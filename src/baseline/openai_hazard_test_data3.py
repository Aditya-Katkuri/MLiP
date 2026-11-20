#!/usr/bin/env python3
"""Apply the frozen validation-selected hazard prompt to pitt-test.

No prompt or threshold selection occurs here. Reuses the original subtype scores.
"""
import argparse,base64,fcntl,hashlib,io,json,math,os,shutil,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import openai_hazard_validation_data3 as v
import openai_baseline_data3 as old

SELECTION_SHA='963bd410b0cb2f389a07df9463782d68b731d22d9055a18a61ffd29f87814381'
HELPER_HASHES={'openai_hazard_validation_data3.py':'a60199a42cf2462ef1285e341500cbc9b942d2d1af30e6b7bd63fdc7bfd904e6',
              'openai_baseline_data3.py':'89e72cb2402347d690cc0b2a47f10fb7f87fe399c716cb4ce2ccf0d2f23e14aa'}

def load_test(root,selected,original):
    from PIL import Image
    root=Path(root).resolve();selected=Path(selected);original=Path(original)
    if v.sha(selected.read_bytes())!=SELECTION_SHA:raise ValueError('Selection differs from the frozen validation winner')
    cfg=json.loads(selected.read_text())
    if cfg['candidate']!='evidence_checklist' or cfg['selection_split']!='pitt-val' or cfg['test_used'] is not False or cfg['hazard_threshold']!=.3:
        raise ValueError('Invalid frozen selection')
    if v.sha((root/'label_schema.json').read_bytes())!=v.SCHEMA_SHA:raise ValueError('Unexpected label schema')
    schema=json.loads((root/'label_schema.json').read_text())
    if cfg['output_schema']!=v.output_schema(cfg['candidate']):raise ValueError('Output schema mismatch')
    if cfg['prompt']!=v.make_prompts(schema)[cfg['candidate']]:raise ValueError('Frozen prompt mismatch')
    rows=v.read_csv(root/'pitt-test.csv');seen=set()
    if len(rows)!=380:raise ValueError('Expected 380 test images')
    for r in rows:
        sid=r['sample_id'];rel=Path(r['image_path'])
        if sid in seen or r['split']!='pitt-test' or r['city']!='pittsburgh':raise ValueError('Invalid test identity')
        seen.add(sid)
        if rel.is_absolute() or len(rel.parts)!=3 or rel.parts[:2]!=('pitt-test',r['class']):raise ValueError('Invalid test image path')
        path=(root/rel).resolve()
        if not path.is_relative_to(root/'pitt-test'):raise ValueError('Image escapes test directory')
        payload=path.read_bytes()
        if v.sha(payload)!=r['output_file_sha256']:raise ValueError('Test image changed')
        with Image.open(io.BytesIO(payload)) as im:
            im.load()
            if im.mode!='RGB' or im.size!=(224,224) or im.format!='PNG':raise ValueError('Unexpected test preprocessing')
        groups=[int(any(r[t]=='1' for t in schema['groups'][g]['positive_tags'])) for g in v.GROUPS]
        if groups!=[int(r['group_'+g]) for g in v.GROUPS] or int(r['hazard_label'])!=int(any(groups)):raise ValueError('Test target mismatch')
    roster={r['sample_id']:r for r in rows}
    originals=v.read_csv(original/'pitt-test_predictions.csv')
    if len(originals)!=380 or {r['sample_id'] for r in originals}!=set(roster):raise ValueError('Original subtype roster mismatch')
    for r in originals:
        t=roster[r['sample_id']]
        for k in v.IDENTITY+old.GROUP_COLUMNS:
            if r[k]!=t[k]:raise ValueError('Original subtype identity/target mismatch')
        for key in old.PROBABILITY_FIELDS:r[key]=float(r[key])
    old_selection=json.loads((original/'selection.json').read_text())
    if old_selection['test_used_for_selection'] or old_selection['validation_split']!='pitt-val':raise ValueError('Subtype thresholds were not selected on validation')
    return cfg,sorted(rows,key=lambda r:r['sample_id']),{r['sample_id']:r for r in originals},old_selection

def request(client,root,row,cfg):
    record={'sample_id':row['sample_id'],'variant':cfg['candidate'],'split':'pitt-test','image_sha256':row['output_file_sha256'],'requested_at_utc':v.now()}
    try:
        payload=(root/row['image_path']).read_bytes()
        if v.sha(payload)!=row['output_file_sha256']:raise ValueError('Test image changed during run')
        response=client.responses.create(model=cfg['model'],reasoning={'effort':cfg['reasoning_effort']},store=False,
            input=[{'role':'user','content':[{'type':'input_text','text':cfg['prompt']},
                {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(payload).decode(),'detail':cfg['image_detail']}]}],
            text={'format':{'type':'json_schema','name':cfg['candidate'],'strict':True,'schema':cfg['output_schema']}})
        record['response']=response.model_dump(mode='json')
        if response.status!='completed' or response.model!=cfg['model'] or not response.output_text:raise ValueError('Incomplete or unexpected response')
        record['prediction']=v.validate_output(json.loads(response.output_text),cfg['candidate'])
        record['status']='ok'
    except Exception as exc:
        import re
        record.update(status='error',error_type=type(exc).__name__,error=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',str(exc)))
    record['completed_at_utc']=v.now();return record

def evaluate_fixed(rows,cfg,original_selection):
    """Fixed validation thresholds only; never calls a threshold-search helper."""
    ht=cfg['hazard_threshold'];gt=original_selection['group_thresholds']
    metrics=old.evaluate(rows,ht,gt)
    metrics['hazard_top10']=old.hazard_metrics(rows,ht,k=math.ceil(len(rows)*.1))
    positives=[r['p_hazard'] for r in rows if int(r['hazard_label'])]
    negatives=[r['p_hazard'] for r in rows if not int(r['hazard_label'])]
    metrics['hazard_roc_auc']=sum((a>b)+.5*(a==b) for a in positives for b in negatives)/(len(positives)*len(negatives))
    metrics.update(hazard_prompt=cfg['candidate'],thresholds_selected_on='pitt-val',test_used_for_selection=False,
                   subtype_scores_reused_from_original=True)
    return metrics,old.classify_rows(rows,ht,gt)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=Path('/data/datasets/sidewalk-data-3'))
    ap.add_argument('--selected',type=Path,required=True)
    ap.add_argument('--original-run',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    if not 1<=a.workers<=6:ap.error('workers must be 1..6')
    for mod in [v,old]:
        if v.sha(Path(mod.__file__).read_bytes())!=HELPER_HASHES[Path(mod.__file__).name]:raise ValueError('Evaluation helper changed')
    root=a.data_root.resolve();cfg,rows,original,original_selection=load_test(root,a.selected,a.original_run)
    if a.dry_run:
        print(json.dumps({'valid':True,'test_images':len(rows),'new_api_calls':len(rows),'hazard_threshold':cfg['hazard_threshold'],
            'candidate':cfg['candidate'],'group_thresholds':original_selection['group_thresholds'],'subtype_api_calls':0}));return
    if not os.environ.get('OPENAI_API_KEY','').strip():raise ValueError('OPENAI_API_KEY is missing')
    import openai
    out=a.out.resolve()
    if a.resume:
        if not (out/'config.json').exists():raise ValueError('Resume configuration missing')
    else:out.mkdir(parents=True,exist_ok=False)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding={'experiment':'frozen-hazard-checklist-test-v1','selected_config_sha256':SELECTION_SHA,
            'runner_sha256':v.sha(Path(__file__).read_bytes()),'helper_hashes':HELPER_HASHES,
            'test_manifest_sha256':v.sha((root/'pitt-test.csv').read_bytes()),'schema_sha256':v.SCHEMA_SHA,
            'original_predictions_sha256':v.sha((a.original_run/'pitt-test_predictions.csv').read_bytes()),
            'original_selection_sha256':v.sha((a.original_run/'selection.json').read_bytes()),
            'original_run':str(a.original_run.resolve()),'data_root':str(root),'openai_version':openai.__version__,
            'model':cfg['model'],'reasoning_effort':cfg['reasoning_effort'],'image_detail':cfg['image_detail'],
            'hazard_threshold':cfg['hazard_threshold'],'group_thresholds':original_selection['group_thresholds'],
            'primary_k':38,'secondary_k':46,'test_used_for_selection':False,'workers':a.workers}
        if a.resume:
            if json.loads((out/'config.json').read_text())['binding']!=binding:raise ValueError('Resume settings changed')
        else:
            v.save_json(out/'config.json',{'created_at_utc':v.now(),'binding':binding})
            for src,name in [(a.selected,'selected_hazard_model.json'),(root/'pitt-test.csv','pitt-test_manifest.csv'),
                (root/'label_schema.json','label_schema.json'),(a.original_run/'pitt-test_predictions.csv','original_test_predictions.csv'),
                (a.original_run/'selection.json','original_selection.json')]:shutil.copy2(src,out/name)
            for mod in [v,old]:shutil.copy2(mod.__file__,out/Path(mod.__file__).name)
            shutil.copy2(__file__,out/Path(__file__).name)
            (out/'prompt.txt').write_text(cfg['prompt']+'\n');v.save_json(out/'output_schema.json',cfg['output_schema'])
        roster={r['sample_id']:r for r in rows};journal_path=out/'predictions.jsonl'
        completed=v.load_completed(journal_path,roster)
        if any(rec['split']!='pitt-test' or key[0]!=cfg['candidate'] for key,rec in completed.items()):raise ValueError('Wrong journal split or variant')
        start=time.monotonic();initial=len(completed);errors=[]
        def progress(state):
            exports=[]
            for r in rows:
                rec=completed.get((cfg['candidate'],r['sample_id']))
                if rec:exports.append({**{k:r[k] for k in v.IDENTITY},'p_hazard':rec['prediction']['p_hazard'],
                    'pred_hazard':int(rec['prediction']['p_hazard']>=cfg['hazard_threshold']),'evidence_json':v.canonical(rec['prediction'])})
            v.save_csv(out/'pitt-test_hazard_predictions.csv',exports,v.IDENTITY+['p_hazard','pred_hazard','evidence_json'])
            elapsed=time.monotonic()-start;n=len(completed)-initial
            v.save_json(out/'status.json',{'state':state,'updated_at_utc':v.now(),'completed':len(completed),'total':len(rows),
                'errors':len(errors),'elapsed_seconds':round(elapsed,1),'eta_seconds':round(elapsed/n*(len(rows)-len(completed)),1) if n else None})
        progress('running')
        pending=iter([r for r in rows if (cfg['candidate'],r['sample_id']) not in completed])
        with openai.OpenAI(timeout=120,max_retries=2) as client,ThreadPoolExecutor(max_workers=a.workers) as pool,journal_path.open('a') as journal:
            futures={}
            def submit():
                try:r=next(pending)
                except StopIteration:return
                futures[pool.submit(request,client,root,r,cfg)]=r['sample_id']
            for _ in range(a.workers):submit()
            while futures:
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for f in ready:
                    sid=futures.pop(f);rec=f.result()
                    journal.write(json.dumps(rec,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                    if rec['status']=='ok':completed[(cfg['candidate'],sid)]=rec
                    else:errors.append({k:rec[k] for k in ['sample_id','error_type','error']})
                    print(v.now(),len(completed),'/',len(rows),sid,rec['status'],flush=True)
                progress('error' if errors else 'running')
                if not errors:
                    for _ in ready:submit()
        if errors:v.save_json(out/'errors.json',errors);raise RuntimeError('API errors saved; successful responses can be resumed')
        assert len(completed)==380
        merged=[]
        for r in rows:
            sid=r['sample_id'];rec=completed[(cfg['candidate'],sid)];base=dict(original[sid])
            base['original_p_hazard']=base['p_hazard'];base['p_hazard']=rec['prediction']['p_hazard']
            base['hazard_evidence_json']=v.canonical(rec['prediction']);base['hazard_method']='validation_selected_evidence_checklist'
            base['subtype_method']='original_frozen_vlm_probabilities';merged.append(base)
        metrics,classified=evaluate_fixed(merged,cfg,original_selection)
        old_metrics=old.evaluate(list(original.values()),original_selection['hazard_threshold'],original_selection['group_thresholds'])
        old_metrics['hazard_top10']=old.hazard_metrics(list(original.values()),original_selection['hazard_threshold'],k=38)
        assert math.isclose(metrics['groups_on_all_true_hazards']['balanced_accuracy'],old_metrics['groups_on_all_true_hazards']['balanced_accuracy'],abs_tol=1e-12)
        for r in classified:
            previous=original[r['sample_id']]
            for g in v.GROUPS:
                assert r['p_'+g]==previous['p_'+g] and int(r['pred_'+g])==int(previous['pred_'+g])
        v.save_csv(out/'pitt-test_predictions.csv',classified,list(classified[0]))
        ranked=[{'rank':i,**r} for i,r in enumerate(sorted(classified,key=lambda r:(-r['p_hazard'],r['sample_id'])),1)]
        for name,rs in [('pitt-test_hazard_ranking.csv',ranked),('pitt-test_top38.csv',ranked[:38]),('pitt-test_top46.csv',ranked[:46])]:
            v.save_csv(out/name,rs,list(rs[0]))
        v.save_json(out/'pitt-test_metrics.json',metrics)
        v.save_csv(out/'pitt-test_per_group_metrics.csv',metrics['group_metrics_on_all_true_hazards'],list(metrics['group_metrics_on_all_true_hazards'][0]))
        v.save_json(out/'original_test_metrics.json',old_metrics)
        comparison=[]
        for name,m in [('original',old_metrics),('revised_hazard',metrics)]:
            comparison.append({'method':name,'images':380,'hazard_threshold':m['hazard_threshold'],
                'hazard_precision_top10':m['hazard_top10']['precision_at_k'],'hazard_top38_hits':m['hazard_top10']['topk_hits'],
                'hazard_precision_at46':m['hazard']['precision_at_k'],'hazard_balanced_accuracy':m['hazard']['balanced_accuracy'],
                'hazard_recall':m['hazard']['recall'],'hazard_precision':m['hazard']['precision'],
                'subtype_balanced_accuracy':m['groups_on_all_true_hazards']['balanced_accuracy']})
        v.save_csv(out/'test_comparison.csv',comparison,list(comparison[0]))
        recovered=v.load_completed(journal_path,roster)
        assert recovered.keys()==completed.keys()
        usage=Counter()
        for key,rec in recovered.items():
            resp=rec['response'];assert resp['model']==cfg['model'] and resp['reasoning']['effort']==cfg['reasoning_effort']
            text=''.join(c['text'] for m in resp['output'] if m['type']=='message' for c in m['content'] if c['type']=='output_text')
            assert json.loads(text)==rec['prediction']
            for field in ['input_tokens','output_tokens','total_tokens']:usage[field]+=resp.get('usage',{}).get(field,0)
        v.save_json(out/'results.json',{'comparison':comparison,'test_metrics':metrics,'test_used_for_selection':False,'subtype_unchanged':True})
        v.save_json(out/'final_verification.json',{'passed':True,'successful_calls':380,'test_images':380,'duplicate_saved_calls':0,
            'selected_config_sha256':SELECTION_SHA,'prediction_journal_sha256':v.sha(journal_path.read_bytes()),
            'subtype_probabilities_and_primary_decisions_unchanged':True,'test_used_for_selection':False,'usage':dict(usage),'completed_at_utc':v.now()})
        progress('complete');print('COMPLETE',json.dumps(comparison),flush=True)

if __name__=='__main__':main()
