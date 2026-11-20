#!/usr/bin/env python3
"""Validation-only image-level hazard prompt experiment; never opens test data.

Two predeclared prompts, same model, reasoning and 224px inputs as baseline.
Preserves subtype baseline; produces only an alternative hazard score.
"""
import argparse
import base64
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from pathlib import Path

MODEL='gpt-6.1-sol'
SCHEMA_SHA='b429abdd786bc6c6f191b5fec0d0104171f2722f03ce819ea71f53bdeab911db'
VARIANTS=['route_focused','evidence_checklist']
GROUPS=['surface_and_level_problems','restricted_passage','missing_tactile_warning','ramp_points_into_traffic']
IDENTITY=['sample_id','image_path','split','hazard_label','output_file_sha256']

def now():return datetime.now(timezone.utc).isoformat()
def sha(x):return hashlib.sha256(x).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def save_json(path,value):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def save_csv(path,rows,fields):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)

DEFINITIONS='''You are assessing accessibility problems in a street-view photograph.
Predict whether the PUBLIC PEDESTRIAN WALKING ROUTE visible in the image contains
at least one of the four study-defined problems below. This is an image-only
assessment, not a general impression of how old, untidy, urban, or unsafe the
scene looks. The route includes the public sidewalk and pedestrian curb ramps
or crossing transitions. Inspect the route before assigning the confidence.

Spatial scope:
- Evaluate the usable walking surface and clearance along that public route.
- Do not count stairs to a private house, a porch, a yard, a retaining wall,
  or a sloping garage driveway outside the walking route. Stairs that actually
  form part of the public walking route DO count.
- A road crack, pothole, road slope, or parked vehicle matters only if it affects
  the pedestrian route or its crossing transition. Camera perspective, a tilted
  horizon, shadows and normal curb height are not evidence of a walking defect.

The four qualifying groups are:
1. Surface and level problems: a visibly broken, bumpy or uneven walking surface,
   displaced/lifted slabs, an abrupt level change on the route, a curb-ramp lip
   that is not flush with the street, pooled water on the route, a clearly steep
   or slanted walking surface, stairs on the public route, or an
   uncovered manhole/open hole on the route. Ordinary paving joints, cosmetic
   cracks, patch color, grass between otherwise usable slabs and brick or
   cobblestone texture alone are NOT sufficient. There must be visible evidence
   of the qualifying defect, not a guess that a crack might hide one.
2. Restricted passage: the remaining usable sidewalk, curb ramp, or landing is
   visibly narrow or has insufficient turning/landing space. Judge the clear
   passage, not merely whether a pole, tree, bush, parked car or object appears
   anywhere in the scene. Objects next to an adequately open route do not count.
   Do not invent measurements from perspective. When width is ambiguous, retain
   that uncertainty instead of assuming a restriction.
3. Missing tactile warning: an identifiable pedestrian curb ramp or pedestrian
   crossing transition lacks the expected detectable warning surface (typically
   a field of raised truncated domes). A colored patch alone is not proof either
   way. Ordinary mid-block pavement, a driveway/garage entrance, a view without
   a visible pedestrian transition, or an obscured transition does not establish
   a missing warning. Only assess absence where the relevant surface is visible.
4. Ramp points into traffic: the direction of travel down a pedestrian curb ramp
   would lead into the vehicle travel area instead of aligning with the intended
   pedestrian crossing. Consider the ramp direction and crossing geometry
   together. A diagonal corner ramp can qualify when it aims into the junction
   rather than along a crossing. Simply meeting the street, lacking painted
   crossing lines, or having traffic nearby is not sufficient evidence.

The four groups may overlap. Hazard means at least one is present.
Non-hazard means none is established by the visible pedestrian-route evidence.
Some details are difficult to see in this low-resolution image; uncertainty is
not itself evidence of a hazard. Do not confidently invent a defect when its
presence or location on the route cannot be established.
'''

def make_prompts(schema):
    tags='\nExact study tag mapping (interpret using the definitions above):\n'
    for group in GROUPS:tags+=group+': '+', '.join(schema['groups'][group]['positive_tags'])+'.\n'
    tags+='These tag concepts alone are non-hazard: '+', '.join(schema['non_hazard_tags'])+'.\n'
    tail='''
Return the requested JSON only. p_hazard is a confidence score between 0 and 1
for the presence of at least one qualifying hazard in this image. It is not a
severity or urgency score. A value near 1 requires clear visible evidence on
the route; a value near 0 means an ordinary route without an established
qualifying problem. Intermediate values express uncertainty about whether a
qualifying problem is actually present. Do not round every image to the same
few confidence levels; use the confidence warranted by its evidence.
'''
    return {
        'route_focused':DEFINITIONS+tags+'''
First identify the relevant public walking route and the most decisive visible
evidence. In decisive_evidence, write one short factual sentence (at most 240
characters) naming the qualifying defect and its location, or explain why the
visible route does not establish one. Then give p_hazard.
'''+tail,
        'evidence_checklist':DEFINITIONS+tags+'''
Check every group separately before producing the hazard score. For each group
return an observation with status present, absent, uncertain, or not_assessable,
and a short description of visible evidence and its location (at most 160
characters). "Absent" requires that the relevant route feature is visible and
does not have the defect. Use "not_assessable" when the required feature or
detail cannot be seen. A feature outside the public route cannot establish
presence. Then combine the four observations into p_hazard. Do not convert an
uncertain or not-assessable entry into a confirmed hazard merely to be cautious.
'''+tail}

def output_schema(variant):
    probability={'type':'number','minimum':0,'maximum':1}
    if variant=='route_focused':
        props={'decisive_evidence':{'type':'string','maxLength':240},'p_hazard':probability}
    else:
        obs={'type':'object','properties':{'status':{'type':'string','enum':['present','absent','uncertain','not_assessable']},
             'evidence':{'type':'string','maxLength':160}},'required':['status','evidence'],'additionalProperties':False}
        props={g:obs for g in GROUPS};props['p_hazard']=probability
    return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}

def validate_output(value,variant):
    expected=set(output_schema(variant)['properties'])
    if not isinstance(value,dict) or set(value)!=expected:raise ValueError('Incorrect output fields')
    p=value['p_hazard']
    if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0<=p<=1:raise ValueError('Invalid hazard confidence')
    if variant=='route_focused':
        if not isinstance(value['decisive_evidence'],str) or len(value['decisive_evidence'])>240:raise ValueError('Invalid evidence')
    else:
        for g in GROUPS:
            o=value[g]
            if set(o)!= {'status','evidence'} or o['status'] not in ['present','absent','uncertain','not_assessable']:raise ValueError('Invalid observation')
            if not isinstance(o['evidence'],str) or len(o['evidence'])>160:raise ValueError('Invalid observation text')
    return value

def load_validation(root):
    from PIL import Image
    root=Path(root).resolve();schema_bytes=(root/'label_schema.json').read_bytes()
    if sha(schema_bytes)!=SCHEMA_SHA:raise ValueError('Unexpected label schema')
    schema=json.loads(schema_bytes);rows=read_csv(root/'pitt-val.csv');seen=set()
    if len(rows)!=382:raise ValueError('Expected the frozen 382-image validation split')
    for r in rows:
        sid=r['sample_id'];rel=Path(r['image_path'])
        if sid in seen or r['split']!='pitt-val' or r['city']!='pittsburgh':raise ValueError('Validation identity mismatch')
        seen.add(sid)
        if rel.is_absolute() or len(rel.parts)!=3 or rel.parts[0]!='pitt-val' or rel.parts[1]!=r['class']:raise ValueError('Not a validation image path')
        path=(root/rel).resolve()
        if not path.is_relative_to(root/'pitt-val'):raise ValueError('Path escapes validation')
        payload=path.read_bytes()
        if sha(payload)!=r['output_file_sha256']:raise ValueError('Changed validation image')
        with Image.open(io.BytesIO(payload)) as im:
            im.load()
            if im.format!='PNG' or im.mode!='RGB' or im.size!=(224,224):raise ValueError('Unexpected preprocessing')
        y=[int(any(r[t]=='1' for t in schema['groups'][g]['positive_tags'])) for g in GROUPS]
        if y!=[int(r['group_'+g]) for g in GROUPS] or int(r['hazard_label'])!=int(any(y)):raise ValueError('Incorrect ground truth')
    return schema,sorted(rows,key=lambda r:r['sample_id'])

def binary_metrics(rows,threshold):
    tp=sum(int(r['hazard_label'])==1 and float(r['p_hazard'])>=threshold for r in rows)
    fn=sum(int(r['hazard_label'])==1 and float(r['p_hazard'])<threshold for r in rows)
    tn=sum(int(r['hazard_label'])==0 and float(r['p_hazard'])<threshold for r in rows)
    fp=len(rows)-tp-fn-tn
    return {'threshold':threshold,'tp':tp,'fn':fn,'tn':tn,'fp':fp,'recall':tp/(tp+fn),'specificity':tn/(tn+fp),
            'precision':tp/(tp+fp) if tp+fp else 0.,'balanced_accuracy':.5*(tp/(tp+fn)+tn/(tn+fp)),
            'accuracy':(tp+tn)/len(rows)}

def evaluate(rows):
    scores={float(r['p_hazard']) for r in rows}
    thresholds=sorted(scores|{0.,1.,math.nextafter(max(scores),math.inf)})
    metrics=[binary_metrics(rows,t) for t in thresholds]
    selected=min(metrics,key=lambda r:(-round(r['balanced_accuracy'],12),r['threshold']))
    ranking=sorted(rows,key=lambda r:(-float(r['p_hazard']),r['sample_id']))
    top={}
    for k in [math.ceil(len(rows)*.1),46]:
        c=float(ranking[k-1]['p_hazard']);above=[r for r in rows if float(r['p_hazard'])>c];ties=[r for r in rows if float(r['p_hazard'])==c]
        slots=k-len(above);a=sum(int(r['hazard_label']) for r in above);h=sum(int(r['hazard_label']) for r in ties)
        hits=sum(int(r['hazard_label']) for r in ranking[:k])
        top[str(k)]={'k':k,'hits':hits,'precision':hits/k,'cutoff':c,'tied_images':len(ties),'selected_tied_images':slots,
            'precision_min_over_ties':(a+max(0,slots-(len(ties)-h)))/k,'precision_max_over_ties':(a+min(slots,h))/k}
    positives=[float(r['p_hazard']) for r in rows if int(r['hazard_label'])]
    negatives=[float(r['p_hazard']) for r in rows if not int(r['hazard_label'])]
    auc=sum((a>b)+.5*(a==b) for a in positives for b in negatives)/(len(positives)*len(negatives))
    return {'n':len(rows),'positive_count':len(positives),'ranking':top,'selected_binary':selected,'roc_auc':auc},metrics,ranking

def load_completed(path,roster):
    completed={}
    if not path.exists():return completed
    offset=0;lines=path.read_bytes().splitlines(keepends=True)
    for i,line in enumerate(lines):
        try:r=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if i!=len(lines)-1 or line.endswith(b'\n'):raise ValueError('Corrupt journal')
            path.with_name('truncated-'+str(time.time_ns())+'.txt').write_bytes(line)
            with path.open('r+b') as f:f.truncate(offset)
            break
        offset+=len(line)
        if r['status']!='ok':continue
        key=(r['variant'],r['sample_id'])
        if r['variant'] not in VARIANTS or r['sample_id'] not in roster or key in completed:raise ValueError('Duplicate or unknown journal response')
        if r['image_sha256']!=roster[r['sample_id']]['output_file_sha256']:raise ValueError('Journal image mismatch')
        validate_output(r['prediction'],r['variant']);completed[key]=r
    if path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:f.write(b'\n')
    return completed

def request(client,root,row,variant,prompt):
    record={'variant':variant,'sample_id':row['sample_id'],'split':'pitt-val','image_sha256':row['output_file_sha256'],'requested_at_utc':now()}
    try:
        payload=(root/row['image_path']).read_bytes()
        if sha(payload)!=row['output_file_sha256']:raise ValueError('Input changed after validation')
        response=client.responses.create(model=MODEL,reasoning={'effort':'low'},store=False,
            input=[{'role':'user','content':[{'type':'input_text','text':prompt},
                {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(payload).decode(),'detail':'high'}]}],
            text={'format':{'type':'json_schema','name':variant,'strict':True,'schema':output_schema(variant)}})
        record['response']=response.model_dump(mode='json')
        if response.status!='completed' or not response.output_text:raise ValueError('Incomplete or refused response')
        if response.model!=MODEL:raise ValueError('Unexpected response model')
        record['prediction']=validate_output(json.loads(response.output_text),variant)
        record['status']='ok'
    except Exception as e:
        import re
        record.update(status='error',error_type=type(e).__name__,error=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',str(e)))
    record['completed_at_utc']=now()
    return record

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=Path('/data/datasets/sidewalk-data-3'))
    ap.add_argument('--baseline-val',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if not 1<=a.workers<=6:ap.error('workers must be 1..6')
    root=a.data_root.resolve();schema,rows=load_validation(root);roster={r['sample_id']:r for r in rows}
    baseline=read_csv(a.baseline_val)
    if len(baseline)!=len(rows) or {r['sample_id'] for r in baseline}!=set(roster):raise ValueError('Baseline validation roster mismatch')
    for r in baseline:
        if r['split']!='pitt-val':raise ValueError('Baseline must contain validation only')
        for k in IDENTITY:
            if r[k]!=roster[r['sample_id']][k]:raise ValueError('Baseline identity mismatch')
    prompts=make_prompts(schema)
    if a.dry_run:
        print(json.dumps({'valid':True,'images':len(rows),'variants':VARIANTS,'new_api_calls':len(rows)*len(VARIANTS),
            'test_images':0,'model':MODEL,'prompt_words':{k:len(v.split()) for k,v in prompts.items()},'primary_k':math.ceil(.1*len(rows))}));return
    if not os.environ.get('OPENAI_API_KEY','').strip():raise ValueError('OPENAI_API_KEY is missing')
    import openai,PIL
    out=a.out.resolve()
    if a.resume:
        if not (out/'config.json').exists():raise ValueError('Resume configuration missing')
    else:out.mkdir(parents=True,exist_ok=False)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        binding={'experiment':'hazard-validation-revision-v1','data_root':str(root),'split':'pitt-val','model':MODEL,'reasoning_effort':'low','image_detail':'high',
            'manifest_sha256':sha((root/'pitt-val.csv').read_bytes()),'schema_sha256':SCHEMA_SHA,
            'baseline_csv_sha256':sha(a.baseline_val.read_bytes()),'runner_sha256':sha(Path(__file__).read_bytes()),'prompts':prompts,
            'output_schemas':{v:output_schema(v) for v in VARIANTS},'image_inventory_sha256':sha(canonical([(r['sample_id'],r['output_file_sha256']) for r in rows]).encode()),
            'primary_metric':'precision at ceil(0.10 * 382) = top39','secondary_metric':'hazard balanced accuracy at validation-selected threshold',
            'threshold_search':'All distinct validation scores, 0, 1, and nextafter(max,+inf); smallest threshold on BA ties',
            'ranking_tie_break':'sample_id ascending','candidate_tie_break':'original then route_focused then evidence_checklist',
            'test_evaluation':False,'subtype_changes':False,'openai_version':openai.__version__,'pillow_version':PIL.__version__}
        if a.resume:
            if json.loads((out/'config.json').read_text())['binding']!=binding:raise ValueError('Resume binding changed')
        else:
            save_json(out/'config.json',{'created_at_utc':now(),'binding':binding})
            shutil.copy2(__file__,out/Path(__file__).name)
            shutil.copy2(root/'pitt-val.csv',out/'pitt-val_manifest.csv')
            shutil.copy2(root/'label_schema.json',out/'label_schema.json')
            shutil.copy2(a.baseline_val,out/'original_validation_predictions.csv')
            for v,prompt in prompts.items():
                (out/(v+'_prompt.txt')).write_text(prompt+'\n');save_json(out/(v+'_schema.json'),output_schema(v))
        path=out/'predictions.jsonl';completed=load_completed(path,roster);initial=len(completed)
        start=time.monotonic();errors=[]
        # Interleave prompts for each sample to reduce temporal ordering effects.
        pending=[(v,r) for r in rows for v in VARIANTS if (v,r['sample_id']) not in completed]
        total=len(rows)*len(VARIANTS)
        def export_progress(state):
            for v in VARIANTS:
                exported=[]
                for r in rows:
                    rec=completed.get((v,r['sample_id']))
                    if rec:exported.append({**{k:r[k] for k in IDENTITY},'p_hazard':rec['prediction']['p_hazard'],'evidence_json':canonical(rec['prediction'])})
                save_csv(out/(v+'_validation_predictions.csv'),exported,IDENTITY+['p_hazard','evidence_json'])
            elapsed=time.monotonic()-start;n=len(completed)-initial
            save_json(out/'status.json',{'state':state,'updated_at_utc':now(),'completed':len(completed),'total':total,
                'by_variant':{v:sum(k[0]==v for k in completed) for v in VARIANTS},'errors':len(errors),
                'elapsed_seconds':round(elapsed,1),'eta_seconds':round(elapsed/n*(total-len(completed)),1) if n else None,'test_images':0})
        export_progress('running')
        with openai.OpenAI(timeout=120,max_retries=2) as client,ThreadPoolExecutor(max_workers=a.workers) as pool,path.open('a') as journal:
            iterator=iter(pending);futures={}
            def submit_one():
                try:v,r=next(iterator)
                except StopIteration:return
                futures[pool.submit(request,client,root,r,v,prompts[v])]=(v,r['sample_id'])
            for _ in range(a.workers):submit_one()
            while futures:
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for f in ready:
                    key=futures.pop(f);record=f.result()
                    journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                    if record['status']=='ok':completed[key]=record
                    else:errors.append({k:record[k] for k in ['variant','sample_id','error_type','error']})
                    print(now(),len(completed),'/',total,key,record['status'],flush=True)
                export_progress('error' if errors else 'running')
                if not errors:
                    for _ in ready:submit_one()
        if errors:
            save_json(out/'errors.json',errors);raise RuntimeError('API errors saved; successful calls retained for resume')
        assert len(completed)==total
        metrics={}
        for name,rs in [('original',baseline)]+[(v,read_csv(out/(v+'_validation_predictions.csv'))) for v in VARIANTS]:
            m,grid,ranked=evaluate(rs);metrics[name]=m
            save_json(out/(name+'_validation_metrics.json'),m)
            save_csv(out/(name+'_threshold_sweep.csv'),grid,list(grid[0]))
            ranked=[{'rank':i,**r} for i,r in enumerate(ranked,1)]
            save_csv(out/(name+'_validation_ranking.csv'),ranked,['rank']+IDENTITY+['p_hazard'])
        order=['original']+VARIANTS;k=str(math.ceil(.1*len(rows)))
        winner=min(order,key=lambda n:(-metrics[n]['ranking'][k]['hits'],-round(metrics[n]['selected_binary']['balanced_accuracy'],12),order.index(n)))
        comparison=[]
        for name in order:
            m=metrics[name];b=m['selected_binary']
            comparison.append({'candidate':name,'n':m['n'],'top10_k':int(k),'top10_hits':m['ranking'][k]['hits'],'precision_top10':m['ranking'][k]['precision'],
                'precision_at46':m['ranking']['46']['precision'],'threshold':b['threshold'],'balanced_accuracy':b['balanced_accuracy'],
                'recall':b['recall'],'specificity':b['specificity'],'roc_auc':m['roc_auc'],'selected':int(name==winner)})
        save_csv(out/'validation_comparison.csv',comparison,list(comparison[0]))
        selected={'candidate':winner,'hazard_threshold':metrics[winner]['selected_binary']['threshold'],'metrics':metrics[winner],
            'selection_split':'pitt-val','test_used':False,'subtype_model':'unchanged original VLM outputs',
            'note':'Development/selection scores on validation; no independent test claim. Original remains selected if revisions do not improve the fixed objective.'}
        if winner!='original':selected.update(prompt=prompts[winner],output_schema=output_schema(winner),model=MODEL,reasoning_effort='low',image_detail='high')
        save_json(out/'selected_hazard_model.json',selected)
        restored=load_completed(path,roster)
        if restored.keys()!=completed.keys():raise ValueError('Journal recovery verification failed')
        usage=Counter();models=Counter()
        for rec in restored.values():
            response=rec['response'];models[response['model']]+=1
            text=''.join(c['text'] for m in response['output'] if m['type']=='message' for c in m['content'] if c['type']=='output_text')
            if json.loads(text)!=rec['prediction']:raise ValueError('Raw response does not match saved scores')
            for key in ['input_tokens','output_tokens','total_tokens']:usage[key]+=response.get('usage',{}).get(key,0)
        save_json(out/'final_verification.json',{'passed':True,'successful_calls':len(restored),'validation_images':len(rows),'test_images':0,
            'journal_sha256':sha(path.read_bytes()),'response_models':dict(models),'usage':dict(usage),'completed_at_utc':now()})
        save_json(out/'results.json',{'selected':winner,'comparison':comparison,'metrics':metrics,'test_used':False})
        export_progress('complete');print('COMPLETE',json.dumps(comparison),flush=True)

if __name__=='__main__':main()
