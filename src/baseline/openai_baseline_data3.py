#!/usr/bin/env python3
"""Zero-shot OpenAI hazard + conditional multi-label tag-group baseline.

Uses sidewalk-data-3 pitt-val for threshold selection, then pitt-test for final
evaluation. No training images, filenames, annotations, or labels go to the API.
Use --dry-run for a full offline input check. --resume skips durable successes.
The previous openai_baseline.py and its results are left unchanged.
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
import re
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = 'gpt-6.1-sol'
DEFAULT_ROOT = Path('/data/datasets/sidewalk-data-3')
REASONING_EFFORT = 'low'
SPLITS = ['pitt-val', 'pitt-test']
THRESHOLDS = [0.3, 0.5, 0.7]
K = 46
SCHEMA_SHA256 = 'b429abdd786bc6c6f191b5fec0d0104171f2722f03ce819ea71f53bdeab911db'
GROUPS = ['surface_and_level_problems', 'restricted_passage',
          'missing_tactile_warning', 'ramp_points_into_traffic']
GROUP_COLUMNS = ['group_' + g for g in GROUPS]
PROBABILITY_FIELDS = ['p_hazard'] + ['p_' + g for g in GROUPS]
IDENTITY_FIELDS = ['sample_id', 'image_path', 'split', 'class', 'hazard_label',
                   'original_class', 'source_image_path', 'city', 'is_manual_collection',
                   'label_basis', 'annotation_status', 'output_file_sha256'] + GROUP_COLUMNS
RAW_FIELDS = IDENTITY_FIELDS + PROBABILITY_FIELDS
OUTPUT_SCHEMA = {'type': 'object', 'properties': {
    f: {'type': 'number', 'minimum': 0, 'maximum': 1} for f in PROBABILITY_FIELDS},
    'required': PROBABILITY_FIELDS, 'additionalProperties': False}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def save_json(path, value):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def save_csv(path, rows, fields=None):
    path = Path(path)
    tmp = path.with_name('.' + path.name + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def make_prompt(schema):
    definitions = {
        'surface_and_level_problems': (
            'A walking-surface or level defect: broken or very broken pavement, bumps, '
            'uneven surface, a curb ramp not level with the street, pooled water, a steep '
            'slope, a surface problem, an abrupt height difference, stairs, an uneven or '
            'slanted walking surface, or an uncovered manhole.'),
        'restricted_passage': (
            'A narrow pedestrian passage or sidewalk, or insufficient landing/turning '
            'space at a ramp. The mere presence of an object does not establish this '
            'group: the usable pedestrian passage or maneuvering space must be restricted.'),
        'missing_tactile_warning': (
            'A curb ramp or pedestrian crossing transition lacks its tactile warning '
            'surface. Do not infer this from an ordinary mid-block sidewalk or a '
            'ramp/crossing that is not visible.'),
        'ramp_points_into_traffic': (
            'A curb ramp points into vehicular traffic instead of aligning with its '
            'intended pedestrian crossing. Do not infer direction when the ramp or '
            'crossing geometry cannot be seen.'),
    }
    lines = [
        'Inspect the pedestrian sidewalk, curb transition, or crossing in this image.',
        'Use the following study-specific definition of hazard, not a general safety judgment.',
        'A hazard is the visible presence of at least one problem in these four groups.',
        'The groups can overlap: an image may have several groups simultaneously.\n',
    ]
    for group in GROUPS:
        lines += [group + ': ' + definitions[group],
                  'Included tag concepts: ' + ', '.join(schema['groups'][group]['positive_tags']) + '.\n']
    lines += [
        'The following feature/tag concepts alone are NON-HAZARD under this study policy:',
        ', '.join(schema['non_hazard_tags']) + '.',
        'For example, a pole, parked car, vegetation, cracks, grass, or brick paving '
        'alone is not sufficient. If it also visibly causes a qualifying narrow passage '
        'or surface/level problem, that qualifying problem makes the image a hazard.',
        'A crosswalk or curb ramp is not itself a hazard; assess the four groups above. '
        'An ordinary sidewalk with none of those problems is non-hazard.\n',
        'Return only a JSON object with exactly these five numeric fields:',
        'p_hazard: P(at least one of the four hazard groups is present | image).',
        'For each of the four other fields, estimate the probability that its group '
        'is present CONDITIONAL ON the image being a hazard:',
        *['p_' + g + ': P(' + g + ' is present | hazard, image).' for g in GROUPS],
        'Return all four conditional group estimates even when p_hazard is low. '
        'Do not multiply the group estimates by p_hazard. They are overlapping binary '
        'attributes, NOT mutually exclusive classes, and must NOT be normalized to sum to one.',
        'Every value must be finite and between 0 and 1 inclusive. Use up to six decimal '
        'places. Judge only visible evidence and express ambiguity through the probabilities. '
        'Do not return explanations, tag lists, class names, or any extra keys.',
    ]
    return '\n'.join(lines)


def bit(value):
    if value not in ('0', '1', 0, 1):
        raise ValueError(f'Not a binary label: {value!r}')
    return int(value)


def load_inputs(root):
    """Recompute targets from tags and validate the exact manifest/image roster."""
    from PIL import Image
    root = Path(root).resolve()
    schema_bytes = (root / 'label_schema.json').read_bytes()
    if sha(schema_bytes) != SCHEMA_SHA256:
        raise ValueError('Label schema differs from the reviewed dataset-3 task contract.')
    schema = json.loads(schema_bytes)
    assert schema['group_order'] == GROUPS and schema['multi_label'] is True
    data = {}; seen = set(); pixel_splits = {}
    for split in SPLITS:
        with (root / (split + '.csv')).open(newline='') as f:
            reader = csv.DictReader(f)
            required = set(IDENTITY_FIELDS + schema['all_tags'] + ['label_rule_version', 'output_rgb_pixel_sha256'])
            if not required <= set(reader.fieldnames or []):
                raise ValueError('Missing manifest columns: ' + str(sorted(required - set(reader.fieldnames or []))))
            rows = list(reader)
        paths = set()
        for row in rows:
            sid = row['sample_id']; rel = Path(row['image_path'])
            if sid in seen or not sid or row['split'] != split or row['city'] != 'pittsburgh':
                raise ValueError(f'Duplicate identity or non-Pittsburgh/split input: {sid}')
            if row['label_rule_version'] != schema['rule_version']:
                raise ValueError('Mixed label versions')
            seen.add(sid)
            expected = [int(any(row[t] == '1' for t in schema['groups'][g]['positive_tags'])) for g in GROUPS]
            if any(row[t] not in ('', '0', '1') for t in schema['all_tags']):
                raise ValueError('Unexpected raw tag value')
            yg = [bit(row[c]) for c in GROUP_COLUMNS]; yh = bit(row['hazard_label'])
            if expected != yg or yh != int(any(yg)) or row['class'] != ('hazard' if yh else 'non_hazard'):
                raise ValueError(f'Tag/target mismatch: {sid}')
            if bit(row['is_manual_collection']) and yh:
                raise ValueError('Hand-collected ordinary sidewalk labeled as hazard')
            if rel.is_absolute() or len(rel.parts) != 3 or rel.parts[:2] != (split, row['class']) or rel.suffix != '.png':
                raise ValueError(f'Invalid image path: {rel}')
            image_path = (root / rel).resolve()
            if not image_path.is_relative_to(root / split / row['class']) or row['image_path'] in paths:
                raise ValueError('Escaping or duplicate image path')
            paths.add(row['image_path'])
            payload = image_path.read_bytes()
            if sha(payload) != row['output_file_sha256']:
                raise ValueError(f'Image checksum mismatch: {sid}')
            with Image.open(io.BytesIO(payload)) as im:
                im.load()
                if im.format != 'PNG' or im.mode != 'RGB' or im.size != (224, 224):
                    raise ValueError(f'Expected existing 224x224 RGB PNG: {sid}')
                # PNG is sent as-is; no new resize, crop, annotation or class-dependent processing.
            pixel = row['output_rgb_pixel_sha256']
            if pixel in pixel_splits and pixel_splits[pixel] != split:
                raise ValueError('Pixel-identical image across validation and test')
            pixel_splits[pixel] = split
        disk = {p.relative_to(root).as_posix() for p in (root / split).rglob('*') if p.suffix.lower() in ('.png','.jpg','.jpeg','.webp')}
        if disk != paths or len(rows) < K:
            raise ValueError(f'Incomplete manifest or image roster: {split}')
        for col in ['hazard_label'] + GROUP_COLUMNS:
            subset = rows if col == 'hazard_label' else [r for r in rows if bit(r['hazard_label'])]
            if {bit(r[col]) for r in subset} != {0, 1}:
                raise ValueError(f'Balanced accuracy undefined for {split} {col}')
        data[split] = sorted(rows, key=lambda r: r['sample_id'])
    return schema, data


def validate_prediction(prediction):
    if not isinstance(prediction, dict) or set(prediction) != set(PROBABILITY_FIELDS):
        raise ValueError('Expected exactly p_hazard and the four conditional group probabilities')
    for key, value in prediction.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid probability: {key}={value!r}')
    # Overlapping conditional group probabilities are deliberately not normalized.
    return {f: float(prediction[f]) for f in PROBABILITY_FIELDS}


def request_prediction(client, model, payload, prompt, detail):
    return client.responses.create(
        model=model, reasoning={'effort': REASONING_EFFORT}, store=False,
        input=[{'role': 'user', 'content': [
            {'type': 'input_text', 'text': prompt},
            {'type': 'input_image', 'image_url': 'data:image/png;base64,' + base64.b64encode(payload).decode(), 'detail': detail}]}],
        text={'format': {'type': 'json_schema', 'name': 'sidewalk_data3_prediction', 'strict': True, 'schema': OUTPUT_SCHEMA}})


def binary_metrics(y, pred):
    if not y or len(y) != len(pred):
        raise ValueError('Empty or mismatched evaluation arrays')
    tp = sum(a == 1 and b == 1 for a, b in zip(y, pred)); fn = sum(a == 1 and b == 0 for a, b in zip(y, pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y, pred)); fp = sum(a == 0 and b == 1 for a, b in zip(y, pred))
    div = lambda a, b: a / b if b else 0.0
    recall = div(tp, tp+fn); specificity = div(tn, tn+fp)
    return dict(tp=tp, fn=fn, tn=tn, fp=fp, support=tp+fn, negative_support=tn+fp,
                precision=div(tp,tp+fp), recall=recall, specificity=specificity,
                f1=div(2*tp,2*tp+fp+fn), accuracy=(tp+tn)/len(y),
                balanced_accuracy=(recall+specificity)/2 if tp+fn and tn+fp else None)


def hazard_metrics(rows, threshold, k=K):
    if len(rows) < k:
        raise ValueError('Not enough images for top-k evaluation')
    ranked = sorted(rows, key=lambda r: (-r['p_hazard'], r['sample_id']))
    y = [bit(r['hazard_label']) for r in rows]
    hits = sum(bit(r['hazard_label']) for r in ranked[:k])
    cutoff = ranked[k-1]['p_hazard']
    above = [r for r in ranked if r['p_hazard'] > cutoff]
    tied = [r for r in ranked if r['p_hazard'] == cutoff]
    slots = k-len(above); positives = sum(bit(r['hazard_label']) for r in tied)
    above_hits = sum(bit(r['hazard_label']) for r in above)
    m = binary_metrics(y, [int(r['p_hazard'] >= threshold) for r in rows])
    return {**m, 'threshold': threshold, 'k': k, 'topk_hits': hits,
            'precision_at_k': hits/k, 'recall_at_k': hits/sum(y),
            'cutoff_ties': {'probability': cutoff, 'tied_images':len(tied),'selected_tied_images':slots,
                'tie_break':'sample_id ascending',
                'possible_precision_min':(above_hits+max(0,slots-(len(tied)-positives)))/k,
                'possible_precision_max':(above_hits+min(slots,positives))/k}}


def group_metrics(rows, thresholds, *, gated=False):
    # The primary subtype metric includes ALL ground-truth hazards, not only detected ones.
    subset = rows if gated else [r for r in rows if bit(r['hazard_label'])]
    per_group=[]; decisions=[]
    for g, t in zip(GROUPS, thresholds):
        pred=[int(r['p_'+g] >= t) * (int(r['pred_hazard']) if gated else 1) for r in subset]
        decisions.append(pred)
        per_group.append({'tag_group':g, **binary_metrics([bit(r['group_'+g]) for r in subset],pred)})
    summary={name:sum(r[name] for r in per_group)/len(GROUPS)
             for name in ['balanced_accuracy','precision','recall','f1','accuracy']}
    summary['exact_match_accuracy']=sum(all(decisions[j][i]==bit(r['group_'+g]) for j,g in enumerate(GROUPS))
                                        for i,r in enumerate(subset))/len(subset)
    summary['n_images']=len(subset)
    return summary,per_group


def choose_thresholds(validation_rows):
    if not validation_rows or {r['split'] for r in validation_rows} != {'pitt-val'}:
        raise ValueError('Threshold selection accepts validation predictions only')
    hs=[]; gs=[]
    for t in THRESHOLDS:
        hs.append({'task':'hazard','threshold':t,**hazard_metrics(validation_rows,t)})
        _, per=group_metrics(validation_rows,[t]*4)
        gs.extend({'task':r['tag_group'],'threshold':t,**r} for r in per)
    pick=lambda rs:min(rs,key=lambda r:(-round(r['balanced_accuracy'],12),r['threshold']))['threshold']
    selection={'validation_split':'pitt-val','test_used_for_selection':False,'threshold_grid':THRESHOLDS,
               'hazard_threshold':pick(hs),'group_thresholds':[pick([r for r in gs if r['task']==g]) for g in GROUPS],
               'group_order':GROUPS,'k':K,
               'hazard_rule':'One fixed zero-shot model/prompt. Precision@46 is threshold-independent; choose threshold by validation balanced accuracy. Lower threshold breaks exact BA ties.',
               'group_rule':'Independently maximize each group binary BA on all true validation hazards; lower threshold breaks exact BA ties.'}
    return selection, hs, gs


def classify_rows(rows, ht, gt):
    output=[]
    for row in rows:
        item=dict(row); item['p_non_hazard']=1-row['p_hazard']; item['pred_hazard']=int(row['p_hazard']>=ht)
        item['predicted_class']='hazard' if item['pred_hazard'] else 'non_hazard'
        for g,t in zip(GROUPS,gt):
            item['pred_'+g]=int(row['p_'+g]>=t)
            item['pipeline_pred_'+g]=item['pred_'+g]*item['pred_hazard']
        item['hazard_without_group']=int(item['pred_hazard'] and not any(item['pred_'+g] for g in GROUPS))
        output.append(item)
    return output


def evaluate(rows, ht, gt):
    classified=classify_rows(rows,ht,gt)
    hm=hazard_metrics(rows,ht)
    gm,per=group_metrics(rows,gt); pm,pper=group_metrics(classified,gt,gated=True)
    non=binary_metrics([1-bit(r['hazard_label']) for r in rows],[1-r['pred_hazard'] for r in classified])
    return {'n_images':len(rows),'hazard':hm,
            'binary_per_class':{'non_hazard':non,'hazard':{k:v for k,v in hm.items() if k in non}},
            'binary_macro_f1':(non['f1']+hm['f1'])/2,
            'binary_confusion':{'labels':['non_hazard','hazard'],'matrix':[[hm['tn'],hm['fp']],[hm['fn'],hm['tp']]]},
            'groups_on_all_true_hazards':gm,'group_metrics_on_all_true_hazards':per,
            'gated_pipeline_groups_on_all_images':pm,'pipeline_group_metrics':pper,
            'hazard_without_group_count':sum(r['hazard_without_group'] for r in classified),
            'hazard_threshold':ht,'group_thresholds':gt,'test_used_for_selection':False}


def recover_journal(path, roster):
    """Read successful responses, repairing only an interrupted final JSON line."""
    completed={}
    if not path.exists(): return completed
    lines=path.read_bytes().splitlines(keepends=True); offset=0
    for i,line in enumerate(lines):
        try: record=json.loads(line)
        except (ValueError,UnicodeDecodeError):
            if i != len(lines)-1 or line.endswith(b'\n'): raise ValueError(f'Corrupt journal line {i+1}')
            path.with_name('truncated-final-line-'+str(time.time_ns())+'.txt').write_bytes(line)
            with path.open('r+b') as f: f.truncate(offset)
            break
        offset+=len(line)
        if record.get('status') != 'ok': continue
        sid=record['sample_id']
        if sid not in roster or sid in completed: raise ValueError('Unknown or duplicate journal identity')
        row=roster[sid]
        if record['metadata'] != row or record['image_sha256'] != row['output_file_sha256']:
            raise ValueError('Journal input/target mismatch')
        completed[sid]={**{f:row[f] for f in IDENTITY_FIELDS},**validate_prediction(record['prediction'])}
    if path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:f.write(b'\n');f.flush();os.fsync(f.fileno())
    return completed


def export_split(out, split, rows, selection):
    ht=selection['hazard_threshold'];gt=selection['group_thresholds']
    classified=classify_rows(rows,ht,gt)
    save_csv(out/(split+'_predictions.csv'),classified)
    ranked=[{'rank':i,**r} for i,r in enumerate(sorted(classified,key=lambda r:(-r['p_hazard'],r['sample_id'])),1)]
    save_csv(out/(split+'_hazard_ranking.csv'),ranked)
    save_csv(out/(split+'_top46.csv'),ranked[:K])
    result=evaluate(rows,ht,gt)
    save_json(out/(split+'_metrics.json'),result)
    save_csv(out/(split+'_per_group_metrics.csv'),result['group_metrics_on_all_true_hazards'])
    save_json(out/(split+'_metrics_fixed_0.5.json'),evaluate(rows,.5,[.5]*4))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--model',default=DEFAULT_MODEL)
    ap.add_argument('--detail',choices=['low','high','auto'],default='high')
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args();root=a.data_root.resolve()
    schema,data=load_inputs(root);prompt=make_prompt(schema)
    counts={s:{'images':len(rs),'hazard':sum(bit(r['hazard_label']) for r in rs),
               'non_hazard':sum(not bit(r['hazard_label']) for r in rs),
               'groups':{g:sum(bit(r['group_'+g]) for r in rs) for g in GROUPS}} for s,rs in data.items()}
    if a.dry_run:
        print(json.dumps({'dry_run':True,'api_requests':0,'model':a.model,'reasoning_effort':REASONING_EFFORT,
                          'split_counts':counts,'probability_fields':PROBABILITY_FIELDS,'prompt':prompt},indent=2));return
    if a.out is None:ap.error('--out is required')
    if not os.environ.get('OPENAI_API_KEY','').strip():ap.error('OPENAI_API_KEY is missing from the environment')
    import openai
    import PIL
    out=a.out.resolve()
    if a.resume:
        if not (out/'run_config.json').is_file():ap.error('--resume requires an existing run_config.json')
    else:out.mkdir(parents=True,exist_ok=False)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        input_names=['pitt-val.csv','pitt-test.csv','label_schema.json','build_config.json','LABELING_RULES.md','PREPROCESSING.md']
        inventory=[{'sample_id':r['sample_id'],'split':s,'image_path':r['image_path'],'sha256':r['output_file_sha256']}
                   for s,rs in data.items() for r in rs]
        binding={'task_version':'sidewalk-data-3-vlm-v1','data_root':str(root),
                 'input_hashes':{n:sha((root/n).read_bytes()) for n in input_names},
                 'image_inventory_sha256':sha(canonical(inventory).encode()),
                 'runner_sha256':sha(Path(__file__).read_bytes()),
                 'model_requested':a.model,'reasoning_effort':REASONING_EFFORT,'image_detail':a.detail,
                 'prompt':prompt,'output_schema':OUTPUT_SCHEMA,'groups':GROUPS,'threshold_grid':THRESHOLDS,'k':K,
                 'split_counts':counts,'openai_version':openai.__version__,'pillow_version':PIL.__version__,
                 'python_version':sys.version,'store':False,
                 'probability_semantics':'p_hazard = P(hazard|image); p_group = P(group|hazard,image), independent multi-label outputs',
                 'ranking':'p_hazard descending, sample_id ascending','training':'none; fixed zero-shot prompt',
                 'threshold_selection':'pitt-val only; test labels never used for thresholds or prompt changes'}
        if a.resume:
            old=json.loads((out/'run_config.json').read_text())
            if old['binding'] != binding:
                raise ValueError('Resume input/settings/code mismatch: '+str([k for k in binding if old['binding'].get(k)!=binding[k]]))
        else:
            (out/'inputs').mkdir()
            for name in input_names:shutil.copy2(root/name,out/'inputs'/name)
            shutil.copy2(__file__,out/'openai_baseline_data3.py')
            save_json(out/'run_config.json',{'created_at_utc':now(),'binding':binding})
            save_json(out/'image_inventory.json',inventory)
            (out/'prompt.txt').write_text(prompt+'\n')
            save_json(out/'output_schema.json',OUTPUT_SCHEMA)
            (out/'README.md').write_text(RUN_README)
        roster={r['sample_id']:r for rows in data.values() for r in rows}
        completed=recover_journal(out/'predictions.jsonl',roster)
        # Rebuild CSVs from the durable journal before any new paid requests.
        for split,rs in data.items():
            save_csv(out/(split+'_probabilities.csv'),[completed[r['sample_id']] for r in rs if r['sample_id'] in completed],RAW_FIELDS)
        start=time.monotonic(); prior=len(completed); selection=None
        def status(state, **extra):
            elapsed=time.monotonic()-start;new=len(completed)-prior;total=len(roster)
            save_json(out/'status.json',{'state':state,'updated_at_utc':now(),'completed':len(completed),'total':total,
                'completed_by_split':{s:sum(r['sample_id'] in completed for r in rs) for s,rs in data.items()},
                'seconds_this_process':round(elapsed,1),'eta_seconds':round(elapsed/new*(total-len(completed)),1) if new else None,**extra})
        try:
            status('running')
            with openai.OpenAI(timeout=120.0,max_retries=2) as client, (out/'predictions.jsonl').open('a') as journal:
                reports={}
                for split,rs in data.items():
                    if split=='pitt-test' and selection is None:raise RuntimeError('Validation selection must precede test requests')
                    print(f'{now()} {split}: {len(rs)} images; {sum(r["sample_id"] in completed for r in rs)} already saved',flush=True)
                    csv_path=out/(split+'_probabilities.csv')
                    with csv_path.open('a',newline='') as cf:
                        writer=csv.DictWriter(cf,fieldnames=RAW_FIELDS)
                        for row in rs:
                            sid=row['sample_id']
                            if sid in completed:continue
                            record={'sample_id':sid,'metadata':row,'image_sha256':row['output_file_sha256'],'requested_at_utc':now()}
                            try:
                                payload=(root/row['image_path']).read_bytes()
                                if sha(payload)!=row['output_file_sha256']:raise ValueError('Image changed after preflight')
                                response=request_prediction(client,a.model,payload,prompt,a.detail)
                                record['response']=response.model_dump(mode='json')
                                if response.status!='completed' or not response.output_text:raise ValueError('Incomplete/refused/empty model output')
                                prediction=validate_prediction(json.loads(response.output_text))
                                record.update(status='ok',prediction=prediction,completed_at_utc=now())
                            except Exception as exc:
                                record.update(status='error',error_type=type(exc).__name__,error=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',str(exc)))
                                raise
                            finally:
                                journal.write(json.dumps(record,allow_nan=False)+'\n');journal.flush();os.fsync(journal.fileno())
                            result={**{f:row[f] for f in IDENTITY_FIELDS},**prediction}
                            writer.writerow(result);cf.flush();os.fsync(cf.fileno())
                            completed[sid]=result
                            status('running',current_split=split)
                            print(f'{now()} [{len(completed)}/{len(roster)}] {split}/{sid} p_hazard={prediction["p_hazard"]:.6f}',flush=True)
                    rows=[completed[r['sample_id']] for r in rs]
                    if split=='pitt-val':
                        selection,hs,gs=choose_thresholds(rows)
                        save_json(out/'selection.json',selection)
                        save_json(out/'validation_threshold_metrics.json',{'hazard':hs,'groups':gs})
                        print('VALIDATION_SELECTION '+json.dumps(selection),flush=True)
                    reports[split]=export_split(out,split,rows,selection)
                recovered=recover_journal(out/'predictions.jsonl',roster)
                if recovered != completed or set(recovered)!=set(roster):raise ValueError('Final journal verification failed')
                usage=Counter();models=Counter()
                for line in (out/'predictions.jsonl').read_text().splitlines():
                    record=json.loads(line);response=record.get('response',{})
                    models[response.get('model','unknown')]+=1
                    for key in ['input_tokens','output_tokens','total_tokens']:usage[key]+=response.get('usage',{}).get(key,0)
                save_json(out/'results.json',{'model_requested':a.model,'response_models':dict(models),'usage_including_recorded_errors':dict(usage),
                    'selection':selection,'validation':reports['pitt-val'],'test':reports['pitt-test']})
                save_json(out/'final_verification.json',{'passed':True,'images':len(roster),'all_successes_saved_once':True,
                    'counts':counts,'input_hashes':binding['input_hashes'],'test_used_for_selection':False,
                    'prediction_journal_sha256':sha((out/'predictions.jsonl').read_bytes()),'completed_at_utc':now()})
                status('complete')
                print('COMPLETE '+json.dumps({'test_precision_at_46':reports['pitt-test']['hazard']['precision_at_k'],
                    'test_hazard_balanced_accuracy':reports['pitt-test']['hazard']['balanced_accuracy'],
                    'test_group_macro_balanced_accuracy':reports['pitt-test']['groups_on_all_true_hazards']['balanced_accuracy']}),flush=True)
        except BaseException as exc:
            status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',error_type=type(exc).__name__)
            raise


RUN_README = '''# OpenAI VLM on sidewalk-data-3

Independent zero-shot GPT-6.1 Sol requests, low reasoning, high image detail by
default. The exact requested model/settings are in run_config.json. No VLM weights
are trained. The previous five-class baseline and results are unchanged.

## Task and inputs

One hazard probability and four overlapping conditional group probabilities:
surface_and_level_problems, restricted_passage, missing_tactile_warning,
ramp_points_into_traffic. Group estimates are P(group | hazard, image); they are
not a softmax and are not multiplied by p_hazard. Elicited scores are not token
log probabilities or calibrated confidence guarantees. Full prompt/schema saved.

All 382 pitt-val images are evaluated first; all 380 pitt-test images follow.
Every original feature class, including Crosswalk, is included. Ground truth is
the dataset-3 hazard_label and four group_* columns, checked against the original
tags. Original classes are provenance only. Hand-collected ordinary sidewalks
are non-hazard. Missing tag annotations follow the existing dataset policy;
annotations and their absence are never sent to the model. Existing contradictory
duplicate annotations and tagless records are retained, as in the DINO experiment.

Exact existing 224x224 RGB PNG bytes are sent without additional image processing.
Only the fixed prompt and one image go to the API, store=False. No filenames,
labels, tags, coordinates, class proportions, or prior predictions are transmitted.
Credentials are read from OPENAI_API_KEY, never saved in the artifacts.

## Selection and metrics

Same threshold grid as DINO: 0.3, 0.5, 0.7, selected using pitt-val only. With one
fixed model/prompt, precision@46 does not vary with threshold; hazard balanced
accuracy selects the threshold, with lower threshold breaking exact ties. Each
group's threshold independently maximizes its binary balanced accuracy on ALL
true validation hazards. Selection is saved before the first test request.

Hazard precision@46 ranks ALL images by p_hazard descending, then sample_id
ascending for ties. A threshold never filters the top-46 ranking. Cutoff ties and
possible precision bounds are saved. Per-label precision/recall/F1, specificity,
accuracy, balanced accuracy, and confusion counts are retained without rounding.

Primary subtype score = mean of four binary balanced accuracies, evaluated on ALL
true hazards, irrespective of whether the VLM detected them. This differs from
the historical two-class Obstacle-vs-SurfaceProblem accuracy. Gated pipeline
metrics over all images and exact multi-label match are reported separately.
Group predictions are suppressed on predicted non-hazards only in the gated
pipeline. Hazard-with-no-group cases are flagged, never forced to an argmax.
Fixed-0.5 metrics are saved separately for reference, not used for model selection.

## Files and restart

- inputs/: complete manifests, label schema, labeling/preprocessing rules and build config.
- run_config.json, image_inventory.json, prompt.txt, output_schema.json, runner snapshot.
- predictions.jsonl: durable per-image metadata, raw response, model identity and token usage.
- pitt-{val,test}_probabilities.csv: saved incrementally, with five raw scores and ground truth.
- selection.json, validation_threshold_metrics.json: validation-only decisions.
- pitt-{val,test}_predictions.csv, *_hazard_ranking.csv, *_top46.csv.
- pitt-{val,test}_metrics.json, *_per_group_metrics.csv, *_metrics_fixed_0.5.json.
- results.json, final_verification.json: present only after full completion and checks.
- status.json: progress/ETA; run.log: server tmux execution log.

Resume with the saved runner and the same --out directory plus --resume. Successful
JSONL records are reused, CSVs rebuilt, and only remaining images requested. Input,
code, settings and dependency fingerprints must match. A directory lock prevents
two workers charging for the same run. API errors/refusals/invalid scores are
saved and stop the run. Only an incomplete final journal line is repaired on resume.
'''


if __name__=='__main__':
    main()
