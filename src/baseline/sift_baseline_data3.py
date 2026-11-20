#!/usr/bin/env python3
"""SIFT bag-of-words prototypes for dataset-3 hazard and overlapping tag groups.

Preserves the existing SIFT feature extractor, 200-word codebook, L1 histograms
and symmetric chi-square distance. Compares Pitt-only and pooled Other+Pitt on
validation, selecting hazard and group models independently before test scoring.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openai_baseline_data3 as task
try:
    import sift_tagger_baseline as engine
except ModuleNotFoundError:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
    import sift_tagger_baseline as engine

PLANS={'pitt_only':['pitt-train'],'pooled_other_and_pitt':['other-cities-train','pitt-train']}
TASKS=['hazard']+task.GROUPS
TRAIN_CAP=500
CODEBOOK_CAP=200
N_WORDS=200


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save_npz(path,**arrays):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def load_training(root,schema,evaluation):
    data={};seen={r['sample_id'] for rs in evaluation.values() for r in rs}
    pixels={r['output_rgb_pixel_sha256']:s for s,rs in evaluation.items() for r in rs}
    for split in ['other-cities-train','pitt-train']:
        with (root/(split+'.csv')).open(newline='') as f:rows=list(csv.DictReader(f))
        for row in rows:
            sid=row['sample_id'];rel=Path(row['image_path'])
            if sid in seen or row['split']!=split:raise ValueError('Training split/identity mismatch')
            seen.add(sid)
            if (row['city']=='pittsburgh')!=(split=='pitt-train') or not row['city']:raise ValueError('Wrong training city')
            if row['label_rule_version']!=schema['rule_version']:raise ValueError('Mixed label rules')
            if any(row[t] not in ('','0','1') for t in schema['all_tags']):raise ValueError('Invalid raw tag')
            groups=[int(any(row[t]=='1' for t in schema['groups'][g]['positive_tags'])) for g in task.GROUPS]
            if groups!=[task.bit(row[c]) for c in task.GROUP_COLUMNS] or task.bit(row['hazard_label'])!=int(any(groups)):
                raise ValueError('Training tag targets disagree')
            if row['class']!=('hazard' if any(groups) else 'non_hazard'):raise ValueError('Training class mismatch')
            if task.bit(row['is_manual_collection']) and any(groups):raise ValueError('Manual ordinary sidewalk is hazard')
            if rel.is_absolute() or len(rel.parts)!=3 or rel.parts[:2]!=(split,row['class']) or rel.suffix!='.png':
                raise ValueError('Invalid training path')
            p=(root/rel).resolve()
            if not p.is_relative_to(root/split/row['class']) or not p.is_file():raise ValueError('Missing/escaping training image')
            pixel=row['output_rgb_pixel_sha256']
            if pixel in pixels and pixels[pixel]!=split:raise ValueError('Pixel-identical record across splits')
            pixels[pixel]=split
        data[split]=rows
    return data


def select_training(pool,plan):
    """Overlapping group memberships are retained; group negatives are true hazards."""
    selections=[];sets={};codebook=set()
    for target in TASKS:
        eligible=pool if target=='hazard' else [r for r in pool if task.bit(r['hazard_label'])]
        column='hazard_label' if target=='hazard' else 'group_'+target
        for polarity in [0,1]:
            candidates=sorted([r for r in eligible if task.bit(r[column])==polarity],key=lambda r:r['sample_id'])
            seed=f'sift-data3-v1-{target}-{polarity}'
            random.Random(seed).shuffle(candidates)
            chosen=candidates[:TRAIN_CAP]
            if not chosen:raise ValueError(f'No training samples for {target}={polarity}')
            ids=[r['sample_id'] for r in chosen];sets[(target,polarity)]=ids
            for i,row in enumerate(chosen):
                used=i<CODEBOOK_CAP
                if used:codebook.add(row['sample_id'])
                selections.append({'plan':plan,'task':target,'polarity':polarity,'sample_id':row['sample_id'],
                    'split':row['split'],'image_path':row['image_path'],'image_sha256':row['output_file_sha256'],
                    'eligible_count':len(candidates),'selected_count':len(chosen),'selection_seed':seed,'in_codebook':int(used)})
    return selections,sets,codebook


def get_descriptors(rows,root,cache,workers):
    ids=[r['sample_id'] for r in rows]
    items=[{**r,'path':str(root/r['image_path'])} for r in rows]
    if cache.exists():
        with np.load(cache,allow_pickle=False) as z:
            if z['sample_ids'].tolist()!=ids:raise ValueError('Descriptor cache roster mismatch')
            desc=z['descriptors'];offsets=z['offsets'];ns=z['n_sparse'];nd=z['n_dense']
        if desc.ndim!=2 or desc.shape[1]!=128 or not np.isfinite(desc).all() or len(offsets)!=len(ids)+1 or offsets[0]!=0 or offsets[-1]!=len(desc):
            raise ValueError('Invalid descriptor cache')
        records=[{**r,'desc':np.ascontiguousarray(desc[offsets[i]:offsets[i+1]],dtype=np.float32),
                  'n_desc':int(offsets[i+1]-offsets[i]),'n_sparse':int(ns[i]),'n_dense':int(nd[i]),'w':224,'h':224}
                 for i,r in enumerate(items)]
    else:
        records=engine.extract_all(items,workers)
        if any(r['n_desc']<engine.MIN_DESCRIPTORS or (r['w'],r['h'])!=(224,224) for r in records):
            raise ValueError('Unsupported input dimensions or too few descriptors; no image is silently omitted')
        offsets=np.cumsum([0]+[r['n_desc'] for r in records])
        save_npz(cache,sample_ids=np.asarray(ids),descriptors=np.concatenate([r['desc'] for r in records]),offsets=offsets,
                 n_sparse=np.array([r['n_sparse'] for r in records]),n_dense=np.array([r['n_dense'] for r in records]))
    if any(r['n_desc']<engine.MIN_DESCRIPTORS for r in records):raise ValueError('Low-descriptor cached image')
    return records


def mean_prototypes(hist_by_id,sets):
    prototypes=[]
    for target in TASKS:
        pair=[]
        for polarity in [0,1]:
            hist=np.stack([hist_by_id[sid] for sid in sets[(target,polarity)]])
            mean=hist.mean(axis=0)
            if not np.isfinite(mean).all() or mean.sum()<=0:raise ValueError('Invalid prototype')
            pair.append(mean/mean.sum())
        prototypes.append(pair)
    return np.asarray(prototypes,dtype=np.float64)


def score_histograms(rows,histograms,prototypes):
    if histograms.shape!=(len(rows),N_WORDS) or prototypes.shape!=(5,2,N_WORDS):raise ValueError('Unexpected histogram/prototype shape')
    output=[]
    for row,hist in zip(rows,histograms):
        result={f:row[f] for f in task.IDENTITY_FIELDS}
        for j,target in enumerate(TASKS):
            neg=engine.chi2(hist,prototypes[j,0]);pos=engine.chi2(hist,prototypes[j,1]);margin=neg-pos
            if not -1-1e-12<=margin<=1+1e-12:raise ValueError('Chi-square margin out of range')
            # Affine storage transform only: not a probability or fitted calibration.
            result['p_'+target]=float(np.clip((margin+1)/2,0,1))
            result['margin_'+target]=margin;result['distance_negative_'+target]=neg;result['distance_positive_'+target]=pos
        task.validate_prediction({f:result[f] for f in task.PROBABILITY_FIELDS})
        output.append(result)
    return output


def best_cutoff(labels,scores):
    if set(map(int,labels))!={0,1}:raise ValueError('Need positive and negative validation labels')
    # Raw distance margins are not calibrated probabilities. Evaluate every
    # distinct validation decision boundary, plus nearest-prototype and extremes.
    cuts=sorted(set([0.,.5,float(np.nextafter(max(scores),np.inf)),*map(float,scores)]))
    table=[{'threshold':t,**task.binary_metrics(labels,[int(s>=t) for s in scores])} for t in cuts]
    best=min(table,key=lambda m:(-round(m['balanced_accuracy'],12),m['threshold']))
    return best['threshold'],table


def select_thresholds(rows):
    if not rows or {r['split'] for r in rows}!={'pitt-val'}:raise ValueError('Validation only for threshold selection')
    thresholds=[];table=[]
    for target in TASKS:
        subset=rows if target=='hazard' else [r for r in rows if task.bit(r['hazard_label'])]
        col='hazard_label' if target=='hazard' else 'group_'+target
        t,scores=best_cutoff([task.bit(r[col]) for r in subset],[r['p_'+target] for r in subset])
        thresholds.append(t);table.extend({'task':target,**m} for m in scores)
    return thresholds[0],thresholds[1:],table


def choose_plans(summaries):
    if {r['plan'] for r in summaries}!=set(PLANS):raise ValueError('Both predeclared training plans required')
    order={p:i for i,p in enumerate(PLANS)}
    h=min(summaries,key=lambda r:(-r['hazard_top46_hits'],-round(r['hazard_balanced_accuracy'],12),order[r['plan']]))
    g=min(summaries,key=lambda r:(-round(r['group_macro_balanced_accuracy'],12),order[r['plan']]))
    return {'hazard_model_plan':h['plan'],'group_model_plan':g['plan'],
            'hazard_threshold':h['hazard_threshold'],'group_thresholds':json.loads(g['group_thresholds']),
            'hazard_margin_threshold':2*h['hazard_threshold']-1,
            'group_margin_thresholds':[2*t-1 for t in json.loads(g['group_thresholds'])],
            'group_order':task.GROUPS,'k':task.K,'validation_split':'pitt-val','test_used_for_selection':False,
            'hazard_rule':'Maximize validation top-46 hits; break ties with hazard balanced accuracy at its validation-selected cutoff, then prefer Pitt-only.',
            'group_rule':'Maximize macro binary balanced accuracy on ALL true validation hazards; each group cutoff selected independently; final ties prefer Pitt-only.',
            'threshold_search':'All distinct validation scores for the task, plus 0, 0.5 and nextafter(max,+inf); lower cutoff breaks exact BA ties.',
            'score_transform':'p_* columns store (1 + chi2_distance_negative - chi2_distance_positive)/2; NOT calibrated probabilities.',
            'threshold_grid_note':'Distance-derived scores require cutoffs on their native scale; unlike DINO/VLM/CLIP this uses all validation decision boundaries, not 0.3/0.5/0.7.',
            'training_order_note':'Pooled Other+Pitt is a union training pool, not sequential neural fine-tuning.'}


def combine_predictions(by_plan,selection):
    hrows=by_plan[selection['hazard_model_plan']];grows=by_plan[selection['group_model_plan']]
    if [r['sample_id'] for r in hrows]!=[r['sample_id'] for r in grows]:raise ValueError('Selected model rosters differ')
    output=[]
    for h,g in zip(hrows,grows):
        if any(h[k]!=g[k] for k in task.IDENTITY_FIELDS):raise ValueError('Selected model ground truth differs')
        row=dict(h)
        for group in task.GROUPS:
            for prefix in ['p_','margin_','distance_negative_','distance_positive_']:row[prefix+group]=g[prefix+group]
        row['hazard_model_plan']=selection['hazard_model_plan'];row['group_model_plan']=selection['group_model_plan']
        output.append(row)
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=task.DEFAULT_ROOT)
    ap.add_argument('--reference-vlm-run',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--workers',type=int,default=16)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if a.workers<1:ap.error('Positive worker count required')
    root=a.data_root.resolve();schema,data=task.load_inputs(root);training=load_training(root,schema,data)
    fits={};all_train={r['sample_id']:r for rs in training.values() for r in rs};selected_ids=set()
    for plan,splits in PLANS.items():
        pool=[r for split in splits for r in training[split]]
        selections,sets,codebook=select_training(pool,plan)
        ids={sid for values in sets.values() for sid in values};selected_ids|=ids
        fits[plan]={'selections':selections,'sets':sets,'codebook':codebook,'ids':ids}
    eval_inventory=[{'sample_id':r['sample_id'],'split':s,'image_path':r['image_path'],'sha256':r['output_file_sha256']}
                    for s,rs in data.items() for r in rs]
    reference={'checked':False}
    if a.reference_vlm_run:
        cfg=json.loads((a.reference_vlm_run/'run_config.json').read_text())['binding']
        if json.loads((a.reference_vlm_run/'image_inventory.json').read_text())!=eval_inventory:raise ValueError('Evaluation roster differs from VLM')
        for name,h in cfg['input_hashes'].items():
            if file_sha(root/name)!=h:raise ValueError('Labels/preprocessing differ from VLM')
        reference={'checked':True,'run':str(a.reference_vlm_run.resolve()),'exact_evaluation_images_and_targets':True}
    from PIL import Image
    train_rows=[all_train[sid] for sid in sorted(selected_ids)]
    for row in train_rows:
        p=root/row['image_path']
        if file_sha(p)!=row['output_file_sha256']:raise ValueError('Training image hash mismatch')
        with Image.open(p) as im:
            im.load()
            if im.size!=(224,224) or im.mode!='RGB' or im.format!='PNG':raise ValueError('Training preprocessing mismatch')
    counts={s:{'images':len(rs),'hazard':sum(task.bit(r['hazard_label']) for r in rs)} for s,rs in {**training,**data}.items()}
    fit_counts={p:{'training_images_available':sum(len(training[s]) for s in PLANS[p]),
                  'unique_selected_training_images':len(f['ids']),'unique_codebook_images':len(f['codebook']),
                  'prototype_sizes':{target:{str(pol):len(f['sets'][(target,pol)]) for pol in [0,1]} for target in TASKS}}
                for p,f in fits.items()}
    if a.dry_run:
        print(json.dumps({'dry_run':True,'counts':counts,'fit_counts':fit_counts,'selected_train_union':len(train_rows),
                          'reference':reference,'training_started':False},indent=2));return
    if a.out is None:ap.error('--out required')
    out=a.out.resolve()
    if a.resume:
        if not (out/'run_config.json').is_file():ap.error('--resume requires an existing run')
    else:out.mkdir(parents=True,exist_ok=False)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        input_names=[s+'.csv' for s in ['other-cities-train','pitt-train',*task.SPLITS]]+['label_schema.json','build_config.json','LABELING_RULES.md','PREPROCESSING.md']
        inventory=[{'sample_id':r['sample_id'],'split':r['split'],'image_path':r['image_path'],'sha256':r['output_file_sha256']}
                   for r in train_rows]+eval_inventory
        binding={'task_version':'sidewalk-data-3-sift-v1','data_root':str(root),'input_hashes':{n:file_sha(root/n) for n in input_names},
                 'inventory_sha256':task.sha(task.canonical(inventory).encode()),'plans':PLANS,'fit_counts':fit_counts,
                 'source_hashes':{Path(p).name:file_sha(p) for p in [__file__,engine.__file__,task.__file__]},
                 'reference':reference,'training_cap_per_task_polarity':TRAIN_CAP,'codebook_cap_per_task_polarity':CODEBOOK_CAP,
                 'visual_words':N_WORDS,'grid_step':engine.GRID_STEP,'sparse_requested':engine.CAP_SPARSE,'dense_cap':engine.CAP_DENSE,
                 'descriptor_cap':engine.DESC_CAP,'minimum_descriptors':engine.MIN_DESCRIPTORS,'workers':a.workers,
                 'opencv':cv2.__version__,'numpy':np.__version__,'python':sys.version,
                 'thread_environment':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']},
                 'kmeans':'OpenCV BOWKMeansTrainer, kmeans++, seed 0, 1 attempt, max20 iterations, epsilon0.1; training descriptors only',
                 'prototypes':'Per task/polarity mean of selected training L1 histograms, then L1 normalize; group models fit only true hazards',
                 'ranking':'p_hazard (affine distance margin) descending, sample_id ascending',
                 'selection':'Independent validation selection for hazard and groups; raw-distance cutoffs use all validation boundaries; no test selection',
                 'split_counts':counts}
        if a.resume:
            old=json.loads((out/'run_config.json').read_text())['binding']
            if old!=binding:raise ValueError('Resume code/data/settings mismatch: '+str([k for k in binding if old.get(k)!=binding[k]]))
        else:
            (out/'inputs').mkdir();(out/'code').mkdir();(out/'descriptor_cache').mkdir();(out/'inference').mkdir()
            for name in input_names:shutil.copy2(root/name,out/'inputs'/name)
            for source in [__file__,engine.__file__,task.__file__]:shutil.copy2(source,out/'code'/Path(source).name)
            task.save_json(out/'run_config.json',{'created_at_utc':task.now(),'binding':binding})
            task.save_json(out/'image_inventory.json',inventory);task.save_json(out/'evaluation_image_inventory.json',eval_inventory)
            (out/'README.md').write_text(RUN_README);(out/'opencv_build.txt').write_text(cv2.getBuildInformation())
            for plan,fit in fits.items():
                folder=out/plan;folder.mkdir();task.save_csv(folder/'train_selection.csv',fit['selections'])
        started=time.monotonic()
        def status(state,**extra):task.save_json(out/'status.json',{'state':state,'updated_at_utc':task.now(),'elapsed_seconds':time.monotonic()-started,**extra})
        try:
            engine.check_sift();status('training_descriptors',training_images=len(train_rows))
            records=get_descriptors(train_rows,root,out/'descriptor_cache/train.npz',a.workers)
            byid={r['sample_id']:r for r in records}
            models={}
            for plan,fit in fits.items():
                status('fit_codebook_and_prototypes',plan=plan)
                p=out/plan/'model.npz';ids=sorted(fit['ids'])
                if p.exists():
                    with np.load(p,allow_pickle=False) as z:
                        vocab=z['vocabulary'];protos=z['prototypes'];hist=z['training_histograms']
                        if z['training_ids'].tolist()!=ids:raise ValueError('Saved model training roster mismatch')
                else:
                    chosen=[{**byid[sid],'in_codebook':sid in fit['codebook']} for sid in ids]
                    if any(r['split'] not in PLANS[plan] for r in chosen):raise ValueError('Held-out image in fit')
                    vocab=engine.build_vocabulary(chosen,N_WORDS);engine.l1_histograms(chosen,vocab)
                    if any(r['hist'] is None for r in chosen):raise ValueError('Training histogram missing')
                    hist=np.stack([r['hist'] for r in chosen]);protos=mean_prototypes(dict(zip(ids,hist)),fit['sets'])
                    save_npz(p,vocabulary=vocab,prototypes=protos,training_histograms=hist,training_ids=np.asarray(ids),
                             codebook_ids=np.asarray(sorted(fit['codebook'])),task_order=np.asarray(TASKS))
                if vocab.shape!=(N_WORDS,128) or protos.shape!=(5,2,N_WORDS) or not np.isfinite(protos).all():raise ValueError('Invalid saved model shape')
                reproduced=mean_prototypes(dict(zip(ids,hist)),fit['sets'])
                if not np.allclose(protos,reproduced,rtol=0,atol=1e-12):raise ValueError('Prototypes differ from fitted training means')
                models[plan]=(vocab,protos)
            status('validation_descriptors')
            val_records=get_descriptors(data['pitt-val'],root,out/'descriptor_cache/pitt-val.npz',a.workers)
            val_scores={};summaries=[]
            for plan,(vocab,protos) in models.items():
                local=[dict(r) for r in val_records];engine.l1_histograms(local,vocab)
                hist=np.stack([r['hist'] for r in local]);rows=score_histograms(data['pitt-val'],hist,protos)
                save_npz(out/plan/'pitt-val_histograms.npz',histograms=hist,sample_ids=np.asarray([r['sample_id'] for r in rows]))
                task.save_csv(out/plan/'pitt-val_scores.csv',rows)
                ht,gt,threshold_table=select_thresholds(rows)
                task.save_csv(out/plan/'validation_threshold_metrics.csv',threshold_table)
                metrics=task.evaluate(rows,ht,gt);task.save_json(out/plan/'validation_metrics.json',metrics)
                summaries.append({'plan':plan,'hazard_top46_hits':metrics['hazard']['topk_hits'],
                    'hazard_precision_at_46':metrics['hazard']['precision_at_k'],'hazard_balanced_accuracy':metrics['hazard']['balanced_accuracy'],
                    'group_macro_balanced_accuracy':metrics['groups_on_all_true_hazards']['balanced_accuracy'],
                    'hazard_threshold':ht,'group_thresholds':json.dumps(gt)})
                val_scores[plan]=rows
            task.save_csv(out/'validation_model_selection.csv',summaries)
            selection=choose_plans(summaries);task.save_json(out/'selection.json',selection)
            task.save_json(out/'inference/selection.json',selection);shutil.copy2(root/'label_schema.json',out/'inference/label_schema.json')
            selected=set([selection['hazard_model_plan'],selection['group_model_plan']])
            for plan in selected:shutil.copy2(out/plan/'model.npz',out/'inference'/(plan+'.npz'))
            print('VALIDATION_SELECTION '+json.dumps(selection),flush=True)
            val_combined=combine_predictions(val_scores,selection);task.save_csv(out/'pitt-val_scores.csv',val_combined)
            reports={'pitt-val':task.export_split(out,'pitt-val',val_combined,selection)}
            status('test_descriptors',selection_frozen=True)
            test_records=get_descriptors(data['pitt-test'],root,out/'descriptor_cache/pitt-test.npz',a.workers)
            test_scores={}
            for plan in sorted(selected):
                vocab,protos=models[plan];local=[dict(r) for r in test_records];engine.l1_histograms(local,vocab)
                hist=np.stack([r['hist'] for r in local]);rows=score_histograms(data['pitt-test'],hist,protos)
                save_npz(out/plan/'pitt-test_histograms.npz',histograms=hist,sample_ids=np.asarray([r['sample_id'] for r in rows]))
                task.save_csv(out/plan/'pitt-test_scores.csv',rows);test_scores[plan]=rows
            test_combined=combine_predictions(test_scores,selection);task.save_csv(out/'pitt-test_scores.csv',test_combined)
            reports['pitt-test']=task.export_split(out,'pitt-test',test_combined,selection)
            # Reproduce every selected score from saved model/histogram arrays.
            for split,expected in [('pitt-val',val_combined),('pitt-test',test_combined)]:
                reproduced={}
                for plan in selected:
                    with np.load(out/plan/'model.npz',allow_pickle=False) as z:protos=z['prototypes']
                    with np.load(out/plan/(split+'_histograms.npz'),allow_pickle=False) as z:
                        hist=z['histograms'];ids=z['sample_ids'].tolist()
                    if ids!=[r['sample_id'] for r in data[split]]:raise ValueError('Saved histogram roster mismatch')
                    reproduced[plan]=score_histograms(data[split],hist,protos)
                if combine_predictions(reproduced,selection)!=expected:raise ValueError('Score reproduction mismatch')
                with (out/(split+'_scores.csv')).open(newline='') as f:saved=list(csv.DictReader(f))
                for row in saved:
                    for k in task.PROBABILITY_FIELDS:row[k]=float(row[k])
                if task.evaluate(saved,selection['hazard_threshold'],selection['group_thresholds'])!=reports[split]:raise ValueError('Metric reproduction mismatch')
            for name,h in binding['input_hashes'].items():
                if file_sha(root/name)!=h:raise ValueError('Input metadata changed during run')
            for row in train_rows+[r for rs in data.values() for r in rs]:
                if file_sha(root/row['image_path'])!=row['output_file_sha256']:raise ValueError('Image changed during run')
            result={'method':'SIFT bag-of-visual-words nearest prototypes','selection':selection,'reference':reference,
                    'validation_configurations':summaries,'validation':reports['pitt-val'],'test':reports['pitt-test'],
                    'fit_counts':fit_counts,'wall_seconds':time.monotonic()-started}
            task.save_json(out/'results.json',result)
            task.save_json(out/'final_verification.json',{'passed':True,'validation_images':382,'test_images':380,
                'selected_training_images':len(train_rows),'all_evaluation_records_retained':True,'no_cross_split_pixel_duplicates':True,
                'no_validation_or_test_images_fit_codebooks_or_prototypes':True,'prototypes_reproduced_from_training_means':True,
                'scores_reproduced_from_saved_models_and_histograms':True,'metrics_reproduced_from_saved_csvs':True,
                'image_hashes_rechecked':True,'test_used_for_selection':False,'finished_at_utc':task.now()})
            status('complete',validation_images=382,test_images=380)
            print('COMPLETE '+json.dumps({'test_precision_at_46':reports['pitt-test']['hazard']['precision_at_k'],
                'test_hazard_balanced_accuracy':reports['pitt-test']['hazard']['balanced_accuracy'],
                'test_group_macro_balanced_accuracy':reports['pitt-test']['groups_on_all_true_hazards']['balanced_accuracy']}),flush=True)
        except BaseException as exc:status('error',error_type=type(exc).__name__);raise


RUN_README='''# SIFT on sidewalk-data-3

Server search found only older SIFT results (latest November 17 five-class run),
with no existing dataset-3 result or active SIFT job. This is a new training and
evaluation; the previous scripts, worktree, and results are preserved.

## Method and training

Unchanged legacy SIFT extractor: existing 224x224 RGB images converted to grayscale,
40 requested sparse features plus up to 40 dense descriptors on a 32-pixel grid,
80 total cap, at least 5 descriptors. OpenCV can exceed the sparse request on ties;
the original concatenation/truncation behavior is retained. No geometric recrop.
200-word OpenCV kmeans++ codebook, seed 0, one attempt, 20 iterations, epsilon 0.1.
Per-image histograms are L1 normalized. Prototypes are means of training histograms,
L1 normalized again. Distances use symmetric chi-square. No SVM or neural net.

Two predeclared training pools: Pitt only, and pooled Other+Pitt. The pooled variant
is not sequential fine-tuning. Within each pool, five independent binary prototype
pairs are fitted: hazard, surface_and_level_problems, restricted_passage,
missing_tactile_warning, ramp_points_into_traffic. Hazard uses all eligible training
labels; group pairs use only TRUE training hazards, including overlapping group
memberships. Non-hazards never enter the group positive or negative prototypes.

Preserve the original sample caps: up to 500 training examples per task/polarity,
with up to the first 200 contributing to that plan's shared codebook. Candidate
rows are sorted by sample_id then shuffled with random.Random using the saved
task/polarity seed. An image contributes to each applicable prototype, but at most
once to the shared vocabulary. train_selection.csv identifies every contribution,
eligible/selected counts and codebook membership. Actual counts are in run_config.

## Targets, scores, and model selection

Dataset-3 hazard_label/group_* columns, verified against tags, are authoritative;
old feature classes are provenance only. All validation/test images, including
Crosswalk and manually collected ordinary sidewalks, are retained. Missing-tag
rules and conflicting annotations within splits are unchanged. Every fitted image
and every validation/test image is hash checked; no cross-split pixel duplicates.

Each pair's raw score is distance_to_negative minus distance_to_positive. Larger
means more positive. p_* columns use the affine transform (1+margin)/2 solely for
the common evaluator: these are NOT probabilities or fitted calibration. The
four group scores overlap and do not sum to one. Raw distances and margins are
saved too. Independent hazard/group predictions may disagree; the gated pipeline
and hazard-without-group flags expose those cases instead of forcing an argmax.

All model selection uses pitt-val. Hazard: largest precision@46, then largest
balanced accuracy at a validation-selected threshold. Groups: largest mean of
four binary balanced accuracies on ALL true hazards, with an independent cutoff
for each group. Final plan ties prefer Pitt only; cutoff ties choose the lower
threshold. Hazard and group models may select different training pools/codebooks.

Because chi-square margins have a different scale from neural probabilities,
cutoff candidates are all observed validation scores plus 0, 0.5 (nearest
prototype), and the next representable score above the maximum. This differs from
the 0.3/0.5/0.7 grid used for DINO/VLM/CLIP; the selection objectives and evaluation
populations match. Selection and chosen model artifacts are saved BEFORE test
descriptor extraction. There is one deterministic sample/codebook per plan, not
a multi-seed sweep. No prompt or parameter is selected using test performance.

## Metrics and artifacts

All 382 validation and 380 test images are scored. Rank p_hazard descending, then
sample_id ascending; top 46 is never filtered by the hazard threshold. Tag-group
macro balanced accuracy is mean binary BA across all four groups on ALL 174 true
test hazards, regardless of hazard detection. Per-group precision/recall/F1,
specificity, confusion counts, exact match, and gated pipeline metrics are saved.
Fixed-0.5 metrics mean nearest-prototype decisions and are provided separately.
The shared dataset/evaluation helper is the updated VLM code; no API is called.

inputs/, code/, run_config.json, image_inventory.json and opencv_build.txt retain
data, code, software and fit provenance. Each plan has train_selection.csv,
model.npz (vocabulary, five prototype pairs, selected training IDs/histograms),
validation scores/threshold sweep/metrics, and held-out histograms. The root has
validation_model_selection.csv, selection.json, pitt-{val,test}_scores.csv,
*_predictions.csv, *_hazard_ranking.csv, *_top46.csv and complete metrics.
inference/ contains the selected model file(s), label schema and selection rules.

Descriptor caches are atomic NPZ files keyed by the run's input/code fingerprints.
--resume reuses caches/fitted models only if the full binding matches. A run lock
prevents concurrent writers. status.json/run.log track progress. results.json and
final_verification.json are written after saved-model score reproduction, saved-CSV
metric reproduction, training-prototype checks, and final input hash checks pass.
'''


if __name__=='__main__':main()
