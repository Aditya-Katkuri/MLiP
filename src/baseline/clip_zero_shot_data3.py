#!/usr/bin/env python3
"""Zero-shot CLIP on sidewalk-data-3; independent binary hazard and tag-group tasks.

Pinned cached ViT-B/16, fixed prompt pairs, validation-only thresholds, held-out
test evaluation. Reuses the dataset contract and metrics from the updated VLM.
No API, API key, training, prompt sweep, or network access is required.
"""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import openai_baseline_data3 as task

MODEL='openai/clip-vit-base-patch16'
REVISION='57c216476eefef5ab752ec549e440a49ae4ae5f3'
MODEL_PATH=Path('/data/hf_cache/hub/models--openai--clip-vit-base-patch16/snapshots')/REVISION
TASKS=['hazard']+task.GROUPS
POLARITIES=['negative','positive']
# Exactly two fixed prompts per polarity and task. Negative means the named
# attribute is absent, not that all four other attributes must also be absent.
PROMPTS={
 'hazard':{
  'negative':[
   'a street photo of an even sidewalk with ample walking space and a curb ramp with tactile paving aligned with the pedestrian crossing',
   'a street photo of an ordinary level sidewalk and accessible street corner with sufficient passage and turning space'],
  'positive':[
   'a street photo of a sidewalk with uneven pavement, a narrow passage, missing tactile paving at a curb ramp, or a curb ramp pointing into traffic',
   'a street photo showing a sidewalk surface or level defect, restricted walking space, a ramp without a tactile warning, or a ramp misaligned with a pedestrian crossing']},
 'surface_and_level_problems':{
  'negative':[
   'a street photo of an even, level sidewalk surface and a curb ramp flush with the street',
   'a street photo of intact flat pavement with a smooth pedestrian walking surface'],
  'positive':[
   'a street photo of broken, bumpy or uneven sidewalk pavement, an abrupt height difference, or an uncovered manhole',
   'a street photo of a steep or slanted sidewalk, stairs, pooled water, or a curb ramp with a raised edge at the street']},
 'restricted_passage':{
  'negative':[
   'a street photo of a wide sidewalk with ample space for pedestrians to pass by street furniture',
   'a street photo of a spacious curb ramp landing with enough room to turn and pass'],
  'positive':[
   'a street photo of a narrow sidewalk or pedestrian passage with very little space to pass',
   'a street photo of a cramped curb ramp landing with insufficient space to turn or maneuver']},
 'missing_tactile_warning':{
  'negative':[
   'a street photo of a curb ramp at a crossing with a visible textured tactile warning surface',
   'a street photo of an ordinary mid-block sidewalk away from a crossing or curb ramp'],
  'positive':[
   'a street photo of a bare concrete curb ramp at a pedestrian crossing without tactile warning paving',
   'a street photo of a curb ramp entering a roadway with the detectable tactile warning surface missing']},
 'ramp_points_into_traffic':{
  'negative':[
   'a street photo of a curb ramp aligned directly with the intended pedestrian crosswalk',
   'a street photo of an ordinary sidewalk continuing along the street away from a curb ramp'],
  'positive':[
   'a street photo of a curb ramp pointing into a vehicle traffic lane instead of the pedestrian crossing',
   'a street photo of a misaligned curb ramp directing pedestrians into the road outside the intended crosswalk']},
}


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def normalize(x):
    x=np.asarray(x,dtype=np.float32)
    norms=np.linalg.norm(x,axis=-1,keepdims=True)
    if not np.isfinite(x).all() or (norms<=0).any():raise ValueError('Invalid embedding')
    return x/norms


def make_prototypes(prompt_features):
    if prompt_features.ndim!=2 or len(prompt_features)!=len(TASKS)*len(POLARITIES)*2:
        raise ValueError('Expected two prompts for both polarities of all five tasks')
    # Same per-prompt normalization, mean, re-normalization as the older CLIP baseline.
    return normalize(normalize(prompt_features).reshape(len(TASKS),2,2,-1).mean(axis=2))


def score_pairs(features, prototypes, scale):
    if features.ndim!=2 or prototypes.shape!=(len(TASKS),2,features.shape[1]):
        raise ValueError('Unexpected image/text feature shape')
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid CLIP logit scale')
    cosines=np.einsum('nd,tpd->ntp',features,prototypes)
    logits=scale*cosines
    weights=np.exp(logits-logits.max(axis=2,keepdims=True))
    probabilities=weights[:,:,1]/weights.sum(axis=2)
    if not np.isfinite(probabilities).all():raise ValueError('Non-finite scores')
    return probabilities,cosines


def save_array(path,value):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('wb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def save_chunk(path,features,sample_ids):
    path=Path(path);tmp=path.with_name('.'+path.name+'.tmp')
    with tmp.open('wb') as f:
        np.savez_compressed(f,features=features,sample_ids=np.asarray(sample_ids,dtype=str))
        f.flush();os.fsync(f.fileno())
    tmp.replace(path)


def load_chunk(path,sample_ids,dim):
    with np.load(path,allow_pickle=False) as z:
        features=z['features'];ids=z['sample_ids'].tolist()
    if ids!=sample_ids or features.shape!=(len(sample_ids),dim) or not np.isfinite(features).all():
        raise ValueError('Cached feature chunk identity/shape mismatch')
    if not np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5):
        raise ValueError('Cached features not normalized')
    return features


def verify_reference(reference,root,inventory):
    if reference is None:return {'checked':False}
    reference=Path(reference)
    binding=json.loads((reference/'run_config.json').read_text())['binding']
    expected=json.loads((reference/'image_inventory.json').read_text())
    if inventory!=expected:raise ValueError('CLIP and VLM input identities or image bytes differ')
    for name,h in binding['input_hashes'].items():
        if file_sha(root/name)!=h:raise ValueError('CLIP and VLM manifest/label/preprocessing mismatch: '+name)
    return {'checked':True,'reference_vlm_run':str(reference.resolve()),'exact_images_targets_and_preprocessing':True}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,default=task.DEFAULT_ROOT)
    ap.add_argument('--model-path',type=Path,default=MODEL_PATH)
    ap.add_argument('--reference-vlm-run',type=Path)
    ap.add_argument('--out',type=Path)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--resume',action='store_true')
    a=ap.parse_args()
    if a.batch_size<1 or a.threads<1:ap.error('Positive batch size and thread count required')
    root=a.data_root.resolve();model_path=a.model_path.resolve()
    schema,data=task.load_inputs(root)
    inventory=[{'sample_id':r['sample_id'],'split':s,'image_path':r['image_path'],'sha256':r['output_file_sha256']}
               for s,rs in data.items() for r in rs]
    reference=verify_reference(a.reference_vlm_run,root,inventory)
    counts={s:{'images':len(rs),'hazard':sum(task.bit(r['hazard_label']) for r in rs),
               'non_hazard':sum(not task.bit(r['hazard_label']) for r in rs),
               'groups':{g:sum(task.bit(r['group_'+g]) for r in rs) for g in task.GROUPS}} for s,rs in data.items()}
    if a.dry_run:
        print(json.dumps({'dry_run':True,'model':MODEL,'revision':REVISION,'split_counts':counts,
                          'tasks':TASKS,'text_pairs':{t:{p:len(PROMPTS[t][p]) for p in POLARITIES} for t in TASKS},
                          'reference':reference,'api_requests':0,'network_access_required':False},indent=2));return
    if a.out is None:ap.error('--out is required')
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from PIL import Image
    from transformers import CLIPModel,CLIPProcessor
    torch.set_num_threads(a.threads);torch.manual_seed(42)
    out=a.out.resolve()
    if a.resume:
        if not (out/'run_config.json').is_file():ap.error('--resume requires an existing run')
    else:out.mkdir(parents=True,exist_ok=False)
    with (out/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        input_names=['pitt-val.csv','pitt-test.csv','label_schema.json','build_config.json','LABELING_RULES.md','PREPROCESSING.md']
        model_files=['pytorch_model.bin','config.json','preprocessor_config.json','tokenizer_config.json',
                     'special_tokens_map.json','vocab.json','merges.txt','tokenizer.json']
        binding={'task_version':'sidewalk-data-3-clip-zero-shot-v1','model':MODEL,'revision':REVISION,'model_path':str(model_path),
                 'model_file_hashes':{n:file_sha(model_path/n) for n in model_files},
                 'data_root':str(root),'input_hashes':{n:file_sha(root/n) for n in input_names},
                 'image_inventory_sha256':task.sha(task.canonical(inventory).encode()),
                 'runner_sha256':file_sha(__file__),'evaluation_helper_sha256':file_sha(task.__file__),
                 'split_counts':counts,'reference':reference,'prompts':PROMPTS,'tasks':TASKS,'polarities':POLARITIES,
                 'batch_size':a.batch_size,'threads':a.threads,'device':'cpu','precision':'float32','seed':42,
                 'threshold_grid':task.THRESHOLDS,'k':task.K,'mode':'zero_shot','training':'none',
                 'packages':{p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','Pillow']},
                 'python_version':sys.version,'local_files_only':True,
                 'preprocessing':'Existing RGB 224x224 images; CLIP rescaling and normalization only; do_resize=False, do_center_crop=False.',
                 'score_definition':'For each task, softmax over negative/positive text cosine similarities times exp(CLIP logit_scale); return positive score. Five independent comparisons, not a five-class softmax.',
                 'group_score_interpretation':'Independent group-presence scores used for the conditional group task on true hazards; not calibrated conditional probability estimates.',
                 'text_ensemble':'L2 normalize each prompt; average two prompts per polarity; L2 normalize the mean.',
                 'selection':'Fixed prompts before inference. Independent thresholds selected on pitt-val; no test tuning.'}
        if a.resume:
            old=json.loads((out/'run_config.json').read_text())
            if old['binding']!=binding:
                raise ValueError('Resume settings/input mismatch: '+str([k for k in binding if old['binding'].get(k)!=binding[k]]))
        else:
            (out/'inputs').mkdir();(out/'feature_chunks').mkdir();(out/'model_config').mkdir()
            for name in input_names:shutil.copy2(root/name,out/'inputs'/name)
            for name in model_files:
                if name!='pytorch_model.bin':shutil.copy2(model_path/name,out/'model_config'/name)
            shutil.copy2(__file__,out/'clip_zero_shot_data3.py')
            shutil.copy2(task.__file__,out/'openai_baseline_data3.py')
            task.save_json(out/'run_config.json',{'created_at_utc':task.now(),'binding':binding})
            task.save_json(out/'image_inventory.json',inventory)
            task.save_json(out/'prompts.json',PROMPTS)
            (out/'README.md').write_text(RUN_README)
        started=time.monotonic();completed=0;total=len(inventory)
        def status(state,**extra):
            elapsed=time.monotonic()-started
            task.save_json(out/'status.json',{'state':state,'updated_at_utc':task.now(),'completed':completed,'total':total,
                'elapsed_seconds':elapsed,'eta_seconds':elapsed/completed*(total-completed) if completed else None,**extra})
        try:
            status('loading_model')
            print('Loading pinned cached CLIP ViT-B/16 on CPU; no network or API.',flush=True)
            model=CLIPModel.from_pretrained(str(model_path),local_files_only=True,use_safetensors=False).eval().to('cpu')
            processor=CLIPProcessor.from_pretrained(str(model_path),local_files_only=True)
            texts=[p for t in TASKS for pol in POLARITIES for p in PROMPTS[t][pol]]
            tokens=processor.tokenizer(texts,padding=True,truncation=False,return_tensors='pt')
            lengths=tokens['attention_mask'].sum(dim=1).tolist()
            if max(lengths)>model.config.text_config.max_position_embeddings:raise ValueError('Prompt exceeds CLIP token limit; no silent truncation allowed')
            with torch.inference_mode():
                raw=model.get_text_features(**tokens)
                raw=getattr(raw,'pooler_output',raw).cpu().numpy()
                scale=float(model.logit_scale.exp().cpu())
            text_features=make_prototypes(raw);dim=text_features.shape[-1]
            save_array(out/'prompt_features.npy',normalize(raw));save_array(out/'text_features.npy',text_features)
            task.save_json(out/'text_encoding.json',{'scale':scale,'dimension':dim,'task_order':TASKS,
                'polarity_order':POLARITIES,'prompts':texts,'token_lengths':lengths,'truncated':False})
            selection=None;reports={};all_features=[];all_rows=[]
            with (out/'predictions.jsonl').open('w') as journal:
                for split,source_rows in data.items():
                    if split=='pitt-test' and selection is None:raise RuntimeError('Validation thresholds must precede test inference')
                    rows=[];features_split=[];cosines_split=[]
                    chunk_dir=out/'feature_chunks'/split;chunk_dir.mkdir(exist_ok=True)
                    with (out/(split+'_probabilities.csv')).open('w',newline='') as cf:
                        writer=csv.DictWriter(cf,fieldnames=task.RAW_FIELDS);writer.writeheader()
                        for begin in range(0,len(source_rows),a.batch_size):
                            batch=source_rows[begin:begin+a.batch_size];ids=[r['sample_id'] for r in batch]
                            chunk=chunk_dir/(f'{begin:06d}.npz')
                            if chunk.exists():features=load_chunk(chunk,ids,dim)
                            else:
                                pictures=[]
                                for r in batch:
                                    p=root/r['image_path']
                                    if file_sha(p)!=r['output_file_sha256']:raise ValueError('Image changed after preflight')
                                    with Image.open(p) as im:pictures.append(im.copy())
                                inputs=processor.image_processor(images=pictures,do_resize=False,do_center_crop=False,return_tensors='pt')
                                if tuple(inputs['pixel_values'].shape)!=(len(batch),3,224,224):raise ValueError('Unexpected CLIP input shape')
                                with torch.inference_mode():
                                    f=model.get_image_features(**inputs)
                                    features=normalize(getattr(f,'pooler_output',f).cpu().numpy())
                                save_chunk(chunk,features,ids)
                            probabilities,cosines=score_pairs(features,text_features,scale)
                            features_split.append(features);cosines_split.append(cosines)
                            for r,p in zip(batch,probabilities):
                                scores=task.validate_prediction(dict(zip(task.PROBABILITY_FIELDS,map(float,p))))
                                result={**{f:r[f] for f in task.IDENTITY_FIELDS},**scores}
                                journal.write(json.dumps({'sample_id':r['sample_id'],'metadata':r,'image_sha256':r['output_file_sha256'],
                                                          'prediction':scores,'status':'ok'})+'\n')
                                writer.writerow(result);rows.append(result)
                            journal.flush();os.fsync(journal.fileno());cf.flush();os.fsync(cf.fileno())
                            completed+=len(batch);status('running',current_split=split)
                            print(f'{task.now()} [{completed}/{total}] {split}: saved scores and recoverable feature chunks',flush=True)
                    features_split=np.concatenate(features_split);save_array(out/(split+'_image_features.npy'),features_split)
                    save_array(out/(split+'_cosine_similarities.npy'),np.concatenate(cosines_split))
                    all_features.append(features_split);all_rows.extend(rows)
                    if split=='pitt-val':
                        selection,hs,gs=task.choose_thresholds(rows)
                        selection['hazard_rule']='One fixed zero-shot CLIP model/prompt set. Precision@46 is threshold-independent; select hazard threshold by validation balanced accuracy; lower threshold breaks exact ties.'
                        task.save_json(out/'selection.json',selection)
                        task.save_json(out/'validation_threshold_metrics.json',{'hazard':hs,'groups':gs})
                        print('VALIDATION_SELECTION '+json.dumps(selection),flush=True)
                    reports[split]=task.export_split(out,split,rows,selection)
            save_array(out/'image_features.npy',np.concatenate(all_features))
            # Reproduce scores from SAVED image/text embeddings, and metrics from SAVED CSVs.
            restored_features=np.load(out/'image_features.npy',allow_pickle=False)
            restored_text=np.load(out/'text_features.npy',allow_pickle=False)
            reproduced,_=score_pairs(restored_features,restored_text,scale)
            max_error=0.;offset=0
            for split in task.SPLITS:
                with (out/(split+'_probabilities.csv')).open(newline='') as f:saved=list(csv.DictReader(f))
                for r in saved:
                    for k in task.PROBABILITY_FIELDS:r[k]=float(r[k])
                if [r['sample_id'] for r in saved]!=[r['sample_id'] for r in data[split]]:raise ValueError('Saved CSV roster mismatch')
                actual=np.array([[r[k] for k in task.PROBABILITY_FIELDS] for r in saved])
                error=float(np.max(np.abs(actual-reproduced[offset:offset+len(saved)])));max_error=max(max_error,error)
                if error>2e-6:raise ValueError('Saved features do not reproduce scores')
                check=task.evaluate(saved,selection['hazard_threshold'],selection['group_thresholds'])
                if check!=reports[split]:raise ValueError('Saved predictions do not reproduce metrics')
                offset+=len(saved)
            if offset!=total:raise ValueError('Incomplete output')
            for name,h in binding['input_hashes'].items():
                if file_sha(root/name)!=h:raise ValueError('Metadata changed during run')
            records=[json.loads(line) for line in (out/'predictions.jsonl').read_text().splitlines()]
            if [r['sample_id'] for r in records]!=[r['sample_id'] for r in all_rows]:raise ValueError('Journal roster mismatch')
            for rec,row in zip(records,all_rows):
                if rec['prediction']!={f:row[f] for f in task.PROBABILITY_FIELDS}:raise ValueError('Journal scores mismatch')
            result={'model':MODEL,'revision':REVISION,'mode':'zero_shot','reference':reference,'selection':selection,
                    'validation':reports['pitt-val'],'test':reports['pitt-test'],'wall_seconds':time.monotonic()-started}
            task.save_json(out/'results.json',result)
            task.save_json(out/'final_verification.json',{'passed':True,'images':total,'split_counts':counts,
                'scores_reproduced_from_saved_embeddings':True,'maximum_score_error':max_error,
                'metrics_reproduced_from_saved_predictions':True,'all_inputs_match_vlm':reference.get('checked',False),
                'no_test_selection':True,'prompts_not_truncated':True,'journal_rows':len(records),
                'finished_at_utc':task.now(),'prediction_journal_sha256':file_sha(out/'predictions.jsonl')})
            status('complete')
            print('COMPLETE '+json.dumps({'test_precision_at_46':reports['pitt-test']['hazard']['precision_at_k'],
                'test_hazard_balanced_accuracy':reports['pitt-test']['hazard']['balanced_accuracy'],
                'test_group_macro_balanced_accuracy':reports['pitt-test']['groups_on_all_true_hazards']['balanced_accuracy']}),flush=True)
        except BaseException as exc:
            status('error',error_type=type(exc).__name__);raise


RUN_README='''# Zero-shot CLIP on sidewalk-data-3

Pinned openai/clip-vit-base-patch16 revision 57c216476eefef5ab752ec549e440a49ae4ae5f3,
matching the previous zero-shot comparison. No model training or API calls.
All prompts are fixed before inference and saved in prompts.json. For each of
five binary tasks there are two negative and two positive text descriptions.
Each prompt embedding is normalized, averaged within its polarity, and normalized
again. Softmax over that task's negative/positive cosine logits (using the saved
CLIP logit scale) gives its positive score. There is no five-class softmax and no
requirement for the four overlapping group scores to sum to one.

The tasks are hazard, surface_and_level_problems, restricted_passage,
missing_tactile_warning, and ramp_points_into_traffic. Scores are uncalibrated
text-image similarities transformed by paired softmax; group scores support the
conditional task evaluated on true hazards, not calibrated conditional
probabilities. Hazard is scored independently, not forced to the sum/max of the
group scores. Predicted hazards with no positive group are flagged.

## Inputs and evaluation

All 382 pitt-val images first, then all 380 pitt-test images. Dataset-3 tags and
explicit hazard/group columns define ground truth, including images whose old
feature class was Crosswalk. Ordinary hand-collected sidewalks remain non-hazard.
Missing-tag and duplicate-label policies are unchanged. Complete manifests,
schema, labeling/preprocessing rules, image identities and checksums are saved.
The actual run checks exact input images and targets against the new VLM run.

Use stored RGB 224x224 PNGs with CLIP's standard pixel rescaling/normalization;
resize and center crop are explicitly disabled. No label-dependent cropping.
Image paths/labels are never passed to the text encoder. Token lengths are checked
against the model limit; no prompt is silently truncated. Model and processor are
loaded only from the pinned local cache. Weight/config hashes and package versions
are recorded. Old CLIP image caches and old five-class probabilities are not reused.

The dataset contract and metric implementation are imported from
openai_baseline_data3.py, snapshotted with this runner. Importing it does not invoke
OpenAI or require an API key. No change to the running VLM experiment is needed.

Threshold grid: 0.3, 0.5, 0.7, selected exclusively on pitt-val. With one fixed
CLIP model/prompt set, precision@46 is threshold-independent, so hazard balanced
accuracy chooses its threshold. Each group's threshold maximizes binary balanced
accuracy on all true validation hazards. Lower thresholds break exact BA ties.
Selection is saved before test feature extraction; no test-based prompt tuning.

Hazard precision@46 ranks all images by p_hazard descending, then sample_id
ascending. Primary subtype metric is the macro average of four binary balanced
accuracies on ALL true hazards, regardless of predicted hazard status. Per-group
precision/recall/F1, specificity, accuracy and confusion counts are saved. Gated
pipeline metrics on all images are separate. Fixed-0.5 metrics are also saved for
reference. These definitions exactly match the updated VLM and DINO tasks.

## Outputs and restart

run_config.json, prompts.json, selection.json, validation_threshold_metrics.json,
inputs/, model_config/, and runner/evaluation snapshots record the protocol.
predictions.jsonl and pitt-{val,test}_probabilities.csv store original metadata,
ground truth and all five scores. Each split also has *_predictions.csv,
*_hazard_ranking.csv, *_top46.csv, *_metrics.json, *_per_group_metrics.csv and
*_metrics_fixed_0.5.json. status.json and run.log track execution.

prompt_features.npy, text_features.npy, image_features.npy, per-split image
features and cosine similarities allow exact score reproduction without loading
CLIP again. Array orders are in image_inventory.json and text_encoding.json.
feature_chunks/ contains atomically saved, identity-checked inference batches.
--resume checks data/model/code/environment fingerprints, reuses saved chunks,
and rebuilds predictions/metrics. A lock prevents concurrent writers.

results.json and final_verification.json are written only after all 762 images,
saved-embedding score reproduction, and saved-CSV metric reproduction pass.
The original CLIP scripts and results are preserved.
'''


if __name__=='__main__':main()
