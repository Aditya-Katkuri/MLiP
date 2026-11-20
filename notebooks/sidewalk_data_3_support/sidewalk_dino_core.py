"""Frozen DINOv3 linear-head sweep and validation-only model selection for dataset 3."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from itertools import product
import hashlib, io, json, os, shutil, tempfile, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image

SPLITS = ['other-cities-train', 'pitt-train', 'pitt-val', 'pitt-test']
GROUPS = ['surface_and_level_problems', 'restricted_passage', 'missing_tactile_warning', 'ramp_points_into_traffic']
GROUP_COLUMNS = ['group_' + g for g in GROUPS]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def plain(x):
    if isinstance(x, Path): return str(x)
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    raise TypeError(type(x).__name__)


def canonical(x): return json.dumps(x, sort_keys=True, separators=(',', ':'), default=plain)
def fingerprint(x): return hashlib.sha256(canonical(x).encode()).hexdigest()
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()


def atomic_bytes(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.tmp')
    tmp.write_bytes(payload); os.replace(tmp, path)


def save_json(path, value): atomic_bytes(path, (json.dumps(value, indent=2, default=plain, allow_nan=False)+'\n').encode())
def save_csv(path, frame): atomic_bytes(path, frame.to_csv(index=False).encode())
def save_torch(path, value):
    with tempfile.TemporaryDirectory() as temp:
        local = Path(temp)/'model.pt'; torch.save(value, local)
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name('.'+path.name+'.tmp'); shutil.copyfile(local, tmp); os.replace(tmp, path)
def save_npz(path, **arrays):
    buf = io.BytesIO(); np.savez_compressed(buf, **arrays); atomic_bytes(path, buf.getvalue())


def configs(settings):
    return [dict(plan=p, lr=float(lr), weight_decay=float(wd), seed=int(seed), batch_size=settings['batch_size'])
            for p,lr,wd,seed in product(settings['train_plans'], settings['learning_rates'], settings['weight_decays'], settings['seeds'])]

def config_id(cfg): return fingerprint({k:v for k,v in cfg.items() if k!='seed'})[:16]
def trial_id(cfg): return fingerprint(cfg)[:16]


def load_data(root, expected_hashes):
    root = Path(root)
    for filename, expected in expected_hashes.items():
        if file_sha(root/filename) != expected: raise ValueError(f'Dataset identity mismatch: {filename}')
    schema = json.loads((root/'label_schema.json').read_text())
    assert schema['group_order']==GROUPS and schema['group_columns']==GROUP_COLUMNS
    assert schema['multi_label'] is True and schema['binary_labels']=={'non_hazard':0,'hazard':1}
    data = {}
    for split in SPLITS:
        df = pd.read_csv(root/(split+'.csv'), low_memory=False, keep_default_na=False)
        assert df['sample_id'].is_unique and df['image_path'].is_unique
        assert set(df['split'])=={split}
        y = df['hazard_label'].to_numpy(dtype=np.int64)
        yg = df[GROUP_COLUMNS].to_numpy(dtype=np.int64)
        assert set(np.unique(y))=={0,1} and set(np.unique(yg))<= {0,1}
        assert np.array_equal(y, yg.any(axis=1).astype(np.int64))
        assert np.array_equal(df['class'].to_numpy(), np.where(y, 'hazard', 'non_hazard'))
        for g in GROUPS:
            tagcols = schema['groups'][g]['positive_tags']
            expected = df[tagcols].astype(str).replace('', '0').astype(float).eq(1).any(axis=1).to_numpy()
            assert np.array_equal(expected, df['group_'+g].to_numpy(dtype=bool))
        assert not df.loc[df.is_manual_collection.astype(int)==1, 'hazard_label'].astype(int).any()
        # A missing class makes balanced accuracy ill-defined for the planned task.
        assert ((yg[y==1].sum(0)>0)&(yg[y==1].sum(0)<int(y.sum()))).all()
        assert all((root/p).is_file() and (root/p).resolve().is_relative_to(root.resolve()) for p in df.image_path)
        data[split] = df
    assert len(set().union(*(set(df.sample_id) for df in data.values()))) == sum(map(len,data.values()))
    return data, schema


def targets(df):
    return df.hazard_label.to_numpy(dtype=np.float32), df[GROUP_COLUMNS].to_numpy(dtype=np.float32)


def binary_metrics(y, pred):
    y=np.asarray(y,dtype=bool); pred=np.asarray(pred,dtype=bool)
    assert y.shape==pred.shape and y.ndim==1 and y.any() and (~y).any()
    tp=int((y&pred).sum()); fn=int((y&~pred).sum()); tn=int((~y&~pred).sum()); fp=int((~y&pred).sum())
    recall=tp/(tp+fn); specificity=tn/(tn+fp)
    precision=tp/(tp+fp) if tp+fp else 0.
    f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    return dict(tp=tp,fn=fn,tn=tn,fp=fp,support=tp+fn,negative_support=tn+fp,
                precision=precision,recall=recall,specificity=specificity,f1=f1,
                accuracy=(tp+tn)/len(y),balanced_accuracy=(recall+specificity)/2)


def hazard_metrics(y, p, threshold, sample_ids, k):
    y=np.asarray(y,dtype=bool); p=np.asarray(p)
    assert len(y)>=k>0 and p.shape==y.shape and np.isfinite(p).all()
    # sample_id is the deterministic secondary key for equal confidence scores.
    order=np.lexsort((np.asarray(sample_ids,dtype=str), -p))
    hits=int(y[order[:k]].sum())
    return {**binary_metrics(y,p>=threshold),'topk_hits':hits,'precision_at_k':hits/k,'recall_at_k':hits/int(y.sum()),'k':int(k)}


def group_metrics(yh, yg, p, thresholds, hazard_only=True):
    yh=np.asarray(yh,dtype=bool); yg=np.asarray(yg,dtype=bool); p=np.asarray(p)
    assert yg.shape==p.shape and yg.shape[1]==len(GROUPS) and np.isfinite(p).all()
    mask=yh if hazard_only else np.ones(len(yh),dtype=bool)
    pred=p[mask]>=np.asarray(thresholds)
    per_group=[{'tag_group':g,**binary_metrics(yg[mask,j],pred[:,j])} for j,g in enumerate(GROUPS)]
    summary={name:float(np.mean([r[name] for r in per_group])) for name in ['balanced_accuracy','precision','recall','f1','accuracy']}
    summary['exact_match_accuracy']=float(np.mean(np.all(pred==yg[mask],axis=1)))
    summary['n_images']=int(mask.sum())
    return summary, per_group


def choose_threshold(scores):
    """scores is a list of (threshold, mean validation BA); lower threshold breaks exact ties."""
    return min(scores, key=lambda x:(-round(float(x[1]),12),float(x[0])))[0]


def select_models(entries, val_df, thresholds, seeds, k):
    """Only final-checkpoint validation probabilities enter this function."""
    yh,yg=targets(val_df); ids=val_df.sample_id.to_numpy(); bycfg=defaultdict(list)
    for entry in entries: bycfg[config_id(entry['cfg'])].append(entry)
    summary=[]; hazard_table=[]; group_table=[]; selection_details={}
    for cid, group in sorted(bycfg.items()):
        group=sorted(group,key=lambda e:e['cfg']['seed'])
        assert [e['cfg']['seed'] for e in group]==list(seeds), f'Incomplete seed group: {cid}'
        hscores=[]; per_group_scores={g:[] for g in GROUPS}
        for t in thresholds:
            h=[]; gs=[[] for _ in GROUPS]
            for e in group:
                p=e['val_probabilities']; assert p.shape==(len(val_df),5)
                m=hazard_metrics(yh,p[:,0],t,ids,k); h.append(m['balanced_accuracy'])
                hazard_table.append(dict(config_id=cid,trial_id=e['trial_id'],**e['cfg'],threshold=t,**m))
                _, gm=group_metrics(yh,yg,p[:,1:],[t]*4)
                for j,g in enumerate(GROUPS):
                    gs[j].append(gm[j]['balanced_accuracy'])
                    group_table.append(dict(config_id=cid,trial_id=e['trial_id'],**e['cfg'],threshold=t,**gm[j]))
            hscores.append((t,float(np.mean(h))))
            for j,g in enumerate(GROUPS):per_group_scores[g].append((t,float(np.mean(gs[j]))))
        ht=float(choose_threshold(hscores)); gt=[float(choose_threshold(per_group_scores[g])) for g in GROUPS]
        hm=[hazard_metrics(yh,e['val_probabilities'][:,0],ht,ids,k) for e in group]
        gm=[group_metrics(yh,yg,e['val_probabilities'][:,1:],gt)[0] for e in group]
        base={key:value for key,value in group[0]['cfg'].items() if key!='seed'}
        row=dict(config_id=cid,**base,hazard_threshold=ht,group_thresholds=json.dumps(gt),
                 topk_hits_sum=sum(m['topk_hits'] for m in hm),
                 precision_at_k_mean=float(np.mean([m['precision_at_k'] for m in hm])),
                 precision_at_k_std=float(np.std([m['precision_at_k'] for m in hm],ddof=1)),
                 hazard_ba_mean=float(np.mean([m['balanced_accuracy'] for m in hm])),
                 hazard_ba_std=float(np.std([m['balanced_accuracy'] for m in hm],ddof=1)),
                 group_macro_ba_mean=float(np.mean([m['balanced_accuracy'] for m in gm])),
                 group_macro_ba_std=float(np.std([m['balanced_accuracy'] for m in gm],ddof=1)))
        summary.append(row)
        # Configuration selection uses seed means. A deployment replica is then
        # selected from that winning configuration on validation alone.
        hrep=min(zip(group,hm),key=lambda em:(-em[1]['topk_hits'],-round(em[1]['balanced_accuracy'],12),em[0]['cfg']['seed']))[0]
        grep=min(zip(group,gm),key=lambda em:(-round(em[1]['balanced_accuracy'],12),em[0]['cfg']['seed']))[0]
        selection_details[cid]=dict(summary=row,trial_ids=[e['trial_id'] for e in group],
                                   hazard_deployment_trial=hrep['trial_id'],group_deployment_trial=grep['trial_id'])
    # Final deterministic ties prefer Pitt-only, lower LR, then lower weight decay.
    fallback=lambda r:(0 if r['plan']=='pitt' else 1,r['lr'],r['weight_decay'],r['config_id'])
    best_h=min(summary,key=lambda r:(-r['topk_hits_sum'],-round(r['hazard_ba_mean'],12),*fallback(r)))
    best_g=min(summary,key=lambda r:(-round(r['group_macro_ba_mean'],12),*fallback(r)))
    selection=dict(hazard=selection_details[best_h['config_id']],tag_groups=selection_details[best_g['config_id']],
                   k=k,group_order=GROUPS,validation_split='pitt-val',test_used_for_selection=False,
                   hazard_rule='Maximum mean validation precision@46 (integer summed top-k hits); tie-break maximum mean hazard BA at selected threshold.',
                   group_rule='Maximum mean macro group BA on ALL true validation hazards, each group threshold selected by its mean BA.',
                   seed_rule='Configuration selected across three seeds. Single deployment replica selected within winner by same validation objective; smaller seed breaks ties. All selected replicas also saved.',
                   threshold_grid=list(thresholds),checkpoint_rule='Final update only; no early-stopping/checkpoint search.')
    return selection,pd.DataFrame(summary),pd.DataFrame(hazard_table),pd.DataFrame(group_table)


def batch_indices(n, steps, batch_size, generator, device):
    assert n>=batch_size
    order=torch.randperm(n,generator=generator); offset=0
    for _ in range(steps):
        if offset+batch_size>n: order=torch.randperm(n,generator=generator);offset=0
        yield order[offset:offset+batch_size].to(device)
        offset+=batch_size


def task_losses(logits, yh, yg):
    h=nn.functional.binary_cross_entropy_with_logits(logits[:,0],yh)
    per_image=nn.functional.binary_cross_entropy_with_logits(logits[:,1:],yg,reduction='none').mean(1)
    g=(per_image*yh).sum()/yh.sum().clamp_min(1)
    return h,g


@torch.inference_mode()
def predict_head(state, features):
    head=nn.Linear(features.shape[1],5).to(features.device)
    head.load_state_dict(state);head.eval()
    return head(features).sigmoid().float().cpu().numpy()


def train_trial(cfg, settings, features, data):
    device=features['pitt-train'].device
    torch.manual_seed(cfg['seed'])
    generator=torch.Generator().manual_seed(cfg['seed'])
    head=nn.Linear(features['pitt-train'].shape[1],5).to(device)
    optimizer=torch.optim.Adam(head.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    valx=features['pitt-val']; vyh,vyg=targets(data['pitt-val'])
    val_targets=[torch.tensor(x,device=device) for x in [vyh,vyg]]
    step=0; history=[];window=[]
    total=sum(s for _,s in settings['train_plans'][cfg['plan']])
    for split,updates in settings['train_plans'][cfg['plan']]:
        x=features[split]; yh,yg=[torch.tensor(t,device=device) for t in targets(data[split])]
        for idx in batch_indices(len(x),updates,cfg['batch_size'],generator,device):
            h,g=task_losses(head(x[idx]),yh[idx],yg[idx]);loss=h+g
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            window.append(torch.stack([h.detach(),g.detach()]));step+=1
            if step%settings['eval_every']==0 or step==total:
                with torch.inference_mode():
                    vh,vg=task_losses(head(valx),*val_targets)
                    probs=head(valx).sigmoid().float().cpu().numpy()
                train_loss=torch.stack(window).mean(0).cpu().numpy();window=[]
                hm=hazard_metrics(vyh,probs[:,0],.5,data['pitt-val'].sample_id.to_numpy(),settings['k'])
                gm,_=group_metrics(vyh,vyg,probs[:,1:],[.5]*4)
                history.append(dict(step=step,phase=split,hazard_train_loss=float(train_loss[0]),group_train_loss=float(train_loss[1]),hazard_val_loss=float(vh),group_val_loss=float(vg),
                    hazard_precision_at_k=hm['precision_at_k'],hazard_ba_at_05=hm['balanced_accuracy'],group_macro_ba_at_05=gm['balanced_accuracy']))
    state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
    return state,predict_head(state,valx),pd.DataFrame(history)


def run_sweep(settings, binding, features, data, run_dir):
    entries=[];histories=[]; all_cfg=configs(settings)
    for i,cfg in enumerate(all_cfg):
        tid=trial_id(cfg);folder=Path(run_dir)/'trials'/tid;folder.mkdir(parents=True,exist_ok=True)
        done=folder/'complete.json';start=time.perf_counter()
        if done.is_file():
            completion=json.loads(done.read_text()); assert completion['binding']==binding and completion['cfg']==cfg
            for name,sha in completion['file_hashes'].items():assert file_sha(folder/name)==sha,f'Damaged trial {tid}: {name}'
            payload=torch.load(folder/'head.pt',map_location='cpu',weights_only=True)
            assert payload['cfg']==cfg and payload['binding']==binding
            state=payload['state_dict'];p=np.load(folder/'val_probabilities.npz')['probabilities'];history=pd.read_csv(folder/'history.csv');action='reused'
        else:
            state,p,history=train_trial(cfg,settings,features,data)
            save_torch(folder/'head.pt',dict(cfg=cfg,binding=binding,state_dict=state,final_step=sum(n for _,n in settings['train_plans'][cfg['plan']]),group_order=GROUPS))
            save_npz(folder/'val_probabilities.npz',probabilities=p,sample_ids=data['pitt-val'].sample_id.to_numpy(dtype=str))
            save_csv(folder/'history.csv',history)
            save_json(done,dict(cfg=cfg,binding=binding,file_hashes={name:file_sha(folder/name) for name in ['head.pt','val_probabilities.npz','history.csv']}));action='trained'
        cached_ids=np.load(folder/'val_probabilities.npz')['sample_ids']
        assert np.array_equal(cached_ids,data['pitt-val'].sample_id.to_numpy(dtype=str))
        entries.append(dict(cfg=cfg,trial_id=tid,state_dict=state,val_probabilities=p))
        histories.append(history.assign(trial_id=tid,**cfg))
        save_json(Path(run_dir)/'status.json',dict(status='sweeping',trials_completed=len(entries),trials_total=len(all_cfg),last_trial=tid))
        print(f'[{i+1}/{len(all_cfg)}] {action}: {cfg} ({time.perf_counter()-start:.1f}s)',flush=True)
    history=pd.concat(histories,ignore_index=True);save_csv(Path(run_dir)/'training_history.csv',history)
    return entries,history


def cache_features(encoder, df, split, root, run_dir, binding, batch_size, device):
    cache=Path(run_dir)/'features'/(split+'.pt')
    cache_binding={**binding,'split':split,'sample_ids_sha256':fingerprint(df.sample_id.tolist())}
    if cache.exists():
        cached=torch.load(cache,map_location='cpu',weights_only=True)
        assert cached['binding']==cache_binding,'Feature cache identity mismatch; start a new run.'
        x=cached['features']; assert x.shape==(len(df),encoder.config.hidden_size) and torch.isfinite(x).all()
        print('Reused cached features:',split)
        return x.to(device).clone()
    root=Path(root);mean=torch.tensor(MEAN,device=device).view(1,3,1,1);std=torch.tensor(STD,device=device).view(1,3,1,1)
    def read(row):
        payload=(root/row.image_path).read_bytes()
        assert hashlib.sha256(payload).hexdigest()==row.output_file_sha256,f'Image checksum mismatch: {row.image_path}'
        with Image.open(io.BytesIO(payload)) as im:
            im.load();assert im.size==(224,224) and im.mode=='RGB' and im.format=='PNG'
            return np.array(im,copy=True)
    pieces=[];start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
        for begin in range(0,len(df),batch_size):
            pixels=np.stack(list(pool.map(read,df.iloc[begin:begin+batch_size].itertuples(index=False))))
            x=torch.from_numpy(pixels).permute(0,3,1,2).to(device).float()/255
            x=(x-mean)/std
            with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=str(device).startswith('cuda')):
                feats=encoder(pixel_values=x).last_hidden_state[:,0].float()
            assert torch.isfinite(feats).all(),'Nonfinite backbone features; stop rather than saving a bad cache.'
            pieces.append(feats.cpu())
            if begin%(batch_size*20)==0:print(f'{split}: {min(begin+batch_size,len(df))}/{len(df)} embedded',flush=True)
    # clone outside inference_mode so head autograd can save its inputs.
    output=torch.cat(pieces).clone()
    save_torch(cache,dict(binding=cache_binding,features=output))
    print(f'Cached {split} in {time.perf_counter()-start:.1f}s')
    return output.to(device).clone()


def export_selection(selection, entries, schema, binding, run_dir, val_features):
    root=Path(run_dir)/'inference';root.mkdir(parents=True,exist_ok=True)
    byid={e['trial_id']:e for e in entries}
    save_json(Path(run_dir)/'selection.json',selection)
    save_json(root/'selection.json',selection);save_json(root/'label_schema.json',schema)
    for task,key in [('hazard','hazard_deployment_trial'),('tag_groups','group_deployment_trial')]:
        chosen=selection[task]
        for tid in chosen['trial_ids']:
            e=byid[tid];state=e['state_dict'];indices=slice(0,1) if task=='hazard' else slice(1,5)
            sliced={name:tensor[indices].clone() for name,tensor in state.items()}
            threshold=chosen['summary']['hazard_threshold'] if task=='hazard' else json.loads(chosen['summary']['group_thresholds'])
            payload=dict(format_version=1,task=task,trial_id=tid,cfg=e['cfg'],binding=binding,state_dict=sliced,thresholds=threshold,
                         output_labels=['hazard'] if task=='hazard' else GROUPS,feature_dim=state['weight'].shape[1],pooling='cls')
            file=root/'replicas'/task/(f'seed-{e["cfg"]["seed"]}.pt');save_torch(file,payload)
            if tid==chosen[key]:save_torch(root/('best_'+task+'.pt'),payload)
    # Reload the deployed heads and verify against the original validation scores.
    checks={}
    for task,key,col in [('hazard','hazard_deployment_trial',slice(0,1)),('tag_groups','group_deployment_trial',slice(1,5))]:
        stored=torch.load(root/('best_'+task+'.pt'),map_location='cpu',weights_only=True)
        e=byid[selection[task][key]];head=nn.Linear(stored['feature_dim'],len(stored['output_labels'])).to(val_features.device)
        head.load_state_dict(stored['state_dict'])
        with torch.inference_mode():p=head(val_features).sigmoid().cpu().numpy()
        original=e['val_probabilities'][:,col]
        error=float(np.max(np.abs(p-original)));assert error<2e-6,f'Checkpoint reload error: {task} {error}'
        checks[task]=dict(passed=True,max_probability_difference=error,trial_id=stored['trial_id'])
    save_json(root/'checkpoint_reload_verification.json',checks)
    return root


def prediction_frame(df, hp, gp, ht, gt, **context):
    cols=['sample_id','image_path','split','class','hazard_label','original_class','source_image_path','city','is_manual_collection']+GROUP_COLUMNS
    out=df[cols].copy();out['p_hazard']=hp;out['pred_hazard']=(hp>=ht).astype(int)
    for j,g in enumerate(GROUPS):
        out['p_'+g]=gp[:,j];out['pred_'+g]=(gp[:,j]>=gt[j]).astype(int)
        out['pipeline_pred_'+g]=out['pred_'+g]*out['pred_hazard']
    out['hazard_without_group']=((out.pred_hazard==1)&~(gp>=np.array(gt)).any(1)).astype(int)
    for name,value in context.items():out[name]=value
    return out


def evaluate_selected(selection, entries, features, data, run_dir, split):
    df=data[split];yh,yg=targets(df);byid={e['trial_id']:e for e in entries}
    rows=[];per_group=[];predictions=[];scores={}
    for task in ['hazard','tag_groups']:
        chosen=selection[task];ht=chosen['summary']['hazard_threshold'];gt=json.loads(chosen['summary']['group_thresholds'])
        deployment=chosen['hazard_deployment_trial' if task=='hazard' else 'group_deployment_trial']
        for tid in chosen['trial_ids']:
            e=byid[tid];p=predict_head(e['state_dict'],features[split]);scores[tid]=p
            context=dict(task=task,trial_id=tid,seed=e['cfg']['seed'],config_id=config_id(e['cfg']),deployment_replica=(tid==deployment),split=split)
            if task=='hazard':m=hazard_metrics(yh,p[:,0],ht,df.sample_id.to_numpy(),selection['k'])
            else:
                m,groups=group_metrics(yh,yg,p[:,1:],gt)
                per_group.extend([{**context,**item} for item in groups])
            rows.append({**context,**m})
            predictions.append(prediction_frame(df,p[:,0],p[:,1:],ht,gt,**{k:v for k,v in context.items() if k!='split'}))
    metrics=pd.DataFrame(rows);save_csv(Path(run_dir)/(split+'_selected_seed_metrics.csv'),metrics)
    save_csv(Path(run_dir)/(split+'_per_group_metrics.csv'),pd.DataFrame(per_group))
    save_csv(Path(run_dir)/(split+'_selected_predictions.csv'),pd.concat(predictions,ignore_index=True))
    columns=[c for c in ['precision_at_k','recall_at_k','balanced_accuracy','precision','recall','f1','exact_match_accuracy'] if c in metrics]
    summary=metrics.groupby('task')[columns].agg(['mean','std']);atomic_bytes(Path(run_dir)/(split+'_selected_seed_summary.csv'),summary.to_csv().encode())
    hid=selection['hazard']['hazard_deployment_trial'];gid=selection['tag_groups']['group_deployment_trial']
    hp=scores[hid][:,0];gp=scores[gid][:,1:];ht=selection['hazard']['summary']['hazard_threshold'];gt=json.loads(selection['tag_groups']['summary']['group_thresholds'])
    deployed=prediction_frame(df,hp,gp,ht,gt,hazard_trial=hid,group_trial=gid)
    save_csv(Path(run_dir)/(split+'_deployment_predictions.csv'),deployed)
    gm,groups=group_metrics(yh,yg,gp,gt)
    gated=(gp>=np.array(gt)) & (hp>=ht)[:,None]
    pm,p_groups=group_metrics(yh,yg,gated.astype(float),[.5]*4,hazard_only=False)
    report={'hazard':hazard_metrics(yh,hp,ht,df.sample_id.to_numpy(),selection['k']),
            'groups_on_all_true_hazards':gm,'group_metrics_on_all_true_hazards':groups,
            'gated_pipeline_groups_on_all_images':pm,'pipeline_group_metrics':p_groups,
            'hazard_trial':hid,'group_trial':gid,'hazard_threshold':ht,'group_thresholds':gt,
            'hazard_without_group_count':int(deployed.hazard_without_group.sum()),'test_used_for_selection':False}
    save_json(Path(run_dir)/(split+'_deployment_metrics.json'),report)
    return metrics,report


def plot_curves(history, selection, run_dir):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,2,figsize=(13,8))
    for col,(task,prefix) in enumerate([('hazard','hazard'),('tag_groups','group')]):
        ids=selection[task]['trial_ids'];subset=history[history.trial_id.isin(ids)]
        for seed,frame in subset.groupby('seed'):
            axs[0,col].plot(frame.step,frame[prefix+'_train_loss'],alpha=.55,label=f'Train seed {seed}')
            axs[0,col].plot(frame.step,frame[prefix+'_val_loss'],ls='--',alpha=.55,label=f'Val seed {seed}')
            name='hazard_precision_at_k' if task=='hazard' else 'group_macro_ba_at_05'
            axs[1,col].plot(frame.step,frame[name],label=f'Seed {seed}')
        axs[0,col].set_title(task+' — selected configuration');axs[0,col].set_ylabel('BCE loss');axs[0,col].legend(fontsize=8)
        axs[1,col].set_ylabel('Validation precision@46' if task=='hazard' else 'Validation macro group BA (thresholds 0.5)')
        for row in range(2):axs[row,col].set_xlabel('Optimizer updates');axs[row,col].grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(Path(run_dir)/('selected_learning_curves.'+ext),dpi=160)
    plt.show();plt.close(fig)
