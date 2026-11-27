"""Verify 128-head raw prediction export, data separation and checkpoint resumption."""
from pathlib import Path
from types import SimpleNamespace
import ast,importlib.util,json,sys
import numpy as np
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'notebooks/sidewalk_data_3_support/hazard_bootstrap_core.py'
spec=importlib.util.spec_from_file_location('hazard_bootstrap_core',MODULE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)


def frame(n=48,split='pitt-train'):
    return pd.DataFrame(dict(sample_id=[f'{split}-{i:03d}' for i in range(n)],
        image_path=[f'{split}/{i}.png' for i in range(n)],split=[split]*n,
        hazard_label=np.arange(n)%2,output_rgb_pixel_sha256=[f'{i:064x}' for i in range(n)]))


def test_all_128_bootstrap_samples_preserve_classes_and_duplicate_groups():
    df=frame()
    df.loc[1,'output_rgb_pixel_sha256']=df.loc[0,'output_rgb_pixel_sha256']
    df.loc[3,'output_rgb_pixel_sha256']=df.loc[2,'output_rgb_pixel_sha256']
    df.loc[6,'output_rgb_pixel_sha256']=df.loc[4,'output_rgb_pixel_sha256']
    fingerprints=set()
    for member in range(1,129):
        draws,counts=core.bootstrap_sample(df,1000+member)
        assert len(draws)==len(df)
        assert counts[0]==counts[1] and counts[2]==counts[3] and counts[4]==counts[6]
        np.testing.assert_array_equal(np.bincount(df.hazard_label.iloc[draws]),np.bincount(df.hazard_label))
        np.testing.assert_array_equal(counts,np.bincount(draws,minlength=len(df)))
        np.testing.assert_array_equal(draws,core.bootstrap_sample(df,1000+member)[0])
        fingerprints.add(core.fingerprint(draws))
    assert len(fingerprints)==128


def test_bootstrap_and_training_reject_nontraining_rows():
    df=frame(split='pitt-val')
    with pytest.raises(AssertionError):core.bootstrap_sample(df,1001)
    with pytest.raises(AssertionError):core.train_member({},torch.zeros(48,8),df,np.arange(48),{})


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    torch.set_num_threads(1)
    root=tmp_path_factory.mktemp('raw128')
    df=frame()
    x=torch.tensor(np.random.default_rng(77).normal(size=(len(df),8)),dtype=torch.float32)
    settings=dict(n_members=128,lr=.03,weight_decay=.01,batch_size=8,updates=2,log_every=1,
                  bootstrap_seed_base=1000,init_seed_base=2000,batch_seed_base=3000)
    entries=core.run_ensemble(settings,{'test':'fixture'},x,df,root)
    return root,df,x,settings,entries


def test_128_heads_are_saved_reloaded_and_resumed_without_retraining(trained,monkeypatch):
    root,df,x,settings,entries=trained
    assert len(entries)==128 and len(list((root/'members').glob('*/head.pt')))==128
    assert all(e['state_dict']['weight'].shape==(1,8) for e in entries)
    inventory=pd.read_csv(root/'bootstrap_members.csv')
    assert inventory.bootstrap_seed.tolist()==list(range(1001,1129))
    assert inventory.init_seed.tolist()==list(range(2001,2129))
    assert inventory.batch_seed.tolist()==list(range(3001,3129))
    for e in entries:
        saved=torch.load(root/'members'/f"head_{e['cfg']['member']:03d}"/'head.pt',weights_only=True)
        assert all(torch.equal(e['state_dict'][k],saved['state_dict'][k]) for k in e['state_dict'])
    def forbidden(*args,**kwargs):raise AssertionError('Retrained a completed head')
    monkeypatch.setattr(core,'train_member',forbidden)
    again=core.run_ensemble(settings,{'test':'fixture'},x,df,root)
    np.testing.assert_array_equal(core.predict_ensemble(entries,x),core.predict_ensemble(again,x))
    with pytest.raises(AssertionError,match='Resume identity'):
        core.run_ensemble({**settings,'lr':.2},{'test':'fixture'},x,df,root)


def test_interrupted_member_only_is_restarted(trained,monkeypatch):
    root,df,x,settings,entries=trained
    (root/'members/head_128/complete.json').unlink()
    before=core.file_sha(root/'members/head_001/head.pt')
    real=core.train_member;called=[]
    def record(cfg,*args,**kwargs):
        called.append(cfg['member']);return real(cfg,*args,**kwargs)
    monkeypatch.setattr(core,'train_member',record)
    again=core.run_ensemble(settings,{'test':'fixture'},x,df,root)
    assert called==[128] and core.file_sha(root/'members/head_001/head.pt')==before
    np.testing.assert_array_equal(core.predict_ensemble(entries,x),core.predict_ensemble(again,x))


def test_raw_probability_export_preserves_manifest_order_and_has_no_analysis(trained,tmp_path):
    root,train,x,settings,entries=trained
    cols=core.probability_columns(np.arange(1,129))
    assert cols[0]=='p_hazard_head_001' and cols[-1]=='p_hazard_head_128'
    for split in ['pitt-train','pitt-val','pitt-test']:
        df=frame(48,split).iloc[::-1].reset_index(drop=True)
        out=core.export_predictions(entries,x,df,tmp_path,split)
        assert out.sample_id.tolist()==df.sample_id.tolist()
        assert [c for c in out if c.startswith('p_hazard_')]==cols
        assert set(out.columns)==set(df.columns)|set(cols)
        assert out[cols].shape==(48,128)
        p=np.load(tmp_path/(split+'_probabilities.npz'))
        np.testing.assert_array_equal(p['member_ids'],np.arange(1,129))
        np.testing.assert_array_equal(p['sample_ids'],df.sample_id)
        np.testing.assert_array_equal(p['probabilities'],out[cols].to_numpy())
        # Check representative ensemble columns against separate single-head forwards.
        for index in [0,63,127]:
            np.testing.assert_allclose(p['probabilities'][:,index],core.predict_member(entries[index]['state_dict'],x),atol=2e-6,rtol=0)
    assert sorted(p.name for p in tmp_path.iterdir())==sorted(s+suffix for s in ['pitt-train','pitt-val','pitt-test'] for suffix in ['_predictions.csv','_probabilities.npz'])
    oob=np.load(root/'pitt-train_bootstrap_counts.npz')
    np.testing.assert_array_equal(oob['sample_ids'],train.sample_id)
    np.testing.assert_array_equal(oob['member_ids'],np.arange(1,129))
    np.testing.assert_array_equal(oob['counts'],np.column_stack([e['counts'] for e in entries]))


def test_exported_weights_are_idempotent_and_match_predictions(trained,tmp_path):
    root,df,x,settings,entries=trained
    bundle=core.export_ensemble(entries,{'test':'fixture'},x[:8],tmp_path)
    digest=core.file_sha(bundle/'ensemble.pt')
    core.export_ensemble(entries,{'test':'fixture'},x[:8],tmp_path)
    assert core.file_sha(bundle/'ensemble.pt')==digest
    saved=torch.load(bundle/'ensemble.pt',weights_only=True)
    assert saved['n_members']==128 and saved['weight'].shape==(128,8)
    assert saved['member_ids']==list(range(1,129))
    expected=(x@saved['weight'].T+saved['bias']).sigmoid().numpy()
    np.testing.assert_array_equal(expected,core.predict_ensemble(entries,x))


def test_offline_inference_emits_128_raw_probabilities(tmp_path,monkeypatch):
    from PIL import Image
    inspec=importlib.util.spec_from_file_location('hazard_bootstrap_inference',MODULE.parent/'hazard_bootstrap_inference.py')
    inference=importlib.util.module_from_spec(inspec);inspec.loader.exec_module(inference)
    class ToyEncoder(torch.nn.Module):
        def forward(self,pixel_values):return SimpleNamespace(last_hidden_state=pixel_values.mean((2,3))[:,None,:])
    class AutoModel:
        @staticmethod
        def from_pretrained(path,local_files_only=False):
            assert local_files_only;return ToyEncoder()
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoModel=AutoModel))
    weights=torch.arange(128*3,dtype=torch.float32).reshape(128,3)/384
    bias=torch.linspace(-.5,.5,128)
    columns=core.probability_columns(range(1,129))
    core.save_torch(tmp_path/'ensemble.pt',dict(format_version=2,task='hazard',n_members=128,
        probability_columns=columns,weight=weights,bias=bias))
    core.save_json(tmp_path/'preprocessing.json',dict(size=[224,224],mean=core.MEAN,std=core.STD))
    path=tmp_path/'image.png';color=(128,64,255);Image.new('RGB',(280,160),color).save(path)
    rows=inference.HazardEnsemble(tmp_path,'cpu').predict([path])
    assert len(rows)==1 and list(rows[0])==['image_path']+columns
    x=(torch.tensor(color,dtype=torch.float32)/255-torch.tensor(core.MEAN))/torch.tensor(core.STD)
    expected=(x@weights.T+bias).sigmoid().numpy()
    np.testing.assert_allclose([rows[0][c] for c in columns],expected,atol=2e-6,rtol=0)


def test_notebook_embeds_selected_source_and_only_training_and_raw_export():
    import nbformat
    nb=json.loads((ROOT/'notebooks/dino_hazard_bootstrap.ipynb').read_text())
    namespace={};markdown='';code=[]
    for i,c in enumerate(nb['cells']):
        text=''.join(c['source'])
        if c['cell_type']=='markdown':markdown+=text;continue
        assert c['execution_count'] is None and not c['outputs']
        compile(''.join(line for line in text.splitlines(True) if not line.lstrip().startswith('%')),f'cell-{i}','exec')
        code.append(text)
        if any(line.startswith('DATA_SOURCE = ') for line in text.splitlines()):exec(text,namespace)
        if text.startswith('CORE_SOURCE ='):
            tree=ast.parse(text)
            for node in tree.body[:5]:
                namespace[node.targets[0].id]=ast.literal_eval(node.value)
    assert namespace['CORE_SOURCE']==MODULE.read_text()
    assert namespace['INFERENCE_SOURCE']==(MODULE.parent/'hazard_bootstrap_inference.py').read_text()
    assert namespace['IO_SOURCE']==(MODULE.parent/'hazard_bootstrap_io.py').read_text()
    assert namespace['SETTINGS']['n_members']==128 and namespace['SETTINGS']['updates']==5000
    assert namespace['SETTINGS']['lr']==3e-5 and namespace['SETTINGS']['weight_decay']==.01
    assert namespace['PREDICTION_SPLITS']==['pitt-train','pitt-val','pitt-test']
    selected=namespace['SOURCE_RECORDS']['selection.json']['hazard']['summary']
    assert selected['config_id']=='68caf0783493ae93' and selected['plan']=='pitt'
    for key in ['lr','weight_decay','batch_size']:assert namespace['SETTINGS'][key]==selected[key]
    assert namespace['SOURCE_RECORDS']['config.json']['settings']['train_plans']['pitt']==[['pitt-train',5000]]
    assert 'selection.json' in markdown and 'config.json' in markdown and '68caf0783493ae93' in markdown
    allcode='\n'.join(code)
    for forbidden in ['RANKING_LAMBDA','VALIDATION_LAMBDAS','ranking_sweep','plot_diagnostics','probability_metrics',
                      'ensemble_statistics','p_hazard_mean','p_hazard_std','pred_hazard','matplotlib','drive.mount(',
                      'authenticate_user(','files.download(']:
        assert forbidden not in allcode
    assert 'use_cookies=False' in allcode
    assert not hasattr(core,'ranking_metrics') and not hasattr(core,'dataset_audit')
    nbformat.validate(nbformat.from_dict(nb))
