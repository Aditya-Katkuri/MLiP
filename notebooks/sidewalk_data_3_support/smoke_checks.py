"""Small synthetic tests; does not train on sidewalk images or run the actual sweep."""
import ast,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np,pandas as pd,torch
from PIL import Image
import sidewalk_dino_core as c
import sidewalk_inference as inference

torch.set_num_threads(1)

def frame(n=12):
    y=np.zeros(n,dtype=np.float32);y[:n//2]=1
    g=np.zeros((n,4),dtype=np.float32)
    for i in range(n//2):g[i,i%4]=1
    g[0,1]=1;g[1,2]=1
    df=pd.DataFrame(dict(sample_id=[f'id-{i:04}' for i in range(n)],image_path=[f'p{i}.png' for i in range(n)],
                        split=['pitt-val']*n,**{'class':np.where(y,'hazard','non_hazard')},hazard_label=y,
                        original_class=['source']*n,source_image_path=[f'raw{i}' for i in range(n)],city=['pittsburgh']*n,is_manual_collection=[0]*n))
    for j,col in enumerate(c.GROUP_COLUMNS):df[col]=g[:,j]
    return df

class Checks(unittest.TestCase):
    def test_binary_formula_and_topk_ties(self):
        m=c.binary_metrics([1,1,0,0],[1,0,0,0]);self.assertEqual(m['balanced_accuracy'],.75)
        self.assertEqual(m['precision'],1.);self.assertEqual(m['recall'],.5)
        m=c.hazard_metrics([1,0,1,0],[.5]*4,.5,['d','a','c','b'],2);self.assertEqual(m['topk_hits'],0)
        m2=c.hazard_metrics([0,1,0,1],[.5]*4,.5,['b','c','a','d'],2);self.assertEqual(m,m2)
        with self.assertRaises(AssertionError):c.hazard_metrics([1,0],[.7,.3],.5,['a','b'],46)

    def test_multilabel_metric_population(self):
        df=frame();y,g=c.targets(df);p=g*.8+.1
        p[y==0]=1 # These intentionally wrong null predictions must not affect subtype selection.
        m,_=c.group_metrics(y,g,p,[.5]*4);self.assertEqual(m['balanced_accuracy'],1.)
        self.assertEqual(m['exact_match_accuracy'],1.);self.assertEqual(m['n_images'],6)
        gated,_=c.group_metrics(y,g,p,[.5]*4,hazard_only=False);self.assertLess(gated['balanced_accuracy'],1.)
        all_negative,_=c.group_metrics(y,g,np.zeros_like(p),[.5]*4);self.assertEqual(all_negative['balanced_accuracy'],.5)

    def test_loss_masks_nonhazards(self):
        logits=torch.zeros(4,5,requires_grad=True);y=torch.tensor([0.,0.,1.,1.]);g=torch.ones(4,4)
        h,s=c.task_losses(logits,y,g);(h+s).backward()
        self.assertTrue(torch.equal(logits.grad[:2,1:],torch.zeros(2,4)))
        self.assertTrue((logits.grad[:,0]!=0).all())
        logits=torch.randn(4,5,requires_grad=True);h,s=c.task_losses(logits,torch.zeros(4),g)
        self.assertEqual(float(s.detach()),0.);(h+s).backward();self.assertTrue(torch.isfinite(logits.grad).all())

    def test_independent_winners_precision_tie_and_seed_rules(self):
        df=frame();y,g=c.targets(df);entries=[]
        scores=[np.array([.9,.8,.55,.54,.53,.52,.7,.6,.51,.49,.48,.47]),
                np.array([.9,.8,.7,.65,.6,.55,.25,.2,.15,.1,.05,.01]),
                np.array([.89,.88,.7,.6,.55,.54,.95,.2,.15,.1,.05,.01])]
        for i in range(3):
            for seed in [0,1,2]:
                cfg=dict(plan='other_then_pitt' if i==2 else 'pitt',lr=[1e-5,3e-5,1e-4][i],weight_decay=.01 if i==1 else 0.,seed=seed,batch_size=4)
                gp=g*.8+.1 if i==2 else np.full_like(g,.2)
                gp[y==0]=1.
                entries.append(dict(cfg=cfg,trial_id=c.trial_id(cfg),val_probabilities=np.column_stack([scores[i],gp])))
        selected,summary,ht,gt=c.select_models(entries,df,[.3,.5,.7],[0,1,2],2)
        self.assertEqual(selected['hazard']['summary']['lr'],3e-5)
        self.assertEqual(selected['tag_groups']['summary']['lr'],1e-4)
        self.assertEqual(selected['hazard']['summary']['hazard_threshold'],.3)
        self.assertEqual(selected['hazard']['summary']['topk_hits_sum'],6)
        self.assertEqual(json.loads(selected['tag_groups']['summary']['group_thresholds']),[.3]*4)
        self.assertEqual(len(ht),27);self.assertEqual(len(gt),108)
        byid={e['trial_id']:e for e in entries}
        self.assertEqual(byid[selected['hazard']['hazard_deployment_trial']]['cfg']['seed'],0)
        self.assertEqual(byid[selected['tag_groups']['group_deployment_trial']]['cfg']['seed'],0)
        self.assertFalse(selected['test_used_for_selection'])
        with self.assertRaises(AssertionError):c.select_models(entries[:-1],df,[.3,.5,.7],[0,1,2],2)

    def test_small_training_resume_export_and_offline_inference(self):
        settings=dict(train_plans={'pitt':[('pitt-train',4)],'other_then_pitt':[('other-cities-train',4),('pitt-train',4)]},
                      learning_rates=[.01],weight_decays=[0.],seeds=[0,1,2],batch_size=4,eval_every=2,thresholds=[.3,.5,.7],k=2)
        data={s:frame(32 if s.endswith('train') else 12).assign(split=s) for s in c.SPLITS}
        torch.manual_seed(123);features={s:torch.randn(len(df),8) for s,df in data.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);binding={'test':'synthetic'}
            entries,history=c.run_sweep(settings,binding,features,data,root)
            self.assertEqual(len(entries),6);self.assertEqual(len(history),18)
            with patch.object(c,'train_trial',side_effect=RuntimeError('Completed trials must not retrain')):
                again,_=c.run_sweep(settings,binding,features,data,root)
            for a,b in zip(entries,again):np.testing.assert_array_equal(a['val_probabilities'],b['val_probabilities'])
            selected,_,_,_=c.select_models(entries,data['pitt-val'],settings['thresholds'],settings['seeds'],settings['k'])
            schema={'group_order':c.GROUPS};bundle=c.export_selection(selected,entries,schema,binding,root,features['pitt-val'])
            heads,meta=inference.load_heads(bundle,'cpu')
            self.assertEqual(len(meta['tag_groups']['output_labels']),4)
            self.assertEqual(heads['hazard'].weight.shape,(1,8))
            self.assertTrue((bundle/'replicas/tag_groups/seed-2.pt').exists())
            metrics,report=c.evaluate_selected(selected,entries,features,data,root,'pitt-test')
            self.assertEqual(len(metrics),6);self.assertEqual(report['groups_on_all_true_hazards']['n_images'],6)
            class FakeEncoder(torch.nn.Module):
                def forward(self,pixel_values):
                    means=pixel_values.mean((2,3));features=torch.cat([means,means,means[:,:2]],1)
                    return SimpleNamespace(last_hidden_state=features[:,None,:])
            (bundle/'backbone').mkdir();c.save_json(bundle/'preprocessing.json',dict(mean=c.MEAN,std=c.STD,size=[224,224]))
            image=root/'test_rgba.png';Image.new('RGBA',(313,187),(90,120,150,255)).save(image)
            with patch('transformers.AutoModel.from_pretrained',return_value=FakeEncoder()):
                model=inference.SidewalkPredictor(bundle,'cpu');prediction=model.predict([image])[0]
            self.assertEqual(len([k for k in prediction if k.startswith('p_')]),5)
            self.assertIn(prediction['predicted_class'],['hazard','non_hazard'])
            if not prediction['pred_hazard']:self.assertEqual(json.loads(prediction['predicted_tag_groups']),[])
            # A changed binding is rejected instead of silently mixing prior results.
            with self.assertRaises(AssertionError):c.run_sweep(settings,{'test':'changed'},features,data,root)

    def test_notebook_compiles_and_contains_current_sources(self):
        path=Path(__file__).with_name('dino_sidewalk_data_3.ipynb')
        if not path.exists():path=Path(__file__).parent.parent/'dino_sidewalk_data_3.ipynb'
        nb=json.loads(path.read_text())
        for i,cell in enumerate(nb['cells']):
            if cell['cell_type']=='code':
                self.assertIsNone(cell['execution_count']);self.assertEqual(cell['outputs'],[])
                ast.parse('\n'.join(x for x in cell['source'].splitlines() if not x.startswith('%')),filename=f'cell-{i}')
        helper=next(cell for cell in nb['cells'] if cell['cell_type']=='code' and cell['source'].startswith('CORE_SOURCE ='))
        parsed=ast.parse(helper['source'])
        assignments={x.targets[0].id:ast.literal_eval(x.value) for x in parsed.body if isinstance(x,ast.Assign) and isinstance(x.value,ast.Constant)}
        self.assertEqual(assignments['CORE_SOURCE'],Path(c.__file__).read_text())
        self.assertEqual(assignments['INFERENCE_SOURCE'],Path(inference.__file__).read_text())
        source='\n'.join(x['source'] for x in nb['cells'])
        self.assertNotIn('hf_VC',source);self.assertNotIn('REPORTED_CHOICES',source)
        self.assertIn('precision@46',source);self.assertIn('best_tag_groups.pt',source)

if __name__=='__main__':unittest.main(verbosity=2)
