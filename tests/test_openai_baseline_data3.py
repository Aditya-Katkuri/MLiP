"""Offline contract, metric and interruption tests; never calls the real API."""
import base64
import contextlib
import csv
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'openai_baseline_data3.py'
if not SOURCE.exists(): SOURCE=HERE.parent/'src/baseline/openai_baseline_data3.py'
spec=importlib.util.spec_from_file_location('openai_data3',SOURCE)
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        candidate=HERE/'data_reference/label_schema.json'
        if not candidate.exists(): candidate=Path('/data/datasets/sidewalk-data-3/label_schema.json')
        cls.schema_bytes=candidate.read_bytes();cls.schema=json.loads(cls.schema_bytes)

    def test_overlapping_scores_are_not_softmax(self):
        p={f:.8 for f in b.PROBABILITY_FIELDS}
        self.assertEqual(b.validate_prediction(p),p)
        for invalid in [float('nan'),float('inf'),1.01,-.01,True,'0.4',None]:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                b.validate_prediction({**p,'p_hazard':invalid})
        with self.assertRaises(ValueError): b.validate_prediction({'p_Obstacle':.5,'p_SurfaceProblem':.5})
        with self.assertRaises(ValueError): b.validate_prediction({**p,'label':'hazard'})

    def test_request_contains_only_fixed_prompt_and_image(self):
        observed={}
        def create(**kw):observed.update(kw);return 'response'
        client=types.SimpleNamespace(responses=types.SimpleNamespace(create=create))
        payload=b'PNG bytes for unit test'
        self.assertEqual(b.request_prediction(client,b.DEFAULT_MODEL,payload,'fixed prompt','high'),'response')
        content=observed['input'][0]['content']
        self.assertEqual(content,[{'type':'input_text','text':'fixed prompt'},
            {'type':'input_image','image_url':'data:image/png;base64,'+base64.b64encode(payload).decode(),'detail':'high'}])
        self.assertFalse(observed['store']);self.assertEqual(observed['reasoning'],{'effort':'low'})
        self.assertEqual(observed['text']['format']['schema'],b.OUTPUT_SCHEMA)

    def test_fixed_k_and_sample_id_tie_break(self):
        rows=[{'sample_id':'d','hazard_label':'1','p_hazard':.8},
              {'sample_id':'b','hazard_label':'0','p_hazard':.8},
              {'sample_id':'c','hazard_label':'1','p_hazard':.8},
              {'sample_id':'a','hazard_label':'1','p_hazard':.9}]
        low=b.hazard_metrics(rows,.3,k=2); high=b.hazard_metrics(rows,.95,k=2)
        self.assertEqual(low['precision_at_k'],.5);self.assertEqual(high['precision_at_k'],.5)
        self.assertEqual(low['recall_at_k'],1/3)
        self.assertEqual(low['cutoff_ties']['possible_precision_min'],.5)
        self.assertEqual(low['cutoff_ties']['possible_precision_max'],1.)

    def test_group_metric_includes_missed_hazards_and_all_four_groups(self):
        rows=[]
        # Four hazards: group 0 is inverted, other groups are perfect.
        for i in range(4):
            r={'hazard_label':1,'p_hazard':.1,'pred_hazard':0}
            for j,g in enumerate(b.GROUPS):
                r['group_'+g]=int(i==j)
                r['p_'+g]=float((i!=j) if j==0 else (i==j))
            rows.append(r)
        rows.append({'hazard_label':0,'p_hazard':.9,'pred_hazard':1,
                     **{'group_'+g:0 for g in b.GROUPS},**{'p_'+g:1. for g in b.GROUPS}})
        primary,per=b.group_metrics(rows,[.5]*4)
        self.assertEqual(primary['n_images'],4);self.assertEqual(primary['balanced_accuracy'],.75)
        self.assertEqual([r['balanced_accuracy'] for r in per],[0.,1.,1.,1.])
        pipeline,_=b.group_metrics(rows,[.5]*4,gated=True)
        self.assertEqual(pipeline['n_images'],5);self.assertEqual(pipeline['balanced_accuracy'],.375)

    def test_test_split_cannot_select_thresholds(self):
        with self.assertRaises(ValueError): b.choose_thresholds([{'split':'pitt-test'}])

    def fixture(self,root):
        from PIL import Image
        (root/'label_schema.json').write_bytes(self.schema_bytes)
        for f in ['build_config.json','LABELING_RULES.md','PREPROCESSING.md']:(root/f).write_text('{}\n')
        for sidx,split in enumerate(b.SPLITS):
            rows=[]
            for i in range(64):
                mask=i%16;sid=f'{sidx:02}-{i:03}';cls='hazard' if mask else 'non_hazard'
                rel=f'{split}/{cls}/{sid}.png';p=root/rel;p.parent.mkdir(parents=True,exist_ok=True)
                im=Image.new('RGB',(224,224),(mask,i,sidx));im.save(p)
                r={f:'' for f in b.IDENTITY_FIELDS}
                r.update(sample_id=sid,image_path=rel,split=split,**{'class':cls},hazard_label=str(int(mask>0)),
                         original_class='Crosswalk',city='pittsburgh',is_manual_collection='0',
                         output_file_sha256=b.sha(p.read_bytes()),output_rgb_pixel_sha256=b.sha(im.tobytes()),
                         label_rule_version=self.schema['rule_version'])
                r.update({t:'0' for t in self.schema['all_tags']})
                for j,g in enumerate(b.GROUPS):
                    r['group_'+g]=str(int(bool(mask & (1<<j))))
                    r[self.schema['groups'][g]['positive_tags'][0]]=r['group_'+g]
                rows.append(r)
            b.save_csv(root/(split+'.csv'),rows)

    def test_input_target_validation_and_image_checksums(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);self.fixture(root)
            _,data=b.load_inputs(root)
            self.assertEqual(len(data['pitt-test']),64)
            self.assertTrue(any(r['hazard_label']=='1' and r['original_class']=='Crosswalk' for r in data['pitt-test']))
            row=data['pitt-test'][0];p=root/row['image_path'];p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'checksum'):b.load_inputs(root)

    def test_resume_after_api_error_without_repeating_successes(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'data';root.mkdir();self.fixture(root);out=Path(td)/'run'
            calls=[];fail=[True]
            class FakeClient:
                def __init__(self,**kw):self.responses=self
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def create(self,**kw):
                    raw=base64.b64decode(kw['input'][0]['content'][1]['image_url'].split(',')[1])
                    pixel=Image.open(io.BytesIO(raw)).getpixel((0,0));calls.append(pixel)
                    if len(calls)==4 and fail[0]:raise RuntimeError('simulated API failure')
                    mask=pixel[0]
                    pred={'p_hazard':.9 if mask else .1,
                          **{'p_'+g:.9 if mask & (1<<j) else .1 for j,g in enumerate(b.GROUPS)}}
                    body={'status':'completed','output_text':json.dumps(pred),'model':'offline-test',
                          'usage':{'input_tokens':1,'output_tokens':1,'total_tokens':2}}
                    return types.SimpleNamespace(**body,model_dump=lambda **kwargs:body)
            fake=types.SimpleNamespace(__version__='offline-test',OpenAI=FakeClient)
            argv=['runner','--data-root',str(root),'--out',str(out)]
            with patch.dict(sys.modules,{'openai':fake}),patch.dict(os.environ,{'OPENAI_API_KEY':'offline-test-key'}),contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys,'argv',argv),self.assertRaisesRegex(RuntimeError,'simulated'):b.main()
                self.assertEqual(json.loads((out/'status.json').read_text())['completed'],3)
                fail[0]=False
                with patch.object(sys,'argv',argv+['--resume']):b.main()
                after=len(calls)
                with patch.object(sys,'argv',argv+['--resume']):b.main()
                self.assertEqual(len(calls),after)
            self.assertEqual(len(calls),129)  # 128 successes + one failed request.
            self.assertEqual(len(set(calls)),128)
            result=json.loads((out/'results.json').read_text())
            self.assertEqual(result['test']['hazard']['precision_at_k'],1.)
            self.assertEqual(result['test']['groups_on_all_true_hazards']['balanced_accuracy'],1.)
            self.assertEqual(result['selection']['hazard_threshold'],.3)
            self.assertEqual(result['selection']['group_thresholds'],[.3]*4)
            self.assertEqual(json.loads((out/'status.json').read_text())['state'],'complete')
            self.assertTrue(json.loads((out/'final_verification.json').read_text())['passed'])

    def test_journal_partial_line_repair_and_duplicate_rejection(self):
        r={f:'' for f in b.IDENTITY_FIELDS};r.update(sample_id='one',output_file_sha256='abc')
        rec={'sample_id':'one','metadata':r,'image_sha256':'abc','status':'ok',
             'prediction':{f:.7 for f in b.PROBABILITY_FIELDS}}
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'predictions.jsonl';line=json.dumps(rec)+'\n'
            p.write_bytes(line.encode()+b'{"incomplete":')
            recovered=b.recover_journal(p,{'one':r})
            self.assertEqual(set(recovered),{'one'});self.assertEqual(p.read_text(),line)
            p.write_text(line+line)
            with self.assertRaises(ValueError):b.recover_journal(p,{'one':r})


if __name__=='__main__':unittest.main()
