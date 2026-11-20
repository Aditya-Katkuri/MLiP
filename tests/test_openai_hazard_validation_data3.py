import json, tempfile, unittest, sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/baseline'))
import openai_hazard_validation_data3 as m

class ValidationExperimentTests(unittest.TestCase):
    def test_ranking_does_not_depend_on_threshold(self):
        rows=[{'sample_id':str(i),'hazard_label':int(i<5),'p_hazard':1-i/100} for i in range(50)]
        result,_,_=m.evaluate(rows)
        self.assertEqual(result['ranking']['5']['precision'],1)
        self.assertEqual(result['ranking']['46']['hits'],5)
        self.assertEqual(result['selected_binary']['balanced_accuracy'],1)

    def test_dense_threshold_finds_low_scoring_hazards(self):
        rows=[{'sample_id':str(i),'hazard_label':int(i<25),'p_hazard':.07 if i<25 else .02} for i in range(50)]
        result,_,_=m.evaluate(rows)
        self.assertEqual(result['selected_binary']['threshold'],.07)
        self.assertEqual(result['selected_binary']['balanced_accuracy'],1)

    def test_ties_use_sample_identity_and_report_bounds(self):
        rows=[{'sample_id':f'{i:03d}','hazard_label':int(i%2==0),'p_hazard':.8} for i in range(50)]
        result,_,ranked=m.evaluate(rows[::-1]);top=result['ranking']['5']
        self.assertEqual(top['hits'],3)
        self.assertEqual(top['precision_min_over_ties'],0)
        self.assertEqual(top['precision_max_over_ties'],1)
        self.assertEqual(ranked[0]['sample_id'],'000')

    def test_invalid_probabilities_rejected(self):
        for p in [True,float('nan'),float('inf'),-.1,1.1]:
            with self.assertRaises(ValueError):m.validate_output({'decisive_evidence':'x','p_hazard':p},'route_focused')

    def test_request_contains_only_prompt_and_image(self):
        captured={}
        prediction={'decisive_evidence':'The visible route is clear.','p_hazard':.1}
        def create(**kw):
            captured.update(kw)
            return SimpleNamespace(status='completed',model=m.MODEL,output_text=json.dumps(prediction),model_dump=lambda **k:{'model':m.MODEL})
        client=SimpleNamespace(responses=SimpleNamespace(create=create))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'image.png').write_bytes(b'image')
            row={'sample_id':'SECRET_ID','image_path':'image.png','output_file_sha256':m.sha(b'image'),'hazard_label':'SECRET_LABEL'}
            record=m.request(client,root,row,'route_focused','FIXED_PROMPT')
        self.assertEqual(record['status'],'ok')
        self.assertEqual(captured['reasoning'],{'effort':'low'})
        self.assertEqual(captured['input'][0]['content'][0]['text'],'FIXED_PROMPT')
        self.assertNotIn('SECRET',json.dumps(captured))
        self.assertFalse(captured['store'])

    def test_journal_recovers_success_and_repairs_partial_tail(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'predictions.jsonl';r={'variant':'route_focused','sample_id':'a','image_sha256':'123','status':'ok',
                'prediction':{'decisive_evidence':'x','p_hazard':.1}}
            p.write_bytes((json.dumps(r)+'\n').encode()+b'{"partial":')
            result=m.load_completed(p,{'a':{'output_file_sha256':'123'}})
            self.assertEqual(len(result),1)
            self.assertEqual(p.read_text(),json.dumps(r)+'\n')

    def test_validation_loader_cannot_read_test_data(self):
        root=Path(__file__).resolve().parent.parent/'data_reference'
        if not root.exists():self.skipTest('Local dataset snapshot unavailable')
        original=Path.open
        def guard(path,*args,**kwargs):
            self.assertNotIn('pitt-test',str(path))
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guard):schema,rows=m.load_validation(root)
        self.assertEqual(len(rows),382)
        self.assertEqual(sum(int(r['hazard_label']) for r in rows),157)
        prompts=m.make_prompts(schema)
        for text in prompts.values():
            for tag in schema['all_tags']:self.assertIn(tag,text)

if __name__=='__main__':unittest.main()
