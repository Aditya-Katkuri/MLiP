import hashlib,json,os,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE.parent),str(HERE.parent/'validation_revision'),str(HERE.parent/'src/baseline')]
import openai_hazard_test_data3 as m

class FakeClient:
    calls=0
    def __init__(self,**kw):pass
    def __enter__(self):self.responses=self;return self
    def __exit__(self,*args):return False
    def create(self,**kw):
        type(self).calls+=1
        assert kw['reasoning']=={'effort':'low'} and kw['store'] is False
        assert kw['text']['format']['name']=='evidence_checklist'
        assert len(kw['input'])==1 and len(kw['input'][0]['content'])==2
        score=int(hashlib.sha256(kw['input'][0]['content'][1]['image_url'].encode()).hexdigest()[:4],16)%100/100
        prediction={g:{'status':'uncertain','evidence':'Offline synthetic observation.'} for g in m.v.GROUPS}
        prediction['p_hazard']=score;text=json.dumps(prediction)
        raw={'model':m.v.MODEL,'reasoning':{'effort':'low'},'status':'completed','store':False,
             'output':[{'type':'message','content':[{'type':'output_text','text':text}]}],
             'usage':{'input_tokens':0,'output_tokens':0,'total_tokens':0}}
        return SimpleNamespace(model=m.v.MODEL,status='completed',output_text=text,model_dump=lambda **kwargs:raw)

class TestFrozenTestRun(unittest.TestCase):
    def test_uses_fixed_threshold_without_search(self):
        rows=[]
        for i in range(60):
            row={'sample_id':f'{i:03d}','hazard_label':int(i%3!=0),'p_hazard':.8 if i%2 else .2}
            for j,g in enumerate(m.v.GROUPS):row['group_'+g]=int(i%3!=0 and (i+j)%2==0);row['p_'+g]=.9 if (i+j)%2==0 else .1
            rows.append(row)
        with patch.object(m.v,'evaluate',side_effect=AssertionError('No threshold selection on test')):
            metrics,classified=m.evaluate_fixed(rows,{'hazard_threshold':.3,'candidate':'evidence_checklist'}, {'group_thresholds':[.5]*4})
        self.assertEqual(metrics['hazard_threshold'],.3)
        self.assertEqual(metrics['hazard_top10']['k'],6)
        self.assertTrue(all(r['pred_hazard']==int(r['p_hazard']>=.3) for r in classified))

    def test_full_offline_run_and_resume_preserve_subtypes(self):
        data=HERE.parent/'data_reference'
        if not data.exists():self.skipTest('Local offline data fixtures unavailable')
        selected=HERE.parent/'validation_revision/results/selected_hazard_model.json'
        original=HERE.parent/'audit_20261120/source'
        FakeClient.calls=0
        with tempfile.TemporaryDirectory() as d,patch.dict(sys.modules,{'openai':SimpleNamespace(__version__='offline-test',OpenAI=FakeClient)}),patch.dict(os.environ,{'OPENAI_API_KEY':'offline-test'}),patch.object(m.v,'evaluate',side_effect=AssertionError('Test must not select thresholds')),patch.object(m.old,'choose_thresholds',side_effect=AssertionError('Test must not select thresholds')):
            out=Path(d)/'output';args=['runner','--data-root',str(data),'--selected',str(selected),'--original-run',str(original),'--out',str(out)]
            with patch.object(sys,'argv',args):m.main()
            first=json.loads((out/'results.json').read_text());self.assertEqual(FakeClient.calls,380)
            self.assertTrue(first['subtype_unchanged']);self.assertFalse(first['test_used_for_selection'])
            self.assertEqual(first['test_metrics']['hazard_threshold'],.3)
            self.assertAlmostEqual(first['test_metrics']['groups_on_all_true_hazards']['balanced_accuracy'],.683038474041105)
            with patch.object(sys,'argv',args+['--resume']):m.main()
            self.assertEqual(FakeClient.calls,380)
            self.assertEqual(json.loads((out/'results.json').read_text()),first)
            self.assertEqual(len((out/'predictions.jsonl').read_text().splitlines()),380)

if __name__=='__main__':unittest.main()
