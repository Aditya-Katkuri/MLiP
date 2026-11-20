"""Offline SIFT task, selection, serialization and input-scope checks."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'sift_baseline_data3.py'
if not SOURCE.exists():SOURCE=HERE.parent/'src/baseline/sift_baseline_data3.py'
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('sift_data3',SOURCE)
s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)


def row(i,mask):
    r={k:'' for k in s.task.IDENTITY_FIELDS}
    r.update(sample_id=str(i),hazard_label=str(int(mask>0)),split='pitt-train',image_path=f'pitt-train/x/{i}.png',output_file_sha256='hash')
    for j,g in enumerate(s.task.GROUPS):r['group_'+g]=str(int(bool(mask&(1<<j))))
    return r


class SiftTests(unittest.TestCase):
    def test_overlapping_groups_and_hazard_only_group_negatives(self):
        rows=[row(i,mask) for i,mask in enumerate([0,1,2,3,4,8])]
        contributions,sets,codebook=s.select_training(rows,'pitt_only')
        self.assertEqual(set(sets[('hazard',0)]),{'0'})
        self.assertEqual(set(sets[('hazard',1)]),{'1','2','3','4','5'})
        self.assertEqual(set(sets[(s.task.GROUPS[0],1)]),{'1','3'})
        self.assertEqual(set(sets[(s.task.GROUPS[1],1)]),{'2','3'})
        for group in s.task.GROUPS:
            self.assertNotIn('0',sets[(group,0)]);self.assertNotIn('0',sets[(group,1)])
        self.assertEqual(codebook,{'0','1','2','3','4','5'})

    def test_selection_deterministic_and_caps_each_polarity(self):
        rows=[row(i,i%16) for i in range(9600)]
        a=s.select_training(rows,'pooled_other_and_pitt')
        b=s.select_training(list(reversed(rows)),'pooled_other_and_pitt')
        self.assertEqual(a,b)
        self.assertTrue(all(len(v)==500 for v in a[1].values()))
        for target in s.TASKS:
            for polarity in [0,1]:
                self.assertEqual(sum(r['in_codebook'] for r in a[0] if r['task']==target and r['polarity']==polarity),200)

    def test_distances_and_independent_scores(self):
        rows=[row(0,0),row(1,15),row(2,3)]
        h=np.array([[1.,0.],[0.,1.],[.5,.5]])
        p=np.array([[[1.,0.],[0.,1.]]]*5)
        with patch.object(s,'N_WORDS',2):scores=s.score_histograms(rows,h,p)
        for field in s.task.PROBABILITY_FIELDS:
            self.assertEqual([r[field] for r in scores],[0.,1.,.5])
        self.assertEqual(scores[1]['margin_hazard'],1.)
        self.assertEqual(scores[0]['distance_negative_hazard'],0.)
        self.assertEqual(scores[0]['distance_positive_hazard'],1.)
        self.assertEqual(sum(scores[1]['p_'+g] for g in s.task.GROUPS),4.)
        changed=[dict(r,hazard_label='1',original_class='Different') for r in rows]
        with patch.object(s,'N_WORDS',2):after=s.score_histograms(changed,h,p)
        for a,b in zip(scores,after):
            self.assertEqual([a[k] for k in s.task.PROBABILITY_FIELDS],[b[k] for k in s.task.PROBABILITY_FIELDS])

    def test_prototypes_reproduce_training_means_after_save_reload(self):
        hist={'a':np.array([1.,0.]),'b':np.array([.5,.5]),'c':np.array([0.,1.])}
        sets={(t,0):['a','b'] for t in s.TASKS};sets.update({(t,1):['c'] for t in s.TASKS})
        p=s.mean_prototypes(hist,sets)
        np.testing.assert_array_equal(p[:,0],np.array([[.75,.25]]*5))
        np.testing.assert_array_equal(p[:,1],np.array([[0.,1.]]*5))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'model.npz';s.save_npz(path,prototypes=p)
            with np.load(path,allow_pickle=False) as z:np.testing.assert_array_equal(z['prototypes'],p)

    def test_validation_cutoff_and_rejection_of_test_selection(self):
        threshold,table=s.best_cutoff([0,0,1,1],[.40,.45,.46,.47])
        self.assertEqual(threshold,.46)
        self.assertEqual(next(r for r in table if r['threshold']==threshold)['balanced_accuracy'],1.)
        with self.assertRaises(ValueError):s.select_thresholds([{'split':'pitt-test'}])

    def test_independent_plan_selection_and_precision_tie_break(self):
        summaries=[{'plan':'pitt_only','hazard_top46_hits':30,'hazard_balanced_accuracy':.6,'group_macro_balanced_accuracy':.8,
                    'hazard_threshold':.45,'group_thresholds':'[0.4,0.5,0.6,0.7]'},
                   {'plan':'pooled_other_and_pitt','hazard_top46_hits':30,'hazard_balanced_accuracy':.7,'group_macro_balanced_accuracy':.6,
                    'hazard_threshold':.48,'group_thresholds':'[0.5,0.5,0.5,0.5]'}]
        result=s.choose_plans(summaries)
        self.assertEqual(result['hazard_model_plan'],'pooled_other_and_pitt')
        self.assertEqual(result['group_model_plan'],'pitt_only')
        self.assertEqual(result['group_thresholds'],[.4,.5,.6,.7])
        summaries[0]['hazard_top46_hits']=31
        self.assertEqual(s.choose_plans(summaries)['hazard_model_plan'],'pitt_only')

    def test_descriptor_cache_identity_and_real_dense_sift(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);path=root/'image.png'
            img=np.zeros((224,224,3),dtype=np.uint8)
            img[::8,:,:]=255;img[:,::8,:]=255;s.cv2.imwrite(str(path),img)
            raw=s.engine._descriptors(str(path))
            self.assertEqual(raw['desc'].shape[1],128)
            self.assertGreaterEqual(raw['n_desc'],s.engine.MIN_DESCRIPTORS);self.assertLessEqual(raw['n_desc'],80)
            cache=root/'cache.npz'
            s.save_npz(cache,sample_ids=np.array(['a']),descriptors=raw['desc'],offsets=np.array([0,raw['n_desc']]),
                       n_sparse=np.array([raw['n_sparse']]),n_dense=np.array([raw['n_dense']]))
            restored=s.get_descriptors([{'sample_id':'a','image_path':'image.png'}],root,cache,1)
            np.testing.assert_array_equal(restored[0]['desc'],raw['desc'])
            with self.assertRaises(ValueError):s.get_descriptors([{'sample_id':'b','image_path':'image.png'}],root,cache,1)


if __name__=='__main__':unittest.main()
