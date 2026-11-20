"""Offline tests for independent CLIP comparisons and inference cache identity."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'clip_zero_shot_data3.py'
if not SOURCE.exists():SOURCE=HERE.parent/'src/baseline/clip_zero_shot_data3.py'
sys.path.insert(0,str(SOURCE.parent))
spec=importlib.util.spec_from_file_location('clip_data3',SOURCE)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class ClipTests(unittest.TestCase):
    def test_five_independent_pair_scores_not_a_five_class_softmax(self):
        features=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
        prototypes=np.array([[[0.,1.],[1.,0.]]]*5,dtype=np.float32)
        scores,cosines=c.score_pairs(features,prototypes,1.)
        self.assertEqual(scores.shape,(2,5));self.assertEqual(cosines.shape,(2,5,2))
        np.testing.assert_allclose(scores[0],1/(1+np.exp(-1)),rtol=1e-6)
        np.testing.assert_allclose(scores[1],1/(1+np.exp(1)),rtol=1e-6)
        self.assertTrue(np.all(scores[0]>.5));self.assertGreater(scores[0].sum(),1.)

    def test_one_group_comparison_cannot_change_other_group_or_hazard_scores(self):
        features=np.array([[1.,0.]],dtype=np.float32)
        prototypes=np.array([[[0.,1.],[1.,0.]]]*5,dtype=np.float32)
        before,_=c.score_pairs(features,prototypes,100.)
        prototypes[2]=prototypes[2,::-1]
        after,_=c.score_pairs(features,prototypes,100.)
        np.testing.assert_array_equal(before[:,[0,1,3,4]],after[:,[0,1,3,4]])
        self.assertLess(after[0,2],before[0,2])

    def test_pair_softmax_is_finite_at_large_scale_and_equal_scores(self):
        features=np.array([[1.,0.]],dtype=np.float32)
        same=np.array([[[1.,0.],[1.,0.]]]*5,dtype=np.float32)
        scores,_=c.score_pairs(features,same,100000.)
        np.testing.assert_array_equal(scores,.5*np.ones((1,5)))
        for scale in [0.,-1.,float('nan')]:
            with self.subTest(scale=scale),self.assertRaises(ValueError):c.score_pairs(features,same,scale)

    def test_prompt_mean_normalizes_each_prompt_before_averaging(self):
        # Unequal input magnitudes must not change equal weighting of prompt directions.
        raw=np.array([[10.,0.],[0.,1.]]*10,dtype=np.float32)
        result=c.make_prototypes(raw)
        self.assertEqual(result.shape,(5,2,2))
        np.testing.assert_allclose(result,np.full((5,2,2),1/np.sqrt(2)),rtol=1e-6)
        with self.assertRaises(ValueError):c.make_prototypes(raw[:-1])
        with self.assertRaises(ValueError):c.normalize(np.zeros((1,2)))

    def test_feature_cache_rejects_wrong_identity_and_unnormalized_features(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'chunk.npz';features=np.eye(2,dtype=np.float32)
            c.save_chunk(p,features,['a','b'])
            np.testing.assert_array_equal(c.load_chunk(p,['a','b'],2),features)
            with self.assertRaises(ValueError):c.load_chunk(p,['b','a'],2)
            with self.assertRaises(ValueError):c.load_chunk(p,['a','b'],3)
            c.save_chunk(p,features*2,['a','b'])
            with self.assertRaises(ValueError):c.load_chunk(p,['a','b'],2)

    def test_vlm_comparison_checks_images_and_label_manifests(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'data';ref=Path(td)/'vlm';root.mkdir();ref.mkdir()
            (root/'pitt-test.csv').write_text('ground truth labels')
            inv=[{'sample_id':'one','sha256':'imagehash'}]
            (ref/'image_inventory.json').write_text(json.dumps(inv))
            (ref/'run_config.json').write_text(json.dumps({'binding':{'input_hashes':{'pitt-test.csv':c.file_sha(root/'pitt-test.csv')}}}))
            self.assertTrue(c.verify_reference(ref,root,inv)['checked'])
            with self.assertRaises(ValueError):c.verify_reference(ref,root,[{'sample_id':'one','sha256':'different'}])
            (root/'pitt-test.csv').write_text('changed label')
            with self.assertRaises(ValueError):c.verify_reference(ref,root,inv)

    def test_prompt_roster_has_exactly_two_prompts_for_each_polarity(self):
        self.assertEqual(list(c.PROMPTS),c.TASKS)
        for t in c.TASKS:
            self.assertEqual(set(c.PROMPTS[t]),{'negative','positive'})
            for p in c.POLARITIES:self.assertEqual(len(c.PROMPTS[t][p]),2)


if __name__=='__main__':unittest.main()
