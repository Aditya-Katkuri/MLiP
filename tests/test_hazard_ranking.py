import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fraction = load('hazard_fraction', 'scripts/hazard_top_fraction.py')
review = load('pilot_review', 'scripts/sidewalk_pilot/evaluate_precision_at_12.py')


class RankingTests(unittest.TestCase):
    def test_top_fraction_uses_ground_truth_and_stable_ties(self):
        rows = [dict(sample_id=s, p_hazard=p, hazard_label=y, pred_hazard='1')
                for s, p, y in [('b', '.9', '0'), ('a', '.9', '1'), ('c', '.1', '1')]]
        result = fraction.evaluate(rows, '.1')
        self.assertEqual(result['k'], 1)
        self.assertEqual(result['precision_at_top_fraction'], 1)
        self.assertTrue(result['tie_crosses_cutoff'])
        self.assertEqual(fraction.evaluate(list(reversed(rows)), '.1'), result)
        self.assertEqual(fraction.evaluate(rows, '.5')['precision_at_top_fraction'], .5)

    def test_reject_invalid_or_mixed_predictions(self):
        row = dict(sample_id='a', p_hazard='.8', hazard_label='1', split='pitt-test')
        for rows in [[], [row, row], [dict(row, p_hazard='nan')],
                     [dict(row, hazard_label='uncertain')],
                     [row, dict(row, sample_id='b', split='pitt-val')]]:
            with self.assertRaises(ValueError): fraction.evaluate(rows)
        for value in ['0', '-1', '1.1', 'NaN', 'Infinity']:
            with self.assertRaises(ValueError): fraction.evaluate([row], value)

    def test_review_denominator_and_within_stretch_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'results').mkdir()
            mapping = [dict(stretch=s, image_number=i,
                            natural_order=25-i if s in 'BD' else i)
                       for s in 'ABCD' for i in range(1, 25)]
            with (root/'results/review_mapping.csv').open('w', newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(mapping[0])); w.writeheader(); w.writerows(mapping)
            labels = [dict(x, hazard_label='hazard' if x['image_number']<=12 else 'non_hazard') for x in mapping]
            result = review.evaluate(root, labels)
            self.assertEqual(len(result), 6)
            for row in result:
                self.assertEqual(row['precision_at_12'], 1 if row['ordering']=='presented_order' else 0)
            labels[0]['hazard_label']='uncertain'
            row=review.evaluate(root, labels)[0]
            self.assertEqual(row['precision_at_12'], '')
            self.assertEqual(row['precision_at_12_lower_bound'], 11/12)
            self.assertEqual(row['precision_at_12_upper_bound'], 1)
            with self.assertRaises(ValueError): review.evaluate(root, labels+[labels[0]])
            with self.assertRaises(ValueError): review.evaluate(root, [dict(labels[0], image_number=99)])


if __name__ == '__main__':
    unittest.main()
