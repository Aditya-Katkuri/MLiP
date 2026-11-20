"""Offline checks for the processed five-class CLIP evaluation; no model download."""
from pathlib import Path
import csv
import importlib.util
import tempfile
import unittest

from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / 'src/baseline/clip_zero_shot.py'
spec = importlib.util.spec_from_file_location('clip_zero_shot_under_test', SCRIPT)
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


class ClipEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index, cls in enumerate(baseline.CLASSES):
            directory = self.root / 'pitt-test' / cls
            directory.mkdir(parents=True)
            filename = f'gsv-pittsburgh-{index}-{cls}.png'
            Image.new('RGB', (224, 224), (index, 0, 0)).save(directory / filename)
            self.rows.append({'filename': filename, 'class': cls, 'split': 'pitt-test',
                              'image_path': f'pitt-test/{cls}/{filename}'})
        self.manifest = self.root / 'pitt-test.csv'
        self.write_manifest()

    def write_manifest(self, rows=None):
        with self.manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.rows[0])
            writer.writeheader(); writer.writerows(self.rows if rows is None else rows)

    def test_loads_all_five_classes_and_uses_current_processed_filename(self):
        self.rows[-1]['filename'] = self.rows[-1]['filename'].replace('.png', '.webp')
        self.write_manifest()
        images = baseline.load_images(self.root, self.manifest)
        self.assertEqual([image['y'] for image in images], baseline.CLASSES)
        self.assertTrue(images[-1]['metadata']['filename'].endswith('.png'))
        self.assertTrue(images[-1]['source_metadata']['filename'].endswith('.webp'))
        self.assertTrue(all(len(image['image_sha256']) == 64 for image in images))

    def test_rejects_wrong_split_missing_roster_and_wrong_size(self):
        self.rows[0]['split'] = 'pitt-val'
        self.write_manifest()
        with self.assertRaises(ValueError): baseline.load_images(self.root, self.manifest)
        self.rows[0]['split'] = 'pitt-test'
        self.write_manifest(self.rows[1:])
        with self.assertRaisesRegex(ValueError, 'roster'): baseline.load_images(self.root, self.manifest)
        self.write_manifest()
        Image.new('RGB', (1440, 960)).save(self.root / self.rows[0]['image_path'])
        with self.assertRaisesRegex(ValueError, '224x224'): baseline.load_images(self.root, self.manifest)

    def test_probability_vector_requires_five_values_and_consistent_argmax(self):
        self.assertEqual(baseline.validate_probabilities([0.2] * 5)['predicted_class'], 'Crosswalk')
        self.assertEqual(baseline.validate_probabilities([0, 0, 0, 1, 0])['predicted_class'], 'SurfaceProblem')
        for bad in [[0.25] * 4, [0.1] * 5, [float('nan'), 0, 0, 0, 1], [2, 0, 0, 0, -1]]:
            with self.subTest(values=bad), self.assertRaises(ValueError): baseline.validate_probabilities(bad)

    def test_crosswalk_false_positives_count_in_precision_but_not_subtype_accuracy(self):
        rows = []
        pairs = [('Crosswalk', 'Obstacle'), ('CurbRamp', 'SurfaceProblem'), ('Obstacle', 'Obstacle'),
                 ('Obstacle', 'SurfaceProblem'), ('SurfaceProblem', 'Obstacle'), ('SurfaceProblem', 'SurfaceProblem')]
        for index, (actual, pred) in enumerate(pairs):
            values = dict.fromkeys(baseline.CLASSES, 0.0)
            values[pred] = 0.9 - 0.05 * index
            values['SurfaceProblem' if pred == 'Obstacle' else 'Obstacle'] = 0.05
            values['Crosswalk'] = 1 - sum(values.values())
            rows.append({'image_path': f'pitt-test/{actual}/{index}.png', 'y': actual, 'predicted_class': pred,
                         **{'p_' + key: value for key, value in values.items()}})
        for index in range(44):
            rows.append({'image_path': f'pitt-test/Crosswalk/filler-{index:02}.png', 'y': 'Crosswalk',
                         'predicted_class': 'Crosswalk', 'p_Crosswalk': 0.9, 'p_CurbRamp': 0.0,
                         'p_Obstacle': 0.05, 'p_SurfaceProblem': 0.05, 'p_no_obstacles': 0.0})
        result, ranked = baseline.hazard_metrics(rows)
        self.assertEqual(result['precision_at_46']['true_hazards'], 4)
        self.assertEqual(result['precision_at_46']['precision'], 4 / 46)
        self.assertEqual(result['all_predicted_hazards']['count'], 6)
        self.assertEqual(result['all_predicted_hazards']['false_positive_nonhazards'], 2)
        self.assertEqual(result['all_predicted_hazards']['subtype']['balanced_accuracy'], 0.5)
        self.assertEqual(result['top_46_hazard_subtype']['balanced_accuracy'], 0.5)
        self.assertEqual([row['image_path'] for row in ranked[-44:]], sorted(row['image_path'] for row in rows[-44:]))

    def test_subtype_balanced_accuracy_is_undefined_without_both_true_subclasses(self):
        result = baseline.subtype_metrics([{'y': 'Obstacle', 'predicted_class': 'Obstacle'},
                                          {'y': 'Crosswalk', 'predicted_class': 'SurfaceProblem'}])
        self.assertIsNone(result['balanced_accuracy'])
        self.assertEqual(result['evaluated_true_hazards'], 1)


if __name__ == '__main__':
    unittest.main()
