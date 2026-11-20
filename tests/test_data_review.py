import base64
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

MODULE = Path(__file__).resolve().parents[1] / 'src/data_review.py'
spec = importlib.util.spec_from_file_location('data_review', MODULE)
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/lGkAAAAASUVORK5CYII=')


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.directory = self.root / 'pitt-train/CurbRamp'
        self.directory.mkdir(parents=True)
        self.filename = 'gsv-pittsburgh-1-CurbRamp.png'
        (self.directory / self.filename).write_bytes(PNG)
        self.store = app.ReviewStore(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def read(self):
        with (self.directory / app.CSV_NAME).open(newline='') as handle:
            return list(csv.DictReader(handle))

    def test_persistence_and_correction_without_duplicate_rows(self):
        before = hashlib.sha256((self.directory / self.filename).read_bytes()).hexdigest()
        self.assertFalse((self.directory / app.CSV_NAME).exists())
        self.store.save('pitt-train/CurbRamp', self.filename, 'CurbRamp', None)
        fresh = app.ReviewStore(self.root)
        image = fresh.listing('pitt-train/CurbRamp')['images'][0]
        self.assertEqual(image['label'], 'CurbRamp')
        fresh.save('pitt-train/CurbRamp', self.filename, 'Obstacle', 'CurbRamp')
        self.assertEqual(len(self.read()), 1)
        self.assertEqual(self.read()[0]['label'], 'Obstacle')
        self.assertTrue(self.read()[0]['updated_at'])
        self.assertEqual(hashlib.sha256((self.directory / self.filename).read_bytes()).hexdigest(), before)

    def test_existing_csv_extra_columns_and_absent_images_are_preserved(self):
        with (self.directory / app.CSV_NAME).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerows([['filename', 'label', 'note'], [self.filename, 'CurbRamp', 'retain note'],
                              ['removed.png', 'Obstacle', 'retain orphan']])
        self.store.save('pitt-train/CurbRamp', self.filename, 'Obstacle', 'CurbRamp')
        rows = {r['filename']: r for r in self.read()}
        self.assertEqual(rows[self.filename]['note'], 'retain note')
        self.assertEqual(rows['removed.png']['note'], 'retain orphan')
        self.assertEqual(len(self.store.listing('pitt-train/CurbRamp')['images']), 1)

    def test_stale_label_is_rejected_without_overwrite(self):
        self.store.save('pitt-train/CurbRamp', self.filename, 'CurbRamp', None)
        with self.assertRaises(app.ReviewError) as caught:
            self.store.save('pitt-train/CurbRamp', self.filename, 'Obstacle', None)
        self.assertEqual(caught.exception.status, 409)
        self.assertEqual(caught.exception.details['label'], 'CurbRamp')
        self.assertEqual(self.read()[0]['label'], 'CurbRamp')

    def test_concurrent_updates_preserve_every_image(self):
        filenames = [f'image-{i}.png' for i in range(12)]
        for filename in filenames:
            (self.directory / filename).write_bytes(PNG)
        def save(filename):
            app.ReviewStore(self.root).save('pitt-train/CurbRamp', filename, 'CurbRamp', None)
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(save, filenames))
        self.assertEqual({r['filename'] for r in self.read()}, set(filenames))
        self.assertFalse(list(self.directory.glob('*.tmp')))

    def test_invalid_csv_is_not_rewritten(self):
        path = self.directory / app.CSV_NAME
        path.write_text('filename,label\n' + self.filename + ',invalid\n')
        before = path.read_bytes()
        with self.assertRaises(app.ReviewError):
            self.store.save('pitt-train/CurbRamp', self.filename, 'CurbRamp', None)
        self.assertEqual(path.read_bytes(), before)

    def test_traversal_symlinks_and_invalid_labels_are_rejected(self):
        for directory in ('../CurbRamp', 'pitt-train/../CurbRamp', 'pitt-train/..'):
            with self.assertRaises(app.ReviewError):
                self.store.directory(directory)
        with self.assertRaises(app.ReviewError):
            self.store.image(self.directory, '../hidden.png')
        outside = self.root / 'outside.png'
        outside.write_bytes(PNG)
        (self.directory / 'link.png').symlink_to(outside)
        with self.assertRaises(app.ReviewError):
            self.store.image(self.directory, 'link.png')
        with self.assertRaises(app.ReviewError):
            self.store.save('pitt-train/CurbRamp', self.filename, 'maybe', None)
        self.assertFalse((self.directory / app.CSV_NAME).exists())

    def test_directory_listing_only_counts_images(self):
        (self.directory / 'readme.txt').write_text('not an image')
        self.assertEqual(self.store.directories()['directories'][0]['image_count'], 1)
        image = self.store.listing('pitt-train/CurbRamp')['images'][0]
        self.assertIsNone(image['label'])
        self.assertFalse((self.directory / app.CSV_NAME).exists())

    def test_all_review_classes_can_be_assigned_and_changed(self):
        previous = None
        self.assertEqual(self.store.directories()['classes'], list(app.CLASSES))
        self.assertEqual(set(app.CLASSES), {'Crosswalk', 'CurbRamp', 'Obstacle', 'SurfaceProblem', 'no_obstacles', 'remove_data'})
        for label in app.CLASSES:
            self.store.save('pitt-train/CurbRamp', self.filename, label, previous)
            fresh = app.ReviewStore(self.root)
            self.assertEqual(fresh.listing('pitt-train/CurbRamp')['images'][0]['label'], label)
            self.assertEqual(len(self.read()), 1)
            previous = label

    def test_binary_labels_remain_separate_and_are_not_assignable(self):
        old_csv = self.directory / 'manual_labels.csv'
        old_csv.write_text('filename,label\n' + self.filename + ',use\n')
        original = old_csv.read_bytes()
        self.assertIsNone(self.store.listing('pitt-train/CurbRamp')['images'][0]['label'])
        for label in ('use', 'do not use'):
            with self.assertRaises(app.ReviewError):
                self.store.save('pitt-train/CurbRamp', self.filename, label, None)
        self.store.save('pitt-train/CurbRamp', self.filename, 'SurfaceProblem', None)
        self.assertEqual(old_csv.read_bytes(), original)
        self.assertEqual(self.read()[0]['label'], 'SurfaceProblem')

    def test_remove_data_is_saved_without_deleting_and_can_be_revised(self):
        image = self.directory / self.filename
        original = image.read_bytes()
        self.store.save('pitt-train/CurbRamp', self.filename, 'remove_data', None)
        self.assertEqual(image.read_bytes(), original)
        fresh = app.ReviewStore(self.root)
        self.assertEqual(fresh.listing('pitt-train/CurbRamp')['images'][0]['label'], 'remove_data')
        fresh.save('pitt-train/CurbRamp', self.filename, 'CurbRamp', 'remove_data')
        self.assertEqual(image.read_bytes(), original)
        self.assertEqual(len(self.read()), 1)
        self.assertEqual(self.read()[0]['label'], 'CurbRamp')


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        directory = self.root / 'pitt-test/no_obstacles'
        directory.mkdir(parents=True)
        (directory / 'sample.png').write_bytes(PNG)
        self.server = app.make_server(self.root, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def request(self, path, body=None, origin=None):
        headers = {'Content-Type': 'application/json'}
        if origin:
            headers['Origin'] = origin
        request = Request(self.url + path, data=json.dumps(body).encode() if body else None, headers=headers)
        return urlopen(request, timeout=5)

    def test_browser_endpoints_and_save_reload_contract(self):
        with self.request('/') as response:
            self.assertIn(b'Class review', response.read())
        with self.request('/api/images?directory=pitt-test/no_obstacles') as response:
            listing = json.load(response)
        with self.request(listing['images'][0]['url']) as response:
            self.assertEqual(response.read(), PNG)
        body = {'directory': 'pitt-test/no_obstacles', 'filename': 'sample.png',
                'label': 'CurbRamp', 'expected_label': None}
        with self.request('/api/label', body, self.url) as response:
            self.assertEqual(json.load(response)['label'], 'CurbRamp')
        with self.request('/api/images?directory=pitt-test/no_obstacles') as response:
            self.assertEqual(json.load(response)['images'][0]['label'], 'CurbRamp')

    def test_cross_origin_mutation_is_rejected(self):
        body = {'directory': 'pitt-test/no_obstacles', 'filename': 'sample.png',
                'label': 'CurbRamp', 'expected_label': None}
        with self.assertRaises(HTTPError) as caught:
            self.request('/api/label', body, 'http://another-origin.invalid')
        self.assertEqual(caught.exception.code, 403)
        caught.exception.close()
        self.assertFalse(list(self.root.rglob(app.CSV_NAME)))


if __name__ == '__main__':
    unittest.main()
