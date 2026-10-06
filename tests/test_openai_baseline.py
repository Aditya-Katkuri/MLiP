"""Offline regression checks; these tests never call the OpenAI API."""

import base64
from collections import Counter
import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

# Keep compiled numerical modules loaded when the fake SDK patch restores sys.modules.
from sklearn import metrics


SCRIPT = Path(__file__).resolve().parents[1] / "src/baseline/openai_baseline.py"
SPEC = importlib.util.spec_from_file_location("openai_baseline_under_test", SCRIPT)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aZVQAAAAASUVORK5CYII="
)
WEBP = base64.b64decode("UklGRh4AAABXRUJQVlA4TBEAAAAvAAAAAAdQuSYVs/+BiOh/AAA=")


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "test.csv"
        self.rows = []
        for index, cls in enumerate([*baseline.CLASSES, "Crosswalk"], 101):
            suffix = ".webp" if cls == "no_obstacles" else ".png"
            filename = f"gsv-pittsburgh-{index}-{cls}{suffix}"
            relative = f"test/{cls}/{filename}"
            path = self.root / relative
            path.parent.mkdir(parents=True)
            path.write_bytes(WEBP if suffix == ".webp" else PNG)
            self.rows.append({"filename": filename, "image_path": relative,
                              "split": "test", "class": cls,
                              "normalized_x": "0.25", "annotation_tag": "LOCAL_ONLY_SENTINEL"})
        self.write_manifest()

    def write_manifest(self, rows=None):
        with self.manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.rows[0].keys())
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def load(self):
        return baseline.load_pittsburgh_images(self.manifest, self.root)

    def probabilities(self, target):
        return {f"p_{cls}": 1.0 if cls == target else 0.0 for cls in baseline.CLASSES}

    def invoke(self, *extra):
        argv = [str(SCRIPT), "--manifest", str(self.manifest), "--data-root", str(self.root), *extra]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output):
            baseline.main()
        return output.getvalue()

    def test_ground_truth_uses_new_classes_and_retains_source_metadata(self):
        images = self.load()
        self.assertEqual([image["y"] for image in images], baseline.CLASSES)
        for image in images:
            self.assertEqual(image["metadata"]["class"], image["y"])
            self.assertEqual(image["metadata"]["city"], "pittsburgh")
            self.assertEqual(image["source_metadata"]["annotation_tag"], "LOCAL_ONLY_SENTINEL")
            self.assertEqual(Path(image["image_path"]).read_bytes(), WEBP if image["y"] == "no_obstacles" else PNG)

    def test_crosswalk_images_are_skipped_without_opening_their_files(self):
        crosswalk = next(row for row in self.rows if row["class"] == "Crosswalk")
        (self.root / crosswalk["image_path"]).unlink()
        excluded = Counter()
        images = baseline.load_pittsburgh_images(self.manifest, self.root, excluded_counts=excluded)
        self.assertEqual(len(images), 4)
        self.assertEqual(dict(excluded), {"Crosswalk": 1})
        self.assertNotIn("Crosswalk", [image["y"] for image in images])
        self.assertNotIn("Crosswalk", baseline.PROMPT)
        self.assertNotIn("p_Crosswalk", baseline.CSV_FIELDS)
        self.write_manifest([crosswalk])
        with self.assertRaisesRegex(ValueError, "No eligible Pittsburgh test images"):
            self.load()

    def test_rejects_wrong_split_city_class_path_and_stale_targets(self):
        for changes in (
            {"split": "train"},
            {"split": "val"},
            {"filename": "gsv-columbus-101-CurbRamp.png"},
            {"class": "missing_curb_ramp"},
            {"class": "Obstacle"},
            {"image_path": "../outside.png"},
            {"image_path": "train/CurbRamp/gsv-pittsburgh-101-CurbRamp.png"},
            {"y": "no_barrier"},
        ):
            with self.subTest(changes=changes):
                rows = [dict(row) for row in self.rows]
                rows[0].update(changes)
                if "y" in changes:
                    fields = [*self.rows[0], "y"]
                    with self.manifest.open("w", newline="") as handle:
                        writer = csv.DictWriter(handle, fieldnames=fields)
                        writer.writeheader()
                        writer.writerows(rows)
                else:
                    self.write_manifest(rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_duplicates_missing_files_and_unsupported_image_bytes(self):
        self.write_manifest(self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.load()
        self.write_manifest()
        first = self.root / self.rows[0]["image_path"]
        first.unlink()
        with self.assertRaises(FileNotFoundError):
            self.load()
        first.write_bytes(b"RIFF-old-webp-file")
        with self.assertRaisesRegex(ValueError, "PNG"):
            self.load()

    def test_rejects_mismatched_image_extension_before_api_calls(self):
        negative = next(row for row in self.rows if row["class"] == "no_obstacles")
        (self.root / negative["image_path"]).write_bytes(PNG)
        with self.assertRaisesRegex(ValueError, "extension does not match"):
            self.load()

    def test_rejects_old_manifest_and_empty_manifest(self):
        self.manifest.write_text("dir,split,verdict,city,label_id\nnocurbramp,test,correct,pittsburgh,101\n")
        with self.assertRaisesRegex(ValueError, "missing columns"):
            self.load()
        self.write_manifest([])
        with self.assertRaisesRegex(ValueError, "No eligible Pittsburgh test images"):
            self.load()

    def test_payload_supports_png_and_webp_without_labels_or_crosswalk(self):
        client = SimpleNamespace(responses=SimpleNamespace(create=Mock()))
        for image in self.load():
            with self.subTest(image_class=image["y"]):
                baseline.request_prediction(client, baseline.DEFAULT_MODEL, Path(image["image_path"]), "high")
                request = client.responses.create.call_args.kwargs
                self.assertEqual(request["model"], "gpt-6.1-sol")
                self.assertEqual(request["reasoning"], {"effort": "low"})
                self.assertFalse(request["store"])
                content = request["input"][0]["content"]
                self.assertEqual(len(content), 2)
                self.assertEqual(content[0], {"type": "input_text", "text": baseline.PROMPT})
                mime, data = ("image/webp", WEBP) if image["y"] == "no_obstacles" else ("image/png", PNG)
                self.assertEqual(content[1]["image_url"], f"data:{mime};base64," + base64.b64encode(data).decode())
                self.assertEqual(content[1]["detail"], "high")
                self.assertEqual(set(request["text"]["format"]["schema"]["properties"]),
                                 {"p_CurbRamp", "p_Obstacle", "p_SurfaceProblem", "p_no_obstacles"})
                self.assertTrue(request["text"]["format"]["strict"])
                serialized = json.dumps(request)
                for private in (image["metadata"]["filename"], image["image_id"], str(self.root), "LOCAL_ONLY_SENTINEL"):
                    self.assertNotIn(private, serialized)

    def test_probability_validation_and_argmax(self):
        for cls in baseline.CLASSES:
            self.assertEqual(baseline.validate_prediction(self.probabilities(cls))["predicted_class"], cls)
        self.assertEqual(baseline.validate_prediction(dict.fromkeys(baseline.PROBABILITY_FIELDS, 0.25))["predicted_class"], "CurbRamp")
        for value in (True, float("nan"), float("inf"), -0.1, 1.1, "0.1"):
            invalid = self.probabilities("CurbRamp")
            invalid["p_Obstacle"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                baseline.validate_prediction(invalid)
        with self.assertRaisesRegex(ValueError, "sum to 1"):
            baseline.validate_prediction(dict.fromkeys(baseline.PROBABILITY_FIELDS, 0.1))
        with self.assertRaises(ValueError):
            baseline.validate_prediction({"p_no_barrier": 1})
        with self.assertRaises(ValueError):
            baseline.validate_prediction({**self.probabilities("no_obstacles"), "p_Crosswalk": 0})

    def test_dry_run_validates_without_api_key_client_or_outputs(self):
        fake_sdk = SimpleNamespace(OpenAI=Mock(side_effect=AssertionError("No API client allowed")))
        # Filtering must precede --limit even when an excluded row appears first.
        self.write_manifest([self.rows[-1], *self.rows[:-1]])
        out = self.root / "unused-output"
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}), patch.dict(sys.modules, {"openai": fake_sdk}):
            result = json.loads(self.invoke("--dry-run", "--limit", "2", "--out", str(out)))
        self.assertEqual(result["total_pittsburgh_images"], 4)
        self.assertEqual(result["total_manifest_images"], 5)
        self.assertEqual(result["excluded_class_counts"], {"Crosswalk": 1})
        self.assertEqual(result["selected_images"], 2)
        self.assertEqual(result["test_class_counts"], dict.fromkeys(baseline.CLASSES, 1))
        self.assertFalse(out.exists())
        fake_sdk.OpenAI.assert_not_called()

    def test_metrics_use_new_class_order_for_incorrect_predictions(self):
        rows = [{"y": cls, "predicted_class": "CurbRamp"} for cls in baseline.CLASSES]
        result = baseline.evaluate_predictions(rows, metrics)
        self.assertEqual(result["pittsburgh_accuracy"], 0.25)
        self.assertEqual(result["pittsburgh_macro_f1"], 0.1)
        self.assertEqual(result["confusion"], {"labels": baseline.CLASSES,
                         "matrix": [[1, 0, 0, 0]] * 4})
        self.assertTrue(all(values["support"] == 1 for values in result["pittsburgh_per_class"].values()))

    def fake_sdk(self, predictions):
        responses = []
        for probabilities in predictions:
            response = SimpleNamespace(status="completed", output_text=json.dumps(probabilities))
            response.model_dump = Mock(return_value={"status": "completed", "usage": {"input_tokens": 1}})
            responses.append(response)
        client = Mock()
        client.responses.create.side_effect = responses
        context = Mock()
        context.__enter__ = Mock(return_value=client)
        context.__exit__ = Mock(return_value=False)
        return SimpleNamespace(__version__="offline-test", OpenAI=Mock(return_value=context)), client

    def test_mocked_full_run_saves_ground_truth_scores_and_four_class_metrics(self):
        sdk, client = self.fake_sdk([self.probabilities(cls) for cls in baseline.CLASSES])
        out = self.root / "results"
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-key"}):
            self.invoke("--out", str(out))
        self.assertEqual(client.responses.create.call_count, 4)
        self.assertEqual(sum(call.kwargs["input"][0]["content"][1]["image_url"].startswith("data:image/webp;")
                             for call in client.responses.create.call_args_list), 1)
        with (out / "pittsburgh_predictions.csv").open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertEqual(reader.fieldnames, baseline.CSV_FIELDS)
            self.assertNotIn("p_Crosswalk", reader.fieldnames)
            rows = list(reader)
        self.assertEqual(len(rows), 4)
        for row, cls in zip(rows, baseline.CLASSES):
            self.assertEqual(row["class"], cls)
            self.assertEqual(row["y"], cls)
            self.assertEqual(row["predicted_class"], cls)
            self.assertEqual(float(row[f"p_{cls}"]), 1.0)
            self.assertTrue((self.root / row["image_path"]).is_file())
        records = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["status"] == "ok" for record in records))
        self.assertTrue(all(record["source_metadata"]["annotation_tag"] == "LOCAL_ONLY_SENTINEL" for record in records))
        metrics = json.loads((out / "results.json").read_text())
        self.assertEqual(metrics["n_test_pittsburgh"], 4)
        self.assertEqual(metrics["total_manifest_images"], 5)
        self.assertEqual(metrics["excluded_classes"], ["Crosswalk"])
        self.assertEqual(metrics["excluded_class_counts"], {"Crosswalk": 1})
        self.assertTrue(metrics["full_pittsburgh_evaluation"])
        self.assertEqual(metrics["openai"]["pittsburgh_accuracy"], 1.0)
        self.assertEqual(metrics["openai"]["pittsburgh_macro_f1"], 1.0)
        self.assertNotIn("pittsburgh_barrier_vs_none", metrics["openai"])
        self.assertEqual(metrics["openai"]["confusion"], {"labels": baseline.CLASSES,
                         "matrix": [[int(i == j) for j in range(4)] for i in range(4)]})
        config = json.loads((out / "run_config.json").read_text())
        self.assertEqual(config["classes"], baseline.CLASSES)
        self.assertEqual(config["evaluation_city"], "pittsburgh")
        self.assertEqual(config["excluded_class_counts"], {"Crosswalk": 1})
        self.assertIn("no_obstacles", config["label_notes"])

    def test_invalid_response_is_logged_without_completed_metrics(self):
        sdk, client = self.fake_sdk([dict.fromkeys(baseline.PROBABILITY_FIELDS, 0.1)])
        out = self.root / "failed-run"
        with patch.dict(sys.modules, {"openai": sdk}), patch.dict("os.environ", {"OPENAI_API_KEY": "offline-test-key"}):
            with self.assertRaisesRegex(RuntimeError, "Stopped on"):
                self.invoke("--out", str(out))
        self.assertEqual(client.responses.create.call_count, 1)
        record = json.loads((out / "predictions.jsonl").read_text())
        self.assertEqual(record["status"], "error")
        self.assertEqual(record["error_type"], "ValueError")
        self.assertFalse((out / "results.json").exists())
        with (out / "pittsburgh_predictions.csv").open(newline="") as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
