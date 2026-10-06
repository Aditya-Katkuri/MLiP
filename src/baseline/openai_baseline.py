#!/usr/bin/env python3
"""Zero-shot OpenAI baseline for the Pittsburgh-only sidewalk-data test split.

Requires the openai and scikit-learn packages and OPENAI_API_KEY in the environment.
In the sidewalk environment:
    python src/baseline/openai_baseline.py --dry-run
    python src/baseline/openai_baseline.py \
        --out /data/eda/openai_baseline/gpt61_sol_low_pittsburgh_v2

Defaults: gpt-6.1-sol, low reasoning, high image detail. Images and ground truth
come from /data/datasets/sidewalk-data/test.csv. Labels are the CSV's class
values, checked against test/<class>/<filename> paths. Crosswalk rows are skipped
before opening their images. Only CurbRamp, Obstacle, SurfaceProblem, and
no_obstacles are evaluated. Train/val and non-Pittsburgh inputs are rejected.
PNG and WebP images are supported. no_obstacles is a proxy label from rejected
annotations; it does not certify the absence of every accessibility barrier.

Each image has exactly one target class. The model returns four probability
estimates, which must sum to one. These are elicited probabilities, not token
logprobs or calibrated classifier scores. The script takes their argmax for
predicted_class; ties follow CLASSES order.

Only the prompt and image bytes are sent, in independent requests. Paths,
filenames, labels, coordinates, tags, and other images' predictions stay local.

Each new output directory contains:
- run_config.json: prompt, schema, model/settings, and manifest fingerprint.
- predictions.jsonl: image identity, prediction, and full response with usage.
- pittsburgh_predictions.csv: filename, image_path (relative to data root), split,
  class, city, label_id, y (same ground truth as class), predicted_class,
  p_CurbRamp, p_Obstacle, p_SurfaceProblem, p_no_obstacles.
- results.json: four-class accuracy, macro-F1, per-class metrics, and confusion
  matrix, after completion. Excluded classes and image counts are recorded.

Each successful prediction is flushed to disk. An API error, refusal, incomplete
response, or invalid probability vector stops the run after saving the error.
Partial runs have no results.json. Use a new output directory for every run.
--dry-run validates eligible images and reports counts without an API
key, API requests, or output files. --limit selects a pilot after this validation.
total_pittsburgh_images and full_pittsburgh_evaluation refer to the eligible
four-class roster; total_manifest_images also counts excluded Crosswalk rows.

Official OpenAI documentation:
https://developers.openai.com/api/docs/models/gpt-6.1-sol
https://developers.openai.com/api/docs/guides/images-vision
https://developers.openai.com/api/docs/guides/structured-outputs
"""

import argparse
import base64
import csv
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "low"
DEFAULT_DATA_ROOT = Path("/data/datasets/sidewalk-data")
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "test.csv"
MANIFEST_FIELDS = ("filename", "image_path", "split", "class")
IDENTITY_FIELDS = (*MANIFEST_FIELDS, "city", "label_id")
CLASSES = ["CurbRamp", "Obstacle", "SurfaceProblem", "no_obstacles"]
EXCLUDED_CLASSES = ["Crosswalk"]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_MIME_TYPES = {".png": "image/png", ".webp": "image/webp"}
PROBABILITY_FIELDS = [f"p_{cls}" for cls in CLASSES]
CSV_FIELDS = [*IDENTITY_FIELDS, "y", "predicted_class", *PROBABILITY_FIELDS]

PROMPT = """Classify the main sidewalk-related feature in the attached street-level image.
For this task, each image belongs to exactly one of these four mutually exclusive
classes:

- CurbRamp: A lowered or sloping curb transition connecting a sidewalk to the
  roadway, commonly at a corner or pedestrian crossing. The ramp belongs to this
  class whether it is in good condition or has accessibility defects.
- Obstacle: An object that blocks or restricts pedestrian passage, such as a pole,
  sign, trash can, vegetation, street furniture, or a parked vehicle on the path.
- SurfaceProblem: A rough, uneven, cracked, broken, or otherwise difficult-to-traverse
  walking surface, for example holes, raised slabs, roots, or grass growing through
  pavement. Focus on the walking surface rather than an object placed on it.
- no_obstacles: The pedestrian path appears clear, without a visible object
  obstruction or surface problem, and a curb ramp is not the main feature shown.

Judge only the visible scene. If multiple features are visible, choose the class
that best describes the main feature in the image. When a curb ramp is the main
feature, choose CurbRamp even if it is usable and the path is otherwise clear.
If the view is ambiguous, express uncertainty across the four classes; still
treat them as alternatives for a single target label, not independent attributes.

Return only a JSON object with these four keys:
p_CurbRamp, p_Obstacle, p_SurfaceProblem, p_no_obstacles.
Each value is your estimated probability that the image's one target label is
that class, between 0 and 1 inclusive. Use at most four decimal places and ensure
the four values sum to 1. Do not return a class name, explanation, or extra keys.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        field: {"type": "number", "minimum": 0, "maximum": 1}
        for field in PROBABILITY_FIELDS
    },
    "required": PROBABILITY_FIELDS,
    "additionalProperties": False,
}


def image_mime_type(header: bytes) -> str:
    """Identify supported image bytes instead of assuming every file is a PNG."""
    if header.startswith(PNG_SIGNATURE):
        return "image/png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("Expected PNG or WebP image bytes.")


def load_pittsburgh_images(
    manifest: Path, data_root: Path, *, excluded_counts: Counter | None = None,
) -> list[dict]:
    """Load the four eligible classes; do not open excluded Crosswalk images."""
    data_root = data_root.resolve()
    images = []
    seen = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) in (None, "") for key in MANIFEST_FIELDS):
                raise ValueError(f"Malformed manifest row {line_number}.")
            if row["split"] != "test":
                raise ValueError(f"Only the test split is accepted: row {line_number}.")
            target = row["class"]
            if target in EXCLUDED_CLASSES:
                if excluded_counts is not None:
                    excluded_counts[target] += 1
                continue
            if target not in CLASSES:
                raise ValueError(f"Unknown class: {target!r}")
            filename = row["filename"]
            suffix = Path(filename).suffix.lower()
            parts = Path(filename).stem.split("-")
            if (
                suffix not in IMAGE_MIME_TYPES
                or len(parts) != 4
                or parts[0] != "gsv"
                or parts[1] != "pittsburgh"
                or not parts[2].isdigit()
                or parts[3] != target
            ):
                raise ValueError(f"Expected a Pittsburgh PNG/WebP matching class {target}: {filename}")
            image_key = f"test/{target}/{filename}"
            if row["image_path"] != image_key:
                raise ValueError(f"Image path/class mismatch: {row['image_path']!r}")
            if "y" in row and row["y"] != target:
                raise ValueError(f"Ground-truth class/y mismatch: {filename}")
            path = (data_root / image_key).resolve()
            if not path.is_relative_to(data_root / "test" / target):
                raise ValueError(f"Image resolves outside its test class directory: {image_key}")
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open("rb") as image_file:
                if image_mime_type(image_file.read(12)) != IMAGE_MIME_TYPES[suffix]:
                    raise ValueError(f"Image extension does not match its bytes: {path}")
            if image_key in seen:
                raise ValueError(f"Duplicate image in manifest: {image_key}")
            seen.add(image_key)
            images.append({
                "image_id": image_key,
                "image_path": str(path),
                "metadata": {
                    **{key: row[key] for key in MANIFEST_FIELDS},
                    "city": "pittsburgh",
                    "label_id": parts[2],
                },
                "source_metadata": row,
                "y": target,
            })
    if not images:
        raise ValueError(f"No eligible Pittsburgh test images found in {manifest}")
    return sorted(images, key=lambda image: image["image_id"])


def request_prediction(client, model: str, image_path: Path, detail: str):
    """Send the image anonymously, without filenames or annotation metadata."""
    image_bytes = image_path.read_bytes()
    mime_type = image_mime_type(image_bytes[:12])
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return client.responses.create(
        model=model,
        reasoning={"effort": REASONING_EFFORT},
        store=False,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPT},
                {
                    "type": "input_image",
                    "image_url": f"data:{mime_type};base64,{encoded}",
                    "detail": detail,
                },
            ],
        }],
        text={
            "format": {
                "type": "json_schema",
                "name": "pittsburgh_prediction",
                "strict": True,
                "schema": OUTPUT_SCHEMA,
            }
        },
    )


def validate_prediction(prediction: dict) -> dict:
    """Reject invalid scores and derive one label without changing probabilities."""
    if not isinstance(prediction, dict) or set(prediction) != set(PROBABILITY_FIELDS):
        raise ValueError("Prediction must contain exactly the four probability fields.")
    for field in PROBABILITY_FIELDS:
        value = prediction[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"Invalid probability for {field}: {value!r}")
    total = math.fsum(prediction.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Probabilities must sum to 1; received {total}.")
    predicted_class = max(CLASSES, key=lambda cls: prediction[f"p_{cls}"])
    return {"predicted_class": predicted_class, **prediction}


def evaluate_predictions(rows: list[dict], metrics) -> dict:
    """Evaluate the four mutually exclusive dataset classes in CLASSES order."""
    targets = [row["y"] for row in rows]
    predictions = [row["predicted_class"] for row in rows]
    report = metrics.classification_report(
        targets, predictions, labels=CLASSES, output_dict=True, zero_division=0
    )
    return {
        "pittsburgh_per_class": {
            cls: {
                **{
                    key: round(report[cls][key], 3)
                    for key in ["precision", "recall", "f1-score"]
                },
                "support": int(report[cls]["support"]),
            }
            for cls in CLASSES
        },
        "pittsburgh_accuracy": round(metrics.accuracy_score(targets, predictions), 3),
        "pittsburgh_macro_f1": round(report["macro avg"]["f1-score"], 3),
        "confusion": {
            "labels": CLASSES,
            "matrix": metrics.confusion_matrix(
                targets, predictions, labels=CLASSES
            ).tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenAI model ID (default: {DEFAULT_MODEL}); reasoning is always low.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, help="New output directory; required unless --dry-run.")
    parser.add_argument("--detail", choices=["auto", "low", "high"], default="high")
    parser.add_argument("--limit", type=int, help="Optional image limit for a pilot.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs without API calls or output files.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run is used.")

    excluded_counts = Counter()
    images = load_pittsburgh_images(
        args.manifest, args.data_root, excluded_counts=excluded_counts,
    )
    total_pittsburgh = len(images)
    selection = {
        "evaluation_scope": "Pittsburgh test images in CLASSES, excluding Crosswalk",
        "total_manifest_images": total_pittsburgh + sum(excluded_counts.values()),
        "total_pittsburgh_images": total_pittsburgh,
        "excluded_classes": EXCLUDED_CLASSES,
        "excluded_class_counts": dict(excluded_counts),
    }
    all_class_counts = dict(Counter(image["y"] for image in images))
    if args.limit is not None:
        images = images[:args.limit]
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "model": args.model,
            "reasoning_effort": REASONING_EFFORT,
            "manifest": str(args.manifest.resolve()),
            "data_root": str(args.data_root.resolve()),
            "classes": CLASSES,
            "probability_fields": PROBABILITY_FIELDS,
            **selection,
            "test_class_counts": all_class_counts,
            "selected_images": len(images),
            "selected_class_counts": dict(Counter(image["y"] for image in images)),
        }, indent=2))
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        parser.error("Set the OPENAI_API_KEY environment variable.")

    # Check both dependencies before any paid requests.
    try:
        import openai
        from sklearn import metrics
    except ImportError as exc:
        parser.error(
            f"Missing dependency: {exc}. In the sidewalk environment, run "
            "'python -m pip install openai scikit-learn'."
        )

    # Preserve earlier results: every invocation must use a new output directory.
    args.out.mkdir(parents=True, exist_ok=False)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_requested": args.model,
        "reasoning_effort": REASONING_EFFORT,
        "openai_sdk_version": openai.__version__,
        "classes": CLASSES,
        "dataset": "sidewalk-data",
        "dataset_sources": {
            "CurbRamp/Obstacle/SurfaceProblem": "projectsidewalk/sidewalk-tagger-ai-validated",
            "no_obstacles": "sidewalk-do-not-use.tar.gz: Pittsburgh incorrect annotations",
        },
        "label_notes": {
            "no_obstacles": "Proxy negatives from rejected designated annotations, not verified absence of every barrier.",
        },
        "evaluation_split": "test",
        "evaluation_city": "pittsburgh",
        "ground_truth_source": "test.csv class column; checked against class folder and filename",
        "prediction_rule": "argmax; ties use classes order",
        "probability_sum_tolerance": 1e-6,
        "prompt": PROMPT,
        "output_schema": OUTPUT_SCHEMA,
        "image_detail": args.detail,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "data_root": str(args.data_root.resolve()),
        **selection,
        "image_count": len(images),
        "test_class_counts": all_class_counts,
        "limit": args.limit,
    }
    (args.out / "run_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    output_path = args.out / "predictions.jsonl"
    csv_path = args.out / "pittsburgh_predictions.csv"
    rows = []
    print(
        f"Evaluating {len(images)}/{total_pittsburgh} Pittsburgh images with "
        f"{args.model}, reasoning={REASONING_EFFORT}, detail={args.detail}. "
        f"Excluded classes: {dict(excluded_counts)}.",
        flush=True,
    )
    with (
        openai.OpenAI(api_key=api_key, timeout=120.0) as client,
        output_path.open("x", encoding="utf-8") as output,
        csv_path.open("x", newline="", encoding="utf-8") as csv_output,
    ):
        writer = csv.DictWriter(csv_output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_output.flush()
        for index, image in enumerate(images, start=1):
            record = dict(image)
            record["prediction"] = None
            try:
                response = request_prediction(
                    client, args.model, Path(image["image_path"]), args.detail
                )
                record["response"] = response.model_dump(mode="json")
                if response.status != "completed":
                    raise RuntimeError(f"Response status: {response.status}")
                if not response.output_text:
                    raise RuntimeError("No prediction text; inspect the saved response.")
                prediction = validate_prediction(json.loads(response.output_text))
                record["prediction"] = prediction
                row = {**image["metadata"], "y": image["y"], **prediction}
                writer.writerow(row)
                csv_output.flush()
                rows.append(row)
                record["status"] = "ok"
            except Exception as exc:
                record["status"] = "error"
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)
                raise RuntimeError(
                    f"Stopped on {image['image_id']}; details saved in {output_path}"
                ) from exc
            finally:
                output.write(json.dumps(record) + "\n")
                output.flush()
            print(
                f"[{index}/{len(images)}] {image['image_id']}: "
                f"{prediction['predicted_class']}",
                flush=True,
            )

    results = {
        "model": args.model,
        "reasoning_effort": REASONING_EFFORT,
        "classes": CLASSES,
        "n_test_pittsburgh": len(rows),
        **selection,
        "full_pittsburgh_evaluation": len(rows) == total_pittsburgh,
        "test_class_counts": dict(Counter(row["y"] for row in rows)),
        "openai": evaluate_predictions(rows, metrics),
    }
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
