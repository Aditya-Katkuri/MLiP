#!/usr/bin/env python3
"""Zero-shot OpenAI baseline for the held-out Pittsburgh crops.

Requires the openai and scikit-learn packages and OPENAI_API_KEY in the environment.
In the sidewalk environment:
    python src/baseline/openai_baseline.py --out /data/eda/openai_baseline/run_001

Defaults: gpt-6.1-sol, low reasoning, high image detail. All Pittsburgh rows in
the existing baseline's roster are evaluated, across original train/val/test
folders. Ground truth follows clip_baseline.py: a correct annotation maps to
its barrier class; an incorrect annotation maps to no_barrier.

Each image has exactly one target class. The model returns four probability
estimates, which must sum to one. These are elicited probabilities, not token
logprobs or calibrated classifier scores. The script takes their argmax for
predicted_class; ties follow CLASSES order, matching the CLIP zero-shot baseline.

Only the prompt and image bytes are sent, in independent requests. Paths,
verdicts, labels, and other images' predictions stay local.

Each new output directory contains:
- run_config.json: prompt, schema, model/settings, and manifest fingerprint.
- predictions.jsonl: image identity, prediction, and full response with usage.
- pittsburgh_predictions.csv: dir, split, verdict, city, label_id, y,
  predicted_class, p_missing_curb_ramp, p_obstacle, p_surface_problem, p_no_barrier.
- results.json: Pittsburgh metrics matching the CLIP baseline, after completion.

Each successful prediction is flushed to disk. An API error, refusal, incomplete
response, or invalid probability vector stops the run after saving the error.
Partial runs have no results.json. Use a new output directory for every run.

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
DEFAULT_MANIFEST = Path("/data/eda/baseline/crops_meta.csv")
DEFAULT_DATA_ROOT = Path("/data/datasets")
IDENTITY_FIELDS = ("dir", "split", "verdict", "city", "label_id")
CLASSES = ["missing_curb_ramp", "obstacle", "surface_problem", "no_barrier"]
DIR_TO_CLASS = {
    "nocurbramp": "missing_curb_ramp",
    "obstacle": "obstacle",
    "surfaceproblem": "surface_problem",
}
PROBABILITY_FIELDS = [f"p_{cls}" for cls in CLASSES]
CSV_FIELDS = [*IDENTITY_FIELDS, "y", "predicted_class", *PROBABILITY_FIELDS]

PROMPT = """Classify the attached street-level image crop for sidewalk accessibility.
For this task, each image belongs to exactly one of these four mutually exclusive
classes:

- missing_curb_ramp: A sidewalk corner or pedestrian crossing has a raised curb
  without a curb ramp providing a wheelchair-accessible transition to the street.
  A curb elsewhere along a street is not by itself evidence of a missing ramp.
- obstacle: An object, such as a pole, sign, trash can, vegetation, or parked
  vehicle, blocks or substantially obstructs the sidewalk or pedestrian path.
- surface_problem: The sidewalk walking surface is cracked, broken, uneven, or
  otherwise damaged, for example by bumps, roots, or grass growing through it.
- no_barrier: None of the three barrier types above is evident; for example,
  the sidewalk is clear and usable, or the corner has a usable curb ramp.

Judge only the visible scene. If multiple features are visible, choose the class
that best describes the main sidewalk accessibility condition in this crop.
If the view is ambiguous, express uncertainty across the four classes; still
treat them as alternatives for a single target label, not independent attributes.

Return only a JSON object with these four keys:
p_missing_curb_ramp, p_obstacle, p_surface_problem, p_no_barrier.
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


def load_pittsburgh_images(manifest: Path, data_root: Path) -> list[dict]:
    """Use the CLIP roster and target mapping; never send targets to the API."""
    images = []
    seen = set()
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(IDENTITY_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for row in reader:
            if row["city"] != "pittsburgh":
                continue
            if row["dir"] not in DIR_TO_CLASS:
                raise ValueError(f"Unknown crop dataset: {row['dir']}")
            if row["verdict"] not in {"correct", "incorrect"}:
                raise ValueError(f"Unknown annotation verdict: {row['verdict']}")
            target = (
                DIR_TO_CLASS[row["dir"]]
                if row["verdict"] == "correct" else "no_barrier"
            )
            if "y" in row and row["y"] != target:
                raise ValueError(f"Target mismatch for label_id={row['label_id']}")
            path = (
                data_root
                / f"sidewalk-validator-ai-dataset-{row['dir']}"
                / row["split"]
                / row["verdict"]
                / f"{row['city']}_{row['label_id']}.webp"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            image_key = str(path.relative_to(data_root))
            if image_key in seen:
                raise ValueError(f"Duplicate image in manifest: {image_key}")
            seen.add(image_key)
            images.append({
                "image_id": image_key,
                "image_path": str(path),
                "metadata": {key: row[key] for key in IDENTITY_FIELDS},
                "y": target,
            })
    if not images:
        raise ValueError(f"No Pittsburgh images found in {manifest}")
    return sorted(images, key=lambda image: image["image_id"])


def request_prediction(client, model: str, image_path: Path, detail: str):
    """Send the image anonymously, without filenames or annotation metadata."""
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
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
                    "image_url": f"data:image/webp;base64,{encoded}",
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
    """Match clip_baseline.py's Pittsburgh metrics and confusion-matrix order."""
    targets = [row["y"] for row in rows]
    predictions = [row["predicted_class"] for row in rows]
    report = metrics.classification_report(
        targets, predictions, labels=CLASSES, output_dict=True, zero_division=0
    )
    true_barriers = [label != "no_barrier" for label in targets]
    predicted_barriers = [label != "no_barrier" for label in predictions]
    tp = sum(a and b for a, b in zip(true_barriers, predicted_barriers))
    return {
        "pittsburgh_per_class": {
            cls: {
                key: round(report[cls][key], 3)
                for key in ["precision", "recall", "f1-score", "support"]
            }
            for cls in CLASSES
        },
        "pittsburgh_accuracy": round(metrics.accuracy_score(targets, predictions), 3),
        "pittsburgh_macro_f1": round(report["macro avg"]["f1-score"], 3),
        "pittsburgh_barrier_vs_none": {
            "precision": round(tp / max(sum(predicted_barriers), 1), 3),
            "recall": round(tp / max(sum(true_barriers), 1), 3),
        },
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
    parser.add_argument("--out", type=Path, required=True, help="New output directory.")
    parser.add_argument("--detail", choices=["auto", "low", "high"], default="high")
    parser.add_argument("--limit", type=int, help="Optional image limit for a pilot.")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
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

    images = load_pittsburgh_images(args.manifest, args.data_root)
    total_pittsburgh = len(images)
    if args.limit is not None:
        images = images[:args.limit]

    # Preserve earlier results: every invocation must use a new output directory.
    args.out.mkdir(parents=True, exist_ok=False)
    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_requested": args.model,
        "reasoning_effort": REASONING_EFFORT,
        "openai_sdk_version": openai.__version__,
        "classes": CLASSES,
        "prediction_rule": "argmax; ties use classes order",
        "probability_sum_tolerance": 1e-6,
        "prompt": PROMPT,
        "output_schema": OUTPUT_SCHEMA,
        "image_detail": args.detail,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "data_root": str(args.data_root.resolve()),
        "total_pittsburgh_images": total_pittsburgh,
        "image_count": len(images),
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
        f"{args.model}, reasoning={REASONING_EFFORT}, detail={args.detail}.",
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
        "n_test_pittsburgh": len(rows),
        "total_pittsburgh_images": total_pittsburgh,
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
