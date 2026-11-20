#!/usr/bin/env python3
"""Five-class zero-shot OpenAI baseline for a Pittsburgh sidewalk test split.

Requires the openai and scikit-learn packages and OPENAI_API_KEY in the environment.
In the sidewalk environment:
    python src/baseline/openai_baseline.py --dry-run
    python src/baseline/openai_baseline.py \
        --out /data/eda/openai_baseline/gpt61_sol_low_naive_merged_nulls_5class

Defaults: gpt-6.1-sol, low reasoning, high image detail. Images and ground truth
come from the supplied manifest and data root. Labels are the CSV's class
values, checked against <split>/<class>/<filename> paths. Crosswalk, CurbRamp,
Obstacle, SurfaceProblem, and no_obstacles are all evaluated; no class is excluded.
Train/val and non-Pittsburgh inputs are rejected.
PNG and WebP images are supported, including processed images whose extension
differs from the original filename column. Manually collected no_obstacles
images are accepted in pitt-test when their source_class records that provenance.
The original legacy test layout remains supported. New defaults select the
224x224 naive-merged-nulls dataset, which combines reviewed archive negatives
and the manually collected no-obstacle screenshots.

Each image has exactly one target class. The model returns five probability
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
  p_Crosswalk, p_CurbRamp, p_Obstacle, p_SurfaceProblem, p_no_obstacles.
- results.json: five-class accuracy, macro-F1, per-class metrics, and confusion
  matrix, after completion. Excluded classes and image counts are recorded.

Each successful prediction is flushed to disk. An API error, refusal, incomplete
response, or invalid probability vector stops the run after saving the error.
Partial runs have no results.json. Use a new output directory for every run;
--resume reuses its successful JSONL records after checking the full input and
experiment fingerprints. Input CSV/config, image hashes, and runner source are
snapshotted with each run. Resume repairs a partially written final JSONL line.
--dry-run validates eligible images and reports counts without an API
key, API requests, or output files. --limit selects a pilot after this validation.
total_pittsburgh_images and full_pittsburgh_evaluation refer to the eligible
five-class roster, including all Crosswalk rows.

Official OpenAI documentation:
https://developers.openai.com/api/docs/models/gpt-6.1-sol
https://developers.openai.com/api/docs/guides/images-vision
https://developers.openai.com/api/docs/guides/structured-outputs
"""

import argparse
import base64
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "low"
DEFAULT_DATA_ROOT = Path("/data/datasets/proc-sidewalk-data-naive-merged-nulls")
DEFAULT_MANIFEST = DEFAULT_DATA_ROOT / "pitt-test.csv"
MANIFEST_FIELDS = ("filename", "image_path", "split", "class")
IDENTITY_FIELDS = (*MANIFEST_FIELDS, "city", "label_id")
CLASSES = ["Crosswalk", "CurbRamp", "Obstacle", "SurfaceProblem", "no_obstacles"]
EXCLUDED_CLASSES = []
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
IMAGE_MIME_TYPES = {".png": "image/png", ".webp": "image/webp"}
PROBABILITY_FIELDS = [f"p_{cls}" for cls in CLASSES]
CSV_FIELDS = [*IDENTITY_FIELDS, "y", "predicted_class", *PROBABILITY_FIELDS]

PROMPT = """Classify the main sidewalk-related feature in the attached street-level image.
For this task, each image belongs to exactly one of these five mutually exclusive
classes:

- Crosswalk: A designated pedestrian crossing across a roadway, often indicated
  by painted stripes or parallel lines. Choose this class when the crossing itself
  is the main feature shown, including crossings with faded or damaged markings.
- CurbRamp: A lowered or sloping curb transition connecting a sidewalk to the
  roadway, commonly at a corner or pedestrian crossing. The ramp belongs to this
  class whether it is in good condition or has accessibility defects.
- Obstacle: An object that blocks or restricts pedestrian passage, such as a pole,
  sign, trash can, vegetation, street furniture, or a parked vehicle on the path.
- SurfaceProblem: A rough, uneven, cracked, broken, or otherwise difficult-to-traverse
  walking surface, for example holes, raised slabs, roots, or grass growing through
  pavement. Focus on the walking surface rather than an object placed on it.
- no_obstacles: The pedestrian path appears clear, without a visible object
  obstruction or surface problem, and neither a curb ramp nor a crosswalk is the
  main feature shown.

Judge only the visible scene. If multiple features are visible, choose the class
that best describes the main feature in the image. When a curb ramp is the main
feature, choose CurbRamp even if it is usable and the path is otherwise clear.
When the pedestrian roadway crossing is the main feature, choose Crosswalk;
a nearby curb ramp alone does not make the crossing the main feature.
If the view is ambiguous, express uncertainty across the five classes; still
treat them as alternatives for a single target label, not independent attributes.

Return only a JSON object with these five keys:
p_Crosswalk, p_CurbRamp, p_Obstacle, p_SurfaceProblem, p_no_obstacles.
Each value is your estimated probability that the image's one target label is
that class, between 0 and 1 inclusive. Use at most four decimal places and ensure
the five values sum to 1. Do not return a class name, explanation, or extra keys.
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
    """Load and validate all five classes, including Crosswalk images."""
    data_root = data_root.resolve()
    images = []
    seen = set()
    manifest_split = None
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if None in row or any(row.get(key) in (None, "") for key in MANIFEST_FIELDS):
                raise ValueError(f"Malformed manifest row {line_number}.")
            split = row["split"]
            if split not in {"test", "pitt-test"}:
                raise ValueError(f"Only test or pitt-test is accepted: row {line_number}.")
            if manifest_split is not None and manifest_split != split:
                raise ValueError("Mixed splits in test manifest.")
            manifest_split = split
            target = row["class"]
            if target in EXCLUDED_CLASSES:
                if excluded_counts is not None:
                    excluded_counts[target] += 1
                continue
            if target not in CLASSES:
                raise ValueError(f"Unknown class: {target!r}")
            filename = row["filename"]
            if Path(filename).name != filename:
                raise ValueError(f"Invalid filename: {filename}")
            relative = Path(row["image_path"])
            if relative.is_absolute() or relative.parts[:2] != (split, target) or len(relative.parts) != 3:
                raise ValueError(f"Image path/class mismatch: {row['image_path']!r}")
            if relative.name != filename and not (
                row.get("source_image_path")
                and Path(row["source_image_path"]).name == filename
                and relative.stem == Path(filename).stem
            ):
                raise ValueError(f"Processed filename/source mismatch: {filename}")
            suffix = relative.suffix.lower()
            parts = Path(filename).stem.split("-")
            is_gsv = (len(parts) == 4 and parts[0] == "gsv"
                      and parts[1] == "pittsburgh" and parts[2].isdigit()
                      and parts[3] == target)
            is_manual = (split == "pitt-test" and target == "no_obstacles"
                         and row.get("source_class") == "no_obstacles_manual_collection"
                         and not filename.startswith("gsv-"))
            if suffix not in IMAGE_MIME_TYPES or not (is_gsv or is_manual):
                raise ValueError(f"Expected a Pittsburgh PNG/WebP matching class {target}: {filename}")
            if row.get("city", "pittsburgh").lower() != "pittsburgh":
                raise ValueError(f"Non-Pittsburgh city in manifest: {filename}")
            image_key = relative.as_posix()
            if "y" in row and row["y"] != target:
                raise ValueError(f"Ground-truth class/y mismatch: {filename}")
            path = (data_root / image_key).resolve()
            if not path.is_relative_to(data_root / split / target):
                raise ValueError(f"Image resolves outside its test class directory: {image_key}")
            if not path.is_file():
                raise FileNotFoundError(path)
            image_bytes = path.read_bytes()
            if image_mime_type(image_bytes[:12]) != IMAGE_MIME_TYPES[suffix]:
                raise ValueError(f"Image extension does not match its bytes: {path}")
            if image_key in seen:
                raise ValueError(f"Duplicate image in manifest: {image_key}")
            seen.add(image_key)
            images.append({
                "image_id": image_key,
                "image_path": str(path),
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "metadata": {
                    **{key: row[key] for key in MANIFEST_FIELDS},
                    "filename": relative.name,
                    "city": "pittsburgh",
                    "label_id": parts[2] if is_gsv else "",
                },
                "source_metadata": row,
                "y": target,
            })
    if not images:
        raise ValueError(f"No eligible Pittsburgh test images found in {manifest}")
    disk_images = {
        str(path.relative_to(data_root))
        for cls in CLASSES
        for path in (data_root / manifest_split / cls).glob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_MIME_TYPES
    }
    if disk_images != seen:
        raise ValueError(f"Manifest/directory image mismatch: {sorted(disk_images ^ seen)[:5]}")
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
        raise ValueError("Prediction must contain exactly the five probability fields.")
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
    """Evaluate the five mutually exclusive dataset classes in CLASSES order."""
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


def saved_predictions(output_dir: Path, images: list[dict]) -> list[dict]:
    """Recover successful records from the durable JSONL journal for a resume."""
    path = output_dir / "predictions.jsonl"
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    roster = {image["image_id"]: image for image in images}
    completed = {}
    offset = 0
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt JSONL record {index + 1}; inspect {path}")
            (output_dir / ("truncated-jsonl-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".txt")).write_bytes(line)
            with path.open("r+b") as handle:
                handle.truncate(offset)
            break
        offset += len(line)
        if record.get("status") != "ok":
            continue
        image_id = record["image_id"]
        if image_id not in roster or image_id in completed:
            raise ValueError(f"Unknown or duplicate successful image: {image_id}")
        image = roster[image_id]
        if record.get("image_sha256") != image["image_sha256"] or record["y"] != image["y"]:
            raise ValueError(f"Saved prediction/input mismatch: {image_id}")
        prediction = validate_prediction({field: record["prediction"][field] for field in PROBABILITY_FIELDS})
        if prediction != record["prediction"]:
            raise ValueError(f"Saved prediction/argmax mismatch: {image_id}")
        completed[image_id] = {**image["metadata"], "y": image["y"], **prediction}
    # A complete last JSON object without its final newline is also recoverable.
    if path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
    return [completed[image["image_id"]] for image in images if image["image_id"] in completed]


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
    parser.add_argument("--resume", action="store_true", help="Resume the same output directory, skipping all saved successes.")
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
        "evaluation_scope": "All Pittsburgh test images in the five dataset classes, including Crosswalk",
        "total_manifest_images": total_pittsburgh + sum(excluded_counts.values()),
        "total_pittsburgh_images": total_pittsburgh,
        "excluded_classes": EXCLUDED_CLASSES,
        "excluded_class_counts": dict(excluded_counts),
    }
    all_class_counts = dict(Counter(image["y"] for image in images))
    if args.limit is not None:
        images = images[:args.limit]
    evaluation_split = images[0]["metadata"]["split"]
    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "model": args.model,
            "reasoning_effort": REASONING_EFFORT,
            "manifest": str(args.manifest.resolve()),
            "data_root": str(args.data_root.resolve()),
            "classes": CLASSES,
            "task_version": "pittsburgh-five-class-v1",
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

    # Explicit resume is the only way to reuse a run directory.
    if args.resume:
        if not (args.out / "run_config.json").is_file():
            parser.error("--resume requires an existing run_config.json.")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / ".run.lock").open("a") as run_lock:
        try:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another process is already using this output directory.")
        inventory = [{key: image[key] for key in ("image_id", "image_sha256", "y")} for image in images]
        preprocessing_path = args.data_root / "config.yaml"
        source_class_counts = dict(Counter(image["source_metadata"].get("source_class", image["y"]) for image in images))
        config = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_requested": args.model,
            "reasoning_effort": REASONING_EFFORT,
            "openai_sdk_version": openai.__version__,
            "classes": CLASSES,
            "dataset": args.data_root.resolve().name,
            "dataset_sources": {
                "Crosswalk/CurbRamp/Obstacle/SurfaceProblem": "projectsidewalk/sidewalk-tagger-ai-validated",
                "no_obstacles": (
                    "Reviewed archive negatives and manual screenshots; source_class_counts records the merged origins."
                    if "no_obstacles_manual_collection" in source_class_counts else
                    "Archive negatives; see supplied dataset documentation for review history."
                ),
            },
            "label_notes": {
                "no_obstacles": "Use the manifest's current class assignment; original source metadata is retained locally.",
            },
            "evaluation_split": evaluation_split,
            "evaluation_city": "pittsburgh",
            "ground_truth_source": f"{args.manifest.name} class column; checked against the split and class folders",
            "prediction_rule": "argmax; ties use classes order",
            "probability_sum_tolerance": 1e-6,
            "prompt": PROMPT,
            "output_schema": OUTPUT_SCHEMA,
            "image_detail": args.detail,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "data_root": str(args.data_root.resolve()),
            "image_inventory_sha256": hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "preprocessing_config_sha256": hashlib.sha256(preprocessing_path.read_bytes()).hexdigest() if preprocessing_path.is_file() else None,
            "source_class_counts": source_class_counts,
            "python_version": sys.version,
            **selection,
            "image_count": len(images),
            "test_class_counts": all_class_counts,
            "limit": args.limit,
        }
        if args.resume:
            saved_config = json.loads((args.out / "run_config.json").read_text())
            differences = [key for key in config if key != "created_at" and config[key] != saved_config.get(key)]
            if differences:
                parser.error(f"Resume settings or inputs changed: {differences}")
        else:
            (args.out / "run_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            (args.out / "image_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
            shutil.copy2(args.manifest, args.out / "input_manifest.csv")
            shutil.copy2(Path(__file__), args.out / "openai_baseline.py")
            if preprocessing_path.is_file():
                shutil.copy2(preprocessing_path, args.out / "dataset_config.yaml")
        output_path = args.out / "predictions.jsonl"
        csv_path = args.out / "pittsburgh_predictions.csv"
        rows = saved_predictions(args.out, images) if args.resume else []
        completed_ids = {row["image_path"] for row in rows}
        pending = [image for image in images if image["image_id"] not in completed_ids]
        # CSV is a derived view of the journal. Rebuilding repairs a crash between writes.
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_csv.replace(csv_path)
        print(
            f"Evaluating {len(images)}/{total_pittsburgh} Pittsburgh images with "
            f"{args.model}, reasoning={REASONING_EFFORT}, detail={args.detail}. "
            f"Excluded classes: {dict(excluded_counts)}. "
            f"Already saved: {len(rows)}; remaining: {len(pending)}.",
            flush=True,
        )
        with (
            openai.OpenAI(api_key=api_key, timeout=120.0) as client,
            output_path.open("a" if args.resume else "x", encoding="utf-8") as output,
            csv_path.open("a", newline="", encoding="utf-8") as csv_output,
        ):
            writer = csv.DictWriter(csv_output, fieldnames=CSV_FIELDS)
            for index, image in enumerate(pending, start=len(rows) + 1):
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
                    os.fsync(output.fileno())
                writer.writerow(row)
                csv_output.flush()
                os.fsync(csv_output.fileno())
                rows.append(row)
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
