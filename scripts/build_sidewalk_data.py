#!/usr/bin/env python3
"""Build the five-class city-held-out dataset using source-directory labels.

Copies and verifies every image before publishing the new output directory.
Existing outputs are never overwritten. No model or API calls are made.
"""

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CLASSES = {
    "nocurbramp": "missing_curb_ramp",
    "curbramp": "curb_ramp",
    "crosswalk": "crosswalk",
    "obstacle": "obstacle",
    "surfaceproblem": "surface_problem",
}
IDENTITY = ("dir", "split", "verdict", "city", "label_id")
FIELDS = [
    "dir", "split", "verdict", "city", "label_id", "y", "source_split",
    "image_path", "source_path", "metadata_source", "sha256",
]


def identity(row):
    return tuple(row[name] for name in IDENTITY)


def read_metadata(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {}
    for row in rows:
        key = identity(row)
        if key in index:
            raise ValueError(f"Duplicate metadata key in {path}: {key}")
        index[key] = row
    return index


def prepare_rows(source_root, metadata_path, features_path):
    metadata = read_metadata(metadata_path)
    features = read_metadata(features_path)
    rows, seen_keys, destinations = [], set(), set()
    for directory, label in CLASSES.items():
        source_dir = source_root / f"sidewalk-validator-ai-dataset-{directory}"
        paths = sorted(source_dir.glob("*/*/*.webp"))
        if not paths:
            raise ValueError(f"No images found in {source_dir}")
        for source in paths:
            source_split, verdict, filename = source.relative_to(source_dir).parts
            city, label_id = source.stem.rsplit("_", 1)
            if source_split not in {"train", "val", "test"} or verdict not in {"correct", "incorrect"}:
                raise ValueError(f"Unexpected source path: {source}")
            original = dict(dir=directory, split=source_split, verdict=verdict,
                            city=city, label_id=label_id)
            key = identity(original)
            feature = features[key]
            if feature["ok"] != "True" or feature["relpath"] != str(source.relative_to(source_dir)):
                raise ValueError(f"Invalid image metadata: {source}")
            if len(feature["sha256"]) != 64:
                raise ValueError(f"Missing source checksum: {source}")
            if key in metadata:
                provenance = metadata_path
                original = {name: metadata[key][name] for name in IDENTITY}
            else:
                if directory not in {"curbramp", "crosswalk"}:
                    raise ValueError(f"Image missing from baseline metadata: {source}")
                provenance = features_path
                original = {name: feature[name] for name in IDENTITY}
            split = "test" if city == "pittsburgh" else "train"
            relative = Path(split) / label / filename
            if str(relative) in destinations:
                raise ValueError(f"Destination filename collision: {relative}")
            destinations.add(str(relative))
            seen_keys.add(key)
            rows.append({
                **original, "split": split, "source_split": source_split, "y": label,
                "image_path": str(relative), "source_path": str(source.resolve()),
                "metadata_source": str(provenance.resolve()), "sha256": feature["sha256"],
            })
    if seen_keys != set(features) or not set(metadata).issubset(seen_keys):
        raise ValueError("Source image inventory and metadata coverage differ.")
    return sorted(rows, key=lambda row: row["image_path"])


def copy_verified(row, staging):
    source, destination = Path(row["source_path"]), staging / row["image_path"]
    shutil.copy2(source, destination)
    checksum = hashlib.sha256()
    with destination.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != row["sha256"]:
        raise ValueError(f"Copied image does not match its source checksum: {source}")
    return destination.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("/data/datasets"))
    parser.add_argument("--metadata", type=Path, default=Path("/data/eda/baseline/crops_meta.csv"))
    parser.add_argument("--features", type=Path, default=Path("/data/eda/crops/tables/image_features.csv.gz"))
    parser.add_argument("--out", type=Path, default=Path("/data/datasets/sidewalk-data"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"Output already exists; refusing to overwrite {args.out}")
    if args.workers < 1:
        parser.error("--workers must be positive")
    rows = prepare_rows(args.source_root, args.metadata, args.features)
    total_bytes = sum(Path(row["source_path"]).stat().st_size for row in rows)
    if shutil.disk_usage(args.out.parent).free < total_bytes * 1.1:
        parser.error("Insufficient disk space for independent image copies")
    staging = Path(tempfile.mkdtemp(prefix=".sidewalk-data-build-", dir=args.out.parent))
    staging.chmod(0o755)  # Keep the shared dataset readable after the atomic rename.
    print(f"Copying {len(rows)} images ({total_bytes:,} bytes) into {staging}", flush=True)
    for split in ["train", "test"]:
        for label in CLASSES.values():
            (staging / split / label).mkdir(parents=True)
    copied_bytes = 0
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(copy_verified, row, staging) for row in rows]
            for count, future in enumerate(as_completed(futures), 1):
                copied_bytes += future.result()
                if count % 1000 == 0 or count == len(rows):
                    print(f"Copied and SHA-256 verified {count}/{len(rows)}", flush=True)
        counts = {}
        for split in ["train", "test"]:
            selected = [row for row in rows if row["split"] == split]
            csv_path = staging / f"{split}.csv"
            with csv_path.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(selected)
            with csv_path.open(newline="", encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
            if written != selected:
                raise ValueError(f"CSV round-trip verification failed: {split}")
            if any((row["city"] == "pittsburgh") != (split == "test") for row in written):
                raise ValueError("Pittsburgh split verification failed")
            expected = {row["image_path"] for row in written}
            actual = {str(p.relative_to(staging)) for p in (staging / split).glob("*/*.webp")}
            if expected != actual:
                raise ValueError(f"Image/CSV correspondence failed: {split}")
            counts[split] = dict(Counter(row["y"] for row in written))
        by_hash = defaultdict(list)
        for row in rows:
            by_hash[row["sha256"]].append(row)
        cross_label = [group for group in by_hash.values() if len({r["y"] for r in group}) > 1]
        cross_split = [group for group in by_hash.values() if len({r["split"] for r in group}) > 1]
        if args.out.exists():
            raise FileExistsError(f"Output appeared during the build: {args.out}")
        staging.rename(args.out)
        print(json.dumps({
            "output": str(args.out), "images": len(rows), "verified_bytes": copied_bytes,
            "counts": counts, "cross_label_duplicate_hash_groups": len(cross_label),
            "train_test_duplicate_hash_groups": len(cross_split),
        }, indent=2), flush=True)
    except Exception:
        print(f"Build failed; unpublished files retained at {staging}", flush=True)
        raise


if __name__ == "__main__":
    main()
