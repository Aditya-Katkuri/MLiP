#!/usr/bin/env python3
"""Read-only review dashboard for the sidewalk-data set and its CLIP baselines.

One local page with two tabs:

  Baselines         CLIP zero-shot and a CLIP linear probe on the city-split sidewalk-data set. The classes
                    come from results.json (currently CurbRamp, Obstacle and SurfaceProblem in every split,
                    plus the test-only proxy class no_obstacles; dataset classes outside them are not shown
                    here). The probe is trained on the known classes only and answers no_obstacles when its
                    top probability falls below tau. The page shows a comparison table (macro F1, balanced
                    accuracy, accuracy, the no_obstacles AUROC on test, and precision and recall for each
                    labelled class, with a majority-class reference), a Test (Pittsburgh) / Val (Columbus +
                    Amsterdam) toggle, and for the selected model and split: stat cards, a detailed metrics
                    panel (per-class precision, recall, F1, support and predicted counts; macro and weighted
                    averages), a clickable confusion matrix, filters (including whether the two models
                    agree), a paginated gallery and CSV export.
  Dataset explorer  Every image of every class, filtered by class, split, city, tag and CLIP outcome, with
                    counts by class, split and city, the dataset's caveat for each noted class, and a note for
                    classes that are not part of the CLIP baselines.

Both tabs open the same lightbox: the image at its own size and aspect ratio with the labeller's point and
the window CLIP saw (each toggleable, both correct when zoomed; classes without a label point and images CLIP
saw whole say so), both models' class probabilities (the probe's with tau marked), tags, provenance and
metadata.

Run from the repository with the project Python. It needs only the standard library, no API key and
no network:

    python src/baseline/vlm_dashboard.py --port 8765

Then forward port 8765 (VS Code Remote SSH, Ports panel) and open the forwarded URL.

Inputs, all read-only:
  --explorer-dir  /data/eda/dashboard/explorer     manifest.json (items with w/h, optional source and
                                                   class_notes), thumbs/<id>.webp, view/<id>.webp
  --clip-dir      /data/eda/clip_sidewalk_data     predictions.csv, results.json

predictions.csv rows are joined to manifest items on `file`. results.json `classes` must be a subset of the
manifest's classes, and every manifest item of those classes needs exactly one row (no other item may appear);
class, split, city and label ID must agree. Probabilities must be valid and agree with each prediction,
and the probe's reject rule (lp_max < tau) must hold on every row. Each split's confusion matrix and metrics
are recomputed from the CSV and must match every field results.json reports (accuracy, balanced accuracy,
macro precision/recall/F1, weighted F1, per-class precision/recall/F1/support, confusion matrices), for every
model and split and for the majority-class blocks, as must the split counts. The page receives only zero-shot, the probe with its reject rule
and the majority reference. Macro averages and balanced accuracy are over the classes that have images in
the split; weighted F1 weights each class by its support. A no_obstacles AUROC that cannot be reproduced is logged and shown as saved.
Either source may be missing when the server starts (its tab then says so). Both are reloaded when their
files change. If the CLIP files exist at startup but fail these checks, the server refuses to start.
A failure after a later reload is logged and shown on the page.

Endpoints: /  /health  /api/explorer  /api/clip  /image/thumb/<id>  /image/view/<id>
Images are served only by integer manifest id, never by a client-supplied path. The server binds to
127.0.0.1, sends a strict Content-Security-Policy, writes nothing and calls no model.
"""

import argparse
import csv
import gzip
import json
import math
import os
import re
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_EXPLORER = Path("/data/eda/dashboard/explorer")
DEFAULT_CLIP = Path("/data/eda/clip_sidewalk_data")
SPLITS = ("train", "val", "test")
DIGITS = re.compile(r"[0-9]{1,9}")
CITY = re.compile(r"[a-z0-9_\-]{1,64}")
CSP = ("default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; "
       "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def stamp(*paths):
    """(mtime_ns, size) of each file, or None for a missing one: cheap change detection."""
    out = []
    for path in paths:
        try:
            st = path.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except OSError:
            out.append(None)
    return tuple(out)


# --------------------------------------------------------------------------- manifest

def validate_manifest(raw):
    """Check the explorer manifest and return (items, warnings). Raises ValueError on anything unsafe."""
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        raise ValueError("manifest has no items list")
    classes = raw.get("classes")
    if not isinstance(classes, list) or not classes or not all(isinstance(c, str) for c in classes):
        raise ValueError("manifest classes must be a list of names")
    tags_by_class = raw.get("tags_by_class") or {}
    if not isinstance(tags_by_class, dict) or not all(
            isinstance(v, list) and all(isinstance(t, str) for t in v) for v in tags_by_class.values()):
        raise ValueError("tags_by_class must map classes to lists of tags")
    class_notes = raw.get("class_notes") or {}
    if not isinstance(class_notes, dict) or not all(k in classes and isinstance(v, str) and len(v) <= 4000 for k, v in class_notes.items()):
        raise ValueError("class_notes must map classes to text")
    default_size = raw.get("image_size")
    if not (isinstance(default_size, list) and len(default_size) == 2 and all(isinstance(v, int) and 0 < v <= 20000 for v in default_size)):
        default_size = None
    split_cities = raw.get("split_cities") or {}
    if not isinstance(split_cities, dict) or not all(
            k in SPLITS and isinstance(v, list) and all(isinstance(c, str) for c in v) for k, v in split_cities.items()):
        raise ValueError("split_cities must map train/val/test to lists of cities")
    items, ids, files, warnings, unknown_tags, misplaced = [], set(), set(), [], 0, 0
    for position, item in enumerate(raw["items"]):
        if not isinstance(item, dict):
            raise ValueError(f"manifest item {position} is not an object")
        item_id = item.get("id")
        if isinstance(item_id, bool) or not isinstance(item_id, int) or not 0 <= item_id < 10**9:
            raise ValueError(f"manifest item {position} has an invalid id {item_id!r}")
        if item_id in ids:
            raise ValueError(f"manifest id {item_id} is duplicated")
        ids.add(item_id)
        if item.get("class") not in classes or item.get("split") not in SPLITS:
            raise ValueError(f"manifest item {item_id} has an unknown class or split")
        city, label_id, file = item.get("city"), item.get("label_id"), item.get("file")
        if not isinstance(city, str) or not CITY.fullmatch(city):
            raise ValueError(f"manifest item {item_id} has an invalid city {city!r}")
        if isinstance(label_id, bool) or not (isinstance(label_id, int) or (isinstance(label_id, str) and DIGITS.fullmatch(label_id))):
            raise ValueError(f"manifest item {item_id} has an invalid label_id {label_id!r}")
        if not isinstance(file, str) or not file or len(file) > 300 or file in files:
            raise ValueError(f"manifest item {item_id} has a missing or duplicated file")
        files.add(file)
        point = []
        for axis in ("x", "y"):
            value = item.get(axis)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"manifest item {item_id} has an invalid {axis} {value!r}")
            point.append(value)
        tags = item.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) and len(t) <= 64 for t in tags):
            raise ValueError(f"manifest item {item_id} has invalid tags")
        unknown_tags += sum(t not in tags_by_class.get(item["class"], []) for t in tags)
        if split_cities:
            misplaced += city not in split_cities.get(item["split"], [])
        size = [item.get("w"), item.get("h")]
        if size == [None, None] and default_size:
            size = list(default_size)
        if not all(isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 20000 for v in size):
            raise ValueError(f"manifest item {item_id} has no valid w/h and there is no image_size to fall back on")
        entry = {"id": item_id, "class": item["class"], "split": item["split"], "city": city, "label_id": str(label_id),
                 "x": point[0], "y": point[1], "tags": tags, "file": file, "w": size[0], "h": size[1]}
        source = item.get("source")
        if source is not None:
            if not isinstance(source, dict):
                raise ValueError(f"manifest item {item_id} has an invalid source")
            entry["source"] = {k: str(source[k])[:120] for k in ("dataset", "split", "verdict", "label_id") if source.get(k) is not None}
        if isinstance(item.get("sources_count"), int) and not isinstance(item.get("sources_count"), bool):
            entry["sources_count"] = item["sources_count"]
        items.append(entry)
    if unknown_tags:
        warnings.append(f"{unknown_tags} tags are not listed in tags_by_class for their class")
    if misplaced:
        warnings.append(f"{misplaced} items are in a split whose split_cities list does not include their city")
    return items, warnings


# --------------------------------------------------------------------------- CLIP results

def metrics(classes, truths, predictions):
    """Accuracy; balanced accuracy, macro precision, macro recall and macro F1 (means over the classes that
    have images in this set); weighted F1 (F1 weighted by support); per-class precision/recall/F1/support and
    the confusion matrix in `classes` order."""
    n = len(classes)
    index = {c: i for i, c in enumerate(classes)}
    matrix = [[0] * n for _ in range(n)]
    for t, p in zip(truths, predictions):
        matrix[index[t]][index[p]] += 1
    per_class = {}
    for i, c in enumerate(classes):
        tp, predicted, support = matrix[i][i], sum(row[i] for row in matrix), sum(matrix[i])
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "predicted": predicted}
    present = [c for c in classes if per_class[c]["support"]]
    total = len(truths)
    return {
        "n": total, "classes": present,
        "accuracy": sum(matrix[i][i] for i in range(n)) / total if total else 0.0,
        "balanced_accuracy": sum(per_class[c]["recall"] for c in present) / len(present) if present else 0.0,
        "macro_precision": sum(per_class[c]["precision"] for c in present) / len(present) if present else 0.0,
        "macro_recall": sum(per_class[c]["recall"] for c in present) / len(present) if present else 0.0,
        "macro_f1": sum(per_class[c]["f1"] for c in present) / len(present) if present else 0.0,
        "weighted_f1": sum(per_class[c]["f1"] * per_class[c]["support"] for c in present) / total if total else 0.0,
        "per_class": per_class,
        "matrix": matrix,
    }


def auroc(scores, positives):
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    if not n_pos or not n_neg:
        return None
    return (sum(r for r, pos in zip(ranks, positives) if pos) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def check_metrics(name, classes, ours, saved):
    """Fail if results.json disagrees with the metrics recomputed from predictions.csv. Every field
    results.json provides is compared; the confusion matrix and per-class block are optional."""
    if "confusion" in saved:
        labels = saved["confusion"]["labels"]
        if not set(labels) <= set(classes):
            raise ValueError(f"{name}: confusion labels {labels} are not all classes {classes}")
        index = [classes.index(c) for c in labels]
        if [[ours["matrix"][i][j] for j in index] for i in index] != saved["confusion"]["matrix"]:
            raise ValueError(f"{name}: confusion matrix in results.json does not match predictions.csv")
    if "n" in saved and saved["n"] != ours["n"]:
        raise ValueError(f"{name}: n is {ours['n']} in predictions.csv but {saved['n']} in results.json")
    summary = ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1")
    pairs = [(key, ours[key], saved[key]) for key in summary if key in saved]
    if not pairs:
        raise ValueError(f"{name}: no summary metrics to check")
    for c, values in saved.get("per_class", {}).items():
        if c not in ours["per_class"]:
            raise ValueError(f"{name}: unknown class {c!r} in per_class")
        for key in ("precision", "recall", "f1"):
            if key in values:
                pairs.append((f"{c} {key}", ours["per_class"][c][key], values[key]))
        if "support" in values and values["support"] != ours["per_class"][c]["support"]:
            raise ValueError(f"{name}: {c} support differs from predictions.csv")
    for label, mine, theirs in pairs:
        if abs(mine - theirs) > 0.0011:
            raise ValueError(f"{name}: {label} is {mine:.4f} from predictions.csv but {theirs} in results.json")


def number(row, key, where):
    try:
        value = float(row[key])
    except ValueError as error:
        raise ValueError(f"{where}: unreadable {key}") from error
    if not math.isfinite(value):
        raise ValueError(f"{where}: {key} is not finite")
    return value


def probabilities(row, prefix, prediction, classes, where):
    p = [number(row, prefix + c, where) for c in classes]
    if not all(0 <= v <= 1 for v in p):
        raise ValueError(f"{where}: {prefix}* outside [0, 1]")
    if abs(sum(p) - 1) > 2e-3:
        raise ValueError(f"{where}: {prefix}* sums to {sum(p):.5f}")
    if p[classes.index(prediction)] < max(p) - 1e-4:
        raise ValueError(f"{where}: {prediction} is not the most probable of {prefix}*")
    return p


def load_clip(clip_dir, items):
    """Validate predictions.csv + results.json against the manifest; return (payload dict, row count, warnings).

    Zero-shot scores every class. The linear probe scores only `known_classes`; with a reject rule in
    results.json (probe.reject.tau) its final prediction is the one class it never saw whenever its top
    probability lp_max falls below tau; otherwise it is the argmax of lp_p_*."""
    results = json.loads((clip_dir / "results.json").read_text())
    classes = results.get("classes")
    manifest_classes = sorted({item["class"] for item in items})
    if not isinstance(classes, list) or not classes or not set(classes) <= set(manifest_classes):
        raise ValueError(f"results.json classes {classes} are not a subset of the manifest classes {manifest_classes}")
    expected = {item["file"] for item in items if item["class"] in classes}
    known = results.get("known_classes") or classes
    if not set(known) <= set(classes):
        raise ValueError(f"known_classes {known} are not all in classes {classes}")
    reject = (results.get("probe") or {}).get("reject")
    unknown = [c for c in classes if c not in known]
    if reject is not None:
        if len(unknown) != 1 or not isinstance(reject.get("tau"), (int, float)):
            raise ValueError("probe.reject needs a numeric tau and exactly one class outside known_classes")
        tau = float(reject["tau"])
    elif unknown:
        raise ValueError(f"classes {unknown} are outside known_classes but probe.reject is missing")
    by_file = {item["file"]: item for item in items}
    required = ["file", "class", "split", "city", "label_id", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
                "zero_shot", *("zs_p_" + c for c in classes), "linear_probe", *("lp_p_" + c for c in known)]
    if reject is not None:
        required += ["lp_max"]
    rows, seen = [], set()
    truth = {s: [] for s in SPLITS}
    preds = {(m, s): [] for m in ("zero_shot", "linear_probe") for s in SPLITS}
    scores = {"zero_shot": [], "lp_max": [], "test_truth": []}
    with (clip_dir / "predictions.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"predictions.csv is missing columns {sorted(missing)}")
        for line, row in enumerate(reader, start=2):
            where = f"predictions.csv line {line}"
            item = by_file.get(row["file"])
            if item is None:
                raise ValueError(f"{where}: file {row['file']!r} is not in the explorer manifest")
            if row["file"] in seen:
                raise ValueError(f"{where}: file {row['file']!r} appears twice")
            if item["class"] not in classes:
                raise ValueError(f"{where}: {row['file']!r} is a {item['class']} image, a class results.json does not list")
            seen.add(row["file"])
            for key in ("class", "split", "city", "label_id"):
                if row[key] != item[key]:
                    raise ValueError(f"{where}: {key} is {row[key]!r} but the manifest says {item[key]!r}")
            box = [number(row, k, where) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")]
            if not all(-1e-6 <= v <= 1 + 1e-6 for v in box) or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"{where}: crop box {box} is not inside the image")
            zs = row["zero_shot"]
            lp = row["linear_probe"]
            lp3 = lp  # the probe's argmax over the known classes; ties within 1e-4 accept the saved answer
            if reject is not None:
                raw = [number(row, "lp_p_" + c, where) for c in known]
                top = [c for c, v in zip(known, raw) if v >= max(raw) - 1e-4]
                lp3 = lp if lp in top else top[0]
            if zs not in classes or lp not in classes or lp3 not in known:
                raise ValueError(f"{where}: unknown predicted class")
            zs_p = probabilities(row, "zs_p_", zs, classes, where)
            lp_p = probabilities(row, "lp_p_", lp3, known, where)
            lp_max = number(row, "lp_max", where) if reject is not None else max(lp_p)
            if abs(lp_max - max(lp_p)) > 1e-3:
                raise ValueError(f"{where}: lp_max {lp_max} is not the largest lp_p_* ({max(lp_p)})")
            if reject is not None:
                if lp_max < tau - 1e-4 and lp != unknown[0]:
                    raise ValueError(f"{where}: lp_max {lp_max} < tau {tau} but linear_probe is {lp!r}, not {unknown[0]!r}")
                if lp_max > tau + 1e-4 and lp != lp3:
                    raise ValueError(f"{where}: lp_max {lp_max} >= tau {tau} but linear_probe {lp!r} differs from the argmax {lp3!r}")
                if lp not in (lp3, unknown[0]):
                    raise ValueError(f"{where}: linear_probe {lp!r} is neither the argmax nor {unknown[0]!r}")
            split = item["split"]
            truth[split].append(item["class"])
            for key, value in (("zero_shot", zs), ("linear_probe", lp)):
                preds[(key, split)].append(value)
            if split == "test" and unknown:
                scores["zero_shot"].append(zs_p[classes.index(unknown[0])])
                scores["lp_max"].append(lp_max)
                scores["test_truth"].append(item["class"])
            rows.append([item["id"], *(round(v, 5) for v in box), classes.index(zs), *zs_p,
                         classes.index(lp), *lp_p, round(lp_max, 5)])
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(f"predictions.csv covers {len(seen)} of the manifest's {len(expected)} images of "
                         f"{', '.join(classes)}; missing e.g. {missing[:3]}")

    model_keys = ["zero_shot", "linear_probe"]
    computed = {s: {m: metrics(classes, truth[s], preds[(m, s)]) for m in model_keys} for s in SPLITS if truth[s]}
    mb = results.get("majority_baseline") or {}
    majority = mb.get("class") or mb.get("majority_class") or mb.get("predicted_class")
    if majority not in classes:
        raise ValueError(f"majority_baseline class {majority!r} is not a class")
    for split in computed:
        computed[split]["majority"] = metrics(classes, truth[split], [majority] * len(truth[split]))
    for split in SPLITS:
        block = next((b for b in (mb.get(split), (mb.get("metrics") or {}).get(split), mb.get(f"{split}_metrics"))
                      if isinstance(b, dict) and "accuracy" in b), None)
        if block is not None:
            if split not in computed:
                raise ValueError(f"majority_baseline has metrics for split {split!r}, which has no rows")
            check_metrics(f"majority {split}", classes, computed[split]["majority"], block)
    aurocs, warnings = {}, []
    if "test" in computed and unknown:
        positives = [c == unknown[0] for c in scores["test_truth"]]
        aurocs = {"zero_shot": auroc(scores["zero_shot"], positives), "linear_probe": auroc([-v for v in scores["lp_max"]], positives)}
    saved_models = results.get("models") or {}
    for model, by_split in saved_models.items():
        if model not in model_keys:
            raise ValueError(f"results.json has metrics for an unknown model {model!r}")
        for split, saved in by_split.items():
            if split not in computed:
                raise ValueError(f"results.json has {model} metrics for split {split!r}, which has no rows")
            check_metrics(f"{model} {split}", classes, computed[split][model], saved)
            if saved.get("no_obstacles_auroc") is not None:
                mine = aurocs.get(model) if split == "test" else None
                if mine is None or abs(mine - saved["no_obstacles_auroc"]) > 0.0011:
                    warnings.append(f"{model} {split}: could not reproduce no_obstacles_auroc {saved['no_obstacles_auroc']} "
                                    f"(got {mine if mine is None else round(mine, 4)}); showing the saved value")
                    aurocs[model] = saved["no_obstacles_auroc"]
    for split in computed:
        for key in ("accuracy", "macro_f1"):
            if f"{split}_{key}" in mb and abs(computed[split]["majority"][key] - mb[f"{split}_{key}"]) > 0.0011:
                raise ValueError(f"majority_baseline {split}_{key} does not match predictions.csv")
    for split, info in (results.get("splits") or {}).items():
        if split not in SPLITS:
            raise ValueError(f"results.json has an unknown split {split!r}")
        counts = {c: truth[split].count(c) for c in classes}
        saved_counts = info.get("class_counts") or {}
        cities = sorted({by_file[f]["city"] for f in seen if by_file[f]["split"] == split})
        if info.get("n") != len(truth[split]) or any(saved_counts.get(c, 0) != counts[c] for c in classes) \
                or set(saved_counts) - set(classes):
            raise ValueError(f"results.json split {split}: n/class_counts differ from predictions.csv ({len(truth[split])}, {counts})")
        if sorted(info.get("cities", [])) != cities:
            raise ValueError(f"results.json split {split}: cities {info.get('cities')} differ from predictions.csv {cities}")
    shown = ("zero_shot", "linear_probe")
    public = results
    fields = ["id", "crop_x0", "crop_y0", "crop_x1", "crop_y1", "zero_shot", *("zs_p_" + c for c in classes),
              "linear_probe", *("lp_p_" + c for c in known), "lp_max"]
    payload = {"available": True, "results": public, "classes": classes, "known_classes": known,
               "reject": ({"class": unknown[0], **reject} if reject is not None else None),
               "majority_class": majority,
               "models": list(shown), "metrics": {s: {m: v for m, v in by.items() if m in (*shown, "majority")} for s, by in computed.items()},
               "auroc": {m: v for m, v in aurocs.items() if m in shown},
               "row_fields": fields, "rows": rows, "warnings": warnings}
    return payload, len(rows), warnings


# --------------------------------------------------------------------------- store

class Store:
    """The manifest and the CLIP output, each reloaded when its files change."""

    def __init__(self, explorer_dir, clip_dir):
        self.root = explorer_dir.resolve()
        self.clip_dir = clip_dir.resolve()
        self.manifest_path = self.root / "manifest.json"
        self.clip_paths = (self.clip_dir / "predictions.csv", self.clip_dir / "results.json")
        self.lock = threading.Lock()
        self.manifest_stamp = self.clip_stamp = "unset"
        self.items, self.ids, self.meta, self.items_json = [], frozenset(), None, b"[]"
        self.manifest_error = "Explorer data has not been loaded yet."
        self.clip_json, self.clip_rows, self.clip_error = None, 0, "CLIP results have not been loaded yet."

    def refresh(self):
        manifest_stamp, clip_stamp = stamp(self.manifest_path), stamp(*self.clip_paths)
        if manifest_stamp == self.manifest_stamp and (clip_stamp, manifest_stamp) == self.clip_stamp:
            return
        with self.lock:
            if manifest_stamp != self.manifest_stamp:
                self.manifest_stamp = manifest_stamp
                self._load_manifest()
            if (clip_stamp, manifest_stamp) != self.clip_stamp:
                self.clip_stamp = (clip_stamp, manifest_stamp)
                self._load_clip()

    def _load_manifest(self):
        self.items, self.ids, self.meta, self.items_json = [], frozenset(), None, b"[]"
        if not self.manifest_path.exists():
            self.manifest_error = f"{self.manifest_path} does not exist yet."
            return
        try:
            raw = json.loads(self.manifest_path.read_text())
            items, warnings = validate_manifest(raw)
            size = raw.get("image_size")
            meta = {
                "generated_at": raw.get("generated_at"), "source_root": raw.get("source_root"),
                "about": raw.get("about") if isinstance(raw.get("about"), str) else "",
                "classes": raw["classes"], "excluded": raw.get("excluded") if isinstance(raw.get("excluded"), dict) else {},
                "tags_by_class": raw.get("tags_by_class") or {}, "split_cities": raw.get("split_cities") or {},
                "class_notes": raw.get("class_notes") or {},
                "image_size": size if isinstance(size, list) and len(size) == 2 else None, "warnings": warnings,
            }
            items_json = json.dumps(items, allow_nan=False, separators=(",", ":")).encode()
        except (OSError, ValueError, TypeError) as error:  # JSONDecodeError is a ValueError
            self.manifest_error = f"The explorer manifest could not be used: {error}"
            print(self.manifest_error, flush=True)
            return
        self.items, self.ids, self.meta, self.items_json = items, frozenset(i["id"] for i in items), meta, items_json
        self.manifest_error = None
        for warning in warnings:
            print(f"Explorer manifest warning: {warning}", flush=True)
        print(f"Explorer manifest loaded: {len(items)} items", flush=True)

    def _load_clip(self):
        self.clip_json, self.clip_rows = None, 0
        missing = [p.name for p in self.clip_paths if not p.exists()]
        if missing:
            self.clip_error = f"Waiting for {', '.join(missing)} in {self.clip_dir}."
            return
        if self.meta is None:
            self.clip_error = "CLIP results need the explorer manifest, which is not loaded."
            return
        try:
            payload, count, warnings = load_clip(self.clip_dir, self.items)
            self.clip_json = json.dumps(payload, allow_nan=False, separators=(",", ":")).encode()
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.clip_error = f"The CLIP results in {self.clip_dir} failed validation: {error}"
            print(self.clip_error, flush=True)
            return
        self.clip_rows, self.clip_error = count, None
        for warning in warnings:
            print(f"CLIP warning: {warning}", flush=True)
        print(f"CLIP results loaded and verified: {count} predictions from {self.clip_dir}", flush=True)

    def explorer_payload(self):
        self.refresh()
        with self.lock:
            meta, items_json, error = self.meta, self.items_json, self.manifest_error
        if meta is None:
            return json.dumps({"available": False, "reason": error}).encode()
        head = {"available": True, **meta, "count": len(self.ids), "ready": self.ready_counts()}
        return json.dumps(head, allow_nan=False)[:-1].encode() + b',"items":' + items_json + b"}"

    def clip_payload(self):
        self.refresh()
        with self.lock:
            body, error = self.clip_json, self.clip_error
        return body if body is not None else json.dumps({"available": False, "reason": error}).encode()

    def ready_counts(self):
        counts = {}
        for sub in ("thumbs", "view"):
            try:
                with os.scandir(self.root / sub) as entries:
                    counts[sub] = sum(1 for e in entries if e.name.endswith(".webp"))
            except OSError:
                counts[sub] = 0
        return counts

    def image_path(self, kind, item_id):
        self.refresh()
        if item_id not in self.ids:
            return None
        path = (self.root / ("thumbs" if kind == "thumb" else "view") / f"{item_id}.webp").resolve()
        return path if path.is_relative_to(self.root) else None


# --------------------------------------------------------------------------- HTTP

class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, html, store, **kwargs):
        self.html, self.store = html, store
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.serve()

    def do_HEAD(self):
        self.serve(head=True)

    def serve(self, head=False):
        store, path = self.store, urlsplit(self.path).path
        if path == "/":
            return self.send(self.html, "text/html; charset=utf-8", "no-cache", head)
        if path == "/health":
            store.refresh()
            body = json.dumps({"status": "ok", "explorer_items": len(store.ids) if store.meta else None,
                               "clip_rows": store.clip_rows or None, "explorer_error": store.manifest_error,
                               "clip_error": store.clip_error}).encode()
            return self.send(body, "application/json", "no-store", head)
        if path == "/api/explorer":
            return self.send(store.explorer_payload(), "application/json; charset=utf-8", "no-store", head)
        if path == "/api/clip":
            return self.send(store.clip_payload(), "application/json; charset=utf-8", "no-store", head)
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return None
        parts = path.split("/")
        if len(parts) == 4 and parts[1] == "image" and parts[2] in {"thumb", "view"} and DIGITS.fullmatch(parts[3]):
            return self.send_image(store.image_path(parts[2], int(parts[3])), head)
        self.send_error(404, "Not found")
        return None

    def send_image(self, image, head):
        if image is None:
            self.send_error(404, "Unknown image")
            return None
        try:
            body = image.read_bytes()
        except OSError:
            self.send_error(404, "Image is not available on this server yet")
            return None
        return self.send(body, "image/webp", "private, max-age=3600", head, compress=False)

    def send(self, body, content_type, cache, head, compress=True):
        encoding = None
        if compress and len(body) > 1024 and "gzip" in self.headers.get("Accept-Encoding", ""):
            body, encoding = gzip.compress(body, compresslevel=5), "gzip"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        if compress:
            self.send_header("Vary", "Accept-Encoding")
        if encoding:
            self.send_header("Content-Encoding", encoding)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
        return None

    def log_message(self, fmt, *args):
        # Keep the log useful: skip successful requests, keep errors and 404s.
        if len(args) >= 2 and str(args[1]) in {"200", "204"}:
            return
        super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--explorer-dir", type=Path, default=DEFAULT_EXPLORER)
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    store = Store(args.explorer_dir, args.clip_dir)
    store.refresh()
    if store.clip_error and all(p.exists() for p in store.clip_paths):
        sys.exit(f"Refusing to start: {store.clip_error}")
    html = Path(__file__).with_suffix(".html").read_bytes()
    ThreadingHTTPServer.daemon_threads = True
    with ThreadingHTTPServer((args.host, args.port), partial(DashboardHandler, html=html, store=store)) as server:
        print(f"Review dashboard: http://{args.host}:{args.port}", flush=True)
        print(f"Explorer: {len(store.ids)} items" if store.meta else f"Explorer: {store.manifest_error}", flush=True)
        print(f"CLIP: {store.clip_rows} predictions" if store.clip_json else f"CLIP: {store.clip_error}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
