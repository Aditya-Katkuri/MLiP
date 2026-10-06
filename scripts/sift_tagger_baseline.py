#!/usr/bin/env python3
"""SIFT bag-of-words nearest-prototype baseline on sidewalk-tagger-ai-validated.

Same method as scripts/sift_baseline.py: grayscale SIFT plus a dense grid,
a 200-word BOWKMeansTrainer codebook, mean L1 histograms, nearest prototype
by symmetric chi-square. No SVM and no neural net.

Split: the dataset's own train and test columns. Val is counted and then
left unused. The surrogate class no_obstacles is counted and left out of
the four-way fit and the four-way metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter
from multiprocessing import get_context

import cv2
import numpy as np

SCORED = ("Crosswalk", "CurbRamp", "Obstacle", "SurfaceProblem")
GRID_STEP = 32
CAP_SPARSE = 40
CAP_DENSE = 40
DESC_CAP = CAP_SPARSE + CAP_DENSE
MIN_DESCRIPTORS = 5
N_WORDS = 200
TRAIN_CAP = 500
CODEBOOK_PER_CLASS = 200

# Published table this run is checked against. A mismatch is reported; the
# CSV columns are still the split that is used.
EXPECTED = {
    "train": {
        "Crosswalk": 1446,
        "CurbRamp": 8324,
        "Obstacle": 2073,
        "SurfaceProblem": 6983,
    },
    "val": {
        "Crosswalk": 124,
        "CurbRamp": 1772,
        "Obstacle": 300,
        "SurfaceProblem": 1500,
    },
    "test": {
        "Crosswalk": 57,
        "CurbRamp": 761,
        "Obstacle": 59,
        "SurfaceProblem": 609,
        "no_obstacles": 204,
    },
}
EXPECTED_N = {"train": 18826, "val": 3696, "test": 1690}


def city_of(filename: str, label: str) -> str | None:
    stem = filename.rsplit(".", 1)[0]
    suffix = "-" + label
    if not stem.startswith("gsv-") or not stem.endswith(suffix):
        return None
    body = stem[4 : -len(suffix)]
    city, id_part = body.rsplit("-", 1)
    if not city or not id_part.isdigit():
        return None
    return city


def load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "class" not in rows[0] or "split" not in rows[0] or "image_path" not in rows[0]:
        raise SystemExit(f"{path} is missing class, split, or image_path")
    return rows


def disk_counts(data_dir: str) -> dict[str, dict[str, int]]:
    out = {}
    for split in ("train", "val", "test"):
        root = os.path.join(data_dir, split)
        counts = {}
        if os.path.isdir(root):
            for name in sorted(os.listdir(root)):
                folder = os.path.join(root, name)
                if os.path.isdir(folder):
                    counts[name] = sum(
                        1
                        for fn in os.listdir(folder)
                        if os.path.isfile(os.path.join(folder, fn))
                    )
        out[split] = counts
    return out


def audit(data_dir: str) -> dict:
    splits = {}
    for split in ("train", "val", "test"):
        rows = load_csv(os.path.join(data_dir, f"{split}.csv"))
        by_class = dict(Counter(r["class"] for r in rows))
        bad_split_col = sum(1 for r in rows if r["split"] != split)
        missing = 0
        path_mismatch = 0
        cities = Counter()
        unparsed = 0
        for r in rows:
            path = os.path.join(data_dir, r["image_path"])
            if not os.path.isfile(path) or os.path.getsize(path) == 0:
                missing += 1
            parts = r["image_path"].split("/")
            if len(parts) != 3 or parts[0] != split or parts[1] != r["class"]:
                path_mismatch += 1
            city = city_of(r["filename"], r["class"])
            if city is None:
                unparsed += 1
            else:
                cities[city] += 1
        expected = EXPECTED[split]
        class_match = by_class == expected and len(rows) == EXPECTED_N[split]
        splits[split] = {
            "n": len(rows),
            "expected_n": EXPECTED_N[split],
            "by_class": by_class,
            "expected_by_class": expected,
            "class_counts_match_table": class_match,
            "bad_split_column": bad_split_col,
            "missing_files": missing,
            "path_mismatch": path_mismatch,
            "unparsed_city": unparsed,
            "cities": dict(cities.most_common()),
            "rows": rows,
        }
    disk = disk_counts(data_dir)
    names = {
        split: {r["filename"] for r in splits[split]["rows"]}
        for split in splits
    }
    overlap = {
        "train_val": len(names["train"] & names["val"]),
        "train_test": len(names["train"] & names["test"]),
        "val_test": len(names["val"] & names["test"]),
    }
    four_match = all(splits[s]["class_counts_match_table"] for s in splits)
    disk_match = all(disk[s] == splits[s]["by_class"] for s in splits)
    return {
        "splits": splits,
        "disk": disk,
        "filename_overlap": overlap,
        "table_match": four_match and disk_match and all(v == 0 for v in overlap.values()),
        "disk_match": disk_match,
    }


def select_items(audit_result: dict, data_dir: str, train_cap: int, codebook_n: int) -> dict:
    train_rows = [r for r in audit_result["splits"]["train"]["rows"] if r["class"] in SCORED]
    test_rows = [r for r in audit_result["splits"]["test"]["rows"] if r["class"] in SCORED]
    excluded = {
        split: {
            label: n
            for label, n in audit_result["splits"][split]["by_class"].items()
            if label not in SCORED
        }
        for split in ("train", "val", "test")
    }
    by_label = {label: [r for r in train_rows if r["class"] == label] for label in SCORED}
    train_items = []
    selection = {}
    for label in SCORED:
        pool = by_label[label]
        rng = random.Random(f"sift-tagger-v1-{label}")
        order = list(pool)
        rng.shuffle(order)
        used_all = len(order) <= train_cap
        chosen = order if used_all else order[:train_cap]
        n_codebook = min(codebook_n, len(chosen))
        items = []
        for i, row in enumerate(chosen):
            items.append(
                {
                    "filename": row["filename"],
                    "label": label,
                    "split": row["split"],
                    "image_path": row["image_path"],
                    "path": os.path.join(data_dir, row["image_path"]),
                    "role": "train",
                    "in_codebook": i < n_codebook,
                    "city": city_of(row["filename"], label),
                }
            )
        names = [it["filename"] for it in items]
        selection[label] = {
            "n_train_available": len(pool),
            "n_train_used": len(items),
            "used_all_train_images": used_all,
            "below_train_cap": len(pool) < train_cap,
            "n_codebook": n_codebook,
            "seed": f"sift-tagger-v1-{label}",
            "filename_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
        }
        train_items.extend(items)
    test_items = []
    for row in test_rows:
        label = row["class"]
        test_items.append(
            {
                "filename": row["filename"],
                "label": label,
                "split": row["split"],
                "image_path": row["image_path"],
                "path": os.path.join(data_dir, row["image_path"]),
                "role": "test",
                "in_codebook": False,
                "city": city_of(row["filename"], label),
            }
        )
    return {
        "train": train_items,
        "test": test_items,
        "selection": selection,
        "excluded_from_four_way": excluded,
    }


def check_sift() -> str:
    if not hasattr(cv2, "SIFT_create"):
        raise SystemExit(
            f"cv2 {cv2.__version__} has no SIFT_create. "
            "Install opencv-python-headless in a user env; do not replace the conda env."
        )
    sift = cv2.SIFT_create()
    blank = np.zeros((64, 64), np.uint8)
    cv2.circle(blank, (32, 32), 12, 255, 2)
    _kps, desc = sift.detectAndCompute(blank, None)
    if desc is None or len(desc) < 1 or desc.shape[1] != 128:
        raise SystemExit("SIFT_create ran but did not return 128-d descriptors")
    return cv2.__version__


def _descriptors(path: str) -> dict:
    cv2.setNumThreads(1)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "imread failed", "path": path}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    sparse = cv2.SIFT_create(nfeatures=CAP_SPARSE)
    sparse_kps = sparse.detect(gray, None)
    _, sparse_desc = sparse.compute(gray, sparse_kps) if sparse_kps else (None, None)

    dense_xy = []
    y = GRID_STEP // 2
    while y < h:
        x = GRID_STEP // 2
        while x < w:
            dense_xy.append((float(x), float(y)))
            x += GRID_STEP
        y += GRID_STEP
    if len(dense_xy) > CAP_DENSE:
        idx = np.linspace(0, len(dense_xy) - 1, CAP_DENSE).astype(int)
        seen = []
        used = set()
        for i in idx.tolist():
            if i not in used:
                used.add(i)
                seen.append(dense_xy[i])
        dense_xy = seen
    dense_kps = [cv2.KeyPoint(x, y, float(GRID_STEP)) for x, y in dense_xy]
    _, dense_desc = sparse.compute(gray, dense_kps) if dense_kps else (None, None)

    parts = []
    n_sparse = 0 if sparse_desc is None else int(len(sparse_desc))
    n_dense = 0 if dense_desc is None else int(len(dense_desc))
    if n_sparse:
        parts.append(sparse_desc)
    if n_dense:
        parts.append(dense_desc)
    if parts:
        desc = np.ascontiguousarray(np.vstack(parts), dtype=np.float32)
    else:
        desc = np.zeros((0, 128), np.float32)
    if len(desc) > DESC_CAP:
        desc = desc[:DESC_CAP]
    return {
        "path": path,
        "h": int(h),
        "w": int(w),
        "n_sparse": n_sparse,
        "n_dense": n_dense,
        "n_desc": int(len(desc)),
        "desc": desc,
    }


def extract_all(items: list[dict], workers: int) -> list[dict]:
    paths = [it["path"] for it in items]
    ctx = get_context("spawn")
    by_path = {}
    done = 0
    t0 = time.time()
    with ctx.Pool(processes=workers, initializer=cv2.setNumThreads, initargs=(1,)) as pool:
        for rec in pool.imap_unordered(_descriptors, paths, chunksize=4):
            done += 1
            if done % 200 == 0 or done == len(paths):
                rate = done / max(time.time() - t0, 1e-6)
                print(f"sift progress {done}/{len(paths)} ({rate:.1f}/s)", flush=True)
            by_path[rec["path"]] = rec
    out = []
    n_err = 0
    for it in items:
        rec = by_path[it["path"]]
        merged = dict(it)
        merged.update(rec)
        if rec.get("error"):
            n_err += 1
        out.append(merged)
    if n_err:
        failed = [r["path"] for r in out if r.get("error")][:8]
        raise SystemExit(f"{n_err} images failed to read, first: {failed}")
    return out


def build_vocabulary(records: list[dict], n_words: int) -> np.ndarray:
    mats = [r["desc"] for r in records if r.get("in_codebook") and r["n_desc"] >= MIN_DESCRIPTORS]
    if not mats:
        raise SystemExit("no codebook descriptors")
    data = np.ascontiguousarray(np.vstack(mats), dtype=np.float32)
    print(f"codebook descriptors {data.shape[0]} x {data.shape[1]} words {n_words}", flush=True)
    if data.shape[0] < n_words:
        raise SystemExit(f"only {data.shape[0]} descriptors for {n_words} words")
    cv2.setRNGSeed(0)
    cv2.setNumThreads(16)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    trainer = cv2.BOWKMeansTrainer(n_words, criteria, 1, cv2.KMEANS_PP_CENTERS)
    trainer.add(data)
    t0 = time.time()
    vocab = trainer.cluster()
    print(f"kmeans seconds {time.time() - t0:.1f}", flush=True)
    vocab = np.ascontiguousarray(vocab, dtype=np.float32)
    if vocab.ndim != 2 or vocab.shape[1] != 128:
        raise SystemExit(f"unexpected vocabulary shape {vocab.shape}")
    print(f"vocabulary {vocab.shape}", flush=True)
    return vocab


def l1_histograms(records: list[dict], vocab: np.ndarray) -> None:
    vnorm = np.sum(vocab.astype(np.float32) ** 2, axis=1).astype(np.float32)
    k = vocab.shape[0]
    for i, rec in enumerate(records):
        desc = rec["desc"]
        if rec["n_desc"] < MIN_DESCRIPTORS:
            rec["hist"] = None
            continue
        d = desc.astype(np.float32, copy=False)
        dnorm = np.sum(d * d, axis=1, keepdims=True)
        dist = dnorm + vnorm[None, :] - 2.0 * (d @ vocab.T)
        nn = np.argmin(dist, axis=1)
        hist = np.bincount(nn, minlength=k).astype(np.float64)
        total = hist.sum()
        if total <= 0:
            rec["hist"] = None
            continue
        rec["hist"] = hist / total
        if (i + 1) % 500 == 0:
            print(f"histogram progress {i + 1}/{len(records)}", flush=True)


def verify_bow_extractor(records: list[dict], vocab: np.ndarray) -> dict:
    sift = cv2.SIFT_create(nfeatures=CAP_SPARSE)
    extractor = cv2.BOWImgDescriptorExtractor(sift, cv2.BFMatcher(cv2.NORM_L2))
    extractor.setVocabulary(vocab)
    checked = 0
    max_abs = 0.0
    for rec in records:
        if rec.get("hist") is None:
            continue
        img = cv2.imread(rec["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        sparse_kps = sift.detect(img, None)
        dense_xy = []
        h, w = img.shape[:2]
        y = GRID_STEP // 2
        while y < h:
            x = GRID_STEP // 2
            while x < w:
                dense_xy.append((float(x), float(y)))
                x += GRID_STEP
            y += GRID_STEP
        if len(dense_xy) > CAP_DENSE:
            idx = np.linspace(0, len(dense_xy) - 1, CAP_DENSE).astype(int)
            seen = []
            used = set()
            for i in idx.tolist():
                if i not in used:
                    used.add(i)
                    seen.append(dense_xy[i])
            dense_xy = seen
        kps = list(sparse_kps) + [cv2.KeyPoint(x, y, float(GRID_STEP)) for x, y in dense_xy]
        bow = extractor.compute(img, kps)
        if bow is None:
            continue
        bow = np.asarray(bow, dtype=np.float64).reshape(-1)
        s = bow.sum()
        if s <= 0:
            continue
        bow = bow / s
        max_abs = max(max_abs, float(np.max(np.abs(bow - rec["hist"]))))
        checked += 1
        if checked >= 5:
            break
    return {"images_checked": checked, "max_abs_l1_diff": max_abs}


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    s = a + b
    d = a - b
    mask = s > 0
    if not np.any(mask):
        return 0.0
    return float(0.5 * np.sum((d[mask] ** 2) / s[mask]))


def prototypes(records: list[dict], labels: list[str]) -> dict[str, np.ndarray]:
    protos = {}
    for label in labels:
        hists = [
            r["hist"]
            for r in records
            if r["role"] == "train" and r["label"] == label and r["hist"] is not None
        ]
        if not hists:
            raise SystemExit(f"no training histograms for {label}")
        mean = np.mean(np.stack(hists, axis=0), axis=0)
        total = mean.sum()
        if total <= 0:
            raise SystemExit(f"empty prototype for {label}")
        protos[label] = mean / total
    return protos


def predict(records: list[dict], protos: dict[str, np.ndarray], labels: list[str]) -> None:
    proto_mat = np.stack([protos[lab] for lab in labels], axis=0)
    for rec in records:
        if rec["role"] != "test":
            continue
        if rec["hist"] is None:
            rec["pred"] = None
            continue
        dists = [chi2(rec["hist"], proto_mat[i]) for i in range(len(labels))]
        rec["pred"] = labels[int(np.argmin(dists))]


def metrics(records: list[dict], labels: list[str], train_counts: dict[str, int]) -> dict:
    test = [r for r in records if r["role"] == "test"]
    confusion = {t: {p: 0 for p in labels} for t in labels}
    abstain = {t: 0 for t in labels}
    for rec in test:
        if rec["pred"] is None:
            abstain[rec["label"]] += 1
        else:
            confusion[rec["label"]][rec["pred"]] += 1
    rows = []
    for label in labels:
        support = sum(confusion[label].values()) + abstain[label]
        tp = confusion[label][label]
        predicted = sum(confusion[t][label] for t in labels)
        precision = None if predicted == 0 else tp / predicted
        recall = None if support == 0 else tp / support
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "support": support,
                "tp": tp,
                "predicted": predicted,
                "abstain": abstain[label],
            }
        )
    recalls = [r["recall"] for r in rows if r["recall"] is not None]
    macro_recall = float(np.mean(recalls)) if recalls else None
    n = len(test)
    correct = sum(r["tp"] for r in rows)
    accuracy = None if n == 0 else correct / n
    majority_label = max(labels, key=lambda lab: (train_counts[lab], lab))
    majority_correct = sum(1 for r in test if r["label"] == majority_label)
    return {
        "per_class": rows,
        "macro_recall": macro_recall,
        "accuracy": accuracy,
        "n_test": n,
        "n_correct": correct,
        "n_abstain": sum(abstain.values()),
        "confusion": confusion,
        "abstain": abstain,
        "majority": {
            "label": majority_label,
            "train_count": train_counts[majority_label],
            "n_correct": majority_correct,
            "n_test": n,
            "accuracy": None if n == 0 else majority_correct / n,
        },
    }


def save_recall_plot(met: dict, out_dir: str, labels: list[str]) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recalls = [row["recall"] if row["recall"] is not None else 0.0 for row in met["per_class"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(labels, recalls, color="#3d6f99")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Recall")
    ax.set_title("Test recall, SIFT nearest prototype")
    for i, row in enumerate(met["per_class"]):
        val = row["recall"]
        txt = "n/a" if val is None else f"{val:.3f}"
        ax.text(i, (0 if val is None else val) + 0.03, txt, ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = os.path.join(out_dir, "sift-tagger-recall.png")
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def fmt(x) -> str:
    if x is None:
        return "n/a"
    return f"{x:.3f}"


def public_audit(audit_result: dict) -> dict:
    splits = {}
    for split, info in audit_result["splits"].items():
        splits[split] = {k: v for k, v in info.items() if k != "rows"}
    return {
        "splits": splits,
        "disk": audit_result["disk"],
        "filename_overlap": audit_result["filename_overlap"],
        "table_match": audit_result["table_match"],
        "disk_match": audit_result["disk_match"],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/data/datasets/sidewalk-data")
    p.add_argument("--out-dir", default="/mnt/scratch/sift-tagger/out")
    p.add_argument("--train-cap", type=int, default=TRAIN_CAP)
    p.add_argument("--codebook-per-class", type=int, default=CODEBOOK_PER_CLASS)
    p.add_argument("--words", type=int, default=N_WORDS)
    p.add_argument("--workers", type=int, default=48)
    args = p.parse_args()

    t0 = time.time()
    version = check_sift()
    same = np.array([0.25, 0.75], dtype=np.float64)
    other = np.array([0.75, 0.25], dtype=np.float64)
    if chi2(same, same) != 0.0 or not np.isclose(chi2(same, other), chi2(other, same)) or chi2(same, other) <= 0.0:
        raise SystemExit("chi-square distance failed a sanity check")
    print(f"SIFT_create ok, cv2 {version}", flush=True)
    if not os.path.isdir(args.data_dir):
        raise SystemExit(f"missing data dir {args.data_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    audited = audit(args.data_dir)
    for split, info in audited["splits"].items():
        print(
            f"count {split}: n {info['n']} classes {info['by_class']} "
            f"table_match {info['class_counts_match_table']} missing {info['missing_files']} "
            f"path_mismatch {info['path_mismatch']} cities {info['cities']}",
            flush=True,
        )
    print(
        f"disk_match {audited['disk_match']} filename_overlap {audited['filename_overlap']} "
        f"table_match {audited['table_match']}",
        flush=True,
    )
    picked = select_items(audited, args.data_dir, args.train_cap, args.codebook_per_class)
    print(f"excluded_from_four_way {picked['excluded_from_four_way']}", flush=True)
    for label, info in picked["selection"].items():
        print(
            f"select {label}: available {info['n_train_available']} used {info['n_train_used']} "
            f"used_all {info['used_all_train_images']} codebook {info['n_codebook']}",
            flush=True,
        )
    items = picked["train"] + picked["test"]
    bad_role = [
        it["image_path"]
        for it in items
        if (it["role"] == "train" and not it["image_path"].startswith("train/"))
        or (it["role"] == "test" and not it["image_path"].startswith("test/"))
        or it["split"] != it["role"]
    ]
    if bad_role:
        raise SystemExit(f"refusing to pool splits; bad rows {bad_role[:8]}")
    if any(it["label"] not in SCORED for it in items):
        raise SystemExit("scored items include a label outside the four classes")

    records = extract_all(items, args.workers)
    vocab = build_vocabulary(records, args.words)
    l1_histograms(records, vocab)
    for rec in records:
        rec.pop("desc", None)
    bow_check = verify_bow_extractor(records, vocab)
    print(f"bow extractor check {bow_check}", flush=True)
    labels = list(SCORED)
    protos = prototypes(records, labels)
    predict(records, protos, labels)
    train_counts = {
        label: audited["splits"]["train"]["by_class"].get(label, 0) for label in labels
    }
    met = metrics(records, labels, train_counts)
    plot = save_recall_plot(met, args.out_dir, labels)

    def used(role: str, label: str) -> dict:
        grp = [r for r in records if r["role"] == role and r["label"] == label]
        return {
            "n": len(grp),
            "n_hist": sum(1 for r in grp if r["hist"] is not None),
            "n_too_few_descriptors": sum(1 for r in grp if r["n_desc"] < MIN_DESCRIPTORS),
            "desc_mean": float(np.mean([r["n_desc"] for r in grp])) if grp else None,
            "desc_min": int(min(r["n_desc"] for r in grp)) if grp else None,
            "sizes": dict(Counter(f"{r['w']}x{r['h']}" for r in grp).most_common()),
        }

    payload = {
        "cv2": version,
        "seconds": round(time.time() - t0, 1),
        "data_dir": args.data_dir,
        "scored_classes": labels,
        "null_class": "no_obstacles",
        "excluded_from_four_way": picked["excluded_from_four_way"],
        "audit": public_audit(audited),
        "method": {
            "grid_step": GRID_STEP,
            "cap_sparse": CAP_SPARSE,
            "cap_dense": CAP_DENSE,
            "desc_cap": DESC_CAP,
            "min_descriptors": MIN_DESCRIPTORS,
            "words": int(vocab.shape[0]),
            "words_requested": args.words,
            "train_cap": args.train_cap,
            "codebook_per_class": args.codebook_per_class,
            "chi2": "0.5 * sum (p-q)^2 / (p+q) over bins with p+q > 0",
            "fit_splits": ["train"],
            "eval_splits": ["test"],
            "val_used_for_fit_or_score": False,
            "kmeans": "cv2.BOWKMeansTrainer attempts=1 PP centers, 20 iter, eps 0.1, RNG seed 0",
        },
        "selection": picked["selection"],
        "bow_extractor_check": bow_check,
        "train_used": {label: used("train", label) for label in labels},
        "test_used": {label: used("test", label) for label in labels},
        "metrics": met,
        "recall_png": plot,
    }
    out_json = os.path.join(args.out_dir, "sift_tagger_baseline.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print("\n| Class | Precision | Recall | Support |", flush=True)
    print("|---|---:|---:|---:|", flush=True)
    for row in met["per_class"]:
        print(
            f"| {row['label']} | {fmt(row['precision'])} | {fmt(row['recall'])} | {row['support']} |",
            flush=True,
        )
    print(f"macro recall {fmt(met['macro_recall'])}", flush=True)
    print(f"accuracy {fmt(met['accuracy'])} ({met['n_correct']}/{met['n_test']})", flush=True)
    maj = met["majority"]
    print(
        f"majority {maj['label']} train_count {maj['train_count']} "
        f"accuracy {fmt(maj['accuracy'])} ({maj['n_correct']}/{maj['n_test']})",
        flush=True,
    )
    print(f"abstain {met['n_abstain']}", flush=True)
    print("confusion rows true, cols pred", labels, flush=True)
    for label in labels:
        print(label, [met["confusion"][label][p] for p in labels], "abstain", met["abstain"][label], flush=True)
    print(f"wrote {out_json}", flush=True)
    print(f"seconds {payload['seconds']}", flush=True)


if __name__ == "__main__":
    main()
