#!/usr/bin/env python3
"""Five-class SIFT reconstruction with complete predictions and fit provenance.

Keeps the original sift_tagger_baseline feature/codebook/prototype functions.
Runs two predefined training pools; neither is selected using test performance.
Distances and margins are scores, not calibrated probabilities.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
import sift_tagger_baseline as engine

CLASSES = ['Crosswalk', 'CurbRamp', 'Obstacle', 'SurfaceProblem', 'no_obstacles']
HAZARDS = ['Obstacle', 'SurfaceProblem']
CLEAN = ['Crosswalk', 'CurbRamp', 'no_obstacles']
PLANS = {'pitt_only': ['pitt-train'],
         'pooled_other_and_pitt': ['other-cities-train', 'pitt-train']}
EXPECTED_HASHES = {
    'other-cities-train': '57e004a3b6d9e5f0482be285f8bea920b371428caf4fe9ab6c44c13ef9af19a2',
    'pitt-train': '081723b38ebf0b5448d56e67be6b0c99dafdf2e3cf239e7d7a239b5fcfaad0cf',
    'pitt-val': 'ae5cc3b382794050c8ad92dcb28ea1a4b82240ed266d329404a67e791a5e69de',
    'pitt-test': '6dc4e536d46a57ebb0d3ca78175d7429a4aed6513a93877ec0ae413562dbb818',
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty CSV: {path}')
    fields = list(dict.fromkeys(k for r in rows for k in r))
    tmp = path.with_suffix('.csv.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def classification(truth, pred, labels):
    cm = [[0 for _ in labels] for _ in labels]
    for a, b in zip(truth, pred, strict=True):
        cm[labels.index(a)][labels.index(b)] += 1
    per_class = []
    for i, label in enumerate(labels):
        tp = cm[i][i]
        support = sum(cm[i])
        predicted = sum(row[i] for row in cm)
        per_class.append(dict(label=label, support=support, predicted_count=predicted,
            true_positive=tp, precision=tp / predicted if predicted else 0.,
            recall=tp / support if support else None,
            f1=2 * tp / (support + predicted) if support + predicted else 0.))
    supported = [r for r in per_class if r['support']]
    return dict(labels=labels, n=len(truth), confusion=cm, per_class=per_class,
        accuracy=sum(cm[i][i] for i in range(len(labels))) / len(truth) if truth else None,
        balanced_accuracy=sum(r['recall'] for r in supported) / len(supported) if supported else None,
        macro_precision=sum(r['precision'] for r in per_class) / len(labels),
        macro_f1=sum(r['f1'] for r in per_class) / len(labels))


def predictions(records, protos):
    rows = []
    for rec in records:
        if rec['role'] != 'test':
            continue
        if rec['hist'] is None:
            raise RuntimeError('Test abstention: cannot silently omit a test image')
        distances = {cl: engine.chi2(rec['hist'], protos[cl]) for cl in CLASSES}
        # Exact distance ties prefer clean, then the fixed class order.
        pred = min(CLASSES, key=lambda cl: (distances[cl], cl in HAZARDS, CLASSES.index(cl)))
        sub = min(HAZARDS, key=lambda cl: distances[cl])  # Obstacle first on ties
        margin = min(distances[c] for c in CLEAN) - min(distances[c] for c in HAZARDS)
        assert (pred in HAZARDS) == (margin > 0)
        rows.append(dict(filename=rec['filename'], image_path=rec['image_path'],
            image_sha256=rec['image_sha256'], split=rec['split'], **{'class': rec['label']},
            source_class=rec['source_class'], predicted_class=pred,
            true_hazard=int(rec['label'] in HAZARDS), predicted_hazard=int(margin > 0),
            hazard_score=margin, forced_subtype=sub, n_descriptors=rec['n_desc'],
            **{'distance_' + cl: distances[cl] for cl in CLASSES}))
    ranked = sorted(rows, key=lambda r: (-r['hazard_score'], r['image_sha256']))
    # Same bytes have identical scores; their order is immaterial to class metrics
    # only if their true labels agree. Refuse ambiguous labels at a cutoff tie.
    by_hash = defaultdict(set)
    for r in ranked:
        by_hash[r['image_sha256']].add(r['class'])
    assert all(len(classes) == 1 for classes in by_hash.values())
    for rank, r in enumerate(ranked, 1):
        r['hazard_rank'] = rank
        r['in_top46'] = int(rank <= 46)
    return rows, ranked


def evaluate(rows, ranked):
    def report(subset, truth_key, pred_key, labels):
        return classification([r[truth_key] for r in subset], [r[pred_key] for r in subset], labels)
    five = report(rows, 'class', 'predicted_class', CLASSES)
    binary = report(rows, 'true_hazard', 'predicted_hazard', [0, 1])
    true_hazards = [r for r in rows if r['true_hazard']]
    gated = [r for r in true_hazards if r['predicted_hazard']]
    all_sub = report(true_hazards, 'class', 'forced_subtype', HAZARDS)
    gated_sub = report(gated, 'class', 'forced_subtype', HAZARDS)
    collapse = lambda cl: cl if cl in HAZARDS else 'clean'
    pipeline = classification([collapse(r['class']) for r in rows],
                              [collapse(r['predicted_class']) for r in rows], ['clean'] + HAZARDS)
    cutoff = ranked[45]['hazard_score']
    tied = [r for r in rows if r['hazard_score'] == cutoff]
    above = [r for r in rows if r['hazard_score'] > cutoff]
    slots = 46 - len(above)
    h_tied = sum(r['true_hazard'] for r in tied)
    h_above = sum(r['true_hazard'] for r in above)
    hits = sum(r['true_hazard'] for r in ranked[:46])
    ranking = dict(k=46, true_hazards_in_top_k=hits, precision_at_46=hits / 46,
        recall_at_46=hits / len(true_hazards), total_true_hazards=len(true_hazards),
        cutoff_score=cutoff, tied_records_at_cutoff=len(tied), slots_at_cutoff=slots,
        possible_hits_min=h_above + max(0, slots - (len(tied) - h_tied)),
        possible_hits_max=h_above + min(slots, h_tied),
        tiebreak='score descending, SHA256(image bytes) ascending; identical-byte same-label duplicates are interchangeable')
    return dict(five_class=five, hazard_binary=binary, subtype_all_true_hazards=all_sub,
                subtype_true_hazards_also_predicted_hazard=gated_sub,
                three_class_pipeline=pipeline, ranking=ranking)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('/data/datasets/proc-sidewalk-data-naive-merged-nulls'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=16)
    args = p.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise SystemExit('Refusing to overwrite an existing run; use a new output directory.')
    started = utc()
    save_json(out / 'status.json', dict(state='running', started_at=started))
    engine.check_sift()
    data = args.data_dir.resolve()
    manifests, manifest_hashes = {}, {}
    for split, expected in EXPECTED_HASHES.items():
        manifest = data / (split + '.csv')
        manifest_hashes[split] = sha(manifest)
        assert manifest_hashes[split] == expected, f'Manifest changed: {split}'
        with manifest.open(newline='') as f:
            manifests[split] = list(csv.DictReader(f))
        rows = manifests[split]
        assert len({r['image_path'] for r in rows}) == len(rows)
        assert {r['class'] for r in rows} == set(CLASSES)
        for r in rows:
            rel = Path(r['image_path'])
            assert r['split'] == split and rel.parts[:2] == (split, r['class'])
            assert (data / rel).is_file() and (data / rel).resolve().is_relative_to(data)
    assert len(manifests['pitt-test']) == 380
    assert sum(r['class'] in HAZARDS for r in manifests['pitt-test']) == 146
    # Audit byte overlap before fitting; don't silently redefine the historical dataset.
    hashes = {}
    for split in ['other-cities-train', 'pitt-train', 'pitt-test']:
        for r in manifests[split]:
            hashes[r['image_path']] = sha(data / r['image_path'])
        print('Hashed', split, len(manifests[split]), flush=True)
    test_hashes = {hashes[r['image_path']] for r in manifests['pitt-test']}
    all_overlap = [dict(split=s, image_path=r['image_path'], image_sha256=hashes[r['image_path']])
                   for s in ['other-cities-train', 'pitt-train'] for r in manifests[s]
                   if hashes[r['image_path']] in test_hashes]
    save_json(out / 'data_audit.json', dict(manifest_sha256=manifest_hashes,
        split_class_counts={s: dict(Counter(r['class'] for r in rr)) for s, rr in manifests.items()},
        test_records=380, unique_test_image_hashes=len(test_hashes),
        train_records_sharing_test_bytes=all_overlap,
        validation_used_for_fit_or_tuning=False))
    # This is a reconstruction, so fail for review if the canonical training data leaks.
    if all_overlap:
        raise RuntimeError(f'{len(all_overlap)} training records share bytes with test; inspect audit before fitting')
    code = out / 'code'
    code.mkdir(exist_ok=True)
    for source in [Path(__file__), Path(engine.__file__)]:
        target = code / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
    save_json(out / 'environment.json', dict(python=sys.version, executable=sys.executable,
        opencv=cv2.__version__, numpy=np.__version__, platform=platform.platform(),
        cpu_count=os.cpu_count(), argv=sys.argv,
        thread_environment={k: os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
        source_sha256={p.name: sha(p) for p in code.glob('*.py')}))
    (out / 'opencv_build.txt').write_text(cv2.getBuildInformation())
    summaries = []
    for plan, splits in PLANS.items():
        folder = out / plan
        folder.mkdir()
        start = time.monotonic()
        pool = [r for s in splits for r in manifests[s]]
        items = []
        selection = {}
        for cl in CLASSES:
            candidates = [r for r in pool if r['class'] == cl]
            random.Random('sift-tagger-v1-' + cl).shuffle(candidates)
            chosen = candidates[:engine.TRAIN_CAP]
            selection[cl] = dict(available=len(candidates), selected=len(chosen),
                codebook=min(engine.CODEBOOK_PER_CLASS, len(chosen)))
            for i, r in enumerate(chosen):
                items.append(dict(r, label=cl, path=str(data / r['image_path']), role='train',
                                  in_codebook=i < engine.CODEBOOK_PER_CLASS,
                                  image_sha256=hashes[r['image_path']]))
        for r in manifests['pitt-test']:
            items.append(dict(r, label=r['class'], path=str(data / r['image_path']), role='test',
                              in_codebook=False, image_sha256=hashes[r['image_path']]))
        save_json(folder / 'config.json', dict(plan=plan, fit_splits=splits,
            test_split='pitt-test', class_order=CLASSES, hazards=HAZARDS,
            train_selection=selection, train_cap_per_class=engine.TRAIN_CAP,
            codebook_cap_per_class=engine.CODEBOOK_PER_CLASS, visual_words=engine.N_WORDS,
            grid_step=engine.GRID_STEP, sparse_requested=engine.CAP_SPARSE,
            dense_cap=engine.CAP_DENSE, total_descriptor_cap=engine.DESC_CAP,
            min_descriptors=engine.MIN_DESCRIPTORS,
            sampling='CSV row order, split concatenation as fit_splits; random.Random(sift-tagger-v1-<class>).shuffle',
            kmeans='OpenCV BOWKMeansTrainer: seed 0, kmeans++ centers, 1 attempt, max20 iterations, epsilon0.1',
            prototype='mean per-image L1 histogram, then L1 normalize',
            distance='0.5 * sum((hist-prototype)**2/(hist+prototype)) over positive denominator bins',
            hazard_score='min(clean distances) - min(hazard distances)',
            binary_decision='hazard_score > 0; exact zero is non-hazard',
            five_class_tie='nearest distance, clean before hazard, then fixed class order',
            forced_subtype='nearest of Obstacle and SurfaceProblem, ties Obstacle; evaluated on all146 true hazards',
            score_type='uncalibrated distances and margins; no class probabilities',
            reconstruction_warning='Original 380-image SIFT model/predictions/training configuration are unavailable. New runs, not exact historical reproduction.',
            selection_policy='Two predefined training pools; no tuning or selection using validation or test',
            image_preprocessing='use existing 224x224 processed images; original feature extractor converts BGR to grayscale',
            workers=args.workers, started_at=utc()))
        print('START', plan, selection, flush=True)
        records = engine.extract_all(items, args.workers)
        assert {(r['w'], r['h']) for r in records} == {(224, 224)}
        vocab = engine.build_vocabulary(records, engine.N_WORDS)
        engine.l1_histograms(records, vocab)
        protos = engine.prototypes(records, CLASSES)
        train = [r for r in records if r['role'] == 'train']
        test = [r for r in records if r['role'] == 'test']
        assert all(r['hist'] is not None for r in records), 'Unexpected low-descriptor image'
        save_csv(folder / 'train_selection.csv', [dict(image_path=r['image_path'],
            image_sha256=r['image_sha256'], split=r['split'], **{'class': r['label']},
            source_class=r['source_class'], in_codebook=int(r['in_codebook']),
            n_descriptors=r['n_desc']) for r in train])
        np.savez_compressed(folder / 'model_and_histograms.npz', vocabulary=vocab,
            classes=np.array(CLASSES), prototypes=np.stack([protos[c] for c in CLASSES]),
            train_histograms=np.stack([r['hist'] for r in train]),
            test_histograms=np.stack([r['hist'] for r in test]),
            train_image_paths=np.array([r['image_path'] for r in train]),
            test_image_paths=np.array([r['image_path'] for r in test]))
        rows, ranked = predictions(records, protos)
        metrics = evaluate(rows, ranked)
        save_csv(folder / 'test_predictions.csv', rows)
        save_csv(folder / 'hazard_ranking.csv', ranked)
        save_json(folder / 'metrics.json', metrics)
        save_csv(folder / 'per_class_metrics.csv', [dict(task=task, **r)
            for task, met in metrics.items() if task != 'ranking' for r in met['per_class']])
        by_class = []
        for cl in CLASSES:
            group = [r for r in rows if r['class'] == cl]
            nh = sum(r['predicted_hazard'] for r in group)
            by_class.append(dict(label=cl, support=len(group), predicted_hazard_count=nh,
                correct_hazard_decision_rate=(nh if cl in HAZARDS else len(group) - nh) / len(group),
                top46_count=sum(r['in_top46'] for r in group)))
        save_csv(folder / 'hazard_decisions_by_original_class.csv', by_class)
        summary = dict(plan=plan, test_records=len(rows), training_images=len(train),
            codebook_images=sum(r['in_codebook'] for r in train),
            precision_at_46=metrics['ranking']['precision_at_46'],
            true_hazards_in_top46=metrics['ranking']['true_hazards_in_top_k'],
            recall_at_46=metrics['ranking']['recall_at_46'],
            subtype_balanced_accuracy=metrics['subtype_all_true_hazards']['balanced_accuracy'],
            gated_subtype_balanced_accuracy=metrics['subtype_true_hazards_also_predicted_hazard']['balanced_accuracy'],
            gated_subtype_n=metrics['subtype_true_hazards_also_predicted_hazard']['n'],
            five_class_accuracy=metrics['five_class']['accuracy'],
            five_class_balanced_accuracy=metrics['five_class']['balanced_accuracy'],
            seconds=time.monotonic() - start)
        summaries.append(summary)
        save_json(folder / 'status.json', dict(state='complete', finished_at=utc(), **summary))
        save_csv(out / 'summary.csv', summaries)
        print('COMPLETE', json.dumps(summary), flush=True)
        del records, train, test, vocab, protos
    save_json(out / 'status.json', dict(state='complete', started_at=started,
        finished_at=utc(), variants=list(PLANS), test_predictions_per_variant=380))


if __name__ == '__main__':
    main()
