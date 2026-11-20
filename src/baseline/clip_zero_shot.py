#!/usr/bin/env python3
"""Five-class zero-shot CLIP evaluation on the processed Pittsburgh test set.

Activate the sidewalk conda environment before running. No API key or training
is needed. Model/processor weights are loaded from the pinned local HF cache.
The supplied images are already 224x224; no old label-centered crop is applied.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import sys
import time

CLASSES = ['Crosswalk', 'CurbRamp', 'Obstacle', 'SurfaceProblem', 'no_obstacles']
HAZARDS = ['Obstacle', 'SurfaceProblem']
MODEL = 'openai/clip-vit-base-patch16'
REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
DATA_ROOT = Path('/data/datasets/proc-sidewalk-data-naive-merged-nulls')
REFERENCE_RUN = Path('/data/eda/openai_baseline/gpt61_sol_low_naive_merged_nulls_pitt_test_5class_20261031T011333Z')
MODEL_PATH = Path('/data/hf_cache/hub/models--openai--clip-vit-base-patch16/snapshots') / REVISION
PROMPTS = {
    'Crosswalk': ['a street photo of a marked pedestrian crosswalk across a road',
                  'painted white stripes or parallel lines marking a pedestrian street crossing'],
    'CurbRamp': ['a street photo of a concrete curb ramp sloping from the sidewalk down to the street at a corner',
                 'a curb cut ramp with a tactile warning strip at a crosswalk'],
    'Obstacle': ['a street photo of a pole, sign, trash can or parked car blocking the sidewalk',
                 'an object obstructing a sidewalk'],
    'SurfaceProblem': ['a street photo of a cracked, broken or uneven sidewalk',
                       'a damaged sidewalk surface with cracks, bumps or grass'],
    'no_obstacles': ['a street photo of a clear, smooth sidewalk with nothing blocking it',
                     'an undamaged sidewalk or street corner with a clear, usable path'],
}
PROBABILITY_FIELDS = ['p_' + cls for cls in CLASSES]
CSV_FIELDS = ['filename', 'image_path', 'split', 'class', 'city', 'label_id', 'y',
              'predicted_class', *PROBABILITY_FIELDS]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def load_images(root, manifest):
    from PIL import Image
    root = root.resolve()
    images = []
    seen = set()
    with manifest.open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'filename', 'image_path', 'split', 'class'} <= set(reader.fieldnames or []):
            raise ValueError('Missing manifest fields.')
        for row in reader:
            relative = Path(row['image_path'])
            if (row['split'] != 'pitt-test' or row['class'] not in CLASSES or
                    relative.parts[:2] != ('pitt-test', row['class']) or len(relative.parts) != 3):
                raise ValueError('Unexpected test split, class, or image path: ' + row['image_path'])
            path = (root / relative).resolve()
            if not path.is_relative_to(root / 'pitt-test' / row['class']):
                raise ValueError('Image path escapes its class directory.')
            if row['image_path'] in seen:
                raise ValueError('Duplicate image in manifest.')
            seen.add(row['image_path'])
            with Image.open(path) as im:
                im.load()
                if im.size != (224, 224) or im.mode != 'RGB' or im.format != 'PNG':
                    raise ValueError('Expected a processed 224x224 RGB PNG: ' + str(path))
            parts = Path(row['filename']).stem.split('-')
            label_id = parts[2] if len(parts) == 4 and parts[:2] == ['gsv', 'pittsburgh'] and parts[2].isdigit() else ''
            images.append({'image_id': row['image_path'], 'image_path': str(path),
                           'image_sha256': sha256(path), 'source_metadata': row, 'y': row['class'],
                           'metadata': {'filename': relative.name, 'image_path': row['image_path'],
                                        'split': 'pitt-test', 'class': row['class'], 'city': 'pittsburgh',
                                        'label_id': label_id}})
    disk = {str(path.relative_to(root)) for cls in CLASSES for path in (root / 'pitt-test' / cls).iterdir()
            if path.is_file() and path.suffix.lower() in {'.png', '.webp', '.jpg', '.jpeg'}}
    if not images or seen != disk:
        raise ValueError('Manifest and test image roster differ.')
    return sorted(images, key=lambda image: image['image_id'])


def validate_probabilities(values):
    if len(values) != len(CLASSES) or any(not math.isfinite(float(v)) or not 0 <= v <= 1 for v in values):
        raise ValueError('Invalid five-class probabilities.')
    if not math.isclose(sum(float(v) for v in values), 1, rel_tol=0, abs_tol=1e-6):
        raise ValueError('Probabilities do not sum to one.')
    index = max(range(len(CLASSES)), key=lambda i: values[i])
    return {'predicted_class': CLASSES[index], **dict(zip(PROBABILITY_FIELDS, map(float, values)))}


def subtype_metrics(rows):
    from sklearn.metrics import balanced_accuracy_score
    eligible = [row for row in rows if row['y'] in HAZARDS and row['predicted_class'] in HAZARDS]
    matrix = [[sum(r['y'] == actual and r['predicted_class'] == predicted for r in eligible)
               for predicted in HAZARDS] for actual in HAZARDS]
    supports = [sum(row) for row in matrix]
    recalls = [matrix[i][i] / count if count else None for i, count in enumerate(supports)]
    score = sum(recalls) / 2 if all(v is not None for v in recalls) else None
    if score is not None:
        assert math.isclose(score, balanced_accuracy_score([r['y'] for r in eligible], [r['predicted_class'] for r in eligible]))
    return {'evaluated_true_hazards': len(eligible), 'class_order': HAZARDS,
            'confusion_true_rows_predicted_columns': matrix, 'support': dict(zip(HAZARDS, supports)),
            'recall': dict(zip(HAZARDS, recalls)), 'balanced_accuracy': score}


def hazard_metrics(rows, k=46):
    if len(rows) < k:
        raise ValueError('Fewer predictions than the requested top-k.')
    def score(row):
        return Decimal(str(row['p_Obstacle'])) + Decimal(str(row['p_SurfaceProblem']))
    ranked = sorted(rows, key=lambda row: (-score(row), row['image_path']))
    top = ranked[:k]
    hits = sum(row['y'] in HAZARDS for row in top)
    cutoff = score(top[-1])
    above = [r for r in rows if score(r) > cutoff]
    tied = [r for r in rows if score(r) == cutoff]
    fixed = sum(r['y'] in HAZARDS for r in above)
    tied_hits = sum(r['y'] in HAZARDS for r in tied)
    slots = k - len(above)
    predicted = [r for r in rows if r['predicted_class'] in HAZARDS]
    result = {'n_images': len(rows), 'hazard_classes': HAZARDS, 'k': k,
              'ranking': 'p_Obstacle + p_SurfaceProblem descending; image_path ascending to break ties',
              'precision_at_46': {'true_hazards': hits, 'false_positive_nonhazards': k - hits,
                                  'precision': hits / k, 'true_class_counts': dict(Counter(r['y'] for r in top)),
                                  'cutoff_score': str(cutoff), 'cutoff_tie_count': len(tied),
                                  'cutoff_tie_slots_selected': slots, 'cutoff_tie_true_hazards': tied_hits,
                                  'precision_range_under_other_tie_breaks': [
                                      (fixed + max(0, slots - (len(tied) - tied_hits))) / k,
                                      (fixed + min(slots, tied_hits)) / k]},
              'all_predicted_hazards': {'count': len(predicted),
                                        'false_positive_nonhazards': sum(r['y'] not in HAZARDS for r in predicted),
                                        'subtype': subtype_metrics(predicted)},
              'top_46_hazard_subtype': subtype_metrics(top)}
    return result, ranked


def classification_metrics(rows):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    targets = [r['y'] for r in rows]
    predictions = [r['predicted_class'] for r in rows]
    report = classification_report(targets, predictions, labels=CLASSES, output_dict=True, zero_division=0)
    return {'accuracy': accuracy_score(targets, predictions), 'macro_f1': report['macro avg']['f1-score'],
            'macro_precision': report['macro avg']['precision'], 'macro_recall': report['macro avg']['recall'],
            'balanced_accuracy': sum(report[c]['recall'] for c in CLASSES) / len(CLASSES),
            'weighted_f1': report['weighted avg']['f1-score'],
            'per_class': {c: report[c] for c in CLASSES},
            'confusion': {'labels': CLASSES, 'matrix': confusion_matrix(targets, predictions, labels=CLASSES).tolist()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=DATA_ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--reference-run', type=Path, default=REFERENCE_RUN)
    parser.add_argument('--model-path', type=Path, default=MODEL_PATH)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--threads', type=int, default=48)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.batch_size <= 0 or args.threads <= 0:
        parser.error('Batch size and thread count must be positive.')
    manifest = args.manifest or args.data_root / 'pitt-test.csv'
    images = load_images(args.data_root, manifest)
    inventory = [{key: image[key] for key in ['image_id', 'image_sha256', 'y']} for image in images]
    reference = json.loads((args.reference_run / 'run_config.json').read_text())
    if reference['classes'] != CLASSES or reference['manifest_sha256'] != sha256(manifest):
        raise ValueError('Classes or manifest differ from the GPT comparison run.')
    if inventory != json.loads((args.reference_run / 'image_inventory.json').read_text()):
        raise ValueError('Test images, bytes, or labels differ from the GPT comparison run.')
    counts = dict(Counter(image['y'] for image in images))
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'images': len(images), 'classes': CLASSES,
                          'class_counts': counts, 'exact_gpt_test_roster_and_image_bytes': True,
                          'model': MODEL, 'model_revision': REVISION, 'mode': 'zero_shot'}, indent=2))
        return
    if args.out is None:
        parser.error('--out is required unless --dry-run is used.')
    import numpy as np
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        config = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'model': MODEL,
                  'model_revision': REVISION, 'model_path': str(args.model_path), 'mode': 'zero_shot',
                  'classes': CLASSES, 'prompts': PROMPTS, 'image_count': len(images), 'class_counts': counts,
                  'data_root': str(args.data_root.resolve()), 'manifest': str(manifest.resolve()),
                  'manifest_sha256': sha256(manifest), 'preprocessing_config_sha256': sha256(args.data_root / 'config.yaml'),
                  'runner_sha256': sha256(Path(__file__)), 'model_weights_sha256': sha256(args.model_path / 'pytorch_model.bin'),
                  'reference_gpt_run': str(args.reference_run), 'exact_gpt_test_roster_and_image_bytes': True,
                  'input': 'Stored 224x224 RGB PNG, standard CLIPProcessor normalization; no additional crop or padding.',
                  'text_ensemble': 'L2-normalize each prompt embedding, mean two prompts per class, L2-normalize each class mean.',
                  'scores': 'softmax(exp(model.logit_scale) * normalized image embedding @ normalized class text embeddings.T)',
                  'prediction_rule': 'argmax; ties follow classes order', 'probability_storage': 'full float32-derived values, not rounded to four decimals',
                  'device': 'cpu', 'batch_size': args.batch_size, 'torch_threads': args.threads, 'seed': 42,
                  'packages': {p: importlib.metadata.version(p) for p in ['torch', 'transformers', 'numpy', 'scikit-learn', 'Pillow']},
                  'python_version': sys.version, 'local_files_only': True}
        write_json(args.out / 'run_config.json', config)
        write_json(args.out / 'image_inventory.json', inventory)
        shutil.copy2(manifest, args.out / 'input_manifest.csv')
        shutil.copy2(args.data_root / 'config.yaml', args.out / 'dataset_config.yaml')
        shutil.copy2(Path(__file__), args.out / 'clip_zero_shot.py')
        for name in ['config.json', 'preprocessor_config.json', 'tokenizer_config.json']:
            shutil.copy2(args.model_path / name, args.out / ('clip_' + name))
        print('Loading pinned cached CLIP model on CPU.', flush=True)
        model = CLIPModel.from_pretrained(str(args.model_path), local_files_only=True, use_safetensors=False).eval()
        processor = CLIPProcessor.from_pretrained(str(args.model_path), local_files_only=True)
        with torch.inference_mode():
            tokens = processor.tokenizer([p for cls in CLASSES for p in PROMPTS[cls]], padding=True, return_tensors='pt')
            text = model.get_text_features(**tokens)
            text = torch.nn.functional.normalize(getattr(text, 'pooler_output', text), dim=-1).numpy()
            scale = float(model.logit_scale.exp())
        text = text.reshape(len(CLASSES), 2, -1).mean(1)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        np.save(args.out / 'text_features.npy', text)
        config['learned_logit_scale_exp'] = scale
        config['projection_dim'] = int(text.shape[1])
        write_json(args.out / 'run_config.json', config)
        rows, embeddings = [], []
        with (args.out / 'predictions.jsonl').open('x') as journal, (args.out / 'pittsburgh_predictions.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for begin in range(0, len(images), args.batch_size):
                batch = images[begin:begin + args.batch_size]
                pictures = []
                for image in batch:
                    with Image.open(image['image_path']) as picture:
                        pictures.append(picture.convert('RGB'))
                inputs = processor.image_processor(images=pictures, return_tensors='pt')
                with torch.inference_mode():
                    features = model.get_image_features(**inputs)
                    features = torch.nn.functional.normalize(getattr(features, 'pooler_output', features), dim=-1).numpy()
                embeddings.append(features)
                logits = scale * features @ text.T
                probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                for image, values in zip(batch, probabilities):
                    prediction = validate_probabilities(values)
                    record = {**image, 'prediction': prediction, 'status': 'ok'}
                    row = {**image['metadata'], 'y': image['y'], **prediction}
                    journal.write(json.dumps(record) + '\n')
                    writer.writerow(row)
                    rows.append(row)
                journal.flush(); os.fsync(journal.fileno())
                stream.flush(); os.fsync(stream.fileno())
                rate = len(rows) / (time.time() - started)
                print(f'[{len(rows)}/{len(images)}] saved; {rate:.1f} images/s including startup', flush=True)
        np.save(args.out / 'image_features.npy', np.concatenate(embeddings))
        # Derive every metric from the saved prediction CSV to make results reproducible.
        with (args.out / 'pittsburgh_predictions.csv').open() as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == len(images) and {r['image_path'] for r in saved} == {r['image_id'] for r in images}
        metrics = classification_metrics(saved)
        hazards, ranked = hazard_metrics(saved)
        hazards['prediction_csv_sha256'] = sha256(args.out / 'pittsburgh_predictions.csv')
        write_json(args.out / 'hazard_metrics_at_46.json', hazards)
        with (args.out / 'hazard_ranking.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['rank', 'image_path', 'true_class', 'predicted_class', 'hazard_probability', 'true_hazard', 'top_46'])
            writer.writeheader()
            for rank, row in enumerate(ranked, 1):
                writer.writerow({'rank': rank, 'image_path': row['image_path'], 'true_class': row['y'],
                                 'predicted_class': row['predicted_class'], 'hazard_probability': str(Decimal(row['p_Obstacle']) + Decimal(row['p_SurfaceProblem'])),
                                 'true_hazard': row['y'] in HAZARDS, 'top_46': rank <= 46})
        results = {'model': MODEL, 'model_revision': REVISION, 'mode': 'zero_shot', 'classes': CLASSES,
                   'n_test_pittsburgh': len(saved), 'class_counts': counts, 'classification': metrics,
                   'hazard_metrics_at_46': hazards, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                   'wall_seconds': time.time() - started}
        write_json(args.out / 'results.json', results)
        print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
