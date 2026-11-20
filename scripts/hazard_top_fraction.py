#!/usr/bin/env python3
"""Measure hazard precision in a fixed top fraction of saved, labeled predictions."""
import argparse
import csv
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
import math
from pathlib import Path


def evaluate(rows, fraction='0.10'):
    fraction = Decimal(str(fraction))
    if not fraction.is_finite() or not 0 < fraction <= 1:
        raise ValueError('fraction must be finite and in (0, 1]')
    if not rows:
        raise ValueError('No predictions')
    ids = [r['sample_id'] for r in rows]
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Expected one unique sample_id per image; do not mix trials')
    if len({r.get('split', '') for r in rows}) != 1:
        raise ValueError('Do not mix evaluation splits')
    if len({r.get('hazard_trial', '') for r in rows}) != 1:
        raise ValueError('Do not mix model trials')
    parsed = []
    for r in rows:
        p = float(r['p_hazard'])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError('Invalid hazard probability')
        if r['hazard_label'] not in {'0', '1'}:
            raise ValueError('hazard_label must be known binary ground truth')
        parsed.append((r['sample_id'], p, int(r['hazard_label'])))
    ranked = sorted(parsed, key=lambda x: (-x[1], x[0]))
    k = int((fraction * len(rows)).to_integral_value(rounding=ROUND_CEILING))
    hits = sum(x[2] for x in ranked[:k])
    return {
        'split': rows[0].get('split', ''),
        'hazard_trial': rows[0].get('hazard_trial', ''),
        'total_images': len(rows), 'total_hazards': sum(x[2] for x in ranked),
        'fraction': str(fraction), 'rounding': 'ceil(fraction * images)', 'k': k,
        'true_hazards_in_top_k': hits, 'precision_at_top_fraction': hits / k,
        'cutoff_probability': ranked[k-1][1],
        'tie_crosses_cutoff': k < len(rows) and ranked[k-1][1] == ranked[k][1],
        'tie_break': 'sample_id ascending', 'top_sample_ids': [x[0] for x in ranked[:k]],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--fraction', default='0.10')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.predictions.open(newline='') as f:
        result = evaluate(list(csv.DictReader(f)), a.fraction)
    result['source_csv'] = str(a.predictions.resolve())
    result['source_csv_sha256'] = hashlib.sha256(a.predictions.read_bytes()).hexdigest()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'top_sample_ids'}, indent=2))


if __name__ == '__main__':
    main()
