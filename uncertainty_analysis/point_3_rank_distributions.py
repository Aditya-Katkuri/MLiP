"""Point 3: rank intervals, top-k frequencies and within-head outranking probabilities."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from .common import (Dataset, SPLITS, parser, order_scores, export_ranking, pairwise_counts,
                     pairwise_exports, save_json, save_csv, setup_plot, save_figure,
                     results_table, shared_notes, md_table, pct)


def rank_distribution(p, k):
    n, b = p.shape
    assert 0 < k <= n
    ranks = rankdata(-p, axis=0, method='average')
    first, topk = np.zeros(n), np.zeros(n)
    for member in range(b):
        values = p[:, member]
        above = (values[None, :] > values[:, None]).sum(axis=1)
        tied = (values[None, :] == values[:, None]).sum(axis=1)
        first += np.clip((1 - above) / tied, 0, 1) / b
        topk += np.clip((k - above) / tied, 0, 1) / b
    summary = dict(expected_rank=ranks.mean(1), rank_median=np.median(ranks, axis=1),
                   rank_q025=np.quantile(ranks, .025, axis=1, method='linear'),
                   rank_q975=np.quantile(ranks, .975, axis=1, method='linear'),
                   p_first=first, p_top_k=topk)
    summary['rank_interval_width'] = summary['rank_q975'] - summary['rank_q025']
    assert np.isclose(first.sum(), 1) and np.isclose(topk.sum(), k)
    return ranks, summary


def main():
    args = parser(__doc__).parse_args()
    folder = args.output / 'point_3'; folder.mkdir(parents=True, exist_ok=True)
    source = Dataset(args.archive)
    metrics, uncertainty, rows, table_rows = {}, {}, [], []
    plt = setup_plot()
    for split in SPLITS:
        data = source.load(split)
        ranks, summary = rank_distribution(data.p, args.k)
        wins, ties = pairwise_counts(data.p)
        # Expected midrank equals one plus the summed half-tie losses.
        expected = 1 + (wins.T + .5 * ties).sum(1) / data.p.shape[1] - .5
        assert np.allclose(summary['expected_rank'], expected)
        pair_stats = pairwise_exports(folder, data, wins, ties)
        np.savez_compressed(folder / (split + '_ranks.npz'), ranks=ranks,
                            sample_ids=data.ids, member_ids=data.member_ids)
        rank_frame = pd.DataFrame(ranks, columns=[f'head_{j:03d}' for j in data.member_ids])
        rank_frame.insert(0, 'sample_id', data.ids); save_csv(folder / (split + '_ranks.csv'), rank_frame)
        orders = dict(mean=order_scores(data.p.mean(1), data.ids),
                      expected_rank=order_scores(summary['expected_rank'], data.ids, descending=False),
                      top_k_frequency=np.lexsort((data.ids, summary['expected_rank'], -summary['p_top_k'])))
        metrics[split] = {}
        for name, order in orders.items():
            metrics[split][name] = export_ranking(folder, data, name, order, summary, args.k)
            rows.append((split, name.replace('_', ' '), metrics[split][name]))
        width = summary['rank_interval_width']
        selected = orders['expected_rank'][:args.k]
        uncertainty[split] = dict(median_interval_width=float(np.median(width)),
            p90_interval_width=float(np.quantile(width, .9)),
            intervals_crossing_cutoff=int(((summary['rank_q025'] <= args.k) & (summary['rank_q975'] > args.k)).sum()),
            observations_p_top_k_ge_095=int((summary['p_top_k'] >= .95).sum()),
            hazards_among_stable_top_k=int(data.y[summary['p_top_k'] >= .95].sum()),
            nonhazards_among_stable_top_k=int((1-data.y[summary['p_top_k'] >= .95]).sum()),
            observations_p_top_k_positive=int((summary['p_top_k'] > 0).sum()),
            expected_rank_top_k_overlap_mean=len(set(selected) & set(orders['mean'][:args.k])),
            mean_p_top_k_in_expected_rank_selection=float(summary['p_top_k'][selected].mean()),
            **pair_stats)
        u = uncertainty[split]
        table_rows.append([split, f"{u['median_interval_width']:.1f}", f"{u['p90_interval_width']:.1f}",
                           u['intervals_crossing_cutoff'], u['observations_p_top_k_ge_095'],
                           u['observations_p_top_k_positive']])
        order = orders['expected_rank']
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(1, len(order) + 1)
        ax.fill_between(x, summary['rank_q025'][order], summary['rank_q975'][order], alpha=.3, label='2.5–97.5% midrank interval')
        ax.plot(x, summary['expected_rank'][order], label='Expected rank', lw=1.4)
        ax.axhline(args.k, color='tab:red', ls='--', label=f'Review cutoff {args.k}')
        ax.set(xlabel='Observation position in expected-rank ordering', ylabel='Rank across fitted heads', title=split)
        ax.invert_yaxis(); ax.legend(); save_figure(fig, folder / (split + '_rank_intervals')); plt.close(fig)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, summary['p_top_k'][order], lw=1.2)
        ax.axvline(args.k, color='tab:red', ls='--')
        ax.set(xlabel='Observation position in expected-rank ordering', ylabel=f'Frequency in top {args.k}', title=split, ylim=(-.02, 1.02))
        save_figure(fig, folder / (split + '_top_k_frequency')); plt.close(fig)
    save_json(folder / 'metrics.json', dict(metrics=metrics, uncertainty=uncertainty, k=args.k))
    save_json(folder / 'provenance.json', source.provenance())
    interpretation = '\n'.join(f"- **{split}:** only {u['observations_p_top_k_ge_095']} observations appear in the top {args.k} with frequency at least 95%; "
                              f"{u['intervals_crossing_cutoff']} rank intervals cross the cutoff. "
                              f"Expected-rank and mean rankings share {u['expected_rank_top_k_overlap_mean']}/{args.k} selected images. "
                              f"Among the ≥95%-stable selections, {u['hazards_among_stable_top_k']} are labelled hazards and "
                              f"{u['nonhazards_among_stable_top_k']} non-hazards."
                              for split, u in uncertainty.items())
    report = f'''# Point 3: uncertainty in ranks and pairwise comparisons

## Method

Rank all observations **within each of the same 128 heads**, highest hazard probability first. Rank 1 is best. For each observation export mean (expected) rank, median rank, the **2.5th and 97.5th empirical percentiles of rank**, interval width, P(first) and P(top {args.k}). Percentiles use linear interpolation.

Exact probability ties receive **midranks**. For P(first) and P(top k), allocate tied slots equally: if a tie group of size t has a items strictly above it, its top-k contribution is `clip((k−a)/t, 0, 1)`. Thus top-k frequencies sum to k and first-place frequencies sum to 1. The reported rank intervals are intervals of midranks across fits; they do not add hypothetical random tie-breaking noise. Final equal expected ranks use sample ID; top-k-frequency ordering breaks frequency ties by expected rank, then sample ID.

Three fixed rankings are evaluated: mean probability; smallest expected rank (the primary point-3 ranking); and largest P(top k). No labels determine these orderings. Expected midrank is exactly one plus the sum of pairwise loss probabilities with half-credit for ties. Thus this ordering is the weighted Borda ranking. It need not be the pairwise-agreement optimum explored in point 4.

For every observation pair, export `P_strict(i,j) = count_b[p_i(b) > p_j(b)] / 128` and `P_tie(i,j)`. Comparisons use corresponding predictions from the **same head**, not all 128×128 cross-head combinations. Consequently `P_strict(i,j) + P_strict(j,i) + P_tie(i,j) = 1`; the strict diagonal is zero and the tie diagonal is one.

## Retrieval results

{results_table(rows, args.k)}

## Rank stability

{md_table(['Split', 'Median interval width', '90th-percentile width', 'Intervals crossing k', 'P(top k) ≥ 95%', 'P(top k) > 0'], table_rows)}

{interpretation}

![Validation rank intervals](pitt-val_rank_intervals.png)

![Test rank intervals](pitt-test_rank_intervals.png)

![Test selection frequencies](pitt-test_top_k_frequency.png)

Ranking by top-k frequency retrieves {metrics['pitt-test']['top_k_frequency']['hits']} test hazards versus {metrics['pitt-test']['mean']['hits']} for mean ranking, a difference of {metrics['pitt-test']['top_k_frequency']['hits']-metrics['pitt-test']['mean']['hits']:+d} image(s). The corresponding validation counts are {metrics['pitt-val']['top_k_frequency']['hits']} and {metrics['pitt-val']['mean']['hits']}. This is a descriptive comparison of prespecified orderings; the test difference is not used to choose a winner or tune the method.

Wide intervals mean the head ensemble disagrees about placement, even when a single averaged ranking looks precise. P(top k) measures selection stability under these refits; it is **not** the probability the image is a true hazard. An observation may be stably misclassified. Point-3 intervals/frequencies and pairwise probabilities can accompany any downstream ordering without replacing it.

## Artifacts and reproducibility

- `pitt-{{val,test}}_expected_rank.csv` contains per-image rank intervals/frequencies, mean/SD probabilities, labels and metadata; `_top{args.k}.csv` gives the selected images. Comparable files exist for `mean` and `top_k_frequency`.
- `pitt-{{val,test}}_ranks.{{npz,csv}}` retains every observation's 128 midranks. NPZ axes are explicitly named by sample IDs and member IDs.
- `pitt-{{val,test}}_pairwise_probability.csv` is the **full strict outranking matrix**, rows/columns named by sample ID. `_pairwise_tie_probability.csv` retains ties. `_pairwise.npz` includes probabilities and exact counts.
- [Metrics and stability summaries](metrics.json), [provenance](provenance.json); figures are saved as PNG and PDF. Heatmaps of these same probabilities and a pairwise consensus ranking are in [point 4](../point_4/analysis.md).
- Run: `python -m uncertainty_analysis.point_3_rank_distributions --k {args.k}`.

{shared_notes(source, args.k)}

Reference: [supplied methods note, points 3–4](../../reference/uncertainty_in_classifier_rankings.pdf).
'''
    (folder / 'analysis.md').write_text(report)
    print(f'Point 3 complete: results in {folder}', flush=True)


if __name__ == '__main__':
    main()
