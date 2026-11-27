"""Point 4: outranking heatmaps and a Kemeny consensus with optimality bounds."""
import numpy as np
import pandas as pd
from .common import (Dataset, SPLITS, parser, order_scores, export_ranking, pairwise_counts,
                     pairwise_exports, save_json, save_csv, setup_plot, save_figure,
                     results_table, shared_notes, md_table)
from .consensus_solver import consensus, agreement


def plot_matrix(folder, split, probability, order, k):
    plt = setup_plot()
    matrix = probability[np.ix_(order, order)].copy()
    np.fill_diagonal(matrix, np.nan)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, origin='upper', vmin=0, vmax=1, cmap='viridis', interpolation='nearest',
                   extent=(.5, len(order)+.5, len(order)+.5, .5))
    ax.axhline(k+.5, color='white', ls='--', lw=.8)
    ax.axvline(k+.5, color='white', ls='--', lw=.8)
    ax.set(xlabel='Column observation: consensus rank', ylabel='Row observation: consensus rank',
           title=f'{split}: P(row outranks column), 128 heads')
    fig.colorbar(im, ax=ax, label='Strict outranking frequency')
    save_figure(fig, folder / (split+'_pairwise_heatmap')); plt.close(fig)
    size = min(20, len(order))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(matrix[:size, :size], vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
    ax.set_xticks(range(size), range(1, size+1)); ax.set_yticks(range(size), range(1, size+1))
    for i in range(size):
        for j in range(size):
            if i != j:
                value = matrix[i,j]
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', fontsize=6.5,
                        color='white' if value < .55 else 'black')
    ax.set(xlabel='Column: consensus rank', ylabel='Row: consensus rank',
           title=f'{split}: top {size} observations, strict outranking frequencies')
    fig.colorbar(im, ax=ax, shrink=.8); save_figure(fig, folder / (split+'_top20_heatmap')); plt.close(fig)


def main():
    p = parser(__doc__)
    p.add_argument('--solver-seconds', type=float, default=240,
                   help='Maximum MILP search budget per split; feasible heuristic retained on timeout.')
    args = p.parse_args()
    assert args.solver_seconds >= 0
    folder = args.output / 'point_4'; folder.mkdir(parents=True, exist_ok=True)
    source = Dataset(args.archive)
    rows, objectives, metrics, pair_stats = [], [], {}, {}
    for split in SPLITS:
        data = source.load(split)
        wins, ties = pairwise_counts(data.p)
        pair_stats[split] = pairwise_exports(folder, data, wins, ties)
        expected = 1 + (wins.T + .5*ties).sum(1)/data.p.shape[1] - .5
        borda = order_scores(expected, data.ids, descending=False)
        print(f'{split}: optimizing {len(data.ids)} observations...', flush=True)
        order, certificate = consensus(wins, data.ids, borda, args.solver_seconds)
        denominator = data.p.shape[1] * len(data.ids) * (len(data.ids)-1) / 2
        tied_constant = float(np.triu(ties, 1).sum()) / 2
        certificate.update(k=args.k, time_budget_seconds=args.solver_seconds,
                           head_pair_comparisons=int(denominator), score_tie_half_credit_constant=tied_constant,
                           agreement_fraction_half_ties=(certificate['agreement']+tied_constant)/denominator,
                           upper_bound_fraction_half_ties=(certificate['upper_bound']+tied_constant)/denominator,
                           top_k_overlap_borda=len(set(order[:args.k])&set(borda[:args.k])),
                           full_order_positions_changed_borda=int((order!=borda).sum()))
        save_json(folder/(split+'_solver.json'), certificate)
        extra = dict(expected_rank=expected, pairwise_borda_score=(wins+.5*ties).sum(1)/data.p.shape[1]-.5)
        methods = dict(mean=order_scores(data.p.mean(1),data.ids), expected_rank=borda, consensus=order)
        metrics[split] = {}
        for name, ranking in methods.items():
            metrics[split][name] = export_ranking(folder, data, name, ranking, extra, args.k)
            rows.append((split, name.replace('_',' '), metrics[split][name]))
            score = agreement(wins,ranking)
            objectives.append([split,name,score,f'{100*(score+tied_constant)/denominator:.5f}%',
                               'Yes' if name=='consensus' and certificate['optimal'] else ('Unproven' if name=='consensus' else '—')])
        mapping = pd.DataFrame(dict(consensus_rank=np.arange(1,len(order)+1), sample_id=data.ids[order],
                                    image_path=data.frame.image_path.to_numpy()[order]))
        save_csv(folder/(split+'_heatmap_index.csv'),mapping)
        plot_matrix(folder,split,wins/data.p.shape[1],order,args.k)
        print(f'{split}: agreement={certificate["agreement"]}, bound={certificate["upper_bound"]}, optimal={certificate["optimal"]}',flush=True)
    save_json(folder/'metrics.json',dict(metrics=metrics,pairwise_summary=pair_stats,k=args.k))
    save_json(folder/'provenance.json',source.provenance())
    certificates = {}
    import json
    for split in SPLITS:
        certificates[split]=json.loads((folder/(split+'_solver.json')).read_text())
    certificate_table = md_table(['Split','Components','Largest component','Feasible objective','Upper bound','Unclosed gap','Certified optimal'],
        [[s,c['n_components'],max(c['component_sizes']),c['agreement'],f"{c['upper_bound']:.3f}",
          f"{c['absolute_gap']:.3f}",'Yes' if c['optimal'] else 'No'] for s,c in certificates.items()])
    implications='\n'.join(f"- **{s}:** consensus changes {c['full_order_positions_changed_borda']} positions relative to expected-rank/Borda ordering; "
                          f"the top-{args.k} sets overlap in {c['top_k_overlap_borda']} images. "
                          f"Consensus retrieves {metrics[s]['consensus']['hits'] - metrics[s]['expected_rank']['hits']:+d} hazards compared with Borda."
                          for s,c in certificates.items())
    report=f'''# Point 4: pairwise outranking and consensus ordering

## Pairwise probabilities and heatmaps

For every ordered pair (i,j), compute the fraction of the **same 128 heads** in which `p_i > p_j`. This preserves dependence between images' predictions. Save strict probability and tie probability separately. A pair with 0.5 preference after half-credit for ties is unresolved by this ensemble; it should not be presented as a confident ordering.

Pairs tied in every head have **strict probabilities zero in both directions**; this means identical predicted scores, not evidence that both observations lose to each other. There are {pair_stats['pitt-val']['pairs_tied_in_all_heads']} such unordered pairs in validation and {pair_stats['pitt-test']['pairs_tied_in_all_heads']} in test. The companion tie matrix distinguishes them from genuine one-sided preferences.

The full matrices have one row/column per observation, with exact sample IDs in CSV headers. The heatmaps reorder both axes by the consensus ranking, use a shared [0,1] color scale, and mask the diagonal (strict self-outranking is zero). Dashed lines mark the top-{args.k} boundary. `*_heatmap_index.csv` maps each plotted position to its sample ID and image path. Enlarged top-20 plots print probabilities inside cells; full-resolution numeric values are in CSV/NPZ.

![Validation pairwise probabilities](pitt-val_pairwise_heatmap.png)

![Test pairwise probabilities](pitt-test_pairwise_heatmap.png)

![Test top-20 pairwise probabilities](pitt-test_top20_heatmap.png)

## Ranking objective and algorithm

Choose a permutation π maximizing `sum_(a<b) C[π[a], π[b]]`, where C is the integer number of heads preferring the row observation. This is a **Kemeny consensus**, maximizing expected pairwise agreement (equivalently minimizing expected pairwise disagreements). Adding half-credit for probability ties adds the same constant to every permutation, so it does not change the optimum. Expected-rank/Borda ordering optimizes a different positional criterion and is a useful starting point, not a general solution to this objective.

Pairwise majority preferences can cycle, so sorting them directly does not always yield a consistent order. The implementation first decomposes the majority graph into strongly connected components; pairwise ties are edges in both directions. Cross-component preferences then form an acyclic order and can all be satisfied. Each component starts with Borda ordering and deterministic improving insertions.

For nontrivial components, a binary linear-ordering program defines x(i,j)=1 when i precedes j. Constraints `0 ≤ x(i,j)+x(j,k)−x(i,k) ≤ 1` eliminate directed triangles. Rather than load all O(n³) constraints, the solver adds violated triangles iteratively. A feasible insertion ranking is retained throughout. The mixed-integer relaxation's dual bound supplies an **upper bound** on the attainable agreement; an acyclic optimum, or an integer objective gap below one, certifies optimality. The search budget is {args.solver_seconds:g} seconds per split. On timeout the script reports its feasible best effort and a valid remaining gap rather than calling it optimal. Fixed sample-ID ordering governs deterministic heuristic ties. Multiple equally optimal permutations may exist.

## Optimization results

{certificate_table}

Objective units above are integer head-pair agreements. The reported upper bound bounds the best possible ordering on these stored predictions, not performance on true hazards.

{md_table(['Split','Ranking','Strict agreements','Agreement with half-ties','Certified optimal'],objectives)}

## Retrieval results and interpretation

{results_table(rows,args.k)}

{implications}

The ranking is selected entirely from predictions. Hazard labels are consulted only for these final retrieval metrics. Maximizing agreement across heads is **not** the same objective as maximizing hazard yield in the first {args.k}; an optimal consensus need not beat mean ranking on precision. A high global pairwise-agreement fraction can also hide instability near the review boundary, since most pairs are far apart in rank. Use the point-3 rank intervals and top-k frequencies alongside the heatmap.

## Artifacts and reproducibility

- `pitt-{{val,test}}_consensus.csv` and `_consensus_top{args.k}.csv`: final ranking, original image paths/labels and probability summaries.
- `pitt-{{val,test}}_pairwise_probability.csv`, `_pairwise_tie_probability.csv`, `_pairwise.npz`: full matrices, exact counts and ID axes (same definition as point 3).
- `pitt-{{val,test}}_solver.json`: component sizes, every solver round, timing, objective and optimality bound; `*_heatmap_index.csv` identifies plotted observations.
- [Retrieval metrics](metrics.json), [provenance](provenance.json); all figures have PNG and vector PDF versions.
- Run: `python -m uncertainty_analysis.point_4_pairwise_consensus --k {args.k} --solver-seconds {args.solver_seconds:g}`.

{shared_notes(source,args.k)}

References: [supplied note, point 4](../../reference/uncertainty_in_classifier_rankings.pdf); [Kemeny ranking algorithms and computational complexity](https://aamas.csc.liv.ac.uk/Proceedings/aamas09/pdf/01_Full%20Papers/11_25_140_FP_0418.pdf); [SciPy mixed-integer programming interface and bound definitions](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html).
'''
    (folder/'analysis.md').write_text(report)
    print(f'Point 4 complete: results in {folder}',flush=True)


if __name__ == '__main__':
    main()
