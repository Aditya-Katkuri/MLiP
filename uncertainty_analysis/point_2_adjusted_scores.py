"""Point 2: choose a mean-minus-SD penalty using validation precision@k only."""
import numpy as np
import pandas as pd
from .common import (Dataset, parser, order_scores, export_ranking, rank_metrics,
                     save_json, save_csv, setup_plot, save_figure, results_table, shared_notes, pct)

LAMBDA_GRID = np.round(np.arange(81) * .05, 10)


def select_lambda(p, y, ids, k, grid=LAMBDA_GRID):
    """Maximize validation hits; ties choose the smallest nonnegative penalty."""
    mean, sd = p.mean(1), p.std(1, ddof=1)
    rows = []
    for value in sorted(set(float(x) for x in grid)):
        assert value >= 0
        metrics = rank_metrics(y, order_scores(mean - value * sd, ids), k)
        rows.append(dict(lambda_value=value, **metrics))
    winner = min(rows, key=lambda r: (-r['hits'], r['lambda_value']))
    return winner['lambda_value'], pd.DataFrame(rows)


def main():
    args = parser(__doc__).parse_args()
    folder = args.output / 'point_2'
    folder.mkdir(parents=True, exist_ok=True)
    source = Dataset(args.archive)
    val = source.load('pitt-val')
    chosen, sweep = select_lambda(val.p, val.y, val.ids, args.k)
    save_csv(folder / 'validation_lambda_sweep.csv', sweep)
    selection = dict(lambda_value=chosen, grid=LAMBDA_GRID, objective=f'validation precision@{args.k}',
                     tie_break='Smallest lambda among maximal integer hazard counts.',
                     selection_split='pitt-val', validation_hazards=int(val.y.sum()), k=args.k,
                     test_labels_used=False, archive_sha256=source.archive_sha256)
    # Freeze the selection before opening the test predictions/labels.
    save_json(folder / 'selection.json', selection)
    test = source.load('pitt-test')
    rows, metrics, details = [], {}, {}
    for data in [val, test]:
        mean, sd = data.p.mean(1), data.p.std(1, ddof=1)
        scores = {'mean': mean, 'adjusted': mean - chosen * sd,
                  'lower_q05': np.quantile(data.p, .05, axis=1, method='linear')}
        orders = {name: order_scores(score, data.ids) for name, score in scores.items()}
        metrics[data.name] = {}
        for name, score in scores.items():
            metrics[data.name][name] = export_ranking(folder, data, name, orders[name],
                                                     {'ranking_score': score}, args.k)
            rows.append((data.name, {'mean': 'Mean probability', 'adjusted': f'Mean − {chosen:g} × SD',
                                     'lower_q05': '5th probability percentile (fixed)'}[name],
                         metrics[data.name][name]))
        mean_set, adjusted_set = set(orders['mean'][:args.k]), set(orders['adjusted'][:args.k])
        details[data.name] = dict(top_k_overlap=len(mean_set & adjusted_set),
                                 entered_top_k=sorted(data.ids[list(adjusted_set - mean_set)].tolist()),
                                 left_top_k=sorted(data.ids[list(mean_set - adjusted_set)].tolist()),
                                 mean_sd_selected_mean=float(sd[orders['mean'][:args.k]].mean()),
                                 mean_sd_selected_adjusted=float(sd[orders['adjusted'][:args.k]].mean()))
    save_json(folder / 'metrics.json', dict(selection=selection, metrics=metrics, changes=details))
    save_json(folder / 'provenance.json', source.provenance())
    plt = setup_plot()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sweep.lambda_value, 100 * sweep.precision_at_k, marker='.', ms=3)
    ax.axvline(chosen, color='tab:red', linestyle='--', label=f'Selected λ={chosen:g}')
    ax.set(xlabel='Uncertainty penalty λ', ylabel=f'Validation precision@{args.k} (%)',
           title='Validation-only selection of the uncertainty penalty')
    ax.legend(); ax.grid(alpha=.2); save_figure(fig, folder / 'validation_lambda_sweep'); plt.close(fig)
    differences = '\n'.join(f"- **{split}:** adjusted ranking retrieves {m['adjusted']['hits'] - m['mean']['hits']:+d} hazards versus the mean ranking, "
                            f"a {(m['adjusted']['precision_at_k']-m['mean']['precision_at_k'])*100:+.2f} percentage-point difference. "
                            f"{details[split]['top_k_overlap']}/{args.k} selected observations overlap."
                            for split, m in metrics.items())
    best = sweep[sweep.hits == sweep.hits.max()]
    report = f'''# Point 2: uncertainty-adjusted probability ranking

## Method and validation selection

For each observation, average the 128 hazard probabilities and compute their **sample standard deviation** (ddof=1). Rank by `mean − lambda × SD`, highest first. SD is variability between heads, not the standard error of their average; scores are not clipped to [0,1]. Exact score ties use ascending sample ID.

Search the predeclared grid **0 to 4 in steps of 0.05 (81 candidates)**. Select the largest validation hazard count in the top {args.k}; ties choose the smallest lambda. This avoids tuning a second unrelated metric. The selected penalty is **λ = {chosen:g}**. There are {len(best)} grid values tied for the best validation precision, spanning λ={best.lambda_value.min():g} to {best.lambda_value.max():g} (see the complete sweep). This is the optimum within this grid, not a claim about every real-valued lambda. `selection.json` is written before loading test labels.

Also report mean-only ranking (λ=0) and the fixed 5th-percentile probability ranking suggested in the PDF. The latter is a descriptive comparator and does not participate in lambda selection. Test data are evaluated once with these prespecified methods; no test lambda sweep is performed.

## Results

{results_table(rows, args.k)}

{differences}

![Validation lambda sweep](validation_lambda_sweep.png)

## Interpretation

{'The validation-selected penalty is zero: this experiment does not support penalizing uncertainty to improve precision at the chosen budget.' if chosen == 0 else 'Validation selects a positive uncertainty penalty. Compare its held-out test result with mean-only ranking before interpreting this as a general benefit.'}

The selected score retrieves {metrics['pitt-val']['adjusted']['hits']} hazards on validation and {metrics['pitt-test']['adjusted']['hits']} on test. The fixed lower-quantile alternative retrieves {metrics['pitt-val']['lower_q05']['hits']} and {metrics['pitt-test']['lower_q05']['hits']}, respectively; it provides no improvement on these sets.

Each additional correct retrieval changes precision@{args.k} by {100/args.k:.2f} percentage points. The validation maximum is optimistic because it was used to select lambda. These are comparisons on the same small held-out sets, without a sampling confidence interval or a significance claim. Variability across heads is a different quantity from uncertainty in population precision.

Mean ranking maximizes expected hazard yield when the probabilities represent the correct predictive expectation. Subtracting SD expresses a preference for stability; it does not in general maximize expected yield. Shared errors of all heads can remain highly confident.

## Artifacts and reproducibility

- [Frozen selection](selection.json), [complete validation sweep](validation_lambda_sweep.csv), [metrics and selection changes](metrics.json), [input and code hashes](provenance.json).
- `pitt-{{val,test}}_{{mean,adjusted,lower_q05}}.csv` contains every observation, metadata, probability summaries and ranking score; `_top{args.k}.csv` contains exactly the selected review set.
- `validation_lambda_sweep.{{png,pdf}}` contains the figure in raster/vector formats.
- Run from the repository root: `python -m uncertainty_analysis.point_2_adjusted_scores --k {args.k}`.

{shared_notes(source, args.k)}

Reference: [supplied methods note, point 2](../../reference/uncertainty_in_classifier_rankings.pdf).
'''
    (folder / 'analysis.md').write_text(report)
    print(f'Point 2 complete: lambda={chosen:g}; results in {folder}', flush=True)


if __name__ == '__main__':
    main()
