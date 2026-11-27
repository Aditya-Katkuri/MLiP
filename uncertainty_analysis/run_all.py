"""Run all three analyses, audit artifacts, and generate a compact results index."""
import json
import subprocess
import sys
from .common import parser, save_csv, md_table, pct
import pandas as pd


def write_summary(output, k):
    results = {p:json.loads((output/p/'metrics.json').read_text()) for p in ['point_2','point_3','point_4']}
    audit = json.loads((output/'verification.json').read_text())
    selection = results['point_2']['selection']
    source = json.loads((output/'point_2/provenance.json').read_text())
    methods = [('Mean probability','point_2','mean'),
               (f'Mean − {selection["lambda_value"]:g} × SD (point 2)','point_2','adjusted'),
               ('5th probability percentile (point 2 comparator)','point_2','lower_q05'),
               ('Expected rank (point 3)','point_3','expected_rank'),
               ('Top-k frequency (point 3 comparator)','point_3','top_k_frequency'),
               ('Optimal pairwise consensus (point 4)','point_4','consensus')]
    rows, records = [], []
    for name,point,method in methods:
        val,test = [results[point]['metrics'][s][method] for s in ['pitt-val','pitt-test']]
        rows.append([name,f'{val["hits"]}/{k}',pct(val['precision_at_k']),f'{test["hits"]}/{k}',pct(test['precision_at_k'])])
        records.append(dict(method=name,point=point,validation_hits=val['hits'],test_hits=test['hits'],
                            validation_precision_at_k=val['precision_at_k'],test_precision_at_k=test['precision_at_k'],k=k))
    save_csv(output/'comparison.csv',pd.DataFrame(records))
    annotation = md_table(['Split','Images','Hazards','Non-hazards','Duplicate pixel groups','Groups with conflicting binary labels'],
                         [[s,a['n_images'],a['hazards'],a['n_images']-a['hazards'],a['duplicate_pixel_groups'],a['conflicting_pixel_groups']]
                          for s,a in audit['splits'].items()])
    report=f'''# Hazard-ranking uncertainty: results index

These analyses use all 128 bootstrap heads from run `{source['run_id']}`. Only hazard probabilities are analyzed; subtype predictions are not involved. The source archive SHA-256 is `{source['archive_sha256']}`.

## Reports

1. [Point 2: uncertainty-adjusted scores](point_2/analysis.md) — validation-only lambda selection, sensitivity plot, mean and lower-quantile comparators.
2. [Point 3: rank uncertainty](point_3/analysis.md) — expected ranks, 2.5th/97.5th-percentile rank intervals, top-k/first-place frequencies and pairwise probabilities.
3. [Point 4: pairwise consensus](point_4/analysis.md) — full matrices, heatmaps, optimality certificates and consensus orderings.

## Precision at the fixed review budget

The review budget is **{k} images**, preserving the earlier precision@46 definition. This is about 12%, not the more recent 10% budget. The exact fraction depends on split size and is recorded in every metric file.

{md_table(['Method','Validation hazards','Validation precision','Test hazards','Test precision'],rows)}

The validation-selected penalty is **λ={selection['lambda_value']:g}**. Thus the point-2 result reduces to mean probability for this archive. Expected-rank and pairwise-consensus orderings have the same precision at this budget. The top-k-frequency comparator retrieves one additional test hazard, but the validation score does not distinguish it from the mean. This is not a test-tuned method recommendation or evidence of statistical significance.

The exact pairwise solver certifies an optimum for both stored prediction matrices. This optimizes agreement between heads, not true hazard precision; both quantities are reported separately.

## Data and interpretation limits

{annotation}

Rows remain the original evaluation observations; duplicate pixel groups and conflicting labels are retained to preserve comparability with the existing dataset. A shared confident mistake or an annotation conflict is not fixed merely by reducing head variance. The point-3 report explicitly counts confident non-hazard selections.

The 128 probabilities per row are refits on overlapping bootstrap samples of the same 1,276 Pitt training records and share a frozen DINOv3 backbone. They are not 128 independent training datasets. Rank/probability intervals describe this model-fit variability; they are not automatically population confidence intervals or calibrated posterior probabilities. The training export is deliberately excluded from evaluation because it contains in-bag predictions.

## Verification and reproducibility

[verification.json](verification.json) recomputes every exported ranking's hazard counts, validates IDs, raw probabilities, pairwise counts and consensus objective values against the archive, and checks the validation-selected lambda. [artifact_manifest.json](artifact_manifest.json) provides output sizes and SHA-256 hashes. Each point has its own provenance file with complete training configuration, input hash and analysis-code hashes. Small synthetic tests compare the optimizer with exhaustive search and check tie handling and rank identities.

Use the commands in the [directory README](../README.md) to rerun all analyses without Colab or a GPU. [comparison.csv](comparison.csv) is the machine-readable version of the table above.
'''
    (output/'README.md').write_text(report)


def main():
    p=parser(__doc__)
    p.add_argument('--solver-seconds',type=float,default=240)
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args()
    if not args.summarize_only:
        for module in ['point_2_adjusted_scores','point_3_rank_distributions','point_4_pairwise_consensus','verify_results']:
            command=[sys.executable,'-m','uncertainty_analysis.'+module,'--archive',str(args.archive),
                     '--output',str(args.output),'--k',str(args.k)]
            if module=='point_4_pairwise_consensus':
                command += ['--solver-seconds',str(args.solver_seconds)]
            subprocess.run(command,check=True)
    write_summary(args.output,args.k)
    # Include the freshly generated index/table in the final artifact inventory.
    subprocess.run([sys.executable,'-m','uncertainty_analysis.verify_results','--archive',str(args.archive),
                    '--output',str(args.output),'--k',str(args.k)],check=True,stdout=subprocess.DEVNULL)
    print(f'Complete: {args.output / "README.md"}')


if __name__=='__main__':
    main()
