# Hazard-ranking uncertainty: results index

These analyses use all 128 bootstrap heads from run `20261126T230142Z-92d44f`. Only hazard probabilities are analyzed; subtype predictions are not involved. The source archive SHA-256 is `806c54b8e09749ef5d677ee0dcbf7394793d7f87a9b9818cfef41b845ab000e0`.

## Reports

1. [Point 2: uncertainty-adjusted scores](point_2/analysis.md) — validation-only lambda selection, sensitivity plot, mean and lower-quantile comparators.
2. [Point 3: rank uncertainty](point_3/analysis.md) — expected ranks, 2.5th/97.5th-percentile rank intervals, top-k/first-place frequencies and pairwise probabilities.
3. [Point 4: pairwise consensus](point_4/analysis.md) — full matrices, heatmaps, optimality certificates and consensus orderings.

## Precision at the fixed review budget

The review budget is **46 images**, preserving the earlier precision@46 definition. This is about 12%, not the more recent 10% budget. The exact fraction depends on split size and is recorded in every metric file.

| Method | Validation hazards | Validation precision | Test hazards | Test precision |
|---|---|---|---|---|
| Mean probability | 34/46 | 73.91% | 27/46 | 58.70% |
| Mean − 0 × SD (point 2) | 34/46 | 73.91% | 27/46 | 58.70% |
| 5th probability percentile (point 2 comparator) | 32/46 | 69.57% | 26/46 | 56.52% |
| Expected rank (point 3) | 34/46 | 73.91% | 27/46 | 58.70% |
| Top-k frequency (point 3 comparator) | 34/46 | 73.91% | 28/46 | 60.87% |
| Optimal pairwise consensus (point 4) | 34/46 | 73.91% | 27/46 | 58.70% |

The validation-selected penalty is **λ=0**. Thus the point-2 result reduces to mean probability for this archive. Expected-rank and pairwise-consensus orderings have the same precision at this budget. The top-k-frequency comparator retrieves one additional test hazard, but the validation score does not distinguish it from the mean. This is not a test-tuned method recommendation or evidence of statistical significance.

The exact pairwise solver certifies an optimum for both stored prediction matrices. This optimizes agreement between heads, not true hazard precision; both quantities are reported separately.

## Data and interpretation limits

| Split | Images | Hazards | Non-hazards | Duplicate pixel groups | Groups with conflicting binary labels |
|---|---|---|---|---|---|
| pitt-val | 382 | 157 | 225 | 18 | 6 |
| pitt-test | 380 | 174 | 206 | 12 | 2 |

Rows remain the original evaluation observations; duplicate pixel groups and conflicting labels are retained to preserve comparability with the existing dataset. A shared confident mistake or an annotation conflict is not fixed merely by reducing head variance. The point-3 report explicitly counts confident non-hazard selections.

The 128 probabilities per row are refits on overlapping bootstrap samples of the same 1,276 Pitt training records and share a frozen DINOv3 backbone. They are not 128 independent training datasets. Rank/probability intervals describe this model-fit variability; they are not automatically population confidence intervals or calibrated posterior probabilities. The training export is deliberately excluded from evaluation because it contains in-bag predictions.

## Verification and reproducibility

[verification.json](verification.json) recomputes every exported ranking's hazard counts, validates IDs, raw probabilities, pairwise counts and consensus objective values against the archive, and checks the validation-selected lambda. [artifact_manifest.json](artifact_manifest.json) provides output sizes and SHA-256 hashes. Each point has its own provenance file with complete training configuration, input hash and analysis-code hashes. Small synthetic tests compare the optimizer with exhaustive search and check tie handling and rank identities.

Use the commands in the [directory README](../README.md) to rerun all analyses without Colab or a GPU. [comparison.csv](comparison.csv) is the machine-readable version of the table above.
