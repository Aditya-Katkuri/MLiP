# Hazard-ranking uncertainty analysis

Post-processing of the 128 bootstrapped DINO hazard-head predictions. These scripts implement points **2, 3 and 4** of the [supplied methods note](reference/uncertainty_in_classifier_rankings.pdf). They do not retrain models or need a GPU, Google Drive, API credentials or a notebook.

## Results and separate reports

The [combined report](latex/uncertainty_analysis.pdf) and [LaTeX source](latex/uncertainty_analysis.tex) describe bootstrap training, the three analyses, validation-based selection and subsequent test results. [Report sources and build instructions](latex/README.md) include the standalone versions.

- [Results index and comparison table](results/README.md)
- [Point 2: mean minus lambda × SD](results/point_2/analysis.md)
- [Point 3: rank intervals and selection frequencies](results/point_3/analysis.md)
- [Point 4: pairwise probabilities, heatmaps and optimal consensus](results/point_4/analysis.md)

Each point has a separate script, Markdown analysis, complete rankings, top-46 review CSVs, metrics JSON, figures (PNG and vector PDF), and provenance. Points 3 and 4 also export the full pairwise matrices with sample-ID axes. The PDFs are editable vector figures.

## Input and model

Default archive: `/data/repo/MLiP/notebooks/colab_results/ensemble/128_bootstrap.zip` (run `20261126T230142Z-92d44f`). It remains intact; the loader reads only relevant ZIP members and validates their stored SHA-256 hashes. Prediction CSVs, probability NPZs and embedded dataset manifests must agree exactly on IDs and labels.

The archive contains 128 independently fitted linear hazard heads over frozen DINOv3 ViT-B/16 features. They use Pitt-only training: 1,276 records (527 hazard, 749 non-hazard), bootstrap resampling of exact-pixel clusters stratified by binary-label composition, Adam learning rate 3e-5, weight decay 0.01, batch size 64, 5,000 updates, unweighted binary cross-entropy, final-update weights. Full configuration and the original configuration-selection source are preserved in each report's `provenance.json`.

Analysis uses **382 validation images (157 hazards)** and **380 test images (174 hazards)**. Training predictions are not evaluated because they include in-bag fits. Image paths in the CSVs resolve relative to `/data/datasets/sidewalk-data-3/`. Original observations, including duplicates and annotation conflicts, are retained.

**Review budget: k=46**, matching the earlier precision@46 metric: 12.0419% of validation and 12.1053% of test. This is intentionally a fixed image count, not a rounded percentage calculation. Only point 2 fits a parameter, using validation labels exclusively. Point 3/4 rankings use predicted probabilities alone.

## Run on the server

From `/data/repo/MLiP`, use the existing analysis environment:

```bash
/data/tools/dino-eval-env/bin/python -m uncertainty_analysis.run_all
```

Or run separately:

```bash
/data/tools/dino-eval-env/bin/python -m uncertainty_analysis.point_2_adjusted_scores
/data/tools/dino-eval-env/bin/python -m uncertainty_analysis.point_3_rank_distributions
/data/tools/dino-eval-env/bin/python -m uncertainty_analysis.point_4_pairwise_consensus --solver-seconds 240
/data/tools/dino-eval-env/bin/python -m uncertainty_analysis.verify_results
/data/tools/dino-eval-env/bin/python -m uncertainty_analysis.run_all --summarize-only
```

All scripts accept `--archive PATH`, `--output DIRECTORY` and `--k 46`; the exact archived 128-member schema is validated. Results default to `uncertainty_analysis/results/`. Rerunning overwrites derived files in that output directory; use a distinct `--output` to preserve an earlier analysis. Dependencies are listed in [requirements.txt](requirements.txt).

For long runs you can use a **server-side** tmux shell. The original computation was launched under server tmux session `uncertainty-analysis`; launch logs are under `/data/logs/uncertainty-ranking-analysis-20261126/`.

## Implementation and checks

- `common.py`: verified loading, ranking metrics, tie-aware pairwise counts, CSV/NPZ exports and plotting helpers.
- `point_2_adjusted_scores.py`: fixed lambda grid 0–4 in 0.05 steps; maximize validation top-46 hits; choose the smallest tied lambda. Save selection before loading test labels.
- `point_3_rank_distributions.py`: within-head midranks; empirical 2.5/97.5% intervals; fractional tie allocation for P(first) and P(top k); expected-rank/Borda and top-k-frequency orderings.
- `point_4_pairwise_consensus.py` / `consensus_solver.py`: full strict outranking matrix, exact tie probabilities, SCC decomposition, improving insertions and triangle-cut mixed-integer optimization. A timeout returns a feasible ranking with a valid upper bound; global optimality is reported only when certified.
- `verify_results.py`: recompute metrics and objectives from raw predictions, check saved IDs/matrices, and hash outputs.

Run meaningful mathematical checks with:

```bash
/data/tools/dino-eval-env/bin/python -m pytest -q uncertainty_analysis/tests
```

Tests include matched-head comparisons, probability ties, expected-rank/Borda equivalence, validation selection, cyclic majorities, exact agreement against exhaustive permutations, row-order invariance and timeout bounds. No analysis is selected using test performance.
