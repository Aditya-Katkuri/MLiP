# SIFT on sidewalk-data-3

Server search found only older SIFT results (latest November 17 five-class run),
with no existing dataset-3 result or active SIFT job. This is a new training and
evaluation; the previous scripts, worktree, and results are preserved.

## Method and training

Unchanged legacy SIFT extractor: existing 224x224 RGB images converted to grayscale,
40 requested sparse features plus up to 40 dense descriptors on a 32-pixel grid,
80 total cap, at least 5 descriptors. OpenCV can exceed the sparse request on ties;
the original concatenation/truncation behavior is retained. No geometric recrop.
200-word OpenCV kmeans++ codebook, seed 0, one attempt, 20 iterations, epsilon 0.1.
Per-image histograms are L1 normalized. Prototypes are means of training histograms,
L1 normalized again. Distances use symmetric chi-square. No SVM or neural net.

Two predeclared training pools: Pitt only, and pooled Other+Pitt. The pooled variant
is not sequential fine-tuning. Within each pool, five independent binary prototype
pairs are fitted: hazard, surface_and_level_problems, restricted_passage,
missing_tactile_warning, ramp_points_into_traffic. Hazard uses all eligible training
labels; group pairs use only TRUE training hazards, including overlapping group
memberships. Non-hazards never enter the group positive or negative prototypes.

Preserve the original sample caps: up to 500 training examples per task/polarity,
with up to the first 200 contributing to that plan's shared codebook. Candidate
rows are sorted by sample_id then shuffled with random.Random using the saved
task/polarity seed. An image contributes to each applicable prototype, but at most
once to the shared vocabulary. train_selection.csv identifies every contribution,
eligible/selected counts and codebook membership. Actual counts are in run_config.

## Targets, scores, and model selection

Dataset-3 hazard_label/group_* columns, verified against tags, are authoritative;
old feature classes are provenance only. All validation/test images, including
Crosswalk and manually collected ordinary sidewalks, are retained. Missing-tag
rules and conflicting annotations within splits are unchanged. Every fitted image
and every validation/test image is hash checked; no cross-split pixel duplicates.

Each pair's raw score is distance_to_negative minus distance_to_positive. Larger
means more positive. p_* columns use the affine transform (1+margin)/2 solely for
the common evaluator: these are NOT probabilities or fitted calibration. The
four group scores overlap and do not sum to one. Raw distances and margins are
saved too. Independent hazard/group predictions may disagree; the gated pipeline
and hazard-without-group flags expose those cases instead of forcing an argmax.

All model selection uses pitt-val. Hazard: largest precision@46, then largest
balanced accuracy at a validation-selected threshold. Groups: largest mean of
four binary balanced accuracies on ALL true hazards, with an independent cutoff
for each group. Final plan ties prefer Pitt only; cutoff ties choose the lower
threshold. Hazard and group models may select different training pools/codebooks.

Because chi-square margins have a different scale from neural probabilities,
cutoff candidates are all observed validation scores plus 0, 0.5 (nearest
prototype), and the next representable score above the maximum. This differs from
the 0.3/0.5/0.7 grid used for DINO/VLM/CLIP; the selection objectives and evaluation
populations match. Selection and chosen model artifacts are saved BEFORE test
descriptor extraction. There is one deterministic sample/codebook per plan, not
a multi-seed sweep. No prompt or parameter is selected using test performance.

## Metrics and artifacts

All 382 validation and 380 test images are scored. Rank p_hazard descending, then
sample_id ascending; top 46 is never filtered by the hazard threshold. Tag-group
macro balanced accuracy is mean binary BA across all four groups on ALL 174 true
test hazards, regardless of hazard detection. Per-group precision/recall/F1,
specificity, confusion counts, exact match, and gated pipeline metrics are saved.
Fixed-0.5 metrics mean nearest-prototype decisions and are provided separately.
The shared dataset/evaluation helper is the updated VLM code; no API is called.

inputs/, code/, run_config.json, image_inventory.json and opencv_build.txt retain
data, code, software and fit provenance. Each plan has train_selection.csv,
model.npz (vocabulary, five prototype pairs, selected training IDs/histograms),
validation scores/threshold sweep/metrics, and held-out histograms. The root has
validation_model_selection.csv, selection.json, pitt-{val,test}_scores.csv,
*_predictions.csv, *_hazard_ranking.csv, *_top46.csv and complete metrics.
inference/ contains the selected model file(s), label schema and selection rules.

Descriptor caches are atomic NPZ files keyed by the run's input/code fingerprints.
--resume reuses caches/fitted models only if the full binding matches. A run lock
prevents concurrent writers. status.json/run.log track progress. results.json and
final_verification.json are written after saved-model score reproduction, saved-CSV
metric reproduction, training-prototype checks, and final input hash checks pass.

## Server command

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/opt/miniconda/envs/sidewalk/bin/python src/baseline/sift_baseline_data3.py \
  --reference-vlm-run /data/eda/openai_baseline/gpt61_sol_low_sidewalk_data_3_20261120T041605Z \
  --out /data/eda/sift_baseline/NEW_RUN
```

## Run location

Results: `/data/eda/sift_baseline/sidewalk_data_3_20261120T043423Z`.

Audit: `/data/logs/sift-sidewalk-data-3-20261120T043423Z`.

Server tmux attach:

```bash
tmux -L sift-data3-20261120T043423Z attach -t sift-data3
```

Resume after interruption:

```bash
tmux -L sift-data3-20261120T043423Z new-session -d -s sift-data3 /opt/miniconda/envs/sidewalk/bin/python /data/logs/sift-sidewalk-data-3-20261120T043423Z/launch_run.py
```

## Completed evaluation

Verified 2026-11-20T04:35:47.229349+00:00. All 382 validation and 380 test outputs are saved; checks passed.

| Split | Hazard precision@46 | Hazard balanced accuracy | Tag-group macro balanced accuracy |
|---|---:|---:|---:|
| pitt-val | 54.35% (25/46) | 60.42% | 64.09% |
| pitt-test | 50.00% (23/46) | 53.41% | 61.39% |

Validation selected pitt_only for hazard detection and pitt_only for tag groups.

Hazard margin cutoff: 0.04035296962931123. Group margin cutoffs (schema order): [0.005757359020423847, 0.019932508231423007, 0.0055662423469260425, 0.008114071657462363]. Decisions use score >= cutoff; exact ties at the fixed nearest-prototype reference cutoff favor positive.

The 500-example cap is per task/polarity: 1,027 unique Pitt-only fit images and 4,375 unique pooled fit images; 5,207 distinct training records across both variants. Codebooks use only selected training descriptors. Models and training contributions are retained for inference and audit.
