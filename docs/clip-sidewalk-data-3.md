# Zero-shot CLIP on sidewalk-data-3

Pinned openai/clip-vit-base-patch16 revision 57c216476eefef5ab752ec549e440a49ae4ae5f3,
matching the previous zero-shot comparison. No model training or API calls.
All prompts are fixed before inference and saved in prompts.json. For each of
five binary tasks there are two negative and two positive text descriptions.
Each prompt embedding is normalized, averaged within its polarity, and normalized
again. Softmax over that task's negative/positive cosine logits (using the saved
CLIP logit scale) gives its positive score. There is no five-class softmax and no
requirement for the four overlapping group scores to sum to one.

The tasks are hazard, surface_and_level_problems, restricted_passage,
missing_tactile_warning, and ramp_points_into_traffic. Scores are uncalibrated
text-image similarities transformed by paired softmax; group scores support the
conditional task evaluated on true hazards, not calibrated conditional
probabilities. Hazard is scored independently, not forced to the sum/max of the
group scores. Predicted hazards with no positive group are flagged.

## Inputs and evaluation

All 382 pitt-val images first, then all 380 pitt-test images. Dataset-3 tags and
explicit hazard/group columns define ground truth, including images whose old
feature class was Crosswalk. Ordinary hand-collected sidewalks remain non-hazard.
Missing-tag and duplicate-label policies are unchanged. Complete manifests,
schema, labeling/preprocessing rules, image identities and checksums are saved.
The actual run checks exact input images and targets against the new VLM run.

Use stored RGB 224x224 PNGs with CLIP's standard pixel rescaling/normalization;
resize and center crop are explicitly disabled. No label-dependent cropping.
Image paths/labels are never passed to the text encoder. Token lengths are checked
against the model limit; no prompt is silently truncated. Model and processor are
loaded only from the pinned local cache. Weight/config hashes and package versions
are recorded. Old CLIP image caches and old five-class probabilities are not reused.

The dataset contract and metric implementation are imported from
openai_baseline_data3.py, snapshotted with this runner. Importing it does not invoke
OpenAI or require an API key. No change to the running VLM experiment is needed.

Threshold grid: 0.3, 0.5, 0.7, selected exclusively on pitt-val. With one fixed
CLIP model/prompt set, precision@46 is threshold-independent, so hazard balanced
accuracy chooses its threshold. Each group's threshold maximizes binary balanced
accuracy on all true validation hazards. Lower thresholds break exact BA ties.
Selection is saved before test feature extraction; no test-based prompt tuning.

Hazard precision@46 ranks all images by p_hazard descending, then sample_id
ascending. Primary subtype metric is the macro average of four binary balanced
accuracies on ALL true hazards, regardless of predicted hazard status. Per-group
precision/recall/F1, specificity, accuracy and confusion counts are saved. Gated
pipeline metrics on all images are separate. Fixed-0.5 metrics are also saved for
reference. These definitions exactly match the updated VLM and DINO tasks.

## Outputs and restart

run_config.json, prompts.json, selection.json, validation_threshold_metrics.json,
inputs/, model_config/, and runner/evaluation snapshots record the protocol.
predictions.jsonl and pitt-{val,test}_probabilities.csv store original metadata,
ground truth and all five scores. Each split also has *_predictions.csv,
*_hazard_ranking.csv, *_top46.csv, *_metrics.json, *_per_group_metrics.csv and
*_metrics_fixed_0.5.json. status.json and run.log track execution.

prompt_features.npy, text_features.npy, image_features.npy, per-split image
features and cosine similarities allow exact score reproduction without loading
CLIP again. Array orders are in image_inventory.json and text_encoding.json.
feature_chunks/ contains atomically saved, identity-checked inference batches.
--resume checks data/model/code/environment fingerprints, reuses saved chunks,
and rebuilds predictions/metrics. A lock prevents concurrent writers.

results.json and final_verification.json are written only after all 762 images,
saved-embedding score reproduction, and saved-CSV metric reproduction pass.
The original CLIP scripts and results are preserved.

## Server command

```bash
/data/tools/dino-eval-env/bin/python src/baseline/clip_zero_shot_data3.py \
  --reference-vlm-run /data/eda/openai_baseline/gpt61_sol_low_sidewalk_data_3_20261120T041605Z \
  --out /data/eda/clip_zero_shot/NEW_RUN
```

Add `--dry-run` to validate inputs without inference, or `--resume` to reuse a run.

## Run location

Results: `/data/eda/clip_zero_shot/sidewalk_data_3_20261120T042341Z`.

Audit: `/data/logs/clip-sidewalk-data-3-20261120T042341Z`.

Server tmux attach:

```bash
tmux -L clip-data3-20261120T042341Z attach -t clip-data3
```

Resume after interruption:

```bash
tmux -L clip-data3-20261120T042341Z new-session -d -s clip-data3 /data/tools/dino-eval-env/bin/python /data/logs/clip-sidewalk-data-3-20261120T042341Z/launch_run.py
```

## Completed evaluation

Verified 2026-11-20T04:24:51.492174+00:00. All 762 predictions and embeddings were saved; final verification passed.

| Split | Hazard precision@46 | Hazard balanced accuracy | Tag-group macro balanced accuracy |
|---|---:|---:|---:|
| pitt-val | 34.78% (16/46) | 49.71% | 56.74% |
| pitt-test | 41.30% (19/46) | 48.67% | 51.91% |

Selected hazard threshold: 0.3. Group thresholds (schema order): [0.5, 0.5, 0.5, 0.3]. All thresholds came from validation.

Tag-group scores average four binary balanced accuracies on all true hazards (157 validation, 174 test). Same CLIP weight checksum as the previous five-class baseline.
