# Zero-shot CLIP: five-class Pittsburgh comparison

Results: `/data/eda/clip_zero_shot/gpt_comparison_five_class_20261031T025143Z/`.
Audit: `/data/logs/clip-five-class-20261031T025143Z/`.

## Protocol

Evaluate all 380 images in `/data/datasets/proc-sidewalk-data-naive-merged-nulls/pitt-test/`, with ground truth from its `pitt-test.csv`.
The manifest, image roster, labels, and per-image SHA-256 hashes are checked against the completed five-class GPT run at `/data/eda/openai_baseline/gpt61_sol_low_naive_merged_nulls_pitt_test_5class_20261031T011333Z/`.

| Class | Images |
|---|---:|
| Crosswalk | 12 |
| CurbRamp | 152 |
| Obstacle | 16 |
| SurfaceProblem | 130 |
| no_obstacles | 70 |
| **Total** | **380** |

This is **zero-shot CLIP only**, as requested; no training or validation images are used.
Model: `openai/clip-vit-base-patch16`, pinned cached revision `57c216476eefef5ab752ec549e440a49ae4ae5f3`, evaluated on CPU.
Weights and tokenizer/processor come from `/data/hf_cache/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3/`, loaded with local_files_only=True. No API key or network inference service is required.

The previous experiment at `/data/eda/clip_sidewalk_data/` also included a trained linear probe. This run preserves its four existing class prompt pairs and adds two Crosswalk prompts:

1. a street photo of a marked pedestrian crosswalk across a road
2. painted white stripes or parallel lines marking a pedestrian street crossing

Prompts were fixed before inference and are saved in run_config.json. Each text embedding is L2-normalized; two embeddings per class are averaged and normalized again. Image embeddings are L2-normalized. Scores are the softmax of the learned CLIP logit scale times image/text cosine similarities, following the previous zero-shot implementation.
Predictions use argmax, with ties resolved in table order. Full float32-derived probabilities are saved, without four-decimal truncation. These are model scores, not calibrated probabilities.

## Input processing

Use the stored **224×224 RGB PNGs** supplied to GPT. CLIPProcessor performs its standard normalization; no new label-centered crop, letterboxing, or full-frame ablation is applied. Existing old feature caches are not reused because those features represented different input images.
The underlying dataset uses bicubic resizing with aspect=null and merges no_obstacles_manual_collection into no_obstacles (20 reviewed archive images and 50 manual screenshots in this test split). The source CSV can retain a .webp filename after processing creates a PNG; image_path identifies the current file, while original metadata stays in JSONL.

## Metrics

- Five-class accuracy, macro precision/recall/F1, balanced accuracy, weighted F1, per-class metrics, and a 5×5 confusion matrix (true rows, predicted columns, in table order).
- **Hazard precision@46**: rank all 380 images by p_Obstacle + p_SurfaceProblem, descending; count true hazards among exactly 46 selected images. Tie-breaker: image_path ascending, matching the GPT calculation. The cutoff tie size and possible precision range are recorded.
- **Hazard subtype balanced accuracy**: average Obstacle and SurfaceProblem recall among true hazards that were predicted as either hazard class. Non-hazard false positives are excluded from this conditional subtype metric and counted separately.
- Also report subtype balanced accuracy restricted to true hazards in the top 46, matching the alternative scope reported for GPT. If either true subtype has zero support, report the score as undefined.

All metrics are derived from the saved prediction CSV. The metric implementation reproduced GPT's precision@46 (40/46), conditional subtype balanced accuracy (71.9125%), and top-46 subtype balanced accuracy (70.8333%) before this CLIP run. Five offline tests plus four subtests passed, and the dry run confirmed exact GPT input equivalence.

## Artifacts

- run_config.json: model/revision/weight hash, all prompts, normalization/scoring method, batch size, CPU threads, environment versions, dataset and code hashes.
- input_manifest.csv, dataset_config.yaml, image_inventory.json, clip_zero_shot.py: immutable input/config/code snapshots.
- clip_config.json, clip_preprocessor_config.json, clip_tokenizer_config.json: pinned model/processor configuration.
- predictions.jsonl and pittsburgh_predictions.csv: image identity, labels, five probabilities, prediction, and original metadata.
- image_features.npy and text_features.npy: normalized embeddings in inventory and class order, respectively.
- results.json, hazard_metrics_at_46.json, hazard_ranking.csv: classification scores, requested hazard metrics, and complete ranking.
- final_verification.json: output/input consistency, checksums, valid probabilities, and scores reproduced from saved embeddings.
- comparison_with_gpt.json: matching hazard metric definitions for both models.
- run.log and status.json: links to execution records in the audit directory.
- README.md: this description, updated with results after completion.

The old CLIP and GPT predictions are preserved. The audit contains tests, dry-run output, metric parity checks, environment/git information, previous result checksums, the historical CLIP source, and the launch script.

## Execution

Activate the existing conda environment: directly invoking its Python without activation failed to resolve MKL libraries. Activation fixed the import; no package installation or environment modification was needed.

```bash
source /opt/miniconda/etc/profile.d/conda.sh
conda activate sidewalk
python src/baseline/clip_zero_shot.py --out /data/eda/clip_zero_shot/gpt_comparison_five_class_20261031T025143Z
```

The actual run executes a fixed code snapshot in server-side tmux, with environment activation inside the launch shell. While running, attach on sidewalk-cpu with:

```bash
tmux -L clip-fiveclass-20261031T025143Z attach -t clip-zero-shot
```

Detach with Ctrl+B then D. The process survives closing the laptop SSH connection. Use a new output directory for a new evaluation; inference is local and the script preserves existing runs.

## Completed evaluation

Verified all 380 predictions at 2026-10-31T02:55:04.259579+00:00. Accuracy: 19.47%; macro-F1: 0.1958.

Hazard precision@46: 25/46 = 54.35%. Hazard subtype balanced accuracy among all true hazards classified as hazards: 50.69%; restricted to the top 46: 42.26%. See hazard_metrics_at_46.json for confusion matrices, conditioning, false positives, and cutoff ties. comparison_with_gpt.json contains the matching GPT comparison.
