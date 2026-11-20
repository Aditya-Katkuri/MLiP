# OpenAI VLM: Pittsburgh test set with Crosswalk

Run: `gpt61_sol_low_naive_merged_nulls_pitt_test_5class_20261031T011333Z`.
Results: `/data/eda/openai_baseline/gpt61_sol_low_naive_merged_nulls_pitt_test_5class_20261031T011333Z/`.
Execution audit: `/data/logs/openai-five-class-20261031T011333Z/`.

## Data and change from the previous run

Evaluate **all 380 images** in `/data/datasets/proc-sidewalk-data-naive-merged-nulls/pitt-test/` using the adjacent `pitt-test.csv` for ground truth.
This reruns every image with a five-class prompt and output schema. The previous four-class predictions remain in `/data/eda/openai_baseline/gpt61_sol_low_naive_merged_nulls_pitt_test_20261031T002557Z/`.

| Class | Images |
|---|---:|
| Crosswalk | 12 |
| CurbRamp | 152 |
| Obstacle | 16 |
| SurfaceProblem | 130 |
| no_obstacles | 70 |
| **Total** | **380** |

Inputs are the existing 224×224 RGB PNGs, sent without further processing by the runner. The saved preprocessing config uses bicubic resizing, `aspect: null`, and merges `no_obstacles_manual_collection` into `no_obstacles`.
The 70 no-obstacle images comprise 20 reviewed archive negatives and 50 manual screenshots. The processed dataset derives from `/data/datasets/sidewalk-data-2`, build revision `ecdf73c2727fe7eaa28928e3805679853cdeee6c`.
Current image paths are used even when the CSV filename column retains an original .webp extension. Original source rows remain in JSONL; screenshots without annotation IDs have blank label_id.

## Model and classification protocol

- Model: `gpt-6.1-sol`; reasoning: `low`; image detail: `high`, matching the preceding run.
- One independent Responses API call per image with `store=False`. Only the prompt and image bytes are submitted; labels and filenames stay local.
- Five mutually exclusive classes in table order. Crosswalk is a pedestrian roadway crossing, often marked with stripes or parallel lines. The prompt distinguishes the crossing from a curb ramp and excludes both from no_obstacles when they are the main feature.
- Strict output fields: `p_Crosswalk`, `p_CurbRamp`, `p_Obstacle`, `p_SurfaceProblem`, `p_no_obstacles`. Values must be finite, between 0 and 1, and sum to one within 1e-6. They are elicited estimates rather than calibrated scores or token log probabilities.
- The predicted class is argmax; ties use table order. All five classes enter evaluation, including Crosswalk. The full prompt/schema are saved in run_config.json and prompt changes in the audit's prompt.diff.
- Authentication uses the server environment's OPENAI_API_KEY. Credentials are not written to experiment artifacts.

## Outputs and metrics

- `run_config.json`: full prompt/schema, requested model, settings, versions, counts, input and code fingerprints.
- `input_manifest.csv`, `dataset_config.yaml`, `image_inventory.json`, `openai_baseline.py`: input/config/code snapshots and image hashes.
- `predictions.jsonl`: durable responses, probabilities, image identity/hash, source annotations, response model identity, usage, and any errors.
- `pittsburgh_predictions.csv`: image identity, ground truth, predicted class, and all five probabilities.
- `results.json`: final accuracy, macro-F1 over **all five classes**, per-class precision/recall/F1/support, and 5×5 confusion matrix (true rows, predicted columns, in table order). Summary metrics use three-decimal rounding, as before.
- `hazard_metrics.json`, `hazard_ranking.csv`: generated automatically after classification. Hazards are Obstacle + SurfaceProblem; Crosswalk, CurbRamp, and no_obstacles are non-hazards. Rank every image by p_Obstacle + p_SurfaceProblem, with image_path as the tie-breaker. Report recall@42 (the requested top-42 definition), literal recall@41, and subtype balanced accuracy among true hazards predicted as hazards, both overall and within the top 42. Non-hazard false positives are counted separately and excluded from subtype accuracy.
- `summarize_run.py`, `final_verification.json`: reproducible metric calculation and final validation of CSV/JSONL, input identity, image checksums, probability vectors, and the confusion matrix.
- `run.log`, `status.json`: links to the audit log and lifecycle status. `README.md` documents the experiment and is updated with final results on completion.

Before launch, 15 offline tests and 19 subtests passed, including Crosswalk inclusion, all five output fields, refusal of old four-class outputs, processed image handling, request contents, metrics, and interruption recovery. A dry run validated all 380 images. The audit records the environment, git state, before/after code, checks, exact command, and launch times.

## Execution and resume

The actual launcher uses a fixed runner snapshot and server-side tmux. It survives closing the laptop's SSH connection. To attach on sidewalk-cpu:

```bash
tmux -L openai-vlm-fiveclass-20261031T011333Z attach -t openai-five-class
```

Detach with Ctrl+B then D. The API key must be available in the server tmux environment. After interruption, the same audited launcher resumes saved predictions and regenerates final metrics:

```bash
tmux -L openai-vlm-fiveclass-20261031T011333Z new-session -d -s openai-five-class \
  /opt/miniconda/envs/sidewalk/bin/python \
  /data/logs/openai-five-class-20261031T011333Z/launch_run.py
```

The launch command is recorded in status.json. The runner supports `--resume`, checks input/settings/code fingerprints, uses a directory lock, and skips all successful JSONL entries. Each success is flushed and fsynced; a stale CSV or truncated final journal line can be repaired on resume.

## Initial live verification

At 2026-10-31T01:17:29.020497+00:00, 12/380 predictions were verified in CSV and JSONL, including 12 Crosswalk inputs. All five probabilities, ground truth, image hashes, and response settings passed validation; no API errors. Key and model access confirmed. See initial_output_verification.json for this snapshot; prediction files show current progress.

## Completed run

Verified 2026-10-31T01:37:49.447817+00:00: 380 predictions, 0 error records. Accuracy: 224/380 (58.95%); macro-F1: 0.470.

Hazard recall@42: 24.6575%. Conditional subtype balanced accuracy among all predicted hazards: 71.9125%; within the top 42: 68.7500%. Non-hazard false positives are excluded from subtype scoring. See results.json, hazard_metrics.json, hazard_ranking.csv, and final_verification.json for full counts and definitions.

## Updated requested metrics: precision@46

Hazards remain Obstacle and SurfaceProblem. Rank all 380 images by p_Obstacle + p_SurfaceProblem, descending, with image_path ascending as the tie-breaker. Precision@46 uses exactly 46 images: 40/46 = 86.9565%. Subtype balanced accuracy among all argmax-predicted hazards, conditioned on their true labels also being hazards, is 71.9125%. Restricted to true hazards within the top 46, subtype balanced accuracy is 70.8333%. Non-hazard false positives are counted in precision and excluded from two-subtype accuracy. Counts, confusion matrices, and cutoff ties are recorded in hazard_metrics_at_46.json; hazard_top46.csv lists the selected images. Earlier recall@41/@42 figures are retained as historical analyses.

Precision@46 cutoff ties: six images share hazard probability 0.94, with two positions available. The documented image_path tie-break selects two true hazards, yielding 40/46 (86.9565%). Other valid selections among the tied images yield 38/46 to 40/46 (82.6087%–86.9565%); keep the tie policy fixed when comparing models.
