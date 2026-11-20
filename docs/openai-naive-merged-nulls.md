# OpenAI VLM: processed Pittsburgh test set

Run: `gpt61_sol_low_naive_merged_nulls_pitt_test_20261031T002557Z`.
Results directory: `/data/eda/openai_baseline/gpt61_sol_low_naive_merged_nulls_pitt_test_20261031T002557Z/`.
Execution audit: `/data/logs/openai-naive-merged-nulls-20261031T002557Z/`.

## Data and task

Input: `/data/datasets/proc-sidewalk-data-naive-merged-nulls/pitt-test/`, with ground truth from the adjacent `pitt-test.csv`.
The class column is checked against the class directory. The runner verifies that the CSV covers every eligible image in the directory.
Only Pittsburgh test images are used. Crosswalk remains excluded, matching the previous four-class VLM experiment.

| Class | Evaluated images |
|---|---:|
| CurbRamp | 152 |
| Obstacle | 16 |
| SurfaceProblem | 130 |
| no_obstacles | 70 |
| **Total** | **368** |

The manifest contains 380 images including 12 excluded Crosswalk images.
`no_obstacles` combines 20 reviewed archive images and 50 manually collected screenshots; the preprocessing config merges `no_obstacles_manual_collection` into `no_obstacles`.
The dataset was built from `/data/datasets/sidewalk-data-2` using `configs/preprocess/naive-merged-nulls.yaml` (build revision `ecdf73c2727fe7eaa28928e3805679853cdeee6c`).
Inputs are the stored 224×224 RGB PNGs, resized with bicubic resampling and `aspect: null`. The VLM runner sends those bytes without further image processing.
The source manifest retains `.webp` filenames for some images that preprocessing converted to PNG. The runner uses `image_path` to locate current files, records current filenames in its predictions CSV, and retains the original CSV row in JSONL.
Manual screenshots have no Project Sidewalk label ID; their output `label_id` is blank.

## Model and protocol

- Model: `gpt-6.1-sol`; reasoning effort: `low`; image detail: `high`.
- Prompt, four-class schema, and tie-breaking order are unchanged from `/data/eda/openai_baseline/gpt61_sol_low_pittsburgh_v2/run_config.json`.
- One independent Responses API request per image, with `store=False`. Only the fixed prompt and image bytes are submitted; filenames, paths, ground truth, and annotations remain local.
- Structured output fields: `p_CurbRamp`, `p_Obstacle`, `p_SurfaceProblem`, `p_no_obstacles`. Values must be finite and in [0, 1], and sum to one within 1e-6. These are elicited probability estimates, not calibrated scores or token log probabilities.
- Prediction is argmax; ties use the class order in the table. Refusals, incomplete responses, or invalid outputs stop the run and are recorded.
- Authentication uses the server's `OPENAI_API_KEY` environment variable. No credentials are written to experiment artifacts.

The runner uses the [OpenAI Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create).
The previous experiment evaluated 1,633 images from an earlier dataset; this run uses a different test partition and processed resolution, so its scores describe a different evaluation set.

## Artifacts and metrics

- `run_config.json`: full prompt/schema, requested model/settings, SDK/Python versions, class counts, manifest and image-inventory fingerprints.
- `input_manifest.csv`, `dataset_config.yaml`, `image_inventory.json`, `openai_baseline.py`: input/config/code snapshots and per-image SHA-256 hashes.
- `predictions.jsonl`: durable per-image journal with ground truth, source metadata, predictions, response/model identity, token usage, and errors.
- `pittsburgh_predictions.csv`: current image identity, ground truth (`class` and `y`), predicted class, and four probabilities.
- `results.json`: generated only after completion; accuracy, macro-F1 over all four classes, per-class precision/recall/F1/support, and confusion matrix (rows=true, columns=predicted, in table order). Summary scores retain the previous runner's three-decimal rounding; saved predictions support exact recalculation.
- `run.log`, `status.json`: links to execution output and lifecycle status in the audit directory.
- `README.md`: this experiment description. The audit also contains pre-edit backups, tests, dry-run output, launch metadata, and the runner/test diff.

All 15 offline tests and 18 subtests passed before launch, including input/provenance handling, request contents, probability validation, metrics, and interrupted-run recovery. The dry run validated all 368 eligible inputs without API calls.

## Running and resuming

Run from `/data/repo/MLiP` with the `sidewalk` Python environment:

```bash
/opt/miniconda/envs/sidewalk/bin/python -u \
  src/baseline/openai_baseline.py \
  --data-root /data/datasets/proc-sidewalk-data-naive-merged-nulls \
  --manifest /data/datasets/proc-sidewalk-data-naive-merged-nulls/pitt-test.csv \
  --out /data/eda/openai_baseline/gpt61_sol_low_naive_merged_nulls_pitt_test_20261031T002557Z
```

The actual launch uses an identical code snapshot in the audit directory and runs in **server-side** tmux, socket `openai-vlm-20261031`, session `openai-naive-merged-nulls`.
It survives closing the laptop SSH connection. On `sidewalk-cpu`, inspect it with:

```bash
tmux -L openai-vlm-20261031 attach -t openai-naive-merged-nulls
```

Detach with Ctrl+B then D. If interrupted, restart the audited launcher in server-side tmux; it detects the existing run and adds `--resume`:

```bash
tmux -L openai-vlm-20261031 new-session -d -s openai-naive-merged-nulls \
  /opt/miniconda/envs/sidewalk/bin/python \
  /data/logs/openai-naive-merged-nulls-20261031T002557Z/launch_run.py
```

The API key must be available to the server tmux environment. Resume verifies the input/settings/code fingerprints, preserves successful JSONL predictions, repairs a truncated final record if necessary, rebuilds the CSV from saved successes, and requests only remaining images. A file lock prevents concurrent runs in the same output directory.

## Initial live verification

At 2026-10-31T00:29:47.265266+00:00, 20/368 saved CSV predictions were checked against the JSONL responses, manifest ground truth, and image SHA-256 hashes. All probabilities passed validation; no API errors were recorded. Model access and the API key were confirmed by successful requests. See initial_output_verification.json for the verification snapshot; use the prediction files for current progress.

## Completed run

Finished 2026-10-31T00:48:43.627626+00:00 (exit code 0): all 368 eligible images have saved predictions; no API errors. Accuracy: 254/368 (69.02%); macro-F1: 0.585 across the four classes. Full per-class metrics and confusion matrix are in results.json. Final verification matched the saved predictions to the input manifest and image inventory; see final_verification.json.

## Hazard metrics

Hazards are Obstacle and SurfaceProblem. Rank all 368 images by p_Obstacle + p_SurfaceProblem (descending), breaking ties by image_path ascending. Ground-truth hazard support is 146 (16 Obstacle, 130 SurfaceProblem).

- Top 42 recall: 34/146 = 23.2877%. These 42 contain 5 true Obstacles, 29 true SurfaceProblems, and 8 non-hazards. The cutoff score is 0.94 with no tie. Literal recall@41 is 33/146 = 22.6027%; the requested top-42 definition is recall@42.
- Subtype balanced accuracy among all argmax-predicted hazards, restricted to true hazards: (10/13 + 81/98)/2 = 79.7881%. Of 142 predicted hazards, 111 are true hazards and 31 are non-hazard false positives excluded from this conditional two-subtype metric.
- If subtype accuracy is restricted to the top 42 instead: (5/5 + 21/29)/2 = 86.2069%, over the 34 true hazards retrieved. The 8 non-hazard false positives are excluded from this subtype metric.
- Counting missed hazards as errors instead of conditioning on detection gives (10/16 + 81/130)/2 = 62.4038% balanced accuracy over all ground-truth hazards.

hazard_metrics.json records formulas, counts, confusion matrices, input checksum, and ranking cutoffs; hazard_ranking.csv records every image's rank and score. hazard_metrics.py reproduces the calculations. Conditional balanced accuracies were independently checked with scikit-learn.
