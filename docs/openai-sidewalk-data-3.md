# OpenAI VLM on sidewalk-data-3

Independent zero-shot GPT-6.1 Sol requests, low reasoning, high image detail by
default. The exact requested model/settings are in run_config.json. No VLM weights
are trained. The previous five-class baseline and results are unchanged.

## Task and inputs

One hazard probability and four overlapping conditional group probabilities:
surface_and_level_problems, restricted_passage, missing_tactile_warning,
ramp_points_into_traffic. Group estimates are P(group | hazard, image); they are
not a softmax and are not multiplied by p_hazard. Elicited scores are not token
log probabilities or calibrated confidence guarantees. Full prompt/schema saved.

All 382 pitt-val images are evaluated first; all 380 pitt-test images follow.
Every original feature class, including Crosswalk, is included. Ground truth is
the dataset-3 hazard_label and four group_* columns, checked against the original
tags. Original classes are provenance only. Hand-collected ordinary sidewalks
are non-hazard. Missing tag annotations follow the existing dataset policy;
annotations and their absence are never sent to the model. Existing contradictory
duplicate annotations and tagless records are retained, as in the DINO experiment.

Exact existing 224x224 RGB PNG bytes are sent without additional image processing.
Only the fixed prompt and one image go to the API, store=False. No filenames,
labels, tags, coordinates, class proportions, or prior predictions are transmitted.
Credentials are read from OPENAI_API_KEY, never saved in the artifacts.

## Selection and metrics

Same threshold grid as DINO: 0.3, 0.5, 0.7, selected using pitt-val only. With one
fixed model/prompt, precision@46 does not vary with threshold; hazard balanced
accuracy selects the threshold, with lower threshold breaking exact ties. Each
group's threshold independently maximizes its binary balanced accuracy on ALL
true validation hazards. Selection is saved before the first test request.

Hazard precision@46 ranks ALL images by p_hazard descending, then sample_id
ascending for ties. A threshold never filters the top-46 ranking. Cutoff ties and
possible precision bounds are saved. Per-label precision/recall/F1, specificity,
accuracy, balanced accuracy, and confusion counts are retained without rounding.

Primary subtype score = mean of four binary balanced accuracies, evaluated on ALL
true hazards, irrespective of whether the VLM detected them. This differs from
the historical two-class Obstacle-vs-SurfaceProblem accuracy. Gated pipeline
metrics over all images and exact multi-label match are reported separately.
Group predictions are suppressed on predicted non-hazards only in the gated
pipeline. Hazard-with-no-group cases are flagged, never forced to an argmax.
Fixed-0.5 metrics are saved separately for reference, not used for model selection.

## Files and restart

- inputs/: complete manifests, label schema, labeling/preprocessing rules and build config.
- run_config.json, image_inventory.json, prompt.txt, output_schema.json, runner snapshot.
- predictions.jsonl: durable per-image metadata, raw response, model identity and token usage.
- pitt-{val,test}_probabilities.csv: saved incrementally, with five raw scores and ground truth.
- selection.json, validation_threshold_metrics.json: validation-only decisions.
- pitt-{val,test}_predictions.csv, *_hazard_ranking.csv, *_top46.csv.
- pitt-{val,test}_metrics.json, *_per_group_metrics.csv, *_metrics_fixed_0.5.json.
- results.json, final_verification.json: present only after full completion and checks.
- status.json: progress/ETA; run.log: server tmux execution log.

Resume with the saved runner and the same --out directory plus --resume. Successful
JSONL records are reused, CSVs rebuilt, and only remaining images requested. Input,
code, settings and dependency fingerprints must match. A directory lock prevents
two workers charging for the same run. API errors/refusals/invalid scores are
saved and stop the run. Only an incomplete final journal line is repaired on resume.

## Server commands

```bash
/opt/miniconda/envs/sidewalk/bin/python src/baseline/openai_baseline_data3.py --dry-run
/opt/miniconda/envs/sidewalk/bin/python src/baseline/openai_baseline_data3.py --out /data/eda/openai_baseline/NEW_RUN
```

Use the run’s saved script plus `--resume` to continue an interrupted experiment.

## Active run

Run ID: `gpt61_sol_low_sidewalk_data_3_20261120T041605Z`.

Results: `/data/eda/openai_baseline/gpt61_sol_low_sidewalk_data_3_20261120T041605Z`.

Audit/log: `/data/logs/openai-sidewalk-data-3-20261120T041605Z`.

Attach on the server:

```bash
tmux -L openai-data3-20261120T041605Z attach -t openai-data3
```

Resume after a stopped job using this exact launcher (successful responses are reused):

```bash
tmux -L openai-data3-20261120T041605Z new-session -d -s openai-data3 /opt/miniconda/envs/sidewalk/bin/python /data/logs/openai-sidewalk-data-3-20261120T041605Z/launch_run.py
```
