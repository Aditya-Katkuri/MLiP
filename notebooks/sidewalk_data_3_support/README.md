# DINOv3 training on sidewalk-data-3

Notebook: `notebooks/dino_sidewalk_data_3.ipynb`. The original
`notebooks/dino_ft_A_report_rerun.ipynb` is unchanged. The new notebook embeds its
helpers, so it is self-contained when opened with the VS Code Colab extension.

## Start in Colab

1. Upload `/data/datasets/sidewalk-data-3.tar` from sidewalk-cpu to
   `MyDrive/MLiP/datasets/sidewalk-data-3.tar`. It is 2,466,887,680 bytes (2.47 GB).
   SHA-256: `d6c88a2b770baefd9ee34b7c846fe8be573258ba6d6ec7cac27e90e2d9eabbbd`.
   Alternatively select `DATA_SOURCE='huggingface'` and set `HF_DATA_REPO` to the
   repository containing this exact archive. The previous dataset link is not used.
2. Open the new notebook, select a Colab T4 GPU kernel, and Run All.
3. Authorize Google Drive and supply `HF_TOKEN` through Colab Secrets, the environment,
   or the hidden prompt. The token must have access to the gated DINOv3 backbone.
   No token is embedded, printed, or included in saved run configuration.

Outputs go to `MyDrive/MLiP/dino_sidewalk_data_3_runs/<run-id>/`.
Set `RESUME_RUN_DIR` to a previous run to reuse features and completed trials.
An interrupted trial restarts. Settings, dataset hashes, code and feature-extraction
environment must match when resuming. The full Colab sweep completed; its uploaded exports live under
`notebooks/colab_results/` on the server and are excluded from Git.

## Training and selection

Frozen `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision
`5931719e67bbdb9737e363e781fb0c67687896bc`, CLS pooling, five linear sigmoid outputs.
One is hazard; four are independent tag groups. Hazard BCE uses every training image;
group BCE is the mean over four outputs on true hazards. The two losses are summed
without class weighting. Adam state continues across Other→Pitt phases.

84 trials = 2 training orders × 7 learning rates × 2 weight decays × 3 seeds.
Pitt-only uses 5,000 updates. Other→Pitt uses 5,000 Other then 5,000 Pitt updates.
Batch size is 64. All comparisons use final checkpoints. Validation snapshots every
250 updates are for learning curves, not early stopping. The three-seed mean is the
configuration selection score.

- Hazard: maximize validation precision@46; break ties by validation balanced
  accuracy at the threshold maximizing its three-seed mean BA. Integer summed
  top-46 hits determine precision ties exactly. Equal probability ranks are
  ordered by sample_id.
- Tag groups: independently maximize mean macro binary balanced accuracy across
  four groups, on all true validation hazards, regardless of hazard-model decisions.
  Each group's BA = (sensitivity + specificity)/2. This is not four-way softmax
  accuracy or macro positive recall.
- Threshold grid is 0.3, 0.5, 0.7. One hazard threshold and four independent group
  thresholds are chosen using validation means over seeds. Equal threshold scores
  choose the lower threshold. Final configuration ties prefer Pitt-only, then lower
  LR, then lower weight decay.
- All three replicas of both winning configurations are retained. For a single
  deployment model per task, choose the best validation replica within the winning
  configuration using the same task objective at the selected thresholds; smaller
  seed breaks ties. Deployment metrics and configuration seed means are saved separately.
- Test features and metrics are computed only after selection/export. Nothing is
  selected, retuned or retrained using test results.

The new dataset's labels are authoritative; old feature class names do not define
hazard labels. All 277 extra Pitt training screenshots are included. The 85 tagless
old Obstacle/SurfaceProblem records and existing duplicate annotation conflicts
remain as documented in dataset 3. No filtering or tag recomputation changes them.

## Inference artifacts

Preserve the entire run's `inference/` folder:

- `best_hazard.pt`, `best_tag_groups.pt`: selected deployment heads and thresholds.
- `replicas/{hazard,tag_groups}/seed-{0,1,2}.pt`: all selected configuration replicas.
- `backbone/`: complete frozen DINOv3 weights and config, enabling offline inference.
- `selection.json`, `label_schema.json`, `preprocessing.json`, `files_sha256.json`.
- `checkpoint_reload_verification.json`: exported weights checked against validation scores.
- `sidewalk_inference.py`, `requirements.txt`, `README.md`.

Run `python sidewalk_inference.py --bundle /path/to/inference --images photo.png --output predictions.csv`.
Preprocessing is EXIF orientation, RGB, full-image bicubic resize to 224×224 and
saved ImageNet normalization. Group probabilities need not sum to one. The pipeline
returns no groups for a predicted non-hazard. A predicted hazard with no group above
threshold is flagged as `hazard_without_group`, rather than forced into one group.

## Reports and verification

The run saves all validation probabilities and threshold scores, all 28 configuration
summaries, loss/metric curves, complete selection identities, per-image test predictions,
per-group metrics and TP/FP/TN/FN counts, seed means/std, and single deployment metrics.
Gated-pipeline metrics are separate from subtype scores on all true hazards.

The support files mirror the notebook's embedded code. Regenerate the notebook after
editing them with `python notebooks/sidewalk_data_3_support/build_notebook.py`.
Run the synthetic checks from `notebooks/sidewalk_data_3_support/` using
`/data/tools/dino-eval-env/bin/python smoke_checks.py` on sidewalk-cpu.
Verified: metric definitions, exact precision tie-breaking, independent selections,
loss masking, completed-trial reuse, checkpoint export/reload, notebook cell compilation,
the complete dataset loader, and offline image inference with the actual cached
DINOv3 backbone. The last inference test used a synthetic image and untrained test heads;
it was not a real training experiment or a Colab/T4 performance benchmark.

References: [DINOv3 API](https://huggingface.co/docs/transformers/model_doc/dinov3),
[balanced accuracy definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html).
