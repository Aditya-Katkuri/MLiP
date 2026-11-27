# 128-head DINO hazard prediction export

Run `notebooks/dino_hazard_bootstrap.ipynb` in Colab with a T4 GPU. The notebook is self-contained.
It trains 128 bootstrap hazard heads and exports raw predictions. Ranking, uncertainty summaries,
decision thresholds, validation/test metrics and plots are left to a separate analysis workflow.

## Fixed configuration and evidence

The notebook's text cell records the complete configuration and its source:

- Frozen DINOv3 ViT-B/16, `facebook/dinov3-vitb16-pretrain-lvd1689m`, revision
  `5931719e67bbdb9737e363e781fb0c67687896bc`; 768-dimensional CLS features.
- Pitt-only training: 1,276 records (527 hazard, 749 non-hazard). Other-city data is not used.
- Adam, learning rate 3e-5, weight decay 0.01, betas (0.9, 0.999), epsilon 1e-8, batch size 64.
- 5,000 updates per head; final checkpoint; unweighted binary cross-entropy with logits.
- Dataset RGB 224x224 images with ImageNet normalization; FP16 backbone, FP32 features/heads.
- 128 separate bootstrap samples and parameter initializations. Bootstrap seeds 1001–1128,
  initialization seeds 2001–2128, and minibatch seeds 3001–3128.

Primary evidence is `hazard.summary` in
`notebooks/colab_results/dino_sidewalk_data_3_runs_extracted/20261119T230431Z-1cd3eb/selection.json`,
configuration ID `68caf0783493ae93`, plus the adjacent `config.json` for the training schedule,
optimizer and preprocessing. The source experiment is `notebooks/dino_sidewalk_data_3.ipynb`.
Historical selection maximized mean validation precision@46 across three seeds, breaking ties
with validation balanced accuracy: 71.7391% and 62.9266%, respectively. Those are historical
selection scores, not measurements of this ensemble. The former 0.5 threshold is not applied.
Source records and original hashes are embedded and exported in `configuration_source/`.

Sampling uses replacement, once per head, and groups exact duplicate pixels using
`output_rgb_pixel_sha256`. Sampling within cluster-label-composition strata preserves 1,276
draws and the original hazard/non-hazard counts while keeping duplicate pairs together.
There are 1,236 training groups, including 40 duplicate pairs. No geographic IDs are available.
All 128 final heads are retained. Training receives no validation/test features or labels.

## Dataset and output storage

No Google Drive mount or Google account authorization is used. Set `DRIVE_FILE_URL` to the
archive file link for `sidewalk-data-3.tar`; anonymous downloads require Anyone with the link /
Viewer access for this one file. `gdown==6.0.0` uses `use_cookies=False`, and SHA-256 is checked.
To keep the file private, upload it with VS Code's Upload to Colab action, choose
`DATA_SOURCE='local_archive'`, and set `LOCAL_ARCHIVE` to its runtime path. Hugging Face and
already-extracted dataset modes are also supported. The gated backbone still needs `HF_TOKEN`.

Results live on the Colab runtime at `/content/dino_hazard_bootstrap_runs/<run-id>/`.
The final cell creates `/content/downloads/<run-id>.zip`. In VS Code, open the Colab sidebar,
expand the runtime files, open `downloads`, right-click the ZIP and choose Download. On older
extensions, update or enable `colab.serverMounting` and use Mount Server to Workspace. Browser
Colab users can use the Files sidebar. This does not mount Drive.

Download the ZIP before resetting/deleting the runtime; saving the notebook does not save
weights or predictions. `RESUME_RUN_DIR` resumes this implementation's local run;
`RESUME_ARCHIVE` restores an uploaded ZIP. Completed heads are verified/reused; interrupted
heads restart. Start a fresh run for this version: old ten-head archives are not compatible.
The notebook attempts to package completed heads on a training interruption.

## Export contract

All three Pittsburgh splits are exported by default:

| File | Rows | Probability columns |
|---|---:|---:|
| `pitt-train_predictions.csv` | 1,276 | 128 |
| `pitt-val_predictions.csv` | 382 | 128 |
| `pitt-test_predictions.csv` | 380 | 128 |

Columns are `p_hazard_head_001` through `p_hazard_head_128`, ordered by member ID, with the same
member occupying the same column everywhere. Rows retain original manifest order. Identifiers,
paths, split, hazard ground truth and provenance fields accompany the predictions.
`*_probabilities.npz` stores float32 `probabilities` (N,128), `sample_ids` (N) and `member_ids` (128).
No means, standard deviations, decisions, ranks or performance metrics are calculated/exported.

`pitt-train_bootstrap_counts.npz` stores counts (1276,128), sample IDs and member IDs. A zero
count identifies a head that omitted that image's duplicate group. Use it for out-of-bag
analysis elsewhere; unfiltered training predictions include in-bag predictions.

Other artifacts:

- `members/head_001/` through `head_128/`: final weights, seeds, bootstrap draws/counts,
  optimization-loss logs, timing and completion checksums.
- `inference/ensemble.pt`: stacked weights for all heads; `backbone/`, preprocessing, source
  configuration, requirements and an offline raw-prediction CLI are included.
- Original split manifests, embedded source, environment, configuration and feature caches.
- `prediction_schema.json`: column and matrix conventions; `bootstrap_members.csv`: seed and
  membership inventory. No head is selected or discarded based on its outputs.

For future images, run:

```bash
python hazard_bootstrap_inference.py --bundle ./inference --images image1.png image2.jpg --output predictions.csv
```

The full bundle works offline and returns only identifiers plus all raw hazard probabilities.
CPU FP32 feature extraction can differ slightly from the T4 FP16 features used for training.
