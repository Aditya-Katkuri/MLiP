# Exploratory data analysis

These notes cover the data the project pulled onto `sidewalk-cpu` on 2026-09-14. Each write-up
starts with its headline findings and ends with its caveats and source links. The scripts that
produce every number are in `src/eda/`.

| Write-up | Covers | Script | Numbers |
|---|---|---|---|
| [pittsburgh.md](pittsburgh.md) | Project Sidewalk labels, validations and audit coverage; transit, destinations, crashes, 311; equity by census tract | `src/eda/pittsburgh_eda.py` | `/data/eda/pittsburgh/` |
| [rampnet.md](rampnet.md) | rampnet-dataset shards, Stage-1 inputs and the "43%" claim, the paper's gold set, the post-publication benchmark, crop-model datasets, checkpoints | `src/eda/rampnet_eda.py` | `/data/eda/rampnet/` |
| [crops.md](crops.md) | The five validator-AI crop datasets, the five validator models, and the sidewalk-tagger-ai crops | `src/eda/crops_eda.py` | `/data/eda/crops/` |

The Pittsburgh snapshot is fetched by `scripts/fetch_pittsburgh_data.py`, which writes
`/data/datasets/pittsburgh/MANIFEST.json` with the URL, size and sha256 of every file.

## Where the data is

| Path on the VM | Contents | Durable copy |
|---|---|---|
| `/data/datasets/rampnet-dataset/`, `rampnet-benchmark/`, `models/rampnet/` | mirror of blob `datasets/` (474 GB) | blob + Hugging Face |
| `/data/datasets/rampnet-stage1-inputs/`, `rampnet-crop-model-dataset-round{1,2}/`, `models/rampnet-crop-model/` | RampNet pipeline inputs and crop model | Hugging Face only |
| `/data/datasets/sidewalk-validator-ai-dataset-*/`, `models/sidewalk-validator-ai-*/` | validator crops (9.5 GB) and models (18 GB) | Hugging Face only |
| `/data/datasets/pittsburgh/` | dated API snapshot, 68 files | none: the APIs change daily |
| `/data/code/` | RampNet, sidewalk-panorama-tools, sidewalk-cv-2021 clones | GitHub |

## Corrections to the proposal, in one place

1. **Scope.** "334 of 828 km" and "494 km un-audited" count Project Sidewalk's *opened* area. The
   city network is 1,596 km: 271 km (17%) audited, 1,324 km not. 907 km were never opened, and
   153 km of the "828" lies in neighbouring boroughs.
2. **Severity** runs 1–3, not 1–5. NoSidewalk has no severity.
3. **Validation votes.** 48% are Project Sidewalk's AI, and `rawLabels` vote counts include them.
4. **Crop classes.** The `curbramp` crop dataset holds crosswalk crops. Crops exist for
   NoCurbRamp, Obstacle and SurfaceProblem (plus Crosswalk), with 854–5,307 confirmed per class,
   not ~5,100 each. There are no CurbRamp or NoSidewalk crops.
5. **The gold set.** The 0.949 / 0.873 gold set is `manual_labels/` in the RampNet repo. The
   `rampnet-benchmark` in blob is a later out-of-domain benchmark, with recall 0.50–0.77.
6. **"43% never yielded usable data"** is an exact-float-join artifact. The figure is 31.8%.
7. **Held-out Pittsburgh.** Pittsburgh labels trained RampNet's crop model, and 598 Pittsburgh
   crops sit in the validator datasets that all five validator models trained on.
