# EDA: validator-AI crop datasets and validator models

Generated 2026-09-14 by [`src/eda/crops_eda.py`](../../src/eda/crops_eda.py). Numbers are in
`/data/eda/crops/summary.json`, tables in `/data/eda/crops/tables/`, and figures in
`/data/eda/crops/figures/`.

## Headline findings

1. **The `curbramp` crop dataset holds crosswalk crops, so there is no CurbRamp crop set.**
   `sidewalk-validator-ai-dataset-curbramp` and `-crosswalk` are byte-identical: 1,585 of 1,585
   images have the same sha256. I joined crop `label_id`s back to the live label API for
   Pittsburgh, Teaneck and Oradell, and **all 275 of 275 checked `curbramp/` crops are
   `Crosswalk` labels**. The montage confirms it by eye: they show painted crosswalks. The
   crosswalk repo was uploaded at 00:49 UTC on 2025-08-16 and the curbramp repo 3 minutes later,
   so the wrong folder was probably pushed.
2. **Only 4 classes exist, and only 3 of them are barriers.** The classes are Crosswalk,
   NoCurbRamp, Obstacle and SurfaceProblem. There are no CurbRamp crops (point 1) and no
   NoSidewalk dataset at all. The proposal's five-class classifier cannot be built from these
   datasets.
3. **Only "correct" crops show the class.** Each class has `correct/` and `incorrect/` folders,
   and the folder is the crowd's verdict on whether the label was that type. "Incorrect" crops
   can show anything. Counting correct crops only, there are **13,231 positives**:
   NoCurbRamp 5,307, SurfaceProblem 4,124, Obstacle 2,946, Crosswalk 854. That is a **6.2×
   imbalance**, or 1.8× across the three barrier classes. The proposal's "~5,100 per type" and
   "~25k crops" do not hold.
4. **Pittsburgh is already in these crops, in every split.** There are 598 Pittsburgh images
   (2.6%), of which 393 are correct. By split: train 424 (276 correct), val 84 (61 correct),
   test 90 (56 correct). The splits are a random 70/15/15 by file, not by city. A
   held-out-Pittsburgh evaluation must remove these. The published validator models were also
   trained on the 424 Pittsburgh train images, so they are **not clean Pittsburgh baselines**.
5. **The CurbRamp validator model was trained on data that isn't published.** Each model's
   `trainer_state.json` implies how many images it was trained on:
   - **Crosswalk, NoCurbRamp, Obstacle, SurfaceProblem:** the implied train-set size matches
     the published train split exactly.
   - **CurbRamp:** it implies 3,265–3,296 train images, but the `curbramp` repo has 1,108 (the
     crosswalk ones).

   So the Crosswalk model was trained on the crosswalk crops, and the real CurbRamp crops
   (~4.7k at a 70% split) were never uploaded.
6. **The mirror is intact and clean.** All 24,241 files (9.51 GB) match Hugging Face on size
   and sha256. All 24,241 decode as WebP RGB with no corrupt files. The longest side is always
   640 px; 91% are 640×640.
7. **Leakage within the crops is negligible.** There are 3 exact cross-split duplicates within
   a class, all neighbouring `label_id`s (the same object labelled twice). A dHash check with
   Hamming ≤ 4 finds 31 near-duplicate pairs among 22,656 images, 16 of them cross-split and 1
   cross-class.
8. **`sidewalk-tagger-ai` is the bigger crop source, and the only published source of CurbRamp
   crops.** I read the zip central directories and CSVs over HTTP range requests, fetching
   about 28 MB in total and downloading no images.
   - **Validated:** 24,008 PNG crops (71 GB): CurbRamp 10,857, SurfaceProblem 9,092, Obstacle
     2,432, Crosswalk 1,627.
   - **Unvalidated:** 87,495 crops (~251 GB).
   - **Missing:** neither has NoCurbRamp or NoSidewalk.
   - **Labels:** multi-hot accessibility *tags*, not validation verdicts.
   - **Crop geometry:** a flat equirectangular 640 px crop, different from the validator
     crops.
   - **Pittsburgh:** included. The validated set has 1,486 Pittsburgh crops, and the
     unvalidated set has 3,569.

## Integrity (local mirror vs Hugging Face)

| dir | HF files | local | sha256 verified | mismatches | GB | HF commit |
|---|---:|---:|---:|---:|---:|---|
| curbramp | 1,585 | 1,585 | 1,585 | 0 | 0.542 | `d396ee43` |
| nocurbramp | 9,150 | 9,150 | 9,150 | 0 | 3.645 | `e737c59a` |
| obstacle | 5,110 | 5,110 | 5,110 | 0 | 2.004 | `9fdc3c30` |
| surfaceproblem | 6,811 | 6,811 | 6,811 | 0 | 2.776 | `23f32c07` |
| crosswalk | 1,585 | 1,585 | 1,585 | 0 | 0.542 | `1be513a0` |

The `.gitattributes` files are excluded from these counts. The validator models were checked
too: all 10 files in each of the 5 repos match on size, and `model.safetensors` matches on
sha256. This run read the models from their NVMe staging copy at
`/mnt/scratch/datasets/models/sidewalk-validator-ai-*/`. Their permanent home is
`/data/datasets/models/sidewalk-validator-ai-*/`, rsynced from the staging copy after the run.

## Decode

- **Files:** 24,241, all WebP RGB, 0 corrupt. Median size is 393 KB.
- **Dimensions:** 22,065 are 640×640. The other 2,176 are 640 on the long side and 300–639 on
  the short side, because the crop box was clamped at the edge of the perspective render.
- **Short-side histogram:**

| short side (px) | 300–399 | 400–499 | 500–599 | 600–638 | 639–640 |
|---|---:|---:|---:|---:|---:|
| images | 274 | 555 | 893 | 447 | 22,072 |

## Class × split × verdict

| dir | train ✓ | train ✗ | val ✓ | val ✗ | test ✓ | test ✗ | total |
|---|---:|---:|---:|---:|---:|---:|---:|
| crosswalk | 597 | 511 | 128 | 109 | 129 | 111 | 1,585 |
| curbramp (= crosswalk bytes) | 597 | 511 | 128 | 109 | 129 | 111 | 1,585 |
| nocurbramp | 3,714 | 2,690 | 796 | 576 | 797 | 577 | 9,150 |
| obstacle | 2,062 | 1,514 | 441 | 324 | 443 | 326 | 5,110 |
| surfaceproblem | 2,886 | 1,880 | 618 | 403 | 620 | 404 | 6,811 |
| **real total (excl. curbramp/)** | 9,259 | 6,595 | 1,983 | 1,412 | 1,989 | 1,418 | **22,656** |

✓ = `correct`, ✗ = `incorrect`. The share of correct crops is 54–61% per class, close to the
generator's 55:45 target.

## Where the crops come from: 19 Project Sidewalk cities

These are **correct-only** crops, the usable positives, per city:

| city | crosswalk | nocurbramp | obstacle | surfaceproblem | all |
|---|---:|---:|---:|---:|---:|
| sea (Seattle) | 120 | 2,961 | 1,356 | 954 | 5,391 |
| chicago | 100 | 402 | 1,037 | 685 | 2,224 |
| taipei | 225 | 168 | 102 | 295 | 790 |
| teaneck | 89 | 390 | 15 | 259 | 753 |
| oradell | 36 | 84 | 52 | 544 | 716 |
| mendota | 47 | 160 | 10 | 294 | 511 |
| newberg | 0 | 218 | 51 | 226 | 495 |
| columbus | 8 | 241 | 51 | 110 | 410 |
| **pittsburgh** | **16** | **200** | **15** | **162** | **393** |
| keelung | 50 | 25 | 77 | 168 | 320 |
| amsterdam | 29 | 65 | 114 | 71 | 279 |
| new_taipei | 122 | 39 | 3 | 104 | 268 |
| cdmx | 2 | 110 | 31 | 73 | 216 |
| spgg | 1 | 109 | 10 | 89 | 209 |
| st_louis | 7 | 95 | 7 | 27 | 136 |
| knox | 1 | 27 | 5 | 49 | 82 |
| cliffside_park | 0 | 4 | 10 | 10 | 24 |
| blackhawk_hills | 1 | 5 | 0 | 2 | 8 |
| la_piedad | 0 | 4 | 0 | 2 | 6 |
| **total** | **854** | **5,307** | **2,946** | **4,124** | **13,231** |

**Seattle and Chicago make up 57.5% of all positives**, and Seattle alone is 56% of NoCurbRamp
and 46% of Obstacle. Each city appears in train, val and test for every class except
Cliffside Park, which has at least 10 images but is missing a split in three classes. Full
tables: `tables/city_by_class_{correct,all}.csv` and `tables/city_split_share.csv`.

### Pittsburgh (598 real-class images)

| class | train ✓/✗ | val ✓/✗ | test ✓/✗ | total |
|---|---|---|---|---:|
| crosswalk | 10 / 16 | 3 / 4 | 3 / 3 | 39 |
| nocurbramp | 141 / 82 | 30 / 10 | 29 / 17 | 309 |
| obstacle | 14 / 16 | 1 / 3 | 0 / 9 | 43 |
| surfaceproblem | 111 / 34 | 27 / 6 | 24 / 5 | 207 |

I joined these crops to today's Pittsburgh label API, `/data/datasets/pittsburgh/projectsidewalk/rawLabels.csv`.
Current severities of the correct crops:
- **NoCurbRamp:** 30 at severity 1, 53 at 2, 109 at 3.
- **SurfaceProblem:** 49 at 1, 39 at 2, 57 at 3.

The rest have no severity.

## What "correct" and "incorrect" mean, and which crops are positives

Source: `data_generation.py` and `split_data.py` at
[ProjectSidewalk/sidewalk-validator-ai](https://github.com/ProjectSidewalk/sidewalk-validator-ai)
(`master`).

- **Labels:** the generator pulls `rawLabels` for exactly these 19 cities, including Pittsburgh,
  and keeps a single label type per run.
- **`correct`:** `agree_count − disagree_count > 2`, downsampled to a 55:45 ratio against
  incorrect. **`incorrect`:** `disagree_count − agree_count > 1`.
- **Crop:** it fetches the GSV panorama, resizes it to 8192×4096, and renders a 2048×2048
  perspective view with a 90° field of view at the label's heading, pitch 0. It cuts a square
  centred on the label point, with half-side `6100 / depth` from Depth-Anything-V2 (metric,
  vitl). That gives about 11.9 m of physical footprint. The crop is resized to 640 on the long
  side and saved as `<city>_<label_id>.webp`.
- **Split:** `split_data.py` shuffles each `correct/` and `incorrect/` folder (no seed) and
  splits 70/15/15. **It does not split by city or by panorama.**
- **Positives for a multi-class barrier classifier:** only `correct/` crops, and only from
  `nocurbramp/`, `obstacle/` and `surfaceproblem/`, plus `crosswalk/` if a non-barrier class is
  wanted. **Never use `curbramp/`.** `incorrect/` crops are not examples of any known class.
  They are labels the crowd rejected as that type, and can show another barrier, nothing, or an
  occlusion. At most they are a noisy "not this class" pool.
- **Label drift:** agreement counts have moved since generation. Today some `correct/` crops sit
  at net agreement 0 (minimum over 2,002 checked), and some `incorrect/` ones at +1.

## Duplicates and near-duplicates (curbramp/ excluded)

- **Exact duplicates:** 6 sha256 groups. None cross classes and none have conflicting verdicts.
  Three cross splits within a class, and each pair is neighbouring label ids, which means a
  duplicate label on the same object:
  - `surfaceproblem` train/incorrect `taipei_43216` ↔ test/incorrect `taipei_43215`
  - `nocurbramp` train/incorrect `chicago_5763` ↔ val/incorrect `chicago_5762`
  - `obstacle` train/correct `chicago_78847` ↔ test/correct `chicago_78848`
- **Near-duplicates:** 64-bit dHash, Hamming ≤ 4, over 22,656 images, gives **31 pairs**:
  - 16 within a class across splits (3 of them exact), involving 32 images
  - 14 within a class and split (3 exact)
  - 1 across classes
  - 0 with conflicting verdicts

  The pairs are in `tables/near_duplicate_pairs.csv`, and the 10 closest non-same-split pairs
  are in `figures/near_duplicate_pairs.png`.
- **The non-exact pairs are real duplicates, not hash collisions.** Looked at by eye, each shows
  the same view under two different `label_id`s, sometimes far apart (e.g. `newberg_828` ↔
  `newberg_2107`), which means the same corner was labelled twice. The one cross-class pair is
  `nocurbramp` `chicago_7550` ↔ `surfaceproblem` `chicago_7549`: the same spot labelled as both
  types, and both marked incorrect.
- **Caveat:** dHash catches re-crops of the same view. It misses the same object seen from a
  different panorama, which is the more likely leak here.

## Validator models

All five are `Dinov2ForImageClassification` fine-tuned from `facebook/dinov2-large`: 24
layers, hidden size 1024, patch 14, **304.37M parameters**, F32. Each is a binary
`{0: correct, 1: incorrect}` classifier. It answers "is this label of type X correct?" and does
not detect or classify barrier types.

Training came from `fine_tune_classifier.py`: learning rate 1e-5, batch 8 × accumulation 4,
25 epochs, best checkpoint chosen by macro-F1 on **val**. Production uses them:
[`sidewalk-ai-api`](https://github.com/ProjectSidewalk/sidewalk-ai-api) loads
`projectsidewalk/sidewalk-validator-ai-{label_type}`.

| model | best **val** macro-F1 | epoch (of 25) | optimizer steps/epoch | implied train images | published train split | match | best checkpoint |
|---|---:|---:|---:|---|---:|:---:|---|
| crosswalk | 0.949 | 11 | 35 | 1,089–1,120 | 1,108 | ✅ | `Crosswalk/dinov2/checkpoint-385` |
| curbramp | 0.860 | 20 | 103 | **3,265–3,296** | 1,108 | ❌ | `CurbRamp/dinov2/checkpoint-2060` |
| nocurbramp | 0.846 | 10 | 201 | 6,401–6,432 | 6,404 | ✅ | `NoCurbRamp/dinov2/checkpoint-2010` |
| surfaceproblem | 0.808 | 25 | 149 | 4,737–4,768 | 4,766 | ✅ | `SurfaceProblem/dinov2/checkpoint-3725` |
| obstacle | 0.797 | 22 | 112 | 3,553–3,584 | 3,576 | ✅ | `Obstacle/dinov2/checkpoint-2464` |

- **How the implied range is computed:** Hugging Face Trainer takes
  `ceil(ceil(N/8)/4)` optimizer steps per epoch. That pins N to a 32-image window.
- **The CurbRamp weights are different weights, not a copy.** All 5 weight files have distinct
  sha256s. Since the CurbRamp model's implied N doesn't match its published dataset (❌), it was
  trained on a CurbRamp crop set that is not on Hugging Face.
- **The metrics are validation-split numbers for label validation.** They are neither test
  numbers nor barrier-classification numbers. Each repo also ships an `optimizer.pt` of about
  2.4 GB, which this analysis never loads.
- **Preprocessing differs between training and serving, per the committed script.**
  - **Training:** `RandomResizedCrop(256)`, then `ToTensor`, with ImageNet `Normalize`
    **commented out**.
  - **Shipped preprocessor, used at inference:** `BitImageProcessor`, which resizes to 256,
    centre-crops 224, and **does** normalize with ImageNet mean/std.

  This is from `master`; whether the uploaded checkpoints were trained with exactly this script
  is not verifiable.
- **Smaller variants exist.** Quantized DINOv2-tiny ONNX versions of all 5 validators (hidden
  size 384) are on Hugging Face as `projectsidewalk/quantized_{crosswalk,curb_ramp,no_curb_ramp,obstacle,surface_problem}_dinov2_tiny_onnx`.
  They were not downloaded.

## sidewalk-tagger-ai datasets (not downloaded)

These come from the ASSETS '24 paper *Towards Fine-Grained Sidewalk Accessibility Assessment
with Deep Learning*. Code is Apache-2.0 at
[ProjectSidewalk/sidewalk-tagger-ai](https://github.com/ProjectSidewalk/sidewalk-tagger-ai), and
the Hugging Face license tag is MIT. Neither Hugging Face card has content beyond the license
header. The counts below come from the archives' own central directories and tag CSVs.

| set | archive | images | uncompressed | train / test rows | tag columns | Pittsburgh |
|---|---|---:|---:|---|---:|---:|
| validated | Crosswalk.zip | 1,627 | 4.63 GB | 1,306 / 321 | 10 | 57 |
| validated | CurbRamp.zip | 10,857 | 30.79 GB | 8,674 / 2,183 | 10 | 761 |
| validated | Obstacle.zip | 2,432 | 7.30 GB | 1,954 / 478 | 20 | 59 |
| validated | SurfaceProblem.zip | 9,092 | 28.28 GB | 7,282 / 1,810 | 13 | 609 |
| unvalidated | Crosswalk.zip | 7,623 | 20.24 GB | 7,623 / – | 13 | 173 |
| unvalidated | CurbRamp.z01+z02+zip | 43,352 | 121.36 GB | CSV in split archive, not read | – | 1,710 |
| unvalidated | Obstacle.zip | 10,150 | 29.79 GB | 10,150 / – | 20 | 619 |
| unvalidated | SurfaceProblem.z01+zip | 26,370 | 80.10 GB | CSV in split archive, not read | – | 1,067 |

- **Filenames:** `gsv-<city>-<label_id>-<LabelType>.png`, from 12 cities: Seattle, Chicago,
  Columbus, Oradell, Newberg, SPGG, CDMX, Amsterdam, Pittsburgh, Teaneck, St. Louis and Walla
  Walla. Counts per city are in `summary.json` under `tagger.*.by_city`.
- **Most frequent tags in the validated train CSVs:**
  - **CurbRamp:** missing-tactile-warning 3,353; points-into-traffic 1,087; surface-problem 809;
    narrow 775
  - **SurfaceProblem:** grass 3,219; cracks 2,913; bumpy 1,642
  - **Obstacle:** narrow 1,181; pole 461; vegetation 355
- **As a Tier 0 source:**
  - **Pros:** it is larger than the validator crops for CurbRamp (10,857 vs 0) and
    SurfaceProblem (9,092 vs 4,124 correct). It is the only published CurbRamp crop set.
  - **Cons:** no NoCurbRamp, no NoSidewalk, and no val split. Crops are flat equirectangular
    640 px PNG, not depth-scaled perspective WebP, so mixing sources adds a geometry and codec
    domain shift. "Validated" refers to the tag-cleaning pass in the paper. I have not verified
    that every crop's label *type* was crowd-confirmed.

## Tier 0 implications

1. **The class set is three barriers.** NoCurbRamp, Obstacle and SurfaceProblem from the
   validator crops, optionally plus Crosswalk. **CurbRamp crops must come from
   `sidewalk-tagger-ai-validated`** or be regenerated with `sidewalk-validator-ai/data_generation.py`
   or `sidewalk-panorama-tools`. **NoSidewalk has no crop data anywhere published.**
2. **Use correct-only positives and weight for imbalance.** Counts run 854–5,307, and 1.8×
   across the three barriers. Treat `incorrect/` as out of scope, or as a separately modelled
   reject class.
3. **Re-split by city for held-out evaluation.** The shipped splits are random per file, so
   every city is in every split. For "train elsewhere → test Pittsburgh", drop all 598
   Pittsburgh validator crops, and the tagger's 1,486 validated / 3,569 unvalidated Pittsburgh crops, from train and val.
4. **The published validator models are contaminated for a Pittsburgh baseline.** They trained
   on 424 Pittsburgh crops, and on random splits of the same labels. They also solve a
   different task: binary validation given the type.
5. **Expect a city-mix skew.** Seattle plus Chicago are 57.5% of positives, and Seattle alone is
   56% of NoCurbRamp.

## Caveats and gaps

- **Only 3 of 19 cities were checked by label id** (Pittsburgh, Teaneck, Oradell: 2,738 crops).
  In all three, every directory except `curbramp/` matched its type (2,462 of 2,463, one label
  since deleted), and `curbramp/` was 275 of 275 Crosswalk. The other 16 cities are inferred
  from the byte identity.
- **Agreement counts and severities are today's API values, not those at generation time**
  (August 2025).
- **Near-duplicates are detected within these datasets only.** I did not check overlap with
  the tagger crops or with the `rampnet-dataset` panoramas.
- **The training-to-checkpoint link is inferred.** The model rows rely on `trainer_state.json`
  and the committed `fine_tune_classifier.py` constants (batch 8, accumulation 4). I did not
  open `training_args.bin`, a pickle, to confirm them.
- **Two tagger CSVs were not read.** The unvalidated CurbRamp and SurfaceProblem CSVs sit
  inside multi-part archives, so their train/test sizes and tags are unread. Image counts come
  from the central directory.
- **The generation code isn't seeded** (`sample(frac=1)`, `random.shuffle`), so the published
  splits cannot be regenerated. Use them as shipped.

## Reproduce

```bash
/opt/miniconda/envs/sidewalk/bin/python src/eda/crops_eda.py \
    --models-root /mnt/scratch/datasets/models      # default /data/datasets/models once synced
```

It takes about 5 minutes on the CPU box with 16 workers, and most of that is network (the label APIs and the tagger range reads).
Writes: `/data/eda/crops/summary.json`,
`tables/{dir_split_verdict,city_by_class_correct,city_by_class_all,city_split_share,pittsburgh_dir_split_verdict,image_sizes,exact_duplicates,near_duplicate_pairs,label_type_check,validator_models}.csv`,
`tables/image_features.csv.gz`, and `figures/{crops_montage,near_duplicate_pairs}.png`.

## Sources

- **Crop datasets on Hugging Face:**
  - [sidewalk-validator-ai-dataset-curbramp](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-curbramp)
  - [-nocurbramp](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-nocurbramp)
  - [-obstacle](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-obstacle)
  - [-surfaceproblem](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-surfaceproblem)
  - [-crosswalk](https://huggingface.co/datasets/projectsidewalk/sidewalk-validator-ai-dataset-crosswalk)
  - Local copies: `/data/datasets/sidewalk-validator-ai-dataset-*/`
- **Validator models:**
  - [sidewalk-validator-ai-curbramp](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-curbramp)
  - [-nocurbramp](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-nocurbramp)
  - [-obstacle](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-obstacle)
  - [-surfaceproblem](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-surfaceproblem)
  - [-crosswalk](https://huggingface.co/projectsidewalk/sidewalk-validator-ai-crosswalk)
  - Local copies: `/data/datasets/models/sidewalk-validator-ai-*/`. This run read them from
    `/mnt/scratch/datasets/models/`.
- **Quantized ONNX models:** e.g. [quantized_curb_ramp_dinov2_tiny_onnx](https://huggingface.co/projectsidewalk/quantized_curb_ramp_dinov2_tiny_onnx)
- **Tagger datasets:** [sidewalk-tagger-ai-validated](https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated),
  [sidewalk-tagger-ai-unvalidated](https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-unvalidated).
  Paper: [doi:10.1145/3663548.3688531](https://doi.org/10.1145/3663548.3688531)
- **Generation and training code:** [sidewalk-validator-ai](https://github.com/ProjectSidewalk/sidewalk-validator-ai)
  (`data_generation.py`, `split_data.py`, `fine_tune_classifier.py`, `get_stats.py`)
- **Serving code:** [sidewalk-ai-api `validator.py`](https://github.com/ProjectSidewalk/sidewalk-ai-api/blob/main/sidewalk_ai_api/validator.py)
- **Crop geometry analysis:** [sidewalk-panorama-tools `reports/2026-08-09-cropper-consumer-requirements.md`](https://github.com/ProjectSidewalk/sidewalk-panorama-tools/blob/main/reports/2026-08-09-cropper-consumer-requirements.md),
  local copy at `/data/code/sidewalk-panorama-tools/`
- **Label APIs used for the label-type check:**
  - [Pittsburgh](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/rawLabels?filetype=csv)
    (local snapshot `/data/datasets/pittsburgh/projectsidewalk/rawLabels.csv`)
  - [Teaneck](https://sidewalk-teaneck.cs.washington.edu/v3/api/rawLabels?filetype=csv)
  - [Oradell](https://sidewalk-oradell.cs.washington.edu/v3/api/rawLabels?filetype=csv)
