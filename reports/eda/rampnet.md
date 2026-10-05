# RampNet corpus — EDA

What is actually inside the RampNet training shards, the Stage-1 inputs that produced them, the
paper's gold set, the post-publication benchmark, and the checkpoints — measured, not quoted.

- Script: [`src/eda/rampnet_eda.py`](../../src/eda/rampnet_eda.py) (`python src/eda/rampnet_eda.py`, ~1 min on the CPU box once shards are on `/mnt/scratch`)
- Numbers: `/data/eda/rampnet/summary.json`; tables `/data/eda/rampnet/tables/*.csv`; figures `/data/eda/rampnet/figures/`
- Run 2026-09-14 against the blob mirror on `/mnt/scratch/datasets`, RampNet @ `2916ba5`

## Headline findings

1. **The card's counts reproduce exactly — after dropping 9 placeholder rows nobody mentions.**
   The parquet footers hold **214,385 rows / 849,904 points**. Shard `data-00087-of-00128` of
   *every* split contains the same three test-fixture rows: `pano_id` `file_1`/`file_2`/`file_3`,
   image bytes = the 19-byte string `dummy image content`, `pano_coord` (0, 0),
   `record_creation_time` 0, and for `file_1` the points `[[1,2],[3,4],[5,6]]` (out of the unit
   square). Without them: **214,376 panoramas / 849,895 labels = the card exactly.** The
   placeholder "images" do not decode (`PIL.UnidentifiedImageError`), so any loop over the full
   shards will crash or train on garbage unless it filters them. They are also the *only*
   cross-split pano_id overlap.
2. **Splits are 150,063 / 42,875 / 21,438 = 70.0 / 20.0 / 10.0 %**, with zero pano_id overlap
   within or across splits once the placeholders are removed.
3. **"NYC / Portland / Bend" is mostly NYC.** By `pano_coord`: NYC 117,323 panos (54.7 %) and
   **566,832 labels (66.7 %)**; Portland 78,848 (36.8 %) / 263,981 (31.1 %); **Bend 18,205
   (8.5 %) / 19,082 (2.2 %)** — and 74.7 % of Bend's panoramas are negatives.
4. **The proposal's "43 % of records never yielded usable data" is an artifact of an exact
   float join. The right number is ~32 %.** 43.23 % is real in the sense that RampNet's own
   `rampnet-stage1-inputs` card publishes it, and we reproduce it to the record (156,712 of
   276,071 gov records consumed) with the same exact `(lat, lon)` join its
   `gov_provenance.py` uses. But 31,469 of the 188,177 distinct coordinates in `dataset.jsonl`
   were serialised a few ulps off their `all_locations.csv` value (max |Δ| = 1.4×10⁻¹⁴°,
   < 6 nm — e.g. `40.65676284610272` vs `40.656762846102716`), and the exact join drops them.
   A 1 mm KD-tree join (count flat from 1 µm to 10 cm) gives **188,183 consumed → 31.8 % never
   became a label** in an intended panorama, **32.3 %** counting only panoramas that were
   actually published. It is also not "unusable data": it is records for which no panorama
   resolved or the install date failed the predates-capture check. The download step itself
   yielded **97.9 %** of intended panoramas (4,571 GSV refusals).
5. **The per-city Stage-1 numbers in the proposal inherit the same bug.** "NYC 130,527 /
   Portland 21,075 / Bend 5,110" are exact-join counts; the 1 mm join gives **150,496 / 30,380 /
   7,307**. Never-a-label rate by city (published panos, 1 mm): NYC 31.4 %, Portland 32.8 %,
   Bend 45.7 % (exact join: 40.5 / 53.4 / 62.0 %).
6. **The auto-labels are not a 1:1 projection of the government points.** Published panoramas
   carry 936,456 attached gov coordinates but only 849,895 placed points (−9.2 %): the crop model
   placed the same number of points as coordinates on 57.2 % of panoramas, fewer on 30.7 %, more
   on 12.1 %. The quality number that matters for "the ground truth is model-generated" is the
   paper's Stage-1 agreement with the gold set, **P 0.940 / R 0.925** (P ≤ 0.912 once redundant
   points are counted as FPs, per `docs/stage1_generation_cost.md`).
7. **The paper's gold set is in-domain and lives in `test`.** `manual_labels/`: 1,000 panoramas,
   3,919 ramps, 207 negatives; all 1,000 pano_ids are in `rampnet-dataset/test`, none in
   train/val; NYC 571 / Portland 348 / Bend 81. Boxes are tiny — median **11.8 × 9.4 px** at
   4096×2048 (p95 75 × 43). On those same panoramas the auto-labels have 3,972 points; the count
   matches the gold count on 654 / 1,000.
8. **`rampnet-benchmark` (the 11.4 GB in blob) is *not* that gold set** — the proposal and the
   handoff both call it "1,000 hand-labelled panoramas, 3,919 labels". It is a 9-city,
   **1,109-panorama post-publication benchmark** built 2026-07/08. Recomputed from the HF
   `records` config (matches RampNet's own scorer to 3 dp on all 9): **precision 0.873–0.975,
   recall 0.503–0.765** (unbiased subset: recall 0.459–0.740); pooled micro-precision 0.942
   (1,833 TP / 112 FP). Against the paper's in-domain 0.949 / 0.873, **out-of-domain recall is
   11–37 points lower** — the best available prior for what Pittsburgh will look like.
9. **One shard is a fair development sample — for train.** pano_ids are globally sorted across
   shards, so a shard is a contiguous id range; GSV ids are effectively random, so city mix is
   stable per train shard (NYC 50–58 %, σ 1.4 pts; negatives 16–24 %) but noisy per test shard
   (NYC 44–62 %, negatives 13–30 %). Evaluate on the whole test split, not a shard.
10. **Frame geometry is narrow.** All 1,920 sampled images are 4096×2048 RGB JPEG (median
    2.13 MB; p5 1.37, p95 3.08 MB). Labels sit in a thin band just below the horizon: y median
    0.578, p1–p99 0.525–0.728; 0.11 % above the horizon.
11. **The crop model that placed every label was evaluated on a leaky split, and it saw
    Pittsburgh.** In `rampnet-crop-model-dataset-round1` (27,704 crops), 14.7 % of val and 14.2 %
    of test crops are byte-identical to a train crop. Its source labels come from 12 Project
    Sidewalk cities including Pittsburgh, which RampNet's own registry marks as transitively
    contaminated for the whole pipeline — relevant to any "held-out Pittsburgh" claim.

## rampnet-dataset

Raw parquet footers (placeholders included — see finding 1):

| split | shards | rows | rows/shard | GB | labels | zero-label panos | mean labels/pano |
|---|---:|---:|---|---:|---:|---:|---:|
| train | 128 | 150,066 | 1,172–1,173 | 324.05 | 595,900 | 30,373 (20.2 %) | 3.97 |
| val | 128 | 42,878 | 334–335 | 92.50 | 171,946 | 8,408 (19.6 %) | 4.01 |
| test | 128 | 21,441 | 167–168 | 45.92 | 82,058 | 4,658 (21.7 %) | 3.83 |
| **total** | 384 | 214,385 | | 462.47 | 849,904 | 43,439 (20.3 %) | 3.96 |

Clean (placeholders removed): 214,376 panoramas, 849,895 labels, 43,433 zero-label panoramas
(43,000 designed negatives + 433 positive-manifest panoramas where the crop model placed nothing),
≈4.97 labels per positive panorama.

By city (bounding box on `pano_coord`; the shards have no city column):

| city | panos | share | labels | share | zero-label panos | labels / positive pano |
|---|---:|---:|---:|---:|---:|---:|
| NYC | 117,323 | 54.7 % | 566,832 | 66.7 % | 11.2 % | 5.44 |
| Portland | 78,848 | 36.8 % | 263,981 | 31.1 % | 21.1 % | 4.25 |
| Bend | 18,205 | 8.5 % | 19,082 | 2.2 % | 74.7 % | 4.14 |

Labels per panorama (train, share): 0 → 20.2 %, 1 → 2.9 %, 2 → 12.1 %, 3 → 8.8 %, 4 → 12.6 %,
5 → 10.5 %, 6 → 9.9 %, 7 → 9.0 %, 8 → 9.6 %, 9 → 3.1 %, ≥10 → 1.3 %, max 15. Even counts
dominate and mass collapses above 8 — consistent with paired ramps per corner. Full histograms:
`tables/ramps_per_pano_hist.csv`, figure `figures/dataset_ramps_and_y.png`.

Schema: `image` (bytes), `pano_id`, `record_creation_time` (all 2025-06-14…17 — build time, not
capture time), `curb_ramp_points_normalized` (list of [x, y]), `pano_coord` [lat, lon],
`curb_ramp_coords` (the attached gov points), `pano_azimuth` (−180…180).

## Stage 1: from government records to labels

| step | n |
|---|---:|
| government curb-ramp records (`location_data/`) | 276,071 |
| records consumed by ≥1 intended positive pano — exact float join (upstream method) | 156,712 |
| records consumed by ≥1 intended positive pano — 1 mm join | **188,183** |
| records consumed by ≥1 published pano — 1 mm join | 186,776 |
| intended positive panoramas (`dataset.jsonl`) | 175,336 |
| negative candidates sampled from streets (`negativepanos.jsonl`) | 88,125 |
| negatives used (`negativepanosSHORTENED.jsonl`, exactly 20 %) | 43,834 |
| intended panoramas (`finaldataset.jsonl` = positives + negatives, verified) | 219,170 |
| written by `download_dataset.py` (RampNet docs) | 214,599 |
| published (clean) | 214,376 |
| auto-placed curb-ramp points | 849,895 |

Intended-but-unpublished: 4,794 = the docs' 4,571 GSV refusals + 223 unexplained. Every published
row's `curb_ramp_coords` length matches its manifest entry (100 %).

| city | gov records | consumed (exact) | never a label (exact) | consumed (1 mm, published) | never a label (1 mm, published) |
|---|---:|---:|---:|---:|---:|
| NYC | 217,679 | 130,527 | 40.0 % | 149,251 | **31.4 %** |
| Portland | 45,035 | 21,075 | 53.2 % | 30,272 | **32.8 %** |
| Bend | 13,357 | 5,110 | 61.7 % | 7,253 | **45.7 %** |
| **total** | 276,071 | 156,712 | 43.2 % | 186,776 | **32.3 %** |

(The "exact" columns are for intended panos, matching the upstream card; 1 mm for intended panos
is 188,183 / 31.8 %.)

Tolerance sweep, intended panos: exact 156,712 · 1 µm 188,183 · 1 mm 188,183 · 1 cm 188,183 ·
10 cm 188,183 · 1 m 188,184. The mismatched coordinates split NYC 19,969 / Portland 9,305 /
Bend 2,195.

Dates: 22,012 records (8.0 %, all Portland) carry the paper-era `2000-01-01` unknown-date
sentinel, which always passes the predates-capture check; 17,638 of them became labels. NYC
"dates" are the 2018 survey date (217,774 rows dated 2018), not install dates.

## The paper's gold set (`manual_labels/`)

| | |
|---|---|
| panoramas | 1,000 (all in `rampnet-dataset/test`) |
| ramps | 3,919 (single class) |
| negative panoramas | 207 |
| city | NYC 571 · Portland 348 · Bend 81 |
| labels per pano | 0: 207 · 1: 16 · 2: 137 · 3: 88 · 4: 132 · 5: 77 · 6: 107 · 7: 92 · 8: 134 · 9: 7 · ≥10: 3 |
| box size @ 4096×2048 | median 11.8 × 9.4 px; p5–p95 w 3.6–75.3, h 3.0–42.5 |
| auto-labels on the same panos | 3,972 points; count equal on 654 / 1,000 |
| published scores on it | Stage 2 P 0.949 / R 0.873; Stage 1 P 0.940 / R 0.925 |

## rampnet-benchmark (post-publication, out-of-domain)

All-panos P/R recomputed from the HF `records` config; "unbiased" drops the 5 hand-picked `top`
panoramas per city. 95 % Wilson intervals are in `summary.json`. Figure: `figures/benchmark_pr.png`.

| city | source (HF value) | native size | capture range | panos | detections | TP | FP | missed (confident / unsure) | P | R | P unbiased | R unbiased |
|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| richmond | mapillary | 11000×5500 | 2024-07 – 2025-09 | 124 | 267 | 237 | 10 | 73 / 78 | 0.960 | 0.765 | 0.961 | 0.740 |
| bend † | launch (GSV) | 16384×8192 | 2012-05 – 2025-07 | 110 | 265 | 248 | 12 | 79 / 72 | 0.954 | 0.758 | 0.972 | 0.738 |
| morgantown | mapillary | 4096×2048 | 2024-01 – 2025-07 | 125 | 209 | 195 | 5 | 72 / 20 | 0.975 | 0.730 | 0.969 | 0.684 |
| annapolis | mapillary | 8000×4000 | 2020-05 – 2023-10 | 125 | 227 | 214 | 8 | 80 / 34 | 0.964 | 0.728 | 0.961 | 0.692 |
| clovis | mapillary | 5760×2880 | 2019-04 – 2021-05 | 125 | 164 | 139 | 13 | 56 / 21 | 0.914 | 0.713 | 0.889 | 0.650 |
| gainesville | launch (GSV) | 16384×8192 | 2011-03 – 2026-04 | 125 | 205 | 189 | 11 | 83 / 31 | 0.945 | 0.695 | 0.943 | 0.647 |
| paterson | launch (GSV) | 16384×8192 | 2007-10 – 2025-11 | 125 | 284 | 270 | 7 | 125 / 36 | 0.975 | 0.684 | 0.987 | 0.650 |
| sao_paulo | launch (GSV) | 16384×8192 | 2015-10 – 2026-01 | 125 | 251 | 190 | 24 | 91 / 43 | 0.888 | 0.676 | 0.869 | 0.626 |
| budapest_district5 ‡ | mapillary | 5760×2880 | 2019-10 – 2026-07 | 125 | 189 | 151 | 22 | 149 / 48 | 0.873 | 0.503 | 0.885 | 0.459 |
| laurens_mapillary § | — | — | — | 94 | | | | | 0.898 | 0.390 | 0.863 | 0.325 |
| laurens_gsv § | — | — | — | 86 | | | | | 0.925 | 0.505 | 0.904 | 0.459 |

† Bend is a training city: 4 of its 110 panoramas are in `rampnet-dataset` (3 train, 1 val) —
`6WC0hdAYRsSAcluKSs5iRg`, `DJ8Zp111zu6KnMZz-0PHgQ`, `VgWpqFkTwCIROvM0z-DkOw` (train),
`9kW9cxpuj7q8DMzf-ClrQQ` (val). No other split shares a pano_id with the training set.
‡ Budapest's reviewer rated their own pass low-confidence; do not place it beside the US rows
without the caveats in the benchmark README. § Laurens is in git only (not pushed to HF); scored
with RampNet's `scripts/score_validation.py`.

All 1,109 HF panoramas carry `label_type` CurbRamp, model `rampnet-model@08-21-2025`; galleries
hold 314 A/B crops (1024×1024).

## Crop-model datasets

| | round 1 (Project Sidewalk crops) | round 2 (manual) |
|---|---|---|
| crops (train / val / test) | 27,704 (19,392 / 4,155 / 4,157 = 70 / 15 / 15 %) | 1,212 (880 / 142 / 190) |
| keypoints | 35,757 (25,054 / 5,374 / 5,329) | 1,863 (1,356 / 215 / 292) |
| keypoints per crop | 1: 21,751 · 2: 4,629 · 3: 840 · 4: 293 · 5: 106 · ≥6: 85; no empty crops | 1: 615 · 2: 556 · 3: 28 · 4: 13; no empty crops |
| crop size | 683×2048 (all) | 682×2048 (2 at 704×2048) |
| size on disk | 13.37 GB parquet | 453 MB jpg |
| card says | 27,704 crops from 20,698 panoramas — matches | 312 panoramas / "1,212 labels" — 1,212 is the crop count; keypoints are 1,863 |

**Round 1's split leaks.** By `sha256` of the image bytes, 1,188 distinct images appear in more than
one split: **612 of 4,155 val crops (14.7 %) and 591 of 4,157 test crops (14.2 %) are
byte-identical to a train crop**, and 1,645 further rows duplicate an image within their own
split. Consistent with two validated labels on one panorama snapping to the same 30° heading and
rendering the same strip under different uids (the card's own recipe), shuffled by the unseeded
`splititup.sh`. Any crop-model val/test number from this partition is optimistic. Round 1 cities
include Pittsburgh (crop-model pre-training, per RampNet's contamination registry), so Pittsburgh
is already transitively inside every RampNet label.

## Checkpoints

| | backbone | params | input → heatmap | notes |
|---|---|---:|---|---|
| `rampnet-model` (blob `models/rampnet/`) | `convnextv2_base.fcmae_ft_in22k_in1k_384` | 90,050,561 (F32, 380 tensors) | 2048×4096 → 512×1024 | threshold 0.55, min_distance 10, TTA recommended |
| `rampnet-crop-model` round 1 / round 2 | same | 90,050,561 each | 1024×352 → 256×88 | `.pth` and `.safetensors` of each (360 MB); round 2 is what Stage 1 loads |

## Caveats & gaps

- Crop-model round 1 was read from its NVMe staging copy on `/mnt/scratch`, because the `/data`
  disk was saturated at the time. The persistent copy is `/data/datasets/rampnet-crop-model-dataset-round1/`,
  rsynced from that staging copy. Its duplicate count is
  by exact image bytes; near-duplicates (same corner, different strip) are not counted, so the
  leakage figure is a floor.
- City attribution is a bounding box on `pano_coord`; it separates the three cities cleanly (only
  the 9 placeholders fall outside) but is not an official city field.
- The image-size sample is the first 5 rows of the middle row group of each shard (1,920 rows),
  not a random sample. No imagery capture date exists in the shards, so label-vs-imagery vintage
  cannot be measured from them.
- The 1 mm join counts a gov record as consumed if any manifest coordinate lies within 1 mm.
  `all_locations.csv` has 8 duplicate-coordinate rows, so the count may be high by at most 8.
  The upstream card's "23,088 records (8.36 %) with no install date" does not match the 22,012
  `2000-01-01` sentinels counted here; not reconciled.
- Benchmark all-panos numbers include 5 hand-picked high-density panoramas per city; quote the
  unbiased columns for between-city comparison. Recall is per-pano-comprehensive as judged by one
  reviewer per split, with no second rater.
- HF `records` label GSV splits `source = "launch"`, not `"gsv"` as the dataset card documents.

## Sources

Hugging Face
- rampnet-dataset — https://huggingface.co/datasets/projectsidewalk/rampnet-dataset
- rampnet-stage1-inputs — https://huggingface.co/datasets/projectsidewalk/rampnet-stage1-inputs
- rampnet-benchmark — https://huggingface.co/datasets/projectsidewalk/rampnet-benchmark
- rampnet-crop-model-dataset-round1 — https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round1
- rampnet-crop-model-dataset-round2 — https://huggingface.co/datasets/projectsidewalk/rampnet-crop-model-dataset-round2
- rampnet-model — https://huggingface.co/projectsidewalk/rampnet-model
- rampnet-crop-model — https://huggingface.co/projectsidewalk/rampnet-crop-model

GitHub (ProjectSidewalk/RampNet @ `2916ba5`)
- Repo — https://github.com/ProjectSidewalk/RampNet/tree/2916ba5353fe11130a37f61a262bb55698583c23
- Gold labels — https://github.com/ProjectSidewalk/RampNet/tree/2916ba5353fe11130a37f61a262bb55698583c23/manual_labels
- Benchmark README (per-city caveats) — https://github.com/ProjectSidewalk/RampNet/blob/2916ba5353fe11130a37f61a262bb55698583c23/benchmark/README.md
- Stage-1 yield — https://github.com/ProjectSidewalk/RampNet/blob/2916ba5353fe11130a37f61a262bb55698583c23/docs/stage1_generation_cost.md
- Contamination registry — https://github.com/ProjectSidewalk/RampNet/blob/2916ba5353fe11130a37f61a262bb55698583c23/docs/data_provenance.md
- Exact-join provenance script — https://github.com/ProjectSidewalk/RampNet/blob/2916ba5353fe11130a37f61a262bb55698583c23/scripts/analysis/gov_provenance.py
- Scorer — https://github.com/ProjectSidewalk/RampNet/blob/2916ba5353fe11130a37f61a262bb55698583c23/scripts/score_validation.py

Paper — O'Meara et al., *RampNet*, ICCV'25 CV4A11y workshop, https://arxiv.org/abs/2508.09415

Local copies
- Blob `https://sidewalkdata23770.blob.core.windows.net/datasets/{rampnet-dataset,rampnet-benchmark,models/rampnet}/`, mirrored to `/mnt/scratch/datasets/` and `/data/datasets/`
- `/data/datasets/rampnet-stage1-inputs/`, `/data/datasets/rampnet-crop-model-dataset-round2/`, `/data/datasets/models/rampnet-crop-model/`
- `/data/datasets/rampnet-crop-model-dataset-round1/`
- `/data/code/RampNet/` (git clone)
