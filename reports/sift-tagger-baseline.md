---
cursor:
  subagentId: "bc-4b1e074d-a3c5-5129-bdc7-bf021ce5ccd5"
---

# SIFT nearest-prototype baseline on the expert-validated tagger set

This replaces the older validator-set SIFT run (393 Pittsburgh correct crops, different label set). That run scored CurbRamp, NoCurbRamp, Obstacle, and SurfaceProblem on `sidewalk-validator-ai-dataset-*` correct crops. This run scores the official split of `projectsidewalk/sidewalk-tagger-ai-validated`.

OpenCV 4.12 `SIFT_create`. No SVM, logistic regression, or neural net. The sidewalk conda env already provided SIFT.

## Data

Found on `sidewalk-cpu` at `/data/datasets/sidewalk-data`. The Hugging Face zips `Validated/{Crosswalk,CurbRamp,Obstacle,SurfaceProblem}.zip` (revision `6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50`) were already downloaded and CRC-checked there. This run did not download them again and did not use the blob prefixes `sidewalk-validator-ai-dataset-*`.

Images are `<split>/<class>/<filename>`, 1440×960. The split is the CSV `split` column (`train.csv`, `val.csv`, `test.csv`). Val was counted and then left out of fitting and scoring. Train and test filenames do not overlap.

CSV rows, files on disk, and the published table match.

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Crosswalk | 1,446 | 124 | 57 |
| CurbRamp | 8,324 | 1,772 | 761 |
| Obstacle | 2,073 | 300 | 59 |
| SurfaceProblem | 6,983 | 1,500 | 609 |
| no_obstacles | 0 | 0 | 204 |
| **Total** | **18,826** | **3,696** | **1,690** |

Cities in the filenames match the README: train is cdmx, chicago, newberg, oradell, seattle, spgg, walla_walla; val is amsterdam and columbus; test is pittsburgh. The fit still follows the CSV columns.

`no_obstacles` is a surrogate, present only on test (204 images, 12.1% of 1,690). `test_no_obstacles_sources.csv` has 226 provenance rows for those 204 files, every row `label_basis=incorrect_annotation_proxy` and `source_verdict=incorrect`, taken from Pittsburgh `incorrect/*.webp` crops in `/data/datasets/sidewalk-do-not-use.tar.gz`. The class column is `no_obstacles`, which is outside the four published classes. It was left out of the codebook, the prototypes, and the four-way metrics. Scored test support is 1,486 (1,690 − 204).

## SIFT

Grayscale SIFT plus a dense grid every 32 px. Cap 40 sparse and 40 dense descriptors (80 total). Codebook: `cv2.BOWKMeansTrainer`, 200 visual words, fit on 200 train images per class (fixed seed `sift-tagger-v1-<class>`, RNG seed 0 for k-means). Those 200 are a subset of the 500 train images whose mean L1 histogram is the class prototype. Every class had more than 500 train images (Crosswalk 1,446, CurbRamp 8,324, Obstacle 2,073, SurfaceProblem 6,983), so the cap bound and none fell back to the full class. The 800 codebook images contributed 63,949 descriptors. Prediction is the nearest prototype by symmetric chi-square, `0.5 * sum (p-q)^2 / (p+q)`. Fewer than 5 descriptors would be no prediction. None of the 3,486 opened crops (2,000 train + 1,486 test) hit that floor. Abstentions 0. Runtime 37.3 seconds with 48 workers.

## Test metrics

Four scored classes only. Support is the official test column.

| Class | Precision | Recall | Support |
|---|---:|---:|---:|
| Crosswalk | 0.084 | 0.544 | 57 |
| CurbRamp | 0.677 | 0.267 | 761 |
| Obstacle | 0.056 | 0.475 | 59 |
| SurfaceProblem | 0.454 | 0.236 | 609 |

Macro recall 0.380. Accuracy 0.273 (406/1,486). Majority class is CurbRamp, the largest train class (8,324 images). Always predicting CurbRamp scores 0.512 (761/1,486).

Confusion, rows true and columns predicted, in the order Crosswalk, CurbRamp, Obstacle, SurfaceProblem. Abstain is zero for every class.

| True | Crosswalk | CurbRamp | Obstacle | SurfaceProblem |
|---|---:|---:|---:|---:|
| Crosswalk | 31 | 6 | 12 | 8 |
| CurbRamp | 218 | 203 | 183 | 157 |
| Obstacle | 15 | 8 | 28 | 8 |
| SurfaceProblem | 103 | 83 | 279 | 144 |

Crosswalk recall is 31/57. Precision is 31/367: 218 curb ramps and 103 surface problems were called Crosswalk. Obstacle recall is 28/59 and precision is 28/502, including 279 surface problems and 183 curb ramps. CurbRamp precision is 203/300 and recall is 203/761. Accuracy 0.273 is below the CurbRamp majority baseline of 0.512. The measured counts are in `reports/sift_tagger_baseline.json`. Recall chart: [sift-tagger-recall.png](/cursor/stores/bc-e83253f7-58cc-435d-aa0c-6257ce4fccf7/media/sift-tagger-recall.png).
