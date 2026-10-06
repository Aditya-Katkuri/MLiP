---
cursor:
  subagentId: "bc-4b1e074d-a3c5-5129-bdc7-bf021ce5ccd5"
---

# SIFT nearest-prototype baseline on the expert-validated tagger set

This replaces the older validator-set SIFT run (393 Pittsburgh correct crops, different label set). Scored classes are CurbRamp, Obstacle, and SurfaceProblem. Crosswalk is in the dataset and is left out of the codebook, the prototypes, and the test metrics.

OpenCV 4.12 `SIFT_create`. No SVM, logistic regression, or neural net. The sidewalk conda env already provided SIFT.

## Data

Found on `sidewalk-cpu` at `/data/datasets/sidewalk-data`. The Hugging Face zips `Validated/{Crosswalk,CurbRamp,Obstacle,SurfaceProblem}.zip` (revision `6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50`) were already downloaded and CRC-checked there. This run did not download them again and did not use the blob prefixes `sidewalk-validator-ai-dataset-*`.

Images are `<split>/<class>/<filename>`, 1440×960. The split is the CSV `split` column. Val was counted and then left out of fitting and scoring. Train and test filenames do not overlap. CSV rows, files on disk, and the published table match.

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Crosswalk | 1,446 | 124 | 57 |
| CurbRamp | 8,324 | 1,772 | 761 |
| Obstacle | 2,073 | 300 | 59 |
| SurfaceProblem | 6,983 | 1,500 | 609 |
| no_obstacles | 0 | 0 | 204 |
| **Total** | **18,826** | **3,696** | **1,690** |

Cities in the filenames match the README. The fit follows the CSV columns.

Crosswalk is excluded on purpose (1,446 train, 124 val, 57 test). `no_obstacles` is a surrogate, present only on test (204 images). `test_no_obstacles_sources.csv` marks those files `incorrect_annotation_proxy`. Neither class is one of the three scored classes. Scored test support is 1,429.

## Why this was refit

L1 normalization is per image, so class counts do not enter that division. The codebook does mix classes. The earlier four-class fit built 200 visual words from 200 train images of each class, including Crosswalk (63,949 descriptors). This run uses the same 500/200 train sample for CurbRamp, Obstacle, and SurfaceProblem (the per-class filename hashes match) and refits the 200 words on those three classes only (48,000 descriptors, 600 images × 80). Every histogram changes with the visual words, and there is no Crosswalk prototype to compete at prediction time. The old confusion matrix cannot be trimmed into this result.

## SIFT

Grayscale SIFT plus a dense grid every 32 px. Cap 40 sparse and 40 dense descriptors (80 total). Codebook: `cv2.BOWKMeansTrainer`, 200 visual words, fixed seed `sift-tagger-v1-<class>`, RNG seed 0 for k-means. The 200 codebook images are a subset of the 500 whose mean L1 histogram is the class prototype. Every scored class had more than 500 train images, so the cap bound. Prediction is the nearest of the three prototypes by symmetric chi-square, `0.5 * sum (p-q)^2 / (p+q)`. Fewer than 5 descriptors would be no prediction. None of the 2,929 opened crops hit that floor. Abstentions 0. Runtime 31.3 seconds with 48 workers.

## Test metrics

| Class | Precision | Recall | Support |
|---|---:|---:|---:|
| CurbRamp | 0.707 | 0.498 | 761 |
| Obstacle | 0.060 | 0.576 | 59 |
| SurfaceProblem | 0.475 | 0.253 | 609 |

Macro recall 0.442. Accuracy 0.397 (567/1,429). Majority class is CurbRamp, the largest scored train class (8,324 images). Always predicting CurbRamp scores 0.533 (761/1,429).

Confusion, rows true and columns predicted, in the order CurbRamp, Obstacle, SurfaceProblem. Abstain is zero for every class.

| True | CurbRamp | Obstacle | SurfaceProblem |
|---|---:|---:|---:|
| CurbRamp | 379 | 220 | 162 |
| Obstacle | 17 | 34 | 8 |
| SurfaceProblem | 140 | 315 | 154 |

CurbRamp precision is 379/536 and recall is 379/761. Obstacle recall is 34/59 and precision is 34/569, including 315 surface problems and 220 curb ramps. SurfaceProblem recall is 154/609. Accuracy 0.397 is below the CurbRamp majority baseline of 0.533. The measured counts are in `reports/sift_tagger_baseline.json`. Recall chart: [sift-tagger-recall.png](/cursor/stores/bc-e83253f7-58cc-435d-aa0c-6257ce4fccf7/media/sift-tagger-recall.png).
