# Sidewalk data

Location: `/data/datasets/sidewalk-data/` on `sidewalk-cpu`; its `README.md` gives city counts and class percentages.
Layout: `<split>/<class>/<filename>`. Test is Pittsburgh; val is Amsterdam + Columbus.
Original training classes use other cities; the 500 training `no_obstacles` images use any non-Pittsburgh city.
Training no_obstacles additions include validation cities (amsterdam, columbus), so train/validation are no longer fully city-disjoint.

| Class | Train | Val | Test |
|---|---:|---:|---:|
| Crosswalk | 1,446 | 124 | 57 |
| CurbRamp | 8,324 | 1,772 | 761 |
| Obstacle | 2,073 | 300 | 59 |
| SurfaceProblem | 6,983 | 1,500 | 609 |
| no_obstacles | 500 | 0 | 204 |
| **Total** | **19,326** | **3,696** | **1,690** |

The four original classes come from [Validated](https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/tree/6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50/Validated),
revision `6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50` of `projectsidewalk/sidewalk-tagger-ai-validated`.
Each PNG comes from `Validated/<class>.zip` → `<class>/crops/<filename>`; annotation fields come from its class CSVs.

`no_obstacles` uses `incorrect/` WebP images in `/data/datasets/sidewalk-do-not-use.tar.gz`, across all source splits.
It is a proxy based on rejected annotations, not a verified absence of all barriers. Source bytes are unchanged.
The test set has 204 Pittsburgh examples. Train adds exactly 500 non-Pittsburgh examples (shuffle seed 42),
excluding existing annotation IDs and identical images across all splits. Validation has no examples of this class.
The legacy `sea` city code is normalized to `seattle`; original codes remain in training provenance.

`train.csv`, `val.csv`, and `test.csv` contain paths, classes, splits, and annotation fields; unavailable values are blank.
Full archive paths and hashes are in `train_no_obstacles_sources.csv` and `test_no_obstacles_sources.csv`.
All pre-existing CSV rows are retained; validation and test CSVs are unchanged by the training addition.
Training audit and backups: `/data/logs/sidewalk-train-no-obstacles-20261006T030346Z/`.
Test audit and backups: `/data/logs/sidewalk-no-obstacles-20261006T000755Z/`.
Original upstream CSVs: `/data/logs/sidewalk-data-reorganization-20261005T224119Z/original_csvs/`.
Download provenance: `/data/logs/sidewalk-tagger-validated-manifest.json`.
