# Sidewalk data

The current tag-based release is [sidewalk-data-3](../sidewalk-data-3/README.md).
It includes uniform preprocessing, hazard/non-hazard labels and four tag groups.
The curated source dataset is described below.

## Curated source: sidewalk-data-2

Source dataset: `/data/datasets/sidewalk-data-2/` on `sidewalk-cpu` (24,950 images).
The earlier dataset is `/data/datasets/sidewalk-data-old/`.
Layout: `<split>/<class>/<filename>`, with matching `<split>.csv` files.
`other-cities-train` contains non-Pittsburgh cities; all `pitt-*` splits contain Pittsburgh.
The original Pittsburgh partition used a stratified 60/20/20 split with seed 42, keeping identical image pixels together.
Manual review and the new validation/test collection change the final counts below.

| Class | other-cities-train | pitt-train | pitt-val | pitt-test |
|---|---:|---:|---:|---:|
| `Crosswalk` | 1,570 | 34 | 11 | 12 |
| `CurbRamp` | 10,096 | 457 | 152 | 152 |
| `Obstacle` | 2,376 | 39 | 14 | 16 |
| `SurfaceProblem` | 8,521 | 389 | 124 | 130 |
| `no_obstacles` | 349 | 80 | 31 | 20 |
| `no_obstacles_manual_collection` | 0 | 277 | 50 | 50 |
| **Total** | **22,912** | **1,276** | **382** | **380** |

The four original classes come from `projectsidewalk/sidewalk-tagger-ai-validated`, revision
`6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50`: images from `Validated/<class>.zip` -> `<class>/crops/<filename>`;
annotations from the class CSVs. Reviewed `no_obstacles` WebPs originated in `incorrect/` folders of
`/data/datasets/sidewalk-do-not-use.tar.gz`. Manual review of all 704 original no_obstacles images
kept 480 in that class, reassigned 85, and removed 139. `manual_relabeling.csv` records these decisions.

`no_obstacles_manual_collection` contains 100 PNGs supplied from `/home/prateik/screenshots/` on the user's local machine.
Sorted filenames were shuffled with Python `random.Random(42)`: 50 go to validation and 50 to test;
none of that original batch go to training. Another 277 collected images were later added to pitt-train.
These added images were absent from the source split CSV and are explicitly included in dataset 3. Original filenames and bytes are preserved, with unknown coordinates/tags blank.
`no_obstacles_manual_collection_sources.csv` records source paths, assignments, city, hashes, and dimensions.

The dataset `README.md` provides city counts and detailed provenance; `split_config.json` records current counts and split settings.
`<split>_no_obstacles_sources.csv` preserves archive provenance for the original reviewed negatives.
Manual-collection import and verification: `/data/logs/sidewalk-manual-collection-20261029T210139Z/`.
Dataset-2 build and review audit: `/data/logs/sidewalk-data-2-20261029T053608Z/`.
Original split audit: `/data/logs/sidewalk-data-pitt-split-20261029T015555Z/`.
Original upstream CSVs: `/data/logs/sidewalk-data-reorganization-20261005T224119Z/original_csvs/`.
Download provenance: `/data/logs/sidewalk-tagger-validated-manifest.json`.
