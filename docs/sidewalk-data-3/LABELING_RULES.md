# Labeling rules: sidewalk-data-3-tag-groups-v1

Model 1 predicts binary `hazard_label`: 1 = hazard, 0 = non_hazard.
Model 2 predicts four independent tag-group memberships, in the order below.
These are multi-label targets, not mutually exclusive classes. Use four sigmoid
outputs / binary cross-entropy, not a four-way softmax. Train/evaluate the second
stage on hazard images for hazard-subtype classification; a joint model may use
all rows with the explicitly stored all-zero vectors on non-hazards. Report which
population is evaluated. Group-level metrics must not assume one true group.

- **Surface and level problems** (`surface_and_level_problems`): `broken-surface`, `bumpy`, `uneven-surface`, `not-level-with-street`, `pooled-water`, `steep`, `surface-problem`, `height-difference`, `stairs`, `uneven-slanted`, `uncovered-manhole`, `very-broken`.
- **Restricted passage / maneuvering space** (`restricted_passage`): `narrow`, `not-enough-landing-space`, `narrow-sidewalk`.
- **Missing tactile warning** (`missing_tactile_warning`): `missing-tactile-warning`.
- **Ramp points into traffic** (`ramp_points_into_traffic`): `points-into-traffic`.

A group is positive if ANY of its constituent tags is positive. An image is
hazardous if ANY group is positive. Context-needed and non-hazard tags remain
non-hazard, as requested; they do not cancel a co-occurring positive hazard tag.
The complete 46-tag mapping is `tag_mapping.csv` and `label_schema.json`.

All `no_obstacles_manual_collection` images are non-hazard because the user
identified them as ordinary sidewalks. This includes the 277 newly discovered
Pittsburgh training screenshots and the existing 50 validation + 50 test images.
Their original tags remain blank, not invented zero annotations.

Images with no positive hazard tags are non-hazard under this operational rule,
including images with no tag annotations at all. Absence of annotations is not
independent verification of safety. Source class names and old manual class
reviews do not override this tag rule. In particular, tagless images previously
reviewed as Obstacle/SurfaceProblem now receive non-hazard; they are explicitly
listed in `audit/label_review_candidates.csv`, with the old decisions preserved.
No new tags are inferred from original class names, filenames, or image content.

Existing splits and all image records are retained, including exact duplicates;
see `audit/cross_split_duplicates.csv` before interpreting split-independent
performance. No image is duplicated on disk merely because it has several groups.
The non-hazard outcome has the four-vector [0, 0, 0, 0].
