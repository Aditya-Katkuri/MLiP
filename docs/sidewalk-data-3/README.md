# Sidewalk dataset 3

Server release: `/data/datasets/sidewalk-data-3` (24,950 images).
Source: `/data/datasets/sidewalk-data-2`, preserved unchanged. All Pittsburgh
partitions are retained; the other cities are a separate training pool.

| Split | Images | Hazard | Non-hazard | Hand-collected nulls |
|---|---:|---:|---:|---:|
| other-cities-train | 22,912 | 11,554 | 11,358 | 0 |
| pitt-train | 1,276 | 527 | 749 | 277 |
| pitt-val | 382 | 157 | 225 | 50 |
| pitt-test | 380 | 174 | 206 | 50 |

Hazard is the union of 17 positive tags in four groups: surface/level problems,
restricted passage, missing tactile warning, and ramps pointing into traffic.
Groups are multi-label. All other/context-dependent tags and all hand-collected
ordinary sidewalks are non-hazard under the agreed annotation policy.
See [the exact rules](LABELING_RULES.md), [tag mapping](tag_mapping.csv),
[label schema](label_schema.json), and [preprocessing](PREPROCESSING.md).

Every image receives EXIF orientation, RGB conversion, full-image bicubic resize
to 224 x 224, and PNG encoding. There is no label-centered recrop. The 277 added
training screenshots are discovered even though absent from the source CSV.
Manifests retain original tags/classes, review provenance, paths, hashes, image
properties and label rationale. The release includes histograms and audits.
85 images from old hazard classes without tags are assigned non-hazard and flagged;
within-split duplicate annotation conflicts are retained and documented. No cross-split
identical-image groups were found. These issues are not silently filtered in training.

## Rebuild and verify

Use Python with Pillow and matplotlib; build_config.json records release versions.
Choose a new output directory; the builder refuses to overwrite a release.

```sh
python scripts/sidewalk_data_3/build_dataset.py --source /data/datasets/sidewalk-data-2 --output /data/datasets/sidewalk-data-3-rebuild
python scripts/sidewalk_data_3/verify_dataset.py /data/datasets/sidewalk-data-3-rebuild
```

The rebuild preserves source splits. `audit_release.py` is the historical final
release audit; it expects the original server paths and archived tag-review files.
Generated timestamps and source filesystem metadata mean a rebuild need not have
the same archive/manifest hash. The training notebook deliberately verifies the
published release hashes, rather than silently accepting a changed dataset.

## Training and inference

Open [dino_sidewalk_data_3.ipynb](../../notebooks/dino_sidewalk_data_3.ipynb) in Colab.
[Setup and selection details](../../notebooks/sidewalk_data_3_support/README.md)
cover Drive persistence, the 84 trials, validation-only selection, deployment
weights, preprocessing, and offline inference. This experiment is separate from
the older five-class baseline and report-rerun notebook.

For a saved deployment predictions CSV, compute hazard precision at the top 10%:

```sh
python scripts/hazard_top_fraction.py --predictions /path/to/pitt-test_deployment_predictions.csv --output /path/to/top10_metrics.json
```

The selected deployment model has 22 true hazards among the top 38 of the 380
Pittsburgh test images: precision@top10% = 57.89%. This differs from precision@46
(26/46 = 56.52%). These are single-deployment metrics, not means over seeds.
[Pilot inference and blinded review](../sidewalk-pilot.md) uses the exported weights.
