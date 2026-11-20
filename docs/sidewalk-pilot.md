# Pittsburgh sidewalk pilot

The completed server dataset is `/data/datasets/sidewalk-pilot`.
Inputs are the corrected `pitt_inf_AB.zip` and `pitt_inf_CD.zip`; immutable archive
hashes are in [configs/sidewalk-pilot/archives.json](../configs/sidewalk-pilot/archives.json).
The superseded `pitt_inference_data_1.zip` and `pitt_inference_data_2.zip` copies
were removed. C/D metadata comes from the CD archive, not obsolete AB placeholders.

| Stretch | Location | Images | Review numbering |
|---|---|---:|---|
| A | Butler Street from 40th | 94 | Natural order |
| B | Penn Avenue from 40th | 96 | Decreasing hazard score |
| C | Downtown Penn Avenue from 10th | 107 | Natural order |
| D | Downtown Liberty Avenue from 10th | 102 | Decreasing hazard score |

Natural order is station distance, left before right, then view number. All 198
B/D images are scored; A/C are never fed to the model. Ranking uses p_hazard with
image ID breaking exact ties, without threshold filtering. Images are the units;
multiple views of a location are retained. Source visibility flags are not labels.

## Reproduce with saved weights

Use the complete `inference/` folder from the dataset-3 Colab export. It contains
the frozen DINOv3 backbone, independently selected hazard and group heads,
preprocessing, thresholds and checksums. On the server, the working Python is
`/data/tools/dino-eval-env/bin/python`; dependencies are torch, transformers,
numpy, pandas and Pillow (the bundle records exact versions). No API key is needed.

For a new build, place the two ZIPs under `<new-root>/archives/`, then run:

```sh
python scripts/sidewalk_pilot/build_and_infer.py --root /data/datasets/sidewalk-pilot-rebuild --bundle /path/to/exported/inference
```

Existing manifests/review folders are refused. The runner verifies archive and
bundle hashes, preserves originals and provenance, and saves predictions before
building numbered native-resolution RGB PNGs. It strips embedded metadata and
checks that review pixels match the source. Model input is separately resized to
224 x 224 using the exported preprocessing. CPU float32 scores can differ slightly
from Colab float16 scores; the saved pilot ranking is fixed for evaluation.

- `review_images/` and `review_images.zip`: A/B/C/D numbered PNGs, neutral review
  instructions, annotation definitions, blank labels CSV and image attribution.
- `results/`: all B/D probabilities, per-stretch top 12, model configuration, and
  `review_mapping.csv` joining numbered images to sources and natural order.
- `manifest.csv`, `source_metadata/`, `audit/`: source information and checks.

Keep results/mappings private until independent reviews are complete. Only the
review package is intended for reviewers. The existing package preserves ranking;
separately randomized anonymous copies would be preferable when collecting labels
solely to evaluate precision, because those labels can be joined back afterward.

## Evaluate human labels

Each reviewer retains a separate CSV. Assign `hazard`, `non_hazard`, or `uncertain`.
Do not populate labels from model predictions. After labeling:

```sh
python scripts/sidewalk_pilot/evaluate_precision_at_12.py --root /data/datasets/sidewalk-pilot --labels reviewer.csv --output reviewer_precision_at_12.csv
```

Precision@12 uses exactly 12 images per stretch. Missing/uncertain labels yield
bounds, not a reduced denominator. The script also measures natural-order top-12
precision within B and D using the same images and reviews. This avoids treating
differences between streets A/C and B/D as solely a ranking effect. Reviewers
should label independently with the same definitions; retain original annotations
when resolving disagreements. Precision cannot be measured before human labeling.
