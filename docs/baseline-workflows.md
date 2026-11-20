# Baseline and review workflows

These preserve the earlier class-based experiments. Their hazard definition
(Obstacle + SurfaceProblem) differs from [dataset 3's tag-based labels](sidewalk-data-3/README.md).

| Workflow | Code | Documentation |
|---|---|---|
| Five-class OpenAI evaluation, including Crosswalk | `src/baseline/openai_baseline.py` | [Five-class run](openai-naive-merged-nulls-five-class.md) |
| Earlier four-class OpenAI experiment | Historical saved runner beside its results | [Four-class run](openai-naive-merged-nulls.md) |
| Five-class zero-shot CLIP | `src/baseline/clip_zero_shot.py` | [CLIP run](clip-naive-merged-nulls-five-class.md) |
| SIFT reconstruction and per-image predictions | `scripts/sift_report_rerun.py` | [SIFT procedure](sift-report-rerun.txt) |
| DINO report reproduction | `notebooks/dino_ft_A_report_rerun.ipynb` | Instructions embedded in notebook |
| Scratch-trained classifiers | `notebooks/scratch_classifiers_A.ipynb` | Instructions embedded in notebook |
| Manual dataset class review | `src/data_review.py` | [Review guide](data-review.md) |

OpenAI credentials come only from OPENAI_API_KEY. Use `--dry-run` to validate
inputs without requests. Resume verifies inputs/configuration before skipping
already saved successes. Offline tests never call the paid API.

The current CLIP/data dashboard is `src/baseline/vlm_dashboard.py` (port 8766).
The historical VLM dashboard is separate: `src/baseline/vlm_dashboard_legacy.py`
(port 8767). Forward the selected port from VS Code. Legacy proxy labels are
preserved to explain the dataset replacement, not as verified ground truth.

Notebooks in Git have outputs stripped; completed exports stay outside version
control. The report notebook preserves its historical selection procedure; use
the dataset-3 notebook for the current precision@46/balanced-accuracy tie-break
and independently selected multi-label tag-group model.

Offline checks:

```sh
python -m unittest discover -s tests
python notebooks/sidewalk_data_3_support/smoke_checks.py
```
