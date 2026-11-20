# Manual class review

On `sidewalk-cpu`, run from `/data/repo/MLiP`:

```sh
python src/data_review.py --port 8768
```

Forward port **8768** in VS Code and open `http://localhost:8768`.
Choose a split and class directory, then assign any of these review labels:

| Key | Assigned class |
|---|---|
| `1` | `Crosswalk` |
| `2` | `CurbRamp` |
| `3` | `Obstacle` |
| `4` | `SurfaceProblem` |
| `5` | `no_obstacles` |
| `6` | `remove_data` |

The selected directory chooses the images to review; it does not constrain the
class you can assign. The current assigned class is shown above the image and its
button is highlighted. Click another class to change it. Arrow keys and Previous/Next
navigate. Turn off **Advance after labeling** to stay on the image. Filter by assigned
class or **Not reviewed**, and use **Reload labels** to load another session's edits.

Each selection is saved immediately to `manual_class_labels.csv` in that image's directory:

```csv
filename,label,updated_at
gsv-pittsburgh-123-CurbRamp.png,Obstacle,2026-10-29T02:00:00+00:00
```

The `label` value is a dataset class name or `remove_data`.
`remove_data` flags an image for removal; assigning it does not delete or move the image. Existing labels are loaded and
changes update the same row. CSV replacement is atomic and uses a per-directory
`.manual_class_labels.lock` file. A stale label from another session is reported
before overwriting. Extra CSV columns and rows for missing images are preserved;
invalid CSVs are reported without rewriting. Two-column `filename,label` CSVs work too.

Previous binary reviews in `manual_labels.csv` remain intact; they are not converted
into class assignments. Images stay in their current directories, and the main split
CSVs are unchanged. The new CSVs hold manual class assignments separately.

The app binds to `127.0.0.1`, needs no API key, and uses only Python's standard library.
Use `--root PATH` for another dataset with the same split/class layout.

Checks: `python -m unittest discover -s tests -p 'test_data_review.py'`.
