# LaTeX uncertainty-analysis reports

## Combined report

[Combined PDF](uncertainty_analysis.pdf) · [Editable LaTeX](uncertainty_analysis.tex) (7 pages).

This is the integrated document: one bootstrap-training section, three analysis sections, and a final **Results** section. It compares the original non-bootstrap DINO head, the 128-head mean, and every uncertainty-aware ranking. Validation ties at 34/46; the original single head is retained as the simpler reference. All test numbers and test figures appear after that explicit selection. The original-head predictions were checked against identical train/validation/test manifests in the bootstrap archive.

Rebuild it with `python build_combined_report.py` from this directory. This replaces `uncertainty_analysis.tex`; to preserve manual edits, compile that `.tex` file directly with `pdflatex` instead. `combined_selection.json`, `combined_build_manifest.json` and `build/combined_verification.json` record the selection rule, source hashes and checks. Original baseline evidence is in `../inputs/original_dino/`.

## Standalone reports

Three separate reports contain the evaluation setup, methods, validation/test tables, interpretation, limitations and vector figures. They use the saved precision@46 analysis; no models or rankings were rerun.

| Report | PDF | Editable source |
|---|---|---|
| Point 2: uncertainty-adjusted hazard ranking (2 pages) | [PDF](point_2_adjusted_scores.pdf) | [LaTeX](point_2_adjusted_scores.tex) |
| Point 3: uncertainty in hazard ranks (4 pages) | [PDF](point_3_rank_uncertainty.pdf) | [LaTeX](point_3_rank_uncertainty.tex) |
| Point 4: pairwise outranking and consensus (5 pages) | [PDF](point_4_pairwise_consensus.pdf) | [LaTeX](point_4_pairwise_consensus.tex) |

The reports use ordinary article headings and tables. Operational details such as run identifiers and execution hardware remain in the underlying evidence files rather than the report text. All numerical tables are populated from the verified analysis JSON files in `../results/`. Figures in `figures/` are copies of the original vector PDFs.

To regenerate the LaTeX sources and PDFs from the saved analysis:

```bash
python build_reports.py
```

The generator uses Python's standard library, `pdflatex` and `pdftotext`. It compiles twice to resolve references. It intentionally checks the results supporting the written conclusions before generating reports. **Regeneration replaces the `.tex` files.** To preserve manual prose edits, compile the edited source directly instead:

```bash
cd /data/repo/MLiP/uncertainty_analysis/latex
pdflatex -interaction=nonstopmode -halt-on-error point_2_adjusted_scores.tex
pdflatex -interaction=nonstopmode -halt-on-error point_2_adjusted_scores.tex
```

`build/` is populated with compilation logs and extracted PDF text when reports are rebuilt; only the small verification receipts are versioned. `build_manifest.json` records the input/output hashes and compiler version; `build/verification.json` records the final compilation and numerical checks. The final PDFs compiled without warnings, undefined references or overflowing text boxes and were visually inspected for table and figure layout.
