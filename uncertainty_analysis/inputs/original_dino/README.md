# Original non-bootstrap DINO reference

These files were copied from the original local `dino_sidewalk_data_3_runs.zip` export for the combined uncertainty report. The comparison uses the saved single deployment hazard head, not the three-seed configuration average.

`verification.json` records the exact source paths, hashes, selected trial, identical train/validation/test manifest hashes, recomputed precision@46 and rank-cutoff checks. `pitt-val_hazard_ranking.csv` and `pitt-test_hazard_ranking.csv` retain the original head's sorted probabilities and labels. The receipts preserve the verification performed when these inputs were gathered; report generation reads the verified metrics and does not need the original archive.
