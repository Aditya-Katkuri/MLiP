# Sidewalk data

Location: `/data/datasets/sidewalk-data/` on `sidewalk-cpu`.

Source: the complete [Validated folder](https://huggingface.co/datasets/projectsidewalk/sidewalk-tagger-ai-validated/tree/main/Validated)
of `projectsidewalk/sidewalk-tagger-ai-validated` on Hugging Face.

Files are downloaded unchanged: `Crosswalk.zip`, `CurbRamp.zip`, `Obstacle.zip`,
and `SurfaceProblem.zip` (about 71 GB total). All four are now extracted
into their original class folders alongside the retained ZIP archives.
Each filename comes from the corresponding `Validated/<filename>` upstream.
The previous directory-derived train/test dataset and CSVs were deleted.

Download script: `scripts/download_sidewalk_validated.py`.
Server tmux session: `sidewalk-validated-download`.
Log: `/data/logs/sidewalk-tagger-validated-download.log`.
Pinned revision, exact source paths, byte sizes, hashes, and completed files:
`/data/logs/sidewalk-tagger-validated-manifest.json`.

Downloads in progress live under `.download/`; completed archives appear directly
in the dataset root after size and SHA-256 verification. Re-running the script
resumes unfinished downloads and reuses verified completed archives.

Extraction: `scripts/extract_sidewalk_validated.py`.
All 24,016 files (71.0 GB uncompressed) passed ZIP CRC and file-size checks.
Extraction log: `/data/logs/sidewalk-tagger-validated-unzip.log`.
