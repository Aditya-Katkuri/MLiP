#!/usr/bin/env python3
"""Write run_manifest.json for a run (runbook Part 5.7).

The git_sha field is the link between the code (git) and the artifacts (blob).
If git_dirty is true the run is NOT reproducible -- commit before you launch.

usage:
  python scripts/make_manifest.py --run-id 20260915-multiclass-lr3e4-alice \
      --author alice --out /data/runs/<run_id>/checkpoints/run_manifest.json \
      --hparams '{"lr": 3e-4, "batch_size": 8, "epochs": 3, "amp": "bf16"}'
"""
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone

DEFAULT_CLASSES = ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "NoSidewalk"]


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", required=True)
    p.add_argument("--author", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--base-checkpoint", default="projectsidewalk/rampnet-model")
    p.add_argument("--dataset", default="rampnet-dataset")
    p.add_argument("--shard-manifest", default="configs/shards_150.txt")
    p.add_argument("--classes", default=",".join(DEFAULT_CLASSES))
    p.add_argument("--hparams", default="{}", help="JSON object")
    p.add_argument("--allow-dirty", action="store_true")
    a = p.parse_args()

    dirty = bool(git("status", "--porcelain"))
    if dirty and not a.allow_dirty:
        sys.exit("refusing: working tree is dirty, this run would not be reproducible. "
                 "Commit first, or pass --allow-dirty.")

    manifest = {
        "run_id": a.run_id,
        "git_sha": git("rev-parse", "--short", "HEAD"),
        "git_dirty": dirty,
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "author": a.author,
        "base_checkpoint": a.base_checkpoint,
        "dataset": a.dataset,
        "shard_manifest": a.shard_manifest,
        "classes": a.classes.split(","),
        "hparams": json.loads(a.hparams),
    }

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
