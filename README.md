# MLiP

10-718 Machine Learning in Practice project — curb-ramp and sidewalk-obstruction
models built on [RampNet](https://github.com/ProjectSidewalk/RampNet).

Infrastructure procedure lives in [`sidewalk-azure-runbook.md`](sidewalk-azure-runbook.md):
one shared A100 VM on Azure, four users, billed to Microsoft for Startups credits.
This README covers only how to reproduce a run.

## Layout

```
configs/     shard manifests, hyperparameter yamls  <- IN GIT, they define experiments
src/         data loading, model, training, eval
scripts/     infra + run plumbing (see below)
notebooks/   exploration, outputs stripped
reports/     figures and tables for the writeup
```

| Script | Runs on | What it does |
|---|---|---|
| `scripts/az_preflight.sh` | laptop | Asserts the isolated Azure CLI profile is active, then prints GPU SKU availability and current quota |
| `scripts/setup_vm.sh` | VM | Formats/mounts the data disk, shared group, teammate accounts, base tooling |
| `scripts/download_data.sh` | VM | Hugging Face → VM → Blob, so the download is never paid for twice |
| `scripts/sync_run.sh` | VM | Pushes a run's checkpoints and results to Blob |
| `scripts/make_manifest.py` | VM | Stamps a run with its git SHA |

## Where things live

| Thing | Where |
|---|---|
| Code, configs, notebooks | this repo |
| Shard manifests, label filters, split definitions | `configs/` |
| Raw datasets | Blob container `datasets` |
| Model checkpoints | Blob `checkpoints/<run_id>/` |
| Predictions, metrics, figures, logs | Blob `results/<run_id>/` |
| Live training metrics | Weights & Biases |
| `/data` on the VM | scratch working copy, assume it vanishes |

Never commit checkpoints or datasets, and do not reach for git-lfs — we have blob
storage, and lfs on a student repo hits quota limits fast.

**This repo is public.** Nothing secret goes in it: no subscription IDs, no storage
keys, no SAS tokens. Azure config lives in `~/sidewalk-env.sh` on each person's
laptop, which is deliberately outside the repo.

## Reproducing a run from its run_id

1. Fetch the manifest:
   ```bash
   azcopy copy "https://${AZ_SA}.blob.core.windows.net/checkpoints/<run_id>/run_manifest.json" .
   ```
2. Check out the exact code:
   ```bash
   git checkout $(jq -r .git_sha run_manifest.json)
   ```
   If `git_dirty` is `true`, the run is not reproducible. That is a bug, not a footnote.
3. Restore the data referenced by `dataset` / `shard_manifest`:
   ```bash
   azcopy copy "https://${AZ_SA}.blob.core.windows.net/datasets/<name>/*" /data/<name>/ --recursive
   ```
4. Re-run with the `hparams` block from the manifest.

## Starting a run

Run IDs are readable and unique: `YYYYMMDD-<what>-<key hparam>-<who>`,
e.g. `20260915-multiclass-lr3e4-alice`.

```bash
RUN_ID=20260915-multiclass-lr3e4-alice
mkdir -p /data/runs/$RUN_ID/{checkpoints,results}
python scripts/make_manifest.py --run-id $RUN_ID --author alice \
  --out /data/runs/$RUN_ID/checkpoints/run_manifest.json \
  --hparams '{"lr": 3e-4, "batch_size": 8, "epochs": 3, "amp": "bf16"}'
```

`make_manifest.py` refuses to run with a dirty working tree. That is deliberate —
the `git_sha` field is the only link between the code and the artifacts.

When the job finishes (or every N epochs from inside the training loop):

```bash
./scripts/sync_run.sh $RUN_ID
```

## Shared-GPU etiquette

One A100, four people.

- Every long job runs in `tmux` (`tmux new -s <you>-train`), so it survives a disconnect.
- Check `nvidia-smi` before you start, and claim the card in `/data/GPU_CLAIM.txt`
  with your name and expected finish time.
- Only stage-two training needs the whole GPU. Data prep, eval, and the crop
  classifiers can share.
- **Always `az vm deallocate`, never `az vm stop`.** `stop` still bills the GPU;
  a forgotten A100 running for a month is about $2,600.

## Housekeeping

Strip notebook outputs before committing, so four people don't generate merge
conflicts on base64 image blobs. Run once per clone — it installs a git filter and
is not carried in the repo:

```bash
pip install nbstripout && nbstripout --install
```
