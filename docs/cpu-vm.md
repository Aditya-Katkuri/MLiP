# The CPU box: `sidewalk-cpu`

A shared 64-core, 512 GB machine in Azure for everything in this project that
**doesn't need a GPU**. Read the "What belongs here" section before you use it —
putting the wrong work on it wastes money, and putting data in the wrong place
loses it.

| | |
|---|---|
| Name | `sidewalk-cpu` in resource group `sidewalk-rg` |
| Size | `Standard_E64ads_v7` — 64 vCPU, 512 GB RAM |
| Region | East US 2 (same region as our blob storage — this is the point) |
| IP | **20.69.232.144** (static — it does not change when the VM is stopped) |
| Cost | **$4.65/hour**, billed only while it is *running* |
| Login | `azureuser` |

## Why this machine exists

Microsoft refused GPU quota on our subscription — it's a sponsored/benefit
subscription and that's a blanket policy, not a capacity problem (support ticket
2609100040010387). So **training happens elsewhere** (CMU / PSC).

What Azure is still genuinely good for is the *data layer*. All 474 GB of our
datasets already sit in blob storage in East US 2, and a VM in that same region
reads them at multi-Gb/s for free. Doing that work from a laptop means dragging
hundreds of GB across the internet onto a disk that probably can't hold it.

## First-time setup (each person does this once)

Right now **only Aditya can SSH in.** Two things are needed per person, and both
require Aditya to run a command — you can't self-serve these.

**1. Send Aditya your SSH public key.** On your own machine:

```bash
ssh-keygen -t ed25519 -C "$(whoami)@10718" -f ~/.ssh/sidewalk
cat ~/.ssh/sidewalk.pub        # send this line to Aditya
```

Send the `.pub` file only. Never send the other file (`~/.ssh/sidewalk`) to
anyone — that's your private key.

**2. Send Aditya your public IP.** Run `curl -4 ifconfig.me`. SSH is firewalled to
specific addresses, so an unknown IP is refused before it ever reaches a password
or key check. If your connection just hangs, this is almost always why. Your home
IP can change, and CMU wifi will give you a different one than your apartment, so
expect to redo this occasionally.

<details>
<summary>For Aditya: adding someone (click to expand)</summary>

```bash
source ~/sidewalk-env.sh

# 1. add their IP to the SSH allow-list (space-separated, include everyone each time)
az network nsg rule update -g "$AZ_RG" --nsg-name sidewalk-cpuNSG -n default-allow-ssh \
  --subscription "$AZ_SUB" \
  --source-address-prefixes 73.79.201.96/32 <THEIR_IP>/32

# 2. with the VM running, install their key
scp -i ~/.ssh/sidewalk alice.pub azureuser@20.69.232.144:/tmp/
ssh -i ~/.ssh/sidewalk azureuser@20.69.232.144 \
  'sudo adduser --disabled-password --gecos "" alice && sudo usermod -aG sidewalk alice && \
   sudo mkdir -p /home/alice/.ssh && sudo cp /tmp/alice.pub /home/alice/.ssh/authorized_keys && \
   sudo chown -R alice:alice /home/alice/.ssh && sudo chmod 700 /home/alice/.ssh && \
   sudo chmod 600 /home/alice/.ssh/authorized_keys'
```
</details>

## Starting and stopping it

The VM is **stopped (deallocated) by default** and should be left that way. You all
have Contributor on the resource group, so anyone can start it.

**In the portal:** portal.azure.com → search `sidewalk-cpu` → **Start**. Takes about
a minute. To stop it, use the **Stop** button, which deallocates properly.

**From the CLI:**

```bash
source ~/sidewalk-env.sh
az vm start      -g "$AZ_RG" -n sidewalk-cpu --subscription "$AZ_SUB"
az vm deallocate -g "$AZ_RG" -n sidewalk-cpu --subscription "$AZ_SUB"   # stops billing
```

> **`deallocate`, never `stop`.** From the CLI, `az vm stop` shuts down the OS but
> **keeps billing you** for the hardware. Only `deallocate` releases it. The
> portal's Stop button does the right thing; the CLI's `stop` does not.

At $4.65/hr, a box left running for a month is about **$3,300**. There's an
auto-shutdown at 0400 UTC as a safety net, but treat it as a backstop, not a plan.
Stop it when you're done for the day.

> Deallocating stops the compute bill but **does not release the vCPU quota** —
> measured, it still reads 64 of our 65 regional vCPUs as consumed. So we can't run
> a second large VM in East US 2 alongside this one without deleting it first. The
> disks keep costing a few dollars a month either way; that part is unavoidable and
> is what preserves your `/data`.

## Every time you start it: rebuild the scratch disk

```bash
ssh -i ~/.ssh/sidewalk <you>@20.69.232.144
sudo /tmp/setup_cpu_vm.sh
```

This is **not optional and not a one-time thing.** The fast scratch array is built
from *ephemeral* local NVMe that Azure destroys whenever the VM is deallocated. The
script rebuilds it. It's safe to re-run and it will not touch `/data`.

If `/tmp/setup_cpu_vm.sh` is missing (`/tmp` is cleared on reboot), copy it up again
from this repo: `scp -i ~/.ssh/sidewalk scripts/setup_cpu_vm.sh azureuser@20.69.232.144:/tmp/`

## Where to put things

This is the part that bites people. Three storage locations, very different
guarantees:

| Location | Size | Speed | Survives a stop? | Use it for |
|---|---|---|---|---|
| `/mnt/scratch` | 3.4 TB | fastest (striped NVMe) | **NO — erased every stop** | Working copies, HF cache, decoded images, anything regenerable |
| `/data` | 2.0 TB | fast | Yes | Derived data that's expensive to recompute |
| Blob storage | unlimited | network | Yes, durably | **The only real source of truth** |

**If you would be upset to lose it, it goes to blob.** Not `/data`, not
`/mnt/scratch`. `/data` survives a stop but dies with the VM; `/mnt/scratch` dies
every single time anyone clicks Stop, including while you're asleep and
auto-shutdown fires.

Blob needs no passwords or keys — the VM has its own identity:

```bash
azcopy login --identity          # once per login session

# pull the dataset down to scratch to work on it
azcopy copy "https://sidewalkdata23770.blob.core.windows.net/datasets/rampnet-dataset/*" \
  /mnt/scratch/rampnet-dataset/ --recursive

# push results back up
azcopy copy "/data/my-output/*" \
  "https://sidewalkdata23770.blob.core.windows.net/results/<run_id>/" --recursive
```

What's already in blob, verified file-by-file against Hugging Face:

| Container / path | Contents |
|---|---|
| `datasets/rampnet-dataset/` | 385 shards, 462.47 GB — the full training set |
| `datasets/rampnet-benchmark/` | 38 files, 11.41 GB — evaluation benchmark |
| `datasets/models/rampnet/` | 6 files, 0.36 GB — pretrained checkpoint |
| `checkpoints/<run_id>/` | your training checkpoints |
| `results/<run_id>/` | metrics, predictions, figures, logs |

## What belongs on this box

**Yes — this is what it's for:**

- Exploring the 462 GB dataset: shard statistics, label distributions, class balance
- Building the crop datasets from the full corpus
- Image preprocessing — decode, resize, crop, augment (64 cores, use them)
- Constructing train/val/test splits and shard manifests
- Computing metrics, generating figures and tables for the report
- Anything pandas / pyarrow / PIL / OpenCV over the whole dataset

**No — don't:**

- **Training.** There's no GPU. Stage-two training goes to CMU/PSC.
- **Casual browsing.** To eyeball a few hundred images or check one shard's schema,
  pull two or three shards to your laptop. You don't need a $4.65/hr machine to
  open a parquet file. This box earns its cost when you need to touch *all* the
  data at once.
- **Long jobs without `tmux`.** See below.

**Evaluation and the crop classifiers** want a GPU, but only a small one — Colab
Pro handles both comfortably. They were never blocked on Azure.

## Sharing it: four people, one machine

**Run every long job in `tmux`** so it survives your laptop closing or your wifi
dropping:

```bash
tmux new -s alice-prep      # start
# ctrl-b then d to detach
tmux attach -t alice-prep   # come back later
tmux ls                     # see everyone's sessions
```

**Check before you start something big.** `htop` shows what's running and who owns
it. 64 cores is a lot, but two people each running a 64-process pool will make both
jobs slower than running them one after another.

**Don't stop the VM without checking.** `who` and `tmux ls` show if someone else is
connected or has a job running. Stopping it kills their work *and* erases
`/mnt/scratch`.

**Shared files:** everything under `/data` and `/mnt/scratch` belongs to the
`sidewalk` group with setgid set, so files you create there are writable by the rest
of us automatically. Work in those directories, not in your home folder, if you want
anyone else to be able to use the output.

## Python environment

```bash
conda env create -f environment.yml     # first time, ~10 min
conda activate sidewalk
```

`environment.yml` lives in the repo root and pins the same versions RampNet uses, so
code behaves the same here as on whatever GPU machine we end up training on.

## When something doesn't work

| Symptom | Cause | Fix |
|---|---|---|
| SSH hangs with no prompt | Your IP isn't on the allow-list, or the VM is stopped | Check the VM is running; send Aditya `curl -4 ifconfig.me` |
| `Permission denied (publickey)` | Your key isn't installed, or you used the wrong `-i` path | Confirm with Aditya that your key was added |
| `/mnt/scratch` is empty or missing | The VM was stopped — this is expected, not a bug | `sudo /tmp/setup_cpu_vm.sh` |
| `azcopy` gives 403 | You didn't run `azcopy login --identity` this session | Run it |
| `conda: command not found` | Login shell didn't source the profile | `export PATH=/opt/miniconda/bin:$PATH` |
