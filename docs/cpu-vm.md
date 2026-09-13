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

## Connecting: sign in with your Microsoft account

**There are no SSH keys to exchange.** This VM uses Microsoft Entra login, the same
model as Azure ML compute instances: you authenticate with the Microsoft account
you were invited with, and Azure issues a short-lived certificate behind the
scenes. Your Linux account is created automatically the first time you log in.

### One-time, on your own machine

```bash
# 1. accept the Azure invitation emailed to your andrew.cmu.edu address, if you
#    haven't already -- nothing below works until you do
# 2. install the Azure CLI, then:
az extension add --name ssh
az login                       # sign in as your andrew.cmu.edu account
```

### Every time you connect

```bash
az ssh vm -n sidewalk-cpu -g sidewalk-rg
```

That's it. No `-i keyfile`, no username, no IP address to remember.

> If you also use Azure for other projects, add
> `--subscription 62173b9a-6762-4b80-9610-9bd71c826d7a` so you don't land in the
> wrong one.

### VS Code

Export an SSH config once, then use the **Remote - SSH** extension normally:

```bash
az ssh config --file ~/.ssh/config -n sidewalk-cpu -g sidewalk-rg
```

`sidewalk-cpu` then appears in the Remote-SSH host list. The certificate is
short-lived, so re-run that command if VS Code starts refusing to connect.

### What you get

You have the **Virtual Machine Administrator Login** role, so you get `sudo`
without a password. That's needed to rebuild the scratch disk (below). It also
means you can break the box for everyone, so run `sudo` deliberately.

### The one thing Aditya still has to do: the firewall

SSH is restricted by source IP. CMU's campus ranges (`128.2.0.0/16`,
`128.237.0.0/16`) are allowed, so **on campus it just works**. From home or a
café, your connection will hang with no error — that is the firewall, not a
broken machine. Send Aditya the output of `curl -4 ifconfig.me` and he'll add it.

<details>
<summary>For Aditya: allowing another IP (click to expand)</summary>

Portal → `sidewalk-cpu` → **Networking** → **Network settings** → click
`default-allow-ssh` → **Source: IP Addresses** → append to the CIDR list:

```
128.2.0.0/16,128.237.0.0/16,73.79.201.96/32,<NEW_IP>/32
```

Or from the CLI (space-separated, list everyone every time — it replaces, not appends):

```bash
source ~/sidewalk-env.sh
az network nsg rule update -g "$AZ_RG" --nsg-name sidewalk-cpuNSG -n default-allow-ssh \
  --subscription "$AZ_SUB" \
  --source-address-prefixes 128.2.0.0/16 128.237.0.0/16 73.79.201.96/32 <NEW_IP>/32
```

Granting a *new* person access (not just a new IP) needs two role assignments at
resource-group scope: `Virtual Machine Administrator Login` (to SSH — note that
Contributor does **not** grant this; Azure separates managing a VM from logging
into it) and `Storage Blob Data Contributor` on the storage account (for blob).
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
az ssh vm -n sidewalk-cpu -g sidewalk-rg
sudo /tmp/setup_cpu_vm.sh
```

This is **not optional and not a one-time thing.** The fast scratch array is built
from *ephemeral* local NVMe that Azure destroys whenever the VM is deallocated. The
script rebuilds it. It's safe to re-run and it will not touch `/data`.

If `/tmp/setup_cpu_vm.sh` is missing (`/tmp` is cleared on reboot), copy it up again
from this repo. `az ssh config` makes `scp` work too:

```bash
az ssh config --file ~/.ssh/config -n sidewalk-cpu -g sidewalk-rg
scp scripts/setup_cpu_vm.sh sidewalk-cpu:/tmp/
```

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

**Shared files:** everything under `/data` and `/mnt/scratch` is group-writable with
setgid plus a default ACL for `aad_admins`, the group every Entra login lands in. So
files you create there are writable by the rest of us automatically, even for
teammates who have never logged in yet. Work in those directories, not in your home folder, if you want
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
| SSH hangs with no prompt | Your IP isn't on the allow-list, or the VM is stopped | Check the VM is running; on campus you're covered, otherwise send Aditya `curl -4 ifconfig.me` |
| `Connection closed by ... port 22` | You lack the **Virtual Machine Administrator Login** role. Contributor is not enough | Ask Aditya to assign it |
| `Couldn't retrieve token from local cache` | Your CLI session expired | `az login` again |
| `az: 'ssh' is not in the 'az' command group` | Missing CLI extension | `az extension add --name ssh` |
| Azure says you have no access at all | You never accepted the emailed invitation | Accept it, then `az login` |
| `/mnt/scratch` is empty or missing | The VM was stopped — this is expected, not a bug | `sudo /tmp/setup_cpu_vm.sh` |
| `azcopy` gives 403 | You didn't run `azcopy login --identity` this session | Run it |
| `conda: command not found` | Login shell didn't source the profile | `export PATH=/opt/miniconda/bin:$PATH` |
