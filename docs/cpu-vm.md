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
| Login | your `@andrew.cmu.edu` Microsoft account — no SSH key, no VPN |

## Why this machine exists

Microsoft refused GPU quota on our subscription — it's a sponsored/benefit
subscription and that's a blanket policy, not a capacity problem (support ticket
2609100040010387). So **training happens elsewhere** (CMU / PSC).

What Azure is still genuinely good for is the *data layer*. All 474 GB of our
datasets already sit in blob storage in East US 2, and a VM in that same region
reads them at multi-Gb/s for free. Doing that work from a laptop means dragging
hundreds of GB across the internet onto a disk that probably can't hold it.

## Connecting: sign in with your Microsoft account

**There are no SSH keys to exchange.** You sign in with the Microsoft account you
were invited with, and Azure issues a short-lived certificate behind the scenes.
Your Linux account is created automatically the first time you log in.

**Connect from anywhere** — home, campus, a café, a hotspot. There's no VPN, no IP
allow-list, and no key to copy around. Your Microsoft account *is* your login.

### One-time, on your own machine

1. **Accept the Azure invitation** emailed to your `andrew.cmu.edu` address. Nothing
   below works until you do.
2. **Install the Azure CLI** — [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli)
   (macOS: `brew install azure-cli`).
3. Run:

   ```bash
   az extension add --name ssh
   az login                       # sign in as your andrew.cmu.edu account
   ```

### What you get

You have the **Virtual Machine Administrator Login** role, so you get `sudo`
without a password. That's needed to rebuild the scratch disk (below). It also
means you can break the box for everyone, so run `sudo` deliberately.

---

## The session loop: start → connect → work → stop

**The VM is switched off by default.** It costs nothing while stopped, and $4.65/hr
while running. Every session looks like this:

### 1. Start it (Azure portal)

1. Go to **portal.azure.com** and sign in with your **andrew.cmu.edu** account.
2. Type `sidewalk-cpu` in the top search bar and click the virtual machine result.
3. Check **Status** on the Overview page:
   - **Stopped (deallocated)** → click **▶ Start** at the top. Wait ~60 seconds
     until Status reads **Running**.
   - **Running** → someone else is already on it. Don't stop it when you're done
     without checking (see *Sharing it* below).

> Anyone on the team can start it — you all have Contributor on the resource
> group. You do not need Aditya for this.

### 2. Connect

Two ways. Both use your Microsoft account; neither needs an SSH key.

<details open>
<summary><b>A. VS Code (recommended for real work)</b></summary>

**One-time setup**

1. Install the **Remote - SSH** extension (publisher: Microsoft) in VS Code.
2. In a terminal, generate the connection profile:

   ```bash
   az login
   az ssh config --file ~/.ssh/config -n sidewalk-cpu -g sidewalk-rg
   ```

   That adds a `sidewalk-cpu` host entry to your SSH config, together with a
   short-lived certificate proving who you are to Azure.

**Each session** (after starting the VM in the portal)

3. `Cmd + Shift + P` → type **Remote-SSH: Connect to Host...**
4. Pick **`sidewalk-cpu`**. A new VS Code window opens, running on the VM.
5. **File → Open Folder** → `/data` or `/mnt/scratch`.
6. **Terminal → New Terminal** gives you a shell *on the VM*, not your laptop.
   Run the scratch rebuild (step 3 below) there.

Your editor, terminal, notebooks, and extensions now all run on the 64-core
machine. The files you browse are the VM's files, not your laptop's.

> **Make sure you install *Remote - SSH*** (publisher: Microsoft). Several
> extensions have similar names; this is the one that works here.

> **If VS Code connected fine yesterday and refuses today**, your certificate
> expired — they're short-lived by design. Re-run the `az ssh config` command
> above and reconnect.

</details>

<details>
<summary><b>B. Terminal</b></summary>

```bash
az login                                    # first time, or after it expires
az ssh vm -n sidewalk-cpu -g sidewalk-rg
```

No `-i keyfile`, no username, no IP to remember.

> If you use Azure for other projects too, add
> `--subscription 62173b9a-6762-4b80-9610-9bd71c826d7a` so you don't land in the
> wrong subscription.

</details>

### 3. Rebuild the scratch disk

**Every single session, right after connecting:**

```bash
sudo /tmp/setup_cpu_vm.sh
```

Skip this and `/mnt/scratch` won't exist. See the next section for why.

### 4. Work

See *What belongs on this box* below.

### 5. Stop it (Azure portal)

Back on the `sidewalk-cpu` Overview page, click **⏹ Stop**. Confirm Status reads
**Stopped (deallocated)**.

**Check first that nobody else is connected** — `who` and `tmux ls` on the VM will
tell you. Stopping it kills their jobs and erases `/mnt/scratch`.

The portal's **Stop** button deallocates properly, which is what stops billing.
There's an auto-shutdown at 0400 UTC as a safety net, but don't rely on it — that's
potentially a full day of billing you didn't need.

## Starting and stopping from the CLI

The portal is the easy path (see the session loop above). From a terminal:

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

## Why the scratch rebuild is needed every time

`sudo /tmp/setup_cpu_vm.sh` is **not optional and not a one-time thing.** The fast
scratch array is built from *ephemeral* local NVMe that Azure destroys whenever the
VM is deallocated. The script rebuilds it. It's safe to re-run and it will not touch
`/data`.

If `/mnt/scratch` looks empty or missing, this is why — it is expected behaviour,
not a broken machine.

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

## Working with blob storage

Everything durable lives in one storage account, **`sidewalkdata23770`**, split into
three containers:

| Container | What's in it | You mostly |
|---|---|---|
| `datasets` | our source data — treat as read-only | read |
| `checkpoints` | model checkpoints, one folder per run | write |
| `results` | metrics, predictions, figures, logs | write |

### Signing in (once per session, on the VM)

```bash
azcopy login --identity
```

No password, no key, no SAS token. The VM has its own identity and blob recognises
it. If `azcopy` returns **403**, you forgot this step.

### Seeing what's there

```bash
azcopy list "https://sidewalkdata23770.blob.core.windows.net/datasets"
```

Contents, all verified file-by-file against the Hugging Face originals:

| Path | Shards | Size | Per file |
|---|---|---|---|
| `datasets/rampnet-dataset/train/` | 128 | 324.0 GB | ~2.5 GB |
| `datasets/rampnet-dataset/val/` | 128 | 92.5 GB | ~723 MB |
| `datasets/rampnet-dataset/test/` | 128 | 45.9 GB | ~359 MB |
| `datasets/rampnet-benchmark/` | 38 | 11.4 GB | — |
| `datasets/models/rampnet/` | 6 | 0.36 GB | pretrained checkpoint |

All shards are Parquet, named `data-000NN-of-00128.parquet`.

### Pulling data down — start with one shard

**This is the important bit.** The shards are uniform samples of the same data, so
**one shard is enough to develop against.** Copy a single 2.5 GB file, get your code
working, and only then scale up:

```bash
azcopy copy \
  "https://sidewalkdata23770.blob.core.windows.net/datasets/rampnet-dataset/train/data-00000-of-00128.parquet" \
  /mnt/scratch/
```

That takes seconds. Pulling `rampnet-dataset/*` instead pulls **all 462 GB** — close
to an hour, and it fills most of scratch. Do that only when you genuinely need a
full pass over the corpus.

A whole split, when you do need one:

```bash
azcopy copy \
  "https://sidewalkdata23770.blob.core.windows.net/datasets/rampnet-dataset/test/*" \
  /mnt/scratch/test/ --recursive        # 45.9 GB, the smallest split
```

### Pushing results back up

```bash
RUN_ID=20260915-explore-alice
azcopy copy "/data/runs/$RUN_ID/results/*" \
  "https://sidewalkdata23770.blob.core.windows.net/results/$RUN_ID/" --recursive
```

Or use `scripts/sync_run.sh <run_id>` from the repo, which does checkpoints and
results together.

### From Python

`azure-storage-blob` is in our conda env. On the VM the managed identity is picked
up automatically; on your laptop it uses your `az login`. Same code either way:

```python
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

svc = BlobServiceClient(
    "https://sidewalkdata23770.blob.core.windows.net",
    credential=DefaultAzureCredential(),
)
cc = svc.get_container_client("datasets")

# list what's available
for b in cc.list_blobs(name_starts_with="rampnet-dataset/val/"):
    print(b.name, b.size)

# fetch one shard, then read it normally
with open("/mnt/scratch/val-00000.parquet", "wb") as f:
    f.write(cc.download_blob("rampnet-dataset/val/data-00000-of-00128.parquet").readall())

import pandas as pd
df = pd.read_parquet("/mnt/scratch/val-00000.parquet")
```

### From your own laptop

You each hold **Storage Blob Data Contributor** on the storage account personally,
not just through the VM. So the same `azcopy` and Python code works locally after
`az login` — handy for pulling one small shard to poke at without starting the VM
at all. Just don't pull hundreds of GB over your home connection.

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
| Connection hangs with no prompt | The VM is stopped | Start it in the portal, wait for **Running** |
| `Connection closed by ... port 22` | You lack the **Virtual Machine Administrator Login** role. Contributor is *not* enough — Azure separates managing a VM from logging into it | Ask Aditya to assign it |
| VS Code worked yesterday, refuses today | Your certificate expired | Re-run `az ssh config --file ~/.ssh/config -n sidewalk-cpu -g sidewalk-rg` |
| `Couldn't retrieve token from local cache` | Your CLI session expired | `az login` again |
| `az: 'ssh' is not in the 'az' command group` | Missing CLI extension | `az extension add --name ssh` |
| Azure says you have no access at all | You never accepted the emailed invitation | Accept it, then `az login` |
| `/mnt/scratch` is empty or missing | The VM was stopped — this is expected, not a bug | `sudo /tmp/setup_cpu_vm.sh` |
| `azcopy` gives 403 | You didn't run `azcopy login --identity` this session | Run it |
| `conda: command not found` | Login shell didn't source the profile | `export PATH=/opt/miniconda/bin:$PATH` |
