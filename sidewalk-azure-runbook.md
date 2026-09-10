# Azure Runbook: 10-718 Sidewalk Project

One shared A100 VM, four users, billed to the Microsoft for Startups credits.

Everything below is copy-pasteable. Work through Part 0 and Part 1 today; the rest waits on quota approval.

---

## Part 0. Isolate this project from your other Azure account

You already use `az` for a different project under a different email. The danger is not that
the other project drains these credits (it cannot; billing follows the subscription a resource
is created in). The danger is that you accidentally create a **$3.67/hr GPU VM in the wrong
subscription**, possibly one with a real credit card on it.

Fix it once, permanently, with a separate CLI profile directory.

### 0.1 Create a project env file

Save this as `~/sidewalk-env.sh`:

```bash
# ~/sidewalk-env.sh
export AZURE_CONFIG_DIR="$HOME/.azure-sidewalk"   # separate credential + default-sub store
export AZ_SUB="PASTE_SUBSCRIPTION_ID_HERE"
export AZ_RG="sidewalk-rg"
export AZ_LOC="eastus2"
export AZ_VM="sidewalk-gpu"
```

### 0.2 Log in inside that profile

```bash
source ~/sidewalk-env.sh
az login
```

This opens a browser. Sign in as **adityak0030@gmail.com** (the Founders Hub account), not your
other one. Because `AZURE_CONFIG_DIR` points somewhere new, this login is stored completely
separately from your existing `~/.azure` profile. Your other project is untouched.

### 0.3 Find the subscription ID and lock it in

```bash
az account list --query "[].{name:name, id:id, state:state, user:user.name}" -o table
```

Copy the `id` for the Founders Hub subscription into `AZ_SUB` in `~/sidewalk-env.sh`, then:

```bash
source ~/sidewalk-env.sh
az account set --subscription "$AZ_SUB"
az account show --query "{sub:name, id:id, user:user.name}" -o table
```

### 0.4 The rule

**Every new terminal, first command is `source ~/sidewalk-env.sh`.**

If you forget, `az` falls back to `~/.azure` and your other project's default subscription.
Belt and braces: every command in this doc that creates or bills something passes
`--subscription "$AZ_SUB"` explicitly. Keep that habit.

Sanity check you can run any time:

```bash
az account show --query name -o tsv
```

If that does not print the Founders Hub subscription, stop and re-source.

---

## Part 1. Request quota (do this today)

You start at **zero GPU vCPUs in every region**. Nothing can be created until this is approved.

### 1.0 Register the resource providers first

> **Verified 2026-09-10.** On a fresh subscription every provider starts `NotRegistered`, and
> until they are registered `az vm list-usage` returns **zero rows** — not zero quota, *no rows
> at all*. Because 1.2 below says "expect zeros" and pipes the output through `grep`, an
> unregistered provider prints exactly the same thing as a real zero quota. Do this first or
> you will file quota requests against a subscription that cannot report quota.

```bash
source ~/sidewalk-env.sh
for ns in Microsoft.Compute Microsoft.Network Microsoft.Storage \
          Microsoft.Quota Microsoft.ManagedIdentity; do
  az provider register --namespace "$ns" --subscription "$AZ_SUB"
done

# registration is async, 1-3 min. Wait for all five to say Registered:
az provider list --subscription "$AZ_SUB" \
  --query "[?namespace=='Microsoft.Compute' || namespace=='Microsoft.Network' \
           || namespace=='Microsoft.Storage' || namespace=='Microsoft.Quota' \
           || namespace=='Microsoft.ManagedIdentity'].{ns:namespace, state:registrationState}" \
  -o table
```

### 1.1 Check which regions actually have the SKUs

No point requesting quota where there is no capacity.

```bash
source ~/sidewalk-env.sh
for r in eastus2 westus3 centralus southcentralus eastus northcentralus westus2; do
  echo "===== $r"
  az vm list-skus --location "$r" --resource-type virtualMachines \
    --subscription "$AZ_SUB" \
    --query "[?name=='Standard_NC24ads_A100_v4' || name=='Standard_NC4as_T4_v3' || name=='Standard_NC40ads_H100_v5'].{sku:name, restriction:restrictions[0].reasonCode}" \
    -o table
done
```

A blank result means the SKU is not offered there. `NotAvailableForSubscription` means it exists
but your subscription is not entitled. Either way, do not request quota in that region.

**Measured 2026-09-10 on this subscription** (`Sponsored_2016-01-01`, sub `62173b9a`):

| Region | `NC24ads_A100_v4` | `NC4as_T4_v3` | `NC40ads_H100_v5` |
|---|---|---|---|
| East US 2 | **yes** | yes | no |
| West US 3 | **yes** | yes | no |
| Central US | **yes** | no | no |
| South Central US | **no** | yes | no |
| East US | no | yes | no |
| North Central US | no | yes | no |
| West US 2 | no | yes | no |

No restrictions flagged on any of the above, so availability is purely a quota question.

Two consequences, and they change the plan in 1.3: **South Central US does not offer the A100
at all**, so the "second shot" this doc originally sent there was unwinnable on availability
rather than capacity. The real A100 regions are **East US 2, West US 3, and Central US**. And
the H100 is offered in none of them, so the optional H100 ask has nothing to land on. Drop it.

### 1.2 See where you stand now (expect zeros)

```bash
az vm list-usage --location "$AZ_LOC" --subscription "$AZ_SUB" -o table \
  | grep -iE "NCADS|NCAS|NVADS|Total Regional"
```

Note the **Total Regional vCPUs** line. It is a separate ceiling from the per-family quota, and
if it is too low your approved A100 quota still will not deploy.

**Measured 2026-09-10**, identical in East US 2, West US 3, and Central US:

| Row | Current | Limit |
|---|---|---|
| `Standard NCADS_A100_v4 Family vCPUs` | 0 | **0** |
| `Standard NCASv3_T4 Family vCPUs` | 0 | **0** |
| `Total Regional vCPUs` | 0 | **65** |
| `Total Regional Low-priority vCPUs` | 0 | **3** |

So **Total Regional vCPUs is already 65** and does not need raising — the original ask of 48
here was below what the subscription already had. One fewer request to file and wait on.

The low-priority ceiling of **3** matters for the Spot fallback in 1.4: raising the A100 *family*
quota alone will not let a Spot A100 deploy, because 24 vCPUs will not fit under a regional
low-priority ceiling of 3. If you pivot to Spot, raise `Total Regional Low-priority vCPUs` to
at least 24 as a separate request.

### 1.3 File the requests

The portal is the reliable route. `az quota` exists but its syntax shifts between extension
versions; if you want to try it, run `az extension add --name quota` then
`az quota update --help` and verify against your version first.

> **Attempted 2026-09-10 via CLI (quota extension 1.0.0), and it was auto-rejected.**
>
> ```bash
> az quota update --resource-name "Standard NCASv3_T4 Family" \
>   --scope "/subscriptions/$AZ_SUB/providers/Microsoft.Compute/locations/eastus2" \
>   --limit-object value=8 --resource-type dedicated
> # ERROR: (QuotaNotAvailableForResource) Request failed.
> ```
>
> This was not a syntax problem: `az quota show` on the same `--resource-name` returns the row
> fine (`isQuotaApplicable: true`, limit 0), and the failure is identical with and without
> `--resource-type`. It is the sponsored-subscription auto-rejection described in 1.4, and it
> came back in seconds — even for the *small T4 ask* that is normally rubber-stamped. Treat the
> self-serve path (CLI and portal alike) as exhausted on this subscription and go straight to
> the support ticket in 1.4 step 3.
>
> Note the name mismatch if you go hunting in the portal. Three different spellings are in play
> for the same quota:
>
> | Where | String |
> |---|---|
> | CLI `--resource-name` | `StandardNCADSA100v4Family` |
> | Portal / `localizedValue` | `Standard NCADS_A100_v4 Family vCPUs` |
> | ~~What this doc used to say~~ | ~~`Standard NCADSA100v4 Family vCPUs`~~ (matches neither) |
>
> The T4 is `Standard NCASv3_T4 Family` in the CLI and `Standard NCASv3_T4 Family vCPUs` in the
> portal.

Portal path:

> portal.azure.com → **Subscriptions** → your Founders Hub subscription →
> **Usage + quotas** → set Provider to *Compute* and Region to *East US 2* →
> find the row → **Request adjustment**

Submit these. Since you want **one shared VM**, the A100 ask is exactly one machine's worth,
which is also the ask most likely to be approved.

| Quota row (portal spelling) | New limit | Region | Why |
|---|---|---|---|
| `Standard NCASv3_T4 Family vCPUs` | **8** | East US 2 | One T4 box. Normally the easy one. Your insurance |
| `Standard NCADS_A100_v4 Family vCPUs` | **24** | East US 2 | Exactly one `NC24ads_A100_v4`. The real target |
| `Standard NCADS_A100_v4 Family vCPUs` | **24** | West US 3 | Second shot. Approval is per region |
| `Standard NCADS_A100_v4 Family vCPUs` | **24** | Central US | Third shot |
| `Standard NCASv3_T4 Family vCPUs` | **8** | West US 3 | T4 backup |

There is deliberately **no Central US T4 row**: `Standard_NC4as_T4_v3` is not offered in
Central US at all (see the 1.1 table). In the portal that row renders as `0 of 0` with a
disabled checkbox and a "Request access" icon rather than the usual pencil, which reads
like a permissions problem but is really "this SKU does not exist here". Do not file it.

File all of these on the **same day**, not one after another. Approval is per region and a
rejection in one tells you nothing about another, so serial retries only burn calendar days.

~~`Total Regional vCPUs` = 48 in East US 2~~ — **dropped**, it is already 65. See 1.2.

~~Optional: `Standard NCadsH100v5 Family vCPUs` = 40~~ — **dropped**, the H100 is not offered in
any region checked in 1.1, so there is nothing to grant.

**What to write in the justification box:**

> Carnegie Mellon University graduate coursework (10-718 Machine Learning in Practice).
> Fine-tuning a computer vision model on street-level imagery for a semester project.
> Single VM, shared by a four-person student team. Expected total spend under $1,500.

Naming the institution and asking for a modest, specific amount reads far better than a vague
large request.

### 1.3a Outcome, 2026-09-10: all five self-serve requests rejected

All five valid requests were submitted through the Quotas blade and **all five were
rejected**. Nothing was auto-approved, including the 8 vCPU T4 asks that this doc
expected to sail through.

What the portal actually says — note it is *not* the CLI error, and it is not instant:

> **Panel:** "We were unable to complete 1 request. To follow up on quota increase
> requests, contact the support team."
>
> **Notification:** "We were unable to adjust your quota. Submit a support ticket so
> that a support engineer can assist you in adjusting your quota for your Standard
> NCADS_A100_v4 Family vCPUs in East US 2 for Azure subscription 1."

Each took **one to two minutes**, and the Activity Log records each request as
"Accepted" with no error code attached. The CLI path fails differently and faster:
`az quota update` returns `QuotaNotAvailableForResource` in seconds. Two channels, two
messages, same refusal — quote whichever one you actually saw when you open the ticket,
and do not attribute the CLI code to the portal.

That the trivial T4 ask was refused alongside the A100 is the informative part: the
automated system is declining the **subscription**, not weighing the size of the
request. Retrying regions is not going to help. Go to 1.5 step B.

### 1.4 How long approval takes

| Request | Typical outcome |
|---|---|
| T4, small ask | Often **auto-approved in minutes**. Sometimes a few hours |
| A100, 24 vCPUs | **1 to 3 business days** if it goes to a human. Can be instant |
| Total Regional vCPUs | Usually **minutes**, it is a soft ceiling |
| H100 | **Days to weeks**, and frequently rejected outright |

**Expect the A100 request to possibly bounce.** Founders Hub and other benefit subscriptions are
currently getting auto-rejections with a "high capacity, cannot approve" message, often within
minutes. That is not a reflection on your request.

> **This is what happened, 2026-09-10.** The self-serve request was rejected in seconds with
> `QuotaNotAvailableForResource` — and not just for the A100. The **8-vCPU T4** ask, the one
> this table calls "almost always approved", was refused too. That pattern (even the trivial ask
> bouncing instantly) says the automated system is refusing the *subscription*, not judging the
> *request*. So step 1 below is not worth much here, and steps 3 and 4 are where the real
> options are.

If it happens:

1. Retry in the other region. Approval varies by region and by week. (Of limited value if even
   the small T4 ask was refused instantly — that is a subscription-level refusal.)
2. Request **Spot / low-priority** quota for the same family. It is a separate, easier pool.
   Remember from 1.2 that this needs *two* raises: the A100 family quota **and**
   `Total Regional Low-priority vCPUs`, which starts at 3 and must reach at least 24.
3. Open a free support ticket: portal → **Help + support** → **Create a support request** →
   Issue type **Service and subscription limits (quotas)**. This routes to a human instead of
   the automated system and often succeeds where the self-serve flow failed. **Given the
   instant auto-rejection above, this is now the primary path, not a fallback.** Use the
   justification text below, and say explicitly that the self-serve request returned
   `QuotaNotAvailableForResource` — it tells the engineer the automated route is already
   exhausted and saves a round trip.
4. Fall back to CMU compute. Your department has GPU clusters, and CMU students can get
   Pittsburgh Supercomputing Center Bridges-2 GPU hours through an ACCESS allocation, which is
   a short free application. For this project that may genuinely beat Azure.

**While you wait, work on Colab Pro.** The Day 1 evaluation run and all five crop classifiers
need no Azure at all.

### 1.5 Exact portal steps (self-serve), then the support ticket

Do these in order. The self-serve attempt takes two minutes and, if the automated
system has changed its mind since 2026-09-10, saves you the ticket entirely.

**A. Self-serve, via the Quotas blade**

1. portal.azure.com → search **"Quotas"** in the top bar → open the **Quotas** service
   (this is the newer blade; **Subscriptions → your sub → Usage + quotas** lands in the
   same place).
2. Left nav → **Compute**.
3. Set the three filters at the top:
   - **Subscription** = `Azure subscription 1`
   - **Region** = `East US 2`
   - **Provider** = `Microsoft.Compute`
4. In the search box type `NCADS` — the row is **`Standard NCADS_A100_v4 Family vCPUs`**,
   showing `0 / 0`. (Search `NCASv3` for the T4 row.) Note the underscores; the name in
   the CLI is spelled differently, see 1.3.
5. Tick the row's checkbox → **New Quota Request** → **Enter a new limit**.
6. Enter **24** for the A100 (**8** for the T4). Submit.
7. Repeat steps 3-6 with **Region = West US 3**, then **Region = Central US**.
   Do all regions the same day — approval is per region and a rejection in one tells
   you nothing about another.

If it succeeds you will see the limit change within minutes. If you get
*"Quota not available"* / `QuotaNotAvailableForResource`, or an instant rejection,
go to B. Do not keep retrying — it is refusing the subscription, not the request.

**B. The support ticket (free, and the path that actually works)**

Quota tickets are free on every subscription, including Free/Sponsored. You do **not**
need a paid support plan.

1. portal.azure.com → **Help + support** → **Create a support request**.
2. **Issue type**: `Service and subscription limits (quotas)`.
3. **Subscription**: `Azure subscription 1`.
4. **Quota type**: `Compute-VM (cores-vCPUs) subscription limit increases`.
5. **Next** → **Enter details**. In the details panel set:
   - Deployment model: `Resource Manager`
   - Location: `East US 2`
   - SKU family: `NCADS_A100_v4 Series`
   - New limit: `24`
   - Add a second row for `East US 2` / `NCASv3_T4 Series` / `8`, and rows for
     `West US 3` and `Central US` / `NCADS_A100_v4 Series` / `24`.
6. **Severity**: `C - Minimal impact` is fine and does not slow quota tickets down.
7. Paste the description below.
8. Contact info → your email → **Create**.

**What to write in the description box:**

> Requesting a GPU vCPU quota increase for Carnegie Mellon University graduate
> coursework (10-718 Machine Learning in Practice). We are fine-tuning a computer
> vision model on street-level imagery for a semester project, using a single shared
> VM for a four-person student team. Expected total spend is under $1,500 against
> Microsoft for Startups credits.
>
> Requested: Standard NCADS_A100_v4 Family = 24 vCPUs (exactly one
> Standard_NC24ads_A100_v4), in East US 2, with West US 3 and Central US as
> alternatives if East US 2 has no capacity. Also Standard NCASv3_T4 Family = 8 vCPUs
> in East US 2 as a lower-cost fallback.
>
> The self-serve quota request was already attempted and was rejected automatically
> within seconds, returning QuotaNotAvailableForResource — including for the 8 vCPU
> T4 request. Current limits are 0 for every GPU family. Total Regional vCPUs is
> already 65, so no increase is needed there. I have confirmed via az vm list-skus
> that Standard_NC24ads_A100_v4 is offered and unrestricted in all three regions
> requested, so this is purely a quota question.
>
> If A100 capacity is unavailable, I would accept Spot/low-priority quota for the same
> family instead. In that case please also raise Total Regional Low-priority vCPUs,
> which is currently 3 and would otherwise block a 24 vCPU Spot deployment.

That last paragraph matters: it gives the engineer a cheaper way to say yes, and it
pre-empts the follow-up round trip about the low-priority ceiling.

**C. If both fail**, fall back to CMU compute — see step 4 in 1.4.

---

## Part 2. Give your teammates access

They do **not** need Microsoft for Startups, their own credits, or an existing Azure account.
Credits belong to the subscription; access is a separate system (RBAC).

### 2.1 Create the resource group first

Scope teammate access to this group, never to the whole subscription.

```bash
source ~/sidewalk-env.sh
az group create -n "$AZ_RG" -l "$AZ_LOC" --subscription "$AZ_SUB" \
  --tags project=sidewalk course=10718
```

### 2.2 Invite each teammate as a guest

```bash
az rest --method post \
  --url "https://graph.microsoft.com/v1.0/invitations" \
  --headers "Content-Type=application/json" \
  --body '{
    "invitedUserEmailAddress": "TEAMMATE@andrew.cmu.edu",
    "inviteRedirectUrl": "https://portal.azure.com",
    "sendInvitationMessage": true
  }' \
  --query "invitedUser.id" -o tsv
```

That prints an object ID. **Save it**, you need it in the next step. Repeat per teammate.

Portal equivalent if you prefer clicking:
> **Microsoft Entra ID** → **Users** → **New user** → **Invite external user**

### 2.3 Assign the role

> **RBAC grants are silent.** Azure sends **no email** when you assign a role. The only
> message a teammate receives is the guest invitation from 2.2. They will not be told
> that they now have Contributor on the resource group or Storage Blob Data Contributor
> on the storage account, and they cannot discover it without being told where to look.
> Message the team yourself with: the subscription name, the resource group, the storage
> account name, and which containers they can write to. Otherwise the first symptom is
> someone hitting an opaque 403 and assuming the setup is broken.

```bash
az role assignment create \
  --assignee-object-id "PASTE_OBJECT_ID" \
  --assignee-principal-type User \
  --role "Contributor" \
  --scope "/subscriptions/$AZ_SUB/resourceGroups/$AZ_RG"
```

> **Corrected 2026-09-10.** On azure-cli 2.89.1 this command rejects
> `--resource-group`/`--subscription` here with `ERROR: the following arguments are
> required: --scope`. Pass the full resource-group scope path instead, as above.

`Contributor` on the resource group lets them start, stop, and manage the VM but not grant
access to anyone else or touch anything outside the project. That is the right level.

Verify:

```bash
az role assignment list --resource-group "$AZ_RG" --subscription "$AZ_SUB" \
  --query "[].{who:principalName, role:roleDefinitionName}" -o table
```

### 2.4 Collect their SSH public keys

Ask each teammate to run this and send you the output:

```bash
ssh-keygen -t ed25519 -C "$(whoami)@10718" -f ~/.ssh/sidewalk
cat ~/.ssh/sidewalk.pub
```

Save the three keys locally as `alice.pub`, `bob.pub`, `carol.pub`. You will install them in
Part 4.

**Do not share your Azure password with anyone.** Guest invites are the supported path and take
two minutes.

---

## Part 3. Create the VM

Only once quota is approved.

### 3.1 Confirm the image exists

```bash
az vm image list --publisher microsoft-dsvm --offer ubuntu-hpc --all -o table
```

Pick the newest `2204` SKU. This image ships NVIDIA drivers and CUDA preinstalled, which saves
you an afternoon of driver debugging.

### 3.2 Create it

```bash
source ~/sidewalk-env.sh
az vm create \
  --resource-group "$AZ_RG" \
  --name "$AZ_VM" \
  --subscription "$AZ_SUB" \
  --location "$AZ_LOC" \
  --image microsoft-dsvm:ubuntu-hpc:2204:latest \
  --size Standard_NC24ads_A100_v4 \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/id_ed25519.pub \
  --os-disk-size-gb 256 \
  --os-disk-delete-option Delete \
  --storage-sku Premium_LRS \
  --public-ip-sku Standard \
  --accelerated-networking true \
  --nsg-rule SSH \
  --tags project=sidewalk course=10718 owner=aditya
```

### 3.3 Settings, and why each one

| Setting | Value | Reason |
|---|---|---|
| Size | `Standard_NC24ads_A100_v4` | 1x A100 80GB, 24 vCPU, 220 GiB RAM. The 24 vCPUs matter as much as the GPU: at 4k panorama resolution you will be data-loading bound, and a 4-vCPU box would starve the dataloader |
| Image | `ubuntu-hpc:2204` | Drivers and CUDA preinstalled |
| OS disk | 256 GB Premium SSD | Room for conda envs and checkpoints |
| Data disk | 1 TB Standard SSD (Part 3.4) | Holds the 484 GB of datasets. Standard SSD is fine, disk is not your bottleneck |
| Pricing | **On-demand, not Spot** | With four people sharing one box, a spot eviction mid-session ruins everyone's afternoon. Spot is only worth it for solo unattended runs with checkpointing |
| Accelerated networking | on | Faster pulls from Hugging Face and Blob |
| NSG | SSH only | Locked down further in 3.5 |
| Tags | project/course/owner | Makes the cost dashboard readable |

### 3.4 Attach the data disk

```bash
az vm disk attach \
  -g "$AZ_RG" --vm-name "$AZ_VM" --subscription "$AZ_SUB" \
  --name sidewalk-data --new --size-gb 1024 --sku StandardSSD_LRS
```

### 3.5 Restrict SSH to your team's IPs

By default `--nsg-rule SSH` opens port 22 to the entire internet. Tighten it:

```bash
MYIP=$(curl -s ifconfig.me)
az network nsg rule update \
  -g "$AZ_RG" --nsg-name "${AZ_VM}NSG" -n default-allow-ssh \
  --subscription "$AZ_SUB" \
  --source-address-prefixes "$MYIP/32"
```

Add teammates' IPs as a space-separated list. If people work from changing networks, CMU campus
ranges or just re-running this command is easier than a VPN.

### 3.6 Auto-shutdown

```bash
az vm auto-shutdown \
  -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB" \
  --time 0400 --email "YOUR_EMAIL_HERE"
```

0400 UTC-adjusted so it does not kill someone's evening run. The email gives 30 minutes notice
with a "delay shutdown" link.

### 3.7 Budget alert

> portal.azure.com → **Cost Management + Billing** → **Budgets** → **Add**
> Amount **$1,500**, alerts at 50% and 80%, recipients = all four of you.

### 3.8 Stopping the VM correctly

This is the single most expensive mistake to get wrong:

```bash
# WRONG: stops the OS but you are STILL BILLED for the GPU
az vm stop -g "$AZ_RG" -n "$AZ_VM"

# RIGHT: releases the hardware, billing stops
az vm deallocate -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB"

# start it back up
az vm start -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB"
```

**Always `deallocate`, never `stop`.** A forgotten A100 running for a month is about $2,600.

Note: deallocating wipes the local temp disk (`/mnt`). Never keep anything you care about there.
The attached data disk survives.

---

## Part 4. Set the VM up for four users

SSH in:

```bash
az vm show -d -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB" --query publicIps -o tsv
ssh azureuser@<that-ip>
```

### 4.1 Format and mount the data disk

```bash
lsblk                                  # find the 1TB unpartitioned disk, likely /dev/sdc
DISK=/dev/sdc
sudo parted $DISK --script mklabel gpt mkpart primary ext4 0% 100%
sudo mkfs.ext4 ${DISK}1
sudo mkdir -p /data
sudo mount ${DISK}1 /data
echo "UUID=$(sudo blkid -s UUID -o value ${DISK}1) /data ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
```

### 4.2 Shared group so everyone can write

```bash
sudo groupadd -f sidewalk
sudo chgrp -R sidewalk /data
sudo chmod -R 2775 /data     # setgid: new files inherit the group automatically
```

### 4.3 Create teammate accounts

Copy their `.pub` files up first (`scp alice.pub azureuser@<ip>:/tmp/`), then:

```bash
for u in alice bob carol; do
  sudo adduser --disabled-password --gecos "" $u
  sudo usermod -aG sidewalk $u
  sudo mkdir -p /home/$u/.ssh
  sudo cp /tmp/$u.pub /home/$u/.ssh/authorized_keys
  sudo chown -R $u:$u /home/$u/.ssh
  sudo chmod 700 /home/$u/.ssh
  sudo chmod 600 /home/$u/.ssh/authorized_keys
done
sudo usermod -aG sidewalk azureuser
```

Password-less sudo is deliberately not granted. If someone needs to install a system package,
they ask you. Keeps the shared box from drifting.

### 4.4 Sharing one GPU between four people

You have one A100. Four people cannot train on it simultaneously without thrashing. Simplest
workable convention, in order of preference:

1. **`tmux` for every long job.** Sessions survive disconnects.
   ```bash
   tmux new -s alice-train
   # ctrl-b d to detach, tmux attach -t alice-train to return
   ```
2. **Check before you start.** `nvidia-smi` shows who is using what. Add a `/data/GPU_CLAIM.txt`
   file that whoever is training edits with their name and expected finish time. Low-tech, works.
3. **Stagger the work.** Only stage-two training needs the full GPU. Data prep, evaluation, and
   crop-classifier work can run concurrently on the same card with room to spare.

If contention becomes a real problem later, the A100 80GB supports **MIG**, which partitions it
into independent instances (for example two 40 GB slices). Check available profiles with
`nvidia-smi mig -lgip`. It requires no running GPU processes to reconfigure and it lowers peak
performance for any single big job, so do not do this on day one. Only reach for it if the
calendar approach genuinely fails.

### 4.5 Base environment

```bash
sudo apt update && sudo apt install -y tmux git git-lfs htop
nvidia-smi                                    # confirm the A100 is visible
curl -LsSf https://astral.sh/uv/install.sh | sh

echo 'export HF_HOME=/data/hf_cache' | sudo tee /etc/profile.d/sidewalk.sh
source /etc/profile.d/sidewalk.sh

git clone https://github.com/ProjectSidewalk/RampNet.git /data/RampNet
cd /data/RampNet
```

Install the pinned requirements (the repo pins PyTorch >= 2.6, < 3 and CUDA 12.6), then pull
data **directly onto the VM**, never through your laptop. Azure ingress is free.

```bash
pip install -U "huggingface_hub[cli]"

hf download projectsidewalk/rampnet-model      --local-dir /data/models/rampnet
hf download projectsidewalk/rampnet-benchmark  --repo-type dataset --local-dir /data/rampnet-benchmark
```

(If your `huggingface_hub` is older, the command is `huggingface-cli download` with the same args.)

Then the five crop datasets (9.5 GB total) and, only if you need full stage-two training,
`projectsidewalk/rampnet-dataset` (462 GB, 386 shards).

---

## Part 5. Permanent storage: Blob for data and artifacts, Git for code

The VM is disposable. Treat `/data` on it as a scratch working copy that you could lose at any
time, because you can: delete the VM without detaching the disk and it goes with it. Everything
you would be upset to lose lives in one of two places.

### 5.0 What goes where

Decide this once and hold the line, otherwise the two stores drift apart and you cannot
reproduce anything in week 12.

| Thing | Where | Why |
|---|---|---|
| Code, configs, notebooks | **Git repo** | Diffable, reviewable, four people editing |
| Shard manifests, label filters, split definitions | **Git repo** (small text files) | These *define* an experiment. They belong with the code |
| Raw datasets (462 GB + 21 GB) | **Blob: `datasets`** | Too big for git, downloaded once, never edited |
| Model checkpoints | **Blob: `checkpoints/<run_id>/`** | Big, binary, per-run |
| Predictions, metrics, figures, logs | **Blob: `results/<run_id>/`** | The evidence for your report |
| Live metrics during training | **Weights & Biases** (free academic tier) | Better than tailing log files with four people |
| `/data` on the VM | **Working copy only** | Assume it vanishes |

Never put checkpoints or datasets in git, and do not reach for git-lfs here. You already have
blob storage, and lfs on a student repo will hit quota limits fast.

### 5.1 Create the storage account and containers

Add to `~/sidewalk-env.sh`:

```bash
export AZ_SA="sidewalkdata$RANDOM"   # run once, then hardcode the value it picks
```

Storage account names must be globally unique, 3 to 24 characters, lowercase letters and digits
only. Generate once, then paste the literal name back into the env file so it is stable.

```bash
source ~/sidewalk-env.sh
az storage account create \
  -n "$AZ_SA" -g "$AZ_RG" -l "$AZ_LOC" --subscription "$AZ_SUB" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --access-tier Hot \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false \
  --tags project=sidewalk course=10718
```

`Standard_LRS` is the cheapest redundancy tier and is correct here: your datasets are
re-downloadable from Hugging Face and your checkpoints are regenerable. Do not pay for GRS.

### 5.2 Give yourself data-plane access (the step everyone misses)

**`Contributor` on the resource group lets you manage the storage account but not read or write
a single blob.** Management plane and data plane are separate in Azure. Grant yourself the data
role explicitly:

```bash
SA_SCOPE=$(az storage account show -n "$AZ_SA" -g "$AZ_RG" --subscription "$AZ_SUB" --query id -o tsv)
MY_ID=$(az ad signed-in-user show --query id -o tsv)

az role assignment create \
  --assignee-object-id "$MY_ID" --assignee-principal-type User \
  --role "Storage Blob Data Contributor" \
  --scope "$SA_SCOPE"
```

Now create the containers:

```bash
for c in datasets checkpoints results; do
  az storage container create --account-name "$AZ_SA" -n "$c" --auth-mode login
done
```

### 5.3 Do the same for each teammate

Reuse the object IDs you saved from the guest invitations in Part 2.2:

```bash
for OID in OBJ_ID_ALICE OBJ_ID_BOB OBJ_ID_CAROL; do
  az role assignment create \
    --assignee-object-id "$OID" --assignee-principal-type User \
    --role "Storage Blob Data Contributor" \
    --scope "$SA_SCOPE"
done
```

Without this they will get opaque 403s and assume something is broken.

### 5.4 Give the VM keyless access with a managed identity

Do not put storage account keys or SAS tokens on a shared VM that four people can read. Use a
managed identity instead, which needs no secrets at all.

```bash
VM_OID=$(az vm identity assign -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB" \
  --query systemAssignedIdentity -o tsv)

az role assignment create \
  --assignee-object-id "$VM_OID" --assignee-principal-type ServicePrincipal \
  --role "Storage Blob Data Contributor" \
  --scope "$SA_SCOPE"
```

### 5.5 Install azcopy on the VM

`azcopy` is dramatically faster than `az storage blob upload` for bulk transfers. On the VM:

```bash
wget -qO azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
tar -xzf azcopy.tar.gz --strip-components=1 -C /tmp
sudo mv /tmp/azcopy /usr/local/bin/ && sudo chmod +x /usr/local/bin/azcopy
azcopy --version

azcopy login --identity     # uses the managed identity, no secrets
```

Every user on the VM can run `azcopy login --identity` and it just works.

### 5.6 Upload the datasets once

Download from Hugging Face straight onto the VM, then push a durable copy to blob so you never
pay the download time again (VM to blob in the same region is free):

```bash
source /etc/profile.d/sidewalk.sh          # sets HF_HOME=/data/hf_cache
hf download projectsidewalk/rampnet-benchmark --repo-type dataset --local-dir /data/rampnet-benchmark

azcopy copy "/data/rampnet-benchmark/*" \
  "https://${AZ_SA}.blob.core.windows.net/datasets/rampnet-benchmark/" \
  --recursive
```

Pulling it back onto a fresh VM later:

```bash
azcopy copy "https://${AZ_SA}.blob.core.windows.net/datasets/rampnet-benchmark/*" \
  /data/rampnet-benchmark/ --recursive
```

### 5.6a Loading blob before the VM exists (server-side copy)

The flow above assumes a VM. Before quota lands there isn't one, and pushing data
through a laptop is the thing this project explicitly does not want -- it is slow, it
burns home bandwidth, and the 462 GB dataset will not fit on a laptop disk anyway.

Azure Blob can fetch from a public URL **server-side**, so Azure downloads from Hugging
Face directly and no bytes touch your machine. One wrinkle, found the hard way:

```
ERROR: A redirected response (HTTP status code 307) from the copy source is not supported.
ErrorCode:CannotVerifyCopySource
```

Every HF `resolve/main/...` URL is a 307 to their CDN, and Azure refuses to follow it.
The fix is to resolve the redirect yourself and hand Azure the final CDN URL.
`scripts/hf_to_blob.py` in the project repo does this for a whole repo:

```bash
source ~/sidewalk-env.sh
python scripts/hf_to_blob.py --repo projectsidewalk/rampnet-benchmark \
    --repo-type dataset --dest rampnet-benchmark
```

It skips blobs that already exist, so re-running it resumes rather than restarting.
Measured 2026-09-10: `rampnet-model` (0.36 GB, 6 files) landed in about 45 seconds.

Once the VM exists, prefer the `azcopy` flow above for anything already on local disk --
this script is specifically for *remote source to blob* with no middleman.

A note on **BlobFuse2**, which mounts blob as a filesystem: it is tempting but do not train
directly off it. Random-access reads of hundreds of thousands of JPEGs over a network mount will
be far slower than your GPU. Copy to the local SSD to train, use blob as the durable store. If
you want blobfuse for convenience, use it only for writing checkpoints and results.

### 5.7 The run convention that ties code to artifacts

This is the part that actually makes experiments reproducible. Every run gets an ID and a
manifest, and the manifest carries the **git commit SHA**. That single field is the link between
your two storage systems.

```
checkpoints/<run_id>/  epoch_*.pt, best.pt, run_manifest.json
results/<run_id>/      metrics.json, predictions.parquet, figures/, train.log
```

Use a readable run ID: `20260915-multiclass-lr3e4-alice`.

Write `run_manifest.json` at the start of every run:

```json
{
  "run_id": "20260915-multiclass-lr3e4-alice",
  "git_sha": "a1b2c3d",
  "git_dirty": false,
  "started_utc": "2026-09-15T14:02:11Z",
  "author": "alice",
  "base_checkpoint": "projectsidewalk/rampnet-model",
  "dataset": "rampnet-dataset",
  "shard_manifest": "configs/shards_150.txt",
  "classes": ["CurbRamp", "NoCurbRamp", "Obstacle", "SurfaceProblem", "NoSidewalk"],
  "hparams": {"lr": 3e-4, "batch_size": 8, "epochs": 3, "amp": "bf16"}
}
```

**If `git_dirty` is true, the run is not reproducible.** Commit before you launch. Make that a
team rule on day one, not week ten.

### 5.8 A sync script

Save as `/data/RampNet/scripts/sync_run.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:?usage: sync_run.sh <run_id>}"
SA="${AZ_SA:?set AZ_SA}"
BASE="https://${SA}.blob.core.windows.net"

azcopy copy "/data/runs/${RUN_ID}/checkpoints/*" "${BASE}/checkpoints/${RUN_ID}/" --recursive
azcopy copy "/data/runs/${RUN_ID}/results/*"     "${BASE}/results/${RUN_ID}/"     --recursive
echo "synced ${RUN_ID}"
```

```bash
chmod +x /data/RampNet/scripts/sync_run.sh
```

Run it when a job finishes, or call it from your training loop every N epochs so a VM failure
never costs you more than N epochs.

### 5.9 Git repo setup

Create the repo, add all four of you as collaborators, and commit this `.gitignore` first:

```gitignore
# data and artifacts live in blob, never here
data/
*.pt
*.pth
*.ckpt
*.safetensors
*.zip
*.parquet
hf_cache/
runs/
wandb/

# notebooks: strip outputs before committing
.ipynb_checkpoints/

# local
.env
.DS_Store
__pycache__/
*.egg-info/
```

Suggested structure:

```
sidewalk-10718/
  configs/          shard manifests, hyperparameter yamls  <- IN GIT, they define experiments
  src/              data loading, model, training, eval
  scripts/          sync_run.sh, setup_vm.sh, download_data.sh
  notebooks/        exploration, outputs stripped
  reports/          figures and tables for the writeup
  README.md         how to reproduce any run from its run_id
```

Strip notebook outputs automatically so four people do not generate merge conflicts on base64
image blobs:

```bash
pip install nbstripout && nbstripout --install
```

### 5.10 What this costs

Hot LRS blob is about **$0.018 per GB per month**. Realistic footprint:

| | Size | Per month |
|---|---|---|
| `datasets` (benchmark + crops + stage1 inputs) | ~22 GB | $0.40 |
| `datasets` (full rampnet-dataset, only if needed) | 462 GB | $8.30 |
| `checkpoints` (say 30 runs x 2 GB) | ~60 GB | $1.10 |
| `results` | ~5 GB | $0.09 |

Under $10 a month even with the full dataset staged. Transactions are pennies. VM to blob within
the same region is free; you are only charged egress if you pull data *out* of Azure, so download
figures and metrics rather than checkpoints.

---

## Quick reference

```bash
source ~/sidewalk-env.sh                                              # ALWAYS FIRST
az account show --query name -o tsv                                   # verify correct subscription
az vm start      -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB"
az vm deallocate -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB"     # stops billing
az vm show -d    -g "$AZ_RG" -n "$AZ_VM" --subscription "$AZ_SUB" --query publicIps -o tsv
az consumption usage list --subscription "$AZ_SUB" --top 5 -o table   # what you have spent

# on the VM
azcopy login --identity                                               # keyless blob auth
./scripts/sync_run.sh <run_id>                                        # push checkpoints + results
azcopy copy "https://${AZ_SA}.blob.core.windows.net/datasets/*" /data/ --recursive   # restore data
```

**Cost at a glance:** `NC24ads_A100_v4` is about **$3.673/hr** on-demand. Running it 8 hours a
day, 5 days a week, for 14 weeks is roughly **$2,060**. Storage adds about $80/month. You have
$10,000 expiring 8 Sept 2028, so budget is not your constraint. Quota approval and time are.
