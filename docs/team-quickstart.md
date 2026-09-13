# Quickstart: getting onto the CPU box

Short version. The full reference is [cpu-vm.md](cpu-vm.md).

---

Hey — the shared CPU box for our 10-718 project is ready. Here's how to get on it.

It's a 64-core / 512 GB machine in Azure, in the same region as our datasets, so it
reads all 474 GB of them at full speed. No GPU on it — training happens elsewhere.
Full guide is in the repo at `docs/cpu-vm.md`.

**One-time setup (10 min)**

1. Accept the Azure invite that was emailed to your andrew.cmu.edu address.
   (All three of you have already accepted, so you're good.)
2. Install the Azure CLI — https://learn.microsoft.com/cli/azure/install-azure-cli
   On Mac: `brew install azure-cli`
3. Install the **Remote - SSH** extension in VS Code (publisher: Microsoft).
4. Run these:

   az extension add --name ssh
   az login
   az ssh config --file ~/.ssh/config -n sidewalk-cpu -g sidewalk-rg

There's no SSH key to set up and no VPN. You log in with your Andrew Microsoft
account, from anywhere.

**Every time you want to use it**

1. Go to portal.azure.com, search `sidewalk-cpu`, open it, click **Start**.
   Wait ~60s for Status to say **Running**. Any of us can start it.
2. In VS Code: Cmd+Shift+P → "Remote-SSH: Connect to Host" → pick `sidewalk-cpu`.
3. Open a terminal in that VS Code window and run:

   sudo /tmp/setup_cpu_vm.sh

   ^ Do this EVERY session. The fast scratch disk gets wiped each time the VM
   stops, and this rebuilds it. If /mnt/scratch is missing, this is why.
4. Do your work. File → Open Folder → /data or /mnt/scratch
5. **When you're done, go back to the portal and click Stop.**

**Please actually stop it.** It costs $4.65/hour while running — about $110/day if
someone forgets. There's an auto-shutdown at midnight ET as a backstop but don't
lean on it. Before you stop it, run `who` to check nobody else is connected —
stopping it kills their jobs.

**Where to put files**

- /mnt/scratch (3.4 TB, fastest) — working files. WIPED every time the VM stops.
- /data (2 TB) — survives stops, but dies with the VM.
- Blob storage — the only thing that's actually safe. Anything you'd be upset to
  lose goes here. From the VM: `azcopy login --identity` then `azcopy copy ...`

Our datasets are already in blob: the full 462 GB training set, the 11 GB
benchmark, and the pretrained checkpoint.

**Two things that look like bugs but aren't**

- VS Code connected fine yesterday, refuses today → your certificate expired,
  they're short-lived. Re-run the `az ssh config` command from step 4 above.
- /mnt/scratch is empty → the VM was stopped. Run the setup script.

**Long jobs: use tmux** (`tmux new -s yourname`) so they survive your laptop
closing or wifi dropping. `tmux ls` shows everyone's sessions.

Shout if anything doesn't work.
