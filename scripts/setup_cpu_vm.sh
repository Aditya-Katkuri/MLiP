#!/usr/bin/env bash
# Set up the CPU data-processing VM (Standard_E64ads_v7). Run ON THE VM:
#
#   sudo sidewalk-setup                  full setup (first boot, or after changes)
#   sudo sidewalk-setup --scratch-only   just rebuild the ephemeral scratch array
#
# Installed to /usr/local/bin/sidewalk-setup, which lives on the OS disk and so
# survives deallocation. A systemd unit (sidewalk-scratch.service) runs the
# --scratch-only path at every boot, so /mnt/scratch is normally already there
# by the time you log in and you never have to remember this.
#
# Safe to re-run. The persistent /data disk is formatted only once, guarded, and
# never touched again.
#
# Storage layout this produces:
#   /mnt/scratch   ~3.4 TB  RAID0 over 4 local NVMe   EPHEMERAL, wiped on deallocate
#   /data           2.0 TB  managed StandardSSD       persists across deallocate
#   blob                     durable store             the only real source of truth
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }

SCRATCH_ONLY=0
[[ "${1:-}" == "--scratch-only" ]] && SCRATCH_ONLY=1

DATA_DISK=/dev/disk/azure/data/by-lun/0

# ---------- persistent data disk -> /data (format once, guarded) ----------
if ! mountpoint -q /data; then
  DEV=$(readlink -f "$DATA_DISK")
  if blkid "${DEV}" >/dev/null 2>&1; then
    echo ">>> /data disk already has a filesystem, mounting as-is"
  else
    echo ">>> formatting persistent data disk $DEV (one time only)"
    mkfs.ext4 -F -L sidewalk-data "$DEV"
  fi
  mkdir -p /data
  mount "$DEV" /data
  UUID=$(blkid -s UUID -o value "$DEV")
  grep -q "$UUID" /etc/fstab || echo "UUID=$UUID /data ext4 defaults,nofail 0 2" >> /etc/fstab
else
  echo ">>> /data already mounted"
fi

# ---------- ephemeral local NVMe -> /mnt/scratch (RAID0, rebuilt each boot) ----------
if ! mountpoint -q /mnt/scratch; then
  mapfile -t LOCALS < <(ls /dev/disk/azure/local/by-index/* 2>/dev/null | sort -V | xargs -r -n1 readlink -f)
  if (( ${#LOCALS[@]} )); then
    echo ">>> building RAID0 scratch over ${#LOCALS[@]} local NVMe: ${LOCALS[*]}"
    command -v mdadm >/dev/null || { apt-get update -qq; apt-get install -y -qq mdadm; }
    mdadm --stop /dev/md0 2>/dev/null || true
    for d in "${LOCALS[@]}"; do wipefs -a "$d" >/dev/null 2>&1 || true; done
    mdadm --create /dev/md0 --run --level=0 --raid-devices="${#LOCALS[@]}" "${LOCALS[@]}"
    mkfs.ext4 -F -L scratch /dev/md0
    mkdir -p /mnt/scratch
    mount /dev/md0 /mnt/scratch
    # deliberately NOT in fstab: these disks are wiped on deallocate and a stale
    # fstab entry would hang the next boot. Re-run this script instead.
  else
    echo "WARNING: no local NVMe found" >&2
  fi
else
  echo ">>> /mnt/scratch already mounted"
fi

# ---------- shared access so all four of us can write ----------
# Two mechanisms, because there are two kinds of account on this box:
#   * local users (azureuser)      -> the `sidewalk` group
#   * Microsoft Entra SSH logins   -> the `aad_admins` group
# Entra accounts are created on first login, so they can never be added to
# `sidewalk` ahead of time. A default ACL for aad_admins covers them
# automatically, including people who have not logged in yet.
command -v setfacl >/dev/null || apt-get install -y -qq acl
groupadd -f sidewalk
for d in /data /mnt/scratch; do
  [[ -d $d ]] || continue
  chgrp -R sidewalk "$d"; chmod -R 2775 "$d"     # setgid: new files inherit the group
  setfacl -R    -m g:aad_admins:rwx "$d"          # existing files
  setfacl -R -d -m g:aad_admins:rwx "$d"          # and anything created later
done
usermod -aG sidewalk azureuser

# ---------- everything below is one-time setup; skipped at boot ----------
# It lives on the OS disk, which survives deallocation, so re-running it at every
# boot would only waste time -- and running apt at boot races unattended-upgrades
# for the dpkg lock.
if (( SCRATCH_ONLY )); then
  echo "scratch ready:"; df -h /mnt/scratch 2>/dev/null || true
  exit 0
fi

# ---------- tooling ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq tmux git git-lfs htop jq parallel build-essential acl

if ! command -v azcopy >/dev/null; then
  wget -qO /tmp/azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
  tar -xzf /tmp/azcopy.tar.gz --strip-components=1 -C /tmp
  mv /tmp/azcopy /usr/local/bin/ && chmod +x /usr/local/bin/azcopy
fi

if [[ ! -d /opt/miniconda ]]; then
  wget -qO /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/mc.sh -b -p /opt/miniconda
  chgrp -R sidewalk /opt/miniconda && chmod -R 2775 /opt/miniconda
fi

cat > /etc/profile.d/sidewalk.sh <<'PROF'
# Source conda's shell hook rather than prepending /opt/miniconda/bin to PATH.
# Two reasons:
#   1. `conda activate` needs the shell function this defines; with PATH alone
#      it fails with "Run 'conda init' before 'conda activate'".
#   2. Prepending the whole conda bin dir shadows system tools with conda's own
#      builds. conda's `clear` links against libtinfow.so.6, which is not on the
#      default loader path, so `clear` dies with a missing-library error.
# The hook gives you conda without hijacking PATH.
if [ -f /opt/miniconda/etc/profile.d/conda.sh ]; then
    . /opt/miniconda/etc/profile.d/conda.sh
fi

export HF_HOME=/mnt/scratch/hf_cache      # scratch: re-downloadable, wiped on deallocate
export HF_HUB_ENABLE_HF_TRANSFER=1
export AZ_SA=sidewalkdata23770
PROF
mkdir -p /mnt/scratch/hf_cache 2>/dev/null || true
chgrp -R sidewalk /mnt/scratch/hf_cache 2>/dev/null || true
chmod -R 2775 /mnt/scratch/hf_cache 2>/dev/null || true

# ---------- install ourselves persistently + a boot unit ----------
# /tmp is cleared on reboot, so a copy of this script left there disappears
# exactly when it is next needed. Install to the OS disk instead.
SELF=$(readlink -f "$0")
if [[ "$SELF" != /usr/local/bin/sidewalk-setup ]]; then
  install -m 0755 "$SELF" /usr/local/bin/sidewalk-setup
  echo ">>> installed to /usr/local/bin/sidewalk-setup"
fi

cat > /etc/systemd/system/sidewalk-scratch.service <<'UNIT'
[Unit]
Description=Rebuild the ephemeral NVMe scratch array for the sidewalk project
# The local NVMe disks are destroyed on every deallocate, so /mnt/scratch has to
# be recreated at each boot. Deliberately not an fstab entry: a stale fstab line
# pointing at a wiped disk can hang the boot.
After=local-fs.target
Requires=local-fs.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/sidewalk-setup --scratch-only
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable sidewalk-scratch.service >/dev/null 2>&1
echo ">>> enabled sidewalk-scratch.service (rebuilds scratch at every boot)"

echo
echo "done."
df -h /data /mnt/scratch 2>/dev/null || true
echo
echo "Next, as your own user:"
echo "  azcopy login --identity        # keyless, uses the VM managed identity"
echo "  conda env create -f environment.yml && conda activate sidewalk"
echo
echo "/mnt/scratch is rebuilt automatically at boot now. It is still WIPED on"
echo "every stop, so anything you care about goes to blob."
