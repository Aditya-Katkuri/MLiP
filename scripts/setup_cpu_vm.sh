#!/usr/bin/env bash
# Set up the CPU data-processing VM (Standard_E64ads_v7). Run ON THE VM:
#   sudo ./setup_cpu_vm.sh
#
# Safe to re-run, and you WILL need to: the local NVMe scratch array is
# ephemeral and is destroyed every time the VM is deallocated. The persistent
# /data disk is formatted only once and is never touched again.
#
# Storage layout this produces:
#   /mnt/scratch   ~3.4 TB  RAID0 over 4 local NVMe   EPHEMERAL, wiped on deallocate
#   /data           2.0 TB  managed StandardSSD       persists across deallocate
#   blob                     durable store             the only real source of truth
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }

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

# ---------- shared group so all four of us can write ----------
groupadd -f sidewalk
for d in /data /mnt/scratch; do
  [[ -d $d ]] || continue
  chgrp -R sidewalk "$d"; chmod -R 2775 "$d"     # setgid: new files inherit the group
done
usermod -aG sidewalk azureuser

# ---------- tooling ----------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq tmux git git-lfs htop jq parallel build-essential

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
export PATH=/opt/miniconda/bin:$PATH
export HF_HOME=/mnt/scratch/hf_cache      # scratch: re-downloadable, wiped on deallocate
export HF_HUB_ENABLE_HF_TRANSFER=1
export AZ_SA=sidewalkdata23770
PROF
mkdir -p /mnt/scratch/hf_cache 2>/dev/null || true
chgrp -R sidewalk /mnt/scratch/hf_cache 2>/dev/null || true
chmod -R 2775 /mnt/scratch/hf_cache 2>/dev/null || true

echo
echo "done."
df -h /data /mnt/scratch 2>/dev/null || true
echo
echo "Next, as your own user:"
echo "  azcopy login --identity        # keyless, uses the VM managed identity"
echo "  conda env create -f environment.yml && conda activate sidewalk"
echo
echo "REMEMBER: /mnt/scratch is wiped on deallocate. Anything you care about"
echo "goes to blob. Re-run this script after each VM start to rebuild scratch."
