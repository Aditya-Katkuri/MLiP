#!/usr/bin/env bash
# Runbook Part 4 -- run this ON THE VM, once, as azureuser.
#   sudo ./setup_vm.sh /dev/sdc alice bob carol
#
# Formats the given disk (DESTRUCTIVE -- it refuses if the disk is mounted or
# already holds a filesystem), mounts it at /data, creates the shared group and
# the teammate accounts, and installs the base tooling.
set -euo pipefail

DISK="${1:?usage: setup_vm.sh <disk e.g. /dev/sdc> [user ...]}"
shift
USERS=("$@")

[[ $EUID -eq 0 ]] || { echo "run with sudo" >&2; exit 1; }

# ---------- 4.1 format and mount the data disk ----------
if ! mountpoint -q /data; then
  [[ -b "$DISK" ]] || { echo "$DISK is not a block device. Check lsblk." >&2; exit 1; }
  if lsblk -no MOUNTPOINT "$DISK" | grep -q .; then
    echo "REFUSING: $DISK (or a partition of it) is mounted." >&2; exit 1
  fi
  if blkid "$DISK"* >/dev/null 2>&1; then
    echo "REFUSING: $DISK already has a filesystem/partition table:" >&2
    blkid "$DISK"* >&2
    echo "If this really is the fresh 1TB data disk, wipe it deliberately by hand." >&2
    exit 1
  fi
  echo ">>> partitioning and formatting $DISK"
  parted "$DISK" --script mklabel gpt mkpart primary ext4 0% 100%
  udevadm settle
  mkfs.ext4 -F "${DISK}1"
  mkdir -p /data
  mount "${DISK}1" /data
  UUID=$(blkid -s UUID -o value "${DISK}1")
  grep -q "$UUID" /etc/fstab || \
    echo "UUID=$UUID /data ext4 defaults,nofail 0 2" >> /etc/fstab
else
  echo ">>> /data already mounted, skipping format"
fi

# ---------- 4.2 shared group ----------
echo ">>> shared group"
groupadd -f sidewalk
chgrp -R sidewalk /data
chmod -R 2775 /data          # setgid: new files inherit the group

# ---------- 4.3 teammate accounts ----------
for u in "${USERS[@]:-}"; do
  [[ -n "$u" ]] || continue
  echo ">>> user $u"
  id "$u" >/dev/null 2>&1 || adduser --disabled-password --gecos "" "$u"
  usermod -aG sidewalk "$u"
  if [[ -f "/tmp/$u.pub" ]]; then
    mkdir -p "/home/$u/.ssh"
    cp "/tmp/$u.pub" "/home/$u/.ssh/authorized_keys"
    chown -R "$u:$u" "/home/$u/.ssh"
    chmod 700 "/home/$u/.ssh"
    chmod 600 "/home/$u/.ssh/authorized_keys"
  else
    echo "    WARNING: /tmp/$u.pub not found -- $u has no way to log in yet." >&2
  fi
done
usermod -aG sidewalk azureuser
# No password-less sudo, deliberately. System packages go through the owner.

# ---------- 4.5 base environment ----------
echo ">>> base packages"
apt-get update -qq
apt-get install -y -qq tmux git git-lfs htop jq
nvidia-smi || echo "WARNING: nvidia-smi failed -- GPU not visible" >&2

echo 'export HF_HOME=/data/hf_cache' > /etc/profile.d/sidewalk.sh
mkdir -p /data/hf_cache
chgrp -R sidewalk /data/hf_cache && chmod -R 2775 /data/hf_cache

# ---------- 5.5 azcopy ----------
if ! command -v azcopy >/dev/null; then
  echo ">>> azcopy"
  wget -qO /tmp/azcopy.tar.gz https://aka.ms/downloadazcopy-v10-linux
  tar -xzf /tmp/azcopy.tar.gz --strip-components=1 -C /tmp
  mv /tmp/azcopy /usr/local/bin/ && chmod +x /usr/local/bin/azcopy
fi
azcopy --version

# ---------- shared-GPU claim file (4.4) ----------
[[ -f /data/GPU_CLAIM.txt ]] || {
  printf 'who\tstarted (UTC)\texpected finish\tjob\n' > /data/GPU_CLAIM.txt
  chgrp sidewalk /data/GPU_CLAIM.txt && chmod 664 /data/GPU_CLAIM.txt
}

echo
echo "done. Next, as each user:  azcopy login --identity"
echo "Long jobs go in tmux. Claim the GPU in /data/GPU_CLAIM.txt. Check nvidia-smi first."
