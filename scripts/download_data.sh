#!/usr/bin/env bash
# Runbook Parts 4.5 / 5.6 -- run this ON THE VM.
# Pulls from Hugging Face straight onto the VM (never through your laptop --
# Azure ingress is free), then pushes a durable copy to blob so the download
# is never paid for twice. VM -> blob in the same region is free.
#
#   ./download_data.sh model benchmark        # the day-one set, ~21 GB
#   ./download_data.sh full-dataset           # 462 GB, 386 shards. Only if you
#                                             # actually need stage-two training.
set -euo pipefail

: "${AZ_SA:?set AZ_SA (see runbook 5.1)}"
BASE="https://${AZ_SA}.blob.core.windows.net"
source /etc/profile.d/sidewalk.sh   # HF_HOME=/data/hf_cache

command -v hf >/dev/null || pip install -U "huggingface_hub[cli]"
# older huggingface_hub: the command is `huggingface-cli download`, same args
HF=$(command -v hf || command -v huggingface-cli)

push() {  # push <local dir> <blob prefix>
  echo ">>> azcopy $1 -> ${BASE}/datasets/$2/"
  azcopy copy "$1/*" "${BASE}/datasets/$2/" --recursive
}

for what in "$@"; do
  case "$what" in
    model)
      "$HF" download projectsidewalk/rampnet-model --local-dir /data/models/rampnet
      ;;
    benchmark)
      "$HF" download projectsidewalk/rampnet-benchmark --repo-type dataset \
        --local-dir /data/rampnet-benchmark
      push /data/rampnet-benchmark rampnet-benchmark
      ;;
    full-dataset)
      echo "462 GB across 386 shards. This takes a while. Run it in tmux."
      read -rp "type YES to continue: " ok; [[ "$ok" == YES ]] || exit 1
      "$HF" download projectsidewalk/rampnet-dataset --repo-type dataset \
        --local-dir /data/rampnet-dataset
      push /data/rampnet-dataset rampnet-dataset
      ;;
    *)
      echo "unknown target: $what (model|benchmark|full-dataset)" >&2; exit 1
      ;;
  esac
done

echo "done. Restore onto a fresh VM with:"
echo "  azcopy copy \"${BASE}/datasets/<name>/*\" /data/<name>/ --recursive"
