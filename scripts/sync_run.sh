#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:?usage: sync_run.sh <run_id>}"
SA="${AZ_SA:?set AZ_SA}"
BASE="https://${SA}.blob.core.windows.net"

azcopy copy "/data/runs/${RUN_ID}/checkpoints/*" "${BASE}/checkpoints/${RUN_ID}/" --recursive
azcopy copy "/data/runs/${RUN_ID}/results/*"     "${BASE}/results/${RUN_ID}/"     --recursive
echo "synced ${RUN_ID}"
