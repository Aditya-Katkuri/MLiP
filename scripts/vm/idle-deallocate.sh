#!/usr/bin/env bash
# Deallocate this VM after IDLE_HOURS with no SSH connections and a quiet CPU.
# Run every 5 min by sidewalk-idle.timer. Deallocating (not just shutting down the
# OS) is what stops billing, so this calls Azure with the VM's managed identity,
# which needs the "Virtual Machine Contributor" role on this VM.
set -euo pipefail

IDLE_HOURS=${IDLE_HOURS:-3}
LOAD_MAX=${LOAD_MAX:-2.0}             # 15-min load avg; a running job on 64 cores sits well above
STATE=/run/sidewalk-idle-since         # /run is cleared on boot, so the clock restarts after a start
DRY_RUN=${DRY_RUN:-0}

ssh_conns=$(ss -Htn state established '( sport = :22 )' | wc -l)
load15=$(cut -d' ' -f3 /proc/loadavg)

# `touch /data/KEEP_AWAKE` to hold the VM up for a long low-CPU job (e.g. a download).
if [ -e /data/KEEP_AWAKE ] || [ "$ssh_conns" -gt 0 ] || awk -v l="$load15" -v m="$LOAD_MAX" 'BEGIN{exit !(l>=m)}'; then
    rm -f "$STATE"
    exit 0
fi

now=$(date +%s)
[ -f "$STATE" ] || echo "$now" > "$STATE"
idle_s=$(( now - $(cat "$STATE") ))
limit_s=$(( IDLE_HOURS * 3600 ))
logger -t sidewalk-idle "idle ${idle_s}s of ${limit_s}s (ssh=$ssh_conns load15=$load15)"
[ "$idle_s" -ge "$limit_s" ] || exit 0

imds=http://169.254.169.254/metadata
rid=$(curl -sf -H Metadata:true "$imds/instance/compute/resourceId?api-version=2021-02-01&format=text")
if [ "$DRY_RUN" = 1 ]; then
    logger -t sidewalk-idle "DRY_RUN: would deallocate $rid"
    exit 0
fi
token=$(curl -sf -H Metadata:true \
    "$imds/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])')
logger -t sidewalk-idle "idle ${IDLE_HOURS}h, deallocating $rid"
curl -sf -X POST -H "Authorization: Bearer $token" -H "Content-Length: 0" \
    "https://management.azure.com${rid}/deallocate?api-version=2024-07-01" \
    || logger -t sidewalk-idle "deallocate request FAILED (does the VM identity have Virtual Machine Contributor?)"
