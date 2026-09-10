#!/usr/bin/env bash
# Runbook Parts 0.4, 1.1, 1.2 in one command. Run on your laptop, after `az login`.
#   source ~/sidewalk-env.sh && ./scripts/az_preflight.sh
# Read-only: creates and bills nothing.
set -uo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }

# --- 0.4 the rule: are we in the isolated profile at all?
[[ "${AZURE_CONFIG_DIR:-}" == "$HOME/.azure-sidewalk" ]] \
  || fail "AZURE_CONFIG_DIR is '${AZURE_CONFIG_DIR:-unset}'. Run: source ~/sidewalk-env.sh"
[[ "${AZ_SUB:-}" != "PASTE_SUBSCRIPTION_ID_HERE" && -n "${AZ_SUB:-}" ]] \
  || fail "AZ_SUB is not filled in. See runbook 0.3."

echo "=== signed in as / active subscription"
az account show --subscription "$AZ_SUB" \
  --query "{sub:name, id:id, user:user.name, state:state}" -o table \
  || fail "cannot read subscription $AZ_SUB. Logged in as the right account?"

echo
echo "=== 1.1 which regions actually offer the GPU SKUs"
for r in eastus2 southcentralus westus3 eastus centralus; do
  echo "----- $r"
  az vm list-skus --location "$r" --resource-type virtualMachines \
    --subscription "$AZ_SUB" \
    --query "[?name=='Standard_NC24ads_A100_v4' || name=='Standard_NC4as_T4_v3' || name=='Standard_NC40ads_H100_v5'].{sku:name, restriction:restrictions[0].reasonCode}" \
    -o table
done
echo "(blank = SKU not offered there; NotAvailableForSubscription = not entitled."
echo " Either way, do not request quota in that region.)"

echo
echo "=== 1.2 current GPU vCPU quota in ${AZ_LOC} (expect zeros before approval)"
az vm list-usage --location "$AZ_LOC" --subscription "$AZ_SUB" -o table \
  | grep -iE "NCADS|NCAS|NVADS|Total Regional"
echo "(Total Regional vCPUs is a SEPARATE ceiling. Approved A100 quota will not"
echo " deploy if that number is too low. Request both.)"
