#!/usr/bin/env bash
# Run the three schedule arms SEQUENTIALLY. One GPU, one arm at a time.
set -uo pipefail
C="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="$C/campaign.log"
say() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }
say "SCHED3 CAMPAIGN START — arms ${ARMS:-A B C} from the 262M branch, 32,000 micro each"
for ARM in ${ARMS:-A B C}; do
  say "=== ARM $ARM starting ==="
  bash "$C/snap_lineage.sh" "$ARM" &
  SNAPPID=$!
  bash "$C/arm_driver.sh" "$ARM"; rc=$?
  wait "$SNAPPID" 2>/dev/null
  if [ $rc -ne 0 ]; then say "ARM $ARM FAILED rc=$rc — campaign stops here"; exit $rc; fi
  say "=== ARM $ARM done ==="
done
say "SCHED3_CAMPAIGN_COMPLETE"
