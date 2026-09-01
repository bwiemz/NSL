#!/usr/bin/env bash
# Splice each cadence snapshot as it lands, so the gate is followed by scoring
# alone. Idempotent (score_trajectory.sh skips what exists), CPU-only, and it
# exits with the training driver.
set -u
cd /home/brandon/Projects/NSL/.claude/worktrees/int1b/run-out || exit 9
while ps -eo args | grep -q '[b]ash stage4_driver'; do
  bash score_trajectory.sh splice 2>&1 | grep -E 'spliced [a-z0-9]+$|FAILED'
  sleep 1200
done
bash score_trajectory.sh splice 2>&1 | grep -E 'spliced [a-z0-9]+$|FAILED'   # final sweep
echo "AUTOSPLICE: driver gone, final sweep done $(date '+%F %T')"
