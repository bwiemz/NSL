#!/usr/bin/env bash
# Relaunch the 1B chain once the GPU is genuinely free, and not before.
#
# 2026-08-31 09:45: the chain died of a real GPU OOM (73.9 MB free of 31.39 GB)
# because a CONCURRENT session's `castcache_ab.sh` runs ./prog_sel directly
# rather than through scripts/gpu-guard.sh. gpu-guard excludes only COOPERATING
# runs, so it protected nothing here -- and then correctly refused our own
# restart while their unguarded job still held 14.4 GB.
#
# Our chain needs ~27 GB of a 31.4 GB card. It cannot share with a 14 GB
# benchmark. So: wait for a SUSTAINED free period rather than racing into the
# next lull and losing another cadence to the same OOM.
set -u
OUT=/home/brandon/Projects/NSL/.claude/worktrees/int1b/run-out
NEED_FREE_MIB=28000        # our peak is ~27 GB; leave headroom
QUIET_POLLS=10             # 10 x 60 s of sustained quiet before committing
MAX_WAIT_POLLS=720         # give up after 12 h
quiet=0
echo "AWAIT: watching for >=${NEED_FREE_MIB} MiB free, sustained ${QUIET_POLLS} min  $(date '+%F %T')"
for i in $(seq 1 $MAX_WAIT_POLLS); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  free=$(( ${total:-0} - ${used:-0} ))
  if [ "$free" -ge "$NEED_FREE_MIB" ]; then
    quiet=$((quiet+1))
    [ $((quiet % 5)) -eq 0 ] && echo "AWAIT: quiet ${quiet}/${QUIET_POLLS} (free ${free} MiB) $(date '+%T')"
  else
    [ "$quiet" -gt 0 ] && echo "AWAIT: reset — free dropped to ${free} MiB $(date '+%T')"
    quiet=0
  fi
  if [ "$quiet" -ge "$QUIET_POLLS" ]; then
    echo "AWAIT: GPU quiet, relaunching chain $(date '+%F %T')"
    cd "$OUT" || exit 9
    nohup bash stage4_driver.sh bf16 > stage4_driver.log 2>&1 &
    sleep 10
    nohup bash traj_snapshot.sh > traj_snapshot.log 2>&1 &
    echo "AWAIT: launched"
    exit 0
  fi
  sleep 60
done
echo "AWAIT: GPU never freed for ${QUIET_POLLS} min within 12 h — not launching"
exit 1
