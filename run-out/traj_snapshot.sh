#!/usr/bin/env bash
# Snapshot theta at EVERY cadence write so the held-out trajectory can be
# scored densely afterwards.
#
# WHY: the 262M->655M "degradation" was read off TWO points and looked like a
# trend. Scoring the three intermediate checkpoints that already existed showed
# a decelerating excursion instead (+0.43 -> +0.15 -> +0.08 -> -0.21 per 98M),
# i.e. the opposite conclusion, for 15 minutes of GPU. Two points cannot
# distinguish a trend from an excursion, so this run banks every one.
#
# Only the .nslm (4.3 GB) is copied, not the .optim (8.6 GB): scoring needs
# theta, not the moments. 11 cadences to the 1B gate ~= 47 GB.
#
# Runs BESIDE the training driver and never writes anything the driver reads.
set -u
WT=/home/brandon/Projects/NSL/.claude/worktrees/int1b
CK=$WT/models/coder1b/checkpoints/pretrain_prod_state.nslm
DEST=$WT/models/coder1b/checkpoints_backup/traj
OUT=$WT/run-out
mkdir -p "$DEST"
seen=" "
echo "TRAJ watcher started $(date '+%F %T') -> $DEST"
while true; do
  # the driver is the liveness signal; exit with it
  ps -eo args | grep -q '[b]ash stage4_driver' || { echo "TRAJ: driver gone, exiting $(date '+%T')"; exit 0; }
  for n in $(cat "$OUT"/stage4_a*.stderr 2>/dev/null | tr '\r' '\n' \
             | grep -oE 'saved: .* at micro-batch step [0-9]+' \
             | grep -oE '[0-9]+$' | sort -un); do
    case "$seen" in *" $n "*) continue ;; esac
    # let the 12.9 GB write settle before copying, or the snapshot is torn
    prev=""; stable=0
    while [ "$stable" -lt 3 ]; do
      now=$(stat -c '%Y %s' "$CK" 2>/dev/null)
      [ "$now" = "$prev" ] && stable=$((stable+1)) || stable=0
      prev="$now"; sleep 4
    done
    cp "$CK" "$DEST/m${n}.nslm" && echo "TRAJ snapshot micro $n ($((n*4096/1000000))M tokens) $(date '+%T')"
    seen="$seen$n "
  done
  sleep 60
done
