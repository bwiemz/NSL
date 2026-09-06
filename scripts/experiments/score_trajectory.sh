#!/usr/bin/env bash
# Score held-out for every cadence snapshot, producing the DENSE trajectory
# the two-point read could not give.
#
#   score_trajectory.sh splice   # CPU only — safe to run beside training
#   score_trajectory.sh score    # needs the GPU — run when the card is free
#
# Splicing is separated on purpose: it is pure CPU/disk, so it can be done
# while the chain still owns the GPU, and it surfaces any splice failure hours
# before the scoring window rather than inside it.
set -u
MODE=${1:?usage: score_trajectory.sh splice|score}
WT=/home/brandon/Projects/NSL/.claude/worktrees/int1b
OUT=$WT/run-out
TRAJ=$WT/models/coder1b/checkpoints_backup/traj
TEMPLATE=/home/brandon/Projects/NSL/.claude/worktrees/pilot1b/models/coder1b/checkpoints/pilot_lr3e5_final.nslm
NSL=${NSL_BIN:-/home/brandon/Projects/NSL-target-fix2p16/release/nsl}   # NSL_BIN: dry-run hook
export NSL_STDLIB_PATH=$WT/stdlib

shopt -s nullglob
# The gate's own backup is a FALLBACK source for the final point.
# traj_snapshot.sh tests "is the driver alive?" BEFORE it sweeps, so if the
# driver exits between the micro-248,000 cadence write and the watcher's next
# poll, that snapshot is never taken. The driver independently copies the same
# state to checkpoints_backup/lr15_1bgate_<arm>/, so recover it from there
# rather than losing the endpoint of the curve. (Not fixed in the watcher
# itself: it is running right now, and bash reads a script incrementally --
# editing one mid-execution corrupts it.)
GATE_BK=$WT/models/coder1b/checkpoints_backup
for g in "$GATE_BK"/lr15_1bgate_*/pretrain_prod_state.nslm; do
  [ -f "$g" ] || continue
  if [ ! -f "$TRAJ/m248000.nslm" ]; then
    echo "  recovering the gate point from $(dirname "$g" | xargs basename)"
    cp "$g" "$TRAJ/m248000.nslm"
  fi
done
snaps=("$TRAJ"/m*.nslm)
[ ${#snaps[@]} -gt 0 ] || { echo "no snapshots in $TRAJ"; exit 1; }

if [ "$MODE" = splice ]; then
  for f in "${snaps[@]}"; do
    n=$(basename "$f" .nslm)                      # m<micro>
    sp="$OUT/spliced_traj_${n}.nslm"
    [ -f "$sp" ] && { echo "  $n already spliced"; continue; }
    nice -n 19 python3 "$WT/tools/nslm_splice.py" "$f" "$TEMPLATE" "$sp" \
      && echo "  spliced $n" || echo "  SPLICE FAILED $n"
  done
  exit 0
fi

# ── score ────────────────────────────────────────────────────────────────
# TF32 for every point, matching the entire trajectory record. bf16 vs TF32
# was measured to agree to 5 dp on identical weights, so this does not bias
# any point -- it keeps the comparison like-for-like on principle.
cd "$WT/models/coder1b" || exit 9
printf '%10s %8s %12s %12s\n' micro Mtok VAL_STACK VAL_WEB
for f in "${snaps[@]}"; do
  n=$(basename "$f" .nslm); micro=${n#m}
  sp="$OUT/spliced_traj_${n}.nslm"
  [ -f "$sp" ] || { echo "  $n NOT SPLICED — run 'splice' first"; continue; }
  sed -i "s|model_load(m, \"../../run-out/[^\"]*\")|model_load(m, \"../../run-out/spliced_traj_${n}.nslm\")|" val_from_splice.nsl
  ( unset NSL_MATMUL_BF16 NSL_MATMUL_BF16_ROUND
    bash "${GUARD_BIN:-$WT/scripts/gpu-guard.sh}" run -- "$NSL" run --source-ad val_from_splice.nsl \
      > "$OUT/traj_${n}_val.stdout" 2> "$OUT/traj_${n}_val.stderr" )
  s=$(grep -A1 VAL_LOSS_STACK "$OUT/traj_${n}_val.stdout" 2>/dev/null | tail -1)
  w=$(grep -A1 VAL_LOSS_WEB   "$OUT/traj_${n}_val.stdout" 2>/dev/null | tail -1)
  mtok=$(( micro * 4096 / 1000000 ))          # `bc` is NOT installed here
  # A missing measurement must never render as a plausible number: printf
  # "%.4f" on an empty string yields 0.0000, which reads as a real loss of
  # zero in a results table. Say FAILED and point at the log instead.
  case "$s$w" in
    *[0-9]*) printf '%10s %8s %12.4f %12.4f\n' "$micro" "$mtok" "$s" "$w" ;;
    *)       printf '%10s %8s %12s %12s   <- see traj_%s_val.stderr\n' \
                    "$micro" "$mtok" FAILED FAILED "$n" ;;
  esac
done
