#!/usr/bin/env bash
# Schedule campaign: run ONE matched continuation arm from the 262M branch.
#
#   arm_driver.sh {A|B|C}
#
# Design constraints this driver enforces, each one earned by a specific
# failure in the 1B chain campaign:
#
#  * ARM AS AN ARGUMENT, env built from scratch. A copied driver carrying a
#    stale NSL_MATMUL_BF16 ran two legs that were written up as f32.
#  * BANNER ASSERTED BOTH WAYS. bf16+RNE and bf16+SR share a cuBLAS banner;
#    only the SR line separates them, so a non-SR arm asserts its ABSENCE.
#    Same for the cublasLt line, which is deliberately OFF for this campaign:
#    an arithmetic change has no place inside a convergence manipulation.
#  * PERSISTENT BINARY via `nsl build -o`. `nsl run` compiles into a temp dir
#    that is gone by the time you look at a core file.
#  * AUTO-RESUME + STALL DETECTION. A 5.6 h arm that dies at hour 5 and stays
#    dead is the expensive failure. Alive != progressing.
#  * REAP BY PROCESS GROUP resolved from ps, then by /proc/PID/exe match --
#    never "every compute PID on the device", which would kill a peer's job.
#  * INTERRUPTIONS ARE COUNTED, not forgotten. The 1B gate left a 0.272-nat
#    chain-vs-arm gap whose only known cause was a 600-microstep replay after
#    an OOM. Every restart records how many microsteps got replayed, into the
#    lineage record, so that question can be asked of THIS data.
set -uo pipefail

# NOTE: no braces in the :? message -- bash closes ${...} at the FIRST },
# so "{A|B|C}}" appends a literal } to the value and the arm becomes "A}".
ARM="${1:?usage: arm_driver.sh A|B|C}"
case "$ARM" in A|B|C) ;; *) echo "bad arm '$ARM'" >&2; exit 2 ;; esac

WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CAMP="$WT/campaign-sched3"
OUT="$CAMP/out_$ARM"
CKDIR="$CAMP/ckpt_$ARM"
SNAP="$CAMP/snap_$ARM"
BRANCH_CK="/home/brandon/Projects/NSL/.claude/worktrees/int1b/models/coder1b/checkpoints_backup/lr15_stage2_opt16000"
NSL="${NSL_BIN:-/home/brandon/Projects/NSL-target-sched3/release/nsl}"
export NSL_STDLIB_PATH="$WT/stdlib"
# The recipe resolves `from model import NSLCoder` and every corpus path
# RELATIVE TO models/coder1b. Build and run from there, nowhere else.
RECIPE_DIR="$WT/models/coder1b"
GUARD="$WT/scripts/gpu-guard.sh"

BRANCH_MICRO=64000
END_MICRO=96000
SCORE_AT="72000 80000 88000 96000"
MAX_RESTARTS=6
STALL_POLLS=15          # x 60 s of no forward progress => stalled
POLL=60

mkdir -p "$OUT" "$CKDIR" "$SNAP"
LOG="$OUT/driver.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---- 1. arithmetic environment, built from scratch -----------------------
unset NSL_MATMUL_BF16 NSL_MATMUL_BF16_ROUND NSL_MATMUL_BF16_CAST_CACHE \
      NSL_MATMUL_BF16_LT NSL_MATMUL_BF16_LT_WORKSPACE_MIB \
      NSL_RESUME_ALLOW_TRAJECTORY_DRIFT
export NSL_MATMUL_BF16=1                 # the production arithmetic (acquitted)
# RNE (default), cast cache OFF, Lt OFF -- identical across all three arms.
WANT_BANNER="BF16 tensor"
MUSTNOT=( "STOCHASTIC rounding" "cublasLt" )

# Arms B and C legitimately change lr/sp2/sp3 relative to the branch
# checkpoint's record. The control must NOT need this: if arm A ever requires
# it, arm A is not a control and this run is void.
if [ "$ARM" != "A" ]; then export NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1; fi

say "ARM $ARM  nsl=$($NSL --version 2>/dev/null | tr -d '\n')  $(date '+%F %T')"
say "ARM $ARM env: BF16=${NSL_MATMUL_BF16} ROUND=${NSL_MATMUL_BF16_ROUND:-unset} \
CACHE=${NSL_MATMUL_BF16_CAST_CACHE:-unset} LT=${NSL_MATMUL_BF16_LT:-unset} \
DRIFT=${NSL_RESUME_ALLOW_TRAJECTORY_DRIFT:-unset}"

# ---- 2. seed the arm's own copy of the branch checkpoint -----------------
if [ ! -f "$CKDIR/state.nslm" ]; then
  say "ARM $ARM seeding from the 262M branch (12.9 GB copy)"
  cp "$BRANCH_CK/pretrain_prod_state.nslm"       "$CKDIR/state.nslm"
  cp "$BRANCH_CK/pretrain_prod_state.nslm.optim" "$CKDIR/state.nslm.optim"
  # The branch backup is the ONLY copy of the best theta. Prove we copied it
  # rather than moved or truncated it.
  for f in pretrain_prod_state.nslm pretrain_prod_state.nslm.optim; do
    [ -s "$BRANCH_CK/$f" ] || { say "ARM $ARM ABORT: branch source $f missing after copy"; exit 1; }
  done
  a=$(stat -c %s "$BRANCH_CK/pretrain_prod_state.nslm")
  b=$(stat -c %s "$CKDIR/state.nslm")
  [ "$a" = "$b" ] || { say "ARM $ARM ABORT: copy size $b != source $a"; exit 1; }
fi

# ---- 3. build a persistent, symbolizable binary --------------------------
PROG="$OUT/arm_${ARM}_prog"
if [ ! -x "$PROG" ]; then
  say "ARM $ARM building $PROG"
  ( cd "$RECIPE_DIR" && "$NSL" build --source-ad --checkpoint-blocks --checkpoint-selective \
      --fuse-rmsnorm-backward --fuse-wgrad-accum -o "$PROG" "sched3_arm_$ARM.nsl" ) \
      > "$OUT/build.log" 2>&1
  [ -x "$PROG" ] || { say "ARM $ARM ABORT: build failed, see $OUT/build.log"; exit 1; }
fi

# ---- 4. run, with resume, stall detection and interruption accounting ----
last_micro() { grep -oE '^[0-9]+$' "$OUT/run.log" 2>/dev/null | tail -1; }

reap() {  # $1 = launched pid; signal its GROUP, then any straggler on our exe
  local pid=${1:-} pg="" mypg
  [ -n "$pid" ] && pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
  if [ -n "$pg" ] && [ "$pg" != "$mypg" ] && [ "$pg" -gt 1 ] 2>/dev/null; then
    kill -TERM -"$pg" 2>/dev/null; sleep 8; kill -9 -"$pg" 2>/dev/null
  elif [ -n "$pid" ]; then
    kill -TERM "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null
  fi
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(readlink -f "/proc/$p/exe" 2>/dev/null)" = "$(readlink -f "$PROG")" ] \
      && kill -9 "$p" 2>/dev/null
  done
}

INTERRUPTIONS=0
REPLAYED=0
for attempt in $(seq 0 $MAX_RESTARTS); do
  resume_from=$(last_micro)
  if [ -n "$resume_from" ] && [ "$attempt" -gt 0 ]; then
    # The checkpoint we resume from is the last CADENCE write, not the last
    # printed step. Everything between is replayed -- record how much.
    ck_micro=$(python3 - "$CKDIR/state.nslm.optim" <<'PY'
import sys,json,struct
f=open(sys.argv[1],'rb'); f.seek(8); h=struct.unpack('<Q',f.read(8))[0]
print(json.loads(f.read(h).decode().rstrip('\x00'))['step_count'])
PY
)
    d=$(( resume_from - ck_micro )); [ "$d" -gt 0 ] && REPLAYED=$(( REPLAYED + d ))
    INTERRUPTIONS=$(( INTERRUPTIONS + 1 ))
    say "ARM $ARM attempt $attempt: resuming from checkpoint@$ck_micro (last print $resume_from, replaying $d)"
  fi

  ( cd "$RECIPE_DIR" && setsid bash "$GUARD" run -- "$PROG" ) >> "$OUT/run.log" 2>&1 &
  RUNPID=$!
  say "ARM $ARM attempt $attempt started pid $RUNPID"

  # --- assert the arm is what it says it is, once the banner has printed ---
  for _ in $(seq 1 30); do sleep 2; grep -q "math mode" "$OUT/run.log" 2>/dev/null && break; done
  if ! grep -q "$WANT_BANNER" "$OUT/run.log" 2>/dev/null; then
    say "ARM $ARM ABORT: banner does not contain '$WANT_BANNER'"; reap "$RUNPID"; exit 1
  fi
  for bad in "${MUSTNOT[@]}"; do
    if grep -q "$bad" "$OUT/run.log" 2>/dev/null; then
      say "ARM $ARM ABORT: banner contains '$bad', which this arm must NOT have"
      reap "$RUNPID"; exit 1
    fi
  done
  if [ "$ARM" = "A" ] && grep -qi "TRAJECTORY_DRIFT" "$OUT/run.log" 2>/dev/null; then
    say "ARM A ABORT: the control required a trajectory-drift acknowledgment; it is not a control"
    reap "$RUNPID"; exit 1
  fi
  say "ARM $ARM identity asserted: '$WANT_BANNER' present, ${MUSTNOT[*]} absent"

  stalls=0; prev=""
  while kill -0 "$RUNPID" 2>/dev/null; do
    sleep $POLL
    cur=$(last_micro)
    if [ -n "$cur" ]; then
      [ "$cur" = "$prev" ] && stalls=$(( stalls + 1 )) || stalls=0
      prev=$cur
      if [ "$cur" -ge "$END_MICRO" ] 2>/dev/null; then
        say "ARM $ARM reached $cur >= $END_MICRO; waiting for the cadence write to settle"
        sleep 90; reap "$RUNPID"; break
      fi
    fi
    if [ "$stalls" -ge "$STALL_POLLS" ]; then
      say "ARM $ARM STALL: no progress past $prev for $((STALL_POLLS*POLL))s; reaping"
      reap "$RUNPID"; break
    fi
  done
  wait "$RUNPID" 2>/dev/null

  cur=$(last_micro)
  if [ -n "$cur" ] && [ "$cur" -ge "$END_MICRO" ] 2>/dev/null; then
    say "ARM $ARM TARGET $cur reached"
    echo "$INTERRUPTIONS $REPLAYED" > "$OUT/interruptions.txt"
    say "ARM_${ARM}_COMPLETE interruptions=$INTERRUPTIONS replayed_microsteps=$REPLAYED $(date '+%F %T')"
    exit 0
  fi
  say "ARM $ARM attempt $attempt ended at ${cur:-<none>}; restarting"
done

say "ARM $ARM GAVE UP after $MAX_RESTARTS restarts"
exit 1
