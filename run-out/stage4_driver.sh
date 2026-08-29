#!/usr/bin/env bash
# Chain leg from 263M tokens to the 1B hard gate, on the POST-#541 compiler.
#
# ── why this driver exists in this shape ──────────────────────────────────
#
# 1. THE 2^16 CEILING. Every previous leg died at micro-step 65,536 — the
#    untyped `on_step` step counter being swept into
#    `nsl_tensor_free_if_valid` and dereferenced as a pointer (PR #541). The
#    fix is a CODEGEN change, so it travels with the CLI, not the runtime
#    archive; this driver therefore pins the CLI and PROVES the fix is in the
#    binary it is about to spend thirty hours with, by running the 2^16
#    fixture before it builds anything. A path check is not a proof — the
#    two candidate CLIs differ by 72 bytes and link byte-identical archives.
#
# 2. THE PRECISION MISLABEL. Stages 2 and 3 were reported as f32 and ran
#    bf16: `NSL_MATMUL_BF16=1` was inherited from a copied driver and nothing
#    checked. So the arm is an ARGUMENT here, the env is set from scratch
#    (never inherited), and the runtime's own banner must AGREE before the
#    run is allowed to continue. bf16+RNE and bf16+SR print the SAME math-mode
#    banner, so SR gets its own separate witness — and the two non-SR arms
#    must show its ABSENCE, or a stray NSL_MATMUL_BF16_ROUND would go unseen
#    exactly the way NSL_MATMUL_BF16 did.
#
# 3. CRASH AND STALL. A 30 h run that dies at hour 28 and stays dead is the
#    expensive failure mode. A crash costs at most one cadence (2,000
#    optimizer updates) and the leg resumes. The program is built to a
#    PERSISTENT path so a repeat crash can be symbolized against the exact
#    binary that produced it — `nsl run`'s program lives in /tmp and was gone
#    before the last one could be. A live-but-not-progressing process is
#    caught too: "still running" is not the same as "still training".
set -u
ARM=${1:?usage: stage4_driver.sh f32|bf16|bf16sr}
WT=/home/brandon/Projects/NSL/.claude/worktrees/int1b
cd "$WT/models/coder1b" || exit 9
export NSL_STDLIB_PATH=$WT/stdlib

# Set the precision from scratch. Never inherit: that is what mislabelled
# stages 2 and 3. WANT/SRWANT are the two witnesses the runtime must show.
unset NSL_MATMUL_BF16 NSL_MATMUL_BF16_ROUND
case "$ARM" in
  f32)    WANT="TF32 tensor cores"; SRWANT=absent ;;
  bf16)   export NSL_MATMUL_BF16=1; WANT="BF16 tensor"; SRWANT=absent ;;
  bf16sr) export NSL_MATMUL_BF16=1 NSL_MATMUL_BF16_ROUND=sr
          WANT="BF16 tensor"; SRWANT=present ;;
  *) echo "STAGE4 ABORT: unknown arm $ARM"; exit 2 ;;
esac
SRLINE="bf16 operand cast: STOCHASTIC rounding"

GUARD=$WT/scripts/gpu-guard.sh
NSL=/home/brandon/Projects/NSL-target-fix2p16/release/nsl   # POST-#541
OUT=$WT/run-out
PROG=$OUT/stage4_prog            # persistent, symbolizable
# 1B hard gate: micro 248,000 = 62,000 optimizer updates = 1,015,808,000
# tokens. A multiple of checkpoint_every=2000 updates, so the gate lands ON a
# cadence write rather than 2,000 updates past one; and a multiple of 4 micro
# = 16,384 tokens, so it lands on a complete accumulation window.
STOP=248200; CKPT_MICRO=248000; MAX_RESTARTS=6
STALL_POLLS=15                   # 15 x 60 s of no progress = stalled
RESUME_FROM=64000                # the state on disk when this leg starts

echo "STAGE4 arm=$ARM  nsl=$(md5sum "$NSL" | cut -c1-12)  $(date '+%F %T')"
echo "STAGE4 env: NSL_MATMUL_BF16=${NSL_MATMUL_BF16:-unset} NSL_MATMUL_BF16_ROUND=${NSL_MATMUL_BF16_ROUND:-unset}"

# ── pre-flight: prove THIS binary carries the 2^16 fix ────────────────────
# 81,930 micro-steps of a toy model, ~5 s. Without the fix it SIGSEGVs at
# 65,536. This is the whole reason the leg is being restarted, so it is
# checked functionally rather than by trusting a path.
if ! ( cd "$OUT/repro" && timeout 300 "$NSL" run --source-ad --seed 7 min4.nsl \
        > "$OUT/stage4_preflight.out" 2> "$OUT/stage4_preflight.err" ) \
   || ! grep -q REPRO_PASSED_2P16 "$OUT/stage4_preflight.out"; then
  echo "STAGE4 ABORT: $NSL did not survive micro-step 65,536 — wrong binary."
  tail -3 "$OUT/stage4_preflight.out" 2>/dev/null; exit 7
fi
echo "STAGE4 preflight OK: binary crosses 2^16 ($(grep -c '^[0-9]' "$OUT/stage4_preflight.out") prints)"

"$NSL" build --source-ad --checkpoint-blocks --checkpoint-selective \
  --fuse-rmsnorm-backward --fuse-wgrad-accum pretrain_prod.nsl -o "$PROG" \
  > "$OUT/stage4_build.log" 2>&1 || { echo "STAGE4 ABORT: build failed"; tail -5 "$OUT/stage4_build.log"; exit 3; }
echo "STAGE4[$ARM] built $PROG"

# Kill the process GROUP, then verify the GPU actually came back. The run is
# launched under `setsid` so gpu-guard and the training program share one
# group id; TERMing only the guard leaves the trainer orphaned and HOLDING
# VRAM, which makes the next attempt's guard refuse and ends the leg. After
# the group signal, anything still resident whose /proc/PID/exe is our own
# program gets a hard kill -- identified by executable, never by "every
# compute PID on the device".
reap() {   # $1 = the launched pid; its GROUP is what gets signalled
  local pid=${1:-} pg=""
  # Ask the kernel for the real group rather than assuming the pid leads it:
  # `setsid` execs without forking only when it is not already a group
  # leader, and that depends on job control being off. Reading pgid is true
  # either way.
  [ -n "$pid" ] && pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  local mypg; mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
  # Refuse to signal our OWN group -- that would kill this driver mid-leg.
  if [ -n "$pg" ] && [ "$pg" != "$mypg" ] && [ "$pg" -gt 1 ] 2>/dev/null; then
    kill -TERM -"$pg" 2>/dev/null; sleep 8; kill -9 -"$pg" 2>/dev/null
  else
    [ -n "$pid" ] && { kill -TERM "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; }
  fi
  sleep 4
  local p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    [ "$(readlink -f /proc/$p/exe 2>/dev/null)" = "$(readlink -f "$PROG")" ] && kill -9 "$p" 2>/dev/null
  done
  sleep 6
}

for attempt in $(seq 0 $MAX_RESTARTS); do
  LOGO=$OUT/stage4_a${attempt}.stdout; LOGE=$OUT/stage4_a${attempt}.stderr
  setsid bash "$GUARD" run -- "$PROG" > "$LOGO" 2> "$LOGE" &
  GPID=$!
  PG=$GPID          # reap resolves this pid to its real group
  T0=$(date +%s); FIRSTMICRO=""
  echo "STAGE4[$ARM] attempt $attempt started $(date '+%H:%M:%S')"
  engaged=""; laststep=""; stall=0; lastreport=$T0
  while true; do
    sleep 60
    if grep -q "REFUSING" "$LOGE" 2>/dev/null; then
      echo "STAGE4 REFUSED (gpu-guard)"; reap "$PG"; exit 4; fi
    alive=yes; kill -0 "$GPID" 2>/dev/null || alive=""
    cur=$(tr '\r' '\n' < "$LOGO" 2>/dev/null | grep -E "^[0-9]+$" | tail -1)
    if [ -z "$alive" ]; then echo "STAGE4[$ARM] attempt $attempt ENDED $(date '+%H:%M:%S') at micro ${cur:-?}"; break; fi
    [ -z "${cur:-}" ] && continue

    # ── one-time arm verification, on the run's own output ───────────────
    if [ -z "$engaged" ]; then
      E=$(tr '\r' '\n' < "$LOGE")
      case "$E" in *"cuBLAS math mode: $WANT"*) ;; *)
        echo "STAGE4 ABORT: math-mode banner is not '$WANT' — arm/env mismatch: $(printf '%s' "$E" | grep -m1 -oE 'cuBLAS math mode: [A-Za-z0-9 ()]*')"
        reap "$PG"; exit 5 ;; esac
      seen=absent; case "$E" in *"$SRLINE"*) seen=present ;; esac
      if [ "$seen" != "$SRWANT" ]; then
        echo "STAGE4 ABORT: stochastic-rounding witness $seen, wanted $SRWANT — the two bf16 arms share a math-mode banner and this is the only line that separates them"
        reap "$PG"; exit 8; fi
      first=$(tr '\r' '\n' < "$LOGO" | grep -m1 -E "^[0-9]+$")
      if [ "$first" -le "$RESUME_FROM" ]; then
        echo "STAGE4 ABORT: from-scratch or backwards (first print $first, expected > $RESUME_FROM)"
        reap "$PG"; exit 6; fi
      engaged=yes; FIRSTMICRO=$first
      echo "STAGE4[$ARM] RESUME ENGAGED at $first  (math=$WANT, sr=$seen)"
    fi

    # ── stall detector: alive is not the same as progressing ─────────────
    if [ "$cur" = "$laststep" ]; then
      stall=$((stall + 1))
      if [ "$stall" -ge "$STALL_POLLS" ]; then
        echo "STAGE4[$ARM] STALLED at micro $cur for $STALL_POLLS min — killing to force a resume"
        reap "$PG"; break; fi
    else stall=0; laststep=$cur; fi

    # ── periodic rate line (the run prints no throughput of its own) ─────
    now=$(date +%s)
    if [ $((now - lastreport)) -ge 1800 ]; then
      el=$((now - T0)); done_=$((cur - FIRSTMICRO))
      if [ "$el" -gt 0 ] && [ "$done_" -gt 0 ]; then
        rate=$(( done_ * 4096 / el )); left=$(( (STOP - cur) * 4096 ))
        echo "STAGE4[$ARM] micro $cur  ${rate} tok/s  ETA $(( left / (rate>0?rate:1) / 3600 ))h  $(date '+%H:%M:%S')"
      fi
      lastreport=$now
    fi

    if [ "$cur" -ge "$STOP" ]; then
      grep -q "micro-batch step $CKPT_MICRO" "$LOGE" \
        && echo "STAGE4 TARGET $cur with checkpoint@$CKPT_MICRO written" \
        || echo "STAGE4 WARNING: reached $cur but no checkpoint@$CKPT_MICRO in the log"
      reap "$PG"
      mkdir -p "checkpoints_backup/lr15_1bgate_${ARM}"
      cp checkpoints/pretrain_prod_state.nslm checkpoints/pretrain_prod_state.nslm.optim \
         "checkpoints_backup/lr15_1bgate_${ARM}/" && echo "STAGE4 BACKUP OK"
      echo "STAGE4_REACHED_1B_GATE at $cur $(date '+%F %T')"; exit 0
    fi
  done
  reap "$PG"
  last=$(tr '\r' '\n' < "$LOGO" | grep -E "^[0-9]+$" | tail -1)
  echo "STAGE4[$ARM] attempt $attempt died at micro ${last:-?}; resuming from the last cadence checkpoint"
  # After a crash the on-disk state is whatever the last cadence wrote, which
  # is >= RESUME_FROM; keep the guard meaningful but do not demand progress
  # past a point the checkpoint may not have reached.
  sleep 20
done
echo "STAGE4 GAVE UP after $MAX_RESTARTS restarts"; exit 1
