#!/usr/bin/env bash
# Three-arm matched pair: f32 / bf16+RNE / bf16+SR   (PR #540's validation run)
#
# WHY: #540's gates are all operand-level. Nothing yet connects "0.034 ULP
# time-averaged bias" to "recovers the loss bf16 was thought to cost". This is
# the run the PR itself names: ONE checkpoint, IDENTICAL batches (the
# checkpoint restores the loader slot and the RNG sidecar), three arms
# differing ONLY in environment.
#
# READ: if the SR arm's paired delta vs f32 stays flat where the RNE arm's
# grows, the standing-weight-bias mechanism is the whole story and bf16+SR is
# usable for the chain. If SR also grows, the bias is real but not dominant.
# If NEITHER grows, bf16 is simply not costing anything at this learning rate
# -- which is what the (bf16, lr=1.5e-5) leg to 263M tokens already suggests,
# and would retire the bf16 conviction outright.
#
# ── two hard lessons are wired in as assertions ──────────────────────────
#
# 1. THE MISLABEL. The bf16 conviction came from legs that were REPORTED as
#    f32 and RAN bf16, because a copied driver carried NSL_MATMUL_BF16=1 and
#    nothing checked. Every arm here sets its env from scratch and then must
#    prove it on the runtime's own output. bf16+RNE and bf16+SR print the
#    SAME math-mode banner, so SR needs its own witness -- and the arms that
#    must NOT have SR assert its ABSENCE, which is the half that would have
#    caught the original mislabel.
#
# 2. THE 2^16 CEILING. The previous attempt at this exact probe measured
#    NOTHING: all three arms died at micro-step 65,536 (PR #541). The fix is
#    a codegen change carried by the CLI, so the CLI is pinned and PROVEN
#    functionally before any arm starts.
#
# WRITE-FREE: checkpoint_load only, never checkpoint_save. No arm can touch
# the chain's checkpoint. Arms run sequentially behind gpu-guard.
set -u
WT=/home/brandon/Projects/NSL/.claude/worktrees/int1b
cd "$WT/models/coder1b" || exit 9
export NSL_STDLIB_PATH=$WT/stdlib
GUARD=$WT/scripts/gpu-guard.sh
NSL=/home/brandon/Projects/NSL-target-fix2p16/release/nsl   # post-#540 AND post-#541
OUT=$WT/run-out
BK=$WT/models/coder1b/checkpoints_backup
SRLINE="bf16 operand cast: STOCHASTIC rounding"

# ── phase 0: wait for the chain's 1B-gate leg (up to 60 h) ───────────────
#
# Wait on the leg's own TERMINAL MARKER first and only fall back to process
# liveness. Two reasons the naive `pgrep -f stage4_driver` is wrong here:
# it matches anything merely MENTIONING the driver -- a `tail -F
# stage4_driver.log` monitor, another shell's `cat` -- which would park this
# probe for the full 60 h behind a log reader; and the driver's own exit is
# better evidenced by what it wrote than by whether it is still breathing.
# The pattern below is anchored on the script name so a reader of the .log
# cannot satisfy it.
stage4_running() { pgrep -f 'stage4_driver\.sh' >/dev/null 2>&1; }
stage4_done()    { grep -qE 'STAGE4_REACHED_1B_GATE|STAGE4 GAVE UP|STAGE4 ABORT|STAGE4 REFUSED' "$OUT/stage4_driver.log" 2>/dev/null; }
for i in $(seq 1 720); do
  stage4_done && break
  stage4_running || { sleep 30; stage4_running || break; }   # settle: log may lag exit
  sleep 300
done
if stage4_running && ! stage4_done; then echo "SR3ARM: stage4 still running after 60h — not starting"; exit 3; fi
echo "SR3ARM: stage4 finished — $(tail -1 "$OUT/stage4_driver.log" 2>/dev/null)"
[ -x "$NSL" ] || { echo "SR3ARM ABORT: $NSL missing — all three arms must share one SR-capable, 2^16-fixed binary"; exit 5; }

# ── pre-flight: the binary must cross micro-step 65,536 ──────────────────
if ! ( cd "$OUT/repro" && timeout 300 "$NSL" run --source-ad --seed 7 min4.nsl \
        > "$OUT/sr3arm_preflight.out" 2> "$OUT/sr3arm_preflight.err" ) \
   || ! grep -q REPRO_PASSED_2P16 "$OUT/sr3arm_preflight.out"; then
  echo "SR3ARM ABORT: $NSL did not survive micro-step 65,536 — this is the bug that made the last attempt measure nothing"; exit 7
fi
echo "SR3ARM preflight OK: binary crosses 2^16"

# Prefer the 1B-gate state; fall back to 263M if the chain did not reach it.
if [ -f "$BK/lr15_1bgate_bf16/pretrain_prod_state.nslm" ]; then
  CKPT="checkpoints_backup/lr15_1bgate_bf16/pretrain_prod_state.nslm"; BASE=248000
  echo "SR3ARM: base = the 1B gate (micro 248,000, 1.016B tokens)"
else
  CKPT="checkpoints_backup/lr15_stage2_opt16000/pretrain_prod_state.nslm"; BASE=64000
  echo "SR3ARM: 1B-gate checkpoint absent — falling back to micro 64,000 (263M tokens)"
fi
# NOTE FOR THE WRITE-UP: this base theta was trained under BF16 at lr=1.5e-5.
# The probe measures the FORWARD-GOING cost of each arithmetic from a shared
# state; it does not and cannot say what a from-scratch f32 run would have
# done. Say so when reporting it.
STOP=$((BASE + 24000))   # 98.3M tokens/arm — the length that convicted bf16
echo "SR3ARM: base=$BASE stop=$STOP ckpt=$CKPT"

# The probe recipe: production recipe, resume-only, no save.
sed -e "s|checkpoint_load=\"checkpoints/pretrain_prod_state.nslm\", checkpoint_save=\"checkpoints/pretrain_prod_state.nslm\", checkpoint_every=2000|checkpoint_load=\"$CKPT\"|" \
    pretrain_prod.nsl > sr3arm_probe.nsl
grep -q "checkpoint_save" sr3arm_probe.nsl && { echo "SR3ARM ABORT: probe still has checkpoint_save — refusing to risk the chain checkpoint"; exit 4; }
grep -q "checkpoint_load=\"$CKPT\"" sr3arm_probe.nsl || { echo "SR3ARM ABORT: probe has no checkpoint_load"; exit 4; }

reap() {
  local pid=${1:-} pg="" mypg
  [ -n "$pid" ] && pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  mypg=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')
  if [ -n "$pg" ] && [ "$pg" != "$mypg" ] && [ "$pg" -gt 1 ] 2>/dev/null; then
    kill -TERM -"$pg" 2>/dev/null; sleep 8; kill -9 -"$pg" 2>/dev/null
  else [ -n "$pid" ] && { kill -TERM "$pid" 2>/dev/null; sleep 8; kill -9 "$pid" 2>/dev/null; }; fi
  sleep 4
  local p
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    case "$(readlink -f /proc/$p/exe 2>/dev/null)" in */nsl_run_*/*|*sr3arm*) kill -9 "$p" 2>/dev/null ;; esac
  done
  sleep 8
}

# $1=name  $2=math-mode banner fragment  $3=present|absent for the SR line  $4..=env
run_arm() {
  local name=$1 want=$2 srwant=$3; shift 3
  echo "SR3ARM [$name] starting $(date '+%H:%M:%S')  env: ${*:-none}"
  ( # Clear FIRST, then set only this arm's variables: a leaked
    # NSL_MATMUL_BF16 from the caller is exactly how the mislabel happened.
    unset NSL_MATMUL_BF16 NSL_MATMUL_BF16_ROUND
    for kv in "$@"; do export "$kv"; done
    bash "$GUARD" run -- "$NSL" run --source-ad --checkpoint-blocks --checkpoint-selective \
      --fuse-rmsnorm-backward --fuse-wgrad-accum sr3arm_probe.nsl \
      > "$OUT/sr3arm_$name.stdout" 2> "$OUT/sr3arm_$name.stderr" ) &
  local GPID=$! engaged="" laststep="" stall=0
  while true; do
    sleep 30
    if grep -q "panicked\|REFUSING" "$OUT/sr3arm_$name.stderr" 2>/dev/null; then
      echo "SR3ARM [$name] CRASHED: $(tr '\r' '\n' < "$OUT/sr3arm_$name.stderr" | grep -m1 'panicked\|REFUSING')"; reap "$GPID"; return 1; fi
    if ! kill -0 "$GPID" 2>/dev/null; then echo "SR3ARM [$name] EXITED early at micro $(tr '\r' '\n' < "$OUT/sr3arm_$name.stdout" | grep -E '^[0-9]+$' | tail -1)"; return 1; fi
    local cur; cur=$(tr '\r' '\n' < "$OUT/sr3arm_$name.stdout" 2>/dev/null | grep -E "^[0-9]+$" | tail -1)
    [ -z "${cur:-}" ] && continue
    if [ -z "$engaged" ]; then
      local E; E=$(tr '\r' '\n' < "$OUT/sr3arm_$name.stderr")
      case "$E" in *"cuBLAS math mode: $want"*) ;; *)
        echo "SR3ARM [$name] ABORT: math-mode banner is not '$want' — this arm is mislabelled: $(printf '%s' "$E" | grep -m1 -oE 'cuBLAS math mode: [A-Za-z0-9 ()]*')"
        reap "$GPID"; return 2 ;; esac
      local seen=absent; case "$E" in *"$SRLINE"*) seen=present ;; esac
      if [ "$seen" != "$srwant" ]; then
        echo "SR3ARM [$name] ABORT: stochastic-rounding witness $seen, wanted $srwant — the bf16 arms share a math-mode banner and this line is the ONLY thing separating them"
        reap "$GPID"; return 2; fi
      local first; first=$(tr '\r' '\n' < "$OUT/sr3arm_$name.stdout" | grep -m1 -E "^[0-9]+$")
      if [ "$first" -le "$BASE" ]; then
        echo "SR3ARM [$name] ABORT: from-scratch ($first, expected > $BASE)"; reap "$GPID"; return 2; fi
      engaged=yes; echo "SR3ARM [$name] RESUME ENGAGED at $first  (math=$want, sr=$seen)"
    fi
    if [ "$cur" = "$laststep" ]; then
      stall=$((stall+1)); [ "$stall" -ge 20 ] && { echo "SR3ARM [$name] STALLED at $cur"; reap "$GPID"; return 1; }
    else stall=0; laststep=$cur; fi
    if [ "$cur" -ge "$STOP" ]; then echo "SR3ARM [$name] reached $cur — stopping"; reap "$GPID"; return 0; fi
  done
}

run_arm f32     "TF32 tensor cores" absent
run_arm bf16rne "BF16 tensor"       absent  NSL_MATMUL_BF16=1
run_arm bf16sr  "BF16 tensor"       present NSL_MATMUL_BF16=1 NSL_MATMUL_BF16_ROUND=sr
echo "SR3ARM_ALL_DONE $(date '+%F %T')"
