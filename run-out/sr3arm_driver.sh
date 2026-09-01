#!/usr/bin/env bash
# Three-arm matched pair: f32 / bf16+RNE / bf16+SR, from ONE checkpoint on
# IDENTICAL batches (the checkpoint restores the loader slot and the RNG
# sidecar), differing ONLY in arithmetic.
#
# ── why this exists, sharpened 2026-08-30 ────────────────────────────────
#
# The 1B chain's held-out PEAKED at 262M tokens and then went BACKWARDS:
# VAL_STACK/WEB 3.872/5.834 at micro 64,000 -> 4.569/6.291 at micro 160,000,
# +0.70/+0.46 nats WORSE over 393M tokens. Training loss over that same span
# was FLAT to -0.03 +/- 0.13 nats per 100k micro, so the loss stream showed
# nothing. The leg ran bf16 at lr=1.5e-5 throughout.
#
# That is the same signature that convicted bf16 at held lr=3e-5, reappearing
# at half the peak — halving the LR seems to have DELAYED the degradation, not
# prevented it. This run decides whether precision is the cause.
#
# ── two design corrections over the previous version ─────────────────────
#
# 1. EACH ARM SAVES ITS OWN THETA, and held-out is scored on all three at the
#    end. The previous version was write-free and produced no per-arm weights,
#    leaving only the paired training-loss delta as evidence — and the loss
#    stream is exactly the instrument that just failed to see a 0.7 nat
#    generalization change. Held-out is the instrument; training loss is the
#    cheap proxy that missed it. Arms write to their OWN directory, never the
#    chain's, and that is asserted below rather than assumed.
# 2. Held-out is scored under TF32 for every arm, because the whole trajectory
#    record was measured under TF32. (Verified 2026-08-30: bf16 vs TF32 on the
#    same weights differ in the 5th decimal, so this does not bias any arm —
#    but the comparison stays like-for-like on principle.)
#
# The arm's own identity is asserted on the runtime's output before it earns
# wall-clock: bf16+RNE and bf16+SR print the SAME math-mode banner, so the
# stochastic-rounding line is checked separately, and the arms that must NOT
# have SR assert its ABSENCE — the half that would have caught the 30-hour
# mislabel of stages 2 and 3.
set -u
WT=/home/brandon/Projects/NSL/.claude/worktrees/int1b
cd "$WT/models/coder1b" || exit 9
export NSL_STDLIB_PATH=$WT/stdlib
GUARD=$WT/scripts/gpu-guard.sh
NSL=/home/brandon/Projects/NSL-target-fix2p16/release/nsl
OUT=$WT/run-out
BK=$WT/models/coder1b/checkpoints_backup
SRLINE="bf16 operand cast: STOCHASTIC rounding"
ARMDIR=checkpoints_sr3arm
TEMPLATE=/home/brandon/Projects/NSL/.claude/worktrees/pilot1b/models/coder1b/checkpoints/pilot_lr3e5_final.nslm

[ -x "$NSL" ] || { echo "SR3ARM ABORT: $NSL missing"; exit 5; }
if ! ( cd "$OUT/repro" && timeout 300 "$NSL" run --source-ad --seed 7 min4.nsl \
        > "$OUT/sr3arm_preflight.out" 2> "$OUT/sr3arm_preflight.err" ) \
   || ! grep -q REPRO_PASSED_2P16 "$OUT/sr3arm_preflight.out"; then
  echo "SR3ARM ABORT: binary does not survive micro-step 65,536"; exit 7; fi
echo "SR3ARM preflight OK: binary crosses 2^16"

# Newest base first: the probe is most informative from the furthest-trained
# theta, and the question is forward-going ("from HERE, does f32 recover?").
if   [ -f "$BK/lr15_stage4_opt40000/pretrain_prod_state.nslm" ]; then
  CKPT="checkpoints_backup/lr15_stage4_opt40000/pretrain_prod_state.nslm"; BASE=160000
elif [ -f "$BK/lr15_stage4_opt24000/pretrain_prod_state.nslm" ]; then
  CKPT="checkpoints_backup/lr15_stage4_opt24000/pretrain_prod_state.nslm"; BASE=96000
else
  CKPT="checkpoints_backup/lr15_stage2_opt16000/pretrain_prod_state.nslm"; BASE=64000
fi
CKMICRO=$((BASE + 24000))          # 98.3M tokens/arm — the length that convicted bf16
STOP=$((CKMICRO + 200))
echo "SR3ARM: base=$BASE ($((BASE*4096/1000000))M tokens)  stop=$CKMICRO  ckpt=$CKPT"
echo "SR3ARM: base theta is BF16-TRAINED — state that in the write-up; this"
echo "SR3ARM: measures forward-going cost from a shared state, not what a"
echo "SR3ARM: from-scratch f32 run would have done."
mkdir -p "$ARMDIR"

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
    case "$(readlink -f /proc/$p/exe 2>/dev/null)" in */nsl_run_*/*) kill -9 "$p" 2>/dev/null ;; esac
  done
  sleep 8
}

# $1=name $2=banner fragment $3=present|absent for SR $4..=env
run_arm() {
  local name=$1 want=$2 srwant=$3; shift 3
  local save="$ARMDIR/${name}_state.nslm"
  local probe="sr3arm_${name}.nsl"
  # Build this arm's recipe: resume from the shared base, save to the ARM's
  # own file. Never the chain's.
  sed -e "s|checkpoint_load=\"checkpoints/pretrain_prod_state.nslm\", checkpoint_save=\"checkpoints/pretrain_prod_state.nslm\", checkpoint_every=2000|checkpoint_load=\"$CKPT\", checkpoint_save=\"$save\", checkpoint_every=2000|" \
      pretrain_prod.nsl > "$probe"
  grep -q "checkpoint_save=\"$save\"" "$probe" || { echo "SR3ARM [$name] ABORT: recipe rewrite failed"; return 3; }
  grep -q 'checkpoint_save="checkpoints/pretrain_prod_state.nslm"' "$probe" \
    && { echo "SR3ARM [$name] ABORT: recipe still writes the CHAIN checkpoint"; return 3; }

  echo "SR3ARM [$name] start $(date '+%H:%M:%S')  env: ${*:-none}  -> $save"
  ( unset NSL_MATMUL_BF16 NSL_MATMUL_BF16_ROUND
    for kv in "$@"; do export "$kv"; done
    bash "$GUARD" run -- "$NSL" run --source-ad --checkpoint-blocks --checkpoint-selective \
      --fuse-rmsnorm-backward --fuse-wgrad-accum "$probe" \
      > "$OUT/sr3arm_$name.stdout" 2> "$OUT/sr3arm_$name.stderr" ) &
  local GPID=$! engaged="" laststep="" stall=0
  while true; do
    sleep 30
    if grep -q "panicked\|REFUSING" "$OUT/sr3arm_$name.stderr" 2>/dev/null; then
      echo "SR3ARM [$name] CRASHED: $(tr '\r' '\n' < "$OUT/sr3arm_$name.stderr" | grep -m1 'panicked\|REFUSING')"; reap "$GPID"; return 1; fi
    if ! kill -0 "$GPID" 2>/dev/null; then echo "SR3ARM [$name] EXITED early at $(tr '\r' '\n' < "$OUT/sr3arm_$name.stdout" | grep -E '^[0-9]+$' | tail -1)"; return 1; fi
    local cur; cur=$(tr '\r' '\n' < "$OUT/sr3arm_$name.stdout" 2>/dev/null | grep -E "^[0-9]+$" | tail -1)
    [ -z "${cur:-}" ] && continue
    if [ -z "$engaged" ]; then
      local E; E=$(tr '\r' '\n' < "$OUT/sr3arm_$name.stderr")
      case "$E" in *"cuBLAS math mode: $want"*) ;; *)
        echo "SR3ARM [$name] ABORT: banner is not '$want': $(printf '%s' "$E" | grep -m1 -oE 'cuBLAS math mode: [A-Za-z0-9 ()]*')"
        reap "$GPID"; return 2 ;; esac
      local seen=absent; case "$E" in *"$SRLINE"*) seen=present ;; esac
      [ "$seen" = "$srwant" ] || { echo "SR3ARM [$name] ABORT: SR witness $seen, wanted $srwant"; reap "$GPID"; return 2; }
      local first; first=$(tr '\r' '\n' < "$OUT/sr3arm_$name.stdout" | grep -m1 -E "^[0-9]+$")
      [ "$first" -gt "$BASE" ] || { echo "SR3ARM [$name] ABORT: from-scratch ($first)"; reap "$GPID"; return 2; }
      engaged=yes; echo "SR3ARM [$name] RESUME ENGAGED at $first (math=$want, sr=$seen)"
    fi
    if [ "$cur" = "$laststep" ]; then
      stall=$((stall+1)); [ "$stall" -ge 20 ] && { echo "SR3ARM [$name] STALLED at $cur"; reap "$GPID"; return 1; }
    else stall=0; laststep=$cur; fi
    if [ "$cur" -ge "$STOP" ]; then
      grep -q "micro-batch step $CKMICRO" "$OUT/sr3arm_$name.stderr" \
        && echo "SR3ARM [$name] reached $cur, theta saved at $CKMICRO" \
        || echo "SR3ARM [$name] WARNING: reached $cur but no save at $CKMICRO"
      reap "$GPID"; return 0
    fi
  done
}

score() {   # $1=arm name -> splice its theta and score held-out under TF32
  # SEPARATE `local` statements on purpose: the shell expands every word of a
  # `local` command BEFORE the builtin performs any assignment, so a
  # cross-reference like `local name=$1 ck="${name}..."` resolves ${name} in
  # the OUTER scope -- unbound, and under `set -u` that aborts the script.
  # This cost the 2026-08-30 probe its scoring stage after 14 h of arms.
  local name=$1
  local ck="$ARMDIR/${name}_state.nslm"
  local sp="$OUT/spliced_sr3arm_${name}.nslm"
  [ -f "$ck" ] || { echo "SR3ARM [$name] no theta to score"; return 1; }
  python3 "$WT/tools/nslm_splice.py" "$ck" "$TEMPLATE" "$sp" || return 1
  sed -i "s|model_load(m, \"../../run-out/spliced_[^\"]*\")|model_load(m, \"../../run-out/spliced_sr3arm_${name}.nslm\")|" val_from_splice.nsl
  ( unset NSL_MATMUL_BF16 NSL_MATMUL_BF16_ROUND      # TF32: match the record
    bash "$GUARD" run -- "$NSL" run --source-ad val_from_splice.nsl \
      > "$OUT/sr3arm_${name}_val.stdout" 2> "$OUT/sr3arm_${name}_val.stderr" )
  local s w
  s=$(grep -A1 VAL_LOSS_STACK "$OUT/sr3arm_${name}_val.stdout" | tail -1)
  w=$(grep -A1 VAL_LOSS_WEB   "$OUT/sr3arm_${name}_val.stdout" | tail -1)
  echo "SR3ARM_VAL $name STACK=$s WEB=$w"
}

run_arm f32     "TF32 tensor cores" absent
run_arm bf16rne "BF16 tensor"       absent  NSL_MATMUL_BF16=1
run_arm bf16sr  "BF16 tensor"       present NSL_MATMUL_BF16=1 NSL_MATMUL_BF16_ROUND=sr
echo "SR3ARM: all arms done, scoring held-out $(date '+%H:%M:%S')"
for a in f32 bf16rne bf16sr; do score "$a"; done
echo "SR3ARM: reference — base (micro $BASE) scored 4.569/6.291; the 262M peak scored 3.872/5.834"
echo "SR3ARM_ALL_DONE $(date '+%F %T')"
