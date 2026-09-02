#!/usr/bin/env bash
# Interruption equivalence at Coder-50M.
#
#   UNINT1, UNINT2  two uninterrupted runs, micro 0 -> 24.
#                   THE CONTROL. If these two are not bit-identical then
#                   determinism is not holding and NOTHING can be attributed
#                   to the interruption. This arm is what makes the other one
#                   mean something -- the same role arm A played in the
#                   schedule campaign.
#   RESUME          micro 0 -> 12, killed after the cadence write, then
#                   re-run with checkpoint_load= added, 12 -> 24.
#
# All three under --deterministic --seed 1000.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="$W/campaign-resumeq"
N="${NSL_BIN:-/home/brandon/Projects/NSL-target-resumeq/release/nsl}"
export NSL_STDLIB_PATH="$W/stdlib"
R="$W/models/coder50m"
SEED=1000
STOP_AT=12          # micro-step of the mid-run cadence write
FLAGS="--seed $SEED --source-ad --deterministic"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$C/equiv.log"; }

mkrecipe() {  # $1 = arm dir, $2 = "load" to add checkpoint_load=
  local d="$1" ck="$1/state.nslm" out="$R/_re_$(basename "$1").nsl"
  sed "s|RESUME_EQUIV_CKPT|$ck|" "$R/resume_equiv.nsl" > "$out"
  if [ "${2:-}" = "load" ]; then
    # The resume contract: the SAME file, with checkpoint_load= added.
    sed -i "s|checkpoint_save=\"$ck\"|checkpoint_load=\"$ck\", checkpoint_save=\"$ck\"|" "$out"
  fi
  echo "$out"
}

run_full() {  # $1 = arm dir
  local d="$1"; mkdir -p "$d"; rm -f "$d"/state.nslm*
  local rec; rec=$(mkrecipe "$d")
  ( cd "$R" && "$N" run $FLAGS "$(basename "$rec")" ) > "$d/run.log" 2>&1
  say "$(basename "$d"): exit=$? steps=$(grep -cE '^[0-9]+$' "$d/run.log")"
}

run_to_stop() {  # $1 = arm dir -- run and kill after the cadence write at STOP_AT
  local d="$1"; mkdir -p "$d"; rm -f "$d"/state.nslm*
  local rec; rec=$(mkrecipe "$d")
  ( cd "$R" && setsid "$N" run $FLAGS "$(basename "$rec")" ) > "$d/run.log" 2>&1 &
  local pid=$!
  local seen=0
  for _ in $(seq 1 600); do
    sleep 1
    local last; last=$(grep -oE '^[0-9]+$' "$d/run.log" 2>/dev/null | tail -1)
    if [ -n "$last" ] && [ "$last" -ge "$STOP_AT" ] 2>/dev/null; then seen=1; break; fi
    kill -0 "$pid" 2>/dev/null || break
  done
  [ "$seen" = 1 ] || { say "phase1: never reached micro $STOP_AT"; return 1; }
  # let the cadence write settle before killing -- a kill mid-write corrupts
  # the only state the resume has.
  local a b
  for _ in 1 2 3; do
    a=$(stat -c %Y "$d/state.nslm" "$d/state.nslm.optim" 2>/dev/null)
    sleep 2
    b=$(stat -c %Y "$d/state.nslm" "$d/state.nslm.optim" 2>/dev/null)
    [ "$a" = "$b" ] && break
  done
  local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && kill -TERM -"$pg" 2>/dev/null; sleep 2
  [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
  say "phase1: stopped after the cadence write at micro $STOP_AT"
}

resume_rest() {  # $1 = arm dir
  local d="$1"
  local rec; rec=$(mkrecipe "$d" load)
  ( cd "$R" && "$N" run $FLAGS "$(basename "$rec")" ) > "$d/run2.log" 2>&1
  say "$(basename "$d") phase2: exit=$? steps=$(grep -cE '^[0-9]+$' "$d/run2.log")"
}

: > "$C/equiv.log"
say "INTERRUPTION EQUIVALENCE — 50M, shuffle=true, deterministic, seed $SEED"
run_full     "$C/unint1"
run_full     "$C/unint2"
run_to_stop  "$C/resume" && resume_rest "$C/resume"
say "EQUIV_RUNS_DONE"
