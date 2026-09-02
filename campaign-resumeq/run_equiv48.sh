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
C="$W/campaign-resumeq/${SUB:-.}"
N="${NSL_BIN:-/home/brandon/Projects/NSL-target-resumeq/release/nsl}"
export NSL_STDLIB_PATH="$W/stdlib"
R="$W/models/coder50m"
SEED=1000
STOP_AT=24          # micro-step of the mid-run cadence write
FLAGS="--seed $SEED --source-ad ${EXTRA_FLAGS:-}"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$C/equiv.log"; }

mkrecipe() {  # $1 = arm dir, $2 = "load" to add checkpoint_load=
  local d="$1" ck="$1/state.nslm" out="$R/_re_$(basename "$1").nsl"
  sed "s|RESUME_EQUIV_CKPT|$ck|" "$R/resume_equiv48.nsl" > "$out"
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
  # POLL THE CHECKPOINT, NOT THE LOG. The child's stdout is redirected to a
  # file, so it is BLOCK-buffered: for a ~4 s run the log stays empty until
  # exit and a log-watching poll never fires. That is why two earlier attempts
  # ran to completion instead of being interrupted. A checkpoint write is a
  # real filesystem event and cannot be buffered away.
  local seen=0
  for _ in $(seq 1 20000); do
    sleep 0.02
    if [ -f "$d/state.nslm.optim" ]; then
      local cs; cs=$(python3 - "$d/state.nslm.optim" 2>/dev/null <<'PYX'
import sys,json,struct
try:
    f=open(sys.argv[1],'rb'); f.seek(8); h=struct.unpack('<Q',f.read(8))[0]
    print(json.loads(f.read(h).decode().rstrip('\x00'))['step_count'])
except Exception: print(-1)
PYX
)
      if [ -n "$cs" ] && [ "$cs" -ge "$STOP_AT" ] 2>/dev/null; then seen=1; break; fi
    fi
    kill -0 "$pid" 2>/dev/null || break
  done
  [ "$seen" = 1 ] || { say "phase1: never reached micro $STOP_AT"; return 1; }
  # KILL IMMEDIATELY. The previous version waited up to 10 s for the mtime to
  # settle BEFORE killing -- correct for a 5.5 h arm, fatal for a 4 s run,
  # which simply finished during the wait (twice). Instead: kill at once, then
  # VERIFY the checkpoint rather than waiting for it, and redo phase 1 if the
  # kill landed mid-write and tore it.
  local pg; pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  [ -n "$pg" ] && kill -9 -"$pg" 2>/dev/null
  sleep 1
  local intact; intact=$(python3 - "$d/state.nslm" "$d/state.nslm.optim" <<'PYX'
import sys, json, struct, pathlib
def ok(p):
    p = pathlib.Path(p)
    with p.open('rb') as f:
        f.read(4); f.read(4); hs = struct.unpack('<Q', f.read(8))[0]
        j = json.loads(f.read(hs).decode().rstrip('\x00'))
    tbl = j.get('params') or j.get('tensors') or []
    start = ((16 + hs) + 63) // 64 * 64
    need = start + sum(t.get('nbytes', 0) for t in tbl)
    return p.stat().st_size >= need
try: print("yes" if ok(sys.argv[1]) and ok(sys.argv[2]) else "torn")
except Exception: print("torn")
PYX
)
  [ "$intact" = "yes" ] || { say "phase1: checkpoint TORN by the kill; retrying"; return 2; }
  say "phase1: stopped after the cadence write at micro $STOP_AT"
}

resume_rest() {  # $1 = arm dir
  local d="$1"
  local rec; rec=$(mkrecipe "$d" load)
  ( cd "$R" && "$N" run $FLAGS "$(basename "$rec")" ) > "$d/run2.log" 2>&1
  say "$(basename "$d") phase2: exit=$? steps=$(grep -cE '^[0-9]+$' "$d/run2.log")"
}

: > "$C/equiv.log"
say "INTERRUPTION EQUIVALENCE — 50M shuffle=true seed $SEED flags=[${EXTRA_FLAGS:-none}]"
run_full     "$C/unint1"
run_full     "$C/unint2"
for try in 1 2 3; do
  run_to_stop "$C/resume"; rc=$?
  [ $rc -eq 0 ] && { resume_rest "$C/resume"; break; }
  say "phase1 attempt $try failed rc=$rc"
done
say "EQUIV_RUNS_DONE"
