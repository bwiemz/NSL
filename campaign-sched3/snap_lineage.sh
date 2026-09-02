#!/usr/bin/env bash
# Snapshot theta at every cadence write of one arm and emit its lineage record.
#
# The record is what the trajectory builder reads. It is emitted HERE, next to
# the write, rather than reconstructed later from filenames -- a filename is
# exactly the kind of evidence that let a probe-arm checkpoint be tabled as a
# chain checkpoint in the 1B campaign.
set -uo pipefail
ARM="${1:?usage: snap_lineage.sh A|B|C}"
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="$WT/campaign-sched3"
CK="$C/ckpt_$ARM/state.nslm"
SNAP="$C/snap_$ARM"
LOG="$C/out_$ARM/snap.log"
mkdir -p "$SNAP"
say() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

ckstep() { python3 - "$CK.optim" <<'PY' 2>/dev/null
import sys,json,struct
f=open(sys.argv[1],'rb'); f.seek(8); h=struct.unpack('<Q',f.read(8))[0]
print(json.loads(f.read(h).decode().rstrip('\x00'))['step_count'])
PY
}

say "snapshotter for arm $ARM started"
while pgrep -f "bash .*arm_driver\.sh $ARM" >/dev/null 2>&1; do
  sleep 45
  [ -f "$CK" ] || continue
  s=$(ckstep); [ -n "$s" ] || continue
  [ "$s" -gt 64000 ] 2>/dev/null || continue
  [ -f "$SNAP/m$s.nslm" ] && continue
  # Wait for BOTH files' mtimes to settle before copying 12.9 GB mid-write.
  ok=0
  for _ in 1 2 3; do
    a=$(stat -c %Y "$CK" 2>/dev/null; stat -c %Y "$CK.optim" 2>/dev/null)
    sleep 12
    b=$(stat -c %Y "$CK" 2>/dev/null; stat -c %Y "$CK.optim" 2>/dev/null)
    [ "$a" = "$b" ] && ok=$((ok+1)) || ok=0
  done
  [ "$ok" -ge 3 ] || { say "arm $ARM: checkpoint at $s still settling, deferring"; continue; }
  cp --reflink=auto "$CK" "$SNAP/m$s.nslm" || { say "arm $ARM: copy of m$s failed"; continue; }
  cp --reflink=auto "$CK.optim" "$SNAP/m$s.nslm.optim" || { say "arm $ARM: optim copy of m$s failed"; rm -f "$SNAP/m$s.nslm"; continue; }
  read -r ints repl < "$C/out_$ARM/interruptions.txt" 2>/dev/null || { ints=0; repl=0; }
  python3 - "$SNAP/m$s.nslm" "$ARM" "${ints:-0}" "${repl:-0}" <<'PY' >> "$LOG" 2>&1
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
sys.path.insert(0, "/home/brandon/Projects/NSL/.claude/worktrees/sched3/campaign-sched3")
import lineage
theta = pathlib.Path(sys.argv[1]); arm = sys.argv[2]
br = json.loads((pathlib.Path(theta).parents[1] / "branch_lineage.json").read_text())
rec = lineage.record(theta, arm=arm, run_uuid=f"sched3-arm{arm}",
                     parent_run_uuid=br["run_uuid"],
                     parent_ckpt_sha=br["model_checkpoint_sha256"],
                     repo=pathlib.Path("/home/brandon/Projects/NSL/.claude/worktrees/sched3"),
                     binary=pathlib.Path("/home/brandon/Projects/NSL-target-sched3/release/nsl"),
                     interruption_count=int(sys.argv[3]), replayed_microsteps=int(sys.argv[4]))
theta.with_suffix(".lineage.json").write_text(json.dumps(rec, indent=2))
print(f"lineage: arm {rec['arm']} micro {rec['micro_step']} tokens {rec['tokens_seen']} lr {rec['current_lr']:.6e}")
PY
  say "arm $ARM: snapshotted + lineage at micro $s"
done
say "snapshotter for arm $ARM exiting"
