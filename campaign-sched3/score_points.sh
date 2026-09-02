#!/usr/bin/env bash
# Score a list of spliced points on held-out. TF32, matching the whole
# trajectory record. A point that fails prints FAILED and names its stderr --
# never 0.0000, which is a plausible-looking loss and would enter a table.
set -uo pipefail
W="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
C="$W/campaign-sched3"
N="${NSL_BIN:-/home/brandon/Projects/NSL-target-sched3/release/nsl}"
OUTTSV="${1:?usage: score_points.sh OUT.tsv point [point...]}"; shift
export NSL_STDLIB_PATH="$W/stdlib"
unset NSL_MATMUL_BF16 NSL_MATMUL_BF16_ROUND NSL_MATMUL_BF16_CAST_CACHE NSL_MATMUL_BF16_LT
cd "$W/models/coder1b" || exit 1
: > "$OUTTSV"
for pt in "$@"; do
  f="$C/spliced/$pt.nslm"
  [ -f "$f" ] || { printf '%s\tMISSING\tMISSING\n' "$pt" >> "$OUTTSV"; continue; }
  sed -i "s|model_load(m, \"[^\"]*\")|model_load(m, \"$f\")|" val_from_splice.nsl
  NSL_GPU_GUARD_THRESHOLD_MIB=4000 bash "$W/scripts/gpu-guard.sh" run -- \
    "$N" run --source-ad val_from_splice.nsl > "$C/val_$pt.stdout" 2> "$C/val_$pt.stderr"
  s=$(grep -A1 VAL_LOSS_STACK "$C/val_$pt.stdout" | tail -1 | tr -d ' ')
  w=$(grep -A1 VAL_LOSS_WEB   "$C/val_$pt.stdout" | tail -1 | tr -d ' ')
  case "$s$w" in
    *[0-9]*) printf '%s\t%s\t%s\n' "$pt" "$s" "$w" >> "$OUTTSV" ;;
    *)       printf '%s\tFAILED\tFAILED\n' "$pt" >> "$OUTTSV" ;;
  esac
done
echo "SCORING_DONE $OUTTSV"
