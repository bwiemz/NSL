#!/usr/bin/env bash
# M2 — FLOP reduction measurement.
# Runs Nsight Compute on a Tier-A-only baseline and a Tier-B-on variant
# of the standard_3doc fixture; computes FLOP ratio per spec sec 7 M2 formula.
#
# Prerequisites:
#   - $CUDA_PATH points to a CUDA 13.x install with `ncu` in bin/.
#   - A bench binary that launches the Tier B forward kernel with a
#     selectable --fixture flag and --tier-b={on,off} switch. The bench
#     binary is NOT in the current repo; users should adapt the existing
#     `pca_tier_a_forward_correctness::launch_forward` harness with a
#     skip_decisions_ptr arg (deferred to Tier B.1.5; see Task 9 notes).
#
# Spec: §7 M2.
set -euo pipefail

FIXTURE="${FIXTURE:-standard_3doc}"
NCU="${NCU:-$CUDA_PATH/bin/ncu}"
BENCH="${BENCH:-./target/release/nsl-codegen-bench}"

[ -x "$NCU" ] || { echo "ERR: ncu not at $NCU (set NCU= or CUDA_PATH=)"; exit 1; }
[ -x "$BENCH" ] || {
    echo "ERR: bench binary not at $BENCH"
    echo "Build the bench harness first: cargo build --release -p nsl-codegen-bench --features cuda"
    echo "(harness not yet in repo — see Task 9 / Tier B.1.5 deferral notes)"
    exit 1
}

# Spec sec 7 M2 metric list.
METRICS="smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum"

echo "Running Tier-A-only baseline (fixture=$FIXTURE) ..."
"$NCU" --metrics "$METRICS" --csv "$BENCH" --fixture "$FIXTURE" --tier-b=off \
    > /tmp/tier_a_baseline.csv

echo "Running Tier-B-on variant (fixture=$FIXTURE) ..."
"$NCU" --metrics "$METRICS" --csv "$BENCH" --fixture "$FIXTURE" --tier-b=on \
    > /tmp/tier_b_on.csv

# FLOPs = fadd + 2*ffma + fmul (per spec sec 7 M2 formula).
FLOP_A=$(awk -F, 'NR>1 {f+=$2+2*$3+$4} END {print f}' /tmp/tier_a_baseline.csv)
FLOP_B=$(awk -F, 'NR>1 {f+=$2+2*$3+$4} END {print f}' /tmp/tier_b_on.csv)
REDUCTION=$(awk -v a="$FLOP_A" -v b="$FLOP_B" 'BEGIN {printf "%.4f", 1 - b/a}')

echo ""
echo "=== M2 results ==="
echo "Tier-A FLOPs: $FLOP_A"
echo "Tier-B FLOPs: $FLOP_B"
echo "Reduction (1 - B/A): $REDUCTION (pass criterion: >= 0.30)"
awk -v r="$REDUCTION" 'BEGIN {exit (r < 0.30)}' && echo "M2 PASS" || echo "M2 FAIL"
