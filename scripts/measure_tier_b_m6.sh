#!/usr/bin/env bash
# M6 — single_doc wall-time regression.
# Median of 5 runs; pass if (B / A) <= 1.01 (worst-case Tier B overhead
# bound when no skipping is possible).
#
# Spec: §7 M6.
set -euo pipefail

FIXTURE="${FIXTURE:-single_doc}"
N_RUNS="${N_RUNS:-5}"
BENCH="${BENCH:-./target/release/nsl-codegen-bench}"

[ -x "$BENCH" ] || {
    echo "ERR: bench binary not at $BENCH"
    echo "Build first: cargo build --release -p nsl-codegen-bench --features cuda"
    echo "(harness not yet in repo — see Task 9 / Tier B.1.5 deferral notes)"
    exit 1
}

run_median() {
    local mode=$1
    local times=()
    for _ in $(seq "$N_RUNS"); do
        local t
        t=$("$BENCH" --fixture "$FIXTURE" --tier-b="$mode" --emit-time-only)
        times+=("$t")
    done
    printf '%s\n' "${times[@]}" | sort -n | awk -v n=$N_RUNS 'NR == int((n+1)/2) {print}'
}

echo "Measuring Tier-A-only median over $N_RUNS runs (fixture=$FIXTURE) ..."
T_A=$(run_median off)
echo "Measuring Tier-B-on median over $N_RUNS runs (fixture=$FIXTURE) ..."
T_B=$(run_median on)

RATIO=$(awk -v a="$T_A" -v b="$T_B" 'BEGIN {printf "%.4f", b/a}')

echo ""
echo "=== M6 results ==="
echo "Tier-A-only median: ${T_A}us"
echo "Tier-B-on median:   ${T_B}us"
echo "Ratio B/A: $RATIO (pass criterion: <= 1.01)"
awk -v r="$RATIO" 'BEGIN {exit (r > 1.01)}' && echo "M6 PASS" || echo "M6 FAIL"
