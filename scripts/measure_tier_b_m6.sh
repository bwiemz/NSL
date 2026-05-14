#!/usr/bin/env bash
# M6 — wall-time median-of-5 measurement for PCA Tier B.
#
# Spec: §8 of docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md
# Plan: B1.5-5 step 1 of docs/superpowers/plans/2026-05-13-pca-tier-b15-and-b2-implementation.md
#
# Per measurement protocol:
#  - 5 outer runs x tier_b ∈ {on, off}
#  - 100 inner iterations per outer run via bench's --iterations
#  - Parse `tier_b_bench_result:` output line; extract median_us and skip_ratio
#  - Emit CSV to stdout: fixture,outer_run,tier_b,median_us,skip_ratio,seed
#
# Usage:
#   bash scripts/measure_tier_b_m6.sh [FIXTURE]
#
# Default fixture: gate_4096.

set -euo pipefail

FIXTURE="${1:-gate_4096}"
OUTER_RUNS="${OUTER_RUNS:-5}"
INNER_ITERS="${INNER_ITERS:-100}"
SEED="${SEED:-42}"

# Prefer pre-built binary if available; otherwise invoke via cargo run.
BENCH_BIN_PATH="${BENCH_BIN:-target/release/bench.exe}"
if [ -x "$BENCH_BIN_PATH" ]; then
    BENCH_CMD=("$BENCH_BIN_PATH")
elif [ -x "target/release/bench" ]; then
    BENCH_CMD=("target/release/bench")
else
    BENCH_CMD=(cargo run -q -p nsl-codegen --features "cuda debug_kernel_instrumentation" --bin bench --release --)
fi

echo "fixture,outer_run,tier_b,median_us,skip_ratio,seed"
for run in $(seq 1 "$OUTER_RUNS"); do
    for tier_b in on off; do
        OUT=$("${BENCH_CMD[@]}" \
            --fixture "$FIXTURE" \
            --tier-b "$tier_b" \
            --seed "$SEED" \
            --iterations "$INNER_ITERS" \
            --emit-time-only \
            2>/dev/null \
            | grep '^tier_b_bench_result:' || true)
        if [ -z "$OUT" ]; then
            echo "ERR: bench produced no result line for $FIXTURE / $tier_b / run=$run" >&2
            exit 2
        fi
        median=$(echo "$OUT" | sed -n 's/.*:median_us=\([0-9.eE+-]*\):.*/\1/p')
        skip=$(  echo "$OUT" | sed -n 's/.*:skip_ratio=\([0-9.eE+-]*\):.*/\1/p')
        echo "$FIXTURE,$run,$tier_b,$median,$skip,$SEED"
    done
done
