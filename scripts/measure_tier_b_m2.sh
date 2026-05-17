#!/usr/bin/env bash
# M2 — skip-ratio sweep across the three-tier fixture matrix.
#
# Spec: §4 + §8.4 of docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md
# Plan: B1.5-5 step 2 of docs/superpowers/plans/2026-05-13-pca-tier-b15-and-b2-implementation.md
#
# Per measurement protocol:
#  - Sweep across gate + 3 sensitivity + 6 parity fixtures × tier_b ∈ {on, off}
#  - Single outer run per fixture (m2 is structural skip-ratio; not 5-run median)
#  - 100 inner iterations via bench's --iterations (matches m6 protocol)
#  - Emit CSV to stdout: fixture,tier_b,skip_ratio,median_us,seed
#
# Usage:
#   bash scripts/measure_tier_b_m2.sh

set -euo pipefail

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

FIXTURES=(
    gate_4096
    sensitivity_10
    sensitivity_50
    sensitivity_90
    parity_1
    parity_2
    parity_3
    parity_4
    parity_5
    parity_6
)

echo "fixture,tier_b,skip_ratio,median_us,seed"
for fx in "${FIXTURES[@]}"; do
    for tier_b in on off; do
        OUT=$("${BENCH_CMD[@]}" \
            --fixture "$fx" \
            --tier-b "$tier_b" \
            --seed "$SEED" \
            --iterations "$INNER_ITERS" \
            --emit-time-only \
            2>/dev/null \
            | grep '^tier_b_bench_result:' || true)
        if [ -z "$OUT" ]; then
            echo "WARN: bench produced no result line for $fx / $tier_b — emitting NA row" >&2
            echo "$fx,$tier_b,NA,NA,$SEED"
            continue
        fi
        median=$(echo "$OUT" | sed -n 's/.*:median_us=\([0-9.eE+-]*\):.*/\1/p')
        skip=$(  echo "$OUT" | sed -n 's/.*:skip_ratio=\([0-9.eE+-]*\):.*/\1/p')
        echo "$fx,$tier_b,$skip,$median,$SEED"
    done
done
