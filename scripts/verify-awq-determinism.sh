#!/usr/bin/env bash
# verify-awq-determinism.sh — Spec §6.1 precondition for #134.
#
# Runs the AWQ end-to-end pipeline 10 times serially, then once more with
# a higher thread count (threads=8), and verifies the captured Sidecar JSON is
# byte-identical across runs. Exits 0 on success, 1 on any divergence.
#
# Used at commit 1a of #134 to verify the AWQ sidecar is deterministic
# before commit 1b captures it as an insta baseline. Spec §6.1 forbids
# papering over non-determinism with a hash digest — fix the source.

set -euo pipefail

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

TEST_NAME="end_to_end_real_subprocess_matches_analytical_reference"

echo "verify-awq-determinism: capturing 10 sequential SIDECAR_DUMP runs of $TEST_NAME"
for i in $(seq 1 10); do
    SIDECAR_DUMP=1 cargo test -p nsl-codegen --test awq_full_pipeline -- \
        --test-threads 1 --nocapture "$TEST_NAME" \
        2>&1 \
        | awk '/^SIDECAR_JSON_START$/{flag=1; next} /^SIDECAR_JSON_END$/{flag=0} flag' \
        > "$TMP_DIR/run_$i.json"
    if [ ! -s "$TMP_DIR/run_$i.json" ]; then
        echo "FAIL: run $i produced empty SIDECAR_JSON output. Is SIDECAR_DUMP instrumentation wired up?"
        exit 1
    fi
done

echo "verify-awq-determinism: capturing thread-varied SIDECAR_DUMP run (threads=8)"
SIDECAR_DUMP=1 cargo test -p nsl-codegen --test awq_full_pipeline -- \
    --test-threads 8 --nocapture "$TEST_NAME" \
    2>&1 \
    | awk '/^SIDECAR_JSON_START$/{flag=1; next} /^SIDECAR_JSON_END$/{flag=0} flag' \
    > "$TMP_DIR/threads_8.json"
if [ ! -s "$TMP_DIR/threads_8.json" ]; then
    echo "FAIL: threads=8 run produced empty SIDECAR_JSON output. Is SIDECAR_DUMP instrumentation wired up?"
    exit 1
fi

echo "verify-awq-determinism: comparing 11 runs"
DIFFS=0
REF="$TMP_DIR/run_1.json"
for run in run_2 run_3 run_4 run_5 run_6 run_7 run_8 run_9 run_10 threads_8; do
    if ! diff -q "$REF" "$TMP_DIR/${run}.json" > /dev/null; then
        echo "DIVERGENCE: $run differs from run_1"
        diff "$REF" "$TMP_DIR/${run}.json" | head -20
        DIFFS=$((DIFFS + 1))
    fi
done

if [ "$DIFFS" -ne 0 ]; then
    echo ""
    echo "FAIL: $DIFFS run(s) diverged from baseline."
    echo "Likely sources: HashMap iteration order in Sidecar serialization."
    echo "Fix at source per spec §6.1 — do NOT mask with a hash digest."
    exit 1
fi

echo "PASS: all 11 runs produced byte-identical Sidecar JSON."
