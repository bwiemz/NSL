#!/usr/bin/env bash
#
# gpu-tier.sh — the single-GPU certification tiers (roadmap item 14).
#
# One command, three depths. Each tier answers a different question:
#
#   smoke      "is the GPU + harness sane right now?"
#              The curated canary (tools/gpu-canary.txt) via tools/gpu-test.sh:
#              every entry makes a real numerical claim against a CPU
#              reference, one process per test.
#
#   certify    "does everything the tree certifies still hold on hardware?"
#              The full certification lane (scripts/gpu-cert.sh --run): every
#              run-class #[ignore] gate in the tree, batched per cargo target,
#              with per-gate confirmation reruns on red targets.
#
#   endurance  "does production-scale 1B training still complete?"
#              models/benchmarks/endurance_1b.py: 1B params @ seq 2048 —
#              SR-BF16 endurance arm + f32 reference arm + checkpoint-resume
#              in a fresh process.
#
# NAMING NOTE: `gpu-cert.sh --tier` is a DIFFERENT axis — it selects gate
# capability classes (gpu|toolchain|multiproc|isolate|all) WITHIN the certify
# tier. This script's axis is depth/duration. Extra arguments after the tier
# name are passed through to the underlying runner, so
# `gpu-tier.sh certify --tier all` runs the certification tier over all
# capability classes.
#
# Budgets (measured on the reference box — RTX PRO 4500 Blackwell 32GB,
# warm CARGO_TARGET_DIR; see models/benchmarks/GPU_TIERS_2026_08_24.md):
#
#   smoke      ~10 min
#   certify    ~90 min
#   endurance  ~270 min (three phases; --skip-f32/--skip-resume shorten it)
#
# A budget overrun WARNS but does not change the exit status: the budget
# documents expectation on the reference box, while correctness enforcement
# lives in the runners' own per-gate/per-phase timeouts. Failing a green run
# for being slow on a cold cache would train operators to ignore red.
#
# Concurrency: every runner refuses to start when the device is busy or
# another guarded run holds the lock — see scripts/gpu-guard.sh. The refusal
# is a hard exit, not a warning: a 2026-08-19 sweep that warned and proceeded
# lost two arms to an orphaned run's resident 22 GB.
#
# Usage:
#   scripts/gpu-tier.sh smoke|certify|endurance [--dry-run] [runner args...]
#
# --dry-run prints the exact command that would run, and runs nothing.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ $# -eq 0 ]]; then
    echo "gpu-tier: no tier given (want smoke|certify|endurance)" >&2
    exit 2
fi
tier="$1"; shift

dry=0
if [[ "${1:-}" == "--dry-run" ]]; then dry=1; shift; fi

# Budget minutes per tier, measured on the reference box (see header).
case "${tier}" in
    smoke)     cmd=(tools/gpu-test.sh);                          budget_min=10 ;;
    certify)   cmd=(scripts/gpu-cert.sh --run);                  budget_min=90 ;;
    endurance) cmd=(python3 models/benchmarks/endurance_1b.py);  budget_min=270 ;;
    *) echo "gpu-tier: unknown tier '${tier}' (want smoke|certify|endurance)" >&2; exit 2 ;;
esac
cmd+=("$@")

if [[ "${dry}" -eq 1 ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'
    exit 0
fi

echo "gpu-tier: ${tier} — budget ~${budget_min} min on the reference box"
t0="${SECONDS}"
rc=0
"${cmd[@]}" || rc=$?
elapsed=$(( SECONDS - t0 ))
echo "gpu-tier: ${tier} finished in $(( elapsed / 60 ))m$(( elapsed % 60 ))s (exit ${rc})"
if (( elapsed > budget_min * 60 )); then
    echo "gpu-tier: BUDGET OVERRUN — ${tier} took $(( elapsed / 60 )) min against a" >&2
    echo "          ~${budget_min} min budget. If this box matches the reference" >&2
    echo "          hardware and the cache was warm, the budget table in" >&2
    echo "          $0 and models/benchmarks/GPU_TIERS_2026_08_24.md is stale." >&2
fi
exit "${rc}"
