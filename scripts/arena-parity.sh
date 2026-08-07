#!/usr/bin/env bash
# Item 5 — the transient arena's byte-identity gate.
#
# Builds one deterministic coder50m training program TWICE — with and without
# `--transient-arena` — runs both to completion on a small fixed token
# stream, and demands:
#   1. byte-identical loss streams,
#   2. byte-identical .nslm checkpoints,
#   3. the planned run's teardown reconciles (placements == binds,
#      0 guard failures, 0 misplaced),
#   4. a per-step red-zone canary pass (NSL_ARENA_CHECK=1) that is ALSO
#      byte-identical.
#
# This is the gate `--transient-arena`'s default-off status is waiting on,
# and the gate CUDA-graph offset feeding waits on after that. It found two
# real bugs before it ever passed (an allocator entry that bypassed the pin,
# and a BFD layout whose shared offsets aliased across slots) — which is the
# argument for running it, not a reason it is scary.
#
# Usage: scripts/arena-parity.sh [path-to-nsl-binary]
#   NSL defaults to target/release/nsl. Needs a CUDA GPU.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSL="${1:-${REPO}/target/release/nsl}"
WORK="$(mktemp -d /tmp/nsl_arena_parity.XXXXXX)"
trap 'rm -rf "${WORK}"' EXIT

export NSL_STDLIB_PATH="${REPO}/stdlib"

echo "== arena-parity: workdir ${WORK}"
cp "${REPO}/models/coder50m/model.nsl" "${REPO}/models/coder50m/config.nsl" "${WORK}/"

python3 "${REPO}/models/benchmarks/gen_tokens.py" --tokens 40960 "${WORK}/tokens.bin" \
  || python3 "${REPO}/models/benchmarks/gen_tokens.py" --tokens 40960 --out "${WORK}/tokens.bin"

for arm in planned unplanned; do
  sed -e "s|CERT_TOKENS_PATH|${WORK}/tokens.bin|" \
      "${REPO}/models/coder50m/pretrain_cert.nsl" > "${WORK}/${arm}.nsl"
  printf 'model_save(m, "%s/ckpt_%s.nslm")\nprint("CKPT_SAVED")\n' "${WORK}" "${arm}" \
    >> "${WORK}/${arm}.nsl"
done

FLAGS=(--source-ad --deterministic --seed 4242)
( cd "${WORK}" && "${NSL}" build planned.nsl   "${FLAGS[@]}" --transient-arena -o planned )
( cd "${WORK}" && "${NSL}" build unplanned.nsl "${FLAGS[@]}"                   -o unplanned )

( cd "${WORK}" && ./planned   > planned.out   2> planned.err )
( cd "${WORK}" && ./unplanned > unplanned.out 2> unplanned.err )

fail() { echo "arena-parity: FAIL — $1" >&2; exit 1; }

diff <(grep -v CKPT "${WORK}/planned.out") <(grep -v CKPT "${WORK}/unplanned.out") \
  > /dev/null || fail "loss streams differ"
cmp -s "${WORK}/ckpt_planned.nslm" "${WORK}/ckpt_unplanned.nslm" \
  || fail "checkpoints differ"

TEARDOWN="$(grep -E '^\[arena\] teardown:' "${WORK}/planned.err" || true)"
[ -n "${TEARDOWN}" ] || fail "no arena teardown line — placement never ran (vacuous)"
echo "   ${TEARDOWN}"
BINDS="$(sed -E 's/.* from ([0-9]+) bind.*/\1/' <<<"${TEARDOWN}")"
PLACED="$(sed -E 's/.*teardown: ([0-9]+) placement.*/\1/' <<<"${TEARDOWN}")"
[ "${BINDS}" -gt 0 ] || fail "zero binds — nothing was admitted (vacuous)"
[ "${BINDS}" = "${PLACED}" ] || fail "reconciliation: ${PLACED} placements from ${BINDS} binds"
grep -qE '0 guard failure' <<<"${TEARDOWN}" || fail "guard failures at teardown"
grep -qE ' 0 misplaced' <<<"${TEARDOWN}" || fail "misplaced placements (scratch at a stable address)"

( cd "${WORK}" && NSL_ARENA_CHECK=1 ./planned > canary.out 2> canary.err )
grep -qE 'corrupted' "${WORK}/canary.err" && fail "red-zone canary found corruption"
diff <(grep -v CKPT "${WORK}/canary.out") <(grep -v CKPT "${WORK}/planned.out") \
  > /dev/null || fail "canary run changed values"

echo "arena-parity: PASS (streams + checkpoints byte-identical; ${TEARDOWN#\[arena\] })"
