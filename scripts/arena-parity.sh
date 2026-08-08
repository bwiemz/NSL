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
# This is the gate `--transient-arena`'s default-off status is waiting on.
# It found two real bugs before it ever passed (an allocator entry that
# bypassed the pin, and a BFD layout whose shared offsets aliased across
# slots) — which is the argument for running it, not a reason it is scary.
#
# It ALSO gates the `--cuda-graphs` composition (steps 5-6 below): the
# arena's pinned addresses feed graph capture BY CONSTRUCTION (the
# allocator pin returns the same pointer every step, which is exactly what
# the recorder's digest-stability test wants), so the composed arm must be
# byte-identical too, reconcile its placements WHILE regions replay
# captured graphs, and stabilize capture no later than graphs-alone.
# Measured on this fixture 2026-08-07: graphs-alone captures after window
# 3 (35 replays); with the arena the digest stabilizes one window earlier
# (36 replays). The canonical cuda_graph_gate.nsl fixture shows NO delta —
# its transients are exactly the classes admission refuses (forward-region,
# unsized, tape-escaping), which is why THIS fixture hosts the composition
# gate.
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

# ── 5/6. CUDA-graph composition: graphs-alone, then graphs+arena ──────────
sed -e "s|CERT_TOKENS_PATH|${WORK}/tokens.bin|" \
    "${REPO}/models/coder50m/pretrain_cert.nsl" > "${WORK}/gr.nsl"
printf 'model_save(m, "%s/ckpt_gr.nslm")\nprint("CKPT_SAVED")\n' "${WORK}" \
  >> "${WORK}/gr.nsl"
cp "${WORK}/gr.nsl" "${WORK}/grarena.nsl"
sed -i "s|ckpt_gr.nslm|ckpt_grarena.nslm|" "${WORK}/grarena.nsl"
( cd "${WORK}" && "${NSL}" build gr.nsl      "${FLAGS[@]}" --cuda-graphs -o gr )
( cd "${WORK}" && "${NSL}" build grarena.nsl "${FLAGS[@]}" --cuda-graphs --transient-arena -o grarena )
( cd "${WORK}" && ./gr      > gr.out      2> gr.err )
( cd "${WORK}" && ./grarena > grarena.out 2> grarena.err )

for arm in gr grarena; do
  diff <(grep -v CKPT "${WORK}/${arm}.out") <(grep -v CKPT "${WORK}/unplanned.out") \
    > /dev/null || fail "${arm}: loss stream differs from eager"
  cmp -s "${WORK}/ckpt_${arm}.nslm" "${WORK}/ckpt_unplanned.nslm" \
    || fail "${arm}: checkpoint differs from eager"
done

GBANNER="$(grep -E '^\[cuda-graph\] regions=' "${WORK}/gr.err" || true)"
ABANNER="$(grep -E '^\[cuda-graph\] regions=' "${WORK}/grarena.err" || true)"
[ -n "${GBANNER}" ] || fail "graphs arm: no cuda-graph banner (capture never armed)"
[ -n "${ABANNER}" ] || fail "composed arm: no cuda-graph banner (capture never armed)"
echo "   graphs:   ${GBANNER}"
echo "   composed: ${ABANNER}"
gfield() { sed -E "s/.* $2=([0-9]+).*/\1/" <<<"$1"; }
[ "$(gfield "${GBANNER}" captured)" -gt 0 ] || fail "graphs arm captured nothing (vacuous)"
[ "$(gfield "${ABANNER}" captured)" -gt 0 ] || fail "composed arm captured nothing"
[ "$(gfield "${ABANNER}" mismatches)" = 0 ] || fail "composed arm had graph mismatches"
# The arena must never make capture stabilize LATER (its whole contribution
# is address stability); equality is fine — improvement is fixture-shaped.
[ "$(gfield "${ABANNER}" replays)" -ge "$(gfield "${GBANNER}" replays)" ] \
  || fail "composed arm replays fewer than graphs-alone — the arena destabilized a digest"

ATEAR="$(grep -E '^\[arena\] teardown:' "${WORK}/grarena.err" || true)"
[ -n "${ATEAR}" ] || fail "composed arm: no arena teardown line"
ABINDS="$(sed -E 's/.* from ([0-9]+) bind.*/\1/' <<<"${ATEAR}")"
APLACED="$(sed -E 's/.*teardown: ([0-9]+) placement.*/\1/' <<<"${ATEAR}")"
[ "${ABINDS}" -gt 0 ] || fail "composed arm: zero binds"
[ "${ABINDS}" = "${APLACED}" ] || fail "composed arm reconciliation: ${APLACED}/${ABINDS}"
grep -qE ' 0 misplaced' <<<"${ATEAR}" \
  || fail "composed arm: misplaced placements — a captured graph may bake a scratch address"

echo "arena-parity: PASS (streams + checkpoints byte-identical; ${TEARDOWN#\[arena\] }; graphs composed: $(gfield "${ABANNER}" captured) captured, $(gfield "${ABANNER}" replays) replays)"
