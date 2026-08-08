#!/usr/bin/env bash
# NCCL transport certification.
#
# WHY THIS IS SEPARATE FROM gpu-cert.sh
# -------------------------------------
# `scripts/gpu-cert.sh` builds with CERT_FEATURES=cuda,test-hooks,test-helpers
# — no `nccl`. Every NCCL gate is therefore compiled out of that lane and
# reported NOTFOUND. That is the correct status, but it means the standard
# certification lane has never certified NCCL and structurally cannot.
#
# Worse, before this script existed the one gate that mentioned NCCL
# (`nccl_two_rank_parity_or_documented_refusal`) PASSED in that lane while
# proving nothing: its failure arm accepted the string "NCCL backend init
# failed", which is exactly what the `#[cfg(not(feature = "nccl"))]` stub
# prints. A build with no libnccl linked at all reported a green "documented
# NCCL refusal".
#
# WHAT THIS CERTIFIES, AND WHAT IT CANNOT
# ---------------------------------------
# Two independent structural facts bound what any single-GPU box can prove:
#
#   1. The runtime skips backend construction entirely at world_size == 1
#      ("NCCL would be a 1-rank identity clique — skip it", zero.rs), so
#      `--devices 1 --collectives nccl` performs ZERO NCCL calls.
#   2. `--devices 2` needs two GPUs: NCCL refuses two ranks on one device
#      with ncclInvalidUsage.
#
# So through the product path, NCCL is unreachable on one GPU. This script
# takes the third route for the ws=1 tier: it drives the real NcclBackend
# directly from in-crate tests, which DO enter libnccl.
#
#   TIER 1 (any GPU + libnccl): comm init/destroy, real ncclAllReduce /
#     ncclAllGather / ncclReduceScatter on device buffers verified against a
#     poisoned receive buffer, our dtype-mapping refusals, and the documented
#     ws==1 short-circuit. Proves NCCL device code runs on this GPU.
#   TIER 2 (1 GPU): the same-device refusal is loud, symmetric across ranks,
#     fast, and carries the real ncclInvalidUsage code.
#   TIER 3 (>= 2 GPUs): full ZeRO parity under the real transport. NOT
#     RUNNABLE on a single-GPU box — reported UNCERTIFIED, never silently
#     skipped.
#
# Tier 3 is where the properties that actually need multiple devices live:
# inter-rank transport, and NCCL's ring reduction ORDER, which
# `reduce_scatter_sum` documents as per-backend and explicitly NOT the
# SimulatedBackend's fixed rank order. No amount of single-GPU testing
# settles those.
#
# Usage:
#   scripts/nccl-cert.sh                 # build + run every runnable tier
#   NSL_NCCL_LIB_DIR=/path/to/lib scripts/nccl-cert.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-${HOME}/.cache/nsl-target-nccl}"
export CARGO_TARGET_DIR

# ---------------------------------------------------------------------------
# Locate libnccl
# ---------------------------------------------------------------------------
find_nccl_dir() {
    if [[ -n "${NSL_NCCL_LIB_DIR:-}" ]]; then
        printf '%s' "${NSL_NCCL_LIB_DIR}"
        return
    fi
    local d
    for d in /usr/lib /usr/lib64 /usr/lib/x86_64-linux-gnu \
             "${CUDA_PATH:-/opt/cuda}/lib64" "${HOME}/Projects/nccl-local/usr/lib"; do
        if [[ -e "${d}/libnccl.so.2" ]]; then
            printf '%s' "${d}"
            return
        fi
    done
    printf ''
}

NCCL_DIR="$(find_nccl_dir)"
if [[ -z "${NCCL_DIR}" ]]; then
    cat >&2 <<'EOF'
nccl-cert: no libnccl.so.2 found.

NCCL is NOT CERTIFIED by this run — nothing was tested. This is a hard error
rather than a skip on purpose: a silent skip is how the previous gate came to
report green without ever linking NCCL.

Set NSL_NCCL_LIB_DIR to a directory containing libnccl.so.2 and re-run.
EOF
    exit 1
fi
echo "nccl-cert: libnccl at ${NCCL_DIR}"

export NSL_NCCL_LIB_DIR="${NCCL_DIR}"
export LD_LIBRARY_PATH="${NCCL_DIR}:${LD_LIBRARY_PATH:-}"
export RUSTFLAGS="-L ${NCCL_DIR} ${RUSTFLAGS:-}"
export CUDA_PATH="${CUDA_PATH:-/opt/cuda}"
# Fail fast instead of sitting on the 300s defaults when a rank dies.
export NSL_NCCL_TIMEOUT_SECS="${NSL_NCCL_TIMEOUT_SECS:-60}"
export NSL_TP_BARRIER_TIMEOUT_SECS="${NSL_TP_BARRIER_TIMEOUT_SECS:-60}"

# ---------------------------------------------------------------------------
# Device count decides which tiers are reachable
# ---------------------------------------------------------------------------
GPUS=0
if command -v nvidia-smi > /dev/null 2>&1; then
    GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
fi
echo "nccl-cert: ${GPUS} GPU(s) visible"
if [[ "${GPUS}" -lt 1 ]]; then
    echo "nccl-cert: no GPU — NCCL NOT CERTIFIED (nothing ran)" >&2
    exit 1
fi

FAILED=0
run_tier() {
    local name="$1"
    shift
    echo ""
    echo "=== ${name} ==="
    if "$@"; then
        echo "--- ${name}: PASS"
    else
        echo "--- ${name}: FAIL" >&2
        FAILED=1
    fi
}

# ---------------------------------------------------------------------------
# Tier 1 — real libnccl, ws=1, driven directly
# ---------------------------------------------------------------------------
run_tier "TIER 1  real NCCL collectives on this GPU (ws=1)" \
    cargo test -p nsl-runtime --lib --features nccl --release nccl_cert \
        -- --ignored --test-threads=1

# ---------------------------------------------------------------------------
# Tier 2 — the same-device refusal, through the product path
# ---------------------------------------------------------------------------
run_tier "TIER 2  ws=2 behaviour through the product path" \
    cargo test -p nsl-cli --features nccl --release \
        --test zero_gpu_collectives_gate \
        -- --ignored --exact nccl_two_rank_parity_or_documented_refusal \
           --test-threads=1

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
echo ""
echo "==================== NCCL CERTIFICATION VERDICT ===================="
if [[ "${FAILED}" -ne 0 ]]; then
    echo "RESULT: FAILED — see the tier output above."
    exit 1
fi

echo "CERTIFIED on this host (${GPUS} GPU(s)):"
echo "  * libnccl links and ncclCommInitRank/ncclCommDestroy succeed"
echo "  * ncclAllReduce / ncclAllGather / ncclReduceScatter execute on this"
echo "    GPU and write the expected values over a poisoned receive buffer"
echo "  * unmapped dtypes are refused before reaching NCCL"
echo "  * the ws==1 short-circuit really does bypass NCCL"

if [[ "${GPUS}" -ge 2 ]]; then
    echo ""
    echo "TIER 3 (multi-GPU) ran: ZeRO parity under the real transport is"
    echo "covered by the ws=2 arm of nccl_two_rank_parity_or_documented_refusal."
else
    cat <<'EOF'

NOT CERTIFIED — needs >= 2 physical GPUs, and this host has 1:
  * inter-rank NCCL transport (any real data movement between ranks)
  * NCCL ring reduction ORDER, which reduce_scatter_sum documents as
    per-backend and NOT the SimulatedBackend's fixed rank order — every
    ZeRO bit-exactness gate in the tree is sim/sim-gpu-shaped and does not
    transfer to NCCL
  * ZeRO-1/2/3 parity numbers under --collectives nccl
On this host NCCL correctly refuses 2 ranks on 1 device (ncclInvalidUsage),
which Tier 2 asserts. That refusal is the certified behaviour here; it is not
a substitute for the three items above.
EOF
fi
echo "===================================================================="
