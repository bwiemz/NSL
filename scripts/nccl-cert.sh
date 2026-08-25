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
#     ws==1 short-circuit. Proves our calls into NCCL are well-formed and that
#     NCCL honoured them — NOT that an NCCL ring KERNEL ran: at nRanks==1 all
#     three collectives are the identity, so no observable can distinguish a
#     collective kernel from a copy.
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

# Refuse-to-start GPU guard (scripts/gpu-guard.sh): flock against other
# guarded runs + busy-device refusal + own process group. A hard exit, not a
# warning — see the guard's header for the incident that made this policy.
if [[ "${NSL_GPU_GUARD:-1}" != "0" ]] && ! "${ROOT}/scripts/gpu-guard.sh" held; then
    exec "${ROOT}/scripts/gpu-guard.sh" run -- "${ROOT}/scripts/nccl-cert.sh" "$@"
fi

# DEDICATED, and deliberately NOT `${CARGO_TARGET_DIR:-...}`. This build sets
# RUSTFLAGS, so sharing a target dir thrashes fingerprints — and worse, any
# concurrent `cargo test --workspace` (which builds WITHOUT --features cuda)
# replaces release/libnsl_runtime.a with a CUDA-less archive. `nsl` embeds that
# archive into every program it compiles, and linker.rs only probes for ONE
# cuda marker symbol, so the emitted binary silently links a runtime with no
# CUDA and aborts in nsl_tensor_to_device. Observed during this script's own
# development. Override with NSL_NCCL_TARGET_DIR if you really mean to.
CARGO_TARGET_DIR="${NSL_NCCL_TARGET_DIR:-${HOME}/.cache/nsl-target-nccl}"
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
        # Both names matter: cudarc dlopens libnccl.so.2 at runtime, but
        # linker.rs passes -lnccl when linking the COMPILED PROGRAM (tier 2),
        # and `-lnccl` resolves only against the unversioned libnccl.so.
        if [[ -e "${d}/libnccl.so.2" ]]; then
            if [[ ! -e "${d}/libnccl.so" ]]; then
                echo "nccl-cert: ${d} has libnccl.so.2 but no unversioned" >&2
                echo "  libnccl.so — tier 2's \`-lnccl\` link will fail." >&2
                echo "  Fix: ln -s libnccl.so.2 ${d}/libnccl.so" >&2
            fi
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
# No trailing colon: an EMPTY element in LD_LIBRARY_PATH means "the current
# directory" to the loader, and this script has already cd'd to the repo root.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH="${NCCL_DIR}:${LD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="${NCCL_DIR}"
fi
export RUSTFLAGS="-L ${NCCL_DIR} ${RUSTFLAGS:-}"
export CUDA_PATH="${CUDA_PATH:-/opt/cuda}"
# Fail fast instead of sitting on the 300s defaults when a rank dies.
export NSL_NCCL_TIMEOUT_SECS="${NSL_NCCL_TIMEOUT_SECS:-60}"
export NSL_TP_BARRIER_TIMEOUT_SECS="${NSL_TP_BARRIER_TIMEOUT_SECS:-60}"

# ---------------------------------------------------------------------------
# Device count decides which tiers are reachable
# ---------------------------------------------------------------------------
# `|| true` is load-bearing: under `set -euo pipefail` a failing nvidia-smi
# propagates through the pipeline and kills the script HERE, making the
# "no GPU" verdict below unreachable — the one case it exists to report.
GPUS=0
if command -v nvidia-smi > /dev/null 2>&1; then
    GPUS="$( { nvidia-smi --query-gpu=index --format=csv,noheader || true; } | grep -c . || true)"
fi
[[ "${GPUS}" =~ ^[0-9]+$ ]] || GPUS=0
echo "nccl-cert: ${GPUS} GPU(s) visible"
if [[ "${GPUS}" -lt 1 ]]; then
    echo "nccl-cert: no GPU — NCCL NOT CERTIFIED (nothing ran)" >&2
    exit 1
fi

FAILED=0

# Exit status alone is NOT enough: `cargo test <filter>` exits 0 when the
# filter matches NOTHING ("0 passed; 0 failed; N filtered out"). A tier that
# ran zero tests would then be indistinguishable from a tier that passed —
# the exact vacuity this whole script exists to eliminate. So require a
# positive pass count as well.
run_tier() {
    local name="$1"
    shift
    echo ""
    echo "=== ${name} ==="
    local log rc passed
    log="$(mktemp)"
    rc=0
    "$@" > "${log}" 2>&1 || rc=$?
    cat "${log}"
    # awk, not bc: bc is not installed on this box, and `|| echo 0` would have
    # made EVERY tier report "selected ZERO tests" — a script that always fails
    # is no better than one that always passes.
    passed="$(grep -oE 'test result: ok\. [0-9]+ passed' "${log}" \
        | grep -oE '[0-9]+' | awk '{n += $1} END {print n + 0}')"
    passed="${passed:-0}"
    rm -f "${log}"
    if [[ "${rc}" -ne 0 ]]; then
        echo "--- ${name}: FAIL (exit ${rc})" >&2
        FAILED=1
    elif [[ "${passed}" -eq 0 ]]; then
        echo "--- ${name}: FAIL — the filter selected ZERO tests, so this" >&2
        echo "    tier certified nothing. A green exit code here means the" >&2
        echo "    test names moved, not that NCCL works." >&2
        FAILED=1
    else
        echo "--- ${name}: PASS (${passed} test(s))"
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
echo "  * ncclAllReduce / ncclAllGather / ncclReduceScatter are ACCEPTED by"
echo "    NCCL on our stream and write the expected values over a poisoned"
echo "    receive buffer. At nRanks==1 all three are the identity, so this"
echo "    proves the calls are well-formed, NOT that a ring kernel ran."
echo "  * unmapped dtypes are refused before reaching NCCL"
echo "  * the ws==1 short-circuit really does bypass NCCL (proved with a"
echo "    deliberately NULL communicator, not a real one)"

if [[ "${GPUS}" -ge 2 ]]; then
    echo ""
    echo "This host has >= 2 GPUs, so tier 2 MAY have taken the parity arm of"
    echo "nccl_two_rank_parity_or_documented_refusal rather than the refusal"
    echo "arm. This script does not inspect which arm ran — nvidia-smi ignores"
    echo "CUDA_VISIBLE_DEVICES while the spawned ranks honour it — so read the"
    echo "tier 2 output above before claiming multi-GPU parity."
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
