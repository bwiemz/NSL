#!/usr/bin/env bash
#
# gpu-test.sh — POSIX runner for the GPU canary manifest (roadmap item 15).
#
# `tools/gpu-test.ps1` is PowerShell-only and pwsh is not installed on the
# reference Linux box, so the canary set could not actually be run there — the
# manifest existed but nothing on this platform could execute it. This is the
# equivalent.
#
# Reads tools/gpu-canary.txt (pipe-delimited: package | binary | test_fn | note),
# runs each entry as its own `cargo test` invocation, and tees per-test logs to
# target/gpu-test-logs/.
#
# One process per test on purpose. A faulting CUDA kernel poisons the context
# for every subsequent test in the same process, which turns one real failure
# into a cascade of fake ones — the harness defect that made the item-19
# certification lane report results it had not produced.
#
# Usage:
#   tools/gpu-test.sh              # run the whole canary set
#   tools/gpu-test.sh --list       # print the manifest, run nothing
#   tools/gpu-test.sh --filter csha  # only entries whose line matches
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

manifest="tools/gpu-canary.txt"
logdir="${CARGO_TARGET_DIR:-target}/gpu-test-logs"
filter=""
list_only=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list_only=1; shift ;;
    --filter) filter="${2:?--filter needs a pattern}"; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[[ -f "${manifest}" ]] || { echo "missing ${manifest}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Device preflight. REFUSE to start rather than report green having run
# nothing: most GPU tests in this tree early-return as a PASS when no device is
# available, so a sweep under those conditions is worse than not sweeping.
# This is the same refusal scripts/gpu-cert.sh makes, for the same reason.
# ---------------------------------------------------------------------------
if [[ "${list_only}" -eq 0 ]]; then
  # `+x` not `:-`. The gates test PRESENCE (`env::var(..).is_ok()`), which is
  # true for the empty string, so `-n "${VAR:-}"` would let
  # `NSL_SKIP_CUDA_TESTS= tools/gpu-test.sh` through — every gate skips, every
  # gate prints "1 passed", and the sweep reports green having run nothing.
  if [[ -n "${NSL_SKIP_CUDA_TESTS+x}" ]]; then
    echo "REFUSING: NSL_SKIP_CUDA_TESTS is set (even to empty) — every gate would early-return as a pass." >&2
    exit 1
  fi
  # Likewise, and this one was outright broken: `${VAR:-unset}` substitutes on
  # unset OR empty, so the comparison against "" could never be true and the
  # check never fired in ANY case. An empty CUDA_VISIBLE_DEVICES hides the
  # device from CUDA (nvidia-smi ignores the variable, so the probes below
  # still pass), every gate early-returns, and the sweep reports 8/8 green
  # having executed zero kernels — precisely the failure this refusal exists
  # to prevent.
  if [[ -n "${CUDA_VISIBLE_DEVICES+x}" && -z "${CUDA_VISIBLE_DEVICES}" ]]; then
    echo "REFUSING: CUDA_VISIBLE_DEVICES is set but empty — no device would be visible to CUDA." >&2
    exit 1
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "REFUSING: nvidia-smi not found — cannot confirm a device is present." >&2
    exit 1
  fi
  if ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    echo "REFUSING: nvidia-smi reports no usable device." >&2
    exit 1
  fi
  gpu="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  echo "device: ${gpu}"
  echo "logs:   ${logdir}"
  mkdir -p "${logdir}"
fi

entries=()
while IFS= read -r line; do
  [[ "${line}" =~ ^[[:space:]]*# ]] && continue
  [[ -z "${line// }" ]] && continue
  [[ -n "${filter}" && "${line}" != *"${filter}"* ]] && continue
  entries+=("${line}")
done < "${manifest}"

if [[ "${#entries[@]}" -eq 0 ]]; then
  echo "no manifest entries matched" >&2
  exit 1
fi

if [[ "${list_only}" -eq 1 ]]; then
  printf '%s\n' "${entries[@]}"
  echo
  echo "${#entries[@]} entries"
  exit 0
fi

pass=0
fail=0
failed_names=()

for line in "${entries[@]}"; do
  IFS='|' read -r pkg bin fn _note <<< "${line}"
  pkg="$(echo "${pkg}" | xargs)"
  bin="$(echo "${bin}" | xargs)"
  fn="$(echo "${fn}" | xargs)"
  name="${pkg}/${bin}/${fn}"
  log="${logdir}/${pkg}__${bin}__${fn}.log"

  # Feature set is per-package. `nsl-test/cuda` only exists for packages that
  # depend on nsl-test; passing it to nsl-runtime is a hard cargo error
  # ("the package 'nsl-runtime' does not contain this feature"), which is how
  # the first version of this runner reported two false FAILs.
  # `test-hooks` is needed too: nsl-runtime's integration tests reach the
  # tensor builders/readers (`test_build_tensor_2d_f32`, `test_read_tensor_f64`)
  # through it, so a `--features cuda` run compiles those files to nothing and
  # the filter matches no test. That reads as NOTFOUND here rather than a false
  # green only because of the summary-line parse below.
  case "${pkg}" in
    nsl-runtime) feats="cuda,test-hooks" ;;
    *)           feats="cuda,nsl-test/cuda" ;;
  esac

  printf '%-72s ' "${name}"
  if cargo test -p "${pkg}" --features "${feats}" --test "${bin}" \
       -- --ignored --exact "${fn}" --nocapture --test-threads=1 \
       > "${log}" 2>&1; then
    # `cargo test` exits 0 when a filter matches NOTHING, so a renamed or
    # deleted test would otherwise read as a pass — the failure mode that let
    # the item-19 lane report green having executed nothing.
    #
    # Parse the SUMMARY line, not the per-test line: under --nocapture a
    # test's own stdout is interleaved on the same line, so `... ok$` does not
    # match even for a test that passed. That mis-detection made this runner
    # report NOTFOUND for a test that had just run green.
    if grep -qE '^test result: ok\. [1-9][0-9]* passed' "${log}"; then
      echo "PASS"
      pass=$((pass + 1))
    else
      echo "NOTFOUND (filter matched no test)"
      fail=$((fail + 1))
      failed_names+=("${name} [NOTFOUND]")
    fi
  else
    echo "FAIL"
    fail=$((fail + 1))
    failed_names+=("${name}")
  fi
done

echo
echo "${pass} passed, ${fail} failed, ${#entries[@]} total"
if [[ "${fail}" -ne 0 ]]; then
  printf '  %s\n' "${failed_names[@]}"
  echo "logs under ${logdir}" >&2
  exit 1
fi
