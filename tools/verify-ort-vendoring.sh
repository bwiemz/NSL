#!/usr/bin/env bash
#
# verify-ort-vendoring.sh — sanity check that the vendored ORT struct
# definitions in `crates/nsl-runtime/src/onnx_rt_op/vendored.rs` track the
# upstream ORT 1.22.x C API. Run by CI in the `test-onnx-rt` job.
#
# Manual invocation:
#   bash tools/verify-ort-vendoring.sh
#
# CI failure semantics: a non-zero exit means the vendored definitions are
# missing types or function names that exist upstream. Run the script
# locally, examine the diff, and update `vendored.rs` (adding missing types
# as opaque placeholders if Spec C's scope doesn't need their fields typed).
#
# Dependencies: `bindgen` (the CLI). Install with:
#   cargo install bindgen-cli
#
# The check is a symbol-set comparison, not a structural diff — bindgen's
# emitted Rust differs from our hand-rolled layout, but the set of declared
# `Ort*` / `ONNX*` types should overlap. Spec C §3.2 calls this a "sanity
# smoke test"; the binary-layout assertions live as `const _:` checks inside
# `vendored.rs` itself.

set -euo pipefail

ORT_VERSION="1.22.0"
HEADER="third_party/onnxruntime-${ORT_VERSION}/include/onnxruntime_c_api.h"
VENDORED="crates/nsl-runtime/src/onnx_rt_op/vendored.rs"
GENERATED="${TMPDIR:-/tmp}/ort-bindgen-generated.rs"
BINDGEN_ERR="${TMPDIR:-/tmp}/ort-bindgen-stderr.txt"
UPSTREAM_SYMS="${TMPDIR:-/tmp}/ort-upstream-syms.txt"
VENDORED_SYMS="${TMPDIR:-/tmp}/ort-vendored-syms.txt"

if [[ ! -f "${HEADER}" ]]; then
    echo "ERROR: vendored header not found at ${HEADER}"
    echo "Re-vendor ORT 1.22.0 per docs/superpowers/specs/2026-05-18-m62b-onnx-rt-custom-op-design.md §3.1."
    exit 1
fi

if [[ ! -f "${VENDORED}" ]]; then
    echo "ERROR: vendored Rust definitions not found at ${VENDORED}"
    exit 1
fi

if ! command -v bindgen >/dev/null 2>&1; then
    echo "ERROR: bindgen CLI not installed."
    echo "Install with: cargo install bindgen-cli"
    exit 1
fi

# Generate Rust definitions from the upstream header. Allowlist Ort* and
# ONNX* names; ignore standard C library types that bindgen would otherwise
# splat into the output.
#
# Use --formatter=none so bindgen emits raw token output without shelling out
# to rustfmt. Bindgen 0.69+ removed --no-rustfmt-bindings; newer versions
# (0.72+) may emit all declarations on a single line, which is why
# `extract_types` below uses `grep -o` (only-matching) rather than a
# line-anchored `^` pattern.
echo "Running bindgen on ${HEADER}..."
bindgen \
    --allowlist-type 'Ort.*|ONNX.*' \
    --allowlist-function 'Ort.*|ONNX.*' \
    --allowlist-var 'ORT_.*|ONNX_.*' \
    --formatter=none \
    "${HEADER}" \
    > "${GENERATED}" 2>"${BINDGEN_ERR}"

if [[ ! -s "${GENERATED}" ]]; then
    echo "ERROR: bindgen produced no output. Bindgen stderr:"
    cat "${BINDGEN_ERR}"
    exit 1
fi
echo "bindgen output: $(wc -c < "${GENERATED}") bytes in ${GENERATED}"

# Extract the type-name set from both files, restricted to ORT's namespace
# prefix (Ort* / ONNX*). We strip the leading `pub struct ` / `pub type ` /
# `pub enum ` keyword and keep only the type name — this normalizes across
# the stylistic difference between bindgen (which emits `pub type
# OrtErrorCode = c_int;` for C enums) and our hand-rolled file (which uses
# `pub struct OrtErrorCode(pub u32);`). The check is then "does every
# vendored type name still exist upstream in some form?".
#
# Field-by-field structural equality is NOT checked here; the binary-layout
# assertions inside `vendored.rs` (`const _:` blocks) guard that property.
#
# `-o` (only-matching) emits one match per output line so the pattern works
# whether bindgen writes one-declaration-per-line (with rustfmt) or emits
# everything on a single line (--formatter=none / bindgen 0.72+).
extract_types() {
    # shellcheck disable=SC2016
    { grep -ohE '(pub struct |pub enum |pub type )(Ort|ONNX)[A-Za-z0-9_]+' "$1" \
        || true; } \
        | sed -E 's/^(pub struct |pub enum |pub type )//; s/[^A-Za-z0-9_].*$//' \
        | sort -u
}

echo "Extracting upstream symbols from generated output..."
extract_types "${GENERATED}" > "${UPSTREAM_SYMS}"

if [[ ! -s "${UPSTREAM_SYMS}" ]]; then
    echo "ERROR: no Ort*/ONNX* types found in generated output."
    echo "First 200 bytes of generated output:"
    head -c 200 "${GENERATED}"
    echo
    echo "Bindgen stderr:"
    cat "${BINDGEN_ERR}"
    exit 1
fi
echo "Found $(wc -l < "${UPSTREAM_SYMS}") upstream symbols."

echo "Extracting vendored symbols from ${VENDORED}..."
extract_types "${VENDORED}" > "${VENDORED_SYMS}"

# Spec C only vendors a subset of upstream types (Ort{Session,Status,...},
# OrtApi, OrtApiBase, OrtCustomOp, OrtKernelContext, etc.). We assert that
# every symbol we DID vendor still exists upstream, which catches deletions
# and renames. New upstream additions that we haven't vendored are OK —
# they're caught only if Spec C grows to need them.
echo "Checking that all vendored symbols still exist upstream..."
MISSING=$(comm -23 "${VENDORED_SYMS}" "${UPSTREAM_SYMS}" || true)
if [[ -n "${MISSING}" ]]; then
    echo "DRIFT: the following vendored symbols are NOT present in upstream ORT ${ORT_VERSION}:"
    echo "${MISSING}"
    echo
    echo "ORT likely renamed or removed these types. Update vendored.rs."
    exit 1
fi

echo "OK — all vendored ORT definitions are present upstream (ORT ${ORT_VERSION})."
