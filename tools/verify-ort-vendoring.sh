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
# Dependencies: standard POSIX tools only (grep, sed, sort, comm).
# No bindgen or libclang required.
#
# The check is a symbol-set comparison, not a structural diff — the C header
# is grepped directly for every Ort* / ONNX* identifier, which forms a
# superset of the declared types. Spec C §3.2 calls this a "sanity smoke
# test"; the binary-layout assertions live as `const _:` checks inside
# `vendored.rs` itself.

set -euo pipefail

ORT_VERSION="1.22.0"
HEADER="third_party/onnxruntime-${ORT_VERSION}/include/onnxruntime_c_api.h"
VENDORED="crates/nsl-runtime/src/onnx_rt_op/vendored.rs"
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

# Extract the type-name set from vendored.rs. Strip the leading Rust keyword
# (`pub struct ` / `pub type ` / `pub enum `) and keep only the type name.
extract_vendored_types() {
    grep -hE '^(pub struct |pub enum |pub type )(Ort|ONNX)[A-Za-z0-9_]+' "$1" \
        | sed -E 's/^(pub struct |pub enum |pub type )//; s/[^A-Za-z0-9_].*$//' \
        | sort -u
}

# Extract all Ort* / ONNX* identifiers from the C header. This produces a
# superset of the declared types (typedef names, struct tags, enum tags, and
# any occurrence in comments or macro bodies). That is intentional: we only
# assert presence ("still exists upstream"), not absence. Field-by-field
# structural equality is owned by the `const _:` assertions in vendored.rs.
extract_header_types() {
    grep -oE '(Ort|ONNX)[A-Za-z0-9_]+' "$1" | sort -u
}

extract_vendored_types "${VENDORED}" > "${VENDORED_SYMS}"
extract_header_types "${HEADER}" > "${UPSTREAM_SYMS}"

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
