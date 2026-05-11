#!/usr/bin/env bash
# Fetch + cache + verify the pinned BitNet b1.58 3B checkpoint.
#
# Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §6.2.
# Reads model_id + revision SHA from tests/fixtures/bitnet_b158_3b_revision.txt
# and SHA-256s from tests/fixtures/bitnet_b158_3b_sha256.txt. Downloads each
# file via the HF Hub resolve API at the pinned revision (immutable per IR-002).
#
# Linux/macOS only per spec §7.3. Windows support is a follow-on.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REVISION_FILE="${REPO_ROOT}/tests/fixtures/bitnet_b158_3b_revision.txt"
SHA256_FILE="${REPO_ROOT}/tests/fixtures/bitnet_b158_3b_sha256.txt"

if [[ ! -f "${REVISION_FILE}" ]]; then
    echo "FAIL: ${REVISION_FILE} not found." >&2
    exit 1
fi
if [[ ! -f "${SHA256_FILE}" ]]; then
    echo "FAIL: ${SHA256_FILE} not found." >&2
    exit 1
fi

MODEL_ID=$(grep '^model_id=' "${REVISION_FILE}" | cut -d= -f2)
REVISION_SHA=$(grep '^revision=' "${REVISION_FILE}" | cut -d= -f2)

if [[ -z "${MODEL_ID}" ]] || [[ -z "${REVISION_SHA}" ]]; then
    echo "FAIL: model_id or revision missing in ${REVISION_FILE}" >&2
    exit 1
fi

CACHE_DIR="${HOME}/.cache/nsl-tests/bitnet_b158_3b"
mkdir -p "${CACHE_DIR}"

echo "Fetching ${MODEL_ID} @ ${REVISION_SHA} to ${CACHE_DIR}"

# Parse the SHA-256 file (lines starting with # are comments).
# Each non-comment line is: <sha256>  <filename>
grep -v '^#' "${SHA256_FILE}" | grep -v '^[[:space:]]*$' | while IFS=' ' read -r EXPECTED_SHA FILENAME; do
    # Strip leading whitespace from filename (typical sha256sum output has 2 spaces).
    FILENAME=$(echo "${FILENAME}" | sed 's/^[[:space:]]*//')
    CACHED_FILE="${CACHE_DIR}/${FILENAME}"

    if [[ -f "${CACHED_FILE}" ]]; then
        ACTUAL_SHA=$(sha256sum "${CACHED_FILE}" | cut -d' ' -f1)
        if [[ "${ACTUAL_SHA}" == "${EXPECTED_SHA}" ]]; then
            echo "  ${FILENAME}: cached, SHA-256 OK"
            continue
        fi
        echo "  ${FILENAME}: cached but SHA-256 mismatch; re-downloading"
        rm -f "${CACHED_FILE}"
    fi

    URL="https://huggingface.co/${MODEL_ID}/resolve/${REVISION_SHA}/${FILENAME}"
    echo "  Downloading ${FILENAME} from ${URL}"
    curl -L -f --progress-bar -o "${CACHED_FILE}.tmp" "${URL}"
    mv "${CACHED_FILE}.tmp" "${CACHED_FILE}"

    ACTUAL_SHA=$(sha256sum "${CACHED_FILE}" | cut -d' ' -f1)
    if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
        echo "FAIL: ${FILENAME} SHA-256 mismatch" >&2
        echo "  expected: ${EXPECTED_SHA}" >&2
        echo "  actual:   ${ACTUAL_SHA}" >&2
        exit 1
    fi
    echo "  ${FILENAME}: downloaded, SHA-256 OK"
done

echo "PASS: all pinned files validated under ${CACHE_DIR}"
