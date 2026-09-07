#!/usr/bin/env bash
# Run the nsl-codegen integration tests that only exist under the
# `test-helpers` feature.
#
# Two kinds of target hide behind that feature:
#
#   1. files under crates/nsl-codegen/tests/ with a crate-level
#      `#![cfg(feature = "test-helpers")]` (and one file with a per-test
#      `#[cfg]`): without the feature they compile to an EMPTY test binary,
#      so `cargo test --workspace` reports them green while running nothing;
#   2. `[[test]]` entries in crates/nsl-codegen/Cargo.toml carrying
#      `required-features = ["test-helpers"]`: cargo skips them outright.
#
# Both lists are derived from the tree here, so a new gated file is picked
# up without editing CI. Extra arguments are passed through to `cargo test`
# (e.g. `-- --nocapture`).
set -euo pipefail
cd "$(dirname "$0")/.."

gated=$(grep -l 'cfg(feature = "test-helpers")' crates/nsl-codegen/tests/*.rs \
  | xargs -n1 basename | sed 's/\.rs$//')
required=$(awk '
  /^\[\[test\]\]/ { name = "" }
  /^name = /      { gsub(/"/, "", $3); name = $3 }
  /^required-features/ && /test-helpers/ && name != "" { print name }
' crates/nsl-codegen/Cargo.toml)

targets=()
for t in $gated $required; do targets+=(--test "$t"); done
echo "gated-tests: ${#targets[@]} nsl-codegen targets under --features test-helpers" >&2
exec cargo test -p nsl-codegen --features test-helpers --no-fail-fast "${targets[@]}" "$@"
