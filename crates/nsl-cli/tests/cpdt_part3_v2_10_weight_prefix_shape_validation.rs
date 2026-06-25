//! CPDT Part III v2.10 end-to-end CLI gate:
//! `@moe(weight_prefix="…")` malformed prefixes (leading-dot, trailing-
//! dot, consecutive dots, whitespace) refuse loudly at decorator-parse
//! time so the build user sees the diagnostic at the source line of
//! the typo — NOT 30 frames downstream at a `derive_v4_dims` failure.
//!
//! The empty-string refusal is already pinned by v2.7's CLI gate
//! (`weight_prefix_rejects_empty_string`). This file pins the v2.10
//! extension that adds 4 more refusal kinds via `validate_weight_prefix`
//! in `crates/nsl-codegen/src/moe.rs`.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Build a fixture with a malformed `weight_prefix` and assert the
/// build fails with a stderr substring matching the expected refusal
/// message AND a stderr substring containing the verbatim prefix —
/// pinning both the refusal kind AND the "echo the offending input"
/// contract per v2.10 fix F3 (LOW adversarial review).
fn assert_weight_prefix_refusal(
    prefix_literal: &str,
    expected_kind_substr: &str,
    expected_echo_substr: &str,
) {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("malformed_prefix.nsl");
    fs::write(
        &src,
        format!(
            r#"model Block:
    @moe(num_experts=2, top_k=1, capacity_factor=2.0, weight_prefix={prefix_literal})
    experts_dummy: int = 2
    router: Tensor = ones([4, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = Block()
let x = ones([2, 4])
let y = m.forward(x)
"#
        ),
    )
    .unwrap();
    let out = tmp.path().join("malformed_prefix.o");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build").arg(&src).arg("--emit-obj").arg("-o").arg(&out);

    cmd.assert().failure().stderr(
        predicate::str::contains(expected_kind_substr)
            .and(predicate::str::contains(expected_echo_substr)),
    );
}

#[test]
fn weight_prefix_leading_dot_refused_with_pointed_message() {
    // ".hf.layer" composes to `..router.weight` — never a real key.
    // Refuse at the source line, not 30 frames downstream.
    assert_weight_prefix_refusal(r#"".hf.layer""#, "cannot start with '.'", ".hf.layer");
}

#[test]
fn weight_prefix_trailing_dot_refused_with_pointed_message() {
    // "hf.layer." composes to `hf.layer..router.weight` (empty segment
    // before the suffix). Refuse loudly.
    assert_weight_prefix_refusal(r#""hf.layer.""#, "cannot end with '.'", "hf.layer.");
}

#[test]
fn weight_prefix_consecutive_dots_refused_with_pointed_message() {
    // "model..layers" has an empty segment between two dots. Refuse
    // even though no single starts/ends-with check fires.
    assert_weight_prefix_refusal(r#""model..layers""#, "consecutive dots", "model..layers");
}

#[test]
fn weight_prefix_whitespace_refused_with_pointed_message() {
    // Stray space in the prefix — almost always a config-file copy-
    // paste artifact. The refusal must call out whitespace specifically
    // rather than hide it inside a generic "not found" message.
    assert_weight_prefix_refusal(r#""model. layers""#, "whitespace", "model. layers");
}
