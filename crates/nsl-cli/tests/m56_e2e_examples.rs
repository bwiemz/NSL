//! M56 v1 E2E example tests (Task 21).
//!
//! Each test invokes `nsl check --linear-types` on an example file and
//! asserts the expected behavior:
//! - Positive examples: clean check (no M56 error codes).
//! - Negative examples: specific error code in stderr.
//! - v2-pending examples: minimal assertions (parse-only or soft).

use std::path::PathBuf;
use std::process::Command;

/// Returns the workspace root (two levels above CARGO_MANIFEST_DIR, which is
/// `crates/nsl-cli`).
fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run `nsl check --linear-types <path>` and return the full output.
fn run_check(example_name: &str) -> std::process::Output {
    let path = workspace_root().join(format!("examples/{}.nsl", example_name));
    Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["check", "--linear-types"])
        .arg(&path)
        .output()
        .unwrap_or_else(|e| panic!("nsl check spawn failed for '{}': {}", example_name, e))
}

// ── Positive examples ─────────────────────────────────────────────────────────

/// Two-agent linear pipeline — must compile clean with no M56 error codes.
#[test]
fn m56_basic_two_agents_compiles_clean() {
    let out = run_check("m56_basic_two_agents");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Assert no E060X codes (E0601-E0609 + E0610) appear.
    assert!(
        !stderr.contains("E060") && !stderr.contains("E0610"),
        "expected clean check; stderr:\n{}",
        stderr
    );
}

/// @auto_device_transfer opt-in — E0608 must NOT fire, no other M56 errors either.
#[test]
fn m56_device_transfer_opt_in_compiles_clean() {
    let out = run_check("m56_device_transfer_opt_in");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // No M56 agent-memory error codes should fire (E0601–E0609 + E0610).
    assert!(
        !stderr.contains("E060") && !stderr.contains("E0610"),
        "expected clean check; stderr:\n{}",
        stderr
    );
    // Specifically: E0608 must NOT fire — @auto_device_transfer suppresses it.
    // (The above assertion already covers this, but the explicit message aids
    // diagnosis if E0608 reappears after a Task 10 regression.)
    assert!(
        !stderr.contains("E0608"),
        "@auto_device_transfer should suppress E0608; stderr:\n{}",
        stderr
    );
    // Note: v1 may emit an info-level device-transfer note — that is acceptable.
    // We don't assert its presence here; Task 10's unit tests cover the note emission.
}

// ── Negative examples ─────────────────────────────────────────────────────────

/// Cross-agent exclusive field access — must produce E0601.
#[test]
fn m56_cross_agent_access_error_emits_e0601() {
    let out = run_check("m56_cross_agent_access_error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("E0601"),
        "expected E0601 cross-agent exclusive field access; stderr:\n{}",
        stderr
    );
}

/// Circular agent topology — must produce E0603.
#[test]
fn m56_cycle_error_emits_e0603() {
    let out = run_check("m56_cycle_error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("E0603"),
        "expected E0603 circular port topology; stderr:\n{}",
        stderr
    );
}

/// Cross-GPU port connection — must produce E0607.
#[test]
fn m56_cross_gpu_error_emits_e0607() {
    let out = run_check("m56_cross_gpu_error");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("E0607"),
        "expected E0607 cross-GPU port connection; stderr:\n{}",
        stderr
    );
}

// ── v2-pending examples ───────────────────────────────────────────────────────

/// serve block with pool_size=4 — v2-pending. The serve block is commented
/// out in v1; only the agent + @pipeline_agent declarations are active.
/// Assert the file doesn't crash the checker (parse-level sanity only).
#[test]
fn m56_serve_pool_parses_clean_v2_pending() {
    let out = run_check("m56_serve_pool");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Best-effort: the file should at least not crash nsl check.
    // With the serve block commented out, no fatal parse errors are expected.
    // Any remaining errors from other type-system passes are acceptable per
    // v2-pending status. Assert at minimum that the process did not panic
    // (exit code 101 = Rust panic/unwrap in the nsl binary).
    assert!(
        out.status.code() != Some(101),
        "nsl check panicked on m56_serve_pool (exit 101); stderr:\n{}",
        stderr
    );
}

/// Three-agent speculative decoding pipeline — spec §1.4 motivating example.
/// Must compile clean with no M56 error codes.
#[test]
fn m56_speculative_decoding_compiles_clean() {
    let out = run_check("m56_speculative_decoding");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // Three-agent linear pipeline — no E060x codes should fire.
    assert!(!stderr.contains("E060") && !stderr.contains("E0610"),
        "expected clean check for speculative decoding pipeline; stderr:\n{}",
        stderr);
}

/// @shared embedding table — v2-pending runtime, but semantic check (Task 8)
/// must already exempt @shared fields from E0601.
#[test]
fn m56_shared_embeddings_shared_field_exempt_from_e0601() {
    let out = run_check("m56_shared_embeddings");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The @shared field must NOT produce E0601 — Task 8 semantic check
    // recognises @shared and exempts it from cross-agent exclusive field access.
    assert!(
        !stderr.contains("E0601"),
        "@shared field should exempt cross-agent access from E0601; stderr:\n{}",
        stderr
    );
    // Other errors may still fire (e.g. runtime/codegen gaps for @shared are
    // v2-pending) — that is acceptable.
}
