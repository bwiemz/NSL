//! CSHA cycle 19 T1 — grep-hygiene test for the backward FFI symbol.
//!
//! Enforces that every direct-call use of `nsl_flash_attention_csha_backward(`
//! across the workspace lives in ONE of three approved locations:
//!
//!   (a) the runtime definition file itself
//!       (`crates/nsl-runtime/src/flash_attention.rs`);
//!   (b) an existing pre-c19 caller test whose path is on the allow-list
//!       below (the 8 tests + 1 codegen builtin snapshot that pre-date
//!       cycle 19 and legitimately name the symbol);
//!   (c) a companion `nsl_flash_attention_csha_backward_probe(` reference
//!       within 100 lines (variant-B call-site pairing).
//!
//! This prevents accidental new callers of the FROZEN 54-param original
//! from sneaking in past T1 — future backward call sites should use the
//! probe variant so the 8-slot dS/dV probe pointers thread through.
//!
//! Doc/markdown files are excluded (they may legitimately reference the
//! symbol name in prose).

use std::fs;
use std::path::{Path, PathBuf};

const SYMBOL: &str = "nsl_flash_attention_csha_backward";
const PROBE_SYMBOL: &str = "nsl_flash_attention_csha_backward_probe";

/// Files that legitimately name `nsl_flash_attention_csha_backward(` and
/// pre-date the cycle-19 hygiene gate. Every path is workspace-relative
/// (matched by suffix so backslashes vs forward-slashes don't matter on
/// Windows).
const ALLOW_LIST: &[&str] = &[
    // Runtime definition file (the symbol lives here).
    "crates/nsl-runtime/src/flash_attention.rs",
    // Cranelift extern-decl registry.
    "crates/nsl-codegen/src/builtins.rs",
    // AD lowerer emits the call.
    "crates/nsl-codegen/src/wengert_lower.rs",
    // Runtime-side typed-coercion sentinel (pre-c19).
    "crates/nsl-runtime/tests/pca_rope_ffi_sentinel.rs",
    // Codegen-side arity assertion (pre-c19).
    "crates/nsl-codegen/tests/pca_rope_ffi_decls.rs",
    // Existing e2e caller tests.
    "crates/nsl-codegen/tests/csha_cycle15_bug1_ablations.rs",
    "crates/nsl-codegen/tests/csha_checkpoint_recompute_gpu.rs",
    "crates/nsl-codegen/tests/csha_cuda_backward.rs",
    "crates/nsl-codegen/tests/csha_cuda_launch_fused.rs",
    "crates/nsl-codegen/tests/csha_dx_norm_readback_diag.rs",
    "crates/nsl-codegen/tests/pca_tier_a_backward_correctness.rs",
    "crates/nsl-codegen/tests/tier_b2_full_backward_cpu_reference.rs",
    // NOTE: Files without any `nsl_flash_attention_csha_backward(` matches
    // are auto-skipped by the scanner (see `contents.contains(&call_marker)`
    // guard below). Allow-list entries should therefore only include files
    // that *actually contain* the symbol — a stale entry here is harmless
    // but misleading. `pca_backward_kernel_snapshot.rs` was previously
    // listed but contains 0 matches; it has been removed for hygiene.
    // c19 hygiene test itself (mentions the symbol in string constants).
    "crates/nsl-codegen/tests/csha_backward_ffi_hygiene.rs",
];

fn workspace_root() -> PathBuf {
    // `CARGO_MANIFEST_DIR` for this crate is `.../crates/nsl-codegen`; the
    // workspace root sits two levels up.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    Path::new(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn is_allowed(path: &Path) -> bool {
    let s = path.to_string_lossy().replace('\\', "/");
    ALLOW_LIST.iter().any(|allowed| s.ends_with(allowed))
}

fn is_source_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|s| s.to_str()),
        Some("rs")
    )
}

fn walk_dir(root: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(root) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        // Skip build artifacts, git internals, worktree scratch.
        if matches!(name, "target" | ".git" | ".worktrees" | "node_modules") {
            continue;
        }
        if path.is_dir() {
            walk_dir(&path, out);
        } else if is_source_file(&path) {
            out.push(path);
        }
    }
}

/// Returns true if any occurrence of `SYMBOL(` in `contents` has a
/// companion `PROBE_SYMBOL(` occurrence within 100 lines (either
/// direction). This is the pair-with-probe escape hatch.
fn every_call_pairs_with_probe(contents: &str) -> bool {
    let lines: Vec<&str> = contents.lines().collect();
    let call_re = format!("{}(", SYMBOL);
    let probe_re = format!("{}(", PROBE_SYMBOL);

    let probe_lines: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter_map(|(i, l)| if l.contains(&probe_re) { Some(i) } else { None })
        .collect();

    for (i, line) in lines.iter().enumerate() {
        if !line.contains(&call_re) {
            continue;
        }
        // The line contains `SYMBOL(` but might also be a `PROBE_SYMBOL(`
        // line (SYMBOL is a prefix of PROBE_SYMBOL). If so, it's fine.
        if line.contains(&probe_re) {
            continue;
        }
        // Otherwise require a probe call within +/- 100 lines.
        let paired = probe_lines
            .iter()
            .any(|&pi| pi.abs_diff(i) <= 100);
        if !paired {
            return false;
        }
    }
    true
}

#[test]
fn csha_backward_ffi_calls_confined_to_allowlist_or_paired_with_probe() {
    let root = workspace_root();
    let mut files = Vec::new();
    walk_dir(&root, &mut files);

    let mut violations: Vec<String> = Vec::new();
    let call_marker = format!("{}(", SYMBOL);

    for file in &files {
        let contents = match fs::read_to_string(file) {
            Ok(c) => c,
            Err(_) => continue,
        };
        // Any occurrence of `nsl_flash_attention_csha_backward(` including
        // the `_probe(` variant will match `_backward(` as prefix? No —
        // `_backward_probe(` doesn't contain `_backward(` because there's
        // no `(` right after `_backward`. So this marker only catches the
        // 54-param original.
        if !contents.contains(&call_marker) {
            continue;
        }
        if is_allowed(file) {
            continue;
        }
        if every_call_pairs_with_probe(&contents) {
            continue;
        }
        violations.push(file.strip_prefix(&root).unwrap_or(file).display().to_string());
    }

    assert!(
        violations.is_empty(),
        "cycle-19 hygiene: found unapproved call sites of `{}(` — either \
         add the file to ALLOW_LIST in this test (only for pre-c19 callers) \
         or convert the call to use `{}(` with probe pointers threaded. \
         Offending files:\n  {}",
        SYMBOL,
        PROBE_SYMBOL,
        violations.join("\n  "),
    );
}
