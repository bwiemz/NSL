//! CEP G6.1.2 — source-AD engagement on nested indexed-array access.
//!
//! Paper §3.2 (Dead Gradient Elimination for Pruned Training) claims that
//! WRGA's compile-time backward eliminator "activates automatically" on
//! recovery FT. PR #235 shipped the chain-of-builds (G6 — SP1+SP2
//! artifacts feed back into `nsl build`); PR #242 shipped the semantic
//! half of G6.1 (subscript on `[Block; N]` resolves to Model element
//! type). This PR (G6.1.2) closes the source-AD half:
//!
//! - Source-AD now ENGAGES on `self.blocks[i].attn.wq`-style chains
//!   instead of falling back to tape-AD (Discriminant(11) Subscript bail).
//! - Trainable params that ARE touched by the forward get classified as
//!   Params with compound names matching the runtime tensor-path table
//!   (`m.blocks.0.attn.wq`, etc.), so the gradient-collection scan at
//!   `stmt.rs::~line 4856` matches them and reports them as connected.
//!
//! What the fixture pins:
//! `cep_canonical_with_train.nsl` has 2 blocks × (1 attn_norm.w + 4 attn
//! tensors + 1 ffn_norm.w + 3 ffn tensors) + (embed + norm.w) = 20
//! trainable tensor params. The toy forward TOUCHES wq, wo, w_gate, w_up,
//! w_down per block × 2 blocks = 10 tensors. The other 10 (wk, wv,
//! attn_norm.w, ffn_norm.w per block + embed + norm.w) are NOT used by
//! this fixture's forward; source-AD must correctly mark them as
//! "missing-from-forward" — that's the expected count, not a bug.
//!
//! The PRIOR regression to avoid: my first partial fix attempt
//! (subscript prefix resolution alone, without the field-type-aware
//! Param classification + recursive context pre-registration) made
//! source-AD ENGAGE but report `0/20 trainable params connected` because
//! every nested compound was registered as a Param including the array
//! itself. That's strictly worse than tape-AD fallback; this test pins
//! that we report >= 10 connected (the forward-touched count).
//!
//! This test deliberately does NOT pin EXACTLY 10/20 — if a future
//! source-AD improvement also connects backward-only param edges, the
//! count could legitimately grow. We pin a LOWER BOUND on what
//! MUST connect to prove G6.1.2 closed, not an upper bound on what we
//! happen to connect today.

use assert_cmd::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

fn canonical_with_train_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_canonical_with_train.nsl")
}

/// G6.1.2 — paper §3.2 closure: source-AD engages and connects on
/// `self.blocks[i].attn.wq`-style chains.  Pins:
///
/// 1. The diagnostic `[nsl] Using source-to-source AD for backward pass`
///    appears (proves source-AD engaged at all).
/// 2. The diagnostic `[nsl] source AD extraction failed` does NOT appear
///    (proves source-AD did not bail to tape-AD via the Discriminant
///    error path that surfaced before this fix).
/// 3. The diagnostic `source AD gradient summary: X/Y trainable tensor
///    params connected` appears with X >= 10 (the count of forward-
///    touched tensors in the toy fixture) and Y == 20 (total trainable
///    tensor params under TinyCoder + 2 × TransformerBlock).
///
/// The build itself must succeed end-to-end — both an `sgd_*.o` and a
/// model `.o` must be written.
#[test]
fn source_ad_engages_on_nested_indexed_array_field_access() {
    let temp = tempfile::TempDir::new().unwrap();
    let out = temp.path().join("g6_1_2_artifact");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_with_train_fixture())
        .arg("--source-ad")
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out);

    let output = cmd.output().expect("nsl bin executes");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = format!("{stderr}{stdout}");

    assert!(
        output.status.success(),
        "build must succeed with --source-ad on nested indexed-array fixture; \
         stderr:\n{stderr}\nstdout:\n{stdout}"
    );

    // (1) source-AD engagement: explicit "Using source-to-source AD" line.
    assert!(
        combined.contains("Using source-to-source AD"),
        "expected source-AD engagement on the G6 fixture — `[nsl] Using \
         source-to-source AD for backward pass` not found.\nCombined output:\n{combined}"
    );

    // (2) source-AD did NOT bail to tape-AD. The pre-fix path printed
    // `[source-ad] VarDecl 'q0' extraction failed at expr Discriminant(11)`
    // (BinaryOp on `x @ self.blocks[0].attn.wq` failed because the RHS
    // chain wasn't extractable) AND `[nsl] source AD extraction failed,
    // falling back to tape-based AD`. Both must be absent.
    assert!(
        !combined.contains("source AD extraction failed"),
        "source-AD bailed to tape-AD — the G6.1.2 indexed-array chain is \
         not extracting cleanly. Combined output:\n{combined}"
    );
    assert!(
        !combined.contains("extraction failed at expr Discriminant"),
        "source-AD hit an unhandled Discriminant — bail trap surfaced. \
         Combined output:\n{combined}"
    );

    // (3) Gradient-summary line: parse "X/Y trainable tensor params connected".
    let summary_line = combined
        .lines()
        .find(|l| l.contains("source AD gradient summary"))
        .unwrap_or_else(|| {
            panic!(
                "expected `source AD gradient summary: X/Y trainable tensor params connected` \
                 in the diagnostic output; not found. Combined output:\n{combined}"
            )
        });
    // Format: "[nsl] source AD gradient summary: X/Y trainable tensor params connected, ..."
    // Find the X/Y fragment robustly.
    let (connected, total) = parse_connected_over_total(summary_line).unwrap_or_else(|| {
        panic!("could not parse `X/Y` from summary line: {summary_line:?}")
    });

    // Y must be 20 — pins the model topology hasn't drifted under us.
    assert_eq!(
        total, 20,
        "expected 20 total trainable tensor params (TinyCoder embed + norm.w + \
         2 blocks × 9 tensors), got {total}. Summary line: {summary_line:?}"
    );
    // X must be >= 10 — proves the forward-touched chain (wq, wo,
    // w_gate, w_up, w_down per block × 2) connected. PRIOR PARTIAL-FIX
    // REGRESSION reported X = 0 here. The lower-bound (not exact-10)
    // leaves room for future source-AD improvements that also connect
    // unused params via backward-only edges without breaking this test.
    assert!(
        connected >= 10,
        "expected at least 10 trainable params connected (forward touches \
         5 tensors per block × 2 blocks); got {connected}/{total}. \
         Prior partial-fix regression reported 0/20 — this is the G6.1.2 \
         silent grad-miss trap class. Summary line: {summary_line:?}"
    );
}

/// Robustly parse `"X/Y trainable tensor params connected"` out of the
/// summary line.  Returns `(connected, total)` if the line contains a
/// well-formed `N/M` fragment immediately followed by ` trainable`.
fn parse_connected_over_total(line: &str) -> Option<(usize, usize)> {
    // Find the substring ending in " trainable"; everything before it
    // is the leading text plus the "X/Y" fragment we want.
    let trainable_idx = line.find(" trainable")?;
    let before = &line[..trainable_idx];
    // The X/Y fragment is the LAST whitespace-separated token of `before`.
    let token = before.split_whitespace().last()?;
    let (x_str, y_str) = token.split_once('/')?;
    let x: usize = x_str.trim().parse().ok()?;
    let y: usize = y_str.trim().parse().ok()?;
    Some((x, y))
}

#[test]
fn summary_line_parser_handles_canonical_format() {
    // Pin the parser against the actual diagnostic format from
    // crates/nsl-codegen/src/stmt.rs ~line 5116.
    let line = "[nsl] source AD gradient summary: 10/20 trainable tensor params \
                connected, 10 missing-from-forward, 0 no-primal, 0 no-adjoint, \
                0 cascade-skip, 0 ignored config-tensor, 0 ignored non-tensor";
    let (connected, total) = parse_connected_over_total(line).expect("parser accepts canonical");
    assert_eq!(connected, 10);
    assert_eq!(total, 20);
}

#[test]
fn summary_line_parser_rejects_malformed() {
    assert!(parse_connected_over_total("nothing useful").is_none());
    assert!(parse_connected_over_total("foo trainable bar").is_none());
    assert!(parse_connected_over_total("X/Y trainable").is_none());
}
