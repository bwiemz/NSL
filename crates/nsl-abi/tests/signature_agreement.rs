//! Workspace-level ABI gate: every row of the typed table
//! (`nsl_abi::RUNTIME_ABI`, which the codegen renders its declarations from)
//! must agree with the runtime's `extern "C"` implementations, parsed from
//! source text. Since roadmap A3 step 1 the runtime's own build checks the
//! same agreement through `rustc` (`nsl-runtime/src/abi_check.rs`); this
//! gate is the belt-and-braces text check, and the one place that reports
//! every disagreement at once rather than the first.

use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR = <root>/crates/nsl-abi
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates dir")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn runtime_function_signatures_agree_with_extern_impls() {
    let root = workspace_root();
    let report = nsl_abi::check_workspace(&root).expect("read workspace sources");

    // Guard against a silently-short parse making this test vacuously green.
    // `cross_check` visits every declared entry exactly once, so this total IS
    // the number of entries parsed.
    let total = report.verified + report.via_macro + report.mismatches.len();
    // Truncation floor, and a weak one by construction.
    //
    // It was `> 540` under a comment claiming "~558 entries" when the registry
    // already held 682 — drifted 142 behind, so a parse dropping a fifth of
    // the table still passed. Restating it against a recorded count narrows
    // that to 35, but does NOT fix it: `RECORDED_TOTAL` is a hand-typed
    // literal that nothing updates, so it drifts again at exactly the same
    // rate. The honest claim is only that the constant is named and dated, so
    // the drift is legible to whoever reads it next.
    //
    // The real backstop is now `nsl_abi::table::tests::table_is_the_recorded_size`,
    // which pins the row count exactly. This floor stays as defence in depth.
    const RECORDED_TOTAL: usize = 682; // 2026-09-02
    let floor = RECORDED_TOTAL * 95 / 100;
    assert!(
        total >= floor,
        "parsed only {total} declared runtime functions, expected at least {floor} \
         ({RECORDED_TOTAL} recorded on 2026-09-02, less a 5% margin) — a parser or path \
         regression, or a truncated table parse. If the registry legitimately shrank \
         below this, update RECORDED_TOTAL."
    );

    if !report.mismatches.is_empty() {
        let mut msg = format!(
            "\nABI signature drift: {} declared runtime function(s) disagree with their runtime \
             `extern \"C\"` implementation.\n(each line names the declaring file and table; \
             implemented in nsl-runtime)\n\n",
            report.mismatches.len()
        );
        for m in &report.mismatches {
            msg.push_str(&format!("  [{:?}] {} — {}\n", m.kind, m.name, m.detail));
        }
        msg.push_str(
            "\nFix by reconciling the row in crates/nsl-abi/src/table.rs with the extern \"C\" fn \
             (arity + types), or, if the runtime fn is macro-generated/behind a cfg the parser \
             cannot see, extend nsl-abi to recognize it.\n",
        );
        panic!("{msg}");
    }

    eprintln!(
        "nsl-abi: {} signatures verified against extern impls, {} via inplace macro, 0 drift",
        report.verified, report.via_macro
    );
}
