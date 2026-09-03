//! Workspace-level ABI gate: the codegen's declared runtime-function signatures
//! (`nsl-codegen/src/builtins.rs::RUNTIME_FUNCTIONS`) must agree with the
//! runtime's `extern "C"` implementations. These are linked by symbol name
//! only, so nothing else in the build catches a drift — this test does.

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

    // Guard against a silently-empty or short parse (a path/parser regression)
    // making this test vacuously green. `cross_check` visits every declared
    // entry exactly once, so this total IS the number of entries parsed.
    let total = report.verified + report.via_macro + report.mismatches.len();
    // Truncation floor. This was `> 540` against a comment claiming "~558
    // entries" when the registry already held 682: the floor had drifted 142
    // entries behind the table, so a parse that silently dropped a fifth of it
    // still passed. It is kept a floor rather than an equality — a PR that adds
    // a runtime function should not have to edit this file — but a floor is
    // only a backstop while it stays near the real count, so it is stated as a
    // percentage of a recorded count and both move together.
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
             `extern \"C\"` implementation.\n(declared in \
             nsl-codegen/src/builtins.rs::RUNTIME_FUNCTIONS; implemented in nsl-runtime)\n\n",
            report.mismatches.len()
        );
        for m in &report.mismatches {
            msg.push_str(&format!("  [{:?}] {} — {}\n", m.kind, m.name, m.detail));
        }
        msg.push_str(
            "\nFix by reconciling the RUNTIME_FUNCTIONS entry with the extern \"C\" fn (arity + \
             types), or, if the runtime fn is macro-generated/behind a cfg the parser cannot see, \
             extend nsl-abi to recognize it.\n",
        );
        panic!("{msg}");
    }

    eprintln!(
        "nsl-abi: {} signatures verified against extern impls, {} via inplace macro, 0 drift",
        report.verified, report.via_macro
    );
}
