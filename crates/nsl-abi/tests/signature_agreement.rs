//! Workspace-level ABI gate: the codegen's declared runtime-function
//! signatures — every `const RUNTIME_FUNCTIONS*` under `nsl-codegen/src`,
//! spread over `builtins/` and `runtime_abi/` — must agree with the runtime's
//! `extern "C"` implementations. These are linked by symbol name only, so
//! nothing else in the build catches a drift; this test does.

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

    // A declaration that was RECOGNISED but not READ is a parser regression,
    // and every failure path inside the table parser is silent — a missing
    // `=`, an initializer that is not a slice literal, an unbalanced `[...]`
    // all yield "no entries" rather than an error. This catches that exactly,
    // which the count-based floor below can only approximate.
    assert_eq!(
        report.tables_found, report.tables_parsed,
        "{} `const RUNTIME_FUNCTIONS*` declaration(s) were found but {} parsed — a table \
         was recognised and then not read. That is a parser regression, not a registry change.",
        report.tables_found, report.tables_parsed
    );
    assert!(
        report.tables_parsed >= 1,
        "no `const RUNTIME_FUNCTIONS*` table was found under crates/nsl-codegen/src at all — \
         the registry moved out of the scanned tree, and this gate is checking nothing."
    );

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
    // The real backstop is the `tables_found == tables_parsed` assertion
    // above, which catches a truncated parse exactly and needs no constant.
    // This floor stays as defence in depth against a failure that produces a
    // short table without failing to parse one.
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
