//! Spec §6.5.1 — regression guard against silent auto-mode fallback.
//!
//! Three tests, one per §5.4/§5.5/§5.6 trigger condition.  Each test drives
//! `debug_resolve_pre_scan_opts` (which runs only the pre-scan phase, not full
//! codegen) and asserts that `wggo_importance` is demoted from `Auto` to
//! `Magnitude`.  This guards against any future change silently removing the
//! demotion logic.
//!
//! Approach: Approach A (public debug-helper API).  `CARGO_BIN_EXE_nsl` is
//! not used here because the binary target is not always built in the same
//! incremental step and the CLI approach adds subprocess overhead without
//! increasing coverage.

use nsl_codegen::{debug_resolve_pre_scan_opts, CompileOptions, WggoImportance};

fn auto_opts() -> CompileOptions {
    let mut opts = CompileOptions::default();
    opts.wggo_importance = WggoImportance::Auto;
    opts
}

// ── §5.4: no @wggo_target decorators in source ──────────────────────────────

#[test]
fn auto_no_decorators_demotes_to_magnitude() {
    let source = include_str!("../../../tests/fixtures/no_attention_mlp.nsl");
    let resolved = debug_resolve_pre_scan_opts(source, auto_opts());
    assert_eq!(
        resolved.wggo_importance,
        WggoImportance::Magnitude,
        "§5.7/§5.4: Auto must demote to Magnitude when no @wggo_target decorators are present"
    );
}

// ── §5.5: decorators present but decorated model not reachable from main() ──

#[test]
fn auto_orphaned_decorators_demotes_to_magnitude() {
    // orphaned_attention.nsl: Attention has @wggo_target but main() instantiates
    // OtherModel — pre-scan finds zero reachable targets (§5.5 trigger).
    let source = include_str!("../../../tests/fixtures/orphaned_attention.nsl");
    let resolved = debug_resolve_pre_scan_opts(source, auto_opts());
    assert_eq!(
        resolved.wggo_importance,
        WggoImportance::Magnitude,
        "§5.7/§5.5: Auto must demote to Magnitude when decorated model is not reachable from main"
    );
}

// ── §5.6: decorators present + model reachable, but no calibration data ─────

#[test]
fn auto_no_calibration_data_demotes_to_magnitude() {
    // attention_no_calib.nsl: decorated Attention IS reachable from main(), but
    // calibration_data is None (§5.6 trigger).
    let source = include_str!("../../../tests/fixtures/attention_no_calib.nsl");
    let mut opts = auto_opts();
    // Explicitly ensure calibration_data is absent (it is by default, but be
    // explicit for clarity and robustness against future default changes).
    opts.calibration_data = None;
    let resolved = debug_resolve_pre_scan_opts(source, opts);
    assert_eq!(
        resolved.wggo_importance,
        WggoImportance::Magnitude,
        "§5.7/§5.6: Auto must demote to Magnitude when calibration data is absent"
    );
}
