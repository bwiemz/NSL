//! Regression test for the calibration retention arena ordering bug.
//!
//! Bug: `emit_retention_arena` used to run AFTER `compile_user_functions` in
//! every codegen entry point.  Model method bodies — which is where the pipe
//! sites live — compile inside `compile_user_functions`.  The splice in
//! `expr/mod.rs::try_emit_retention_splice` early-returns when
//! `retention_arena_data_id` is `None`, so every splice site silently no-opped
//! and the calibration subprocess read an all-zeros retention arena.
//!
//! Fix: `emit_retention_arena` now runs immediately after `collect_models` and
//! before `compile_user_functions` in every entry point.  When
//! `calibration_retention` is pre-set on `CompileOptions`, the arena data_id
//! is available by the time method bodies lower their pipe targets, and the
//! splice emits `emit_splice_memcpy` IR.
//!
//! This test compiles the `TinyMLP` fixture (two pipes — `|> up_proj` and
//! `|> down_proj` — with a `|> relu` in between that is *not* a projection)
//! with `calibration_retention` pre-set, and asserts the splice counter
//! reports exactly 2 emissions.
//!
//! Not covered here:
//! - Calibration subprocess not calling `model_forward` at all.
//! - No model library linked into the calibration binary.
//!
//! Those are separate design-level gaps tracked in project memory.

use nsl_codegen::calibration::discovery::DiscoveredProjection;
use nsl_codegen::calibration::observation::ProjectionRef;
use nsl_codegen::{compile_returning_splice_count_for_tests, CompileOptions};

// Minimal inline fixture: model with a pipe-based forward whose identifiers
// are declared as free functions so the pipe targets resolve.  The splice
// only cares about the identifier name matching a discovered projection;
// the actual call on the right-hand side is irrelevant to the splice
// counter.  No top-level `main` — `compile_user_functions` lowers only the
// free functions and the model method.
const TINY_MLP_FIXTURE: &str = r#"
fn up_proj(x: Tensor) -> Tensor:
    return x

fn down_proj(x: Tensor) -> Tensor:
    return x

fn relu(x: Tensor) -> Tensor:
    return x

model TinyMLP:
    up_proj: Tensor = zeros([128, 64])
    down_proj: Tensor = zeros([64, 128])

    fn forward(self, x: Tensor) -> Tensor:
        return x |> up_proj |> relu |> down_proj
"#;

#[test]
fn retention_splice_fires_when_arena_declared_before_method_codegen() {
    let source = TINY_MLP_FIXTURE;

    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(&source, nsl_errors::FileId(0), &mut interner);
    assert!(
        lex_diags
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "fixture must pass semantic: {:?}",
        analysis.diagnostics
    );

    // Pre-set calibration_retention on CompileOptions — the harness-driven
    // train-block path would normally set this during compile_main, but the
    // splice needs it BEFORE compile_user_functions runs.  This test bypasses
    // the harness to isolate the ordering fix.
    let projections = vec![
        DiscoveredProjection {
            projection: ProjectionRef::new("TinyMLP.up_proj"),
            weight_shape: [128, 64],
        },
        DiscoveredProjection {
            projection: ProjectionRef::new("TinyMLP.down_proj"),
            weight_shape: [64, 128],
        },
    ];

    let opts = CompileOptions {
        calibration_retention: Some(projections),
        calibration_batch_seq: Some((1, 1)),
        ..Default::default()
    };

    let splice_count = compile_returning_splice_count_for_tests(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &opts,
    )
    .expect("compile must succeed with retention pre-set");

    // TinyMLP.forward body is `x |> up_proj |> relu |> down_proj` — two
    // projection pipes (up_proj, down_proj), one builtin (relu that is not a
    // projection). Each projection pipe must emit exactly one splice.
    assert_eq!(
        splice_count, 2,
        "expected 2 retention splice emissions (up_proj + down_proj); got {splice_count}. \
         This means the arena was not declared before method-body codegen — the \
         ordering fix regressed."
    );
}

/// Negative control: when `calibration_retention` is `None` (the shipped-
/// binary path), the splice must NOT fire even though the source contains
/// pipe expressions.  This proves the counter above isn't incremented by
/// something unrelated and that the shipped path is truly zero-cost.
#[test]
fn retention_splice_does_not_fire_without_calibration_retention() {
    let source = TINY_MLP_FIXTURE;

    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(source, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);

    let opts = CompileOptions::default();

    let splice_count = compile_returning_splice_count_for_tests(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &opts,
    )
    .expect("compile must succeed without retention");

    assert_eq!(
        splice_count, 0,
        "splice must be zero-cost when calibration_retention is None; got {splice_count}",
    );
}
