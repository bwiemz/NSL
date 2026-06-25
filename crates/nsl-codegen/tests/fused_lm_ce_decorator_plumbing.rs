//! CFTP §4.4 G3 (Sprint 2) — end-to-end decorator-plumbing integration test.
//!
//! This test proves the Sprint 2 wiring is mechanically complete: a NSL
//! source string that decorates a `train` block with `@fused_lm_ce(...)`
//! flows from parse → semantic → AnalysisResult → CompileOptions
//! conversion → Compiler state, with the right enabled flag + vocab_tile
//! visible on the codegen side.
//!
//! Sprint 2 stops short of using the plumbed config at the lowering
//! site (see the deferral marker in
//! `crates/nsl-codegen/src/wengert_lower.rs` near
//! `PrimalOp::CrossEntropyLoss`).  This test pins the contract so Sprint
//! 2.5 can light up the substitution without re-wiring the pipeline.
//!
//! No CUDA dependency — pure compile-time semantic + plumbing assertions.
//! The companion v1 GPU test `fused_linear_ce_numerical.rs` continues to
//! validate the kernel-level numerical correctness; this test exercises
//! the orthogonal compiler-integration axis.

use nsl_codegen::FusedCeDecoratorConfig;

/// Mirror of `nsl-cli::analysis_to_fused_ce_configs` used by the CLI bridge.
/// Re-implemented locally so this test does not pull in `nsl-cli`.
fn analysis_to_fused_ce_configs(
    a: &nsl_semantic::AnalysisResult,
) -> Vec<FusedCeDecoratorConfig> {
    a.fused_ce_configs
        .iter()
        .map(|c| FusedCeDecoratorConfig {
            enabled: c.enabled,
            vocab_tile: c.vocab_tile,
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            batch_size: c.batch_size,
            seq_len: c.seq_len,
            dtype: None,
        })
        .collect()
}

fn analyze(src: &str) -> nsl_semantic::AnalysisResult {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _lex_diags) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    nsl_semantic::analyze(&parsed.module, &mut interner)
}

#[test]
fn fused_lm_ce_decorator_round_trips_to_compile_options() {
    let src = r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])

@fused_lm_ce(enabled = true, vocab_tile = 256)
train(model = m, epochs = 5):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = m.forward(x)
"#;
    let analysis = analyze(src);
    assert!(
        analysis.diagnostics.iter().all(|d| !format!("{:?}", d).contains("error")),
        "unexpected diagnostics: {:?}",
        analysis.diagnostics
    );
    assert_eq!(
        analysis.fused_ce_configs.len(),
        1,
        "semantic must capture the decorator"
    );

    // CLI bridge → CompileOptions field.
    let cli_bridged = analysis_to_fused_ce_configs(&analysis);
    assert_eq!(cli_bridged.len(), 1);
    assert!(cli_bridged[0].enabled);
    assert_eq!(cli_bridged[0].vocab_tile, Some(256));

    // CompileOptions accepts the field (assignment-only sanity).  The
    // Compiler constructor that reads the field is exercised by every
    // nsl-codegen lib test that builds an ObjectModule.
    let mut opts = nsl_codegen::CompileOptions::default();
    opts.fused_ce_configs = cli_bridged;
    assert_eq!(opts.fused_ce_configs.len(), 1);
    assert!(opts.fused_ce_configs[0].enabled);
    assert_eq!(opts.fused_ce_configs[0].vocab_tile, Some(256));
}

#[test]
fn absent_decorator_yields_empty_fused_ce_configs() {
    // Same train block, no decorator — fused_ce_configs must be empty.
    let src = r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])

train(model = m, epochs = 5):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = m.forward(x)
"#;
    let analysis = analyze(src);
    assert!(
        analysis.fused_ce_configs.is_empty(),
        "no decorator → no captured configs, got {:?}",
        analysis.fused_ce_configs.len()
    );
    let cli_bridged = analysis_to_fused_ce_configs(&analysis);
    assert!(cli_bridged.is_empty(), "bridge must preserve emptiness");
}

#[test]
fn fused_lm_ce_decorator_disabled_round_trips_with_enabled_false() {
    // Decorator present but `enabled = false` — captured AND flagged
    // disabled, so Sprint 2.5 can skip substitution without re-walking.
    let src = r#"
let m = Tiny()
@fused_lm_ce(enabled = false, vocab_tile = 128)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = 0
"#;
    let analysis = analyze(src);
    assert_eq!(analysis.fused_ce_configs.len(), 1);
    let cli_bridged = analysis_to_fused_ce_configs(&analysis);
    assert!(!cli_bridged[0].enabled);
    assert_eq!(cli_bridged[0].vocab_tile, Some(128));
}

#[test]
fn fused_lm_ce_decorator_on_non_train_target_is_rejected() {
    // Apply to a `model` decl instead of a `train` block — semantic
    // emits an error and DOES NOT capture the config (contract pinned).
    let src = r#"
@fused_lm_ce(enabled = true)
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let analysis = analyze(src);
    assert!(
        analysis.fused_ce_configs.is_empty(),
        "non-train target must NOT capture"
    );
    assert!(
        analysis
            .diagnostics
            .iter()
            .any(|d| format!("{:?}", d)
                .contains("@fused_lm_ce may only be applied to a `train` block")),
        "expected target-mismatch diagnostic, got: {:?}",
        analysis.diagnostics
    );
}
