//! CFTP §4.4 G3 v4-2 — `@fused_lm_ce(dtype = "...")` decorator-plumbing test.
//!
//! Companion to `fused_lm_ce_decorator_plumbing.rs`.  Sprint v4-2 added a
//! `dtype` keyword arg to the decorator; this test pins the
//!
//!   parse → semantic.FusedCeConfig.dtype
//!     → CLI bridge → codegen.FusedCeDecoratorConfig.dtype
//!     → `FusedCeDtypeHint::dtype_tag` FFI sentinel
//!
//! chain at the four accepted values:
//!
//!   * `dtype = "f32"`  → `FusedCeDtypeHint::F32` / sentinel 0
//!   * `dtype = "fp16"` → `FusedCeDtypeHint::F16` / sentinel 1
//!   * `dtype = "bf16"` → `FusedCeDtypeHint::Bf16` / sentinel 2
//!   * `dtype = "foo"`  → semantic diagnostic, no captured hint
//!
//! Absent `dtype` arg preserves pre-v4-2 byte-identical behavior:
//! `FusedCeConfig.dtype = None`, which the wengert dispatch maps to
//! sentinel 0 (F32) via `fused_ce_dtype_for_compiler`.
//!
//! Pure compile-time test; no CUDA dependency.  The wengert dispatch
//! consumption is exercised separately in
//! `fused_linear_ce_dtype_activation.rs`.

use nsl_codegen::{FusedCeDecoratorConfig, FusedCeDtypeHint};

fn analyze(src: &str) -> nsl_semantic::AnalysisResult {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _lex_diags) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    nsl_semantic::analyze(&parsed.module, &mut interner)
}

/// Mirror of `nsl-cli::analysis_to_fused_ce_configs` — kept inline so this
/// test doesn't pull in nsl-cli.
fn bridge(
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
            dtype: c.dtype.map(|d| match d {
                nsl_semantic::cftp::FusedCeDtypeHint::F32 => FusedCeDtypeHint::F32,
                nsl_semantic::cftp::FusedCeDtypeHint::F16 => FusedCeDtypeHint::F16,
                nsl_semantic::cftp::FusedCeDtypeHint::Bf16 => FusedCeDtypeHint::Bf16,
            }),
        })
        .collect()
}

/// Build a minimal NSL source for the dtype-string under test.
fn src_with_dtype(dtype_lit: &str) -> String {
    format!(
        r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])

@fused_lm_ce(enabled = true, vocab_tile = 256, dtype = "{dtype_lit}")
train(model = m, epochs = 5):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = m.forward(x)
"#
    )
}

#[test]
fn fused_lm_ce_dtype_f32_round_trip() {
    let analysis = analyze(&src_with_dtype("f32"));
    let bridged = bridge(&analysis);
    assert_eq!(bridged.len(), 1, "dtype=f32 must still capture the decorator");
    assert_eq!(bridged[0].dtype, Some(FusedCeDtypeHint::F32));
    assert_eq!(
        bridged[0].dtype.unwrap().dtype_tag(),
        0,
        "F32 → FFI sentinel 0 (pre-v4-2 byte-identical)"
    );
}

#[test]
fn fused_lm_ce_dtype_fp16_round_trip() {
    let analysis = analyze(&src_with_dtype("fp16"));
    let bridged = bridge(&analysis);
    assert_eq!(bridged.len(), 1);
    assert_eq!(bridged[0].dtype, Some(FusedCeDtypeHint::F16));
    assert_eq!(
        bridged[0].dtype.unwrap().dtype_tag(),
        1,
        "F16 → FFI sentinel 1 (v3-2 emitters)"
    );

    // Confirm the "f16" alias also resolves to F16.
    let analysis2 = analyze(&src_with_dtype("f16"));
    let bridged2 = bridge(&analysis2);
    assert_eq!(bridged2[0].dtype, Some(FusedCeDtypeHint::F16));
}

#[test]
fn fused_lm_ce_dtype_bf16_round_trip() {
    let analysis = analyze(&src_with_dtype("bf16"));
    let bridged = bridge(&analysis);
    assert_eq!(bridged.len(), 1);
    assert_eq!(bridged[0].dtype, Some(FusedCeDtypeHint::Bf16));
    assert_eq!(
        bridged[0].dtype.unwrap().dtype_tag(),
        2,
        "Bf16 → FFI sentinel 2 (v4-1 emitters)"
    );
}

#[test]
fn fused_lm_ce_dtype_invalid_string_emits_diagnostic() {
    // "foo" is not a recognised dtype string; semantic must emit a
    // diagnostic AND leave `cfg.dtype = None` so codegen falls back to
    // the F32 default (instead of silently lighting up an unrelated path).
    let analysis = analyze(&src_with_dtype("foo"));
    assert!(
        analysis.diagnostics.iter().any(|d| {
            let s = format!("{:?}", d);
            s.contains("dtype")
                && s.contains("'foo'")
                && s.contains("not recognised")
        }),
        "expected invalid-dtype diagnostic, got: {:?}",
        analysis.diagnostics
    );
    // The decorator is still captured (other arg validation succeeded);
    // only the dtype field is None.
    assert_eq!(analysis.fused_ce_configs.len(), 1);
    let bridged = bridge(&analysis);
    assert_eq!(
        bridged[0].dtype, None,
        "invalid dtype string must fall through to None (F32 default)"
    );
}

#[test]
fn fused_lm_ce_dtype_absent_arg_defaults_to_none() {
    // No dtype arg at all → cfg.dtype = None.  Sentinel 0 (F32) by way
    // of `fused_ce_dtype_for_compiler` in wengert_lower.
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
    let bridged = bridge(&analysis);
    assert_eq!(bridged.len(), 1);
    assert_eq!(
        bridged[0].dtype, None,
        "absent dtype arg → None (pre-v4-2 byte-identical default)"
    );
}
