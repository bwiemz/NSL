//! CFTP §4.4 G3 (Sprint v3-1) — auto-substitution coverage.
//!
//! Sprint 4 wired the COMPILER side of the explicit-call path:
//! `fused_linear_ce(x, W, bias, targets)` in user code → recognised in
//! source-AD as a single `PrimalOp::FusedLinearCe`, AD rule emits three
//! backward components.
//!
//! Sprint v3-1 (this test file) covers the implicit-call path: when a
//! `train` block carries `@fused_lm_ce(enabled = true, ...)` AND the
//! user writes the canonical decomposition
//!
//! ```text
//!   W_t = transpose(W, 0, 1)
//!   logits = matmul(x, W_t) + bias
//!   loss   = cross_entropy(logits, targets)
//! ```
//!
//! the compiler should detect the upstream
//! `Add(Matmul(x, Transpose(W, 0, 1)), bias)` chain in the Wengert tape
//! and substitute the `cross_entropy` recognition with a single
//! `PrimalOp::FusedLinearCe`, identical to what the explicit call path
//! emits.  Backward then flows through the same AD rule.
//!
//! These tests assert PrimalOp presence/absence in the extracted
//! `WengertList`.  GPU numerical correctness is already covered by
//! `fused_linear_ce_numerical.rs` (V=4096) and
//! `fused_linear_ce_large_vocab_numerical.rs` (V=49152), so this file
//! deliberately stops at the Wengert layer — exactly mirroring
//! `fused_linear_ce_lowering_end_to_end.rs`'s Sprint 4 strategy.

use nsl_codegen::fused_linear_ce::LARGE_VOCAB_THRESHOLD;
use nsl_codegen::source_ad::WengertExtractor;
use nsl_codegen::wengert::PrimalOp;
use nsl_codegen::{FusedCeDecoratorConfig, FusedCeDtypeHint};
use nsl_lexer::Interner;

/// Extract the Wengert list for the first FnDef in `src`, optionally
/// with a `@fused_lm_ce` decorator config plumbed in.  Returns the
/// extracted list (when extraction succeeded).
///
/// Mirrors `fused_linear_ce_lowering_end_to_end::extract_first_fn`.
fn extract_first_fn(
    src: &str,
    cfg: Option<FusedCeDecoratorConfig>,
) -> Option<nsl_codegen::wengert::WengertList> {
    extract_first_fn_with_ranks(src, cfg, &[])
}

/// CFTP v10 (item 5): extract the first FnDef's Wengert list, threading
/// per-parameter rank annotations through
/// `WengertExtractor::register_input_with_rank`.  Parameter names in
/// `ranks` that don't appear in the fn's parameter list are ignored;
/// parameters not listed default to unknown rank (matching the
/// pre-v10 behaviour that leaves the matcher conservatively firing).
fn extract_first_fn_with_ranks(
    src: &str,
    cfg: Option<FusedCeDecoratorConfig>,
    ranks: &[(&str, usize)],
) -> Option<nsl_codegen::wengert::WengertList> {
    let mut interner_box = Box::new(Interner::new());
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner_box);
    let parsed = nsl_parser::parse(&tokens, &mut interner_box);
    let module = parsed.module;
    // SAFETY: leak the interner so the returned references live for the
    // test duration. This is a one-off test pattern.
    let interner_static: &'static Interner = Box::leak(interner_box);
    let mut extractor =
        WengertExtractor::new(interner_static).with_fused_ce_config(cfg);
    let fn_def = module
        .stmts
        .iter()
        .find_map(|s| match &s.kind {
            nsl_ast::stmt::StmtKind::FnDef(f) => Some(f),
            _ => None,
        })
        .expect("test src must contain a fn");
    for param in &fn_def.params {
        let param_name = interner_static.resolve(param.name.0).unwrap_or("?");
        let rank = ranks
            .iter()
            .find(|(name, _)| *name == param_name)
            .map(|(_, r)| *r);
        extractor.register_input_with_rank(param.name, rank);
    }
    let ok = extractor.extract_stmts(&fn_def.body.stmts);
    if !ok {
        return None;
    }
    extractor.finalize()
}

fn count_fused(list: &nsl_codegen::wengert::WengertList) -> usize {
    list.ops
        .iter()
        .filter(|op| matches!(op.op, PrimalOp::FusedLinearCe { .. }))
        .count()
}

fn count_composite_ce(list: &nsl_codegen::wengert::WengertList) -> usize {
    list.ops
        .iter()
        .filter(|op| matches!(op.op, PrimalOp::CrossEntropyLoss))
        .count()
}

/// Canonical user-written decomposition that the auto-substitution
/// should target.  Mirrors stdlib `fused_linear_ce`'s body exactly so
/// the test exercises the realistic codepath users hit.
const AUTO_SUB_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;

/// Same pattern but with NO bias add — so the matcher should refuse
/// even with the decorator fully populated.
const NO_BIAS_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 1)
    let logits = matmul(x, w_t)
    return cross_entropy(logits, targets)
"#;

/// CE called on a bare ident (no Matmul/Add chain at all) — the matcher
/// must refuse and fall through to the composite path.
const PLAIN_LOGITS_SRC: &str = r#"
fn step(logits: Tensor, targets: Tensor) -> Tensor:
    return cross_entropy(logits, targets)
"#;

/// Sprint 4's explicit-call path — `fused_linear_ce(x, w, bias, targets)`
/// remains a separate code path from the auto-substitution; this guard
/// catches accidental regressions.
const EXPLICIT_CALL_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    return fused_linear_ce(x, w, bias, targets)
"#;

// ─── Test 1: decorator enabled + canonical pattern → auto-substitution fires ─

#[test]
fn auto_substitution_fires_when_decorator_enabled_and_pattern_matches() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(AUTO_SUB_SRC, Some(cfg))
        .expect("extraction must succeed");
    let fused: Vec<_> = list
        .ops
        .iter()
        .filter_map(|op| match &op.op {
            PrimalOp::FusedLinearCe {
                vocab_size,
                hidden_size,
                batch_size,
                seq_len,
                vocab_tile,
                ignore_index,
                is_large,
            } => Some((
                *vocab_size,
                *hidden_size,
                *batch_size,
                *seq_len,
                *vocab_tile,
                *ignore_index,
                *is_large,
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        fused.len(),
        1,
        "expected exactly one FusedLinearCe op after auto-substitution"
    );
    let (v, h, b, s, vt, ignore, is_large) = fused[0];
    assert_eq!(v, 4096);
    assert_eq!(h, 128);
    assert_eq!(b, 2);
    assert_eq!(s, 32);
    assert_eq!(vt, 1024);
    assert_eq!(ignore, -100);
    assert!(
        !is_large,
        "V=4096 < {LARGE_VOCAB_THRESHOLD} must route to small-vocab path"
    );
    // And critically: the composite CE op MUST NOT also appear (we
    // substituted, not duplicated).
    assert_eq!(
        count_composite_ce(&list),
        0,
        "PrimalOp::CrossEntropyLoss must NOT be emitted alongside the fused op"
    );
}

// ─── Test 2: decorator off → composite preserved ────────────────────────

#[test]
fn decorator_disabled_preserves_composite_cross_entropy() {
    let cfg = FusedCeDecoratorConfig {
        enabled: false,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(AUTO_SUB_SRC, Some(cfg))
        .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        0,
        "decorator disabled → no FusedLinearCe should appear"
    );
    assert_eq!(
        count_composite_ce(&list),
        1,
        "composite PrimalOp::CrossEntropyLoss must be emitted instead"
    );
}

// ─── Test 3: decorator absent → composite preserved ─────────────────────

#[test]
fn decorator_absent_preserves_composite_cross_entropy() {
    let list = extract_first_fn(AUTO_SUB_SRC, None)
        .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        0,
        "no decorator → no FusedLinearCe should appear"
    );
    assert_eq!(
        count_composite_ce(&list),
        1,
        "composite PrimalOp::CrossEntropyLoss must be emitted instead"
    );
}

// ─── Test 4: pattern mismatch (no Add/Matmul chain) → composite ─────────

#[test]
fn pattern_mismatch_falls_through_to_composite_even_when_enabled() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    // No upstream Matmul or Add — `cross_entropy` is called on a bare
    // function parameter.  The matcher MUST refuse.
    let list = extract_first_fn(PLAIN_LOGITS_SRC, Some(cfg))
        .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        0,
        "pattern mismatch (plain logits, no Add(Matmul, bias) chain) \
         must NOT trigger auto-substitution"
    );
    assert_eq!(
        count_composite_ce(&list),
        1,
        "composite PrimalOp::CrossEntropyLoss must be emitted as fallback"
    );

    // Sub-case: pattern has Matmul + Transpose but NO bias add.
    // The matcher requires the BiasAdd because v1's FFI signature is
    // 4-input (x, W, bias, targets) with no bias-elision.
    let list_no_bias = extract_first_fn(
        NO_BIAS_SRC,
        Some(FusedCeDecoratorConfig {
            enabled: true,
            vocab_tile: Some(1024),
            vocab_size: Some(4096),
            hidden_size: Some(128),
            batch_size: Some(2),
            seq_len: Some(32),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
        }),
    )
    .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list_no_bias),
        0,
        "Matmul-without-bias-Add pattern must NOT trigger auto-substitution"
    );
    assert_eq!(
        count_composite_ce(&list_no_bias),
        1,
        "no-bias case must lower composite CE"
    );
}

// ─── Test 5: large-vocab routing through auto-substitution ──────────────

#[test]
fn auto_substitution_routes_large_vocab_to_two_kernel_path() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(128),
        vocab_size: Some(49152),
        hidden_size: Some(512),
        batch_size: Some(1),
        seq_len: Some(64),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(AUTO_SUB_SRC, Some(cfg))
        .expect("extraction must succeed");
    let is_large = list
        .ops
        .iter()
        .find_map(|op| match &op.op {
            PrimalOp::FusedLinearCe { is_large, .. } => Some(*is_large),
            _ => None,
        })
        .expect(
            "expected exactly one FusedLinearCe op produced by auto-substitution",
        );
    assert!(
        is_large,
        "V=49152 > {LARGE_VOCAB_THRESHOLD} must route to large-vocab \
         path through the auto-substituted op too"
    );
    assert_eq!(
        count_composite_ce(&list),
        0,
        "PrimalOp::CrossEntropyLoss must NOT also appear"
    );
}

// ─── Test 7: review Finding 1 — dead composite chain is pruned ──────────
//
// After auto-substitution emits `PrimalOp::FusedLinearCe`, the upstream
// `Transpose → Matmul → Add` chain that produced `logits_var` must be
// removed from the Wengert tape.  The lowerer has no DCE, so leaving
// the chain in place causes the train step to physically run a
// `[B*S, V]` matmul + bias-add that nothing consumes — wasting ~1.5 GB
// of HBM per step at V=49152/H=4096/B=2/S=4096.

#[test]
fn auto_substitution_prunes_dead_composite_upstream_chain() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(AUTO_SUB_SRC, Some(cfg))
        .expect("extraction must succeed");

    // Sanity: exactly one FusedLinearCe.
    assert_eq!(
        count_fused(&list),
        1,
        "auto-substitution must fire on the canonical pattern"
    );

    // Walk the tape — none of these three composite ops should remain.
    // (Without the prune they would all still be present alongside the
    // fused op, since substitution emits an additional op rather than
    // structurally replacing the chain.)
    let has_matmul = list
        .ops
        .iter()
        .any(|op| matches!(op.op, PrimalOp::Matmul));
    let has_transpose = list
        .ops
        .iter()
        .any(|op| matches!(op.op, PrimalOp::Transpose { dim0: 0, dim1: 1 }));
    let has_add = list
        .ops
        .iter()
        .any(|op| matches!(op.op, PrimalOp::Add));

    assert!(
        !has_matmul,
        "PrimalOp::Matmul from the substituted chain must be pruned \
         after auto-substitution (Finding 1 — would otherwise allocate \
         a wasted [B*S, V] tensor each training step)"
    );
    assert!(
        !has_transpose,
        "PrimalOp::Transpose {{ dim0: 0, dim1: 1 }} from the substituted \
         chain must be pruned after auto-substitution (Finding 1)"
    );
    assert!(
        !has_add,
        "PrimalOp::Add (the bias-add producing logits_var) must be pruned \
         after auto-substitution (Finding 1)"
    );

    // Composite CE must still NOT be present (already covered by Test 1
    // but re-asserted here so this test stands on its own).
    assert_eq!(
        count_composite_ce(&list),
        0,
        "PrimalOp::CrossEntropyLoss must not appear after substitution"
    );
}

// ─── Test 8: review Finding 3 — Add(Matmul, Matmul) is rejected ────────
//
// `try_match_fused_linear_ce_pattern` previously accepted whatever the
// non-Matmul operand of the Add was as `bias_var`.  If BOTH Add
// operands are Matmuls (e.g. `matmul(x1, W1^T) + matmul(x2, W2^T)`),
// the wrong-shape RHS would silently be passed in the bias slot to the
// fused FFI, which dereferences it as a `[V]` vector — reading garbage.
//
// The fix: detect the bias-side Matmul producer and reject the
// substitution, falling through to the composite path.

const ADD_MATMUL_MATMUL_SRC: &str = r#"
fn step(x1: Tensor, w1: Tensor, x2: Tensor, w2: Tensor, targets: Tensor) -> Tensor:
    let w1_t = transpose(w1, 0, 1)
    let w2_t = transpose(w2, 0, 1)
    let logits = matmul(x1, w1_t) + matmul(x2, w2_t)
    return cross_entropy(logits, targets)
"#;

#[test]
fn ambiguous_add_matmul_matmul_pattern_is_rejected() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(ADD_MATMUL_MATMUL_SRC, Some(cfg))
        .expect("extraction must succeed");

    // The matcher must refuse — the "bias" operand is itself a Matmul
    // result, so passing it as a `[V]` bias would corrupt the kernel.
    assert_eq!(
        count_fused(&list),
        0,
        "Add(Matmul, Matmul) is ambiguous — the bias slot would receive \
         a high-rank tensor.  Substitution must be refused (Finding 3)."
    );
    // Composite CE must take over.
    assert_eq!(
        count_composite_ce(&list),
        1,
        "composite PrimalOp::CrossEntropyLoss must be emitted as fallback"
    );
}

// ─── Test 9 (Sprint v4-3): negative-dim transpose substitutes ───────────
//
// Sprint v4-3 widens the matcher to accept all four 2D-equivalent
// transpose forms: `{0,1}`, `{1,0}`, and the negative-dim encodings
// `{-2,-1}` / `{-1,-2}` (which `encode_transpose_dim` rewrites to
// `{usize::MAX - 1, usize::MAX}` and its swap).  Pre-v4-3 only the
// first was accepted, silently routing idiomatic NSL code onto the
// composite path.
//
// All four assertions hit the same matcher arm, so we parameterise
// over the four user-written transpose-arg pairs to pin all four.
//
// `transpose(W, 1, 0)` produces an actual permutation that differs
// from the stdlib `transpose(W, 0, 1)` if W is non-square, but on a
// 2D weight tensor of any shape the kernel sees the same memory layout
// either way — both produce `W^T`.  The semantic equivalence is what
// the matcher is asserting; we are NOT asserting the kernel runs (the
// numerical regression tests in `fused_linear_ce_numerical.rs` cover
// the kernel layer).

const NEG_DIM_SRC_LAST_TWO: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, -2, -1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;

const NEG_DIM_SRC_LAST_TWO_SWAP: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, -1, -2)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;

const POS_DIM_SRC_SWAPPED: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 1, 0)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;

#[test]
fn negative_dim_transpose_substitutes() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };

    for (label, src) in [
        ("transpose(W, -2, -1)", NEG_DIM_SRC_LAST_TWO),
        ("transpose(W, -1, -2)", NEG_DIM_SRC_LAST_TWO_SWAP),
        ("transpose(W, 1, 0)", POS_DIM_SRC_SWAPPED),
    ] {
        let list = extract_first_fn(src, Some(cfg.clone()))
            .unwrap_or_else(|| panic!("{label}: extraction must succeed"));
        assert_eq!(
            count_fused(&list),
            1,
            "{label}: Sprint v4-3 widened matcher MUST substitute this \
             2D-equivalent transpose form into a fused op"
        );
        assert_eq!(
            count_composite_ce(&list),
            0,
            "{label}: composite CE must NOT also appear post-substitution"
        );
    }
}

// ─── Test 10 (Sprint v4-3): bias-free DEFERRAL pin ──────────────────────
//
// A bare `Matmul(x, Transpose(W))` (no bias add) is NOT substituted in
// v4-3 — the runtime FFI emitters at
// `crates/nsl-runtime/src/fused_linear_ce.rs` and the corresponding PTX
// emitters in `crates/nsl-codegen/src/fused_linear_ce.rs` unconditionally
// declare `param_bias` with no null-guard predicate; a true bias-free
// fast path needs new emitter variants.  This test pins the deferral
// (NOT a regression of the v4-3 widening) — if a future sprint adds a
// bias-free emitter, this test should be UPDATED, not deleted, to
// document the v4-3 → v5+ behaviour transition.

#[test]
fn bias_free_pattern_deferred_to_v5() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(NO_BIAS_SRC, Some(cfg))
        .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        0,
        "Sprint v4-3 deferral: bias-free Matmul(x, W^T) → CE chain does \
         NOT substitute (needs a no-bias PTX emitter variant; tracked v5+). \
         If a future sprint lands the variant, update this assertion to \
         `count_fused == 1` and document the behaviour flip in the test \
         body."
    );
    assert_eq!(
        count_composite_ce(&list),
        1,
        "bias-free case falls through to composite CE (correct + safe — \
         slower but bit-equivalent)"
    );
}

// ─── Test 11 (Sprint v4-3): no-transpose DEFERRAL pin ───────────────────
//
// `Matmul(x, W)` where W is already in `[H, V]` layout is NOT
// substituted in v4-3 — every PTX emitter in
// `crates/nsl-codegen/src/fused_linear_ce.rs` indexes W as `W[v*H + h]`,
// i.e. requires W in `[V, H]` layout.  Supporting `[H, V]` needs a
// parallel emitter variant + new FFI symbol; tracked v5+.
//
// Same pinning contract as Test 10: if a future sprint lands the
// variant, update the assertion + document the transition.

const NO_TRANSPOSE_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let logits = matmul(x, w) + bias
    return cross_entropy(logits, targets)
"#;

// ─── Test 12 (Adversarial review Findings 1 + 8): rank-3 W LATENT-HAZARD pin ─
//
// The auto-substitution matcher does NOT structurally verify that W is
// rank-2 — it accepts whatever VarId sits in the transpose op's first
// input slot.  Every PTX emitter indexes W as W[v*H + h] (rank-2 [V, H]);
// a user who wrote `matmul(x, transpose(W3D, -2, -1)) + bias` over a
// rank-3 W3D (e.g. an MoE expert stack [D, V, H]) would have the matcher
// silently fire, the fused FFI stride through W3D as if it were [V, H],
// and produce wrong forward logits + wrong dW gradients with no
// diagnostic.
//
// The source-AD layer has no shape table available at the matcher site
// (only the wengert PrimalOp graph), so we cannot structurally enforce
// rank-2 today.  Current mitigations:
//   * The upstream type system rejects most non-2D Matmul shape configs.
//   * The decorator's (vocab_size, hidden_size) shape contract pins the
//     expected [V, H] shape at the caller before substitution.
// A 3-D W that satisfies neither check would still slip through.
//
// This test EXISTS to document the deferral surface — the source NSL
// code below does NOT actually construct a rank-3 W (it's just a parameter
// the matcher sees as opaque), and the matcher still fires.  The test
// asserts the CURRENT documented-bug behaviour so a future structural
// rank-2 enforcement lands with an explicit assertion FLIP (not a silent
// behaviour change).  IF structural enforcement is added: replace
// `count_fused == 1` with `count_fused == 0` and add the rank-3 negative
// path as the new safety net.

const RANK_AGNOSTIC_NEG_DIM_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, -2, -1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;

#[test]
fn rank_check_absent_matcher_fires_regardless_of_w_rank() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(RANK_AGNOSTIC_NEG_DIM_SRC, Some(cfg))
        .expect("extraction must succeed");
    // CFTP v10 (item 5): CURRENT BEHAVIOUR — rank enforcement is
    // conservative-fire when rank is UNKNOWN.  The parser+extractor
    // here don't infer rank from the bare `Tensor` param annotation, so
    // `known_ranks` stays empty and the matcher preserves its pre-v10
    // fire behaviour.  This is intentional: unannotated code must not
    // regress on the auto-substitution optimisation.  Programs with an
    // annotated `Tensor<[V, H]>` param get the structural check via the
    // compiler's `register_input_with_rank` plumbing in `stmt.rs`
    // (`resolvable_tensor_rank`) — pinned by
    // `rank_3_annotated_w_refuses_substitution` +
    // `rank_2_annotated_w_still_substitutes` below.
    assert_eq!(
        count_fused(&list),
        1,
        "unknown-rank path stays fire-conservative (rank annotations \
         are the frontend's job; the matcher only refuses when it can \
         PROVE non-2D)"
    );
}

// ─── CFTP v10 item 5: rank-3 W refuses substitution ─────────────────────
//
// When the compiler threads a rank-3 annotation via
// `register_input_with_rank` (mirroring the AST/semantic path in
// `resolvable_tensor_rank`), the matcher must REFUSE the substitution.
// This is the structural close on the LATENT 3-D+ RISK documented in
// `source_ad.rs::try_match_fused_linear_ce_pattern`.

#[test]
fn rank_3_annotated_w_refuses_substitution() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn_with_ranks(
        AUTO_SUB_SRC,
        Some(cfg),
        // Announce rank-3 W (e.g. an MoE expert stack [D, V, H]).  Every
        // fused-CE PTX emitter indexes W as `W[v*H + h]` (rank-2 [V, H]) —
        // firing on rank-3 would produce wrong forward logits + wrong dW
        // gradients with no diagnostic.  Item 5 refuses.
        &[("w", 3)],
    )
    .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        0,
        "item 5 structural enforcement: rank-3 W must refuse the fused \
         substitution — the fused FFI would silently stride through the \
         rank-3 tensor as if it were [V, H]"
    );
    assert_eq!(
        count_composite_ce(&list),
        1,
        "rank-3 W falls through to composite CE (correct + safe)"
    );
}

// ─── CFTP v10 item 5: rank-2 W still substitutes ────────────────────────
//
// The positive counterpart — when the compiler PROVES rank-2, the
// matcher fires as before.  Guards against a future refactor that
// accidentally makes the check refuse rank-2 too (or defaults `rank`
// to `Some(0)` instead of `None` for the unknown case, causing every
// program to be refused).

// ─── CFTP v10 item 5 (reviewer-flagged gap closure): rank-3 MODEL FIELD W refuses ─
//
// The initial item-5 fix only populated `known_ranks` from
// step-body parameter type annotations.  Reviewer flagged this
// as leaving the exact MoE expert stack scenario unprotected —
// weight tensors accessed via `self.field` bypass
// `register_input_with_rank` entirely and land in
// `named_param_vars` via `extract_expr`'s MemberAccess arm.
//
// This test walks the FULL member-access Param-registration
// path: it declares a model with a 3-D weight, wires the
// extractor with `set_model_field_ranks` (mirroring what
// `stmt.rs::compile_train_block` does at runtime from
// `Compiler::models.model_field_ranks`), then extracts a
// forward body that does `matmul(x, transpose(self.w, -2, -1))
// + bias` → `cross_entropy(...)`.
//
// Pre-fix: matcher fires (0 → 1 fused ops), silently striding
// through a rank-3 tensor as if it were `[V, H]`.
// Post-fix: matcher refuses; composite CE takes over.

const MODEL_FIELD_MEMBER_ACCESS_SRC: &str = r#"
model Tiny:
    w: Tensor

    fn forward(self, x: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
        let w_t = transpose(self.w, -2, -1)
        let logits = matmul(x, w_t) + bias
        return cross_entropy(logits, targets)
"#;

/// Extract the first model's `forward` method as if
/// `compile_train_block` were calling it — populates
/// `model_field_ranks` for the model's tensor fields.
fn extract_first_model_forward(
    src: &str,
    cfg: Option<FusedCeDecoratorConfig>,
    // Model-field ranks keyed as (model_type, field) -> rank.
    model_field_ranks: &[(&'static str, &'static str, usize)],
) -> Option<nsl_codegen::wengert::WengertList> {
    let mut interner_box = Box::new(Interner::new());
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner_box);
    let parsed = nsl_parser::parse(&tokens, &mut interner_box);
    let module = parsed.module;
    let interner_static: &'static Interner = Box::leak(interner_box);

    // Locate the first ModelDef and its `forward` method.  Models keep
    // fields + methods in a single `members` vec, so we scan for the
    // `Method` variant matching `forward`.
    let (model_type, forward) = module
        .stmts
        .iter()
        .find_map(|s| match &s.kind {
            nsl_ast::stmt::StmtKind::ModelDef(model) => {
                let model_name =
                    interner_static.resolve(model.name.0).unwrap_or("?").to_string();
                let forward = model.members.iter().find_map(|m| match m {
                    nsl_ast::decl::ModelMember::Method(fn_def, _) => {
                        if interner_static.resolve(fn_def.name.0).unwrap_or("") == "forward" {
                            Some(fn_def.clone())
                        } else {
                            None
                        }
                    }
                    _ => None,
                })?;
                Some((model_name, forward))
            }
            _ => None,
        })
        .expect("test src must contain a model with a `forward` method");

    let mut extractor =
        WengertExtractor::new(interner_static).with_fused_ce_config(cfg);
    // Mirror what stmt.rs::compile_train_block installs so the
    // MemberAccess arm can resolve `self.w`.
    let mut ranks: std::collections::HashMap<
        String,
        std::collections::HashMap<String, usize>,
    > = std::collections::HashMap::new();
    for (model, field, rank) in model_field_ranks {
        ranks
            .entry((*model).to_string())
            .or_default()
            .insert((*field).to_string(), *rank);
    }
    extractor.set_model_field_ranks(ranks);

    // Set up the self context so `self.w` resolves via
    // `context_to_model_type["self"] = "Tiny"`.
    let self_sym = forward.params.first().expect("`self` param").name;
    extractor.register_model_instance(self_sym, &model_type);
    extractor.set_self_context(Some("self".to_string()));

    // Register the remaining step-body inputs.
    for param in forward.params.iter().skip(1) {
        extractor.register_input_with_rank(param.name, None);
    }

    let ok = extractor.extract_stmts(&forward.body.stmts);
    if !ok {
        return None;
    }
    extractor.finalize()
}

#[test]
fn rank_3_model_field_w_refuses_substitution() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_model_forward(
        MODEL_FIELD_MEMBER_ACCESS_SRC,
        Some(cfg),
        // Announce rank-3 for `Tiny.w` — mimicking what
        // `collection.rs` would record for `w: Tensor<[D, V, H]>`
        // or `w = zeros([D, V, H])`.
        &[("Tiny", "w", 3)],
    )
    .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        0,
        "reviewer-flagged gap: rank-3 W accessed via `self.w` must \
         refuse the fused substitution — the fused FFI would silently \
         stride through the rank-3 tensor as if it were [V, H]"
    );
    assert_eq!(
        count_composite_ce(&list),
        1,
        "composite CE must take over on the refused path"
    );
}

#[test]
fn rank_2_model_field_w_still_substitutes() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_model_forward(
        MODEL_FIELD_MEMBER_ACCESS_SRC,
        Some(cfg),
        &[("Tiny", "w", 2)],
    )
    .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        1,
        "rank-2 model-field W must still substitute — item 5's refusal \
         is scoped to provable non-2D"
    );
    assert_eq!(
        count_composite_ce(&list),
        0,
        "composite CE must NOT appear alongside the fused op"
    );
}

#[test]
fn rank_2_annotated_w_still_substitutes() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn_with_ranks(
        AUTO_SUB_SRC,
        Some(cfg),
        &[("w", 2)],
    )
    .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        1,
        "rank-2 W must still substitute — the item 5 check only refuses \
         provable-non-2D operands"
    );
    assert_eq!(
        count_composite_ce(&list),
        0,
        "composite CE must NOT also appear alongside the fused op"
    );
}

#[test]
fn no_transpose_pattern_deferred_to_v5() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: None,
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(NO_TRANSPOSE_SRC, Some(cfg))
        .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        0,
        "Sprint v4-3 deferral: Matmul(x, W) + bias → CE with W already \
         in [H, V] layout does NOT substitute (every fused-CE PTX emitter \
         indexes W as `W[v*H + h]`, requiring [V, H]; a parallel emitter \
         + new FFI symbol tracked v5+).  If a future sprint lands the \
         variant, update this assertion to `count_fused == 1`."
    );
    assert_eq!(
        count_composite_ce(&list),
        1,
        "no-transpose case falls through to composite CE (correct + safe)"
    );
}

// ─── Test 6: Sprint 4's explicit-call path still works ──────────────────

// CFTP v5 follow-on Finding 9 (LOW): all pre-existing tests in this
// file pass `dtype: None`, which maps to F32 via
// `fused_ce_dtype_for_compiler` — so the auto-substitution path's
// behaviour under fp16/bf16 dtype hints was never exercised at the
// wengert layer.  The two tests below close that gap by asserting the
// auto-substitution STILL fires under fp16 / bf16 hints (the lift in
// v5 should NOT have changed substitution behaviour — substitution is
// orthogonal to dtype, the dtype only affects downstream lowering).
//
// CFTP v6 Finding 13 (MEDIUM): these tests exercise the WengertExtractor
// only (`extract_first_fn` stops there); they do NOT invoke
// `compile_wengert_ops` and therefore do NOT exercise the v6
// `maybe_precision_cast_inputs` lowering.  The claim they pin is
// narrow: "auto-substitution recognition is dtype-orthogonal at the
// wengert layer".  The end-to-end cast-emission claim is pinned in
// `fused_lm_ce_e2e_{fp16,bf16}_activation.rs` (Phase 1 IR-shape
// assertions) and in `fused_linear_ce_precision_cast_lowering.rs` (the
// IR-emission contract for the cast wrappers themselves).

/// Narrow scope per Finding 13: the WengertExtractor sees `dtype=Bf16`
/// on the decorator config and STILL recognises the fused_linear_ce
/// pattern (substitution is dtype-orthogonal at the wengert layer).
/// This test does NOT exercise the v6 precision-cast lowering — see
/// the module comment block above.
#[test]
fn wengert_recognition_is_dtype_orthogonal_under_bf16_hint() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: Some(FusedCeDtypeHint::Bf16),
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(AUTO_SUB_SRC, Some(cfg))
        .expect("extraction must succeed under bf16 hint");
    assert_eq!(
        count_fused(&list),
        1,
        "auto-substitution must still fire when dtype=Bf16 — the v5 lift \
         only affects downstream wengert lowering, NOT the upstream \
         pattern recognition.  A regression here would mean the dtype \
         hint accidentally became a gate on substitution itself."
    );
    assert_eq!(
        count_composite_ce(&list),
        0,
        "bf16 hint must NOT cause a fallback to composite cross_entropy"
    );
}

/// Mirror of the bf16 test under fp16 — pins the same wengert-layer
/// dtype-orthogonal recognition (Finding 13: this is NOT an end-to-end
/// cast-emission test; see module comment).
#[test]
fn wengert_recognition_is_dtype_orthogonal_under_fp16_hint() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: Some(FusedCeDtypeHint::F16),
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(AUTO_SUB_SRC, Some(cfg))
        .expect("extraction must succeed under fp16 hint");
    assert_eq!(
        count_fused(&list),
        1,
        "auto-substitution must still fire when dtype=F16",
    );
    assert_eq!(
        count_composite_ce(&list),
        0,
        "fp16 hint must NOT cause a fallback to composite cross_entropy",
    );
}

#[test]
fn sprint4_explicit_call_path_still_works() {
    let cfg = FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
            dtype: None,
            train_block_stmt_id: nsl_ast::NodeId::dummy(),
    };
    let list = extract_first_fn(EXPLICIT_CALL_SRC, Some(cfg))
        .expect("extraction must succeed");
    assert_eq!(
        count_fused(&list),
        1,
        "explicit fused_linear_ce(...) call must still emit FusedLinearCe \
         (Sprint 4 path is independent of Sprint v3-1 auto-substitution)"
    );
    assert_eq!(
        count_composite_ce(&list),
        0,
        "explicit call path must NOT emit a composite CE"
    );
}
