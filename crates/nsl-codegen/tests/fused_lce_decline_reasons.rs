//! Item 6 — the fused linear-CE substitution must say WHY it declined.
//!
//! # What this file is defending
//!
//! `@fused_lm_ce(enabled = true)` exists to delete the `[batch*seq, vocab]`
//! logits-gradient surface (402 MB per step at V=49152, H=2048). Its
//! substitution declines for twelve distinct reasons, and before item 6 every
//! one of them fell through to `PrimalOp::CrossEntropyLoss` with no output
//! whatsoever.
//!
//! That mattered because the shapes that decline are not exotic: NSL's own
//! `models/coder50m/model.nsl` has a biasless weight-tied head, and its
//! pretraining scripts reshape `[B,S,V]` logits before the loss. Both used to
//! decline. A user who added the decorator to either got a clean compile, no
//! diagnostic, and the full composite path — while believing the fused kernel
//! was live. That is the failure shape `feedback_deferral_must_refuse` names:
//! worse than a refusal, because it looks like success.
//!
//! Sprint 2.5 flipped both of those shapes into substitutions (see the
//! BEHAVIOUR FLIP notes on the first two tests), and `coder50m/pretrain.nsl`
//! now carries the decorator. The fixtures below are kept as the regression
//! pin on that flip; the remaining declines are the genuinely unmatchable
//! heads.
//!
//! # What is asserted here, and what is NOT
//!
//! These tests pin the DECLINE REASON at the extractor layer — that the right
//! `FusedLceDecline` variant comes back for each shape. Falling through to the
//! composite is still the extractor's behaviour and is still correct: the
//! composite path computes the same loss, just with the `[N, V]` surface. The
//! REFUSAL (turning "nothing fused" into a hard `CodegenError`) lives at the
//! train-block lowering in `stmt.rs` and is gated end-to-end by
//! `crates/nsl-cli/tests/fused_lm_ce_decline_gate.rs`.
//!
//! Splitting it this way is deliberate: the extractor is used by the WGGO
//! pre-pass too, where a decline is informational, so the policy decision
//! belongs at the lowering that knows a `train` block asked for the feature.

use nsl_codegen::source_ad::{FusedLceDecline, WengertExtractor};
use nsl_codegen::{FusedCeDecoratorConfig, FusedCeDtypeHint};
use nsl_lexer::Interner;

/// Fully-populated decorator config — the state in which a decline is a
/// broken promise rather than an opt-out.
fn enabled_cfg() -> FusedCeDecoratorConfig {
    FusedCeDecoratorConfig {
        enabled: true,
        vocab_tile: Some(1024),
        vocab_size: Some(4096),
        hidden_size: Some(128),
        batch_size: Some(2),
        seq_len: Some(32),
        dtype: Some(FusedCeDtypeHint::F32),
        train_block_stmt_id: nsl_ast::NodeId::dummy(),
    }
}

/// Extract the first `fn` in `src` and return the declines the extractor
/// recorded, plus how many substitutions succeeded.
///
/// `ranks` threads per-parameter rank annotations through
/// `register_input_with_rank`, mirroring what the train-block lowering does
/// from the semantic `variable_types` table.
fn declines_for(
    src: &str,
    cfg: Option<FusedCeDecoratorConfig>,
    ranks: &[(&str, usize)],
) -> (Vec<FusedLceDecline>, usize) {
    let mut interner_box = Box::new(Interner::new());
    let (tokens, _) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner_box);
    let parsed = nsl_parser::parse(&tokens, &mut interner_box);
    let module = parsed.module;
    // SAFETY: leak the interner so borrows live for the test duration. Same
    // one-off pattern the neighbouring auto-substitution tests use.
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
    assert!(
        extractor.extract_stmts(&fn_def.body.stmts),
        "fixture must extract cleanly — a failed extraction would make the \
         decline assertion below vacuous (no cross_entropy arm reached)"
    );
    (
        extractor.fused_lce_declines().to_vec(),
        extractor.fused_lce_substitution_count(),
    )
}

/// Assert exactly one decline came back, and that it is `expected`.
fn assert_single_decline(
    src: &str,
    ranks: &[(&str, usize)],
    expected: &FusedLceDecline,
) {
    let (declines, subs) = declines_for(src, Some(enabled_cfg()), ranks);
    assert_eq!(
        subs, 0,
        "fixture was supposed to DECLINE but the matcher substituted {subs} \
         time(s); the decline assertion below would be testing nothing"
    );
    assert_eq!(
        declines.len(),
        1,
        "expected exactly one decline, got {declines:?}"
    );
    assert_eq!(
        &declines[0], expected,
        "wrong decline reason.\n  expected: {}\n  actual:   {}",
        expected.describe(),
        declines[0].describe()
    );
}

// ─── The two shapes every production coder model actually has ───────────────

/// `models/coder50m/model.nsl`'s `forward_core` — biasless weight-tied head.
/// `x @ W.transpose(0, 1)` produces a bare `Matmul`, not the `Add` the
/// matcher used to require.
const BIASLESS_HEAD_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 1)
    let logits = matmul(x, w_t)
    return cross_entropy(logits, targets)
"#;

#[test]
fn biasless_head_substitutes_with_no_decline() {
    // BEHAVIOUR FLIP (Sprint 2.5): the biasless head — previously the
    // second most common decline — substitutes with has_bias=false via
    // the GEMM-chunked runtime path.
    let (declines, subs) = declines_for(BIASLESS_HEAD_SRC, Some(enabled_cfg()), &[]);
    assert_eq!(subs, 1, "Sprint 2.5: the biasless head substitutes");
    assert!(
        declines.is_empty(),
        "no decline may be recorded for a successful biasless match, got \
         {declines:?}"
    );
}

/// `models/coder50m/pretrain.nsl`'s step body — a reshape between the head
/// and the loss. The producer of `logits` is then the reshape, not the head
/// `Add`.
const RESHAPE_BEFORE_CE_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 1)
    let logits3d = matmul(x, w_t) + bias
    let logits = logits3d.reshape([64, 4096])
    return cross_entropy(logits, targets)
"#;

#[test]
fn reshape_between_head_and_ce_is_seen_through() {
    // BEHAVIOUR FLIP (Sprint 2.5): the flatten between a 3-D head and the
    // loss — previously THE single most common decline in real code — is
    // now seen through (up to two chained reshapes) and substitutes with
    // x_rank3=true.
    let (declines, subs) =
        declines_for(RESHAPE_BEFORE_CE_SRC, Some(enabled_cfg()), &[]);
    assert_eq!(subs, 1, "Sprint 2.5: the reshape is seen through");
    assert!(
        declines.is_empty(),
        "no decline may be recorded for a successful see-through, got \
         {declines:?}"
    );
}

/// A reshape whose INPUT chain still cannot match (bare parameter under the
/// reshape) — see-through must descend and then report the INNER producer,
/// keeping the diagnostic actionable.
const RESHAPE_OF_PARAM_SRC: &str = r#"
fn step(logits3d: Tensor, targets: Tensor) -> Tensor:
    let logits = logits3d.reshape([64, 4096])
    return cross_entropy(logits, targets)
"#;

#[test]
fn reshape_of_unmatchable_chain_declines_with_inner_producer() {
    assert_single_decline(
        RESHAPE_OF_PARAM_SRC,
        &[],
        &FusedLceDecline::LogitsProducerNotAdd {
            producer: "Input(\"logits3d\")".to_string(),
        },
    );
}

// ─── The remaining reachable bail points ───────────────────────────────────

/// `cross_entropy` on a bare parameter — the head chain is computed
/// somewhere this extractor cannot see.
///
/// Note the decline is `LogitsProducerNotAdd { producer: Input(...) }`, NOT
/// `LogitsHasNoProducer`: a registered parameter DOES get a producing op
/// (`PrimalOp::Input`) on the tape. Writing this test against the
/// intuitive-but-wrong variant is what surfaced that, and is why
/// `LogitsHasNoProducer` is classified unreachable below.
const PLAIN_LOGITS_SRC: &str = r#"
fn step(logits: Tensor, targets: Tensor) -> Tensor:
    return cross_entropy(logits, targets)
"#;

#[test]
fn logits_as_a_parameter_declines_naming_the_binding() {
    assert_single_decline(
        PLAIN_LOGITS_SRC,
        &[],
        &FusedLceDecline::LogitsProducerNotAdd {
            producer: "Input(\"logits\")".to_string(),
        },
    );
}

/// A pre-transposed `[H, V]` weight: the matmul's right operand is a leaf, so
/// there is no transpose to recognise.
const NO_TRANSPOSE_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let logits = matmul(x, w) + bias
    return cross_entropy(logits, targets)
"#;

#[test]
fn pre_transposed_weight_declines_with_not_transposed() {
    assert_single_decline(
        NO_TRANSPOSE_SRC,
        &[],
        &FusedLceDecline::WeightNotTransposed,
    );
}

/// `Add(Matmul, Matmul)` — whichever operand lands in the bias slot would be
/// dereferenced as a dense `[V]` vector. Deliberate guard.
const AMBIGUOUS_ADD_SRC: &str = r#"
fn step(x1: Tensor, w1: Tensor, x2: Tensor, w2: Tensor, targets: Tensor) -> Tensor:
    let w1t = transpose(w1, 0, 1)
    let w2t = transpose(w2, 0, 1)
    let logits = matmul(x1, w1t) + matmul(x2, w2t)
    return cross_entropy(logits, targets)
"#;

#[test]
fn ambiguous_add_declines_with_high_rank_bias_slot() {
    assert_single_decline(
        AMBIGUOUS_ADD_SRC,
        &[],
        &FusedLceDecline::BiasSlotIsHighRank {
            producer: "Matmul".to_string(),
        },
    );
}

/// Neither `Add` operand is a matmul — no LM-head GEMM to absorb.
const NO_MATMUL_SRC: &str = r#"
fn step(h: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let logits = h + bias
    return cross_entropy(logits, targets)
"#;

#[test]
fn add_without_a_matmul_declines_with_no_matmul_operand() {
    assert_single_decline(
        NO_MATMUL_SRC,
        &[],
        &FusedLceDecline::NoMatmulOperand,
    );
}

/// A transpose over dims other than the last two. `[V, H]` indexing cannot
/// honour it.
const WRONG_DIM_TRANSPOSE_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 2)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;

#[test]
fn wrong_dim_transpose_declines_and_renders_dims_readably() {
    let (declines, subs) =
        declines_for(WRONG_DIM_TRANSPOSE_SRC, Some(enabled_cfg()), &[]);
    assert_eq!(subs, 0, "transpose(0, 2) must not substitute");
    assert_eq!(declines.len(), 1, "expected one decline, got {declines:?}");
    assert_eq!(
        declines[0],
        FusedLceDecline::TransposeNotLastTwoDims {
            dims: "(0, 2)".to_string()
        },
        "got: {}",
        declines[0].describe()
    );
}

/// CFTP v10 item 5 — an annotated rank-3 `W` (MoE expert stack) is refused so
/// the kernel never strides it as `[V, H]`.
const RANK3_W_SRC: &str = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, -2, -1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;

#[test]
fn rank_3_weight_declines_with_rank_not_2() {
    assert_single_decline(
        RANK3_W_SRC,
        &[("w", 3)],
        &FusedLceDecline::WeightRankNot2 { rank: 3 },
    );
}

/// The canonical head, but the decorator is missing shape hints. The
/// substitution cannot proceed and the message must name which are absent.
#[test]
fn missing_shape_hints_are_named_individually() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;
    let mut cfg = enabled_cfg();
    cfg.vocab_size = None;
    cfg.seq_len = None;
    let (declines, subs) = declines_for(src, Some(cfg), &[]);
    assert_eq!(subs, 0, "missing hints must block the substitution");
    assert_eq!(
        declines,
        vec![FusedLceDecline::MissingShapeHints {
            missing: vec!["vocab_size", "seq_len"]
        }],
        "the decline must list exactly the absent hints"
    );
    // Assert against the LIST portion only. The message goes on to show a
    // complete example decorator, which necessarily names all four hints —
    // checking the whole string would pass for a message that listed every
    // hint as missing.
    let msg = declines[0].describe();
    let list = msg
        .split("hint(s): ")
        .nth(1)
        .and_then(|rest| rest.split('.').next())
        .unwrap_or_else(|| panic!("message lost its hint list: {msg}"));
    assert!(
        list.contains("vocab_size") && list.contains("seq_len"),
        "the list must name both missing hints, got {list:?}"
    );
    assert!(
        !list.contains("hidden_size") && !list.contains("batch_size"),
        "the list must not name hints that WERE supplied, got {list:?}"
    );
}

// ─── Anti-vacuity: a decline is only recorded when one was promised ─────────

/// The canonical head with a complete decorator substitutes, and records NO
/// decline. Without this the assertions above could all be satisfied by an
/// extractor that records a decline unconditionally.
#[test]
fn canonical_head_substitutes_and_records_no_decline() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;
    let (declines, subs) = declines_for(src, Some(enabled_cfg()), &[]);
    assert_eq!(subs, 1, "canonical head must substitute");
    assert!(
        declines.is_empty(),
        "a successful substitution must record no decline, got {declines:?}"
    );
}

/// With the decorator ABSENT, the composite path is what the program asked
/// for — declining is not a broken promise and must not be recorded.
/// Otherwise every ordinary `cross_entropy` in the codebase would accumulate
/// a decline and the train-block refusal would fire on programs that never
/// mentioned the feature.
#[test]
fn no_decorator_records_no_decline() {
    let (declines, subs) = declines_for(BIASLESS_HEAD_SRC, None, &[]);
    assert_eq!(subs, 0);
    assert!(
        declines.is_empty(),
        "no decorator means no promise to break, got {declines:?}"
    );
}

/// Same for an explicitly disabled decorator.
#[test]
fn disabled_decorator_records_no_decline() {
    let mut cfg = enabled_cfg();
    cfg.enabled = false;
    let (declines, subs) = declines_for(BIASLESS_HEAD_SRC, Some(cfg), &[]);
    assert_eq!(subs, 0);
    assert!(
        declines.is_empty(),
        "enabled = false is a deliberate opt-out, got {declines:?}"
    );
}

// ─── Coverage bookkeeping over the taxonomy ────────────────────────────────

/// Whether a decline variant is exercised by a fixture above, or is
/// structurally unreachable from NSL source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Coverage {
    /// A fixture in this file produces it. The `&str` names the test.
    Exercised(&'static str),
    /// No NSL program can produce it; the arm is a defensive guard against a
    /// malformed tape. The `&str` says why it cannot be reached.
    Unreachable(&'static str),
}

/// Classify every variant.
///
/// The match is exhaustive with no `_` arm, so adding a bail point to
/// `FusedLceDecline` fails to compile HERE until it is classified. That is
/// the forcing function — it does not by itself prove the fixture exists,
/// which is what `every_exercised_variant_has_a_fixture` checks below.
fn classify_coverage(d: &FusedLceDecline) -> Coverage {
    match d {
        // VERIFIED, not assumed: `nsl build` on a fixture calling
        // `cross_entropy(logits, labels, labels)` fails with
        // "error: too many arguments: expected 2, got 3" and never reaches
        // codegen. The check is `crates/nsl-semantic/src/checker/ops.rs:300`
        // (`arg_types.len() > params.len() && !is_variadic`).
        //
        // Note this covers only the TOO-MANY direction; that file does not
        // reject too FEW arguments. A 1-arg `cross_entropy(logits)` would
        // reach the substitution arm — but `input_vars` is built from `args`,
        // so it declines as `Arity { args: 1 }` and, being the only CE call,
        // becomes a refusal. That is a behaviour CHANGE from silently
        // compiling, and it is deliberate: a 1-arg cross_entropy has no
        // targets and cannot be a working loss.
        FusedLceDecline::Arity { .. } => Coverage::Unreachable(
            "nsl-semantic checker/ops.rs:300 rejects a 3-arg cross_entropy              before codegen (verified by running nsl build); the 1-arg              direction is reachable but is a deliberate new refusal",
        ),
        FusedLceDecline::MissingShapeHints { .. } => {
            Coverage::Exercised("missing_shape_hints_are_named_individually")
        }
        // A defensive arm, not a reachable one: `find_producer` returns `None`
        // only for a VarId no op produced, and every VarId that reaches the
        // substitution came out of `extract_expr`, which always pushes a
        // producing op — a bare parameter gets `PrimalOp::Input`. Kept because
        // `find_producer` is `Option`-typed and the alternative is an
        // `expect` at a site where a malformed tape should diagnose, not
        // panic.
        FusedLceDecline::LogitsHasNoProducer => Coverage::Unreachable(
            "extract_expr always pushes a producing op; a bare parameter \
             produces PrimalOp::Input, so find_producer never returns None here",
        ),
        // Produced by RESHAPE_OF_PARAM_SRC and PLAIN_LOGITS_SRC. It was
        // named for `biasless_head_declines_with_producer_not_add` until
        // Sprint 2.5 flipped that fixture into a substitution and renamed it;
        // the claim outlived the test it cited, which the assertion in
        // `every_exercised_variant_has_a_fixture` cannot catch because it
        // only checks that SOMETHING produces the variant.
        FusedLceDecline::LogitsProducerNotAdd { .. } => {
            Coverage::Exercised("logits_as_a_parameter_declines_naming_the_binding")
        }
        // `PrimalOp::Add` is pushed only from the binary `+` arm of
        // `extract_expr`, which supplies exactly two inputs.
        FusedLceDecline::AddArity { .. } => Coverage::Unreachable(
            "PrimalOp::Add is only emitted from a binary +, always arity 2",
        ),
        FusedLceDecline::NoMatmulOperand => {
            Coverage::Exercised("add_without_a_matmul_declines_with_no_matmul_operand")
        }
        FusedLceDecline::BiasSlotIsHighRank { .. } => {
            Coverage::Exercised("ambiguous_add_declines_with_high_rank_bias_slot")
        }
        // VERIFIED: `matmul` is not a resolvable name in NSL source — `nsl
        // build` on a fixture using `matmul(a, b, c)` fails with
        // "error: undefined variable `matmul`". The only surface syntax that
        // produces `PrimalOp::Matmul` is the binary `@` operator, which is
        // 2-ary by construction. (The extractor recognises the NAME "matmul"
        // because the parser-level tests in this crate build tapes directly,
        // bypassing semantic analysis.)
        FusedLceDecline::MatmulArity { .. } => Coverage::Unreachable(
            "`matmul` is not a resolvable NSL name (verified by running nsl              build); PrimalOp::Matmul comes only from the binary @, always 2-ary",
        ),
        FusedLceDecline::WeightNotTransposed => {
            Coverage::Exercised("pre_transposed_weight_declines_with_not_transposed")
        }
        FusedLceDecline::TransposeNotLastTwoDims { .. } => {
            Coverage::Exercised("wrong_dim_transpose_declines_and_renders_dims_readably")
        }
        // `transpose(w, i, j)` always records `w` in input slot 0.
        FusedLceDecline::TransposeHasNoOperand => Coverage::Unreachable(
            "the transpose extraction always records its operand in slot 0",
        ),
        FusedLceDecline::WeightRankNot2 { .. } => {
            Coverage::Exercised("rank_3_weight_declines_with_rank_not_2")
        }
    }
}

/// One constructed sample per variant.
///
/// Kept in sync with the enum by `classify_coverage`'s exhaustive match: a
/// new variant breaks that match's compile, and the comment there points
/// here. This list cannot itself be compiler-verified as complete, so it is
/// a forcing function rather than a proof — stated plainly so nobody reads
/// more assurance into it than it carries.
fn all_variants() -> Vec<FusedLceDecline> {
    vec![
        FusedLceDecline::Arity { args: 3 },
        FusedLceDecline::MissingShapeHints {
            missing: vec!["vocab_size"],
        },
        FusedLceDecline::LogitsHasNoProducer,
        FusedLceDecline::LogitsProducerNotAdd {
            producer: "Matmul".to_string(),
        },
        FusedLceDecline::AddArity { inputs: 3 },
        FusedLceDecline::NoMatmulOperand,
        FusedLceDecline::BiasSlotIsHighRank {
            producer: "Matmul".to_string(),
        },
        FusedLceDecline::MatmulArity { inputs: 3 },
        FusedLceDecline::WeightNotTransposed,
        FusedLceDecline::TransposeNotLastTwoDims {
            dims: "(0, 2)".to_string(),
        },
        FusedLceDecline::TransposeHasNoOperand,
        FusedLceDecline::WeightRankNot2 { rank: 3 },
    ]
}

/// Every variant classified `Exercised` must actually be produced by one of
/// the fixtures above.
///
/// This is the check that stops the taxonomy from claiming coverage it does
/// not have: a variant can be added, classified `Exercised`, and pass
/// `classify_coverage`'s compile gate while no fixture reaches it. Here the
/// fixtures are re-run and the produced set compared against the claim.
#[test]
fn every_exercised_variant_has_a_fixture() {
    let cfg = || Some(enabled_cfg());
    let mut hint_cfg = enabled_cfg();
    hint_cfg.vocab_size = None;

    let mut produced: Vec<&'static str> = Vec::new();
    let mut record = |src: &str, c: Option<FusedCeDecoratorConfig>, ranks: &[(&str, usize)]| {
        for d in declines_for(src, c, ranks).0 {
            produced.push(d.discriminant_name());
        }
    };
    // BIASLESS_HEAD_SRC / RESHAPE_BEFORE_CE_SRC substitute since Sprint 2.5
    // and record no declines; RESHAPE_OF_PARAM_SRC keeps the see-through
    // decline path (inner producer) exercised.
    record(RESHAPE_OF_PARAM_SRC, cfg(), &[]);
    record(PLAIN_LOGITS_SRC, cfg(), &[]);
    record(NO_TRANSPOSE_SRC, cfg(), &[]);
    record(AMBIGUOUS_ADD_SRC, cfg(), &[]);
    record(NO_MATMUL_SRC, cfg(), &[]);
    record(WRONG_DIM_TRANSPOSE_SRC, cfg(), &[]);
    record(RANK3_W_SRC, cfg(), &[("w", 3)]);
    record(
        r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, 0, 1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#,
        Some(hint_cfg),
        &[],
    );

    let mut missing: Vec<String> = Vec::new();
    for v in all_variants() {
        if let Coverage::Exercised(test_name) = classify_coverage(&v)
            && !produced.contains(&v.discriminant_name())
        {
            missing.push(format!(
                "{} (claimed exercised by `{}`)",
                v.discriminant_name(),
                test_name
            ));
        }
    }
    assert!(
        missing.is_empty(),
        "these decline variants claim to be exercised but no fixture in this \
         file produces them: {missing:?}\nEither add a fixture, or reclassify \
         them in `classify_coverage` with the reason they are unreachable."
    );

    // Anti-vacuity: if the fixtures stopped producing declines entirely (a
    // refactor that silently stopped recording them), `produced` would be
    // empty and the loop above would pass for a taxonomy of all-Unreachable.
    // Pin the count so that collapse is a failure.
    assert_eq!(
        produced.len(),
        8,
        "expected exactly 8 declines across the fixtures, got {produced:?}. \
         Two fixtures share LogitsProducerNotAdd, so a per-discriminant check \
         alone cannot notice one of them silently ceasing to decline — pin the \
         count. Update this number deliberately when adding a fixture. \
         (Was 9 before Sprint 2.5 flipped the biasless + reshape fixtures \
         into substitutions.)"
    );
}

/// Every `Unreachable` classification must carry a non-empty reason, and no
/// variant may be classified `Unreachable` while a fixture still produces it
/// (that would mean the stated reason is wrong).
///
/// LIMITATION, stated plainly because it would otherwise read as stronger than
/// it is: this CANNOT verify unreachability. "No fixture produces it" is
/// circular when the reason no fixture produces it is that nobody wrote one.
/// The assurance for `Unreachable` comes from the reasons themselves, which
/// were checked by running `nsl build` against a program that would need to
/// reach each arm — see the citations in `classify_coverage`. A review
/// correctly flagged an earlier version where two of those reasons were
/// asserted rather than verified, and one of them was wrong about which layer
/// enforced the check.
#[test]
fn unreachable_classifications_are_justified_and_honest() {
    let mut produced: Vec<&'static str> = Vec::new();
    for (src, ranks) in [
        (BIASLESS_HEAD_SRC, &[][..]),
        (RESHAPE_BEFORE_CE_SRC, &[]),
        (PLAIN_LOGITS_SRC, &[]),
        (NO_TRANSPOSE_SRC, &[]),
        (AMBIGUOUS_ADD_SRC, &[]),
        (NO_MATMUL_SRC, &[]),
        (WRONG_DIM_TRANSPOSE_SRC, &[]),
        (RANK3_W_SRC, &[("w", 3)]),
    ] {
        for d in declines_for(src, Some(enabled_cfg()), ranks).0 {
            produced.push(d.discriminant_name());
        }
    }
    for v in all_variants() {
        if let Coverage::Unreachable(reason) = classify_coverage(&v) {
            assert!(
                !reason.trim().is_empty(),
                "{} is classified unreachable with no reason",
                v.discriminant_name()
            );
            assert!(
                !produced.contains(&v.discriminant_name()),
                "{} is classified UNREACHABLE ({reason}) but a fixture in this \
                 file produces it — the classification is wrong",
                v.discriminant_name()
            );
        }
    }
}

/// Every decline message must be usable: non-empty, and it must not leak the
/// raw `usize::MAX` sentinel that `encode_transpose_dim` uses for negative
/// dims (printing 18446744073709551615 where the user wrote `-1` would send
/// them hunting for a bug that isn't there).
#[test]
fn decline_messages_are_readable() {
    for v in all_variants() {
        let msg = v.describe();
        assert!(
            msg.len() > 20,
            "{} has a uselessly short message: {msg:?}",
            v.discriminant_name()
        );
        assert!(
            !msg.contains("18446744073709551615"),
            "{} leaks the usize::MAX negative-dim sentinel: {msg}",
            v.discriminant_name()
        );
    }
    // `transpose(w, -2, -1)` SUBSTITUTES (an accepted form) — asserted so the
    // sentinel range stays wired to the accept list, not just the printer.
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, -2, -1)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;
    let (declines, subs) = declines_for(src, Some(enabled_cfg()), &[]);
    assert_eq!(subs, 1, "transpose(-2, -1) is an accepted form");
    assert!(declines.is_empty());
}

/// The negative-dim SENTINEL DECODING, through a real decline.
///
/// `format_transpose_dims` turns `encode_transpose_dim`'s sentinels
/// (`usize::MAX` = -1, `usize::MAX - 1` = -2) back into what the user wrote.
/// Getting this wrong prints 18446744073709551615 and sends the reader hunting
/// for a bug that isn't there.
///
/// The `decline_messages_are_readable` test above CANNOT catch that: its
/// `TransposeNotLastTwoDims` sample carries a hardcoded `"(0, 2)"` string that
/// never passes through the decoder, and its `transpose(-2, -1)` case is an
/// accepted form producing no decline at all. `transpose(w, -1, 0)` is the
/// reachable combination — a sentinel in a REJECTED dim pair.
#[test]
fn negative_dim_sentinel_is_decoded_in_the_decline_message() {
    let src = r#"
fn step(x: Tensor, w: Tensor, bias: Tensor, targets: Tensor) -> Tensor:
    let w_t = transpose(w, -1, 0)
    let logits = matmul(x, w_t) + bias
    return cross_entropy(logits, targets)
"#;
    let (declines, subs) = declines_for(src, Some(enabled_cfg()), &[]);
    assert_eq!(subs, 0, "transpose(-1, 0) is not a last-two-dims transpose");
    assert_eq!(
        declines,
        vec![FusedLceDecline::TransposeNotLastTwoDims {
            dims: "(-1, 0)".to_string()
        }],
        "the sentinel must decode back to -1, not to a raw usize::MAX. Got: {}",
        declines
            .first()
            .map(|d| d.describe())
            .unwrap_or_else(|| "<none>".into())
    );
}
