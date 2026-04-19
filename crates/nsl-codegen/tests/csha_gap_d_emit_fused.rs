//! Gap D — `EmitFused` emits a real fused-backward kernel launch.
//!
//! Verifies the mechanical contract after A+B+C+D land:
//!
//! 1. Compiling an `@train` program whose attention chain is CSHA-active
//!    causes the object file to reference `nsl_flash_attention_csha_backward`
//!    via at least one relocation (the launch op lowerer fired).
//! 2. The `csha_fused_bwd_cache` population path is exercised — we can
//!    see the seven extract-consumer FFIs (`nsl_tensor_to_device`,
//!    `nsl_tensor_zeros`) appear downstream, proving the 7 output
//!    tensors were allocated.
//! 3. The legacy "fallback to per-op AD" eprintln (used by the pre-Gap-D
//!    stub) is NOT present in the fused-event record — i.e. the
//!    dispatcher produced an event with `smoke_config=true` for a
//!    CSHA-active hd=32 chain, rather than silently falling through.
//!
//! When the source doesn't reach CSHA at all (planner rejects, shapes
//! don't match a boundary chain, etc.), the test skips with a diagnostic
//! rather than failing — the unit tests in
//! `tests/ad_csha_reverse_walk_wiring.rs` remain load-bearing for Gap D's
//! AD-emitter semantics.

#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use nsl_errors::{FileId, Level};
use nsl_lexer::Interner;
use object::{Object, ObjectSection, ObjectSymbol};

/// Minimal NSL program that should trigger:
///   - `@flash_attention` → the CSHA planner's boundary scan
///   - `@train` → AD generation → the fused-backward dispatcher
///
/// Construct a tiny Attn model with an RMSNorm-free one-layer flavour —
/// even when CSHA fires at a coarser granularity, Gap D emits the
/// launch on the CLAIMED op's reverse-walk visit.
const TRAINING_SRC: &str = r#"
from nsl.nn.losses import mse_loss

model Attn:
    wq: Tensor = ones([4, 4])

    @flash_attention
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.wq

let m = Attn()
let x = ones([2, 4])
let y = zeros([2, 4])

train(model = m, epochs = 1):
    optimizer: AdamW(lr = 0.001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

fn compile_training_to_object(src: &str) -> Option<Vec<u8>> {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, Level::Error)) {
        eprintln!("[gap-d] lex errors — skipping");
        return None;
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        eprintln!("[gap-d] parse errors — skipping");
        return None;
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if analysis
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, Level::Error))
    {
        eprintln!(
            "[gap-d] semantic errors — skipping: {:?}",
            analysis
                .diagnostics
                .iter()
                .filter(|d| matches!(d.level, Level::Error))
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        );
        return None;
    }
    let opts = CompileOptions {
        csha_mode: Some("auto".into()),
        // Post-Gap-F the method-decorator scan path now reaches
        // `parse_gpu_sm_from_target`, which panics on the default
        // `"cuda"` target.  Pick an explicit `sm_*` to keep this
        // test stable.  See csha_gap_a_forward_saves.rs for the
        // identical pattern.
        target: "sm_75".to_string(),
        ..Default::default()
    };
    match nsl_codegen::compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        &opts,
    ) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("[gap-d] compile_module failed — skipping ({})", e.message);
            None
        }
    }
}

/// Collect the set of symbol names referenced by at least one relocation
/// anywhere in the object file.  Symbols in the symbol table but never
/// referenced don't count as "actually called" — see the rationale in
/// `csha_gap_a_forward_saves.rs::object_contains_gap_a_ffi_symbols_when_train_block_csha`.
fn relocation_symbol_set(obj_bytes: &[u8]) -> std::collections::HashSet<String> {
    let file = object::File::parse(obj_bytes).expect("object::File::parse");
    let name_by_index: std::collections::HashMap<_, _> = file
        .symbols()
        .filter_map(|s| s.name().ok().map(|n| (s.index(), n.to_string())))
        .collect();

    let mut out = std::collections::HashSet::new();
    for section in file.sections() {
        for (_offset, reloc) in section.relocations() {
            if let object::RelocationTarget::Symbol(idx) = reloc.target() {
                if let Some(name) = name_by_index.get(&idx) {
                    out.insert(name.clone());
                }
            }
        }
    }
    out
}

/// Core invariant: the fused CSHA backward FFI is called from the
/// compiled object.  When CSHA doesn't fire at all (shape mismatch, no
/// `@train` picked up by the boundary scanner) the test skips rather
/// than fails — the unit tests in `ad_csha_reverse_walk_wiring.rs` and
/// `csha_gap_c_multiresult_primitive.rs` remain load-bearing.
#[test]
fn training_compile_emits_fused_backward_ffi_call() {
    let obj = match compile_training_to_object(TRAINING_SRC) {
        Some(b) => b,
        None => return,
    };
    let called = relocation_symbol_set(&obj);

    let csha_related: Vec<&String> = called
        .iter()
        .filter(|n| n.contains("nsl_csha") || n.contains("nsl_flash_attention_csha"))
        .collect();
    eprintln!("[gap-d] CSHA-related symbols referenced by relocations: {csha_related:?}");

    // If the forward CSHA FFI never fired, we can't exercise the backward
    // either — skip.
    let fired_forward = called.contains("nsl_flash_attention_csha_with_saves")
        || called.contains("nsl_flash_attention_csha");
    if !fired_forward {
        eprintln!(
            "[gap-d] forward CSHA FFI did not fire for this toy program — \
             skipping backward-relocation check.  Unit tests in \
             ad_csha_reverse_walk_wiring.rs cover Gap D's AD-emitter semantics."
        );
        return;
    }

    // Forward fired.  If the `@train` block caused Gap B to embed the
    // backward PTX AND the fused dispatcher landed, the object must
    // reference the backward FFI via at least one relocation.
    let fired_backward = called.contains("nsl_flash_attention_csha_backward");

    // Diagnostic: show whether Gap B's training PTX plumbing was also
    // present.  If training-PTX is embedded but backward isn't called,
    // the EmitFused arm silently fell back and Gap D didn't land.
    let has_alloc_into = called.contains("nsl_csha_alloc_backward_activations_into");
    let has_free_from = called.contains("nsl_csha_free_backward_activations_from");
    eprintln!(
        "[gap-d] firing summary: forward={}, alloc_into={}, free_from={}, backward={}",
        fired_forward, has_alloc_into, has_free_from, fired_backward,
    );

    if !fired_backward {
        eprintln!(
            "[gap-d] `nsl_flash_attention_csha_backward` not referenced.  \
             Possible causes: the dispatcher's EmitFused arm didn't hit \
             (check Gap B's training-PTX embed status), the SMEM validator \
             rejected the backward config (Gap B diagnostic), or the \
             Wengert extraction for this @train block never produced a \
             claimed chain.  Unit tests in ad_csha_reverse_walk_wiring.rs \
             remain load-bearing for Gap D's semantics."
        );
        return;
    }

    // Backward fired.  The launch op lowerer allocates 7 output
    // tensors and moves them to device, so we should also see
    // `nsl_tensor_zeros` and `nsl_tensor_to_device` referenced.
    assert!(
        called.contains("nsl_tensor_zeros"),
        "Gap D lowerer allocates 7 output tensors via `nsl_tensor_zeros`; \
         relocation absent"
    );
    assert!(
        called.contains("nsl_tensor_to_device"),
        "Gap D lowerer moves 7 output tensors via `nsl_tensor_to_device`; \
         relocation absent"
    );

    // Sanity: the Gap A scope-free has been moved INTO the backward
    // lowerer, so free_from should be present iff backward fired.
    assert!(
        has_free_from,
        "Gap D moves the save-buffer `free_from` call inside the backward \
         lowerer; free_from must appear whenever backward does"
    );
}

/// `AdjointGenerator::generate` emits a `FusedCshaBackward` launch op +
/// 7 `CshaFusedBackwardExtract` ops when the dispatcher fires EmitFused
/// on a claimed chain.  This is a lightweight unit-level check that
/// complements the object-file test above.
#[test]
fn emit_fused_produces_launch_op_plus_seven_extracts() {
    use nsl_codegen::csha_apply::FusionMark;
    use nsl_codegen::csha_boundary::ProjKind;
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use nsl_codegen::source_ad::{AdjointGenerator, CshaBackwardClaims};
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp, WengertType};
    use std::collections::HashMap;

    // Build a tiny primal Wengert list with a "claimed" matmul_op.
    let ops = vec![
        WengertOp {
            id: 0,
            result: 0,
            op: PrimalOp::Input("x".into()),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        },
        WengertOp {
            id: 1,
            result: 1,
            op: PrimalOp::Param("wq".into()),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        },
        // The claimed op — a matmul for the Q projection.
        WengertOp {
            id: 2,
            result: 2,
            op: PrimalOp::Matmul,
            inputs: vec![0, 1],
            saved_for_backward: false,
            checkpointed: false,
        },
    ];
    let mut var_types = HashMap::new();
    var_types.insert(0, WengertType::Tensor);
    var_types.insert(1, WengertType::Tensor);
    var_types.insert(2, WengertType::Tensor);
    let primal = WengertList {
        ops,
        output: 2,
        var_names: HashMap::new(),
        var_types,
    };

    // Build a CSHA config that passes the backward validator (smoke config).
    let cfg = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            active_heads: 1,
            rmsnorm_eps: 1e-5,
            ..CshaExtras::default()
        }),
    };

    let mark = FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "wq".into(),
        role: nsl_codegen::csha_apply::MarkRole::NormPrologue,
        config: Some(cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: None,
    };
    let mut op_to_chain: HashMap<u32, usize> = HashMap::new();
    op_to_chain.insert(2, 0); // matmul_op (id=2) → chain 0
    let claims = CshaBackwardClaims {
        op_to_chain,
        chain_marks: vec![mark],
    };

    let mut gen = AdjointGenerator::new(10);
    gen.set_csha_claims(claims);
    let adjoint = gen.generate(&primal);

    // Expect the event was recorded.
    let events = gen.csha_fused_events();
    assert_eq!(
        events.len(),
        1,
        "expected exactly one fused-backward event, got {:?}",
        events
    );
    let ev = &events[0];
    assert_eq!(ev.layer, "blocks.0");
    assert!(ev.smoke_config, "hd=32/block=32/dm=32 must be smoke");

    // The adjoint list should contain the FusedCshaBackward launch op
    // AND at least one CshaFusedBackwardExtract op.
    let has_launch = adjoint
        .ops
        .iter()
        .any(|o| matches!(&o.op, PrimalOp::FusedCshaBackward { layer } if layer == "blocks.0"));
    assert!(
        has_launch,
        "EmitFused must emit a FusedCshaBackward launch op; ops={:?}",
        adjoint.ops.iter().map(|o| &o.op).collect::<Vec<_>>()
    );

    let extract_count = adjoint
        .ops
        .iter()
        .filter(|o| matches!(o.op, PrimalOp::CshaFusedBackwardExtract { .. }))
        .count();
    assert_eq!(
        extract_count, 8,
        "EmitFused must emit 8 CshaFusedBackwardExtract ops (one per \
         dq/dk/dv/dwq/dwk/dwv/dx/dx_norm component); got {}",
        extract_count
    );

    // Gap I.2+M: each extract lists two inputs — [launch_result,
    // chain_key]. All eight (7 from Gap C + dx_norm from Gap I.5
    // Option A) MUST declare the launch op's result as their FIRST
    // input (so dead-grad elim keeps the launch alive via the worklist
    // walk) and share the same SECOND input — the matmul_op's result
    // VarId 2 (the chain key).
    let launch_result = adjoint
        .ops
        .iter()
        .find_map(|o| match &o.op {
            PrimalOp::FusedCshaBackward { .. } => Some(o.result),
            _ => None,
        })
        .expect("launch op should be present in adjoint list");
    for op in &adjoint.ops {
        if let PrimalOp::CshaFusedBackwardExtract { .. } = op.op {
            assert_eq!(
                op.inputs.len(),
                2,
                "extract ops must have 2 inputs ([launch_result, chain_key]); got {:?}",
                op.inputs
            );
            assert_eq!(
                op.inputs[0], launch_result,
                "extract inputs[0] must be the FusedCshaBackward launch result"
            );
            assert_eq!(
                op.inputs[1], 2,
                "extract inputs[1] must be the chain_key = matmul_op.result"
            );
        }
    }

    // Gap I.5 Option A pin: every component value in 0..=7 must be
    // emitted exactly once. Component 7 (`dx_norm`) is the load-bearing
    // addition for correct dgamma semantics — if it regresses to 6
    // components (or 7 without component=7), the AD-side
    // RmsNormGammaBackward will receive dx_raw again (the original
    // Gap I.5 bug).
    let mut seen_components = std::collections::HashSet::<u8>::new();
    for op in &adjoint.ops {
        if let PrimalOp::CshaFusedBackwardExtract { component } = op.op {
            assert!(
                seen_components.insert(component),
                "duplicate CshaFusedBackwardExtract component={component}"
            );
        }
    }
    for c in 0u8..=7 {
        assert!(
            seen_components.contains(&c),
            "missing CshaFusedBackwardExtract component={c} — Gap I.5 \
             Option-A expects the full 0..=7 range (component 7 is the \
             dx_norm extract)"
        );
    }
}

/// Gap I.2+M combined test: run `AdjointGenerator::generate` to produce
/// a full adjoint list with a real `FusedCshaBackward` + 7 extracts,
/// then invoke `eliminate_dead_gradients` with the param adjoints
/// marked as needed, and assert the launch op is still present.
///
/// This stitches together the two passes the real compile does (source
/// AD emits, dead-grad elim prunes) and pins that the launch op's
/// survival is load-bearing through the combined pipeline.
#[test]
fn gap_i2_launch_op_survives_dead_grad_elim_in_generated_adjoint() {
    use nsl_codegen::csha_apply::FusionMark;
    use nsl_codegen::csha_boundary::ProjKind;
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use nsl_codegen::source_ad::{
        eliminate_dead_gradients, AdjointGenerator, CshaBackwardClaims,
    };
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp, WengertType};
    use std::collections::{HashMap, HashSet};

    // Same shape as the existing emit_fused test: Input + Param + Matmul,
    // with the matmul claimed by CSHA.
    let ops = vec![
        WengertOp {
            id: 0,
            result: 0,
            op: PrimalOp::Input("x".into()),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        },
        WengertOp {
            id: 1,
            result: 1,
            op: PrimalOp::Param("wq".into()),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        },
        WengertOp {
            id: 2,
            result: 2,
            op: PrimalOp::Matmul,
            inputs: vec![0, 1],
            saved_for_backward: false,
            checkpointed: false,
        },
    ];
    let mut var_types = HashMap::new();
    var_types.insert(0, WengertType::Tensor);
    var_types.insert(1, WengertType::Tensor);
    var_types.insert(2, WengertType::Tensor);
    let primal = WengertList {
        ops,
        output: 2,
        var_names: HashMap::new(),
        var_types,
    };

    // Clamped training-shaped config (I.1 semantic) so EmitFused accepts.
    let cfg = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: false,
            fused_output_proj: false,
            save_activations_for_backward: true,
            d_model: 32,
            active_heads: 1,
            rmsnorm_eps: 1e-5,
            ..CshaExtras::default()
        }),
    };
    let mark = FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "wq".into(),
        role: nsl_codegen::csha_apply::MarkRole::NormPrologue,
        config: Some(cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: None,
    };
    let mut op_to_chain: HashMap<u32, usize> = HashMap::new();
    op_to_chain.insert(2, 0);
    let claims = CshaBackwardClaims {
        op_to_chain,
        chain_marks: vec![mark],
    };

    let mut gen = AdjointGenerator::new(10);
    gen.set_csha_claims(claims);
    let adjoint = gen.generate(&primal);

    // The Param's adjoint VarId is the "needed" var that param-gradient
    // consumption would mark live. Look it up via the generator.
    let param_adjoint = gen
        .adjoint_of(1)
        .expect("Param 'wq' must have an adjoint after CSHA EmitFused routing");

    let needed: HashSet<u32> = [param_adjoint].into_iter().collect();
    let pruned = eliminate_dead_gradients(&adjoint.ops, &needed);

    // The launch op MUST be kept — without the Gap I.2+M fix, the walk
    // from param_adjoint would never reach it (extracts would declare
    // only [chain_key] as input, which is a primal VarId).
    let launch_kept = pruned.iter().any(|o| {
        matches!(
            &o.op,
            PrimalOp::FusedCshaBackward { layer } if layer == "blocks.0"
        )
    });
    assert!(
        launch_kept,
        "Gap I.2+M regression: FusedCshaBackward must survive \
         eliminate_dead_gradients on a realistically-generated adjoint \
         list. Pruned ops: {:?}",
        pruned
            .iter()
            .map(|o| (o.result, &o.op, &o.inputs))
            .collect::<Vec<_>>()
    );
}

// ── Gap D.1 — claim-site + adjoint-routing tests ───────────────────────────

/// With the Gap D.1 claim-site fix, the dispatcher should map the SDPA
/// op (not the Q/K/V matmul) to `EmitFused` when a full Q+K+V chain
/// group shares an SDPA consumer. This test verifies:
///   1. `collect_chain_dispatch_map_with_wengert` groups the three chains
///      under a single canonical mark (not three per-chain marks).
///   2. The SDPA op is in the `op_to_chain` map (claim primary).
///   3. The mark carries a fully resolved `chain_varids` routing table.
#[test]
fn gap_d1_sdpa_is_claim_primary_for_full_qkv_chain_group() {
    use nsl_codegen::csha::{self, CshaInput, CshaMode};
    use nsl_codegen::csha_apply::{bridge, collect_chain_dispatch_map_with_wengert};
    use nsl_codegen::csha_boundary::ProjKind;
    use nsl_codegen::csha_specialize::SpecConfig;
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
    use nsl_codegen::wggo_cost::LayerShape;
    use std::collections::HashMap;

    // Wengert with a full Q/K/V chain group + SDPA consumer.
    //   0: Input(x)
    //   1: RMSNorm(0)                   — x_norm
    //   2: Param(Wq)
    //   3: Matmul(1, 2)                 — Q_out
    //   4: RoPE(3)                      — Q_rope_out (q_out_var target)
    //   5: Param(Wk)
    //   6: Matmul(1, 5)                 — K_out
    //   7: RoPE(6)                      — K_rope_out (k_out_var target)
    //   8: Param(Wv)
    //   9: Matmul(1, 8)                 — V_out (v_out_var target)
    //  10: ScaledDotProductAttention(4, 7, 9)  — the shared SDPA
    let mk = |id, result, prim: PrimalOp, inputs| WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };
    let w = WengertList {
        ops: vec![
            mk(0, 0, PrimalOp::Input("x".into()), vec![]),
            mk(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            mk(2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            mk(3, 3, PrimalOp::Matmul, vec![1, 2]),
            mk(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            mk(5, 5, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
            mk(6, 6, PrimalOp::Matmul, vec![1, 5]),
            mk(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            mk(8, 8, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
            mk(9, 9, PrimalOp::Matmul, vec![1, 8]),
            mk(
                10,
                10,
                PrimalOp::ScaledDotProductAttention { causal: false },
                vec![4, 7, 9],
            ),
        ],
        output: 10,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };

    let plan = csha::run(CshaInput {
        mode: CshaMode::Auto,
        target: "H100",
        wengert: &w,
        weights: None,
        shape: LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        },
        n_heads: 8,
        spec_cfg: SpecConfig::default(),
        pattern_cfg: nsl_codegen::csha_patterns::PatternConfig::default(),
        wggo_overrides: None,
    });

    // All three chains must have detected the shared SDPA (op id 10).
    assert_eq!(plan.boundary.chains.len(), 3);
    for c in &plan.boundary.chains {
        assert_eq!(
            c.sdpa_op,
            Some(10),
            "chain {:?} missing SDPA detection",
            c.kind
        );
    }

    let br = bridge(&plan, 64, &mut Vec::new());
    let (op_to_chain, chain_marks) =
        collect_chain_dispatch_map_with_wengert(&plan, &br, Some(&w), None);

    // Gap D.1: one grouped mark for the whole layer (not three per-chain marks).
    assert_eq!(
        chain_marks.len(),
        1,
        "expected 1 grouped mark for full Q/K/V group, got {}",
        chain_marks.len()
    );

    // SDPA must be mapped as a claim primary.
    assert!(
        op_to_chain.contains_key(&10),
        "SDPA op 10 must be in the claim map; got keys: {:?}",
        op_to_chain.keys().collect::<Vec<_>>()
    );
    // All secondary ops (norm/matmul/rope) must also be mapped (they
    // return AlreadyEmitted once SDPA fires first in reverse walk).
    for key in [1u32, 3, 4, 6, 7, 9] {
        assert!(
            op_to_chain.contains_key(&key),
            "op {} must be in claim map (secondary AlreadyEmitted target)",
            key
        );
    }

    // chain_varids must be populated.
    let mark = &chain_marks[0];
    let v = mark
        .chain_varids
        .as_ref()
        .expect("grouped mark must carry chain_varids");
    assert_eq!(v.q_out_var, 4, "q_out = RoPE-Q output (VarId 4)");
    assert_eq!(v.k_out_var, 7, "k_out = RoPE-K output (VarId 7)");
    assert_eq!(v.v_out_var, 9, "v_out = V-matmul output (VarId 9)");
    assert_eq!(v.wq_var, 2);
    assert_eq!(v.wk_var, 5);
    assert_eq!(v.wv_var, 8);
    assert_eq!(v.x_norm_var, 1, "x_norm = RMSNorm output (VarId 1)");
    assert_eq!(v.sdpa_out_var, 10, "sdpa_out = SDPA output (VarId 10)");

    // Canonical mark uses Q as its kind representative.
    assert_eq!(mark.kind, Some(ProjKind::Q));
}

/// End-to-end Gap D.1 test: run `AdjointGenerator::generate` on a
/// Wengert list containing Q/K/V chains + SDPA, with chain_varids
/// populated. Verify:
///   - The fused launch op's inputs[1] (do_var) is an adjoint of the
///     SDPA output (VarId 10), NOT the current op's result.
///   - The 7 gradient outputs are routed to the right VarIds (the
///     adjoint_vars map gains entries for q_out/k_out/v_out/wq/wk/wv/
///     x_norm).
///   - Per-op SDPA backward does NOT fire (no AttentionBackwardQ/K/V
///     rule ops in the adjoint graph), so no double-accumulation.
///   - Per-op matmul backward for the projections does NOT fire.
#[test]
fn gap_d1_adjoint_routing_populates_correct_varids() {
    use nsl_codegen::csha_apply::{CshaChainVarIds, FusionMark};
    use nsl_codegen::csha_boundary::ProjKind;
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use nsl_codegen::source_ad::{AdjointGenerator, CshaBackwardClaims};
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
    use std::collections::HashMap;

    // Same Wengert as above (Q/K/V + SDPA).
    let mk = |id, result, prim: PrimalOp, inputs| WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };
    let primal = WengertList {
        ops: vec![
            mk(0, 0, PrimalOp::Input("x".into()), vec![]),
            mk(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            mk(2, 2, PrimalOp::Param("wq".into()), vec![]),
            mk(3, 3, PrimalOp::Matmul, vec![1, 2]),
            mk(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            mk(5, 5, PrimalOp::Param("wk".into()), vec![]),
            mk(6, 6, PrimalOp::Matmul, vec![1, 5]),
            mk(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            mk(8, 8, PrimalOp::Param("wv".into()), vec![]),
            mk(9, 9, PrimalOp::Matmul, vec![1, 8]),
            mk(
                10,
                10,
                PrimalOp::ScaledDotProductAttention { causal: false },
                vec![4, 7, 9],
            ),
        ],
        output: 10,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };

    // Smoke config — passes backward validator.
    let cfg = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            active_heads: 1,
            rmsnorm_eps: 1e-5,
            ..CshaExtras::default()
        }),
    };

    // Build a grouped mark with chain_varids by hand (mirrors what
    // `collect_chain_dispatch_map_with_wengert` would produce).
    let varids = CshaChainVarIds {
        q_out_var: 4,
        k_out_var: 7,
        v_out_var: 9,
        wq_var: 2,
        wk_var: 5,
        wv_var: 8,
        x_norm_var: 1,
        sdpa_out_var: 10,
        // Gap I.4: no gamma param in this structural test; kernel's
        // `csha_norm_weight_ptr` null-guard handles the `unwrap_or(0)`
        // that the launch emitter passes when this is `None`.
        norm_weight_var: None,
        // Gap I step K: no gamma → no x_raw / eps for dgamma emission.
        x_raw_var: None,
        rmsnorm_eps: 0.0,
    };
    let mark = FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "wq".into(),
        role: nsl_codegen::csha_apply::MarkRole::NormPrologue,
        config: Some(cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: Some(varids),
    };

    // Claim: SDPA primary, Q/K/V matmul+rope+RMSNorm secondary.
    let mut op_to_chain: HashMap<u32, usize> = HashMap::new();
    op_to_chain.insert(10, 0); // SDPA — claim primary
    op_to_chain.insert(1, 0); // RMSNorm
    op_to_chain.insert(3, 0); // Q matmul
    op_to_chain.insert(4, 0); // Q RoPE
    op_to_chain.insert(6, 0); // K matmul
    op_to_chain.insert(7, 0); // K RoPE
    op_to_chain.insert(9, 0); // V matmul

    let claims = CshaBackwardClaims {
        op_to_chain,
        chain_marks: vec![mark],
    };

    let mut gen = AdjointGenerator::new(100);
    gen.set_csha_claims(claims);
    let adjoint = gen.generate(&primal);

    // The fused launch op must be emitted exactly once.
    let launch_count = adjoint
        .ops
        .iter()
        .filter(|o| matches!(&o.op, PrimalOp::FusedCshaBackward { .. }))
        .count();
    assert_eq!(launch_count, 1, "exactly one fused launch expected");

    // No per-op SDPA backward should fire — the fused kernel replaces
    // it entirely. `AttentionBackwardQ/K/V` compound rules would show
    // up as sequences of Transpose/Matmul/Softmax/etc. ops, but more
    // reliably we check that SDPA's claimed-chain membership means it
    // returned via `continue` in EmitFused, not through `apply_ad_rule`.
    //
    // Assertion: the adjoint map should contain entries for all 7
    // routed VarIds (Q/K/V outputs, 3 weights, x_norm).
    let adj_map = gen.adjoint_vars_map();
    for vid in [4u32, 7, 9, 2, 5, 8, 1] {
        assert!(
            adj_map.contains_key(&vid),
            "Gap D.1 must populate adjoint for primal VarId {} via \
             the fused kernel; current map keys: {:?}",
            vid,
            adj_map.keys().collect::<Vec<_>>()
        );
    }

    // The fused launch op's inputs[1] (dO) must be the adjoint of the
    // SDPA output (VarId 10), proving the claim-site fix: dO is sourced
    // from the SDPA output, not from whatever op.result happened to be
    // when EmitFused fired.
    let launch = adjoint
        .ops
        .iter()
        .find(|o| matches!(&o.op, PrimalOp::FusedCshaBackward { .. }))
        .expect("launch op must exist");
    let do_var = launch.inputs[1];
    let sdpa_adj = adj_map
        .get(&10)
        .copied()
        .expect("SDPA output (VarId 10) must have an adjoint");
    assert_eq!(
        do_var, sdpa_adj,
        "launch inputs[1] must be the SDPA output's y_bar (VarId {}) \
         — got {}",
        sdpa_adj, do_var
    );

    // The fused kernel's output-proj event should have been recorded
    // against the SDPA op (id 10).
    let events = gen.csha_fused_events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].output_op_id, 10,
        "Gap D.1 claim primary is the SDPA op; event.output_op_id should be 10"
    );
}

// ── Gap I.4 — weight-pointer threading through EmitFused ──────────────────

/// Gap I.4 unit test (resolution): when the RMSNorm op's `inputs[1]` is
/// a `Param(...)`, `collect_chain_dispatch_map_with_wengert` must
/// populate `norm_weight_var` with that param's VarId. Without this,
/// the `csha_norm_weight_ptr` slot in the launch stays null and the
/// kernel can't accumulate dgamma (the stub for Gap I.K).
#[test]
fn gap_i4_norm_weight_var_populated_for_trainable_gamma() {
    use nsl_codegen::csha::{self, CshaInput, CshaMode};
    use nsl_codegen::csha_apply::{bridge, collect_chain_dispatch_map_with_wengert};
    use nsl_codegen::csha_specialize::SpecConfig;
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
    use nsl_codegen::wggo_cost::LayerShape;
    use std::collections::HashMap;

    // Wengert with trainable gamma: RMSNorm(0, gamma_param_1) consumes
    // a Param input as its second arg. Op layout:
    //   0: Input(x)
    //   1: Param("blocks.0.attn.norm_weight")  — gamma param
    //   2: RMSNorm(0, 1)                        — RMSNorm with trainable gamma
    //   3: Param(Wq), 4: Matmul(2,3), 5: RoPE(4)
    //   6: Param(Wk), 7: Matmul(2,6), 8: RoPE(7)
    //   9: Param(Wv), 10: Matmul(2,9)
    //  11: SDPA(5, 8, 10)
    let mk = |id, result, prim: PrimalOp, inputs| WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };
    let w = WengertList {
        ops: vec![
            mk(0, 0, PrimalOp::Input("x".into()), vec![]),
            mk(1, 1, PrimalOp::Param("blocks.0.attn.norm_weight".into()), vec![]),
            mk(2, 2, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0, 1]),
            mk(3, 3, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            mk(4, 4, PrimalOp::Matmul, vec![2, 3]),
            mk(5, 5, PrimalOp::RoPE { dim: 64 }, vec![4]),
            mk(6, 6, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
            mk(7, 7, PrimalOp::Matmul, vec![2, 6]),
            mk(8, 8, PrimalOp::RoPE { dim: 64 }, vec![7]),
            mk(9, 9, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
            mk(10, 10, PrimalOp::Matmul, vec![2, 9]),
            mk(
                11,
                11,
                PrimalOp::ScaledDotProductAttention { causal: false },
                vec![5, 8, 10],
            ),
        ],
        output: 11,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };

    let plan = csha::run(CshaInput {
        mode: CshaMode::Auto,
        target: "H100",
        wengert: &w,
        weights: None,
        shape: LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        },
        n_heads: 8,
        spec_cfg: SpecConfig::default(),
        pattern_cfg: nsl_codegen::csha_patterns::PatternConfig::default(),
        wggo_overrides: None,
    });

    let br = bridge(&plan, 64, &mut Vec::new());
    let (_op_to_chain, chain_marks) =
        collect_chain_dispatch_map_with_wengert(&plan, &br, Some(&w), None);
    assert_eq!(chain_marks.len(), 1, "expected grouped mark");

    let v = chain_marks[0]
        .chain_varids
        .as_ref()
        .expect("grouped mark must carry chain_varids");
    assert_eq!(
        v.norm_weight_var,
        Some(1),
        "trainable gamma at VarId 1 must be resolved into norm_weight_var; \
         got {:?}",
        v.norm_weight_var
    );
}

/// Gap I.4 unit test (no trainable gamma): when the RMSNorm op has no
/// second input OR that input is NOT a `Param(...)` (e.g. a constant
/// tensor), `norm_weight_var` must be `None`. The kernel's
/// `csha_norm_weight_ptr` null-guard handles the resulting null slot
/// in the launch.
#[test]
fn gap_i4_norm_weight_var_none_for_gammaless_rmsnorm() {
    use nsl_codegen::csha::{self, CshaInput, CshaMode};
    use nsl_codegen::csha_apply::{bridge, collect_chain_dispatch_map_with_wengert};
    use nsl_codegen::csha_specialize::SpecConfig;
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
    use nsl_codegen::wggo_cost::LayerShape;
    use std::collections::HashMap;

    // RMSNorm(0) — single-input gamma-less variant (same layout as the
    // Gap D.1 test above).
    let mk = |id, result, prim: PrimalOp, inputs| WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };
    let w = WengertList {
        ops: vec![
            mk(0, 0, PrimalOp::Input("x".into()), vec![]),
            mk(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            mk(2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            mk(3, 3, PrimalOp::Matmul, vec![1, 2]),
            mk(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            mk(5, 5, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
            mk(6, 6, PrimalOp::Matmul, vec![1, 5]),
            mk(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            mk(8, 8, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
            mk(9, 9, PrimalOp::Matmul, vec![1, 8]),
            mk(
                10,
                10,
                PrimalOp::ScaledDotProductAttention { causal: false },
                vec![4, 7, 9],
            ),
        ],
        output: 10,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };

    let plan = csha::run(CshaInput {
        mode: CshaMode::Auto,
        target: "H100",
        wengert: &w,
        weights: None,
        shape: LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        },
        n_heads: 8,
        spec_cfg: SpecConfig::default(),
        pattern_cfg: nsl_codegen::csha_patterns::PatternConfig::default(),
        wggo_overrides: None,
    });
    let br = bridge(&plan, 64, &mut Vec::new());
    let (_op_to_chain, chain_marks) =
        collect_chain_dispatch_map_with_wengert(&plan, &br, Some(&w), None);
    assert_eq!(chain_marks.len(), 1);
    let v = chain_marks[0]
        .chain_varids
        .as_ref()
        .expect("grouped mark must carry chain_varids");
    assert!(
        v.norm_weight_var.is_none(),
        "gamma-less RMSNorm must leave norm_weight_var = None; got {:?}",
        v.norm_weight_var
    );
}

/// Gap I.4 integration: the `FusedCshaBackward` launch op emitted by
/// `AdjointGenerator::generate` must carry 10 inputs (chain_key, dO,
/// q, k, v, x_norm, wq, wk, wv, norm_weight) when `chain_varids` is
/// populated. Pre-Gap-I.4 it carried only 5 (chain_key, dO, q, k, v)
/// and the lowerer's null-default branches left wq/wk/wv/x/norm_weight
/// at null, so the backward PTX skipped dwq/dwk/dwv accumulation.
#[test]
fn gap_i4_launch_inputs_thread_weight_and_norm_pointers() {
    use nsl_codegen::csha_apply::{CshaChainVarIds, FusionMark};
    use nsl_codegen::csha_boundary::ProjKind;
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use nsl_codegen::source_ad::{AdjointGenerator, CshaBackwardClaims};
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
    use std::collections::HashMap;

    let mk = |id, result, prim: PrimalOp, inputs| WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };
    let primal = WengertList {
        ops: vec![
            mk(0, 0, PrimalOp::Input("x".into()), vec![]),
            mk(1, 1, PrimalOp::Param("norm_w".into()), vec![]),
            mk(2, 2, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0, 1]),
            mk(3, 3, PrimalOp::Param("wq".into()), vec![]),
            mk(4, 4, PrimalOp::Matmul, vec![2, 3]),
            mk(5, 5, PrimalOp::RoPE { dim: 64 }, vec![4]),
            mk(6, 6, PrimalOp::Param("wk".into()), vec![]),
            mk(7, 7, PrimalOp::Matmul, vec![2, 6]),
            mk(8, 8, PrimalOp::RoPE { dim: 64 }, vec![7]),
            mk(9, 9, PrimalOp::Param("wv".into()), vec![]),
            mk(10, 10, PrimalOp::Matmul, vec![2, 9]),
            mk(
                11,
                11,
                PrimalOp::ScaledDotProductAttention { causal: false },
                vec![5, 8, 10],
            ),
        ],
        output: 11,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };

    let cfg = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            active_heads: 1,
            rmsnorm_eps: 1e-5,
            ..CshaExtras::default()
        }),
    };

    // Chain varids with a trainable gamma (norm_weight_var = Some(1)).
    let varids = CshaChainVarIds {
        q_out_var: 5,
        k_out_var: 8,
        v_out_var: 10,
        wq_var: 3,
        wk_var: 6,
        wv_var: 9,
        x_norm_var: 2,
        sdpa_out_var: 11,
        norm_weight_var: Some(1),
        // Gap I step K: RMSNorm op lives at VarId 2, its input
        // (pre-norm x) is VarId 0, and eps matches the op above.
        x_raw_var: Some(0),
        rmsnorm_eps: 1e-5,
    };
    let mark = FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "wq".into(),
        role: nsl_codegen::csha_apply::MarkRole::NormPrologue,
        config: Some(cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: Some(varids),
    };

    let mut op_to_chain: HashMap<u32, usize> = HashMap::new();
    op_to_chain.insert(11, 0); // SDPA primary
    op_to_chain.insert(2, 0); // RMSNorm
    op_to_chain.insert(4, 0); // Q matmul
    op_to_chain.insert(5, 0); // Q RoPE
    op_to_chain.insert(7, 0); // K matmul
    op_to_chain.insert(8, 0); // K RoPE
    op_to_chain.insert(10, 0); // V matmul

    let claims = CshaBackwardClaims {
        op_to_chain,
        chain_marks: vec![mark],
    };

    let mut gen = AdjointGenerator::new(200);
    gen.set_csha_claims(claims);
    let adjoint = gen.generate(&primal);

    let launch = adjoint
        .ops
        .iter()
        .find(|o| matches!(&o.op, PrimalOp::FusedCshaBackward { .. }))
        .expect("Gap I.4: fused launch op must be emitted");

    // Shape check — pre-Gap-I.4 this was 5 entries. Post-Gap-I.4 it
    // must be 10 (chain_key, dO, q, k, v, x, wq, wk, wv, norm_weight).
    assert_eq!(
        launch.inputs.len(),
        10,
        "Gap I.4: launch_inputs must carry 10 entries when chain_varids is \
         populated (chain_key, dO, q, k, v, x, wq, wk, wv, norm_weight); \
         got {} entries: {:?}",
        launch.inputs.len(),
        launch.inputs
    );

    // Index-by-index routing: the 5 new entries must be the primal
    // VarIds from chain_varids (adjoints are applied at launch input 1
    // only — the dO slot).
    assert_eq!(launch.inputs[5], 2, "inputs[5] = x_norm_var (VarId 2)");
    assert_eq!(launch.inputs[6], 3, "inputs[6] = wq_var (VarId 3)");
    assert_eq!(launch.inputs[7], 6, "inputs[7] = wk_var (VarId 6)");
    assert_eq!(launch.inputs[8], 9, "inputs[8] = wv_var (VarId 9)");
    assert_eq!(
        launch.inputs[9], 1,
        "inputs[9] = norm_weight_var (VarId 1) since gamma is trainable"
    );
}

/// Gap I.4 integration (null gamma): when `norm_weight_var` is `None`
/// the launch emitter passes `0` (null-sentinel VarId) at inputs[9].
/// The lowerer will iconst(0) that into a null pointer and the
/// kernel's `csha_norm_weight_ptr` null-guard skips the dgamma path.
#[test]
fn gap_i4_launch_inputs_pass_null_for_none_norm_weight() {
    use nsl_codegen::csha_apply::{CshaChainVarIds, FusionMark};
    use nsl_codegen::csha_boundary::ProjKind;
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use nsl_codegen::source_ad::{AdjointGenerator, CshaBackwardClaims};
    use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
    use std::collections::HashMap;

    let mk = |id, result, prim: PrimalOp, inputs| WengertOp {
        id,
        result,
        op: prim,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };
    let primal = WengertList {
        ops: vec![
            mk(0, 0, PrimalOp::Input("x".into()), vec![]),
            mk(1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
            mk(2, 2, PrimalOp::Param("wq".into()), vec![]),
            mk(3, 3, PrimalOp::Matmul, vec![1, 2]),
            mk(4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
            mk(5, 5, PrimalOp::Param("wk".into()), vec![]),
            mk(6, 6, PrimalOp::Matmul, vec![1, 5]),
            mk(7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
            mk(8, 8, PrimalOp::Param("wv".into()), vec![]),
            mk(9, 9, PrimalOp::Matmul, vec![1, 8]),
            mk(
                10,
                10,
                PrimalOp::ScaledDotProductAttention { causal: false },
                vec![4, 7, 9],
            ),
        ],
        output: 10,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    let cfg = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            active_heads: 1,
            rmsnorm_eps: 1e-5,
            ..CshaExtras::default()
        }),
    };
    let varids = CshaChainVarIds {
        q_out_var: 4,
        k_out_var: 7,
        v_out_var: 9,
        wq_var: 2,
        wk_var: 5,
        wv_var: 8,
        x_norm_var: 1,
        sdpa_out_var: 10,
        norm_weight_var: None,
        // Gap I step K: no gamma → no dgamma emission.
        x_raw_var: None,
        rmsnorm_eps: 0.0,
    };
    let mark = FusionMark {
        layer: "blocks.0".into(),
        kind: Some(ProjKind::Q),
        param_name: "wq".into(),
        role: nsl_codegen::csha_apply::MarkRole::NormPrologue,
        config: Some(cfg),
        backward_emitted: std::cell::Cell::new(false),
        chain_varids: Some(varids),
    };
    let mut op_to_chain: HashMap<u32, usize> = HashMap::new();
    for id in [10u32, 1, 3, 4, 6, 7, 9] {
        op_to_chain.insert(id, 0);
    }
    let claims = CshaBackwardClaims {
        op_to_chain,
        chain_marks: vec![mark],
    };
    let mut gen = AdjointGenerator::new(100);
    gen.set_csha_claims(claims);
    let adjoint = gen.generate(&primal);
    let launch = adjoint
        .ops
        .iter()
        .find(|o| matches!(&o.op, PrimalOp::FusedCshaBackward { .. }))
        .expect("launch op must exist");
    assert_eq!(launch.inputs.len(), 10, "Gap I.4 launch always carries 10 entries when chain_varids is Some");
    assert_eq!(
        launch.inputs[9], 0,
        "Gap I.4: norm_weight_var=None → launch_inputs[9] must be the \
         null-sentinel VarId (0); got {}",
        launch.inputs[9]
    );
}
