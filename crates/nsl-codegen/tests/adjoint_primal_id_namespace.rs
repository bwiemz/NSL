//! The primal and adjoint tapes must not share an `OpId` space.
//!
//! # The debt this closes
//!
//! `wengert_lower::compile_wengert_ops` lowers BOTH tapes through one
//! `match`, and inside the `ScaledDotProductAttention` arm it consults
//! `claims.op_to_chain.get(&op.id)` — a table keyed by PRIMAL op ids — to
//! decide whether to emit CSHA's fused forward (which also allocates and
//! registers that layer's backward save buffers) instead of decomposing.
//!
//! Until the id split, both tapes numbered from 0. CCR splices *clones of
//! primal forward ops* into the adjoint and renumbers it positionally, so
//! a recompute clone of an UNCLAIMED SDPA could land on a claimed primal
//! id — at which point the clone takes another layer's fused-forward path
//! and stores its save pointers under that layer's key.
//!
//! What kept that from firing was a single `bus.clear_csha_backward_claims()`
//! sited between the forward and adjoint lowerings: a temporal invariant,
//! at one of the ~8 `compile_wengert_ops` call sites, enforced by nothing.
//! `numbering_alone_admitted_the_collision_before_the_split` below pins the
//! numbering half of that — the half that made the timing load-bearing.
//!
//! The fix gives the adjoint a disjoint half of the id space
//! (`ADJOINT_ID_BASE`), so a primal-keyed lookup cannot match an adjoint op
//! whenever it runs.

use std::collections::{HashMap, HashSet};

use nsl_codegen::ccr::{apply_to_adjoint, BlockSegment, CcrPlan};
use nsl_codegen::source_ad::AdjointGenerator;
use nsl_codegen::wengert::{
    adjoint_op_id, id_space, IdSpace, OpId, PrimalOp, VarId, WengertList, WengertOp, WengertType,
    ADJOINT_ID_BASE,
};

fn op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
    WengertOp {
        id,
        result,
        op,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    }
}

/// A two-block attention-ish primal tape with one SDPA per block, ids
/// positional (`id == index`), which is how extraction leaves it.
///
/// Block k occupies a contiguous span, so CCR's `BlockSegment`s can be
/// handed out over it without a real boundary scan.
fn two_block_primal() -> (WengertList, Vec<BlockSegment>) {
    let mut ops = Vec::new();
    let mut var_types = HashMap::new();
    let mut segments = Vec::new();
    let mut next: u32 = 0;
    let mut push = |ops: &mut Vec<WengertOp>, o: PrimalOp, inputs: Vec<VarId>| -> VarId {
        let id = ops.len() as u32;
        let result = next;
        next += 1;
        ops.push(op(id, result, o, inputs));
        result
    };

    let x = push(&mut ops, PrimalOp::Input("x".into()), vec![]);
    for b in 0..2u32 {
        let start = ops.len();
        let n = push(&mut ops, PrimalOp::RMSNorm { eps: 1e-5 }, vec![x]);
        let wq = push(&mut ops, PrimalOp::Param(format!("blocks.{b}.wq")), vec![]);
        let q = push(&mut ops, PrimalOp::Matmul, vec![n, wq]);
        let wk = push(&mut ops, PrimalOp::Param(format!("blocks.{b}.wk")), vec![]);
        let k = push(&mut ops, PrimalOp::Matmul, vec![n, wk]);
        let wv = push(&mut ops, PrimalOp::Param(format!("blocks.{b}.wv")), vec![]);
        let v = push(&mut ops, PrimalOp::Matmul, vec![n, wv]);
        let scale = push(&mut ops, PrimalOp::Constant(0.125), vec![]);
        let attn = push(
            &mut ops,
            PrimalOp::ScaledDotProductAttention { causal: false },
            vec![q, k, v, scale],
        );
        let out = push(&mut ops, PrimalOp::Relu, vec![attn]);
        let end = ops.len();
        // `Input`/`Param`/`Constant` are skipped BEFORE `interior` is built in
        // the real planner (`ccr::plan`), so a real plan never recomputes a
        // Param. Match that: an interior containing Params drives
        // `apply_to_adjoint` into cloning `Param(..)` ops onto the adjoint, a
        // configuration production never emits.
        let interior: Vec<VarId> = ops[start..end]
            .iter()
            .filter(|o| {
                !matches!(
                    o.op,
                    PrimalOp::Input(_) | PrimalOp::Param(_) | PrimalOp::Constant(_)
                )
            })
            .map(|o| o.result)
            .filter(|r| *r != out)
            .collect();
        segments.push(BlockSegment {
            layer_key: format!("blocks.{b}"),
            start,
            end,
            escaping: vec![out],
            interior,
        });
    }
    let output = ops.last().expect("non-empty tape").result;
    for o in &ops {
        var_types.insert(o.result, WengertType::Tensor);
    }
    (
        WengertList {
            ops,
            output,
            var_names: HashMap::new(),
            var_types,
        },
        segments,
    )
}

/// Every op id on a list, as a set.
fn ids(list: &WengertList) -> HashSet<OpId> {
    list.ops.iter().map(|o| o.id).collect()
}

/// Count `ScaledDotProductAttention` ops on a list.
fn sdpa_count(list: &WengertList) -> usize {
    list.ops
        .iter()
        .filter(|o| matches!(o.op, PrimalOp::ScaledDotProductAttention { .. }))
        .count()
}

/// Build the adjoint for `primal` and splice CCR recompute clones of
/// block 1's interior into it — the configuration the hazard needs.
///
/// Returns `(adjoint, sdpa_clones_added_by_the_splice)`.
fn adjoint_with_recompute_clones(
    primal: &WengertList,
    segments: Vec<BlockSegment>,
) -> (WengertList, usize) {
    let mut fresh: VarId = primal.ops.iter().map(|o| o.result).max().unwrap_or(0) + 1;
    let mut generator = AdjointGenerator::new(fresh);
    let mut adjoint = generator.generate(primal);
    fresh = fresh
        .max(adjoint.ops.iter().map(|o| o.result).max().unwrap_or(0) + 1)
        .max(
            adjoint
                .ops
                .iter()
                .flat_map(|o| o.inputs.iter().copied())
                .max()
                .unwrap_or(0)
                + 1,
        );

    // Recompute the LAST block's interior, SDPA included: the clone of that
    // SDPA is the op the claim table could alias. Victims come from the
    // segment's own `interior` (which already excludes leaves), so the plan
    // has the shape `ccr::plan` produces rather than one this test invented.
    let seg = segments.last().expect("two blocks");
    let victims: Vec<VarId> = seg.interior.clone();
    let recompute: HashSet<VarId> = victims.iter().copied().collect();
    let plan = CcrPlan {
        segments,
        recompute,
        per_segment_recompute: vec![Vec::new(), victims],
        free_eligible: HashSet::new(),
        compress: Vec::new(),
    };
    let before = sdpa_count(&adjoint);
    apply_to_adjoint(primal, &mut adjoint, &plan, &mut fresh).expect("ccr splice");
    let added = sdpa_count(&adjoint) - before;
    (adjoint, added)
}

/// The claim set a CSHA build publishes for block 0.
///
/// Shaped like the real thing: `collect_chain_dispatch_map_with_wengert`
/// claims the boundary chain — RMSNorm, the Q/K/V projection matmuls, the
/// optional RoPE — plus the SDPA primary. It does NOT claim `Param`,
/// `Constant` or the block's output op, and keying on every id in the span
/// (as an earlier version of this file did) makes the claim set a strict
/// superset of any real one, which weakens what a disjointness result means.
///
/// Block 1's SDPA is deliberately NOT claimed — the hazard is an UNCLAIMED
/// op's recompute clone landing on a CLAIMED op's key.
fn block0_claim_keys(primal: &WengertList, segments: &[BlockSegment]) -> HashSet<OpId> {
    let seg = &segments[0];
    primal.ops[seg.start..seg.end]
        .iter()
        .filter(|o| {
            matches!(
                o.op,
                PrimalOp::RMSNorm { .. }
                    | PrimalOp::Matmul
                    | PrimalOp::RoPE { .. }
                    | PrimalOp::ScaledDotProductAttention { .. }
            )
        })
        .map(|o| o.id)
        .collect()
}

#[test]
fn adjoint_op_ids_are_disjoint_from_primal_op_ids() {
    let (primal, segments) = two_block_primal();
    let (adjoint, _) = adjoint_with_recompute_clones(&primal, segments);

    // Non-vacuity first: an empty adjoint is trivially disjoint from
    // anything, and would let this test pass over a broken generator.
    assert!(
        primal.ops.len() >= 20,
        "fixture should build a real two-block tape, got {} ops",
        primal.ops.len()
    );
    assert!(
        adjoint.ops.len() >= 10,
        "adjoint should be non-trivial, got {} ops",
        adjoint.ops.len()
    );

    let overlap: Vec<OpId> = ids(&primal).intersection(&ids(&adjoint)).copied().collect();
    assert!(
        overlap.is_empty(),
        "primal and adjoint id spaces overlap on {overlap:?} — a table \
         keyed by primal ids can alias an adjoint op"
    );
    for o in &primal.ops {
        assert_eq!(id_space(o.id), IdSpace::Primal, "primal op {:?}", o.op);
    }
    for o in &adjoint.ops {
        assert_eq!(id_space(o.id), IdSpace::Adjoint, "adjoint op {:?}", o.op);
    }
}

#[test]
fn a_recompute_clone_of_an_unclaimed_sdpa_cannot_match_a_claim_key() {
    let (primal, segments) = two_block_primal();
    let claims = block0_claim_keys(&primal, &segments);
    let (adjoint, sdpa_clones) = adjoint_with_recompute_clones(&primal, segments);

    // Non-vacuity: the CCR splice must actually have ADDED an SDPA clone
    // to the adjoint. Without this the disjointness assertion below could
    // hold because the scenario never materialized.
    assert!(
        sdpa_clones > 0,
        "the CCR splice added no SDPA clone to the adjoint — the alias \
         scenario never materialized, so the assertion below is vacuous"
    );
    assert!(
        !claims.is_empty(),
        "fixture produced no claim keys — the assertion below would be vacuous"
    );

    let aliased: Vec<OpId> = adjoint
        .ops
        .iter()
        .map(|o| o.id)
        .filter(|id| claims.contains(id))
        .collect();
    assert!(
        aliased.is_empty(),
        "adjoint ops {aliased:?} carry ids that are CSHA claim keys — \
         `wengert_lower`'s SDPA arm would route them through the claimed \
         layer's fused forward and register its save buffers"
    );
}

/// The numbering half of the debt, pinned as a fact about the OLD scheme.
///
/// Renumbering the adjoint positionally from 0 — what every adjoint
/// rewriter did before `renumber_adjoint_ops` — puts adjoint ids squarely
/// inside the primal claim key range. This is what made the
/// `clear_csha_backward_claims()` placement load-bearing; the split is
/// what makes it merely belt-and-braces.
#[test]
fn numbering_alone_admitted_the_collision_before_the_split() {
    let (primal, segments) = two_block_primal();
    let claims = block0_claim_keys(&primal, &segments);
    let (adjoint, sdpa_clones) = adjoint_with_recompute_clones(&primal, segments);
    assert!(sdpa_clones > 0, "no SDPA clone was spliced onto the adjoint");

    // The SPECIFIC op that matters: the recompute clone of block 1's
    // (unclaimed) SDPA, now riding the adjoint. Under the pre-split scheme
    // its id was its POSITION in the adjoint — that is what every adjoint
    // rewriter wrote — so that is the id `wengert_lower`'s SDPA arm would
    // have looked up in the primal-keyed claim table.
    //
    // An earlier version of this test intersected `0..adjoint.ops.len()`
    // with the claim set, which reduces to `adjoint.ops.len() >= 2` and says
    // nothing about the clone at all. Locate the clone and check ITS id.
    let clone_pos = adjoint
        .ops
        .iter()
        .position(|o| matches!(o.op, PrimalOp::ScaledDotProductAttention { .. }))
        .expect("the CCR splice put an SDPA clone on the adjoint");
    let legacy_id = clone_pos as OpId;

    assert!(
        claims.contains(&legacy_id),
        "the SDPA recompute clone sits at adjoint position {clone_pos}, whose \
         pre-split id ({legacy_id}) is NOT one of block 0's claim keys \
         {claims:?} — the fixture no longer reproduces the collision this \
         test exists to pin, so it has stopped describing the debt"
    );

    // And the op that key belongs to is a CLAIMED SDPA in the primal, so the
    // lookup would have resolved to a real chain and taken the fused-forward
    // path under block 0's layer — not merely matched a number.
    let claimed_op = primal
        .ops
        .iter()
        .find(|o| o.id == legacy_id)
        .expect("claim keys are primal op ids");
    assert!(
        matches!(
            claimed_op.op,
            PrimalOp::RMSNorm { .. }
                | PrimalOp::Matmul
                | PrimalOp::RoPE { .. }
                | PrimalOp::ScaledDotProductAttention { .. }
        ),
        "the aliased key must belong to a real boundary-chain op, got {:?}",
        claimed_op.op
    );

    // Post-split, that same clone carries an adjoint-space id instead.
    assert_eq!(
        id_space(adjoint.ops[clone_pos].id),
        IdSpace::Adjoint,
        "the clone should now be numbered in the adjoint half"
    );
    assert!(!claims.contains(&adjoint.ops[clone_pos].id));
}

#[test]
fn publishing_a_claim_keyed_by_an_adjoint_id_is_refused() {
    use nsl_codegen::pass_bus::PassBus;
    use nsl_codegen::source_ad::CshaBackwardClaims;

    let mut bus = PassBus::default();
    let mut op_to_chain = HashMap::new();
    op_to_chain.insert(adjoint_op_id(3), 0usize);
    let claims = CshaBackwardClaims {
        op_to_chain,
        chain_marks: Vec::new(),
    };
    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        bus.publish_csha_backward_claims(claims);
    }))
    .expect_err("publishing an adjoint-keyed claim must panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("ADJOINT-space op id"),
        "panic should name the id space, got: {msg}"
    );
}

/// The adjoint rewriters take a bare `&mut Vec<WengertOp>`, and `primal.ops`
/// has that exact type — so handing them the primal COMPILES. Their trailing
/// positional renumber would then rewrite every primal id into the adjoint
/// half, invalidating every CSHA claim key and CCR's exemption set at once,
/// and after the renumber there is no evidence left that it happened.
///
/// This is why the id-space assertions live at each rewriter's ENTRY. An
/// assertion placed after `renumber_adjoint_ops` only re-reads what that call
/// unconditionally wrote: it cannot fail for any input, which reads like
/// coverage while providing none.
#[test]
fn handing_a_primal_list_to_an_adjoint_rewriter_is_refused() {
    let (primal, _) = two_block_primal();
    let mut primal_ops = primal.ops.clone();
    let needed: HashSet<VarId> = HashSet::new();

    let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        nsl_codegen::source_ad::fuse_swiglu_gate_backward(&mut primal_ops, &needed);
    }))
    .expect_err("a primal list reaching an adjoint rewriter must panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("Primal id space") || msg.contains("expected Adjoint"),
        "the panic should name the id-space mismatch, got: {msg}"
    );

    // And the guard has NOT silently renumbered anything on the way out.
    assert_eq!(
        primal_ops.iter().map(|o| o.id).collect::<Vec<_>>(),
        primal.ops.iter().map(|o| o.id).collect::<Vec<_>>(),
        "the refusal must precede any mutation"
    );
}

#[test]
fn adjoint_ids_stay_positional_within_their_half() {
    // The property `ccr.rs`'s splice renumbering relies on: ids are still
    // dense and ascending, only rebased. Anything that walked the adjoint
    // expecting `id - base == index` keeps working.
    let mut ops: Vec<WengertOp> = (0..5)
        .map(|i| op(0, i, PrimalOp::Relu, vec![]))
        .collect();
    nsl_codegen::wengert::renumber_adjoint_ops(&mut ops);
    for (i, o) in ops.iter().enumerate() {
        assert_eq!(o.id, ADJOINT_ID_BASE + i as OpId);
        assert_eq!(id_space(o.id), IdSpace::Adjoint);
    }
}

#[test]
fn an_index_that_would_wrap_into_the_primal_space_panics() {
    // A wrap would land back among primal ids and silently restore the
    // exact aliasing this split removes — louder is strictly better.
    let too_big = (u32::MAX - ADJOINT_ID_BASE) as usize + 1;
    let err = std::panic::catch_unwind(|| adjoint_op_id(too_big))
        .expect_err("an index past the adjoint half must panic, not wrap");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("overflows the adjoint id half"),
        "panic should explain the wrap hazard, got: {msg}"
    );
    // The last representable index is fine.
    assert_eq!(adjoint_op_id(too_big - 1), u32::MAX);
}
