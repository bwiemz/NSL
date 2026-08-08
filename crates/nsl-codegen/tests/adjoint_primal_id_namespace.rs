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
        let interior: Vec<VarId> = ops[start..end]
            .iter()
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
    let mut gen = AdjointGenerator::new(fresh);
    let mut adjoint = gen.generate(primal);
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

    // Recompute the whole interior of the LAST block, SDPA included: the
    // clone of that SDPA is the op the claim table could alias.
    let seg = segments.last().expect("two blocks");
    let victims: Vec<VarId> = primal.ops[seg.start..seg.end]
        .iter()
        .map(|o| o.result)
        .collect();
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

/// The claim set a CSHA build publishes: the primal ids of block 0's
/// chain. Block 1's SDPA is deliberately NOT claimed — the hazard is an
/// unclaimed op's recompute clone matching a claimed op's key.
fn block0_claim_keys(primal: &WengertList, segments: &[BlockSegment]) -> HashSet<OpId> {
    let seg = &segments[0];
    primal.ops[seg.start..seg.end]
        .iter()
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
    let (adjoint, _) = adjoint_with_recompute_clones(&primal, segments);

    let legacy_ids: HashSet<OpId> = (0..adjoint.ops.len() as OpId).collect();
    let would_have_aliased: Vec<OpId> =
        legacy_ids.intersection(&claims).copied().collect();
    assert!(
        !would_have_aliased.is_empty(),
        "expected the pre-split numbering to collide with claim keys on \
         this fixture; if it no longer does, this test has stopped \
         describing the debt it was written for"
    );

    // And the collision matters because a real forward op is replayed on
    // the adjoint under one of those ids, not just a marker.
    assert!(
        sdpa_count(&adjoint) > 0,
        "the collision matters because an SDPA clone rides the adjoint; \
         this fixture has none"
    );
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
