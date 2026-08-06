//! Regression pins for the tape reference discipline (item 2, tape-mediated
//! ordering): positional indices may not outlive the scan that captured
//! them, id-space references must stay unique.
//!
//! Both tests here fail on the code as it stood before the fix:
//!
//! * CSHA's claim tables were POPULATED with positional indices
//!   (`BoundaryChain` fields, from `enumerate()`) and CONSUMED against
//!   `op.id` (`source_ad`'s reverse walk, `wengert_lower`'s fused-forward
//!   lookup, CCR's exemption test). The two spaces agree only while
//!   `op.id == index` — and the fused-LCE dead-chain prune deletes ops
//!   WITHOUT renumbering *before* the CSHA scan on any `@fused_lm_ce`
//!   build, after which an id-keyed consumer can miss the claimed op or
//!   false-hit an unrelated one into the fused backward.
//! * CCR's compressed-save tail minted new ids from `ops.len()`, which
//!   after any deletion sits at-or-below the surviving max id — duplicate
//!   ids, and id-keyed tables can match the appended cast op.

use std::collections::{HashMap, HashSet};

use nsl_codegen::csha::{run, CshaInput, CshaMode};
use nsl_codegen::csha_specialize::SpecConfig;
use nsl_codegen::wggo_cost::LayerShape;
use nsl_codegen::csha_apply::{
    bridge, collect_chain_dispatch_map_with_wengert, collect_claimed_ops,
};
use nsl_codegen::wengert::{OpId, PrimalOp, VarId, WengertList, WengertOp};

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

/// The `csha_apply` toy attention tape, with every op id shifted up by
/// `skew` — exactly the shape a list takes after `skew` earlier ops were
/// deleted without renumbering (the fused-LCE prune's effect): positions
/// run 0..10, ids run skew..skew+10.
fn skewed_attn_wengert(skew: u32) -> WengertList {
    let ops = vec![
        op(skew, 0, PrimalOp::Input("x".into()), vec![]),
        op(skew + 1, 1, PrimalOp::RMSNorm { eps: 1e-5 }, vec![0]),
        op(skew + 2, 2, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
        op(skew + 3, 3, PrimalOp::Matmul, vec![1, 2]),
        op(skew + 4, 4, PrimalOp::RoPE { dim: 64 }, vec![3]),
        op(skew + 5, 5, PrimalOp::Param("blocks.0.attn.wk".into()), vec![]),
        op(skew + 6, 6, PrimalOp::Matmul, vec![1, 5]),
        op(skew + 7, 7, PrimalOp::RoPE { dim: 64 }, vec![6]),
        op(skew + 8, 8, PrimalOp::Param("blocks.0.attn.wv".into()), vec![]),
        op(skew + 9, 9, PrimalOp::Matmul, vec![1, 8]),
    ];
    WengertList {
        ops,
        output: 9,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    }
}

fn plan_for(w: &WengertList) -> nsl_codegen::csha::CshaPlan {
    run(CshaInput {
        mode: CshaMode::Auto,
        target: "H100",
        wengert: w,
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
    })
}

/// On a tape where `op.id != index`, the published claim set must contain
/// the ops' IDS — what every consumer compares against — not their
/// positions. Positions here are {1, 3, 4, 6, 7, 9}; ids are those + 7.
#[test]
fn claims_are_op_ids_not_positions() {
    const SKEW: u32 = 7;
    let w = skewed_attn_wengert(SKEW);
    let plan = plan_for(&w);
    assert!(
        !plan.boundary.chains.is_empty(),
        "the toy tape stopped producing boundary chains — the premise died"
    );
    let claimed = collect_claimed_ops(&plan, &w);
    let expected: HashSet<u32> =
        [1, 3, 4, 6, 7, 9].iter().map(|p| p + SKEW).collect();
    assert_eq!(
        claimed, expected,
        "claim set must be id-space (positions + {SKEW} here); a \
         position-space set means the conversion boundary regressed"
    );
}

/// Same discipline for the reverse-walk dispatch map: its keys are looked
/// up by `op.id` at `source_ad`'s reverse walk and `wengert_lower`'s
/// fused-forward arm, so on a skewed tape every key must be an id. (The
/// grouped-SDPA path needs a fused SDPA primitive the toy tape doesn't
/// carry, so this exercises the legacy per-chain path — which is exactly
/// the arm whose keys went out unconverted.)
#[test]
fn dispatch_map_keys_are_op_ids_not_positions() {
    const SKEW: u32 = 7;
    let w = skewed_attn_wengert(SKEW);
    let plan = plan_for(&w);
    let br = bridge(&plan, 64, &mut Vec::new());
    let (op_to_chain, _marks) =
        collect_chain_dispatch_map_with_wengert(&plan, &br, Some(&w), None);
    assert!(
        !op_to_chain.is_empty(),
        "the dispatch map came back empty — the premise died"
    );
    let ids: HashSet<u32> = w.ops.iter().map(|o| o.id).collect();
    for key in op_to_chain.keys() {
        assert!(
            ids.contains(key),
            "dispatch-map key {key} is not any op's id on this tape — it is \
             a positional index that leaked past the conversion boundary \
             (ids run {SKEW}..{})",
            SKEW + 10
        );
    }
}

/// After a deletion, `ops.len()` is at-or-below the surviving max id, so a
/// len-minted id collides. `fresh_op_id` must mint ABOVE the max, and
/// `append_compressed_saves` must use it — asserted directly here by
/// running the real append on a deletion-shaped tape (the pre-fix code
/// panics in `assert_unique_op_ids` / produced a duplicate).
#[test]
fn ccr_append_mints_unique_ids_after_a_deletion() {
    // Ids 0..6 with op 2 deleted: len = 6, max id = 6 — a len-minted id (6)
    // collides with the surviving last op.
    let mut primal = WengertList {
        ops: vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("w".into()), vec![]),
            op(3, 3, PrimalOp::Matmul, vec![0, 1]),
            op(4, 4, PrimalOp::Relu, vec![3]),
            op(5, 5, PrimalOp::Matmul, vec![4, 1]),
            op(6, 6, PrimalOp::Sum { dim: None }, vec![5]),
        ],
        output: 6,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    assert_eq!(primal.fresh_op_id(), 7, "fresh ids mint above the max id");

    let ccr_plan = nsl_codegen::ccr::CcrPlan {
        segments: Vec::new(),
        recompute: HashSet::new(),
        per_segment_recompute: Vec::new(),
        free_eligible: HashSet::new(),
        compress: vec![3],
    };
    let mut fresh: VarId = 100;
    let map =
        nsl_codegen::ccr::append_compressed_saves(&mut primal, &ccr_plan, "fp16", &mut fresh);
    assert_eq!(map.len(), 1);
    // The two appended ops (cast + free) carry ids 7 and 8 — unique by
    // construction; the old len-minted scheme would have handed out 6.
    primal.assert_unique_op_ids("test: post-append");
    let appended: Vec<u32> = primal.ops[6..].iter().map(|o| o.id).collect();
    assert_eq!(appended, vec![7, 8]);
}

/// The uniqueness belt itself: duplicate ids panic, naming the context.
#[test]
fn duplicate_op_ids_panic_with_the_mutation_context() {
    let list = WengertList {
        ops: vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(0, 1, PrimalOp::Relu, vec![0]),
        ],
        output: 1,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    };
    let err = std::panic::catch_unwind(|| list.assert_unique_op_ids("test-context"))
        .expect_err("duplicate ids must panic");
    let msg = err
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(
        msg.contains("test-context") && msg.contains("duplicate Wengert OpId"),
        "the panic must name the mutation site: {msg}"
    );
}
