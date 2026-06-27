// Integration tests for wggo_prune::run() — spec §7.5, Task 14a slice.
//
// Harness variant: STRUCTURAL (not numerical).
//
// This file builds synthetic WengertLists that mirror the structure
// `prune_rewrite_toy.nsl` would produce if it were compiled through the
// WengertExtractor pipeline. The structural approach avoids full NSL
// compilation (no lex/parse/semantic/extractor chain), making the tests
// fast, hermetic, and free of GPU/CUDA setup.
//
// Known deviation from spec §7.5 full intent:
//   Layer 3 "bit-exact numerical forward output comparison" is NOT
//   implemented in this slice. These tests assert IR structural properties
//   after rewrite (rewrites.len(), ops_deleted, h_before/h_after VarIds,
//   layer names) rather than executing a forward pass and comparing float
//   outputs. See Task 14b for the numerical comparison extension.
//
// What IS tested here (Task 14a):
//   - harness: mk_toy_wengert() builds a 4-block pre-norm transformer
//     Wengert structure matching prune_rewrite_toy.nsl (d_model=32, 4 heads,
//     blocks.0..3, each with attn + ffn subgraph + residual Add boundary)
//   - blocks_1_attn_prunes_without_refusal: single-layer prune, correct
//     structural rewrite (rewrites.len==1, refusals empty, ops_deleted>0)
//   - blocks_1_attn_rewrite_removes_correct_layer: asserts layer_name and
//     h_before/h_after VarIds are the ones belonging to blocks.1.attn
//   - blocks_1_attn_wengert_output_unchanged: the wengert.output stays
//     consistent (pointing at the final block output, not a pruned var)
//
// Deferred to Task 14b:
//   - Remaining 7 fixture variants (blocks.0.attn, blocks.0.ffn, etc.)
//   - 20 Layer 1 insta snapshots
//   - Strong adjoint assertion (source-AD input reduced after prune)
//   - Layer 4 stderr diagnostic tests
//   - Numerical (bit-exact forward) comparison to reference fixtures

use nsl_codegen::wengert::{PrimalOp, VarId, WengertList, WengertOp};
use nsl_codegen::wggo_apply::{AppliedLayer, AppliedPlan};
use nsl_codegen::wggo_dp::LayerDecision;
use nsl_codegen::weight_aware::WeightMap;
use nsl_codegen::wggo_prune::{PruneRefusal, run};
use std::collections::HashMap;

// --------------------------------------------------------------------------
// Toy Wengert builder — mirrors prune_rewrite_toy.nsl structure.
//
// The toy NSL model is:
//   4 TinyBlock instances (blocks.0 .. blocks.3)
//   Each block:
//     h1 = h + block.attn(h)   — attn residual add
//     h2 = h1 + block.ffn(h1)  — ffn residual add
//
// VarId layout (8 vars per block + 2 shared stream vars per add = compact):
//   We assign var IDs in ranges to keep arithmetic simple and readable.
//   Per block B (0..3):
//     stream_in[B]  = B * 20        (h entering block B)
//     attn_param[B] = B * 20 + 1   (first attn param op result; named blocks.B.attn.wq)
//     attn_out[B]   = B * 20 + 2   (attn sub-result; read by residual add)
//     stream_mid[B] = B * 20 + 3   (h after attn residual add = stream_in[B] + attn_out[B])
//     ffn_param[B]  = B * 20 + 4   (first ffn param op result; named blocks.B.ffn.w_ffn_up)
//     ffn_out[B]    = B * 20 + 5   (ffn sub-result; read by residual add)
//     stream_out[B] = B * 20 + 6   (h after ffn residual add = stream_mid[B] + ffn_out[B])
//   stream_in[0] = 0 (model input x)
//   stream_in[B+1] = stream_out[B]  (chained through blocks)
//
// Op IDs (4 ops per block: attn_param, attn_add, ffn_param, ffn_add):
//   Per block B: op_base = B * 4
//     op_base + 0: Relu  attn_param[B] = relu(stream_in[B])   — proxy for attn subgraph
//     op_base + 1: Add   stream_mid[B] = stream_in[B] + attn_out[B]
//     op_base + 2: Relu  ffn_param[B]  = relu(stream_mid[B])  — proxy for ffn subgraph
//     op_base + 3: Add   stream_out[B] = stream_mid[B] + ffn_out[B]
//
// This is intentionally minimal — the exact ops inside the layer closure don't
// matter for prune correctness; what matters is:
//   (a) var_names map has entries with the right "blocks.B.attn." / "blocks.B.ffn." prefix
//   (b) residual Add has inputs (h_before, block_output) where h_before is
//       from the prior stream and block_output is reachable from a layer param
//   (c) h_after = result of the residual Add

fn mk_toy_wengert() -> WengertList {
    // Stream vars (one per add-output + the initial input):
    //   stream_in[0] = VarId 0  (model input)
    //   stream_mid[B] = VarId B*20 + 3
    //   stream_out[B] = VarId B*20 + 6
    //   stream_in[B+1] = stream_out[B] = B*20 + 6  (= (B+1)*20 would be offset;
    //   we just alias stream_in[B+1] = stream_out[B] by chaining adds properly)
    //
    // To keep the arithmetic clean, use explicit constants:
    const INPUT: VarId = 0;

    // block 0
    const B0_ATTN_PARAM: VarId = 1;
    const B0_ATTN_OUT: VarId   = 2;   // result of the relu (proxy for whole attn path)
    const B0_MID: VarId        = 3;   // h after attn add = INPUT + B0_ATTN_OUT
    const B0_FFN_PARAM: VarId  = 4;
    const B0_FFN_OUT: VarId    = 5;
    const B0_OUT: VarId        = 6;   // h after ffn add  = B0_MID + B0_FFN_OUT

    // block 1
    const B1_ATTN_PARAM: VarId = 7;
    const B1_ATTN_OUT: VarId   = 8;
    const B1_MID: VarId        = 9;   // B0_OUT + B1_ATTN_OUT
    const B1_FFN_PARAM: VarId  = 10;
    const B1_FFN_OUT: VarId    = 11;
    const B1_OUT: VarId        = 12;  // B1_MID + B1_FFN_OUT

    // block 2
    const B2_ATTN_PARAM: VarId = 13;
    const B2_ATTN_OUT: VarId   = 14;
    const B2_MID: VarId        = 15;
    const B2_FFN_PARAM: VarId  = 16;
    const B2_FFN_OUT: VarId    = 17;
    const B2_OUT: VarId        = 18;

    // block 3
    const B3_ATTN_PARAM: VarId = 19;
    const B3_ATTN_OUT: VarId   = 20;
    const B3_MID: VarId        = 21;
    const B3_FFN_PARAM: VarId  = 22;
    const B3_FFN_OUT: VarId    = 23;
    const B3_OUT: VarId        = 24;

    // Helper: build a WengertOp.
    let make_op = |id: u32, result: VarId, op: PrimalOp, inputs: Vec<VarId>| WengertOp {
        id,
        result,
        op,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    };

    let ops = vec![
        // --- block 0 attn ---
        // B0_ATTN_PARAM (proxy for wq matmul etc.) — read stream_in=INPUT
        make_op(0, B0_ATTN_PARAM, PrimalOp::Relu, vec![INPUT]),
        // B0_ATTN_OUT reads B0_ATTN_PARAM
        make_op(1, B0_ATTN_OUT, PrimalOp::Relu, vec![B0_ATTN_PARAM]),
        // attn residual add: B0_MID = INPUT + B0_ATTN_OUT
        make_op(2, B0_MID, PrimalOp::Add, vec![INPUT, B0_ATTN_OUT]),

        // --- block 0 ffn ---
        make_op(3, B0_FFN_PARAM, PrimalOp::Relu, vec![B0_MID]),
        make_op(4, B0_FFN_OUT,   PrimalOp::Relu, vec![B0_FFN_PARAM]),
        // ffn residual add: B0_OUT = B0_MID + B0_FFN_OUT
        make_op(5, B0_OUT, PrimalOp::Add, vec![B0_MID, B0_FFN_OUT]),

        // --- block 1 attn ---
        make_op(6,  B1_ATTN_PARAM, PrimalOp::Relu, vec![B0_OUT]),
        make_op(7,  B1_ATTN_OUT,   PrimalOp::Relu, vec![B1_ATTN_PARAM]),
        // attn residual add: B1_MID = B0_OUT + B1_ATTN_OUT
        make_op(8,  B1_MID, PrimalOp::Add, vec![B0_OUT, B1_ATTN_OUT]),

        // --- block 1 ffn ---
        make_op(9,  B1_FFN_PARAM, PrimalOp::Relu, vec![B1_MID]),
        make_op(10, B1_FFN_OUT,   PrimalOp::Relu, vec![B1_FFN_PARAM]),
        // ffn residual add: B1_OUT = B1_MID + B1_FFN_OUT
        make_op(11, B1_OUT, PrimalOp::Add, vec![B1_MID, B1_FFN_OUT]),

        // --- block 2 attn ---
        make_op(12, B2_ATTN_PARAM, PrimalOp::Relu, vec![B1_OUT]),
        make_op(13, B2_ATTN_OUT,   PrimalOp::Relu, vec![B2_ATTN_PARAM]),
        make_op(14, B2_MID, PrimalOp::Add, vec![B1_OUT, B2_ATTN_OUT]),

        // --- block 2 ffn ---
        make_op(15, B2_FFN_PARAM, PrimalOp::Relu, vec![B2_MID]),
        make_op(16, B2_FFN_OUT,   PrimalOp::Relu, vec![B2_FFN_PARAM]),
        make_op(17, B2_OUT, PrimalOp::Add, vec![B2_MID, B2_FFN_OUT]),

        // --- block 3 attn ---
        make_op(18, B3_ATTN_PARAM, PrimalOp::Relu, vec![B2_OUT]),
        make_op(19, B3_ATTN_OUT,   PrimalOp::Relu, vec![B3_ATTN_PARAM]),
        make_op(20, B3_MID, PrimalOp::Add, vec![B2_OUT, B3_ATTN_OUT]),

        // --- block 3 ffn ---
        make_op(21, B3_FFN_PARAM, PrimalOp::Relu, vec![B3_MID]),
        make_op(22, B3_FFN_OUT,   PrimalOp::Relu, vec![B3_FFN_PARAM]),
        make_op(23, B3_OUT, PrimalOp::Add, vec![B3_MID, B3_FFN_OUT]),
    ];

    // var_names: label each param producer and key stream vars so
    // wggo_prune can identify layer closures by prefix.
    let var_names: HashMap<VarId, String> = [
        (INPUT,         "x".to_string()),
        // block 0
        (B0_ATTN_PARAM, "blocks.0.attn.wq".to_string()),
        (B0_FFN_PARAM,  "blocks.0.ffn.w_ffn_up".to_string()),
        // block 1
        (B1_ATTN_PARAM, "blocks.1.attn.wq".to_string()),
        (B1_FFN_PARAM,  "blocks.1.ffn.w_ffn_up".to_string()),
        // block 2
        (B2_ATTN_PARAM, "blocks.2.attn.wq".to_string()),
        (B2_FFN_PARAM,  "blocks.2.ffn.w_ffn_up".to_string()),
        // block 3
        (B3_ATTN_PARAM, "blocks.3.attn.wq".to_string()),
        (B3_FFN_PARAM,  "blocks.3.ffn.w_ffn_up".to_string()),
        // stream labels for human readability in error messages
        (B0_MID,        "h_after_b0_attn".to_string()),
        (B0_OUT,        "h_after_b0".to_string()),
        (B1_MID,        "h_after_b1_attn".to_string()),
        (B1_OUT,        "h_after_b1".to_string()),
        (B2_MID,        "h_after_b2_attn".to_string()),
        (B2_OUT,        "h_after_b2".to_string()),
        (B3_MID,        "h_after_b3_attn".to_string()),
        (B3_OUT,        "h_after_b3".to_string()),
    ].into_iter().collect();

    WengertList {
        ops,
        output: B3_OUT,
        var_names,
        var_types: HashMap::new(),
    }
}

/// Build a minimal AppliedLayer for a Prune decision.
fn mk_prune_layer(idx: u32, name: &str) -> AppliedLayer {
    AppliedLayer {
        layer_index: idx,
        layer_name: name.to_string(),
        coarse: LayerDecision::Prune,
        pipeline_stage: 0,
        shard_factor: 1,
        shard_grads: 1,
        shard_optim: 1,
        active_heads: 0,
        ffn_width: 0,
        csha_level: 0,
        adapter_rank: 0,
        optim_m_bits: 0,
        optim_v_bits: 0,
        fase_fused: false,
        packing_mode: 0,
        estimated_us: 0.0,
        param_bytes: 0,
        activation_bytes: 0,
    }
}

/// Build a minimal AppliedLayer for a KeepFull (non-pruned) decision.
fn mk_keep_layer(idx: u32, name: &str) -> AppliedLayer {
    AppliedLayer {
        layer_index: idx,
        layer_name: name.to_string(),
        coarse: LayerDecision::KeepFull,
        pipeline_stage: 0,
        shard_factor: 1,
        shard_grads: 1,
        shard_optim: 1,
        active_heads: 4,
        ffn_width: 64,
        csha_level: 1,
        adapter_rank: 0,
        optim_m_bits: 8,
        optim_v_bits: 8,
        fase_fused: false,
        packing_mode: 0,
        estimated_us: 10.0,
        param_bytes: 1024,
        activation_bytes: 512,
    }
}

// --------------------------------------------------------------------------
// Test 1 (Task 14a PRIMARY): single-layer prune of blocks.1.attn
//   - Plan has 8 layers total (blocks.0.attn, blocks.0.ffn, ... blocks.3.ffn)
//   - Only blocks.1.attn is marked Prune; rest are KeepFull
//   - Asserts: no refusals, 1 rewrite, ops_deleted > 0
// --------------------------------------------------------------------------

#[test]
fn blocks_1_attn_prunes_without_refusal() {
    let mut wengert = mk_toy_wengert();
    let weight_map = WeightMap::default();

    let applied = AppliedPlan {
        layers: vec![
            mk_keep_layer(0, "blocks.0.attn"),
            mk_keep_layer(1, "blocks.0.ffn"),
            mk_prune_layer(2, "blocks.1.attn"),   // <-- prune target
            mk_keep_layer(3, "blocks.1.ffn"),
            mk_keep_layer(4, "blocks.2.attn"),
            mk_keep_layer(5, "blocks.2.ffn"),
            mk_keep_layer(6, "blocks.3.attn"),
            mk_keep_layer(7, "blocks.3.ffn"),
        ],
        total_us: 80.0,
        peak_memory_bytes: 8192,
    };

    let result = run(&mut wengert, &applied, &weight_map);

    assert!(
        result.refusals.is_empty(),
        "expected no refusals, got: {:?}",
        result.refusals
    );
    assert_eq!(
        result.rewrites.len(),
        1,
        "expected exactly 1 rewrite for blocks.1.attn prune"
    );
    assert!(
        result.ops_deleted > 0,
        "expected ops_deleted > 0 after prune"
    );
    assert!(
        !result.pruned_forward_var_ids.is_empty(),
        "expected pruned_forward_var_ids to be populated"
    );
}

// --------------------------------------------------------------------------
// Test 2: rewrite targets the CORRECT layer (blocks.1.attn) and records the
//   right h_before / h_after VarIds.
//   h_before for blocks.1.attn is B0_OUT (VarId 6 = left input of op 8).
//   h_after  for blocks.1.attn is B1_MID (VarId 9 = result of op 8).
// --------------------------------------------------------------------------

#[test]
fn blocks_1_attn_rewrite_removes_correct_layer() {
    let mut wengert = mk_toy_wengert();
    let weight_map = WeightMap::default();

    let applied = AppliedPlan {
        layers: vec![
            mk_keep_layer(0, "blocks.0.attn"),
            mk_keep_layer(1, "blocks.0.ffn"),
            mk_prune_layer(2, "blocks.1.attn"),
            mk_keep_layer(3, "blocks.1.ffn"),
            mk_keep_layer(4, "blocks.2.attn"),
            mk_keep_layer(5, "blocks.2.ffn"),
            mk_keep_layer(6, "blocks.3.attn"),
            mk_keep_layer(7, "blocks.3.ffn"),
        ],
        total_us: 80.0,
        peak_memory_bytes: 8192,
    };

    let result = run(&mut wengert, &applied, &weight_map);

    assert!(result.refusals.is_empty(), "unexpected refusals: {:?}", result.refusals);
    assert_eq!(result.rewrites.len(), 1);

    let rewrite = &result.rewrites[0];
    assert_eq!(
        rewrite.layer_name, "blocks.1.attn",
        "rewrite should target blocks.1.attn, got: {}",
        rewrite.layer_name
    );

    // h_before for blocks.1.attn is B0_OUT = VarId 6
    // (op 8 is the residual Add: Add(B0_OUT=6, B1_ATTN_OUT=8) -> B1_MID=9)
    assert_eq!(
        rewrite.h_before_var, 6,
        "h_before should be B0_OUT (VarId 6), got {}",
        rewrite.h_before_var
    );
    // h_after for blocks.1.attn is B1_MID = VarId 9
    assert_eq!(
        rewrite.h_after_var, 9,
        "h_after should be B1_MID (VarId 9), got {}",
        rewrite.h_after_var
    );
    // residual_add_op should be op 8
    assert_eq!(
        rewrite.residual_add_op, 8,
        "residual_add_op should be op 8 (the attn Add), got {}",
        rewrite.residual_add_op
    );
}

// --------------------------------------------------------------------------
// Test 3: wengert.output is NOT a pruned var after the rewrite. The pruned
//   layer (blocks.1.attn) produces B1_MID which is NOT the terminal output
//   (B3_OUT = VarId 24). Verify output is consistent.
// --------------------------------------------------------------------------

#[test]
fn blocks_1_attn_wengert_output_unchanged() {
    let mut wengert = mk_toy_wengert();
    let weight_map = WeightMap::default();

    // Capture output before prune.
    let original_output = wengert.output;
    assert_eq!(original_output, 24, "pre-condition: output should be B3_OUT=24");

    let applied = AppliedPlan {
        layers: vec![
            mk_keep_layer(0, "blocks.0.attn"),
            mk_keep_layer(1, "blocks.0.ffn"),
            mk_prune_layer(2, "blocks.1.attn"),
            mk_keep_layer(3, "blocks.1.ffn"),
            mk_keep_layer(4, "blocks.2.attn"),
            mk_keep_layer(5, "blocks.2.ffn"),
            mk_keep_layer(6, "blocks.3.attn"),
            mk_keep_layer(7, "blocks.3.ffn"),
        ],
        total_us: 80.0,
        peak_memory_bytes: 8192,
    };

    let result = run(&mut wengert, &applied, &weight_map);

    assert!(result.refusals.is_empty());
    assert_eq!(result.rewrites.len(), 1);

    // B1_MID (VarId 9) IS in pruned_forward_var_ids (h_after of the pruned layer).
    assert!(
        result.pruned_forward_var_ids.contains(&9),
        "B1_MID (VarId 9, h_after of blocks.1.attn) should be in pruned_forward_var_ids"
    );
    // B3_OUT (VarId 24) should NOT be pruned — it's the terminal output.
    assert!(
        !result.pruned_forward_var_ids.contains(&24),
        "B3_OUT (VarId 24, the final output) should NOT be in pruned_forward_var_ids"
    );

    // wengert.output should still be B3_OUT (24) — we didn't prune the output layer.
    assert_eq!(
        wengert.output, 24,
        "wengert.output should still be 24 (B3_OUT) after pruning a non-terminal layer"
    );
}

// --------------------------------------------------------------------------
// Test 4: empty plan (no Prune decisions) → no rewrites, no refusals.
// --------------------------------------------------------------------------

#[test]
fn empty_plan_no_rewrites() {
    let mut wengert = mk_toy_wengert();
    let weight_map = WeightMap::default();
    let initial_op_count = wengert.ops.len();

    let applied = AppliedPlan {
        layers: vec![
            mk_keep_layer(0, "blocks.0.attn"),
            mk_keep_layer(1, "blocks.0.ffn"),
            mk_keep_layer(2, "blocks.1.attn"),
            mk_keep_layer(3, "blocks.1.ffn"),
            mk_keep_layer(4, "blocks.2.attn"),
            mk_keep_layer(5, "blocks.2.ffn"),
            mk_keep_layer(6, "blocks.3.attn"),
            mk_keep_layer(7, "blocks.3.ffn"),
        ],
        total_us: 80.0,
        peak_memory_bytes: 8192,
    };

    let result = run(&mut wengert, &applied, &weight_map);

    assert!(result.refusals.is_empty(), "no refusals expected for keep-all plan");
    assert!(result.rewrites.is_empty(), "no rewrites expected for keep-all plan");
    assert_eq!(result.ops_deleted, 0);
    assert_eq!(
        wengert.ops.len(),
        initial_op_count,
        "wengert should be unchanged when no layers are pruned"
    );
}

// --------------------------------------------------------------------------
// Test 5: whole-block prune attempt (blocks.1 with role=Block) → refusal
//   with WholeBlockUnsupported. Spec §3.6: v1 does not support whole-block
//   prune; only attn / ffn sub-layer prune is implemented.
// --------------------------------------------------------------------------

#[test]
fn whole_block_prune_refused() {
    let mut wengert = mk_toy_wengert();
    let weight_map = WeightMap::default();

    // "blocks.1" has role Block (inferred by wggo_graph::infer_role).
    let applied = AppliedPlan {
        layers: vec![
            mk_prune_layer(0, "blocks.1"),  // whole-block prune — should be refused
        ],
        total_us: 0.0,
        peak_memory_bytes: 0,
    };

    let result = run(&mut wengert, &applied, &weight_map);

    assert_eq!(result.rewrites.len(), 0, "no rewrites expected on refusal");
    assert_eq!(result.refusals.len(), 1, "expected exactly 1 refusal");
    assert!(
        matches!(&result.refusals[0], PruneRefusal::WholeBlockUnsupported { layer_name }
            if layer_name == "blocks.1"),
        "expected WholeBlockUnsupported for blocks.1, got: {:?}",
        result.refusals[0]
    );
}

// --------------------------------------------------------------------------
// Test 6 (Task 16 — merge-gate smoke test): mixed plan with one supported
//   sub-block prune (blocks.1.attn) AND one unsupported whole-block prune
//   (blocks.0) must:
//     (a) emit the WholeBlockUnsupported refusal for blocks.0
//     (b) apply zero rewrites (dry-run-then-commit invariant — any refusal
//         aborts the entire commit phase)
//     (c) leave wengert.ops and wengert.output completely unchanged
//
// Spec §11 criterion #7. Validates the three-phase run() orchestration from
// Task 11 under the realistic scenario of a partially-unsupported plan.
// --------------------------------------------------------------------------

#[test]
fn mixed_plan_emits_all_refusals_in_one_pass_with_wengert_untouched() {
    let mut wengert = mk_toy_wengert();
    let baseline_op_count = wengert.ops.len();
    let baseline_output = wengert.output;

    // Mixed plan: blocks.1.attn is a valid sub-block prune; blocks.0 is a
    // whole-block prune which is unsupported in v1 (LayerRole::Block).
    let plan = AppliedPlan {
        layers: vec![
            mk_prune_layer(2, "blocks.1.attn"), // supported sub-block prune
            mk_prune_layer(0, "blocks.0"),      // whole-block — should be refused
        ],
        total_us: 0.0,
        peak_memory_bytes: 0,
    };

    let result = run(&mut wengert, &plan, &WeightMap::default());

    // (a) WholeBlockUnsupported refusal must be present for blocks.0.
    let has_whole_block_refusal = result.refusals.iter().any(|r| {
        matches!(r, PruneRefusal::WholeBlockUnsupported { layer_name }
            if layer_name == "blocks.0")
    });
    assert!(
        has_whole_block_refusal,
        "expected WholeBlockUnsupported for blocks.0; got refusals: {:?}",
        result.refusals
    );

    // (b) Zero rewrites applied (dry-run-then-commit invariant).
    assert_eq!(
        result.rewrites.len(), 0,
        "expected zero rewrites applied in a mixed-refusal plan (dry-run-then-commit); got {}",
        result.rewrites.len()
    );
    assert_eq!(
        result.ops_deleted, 0,
        "expected ops_deleted=0 in a mixed-refusal plan; got {}",
        result.ops_deleted
    );

    // (c) wengert unchanged — op count and output pointer both preserved.
    assert_eq!(
        wengert.ops.len(), baseline_op_count,
        "wengert.ops should be untouched on refusal; count changed from {} to {}",
        baseline_op_count, wengert.ops.len()
    );
    assert_eq!(
        wengert.output, baseline_output,
        "wengert.output should be untouched on refusal"
    );
}
