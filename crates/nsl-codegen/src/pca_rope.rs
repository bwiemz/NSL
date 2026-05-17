//! PCA — document-aware RoPE position reset.
//!
//! When sequences are packed, positions must *reset* at each document
//! boundary so that each document sees RoPE positions `0..len_i`.  In the
//! standard flow this requires a separate `position_ids: [S]` tensor
//! constructed by the DataLoader.  PCA eliminates this tensor by fusing
//! the position-reset computation into the RoPE kernel's epilogue (the
//! same one CSHA Level 1 already fuses into the Q/K projections).
//!
//! This module computes the per-position offset function that the fused
//! RoPE kernel evaluates.  Rather than emitting PTX directly, we produce
//! a [`RopePositionPlan`] with three representations:
//!
//!   * `doc_starts: [num_docs + 1]` — cumulative start offsets
//!   * `position_offsets: [S]` — resolved `i - doc_start(segment_id(i))`
//!   * `formula` — a textual representation for CLI reports
//!
//! Only `doc_starts` is consumed by the kernel (the other representations
//! are for testing and diagnostics).

use cranelift_codegen::ir::{types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataId, Module};
use serde::Serialize;

use crate::pca_tileskip::PackingLayout;

/// PCA §4.3 — RoPE position-reset compile-time bound.
///
/// Upper bound on the number of documents per packed batch. The fixed
/// `[MAX_NUM_DOCS + 1]` SMEM layout (1028 bytes) lets the kernel emit a
/// constant-size doc_starts load independent of runtime num_docs. The
/// packer asserts num_docs ≤ MAX_NUM_DOCS at batch-construction time.
///
/// 256 covers pretraining (avg doc ~256-2048 tok/doc at seq=16384), chat
/// SFT (avg ~50-150 tok/turn × 4-8 turns), and short-prompt instruction
/// tuning. SMEM cost 1028 bytes — trivial against the joint Tier A + Tier B
/// + RoPE-reset budget.
pub const MAX_NUM_DOCS: u32 = 256;

const _: () = assert!(MAX_NUM_DOCS <= 4096, "MAX_NUM_DOCS bound — keeps SMEM layout small");
const _: () = assert!(
    (MAX_NUM_DOCS + 1) * 4 <= 2048,
    "doc_starts SMEM bake must stay well under per-CTA budget"
);

// SMEM joint bake bound — CONSERVATIVE CEILING. The project ships kernels
// ptxas-clean on sm_75 (48 KB usable per CTA). The 16384+16384 figures are
// upper-bound ceilings for Tier B's range-table region (TIER_B_MAX_BAKED_
// SEQ_LEN per the planner spec, PR #175) and Tier A's segment_ids region
// (≤2*seq_len bytes at u16 packing, capped at seq_len=8192 for conservatism).
// Actual shipped joint usage is lower; this assert protects against
// catastrophic regression, not as a precise budget. Active development
// target is sm_120 (RTX 5070 Ti, 100 KB) but the floor stays sm_75.
const _: () = assert!(
    (MAX_NUM_DOCS + 1) * 4 + 16384 + 16384 < 48 * 1024,
    "Tier B + Tier A + RoPE-reset SMEM joint bake must fit sm_75 limit"
);

/// Construct a sentinel-zero `doc_starts_ptr` Value at a Cranelift call site.
/// Identity-position semantics (matches pre-spec behavior).
#[inline]
pub fn doc_starts_disabled_sentinel(builder: &mut FunctionBuilder<'_>) -> Value {
    builder.ins().iconst(types::I64, 0)
}

/// Construct an enabled `doc_starts_ptr` Value pointing at a device tensor.
///
/// The caller MUST ensure `data_id` was declared with element type `i32`.
/// This helper does NOT type-check the data — passing a `DataId` for a
/// different element type will silently corrupt loads downstream. See
/// the spec's §2 for the producer-side i32 invariant.
#[inline]
pub fn doc_starts_enabled<M: Module>(
    builder: &mut FunctionBuilder<'_>,
    module: &mut M,
    data_id: DataId,
) -> Value {
    let gv = module.declare_data_in_func(data_id, builder.func);
    builder.ins().symbol_value(types::I64, gv)
}

/// RoPE reset plan: how each token's position is computed inside the
/// fused RoPE kernel.
#[derive(Debug, Clone, Serialize)]
pub struct RopePositionPlan {
    /// Document starts `[num_docs + 1]`.
    pub doc_starts: Vec<u32>,
    /// Resolved per-position offsets, for testing.
    pub position_offsets: Vec<u32>,
    /// Pseudocode describing the kernel-level computation.
    pub formula: String,
    /// Whether the plan is a true packing plan (true) or a degenerate
    /// identity (false — single document, no reset needed).
    pub needs_reset: bool,
}

impl RopePositionPlan {
    /// Number of documents in the packed sample.
    pub fn num_docs(&self) -> u32 {
        self.doc_starts.len().saturating_sub(1) as u32
    }

    /// Packed sequence length.
    pub fn packed_length(&self) -> u32 {
        self.doc_starts.last().copied().unwrap_or(0)
    }
}

/// Compute the per-position offset `i - doc_start(segment_id(i))` for a
/// packing layout.  Returns a plan the fused RoPE kernel consumes.
pub fn plan(layout: &PackingLayout) -> RopePositionPlan {
    let doc_starts = layout.doc_starts();
    let segments = layout.segment_ids();
    let mut offsets = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let start = doc_starts[*seg as usize];
        offsets.push(i as u32 - start);
    }
    let needs_reset = layout.doc_lengths.len() > 1;
    let formula = if needs_reset {
        "position[i] = i - doc_starts[segment_ids[i]]".to_string()
    } else {
        "position[i] = i  (single document — no reset)".to_string()
    };
    RopePositionPlan {
        doc_starts,
        position_offsets: offsets,
        formula,
        needs_reset,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_doc_is_identity() {
        let p = plan(&PackingLayout::from_docs(vec![5]));
        assert!(!p.needs_reset);
        assert_eq!(p.position_offsets, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn three_docs_reset_positions() {
        let p = plan(&PackingLayout::from_docs(vec![3, 2, 4]));
        assert!(p.needs_reset);
        assert_eq!(
            p.position_offsets,
            vec![0, 1, 2, 0, 1, 0, 1, 2, 3]
        );
    }

    #[test]
    fn doc_starts_match_cumulative_sum() {
        let p = plan(&PackingLayout::from_docs(vec![3, 2, 4]));
        assert_eq!(p.doc_starts, vec![0, 3, 5, 9]);
        assert_eq!(p.packed_length(), 9);
        assert_eq!(p.num_docs(), 3);
    }

    #[test]
    fn empty_layout_is_trivial() {
        let p = plan(&PackingLayout::from_docs(Vec::new()));
        assert_eq!(p.num_docs(), 0);
        assert_eq!(p.packed_length(), 0);
        assert!(p.position_offsets.is_empty());
    }

    #[test]
    fn long_single_doc_keeps_linear() {
        let p = plan(&PackingLayout::from_docs(vec![128]));
        for (i, off) in p.position_offsets.iter().enumerate() {
            assert_eq!(*off, i as u32);
        }
    }

    #[test]
    fn formula_contains_reset_text_when_reset_needed() {
        let p = plan(&PackingLayout::from_docs(vec![3, 2]));
        assert!(p.formula.contains("segment_ids"));
        let p2 = plan(&PackingLayout::from_docs(vec![5]));
        assert!(p2.formula.contains("single document"));
    }

    #[test]
    fn max_num_docs_is_256() {
        assert_eq!(MAX_NUM_DOCS, 256);
    }

    #[test]
    fn doc_starts_smem_size_bytes_is_1028() {
        assert_eq!((MAX_NUM_DOCS + 1) * 4, 1028);
    }

    #[test]
    fn doc_starts_disabled_sentinel_is_zero_constant() {
        use cranelift_codegen::ir::Function;
        use cranelift_codegen::ir::Signature;
        use cranelift_codegen::ir::UserFuncName;
        use cranelift_codegen::isa::CallConv;
        use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

        let mut fn_ctx = FunctionBuilderContext::new();
        let sig = Signature::new(CallConv::SystemV);
        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
        let mut builder = FunctionBuilder::new(&mut func, &mut fn_ctx);
        let block = builder.create_block();
        builder.switch_to_block(block);

        let v = doc_starts_disabled_sentinel(&mut builder);
        // The returned Value should be the result of an iconst(I64, 0).
        let inst = builder.func.dfg.value_def(v).unwrap_inst();
        let opcode = builder.func.dfg.insts[inst].opcode();
        assert_eq!(opcode.to_string(), "iconst");

        // Verify value == 0 (the spec's identity-position sentinel value).
        let inst_data = builder.func.dfg.insts[inst];
        match inst_data {
            cranelift_codegen::ir::InstructionData::UnaryImm { imm, .. } => {
                assert_eq!(imm.bits(), 0, "sentinel must be value 0, got {}", imm.bits());
            }
            other => panic!("expected UnaryImm for iconst, got {:?}", other),
        }
        // Verify type is I64.
        let result_type = builder.func.dfg.value_type(v);
        assert_eq!(result_type, types::I64, "sentinel must be I64");
    }
}
