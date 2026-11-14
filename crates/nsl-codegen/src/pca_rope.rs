//! PCA — document-aware RoPE position reset, kernel side.
//!
//! When sequences are packed, the CSHA kernels can reset RoPE positions at
//! each document boundary from a per-row `doc_starts` table held in shared
//! memory, instead of reading a `position_ids` tensor. This module holds
//! the table's compile-time bound ([`MAX_NUM_DOCS`]), the PTX that loads it
//! ([`emit_doc_starts_smem_load`]) and the Cranelift helpers that pass its
//! pointer (or the "no reset" sentinel) at a call site.
//!
//! No program reaches the table today: the route that would feed it from a
//! `DataLoader` is refused at compile time, and the working packed route
//! (`GroupedQueryAttention.forward_packed`) takes `position_ids` as a
//! tensor. With RoPE the reset changes nothing measurable in any case,
//! because RoPE scores depend only on the distance between two positions
//! of the same document.

use cranelift_codegen::ir::{types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataId, Module};

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

/// Emit the CTA-prologue PTX that loads ONE ROW's slice of `doc_starts`
/// from HBM into SMEM. Each CTA owns one (batch_idx, head_idx) element;
/// the CTA reads only its row's 1028-byte subtable from HBM offset
/// `batch_idx * (MAX_NUM_DOCS + 1) * 4` (= `batch_idx * 1028`).
///
/// `batch_idx` is reused from the existing u64 `%batch_idx` register
/// computed earlier in the prelude (`bid_y / heads`), so the caller does
/// NOT need to pass `heads` — the runtime quantity is already factored
/// into `%batch_idx`. This is a deviation from the original T5+T6 plan
/// signature `(ptx, heads)` and is the correct one (heads is not a
/// compile-time constant on FlashAttentionConfig / CshaExtras).
///
/// Sites 1-4 (Tasks 7-9) read from `smem_doc_starts[0..1028]` after this
/// prologue runs. Null `doc_starts_ptr` (== 0) is guarded: the per-iteration
/// `ld.global.s32` is predicated on `%p_doc_null` (`setp.eq.u64 ..., 0` on the
/// raw param) and writes `0` to `smem_doc_starts` instead of dereferencing,
/// so RoPE positions become `i - 0 = i` (standard) — the identity path for any
/// non-packed step. (PCA Tier A activation, spec 2026-05-25 §4.)
///
/// Requires registers declared by the sibling block in
/// `flash_attention_v2/phases/forward/prelude.rs` (gated on
/// `segment_masked && rope_q`).
pub fn emit_doc_starts_smem_load(ptx: &mut String) {
    // This 1028-byte `smem_doc_starts` region IS now counted in the static-vs-
    // extern budget: `fwd_needs_dynamic_smem` / `backward_needs_dynamic_smem`
    // route through `smem_layout::pca_smem_layout(.., rope_q).total`, which adds
    // the doc region when rope_q (2026-05-27 budget fix). It remains a SEPARATE
    // static `.shared` (NOT embedded in shmem[]); on the extern path that means
    // a static `.shared` coexists with `extern .shared shmem[]` — the Blackwell
    // mixed-layout pattern — which the 2026-05-12 SMEM probe + a 2026-05-27
    // reproduce found is SAFE on CUDA 13.2 / driver 591.86 / RTX 5070 Ti (the
    // illegal-address premise did not reproduce), so the embed was intentionally
    // not built. See `project_pca_smem_mixed_layout_crash_disproven` (memory).
    ptx.push_str("    // PCA sec.4.3 -- CTA prologue: load this row's doc_starts to SMEM\n");
    ptx.push_str("    .shared .align 4 .b8 smem_doc_starts[1028];\n");
    ptx.push_str("    ld.param.u64 %rd_doc_starts_ptr, [doc_starts_ptr];\n");
    // PCA §4.3 null-guard (spec §4.2): null doc_starts_ptr → write the all-zero
    // sentinel (doc_starts[0]=0 → RoPE position = i - 0 = i → standard positions).
    // Check the RAW param here, BEFORE the batch-row offset add below.
    ptx.push_str("    setp.eq.u64 %p_doc_null, %rd_doc_starts_ptr, 0;\n");
    // Narrow %batch_idx (u64) → %r_batch_idx (u32). batch_idx is small.
    ptx.push_str("    cvt.u32.u64 %r_batch_idx, %batch_idx;\n");
    // row_base_offset_bytes = batch_idx * 1028  (= (MAX_NUM_DOCS+1) * 4)
    ptx.push_str("    mul.lo.u32 %r_row_offset_elems, %r_batch_idx, 1028;\n");
    ptx.push_str("    cvt.u64.u32 %rd_doc_starts_addr, %r_row_offset_elems;\n");
    ptx.push_str("    add.u64 %rd_doc_starts_ptr, %rd_doc_starts_ptr, %rd_doc_starts_addr;\n");
    // Convert the smem_doc_starts label to a generic-space u64 base so
    // store addresses can be built via plain u64 add — ptxas rejects the
    // `[symbol + %reg]` mixed form (same constraint that caused the
    // Tier A `seg_smem` cvta.shared.u64 pattern; see prelude.rs note).
    ptx.push_str("    cvta.shared.u64 %r_doc_smem_base, smem_doc_starts;\n");
    // Parallel cooperative load: each thread loads multiple i32s from the row.
    // block_x >= 128 for all CSHA configs; (MAX_NUM_DOCS+1)=257 entries to load.
    // With block_x=128 each thread issues ceil(257/128)=3 i32 loads max.
    //
    // Materialize %ntid.x into a regular register up-front — ptxas rejects
    // special registers (%ntid.x, %tid.x, etc.) as direct add/sub operands.
    ptx.push_str("    mov.u32 %r_doc_starts_idx, %tid.x;\n");
    ptx.push_str("    mov.u32 %r_doc_starts_stride, %ntid.x;\n");
    ptx.push_str("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP:\n");
    ptx.push_str("    setp.ge.u32 %p_doc_load_done, %r_doc_starts_idx, 257;\n");
    ptx.push_str("    @%p_doc_load_done bra V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP_END;\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
    // Build the HBM (global) source address.
    ptx.push_str("    cvt.u64.u32 %rd_doc_starts_addr, %r_doc_starts_byte_off;\n");
    ptx.push_str("    add.u64 %rd_doc_starts_addr, %rd_doc_starts_addr, %rd_doc_starts_ptr;\n");
    ptx.push_str("    @%p_doc_null bra V2_PCA_ROPE_DOC_NULL_LD;\n");
    ptx.push_str("    ld.global.s32 %r_doc_start, [%rd_doc_starts_addr];\n");
    ptx.push_str("    bra V2_PCA_ROPE_DOC_LD_DONE;\n");
    ptx.push_str("V2_PCA_ROPE_DOC_NULL_LD:\n");
    ptx.push_str("    mov.s32 %r_doc_start, 0;\n");
    ptx.push_str("V2_PCA_ROPE_DOC_LD_DONE:\n");
    // Build the SMEM destination address (generic-space u64 from cvta.shared
    // earlier; offset added in u64 to avoid mixed-width arithmetic).
    ptx.push_str("    cvt.u64.u32 %rd_doc_smem_addr, %r_doc_starts_byte_off;\n");
    ptx.push_str("    add.u64 %rd_doc_smem_addr, %rd_doc_smem_addr, %r_doc_smem_base;\n");
    ptx.push_str("    st.shared.s32 [%rd_doc_smem_addr], %r_doc_start;\n");
    ptx.push_str("    add.u32 %r_doc_starts_idx, %r_doc_starts_idx, %r_doc_starts_stride;\n");
    ptx.push_str("    bra V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP;\n");
    ptx.push_str("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP_END:\n");
    ptx.push_str("    bar.sync 0;\n");
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
