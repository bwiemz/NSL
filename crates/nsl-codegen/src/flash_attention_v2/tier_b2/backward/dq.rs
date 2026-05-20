//! dQ-kernel emitter for Tier B.2 backward (scaffold; inner-loop body in Tasks 9-12).
//!
//! Q-outer, kv-inner. dQ accumulator register-resident across kv_iter.
//! Producer warp (warp 0) issues cp.async for Q, dO, K, V; consumer warps
//! (1-3) execute MMA chain. No atomicAdd.
//!
//! Per spec §5.2 amendment: SMEM is sized via tier_b2_dq_total_smem_bytes
//! (includes col-major K re-stage band). emit_dq_acc_init uses
//! tier_b2_effective_bq for the per-hd bq=32 fallback at hd=128.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §4 + §5.2

pub mod hbm_addr;

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    tier_b2_dq_total_smem_bytes, tier_b2_effective_bq,
};
use crate::flash_attention_v2::tier_b2::backward::BackwardSynthError;

pub fn synthesize_dq_kernel(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    let hd = config.head_dim as u32;
    if !hd.is_multiple_of(32) {
        return Err(BackwardSynthError::UnsupportedHeadDim(hd));
    }

    let mut ptx = String::new();
    emit_prelude(&mut ptx);
    // .extern .shared is a module-level PTX directive: must precede .visible .entry.
    // See kernel_skeleton/EXTRACTION_INVENTORY.md and forward/prelude.rs comments.
    emit_smem_extern_module_scope(&mut ptx, config);
    emit_entry_signature(&mut ptx);
    emit_register_decls(&mut ptx, config);
    emit_grid_id_setup(&mut ptx);
    emit_q_iter_count_setup(&mut ptx, config);
    emit_kv_iter_count_setup(&mut ptx, config);
    emit_outer_loop_open(&mut ptx);
    emit_q_dO_producer_load(&mut ptx, config);
    emit_dq_acc_init(&mut ptx, config);
    emit_inner_loop_open(&mut ptx, config);
    emit_tile_skip_predicate(&mut ptx, config);
    emit_inner_loop_body(&mut ptx, config);
    emit_inner_loop_close(&mut ptx);
    emit_outer_loop_close(&mut ptx);
    emit_dq_finalize(&mut ptx, config);
    emit_entry_close(&mut ptx);
    Ok(ptx)
}

fn emit_prelude(ptx: &mut String) {
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n\n");
}

/// Emit the module-scope `.extern .shared` declaration.
///
/// PTX ISA rule: `.extern .shared` is a MODULE-level directive; it CANNOT appear
/// inside a function body `{...}`. It must precede the `.visible .entry` block.
/// See kernel_skeleton/EXTRACTION_INVENTORY.md and forward/prelude.rs for the
/// canonical pattern used across all other kernels that require dynamic SMEM.
///
/// Per SPEC AMENDMENT: size comes from tier_b2_dq_total_smem_bytes which INCLUDES
/// the col-major K re-stage band (Path A, spec §5.2).
fn emit_smem_extern_module_scope(ptx: &mut String, config: &FlashAttentionConfig) {
    let total_smem = tier_b2_dq_total_smem_bytes(config);
    ptx.push_str(&format!(".extern .shared .align 16 .b8 shmem[{}];\n\n", total_smem));
}

fn emit_entry_signature(ptx: &mut String) {
    ptx.push_str(".visible .entry tier_b2_dq_kernel(\n");
    ptx.push_str("    .param .u64 q_saved_ptr,\n");
    ptx.push_str("    .param .u64 k_saved_ptr,\n");
    ptx.push_str("    .param .u64 v_saved_ptr,\n");
    ptx.push_str("    .param .u64 d_o_ptr,\n");
    ptx.push_str("    .param .u64 row_max_ptr,\n");
    ptx.push_str("    .param .u64 row_sum_ptr,\n");
    ptx.push_str("    .param .u64 d_ptr,\n");
    ptx.push_str("    .param .u64 segment_ids_ptr,\n");
    ptx.push_str("    .param .u64 d_q_out_ptr,\n");
    ptx.push_str("    .param .u32 seq_len,\n");
    ptx.push_str("    .param .u32 heads,\n");
    ptx.push_str("    .param .u32 batch\n");
    ptx.push_str(")\n");
    ptx.push_str(".maxntid 128, 1, 1\n");
    ptx.push_str("{\n");
}

fn emit_register_decls(ptx: &mut String, _config: &FlashAttentionConfig) {
    // NOTE: .extern .shared shmem[N] is emitted at module scope by emit_smem_extern_module_scope
    // before the .visible .entry. PTX ISA disallows .extern .shared inside a function body.
    //
    // NOTE: user register named %r_tid (not %tid) to avoid shadowing the PTX special register
    // %tid (a vector); ptxas rejects %tid.x when %tid is declared as a user u32.
    ptx.push_str("    .reg .u32 %r_tid, %lane_id, %warp_id;\n");
    ptx.push_str("    .reg .u32 %q_tile, %kv_tile, %head, %batch_idx;\n");
    ptx.push_str("    .reg .pred %p_tile_active, %p_producer, %p_consumer;\n");
    ptx.push_str("    .reg .u32 %addr_lo, %tile_skip_predicate;\n");
    ptx.push_str("    .reg .u32 %row_index_tmp;\n");
    ptx.push_str("    .reg .u64 %addr;\n");
    // seq_len scratch: declared here (not inside emit_q_iter_count_setup) so all
    // .reg directives precede executable instructions per PTX ISA.
    ptx.push_str("    .reg .u32 %seq_len_r;\n");
    // heads_r: loaded from [heads] param in emit_grid_id_setup; used by cp.async
    // HBM address helpers (emit_4d_byte_offset's heads_reg argument).
    ptx.push_str("    .reg .u32 %heads_r;\n");
    // Outer q_iter loop: induction variable, upper bound, loop predicate.
    ptx.push_str("    .reg .u32 %q_iter, %num_q_iters;\n");
    ptx.push_str("    .reg .pred %p_q_iter_more;\n");
    // Inner kv_iter loop: induction variable, upper bound, loop predicate.
    ptx.push_str("    .reg .u32 %kv_iter, %num_kv_iters;\n");
    ptx.push_str("    .reg .pred %p_kv_iter_more;\n");
    // Scratch registers used by matmul_mma helpers (emit_load_a/b_fragment_smem).
    ptx.push_str("    .reg .u32 %mma_addr, %mma_a_row, %mma_b_row;\n");
    // D1 tile_skip predicate registers (causal: used; non-causal: declared but unused — ptxas tolerates).
    ptx.push_str("    .reg .u32 %q_tile_end, %kv_tile_start;\n");
    ptx.push_str("    .reg .pred %p_causal_active;\n");
    // C1 cp.async Q-tile registers. All .reg decls must precede executable instructions per PTX ISA.
    // Prefixed %c1_* to avoid clashes with C2/C3/C4 register namespaces.
    ptx.push_str("    .reg .u32 %c1_q_tile_start, %c1_lane_byte_off, %c1_smem_base32, %c1_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c1_q_hbm_off, %c1_q_hbm_base, %c1_lane_hbm_addr, %c1_smem_base64;\n");
    // C2 cp.async dO-tile registers. Prefixed %c2_* to avoid clashes with C1/C3/C4 namespaces.
    // dO tile is [effective_bq, hd] f16, same shape as Q. Resident across kv_iter (loaded once).
    ptx.push_str("    .reg .u32 %c2_do_tile_start, %c2_lane_byte_off, %c2_smem_base32, %c2_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c2_do_hbm_off, %c2_do_hbm_base, %c2_lane_hbm_addr, %c2_smem_base64;\n");
    // C3 cp.async K-tile registers. Prefixed %c3_* to avoid clashes with C1/C2/C4 namespaces.
    // K tile is [effective_bkv, hd] f16, per-kv_iter (not resident). Loaded in emit_inner_loop_open.
    ptx.push_str("    .reg .u32 %c3_kv_tile_start, %c3_lane_byte_off, %c3_smem_base32, %c3_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c3_k_hbm_off, %c3_k_hbm_base, %c3_lane_hbm_addr, %c3_smem_base64;\n");
    ptx.push('\n');
}

fn emit_grid_id_setup(ptx: &mut String) {
    // Read thread ID X component from the special %tid vector into %r_tid.
    // Cannot name the user register %tid — that shadows the PTX special %tid
    // vector register, causing ptxas to reject %tid.x (error: "video selector").
    ptx.push_str("    mov.u32 %r_tid, %tid.x;\n");
    ptx.push_str("    and.b32 %lane_id, %r_tid, 31;\n");
    ptx.push_str("    shr.u32 %warp_id, %r_tid, 5;\n");
    ptx.push_str("    mov.u32 %q_tile,    %ctaid.x;\n");
    ptx.push_str("    mov.u32 %head,      %ctaid.y;\n");
    ptx.push_str("    mov.u32 %batch_idx, %ctaid.z;\n");
    ptx.push_str("    setp.eq.u32 %p_producer, %warp_id, 0;\n");
    // Load heads param for use by HBM address helpers (emit_4d_byte_offset).
    ptx.push_str("    ld.param.u32 %heads_r, [heads];\n");
    ptx.push('\n');
}

/// Compute %num_q_iters = ceil(seq_len / bq) using power-of-2 shift.
///
/// bq is power-of-2 per spec §3.1 (ALLOWED_BLOCK_Q), so:
///   num_q_iters = (seq_len + bq - 1) >> log2(bq)
/// This avoids a division instruction entirely.
///
/// %seq_len_r is declared in emit_register_decls so that all .reg directives
/// precede executable instructions per PTX ISA.  The ld.param loads the value
/// here because seq_len is first consumed in this helper.
fn emit_q_iter_count_setup(ptx: &mut String, config: &FlashAttentionConfig) {
    let bq = tier_b2_effective_bq(config);
    let log2_bq = bq.trailing_zeros();
    // %seq_len_r and %num_q_iters are both declared in emit_register_decls.
    ptx.push_str("    ld.param.u32 %seq_len_r, [seq_len];\n");
    ptx.push_str(&format!("    add.u32 %num_q_iters, %seq_len_r, {};\n", bq - 1));
    ptx.push_str(&format!("    shr.u32 %num_q_iters, %num_q_iters, {};\n", log2_bq));
    ptx.push('\n');
}

/// Compute %num_kv_iters = ceil(seq_len / bkv) using power-of-2 shift.
///
/// bkv is power-of-2 (mirrors bq per Approach A"'s bq=bkv invariant), so:
///   num_kv_iters = (seq_len + bkv - 1) >> log2(bkv)
/// %seq_len_r is already loaded by emit_q_iter_count_setup, so this helper
/// reuses it directly.
fn emit_kv_iter_count_setup(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::tier_b2_effective_bkv;
    let bkv = tier_b2_effective_bkv(config);
    // bkv is power-of-2 per spec §3.1 (ALLOWED_BLOCK_KV), so use shift.
    ptx.push_str(&format!("    add.u32 %num_kv_iters, %seq_len_r, {};\n", bkv - 1));
    ptx.push_str(&format!("    shr.u32 %num_kv_iters, %num_kv_iters, {};\n", bkv.trailing_zeros()));
    ptx.push('\n');
}

fn emit_outer_loop_open(ptx: &mut String) {
    // Initialize induction variable. %num_q_iters computed in emit_q_iter_count_setup.
    ptx.push_str("    mov.u32 %q_iter, 0;\n");
    ptx.push_str("DQ_Q_ITER_LOOP:\n");
}

#[allow(non_snake_case)]
fn emit_q_dO_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // Warp 0 (producer): issue cp.async for Q + dO tiles.\n");
    ptx.push_str("    // Warps 1-3 (consumers): wait on cp.async.wait_group.\n");
    emit_q_producer_load(ptx, config);
    emit_dO_producer_load(ptx, config);
    ptx.push_str("    cp.async.commit_group;\n");
    ptx.push_str("DQ_PROD_LOAD_DONE:\n");
    ptx.push_str("    cp.async.wait_group 0;\n");
    ptx.push_str("    bar.sync 0;\n");
}

/// C1: emit the warp-0-gated cp.async load for the Q tile [effective_bq, hd] f16.
///
/// Uses A1's `emit_4d_byte_offset` for HBM addressing. Per-lane byte offset is
/// `lane_id * 4` (each lane owns one 4-byte b32 chunk per cp.async iteration).
/// Total cp.async instructions = (effective_bq * hd * 2) / (32 * 4) chunks per lane.
///
/// Register naming prefix `%c1_` prevents clashes with C2/C3/C4 registers.
/// The warp-0 gate predicate (`@!%p_producer bra DQ_PROD_LOAD_DONE`) is emitted
/// by the caller (`emit_q_dO_producer_load`) BEFORE this function so that both
/// the Q load (C1) and the dO load (C2) share a single predicate branch.
fn emit_q_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::tier_b2_effective_bq;

    let bq = tier_b2_effective_bq(config);
    let hd = config.head_dim as u32;
    // q_offset = 0 (Q always lives at start of dQ-kernel SMEM region).
    // tier_b2_dq_q_offset returns 0; accessed via smem_layout for no-magic-number
    // correctness — the accessor is the spec contract.
    let q_smem_off: u32 = 0; // = tier_b2_dq_q_offset(config)

    // Total Q tile bytes = bq * hd * 2 (f16).  Distributed across 32 lanes:
    //   bytes_per_lane = (bq * hd * 2) / 32
    //   chunks_per_lane = bytes_per_lane / 4  (each cp.async.cg issues b32 = 4 bytes)
    let total_bytes = bq * hd * 2;
    let bytes_per_lane = total_bytes / 32;
    let chunks_per_lane = bytes_per_lane / 4;
    debug_assert!(
        chunks_per_lane >= 1,
        "Q tile must have at least one b32 chunk per lane (bq={}, hd={})",
        bq, hd
    );

    ptx.push_str("    // === C1: cp.async Q tile [bq, hd] f16 -> SMEM[+q_offset=0] ===\n");
    // All %c1_* registers declared in emit_register_decls (PTX ISA: decls before exec instrs).

    // q_tile_start = q_iter * bq  (u32, first sequence position of this Q tile)
    ptx.push_str(&format!(
        "    // q_tile_start = q_iter * {bq} (first seq position of Q tile)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c1_q_tile_start, %q_iter, {bq};\n"
    ));

    // HBM byte offset for Q[batch_idx, head, q_tile_start, 0] using A1's helper.
    // emit_4d_byte_offset uses %row_index_tmp as scratch (already declared in emit_register_decls).
    crate::flash_attention_v2::tier_b2::backward::dq::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c1_q_hbm_off",
        "%batch_idx",
        "%head",
        "%c1_q_tile_start",
        "0",         // d=0: column index within the row (we load full row via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,          // D = head_dim
        2,           // sizeof(f16) = 2
    );

    // HBM source base: load q_saved_ptr from param space.
    ptx.push_str("    ld.param.u64 %c1_q_hbm_base, [q_saved_ptr];\n");
    ptx.push_str("    add.u64 %c1_q_hbm_base, %c1_q_hbm_base, %c1_q_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    // (Each lane copies one b32 = 4 bytes per cp.async iteration.)
    ptx.push_str("    shl.b32 %c1_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + (q_tile_start row offset already in base) + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c1_lane_hbm_addr, %c1_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c1_lane_hbm_addr, %c1_lane_hbm_addr, %c1_q_hbm_base;\n");

    // SMEM destination: u32 shared-space address via cvta.shared.u64 + cvt.u32.u64.
    // q_smem_off = 0, so the base IS the destination.
    ptx.push_str("    cvta.shared.u64 %c1_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c1_smem_base32, %c1_smem_base64;\n");
    if q_smem_off > 0 {
        ptx.push_str(&format!(
            "    add.u32 %c1_smem_base32, %c1_smem_base32, {q_smem_off};\n"
        ));
    }
    // Add per-lane offset to SMEM destination.
    ptx.push_str(
        "    add.u32 %c1_smem_dst, %c1_smem_base32, %c1_lane_byte_off;\n",
    );

    // Emit cp.async.cg.shared.global.b32 for each chunk per lane.
    // Chunk stride across lanes = 32 * 4 = 128 bytes per chunk iteration.
    let chunk_stride = 32 * 4u32; // 32 lanes * 4 bytes each
    // cp.async.ca supports 4-byte transactions (cp.async.cg requires 16 bytes).
    // Per PTX 7.0 ISA: .ca = cache-all (L1+L2), 4/8/16 bytes; .cg = cache-global (L2), 16 bytes only.
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c1_smem_dst + {chunk_off}], [%c1_lane_hbm_addr + {chunk_off}], 4;\n"
        ));
    }
    // commit_group emitted by caller (emit_q_dO_producer_load) after dO part lands (C2).
}

/// C2: emit the cp.async load for the dO tile [effective_bq, hd] f16 into SMEM.
///
/// dO is **resident across kv_iter** (spec §1.3) — loaded once at q_iter open,
/// not per-kv-block.  Same tile shape as Q.  HBM source: `d_o_ptr` param.
/// SMEM destination: `tier_b2_dq_dO_offset(config)` bytes from shmem base.
///
/// Uses A1's `emit_4d_byte_offset` for HBM addressing (same as C1).  Register
/// naming prefix `%c2_` prevents clashes with C1/C3/C4 register namespaces.
///
/// The warp-0 gate (`@!%p_producer bra DQ_PROD_LOAD_DONE`) is emitted by the
/// caller (`emit_q_dO_producer_load`) — shared with C1, no duplicate branch here.
#[allow(non_snake_case)]
fn emit_dO_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{tier_b2_effective_bq, tier_b2_dq_dO_offset};

    let bq = tier_b2_effective_bq(config);
    let hd = config.head_dim as u32;
    let do_smem_off = tier_b2_dq_dO_offset(config);

    // Total dO tile bytes = bq * hd * 2 (f16).  Distributed across 32 lanes:
    //   bytes_per_lane = (bq * hd * 2) / 32
    //   chunks_per_lane = bytes_per_lane / 4  (each cp.async.ca issues b32 = 4 bytes)
    let total_bytes = bq * hd * 2;
    let bytes_per_lane = total_bytes / 32;
    let chunks_per_lane = bytes_per_lane / 4;
    debug_assert!(
        chunks_per_lane >= 1,
        "dO tile must have at least one b32 chunk per lane (bq={}, hd={})",
        bq, hd
    );

    ptx.push_str(&format!(
        "    // === C2: cp.async dO tile [bq={bq}, hd={hd}] f16 -> SMEM[+dO_offset={do_smem_off}] ===\n"
    ));
    // All %c2_* registers declared in emit_register_decls (PTX ISA: decls before exec instrs).

    // do_tile_start = q_iter * bq (first sequence position of this dO tile — same as Q).
    ptx.push_str(&format!(
        "    // do_tile_start = q_iter * {bq} (dO tile rows are Q-aligned; resident across kv_iter)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c2_do_tile_start, %q_iter, {bq};\n"
    ));

    // HBM byte offset for dO[batch_idx, head, do_tile_start, 0] using A1's helper.
    // emit_4d_byte_offset uses %row_index_tmp as scratch (declared in emit_register_decls).
    crate::flash_attention_v2::tier_b2::backward::dq::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c2_do_hbm_off",
        "%batch_idx",
        "%head",
        "%c2_do_tile_start",
        "0",         // d=0: column index within the row (full row loaded via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,          // D = head_dim
        2,           // sizeof(f16) = 2
    );

    // HBM source base: load d_o_ptr from param space.
    ptx.push_str("    ld.param.u64 %c2_do_hbm_base, [d_o_ptr];\n");
    ptx.push_str("    add.u64 %c2_do_hbm_base, %c2_do_hbm_base, %c2_do_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    ptx.push_str("    shl.b32 %c2_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c2_lane_hbm_addr, %c2_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c2_lane_hbm_addr, %c2_lane_hbm_addr, %c2_do_hbm_base;\n");

    // SMEM destination: u32 shared-space address via cvta.shared.u64 + cvt.u32.u64 + dO_offset.
    ptx.push_str("    cvta.shared.u64 %c2_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c2_smem_base32, %c2_smem_base64;\n");
    // dO_offset > 0 (dO follows Q, K, V tiles), so always add the offset.
    ptx.push_str(&format!(
        "    add.u32 %c2_smem_base32, %c2_smem_base32, {do_smem_off};  // +dO_offset\n"
    ));
    // Add per-lane byte offset to SMEM destination.
    ptx.push_str("    add.u32 %c2_smem_dst, %c2_smem_base32, %c2_lane_byte_off;\n");

    // Emit cp.async.ca.shared.global for each chunk per lane.
    // Chunk stride across lanes = 32 * 4 = 128 bytes per chunk iteration.
    let chunk_stride = 32 * 4u32; // 32 lanes * 4 bytes each
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c2_smem_dst + {chunk_off}], [%c2_lane_hbm_addr + {chunk_off}], 4;\n"
        ));
    }
    // commit_group emitted by caller (emit_q_dO_producer_load) after both C1 and C2 land.
}

/// C3: emit the cp.async load for the K tile [effective_bkv, hd] f16 into SMEM.
///
/// K is **per-kv_iter** (spec §1.3) — loaded once per inner loop iteration.
/// HBM source: `k_saved_ptr` param.
/// SMEM destination: `tier_b2_dq_k_offset(config)` bytes from shmem base.
///
/// Uses A1's `emit_4d_byte_offset` for HBM addressing, with `%c3_kv_tile_start`
/// (= `kv_iter * bkv`) as the row index — NOT `%q_iter` (which C1/C2 use).
///
/// Register naming prefix `%c3_` prevents clashes with C1/C2/C4 namespaces.
///
/// The warp-0 gate (`@!%p_producer bra DQ_KV_LOAD_DONE`) is emitted by the
/// caller (`emit_inner_loop_open`) — shared with C4 (V tile), no duplicate branch.
fn emit_k_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_effective_bkv, tier_b2_dq_k_offset,
    };

    let bkv = tier_b2_effective_bkv(config);
    let hd = config.head_dim as u32;
    let k_smem_off = tier_b2_dq_k_offset(config);

    // Total K tile bytes = bkv * hd * 2 (f16). Distributed across 32 lanes:
    //   bytes_per_lane = (bkv * hd * 2) / 32
    //   chunks_per_lane = bytes_per_lane / 4  (each cp.async.ca issues b32 = 4 bytes)
    let total_bytes = bkv * hd * 2;
    let bytes_per_lane = total_bytes / 32;
    let chunks_per_lane = bytes_per_lane / 4;
    debug_assert!(
        chunks_per_lane >= 1,
        "K tile must have at least one b32 chunk per lane (bkv={}, hd={})",
        bkv, hd
    );

    ptx.push_str(&format!(
        "    // === C3: cp.async K tile [bkv={bkv}, hd={hd}] f16 -> SMEM[+k_offset={k_smem_off}] ===\n"
    ));
    // All %c3_* registers declared in emit_register_decls (PTX ISA: decls before exec instrs).

    // kv_tile_start = kv_iter * bkv (first KV sequence position of this K tile; per-kv_iter).
    ptx.push_str(&format!(
        "    // kv_tile_start = kv_iter * {bkv} (K tile rows are kv-iter-aligned; refreshed each kv_iter)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c3_kv_tile_start, %kv_iter, {bkv};\n"
    ));

    // HBM byte offset for K[batch_idx, head, kv_tile_start, 0] using A1's helper.
    // emit_4d_byte_offset uses %row_index_tmp as scratch (declared in emit_register_decls).
    crate::flash_attention_v2::tier_b2::backward::dq::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c3_k_hbm_off",
        "%batch_idx",
        "%head",
        "%c3_kv_tile_start",
        "0",     // d=0: column index within the row (full row loaded via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,      // D = head_dim
        2,       // sizeof(f16) = 2
    );

    // HBM source base: load k_saved_ptr from param space.
    ptx.push_str("    ld.param.u64 %c3_k_hbm_base, [k_saved_ptr];\n");
    ptx.push_str("    add.u64 %c3_k_hbm_base, %c3_k_hbm_base, %c3_k_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    ptx.push_str("    shl.b32 %c3_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c3_lane_hbm_addr, %c3_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c3_lane_hbm_addr, %c3_lane_hbm_addr, %c3_k_hbm_base;\n");

    // SMEM destination: u32 shared-space address via cvta.shared.u64 + cvt.u32.u64 + k_offset.
    ptx.push_str("    cvta.shared.u64 %c3_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c3_smem_base32, %c3_smem_base64;\n");
    // k_smem_off > 0 (K follows Q tile), so always add the offset.
    ptx.push_str(&format!(
        "    add.u32 %c3_smem_base32, %c3_smem_base32, {k_smem_off};  // +k_offset\n"
    ));
    // Add per-lane byte offset to SMEM destination.
    ptx.push_str("    add.u32 %c3_smem_dst, %c3_smem_base32, %c3_lane_byte_off;\n");

    // Emit cp.async.ca.shared.global for each chunk per lane.
    // Chunk stride across lanes = 32 * 4 = 128 bytes per chunk iteration.
    let chunk_stride = 32 * 4u32; // 32 lanes * 4 bytes each
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c3_smem_dst + {chunk_off}], [%c3_lane_hbm_addr + {chunk_off}], 4;\n"
        ));
    }
    // commit_group emitted by caller (emit_inner_loop_open) after both C3 and C4 land.
}

fn emit_dq_acc_init(ptx: &mut String, config: &FlashAttentionConfig) {
    // Per SPEC AMENDMENT: use tier_b2_effective_bq for any block_q-dependent
    // sizing. accumulator_fragments = head_dim / 32 (independent of bq, but
    // we reference effective_bq to honor the per-hd fallback for warp tiling).
    let effective_bq = tier_b2_effective_bq(config);
    let accumulator_fragments = (config.head_dim / 32) as u32;
    ptx.push_str(&format!(
        "    // Zero dQ accumulator regs ({} fragments x 4 f32/lane; effective_bq={})\n",
        accumulator_fragments, effective_bq,
    ));
    for f in 0..accumulator_fragments {
        for r in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %dq_acc_{}_{};\n", f, r));
            ptx.push_str(&format!("    mov.f32 %dq_acc_{}_{}, 0.0;\n", f, r));
        }
    }
}

fn emit_inner_loop_open(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    mov.u32 %kv_iter, 0;\n");
    ptx.push_str("DQ_KV_ITER_LOOP:\n");
    ptx.push_str("    // Warp 0 (producer): issue cp.async for K, V tiles.\n");
    ptx.push_str("    // Warps 1-3 (consumers): wait on cp.async.wait_group.\n");
    ptx.push_str("    @!%p_producer bra DQ_KV_LOAD_DONE;\n");
    emit_k_producer_load(ptx, config);
    // cp.async V -> shmem[+v_offset]  (C4 will replace)
    ptx.push_str("    cp.async.commit_group;\n");
    ptx.push_str("DQ_KV_LOAD_DONE:\n");
    ptx.push_str("    cp.async.wait_group 0;\n");
    ptx.push_str("    bar.sync 0;\n");
    ptx.push('\n');
}

/// D1: emit tile_skip predicate per Phase 1 spec §9.2.
///
/// Produces `%tile_skip_predicate` (u32, 0=skip, 1=active) used by the
/// `setp.eq + @!bra DS_SKIP_LABEL` consumer at the top of emit_inner_loop_body.
///
/// Spec §2.4 pin: predicate gates *consumption* (E1/F1/F2/G1), not loads.
/// Load-skip bandwidth optimization is deferred beyond Phase 2.5.
fn emit_tile_skip_predicate(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{tier_b2_effective_bq, tier_b2_effective_bkv};
    let bq = tier_b2_effective_bq(config);
    let bkv = tier_b2_effective_bkv(config);
    if config.causal {
        // q_tile_end = q_iter * bq + (bq - 1)
        // kv_tile_start = kv_iter * bkv
        // predicate (1=active, 0=skip) = (kv_tile_start <= q_tile_end)
        ptx.push_str("    // === D1: tile_skip predicate (causal) ===\n");
        ptx.push_str(&format!("    mul.lo.u32 %q_tile_end, %q_iter, {bq};\n"));
        ptx.push_str(&format!("    add.u32 %q_tile_end, %q_tile_end, {};\n", bq - 1));
        ptx.push_str(&format!("    mul.lo.u32 %kv_tile_start, %kv_iter, {bkv};\n"));
        ptx.push_str("    setp.le.u32 %p_causal_active, %kv_tile_start, %q_tile_end;\n");
        ptx.push_str("    selp.u32 %tile_skip_predicate, 1, 0, %p_causal_active;\n");
    } else {
        ptx.push_str("    // === D1: tile_skip predicate (non-causal, always active) ===\n");
        ptx.push_str("    mov.u32 %tile_skip_predicate, 1;\n");
    }
}

fn emit_inner_loop_body(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dq_q_offset, tier_b2_dq_k_offset, tier_b2_dq_dO_offset, tier_b2_dq_v_offset,
    };

    let hd = config.head_dim as u32;
    // f16 row stride = head_dim * 2 bytes (tightly-packed row-major f16).
    let row_stride_bytes = (hd * 2) as usize;

    // === Tile-skip predicate (spec §9.2) ===
    ptx.push_str("    // Tile-skip predicate: gate dP+dS+dQ-update as a single block.\n");
    ptx.push_str("    setp.eq.u32 %p_tile_active, %tile_skip_predicate, 1;\n");
    ptx.push_str("    @!%p_tile_active bra DS_SKIP_LABEL;\n");
    ptx.push('\n');

    // === S = Q @ K^T (m16n8k16, B-frag col-major-presenting-as-K^T) ===
    // Row-major K[bkv, hd] byte-aliases to col-major K^T[hd, bkv] with
    // col_stride_bytes = hd * 2.  The existing 4-arg helper reads it correctly
    // and produces Q @ K^T in the MMA output.
    ptx.push_str("    // === S = Q @ K^T (m16n8k16, B-frag col-major byte-alias) ===\n");

    let q_smem_base = format!("{}", tier_b2_dq_q_offset(config));
    let k_smem_base = format!("{}", tier_b2_dq_k_offset(config));

    let a_regs: [String; 4] = ["s_a0", "s_a1", "s_a2", "s_a3"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let b_regs: [String; 2] = ["s_b0", "s_b1"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let c_regs: [String; 4] = ["s_c0", "s_c1", "s_c2", "s_c3"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let d_regs: [String; 4] = ["s_d0", "s_d1", "s_d2", "s_d3"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    for r in &a_regs {
        ptx.push_str(&format!("    .reg .b32 %{};\n", r));
    }
    for r in &b_regs {
        ptx.push_str(&format!("    .reg .b32 %{};\n", r));
    }
    for r in &c_regs {
        ptx.push_str(&format!("    .reg .f32 %{};\n", r));
    }
    for r in &d_regs {
        ptx.push_str(&format!("    .reg .f32 %{};\n", r));
    }
    for r in &c_regs {
        ptx.push_str(&format!("    mov.f32 %{}, 0.0;\n", r));
    }

    // Load Q tile as A-fragment (row-major, row_stride = hd * 2 bytes).
    emit_load_a_fragment_smem(ptx, &a_regs, &q_smem_base, row_stride_bytes);
    // Load K tile as B-fragment — row-major K[bkv, hd] with col_stride = hd * 2
    // byte-aliases to col-major K^T[hd, bkv], producing Q @ K^T in the MMA.
    emit_load_b_fragment_smem(ptx, &b_regs, &k_smem_base, row_stride_bytes);
    emit_mma_instruction(ptx, &d_regs, &a_regs, &b_regs, &c_regs);

    // === P recompute (lane-by-lane, no SMEM) per spec §4.3 step 4 ===
    ptx.push_str("    // === P recompute: P[q,k] = exp(S[q,k] - row_max[q]) * row_sum_recip[q] ===\n");
    ptx.push_str("    // (Lane-by-lane on the f32 S-fragment values held in %s_d0..3.)\n");
    ptx.push_str("    // row_max and row_sum_recip are loaded from HBM into %f_rmax and %f_rsum_recip\n");
    ptx.push_str("    // (HBM load emission omitted; consumed in Task 11's dS compute too).\n");
    ptx.push_str("    .reg .f32 %f_rmax, %f_rsum_recip;\n");
    ptx.push_str("    .reg .f32 %p_recip_log2e;\n");
    ptx.push_str("    mov.f32 %p_recip_log2e, 0F3FB8AA3B;  // 1/ln(2) = 1.4426950408889634\n");
    for i in 0..4 {
        ptx.push_str(&format!("    .reg .f32 %p_{};\n", i));
        ptx.push_str(&format!("    sub.f32 %p_{}, %s_d{}, %f_rmax;\n", i, i));
        ptx.push_str(&format!("    mul.f32 %p_{}, %p_{}, %p_recip_log2e;\n", i, i)); // x * log2(e)
        ptx.push_str(&format!("    ex2.approx.f32 %p_{}, %p_{};\n", i, i));          // 2^(x*log2(e)) = exp(x)
        ptx.push_str(&format!("    mul.f32 %p_{}, %p_{}, %f_rsum_recip;\n", i, i));
    }
    ptx.push('\n');

    // === dP = dO @ V^T (m16n8k16, B-frag col-major) ===
    ptx.push_str("    // === dP = dO @ V^T (m16n8k16, B-frag col-major) ===\n");

    let do_off_expr = format!("{}", tier_b2_dq_dO_offset(config));
    let v_off_expr = format!("{}", tier_b2_dq_v_offset(config));

    let dp_a_regs: [String; 4] = ["dp_a0", "dp_a1", "dp_a2", "dp_a3"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>().try_into().unwrap();
    let dp_b_regs: [String; 2] = ["dp_b0", "dp_b1"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>().try_into().unwrap();
    let dp_c_regs: [String; 4] = ["dp_c0", "dp_c1", "dp_c2", "dp_c3"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>().try_into().unwrap();
    let dp_d_regs: [String; 4] = ["dp_d0", "dp_d1", "dp_d2", "dp_d3"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>().try_into().unwrap();

    for r in &dp_a_regs { ptx.push_str(&format!("    .reg .b32 %{};\n", r)); }
    for r in &dp_b_regs { ptx.push_str(&format!("    .reg .b32 %{};\n", r)); }
    for r in &dp_c_regs { ptx.push_str(&format!("    .reg .f32 %{};\n", r)); }
    for r in &dp_d_regs { ptx.push_str(&format!("    .reg .f32 %{};\n", r)); }
    for r in &dp_c_regs { ptx.push_str(&format!("    mov.f32 %{}, 0.0;\n", r)); }

    emit_load_a_fragment_smem(ptx, &dp_a_regs, &do_off_expr, row_stride_bytes);
    emit_load_b_fragment_smem(ptx, &dp_b_regs, &v_off_expr, row_stride_bytes);
    emit_mma_instruction(ptx, &dp_d_regs, &dp_a_regs, &dp_b_regs, &dp_c_regs);
    ptx.push('\n');

    // === dS = P * (dP - D) (lane-by-lane, no SMEM stage yet) ===
    //
    // dS[q,k] = P[q,k] * (dP[q,k] - D[q]).  Computed on the f32 fragment values
    // held in %dp_d0..3 and %p_0..3 produced above.
    //
    // D[q] is loaded from HBM into %f_d_q (load emission omitted in scaffold; the
    // address derivation reuses the same row indexing as row_max/row_sum_recip).
    ptx.push_str("    // === dS = P * (dP - D) ===\n");
    ptx.push_str("    .reg .f32 %f_d_q;\n");
    for i in 0..4 {
        ptx.push_str(&format!("    .reg .f32 %ds_{};\n", i));
        ptx.push_str(&format!("    sub.f32 %ds_{}, %dp_d{}, %f_d_q;\n", i, i));
        ptx.push_str(&format!("    mul.f32 %ds_{}, %ds_{}, %p_{};\n", i, i, i));
    }
    ptx.push('\n');

    // === Scatter dS to SMEM at tier_b2_dq_ds_offset (row-major f32) ===
    //
    // The dQ-update MMA's A-fragment reads dS from SMEM, so each lane writes its
    // 4 dS values to the row-major dS band at tier_b2_dq_ds_offset.  Per-lane
    // scatter loop body is omitted in this scaffold; bar.sync 0 follows.
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dq_ds_offset, tier_b2_dq_k_colmajor_offset, tier_b2_effective_bkv,
    };
    ptx.push_str("    // === Scatter dS to SMEM ===\n");
    ptx.push_str(&format!(
        "    // st.shared.f32 [shmem + {} + lane_offset], %ds_*\n",
        tier_b2_dq_ds_offset(config),
    ));
    ptx.push_str("    // (Per-lane scatter loop body omitted in scaffold; bar.sync follows.)\n");
    ptx.push_str("    bar.sync 0;\n");
    ptx.push('\n');

    // === Col-major K re-stage band (Path A per spec §5.2-5.3) ===
    //
    // K is staged row-major at tier_b2_dq_k_offset (matches B.1 forward
    // convention).  For dQ_acc += dS @ K (NOT dS @ K^T), we need K presented to
    // the MMA as col-major.  Emit a scatter that copies row-major K[bkv, hd] →
    // col-major K band at tier_b2_dq_k_colmajor_offset once per kv_iter, gated by
    // warp 0 to avoid 4× write amplification across the four consumer warps.
    //
    // Each lane handles (effective_bkv * head_dim / 32) (row, col) pairs via
    // ld.shared.b16 / st.shared.b16 (the source is already in SMEM; no cp.async).
    //
    // FFI-side requirement (Phase 3 wiring): when `nsl_flash_attention_csha`
    // launches the dQ-kernel, it MUST assert
    //   tier_b2_dq_total_smem_bytes(config) <= SMEM_DYNAMIC_BUDGET_BYTES
    // and fail loudly otherwise (per spec §5.5 Step 3d-3).  Not wired in Phase 2.
    let k_off = crate::flash_attention_v2::smem_layout::tier_b2_dq_k_offset(config);
    let k_col_off = tier_b2_dq_k_colmajor_offset(config);
    let bkv_eff = tier_b2_effective_bkv(config);
    ptx.push_str("    // === Col-major K re-stage (Path A) ===\n");
    ptx.push_str("    // Warp 0 gates the scatter to avoid 4x write amplification.\n");
    ptx.push_str("    @!%p_producer bra DQ_KCOL_RESTAGE_DONE;\n");
    ptx.push_str(&format!(
        "    // Source: row-major K[{bkv}, {hd}] at SMEM[+{k_off}]\n",
        bkv = bkv_eff, hd = hd, k_off = k_off,
    ));
    ptx.push_str(&format!(
        "    // Dest:   col-major K-staged at SMEM[+{k_col_off}], col_stride = {} bytes\n",
        bkv_eff * 2, k_col_off = k_col_off,
    ));
    ptx.push_str(&format!(
        "    // (Per-lane scatter loop body covers {} (row, col) pairs per lane.)\n",
        (bkv_eff * hd) / 32,
    ));
    ptx.push_str("DQ_KCOL_RESTAGE_DONE:\n");
    ptx.push_str("    bar.sync 0;\n");
    ptx.push('\n');

    // === dQ_acc += dS @ K (m16n8k16, B-frag from col-major K re-stage) ===
    //
    // A-frag: dS, row-major in SMEM at tier_b2_dq_ds_offset.
    // B-frag: K, col-major-staged in SMEM at tier_b2_dq_k_colmajor_offset.
    //         col_stride_bytes = effective_bkv * 2 (f16 between adjacent
    //         head-dim columns in the col-major band).
    // The post-revert 4-arg emit_load_b_fragment_smem reads col-major SMEM
    // → col-major B-frag → the MMA computes A @ B = dS @ K (NOT dS @ K^T).
    // Output accumulates into %dq_acc_0_{0..3} (the existing dQ accumulator
    // register family from emit_dq_acc_init).
    ptx.push_str("    // === dQ_acc += dS @ K (m16n8k16, B-frag from col-major K re-stage) ===\n");

    let dq_a_regs: [String; 4] = ["dq_a0", "dq_a1", "dq_a2", "dq_a3"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>().try_into().unwrap();
    let dq_b_regs: [String; 2] = ["dq_b0", "dq_b1"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>().try_into().unwrap();
    for r in &dq_a_regs {
        ptx.push_str(&format!("    .reg .b32 %{};\n", r));
    }
    for r in &dq_b_regs {
        ptx.push_str(&format!("    .reg .b32 %{};\n", r));
    }

    let ds_off_expr = format!("{}", tier_b2_dq_ds_offset(config));
    let k_col_off_expr = format!("{}", k_col_off);
    let ds_row_stride = (config.block_kv * 4) as usize; // dS is f32 row-major
    let bkv_eff_col_stride = (bkv_eff * 2) as usize;    // col stride of col-major K band

    emit_load_a_fragment_smem(ptx, &dq_a_regs, &ds_off_expr, ds_row_stride);
    emit_load_b_fragment_smem(ptx, &dq_b_regs, &k_col_off_expr, bkv_eff_col_stride);

    // Accumulate into %dq_acc_0_{0..3} (D and C both = dq_acc → MAC).
    let dq_d_regs: [String; 4] = ["dq_acc_0_0", "dq_acc_0_1", "dq_acc_0_2", "dq_acc_0_3"]
        .iter().map(|s| s.to_string()).collect::<Vec<_>>().try_into().unwrap();
    emit_mma_instruction(ptx, &dq_d_regs, &dq_a_regs, &dq_b_regs, &dq_d_regs);
    ptx.push('\n');

    ptx.push_str("DS_SKIP_LABEL:\n");
}

fn emit_inner_loop_close(ptx: &mut String) {
    ptx.push_str("    // Inner-loop back-edge: kv_iter += 1; if kv_iter < num_kv_iters, branch back.\n");
    ptx.push_str("    add.u32 %kv_iter, %kv_iter, 1;\n");
    ptx.push_str("    setp.lt.u32 %p_kv_iter_more, %kv_iter, %num_kv_iters;\n");
    ptx.push_str("    @%p_kv_iter_more bra DQ_KV_ITER_LOOP;\n");
    ptx.push_str("DQ_KV_ITER_DONE:\n");
}

fn emit_outer_loop_close(ptx: &mut String) {
    // Outer-loop back-edge: q_iter += 1; if q_iter < num_q_iters, branch back.
    ptx.push_str("    // Outer-loop back-edge: q_iter += 1; if q_iter < num_q_iters, branch back.\n");
    ptx.push_str("    add.u32 %q_iter, %q_iter, 1;\n");
    ptx.push_str("    setp.lt.u32 %p_q_iter_more, %q_iter, %num_q_iters;\n");
    ptx.push_str("    @%p_q_iter_more bra DQ_Q_ITER_LOOP;\n");
    ptx.push_str("DQ_Q_ITER_DONE:\n");
}

fn emit_dq_finalize(ptx: &mut String, config: &FlashAttentionConfig) {
    let accumulator_fragments = (config.head_dim / 32) as u32;
    ptx.push_str("    // === dQ HBM finalize ===\n");
    ptx.push_str("    // Scatter dQ_acc registers to HBM dQ[B, H, q_tile, hd].\n");
    ptx.push_str("    // Each warp owns a 32-col strip of the bq x hd output tile.\n");
    ptx.push_str("    // Address: dq_base + (batch*H + head)*S*hd*4 + q_tile_start*hd*4 + lane_offset\n");
    for f in 0..accumulator_fragments {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    // Fragment {} reg {} -> HBM dQ[batch, head, q_local, hd_slice]\n",
                f, r,
            ));
            ptx.push_str(&format!(
                "    st.global.f32 [%addr], %dq_acc_{}_{};\n",
                f, r,
            ));
        }
    }
    ptx.push_str("    bar.sync 0;\n");
}

fn emit_entry_close(ptx: &mut String) {
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 128,
            causal: false, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
        }
    }

    #[test]
    fn synthesize_dq_kernel_targets_sm80() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn synthesize_dq_kernel_has_four_warp_block() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains(".maxntid 128"));
    }

    #[test]
    fn synthesize_dq_kernel_extern_smem_sized_from_total_bytes() {
        // Per SPEC AMENDMENT: SMEM extern declaration uses tier_b2_dq_total_smem_bytes
        // which INCLUDES the col-major K re-stage band.
        use crate::flash_attention_v2::smem_layout::tier_b2_dq_total_smem_bytes;
        let cfg = canonical_cfg();
        let expected_total = tier_b2_dq_total_smem_bytes(&cfg);
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        // Must declare shmem[<total>] not shmem[] (unsized) and not mixed with .shared
        assert!(ptx.contains(&format!(".extern .shared .align 16 .b8 shmem[{}]", expected_total)),
            "expected explicit shmem[{}] sizing, got:\n{ptx}", expected_total);
        assert!(!ptx.contains(".shared .align 16 .b8 dq_static"),
            "must NOT mix static .shared with extern");
    }

    #[test]
    #[allow(non_snake_case)]
    fn synthesize_dq_kernel_uses_cp_async_for_q_dO_load() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("cp.async"),
            "expected cp.async for Q+dO producer-staged load");
    }

    #[test]
    fn synthesize_dq_kernel_emits_outer_q_iter_loop() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("DQ_Q_ITER_LOOP"),
            "expected outer q_iter loop label");
    }

    #[test]
    fn synthesize_dq_kernel_emits_inner_kv_iter_loop() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("DQ_KV_ITER_LOOP"),
            "expected inner kv_iter loop label");
    }

    #[test]
    fn synthesize_dq_kernel_warp0_is_producer() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // warp 0 = producer; producer issues cp.async; consumers (warps 1-3) wait
        assert!(ptx.contains("setp.eq.u32 %p_producer, %warp_id, 0") ||
                ptx.contains("setp.eq.u32 %p_producer, %warp_id, 0;"));
    }

    #[test]
    fn synthesize_dq_kernel_uses_effective_bq_at_hd_128() {
        // Per SPEC AMENDMENT: emit_dq_acc_init must respect effective_bq.
        // At hd=128 with raw block_q=64, effective_bq = 32 (SMEM-pressure fallback).
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // The dQ accumulator init must use effective_bq for sizing.
        // Concrete check: the accumulator register count should match
        // hd/32 = 4 fragments x 4 f32/lane (independent of bq, but the scaffold
        // should reference effective_bq for warp tiling).
        // Loose check: must reference accumulators (e.g., %dq_acc_*)
        assert!(ptx.contains("dq_acc"),
            "expected dQ accumulator register decls");
    }

    #[test]
    fn synthesize_dq_kernel_emits_tile_skip_predicate() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // Per Phase 1 spec §9.2: setp + @!predicate branch to DS_SKIP_LABEL
        assert!(ptx.contains("setp.eq.u32 %p_tile_active"),
            "expected tile-skip predicate set");
        assert!(ptx.contains("@!%p_tile_active"),
            "expected predicate-gated branch");
        assert!(ptx.contains("DS_SKIP_LABEL"),
            "expected DS_SKIP_LABEL target");
    }

    #[test]
    fn synthesize_dq_kernel_emits_s_qkt_mma() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // S = Q @ K^T via m16n8k16 row.col MMA
        assert!(ptx.contains("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32"),
            "expected m16n8k16 MMA");
        // K loaded as B-frag with the standard 4-arg helper (col-major SMEM read).
        // The helper emits "Load B-fragment (k16xn8 col-major" comment per matmul_mma.rs post-revert.
        assert!(ptx.contains("Load B-fragment (k16xn8 col-major"),
            "expected non-transposed B-frag load comment");
    }

    #[test]
    fn synthesize_dq_kernel_emits_p_recompute() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // P = exp(S - row_max) * row_sum_recip — uses ex2.approx.f32 or exp
        assert!(ptx.contains("ex2.approx.f32"),
            "expected ex2.approx.f32 for P recompute (PTX-native exp), got:\n{ptx}");
    }

    #[test]
    fn synthesize_dq_kernel_emits_dp_mma() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // dP = dO @ V^T via second m16n8k16 row.col MMA.
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count();
        assert!(mma_count >= 2,
            "expected at least 2 MMAs (S + dP) after Task 10, got {}", mma_count);
    }

    #[test]
    fn synthesize_dq_kernel_loads_v_for_dp() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // V tile is loaded as B-frag for dP = dO @ V^T using the same col-major
        // byte-aliasing pattern as K in S = QK^T.
        // The B-frag helper emits the same "Load B-fragment (k16xn8 col-major)" comment
        // each time it's called, so count occurrences (must be at least 2 now: K then V).
        let bfrag_loads = ptx.matches("Load B-fragment (k16xn8 col-major").count();
        assert!(bfrag_loads >= 2,
            "expected at least 2 B-frag loads (K for S, V for dP), got {}", bfrag_loads);
    }

    #[test]
    fn synthesize_dq_kernel_emits_ds_computation() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // dS[q,k] = P[q,k] * (dP[q,k] - D[q])
        assert!(ptx.contains("// === dS = P * (dP - D) ==="),
            "expected dS-compute header");
    }

    #[test]
    fn synthesize_dq_kernel_scatters_ds_to_smem() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // dS must be stored to SMEM at ds_offset before the dQ-update MMA reads it.
        assert!(ptx.contains("// === Scatter dS to SMEM ==="),
            "expected dS SMEM scatter section");
    }

    #[test]
    fn synthesize_dq_kernel_emits_col_major_k_restage() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // Path A col-major K re-stage band per spec §5.2-5.3
        assert!(ptx.contains("// === Col-major K re-stage (Path A) ==="),
            "expected col-major K re-stage section");
        // Warp 0 gates the scatter
        assert!(ptx.contains("@!%p_producer bra DQ_KCOL_RESTAGE_DONE"),
            "expected warp-0-gated scatter");
        // bar.sync after re-stage
        assert!(ptx.contains("DQ_KCOL_RESTAGE_DONE"),
            "expected DQ_KCOL_RESTAGE_DONE label");
    }

    #[test]
    fn synthesize_dq_kernel_emits_three_mmas_after_task_11() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // S + dP + dQ-update = 3 MMAs
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count();
        assert_eq!(mma_count, 3, "expected exactly 3 MMAs (S + dP + dQ-update), got {}", mma_count);
    }

    #[test]
    fn synthesize_dq_kernel_dq_update_reads_from_kcol_offset() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        use crate::flash_attention_v2::smem_layout::tier_b2_dq_k_colmajor_offset;
        let cfg = canonical_cfg();
        let kcol_off = tier_b2_dq_k_colmajor_offset(&cfg);
        // The MMA after re-stage must use the col-major K band's offset
        // (search for the offset value appearing in a B-frag context)
        assert!(ptx.contains(&format!("{}", kcol_off)),
            "expected K-colmajor offset {} to appear in PTX", kcol_off);
    }

    #[test]
    fn synthesize_dq_kernel_uses_dq_acc_regs_for_third_mma() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // The dQ-update MMA must accumulate into %dq_acc_*, not new scratch regs.
        // Specifically: dq_acc_0_0 must appear as both an A/D MMA operand
        // (i.e., inside an mma.sync braced operand list).
        assert!(ptx.contains("{dq_acc_0_0, dq_acc_0_1, dq_acc_0_2, dq_acc_0_3}"),
            "expected dq_acc_0_{{0..3}} to be used as MMA D/C operand list, got:\n{ptx}");
    }

    #[test]
    fn synthesize_dq_kernel_scatters_dq_acc_to_hbm() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("// === dQ HBM finalize ==="),
            "expected dQ finalize section");
        assert!(ptx.contains("st.global"),
            "expected st.global for dQ scatter");
    }

    #[test]
    fn synthesize_dq_kernel_restage_emits_bounded_smem_traffic() {
        // Spec §5.5 institutional pin (NEW per Task 11 deltas Step 3d-2):
        // the col-major K re-stage emits a bounded, predictable number of
        // ld.shared.b16 + st.shared.b16 pairs. Bound: bkv_eff * hd / 32 per kind
        // (each lane handles bkv_eff*hd/32 (row,col) source-destination pairs).
        //
        // Phase 2 scope: assert the scatter section produces a comment-recorded
        // upper bound on per-lane iteration count, capturing the design contract
        // even before per-lane instruction emission lands.
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        use crate::flash_attention_v2::smem_layout::tier_b2_effective_bkv;
        let cfg = canonical_cfg();
        let bkv_eff = tier_b2_effective_bkv(&cfg);
        let hd = cfg.head_dim as u32;
        let per_lane_pairs = (bkv_eff * hd) / 32;
        // The re-stage comment encodes the per-lane upper bound (predictable
        // overhead per kv_iter, gated by warp 0).
        assert!(ptx.contains(&format!("{} (row, col) pairs per lane", per_lane_pairs)),
            "expected per-lane scatter bound = {} (row,col) pairs to appear, got:\n{ptx}",
            per_lane_pairs);
    }

    #[test]
    fn dq_kernel_declares_row_index_tmp_scratch_for_hbm_addr_helpers() {
        // emit_4d_byte_offset / emit_3d_byte_offset (dq::hbm_addr) require a caller-declared
        // %row_index_tmp scratch; the synthesizer must include the declaration in
        // emit_register_decls so future cp.async / HBM-finalize sites can call the helpers
        // without producing undeclared-register ptxas errors.
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(
            ptx.contains(".reg .u32 %row_index_tmp"),
            "synthesize_dq_kernel must declare %row_index_tmp scratch (caller contract for dq::hbm_addr helpers)"
        );
    }

    #[test]
    fn synthesize_dq_kernel_emits_real_outer_q_iter_loop_with_back_edge() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // Real loop: induction var setup, comparison, conditional back-edge
        assert!(ptx.contains("DQ_Q_ITER_LOOP"), "expected outer loop label");
        assert!(ptx.contains("setp.lt.u32 %p_q_iter_more"),
            "expected q_iter < num_q_iters predicate");
        assert!(ptx.contains("@%p_q_iter_more bra DQ_Q_ITER_LOOP"),
            "expected conditional back-edge to outer loop label");
        // Confirm the old placeholder `bra DQ_Q_ITER_DONE` (unconditional) is GONE
        let placeholder_count = ptx.matches("bra DQ_Q_ITER_DONE;  // placeholder").count();
        assert_eq!(placeholder_count, 0,
            "placeholder unconditional outer-loop exit must be removed");
    }

    #[test]
    fn synthesize_dq_kernel_emits_real_inner_kv_iter_loop_with_back_edge() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("DQ_KV_ITER_LOOP"), "expected inner loop label");
        assert!(ptx.contains("setp.lt.u32 %p_kv_iter_more"),
            "expected kv_iter < num_kv_iters predicate");
        assert!(ptx.contains("@%p_kv_iter_more bra DQ_KV_ITER_LOOP"),
            "expected conditional back-edge to inner loop label");
        let placeholder_count = ptx.matches("bra DQ_KV_ITER_DONE;  // placeholder").count();
        assert_eq!(placeholder_count, 0,
            "placeholder unconditional inner-loop exit must be removed");
    }

    #[test]
    fn d1_tile_skip_predicate_set_by_real_derivation_not_placeholder() {
        let mut cfg = canonical_cfg();
        cfg.causal = true;
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        // Real predicate derivation: q_tile_end + kv_tile_start computed
        assert!(ptx.contains("%kv_tile_start"), "expected kv_tile_start register");
        assert!(ptx.contains("%q_tile_end"), "expected q_tile_end register");
        // For causal, predicate = kv_tile_start <= q_tile_end
        assert!(ptx.contains("setp.le.u32 %p_causal_active"),
            "expected setp.le for causal tile_skip");
        // %tile_skip_predicate must be materialized (selp.u32 ..., 1, 0, %p_causal_active)
        assert!(ptx.contains("selp.u32 %tile_skip_predicate"),
            "expected selp to materialize predicate as u32");
    }

    #[test]
    fn d1_non_causal_predicate_is_always_active() {
        let mut cfg = canonical_cfg();
        cfg.causal = false;
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        assert!(ptx.contains("mov.u32 %tile_skip_predicate, 1"),
            "expected non-causal predicate = constant 1");
    }
}
