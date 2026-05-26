//! dK/dV-kernel emitter for Tier B.2 backward.
//!
//! kv-OUTER, q-INNER. dK and dV accumulators are register-resident across the
//! q sweep (one kv tile fixed, sweep over all q tiles). No atomicAdd.
//!
//! Loop order (contrast with dq.rs which is q-outer/kv-inner):
//!   outer = kv_iter  -- loads K+V once, holds resident across q sweep
//!   inner = q_iter   -- loads Q+dO per iteration; also loads per-row stats
//!
//! Accumulators dV_acc and dK_acc are zeroed ONCE at kv-outer loop open
//! (before the q-inner loop). Finalize (HBM scatter) runs AFTER the q-inner
//! loop closes but STILL INSIDE the kv-outer loop (before its back-edge),
//! so each kv tile's dK/dV is written with the correct in-range kv row.
//!
//! This module is a SCAFFOLD (Phase 3a Task 3). The finalize body is a stub;
//! Tasks 4-8 add S/dP/dS/dV/dK MMA math.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md ss4 + ss5.2

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    tier_b2_dkdv_total_smem_bytes, tier_b2_effective_bkv, tier_b2_effective_bq,
};
use crate::flash_attention_v2::tier_b2::backward::BackwardSynthError;

pub fn synthesize_dkdv_kernel(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    let hd = config.head_dim as u32;
    if !hd.is_multiple_of(32) {
        return Err(BackwardSynthError::UnsupportedHeadDim(hd));
    }

    let mut ptx = String::new();
    emit_prelude(&mut ptx);
    // .extern .shared is a module-level PTX directive: must precede .visible .entry.
    emit_smem_extern_module_scope(&mut ptx, config);
    emit_entry_signature(&mut ptx);
    emit_register_decls(&mut ptx, config);
    emit_grid_id_setup(&mut ptx);
    emit_warp_band_setup(&mut ptx, config);
    emit_kv_iter_count_setup(&mut ptx, config);
    emit_q_iter_count_setup(&mut ptx, config);
    // kv-outer loop open: load K+V (resident across q sweep), zero dK/dV accumulators.
    emit_outer_loop_open(&mut ptx, config);
    // q-inner loop open: load Q+dO per iteration + stats.
    emit_inner_loop_open(&mut ptx, config);
    emit_tile_skip_predicate(&mut ptx, config);
    emit_inner_loop_body(&mut ptx, config);
    emit_inner_loop_close(&mut ptx);
    // Finalize MUST run INSIDE the kv-outer loop: dK/dV accumulators are per-kv-tile
    // (zeroed at kv-outer open, accumulated over the q sweep). Finalize's HBM row =
    // kv_iter*bkv + band_row_base + lane/4. Placing it after the kv-outer close would
    // use the post-loop %kv_iter (= num_kv_iters), writing OOB rows. Run it per kv_iter
    // before the increment so each kv-tile's dK/dV is written with the correct in-range row.
    emit_dkdv_finalize_stub(&mut ptx);
    emit_outer_loop_close(&mut ptx);
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
fn emit_smem_extern_module_scope(ptx: &mut String, config: &FlashAttentionConfig) {
    let total_smem = tier_b2_dkdv_total_smem_bytes(config);
    ptx.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
    let _ = total_smem; // Size is communicated via cuFuncSetAttribute at launch time, not baked into the extern declaration.
}

fn emit_entry_signature(ptx: &mut String) {
    ptx.push_str(".visible .entry tier_b2_dkdv_kernel(\n");
    ptx.push_str("    .param .u64 q_saved_ptr,\n");
    ptx.push_str("    .param .u64 k_saved_ptr,\n");
    ptx.push_str("    .param .u64 v_saved_ptr,\n");
    ptx.push_str("    .param .u64 d_o_ptr,\n");
    ptx.push_str("    .param .u64 row_max_ptr,\n");
    ptx.push_str("    .param .u64 row_sum_ptr,\n");
    ptx.push_str("    .param .u64 d_ptr,\n");
    ptx.push_str("    .param .u64 segment_ids_ptr,\n");
    ptx.push_str("    .param .u64 d_k_out_ptr,\n");
    ptx.push_str("    .param .u64 d_v_out_ptr,\n");
    ptx.push_str("    .param .u32 seq_len,\n");
    ptx.push_str("    .param .u32 heads,\n");
    ptx.push_str("    .param .u32 batch\n");
    ptx.push_str(")\n");
    ptx.push_str(".maxntid 128, 1, 1\n");
    ptx.push_str("{\n");
}

fn emit_register_decls(ptx: &mut String, config: &FlashAttentionConfig) {
    // NOTE: .extern .shared shmem[] is emitted at module scope by emit_smem_extern_module_scope.
    // NOTE: user register named %r_tid (not %tid) to avoid shadowing the PTX special register
    // %tid (a vector); ptxas rejects %tid.x when %tid is declared as a user u32.
    ptx.push_str("    .reg .u32 %r_tid, %lane_id, %warp_id;\n");
    ptx.push_str("    .reg .u32 %band_row_base;\n");
    ptx.push_str("    .reg .pred %p_warp_active;\n");
    ptx.push_str("    .reg .u32 %kv_tile, %q_tile, %head, %batch_idx;\n");
    ptx.push_str("    .reg .pred %p_tile_active, %p_producer, %p_consumer;\n");
    ptx.push_str("    .reg .u32 %addr_lo, %tile_skip_predicate;\n");
    ptx.push_str("    .reg .u32 %row_index_tmp;\n");
    ptx.push_str("    .reg .u64 %addr;\n");
    // seq_len scratch: declared here so all .reg directives precede executable instructions.
    ptx.push_str("    .reg .u32 %seq_len_r;\n");
    // heads_r: loaded from [heads] param; used by cp.async HBM address helpers.
    ptx.push_str("    .reg .u32 %heads_r;\n");
    // Outer kv_iter loop: induction variable, upper bound, loop predicate.
    ptx.push_str("    .reg .u32 %kv_iter, %num_kv_iters;\n");
    ptx.push_str("    .reg .pred %p_kv_iter_more;\n");
    // Inner q_iter loop: induction variable, upper bound, loop predicate.
    ptx.push_str("    .reg .u32 %q_iter, %num_q_iters;\n");
    ptx.push_str("    .reg .pred %p_q_iter_more;\n");
    // Scratch registers used by matmul_mma helpers.
    ptx.push_str("    .reg .u32 %mma_addr, %mma_a_row, %mma_b_row;\n");
    // Converted shared-memory base for MMA fragment addressing (loop-invariant).
    ptx.push_str("    .reg .u64 %mma_smem_base64;\n");
    ptx.push_str("    .reg .u32 %mma_smem_base;\n");
    // n-tile streaming loop registers (for future MMA tiling Tasks 4-8).
    ptx.push_str("    .reg .u32 %n_tile, %num_n_tiles, %a_base, %b_base;\n");
    ptx.push_str("    .reg .pred %p_ntile_more;\n");
    // D1 tile_skip predicate registers (causal; declared but unused in scaffold).
    ptx.push_str("    .reg .u32 %q_tile_end, %kv_tile_start;\n");
    ptx.push_str("    .reg .pred %p_causal_active;\n");
    // E1 S=Q@K^T + P-recompute MMA fragment registers (Task 4).
    // Per PTX ISA all .reg directives must precede executable instructions.
    // S = Q@K^T fragment family (A: k16 rows b32-packed; B: n8 cols; C/D: f32 acc).
    ptx.push_str("    .reg .b32 %s_a0, %s_a1, %s_a2, %s_a3, %s_b0, %s_b1;\n");
    ptx.push_str("    .reg .f32 %s_c0, %s_c1, %s_c2, %s_c3, %s_d0, %s_d1, %s_d2, %s_d3;\n");
    // P recompute scalars + per-lane P values.
    ptx.push_str("    .reg .f32 %p_recip_log2e, %f_scale, %p_0, %p_1, %p_2, %p_3;\n");
    // C1 cp.async K-tile registers (kv-outer resident; prefixed %c1_*).
    ptx.push_str("    .reg .u32 %c1_kv_tile_start, %c1_lane_byte_off, %c1_smem_base32, %c1_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c1_k_hbm_off, %c1_k_hbm_base, %c1_lane_hbm_addr, %c1_smem_base64;\n");
    // C2 cp.async V-tile registers (kv-outer resident; prefixed %c2_*).
    ptx.push_str("    .reg .u32 %c2_kv_tile_start, %c2_lane_byte_off, %c2_smem_base32, %c2_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c2_v_hbm_off, %c2_v_hbm_base, %c2_lane_hbm_addr, %c2_smem_base64;\n");
    // C3 cp.async Q-tile registers (q-inner per-iteration; prefixed %c3_*).
    ptx.push_str("    .reg .u32 %c3_q_tile_start, %c3_lane_byte_off, %c3_smem_base32, %c3_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c3_q_hbm_off, %c3_q_hbm_base, %c3_lane_hbm_addr, %c3_smem_base64;\n");
    // C4 cp.async dO-tile registers (q-inner per-iteration; prefixed %c4_*).
    // dO tile is [effective_bq, hd] f16, same shape as Q.
    ptx.push_str("    .reg .u32 %c4_q_tile_start, %c4_lane_byte_off, %c4_smem_base32, %c4_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c4_do_hbm_off, %c4_do_hbm_base, %c4_lane_hbm_addr, %c4_smem_base64;\n");
    // Stats-load registers (emit_stats_addr_load). Declared here per PTX ISA rule.
    ptx.push_str("    .reg .u64 %stats_rmax_base, %stats_rsum_base, %stats_d_base;\n");
    ptx.push_str("    .reg .u64 %stats_off_lo, %stats_off_hi, %stats_addr;\n");
    ptx.push_str("    .reg .u32 %s_lo, %s_hi, %stat_lane_div4;\n");
    ptx.push_str("    .reg .f32 %rmax_lo, %rmax_hi, %rsum_lo, %rsum_hi;\n");
    ptx.push_str("    .reg .f32 %rsum_recip_lo, %rsum_recip_hi, %d_lo, %d_hi;\n");
    // G1 dV HBM finalize registers (prefixed %g1_*).
    ptx.push_str("    .reg .u32 %g1_lane_mod4, %g1_lane_div4;\n");
    ptx.push_str("    .reg .u32 %g1_kv_tile_start, %g1_row_lo, %g1_row_hi;\n");
    ptx.push_str("    .reg .u32 %g1_col_lo, %g1_col_hi;\n");
    ptx.push_str("    .reg .u32 %g1_d_tmp;\n");
    ptx.push_str("    .reg .u64 %g1_dv_byte_off, %g1_dv_addr;\n");
    ptx.push_str("    .reg .u64 %g1_dv_base;\n");
    // G2 dK HBM finalize registers (prefixed %g2_*).
    ptx.push_str("    .reg .u32 %g2_lane_mod4, %g2_lane_div4;\n");
    ptx.push_str("    .reg .u32 %g2_kv_tile_start, %g2_row_lo, %g2_row_hi;\n");
    ptx.push_str("    .reg .u32 %g2_col_lo, %g2_col_hi;\n");
    ptx.push_str("    .reg .u32 %g2_d_tmp;\n");
    ptx.push_str("    .reg .u64 %g2_dk_byte_off, %g2_dk_addr;\n");
    ptx.push_str("    .reg .u64 %g2_dk_base;\n");
    // dP MMA fragment registers (Task 5). All .reg directives must precede executable instructions.
    ptx.push_str("    .reg .b32 %dp_a0, %dp_a1, %dp_a2, %dp_a3, %dp_b0, %dp_b1;\n");
    ptx.push_str("    .reg .f32 %dp_c0, %dp_c1, %dp_c2, %dp_c3, %dp_d0, %dp_d1, %dp_d2, %dp_d3;\n");
    // dS scalars (Task 5): dS_i = (1/sqrt(D)) * P_i * (dP_i - D[q]).
    ptx.push_str("    .reg .f32 %ds_0, %ds_1, %ds_2, %ds_3;\n");
    // dV accumulators: hd/8 contiguous n-tiles x 4 f32/lane (Task 5 fills MMA; zeroed here).
    let n_acc = (config.head_dim / 8) as u32;
    for n in 0..n_acc {
        ptx.push_str(&format!(
            "    .reg .f32 %dv_acc_{n}_0, %dv_acc_{n}_1, %dv_acc_{n}_2, %dv_acc_{n}_3;\n"
        ));
    }
    // dK accumulators: hd/8 contiguous n-tiles x 4 f32/lane (Task 6 fills MMA; zeroed here).
    for n in 0..n_acc {
        ptx.push_str(&format!(
            "    .reg .f32 %dk_acc_{n}_0, %dk_acc_{n}_1, %dk_acc_{n}_2, %dk_acc_{n}_3;\n"
        ));
    }
    ptx.push('\n');
}

fn emit_grid_id_setup(ptx: &mut String) {
    // Read thread ID X component from the special %tid vector into %r_tid.
    ptx.push_str("    mov.u32 %r_tid, %tid.x;\n");
    ptx.push_str("    and.b32 %lane_id, %r_tid, 31;\n");
    ptx.push_str("    shr.u32 %warp_id, %r_tid, 5;\n");
    // For dK/dV the grid is launched with ctaid.x = kv_tile, ctaid.y = head, ctaid.z = batch.
    ptx.push_str("    mov.u32 %kv_tile,   %ctaid.x;\n");
    ptx.push_str("    mov.u32 %head,      %ctaid.y;\n");
    ptx.push_str("    mov.u32 %batch_idx, %ctaid.z;\n");
    ptx.push_str("    setp.eq.u32 %p_producer, %warp_id, 0;\n");
    // Load heads param for use by HBM address helpers.
    ptx.push_str("    ld.param.u32 %heads_r, [heads];\n");
    // Converted SMEM base for the MMA fragment-load helpers (loop-invariant).
    ptx.push_str("    cvta.shared.u64 %mma_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %mma_smem_base, %mma_smem_base64;\n");
    ptx.push('\n');
}

/// Emit warp-band ownership base and the idle-warp predicate from %warp_id.
///
/// For dK/dV the band covers kv rows (not q rows). The bkv/16 active warps each
/// own a 16-row kv band: warp w owns kv rows [w*16, w*16+16).
fn emit_warp_band_setup(ptx: &mut String, config: &FlashAttentionConfig) {
    let active_warps = tier_b2_effective_bkv(config) / 16; // 4 at bkv=64, 2 at bkv=32
    ptx.push_str("    // Warp-per-m16-band: warp w owns kv-rows [w*16, w*16+16).\n");
    ptx.push_str("    mul.lo.u32 %band_row_base, %warp_id, 16;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p_warp_active, %warp_id, {active_warps};  // active = warp_id < bkv/16\n"
    ));
    ptx.push('\n');
}

/// Compute %num_kv_iters = ceil(seq_len / bkv) using power-of-2 shift.
///
/// bkv is power-of-2 per spec ss3.1, so:
///   num_kv_iters = (seq_len + bkv - 1) >> log2(bkv)
/// %seq_len_r is declared in emit_register_decls; loaded here.
fn emit_kv_iter_count_setup(ptx: &mut String, config: &FlashAttentionConfig) {
    let bkv = tier_b2_effective_bkv(config);
    let log2_bkv = bkv.trailing_zeros();
    ptx.push_str("    ld.param.u32 %seq_len_r, [seq_len];\n");
    ptx.push_str(&format!("    add.u32 %num_kv_iters, %seq_len_r, {};\n", bkv - 1));
    ptx.push_str(&format!("    shr.u32 %num_kv_iters, %num_kv_iters, {};\n", log2_bkv));
    ptx.push('\n');
}

/// Compute %num_q_iters = ceil(seq_len / bq) using power-of-2 shift.
///
/// bq is power-of-2 per spec ss3.1. %seq_len_r is already loaded by
/// emit_kv_iter_count_setup, so this helper reuses it directly.
fn emit_q_iter_count_setup(ptx: &mut String, config: &FlashAttentionConfig) {
    let bq = tier_b2_effective_bq(config);
    ptx.push_str(&format!("    add.u32 %num_q_iters, %seq_len_r, {};\n", bq - 1));
    ptx.push_str(&format!("    shr.u32 %num_q_iters, %num_q_iters, {};\n", bq.trailing_zeros()));
    ptx.push('\n');
}

/// kv-OUTER loop open: initialize kv_iter, load K+V (resident across q sweep),
/// and zero the dV/dK accumulator registers once before the q-inner loop.
fn emit_outer_loop_open(ptx: &mut String, config: &FlashAttentionConfig) {
    // Initialize induction variable. %num_kv_iters computed in emit_kv_iter_count_setup.
    ptx.push_str("    mov.u32 %kv_iter, 0;\n");
    ptx.push_str("DKDV_KV_ITER_LOOP:\n");
    // Warp 0 (producer): issue cp.async for K+V tiles (resident across q sweep).
    // Warps 1-3 (consumers): wait on cp.async.wait_group.
    ptx.push_str("    @!%p_producer bra DKDV_KV_LOAD_DONE;\n");
    emit_k_producer_load(ptx, config);
    emit_v_producer_load(ptx, config);
    ptx.push_str("    cp.async.commit_group;\n");
    ptx.push_str("DKDV_KV_LOAD_DONE:\n");
    ptx.push_str("    cp.async.wait_group 0;\n");
    ptx.push_str("    bar.sync 0;\n");
    ptx.push('\n');
    // Zero dV and dK accumulator registers ONCE here (before q sweep).
    emit_dkdv_acc_init(ptx, config);
}

/// C1: emit the cp.async load for the K tile [effective_bkv, hd] f16 into SMEM.
///
/// K is resident across the q sweep (kv-outer). HBM source: `k_saved_ptr` param.
/// SMEM destination: `tier_b2_dkdv_k_offset(config)` bytes from shmem base.
///
/// Register naming prefix `%c1_` prevents clashes with C2/C3/C4 register namespaces.
fn emit_k_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dkdv_k_offset,
    };

    let bkv = tier_b2_effective_bkv(config);
    let hd = config.head_dim as u32;
    let k_smem_off = tier_b2_dkdv_k_offset(config);

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
        "    // === C1: cp.async K tile [bkv={bkv}, hd={hd}] f16 -> SMEM[+k_offset={k_smem_off}] (kv-outer resident) ===\n"
    ));

    // kv_tile_start = kv_iter * bkv (first KV sequence position; per-kv_iter outer loop).
    ptx.push_str(&format!(
        "    // kv_tile_start = kv_iter * {bkv} (K tile rows are kv-iter-aligned; resident across q sweep)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c1_kv_tile_start, %kv_iter, {bkv};\n"
    ));

    // HBM byte offset for K[batch_idx, head, kv_tile_start, 0].
    crate::flash_attention_v2::tier_b2::backward::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c1_k_hbm_off",
        "%batch_idx",
        "%head",
        "%c1_kv_tile_start",
        "0",        // d=0: column index (full row loaded via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,         // D = head_dim
        2,          // sizeof(f16) = 2
    );

    // HBM source base: load k_saved_ptr from param space.
    ptx.push_str("    ld.param.u64 %c1_k_hbm_base, [k_saved_ptr];\n");
    ptx.push_str("    add.u64 %c1_k_hbm_base, %c1_k_hbm_base, %c1_k_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    ptx.push_str("    shl.b32 %c1_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c1_lane_hbm_addr, %c1_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c1_lane_hbm_addr, %c1_lane_hbm_addr, %c1_k_hbm_base;\n");

    // SMEM destination: u32 shared-space address.
    ptx.push_str("    cvta.shared.u64 %c1_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c1_smem_base32, %c1_smem_base64;\n");
    if k_smem_off > 0 {
        ptx.push_str(&format!(
            "    add.u32 %c1_smem_base32, %c1_smem_base32, {k_smem_off};  // +k_offset\n"
        ));
    }
    ptx.push_str("    add.u32 %c1_smem_dst, %c1_smem_base32, %c1_lane_byte_off;\n");

    // Emit cp.async.ca.shared.global.b32 for each chunk per lane.
    let chunk_stride = 32 * 4u32; // 32 lanes * 4 bytes each
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c1_smem_dst + {chunk_off}], [%c1_lane_hbm_addr + {chunk_off}], 4;\n"
        ));
    }
    // commit_group emitted by caller after V part lands (C2).
}

/// C2: emit the cp.async load for the V tile [effective_bkv, hd] f16 into SMEM.
///
/// V is resident across the q sweep (kv-outer). Mirrors C1 with V source.
/// HBM source: `v_saved_ptr` param.
/// SMEM destination: `tier_b2_dkdv_v_offset(config)` bytes from shmem base.
///
/// Register naming prefix `%c2_` prevents clashes with C1/C3/C4 namespaces.
fn emit_v_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dkdv_v_offset,
    };

    let bkv = tier_b2_effective_bkv(config);
    let hd = config.head_dim as u32;
    let v_smem_off = tier_b2_dkdv_v_offset(config);

    // Total V tile bytes = bkv * hd * 2 (f16). Distributed across 32 lanes.
    let total_bytes = bkv * hd * 2;
    let bytes_per_lane = total_bytes / 32;
    let chunks_per_lane = bytes_per_lane / 4;
    debug_assert!(
        chunks_per_lane >= 1,
        "V tile must have at least one b32 chunk per lane (bkv={}, hd={})",
        bkv, hd
    );

    ptx.push_str(&format!(
        "    // === C2: cp.async V tile [bkv={bkv}, hd={hd}] f16 -> SMEM[+v_offset={v_smem_off}] (kv-outer resident) ===\n"
    ));

    // kv_tile_start = kv_iter * bkv.
    ptx.push_str(&format!(
        "    // kv_tile_start = kv_iter * {bkv} (V tile rows are kv-iter-aligned; resident across q sweep)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c2_kv_tile_start, %kv_iter, {bkv};\n"
    ));

    // HBM byte offset for V[batch_idx, head, kv_tile_start, 0].
    crate::flash_attention_v2::tier_b2::backward::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c2_v_hbm_off",
        "%batch_idx",
        "%head",
        "%c2_kv_tile_start",
        "0",        // d=0: column index (full row loaded via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,         // D = head_dim
        2,          // sizeof(f16) = 2
    );

    // HBM source base: load v_saved_ptr from param space.
    ptx.push_str("    ld.param.u64 %c2_v_hbm_base, [v_saved_ptr];\n");
    ptx.push_str("    add.u64 %c2_v_hbm_base, %c2_v_hbm_base, %c2_v_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    ptx.push_str("    shl.b32 %c2_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c2_lane_hbm_addr, %c2_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c2_lane_hbm_addr, %c2_lane_hbm_addr, %c2_v_hbm_base;\n");

    // SMEM destination: u32 shared-space address + v_offset.
    ptx.push_str("    cvta.shared.u64 %c2_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c2_smem_base32, %c2_smem_base64;\n");
    // v_smem_off > 0 (V follows Q + K tiles), so always add.
    ptx.push_str(&format!(
        "    add.u32 %c2_smem_base32, %c2_smem_base32, {v_smem_off};  // +v_offset\n"
    ));
    ptx.push_str("    add.u32 %c2_smem_dst, %c2_smem_base32, %c2_lane_byte_off;\n");

    // Emit cp.async.ca.shared.global.b32 for each chunk per lane.
    let chunk_stride = 32 * 4u32; // 32 lanes * 4 bytes each
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c2_smem_dst + {chunk_off}], [%c2_lane_hbm_addr + {chunk_off}], 4;\n"
        ));
    }
    // commit_group emitted by caller after both C1 and C2 land.
}

/// Zero dV and dK accumulator registers once at kv-outer loop open.
///
/// dV_acc and dK_acc accumulate over the q axis for a fixed kv tile, so they
/// must be zeroed exactly ONCE per kv_iter (before the q-inner loop). This
/// mirrors the dQ pattern (emit_dq_acc_init in dq.rs) but produces two sets
/// of accumulators (dV and dK each need hd/8 n-tiles x 4 f32/lane).
fn emit_dkdv_acc_init(ptx: &mut String, config: &FlashAttentionConfig) {
    let n_acc = (config.head_dim / 8) as u32;
    ptx.push_str(&format!(
        "    // Zero dV+dK accumulator regs (hd/8={} contiguous n-tiles x 4 f32/lane each)\n",
        n_acc,
    ));
    // .reg decls for %dv_acc_* and %dk_acc_* are hoisted to emit_register_decls; only zero here.
    for n in 0..n_acc {
        for r in 0..4 {
            ptx.push_str(&format!("    mov.f32 %dv_acc_{n}_{r}, 0.0;\n"));
        }
    }
    for n in 0..n_acc {
        for r in 0..4 {
            ptx.push_str(&format!("    mov.f32 %dk_acc_{n}_{r}, 0.0;\n"));
        }
    }
}

/// q-INNER loop open: initialize q_iter, issue cp.async for Q+dO tiles (per-q-iter),
/// and load per-row softmax stats from HBM.
fn emit_inner_loop_open(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    mov.u32 %q_iter, 0;\n");
    ptx.push_str("DKDV_Q_ITER_LOOP:\n");
    // Warp 0 (producer): issue cp.async for Q+dO tiles (refreshed each q_iter).
    // Warps 1-3 (consumers): wait on cp.async.wait_group.
    ptx.push_str("    @!%p_producer bra DKDV_Q_LOAD_DONE;\n");
    emit_q_producer_load(ptx, config);
    emit_do_producer_load(ptx, config);
    ptx.push_str("    cp.async.commit_group;\n");
    ptx.push_str("DKDV_Q_LOAD_DONE:\n");
    ptx.push_str("    cp.async.wait_group 0;\n");
    ptx.push_str("    bar.sync 0;\n");
    // Load per-row softmax stats (row_max, row_sum, D) from HBM for this q tile.
    // Stats are q-row-indexed; they must be loaded inside the q-inner loop (not
    // outer), because each q_iter covers a different set of q rows.
    emit_stats_addr_load(ptx, config);
    ptx.push('\n');
}

/// C3: emit the cp.async load for the Q tile [effective_bq, hd] f16 into SMEM.
///
/// Q is per-q_iter (inner loop). HBM source: `q_saved_ptr` param.
/// SMEM destination: `tier_b2_dkdv_q_offset(config)` bytes from shmem base.
///
/// Register naming prefix `%c3_` prevents clashes with C1/C2/C4 namespaces.
fn emit_q_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::tier_b2_dkdv_q_offset;

    let bq = tier_b2_effective_bq(config);
    let hd = config.head_dim as u32;
    let q_smem_off = tier_b2_dkdv_q_offset(config);

    // Total Q tile bytes = bq * hd * 2 (f16). Distributed across 32 lanes.
    let total_bytes = bq * hd * 2;
    let bytes_per_lane = total_bytes / 32;
    let chunks_per_lane = bytes_per_lane / 4;
    debug_assert!(
        chunks_per_lane >= 1,
        "Q tile must have at least one b32 chunk per lane (bq={}, hd={})",
        bq, hd
    );

    ptx.push_str(&format!(
        "    // === C3: cp.async Q tile [bq={bq}, hd={hd}] f16 -> SMEM[+q_offset={q_smem_off}] (q-inner per-iter) ===\n"
    ));

    // q_tile_start = q_iter * bq (first sequence position of this Q tile).
    ptx.push_str(&format!(
        "    // q_tile_start = q_iter * {bq} (Q tile rows are q-iter-aligned; refreshed each q_iter)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c3_q_tile_start, %q_iter, {bq};\n"
    ));

    // HBM byte offset for Q[batch_idx, head, q_tile_start, 0].
    crate::flash_attention_v2::tier_b2::backward::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c3_q_hbm_off",
        "%batch_idx",
        "%head",
        "%c3_q_tile_start",
        "0",        // d=0: column index (full row loaded via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,         // D = head_dim
        2,          // sizeof(f16) = 2
    );

    // HBM source base: load q_saved_ptr from param space.
    ptx.push_str("    ld.param.u64 %c3_q_hbm_base, [q_saved_ptr];\n");
    ptx.push_str("    add.u64 %c3_q_hbm_base, %c3_q_hbm_base, %c3_q_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    ptx.push_str("    shl.b32 %c3_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c3_lane_hbm_addr, %c3_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c3_lane_hbm_addr, %c3_lane_hbm_addr, %c3_q_hbm_base;\n");

    // SMEM destination: u32 shared-space address.
    ptx.push_str("    cvta.shared.u64 %c3_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c3_smem_base32, %c3_smem_base64;\n");
    if q_smem_off > 0 {
        ptx.push_str(&format!(
            "    add.u32 %c3_smem_base32, %c3_smem_base32, {q_smem_off};  // +q_offset\n"
        ));
    }
    ptx.push_str("    add.u32 %c3_smem_dst, %c3_smem_base32, %c3_lane_byte_off;\n");

    // Emit cp.async.ca.shared.global.b32 for each chunk per lane.
    let chunk_stride = 32 * 4u32;
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c3_smem_dst + {chunk_off}], [%c3_lane_hbm_addr + {chunk_off}], 4;\n"
        ));
    }
    // commit_group emitted by caller (emit_inner_loop_open) after dO part lands (C4).
}

/// C4: emit the cp.async load for the dO tile [effective_bq, hd] f16 into SMEM.
///
/// dO is per-q_iter (inner loop). Mirrors C3 with dO source.
/// HBM source: `d_o_ptr` param.
/// SMEM destination: `tier_b2_dkdv_dO_offset(config)` bytes from shmem base.
///
/// Register naming prefix `%c4_` prevents clashes with C1/C2/C3 namespaces.
#[allow(non_snake_case)]
fn emit_do_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::tier_b2_dkdv_dO_offset;

    let bq = tier_b2_effective_bq(config);
    let hd = config.head_dim as u32;
    let do_smem_off = tier_b2_dkdv_dO_offset(config);

    // Total dO tile bytes = bq * hd * 2 (f16). Distributed across 32 lanes.
    let total_bytes = bq * hd * 2;
    let bytes_per_lane = total_bytes / 32;
    let chunks_per_lane = bytes_per_lane / 4;
    debug_assert!(
        chunks_per_lane >= 1,
        "dO tile must have at least one b32 chunk per lane (bq={}, hd={})",
        bq, hd
    );

    ptx.push_str(&format!(
        "    // === C4: cp.async dO tile [bq={bq}, hd={hd}] f16 -> SMEM[+dO_offset={do_smem_off}] (q-inner per-iter) ===\n"
    ));

    // q_tile_start = q_iter * bq (same as Q; dO rows are Q-aligned).
    ptx.push_str(&format!(
        "    // q_tile_start = q_iter * {bq} (dO tile rows are Q-aligned; refreshed each q_iter)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c4_q_tile_start, %q_iter, {bq};\n"
    ));

    // HBM byte offset for dO[batch_idx, head, q_tile_start, 0].
    crate::flash_attention_v2::tier_b2::backward::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c4_do_hbm_off",
        "%batch_idx",
        "%head",
        "%c4_q_tile_start",
        "0",        // d=0: column index (full row loaded via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,         // D = head_dim
        2,          // sizeof(f16) = 2
    );

    // HBM source base: load d_o_ptr from param space.
    ptx.push_str("    ld.param.u64 %c4_do_hbm_base, [d_o_ptr];\n");
    ptx.push_str("    add.u64 %c4_do_hbm_base, %c4_do_hbm_base, %c4_do_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    ptx.push_str("    shl.b32 %c4_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c4_lane_hbm_addr, %c4_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c4_lane_hbm_addr, %c4_lane_hbm_addr, %c4_do_hbm_base;\n");

    // SMEM destination: u32 shared-space address + dO_offset.
    ptx.push_str("    cvta.shared.u64 %c4_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c4_smem_base32, %c4_smem_base64;\n");
    // do_smem_off > 0 (dO follows Q + K + V tiles), so always add.
    ptx.push_str(&format!(
        "    add.u32 %c4_smem_base32, %c4_smem_base32, {do_smem_off};  // +dO_offset\n"
    ));
    ptx.push_str("    add.u32 %c4_smem_dst, %c4_smem_base32, %c4_lane_byte_off;\n");

    // Emit cp.async.ca.shared.global.b32 for each chunk per lane.
    let chunk_stride = 32 * 4u32;
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c4_smem_dst + {chunk_off}], [%c4_lane_hbm_addr + {chunk_off}], 4;\n"
        ));
    }
    // commit_group emitted by caller after both C3 and C4 land.
}

/// Load per-row softmax stats (row_max, row_sum, D) from HBM for the current q tile.
///
/// Stats are indexed by q rows: s_lo = q_iter*bq + band_row_base + lane/4;
/// s_hi = s_lo + 8.  Called inside the q-inner loop (per-q_iter), because
/// each q_iter covers a different set of q rows.
fn emit_stats_addr_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use super::hbm_addr::emit_3d_byte_offset;
    let bq = tier_b2_effective_bq(config);
    ptx.push_str("    // === Per-row stats load (s_lo/s_hi = q_iter*bq + band_row_base + lane/4 [+8]) ===\n");
    ptx.push_str("    ld.param.u64 %stats_rmax_base, [row_max_ptr];\n");
    ptx.push_str("    ld.param.u64 %stats_rsum_base, [row_sum_ptr];\n");
    ptx.push_str("    ld.param.u64 %stats_d_base, [d_ptr];\n");
    ptx.push_str("    shr.u32 %stat_lane_div4, %lane_id, 2;\n");
    ptx.push_str(&format!("    mul.lo.u32 %s_lo, %q_iter, {bq};\n"));
    ptx.push_str("    add.u32 %s_lo, %s_lo, %band_row_base;\n");
    ptx.push_str("    add.u32 %s_lo, %s_lo, %stat_lane_div4;\n");
    ptx.push_str("    add.u32 %s_hi, %s_lo, 8;\n");
    emit_3d_byte_offset(ptx, "%stats_off_lo", "%batch_idx", "%head", "%s_lo", "%heads_r", "%seq_len_r", 4);
    emit_3d_byte_offset(ptx, "%stats_off_hi", "%batch_idx", "%head", "%s_hi", "%heads_r", "%seq_len_r", 4);
    for (base, reg, off) in [
        ("%stats_rmax_base", "%rmax_lo", "%stats_off_lo"),
        ("%stats_rmax_base", "%rmax_hi", "%stats_off_hi"),
        ("%stats_rsum_base", "%rsum_lo", "%stats_off_lo"),
        ("%stats_rsum_base", "%rsum_hi", "%stats_off_hi"),
        ("%stats_d_base",    "%d_lo",    "%stats_off_lo"),
        ("%stats_d_base",    "%d_hi",    "%stats_off_hi"),
    ] {
        ptx.push_str(&format!("    add.u64 %stats_addr, {base}, {off};\n"));
        ptx.push_str(&format!("    ld.global.f32 {reg}, [%stats_addr];\n"));
    }
    ptx.push_str("    rcp.approx.f32 %rsum_recip_lo, %rsum_lo;\n");
    ptx.push_str("    rcp.approx.f32 %rsum_recip_hi, %rsum_hi;\n");
}

/// D1: emit the tile_skip predicate (causal masking) per Phase 1 spec ss9.2.
///
/// Produces `%tile_skip_predicate` (u32, 0=skip, 1=active) used by the
/// `setp.eq + @!bra DKDV_DS_SKIP_LABEL` consumer at the top of emit_inner_loop_body.
///
/// For dK/dV the loop nesting is kv-outer / q-inner, but the causal predicate
/// is structurally identical to dq: kv_tile_start <= q_tile_end means the current
/// kv-tile can interact with the current q-tile (they overlap in the causal mask).
/// Both %q_iter and %kv_iter exist in the dkdv inner loop context.
fn emit_tile_skip_predicate(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{tier_b2_effective_bkv, tier_b2_effective_bq};
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

/// E1: emit the tiled S = Q @ K^T m16n8k16 matmul for ONE runtime n_tile.
///
/// Mirrors dq.rs::emit_s_matmul_tiled exactly, with SMEM offsets swapped to
/// the dK/dV layout (tier_b2_dkdv_q_offset for Q, tier_b2_dkdv_k_offset for K).
///
/// Bug-fix invariants preserved from dq (the whole reason for this fresh branch):
///   1. `%a_base`/`%b_base` each get `add.u32 ..., %mma_smem_base;` — the
///      cvta'd shared base computed once in grid setup. A raw byte offset would
///      read uninitialized SMEM.
///   2. row_stride = hd*2 for both A (Q rows) and B (K^T cols).
///   3. n_k_tiles = hd/16 (contraction over head_dim in 16-wide k-tiles).
fn emit_s_matmul_tiled(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    a_regs: &[String; 4],
    b_regs: &[String; 2],
    c_regs: &[String; 4],
    d_regs: &[String; 4],
) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dkdv_k_offset, tier_b2_dkdv_q_offset,
    };
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    let hd = config.head_dim as u32;
    let row_stride = (hd * 2) as usize;
    let q_off = tier_b2_dkdv_q_offset(config);
    let k_off = tier_b2_dkdv_k_offset(config);
    let n_k_tiles = hd / 16;
    let pct4 = |r: &[String; 4]| {
        [
            format!("%{}", r[0]),
            format!("%{}", r[1]),
            format!("%{}", r[2]),
            format!("%{}", r[3]),
        ]
    };
    let pct2 = |r: &[String; 2]| [format!("%{}", r[0]), format!("%{}", r[1])];
    for r in c_regs {
        ptx.push_str(&format!("    mov.f32 %{r}, 0.0;\n"));
    }
    for k in 0..n_k_tiles {
        // A base = cvta(shmem) + q_off + band_row_base*hd*2 + k*16*2 (Q-band rows, k-tile cols)
        ptx.push_str(&format!(
            "    mul.lo.u32 %a_base, %band_row_base, {};\n",
            hd * 2
        ));
        ptx.push_str(&format!(
            "    add.u32 %a_base, %a_base, {};\n",
            q_off + k * 16 * 2
        ));
        ptx.push_str("    add.u32 %a_base, %a_base, %mma_smem_base;  // + cvta(shmem) base\n");
        // B base = cvta(shmem) + k_off + n_tile*8*hd*2 + k*16*2 (K^T: kv cols via %n_tile, hd k)
        ptx.push_str(&format!(
            "    mul.lo.u32 %b_base, %n_tile, {};\n",
            8 * hd * 2
        ));
        ptx.push_str(&format!(
            "    add.u32 %b_base, %b_base, {};\n",
            k_off + k * 16 * 2
        ));
        ptx.push_str("    add.u32 %b_base, %b_base, %mma_smem_base;  // + cvta(shmem) base\n");
        emit_load_a_fragment_smem(ptx, a_regs, "%a_base", row_stride);
        // col_stride = hd*2 (row-major K byte-aliases K^T)
        emit_load_b_fragment_smem(ptx, b_regs, "%b_base", row_stride);
        emit_mma_instruction(ptx, &pct4(d_regs), &pct4(a_regs), &pct2(b_regs), &pct4(c_regs));
        for (c, d) in c_regs.iter().zip(d_regs.iter()) {
            // MAC: next k accumulates onto this D
            ptx.push_str(&format!("    mov.f32 %{c}, %{d};\n"));
        }
    }
}

/// Emit the codegen-unrolled k-tile contraction for dP = dO @ V^T for ONE runtime
/// n_tile (the kv-column tile selected by the live `%n_tile` register).
///
/// Mirrors `emit_s_matmul_tiled` for the dK/dV layout:
///   - A = dO  (A-base: `tier_b2_dkdv_dO_offset`, `%band_row_base` q-row offset)
///   - B = V   (B-base: `tier_b2_dkdv_v_offset`, runtime `%n_tile` kv-column offset)
///   - row-major V byte-aliases V^T with col_stride = hd*2
///
/// Bug-fix invariants preserved from dq:
///   1. Both `%a_base` and `%b_base` get `add.u32 ..., %mma_smem_base;` (the cvta'd
///      shared base computed once in grid setup). A raw byte offset would read
///      uninitialized SMEM.
///   2. row_stride = hd*2 for both A (dO rows) and B (V^T cols).
///   3. n_k_tiles = hd/16 (contraction over head_dim in 16-wide k-tiles).
#[allow(non_snake_case)]
fn emit_dp_matmul_tiled(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    a_regs: &[String; 4],
    b_regs: &[String; 2],
    c_regs: &[String; 4],
    d_regs: &[String; 4],
) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dkdv_dO_offset, tier_b2_dkdv_v_offset,
    };
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    let hd = config.head_dim as u32;
    let row_stride = (hd * 2) as usize;
    let do_off = tier_b2_dkdv_dO_offset(config);
    let v_off = tier_b2_dkdv_v_offset(config);
    let n_k_tiles = hd / 16;
    let pct4 = |r: &[String; 4]| {
        [
            format!("%{}", r[0]),
            format!("%{}", r[1]),
            format!("%{}", r[2]),
            format!("%{}", r[3]),
        ]
    };
    let pct2 = |r: &[String; 2]| [format!("%{}", r[0]), format!("%{}", r[1])];
    for r in c_regs {
        ptx.push_str(&format!("    mov.f32 %{r}, 0.0;\n"));
    }
    for k in 0..n_k_tiles {
        // A base = cvta(shmem) + do_off + band_row_base*hd*2 + k*16*2  (dO-band rows, k-tile cols)
        ptx.push_str(&format!(
            "    mul.lo.u32 %a_base, %band_row_base, {};\n",
            hd * 2
        ));
        ptx.push_str(&format!(
            "    add.u32 %a_base, %a_base, {};\n",
            do_off + k * 16 * 2
        ));
        ptx.push_str("    add.u32 %a_base, %a_base, %mma_smem_base;  // + cvta(shmem) base\n");
        // B base = cvta(shmem) + v_off + n_tile*8*hd*2 + k*16*2  (V^T: kv cols via runtime %n_tile, hd k)
        ptx.push_str(&format!(
            "    mul.lo.u32 %b_base, %n_tile, {};\n",
            8 * hd * 2
        ));
        ptx.push_str(&format!(
            "    add.u32 %b_base, %b_base, {};\n",
            v_off + k * 16 * 2
        ));
        ptx.push_str("    add.u32 %b_base, %b_base, %mma_smem_base;  // + cvta(shmem) base\n");
        emit_load_a_fragment_smem(ptx, a_regs, "%a_base", row_stride);
        emit_load_b_fragment_smem(ptx, b_regs, "%b_base", row_stride); // col_stride = hd*2 (row-major V byte-aliases V^T)
        emit_mma_instruction(ptx, &pct4(d_regs), &pct4(a_regs), &pct2(b_regs), &pct4(c_regs));
        for (c, d) in c_regs.iter().zip(d_regs.iter()) {
            ptx.push_str(&format!("    mov.f32 %{c}, %{d};\n")); // MAC: next k accumulates onto this D
        }
    }
}

/// Emit the dK/dV inner-loop body: tile-skip gate, S=Q@K^T MMA, P-recompute.
///
/// This is Task 4. The structure mirrors dq.rs::emit_inner_loop_body (top portion
/// only: S + P-recompute). The dP/dS/scatter/restage/MMA-3/4 are Tasks 5-8.
///
/// Key differences vs dq:
///   - Labels use the DKDV_ prefix (both kernels concatenate into one PTX module).
///   - SMEM offsets use tier_b2_dkdv_q_offset / tier_b2_dkdv_k_offset.
///   - num_n_tiles = bkv/8 (contraction over the kv axis, one 8-col n-tile at a time).
///   - %band_row_base maps to q-rows (per emit_warp_band_setup: warp_id*16 q-rows).
///   - Per-row stats (%rmax_lo/hi, %rsum_recip_lo/hi) are loaded by emit_stats_addr_load
///     inside the q-inner loop (indexed by %q_iter), so they are fresh here.
fn emit_inner_loop_body(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::tier_b2_effective_bkv;

    let hd = config.head_dim as u32;
    let _ = hd; // used indirectly via emit_s_matmul_tiled

    // === Tile-skip gate (spec ss9.2) ===
    ptx.push_str("    // Tile-skip gate: skip S/P/dP/dS/dK/dV-update when tile is masked.\n");
    ptx.push_str("    setp.eq.u32 %p_tile_active, %tile_skip_predicate, 1;\n");
    ptx.push_str("    @!%p_tile_active bra DKDV_DS_SKIP_LABEL;\n");
    ptx.push('\n');

    // === S register family ===
    // %s_d0..3 holds the raw S result; P-recompute reads it directly.
    // .reg decls are hoisted to emit_register_decls (PTX ISA requires all .reg
    // directives to precede executable instructions). These arrays only name the regs.
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

    // === Open runtime n-tile streaming loop (DKDV_NTILE_LOOP) ===
    // One iteration per 8-column kv n-tile (bkv/8 iterations at runtime). The body
    // computes S(tiled) -> P-recompute for this %n_tile.
    // Tasks 5-8 will add dP/dS/scatter/MMA-3/4 inside the n-tile loop.
    let num_n_tiles = tier_b2_effective_bkv(config) / 8;
    // Idle-warp gate: warps with warp_id >= bkv/16 skip the n-tile loop entirely.
    // The loop has NO bar.sync inside, so skipping is deadlock-safe; idle warps
    // reconverge at DKDV_IDLE_SKIP_NTILE.
    ptx.push_str("    @!%p_warp_active bra DKDV_IDLE_SKIP_NTILE;\n");
    ptx.push_str(&format!("    mov.u32 %num_n_tiles, {num_n_tiles};\n"));
    ptx.push_str("    mov.u32 %n_tile, 0;\n");
    ptx.push_str("DKDV_NTILE_LOOP:\n");

    // === S = Q @ K^T (m16n8k16, k-tile contraction codegen-unrolled per n_tile) ===
    // Row-major K[bkv, hd] byte-aliases to col-major K^T[hd, bkv] with
    // col_stride_bytes = hd*2. A-frag base folds %band_row_base (q-rows);
    // B-frag base derives from runtime %n_tile (kv columns).
    ptx.push_str("    // === E1: S = Q @ K^T (tiled m16n8k16, k-tiles unrolled, per runtime n_tile) ===\n");
    emit_s_matmul_tiled(ptx, config, &a_regs, &b_regs, &c_regs, &d_regs);

    // === P recompute (lane-by-lane, no SMEM) per spec ss4.3 step 4 ===
    // P[q,k] = softmax(scale * S[q,k]) = exp(scale*S[q,k] - row_max[q]) * row_sum_recip[q].
    // The forward computed row_max/row_sum on SCALED scores, so we must scale
    // the raw MMA S before exp. Omitting the scale leaves P ~sqrt(D)x too large.
    // Stats loaded by emit_stats_addr_load: %rmax_lo/%rmax_hi, %rsum_recip_lo/%rsum_recip_hi.
    ptx.push_str("    // === P recompute: P[q,k] = exp(scale*S[q,k] - row_max[q]) * row_sum_recip[q] ===\n");
    ptx.push_str("    // Lane-by-lane on f32 S-fragment values in %s_d0..3.\n");
    ptx.push_str("    // Elements {0,1} -> row lane/4 (lo); elements {2,3} -> row lane/4+8 (hi).\n");
    // %p_recip_log2e/%f_scale/%p_0..3 declared in emit_register_decls (hoisted).
    ptx.push_str("    mov.f32 %p_recip_log2e, 0F3FB8AA3B;  // 1/ln(2) = 1.4426950408889634\n");
    // Attention scale = 1/sqrt(head_dim).
    let scale_bits = (1.0f32 / (config.head_dim as f32).sqrt()).to_bits();
    ptx.push_str(&format!(
        "    mov.f32 %f_scale, 0F{scale_bits:08X};  // 1/sqrt(head_dim)\n"
    ));
    for i in 0..4 {
        let (rmax, rrecip) = if i < 2 {
            ("%rmax_lo", "%rsum_recip_lo")
        } else {
            ("%rmax_hi", "%rsum_recip_hi")
        };
        // Bug fix 2: multiply by %f_scale BEFORE subtracting %rmax (match forward's scaled stats).
        ptx.push_str(&format!(
            "    mul.f32 %p_{i}, %s_d{i}, %f_scale;  // scale * S (match forward scaled stats)\n"
        ));
        ptx.push_str(&format!("    sub.f32 %p_{i}, %p_{i}, {rmax};\n"));
        ptx.push_str(&format!(
            "    mul.f32 %p_{i}, %p_{i}, %p_recip_log2e;\n"
        ));
        ptx.push_str(&format!("    ex2.approx.f32 %p_{i}, %p_{i};\n"));
        ptx.push_str(&format!("    mul.f32 %p_{i}, %p_{i}, {rrecip};\n"));
    }
    ptx.push('\n');

    // === dP = dO @ V^T (tiled m16n8k16, per runtime n_tile) ===
    ptx.push_str("    // === dP = dO @ V^T (tiled m16n8k16, k-tiles unrolled, per runtime n_tile) ===\n");
    let dp_a_regs: [String; 4] = ["dp_a0", "dp_a1", "dp_a2", "dp_a3"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let dp_b_regs: [String; 2] = ["dp_b0", "dp_b1"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let dp_c_regs: [String; 4] = ["dp_c0", "dp_c1", "dp_c2", "dp_c3"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    let dp_d_regs: [String; 4] = ["dp_d0", "dp_d1", "dp_d2", "dp_d3"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    // .reg decls for the dP family are hoisted to emit_register_decls (PTX ISA).
    emit_dp_matmul_tiled(ptx, config, &dp_a_regs, &dp_b_regs, &dp_c_regs, &dp_d_regs);
    ptx.push('\n');

    // === dS = P * (dP - D) ===
    //
    // dS[q,k] = (1/sqrt(D)) * P[q,k] * (dP[q,k] - D[q]).
    // The 1/sqrt(D) factor (%f_scale) is set in the P-recompute above and still live.
    // Omitting it makes dK ~sqrt(D)x too large.
    // D[q] is per-row from HBM: %d_lo for elements {0,1} (row lane/4),
    // %d_hi for elements {2,3} (row lane/4+8) — loaded by emit_stats_addr_load.
    ptx.push_str("    // === dS = P * (dP - D) ===\n");
    // %ds_0..3 declared in emit_register_decls (hoisted).
    for i in 0..4 {
        let d_row = if i < 2 { "%d_lo" } else { "%d_hi" };
        ptx.push_str(&format!("    sub.f32 %ds_{i}, %dp_d{i}, {d_row};\n"));
        ptx.push_str(&format!("    mul.f32 %ds_{i}, %ds_{i}, %p_{i};\n"));
        ptx.push_str(&format!("    mul.f32 %ds_{i}, %ds_{i}, %f_scale;  // dS = (1/sqrt(D)) * P * (dP - D)\n"));
    }
    ptx.push('\n');

    // (Task 6: Q/dO re-stage for MMA-3/4)
    // (Task 7: P/dS col scatter for dV/dK MMA inputs)
    // (Task 8: MMA-3 dV_acc += P^T @ dO + MMA-4 dK_acc += dS^T @ Q, after the loop)

    // === Close runtime n-tile streaming loop (DKDV_NTILE_LOOP) ===
    ptx.push_str("    add.u32 %n_tile, %n_tile, 1;\n");
    ptx.push_str("    setp.lt.u32 %p_ntile_more, %n_tile, %num_n_tiles;\n");
    ptx.push_str("    @%p_ntile_more bra DKDV_NTILE_LOOP;\n");
    // Idle-warp reconvergence. No bar.sync inside the n-tile loop, so deadlock-safe.
    ptx.push_str("DKDV_IDLE_SKIP_NTILE:\n");
    ptx.push('\n');

    ptx.push_str("DKDV_DS_SKIP_LABEL:\n");
}

fn emit_inner_loop_close(ptx: &mut String) {
    ptx.push_str("    // q-inner loop back-edge: q_iter += 1; if q_iter < num_q_iters, branch back.\n");
    ptx.push_str("    add.u32 %q_iter, %q_iter, 1;\n");
    ptx.push_str("    setp.lt.u32 %p_q_iter_more, %q_iter, %num_q_iters;\n");
    ptx.push_str("    @%p_q_iter_more bra DKDV_Q_ITER_LOOP;\n");
    ptx.push_str("DKDV_Q_ITER_DONE:\n");
}

/// Finalize stub: placeholder for dV + dK HBM scatter (Task 8 fills this).
///
/// Placed INSIDE the kv-outer loop (after q-inner loop closes, before the
/// kv-outer back-edge), so each kv tile's accumulated dK/dV is written with
/// the correct, in-range kv row index.
fn emit_dkdv_finalize_stub(ptx: &mut String) {
    ptx.push_str("    // === Finalize dV (stub: HBM scatter of dV_acc -- Task 8) ===\n");
    ptx.push_str("DKDV_FINALIZE_DV:\n");
    ptx.push_str("    // === Finalize dK (stub: HBM scatter of dK_acc -- Task 8) ===\n");
    ptx.push_str("DKDV_FINALIZE_DK:\n");
}

fn emit_outer_loop_close(ptx: &mut String) {
    // kv-outer back-edge: kv_iter += 1; if kv_iter < num_kv_iters, branch back.
    ptx.push_str("    // kv-outer loop back-edge: kv_iter += 1; if kv_iter < num_kv_iters, branch back.\n");
    ptx.push_str("    add.u32 %kv_iter, %kv_iter, 1;\n");
    ptx.push_str("    setp.lt.u32 %p_kv_iter_more, %kv_iter, %num_kv_iters;\n");
    ptx.push_str("    @%p_kv_iter_more bra DKDV_KV_ITER_LOOP;\n");
    ptx.push_str("DKDV_KV_ITER_DONE:\n");
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
    fn synthesize_dkdv_kernel_has_kv_outer_q_inner_structure() {
        let ptx = synthesize_dkdv_kernel(&canonical_cfg()).expect("synth ok");
        // kv-outer loop label must come before q-inner loop label.
        let kv_pos = ptx.find("DKDV_KV_ITER_LOOP:").expect("missing kv-outer label");
        let q_pos  = ptx.find("DKDV_Q_ITER_LOOP:").expect("missing q-inner label");
        assert!(kv_pos < q_pos, "kv-outer loop must precede q-inner loop");
        // The q-inner back-edge must come before the kv-outer back-edge.
        let q_done_pos  = ptx.find("DKDV_Q_ITER_DONE:").expect("missing q-inner done label");
        let kv_done_pos = ptx.find("DKDV_KV_ITER_DONE:").expect("missing kv-outer done label");
        assert!(q_done_pos < kv_done_pos, "q-inner close must precede kv-outer close");
        // K+V cp.async loads appear before Q+dO loads (outer before inner).
        let k_cp_pos  = ptx.find("cp.async.ca.shared.global [%c1_smem_dst").expect("missing K cp.async");
        let q_cp_pos  = ptx.find("cp.async.ca.shared.global [%c3_smem_dst").expect("missing Q cp.async");
        assert!(k_cp_pos < q_cp_pos, "K load (c1) must precede Q load (c3) in emitted PTX");
        // Finalize stubs must be between q-inner done and kv-outer done.
        let finalize_pos = ptx.find("Finalize dV").expect("missing finalize dV stub");
        assert!(finalize_pos > q_done_pos, "finalize must appear after q-inner close");
        assert!(finalize_pos < kv_done_pos, "finalize must appear before kv-outer close");
    }

    #[test]
    fn synthesize_dkdv_kernel_rejects_invalid_head_dim() {
        let mut cfg = canonical_cfg();
        cfg.head_dim = 48; // not divisible by 32
        let result = synthesize_dkdv_kernel(&cfg);
        assert_eq!(result, Err(BackwardSynthError::UnsupportedHeadDim(48)));
    }

    #[test]
    fn synthesize_dkdv_kernel_entry_name_present() {
        let ptx = synthesize_dkdv_kernel(&canonical_cfg()).expect("synth ok");
        assert!(ptx.contains("tier_b2_dkdv_kernel"), "entry name must appear in PTX");
    }
}
