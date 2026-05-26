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
    // (Task 4+: tile-skip predicate, S/dP/dS/dV/dK MMA body here)
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
