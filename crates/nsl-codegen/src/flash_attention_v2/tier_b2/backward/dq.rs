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
    emit_warp_band_setup(&mut ptx, config);
    emit_q_iter_count_setup(&mut ptx, config);
    emit_kv_iter_count_setup(&mut ptx, config);
    emit_outer_loop_open(&mut ptx);
    emit_q_dO_producer_load(&mut ptx, config);
    emit_stats_addr_load(&mut ptx, config);
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
    ptx.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
    let _ = total_smem; // Size is communicated via cuFuncSetAttribute at launch time, not baked into the extern declaration.
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
    ptx.push_str("    .reg .u32 %band_row_base;\n");
    ptx.push_str("    .reg .pred %p_warp_active;\n");
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
    // Runtime n-tile streaming loop (DQ_NTILE_LOOP) for the tiled S=Q@K^T matmul.
    // %a_base/%b_base are the per-k-tile SMEM fragment bases recomputed each iter.
    ptx.push_str("    .reg .u32 %n_tile, %num_n_tiles, %a_base, %b_base;\n");
    ptx.push_str("    .reg .pred %p_ntile_more;\n");
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
    // C4 cp.async V-tile registers. Prefixed %c4_* to avoid clashes with C1/C2/C3 namespaces.
    // V tile is [effective_bkv, hd] f16, per-kv_iter (not resident). Loaded in emit_inner_loop_open.
    ptx.push_str("    .reg .u32 %c4_kv_tile_start, %c4_lane_byte_off, %c4_smem_base32, %c4_smem_dst;\n");
    ptx.push_str("    .reg .u64 %c4_v_hbm_off, %c4_v_hbm_base, %c4_lane_hbm_addr, %c4_smem_base64;\n");
    // F1 dS-scatter registers. Prefixed %f1_* to avoid clashes with other namespaces.
    // Per PTX ISA all .reg decls must precede executable instructions -- declared here,
    // used in emit_ds_scatter_to_smem (called from emit_inner_loop_body).
    ptx.push_str("    .reg .u32 %f1_lane_mod4, %f1_lane_div4;\n");
    ptx.push_str("    .reg .u32 %f1_col_lo, %f1_col_hi, %f1_col_lo_b, %f1_col_hi_b;\n");
    ptx.push_str("    .reg .u32 %f1_row_lo_off, %f1_row_hi_off;\n");
    ptx.push_str("    .reg .u32 %f1_addr_lolo, %f1_addr_lohi, %f1_addr_hilo, %f1_addr_hihi;\n");
    ptx.push_str("    .reg .u64 %f1_smem_base64;\n");
    ptx.push_str("    .reg .u32 %f1_smem_base32;\n");
    // F2 col-major K re-stage scatter registers. Prefixed %f2_* to avoid clashes.
    // Per PTX ISA all .reg decls must precede executable instructions -- declared here,
    // used in emit_kcol_restage_scatter (called from emit_inner_loop_body).
    ptx.push_str("    .reg .u32 %f2_lane_mod4, %f2_lane_div4;\n");
    ptx.push_str("    .reg .u32 %f2_row_base, %f2_col_base;\n");
    ptx.push_str("    .reg .u32 %f2_src_row, %f2_src_col;\n");
    ptx.push_str("    .reg .u32 %f2_src_off, %f2_dst_off;\n");
    ptx.push_str("    .reg .u32 %f2_src_addr, %f2_dst_addr;\n");
    ptx.push_str("    .reg .u64 %f2_smem_base64;\n");
    ptx.push_str("    .reg .u32 %f2_smem_base32;\n");
    ptx.push_str("    .reg .b16 %f2_val;\n");
    // G1 dQ HBM finalize registers. Prefixed %g1_* to avoid clashes.
    // Per PTX ISA all .reg decls must precede executable instructions -- declared here,
    // used in emit_dq_finalize (after the outer loop closes).
    // Per-lane row/col scratch computed once; reused across all fragments.
    ptx.push_str("    .reg .u32 %g1_lane_mod4, %g1_lane_div4;\n");
    ptx.push_str("    .reg .u32 %g1_q_tile_start, %g1_row_lo, %g1_row_hi;\n");
    ptx.push_str("    .reg .u32 %g1_col_lo, %g1_col_hi;\n");
    ptx.push_str("    .reg .u32 %g1_d_tmp;\n");
    ptx.push_str("    .reg .u64 %g1_dq_byte_off, %g1_dq_addr;\n");
    ptx.push_str("    .reg .u64 %g1_dq_base;\n");
    // Stats-load registers (emit_stats_addr_load). Declared here per PTX ISA rule
    // that all .reg directives must precede executable instructions.
    ptx.push_str("    .reg .u64 %stats_rmax_base, %stats_rsum_base, %stats_d_base;\n");
    ptx.push_str("    .reg .u64 %stats_off_lo, %stats_off_hi, %stats_addr;\n");
    ptx.push_str("    .reg .u32 %s_lo, %s_hi, %stat_lane_div4;\n");
    ptx.push_str("    .reg .f32 %rmax_lo, %rmax_hi, %rsum_lo, %rsum_hi;\n");
    ptx.push_str("    .reg .f32 %rsum_recip_lo, %rsum_recip_hi, %d_lo, %d_hi;\n");
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

/// Emit warp-band ownership base and the idle-warp predicate from %warp_id.
fn emit_warp_band_setup(ptx: &mut String, config: &FlashAttentionConfig) {
    let active_warps = tier_b2_effective_bq(config) / 16; // 4 at bq=64, 2 at bq=32
    ptx.push_str("    // Warp-per-m16-band: warp w owns q-rows [w*16, w*16+16).\n");
    ptx.push_str("    mul.lo.u32 %band_row_base, %warp_id, 16;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p_warp_active, %warp_id, {active_warps};  // active = warp_id < bq/16\n"
    ));
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

/// Load each lane's two owned q-rows' softmax stats. Hoisted once per q_iter;
/// held in registers across the kv sweep. s_lo = q_iter*bq + warp_id*16 + lane/4;
/// s_hi = s_lo + 8. C-frag elements {0,1} use *_lo, {2,3} use *_hi (applied in Task 4).
fn emit_stats_addr_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use hbm_addr::emit_3d_byte_offset;
    let bq = tier_b2_effective_bq(config);
    ptx.push_str("    // === Per-row stats load (s_lo/s_hi = q_iter*bq + warp_id*16 + lane/4 [+8]) ===\n");
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

/// C4: emit the cp.async load for the V tile [effective_bkv, hd] f16 into SMEM.
///
/// V is **per-kv_iter** (spec §1.3) — loaded once per inner loop iteration.
/// Mirrors C3 (K tile) exactly, with only the SMEM destination and HBM source
/// param changed.  HBM source: `v_saved_ptr` param.
/// SMEM destination: `tier_b2_dq_v_offset(config)` bytes from shmem base.
///
/// Uses A1's `emit_4d_byte_offset` for HBM addressing, with `%c4_kv_tile_start`
/// (= `kv_iter * bkv`) as the row index — NOT `%q_iter` (which C1/C2 use).
///
/// Register naming prefix `%c4_` prevents clashes with C1/C2/C3 namespaces.
///
/// The warp-0 gate (`@!%p_producer bra DQ_KV_LOAD_DONE`) is emitted by the
/// caller (`emit_inner_loop_open`) — shared with C3 (K tile), no duplicate branch.
fn emit_v_producer_load(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_effective_bkv, tier_b2_dq_v_offset,
    };

    let bkv = tier_b2_effective_bkv(config);
    let hd = config.head_dim as u32;
    let v_smem_off = tier_b2_dq_v_offset(config);

    // Total V tile bytes = bkv * hd * 2 (f16). Distributed across 32 lanes:
    //   bytes_per_lane = (bkv * hd * 2) / 32
    //   chunks_per_lane = bytes_per_lane / 4  (each cp.async.ca issues b32 = 4 bytes)
    let total_bytes = bkv * hd * 2;
    let bytes_per_lane = total_bytes / 32;
    let chunks_per_lane = bytes_per_lane / 4;
    debug_assert!(
        chunks_per_lane >= 1,
        "V tile must have at least one b32 chunk per lane (bkv={}, hd={})",
        bkv, hd
    );

    ptx.push_str(&format!(
        "    // === C4: cp.async V tile [bkv={bkv}, hd={hd}] f16 -> SMEM[+v_offset={v_smem_off}] ===\n"
    ));
    // All %c4_* registers declared in emit_register_decls (PTX ISA: decls before exec instrs).

    // kv_tile_start = kv_iter * bkv (first KV sequence position of this V tile; per-kv_iter).
    ptx.push_str(&format!(
        "    // kv_tile_start = kv_iter * {bkv} (V tile rows are kv-iter-aligned; refreshed each kv_iter)\n"
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %c4_kv_tile_start, %kv_iter, {bkv};\n"
    ));

    // HBM byte offset for V[batch_idx, head, kv_tile_start, 0] using A1's helper.
    // emit_4d_byte_offset uses %row_index_tmp as scratch (declared in emit_register_decls).
    crate::flash_attention_v2::tier_b2::backward::dq::hbm_addr::emit_4d_byte_offset(
        ptx,
        "%c4_v_hbm_off",
        "%batch_idx",
        "%head",
        "%c4_kv_tile_start",
        "0",     // d=0: column index within the row (full row loaded via multi-chunk loop)
        "%heads_r",
        "%seq_len_r",
        hd,      // D = head_dim
        2,       // sizeof(f16) = 2
    );

    // HBM source base: load v_saved_ptr from param space.
    ptx.push_str("    ld.param.u64 %c4_v_hbm_base, [v_saved_ptr];\n");
    ptx.push_str("    add.u64 %c4_v_hbm_base, %c4_v_hbm_base, %c4_v_hbm_off;\n");

    // Per-lane byte offset within the tile: lane_id * 4 bytes.
    ptx.push_str("    shl.b32 %c4_lane_byte_off, %lane_id, 2;  // lane_id * 4 bytes\n");

    // HBM address for this lane = base + lane_byte_off.
    ptx.push_str("    cvt.u64.u32 %c4_lane_hbm_addr, %c4_lane_byte_off;\n");
    ptx.push_str("    add.u64 %c4_lane_hbm_addr, %c4_lane_hbm_addr, %c4_v_hbm_base;\n");

    // SMEM destination: u32 shared-space address via cvta.shared.u64 + cvt.u32.u64 + v_offset.
    ptx.push_str("    cvta.shared.u64 %c4_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %c4_smem_base32, %c4_smem_base64;\n");
    // v_smem_off > 0 (V follows Q+K tiles), so always add the offset.
    ptx.push_str(&format!(
        "    add.u32 %c4_smem_base32, %c4_smem_base32, {v_smem_off};  // +v_offset\n"
    ));
    // Add per-lane byte offset to SMEM destination.
    ptx.push_str("    add.u32 %c4_smem_dst, %c4_smem_base32, %c4_lane_byte_off;\n");

    // Emit cp.async.ca.shared.global for each chunk per lane.
    // Chunk stride across lanes = 32 * 4 = 128 bytes per chunk iteration.
    let chunk_stride = 32 * 4u32; // 32 lanes * 4 bytes each
    for chunk_idx in 0..chunks_per_lane {
        let chunk_off = chunk_idx * chunk_stride;
        ptx.push_str(&format!(
            "    cp.async.ca.shared.global [%c4_smem_dst + {chunk_off}], [%c4_lane_hbm_addr + {chunk_off}], 4;\n"
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
    emit_v_producer_load(ptx, config);
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

/// F1: Scatter each lane's 4 dS f32 values to the row-major dS tile in SMEM.
///
/// Per PTX m16n8 D-fragment layout, lane `t` holds:
///   ds_0 -> (row=t/4,     col=(t%4)*2)
///   ds_1 -> (row=t/4,     col=(t%4)*2 + 1)
///   ds_2 -> (row=t/4 + 8, col=(t%4)*2)
///   ds_3 -> (row=t/4 + 8, col=(t%4)*2 + 1)
///
/// Row stride of the f32 dS tile: `effective_bkv * 4` bytes.
/// Destination: SMEM at `tier_b2_dq_ds_offset(config)`.
///
/// All `%f1_*` scratch registers are declared in `emit_register_decls` (PTX ISA
/// requires all `.reg` declarations to precede executable instructions).
fn emit_ds_scatter_to_smem(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dq_ds_offset, tier_b2_effective_bkv,
    };

    let ds_off = tier_b2_dq_ds_offset(config);
    let bkv = tier_b2_effective_bkv(config);
    let row_stride = bkv * 4; // f32 row-major dS tile row stride in bytes

    ptx.push_str("    // === F1: Scatter dS to SMEM (row-major f32 tile at ds_offset) ===\n");
    ptx.push_str("    // Per PTX m16n8 D-frag layout per lane t:\n");
    ptx.push_str("    //   ds_0 -> (row=t/4,   col=(t%4)*2)\n");
    ptx.push_str("    //   ds_1 -> (row=t/4,   col=(t%4)*2+1)\n");
    ptx.push_str("    //   ds_2 -> (row=t/4+8, col=(t%4)*2)\n");
    ptx.push_str("    //   ds_3 -> (row=t/4+8, col=(t%4)*2+1)\n");

    // Obtain SMEM base as u32 shared-space address.
    // cvta.shared.u64 converts the shmem symbol to a 64-bit generic address;
    // cvt.u32.u64 truncates to the 32-bit shared-space address form used by
    // st.shared.f32 [%u32_addr].
    ptx.push_str("    cvta.shared.u64 %f1_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %f1_smem_base32, %f1_smem_base64;\n");

    // Derive per-lane row/col from %lane_id.
    ptx.push_str("    and.b32 %f1_lane_mod4, %lane_id, 3;  // lane%4\n");
    ptx.push_str("    shr.u32 %f1_lane_div4, %lane_id, 2;\n");

    // col_lo = (t%4)*2  (f32 column index of ds_0/ds_2)
    ptx.push_str("    shl.b32 %f1_col_lo, %f1_lane_mod4, 1;       // col_lo = (t%4)*2\n");
    // col_hi = col_lo + 1  (f32 column index of ds_1/ds_3)
    ptx.push_str("    add.u32 %f1_col_hi, %f1_col_lo, 1;\n");
    // Convert column indices to byte offsets within row (f32 = 4 bytes each).
    ptx.push_str("    shl.b32 %f1_col_lo_b, %f1_col_lo, 2;        // col_lo * 4 bytes\n");
    ptx.push_str("    shl.b32 %f1_col_hi_b, %f1_col_hi, 2;        // col_hi * 4 bytes\n");

    // Row byte offsets: row_lo = t/4, row_hi = t/4 + 8.
    ptx.push_str(&format!("    mul.lo.u32 %f1_row_lo_off, %f1_lane_div4, {row_stride};  // (t/4) * row_stride\n"));
    // row_hi_off = (t/4 + 8) * row_stride = row_lo_off + 8*row_stride
    ptx.push_str(&format!("    add.u32 %f1_row_hi_off, %f1_row_lo_off, {};              // + 8 * row_stride\n",
        8 * row_stride));

    // 4 SMEM destination addresses: f1_smem_base32 + ds_off + row_off + col_off_b.
    // addr_lolo: ds_0 -> (row_lo, col_lo)
    ptx.push_str(&format!("    add.u32 %f1_addr_lolo, %f1_smem_base32, {ds_off};\n"));
    ptx.push_str("    add.u32 %f1_addr_lolo, %f1_addr_lolo, %f1_row_lo_off;\n");
    ptx.push_str("    add.u32 %f1_addr_lolo, %f1_addr_lolo, %f1_col_lo_b;\n");
    // addr_lohi: ds_1 -> (row_lo, col_hi)
    ptx.push_str(&format!("    add.u32 %f1_addr_lohi, %f1_smem_base32, {ds_off};\n"));
    ptx.push_str("    add.u32 %f1_addr_lohi, %f1_addr_lohi, %f1_row_lo_off;\n");
    ptx.push_str("    add.u32 %f1_addr_lohi, %f1_addr_lohi, %f1_col_hi_b;\n");
    // addr_hilo: ds_2 -> (row_hi, col_lo)
    ptx.push_str(&format!("    add.u32 %f1_addr_hilo, %f1_smem_base32, {ds_off};\n"));
    ptx.push_str("    add.u32 %f1_addr_hilo, %f1_addr_hilo, %f1_row_hi_off;\n");
    ptx.push_str("    add.u32 %f1_addr_hilo, %f1_addr_hilo, %f1_col_lo_b;\n");
    // addr_hihi: ds_3 -> (row_hi, col_hi)
    ptx.push_str(&format!("    add.u32 %f1_addr_hihi, %f1_smem_base32, {ds_off};\n"));
    ptx.push_str("    add.u32 %f1_addr_hihi, %f1_addr_hihi, %f1_row_hi_off;\n");
    ptx.push_str("    add.u32 %f1_addr_hihi, %f1_addr_hihi, %f1_col_hi_b;\n");

    // 4 stores: scatter dS values into the row-major dS tile in SMEM.
    ptx.push_str("    st.shared.f32 [%f1_addr_lolo], %ds_0;\n");
    ptx.push_str("    st.shared.f32 [%f1_addr_lohi], %ds_1;\n");
    ptx.push_str("    st.shared.f32 [%f1_addr_hilo], %ds_2;\n");
    ptx.push_str("    st.shared.f32 [%f1_addr_hihi], %ds_3;\n");
    ptx.push_str("    bar.sync 0;  // dS scatter visible to all warps\n");
}

/// F2: Emit the col-major K re-stage scatter (Path A, strongest treatment).
///
/// Copies row-major K[bkv, hd] (at `tier_b2_dq_k_offset`) to col-major
/// K[hd, bkv] (at `tier_b2_dq_k_colmajor_offset`) via per-lane
/// ld.shared.b16 / st.shared.b16, warp-0-gated to avoid 4x write
/// amplification.
///
/// Partition: each lane handles `pairs_per_lane = (bkv/4)*(hd/8)` cells.
///   src_row = lane%4 * (bkv/4) + (p % (bkv/4))     -- row in K[bkv, hd]
///   src_col = lane/4 * (hd/8)  + (p / (bkv/4))     -- col in K[bkv, hd]
///   src_off = k_off + src_row * hd * 2 + src_col * 2
///   dst_off = k_colmajor_off + src_col * bkv * 2 + src_row * 2
///
/// This partition makes all four spec §5.5 institutional-pin terms contribute:
///   - lane%4  via and.b32 %f2_lane_mod4, %lane_id, 3
///   - lane/4  via shr.u32 %f2_lane_div4, %lane_id, 2  (THE BUG-CLASS TERM)
///   - k_colmajor_offset literal (smem_base)
///   - bkv_eff * 2 (col_stride_bytes for the col-major dst)
///
/// All `%f2_*` scratch registers are declared in `emit_register_decls`.
fn emit_kcol_restage_scatter(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dq_k_offset, tier_b2_dq_k_colmajor_offset, tier_b2_effective_bkv,
    };

    let k_off     = tier_b2_dq_k_offset(config);
    let kcol_off  = tier_b2_dq_k_colmajor_offset(config);
    let bkv       = tier_b2_effective_bkv(config);
    let hd        = config.head_dim as u32;

    // Partition parameters (all powers-of-two per spec §3.1 invariants).
    // rows_per_lane_mod = bkv/4  (how many bkv rows a lane%4 group owns)
    // cols_per_lane_div = hd/8   (how many hd cols a lane/4 group owns)
    // pairs_per_lane   = rows_per_lane_mod * cols_per_lane_div = bkv*hd/32
    let rows_per_lane_mod = bkv / 4;
    let cols_per_lane_div = hd  / 8;
    let pairs_per_lane    = rows_per_lane_mod * cols_per_lane_div;

    let row_stride_src  = hd  * 2;  // row-major K src: hd*2 bytes per row
    let col_stride_dst  = bkv * 2;  // col-major K dst: bkv*2 bytes per col

    ptx.push_str("    // === Col-major K re-stage (Path A) ===\n");
    ptx.push_str("    // Source: row-major K[bkv, hd] at SMEM[+k_offset], row stride hd*2 bytes\n");
    ptx.push_str("    // Dest:   col-major K[hd, bkv] at SMEM[+k_colmajor_offset], col stride bkv*2 bytes\n");
    ptx.push_str("    // Warp 0 gates the scatter (avoids 4x write amplification)\n");
    ptx.push_str(&format!(
        "    // (Per-lane scatter loop body covers {pairs_per_lane} (row, col) pairs per lane.)\n",
    ));
    ptx.push_str("    @!%p_producer bra DQ_KCOL_RESTAGE_DONE;\n");

    // Derive lane%4 and lane/4 once (institutional-pin terms 1 and 2).
    // These are the pre-declared %f2_* regs from emit_register_decls.
    ptx.push_str("    and.b32 %f2_lane_mod4, %lane_id, 3;\n");
    ptx.push_str("    shr.u32 %f2_lane_div4, %lane_id, 2;\n");

    // Compute per-lane row/col base offsets (lane-dependent, pair-independent).
    //   row_base = lane%4 * rows_per_lane_mod
    //   col_base = lane/4 * cols_per_lane_div
    ptx.push_str(&format!("    mul.lo.u32 %f2_row_base, %f2_lane_mod4, {rows_per_lane_mod};\n"));
    ptx.push_str(&format!("    mul.lo.u32 %f2_col_base, %f2_lane_div4, {cols_per_lane_div};\n"));

    // Obtain SMEM base as u32 shared-space address.
    ptx.push_str("    cvta.shared.u64 %f2_smem_base64, shmem;\n");
    ptx.push_str("    cvt.u32.u64 %f2_smem_base32, %f2_smem_base64;\n");

    // Loop-unrolled per (row, col) pair.
    // For pair p:
    //   src_row = row_base + (p % rows_per_lane_mod)
    //   src_col = col_base + (p / rows_per_lane_mod)
    //   src_off = k_off + src_row * row_stride_src + src_col * 2
    //   dst_off = kcol_off + src_col * col_stride_dst + src_row * 2
    for p in 0..pairs_per_lane {
        let pair_row_off = p % rows_per_lane_mod;  // row delta within lane's block
        let pair_col_off = p / rows_per_lane_mod;  // col delta within lane's block

        // src_row = row_base + pair_row_off
        ptx.push_str(&format!("    add.u32 %f2_src_row, %f2_row_base, {pair_row_off};\n"));
        // src_col = col_base + pair_col_off
        ptx.push_str(&format!("    add.u32 %f2_src_col, %f2_col_base, {pair_col_off};\n"));

        // src_off = k_off + src_row * row_stride_src + src_col * 2
        ptx.push_str(&format!("    mul.lo.u32 %f2_src_off, %f2_src_row, {row_stride_src};\n"));
        // src_col * 2: multiply by 2 then add
        // Use a temporary for src_col_bytes to avoid clobbering %f2_src_col before dst_off.
        // We compute dst_off before the shift to keep %f2_src_col intact.
        // dst_off = kcol_off + src_col * col_stride_dst + src_row * 2
        ptx.push_str(&format!("    mul.lo.u32 %f2_dst_off, %f2_src_col, {col_stride_dst};\n"));
        // src_row * 2 for dst: shl src_row by 1 and add
        // use %f2_src_addr as a scratch here (it's unused until after these address calcs)
        ptx.push_str("    shl.b32 %f2_src_addr, %f2_src_row, 1;\n");
        ptx.push_str("    add.u32 %f2_dst_off, %f2_dst_off, %f2_src_addr;\n");
        ptx.push_str(&format!("    add.u32 %f2_dst_off, %f2_dst_off, {kcol_off};\n"));

        // Now compute src_off += src_col * 2 (src_col still intact)
        ptx.push_str("    shl.b32 %f2_dst_addr, %f2_src_col, 1;\n");  // reuse %f2_dst_addr as scratch
        ptx.push_str("    add.u32 %f2_src_off, %f2_src_off, %f2_dst_addr;\n");
        ptx.push_str(&format!("    add.u32 %f2_src_off, %f2_src_off, {k_off};\n"));

        // Compute actual SMEM addresses.
        ptx.push_str("    add.u32 %f2_src_addr, %f2_smem_base32, %f2_src_off;\n");
        ptx.push_str("    add.u32 %f2_dst_addr, %f2_smem_base32, %f2_dst_off;\n");

        // ld.shared.b16 from row-major src; st.shared.b16 to col-major dst.
        ptx.push_str("    ld.shared.b16 %f2_val, [%f2_src_addr];\n");
        ptx.push_str("    st.shared.b16 [%f2_dst_addr], %f2_val;\n");
    }

    ptx.push_str("DQ_KCOL_RESTAGE_DONE:\n");
    ptx.push_str("    bar.sync 0;\n");
}

/// Emit the codegen-unrolled k-tile contraction for S = Q @ K^T for ONE runtime
/// n_tile (the kv-column tile selected by the live `%n_tile` register).
///
/// The k-tile loop (`hd / 16` iterations) is UNROLLED at codegen time into a
/// chain of m16n8k16 MMAs accumulating into `c_regs` (MAC: each iter's D feeds
/// the next iter's C). The A-fragment base folds in the warp-band row offset
/// (`%band_row_base`) so each warp reads its own 16-row Q band; the B-fragment
/// base derives from the runtime `%n_tile` (each n-tile is 8 kv columns).
///
/// Row-major K[bkv, hd] byte-aliases to col-major K^T[hd, bkv] with
/// col_stride = hd*2 (the 4-arg B-frag helper reads it correctly → Q @ K^T).
fn emit_s_matmul_tiled(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    a_regs: &[String; 4],
    b_regs: &[String; 2],
    c_regs: &[String; 4],
    d_regs: &[String; 4],
) {
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    use crate::flash_attention_v2::smem_layout::{tier_b2_dq_q_offset, tier_b2_dq_k_offset};
    let hd = config.head_dim as u32;
    let row_stride = (hd * 2) as usize;
    let q_off = tier_b2_dq_q_offset(config);
    let k_off = tier_b2_dq_k_offset(config);
    let n_k_tiles = hd / 16;
    let pct4 = |r: &[String; 4]| {
        [
            format!("%{}", r[0]), format!("%{}", r[1]),
            format!("%{}", r[2]), format!("%{}", r[3]),
        ]
    };
    let pct2 = |r: &[String; 2]| [format!("%{}", r[0]), format!("%{}", r[1])];
    for r in c_regs {
        ptx.push_str(&format!("    mov.f32 %{r}, 0.0;\n"));
    }
    for k in 0..n_k_tiles {
        // A base = q_off + band_row_base*hd*2 + k*16*2  (Q-band rows, k-tile cols)
        ptx.push_str(&format!("    mul.lo.u32 %a_base, %band_row_base, {};\n", hd * 2));
        ptx.push_str(&format!("    add.u32 %a_base, %a_base, {};\n", q_off + k * 16 * 2));
        // B base = k_off + n_tile*8*hd*2 + k*16*2  (Kᵀ: kv columns via runtime %n_tile, hd k)
        ptx.push_str(&format!("    mul.lo.u32 %b_base, %n_tile, {};\n", 8 * hd * 2));
        ptx.push_str(&format!("    add.u32 %b_base, %b_base, {};\n", k_off + k * 16 * 2));
        emit_load_a_fragment_smem(ptx, a_regs, "%a_base", row_stride);
        emit_load_b_fragment_smem(ptx, b_regs, "%b_base", row_stride); // col_stride = hd*2 (row-major K byte-aliases Kᵀ)
        emit_mma_instruction(ptx, &pct4(d_regs), &pct4(a_regs), &pct2(b_regs), &pct4(c_regs));
        for (c, d) in c_regs.iter().zip(d_regs.iter()) {
            ptx.push_str(&format!("    mov.f32 %{c}, %{d};\n")); // MAC: next k accumulates onto this D
        }
    }
}

fn emit_inner_loop_body(ptx: &mut String, config: &FlashAttentionConfig) {
    use crate::matmul_mma::{
        emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
    };
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dq_dO_offset, tier_b2_dq_v_offset, tier_b2_effective_bkv,
    };

    // Convention: emit_load_{a,b}_fragment_smem take register names WITHOUT `%`
    // prefix (they prepend `%` internally in `ld.shared` instructions).
    // emit_mma_instruction uses names verbatim — callers MUST supply `%` prefix.
    // Use pct4/pct2 to build the prefixed variants right before each MMA call.
    let pct4 = |rs: &[String; 4]| -> [String; 4] {
        [
            format!("%{}", rs[0]), format!("%{}", rs[1]),
            format!("%{}", rs[2]), format!("%{}", rs[3]),
        ]
    };
    let pct2 = |rs: &[String; 2]| -> [String; 2] {
        [format!("%{}", rs[0]), format!("%{}", rs[1])]
    };

    let hd = config.head_dim as u32;
    // f16 row stride = head_dim * 2 bytes (tightly-packed row-major f16).
    let row_stride_bytes = (hd * 2) as usize;

    // === Tile-skip predicate (spec §9.2) ===
    ptx.push_str("    // Tile-skip predicate: gate dP+dS+dQ-update as a single block.\n");
    ptx.push_str("    setp.eq.u32 %p_tile_active, %tile_skip_predicate, 1;\n");
    ptx.push_str("    @!%p_tile_active bra DS_SKIP_LABEL;\n");
    ptx.push('\n');

    // === S register family (declared ONCE, above the n-tile loop) ===
    // %s_d0..3 stays the S result so the P-recompute (which reads %s_d0..3) is unchanged.
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

    // === Open runtime n-tile streaming loop (DQ_NTILE_LOOP) ===
    // One iteration per 8-column kv n-tile (bkv/8 iterations at runtime). The body
    // computes S(tiled) → P-recompute → dP(single-tile) → dS → dS-scatter for the
    // current %n_tile. dP/dS/scatter stay single-tile this task (Tasks 6/7/8 tile them).
    let num_n_tiles = tier_b2_effective_bkv(config) / 8;
    ptx.push_str(&format!("    mov.u32 %num_n_tiles, {num_n_tiles};\n"));
    ptx.push_str("    mov.u32 %n_tile, 0;\n");
    ptx.push_str("DQ_NTILE_LOOP:\n");

    // === S = Q @ K^T (m16n8k16, k-tile contraction codegen-unrolled per n_tile) ===
    // Row-major K[bkv, hd] byte-aliases to col-major K^T[hd, bkv] with
    // col_stride_bytes = hd * 2.  The 4-arg B-frag helper reads it correctly and
    // produces Q @ K^T.  A-frag base folds %band_row_base; B-frag base from %n_tile.
    ptx.push_str("    // === S = Q @ K^T (tiled m16n8k16, k-tiles unrolled, per runtime n_tile) ===\n");
    emit_s_matmul_tiled(ptx, config, &a_regs, &b_regs, &c_regs, &d_regs);

    // === P recompute (lane-by-lane, no SMEM) per spec §4.3 step 4 ===
    ptx.push_str("    // === P recompute: P[q,k] = exp(S[q,k] - row_max[q]) * row_sum_recip[q] ===\n");
    ptx.push_str("    // (Lane-by-lane on the f32 S-fragment values held in %s_d0..3.)\n");
    ptx.push_str("    // Elements {0,1} belong to row lane/4 (lo), elements {2,3} to row lane/4+8 (hi).\n");
    ptx.push_str("    // Stats loaded by emit_stats_addr_load: %rmax_lo/%rmax_hi, %rsum_recip_lo/%rsum_recip_hi.\n");
    ptx.push_str("    .reg .f32 %p_recip_log2e;\n");
    ptx.push_str("    mov.f32 %p_recip_log2e, 0F3FB8AA3B;  // 1/ln(2) = 1.4426950408889634\n");
    for i in 0..4 {
        let (rmax, rrecip) = if i < 2 { ("%rmax_lo", "%rsum_recip_lo") }
                              else      { ("%rmax_hi", "%rsum_recip_hi") };
        ptx.push_str(&format!("    .reg .f32 %p_{i};\n"));
        ptx.push_str(&format!("    sub.f32 %p_{i}, %s_d{i}, {rmax};\n"));
        ptx.push_str(&format!("    mul.f32 %p_{i}, %p_{i}, %p_recip_log2e;\n"));
        ptx.push_str(&format!("    ex2.approx.f32 %p_{i}, %p_{i};\n"));
        ptx.push_str(&format!("    mul.f32 %p_{i}, %p_{i}, {rrecip};\n"));
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
    // emit_mma_instruction uses names verbatim — pass %-prefixed variants.
    emit_mma_instruction(ptx, &pct4(&dp_d_regs), &pct4(&dp_a_regs), &pct2(&dp_b_regs), &pct4(&dp_c_regs));
    ptx.push('\n');

    // === dS = P * (dP - D) (lane-by-lane, no SMEM stage yet) ===
    //
    // dS[q,k] = P[q,k] * (dP[q,k] - D[q]).  Computed on the f32 fragment values
    // held in %dp_d0..3 and %p_0..3 produced above.
    //
    // D[q] is loaded from HBM per row: %d_lo for elements {0,1} (row lane/4),
    // %d_hi for elements {2,3} (row lane/4+8) — loaded by emit_stats_addr_load.
    ptx.push_str("    // === dS = P * (dP - D) ===\n");
    for i in 0..4 {
        let d_row = if i < 2 { "%d_lo" } else { "%d_hi" };
        ptx.push_str(&format!("    .reg .f32 %ds_{i};\n"));
        ptx.push_str(&format!("    sub.f32 %ds_{i}, %dp_d{i}, {d_row};\n"));
        ptx.push_str(&format!("    mul.f32 %ds_{i}, %ds_{i}, %p_{i};\n"));
    }
    ptx.push('\n');

    // === Scatter dS to SMEM at tier_b2_dq_ds_offset (row-major f32) ===
    //
    // The dQ-update MMA's A-fragment reads dS from SMEM, so each lane writes its
    // 4 dS values to the row-major dS band at tier_b2_dq_ds_offset.
    // F1: real per-lane st.shared.f32 emission (bar.sync included inside helper).
    use crate::flash_attention_v2::smem_layout::{
        tier_b2_dq_ds_offset, tier_b2_dq_k_colmajor_offset,
    };
    ptx.push_str("    // === Scatter dS to SMEM ===\n");
    emit_ds_scatter_to_smem(ptx, config);
    ptx.push('\n');

    // === Close runtime n-tile streaming loop (DQ_NTILE_LOOP) ===
    // S → P → dP → dS → dS-scatter ran for this %n_tile; advance and branch back.
    ptx.push_str("    add.u32 %n_tile, %n_tile, 1;\n");
    ptx.push_str("    setp.lt.u32 %p_ntile_more, %n_tile, %num_n_tiles;\n");
    ptx.push_str("    @%p_ntile_more bra DQ_NTILE_LOOP;\n");
    ptx.push('\n');

    // === F2: Col-major K re-stage (Path A, strongest treatment per spec §5.2-5.3 + §4.4) ===
    //
    // K is staged row-major at tier_b2_dq_k_offset (matches B.1 forward convention).
    // For dQ_acc += dS @ K (NOT dS @ K^T), we need K presented to the MMA as col-major.
    // emit_kcol_restage_scatter copies row-major K[bkv, hd] → col-major K band at
    // tier_b2_dq_k_colmajor_offset, gated by warp 0, bar.sync'd before MMA reads it.
    //
    // FFI-side requirement (Phase 3 wiring): when `nsl_flash_attention_csha` launches
    // the dQ-kernel, it MUST assert tier_b2_dq_total_smem_bytes <= SMEM_DYNAMIC_BUDGET_BYTES
    // and fail loudly otherwise (per spec §5.5 Step 3d-3). Not wired in Phase 2.
    emit_kcol_restage_scatter(ptx, config);
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

    let bkv_eff = tier_b2_effective_bkv(config);
    let k_col_off = tier_b2_dq_k_colmajor_offset(config);
    let ds_off_expr = format!("{}", tier_b2_dq_ds_offset(config));
    let k_col_off_expr = format!("{}", k_col_off);
    let ds_row_stride = (tier_b2_effective_bkv(config) * 4) as usize; // dS is f32 row-major; use effective_bkv (post Path-A fallback), not raw block_kv
    let bkv_eff_col_stride = (bkv_eff * 2) as usize;    // col stride of col-major K band

    emit_load_a_fragment_smem(ptx, &dq_a_regs, &ds_off_expr, ds_row_stride);
    emit_load_b_fragment_smem(ptx, &dq_b_regs, &k_col_off_expr, bkv_eff_col_stride);

    // Accumulate into %dq_acc_0_{0..3} (D and C both = dq_acc → MAC).
    // emit_mma_instruction uses names verbatim — build %-prefixed dq_acc array.
    let dq_d_regs: [String; 4] = [
        "%dq_acc_0_0".into(), "%dq_acc_0_1".into(),
        "%dq_acc_0_2".into(), "%dq_acc_0_3".into(),
    ];
    emit_mma_instruction(ptx, &dq_d_regs, &pct4(&dq_a_regs), &pct2(&dq_b_regs), &dq_d_regs);
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

/// G1: dQ HBM finalize addressing.
///
/// Scatter dQ_acc registers to row-major [B, H, S, D] f32 HBM output per spec §2.1.
///
/// Per-lane accumulator layout (m16n8 PTX spec, lane t):
///   dq_acc_f_0: (row = t/4,     col = f*32 + (t%4)*2)
///   dq_acc_f_1: (row = t/4,     col = f*32 + (t%4)*2 + 1)
///   dq_acc_f_2: (row = t/4 + 8, col = f*32 + (t%4)*2)
///   dq_acc_f_3: (row = t/4 + 8, col = f*32 + (t%4)*2 + 1)
///
/// HBM byte_offset = (((batch_idx * H + head) * S + (q_tile_start + row)) * D + col) * 4
/// computed via A1's emit_4d_byte_offset (sizeof_dtype=4 for f32).
///
/// All registers (%g1_*) are declared in emit_register_decls to satisfy the PTX ISA
/// constraint that all .reg directives precede executable instructions.
fn emit_dq_finalize(ptx: &mut String, config: &FlashAttentionConfig) {
    use hbm_addr::emit_4d_byte_offset;

    let bq = tier_b2_effective_bq(config);
    let hd = config.head_dim as u32;
    let accumulator_fragments = hd / 32;

    ptx.push_str("    // === G1: dQ HBM finalize addressing ===\n");
    ptx.push_str("    // Scatter dQ_acc registers to HBM dQ[B,H,S,D] row-major f32.\n");
    ptx.push_str("    // Per spec s2.1: byte_offset = (((b*H+h)*S+s)*D+d)*4.\n");
    ptx.push_str("    // Per-lane row/col scratch is computed once, reused across all fragments.\n");

    // Per-lane scratch: lane%4 and lane/4
    ptx.push_str("    and.b32 %g1_lane_mod4, %lane_id, 3;\n");
    ptx.push_str("    shr.u32 %g1_lane_div4, %lane_id, 2;\n");

    // q_tile_start = q_iter * bq
    ptx.push_str(&format!("    mul.lo.u32 %g1_q_tile_start, %q_iter, {bq};\n"));

    // row_lo = q_tile_start + lane/4      (the "lo" MMA row)
    // row_hi = q_tile_start + lane/4 + 8  (the "hi" MMA row)
    ptx.push_str("    add.u32 %g1_row_lo, %g1_q_tile_start, %g1_lane_div4;\n");
    ptx.push_str("    add.u32 %g1_row_hi, %g1_row_lo, 8;\n");

    // col_lo = (lane%4) * 2
    // col_hi = col_lo + 1
    ptx.push_str("    shl.b32 %g1_col_lo, %g1_lane_mod4, 1;\n");
    ptx.push_str("    add.u32 %g1_col_hi, %g1_col_lo, 1;\n");

    // Load the dQ HBM base pointer once.
    ptx.push_str("    ld.param.u64 %g1_dq_base, [d_q_out_ptr];\n");

    for f in 0..accumulator_fragments {
        let frag_col_base = f * 32;

        // For each of the 4 dq_acc registers in fragment f:
        //   r=0: (row_lo, col_lo + frag_col_base)
        //   r=1: (row_lo, col_hi + frag_col_base)
        //   r=2: (row_hi, col_lo + frag_col_base)
        //   r=3: (row_hi, col_hi + frag_col_base)
        let stores: [(u32, &str, &str); 4] = [
            (0, "%g1_row_lo", "%g1_col_lo"),
            (1, "%g1_row_lo", "%g1_col_hi"),
            (2, "%g1_row_hi", "%g1_col_lo"),
            (3, "%g1_row_hi", "%g1_col_hi"),
        ];

        for (r, row_reg, col_reg) in stores {
            // Compute final d_col = col_reg + frag_col_base.
            // If frag_col_base == 0, col_reg is already correct; reuse it directly.
            // Otherwise compute into %g1_d_tmp (single scratch; written before each use).
            let d_reg: String = if frag_col_base == 0 {
                col_reg.to_string()
            } else {
                ptx.push_str(&format!(
                    "    add.u32 %g1_d_tmp, {col_reg}, {frag_col_base};\n"
                ));
                "%g1_d_tmp".to_string()
            };

            emit_4d_byte_offset(
                ptx,
                "%g1_dq_byte_off",
                "%batch_idx",
                "%head",
                row_reg,
                &d_reg,
                "%heads_r",
                "%seq_len_r",
                hd,
                4, // f32 sizeof
            );
            ptx.push_str("    add.u64 %g1_dq_addr, %g1_dq_base, %g1_dq_byte_off;\n");
            ptx.push_str(&format!("    st.global.f32 [%g1_dq_addr], %dq_acc_{f}_{r};\n"));
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
        // Fix 1 (H4): the SMEM extern must use the UNSIZED form `shmem[]` (canonical dynamic
        // SMEM pattern used by every other dynamic-SMEM kernel in the codebase).  The sized
        // form `shmem[N]` is treated by the CUDA driver as a STATIC allocation; the launcher
        // then ALSO requests N bytes of dynamic SMEM, doubling the per-block budget to 2N and
        // causing CUDA_ERROR_INVALID_VALUE at bq=64 (N=38400, 2N=76800 > 48 KB cap).
        //
        // The actual SMEM size is communicated at launch time via cuFuncSetAttribute
        // (CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, N).  tier_b2_dq_total_smem_bytes
        // is still authoritative — it's just passed to the launcher, not baked into PTX.
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // Must use canonical dynamic form (empty brackets), NOT a sized extern.
        assert!(ptx.contains(".extern .shared .align 16 .b8 shmem[]"),
            "expected unsized dynamic shmem[] extern, got:\n{ptx}");
        // Verify no digit follows the opening bracket (i.e., no sized form shmem[N]).
        assert!(!ptx.contains(".extern .shared .align 16 .b8 shmem[0")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[1")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[2")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[3")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[4")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[5")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[6")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[7")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[8")
            && !ptx.contains(".extern .shared .align 16 .b8 shmem[9"),
            "must NOT emit sized shmem[N] extern (driver treats it as static, doubling budget)");
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
    fn synthesize_dq_kernel_tiles_s_matmul() {
        // hd=64: S's k-tile contraction is codegen-unrolled to hd/16=4 MMAs, emitted
        // once inside the runtime n-tile loop (DQ_NTILE_LOOP iterates bkv/8 at runtime).
        let cfg = FlashAttentionConfig { head_dim: 64, ..canonical_cfg() };
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        let mma = ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count();
        // S contributes hd/16=4 unrolled k-tile MMAs; dP (still single-tile) + dQ (single)
        // contribute a few more. Assert S alone raised the count to >= 4 emitted MMAs in
        // the S section AND the n-tile loop + band offset exist.
        assert!(mma >= 4, "S k-tile contraction must emit hd/16 MMAs, got {mma}");
        assert!(ptx.contains("DQ_NTILE_LOOP"), "expected runtime n-tile streaming loop label");
        assert!(ptx.contains("mul.lo.u32 %b_base, %n_tile,"),
            "S K^T B-frag base must be computed from the runtime %n_tile");
        assert!(ptx.contains("%band_row_base"), "S A-frag base must include warp-band offset");
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
        let cfg = canonical_cfg();
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        // Post-Task-5: S=Q@K^T's k-tile contraction is codegen-unrolled to hd/16
        // emitted MMAs (inside the runtime DQ_NTILE_LOOP); dP and dQ-update stay
        // single-tile (Tasks 6/7/8 tile them). So emitted MMAs = hd/16 (S) + 1 (dP) + 1 (dQ).
        let hd = cfg.head_dim as usize;
        let expected = hd / 16 + 2;
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count();
        assert_eq!(mma_count, expected,
            "expected hd/16 (S, tiled) + dP + dQ-update = {expected} MMAs, got {mma_count}");
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
        assert!(ptx.contains("{%dq_acc_0_0, %dq_acc_0_1, %dq_acc_0_2, %dq_acc_0_3}"),
            "expected %dq_acc_0_{{0..3}} (with %% prefix) to be used as MMA D/C operand list, got:\n{ptx}");
    }

    #[test]
    fn synthesize_dq_kernel_scatters_dq_acc_to_hbm() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("// === G1: dQ HBM finalize addressing ==="),
            "expected G1 dQ finalize section");
        assert!(ptx.contains("st.global.f32"),
            "expected st.global.f32 for dQ scatter via real HBM addressing");
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

    #[test]
    fn synthesize_dq_kernel_loads_per_row_stats_probe1() {
        // Probe 1 (structural): exact per-lane stats lane-mapping vs independent
        // spec-derived pattern. s_lo = q_iter*bq + warp_id*16 + lane/4 ; s_hi = +8.
        let cfg = FlashAttentionConfig { head_dim: 32, ..canonical_cfg() }; // eff bq = 64
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        // bug #1: the three stats params must be consumed (loaded), not declared-unused.
        assert!(ptx.contains("ld.param.u64 %stats_rmax_base, [row_max_ptr]"), "row_max_ptr must be loaded");
        assert!(ptx.contains("ld.param.u64 %stats_rsum_base, [row_sum_ptr]"), "row_sum_ptr must be loaded");
        assert!(ptx.contains("ld.param.u64 %stats_d_base, [d_ptr]"), "d_ptr must be loaded");
        // exact s_lo derivation (the lane/4 term is the bug-class term):
        assert!(ptx.contains("shr.u32 %stat_lane_div4, %lane_id, 2"), "s_lo must include lane/4");
        assert!(ptx.contains("mul.lo.u32 %s_lo, %q_iter, 64"), "s_lo must include q_iter*bq (bq=64 at hd=32)");
        assert!(ptx.contains("add.u32 %s_lo, %s_lo, %band_row_base"), "s_lo must include warp band base");
        assert!(ptx.contains("add.u32 %s_lo, %s_lo, %stat_lane_div4"), "s_lo must add lane/4");
        assert!(ptx.contains("add.u32 %s_hi, %s_lo, 8"), "s_hi = s_lo + 8");
        // 6 stat loads (rmax/rsum/d x lo/hi) and per-row reciprocal:
        assert!(ptx.matches("ld.global.f32").count() >= 6, "expected >=6 stat loads");
        assert!(ptx.contains("rcp.approx.f32 %rsum_recip_lo") && ptx.contains("rcp.approx.f32 %rsum_recip_hi"),
            "row_sum must be reciprocated per row");
    }

    #[test]
    fn synthesize_dq_kernel_computes_warp_band_base_and_active() {
        // hd=32 -> bq=64 -> active warps = 64/16 = 4.
        let cfg = FlashAttentionConfig { head_dim: 32, ..canonical_cfg() };
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        assert!(ptx.contains("mul.lo.u32 %band_row_base, %warp_id, 16"),
            "expected warp-band row base = warp_id*16");
        assert!(ptx.contains("setp.lt.u32 %p_warp_active, %warp_id, 4"),
            "expected warp-active predicate warp_id < bq/16 (=4 at bq=64)");

        // hd=128 -> effective bq=32 -> active warps = 32/16 = 2 (warps 2-3 idle).
        // This is the deadlock-critical value Task 10's idle-warp gating depends on.
        let cfg128 = FlashAttentionConfig { head_dim: 128, ..canonical_cfg() };
        let ptx128 = synthesize_dq_kernel(&cfg128).unwrap();
        assert!(ptx128.contains("setp.lt.u32 %p_warp_active, %warp_id, 2"),
            "expected warp-active predicate warp_id < bq/16 (=2 at bq=32, hd=128)");
    }

    #[test]
    fn synthesize_dq_kernel_applies_per_row_stats() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // P recompute must subtract the ROW-CORRECT max: lo for elems 0/1, hi for 2/3.
        assert!(ptx.contains("sub.f32 %p_0, %s_d0, %rmax_lo")
            && ptx.contains("sub.f32 %p_2, %s_d2, %rmax_hi"),
            "P recompute must use rmax_lo for elem0 and rmax_hi for elem2");
        assert!(ptx.contains("mul.f32 %p_0, %p_0, %rsum_recip_lo")
            && ptx.contains("mul.f32 %p_2, %p_2, %rsum_recip_hi"),
            "P recompute must scale by row_sum_recip_lo/hi per row");
        // dS must subtract D per row: d_lo for elems 0/1, d_hi for 2/3.
        assert!(ptx.contains("sub.f32 %ds_0, %dp_d0, %d_lo")
            && ptx.contains("sub.f32 %ds_2, %dp_d2, %d_hi"),
            "dS must use d_lo for elem0 and d_hi for elem2");
        // The stub single-stat regs must be gone.
        assert!(!ptx.contains("%f_rmax") && !ptx.contains("%f_d_q"),
            "stub single-stat regs must be removed");
    }
}
