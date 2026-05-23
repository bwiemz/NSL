//! QK^T + softmax + PV MMA emitter (Phase B of the main FSM).
//!
//! Per spec section 4.2 main loop, Phase B per kv_iter:
//!   1. QK^T MMA: Q tile (SMEM, from prior projection) @ K^T tile (SMEM, slot[curr]).
//!      Produces S fragments in per-warp registers.
//!   2. Online softmax: row-max -> exp(S - max) -> row-sum, with running
//!      correction across kv_iters. Reduction across lanes via shfl.sync.bfly.
//!   3. PV MMA: P (packed f16, registers) @ V tile (SMEM, slot[curr]).
//!      Accumulates into O_acc registers -- register-resident per spec section 3.1.
//!
//! ## B1.5 Task 5.2 scope
//!
//! This file ships the STRUCTURAL scaffold for Phase B. Each sub-helper
//! emits a minimal placeholder body sufficient for ptxas to validate the
//! kernel and for snapshot tests to lock the FSM shape. Numerical
//! correctness is intentionally deferred -- see the per-sub-helper rustdoc
//! for the specific B1.6 swap targets.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    tier_b1_k_offset_ping, tier_b1_k_offset_pong, tier_b1_n_tiles_kv, tier_b1_p_offset,
    tier_b1_q_offset, tier_b1_reduced_stats_offset, tier_b1_reduced_stats_sum_offset,
    tier_b1_softmax_scratch_bytes, tier_b1_softmax_scratch_offset, tier_b1_v_offset_ping,
    tier_b1_v_offset_pong,
};
use crate::matmul_mma::{
    emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
};

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the QK^T
/// output (bq x bkv). Always at least 1.
pub(crate) fn tiles_per_warp_qkt(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let m_tiles = bq / 16;
    let n_tiles = bkv / 8;
    let total = m_tiles.max(1) * n_tiles.max(1);
    (total / 8).max(1)
}

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the PV
/// output (bq x hd). Identical to Q-projection's tile count.
pub(crate) fn tiles_per_warp_pv(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let hd = config.head_dim as u32;
    let m_tiles = bq / 16;
    let n_tiles = hd / 8;
    let total = m_tiles.max(1) * n_tiles.max(1);
    (total / 8).max(1)
}

/// Phase B: QK^T MMA -> online softmax -> PV MMA. Per spec section 5.3
/// (tile distribution) + section 4.2 main loop Phase B.
///
/// Reads Q tile (SMEM, already projected) and slot[curr]'s K and V tiles
/// (SMEM, written by Phase A). Accumulates into per-warp O_acc registers
/// (NOT SMEM, per spec section 3.1's load-bearing register-residency
/// decision).
///
/// # B1.5 scaffold scope
///
/// The structural FSM shape -- header comment, QK^T MMA block, softmax
/// block, PV MMA block -- is emitted. Each of the three phases has FOUR
/// classes of placeholder identical to those in `projection_mma.rs`:
///
/// 1. Fragment loads use uniform SMEM addresses (B1.6 swap to
///    `matmul_mma::emit_load_{a,b}_fragment_smem`).
/// 2. Warp-ownership predicate (`t %% 8 == warp_id`) not emitted (B1.6
///    add `setp` + `@%pred` gates).
/// 3. Softmax row-reductions are placeholder lane-collective shfl ops
///    rather than real bfly trees (B1.6 implement spec section 5.3's
///    intra-warp reduction pattern).
/// 4. **All `.reg` declarations inside this file's sub-helpers will collide
///    if `emit_phase_b_attention` is called more than once on the same PTX
///    string.** ptxas rejects duplicate `.reg` declarations. The set
///    includes `%tb1_phase_b_smem_q/k/v`, `%s_acc_<t>_<lane>`,
///    `%p_packed_<t>`, `%o_acc_<t>_<lane>`, and all `%tb1_qkt_a/b` /
///    `%tb1_pv_a/b` fragment regs. Until B1.6 hoists these, the Task 5.3
///    orchestrator MUST call this helper exactly once (single-iteration
///    scaffold). The multi-`kv_iter` loop the plan template draws is
///    deferred to B1.6 alongside the register-hoist refactor.
///
///    Additionally, O_acc semantically must persist across kv_iters per
///    spec section 3.1 (register-resident O_acc); the same hoist that
///    fixes the collision also satisfies the lifetime requirement.
pub fn emit_phase_b_attention(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    kv_iter: u32,
    slot: u32,
) {
    ptx.push_str(&format!(
        "    // === Phase B: attention compute on slot[{}] (kv_iter={}) ===\n",
        slot, kv_iter
    ));
    emit_qkt_mma(ptx, config, slot);
    emit_online_softmax(ptx, config, kv_iter);
    emit_scatter_p_to_smem(ptx, config);
    emit_pv_mma(ptx, config, slot);
}

/// Scatter the D-fragment-shaped softmax output `%s_acc_<t>_<i>` to SMEM
/// at `tier_b1_p_offset` so the PV MMA can load it as an A-fragment via
/// `matmul_mma::emit_load_a_fragment_smem`. Per spec section 3.5 the
/// PV A-fragment k=16 span crosses two consecutive QK^T n-tiles, which
/// can't be sourced from a single D-fragment in-register — the SMEM
/// round-trip is the standard FA-2 bridge.
///
/// Issues a `bar.sync 0` at the end to ensure cross-warp visibility of
/// the scattered P data before the PV MMA reads from `P_smem`.
///
/// **N3 resolution:** each warp owns slot `local_t` and maps it to
/// `global_t = warp_id + local_t * 8` (runtime), then derives a
/// distinct `(m_tile, n_tile)` per warp. No warp gate is needed; each
/// warp writes its own slice of P_smem to a DISJOINT region. Requires
/// `block_kv / 8` to be a power of 2 (asserted at codegen).
fn emit_scatter_p_to_smem(ptx: &mut String, config: &FlashAttentionConfig) {
    let tpw = tiles_per_warp_qkt(config);
    let bkv = config.block_kv as u32;
    let n_tiles_kv = (bkv / 8).max(1);
    let p_off = tier_b1_p_offset(config);
    assert!(
        n_tiles_kv.is_power_of_two(),
        "N3: P scatter requires block_kv/8 to be a power of 2; got {}",
        n_tiles_kv
    );
    let log2_n_tiles_kv = n_tiles_kv.trailing_zeros();
    let n_tiles_kv_mask = n_tiles_kv - 1;
    // m_tile stride within P_smem (each tile spans 16 rows of bkv-wide).
    let m_tile_stride_bytes = 16 * bkv * 2;
    assert!(
        m_tile_stride_bytes.is_power_of_two(),
        "N3: P scatter requires 16 * block_kv * 2 to be a power of 2; got {}",
        m_tile_stride_bytes
    );
    let log2_m_tile_stride = m_tile_stride_bytes.trailing_zeros();

    ptx.push_str("    // === P scatter to SMEM (PV A-fragment bridge + N3) ===\n");
    ptx.push_str("    .reg .u32 %sp_laneid, %sp_lo_row, %sp_lo_col_base;\n");
    ptx.push_str("    .reg .u32 %sp_row_off, %sp_col_off, %sp_lane_off;\n");
    ptx.push_str("    .reg .u32 %sp_addr_u32, %sp_addr_i;\n");
    ptx.push_str("    .reg .u32 %sp_global_t, %sp_m_tile, %sp_n_tile;\n");
    ptx.push_str("    .reg .u32 %sp_tile_off, %sp_m_off, %sp_n_off;\n");
    ptx.push_str("    .reg .u64 %sp_addr_u64;\n");
    ptx.push_str("    .reg .b16 %sp_h;\n");

    // Per-lane addressing setup (constant across all tiles for a lane).
    ptx.push_str("    mov.u32 %sp_laneid, %tid.x;\n");
    ptx.push_str("    and.b32 %sp_laneid, %sp_laneid, 31;\n");
    ptx.push_str("    shr.u32 %sp_lo_row, %sp_laneid, 2;       // l/4 = D-fragment lo-row\n");
    ptx.push_str("    and.b32 %sp_lo_col_base, %sp_laneid, 3;  // l%4\n");
    ptx.push_str("    shl.b32 %sp_lo_col_base, %sp_lo_col_base, 1; // (l%4)*2 = col base\n");
    ptx.push_str(&format!(
        "    mul.lo.u32 %sp_row_off, %sp_lo_row, {};  // row * block_kv * 2 bytes\n",
        bkv * 2
    ));
    ptx.push_str("    shl.b32 %sp_col_off, %sp_lo_col_base, 1;     // col * 2 bytes (f16)\n");
    ptx.push_str("    add.u32 %sp_lane_off, %sp_row_off, %sp_col_off;\n");

    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // P scatter slot local_t={} (global_t = warp_id + {}*8 at runtime; N3)\n",
            t,
            t
        ));

        // global_t = warp_id + local_t * 8
        if t == 0 {
            ptx.push_str("    mov.u32 %sp_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %sp_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        // (m_tile, n_tile) from global_t.
        ptx.push_str(&format!(
            "    shr.u32 %sp_m_tile, %sp_global_t, {};\n",
            log2_n_tiles_kv
        ));
        ptx.push_str(&format!(
            "    and.b32 %sp_n_tile, %sp_global_t, {};\n",
            n_tiles_kv_mask
        ));
        // m_tile * 16 * block_kv * 2 (shl since power of 2).
        ptx.push_str(&format!(
            "    shl.b32 %sp_m_off, %sp_m_tile, {};\n",
            log2_m_tile_stride
        ));
        // n_tile * 8 * 2 = n_tile * 16 bytes (shl 4).
        ptx.push_str("    shl.b32 %sp_n_off, %sp_n_tile, 4;\n");
        ptx.push_str("    add.u32 %sp_tile_off, %sp_m_off, %sp_n_off;\n");

        // P_smem base + p_offset + tile_off.
        ptx.push_str(&format!(
            "    add.u64 %sp_addr_u64, %shmem_base, {};\n",
            p_off
        ));
        ptx.push_str("    cvt.u32.u64 %sp_addr_u32, %sp_addr_u64;\n");
        ptx.push_str("    add.u32 %sp_addr_u32, %sp_addr_u32, %sp_tile_off;\n");
        ptx.push_str("    add.u32 %sp_addr_u32, %sp_addr_u32, %sp_lane_off;\n");

        // 4 D-fragment positions per lane:
        //   i=0: row=lo_row,   col=lo_col_base   → +0
        //   i=1: row=lo_row,   col=lo_col_base+1 → +2 bytes
        //   i=2: row=lo_row+8, col=lo_col_base   → +8*bkv*2 bytes
        //   i=3: row=lo_row+8, col=lo_col_base+1 → +8*bkv*2 + 2 bytes
        for i in 0..4u32 {
            let off_bytes: u32 = match i {
                0 => 0,
                1 => 2,
                2 => 8 * bkv * 2,
                3 => 8 * bkv * 2 + 2,
                _ => unreachable!(),
            };
            if off_bytes > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %sp_addr_i, %sp_addr_u32, {};\n",
                    off_bytes
                ));
            } else {
                ptx.push_str("    mov.u32 %sp_addr_i, %sp_addr_u32;\n");
            }
            ptx.push_str(&format!(
                "    cvt.rn.f16.f32 %sp_h, %s_acc_{}_{};\n",
                t, i
            ));
            // N3: no warp gate — each warp writes to a distinct P_smem
            // region for its own (m_tile, n_tile).
            ptx.push_str("    st.shared.b16 [%sp_addr_i], %sp_h;\n");
        }
    }

    // CTA-wide sync: ensure all warps see all scattered P values before
    // any warp reads P_smem in the PV MMA.
    ptx.push_str("    bar.sync 0;\n");
}

/// QK^T: m16n8k16 MMA against Q (SMEM, tier_b1_q_offset) and K (SMEM,
/// tier_b1_k_offset_<ping|pong> selected by slot). Output S-fragments
/// accumulate into per-warp `%s_acc_<t>_<lane>` registers.
///
/// **B1.6 TODO (deferral #1):** fragment loads use uniform SMEM address;
/// swap to `matmul_mma::emit_load_a_fragment_smem` (Q) +
/// `emit_load_b_fragment_smem` (K^T) for real per-thread addressing.
/// Also requires importing `smem_layout::tier_b1_q_offset` and
/// `tier_b1_k_offset_{ping,pong}` to compute the per-tile base address
/// — the placeholder currently uses `mov.u64 ..., 0` which would yield
/// reads from offset 0 even after the fragment-load swap.
///
/// **B1.6 TODO (deferral #2):** warp-ownership predicate not yet emitted.
/// All 8 warps execute every MMA; gate each on `setp.eq.u32` + `@%pred`.
///
/// **B1.6 deferral #5 RESOLVED:** the QK^T tail applies the
/// `1/sqrt(head_dim)` scaling via per-lane `mul.f32` immediately after
/// the MMA loop — semantically equivalent to applying the scale to the
/// softmax input. The scale constant is computed at codegen time from
/// `config.head_dim` and emitted as the IEEE-754 f32 hex literal.
fn emit_qkt_mma(ptx: &mut String, config: &FlashAttentionConfig, slot: u32) {
    let tpw = tiles_per_warp_qkt(config);
    let hd = config.head_dim as u32;
    let bkv = config.block_kv as u32;
    let n_tiles_bkv = (bkv / 8).max(1);
    assert!(
        n_tiles_bkv.is_power_of_two(),
        "N3: QK^T requires block_kv/8 to be a power of 2; got {}",
        n_tiles_bkv
    );
    let log2_n_tiles_bkv = n_tiles_bkv.trailing_zeros();
    let n_tiles_bkv_mask = n_tiles_bkv - 1;
    // m_tile stride in Q SMEM = 16 rows * hd cols * 2 bytes.
    let q_m_stride_bytes = 16 * hd * 2;
    // n_tile stride in K SMEM = 8 K-rows (== 8 bkv positions) * hd cols * 2 bytes.
    let k_n_stride_bytes = 8 * hd * 2;
    assert!(
        q_m_stride_bytes.is_power_of_two(),
        "N3: QK^T requires 16*hd*2 to be a power of 2; got {}",
        q_m_stride_bytes
    );
    assert!(
        k_n_stride_bytes.is_power_of_two(),
        "N3: QK^T requires 8*hd*2 to be a power of 2; got {}",
        k_n_stride_bytes
    );
    let log2_q_m_stride = q_m_stride_bytes.trailing_zeros();
    let log2_k_n_stride = k_n_stride_bytes.trailing_zeros();
    // Codegen-time computation: 1 / sqrt(head_dim) as f32 IEEE-754 bits.
    // Spec section 4.2: scale QK^T output by 1/sqrt(d_k) before softmax.
    let scale_bits = (1.0_f32 / (hd as f32).sqrt()).to_bits();
    ptx.push_str(&format!(
        "    // QK^T MMA: bq={} bkv={} tpw_qkt={} slot={} (scale=1/sqrt({})=0f{:08X}) (N3)\n",
        config.block_q, config.block_kv, tpw, slot, hd, scale_bits
    ));
    // %tb1_phase_b_smem_q/k and %s_acc_<t>_<lane> are declared + zero-init'd
    // by register_budget::declare_registers at orchestrator scope (B1.6
    // deferral #4 resolution).
    let q_off = tier_b1_q_offset(config);
    let k_off = if slot == 0 {
        tier_b1_k_offset_ping(config)
    } else {
        tier_b1_k_offset_pong(config)
    };
    ptx.push_str(&format!(
        "    add.u64 %tb1_phase_b_smem_q, %shmem_base, {}; // Q tile @ tier_b1_q_offset\n",
        q_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_phase_b_smem_k, %shmem_base, {}; // K tile @ slot={} offset\n",
        k_off, slot
    ));
    // u32 sister regs for the matmul_mma fragment-load helpers.
    ptx.push_str("    .reg .u32 %tb1_phase_b_smem_q_u32, %tb1_phase_b_smem_k_u32;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_phase_b_smem_q_u32, %tb1_phase_b_smem_q;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_phase_b_smem_k_u32, %tb1_phase_b_smem_k;\n");
    // N3 runtime tile coords: each warp's slot t maps to a DISTINCT
    // (m_tile, n_tile) of the bq × bkv output via
    // global_t = warp_id + local_t*8.
    ptx.push_str("    .reg .u32 %qkt_global_t, %qkt_m_tile, %qkt_n_tile;\n");
    ptx.push_str("    .reg .u32 %qkt_q_warp_off, %qkt_k_warp_off;\n");
    ptx.push_str("    .reg .u32 %qkt_a_base, %qkt_b_base;\n");
    // N1b K-iter shifted bases (for head_dim > 16).
    ptx.push_str("    .reg .u32 %qkt_a_base_k, %qkt_b_base_k;\n");

    // N1b: QK^T K-loop count = head_dim / 16.
    assert!(
        hd.is_multiple_of(16),
        "N1b: QK^T K-loop requires head_dim divisible by 16; got {}",
        hd
    );
    let n_k_iters_qkt = hd / 16;

    for t in 0..tpw {
        // N3: compute warp-specific (m_tile, n_tile) at runtime.
        if t == 0 {
            ptx.push_str("    mov.u32 %qkt_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %qkt_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        ptx.push_str(&format!(
            "    shr.u32 %qkt_m_tile, %qkt_global_t, {};\n",
            log2_n_tiles_bkv
        ));
        ptx.push_str(&format!(
            "    and.b32 %qkt_n_tile, %qkt_global_t, {};\n",
            n_tiles_bkv_mask
        ));
        // q_warp_off = m_tile * 16 * hd * 2 (Q tile m-stride)
        ptx.push_str(&format!(
            "    shl.b32 %qkt_q_warp_off, %qkt_m_tile, {};\n",
            log2_q_m_stride
        ));
        ptx.push_str("    add.u32 %qkt_a_base, %tb1_phase_b_smem_q_u32, %qkt_q_warp_off;\n");
        // k_warp_off = n_tile * 8 * hd * 2 (K tile n-stride: 8 K-rows × hd × 2 bytes)
        ptx.push_str(&format!(
            "    shl.b32 %qkt_k_warp_off, %qkt_n_tile, {};\n",
            log2_k_n_stride
        ));
        ptx.push_str("    add.u32 %qkt_b_base, %tb1_phase_b_smem_k_u32, %qkt_k_warp_off;\n");

        // B1.6 deferral #1 resolution: per-lane fragment loads via
        // matmul_mma helpers.
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_qkt_a_{}_0, %tb1_qkt_a_{}_1, %tb1_qkt_a_{}_2, %tb1_qkt_a_{}_3;\n",
            t, t, t, t
        ));
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_qkt_b_{}_0, %tb1_qkt_b_{}_1;\n",
            t, t
        ));

        // N1b: QK^T K-loop. m16n8k16 K-dim = 16; for head_dim > 16 we
        // need head_dim/16 K-iters, accumulating into the same
        // %s_acc_<t>_*. Both A (Q) and B (K) shift by k_iter * 32 bytes
        // along the k=hd dim (16 cols × 2 bytes per col).
        for k_iter in 0..n_k_iters_qkt {
            let a_base_expr: String = if k_iter == 0 {
                "%qkt_a_base".to_string()
            } else {
                let off = k_iter * 32;
                ptx.push_str(&format!(
                    "    add.u32 %qkt_a_base_k, %qkt_a_base, {}; // QK^T A k_iter={}\n",
                    off, k_iter
                ));
                "%qkt_a_base_k".to_string()
            };
            let b_base_expr: String = if k_iter == 0 {
                "%qkt_b_base".to_string()
            } else {
                let off = k_iter * 32;
                ptx.push_str(&format!(
                    "    add.u32 %qkt_b_base_k, %qkt_b_base, {}; // QK^T B k_iter={}\n",
                    off, k_iter
                ));
                "%qkt_b_base_k".to_string()
            };

            let a_fragment_regs = [
                format!("tb1_qkt_a_{}_0", t),
                format!("tb1_qkt_a_{}_1", t),
                format!("tb1_qkt_a_{}_2", t),
                format!("tb1_qkt_a_{}_3", t),
            ];
            emit_load_a_fragment_smem(
                ptx,
                &a_fragment_regs,
                &a_base_expr,
                (hd * 2) as usize,
            );
            let b_fragment_regs = [
                format!("tb1_qkt_b_{}_0", t),
                format!("tb1_qkt_b_{}_1", t),
            ];
            emit_load_b_fragment_smem(
                ptx,
                &b_fragment_regs,
                &b_base_expr,
                (hd * 2) as usize,
            );
            let d_regs = [
                format!("%s_acc_{}_0", t),
                format!("%s_acc_{}_1", t),
                format!("%s_acc_{}_2", t),
                format!("%s_acc_{}_3", t),
            ];
            let a_regs = [
                format!("%tb1_qkt_a_{}_0", t),
                format!("%tb1_qkt_a_{}_1", t),
                format!("%tb1_qkt_a_{}_2", t),
                format!("%tb1_qkt_a_{}_3", t),
            ];
            let b_regs = [
                format!("%tb1_qkt_b_{}_0", t),
                format!("%tb1_qkt_b_{}_1", t),
            ];
            let c_regs = d_regs.clone();
            // N1b: accumulate into same %s_acc_<t>_* across K-iters.
            // N3: unconditional MMA — each warp writes its OWN slot.
            emit_mma_instruction(ptx, &d_regs, &a_regs, &b_regs, &c_regs);
        }
    }

    // B1.6 deferral #5 resolution: scale S accumulators by 1/sqrt(head_dim).
    // Per spec section 4.2 the scale belongs between QK^T and softmax. We
    // apply it to the f32 S accumulators here so the downstream softmax
    // input is already pre-scaled. Mathematically equivalent to scaling
    // inside exp(); architecturally simpler because the constant is known
    // at codegen time and the loop already has the accumulator regs hot.
    ptx.push_str(&format!(
        "    // Apply 1/sqrt(head_dim={}) scale to S accumulators (deferral #5)\n",
        hd
    ));
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!(
                "    mul.f32 %s_acc_{}_{}, %s_acc_{}_{}, 0f{:08X};\n",
                t, lane, t, lane, scale_bits
            ));
        }
    }

    // Causal mask: set S[q, k] := -INF where k > q.
    //
    // Each lane holds 4 D-fragment positions per tile per spec §5.5:
    //   D0 = S[m_tile*16 + lo_row,     n_tile*8 + lo_col_base + 0]
    //   D1 = S[m_tile*16 + lo_row,     n_tile*8 + lo_col_base + 1]
    //   D2 = S[m_tile*16 + lo_row + 8, n_tile*8 + lo_col_base + 0]
    //   D3 = S[m_tile*16 + lo_row + 8, n_tile*8 + lo_col_base + 1]
    //
    // The position-in-tile (lo_row, lo_col_base) is derived from %laneid;
    // (m_tile, n_tile) was computed per-t above via warp-distribution.
    // q_global = %q_start + m_tile*16 + row_in_tile; k_global = kv_iter *
    // block_kv + n_tile*8 + col_in_tile (the kv_iter offset is folded in
    // via %kv_start_runtime if the caller writes it; for the single-iter
    // scaffold kv_iter=0 so the offset is 0).
    if config.causal {
        ptx.push_str("    // Causal mask: -INF where k_global > q_global\n");
        ptx.push_str("    .reg .u32 %cm_laneid, %cm_lo_row, %cm_lo_col_base;\n");
        ptx.push_str("    .reg .u32 %cm_q_global, %cm_k_global;\n");
        ptx.push_str("    .reg .pred %cm_mask;\n");
        ptx.push_str("    mov.u32 %cm_laneid, %tid.x;\n");
        ptx.push_str("    and.b32 %cm_laneid, %cm_laneid, 31;\n");
        ptx.push_str("    shr.u32 %cm_lo_row, %cm_laneid, 2;       // lo_row = lane/4\n");
        ptx.push_str("    and.b32 %cm_lo_col_base, %cm_laneid, 3;  // lane%4\n");
        ptx.push_str("    shl.b32 %cm_lo_col_base, %cm_lo_col_base, 1; // *2\n");
        // -inf as f32 IEEE 754 = 0xFF800000. Use a large negative value
        // that survives the subsequent (S - max) - softmax_scale path
        // without underflowing to -inf prematurely (we don't want NaN
        // from inf-inf). 0xC2C80000 ≈ -100 is conservative for all
        // post-scale S magnitudes the kernel produces.
        let mask_neg_bits: u32 = 0xC2C80000; // -100.0f
        for t in 0..tpw {
            // Recompute (m_tile, n_tile) for this tile slot — same formula
            // as the QK^T MMA above. Reuse %qkt_global_t/m_tile/n_tile
            // (declared above; lives in scope here).
            if t == 0 {
                ptx.push_str("    mov.u32 %qkt_global_t, %warp_id;\n");
            } else {
                ptx.push_str(&format!(
                    "    add.u32 %qkt_global_t, %warp_id, {};\n",
                    t * 8
                ));
            }
            ptx.push_str(&format!(
                "    shr.u32 %qkt_m_tile, %qkt_global_t, {};\n",
                log2_n_tiles_bkv
            ));
            ptx.push_str(&format!(
                "    and.b32 %qkt_n_tile, %qkt_global_t, {};\n",
                n_tiles_bkv_mask
            ));
            // m_tile*16 and n_tile*8 — reuse %qkt_q_warp_off/%qkt_k_warp_off
            // as scratch (we no longer need them after the MMA).
            ptx.push_str("    shl.b32 %qkt_q_warp_off, %qkt_m_tile, 4;\n");
            ptx.push_str("    shl.b32 %qkt_k_warp_off, %qkt_n_tile, 3;\n");

            for d_idx in 0..4u32 {
                let row_offset = if d_idx < 2 { 0u32 } else { 8 };
                let col_offset = d_idx % 2;
                // q_global = q_start_low_u32 + m_tile*16 + lo_row + row_offset
                ptx.push_str(
                    "    add.u32 %cm_q_global, %qkt_q_warp_off, %cm_lo_row;\n",
                );
                if row_offset > 0 {
                    ptx.push_str(&format!(
                        "    add.u32 %cm_q_global, %cm_q_global, {};\n",
                        row_offset
                    ));
                }
                ptx.push_str(
                    "    cvt.u32.u64 %cm_k_global, %q_start;  // q_start (low 32 bits)\n",
                );
                ptx.push_str("    add.u32 %cm_q_global, %cm_q_global, %cm_k_global;\n");
                // k_global = n_tile*8 + lo_col_base + col_offset
                // (single-iter kv_iter=0; kv_start = 0).
                ptx.push_str(
                    "    add.u32 %cm_k_global, %qkt_k_warp_off, %cm_lo_col_base;\n",
                );
                if col_offset > 0 {
                    ptx.push_str(&format!(
                        "    add.u32 %cm_k_global, %cm_k_global, {};\n",
                        col_offset
                    ));
                }
                // mask if k_global > q_global → set -INF-substitute.
                ptx.push_str("    setp.gt.u32 %cm_mask, %cm_k_global, %cm_q_global;\n");
                ptx.push_str(&format!(
                    "    @%cm_mask mov.f32 %s_acc_{}_{}, 0f{:08X};\n",
                    t, d_idx, mask_neg_bits
                ));
            }
        }
    }
}

/// Online softmax: row-max -> exp(S - max) -> row-sum + running
/// correction across kv_iters. Reduction across lanes via
/// `shfl.sync.bfly.b32`. Output: P fragments packed f16 in
/// `%p_packed_<t>` registers (one b32 per tile, holding 2 f16 lanes).
///
/// **B1.6 deferral #6 PARTIAL:** the row-max bfly reduction tree is real
/// (2 steps, mask=1 then mask=2 across the 4-lane row group per spec
/// §5.5 lane mapping). For each tile, each lane folds its 2 cols held
/// for `row_lo` into a local max, then bfly-reduces across the 4-lane
/// group sharing that row. Same for `row_lo + 8`. After the tree, every
/// lane in a row group has the per-row max.
///
/// **Still deferred** (full softmax pipeline): subtract row_max, exp
/// (PTX `ex2.approx.f32` after `mul.f32` by `log2(e)`), row_sum
/// reduction (same 2-step bfly), divide. P-packed emission stays a
/// `mov` placeholder until that pipeline lands.
///
/// **B1.6 TODO (deferral #7):** running max + sum across kv_iters needs
/// state in `%m_acc_<row>` and `%l_acc_<row>` registers declared at
/// kernel-prologue scope. The scaffold here re-zeros them per call,
/// which is correct for kv_iter=0 only.
fn emit_online_softmax(ptx: &mut String, config: &FlashAttentionConfig, kv_iter: u32) {
    let tpw = tiles_per_warp_qkt(config);
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let n_tiles_kv = tier_b1_n_tiles_kv(config);
    let softmax_off = tier_b1_softmax_scratch_offset(config);
    let scratch_max_size = bq * n_tiles_kv * 4; // first half of scratch holds maxes
    // The sum region lives at +scratch_max_size from the softmax base; that
    // offset is added inline at the SMEM addressing site below.
    // sanity: total reserved must match smem_layout helper
    assert_eq!(
        scratch_max_size + scratch_max_size,
        tier_b1_softmax_scratch_bytes(config),
        "softmax scratch byte budget mismatch"
    );

    ptx.push_str(&format!(
        "    // Online softmax (cross-warp combine): bq={} bkv={} n_tiles_kv={} tpw_qkt={} kv_iter={}\n",
        bq, bkv, n_tiles_kv, tpw, kv_iter
    ));

    // ---- shared scratch register prelude (per-call but cheap; ptxas SSA
    // collapses repeats across multi-iter when those are added). ----
    ptx.push_str("    .reg .u32 %sm_laneid, %sm_lane_mod4, %sm_lo_row;\n");
    ptx.push_str("    .reg .pred %sm_writer_pred;\n");
    ptx.push_str("    .reg .u32 %sm_scratch_max_base, %sm_scratch_sum_base;\n");
    ptx.push_str("    .reg .u32 %sm_addr, %sm_addr2, %sm_global_row;\n");
    ptx.push_str("    .reg .f32 %sm_global_max_lo, %sm_global_max_hi;\n");
    ptx.push_str("    .reg .f32 %sm_global_sum_lo, %sm_global_sum_hi;\n");
    ptx.push_str("    .reg .f32 %sm_part_max, %sm_part_sum;\n");
    // Convert SMEM scratch base from %shmem_base (u64) to u32 once.
    ptx.push_str(&format!(
        "    add.u64 %tb1_phase_b_smem_q, %shmem_base, {}; // softmax scratch base (reusing %tb1_phase_b_smem_q as u64 scratch)\n",
        softmax_off
    ));
    ptx.push_str("    cvt.u32.u64 %sm_scratch_max_base, %tb1_phase_b_smem_q;\n");
    ptx.push_str(&format!(
        "    add.u32 %sm_scratch_sum_base, %sm_scratch_max_base, {};\n",
        scratch_max_size
    ));
    ptx.push_str("    mov.u32 %sm_laneid, %tid.x;\n");
    ptx.push_str("    and.b32 %sm_laneid, %sm_laneid, 31;\n");
    ptx.push_str("    shr.u32 %sm_lo_row, %sm_laneid, 2;       // lo_row = lane/4\n");
    ptx.push_str("    and.b32 %sm_lane_mod4, %sm_laneid, 3;    // lane%4\n");
    // Writer-lane gate: only lane%4==0 writes (4 lanes per row group share
    // the same per-warp partial after bfly; one of them does the SMEM store).
    ptx.push_str("    setp.eq.u32 %sm_writer_pred, %sm_lane_mod4, 0;\n");

    // ============================================================
    // STEP 1: per-warp partial row-max + cross-warp combine to global_max.
    // ============================================================
    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // === softmax tile t={} : step 1 = per-warp partial row-max ===\n",
            t
        ));
        ptx.push_str(&format!(
            "    .reg .f32 %s_max_{}_lo, %s_max_{}_hi, %s_shfl_{}_tmp;\n",
            t, t, t
        ));
        // Intra-thread max across the 2 cols this lane holds for row_lo
        // (frag indices 0,1) and row_lo+8 (frag indices 2,3).
        ptx.push_str(&format!(
            "    max.f32 %s_max_{}_lo, %s_acc_{}_0, %s_acc_{}_1;\n",
            t, t, t
        ));
        ptx.push_str(&format!(
            "    max.f32 %s_max_{}_hi, %s_acc_{}_2, %s_acc_{}_3;\n",
            t, t, t
        ));
        // Bfly within 4-lane row group → partial max for warp's 8-col slice.
        for (mask, comment) in [(1u32, "lane k <-> k^1"), (2u32, "lane k <-> k^2")].iter() {
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %s_shfl_{}_tmp, %s_max_{}_lo, {}, 0x1f, 0xffffffff; // {}\n",
                t, t, mask, comment
            ));
            ptx.push_str(&format!(
                "    max.f32 %s_max_{}_lo, %s_max_{}_lo, %s_shfl_{}_tmp;\n",
                t, t, t
            ));
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %s_shfl_{}_tmp, %s_max_{}_hi, {}, 0x1f, 0xffffffff; // {}\n",
                t, t, mask, comment
            ));
            ptx.push_str(&format!(
                "    max.f32 %s_max_{}_hi, %s_max_{}_hi, %s_shfl_{}_tmp;\n",
                t, t, t
            ));
        }

        // Writer-lane stores partial max for (m_tile*16 + lo_row, n_tile)
        // and (m_tile*16 + lo_row + 8, n_tile). Recompute (m_tile, n_tile)
        // from %warp_id and t (mirrors the QK^T MMA distribution).
        if t == 0 {
            ptx.push_str("    mov.u32 %qkt_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %qkt_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        // m_tile = global_t / n_tiles_kv ; n_tile = global_t & (n_tiles_kv-1)
        let log2_n_tiles_kv = n_tiles_kv.trailing_zeros();
        let n_tiles_kv_mask = n_tiles_kv - 1;
        ptx.push_str(&format!(
            "    shr.u32 %qkt_m_tile, %qkt_global_t, {};\n",
            log2_n_tiles_kv
        ));
        ptx.push_str(&format!(
            "    and.b32 %qkt_n_tile, %qkt_global_t, {};\n",
            n_tiles_kv_mask
        ));
        // global_row_lo = m_tile*16 + lo_row.
        ptx.push_str("    shl.b32 %qkt_q_warp_off, %qkt_m_tile, 4;\n");
        ptx.push_str("    add.u32 %sm_global_row, %qkt_q_warp_off, %sm_lo_row;\n");
        // addr = scratch_max_base + (global_row * n_tiles_kv + n_tile) * 4.
        ptx.push_str(&format!(
            "    mul.lo.u32 %sm_addr, %sm_global_row, {};\n",
            n_tiles_kv * 4
        ));
        ptx.push_str("    shl.b32 %sm_addr2, %qkt_n_tile, 2; // n_tile * 4 bytes\n");
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_addr2;\n");
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_scratch_max_base;\n");
        ptx.push_str(&format!(
            "    @%sm_writer_pred st.shared.f32 [%sm_addr], %s_max_{}_lo;\n",
            t
        ));
        // Now the HI row: global_row + 8.
        ptx.push_str("    add.u32 %sm_global_row, %sm_global_row, 8;\n");
        ptx.push_str(&format!(
            "    mul.lo.u32 %sm_addr, %sm_global_row, {};\n",
            n_tiles_kv * 4
        ));
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_addr2;\n");
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_scratch_max_base;\n");
        ptx.push_str(&format!(
            "    @%sm_writer_pred st.shared.f32 [%sm_addr], %s_max_{}_hi;\n",
            t
        ));
    }
    // Make all partial maxes visible to all warps.
    ptx.push_str("    bar.sync 0; // softmax cross-warp partial-max visibility\n");

    // ============================================================
    // STEP 2: each lane reads all n_tiles_kv partial maxes for its rows
    // and computes the global_max. Two rows per lane (lo and hi).
    // ============================================================
    for t in 0..tpw {
        // Recompute m_tile for this slot (same formula).
        if t == 0 {
            ptx.push_str("    mov.u32 %qkt_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %qkt_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        let log2_n_tiles_kv = n_tiles_kv.trailing_zeros();
        ptx.push_str(&format!(
            "    shr.u32 %qkt_m_tile, %qkt_global_t, {};\n",
            log2_n_tiles_kv
        ));
        ptx.push_str("    shl.b32 %qkt_q_warp_off, %qkt_m_tile, 4;\n");
        // For each half (lo/hi), read all n_tiles_kv partials and reduce.
        for (half_name, row_offset) in [("lo", 0u32), ("hi", 8u32)].iter() {
            // sm_global_row = m_tile*16 + lo_row + row_offset
            ptx.push_str("    add.u32 %sm_global_row, %qkt_q_warp_off, %sm_lo_row;\n");
            if *row_offset > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %sm_global_row, %sm_global_row, {};\n",
                    row_offset
                ));
            }
            // Initialize global_max from the n=0 partial; then max-fold n=1..
            ptx.push_str(&format!(
                "    mul.lo.u32 %sm_addr, %sm_global_row, {};\n",
                n_tiles_kv * 4
            ));
            ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_scratch_max_base;\n");
            ptx.push_str(&format!(
                "    ld.shared.f32 %sm_global_max_{}, [%sm_addr];   // n=0\n",
                half_name
            ));
            for n in 1..n_tiles_kv {
                ptx.push_str(&format!(
                    "    ld.shared.f32 %sm_part_max, [%sm_addr + {}];   // n={}\n",
                    n * 4,
                    n
                ));
                ptx.push_str(&format!(
                    "    max.f32 %sm_global_max_{}, %sm_global_max_{}, %sm_part_max;\n",
                    half_name, half_name
                ));
            }
            // Stash the global_max in %s_max_<t>_<half> for downstream use.
            ptx.push_str(&format!(
                "    mov.f32 %s_max_{}_{}, %sm_global_max_{};\n",
                t, half_name, half_name
            ));
        }
    }

    // ============================================================
    // STEP 3: P = exp((S - global_max) * log2(e)) per lane.
    // ============================================================
    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // === softmax tile t={} : step 3 = exp(S - global_max) ===\n",
            t
        ));
        for (lane, max_var) in [(0u32, "lo"), (1u32, "lo"), (2u32, "hi"), (3u32, "hi")].iter() {
            ptx.push_str(&format!(
                "    sub.f32 %s_acc_{}_{}, %s_acc_{}_{}, %s_max_{}_{};\n",
                t, lane, t, lane, t, max_var
            ));
            ptx.push_str(&format!(
                "    mul.f32 %s_acc_{}_{}, %s_acc_{}_{}, 0f3FB8AA3B; // * log2(e)\n",
                t, lane, t, lane
            ));
            ptx.push_str(&format!(
                "    ex2.approx.f32 %s_acc_{}_{}, %s_acc_{}_{};\n",
                t, lane, t, lane
            ));
        }
    }

    // ============================================================
    // STEP 4: per-warp partial row-sum + cross-warp combine to global_sum.
    // No correction factor needed since each warp's partial_sum was computed
    // from exp(S - global_max) directly (not exp(S - partial_max)).
    // ============================================================
    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // === softmax tile t={} : step 4 = per-warp partial row-sum ===\n",
            t
        ));
        ptx.push_str(&format!(
            "    .reg .f32 %s_sum_{}_lo, %s_sum_{}_hi;\n",
            t, t
        ));
        ptx.push_str(&format!(
            "    add.f32 %s_sum_{}_lo, %s_acc_{}_0, %s_acc_{}_1;\n",
            t, t, t
        ));
        ptx.push_str(&format!(
            "    add.f32 %s_sum_{}_hi, %s_acc_{}_2, %s_acc_{}_3;\n",
            t, t, t
        ));
        for (mask, comment) in [(1u32, "lane k <-> k^1"), (2u32, "lane k <-> k^2")].iter() {
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %s_shfl_{}_tmp, %s_sum_{}_lo, {}, 0x1f, 0xffffffff; // {}\n",
                t, t, mask, comment
            ));
            ptx.push_str(&format!(
                "    add.f32 %s_sum_{}_lo, %s_sum_{}_lo, %s_shfl_{}_tmp;\n",
                t, t, t
            ));
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %s_shfl_{}_tmp, %s_sum_{}_hi, {}, 0x1f, 0xffffffff; // {}\n",
                t, t, mask, comment
            ));
            ptx.push_str(&format!(
                "    add.f32 %s_sum_{}_hi, %s_sum_{}_hi, %s_shfl_{}_tmp;\n",
                t, t, t
            ));
        }
        // Writer-lane writes partial_sum_lo at (global_row=m_tile*16+lo_row,
        // n_tile) and partial_sum_hi at (global_row+8, n_tile).
        if t == 0 {
            ptx.push_str("    mov.u32 %qkt_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %qkt_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        let log2_n_tiles_kv = n_tiles_kv.trailing_zeros();
        let n_tiles_kv_mask = n_tiles_kv - 1;
        ptx.push_str(&format!(
            "    shr.u32 %qkt_m_tile, %qkt_global_t, {};\n",
            log2_n_tiles_kv
        ));
        ptx.push_str(&format!(
            "    and.b32 %qkt_n_tile, %qkt_global_t, {};\n",
            n_tiles_kv_mask
        ));
        ptx.push_str("    shl.b32 %qkt_q_warp_off, %qkt_m_tile, 4;\n");
        ptx.push_str("    add.u32 %sm_global_row, %qkt_q_warp_off, %sm_lo_row;\n");
        ptx.push_str(&format!(
            "    mul.lo.u32 %sm_addr, %sm_global_row, {};\n",
            n_tiles_kv * 4
        ));
        ptx.push_str("    shl.b32 %sm_addr2, %qkt_n_tile, 2;\n");
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_addr2;\n");
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_scratch_sum_base;\n");
        ptx.push_str(&format!(
            "    @%sm_writer_pred st.shared.f32 [%sm_addr], %s_sum_{}_lo;\n",
            t
        ));
        ptx.push_str("    add.u32 %sm_global_row, %sm_global_row, 8;\n");
        ptx.push_str(&format!(
            "    mul.lo.u32 %sm_addr, %sm_global_row, {};\n",
            n_tiles_kv * 4
        ));
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_addr2;\n");
        ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_scratch_sum_base;\n");
        ptx.push_str(&format!(
            "    @%sm_writer_pred st.shared.f32 [%sm_addr], %s_sum_{}_hi;\n",
            t
        ));
    }
    ptx.push_str("    bar.sync 0; // softmax cross-warp partial-sum visibility\n");

    // ============================================================
    // STEP 5: each lane reads all n_tiles_kv partial sums for its rows
    // and computes the global_sum. Store in %s_sum_<t>_<half> so that
    // (a) downstream code (P_packed packing — vestigial) sees consistent
    // values and (b) finalize divides by the GLOBAL sum.
    // ============================================================
    for t in 0..tpw {
        if t == 0 {
            ptx.push_str("    mov.u32 %qkt_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %qkt_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        let log2_n_tiles_kv = n_tiles_kv.trailing_zeros();
        ptx.push_str(&format!(
            "    shr.u32 %qkt_m_tile, %qkt_global_t, {};\n",
            log2_n_tiles_kv
        ));
        ptx.push_str("    shl.b32 %qkt_q_warp_off, %qkt_m_tile, 4;\n");
        for (half_name, row_offset) in [("lo", 0u32), ("hi", 8u32)].iter() {
            ptx.push_str("    add.u32 %sm_global_row, %qkt_q_warp_off, %sm_lo_row;\n");
            if *row_offset > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %sm_global_row, %sm_global_row, {};\n",
                    row_offset
                ));
            }
            ptx.push_str(&format!(
                "    mul.lo.u32 %sm_addr, %sm_global_row, {};\n",
                n_tiles_kv * 4
            ));
            ptx.push_str("    add.u32 %sm_addr, %sm_addr, %sm_scratch_sum_base;\n");
            ptx.push_str(&format!(
                "    ld.shared.f32 %sm_global_sum_{}, [%sm_addr];   // n=0\n",
                half_name
            ));
            for n in 1..n_tiles_kv {
                ptx.push_str(&format!(
                    "    ld.shared.f32 %sm_part_sum, [%sm_addr + {}];   // n={}\n",
                    n * 4,
                    n
                ));
                ptx.push_str(&format!(
                    "    add.f32 %sm_global_sum_{}, %sm_global_sum_{}, %sm_part_sum;\n",
                    half_name, half_name
                ));
            }
            ptx.push_str(&format!(
                "    mov.f32 %s_sum_{}_{}, %sm_global_sum_{};\n",
                t, half_name, half_name
            ));
        }
    }

    // ============================================================
    // STEP 4.5: persist reduced row-max/row-sum to SMEM keyed by ABSOLUTE
    // global query row (hd > block_kv fix).
    //
    // %s_max_<t>_<half> / %s_sum_<t>_<half> are per-query-ROW reduced stats,
    // but their register index <t> is a per-warp QK^T output-tile SLOT
    // (m_tile = global_t / (block_kv/8)). finalize re-derives a DIFFERENT
    // m_tile from the same slot under the PV decomposition
    // (m_tile = global_t / (head_dim/8)); the two agree only when
    // head_dim == block_kv. To make finalize basis-independent, persist each
    // reduced stat here keyed by its ABSOLUTE global_row so finalize can read
    // it back by its PV m-tile's global_row.
    //
    // Gating: one lane per 4-lane row quad (lane%4 == 0) owns the row, so each
    // (global_row, stat) is written exactly once per (warp, slot). Across all
    // warps + QK^T slots the m-tiles tile the full bq row range, so every row
    // 0..bq is written exactly once. Visibility to finalize is guaranteed by
    // the P-scatter / Phase-C / projection-save bar.sync fences that follow.
    // ============================================================
    {
        let reduced_max_base = tier_b1_reduced_stats_offset(config);
        let reduced_sum_base = tier_b1_reduced_stats_sum_offset(config);
        let log2_n_tiles_kv = n_tiles_kv.trailing_zeros();
        ptx.push_str(
            "    // === STEP 4.5: persist reduced row-max/row-sum to SMEM by absolute global_row ===\n",
        );
        // Producer-side namespace (%rstp_*) is DISTINCT from finalize's reader
        // namespace (%rstat_*): both live in the SAME PTX function scope, so a
        // shared register name would be a duplicate `.reg` definition (ptxas
        // rejects "Duplicate definition of variable").
        ptx.push_str("    .reg .u32 %rstp_max_base, %rstp_sum_base, %rstp_addr;\n");
        ptx.push_str("    .reg .u64 %rstp_base_u64;\n");
        // reduced_max_base / reduced_sum_base as u32 SMEM addresses (once).
        ptx.push_str(&format!(
            "    add.u64 %rstp_base_u64, %shmem_base, {}; // reduced row-max base\n",
            reduced_max_base
        ));
        ptx.push_str("    cvt.u32.u64 %rstp_max_base, %rstp_base_u64;\n");
        ptx.push_str(&format!(
            "    add.u64 %rstp_base_u64, %shmem_base, {}; // reduced row-sum base\n",
            reduced_sum_base
        ));
        ptx.push_str("    cvt.u32.u64 %rstp_sum_base, %rstp_base_u64;\n");
        for t in 0..tpw {
            // Recompute qkt m_tile for this slot (same N3 formula as Step 5).
            if t == 0 {
                ptx.push_str("    mov.u32 %qkt_global_t, %warp_id;\n");
            } else {
                ptx.push_str(&format!(
                    "    add.u32 %qkt_global_t, %warp_id, {};\n",
                    t * 8
                ));
            }
            ptx.push_str(&format!(
                "    shr.u32 %qkt_m_tile, %qkt_global_t, {};\n",
                log2_n_tiles_kv
            ));
            ptx.push_str("    shl.b32 %qkt_q_warp_off, %qkt_m_tile, 4; // m_tile * 16\n");
            for (half_name, row_offset) in [("lo", 0u32), ("hi", 8u32)].iter() {
                // global_row = m_tile*16 + lo_row + row_offset.
                ptx.push_str("    add.u32 %sm_global_row, %qkt_q_warp_off, %sm_lo_row;\n");
                if *row_offset > 0 {
                    ptx.push_str(&format!(
                        "    add.u32 %sm_global_row, %sm_global_row, {};\n",
                        row_offset
                    ));
                }
                // addr offset = global_row * 4 bytes (f32).
                ptx.push_str("    shl.b32 %rstp_addr, %sm_global_row, 2;\n");
                // row-max store: [reduced_max_base + global_row*4] = %s_max.
                ptx.push_str("    add.u32 %sm_addr, %rstp_max_base, %rstp_addr;\n");
                ptx.push_str(&format!(
                    "    @%sm_writer_pred st.shared.f32 [%sm_addr], %s_max_{}_{};\n",
                    t, half_name
                ));
                // row-sum store: [reduced_sum_base + global_row*4] = %s_sum.
                ptx.push_str("    add.u32 %sm_addr, %rstp_sum_base, %rstp_addr;\n");
                ptx.push_str(&format!(
                    "    @%sm_writer_pred st.shared.f32 [%sm_addr], %s_sum_{}_{};\n",
                    t, half_name
                ));
            }
        }
        // The subsequent P-scatter (emit_scatter_p_to_smem) issues a `bar.sync 0`
        // before the PV MMA, and the Phase-C swap issues another before finalize.
        // Either fence makes these reduced-stat writes visible CTA-wide before
        // finalize reads them; no extra bar.sync is needed here.
    }

    // NO per-warp divide here. P (=exp(S - global_max)) is left in
    // %s_acc_<t>_<lane>; finalize divides by %s_sum_<t>_<half> (global)
    // as the SINGLE normalization step. The prior emission divided
    // here AND in finalize, producing 1/global_sum^2 instead of
    // 1/global_sum. The per-warp divide also normalized by a partial
    // sum, which combined with the per-warp partial-only path produced
    // n_tiles_kv-times-too-large attention contributions.

    // ----- pack 2 f32 P values into one b32 holding 2 f16 lanes -----
    // (Vestigial: %p_packed_<t> is no longer consumed by PV after the
    // SMEM-bridge refactor. Kept until the next pass cleans it up; ptxas
    // prunes dead registers so this is free at runtime.)
    for t in 0..tpw {
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %mma_h0, %s_acc_{}_0;\n",
            t
        ));
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %mma_h1, %s_acc_{}_1;\n",
            t
        ));
        ptx.push_str(&format!(
            "    mov.b32 %p_packed_{}, {{%mma_h0, %mma_h1}};\n",
            t
        ));
    }
}

/// PV: m16n8k16 MMA against P (packed f16, registers) and V (SMEM,
/// tier_b1_v_offset_<ping|pong>). Accumulates into per-warp
/// `%o_acc_<t>_<lane>` registers -- these MUST persist across all
/// kv_iters per spec section 3.1 (register-resident O_acc).
///
/// **B1.6 TODO (deferral #1):** B-fragment load from V SMEM uses uniform
/// address; swap to `matmul_mma::emit_load_b_fragment_smem`. Also
/// requires `smem_layout::tier_b1_v_offset_{ping,pong}` to compute the
/// per-tile base address — the placeholder uses `mov.u64 ..., 0` so the
/// fragment-load swap alone leaves the kernel reading from offset 0.
///
/// **B1.6 TODO (deferral #2):** warp-ownership predicate not yet emitted.
/// All 8 warps execute every MMA; gate each on `setp.eq.u32` + `@%pred`.
///
/// **B1.6 TODO (deferral #8 -- Phase-B-specific):** the O_acc registers
/// are declared inside this function; they must move to the kernel
/// orchestrator (`tier_b1::synthesize`) so their lifetime spans all
/// kv_iter loop iterations. Currently re-zero'd per call.
fn emit_pv_mma(ptx: &mut String, config: &FlashAttentionConfig, slot: u32) {
    let tpw = tiles_per_warp_pv(config);
    let hd = config.head_dim as u32;
    let bkv = config.block_kv as u32;
    let n_tiles_d_pv = (hd / 8).max(1);
    assert!(
        n_tiles_d_pv.is_power_of_two(),
        "N3: PV requires head_dim/8 to be a power of 2; got {}",
        n_tiles_d_pv
    );
    let log2_n_tiles_d_pv = n_tiles_d_pv.trailing_zeros();
    let n_tiles_d_pv_mask = n_tiles_d_pv - 1;
    // m_tile stride for P (A-fragment) = 16 rows × bkv cols × 2 bytes.
    let p_m_stride_bytes = 16 * bkv * 2;
    assert!(
        p_m_stride_bytes.is_power_of_two(),
        "N3: PV requires 16*bkv*2 to be a power of 2; got {}",
        p_m_stride_bytes
    );
    let log2_p_m_stride = p_m_stride_bytes.trailing_zeros();
    // n_tile stride for V (B-fragment) = 8 cols * 2 bytes = 16 bytes.
    // (V is bkv × hd row-major; n_tile shifts within the head_dim cols.)
    ptx.push_str(&format!(
        "    // PV MMA: bq={} hd={} tpw_pv={} slot={} (N3)\n",
        config.block_q, config.head_dim, tpw, slot
    ));
    // B1.6 deferral #1 (Phase B PV): populate %tb1_phase_b_smem_v with the
    // real V tile SMEM offset for the selected slot.
    let v_off = if slot == 0 {
        tier_b1_v_offset_ping(config)
    } else {
        tier_b1_v_offset_pong(config)
    };
    ptx.push_str(&format!(
        "    add.u64 %tb1_phase_b_smem_v, %shmem_base, {}; // V tile @ slot={} offset\n",
        v_off, slot
    ));
    // u32 sister reg for the B-fragment helper.
    ptx.push_str("    .reg .u32 %tb1_phase_b_smem_v_u32;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_phase_b_smem_v_u32, %tb1_phase_b_smem_v;\n");

    // P_smem base for A-fragment loads. Bridges softmax's D-fragment
    // output to the PV A-fragment via `tier_b1_p_offset`.
    let p_off = tier_b1_p_offset(config);
    ptx.push_str(&format!(
        "    add.u64 %tb1_phase_b_smem_q, %shmem_base, {}; // P_smem base (reusing q-slot u64 reg)\n",
        p_off
    ));
    ptx.push_str("    .reg .u32 %tb1_phase_b_smem_p_u32;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_phase_b_smem_p_u32, %tb1_phase_b_smem_q;\n");

    // N3 runtime tile coords for PV: each warp's slot maps to a DISTINCT
    // (m_tile_pv, n_tile_pv) via global_t = warp_id + local_t*8.
    ptx.push_str("    .reg .u32 %pv_global_t, %pv_m_tile, %pv_n_tile;\n");
    ptx.push_str("    .reg .u32 %pv_p_warp_off, %pv_v_warp_off;\n");
    ptx.push_str("    .reg .u32 %pv_a_base, %pv_b_base;\n");
    // N1a K-iter shifted bases (only emitted/used for k_iter > 0; the
    // declaration is unconditional since `.reg` emission is cheap).
    ptx.push_str("    .reg .u32 %pv_a_base_k, %pv_b_base_k;\n");

    for t in 0..tpw {
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_pv_a_{}_0, %tb1_pv_a_{}_1, %tb1_pv_a_{}_2, %tb1_pv_a_{}_3;\n",
            t, t, t, t
        ));
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_pv_b_{}_0, %tb1_pv_b_{}_1;\n",
            t, t
        ));

        // N3: per-warp runtime tile coords.
        if t == 0 {
            ptx.push_str("    mov.u32 %pv_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %pv_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        ptx.push_str(&format!(
            "    shr.u32 %pv_m_tile, %pv_global_t, {};\n",
            log2_n_tiles_d_pv
        ));
        ptx.push_str(&format!(
            "    and.b32 %pv_n_tile, %pv_global_t, {};\n",
            n_tiles_d_pv_mask
        ));
        // A-fragment base = P_smem + m_tile_pv * 16 * bkv * 2 bytes.
        ptx.push_str(&format!(
            "    shl.b32 %pv_p_warp_off, %pv_m_tile, {};\n",
            log2_p_m_stride
        ));
        ptx.push_str(
            "    add.u32 %pv_a_base, %tb1_phase_b_smem_p_u32, %pv_p_warp_off;\n",
        );
        // B-fragment base = V_smem + n_tile_pv * 8 * (bkv*2) bytes. V is
        // stored COL-MAJOR [bkv × hd] so n_tile advance = 8 cols ×
        // bkv*2 byte-stride per col.
        ptx.push_str(&format!(
            "    mul.lo.u32 %pv_v_warp_off, %pv_n_tile, {};\n",
            8 * bkv * 2
        ));
        ptx.push_str(
            "    add.u32 %pv_b_base, %tb1_phase_b_smem_v_u32, %pv_v_warp_off;\n",
        );

        // N1a: PV K-loop. m16n8k16 K-dim = 16; for block_kv > 16 we need
        // ceil(block_kv / 16) K-iters per output tile, accumulating into
        // the same %o_acc_<t>_*. Each K-iter shifts:
        //   - A-base in P_smem by k_iter * 16 * 2 = k_iter * 32 bytes
        //     (k-stride = 2 bytes per P col within the 16-row m-tile)
        //   - B-base in V_smem by k_iter * 16 * hd * 2 bytes
        //     (k-iter advances bkv-row position by 16, each bkv-row spans
        //     hd cols × 2 bytes)
        assert!(
            bkv.is_multiple_of(16),
            "N1a: PV K-loop requires block_kv divisible by 16; got {}",
            bkv
        );
        let n_k_iters_pv = bkv / 16;

        for k_iter in 0..n_k_iters_pv {
            // Per-K-iter shifted bases.
            let a_base_expr: String = if k_iter == 0 {
                "%pv_a_base".to_string()
            } else {
                let off = k_iter * 32;
                ptx.push_str(&format!(
                    "    add.u32 %pv_a_base_k, %pv_a_base, {}; // PV A-base shift k_iter={}\n",
                    off, k_iter
                ));
                "%pv_a_base_k".to_string()
            };
            let b_base_expr: String = if k_iter == 0 {
                "%pv_b_base".to_string()
            } else {
                // V is COL-MAJOR [bkv × hd]: K-iter advances by 16
                // k-positions = 16 * 2 = 32 bytes along the inner (m)
                // dim. Prior emission used `k_iter * 16 * hd * 2` which
                // matched the row-major V interpretation; under
                // col-major that overshot by `hd` (= one full row span
                // of 8 n-tiles) per K-iter.
                let off = k_iter * 16 * 2;
                ptx.push_str(&format!(
                    "    add.u32 %pv_b_base_k, %pv_b_base, {}; // PV B-base shift k_iter={}\n",
                    off, k_iter
                ));
                "%pv_b_base_k".to_string()
            };

            let a_fragment_regs = [
                format!("tb1_pv_a_{}_0", t),
                format!("tb1_pv_a_{}_1", t),
                format!("tb1_pv_a_{}_2", t),
                format!("tb1_pv_a_{}_3", t),
            ];
            emit_load_a_fragment_smem(
                ptx,
                &a_fragment_regs,
                &a_base_expr,
                (bkv * 2) as usize,
            );

            // B1.6 deferral #1 (PV B-fragment): per-lane V load via helper.
            // V is stored COL-MAJOR [bkv × hd] f16 in SMEM (the V-scatter
            // uses `emit_dfragment_scatter_col_major`). B-fragment helper
            // expects col_stride = bytes between adjacent N cols = bkv*2,
            // and within each col, 2 k-adjacent f16 are contiguous (which
            // they ARE in col-major). Prior emission passed (hd*2) under
            // the assumption that V was row-major, which produced
            // effectively P @ scrambled-V (helper read V at byte
            // n*(hd*2) + k*2 — that byte position in row-major-stored V
            // resolved to V[n, k] not V[k, n]).
            let b_fragment_regs = [
                format!("tb1_pv_b_{}_0", t),
                format!("tb1_pv_b_{}_1", t),
            ];
            emit_load_b_fragment_smem(
                ptx,
                &b_fragment_regs,
                &b_base_expr,
                (bkv * 2) as usize,
            );
            let d_regs = [
                format!("%o_acc_{}_0", t),
                format!("%o_acc_{}_1", t),
                format!("%o_acc_{}_2", t),
                format!("%o_acc_{}_3", t),
            ];
            let a_regs = [
                format!("%tb1_pv_a_{}_0", t),
                format!("%tb1_pv_a_{}_1", t),
                format!("%tb1_pv_a_{}_2", t),
                format!("%tb1_pv_a_{}_3", t),
            ];
            let b_regs = [
                format!("%tb1_pv_b_{}_0", t),
                format!("%tb1_pv_b_{}_1", t),
            ];
            let c_regs = d_regs.clone();
            // N1a: accumulate into same %o_acc_<t>_* across K-iters.
            // N3: unconditional MMA — each warp writes its OWN slot.
            emit_mma_instruction(ptx, &d_regs, &a_regs, &b_regs, &c_regs);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn make_config(bq: i64, bkv: i64, hd: i64, dm: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: bq,
            block_kv: bkv,
            head_dim: hd,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: dm,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn phase_b_emits_three_subphase_headers() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_phase_b_attention(&mut ptx, &cfg, 0, 0);
        assert!(ptx.contains("Phase B: attention compute"));
        assert!(ptx.contains("QK^T MMA"));
        assert!(ptx.contains("Online softmax"));
        assert!(ptx.contains("PV MMA"));
    }

    #[test]
    fn phase_b_emits_qkt_and_pv_mmas() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_phase_b_attention(&mut ptx, &cfg, 0, 0);
        // N1: QK^T iterates head_dim/16 K-iters, PV iterates block_kv/16.
        // At canonical 32x32x32: QK^T = 1 tpw * 2 K-iters = 2 MMAs;
        // PV = 1 tpw * 2 K-iters = 2 MMAs. Total = 4.
        assert_eq!(
            ptx.matches("mma.sync.aligned.m16n8k16").count(),
            4,
            "expected 4 MMAs (2 QK^T K-iters + 2 PV K-iters) at canonical 32x32x32; got:\n{}",
            ptx
        );
    }

    #[test]
    fn phase_b_o_acc_declared_by_register_budget() {
        // After B1.6 deferral #4 hoist, the O_acc registers are declared
        // by register_budget::declare_registers at orchestrator scope.
        // emit_phase_b_attention writes into them but no longer declares
        // them — confirm both halves of the contract.
        let cfg = make_config(32, 32, 32, 2048);
        let mut helper_ptx = String::new();
        emit_phase_b_attention(&mut helper_ptx, &cfg, 0, 0);
        assert!(
            !helper_ptx.contains(".reg .f32 %o_acc_0_0"),
            "emit_phase_b_attention must NOT declare %o_acc post-hoist; declaration belongs to register_budget::declare_registers"
        );
        let mut prelude_ptx = String::new();
        crate::flash_attention_v2::tier_b1::register_budget::declare_registers(&mut prelude_ptx, &cfg);
        assert!(
            prelude_ptx.contains(".reg .f32 %o_acc_0_0"),
            "register_budget::declare_registers must emit the %o_acc declaration"
        );
    }

    #[test]
    fn phase_b_qkt_scale_count_equals_tpw_times_four() {
        // B1.6 deferral #5: 1/sqrt(head_dim) scale per S-accumulator lane
        // emitted after the MMA loop. Count must equal tpw_qkt * 4 lanes.
        // Match on the QK^T-specific hex literal (different from the
        // log2(e) scale that the softmax pipeline also emits, per #6).
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_phase_b_attention(&mut ptx, &cfg, 0, 0);
        let tpw = tiles_per_warp_qkt(&cfg);
        let qkt_scale_bits = (1.0_f32 / 32.0_f32.sqrt()).to_bits();
        let qkt_scale_marker = format!("0f{:08X}", qkt_scale_bits);
        // Match `mul.f32 ..., 0f<bits>` to exclude the diagnostic comment
        // header that also prints the hex literal.
        let scale_count = ptx
            .matches(&format!(", {};", qkt_scale_marker)[..])
            .count();
        assert_eq!(
            scale_count,
            (tpw * 4) as usize,
            "expected {} QK^T scale ops ({} tile(s) * 4 lanes); got {}",
            tpw * 4,
            tpw,
            scale_count
        );
        // The scale literal must be a non-zero f32. For hd=32, 1/sqrt(32) bits.
        let expected_scale = (1.0_f32 / 32.0_f32.sqrt()).to_bits();
        assert!(
            ptx.contains(&format!("0f{:08X}", expected_scale)),
            "QK^T tail must embed the IEEE-754 hex for 1/sqrt(32) = 0f{:08X}",
            expected_scale
        );
    }

    #[test]
    fn phase_b_softmax_emits_bfly_placeholder() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_phase_b_attention(&mut ptx, &cfg, 0, 0);
        assert!(
            ptx.contains("shfl.sync.bfly.b32"),
            "softmax placeholder must emit at least one bfly reduction"
        );
    }

    #[test]
    fn tiles_per_warp_qkt_canonical_small() {
        let cfg = make_config(32, 32, 32, 2048);
        assert_eq!(tiles_per_warp_qkt(&cfg), 1);
    }

    #[test]
    fn tiles_per_warp_pv_canonical_small() {
        let cfg = make_config(32, 32, 32, 2048);
        assert_eq!(tiles_per_warp_pv(&cfg), 1);
    }
}
