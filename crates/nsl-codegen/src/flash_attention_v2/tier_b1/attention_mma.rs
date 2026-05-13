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
    tier_b1_k_offset_ping, tier_b1_k_offset_pong, tier_b1_p_offset, tier_b1_q_offset,
    tier_b1_v_offset_ping, tier_b1_v_offset_pong,
};
use crate::matmul_mma::{
    emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction_predicated,
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
/// Each tile is warp-gated on `%warp_id == (t % 8)` to match the QK^T
/// MMA's warp ownership pattern — only the owning warp's `%s_acc_*`
/// registers hold meaningful post-softmax values, and only that warp
/// writes its tile's slice of P_smem.
fn emit_scatter_p_to_smem(ptx: &mut String, config: &FlashAttentionConfig) {
    let tpw = tiles_per_warp_qkt(config);
    let bkv = config.block_kv as u32;
    let n_tiles_kv = (bkv / 8).max(1);
    let p_off = tier_b1_p_offset(config);

    ptx.push_str("    // === P scatter to SMEM (PV A-fragment bridge) ===\n");
    ptx.push_str("    .reg .u32 %sp_laneid, %sp_lo_row, %sp_lo_col_base;\n");
    ptx.push_str("    .reg .u32 %sp_row_off, %sp_col_off, %sp_lane_off;\n");
    ptx.push_str("    .reg .u32 %sp_addr_u32, %sp_addr_i;\n");
    ptx.push_str("    .reg .u64 %sp_addr_u64;\n");
    ptx.push_str("    .reg .b16 %sp_h;\n");
    ptx.push_str("    .reg .pred %sp_warp_pred;\n");

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
        let m_tile = t / n_tiles_kv;
        let n_tile = t % n_tiles_kv;
        let tile_off_bytes = p_off + m_tile * 16 * bkv * 2 + n_tile * 8 * 2;

        ptx.push_str(&format!(
            "    // P scatter tile t={} (m_tile={}, n_tile={}, warp owner = {})\n",
            t,
            m_tile,
            n_tile,
            t % 8
        ));
        ptx.push_str(&format!(
            "    setp.eq.u32 %sp_warp_pred, %warp_id, {};\n",
            t % 8
        ));
        ptx.push_str(&format!(
            "    add.u64 %sp_addr_u64, %shmem_base, {};\n",
            tile_off_bytes
        ));
        ptx.push_str("    cvt.u32.u64 %sp_addr_u32, %sp_addr_u64;\n");
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
            ptx.push_str("    @%sp_warp_pred st.shared.b16 [%sp_addr_i], %sp_h;\n");
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
    // Codegen-time computation: 1 / sqrt(head_dim) as f32 IEEE-754 bits.
    // Spec section 4.2: scale QK^T output by 1/sqrt(d_k) before softmax.
    let scale_bits = (1.0_f32 / (hd as f32).sqrt()).to_bits();
    ptx.push_str(&format!(
        "    // QK^T MMA: bq={} bkv={} tpw_qkt={} slot={} (scale=1/sqrt({})=0f{:08X})\n",
        config.block_q, config.block_kv, tpw, slot, hd, scale_bits
    ));
    // %tb1_phase_b_smem_q/k and %s_acc_<t>_<lane> are declared + zero-init'd
    // by register_budget::declare_registers at orchestrator scope (B1.6
    // deferral #4 resolution). The placeholder zero-bases there are
    // overwritten here with real SMEM offsets (B1.6 deferral #1).
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

    for t in 0..tpw {
        // B1.6 deferral #1 resolution: per-lane fragment loads via
        // matmul_mma helpers. A-fragment from Q SMEM (stride hd*2 bytes);
        // B-fragment from K SMEM (stride hd*2 bytes for the K-row layout
        // that QK^T uses).
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_qkt_a_{}_0, %tb1_qkt_a_{}_1, %tb1_qkt_a_{}_2, %tb1_qkt_a_{}_3;\n",
            t, t, t, t
        ));
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_qkt_b_{}_0, %tb1_qkt_b_{}_1;\n",
            t, t
        ));
        let a_fragment_regs = [
            format!("tb1_qkt_a_{}_0", t),
            format!("tb1_qkt_a_{}_1", t),
            format!("tb1_qkt_a_{}_2", t),
            format!("tb1_qkt_a_{}_3", t),
        ];
        emit_load_a_fragment_smem(
            ptx,
            &a_fragment_regs,
            "%tb1_phase_b_smem_q_u32",
            (hd * 2) as usize,
        );
        let b_fragment_regs = [
            format!("tb1_qkt_b_{}_0", t),
            format!("tb1_qkt_b_{}_1", t),
        ];
        emit_load_b_fragment_smem(
            ptx,
            &b_fragment_regs,
            "%tb1_phase_b_smem_k_u32",
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
        // B1.6 deferral #2 resolution: warp-ownership gate. Only the warp
        // owning tile `t` executes its MMA; the predicate `%wo_pred` is
        // set per-tile and the MMA is prefixed with `@%wo_pred`.
        ptx.push_str(&format!(
            "    setp.eq.u32 %wo_pred, %warp_id, {}; // QK^T tile t={} ownership\n",
            t % 8, t
        ));
        emit_mma_instruction_predicated(ptx, &d_regs, &a_regs, &b_regs, &c_regs, "wo_pred");
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
    ptx.push_str(&format!(
        "    // Online softmax: bq={} tpw_qkt={} kv_iter={}\n",
        config.block_q, tpw, kv_iter
    ));
    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // === softmax tile t={} : row-max bfly tree (deferral #6 partial) ===\n",
            t
        ));
        // Per-tile scratch for the row-max reduction.
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
        // Row-max bfly tree across the 4-lane row group. Per spec §5.5
        // lanes (4k, 4k+1, 4k+2, 4k+3) share the same row_lo; pairing
        // via bfly mask=1 then mask=2 reduces those 4 lanes to a common
        // max. Same pattern applies independently to row_lo+8.
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
        // ----- subtract row_max + exp ----- (B1.6 deferral #6 cont'd)
        //
        // Spec section 4.2: P = exp(S - row_max). PTX has no native exp,
        // so we use the identity exp(x) = 2^(x * log2(e)) with the fast
        // `ex2.approx.f32` instruction. The log2(e) constant is folded
        // into the subtract: (S - row_max) * log2(e) → ex2.approx.f32.
        //
        // log2(e) = 1.4426950408889634 (IEEE-754 single = 0f3FB8AA3B).
        ptx.push_str(&format!(
            "    // subtract row_max, scale by log2(e), apply ex2.approx (deferral #6)\n",
        ));
        // frag indices 0,1 belong to row_lo (subtract %s_max_<t>_lo);
        // frag indices 2,3 belong to row_lo+8 (subtract %s_max_<t>_hi).
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

        // ----- row_sum bfly tree ----- (deferral #6 cont'd)
        //
        // After exp, the lane's 4 fragment slots hold P values for
        // (row_lo, cols col_pair*2..col_pair*2+1) and
        // (row_lo+8, cols col_pair*2..col_pair*2+1). row_sum across 8
        // cols of one row requires the same 4-lane bfly tree as row_max.
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

        // ----- divide by row_sum ----- (deferral #6 cont'd)
        //
        // P_ij = exp(S_ij - row_max_i) / row_sum_i.
        for (lane, sum_var) in [(0u32, "lo"), (1u32, "lo"), (2u32, "hi"), (3u32, "hi")].iter() {
            ptx.push_str(&format!(
                "    div.approx.f32 %s_acc_{}_{}, %s_acc_{}_{}, %s_sum_{}_{};\n",
                t, lane, t, lane, t, sum_var
            ));
        }

        // ----- pack 2 f32 P values into one b32 holding 2 f16 lanes -----
        //
        // Frag indices 0,1 hold cols col_pair*2 and col_pair*2+1 of
        // row_lo (post-divide). The PV MMA's A-fragment expects f16 col
        // pairs packed into b32 registers. Convert each f32 -> f16 then
        // pack two f16 into one b32 via the PTX mov.b32 {h0,h1} syntax.
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
    ptx.push_str(&format!(
        "    // PV MMA: bq={} hd={} tpw_pv={} slot={}\n",
        config.block_q, config.head_dim, tpw, slot
    ));
    // B1.6 deferral #1 (Phase B PV): populate %tb1_phase_b_smem_v with the
    // real V tile SMEM offset for the selected slot. The orchestrator's
    // placeholder zero-base is overwritten here.
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

    // P_smem base for A-fragment loads (set once; per-tile offset added
    // inside the loop). Bridges softmax's D-fragment output to the PV
    // A-fragment via `tier_b1_p_offset` (see emit_scatter_p_to_smem).
    let p_off = tier_b1_p_offset(config);
    ptx.push_str(&format!(
        "    add.u64 %tb1_phase_b_smem_q, %shmem_base, {}; // P_smem base (reusing q-slot u64 reg)\n",
        p_off
    ));
    ptx.push_str("    .reg .u32 %tb1_phase_b_smem_p_u32, %tb1_phase_b_smem_p_tile_u32;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_phase_b_smem_p_u32, %tb1_phase_b_smem_q;\n");

    for t in 0..tpw {
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_pv_a_{}_0, %tb1_pv_a_{}_1, %tb1_pv_a_{}_2, %tb1_pv_a_{}_3;\n",
            t, t, t, t
        ));
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_pv_b_{}_0, %tb1_pv_b_{}_1;\n",
            t, t
        ));

        // PV A-fragment via SMEM round-trip. PV tile t has output position
        // (m_tile_pv = t / (hd/8), n_tile_pv = t % (hd/8)) where m-dim
        // covers q-rows and n-dim covers head_dim cols. The A-fragment
        // K-dim spans 16 cols of P starting at k_iter*16 (k_iter = 0 in
        // the current single-K-iter scaffold).
        //
        // A-fragment base = P_smem + m_tile_pv*16*block_kv*2 + 0*2 bytes
        // (k_iter=0). emit_load_a_fragment_smem then adds `%mma_a_row *
        // (block_kv * 2)` for the per-lane row offset.
        let m_tile_pv = t / n_tiles_d_pv;
        let p_tile_byte_off = m_tile_pv * 16 * bkv * 2;
        ptx.push_str(&format!(
            "    // PV A-frag tile t={}: m_tile_pv={} → P_smem +{} bytes\n",
            t, m_tile_pv, p_tile_byte_off
        ));
        if p_tile_byte_off > 0 {
            ptx.push_str(&format!(
                "    add.u32 %tb1_phase_b_smem_p_tile_u32, %tb1_phase_b_smem_p_u32, {};\n",
                p_tile_byte_off
            ));
        } else {
            ptx.push_str(
                "    mov.u32 %tb1_phase_b_smem_p_tile_u32, %tb1_phase_b_smem_p_u32;\n",
            );
        }
        let a_fragment_regs = [
            format!("tb1_pv_a_{}_0", t),
            format!("tb1_pv_a_{}_1", t),
            format!("tb1_pv_a_{}_2", t),
            format!("tb1_pv_a_{}_3", t),
        ];
        emit_load_a_fragment_smem(
            ptx,
            &a_fragment_regs,
            "%tb1_phase_b_smem_p_tile_u32",
            (bkv * 2) as usize,
        );

        // B1.6 deferral #1 (PV B-fragment): real per-lane V load.
        // V is bkv rows x hd cols; B-fragment col-major stride is hd*2.
        let b_fragment_regs = [
            format!("tb1_pv_b_{}_0", t),
            format!("tb1_pv_b_{}_1", t),
        ];
        emit_load_b_fragment_smem(
            ptx,
            &b_fragment_regs,
            "%tb1_phase_b_smem_v_u32",
            (hd * 2) as usize,
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
        // B1.6 deferral #2 resolution: warp-ownership gate on PV MMA.
        ptx.push_str(&format!(
            "    setp.eq.u32 %wo_pred, %warp_id, {}; // PV tile t={} ownership\n",
            t % 8, t
        ));
        emit_mma_instruction_predicated(ptx, &d_regs, &a_regs, &b_regs, &c_regs, "wo_pred");
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
        // QK^T tpw=1 + PV tpw=1 = 2 MMAs at canonical small config.
        assert_eq!(
            ptx.matches("mma.sync.aligned.m16n8k16").count(),
            2,
            "expected 2 MMAs (1 QK^T + 1 PV) at canonical 32x32x32; got:\n{}",
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
