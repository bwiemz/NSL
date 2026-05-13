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
    tier_b1_k_offset_ping, tier_b1_k_offset_pong, tier_b1_q_offset, tier_b1_v_offset_ping,
    tier_b1_v_offset_pong,
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
    emit_pv_mma(ptx, config, slot);
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
/// **B1.6 TODO (deferral #6 -- Phase-B-specific):** the bfly reduction
/// tree is currently a single placeholder `shfl.sync.bfly.b32` per row.
/// Real implementation is 3-step log2(8)=3 reductions for the lane group
/// holding one row's S values (spec section 5.3 lane->fragment map).
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
    // Placeholder bfly reduction per tile. Real impl is per spec 5.3.
    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // softmax tile t={} -- placeholder bfly + exp (B1.6 deferral #6)\n",
            t
        ));
        // B1.6 TODO (deferral #6): replace with 3-step bfly tree (log2(8)=3
        // reductions) operating on the lane group holding one row's S values.
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %s_acc_{}_0, %s_acc_{}_0, 4, 0x1f, 0xffffffff;\n",
            t, t
        ));
        // Pack to b32 holding two f16 lanes (placeholder).
        // B1.6 TODO (deferral #7): replace with real exp(S - row_max) + row_sum
        // correction using running %m_acc and %l_acc state from prologue scope.
        // %p_packed_<t> is hoisted to register_budget::declare_registers (B1.6
        // deferral #4 resolution); we just write into it.
        ptx.push_str(&format!(
            "    mov.b32 %p_packed_{}, %s_acc_{}_0;\n",
            t, t
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

    for t in 0..tpw {
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_pv_a_{}_0, %tb1_pv_a_{}_1, %tb1_pv_a_{}_2, %tb1_pv_a_{}_3;\n",
            t, t, t, t
        ));
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_pv_b_{}_0, %tb1_pv_b_{}_1;\n",
            t, t
        ));
        // P-packed is in %p_packed_<t>, reuse as the A-fragment lane 0.
        // (Placeholder: real A-fragment for PV has 4 lanes per thread.
        // Real mapping requires the m16n8k16 register-A-fragment layout
        // which differs from the SMEM-A-fragment helper signature; deferred
        // to a B1.6 follow-on commit.)
        ptx.push_str(&format!(
            "    mov.b32 %tb1_pv_a_{}_0, %p_packed_{};\n",
            t, t
        ));
        ptx.push_str(&format!("    mov.b32 %tb1_pv_a_{}_1, %p_packed_{};\n", t, t));
        ptx.push_str(&format!("    mov.b32 %tb1_pv_a_{}_2, %p_packed_{};\n", t, t));
        ptx.push_str(&format!("    mov.b32 %tb1_pv_a_{}_3, %p_packed_{};\n", t, t));
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
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_phase_b_attention(&mut ptx, &cfg, 0, 0);
        let tpw = tiles_per_warp_qkt(&cfg);
        let scale_count = ptx.matches("mul.f32 %s_acc_").count();
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
