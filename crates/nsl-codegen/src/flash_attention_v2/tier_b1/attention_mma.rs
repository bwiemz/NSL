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
use crate::matmul_mma::emit_mma_instruction;

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the QK^T
/// output (bq x bkv). Always at least 1.
fn tiles_per_warp_qkt(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let m_tiles = bq / 16;
    let n_tiles = bkv / 8;
    let total = m_tiles.max(1) * n_tiles.max(1);
    (total / 8).max(1)
}

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the PV
/// output (bq x hd). Identical to Q-projection's tile count.
fn tiles_per_warp_pv(config: &FlashAttentionConfig) -> u32 {
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
/// **B1.6 TODO (deferral #5 -- Phase-B-specific):** S-fragment dtype.
/// QK^T output should be f32 accumulator (4 lanes per thread), then
/// scaled by `1/sqrt(head_dim)` before softmax. Scaling is a scalar
/// `mul.f32` per lane; ship it as part of the QK^T tail (not yet emitted
/// in the scaffold -- B1.6 must add the scaling line per spec section 4.2).
fn emit_qkt_mma(ptx: &mut String, config: &FlashAttentionConfig, slot: u32) {
    let tpw = tiles_per_warp_qkt(config);
    ptx.push_str(&format!(
        "    // QK^T MMA: bq={} bkv={} tpw_qkt={} slot={}\n",
        config.block_q, config.block_kv, tpw, slot
    ));
    // Declare placeholder SMEM base registers for Q and K input tiles.
    // B1.6 deferral #1: these uniform addresses are replaced by per-thread
    // offset arithmetic via matmul_mma::emit_load_{a,b}_fragment_smem.
    ptx.push_str("    .reg .u64 %tb1_phase_b_smem_q, %tb1_phase_b_smem_k;\n");
    ptx.push_str("    mov.u64 %tb1_phase_b_smem_q, 0; // placeholder (B1.6 deferral #1)\n");
    ptx.push_str("    mov.u64 %tb1_phase_b_smem_k, 0; // placeholder (B1.6 deferral #1)\n");
    // Declare per-warp S accumulators.
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %s_acc_{}_{};\n", t, lane));
        }
    }
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %s_acc_{}_{}, 0f00000000;\n",
                t, lane
            ));
        }
    }
    // Placeholder A/B-fragment registers + uniform loads. (Same deferral
    // class as Q/KV projection -- B1.6 swaps to matmul_mma helpers.)
    for t in 0..tpw {
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_qkt_a_{}_0, %tb1_qkt_a_{}_1, %tb1_qkt_a_{}_2, %tb1_qkt_a_{}_3;\n",
            t, t, t, t
        ));
        ptx.push_str(&format!(
            "    .reg .b32 %tb1_qkt_b_{}_0, %tb1_qkt_b_{}_1;\n",
            t, t
        ));
        for i in 0..4 {
            ptx.push_str(&format!(
                "    ld.shared.b32 %tb1_qkt_a_{}_{}, [%tb1_phase_b_smem_q];\n",
                t, i
            ));
        }
        for i in 0..2 {
            ptx.push_str(&format!(
                "    ld.shared.b32 %tb1_qkt_b_{}_{}, [%tb1_phase_b_smem_k];\n",
                t, i
            ));
        }
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
        // B1.6 TODO (deferral #2): gate this MMA on the warp-ownership
        // predicate (setp.eq.u32 %wo_pred, %warp_id, (t %% 8); @%wo_pred mma.sync ...).
        // Currently every one of the 8 warps executes every MMA tile; the
        // accumulator end-state is therefore 8x the intended sum.
        ptx.push_str(&format!(
            "    // QK^T MMA tile t={} -- intended ownership: t %% 8 == warp_id (gate NOT yet emitted; B1.6)\n",
            t
        ));
        emit_mma_instruction(ptx, &d_regs, &a_regs, &b_regs, &c_regs);
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
        ptx.push_str(&format!("    .reg .b32 %p_packed_{};\n", t));
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
    ptx.push_str(&format!(
        "    // PV MMA: bq={} hd={} tpw_pv={} slot={}\n",
        config.block_q, config.head_dim, tpw, slot
    ));
    // Declare placeholder SMEM base register for V input tile.
    // B1.6 deferral #1: replaced by per-thread offset arithmetic via
    // matmul_mma::emit_load_b_fragment_smem for the V tile.
    ptx.push_str("    .reg .u64 %tb1_phase_b_smem_v;\n");
    ptx.push_str("    mov.u64 %tb1_phase_b_smem_v, 0; // placeholder (B1.6 deferral #1)\n");
    // Declare per-warp O_acc + zero-init.
    // B1.6 deferral #8 (lifetime): move these declarations to
    // tier_b1::synthesize outer prologue so O_acc persists across all
    // kv_iter calls to emit_phase_b_attention. Currently re-zero'd per call.
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %o_acc_{}_{};\n", t, lane));
            ptx.push_str(&format!(
                "    mov.f32 %o_acc_{}_{}, 0f00000000;\n",
                t, lane
            ));
        }
    }
    // Placeholder PV MMA per tile.
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
        // (Placeholder: real A-fragment for PV has 4 lanes per thread.)
        ptx.push_str(&format!(
            "    mov.b32 %tb1_pv_a_{}_0, %p_packed_{};\n",
            t, t
        ));
        ptx.push_str(&format!("    mov.b32 %tb1_pv_a_{}_1, %p_packed_{};\n", t, t));
        ptx.push_str(&format!("    mov.b32 %tb1_pv_a_{}_2, %p_packed_{};\n", t, t));
        ptx.push_str(&format!("    mov.b32 %tb1_pv_a_{}_3, %p_packed_{};\n", t, t));
        for i in 0..2 {
            ptx.push_str(&format!(
                "    ld.shared.b32 %tb1_pv_b_{}_{}, [%tb1_phase_b_smem_v];\n",
                t, i
            ));
        }
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
        // B1.6 TODO (deferral #2): gate this MMA on the warp-ownership
        // predicate (setp.eq.u32 %wo_pred, %warp_id, (t %% 8); @%wo_pred mma.sync ...).
        // Currently every one of the 8 warps executes every MMA tile; the
        // accumulator end-state is therefore 8x the intended sum.
        ptx.push_str(&format!(
            "    // PV MMA tile t={} -- intended ownership: t %% 8 == warp_id (gate NOT yet emitted; B1.6)\n",
            t
        ));
        emit_mma_instruction(ptx, &d_regs, &a_regs, &b_regs, &c_regs);
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
    fn phase_b_o_acc_declared_in_pv_helper() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_phase_b_attention(&mut ptx, &cfg, 0, 0);
        assert!(
            ptx.contains(".reg .f32 %o_acc_0_0"),
            "O_acc registers must be declared by emit_pv_mma (B1.6 deferral #8 will hoist)"
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
