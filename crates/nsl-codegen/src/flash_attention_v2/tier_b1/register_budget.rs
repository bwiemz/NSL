//! Static spill analysis at codegen time. Counts live MMA accumulators
//! per warp and rejects configs predicted to exceed the 255/thread cap.
//! Per spec section 5.4 register budget.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::tier_b1::attention_mma::{tiles_per_warp_pv, tiles_per_warp_qkt};
use crate::flash_attention_v2::tier_b1::projection_mma::{tiles_per_warp, tiles_per_warp_kv};

#[derive(Debug)]
pub struct SpillRisk(pub String);

/// Declare and zero-initialize all Tier B.1 per-warp accumulator
/// registers + Phase B placeholder SMEM base regs at orchestrator scope.
/// Resolves B1.6 deferrals #4 (multi-call .reg collision), #7 (running
/// max/sum state lifetime), and #8 (O_acc cross-iter lifetime) — all
/// three were architecturally the same hoist viewed from different
/// sub-helpers.
///
/// Emission order:
///   1. Q projection accumulators (`%q_acc_<t>_<lane>`)
///   2. K projection accumulators (`%k_acc_<t>_<lane>`)
///   3. V projection accumulators (`%v_acc_<t>_<lane>`)
///   4. QK^T S accumulators (`%s_acc_<t>_<lane>`)
///   5. PV O accumulators (`%o_acc_<t>_<lane>`)
///   6. Softmax P-packed (`%p_packed_<t>`)
///   7. Phase B placeholder SMEM base regs (B1.6 deferral #1 swaps these)
///
/// Each accumulator tile is 4 f32 lanes per thread (m16n8k16 fragment
/// shape per spec section 5.5). Zero-init is emitted after the
/// declarations so the snapshot diff has clean structure.
pub fn declare_registers(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str(
        "    // === Tier B.1 accumulator + scratch register declarations (B1.6 deferral #4) ===\n",
    );

    let tpw_q = tiles_per_warp(config);
    let tpw_kv = tiles_per_warp_kv(config);
    let tpw_qkt = tiles_per_warp_qkt(config);
    let tpw_pv = tiles_per_warp_pv(config);

    ptx.push_str(&format!(
        "    // Q projection accumulators ({} tile(s)/warp x 4 f32 lanes)\n",
        tpw_q
    ));
    for t in 0..tpw_q {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %q_acc_{}_{};\n", t, lane));
        }
    }

    ptx.push_str(&format!(
        "    // K projection accumulators ({} tile(s)/warp x 4 f32 lanes)\n",
        tpw_kv
    ));
    for t in 0..tpw_kv {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %k_acc_{}_{};\n", t, lane));
        }
    }

    ptx.push_str(&format!(
        "    // V projection accumulators ({} tile(s)/warp x 4 f32 lanes)\n",
        tpw_kv
    ));
    for t in 0..tpw_kv {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %v_acc_{}_{};\n", t, lane));
        }
    }

    ptx.push_str(&format!(
        "    // QK^T S accumulators ({} tile(s)/warp x 4 f32 lanes)\n",
        tpw_qkt
    ));
    for t in 0..tpw_qkt {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %s_acc_{}_{};\n", t, lane));
        }
    }

    ptx.push_str(&format!("    // PV O accumulators (REGISTER-RESIDENT across kv_iters; {} tile(s)/warp x 4 f32 lanes)\n", tpw_pv));
    for t in 0..tpw_pv {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %o_acc_{}_{};\n", t, lane));
        }
    }

    ptx.push_str(&format!(
        "    // Softmax P-packed ({} b32 holding 2 f16 lanes each)\n",
        tpw_qkt
    ));
    for t in 0..tpw_qkt {
        ptx.push_str(&format!("    .reg .b32 %p_packed_{};\n", t));
    }

    ptx.push_str("    // Phase B placeholder SMEM base regs (B1.6 deferral #1 will replace)\n");
    ptx.push_str("    .reg .u64 %tb1_phase_b_smem_q, %tb1_phase_b_smem_k, %tb1_phase_b_smem_v;\n");
    ptx.push_str("    mov.u64 %tb1_phase_b_smem_q, 0;\n");
    ptx.push_str("    mov.u64 %tb1_phase_b_smem_k, 0;\n");
    ptx.push_str("    mov.u64 %tb1_phase_b_smem_v, 0;\n");

    // Warp-ownership gating (B1.6 deferral #2). `%warp_id` is already
    // declared + populated by `phases::forward::prelude::emit` (it's a
    // Tier A primitive); we just add the per-tile predicate register.
    // Each per-tile MMA gates on `%warp_id == (t % 8)`.
    ptx.push_str("    // Warp-ownership gating predicate (B1.6 deferral #2)\n");
    ptx.push_str("    .reg .pred %wo_pred;\n");

    // MMA fragment-load temps (B1.6 deferral #1 resolution). The helpers
    // `matmul_mma::emit_load_{a,b}_fragment_smem` write into
    // %mma_a_row / %mma_b_row / %mma_addr; we precompute the per-lane
    // %mma_a_row from %tid.x using the m16n8k16 lane-to-row mapping per
    // spec section 5.5 (row = (laneid % 4) * 2 + laneid / 16; B row
    // matches A row for the k-dim).
    ptx.push_str("    // MMA fragment-load temps (B1.6 deferral #1)\n");
    ptx.push_str("    .reg .f16 %mma_h0, %mma_h1;          // f32->f16 temps\n");
    ptx.push_str("    .reg .u32 %mma_a_row, %mma_b_row;    // fragment row indices\n");
    ptx.push_str("    .reg .u32 %mma_addr;                  // SMEM address scratch\n");
    ptx.push_str("    .reg .u32 %mma_laneid;                // laneid = tid.x % 32\n");
    ptx.push_str("    mov.u32 %mma_laneid, %tid.x;\n");
    ptx.push_str("    and.b32 %mma_laneid, %mma_laneid, 31;\n");
    ptx.push_str("    and.b32 %mma_a_row, %mma_laneid, 3;   // laneid % 4\n");
    ptx.push_str("    shl.b32 %mma_a_row, %mma_a_row, 1;    // * 2\n");
    ptx.push_str("    shr.u32 %mma_addr, %mma_laneid, 4;    // laneid / 16 (scratch)\n");
    ptx.push_str(
        "    add.u32 %mma_a_row, %mma_a_row, %mma_addr; // row = (laneid%4)*2 + laneid/16\n",
    );
    ptx.push_str("    mov.u32 %mma_b_row, %mma_a_row;       // B row matches A row for k-dim\n");

    ptx.push_str("    // === Zero-init all f32 accumulators ===\n");
    for t in 0..tpw_q {
        for lane in 0..4 {
            ptx.push_str(&format!("    mov.f32 %q_acc_{}_{}, 0f00000000;\n", t, lane));
        }
    }
    for t in 0..tpw_kv {
        for lane in 0..4 {
            ptx.push_str(&format!("    mov.f32 %k_acc_{}_{}, 0f00000000;\n", t, lane));
            ptx.push_str(&format!("    mov.f32 %v_acc_{}_{}, 0f00000000;\n", t, lane));
        }
    }
    for t in 0..tpw_qkt {
        for lane in 0..4 {
            ptx.push_str(&format!("    mov.f32 %s_acc_{}_{}, 0f00000000;\n", t, lane));
        }
    }
    for t in 0..tpw_pv {
        for lane in 0..4 {
            ptx.push_str(&format!("    mov.f32 %o_acc_{}_{}, 0f00000000;\n", t, lane));
        }
    }
}

/// Per-thread register cap on sm_80+ NVIDIA GPUs.
const REG_CAP_PER_THREAD: u32 = 255;

/// Headroom reserved for compiler spill-prevention margin (live ranges
/// the static estimate doesn't see).
const REG_HEADROOM: u32 = 15;

/// Estimate live registers per thread for the given Tier B.1 config.
/// Per spec §5.4 register budget table (8 warps per CTA, MMA m16n8k16
/// fragment layout). Returns Ok(regs_estimate) if under the cap minus
/// headroom; Err(SpillRisk) otherwise.
///
/// **INVARIANT (load-bearing for `chunk_config::select` short-circuit):**
/// this function MUST remain chunk-independent. `select` short-circuits
/// the descending search with `break` on register failure under the
/// assumption that all chunks produce the same register estimate. If a
/// future revision introduces chunk-dependent register pressure (e.g.,
/// chunk affects how many MMA accumulators are co-live), the `break` at
/// `chunk_config.rs::select` becomes wrong — REMOVE it there before
/// introducing chunk dependence here.
///
/// Budget terms (per-thread, with 8 warps × 32 threads = 256 threads/CTA):
///   * Q resident accumulators       : (bq/16) * (hd/8) / 8 * 4 (f32)
///   * K projection in-flight chunk  : 4 (f32; one m16n8k16 accumulator
///                                       fragment = 4 f32 lanes per thread)
///   * V projection in-flight chunk  : 4 (f32; same shape as K)
///   * QK^T S fragments              : (bq/16) * (bkv/8) / 8 * 4 (f32)
///   * P_f16 packed (post-softmax)   : 8 (b32 holding 16 f16 lanes)
///   * O_acc (lives across kv_iters) : (bq/16) * (hd/8) / 8 * 4 (f32)
///   * Row stats + correction        : ~16 (f32 + scratch)
///   * Addressing + predicates       : ~32 (mixed)
///
/// At the canonical (bq=64, bkv=64, hd=128) the formula yields 144
/// regs/thread; at (64,64,64) it yields 112. Both fit comfortably under
/// the 255-15=240 cap with headroom.
///
/// The formula is integer-only (no floor/ceil); MMA tile counts
/// (bq/16, bkv/8, hd/8) are assumed integral — the caller (validator
/// chain or chunk_config::select) is responsible for rejecting configs
/// where they aren't.
pub fn analyze(config: &FlashAttentionConfig) -> Result<u32, SpillRisk> {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;

    // Guard against non-MMA-aligned tiles. MMA m16n8k16 needs bq%16==0,
    // bkv%8==0, hd%8==0. If any fails, the kernel can't be emitted at all
    // — but register_budget is the wrong layer to enforce this; just
    // produce a conservative estimate.
    let q_resident = (bq / 16).max(1) * (hd / 8).max(1) / 8 * 4;
    let k_inflight = 4; // one chunk's K accumulator fragment
    let v_inflight = 4; // one chunk's V accumulator fragment
    let s_frags = (bq / 16).max(1) * (bkv / 8).max(1) / 8 * 4;
    let p_packed = 8;
    let o_acc = (bq / 16).max(1) * (hd / 8).max(1) / 8 * 4;
    let row_stats = 16;
    let addressing = 32;

    let total =
        q_resident + k_inflight + v_inflight + s_frags + p_packed + o_acc + row_stats + addressing;

    if total + REG_HEADROOM > REG_CAP_PER_THREAD {
        return Err(SpillRisk(format!(
            "tier_b1 estimated {} regs/thread for (bq={}, bkv={}, hd={}); exceeds {} cap minus {} headroom",
            total, bq, bkv, hd, REG_CAP_PER_THREAD, REG_HEADROOM
        )));
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn make_config(bq: i64, bkv: i64, hd: i64) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: bq,
            block_kv: bkv,
            head_dim: hd,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: 2048,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn canonical_64_64_64_fits_budget() {
        // Hand-derive: q_resident=16, k=4, v=4, s=16, p=8, o=16,
        // row_stats=16, addressing=32 -> total = 112 regs/thread.
        // Cap-with-headroom = 255 - 15 = 240; comfortable margin.
        let cfg = make_config(64, 64, 64);
        let regs = analyze(&cfg).expect("canonical config should fit");
        assert_eq!(regs, 112, "formula drift in canonical (64,64,64) budget");
    }

    #[test]
    fn canonical_64_64_128_fits_budget() {
        // Hand-derive: q_resident=32, k=4, v=4, s=16, p=8, o=32,
        // row_stats=16, addressing=32 -> total = 144 regs/thread.
        // hd=128 doubles q_resident and o_acc vs hd=64 (both terms
        // scale with hd/8); other terms unchanged. Still under cap.
        let cfg = make_config(64, 64, 128);
        let regs = analyze(&cfg).expect("canonical hd=128 should fit");
        assert_eq!(regs, 144, "formula drift in canonical (64,64,128) budget");
        assert!(regs < REG_CAP_PER_THREAD - REG_HEADROOM);
    }

    #[test]
    fn small_config_fits_with_margin() {
        let cfg = make_config(32, 32, 32);
        let regs = analyze(&cfg).expect("small config should fit");
        assert!(
            regs < 100,
            "small config should use way under 100 regs, got {}",
            regs
        );
    }
}
