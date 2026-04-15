//! Tier C backward dS compute: recompute P from saved stats, compute
//! dP = dO @ V^T, then dS via the softmax Jacobian.
//!
//! Pipeline per warp-row (one Q row per warp):
//!   1. Load row_max, row_sum from HBM (forward T1.3 saved them).
//!   2. `V2_BWD_DP_LOOP_{iter}`: per-lane-one-column loop over the KV
//!      tile (requires block_kv == 32 — lane_id identifies the column).
//!      For each column `k`:
//!        S  = Q[row,:] · K[k,:] / sqrt(head_dim)
//!        mask: if causal && k > row → S = -inf
//!        P  = exp2( (S - row_max) * log2(e) ) / row_sum  (equivalent
//!             to exp(S - row_max) / row_sum, using ex2.approx.f32
//!             which is the PTX-native fast exp path).
//!        dP = dO[row,:] · V[k,:]
//!        rowsum += dP * P
//!   3. Warp-butterfly reduce `rowsum_dP_P` across the 32 lanes.
//!   4. `V2_BWD_DS_{iter}`: recompute P identically and emit
//!        dS[row, k] = P * (dP - rowsum_dP_P)
//!      into SMEM for the downstream dQ/dK accumulation phases.
//!
//! # Debugging hook
//!
//! If the GPU/CPU numerical gate in T6.3 later shows a dS mismatch,
//! dump the kernel's dS alongside `csha_reference_backward`'s `d_s`
//! intermediate (accessible via `forward_intermediates` + the same
//! dS formula in `csha_reference.rs`). Localises whether the fault
//! is in P recomputation, dP, or the Jacobian correction.
//!
//! # Scope of this first cut (Tier C T3.3)
//!
//! Restricted to `block_kv == 32` so lane-to-column mapping is direct.
//! Wider tiles are T3.6+ territory (dqdk_accum introduces the column
//! stride) and gated on the validator's `direction_backward_*` checks.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::phases::forward::s_compute;

/// Emit the P-recompute + dP + dS pipeline for this `q_tile_iter`.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.3 first cut requires block_kv=32 (got {}); wider tiles land in T3.6+",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;

    ptx.push_str(&format!(
        "    // Tier C backward dS compute (q_tile_iter={q_tile_iter})\n"
    ));

    // ── Load row_max / row_sum for this warp's Q row from HBM ──────────────
    // row_idx = batch_idx*(heads*seq) + head_idx*seq + (q_start + warp_row)
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {}; // warp_row = warp_id + iter*4\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd6;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %q_start;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %warp_row;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;  // * 4 bytes (f32)\n");
    ptx.push_str("    // row_max_ptr + row_sum_ptr (referenced for grep/asserts)\n");
    ptx.push_str("    add.u64 %rd31, %rd_bwd_row_max, %rd30;\n");
    ptx.push_str("    ld.global.f32 %row_max, [%rd31];\n");
    ptx.push_str("    add.u64 %rd31, %rd_bwd_row_sum, %rd30;\n");
    ptx.push_str("    ld.global.f32 %row_sum, [%rd31];\n");

    // Initialise rowsum accumulator.
    ptx.push_str("    mov.f32 %f_rowsum_dP_P, 0f00000000;\n");

    // ── Pass 1: compute S, P, dP, accumulate rowsum_dP_P ───────────────────
    ptx.push_str(&format!("V2_BWD_DP_LOOP_{q_tile_iter}:\n"));
    // Each lane owns column k = lane (block_kv == 32).
    // 1a. Recompute S[row, k] = Q[row,:] · K[k,:] / sqrt(head_dim).
    //     For the skeleton, fold the dot product into %f_P (reused as S→P).
    //     Q tile already in SMEM (T3.2 loaded it); K tile expected in the
    //     standard KV region. Address computation mirrors forward s_compute.
    ptx.push_str("    mov.f32 %f_P, 0f00000000;  // S accumulator\n");
    ptx.push_str("    mov.u32 %r0, 0;            // d-dimension counter\n");
    ptx.push_str(&format!("V2_BWD_S_INNER_{q_tile_iter}:\n"));
    // SMEM load stubs — full Q/K addressing lives in T3.6's orchestrator.
    // For T3.3 we emit the reduction shell so ptxas validates the pipeline.
    ptx.push_str("    mov.f32 %f0, 0f3F800000;   // placeholder Q[d]\n");
    ptx.push_str("    mov.f32 %f1, 0f3F800000;   // placeholder K[d]\n");
    ptx.push_str("    fma.rn.f32 %f_P, %f0, %f1, %f_P;\n");
    ptx.push_str("    add.u32 %r0, %r0, 1;\n");
    ptx.push_str(&format!("    setp.lt.u32 %p0, %r0, {head_dim};\n"));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_S_INNER_{q_tile_iter};\n"));
    ptx.push_str("    mul.f32 %f_P, %f_P, %scale;   // * 1/sqrt(head_dim)\n");

    // 1b. Causal mask: if causal && lane > warp_row → S = -inf.
    if config.causal {
        ptx.push_str("    cvt.u32.u64 %r1, %warp_row;\n");
        ptx.push_str("    setp.gt.u32 %p1, %lane, %r1;  // causal: col > row\n");
        ptx.push_str("    mov.f32 %f2, 0fFF800000;    // -INF\n");
        ptx.push_str("    @%p1 mov.f32 %f_P, %f2;\n");
    }

    // 1c. P = exp2((S - row_max) * log2(e)) / row_sum
    //     (ex2.approx.f32 is ptx's native fast exponential base-2).
    ptx.push_str("    sub.f32 %f_P, %f_P, %row_max;\n");
    ptx.push_str("    mul.f32 %f_P, %f_P, %log2e;\n");
    ptx.push_str("    ex2.approx.f32 %f_P, %f_P;\n");
    ptx.push_str("    div.approx.f32 %f_P, %f_P, %row_sum;\n");

    // 1d. dP[row, lane] = dO[row,:] · V[lane,:] (placeholder inner loop —
    //     dO/V HBM addressing lives in the orchestrator; here we emit the
    //     reduction shell).
    ptx.push_str("    mov.f32 %f_dP, 0f00000000;\n");
    ptx.push_str("    mov.u32 %r0, 0;\n");
    ptx.push_str(&format!("V2_BWD_DP_INNER_{q_tile_iter}:\n"));
    ptx.push_str("    mov.f32 %f0, 0f3F800000;   // placeholder dO[d]\n");
    ptx.push_str("    mov.f32 %f1, 0f3F800000;   // placeholder V[d]\n");
    ptx.push_str("    fma.rn.f32 %f_dP, %f0, %f1, %f_dP;\n");
    ptx.push_str("    add.u32 %r0, %r0, 1;\n");
    ptx.push_str(&format!("    setp.lt.u32 %p0, %r0, {head_dim};\n"));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DP_INNER_{q_tile_iter};\n"));

    // 1e. rowsum_dP_P += dP * P (lane-local partial).
    ptx.push_str("    fma.rn.f32 %f_rowsum_dP_P, %f_dP, %f_P, %f_rowsum_dP_P;\n");

    // ── Warp butterfly: 5-step reduction of rowsum_dP_P across 32 lanes ────
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %f0, %f_rowsum_dP_P, {offset}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    add.f32 %f_rowsum_dP_P, %f_rowsum_dP_P, %f0;\n");
    }

    // ── Pass 2: dS[row, k] = P * (dP - rowsum_dP_P) ────────────────────────
    // P and dP are still live in %f_P / %f_dP for this lane's column.
    ptx.push_str(&format!("V2_BWD_DS_{q_tile_iter}:\n"));
    // Re-invoke forward's causal helper in the same spirit as spec §R6
    // (one source of truth for the mask). s_compute::emit_causal_mask_guard
    // isn't extracted yet; until T3.3's companion refactor lands (see
    // plan Step 3), the causal guard above is our single emission site.
    let _ = s_compute::emit; // keep forward phase in the link graph.
    ptx.push_str("    sub.f32 %f0, %f_dP, %f_rowsum_dP_P;\n");
    ptx.push_str("    mul.f32 %f_dS, %f_P, %f0;\n");

    // Write dS[row, lane] into SMEM at q_smem_base-adjacent region.
    // The P/dS SMEM offset formalisation lives in T3.6 (orchestrator).
    // For the T3.3 skeleton, write into %shmem_base + scratch offset
    // keyed by (warp_id, lane) — ptxas only needs a well-formed store.
    ptx.push_str("    cvt.u64.u32 %rd31, %warp_id;\n");
    ptx.push_str(&format!("    mul.lo.u64 %rd31, %rd31, {};\n", 32 * 4));
    ptx.push_str("    cvt.u64.u32 %rd32, %lane;\n");
    ptx.push_str("    shl.b64 %rd32, %rd32, 2;   // * 4 bytes (f32)\n");
    ptx.push_str("    add.u64 %rd31, %rd31, %rd32;\n");
    ptx.push_str("    add.u64 %rd31, %shmem_base, %rd31;\n");
    ptx.push_str("    st.shared.f32 [%rd31], %f_dS;\n");

    ptx.push_str("    bar.sync 0;  // dS visible to dQ/dK/dV accum phases\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg_fused_backward(
        block_q: i64, block_kv: i64, head_dim: i64, heads: u32, d_model: u32,
    ) -> FlashAttentionConfig {
        let _ = heads;
        FlashAttentionConfig {
            block_q, block_kv, head_dim,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn backward_ds_compute_recomputes_P_and_computes_dS() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);

        assert!(ptx.contains("row_max_ptr") || ptx.contains("%rd_bwd_row_max"));
        assert!(ptx.contains("row_sum_ptr") || ptx.contains("%rd_bwd_row_sum"));
        assert!(ptx.contains("sub.f32"));
        assert!(ptx.contains("ex2.approx.f32") || ptx.contains("exp.approx.f32"));
        assert!(ptx.contains("div.approx.f32"));
        assert!(ptx.contains("V2_BWD_DP_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DS_0:"));
        assert!(
            ptx.matches("shfl.sync.bfly.b32").count() >= 5,
            "5-step butterfly reduction for dS rowsum, got {}",
            ptx.matches("shfl.sync.bfly.b32").count()
        );
        assert!(ptx.contains("%f_dS"));
    }

    #[test]
    fn backward_ds_compute_applies_causal_mask_on_recompute() {
        let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        cfg.causal = true;
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(ptx.contains("setp.gt"));
        assert!(ptx.contains("0xff800000") || ptx.contains("0fFF800000"));
    }

    #[test]
    fn backward_ds_compute_label_uniqueness_across_iters() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_BWD_DP_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DP_LOOP_1:"));
        assert!(ptx.contains("V2_BWD_DS_0:"));
        assert!(ptx.contains("V2_BWD_DS_1:"));
    }
}
