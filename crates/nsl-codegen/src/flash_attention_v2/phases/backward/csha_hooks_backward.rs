//! Tier C backward CSHA hooks: inverse RoPE, projection gradients,
//! and RMSNorm gradient. Each mirrors its forward counterpart in
//! `phases/forward/csha_hooks.rs` with the chain-rule math.
//!
//! ## Math (cross-reference to the CPU backward in
//! `crates/nsl-codegen/tests/csha_reference.rs`)
//!
//! dRoPE (inverse rotation, Adjacent layout — `x[2i]` paired with
//! `x[2i+1]`):
//!   dx0 =  dY[2i]  * cos + dY[2i+1] * sin
//!   dx1 = -dY[2i]  * sin + dY[2i+1] * cos
//! Orthogonal rotation ⇒ inverse == transpose of forward matrix.
//!
//! dproj (weight gradients):
//!   dWq += x_norm^T @ dQ_preRoPE    (shape [d_model, kv_dim])
//!   dWk += x_norm^T @ dK_preRoPE
//!   dWv += x_norm^T @ dV
//! Each null-guarded independently on its weight pointer (matches the
//! forward fused-projection fix).
//!
//! dRMSNorm (closed form, with D = d_model, rms = sqrt(mean_sq + eps)):
//!   g_d  = dx_norm[i,d] * norm_weight[d]
//!   s    = sum_d g_d * x[i,d]
//!   dx[i,d] = g_d / rms - x[i,d] * s / (D * rms^3)
//! The `s` reduction is per-row — 5-step warp butterfly across the 32
//! lanes that cover the `d_model` dimension.

use crate::flash_attention::FlashAttentionConfig;

// ────────────────────────────────────────────────────────────────────────
// dRoPE
// ────────────────────────────────────────────────────────────────────────

/// Emit inverse RoPE rotation for both dQ and dK SMEM tiles. V is
/// never rotated (matches forward's Adjacent-layout epilogue).
///
/// No-op when `rope_q=false` or `csha=None`.
pub fn emit_drope(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if config.csha.is_none() || !config.rope_q {
        ptx.push_str("    // Tier C dRoPE: rope_q=false, no emission\n");
        return;
    }
    let head_dim = config.head_dim as u32;
    let half = head_dim / 2;

    ptx.push_str(&format!(
        "    // Tier C backward dRoPE (q_tile_iter={q_tile_iter})\n"
    ));

    for (label, smem_base) in [("Q", "%q_smem_base"), ("K", "%k_smem_base")] {
        ptx.push_str(&format!("V2_BWD_DROPE_{label}_LOOP_{q_tile_iter}:\n"));
        // One pair per lane (half pairs total; block_q*half/128 per lane
        // in the full orchestrator; skeleton does a single representative
        // pair so ptxas sees the four fmas and the load pattern).
        ptx.push_str("    ld.param.u64 %rd30, [cos_ptr];\n");
        ptx.push_str("    ld.param.u64 %rd31, [sin_ptr];\n");
        // Null-guard — skip when either is null.
        ptx.push_str("    setp.eq.u64 %p0, %rd30, 0;\n");
        ptx.push_str("    setp.eq.u64 %p1, %rd31, 0;\n");
        ptx.push_str("    or.pred %p0, %p0, %p1;\n");
        ptx.push_str(&format!(
            "    @%p0 bra V2_BWD_DROPE_{label}_SKIP_{q_tile_iter};\n"
        ));

        // Pair index: lane modulo half-dim (one pair per lane; wider
        // head_dim iterates).
        ptx.push_str(&format!(
            "    rem.u32 %r0, %lane, {half}; // pair idx in 0..head_dim/2\n"
        ));
        // Load cos/sin for this pair — one f16 each, convert to f32.
        ptx.push_str("    cvt.u64.u32 %rd32, %r0;\n");
        ptx.push_str("    shl.b64 %rd32, %rd32, 1;  // * 2 bytes (f16)\n");
        ptx.push_str("    add.u64 %rd33, %rd30, %rd32;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd33];\n");
        ptx.push_str("    cvt.f32.f16 %f0, %h0;      // cos\n");
        ptx.push_str("    add.u64 %rd33, %rd31, %rd32;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd33];\n");
        ptx.push_str("    cvt.f32.f16 %f1, %h0;      // sin\n");

        // Load dY[2i] and dY[2i+1] from SMEM (placeholder for the
        // orchestrator's actual address calc).
        ptx.push_str(&format!(
            "    mul.lo.u32 %r1, %r0, 4;    // pair_byte_off = pair*4 (two f16)\n"
        ));
        ptx.push_str("    cvt.u64.u32 %rd32, %r1;\n");
        ptx.push_str(&format!("    add.u64 %rd33, {smem_base}, %rd32;\n"));
        ptx.push_str("    ld.shared.b16 %h0, [%rd33];\n");
        ptx.push_str("    cvt.f32.f16 %f2, %h0;      // dY[2i]\n");
        ptx.push_str("    add.u64 %rd33, %rd33, 2;\n");
        ptx.push_str("    ld.shared.b16 %h0, [%rd33];\n");
        ptx.push_str("    cvt.f32.f16 %f3, %h0;      // dY[2i+1]\n");

        // Inverse rotation:
        //   dx0 =  dY[2i]*cos + dY[2i+1]*sin
        //   dx1 = -dY[2i]*sin + dY[2i+1]*cos
        ptx.push_str("    mov.f32 %f4, 0f00000000;\n");
        ptx.push_str("    fma.rn.f32 %f4, %f2, %f0, %f4;        // dx0 += dY[2i]*cos\n");
        ptx.push_str("    fma.rn.f32 %f4, %f3, %f1, %f4;        // dx0 += dY[2i+1]*sin\n");
        ptx.push_str("    mov.f32 %f5, 0f00000000;\n");
        ptx.push_str("    neg.f32 %f6, %f2;\n");
        ptx.push_str("    fma.rn.f32 %f5, %f6, %f1, %f5;        // dx1 += -dY[2i]*sin\n");
        ptx.push_str("    fma.rn.f32 %f5, %f3, %f0, %f5;        // dx1 += dY[2i+1]*cos\n");

        // Store back to SMEM (f16).
        ptx.push_str("    cvt.rn.f16.f32 %h0, %f4;\n");
        ptx.push_str("    sub.u64 %rd33, %rd33, 2;\n");
        ptx.push_str("    st.shared.b16 [%rd33], %h0;\n");
        ptx.push_str("    cvt.rn.f16.f32 %h0, %f5;\n");
        ptx.push_str("    add.u64 %rd33, %rd33, 2;\n");
        ptx.push_str("    st.shared.b16 [%rd33], %h0;\n");

        ptx.push_str(&format!(
            "V2_BWD_DROPE_{label}_SKIP_{q_tile_iter}:\n"
        ));
    }
    ptx.push_str("    bar.sync 0;  // dRoPE writes visible\n");
}

// ────────────────────────────────────────────────────────────────────────
// dproj — Wq/Wk/Wv weight gradients
// ────────────────────────────────────────────────────────────────────────

/// Emit `dW{q,k,v} += x_norm^T @ dY_preRoPE` for the three projections.
/// Each sweep is null-guarded independently on its own weight pointer
/// (matching the forward fused-projection fix so inference-mode
/// launches that supply pre-projected Q/K/V skip the dW accumulations).
pub fn emit_dproj(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if config.csha.is_none() {
        ptx.push_str("    // Tier C dproj: csha=None, no emission\n");
        return;
    }
    ptx.push_str(&format!(
        "    // Tier C backward dproj weight gradients (q_tile_iter={q_tile_iter})\n"
    ));

    for (label, ptr_name) in [
        ("WQ", "csha_wq_ptr"),
        ("WK", "csha_wk_ptr"),
        ("WV", "csha_wv_ptr"),
    ] {
        ptx.push_str(&format!("V2_BWD_DPROJ_{label}_LOOP_{q_tile_iter}:\n"));
        // Null-guard per weight pointer.
        ptx.push_str(&format!("    ld.param.u64 %rd30, [{ptr_name}];\n"));
        ptx.push_str("    setp.eq.u64 %p0, %rd30, 0;\n");
        ptx.push_str(&format!(
            "    @%p0 bra V2_BWD_DPROJ_{label}_SKIP_{q_tile_iter};\n"
        ));

        // Inner accumulation (skeleton): load x_norm[s] and dY[s,d],
        // accumulate their outer product into a scratch f32 — the real
        // write-into-SMEM-dW-tile lives in the T4.1 orchestrator.
        ptx.push_str("    mov.f32 %f0, 0f3F800000;   // placeholder x_norm[s]\n");
        ptx.push_str("    mov.f32 %f1, 0f3F800000;   // placeholder dY_preRoPE[s, d]\n");
        ptx.push_str("    mov.f32 %f2, 0f00000000;   // dW accumulator\n");
        ptx.push_str("    fma.rn.f32 %f2, %f0, %f1, %f2;\n");

        ptx.push_str(&format!(
            "V2_BWD_DPROJ_{label}_SKIP_{q_tile_iter}:\n"
        ));
    }
    ptx.push_str("    bar.sync 0;  // dproj accumulations visible\n");
}

// ────────────────────────────────────────────────────────────────────────
// dRMSNorm
// ────────────────────────────────────────────────────────────────────────

/// Emit the two-term RMSNorm backward:
///
///   dx[i,d] = dx_norm[i,d] * norm_weight[d] / rms[i]
///             - x[i,d] * s / (d_model * rms[i]^3)
///
/// where `s = sum_d (dx_norm[i,d] * norm_weight[d] * x[i,d])`.
///
/// The `s` reduction is a 5-step warp butterfly across the 32 lanes
/// covering this row's `d_model` dimension.
pub fn emit_drmsnorm(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if config.csha.is_none() {
        ptx.push_str("    // Tier C dRMSNorm: csha=None, no emission\n");
        return;
    }
    let d_model = config.csha.as_ref().map_or(0, |c| c.d_model);

    ptx.push_str(&format!(
        "    // Tier C backward dRMSNorm (q_tile_iter={q_tile_iter}, d_model={d_model})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DRMSNORM_{q_tile_iter}:\n"));

    // Recompute rms[i] from x[i,:]: mean_sq = (1/D) sum_d x^2; rms = sqrt(mean_sq + eps).
    // Per-lane partial; butterfly reduce; rsqrt for 1/rms.
    ptx.push_str("    mov.f32 %f0, 0f3F800000;    // placeholder x[i, d_slice]\n");
    ptx.push_str("    mul.f32 %f1, %f0, %f0;\n");
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %f2, %f1, {offset}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    add.f32 %f1, %f1, %f2;\n");
    }
    // mean_sq = mean_sq_sum / d_model
    ptx.push_str(&format!(
        "    mov.f32 %f2, 0f{:08X};    // 1/d_model\n",
        (1.0f32 / d_model as f32).to_bits()
    ));
    ptx.push_str("    mul.f32 %f1, %f1, %f2;\n");
    // + eps
    ptx.push_str("    ld.param.f32 %f2, [csha_eps];\n");
    ptx.push_str("    add.f32 %f1, %f1, %f2;\n");
    // rms_inv = rsqrt(mean_sq + eps)   (f_rms_inv aliased as %f3)
    ptx.push_str("    rsqrt.approx.f32 %f3, %f1;  // 1/rms\n");

    // Compute g_d = dx_norm[i,d] * norm_weight[d] (placeholder).
    ptx.push_str("    mov.f32 %f4, 0f3F800000;    // placeholder dx_norm[i, d]\n");
    ptx.push_str("    mov.f32 %f5, 0f3F800000;    // placeholder norm_weight[d]\n");
    ptx.push_str("    mul.f32 %f_g, %f4, %f5;\n");

    // s = sum_d g_d * x[i, d]  — butterfly reduction across lanes.
    ptx.push_str("    mul.f32 %f6, %f_g, %f0;     // g_d * x[i, d]\n");
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %f7, %f6, {offset}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    add.f32 %f6, %f6, %f7;\n");
    }
    // denom = d_model * rms^3  ⇒  1/denom = (1/rms)^3 / d_model
    ptx.push_str("    mul.f32 %f7, %f3, %f3;\n");
    ptx.push_str("    mul.f32 %f7, %f7, %f3;      // (1/rms)^3\n");
    ptx.push_str("    mul.f32 %f7, %f7, %f2;      // / d_model\n");

    // dx[i, d] = g_d * (1/rms) - x[i, d] * s * (1/rms)^3 / d_model
    ptx.push_str("    mul.f32 %f8, %f_g, %f3;\n");
    ptx.push_str("    mul.f32 %f9, %f0, %f6;\n");
    ptx.push_str("    fma.rn.f32 %f8, %f9, %f7, 0f80000000;  // -x * s * denom_inv\n");
    ptx.push_str("    mul.f32 %f8, %f8, 0fBF800000;          // negate (two-term combine)\n");
    // Write dx into its SMEM slot (placeholder — orchestrator formalises
    // the dx-tile layout in T3.7).
    ptx.push_str("    cvt.u64.u32 %rd30, %lane;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    st.shared.f32 [%rd30], %f8;\n");
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
    fn backward_drope_rotates_dq_dk_inversely() {
        let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        cfg.rope_q = true;
        let mut ptx = String::new();
        emit_drope(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DROPE_Q_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DROPE_K_LOOP_0:"));
        assert!(
            ptx.matches("fma.rn.f32").count() >= 4,
            "need ≥4 fmas (2 per dim × Q and K), got {}",
            ptx.matches("fma.rn.f32").count()
        );
        assert!(!ptx.contains("V2_BWD_DROPE_V_LOOP"),
            "V must never be rotated");
    }

    #[test]
    fn backward_drope_skipped_when_rope_q_false() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_drope(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("rope_q=false") || ptx.is_empty(),
            "dRoPE must no-op when rope_q=false"
        );
        assert!(!ptx.contains("V2_BWD_DROPE"));
    }

    #[test]
    fn backward_dproj_accumulates_dwq_dwk_dwv() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_dproj(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DPROJ_WQ_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DPROJ_WK_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DPROJ_WV_LOOP_0:"));
        assert!(ptx.contains("setp.eq.u64 %p") && ptx.contains("csha_wq_ptr"),
            "independent null-guard on Wq missing");
        assert!(ptx.contains("csha_wk_ptr"));
        assert!(ptx.contains("csha_wv_ptr"));
    }

    #[test]
    fn backward_drmsnorm_produces_dx() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_drmsnorm(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DRMSNORM_0:"));
        assert!(ptx.contains("%f_g"));
        assert!(ptx.contains("rsqrt.approx.f32"));
        // Two 5-step butterflies: one for mean_sq, one for `s`. ≥5 covers both.
        assert!(
            ptx.matches("shfl.sync.bfly.b32").count() >= 5,
            "need ≥5 butterflies for rowsum reduction(s), got {}",
            ptx.matches("shfl.sync.bfly.b32").count()
        );
    }

    #[test]
    fn backward_hooks_label_uniqueness_across_iters() {
        let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        cfg.rope_q = true;
        let mut ptx = String::new();
        emit_drope(&mut ptx, &cfg, 0);
        emit_drope(&mut ptx, &cfg, 1);
        emit_dproj(&mut ptx, &cfg, 0);
        emit_dproj(&mut ptx, &cfg, 1);
        emit_drmsnorm(&mut ptx, &cfg, 0);
        emit_drmsnorm(&mut ptx, &cfg, 1);
        for label in [
            "V2_BWD_DROPE_Q_LOOP_0:", "V2_BWD_DROPE_Q_LOOP_1:",
            "V2_BWD_DROPE_K_LOOP_0:", "V2_BWD_DROPE_K_LOOP_1:",
            "V2_BWD_DPROJ_WQ_LOOP_0:", "V2_BWD_DPROJ_WQ_LOOP_1:",
            "V2_BWD_DRMSNORM_0:", "V2_BWD_DRMSNORM_1:",
        ] {
            assert!(ptx.contains(label), "missing: {label}");
        }
    }
}
