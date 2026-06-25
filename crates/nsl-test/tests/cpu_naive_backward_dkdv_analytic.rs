//! CPU-only analytic check for `cpu_naive_backward_dkdv` (Phase 3a Task 9).
//!
//! Lives in nsl-test's integration tests (NOT the cuda-gated nsl-codegen
//! `tier_b2_dkdv_kernel_cpu_reference.rs`) so the CPU reference math is
//! regression-tested in a standard `cargo test` run, with no GPU/cuda feature.

use half::f16;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_test::cpu_naive_backward::cpu_naive_backward_dkdv;

fn cfg(hd: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: hd,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
        checkpoint: None,
    }
}

#[test]
fn cpu_naive_dkdv_s1_collapses_to_dv_eq_do_and_dk_zero() {
    // At seq=1 the softmax collapses: P=[[1]], O = P@V = V, so
    //   d_local = sum_d dO*O = sum_d dO*V = dP[0,0]  =>  dS = scale*P*(dP - d_local) = 0.
    // Therefore dV[0,:] = P[0,0]*dO[0,:] = dO[0,:]  and  dK[0,:] = dS*Q = 0, exactly.
    let hd = 4usize;
    let to16 = |xs: &[f32]| xs.iter().map(|&x| f16::from_f32(x)).collect::<Vec<_>>();
    let q = to16(&[0.1, 0.2, 0.3, 0.4]);
    let k = to16(&[0.5, 0.6, 0.7, 0.8]);
    let v = to16(&[1.0, 2.0, 3.0, 4.0]);
    let o = v.clone(); // seq=1: O = P@V = V
    let d_o = to16(&[0.9, 0.8, 0.7, 0.6]);
    let cfg = cfg(hd as i64);

    let (dv, dk) = cpu_naive_backward_dkdv(&q, &k, &v, &o, &d_o, 1, 1, 1, &cfg);

    for di in 0..hd {
        assert!(
            (dv[di] - d_o[di].to_f32()).abs() < 1e-6,
            "s=1: dV must equal dO at d={di} (got {} vs {})",
            dv[di],
            d_o[di].to_f32()
        );
        assert!(dk[di].abs() < 1e-6, "s=1: dK must be 0 at d={di} (got {})", dk[di]);
    }
}
