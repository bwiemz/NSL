//! Tests for cpu_naive_backward_dq (Phase 2.6 T5).
use half::f16;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_test::cpu_naive_backward::cpu_naive_backward_dq;
use nsl_test::cpu_naive_forward::cpu_naive_forward;

fn cfg_hd(d: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: d, causal: false, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
    }
}

#[test]
fn cpu_naive_backward_dq_finite_and_correct_shape() {
    let q: Vec<f16> = vec![1.0, 0.0, 0.0, 1.0].into_iter().map(f16::from_f32).collect();
    let k = q.clone(); let v = q.clone(); let d_o = q.clone();
    let fwd = cpu_naive_forward(&q, &k, &v, 1, 1, 2, 2, false);
    let cfg = cfg_hd(2);
    let dq = cpu_naive_backward_dq(&fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, 1, 1, 2, &cfg);
    assert_eq!(dq.len(), 4);
    for (i, &x) in dq.iter().enumerate() { assert!(x.is_finite(), "dq[{}]={} not finite", i, x); }
}

#[test]
fn cpu_naive_backward_dq_matches_inline_reference_on_random_config() {
    let b = 1usize; let h = 1usize; let s = 8usize; let d = 4usize;
    let q: Vec<f16> = (0..b*h*s*d).map(|i| f16::from_f32((i as f32 * 0.137).sin() * 0.1)).collect();
    let k: Vec<f16> = (0..b*h*s*d).map(|i| f16::from_f32((i as f32 * 0.211).cos() * 0.1)).collect();
    let v: Vec<f16> = (0..b*h*s*d).map(|i| f16::from_f32((i as f32 * 0.317).sin() * 0.1)).collect();
    let d_o: Vec<f16> = (0..b*h*s*d).map(|i| f16::from_f32((i as f32 * 0.419).cos() * 0.1)).collect();
    let fwd = cpu_naive_forward(&q, &k, &v, b, h, s, d, false);
    let cfg = cfg_hd(d as i64);
    let dq = cpu_naive_backward_dq(&fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, b, h, s, &cfg);
    let dq_inline = inline_naive_dq(&q, &k, &v, &fwd.o, &d_o, b, h, s, d);
    let max_abs = dq.iter().zip(dq_inline.iter()).map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max);
    assert!(max_abs < 1e-5, "max_abs={} between fn and inline reference", max_abs);
}

// Standalone inline reference -- intentionally takes the same wide arg list as the
// function under test, and indexes ds[ki] by the flat-strided loop var.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
fn inline_naive_dq(q: &[f16], k: &[f16], v: &[f16], o: &[f16], d_o: &[f16],
                   b: usize, h: usize, s: usize, d: usize) -> Vec<f32> {
    let mut dq = vec![0.0f32; b*h*s*d];
    let scale = 1.0f32 / (d as f32).sqrt();
    for bi in 0..b { for hi in 0..h { for qi in 0..s {
        let qbase = ((bi*h+hi)*s+qi)*d;
        let s_row: Vec<f32> = (0..s).map(|ki| {
            let kb=((bi*h+hi)*s+ki)*d;
            let dot: f32 = (0..d).map(|di| q[qbase+di].to_f32()*k[kb+di].to_f32()).sum();
            dot*scale
        }).collect();
        let rmax = s_row.iter().fold(f32::NEG_INFINITY, |a,&x| a.max(x));
        let rsum: f32 = s_row.iter().map(|&x| (x-rmax).exp()).sum();
        let p: Vec<f32> = s_row.iter().map(|&x| (x-rmax).exp()/rsum).collect();
        let dl: f32 = (0..d).map(|di| d_o[qbase+di].to_f32()*o[qbase+di].to_f32()).sum();
        let dp: Vec<f32> = (0..s).map(|ki| {
            let kb=((bi*h+hi)*s+ki)*d;
            (0..d).map(|di| d_o[qbase+di].to_f32()*v[kb+di].to_f32()).sum()
        }).collect();
        let ds: Vec<f32> = (0..s).map(|ki| scale*p[ki]*(dp[ki]-dl)).collect();
        for ki in 0..s { let kb=((bi*h+hi)*s+ki)*d; for di in 0..d { dq[qbase+di]+=ds[ki]*k[kb+di].to_f32(); } }
    }}}
    dq
}
