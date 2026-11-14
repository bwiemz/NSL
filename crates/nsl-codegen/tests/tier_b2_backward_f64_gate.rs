//! GPU gate for the tensor-core CSHA backward ("Tier B.2": `d_prepass`,
//! `dq`, `dkdv`, `proj_backward`, launched by
//! `nsl_flash_attention_csha_backward` with `tier_b2_active = 1`) at the
//! gradient magnitudes of a real pretraining step.
//!
//! The dQ and dK/dV kernels run their matmuls on `mma.sync ... f16.f16`, so
//! the gradient operands dO and dS are converted to f16. Unscaled, that
//! conversion is exactly zero below 2.98e-8 and keeps a handful of mantissa
//! bits below 6.1e-5, which is where dO sits in training. The launcher
//! therefore multiplies each (batch, head) slice of dO by the power of two
//! that puts its largest element in [64, 128) before narrowing it, and
//! multiplies dQ, dK and dV by the reciprocal as they come back; the backward
//! is linear in dO, so the rescale is exact.
//!
//! The sibling gate `tier_b2_full_backward_cpu_reference.rs` multiplies dO up
//! "so the reference magnitudes clear the f16 noise floor", which is the one
//! regime where the missing prescale could not be seen. This file draws dO at
//! 1e-6 and 1e-7 and compares with `csha_f64_reference::fused_backward_f64`
//! (an f64 evaluation from the unrounded inputs, pinned to PyTorch) under
//! relative tolerances.
//!
//! The forward is not launched: Tier B.2 is selected only for one tile, one
//! head and `d_model == head_dim`, and the saves the backward reads (Q, K, V
//! and the attention output as f16, the softmax row statistics as f32) are
//! staged from the reference exactly as the forward kernel stores them.
//!
//! ```text
//! cargo test -p nsl-codegen --release --features cuda \
//!     --test tier_b2_backward_f64_gate -- --ignored --test-threads=1
//! ```

#![cfg(feature = "cuda")]

#[path = "csha_f64_reference.rs"]
mod csha_f64_reference;
use csha_f64_reference::{
    d_o_seq, det_seq, fused_backward_f64, gamma_seq, rel_l2, zero_fraction, FusedCase, Grads64,
};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::shared_mem_bytes_v2_backward;
use nsl_codegen::flash_attention_v2::tier_b2::backward::synthesize_tier_b2_backward;
use nsl_runtime::flash_attention::nsl_flash_attention_csha_backward;
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
};

const NORM_EPS: f32 = 1e-5;

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

/// The only shape Tier B.2 is dispatched for: one tile, one head,
/// `d_model == head_dim`, causal.
fn tier_b2_config(head_dim: usize, block: usize) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: block as i64,
        block_kv: block as i64,
        head_dim: head_dim as i64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            fused_rmsnorm: true,
            fused_projections: true,
            fused_output_proj: false,
            save_activations_for_backward: true,
            active_heads: 1,
            rmsnorm_eps: NORM_EPS,
            d_model: head_dim as u32,
            skip_rmsnorm_prologue: false,
            static_seq_len: None,
        }),
        checkpoint: None,
    }
}

struct Inputs {
    x: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    gamma: Vec<f32>,
}

impl Inputs {
    fn new(seq: usize, hd: usize) -> Self {
        // Weights at a quarter of det_seq's range keep the attention logits
        // of order 1, so the attention weights P stay in f16's normal range.
        // P is an f16 operand of the tensor-core matmuls too, and this gate
        // is about the gradient operands, not about P.
        let w = |seed: u32| -> Vec<f32> {
            det_seq(seed, hd * hd).into_iter().map(|u| 0.25 * u).collect()
        };
        Inputs { x: det_seq(42, seq * hd), wq: w(43), wk: w(44), wv: w(45), gamma: gamma_seq(hd) }
    }

    fn reference(&self, seq: usize, hd: usize, d_o: &[f32]) -> Grads64 {
        fused_backward_f64(&FusedCase {
            batch: 1,
            seq,
            head_dim: hd,
            d_model: hd,
            causal: true,
            eps: NORM_EPS,
            x: &self.x,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            gamma: &self.gamma,
            d_o,
            rope: None,
        })
    }
}

fn upload<T: Copy>(host: &[T]) -> i64 {
    let bytes = std::mem::size_of_val(host) as i64;
    let dev = nsl_test_cuda_alloc(bytes);
    assert_ne!(dev, 0, "device allocation failed");
    nsl_test_cuda_h2d(dev, host.as_ptr() as i64, bytes);
    dev
}

fn to_f16(v: &[f64]) -> Vec<u16> {
    v.iter().map(|&a| half::f16::from_f64(a).to_bits()).collect()
}

fn f32_to_f16(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&a| half::f16::from_f32(a).to_bits()).collect()
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|&a| a as f32).collect()
}

fn read_f32(dev: i64, elems: usize) -> Vec<f32> {
    let mut out = vec![0f32; elems];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dev, (elems * 4) as i64);
    out
}

/// Launch the four-kernel backward and read back the eight gradients, in
/// the order dq, dk, dv, dwq, dwk, dwv, dx, dx_norm.
fn launch(
    cfg: &FlashAttentionConfig,
    inputs: &Inputs,
    reference: &Grads64,
    d_o: &[f32],
    seq: usize,
) -> [Vec<f32>; 8] {
    let hd = cfg.head_dim as usize;
    let qkv = seq * hd;
    let w = hd * hd;

    // Forward saves, in the formats the forward kernel writes them.
    let q_dev = upload(&to_f16(&reference.q));
    let k_dev = upload(&to_f16(&reference.k));
    let v_dev = upload(&to_f16(&reference.v));
    let o_dev = upload(&to_f16(&reference.o));
    let rmax_dev = upload(&to_f32(&reference.row_max));
    let rsum_dev = upload(&to_f32(&reference.row_sum));
    // Projection inputs: x and the RMSNorm weight f32, the weights f16.
    let xraw_dev = upload(&inputs.x);
    let nw_dev = upload(&inputs.gamma);
    let wq_dev = upload(&f32_to_f16(&inputs.wq));
    let wk_dev = upload(&f32_to_f16(&inputs.wk));
    let wv_dev = upload(&f32_to_f16(&inputs.wv));
    // The gradient in, f32, and the eight gradients out, f32.
    let do_dev = upload(d_o);
    let zeros = |n: usize| upload(&vec![0f32; n]);
    let outs = [zeros(qkv), zeros(qkv), zeros(qkv), zeros(w), zeros(w), zeros(w), zeros(qkv), zeros(qkv)];

    let mut ptx = synthesize_tier_b2_backward(cfg).expect("synthesize_tier_b2_backward").into_bytes();
    if ptx.last() != Some(&0) {
        ptx.push(0);
    }
    let rc = nsl_flash_attention_csha_backward(
        0, 0, 0,
        o_dev,
        0,
        (1.0f32 / (hd as f32).sqrt()).to_bits() as i64,
        1, 1, seq as i64, hd as i64,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        shared_mem_bytes_v2_backward(cfg) as i64,
        ptx.as_ptr() as i64,
        0,
        cfg.block_q, cfg.block_kv,
        1,
        0, nw_dev,
        wq_dev, wk_dev, wv_dev,
        0,
        NORM_EPS.to_bits() as i64,
        1, hd as i64,
        q_dev, k_dev, v_dev,
        rmax_dev, rsum_dev,
        xraw_dev,
        do_dev,
        outs[0], outs[1], outs[2],
        outs[3], outs[4], outs[5],
        outs[6], outs[7],
        0,
        0, 0,
        0,
        1, // tier_b2_active
        0,
    );
    let sizes = [qkv, qkv, qkv, w, w, w, qkv, qkv];
    let read: [Vec<f32>; 8] = std::array::from_fn(|i| read_f32(outs[i], sizes[i]));
    for p in [
        q_dev, k_dev, v_dev, o_dev, rmax_dev, rsum_dev, xraw_dev, nw_dev, wq_dev, wk_dev, wv_dev,
        do_dev,
    ]
    .into_iter()
    .chain(outs)
    {
        nsl_test_cuda_free(p);
    }
    assert_eq!(rc, 0, "Tier B.2 backward launch failed, rc={rc}");
    read
}

const NAMES: [&str; 8] = ["dq", "dk", "dv", "dwq", "dwk", "dwv", "dx", "dx_norm"];

/// Relative L2 error allowed on a gradient.
///
/// dO, dS, P, Q, K and V are all f16 operands of the tensor-core matmuls
/// here (11 significant bits each once the gradient operands are prescaled),
/// so the floor is a few 1e-4 per operand; measured on an RTX PRO 4500 the
/// worst gradient is below 2e-3. Without the prescale dQ is 60% off at
/// dO = 1e-6 and entirely zero at 1e-7.
const REL_TOL: f64 = 5e-3;

fn check(label: &str, got: &[Vec<f32>; 8], want: &Grads64) -> Vec<String> {
    let wants: [&[f64]; 8] =
        [&want.dq, &want.dk, &want.dv, &want.dwq, &want.dwk, &want.dwv, &want.dx, &want.dx_norm];
    let mut failures = Vec::new();
    let mut line = format!("[{label}]");
    for ((name, g), w) in NAMES.into_iter().zip(got).zip(wants) {
        let err = rel_l2(name, g, w);
        line.push_str(&format!(" {name}={err:.2e}"));
        if err.is_nan() || err >= REL_TOL {
            failures.push(format!(
                "{label}: {name} off by {err:.3e} relative (tol {REL_TOL:.0e}), {:.0}% of its \
                 entries exactly zero",
                100.0 * zero_fraction(g)
            ));
        }
    }
    eprintln!("{line}");
    failures
}

/// Every gradient of the four-kernel backward, across the magnitudes a
/// gradient takes: 1.0 (the only regime the older gate covered), 1e-6 and
/// 1e-7 (pretraining), and 1e-9 (below f16's smallest subnormal).
#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b2_backward_matches_reference_at_realistic_gradient_magnitudes() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let mut failures = Vec::new();
    // hd=128 runs at block 32: it keeps the backward's shared memory under
    // the device cap and matches the kernels' effective tile at that width.
    for (hd, seq) in [(64usize, 64usize), (128, 32)] {
        let cfg = tier_b2_config(hd, seq);
        let inputs = Inputs::new(seq, hd);
        for magnitude in [1.0f32, 1e-6, 1e-7, 1e-9] {
            let d_o = d_o_seq(seq * hd, magnitude);
            let want = inputs.reference(seq, hd, &d_o);
            let got = launch(&cfg, &inputs, &want, &d_o, seq);
            failures.extend(check(&format!("hd={hd} seq={seq} dO~{magnitude:e}"), &got, &want));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The prescale is chosen from the exponent of the largest |dO| only, so a
/// power-of-two rescale of dO changes nothing but the exponents: every
/// output must be the exact multiple, bit for bit.
#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b2_backward_is_bit_exact_under_a_power_of_two_rescale_of_do() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let (hd, seq) = (64usize, 64usize);
    let cfg = tier_b2_config(hd, seq);
    let inputs = Inputs::new(seq, hd);
    let base = d_o_seq(seq * hd, 1e-2);
    let k = 2.0f32.powi(-20);
    let scaled: Vec<f32> = base.iter().map(|&v| v * k).collect();
    let want = inputs.reference(seq, hd, &base);
    let a = launch(&cfg, &inputs, &want, &base, seq);
    let b = launch(&cfg, &inputs, &want, &scaled, seq);
    for ((name, big), small) in NAMES.into_iter().zip(&a).zip(&b) {
        assert!(big.iter().any(|v| *v != 0.0), "{name} is all zero");
        let differing =
            big.iter().zip(small).filter(|(x, y)| (**x * k).to_bits() != y.to_bits()).count();
        assert_eq!(
            differing, 0,
            "{name}: {differing} of {} entries are not the exact 2^-20 multiple",
            big.len()
        );
    }
}
