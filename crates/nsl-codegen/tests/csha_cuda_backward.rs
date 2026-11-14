//! GPU gate for the CSHA fused backward (`nsl_flash_attention_csha_backward`,
//! the scalar "Tier C" kernel) against an independent reference.
//!
//! What is compared: the eight gradients the kernel produces (dQ, dK, dV,
//! dWq, dWk, dWv, dx, dx_norm) and the forward output, against
//! `csha_f64_reference::fused_backward_f64` — an f64 evaluation from the
//! UNROUNDED inputs that is itself held to PyTorch float64 autograd literals
//! by CPU tests in that file.
//!
//! What the comparison is able to see, which its predecessor was not:
//!
//! * The incoming gradient dO is drawn at the magnitudes a real pretraining
//!   step has (1e-6, 1e-7), not O(1). The kernel used to read dO as f16 and
//!   store dQ/dK/dV/dW as f16; f16 is exactly zero below 2.98e-8, so at these
//!   magnitudes most of dQ was zero. The old gate drew dO of order 1, rounded
//!   it to f16, and handed the ROUNDED values to its CPU oracle too.
//! * Every tolerance is relative (`|gpu - ref|_2 / |ref|_2`). An absolute
//!   tolerance of 5e-3 passes any gradient of magnitude 1e-6, including zero.
//! * The reference does not share the kernel's rounding. Q, K, V and the
//!   weights still reach the kernel as f16 operands (weights narrowed by the
//!   launcher, projections saved as f16 by the forward), which is the
//!   remaining error and what the tolerances below are sized for.
//!
//! The file is `#[cfg(feature = "cuda")]` and its GPU tests are `#[ignore]`:
//!
//! ```text
//! cargo test -p nsl-codegen --release --features cuda \
//!     --test csha_cuda_backward -- --ignored --test-threads=1
//! ```

#![cfg(feature = "cuda")]

#[path = "csha_f64_reference.rs"]
mod csha_f64_reference;
use csha_f64_reference::{
    d_o_seq, det_seq, fused_backward_f64, gamma_seq, rel_l2, rope_tables, zero_fraction, FusedCase,
    Grads64,
};

use std::ffi::CString;

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2,
    smem_layout::{self, needs_dynamic_smem, Direction},
    synthesize_backward,
};

use nsl_runtime::flash_attention::{
    nsl_csha_alloc_backward_activations, nsl_csha_free_backward_activations,
    nsl_flash_attention_csha_backward, nsl_flash_attention_csha_with_saves,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
    nsl_test_cuda_jit_log,
};

const NORM_EPS: f32 = 1e-5;

fn f16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

fn f32_to_f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

fn free_all(ptrs: &[i64]) {
    for &p in ptrs {
        if p != 0 {
            nsl_test_cuda_free(p);
        }
    }
}

fn backward_kernel_name(cfg: &FlashAttentionConfig) -> String {
    let fw = flash_attention_kernel_name_v2(cfg);
    match fw.strip_prefix("flash_attn_") {
        Some(rest) => format!("flash_attn_backward_{rest}"),
        None => format!("flash_attn_backward_{fw}"),
    }
}

/// One fused-projection layer: a single head, square tiles of `block`.
/// The fused kernel pair reads `x` as rows `head_dim` wide, so the numerical
/// cases all have `d_model == head_dim`; a wider `d_model` is refused (see
/// `fused_kernels_refuse_a_d_model_above_head_dim`).
#[derive(Clone, Copy, Debug)]
struct Case {
    block: u32,
    head_dim: u32,
    d_model: u32,
    seq: usize,
    causal: bool,
    rope: bool,
}

/// Host-side inputs of a case, as the unrounded f32 the reference reads.
struct Inputs {
    x: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
    gamma: Vec<f32>,
    cos: Vec<f32>,
    sin: Vec<f32>,
}

impl Inputs {
    fn new(case: &Case) -> Self {
        let hd = case.head_dim as usize;
        let dm = case.d_model as usize;
        // Without RoPE the kernel still receives tables; the identity
        // rotation leaves Q and K alone.
        let (cos, sin) = if case.rope {
            rope_tables(case.seq, hd)
        } else {
            (vec![1.0f32; case.seq * hd / 2], vec![0.0f32; case.seq * hd / 2])
        };
        Inputs {
            x: det_seq(42, case.seq * dm),
            wq: det_seq(43, dm * hd),
            wk: det_seq(44, dm * hd),
            wv: det_seq(45, dm * hd),
            gamma: gamma_seq(dm),
            cos,
            sin,
        }
    }

    fn reference(&self, case: &Case, d_o: &[f32]) -> Grads64 {
        let hd = case.head_dim as usize;
        fused_backward_f64(&FusedCase {
            batch: 1,
            seq: case.seq,
            head_dim: hd,
            d_model: case.d_model as usize,
            causal: case.causal,
            eps: NORM_EPS,
            x: &self.x,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            gamma: &self.gamma,
            d_o,
            rope: case.rope.then_some((&self.cos, &self.sin)),
        })
    }
}

/// What the kernels wrote, read back as f32.
struct GpuRun {
    o: Vec<f32>,
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
    dwq: Vec<f32>,
    dwk: Vec<f32>,
    dwv: Vec<f32>,
    dx: Vec<f32>,
    dx_norm: Vec<f32>,
}

impl GpuRun {
    fn gradients(&self) -> [(&'static str, &[f32]); 8] {
        [
            ("dq", &self.dq),
            ("dk", &self.dk),
            ("dv", &self.dv),
            ("dwq", &self.dwq),
            ("dwk", &self.dwk),
            ("dwv", &self.dwv),
            ("dx", &self.dx),
            ("dx_norm", &self.dx_norm),
        ]
    }
}

fn reference_gradients(g: &Grads64) -> [(&'static str, &[f64]); 8] {
    [
        ("dq", &g.dq),
        ("dk", &g.dk),
        ("dv", &g.dv),
        ("dwq", &g.dwq),
        ("dwk", &g.dwk),
        ("dwv", &g.dwv),
        ("dx", &g.dx),
        ("dx_norm", &g.dx_norm),
    ]
}

/// Forward with saves, then the fused backward, through the two FFI entry
/// points with raw device buffers.
///
/// Buffer dtypes are the FFI contract for raw pointers: the weights and the
/// rotary tables are f16 and the forward output is f16 (the kernels' operand
/// formats); x, the RMSNorm weight, dO and every gradient are f32.
fn launch(case: &Case, inputs: &Inputs, d_o: &[f32]) -> Result<GpuRun, String> {
    let hd = case.head_dim as usize;
    let dm = case.d_model as usize;
    let seq = case.seq;
    let block = case.block as usize;
    assert!(seq >= block, "seq {seq} must cover at least one block");
    assert_eq!(d_o.len(), seq * hd);
    let scale = 1.0f32 / (hd as f32).sqrt();

    let config = FlashAttentionConfig {
        block_q: block as i64,
        block_kv: block as i64,
        head_dim: hd as i64,
        causal: case.causal,
        paged: false,
        rope_q: case.rope,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            fused_rmsnorm: true,
            fused_projections: true,
            fused_output_proj: false,
            save_activations_for_backward: true,
            active_heads: 1,
            rmsnorm_eps: NORM_EPS,
            d_model: dm as u32,
            skip_rmsnorm_prologue: false,
            static_seq_len: None,
        }),
        checkpoint: None,
    };
    smem_layout::validate_scalar_v2_config(&config, Direction::Backward)
        .map_err(|e| format!("backward validator: {e}"))?;

    let to_f16 = |v: &[f32]| -> Vec<u16> { v.iter().map(|&a| f32_to_f16_bits(a)).collect() };
    let wq_f16 = to_f16(&inputs.wq);
    let wk_f16 = to_f16(&inputs.wk);
    let wv_f16 = to_f16(&inputs.wv);
    let cos_f16 = to_f16(&inputs.cos);
    let sin_f16 = to_f16(&inputs.sin);

    nsl_cuda_init();
    let qkv_elems = seq * hd;
    let x_elems = seq * dm;
    let w_elems = dm * hd;
    let f16_bytes = |n: usize| (n * 2) as i64;
    let f32_bytes = |n: usize| (n * 4) as i64;

    // The forward kernel RMS-normalises x in place, so it gets its own copy.
    let x_dev = nsl_test_cuda_alloc(f32_bytes(x_elems));
    let nw_dev = nsl_test_cuda_alloc(f32_bytes(dm));
    let wq_dev = nsl_test_cuda_alloc(f16_bytes(w_elems));
    let wk_dev = nsl_test_cuda_alloc(f16_bytes(w_elems));
    let wv_dev = nsl_test_cuda_alloc(f16_bytes(w_elems));
    let cos_dev = nsl_test_cuda_alloc(f16_bytes(seq * hd / 2));
    let sin_dev = nsl_test_cuda_alloc(f16_bytes(seq * hd / 2));
    // q/k/v inputs are unused under fused projections but are part of the ABI.
    let q_dev = nsl_test_cuda_alloc(f16_bytes(qkv_elems));
    let k_dev = nsl_test_cuda_alloc(f16_bytes(qkv_elems));
    let v_dev = nsl_test_cuda_alloc(f16_bytes(qkv_elems));
    let out_dev = nsl_test_cuda_alloc(f16_bytes(qkv_elems));
    let lse_dev = nsl_test_cuda_alloc(f32_bytes(seq));
    let do_dev = nsl_test_cuda_alloc(f32_bytes(qkv_elems));
    let dq_dev = nsl_test_cuda_alloc(f32_bytes(qkv_elems));
    let dk_dev = nsl_test_cuda_alloc(f32_bytes(qkv_elems));
    let dv_dev = nsl_test_cuda_alloc(f32_bytes(qkv_elems));
    let dwq_dev = nsl_test_cuda_alloc(f32_bytes(w_elems));
    let dwk_dev = nsl_test_cuda_alloc(f32_bytes(w_elems));
    let dwv_dev = nsl_test_cuda_alloc(f32_bytes(w_elems));
    let dx_dev = nsl_test_cuda_alloc(f32_bytes(x_elems));
    let dxn_dev = nsl_test_cuda_alloc(f32_bytes(x_elems));
    let all_dev = [
        x_dev, nw_dev, wq_dev, wk_dev, wv_dev, cos_dev, sin_dev, q_dev, k_dev, v_dev, out_dev,
        lse_dev, do_dev, dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev, dx_dev, dxn_dev,
    ];
    let saves = unsafe { nsl_csha_alloc_backward_activations(1, 1, seq as i64, hd as i64) };

    nsl_test_cuda_h2d(x_dev, inputs.x.as_ptr() as i64, f32_bytes(x_elems));
    nsl_test_cuda_h2d(nw_dev, inputs.gamma.as_ptr() as i64, f32_bytes(dm));
    nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, f16_bytes(w_elems));
    nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, f16_bytes(w_elems));
    nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, f16_bytes(w_elems));
    nsl_test_cuda_h2d(cos_dev, cos_f16.as_ptr() as i64, f16_bytes(seq * hd / 2));
    nsl_test_cuda_h2d(sin_dev, sin_f16.as_ptr() as i64, f16_bytes(seq * hd / 2));
    nsl_test_cuda_h2d(do_dev, d_o.as_ptr() as i64, f32_bytes(qkv_elems));

    let cleanup = |saves| {
        unsafe { nsl_csha_free_backward_activations(saves) };
        free_all(&all_dev);
    };
    let jit_log = |ptx: &[u8]| -> String {
        let p = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
        if p != 0 {
            unsafe { std::ffi::CStr::from_ptr(p as *const i8) }.to_string_lossy().into_owned()
        } else {
            "<no log>".into()
        }
    };

    // Combined module: the fused kernel plus its `_mt_attn` twin, which the
    // runtime's two-launch dispatch resolves by name when seq exceeds a tile.
    let fwd_ptx = nsl_codegen::flash_attention_v2::synthesize_forward_multi_tile_combined(&config);
    let fwd_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    let fwd_smem =
        if needs_dynamic_smem(&config) { smem_layout::total_bytes(&config) as i64 } else { 0 };
    let causal = i64::from(case.causal);

    let rc_fwd = nsl_flash_attention_csha_with_saves(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        1, 1, seq as i64, hd as i64,
        0, 0, 0, 0,
        cos_dev, sin_dev,
        0, 0,
        fwd_smem,
        fwd_ptx.as_ptr() as i64, fwd_name.as_ptr() as i64,
        block as i64, block as i64,
        causal,
        x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
        0, NORM_EPS.to_bits() as i64,
        1, dm as i64,
        saves.q_proj, saves.k_proj, saves.v_proj,
        saves.row_max, saves.row_sum,
        saves.x_raw,
        0, // segment_ids
        0, 0, // Tier B variant
        0, // doc_starts
        0, // num_docs
    );
    if rc_fwd != 0 {
        let log = jit_log(&fwd_ptx);
        cleanup(saves);
        return Err(format!("forward rc={rc_fwd}\nJIT log:\n{log}"));
    }

    let mut bwd_ptx = synthesize_backward(&config).map_err(|e| format!("synth backward: {e}"))?;
    if !bwd_ptx.ends_with('\0') {
        bwd_ptx.push('\0');
    }
    let bwd_ptx = bwd_ptx.into_bytes();
    let bwd_name = CString::new(backward_kernel_name(&config)).unwrap();
    // The backward kernel declares `.extern .shared` once its tiles exceed
    // the static cap, and the launch must then supply the size.
    let bwd_smem = nsl_codegen::flash_attention_v2::shared_mem_bytes_v2_backward(&config);
    let bwd_smem_dyn =
        if bwd_smem > smem_layout::SMEM_BUDGET_BYTES { bwd_smem as i64 } else { 0 };

    let rc_bwd = nsl_flash_attention_csha_backward(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        1, 1, seq as i64, hd as i64,
        0, 0, 0, 0,
        cos_dev, sin_dev,
        0, 0,
        bwd_smem_dyn,
        bwd_ptx.as_ptr() as i64, bwd_name.as_ptr() as i64,
        block as i64, block as i64,
        causal,
        x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
        0, NORM_EPS.to_bits() as i64,
        1, dm as i64,
        saves.q_proj, saves.k_proj, saves.v_proj,
        saves.row_max, saves.row_sum,
        saves.x_raw,
        do_dev, dq_dev, dk_dev, dv_dev,
        dwq_dev, dwk_dev, dwv_dev, dx_dev,
        dxn_dev,
        0, // segment_ids
        0, 0, // Tier B variant
        0, // doc_starts
        0, // tier_b2_active: the scalar kernel
        0, // num_docs
    );
    if rc_bwd != 0 {
        let log = jit_log(&bwd_ptx);
        cleanup(saves);
        return Err(format!("backward rc={rc_bwd}\nJIT log:\n{log}"));
    }

    let read_f32 = |dev: i64, elems: usize| -> Vec<f32> {
        let mut out = vec![0f32; elems];
        nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dev, f32_bytes(elems));
        out
    };
    let o = {
        let mut raw = vec![0u16; qkv_elems];
        nsl_test_cuda_d2h(raw.as_mut_ptr() as i64, out_dev, f16_bytes(qkv_elems));
        raw.iter().map(|&b| f16_to_f32(b)).collect()
    };
    let run = GpuRun {
        o,
        dq: read_f32(dq_dev, qkv_elems),
        dk: read_f32(dk_dev, qkv_elems),
        dv: read_f32(dv_dev, qkv_elems),
        dwq: read_f32(dwq_dev, w_elems),
        dwk: read_f32(dwk_dev, w_elems),
        dwv: read_f32(dwv_dev, w_elems),
        dx: read_f32(dx_dev, x_elems),
        dx_norm: read_f32(dxn_dev, x_elems),
    };
    cleanup(saves);
    Ok(run)
}

/// Relative L2 error allowed on a gradient.
///
/// The gradients are f32 from end to end, so what is left is the operands'
/// precision: Q, K and V are saved by the forward as f16 and the weights are
/// narrowed to f16, a relative rounding of at most 2^-11 per element, plus
/// the kernel's `ex2.approx` / `div.approx` / `rsqrt.approx`. Measured on an
/// RTX PRO 4500 the worst gradient across every case in this file is 2.1e-3
/// (head_dim 64 with RoPE; 8e-4 to 1.1e-3 at head_dim 32), the same at every
/// magnitude of dO. With the f16 gradient path that this gate replaced, the
/// same cases read 4% to 13% at dO = 1e-6, 53% to 100% at 1e-7 and 100%
/// (every gradient all zero) at 1e-9.
const GRAD_REL_TOL: f64 = 4e-3;

/// Relative L2 error allowed on the forward output, which the kernel stores
/// as f16. Measured: 6e-4 to 1.3e-3.
const FWD_REL_TOL: f64 = 2e-3;

/// Compare one run with the reference. Returns the failures and prints one
/// line of measurements.
fn compare(label: &str, run: &GpuRun, want: &Grads64) -> Vec<String> {
    let mut failures = Vec::new();
    let fwd = rel_l2("o", &run.o, &want.o);
    let mut line = format!("[{label}] fwd={fwd:.2e}");
    if fwd.is_nan() || fwd >= FWD_REL_TOL {
        failures.push(format!("{label}: forward output off by {fwd:.3e} (tol {FWD_REL_TOL:.0e})"));
    }
    for ((name, got), (_, reference)) in run.gradients().into_iter().zip(reference_gradients(want)) {
        let err = rel_l2(name, got, reference);
        line.push_str(&format!(" {name}={err:.2e}"));
        if err.is_nan() || err >= GRAD_REL_TOL {
            failures.push(format!(
                "{label}: {name} off by {err:.3e} relative (tol {GRAD_REL_TOL:.0e}), \
                 {:.0}% of its entries exactly zero",
                100.0 * zero_fraction(got)
            ));
        }
    }
    eprintln!("{line}");
    failures
}

fn run_and_compare(case: &Case, magnitude: f32) -> Vec<String> {
    let inputs = Inputs::new(case);
    let d_o = d_o_seq(case.seq * case.head_dim as usize, magnitude);
    let label = format!(
        "hd={} d_model={} block={} seq={} causal={} rope={} dO~{magnitude:e}",
        case.head_dim, case.d_model, case.block, case.seq, case.causal as u8, case.rope as u8
    );
    match launch(case, &inputs, &d_o) {
        Ok(run) => compare(&label, &run, &inputs.reference(case, &d_o)),
        Err(e) => vec![format!("{label}: did not launch: {e}")],
    }
}

/// The headline gate: gradients at the magnitudes of a real pretraining step.
///
/// 1e-6 and 1e-7 are where the f16 path failed (57% error on dQ with 80% of
/// it exactly zero, then all of it); 1e-9 is below f16's smallest subnormal,
/// where every f16 gradient is zero. 1.0 is kept so the gate still covers the
/// one magnitude the old test used.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_backward_matches_reference_at_realistic_gradient_magnitudes() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let mut failures = Vec::new();
    for causal in [false, true] {
        let case = Case { block: 32, head_dim: 32, d_model: 32, seq: 32, causal, rope: false };
        for magnitude in [1.0f32, 1e-6, 1e-7, 1e-9] {
            failures.extend(run_and_compare(&case, magnitude));
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// The backward is linear in dO, and with f32 gradients a power-of-two
/// rescale of dO must rescale every output exactly: the same bits, shifted
/// exponent. Any f16 stage on the gradient path breaks this, because 2^-20
/// moves the values across f16's normal range.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_backward_is_bit_exact_under_a_power_of_two_rescale_of_do() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let case = Case { block: 32, head_dim: 32, d_model: 32, seq: 32, causal: true, rope: false };
    let inputs = Inputs::new(&case);
    let base = d_o_seq(case.seq * 32, 1e-2);
    let k = 2.0f32.powi(-20);
    let scaled: Vec<f32> = base.iter().map(|&v| v * k).collect();
    let a = launch(&case, &inputs, &base).expect("launch at 1e-2");
    let b = launch(&case, &inputs, &scaled).expect("launch at 1e-2 * 2^-20");
    for ((name, big), (_, small)) in a.gradients().into_iter().zip(b.gradients()) {
        assert!(big.iter().any(|v| *v != 0.0), "{name} is all zero");
        let differing = big
            .iter()
            .zip(small)
            .filter(|(x, y)| (**x * k).to_bits() != y.to_bits())
            .count();
        assert_eq!(
            differing, 0,
            "{name}: {differing} of {} entries are not the exact 2^-20 multiple",
            big.len()
        );
    }
}

/// Several query tiles and several key tiles: dK and dV accumulate across
/// query-tile launches, and dWk, dWv, dx and dx_norm come from a post-pass
/// over the totals.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_backward_matches_reference_over_four_tiles() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let mut failures = Vec::new();
    for head_dim in [32u32, 64] {
        for causal in [false, true] {
            for rope in [false, true] {
                let case = Case { block: 32, head_dim, d_model: head_dim, seq: 128, causal, rope };
                failures.extend(run_and_compare(&case, 1e-6));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// Single tile across head_dim, the causal mask and RoPE.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_backward_matches_reference_across_head_dim_causal_and_rope() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let mut failures = Vec::new();
    for head_dim in [32u32, 64] {
        for causal in [false, true] {
            for rope in [false, true] {
                let case = Case { block: 32, head_dim, d_model: head_dim, seq: 32, causal, rope };
                failures.extend(run_and_compare(&case, 1e-6));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}

/// `d_model` above `head_dim` is refused by both launchers.
///
/// The fused kernels read `x`, and the backward its saved copy, as rows
/// `head_dim` wide while indexing by `d_model`, and the save buffer is sized
/// by `head_dim`. At `d_model = 64`, `head_dim = 32` the forward output was
/// 126% off the reference and every gradient between 125% and 215%, with a
/// success return code. The launch must fail instead.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_kernels_refuse_a_d_model_above_head_dim() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA");
        return;
    }
    let case = Case { block: 32, head_dim: 32, d_model: 64, seq: 32, causal: true, rope: false };
    let inputs = Inputs::new(&case);
    let d_o = d_o_seq(case.seq * 32, 1e-6);
    let err = match launch(&case, &inputs, &d_o) {
        Ok(_) => panic!("a fused launch with d_model 64 and head_dim 32 was accepted"),
        Err(e) => e,
    };
    assert!(err.starts_with("forward rc=-1"), "expected the forward to refuse, got: {err}");

    // The backward on its own: it refuses on the kernel's name and the two
    // dimensions, before it touches a buffer or loads the module.
    let name = CString::new(
        "flash_attn_backward_p0_r0_hs_g1_c1_t0_q32_kv32_cshaL2_n1_p1_o0_h1_v2",
    )
    .unwrap();
    let rc = nsl_flash_attention_csha_backward(
        0, 0, 0, 0, 0,
        (1.0f32 / 32f32.sqrt()).to_bits() as i64,
        1, 1, 32, 32,
        0, 0, 0, 0,
        0, 0,
        0, 0,
        0,
        0, name.as_ptr() as i64,
        32, 32,
        1,
        0, 0, 0, 0, 0,
        0, NORM_EPS.to_bits() as i64,
        1, 64,
        0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0,
    );
    assert_eq!(rc, -1, "the backward must refuse d_model 64 with head_dim 32");
}
