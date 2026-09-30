//! PCA Stage C on silicon: the packed (segment-masked) fused SDPA forward and
//! backward, launched through the runtime FFIs exactly as a packed train step
//! launches them, against f64 oracles — outputs and RAW gradients, on a
//! geometry where documents straddle KV tiles.
//!
//! # Why this file exists
//!
//! Until now the packed path's silicon evidence was
//! `stage_c_packed_parity.rs::packed_fused_training_smoke_on_gpu` (until this
//! change `packed_fused_matches_decomposed_on_gpu`): one epoch at seq 64
//! (16 micro-batches, 8 optimizer steps, one KV tile), fused against
//! decomposed, checkpoint-compared at 2e-2. Eight clipped optimizer steps
//! average a gradient error away, and a single tile never exercises the
//! cross-tile softmax state or the m-tile walk. The numerics
//! were proved only on the CTA interpreter (`sdpa_fused_forward_interp.rs`,
//! `sdpa_fused_backward_interp.rs`), which is not the hardware: ptxas may
//! contract, the MMA units round as they round, and warps race as they race.
//! Every GPU gate of this backward (`synthesize_flash_attention_backward_ptx`)
//! runs `segment_masked: false`, against `flash_attention_backward_cpu`, the
//! runtime's own CPU fallback. The packed backward gates in the hardware
//! bundle test other kernels: the CSHA v2 Tier-A backward (against itself,
//! run unpacked) and the per-document CTA kernels.
//!
//! # What runs
//!
//! * The production PTX and launch parameters for a decorator-free packed
//!   train compile at the default target (`cuda` → `gpu_sm = 80`, the MMA
//!   backward): the forward config `ensure_sdpa_fwd_variant_table` admits
//!   (first of its tile candidates that validates), its base and Tier-B-on
//!   kernels with the shared-memory request widened for the Tier-B range
//!   table, and the backward config `ensure_sdpa_bwd_variant_table` builds.
//! * `S = 448`: seven 64-wide tiles, not a multiple of 128, and above the
//!   Tier-B floor, so production's forward takes the Tier-B-on kernel. 100-
//!   token documents at a different phase per batch row put boundaries
//!   mid-tile (row 0 at 100/200/300/400, row 1 one token before a tile edge).
//! * Oracles in f64, independent of the runtime: the exact gradients of
//!   causal-within-document softmax attention, and the same with every MMA
//!   operand rounded to f16 where the kernel rounds it (the interpreter
//!   gate's oracle, which pins the storage story).
//! * The launch census proves each kernel ran. The backward FFI falls back
//!   to its CPU reference on any refusal (a missing logsumexp, a view input,
//!   a failed launch), and the fallback's gradients are close enough to pass
//!   a loose bound, so a gate without this proof can certify the CPU.
//!
//! The training comparison in `stage_c_packed_parity.rs` stays, as the
//! integration smoke it is.
//!
//! ```bash
//! cargo test -p nsl-codegen --features cuda,test-hooks \
//!     --test sdpa_fused_packed_gpu_parity -- --ignored --test-threads=1
//! ```

#![cfg(all(feature = "cuda", feature = "test-hooks"))]

use std::ffi::CString;

use half::f16;

use nsl_codegen::flash_attention::{
    backward_select_blocks, flash_attention_bwd_d_kernel_name, flash_attention_bwd_main_kernel_name,
    synthesize_flash_attention_backward_ptx, FlashAttentionBackwardConfig, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_selector::shared_mem_bytes_selected;
use nsl_codegen::flash_attention_v2::smem_layout::{tier_b_range_table_offset, validate_scalar_v2_config, Direction};
use nsl_codegen::pca_tier_b::emit_tier_b_variants_for_config;
use nsl_codegen::pca_tilerange::tier_b_range_table_bytes;

use nsl_runtime::flash_attention::{nsl_flash_attention_backward, nsl_sdpa_fused_forward, nsl_sdpa_fused_launch_count};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::pca_tier_b_runtime::TIER_B_MAX_BAKED_SEQ_LEN;
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_to_device, nsl_tensor_zeros_on};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d, nsl_test_cuda_jit_log, test_kernel_launch_census_arm,
    test_kernel_launch_count,
};

const B: usize = 2;
const H: usize = 2;
const S: usize = 448;
const D: usize = 32;
const DOC_LEN: usize = 100;

/// `parse_gpu_sm_from_target("cuda")`: what a default `nsl build` embeds.
const GPU_SM: u32 = 80;

/// The forward tile candidates `ensure_sdpa_fwd_variant_table` probes, in order.
const FWD_TILE_CANDIDATES: [(i64, i64); 4] = [(64, 64), (32, 32), (32, 16), (16, 16)];

fn init_cuda() {
    assert_eq!(nsl_cuda_init(), 0, "nsl_cuda_init failed; this gate needs a CUDA device");
}

/// Deterministic values in about [-2, 2) (the interpreter gates' generator).
fn values(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((state >> 33) as u32 as f64 / u32::MAX as f64) * 4.0 - 2.0) as f32
        })
        .collect()
}

/// `DOC_LEN`-token documents, the second row shifted by 37 tokens.
fn segments() -> Vec<u16> {
    (0..B).flat_map(|b| (0..S).map(move |i| ((i + 37 * b) / DOC_LEN) as u16)).collect()
}

struct Grads {
    dq: Vec<f32>,
    dk: Vec<f32>,
    dv: Vec<f32>,
}

/// The forward (O, natural logsumexp) and the gradients, in f64. With
/// `staged`, every operand of the kernel's five MMA products is rounded to
/// f16 first: Q and K in S, dO and V in dP, P and dO in dV, dS and K in dQ,
/// dS and Q in dK. O, D and the logsumexp stay exact — the kernel reads
/// them in f32. The same oracle as `sdpa_fused_backward_interp.rs`.
fn oracle(q: &[f32], k: &[f32], v: &[f32], dout: &[f32], seg: &[u16], staged: bool) -> (Vec<f32>, Vec<f32>, Grads) {
    let r = |x: f64| if staged { f16::from_f64(x).to_f64() } else { x };
    let scale = 1.0 / (D as f64).sqrt();
    let n = B * H * S * D;
    let (mut out, mut lse) = (vec![0f32; n], vec![0f32; B * H * S]);
    let (mut dq, mut dk, mut dv) = (vec![0f64; n], vec![0f64; n], vec![0f64; n]);
    for b in 0..B {
        for h in 0..H {
            let base = (b * H + h) * S * D;
            let at = |x: &[f32], i: usize, d: usize| x[base + i * D + d] as f64;
            for i in 0..S {
                let vis: Vec<usize> = (0..=i).filter(|&j| seg[b * S + i] == seg[b * S + j]).collect();
                let s_exact: Vec<f64> =
                    vis.iter().map(|&j| (0..D).map(|d| at(q, i, d) * at(k, j, d)).sum::<f64>() * scale).collect();
                let m = s_exact.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let l = m + s_exact.iter().map(|s| (s - m).exp()).sum::<f64>().ln();
                lse[(b * H + h) * S + i] = l as f32;
                let o: Vec<f64> = (0..D)
                    .map(|d| vis.iter().zip(&s_exact).map(|(&j, s)| (s - l).exp() * at(v, j, d)).sum())
                    .collect();
                for d in 0..D {
                    out[base + i * D + d] = o[d] as f32;
                }
                let (o32, l32): (Vec<f64>, f64) = (o.iter().map(|&x| x as f32 as f64).collect(), l as f32 as f64);
                let dd: f64 = (0..D).map(|d| at(dout, i, d) * o32[d]).sum();
                for &j in &vis {
                    let s = (0..D).map(|d| r(at(q, i, d)) * r(at(k, j, d))).sum::<f64>() * scale;
                    let p = (s - l32).exp();
                    let dp: f64 = (0..D).map(|d| r(at(dout, i, d)) * r(at(v, j, d))).sum();
                    let ds = p * (dp - dd);
                    for d in 0..D {
                        dv[base + j * D + d] += r(p) * r(at(dout, i, d));
                        dq[base + i * D + d] += r(ds) * r(at(k, j, d)) * scale;
                        dk[base + j * D + d] += r(ds) * r(at(q, i, d)) * scale;
                    }
                }
            }
        }
    }
    let f = |x: Vec<f64>| x.into_iter().map(|v| v as f32).collect();
    (out, lse, Grads { dq: f(dq), dk: f(dk), dv: f(dv) })
}

/// max |got − want| over max |want|.
fn rel(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, x| m.max(x.abs()));
    assert!(scale > 0.0, "an all-zero reference");
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max) / scale
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    got.iter().zip(want).map(|(g, w)| (g - w).abs()).fold(0f32, f32::max)
}

fn gpu_tensor(shape: &[i64], vals: &[f32]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1);
    nsl_list_free(shape_list);
    assert_ne!(t, 0, "GPU tensor alloc failed");
    nsl_test_cuda_h2d(nsl_tensor_data_ptr(t), vals.as_ptr() as i64, (vals.len() * 4) as i64);
    t
}

fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, nsl_tensor_data_ptr(t), (len * 4) as i64);
    out
}

/// Minimal mirror of the runtime's `#[repr(C)]` `NslTensor` header, to read
/// the device tag (the same mirror `gpu_dtype_refusal.rs` carries).
#[repr(C)]
struct TensorHeader {
    _magic: u32,
    _data: *mut std::ffi::c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    _ndim: i64,
    _len: i64,
    _refcount: std::sync::atomic::AtomicI64,
    device: u8,
}

fn tensor_device(ptr: i64) -> u8 {
    unsafe { (*(ptr as *const TensorHeader)).device }
}

/// The `[B, S]` segment-id tensor as the attention ops receive it in a GPU
/// packed step: `packing.rs` emits f32 ids on the host, and
/// `nsl_packed_batch_align_device` moves them to the parameters' device
/// before anything consumes them. So the runtime's device-to-host staging
/// branch (`segment_ids_host_u16`, `device > 0`) is the one a real step takes.
fn device_segment_tensor(seg: &[u16]) -> i64 {
    let shape_list = nsl_list_new();
    nsl_list_push(shape_list, B as i64);
    nsl_list_push(shape_list, S as i64);
    let t = nsl_tensor_zeros_on(shape_list, 0);
    nsl_list_free(shape_list);
    assert_ne!(t, 0, "CPU tensor alloc failed");
    let p = nsl_tensor_data_ptr(t) as *mut f32;
    for (i, &v) in seg.iter().enumerate() {
        unsafe { *p.add(i) = f32::from(v) };
    }
    let on_device = nsl_tensor_to_device(t, 1);
    nsl_tensor_free(t);
    assert_eq!(tensor_device(on_device), 1, "segment ids did not reach the device");
    on_device
}

fn jit_log(ptx: &[u8]) -> String {
    let p = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
    if p == 0 {
        return "<no log>".into();
    }
    unsafe { std::ffi::CStr::from_ptr(p as *const std::ffi::c_char).to_string_lossy().into_owned() }
}

/// NUL-terminated PTX, as `cuModuleLoadData` wants it.
fn terminated(mut ptx: Vec<u8>) -> Vec<u8> {
    while ptx.last() == Some(&0) {
        ptx.pop();
    }
    if ptx.last() != Some(&b'\n') {
        ptx.push(b'\n');
    }
    ptx.push(0);
    ptx
}

/// The forward config `ensure_sdpa_fwd_variant_table` admits for head_dim `D`.
fn forward_config() -> FlashAttentionConfig {
    FWD_TILE_CANDIDATES
        .iter()
        .map(|&(block_q, block_kv)| FlashAttentionConfig {
            block_q,
            block_kv,
            head_dim: D as i64,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: GPU_SM,
            segment_masked: true,
            csha: None,
            checkpoint: None,
        })
        .find(|c| validate_scalar_v2_config(c, Direction::Forward).is_ok())
        .expect("the packed forward admits a tile at head_dim 32")
}

/// The backward config `ensure_sdpa_bwd_variant_table` builds for head_dim `D`.
fn backward_config() -> FlashAttentionBackwardConfig {
    let (block_q, block_kv) = backward_select_blocks(D as i64);
    FlashAttentionBackwardConfig { block_q, block_kv, head_dim: D as i64, causal: true, gpu_sm: GPU_SM, segment_masked: true }
}

struct Inputs {
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    dout: Vec<f32>,
    seg: Vec<u16>,
}

fn inputs() -> Inputs {
    let n = B * H * S * D;
    Inputs { q: values(1, n), k: values(2, n), v: values(3, n), dout: values(4, n), seg: segments() }
}

/// Launch the fused forward. With `tier_b`, pass the Tier-B-on pair as
/// production does (the runtime then selects it at this seq); without, the
/// base kernel. Returns (out, lse) and asserts which variant the runtime
/// counted.
fn fused_forward(x: &Inputs, tier_b: bool) -> (Vec<f32>, Vec<f32>) {
    let config = forward_config();
    let emission = emit_tier_b_variants_for_config(&config);
    let base_ptx = terminated(emission.base_ptx.clone());
    // The Tier-B-on module names its entry with the base name (see
    // `ensure_sdpa_fwd_variant_table`).
    let name = CString::new(emission.base_kernel_name.clone()).unwrap();
    let tb_ptx = tier_b.then(|| terminated(emission.tier_b_on_ptx.clone().expect("the packed config emits Tier-B")));
    // One request serves both kernels: the base request widened to cover the
    // Tier-B range table at the baked maximum seq, as production sizes it.
    let tb_off = tier_b_range_table_offset(&config, Direction::Forward);
    let smem = (shared_mem_bytes_selected(&config) as i64)
        .max((tb_off + tier_b_range_table_bytes(&config, TIER_B_MAX_BAKED_SEQ_LEN)) as i64);

    let shape = [B as i64, H as i64, S as i64, D as i64];
    let (q_t, k_t, v_t) = (gpu_tensor(&shape, &x.q), gpu_tensor(&shape, &x.k), gpu_tensor(&shape, &x.v));
    let seg_t = device_segment_tensor(&x.seg);
    let variant = usize::from(tier_b) as i64;
    let before = nsl_sdpa_fused_launch_count(variant);
    let scale = 1.0f32 / (D as f32).sqrt();
    let list = nsl_sdpa_fused_forward(
        q_t,
        k_t,
        v_t,
        scale.to_bits() as i64,
        1,
        seg_t,
        base_ptx.as_ptr() as i64,
        name.as_ptr() as i64,
        tb_ptx.as_ref().map_or(0, |p| p.as_ptr() as i64),
        if tier_b { name.as_ptr() as i64 } else { 0 },
        config.block_q,
        config.block_kv,
        smem,
    );
    if list == 0 {
        panic!(
            "nsl_sdpa_fused_forward declined (tier_b={tier_b}); JIT log:\n{}",
            jit_log(tb_ptx.as_deref().unwrap_or(&base_ptx))
        );
    }
    assert_eq!(
        nsl_sdpa_fused_launch_count(variant),
        before + 1,
        "the runtime did not launch the {} forward it was given",
        if tier_b { "Tier-B" } else { "base" }
    );
    let (out_t, lse_t) = (nsl_list_get(list, 0), nsl_list_get(list, 1));
    let (out, lse) = (read_gpu(out_t, B * H * S * D), read_gpu(lse_t, B * H * S));
    for t in [out_t, lse_t, q_t, k_t, v_t, seg_t] {
        nsl_tensor_free(t);
    }
    nsl_list_free(list);
    (out, lse)
}

/// Launch the packed backward from `out` and `lse`, and require both phases
/// to have run on the GPU.
fn fused_backward(x: &Inputs, out: &[f32], lse: &[f32]) -> Grads {
    let config = backward_config();
    let (p1, p2) = synthesize_flash_attention_backward_ptx(&config);
    let (p1, p2) = (terminated(p1), terminated(p2));
    let name1 = flash_attention_bwd_d_kernel_name(&config);
    let name2 = flash_attention_bwd_main_kernel_name(&config);
    let (c1, c2) = (CString::new(name1.clone()).unwrap(), CString::new(name2.clone()).unwrap());

    let shape = [B as i64, H as i64, S as i64, D as i64];
    let (q_t, k_t, v_t) = (gpu_tensor(&shape, &x.q), gpu_tensor(&shape, &x.k), gpu_tensor(&shape, &x.v));
    let (out_t, dout_t) = (gpu_tensor(&shape, out), gpu_tensor(&shape, &x.dout));
    let lse_t = gpu_tensor(&[B as i64, H as i64, S as i64], lse);
    let seg_t = device_segment_tensor(&x.seg);
    let (before1, before2) = (test_kernel_launch_count(&name1), test_kernel_launch_count(&name2));
    let scale = 1.0f32 / (D as f32).sqrt();
    let list = nsl_flash_attention_backward(
        dout_t,
        q_t,
        k_t,
        v_t,
        out_t,
        lse_t,
        scale.to_bits() as i64,
        B as i64,
        H as i64,
        S as i64,
        D as i64,
        1,
        p1.as_ptr() as i64,
        c1.as_ptr() as i64,
        p2.as_ptr() as i64,
        c2.as_ptr() as i64,
        0,
        0,
        seg_t,
    );
    assert_ne!(list, 0, "nsl_flash_attention_backward returned null; phase-2 JIT log:\n{}", jit_log(&p2));
    let (l1, l2) = (test_kernel_launch_count(&name1) - before1, test_kernel_launch_count(&name2) - before2);
    assert!(
        l1 >= 1 && l2 >= 1,
        "the packed backward did not run on the GPU ({name1}: {l1} launch(es), {name2}: {l2}); \
         the gradients below would be the CPU fallback's"
    );
    let n = B * H * S * D;
    let grads = Grads {
        dq: read_gpu(nsl_list_get(list, 0), n),
        dk: read_gpu(nsl_list_get(list, 1), n),
        dv: read_gpu(nsl_list_get(list, 2), n),
    };
    for i in 0..3 {
        nsl_tensor_free(nsl_list_get(list, i));
    }
    nsl_list_free(list);
    for t in [q_t, k_t, v_t, out_t, dout_t, lse_t, seg_t] {
        nsl_tensor_free(t);
    }
    grads
}

fn report(name: &str, got: &Grads, exact: &Grads, staged: &Grads) -> ([f32; 3], [f32; 3]) {
    let e = [rel(&got.dq, &exact.dq), rel(&got.dk, &exact.dk), rel(&got.dv, &exact.dv)];
    let s = [rel(&got.dq, &staged.dq), rel(&got.dk, &staged.dk), rel(&got.dv, &staged.dv)];
    eprintln!(
        "{name}: vs exact dq {:.2e} dk {:.2e} dv {:.2e}; vs f16-operand dq {:.2e} dk {:.2e} dv {:.2e}",
        e[0], e[1], e[2], s[0], s[1], s[2]
    );
    for (t, g) in [("dq", &got.dq), ("dk", &got.dk), ("dv", &got.dv)] {
        assert!(g.iter().all(|v| v.is_finite()), "{name}: {t} has a non-finite entry");
    }
    (e, s)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn packed_forward_matches_the_oracle_across_kv_tiles() {
    init_cuda();
    let x = inputs();
    let (want_out, want_lse, _) = oracle(&x.q, &x.k, &x.v, &x.dout, &x.seg, false);
    for tier_b in [false, true] {
        let (out, lse) = fused_forward(&x, tier_b);
        let (out_err, lse_err) = (rel(&out, &want_out), max_abs(&lse, &want_lse));
        eprintln!("forward (tier_b={tier_b}): out {out_err:.2e} of max |out|, lse {lse_err:.2e} absolute");
        // The forward stores O in f16, so O is the f16 rounding of the
        // attention output: 2^-11 relative, plus the online-softmax rescale.
        assert!(out_err < 2e-3, "forward (tier_b={tier_b}): out {out_err:.2e} from the oracle");
        // lse is f32 through ex2/lg2.approx over scores of a few units.
        assert!(lse_err < 5e-3, "forward (tier_b={tier_b}): lse {lse_err:.2e} from the oracle");
    }
}

/// The backward alone, fed the oracle's O and logsumexp, so every error is
/// the backward's own. The MMA path is the f16-operand gradient: dV to f32
/// noise, dQ/dK within a few f16 roundings of dS; against the exact
/// gradient all three miss by f16-sized errors, not the ~100% of the
/// double-counted m-tiles the interpreter gate caught.
#[test]
#[ignore = "requires CUDA GPU"]
fn packed_backward_is_the_f16_operand_gradient_across_kv_tiles() {
    init_cuda();
    test_kernel_launch_census_arm();
    let x = inputs();
    let (out, lse, exact) = oracle(&x.q, &x.k, &x.v, &x.dout, &x.seg, false);
    let (_, _, staged) = oracle(&x.q, &x.k, &x.v, &x.dout, &x.seg, true);
    let got = fused_backward(&x, &out, &lse);
    let (e, s) = report("packed backward", &got, &exact, &staged);
    assert!(s[2] < 2e-5, "dV {:.2e} from the f16-operand oracle", s[2]);
    assert!(s[0].max(s[1]) < 5e-4, "dQ/dK {:.2e} from the f16-operand oracle", s[0].max(s[1]));
    let e_max = e[0].max(e[1]).max(e[2]);
    assert!(e_max < 5e-3, "{e_max:.2e} from the exact gradients");
    assert!(e_max > 1e-4, "within {e_max:.2e} of exact: no longer f16 operands, or not the MMA kernel");
}

/// Production's composition: the Tier-B forward's own O and logsumexp feed
/// the backward. This is where a disagreement between the two about the
/// logsumexp's base or layout, or about O's staging, would show; each kernel
/// alone agrees with its oracle.
#[test]
#[ignore = "requires CUDA GPU"]
fn packed_forward_then_backward_is_the_attention_gradient() {
    init_cuda();
    test_kernel_launch_census_arm();
    let x = inputs();
    let (_, _, exact) = oracle(&x.q, &x.k, &x.v, &x.dout, &x.seg, false);
    let (_, _, staged) = oracle(&x.q, &x.k, &x.v, &x.dout, &x.seg, true);
    let (out, lse) = fused_forward(&x, true);
    let got = fused_backward(&x, &out, &lse);
    let (e, _) = report("forward then backward", &got, &exact, &staged);
    let e_max = e[0].max(e[1]).max(e[2]);
    // The f16 O the forward stores enters D = rowsum(dO ∘ O); that and the
    // f16 operands bound the error. A base-2 logsumexp or a transposed O
    // would put it at O(1).
    assert!(e_max < 1e-2, "{e_max:.2e} from the exact gradients");
}
