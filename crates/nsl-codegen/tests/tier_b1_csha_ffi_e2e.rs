//! End-to-end production-FFI test for the CSHA Tier B.1 forward kernel.
//!
//! Where `tier_b1_full_kernel_e2e.rs` validates the kernel via direct
//! `nsl_kernel_launch` with manually-prepared chunkified inputs, this
//! test exercises the **production** path through
//! `nsl_flash_attention_csha`:
//!
//!   1. Caller supplies raw f32 `x` `[seq, d_model]`, f32 `gamma` `[d_model]`,
//!      and f32 `Wq/Wk/Wv` `[d_model, hd]` in row-major layout — the
//!      standard CSHA pipeline format.
//!   2. The FFI inspects the kernel name (`_tier_b1_chunk<N>` suffix) to
//!      detect Tier B.1 dispatch, runs the GPU pre-pass kernels (RMSNorm
//!      + narrow + chunkify on x; narrow + col-major chunkify on Wq/Wk/Wv),
//!      then launches the main Tier B.1 kernel with the chunkified
//!      pointers substituted in.
//!   3. Output is compared element-wise against `csha_reference` (the
//!      f32 CPU implementation).
//!
//! When this test passes, the Tier B.1 pipeline is usable by production
//! callers without requiring host-side pre-processing.
//!
//! Running:
//! ```bash
//! cargo test --package nsl-codegen --features cuda \
//!     --test tier_b1_csha_ffi_e2e -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

#[path = "csha_f64_reference.rs"]
mod csha_f64_reference;
#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference, CshaInputs, CshaShape};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, shared_mem_bytes_v2,
    synthesize_flash_attention_ptx_v2,
};
use nsl_codegen::flash_attention_v2::smem_layout::tier_b1_total_smem_bytes;
use nsl_runtime::flash_attention::nsl_flash_attention_csha;
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use std::ffi::CString;

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = -1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let e = (127 + e - 14) as u32;
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        (sign << 31) | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

fn canonical_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 2048,
            // The production FFI's pre-pass orchestration always feeds the
            // kernel a pre-RMSNormalized + chunkified x. The codegen
            // dispatch path in `synthesize_flash_attention_ptx_v2` forces
            // this flag to `true` when emitting Tier B.1 PTX (see
            // `flash_attention_v2::synthesize_flash_attention_ptx_v2_with_tier_b`),
            // so the caller's value here is overridden. We leave it `false`
            // to exercise the override path.
            skip_rmsnorm_prologue: false,
            ..CshaExtras::default()
        }),
        checkpoint: None,
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_through_csha_ffi_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }

    let config = canonical_config();
    let batch = 1usize;
    let heads = 1usize;
    let seq = config.block_q as usize;
    let head_dim = config.head_dim as usize;
    let kv_dim = heads * head_dim;
    let d_model = config
        .csha
        .as_ref()
        .map(|c| c.d_model as usize)
        .expect("canonical config has csha");
    let causal = config.causal;
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Host inputs (identical fixture to `tier_b1_full_kernel_e2e`).
    let mut x_host = vec![0f32; seq * d_model];
    for s in 0..seq {
        for d in 0..d_model {
            x_host[s * d_model + d] = ((s + d) as f32).sin() * 0.1 + 0.05;
        }
    }
    let mut wq_f32 = vec![0f32; d_model * kv_dim];
    let mut wk_f32 = vec![0f32; d_model * kv_dim];
    let mut wv_f32 = vec![0f32; d_model * kv_dim];
    for d in 0..d_model {
        for n in 0..kv_dim {
            wq_f32[d * kv_dim + n] = ((d * 7 + n) as f32).sin() * 0.05;
            wk_f32[d * kv_dim + n] = ((d * 11 + n) as f32).cos() * 0.05;
            wv_f32[d * kv_dim + n] = ((d * 13 + n) as f32).sin() * 0.05;
        }
    }
    let norm_weight = vec![1.0f32; d_model];

    // CPU reference.
    let cos = vec![1.0f32; seq * (head_dim / 2)];
    let sin = vec![0.0f32; seq * (head_dim / 2)];
    let cpu_out = csha_reference(
        &CshaInputs {
            x: &x_host,
            wq: &wq_f32,
            wk: &wk_f32,
            wv: &wv_f32,
            norm_weight: &norm_weight,
            cos: &cos,
            sin: &sin,
        },
        &CshaShape {
            seq, heads, head_dim, d_model, causal, norm_eps,
            rope_q: true,
            // Mirror the kernel config above. Identity RoPE (cos=1, sin=0)
            // makes the pairing unobservable here, but oracle and kernel must
            // agree on it by construction.
            rope_style: RopeStyle::HalfSplit,
        },
    );

    // GPU buffers — all f32 raw inputs (standard CSHA pipeline format).
    let x_bytes = (x_host.len() * 4) as i64;
    let nw_bytes = (norm_weight.len() * 4) as i64;
    let w_bytes = (wq_f32.len() * 4) as i64;
    let qkv_elems = batch * heads * seq * head_dim;
    let qkv_f32_bytes = (qkv_elems * 4) as i64;
    let out_bytes = (qkv_elems * 2) as i64; // f16
    let lse_bytes = (batch * heads * seq * 4) as i64;

    let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let nw_dev = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    let wq_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wk_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wv_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let q_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let k_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let v_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };

    let all_ptrs = [x_dev, nw_dev, wq_dev, wk_dev, wv_dev, q_dev, k_dev, v_dev, out_dev, lse_dev];
    if !all_ptrs.iter().all(|&p| p != 0) {
        for &p in &all_ptrs { if p != 0 { unsafe { nsl_test_cuda_free(p) }; } }
        panic!("device alloc returned null");
    }

    unsafe {
        nsl_test_cuda_h2d(x_dev, x_host.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(nw_dev, norm_weight.as_ptr() as i64, nw_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f32.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f32.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f32.as_ptr() as i64, w_bytes);
    }

    // PTX synthesis — codegen will see Tier B.1 dispatch criteria and
    // emit the chunked variant with `_tier_b1_chunk<N>` in the name.
    // Crucially, the codegen also force-overrides
    // `skip_rmsnorm_prologue=true` for this path so the kernel does
    // NOT re-RMSNormalize (the runtime FFI's pre-pass already did).
    let mut ptx = synthesize_flash_attention_ptx_v2(&config);
    while ptx.last() == Some(&0) { ptx.pop(); }
    if ptx.last() != Some(&b'\n') { ptx.push(b'\n'); }
    let dump = std::env::temp_dir().join("tier_b1_csha_ffi_e2e.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("[csha-ffi] PTX dumped to {}", dump.display());
    ptx.push(0);
    let kernel_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    eprintln!("[csha-ffi] kernel name = {:?}", kernel_name);
    // Sanity: the name MUST encode the Tier B.1 chunk suffix; without it
    // the runtime's prepass orchestration is skipped.
    let name_str = kernel_name.to_string_lossy();
    assert!(
        name_str.contains("_tier_b1_chunk"),
        "Tier B.1 dispatch should have appended _tier_b1_chunk<N> to the kernel name; got {:?}",
        name_str
    );

    // Tier B.1 declares `.shared .b8 shmem[N]` statically (N <= 48KB
    // for canonical 32x32x32 — 45056 bytes fits the static cap). Pass
    // `shared_mem_bytes=0` so the runtime does NOT opt-in to dynamic
    // SMEM via cuFuncSetAttribute (which would also try to call
    // `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` and fail with
    // CUDA_ERROR_INVALID_VALUE on a static-SMEM kernel).
    let smem = 0i64;
    let _ = tier_b1_total_smem_bytes(&config, 128); // suppress unused-import warn
    let _ = shared_mem_bytes_v2; // suppress unused-import warn under cfg(cuda)

    // ---- Launch through the production FFI ----
    let rc = nsl_flash_attention_csha(
        q_dev, k_dev, v_dev, out_dev, lse_dev,
        scale.to_bits() as i64,
        batch as i64, heads as i64, seq as i64, head_dim as i64,
        0, 0, 0, 0,             // paging
        0, 0, 0, 0,             // rope + ragged
        smem,
        ptx.as_ptr() as i64,
        kernel_name.as_ptr() as i64,
        config.block_q as i64, config.block_kv as i64,
        if causal { 1 } else { 0 },
        // CSHA extras: x, norm_weight, wq, wk, wv, wo, eps, ah, dm
        x_dev, nw_dev, wq_dev, wk_dev, wv_dev, 0,
        norm_eps.to_bits() as i64,
        0, d_model as i64,
        // segment_ids_ptr, tier_b sentinels, doc_starts, num_docs_or_zero
        0, 0, 0, 0, 0,
    );

    if rc != 0 {
        let log_ptr = unsafe { nsl_test_cuda_jit_log(ptx.as_ptr() as i64) };
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no log>".into()
        };
        for &p in &all_ptrs { unsafe { nsl_test_cuda_free(p) }; }
        panic!("csha FFI launch rc={}\nJIT log:\n{}", rc, log);
    }

    // Readback + compare.
    let mut out_f16 = vec![0u16; qkv_elems];
    unsafe { nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, out_bytes) };
    let out_gpu: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();

    for &p in &all_ptrs { unsafe { nsl_test_cuda_free(p) }; }

    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    let mut max_idx = 0usize;
    let mut n_nan = 0usize;
    for (i, (&g, &c)) in out_gpu.iter().zip(cpu_out.iter()).enumerate() {
        if !g.is_finite() {
            n_nan += 1;
            continue;
        }
        let diff = (g - c).abs();
        sum_abs += diff;
        if diff > max_abs {
            max_abs = diff;
            max_idx = i;
        }
    }
    let mean_abs = sum_abs / qkv_elems as f32;
    eprintln!("\n[csha-ffi] ── numerical summary ──");
    eprintln!("[csha-ffi] total elems          = {}", qkv_elems);
    eprintln!("[csha-ffi] non-finite GPU elems = {}", n_nan);
    eprintln!("[csha-ffi] max_abs              = {:.4e}", max_abs);
    eprintln!("[csha-ffi] mean_abs             = {:.4e}", mean_abs);
    if max_idx < out_gpu.len() {
        eprintln!(
            "[csha-ffi] worst idx={} gpu={:.6} cpu={:.6}",
            max_idx, out_gpu[max_idx], cpu_out[max_idx]
        );
    }
    eprintln!("\n[csha-ffi] first 8 GPU: {:?}", &out_gpu[..8.min(out_gpu.len())]);
    eprintln!("[csha-ffi] first 8 CPU: {:?}", &cpu_out[..8.min(cpu_out.len())]);

    assert_eq!(n_nan, 0, "FFI path produced {} non-finite outputs", n_nan);
    // Same tolerance as `tier_b1_full_kernel_e2e` — the production FFI
    // adds an RMSNorm + narrow + chunkify GPU pre-pass which contributes
    // negligible additional noise (the f16 narrowing dominates).
    const MAX_ABS_TOL: f32 = 0.6;
    const MEAN_ABS_TOL: f32 = 0.25;
    assert!(
        max_abs <= MAX_ABS_TOL,
        "FFI-path max_abs {} > {} tolerance",
        max_abs, MAX_ABS_TOL
    );
    assert!(
        mean_abs <= MEAN_ABS_TOL,
        "FFI-path mean_abs {} > {} tolerance",
        mean_abs, MEAN_ABS_TOL
    );
    eprintln!(
        "[csha-ffi] PASS (max_abs={:.4e} ≤ {:.4e}; mean_abs={:.4e} ≤ {:.4e})",
        max_abs, MAX_ABS_TOL, mean_abs, MEAN_ABS_TOL
    );
}

// ── Launcher contract: weights are re-read on every launch, and shapes the
//    pre-passes cannot serve are refused ────────────────────────────────────

/// Device buffers for one Tier B.1 launch at the canonical config, in the
/// raw f32 layout the production FFI takes.
struct B1Devices {
    x: i64,
    nw: i64,
    wq: i64,
    wk: i64,
    wv: i64,
    qkv: [i64; 3],
    out: i64,
    lse: i64,
    seq: usize,
    head_dim: usize,
    d_model: usize,
}

impl B1Devices {
    fn new(config: &FlashAttentionConfig) -> Self {
        let seq = config.block_q as usize;
        let head_dim = config.head_dim as usize;
        let d_model = config.csha.as_ref().expect("csha").d_model as usize;
        let alloc = |bytes: usize| {
            let p = nsl_test_cuda_alloc(bytes as i64);
            assert_ne!(p, 0, "device alloc returned null");
            p
        };
        let qkv = seq * head_dim;
        B1Devices {
            x: alloc(seq * d_model * 4),
            nw: alloc(d_model * 4),
            wq: alloc(d_model * head_dim * 4),
            wk: alloc(d_model * head_dim * 4),
            wv: alloc(d_model * head_dim * 4),
            qkv: [alloc(qkv * 4), alloc(qkv * 4), alloc(qkv * 4)],
            out: alloc(qkv * 2),
            lse: alloc(seq * 4),
            seq,
            head_dim,
            d_model,
        }
    }

    fn upload(dev: i64, host: &[f32]) {
        nsl_test_cuda_h2d(dev, host.as_ptr() as i64, (host.len() * 4) as i64);
    }

    /// Launch through the production FFI with the given batch and head
    /// counts; returns the launcher's return code.
    fn launch(&self, config: &FlashAttentionConfig, batch: i64, heads: i64) -> i64 {
        let mut ptx = synthesize_flash_attention_ptx_v2(config);
        while ptx.last() == Some(&0) {
            ptx.pop();
        }
        ptx.push(b'\n');
        ptx.push(0);
        let name = CString::new(flash_attention_kernel_name_v2(config)).unwrap();
        assert!(name.to_string_lossy().contains("_tier_b1_chunk"), "not a Tier B.1 kernel: {name:?}");
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        nsl_flash_attention_csha(
            self.qkv[0], self.qkv[1], self.qkv[2], self.out, self.lse,
            scale.to_bits() as i64,
            batch, heads, self.seq as i64, self.head_dim as i64,
            0, 0, 0, 0,
            0, 0, 0, 0,
            0,
            ptx.as_ptr() as i64,
            name.as_ptr() as i64,
            config.block_q, config.block_kv,
            i64::from(config.causal),
            self.x, self.nw, self.wq, self.wk, self.wv, 0,
            1e-5f32.to_bits() as i64,
            0, self.d_model as i64,
            0, 0, 0, 0, 0,
        )
    }

    fn read_out(&self) -> Vec<f32> {
        let n = self.seq * self.head_dim;
        let mut raw = vec![0u16; n];
        nsl_test_cuda_d2h(raw.as_mut_ptr() as i64, self.out, (n * 2) as i64);
        raw.iter().map(|&b| f16_to_f32(b)).collect()
    }
}

impl Drop for B1Devices {
    fn drop(&mut self) {
        for p in [self.x, self.nw, self.wq, self.wk, self.wv, self.out, self.lse]
            .into_iter()
            .chain(self.qkv)
        {
            nsl_test_cuda_free(p);
        }
    }
}

/// Host inputs at the canonical config. `variant` selects a second, unrelated
/// set of projection weights.
struct B1Inputs {
    x: Vec<f32>,
    gamma: Vec<f32>,
    wq: Vec<f32>,
    wk: Vec<f32>,
    wv: Vec<f32>,
}

impl B1Inputs {
    fn new(seq: usize, head_dim: usize, d_model: usize, variant: u32) -> Self {
        use csha_f64_reference::{det_seq, gamma_seq};
        // Weights scaled so the 2048-wide projections stay of order 1.
        let w = |seed: u32| -> Vec<f32> {
            det_seq(seed + 100 * variant, d_model * head_dim).into_iter().map(|u| 0.1 * u).collect()
        };
        B1Inputs {
            x: det_seq(42, seq * d_model),
            gamma: gamma_seq(d_model),
            wq: w(43),
            wk: w(44),
            wv: w(45),
        }
    }

    /// The attention output, in f64 from the unrounded inputs.
    fn reference(&self, seq: usize, head_dim: usize, d_model: usize) -> Vec<f64> {
        let d_o = vec![0.0f32; seq * head_dim];
        csha_f64_reference::fused_backward_f64(&csha_f64_reference::FusedCase {
            batch: 1,
            seq,
            head_dim,
            d_model,
            causal: true,
            eps: 1e-5,
            x: &self.x,
            wq: &self.wq,
            wk: &self.wk,
            wv: &self.wv,
            gamma: &self.gamma,
            d_o: &d_o,
            rope: None,
        })
        .o
    }

    fn upload(&self, dev: &B1Devices) {
        // The pre-pass leaves `x` alone (it writes a re-laid-out copy), but
        // upload it each time so a launch never depends on the previous one.
        B1Devices::upload(dev.x, &self.x);
        B1Devices::upload(dev.nw, &self.gamma);
        B1Devices::upload(dev.wq, &self.wq);
        B1Devices::upload(dev.wk, &self.wk);
        B1Devices::upload(dev.wv, &self.wv);
    }
}

/// Relative L2 error allowed on the Tier B.1 output against the f64
/// reference. Every operand is f16 on tensor cores here and the output is
/// stored as f16; measured on an RTX PRO 4500 the error is 6e-4 to 7e-4.
const B1_REL_TOL: f64 = 5e-3;

/// Two launches with different weights at the SAME device addresses: the
/// second must use the second weights. The launcher used to keep its
/// re-laid-out copy of each weight in a cache keyed on the device address
/// and never invalidated it, so an in-place weight update (an optimizer
/// step, a checkpoint load) left the kernel on the old weights.
#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_rereads_weights_updated_in_place() {
    if !cuda_available() {
        return;
    }
    let config = canonical_config();
    let dev = B1Devices::new(&config);
    let (seq, hd, dm) = (dev.seq, dev.head_dim, dev.d_model);

    let first = B1Inputs::new(seq, hd, dm, 0);
    first.upload(&dev);
    assert_eq!(dev.launch(&config, 1, 1), 0, "first launch");
    let out_first = dev.read_out();
    let err_first = csha_f64_reference::rel_l2("first", &out_first, &first.reference(seq, hd, dm));

    let second = B1Inputs::new(seq, hd, dm, 1);
    second.upload(&dev);
    assert_eq!(dev.launch(&config, 1, 1), 0, "second launch");
    let out_second = dev.read_out();
    let want_second = second.reference(seq, hd, dm);
    let err_second = csha_f64_reference::rel_l2("second", &out_second, &want_second);
    // What a stale cache produces: the first launch's output again.
    let err_if_stale = csha_f64_reference::rel_l2("stale", &out_first, &want_second);

    eprintln!(
        "[tier-b1 reread] first={err_first:.3e} second={err_second:.3e} \
         (first output against second reference: {err_if_stale:.3e})"
    );
    assert!(err_if_stale > 0.5, "the two weight sets must give different outputs");
    assert!(err_first < B1_REL_TOL, "first launch off by {err_first:.3e}");
    assert!(
        err_second < B1_REL_TOL,
        "second launch off by {err_second:.3e}: it did not use the weights now at those addresses"
    );
}

/// The pre-passes handle one batch element and one head. A larger batch or
/// head count must be refused, not launched: the `x` pre-pass has no batch
/// dimension and would leave every batch element after the first unread.
#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_refuses_batch_or_heads_above_one() {
    if !cuda_available() {
        return;
    }
    let config = canonical_config();
    let dev = B1Devices::new(&config);
    B1Inputs::new(dev.seq, dev.head_dim, dev.d_model, 0).upload(&dev);
    assert_eq!(dev.launch(&config, 2, 1), -1, "batch 2 must be refused");
    assert_eq!(dev.launch(&config, 1, 2), -1, "2 heads must be refused");
    assert_eq!(dev.launch(&config, 1, 1), 0, "batch 1, 1 head is the supported shape");
}
