//! Cycle-15 Bug-1 ablation harness.
//!
//! Bug 1: `csha_checkpoint_recompute_gpu::t_recompute_hd64_s512_bq32` Path A
//! (checkpoint=None baseline) returned wrong gradients on RTX 5070 Ti
//! (Blackwell sm_120, CUDA 13.2) with config:
//!   causal=true, rope_q=true, fused_projections=true, hd=64, S=512, bq=32.
//!
//! All 7 gradients diverged by orders of magnitude from cpu_reference:
//!   dq:  max_abs=4.420e0  max_rel=1.000e0
//!   dk:  max_abs=4.356e0  max_rel=2.040e1
//!   dv:  max_abs=8.571e0  max_rel=3.979e3
//!   dwq: max_abs=3.403e1  max_rel=1.000e0
//!   dwk: max_abs=3.664e1  max_rel=1.020e0
//!   dwv: max_abs=1.733e1  max_rel=1.032e0
//!   dx:  max_abs=4.691e1  max_rel=1.118e0
//!
//! Three candidate sites:
//!   A: Causal mask predicate in ds_compute.rs:212-232 (%rd42 vs %rd39).
//!   B: dRoPE inverse rotation in csha_hooks_backward.rs:191-365.
//!   C: Projection chain (xnorm_recompute, dproj, drmsnorm) at hd<128.
//!
//! Ablation protocol: toggle ONE feature off per test, hold all others at the
//! cycle-14 failing config (hd=64, S=512, bq=32, causal=true, rope_q=true,
//! fused_projections=true). Each ablation runs Path A baseline
//! (checkpoint=None) vs cpu_reference oracle. Pass/fail pattern narrows the
//! candidate site.
//!
//! All four tests are gated on `#[cfg(feature = "cuda")]` + `#[ignore]` so
//! they never run in default CI and only execute when `--ignored` is passed
//! with a real CUDA device.
//!
//! ─────────────────────────────────────────────────────────────────────────
//! RESOLUTION (2026-07-25): **Bug 1 was not a numerics bug.**
//!
//! The magnitudes quoted above are not error magnitudes — they are the max
//! MAGNITUDES of the reference tensors. Compare, from the A0 control:
//!
//!     reported "max_abs"   A0 max|ref|
//!     dq   4.420e0         4.420e0
//!     dk   4.356e0         4.356e0
//!     dwq  3.403e1         3.403e1
//!     dwk  3.664e1         3.664e1
//!     dwv  1.733e1         1.733e1
//!     dx   4.691e1         4.691e1
//!
//! Six exact matches. `max_abs_diff(gpu, ref) == max|ref|` together with
//! `dq max_rel = 1.000e0` is the arithmetic signature of `gpu == 0`: the
//! kernel never wrote its outputs. That is the harness defect fixed below —
//! a multi-tile fused config was handed single-kernel PTX, the `_mt_attn`
//! twin lookup failed, and the comparison dutifully reported the reference
//! against a buffer of zeros.
//!
//! With the correct PTX, A0/A1/A2 agree with the CPU oracle to 3-11x the f16
//! storage floor, which is ordinary accumulation growth. They previously
//! "failed" for a second, independent reason: six of the seven compared
//! tensors are read back as f16, but `atol` was calibrated for f32 and sits
//! BELOW the f16 representation floor, so the absolute half of the criterion
//! was unsatisfiable no matter how correct the arithmetic. See F16_ACCUM_AMP.
//!
//! STILL OPEN — a separate, pre-existing bug that A3 surfaces:
//! `fused_rmsnorm=true` + `fused_projections=false` (kernel suffix `n1_p0`)
//! synthesizes without complaint and then faults with an invalid __shared__
//! read of 2 bytes at the shared-window base 0xff800000, inside the FORWARD
//! kernel (localized with compute-sanitizer:
//! `nsl_flash_attention_csha_with_saves` -> flash_attention.rs:2327).
//!
//! A3's doc comment below predicted this failure mode and expected a
//! synthesizer refusal to catch it. **A refusal would be the WRONG fix.**
//! `CshaExtras::level1(eps)` constructs precisely this combination
//! (fused_rmsnorm=true, fused_projections=false) and
//! flash_attention_selector.rs:139 selects it in production — n1_p0 is a
//! first-class supported configuration, so refusing it would break a shipping
//! path rather than protect one. R3 in `validate_checkpoint_eligibility`
//! demands both flags, but only under @checkpoint, and A3 runs Path A with
//! checkpoint stripped, which is why nothing gates it here.
//!
//! What is NOT yet established: whether the fault needs rope_q, needs
//! multi-tile (S=512 with 32x32 tiles), or reproduces for plain level-1 at
//! single tile. Those discriminators come first — the fix depends on which.

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference_backward, CshaGradients, CshaInputs, CshaShape};

// Re-use harness helpers from the cycle-14 test (same FFI block).
#[path = "csha_checkpoint_recompute_gpu.rs"]
mod cycle14;

use std::ffi::CString;

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, smem_layout, synthesize_backward_with_tier_b,
    synthesize_forward_multi_tile_combined,
};
use nsl_test::cpu_naive_prologue::{cpu_naive_norm_proj_rope, PrologueConfig};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h,
    nsl_test_cuda_free, nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};
use nsl_runtime::flash_attention::{
    nsl_csha_alloc_backward_activations, nsl_csha_free_backward_activations,
    nsl_flash_attention_csha_backward, nsl_flash_attention_csha_with_saves,
};

// ── Helpers (mirrors csha_checkpoint_recompute_gpu.rs) ───────────────────────

fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s: u32 = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((s >> 16) as f32 / 65535.0) - 0.5
    }).collect()
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    unsafe { nsl_cuda_init() == 0 }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 { sign << 31 } else {
            let mut m = mant;
            let mut e: i32 = -1;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
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

fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() { return 0x7E00; }
    let b = x.to_bits();
    let sign = (b >> 31) & 1;
    let exp = ((b >> 23) & 0xFF) as i32;
    let mant = b & 0x7FFFFF;
    if exp == 255 { return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16; }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 { return ((sign << 15) | 0x7C00) as u16; }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

fn free_all(ptrs: &[i64]) {
    for &p in ptrs { if p != 0 { unsafe { nsl_test_cuda_free(p); } } }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y).abs()).fold(0f32, f32::max)
}

fn max_rel_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter())
        .map(|(&x, &y)| (x - y).abs() / y.abs().max(1e-6))
        .fold(0f32, f32::max)
}

fn tol_dqkv(head_dim: usize) -> (f32, f32) {
    if head_dim >= 128 { (2e-3, 1e-2) } else { (5e-4, 5e-3) }
}

/// The SMALLEST `max_abs` any correct f16-storing kernel can achieve against an
/// f32 reference: round-trip the reference through f16 and measure. Six of the
/// seven compared tensors are read back as f16 (only `dx` is f32), so a tensor
/// whose floor already exceeds `atol` cannot pass on absolute error no matter
/// how correct the arithmetic is — and the OR in `check` then rests entirely on
/// `max_rel`, which `max_rel_diff` inflates without bound near zero because it
/// divides by `y.abs().max(1e-6)`.
fn f16_floor(reference: &[f32]) -> f32 {
    reference
        .iter()
        .map(|&y| (f16_to_f32(f32_to_f16_bits(y)) - y).abs())
        .fold(0f32, f32::max)
}

fn max_mag(v: &[f32]) -> f32 {
    v.iter().map(|x| x.abs()).fold(0f32, f32::max)
}

fn backward_kernel_name(cfg: &FlashAttentionConfig) -> String {
    let fw = flash_attention_kernel_name_v2(cfg);
    match fw.strip_prefix("flash_attn_") {
        Some(rest) => format!("flash_attn_backward_{}", rest),
        None => format!("flash_attn_backward_{fw}"),
    }
}

// ── Shared launch + compare driver ──────────────────────────────────────────

/// Build the cycle-14 failing config and then apply `mutate_fn` to produce
/// the ablation variant. Returns PASS/FAIL per tensor.
fn run_ablation(
    ablation_label: &str,
    head_dim: u32,
    seq_len: u32,
    mutate_fn: impl Fn(&mut FlashAttentionConfig),
) {
    if !cuda_available() {
        eprintln!(
            "[{ablation_label}] SKIPPED — no CUDA device (or NSL_SKIP_CUDA_TESTS set)"
        );
        return;
    }

    // Base: cycle-14 failing config.
    let d_model = head_dim; // 1 head, dm == hd
    let mut config = FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: head_dim as i64,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some({
            let mut e = CshaExtras::level1_with_fused_proj(1e-6);
            e.d_model = d_model;
            e
        }),
        checkpoint: Some(CheckpointExtras::full()),
    };
    // Apply the ablation mutation.
    mutate_fn(&mut config);
    // Path A baseline: strip checkpoint so dispatch takes kv_load branch.
    config.checkpoint = None;

    let batch = 1usize;
    let heads = 1usize;
    let seq = seq_len as usize;
    let hd = head_dim as usize;
    let dm = hd;
    let kv_dim = heads * hd;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let norm_eps = 1e-6f32;
    let causal = config.causal;

    eprintln!("[{ablation_label}] causal={causal} rope_q={} fused_proj={} hd={head_dim} S={seq_len}",
        config.rope_q,
        config.csha.as_ref().map(|c| c.fused_projections).unwrap_or(false));

    // Check if PTX synthesis succeeds before allocating GPU memory.
    let bwd_ptx_check = synthesize_backward_with_tier_b(&config, None);
    match &bwd_ptx_check {
        Ok(_) => eprintln!("[{ablation_label}] backward PTX synthesis OK"),
        Err(e) => {
            eprintln!("[{ablation_label}] backward PTX synthesis REFUSED: {e}");
            eprintln!("[{ablation_label}] SKIPPED — ablation config refused by synthesizer");
            return;
        }
    }

    // ── Deterministic host data ──────────────────────────────────────────────
    let x_host = det_seq(42, heads * seq * hd);
    let wq_f32 = det_seq(43, dm * kv_dim);
    let wk_f32 = det_seq(44, dm * kv_dim);
    let wv_f32 = det_seq(45, dm * kv_dim);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let nw_host = vec![1.0f32; hd];

    // cos/sin only needed when rope_q=true. When rope_q=false the kernel
    // null-guards on cos_ptr/sin_ptr, so we can pass zero device pointers.
    let cos_f32: Vec<f32>;
    let sin_f32: Vec<f32>;
    let cos_f16: Vec<u16>;
    let sin_f16: Vec<u16>;
    let rope_bytes_alloc: i64;
    if config.rope_q {
        let cos_raw: Vec<f32> = (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).cos()).collect();
        let sin_raw: Vec<f32> = (0..seq * hd / 2).map(|i| ((i as f32) * 0.1).sin()).collect();
        cos_f16 = cos_raw.iter().map(|&v| f32_to_f16_bits(v)).collect();
        sin_f16 = sin_raw.iter().map(|&v| f32_to_f16_bits(v)).collect();
        cos_f32 = cos_f16.iter().map(|&b| f16_to_f32(b)).collect();
        sin_f32 = sin_f16.iter().map(|&b| f16_to_f32(b)).collect();
        rope_bytes_alloc = (seq * hd / 2 * 2) as i64;
    } else {
        cos_f32 = vec![];
        sin_f32 = vec![];
        cos_f16 = vec![];
        sin_f16 = vec![];
        rope_bytes_alloc = 0;
    }

    let do_raw = det_seq(99, seq * kv_dim);
    let do_f16: Vec<u16> = do_raw.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let do_f32: Vec<f32> = do_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // ── Device allocation ────────────────────────────────────────────────────
    let qkv_bytes = (heads * seq * hd * 2) as i64;
    let lse_bytes = (batch * heads * seq * 4) as i64;
    let x_bytes = (heads * seq * hd * 4) as i64;
    let w_bytes = (dm * kv_dim * 2) as i64;
    let nw_bytes = (hd * 4) as i64;
    let dw_bytes = (dm * kv_dim * 2) as i64;
    let dx_bytes = (heads * seq * hd * 4) as i64;
    let dxn_bytes = (batch * seq * dm * 4) as i64;

    unsafe { nsl_cuda_init(); }

    let q_dev  = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let k_dev  = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let v_dev  = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };
    let x_dev  = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let nw_dev = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    let wq_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wk_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wv_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };

    let (cos_dev, sin_dev) = if config.rope_q {
        (
            unsafe { nsl_test_cuda_alloc(rope_bytes_alloc) },
            unsafe { nsl_test_cuda_alloc(rope_bytes_alloc) },
        )
    } else {
        (0i64, 0i64)
    };

    let do_dev = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let saves = unsafe {
        nsl_csha_alloc_backward_activations(
            batch as i64, heads as i64, seq as i64, hd as i64,
        )
    };

    // ── H2D ─────────────────────────────────────────────────────────────────
    unsafe {
        nsl_test_cuda_h2d(x_dev,  x_host.as_ptr()  as i64, x_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(nw_dev, nw_host.as_ptr() as i64, nw_bytes);
        nsl_test_cuda_h2d(do_dev, do_f16.as_ptr() as i64, qkv_bytes);
    }
    if config.rope_q && rope_bytes_alloc > 0 {
        unsafe {
            nsl_test_cuda_h2d(cos_dev, cos_f16.as_ptr() as i64, rope_bytes_alloc);
            nsl_test_cuda_h2d(sin_dev, sin_f16.as_ptr() as i64, rope_bytes_alloc);
        }
    }

    // ── Forward launch ───────────────────────────────────────────────────────
    // These ablations run at S=512 with 32x32 tiles, i.e. MULTI-TILE. For a
    // fused-projection config `nsl_flash_attention_csha_with_saves` then does
    // a two-launch dispatch and looks up a `<name>_mt_attn` twin entry, which
    // `synthesize_flash_attention_ptx_v2` alone does not emit — the lookup
    // fails with CUDA_ERROR_NOT_FOUND (500), exactly as flash_attention.rs
    // documents ("if the module lacks the twin ... the name lookup fails and
    // the error surfaces loudly in the rc"). That refusal is correct; the
    // harness was simply handing it single-kernel PTX. The combined
    // synthesizer emits both entries and degrades to the plain kernel for
    // configs with no twin (non-fused / sinks / segment / checkpoint / paged).
    let fwd_ptx = synthesize_forward_multi_tile_combined(&config);
    let fwd_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    let fwd_smem_total = smem_layout::total_bytes(&config);
    let fwd_smem_dyn = if smem_layout::needs_dynamic_smem(&config) {
        fwd_smem_total as i64
    } else { 0 };

    let rc_fwd = unsafe {
        nsl_flash_attention_csha_with_saves(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, hd as i64,
            0, 0, 0, 0,
            cos_dev, sin_dev,
            0, 0,
            fwd_smem_dyn,
            fwd_ptx.as_ptr() as i64, fwd_name.as_ptr() as i64,
            config.block_q, config.block_kv,
            if causal { 1 } else { 0 },
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, dm as i64,
            saves.q_proj, saves.k_proj, saves.v_proj,
            saves.row_max, saves.row_sum,
            saves.x_raw,
            // segment_ids_ptr, tier_b_ptx_ptr, tier_b_name_ptr, doc_starts_ptr,
            // num_docs_or_zero (PCA per-doc CTA Strategy 3 v1 — 5 trailing args).
            0, 0, 0, 0, 0,
        )
    };

    if rc_fwd != 0 {
        let log = unsafe {
            let p = nsl_test_cuda_jit_log(fwd_ptx.as_ptr() as i64);
            if p != 0 {
                std::ffi::CStr::from_ptr(p as *const i8).to_string_lossy().into_owned()
            } else { "<no log>".into() }
        };
        unsafe { nsl_csha_free_backward_activations(saves); }
        free_all(&[q_dev, k_dev, v_dev, out_dev, lse_dev, x_dev, nw_dev,
            wq_dev, wk_dev, wv_dev, cos_dev, sin_dev, do_dev]);
        panic!("[{ablation_label}] forward launch FAILED rc={rc_fwd}\nJIT log:\n{log}");
    }
    eprintln!("[{ablation_label}] forward launch OK rc=0");

    // ── Backward launch ──────────────────────────────────────────────────────
    let dq_dev  = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let dk_dev  = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let dv_dev  = unsafe { nsl_test_cuda_alloc(qkv_bytes) };
    let dwq_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dwk_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dwv_dev = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dx_dev  = unsafe { nsl_test_cuda_alloc(dx_bytes) };
    let dxn_dev = unsafe { nsl_test_cuda_alloc(dxn_bytes) };

    let mut bwd_ptx_str = bwd_ptx_check.unwrap();

    // PTX ASCII audit (same pattern as cycle-14 harness).
    for (line_idx, line) in bwd_ptx_str.lines().enumerate() {
        if line.chars().any(|c| !c.is_ascii()) {
            let bad: Vec<(usize, char)> = line.char_indices()
                .filter(|(_, c)| !c.is_ascii())
                .collect();
            eprintln!(
                "  [{ablation_label}] PTX line {} non-ASCII: {:?}",
                line_idx + 1, bad
            );
        }
    }

    let dump_path = std::env::temp_dir().join(format!(
        "cycle15_{ablation_label}_hd{head_dim}_S{seq_len}.ptx"
    ).replace(' ', "_"));
    std::fs::write(&dump_path, &bwd_ptx_str).ok();
    eprintln!("  [{ablation_label}] backward PTX dump: {}", dump_path.display());

    if !bwd_ptx_str.ends_with('\0') { bwd_ptx_str.push('\0'); }
    let bwd_ptx = bwd_ptx_str.into_bytes();
    let bwd_name = CString::new(backward_kernel_name(&config)).unwrap();

    let bwd_smem_total = smem_layout::total_bytes(&config)
        + smem_layout::backward_extra_bytes(&config);
    let bwd_needs_dyn = bwd_smem_total > 48 * 1024;
    let bwd_smem_dyn = if bwd_needs_dyn { bwd_smem_total as i64 } else { 0 };
    eprintln!(
        "  [{ablation_label}] bwd SMEM total={bwd_smem_total} bytes; dyn={bwd_smem_dyn}"
    );

    let rc_bwd = unsafe {
        nsl_flash_attention_csha_backward(
            q_dev, k_dev, v_dev, out_dev, lse_dev,
            scale.to_bits() as i64,
            batch as i64, heads as i64, seq as i64, hd as i64,
            0, 0, 0, 0,
            cos_dev, sin_dev,
            0, 0,
            bwd_smem_dyn,
            bwd_ptx.as_ptr() as i64, bwd_name.as_ptr() as i64,
            config.block_q, config.block_kv,
            if causal { 1 } else { 0 },
            x_dev, nw_dev, wq_dev, wk_dev, wv_dev,
            0, norm_eps.to_bits() as i64,
            heads as i64, dm as i64,
            saves.q_proj, saves.k_proj, saves.v_proj,
            saves.row_max, saves.row_sum,
            saves.x_raw,
            do_dev,
            dq_dev, dk_dev, dv_dev,
            dwq_dev, dwk_dev, dwv_dev,
            dx_dev,
            dxn_dev,
            // segment_ids, tier_b_ptx, tier_b_name, doc_starts, tier_b2_active,
            // num_docs_or_zero (PCA per-doc CTA Sprint 5 — 6 trailing args).
            0, 0, 0, 0, 0, 0,
        )
    };
    if rc_bwd != 0 {
        let log = unsafe {
            let p = nsl_test_cuda_jit_log(bwd_ptx.as_ptr() as i64);
            if p != 0 {
                std::ffi::CStr::from_ptr(p as *const i8).to_string_lossy().into_owned()
            } else { "<no log>".into() }
        };
        free_all(&[dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev, dx_dev, dxn_dev]);
        unsafe { nsl_csha_free_backward_activations(saves); }
        free_all(&[q_dev, k_dev, v_dev, out_dev, lse_dev, x_dev, nw_dev,
            wq_dev, wk_dev, wv_dev, cos_dev, sin_dev, do_dev]);
        panic!("[{ablation_label}] backward launch FAILED rc={rc_bwd}\nJIT log:\n{log}");
    }
    eprintln!("[{ablation_label}] backward launch OK rc=0");

    // ── Readback gradients ───────────────────────────────────────────────────
    let qkv_elems = heads * seq * hd;
    let dw_elems = dm * kv_dim;
    let dx_elems = heads * seq * hd;

    let read_f16 = |dev: i64, elems: usize| -> Vec<f32> {
        let mut raw = vec![0u16; elems];
        unsafe { nsl_test_cuda_d2h(raw.as_mut_ptr() as i64, dev, (elems * 2) as i64); }
        raw.iter().map(|&b| f16_to_f32(b)).collect()
    };
    let read_f32 = |dev: i64, elems: usize| -> Vec<f32> {
        let mut out = vec![0f32; elems];
        unsafe { nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dev, (elems * 4) as i64); }
        out
    };

    let gpu_grads = CshaGradients {
        dq:  read_f16(dq_dev,  qkv_elems),
        dk:  read_f16(dk_dev,  qkv_elems),
        dv:  read_f16(dv_dev,  qkv_elems),
        dwq: read_f16(dwq_dev, dw_elems),
        dwk: read_f16(dwk_dev, dw_elems),
        dwv: read_f16(dwv_dev, dw_elems),
        dx:  read_f32(dx_dev,  dx_elems),
    };

    free_all(&[dq_dev, dk_dev, dv_dev, dwq_dev, dwk_dev, dwv_dev, dx_dev, dxn_dev]);
    unsafe { nsl_csha_free_backward_activations(saves); }
    free_all(&[q_dev, k_dev, v_dev, out_dev, lse_dev, x_dev, nw_dev,
        wq_dev, wk_dev, wv_dev, cos_dev, sin_dev, do_dev]);

    // ── CPU reference oracle ─────────────────────────────────────────────────
    // When rope_q=false, pass zero cos/sin so the reference skips rotation.
    let cos_ref = if config.rope_q { cos_f32.as_slice() } else { &[] };
    let sin_ref = if config.rope_q { sin_f32.as_slice() } else { &[] };
    let inputs = CshaInputs {
        x: &x_host,
        wq: &wq_f32, wk: &wk_f32, wv: &wv_f32,
        norm_weight: &nw_host,
        cos: cos_ref, sin: sin_ref,
    };
    let shape = CshaShape {
        seq, heads, head_dim: hd, d_model: dm,
        causal,
        norm_eps: 1e-6,
        rope_q: config.rope_q,
    };
    let cpu_grads = csha_reference_backward(&inputs, &shape, &do_f32);

    // ── Compare ──────────────────────────────────────────────────────────────
    let (atol_qkv, rtol_qkv) = tol_dqkv(hd);
    let atol_dw = 1e-3f32; let rtol_dw = 1e-2f32;
    let atol_dx = 1e-2f32; let rtol_dx = 2e-2f32;

    // Accumulation amplification allowed over the f16 storage floor. The
    // reductions here run over S=512 (attention) and d_model=64 (projections);
    // f16 error through an N-term reduction grows ~sqrt(N), so sqrt(512)=22.6
    // is the loose bound and 16 is a deliberately TIGHTER pre-registered
    // choice, fixed before checking whether it passes rather than fitted to
    // the data. Observed worst case across A0/A1/A2 is 11.10 (A2 dq), i.e.
    // the bound is not tight against the measurements but does constrain them.
    const F16_ACCUM_AMP: f32 = 16.0;

    let check = |name: &str, x: &[f32], y: &[f32], atol: f32, rtol: f32| -> bool {
        let abs = max_abs_diff(x, y);
        let rel = max_rel_diff(x, y);
        // A tolerance below the storage floor is unsatisfiable by construction,
        // so raise `atol` to the floor-derived bound when it is the smaller of
        // the two. `dx` is f32 and has floor ~0, so it keeps its original atol.
        let atol = atol.max(F16_ACCUM_AMP * f16_floor(y));
        let ok = abs <= atol || rel <= rtol;
        // `floor` is what a PERFECT f16 kernel would score; `ratio` is how many
        // times worse this run is than that. ratio ~1 means the discrepancy is
        // storage precision, not arithmetic — and `atol` below `floor` means
        // the absolute half of the criterion was never satisfiable.
        let floor = f16_floor(y);
        let ratio = if floor > 0.0 { abs / floor } else { f32::INFINITY };
        eprintln!(
            "  [{ablation_label}] {name}: max_abs={abs:.3e} max_rel={rel:.3e} \
             (atol={atol:.0e} rtol={rtol:.0e}) {} \
             | max|ref|={:.3e} f16_floor={floor:.3e} abs/floor={ratio:.2}{}",
            if ok { "PASS" } else { "FAIL" },
            max_mag(y),
            if floor > atol { "  <-- atol BELOW f16 floor" } else { "" }
        );
        ok
    };

    let r_dq  = check("dq",  &gpu_grads.dq,  &cpu_grads.dq,  atol_qkv, rtol_qkv);
    let r_dk  = check("dk",  &gpu_grads.dk,  &cpu_grads.dk,  atol_qkv, rtol_qkv);
    let r_dv  = check("dv",  &gpu_grads.dv,  &cpu_grads.dv,  atol_qkv, rtol_qkv);
    let r_dwq = check("dwq", &gpu_grads.dwq, &cpu_grads.dwq, atol_dw,  rtol_dw);
    let r_dwk = check("dwk", &gpu_grads.dwk, &cpu_grads.dwk, atol_dw,  rtol_dw);
    let r_dwv = check("dwv", &gpu_grads.dwv, &cpu_grads.dwv, atol_dw,  rtol_dw);
    let r_dx  = check("dx",  &gpu_grads.dx,  &cpu_grads.dx,  atol_dx,  rtol_dx);

    let fails: Vec<&str> = [
        ("dq", r_dq), ("dk", r_dk), ("dv", r_dv),
        ("dwq", r_dwq), ("dwk", r_dwk), ("dwv", r_dwv),
        ("dx", r_dx),
    ].iter().filter_map(|(n, ok)| if !ok { Some(*n) } else { None }).collect();

    let overall = if fails.is_empty() { "PASS" } else { "FAIL" };
    eprintln!(
        "[{ablation_label}] OVERALL: {overall} \
         (causal={causal} rope_q={} fused_proj={} hd={head_dim})",
        config.rope_q,
        config.csha.as_ref().map(|c| c.fused_projections).unwrap_or(false),
    );

    assert!(
        fails.is_empty(),
        "[{ablation_label}] FAIL on {} tensor(s): {:?}",
        fails.len(), fails
    );
}

// ── Ablation tests ────────────────────────────────────────────────────────────

/// A1: causal=true, rope_q=FALSE, fused_projections=true, hd=64
///
/// Toggles RoPE off while keeping causal mask and fused projections.
/// Pure-synthesis probe (no GPU): does the FORWARD kernel's shared-memory
/// declaration agree with what `needs_dynamic_smem` tells the launcher to
/// allocate? A3 faults with an invalid `__shared__` read at 0xff800000 — the
/// base of the shared window, which is what a read hits when the dynamic
/// allocation is zero. That is the signature of an `.extern .shared` kernel
/// launched with dyn=0.
#[test]
fn probe_forward_smem_decl_matches_launcher_for_ablation_configs() {
    for (label, mutate) in [
        ("A0-base", &(|_: &mut FlashAttentionConfig| {}) as &dyn Fn(&mut FlashAttentionConfig)),
        // EXACTLY what a3 does — only `fused_projections` is cleared, so
        // `fused_rmsnorm` stays TRUE. That n1_p0 combination is the one whose
        // kernel faults; clearing both (n0_p0) is a different, working config.
        ("A3-fused-proj-off", &|cfg: &mut FlashAttentionConfig| {
            if let Some(c) = cfg.csha.as_mut() {
                c.fused_projections = false;
            }
        }),
        ("A3b-both-off", &|cfg: &mut FlashAttentionConfig| {
            if let Some(c) = cfg.csha.as_mut() {
                c.fused_projections = false;
                c.fused_rmsnorm = false;
            }
        }),
    ] {
        let mut config = FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 64,
            causal: true,
            paged: false,
            rope_q: true,
            rope_style: RopeStyle::Adjacent,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some({
                let mut e = CshaExtras::level1_with_fused_proj(1e-6);
                e.d_model = 64;
                e
            }),
            checkpoint: None,
        };
        mutate(&mut config);

        let ptx = synthesize_forward_multi_tile_combined(&config);
        let end = ptx.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        let txt = String::from_utf8_lossy(&ptx[..end]).into_owned();

        let declares_extern = txt.contains(".extern .shared");
        let static_decl = txt.contains(".shared .align 16 .b8 shmem[");
        let total = smem_layout::total_bytes(&config);
        let needs_dyn = smem_layout::needs_dynamic_smem(&config);
        eprintln!(
            "[{label}] total_bytes={total} needs_dynamic_smem={needs_dyn} \
             ptx_extern_shared={declares_extern} ptx_static_shmem={static_decl}"
        );
        // Reported, NOT asserted: a combined module legitimately carries two
        // entries with different SMEM shapes, so a module-scope
        // `.extern .shared` does not imply the launched entry uses it. A0
        // declares extern AND launches at dyn=0 and is correct.
        let entry = flash_attention_kernel_name_v2(&config);
        let uses_extern_in_entry = txt
            .split_once(&format!(".visible .entry {entry}"))
            .map(|(_, rest)| {
                let body = rest.split(".visible .entry").next().unwrap_or(rest);
                !body.contains(".shared .align 16 .b8 shmem[")
            })
            .unwrap_or(false);
        eprintln!(
            "[{label}]   entry={entry}\n[{label}]   \
             launched_entry_relies_on_dynamic_shmem={uses_extern_in_entry} \
             (launcher would pass dyn={})",
            if needs_dyn { total } else { 0 }
        );
    }
}

/// A0 — THE MISSING CONTROL. Every other test here mutates the config, but
/// nothing ran it unmutated, so "A1 passes / A1 fails" had no baseline inside
/// this harness to be measured against. The cycle-14 magnitudes quoted in the
/// module header (dq max_abs=4.420e0) come from a DIFFERENT test binary with
/// its own inputs; comparing an ablation here against those numbers silently
/// assumes the two harnesses agree, which is exactly the kind of assumption
/// that produced the fake results earlier in this investigation.
///
/// A0 makes the comparison internal: same inputs, same oracle, same
/// tolerances, only the ablation removed. Read A1/A2/A3 against THIS.
#[test]
#[ignore = "requires CUDA GPU"]
fn a0_base_config_control_causal_rope_q_fused_proj_hd64() {
    run_ablation("A0-base-control", 64, 512, |_cfg| {});
}

/// If A1 PASSES but the base config FAILs, the bug is in the dRoPE inverse
/// rotation (Candidate B: csha_hooks_backward.rs:191-365 emit_drope).
#[test]
#[ignore = "requires CUDA GPU"]
fn a1_rope_q_off_causal_true_fused_proj_true_hd64() {
    run_ablation("A1-rope-q-off", 64, 512, |cfg| {
        cfg.rope_q = false;
    });
}

/// A2: causal=FALSE, rope_q=true, fused_projections=true, hd=64
///
/// Toggles causal mask off while keeping RoPE and fused projections.
/// If A2 PASSES but the base config FAILs, the bug is in the causal mask
/// predicate (Candidate A: ds_compute.rs:212-232 setp.gt / or.pred).
#[test]
#[ignore = "requires CUDA GPU"]
fn a2_causal_off_rope_q_true_fused_proj_true_hd64() {
    run_ablation("A2-causal-off", 64, 512, |cfg| {
        cfg.causal = false;
    });
}

/// A3: causal=true, rope_q=true, fused_projections=FALSE, hd=64
///
/// Toggles fused_projections off. This changes the kernel significantly --
/// the dproj/dRMSNorm hooks are skipped, and dq/dk/dv must still be correct
/// without the projection chain. If A3 PASSES but the base config FAILs,
/// the bug is in emit_dproj / emit_drmsnorm / emit_xnorm_recompute
/// (Candidate C: csha_hooks_backward.rs:60-159, :385-505, :525-879).
///
/// Structural note: disabling fused_projections makes d_model=0 (via level1
/// not level1_with_fused_proj), so the x_norm/rms/dproj/drmsnorm hooks
/// short-circuit. This does change the kernel parameter surface (no csha_wq_ptr
/// etc.) which may affect PTX synthesis if the backward prelude
/// unconditionally declares those registers. The synthesizer refusal check
/// at the start of run_ablation will catch this and mark A3 STRUCTURALLY-BLOCKED.
#[test]
#[ignore = "broken: fused_rmsnorm=true + fused_projections=false (n1_p0) synthesizes then faults with an invalid __shared__ read at the window base; needs a synthesis-time refusal, see module header"]
fn a3_fused_proj_off_causal_true_rope_q_true_hd64() {
    run_ablation("A3-fused-proj-off", 64, 512, |cfg| {
        if let Some(csha) = cfg.csha.as_mut() {
            csha.fused_projections = false;
            csha.d_model = 0; // no projection weights
            csha.save_activations_for_backward = true; // keep x_raw save
        }
    });
}

/// A4: causal=true, rope_q=true, fused_projections=true, hd=128
///
/// Uses the cycle-12 known-passing config (hd=128) instead of hd=64.
/// If A4 PASSES but A1/A2/A3 all FAIL, the bug is hd-stride sensitive --
/// likely a stride that assumes head_dim=128 in some emitter
/// (Candidate C sub-case).
///
/// Note: hd=128 + bq=32 is below the Blackwell 99 KB SMEM cap (verified
/// by G14-D for hd=64; the budget scales linearly with head_dim so hd=128
/// bq=32 stays well within bounds).
#[test]
#[ignore = "requires CUDA GPU"]
fn a4_hd128_causal_true_rope_q_true_fused_proj_true() {
    run_ablation("A4-hd128", 128, 512, |cfg| {
        cfg.head_dim = 128;
        if let Some(csha) = cfg.csha.as_mut() {
            csha.d_model = 128; // dm == hd for 1-head case
        }
    });
}
