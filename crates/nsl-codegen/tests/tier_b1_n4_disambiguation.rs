//! N4 disambiguation: launch the Tier B.1 forward kernel at the canonical
//! 32×32×32 (d_model=2048, chunk=128) config with deterministic inputs and
//! compare numerically against the CPU naive-attention reference.
//!
//! ## Status (2026-05-15)
//!
//! Kernel launches and writes all output positions (no zeros, no crash on
//! sm_120). Two GPU-correctness bugs were uncovered + fixed during the
//! initial run:
//!   1. `phases::forward::prelude::emit` was declaring `shmem[N]` at the
//!      Tier-A baseline (`smem_layout::total_bytes`), not Tier B.1's
//!      chunk-aware total. Writes to offsets past 36 KB walked off the
//!      end of the static allocation, silently stomping neighbouring
//!      GPU state. Fixed by `emit_with_smem_override` + the new
//!      `smem_layout::tier_b1_total_smem_bytes` helper.
//!   2. `tier_b1::projection_mma::emit_cp_async_multi_iter` was narrowing
//!      the 64-bit HBM source pointer to u32 (`cvt.u32.u64`), then
//!      sign-extending back to u64. On Blackwell HBM allocations live
//!      near `0x7f00_0000_0000`; the upper 32 bits are non-zero, so the
//!      narrowing produced `CUDA_ERROR_ILLEGAL_ADDRESS` at launch.
//!      Fixed by switching the HBM address path to u64 throughout.
//!
//! After both fixes the kernel produces all-NaN output. This is a
//! THIRD, ORTHOGONAL design gap discovered while running N4 (NOT the
//! helper-convention question this test exists to answer):
//!
//!   * `csha_hooks::emit_prologue` reads + writes `csha_x_ptr` as `f32`
//!     (RMSNorm prologue uses `ld.global.f32` / `st.global.f32`).
//!   * `tier_b1::projection_mma` stages `csha_x_ptr` to SMEM as `f16`
//!     (cp.async transfers raw bytes; the MMA reads them as `f16`).
//!     The pre-existing rustdoc at `projection_mma.rs:590-596` says
//!     "narrowing must happen upstream (csha RMSNorm path)" — but no
//!     `f32 -> f16` narrowing pass exists. The MMA reads raw `f32` bit
//!     patterns as `f16` values, producing arbitrary results that often
//!     overflow to NaN through softmax.
//!
//! That bridge is in-scope for a follow-up CSHA Tier B.1 design pass —
//! either add an `f32 -> f16` narrowing in `csha_hooks::emit_prologue`
//! (writing a SEPARATE `f16` buffer + threading a new pointer through
//! the FFI) or rewrite RMSNorm + projection to share a single `f16`
//! convention from prologue through epilogue. Until that lands, the
//! helper-convention disambiguation that this test exists for cannot be
//! cleanly answered: any NaN-vs-NaN comparison is dominated by the
//! dtype-mismatch noise, not the helper's correctness.
//!
//! The kernel launches without faulting and the data-flow plumbing is
//! intact, so the test is kept in-tree as a regression sentinel for the
//! two fixes above. When the narrowing bridge lands, this test should
//! be re-run and the DIAGNOSIS branch below will classify the helper.
//!
//! ## What N4 disambiguates
//!
//! `matmul_mma::emit_load_a_fragment_smem` reads 4 b32 from a SINGLE SMEM
//! row at column offsets {0, 8, 16, 24} bytes. The PTX m16n8k16 spec for
//! the A-fragment (16×16 f16, row-major) instead expects each thread to
//! hold elements from TWO rows × TWO column-pair positions:
//!     a0: row=(t/4),     col=(t%4)*2      (lo: f16 at col, hi: f16 at col+1)
//!     a1: row=(t/4)+8,   col=(t%4)*2
//!     a2: row=(t/4),     col=(t%4)*2 + 8
//!     a3: row=(t/4)+8,   col=(t%4)*2 + 8
//!
//! The helper's convention may be:
//!   (a) Correct — SMEM was pre-shuffled or alternate spec interpretation
//!       (kernel output matches CPU within f16 tolerance).
//!   (b) Locally buggy — output drifts in a structured way (e.g., zeros at
//!       specific positions, doubled values, predictable swaps).
//!   (c) Fundamentally wrong — output is unstructured garbage / NaN; the
//!       helper must be rewritten to the PTX-spec convention.
//!
//! ## Running
//!
//! ```bash
//! cargo test --package nsl-codegen --features cuda --test tier_b1_n4_disambiguation \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `#[ignore]`-gated because it requires a live CUDA GPU. Skips gracefully
//! when `nsl_cuda_init()` fails. `--test-threads=1` because the CUDA
//! driver singleton is process-global.
//!
//! ## Why this test calls `nsl_kernel_launch` directly
//!
//! Tier B.1 codegen assumes 8 warps (256 threads) per CTA — see the
//! `global_t = warp_id + local_t * 8` warp distribution in
//! `tier_b1::attention_mma`, `tier_b1::projection_mma`, and
//! `tier_b1::finalize`. The shared `nsl_flash_attention_csha` FFI in
//! `nsl-runtime` hardcodes `block_x = 128i64` (4 warps, the Tier-A value);
//! launching Tier B.1 through it leaves warps 4–7 missing and the kernel
//! silently writes zeros for any tile owned by warp_id >= 4. This test
//! bypasses the shared FFI and uses `nsl_kernel_launch` directly so the
//! per-Tier launch geometry can be set without modifying the runtime
//! crate.

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference, det_seq, CshaInputs, CshaShape};

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_flash_attention_ptx_v2,
};
use std::ffi::{c_void, CString};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};

// `nsl_kernel_launch` is `#[no_mangle] pub extern "C"` in nsl-runtime but
// not re-exported via the doc-hidden test seam. Declare the FFI here.
extern "C" {
    fn nsl_kernel_launch(
        ptx_ptr: i64,
        name_ptr: i64,
        grid_x: i64,
        grid_y: i64,
        grid_z: i64,
        block_x: i64,
        block_y: i64,
        block_z: i64,
        args_ptr: i64,
        num_args: i64,
        shared_mem_bytes: i64,
    ) -> i64;
}

/// IEEE 754 f16 → f32.
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

/// f32 → f16 (round-to-nearest, saturate).
fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 255 {
        return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16;
    }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

fn free_all(ptrs: &[i64]) {
    for &p in ptrs {
        if p != 0 {
            unsafe { nsl_test_cuda_free(p) };
        }
    }
}

/// Canonical Tier B.1 dispatch: block_q=block_kv=head_dim=32, d_model=2048,
/// causal=true, gpu_sm=120, csha.level=2.
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
            ..CshaExtras::default()
        }),
        checkpoint: None,
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_canonical_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }

    let config = canonical_config();
    let batch = 1usize;
    let heads = 1usize;
    let seq = config.block_q as usize; // single-tile (seq == block_q == block_kv)
    let head_dim = config.head_dim as usize;
    let kv_dim = heads * head_dim;
    let d_model = config
        .csha
        .as_ref()
        .map(|c| c.d_model as usize)
        .expect("canonical config has csha");
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let causal = config.causal;

    // ── Host inputs ──────────────────────────────────────────────────────
    // x: [seq, d_model] f32. Small magnitudes keep softmax numerics calm.
    let x_host = det_seq(0xA1B2_C3D4, seq * d_model);
    // Wq, Wk, Wv: [d_model, kv_dim] f16. Independent seeds.
    let wq_f32 = det_seq(0x1111_1111, d_model * kv_dim);
    let wk_f32 = det_seq(0x2222_2222, d_model * kv_dim);
    let wv_f32 = det_seq(0x3333_3333, d_model * kv_dim);
    let wq_f16: Vec<u16> = wq_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wk_f16: Vec<u16> = wk_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    let wv_f16: Vec<u16> = wv_f32.iter().map(|&v| f32_to_f16_bits(v)).collect();
    // RMSNorm weight: all 1.0 to keep the reference clean.
    let norm_weight = vec![1.0f32; d_model];

    // ── CPU reference (identity RoPE, head-wise normalisation matches kernel) ──
    // Per csha_cuda_launch_fused convention: pass identity RoPE to the
    // reference, use the same d_model as the kernel, normalise over d_model.
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
            seq,
            heads,
            head_dim,
            d_model,
            causal,
            norm_eps,
            rope_q: true,
        },
    );

    // ── Device allocations ───────────────────────────────────────────────
    let x_bytes = (x_host.len() * 4) as i64;
    let w_bytes = (wq_f16.len() * 2) as i64;
    let nw_bytes = (norm_weight.len() * 4) as i64;
    let qkv_elems = batch * heads * seq * head_dim;
    let qkv_f32_bytes = (qkv_elems * 4) as i64;
    let out_bytes = (qkv_elems * 2) as i64; // f16
    let lse_bytes = (batch * heads * seq * 4) as i64;

    let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let wq_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wk_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let wv_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let nw_dev = unsafe { nsl_test_cuda_alloc(nw_bytes) };
    // Q/K/V are unused by Tier B.1 (it projects from x) but the prelude's
    // param block still declares them; allocate so the pointers are valid
    // even though the kernel never dereferences them.
    let q_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let k_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let v_dev = unsafe { nsl_test_cuda_alloc(qkv_f32_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    let lse_dev = unsafe { nsl_test_cuda_alloc(lse_bytes) };

    let all_ptrs = [
        x_dev, wq_dev, wk_dev, wv_dev, nw_dev, q_dev, k_dev, v_dev, out_dev, lse_dev,
    ];
    if !all_ptrs.iter().all(|&p| p != 0) {
        free_all(&all_ptrs);
        panic!("device allocation returned null");
    }

    unsafe {
        nsl_test_cuda_h2d(x_dev, x_host.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(wq_dev, wq_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wk_dev, wk_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(wv_dev, wv_f16.as_ptr() as i64, w_bytes);
        nsl_test_cuda_h2d(nw_dev, norm_weight.as_ptr() as i64, nw_bytes);
    }

    // ── PTX synthesis (routes to tier_b1::synthesize) ────────────────────
    let mut ptx = synthesize_flash_attention_ptx_v2(&config);
    while ptx.last() == Some(&0) {
        ptx.pop();
    }
    if ptx.last() != Some(&b'\n') {
        ptx.push(b'\n');
    }
    // Dump for offline debugging.
    let dump = std::env::temp_dir().join("tier_b1_n4_canonical.ptx");
    std::fs::write(&dump, &ptx).ok();
    eprintln!("[N4] PTX dumped to: {}", dump.display());
    ptx.push(0); // null-terminate for cuModuleLoadData

    let kernel_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();

    // ── Build the 37-arg list (mirrors `nsl_flash_attention_csha`) ───────
    // All raw HBM device pointers — the test seam allocates on the device
    // and `nsl_test_cuda_alloc` returns the raw device base, not a wrapped
    // tensor handle, so no `csha_tensor_data_ptr` resolution is needed.
    let mut q = q_dev as u64;
    let mut k = k_dev as u64;
    let mut v = v_dev as u64;
    let mut out = out_dev as u64;
    let mut s = scale;
    let mut b = batch as u64;
    let mut h = heads as u64;
    let mut sl = seq as u64;
    let mut hd = head_dim as u64;
    let mut bt: u64 = 0;
    let mut kp: u64 = 0;
    let mut vp: u64 = 0;
    let mut bs: u64 = 0;
    let mut cos_ptr: u64 = 0;
    let mut sin_ptr: u64 = 0;
    let mut sids: u64 = 0;
    let mut slens: u64 = 0;
    let mut dfs_enter: u64 = 0;
    let mut dfs_exit: u64 = 0;
    let mut num_tree_nodes: u64 = 0;
    let mut lse = lse_dev as u64;
    let mut x = x_dev as u64;
    let mut nw = nw_dev as u64;
    let mut wq = wq_dev as u64;
    let mut wk = wk_dev as u64;
    let mut wv = wv_dev as u64;
    let mut wo: u64 = 0;
    let mut eps = norm_eps;
    let mut ah: u32 = 0;
    let mut dm = d_model as u32;
    let mut q_proj: u64 = 0;
    let mut k_proj: u64 = 0;
    let mut v_proj: u64 = 0;
    let mut rmax: u64 = 0;
    let mut rsum: u64 = 0;
    let mut xraw: u64 = 0;

    let args: [*mut c_void; 37] = [
        &mut q as *mut _ as *mut c_void,
        &mut k as *mut _ as *mut c_void,
        &mut v as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut s as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut sl as *mut _ as *mut c_void,
        &mut hd as *mut _ as *mut c_void,
        &mut bt as *mut _ as *mut c_void,
        &mut kp as *mut _ as *mut c_void,
        &mut vp as *mut _ as *mut c_void,
        &mut bs as *mut _ as *mut c_void,
        &mut cos_ptr as *mut _ as *mut c_void,
        &mut sin_ptr as *mut _ as *mut c_void,
        &mut sids as *mut _ as *mut c_void,
        &mut slens as *mut _ as *mut c_void,
        &mut dfs_enter as *mut _ as *mut c_void,
        &mut dfs_exit as *mut _ as *mut c_void,
        &mut num_tree_nodes as *mut _ as *mut c_void,
        &mut lse as *mut _ as *mut c_void,
        &mut x as *mut _ as *mut c_void,
        &mut nw as *mut _ as *mut c_void,
        &mut wq as *mut _ as *mut c_void,
        &mut wk as *mut _ as *mut c_void,
        &mut wv as *mut _ as *mut c_void,
        &mut wo as *mut _ as *mut c_void,
        &mut eps as *mut _ as *mut c_void,
        &mut ah as *mut _ as *mut c_void,
        &mut dm as *mut _ as *mut c_void,
        &mut q_proj as *mut _ as *mut c_void,
        &mut k_proj as *mut _ as *mut c_void,
        &mut v_proj as *mut _ as *mut c_void,
        &mut rmax as *mut _ as *mut c_void,
        &mut rsum as *mut _ as *mut c_void,
        &mut xraw as *mut _ as *mut c_void,
        // segment_ids_ptr (always last, always 0 for non-PCA configs).
        &mut sids as *mut _ as *mut c_void,
    ];

    // ── Launch with Tier-B.1 geometry: grid=[1,1,1], block=[256,1,1] ─────
    // 256 threads = 8 warps, matching the `global_t = warp_id + local_t*8`
    // distribution baked into the codegen. Static SMEM is declared inside
    // the kernel; pass 0 for `shared_mem_bytes`.
    let rc = unsafe {
        nsl_kernel_launch(
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            /* grid */ 1, 1, 1,
            /* block */ 256, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            /* smem_dynamic */ 0,
        )
    };

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
        free_all(&all_ptrs);
        panic!("Tier B.1 launch failed rc={}\nJIT log:\n{}", rc, log);
    }

    // ── Readback + compare ───────────────────────────────────────────────
    let mut out_f16 = vec![0u16; qkv_elems];
    unsafe { nsl_test_cuda_d2h(out_f16.as_mut_ptr() as i64, out_dev, out_bytes) };
    let out_gpu: Vec<f32> = out_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // Stats: max-abs, mean-abs, fraction zero, fraction NaN/inf, structural
    // pattern (zeros at warp_id >= 4 owned tiles would indicate the
    // thread-count fix didn't take).
    let mut max_abs = 0f32;
    let mut sum_abs = 0f32;
    let mut max_idx = 0usize;
    let mut n_nan = 0usize;
    let mut n_zero = 0usize;
    for (i, (&g, &c)) in out_gpu.iter().zip(cpu_out.iter()).enumerate() {
        if !g.is_finite() {
            n_nan += 1;
            continue;
        }
        if g == 0.0 {
            n_zero += 1;
        }
        let diff = (g - c).abs();
        sum_abs += diff;
        if diff > max_abs {
            max_abs = diff;
            max_idx = i;
        }
    }
    let mean_abs = sum_abs / qkv_elems as f32;
    eprintln!("\n[N4] ── numerical disambiguation summary ──");
    eprintln!("[N4] total elems          = {}", qkv_elems);
    eprintln!("[N4] non-finite GPU elems = {}", n_nan);
    eprintln!("[N4] zero GPU elems       = {}", n_zero);
    eprintln!("[N4] max_abs              = {:.4e}", max_abs);
    eprintln!("[N4] mean_abs             = {:.4e}", mean_abs);
    eprintln!(
        "[N4] worst idx={} gpu={:.6} cpu={:.6}",
        max_idx, out_gpu[max_idx], cpu_out[max_idx]
    );

    let diag_n = 8.min(qkv_elems);
    eprintln!("[N4] first {} GPU: {:?}", diag_n, &out_gpu[..diag_n]);
    eprintln!("[N4] first {} CPU: {:?}", diag_n, &cpu_out[..diag_n]);

    // Per-row max_abs to expose structured zero patterns (e.g., all-zero
    // for rows owned by missing warps).
    eprintln!("[N4] per-row max_abs:");
    for row in 0..seq {
        let base = row * head_dim;
        let row_max = out_gpu[base..base + head_dim]
            .iter()
            .zip(cpu_out[base..base + head_dim].iter())
            .map(|(&g, &c)| (g - c).abs())
            .fold(0f32, f32::max);
        let row_gpu_max = out_gpu[base..base + head_dim]
            .iter()
            .fold(0f32, |acc, &v| acc.max(v.abs()));
        eprintln!(
            "[N4]   row {:2}: max_abs={:.4e}  gpu_row_max_abs={:.4e}",
            row, row_max, row_gpu_max
        );
    }

    free_all(&all_ptrs);

    // Diagnostic assertions — these classify the outcome. We don't gate on
    // a tight tolerance because N4's purpose is observation, not regression.
    // The interpretation goes in the test output and the project memory.
    if n_nan > 0 {
        eprintln!("[N4] DIAGNOSIS: non-finite GPU outputs — kernel produced NaN/Inf");
    } else if n_zero == qkv_elems {
        eprintln!("[N4] DIAGNOSIS: all-zero GPU output — kernel may have crashed / never wrote");
    } else if max_abs < 5e-3 {
        eprintln!("[N4] DIAGNOSIS (a): helper convention is CORRECT (max_abs={:.4e} within f16 tolerance)", max_abs);
    } else if max_abs < 1.0 {
        eprintln!("[N4] DIAGNOSIS (b): STRUCTURED drift (max_abs={:.4e}); see per-row pattern above", max_abs);
    } else {
        eprintln!("[N4] DIAGNOSIS (c): UNSTRUCTURED drift (max_abs={:.4e}); helper likely needs PTX-spec rewrite", max_abs);
    }
}
