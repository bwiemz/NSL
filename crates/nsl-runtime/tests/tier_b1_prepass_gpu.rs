//! GPU validation for the CSHA Tier B.1 production pre-pass kernels.
//!
//! Two independent tests:
//!   * `x_prepass_matches_cpu_reference` — launches `csha_tier_b1_prepass_x`
//!     and compares the f16 chunkified output against a CPU implementation
//!     of RMSNorm + narrow + chunkify.
//!   * `w_prepass_matches_cpu_reference` — launches `csha_tier_b1_prepass_w`
//!     and compares against a CPU implementation of narrow + col-major
//!     chunkify.
//!
//! These tests validate the pre-pass kernels in ISOLATION, before they are
//! orchestrated together inside `nsl_flash_attention_csha`. The full
//! orchestration test lives at `tier_b1_n4_disambiguation` (re-enabled
//! once the orchestration wiring lands).
//!
//! Running:
//! ```bash
//! cargo test --package nsl-runtime --features cuda --test tier_b1_prepass_gpu \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, CSHA_TIER_B1_PREPASS_W_PTX, CSHA_TIER_B1_PREPASS_X_PTX,
};

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

#[test]
#[ignore = "requires CUDA GPU"]
fn x_prepass_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }
    let ptx = CSHA_TIER_B1_PREPASS_X_PTX;
    let kernel_name = std::ffi::CString::new("csha_tier_b1_prepass_x").unwrap();

    let seq: usize = 32;
    let d_model: usize = 2048;
    let chunk: usize = 128;
    let eps: f32 = 1e-5;

    // Host inputs.
    let mut x_host = vec![0f32; seq * d_model];
    for s in 0..seq {
        for d in 0..d_model {
            x_host[s * d_model + d] = ((s + d) as f32).sin() * 0.1 + 0.05;
        }
    }
    let gamma_host = vec![1.0f32; d_model];

    // CPU reference: RMSNorm + narrow + chunkify.
    let mut expected_chunked = vec![0u16; (d_model / chunk) * seq * chunk];
    for s in 0..seq {
        let row = &x_host[s * d_model..(s + 1) * d_model];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / d_model as f32;
        let rms_inv = (mean_sq + eps).sqrt().recip();
        for d in 0..d_model {
            let val = row[d] * rms_inv * gamma_host[d];
            let chunk_idx = d / chunk;
            let c = d % chunk;
            let idx = chunk_idx * seq * chunk + s * chunk + c;
            expected_chunked[idx] = f32_to_f16_bits(val);
        }
    }

    // GPU buffers.
    let x_bytes = (seq * d_model * 4) as i64;
    let gamma_bytes = (d_model * 4) as i64;
    let out_bytes = (expected_chunked.len() * 2) as i64;
    let x_dev = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let gamma_dev = unsafe { nsl_test_cuda_alloc(gamma_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    assert!(x_dev != 0 && gamma_dev != 0 && out_dev != 0);
    unsafe {
        nsl_test_cuda_h2d(x_dev, x_host.as_ptr() as i64, x_bytes);
        nsl_test_cuda_h2d(gamma_dev, gamma_host.as_ptr() as i64, gamma_bytes);
    }

    // Args: x_in, gamma, x_out, seq, d_model, chunk, log2_chunk, eps.
    let mut x_in = x_dev as u64;
    let mut gamma_p = gamma_dev as u64;
    let mut x_out = out_dev as u64;
    let mut seq_v = seq as u64;
    let mut dm_v = d_model as u64;
    let mut chunk_v = chunk as u64;
    let mut log2c: u32 = (chunk as u64).trailing_zeros();
    let mut eps_v = eps;
    let args: [*mut std::ffi::c_void; 8] = [
        &mut x_in as *mut _ as *mut std::ffi::c_void,
        &mut gamma_p as *mut _ as *mut std::ffi::c_void,
        &mut x_out as *mut _ as *mut std::ffi::c_void,
        &mut seq_v as *mut _ as *mut std::ffi::c_void,
        &mut dm_v as *mut _ as *mut std::ffi::c_void,
        &mut chunk_v as *mut _ as *mut std::ffi::c_void,
        &mut log2c as *mut _ as *mut std::ffi::c_void,
        &mut eps_v as *mut _ as *mut std::ffi::c_void,
    ];

    let rc = unsafe {
        nsl_kernel_launch(
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            seq as i64, 1, 1,
            256, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            0,
        )
    };
    assert_eq!(rc, 0, "x prepass launch failed: rc={}", rc);

    let mut got = vec![0u16; expected_chunked.len()];
    unsafe { nsl_test_cuda_d2h(got.as_mut_ptr() as i64, out_dev, out_bytes) };

    unsafe {
        nsl_test_cuda_free(x_dev);
        nsl_test_cuda_free(gamma_dev);
        nsl_test_cuda_free(out_dev);
    }

    // Compare bit-exact (CPU uses the same RMSNorm formula + f16 narrowing;
    // GPU uses rsqrt.approx which is ~24-bit, slightly diff from libm sqrt;
    // allow a small ULP-level tolerance).
    let mut max_abs = 0f32;
    let mut mismatches = 0;
    for (i, (&e, &g)) in expected_chunked.iter().zip(got.iter()).enumerate() {
        let e_f = f16_to_f32(e);
        let g_f = f16_to_f32(g);
        let d = (e_f - g_f).abs();
        if d > max_abs {
            max_abs = d;
        }
        if d > 1e-2 {
            if mismatches < 5 {
                eprintln!("[x-prepass] mismatch @ {}: expected {:.6} got {:.6}", i, e_f, g_f);
            }
            mismatches += 1;
        }
    }
    eprintln!(
        "[x-prepass] max_abs={:.4e} mismatches(>1e-2)={}",
        max_abs, mismatches
    );
    assert!(
        max_abs < 5e-2,
        "x prepass output drift exceeds 5e-2 (rsqrt.approx + f16 round-trip noise budget); got {:.4e}",
        max_abs
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn w_prepass_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }
    let ptx = CSHA_TIER_B1_PREPASS_W_PTX;
    let kernel_name = std::ffi::CString::new("csha_tier_b1_prepass_w").unwrap();

    let d_model: usize = 2048;
    let hd: usize = 32;
    let chunk: usize = 128;

    let mut w_host = vec![0f32; d_model * hd];
    for d in 0..d_model {
        for n in 0..hd {
            w_host[d * hd + n] = ((d * 7 + n) as f32).sin() * 0.05;
        }
    }

    // CPU reference: narrow + col-major chunkify.
    let n_chunks = d_model / chunk;
    let mut expected = vec![0u16; n_chunks * hd * chunk];
    for chunk_idx in 0..n_chunks {
        for n in 0..hd {
            for k_in_chunk in 0..chunk {
                let d_row = chunk_idx * chunk + k_in_chunk;
                let val = w_host[d_row * hd + n];
                expected[chunk_idx * hd * chunk + n * chunk + k_in_chunk] = f32_to_f16_bits(val);
            }
        }
    }

    let w_bytes = (w_host.len() * 4) as i64;
    let out_bytes = (expected.len() * 2) as i64;
    let w_dev = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let out_dev = unsafe { nsl_test_cuda_alloc(out_bytes) };
    assert!(w_dev != 0 && out_dev != 0);
    unsafe { nsl_test_cuda_h2d(w_dev, w_host.as_ptr() as i64, w_bytes) };

    let mut w_in = w_dev as u64;
    let mut w_out = out_dev as u64;
    let mut dm_v = d_model as u64;
    let mut hd_v = hd as u64;
    let mut chunk_v = chunk as u64;
    let mut log2h: u32 = (hd as u64).trailing_zeros();
    let mut log2c: u32 = (chunk as u64).trailing_zeros();
    let args: [*mut std::ffi::c_void; 7] = [
        &mut w_in as *mut _ as *mut std::ffi::c_void,
        &mut w_out as *mut _ as *mut std::ffi::c_void,
        &mut dm_v as *mut _ as *mut std::ffi::c_void,
        &mut hd_v as *mut _ as *mut std::ffi::c_void,
        &mut chunk_v as *mut _ as *mut std::ffi::c_void,
        &mut log2h as *mut _ as *mut std::ffi::c_void,
        &mut log2c as *mut _ as *mut std::ffi::c_void,
    ];

    let total: i64 = (d_model * hd) as i64;
    let grid_x = (total + 255) / 256;
    let rc = unsafe {
        nsl_kernel_launch(
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            grid_x, 1, 1,
            256, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            0,
        )
    };
    assert_eq!(rc, 0, "w prepass launch failed: rc={}", rc);

    let mut got = vec![0u16; expected.len()];
    unsafe { nsl_test_cuda_d2h(got.as_mut_ptr() as i64, out_dev, out_bytes) };

    unsafe {
        nsl_test_cuda_free(w_dev);
        nsl_test_cuda_free(out_dev);
    }

    // W pre-pass is pure narrow+chunkify (no RMSNorm) → should be bit-exact
    // (every byte comes from a single f32->f16 cvt.rn with the same input).
    let mut mismatches = 0;
    for (i, (&e, &g)) in expected.iter().zip(got.iter()).enumerate() {
        if e != g {
            if mismatches < 5 {
                let e_f = f16_to_f32(e);
                let g_f = f16_to_f32(g);
                eprintln!("[w-prepass] mismatch @ {}: expected 0x{:04x}={:.6} got 0x{:04x}={:.6}", i, e, e_f, g, g_f);
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{} elements differ between CPU and GPU w prepass", mismatches);
    eprintln!("[w-prepass] all {} elements bit-exact match", expected.len());
}

// Local copy of f32_to_f16_bits matching the kernel's cvt.rn.f16.f32 rounding.
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
