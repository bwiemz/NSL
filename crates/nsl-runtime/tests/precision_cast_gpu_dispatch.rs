//! CFTP v7 — GPU dispatch validation for the precision-cast FFIs.
//!
//! These tests exercise `nsl_tensor_to_{bf16,fp16,f32}` on device tensors,
//! confirming the new GPU branch in `cast_and_publish` produces results
//! within bf16/fp16 numeric tolerance after a round-trip through the PTX
//! kernel.
//!
//! All tests are `#[ignore]` so they only run when CUDA is available; CI
//! invokes them with `--ignored`. Running locally:
//!
//! ```bash
//! cargo test --package nsl-runtime --features cuda \
//!     --test precision_cast_gpu_dispatch -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Critical invariants pinned here:
//! * Round-trip through bf16 stays within bf16 truncating eps (~7.9e-3).
//! * Round-trip through fp16 stays within fp16 eps (~1.1e-3).
//! * Result tensor has correct shape, len, device, dtype.
//! * Coverage at numel = 1 / 17 (not block-aligned) / 4096 / 1 << 20.

#![cfg(feature = "cuda")]

use nsl_runtime::nsl_cuda_init;

// Cast FFIs under test (from precision_cast.rs).
extern "C" {
    fn nsl_tensor_to_bf16(src_ptr: i64) -> i64;
    fn nsl_tensor_to_fp16(src_ptr: i64) -> i64;
    fn nsl_tensor_to_f32(src_ptr: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
}

// NslTensor layout — minimal mirror to read fields we care about.
// The runtime exposes `crate::tensor::NslTensor` but it's not in the public
// API; reproduce just the fields this test reads.
#[repr(C)]
struct TensorView {
    _magic: u32,
    _data: *mut std::ffi::c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    ndim: i64,
    len: i64,
    _refcount: std::sync::atomic::AtomicI64,
    device: u8,
    dtype: u16,
    _owns_data: u8,
    _data_owner: i64,
    _slab_managed: u8,
    _tape_id: i64,
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

// Public C ABI tensor constructors used here. `nsl_tensor_from_static`
// wraps a static (non-owning) buffer; we leak a heap-allocated f32 slice
// so the buffer outlives the test (the resulting tensor has owns_data=0
// and never frees `data`). The leaks are tiny and per-test.
extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
}

const DTYPE_F32: i64 = 1;

/// Build a CPU f32 tensor from a slice via `nsl_tensor_from_static`. The
/// underlying buffer is leaked (owns_data=0); per-test leak is small.
fn cpu_f32_tensor(vals: &[f32]) -> i64 {
    let boxed: Box<[f32]> = vals.to_vec().into_boxed_slice();
    let leaked: &'static [f32] = Box::leak(boxed);
    let data_ptr = leaked.as_ptr() as i64;
    let shape_list = unsafe { nsl_list_new() };
    unsafe { nsl_list_push(shape_list, vals.len() as i64) };
    let t = unsafe { nsl_tensor_from_static(data_ptr, shape_list, DTYPE_F32) };
    unsafe { nsl_list_free(shape_list) };
    t
}

/// Read a GPU f32 tensor back to host as `Vec<f32>`.
///
/// `nsl_tensor_to_device(ptr, 0)` upcasts GPU f32 -> CPU f64 (runtime
/// invariant from MEMORY.md: CPU=f64, GPU=f32). We read the f64 buffer
/// and narrow to f32 for the round-trip-error comparison.
fn read_gpu_f32_as_host_f32(ptr: i64) -> Vec<f32> {
    let cpu = unsafe { nsl_tensor_to_device(ptr, 0) };
    let t = unsafe { &*(cpu as *const TensorView) };
    assert_eq!(t.dtype, 0, "to_device(_, 0) must yield CPU f64 (dtype=0)");
    let len = t.len as usize;
    let slice = unsafe { std::slice::from_raw_parts(t._data as *const f64, len) };
    let out: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
    if cpu != ptr {
        unsafe { nsl_tensor_free(cpu) };
    }
    out
}

/// Round-trip f32 -> bf16 -> f32 ON THE GPU.
/// bf16 has 7 explicit mantissa bits; truncating cvt gives worst-case ~7.8e-3
/// relative error. The kernel uses `cvt.rn.bf16.f32` (RTE), which is slightly
/// tighter than the CPU primitive's truncation, so we keep the same 7.9e-3
/// bound the CPU round-trip test uses (a tighter bound would diverge from the
/// CPU contract and create a hard-to-debug numerical gap).
#[test]
#[ignore]
fn gpu_to_bf16_round_trip_within_epsilon() {
    if !cuda_available() {
        return;
    }
    let vals = [1.0001_f32, 3.14159265, -2.71828, 1234.5, 1e-3, -1e3];
    let cpu_src = cpu_f32_tensor(&vals);
    let gpu_src = unsafe { nsl_tensor_to_device(cpu_src, 1) };

    let gpu_bf16 = unsafe { nsl_tensor_to_bf16(gpu_src) };
    let t = unsafe { &*(gpu_bf16 as *const TensorView) };
    assert_eq!(t.dtype, 3, "result must be tagged bf16 (dtype=3)");
    assert_eq!(t.device, 1, "result must remain on GPU (device=1)");
    assert_eq!(t.len, vals.len() as i64);
    assert_eq!(t.ndim, 1);

    let gpu_back = unsafe { nsl_tensor_to_f32(gpu_bf16) };
    let got = read_gpu_f32_as_host_f32(gpu_back);
    for (i, &v) in vals.iter().enumerate() {
        let rel = ((got[i] - v) / v).abs();
        assert!(
            rel <= 7.9e-3,
            "elem {i}: v={v} got={} rel_err={} (bf16 eps ~7.8e-3)",
            got[i],
            rel
        );
    }

    unsafe {
        nsl_tensor_free(cpu_src);
        nsl_tensor_free(gpu_src);
        nsl_tensor_free(gpu_bf16);
        nsl_tensor_free(gpu_back);
    }
}

/// Round-trip f32 -> fp16 -> f32 ON THE GPU.
/// fp16 has 10 mantissa bits, eps ~9.8e-4. The CPU primitive truncates, the
/// GPU `cvt.rn.f16.f32` is RTE, but we keep the CPU-test bound (1.1e-3) so
/// the contract is uniform.
#[test]
#[ignore]
fn gpu_to_fp16_round_trip_within_epsilon() {
    if !cuda_available() {
        return;
    }
    let vals = [1.0001_f32, 3.14159265, -2.71828, 1234.5, 1.5e-2, -512.0];
    let cpu_src = cpu_f32_tensor(&vals);
    let gpu_src = unsafe { nsl_tensor_to_device(cpu_src, 1) };

    let gpu_fp16 = unsafe { nsl_tensor_to_fp16(gpu_src) };
    let t = unsafe { &*(gpu_fp16 as *const TensorView) };
    assert_eq!(t.dtype, 2, "result must be tagged fp16 (dtype=2)");
    assert_eq!(t.device, 1, "result must remain on GPU (device=1)");
    assert_eq!(t.len, vals.len() as i64);

    let gpu_back = unsafe { nsl_tensor_to_f32(gpu_fp16) };
    let got = read_gpu_f32_as_host_f32(gpu_back);
    for (i, &v) in vals.iter().enumerate() {
        let rel = ((got[i] - v) / v).abs();
        assert!(
            rel <= 1.1e-3,
            "elem {i}: v={v} got={} rel_err={} (fp16 eps ~9.8e-4)",
            got[i],
            rel
        );
    }

    unsafe {
        nsl_tensor_free(cpu_src);
        nsl_tensor_free(gpu_src);
        nsl_tensor_free(gpu_fp16);
        nsl_tensor_free(gpu_back);
    }
}

/// Edge cases: numel = 1, 17 (not a multiple of 256), 4096 (block-aligned),
/// 1 << 20 (multi-grid). Confirms the grid-stride loop covers every regime.
#[test]
#[ignore]
fn gpu_to_bf16_size_sweep() {
    if !cuda_available() {
        return;
    }
    for &n in &[1_usize, 17, 4096, 1 << 20] {
        // Pattern: small magnitudes inside bf16's representable range.
        let vals: Vec<f32> = (0..n).map(|i| ((i % 1024) as f32) * 0.125 + 0.5).collect();
        let cpu_src = cpu_f32_tensor(&vals);
        let gpu_src = unsafe { nsl_tensor_to_device(cpu_src, 1) };

        let gpu_bf16 = unsafe { nsl_tensor_to_bf16(gpu_src) };
        let gpu_back = unsafe { nsl_tensor_to_f32(gpu_bf16) };
        let got = read_gpu_f32_as_host_f32(gpu_back);

        assert_eq!(got.len(), n, "n={n}: length mismatch");
        // Spot-check a sample of elements (full-sweep at n=1M would be slow
        // but still correct). For n <= 4096 we check every element.
        let stride = if n > 4096 { n / 256 } else { 1 };
        for i in (0..n).step_by(stride) {
            let rel = ((got[i] - vals[i]) / vals[i]).abs();
            assert!(
                rel <= 7.9e-3,
                "n={n} elem {i}: v={} got={} rel_err={}",
                vals[i],
                got[i],
                rel
            );
        }
        unsafe {
            nsl_tensor_free(cpu_src);
            nsl_tensor_free(gpu_src);
            nsl_tensor_free(gpu_bf16);
            nsl_tensor_free(gpu_back);
        }
    }
}

/// Same-dtype "cast" — f32 -> f32 on GPU must produce a NEW buffer (not
/// alias) and the values must be byte-identical (memcpy_dtod fast path).
#[test]
#[ignore]
fn gpu_to_f32_same_dtype_is_copy() {
    if !cuda_available() {
        return;
    }
    let vals = [1.0_f32, -2.5, 3.14, 0.0, -0.0, 1e10];
    let cpu_src = cpu_f32_tensor(&vals);
    let gpu_src = unsafe { nsl_tensor_to_device(cpu_src, 1) };
    let gpu_src_data = unsafe { (*(gpu_src as *const TensorView))._data };

    let gpu_copy = unsafe { nsl_tensor_to_f32(gpu_src) };
    let gpu_copy_data = unsafe { (*(gpu_copy as *const TensorView))._data };
    assert_ne!(
        gpu_src_data, gpu_copy_data,
        "GPU same-dtype cast must allocate a NEW buffer, not alias"
    );

    let back = read_gpu_f32_as_host_f32(gpu_copy);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(back[i].to_bits(), v.to_bits(), "elem {i}: bit-identical");
    }

    unsafe {
        nsl_tensor_free(cpu_src);
        nsl_tensor_free(gpu_src);
        nsl_tensor_free(gpu_copy);
    }
}

// ── Deferral-closure 2026-07-14: the GPU optimizer-moment envelope ──────────
//
// PR #367 refused reduced-precision optimizer moments on GPU because the
// dequant→step→quant envelope (`nsl_tensor_zeros_like_dtype` →
// `nsl_tensor_cast` → `nsl_tensor_cast_into`) was CPU-only. All three
// primitives now dispatch to the CFTP-v7 PTX cast kernels; this test
// simulates one full envelope step exactly as `stmt_fase.rs` emits it.

extern "C" {
    fn nsl_tensor_zeros_like_dtype(template_ptr: i64, dtype: i64) -> i64;
    fn nsl_tensor_cast(src_ptr: i64, target_dtype: i64) -> i64;
    fn nsl_tensor_cast_into(dst_ptr: i64, src_ptr: i64);
    fn nsl_test_cuda_h2d(dst_device: i64, src_host: i64, size_bytes: i64) -> i64;
}

const DTYPE_FP16: i64 = 2;

/// f32 -> fp16 (truncating, mirrors the CPU primitive's contract loosely —
/// the GPU kernel uses RTE which is tighter; used only to build EXPECTED
/// values with an fp16-eps tolerance).
fn fp16_round_trip(v: f32) -> f32 {
    // Go through Rust's half-emulation via bit twiddling is overkill here;
    // an fp16 storage step keeps |err| <= max(|v|, 6.1e-5) * 9.8e-4 for
    // RTE. The assertion below uses that bound rather than exact bits.
    v
}

/// One simulated FASE step with FP16 moment storage, fully on-device:
///   m_fp16 = zeros_like_dtype(param_gpu, FP16)      [allocation]
///   m_f32  = cast(m_fp16, F32)                       [dequant]
///   m_f32 <- simulated update (h2d write)            [the "step"]
///   cast_into(m_fp16, m_f32)                         [quant-back, in place]
///   verify: cast(m_fp16, F32) matches update within fp16 eps,
///           m_fp16 buffer identity is PRESERVED across the step.
#[test]
#[ignore]
fn gpu_moment_envelope_fp16_round_trip() {
    if !cuda_available() {
        return;
    }
    let n = 4096 + 17; // not block-aligned on purpose
    let param_vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 2.0).collect();
    let cpu_param = cpu_f32_tensor(&param_vals);
    let gpu_param = unsafe { nsl_tensor_to_device(cpu_param, 1) };

    // 1. Allocation: FP16 moment buffer on-device, zero-filled.
    let m_fp16 = unsafe { nsl_tensor_zeros_like_dtype(gpu_param, DTYPE_FP16) };
    let mv = unsafe { &*(m_fp16 as *const TensorView) };
    assert_eq!(mv.device, 1, "moment buffer must be GPU-resident");
    assert_eq!(mv.dtype, DTYPE_FP16 as u16, "moment buffer must be FP16 storage");
    assert_eq!(mv.len as usize, n);
    let m_fp16_data_before = mv._data;

    // 2. Dequant: widen to f32 working tensor (new device buffer).
    let m_f32 = unsafe { nsl_tensor_cast(m_fp16, DTYPE_F32) };
    let wv = unsafe { &*(m_f32 as *const TensorView) };
    assert_eq!(wv.device, 1);
    assert_eq!(wv.dtype, DTYPE_F32 as u16);
    let zeros = read_gpu_f32_as_host_f32(m_f32);
    assert!(
        zeros.iter().all(|&x| x == 0.0),
        "freshly allocated FP16 moments must dequant to exact zeros"
    );

    // 3. Simulated optimizer update: overwrite the f32 working buffer.
    let update: Vec<f32> = (0..n).map(|i| ((i % 613) as f32) * 3e-3 - 0.9).collect();
    unsafe {
        nsl_test_cuda_h2d(
            wv._data as i64,
            update.as_ptr() as i64,
            (n * 4) as i64,
        )
    };

    // 4. Quant-back IN PLACE into the persistent FP16 buffer.
    unsafe { nsl_tensor_cast_into(m_fp16, m_f32) };
    let mv_after = unsafe { &*(m_fp16 as *const TensorView) };
    assert_eq!(
        m_fp16_data_before, mv_after._data,
        "cast_into must preserve the moment buffer's identity (no realloc)"
    );

    // 5. Re-dequant and verify within fp16 RTE eps.
    let m_f32_check = unsafe { nsl_tensor_cast(m_fp16, DTYPE_F32) };
    let got = read_gpu_f32_as_host_f32(m_f32_check);
    for (i, (&g, &u)) in got.iter().zip(update.iter()).enumerate() {
        let _ = fp16_round_trip(u);
        let tol = f32::max(u.abs(), 6.1e-5) * 9.8e-4 + 1e-7;
        assert!(
            (g - u).abs() <= tol,
            "elem {i}: fp16 moment round-trip {g} vs update {u} (tol {tol})"
        );
    }

    unsafe {
        nsl_tensor_free(m_f32_check);
        nsl_tensor_free(m_f32);
        nsl_tensor_free(m_fp16);
        nsl_tensor_free(gpu_param);
        nsl_tensor_free(cpu_param);
    }
}
