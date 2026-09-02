//! GPU dispatch validation for the gradient-accumulation FFIs in zero.rs.
//!
//! Pre-fix, the single-GPU FullBuffer accumulation path was silently wrong:
//! `nsl_grad_accumulate_add` returned -1 (discarded by codegen) for any
//! GPU-resident tensor, so accumulation buffers stayed zero and the
//! optimizer stepped on zero gradients — loss printed normally, params
//! frozen. `nsl_grad_zero` was likewise a refusing no-op on GPU buffers.
//! These tests pin the corrected behavior: real device accumulation /
//! zeroing with return code 0 and no refcount leaks from the internal
//! `nsl_tensor_to_device_like` migration.
//!
//! All tests are `#[ignore]` so they only run when CUDA is available.
//! Running locally:
//!
//! ```bash
//! cargo test --package nsl-runtime --features cuda \
//!     --test zero_grad_gpu -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_runtime::nsl_cuda_init;

// FFIs under test (zero.rs) plus tensor/list helpers (tensor/mod.rs, list.rs).
unsafe extern "C" {
    fn nsl_grad_accumulate_add(dst_ptr: i64, src_ptr: i64, num_elems: i64) -> i64;
    fn nsl_grad_zero(grad_ptr: i64, num_elems: i64) -> i64;
    fn nsl_zero_init(stage: i64, world_size: i64) -> i64;
    fn nsl_zero_destroy() -> i64;
    fn nsl_zero_reduce_grads(grads_list_ptr: i64, num_params: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
}

// NslTensor layout — minimal mirror to read fields we care about.
#[repr(C)]
struct TensorView {
    _magic: u32,
    _data: *mut std::ffi::c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    _ndim: i64,
    len: i64,
    refcount: std::sync::atomic::AtomicI64,
    device: u8,
    dtype: u16,
    _owns_data: u8,
    _data_owner: i64,
    _slab_managed: u8,
    _tape_id: i64,
}

const DTYPE_F64: i64 = 0;
const DTYPE_F32: i64 = 1;

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

fn refcount(ptr: i64) -> i64 {
    let t = unsafe { &*(ptr as *const TensorView) };
    t.refcount.load(std::sync::atomic::Ordering::SeqCst)
}

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

/// Build a CPU f64 tensor (the canonical CPU dtype tape-AD gradients use).
fn cpu_f64_tensor(vals: &[f64]) -> i64 {
    let boxed: Box<[f64]> = vals.to_vec().into_boxed_slice();
    let leaked: &'static [f64] = Box::leak(boxed);
    let data_ptr = leaked.as_ptr() as i64;
    let shape_list = unsafe { nsl_list_new() };
    unsafe { nsl_list_push(shape_list, vals.len() as i64) };
    let t = unsafe { nsl_tensor_from_static(data_ptr, shape_list, DTYPE_F64) };
    unsafe { nsl_list_free(shape_list) };
    t
}

/// Read a GPU f32 tensor back to host as `Vec<f32>`.
/// `nsl_tensor_to_device(ptr, 0)` upcasts GPU f32 -> CPU f64 (runtime
/// invariant: CPU=f64, GPU=f32); narrow back to f32 for comparison.
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

/// (a) GPU dst += GPU src: the exact shape of the live single-GPU
/// FullBuffer bug (accum buffer and tape gradient both device-resident).
/// Also (e): the internal to_device_like migration is a same-device
/// refcount++ that must be balanced by exactly one free.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dst_gpu_src_accumulate_matches_cpu_reference() {
    if !cuda_available() {
        return;
    }
    let dst_vals = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
    let src_vals = [10.0_f32, 20.0, 30.0, 40.0, 50.0];

    let cpu_dst = cpu_f32_tensor(&dst_vals);
    let cpu_src = cpu_f32_tensor(&src_vals);
    let gpu_dst = unsafe { nsl_tensor_to_device(cpu_dst, 1) };
    let gpu_src = unsafe { nsl_tensor_to_device(cpu_src, 1) };

    let rc_before = refcount(gpu_src);
    let rc = unsafe { nsl_grad_accumulate_add(gpu_dst, gpu_src, dst_vals.len() as i64) };
    assert_eq!(rc, 0, "GPU accumulate must succeed (pre-fix: silent -1)");
    assert_eq!(
        refcount(gpu_src),
        rc_before,
        "migrated ref must be freed exactly once (no leak, no double-free)"
    );

    let got = read_gpu_f32_as_host_f32(gpu_dst);
    for (i, (&d, &s)) in dst_vals.iter().zip(src_vals.iter()).enumerate() {
        assert_eq!(got[i], d + s, "elem {i}: {} + {} != {}", d, s, got[i]);
    }

    unsafe {
        nsl_tensor_free(cpu_dst);
        nsl_tensor_free(cpu_src);
        nsl_tensor_free(gpu_dst);
        nsl_tensor_free(gpu_src);
    }
}

/// (b) GPU dst += CPU f64 src: the tape-AD shape of the bug (backward
/// produces CPU f64 gradients while the accum buffer sits on the GPU).
/// The src must be migrated (f64 -> GPU f32) and freed, leaving the
/// caller's tensor untouched.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_dst_cpu_f64_src_accumulate() {
    if !cuda_available() {
        return;
    }
    let dst_vals = [0.5_f32, 1.5, 2.5, 3.5];
    let src_vals = [100.0_f64, 200.0, 300.0, 400.0];

    let cpu_dst = cpu_f32_tensor(&dst_vals);
    let gpu_dst = unsafe { nsl_tensor_to_device(cpu_dst, 1) };
    let cpu_src = cpu_f64_tensor(&src_vals);

    let rc_before = refcount(cpu_src);
    let rc = unsafe { nsl_grad_accumulate_add(gpu_dst, cpu_src, dst_vals.len() as i64) };
    assert_eq!(rc, 0, "GPU dst + CPU f64 src accumulate must succeed");
    assert_eq!(
        refcount(cpu_src),
        rc_before,
        "cross-device migration must not leak refs on the source tensor"
    );
    // Source data untouched.
    let src_view = unsafe { &*(cpu_src as *const TensorView) };
    let src_now =
        unsafe { std::slice::from_raw_parts(src_view._data as *const f64, src_vals.len()) };
    assert_eq!(src_now, &src_vals, "src gradient must not be mutated");

    let got = read_gpu_f32_as_host_f32(gpu_dst);
    for (i, (&d, &s)) in dst_vals.iter().zip(src_vals.iter()).enumerate() {
        assert_eq!(got[i], d + s as f32, "elem {i}");
    }

    unsafe {
        nsl_tensor_free(cpu_dst);
        nsl_tensor_free(gpu_dst);
        nsl_tensor_free(cpu_src);
    }
}

/// (c) nsl_grad_zero on a GPU buffer: device memset, rc 0, all zeros on
/// readback (pre-fix: rc -1 and the buffer kept its stale values, so the
/// next accumulation window started from the previous window's sum).
#[test]
#[ignore = "requires CUDA GPU"]
fn grad_zero_gpu_clears_buffer() {
    if !cuda_available() {
        return;
    }
    let vals = [5.5_f32, -3.25, 7.0, 0.125, 9.0, 42.0, -1.0];
    let cpu = cpu_f32_tensor(&vals);
    let gpu = unsafe { nsl_tensor_to_device(cpu, 1) };

    let rc = unsafe { nsl_grad_zero(gpu, vals.len() as i64) };
    assert_eq!(rc, 0, "GPU grad_zero must succeed (pre-fix: silent -1)");

    let got = read_gpu_f32_as_host_f32(gpu);
    assert_eq!(got.len(), vals.len());
    for (i, &g) in got.iter().enumerate() {
        assert_eq!(g.to_bits(), 0.0_f32.to_bits(), "elem {i} must be +0.0");
    }

    unsafe {
        nsl_tensor_free(cpu);
        nsl_tensor_free(gpu);
    }
}

/// (f) `nsl_zero_reduce_grads` must never SILENTLY skip a device-resident
/// gradient. The original bug was exactly that: GPU tensors fell through the
/// reduce loop untouched, so ranks kept unsynchronised gradients and training
/// diverged with nothing in the logs.
///
/// This gate used to assert the divide succeeded under a single-process
/// `nsl_zero_init(1, 4)`. Two deliberate hardenings since made that
/// unreachable, and asserting it was asserting behaviour the codebase has
/// intentionally removed:
///
///   1. `nsl_zero_init` refuses world_size > 1 without `NSL_TP_SHM_PATH`
///      (-3) — pinned by zero.rs's own `test_zero_ffi_lifecycle`.
///   2. A device-resident gradient under a non-CUDA-aware backend is REFUSED
///      with -5 ("GPU-resident ZeRO SPMD needs real collectives") instead of
///      being skipped. That refusal IS the fix for the original bug.
///
/// Driving a genuine CUDA-aware reduce needs a real multi-rank spawn: the
/// `sim-gpu` backend aborts when a single process claims ws=4. That path is
/// covered end-to-end by `nsl-cli/tests/zero_gpu_collectives_gate.rs`, which
/// goes through `nsl run --devices 2 --collectives sim-gpu`.
///
/// So what this gate pins now is the property that actually regressed once
/// and could regress again: the failure is LOUD and NON-DESTRUCTIVE — a
/// refusal code, with the gradient left bit-for-bit untouched rather than
/// half-reduced.
#[test]
#[ignore = "requires CUDA GPU"]
fn zero_reduce_grads_refuses_gpu_tensor_without_cuda_aware_backend() {
    if !cuda_available() {
        return;
    }
    unsafe { nsl_zero_destroy() };
    let shm_path = std::env::temp_dir().join(format!(
        "nsl_zero_grad_gpu_{}.shm",
        std::process::id()
    ));
    {
        let f = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&shm_path)
            .expect("create shm file");
        // Header + 64 MiB per rank, matching commands/run.rs's spawner.
        f.set_len(64 + 4 * 64 * 1024 * 1024).expect("size shm file");
    }
    unsafe {
        std::env::set_var("NSL_TP_SHM_PATH", &shm_path);
        std::env::set_var("NSL_LOCAL_RANK", "1");
        std::env::set_var("NSL_COLLECTIVES", "sim");
    }
    // Teardown must survive a failing assertion. These vars are PROCESS-global
    // and three other tests share this binary, so leaking them on the error
    // path would silently reconfigure whatever runs next — and the shm file is
    // ~256 MiB to strand on disk. Cleanup matters most exactly when the test
    // fails, so it belongs in a Drop guard rather than at the happy-path end.
    struct EnvShmGuard(std::path::PathBuf);
    impl Drop for EnvShmGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("NSL_TP_SHM_PATH");
                std::env::remove_var("NSL_LOCAL_RANK");
                std::env::remove_var("NSL_COLLECTIVES");
            }
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _guard = EnvShmGuard(shm_path.clone());

    assert_eq!(
        unsafe { nsl_zero_init(1, 4) },
        0,
        "ws=4 init must succeed once NSL_TP_SHM_PATH exists"
    );

    let vals = [4.0_f32, 8.0, 12.0, 16.0];
    let cpu = cpu_f32_tensor(&vals);
    let gpu = unsafe { nsl_tensor_to_device(cpu, 1) };
    let list = unsafe { nsl_list_new() };
    unsafe { nsl_list_push(list, gpu) };

    let rc = unsafe { nsl_zero_reduce_grads(list, 1) };
    assert_eq!(
        rc, -5,
        "a device-resident gradient under the CPU-shm `sim` backend must be \
         REFUSED, not silently skipped"
    );

    // The load-bearing half: refusing must not have partially mutated the
    // gradient. A half-applied divisor would be worse than either extreme.
    let got = read_gpu_f32_as_host_f32(gpu);
    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(got[i], v, "elem {i} must be untouched by a refused reduce");
    }

    assert_eq!(unsafe { nsl_zero_destroy() }, 0);
    unsafe {
        nsl_list_free(list);
        nsl_tensor_free(cpu);
        nsl_tensor_free(gpu);
    }
    // env vars + shm file released by `_guard`.
}
