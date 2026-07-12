//! Regression gate: GPU maxpool2d must not host-dereference its device argmax
//! buffer.
//!
//! `gpu_maxpool2d_f32` allocates the argmax index buffer with `alloc_managed`,
//! which despite its name returns a plain DEVICE pointer (it routes through the
//! caching allocator's `cuMemAlloc`, never `cuMemAllocManaged`). The readback
//! used to build a host slice straight over that device pointer with
//! `std::slice::from_raw_parts` — a host dereference of device memory, which
//! SIGSEGVs on a discrete GPU and aborts the whole process (the same bug class
//! already fixed in `test_vec_add_kernel_launch`). The fix stages it through
//! `memcpy_dtoh`. The readback runs unconditionally on every GPU maxpool call,
//! so simply completing this call without a fault is the regression signal;
//! the value check confirms the pool is still correct.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test gpu_maxpool2d_argmax_readback
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_maxpool2d, nsl_tensor_zeros_on,
};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

fn gpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    nsl_test_cuda_h2d(
        nsl_tensor_data_ptr(t),
        vals.as_ptr() as i64,
        (vals.len() * 4) as i64,
    );
    t
}

fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    nsl_test_cuda_d2h(
        out.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(t),
        (len * 4) as i64,
    );
    out
}

/// A 4x4 single-channel input pooled 2x2 (stride 2, no pad) -> 2x2. Reaching
/// the value assertion at all means the device-side argmax readback completed
/// without a host-deref SIGSEGV.
#[test]
#[ignore = "requires a CUDA GPU; run with --features cuda -- --ignored"]
fn gpu_maxpool2d_argmax_readback_does_not_segfault() {
    if !cuda_available() {
        eprintln!("skipping: no CUDA device");
        return;
    }

    // 0  1  2  3
    // 4  5  6  7
    // 8  9 10 11
    // 12 13 14 15
    let input_vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let input = gpu_tensor(&input_vals, &[1, 1, 4, 4]);

    // kernel 2x2, stride 2, padding 0 -> [1,1,2,2].
    let out = nsl_tensor_maxpool2d(input, 2, 2, 2, 0);

    let got = read_gpu(out, 4);
    // Per-window maxima: max(0,1,4,5)=5, max(2,3,6,7)=7,
    //                    max(8,9,12,13)=13, max(10,11,14,15)=15.
    let expected = [5.0f32, 7.0, 13.0, 15.0];
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-6,
            "maxpool2d output[{i}] = {g}, expected {e} (full output: {got:?})"
        );
    }

    nsl_tensor_free(input);
    nsl_tensor_free(out);
}
