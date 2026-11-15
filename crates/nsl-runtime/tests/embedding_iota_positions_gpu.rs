//! A gather by `arange(0, n)` from a GPU table reads device-resident positions.
//!
//! RotaryEmbedding gathers rows `0..seq_len` of its cached cos/sin tables with
//! `embedding_lookup(table, arange(0, seq_len))`, twice per attention layer and
//! again in the recompute. The positions are built on the host, and the GPU
//! arm used to upload them on every call: an 8 KB `cuMemcpyHtoD_v2` from
//! pageable memory, which waits for the device to drain first. At Coder-1B
//! that was 128 device synchronisations per micro-batch (measured with Nsight
//! Systems, `models/benchmarks/NSL_VS_PYTORCH_2026_11.md`).
//!
//! The values `0.0, 1.0, ..., n-1` never change, so the runtime now keeps one
//! device copy per device and uploads nothing on later calls. These gates check
//! that (1) the second gather of a prefix moves no bytes host-to-device, and
//! (2) every gather, iota or not, returns exactly the table rows a host-side
//! reference names.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test embedding_iota_positions_gpu
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::host_profile::pcie_bytes;
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::creation::nsl_tensor_arange;
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_embedding_lookup, nsl_tensor_free, nsl_tensor_len,
    nsl_tensor_ndim, nsl_tensor_shape_dim, nsl_tensor_zeros_on,
};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

fn cuda_available() -> bool {
    std::env::var("NSL_SKIP_CUDA_TESTS").is_err() && nsl_cuda_init() == 0
}

/// A `[rows, dim]` GPU table whose element (r, c) is `r * dim + c`: every value
/// is an exact small integer in f32, so a gathered row names its own index.
fn gpu_table(rows: i64, dim: i64) -> (i64, Vec<f32>) {
    let shape_list = nsl_list_new();
    nsl_list_push(shape_list, rows);
    nsl_list_push(shape_list, dim);
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    let host: Vec<f32> = (0..rows * dim).map(|i| i as f32).collect();
    nsl_test_cuda_h2d(nsl_tensor_data_ptr(t), host.as_ptr() as i64, (host.len() * 4) as i64);
    (t, host)
}

/// A host f32 index vector holding `vals`.
fn host_indices(vals: &[f32]) -> i64 {
    let t = nsl_tensor_arange(0.0, vals.len() as f64, 1.0);
    let data = nsl_tensor_data_ptr(t) as *mut f32;
    for (i, &v) in vals.iter().enumerate() {
        // SAFETY: `t` is a host f32 tensor of exactly `vals.len()` elements.
        unsafe { *data.add(i) = v };
    }
    t
}

fn d2h(t: i64) -> (Vec<i64>, Vec<f32>) {
    let shape = (0..nsl_tensor_ndim(t)).map(|d| nsl_tensor_shape_dim(t, d)).collect();
    let mut out = vec![0.0f32; nsl_tensor_len(t) as usize];
    nsl_test_cuda_d2h(out.as_mut_ptr() as i64, nsl_tensor_data_ptr(t), (out.len() * 4) as i64);
    (shape, out)
}

/// The rows of `table` (`dim` wide) named by `idx`, gathered on the host.
fn host_gather(table: &[f32], dim: usize, idx: &[f32]) -> Vec<f32> {
    idx.iter()
        .flat_map(|&i| table[i as usize * dim..(i as usize + 1) * dim].iter().copied())
        .collect()
}

/// The gate's point: after the first gather of a prefix, a repeat moves no
/// bytes host-to-device. Before the change every call uploaded `4 * n` bytes.
#[test]
#[ignore = "requires CUDA GPU"]
fn repeat_arange_gather_uploads_nothing() {
    if !cuda_available() {
        return;
    }
    let (rows, dim) = (64i64, 8usize);
    let (table, host) = gpu_table(rows, dim as i64);
    let n = 40usize;
    let want = host_gather(&host, dim, &(0..n).map(|i| i as f32).collect::<Vec<_>>());
    for call in 0..3 {
        let idx = nsl_tensor_arange(0.0, n as f64, 1.0);
        let (h2d_before, _) = pcie_bytes();
        let out = nsl_tensor_embedding_lookup(table, idx);
        let (h2d_after, _) = pcie_bytes();
        if call > 0 {
            assert_eq!(
                h2d_after - h2d_before,
                0,
                "call {call}: a gather by arange(0, {n}) uploaded its positions again"
            );
        }
        let (shape, got) = d2h(out);
        assert_eq!(shape, [n as i64, dim as i64]);
        assert_eq!(got, want, "call {call}: rows 0..{n} of the table");
        nsl_tensor_free(out);
        nsl_tensor_free(idx);
    }
    nsl_tensor_free(table);
}

/// A longer prefix than any seen before grows the device copy and is still
/// exact, at every length up to the table's, including the full table.
#[test]
#[ignore = "requires CUDA GPU"]
fn longer_prefixes_grow_the_device_positions() {
    if !cuda_available() {
        return;
    }
    let (rows, dim) = (4096i64, 4usize);
    let (table, host) = gpu_table(rows, dim as i64);
    for n in [3usize, 1000, 1025, 2048, 4096, 7] {
        let idx = nsl_tensor_arange(0.0, n as f64, 1.0);
        let out = nsl_tensor_embedding_lookup(table, idx);
        let (shape, got) = d2h(out);
        let want = host_gather(&host, dim, &(0..n).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(shape, [n as i64, dim as i64]);
        assert_eq!(got, want, "arange(0, {n})");
        nsl_tensor_free(out);
        nsl_tensor_free(idx);
    }
    nsl_tensor_free(table);
}

/// Index vectors that are NOT `0..n` (permuted, offset, or holding a
/// non-integer) take the upload path and gather exactly what they name. The
/// GPU kernel truncates an f32 index toward zero, so `2.5` reads row 2.
#[test]
#[ignore = "requires CUDA GPU"]
fn other_index_vectors_gather_what_they_name() {
    if !cuda_available() {
        return;
    }
    let (rows, dim) = (16i64, 3usize);
    let (table, host) = gpu_table(rows, dim as i64);
    let cases: [&[f32]; 4] = [
        &[0.0, 1.0, 3.0, 2.0],
        &[1.0, 2.0, 3.0],
        &[0.0, 1.0, 2.5, 3.0],
        &[15.0, 0.0, 15.0],
    ];
    for vals in cases {
        let idx = host_indices(vals);
        let (h2d_before, _) = pcie_bytes();
        let out = nsl_tensor_embedding_lookup(table, idx);
        let (h2d_after, _) = pcie_bytes();
        assert!(h2d_after > h2d_before, "{vals:?} is not 0..n: its values must be uploaded");
        let (_, got) = d2h(out);
        assert_eq!(got, host_gather(&host, dim, vals), "indices {vals:?}");
        nsl_tensor_free(out);
        nsl_tensor_free(idx);
    }
    nsl_tensor_free(table);
}
