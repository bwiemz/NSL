//! A CPU -> GPU -> CPU round trip must hand back the dtype tag it started
//! with and the exact bytes (C5 step 2a,
//! `docs/superpowers/specs/2026-09-26-dtype-semantics-design.md`).
//!
//! Before step 2a the download arm of `nsl_tensor_to_device` widened every f32
//! device tensor to f64, so an f32 tensor that made the trip came back as f64
//! and every later op on it took the f64 path. The `l1_backward` regression in
//! the CHANGELOG came from exactly that. Every other dtype was already a byte
//! copy in both directions; f32 was the one exception.
//!
//! The gates compare BIT PATTERNS, not values, and the f32 inputs are chosen
//! so that any conversion on the way shows up: -0.0, a quiet NaN with a
//! payload, a signalling NaN, both infinities, the smallest and largest
//! subnormals, and significands that use all 23 bits. A widen-then-narrow
//! through f64 is exact for every finite f32, so only the NaNs would catch
//! that; the tag assertion catches the plain widen.
//!
//! Not covered here: `DTYPE_U16_TOKEN`, which the upload deliberately widens to
//! i32 for the index kernels, and the FP8 tags, which `dtype_element_size`
//! does not know and so cannot be transferred in either direction.
//!
//! Running locally:
//!
//! ```bash
//! cargo test --package nsl-runtime --features cuda \
//!     --test transfer_preserves_dtype_gpu -- --ignored --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_abi::wire::dtype::{
    DTYPE_BF16, DTYPE_F32, DTYPE_F64, DTYPE_FP16, DTYPE_I32, DTYPE_INT8, DTYPE_U16_SEGMENT,
};
use nsl_runtime::nsl_cuda_init;

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
}

/// Minimal mirror of the runtime's `NslTensor` header, the same one
/// `gpu_dtype_refusal.rs` carries. Field order and widths follow the
/// `#[repr(C)]` layout in `crates/nsl-runtime/src/tensor/mod.rs`.
#[repr(C)]
struct TensorView {
    _magic: u32,
    data: *mut std::ffi::c_void,
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

fn view(ptr: i64) -> &'static TensorView {
    unsafe { &*(ptr as *const TensorView) }
}

/// A gate that skipped when CUDA is missing would pass vacuously in the cert
/// lane, so a failed init is a failure here, not a skip.
fn init_cuda() {
    let rc = nsl_cuda_init();
    assert_eq!(rc, 0, "nsl_cuda_init returned {rc}; these gates need a CUDA device");
}

/// A CPU tensor over a leaked copy of `bytes`, tagged `dtype`. The tensor does
/// not own the buffer (`owns_data = 0`), so the leak outlives it by design.
fn cpu_tensor(bytes: &[u8], dims: &[i64], dtype: u16) -> i64 {
    let data: &'static mut [u8] = Box::leak(bytes.to_vec().into_boxed_slice());
    let list = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(list, d) };
    }
    let t = unsafe { nsl_tensor_from_static(data.as_mut_ptr() as i64, list, i64::from(dtype)) };
    unsafe { nsl_list_free(list) };
    t
}

/// The host bytes of a CPU tensor: `len` elements of `elem_size` bytes.
fn host_bytes(ptr: i64, elem_size: usize) -> Vec<u8> {
    let v = view(ptr);
    assert_eq!(v.device, 0, "host_bytes needs a CPU tensor");
    let n = usize::try_from(v.len).expect("negative len") * elem_size;
    unsafe { std::slice::from_raw_parts(v.data as *const u8, n) }.to_vec()
}

/// Upload `bytes` tagged `dtype`, download it again, and require the same tag,
/// shape and bytes back. Returns nothing; every check is an assert.
fn assert_round_trip(label: &str, bytes: &[u8], dims: &[i64], dtype: u16, elem_size: usize) {
    let numel: i64 = dims.iter().product();
    assert_eq!(bytes.len(), usize::try_from(numel).unwrap() * elem_size, "{label}: fixture size");

    let cpu = cpu_tensor(bytes, dims, dtype);
    let gpu = unsafe { nsl_tensor_to_device(cpu, 1) };
    assert_eq!(view(gpu).device, 1, "{label}: upload did not land on the device");
    assert_eq!(view(gpu).dtype, dtype, "{label}: upload changed the tag");

    let back = unsafe { nsl_tensor_to_device(gpu, 0) };
    let b = view(back);
    assert_eq!(b.device, 0, "{label}: download did not land on the host");
    assert_eq!(
        b.dtype, dtype,
        "{label}: download changed the tag (tag {dtype} went up, tag {} came back)",
        b.dtype
    );
    assert_eq!(b.ndim, dims.len() as i64, "{label}: rank changed");
    assert_eq!(b.len, numel, "{label}: element count changed");

    let got = host_bytes(back, elem_size);
    if let Some(i) = got.iter().zip(bytes).position(|(g, w)| g != w) {
        let e = i / elem_size;
        panic!(
            "{label}: byte {i} (element {e}) differs after the round trip: sent {:02x?}, got {:02x?}",
            &bytes[e * elem_size..(e + 1) * elem_size],
            &got[e * elem_size..(e + 1) * elem_size],
        );
    }

    unsafe {
        nsl_tensor_free(back);
        nsl_tensor_free(gpu);
        nsl_tensor_free(cpu);
    }
}

/// f32 values that expose a conversion, as bit patterns.
const F32_EDGE_BITS: [u32; 17] = [
    0x0000_0000, // +0.0
    0x8000_0000, // -0.0
    0x3F80_0000, // 1.0
    0x3F80_0001, // 1.0 + 2^-23: the last significand bit
    0x3F7F_FFFF, // largest value below 1.0: every significand bit set
    0x3EAA_AAAB, // 1/3 rounded to nearest
    0x0000_0001, // smallest subnormal
    0x807F_FFFF, // largest negative subnormal
    0x0080_0000, // f32::MIN_POSITIVE
    0x7F7F_FFFF, // f32::MAX
    0xFF7F_FFFF, // f32::MIN
    0x7F80_0000, // +inf
    0xFF80_0000, // -inf
    0x7FC0_1234, // quiet NaN with a payload
    0xFFC0_0001, // negative quiet NaN with a payload
    0x7F80_0001, // signalling NaN: quieted by any arithmetic on the way
    0x4049_0FDB, // pi
];

/// A deterministic bit-pattern stream (SplitMix64), so the large case covers
/// every f32 class, NaNs and subnormals included, without a fixture file.
fn splitmix_bytes(n_bytes: usize, mut seed: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(n_bytes + 8);
    while out.len() < n_bytes {
        seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        out.extend_from_slice(&z.to_le_bytes());
    }
    out.truncate(n_bytes);
    out
}

#[test]
#[ignore = "requires CUDA GPU"]
fn f32_round_trip_keeps_the_tag_and_every_bit() {
    init_cuda();

    let edge: Vec<u8> = F32_EDGE_BITS.iter().flat_map(|b| b.to_le_bytes()).collect();
    assert_round_trip("f32 edge values", &edge, &[F32_EDGE_BITS.len() as i64], DTYPE_F32, 4);

    // Not a multiple of any block or warp size, and rank 2, so a transfer that
    // assumed a padded or rank-1 layout would show.
    let dims = [1027_i64, 131];
    let n = usize::try_from(dims[0] * dims[1]).unwrap();
    let random = splitmix_bytes(n * 4, 0x00C5_2A00);
    assert_round_trip("f32 random bit patterns", &random, &dims, DTYPE_F32, 4);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn every_byte_copied_dtype_round_trips_unchanged() {
    init_cuda();

    // f32 is gated above. These were byte copies before step 2a too; pinning
    // them keeps the download arm from growing a new special case unnoticed.
    let cases: [(&str, u16, usize); 5] = [
        ("fp16", DTYPE_FP16, 2),
        ("bf16", DTYPE_BF16, 2),
        ("u16 segment ids", DTYPE_U16_SEGMENT, 2),
        ("int8", DTYPE_INT8, 1),
        ("i32", DTYPE_I32, 4),
    ];
    for (i, (label, dtype, elem)) in cases.into_iter().enumerate() {
        let dims = [37_i64, 29];
        let n = usize::try_from(dims[0] * dims[1]).unwrap();
        let bytes = splitmix_bytes(n * elem, 0x00C5_2A10 + i as u64);
        assert_round_trip(label, &bytes, &dims, dtype, elem);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_transposed_f32_device_view_downloads_as_contiguous_f32() {
    init_cuda();

    // A non-contiguous device tensor is made contiguous on the device before
    // the copy, then the temporary is freed. That is a second route into the
    // download arm, and the one training's transposed weights take.
    let (rows, cols) = (48_usize, 80_usize);
    let bits: Vec<u32> = splitmix_bytes(rows * cols * 4, 0x00C5_2A20)
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let bytes: Vec<u8> = bits.iter().flat_map(|b| b.to_le_bytes()).collect();

    let cpu = cpu_tensor(&bytes, &[rows as i64, cols as i64], DTYPE_F32);
    let gpu = unsafe { nsl_tensor_to_device(cpu, 1) };
    let gpu_t = unsafe { nsl_tensor_transpose(gpu, 0, 1) };
    assert_eq!(view(gpu_t).device, 1);

    let back = unsafe { nsl_tensor_to_device(gpu_t, 0) };
    let b = view(back);
    assert_eq!((b.device, b.dtype), (0, DTYPE_F32), "transposed download: device/tag");
    assert_eq!(b.len, (rows * cols) as i64);

    let got = host_bytes(back, 4);
    for c in 0..cols {
        for r in 0..rows {
            let want = bits[r * cols + c];
            let o = (c * rows + r) * 4;
            let have = u32::from_le_bytes([got[o], got[o + 1], got[o + 2], got[o + 3]]);
            assert_eq!(
                have, want,
                "transposed download: element ({c},{r}) is {have:#010x}, want {want:#010x}"
            );
        }
    }

    unsafe {
        nsl_tensor_free(back);
        nsl_tensor_free(gpu_t);
        nsl_tensor_free(gpu);
        nsl_tensor_free(cpu);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn an_f64_upload_narrows_and_the_download_keeps_the_narrowed_tag() {
    init_cuda();

    // The interim state between steps 2a and 2b: the upload still narrows f64
    // to f32 (step 2b turns it into a refusal naming `.to(f32)`), and the
    // download no longer widens it back. So the host copy is f32, holding the
    // round-to-nearest narrowing of each input.
    let values: [f64; 6] = [1.0, -0.0, 0.1, 1.0 / 3.0, 1e-40, f64::INFINITY];
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

    let cpu = cpu_tensor(&bytes, &[values.len() as i64], DTYPE_F64);
    let gpu = unsafe { nsl_tensor_to_device(cpu, 1) };
    assert_eq!(view(gpu).dtype, DTYPE_F32, "f64 upload is expected to narrow until step 2b");

    let back = unsafe { nsl_tensor_to_device(gpu, 0) };
    assert_eq!(
        (view(back).device, view(back).dtype),
        (0, DTYPE_F32),
        "the download must not widen the narrowed tensor back to f64"
    );
    let got = host_bytes(back, 4);
    for (i, v) in values.iter().enumerate() {
        let want = (*v as f32).to_bits();
        let have = u32::from_le_bytes([got[4 * i], got[4 * i + 1], got[4 * i + 2], got[4 * i + 3]]);
        assert_eq!(have, want, "element {i} ({v:e}): got {have:#010x}, want {want:#010x}");
    }

    unsafe {
        nsl_tensor_free(back);
        nsl_tensor_free(gpu);
        nsl_tensor_free(cpu);
    }
}
