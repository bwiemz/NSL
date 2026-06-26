//! CPDT precision-adaptive optimizer state casts (v1: FP16 <-> FP32).
//!
//! Tensor-level (whole-tensor) dequant/quant for FASE's optimizer-state
//! storage. Reuses the crate's `f16_bits_to_f32` (exact widening) and
//! `f32_to_f16_bits` so CPU and GPU paths share identical numerics. NOTE:
//! despite its docstring, `f32_to_f16_bits` *truncates* the mantissa (it is
//! not round-to-nearest-even); v1 reuses it as-is. Truncation is biased
//! toward zero — proper rounding for optimizer state is a deferred ladder
//! step (see the design doc §10). Models its allocation on
//! `crate::fp8::nsl_fp8_cast`.
//!
//! # Adversarial review v6 — remaining deferred items
//!
//! Finding 3 (MEDIUM, HBM growth in long-scope loops): each forward+
//! backward step allocates one set of cast shadow buffers per
//! FusedLinearCe op via `cast_and_publish`'s `NslTensor::publish`,
//! which `scope_track`s the tensor; the scope sweep runs at function
//! exit. A training loop body lowered into one Cranelift function
//! that runs N iterations within ONE scope (e.g., gradient
//! accumulation) accumulates the cast tensors over the entire loop
//! rather than per iteration.  The Finding 10/14 forward-to-backward
//! cache halves the per-iteration cost (one cast set per dispatch
//! instead of two), but the per-loop accumulation hazard remains.
//! Mitigations: (a) ensure the lowered loop body emits per-iteration
//! `nsl_tensor_scope_begin`/`scope_end` boundaries; (b) longer term,
//! a v7 on-device cast kernel + a model-weight-pinned W shadow buffer
//! lifts the cost out of the loop entirely.
//!
//! Finding 7 (HIGH, RTE rounding gap): the underlying CPU primitives
//! `f32_to_f16_bits` / `f32_to_bf16_bits` TRUNCATE rather than round-to-
//! nearest-even, producing a one-sided bias of up to one full unit
//! roundoff per element.  CFTP v7 closes the GAP on the production GPU
//! path by routing GPU tensors through `gpu_cast_and_publish` →
//! PTX `cvt.rn.{f16,bf16}.f32` (IEEE-754 default RTE), so the
//! production wengert lowering matches the `half::*::from_f32`
//! rounding used by the V=49152 baselines.  The CPU primitives'
//! truncation lives only on CPU staging paths that real training runs
//! do not exercise; aligning the CPU primitives to RTE is the §10
//! deferred ladder step.  With v7 the FFI refusal at
//! `fused_linear_ce.rs` is LIFTED — `NSL_FUSED_LCE_REFUSE_NON_F32=1`
//! remains as an opt-in diagnostic / safety override.

use std::ffi::c_void;

use crate::memory::{checked_alloc, checked_alloc_zeroed};

use super::{
    bf16_bits_to_f32, f16_bits_to_f32, f32_to_bf16_bits, f32_to_f16_bits, NslTensor, DTYPE_BF16,
    DTYPE_F32, DTYPE_FP16,
};

/// Cast a tensor to `target_dtype`, returning a NEW owned tensor.
///
/// v1 supports F32 and FP16 storage. FP16 storage holds IEEE-754 half bits
/// as `u16`; the f32->f16 conversion uses the crate's `f32_to_f16_bits`
/// (which truncates the mantissa — see the module note). An F32->F32 cast
/// yields a faithful copy. The source tensor is NOT consumed.
#[no_mangle]
pub extern "C" fn nsl_tensor_cast(src_ptr: i64, target_dtype: i64) -> i64 {
    let t = unsafe { &*(src_ptr as *const NslTensor) };
    // v1 is CPU-only: `t.data` is dereferenced as a host pointer below. A GPU
    // tensor (device > 0) would be silent UB. Fail loudly instead. GPU support
    // (host round-trip / on-device conversion) is the Task 15 / v2 boundary.
    assert_eq!(
        t.device, 0,
        "nsl_tensor_cast: GPU tensors not supported in v1 (device={})",
        t.device
    );
    // CFTP v6 Finding 2 (HIGH): the cast reads `t.data` LINEARLY for `len`
    // contiguous elements then overwrites the output strides with
    // `compute_strides(shape)` (always row-major contiguous). A non-contiguous
    // source (transposed view, sliced view, broadcast) would silently produce a
    // tensor whose values are wrong — the cast reads physical layout but claims
    // logical contiguous semantics. Refuse non-contiguous sources loudly so a
    // future caller that legitimately passes a strided view surfaces here
    // rather than silently corrupting training numerics.
    assert!(
        t.is_contiguous(),
        "nsl_tensor_cast: source tensor must be contiguous (ndim={}); \
         non-contiguous sources are NOT supported — call \
         nsl_tensor_contiguous(src) before nsl_tensor_cast.",
        t.ndim
    );
    // CFTP v6 Finding 4 (MEDIUM): len/shape mismatch would size the alloc
    // by `t.len` while downstream callers (PTX kernels) trust `product(shape)`
    // — refuse such tensors at the cast site to catch the violation early.
    assert_eq!(
        t.len,
        NslTensor::total_elements(t.shape, t.ndim),
        "nsl_tensor_cast: t.len ({}) != product(t.shape) — invariant violation",
        t.len
    );
    let len = t.len as usize;
    let target = target_dtype as u16;

    let src_f32: Vec<f32> = match t.dtype {
        DTYPE_F32 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
            d.to_vec()
        }
        DTYPE_FP16 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const u16, len) };
            d.iter().map(|&b| f16_bits_to_f32(b)).collect()
        }
        DTYPE_BF16 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const u16, len) };
            d.iter().map(|&b| bf16_bits_to_f32(b)).collect()
        }
        other => panic!("nsl_tensor_cast: unsupported source dtype {other} (v6: F32/FP16/BF16)"),
    };

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let data: *mut c_void = match target {
        DTYPE_F32 => {
            let p = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = v };
            }
            p as *mut c_void
        }
        DTYPE_FP16 => {
            let p = checked_alloc(len * std::mem::size_of::<u16>()) as *mut u16;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = f32_to_f16_bits(v) };
            }
            p as *mut c_void
        }
        DTYPE_BF16 => {
            let p = checked_alloc(len * std::mem::size_of::<u16>()) as *mut u16;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = f32_to_bf16_bits(v) };
            }
            p as *mut c_void
        }
        other => panic!("nsl_tensor_cast: unsupported target dtype {other} (v6: F32/FP16/BF16)"),
    };

    let out = Box::new(NslTensor::new(
        data, shape, strides, t.ndim, t.len, t.device, target, 1, 0,
    ));
    // Bare `Box::into_raw` — NOT `NslTensor::publish`. The caller owns the
    // result and frees it explicitly (the FASE optimizer wrapping frees the
    // dequant/quant working tensors each step). `publish` would `scope_track`
    // the pointer; combined with the caller's explicit `nsl_tensor_free`, the
    // scope-end sweep would touch an already-freed tensor (use-after-free).
    Box::into_raw(out) as i64
}

/// CFTP v6 convenience: cast a tensor to BF16, returning a NEW owned tensor
/// published into the scope sweep (so the shadow buffer is reclaimed at
/// function-scope exit — the right lifecycle for forward inline-cast). The
/// source tensor is NOT consumed. v6 is CPU-only (the underlying
/// `nsl_tensor_cast` is CPU-only); a device-side PTX cast kernel is deferred
/// to v7. Supports F32/FP16/BF16 source dtypes (BF16->BF16 is a faithful copy).
#[no_mangle]
pub extern "C" fn nsl_tensor_to_bf16(src_ptr: i64) -> i64 {
    cast_and_publish(src_ptr, DTYPE_BF16)
}

/// CFTP v6 convenience: cast a tensor to FP16, returning a NEW owned tensor
/// published into the scope sweep. See `nsl_tensor_to_bf16`. Uses the crate's
/// `f32_to_f16_bits` primitive (truncating, not RTE — see module note).
#[no_mangle]
pub extern "C" fn nsl_tensor_to_fp16(src_ptr: i64) -> i64 {
    cast_and_publish(src_ptr, DTYPE_FP16)
}

/// CFTP v6 convenience: cast a tensor to F32, returning a NEW owned tensor
/// published into the scope sweep. Needed for backward symmetry: if dx is
/// emitted as BF16/FP16, the wengert tape may need an F32 widened copy.
#[no_mangle]
pub extern "C" fn nsl_tensor_to_f32(src_ptr: i64) -> i64 {
    cast_and_publish(src_ptr, DTYPE_F32)
}

/// CFTP v7: GPU dispatch for the v6 cast wrappers.
///
/// Allocates a device-side destination buffer, launches the appropriate PTX
/// cast kernel, and publishes the result so the scope sweep reclaims it.
/// `t` MUST already be device-resident and contiguous (the caller guards
/// both before reaching this function).
///
/// Same-dtype "casts" (e.g. F32->F32) are handled via `cuMemcpyDtoD` rather
/// than a kernel — there is no PTX cast pair for them. bf16<->fp16 are NOT
/// supported by a single kernel; the caller would have to stage via f32.
/// For the v7 surface (which is only reachable from the three v6 wrappers)
/// every legitimate (src,target) pair is covered.
#[cfg(feature = "cuda")]
fn gpu_cast_and_publish(t: &NslTensor, target_dtype: u16) -> i64 {
    use crate::cuda::precision_cast_kernels;

    let len = t.len as usize;
    // CFTP v7 follow-on (finding-4): zero-length casts must not produce a
    // published tensor pointing at stale HBM. They're a programming error
    // in this path (the wengert lowering never emits a zero-length cast),
    // so refuse loudly rather than silently exposing caching-allocator
    // garbage in the dst buffer.
    assert!(
        len > 0,
        "gpu_cast_and_publish: zero-length cast (t.len == 0) is not supported; \
         caller must not request a cast of an empty tensor"
    );
    let target_elem_size: usize = match target_dtype {
        DTYPE_F32 => 4,
        DTYPE_FP16 => 2,
        DTYPE_BF16 => 2,
        other => panic!(
            "gpu_cast_and_publish: unsupported target dtype {other} (v7: F32/FP16/BF16)"
        ),
    };

    // Allocate the device-side destination buffer via the runtime's managed
    // allocator (same path as every other GPU op; participates in the
    // caching allocator + OOM recovery).
    let dst_bytes = len * target_elem_size;
    let dst_data = crate::cuda::inner::alloc_managed(dst_bytes);
    // CFTP v7 follow-on (finding-4): defensive zero-fill closes the
    // partial-write window if a future change to the cast kernel ever
    // leaves bytes unwritten (e.g. if the u64 bounds check above were
    // accidentally re-narrowed to u32 by a refactor). Cheap — memset is
    // parallel-async on every Blackwell/Hopper SKU we target.
    crate::cuda::inner::memset_d8(dst_data, dst_bytes);

    // Same-dtype path: device-to-device memcpy (no conversion).
    if t.dtype == target_dtype {
        crate::cuda::inner::memcpy_dtod(
            dst_data,
            t.data as *const std::ffi::c_void,
            dst_bytes,
        );
    } else {
        let (ptx, kname) =
            precision_cast_kernels::pick_cast_kernel(t.dtype, target_dtype).unwrap_or_else(|| {
                panic!(
                    "gpu_cast_and_publish: no GPU cast kernel for src dtype {} -> target {}; \
                     v7 supports {{f32,fp16,bf16}} pairs through f32 only",
                    t.dtype, target_dtype
                )
            });
        let rc = precision_cast_kernels::launch_cast(
            ptx,
            kname,
            t.data as u64,
            dst_data as u64,
            len as u64,
        );
        assert_eq!(
            rc, 0,
            "gpu_cast_and_publish: cast kernel '{}' launch failed (rc={rc})",
            kname.trim_end_matches('\0')
        );
        // Synchronise so callers reading the destination see consistent
        // results (mirrors `nsl_fused_linear_ce_forward`'s post-launch sync).
        unsafe {
            cudarc::driver::sys::cuCtxSynchronize();
        }
    }

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);
    let out = Box::new(NslTensor::new(
        dst_data,
        shape,
        strides,
        t.ndim,
        t.len,
        t.device, // preserve device (1+ for CUDA)
        target_dtype,
        1, // owns_data
        0, // data_owner
    ));
    NslTensor::publish(out)
}

/// `cuda` feature off: GPU dispatch is unreachable (no device tensors can
/// exist without the `cuda` feature). Provide a stub for the non-cuda build
/// so the device-branch in `cast_and_publish` still compiles.
#[cfg(not(feature = "cuda"))]
fn gpu_cast_and_publish(t: &NslTensor, _target_dtype: u16) -> i64 {
    panic!(
        "gpu_cast_and_publish: device tensor (device={}) reached cast path \
         but runtime was compiled without the `cuda` feature",
        t.device
    );
}

/// Shared helper for v6 cast wrappers. Performs a CPU cast (via the same
/// host arithmetic as `nsl_tensor_cast`), but allocates the result via
/// `NslTensor::publish` so the new tensor participates in the scope sweep
/// (in contrast to `nsl_tensor_cast` which uses bare `Box::into_raw` for
/// FASE optimizer-state buffers that are explicitly freed each step).
///
/// CFTP v7 (2026-06-26): GPU tensors are now dispatched to
/// `gpu_cast_and_publish` (which lowers to a PTX cast kernel). CPU tensors
/// continue to take the host-staged path below — byte-identical to v6.
fn cast_and_publish(src_ptr: i64, target_dtype: u16) -> i64 {
    let t = unsafe { &*(src_ptr as *const NslTensor) };
    // CFTP v6 Finding 2 (HIGH) + Finding 4 (MEDIUM): the contiguity and
    // len/shape invariants are correctness preconditions for BOTH the CPU
    // host-staged path AND the v7 GPU PTX cast — keep them before any
    // device-branch so a violating input is refused regardless of device.
    // (Linear-read + row-major-strides assumption applies to GPU too: the
    // cast kernels index `i in [0, numel)` over the source buffer without
    // consulting strides.)
    assert!(
        t.is_contiguous(),
        "cast_and_publish: source tensor must be contiguous (ndim={}); \
         the wengert lowering must pass contiguous x/W/bias buffers.",
        t.ndim
    );
    assert_eq!(
        t.len,
        NslTensor::total_elements(t.shape, t.ndim),
        "cast_and_publish: t.len ({}) != product(t.shape) — invariant violation",
        t.len
    );

    // CFTP v7: GPU dispatch — must come BEFORE any host slice read on
    // `t.data` (lines below treat `t.data` as a HOST pointer; a GPU
    // tensor would be silent UB otherwise).
    if t.device > 0 {
        return gpu_cast_and_publish(t, target_dtype);
    }

    let len = t.len as usize;

    let src_f32: Vec<f32> = match t.dtype {
        DTYPE_F32 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
            d.to_vec()
        }
        DTYPE_FP16 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const u16, len) };
            d.iter().map(|&b| f16_bits_to_f32(b)).collect()
        }
        DTYPE_BF16 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const u16, len) };
            d.iter().map(|&b| bf16_bits_to_f32(b)).collect()
        }
        other => panic!("cast_and_publish: unsupported source dtype {other} (v6: F32/FP16/BF16)"),
    };

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let data: *mut c_void = match target_dtype {
        DTYPE_F32 => {
            let p = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = v };
            }
            p as *mut c_void
        }
        DTYPE_FP16 => {
            let p = checked_alloc(len * std::mem::size_of::<u16>()) as *mut u16;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = f32_to_f16_bits(v) };
            }
            p as *mut c_void
        }
        DTYPE_BF16 => {
            let p = checked_alloc(len * std::mem::size_of::<u16>()) as *mut u16;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = f32_to_bf16_bits(v) };
            }
            p as *mut c_void
        }
        other => panic!("cast_and_publish: unsupported target dtype {other} (v6: F32/FP16/BF16)"),
    };

    let out = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        t.ndim,
        t.len,
        t.device,
        target_dtype,
        1, // owns_data
        0, // data_owner
    ));
    // publish (not bare Box::into_raw) — scope_track participation so the
    // forward inline-cast shadow buffer is reclaimed on function-scope exit.
    NslTensor::publish(out)
}

/// Convert `src` (read as f32) into `dst`'s dtype and write it IN PLACE into
/// `dst`'s existing buffer. `dst` keeps its identity (pointer/buffer), which
/// preserves persistent optimizer-state identity across steps. `len` must
/// match. Neither tensor is freed. v1: F32/FP16 only, CPU only.
#[no_mangle]
pub extern "C" fn nsl_tensor_cast_into(dst_ptr: i64, src_ptr: i64) {
    let dst = unsafe { &*(dst_ptr as *const NslTensor) };
    let src = unsafe { &*(src_ptr as *const NslTensor) };
    assert_eq!(dst.device, 0, "nsl_tensor_cast_into: GPU dst not supported in v1 (device={})", dst.device);
    assert_eq!(src.device, 0, "nsl_tensor_cast_into: GPU src not supported in v1 (device={})", src.device);
    // CFTP v6 Finding 2: refuse non-contiguous src/dst — `nsl_tensor_cast_into`
    // does a linear read and a linear in-place write; a non-contiguous view
    // would silently corrupt the underlying buffer.
    assert!(
        src.is_contiguous(),
        "nsl_tensor_cast_into: src must be contiguous (ndim={})",
        src.ndim
    );
    assert!(
        dst.is_contiguous(),
        "nsl_tensor_cast_into: dst must be contiguous (ndim={})",
        dst.ndim
    );
    let len = dst.len as usize;
    assert_eq!(dst.len, src.len, "nsl_tensor_cast_into: length mismatch (dst={}, src={})", dst.len, src.len);

    let src_f32: Vec<f32> = match src.dtype {
        DTYPE_F32 => unsafe { std::slice::from_raw_parts(src.data as *const f32, len) }.to_vec(),
        DTYPE_FP16 => {
            let d = unsafe { std::slice::from_raw_parts(src.data as *const u16, len) };
            d.iter().map(|&b| f16_bits_to_f32(b)).collect()
        }
        DTYPE_BF16 => {
            let d = unsafe { std::slice::from_raw_parts(src.data as *const u16, len) };
            d.iter().map(|&b| bf16_bits_to_f32(b)).collect()
        }
        other => panic!("nsl_tensor_cast_into: unsupported src dtype {other} (v6: F32/FP16/BF16)"),
    };

    match dst.dtype {
        DTYPE_F32 => {
            let p = dst.data as *mut f32;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = v };
            }
        }
        DTYPE_FP16 => {
            let p = dst.data as *mut u16;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = f32_to_f16_bits(v) };
            }
        }
        DTYPE_BF16 => {
            let p = dst.data as *mut u16;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = f32_to_bf16_bits(v) };
            }
        }
        other => panic!("nsl_tensor_cast_into: unsupported dst dtype {other} (v6: F32/FP16/BF16)"),
    }
}

/// Allocate a new zero-filled tensor with the same shape/device as
/// `template_ptr` but the given `dtype` (v1: F32=1 or FP16=2). Mirrors
/// `nsl_tensor_zeros_like`'s allocation/ownership; the result is a persistent
/// optimizer-state buffer (same lifecycle as the FP32 zeros_like output).
///
/// Ownership pattern: `NslTensor::publish` — identical to the path
/// `nsl_tensor_zeros_like` → `nsl_tensor_zeros_on` → `nsl_tensor_zeros` →
/// `tensor_from_shape_list` (see `crates/nsl-runtime/src/tensor/creation.rs:46`).
/// `publish` calls `scope_track` so the tensor participates in the scope sweep
/// (freed on function-scope exit or by an explicit `nsl_tensor_free`). Using
/// bare `Box::into_raw` here instead would break parity: the scope sweep
/// would never reclaim an optimizer-state buffer allocated mid-training-loop.
///
/// Zeroing: uses `checked_alloc_zeroed` (OS-guaranteed zero fill via
/// `alloc_zeroed`). 0-bits == 0.0 for both f32 and IEEE-754 f16, so the
/// buffer is numerically zero for both dtypes without an explicit memset.
///
/// Device: mirrors `tensor_from_shape_list` — CPU only (device=0). GPU
/// template support requires a CUDA codepath (deferred to v2; only shape
/// and device are needed from the template, not its data, so a GPU template
/// whose shape is host-readable in principle could be supported, but parity
/// with the existing CPU-only creation helpers is the v1 constraint).
#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_like_dtype(template_ptr: i64, dtype: i64) -> i64 {
    let t = unsafe { &*(template_ptr as *const NslTensor) };
    let dtype = dtype as u16;
    let elem_size: usize = match dtype {
        DTYPE_F32 => 4,
        DTYPE_FP16 => 2,
        DTYPE_BF16 => 2,
        other => panic!("nsl_tensor_zeros_like_dtype: unsupported dtype {other} (v6: F32=1, FP16=2, BF16=3)"),
    };

    let ndim = t.ndim;
    let len = t.len;

    // Copy shape from template (same pattern as tensor_from_shape_list).
    let shape = NslTensor::copy_shape(t.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    // Zero-fill: 0-bits == 0.0 for both f32 and FP16.
    let data_size = (len as usize) * elem_size;
    let data = checked_alloc_zeroed(data_size) as *mut std::ffi::c_void;

    let tensor = Box::new(NslTensor::new(
        data, shape, strides, ndim, len,
        t.device, // preserve template's device
        dtype,
        1, // owns_data
        0, // data_owner
    ));
    // Use NslTensor::publish — NOT bare Box::into_raw — to mirror
    // nsl_tensor_zeros_like's lifecycle (scope_track participation).
    // Persistent optimizer-state buffers must have the same lifecycle as
    // the FP32 zeros_like output; publish ensures they do.
    NslTensor::publish(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an owned f32 CPU tensor from a slice (helper for tests).
    fn f32_tensor(vals: &[f32]) -> i64 {
        let len = vals.len();
        let data = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *data.add(i) = v };
        }
        let shape = NslTensor::copy_shape([len as i64].as_ptr() as *mut i64, 1);
        let strides = NslTensor::compute_strides(shape, 1);
        let t = Box::new(NslTensor::new(
            data as *mut c_void, shape, strides, 1, len as i64, 0, DTYPE_F32, 1, 0,
        ));
        Box::into_raw(t) as i64
    }

    fn fp16_tensor(vals: &[f32]) -> i64 {
        let len = vals.len();
        let data = checked_alloc(len * std::mem::size_of::<u16>()) as *mut u16;
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *data.add(i) = f32_to_f16_bits(v) };
        }
        let shape = NslTensor::copy_shape([len as i64].as_ptr() as *mut i64, 1);
        let strides = NslTensor::compute_strides(shape, 1);
        let t = Box::new(NslTensor::new(
            data as *mut c_void, shape, strides, 1, len as i64, 0, DTYPE_FP16, 1, 0,
        ));
        Box::into_raw(t) as i64
    }

    fn read_f32(ptr: i64) -> Vec<f32> {
        let t = unsafe { &*(ptr as *const NslTensor) };
        let len = t.len as usize;
        let d = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        d.to_vec()
    }

    #[test]
    fn cast_f32_to_fp16_then_back_matches_primitive_round_trip() {
        let vals = [1.0001_f32, 3.14159265_f32, -2.71828_f32, 65504.0, 1e-5];
        let src = f32_tensor(&vals);
        let fp16 = nsl_tensor_cast(src, DTYPE_FP16 as i64);
        let back = nsl_tensor_cast(fp16, DTYPE_F32 as i64);
        let got = read_f32(back);
        for (i, &v) in vals.iter().enumerate() {
            // Expected == exact round-trip through the crate's f32->f16 primitive
            // (which truncates) and back. Bit-exact: the cast must delegate to it.
            let expected = f16_bits_to_f32(f32_to_f16_bits(v));
            assert!(
                (got[i] - expected).abs() <= 0.0,
                "elem {i}: got {} expected {} (exact primitive round-trip)",
                got[i],
                expected
            );
        }
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(fp16);
        crate::tensor::nsl_tensor_free(back);
    }

    #[test]
    fn cast_f32_to_f32_is_value_identity() {
        let vals = [1.0_f32, -0.5, 12345.678];
        let src = f32_tensor(&vals);
        let out = nsl_tensor_cast(src, DTYPE_F32 as i64);
        let got = read_f32(out);
        assert_eq!(got, vals.to_vec(), "F32->F32 cast must preserve bits");
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(out);
    }

    /// Negative control: verify that the crate primitive `f32_to_f16_bits` produces
    /// bit-exact results consistent with the ACTUAL crate behavior (truncation of the
    /// lower 13 mantissa bits). Two distinct f32 values in the same f16 truncation
    /// bucket must produce identical f16 bits, confirming the cast path is faithful
    /// to the primitive rather than applying independent rounding.
    ///
    /// Note: `f32_to_f16_bits` is documented "round-to-nearest-even" but the
    /// implementation truncates (mant >> 13, no guard/sticky logic). This test
    /// documents that `nsl_tensor_cast` faithfully delegates to the primitive.
    #[test]
    fn cast_fp16_round_trip_is_faithful_to_primitive() {
        // Two f32 values that differ only in the lower 13 mantissa bits.
        // Both should produce the same f16 bits (truncation bucket).
        let v_lo = f32::from_bits(0x3F800001_u32); // 1.0 + 1 ULP (mant = 0x000001)
        let v_hi = f32::from_bits(0x3F801000_u32); // 1.0 + 0x1000 ULP (mant = 0x001000)
        // Both have upper 10 mant bits = 0 -> same f16 bucket
        assert_eq!(
            f32_to_f16_bits(v_lo),
            f32_to_f16_bits(v_hi),
            "both values must land in the same f16 truncation bucket"
        );
        // nsl_tensor_cast must match what the primitive produces
        let src_lo = {
            let data = checked_alloc(std::mem::size_of::<f32>()) as *mut f32;
            unsafe { *data = v_lo };
            let shape = NslTensor::copy_shape([1_i64].as_ptr() as *mut i64, 1);
            let strides = NslTensor::compute_strides(shape, 1);
            Box::into_raw(Box::new(NslTensor::new(
                data as *mut c_void, shape, strides, 1, 1, 0, DTYPE_F32, 1, 0,
            ))) as i64
        };
        let src_hi = {
            let data = checked_alloc(std::mem::size_of::<f32>()) as *mut f32;
            unsafe { *data = v_hi };
            let shape = NslTensor::copy_shape([1_i64].as_ptr() as *mut i64, 1);
            let strides = NslTensor::compute_strides(shape, 1);
            Box::into_raw(Box::new(NslTensor::new(
                data as *mut c_void, shape, strides, 1, 1, 0, DTYPE_F32, 1, 0,
            ))) as i64
        };
        let fp16_lo = nsl_tensor_cast(src_lo, DTYPE_FP16 as i64);
        let fp16_hi = nsl_tensor_cast(src_hi, DTYPE_FP16 as i64);
        let bits_lo = unsafe { *((&*(fp16_lo as *const NslTensor)).data as *const u16) };
        let bits_hi = unsafe { *((&*(fp16_hi as *const NslTensor)).data as *const u16) };
        assert_eq!(
            bits_lo, bits_hi,
            "cast must be faithful to f32_to_f16_bits: both values land in same f16 bucket"
        );
        crate::tensor::nsl_tensor_free(src_lo);
        crate::tensor::nsl_tensor_free(src_hi);
        crate::tensor::nsl_tensor_free(fp16_lo);
        crate::tensor::nsl_tensor_free(fp16_hi);
    }

    #[test]
    fn cast_into_quantizes_f32_into_existing_fp16_buffer() {
        let dst = fp16_tensor(&[0.0, 0.0, 0.0]);
        let dst_data_before = unsafe { (*(dst as *const NslTensor)).data };
        let src = f32_tensor(&[1.0001, 3.14159, -2.5]);

        nsl_tensor_cast_into(dst, src);

        let dst_data_after = unsafe { (*(dst as *const NslTensor)).data };
        assert_eq!(dst_data_before, dst_data_after, "dst buffer must not be reallocated");

        let t = unsafe { &*(dst as *const NslTensor) };
        let bits = unsafe { std::slice::from_raw_parts(t.data as *const u16, 3) };
        for (i, &v) in [1.0001_f32, 3.14159, -2.5].iter().enumerate() {
            assert_eq!(bits[i], f32_to_f16_bits(v), "elem {i} not quantized to match the primitive");
        }
        crate::tensor::nsl_tensor_free(dst);
        crate::tensor::nsl_tensor_free(src);
    }

    #[test]
    fn cast_into_f32_dst_is_copy() {
        let dst = f32_tensor(&[9.0, 9.0]);
        let src = f32_tensor(&[1.5, -2.5]);
        nsl_tensor_cast_into(dst, src);
        let got = read_f32(dst);
        assert_eq!(got, vec![1.5_f32, -2.5]);
        crate::tensor::nsl_tensor_free(dst);
        crate::tensor::nsl_tensor_free(src);
    }

    #[test]
    fn zeros_like_dtype_matches_shape_sets_dtype_and_zeroes() {
        let template = f32_tensor(&[5.0, 6.0, 7.0, 8.0]); // shape [4]
        let z = nsl_tensor_zeros_like_dtype(template, DTYPE_FP16 as i64);
        let t = unsafe { &*(z as *const NslTensor) };
        assert_eq!(t.len, 4, "len matches template");
        assert_eq!(t.dtype, DTYPE_FP16, "dtype is FP16");
        let bits = unsafe { std::slice::from_raw_parts(t.data as *const u16, 4) };
        assert!(bits.iter().all(|&b| f16_bits_to_f32(b) == 0.0), "all zero");
        crate::tensor::nsl_tensor_free(template);
        crate::tensor::nsl_tensor_free(z);
    }

    fn bf16_tensor(vals: &[f32]) -> i64 {
        let len = vals.len();
        let data = checked_alloc(len * std::mem::size_of::<u16>()) as *mut u16;
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *data.add(i) = f32_to_bf16_bits(v) };
        }
        let shape = NslTensor::copy_shape([len as i64].as_ptr() as *mut i64, 1);
        let strides = NslTensor::compute_strides(shape, 1);
        let t = Box::new(NslTensor::new(
            data as *mut c_void, shape, strides, 1, len as i64, 0, DTYPE_BF16, 1, 0,
        ));
        Box::into_raw(t) as i64
    }

    /// CFTP v6: bit-pattern equality vs hand-computed bf16 reference.
    /// bf16 = top 16 bits of f32, so 1.0 = 0x3F800000_f32 -> 0x3F80_bf16.
    #[test]
    fn to_bf16_produces_expected_bit_patterns() {
        // Hand-computed references: bf16 bits = high 16 bits of the f32 bits.
        let vals = [1.0_f32, -2.0, 0.5, 0.0];
        let expected_bits: [u16; 4] = [
            (1.0_f32.to_bits() >> 16) as u16,   // 0x3F80
            ((-2.0_f32).to_bits() >> 16) as u16, // 0xC000
            (0.5_f32.to_bits() >> 16) as u16,   // 0x3F00
            0x0000,
        ];
        let src = f32_tensor(&vals);
        let bf16 = nsl_tensor_to_bf16(src);
        let t = unsafe { &*(bf16 as *const NslTensor) };
        assert_eq!(t.dtype, DTYPE_BF16, "result must be tagged bf16");
        assert_eq!(t.len, vals.len() as i64);
        let bits = unsafe { std::slice::from_raw_parts(t.data as *const u16, vals.len()) };
        for i in 0..vals.len() {
            assert_eq!(
                bits[i], expected_bits[i],
                "elem {i}: got 0x{:04X} expected 0x{:04X}",
                bits[i], expected_bits[i]
            );
        }
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(bf16);
    }

    /// Round-trip f32 -> bf16 -> f32 closes within bf16's truncating eps.
    /// bf16 has 7 explicit mantissa bits so unit roundoff = 2^-7 ≈ 7.8e-3.
    /// The crate's `f32_to_bf16_bits` truncates (drops low 16 bits), so worst
    /// case relative error approaches the unit roundoff itself, not 0.5 ULP.
    /// We use 7.9e-3 (one ULP above 2^-7) to be safe.
    #[test]
    fn to_bf16_round_trip_within_epsilon() {
        let vals = [1.0001_f32, 3.14159265, -2.71828, 1234.5, 1e-3, -1e3];
        let src = f32_tensor(&vals);
        let bf16 = nsl_tensor_to_bf16(src);
        let back = nsl_tensor_to_f32(bf16);
        let got = read_f32(back);
        for (i, &v) in vals.iter().enumerate() {
            let rel = ((got[i] - v) / v).abs();
            assert!(
                rel <= 7.9e-3,
                "elem {i}: v={v} got={} rel_err={} (bf16 truncating eps ~7.8e-3)",
                got[i],
                rel
            );
        }
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(bf16);
        crate::tensor::nsl_tensor_free(back);
    }

    /// Round-trip f32 -> fp16 -> f32 closes within fp16's epsilon (~9.8e-4).
    /// fp16 has 10 mantissa bits, eps ~= 2^-10 ≈ 9.77e-4. Truncating primitive
    /// gives worst-case ~2 * eps but real-world values stay within ~1.1e-3.
    #[test]
    fn to_fp16_round_trip_within_epsilon() {
        // Avoid subnormal and overflow regions: fp16 range is ~6e-5 to 65504.
        let vals = [1.0001_f32, 3.14159265, -2.71828, 1234.5, 1.5e-2, -512.0];
        let src = f32_tensor(&vals);
        let fp16 = nsl_tensor_to_fp16(src);
        let back = nsl_tensor_to_f32(fp16);
        let got = read_f32(back);
        for (i, &v) in vals.iter().enumerate() {
            let rel = ((got[i] - v) / v).abs();
            // Truncating primitive can lose up to ~2 ULP at worst; use 1.1e-3.
            assert!(
                rel <= 1.1e-3,
                "elem {i}: v={v} got={} rel_err={} (fp16 eps ~9.8e-4)",
                got[i],
                rel
            );
        }
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(fp16);
        crate::tensor::nsl_tensor_free(back);
    }

    /// Multi-size coverage: small (1), aligned (16), non-aligned (17), large (1024).
    #[test]
    fn to_bf16_multi_size_round_trip() {
        for &n in &[1_usize, 16, 17, 1024] {
            let vals: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125 + 0.1).collect();
            let src = f32_tensor(&vals);
            let bf16 = nsl_tensor_to_bf16(src);
            let back = nsl_tensor_to_f32(bf16);
            let got = read_f32(back);
            for (i, &v) in vals.iter().enumerate() {
                let rel = ((got[i] - v) / v).abs();
                assert!(
                    rel <= 7.9e-3,
                    "n={n} elem {i}: v={v} got={} rel_err={}",
                    got[i],
                    rel
                );
            }
            // Cast also preserves shape + strides.
            let t = unsafe { &*(bf16 as *const NslTensor) };
            assert_eq!(t.len as usize, n, "len preserved (n={n})");
            assert_eq!(t.ndim, 1, "ndim preserved (n={n})");
            crate::tensor::nsl_tensor_free(src);
            crate::tensor::nsl_tensor_free(bf16);
            crate::tensor::nsl_tensor_free(back);
        }
    }

    /// Confirm the v6 wrappers DELEGATE to the same primitives as nsl_tensor_cast.
    /// `nsl_tensor_to_bf16(x)` and `nsl_tensor_cast(x, DTYPE_BF16)` must produce
    /// bit-identical buffers.
    #[test]
    fn to_bf16_matches_nsl_tensor_cast_bit_for_bit() {
        let vals = [1.0_f32, -3.14, 1e-2, 65500.0];
        let src_a = f32_tensor(&vals);
        let src_b = f32_tensor(&vals);
        let via_wrapper = nsl_tensor_to_bf16(src_a);
        let via_cast = nsl_tensor_cast(src_b, DTYPE_BF16 as i64);
        let bits_a = unsafe {
            std::slice::from_raw_parts(
                (*(via_wrapper as *const NslTensor)).data as *const u16,
                vals.len(),
            )
        };
        let bits_b = unsafe {
            std::slice::from_raw_parts(
                (*(via_cast as *const NslTensor)).data as *const u16,
                vals.len(),
            )
        };
        assert_eq!(bits_a, bits_b, "wrapper must be bit-identical to nsl_tensor_cast");
        crate::tensor::nsl_tensor_free(src_a);
        crate::tensor::nsl_tensor_free(src_b);
        // Note: via_wrapper was published into scope_track. In tests there is
        // no enclosing function scope sweep, so an explicit free is fine.
        crate::tensor::nsl_tensor_free(via_wrapper);
        crate::tensor::nsl_tensor_free(via_cast);
    }

    /// Source dtype coverage: bf16 source -> f32 cast (backward symmetry path).
    #[test]
    fn nsl_tensor_cast_supports_bf16_source() {
        let vals = [1.0_f32, -2.0, 0.5];
        let src = bf16_tensor(&vals);
        let out = nsl_tensor_cast(src, DTYPE_F32 as i64);
        let got = read_f32(out);
        // Exact: bf16 1.0/-2.0/0.5 round-trip with zero error (all exactly representable).
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(got[i], v, "elem {i}: exact bf16 value lost");
        }
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(out);
    }

    /// zeros_like_dtype now supports BF16.
    #[test]
    fn zeros_like_dtype_bf16() {
        let template = f32_tensor(&[5.0, 6.0, 7.0]);
        let z = nsl_tensor_zeros_like_dtype(template, DTYPE_BF16 as i64);
        let t = unsafe { &*(z as *const NslTensor) };
        assert_eq!(t.dtype, DTYPE_BF16, "dtype is BF16");
        assert_eq!(t.len, 3);
        let bits = unsafe { std::slice::from_raw_parts(t.data as *const u16, 3) };
        assert!(bits.iter().all(|&b| bf16_bits_to_f32(b) == 0.0), "all zero");
        crate::tensor::nsl_tensor_free(template);
        crate::tensor::nsl_tensor_free(z);
    }

    #[test]
    fn zeros_like_dtype_f32_is_f32_zeros() {
        let template = f32_tensor(&[5.0, 6.0]);
        let z = nsl_tensor_zeros_like_dtype(template, DTYPE_F32 as i64);
        let got = read_f32(z);
        assert_eq!(got, vec![0.0_f32, 0.0]);
        crate::tensor::nsl_tensor_free(template);
        crate::tensor::nsl_tensor_free(z);
    }

    /// CFTP v6 Finding 2 (HIGH): the contiguity guard precondition is a
    /// load-bearing invariant — verify both positive forms work today.  The
    /// negative (refusal) cases cannot use `#[should_panic]` because the
    /// cast FFIs are `extern "C"` (panic-on-unwind aborts the test runner);
    /// the assertion itself is verified at the function-prologue line.
    /// This test exercises the happy path (contiguous tensor) and confirms
    /// the new `is_contiguous` check does not over-trigger on row-major
    /// rank-2 inputs (a stride layout the prior tests did not cover).
    #[test]
    fn cast_accepts_rank2_row_major_contiguous() {
        // 2x2 row-major contiguous: shape=[2,2], strides=[2,1], len=4.
        let vals = [1.0_f32, 2.0, 3.0, 4.0];
        let len = vals.len();
        let data = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *data.add(i) = v };
        }
        let shape = NslTensor::copy_shape([2_i64, 2].as_ptr() as *mut i64, 2);
        let strides = NslTensor::compute_strides(shape, 2);
        let t = Box::new(NslTensor::new(
            data as *mut c_void, shape, strides, 2, len as i64, 0, DTYPE_F32, 1, 0,
        ));
        let src = Box::into_raw(t) as i64;
        // Should NOT panic — the contiguous, len==product(shape) input is valid.
        let bf16 = nsl_tensor_to_bf16(src);
        let t = unsafe { &*(bf16 as *const NslTensor) };
        assert_eq!(t.dtype, DTYPE_BF16);
        assert_eq!(t.len, 4);
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(bf16);
    }
}
