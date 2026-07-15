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

    // GPU dispatch (deferral-closure 2026-07-14, lifts the #367 refusal):
    // route through the CFTP-v7 PTX cast kernels, returning a BARE
    // `Box::into_raw` tensor exactly like the CPU path below — the FASE
    // optimizer envelope frees the dequant working tensor explicitly each
    // step, so `publish`'s scope tracking would double-free. Must run
    // BEFORE any host slice read on `t.data`.
    if t.device > 0 {
        return gpu_cast_bare(t, target);
    }

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
/// Device-side conversion core shared by every GPU cast entry: writes
/// `len` elements of `src_dtype` at `src_data` into `dst_data` as
/// `dst_dtype`. Same-dtype pairs use `cuMemcpyDtoD`; converting pairs
/// launch the CFTP-v7 PTX cast kernels. No sync — callers on the default
/// in-order stream are ordered against subsequent kernels; callers that
/// free `src_data` immediately after MUST sync first (see
/// `nsl_tensor_cast_into`'s GPU branch).
#[cfg(feature = "cuda")]
fn gpu_cast_core(src_data: u64, src_dtype: u16, dst_data: u64, dst_dtype: u16, len: usize) {
    use crate::cuda::precision_cast_kernels;
    if src_dtype == dst_dtype {
        let elem: usize = if src_dtype == DTYPE_F32 { 4 } else { 2 };
        crate::cuda::inner::memcpy_dtod(
            dst_data as *mut std::ffi::c_void,
            src_data as *const std::ffi::c_void,
            len * elem,
        );
        return;
    }
    let (ptx, kname) = precision_cast_kernels::pick_cast_kernel(src_dtype, dst_dtype)
        .unwrap_or_else(|| {
            panic!(
                "gpu_cast_core: no GPU cast kernel for src dtype {src_dtype} -> \
                 target {dst_dtype}; supported pairs go through f32 only"
            )
        });
    let rc = precision_cast_kernels::launch_cast(ptx, kname, src_data, dst_data, len as u64);
    assert_eq!(
        rc,
        0,
        "gpu_cast_core: cast kernel '{}' launch failed (rc={rc})",
        kname.trim_end_matches('\0')
    );
}

/// GPU analog of the CPU `nsl_tensor_cast` body: new device tensor via
/// bare `Box::into_raw` (caller frees explicitly — the FASE optimizer
/// envelope's dequant working tensors; `publish` would double-free at the
/// scope sweep). Contiguity/len invariants were already checked by the
/// caller.
#[cfg(feature = "cuda")]
fn gpu_cast_bare(t: &NslTensor, target_dtype: u16) -> i64 {
    let len = t.len as usize;
    assert!(
        len > 0,
        "gpu_cast_bare: zero-length cast is not supported (optimizer moments are never empty)"
    );
    let elem: usize = match target_dtype {
        DTYPE_F32 => 4,
        DTYPE_FP16 | DTYPE_BF16 => 2,
        other => panic!("gpu_cast_bare: unsupported target dtype {other} (F32/FP16/BF16)"),
    };
    let dst_bytes = len * elem;
    let dst_data = crate::cuda::inner::alloc_managed(dst_bytes);
    crate::cuda::inner::memset_d8(dst_data, dst_bytes);
    gpu_cast_core(t.data as u64, t.dtype, dst_data as u64, target_dtype, len);
    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);
    let out = Box::new(NslTensor::new(
        dst_data, shape, strides, t.ndim, t.len, t.device, target_dtype, 1, 0,
    ));
    Box::into_raw(out) as i64
}

/// `cuda` off: a device tensor cannot exist — loud stub (mirrors
/// `gpu_cast_and_publish`'s non-cuda stub).
#[cfg(not(feature = "cuda"))]
fn gpu_cast_bare(t: &NslTensor, _target_dtype: u16) -> i64 {
    panic!(
        "gpu_cast_bare: device tensor (device={}) reached cast path but the \
         runtime was compiled without the `cuda` feature",
        t.device
    );
}

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
        // CFTP v7 follow-on (findings 12/17): no per-cast cuCtxSynchronize.
        //
        // The production consumer (fused-LCE FFI) runs on CUDA's default
        // (in-order) stream and issues its own terminal cuCtxSynchronize
        // before returning. A pre-cast sync here adds 3-6 host stalls per
        // fused-LCE step (one per x/W/bias cast, doubled in forward+backward)
        // that the wengert precision-cast cache was specifically introduced
        // to amortise. Tests that read dst_data directly on the host (e.g.
        // `precision_cast_gpu_dispatch.rs`) route through
        // `nsl_tensor_to_device(_, 0)` which itself syncs before the dtoh
        // copy, so they still observe consistent results.
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
    // Mixed-device pairs have no meaning in the dequant→step→quant
    // envelope (both live where the parameters live) — refuse loudly.
    assert_eq!(
        dst.device, src.device,
        "nsl_tensor_cast_into: dst (device={}) and src (device={}) must be co-resident",
        dst.device, src.device
    );
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

    // GPU dispatch (deferral-closure 2026-07-14): quant-back writes into
    // `dst`'s EXISTING device buffer, preserving the persistent
    // optimizer-state identity across steps — no allocation. The terminal
    // sync makes the immediately-following `nsl_tensor_free(src)` in the
    // FASE envelope safe even if a future allocator change breaks the
    // in-order-stream reuse argument (once per param per optimizer step —
    // cold path relative to per-op kernels).
    if dst.device > 0 {
        #[cfg(feature = "cuda")]
        {
            gpu_cast_core(src.data as u64, src.dtype, dst.data as u64, dst.dtype, len);
            unsafe { crate::cuda::inner::cu_ctx_synchronize() };
            return;
        }
        #[cfg(not(feature = "cuda"))]
        panic!(
            "nsl_tensor_cast_into: device tensors (device={}) but the runtime \
             was compiled without the `cuda` feature",
            dst.device
        );
    }

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

/// Allocate a new zero-filled tensor with the same shape and device as
/// `template_ptr` but the given `dtype` (F32=1, FP16=2, BF16=3). Mirrors
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
/// Device: follows the template. A CPU template mirrors
/// `tensor_from_shape_list` (host `checked_alloc_zeroed`). A GPU template
/// allocates a zero-filled DEVICE buffer via the managed allocator
/// (deferral-closure 2026-07-14 — this lifted the #367
/// `refuse_gpu_moment_template` guard, which existed because the whole
/// dequant→step→quant envelope was CPU-only; `nsl_tensor_cast` /
/// `nsl_tensor_cast_into` now have GPU paths through the CFTP-v7 PTX cast
/// kernels, so reduced-precision moments genuinely live on-device).
/// History preserved for the audit trail: pre-#367 this function silently
/// allocated HOST memory but stamped the result `device = template.device`
/// — a device=1 tensor backed by a host pointer that aborted mid-step.
#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_like_dtype(template_ptr: i64, dtype: i64) -> i64 {
    let t = unsafe { &*(template_ptr as *const NslTensor) };
    let dtype = dtype as u16;
    if t.device > 0 {
        return gpu_zeros_like_dtype(t, dtype);
    }
    zeros_like_dtype_body(t, dtype)
}

/// GPU branch of [`nsl_tensor_zeros_like_dtype`]: zero-filled device
/// buffer, same `publish` lifecycle as the CPU body (persistent
/// optimizer-state buffers participate in the scope sweep either way).
#[cfg(feature = "cuda")]
fn gpu_zeros_like_dtype(t: &NslTensor, dtype: u16) -> i64 {
    let elem_size: usize = match dtype {
        DTYPE_F32 => 4,
        DTYPE_FP16 | DTYPE_BF16 => 2,
        other => panic!(
            "nsl_tensor_zeros_like_dtype: unsupported dtype {other} (F32=1, FP16=2, BF16=3)"
        ),
    };
    let data_size = (t.len as usize) * elem_size;
    let data = crate::cuda::inner::alloc_managed(data_size);
    crate::cuda::inner::memset_d8(data, data_size); // 0-bits == 0.0 for f32/f16/bf16
    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);
    let tensor = Box::new(NslTensor::new(
        data, shape, strides, t.ndim, t.len, t.device, dtype, 1, 0,
    ));
    NslTensor::publish(tensor)
}

/// `cuda` off: a device template cannot exist — loud stub.
#[cfg(not(feature = "cuda"))]
fn gpu_zeros_like_dtype(t: &NslTensor, _dtype: u16) -> i64 {
    panic!(
        "nsl_tensor_zeros_like_dtype: GPU template (device={}) but the \
         runtime was compiled without the `cuda` feature",
        t.device
    );
}

fn zeros_like_dtype_body(t: &NslTensor, dtype: u16) -> i64 {
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
        t.device, // always 0 here: GPU templates take gpu_zeros_like_dtype
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

// ---------------------------------------------------------------------------
// P0.3 — offload x reduced-precision composition: cross-device cast envelope.
//
// The single-purpose envelopes cannot compose: `nsl_tensor_cast_into`
// asserts co-residency (quant-back is an in-place same-device write) and
// `nsl_tensor_copy_data` asserts dtype equality (offload copy-back is a
// raw memcpy). These two FFIs are the composition: each stages through a
// transient DEVICE buffer at the host dtype, so the PCIe leg moves the
// 2-byte payload (half the f32 offload traffic) and the quant/dequant
// runs on-device through the CFTP-v7 PTX cast kernels.
// ---------------------------------------------------------------------------

/// Dequant-stage a HOST-resident optimizer-state tensor to a NEW
/// device-resident F32 working tensor (offload+precision stage-in).
///
/// * `host_src`: persistent host state buffer (F32/FP16/BF16, device 0).
///   NOT consumed.
/// * `device_template`: placement donor (theta) — only its device is used.
///
/// GPU template: HtoD the raw half payload into a transient device buffer,
/// dequant to F32 on-device, free the transient. F32 host state skips the
/// transient (plain HtoD into the result). CPU template (GPU-less run):
/// degrades to the CPU `nsl_tensor_cast(src, F32)` — placement-transparent
/// like the offload-only envelope.
///
/// Ownership: BARE `Box::into_raw`, exactly like `nsl_tensor_cast` — the
/// FASE envelope frees the working tensor each step (via the consuming
/// `nsl_tensor_cast_to_host_into`); `publish` would double-free at the
/// scope sweep.
#[no_mangle]
pub extern "C" fn nsl_tensor_cast_from_host(host_src_ptr: i64, device_template_ptr: i64) -> i64 {
    let src = unsafe { &*(host_src_ptr as *const NslTensor) };
    let template = unsafe { &*(device_template_ptr as *const NslTensor) };
    assert!(
        src.is_contiguous(),
        "nsl_tensor_cast_from_host: src must be contiguous (ndim={})",
        src.ndim
    );
    assert_eq!(
        src.len,
        NslTensor::total_elements(src.shape, src.ndim),
        "nsl_tensor_cast_from_host: src.len ({}) != product(src.shape) — invariant violation",
        src.len
    );
    assert_eq!(
        src.device, 0,
        "nsl_tensor_cast_from_host: src must be HOST-resident (device={})",
        src.device
    );
    if template.device == 0 {
        // CPU run: the whole envelope degrades to the co-resident CPU cast.
        return nsl_tensor_cast(host_src_ptr, DTYPE_F32 as i64);
    }
    assert_eq!(
        src.len, template.len,
        "nsl_tensor_cast_from_host: src len {} != template len {}",
        src.len, template.len
    );
    #[cfg(feature = "cuda")]
    {
        let len = src.len as usize;
        assert!(
            len > 0,
            "nsl_tensor_cast_from_host: zero-length cast is not supported \
             (optimizer moments are never empty)"
        );
        let dst_bytes = len * 4;
        let dst_data = crate::cuda::inner::alloc_managed(dst_bytes);
        match src.dtype {
            DTYPE_F32 => {
                // f32-tier param under the combined plan: no cast needed —
                // identical to the offload-only stage-in transfer.
                crate::cuda::inner::memcpy_htod(dst_data, src.data, dst_bytes);
            }
            DTYPE_FP16 | DTYPE_BF16 => {
                let half_bytes = len * 2;
                let transient = crate::cuda::inner::alloc_managed(half_bytes);
                crate::cuda::inner::memcpy_htod(transient, src.data, half_bytes);
                gpu_cast_core(transient as u64, src.dtype, dst_data as u64, DTYPE_F32, len);
                // Inline free is safe: the dequant kernel runs on the NULL
                // stream, and any reuse of the recycled block is written by
                // later NULL-stream-ordered work (same in-order argument as
                // gpu_cast_bare's caller-side frees).
                crate::cuda::inner::free_managed(transient);
            }
            other => panic!(
                "nsl_tensor_cast_from_host: unsupported host state dtype {other} \
                 (F32=1, FP16=2, BF16=3)"
            ),
        }
        let shape = NslTensor::copy_shape(src.shape, src.ndim);
        let strides = NslTensor::compute_strides(shape, src.ndim);
        let out = Box::new(NslTensor::new(
            dst_data, shape, strides, src.ndim, src.len, template.device, DTYPE_F32, 1, 0,
        ));
        return Box::into_raw(out) as i64;
    }
    #[cfg(not(feature = "cuda"))]
    panic!(
        "nsl_tensor_cast_from_host: device template (device={}) but the runtime \
         was compiled without the `cuda` feature",
        template.device
    );
}

/// Quant-store a device-resident F32 working tensor back into its
/// persistent HOST-resident reduced-precision buffer (offload+precision
/// stage-out). CONSUMES `device_src` — the staged working tensor is freed
/// inline on the sync path or drain-deferred (freed by
/// `nsl_offload_drain`) on the async path, mirroring
/// `nsl_tensor_copy_data_async`'s ownership contract.
///
/// GPU src: on-device quant-cast F32 -> host dtype into a transient device
/// buffer, then DtoH into `host_dst` — async on the transfer stream when
/// `host_dst` is pinned and `NSL_OFFLOAD_SYNC` is unset (transient +
/// staged tensor ride the drain list), synchronous otherwise. Same-dtype
/// pairs delegate to `nsl_tensor_copy_data_async` (no cast kernel).
/// CPU src (GPU-less run): degrades to the co-resident CPU
/// `nsl_tensor_cast_into` + inline free.
#[no_mangle]
pub extern "C" fn nsl_tensor_cast_to_host_into(host_dst_ptr: i64, device_src_ptr: i64) {
    let dst = unsafe { &*(host_dst_ptr as *const NslTensor) };
    let src = unsafe { &*(device_src_ptr as *const NslTensor) };
    assert!(
        dst.is_contiguous(),
        "nsl_tensor_cast_to_host_into: dst must be contiguous (ndim={})",
        dst.ndim
    );
    assert!(
        src.is_contiguous(),
        "nsl_tensor_cast_to_host_into: src must be contiguous (ndim={})",
        src.ndim
    );
    assert_eq!(
        dst.len, src.len,
        "nsl_tensor_cast_to_host_into: length mismatch (dst={}, src={})",
        dst.len, src.len
    );
    if src.device == 0 {
        // CPU run: co-resident quant-back, then the consuming free.
        nsl_tensor_cast_into(host_dst_ptr, device_src_ptr);
        crate::tensor::nsl_tensor_free(device_src_ptr);
        return;
    }
    assert_eq!(
        dst.device, 0,
        "nsl_tensor_cast_to_host_into: dst must be HOST-resident (device={})",
        dst.device
    );
    if dst.dtype == src.dtype {
        // f32-tier param under the combined plan: plain copy-back
        // (consuming, pinned-aware, async-capable).
        crate::tensor::nsl_tensor_copy_data_async(host_dst_ptr, device_src_ptr);
        return;
    }
    #[cfg(feature = "cuda")]
    {
        assert_eq!(
            src.dtype, DTYPE_F32,
            "nsl_tensor_cast_to_host_into: staged working tensor must be F32 \
             (got dtype {}) — the FASE envelope updates in F32",
            src.dtype
        );
        assert!(
            dst.dtype == DTYPE_FP16 || dst.dtype == DTYPE_BF16,
            "nsl_tensor_cast_to_host_into: unsupported host state dtype {} \
             (FP16=2, BF16=3; F32 pairs take the copy_data path above)",
            dst.dtype
        );
        let len = src.len as usize;
        assert!(
            len > 0,
            "nsl_tensor_cast_to_host_into: zero-length cast is not supported \
             (optimizer moments are never empty)"
        );
        let half_bytes = len * 2;
        // Bound in-flight staged buffers BEFORE allocating the transient
        // (review M1) — a saturated flush both caps device residency and
        // returns the oldest transients to the caching allocator so this
        // alloc can reuse them.
        crate::tensor::offload_flush_if_saturated();
        let transient = crate::cuda::inner::alloc_managed(half_bytes);
        // Quant-cast on the NULL stream (ordered after the update kernels
        // that produced src).
        gpu_cast_core(src.data as u64, DTYPE_F32, transient as u64, dst.dtype, len);
        if crate::cuda::inner::is_pinned(dst.data) && !crate::tensor::offload_sync_forced() {
            // Async DtoH on the transfer stream (event-ordered after the
            // cast kernel). Both device buffers stay alive until the
            // per-step nsl_offload_drain().
            crate::cuda::inner::memcpy_dtoh_async(dst.data, transient, half_bytes);
            crate::tensor::offload_defer_free_device_buf(transient);
            crate::tensor::offload_defer_free_tensor(device_src_ptr);
        } else {
            // Sync fallback (pageable host buffer or NSL_OFFLOAD_SYNC=1):
            // the synchronous DtoH serializes with the NULL stream, so the
            // cast kernel is complete when it returns — inline frees safe.
            crate::cuda::inner::memcpy_dtoh(dst.data, transient, half_bytes);
            crate::cuda::inner::free_managed(transient);
            crate::tensor::nsl_tensor_free(device_src_ptr);
        }
    }
    #[cfg(not(feature = "cuda"))]
    panic!(
        "nsl_tensor_cast_to_host_into: device src (device={}) but the runtime \
         was compiled without the `cuda` feature",
        src.device
    );
}

/// Allocate a zero-filled HOST-resident tensor with `template`'s shape at
/// the given `dtype` (F32=1, FP16=2, BF16=3) — the offload+precision
/// analog of `nsl_tensor_zeros_like_host_f32` (P0.3 item 8).
///
/// GPU template: host buffer via the pinned-aware offload allocator
/// (pinned unless NSL_OFFLOAD_PAGEABLE=1 or no CUDA context; element size
/// 2 for FP16/BF16). CPU template: delegates to
/// `nsl_tensor_zeros_like_dtype` so a GPU-less run with both flags behaves
/// exactly like precision-only (mirrors `nsl_tensor_zeros_like_host_f32`'s
/// CPU delegation rationale). Ownership: `NslTensor::publish`, same
/// persistent-state lifecycle as both parents.
#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_like_host_dtype(template_ptr: i64, dtype: i64) -> i64 {
    let t = unsafe { &*(template_ptr as *const NslTensor) };
    let dtype = dtype as u16;
    let elem_size: usize = match dtype {
        DTYPE_F32 => 4,
        DTYPE_FP16 | DTYPE_BF16 => 2,
        other => panic!(
            "nsl_tensor_zeros_like_host_dtype: unsupported dtype {other} (F32=1, FP16=2, BF16=3)"
        ),
    };
    if t.device == 0 {
        return nsl_tensor_zeros_like_dtype(template_ptr, dtype as i64);
    }
    let len = t.len as usize;
    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);
    let data = crate::tensor::alloc_host_state_buffer(len * elem_size);
    let out = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        t.ndim,
        t.len,
        0, // CPU
        dtype,
        1, // owns_data
        0, // data_owner
    ));
    NslTensor::publish(out)
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

    /// CFTP v7 follow-on (finding 7/2): CPU `f32_to_bf16_bits` now uses
    /// IEEE-754 RTE (round-to-nearest-even) via `half::bf16::from_f32`,
    /// matching PTX `cvt.rn.bf16.f32` bit-for-bit on every input. Pin
    /// the rounding so a regression to the truncating implementation is
    /// caught immediately.
    #[test]
    fn f32_to_bf16_bits_uses_rte_not_truncation() {
        use crate::tensor::f32_to_bf16_bits;
        // f32 = 0x3F80FFFF (just below 0x3F810000). Lower 16 bits = 0xFFFF
        // > 0x8000, so RTE rounds UP from 0x3F80 to 0x3F81. Truncation
        // would yield 0x3F80 (the buggy v6 behavior).
        let v = f32::from_bits(0x3F80FFFF);
        let bits = f32_to_bf16_bits(v);
        assert_eq!(
            bits, 0x3F81,
            "f32_to_bf16_bits MUST round-to-nearest-even (got 0x{bits:04X}, \
             expected 0x3F81). Truncation here would silently disagree with \
             PTX cvt.rn.bf16.f32 by one full ULP on the CPU staging path."
        );
        // Exact halfway: f32 = 0x3F808000. Lower 16 = 0x8000 == halfway.
        // RTE: round to even (mant LSB = 0) → keep 0x3F80.
        let v2 = f32::from_bits(0x3F808000);
        assert_eq!(
            f32_to_bf16_bits(v2),
            0x3F80,
            "RTE halfway with even-LSB neighbour must round DOWN (banker's rounding)"
        );
        // Exact halfway with odd LSB: f32 = 0x3F818000. Lower 16 = 0x8000;
        // mant LSB = 1 (odd) → round UP to 0x3F82.
        let v3 = f32::from_bits(0x3F818000);
        assert_eq!(
            f32_to_bf16_bits(v3),
            0x3F82,
            "RTE halfway with odd-LSB neighbour must round UP to the even mantissa"
        );
    }

    /// CFTP v7 follow-on (finding 7/2): same RTE pin for fp16.
    #[test]
    fn f32_to_f16_bits_uses_rte_not_truncation() {
        use crate::tensor::f32_to_f16_bits;
        // f32 = 0x3F80FFFF (mant bits: 0x00FFFF). f16 mant = top 10 of 23 =
        // 0x00FFFF >> 13 = 7; round bit = bit 12 = 1; sticky = bits 0-11 =
        // 0xFFF (non-zero). > halfway → round UP from 7 to 8. Expected 0x3C08.
        // v6 truncation would have produced 0x3C07.
        let v = f32::from_bits(0x3F80FFFF);
        let bits = f32_to_f16_bits(v);
        assert_eq!(
            bits, 0x3C08,
            "f32_to_f16_bits MUST round-to-nearest-even (got 0x{bits:04X}, \
             expected 0x3C08). v6 truncation produced 0x3C07."
        );
        // Halfway tie with even neighbour: f32 = 0x3F801000 (mant LSB=0 in
        // f16). Lower 13 = 0x1000 exactly; sticky = 0; tie → round to even.
        // f16 mant stays 0 → 0x3C00.
        let v2 = f32::from_bits(0x3F801000);
        assert_eq!(
            f32_to_f16_bits(v2),
            0x3C00,
            "RTE halfway with even neighbour must round DOWN"
        );
        // Halfway tie with odd neighbour: f32 = 0x3F803000 (f16 base mant=1).
        // Lower 13 = 0x1000 exact halfway; tie → round to even. Round UP
        // from 1 to 2 → 0x3C02.
        let v3 = f32::from_bits(0x3F803000);
        assert_eq!(
            f32_to_f16_bits(v3),
            0x3C02,
            "RTE halfway with odd neighbour must round UP to even"
        );
    }

    /// CFTP v7 follow-on (finding 8): NaN preservation — both bf16 and f16
    /// primitives previously silently flushed NaN to ±Inf because the
    /// finite-overflow check (`exp >= 143` for f16; `bits >> 16` for bf16)
    /// also matched `exp == 255`. Delegating to `half` preserves NaN.
    #[test]
    fn f32_to_f16_bits_preserves_nan() {
        use crate::tensor::f32_to_f16_bits;
        let nan_bits = f32_to_f16_bits(f32::NAN);
        // f16 NaN: exp = 0x1F (top 5 bits), mant != 0. f16 +Inf is 0x7C00
        // (mant == 0); any quiet NaN has mant != 0.
        let exp = (nan_bits >> 10) & 0x1F;
        let mant = nan_bits & 0x3FF;
        assert_eq!(exp, 0x1F, "NaN must produce f16 exp=0x1F (not zero)");
        assert_ne!(
            mant, 0,
            "NaN must produce f16 NaN (mant != 0) — finding-8: previous \
             implementation collapsed NaN to ±Inf (mant == 0)"
        );
    }

    #[test]
    fn f32_to_bf16_bits_preserves_nan() {
        use crate::tensor::f32_to_bf16_bits;
        // sNaN with payload only in low 16 mantissa bits: 0x7F800001.
        // v6 truncation: bits >> 16 = 0x7F80 = bf16 +Inf (silent NaN→Inf).
        // v7 (half crate): preserves NaN.
        let snan = f32::from_bits(0x7F80_0001);
        let bits = f32_to_bf16_bits(snan);
        let exp = (bits >> 7) & 0xFF;
        let mant = bits & 0x7F;
        assert_eq!(exp, 0xFF, "bf16 NaN must have exp=0xFF");
        assert_ne!(
            mant, 0,
            "bf16 NaN MUST keep a non-zero mantissa (got 0x{bits:04X}). \
             v6 truncation produced 0x7F80 (= bf16 +Inf), silently \
             converting NaN to Inf."
        );
        // Standard qNaN — always preserved.
        let qnan_bits = f32_to_bf16_bits(f32::NAN);
        let qexp = (qnan_bits >> 7) & 0xFF;
        let qmant = qnan_bits & 0x7F;
        assert_eq!(qexp, 0xFF);
        assert_ne!(qmant, 0);
    }

    // Cost-model audit finding 2 HISTORY: `refuse_gpu_moment_template`
    // (and its `#[should_panic]` guard test) lived here between PR #367
    // and the 2026-07-14 deferral closure. The refusal existed because
    // the dequant→step→quant envelope was CPU-only; the envelope's three
    // primitives (`nsl_tensor_zeros_like_dtype`, `nsl_tensor_cast`,
    // `nsl_tensor_cast_into`) now have real GPU paths through the
    // CFTP-v7 PTX cast kernels, so a GPU template is a supported input —
    // covered by `precision_cast_gpu_dispatch.rs` (cuda, --ignored).

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

    // -----------------------------------------------------------------
    // P0.3 — offload x precision composition, CPU (GPU-less) degradations.
    // The GPU fast paths live in the `gpu` submodule below (cuda-gated).
    // -----------------------------------------------------------------

    /// cast_from_host with a CPU template degrades to the co-resident CPU
    /// `nsl_tensor_cast(src, F32)` — a fresh owned F32 widening of the host
    /// reduced-precision state (placement-transparent GPU-less run).
    #[test]
    fn cast_from_host_cpu_template_widens_to_f32() {
        let vals = [1.5_f32, -2.25, 0.5];
        let src = fp16_tensor(&vals);
        let template = f32_tensor(&[0.0; 3]); // CPU placement donor
        let work = nsl_tensor_cast_from_host(src, template);
        let t = unsafe { &*(work as *const NslTensor) };
        assert_eq!(t.dtype, DTYPE_F32, "working tensor must be F32");
        assert_eq!(t.device, 0, "CPU template keeps the working tensor on CPU");
        let got = read_f32(work);
        for (i, &v) in vals.iter().enumerate() {
            // Exact: these values round-trip fp16 losslessly.
            let expected = f16_bits_to_f32(f32_to_f16_bits(v));
            assert_eq!(got[i], expected, "elem {i}");
        }
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(template);
        crate::tensor::nsl_tensor_free(work);
    }

    /// cast_to_host_into on a CPU run = co-resident `nsl_tensor_cast_into`
    /// plus the consuming free of the staged working tensor (ownership
    /// contract shared with the GPU paths).
    #[test]
    fn cast_to_host_into_cpu_quantizes_and_consumes_src() {
        let dst = fp16_tensor(&[0.0, 0.0, 0.0]);
        let src = f32_tensor(&[1.0001, 3.14159, -2.5]);
        // Keep src alive across the consuming call so the refcount drop is
        // observable (rc 2 -> 1) rather than an actual free.
        crate::tensor::nsl_tensor_retain(src);
        nsl_tensor_cast_to_host_into(dst, src);
        let src_rc = unsafe { &*(src as *const NslTensor) }
            .refcount
            .load(std::sync::atomic::Ordering::SeqCst);
        assert_eq!(src_rc, 1, "cast_to_host_into must CONSUME src (one refcount drop)");
        let t = unsafe { &*(dst as *const NslTensor) };
        let bits = unsafe { std::slice::from_raw_parts(t.data as *const u16, 3) };
        for (i, &v) in [1.0001_f32, 3.14159, -2.5].iter().enumerate() {
            assert_eq!(bits[i], f32_to_f16_bits(v), "elem {i} not quantized to the primitive");
        }
        crate::tensor::nsl_tensor_free(dst);
        crate::tensor::nsl_tensor_free(src);
    }

    /// zeros_like_host_dtype with a CPU template delegates to
    /// `nsl_tensor_zeros_like_dtype` so a GPU-less run with offload +
    /// precision behaves exactly like precision-only.
    #[test]
    fn zeros_like_host_dtype_cpu_template_delegates() {
        let template = f32_tensor(&[5.0, 6.0, 7.0]);
        let z = nsl_tensor_zeros_like_host_dtype(template, DTYPE_BF16 as i64);
        let t = unsafe { &*(z as *const NslTensor) };
        assert_eq!(t.dtype, DTYPE_BF16, "dtype is BF16");
        assert_eq!(t.device, 0, "host-resident");
        assert_eq!(t.len, 3);
        let bits = unsafe { std::slice::from_raw_parts(t.data as *const u16, 3) };
        assert!(bits.iter().all(|&b| bf16_bits_to_f32(b) == 0.0), "all zero");
        crate::tensor::nsl_tensor_free(template);
        crate::tensor::nsl_tensor_free(z);
    }

    /// P0.3 GPU coverage — gated exactly like `cuda/mod.rs`'s unit tests:
    /// compiled only under `--features cuda`, requires a live GPU at runtime
    /// (`cargo test -p nsl-runtime --features cuda`).
    #[cfg(feature = "cuda")]
    mod gpu {
        use super::*;

        /// Offloaded state buffers must come back PINNED once a CUDA
        /// context is live, and `nsl_tensor_free` must route the buffer
        /// through the pinned registry to `cuMemFreeHost`.
        #[test]
        fn zeros_like_host_f32_is_pinned_and_free_unregisters() {
            if crate::tensor::offload_pageable_forced() {
                eprintln!("skipping: NSL_OFFLOAD_PAGEABLE=1 in the environment");
                return;
            }
            let cpu = f32_tensor(&[1.0; 8]);
            let gpu_t = crate::tensor::nsl_tensor_to_device(cpu, 1);
            let host_state = crate::tensor::nsl_tensor_zeros_like_host_f32(gpu_t);
            let t = unsafe { &*(host_state as *const NslTensor) };
            assert_eq!(t.device, 0, "state is host-resident");
            assert_eq!(t.dtype, DTYPE_F32);
            assert!(
                crate::cuda::inner::is_pinned(t.data),
                "offloaded state must be pinned when a CUDA context is live"
            );
            let data = t.data;
            crate::tensor::nsl_tensor_free(host_state);
            assert!(
                !crate::cuda::inner::is_pinned(data),
                "nsl_tensor_free must unregister + cuMemFreeHost the pinned buffer"
            );
            crate::tensor::nsl_tensor_free(gpu_t);
            crate::tensor::nsl_tensor_free(cpu);
        }

        /// P0.2 ownership contract: on the pinned async path the staged
        /// tensor's free is DEFERRED until nsl_offload_drain(); the copy
        /// itself must land after the drain.
        #[test]
        fn copy_data_async_pinned_defers_free_until_drain() {
            if crate::tensor::offload_pageable_forced() || crate::tensor::offload_sync_forced() {
                eprintln!("skipping: offload kill-switch env set");
                return;
            }
            let vals = [3.0_f32, -4.0, 5.5, 0.25];
            let cpu = f32_tensor(&vals);
            let dev = crate::tensor::nsl_tensor_to_device(cpu, 1);
            let host = crate::tensor::nsl_tensor_zeros_like_host_f32(dev);
            // rc 2 so the consuming free is observable, not destructive.
            crate::tensor::nsl_tensor_retain(dev);
            crate::tensor::nsl_tensor_copy_data_async(host, dev);
            let rc_before = unsafe { &*(dev as *const NslTensor) }
                .refcount
                .load(std::sync::atomic::Ordering::SeqCst);
            assert_eq!(
                rc_before, 2,
                "async path must NOT free the staged tensor before the drain"
            );
            crate::tensor::nsl_offload_drain();
            let rc_after = unsafe { &*(dev as *const NslTensor) }
                .refcount
                .load(std::sync::atomic::Ordering::SeqCst);
            assert_eq!(rc_after, 1, "drain must perform the deferred free");
            let got = read_f32(host);
            assert_eq!(got, vals.to_vec(), "async DtoH must land by drain time");
            crate::tensor::nsl_tensor_free(host);
            crate::tensor::nsl_tensor_free(dev);
            crate::tensor::nsl_tensor_free(cpu);
        }

        /// P0.3 combined envelope end-to-end on the GPU: pinned host FP16
        /// state -> device F32 working tensor -> quant-cast back into the
        /// same host buffer (async, drain-deferred frees). Values chosen
        /// exactly representable in fp16 so the round-trip is bit-exact.
        #[test]
        fn cast_from_host_and_back_gpu_round_trip_fp16() {
            let vals = [1.0_f32, -2.5, 0.75, 1234.5];
            let cpu = f32_tensor(&vals);
            let theta = crate::tensor::nsl_tensor_to_device(cpu, 1);
            // Pinned host FP16 state buffer (the P0.3 allocation path).
            let host = nsl_tensor_zeros_like_host_dtype(theta, DTYPE_FP16 as i64);
            let ht = unsafe { &*(host as *const NslTensor) };
            assert_eq!(ht.dtype, DTYPE_FP16);
            {
                let bits = unsafe { std::slice::from_raw_parts_mut(ht.data as *mut u16, 4) };
                for (i, &v) in vals.iter().enumerate() {
                    bits[i] = f32_to_f16_bits(v);
                }
            }
            // Stage-in: host fp16 -> device f32.
            let work = nsl_tensor_cast_from_host(host, theta);
            let wt = unsafe { &*(work as *const NslTensor) };
            assert!(wt.device > 0, "working tensor must be device-resident");
            assert_eq!(wt.dtype, DTYPE_F32);
            // Stage-out: device f32 -> host fp16 (consumes `work`).
            nsl_tensor_cast_to_host_into(host, work);
            crate::tensor::nsl_offload_drain();
            let bits = unsafe { std::slice::from_raw_parts(ht.data as *const u16, 4) };
            for (i, &v) in vals.iter().enumerate() {
                assert_eq!(
                    bits[i],
                    f32_to_f16_bits(v),
                    "elem {i}: fp16 -> f32 -> fp16 GPU round-trip must be bit-exact"
                );
            }
            crate::tensor::nsl_tensor_free(host);
            crate::tensor::nsl_tensor_free(theta);
            crate::tensor::nsl_tensor_free(cpu);
        }
    }
}
