//! D2b: weight layer-streaming — pointer-identity host offload for model
//! parameters.
//!
//! The CSLA layer-major schedule (`--layerwise-accum`) touches each
//! layer-grouped parameter in exactly two bracketed regions: its layer's
//! primal segment (forward, every micro-batch) and its layer's replay range
//! + per-layer update (window backward). Between those brackets the device
//! copy is dead weight — at 1B it is 4 GiB of f32 that forced the m/v
//! streaming compromise; at 7B it is 28.7 GiB and the difference between
//! possible and impossible.
//!
//! Part 2 (forward segment streaming) closes the loop: the forward is
//! lowered per CCR block segment with an upload before each layer's
//! segment and a read-only evict after its last primal read, so a streamed
//! parameter holds device memory ONLY inside its brackets for the whole
//! training loop (registration happens idempotently at step-body top;
//! teardown restores residency for `model_save`/eval). Consequence: any
//! OTHER reader of θ during training — a user callback dereferencing
//! model fields in `on_step`/`on_epoch`, a mid-training `model_save` —
//! hits an evicted tensor and crashes loudly on the null data pointer.
//!
//! Mechanism: a runtime SIDE TABLE mapping tensor pointer → pinned host
//! mirror. The `NslTensor` ABI (13 fields, `#[repr(C)]`, documented layout)
//! does NOT change, and the tensor POINTER never changes — `param_list`,
//! struct fields, the CSLA tie-guard, and every captured SSA value stay
//! valid. Eviction frees the device buffer and nulls `t.data`; upload
//! re-allocates and copies the mirror back. A read of an evicted tensor
//! dereferences/launches on a null data pointer — loud, not silent (and
//! `nsl_tensor_free` on an evicted tensor is safe: the device-free path
//! no-ops on null).
//!
//! All transfers are SYNCHRONOUS in v1 (correctness first; the async
//! overlap belongs to a follow-up with the transfer-stream/drain protocol
//! the m/v path uses).

#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::os::raw::c_void;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

use crate::tensor::NslTensor;

#[cfg(feature = "cuda")]
struct Mirror {
    host: *mut u8,
    bytes: usize,
    /// The device pointer this param last held (0 = currently evicted). Used
    /// to count actual pointer MOVES: an upload that hands back a different
    /// device address than the param held before is concrete evidence the
    /// storage was freed and reallocated (the exact hazard the #397 view-of-θ
    /// exclusion protects a live buffered view from). A view-rooted param is
    /// never registered, so its pointer never moves.
    last_dev: i64,
}
// Raw pointers in the table are only touched under the lock on the
// (single-threaded) training path.
#[cfg(feature = "cuda")]
unsafe impl Send for Mirror {}

#[cfg(feature = "cuda")]
static MIRRORS: Mutex<Option<HashMap<i64, Mirror>>> = Mutex::new(None);

/// Anti-vacuity counters for the gates (uploads / evictions performed).
/// WS_EVICTS_WB counts the writeback subset separately: the total evict
/// count alone cannot distinguish the post-update writeback evict from a
/// read-only one absorbed by the idempotent step-top register belt (review
/// D2b-2-5) — dropping the writeback leg would leave totals unchanged but
/// silently train on stale mirrors after the first window.
pub static WS_UPLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WS_EVICTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WS_EVICTS_WB: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Distinct parameters that entered the streaming table (were evicted at least
/// once → their original device storage was freed). A view-rooted param that
/// the #397 exclusion keeps resident is NEVER registered, so it never appears
/// here — this is the per-param pointer-STABILITY signal the view gate asserts.
pub static WS_REGISTERED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Uploads that returned a device address DIFFERENT from the one the param
/// last held — concrete "the pointer moved" evidence for streamed params.
pub static WS_PTR_MOVES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[no_mangle]
pub extern "C" fn nsl_weight_stream_upload_count() -> i64 {
    WS_UPLOADS.load(std::sync::atomic::Ordering::Relaxed) as i64
}

#[no_mangle]
pub extern "C" fn nsl_weight_stream_registered_count() -> i64 {
    WS_REGISTERED.load(std::sync::atomic::Ordering::Relaxed) as i64
}

#[no_mangle]
pub extern "C" fn nsl_weight_stream_ptr_moves() -> i64 {
    WS_PTR_MOVES.load(std::sync::atomic::Ordering::Relaxed) as i64
}

/// Register a GPU-resident f32 parameter for streaming: allocate a pinned
/// host mirror, copy the current device bytes into it, free the device
/// buffer, and null `t.data` (evicted state).
///
/// IDEMPOTENT: on an already-registered RESIDENT tensor this degenerates
/// to a writeback-free evict — the mirror is current by construction
/// (post-update evicts write back; forwards never mutate θ), so the
/// device bytes equal the mirror and a pure free suffices; on an
/// already-registered EVICTED tensor it is a no-op. Emitted at step-body
/// top every micro-batch (part 2: makes the very first forward stream)
/// and again at each window start as a belt.
///
/// Aborts loudly on CPU tensors, views, or non-owning tensors — the
/// codegen admission routes only plain owning device params here.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_register(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let t = NslTensor::from_ptr(tensor_ptr);
    if t.device == 0 {
        eprintln!(
            "[weight-stream] FATAL: --weight-stream requires GPU placement — \
             call m.to(cuda) before the train block (param tensor {tensor_ptr} \
             is CPU-resident)"
        );
        std::process::abort();
    }
    if t.owns_data == 0 || t.data_owner != 0 || t.slab_managed != 0 {
        eprintln!(
            "[weight-stream] refusing to register tensor {tensor_ptr}: \
             owns_data={} data_owner={} slab_managed={} — only plain owning \
             non-slab GPU tensors stream",
            t.owns_data, t.data_owner, t.slab_managed
        );
        std::process::abort();
    }
    #[cfg(feature = "cuda")]
    {
        {
            let guard = MIRRORS.lock().unwrap();
            if guard.as_ref().is_some_and(|g| g.contains_key(&tensor_ptr)) {
                drop(guard);
                // Already registered: window-start re-evict, mirror current.
                nsl_weight_stream_evict(tensor_ptr, 0);
                return;
            }
        }
        let bytes = t.data_byte_size();
        let host = crate::tensor::alloc_host_state_buffer(bytes);
        crate::cuda::inner::ensure_context();
        crate::cuda::inner::memcpy_dtoh(host as *mut c_void, t.data, bytes);
        crate::cuda::inner::free_managed(t.data);
        t.data = std::ptr::null_mut();
        let mut guard = MIRRORS.lock().unwrap();
        guard
            .get_or_insert_with(HashMap::new)
            .insert(tensor_ptr, Mirror { host, bytes, last_dev: 0 });
        // First registration = this param's original device storage was freed
        // (its pointer is now invalid). A view-rooted param excluded by #397
        // never reaches here.
        WS_REGISTERED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[weight-stream] register called in a non-CUDA build");
        std::process::abort();
    }
}

/// Upload an evicted parameter's mirror back to a fresh device buffer.
/// Idempotent when already resident. Aborts on an unregistered tensor —
/// codegen must only stream registered params.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_upload(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let t = NslTensor::from_ptr(tensor_ptr);
    if !t.data.is_null() {
        return; // already resident
    }
    #[cfg(feature = "cuda")]
    {
        let mut guard = MIRRORS.lock().unwrap();
        let Some(m) = guard.as_mut().and_then(|g| g.get_mut(&tensor_ptr)) else {
            eprintln!(
                "[weight-stream] FATAL: upload of unregistered tensor {tensor_ptr}"
            );
            std::process::abort();
        };
        crate::cuda::inner::ensure_context();
        let dev = crate::cuda::inner::alloc_managed(m.bytes);
        crate::cuda::inner::memcpy_htod(dev, m.host as *const c_void, m.bytes);
        t.data = dev;
        // Pointer-move accounting: a fresh allocation at a different address
        // than the param last held is concrete evidence the storage moved.
        let dev_i = dev as i64;
        if m.last_dev != 0 && m.last_dev != dev_i {
            WS_PTR_MOVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        m.last_dev = dev_i;
        WS_UPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("[weight-stream] upload called in a non-CUDA build");
        std::process::abort();
    }
}

/// Evict a resident parameter: optionally write the device bytes back to
/// the mirror (θ after its per-layer update; forward-side evictions are
/// read-only and skip it), free the device buffer, null `t.data`.
/// Idempotent when already evicted.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_evict(tensor_ptr: i64, writeback: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let t = NslTensor::from_ptr(tensor_ptr);
    if t.data.is_null() {
        return; // already evicted
    }
    #[cfg(feature = "cuda")]
    {
        let guard = MIRRORS.lock().unwrap();
        let Some(m) = guard.as_ref().and_then(|g| g.get(&tensor_ptr)) else {
            // No call site legitimately evicts an unregistered tensor —
            // a silent pass here would mask a codegen slip (review D2b-6).
            eprintln!(
                "[weight-stream] FATAL: evict of unregistered tensor {tensor_ptr}"
            );
            std::process::abort();
        };
        crate::cuda::inner::ensure_context();
        if writeback != 0 {
            crate::cuda::inner::memcpy_dtoh(m.host as *mut c_void, t.data, m.bytes);
        }
        crate::cuda::inner::free_managed(t.data);
        t.data = std::ptr::null_mut();
        WS_EVICTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if writeback != 0 {
            WS_EVICTS_WB.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = writeback;
        eprintln!("[weight-stream] evict called in a non-CUDA build");
        std::process::abort();
    }
}

/// Restore every registered, currently-evicted parameter to device
/// residency WITHOUT dropping the table. Part 1 emitted this after the
/// window's epilogue updates (window-scoped cycle); part 2's segment-
/// streamed forward re-uploads per layer instead, so codegen no longer
/// emits it — kept as a runtime API for tooling and as the documented
/// "make everything resident now" escape hatch.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_upload_all() {
    #[cfg(feature = "cuda")]
    {
        let keys: Vec<i64> = {
            let guard = MIRRORS.lock().unwrap();
            guard
                .as_ref()
                .map(|g| g.keys().copied().collect())
                .unwrap_or_default()
        };
        for ptr in keys {
            nsl_weight_stream_upload(ptr);
        }
    }
}

/// Is `tensor_ptr` a currently-registered streamed parameter? Used by
/// `nsl_model_save` to materialize an evicted param from its mirror for the
/// duration of the serialization read (Item 12: mid-loop model_save no
/// longer crashes on a null data pointer).
#[no_mangle]
pub extern "C" fn nsl_weight_stream_is_registered(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    #[cfg(feature = "cuda")]
    {
        let guard = MIRRORS.lock().unwrap();
        return guard.as_ref().is_some_and(|g| g.contains_key(&tensor_ptr)) as i64;
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Re-evict every registered, currently-RESIDENT parameter, restoring the
/// streamed (evicted) invariant after a scoped `upload_all`. This is the
/// close bracket of the Item 12 callback guard: a callback that touches
/// model θ under `--weight-stream` runs between `upload_all` (open) and this
/// (close), so its field reads see resident data and its writes are captured.
///
/// `writeback`: pass 1 when the guarded callback might MUTATE θ (any assign
/// to a model field, or a method call on one) — the device bytes are copied
/// back to the mirror so the mutation survives the next window's upload; pass
/// 0 for a read-only callback (logging, `model_save`) to skip the DtoH. The
/// codegen picks this from a compile-time read-only analysis of the body.
///
/// Idempotent: an already-evicted param is skipped by `evict`. Emitted only
/// when a callback actually references the model, so the steady-state
/// transfer arithmetic the gates assert is untouched.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_reevict_all(writeback: i64) {
    #[cfg(feature = "cuda")]
    {
        let keys: Vec<i64> = {
            let guard = MIRRORS.lock().unwrap();
            guard
                .as_ref()
                .map(|g| g.keys().copied().collect())
                .unwrap_or_default()
        };
        for ptr in keys {
            nsl_weight_stream_evict(ptr, writeback);
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = writeback;
    }
}

/// Restore every registered parameter to device residency (final upload +
/// table drop). Emitted at train-block teardown so post-training code
/// (model_save, eval) sees ordinary device tensors, and the pinned mirrors
/// are released.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_teardown() {
    #[cfg(feature = "cuda")]
    {
        let mut guard = MIRRORS.lock().unwrap();
        let Some(table) = guard.take() else { return };
        for (ptr, m) in table {
            let t = NslTensor::from_ptr(ptr);
            crate::cuda::inner::ensure_context();
            if t.data.is_null() {
                // NOTE: deliberately NOT counted in WS_UPLOADS — under
                // part-2 whole-loop streaming EVERY streamed param arrives
                // here evicted (this branch is the steady state, and this
                // restore is load-bearing for model_save/eval), and the
                // gates' designed transfer arithmetic (uploads =
                // fwd_microbatches × streamed + windows × streamed)
                // depends on teardown restores staying out of the count.
                // Allocations here run under the ambient surface (post-
                // training accounting only; frees reconcile by recorded
                // surface).
                let dev = crate::cuda::inner::alloc_managed(m.bytes);
                crate::cuda::inner::memcpy_htod(dev, m.host as *const c_void, m.bytes);
                t.data = dev;
            }
            // Mirror free: pinned buffers go back to the driver, pageable
            // fallbacks to the heap (same routing as free_host_tensor_data).
            if crate::cuda::inner::is_pinned(m.host as *mut c_void) {
                crate::cuda::inner::free_pinned(m.host as *mut c_void);
                continue;
            }
            unsafe { crate::memory::checked_free(m.host, m.bytes) };
        }
    }
}
