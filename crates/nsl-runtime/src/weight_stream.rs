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
}
// Raw pointers in the table are only touched under the lock on the
// (single-threaded) training path.
#[cfg(feature = "cuda")]
unsafe impl Send for Mirror {}

#[cfg(feature = "cuda")]
static MIRRORS: Mutex<Option<HashMap<i64, Mirror>>> = Mutex::new(None);

/// Anti-vacuity counters for the gates (uploads / evictions performed).
pub static WS_UPLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WS_EVICTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[no_mangle]
pub extern "C" fn nsl_weight_stream_upload_count() -> i64 {
    WS_UPLOADS.load(std::sync::atomic::Ordering::Relaxed) as i64
}

/// Register a GPU-resident f32 parameter for streaming: allocate a pinned
/// host mirror, copy the current device bytes into it, free the device
/// buffer, and null `t.data` (evicted state).
///
/// IDEMPOTENT ACROSS WINDOWS: on an already-registered RESIDENT tensor this
/// degenerates to a writeback-free evict — the mirror is current by
/// construction (post-update evicts write back; forwards never mutate θ;
/// the window-end restore uploads from that same mirror), so the device
/// bytes equal the mirror and a pure free suffices. Emitted at every
/// window start for each layer-grouped param.
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
            .insert(tensor_ptr, Mirror { host, bytes });
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
        let guard = MIRRORS.lock().unwrap();
        let Some(m) = guard.as_ref().and_then(|g| g.get(&tensor_ptr)) else {
            eprintln!(
                "[weight-stream] FATAL: upload of unregistered tensor {tensor_ptr}"
            );
            std::process::abort();
        };
        crate::cuda::inner::ensure_context();
        let dev = crate::cuda::inner::alloc_managed(m.bytes);
        crate::cuda::inner::memcpy_htod(dev, m.host as *const c_void, m.bytes);
        t.data = dev;
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
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = writeback;
        eprintln!("[weight-stream] evict called in a non-CUDA build");
        std::process::abort();
    }
}

/// Restore every registered, currently-evicted parameter to device
/// residency WITHOUT dropping the table — emitted after the window's
/// epilogue updates so the next iterations' forwards see ordinary resident
/// weights (the window-scoped eviction cycle).
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
                // NOTE: deliberately NOT counted in WS_UPLOADS — steady
                // state is resident at teardown, and the gates' designed
                // transfer arithmetic (uploads = windows x params x 2)
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
