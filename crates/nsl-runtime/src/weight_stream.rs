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
    /// Item 10 (layer arenas): when this param is resident inside a shared
    /// device arena slot rather than its own `alloc_managed` buffer, the slot
    /// index (else -1). `arena_off` is its byte offset within that slot.
    /// Evicting an arena-resident param releases a reference on the slot
    /// instead of `free_managed`-ing an interior pointer (which is not a valid
    /// allocation base).
    arena_slot: i64,
    arena_off: usize,
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
/// Item 10 (layer arenas): contiguous-layer-pack transfers. Each
/// `upload_pack` / `evict_pack` is ONE HtoD / DtoH covering a whole layer's
/// params, so these stay far below the param-granular `WS_UPLOADS`/`WS_EVICTS`
/// (which still count per-param) — the compile-time-verified batching win.
pub static WS_PACK_UPLOADS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WS_PACK_EVICTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Item 11: pack uploads issued ASYNCHRONOUSLY as prefetches (a subset of
/// `WS_PACK_UPLOADS`) — the overlap evidence the double-buffer gate asserts.
pub static WS_PREFETCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

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
    #[cfg(feature = "cuda")]
    {
        {
            let guard = MIRRORS.lock().unwrap();
            if guard.as_ref().is_some_and(|g| g.contains_key(&tensor_ptr)) {
                drop(guard);
                // Already registered: window-start re-evict, mirror current.
                // The admission check below runs ONLY on first registration —
                // once streaming, a param's `owns_data` legitimately flips to 0
                // while it is a non-owning view inside an arena slot (Item 10),
                // so re-checking it here would abort every step after the first.
                nsl_weight_stream_evict(tensor_ptr, 0);
                return;
            }
        }
        // First-registration admission: only plain owning non-slab GPU tensors
        // may enter the streaming table (a view/slab param would corrupt on
        // free-and-reupload — the #397 hazard).
        if t.owns_data == 0 || t.data_owner != 0 || t.slab_managed != 0 {
            eprintln!(
                "[weight-stream] refusing to register tensor {tensor_ptr}: \
                 owns_data={} data_owner={} slab_managed={} — only plain owning \
                 non-slab GPU tensors stream",
                t.owns_data, t.data_owner, t.slab_managed
            );
            std::process::abort();
        }
        let bytes = t.data_byte_size();
        let host = crate::tensor::alloc_host_state_buffer(bytes);
        crate::cuda::inner::ensure_context();
        crate::cuda::inner::memcpy_dtoh(host as *mut c_void, t.data, bytes);
        crate::cuda::inner::free_managed(t.data);
        t.data = std::ptr::null_mut();
        let mut guard = MIRRORS.lock().unwrap();
        guard.get_or_insert_with(HashMap::new).insert(
            tensor_ptr,
            Mirror {
                host,
                bytes,
                last_dev: 0,
                arena_slot: -1,
                arena_off: 0,
            },
        );
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
        // This is a plain OWNED buffer (not an arena view): restore owns_data
        // so a param that was last arena-resident (owns_data=0) is a
        // well-formed owned tensor again — the mirror-buffer flag must always
        // match reality (review Item-10 finding C).
        t.owns_data = 1;
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
        let mut guard = MIRRORS.lock().unwrap();
        let Some(m) = guard.as_mut().and_then(|g| g.get_mut(&tensor_ptr)) else {
            // No call site legitimately evicts an unregistered tensor —
            // a silent pass here would mask a codegen slip (review D2b-6).
            eprintln!(
                "[weight-stream] FATAL: evict of unregistered tensor {tensor_ptr}"
            );
            std::process::abort();
        };
        crate::cuda::inner::ensure_context();
        if writeback != 0 {
            // Reading the arena INTERIOR pointer for a DtoH is valid; only
            // FREEING an interior pointer would be illegal.
            crate::cuda::inner::memcpy_dtoh(m.host as *mut c_void, t.data, m.bytes);
        }
        if m.arena_slot >= 0 {
            // Item 10: arena-resident — release a reference on the shared slot
            // rather than free_managed-ing an interior pointer. The slot's
            // device/host buffers are pooled for reuse, not freed here.
            let slot = m.arena_slot as usize;
            m.arena_slot = -1;
            arena_release_ref(slot);
        } else {
            crate::cuda::inner::free_managed(t.data);
        }
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

// ─────────────────────────────────────────────────────────────────────────
// Item 10 — layer arenas: stable device staging slots + contiguous layer-pack
// transfers. Instead of one `alloc_managed`/`memcpy` per parameter (hundreds
// per step), a whole layer's params are staged into one contiguous PINNED
// host buffer and moved with ONE HtoD into a reused device slot; each param's
// `t.data` then points at its offset inside the slot (a non-owning view). This
// cuts CUDA calls, lands one large PCIe transaction, keeps device addresses
// stable across steps (CUDA-Graph-friendly), and bounds fragmentation. The
// host mirror (from `register`) stays the source of truth, so registration and
// teardown are shared with the per-param path — only residency changes.
// ─────────────────────────────────────────────────────────────────────────

/// One reusable staging slot: a device buffer + a matching pinned host buffer.
/// A slot holds exactly one resident layer pack at a time (`live` = params in
/// it); when `live` hits 0 the buffers stay allocated for the next pack.
#[cfg(feature = "cuda")]
struct ArenaSlot {
    dev: *mut c_void,
    host_stage: *mut u8,
    cap: usize,
    live: usize,
    /// Item 11: a prefetch HtoD's completion event, recorded on the transfer
    /// stream and not yet awaited (0 = none). `await_pack` discharges it onto
    /// the compute stream before any kernel reads this slot.
    pending_event: u64,
}
#[cfg(feature = "cuda")]
unsafe impl Send for ArenaSlot {}

#[cfg(feature = "cuda")]
static ARENA_POOL: Mutex<Vec<ArenaSlot>> = Mutex::new(Vec::new());

/// Per-param device sub-buffer alignment inside a slot. cudaMalloc bases are
/// ≥256-byte aligned; a mid-slot offset must be too so kernels that assume a
/// vectorized/aligned base on `t.data` stay correct. 512 is a safe multiple.
#[cfg(feature = "cuda")]
const ARENA_ALIGN: usize = 512;

#[cfg(feature = "cuda")]
fn align_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// Reserve a slot with capacity ≥ `bytes`, marking it holding `live` params.
/// Reuses a free slot (stable address) or grows the pool by one. Returns the
/// slot index and its (device, pinned-host) buffers. Caller holds MIRRORS;
/// lock order is MIRRORS → ARENA_POOL → allocator (never reversed).
#[cfg(feature = "cuda")]
fn arena_acquire(bytes: usize, live: usize) -> (usize, *mut c_void, *mut u8) {
    let mut pool = ARENA_POOL.lock().unwrap();
    if let Some(idx) = pool.iter().position(|s| s.live == 0 && s.cap >= bytes) {
        pool[idx].live = live;
        return (idx, pool[idx].dev, pool[idx].host_stage);
    }
    // Round up so slightly-different pack sizes can reuse the same slot.
    let cap = align_up(bytes.max(ARENA_ALIGN), 256 * 1024);
    crate::cuda::inner::ensure_context();
    let dev = crate::cuda::inner::alloc_managed(cap);
    let host_stage = crate::tensor::alloc_host_state_buffer(cap);
    pool.push(ArenaSlot {
        dev,
        host_stage,
        cap,
        live,
        pending_event: 0,
    });
    (pool.len() - 1, dev, host_stage)
}

/// Buffers of an in-use slot (device, pinned-host).
#[cfg(feature = "cuda")]
fn arena_slot_bufs(slot: usize) -> (*mut c_void, *mut u8) {
    let pool = ARENA_POOL.lock().unwrap();
    let s = &pool[slot];
    (s.dev, s.host_stage)
}

/// Drop `count` references from a slot; buffers persist for reuse.
#[cfg(feature = "cuda")]
fn arena_release_pack(slot: usize, count: usize) {
    let mut pool = ARENA_POOL.lock().unwrap();
    if let Some(s) = pool.get_mut(slot) {
        s.live = s.live.saturating_sub(count);
    }
}

/// Drop ONE reference (the defensive single-param `evict` path).
#[cfg(feature = "cuda")]
fn arena_release_ref(slot: usize) {
    arena_release_pack(slot, 1);
}

/// Free every slot's device + pinned-host buffer and clear the pool. Called
/// from teardown after all params are restored to owned buffers.
#[cfg(feature = "cuda")]
fn arena_teardown() {
    // Item 11: ensure no prefetch HtoD is still in flight into a slot before
    // its device buffer is freed (codegen awaits every pack before its
    // compute, so this is belt-and-suspenders for the async path).
    crate::cuda::inner::transfer_stream_synchronize();
    let mut pool = ARENA_POOL.lock().unwrap();
    for s in pool.drain(..) {
        crate::cuda::inner::ensure_context();
        crate::cuda::inner::free_managed(s.dev);
        if crate::cuda::inner::is_pinned(s.host_stage as *mut c_void) {
            crate::cuda::inner::free_pinned(s.host_stage as *mut c_void);
        } else {
            unsafe { crate::memory::checked_free(s.host_stage, s.cap) };
        }
    }
}

/// Upload a whole layer pack in ONE HtoD. `pw_list_ptr` is an `NslList` of the
/// pack's parameter tensor pointers (built by codegen from a CSLA layer
/// group). All must be registered and currently evicted. After this returns
/// each param's `t.data` points inside a shared, reused device slot (a
/// non-owning view) and the pack shares one slot for its whole residency.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_upload_pack(pw_list_ptr: i64) {
    #[cfg(feature = "cuda")]
    upload_pack_inner(pw_list_ptr, false);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = pw_list_ptr;
    }
}

/// Item 11: async sibling of `upload_pack`. Issues the pack's HtoD on the
/// transfer stream (overlapping current compute) and records a completion
/// event in the slot; `t.data` is set immediately but MUST NOT be read until
/// `nsl_weight_stream_await_pack` discharges the event onto the compute
/// stream. Prefetches only into a slot freed by a SYNCHRONOUS evict, so the
/// destination has no other pending writer.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_prefetch_pack(pw_list_ptr: i64) {
    #[cfg(feature = "cuda")]
    upload_pack_inner(pw_list_ptr, true);
    #[cfg(not(feature = "cuda"))]
    {
        let _ = pw_list_ptr;
    }
}

#[cfg(feature = "cuda")]
fn upload_pack_inner(pw_list_ptr: i64, prefetch: bool) {
    if pw_list_ptr == 0 {
        return;
    }
    let list = crate::list::NslList::from_ptr(pw_list_ptr);
    let n = list.len as usize;
    if n == 0 {
        return;
    }
    crate::cuda::inner::ensure_context();
    let mut guard = MIRRORS.lock().unwrap();
    let table = guard
        .as_mut()
        .expect("[weight-stream] upload_pack before any register");

    // Aligned offsets within the pack + total bytes.
    let mut layout: Vec<(i64, usize, usize)> = Vec::with_capacity(n); // (ptr, off, bytes)
    let mut total = 0usize;
    for i in 0..n {
        let ptr = unsafe { *list.data.add(i) };
        let Some(m) = table.get(&ptr) else {
            eprintln!("[weight-stream] FATAL: upload_pack of unregistered tensor {ptr}");
            std::process::abort();
        };
        let off = align_up(total, ARENA_ALIGN);
        layout.push((ptr, off, m.bytes));
        total = off + m.bytes;
    }

    let (slot_idx, dev, host_stage) = arena_acquire(total, n);

    // Gather each mirror into the contiguous pinned host buffer, then ONE HtoD
    // for the whole pack — synchronous for `upload_pack`, async (overlapping
    // compute) for `prefetch_pack`.
    for &(ptr, off, bytes) in &layout {
        let host = table.get(&ptr).unwrap().host;
        unsafe { std::ptr::copy_nonoverlapping(host, host_stage.add(off), bytes) };
    }
    if prefetch {
        let ev =
            crate::cuda::inner::prefetch_htod_on_transfer(dev, host_stage as *const c_void, total);
        arena_set_pending_event(slot_idx, ev);
        WS_PREFETCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    } else {
        crate::cuda::inner::memcpy_htod(dev, host_stage as *const c_void, total);
    }

    // Point each param at its arena offset (non-owning view).
    for &(ptr, off, _bytes) in &layout {
        let t = NslTensor::from_ptr(ptr);
        let interior = unsafe { (dev as *mut u8).add(off) } as *mut c_void;
        t.data = interior;
        // A view into the arena: a stray `nsl_tensor_free` must NOT free an
        // interior pointer. Restored to owned (=1) at teardown.
        t.owns_data = 0;
        let m = table.get_mut(&ptr).unwrap();
        let dev_i = interior as i64;
        if m.last_dev != 0 && m.last_dev != dev_i {
            WS_PTR_MOVES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        m.last_dev = dev_i;
        m.arena_slot = slot_idx as i64;
        m.arena_off = off;
        WS_UPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    WS_PACK_UPLOADS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Item 11: discharge a pack's pending prefetch event onto the compute stream
/// so every subsequent kernel that reads the pack is ordered after its HtoD.
/// A no-op if the pack was uploaded synchronously (no pending event) or is not
/// arena-resident. The pack's members share one slot; the event lives on it.
#[no_mangle]
pub extern "C" fn nsl_weight_stream_await_pack(pw_list_ptr: i64) {
    if pw_list_ptr == 0 {
        return;
    }
    #[cfg(feature = "cuda")]
    {
        let list = crate::list::NslList::from_ptr(pw_list_ptr);
        if list.len == 0 {
            return;
        }
        let first = unsafe { *list.data };
        let slot = {
            let guard = MIRRORS.lock().unwrap();
            match guard.as_ref().and_then(|g| g.get(&first)) {
                Some(m) if m.arena_slot >= 0 => m.arena_slot as usize,
                _ => return,
            }
        };
        let ev = arena_take_pending_event(slot);
        if ev != 0 {
            crate::cuda::inner::compute_stream_wait_event(ev);
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = pw_list_ptr;
    }
}

/// Record a slot's pending prefetch event.
#[cfg(feature = "cuda")]
fn arena_set_pending_event(slot: usize, ev: u64) {
    let mut pool = ARENA_POOL.lock().unwrap();
    if let Some(s) = pool.get_mut(slot) {
        s.pending_event = ev;
    }
}

/// Take (and clear) a slot's pending prefetch event.
#[cfg(feature = "cuda")]
fn arena_take_pending_event(slot: usize) -> u64 {
    let mut pool = ARENA_POOL.lock().unwrap();
    match pool.get_mut(slot) {
        Some(s) => std::mem::replace(&mut s.pending_event, 0),
        None => 0,
    }
}

/// Evict a whole layer pack in ONE DtoH (when `writeback != 0`). Params not
/// currently arena-resident are skipped (keeps the belt idempotent). All
/// resident members must share one slot (codegen invariant — they were
/// uploaded together).
#[no_mangle]
pub extern "C" fn nsl_weight_stream_evict_pack(pw_list_ptr: i64, writeback: i64) {
    if pw_list_ptr == 0 {
        return;
    }
    #[cfg(feature = "cuda")]
    {
        let list = crate::list::NslList::from_ptr(pw_list_ptr);
        let n = list.len as usize;
        if n == 0 {
            return;
        }
        crate::cuda::inner::ensure_context();
        let mut guard = MIRRORS.lock().unwrap();
        let table = guard
            .as_mut()
            .expect("[weight-stream] evict_pack before any register");

        let mut slot: i64 = -1;
        let mut regions: Vec<(i64, usize, usize)> = Vec::with_capacity(n); // (ptr, off, bytes)
        for i in 0..n {
            let ptr = unsafe { *list.data.add(i) };
            let Some(m) = table.get(&ptr) else {
                eprintln!("[weight-stream] FATAL: evict_pack of unregistered tensor {ptr}");
                std::process::abort();
            };
            if m.arena_slot < 0 {
                // Legitimately evicted (idempotent belt) → skip. But a member
                // that is RESIDENT via a non-arena owned buffer (data != null
                // while arena_slot < 0) is a codegen slip that would silently
                // leak it and skip its writeback — abort loudly instead
                // (review Item-10 finding B; matters now that --stream-prefetch
                // and the callback guard mint per-param owned buffers).
                if !NslTensor::from_ptr(ptr).data.is_null() {
                    eprintln!(
                        "[weight-stream] FATAL: evict_pack member {ptr} is resident via a \
                         non-arena buffer (arena_slot=-1, data!=null) — pack grouping mismatch"
                    );
                    std::process::abort();
                }
                continue;
            }
            if slot < 0 {
                slot = m.arena_slot;
            } else if slot != m.arena_slot {
                eprintln!(
                    "[weight-stream] FATAL: evict_pack members span slots {slot} and {} \
                     — pack upload/evict grouping mismatch",
                    m.arena_slot
                );
                std::process::abort();
            }
            regions.push((ptr, m.arena_off, m.bytes));
        }
        if regions.is_empty() {
            return;
        }
        let (dev, host_stage) = arena_slot_bufs(slot as usize);

        if writeback != 0 {
            // ONE DtoH of the occupied span, then scatter to per-param mirrors
            // (interior gaps between aligned regions are copied but ignored).
            let span = regions.iter().map(|&(_, o, b)| o + b).max().unwrap_or(0);
            crate::cuda::inner::memcpy_dtoh(host_stage as *mut c_void, dev as *const c_void, span);
            for &(ptr, off, bytes) in &regions {
                let host = table.get(&ptr).unwrap().host;
                unsafe { std::ptr::copy_nonoverlapping(host_stage.add(off), host, bytes) };
            }
        }
        for &(ptr, _off, _bytes) in &regions {
            let t = NslTensor::from_ptr(ptr);
            t.data = std::ptr::null_mut();
            let m = table.get_mut(&ptr).unwrap();
            m.arena_slot = -1;
            WS_EVICTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if writeback != 0 {
                WS_EVICTS_WB.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        arena_release_pack(slot as usize, regions.len());
        WS_PACK_EVICTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = writeback;
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
        guard.as_ref().is_some_and(|g| g.contains_key(&tensor_ptr)) as i64
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
            if m.arena_slot >= 0 {
                // Item 10: still resident in a shared arena slot (defensive —
                // the loop evicts every pack after its update, so this is rare).
                // Copy its bytes into a fresh OWNED device buffer (DtoD) and
                // restore the ownership flag so post-training code sees an
                // ordinary tensor. NOT counted in WS_UPLOADS (post-training).
                let (dev_base, _hs) = arena_slot_bufs(m.arena_slot as usize);
                let interior =
                    unsafe { (dev_base as *const u8).add(m.arena_off) } as *const c_void;
                let owned = crate::cuda::inner::alloc_managed(m.bytes);
                crate::cuda::inner::memcpy_dtod(owned, interior, m.bytes);
                t.data = owned;
                t.owns_data = 1;
            } else if t.data.is_null() {
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
                // Arena-mode evicts leave owns_data=0 (set at upload_pack);
                // restore it so the tensor owns its fresh buffer.
                t.owns_data = 1;
            }
            // Mirror free: pinned buffers go back to the driver, pageable
            // fallbacks to the heap (same routing as free_host_tensor_data).
            if crate::cuda::inner::is_pinned(m.host as *mut c_void) {
                crate::cuda::inner::free_pinned(m.host as *mut c_void);
                continue;
            }
            unsafe { crate::memory::checked_free(m.host, m.bytes) };
        }
        // Free the arena slots now that every param owns its buffer again.
        arena_teardown();
    }
}
