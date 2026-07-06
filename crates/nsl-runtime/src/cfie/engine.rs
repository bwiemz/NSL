//! CFIE decode-loop engine — kernel registry, KV pool, cached launches.
//!
//! CFIE Cycle 6 runtime half (ABI frozen 2026-07-01).  The codegen side
//! registers each compiled PTX kernel once at serve startup
//! (`nsl_cfie_register_kernel`), allocates the KV pool
//! (`nsl_cfie_kv_pool_alloc`), then finalizes
//! (`nsl_cfie_engine_finalize`) — which loads every module ONCE and
//! resolves every CUfunction ONCE.  The per-token launch FFIs and the
//! `nsl_cfie_decode_step` host loop then launch through the cached
//! handles without re-resolving anything (the existing
//! `cuda::inner::kernel_launch` re-runs `cuModuleGetFunction` on every
//! call, which is too slow for a decode loop).
//!
//! Kernel kinds: 0=decode_attn, 1=fused_sample, 2=decode_block,
//! 3=spec_verify, 4=spec_reject, 5=quant_attn, 6=draft decode_block,
//! 7=draft_sample, 8=verify_probs (6-8: CFIE Cycle 13, G15
//! draft-model-in-binary).  `layer_idx` is meaningful ONLY for kind 5
//! (the quant emitter bakes the layer into the PTX, so there is one
//! registration per layer); all other kinds register with layer_idx 0.
//!
//! Non-GPU builds are honest refusals: `finalize`/`kv_pool_alloc` print
//! one clear warning and return -1 — they never pretend success
//! (deferral-must-refuse invariant).  Registration and destruction are
//! pure CPU bookkeeping and work on every build so the lifecycle is
//! unit-testable without a GPU.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
use std::ffi::c_void;

use crate::cfie::ffi::kv_slots_global;

// Kernel kinds — frozen by the CFIE Cycle 6 ABI contract.
pub(crate) const KIND_DECODE_ATTN: i64 = 0;
pub(crate) const KIND_FUSED_SAMPLE: i64 = 1;
pub(crate) const KIND_DECODE_BLOCK: i64 = 2;
pub(crate) const KIND_SPEC_VERIFY: i64 = 3;
pub(crate) const KIND_SPEC_REJECT: i64 = 4;
pub(crate) const KIND_QUANT_ATTN: i64 = 5;
// CFIE Cycle 13 (G15 draft-model-in-binary) kinds.  Kind 6 is the
// EXISTING decode-block emitter (cfie_persistent_ptx.rs) instantiated
// with the DRAFT model's DecodeBlockConfig (draft dims,
// per_slot_max_tokens = draft pool capacity, max_slots = 1); kinds 7/8
// are the draft greedy sampler and the verification prob-row writer
// (cfie_spec_sampler_ptx.rs) — sampler-family kernels, no KV access.
pub(crate) const KIND_DRAFT_BLOCK: i64 = 6;
pub(crate) const KIND_DRAFT_SAMPLE: i64 = 7;
pub(crate) const KIND_VERIFY_PROBS: i64 = 8;

/// One registered kernel: an owned NUL-terminated PTX copy + launch
/// metadata, plus the driver handles cached at finalize time.
///
/// Non-cuda builds record registrations (pure CPU bookkeeping) but the
/// finalize/launch readers are compiled out, so the fields look dead
/// to that build's lint pass.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct KernelRegistration {
    /// PTX bytes copied out of caller memory with a runtime-appended
    /// trailing NUL (the ABI's `ptx_len` EXCLUDES any NUL).
    ptx: Vec<u8>,
    /// Kernel entry-point name (CString appends the NUL).
    name: CString,
    grid_x: u32,
    block_x: u32,
    /// DYNAMIC shared memory for the launch — 0 for all current CFIE
    /// kernels (they declare static .shared in-module); kept for future
    /// extern-.shared kernels.
    smem_dyn_bytes: u32,
    /// CUmodule / CUfunction as `usize` — opaque driver pointers are
    /// not `Send`, so they cross module boundaries as integer casts
    /// guarded by the ENGINE mutex (same pattern as
    /// `CudaState.module_cache`).  0 = not resolved yet.
    module: usize,
    func: usize,
}

struct EngineState {
    /// Registrations keyed by (kind, layer_idx); duplicate registration
    /// replaces the previous entry.
    kernels: HashMap<(i64, i64), KernelRegistration>,
    finalized: bool,
    /// Device address of the fused-sample module's
    /// `nsl_cfie_grammar_mask` global; 0 when the symbol is absent
    /// (grammar disabled — legal).
    grammar_mask_dev: u64,
    /// KV pool allocation record: device base from
    /// `nsl_cfie_kv_pool_alloc`, 0 when unallocated.
    pool_base: u64,
    /// Recorded for diagnostics/symmetry with the slot allocator's
    /// attach record; `free_managed` only needs the base.
    #[allow(dead_code)]
    pool_bytes: u64,
    /// Device addresses (as `usize`) of every weight buffer uploaded via
    /// `nsl_cfie_upload_weight_f16` / `_f32`.  Engine-owned, freed by
    /// `nsl_cfie_weights_reset` and by `nsl_cfie_engine_destroy` so a
    /// serve session leaves no device leak.  Non-cuda builds never push
    /// (the uploads refuse), so the vec is always empty there.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    weight_allocs: Vec<usize>,
    /// Bound-model state (CFIE Cycle 10), populated by
    /// `nsl_cfie_bind_model`.  `None` until a model is bound; cleared by
    /// `nsl_cfie_generate_reset` and `nsl_cfie_engine_destroy` WITHOUT
    /// freeing the device weight buffers (those are owned by
    /// `weight_allocs` / `nsl_cfie_weights_reset`).  Non-cuda builds never
    /// populate it (bind_model refuses), so it is always `None` there.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    bound_model: Option<BoundModel>,
    /// DRAFT-model binding (CFIE Cycle 13, G15 draft-model-in-binary),
    /// populated by `nsl_cfie_bind_draft_model` — the same record shape
    /// as the target binding (the draft is a full HF-Llama model with
    /// its own dims; only the vocab must match the target's).  Its
    /// device pointers alias entries in `weight_allocs` (uploaded via
    /// the same Cycle-9 FFIs), so clearing the binding
    /// (`nsl_cfie_draft_reset` / `nsl_cfie_engine_destroy`) drops the
    /// RECORDS only; the buffers are freed by `nsl_cfie_weights_reset`
    /// / `nsl_cfie_engine_destroy`.
    draft_model: Option<BoundModel>,
    /// DRAFT KV pool device base, allocated + zeroed by
    /// `nsl_cfie_draft_pool_alloc` in the PERSISTENT caching-allocator
    /// bracket.  This pool is NOT tracked by the target KV slot
    /// allocator — draft position bookkeeping is host-side in
    /// `nsl_cfie_speculative_generate`, and kind-6 launches inject this
    /// base as their `kv_base`.  0 = unallocated.  Freed by
    /// `nsl_cfie_draft_reset` and `nsl_cfie_engine_destroy`.
    draft_pool_base: u64,
    /// Recorded for diagnostics/symmetry with `pool_bytes`; the frees
    /// only need the base.
    #[allow(dead_code)]
    draft_pool_bytes: u64,
}

/// The runtime binding of an `NslModel` to the finalized CFIE engine —
/// the device weight table `nsl_cfie_generate` drives `decode_step` with,
/// plus the host-resident token-embedding table the per-step gather
/// reads.  Assembled by `nsl_cfie_bind_model`; the device pointers here
/// alias entries in `EngineState::weight_allocs` (the upload FFIs record
/// them for cleanup), so tearing the binding down only clears these
/// records — the device buffers are freed by `nsl_cfie_weights_reset`.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct BoundModel {
    /// `n_layers * 9` DEVICE pointers in the exact order
    /// `nsl_cfie_decode_step` reads each layer record:
    /// wq, wk, wv, wo, w_gate, w_up, w_down, norm1_w, norm2_w.
    weight_table: Vec<u64>,
    /// Final-norm gamma (f32) device pointer -> decode_step `norm_w_ptr`.
    final_norm_dev: u64,
    /// LM-head (f16) device pointer -> decode_step `lm_head_ptr`.
    lm_head_dev: u64,
    /// Token-embedding table kept on the HOST as f32, laid out
    /// `[vocab][d_model]` row-major.  The v1 embedding gather memcpys
    /// row `t` (d_model f32 = d_model*4 bytes) host->device into `x_a`
    /// before each decode_step — no kernel, no f16 convert (x_a is f32).
    embed_host: Vec<f32>,
    n_layers: i64,
    d_model: i64,
    vocab_size: i64,
}

static ENGINE: OnceLock<Mutex<EngineState>> = OnceLock::new();

fn engine() -> &'static Mutex<EngineState> {
    ENGINE.get_or_init(|| {
        Mutex::new(EngineState {
            kernels: HashMap::new(),
            finalized: false,
            grammar_mask_dev: 0,
            pool_base: 0,
            pool_bytes: 0,
            weight_allocs: Vec::new(),
            bound_model: None,
            draft_model: None,
            draft_pool_base: 0,
            draft_pool_bytes: 0,
        })
    })
}

// ---------------------------------------------------------------------------
// Registration / lifecycle FFIs
// ---------------------------------------------------------------------------

/// Register one compiled CFIE kernel.  The runtime COPIES the PTX bytes
/// `[ptx_ptr, ptx_ptr + ptx_len)` and the name bytes, appending its own
/// trailing NUL to each (`ptx_len`/`name_len` exclude any NUL).
/// Duplicate `(kind, layer_idx)` replaces the previous registration and
/// returns 0.  Returns 0 ok; -1 invalid (bad kind, null/zero ptx or
/// name, non-positive grid/block).
///
/// # Safety
/// `ptx_ptr`/`name_ptr` must point to at least `ptx_len`/`name_len`
/// readable bytes; the compiled serve prologue passes data-section
/// addresses.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_cfie_register_kernel(
    kind: i64,
    layer_idx: i64,
    ptx_ptr: i64,
    ptx_len: i64,
    name_ptr: i64,
    name_len: i64,
    grid_x: i64,
    block_x: i64,
    smem_dyn_bytes: i64,
) -> i64 {
    match kind {
        KIND_DECODE_ATTN | KIND_FUSED_SAMPLE | KIND_DECODE_BLOCK | KIND_SPEC_VERIFY
        | KIND_SPEC_REJECT | KIND_QUANT_ATTN | KIND_DRAFT_BLOCK | KIND_DRAFT_SAMPLE
        | KIND_VERIFY_PROBS => {}
        _ => return -1,
    }
    if layer_idx < 0 {
        return -1;
    }
    if ptx_ptr == 0 || ptx_len <= 0 || name_ptr == 0 || name_len <= 0 {
        return -1;
    }
    if grid_x <= 0 || block_x <= 0 || smem_dyn_bytes < 0 {
        return -1;
    }
    if grid_x > u32::MAX as i64 || block_x > u32::MAX as i64 || smem_dyn_bytes > u32::MAX as i64 {
        return -1;
    }
    // Copy PTX out of caller memory, appending our own trailing NUL so
    // cuModuleLoadData sees a C string regardless of the caller's
    // buffer lifetime.
    let ptx = unsafe {
        let src = std::slice::from_raw_parts(ptx_ptr as *const u8, ptx_len as usize);
        let mut v = Vec::with_capacity(src.len() + 1);
        v.extend_from_slice(src);
        v.push(0);
        v
    };
    let name_bytes =
        unsafe { std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize) };
    // CString::new appends the NUL and refuses interior NULs (an
    // interior NUL would silently truncate the cuModuleGetFunction
    // lookup — refuse instead).
    let name = match CString::new(name_bytes) {
        Ok(c) => c,
        Err(_) => return -1,
    };
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    g.kernels.insert(
        (kind, layer_idx),
        KernelRegistration {
            ptx,
            name,
            grid_x: grid_x as u32,
            block_x: block_x as u32,
            smem_dyn_bytes: smem_dyn_bytes as u32,
            module: 0,
            func: 0,
        },
    );
    // A registration landing after finalize would otherwise never be
    // resolved (the idempotent second finalize only counts) while
    // finalize keeps reporting success — force a re-finalize instead.
    g.finalized = false;
    0
}

/// Allocate the KV device pool.  Returns 0 ok; -1 when there is no
/// CUDA build/GPU, `bytes <= 0`, the KV slot allocator is
/// uninitialized, or a pool is already allocated (destroy first).
///
/// The pool lives for the whole serve session, so it is allocated in
/// the PERSISTENT caching-allocator pool (same bracket
/// `nsl_gpu_set_persistent_pool` uses for model weights — per-step
/// transient drains never release it), zeroed with memset_d8(0), and
/// recorded via `KvSlotAllocator::attach_device_buffer`.
#[no_mangle]
pub extern "C" fn nsl_cfie_kv_pool_alloc(bytes: i64) -> i64 {
    #[cfg(feature = "cuda")]
    {
        if bytes <= 0 {
            return -1;
        }
        let mut g = match engine().lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        if g.pool_base != 0 {
            return -1; // already allocated — nsl_cfie_engine_destroy first
        }
        // KV slots must be initialized before the pool exists so the
        // base has somewhere to attach.
        {
            let kv = match kv_slots_global().lock() {
                Ok(k) => k,
                Err(_) => return -1,
            };
            if kv.is_none() {
                return -1;
            }
        }
        use crate::cuda::caching_allocator::{get_alloc_pool, set_alloc_pool, AllocPool};
        let prev = get_alloc_pool();
        set_alloc_pool(AllocPool::Persistent);
        let ptr = crate::cuda::inner::alloc_managed(bytes as usize);
        set_alloc_pool(prev);
        if ptr.is_null() {
            return -1;
        }
        crate::cuda::inner::memset_d8(ptr, bytes as usize);
        {
            let mut kv = match kv_slots_global().lock() {
                Ok(k) => k,
                Err(_) => {
                    crate::cuda::inner::free_managed(ptr);
                    return -1;
                }
            };
            match kv.as_mut() {
                Some(a) => a.attach_device_buffer(ptr as u64, bytes as u64),
                None => {
                    // Slot allocator was torn down between the check and
                    // the attach — refuse rather than leak or dangle.
                    drop(kv);
                    crate::cuda::inner::free_managed(ptr);
                    return -1;
                }
            }
        }
        g.pool_base = ptr as u64;
        g.pool_bytes = bytes as u64;
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = bytes;
        eprintln!("CFIE: kv_pool_alloc requires a CUDA-enabled build and GPU — refusing (no pool allocated)");
        -1
    }
}

/// Finalize the engine: load each registered kernel's module ONCE
/// (`cuModuleLoadData` on the NUL-terminated copy) and resolve its
/// CUfunction ONCE, caching the handles for the launch FFIs.  For the
/// kind-1 (fused_sample) module, additionally resolve the optional
/// `nsl_cfie_grammar_mask` global (absent => 0, grammar disabled).
/// Returns the count (>= 0) of kernels resolved, or -1 (no CUDA
/// build/GPU, zero registrations, or a driver load/lookup failure —
/// diagnosed on stderr).
#[no_mangle]
pub extern "C" fn nsl_cfie_engine_finalize() -> i64 {
    #[cfg(feature = "cuda")]
    {
        let mut g = match engine().lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        if g.kernels.is_empty() {
            eprintln!("CFIE: engine_finalize called with zero kernel registrations — refusing");
            return -1;
        }
        if g.finalized {
            // Idempotent second call: everything is already resolved.
            return g.kernels.values().filter(|k| k.func != 0).count() as i64;
        }
        crate::cuda::inner::ensure_context();
        let keys: Vec<(i64, i64)> = g.kernels.keys().copied().collect();
        for key in keys {
            let (module, func) = {
                let k = &g.kernels[&key];
                let module = match crate::cuda::inner::load_module_once(&k.ptx) {
                    Ok(m) => m,
                    Err(code) => {
                        eprintln!(
                            "CFIE: cuModuleLoadData failed for kernel (kind {}, layer {}): CUresult {}",
                            key.0, key.1, code
                        );
                        return -1;
                    }
                };
                let func = match crate::cuda::inner::get_function(module, &k.name) {
                    Ok(f) => f,
                    Err(code) => {
                        eprintln!(
                            "CFIE: cuModuleGetFunction('{}') failed for kernel (kind {}, layer {}): CUresult {}",
                            k.name.to_string_lossy(),
                            key.0,
                            key.1,
                            code
                        );
                        return -1;
                    }
                };
                (module, func)
            };
            let k = g.kernels.get_mut(&key).expect("key came from the map");
            k.module = module;
            k.func = func;
        }
        // Grammar mask: module-scope global in the fused_sample module.
        // Absent (CUDA_ERROR_NOT_FOUND) is legal — grammar disabled.
        g.grammar_mask_dev = 0;
        if let Some(k) = g.kernels.get(&(KIND_FUSED_SAMPLE, 0)) {
            let name = c"nsl_cfie_grammar_mask";
            match crate::cuda::inner::module_get_global(k.module, name) {
                Ok((addr, _bytes)) => g.grammar_mask_dev = addr,
                Err(code)
                    if code == cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_FOUND as u32 => {}
                Err(code) => {
                    // A real driver error here must NOT degrade to
                    // "mask absent": the sampler would silently run
                    // UNCONSTRAINED (mask ptr 0 = grammar disabled).
                    // Refuse finalize like the load/lookup failures.
                    eprintln!(
                        "CFIE: cuModuleGetGlobal('nsl_cfie_grammar_mask') failed: CUresult {} — refusing finalize",
                        code
                    );
                    return -1;
                }
            }
        }
        g.finalized = true;
        g.kernels.len() as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CFIE: engine_finalize requires a CUDA-enabled build and GPU — refusing (no kernels resolved)");
        -1
    }
}

/// Tear the engine down.  Frees the pool allocation if any, detaches
/// the slot allocator's device buffer (`attach_device_buffer(0, 0)`),
/// and clears registrations and cached handles so a fresh
/// register→finalize cycle can run.  CUmodules are left loaded
/// (leak-by-design precedent: `cuda/mod.rs` `module_cache`).  Always
/// returns 0.
#[no_mangle]
pub extern "C" fn nsl_cfie_engine_destroy() -> i64 {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(), // still tear down after a poisoned panic
    };
    #[cfg(feature = "cuda")]
    {
        if g.pool_base != 0 {
            crate::cuda::inner::free_managed(g.pool_base as *mut std::ffi::c_void);
        }
        // The DRAFT KV pool is engine-held (no slot-allocator attach) —
        // free it here so a serve session leaves no device leak.
        if g.draft_pool_base != 0 {
            crate::cuda::inner::free_managed(g.draft_pool_base as *mut std::ffi::c_void);
        }
        // Weights are engine-owned device resources, same as the KV pool
        // and module handles — a serve session must leave no device leak.
        free_weight_allocs(&mut g);
    }
    #[cfg(not(feature = "cuda"))]
    {
        g.weight_allocs.clear();
    }
    // Clear the bound-model records too (device buffers already freed
    // above via free_weight_allocs / weight_allocs.clear()).  Same for
    // the draft binding — its device pointers were in weight_allocs.
    g.bound_model = None;
    g.draft_model = None;
    g.pool_base = 0;
    g.pool_bytes = 0;
    g.draft_pool_base = 0;
    g.draft_pool_bytes = 0;
    if let Ok(mut kv) = kv_slots_global().lock() {
        if let Some(a) = kv.as_mut() {
            a.attach_device_buffer(0, 0);
        }
    }
    g.kernels.clear();
    g.finalized = false;
    g.grammar_mask_dev = 0;
    0
}

/// Test/diagnostic accessor: the KV pool device base recorded by
/// `nsl_cfie_kv_pool_alloc`, or 0 when no pool is allocated.  The GPU
/// parity tests seed K/V rows by uploading through this base; the
/// production launch path never needs it (the launch FFIs inject the
/// base from the slot allocator's attach record).
#[doc(hidden)]
#[no_mangle]
pub extern "C" fn nsl_cfie_kv_pool_base() -> i64 {
    match engine().lock() {
        Ok(g) => g.pool_base as i64,
        Err(_) => 0,
    }
}

// ---------------------------------------------------------------------------
// Weight binding (CFIE Cycle 9)
//
// PRODUCTION upload primitives: put model weights on the GPU in the
// EXACT layout `nsl_cfie_decode_step` already consumes.  The caller lays
// each weight out [out][in] row-major on the host; the upload copies it
// element-for-element, so the device buffer is f16 (or f32) [out][in]
// row-major — exactly what the decode-block kernel `ld.global.b16`
// addressing expects (cfie_persistent_ptx.rs).  This cycle ships the
// upload/lifecycle primitives + the GPU proof; resolving weights from a
// loaded NslModel by name is Cycle 10.
// ---------------------------------------------------------------------------

/// Round-to-nearest-even f32 -> f16 (IEEE 754 half), returned as raw
/// bits.  `half` is already a nsl-runtime dependency, so delegate to its
/// audited converter (correct RNE including subnormals, overflow to inf,
/// and NaN) rather than hand-rolling and risking a rounding-edge bug.
#[cfg(feature = "cuda")]
#[inline]
fn f32_to_f16_bits(x: f32) -> u16 {
    half::f16::from_f32(x).to_bits()
}

/// Free every recorded weight allocation and clear the tracking list.
/// Operates on an already-locked engine guard so `nsl_cfie_weights_reset`
/// and `nsl_cfie_engine_destroy` share one code path.  Idempotent.
#[cfg(feature = "cuda")]
fn free_weight_allocs(g: &mut EngineState) {
    for ptr in g.weight_allocs.drain(..) {
        crate::cuda::inner::free_managed(ptr as *mut c_void);
    }
}

/// Allocate `bytes` in the PERSISTENT caching-allocator pool (the same
/// bracket the KV pool uses — per-step transient drains never release
/// it).  Returns a null pointer only for a zero request (`alloc_managed`
/// panics on real OOM, matching `nsl_cfie_kv_pool_alloc`).
#[cfg(feature = "cuda")]
fn alloc_persistent(bytes: usize) -> *mut c_void {
    use crate::cuda::caching_allocator::{get_alloc_pool, set_alloc_pool, AllocPool};
    let prev = get_alloc_pool();
    set_alloc_pool(AllocPool::Persistent);
    let ptr = crate::cuda::inner::alloc_managed(bytes);
    set_alloc_pool(prev);
    ptr
}

/// Byte count for a weight upload of `n_elems` f32 read from host.
///
/// The upload reads `n_elems * size_of::<f32>()` host bytes via
/// `slice::from_raw_parts`/`memcpy`, which requires that total to be
/// `<= isize::MAX` — exceeding it is undefined behaviour (a process
/// abort), NOT a recoverable error.  Both upload FFIs must therefore
/// guard the HOST read size, not just their (never-larger) device
/// buffer: the f16 device buffer is `n_elems * 2` and the f32 one is
/// `n_elems * 4 == host_bytes`, so a single host-bytes guard bounds both.
/// Returns the host byte count, or `None` to refuse (`n_elems <= 0` or
/// the read would exceed `isize::MAX`).
#[cfg(feature = "cuda")]
fn checked_host_f32_bytes(n_elems: i64) -> Option<usize> {
    if n_elems <= 0 {
        return None;
    }
    (n_elems as u64)
        .checked_mul(std::mem::size_of::<f32>() as u64)
        .filter(|&b| b <= isize::MAX as u64)
        .map(|b| b as usize)
}

/// Upload `n_elems` f32 read from host memory at `host_f32_ptr` to the
/// GPU as f16, round-to-nearest-even (via `half`).  Allocates
/// `n_elems * 2` device bytes in the PERSISTENT pool, uploads the f16
/// bytes, records the allocation for cleanup, and returns the device
/// pointer as `i64` (> 0).
///
/// Layout is preserved verbatim: the caller lays weights out [out][in]
/// row-major on the host, this copies element-for-element, so the device
/// buffer is f16 [out][in] row-major — exactly what the decode-block
/// kernel's `ld.global.b16` addressing expects.
///
/// Returns -1 on: non-cuda build, `host_f32_ptr == 0`, `n_elems <= 0`,
/// host read size (`n_elems * 4`) exceeding `isize::MAX`, or any driver
/// failure (freeing the partial device allocation first).
///
/// # Safety
/// `host_f32_ptr` must point to at least `n_elems` readable `f32`s.
#[no_mangle]
pub extern "C" fn nsl_cfie_upload_weight_f16(host_f32_ptr: i64, n_elems: i64) -> i64 {
    #[cfg(feature = "cuda")]
    {
        if host_f32_ptr == 0 {
            return -1;
        }
        // Guard the HOST read size (n_elems * 4) against isize::MAX — the
        // slice below is UB (process abort) past that, so an overflowing
        // n_elems must refuse, not abort.  Device bytes (f16 = n_elems*2)
        // are half the host bytes and can't overflow once this passes.
        let host_bytes = match checked_host_f32_bytes(n_elems) {
            Some(b) => b,
            None => return -1,
        };
        let dev_bytes = host_bytes / 2; // f16 is 2 B/elem; host f32 is 4 B/elem
        // Cast host f32 -> f16 bits with round-to-nearest-even.  Read the
        // host f32s into an owned buffer we control (the caller's memory
        // may be reclaimed after this call returns).
        let src = unsafe { std::slice::from_raw_parts(host_f32_ptr as *const f32, n_elems as usize) };
        let bits: Vec<u16> = src.iter().map(|&x| f32_to_f16_bits(x)).collect();

        let mut g = match engine().lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        let ptr = alloc_persistent(dev_bytes);
        if ptr.is_null() {
            return -1;
        }
        // htod copy of the f16 bytes.  On a driver failure `memcpy_htod`
        // panics (house contract, same as the KV pool's memset) — guard
        // it so the partial allocation is freed before the panic
        // unwinds, honoring the "free partial alloc first" contract.
        let copy = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::cuda::inner::memcpy_htod(
                ptr,
                bits.as_ptr() as *const c_void,
                dev_bytes,
            );
        }));
        if copy.is_err() {
            crate::cuda::inner::free_managed(ptr);
            return -1;
        }
        g.weight_allocs.push(ptr as usize);
        ptr as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (host_f32_ptr, n_elems);
        eprintln!("CFIE: upload_weight_f16 requires a CUDA-enabled build and GPU — refusing (no upload)");
        -1
    }
}

/// Upload `n_elems` f32 read from host memory at `host_f32_ptr` to the
/// GPU, KEEPING f32 (`n_elems * 4` device bytes, straight htod copy).
/// For the RMSNorm gammas (norm1_w, norm2_w, final norm) which the
/// decode/sample kernels read as f32.  Records the allocation for
/// cleanup and returns the device pointer as `i64` (> 0).  Same error
/// contract as `nsl_cfie_upload_weight_f16`: refuses (-1) when the host
/// read size (`n_elems * 4`) would exceed `isize::MAX`.
///
/// # Safety
/// `host_f32_ptr` must point to at least `n_elems` readable `f32`s.
#[no_mangle]
pub extern "C" fn nsl_cfie_upload_weight_f32(host_f32_ptr: i64, n_elems: i64) -> i64 {
    #[cfg(feature = "cuda")]
    {
        if host_f32_ptr == 0 {
            return -1;
        }
        // f32 device bytes == host read bytes (n_elems * 4); the shared
        // guard bounds both against isize::MAX so a huge n_elems refuses
        // (-1) instead of panicking in alloc/memcpy.
        let dev_bytes = match checked_host_f32_bytes(n_elems) {
            Some(b) => b,
            None => return -1,
        };
        let mut g = match engine().lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        let ptr = alloc_persistent(dev_bytes);
        if ptr.is_null() {
            return -1;
        }
        let copy = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::cuda::inner::memcpy_htod(
                ptr,
                host_f32_ptr as *const c_void,
                dev_bytes,
            );
        }));
        if copy.is_err() {
            crate::cuda::inner::free_managed(ptr);
            return -1;
        }
        g.weight_allocs.push(ptr as usize);
        ptr as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (host_f32_ptr, n_elems);
        eprintln!("CFIE: upload_weight_f32 requires a CUDA-enabled build and GPU — refusing (no upload)");
        -1
    }
}

/// Free every weight allocation recorded by the two upload FFIs (via
/// `free_managed`) and clear the tracking list.  Returns 0 always
/// (idempotent; safe with no CUDA / nothing uploaded).  Does NOT touch
/// the KV pool or kernel registrations.
#[no_mangle]
pub extern "C" fn nsl_cfie_weights_reset() -> i64 {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    #[cfg(feature = "cuda")]
    {
        free_weight_allocs(&mut g);
    }
    #[cfg(not(feature = "cuda"))]
    {
        // No cuda: no device allocations were ever recorded, but keep the
        // vec cleared for symmetry (always empty here).
        g.weight_allocs.clear();
    }
    0
}

// ---------------------------------------------------------------------------
// Model binding + generation driver (CFIE Cycle 10)
//
// bind_model resolves an NslModel's host f32 weights BY NAME (HF-Llama
// convention), uploads each to the device in the layout decode_step
// consumes (attention/FFN/lm_head as f16, the RMSNorm gammas as f32),
// keeps the token-embedding table on the HOST as f32, and records the
// device weight table + shape in engine state.  generate then drives the
// single decode loop: gather embed row -> x_a, decode_step, sample,
// prefill-boundary + EOS + capacity handling.
// ---------------------------------------------------------------------------

/// HF-Llama weight naming convention (documented ABI).  bind_model
/// resolves EXACTLY these names; a real NslModel using different names is
/// refused with the expected-vs-missing list rather than guessed at.
#[cfg(feature = "cuda")]
mod names {
    /// Per-layer attention/FFN projections + the two RMSNorm gammas.
    /// `{i}` is the layer index.
    pub const WQ: &str = "model.layers.{i}.self_attn.q_proj.weight";
    pub const WK: &str = "model.layers.{i}.self_attn.k_proj.weight";
    pub const WV: &str = "model.layers.{i}.self_attn.v_proj.weight";
    pub const WO: &str = "model.layers.{i}.self_attn.o_proj.weight";
    pub const W_GATE: &str = "model.layers.{i}.mlp.gate_proj.weight";
    pub const W_UP: &str = "model.layers.{i}.mlp.up_proj.weight";
    pub const W_DOWN: &str = "model.layers.{i}.mlp.down_proj.weight";
    pub const NORM1: &str = "model.layers.{i}.input_layernorm.weight";
    pub const NORM2: &str = "model.layers.{i}.post_attention_layernorm.weight";
    /// Model-level: final norm, LM head, token embedding.
    pub const FINAL_NORM: &str = "model.norm.weight";
    pub const LM_HEAD: &str = "lm_head.weight";
    pub const EMBED: &str = "model.embed_tokens.weight";

    /// Substitute the layer index into a `{i}` template name.
    pub fn layer(name: &str, i: i64) -> String {
        name.replace("{i}", &i.to_string())
    }
}

/// Read a bound weight tensor's host f32 data pointer + element count by
/// name from an NslModel handle.  Returns `Err(name)` (the resolved
/// name) when the weight is absent or its `data`/`len` is unusable so the
/// caller can list the first missing tensor.  The model's loaded weights
/// are host f32 NslTensors (dtype 1, device 0), matching the upload FFIs'
/// `host_f32_ptr` contract.
#[cfg(feature = "cuda")]
fn resolve_weight(model_handle: i64, name: &str) -> Result<(*const f32, i64), String> {
    let tptr = crate::c_api::nsl_model_get_weight(
        model_handle,
        name.as_ptr() as i64,
        name.len() as i64,
    );
    if tptr == 0 {
        return Err(name.to_string());
    }
    // The tensor lives in this crate; read its fields directly.  Loaded
    // safetensors weights are contiguous host f32.
    let t = unsafe { &*(tptr as *const crate::tensor::NslTensor) };
    if t.data.is_null() || t.len <= 0 {
        return Err(name.to_string());
    }
    Ok((t.data as *const f32, t.len))
}

/// Resolve a weight, verify its element count equals `expect_elems`, then
/// upload it to the device via `uploader` (f16 or f32).  On a shape
/// mismatch or a failed upload the FIRST offending name is returned so
/// bind_model can refuse with it (deferral-must-refuse: never fabricate a
/// weight).  Returns the device pointer on success.
#[cfg(feature = "cuda")]
fn upload_named(
    model_handle: i64,
    name: &str,
    expect_elems: i64,
    uploader: extern "C" fn(i64, i64) -> i64,
) -> Result<u64, String> {
    let (ptr, len) = resolve_weight(model_handle, name)?;
    if len != expect_elems {
        return Err(format!(
            "{name} (shape mismatch: has {len} elems, expected {expect_elems})"
        ));
    }
    let dev = uploader(ptr as i64, len);
    if dev <= 0 {
        return Err(format!("{name} (device upload failed)"));
    }
    Ok(dev as u64)
}

/// Resolve + upload one full HF-Llama model (target OR draft) into a
/// `BoundModel` record: every weight resolved by name from the
/// `NslModel` handle, shape-verified against the passed dims, and
/// uploaded in the layout the decode-block/sampler kernels consume
/// (q/k/v/o + gate/up/down + lm_head as f16; RMSNorm gammas + final
/// norm as f32; the token-embedding table kept HOST-resident as f32).
/// Returns `Err(first offending name)` on any resolution/shape/upload
/// failure — the CALLER owns cleanup of whatever partial uploads landed
/// in `weight_allocs` (bind_model resets ALL weights; bind_draft_model
/// frees only the uploads recorded after its snapshot).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn resolve_and_upload_model(
    model_handle: i64,
    n_layers: i64,
    d_model: i64,
    n_heads: i64,
    n_kv_heads: i64,
    head_dim: i64,
    d_ff: i64,
    vocab_size: i64,
) -> Result<BoundModel, String> {
    // Per-weight expected element counts (from the [out][in] shapes).
    let nhd = n_heads * head_dim; // q/attention width
    let nkvd = n_kv_heads * head_dim; // k/v width
    let attn_in = d_model; // projections read the residual stream
    let wq_elems = nhd * attn_in;
    let wkv_elems = nkvd * attn_in;
    let wo_elems = d_model * nhd;
    let gate_up_elems = d_ff * d_model;
    let down_elems = d_model * d_ff;
    let norm_elems = d_model;
    let lm_head_elems = vocab_size * d_model;
    let embed_elems = vocab_size * d_model;

    let mut weight_table = Vec::with_capacity((n_layers * 9) as usize);
    for i in 0..n_layers {
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::WQ, i),
            wq_elems,
            nsl_cfie_upload_weight_f16,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::WK, i),
            wkv_elems,
            nsl_cfie_upload_weight_f16,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::WV, i),
            wkv_elems,
            nsl_cfie_upload_weight_f16,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::WO, i),
            wo_elems,
            nsl_cfie_upload_weight_f16,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::W_GATE, i),
            gate_up_elems,
            nsl_cfie_upload_weight_f16,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::W_UP, i),
            gate_up_elems,
            nsl_cfie_upload_weight_f16,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::W_DOWN, i),
            down_elems,
            nsl_cfie_upload_weight_f16,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::NORM1, i),
            norm_elems,
            nsl_cfie_upload_weight_f32,
        )?);
        weight_table.push(upload_named(
            model_handle,
            &names::layer(names::NORM2, i),
            norm_elems,
            nsl_cfie_upload_weight_f32,
        )?);
    }
    let final_norm_dev = upload_named(
        model_handle,
        names::FINAL_NORM,
        norm_elems,
        nsl_cfie_upload_weight_f32,
    )?;
    let lm_head_dev = upload_named(
        model_handle,
        names::LM_HEAD,
        lm_head_elems,
        nsl_cfie_upload_weight_f16,
    )?;
    // Embedding table stays HOST-resident as f32 (v1 gather is a
    // per-step host->device row memcpy — no upload, no f16 convert).
    // Resolve + shape-check, then copy into an owned Vec the engine
    // keeps for the session.
    let (embed_ptr, embed_len) = resolve_weight(model_handle, names::EMBED)?;
    if embed_len != embed_elems {
        return Err(format!(
            "{} (shape mismatch: has {embed_len} elems, expected {embed_elems})",
            names::EMBED
        ));
    }
    let embed_host = unsafe { std::slice::from_raw_parts(embed_ptr, embed_len as usize) }.to_vec();

    Ok(BoundModel {
        weight_table,
        final_norm_dev,
        lm_head_dev,
        embed_host,
        n_layers,
        d_model,
        vocab_size,
    })
}

/// Bind a loaded `NslModel` to the finalized CFIE engine: resolve every
/// weight by the HF-Llama naming convention, upload it in the layout
/// `nsl_cfie_decode_step` consumes (q/k/v/o + gate/up/down + lm_head as
/// f16; the RMSNorm gammas + final norm as f32), keep the token-embedding
/// table host-resident as f32, and record the device weight table +
/// shape in engine state.  `nsl_cfie_generate` then drives decode_step
/// over a prompt with this binding.
///
/// Weight names (HF-Llama; `{i}` = layer index):
///   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight  (f16)
///   model.layers.{i}.mlp.{gate,up,down}_proj.weight    (f16)
///   model.layers.{i}.input_layernorm.weight            (f32)
///   model.layers.{i}.post_attention_layernorm.weight   (f32)
///   model.norm.weight                                  (f32)
///   lm_head.weight                                     (f16)
///   model.embed_tokens.weight                          (host f32)
///
/// Shapes verified against the passed dims (weights are `[out][in]`
/// row-major, matching the PyTorch/safetensors `nn.Linear` convention the
/// decode-block kernel's `ld.global.b16` addressing expects):
///   q_proj  [n_heads*head_dim][d_model]
///   k/v_proj [n_kv_heads*head_dim][d_model]
///   o_proj  [d_model][n_heads*head_dim]
///   gate/up [d_ff][d_model], down [d_model][d_ff]
///   norms   [d_model], final norm [d_model]
///   lm_head [vocab][d_model], embed [vocab][d_model]
///
/// Returns 0 on success; -1 (with an eprintln LISTING the first
/// missing/mis-shaped tensor) on any resolution/shape/upload failure or
/// on a non-cuda build.  A partial upload IS torn down on failure (the
/// uploads are engine-tracked, and this refusal clears the whole binding
/// via `nsl_cfie_weights_reset` so no half-bound state survives).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_cfie_bind_model(
    model_handle: i64,
    n_layers: i64,
    d_model: i64,
    n_heads: i64,
    n_kv_heads: i64,
    head_dim: i64,
    d_ff: i64,
    vocab_size: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        if model_handle == 0 {
            eprintln!("CFIE: bind_model refused — null model handle");
            return -1;
        }
        if n_layers <= 0
            || d_model <= 0
            || n_heads <= 0
            || n_kv_heads <= 0
            || head_dim <= 0
            || d_ff <= 0
            || vocab_size <= 0
        {
            eprintln!("CFIE: bind_model refused — non-positive dimension in the model shape");
            return -1;
        }

        // Assemble the binding into locals first; only on a fully clean
        // resolve do we commit it to engine state.  On ANY failure we
        // reset the (partially uploaded) weights so no half-bound state
        // survives.
        match resolve_and_upload_model(
            model_handle,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            d_ff,
            vocab_size,
        ) {
            Ok(bm) => {
                let mut g = match engine().lock() {
                    Ok(g) => g,
                    Err(_) => {
                        // Lock poisoned after uploads landed — free them
                        // so the refusal leaves no device leak.  The reset
                        // frees the DRAFT weights too, so both binding
                        // records must go (dangling-pointer hazard).
                        nsl_cfie_weights_reset();
                        let mut g = engine().lock().unwrap_or_else(|e| e.into_inner());
                        g.bound_model = None;
                        g.draft_model = None;
                        eprintln!("CFIE: bind_model refused — engine lock poisoned");
                        return -1;
                    }
                };
                // A re-bind replaces any prior binding; the old device
                // buffers were freed by the caller's weights_reset (Cycle
                // 11 rebinds via bind_model after weights_reset) or are
                // superseded here — we only overwrite the records.
                g.bound_model = Some(bm);
                0
            }
            Err(missing) => {
                // Deferral-must-refuse: a missing/mis-shaped weight must
                // refuse, never fabricate.  Free whatever partial uploads
                // landed so the refusal leaves no device leak, and clear
                // any stale binding.  weights_reset frees EVERY recorded
                // device weight — including a bound draft model's — so
                // the draft records must be cleared too or they dangle.
                nsl_cfie_weights_reset();
                {
                    let mut g = engine().lock().unwrap_or_else(|e| e.into_inner());
                    g.bound_model = None;
                    g.draft_model = None;
                }
                eprintln!(
                    "CFIE: bind_model refused — missing or mis-shaped weight: {missing}"
                );
                -1
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            model_handle,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            d_ff,
            vocab_size,
        );
        eprintln!("CFIE: bind_model requires a CUDA-enabled build and GPU — refusing (no binding)");
        -1
    }
}

/// Drive the CFIE decode loop over `prompt_tokens` + up to
/// `max_new_tokens` generated tokens, writing the GENERATED token ids to
/// `out_tokens` (host array of capacity `out_cap` i64).  Requires
/// `nsl_cfie_bind_model` + `nsl_cfie_engine_finalize` to have run.
///
/// The loop (see the ABI): acquire a KV slot; for pos in
/// 0..(prompt_len + max_new_tokens), gather the embedding row of the
/// current input token (prompt token while prefilling, else the last
/// sampled token) into `x_a`, run `nsl_cfie_decode_step`, read the
/// sampled token, and — from the LAST prefill step onward (`pos >=
/// prompt_len - 1`, which already produces the first NEW token) — record
/// it, stopping on EOS or `max_new_tokens`.  Cleanly stops (breaks) on a
/// `-2` KV-capacity refusal; a real failure (`rc < 0`, `rc != -2`)
/// releases the slot and returns -1.
///
/// grammar_state stays 0 in v1 (the sampler supports a grammar mask when
/// grammar_states>0, but wiring the per-step DFA transition through
/// generate is a later cycle — DEFERRED, documented here).
///
/// Return value: the TRUE generated-token count (may exceed `out_cap`).
/// Writes are CLAMPED to `out_cap` (tokens past the capacity are counted
/// but not written — no buffer overrun); the returned count is the true
/// number generated so the caller can detect truncation.  Returns -1 on a
/// bad argument, missing binding/finalize, slot-acquire failure, or a
/// real decode failure; on a non-cuda build.
///
/// # Safety
/// `prompt_tokens_ptr` must point to `prompt_len` readable i64s;
/// `out_tokens_ptr` to `out_cap` writable i64s.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_cfie_generate(
    prompt_tokens_ptr: i64,
    prompt_len: i64,
    max_new_tokens: i64,
    eos_token_id: i64,
    rng_seed: i64,
    out_tokens_ptr: i64,
    out_cap: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        // Argument validation up front (before any device work).
        if prompt_tokens_ptr == 0
            || out_tokens_ptr == 0
            || prompt_len <= 0
            || out_cap <= 0
            || max_new_tokens <= 0
        {
            eprintln!("CFIE: generate refused — bad argument (null ptr or non-positive length/cap)");
            return -1;
        }

        // Snapshot the binding + shape under the engine lock (the device
        // pointers/host embed are stable for the call; we release the lock
        // before the decode loop so decode_step's own engine locking is
        // not re-entrant).
        let (weight_table, final_norm_dev, lm_head_dev, embed_host, n_layers, d_model, vocab_size) = {
            let g = match engine().lock() {
                Ok(g) => g,
                Err(_) => {
                    eprintln!("CFIE: generate refused — engine lock poisoned");
                    return -1;
                }
            };
            match g.bound_model.as_ref() {
                Some(bm) => (
                    bm.weight_table.clone(),
                    bm.final_norm_dev,
                    bm.lm_head_dev,
                    bm.embed_host.clone(),
                    bm.n_layers,
                    bm.d_model,
                    bm.vocab_size,
                ),
                None => {
                    eprintln!("CFIE: generate refused — no model bound (call nsl_cfie_bind_model first)");
                    return -1;
                }
            }
        };
        // Engine must be finalized (kinds 1 + 2 resolved).  resolved()
        // checks finalize; probe kind 2 so a not-finalized engine refuses
        // here rather than deep in the loop.
        if resolved(KIND_DECODE_BLOCK, 0).is_err() || resolved(KIND_FUSED_SAMPLE, 0).is_err() {
            eprintln!("CFIE: generate refused — engine not finalized (kinds 1+2 unresolved)");
            return -1;
        }

        let d = d_model as usize;
        // Read the prompt token ids (host i64 array).
        let prompt =
            unsafe { std::slice::from_raw_parts(prompt_tokens_ptr as *const i64, prompt_len as usize) };
        // Output slice (host i64 array of capacity out_cap).
        let out = unsafe {
            std::slice::from_raw_parts_mut(out_tokens_ptr as *mut i64, out_cap as usize)
        };

        // Acquire a fresh KV slot for this sequence.
        let slot = crate::cfie::ffi::nsl_cfie_kv_slot_acquire();
        if slot < 0 {
            eprintln!("CFIE: generate refused — KV slot acquire failed (pool exhausted?)");
            return -1;
        }

        // Device scratch: x_a / x_b (f32 [d_model]) + one u32 out-token.
        // Freed on EVERY exit path below (including early returns).
        let x_a = crate::cuda::inner::alloc_device(d * 4) as i64;
        let x_b = crate::cuda::inner::alloc_device(d * 4) as i64;
        let tok_dev = crate::cuda::inner::alloc_device(4) as i64;

        // Single cleanup closure — release slot + free scratch.  Called on
        // every return so no device buffer or KV slot leaks.
        let cleanup = |x_a: i64, x_b: i64, tok_dev: i64, slot: i64| {
            crate::cuda::inner::free_device(x_a as *mut c_void);
            crate::cuda::inner::free_device(x_b as *mut c_void);
            crate::cuda::inner::free_device(tok_dev as *mut c_void);
            crate::cfie::ffi::nsl_cfie_kv_slot_release(slot);
        };

        let mut generated: i64 = 0;
        let mut last_sampled: i64 = 0;
        let total_steps = prompt_len + max_new_tokens;
        for pos in 0..total_steps {
            let input_token = if pos < prompt_len {
                prompt[pos as usize]
            } else {
                last_sampled
            };
            // Embedding gather: memcpy the host f32 row `input_token`
            // (d_model elems) into x_a.  Guard the token id against the
            // vocab so a bad id refuses rather than reading OOB host mem.
            if input_token < 0 || input_token >= vocab_size {
                cleanup(x_a, x_b, tok_dev, slot);
                eprintln!(
                    "CFIE: generate refused — input token id {input_token} out of range [0, {vocab_size})"
                );
                return -1;
            }
            let row_start = input_token as usize * d;
            crate::cuda::inner::memcpy_htod(
                x_a as *mut c_void,
                embed_host[row_start..row_start + d].as_ptr() as *const c_void,
                d * 4,
            );

            let rc = nsl_cfie_decode_step(
                x_a,
                x_b,
                weight_table.as_ptr() as i64,
                n_layers,
                final_norm_dev as i64,
                lm_head_dev as i64,
                slot,
                pos,
                rng_seed,
                0, // grammar_state — DEFERRED (v1 unconstrained sampling)
                tok_dev,
            );
            if rc == -2 {
                // KV slot capacity reached -> stop cleanly (not a failure).
                break;
            }
            if rc < 0 {
                // Real decode failure -> release slot + free scratch, -1.
                cleanup(x_a, x_b, tok_dev, slot);
                eprintln!("CFIE: generate — decode_step failed with rc {rc}");
                return -1;
            }

            // Copy the sampled device u32 -> host.
            let mut sampled_u32 = [0u32; 1];
            crate::cuda::inner::memcpy_dtoh(
                sampled_u32.as_mut_ptr() as *mut c_void,
                tok_dev as *const c_void,
                4,
            );
            let sampled = sampled_u32[0] as i64;

            // The last prefill step (pos == prompt_len - 1) already
            // produces the FIRST new token, so recording starts there.
            if pos >= prompt_len - 1 {
                if generated < out_cap {
                    out[generated as usize] = sampled;
                }
                generated += 1;
                if sampled == eos_token_id {
                    break;
                }
                if generated >= max_new_tokens {
                    break;
                }
            }
            last_sampled = sampled;
        }

        cleanup(x_a, x_b, tok_dev, slot);
        generated
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            prompt_tokens_ptr,
            prompt_len,
            max_new_tokens,
            eos_token_id,
            rng_seed,
            out_tokens_ptr,
            out_cap,
        );
        eprintln!("CFIE: generate requires a CUDA-enabled build and GPU — refusing (no generation)");
        -1
    }
}

/// Clear the bound-model state (weight table, embed/final_norm/lm_head,
/// shape) WITHOUT freeing the device weight buffers — those are owned by
/// `nsl_cfie_weights_reset` / `nsl_cfie_engine_destroy`.  Idempotent;
/// always returns 0.  Cycle 11 rebinds via `nsl_cfie_bind_model` after a
/// weights reset; this FFI just drops the table records so a stale
/// binding cannot drive `nsl_cfie_generate`.
#[no_mangle]
pub extern "C" fn nsl_cfie_generate_reset() -> i64 {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    g.bound_model = None;
    0
}

// ---------------------------------------------------------------------------
// Draft-model binding + draft KV pool (CFIE Cycle 13, G15)
//
// The paper's serve speculative config names a DRAFT model compiled into
// the SAME binary (same CUDA context, same memory pool).  The engine
// holds it as a SECOND BoundModel + a SECOND, engine-held KV pool: the
// draft decode chain (kind 6 — the existing decode-block emitter
// instantiated with the draft dims, max_slots = 1) injects the draft
// pool base as kv_base and always runs slot 0; draft position
// bookkeeping is host-side in `nsl_cfie_speculative_generate` (no slot
// allocator — appends at an explicit pos, overwriting stale rows by
// position IS the rollback).
// ---------------------------------------------------------------------------

/// Free every weight allocation recorded at index >= `start` and
/// truncate the tracking list back to `start`.  Cleanup path for a
/// failed DRAFT bind: the target model's uploads (recorded before
/// `start`) must survive, so the full `free_weight_allocs` /
/// `nsl_cfie_weights_reset` hammer cannot be used.
#[cfg(feature = "cuda")]
fn free_weight_allocs_from(g: &mut EngineState, start: usize) {
    let start = start.min(g.weight_allocs.len());
    for ptr in g.weight_allocs.drain(start..) {
        crate::cuda::inner::free_managed(ptr as *mut c_void);
    }
}

/// Bind a loaded `NslModel` as the DRAFT model for speculative decoding:
/// same resolution + upload as `nsl_cfie_bind_model` (HF-Llama names,
/// Cycle-9 upload FFIs, refuse-on-missing) into a SEPARATE draft binding
/// (weight table + final norm + lm_head + host f32 embed).
///
/// Refuses (-1, eprintln) when: null handle / non-positive dim; NO
/// target model is bound yet (speculative decoding verifies against the
/// target — bind it first); `vocab_size` differs from the target
/// binding's vocab (rejection sampling compares prob rows over ONE
/// shared vocab); any weight is missing/mis-shaped (only the DRAFT
/// uploads recorded by this call are freed — the target binding
/// survives); non-cuda build.  A re-bind replaces the previous draft
/// records (the superseded device buffers stay in `weight_allocs` until
/// `nsl_cfie_weights_reset` / destroy, matching bind_model's re-bind
/// discipline).
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_cfie_bind_draft_model(
    model_handle: i64,
    n_layers: i64,
    d_model: i64,
    n_heads: i64,
    n_kv_heads: i64,
    head_dim: i64,
    d_ff: i64,
    vocab_size: i64,
) -> i64 {
    // Cross-build guards (pure bookkeeping — meaningful CPU-only tests):
    // arg validation, target-bound ordering, shared-vocab invariant.
    if model_handle == 0 {
        eprintln!("CFIE: bind_draft_model refused — null model handle");
        return -1;
    }
    if n_layers <= 0
        || d_model <= 0
        || n_heads <= 0
        || n_kv_heads <= 0
        || head_dim <= 0
        || d_ff <= 0
        || vocab_size <= 0
    {
        eprintln!("CFIE: bind_draft_model refused — non-positive dimension in the model shape");
        return -1;
    }
    {
        let g = match engine().lock() {
            Ok(g) => g,
            Err(_) => {
                eprintln!("CFIE: bind_draft_model refused — engine lock poisoned");
                return -1;
            }
        };
        match g.bound_model.as_ref() {
            None => {
                eprintln!(
                    "CFIE: bind_draft_model refused — no target model bound (call nsl_cfie_bind_model first)"
                );
                return -1;
            }
            Some(bm) if bm.vocab_size != vocab_size => {
                eprintln!(
                    "CFIE: bind_draft_model refused — draft vocab {} != target vocab {} (speculative decoding requires a shared vocab)",
                    vocab_size, bm.vocab_size
                );
                return -1;
            }
            Some(_) => {}
        }
    }
    #[cfg(feature = "cuda")]
    {
        // Snapshot the upload-tracking length so a failed draft bind can
        // free ONLY its own partial uploads (the target's stay).
        let start = match engine().lock() {
            Ok(g) => g.weight_allocs.len(),
            Err(_) => {
                eprintln!("CFIE: bind_draft_model refused — engine lock poisoned");
                return -1;
            }
        };
        match resolve_and_upload_model(
            model_handle,
            n_layers,
            d_model,
            n_heads,
            n_kv_heads,
            head_dim,
            d_ff,
            vocab_size,
        ) {
            Ok(bm) => {
                let mut g = engine().lock().unwrap_or_else(|e| e.into_inner());
                // Re-verify the ordering invariants under the commit lock
                // (the target could have been unbound/re-bound between the
                // guard above and here) — refuse + free our uploads rather
                // than commit a draft that no longer matches.
                let target_ok = matches!(
                    g.bound_model.as_ref(),
                    Some(t) if t.vocab_size == vocab_size
                );
                if !target_ok {
                    free_weight_allocs_from(&mut g, start);
                    eprintln!(
                        "CFIE: bind_draft_model refused — target binding changed during the draft bind"
                    );
                    return -1;
                }
                g.draft_model = Some(bm);
                0
            }
            Err(missing) => {
                // Deferral-must-refuse; free ONLY this call's uploads.
                {
                    let mut g = engine().lock().unwrap_or_else(|e| e.into_inner());
                    free_weight_allocs_from(&mut g, start);
                    g.draft_model = None;
                }
                eprintln!(
                    "CFIE: bind_draft_model refused — missing or mis-shaped weight: {missing}"
                );
                -1
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!(
            "CFIE: bind_draft_model requires a CUDA-enabled build and GPU — refusing (no binding)"
        );
        -1
    }
}

/// Allocate + zero the DRAFT KV pool (persistent caching-allocator
/// bracket, same as the target pool).  Engine-held: NOT attached to the
/// target KV slot allocator — kind-6 launches inject this base as their
/// `kv_base`, and draft position bookkeeping is host-side.  Returns 0
/// ok; -1 when `bytes <= 0`, a draft pool is already allocated (call
/// `nsl_cfie_draft_reset` / destroy first), or on a non-cuda build.
///
/// SIZING CONTRACT: the registered kind-6 kernel's baked
/// `per_slot_max_tokens` (which `bytes` must match:
/// `draft_n_layers * 2 * per_slot * draft_n_kv_heads * draft_head_dim
/// * 2`) must be >= the TARGET's per-slot token capacity —
/// `nsl_cfie_speculative_generate`'s capacity probe bounds draft-KV
/// writes by the TARGET capacity only.  The serve wiring bakes the
/// same per-slot value into both DecodeBlockConfigs.
#[no_mangle]
pub extern "C" fn nsl_cfie_draft_pool_alloc(bytes: i64) -> i64 {
    if bytes <= 0 {
        return -1;
    }
    #[cfg(feature = "cuda")]
    {
        let mut g = match engine().lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        if g.draft_pool_base != 0 {
            return -1; // already allocated — nsl_cfie_draft_reset first
        }
        let ptr = alloc_persistent(bytes as usize);
        if ptr.is_null() {
            return -1;
        }
        crate::cuda::inner::memset_d8(ptr, bytes as usize);
        g.draft_pool_base = ptr as u64;
        g.draft_pool_bytes = bytes as u64;
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CFIE: draft_pool_alloc requires a CUDA-enabled build and GPU — refusing (no pool allocated)");
        -1
    }
}

/// Test/diagnostic accessor: the DRAFT KV pool device base recorded by
/// `nsl_cfie_draft_pool_alloc`, or 0 when no draft pool is allocated.
/// Mirrors `nsl_cfie_kv_pool_base` — GPU parity tests seed draft K/V
/// rows through it; production launches inject the base internally.
#[doc(hidden)]
#[no_mangle]
pub extern "C" fn nsl_cfie_draft_pool_base() -> i64 {
    match engine().lock() {
        Ok(g) => g.draft_pool_base as i64,
        Err(_) => 0,
    }
}

/// Clear the draft-model binding and free the draft KV pool.
/// Idempotent; always returns 0.  Does NOT free the draft WEIGHT device
/// buffers — those live in `weight_allocs` (shared with the target's)
/// and are freed by `nsl_cfie_weights_reset` / `nsl_cfie_engine_destroy`;
/// this drops the binding records so a stale draft cannot drive
/// `nsl_cfie_speculative_generate`.
#[no_mangle]
pub extern "C" fn nsl_cfie_draft_reset() -> i64 {
    let mut g = match engine().lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    #[cfg(feature = "cuda")]
    {
        if g.draft_pool_base != 0 {
            crate::cuda::inner::free_managed(g.draft_pool_base as *mut std::ffi::c_void);
        }
    }
    g.draft_pool_base = 0;
    g.draft_pool_bytes = 0;
    g.draft_model = None;
    0
}

// ---------------------------------------------------------------------------
// Launch plumbing (cuda builds only)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
struct ResolvedKernel {
    func: usize,
    grid_x: u32,
    block_x: u32,
    smem_dyn_bytes: u32,
}

/// Snapshot the cached handle + launch meta for `(kind, layer_idx)`.
/// Err(-1) when the engine is not finalized or the kind is missing.
#[cfg(feature = "cuda")]
fn resolved(kind: i64, layer_idx: i64) -> Result<ResolvedKernel, i64> {
    let g = engine().lock().map_err(|_| -1i64)?;
    if !g.finalized {
        return Err(-1);
    }
    match g.kernels.get(&(kind, layer_idx)) {
        Some(k) if k.func != 0 => Ok(ResolvedKernel {
            func: k.func,
            grid_x: k.grid_x,
            block_x: k.block_x,
            smem_dyn_bytes: k.smem_dyn_bytes,
        }),
        _ => Err(-1),
    }
}

/// The grammar-mask device address resolved at finalize (0 = absent).
#[cfg(feature = "cuda")]
fn grammar_mask_dev() -> Result<u64, i64> {
    let g = engine().lock().map_err(|_| -1i64)?;
    if !g.finalized {
        return Err(-1);
    }
    Ok(g.grammar_mask_dev)
}

/// KV pool base injected as the kernels' `kv_base` parameter.  Err(-1)
/// when the pool is missing (device_base 0) — refuse rather than hand
/// the kernel a null KV region.
#[cfg(feature = "cuda")]
fn kv_base() -> Result<u64, i64> {
    let g = kv_slots_global().lock().map_err(|_| -1i64)?;
    match g.as_ref() {
        Some(a) if a.device_base() != 0 => Ok(a.device_base()),
        _ => Err(-1),
    }
}

/// grid=(grid_x,1,1) block=(block_x,1,1) from the registration meta.
#[cfg(feature = "cuda")]
fn launch(meta: &ResolvedKernel, args: &[*mut c_void]) -> i64 {
    crate::cuda::inner::launch_function_raw(
        meta.func,
        [meta.grid_x, 1, 1],
        [meta.block_x, 1, 1],
        args,
        meta.smem_dyn_bytes,
    ) as i64
}

// ---------------------------------------------------------------------------
// Launch FFIs — kernel params marshalled EXACTLY per each emitter's
// frozen .param list; u32 params passed as u32 locals, f32 params as
// f32 locals (bit-punned from the i64 low half), pointers as u64
// locals; args = pointers TO those locals (cuLaunchKernel convention).
// Returns: 0 ok, -1 engine-not-finalized / kind-missing /
// pool-missing-when-needed, else the positive CUresult.
// ---------------------------------------------------------------------------

/// Kind 0 — kernel params (cfie_decode_attention.rs, 6): q_ptr.u64,
/// kv_base.u64 (INJECTED from the slot allocator), out_ptr.u64,
/// layer_idx.u32, slot_idx.u32, seq_len.u32.
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_decode_attn(
    q_ptr: i64,
    out_ptr: i64,
    layer_idx: i64,
    slot_idx: i64,
    seq_len: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_DECODE_ATTN, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let kv = match kv_base() {
            Ok(k) => k,
            Err(e) => return e,
        };
        let mut q = q_ptr as u64;
        let mut kvb = kv;
        let mut out = out_ptr as u64;
        let mut layer = layer_idx as u32;
        let mut slot = slot_idx as u32;
        let mut slen = seq_len as u32;
        let args: [*mut c_void; 6] = [
            &mut q as *mut _ as *mut c_void,
            &mut kvb as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut layer as *mut _ as *mut c_void,
            &mut slot as *mut _ as *mut c_void,
            &mut slen as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, out_ptr, layer_idx, slot_idx, seq_len);
        -1
    }
}

/// Kind 1 — kernel params (cfie_sample_ptx.rs, 7): hidden.u64,
/// norm_w.u64, lm_head.u64, out_token.u64, rng_seed.u64,
/// grammar_mask_ptr.u64 (INJECTED: the cuModuleGetGlobal address or 0),
/// grammar_state.u32.
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_fused_sample(
    hidden_ptr: i64,
    norm_w_ptr: i64,
    lm_head_ptr: i64,
    out_token_ptr: i64,
    rng_seed: i64,
    grammar_state: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_FUSED_SAMPLE, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let mask = match grammar_mask_dev() {
            Ok(m) => m,
            Err(e) => return e,
        };
        let mut hidden = hidden_ptr as u64;
        let mut norm_w = norm_w_ptr as u64;
        let mut lm_head = lm_head_ptr as u64;
        let mut out_tok = out_token_ptr as u64;
        let mut seed = rng_seed as u64;
        let mut mask_ptr = mask;
        let mut gstate = grammar_state as u32;
        let args: [*mut c_void; 7] = [
            &mut hidden as *mut _ as *mut c_void,
            &mut norm_w as *mut _ as *mut c_void,
            &mut lm_head as *mut _ as *mut c_void,
            &mut out_tok as *mut _ as *mut c_void,
            &mut seed as *mut _ as *mut c_void,
            &mut mask_ptr as *mut _ as *mut c_void,
            &mut gstate as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            hidden_ptr,
            norm_w_ptr,
            lm_head_ptr,
            out_token_ptr,
            rng_seed,
            grammar_state,
        );
        -1
    }
}

/// Kind 2 — kernel params (cfie_persistent_ptx.rs, 15): x_in.u64,
/// x_out.u64, wq.u64, wk.u64, wv.u64, wo.u64, w_gate.u64, w_up.u64,
/// w_down.u64, norm1_w.u64, norm2_w.u64, kv_base.u64 (INJECTED),
/// layer_idx.u32, slot_idx.u32, pos.u32.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_cfie_launch_decode_block(
    x_in: i64,
    x_out: i64,
    wq: i64,
    wk: i64,
    wv: i64,
    wo: i64,
    w_gate: i64,
    w_up: i64,
    w_down: i64,
    norm1_w: i64,
    norm2_w: i64,
    layer_idx: i64,
    slot_idx: i64,
    pos: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_DECODE_BLOCK, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let kv = match kv_base() {
            Ok(k) => k,
            Err(e) => return e,
        };
        let mut xin = x_in as u64;
        let mut xout = x_out as u64;
        let mut p_wq = wq as u64;
        let mut p_wk = wk as u64;
        let mut p_wv = wv as u64;
        let mut p_wo = wo as u64;
        let mut p_wg = w_gate as u64;
        let mut p_wu = w_up as u64;
        let mut p_wd = w_down as u64;
        let mut p_n1 = norm1_w as u64;
        let mut p_n2 = norm2_w as u64;
        let mut kvb = kv;
        let mut layer = layer_idx as u32;
        let mut slot = slot_idx as u32;
        let mut p_pos = pos as u32;
        let args: [*mut c_void; 15] = [
            &mut xin as *mut _ as *mut c_void,
            &mut xout as *mut _ as *mut c_void,
            &mut p_wq as *mut _ as *mut c_void,
            &mut p_wk as *mut _ as *mut c_void,
            &mut p_wv as *mut _ as *mut c_void,
            &mut p_wo as *mut _ as *mut c_void,
            &mut p_wg as *mut _ as *mut c_void,
            &mut p_wu as *mut _ as *mut c_void,
            &mut p_wd as *mut _ as *mut c_void,
            &mut p_n1 as *mut _ as *mut c_void,
            &mut p_n2 as *mut _ as *mut c_void,
            &mut kvb as *mut _ as *mut c_void,
            &mut layer as *mut _ as *mut c_void,
            &mut slot as *mut _ as *mut c_void,
            &mut p_pos as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            x_in, x_out, wq, wk, wv, wo, w_gate, w_up, w_down, norm1_w, norm2_w, layer_idx,
            slot_idx, pos,
        );
        -1
    }
}

/// Kind 3 — kernel params (cfie_speculative_ptx.rs, 6): q.u64,
/// kv_base.u64 (INJECTED), out.u64, layer_idx.u32, slot_idx.u32,
/// seq_len.u32.
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_spec_verify(
    q_ptr: i64,
    out_ptr: i64,
    layer_idx: i64,
    slot_idx: i64,
    seq_len: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_SPEC_VERIFY, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let kv = match kv_base() {
            Ok(k) => k,
            Err(e) => return e,
        };
        let mut q = q_ptr as u64;
        let mut kvb = kv;
        let mut out = out_ptr as u64;
        let mut layer = layer_idx as u32;
        let mut slot = slot_idx as u32;
        let mut slen = seq_len as u32;
        let args: [*mut c_void; 6] = [
            &mut q as *mut _ as *mut c_void,
            &mut kvb as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut layer as *mut _ as *mut c_void,
            &mut slot as *mut _ as *mut c_void,
            &mut slen as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, out_ptr, layer_idx, slot_idx, seq_len);
        -1
    }
}

/// Kind 4 — kernel params (cfie_speculative_ptx.rs, 6, no injection):
/// target_probs.u64, draft_probs.u64, draft_tokens.u64, rng_seed.u64,
/// out_accepted.u64, out_correction_token.u64.
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_spec_reject(
    target_probs_ptr: i64,
    draft_probs_ptr: i64,
    draft_tokens_ptr: i64,
    rng_seed: i64,
    out_accepted_ptr: i64,
    out_correction_token_ptr: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_SPEC_REJECT, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let mut tp = target_probs_ptr as u64;
        let mut dp = draft_probs_ptr as u64;
        let mut dt = draft_tokens_ptr as u64;
        let mut seed = rng_seed as u64;
        let mut oa = out_accepted_ptr as u64;
        let mut oc = out_correction_token_ptr as u64;
        let args: [*mut c_void; 6] = [
            &mut tp as *mut _ as *mut c_void,
            &mut dp as *mut _ as *mut c_void,
            &mut dt as *mut _ as *mut c_void,
            &mut seed as *mut _ as *mut c_void,
            &mut oa as *mut _ as *mut c_void,
            &mut oc as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            target_probs_ptr,
            draft_probs_ptr,
            draft_tokens_ptr,
            rng_seed,
            out_accepted_ptr,
            out_correction_token_ptr,
        );
        -1
    }
}

/// Kind 5 — `layer_idx` selects the `(5, layer_idx)` registration (the
/// quant emitter bakes the layer into the PTX).  Kernel params
/// (cfie_kv_quant_ptx.rs, 7): q.u64, kv_base.u64 (INJECTED), out.u64,
/// slot_idx.u32, seq_len.u32, k_scale.f32, v_scale.f32 (each f32
/// bit-punned from the low 32 bits of its i64 arg).
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_quant_attn(
    layer_idx: i64,
    q_ptr: i64,
    out_ptr: i64,
    slot_idx: i64,
    seq_len: i64,
    k_scale_bits: i64,
    v_scale_bits: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_QUANT_ATTN, layer_idx) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let kv = match kv_base() {
            Ok(k) => k,
            Err(e) => return e,
        };
        let mut q = q_ptr as u64;
        let mut kvb = kv;
        let mut out = out_ptr as u64;
        let mut slot = slot_idx as u32;
        let mut slen = seq_len as u32;
        // `as u32` on i64 truncates to the low 32 bits — the ABI's
        // f32::to_bits transport convention.
        let mut k_scale = f32::from_bits(k_scale_bits as u32);
        let mut v_scale = f32::from_bits(v_scale_bits as u32);
        let args: [*mut c_void; 7] = [
            &mut q as *mut _ as *mut c_void,
            &mut kvb as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut slot as *mut _ as *mut c_void,
            &mut slen as *mut _ as *mut c_void,
            &mut k_scale as *mut _ as *mut c_void,
            &mut v_scale as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            layer_idx,
            q_ptr,
            out_ptr,
            slot_idx,
            seq_len,
            k_scale_bits,
            v_scale_bits,
        );
        -1
    }
}

/// Kind 6 — the DRAFT decode block.  Same 15-param kernel as kind 2
/// (cfie_persistent_ptx.rs — the emitter is instantiated with the DRAFT
/// DecodeBlockConfig: draft dims, per_slot_max_tokens = draft pool
/// capacity, max_slots = 1), but the injection differs: `kv_base` = the
/// DRAFT pool base (engine-held, `nsl_cfie_draft_pool_alloc`), the nine
/// weight pointers come from the DRAFT binding's table for `layer_idx`
/// (engine-held — unlike kind 2 the weights are NOT FFI arguments,
/// keeping this FFI small), and `slot_idx` is fixed 0 (single draft
/// sequence).  Returns 0 ok; -1 not-finalized / kind missing / no draft
/// binding / `layer_idx` out of range / negative pos / draft pool
/// unallocated; else the positive CUresult.
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_draft_block(
    x_in: i64,
    x_out: i64,
    layer_idx: i64,
    pos: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_DRAFT_BLOCK, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        if pos < 0 {
            return -1; // would wrap to a huge u32 KV row index
        }
        // Snapshot the draft layer record + pool base under one lock.
        let (rec, kvb) = {
            let g = match engine().lock() {
                Ok(g) => g,
                Err(_) => return -1,
            };
            let dm = match g.draft_model.as_ref() {
                Some(dm) => dm,
                None => return -1,
            };
            if layer_idx < 0 || layer_idx >= dm.n_layers {
                return -1;
            }
            if g.draft_pool_base == 0 {
                return -1; // refuse rather than hand the kernel a null KV region
            }
            let base = layer_idx as usize * 9;
            let rec: [u64; 9] = match dm
                .weight_table
                .get(base..base + 9)
                .and_then(|s| s.try_into().ok())
            {
                Some(r) => r,
                None => return -1, // malformed binding — refuse
            };
            (rec, g.draft_pool_base)
        };
        let mut xin = x_in as u64;
        let mut xout = x_out as u64;
        let mut p_wq = rec[0];
        let mut p_wk = rec[1];
        let mut p_wv = rec[2];
        let mut p_wo = rec[3];
        let mut p_wg = rec[4];
        let mut p_wu = rec[5];
        let mut p_wd = rec[6];
        let mut p_n1 = rec[7];
        let mut p_n2 = rec[8];
        let mut kv = kvb;
        let mut layer = layer_idx as u32;
        let mut slot = 0u32; // draft config is max_slots = 1
        let mut p_pos = pos as u32;
        let args: [*mut c_void; 15] = [
            &mut xin as *mut _ as *mut c_void,
            &mut xout as *mut _ as *mut c_void,
            &mut p_wq as *mut _ as *mut c_void,
            &mut p_wk as *mut _ as *mut c_void,
            &mut p_wv as *mut _ as *mut c_void,
            &mut p_wo as *mut _ as *mut c_void,
            &mut p_wg as *mut _ as *mut c_void,
            &mut p_wu as *mut _ as *mut c_void,
            &mut p_wd as *mut _ as *mut c_void,
            &mut p_n1 as *mut _ as *mut c_void,
            &mut p_n2 as *mut _ as *mut c_void,
            &mut kv as *mut _ as *mut c_void,
            &mut layer as *mut _ as *mut c_void,
            &mut slot as *mut _ as *mut c_void,
            &mut p_pos as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (x_in, x_out, layer_idx, pos);
        -1
    }
}

/// Kind 7 — draft greedy sampler.  Kernel params
/// (cfie_spec_sampler_ptx.rs `nsl_cfie_draft_sample`, 6): hidden.u64,
/// norm_w.u64 (INJECTED: draft binding's final-norm gamma), lm_head.u64
/// (INJECTED: draft binding's f16 lm_head), out_token.u64 (st u32
/// argmax), out_prob.u64 (st f32 = p(argmax)), rng_seed.u64 (ACCEPTED
/// for ABI symmetry; UNUSED by the kernel — v1 drafting is greedy per
/// the paper's temperature 0.0).  Sampler-family: no KV access, no pool
/// needed.  Returns 0 ok; -1 not-finalized / kind missing / no draft
/// binding; else the positive CUresult.
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_draft_sample(
    hidden_ptr: i64,
    out_token_ptr: i64,
    out_prob_ptr: i64,
    rng_seed: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_DRAFT_SAMPLE, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let (norm, lm) = {
            let g = match engine().lock() {
                Ok(g) => g,
                Err(_) => return -1,
            };
            match g.draft_model.as_ref() {
                Some(dm) => (dm.final_norm_dev, dm.lm_head_dev),
                None => return -1,
            }
        };
        let mut hidden = hidden_ptr as u64;
        let mut norm_w = norm;
        let mut lm_head = lm;
        let mut out_tok = out_token_ptr as u64;
        let mut out_prob = out_prob_ptr as u64;
        let mut seed = rng_seed as u64;
        let args: [*mut c_void; 6] = [
            &mut hidden as *mut _ as *mut c_void,
            &mut norm_w as *mut _ as *mut c_void,
            &mut lm_head as *mut _ as *mut c_void,
            &mut out_tok as *mut _ as *mut c_void,
            &mut out_prob as *mut _ as *mut c_void,
            &mut seed as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (hidden_ptr, out_token_ptr, out_prob_ptr, rng_seed);
        -1
    }
}

/// Kind 8 — verification prob-row writer.  Kernel params
/// (cfie_spec_sampler_ptx.rs `nsl_cfie_verify_probs`, 4): hidden.u64,
/// norm_w.u64 (INJECTED: TARGET binding's final-norm gamma), lm_head.u64
/// (INJECTED: TARGET binding's f16 lm_head), out_probs.u64 (st f32 x
/// vocab, the full softmaxed row).  The fused sampler never materializes
/// probs by design; this kernel exists so the kind-4 rejection kernel
/// has target rows to compare against.  Returns 0 ok; -1 not-finalized /
/// kind missing / no target binding; else the positive CUresult.
#[no_mangle]
pub extern "C" fn nsl_cfie_launch_verify_probs(hidden_ptr: i64, out_probs_ptr: i64) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let meta = match resolved(KIND_VERIFY_PROBS, 0) {
            Ok(m) => m,
            Err(e) => return e,
        };
        let (norm, lm) = {
            let g = match engine().lock() {
                Ok(g) => g,
                Err(_) => return -1,
            };
            match g.bound_model.as_ref() {
                Some(bm) => (bm.final_norm_dev, bm.lm_head_dev),
                None => return -1,
            }
        };
        let mut hidden = hidden_ptr as u64;
        let mut norm_w = norm;
        let mut lm_head = lm;
        let mut out_probs = out_probs_ptr as u64;
        let args: [*mut c_void; 4] = [
            &mut hidden as *mut _ as *mut c_void,
            &mut norm_w as *mut _ as *mut c_void,
            &mut lm_head as *mut _ as *mut c_void,
            &mut out_probs as *mut _ as *mut c_void,
        ];
        launch(&meta, &args)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (hidden_ptr, out_probs_ptr);
        -1
    }
}

// ---------------------------------------------------------------------------
// Host decode loop
// ---------------------------------------------------------------------------

/// THE host decode loop for one token.  Requires kinds 2 (decode_block)
/// and 1 (fused_sample) finalized; NULL-stream ordering serializes the
/// per-layer chain (fused_ce precedent).
///
/// 1. `kv_slot_advance(slot, 1)` FIRST — refusal returns -2 without
///    launching anything; a length that disagrees with `pos + 1`
///    rolls back and returns -3 (caller/kernel pos mismatch).
/// 2. One decode_block launch per layer, ping-ponging x between the
///    two buffers (layer 0: a->b, layer 1: b->a, ...), with weights
///    read from `layer_weights_ptr` — a HOST-memory array of
///    `n_layers` records x 9 u64 DEVICE pointers in order
///    (wq, wk, wv, wo, w_gate, w_up, w_down, norm1_w, norm2_w).
/// 3. fused_sample on the final hidden buffer (`norm_w_ptr` =
///    final-norm gamma), writing `out_token_ptr` (device u32).
///
/// On any launch failure: `rollback(slot, 1)` and return the launch's
/// error code.
///
/// # Safety
/// `layer_weights_ptr` must point to `n_layers * 9` readable host
/// `u64`s; all other pointers are opaque device addresses passed
/// through to the kernels.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_cfie_decode_step(
    x_buf_a: i64,
    x_buf_b: i64,
    layer_weights_ptr: i64,
    n_layers: i64,
    norm_w_ptr: i64,
    lm_head_ptr: i64,
    slot_idx: i64,
    pos: i64,
    rng_seed: i64,
    grammar_state: i64,
    out_token_ptr: i64,
) -> i64 {
    if layer_weights_ptr == 0 || n_layers <= 0 {
        return -1;
    }
    // (1) Book the token in the slot allocator BEFORE any launch — a
    // capacity refusal must abort before any kernel writes KV entries.
    let new_len = crate::cfie::ffi::nsl_cfie_kv_slot_advance(slot_idx, 1);
    if new_len < 0 {
        return -2;
    }
    if new_len != pos + 1 {
        // The caller's pos and the allocator's length disagree — the
        // kernels would index the wrong KV row.  Un-book and refuse.
        crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot_idx, 1);
        return -3;
    }
    // (2) Per-layer decode_block launches, ping-ponging the hidden
    // state between the two device buffers.
    for layer in 0..n_layers {
        let rec = unsafe {
            std::slice::from_raw_parts(
                (layer_weights_ptr as *const u64).add(layer as usize * 9),
                9,
            )
        };
        let (x_in, x_out) = if layer % 2 == 0 {
            (x_buf_a, x_buf_b)
        } else {
            (x_buf_b, x_buf_a)
        };
        let rc = nsl_cfie_launch_decode_block(
            x_in,
            x_out,
            rec[0] as i64, // wq
            rec[1] as i64, // wk
            rec[2] as i64, // wv
            rec[3] as i64, // wo
            rec[4] as i64, // w_gate
            rec[5] as i64, // w_up
            rec[6] as i64, // w_down
            rec[7] as i64, // norm1_w
            rec[8] as i64, // norm2_w
            layer,
            slot_idx,
            pos,
        );
        if rc != 0 {
            crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot_idx, 1);
            return rc;
        }
    }
    // (3) Sample from the final hidden buffer: after an even number of
    // layers the last write landed in x_buf_a (layer n-1 was odd,
    // b->a); after an odd number it landed in x_buf_b.
    let final_hidden = if n_layers % 2 == 0 { x_buf_a } else { x_buf_b };
    let rc = nsl_cfie_launch_fused_sample(
        final_hidden,
        norm_w_ptr,
        lm_head_ptr,
        out_token_ptr,
        rng_seed,
        grammar_state,
    );
    if rc != 0 {
        crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot_idx, 1);
        return rc;
    }
    0
}

// ---------------------------------------------------------------------------
// Speculative generation driver (CFIE Cycle 13, G15)
// ---------------------------------------------------------------------------

/// Byte count for a host read of `n_elems` i64 — same isize::MAX
/// refusal contract as `checked_host_f32_bytes` (Cycle-9 overflow-guard
/// pattern) for the prompt/out token arrays.
#[cfg(feature = "cuda")]
fn checked_host_i64_bytes(n_elems: i64) -> Option<usize> {
    if n_elems <= 0 {
        return None;
    }
    (n_elems as u64)
        .checked_mul(std::mem::size_of::<i64>() as u64)
        .filter(|&b| b <= isize::MAX as u64)
        .map(|b| b as usize)
}

/// Embedding gather: memcpy the host f32 row `token` (`d` elems) into
/// the device buffer `x_dev`.  The CALLER guards `token` against the
/// vocab — this indexes `embed_host` directly.
#[cfg(feature = "cuda")]
fn gather_embed_row(embed_host: &[f32], d: usize, token: i64, x_dev: i64) {
    let row_start = token as usize * d;
    crate::cuda::inner::memcpy_htod(
        x_dev as *mut c_void,
        embed_host[row_start..row_start + d].as_ptr() as *const c_void,
        d * 4,
    );
}

/// Run the DRAFT decode-block chain (kind 6, all draft layers) at `pos`,
/// ping-ponging the hidden state between the two draft buffers exactly
/// like `nsl_cfie_decode_step` does for the target.  Returns the final
/// hidden buffer on success, or `Err(rc)` from the failing launch.  No
/// slot booking: draft position bookkeeping is host-side (the caller's
/// `draft_pos` counter), and appending at an explicit `pos` overwrites
/// stale rows by position.
#[cfg(feature = "cuda")]
fn draft_chain(x_da: i64, x_db: i64, n_layers: i64, pos: i64) -> Result<i64, i64> {
    for layer in 0..n_layers {
        let (x_in, x_out) = if layer % 2 == 0 {
            (x_da, x_db)
        } else {
            (x_db, x_da)
        };
        let rc = nsl_cfie_launch_draft_block(x_in, x_out, layer, pos);
        if rc != 0 {
            return Err(rc);
        }
    }
    Ok(if n_layers % 2 == 0 { x_da } else { x_db })
}

/// One VERIFY-position target forward: identical to
/// `nsl_cfie_decode_step` (advance-first slot booking, kind-2 chain,
/// same rollback-on-failure discipline) except the terminal kernel is
/// kind 8 — the full softmaxed prob ROW writer — instead of the fused
/// sampler.  Return contract mirrors decode_step: 0 ok; -2 capacity
/// refusal (nothing launched); -3 pos/len mismatch (booking rolled
/// back); else the failing launch's code (booking rolled back).
#[cfg(feature = "cuda")]
fn target_verify_step(
    x_buf_a: i64,
    x_buf_b: i64,
    weight_table: &[u64],
    n_layers: i64,
    slot_idx: i64,
    pos: i64,
    out_probs_ptr: i64,
) -> i64 {
    let new_len = crate::cfie::ffi::nsl_cfie_kv_slot_advance(slot_idx, 1);
    if new_len < 0 {
        return -2;
    }
    if new_len != pos + 1 {
        crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot_idx, 1);
        return -3;
    }
    for layer in 0..n_layers {
        let base = layer as usize * 9;
        let rec = &weight_table[base..base + 9];
        let (x_in, x_out) = if layer % 2 == 0 {
            (x_buf_a, x_buf_b)
        } else {
            (x_buf_b, x_buf_a)
        };
        let rc = nsl_cfie_launch_decode_block(
            x_in,
            x_out,
            rec[0] as i64, // wq
            rec[1] as i64, // wk
            rec[2] as i64, // wv
            rec[3] as i64, // wo
            rec[4] as i64, // w_gate
            rec[5] as i64, // w_up
            rec[6] as i64, // w_down
            rec[7] as i64, // norm1_w
            rec[8] as i64, // norm2_w
            layer,
            slot_idx,
            pos,
        );
        if rc != 0 {
            crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot_idx, 1);
            return rc;
        }
    }
    let final_hidden = if n_layers % 2 == 0 { x_buf_a } else { x_buf_b };
    let rc = nsl_cfie_launch_verify_probs(final_hidden, out_probs_ptr);
    if rc != 0 {
        crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot_idx, 1);
        return rc;
    }
    0
}

/// Speculative decode loop (the paper's serve `speculative:` driver):
/// draft `k_tokens` greedily with the DRAFT model, verify them with one
/// target forward per position + ONE kind-4 rejection launch, emit the
/// accepted prefix (+ the residual correction, or a bonus token on
/// all-accept), and roll both KV sides back to the emitted sequence.
/// Same conventions as `nsl_cfie_generate`: advance-first target slot
/// booking, `out_cap`-clamped writes with the TRUE generated count
/// returned, scratch freed on every path, slot released, vocab-range
/// guards on every token id used as an input.
///
/// Requires: `nsl_cfie_bind_model` + `nsl_cfie_bind_draft_model` +
/// `nsl_cfie_draft_pool_alloc` + `nsl_cfie_engine_finalize` (kinds
/// 1/2/4/6/7/8).  `k_tokens` must be 1..=32 and MUST equal the K the
/// registered kind-4 rejection kernel was compiled for (the serve
/// wiring passes the same speculative-config value to both).
///
/// The loop:
///   1. Prefill BOTH models over the prompt (target: `decode_step` at
///      pos 0..prompt_len — byte-identical to `nsl_cfie_generate`'s
///      prefix, so the first new token, sampled at pos prompt_len-1,
///      agrees token-for-token; draft: kind-6 chain, outputs discarded).
///   2. Rounds until max_new/EOS/capacity.  Each round STARTS with a
///      capacity probe: the draft phase writes K draft-KV rows BEFORE
///      any verify step books the target slot, and the kind-6 kernel
///      appends at an explicit pos with NO bounds check (the plane
///      strides are baked from its per_slot_max_tokens) — so the round
///      may only start when the TARGET slot can book K more rows
///      (`kv_slot_advance(slot, K)` probe, rolled straight back).  When
///      the probe fails, the TAIL of the generation falls back to plain
///      per-token `decode_step`s — exact parity with
///      `nsl_cfie_generate` up to capacity (no truncated round, no
///      draft-KV write past the pool).  Otherwise: draft K tokens (per
///      token: draft embed gather -> kind-6 all layers -> kind-7 ->
///      (token, p_draft)); target-verify K positions (embed gather of
///      the PREVIOUS token -> kind-2 chain appending target KV ->
///      kind-8 prob row); ONE kind-4 launch -> (accepted, correction).
///   3. Rejection: emit the accepted prefix + the correction.  The
///      round appended K KV rows for inputs (last, d0..d[K-2]); the
///      emitted sequence keeps accepted+1 of them, so
///      `K - accepted - 1` rows are STALE on BOTH sides —
///      `nsl_cfie_kv_slot_rollback` for the target, and the draft
///      position counter rewinds by the same count (the correction
///      token is fed at the next round boundary, overwriting the first
///      stale draft row by position — that overwrite IS the draft
///      rollback).
///   4. All-accept (0xFFFFFFFF sentinel): emit the K drafts, then one
///      BONUS token via a plain target `decode_step` (kind-1 fused
///      sampler — generate's step), drafting d[K-1] through the draft
///      chain to keep the two KV sides in lock-step.
///
/// DETERMINISM CONTRACT (the lossless self-speculation invariant): with
/// greedy target sampling AND draft == target weights, p_t/p_d == 1 for
/// every drafted token, the reject kernel accepts all K every round
/// regardless of seed, and the output equals `nsl_cfie_generate`'s for
/// the same prompt/seed/max_new EXACTLY — same positions, same embed
/// gathers, same sampler calls at round boundaries.  The kind-4 seed is
/// varied per round (`rng_seed + round`, wrapping) so real draft/target
/// pairs do not replay identical uniforms each round; the invariant is
/// seed-independent.
///
/// SIZING CONTRACT (enforced by the serve wiring, documented here
/// because the engine cannot verify it): the draft pool must be sized
/// with per_slot_max_tokens >= the TARGET's per-slot token capacity —
/// the capacity probe bounds every draft-KV write by the TARGET
/// capacity, so a smaller draft pool would still overflow.  The serve
/// wiring bakes the SAME per-slot value into both DecodeBlockConfigs.
///
/// SINGLE-FLIGHT: concurrent speculative calls would race on slot 0 of
/// the single engine-held draft pool (kind-6 launches hardcode slot 0),
/// so overlapping calls refuse (-1) instead of silently corrupting
/// draft KV; `nsl_cfie_generate` stays per-slot concurrent.
///
/// Returns the TRUE generated-token count (writes clamped to
/// `out_cap`); EOS anywhere in the emitted tokens stops generation
/// (truncated after the EOS).  -1 on bad args (incl. `k_tokens`
/// outside 1..=32), missing target/draft binding, vocab mismatch,
/// missing finalize/pool, slot-acquire failure, a real launch failure,
/// or a non-cuda build.  A -2 capacity refusal anywhere stops CLEANLY
/// with the tokens emitted so far (via the probe + tail fallback the
/// emitted prefix equals `nsl_cfie_generate`'s output truncated at
/// capacity under the determinism contract).
///
/// # Safety
/// `prompt_tokens_ptr` must point to `prompt_len` readable i64s;
/// `out_tokens_ptr` to `out_cap` writable i64s.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_cfie_speculative_generate(
    prompt_tokens_ptr: i64,
    prompt_len: i64,
    max_new_tokens: i64,
    eos_token_id: i64,
    rng_seed: i64,
    k_tokens: i64,
    out_tokens_ptr: i64,
    out_cap: i64,
) -> i64 {
    // Cross-build guards (pure bookkeeping — CPU-unit-testable): args,
    // binding order, shared-vocab invariant.
    if prompt_tokens_ptr == 0
        || out_tokens_ptr == 0
        || prompt_len <= 0
        || out_cap <= 0
        || max_new_tokens <= 0
    {
        eprintln!(
            "CFIE: speculative_generate refused — bad argument (null ptr or non-positive length/cap)"
        );
        return -1;
    }
    if !(1..=32).contains(&k_tokens) {
        eprintln!(
            "CFIE: speculative_generate refused — k_tokens {k_tokens} outside the frozen 1..=32 range"
        );
        return -1;
    }
    {
        let g = match engine().lock() {
            Ok(g) => g,
            Err(_) => {
                eprintln!("CFIE: speculative_generate refused — engine lock poisoned");
                return -1;
            }
        };
        let bm = match g.bound_model.as_ref() {
            Some(bm) => bm,
            None => {
                eprintln!(
                    "CFIE: speculative_generate refused — no model bound (call nsl_cfie_bind_model first)"
                );
                return -1;
            }
        };
        match g.draft_model.as_ref() {
            Some(dm) if dm.vocab_size == bm.vocab_size => {}
            Some(dm) => {
                eprintln!(
                    "CFIE: speculative_generate refused — draft vocab {} != target vocab {} (rebind required)",
                    dm.vocab_size, bm.vocab_size
                );
                return -1;
            }
            None => {
                eprintln!(
                    "CFIE: speculative_generate refused — no draft model bound (call nsl_cfie_bind_draft_model first)"
                );
                return -1;
            }
        }
    }
    #[cfg(feature = "cuda")]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        // Single-flight guard (see the doc comment): the draft pool is
        // single-slot, so a second concurrent speculative call would
        // race the first's draft KV rows.  Released on EVERY exit path
        // via the drop guard.
        static SPEC_IN_FLIGHT: AtomicBool = AtomicBool::new(false);
        struct SpecFlight;
        impl Drop for SpecFlight {
            fn drop(&mut self) {
                SPEC_IN_FLIGHT.store(false, Ordering::Release);
            }
        }
        if SPEC_IN_FLIGHT.swap(true, Ordering::AcqRel) {
            eprintln!(
                "CFIE: speculative_generate refused — another speculative call is in flight (single-slot draft KV pool; serialize calls)"
            );
            return -1;
        }
        let _flight = SpecFlight;

        // Snapshot BOTH bindings under one lock (device pointers/host
        // embeds are stable for the call; the lock is released before
        // the loop so the per-launch engine locking is not re-entrant).
        let (
            weight_table,
            final_norm_dev,
            lm_head_dev,
            embed_host,
            n_layers,
            d_model,
            vocab_size,
            draft_embed,
            draft_n_layers,
            draft_d_model,
        ) = {
            let g = match engine().lock() {
                Ok(g) => g,
                Err(_) => return -1,
            };
            let (bm, dm) = match (g.bound_model.as_ref(), g.draft_model.as_ref()) {
                (Some(b), Some(d)) => (b, d),
                _ => return -1, // unbound between the guard and here
            };
            (
                bm.weight_table.clone(),
                bm.final_norm_dev,
                bm.lm_head_dev,
                bm.embed_host.clone(),
                bm.n_layers,
                bm.d_model,
                bm.vocab_size,
                dm.embed_host.clone(),
                dm.n_layers,
                dm.d_model,
            )
        };
        // Every kernel kind the loop drives must be finalized.
        if resolved(KIND_DECODE_BLOCK, 0).is_err()
            || resolved(KIND_FUSED_SAMPLE, 0).is_err()
            || resolved(KIND_DRAFT_BLOCK, 0).is_err()
            || resolved(KIND_DRAFT_SAMPLE, 0).is_err()
            || resolved(KIND_VERIFY_PROBS, 0).is_err()
            || resolved(KIND_SPEC_REJECT, 0).is_err()
        {
            eprintln!(
                "CFIE: speculative_generate refused — engine not finalized (kinds 1/2/4/6/7/8 unresolved)"
            );
            return -1;
        }
        // The draft KV pool must exist (kind-6 kv_base injection).
        {
            let g = match engine().lock() {
                Ok(g) => g,
                Err(_) => return -1,
            };
            if g.draft_pool_base == 0 {
                eprintln!(
                    "CFIE: speculative_generate refused — draft KV pool not allocated (call nsl_cfie_draft_pool_alloc)"
                );
                return -1;
            }
        }
        // Host-read overflow guards (Cycle-9 pattern) for the token
        // arrays, and the [K][vocab] f32 prob-matrix size.
        if checked_host_i64_bytes(prompt_len).is_none()
            || checked_host_i64_bytes(out_cap).is_none()
        {
            eprintln!("CFIE: speculative_generate refused — token array size overflows");
            return -1;
        }
        let probs_bytes = match (k_tokens as u64)
            .checked_mul(vocab_size as u64)
            .and_then(|x| x.checked_mul(4))
            .filter(|&b| b <= isize::MAX as u64)
        {
            Some(b) => b as usize,
            None => {
                eprintln!("CFIE: speculative_generate refused — K x vocab prob matrix overflows");
                return -1;
            }
        };

        let d = d_model as usize;
        let dd = draft_d_model as usize;
        let k = k_tokens as usize;
        let prompt = unsafe {
            std::slice::from_raw_parts(prompt_tokens_ptr as *const i64, prompt_len as usize)
        };
        let out = unsafe {
            std::slice::from_raw_parts_mut(out_tokens_ptr as *mut i64, out_cap as usize)
        };

        // Acquire a fresh TARGET KV slot; the draft side is slot 0 of
        // its own single-slot pool.
        let slot = crate::cfie::ffi::nsl_cfie_kv_slot_acquire();
        if slot < 0 {
            eprintln!("CFIE: speculative_generate refused — KV slot acquire failed (pool exhausted?)");
            return -1;
        }

        // Device scratch — freed on EVERY exit path via `cleanup`.
        let x_a = crate::cuda::inner::alloc_device(d * 4) as i64; // target hidden ping
        let x_b = crate::cuda::inner::alloc_device(d * 4) as i64; // target hidden pong
        let tok_dev = crate::cuda::inner::alloc_device(4) as i64; // kind-1 sampled u32
        let x_da = crate::cuda::inner::alloc_device(dd * 4) as i64; // draft hidden ping
        let x_db = crate::cuda::inner::alloc_device(dd * 4) as i64; // draft hidden pong
        let d_tokens_dev = crate::cuda::inner::alloc_device(k * 4) as i64; // draft tokens u32[K]
        let d_probs_dev = crate::cuda::inner::alloc_device(k * 4) as i64; // p_draft f32[K]
        let t_probs_dev = crate::cuda::inner::alloc_device(probs_bytes) as i64; // target rows f32[K][vocab]
        let rj_dev = crate::cuda::inner::alloc_device(8) as i64; // accepted u32 + correction u32

        let scratch = [
            x_a, x_b, tok_dev, x_da, x_db, d_tokens_dev, d_probs_dev, t_probs_dev, rj_dev,
        ];
        let cleanup = move || {
            for p in scratch.iter() {
                crate::cuda::inner::free_device(*p as *mut c_void);
            }
            crate::cfie::ffi::nsl_cfie_kv_slot_release(slot);
        };
        // Device u32 -> host readback (NULL-stream ordering serializes
        // behind the preceding launches, same as generate's readback).
        let read_u32 = |dev: i64| -> u32 {
            let mut v = [0u32; 1];
            crate::cuda::inner::memcpy_dtoh(v.as_mut_ptr() as *mut c_void, dev as *const c_void, 4);
            v[0]
        };
        // Record one emitted token; returns true when generation must
        // STOP (EOS emitted, or max_new reached).  Same clamp/count/EOS
        // ordering as nsl_cfie_generate.
        fn record(
            tok: i64,
            out: &mut [i64],
            generated: &mut i64,
            out_cap: i64,
            eos: i64,
            max_new: i64,
        ) -> bool {
            if *generated < out_cap {
                out[*generated as usize] = tok;
            }
            *generated += 1;
            tok == eos || *generated >= max_new
        }

        let mut generated: i64 = 0;

        // ---- Phase 1: TARGET prefill — byte-identical to
        // nsl_cfie_generate's first prompt_len decode_steps, so the
        // pre-speculation prefix agrees token-for-token.  The sampler
        // output only matters at the last position (the FIRST new
        // token, sampled at pos prompt_len-1).
        for pos in 0..prompt_len {
            let t = prompt[pos as usize];
            if t < 0 || t >= vocab_size {
                cleanup();
                eprintln!(
                    "CFIE: speculative_generate refused — prompt token id {t} out of range [0, {vocab_size})"
                );
                return -1;
            }
            gather_embed_row(&embed_host, d, t, x_a);
            let rc = nsl_cfie_decode_step(
                x_a,
                x_b,
                weight_table.as_ptr() as i64,
                n_layers,
                final_norm_dev as i64,
                lm_head_dev as i64,
                slot,
                pos,
                rng_seed,
                0, // grammar_state — DEFERRED, same as generate
                tok_dev,
            );
            if rc == -2 {
                // Prompt alone filled the slot: 0 new tokens, cleanly.
                cleanup();
                return 0;
            }
            if rc < 0 {
                cleanup();
                eprintln!("CFIE: speculative_generate — prefill decode_step failed with rc {rc}");
                return -1;
            }
        }
        let t0 = read_u32(tok_dev) as i64;
        if record(t0, out, &mut generated, out_cap, eos_token_id, max_new_tokens) {
            cleanup();
            return generated;
        }

        // ---- Phase 2: DRAFT prefill — kind-6 chain per prompt token,
        // outputs discarded (this only appends draft KV).  Prompt ids
        // were vocab-guarded in phase 1 (shared vocab).
        for pos in 0..prompt_len {
            gather_embed_row(&draft_embed, dd, prompt[pos as usize], x_da);
            if let Err(rc) = draft_chain(x_da, x_db, draft_n_layers, pos) {
                cleanup();
                eprintln!("CFIE: speculative_generate — draft prefill launch failed with rc {rc}");
                return -1;
            }
        }

        // Fed-token counters.  target_pos mirrors the slot allocator's
        // booked length EXACTLY (each verify/decode_step advances 1;
        // rejection rolls back `stale`); draft_pos is the host-side
        // equivalent for the single-slot draft pool.  Invariant at every
        // round boundary: target_pos == draft_pos == (fed tokens), and
        // the last emitted token has NOT yet been fed to either model.
        let mut target_pos: i64 = prompt_len;
        let mut draft_pos: i64 = prompt_len;
        let mut last = t0;
        let mut round: i64 = 0;

        'rounds: loop {
            // ---- Capacity probe (draft-KV OOB guard). ----
            // The draft phase below writes K draft-KV rows at
            // draft_pos..draft_pos+K BEFORE any advance-first booking,
            // and the kind-6 kernel has NO bounds check — so the round
            // may only start when the TARGET slot (whose booked length
            // equals draft_pos at every round boundary) can book K more
            // rows.  Probe by advancing K and rolling straight back
            // (both pure bookkeeping); with the single-flight guard the
            // slot is exclusively ours, so a passing probe guarantees
            // the verify bookings succeed.
            let probe = crate::cfie::ffi::nsl_cfie_kv_slot_advance(slot, k_tokens);
            if probe >= 0 {
                crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot, k_tokens);
            } else {
                // TAIL FALLBACK: fewer than K rows remain.  Finish with
                // plain per-token decode_steps — exactly what
                // nsl_cfie_generate runs from this state, so the
                // determinism contract holds all the way to the -2
                // capacity stop.  No draft sync needed: speculation
                // never resumes (capacity only shrinks from here).
                loop {
                    if last < 0 || last >= vocab_size {
                        cleanup();
                        eprintln!(
                            "CFIE: speculative_generate refused — tail input token id {last} out of range [0, {vocab_size})"
                        );
                        return -1;
                    }
                    gather_embed_row(&embed_host, d, last, x_a);
                    let rc = nsl_cfie_decode_step(
                        x_a,
                        x_b,
                        weight_table.as_ptr() as i64,
                        n_layers,
                        final_norm_dev as i64,
                        lm_head_dev as i64,
                        slot,
                        target_pos,
                        rng_seed,
                        0,
                        tok_dev,
                    );
                    if rc == -2 {
                        break 'rounds; // capacity — clean stop
                    }
                    if rc < 0 {
                        cleanup();
                        eprintln!(
                            "CFIE: speculative_generate — tail decode_step failed with rc {rc}"
                        );
                        return -1;
                    }
                    target_pos += 1;
                    let t = read_u32(tok_dev) as i64;
                    if t >= vocab_size {
                        cleanup();
                        eprintln!(
                            "CFIE: speculative_generate — sampler produced out-of-vocab token {t} (kernel contract violation)"
                        );
                        return -1;
                    }
                    if record(t, out, &mut generated, out_cap, eos_token_id, max_new_tokens) {
                        break 'rounds;
                    }
                    last = t;
                }
            }

            // ---- Draft K tokens greedily, recording p_draft[j]. ----
            let mut draft_toks: Vec<i64> = Vec::with_capacity(k);
            let mut prev = last;
            for j in 0..k {
                if prev < 0 || prev >= vocab_size {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate refused — draft input token id {prev} out of range [0, {vocab_size})"
                    );
                    return -1;
                }
                gather_embed_row(&draft_embed, dd, prev, x_da);
                let hidden = match draft_chain(x_da, x_db, draft_n_layers, draft_pos) {
                    Ok(h) => h,
                    Err(rc) => {
                        cleanup();
                        eprintln!(
                            "CFIE: speculative_generate — draft block launch failed with rc {rc}"
                        );
                        return -1;
                    }
                };
                let rc = nsl_cfie_launch_draft_sample(
                    hidden,
                    d_tokens_dev + (j as i64) * 4,
                    d_probs_dev + (j as i64) * 4,
                    rng_seed, // ABI symmetry — kind 7 is greedy, seed unused
                );
                if rc != 0 {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — draft sample launch failed with rc {rc}"
                    );
                    return -1;
                }
                draft_pos += 1;
                let dt = read_u32(d_tokens_dev + (j as i64) * 4) as i64;
                if dt >= vocab_size {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — draft sampler produced out-of-vocab token {dt} (kernel contract violation)"
                    );
                    return -1;
                }
                draft_toks.push(dt);
                prev = dt;
            }

            // ---- Target verify: kind-2 chain + kind-8 prob row per
            // position.  Row j is the target distribution over the
            // sequence slot draft_toks[j] occupies, so the INPUT is the
            // PREVIOUS token (row 0 feeds `last`).  Each step appends
            // target KV (advance-first, exactly like decode_step).
            for j in 0..k {
                let inp = if j == 0 { last } else { draft_toks[j - 1] };
                // (vocab-guarded when produced, on every build path)
                gather_embed_row(&embed_host, d, inp, x_a);
                let rc = target_verify_step(
                    x_a,
                    x_b,
                    &weight_table,
                    n_layers,
                    slot,
                    target_pos,
                    t_probs_dev + (j as i64) * vocab_size * 4,
                );
                if rc == -2 {
                    // Unreachable given the round-start capacity probe
                    // + single-flight (K rows were bookable); kept as
                    // defense-in-depth — a clean stop, never corruption.
                    break 'rounds;
                }
                if rc != 0 {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — verify step failed with rc {rc}"
                    );
                    return -1;
                }
                target_pos += 1;
            }

            // ---- ONE kind-4 rejection launch over the K rows.  The
            // seed varies per round (wrapping) so real draft/target
            // pairs do not replay identical uniforms; the
            // self-speculation invariant is seed-independent.
            let rc = nsl_cfie_launch_spec_reject(
                t_probs_dev,
                d_probs_dev,
                d_tokens_dev,
                rng_seed.wrapping_add(round),
                rj_dev,
                rj_dev + 4,
            );
            if rc != 0 {
                cleanup();
                eprintln!("CFIE: speculative_generate — reject launch failed with rc {rc}");
                return -1;
            }
            round += 1;
            let accepted = read_u32(rj_dev) as i64;
            let correction = read_u32(rj_dev + 4);
            if accepted > k_tokens {
                cleanup();
                eprintln!(
                    "CFIE: speculative_generate — reject kernel reported {accepted} accepted of {k_tokens} (kernel contract violation)"
                );
                return -1;
            }

            if correction == 0xFFFF_FFFF {
                // All K accepted (sentinel).  Emit the K drafts, then
                // ONE bonus token via a plain target step, drafting
                // d[K-1] through the draft chain to keep both KV sides
                // in lock-step.
                if accepted != k_tokens {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — all-accept sentinel with accepted {accepted} != K {k_tokens} (kernel contract violation)"
                    );
                    return -1;
                }
                for &t in &draft_toks {
                    if record(t, out, &mut generated, out_cap, eos_token_id, max_new_tokens) {
                        break 'rounds;
                    }
                }
                let inp = draft_toks[k - 1]; // vocab-guarded at draft time
                gather_embed_row(&embed_host, d, inp, x_a);
                let rc = nsl_cfie_decode_step(
                    x_a,
                    x_b,
                    weight_table.as_ptr() as i64,
                    n_layers,
                    final_norm_dev as i64,
                    lm_head_dev as i64,
                    slot,
                    target_pos,
                    rng_seed,
                    0,
                    tok_dev,
                );
                if rc == -2 {
                    break 'rounds;
                }
                if rc < 0 {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — bonus decode_step failed with rc {rc}"
                    );
                    return -1;
                }
                target_pos += 1;
                let bonus = read_u32(tok_dev) as i64;
                if bonus >= vocab_size {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — sampler produced out-of-vocab token {bonus} (kernel contract violation)"
                    );
                    return -1;
                }
                // Draft KV sync: feed d[K-1] (chain only — its sample is
                // never needed; the bonus came from the TARGET sampler).
                gather_embed_row(&draft_embed, dd, inp, x_da);
                if let Err(rc) = draft_chain(x_da, x_db, draft_n_layers, draft_pos) {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — draft sync launch failed with rc {rc}"
                    );
                    return -1;
                }
                draft_pos += 1;
                if record(bonus, out, &mut generated, out_cap, eos_token_id, max_new_tokens) {
                    break 'rounds;
                }
                last = bonus;
            } else {
                // Rejection at index `accepted`: emit the accepted
                // prefix, then the residual correction sample.
                if accepted >= k_tokens {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — correction with accepted {accepted} >= K {k_tokens} (kernel contract violation)"
                    );
                    return -1;
                }
                for &t in &draft_toks[..accepted as usize] {
                    if record(t, out, &mut generated, out_cap, eos_token_id, max_new_tokens) {
                        break 'rounds;
                    }
                }
                let corr = correction as i64;
                if corr >= vocab_size {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — reject kernel produced out-of-vocab correction {corr} (kernel contract violation)"
                    );
                    return -1;
                }
                // Roll back the STALE rows.  The round appended K rows
                // for inputs (last, d0..d[K-2]); the emitted sequence
                // keeps (last, d0..d[accepted-1]) = accepted+1 of them,
                // so K - accepted - 1 rows are stale on BOTH sides.  The
                // correction is fed at the next round boundary,
                // overwriting the first stale draft row by position —
                // that overwrite IS the draft-side rollback.
                let stale = k_tokens - accepted - 1;
                if stale > 0
                    && crate::cfie::ffi::nsl_cfie_kv_slot_rollback(slot, stale) < 0
                {
                    cleanup();
                    eprintln!(
                        "CFIE: speculative_generate — KV rollback of {stale} rejected rows failed"
                    );
                    return -1;
                }
                target_pos -= stale;
                draft_pos -= stale;
                if record(corr, out, &mut generated, out_cap, eos_token_id, max_new_tokens) {
                    break 'rounds;
                }
                last = corr;
            }
        }

        cleanup();
        generated
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (eos_token_id, rng_seed);
        eprintln!(
            "CFIE: speculative_generate requires a CUDA-enabled build and GPU — refusing (no generation)"
        );
        -1
    }
}

// ---------------------------------------------------------------------------
// Tests (CPU-only — run without the cuda feature)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::MutexGuard;

    // These tests share the global ENGINE and GLOBAL_KV_SLOTS with the
    // kv_slot_tests in ffi.rs — every module mutating that state must
    // serialize under the SAME crate-wide lock (separate locks let
    // filtered `cargo test --lib cfie` runs interleave and flake).
    fn engine_serial_lock() -> MutexGuard<'static, ()> {
        crate::cfie::test_serial_lock()
    }

    fn reset_engine() {
        assert_eq!(nsl_cfie_engine_destroy(), 0);
    }

    fn deinit_kv_slots() {
        *kv_slots_global().lock().unwrap_or_else(|e| e.into_inner()) = None;
    }

    const PTX: &[u8] = b".version 7.0 // fake ptx for registry tests";
    const NAME: &[u8] = b"nsl_cfie_test_kernel";

    fn register(kind: i64, layer: i64) -> i64 {
        nsl_cfie_register_kernel(
            kind,
            layer,
            PTX.as_ptr() as i64,
            PTX.len() as i64,
            NAME.as_ptr() as i64,
            NAME.len() as i64,
            4,
            128,
            0,
        )
    }

    fn kernel_count() -> usize {
        engine()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .kernels
            .len()
    }

    #[test]
    fn register_validates_kind_range() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(register(-1, 0), -1);
        assert_eq!(register(9, 0), -1); // one past the Cycle-13 range
        assert_eq!(register(0, -1), -1); // negative layer index
        for kind in 0..=8 {
            assert_eq!(register(kind, 0), 0, "kind {} must register", kind);
        }
        assert_eq!(kernel_count(), 9);
        reset_engine();
    }

    #[test]
    fn register_validates_null_and_len_args() {
        let _serial = engine_serial_lock();
        reset_engine();
        let p = PTX.as_ptr() as i64;
        let pl = PTX.len() as i64;
        let n = NAME.as_ptr() as i64;
        let nl = NAME.len() as i64;
        assert_eq!(nsl_cfie_register_kernel(0, 0, 0, pl, n, nl, 1, 1, 0), -1); // null ptx
        assert_eq!(nsl_cfie_register_kernel(0, 0, p, 0, n, nl, 1, 1, 0), -1); // zero ptx_len
        assert_eq!(nsl_cfie_register_kernel(0, 0, p, -4, n, nl, 1, 1, 0), -1); // negative ptx_len
        assert_eq!(nsl_cfie_register_kernel(0, 0, p, pl, 0, nl, 1, 1, 0), -1); // null name
        assert_eq!(nsl_cfie_register_kernel(0, 0, p, pl, n, 0, 1, 1, 0), -1); // zero name_len
        assert_eq!(nsl_cfie_register_kernel(0, 0, p, pl, n, nl, 0, 1, 0), -1); // zero grid_x
        assert_eq!(nsl_cfie_register_kernel(0, 0, p, pl, n, nl, 1, 0, 0), -1); // zero block_x
        assert_eq!(nsl_cfie_register_kernel(0, 0, p, pl, n, nl, 1, 1, -1), -1); // negative smem
        // Interior NUL in the name would truncate the function lookup.
        let bad_name = b"nsl\0bad";
        assert_eq!(
            nsl_cfie_register_kernel(
                0,
                0,
                p,
                pl,
                bad_name.as_ptr() as i64,
                bad_name.len() as i64,
                1,
                1,
                0
            ),
            -1
        );
        assert_eq!(kernel_count(), 0, "no bad registration may be recorded");
        reset_engine();
    }

    #[test]
    fn register_copies_ptx_and_appends_nul() {
        let _serial = engine_serial_lock();
        reset_engine();
        // Register from a scratch buffer, then clobber it — the engine
        // must have copied the bytes (and appended its own NUL).
        let mut scratch = PTX.to_vec();
        assert_eq!(
            nsl_cfie_register_kernel(
                2,
                0,
                scratch.as_ptr() as i64,
                scratch.len() as i64,
                NAME.as_ptr() as i64,
                NAME.len() as i64,
                4,
                128,
                0
            ),
            0
        );
        for b in scratch.iter_mut() {
            *b = 0xAA;
        }
        let g = engine().lock().unwrap_or_else(|e| e.into_inner());
        let k = g.kernels.get(&(2, 0)).expect("registered");
        assert_eq!(&k.ptx[..PTX.len()], PTX, "ptx bytes must be an owned copy");
        assert_eq!(k.ptx.last(), Some(&0u8), "runtime appends the trailing NUL");
        assert_eq!(k.ptx.len(), PTX.len() + 1);
        assert_eq!(k.name.to_bytes(), NAME);
        assert_eq!((k.grid_x, k.block_x, k.smem_dyn_bytes), (4, 128, 0));
        drop(g);
        reset_engine();
    }

    #[test]
    fn duplicate_kind_layer_replaces_previous() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(register(5, 3), 0);
        assert_eq!(kernel_count(), 1);
        // Same (kind, layer), different meta — must replace, not add.
        let other_name = b"nsl_cfie_replacement";
        assert_eq!(
            nsl_cfie_register_kernel(
                5,
                3,
                PTX.as_ptr() as i64,
                PTX.len() as i64,
                other_name.as_ptr() as i64,
                other_name.len() as i64,
                8,
                256,
                0
            ),
            0
        );
        assert_eq!(kernel_count(), 1);
        let g = engine().lock().unwrap_or_else(|e| e.into_inner());
        let k = g.kernels.get(&(5, 3)).expect("registered");
        assert_eq!(k.name.to_bytes(), other_name);
        assert_eq!((k.grid_x, k.block_x), (8, 256));
        drop(g);
        // Distinct layer under the same kind is a separate registration.
        assert_eq!(register(5, 4), 0);
        assert_eq!(kernel_count(), 2);
        reset_engine();
    }

    #[test]
    fn finalize_refuses_with_zero_registrations() {
        let _serial = engine_serial_lock();
        reset_engine();
        // Zero registrations => -1 on every build; on non-cuda builds
        // finalize refuses unconditionally.
        assert_eq!(nsl_cfie_engine_finalize(), -1);
        reset_engine();
    }

    #[test]
    fn launches_return_error_before_finalize() {
        let _serial = engine_serial_lock();
        reset_engine();
        // Register every kind, but never finalize: all launches refuse.
        for kind in 0..=8 {
            assert_eq!(register(kind, 0), 0);
        }
        assert_eq!(nsl_cfie_launch_decode_attn(0x10, 0x20, 0, 0, 1), -1);
        assert_eq!(nsl_cfie_launch_fused_sample(0x10, 0x20, 0x30, 0x40, 7, 0), -1);
        assert_eq!(
            nsl_cfie_launch_decode_block(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0),
            -1
        );
        assert_eq!(nsl_cfie_launch_spec_verify(0x10, 0x20, 0, 0, 1), -1);
        assert_eq!(nsl_cfie_launch_spec_reject(1, 2, 3, 4, 5, 6), -1);
        assert_eq!(nsl_cfie_launch_quant_attn(0, 0x10, 0x20, 0, 1, 0, 0), -1);
        // Cycle-13 draft/verify launches refuse before finalize too.
        assert_eq!(nsl_cfie_launch_draft_block(0x10, 0x20, 0, 0), -1);
        assert_eq!(nsl_cfie_launch_draft_sample(0x10, 0x20, 0x30, 7), -1);
        assert_eq!(nsl_cfie_launch_verify_probs(0x10, 0x20), -1);
        reset_engine();
    }

    #[test]
    fn destroy_clears_state_for_reregistration() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(register(0, 0), 0);
        assert_eq!(register(1, 0), 0);
        assert_eq!(kernel_count(), 2);
        assert_eq!(nsl_cfie_engine_destroy(), 0);
        assert_eq!(kernel_count(), 0);
        {
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(!g.finalized);
            assert_eq!(g.grammar_mask_dev, 0);
            assert_eq!(g.pool_base, 0);
        }
        // A fresh register + finalize cycle must be able to run.  (On
        // this CPU-only build finalize still refuses with -1 — the
        // point is that destroy left no stale state behind, so the
        // cycle behaves exactly like a first run.)
        assert_eq!(register(1, 0), 0);
        assert_eq!(kernel_count(), 1);
        let rc = nsl_cfie_engine_finalize();
        #[cfg(not(feature = "cuda"))]
        assert_eq!(rc, -1);
        #[cfg(feature = "cuda")]
        let _ = rc; // GPU builds resolve for real; count depends on the driver
        reset_engine();
    }

    #[test]
    fn destroy_detaches_kv_device_buffer() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(crate::cfie::ffi::nsl_cfie_kv_slots_init(2, 8), 0);
        assert_eq!(crate::cfie::ffi::nsl_cfie_kv_attach_device(0x9000, 4096), 0);
        assert_eq!(nsl_cfie_engine_destroy(), 0);
        {
            let kv = kv_slots_global().lock().unwrap_or_else(|e| e.into_inner());
            let a = kv.as_ref().expect("allocator still initialized");
            assert_eq!(a.device_base(), 0, "destroy must detach the device buffer");
            assert_eq!(a.device_bytes(), 0);
        }
        deinit_kv_slots();
        reset_engine();
    }

    #[test]
    fn pool_alloc_refuses_before_kv_slots_init() {
        let _serial = engine_serial_lock();
        reset_engine();
        deinit_kv_slots();
        // Non-cuda builds refuse unconditionally; cuda builds must also
        // refuse while the slot allocator is uninitialized.
        assert_eq!(nsl_cfie_kv_pool_alloc(4096), -1);
        assert_eq!(nsl_cfie_kv_pool_alloc(0), -1);
        assert_eq!(nsl_cfie_kv_pool_alloc(-1), -1);
        reset_engine();
    }

    #[test]
    fn pool_alloc_refuses_double_alloc() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(crate::cfie::ffi::nsl_cfie_kv_slots_init(2, 8), 0);
        #[cfg(not(feature = "cuda"))]
        {
            // No GPU: honest refusal both times.
            assert_eq!(nsl_cfie_kv_pool_alloc(4096), -1);
            assert_eq!(nsl_cfie_kv_pool_alloc(4096), -1);
        }
        #[cfg(feature = "cuda")]
        {
            // GPU build: first alloc succeeds, second refuses until
            // destroy releases the pool.
            assert_eq!(nsl_cfie_kv_pool_alloc(4096), 0);
            assert_eq!(nsl_cfie_kv_pool_alloc(4096), -1);
            assert_eq!(nsl_cfie_engine_destroy(), 0);
            assert_eq!(nsl_cfie_kv_pool_alloc(4096), 0);
        }
        deinit_kv_slots();
        reset_engine();
    }

    #[test]
    fn decode_step_validates_args_and_books_slots_first() {
        let _serial = engine_serial_lock();
        reset_engine();
        deinit_kv_slots();

        let weights = [0u64; 2 * 9]; // 2 layers x 9 device pointers
        let wp = weights.as_ptr() as i64;

        // Null weights / non-positive layer count refuse up front.
        assert_eq!(nsl_cfie_decode_step(1, 2, 0, 2, 3, 4, 0, 0, 7, 0, 5), -1);
        assert_eq!(nsl_cfie_decode_step(1, 2, wp, 0, 3, 4, 0, 0, 7, 0, 5), -1);

        // Uninitialized slot allocator => advance refuses => -2.
        assert_eq!(nsl_cfie_decode_step(1, 2, wp, 2, 3, 4, 0, 0, 7, 0, 5), -2);

        // Active slot but caller pos disagreeing with the booked length
        // => -3, and the booking must have been rolled back.
        assert_eq!(crate::cfie::ffi::nsl_cfie_kv_slots_init(1, 4), 0);
        let slot = crate::cfie::ffi::nsl_cfie_kv_slot_acquire();
        assert_eq!(slot, 0);
        assert_eq!(
            nsl_cfie_decode_step(1, 2, wp, 2, 3, 4, slot, 5, 7, 0, 5),
            -3,
            "pos=5 against an empty slot is a mismatch"
        );
        {
            let kv = kv_slots_global().lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                kv.as_ref().unwrap().seq_len(0),
                Some(0),
                "-3 path must roll the advance back"
            );
        }

        // pos consistent but engine never finalized: the first
        // decode_block launch refuses (-1) and the booking rolls back.
        assert_eq!(nsl_cfie_decode_step(1, 2, wp, 2, 3, 4, slot, 0, 7, 0, 5), -1);
        {
            let kv = kv_slots_global().lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(
                kv.as_ref().unwrap().seq_len(0),
                Some(0),
                "launch-failure path must roll the advance back"
            );
        }

        // Slot at capacity => -2 without touching the length.
        assert_eq!(crate::cfie::ffi::nsl_cfie_kv_slot_advance(slot, 4), 4);
        assert_eq!(nsl_cfie_decode_step(1, 2, wp, 2, 3, 4, slot, 4, 7, 0, 5), -2);
        {
            let kv = kv_slots_global().lock().unwrap_or_else(|e| e.into_inner());
            assert_eq!(kv.as_ref().unwrap().seq_len(0), Some(4));
        }

        deinit_kv_slots();
        reset_engine();
    }

    #[test]
    fn pool_base_accessor_returns_zero_when_unallocated() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(nsl_cfie_kv_pool_base(), 0);
        reset_engine();
    }

    #[test]
    fn destroy_is_idempotent_and_always_zero() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(nsl_cfie_engine_destroy(), 0);
        assert_eq!(nsl_cfie_engine_destroy(), 0);
        assert_eq!(register(4, 0), 0);
        assert_eq!(nsl_cfie_engine_destroy(), 0);
        assert_eq!(kernel_count(), 0);
    }

    // -----------------------------------------------------------------
    // CFIE Cycle 9 weight binding
    // -----------------------------------------------------------------

    /// The f32 -> f16 cast used by `nsl_cfie_upload_weight_f16` delegates
    /// to `half::f16::from_f32(x).to_bits()`.  Pin its round-to-nearest-
    /// even behavior against known-correct half bit patterns so a future
    /// `half` bump (or an accidental swap to a hand-rolled converter)
    /// cannot silently change the device weights.  This runs on every
    /// build (the helper itself is cuda-gated; `half` is not).
    #[test]
    fn f32_to_f16_cast_is_correct_rne() {
        let case = |x: f32| half::f16::from_f32(x).to_bits();
        // Exact representables.
        assert_eq!(case(1.0), 0x3C00, "1.0");
        assert_eq!(case(0.5), 0x3800, "0.5");
        assert_eq!(case(-0.0), 0x8000, "-0.0 keeps the sign bit, mantissa 0");
        assert_eq!(case(0.0), 0x0000, "+0.0");
        assert_eq!(case(2.0), 0x4000, "2.0");
        // A subnormal f16: 2^-24 is the smallest positive subnormal
        // (bits 0x0001); RNE of exactly 2^-24 lands on it.
        assert_eq!(
            case(2f32.powi(-24)),
            0x0001,
            "smallest positive f16 subnormal 2^-24"
        );
        // Overflow: 70000 > f16 max (65504) rounds to +inf.
        assert_eq!(case(70000.0), 0x7C00, "overflow to +inf");
        assert_eq!(case(-70000.0), 0xFC00, "overflow to -inf");
        // A mid value that is NOT exactly representable: 0.1 -> nearest
        // half is 0x2E66 (0.0999755859375), the documented RNE result.
        assert_eq!(case(0.1), 0x2E66, "0.1 rounds to nearest even half");
        // NaN stays NaN (exponent all ones, non-zero mantissa).
        let nan = case(f32::NAN);
        assert_eq!(nan & 0x7C00, 0x7C00, "NaN exponent all ones");
        assert_ne!(nan & 0x03FF, 0, "NaN mantissa non-zero");
        // Round-trip sanity: casting the half bits back to f32 is close.
        assert!((half::f16::from_bits(case(0.1)).to_f32() - 0.1).abs() < 1e-3);
    }

    #[test]
    fn weights_reset_idempotent_and_zero_with_nothing_uploaded() {
        let _serial = engine_serial_lock();
        reset_engine();
        // Never uploaded anything: reset is a no-op returning 0, twice.
        assert_eq!(nsl_cfie_weights_reset(), 0);
        assert_eq!(nsl_cfie_weights_reset(), 0);
        {
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(g.weight_allocs.is_empty(), "no allocations recorded");
        }
        reset_engine();
    }

    #[test]
    fn uploads_refuse_on_bad_args_and_noncuda() {
        let _serial = engine_serial_lock();
        reset_engine();
        // A real host f32 buffer so the ptr is non-null; on non-cuda
        // builds the upload refuses before touching it, and on cuda
        // builds the bad-arg guards fire before any device work.
        let host = [1.0f32, 2.0, 3.0, 4.0];
        let hp = host.as_ptr() as i64;

        // Bad args: null host ptr, zero and negative n_elems.
        assert_eq!(nsl_cfie_upload_weight_f16(0, 4), -1, "null host ptr");
        assert_eq!(nsl_cfie_upload_weight_f16(hp, 0), -1, "zero n_elems");
        assert_eq!(nsl_cfie_upload_weight_f16(hp, -1), -1, "negative n_elems");
        assert_eq!(nsl_cfie_upload_weight_f32(0, 4), -1, "null host ptr (f32)");
        assert_eq!(nsl_cfie_upload_weight_f32(hp, 0), -1, "zero n_elems (f32)");
        assert_eq!(nsl_cfie_upload_weight_f32(hp, -1), -1, "negative n_elems (f32)");

        // Host read size (n_elems * 4) must fit isize::MAX, else the
        // internal slice/memcpy is UB (a process ABORT, not a panic).
        // i64::MAX overflows u64 * 4 outright:
        assert_eq!(
            nsl_cfie_upload_weight_f16(hp, i64::MAX),
            -1,
            "i64::MAX host-bytes overflow (f16)"
        );
        assert_eq!(
            nsl_cfie_upload_weight_f32(hp, i64::MAX),
            -1,
            "i64::MAX host-bytes overflow (f32)"
        );
        // The dangerous band: n_elems where n_elems*4 does NOT overflow
        // u64 but DOES exceed isize::MAX.  The old guard (device bytes vs
        // usize::MAX) passed this and reached from_raw_parts -> abort; the
        // host-bytes guard must refuse with -1 without aborting.
        let band = (isize::MAX as i64) / 4 + 1;
        assert_eq!(
            nsl_cfie_upload_weight_f16(hp, band),
            -1,
            "host bytes > isize::MAX must refuse, not abort (f16)"
        );
        assert_eq!(
            nsl_cfie_upload_weight_f32(hp, band),
            -1,
            "host bytes > isize::MAX must refuse, not abort (f32)"
        );

        #[cfg(not(feature = "cuda"))]
        {
            // Non-cuda build: even a valid request refuses (no GPU).
            assert_eq!(nsl_cfie_upload_weight_f16(hp, 4), -1, "no cuda => -1");
            assert_eq!(nsl_cfie_upload_weight_f32(hp, 4), -1, "no cuda => -1");
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(g.weight_allocs.is_empty(), "non-cuda never records");
        }
        reset_engine();
    }

    // -----------------------------------------------------------------
    // CFIE Cycle 10 model binding + generation driver
    // -----------------------------------------------------------------

    #[test]
    fn generate_reset_is_idempotent_and_zero() {
        let _serial = engine_serial_lock();
        reset_engine();
        // No binding yet: reset is a no-op returning 0, twice.
        assert_eq!(nsl_cfie_generate_reset(), 0);
        assert_eq!(nsl_cfie_generate_reset(), 0);
        {
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(g.bound_model.is_none(), "no binding recorded");
        }
        reset_engine();
    }

    #[test]
    fn bind_model_validates_args_and_refuses_noncuda() {
        let _serial = engine_serial_lock();
        reset_engine();
        // Null model handle refuses on every build (guard fires before any
        // device work / any weight resolution).
        assert_eq!(
            nsl_cfie_bind_model(0, 2, 64, 2, 1, 32, 128, 128),
            -1,
            "null model handle must refuse"
        );
        // A non-null handle with a non-positive dimension refuses on cuda
        // builds via the shape guard; on non-cuda builds the whole FFI
        // refuses regardless.  Use a bogus (never-dereferenced on these
        // paths) handle value.
        let bogus = 0x1000i64;
        assert_eq!(
            nsl_cfie_bind_model(bogus, 0, 64, 2, 1, 32, 128, 128),
            -1,
            "non-positive n_layers must refuse"
        );
        assert_eq!(
            nsl_cfie_bind_model(bogus, 2, 0, 2, 1, 32, 128, 128),
            -1,
            "non-positive d_model must refuse"
        );
        #[cfg(not(feature = "cuda"))]
        {
            // Non-cuda: even a fully valid-looking request refuses (no GPU)
            // BEFORE dereferencing the handle.
            assert_eq!(
                nsl_cfie_bind_model(bogus, 2, 64, 2, 1, 32, 128, 128),
                -1,
                "no cuda => -1"
            );
        }
        // No binding may have been recorded by any refusal.
        {
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(g.bound_model.is_none(), "a refused bind must record nothing");
        }
        reset_engine();
    }

    #[test]
    fn generate_validates_args_and_refuses_before_bind() {
        let _serial = engine_serial_lock();
        reset_engine();
        // A real host prompt + output buffer so the ptrs are non-null; the
        // arg guards / no-binding refusal fire before any device work.
        let prompt = [1i64, 2, 3];
        let pp = prompt.as_ptr() as i64;
        let mut out = [0i64; 4];
        let op = out.as_mut_ptr() as i64;

        // Bad args refuse (-1): null prompt ptr, null out ptr, non-positive
        // prompt_len / out_cap / max_new_tokens.
        assert_eq!(nsl_cfie_generate(0, 3, 4, 99, 7, op, 4), -1, "null prompt ptr");
        assert_eq!(nsl_cfie_generate(pp, 3, 4, 99, 7, 0, 4), -1, "null out ptr");
        assert_eq!(nsl_cfie_generate(pp, 0, 4, 99, 7, op, 4), -1, "zero prompt_len");
        assert_eq!(nsl_cfie_generate(pp, -1, 4, 99, 7, op, 4), -1, "neg prompt_len");
        assert_eq!(nsl_cfie_generate(pp, 3, 4, 99, 7, op, 0), -1, "zero out_cap");
        assert_eq!(nsl_cfie_generate(pp, 3, 0, 99, 7, op, 4), -1, "zero max_new_tokens");

        // Valid args but NO model bound (and on cuda, not finalized): must
        // refuse -1 before touching a KV slot.  On non-cuda the whole FFI
        // refuses regardless.
        assert_eq!(
            nsl_cfie_generate(pp, 3, 4, 99, 7, op, 4),
            -1,
            "generate must refuse before bind_model + finalize"
        );
        reset_engine();
    }

    // -----------------------------------------------------------------
    // CFIE Cycle 13 draft binding + speculative generation
    // -----------------------------------------------------------------

    /// A fake binding record for the CROSS-BUILD guard paths (binding
    /// order, vocab agreement).  Empty tables/embeds are fine: every
    /// test that injects one only exercises refusals that fire BEFORE
    /// any table/embed access (non-cuda refuses outright; cuda refuses
    /// at the not-finalized check).
    fn fake_binding(vocab: i64) -> BoundModel {
        BoundModel {
            weight_table: Vec::new(),
            final_norm_dev: 0,
            lm_head_dev: 0,
            embed_host: Vec::new(),
            n_layers: 1,
            d_model: 8,
            vocab_size: vocab,
        }
    }

    fn inject_fake_target(vocab: i64) {
        engine()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .bound_model = Some(fake_binding(vocab));
    }

    fn inject_fake_draft(vocab: i64) {
        engine()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .draft_model = Some(fake_binding(vocab));
    }

    #[test]
    fn bind_draft_refuses_bad_args_ordering_and_vocab_mismatch() {
        let _serial = engine_serial_lock();
        reset_engine();
        let bogus = 0x1000i64; // never dereferenced on these paths
        // Null handle / non-positive dims refuse on every build.
        assert_eq!(nsl_cfie_bind_draft_model(0, 1, 8, 1, 1, 8, 16, 32), -1);
        assert_eq!(
            nsl_cfie_bind_draft_model(bogus, 0, 8, 1, 1, 8, 16, 32),
            -1,
            "non-positive n_layers"
        );
        assert_eq!(
            nsl_cfie_bind_draft_model(bogus, 1, 8, 1, 1, 8, 16, 0),
            -1,
            "non-positive vocab"
        );
        // Ordering: no TARGET model bound yet => refuse on every build.
        assert_eq!(
            nsl_cfie_bind_draft_model(bogus, 1, 8, 1, 1, 8, 16, 32),
            -1,
            "bind_draft before bind_model must refuse"
        );
        // Shared-vocab invariant: target vocab 32, draft vocab 64.
        inject_fake_target(32);
        assert_eq!(
            nsl_cfie_bind_draft_model(bogus, 1, 8, 1, 1, 8, 16, 64),
            -1,
            "vocab mismatch must refuse"
        );
        #[cfg(not(feature = "cuda"))]
        {
            // Matching vocab, non-cuda: honest refusal (never touches
            // the bogus handle).
            assert_eq!(
                nsl_cfie_bind_draft_model(bogus, 1, 8, 1, 1, 8, 16, 32),
                -1,
                "no cuda => -1"
            );
        }
        // No draft binding may have been recorded by any refusal.
        {
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(g.draft_model.is_none(), "a refused draft bind records nothing");
        }
        reset_engine();
    }

    #[test]
    fn draft_pool_alloc_refuses_bad_args_and_double_alloc() {
        let _serial = engine_serial_lock();
        reset_engine();
        assert_eq!(nsl_cfie_draft_pool_alloc(0), -1);
        assert_eq!(nsl_cfie_draft_pool_alloc(-4096), -1);
        assert_eq!(nsl_cfie_draft_pool_base(), 0, "no pool recorded");
        #[cfg(not(feature = "cuda"))]
        {
            // No GPU: honest refusal for a valid size too.
            assert_eq!(nsl_cfie_draft_pool_alloc(4096), -1);
            assert_eq!(nsl_cfie_draft_pool_base(), 0);
        }
        #[cfg(feature = "cuda")]
        {
            // GPU build: first alloc succeeds, second refuses until
            // draft_reset (or destroy) releases the pool.
            assert_eq!(nsl_cfie_draft_pool_alloc(4096), 0);
            assert_ne!(nsl_cfie_draft_pool_base(), 0);
            assert_eq!(nsl_cfie_draft_pool_alloc(4096), -1);
            assert_eq!(nsl_cfie_draft_reset(), 0);
            assert_eq!(nsl_cfie_draft_pool_base(), 0);
            assert_eq!(nsl_cfie_draft_pool_alloc(4096), 0);
        }
        reset_engine();
    }

    #[test]
    fn draft_reset_is_idempotent_and_clears_binding() {
        let _serial = engine_serial_lock();
        reset_engine();
        // Nothing bound: reset is a no-op returning 0, twice.
        assert_eq!(nsl_cfie_draft_reset(), 0);
        assert_eq!(nsl_cfie_draft_reset(), 0);
        // With a (fake) draft binding: reset clears it, target untouched.
        inject_fake_target(32);
        inject_fake_draft(32);
        assert_eq!(nsl_cfie_draft_reset(), 0);
        {
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(g.draft_model.is_none(), "draft binding cleared");
            assert_eq!(g.draft_pool_base, 0);
            assert_eq!(g.draft_pool_bytes, 0);
            assert!(g.bound_model.is_some(), "target binding must survive draft_reset");
        }
        reset_engine();
    }

    #[test]
    fn destroy_clears_draft_state() {
        let _serial = engine_serial_lock();
        reset_engine();
        inject_fake_target(32);
        inject_fake_draft(32);
        assert_eq!(nsl_cfie_engine_destroy(), 0);
        {
            let g = engine().lock().unwrap_or_else(|e| e.into_inner());
            assert!(g.bound_model.is_none());
            assert!(g.draft_model.is_none(), "destroy must clear the draft binding");
            assert_eq!(g.draft_pool_base, 0);
            assert_eq!(g.draft_pool_bytes, 0);
        }
        reset_engine();
    }

    #[test]
    fn speculative_generate_validates_args_and_binding_order() {
        let _serial = engine_serial_lock();
        reset_engine();
        let prompt = [1i64, 2, 3];
        let pp = prompt.as_ptr() as i64;
        let mut out = [0i64; 8];
        let op = out.as_mut_ptr() as i64;

        // Bad args refuse (-1) on every build.
        assert_eq!(nsl_cfie_speculative_generate(0, 3, 4, 99, 7, 4, op, 8), -1, "null prompt");
        assert_eq!(nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 4, 0, 8), -1, "null out");
        assert_eq!(nsl_cfie_speculative_generate(pp, 0, 4, 99, 7, 4, op, 8), -1, "zero prompt_len");
        assert_eq!(nsl_cfie_speculative_generate(pp, -1, 4, 99, 7, 4, op, 8), -1, "neg prompt_len");
        assert_eq!(nsl_cfie_speculative_generate(pp, 3, 0, 99, 7, 4, op, 8), -1, "zero max_new");
        assert_eq!(nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 4, op, 0), -1, "zero out_cap");
        // k_tokens outside the frozen 1..=32 range refuses.
        assert_eq!(nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 0, op, 8), -1, "k = 0");
        assert_eq!(nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, -3, op, 8), -1, "k < 0");
        assert_eq!(nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 33, op, 8), -1, "k > 32");

        // Binding order: no target model bound.
        assert_eq!(
            nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 4, op, 8),
            -1,
            "must refuse before bind_model"
        );
        // Target bound but NO draft model bound.
        inject_fake_target(32);
        assert_eq!(
            nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 4, op, 8),
            -1,
            "must refuse before bind_draft_model"
        );
        // Both bound but vocab disagreement (a target re-bind after the
        // draft bind can cause this) refuses.
        inject_fake_draft(64);
        assert_eq!(
            nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 4, op, 8),
            -1,
            "vocab mismatch must refuse"
        );
        // Both bound + matching: on non-cuda the FFI refuses (no GPU);
        // on cuda builds it refuses at the not-finalized check — either
        // way -1 with no device work and no KV slot touched.
        inject_fake_draft(32);
        assert_eq!(
            nsl_cfie_speculative_generate(pp, 3, 4, 99, 7, 4, op, 8),
            -1,
            "must refuse before finalize"
        );
        // Nothing was written to the output buffer by any refusal.
        assert_eq!(out, [0i64; 8]);
        reset_engine();
    }
}
