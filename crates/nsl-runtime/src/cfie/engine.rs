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
//! 3=spec_verify, 4=spec_reject, 5=quant_attn.  `layer_idx` is
//! meaningful ONLY for kind 5 (the quant emitter bakes the layer into
//! the PTX, so there is one registration per layer); all other kinds
//! register with layer_idx 0.
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
        | KIND_SPEC_REJECT | KIND_QUANT_ATTN => {}
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
    }
    g.pool_base = 0;
    g.pool_bytes = 0;
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
        assert_eq!(register(6, 0), -1);
        assert_eq!(register(0, -1), -1); // negative layer index
        for kind in 0..=5 {
            assert_eq!(register(kind, 0), 0, "kind {} must register", kind);
        }
        assert_eq!(kernel_count(), 6);
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
        for kind in 0..=5 {
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
}
