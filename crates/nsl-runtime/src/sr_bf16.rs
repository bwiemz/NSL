//! P4 item 17: SR-BF16 authoritative weights — counter-based stochastic
//! rounding core (CPU reference implementation).
//!
//! The authoritative copy of every trained parameter is a device-resident
//! BF16 buffer (no FP32 master). The optimizer computes each update in f32
//! and rounds the result to bf16 STOCHASTICALLY, so quantization error is
//! zero-mean instead of systematically biased the way round-to-nearest is —
//! the property that makes master-copy-free bf16 training converge.
//!
//! ## Counter-based randomness (compiler-owned, deterministic)
//!
//! The dither for element `i` of parameter `p` at optimizer step `t` is a
//! pure function `mix64(seed ^ (t · STEP_SALT), param_base(p) + i)` — no
//! RNG state, no launch-order dependence. Codegen assigns each parameter a
//! stable `param_base` (its index in the train block's canonical param list
//! `<< 40`, far above any element count), the runtime threads the CLI
//! `--seed` and the step counter, and the SAME function runs on CPU (here)
//! and in PTX (`FASE_FUSED_ADAMW_STEP_BF16SR_PTX`), bit-for-bit. Reruns with
//! equal (seed, step, param, element) produce identical weights on every
//! rank and every backend.
//!
//! ## Rounding rule (bit-dither truncation)
//!
//! bf16 is the top 16 bits of an f32. For a finite f32 value `x`:
//!
//!   out = truncate_low16( bits(x) + (rand & 0xFFFF) )
//!
//! Adding a uniform 16-bit integer to the raw bits then truncating rounds
//! DOWN (in magnitude) with probability `1 - frac` and UP with probability
//! `frac`, where `frac` is the truncated remainder / ulp — exactly
//! stochastic rounding, including across the subnormal/normal boundary
//! (IEEE-754 bit patterns of one sign are ordered by magnitude, and the
//! carry out of the mantissa lands in the exponent precisely at the binade
//! step). Sign-magnitude layout means the dither always moves AWAY from
//! zero, matching truncation-toward-zero: E[out] == x.
//!
//! ## Explicit overflow / underflow behavior
//!
//!   * **Rounding-induced overflow** (finite f32 whose dithered bits would
//!     carry into exponent 0xFF — only reachable in the largest normal
//!     binade): SATURATE to ±BF16_MAX_NORMAL (0x7F7F). A finite update must
//!     never become Inf because of the rounding dice.
//!   * **Arithmetic ±Inf** (the f32 update itself overflowed): PROPAGATE as
//!     bf16 ±Inf (0x7F80). Divergence must stay loud, not be masked by
//!     saturation.
//!   * **NaN**: PROPAGATE as the canonical quiet bf16 NaN (0x7FC0 | sign).
//!     Truncating an arbitrary NaN's mantissa could zero its payload's top
//!     bits and mint an Inf — forced to a fixed qNaN instead.
//!   * **Underflow**: gradual. Subnormal bf16 results fall out of the same
//!     bit arithmetic; values whose dithered magnitude truncates to 0 round
//!     to ±0 with the correct probability.

/// Salt folded into the step counter so (seed, step) pairs decorrelate even
/// for adjacent steps. Odd constant from the splitmix64 increment.
pub const SR_STEP_SALT: u64 = 0x9E37_79B9_7F4A_7C15;

/// Stable per-parameter counter stride: parameter index `<< 40` leaves room
/// for 2^40 elements per parameter — far beyond any real tensor — so
/// (param, element) counters never collide across parameters.
pub const SR_PARAM_SHIFT: u32 = 40;

/// splitmix64 finalizer — the same three multiply/xorshift rounds the PTX
/// kernel performs. Stateless: mixes (key, counter) into 64 uniform bits.
#[inline]
pub fn sr_mix64(key: u64, counter: u64) -> u64 {
    let mut z = key.wrapping_add(counter.wrapping_mul(SR_STEP_SALT));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The 16-bit dither for (seed, step, param_base + element).
#[inline]
pub fn sr_dither16(seed: u64, step: u64, counter: u64) -> u16 {
    sr_mix64(seed ^ step.wrapping_mul(SR_STEP_SALT), counter) as u16
}

/// Stochastically round an f32 to bf16 storage bits using a caller-supplied
/// 16-bit dither. This is the REFERENCE for the PTX kernel's rounding tail —
/// the parity gate runs both on identical inputs and dithers.
#[inline]
pub fn sr_bf16_round(x: f32, dither: u16) -> u16 {
    let bits = x.to_bits();
    let sign16 = ((bits >> 16) & 0x8000) as u16;
    let exp = bits & 0x7F80_0000;
    if exp == 0x7F80_0000 {
        // Arithmetic Inf/NaN: propagate explicitly (see module doc).
        return if bits & 0x007F_FFFF != 0 {
            sign16 | 0x7FC0 // canonical quiet NaN
        } else {
            sign16 | 0x7F80 // ±Inf
        };
    }
    let dithered = bits.wrapping_add(dither as u32);
    if dithered & 0x7F80_0000 == 0x7F80_0000 {
        // Rounding-induced overflow: saturate to ±max normal.
        return sign16 | 0x7F7F;
    }
    (dithered >> 16) as u16
}

/// Full reference step: dither from the counter stream, then round.
#[inline]
pub fn sr_bf16_round_counter(x: f32, seed: u64, step: u64, counter: u64) -> u16 {
    sr_bf16_round(x, sr_dither16(seed, step, counter))
}

// ---------------------------------------------------------------------------
// BF16 authoritative-mirror residency backend
// ---------------------------------------------------------------------------
//
// Under `--param-dtype bf16-sr` every streamed parameter's AUTHORITATIVE copy
// is a device-resident bf16 buffer (the "mirror" — 2 bytes/elem, no FP32
// master and no host copy). The weight-stream entry points redirect here
// (same shape as the ZeRO-3 redirect):
//
//   register: quantize the initial f32 device params to a fresh bf16 mirror
//             (round-to-nearest-even, the standard initial quantization) and
//             free the f32 buffer;
//   upload:   allocate a TRANSIENT f32 working view and widen the mirror
//             into it (device-to-device cast, no host, no PCIe);
//   evict:    free the working view — NO writeback, because the fused SR
//             optimizer step (`nsl_sr_bf16_step_adamw`) writes the bf16
//             mirror directly and then refreshes the resident working view,
//             so the view is coherent (view == widen(mirror)) at all times.
//
// The ONLY sanctioned mutation of a registered parameter is that fused step;
// codegen refuses every composition that would update theta through another
// path (interpreted updates, muon, reduced-precision moments, ZeRO, offload).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::tensor::NslTensor;

// Fields are read only on the CUDA paths; the table TYPE (and thus the
// struct) must exist in every build for the unconditional count getters.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
struct Bf16Mirror {
    /// Raw device pointer of the bf16 authoritative buffer (len * 2 bytes).
    dev_bf16: u64,
    /// Element count.
    len: usize,
    /// Stable parameter index assigned by codegen (SR counter block).
    param_idx: u64,
}

static SRBF16_ACTIVE: AtomicBool = AtomicBool::new(false);
static SRBF16_TABLE: Mutex<Option<HashMap<i64, Bf16Mirror>>> = Mutex::new(None);
/// param ptr -> stable index, noted by codegen BEFORE first registration.
static SRBF16_PENDING_IDX: Mutex<Option<HashMap<i64, u64>>> = Mutex::new(None);
/// Anti-vacuity counters (gates assert these moved).
pub static SRBF16_STEPS: AtomicU64 = AtomicU64::new(0);
pub static SRBF16_UPLOADS: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn srbf16_active() -> bool {
    SRBF16_ACTIVE.load(Ordering::Relaxed)
}

/// Emitted at train-block setup when `--param-dtype bf16-sr` is on.
/// Idempotent and quiet on re-entry (the register belts run every step).
#[no_mangle]
pub extern "C" fn nsl_sr_bf16_enable() {
    if !SRBF16_ACTIVE.swap(true, Ordering::Relaxed) {
        eprintln!(
            "[sr-bf16] authoritative parameter dtype: bf16 with counter-based \
             stochastic rounding (no FP32 master copy)"
        );
    }
}

/// Emitted per parameter before its first weight-stream registration:
/// assigns the stable SR counter block (`param_idx << SR_PARAM_SHIFT`).
#[no_mangle]
pub extern "C" fn nsl_sr_bf16_note_param(tensor_ptr: i64, param_idx: i64) {
    let mut guard = SRBF16_PENDING_IDX.lock().unwrap();
    guard
        .get_or_insert_with(HashMap::new)
        .insert(tensor_ptr, param_idx as u64);
}

#[no_mangle]
pub extern "C" fn nsl_sr_bf16_step_count() -> i64 {
    SRBF16_STEPS.load(Ordering::Relaxed) as i64
}

#[no_mangle]
pub extern "C" fn nsl_sr_bf16_registered_count() -> i64 {
    let guard = SRBF16_TABLE.lock().unwrap();
    guard.as_ref().map_or(0, |g| g.len() as i64)
}

pub(crate) fn srbf16_is_registered(tensor_ptr: i64) -> bool {
    let guard = SRBF16_TABLE.lock().unwrap();
    guard.as_ref().is_some_and(|g| g.contains_key(&tensor_ptr))
}

// Non-CUDA builds: the residency backend is GPU-only; codegen refuses
// `--param-dtype bf16-sr` off-GPU, so reaching any of these without the
// cuda feature is a wiring bug — abort loudly (deferral-must-refuse).
#[cfg(not(feature = "cuda"))]
mod no_cuda {
    fn refuse(op: &str) -> ! {
        eprintln!("[sr-bf16] {op} requires the cuda feature");
        std::process::abort();
    }
    // register's redirect lives inside weight_stream's cfg(cuda) block, so
    // this arm is unreachable-by-construction — kept for API symmetry.
    #[allow(dead_code)]
    pub(crate) fn srbf16_register(_: i64) { refuse("register") }
    pub(crate) fn srbf16_upload(_: i64) { refuse("upload") }
    pub(crate) fn srbf16_evict(_: i64) { refuse("evict") }
    pub(crate) fn srbf16_upload_all() { refuse("upload_all") }
    pub(crate) fn srbf16_evict_all() { refuse("evict_all") }
}
#[cfg(not(feature = "cuda"))]
pub(crate) use no_cuda::*;

/// Register: quantize the f32 device param into a fresh bf16 mirror (RTE)
/// and free the f32 storage. Idempotent: an already-registered RESIDENT
/// param degenerates to a pure evict (the mirror is current by the
/// coherence invariant); an already-registered EVICTED param is a no-op.
#[cfg(feature = "cuda")]
pub(crate) fn srbf16_register(tensor_ptr: i64) {
    let t = NslTensor::from_ptr(tensor_ptr);
    if srbf16_is_registered(tensor_ptr) {
        srbf16_evict(tensor_ptr);
        return;
    }
    if t.owns_data == 0 || t.data_owner != 0 || t.slab_managed != 0 {
        eprintln!(
            "[sr-bf16] refusing to register tensor {tensor_ptr}: owns_data={} \
             data_owner={} slab_managed={} — only plain owning non-slab GPU \
             tensors take a bf16 mirror",
            t.owns_data, t.data_owner, t.slab_managed
        );
        std::process::abort();
    }
    if t.dtype != crate::tensor::DTYPE_F32 {
        eprintln!(
            "[sr-bf16] FATAL: register expects an f32 GPU param (dtype 1), \
             got dtype {} for tensor {tensor_ptr}",
            t.dtype
        );
        std::process::abort();
    }
    let param_idx = {
        let guard = SRBF16_PENDING_IDX.lock().unwrap();
        match guard.as_ref().and_then(|g| g.get(&tensor_ptr)).copied() {
            Some(i) => i,
            None => {
                eprintln!(
                    "[sr-bf16] FATAL: tensor {tensor_ptr} registered without a \
                     prior nsl_sr_bf16_note_param — codegen must assign the \
                     stable SR counter index first"
                );
                std::process::abort();
            }
        }
    };
    let len = t.len as usize;
    assert!(
        (len as u64) < (1u64 << SR_PARAM_SHIFT),
        "[sr-bf16] parameter of {len} elements exceeds the per-param SR \
         counter block (2^{SR_PARAM_SHIFT})"
    );
    crate::cuda::inner::ensure_context();
    let dev_bf16 = crate::cuda::inner::alloc_managed(len * 2) as u64;
    crate::cuda::gpu_cast_raw_f32_to_bf16(t.data as u64, dev_bf16, len);
    crate::cuda::inner::free_managed(t.data);
    t.data = std::ptr::null_mut();
    let mut guard = SRBF16_TABLE.lock().unwrap();
    guard
        .get_or_insert_with(HashMap::new)
        .insert(tensor_ptr, Bf16Mirror { dev_bf16, len, param_idx });
}

/// Upload: widen the mirror into a fresh transient f32 working view.
#[cfg(feature = "cuda")]
pub(crate) fn srbf16_upload(tensor_ptr: i64) {
    let t = NslTensor::from_ptr(tensor_ptr);
    if !t.data.is_null() {
        return; // already resident
    }
    let guard = SRBF16_TABLE.lock().unwrap();
    let Some(m) = guard.as_ref().and_then(|g| g.get(&tensor_ptr)) else {
        eprintln!("[sr-bf16] FATAL: upload of unregistered tensor {tensor_ptr}");
        std::process::abort();
    };
    crate::cuda::inner::ensure_context();
    let dev = crate::cuda::inner::alloc_managed(m.len * 4);
    crate::cuda::gpu_cast_raw_bf16_to_f32(m.dev_bf16, dev as u64, m.len);
    drop(guard);
    t.data = dev;
    t.owns_data = 1;
    SRBF16_UPLOADS.fetch_add(1, Ordering::Relaxed);
}

/// Evict: free the transient working view. Never writes back — the fused SR
/// step already updated the mirror (coherence invariant, module doc above).
#[cfg(feature = "cuda")]
pub(crate) fn srbf16_evict(tensor_ptr: i64) {
    let t = NslTensor::from_ptr(tensor_ptr);
    if t.data.is_null() {
        return;
    }
    if !srbf16_is_registered(tensor_ptr) {
        eprintln!("[sr-bf16] FATAL: evict of unregistered tensor {tensor_ptr}");
        std::process::abort();
    }
    crate::cuda::inner::ensure_context();
    crate::cuda::inner::free_managed(t.data);
    t.data = std::ptr::null_mut();
}

#[cfg(feature = "cuda")]
pub(crate) fn srbf16_upload_all() {
    let ptrs: Vec<i64> = {
        let guard = SRBF16_TABLE.lock().unwrap();
        guard.as_ref().map_or_else(Vec::new, |g| g.keys().copied().collect())
    };
    for p in ptrs {
        srbf16_upload(p);
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn srbf16_evict_all() {
    let ptrs: Vec<i64> = {
        let guard = SRBF16_TABLE.lock().unwrap();
        guard.as_ref().map_or_else(Vec::new, |g| g.keys().copied().collect())
    };
    for p in ptrs {
        srbf16_evict(p);
    }
}

/// Train-block teardown: re-materialize every param as an ordinary owned
/// f32 device tensor (post-training code — model_save, eval — sees plain
/// tensors), report anti-vacuity counters, free the mirrors, drop the mode.
#[no_mangle]
pub extern "C" fn nsl_sr_bf16_teardown() {
    #[cfg(feature = "cuda")]
    {
        srbf16_upload_all();
        let mut guard = SRBF16_TABLE.lock().unwrap();
        if let Some(table) = guard.take() {
            eprintln!(
                "[sr-bf16] teardown: {} bf16-authoritative param(s), {} SR \
                 optimizer step(s), {} widen-upload(s)",
                table.len(),
                SRBF16_STEPS.load(Ordering::Relaxed),
                SRBF16_UPLOADS.load(Ordering::Relaxed),
            );
            crate::cuda::inner::ensure_context();
            for (_, m) in table {
                crate::cuda::inner::free_managed(m.dev_bf16 as *mut std::ffi::c_void);
            }
        }
    }
    SRBF16_ACTIVE.store(false, Ordering::Relaxed);
}

/// P4 item 17: the ONLY sanctioned weight mutation under bf16-sr — fused
/// AdamW step against the bf16 mirror with counter-based SR, then a refresh
/// of the resident f32 working view so the coherence invariant holds.
///
/// Mirrors `nsl_fase_fused_adamw_step`'s scalar contract (f64 scalars,
/// bc1/bc2 precomputed by codegen); adds `step` (the optimizer step counter
/// the SR stream is keyed on). Seed comes from the global `--seed` store.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_sr_bf16_step_adamw(
    theta_ptr: i64,
    m_ptr: i64,
    v_ptr: i64,
    mp_ptr: i64,
    lr: f64,
    beta1: f64,
    one_minus_beta1: f64,
    beta2: f64,
    one_minus_beta2: f64,
    eps: f64,
    wd: f64,
    bc1_inv: f64,
    bc2_inv: f64,
    step: i64,
) {
    #[cfg(feature = "cuda")]
    {
        let th = NslTensor::from_ptr(theta_ptr);
        let m = unsafe { &*(m_ptr as *const NslTensor) };
        let v = unsafe { &*(v_ptr as *const NslTensor) };
        let mp = unsafe { &*(mp_ptr as *const NslTensor) };
        let (dev_bf16, len, param_idx) = {
            let guard = SRBF16_TABLE.lock().unwrap();
            match guard.as_ref().and_then(|g| g.get(&theta_ptr)) {
                Some(mi) => (mi.dev_bf16, mi.len, mi.param_idx),
                None => {
                    // Un-streamed params (view-rooted / tied / epilogue
                    // residents — the same set ZeRO-3 keeps Replicated) have
                    // no bf16 mirror: they keep f32 authoritative storage
                    // and take the plain fused step. bf16-sr's memory story
                    // applies to the STREAMED set, exactly like sharding.
                    drop(guard);
                    crate::fase_step::nsl_fase_fused_adamw_step(
                        theta_ptr, m_ptr, v_ptr, mp_ptr,
                        lr, beta1, one_minus_beta1, beta2, one_minus_beta2,
                        eps, wd, bc1_inv, bc2_inv,
                    );
                    return;
                }
            }
        };
        assert!(
            m.len as usize == len && v.len as usize == len && mp.len as usize == len,
            "[sr-bf16] step length mismatch (theta={len}, m={}, v={}, mp={})",
            m.len, v.len, mp.len
        );
        assert!(
            m.device > 0 && v.device > 0 && mp.device > 0,
            "[sr-bf16] m/v/mp must be GPU-resident"
        );
        assert!(
            m.dtype == 1 && v.dtype == 1 && mp.dtype == 1,
            "[sr-bf16] m/v/mp must be f32 (got {}/{}/{})",
            m.dtype, v.dtype, mp.dtype
        );
        assert!(
            m.is_contiguous() && v.is_contiguous() && mp.is_contiguous(),
            "[sr-bf16] m/v/mp must be contiguous"
        );
        let seed = crate::deterministic_ops::get_rng_seed();
        let key = seed ^ (step as u64).wrapping_mul(SR_STEP_SALT);
        let ctr_base = param_idx << SR_PARAM_SHIFT;
        let has_wd = wd != 0.0;
        crate::cuda::gpu_fase_fused_adamw_step_bf16sr(
            dev_bf16, m_ptr, v_ptr, mp_ptr, len,
            beta1 as f32,
            one_minus_beta1 as f32,
            beta2 as f32,
            one_minus_beta2 as f32,
            eps as f32,
            (-lr) as f32,
            ((-lr) * wd) as f32,
            bc1_inv as f32,
            bc2_inv as f32,
            has_wd,
            key,
            ctr_base,
        );
        // Coherence refresh: the resident f32 working view must equal
        // widen(mirror) at all times so callbacks / model_save / the next
        // forward all read post-update values.
        if !th.data.is_null() {
            crate::cuda::gpu_cast_raw_bf16_to_f32(dev_bf16, th.data as u64, len);
        }
        SRBF16_STEPS.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            theta_ptr, m_ptr, v_ptr, mp_ptr, lr, beta1, one_minus_beta1, beta2,
            one_minus_beta2, eps, wd, bc1_inv, bc2_inv, step,
        );
        eprintln!("[sr-bf16] nsl_sr_bf16_step_adamw requires the cuda feature");
        std::process::abort();
    }
}

// ---------------------------------------------------------------------------
// P4 item 18 rung 2: compressed Muon state — BF16 momentum, FP32 working
// ---------------------------------------------------------------------------

/// Distinct salt for the Muon STATE dither stream so it never correlates
/// with the WEIGHT stream (item 17) at equal (seed, step, param, element).
pub const SR_MUON_STATE_SALT: u64 = 0xA5A5_5A5A_C3C3_3C3C;

/// Anti-vacuity counter: SR stores of the bf16 momentum.
pub static MUON_STATE_SR_STORES: AtomicU64 = AtomicU64::new(0);

#[no_mangle]
pub extern "C" fn nsl_muon_state_sr_count() -> i64 {
    MUON_STATE_SR_STORES.load(Ordering::Relaxed) as i64
}

/// Quant-store the post-step FP32 working momentum back into its BF16
/// authoritative buffer with counter-based stochastic rounding
/// (`--muon-state-dtype bf16`). SR — not RTE — because the momentum EMA's
/// per-step increment `(1-β)·g` routinely falls below a bf16 ulp of m;
/// nearest-rounding would drop it systematically, SR preserves it in
/// expectation. Same mixer as the weight stream, decorrelated by
/// `SR_MUON_STATE_SALT`.
#[no_mangle]
pub extern "C" fn nsl_muon_state_sr_store(
    src_f32_ptr: i64,
    dst_bf16_ptr: i64,
    step: i64,
    param_idx: i64,
) {
    let src = unsafe { &*(src_f32_ptr as *const NslTensor) };
    let dst = unsafe { &*(dst_bf16_ptr as *const NslTensor) };
    assert!(
        src.device == dst.device,
        "[muon-state] SR store device mismatch (src dev={}, dst dev={})",
        src.device, dst.device
    );
    assert!(
        src.dtype == crate::tensor::DTYPE_F32 && dst.dtype == crate::tensor::DTYPE_BF16,
        "[muon-state] SR store expects f32 -> bf16 (got {} -> {})",
        src.dtype, dst.dtype
    );
    assert!(
        src.len == dst.len && src.is_contiguous() && dst.is_contiguous(),
        "[muon-state] SR store shape/layout mismatch (src len={}, dst len={})",
        src.len, dst.len
    );
    let len = src.len as usize;
    assert!(
        (len as u64) < (1u64 << SR_PARAM_SHIFT),
        "[muon-state] parameter exceeds the per-param SR counter block"
    );
    let seed = crate::deterministic_ops::get_rng_seed() ^ SR_MUON_STATE_SALT;
    let key = seed ^ (step as u64).wrapping_mul(SR_STEP_SALT);
    let ctr_base = (param_idx as u64) << SR_PARAM_SHIFT;
    if src.device > 0 {
        #[cfg(feature = "cuda")]
        crate::cuda::gpu_sr_bf16_round_probe(
            src.data as u64, dst.data as u64, len, key, ctr_base,
        );
        #[cfg(not(feature = "cuda"))]
        {
            eprintln!("[muon-state] GPU SR store requires the cuda feature");
            std::process::abort();
        }
    } else {
        // CPU-resident params (e.g. rank-1 norm vectors a model kept on the
        // host) take the reference implementation — BIT-IDENTICAL to the
        // PTX tail by construction (pure integer math, same counters), so
        // device placement never changes the trained bits.
        let s = src.data as *const f32;
        let d = dst.data as *mut u16;
        for i in 0..len {
            let x = unsafe { *s.add(i) };
            let bits = sr_bf16_round(x, sr_mix64(key, ctr_base + i as u64) as u16);
            unsafe { *d.add(i) = bits };
        }
    }
    if MUON_STATE_SR_STORES.fetch_add(1, Ordering::Relaxed) == 0 {
        eprintln!(
            "[muon-state] bf16 momentum active: f32 working buffer + \
             counter-based SR store (item 18 rung 2)"
        );
    }
}

/// Parity-gate hook: run the GPU SR rounding tail over `vals` with explicit
/// counters and return the bf16 storage bits. Host-side staging only — the
/// integration gate compares this against `sr_bf16_round_counter`.
#[cfg(feature = "cuda")]
pub fn sr_bf16_gpu_probe_host(vals: &[f32], seed: u64, step: u64, ctr_base: u64) -> Vec<u16> {
    let n = vals.len();
    if n == 0 {
        return Vec::new();
    }
    crate::cuda::inner::ensure_context();
    let src = crate::cuda::inner::alloc_managed(n * 4);
    crate::cuda::inner::memcpy_htod(src, vals.as_ptr() as *const std::ffi::c_void, n * 4);
    let dst = crate::cuda::inner::alloc_managed(n * 2);
    let key = seed ^ step.wrapping_mul(SR_STEP_SALT);
    crate::cuda::gpu_sr_bf16_round_probe(src as u64, dst as u64, n, key, ctr_base);
    let mut out = vec![0u16; n];
    crate::cuda::inner::memcpy_dtoh(out.as_mut_ptr() as *mut std::ffi::c_void, dst, n * 2);
    crate::cuda::inner::free_managed(src);
    crate::cuda::inner::free_managed(dst);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::bf16_bits_to_f32;

    /// Same (seed, step, counter) → same output, across calls and value
    /// reuse; different step or counter → (overwhelmingly) different dither.
    #[test]
    fn sr_is_deterministic_in_its_counters() {
        let x = 1.234_567_9_f32;
        let a = sr_bf16_round_counter(x, 42, 7, 1_000_000);
        let b = sr_bf16_round_counter(x, 42, 7, 1_000_000);
        assert_eq!(a, b);
        let mut distinct = std::collections::HashSet::new();
        for step in 0..64u64 {
            distinct.insert(sr_dither16(42, step, 1_000_000));
        }
        assert!(distinct.len() > 48, "step must vary the dither stream");
    }

    /// E[SR(x)] == x: over the full 2^16 dither space the AVERAGE of the
    /// rounded values equals x exactly (the dither enumerates every
    /// remainder threshold once). Uses exhaustive enumeration, not
    /// sampling, so the test is deterministic.
    #[test]
    fn sr_is_exactly_unbiased_over_the_dither_space() {
        for &x in &[1.000_01_f32, -3.999_9, 0.333_333_34, 1.5e-40, -2.7e38] {
            let lo = (x.to_bits() >> 16) as u16; // truncation
            let mut acc = 0.0_f64;
            for d in 0..=u16::MAX {
                acc += bf16_bits_to_f32(sr_bf16_round(x, d)) as f64;
            }
            let mean = acc / 65536.0;
            // Exact identity: mean = lo + frac*ulp. Compare against x with
            // a tolerance far below half a bf16 ulp of x.
            let ulp = (bf16_bits_to_f32(lo.wrapping_add(1)) - bf16_bits_to_f32(lo)).abs() as f64;
            assert!(
                (mean - x as f64).abs() < ulp * 1e-3,
                "biased SR for {x}: mean={mean}",
            );
        }
    }

    /// A value already representable in bf16 must round to itself under
    /// every dither except... none — remainder is 0, dither never carries.
    #[test]
    fn sr_exact_values_are_fixed_points() {
        for &x in &[1.0_f32, -2.5, 0.0, -0.0, 65280.0] {
            let expect = (x.to_bits() >> 16) as u16;
            for d in [0u16, 1, 0x7FFF, 0xFFFF] {
                assert_eq!(sr_bf16_round(x, d), expect, "x={x} d={d}");
            }
        }
    }

    #[test]
    fn sr_edge_cases_follow_the_documented_policy() {
        // NaN → canonical quiet NaN, sign preserved, NEVER Inf.
        let nan_payload = f32::from_bits(0x7F80_0001); // signaling-ish NaN
        assert_eq!(sr_bf16_round(nan_payload, 0xFFFF), 0x7FC0);
        assert_eq!(sr_bf16_round(f32::from_bits(0xFF80_0001), 0xFFFF), 0xFFC0);
        // ±Inf propagates.
        assert_eq!(sr_bf16_round(f32::INFINITY, 0xFFFF), 0x7F80);
        assert_eq!(sr_bf16_round(f32::NEG_INFINITY, 0xFFFF), 0xFF80);
        // Rounding-induced overflow saturates: largest finite f32 with a
        // dither that would carry into the Inf encoding.
        assert_eq!(sr_bf16_round(f32::MAX, 0xFFFF), 0x7F7F);
        assert_eq!(sr_bf16_round(-f32::MAX, 0xFFFF), 0xFF7F);
        // ...but the same value with zero dither truncates finitely.
        assert_eq!(sr_bf16_round(f32::MAX, 0), 0x7F7F);
        // Subnormal underflow is gradual: tiny positive value rounds to
        // either 0 or the smallest subnormal, never something else.
        let tiny = f32::from_bits(0x0000_5000); // deep f32 subnormal
        for d in [0u16, 0x8000, 0xFFFF] {
            let out = sr_bf16_round(tiny, d);
            assert!(out == 0x0000 || out == 0x0001, "tiny→{out:#06x}");
        }
    }

    /// The dither stream must not collide across parameters: distinct
    /// param_base blocks give distinct streams for the same element index.
    #[test]
    fn sr_param_bases_decorrelate_streams() {
        let e = 12345u64;
        let a = sr_dither16(42, 3, (1u64 << SR_PARAM_SHIFT) + e);
        let b = sr_dither16(42, 3, (2u64 << SR_PARAM_SHIFT) + e);
        let c = sr_dither16(42, 3, (3u64 << SR_PARAM_SHIFT) + e);
        assert!(!(a == b && b == c), "param bases must decorrelate dithers");
    }
}
