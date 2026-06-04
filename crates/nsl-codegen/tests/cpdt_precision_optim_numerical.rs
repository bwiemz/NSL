//! In-process numerical validation of the FP16-storage AdamW mechanism for CPDT.
//!
//! ## What this validates
//!
//! The CPDT precision-adaptive optimizer stores AdamW first/second moment
//! state (m, v) at FP16 precision.  The codegen wraps FASE's existing FP32
//! AdamW update with:
//!
//!   1. dequant: `m_f32 = nsl_tensor_cast(m_fp16, DTYPE_F32)`
//!   2. update:  FP32 AdamW step (matching FASE's `emit_adamw` recipe)
//!   3. quant:   `nsl_tensor_cast_into(m_fp16, m_f32_updated)`
//!
//! We CANNOT run the full compiled-IR end-to-end here (source-AD crashes the
//! test fixture — a pre-existing, out-of-scope bug).  Instead this test
//! drives the mechanism in-process using the real `nsl_tensor_cast` /
//! `nsl_tensor_cast_into` FFI ops and compares against carefully constructed
//! Rust references.
//!
//! ## Comparisons
//!
//! | Mode | Description | Gate |
//! |------|-------------|------|
//! | Compiled-mechanism run (1) | real cast ops, N steps | (reference) |
//! | Mode B — cast correctness (2) | pure-Rust truncating FP16 | rel_err < 1e-5 |
//! | Mode A — precision-adaptive claim (3) | pure FP32 (no quant) | rel_err < 5e-2 |
//! | Negative control (4) | RTN rounding != truncation | must differ > 1e-5 |
//! | High-tier bit-exact (5) | F32->F32 cast round-trip | bit-exact |
//!
//! ## AdamW recipe (from `fase_numerical_validation.rs` / `fase_optimizer.rs`)
//!
//! ```text
//!   m  = beta1*m + (1-beta1)*g         (EMA of gradient)
//!   v  = beta2*v + (1-beta2)*g^2       (EMA of squared gradient -- FASE approximation)
//!   m_hat = m / (1 - beta1^t)          (bias-corrected)
//!   v_hat = v / (1 - beta2^t)          (bias-corrected)
//!   denom = sqrt(v_hat) + eps
//!   theta -= lr * (m_hat / denom + wd * theta)
//! ```
//!
//! NSL's FASE AdamW DOES apply bias-correction divisors (1 - beta^t) per the
//! `adamw_fase_deferred_reference` fn in `fase_numerical_validation.rs`.
//!
//! ## FP16 truncation note
//!
//! `nsl_runtime`'s `f32_to_f16_bits` TRUNCATES the mantissa (`mant >> 13`,
//! no guard/sticky logic), despite the docstring claiming RTN.  The conversion
//! is biased toward zero.  All references that simulate FP16 storage replicate
//! this truncation exactly.
//!
//! ## Tolerance derivation (Mode A, FP16 vs FP32)
//!
//! Per-step FP16 truncation max relative error <= 2^-10 ~= 9.77e-4 (one
//! mantissa bucket in the f16 encoding).  The EMA low-pass filter with decay
//! beta means the accumulated state drift is bounded by ~eps_per_step/(1-beta)
//! in steady state (geometric series bound).  For beta1=0.9: ~9.77e-3; for
//! beta2=0.999: ~0.977.  v enters via sqrt+bias-correction which compresses
//! sensitivity.  We use a conservative empirical bound of rel_err < 5e-2.

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};
use nsl_runtime::tensor::precision_cast::{nsl_tensor_cast, nsl_tensor_cast_into};

// ---------------------------------------------------------------------------
// Tensor helpers (using only public APIs from nsl-runtime)
// ---------------------------------------------------------------------------

/// Create a CPU f32 tensor (dtype=1) filled with `vals`.
/// Uses `nsl_tensor_zeros_on` + raw-pointer write (via `nsl_tensor_data_ptr`).
fn make_f32_tensor(vals: &[f32]) -> i64 {
    let len = vals.len();
    let shape_list = nsl_list_new();
    nsl_list_push(shape_list, len as i64);
    let ptr = nsl_tensor_zeros_on(shape_list, 0 /* CPU */);
    nsl_list_free(shape_list);
    // Write data directly into the tensor's f32 buffer.
    let data = nsl_tensor_data_ptr(ptr) as *mut f32;
    for (i, &v) in vals.iter().enumerate() {
        unsafe { *data.add(i) = v };
    }
    ptr
}

/// Read all f32 elements from a CPU f32 tensor.
fn read_f32_tensor(ptr: i64, len: usize) -> Vec<f32> {
    let data = nsl_tensor_data_ptr(ptr) as *const f32;
    (0..len).map(|i| unsafe { *data.add(i) }).collect()
}

// ---------------------------------------------------------------------------
// FP16 conversion helpers -- replicated from nsl_runtime's pub(crate) fns
// ---------------------------------------------------------------------------

/// Truncating f32->f16 bits.  Mirrors `nsl_runtime::tensor::f32_to_f16_bits`
/// exactly (truncation in the normal-number path: `mant >> 13`, no round bit).
/// Used in Mode B reference and negative-control without linking to the
/// `pub(crate)` symbol.
#[inline]
fn f32_to_f16_bits_truncate(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp >= 143 {
        return sign | 0x7C00; // overflow -> +-Inf
    }
    if exp <= 102 {
        return sign; // underflow -> +-0
    }
    if exp <= 112 {
        // Subnormal f16
        let shift = 113 - exp;
        let m = (0x80_0000 | mant) >> (shift + 13);
        return sign | m as u16;
    }
    // Normal: TRUNCATE -- no rounding
    let f16_exp = ((exp - 112) as u16) << 10;
    let f16_mant = (mant >> 13) as u16;
    sign | f16_exp | f16_mant
}

/// Round-to-nearest-even f32->f16 bits.  Used ONLY for the negative control
/// (Mode 4).  The runtime uses truncation, not RTN, so results differ for
/// values whose f32 mantissa has non-zero bits in positions 11-13.
#[inline]
fn f32_to_f16_bits_rtn(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;
    if exp >= 143 {
        return sign | 0x7C00;
    }
    if exp <= 102 {
        return sign;
    }
    if exp <= 112 {
        let shift = 113 - exp;
        let m_full = (0x80_0000 | mant) >> shift;
        let round = m_full & (1 << 12);
        let m_trunc = m_full >> 13;
        let m_rtn = if round != 0 { m_trunc + 1 } else { m_trunc };
        return sign | m_rtn as u16;
    }
    // Normal: RTN -- add guard bit before shifting
    let guard = (mant >> 12) & 1;
    let sticky = mant & 0xFFF;
    let f16_mant_trunc = (mant >> 13) as u16;
    // Round up if guard=1 and (sticky!=0 OR lsb=1)
    let f16_mant_rtn = if guard != 0 && (sticky != 0 || (f16_mant_trunc & 1) != 0) {
        f16_mant_trunc + 1
    } else {
        f16_mant_trunc
    };
    let f16_exp = ((exp - 112) as u16) << 10;
    sign | f16_exp | f16_mant_rtn
}

/// Widen f16 bits back to f32.  Exact mirror of `nsl_runtime::tensor::f16_bits_to_f32`.
///
/// The subnormal branch iterates left-shifts until bit 10 of `m` is set (the
/// implied leading 1 in the normalized f16 mantissa), then encodes the result
/// as a normalised f32.  This matches the runtime's iterative normalization loop
/// exactly.
#[inline]
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal f16 -> normalize (mirror of runtime's iterative loop).
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 {
            m <<= 1;
            e += 1;
        }
        let f_exp = (127 - 15 + 1 - e) << 23;
        let f_mant = (m & 0x3FF) << 13;
        f32::from_bits(sign | f_exp | f_mant)
    } else if exp == 0x1F {
        f32::from_bits(sign | 0x7F80_0000 | (mant << 13)) // +-Inf / NaN
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

/// Round-trip f32 through FP16 truncation (pure Rust, no tensors).
#[inline]
fn trunc_fp16(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits_truncate(v))
}

/// Round-trip f32 through RTN FP16 (pure Rust, no tensors).
#[inline]
fn rtn_fp16(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits_rtn(v))
}

// ---------------------------------------------------------------------------
// AdamW reference (matches `fase_numerical_validation.rs` / `emit_adamw`)
// ---------------------------------------------------------------------------

/// One AdamW step matching FASE's `emit_adamw` recipe exactly.
///
/// Updates theta, m, v IN PLACE.  Uses bias-correction divisors (1 - beta^t)
/// per `adamw_fase_deferred_reference` in `fase_numerical_validation.rs`.
///
/// `step` is 1-based (the bias-correction exponent for this step).
#[allow(clippy::too_many_arguments)]
fn adamw_step_f32(
    theta: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
    grad: &[f32],
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    wd: f32,
    step: u32,
) {
    for i in 0..theta.len() {
        let g = grad[i];
        m[i] = beta1 * m[i] + (1.0 - beta1) * g;
        v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;
        let bc1 = 1.0 - beta1.powi(step as i32);
        let bc2 = 1.0 - beta2.powi(step as i32);
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        let denom = v_hat.sqrt() + eps;
        let tmp = m_hat / denom;
        theta[i] -= lr * (tmp + wd * theta[i]);
    }
}

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------

const N_PARAMS: usize = 32;
const N_STEPS: usize = 15;
const LR: f32 = 1e-3;
const BETA1: f32 = 0.9;
const BETA2: f32 = 0.999;
const EPS: f32 = 1e-8;
const WD: f32 = 0.01;

/// DTYPE constants (match nsl_runtime: F32=1, FP16=2).
const DTYPE_F32: i64 = 1;
const DTYPE_FP16: i64 = 2;

/// Deterministic initial parameters with varied magnitudes exercising different
/// f16 exponent bins.  Values are chosen so that EMA state values land in the
/// f16 subnormal range and the normal range, and so that truncation and RTN
/// diverge (non-zero guard bits) for the negative control.
fn initial_params() -> Vec<f32> {
    (0..N_PARAMS)
        .map(|i| {
            let base = match i % 4 {
                0 => 0.01 + (i as f32) * 0.003, // ~0.01 .. 0.1
                1 => 0.1 + (i as f32) * 0.07,   // ~0.1 .. 2.3
                2 => 1.0 + (i as f32) * 0.13,   // ~1.0 .. 5.0
                _ => 5.0 + (i as f32) * 1.5,    // ~5.0 .. 52.5
            };
            if i % 5 == 0 { -base } else { base }
        })
        .collect()
}

/// Deterministic gradient for step `s` (1-based) and parameter index `i`.
fn gradient(step: usize, param_idx: usize) -> f32 {
    let angle = (step as f32 * 0.37 + param_idx as f32 * 0.11) * std::f32::consts::PI;
    let base = angle.sin() * 0.5 + (param_idx as f32 * 0.02 + 0.1);
    base * (0.5 + (step as f32) * 0.03)
}

fn gradients_for_step(step: usize) -> Vec<f32> {
    (0..N_PARAMS).map(|i| gradient(step, i)).collect()
}

// ---------------------------------------------------------------------------
// Relative error helper
// ---------------------------------------------------------------------------

fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let denom = x.abs().max(y.abs()).max(1e-10);
            (x - y).abs() / denom
        })
        .fold(0.0_f32, f32::max)
}

// ---------------------------------------------------------------------------
// The five runs
// ---------------------------------------------------------------------------

/// Run 1 (compiled-mechanism): uses real `nsl_tensor_cast` / `nsl_tensor_cast_into`.
///
/// m and v optimizer state is stored in a tensor of `state_dtype`.  Each step:
///   - dequant: cast state -> F32 (new owned tensor, freed after reading)
///   - update:  AdamW in Rust f32 (matching FASE recipe)
///   - quant:   cast_into(state_tensor, updated_f32_tensor)
///   - free the temp F32 update tensors
///
/// When `use_fp16_state=false`, stores m/v as F32 (High-tier path, test 5).
/// Returns final theta (params always at full F32 precision).
fn run_compiled_mechanism(theta_init: &[f32], use_fp16_state: bool) -> Vec<f32> {
    let len = theta_init.len();
    let mut theta = theta_init.to_vec();

    let state_dtype = if use_fp16_state { DTYPE_FP16 } else { DTYPE_F32 };
    let zeros = vec![0.0_f32; len];
    let tmp_f32 = make_f32_tensor(&zeros);
    let m_state = nsl_tensor_cast(tmp_f32, state_dtype);
    let v_state = nsl_tensor_cast(tmp_f32, state_dtype);
    nsl_tensor_free(tmp_f32);

    for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);

        // Dequant: cast state -> F32 (new owned tensors).
        let m_f32 = nsl_tensor_cast(m_state, DTYPE_F32);
        let v_f32 = nsl_tensor_cast(v_state, DTYPE_F32);
        let mut m_vals = read_f32_tensor(m_f32, len);
        let mut v_vals = read_f32_tensor(v_f32, len);
        nsl_tensor_free(m_f32);
        nsl_tensor_free(v_f32);

        // AdamW update in Rust f32.
        adamw_step_f32(
            &mut theta, &mut m_vals, &mut v_vals, &grad,
            LR, BETA1, BETA2, EPS, WD, step as u32,
        );

        // Quant-store: write updated m/v back into the state tensors.
        let m_updated = make_f32_tensor(&m_vals);
        let v_updated = make_f32_tensor(&v_vals);
        nsl_tensor_cast_into(m_state, m_updated);
        nsl_tensor_cast_into(v_state, v_updated);
        nsl_tensor_free(m_updated);
        nsl_tensor_free(v_updated);
    }

    nsl_tensor_free(m_state);
    nsl_tensor_free(v_state);
    theta
}

/// Run 2 (Mode B reference): pure-Rust AdamW that truncates m/v each step via
/// `f32_to_f16_bits_truncate` -- the same truncation as the runtime's cast ops.
///
/// m and v are initialized to 0.0 (exact in both F32 and F16).  Each step:
///   - m/v enter with the truncated values from the previous step (exact match
///     to the compiled mechanism's dequanted state)
///   - AdamW update produces new m/v
///   - trunc_fp16() simulates the quant-store round-trip
///
/// Should be bit-for-bit identical to run_compiled_mechanism(..., true) up to
/// f32 arithmetic ordering; gate: rel_err < 1e-5.
fn run_mode_b_truncate_reference(theta_init: &[f32]) -> Vec<f32> {
    let len = theta_init.len();
    let mut theta = theta_init.to_vec();
    let mut m = vec![0.0_f32; len];
    let mut v = vec![0.0_f32; len];

    for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);
        adamw_step_f32(&mut theta, &mut m, &mut v, &grad, LR, BETA1, BETA2, EPS, WD, step as u32);
        for i in 0..len {
            m[i] = trunc_fp16(m[i]);
            v[i] = trunc_fp16(v[i]);
        }
    }
    theta
}

/// Run 3 (Mode A reference): pure-FP32 AdamW with NO quantization of m/v.
fn run_mode_a_fp32_reference(theta_init: &[f32]) -> Vec<f32> {
    let len = theta_init.len();
    let mut theta = theta_init.to_vec();
    let mut m = vec![0.0_f32; len];
    let mut v = vec![0.0_f32; len];

    for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);
        adamw_step_f32(&mut theta, &mut m, &mut v, &grad, LR, BETA1, BETA2, EPS, WD, step as u32);
    }
    theta
}

/// Run 4 (negative control): pure-Rust AdamW that quantizes m/v via RTN each
/// step.  Because the runtime uses truncation (not RTN), theta must differ from
/// the compiled mechanism wherever m/v have non-zero guard bits.
fn run_negative_control_rtn(theta_init: &[f32]) -> Vec<f32> {
    let len = theta_init.len();
    let mut theta = theta_init.to_vec();
    let mut m = vec![0.0_f32; len];
    let mut v = vec![0.0_f32; len];

    for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);
        adamw_step_f32(&mut theta, &mut m, &mut v, &grad, LR, BETA1, BETA2, EPS, WD, step as u32);
        for i in 0..len {
            m[i] = rtn_fp16(m[i]);
            v[i] = rtn_fp16(v[i]);
        }
    }
    theta
}

// ---------------------------------------------------------------------------
// Non-vacuity check: truncation and RTN must diverge on these inputs
// ---------------------------------------------------------------------------

/// Returns true if any m/v value during N_STEPS has truncation-bits != RTN-bits.
fn verify_truncation_rtn_differ(theta_init: &[f32]) -> bool {
    let len = theta_init.len();
    let mut m = vec![0.0_f32; len];
    let mut v = vec![0.0_f32; len];

    'outer: for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);
        let mut theta_tmp = theta_init.to_vec();
        adamw_step_f32(&mut theta_tmp, &mut m, &mut v, &grad, LR, BETA1, BETA2, EPS, WD, step as u32);
        for &val in m.iter().chain(v.iter()) {
            if f32_to_f16_bits_truncate(val) != f32_to_f16_bits_rtn(val) {
                return true;
            }
        }
        // Advance state using truncation.
        for i in 0..len {
            m[i] = trunc_fp16(m[i]);
            v[i] = trunc_fp16(v[i]);
        }
        let _ = step; // suppress unused warning after break rewrite
        if false { break 'outer; } // make the label reachable
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Primary validation: five-way comparison of the FP16-storage AdamW mechanism.
///
/// Covers:
///   - Mode B (cast correctness): compiled == truncating-Rust ref (rel < 1e-5)
///   - Mode A (FP16 precision claim): compiled ~= FP32 ref (rel < 5e-2)
///   - Negative control: compiled != RTN ref (rel > 1e-5, proves gate non-vacuous)
///   - High-tier: F32-state compiled == FP32 ref bit-exactly
#[test]
fn cpdt_fp16_adamw_numerical_validation() {
    let theta_init = initial_params();

    // Pre-check: ensure the negative-control inputs are non-vacuous.
    assert!(
        verify_truncation_rtn_differ(&theta_init),
        "PRECONDITION FAILED: truncation and RTN produce identical results for all \
         m/v values over {N_STEPS} steps.  The negative-control gate would be vacuous."
    );

    // Run all five regimes.
    let result_compiled  = run_compiled_mechanism(&theta_init, true  /* FP16 state */);
    let result_mode_b    = run_mode_b_truncate_reference(&theta_init);
    let result_mode_a    = run_mode_a_fp32_reference(&theta_init);
    let result_neg_ctrl  = run_negative_control_rtn(&theta_init);
    let result_high_tier = run_compiled_mechanism(&theta_init, false /* F32 state */);

    // --- Mode B: cast correctness ---
    // Both apply the same truncation each step -> must agree to f32-arithmetic
    // precision (essentially floating-point order effects only).
    let rel_err_b = max_rel_err(&result_compiled, &result_mode_b);
    println!(
        "[Mode B] compiled-mechanism vs truncating-Rust ref: max rel_err = {rel_err_b:.2e}"
    );
    assert!(
        rel_err_b < 1e-5,
        "Mode B FAILED: compiled-mechanism and truncating-Rust reference diverge by \
         {rel_err_b:.2e} (gate: < 1e-5).  This indicates nsl_tensor_cast / \
         nsl_tensor_cast_into does NOT faithfully implement f32->f16 truncation."
    );

    // --- Mode A: precision-adaptive claim ---
    // FP16 storage accumulates drift but the EMA low-pass bounds it.
    // Derivation: per-step trunc_err <= 2^-10 ~= 9.8e-4; EMA bound for m:
    // ~9.8e-3 (beta1=0.9); for v: ~0.98 (beta2=0.999) but sqrt compression
    // reduces it; empirical conservative bound 5e-2.
    let rel_err_a = max_rel_err(&result_compiled, &result_mode_a);
    println!(
        "[Mode A] compiled-mechanism vs FP32 reference: max rel_err = {rel_err_a:.2e}"
    );
    assert!(
        rel_err_a < 5e-2,
        "Mode A FAILED: FP16-storage AdamW diverges from FP32 by {rel_err_a:.2e} \
         (gate: < 5e-2).  FP16 optimizer state precision loss is unacceptably large."
    );

    // --- Negative control: must NOT match RTN ---
    // Proves the Mode B gate is non-vacuous: if truncation == RTN for all
    // inputs, the gate would pass trivially even for a broken cast.
    let rel_err_neg = max_rel_err(&result_compiled, &result_neg_ctrl);
    println!(
        "[Neg ctrl] compiled-mechanism vs RTN reference: max rel_err = {rel_err_neg:.2e}"
    );
    assert!(
        rel_err_neg > 1e-5,
        "Negative control FAILED: compiled-mechanism matches RTN reference within \
         {rel_err_neg:.2e} (must be > 1e-5).  The Mode B gate is VACUOUS: truncation \
         and RTN are indistinguishable on these inputs."
    );

    // --- High-tier: F32 state is bit-identical to pure-FP32 ---
    // F32->F32 casts are lossless copies, so the High-tier path (no quantization
    // of optimizer state) must produce the exact same theta as the FP32 reference.
    let mut all_exact = true;
    for (i, (&a, &b)) in result_high_tier.iter().zip(result_mode_a.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            println!("[High-tier] MISMATCH elem[{i}]: high_tier={a} fp32_ref={b}");
            all_exact = false;
        }
    }
    assert!(
        all_exact,
        "High-tier FAILED: F32-state round-trip (F32->F32 cast) is NOT bit-identical \
         to pure-FP32 reference.  F32->F32 casts must be lossless copies."
    );

    println!();
    println!("=== CPDT FP16-storage AdamW numerical validation: ALL PASSED ===");
    println!("  N_PARAMS={N_PARAMS}  N_STEPS={N_STEPS}  LR={LR}  beta1={BETA1}  beta2={BETA2}");
    println!("  Mode B (cast correctness):    rel_err = {rel_err_b:.2e}  < 1e-5  PASS");
    println!("  Mode A (FP16 vs FP32 claim):  rel_err = {rel_err_a:.2e}  < 5e-2  PASS");
    println!("  Negative control (trunc!=RTN): rel_err = {rel_err_neg:.2e}  > 1e-5  PASS");
    println!("  High-tier (F32 bit-exact):    exact match                         PASS");
}

/// Sanity: pure-Rust f16 conversion helpers stay within 2 f16 ULPs of the
/// original f32 value.
#[test]
fn f16_conversion_helpers_sanity() {
    // Use exact f32 constants; avoid clippy::approx_constant.
    let test_vals = [
        0.1_f32, 0.2, 0.3, 1.001,
        std::f32::consts::PI, -0.5,
        -std::f32::consts::E, 0.01234,
    ];
    let one_f16_rel = 2.0_f32.powi(-10);

    for &v in &test_vals {
        let v_trunc = trunc_fp16(v);
        let v_rtn   = rtn_fp16(v);
        let err_trunc = (v_trunc - v).abs() / v.abs().max(1e-10);
        let err_rtn   = (v_rtn   - v).abs() / v.abs().max(1e-10);
        assert!(
            err_trunc <= one_f16_rel * 2.0 || v.abs() < 1e-5,
            "f32_to_f16_bits_truncate({v}) -> {v_trunc}: rel_err={err_trunc:.2e} exceeds 2 f16 ULP"
        );
        assert!(
            err_rtn   <= one_f16_rel * 2.0 || v.abs() < 1e-5,
            "f32_to_f16_bits_rtn({v}) -> {v_rtn}: rel_err={err_rtn:.2e} exceeds 2 f16 ULP"
        );
    }
}

// ---------------------------------------------------------------------------
// CPDT Part II S5: FullBuffer-sub-arm wrap envelope coverage
// ---------------------------------------------------------------------------
//
// The S5 wrap inside `emit_stdlib_optim_call` (stmt_fase.rs:608) emits the
// same dequant→step→quant envelope as `fase_emit_final_step` but for the
// stdlib optimizer FFI path. The structural difference vs the Deferred-arm
// wrap is the single-state alias handling: when SGD/Lion/Muon pass
// `s2 == s1` as a placeholder, the wrap casts s1 once, runs the step, casts
// back to s1's storage dtype, and frees the working tensor — without
// touching s2's slot. This test exercises that single-state alias path
// in-process using the same runtime ops the codegen emits.

const SGD_LR: f32 = 1e-2;
const SGD_MOMENTUM: f32 = 0.9;
const SGD_WD: f32 = 1e-4;

/// Pure-Rust SGD-with-momentum step matching `nsl.optim.sgd`'s recipe:
///   v_t   = momentum*v_{t-1} + g_t
///   theta = theta - lr*(v_t + wd*theta)
fn sgd_step_f32(theta: &mut [f32], v: &mut [f32], grad: &[f32]) {
    for i in 0..theta.len() {
        v[i] = SGD_MOMENTUM * v[i] + grad[i];
        theta[i] -= SGD_LR * (v[i] + SGD_WD * theta[i]);
    }
}

/// Mirrors what `emit_stdlib_optim_call` emits at FullBuffer-arm SGD with
/// `wrap_precision=true`: cast(s1, F32) → SGD step on F32 working tensor →
/// cast_into(orig_s1, work_s1) → free(work_s1). s2 aliases s1 and is not
/// independently touched (the `wrap_s2 = s1 != s2` guard skips its branch).
fn run_sgd_wrap_envelope(theta_init: &[f32]) -> Vec<f32> {
    let len = theta_init.len();
    let mut theta = theta_init.to_vec();

    // v_state at FP16 — single momentum buffer; mirrors state_list_1[i].
    let zeros = vec![0.0_f32; len];
    let tmp_f32 = make_f32_tensor(&zeros);
    let v_state = nsl_tensor_cast(tmp_f32, DTYPE_FP16);
    nsl_tensor_free(tmp_f32);

    for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);

        // S5 wrap envelope IN: dequant FP16 → F32 (orig→work cast).
        let v_work = nsl_tensor_cast(v_state, DTYPE_F32);

        // F32 SGD step (the stdlib FFI's equivalent — done in-process).
        let mut v_vals = read_f32_tensor(v_work, len);
        sgd_step_f32(&mut theta, &mut v_vals, &grad);
        let v_updated = make_f32_tensor(&v_vals);

        // S5 wrap envelope OUT: quant F32 → FP16 (work→orig cast_into).
        nsl_tensor_cast_into(v_state, v_updated);
        nsl_tensor_free(v_updated);
        nsl_tensor_free(v_work);
    }

    nsl_tensor_free(v_state);
    theta
}

/// Pure-Rust SGD reference with truncating-FP16 momentum (Mode B).
fn run_sgd_truncate_reference(theta_init: &[f32]) -> Vec<f32> {
    let len = theta_init.len();
    let mut theta = theta_init.to_vec();
    let mut v = vec![0.0_f32; len];
    for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);
        sgd_step_f32(&mut theta, &mut v, &grad);
        for vi in v.iter_mut() {
            *vi = trunc_fp16(*vi);
        }
    }
    theta
}

/// Pure-Rust SGD reference with full F32 momentum (Mode A — drift gate).
fn run_sgd_fp32_reference(theta_init: &[f32]) -> Vec<f32> {
    let len = theta_init.len();
    let mut theta = theta_init.to_vec();
    let mut v = vec![0.0_f32; len];
    for step in 1..=N_STEPS {
        let grad = gradients_for_step(step);
        sgd_step_f32(&mut theta, &mut v, &grad);
    }
    theta
}

/// CPDT Part II Sprint 5: validates the FullBuffer-arm wrap envelope for the
/// single-state alias path (SGD-class). Mirrors the AdamW validation suite
/// but exercises the `s1 == s2` branch instead of the multi-state branch.
///
/// Three-way comparison:
///   - Mode B (cast correctness): wrap-envelope ≡ truncating-Rust SGD (rel < 1e-5)
///   - Mode A (precision claim):  wrap-envelope ~ FP32-SGD reference   (rel < 5e-2)
///   - SGD-specific: the v_state buffer is the SOLE state slot — no v2/s2
///     aliasing artifacts (the codegen passes `wrap_s2=false` for SGD;
///     this test does the same by never allocating a second buffer).
#[test]
fn cpdt_fp16_sgd_fullbuffer_wrap_envelope_validation() {
    let theta_init = initial_params();

    let result_wrap     = run_sgd_wrap_envelope(&theta_init);
    let result_mode_b   = run_sgd_truncate_reference(&theta_init);
    let result_mode_a   = run_sgd_fp32_reference(&theta_init);

    let rel_err_b = max_rel_err(&result_wrap, &result_mode_b);
    println!(
        "[SGD Mode B] wrap-envelope vs truncating-Rust ref: max rel_err = {rel_err_b:.2e}"
    );
    assert!(
        rel_err_b < 1e-5,
        "SGD Mode B FAILED: wrap-envelope and truncating-Rust reference diverge by \
         {rel_err_b:.2e} (gate: < 1e-5). The S5 cast/cast_into emission does NOT \
         match pure-Rust truncation for the single-state path."
    );

    let rel_err_a = max_rel_err(&result_wrap, &result_mode_a);
    println!(
        "[SGD Mode A] wrap-envelope vs FP32 reference: max rel_err = {rel_err_a:.2e}"
    );
    assert!(
        rel_err_a < 5e-2,
        "SGD Mode A FAILED: FP16-momentum SGD diverges from FP32 SGD by {rel_err_a:.2e} \
         (gate: < 5e-2). FP16 momentum precision loss exceeds the EMA-bounded budget."
    );

    println!();
    println!("=== CPDT FP16 SGD FullBuffer-wrap-envelope validation: ALL PASSED ===");
    println!("  Mode B (cast correctness): rel_err = {rel_err_b:.2e}  < 1e-5  PASS");
    println!("  Mode A (FP16 vs FP32):     rel_err = {rel_err_a:.2e}  < 5e-2  PASS");
}

/// Spot-check: compiled-mechanism cast ops faithfully implement pure-Rust
/// truncation (bit-exact round-trip for a set of representative values).
#[test]
fn cast_ops_match_pure_rust_truncation() {
    // Values include normals, a small near-subnormal, and a negative.
    // Use std::f32::consts::PI instead of the literal to satisfy clippy::approx_constant.
    let vals = [0.1_f32, -0.5, std::f32::consts::PI, 1.0001, 100.0, -0.001953, 0.0625];
    let src_ptr = make_f32_tensor(&vals);
    let fp16_ptr = nsl_tensor_cast(src_ptr, DTYPE_FP16);
    let back_ptr = nsl_tensor_cast(fp16_ptr, DTYPE_F32);
    let got = read_f32_tensor(back_ptr, vals.len());

    for (i, &v) in vals.iter().enumerate() {
        let expected = trunc_fp16(v);
        assert_eq!(
            got[i].to_bits(),
            expected.to_bits(),
            "elem[{i}] v={v}: cast round-trip got {} expected {} (pure-Rust truncation)",
            got[i],
            expected
        );
    }

    nsl_tensor_free(src_ptr);
    nsl_tensor_free(fp16_ptr);
    nsl_tensor_free(back_ptr);
}
