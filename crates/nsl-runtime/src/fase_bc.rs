//! FASE bias-correction scalar helper.
//!
//! `nsl_bias_correction_inv(base, step)` returns `1 / (1 - base^step)` —
//! the scalar factor that turns a raw moment into a bias-corrected moment
//! in Adam/AdamW.  Computed once per optimizer step (not per parameter),
//! so a single FFI call per β is negligible cost.

/// Compute `1.0 / (1.0 - base^step)` — the bias-correction divisor's
/// inverse, so callers can multiply rather than divide.
///
/// Called once per optimizer step from compiled FASE Deferred code.
/// No unwinding: returns `f64::INFINITY` or `f64::NAN` for degenerate
/// inputs rather than panicking (FFI boundary).
#[no_mangle]
pub extern "C" fn nsl_bias_correction_inv(base: f64, step: i64) -> f64 {
    let exponent = step as f64;
    let denom = 1.0 - base.powf(exponent);
    1.0 / denom
}

#[cfg(test)]
mod tests {
    use super::nsl_bias_correction_inv;

    #[test]
    fn matches_reference_for_known_step() {
        let v = nsl_bias_correction_inv(0.9, 1);
        assert!((v - 10.0).abs() < 1e-12, "got {}", v);
    }

    #[test]
    fn matches_reference_for_large_step() {
        let expected = 1.0 / (1.0 - 0.999_f64.powf(100.0));
        let v = nsl_bias_correction_inv(0.999, 100);
        assert!((v - expected).abs() < 1e-9, "got {} want {}", v, expected);
    }

    #[test]
    fn step_zero_produces_infinity_or_nan_without_panic() {
        let v = nsl_bias_correction_inv(0.9, 0);
        assert!(v.is_infinite() || v.is_nan());
    }
}
