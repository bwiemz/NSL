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
#[unsafe(no_mangle)]
pub extern "C" fn nsl_bias_correction_inv(base: f64, step: i64) -> f64 {
    let exponent = step as f64;
    let denom = 1.0 - base.powf(exponent);
    1.0 / denom
}

/// The text `nsl_train_partial_window_dropped` prints, without the
/// newline. Separate so a test can read it. `reason` is the code the
/// compiler bakes in: 0 for the layer-major window schedule, 1 for the
/// `NSL_FASE_FLUSH_TAIL=0` switch.
fn partial_window_dropped_message(pending: i64, window: i64, reason: i64) -> String {
    let why = match reason {
        1 => "NSL_FASE_FLUSH_TAIL=0 was set when this program was compiled, which turns \
              the final step on a partial window off",
        _ => "--layerwise-accum replays whole windows only and cannot step on a partial one",
    };
    format!(
        "nsl: warning: training ended {pending} micro-batch(es) into a \
         gradient-accumulation window of {window}, and {why}. The gradients of \
         those {pending} micro-batch(es) were not applied. Make the number of \
         micro-batches a multiple of grad_accumulation to use them."
    )
}

/// Report a trailing partial accumulation window that was not applied.
///
/// A train block normally steps once more at the end of the run on the
/// micro-batches a partial window holds. The layer-major window schedule
/// (`--layerwise-accum`) cannot, and a program compiled with
/// `NSL_FASE_FLUSH_TAIL=0` does not; both call this instead, so the
/// dropped gradients are at least not silent.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_train_partial_window_dropped(pending: i64, window: i64, reason: i64) {
    crate::nsl_log!(WARN, "train", "{}", partial_window_dropped_message(pending, window, reason));
}

#[cfg(test)]
mod tests {
    use super::{nsl_bias_correction_inv, partial_window_dropped_message};

    #[test]
    fn the_dropped_window_message_names_both_counts() {
        let m = partial_window_dropped_message(2, 4, 0);
        assert!(m.contains("ended 2 micro-batch(es)"), "{m}");
        assert!(m.contains("window of 4"), "{m}");
        assert!(m.contains("were not applied"), "{m}");
        assert!(m.contains("--layerwise-accum"), "{m}");
        let m = partial_window_dropped_message(1, 8, 1);
        assert!(m.contains("NSL_FASE_FLUSH_TAIL=0") && !m.contains("--layerwise-accum"), "{m}");
    }

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
