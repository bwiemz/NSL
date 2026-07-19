//! P0.3: gradient-integrity gate — a runtime diagnostic that verifies, every
//! optimizer step, that the trainable parameters actually receive usable
//! gradients. It converts the silent failure mode from PR #396 (all parameter
//! gradients dropped, model "trains" on weight decay) into an observable,
//! assertable signal.
//!
//! Enabled by `--grad-integrity`. At process exit it prints, when
//! `NSL_GRAD_INTEGRITY=1`, the worst-case-over-steps snapshot:
//!
//! ```text
//! [grad-integrity]
//! checks=100
//! expected_params=146
//! gradient_params=146
//! finite=146
//! nonzero=144
//! missing=[]
//! ```
//!
//! Semantics (all counts are the WORST step observed, so a gate that reads
//! them proves the property held on EVERY step, not just the last):
//!   - `checks`          number of steps the gate ran (anti-vacuity: 0 ⇒ the
//!                       hook never fired, so the other lines mean nothing).
//!   - `expected_params` trainable parameter count (max seen; constant).
//!   - `gradient_params` params that received a non-null gradient tensor.
//!   - `finite`          of those, how many had no NaN/Inf.
//!   - `nonzero`         of those, how many had at least one nonzero element.
//!   - `missing`         param indices that received NO gradient on some step
//!                       (the #396 symptom) — the union over all steps.
//!
//! Two feed paths, one accumulator:
//!   - FullBuffer / composite path: `nsl_grad_integrity_check(grads, n)` scans
//!     the whole materialized grads list in one call.
//!   - FASE-Deferred path: the grads list is a null sentinel (gradients are
//!     consumed during backward lowering), so the emitted backward brackets a
//!     step with `nsl_grad_integrity_step_begin(n)`, calls
//!     `nsl_grad_integrity_note(grad_ptr, idx)` per parameter from the FASE
//!     hook, and closes it with `nsl_grad_integrity_step_end()`.

use std::collections::BTreeSet;
use std::sync::{Mutex, Once};

use crate::list::NslList;
use crate::tensor::{nsl_tensor_free, nsl_tensor_to_device, NslTensor};

#[derive(Default)]
struct StepAcc {
    /// Trainable parameter count for the in-progress step.
    expected: usize,
    /// One flag per parameter index: did it receive a non-null gradient?
    present: Vec<bool>,
    /// Of the present gradients, how many are finite (no NaN/Inf).
    finite: usize,
    /// Of the present gradients, how many have a nonzero element.
    nonzero: usize,
}

#[derive(Default)]
struct Aggregate {
    checks: u64,
    expected: usize,
    /// Worst (minimum) counts observed across all steps.
    min_gradient: Option<usize>,
    min_finite: Option<usize>,
    min_nonzero: Option<usize>,
    /// Union of parameter indices that were missing on some step.
    missing_union: BTreeSet<usize>,
}

#[derive(Default)]
struct State {
    agg: Aggregate,
    /// Set between `step_begin` and `step_end` on the FASE feed path.
    cur: Option<StepAcc>,
}

static STATE: Mutex<State> = Mutex::new(State {
    agg: Aggregate {
        checks: 0,
        expected: 0,
        min_gradient: None,
        min_finite: None,
        min_nonzero: None,
        missing_union: BTreeSet::new(),
    },
    cur: None,
});

/// Read one gradient tensor (device → host if needed) and classify it.
/// Returns `(finite, any_nonzero)`; a null pointer classifies as
/// `(false, false)` and is handled by the caller as "missing".
fn classify_grad(grad_ptr: i64) -> (bool, bool) {
    if grad_ptr == 0 {
        return (false, false);
    }
    let grad = NslTensor::from_ptr(grad_ptr);
    let (actual_ptr, needs_free) = if grad.device > 0 {
        (nsl_tensor_to_device(grad_ptr, 0), true)
    } else {
        (grad_ptr, false)
    };
    let grad = NslTensor::from_ptr(actual_ptr);
    let len = grad.len as usize;
    let mut has_nan = false;
    let mut has_inf = false;
    let mut any_nonzero = false;
    if grad.dtype == 1 {
        for j in 0..len {
            let v = unsafe { *grad.data_f32().add(j) } as f64;
            if v.is_nan() {
                has_nan = true;
            }
            if v.is_infinite() {
                has_inf = true;
            }
            if v != 0.0 {
                any_nonzero = true;
            }
        }
    } else {
        for j in 0..len {
            let v = unsafe { *grad.data_f64().add(j) };
            if v.is_nan() {
                has_nan = true;
            }
            if v.is_infinite() {
                has_inf = true;
            }
            if v != 0.0 {
                any_nonzero = true;
            }
        }
    }
    if needs_free {
        nsl_tensor_free(actual_ptr);
    }
    (!has_nan && !has_inf, any_nonzero)
}

/// Roll a finished step's accumulator into the global worst-case aggregate.
fn finalize(agg: &mut Aggregate, step: StepAcc) {
    let gradient = step.present.iter().filter(|p| **p).count();
    agg.checks += 1;
    agg.expected = agg.expected.max(step.expected);
    agg.min_gradient = Some(agg.min_gradient.map_or(gradient, |m| m.min(gradient)));
    agg.min_finite = Some(agg.min_finite.map_or(step.finite, |m| m.min(step.finite)));
    agg.min_nonzero = Some(agg.min_nonzero.map_or(step.nonzero, |m| m.min(step.nonzero)));
    for (i, present) in step.present.iter().enumerate() {
        if !present {
            agg.missing_union.insert(i);
        }
    }
}

/// FullBuffer / composite path: scan the whole materialized grads list once,
/// classify each parameter, and finalize the step.
#[no_mangle]
pub extern "C" fn nsl_grad_integrity_check(grads_list: i64, num_params: i64) {
    if grads_list == 0 || num_params <= 0 {
        return;
    }
    let grads = NslList::from_ptr(grads_list);
    let n = num_params as usize;
    let mut step = StepAcc {
        expected: n,
        present: vec![false; n],
        finite: 0,
        nonzero: 0,
    };
    for i in 0..n.min(grads.len as usize) {
        let grad_ptr = unsafe { *grads.data.add(i) };
        if grad_ptr == 0 {
            continue; // missing (present stays false)
        }
        step.present[i] = true;
        let (finite, nonzero) = classify_grad(grad_ptr);
        if finite {
            step.finite += 1;
        }
        if nonzero {
            step.nonzero += 1;
        }
    }
    let mut state = STATE.lock().unwrap();
    finalize(&mut state.agg, step);
}

/// FASE feed path: open a step accumulator sized to `num_params`.
#[no_mangle]
pub extern "C" fn nsl_grad_integrity_step_begin(num_params: i64) {
    if num_params <= 0 {
        return;
    }
    let n = num_params as usize;
    let mut state = STATE.lock().unwrap();
    state.cur = Some(StepAcc {
        expected: n,
        present: vec![false; n],
        finite: 0,
        nonzero: 0,
    });
}

/// FASE feed path: record one parameter's gradient by its index.
#[no_mangle]
pub extern "C" fn nsl_grad_integrity_note(grad_ptr: i64, param_idx: i64) {
    if param_idx < 0 {
        return;
    }
    let idx = param_idx as usize;
    // classify before taking the lock (the d2h copy is the slow part).
    let (finite, nonzero) = classify_grad(grad_ptr);
    let mut state = STATE.lock().unwrap();
    let Some(cur) = state.cur.as_mut() else {
        return; // note outside a begin/end bracket — ignore
    };
    if idx >= cur.present.len() {
        return;
    }
    if grad_ptr != 0 && !cur.present[idx] {
        cur.present[idx] = true;
        if finite {
            cur.finite += 1;
        }
        if nonzero {
            cur.nonzero += 1;
        }
    }
}

/// FASE feed path: close and finalize the current step accumulator.
#[no_mangle]
pub extern "C" fn nsl_grad_integrity_step_end() {
    let mut state = STATE.lock().unwrap();
    if let Some(step) = state.cur.take() {
        finalize(&mut state.agg, step);
    }
}

static ARM_ONCE: Once = Once::new();

/// Register the process-exit `[grad-integrity]` report exactly once. Emitted at
/// train setup when `--grad-integrity` is on, and also called from
/// `nsl_args_init` when `NSL_GRAD_INTEGRITY=1`, so either path prints the
/// snapshot and neither double-registers.
#[no_mangle]
pub extern "C" fn nsl_grad_integrity_arm() {
    ARM_ONCE.call_once(|| {
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(grad_integrity_atexit);
        }
    });
}

extern "C" fn grad_integrity_atexit() {
    let (checks, expected, gradient, finite, nonzero, missing) = grad_integrity_snapshot();
    let missing_str = missing
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    eprintln!("[grad-integrity]");
    eprintln!("checks={checks}");
    eprintln!("expected_params={expected}");
    eprintln!("gradient_params={gradient}");
    eprintln!("finite={finite}");
    eprintln!("nonzero={nonzero}");
    eprintln!("missing=[{missing_str}]");
}

/// Snapshot for the atexit report and for tests:
/// `(checks, expected, gradient, finite, nonzero, missing_indices)`.
pub fn grad_integrity_snapshot() -> (u64, usize, usize, usize, usize, Vec<usize>) {
    let state = STATE.lock().unwrap();
    let a = &state.agg;
    (
        a.checks,
        a.expected,
        a.min_gradient.unwrap_or(0),
        a.min_finite.unwrap_or(0),
        a.min_nonzero.unwrap_or(0),
        a.missing_union.iter().copied().collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests exercise the accumulator arithmetic directly (no tensors).
    fn reset() {
        let mut s = STATE.lock().unwrap();
        *s = State::default();
    }

    #[test]
    fn worst_case_over_steps_and_missing_union() {
        reset();
        // Step 1: 3 params, all present/finite, 2 nonzero.
        {
            let mut s = STATE.lock().unwrap();
            finalize(
                &mut s.agg,
                StepAcc { expected: 3, present: vec![true, true, true], finite: 3, nonzero: 2 },
            );
        }
        // Step 2: param 1 missing, finite 2, nonzero 2.
        {
            let mut s = STATE.lock().unwrap();
            finalize(
                &mut s.agg,
                StepAcc { expected: 3, present: vec![true, false, true], finite: 2, nonzero: 2 },
            );
        }
        let (checks, expected, gradient, finite, nonzero, missing) = grad_integrity_snapshot();
        assert_eq!(checks, 2);
        assert_eq!(expected, 3);
        assert_eq!(gradient, 2, "worst step had 2 present");
        assert_eq!(finite, 2, "worst step had 2 finite");
        assert_eq!(nonzero, 2);
        assert_eq!(missing, vec![1], "param 1 was missing on step 2");
    }
}
