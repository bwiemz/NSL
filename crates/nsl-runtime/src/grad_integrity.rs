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
//! notes_expected=4
//! notes_observed=4..4
//! under_noted=[]
//! over_noted=[]
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
//!   - `notes_expected`  gradient contributions each param must receive per
//!                       bracket, declared by the emitter (0 ⇒ undeclared, and
//!                       then the two verdict lines below are always empty).
//!   - `notes_observed`  `min..max` of the per-(step, param) contribution
//!                       count over params that got at least one (params with
//!                       none are `missing`, not under-noted — see below).
//!   - `under_noted`     params that got 1..expected-1 contributions on some
//!                       step, i.e. a STRICT SUBSET of the window's
//!                       micro-batches reached them.
//!   - `over_noted`      params whose gradient was consumed MORE than
//!                       `notes_expected` times on some step.
//!
//! The first six lines are the frozen prefix: three consumers parse a fixed
//! six-line window after the header (two Rust gates plus
//! `models/benchmarks/p0_campaign.py`), so anything new must be APPENDED.
//!
//! Why the note COUNT is load-bearing, not decoration. Under CSLA the bracket
//! spans a whole accumulation window (see below), and repeat notes for one
//! param merge (`fin &= finite`, `nz |= nonzero`). A param that receives k of
//! the window's N micro-batch contributions therefore reads green on
//! `present` / `finite` / `nonzero` — while `fase_emit_accumulate` applies the
//! `1/N` scale PER NOTE, so its accumulated gradient is k/N of the correct
//! magnitude and biased toward the micro-batches that did contribute. That is
//! precisely the #396-class silent gradient loss this gate exists to convert
//! into a signal, and the window bracket had traded away the resolution to see
//! it (the per-micro-batch baseline bracket still catches it, as a `missing`).
//! `under_noted` restores it. `over_noted` is its mirror: a gradient consumed
//! twice double-counts into `m_partial` at the same `1/N` scale.
//!
//! `missing` deliberately keeps sole ownership of the count==0 case, so the
//! two verdict lists stay disjoint from it and from each other.
//!
//! Two feed paths, one accumulator:
//!   - FullBuffer / composite path: `nsl_grad_integrity_check(grads, n)` scans
//!     the whole materialized grads list in one call. Each param appears in
//!     the list exactly once, so `notes_expected` is 1 there.
//!   - FASE-Deferred path: the grads list is a null sentinel (gradients are
//!     consumed during backward lowering), so the emitted backward brackets a
//!     step with `nsl_grad_integrity_step_begin(n, expected_notes_per_param)`,
//!     calls `nsl_grad_integrity_note(grad_ptr, idx)` per parameter from the
//!     FASE hook, and closes it with `nsl_grad_integrity_step_end()`.
//!
//! Bracket granularity differs by path: the FASE-interleaved backward
//! brackets each MICRO-batch (checks counts micros, `notes_expected` = 1), the
//! CSLA `--layerwise-accum` windowed replay brackets each accumulation WINDOW
//! (checks counts optimizer steps, `notes_expected` = the window's
//! `grad_accumulation`; each param is noted once per micro-batch and repeat
//! notes MERGE — finite ANDs, nonzero ORs — so the report attests the whole
//! window's partials, not just the first). Under CSLA on GPU each note pays a
//! device sync + D2H copy per (param, micro-batch) — an acceptable
//! observe-only diagnostic cost, but expect it to serialize streams and taint
//! (self-healing) cuda-graph capture regions.

use std::collections::BTreeSet;
use std::sync::{Mutex, Once, OnceLock};

use crate::list::NslList;
use crate::tensor::{nsl_tensor_free, nsl_tensor_to_device, NslTensor};

#[derive(Default)]
struct StepAcc {
    /// Trainable parameter count for the in-progress step.
    expected: usize,
    /// How many gradient contributions each parameter index received in this
    /// bracket. `present` is `count > 0`; the count itself is what makes a
    /// partially-noted parameter (k of N micro-batches) visible — see the
    /// module docs.
    counts: Vec<u32>,
    /// Contributions each parameter is expected to receive in this bracket,
    /// declared by the emitter at `step_begin`. 0 means "not declared", which
    /// disables the under/over verdict rather than guessing a value.
    expected_notes: u32,
    /// Per-index classification, valid where the count is nonzero. Repeat
    /// notes for the same index MERGE (`fin &= finite`, `nz |= nonzero`):
    /// under the CSLA window-scoped bracket a parameter is noted once per
    /// micro-batch, and a NaN/Inf in ANY partial poisons the accumulated
    /// gradient, while a nonzero in ANY partial makes the window sum
    /// (approximately) nonzero. First-wins here would attest only micro-batch
    /// 0's partial.
    fin: Vec<bool>,
    nz: Vec<bool>,
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
    /// Largest `expected_notes` any bracket declared (0 ⇒ none did).
    notes_expected: u32,
    /// Extremes of the per-(step, param) contribution count, over the params
    /// that received at least one. `None` until the first note lands.
    notes_min: Option<u32>,
    notes_max: Option<u32>,
    /// Union of parameter indices whose contribution count deviated from the
    /// declared expectation on some step. Disjoint from `missing_union` by
    /// construction: a count of 0 is `missing`, never `under`.
    under_union: BTreeSet<usize>,
    over_union: BTreeSet<usize>,
}

#[derive(Default)]
struct State {
    agg: Aggregate,
    /// Set between `step_begin` and `step_end` on the FASE feed path.
    cur: Option<StepAcc>,
    /// `NSL_GRAD_INTEGRITY_FAULT=drop:<idx>` only: has this bracket already
    /// swallowed the target's one suppressed contribution? Held here rather
    /// than inferred from `counts[idx] == 0`, which never advances on a
    /// suppressed note and would therefore swallow ALL of them — turning the
    /// k-of-N fault into a plain `missing` and testing the wrong field.
    fault_dropped_this_bracket: bool,
}

static STATE: Mutex<State> = Mutex::new(State {
    agg: Aggregate {
        checks: 0,
        expected: 0,
        min_gradient: None,
        min_finite: None,
        min_nonzero: None,
        missing_union: BTreeSet::new(),
        notes_expected: 0,
        notes_min: None,
        notes_max: None,
        under_union: BTreeSet::new(),
        over_union: BTreeSet::new(),
    },
    cur: None,
    fault_dropped_this_bracket: false,
});

/// Test-only fault injection, same convention as `NSL_PARAM_PLAN_FAULT` /
/// `NSL_CPDT_FORCE_STALE_PLAN`: `NSL_GRAD_INTEGRITY_FAULT=double:<idx>` or
/// `=drop:<idx>`.
///
/// It exists because the one-contribution-per-(param, micro-batch) invariant
/// is STRUCTURALLY true today — `partition_ranges` is a proven partition and
/// a Wengert result VarId has exactly one producing op, so the FASE hook
/// fires once per param per replay. The under/over verdict therefore reads
/// green on every fixture in the tree, and a stub that always reported `[]`
/// would be indistinguishable from the real check. These two knobs make the
/// failing arm reachable: `drop` suppresses a param's FIRST contribution in
/// each bracket (leaving the other N-1 — the k-of-N silent-scaling defect,
/// which `missing` cannot see because the count is still nonzero), and
/// `double` records every contribution twice (consumed-more-than-once).
///
/// Applies to the FASE note path only — the composite scan reads the
/// materialized grads list directly and has no per-contribution hook.
#[derive(Clone, Copy)]
enum Fault {
    Double(usize),
    Drop(usize),
}

fn fault() -> Option<Fault> {
    static FAULT: OnceLock<Option<Fault>> = OnceLock::new();
    *FAULT.get_or_init(|| {
        let raw = std::env::var("NSL_GRAD_INTEGRITY_FAULT").ok()?;
        let (kind, idx) = raw.trim().split_once(':')?;
        let idx: usize = idx.trim().parse().ok()?;
        let f = match kind.trim() {
            "double" => Fault::Double(idx),
            "drop" => Fault::Drop(idx),
            _ => return None,
        };
        eprintln!(
            "[grad-integrity] FAULT INJECTION ACTIVE (NSL_GRAD_INTEGRITY_FAULT={}): \
             parameter {idx}'s gradient contributions will be miscounted on purpose. \
             This is a test hook; never set it for a real run.",
            raw.trim()
        );
        Some(f)
    })
}

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

/// Record one gradient contribution for `idx`. Split out of
/// [`nsl_grad_integrity_note`] so the `double` fault can replay it verbatim
/// instead of hand-rolling a second, subtly-different update.
fn record(cur: &mut StepAcc, idx: usize, finite: bool, nonzero: bool) {
    if cur.counts[idx] == 0 {
        cur.fin[idx] = finite;
        cur.nz[idx] = nonzero;
    } else {
        // Repeat note (CSLA window bracket: one per micro-batch): a NaN/Inf
        // in ANY partial poisons the accumulated gradient; a nonzero in ANY
        // partial makes the window sum nonzero.
        cur.fin[idx] &= finite;
        cur.nz[idx] |= nonzero;
    }
    cur.counts[idx] = cur.counts[idx].saturating_add(1);
}

/// Roll a finished step's accumulator into the global worst-case aggregate.
fn finalize(agg: &mut Aggregate, step: StepAcc) {
    let gradient = step.counts.iter().filter(|c| **c > 0).count();
    let finite = step
        .counts
        .iter()
        .zip(step.fin.iter())
        .filter(|(c, f)| **c > 0 && **f)
        .count();
    let nonzero = step
        .counts
        .iter()
        .zip(step.nz.iter())
        .filter(|(c, z)| **c > 0 && **z)
        .count();
    agg.checks += 1;
    agg.expected = agg.expected.max(step.expected);
    agg.min_gradient = Some(agg.min_gradient.map_or(gradient, |m| m.min(gradient)));
    agg.min_finite = Some(agg.min_finite.map_or(finite, |m| m.min(finite)));
    agg.min_nonzero = Some(agg.min_nonzero.map_or(nonzero, |m| m.min(nonzero)));
    agg.notes_expected = agg.notes_expected.max(step.expected_notes);
    for (i, &count) in step.counts.iter().enumerate() {
        if count == 0 {
            // `missing` owns the zero case exclusively — reporting it in
            // `under_noted` too would make the new line restate the old one
            // on every frozen/unreached parameter and bury the k-of-N signal.
            agg.missing_union.insert(i);
            continue;
        }
        agg.notes_min = Some(agg.notes_min.map_or(count, |m| m.min(count)));
        agg.notes_max = Some(agg.notes_max.map_or(count, |m| m.max(count)));
        if step.expected_notes == 0 {
            continue; // undeclared bracket: observe only, no verdict
        }
        if count < step.expected_notes {
            agg.under_union.insert(i);
        } else if count > step.expected_notes {
            agg.over_union.insert(i);
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
        counts: vec![0; n],
        // The list holds at most one gradient per parameter slot, so a slot
        // can never be over-noted here; declaring 1 keeps the verdict lines
        // meaningful (rather than "undeclared") on this path too.
        expected_notes: 1,
        fin: vec![false; n],
        nz: vec![false; n],
    };
    for i in 0..n.min(grads.len as usize) {
        let grad_ptr = unsafe { *grads.data.add(i) };
        if grad_ptr == 0 {
            continue; // missing (the count stays 0)
        }
        let (finite, nonzero) = classify_grad(grad_ptr);
        record(&mut step, i, finite, nonzero);
    }
    let mut state = STATE.lock().unwrap();
    finalize(&mut state.agg, step);
}

/// FASE feed path: open a step accumulator sized to `num_params`.
///
/// `expected_notes_per_param` is how many gradient contributions each
/// parameter must receive before the matching `step_end` — 1 for the
/// per-micro-batch baseline bracket, `grad_accumulation` for the CSLA
/// window bracket. Pass 0 only if the emitter genuinely cannot know it;
/// the report then observes the counts without judging them.
#[no_mangle]
pub extern "C" fn nsl_grad_integrity_step_begin(num_params: i64, expected_notes_per_param: i64) {
    if num_params <= 0 {
        return;
    }
    let n = num_params as usize;
    let mut state = STATE.lock().unwrap();
    state.cur = Some(StepAcc {
        expected: n,
        counts: vec![0; n],
        expected_notes: expected_notes_per_param.max(0) as u32,
        fin: vec![false; n],
        nz: vec![false; n],
    });
    state.fault_dropped_this_bracket = false;
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
    // Split the borrow: the drop fault reads/writes the bracket flag on
    // `State` while recording borrows `cur`.
    let State {
        cur,
        fault_dropped_this_bracket,
        ..
    } = &mut *state;
    let Some(cur) = cur.as_mut() else {
        return; // note outside a begin/end bracket — ignore
    };
    if idx >= cur.counts.len() || grad_ptr == 0 {
        return;
    }
    match fault() {
        // Suppress the bracket's FIRST contribution for the target param,
        // exactly as if the emitted backward had skipped one micro-batch's
        // hook call. The bracket's remaining N-1 still land, so the count
        // stays nonzero and only `under_noted` can see the loss.
        Some(Fault::Drop(t)) if t == idx && !*fault_dropped_this_bracket => {
            *fault_dropped_this_bracket = true;
            return;
        }
        Some(Fault::Double(t)) if t == idx => {
            record(cur, idx, finite, nonzero);
            record(cur, idx, finite, nonzero);
            return;
        }
        _ => {}
    }
    record(cur, idx, finite, nonzero);
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
    let notes = grad_integrity_notes_snapshot();
    let list = |v: &[usize]| {
        v.iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    };
    eprintln!("[grad-integrity]");
    eprintln!("checks={checks}");
    eprintln!("expected_params={expected}");
    eprintln!("gradient_params={gradient}");
    eprintln!("finite={finite}");
    eprintln!("nonzero={nonzero}");
    eprintln!("missing=[{}]", list(&missing));
    // APPEND-ONLY below this line: three consumers parse a fixed six-line
    // window after the header (crates/nsl-cli/tests/grad_integrity_gate.rs,
    // crates/nsl-cli/tests/csla_layerwise_gate.rs, and
    // models/benchmarks/p0_campaign.py). Inserting a field above would push
    // `missing` out of that window and break all three silently.
    eprintln!("notes_expected={}", notes.expected);
    eprintln!("notes_observed={}..{}", notes.min, notes.max);
    eprintln!("under_noted=[{}]", list(&notes.under));
    eprintln!("over_noted=[{}]", list(&notes.over));
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

/// Contribution-count half of the snapshot (kept separate so the six-field
/// tuple every existing caller destructures keeps its shape).
pub struct NotesSnapshot {
    /// Declared contributions per parameter per bracket (0 ⇒ undeclared).
    pub expected: u32,
    /// Extremes of the observed count over params that got at least one; both
    /// 0 when no note ever landed.
    pub min: u32,
    pub max: u32,
    /// Params that got strictly fewer / more than `expected` on some step.
    pub under: Vec<usize>,
    pub over: Vec<usize>,
}

pub fn grad_integrity_notes_snapshot() -> NotesSnapshot {
    let state = STATE.lock().unwrap();
    let a = &state.agg;
    NotesSnapshot {
        expected: a.notes_expected,
        min: a.notes_min.unwrap_or(0),
        max: a.notes_max.unwrap_or(0),
        under: a.under_union.iter().copied().collect(),
        over: a.over_union.iter().copied().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests exercise the accumulator arithmetic directly (no tensors).
    // `STATE` is process-global, so they must not interleave: cargo runs the
    // module's tests on parallel threads and a second `reset()` mid-test
    // silently erases the first test's aggregate. `reset` hands back a guard
    // held for the caller's whole body. `into_inner` on poison keeps one
    // failing test from cascading into the others.
    static TEST_SERIAL: Mutex<()> = Mutex::new(());

    fn reset() -> std::sync::MutexGuard<'static, ()> {
        let g = TEST_SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut s = STATE.lock().unwrap();
        *s = State::default();
        g
    }

    #[test]
    fn worst_case_over_steps_and_missing_union() {
        let _serial = reset();
        // Step 1: 3 params, all present/finite, 2 nonzero.
        {
            let mut s = STATE.lock().unwrap();
            finalize(
                &mut s.agg,
                StepAcc {
                    expected: 3,
                    counts: vec![1, 1, 1],
                    expected_notes: 1,
                    fin: vec![true, true, true],
                    nz: vec![true, true, false],
                },
            );
        }
        // Step 2: param 1 missing, finite 2, nonzero 2.
        {
            let mut s = STATE.lock().unwrap();
            finalize(
                &mut s.agg,
                StepAcc {
                    expected: 3,
                    counts: vec![1, 0, 1],
                    expected_notes: 1,
                    fin: vec![true, false, true],
                    nz: vec![true, false, true],
                },
            );
        }
        let (checks, expected, gradient, finite, nonzero, missing) = grad_integrity_snapshot();
        assert_eq!(checks, 2);
        assert_eq!(expected, 3);
        assert_eq!(gradient, 2, "worst step had 2 present");
        assert_eq!(finite, 2, "worst step had 2 finite");
        assert_eq!(nonzero, 2);
        assert_eq!(missing, vec![1], "param 1 was missing on step 2");
        // A count of 0 belongs to `missing` alone — the two verdict lists
        // must stay disjoint from it, or the new line just restates the old.
        let n = grad_integrity_notes_snapshot();
        assert!(n.under.is_empty(), "count 0 is `missing`, not under-noted");
        assert!(n.over.is_empty());
    }

    /// The k-of-N case the CSLA window bracket could not see: `present`,
    /// `finite` and `nonzero` are all fully green while param 1 received one
    /// of the window's four micro-batch contributions.
    #[test]
    fn partial_contributions_are_under_noted_while_every_old_field_reads_green() {
        let _serial = reset();
        {
            let mut s = STATE.lock().unwrap();
            finalize(
                &mut s.agg,
                StepAcc {
                    expected: 3,
                    counts: vec![4, 1, 8],
                    expected_notes: 4,
                    fin: vec![true, true, true],
                    nz: vec![true, true, true],
                },
            );
        }
        let (checks, expected, gradient, finite, nonzero, missing) = grad_integrity_snapshot();
        assert_eq!((checks, expected), (1, 3));
        assert_eq!(
            (gradient, finite, nonzero, missing.len()),
            (3, 3, 3, 0),
            "the pre-existing fields cannot distinguish k-of-N — that is the defect"
        );
        let n = grad_integrity_notes_snapshot();
        assert_eq!(n.expected, 4);
        assert_eq!((n.min, n.max), (1, 8));
        assert_eq!(n.under, vec![1], "param 1 got 1 of 4 contributions");
        assert_eq!(n.over, vec![2], "param 2 was consumed 8 times, not 4");
    }

    /// An undeclared bracket (`expected_notes == 0`) observes without judging:
    /// a wrong verdict is worse than none, so no expectation ⇒ no verdict.
    #[test]
    fn undeclared_expectation_reports_range_but_no_verdict() {
        let _serial = reset();
        {
            let mut s = STATE.lock().unwrap();
            finalize(
                &mut s.agg,
                StepAcc {
                    expected: 2,
                    counts: vec![1, 7],
                    expected_notes: 0,
                    fin: vec![true, true],
                    nz: vec![true, true],
                },
            );
        }
        let n = grad_integrity_notes_snapshot();
        assert_eq!(n.expected, 0);
        assert_eq!((n.min, n.max), (1, 7));
        assert!(n.under.is_empty() && n.over.is_empty());
    }
}
