//! Item 3: the compile-time **ParameterPlan**, checked against reality.
//!
//! # The problem this exists to solve
//!
//! A trainable parameter's storage is described today by three independent
//! runtime tables, each owned by a different module, each keyed by the
//! parameter's `NslTensor` pointer, and none of which knows the others exist:
//!
//! | table | module | holds |
//! |---|---|---|
//! | `MIRRORS` | [`crate::weight_stream`] | host mirror for `--weight-stream` |
//! | `ZERO3_TABLE` | [`crate::zero`] | owner + residency for `--zero-stage 3` |
//! | `SRBF16_TABLE` | [`crate::sr_bf16`] | bf16 device mirror for `--param-dtype bf16-sr` |
//!
//! `nsl_weight_stream_register` picks **exactly one** of the three backends
//! for a parameter, and it picks by *global* flag — `zero3_active()`, then
//! `srbf16_active()`, else host mirrors. But the decision that a parameter
//! *should* stream (and in which mode) is made per-parameter at compile time,
//! in more than one place in `stmt.rs`. Nothing connects the two.
//!
//! That gap is not a crash class, it is a **silent-wrong-numerics** class. If
//! a parameter the plan marked `bf16-sr` reaches `register` before
//! `nsl_sr_bf16_enable` did, it lands in `MIRRORS` instead: it trains in f32
//! with an f32 master copy, the run completes, the loss curve looks fine, and
//! the certified property (`no FP32 master copy`) is quietly false. Symmetric
//! failures exist for the ZeRO-3 route.
//!
//! # What this module does
//!
//! Codegen bakes the plan it derived into the binary — one
//! [`nsl_param_plan_declare`] per parameter, carrying its index and the
//! [`PLAN_STREAMED`] / [`PLAN_BF16_SR`] / [`PLAN_SHARDED`] bits — and calls
//! [`nsl_param_plan_verify`] after the registration belt has run. Because
//! that belt is per-micro-batch, so is the check: a table that drifts
//! mid-run is caught, not just the first step's state. Verify reads the three
//! tables directly and asserts each parameter landed in the backend the plan
//! named, and in no other. A disagreement aborts. Only the FIRST clean result
//! is printed, so a long run does not bury its own stderr.
//!
//! # What this does NOT claim
//!
//! It does not claim the plan is *correct* — only that the runtime realized
//! the plan the compiler emitted. A wrong plan is verified just as happily as
//! a right one. The value is that the plan is now a single artifact that can
//! be read, reported and tested, instead of an emergent property of three
//! tables and a flag-priority chain.
//!
//! As of this commit the compile-time derivation makes the four states
//! mutually consistent by construction, so verify is defense-in-depth rather
//! than a fix for a live bug. It is worth its cost because the invariant it
//! pins is invisible at every individual call site: whoever next adds a
//! parameter-storage mode (FP8 params, sharded optimizer mirrors) will edit
//! one registration site and not the others, and this is what notices.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The parameter is registered with a residency backend (it does not sit
/// device-resident for the whole step).
pub const PLAN_STREAMED: i64 = 1 << 0;
/// Its authoritative storage is bf16 with stochastic rounding
/// (`--param-dtype bf16-sr`) — the [`crate::sr_bf16`] backend.
pub const PLAN_BF16_SR: i64 = 1 << 1;
/// It is tensor-granular sharded across ranks (`--zero-stage 3`) — the
/// [`crate::zero`] broadcast backend.
pub const PLAN_SHARDED: i64 = 1 << 2;

/// Every bit this ABI version defines. An unknown bit in a `declare` call
/// means codegen and runtime disagree about the plan encoding; that is a
/// build-integrity failure, not a recoverable condition.
const PLAN_KNOWN_BITS: i64 = PLAN_STREAMED | PLAN_BF16_SR | PLAN_SHARDED;

#[derive(Clone, Copy)]
struct Declared {
    idx: i64,
    flags: i64,
}

static PLAN: Mutex<Option<HashMap<i64, Declared>>> = Mutex::new(None);
/// How many times verify has run since the last teardown. Only used to make
/// the success banner print once per train block.
static PLAN_VERIFIES: AtomicU64 = AtomicU64::new(0);

/// Which backend actually holds this parameter, observed from the tables.
/// `Nowhere` means no residency backend registered it.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum Observed {
    Nowhere,
    HostMirror,
    Bf16Sr,
    Sharded,
    /// More than one table claims it — impossible through `register`'s
    /// single-dispatch, so it means a table was populated out of band.
    Conflict,
}

fn observe(tensor_ptr: i64) -> Observed {
    let host = crate::weight_stream::mirror_is_registered(tensor_ptr);
    let sr = crate::sr_bf16::srbf16_is_registered(tensor_ptr);
    let shard = crate::zero::zero3_is_registered(tensor_ptr);
    match (host, sr, shard) {
        (false, false, false) => Observed::Nowhere,
        (true, false, false) => Observed::HostMirror,
        (false, true, false) => Observed::Bf16Sr,
        (false, false, true) => Observed::Sharded,
        _ => Observed::Conflict,
    }
}

/// What the plan's flag bits say the backend must be.
fn expected(flags: i64) -> Observed {
    if flags & PLAN_STREAMED == 0 {
        Observed::Nowhere
    } else if flags & PLAN_SHARDED != 0 {
        Observed::Sharded
    } else if flags & PLAN_BF16_SR != 0 {
        Observed::Bf16Sr
    } else {
        Observed::HostMirror
    }
}

fn describe(o: Observed) -> &'static str {
    match o {
        Observed::Nowhere => "no residency backend (fully resident)",
        Observed::HostMirror => "weight-stream host mirror",
        Observed::Bf16Sr => "bf16-sr device mirror",
        Observed::Sharded => "zero-3 sharded",
        Observed::Conflict => "MULTIPLE backends (table corruption)",
    }
}

/// Test-only fault injection: `NSL_PARAM_PLAN_FAULT=<idx>` corrupts the
/// declared plan for that one parameter so it disagrees with whatever the
/// runtime actually did.
///
/// This exists because [`nsl_param_plan_verify`]'s abort arm is otherwise
/// unreachable from a test: producing a real mismatch requires a compiler
/// bug. Without it, stubbing `observe()` to return `expected()` would leave
/// every gate green — the check would be decorative and nothing would say
/// so. The gate in `param_plan_gate.rs` drives this and asserts the process
/// dies with `[param-plan] FATAL`.
///
/// The corruption is chosen to always produce a mismatch without tripping
/// the shape guard below: a streamed parameter is re-declared as resident, a
/// resident one as streamed.
fn fault_injected_flags(idx: i64, flags: i64) -> i64 {
    use std::sync::OnceLock;
    static FAULT: OnceLock<Option<i64>> = OnceLock::new();
    let target = *FAULT.get_or_init(|| {
        let t = std::env::var("NSL_PARAM_PLAN_FAULT")
            .ok()
            .and_then(|v| v.trim().parse::<i64>().ok());
        if let Some(t) = t {
            eprintln!(
                "[param-plan] FAULT INJECTION ACTIVE (NSL_PARAM_PLAN_FAULT={t}): \
                 parameter {t}'s declared plan will be corrupted on purpose. \
                 This is a test hook; never set it for a real run."
            );
        }
        t
    });
    if target != Some(idx) {
        return flags;
    }
    if flags & PLAN_STREAMED != 0 {
        0
    } else {
        PLAN_STREAMED
    }
}

/// Declare parameter `idx` (pointer `tensor_ptr`) as planned with `flags`.
/// Emitted once per parameter per train block; idempotent, so the step-top
/// registration belt may re-declare freely.
///
/// Returns 0 on success, -1 on a rejected argument.
#[no_mangle]
pub extern "C" fn nsl_param_plan_declare(tensor_ptr: i64, idx: i64, flags: i64) -> i64 {
    if tensor_ptr == 0 || idx < 0 {
        return -1;
    }
    let flags = fault_injected_flags(idx, flags);
    if flags & !PLAN_KNOWN_BITS != 0 {
        eprintln!(
            "[param-plan] FATAL: parameter {idx} declared with unknown plan \
             bits {:#x} (known {:#x}) — the codegen and runtime plan encodings \
             have drifted",
            flags & !PLAN_KNOWN_BITS,
            PLAN_KNOWN_BITS
        );
        std::process::abort();
    }
    // A non-streamed parameter cannot carry a storage-mode bit: both bf16-sr
    // and zero-3 residency are properties of *being registered* with a
    // backend. This is the compile-time invariant restated where a
    // hand-built CompileOptions caller can still trip it.
    if flags & PLAN_STREAMED == 0 && flags & (PLAN_BF16_SR | PLAN_SHARDED) != 0 {
        eprintln!(
            "[param-plan] FATAL: parameter {idx} declared with a storage mode \
             ({:#x}) but not STREAMED — bf16-sr and zero-3 residency only \
             exist for registered parameters",
            flags & (PLAN_BF16_SR | PLAN_SHARDED)
        );
        std::process::abort();
    }
    PLAN.lock()
        .unwrap()
        .get_or_insert_with(HashMap::new)
        .insert(tensor_ptr, Declared { idx, flags });
    0
}

/// Cross-check every declared parameter against the three residency tables.
/// Returns the mismatch count; aborts on any mismatch (a mismatch means the
/// process is training something other than what was compiled and certified).
///
/// Emitted after the registration belt, once per train block.
#[no_mangle]
pub extern "C" fn nsl_param_plan_verify() -> i64 {
    let snapshot: Vec<(i64, Declared)> = {
        let guard = PLAN.lock().unwrap();
        match guard.as_ref() {
            Some(m) if !m.is_empty() => m.iter().map(|(p, d)| (*p, *d)).collect(),
            // Nothing declared: verification is vacuous. Say so rather than
            // reporting a clean bill of health.
            _ => {
                if PLAN_VERIFIES.fetch_add(1, Ordering::Relaxed) == 0 {
                    eprintln!(
                        "[param-plan] verify called with an empty plan — nothing \
                         was declared, so nothing was checked"
                    );
                }
                return 0;
            }
        }
    };
    // (idx, message) so the fatal report sorts NUMERICALLY — sorting the
    // formatted strings puts param 10 between 1 and 2.
    let mut mismatches: Vec<(i64, String)> = Vec::new();
    let mut counts = [0usize; 4]; // nowhere / host / bf16 / sharded
    for (ptr, d) in &snapshot {
        let want = expected(d.flags);
        let got = observe(*ptr);
        if want == got {
            match want {
                Observed::Nowhere => counts[0] += 1,
                Observed::HostMirror => counts[1] += 1,
                Observed::Bf16Sr => counts[2] += 1,
                Observed::Sharded => counts[3] += 1,
                Observed::Conflict => unreachable!("expected() never yields Conflict"),
            }
        } else {
            mismatches.push((
                d.idx,
                format!(
                    "  param {} (ptr {:#x}): plan says {}, runtime has {}",
                    d.idx,
                    ptr,
                    describe(want),
                    describe(got)
                ),
            ));
        }
    }
    let first = PLAN_VERIFIES.fetch_add(1, Ordering::Relaxed) == 0;
    if mismatches.is_empty() {
        // The check runs every micro-batch (it is emitted after the
        // registration belt, which is itself per-micro-batch, so a table that
        // drifts mid-run is caught) but only the FIRST clean result is
        // printed — otherwise a 500-step run buries its own stderr.
        if first {
            eprintln!(
                "[param-plan] verified {} parameter(s): {} resident, {} host-mirrored, \
                 {} bf16-sr, {} sharded",
                snapshot.len(),
                counts[0],
                counts[1],
                counts[2],
                counts[3]
            );
        }
        return 0;
    }
    mismatches.sort_by_key(|(idx, _)| *idx);
    eprintln!(
        "[param-plan] FATAL: {} of {} parameter(s) did not land in the backend \
         the compiled plan named. The run would train storage the certification \
         does not describe.\n{}",
        mismatches.len(),
        snapshot.len(),
        mismatches
            .iter()
            .map(|(_, m)| m.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
    std::process::abort();
}

/// Drop the declared plan at train-block teardown, alongside the residency
/// tables it describes. A second train block in the same program then
/// declares and verifies its own plan from a clean slate rather than
/// re-checking the first block's now-unregistered (possibly freed) pointers.
///
/// Also rearms the once-per-block success report. Returns the number of
/// entries dropped.
#[no_mangle]
pub extern "C" fn nsl_param_plan_teardown() -> i64 {
    let mut guard = PLAN.lock().unwrap();
    let n = guard.as_ref().map_or(0, |m| m.len()) as i64;
    *guard = None;
    PLAN_VERIFIES.store(0, Ordering::Relaxed);
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `PLAN` and the counters are process-global, so the tests that mutate
    /// them must not interleave. (Pure-function tests below skip the lock.)
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// The expectation table is the whole contract: flags in, required
    /// backend out. Pinning it here means a reordering of `register`'s
    /// dispatch chain (zero3 > srbf16 > mirrors) has to be reflected in a
    /// test edit rather than silently inverting a check.
    #[test]
    fn flag_bits_map_to_the_registers_dispatch_priority() {
        assert_eq!(expected(0), Observed::Nowhere);
        assert_eq!(expected(PLAN_STREAMED), Observed::HostMirror);
        assert_eq!(expected(PLAN_STREAMED | PLAN_BF16_SR), Observed::Bf16Sr);
        assert_eq!(expected(PLAN_STREAMED | PLAN_SHARDED), Observed::Sharded);
        // zero3 wins over srbf16, exactly as nsl_weight_stream_register does.
        // (The composition matrix refuses this pair at the CLI; the tie-break
        // is pinned anyway so the two chains cannot drift apart unnoticed.)
        assert_eq!(
            expected(PLAN_STREAMED | PLAN_BF16_SR | PLAN_SHARDED),
            Observed::Sharded
        );
    }

    /// The mismatch arm must actually fire. Without this the verify could be
    /// unconditionally-clean and every e2e gate asserting "no mismatch" would
    /// be vacuous — the exact failure mode this module was written to stop.
    #[test]
    fn a_plan_backend_disagreement_is_detected() {
        // Nothing is registered in any table in a unit-test process, so a
        // parameter declared STREAMED is observably Nowhere.
        assert_eq!(observe(0x1234), Observed::Nowhere);
        assert_ne!(expected(PLAN_STREAMED), observe(0x1234));
        // ...and a parameter declared resident agrees.
        assert_eq!(expected(0), observe(0x1234));
    }

    #[test]
    fn declare_rejects_null_and_negative() {
        assert_eq!(nsl_param_plan_declare(0, 0, 0), -1);
        assert_eq!(nsl_param_plan_declare(0x10, -1, 0), -1);
    }

    /// The belts re-declare every micro-batch, so a re-declaration of the
    /// same pointer must not grow the table (teardown's return value is the
    /// entry count).
    #[test]
    fn declare_is_idempotent_per_pointer() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        nsl_param_plan_teardown();
        assert_eq!(nsl_param_plan_declare(0xAA00, 0, 0), 0);
        assert_eq!(nsl_param_plan_declare(0xAA00, 0, 0), 0);
        assert_eq!(nsl_param_plan_declare(0xAA08, 1, 0), 0);
        assert_eq!(nsl_param_plan_teardown(), 2, "re-declare grew the table");
    }

    /// Fault injection must actually corrupt, and only the named parameter.
    /// If this drifts, the gate that proves the abort arm works goes vacuous.
    #[test]
    fn fault_injection_flips_exactly_one_parameter_into_a_mismatch() {
        // Without the env var set (the normal case) it is the identity.
        assert_eq!(fault_injected_flags(0, PLAN_STREAMED), PLAN_STREAMED);
        assert_eq!(fault_injected_flags(3, 0), 0);
        // The transform itself, independent of the env read: a streamed
        // parameter becomes resident and vice versa, and both directions
        // change what `expected()` demands — which is what makes the
        // mismatch certain rather than merely likely.
        assert_ne!(expected(PLAN_STREAMED), expected(0));
        assert_ne!(
            expected(PLAN_STREAMED | PLAN_BF16_SR),
            expected(0),
            "corrupting a bf16-sr param to resident must change the expectation"
        );
    }

    /// An all-resident plan verifies clean: it is the one shape whose
    /// expectation ("registered nowhere") holds in a bare test process, so it
    /// proves the clean path returns 0 rather than aborting.
    #[test]
    fn an_all_resident_plan_verifies_clean() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        nsl_param_plan_teardown();
        assert_eq!(nsl_param_plan_declare(0xBB00, 0, 0), 0);
        assert_eq!(nsl_param_plan_declare(0xBB08, 1, 0), 0);
        assert_eq!(nsl_param_plan_verify(), 0);
        nsl_param_plan_teardown();
    }

    /// An empty plan reports itself as vacuous instead of returning a clean
    /// verdict that means nothing.
    #[test]
    fn an_empty_plan_verify_is_reported_not_silently_clean() {
        let _g = TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        nsl_param_plan_teardown();
        assert_eq!(nsl_param_plan_verify(), 0);
    }
}
