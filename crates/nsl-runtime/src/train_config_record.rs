//! The resolved train/optimizer/scheduler configuration as checkpoint
//! identity (next-roadmap item 4).
//!
//! Item 8 made the `.optim` sidecar refuse corpus/geometry/seed drift and
//! PR #519 made it refuse compile-flag arithmetic drift — but the resolved
//! CONFIG (lr, betas, weight decay, grad_accumulation, the schedule, the
//! clip) was recorded nowhere: editing any of them between save and resume
//! silently produced a run that is not a continuation of the checkpoint.
//!
//! Same shape as the execution fingerprint: codegen renders one fixed-order
//! `k=v,k=v` record from the RESOLVED config (the values are compile-time
//! constants — the runtime cannot recover them, so codegen installs the
//! record at train-block entry), the sidecar carries it verbatim inside an
//! allowlisted string field, and resume diffs saved-vs-live per key. Floats
//! ride INSIDE the string because the sidecar's needle scanners are
//! digit-only — a bare `"lr":0.0001` header number would parse as 0.
//!
//! Two severity classes, split by WHAT the drift corrupts (mirroring
//! #519's arithmetic/placement doctrine):
//!
//!  * **moment-meaning** (`opt, accum, beta1, beta2, eps, wd, momentum,
//!    dampening, nesterov, ns_steps, adamw_lr, no_decay`): these change
//!    what the restored m/v moments and the restored step counter MEAN —
//!    `accum` is the optimizer-step divisor and the bias-correction clock.
//!    A resume under different values is not a continuation and nothing
//!    downstream can tell. ABORT, no escape; `model_load(...)` is the
//!    documented weights-only warm start.
//!  * **trajectory** (`lr, sched, sp1, sp2, sp3, clip`): these change the
//!    FUTURE trajectory, not the meaning of restored state — and the
//!    repo's own history is full of deliberate mid-resume lr experiments.
//!    ABORT by default naming `NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1`, which
//!    converts the refusal into a loud acknowledgment (the
//!    `NSL_TAPE_ALLOW_DISCONNECTED` prior art). Note the acknowledgment
//!    prints the schedule-reinterpretation hazard: the schedule is a pure
//!    function of (base_lr, restored step, constants), so changing
//!    warmup/total re-reads the restored step counter under new rules.

use std::sync::Mutex;

use crate::exec_fingerprint::{diff, render, FieldDiff};

static TRAIN_CONFIG: Mutex<String> = Mutex::new(String::new());

/// Keys whose drift changes the meaning of restored optimizer state or the
/// restored step counter. Explicit list, not the negation of the other —
/// a key in neither class is deliberately silent (the #519 doctrine).
pub const MOMENT_KEYS: &[&str] = &[
    "opt", "accum", "beta1", "beta2", "eps", "wd", "momentum", "dampening",
    "nesterov", "ns_steps", "adamw_lr", "no_decay",
];

/// Keys whose drift changes the future trajectory only. sp4..sp6 are
/// reserved ahead of any 4+-parameter scheduler: a parameter rendered
/// under a key in neither class would be silently unguarded (review
/// finding — absent-on-both-sides is not a difference by design).
pub const TRAJECTORY_KEYS: &[&str] =
    &["lr", "sched", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "clip"];

/// Install the record for the CURRENT train block. Codegen emits this at
/// train-block entry — per block, not per program: a module can hold more
/// than one `train(...)`, and blocks execute sequentially, so the global is
/// correct whenever this block's save/load runs. Null/invalid input leaves
/// the previous value (checked and reported downstream, never guessed).
#[no_mangle]
pub extern "C" fn nsl_set_train_config_record(ptr: i64, len: i64) -> i64 {
    if ptr == 0 || len <= 0 {
        return 0;
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len as usize) };
    match std::str::from_utf8(bytes) {
        Ok(s) => {
            *TRAIN_CONFIG.lock().unwrap() = s.to_string();
            0
        }
        Err(_) => 0,
    }
}

/// The installed record; empty when the build predates item 4.
pub fn train_config_record() -> String {
    TRAIN_CONFIG.lock().unwrap().clone()
}

pub fn moment_diff(saved: &str, live: &str) -> Vec<FieldDiff> {
    diff(saved, live, MOMENT_KEYS)
}

pub fn trajectory_diff(saved: &str, live: &str) -> Vec<FieldDiff> {
    diff(saved, live, TRAJECTORY_KEYS)
}

pub use crate::exec_fingerprint::render as render_diffs;

/// The resume-side policy. Called from `nsl_train_checkpoint_load` inside
/// the validated-but-nothing-mutated window. Aborts on refusal.
pub fn check_on_resume(saved: &str) {
    let live = train_config_record();
    // Empty on either side => the check cannot run; say so loudly rather
    // than silently passing (the #519 vacuity lesson).
    if saved.is_empty() || live.is_empty() {
        let side = if saved.is_empty() { "checkpoint" } else { "this build" };
        eprintln!(
            "nsl: train_checkpoint_load: the {side} carries no train-config \
             record, so the lr/optimizer/schedule check is SKIPPED. Verify \
             the training configuration matches the saved run by hand."
        );
        return;
    }
    let moment = moment_diff(saved, &live);
    if !moment.is_empty() {
        eprintln!(
            "nsl: train_checkpoint_load: this run's OPTIMIZER CONFIGURATION \
             differs from the checkpoint's:\n{}\n\
             These change what the restored optimizer moments and step \
             counter MEAN — the resume would not be a continuation of the \
             saved run, and nothing downstream could tell. There is no \
             override for this class; use `model_load(...)` for a \
             weights-only warm start under a new configuration.",
            render(&moment)
        );
        std::process::abort();
    }
    let traj = trajectory_diff(saved, &live);
    if traj.is_empty() {
        return;
    }
    let allowed = std::env::var("NSL_RESUME_ALLOW_TRAJECTORY_DRIFT")
        .ok()
        .as_deref()
        == Some("1");
    if allowed {
        eprintln!(
            "nsl: train_checkpoint_load: TRAJECTORY drift acknowledged \
             (NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1):\n{}\n\
             The resume continues under the NEW values. Note: the schedule \
             is recomputed from (base_lr, restored step, schedule \
             constants) every step, so a changed warmup/total re-interprets \
             the restored step counter under the new schedule.",
            render(&traj)
        );
        return;
    }
    eprintln!(
        "nsl: train_checkpoint_load: this run's LR/SCHEDULE/CLIP differs \
         from the checkpoint's:\n{}\n\
         If this change is intentional (an lr drop, a schedule extension), \
         re-run with NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1 to resume under \
         the new values with an acknowledgment. Otherwise restore the saved \
         configuration — the resume contract is 're-run the file UNCHANGED \
         with checkpoint_load= added'.",
        render(&traj)
    );
    std::process::abort();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact class contents are the policy — the review proved that
    /// deleting most keys survived every behavioral gate (absent-on-both
    /// is no-diff by design), so the arrays themselves are pinned here.
    #[test]
    fn class_membership_is_pinned_exactly() {
        assert_eq!(
            MOMENT_KEYS,
            &["opt", "accum", "beta1", "beta2", "eps", "wd", "momentum",
              "dampening", "nesterov", "ns_steps", "adamw_lr", "no_decay"],
        );
        assert_eq!(
            TRAJECTORY_KEYS,
            &["lr", "sched", "sp1", "sp2", "sp3", "sp4", "sp5", "sp6", "clip"],
        );
    }

    #[test]
    fn classes_are_disjoint_and_diff_routes_by_class() {
        for k in MOMENT_KEYS {
            assert!(!TRAJECTORY_KEYS.contains(k), "{k} in both classes");
        }
        let saved = "opt=adamw,lr=0.0001,accum=4,beta2=0.95,clip=1,sched=warmup_cosine,sp1=200,sp2=2048,sp3=0.00003";
        let live_lr = "opt=adamw,lr=0.0002,accum=4,beta2=0.95,clip=1,sched=warmup_cosine,sp1=200,sp2=2048,sp3=0.00003";
        assert!(moment_diff(saved, live_lr).is_empty());
        let t = trajectory_diff(saved, live_lr);
        assert_eq!(t.len(), 1);
        assert_eq!(t[0].key, "lr");

        let live_accum = saved.replace("accum=4", "accum=2");
        let m = moment_diff(saved, &live_accum);
        assert_eq!(m.len(), 1);
        assert_eq!(m[0].key, "accum");
        assert!(trajectory_diff(saved, &live_accum).is_empty());
    }

    #[test]
    fn absent_on_both_sides_is_not_a_difference() {
        // Neither record carries sp2/sp3 (a paramless scheduler): no diff.
        let a = "opt=sgd,lr=0.05,accum=2,sched=none,clip=none";
        let b = "opt=sgd,lr=0.05,accum=2,sched=none,clip=none";
        assert!(moment_diff(a, b).is_empty());
        assert!(trajectory_diff(a, b).is_empty());
    }
}
