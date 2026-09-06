//! The runtime-read behavior-tier environment as checkpoint identity
//! (roadmap A5, increment 3).
//!
//! Item 8 guarded the corpus, geometry and seed; #519 guarded the compile
//! flags (`exec_fingerprint`); item 4 guarded the resolved train config
//! (`train_config_record`). One input to a training step was still
//! recorded nowhere: the `NSL_*` variables the RUNTIME reads while it runs.
//! `NSL_FLASH_BWD_CPU=1`, `NSL_SUM_SQ_CPU=1`, `NSL_MATMUL_TF32=0`,
//! `NSL_GPU_GRAD_CLIP=0` each route a kernel to a different implementation
//! with a different reduction order, and each is a word in the launching
//! shell, not in the recipe. A resume under a different set of them
//! continued theta and the moments under different arithmetic and said
//! nothing — the same gap #519 closed for the command line, one layer out.
//!
//! WHICH VARIABLES. The registry in `nsl-env` is the policy, not a second
//! list here: every variable whose tier is [`nsl_env::Tier::Behavior`]
//! ("changes what a run computes") and whose `read_at` is
//! [`nsl_env::ReadAt::Runtime`]. Variables read at BOTH compile and run
//! time (the `NSL_MATMUL_BF16*` family) are excluded on purpose: #583
//! resolved them into `CompileOptions` at compile time, the exec fingerprint
//! already carries the resolved values, and the runtime read is only the
//! embedder fallback — recording the raw environment a second time would
//! refuse a resume whose PROGRAM carries the setting but whose shell no
//! longer does. Compile-time-only behavior reads are the exec fingerprint's
//! job (see `CompileOptions::exec_fingerprint`, keys `fa_mma`, `lce_gemm`,
//! `fase_sumsq`, `fase_override`, `csha_save`).
//!
//! WHAT IS RECORDED. `NAME=value` for every such variable that is SET, in
//! registry (name) order, joined by `,`; an unset variable is absent from
//! the record. So the record for the common case — nothing exported — is
//! the empty string, and "empty" here means "nothing set", NOT "predates
//! the feature": the sidecar distinguishes the two by whether the `env` key
//! exists in the header at all ([`check_on_resume`] takes an `Option`).
//! (Under the CLI the record is rarely literally empty: `nsl run` exports
//! `NSL_COLLECTIVES` to the program it launches, so `NSL_COLLECTIVES=sim`
//! is the usual single-GPU record — and a resume that switches the
//! collective backend is, correctly, a difference.)
//!
//! WHAT DRIFT MEANS. Every variable in the set is arithmetic by the tier's
//! definition, so any difference — set on one side only, or set to
//! different values — refuses. The values are compared as the raw strings
//! the shell exported: `NSL_SUM_SQ_CPU=1` on one side and unset on the
//! other is a real difference; `=1` versus `=true` is reported as one even
//! where a read site would accept both, because this module does not
//! re-implement 19 parsers, and a spurious refusal names exactly what to
//! fix. `NSL_RESUME_ALLOW_ENV_DRIFT=1` converts the refusal into a loud
//! acknowledgment, mirroring `NSL_RESUME_ALLOW_TRAJECTORY_DRIFT` — an
//! operator who moved a run from a machine where `NSL_FLASH_BWD_CPU=1` was
//! the workaround to one where it is not should be able to say so.

use crate::exec_fingerprint::{diff, render, FieldDiff};

/// The names this module records, in registry order.
pub fn guarded_names() -> Vec<&'static str> {
    nsl_env::by_tier(nsl_env::Tier::Behavior)
        .filter(|v| v.read_at == nsl_env::ReadAt::Runtime)
        .map(|v| v.name)
        .collect()
}

/// Render an environment as the record: `NAME=value,...` for the guarded
/// names that `lookup` resolves, name order. `lookup` is a parameter so the
/// tests can hand in an environment without mutating the process's.
pub fn render_record(lookup: impl Fn(&str) -> Option<String>) -> String {
    guarded_names()
        .into_iter()
        .filter_map(|name| lookup(name).map(|v| format!("{name}={}", sanitize(&v))))
        .collect::<Vec<_>>()
        .join(",")
}

/// The live process's record.
pub fn env_record() -> String {
    // The variable NAMES come from the registry (`nsl_env::REGISTRY`), so
    // this is the one read in the workspace whose name is not a literal;
    // `nsl-env`'s agreement gate pins it in `KNOWN_DYNAMIC_READS`.
    render_record(|name| std::env::var(name).ok())
}

/// A value rides inside a `k=v,k=v` record that itself rides inside a JSON
/// string in the sidecar header, so the two record separators and the two
/// JSON-string terminators must not survive. Same allowlist as the exec
/// fingerprint's `dtype`: everything outside `[A-Za-z0-9._-]` becomes `_`.
/// The recorded value is for COMPARISON, never fed back to a read site, so
/// a mangled hostile value costs nothing but a truthful refusal.
fn sanitize(v: &str) -> String {
    v.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Differences between two records over the guarded names — absent on one
/// side is a difference (set versus unset), absent on both is not.
pub fn env_diff(saved: &str, live: &str) -> Vec<FieldDiff> {
    diff(saved, live, &guarded_names())
}

/// What the loader does with a checkpoint's record.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The sidecar has no `env` key: written before this record existed.
    /// The check cannot run; say so.
    Predates,
    /// Identical (both usually empty).
    Same,
    /// Differences, and `NSL_RESUME_ALLOW_ENV_DRIFT=1` was set.
    Acknowledged(Vec<FieldDiff>),
    /// Differences, no acknowledgment: refuse.
    Refuse(Vec<FieldDiff>),
}

/// The policy, pure: `saved` is the sidecar's record (`None` = no `env`
/// key in the header), `live` the running process's, `allow` whether the
/// acknowledgment variable is set to `1`.
pub fn verdict(saved: Option<&str>, live: &str, allow: bool) -> Verdict {
    let Some(saved) = saved else {
        return Verdict::Predates;
    };
    let d = env_diff(saved, live);
    if d.is_empty() {
        Verdict::Same
    } else if allow {
        Verdict::Acknowledged(d)
    } else {
        Verdict::Refuse(d)
    }
}

/// The resume-side check. Called from `nsl_train_checkpoint_load` inside
/// the validated-but-nothing-mutated window, after the compile-flag and
/// train-config checks. Aborts on refusal.
pub fn check_on_resume(saved: Option<&str>) {
    let live = env_record();
    let allow = std::env::var("NSL_RESUME_ALLOW_ENV_DRIFT").ok().as_deref() == Some("1");
    match verdict(saved, &live, allow) {
        Verdict::Predates => {
            eprintln!(
                "nsl: train_checkpoint_load: the checkpoint carries no \
                 environment record (written before one existed), so the \
                 runtime NSL_* behavior check is SKIPPED for this resume. \
                 Re-save from a current build to restore it."
            );
        }
        Verdict::Same => {}
        Verdict::Acknowledged(d) => {
            eprintln!(
                "nsl: train_checkpoint_load: runtime ENVIRONMENT drift \
                 acknowledged (NSL_RESUME_ALLOW_ENV_DRIFT=1):\n{}\n\
                 The resume continues under this run's values. Each of these \
                 routes a kernel to a different implementation, so the \
                 continuation is not bit-comparable with the saved run.",
                render(&d)
            );
        }
        Verdict::Refuse(d) => {
            eprintln!(
                "nsl: train_checkpoint_load: this run's ENVIRONMENT differs \
                 from the one the checkpoint was saved under:\n{}\n  \
                 These are behavior-tier variables (`nsl env list --tier \
                 behavior`): each changes what a training step computes, so \
                 resuming under different ones is not a continuation of the \
                 saved run. Export the saved values (unset = absent above), \
                 or re-run with NSL_RESUME_ALLOW_ENV_DRIFT=1 to resume under \
                 the new ones with an acknowledgment.",
                render(&d)
            );
            std::process::abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let m: HashMap<String, String> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        move |name| m.get(name).cloned()
    }

    #[test]
    fn the_guarded_set_is_the_registry_tier_not_a_local_list() {
        let names = guarded_names();
        // Anti-vacuity: the tier is populated, and it is exactly the
        // runtime-read half — the compile-resolved matmul family is out.
        assert!(names.len() >= 10, "{names:?}");
        for n in ["NSL_FLASH_BWD_CPU", "NSL_SUM_SQ_CPU", "NSL_GPU_GRAD_CLIP", "NSL_MATMUL_TF32"] {
            assert!(names.contains(&n), "{n} missing from {names:?}");
        }
        for n in ["NSL_MATMUL_BF16", "NSL_MATMUL_BF16_ROUND"] {
            assert!(!names.contains(&n), "{n} is compile-resolved (#583) and must not be re-guarded");
        }
        // Diagnostic-tier knobs never enter the record.
        assert!(!names.contains(&"NSL_MEMSTATS"));
        let mut sorted = names.clone();
        sorted.sort_unstable();
        assert_eq!(names, sorted, "record order must be name order");
    }

    #[test]
    fn only_set_guarded_variables_are_recorded() {
        let rec = render_record(env(&[
            ("NSL_SUM_SQ_CPU", "1"),
            ("NSL_MEMSTATS", "1"),   // diagnostic tier: ignored
            ("NSL_MATMUL_BF16", "1"), // compile-resolved: ignored
            ("HOME", "/x"),
        ]));
        assert_eq!(rec, "NSL_SUM_SQ_CPU=1");
        assert_eq!(render_record(env(&[])), "");
    }

    #[test]
    fn a_hostile_value_cannot_forge_a_field_or_break_the_sidecar_json() {
        let rec = render_record(env(&[("NSL_SUM_SQ_CPU", r#"1","evil":"x,NSL_GPU_GRAD_CLIP=0\"#)]));
        assert_eq!(rec.split(',').count(), 1, "{rec}");
        assert!(!rec.contains('"') && !rec.contains('\\'), "{rec}");
        assert_eq!(rec.matches('=').count(), 1, "{rec}");
        assert!(env_diff(&rec, "NSL_GPU_GRAD_CLIP=0").iter().any(|d| d.key == "NSL_GPU_GRAD_CLIP"));
    }

    #[test]
    fn set_versus_unset_is_a_difference_and_unset_on_both_is_not() {
        let d = env_diff("", "NSL_FLASH_BWD_CPU=1");
        assert_eq!(d.len(), 1);
        assert_eq!(d[0].key, "NSL_FLASH_BWD_CPU");
        assert_eq!(d[0].saved, "<absent>");
        assert_eq!(d[0].live, "1");
        assert!(env_diff("", "").is_empty());
        assert!(env_diff("NSL_SUM_SQ_CPU=1", "NSL_SUM_SQ_CPU=1").is_empty());
        let d = env_diff("NSL_SUM_SQ_BLOCKS=256", "NSL_SUM_SQ_BLOCKS=128");
        assert_eq!(d.len(), 1);
        assert_eq!((d[0].saved.as_str(), d[0].live.as_str()), ("256", "128"));
    }

    #[test]
    fn the_verdict_routes_predates_same_acknowledged_refuse() {
        assert_eq!(verdict(None, "", false), Verdict::Predates);
        assert_eq!(verdict(None, "NSL_SUM_SQ_CPU=1", true), Verdict::Predates);
        assert_eq!(verdict(Some(""), "", false), Verdict::Same);
        match verdict(Some(""), "NSL_SUM_SQ_CPU=1", false) {
            Verdict::Refuse(d) => assert_eq!(d[0].key, "NSL_SUM_SQ_CPU"),
            other => panic!("expected Refuse, got {other:?}"),
        }
        match verdict(Some("NSL_SUM_SQ_CPU=1"), "", true) {
            Verdict::Acknowledged(d) => assert_eq!(d[0].live, "<absent>"),
            other => panic!("expected Acknowledged, got {other:?}"),
        }
        // A key that is NOT guarded never produces a verdict, even if it
        // somehow reached a record (a registry retier after the save).
        assert_eq!(verdict(Some("NSL_MEMSTATS=1"), "", false), Verdict::Same);
    }
}
