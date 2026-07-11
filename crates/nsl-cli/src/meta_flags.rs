//! Meta-flag expansion (roadmap 3.3).
//!
//! `--pretrain-optimized` bundles the pretraining stack: `--source-ad`
//! (mandatory — WGGO/CSHA/FASE planning all live in the source-AD branch of
//! train-block lowering, and tape AD leaves GPU params frozen), `--wggo greedy`
//! (the ~500ms planner, within a few percent of `full` per its docs — `full`
//! remains an explicit opt-in), and `--csha auto`.
//!
//! Fill-None semantics: explicit user values always win — the bundle only
//! fills flags the user left unset, mirroring the `--cpdt-report`→`--cpdt`
//! precedent. When the user explicitly disables a member (`--wggo off` /
//! `--csha off`), the bundle respects it and says so once on stderr.
//!
//! One shared helper for `nsl build` and `nsl run` so the two dispatchers
//! cannot drift (the historical failure mode of these copy-pasted option
//! blocks). Expansion runs BEFORE mode-string validation, so filled values go
//! through the exact same validation as hand-written flags.
//!
//! Deliberately NOT bundled (documented refusals, not silent gaps):
//! `@pca(strategy=...)` and packed `dataset` blocks are authorial source
//! constructs a flag cannot synthesize — WGGO packing decisions surface via
//! the `[pca] wggo-override-*` verdicts instead; CPDT needs a cluster spec
//! (`--cpdt-num-gpus`) and stays opt-in.

pub(crate) fn expand_pretrain_optimized(
    pretrain_optimized: bool,
    wggo: &mut Option<String>,
    csha: &mut Option<String>,
    source_ad: &mut bool,
) {
    if !pretrain_optimized {
        return;
    }
    *source_ad = true;
    match wggo.as_deref() {
        None => *wggo = Some("greedy".to_string()),
        Some("off") => eprintln!(
            "note: --pretrain-optimized bundle partially disabled: --wggo off \
             (explicit flag wins; no WGGO plan will drive CSHA/FASE/PCA)"
        ),
        Some(_) => {} // explicit mode wins silently
    }
    match csha.as_deref() {
        None => *csha = Some("auto".to_string()),
        Some("off") => eprintln!(
            "note: --pretrain-optimized bundle partially disabled: --csha off \
             (explicit flag wins)"
        ),
        Some(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_unset_members() {
        let (mut w, mut c, mut s) = (None, None, false);
        expand_pretrain_optimized(true, &mut w, &mut c, &mut s);
        assert_eq!(w.as_deref(), Some("greedy"));
        assert_eq!(c.as_deref(), Some("auto"));
        assert!(s);
    }

    #[test]
    fn explicit_values_win() {
        let (mut w, mut c, mut s) =
            (Some("full".to_string()), Some("boundary".to_string()), false);
        expand_pretrain_optimized(true, &mut w, &mut c, &mut s);
        assert_eq!(w.as_deref(), Some("full"));
        assert_eq!(c.as_deref(), Some("boundary"));
        assert!(s, "source_ad is always forced on (planning needs it)");
    }

    #[test]
    fn explicit_off_respected() {
        let (mut w, mut c, mut s) =
            (Some("off".to_string()), Some("off".to_string()), false);
        expand_pretrain_optimized(true, &mut w, &mut c, &mut s);
        assert_eq!(w.as_deref(), Some("off"));
        assert_eq!(c.as_deref(), Some("off"));
        assert!(s);
    }

    #[test]
    fn noop_without_the_flag() {
        let (mut w, mut c, mut s) = (None, None, false);
        expand_pretrain_optimized(false, &mut w, &mut c, &mut s);
        assert!(w.is_none() && c.is_none() && !s);
    }
}
