//! Meta-flag expansion (roadmap 3.3).
//!
//! `--pretrain-optimized` bundles the pretraining stack: `--source-ad`
//! (mandatory — WGGO/CSHA/FASE planning all live in the source-AD branch of
//! train-block lowering, and tape AD leaves GPU params frozen), `--wggo greedy`
//! (the ~500ms planner, within a few percent of `full` per its docs — `full`
//! remains an explicit opt-in), `--csha auto`, and both backward fusions
//! (`--fuse-rmsnorm-backward`, `--fuse-wgrad-accum`).
//!
//! The two fusions change the ARITHMETIC, not just the schedule, which is why
//! they are opt-in everywhere else. They belong here because this bundle IS the
//! "I want pretraining throughput" request: measured +14% tok/s on Coder-50M at
//! a max relative loss divergence of 1.2e-5 over 24 micro-batches, identical
//! trend. `--training-reference` strips both back out. Because the bundle sets
//! them AFTER clap has finished validating, clap can no longer enforce
//! `--fuse-wgrad-accum`'s conflicts — [`WgradFusionBlockers`] is what replaces
//! that enforcement, with a matching hard error in `stmt.rs` as the backstop.
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

/// Parse `--checkpoint-stride` (`"auto"` or a positive integer) into a
/// [`nsl_codegen::CheckpointStride`]. Clap guarantees the flag's presence and
/// `requires = "checkpoint_blocks"`; this is the value-domain check. Invalid or
/// zero values fall back to `Fixed(1)` (per-block) with a stderr note rather
/// than aborting the build.
pub(crate) fn parse_checkpoint_stride(s: &str) -> nsl_codegen::CheckpointStride {
    use nsl_codegen::CheckpointStride;
    let t = s.trim();
    if t.eq_ignore_ascii_case("auto") {
        return CheckpointStride::Auto;
    }
    if t.eq_ignore_ascii_case("dp") {
        return CheckpointStride::Dp;
    }
    match t.parse::<usize>() {
        Ok(n) if n >= 1 => CheckpointStride::Fixed(n),
        _ => {
            eprintln!(
                "note: --checkpoint-stride '{s}' is not 'auto', 'dp' or a \
                 positive integer; using stride 1 (per-block checkpointing)"
            );
            CheckpointStride::Fixed(1)
        }
    }
}

#[cfg(test)]
mod stride_tests {
    use super::parse_checkpoint_stride;
    use nsl_codegen::CheckpointStride;

    #[test]
    fn parses_auto_and_integers_and_falls_back() {
        assert_eq!(parse_checkpoint_stride("auto"), CheckpointStride::Auto);
        assert_eq!(parse_checkpoint_stride("dp"), CheckpointStride::Dp);
        assert_eq!(parse_checkpoint_stride("DP"), CheckpointStride::Dp);
        assert_eq!(parse_checkpoint_stride("AUTO"), CheckpointStride::Auto);
        assert_eq!(parse_checkpoint_stride("1"), CheckpointStride::Fixed(1));
        assert_eq!(parse_checkpoint_stride("4"), CheckpointStride::Fixed(4));
        // 0 and garbage fall back to Fixed(1).
        assert_eq!(parse_checkpoint_stride("0"), CheckpointStride::Fixed(1));
        assert_eq!(parse_checkpoint_stride("nope"), CheckpointStride::Fixed(1));
    }
}

/// Everything `--fuse-wgrad-accum` refuses (`FEATURE_RULES`), as a single
/// predicate.
///
/// This list is LOAD-BEARING, and not for the reason it looks like. clap
/// conflicts `--fuse-wgrad-accum` with all three at parse time — but the bundle
/// enables the fusion from `expand_pretrain_optimized`, which runs inside
/// `dispatch()`, long after parsing. Setting a local `bool` here is invisible to
/// clap, so clap can no longer refuse these compositions: this list is the ONLY
/// thing standing between `--pretrain-optimized --grad-integrity` and a fused
/// chain that never materializes a gradient tensor.
///
/// What that would cost, if the list ever lapsed:
///   - `--grad-integrity`: the P0.3 report attests parameters whose gradients it
///     never observed — a silently-passing integrity gate, exactly the failure
///     that gate exists to catch.
///   - `--optim-state-offload`: `m_partial` is host-resident; the fused device
///     GEMM cannot write it.
///   - `--layerwise-accum`: pre-sliced tapes defeat the fusion's single-reader
///     proof.
///
/// `stmt.rs` carries a matching hard error (not a `debug_assert!`, which is a
/// no-op in this workspace's release profile) so a gap here fails loudly at
/// compile time rather than mis-training silently.
pub(crate) struct WgradFusionBlockers {
    pub grad_integrity: bool,
    pub optim_state_offload: bool,
    pub layerwise_accum: bool,
}

impl WgradFusionBlockers {
    fn first(&self) -> Option<&'static str> {
        if self.grad_integrity {
            Some("--grad-integrity")
        } else if self.optim_state_offload {
            Some("--optim-state-offload")
        } else if self.layerwise_accum {
            Some("--layerwise-accum")
        } else {
            None
        }
    }
}

pub(crate) fn expand_pretrain_optimized(
    pretrain_optimized: bool,
    wggo: &mut Option<String>,
    csha: &mut Option<String>,
    source_ad: &mut bool,
    fuse_rmsnorm_backward: &mut bool,
    fuse_wgrad_accum: &mut bool,
    blockers: &WgradFusionBlockers,
) {
    if !pretrain_optimized {
        return;
    }
    *source_ad = true;

    // The two backward fusions. Both are opt-in elsewhere because they change
    // the ARITHMETIC, not just the schedule — but "pretraining, optimized" is
    // exactly the context where that trade is intended, `--training-reference`
    // strips them back out for a reference run, and an explicit flag still
    // wins. Measured on Coder-50M (batch 4, seq 1024, RTX PRO 4500):
    // 32,925 -> 36,662 tok/s, max relative loss divergence 1.2e-5 over 24
    // steps with an identical trend.
    //
    // Neither is filled destructively: they are already `false` unless the user
    // asked for them, so setting them here cannot override an explicit choice
    // (there is no `--no-fuse-*` to lose).
    *fuse_rmsnorm_backward = true;
    match blockers.first() {
        None => *fuse_wgrad_accum = true,
        Some(flag) => eprintln!(
            "note: --pretrain-optimized bundle partially disabled: \
             --fuse-wgrad-accum not enabled because {flag} is set ({flag} needs \
             the raw gradient this fusion never materializes; the rest of the \
             bundle still applies)"
        ),
    }
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

/// P1.7 `--training-reference`: force the field-controlled optimizations OFF on
/// an already-built `CompileOptions`, so the emitted training path is the
/// simplest correct baseline. The remaining optimizations that are decorator- or
/// pattern-driven (FBIP, the fused FASE step, fused-CE substitution, @checkpoint
/// decorators) are gated in codegen on `opts.training_reference`.
///
/// Loud-override semantics (mirrors `--pretrain-optimized`): anything actually
/// turned off is listed once on stderr, so a user who also passed e.g.
/// `--checkpoint-blocks` sees exactly what the reference mode overrode rather
/// than silently getting a different path than they asked for.
pub(crate) fn apply_training_reference(opts: &mut nsl_codegen::CompileOptions) {
    if !opts.training_reference {
        return;
    }
    let mut disabled: Vec<&str> = Vec::new();
    macro_rules! off_bool {
        ($field:ident, $name:literal) => {
            if opts.$field {
                opts.$field = false;
                disabled.push($name);
            }
        };
    }
    off_bool!(checkpoint_blocks, "--checkpoint-blocks (CCR)");
    off_bool!(checkpoint_selective, "--checkpoint-selective (CCR)");
    off_bool!(layerwise_accum, "--layerwise-accum (CSLA)");
    off_bool!(weight_stream, "--weight-stream");
    off_bool!(stream_arena, "--stream-arena");
    off_bool!(stream_prefetch, "--stream-prefetch");
    off_bool!(stream_async_writeback, "--stream-async-writeback");
    off_bool!(optim_state_offload, "--optim-state-offload");
    // Both change the ARITHMETIC, not just the schedule — a reference run must
    // not silently keep a deliberately non-bit-exact fusion. (`--fuse-rmsnorm-backward`
    // was missing here too; same reason, same fix.)
    off_bool!(fuse_wgrad_accum, "--fuse-wgrad-accum (non-bit-exact)");
    off_bool!(fuse_rmsnorm_backward, "--fuse-rmsnorm-backward (non-bit-exact)");
    if opts.checkpoint_budget_mib.is_some() {
        opts.checkpoint_budget_mib = None;
        disabled.push("--checkpoint-budget-mib (CCR)");
    }
    if opts.checkpoint_stride != nsl_codegen::CheckpointStride::Fixed(1) {
        opts.checkpoint_stride = nsl_codegen::CheckpointStride::Fixed(1);
        disabled.push("--checkpoint-stride (periodic checkpointing)");
    }
    if opts.checkpoint_compress.is_some() {
        opts.checkpoint_compress = None;
        disabled.push("--checkpoint-compress (CCR)");
    }
    if !opts.disable_fusion {
        opts.disable_fusion = true;
        disabled.push("kernel @fuse fusion");
    }
    // WGGO + reduced-precision moments.
    if opts.wggo.mode.as_deref() != Some("off") {
        opts.wggo.mode = Some("off".to_string());
        disabled.push("WGGO transformations");
    }
    if opts.wggo.moment_precision {
        opts.wggo.moment_precision = false;
        disabled.push("WGGO reduced-precision moments");
    }
    if opts.wggo.memory_budget_bytes.is_some() {
        opts.wggo.memory_budget_bytes = None;
        disabled.push("--wggo-memory-budget");
    }
    // CPDT precision-adaptive path (independent moment-precision lowering).
    if opts.cpdt.mode != nsl_codegen::cpdt::CpdtMode::Off {
        opts.cpdt.mode = nsl_codegen::cpdt::CpdtMode::Off;
        disabled.push("CPDT precision-adaptive training");
    }
    // CSHA attention fusion. `csha.mode = "off"` is the load-bearing gate
    // (forces disabled_by_flag in codegen, covering @csha decorators too); the
    // decorator config maps are populated LATER in the pipeline, so clearing
    // them here would be a no-op — the mode gate is what actually disables them.
    if opts.csha.mode.as_deref() != Some("off") {
        opts.csha.mode = Some("off".to_string());
        disabled.push("CSHA attention fusion (mode + @csha)");
    }

    eprintln!(
        "note: --training-reference forces the simplest correct training path. \
         Disabled: {}. Also disabled in codegen: FBIP in-place, the fused FASE \
         optimizer step, and @fused_lm_ce / @fused_kl_ce / @checkpoint decorators.",
        if disabled.is_empty() {
            "(nothing was on)".to_string()
        } else {
            disabled.join(", ")
        }
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_reference_forces_field_opts_off() {
        let mut opts = nsl_codegen::CompileOptions {
            training_reference: true,
            checkpoint_blocks: true,
            layerwise_accum: true,
            weight_stream: true,
            stream_arena: true,
            stream_prefetch: true,
            stream_async_writeback: true,
            optim_state_offload: true,
            disable_fusion: false,
            ..Default::default()
        };
        opts.wggo.mode = Some("greedy".to_string());
        opts.wggo.moment_precision = true;
        apply_training_reference(&mut opts);
        assert!(!opts.checkpoint_blocks);
        assert!(!opts.layerwise_accum);
        assert!(!opts.weight_stream);
        assert!(!opts.stream_arena);
        assert!(!opts.stream_prefetch);
        assert!(!opts.stream_async_writeback);
        assert!(!opts.optim_state_offload);
        assert!(opts.disable_fusion, "kernel fusion disabled");
        assert_eq!(opts.wggo.mode.as_deref(), Some("off"));
        assert!(!opts.wggo.moment_precision);
        assert_eq!(opts.cpdt.mode, nsl_codegen::cpdt::CpdtMode::Off);
        assert_eq!(opts.csha.mode.as_deref(), Some("off"));
    }

    #[test]
    fn training_reference_noop_when_flag_absent() {
        let mut opts = nsl_codegen::CompileOptions {
            training_reference: false,
            checkpoint_blocks: true,
            ..Default::default()
        };
        apply_training_reference(&mut opts);
        assert!(opts.checkpoint_blocks, "no override without the flag");
    }

    /// No blockers: the bundle may fill both fusions.
    fn clear() -> WgradFusionBlockers {
        WgradFusionBlockers {
            grad_integrity: false,
            optim_state_offload: false,
            layerwise_accum: false,
        }
    }

    /// `(wggo, csha, source_ad, fuse_rmsnorm, fuse_wgrad)` after expansion.
    fn expand(
        on: bool,
        w: Option<&str>,
        c: Option<&str>,
        blockers: WgradFusionBlockers,
    ) -> (Option<String>, Option<String>, bool, bool, bool) {
        let (mut w, mut c) = (w.map(str::to_string), c.map(str::to_string));
        let (mut s, mut fr, mut fw) = (false, false, false);
        expand_pretrain_optimized(on, &mut w, &mut c, &mut s, &mut fr, &mut fw, &blockers);
        (w, c, s, fr, fw)
    }

    #[test]
    fn fills_unset_members() {
        let (w, c, s, fr, fw) = expand(true, None, None, clear());
        assert_eq!(w.as_deref(), Some("greedy"));
        assert_eq!(c.as_deref(), Some("auto"));
        assert!(s);
        assert!(fr && fw, "the bundle enables both backward fusions");
    }

    #[test]
    fn explicit_values_win() {
        let (w, c, s, _, _) = expand(true, Some("full"), Some("boundary"), clear());
        assert_eq!(w.as_deref(), Some("full"));
        assert_eq!(c.as_deref(), Some("boundary"));
        assert!(s, "source_ad is always forced on (planning needs it)");
    }

    #[test]
    fn explicit_off_respected() {
        let (w, c, s, _, _) = expand(true, Some("off"), Some("off"), clear());
        assert_eq!(w.as_deref(), Some("off"));
        assert_eq!(c.as_deref(), Some("off"));
        assert!(s);
    }

    #[test]
    fn noop_without_the_flag() {
        let (w, c, s, fr, fw) = expand(false, None, None, clear());
        assert!(w.is_none() && c.is_none() && !s);
        assert!(!fr && !fw, "no bundle, no fusions");
    }

    /// Each blocker suppresses ONLY `--fuse-wgrad-accum`. Filling it anyway
    /// would make clap reject an invocation whose conflicting flag the user
    /// never paired with a fusion flag themselves.
    #[test]
    fn wgrad_fusion_suppressed_by_each_blocker() {
        for (name, b) in [
            ("grad_integrity", WgradFusionBlockers { grad_integrity: true, ..clear() }),
            ("optim_state_offload", WgradFusionBlockers { optim_state_offload: true, ..clear() }),
            ("layerwise_accum", WgradFusionBlockers { layerwise_accum: true, ..clear() }),
        ] {
            let (_, _, s, fr, fw) = expand(true, None, None, b);
            assert!(s, "{name}: the rest of the bundle still applies");
            assert!(fr, "{name}: only the wgrad fusion is blocked");
            assert!(!fw, "{name}: must not enable a fusion clap will reject");
        }
    }

    /// `--training-reference` has to strip what the bundle adds, or a reference
    /// run silently keeps a non-bit-exact fusion it never asked for.
    #[test]
    fn training_reference_strips_the_bundle_fusions() {
        let (_, _, _, fr, fw) = expand(true, None, None, clear());
        assert!(fr && fw);
        let mut opts = nsl_codegen::CompileOptions {
            training_reference: true,
            fuse_rmsnorm_backward: fr,
            fuse_wgrad_accum: fw,
            ..Default::default()
        };
        apply_training_reference(&mut opts);
        assert!(
            !opts.fuse_rmsnorm_backward && !opts.fuse_wgrad_accum,
            "reference mode must disable both bundle-enabled fusions"
        );
    }
}
