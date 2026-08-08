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
//! trend.
//!
//! That +14% is NOT "both fusions" (corrected 2026-08-07). `--fuse-wgrad-accum`
//! fires only through the FASE-*Deferred* `on_param_grad` hook, which needs
//! `grad_accumulation >= 2` in the train block; the measured run declared none,
//! so the wgrad half fused zero chains and the number belongs to
//! `--fuse-rmsnorm-backward` and the rest of the bundle. Codegen now emits
//! `[wgrad-fusion] declined: ...` per train block rather than leaving that
//! invisible. `--training-reference` strips both back out. Because the bundle sets
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

/// The flags `--pretrain-optimized` expands into.
///
/// A struct rather than six `&mut` out-parameters: the sixth (item 4's
/// `--fuse-lm-head`) tipped the signature past clippy's `too_many_arguments`,
/// and six same-shaped `&mut`s in a row is exactly the argument list a caller
/// transposes. Both dispatchers build one of these from their locals and
/// destructure it back, so the shared-helper property that keeps `nsl run` and
/// `nsl build` from drifting is unchanged.
pub(crate) struct PretrainBundle {
    pub(crate) wggo: Option<String>,
    pub(crate) csha: Option<String>,
    pub(crate) source_ad: bool,
    pub(crate) fuse_rmsnorm_backward: bool,
    pub(crate) fuse_wgrad_accum: bool,
    /// OUT-parameter, always built as `false`: set here when the bundle — not
    /// the user — turned `fuse_wgrad_accum` on. Codegen refuses a flag the
    /// user typed that cannot fire, and only warns for one the bundle filled
    /// in; see `CompileOptions::fuse_wgrad_accum_from_bundle`.
    pub(crate) fuse_wgrad_accum_from_bundle: bool,
    pub(crate) fuse_lm_head: Option<String>,
}

pub(crate) fn expand_pretrain_optimized(
    pretrain_optimized: bool,
    b: &mut PretrainBundle,
    blockers: &WgradFusionBlockers,
) {
    if !pretrain_optimized {
        return;
    }
    let PretrainBundle {
        wggo,
        csha,
        source_ad,
        fuse_rmsnorm_backward,
        fuse_wgrad_accum,
        fuse_wgrad_accum_from_bundle,
        fuse_lm_head,
    } = b;
    *source_ad = true;

    // Item 4: the fused LM head belongs in the bundle for the same reason the
    // two backward fusions do — it is a measured win (~8.5% step time on
    // Coder-50M) that until now required hand-editing four numbers into the
    // model source. `auto`, not `require`: the bundle must not turn a program
    // whose head the compiler cannot prove into a build failure.
    //
    // Same loud-override shape as --wggo/--csha below, and the same reason:
    // an explicit `off` is a decision, and silently overriding it would make
    // the bundle a worse citizen than the flags it expands to.
    match fuse_lm_head.as_deref() {
        None => *fuse_lm_head = Some("auto".to_string()),
        Some("off") => eprintln!(
            "note: --pretrain-optimized bundle partially disabled: \
             --fuse-lm-head off (explicit flag wins; the [batch*seq, vocab] \
             logits surface will be materialized unless a @fused_lm_ce \
             decorator is present)"
        ),
        Some(_) => {}
    }

    // The two backward fusions. Both are opt-in elsewhere because they change
    // the ARITHMETIC, not just the schedule — but "pretraining, optimized" is
    // exactly the context where that trade is intended, `--training-reference`
    // strips them back out for a reference run, and an explicit flag still
    // wins.
    //
    // ATTRIBUTION, corrected 2026-08-07. The measurement is real: Coder-50M
    // (batch 4, seq 1024, RTX PRO 4500), 32,925 -> 36,662 tok/s, max relative
    // loss divergence 1.2e-5 over 24 steps with an identical trend. It was
    // credited to "both backward fusions". It was not: that run — like both
    // shipped pretrain scripts and `models/benchmarks/matrix_bench.py`'s
    // `optimized` arm — declares no `grad_accumulation`, so FASE is
    // Passthrough, the `on_param_grad` hook never exists, and
    // `--fuse-wgrad-accum` fused ZERO chains. The number belongs to
    // `--fuse-rmsnorm-backward` (plus whatever else the bundle turns on); the
    // wgrad half contributed nothing to it and cannot until a train block
    // declares `grad_accumulation >= 2`. Codegen now says so out loud per
    // train block (`[wgrad-fusion] declined:` in stmt.rs) rather than leaving
    // this comment as the only record.
    //
    // Neither is filled destructively: they are already `false` unless the user
    // asked for them, so setting them here cannot override an explicit choice
    // (there is no `--no-fuse-*` to lose).
    *fuse_rmsnorm_backward = true;
    match blockers.first() {
        None => {
            // Read the clap value BEFORE overwriting it: `--pretrain-optimized
            // --fuse-wgrad-accum` is an explicit request and must keep the
            // hard refusal when the fusion cannot fire, while a bundle-filled
            // flag only warns (it lands on programs that never asked).
            *fuse_wgrad_accum_from_bundle = !*fuse_wgrad_accum;
            *fuse_wgrad_accum = true;
        }
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
    // Item 4: the reference arm is what the fused head's numerics are
    // validated AGAINST, so it must not contain an inferred fused head — the
    // baseline would be a baseline for itself. Codegen refuses independently
    // (`lm_head_inference_for_train_block`); this arm exists so the override
    // is announced like every other one, rather than being a silent
    // difference between what was asked for and what ran.
    if opts.lm_head_fusion.is_on() {
        opts.lm_head_fusion = nsl_codegen::lm_head_inference::LmHeadFusion::Off;
        disabled.push("--fuse-lm-head (compiler-inferred fused LM head)");
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
        let (w, c, s, fr, fw, _) = expand_all(on, w, c, None, blockers);
        (w, c, s, fr, fw)
    }

    /// Same, plus the item-4 `--fuse-lm-head` slot.
    #[allow(clippy::type_complexity)]
    fn expand_all(
        on: bool,
        w: Option<&str>,
        c: Option<&str>,
        lm: Option<&str>,
        blockers: WgradFusionBlockers,
    ) -> (Option<String>, Option<String>, bool, bool, bool, Option<String>) {
        let mut b = PretrainBundle {
            wggo: w.map(str::to_string),
            csha: c.map(str::to_string),
            source_ad: false,
            fuse_rmsnorm_backward: false,
            fuse_wgrad_accum: false,
            fuse_wgrad_accum_from_bundle: false,
            fuse_lm_head: lm.map(str::to_string),
        };
        expand_pretrain_optimized(on, &mut b, &blockers);
        (
            b.wggo,
            b.csha,
            b.source_ad,
            b.fuse_rmsnorm_backward,
            b.fuse_wgrad_accum,
            b.fuse_lm_head,
        )
    }

    /// `(fuse_wgrad_accum, fuse_wgrad_accum_from_bundle)` for a given starting
    /// value of the clap flag. The provenance bit is what decides whether an
    /// inert fusion refuses (typed) or warns (bundle-filled), so both
    /// directions are pinned here rather than only end-to-end.
    fn wgrad_provenance(on: bool, explicit: bool, blockers: WgradFusionBlockers) -> (bool, bool) {
        let mut b = PretrainBundle {
            wggo: None,
            csha: None,
            source_ad: false,
            fuse_rmsnorm_backward: false,
            fuse_wgrad_accum: explicit,
            fuse_wgrad_accum_from_bundle: false,
            fuse_lm_head: None,
        };
        expand_pretrain_optimized(on, &mut b, &blockers);
        (b.fuse_wgrad_accum, b.fuse_wgrad_accum_from_bundle)
    }

    #[test]
    fn the_bundle_marks_its_own_wgrad_provenance() {
        // Bundle filled it in: the program never asked, so an inert fusion is
        // a warning.
        assert_eq!(wgrad_provenance(true, false, clear()), (true, true));
        // The user typed `--fuse-wgrad-accum` too: an inert fusion is a hard
        // refusal, so the bundle must NOT launder the provenance.
        assert_eq!(wgrad_provenance(true, true, clear()), (true, false));
        // No bundle at all.
        assert_eq!(wgrad_provenance(false, true, clear()), (true, false));
        assert_eq!(wgrad_provenance(false, false, clear()), (false, false));
        // A blocker suppresses the bundle's fill entirely — nothing is turned
        // on, so nothing can claim bundle provenance.
        assert_eq!(
            wgrad_provenance(
                true,
                false,
                WgradFusionBlockers {
                    grad_integrity: true,
                    optim_state_offload: false,
                    layerwise_accum: false,
                }
            ),
            (false, false)
        );
    }

    #[test]
    fn the_bundle_fills_fuse_lm_head_with_auto() {
        let (.., lm) = expand_all(true, None, None, None, clear());
        assert_eq!(lm.as_deref(), Some("auto"));
    }

    #[test]
    fn an_explicit_fuse_lm_head_wins_over_the_bundle() {
        // Both directions: `off` is a deliberate opt-out the bundle must not
        // overturn, and `require` must not be softened to `auto`.
        let (.., lm) = expand_all(true, None, None, Some("off"), clear());
        assert_eq!(lm.as_deref(), Some("off"));
        let (.., lm) = expand_all(true, None, None, Some("require"), clear());
        assert_eq!(lm.as_deref(), Some("require"));
    }

    #[test]
    fn without_the_bundle_fuse_lm_head_is_untouched() {
        let (.., lm) = expand_all(false, None, None, None, clear());
        assert_eq!(lm, None, "no bundle, no fill — the flag defaults to off");
    }

    #[test]
    fn training_reference_turns_an_inferred_head_off() {
        let mut opts = nsl_codegen::CompileOptions {
            training_reference: true,
            lm_head_fusion: nsl_codegen::lm_head_inference::LmHeadFusion::Require,
            ..Default::default()
        };
        apply_training_reference(&mut opts);
        assert!(
            !opts.lm_head_fusion.is_on(),
            "the reference arm must not contain the fusion it is the baseline for"
        );
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
