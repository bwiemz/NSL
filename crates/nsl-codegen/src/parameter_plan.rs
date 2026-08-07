//! Item 3: one **ParameterPlan** describing every trainable parameter's
//! storage, derived once and consumed everywhere.
//!
//! # What was here before
//!
//! Three questions were answered independently, at three points in
//! `stmt.rs`, from raw feature flags:
//!
//! * *Does this parameter stream?* — the CSLA schedule's `ws_streamed` set,
//!   computed from the layer grouping minus view-rooted params.
//! * *Is its authoritative storage bf16?* — `features.param_dtype_bf16sr`,
//!   read directly at each registration site.
//! * *Is it sharded?* — `(zero_stage == Some(3)).then(|| ws_streamed.clone())`,
//!   i.e. re-derived from the streaming set at the point of use.
//!
//! Each site then emitted its own sequence of `nsl_sr_bf16_note_param` /
//! `nsl_weight_stream_register` calls. The three answers agree today, but
//! only because two sites happen to spell the same conditions the same way;
//! nothing computes them together, and nothing checks that the emitted calls
//! and the intended plan describe the same thing.
//!
//! # What this is
//!
//! [`ParameterPlan::derive`] answers all three questions at once, for every
//! parameter, and returns a value that can be printed, tested, and — via
//! [`ParamPlanEntry::runtime_flags`] — baked into the binary so the runtime
//! can confirm each parameter landed where the plan said (see
//! `nsl_runtime::param_plan`).
//!
//! The bit encoding is *shared with*, not mirrored from, the runtime: the
//! `PLAN_*` constants below are re-exports of `nsl_runtime::param_plan`'s, so
//! the two crates cannot drift the way the repo's other cross-crate ABI
//! surfaces have.
//!
//! # What this deliberately is NOT
//!
//! It is not a replacement for the feature-composition refusal matrix in
//! `nsl-cli/src/feature_rules.rs`. Those ~40 pairwise rules encode
//! compositions that are *unimplemented*, and a plan type cannot implement
//! them by describing them. [`ParameterPlan::derive`] re-checks only the
//! handful of rules that are invariants **of the plan's own shape** — a
//! parameter cannot be bf16-sr without streaming, cannot be sharded without
//! streaming, cannot be both bf16-sr and sharded. Those hold at the CLI
//! today; restating them here covers callers that build `CompileOptions`
//! directly, which `feature_rules.rs` explicitly documents as a real caller
//! class the clap layer does not protect.

pub use nsl_runtime::param_plan::{PLAN_BF16_SR, PLAN_ELEMENTWISE, PLAN_SHARDED, PLAN_STREAMED};

/// Where a parameter lives between uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamResidency {
    /// Device-resident for the whole step; no residency backend involved.
    Resident,
    /// Registered with a residency backend and materialized per use.
    Streamed,
}

/// The dtype of the authoritative copy of the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamStorage {
    /// FP32 master weights (the default).
    F32,
    /// BF16 authoritative storage with counter-based stochastic rounding on
    /// the fused update — no FP32 master copy (`--param-dtype bf16-sr`).
    Bf16Sr,
}

/// Whether the parameter is replicated on every rank or split across them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParamSharding {
    Replicated,
    /// Tensor-granular sharded (`--zero-stage 3`): the owner holds the
    /// authoritative bytes, non-owners gather per layer window.
    Sharded,
    /// Item 11 (`--zero-elementwise`): each rank persistently holds a
    /// contiguous 1/world_size slice; the full tensor is a window transient
    /// reconstructed by all_gather, gradients reduce_scatter, and EVERY rank
    /// runs the fused-AdamW step on its own slice. Eligibility (streamed,
    /// static numel divisible by world_size) is decided here in `derive` —
    /// ineligible streamed params fall back to `Sharded`.
    ShardedElementwise,
}

impl ParamSharding {
    /// Both zero-3 modes — the runtime residency table, the reduce site and
    /// the registration redirects treat them identically; only the gather
    /// primitive, the step dispatch and the owner gate differ.
    pub fn is_sharded(self) -> bool {
        matches!(self, ParamSharding::Sharded | ParamSharding::ShardedElementwise)
    }
}

/// The plan for one parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParamPlanEntry {
    /// Position in the runtime `param_list` — the cross-rank ordering key
    /// and the SR counter-block index. This, not the tensor pointer, is the
    /// parameter's identity.
    pub idx: i64,
    /// Dotted model field path (`blocks.0.w1`), for reports and diagnostics.
    pub name: String,
    pub residency: ParamResidency,
    pub storage: ParamStorage,
    pub sharding: ParamSharding,
}

impl ParamPlanEntry {
    /// The plan encoded for `nsl_param_plan_declare`.
    pub fn runtime_flags(&self) -> i64 {
        let mut f = 0;
        if self.residency == ParamResidency::Streamed {
            f |= PLAN_STREAMED;
        }
        if self.storage == ParamStorage::Bf16Sr {
            f |= PLAN_BF16_SR;
        }
        if self.sharding.is_sharded() {
            f |= PLAN_SHARDED;
        }
        if self.sharding == ParamSharding::ShardedElementwise {
            f |= PLAN_ELEMENTWISE;
        }
        f
    }

    /// Does this parameter need `nsl_sr_bf16_note_param` before its first
    /// `nsl_weight_stream_register`? (The SR backend aborts on a register of
    /// an un-noted parameter, so this ordering is load-bearing.)
    pub fn needs_sr_note(&self) -> bool {
        self.storage == ParamStorage::Bf16Sr && self.residency == ParamResidency::Streamed
    }

    /// Item 11: does this parameter need `nsl_zero3_mark_elementwise` before
    /// its first `nsl_weight_stream_register`? (The register carves the
    /// slice, so mark-before-register is load-bearing in the same way as the
    /// SR note.)
    pub fn is_elementwise(&self) -> bool {
        self.sharding == ParamSharding::ShardedElementwise
    }
}

/// The subset of `CompileOptions` that determines parameter storage.
///
/// Taken as a narrow struct rather than the whole options bag so the
/// derivation's inputs are visible in its signature — the pass-bus problem
/// (item 2) is that `stmt.rs` reads ~20 fields off `self` and no reader can
/// tell which ones matter.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlanFeatures {
    pub weight_stream: bool,
    pub param_dtype_bf16sr: bool,
    pub zero_stage: Option<u8>,
    /// Item 11: `--zero-elementwise` — elementwise-shard every ELIGIBLE
    /// streamed parameter (static numel divisible by `world_size`);
    /// ineligible ones stay tensor-granular.
    pub zero_elementwise: bool,
    /// Compile-time world size (`--devices`), the shard divisor. 0 is
    /// treated as 1 (the codegen setup emits `world_size.max(1)` too).
    pub world_size: u32,
}

/// A plan that could not be derived, with a message in the repo's refusal
/// voice (what was asked, why it cannot hold, what to change).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanError(pub String);

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParameterPlan {
    entries: Vec<ParamPlanEntry>,
}

impl ParameterPlan {
    /// Derive the plan for `names` (indexed by `param_list` position), given
    /// the set of indices the streaming schedule selected.
    ///
    /// `streamed` is the schedule's output, not a feature flag: streaming
    /// excludes view-rooted parameters and anything the layer grouping did
    /// not claim, so it is strictly narrower than "every parameter" even
    /// with `--weight-stream` on.
    ///
    /// `elems` carries each parameter's STATIC element count (0 = symbolic /
    /// unknown), aligned with `names` — item 11's elementwise eligibility is
    /// a divisibility fact about it. Only consulted under
    /// `--zero-elementwise`.
    pub fn derive(
        names: &[String],
        streamed: &[i64],
        elems: &[u64],
        f: &PlanFeatures,
    ) -> Result<Self, PlanError> {
        let sharding_on = f.zero_stage == Some(3);
        if f.zero_elementwise && !sharding_on {
            return Err(PlanError(
                "--zero-elementwise requires --zero-stage 3: elementwise \
                 slices are a refinement of the zero-3 residency backend"
                    .into(),
            ));
        }
        if f.zero_elementwise && elems.len() != names.len() {
            return Err(PlanError(format!(
                "--zero-elementwise: {} static element counts for {} \
                 parameter(s) — the derivation inputs are misaligned",
                elems.len(),
                names.len()
            )));
        }
        let ws = f.world_size.max(1) as u64;

        // ── Shape invariants (see the module header on scope) ──────────────
        if f.param_dtype_bf16sr && !f.weight_stream {
            return Err(PlanError(
                "--param-dtype bf16-sr requires --weight-stream: the bf16 \
                 authoritative copy IS the residency backend's mirror, so \
                 there is nowhere for it to live without streaming"
                    .into(),
            ));
        }
        if sharding_on && !f.weight_stream {
            return Err(PlanError(
                "--zero-stage 3 requires --weight-stream: sharded residency \
                 is realized by redirecting the weight-stream register/upload/\
                 evict sites to the broadcast backend"
                    .into(),
            ));
        }
        if f.param_dtype_bf16sr && sharding_on {
            return Err(PlanError(
                "--param-dtype bf16-sr does not compose with --zero-stage 3: \
                 both claim the residency backend, and nsl_weight_stream_register \
                 can only dispatch to one"
                    .into(),
            ));
        }
        if !f.weight_stream && !streamed.is_empty() {
            return Err(PlanError(format!(
                "streaming schedule selected {} parameter(s) but --weight-stream \
                 is off — the registration sites would never be emitted and the \
                 schedule would silently do nothing",
                streamed.len()
            )));
        }

        let mut is_streamed = vec![false; names.len()];
        for &i in streamed {
            let Ok(u) = usize::try_from(i) else {
                return Err(PlanError(format!(
                    "streaming schedule contains negative parameter index {i}"
                )));
            };
            if u >= names.len() {
                return Err(PlanError(format!(
                    "streaming schedule contains parameter index {i}, but the \
                     model has {} parameter(s)",
                    names.len()
                )));
            }
            if std::mem::replace(&mut is_streamed[u], true) {
                return Err(PlanError(format!(
                    "streaming schedule lists parameter index {i} twice — \
                     double registration would double-count the SR counter block"
                )));
            }
        }

        let entries = names
            .iter()
            .enumerate()
            .map(|(u, name)| {
                let streamed = is_streamed[u];
                ParamPlanEntry {
                    idx: u as i64,
                    name: name.clone(),
                    residency: if streamed {
                        ParamResidency::Streamed
                    } else {
                        ParamResidency::Resident
                    },
                    // A storage mode only exists for a parameter that is
                    // actually registered with a backend — a resident
                    // parameter under `--param-dtype bf16-sr` (e.g. a
                    // view-rooted one the schedule excluded) keeps f32
                    // master weights, because nothing ever built it a bf16
                    // mirror. Encoding that here is what stops the runtime
                    // check from demanding a mirror that never existed.
                    storage: if streamed && f.param_dtype_bf16sr {
                        ParamStorage::Bf16Sr
                    } else {
                        ParamStorage::F32
                    },
                    sharding: if streamed && sharding_on {
                        // Item 11: elementwise where the static shape allows
                        // an exact 1/ws split; tensor-granular otherwise
                        // (ragged or symbolic numel). The fallback keeps the
                        // whole streamed set sharded — the two modes coexist
                        // in one run.
                        let elem_ok = f.zero_elementwise
                            && elems
                                .get(u)
                                .is_some_and(|&n| n > 0 && n % ws == 0);
                        if elem_ok {
                            ParamSharding::ShardedElementwise
                        } else {
                            ParamSharding::Sharded
                        }
                    } else {
                        ParamSharding::Replicated
                    },
                }
            })
            .collect();

        Ok(Self { entries })
    }

    pub fn entries(&self) -> &[ParamPlanEntry] {
        &self.entries
    }

    /// Is any parameter registered with a residency backend? When false
    /// there is nothing for the runtime to cross-check and codegen skips
    /// emitting the declare/verify calls entirely.
    pub fn has_streamed(&self) -> bool {
        self.entries
            .iter()
            .any(|e| e.residency == ParamResidency::Streamed)
    }

    /// Must `nsl_sr_bf16_enable` be emitted? True iff some parameter's entry
    /// asks for an SR counter block.
    ///
    /// The SR backend aborts on a `register` of a parameter that never got a
    /// `note_param`, so arming the backend and noting a parameter have to be
    /// decided from one source. Reading both off this plan is what makes that
    /// structural rather than a coincidence of two matching `if` conditions.
    pub fn needs_sr_backend(&self) -> bool {
        self.entries.iter().any(|e| e.needs_sr_note())
    }

    /// Human-readable plan, printed under `NSL_PARAM_PLAN_REPORT=1`.
    pub fn report(&self) -> String {
        let total = self.entries.len();
        let streamed = self
            .entries
            .iter()
            .filter(|e| e.residency == ParamResidency::Streamed)
            .count();
        let bf16 = self
            .entries
            .iter()
            .filter(|e| e.storage == ParamStorage::Bf16Sr)
            .count();
        let sharded = self
            .entries
            .iter()
            .filter(|e| e.sharding.is_sharded())
            .count();
        let elementwise = self
            .entries
            .iter()
            .filter(|e| e.sharding == ParamSharding::ShardedElementwise)
            .count();
        let mut s = format!(
            "[param-plan] {total} parameter(s): {streamed} streamed \
             ({bf16} bf16-sr, {sharded} sharded, {elementwise} elementwise), \
             {} resident\n",
            total - streamed
        );
        let width = self
            .entries
            .iter()
            .map(|e| e.name.len())
            .max()
            .unwrap_or(0)
            .min(48);
        for e in &self.entries {
            let mode = match (e.residency, e.storage, e.sharding) {
                (ParamResidency::Resident, _, _) => "resident",
                (_, ParamStorage::Bf16Sr, _) => "streamed bf16-sr",
                (_, _, ParamSharding::ShardedElementwise) => "streamed sharded-elementwise",
                (_, _, ParamSharding::Sharded) => "streamed sharded",
                _ => "streamed",
            };
            // `param[` rather than indentation: the negative assertion in
            // the gate ("no per-parameter lines without the env var") is
            // pinned to this literal, and a needle made of whitespace runs
            // goes silently always-true the first time someone reformats.
            s.push_str(&format!(
                "[param-plan] param[{:>3}] {:<width$}  {mode}\n",
                e.idx,
                e.name,
                width = width
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("blocks.{i}.w")).collect()
    }

    #[test]
    fn plain_streaming_marks_only_the_scheduled_subset() {
        let f = PlanFeatures {
            weight_stream: true,
            ..Default::default()
        };
        let p = ParameterPlan::derive(&names(4), &[1, 2], &[], &f).unwrap();
        let res: Vec<_> = p.entries().iter().map(|e| e.residency).collect();
        assert_eq!(
            res,
            vec![
                ParamResidency::Resident,
                ParamResidency::Streamed,
                ParamResidency::Streamed,
                ParamResidency::Resident
            ]
        );
        // No storage mode requested: every entry is plain f32/replicated.
        assert!(p.entries().iter().all(|e| e.storage == ParamStorage::F32));
        assert_eq!(p.entries()[1].runtime_flags(), PLAN_STREAMED);
        assert_eq!(p.entries()[0].runtime_flags(), 0);
        assert!(p.has_streamed());
    }

    /// The load-bearing asymmetry: `--param-dtype bf16-sr` is a global flag,
    /// but a parameter the schedule excluded (view-rooted, unclaimed) is
    /// never registered and therefore never gets a bf16 mirror. Marking it
    /// bf16-sr would make the runtime check demand a mirror that does not
    /// and should not exist.
    #[test]
    fn a_storage_mode_never_applies_to_an_unscheduled_parameter() {
        let f = PlanFeatures {
            weight_stream: true,
            param_dtype_bf16sr: true,
            ..Default::default()
        };
        let p = ParameterPlan::derive(&names(3), &[0], &[], &f).unwrap();
        assert_eq!(p.entries()[0].storage, ParamStorage::Bf16Sr);
        assert!(p.entries()[0].needs_sr_note());
        assert_eq!(p.entries()[1].storage, ParamStorage::F32);
        assert!(!p.entries()[1].needs_sr_note());
        assert_eq!(
            p.entries()[0].runtime_flags(),
            PLAN_STREAMED | PLAN_BF16_SR
        );
        assert_eq!(p.entries()[1].runtime_flags(), 0);
    }

    #[test]
    fn zero3_shards_the_streamed_set_and_stages_1_2_do_not() {
        let mk = |stage| PlanFeatures {
            weight_stream: true,
            zero_stage: Some(stage),
            ..Default::default()
        };
        let p3 = ParameterPlan::derive(&names(3), &[0, 1], &[], &mk(3)).unwrap();
        assert_eq!(p3.entries()[0].sharding, ParamSharding::Sharded);
        assert_eq!(p3.entries()[2].sharding, ParamSharding::Replicated);
        assert_eq!(
            p3.entries()[0].runtime_flags(),
            PLAN_STREAMED | PLAN_SHARDED
        );
        // Stages 1 and 2 shard optimizer state / gradients, not parameters.
        for stage in [1, 2] {
            let p = ParameterPlan::derive(&names(3), &[0, 1], &[], &mk(stage)).unwrap();
            assert!(p
                .entries()
                .iter()
                .all(|e| e.sharding == ParamSharding::Replicated));
        }
    }

    /// Item 11: elementwise eligibility is a divisibility fact about the
    /// STATIC element count — ragged and symbolic (0) params fall back to
    /// tensor-granular, unstreamed ones stay replicated, and the flag off
    /// means nobody is elementwise. Mixed-mode in one plan is the norm.
    #[test]
    fn elementwise_eligibility_is_static_divisibility_with_granular_fallback() {
        let f = PlanFeatures {
            weight_stream: true,
            zero_stage: Some(3),
            zero_elementwise: true,
            world_size: 2,
            ..Default::default()
        };
        // elems: [divisible, ragged, symbolic, divisible-but-unstreamed]
        let p = ParameterPlan::derive(&names(4), &[0, 1, 2], &[8192, 127, 0, 64], &f).unwrap();
        assert_eq!(p.entries()[0].sharding, ParamSharding::ShardedElementwise);
        assert_eq!(p.entries()[1].sharding, ParamSharding::Sharded);
        assert_eq!(p.entries()[2].sharding, ParamSharding::Sharded);
        assert_eq!(p.entries()[3].sharding, ParamSharding::Replicated);
        assert_eq!(
            p.entries()[0].runtime_flags(),
            PLAN_STREAMED | PLAN_SHARDED | PLAN_ELEMENTWISE
        );
        assert_eq!(
            p.entries()[1].runtime_flags(),
            PLAN_STREAMED | PLAN_SHARDED
        );
        assert!(p.entries()[0].is_elementwise());
        assert!(!p.entries()[1].is_elementwise());
        // The report names the mode (the GPU gate greps the runtime marker,
        // this pins the compile-side spelling).
        assert!(p.report().contains("streamed sharded-elementwise"), "{}", p.report());
        assert!(p.report().contains("1 elementwise"), "{}", p.report());

        // Flag off: identical inputs, nobody is elementwise.
        let off = PlanFeatures {
            zero_elementwise: false,
            ..f
        };
        let p = ParameterPlan::derive(&names(4), &[0, 1, 2], &[8192, 127, 0, 64], &off).unwrap();
        assert!(p.entries().iter().all(|e| !e.is_elementwise()));
    }

    /// Item 11: the flag's own shape invariants refuse loudly.
    #[test]
    fn elementwise_shape_invariants_refuse() {
        // Without stage 3.
        let e = ParameterPlan::derive(
            &names(2),
            &[],
            &[],
            &PlanFeatures {
                weight_stream: true,
                zero_elementwise: true,
                world_size: 2,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(e.0.contains("requires --zero-stage 3"), "{e}");
        // Misaligned element counts.
        let e = ParameterPlan::derive(
            &names(2),
            &[0],
            &[64],
            &PlanFeatures {
                weight_stream: true,
                zero_stage: Some(3),
                zero_elementwise: true,
                world_size: 2,
                ..Default::default()
            },
        )
        .unwrap_err();
        assert!(e.0.contains("misaligned"), "{e}");
    }

    #[test]
    fn shape_invariants_are_refused_with_an_actionable_message() {
        let bf16_no_ws = PlanFeatures {
            param_dtype_bf16sr: true,
            ..Default::default()
        };
        let e = ParameterPlan::derive(&names(2), &[], &[], &bf16_no_ws).unwrap_err();
        assert!(e.0.contains("requires --weight-stream"), "{e}");

        let z3_no_ws = PlanFeatures {
            zero_stage: Some(3),
            ..Default::default()
        };
        let e = ParameterPlan::derive(&names(2), &[], &[], &z3_no_ws).unwrap_err();
        assert!(e.0.contains("requires --weight-stream"), "{e}");

        let both = PlanFeatures {
            weight_stream: true,
            param_dtype_bf16sr: true,
            zero_stage: Some(3),
            ..Default::default()
        };
        let e = ParameterPlan::derive(&names(2), &[0], &[], &both).unwrap_err();
        assert!(e.0.contains("does not compose"), "{e}");

        let ws_off = PlanFeatures::default();
        let e = ParameterPlan::derive(&names(2), &[0], &[], &ws_off).unwrap_err();
        assert!(e.0.contains("--weight-stream is off"), "{e}");
    }

    #[test]
    fn a_malformed_schedule_is_refused_rather_than_silently_dropped() {
        let f = PlanFeatures {
            weight_stream: true,
            ..Default::default()
        };
        let e = ParameterPlan::derive(&names(2), &[5], &[], &f).unwrap_err();
        assert!(e.0.contains("has 2 parameter(s)"), "{e}");
        let e = ParameterPlan::derive(&names(2), &[-1], &[], &f).unwrap_err();
        assert!(e.0.contains("negative"), "{e}");
        let e = ParameterPlan::derive(&names(2), &[0, 0], &[], &f).unwrap_err();
        assert!(e.0.contains("twice"), "{e}");
    }

    /// With no features on, every parameter is resident and the plan reports
    /// itself as having nothing to check — codegen then emits no declare or
    /// verify calls at all.
    #[test]
    fn a_featureless_plan_has_nothing_to_verify() {
        let p =
            ParameterPlan::derive(&names(3), &[], &[], &PlanFeatures::default()).unwrap();
        assert!(!p.has_streamed());
        assert!(!p.needs_sr_backend());
        assert!(p.entries().iter().all(|e| e.runtime_flags() == 0));
    }

    /// `needs_sr_backend` gates the `nsl_sr_bf16_enable` emission and
    /// `needs_sr_note` gates the per-parameter note. The SR backend aborts
    /// when a parameter is registered without a note, so "armed" must imply
    /// "some parameter will be noted" and vice versa — asserted here rather
    /// than trusted, because the two are emitted at different call sites.
    #[test]
    fn arming_the_sr_backend_agrees_with_noting_parameters() {
        let ws = PlanFeatures {
            weight_stream: true,
            ..Default::default()
        };
        let sr = PlanFeatures {
            weight_stream: true,
            param_dtype_bf16sr: true,
            ..Default::default()
        };
        for (f, streamed) in [
            (ws, &[0i64, 1][..]),
            (ws, &[][..]),
            (sr, &[0i64][..]),
            (sr, &[][..]),
        ] {
            let p = ParameterPlan::derive(&names(3), streamed, &[], &f).unwrap();
            assert_eq!(
                p.needs_sr_backend(),
                p.entries().iter().any(|e| e.needs_sr_note()),
                "arm/note disagreement for {f:?} streamed={streamed:?}"
            );
        }
        // And bf16-sr with an EMPTY schedule must not arm the backend: there
        // is no parameter to note, so arming it would guarantee the abort.
        assert!(!ParameterPlan::derive(&names(3), &[], &[], &sr)
            .unwrap()
            .needs_sr_backend());
    }

    #[test]
    fn the_report_names_each_parameters_mode() {
        let f = PlanFeatures {
            weight_stream: true,
            param_dtype_bf16sr: true,
            ..Default::default()
        };
        let r = ParameterPlan::derive(&names(2), &[1], &[], &f).unwrap().report();
        assert!(r.contains(
            "2 parameter(s): 1 streamed (1 bf16-sr, 0 sharded, 0 elementwise), 1 resident"
        ));
        assert!(r.contains("blocks.0.w"), "{r}");
        assert!(r.contains("streamed bf16-sr"), "{r}");
    }
}
