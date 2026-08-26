//! Item 20: the stderr execution-marker vocabulary.
//!
//! NSL's subsystems announce themselves on stderr with a bracketed tag —
//! `[csla]`, `[zero3]`, `[fase-multi]`. Integration tests assert on those tags
//! to prove a feature actually engaged (rather than the run merely producing
//! the right number), so the tags are a de-facto API between codegen/runtime
//! and ~42 assertions spread across 19 test files.
//!
//! Centralizing them buys two things a scattered string literal cannot:
//!
//! 1. **A renamed tag stops being a silent pass.** A POSITIVE assertion
//!    (`stderr.contains("[csla]")`) fails loudly if the tag is renamed — fine.
//!    A NEGATIVE one (`!stderr.contains("[csha] csha[")`, asserting a feature
//!    did NOT engage) passes forever once the tag no longer exists, which is
//!    exactly backwards. Constants make a rename a compile error at every use
//!    site, and [`EXEC_MARKERS`] pins that each tag is still emitted.
//! 2. **The vocabulary becomes reviewable.** Nineteen tags were in use and
//!    nobody could see the list.
//!
//! # What this does NOT assert
//!
//! That a marker is emitted under the right *conditions* — only that the
//! emitting call site still exists. A marker moved behind a condition that
//! never holds would keep this green; only a test that runs the feature and
//! looks for the tag can catch that, which is what the integration gates do.

/// One stderr marker, and where it comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecMarker {
    /// The literal token as it appears on stderr, brackets included.
    pub token: &'static str,
    /// Repo-relative files whose print macros emit it. At least one must
    /// still contain the token. A LIST because most markers are printed from
    /// several places, and pinning only one lets the others be renamed
    /// silently — which is exactly how the first version of this registry
    /// pointed `[csha]` at a GPU-spec probe while the assertion that mattered
    /// depended on a plan summary in another file entirely.
    pub emitted_by: &'static [&'static str],
    /// What its presence tells a test.
    pub means: &'static str,
}

const fn m(
    token: &'static str,
    emitted_by: &'static [&'static str],
    means: &'static str,
) -> ExecMarker {
    ExecMarker {
        token,
        emitted_by,
        means,
    }
}

/// Every marker an integration test asserts on. Verified present in
/// `emitted_by` by `feature_composition_gate::every_exec_marker_is_still_emitted`.
pub const EXEC_MARKERS: &[ExecMarker] = &[
    m(
        "[wggo]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
            "crates/nsl-codegen/src/wggo.rs",
            "crates/nsl-codegen/src/compiler/kernel.rs",
            // The pre-pass's "no pre-plan for training block #N (distill
            // block)" note. Registered because the list is the point: the
            // gate only requires SOME listed file to still contain the
            // token, so an unlisted emit site can be renamed silently.
            "crates/nsl-codegen/src/wggo_prepass.rs",
        ],
        "the WGGO planner ran; `consumed pre-solved plan` / `wggo[...]` mean it APPLIED one",
    ),
    m(
        "[csla]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
        ],
        "the CSLA window-buffered schedule was lowered or executed",
    ),
    m(
        "[grad-integrity]",
        &[
            "crates/nsl-runtime/src/grad_integrity.rs",
        ],
        "the P0.3 gradient-integrity atexit block (armed by --grad-integrity \
         or NSL_GRAD_INTEGRITY=1): worst-case-over-steps counts proving every \
         trainable parameter received a finite, mostly-nonzero gradient. \
         Registered 2026-08-24 — it had test consumers (csla_layerwise_gate, \
         grad_integrity_gate) but sat OUTSIDE this registry until the \
         events-schema consistency gate required its marker to exist here",
    ),
    m(
        "[zero]",
        &[
            "crates/nsl-runtime/src/zero.rs",
            "crates/nsl-runtime/src/args.rs",
        ],
        "ZeRO stage 1/2 collectives ran",
    ),
    m(
        "[zero3]",
        &[
            "crates/nsl-runtime/src/zero.rs",
        ],
        "ZeRO stage 3 parameter sharding engaged",
    ),
    m(
        "[muon]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
        ],
        "the Muon optimizer arm was lowered",
    ),
    m(
        "[wd-groups]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
        ],
        "AdamW parameter groups (`no_decay=[...]`) resolved a per-parameter \
         weight decay; the table names every parameter it exempted",
    ),
    m(
        "[muon-state]",
        &[
            "crates/nsl-runtime/src/sr_bf16.rs",
        ],
        "the bf16 Muon momentum envelope engaged",
    ),
    m(
        "[fuse]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
        ],
        "a fusion peephole fired",
    ),
    m(
        "[fused-ew]",
        &[
            "crates/nsl-runtime/src/tensor/fused_chain.rs",
        ],
        "the fused elementwise-chain runtime: launch/fallback counters and \
         the decomposed-replay warn-once",
    ),
    m(
        "[csha]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
            "crates/nsl-codegen/src/csha.rs",
            "crates/nsl-codegen/src/compiler/kernel.rs",
        ],
        "CSHA planning ran; `[csha] csha[...]` is the plan SUMMARY (stmt.rs), while csha.rs also prints an unrelated `[csha] gpu spec:` probe",
    ),
    m(
        "[pca]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
            "crates/nsl-codegen/src/compiler/kernel.rs",
        ],
        "packed-context attention engaged",
    ),
    m(
        "[pca-per-doc]",
        &[
            "crates/nsl-codegen/src/compiler/kernel.rs",
        ],
        "PCA per-document admission ran",
    ),
    m(
        "[fase-multi]",
        &[
            "crates/nsl-runtime/src/fase_step.rs",
        ],
        "the multi-tensor FASE optimizer step fired",
    ),
    m(
        "[phase]",
        &[
            "crates/nsl-runtime/src/math.rs",
        ],
        "NSL_PHASE_TIMING instrumentation ran: one `fwd=/bwd=` line per \
         MICRO-batch and one `opt=` line per OPTIMIZER step, so the two counts \
         together witness the accumulation window (stdout, not stderr) — but \
         SOURCE-AD PATH ONLY for the `fwd=/bwd=` half, which stmt.rs emits \
         around the lowered adjoint; the tape path's backward is a single \
         opaque nsl_tape_backward call and emits no such line, so a tape-AD \
         run yields `opt=` lines alone and the micro-batch count is unwitnessed",
    ),
    m(
        "[tape-ad]",
        &[
            "crates/nsl-runtime/src/autodiff/backward.rs",
        ],
        "the tape backward hit a FATAL disconnection (this token has no success form)",
    ),
    m(
        "[sr-bf16]",
        &[
            "crates/nsl-runtime/src/sr_bf16.rs",
        ],
        "stochastic-rounding bf16 parameter storage engaged",
    ),
    m(
        "[pass-trace]",
        &[
            // The report text is built here...
            "crates/nsl-codegen/src/pass_trace.rs",
            // ...and printed here. Pinning only the builder would let the
            // eprint! be deleted with the marker gate still green.
            "crates/nsl-cli/src/commands/build/mod.rs",
        ],
        "the registered compiler passes that actually ran this compile, in \
         first-invocation order (roadmap item 2)",
    ),
    m(
        "[pass-bus]",
        &[
            // The report text is built here...
            "crates/nsl-codegen/src/pass_bus.rs",
            // ...and printed here, from the same emitter as `[pass-trace]`:
            // the bus's SILENT DEFAULT finding is computed by cross-referencing
            // the two, so they must reach every terminal path together.
            "crates/nsl-cli/src/commands/build/mod.rs",
        ],
        "per-channel traffic on the inter-pass bus — which pass published what, \
         how many consumers read it, and the two patterns that are defects \
         (roadmap item 2 step 4)",
    ),
    m(
        "[cpdt]",
        &[
            // The planning driver's diagnostics: tier agreement, the
            // moment-precision arbitration arms (active / NOT lowered), the
            // stale-plan refusal's neighbors, and the no-WGGO skip notice.
            "crates/nsl-codegen/src/stmt.rs",
            // Decorator, MoE capacity, and expert-prune diagnostics.
            "crates/nsl-codegen/src/cpdt_decorator.rs",
            "crates/nsl-codegen/src/cpdt_moe_capacity.rs",
            "crates/nsl-codegen/src/cpdt_expert_prune.rs",
            // CLI-side plan rendering.
            "crates/nsl-cli/src/commands/build/options.rs",
            "crates/nsl-cli/src/commands/build/run.rs",
        ],
        "CPDT planning diagnostics — tier agreement, optimizer-moment \
         precision arbitration, and the skip/refusal notices for plans that \
         cannot apply (roadmap item 12 family)",
    ),
    m(
        "[param-plan]",
        &[
            // The compile-time report...
            "crates/nsl-codegen/src/parameter_plan.rs",
            // ...and the runtime cross-check that confirms each parameter
            // landed in the backend the plan named.
            "crates/nsl-runtime/src/param_plan.rs",
        ],
        "the compiled ParameterPlan was verified against the residency tables \
         (or, with FATAL, a parameter landed in the wrong backend)",
    ),
    m(
        "[flash-bwd]",
        &[
            "crates/nsl-runtime/src/flash_attention.rs",
        ],
        "the flash-attention backward ran",
    ),
    m(
        "[arena]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
        ],
        "the transient-memory arena was used",
    ),
    m(
        "[gpu-mem]",
        &[
            "crates/nsl-runtime/src/tensor/mod.rs",
        ],
        "GPU memory accounting reported",
    ),
    m(
        "[nsl-gpu-launch-count]",
        &[
            "crates/nsl-runtime/src/fused_adapter.rs",
        ],
        "the fused-adapter GPU launch counter reported",
    ),
    m(
        "[cuda-graph]",
        &["crates/nsl-runtime/src/cuda/graph_capture.rs"],
        "CUDA-graph capture/replay reported a region banner",
    ),
    m(
        "[fase-fused]",
        &["crates/nsl-runtime/src/args.rs"],
        "the p9 fused FASE optimizer-step counter reported (NSL_FASE_FUSED_COUNTER=1)",
    ),
    m(
        "[weight-stream]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
            "crates/nsl-runtime/src/weight_stream.rs",
            "crates/nsl-runtime/src/args.rs",
        ],
        "layer-major weight streaming engaged (upload/evict, or its FATAL placement check)",
    ),
    m(
        "[wgrad-accum]",
        &["crates/nsl-runtime/src/args.rs"],
        "item 7 fused-GEMM vs decomposed-fallback counts (NSL_WGRAD_COUNTER=1)",
    ),
    m(
        "[wgrad-fusion]",
        &[
            "crates/nsl-codegen/src/wengert_lower.rs",
            // stmt.rs: the `declined:` note. Registered separately because the
            // two lines are mutually exclusive by construction — the COUNT is
            // gated on the FASE-Deferred hook, so a build without that hook
            // emitted nothing at all and the feature was invisibly inert on
            // exactly the shipped pretraining configuration. Pinning only
            // wengert_lower.rs would let the decline be renamed silently.
            "crates/nsl-codegen/src/stmt.rs",
            // compiler/mod.rs: the SAME `declined:` note for the one case
            // stmt.rs structurally cannot reach — a program with no train
            // block, where `compile_train_block_inner` never runs, so
            // `finish_wgrad_admission` is the only thing left to say the flag
            // did nothing.
            "crates/nsl-codegen/src/compiler/mod.rs",
        ],
        "item 7 reported what `--fuse-wgrad-accum` did: `N chain(s) fused` \
         counts the chains the pre-pass collapsed, `declined:` names the \
         precondition that kept the fusion from firing at all (and which \
         train block, since a decline no longer implies a failed build)",
    ),
    m(
        "[ccr]",
        &["crates/nsl-codegen/src/stmt.rs"],
        "CCR reported its adjoint last-use free placement (NSL_CCR_DEBUG=1), \
         and warns when that placement broke a weight-gradient fusion chain",
    ),
    m(
        "[fused-lm-ce]",
        &[
            // stmt.rs: the partial-decline warning, and the
            // extraction-failed warning.
            "crates/nsl-codegen/src/stmt.rs",
            // source_ad.rs: the explicit `fused_linear_ce(...)` call
            // diagnostics (no decorator / disabled / missing hints). Listed
            // because pinning only one emitting file lets the others be
            // renamed silently — the exact defect a review caught in this
            // registry's first version.
            "crates/nsl-codegen/src/source_ad.rs",
        ],
        "item 6 warned that a fused linear-CE substitution did not happen \
         while @fused_lm_ce was active (the PARTIAL / unextractable / \
         explicit-call cases — a total decline failure is a hard CodegenError, \
         not this marker)",
    ),
    m(
        "[fused-lce]",
        &[
            "crates/nsl-codegen/src/wengert_lower.rs",
        ],
        "item 6 named the fused linear-CE KERNEL FAMILY this compile chose \
         (route=gemm | v1 | v1-large) with the inputs that decided it. \
         Distinct from `[fused-lm-ce]`, which reports a substitution that did \
         NOT happen: this one reports which kernels a substitution that DID \
         happen will run, a decision worth 335x/504x at production shape and \
         previously reported by nothing",
    ),
    m(
        "[lm-head-fusion]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
        ],
        "item 4 reported what `--fuse-lm-head` decided: `inferred:` names the \
         four dims of a fused LM head the compiler installed with no \
         @fused_lm_ce decorator, `declined:` says why it kept the composite \
         path. Deliberately NOT `[fused-lm-ce]`, which means the opposite \
         (a fusion that did not happen) and is asserted ABSENT by \
         fused_lm_ce_decline_gate's success cases",
    ),
    m(
        "[checkpoint]",
        &[
            "crates/nsl-runtime/src/checkpoint.rs",
        ],
        "Milestone B full-train-state checkpointing: `saved: <path> (+.optim) \
         at micro-batch step N` after each boundary save (θ .nslm + AdamW \
         m/v + step counter, tmp-and-rename atomic), `resumed: ...` after \
         checkpoint_load restored all three and seeded the step counter. \
         train_checkpoint_gate.rs asserts both; endurance_1b.py asserts the \
         resume witness at 1B",
    ),
    m(
        "[pass-manager]",
        &[
            "crates/nsl-codegen/src/pass_manager.rs",
        ],
        "Milestone C scheduler trace: `-> PASS phase=... epoch=N tape=... \
         predecessors=...` at each scheduled invocation, `<- PASS \
         disposition=...` (or `postconditions deferred: <reason>`) at its \
         boundary, and the non-LIFO epoch-drop BUG line. OPT-IN \
         (NSL_PASS_TRACE=1, the same switch as the pass trace), \
         which is the point of the entry — pass_scheduler_gate.rs asserts it \
         ABSENT on a default build, and a renamed tag would make that \
         negative assertion pass forever",
    ),
    m(
        "[nsl-kernel-count]",
        &[
            "crates/nsl-runtime/src/fused_adapter.rs",
        ],
        "the kernel counter reported",
    ),
    m(
        "[fase]",
        &[
            "crates/nsl-codegen/src/stmt.rs",
        ],
        "the FASE planner's driver reported a decorator/override decision \
         (the @fase activation witness is `[fase] @fase decorator applied:`)",
    ),
    m(
        "[activation]",
        &[
            "crates/nsl-codegen/src/activation.rs",
        ],
        "the Milestone A reconciler reported a requested surface's outcome \
         (applied / declined / witnessed / UNSATISFIED)",
    ),
    m(
        "[source-ad]",
        &[
            "crates/nsl-codegen/src/source_ad.rs",
            "crates/nsl-codegen/src/wengert_lower.rs",
        ],
        "a source-AD extraction/lowering diagnostic (extraction failures, \
         unresolved VarIds, CSHA claims). The dropout parity gate asserts the \
         NEGATIVE form: the pre-2026-08-16 wrong-backward warning must never \
         return (see its NEGATIVE_NEEDLES tombstone entry)",
    ),
];

/// Look a marker up by token. Panics if it is not registered — call sites are
/// meant to use the constants below, so an unknown token is a bug.
pub fn marker(token: &str) -> &'static ExecMarker {
    EXEC_MARKERS
        .iter()
        .find(|m| m.token == token)
        .unwrap_or_else(|| panic!("unregistered exec marker {token:?}"))
}

/// Named constants for the tokens tests assert on. Using these instead of a
/// string literal means renaming a marker breaks the build at every use site
/// rather than silently flipping a negative assertion to always-true.
pub mod tokens {
    pub const WGGO: &str = "[wggo]";
    pub const CSLA: &str = "[csla]";
    pub const GRAD_INTEGRITY: &str = "[grad-integrity]";
    pub const ZERO: &str = "[zero]";
    pub const ZERO3: &str = "[zero3]";
    pub const MUON: &str = "[muon]";
    pub const WD_GROUPS: &str = "[wd-groups]";
    pub const MUON_STATE: &str = "[muon-state]";
    pub const FUSE: &str = "[fuse]";
    pub const CSHA: &str = "[csha]";
    pub const PCA: &str = "[pca]";
    pub const PCA_PER_DOC: &str = "[pca-per-doc]";
    pub const FASE_MULTI: &str = "[fase-multi]";
    pub const PHASE: &str = "[phase]";
    pub const TAPE_AD: &str = "[tape-ad]";
    pub const SR_BF16: &str = "[sr-bf16]";
    pub const PARAM_PLAN: &str = "[param-plan]";
    pub const CPDT: &str = "[cpdt]";
    pub const PASS_TRACE: &str = "[pass-trace]";
    pub const PASS_BUS: &str = "[pass-bus]";
    pub const FLASH_BWD: &str = "[flash-bwd]";
    pub const ARENA: &str = "[arena]";
    pub const GPU_MEM: &str = "[gpu-mem]";
    pub const GPU_LAUNCH_COUNT: &str = "[nsl-gpu-launch-count]";
    pub const KERNEL_COUNT: &str = "[nsl-kernel-count]";
    pub const CUDA_GRAPH: &str = "[cuda-graph]";
    pub const FASE_FUSED: &str = "[fase-fused]";
    pub const WEIGHT_STREAM: &str = "[weight-stream]";
    pub const WGRAD_ACCUM: &str = "[wgrad-accum]";
    pub const WGRAD_FUSION: &str = "[wgrad-fusion]";
    pub const CCR: &str = "[ccr]";
    pub const FASE: &str = "[fase]";
    pub const ACTIVATION: &str = "[activation]";
}

/// A string a NEGATIVE assertion depends on.
///
/// [`EXEC_MARKERS`] pins that a TOKEN is still emitted somewhere. That is not
/// enough, and the gap is what a review caught in this file's first version:
/// `[csha]` was pinned against `csha.rs`, whose only `[csha]` lines are an
/// unrelated GPU-spec probe, while the assertion that mattered
/// (`!stderr.contains("[csha] csha[")`) depends on a plan summary assembled
/// from `stmt.rs` AND `csha.rs`. Renaming the summary prefix would have left
/// the gate green and that assertion permanently, silently true.
///
/// So the needle itself is pinned, part by part — because a needle can be
/// assembled from more than one emit site, no single file contains it whole.
#[derive(Debug, Clone, Copy)]
pub struct NegativeNeedle {
    /// Test file whose assertion breaks (into always-passing) if this rots.
    pub test: &'static str,
    /// What the assertion is claiming did NOT happen.
    pub asserts: &'static str,
    /// `(literal, file)` — each literal must still appear in a print macro in
    /// that file.
    pub parts: &'static [(&'static str, &'static str)],
}

/// Every negative marker assertion in the suite. A positive assertion fails
/// loudly when its marker rots and needs no entry here; a negative one cannot
/// fail at all, so each is pinned.
pub const NEGATIVE_NEEDLES: &[NegativeNeedle] = &[
    NegativeNeedle {
        test: "crates/nsl-cli/tests/dropout_backward_parity_gate.rs",
        asserts: "the pre-2026-08-16 wrong-backward dropout warning \
                  (\"[source-ad] WARNING: dropout\") never returns. TOMBSTONE: \
                  the banned literal deliberately has NO emit site — the \
                  backward now consumes the exact forward mask, so parts is \
                  empty by design. This needle cannot rot into always-true; \
                  always-true IS its steady state, and it fails exactly when \
                  someone reintroduces the warning verbatim",
        parts: &[],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/dropout_elision_gate.rs",
        asserts: "same tombstone as dropout_backward_parity_gate.rs (that gate \
                  reads the banned literal through a const, which the sweep \
                  cannot see — registered here for the record anyway)",
        parts: &[],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/cpdt_decorator_activation_gate.rs",
        asserts: "a control build without @cpdt carries no decorator-applied marker",
        parts: &[
            // eprintln!("[cpdt] @cpdt decorator applied: mode={} ...")
            ("[cpdt] @cpdt decorator applied", "crates/nsl-codegen/src/cpdt_decorator.rs"),
        ],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/fase_decorator_activation_gate.rs",
        asserts: "a control build without @fase carries no decorator-applied witness",
        parts: &[
            // eprintln!("[fase] @fase decorator applied: mode={:?} ...")
            ("[fase] @fase decorator applied:", "crates/nsl-codegen/src/stmt.rs"),
        ],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/csha_checkpoint_decorator_cli_e2e.rs",
        asserts: "@csha(disable=true) suppressed the CSHA plan summary",
        parts: &[
            // eprintln!("[csha] {}", plan.summary())
            ("[csha] {}", "crates/nsl-codegen/src/stmt.rs"),
            // format!("csha[{}]: {} chains, ...")  <- the summary's own prefix
            ("csha[{}]", "crates/nsl-codegen/src/csha.rs"),
        ],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/lm_head_inference_gate.rs",
        asserts: "no LM head was inferred — the default-off case, and the case \
                  where an @fused_lm_ce(enabled = false) opt-out was honoured",
        // The bare token is the whole needle: those two tests assert no
        // `[lm-head-fusion]` line AT ALL, neither `inferred:` nor `declined:`,
        // because "inference never ran" and "inference ran and passed" have to
        // be distinguishable. One emitting file — the train-block lowering.
        parts: &[("[lm-head-fusion]", "crates/nsl-codegen/src/stmt.rs")],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/mse_leak_gate.rs",
        asserts: "the tape backward did not hit a FATAL disconnection",
        parts: &[("[tape-ad] FATAL", "crates/nsl-runtime/src/autodiff/backward.rs")],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/cuda_graph_gate.rs",
        asserts: "NSL_ASYNC_ALLOC=1 refused capture BEFORE any region armed — \
                  the teardown banner must be absent, not merely zeroed, \
                  because `enable()` returns before ENABLED is set and a run \
                  that armed and captured nothing would print `captured=0`",
        // The banner's own prefix, from the one file that formats it.
        parts: &[(
            "[cuda-graph] regions=",
            "crates/nsl-runtime/src/cuda/graph_capture.rs",
        )],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/pass_trace_gate.rs",
        asserts: "the pass trace stayed silent without NSL_PASS_TRACE=1",
        // The bare token is the whole needle (the test asserts no
        // `[pass-trace]` line at all), and it is emitted from exactly one
        // place: the report builder.
        parts: &[("[pass-trace]", "crates/nsl-codegen/src/pass_trace.rs")],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/pass_bus_gate.rs",
        asserts: "the bus report stayed silent without NSL_PASS_TRACE=1",
        // Same shape as the pass-trace needle above: the test asserts no
        // `[pass-bus]` line at all, and the bare token is emitted from exactly
        // one place, the report builder.
        parts: &[("[pass-bus]", "crates/nsl-codegen/src/pass_bus.rs")],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/param_plan_gate.rs",
        asserts: "the per-parameter plan listing stayed OFF without \
                  NSL_PARAM_PLAN_REPORT=1 (the one-line summary still prints)",
        // The listing's own prefix, which is why it is `param[` and not an
        // indent: a needle made of whitespace would go always-true the first
        // time the report is reformatted, and the paired positive assertion
        // in the same test would keep passing.
        parts: &[(
            "[param-plan] param[",
            "crates/nsl-codegen/src/parameter_plan.rs",
        )],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/wgrad_accum_fusion_gate.rs",
        asserts: "a build that DID fuse reported no decline — the two outcomes \
                  share one token, so a run that both fused and declined would \
                  mean the admission check and the lowerer disagree",
        // The `declined:` suffix, not the bare token: the same token also
        // carries the `N chain(s) fused` count, so pinning `[wgrad-fusion]`
        // alone would stay green if the decline were renamed — and the decline
        // is the whole point, because without it a non-fusing build printed
        // NOTHING (the count is gated on the same FASE hook the fusion needs).
        parts: &[(
            "[wgrad-fusion] declined:",
            "crates/nsl-codegen/src/stmt.rs",
        )],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/rmsnorm_residual_fold_gate.rs",
        asserts: "the rmsnorm dx+residual fold did NOT fire with the flag off",
        parts: &[(
            "[fuse] rmsnorm dx+residual folds:",
            "crates/nsl-codegen/src/stmt.rs",
        )],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/elementwise_chain_fusion_gate.rs",
        asserts: "the elementwise chain fuser did NOT fire with \
                  NSL_FUSE_ELEMENTWISE_BWD=0 (and did not run under \
                  --layerwise-accum)",
        parts: &[(
            "[fuse] elementwise backward chains:",
            "crates/nsl-codegen/src/stmt.rs",
        )],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/scalar_immediate_fold_gate.rs",
        asserts: "the scalar-immediate sweep did NOT fire with \
                  NSL_FUSE_SCALAR_IMM=0",
        parts: &[(
            "[fuse] scalar immediates:",
            "crates/nsl-codegen/src/stmt.rs",
        )],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/rope_neg_fold_gate.rs",
        asserts: "the RoPE backward fold did NOT fire with NSL_FUSE_ROPE_NEG=0",
        parts: &[(
            "[fuse] rope backward folds:",
            "crates/nsl-codegen/src/stmt.rs",
        )],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/fused_lm_ce_decline_gate.rs",
        asserts: "a matching LM head, and a disabled decorator, produced NO \
                  fused linear-CE fallback diagnostic",
        // Both assertions are `!stderr.contains("[fused-lm-ce]")`, so the bare
        // token is the whole needle. Two emitting files: the partial-decline /
        // extraction-failed warnings in stmt.rs and the explicit-call
        // diagnostics in source_ad.rs. Rename either prefix and BOTH negative
        // assertions become permanently true while the positive assertions in
        // the same file still pass — which is why the token is pinned per
        // file rather than once.
        parts: &[
            ("[fused-lm-ce]", "crates/nsl-codegen/src/stmt.rs"),
            ("[fused-lm-ce]", "crates/nsl-codegen/src/source_ad.rs"),
        ],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/fused_lm_ce_hint_pin_gate.rs",
        asserts: "the two control arms fused for real -- a DECLINE would mean \
                  the composite path ran, the hint pin was never reached, and \
                  the arm proved nothing about false positives",
        // Same bare token and same two emitting files as the decline gate
        // above. Pinned separately because these arms carry different weight:
        // there the assertion says a matching head produced no fallback, here
        // it is what stops a false-positive control from passing vacuously
        // against a program that never engaged the kernel at all.
        parts: &[
            ("[fused-lm-ce]", "crates/nsl-codegen/src/stmt.rs"),
            ("[fused-lm-ce]", "crates/nsl-codegen/src/source_ad.rs"),
        ],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/multi_adamw_gate.rs",
        asserts: "the multi-tensor FASE step did NOT fire with the flag off",
        parts: &[("[fase-multi]", "crates/nsl-runtime/src/fase_step.rs")],
    },
    NegativeNeedle {
        test: "crates/nsl-cli/tests/cuda_graph_gate.rs",
        asserts: "the NSL_ASYNC_ALLOC=1 refusal disabled capture — no summary \
                  banner was printed, i.e. capture never ran",
        // The `regions=` suffix, not the bare `[cuda-graph]` token: the same
        // test asserts POSITIVELY on `[cuda-graph] disabled:`, so a
        // bare-token needle would stay satisfied by that line alone even if
        // the summary banner were renamed — exactly the rot this gate exists
        // to catch. One emitting site: the end-of-run report.
        parts: &[(
            "[cuda-graph] regions=",
            "crates/nsl-runtime/src/cuda/graph_capture.rs",
        )],
    },
];

// ───────────────────────────────────────────────────────────────────────────
// Item 17: structured-event schemas (`NSL_EVENTS=<path>` JSONL twins)
// ───────────────────────────────────────────────────────────────────────────

/// Schema of one structured event kind emitted by `nsl_runtime::events`.
///
/// Every kind twins a stderr marker above: the runtime builds both renderings
/// from ONE counter snapshot, so values cannot disagree — but field NAMES can
/// rot independently of the emitting call site, which is exactly the gap the
/// marker registry's gates leave open (they pin that the emit exists, not
/// what it says). `events_stream_gate.rs` runs a fixture with `NSL_EVENTS`
/// set and checks every event of a registered kind carries at least these
/// fields, so renaming or dropping a field fails a GPU-free gate instead of
/// silently breaking whichever consumer read it.
///
/// `fields` is the REQUIRED set — kinds may grow fields (consumers must
/// ignore unknown keys; the envelope version `v` only bumps for envelope
/// changes). Removing or renaming a listed field is a breaking change and
/// must update this table and every consumer in the same commit.
pub struct EventSchema {
    /// `kind` value in the event envelope.
    pub kind: &'static str,
    /// The stderr marker this event twins (must appear in `EXEC_MARKERS`).
    pub marker: &'static str,
    /// Field names REQUIRED in `fields` of every event of this kind.
    pub fields: &'static [&'static str],
}

pub const EVENT_SCHEMAS: &[EventSchema] = &[
    EventSchema {
        kind: "fused_ew_counters",
        marker: "[fused-ew]",
        fields: &["launches", "fallbacks"],
    },
    EventSchema {
        kind: "zero_counters",
        marker: "[zero]",
        fields: &[
            "ws", "rank", "all_reduce", "broadcast", "optim_elems",
            "bucket_members", "reduce_scatter", "all_gather", "repl_optim_elems",
        ],
    },
    EventSchema {
        kind: "weight_stream_counters",
        marker: "[weight-stream]",
        fields: &[
            "uploads", "evicts", "writeback", "registered", "ptr_moves",
            "pack_uploads", "pack_evicts", "prefetches", "async_wb",
            "h2d_bytes", "d2h_bytes",
        ],
    },
    EventSchema {
        kind: "weight_stream_residency",
        // KNOWN GAP: the one kind no gate can exercise — it needs cuda AND
        // registered > 0 (weight streaming engaged), which neither the CPU
        // gate nor the GPU cross-validation fixture reaches. Its value
        // agreement rests on the shared-snapshot construction in
        // nsl_weight_stream_count_atexit; if a streaming gate ever gains an
        // events file, cross-validate this kind there.
        marker: "[weight-stream]",
        fields: &["pinned", "registered", "pinned_bytes", "streamed_bytes", "summary"],
    },
    EventSchema {
        kind: "csla_counters",
        marker: "[csla]",
        fields: &["window_backward_phases"],
    },
    EventSchema {
        kind: "fase_fused_counters",
        marker: "[fase-fused]",
        fields: &[
            "rank", "fused_step_launches", "block_table_builds",
            "multi_batched", "multi_fallback",
        ],
    },
    EventSchema {
        kind: "wgrad_counters",
        marker: "[wgrad-accum]",
        fields: &["fused_gemm", "decomposed_fallback"],
    },
    EventSchema {
        kind: "kernel_launch_count",
        marker: "[nsl-kernel-count]",
        fields: &["count"],
    },
    EventSchema {
        kind: "gpu_launch_count",
        marker: "[nsl-gpu-launch-count]",
        fields: &["count"],
    },
    EventSchema {
        kind: "grad_integrity",
        marker: "[grad-integrity]",
        fields: &[
            "checks", "expected_params", "gradient_params", "finite", "nonzero",
            "missing", "notes_expected", "notes_observed_min",
            "notes_observed_max", "under_noted", "over_noted", "unjudged_checks",
        ],
    },
    EventSchema {
        kind: "gpu_mem_step",
        marker: "[gpu-mem]",
        fields: &[
            "driver_used_bytes", "driver_free_bytes", "driver_total_bytes",
            "allocated_bytes", "reserved_bytes", "live_blocks",
            "persistent_blocks", "drv_allocs", "drv_frees",
            "persistent_pool_bytes", "persistent_pool_segs",
            "transient_pool_bytes", "transient_pool_segs", "free_blocks",
            "cache_hits", "cache_misses", "splits", "coalesces", "surfaces",
            "external_async_bytes", "external_direct_bytes",
            "external_persistent_bytes", "external_count",
            "external_identified", "deferred_free_pending",
        ],
    },
];
