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
        &["crates/nsl-codegen/src/pass_trace.rs"],
        "the registered compiler passes that actually ran this compile, in \
         first-invocation order (roadmap item 2)",
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
        &["crates/nsl-codegen/src/wengert_lower.rs"],
        "item 7 reported how many weight-gradient chains the pre-pass fused",
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
        "[nsl-kernel-count]",
        &[
            "crates/nsl-runtime/src/fused_adapter.rs",
        ],
        "the kernel counter reported",
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
    pub const TAPE_AD: &str = "[tape-ad]";
    pub const SR_BF16: &str = "[sr-bf16]";
    pub const PARAM_PLAN: &str = "[param-plan]";
    pub const PASS_TRACE: &str = "[pass-trace]";
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
        test: "crates/nsl-cli/tests/mse_leak_gate.rs",
        asserts: "the tape backward did not hit a FATAL disconnection",
        parts: &[("[tape-ad] FATAL", "crates/nsl-runtime/src/autodiff/backward.rs")],
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
        test: "crates/nsl-cli/tests/rmsnorm_residual_fold_gate.rs",
        asserts: "the rmsnorm dx+residual fold did NOT fire with the flag off",
        parts: &[(
            "[fuse] rmsnorm dx+residual folds:",
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
        test: "crates/nsl-cli/tests/multi_adamw_gate.rs",
        asserts: "the multi-tensor FASE step did NOT fire with the flag off",
        parts: &[("[fase-multi]", "crates/nsl-runtime/src/fase_step.rs")],
    },
];
