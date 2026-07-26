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
    /// Repo-relative file whose print macros emit it.
    pub emitted_by: &'static str,
    /// What its presence tells a test.
    pub means: &'static str,
}

const fn m(
    token: &'static str,
    emitted_by: &'static str,
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
        "crates/nsl-codegen/src/compiler/kernel.rs",
        "the WGGO planner ran and applied a plan",
    ),
    m(
        "[csla]",
        "crates/nsl-runtime/src/csla_stat.rs",
        "the CSLA window-buffered schedule executed",
    ),
    m(
        "[zero]",
        "crates/nsl-runtime/src/args.rs",
        "ZeRO stage 1/2 collectives ran",
    ),
    m(
        "[zero3]",
        "crates/nsl-runtime/src/zero.rs",
        "ZeRO stage 3 parameter sharding engaged",
    ),
    m(
        "[muon]",
        "crates/nsl-codegen/src/stmt.rs",
        "the Muon optimizer arm was lowered",
    ),
    m(
        "[muon-state]",
        "crates/nsl-runtime/src/sr_bf16.rs",
        "the bf16 Muon momentum envelope engaged",
    ),
    m(
        "[fuse]",
        "crates/nsl-codegen/src/stmt.rs",
        "a fusion peephole fired",
    ),
    m(
        "[csha]",
        "crates/nsl-codegen/src/csha.rs",
        "CSHA claimed an attention block",
    ),
    m(
        "[pca]",
        "crates/nsl-codegen/src/stmt.rs",
        "packed-context attention engaged",
    ),
    m(
        "[pca-per-doc]",
        "crates/nsl-codegen/src/compiler/kernel.rs",
        "PCA per-document admission ran",
    ),
    m(
        "[fase-multi]",
        "crates/nsl-runtime/src/fase_step.rs",
        "the multi-tensor FASE optimizer step fired",
    ),
    m(
        "[tape-ad]",
        "crates/nsl-runtime/src/autodiff/backward.rs",
        "the runtime tape backward ran (NOT source-AD)",
    ),
    m(
        "[sr-bf16]",
        "crates/nsl-runtime/src/sr_bf16.rs",
        "stochastic-rounding bf16 parameter storage engaged",
    ),
    m(
        "[flash-bwd]",
        "crates/nsl-runtime/src/flash_attention.rs",
        "the flash-attention backward ran",
    ),
    m(
        "[arena]",
        "crates/nsl-codegen/src/stmt.rs",
        "the transient-memory arena was used",
    ),
    m(
        "[gpu-mem]",
        "crates/nsl-runtime/src/tensor/mod.rs",
        "GPU memory accounting reported",
    ),
    m(
        "[nsl-gpu-launch-count]",
        "crates/nsl-runtime/src/fused_adapter.rs",
        "the fused-adapter GPU launch counter reported",
    ),
    m(
        "[nsl-kernel-count]",
        "crates/nsl-runtime/src/fused_adapter.rs",
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
    pub const MUON_STATE: &str = "[muon-state]";
    pub const FUSE: &str = "[fuse]";
    pub const CSHA: &str = "[csha]";
    pub const PCA: &str = "[pca]";
    pub const PCA_PER_DOC: &str = "[pca-per-doc]";
    pub const FASE_MULTI: &str = "[fase-multi]";
    pub const TAPE_AD: &str = "[tape-ad]";
    pub const SR_BF16: &str = "[sr-bf16]";
    pub const FLASH_BWD: &str = "[flash-bwd]";
    pub const ARENA: &str = "[arena]";
    pub const GPU_MEM: &str = "[gpu-mem]";
    pub const GPU_LAUNCH_COUNT: &str = "[nsl-gpu-launch-count]";
    pub const KERNEL_COUNT: &str = "[nsl-kernel-count]";
}
