//! The `--matmul-mode` / `--bf16-*` flag group, shared by `nsl run` and
//! `nsl build`.
//!
//! Both subcommands used to declare these seven flags separately and build a
//! `MatmulConfig` from them with byte-identical code. They are flattened here
//! so the two cannot drift, and so ONE place decides what "the user did not
//! pass anything" means.
//!
//! ## Why the cast-cache flag is spelled negatively
//!
//! `MatmulConfig::default()` has `bf16_cast_cache: true` (it was on before
//! #583, whose runtime read was `var != Some("0")` -- true when unset). A bare
//! `#[arg(long)] bool` in clap defaults to FALSE, so the affirmative spelling
//! `--bf16-cast-cache` could not express that default: every invocation that
//! did not pass the flag handed `false` to `MatmulConfig`, silently disabling
//! the cache for the whole product while `MatmulConfig::default()` -- and the
//! unit test guarding it -- still read `true`.
//!
//! That also killed the environment fallback. `with_env_fallback` takes a
//! field from the environment only when it still equals the library default,
//! which is how an explicit flag wins. With clap writing `false` over a `true`
//! default the sentinel never held, so `NSL_MATMUL_BF16_CAST_CACHE` was inert:
//! no effect and no deprecation notice, while its six siblings warned.
//!
//! So a default-ON option MUST be a `--no-*` flag, matching `--no-bf16-lt-tune`
//! below. `cli_defaults_match_the_library_defaults` pins the general rule.

use nsl_codegen::{Bf16Rounding, MatmulConfig, MatmulMode};

/// The matmul arithmetic flag group.
#[derive(clap::Args, Debug, Clone)]
pub(crate) struct MatmulArgs {
    /// Matmul arithmetic for high-intensity GEMMs: tf32 (default), bf16, f32.
    ///
    /// Replaces NSL_MATMUL_BF16. Unlike the environment variable this reaches
    /// the EXECUTION FINGERPRINT, so a checkpoint records which arithmetic
    /// produced it and a resume refuses a silent switch. The fingerprint's
    /// `dtype` key does NOT carry this: it is the model dtype and reads
    /// `bf16` whatever the GEMMs actually do.
    #[arg(long, value_name = "MODE", default_value = "tf32")]
    pub(crate) matmul_mode: String,

    /// Rounding for the bf16 operand cast: rne (default) or sr.
    /// Replaces NSL_MATMUL_BF16_ROUND. SR re-dithers per launch, which blocks
    /// CUDA graph capture and is incompatible with the bf16 cast cache.
    #[arg(long, value_name = "MODE", default_value = "rne")]
    pub(crate) bf16_rounding: String,

    /// Minimum arithmetic intensity mnk/(a+b elements) for a GEMM to take the
    /// bf16 path. Replaces NSL_MATMUL_BF16_MIN_RATIO. ARITHMETIC: it decides
    /// WHICH matmuls are reduced precision.
    #[arg(long, value_name = "RATIO", default_value_t = 512.0)]
    pub(crate) bf16_min_ratio: f64,

    /// Do NOT cache the weight operand's bf16 cast across GEMMs. The cache is
    /// ON by default. Replaces NSL_MATMUL_BF16_CAST_CACHE=0.
    ///
    /// The cache is bit-preserving, so a resume only warns -- but it costs
    /// ~2 GiB of pinned VRAM at 1B, so this switch is the way to fit a run
    /// that would otherwise OOM.
    #[arg(long)]
    pub(crate) no_bf16_cast_cache: bool,

    /// Issue bf16-storage GEMMs through cuBLASLt heuristics rather than
    /// GemmEx. Replaces NSL_MATMUL_BF16_LT. Changes kernel and reduction
    /// order, so it is arithmetic-class for resume.
    #[arg(long)]
    pub(crate) bf16_lt: bool,

    /// Workspace cap in MiB for the cuBLASLt heuristic (clamped to 4096).
    /// Replaces NSL_MATMUL_BF16_LT_WORKSPACE_MIB. ARITHMETIC: the cap FILTERS
    /// candidate kernels, so a smaller value excludes split-k and wide-tile
    /// algorithms and changes the reduction order.
    #[arg(long, value_name = "MIB", default_value_t = 64)]
    pub(crate) bf16_lt_workspace_mib: u32,

    /// Disable cuBLASLt timed first-use plan selection. Replaces
    /// NSL_MATMUL_BF16_LT_TUNE=0. With tuning ON (the default) the winner
    /// depends on live machine state, so plan choice is NOT reproducible
    /// across processes.
    #[arg(long)]
    pub(crate) no_bf16_lt_tune: bool,
}

impl MatmulArgs {
    /// The `MatmulConfig` these flags describe, with the deprecated
    /// `NSL_MATMUL_BF16*` variables filling in anything left at its default so
    /// an env-driven run still reaches the fingerprint, and `.clamped()`
    /// applying the runtime's bounds so the fingerprint records the EFFECTIVE
    /// value rather than the raw one.
    pub(crate) fn to_config(&self) -> MatmulConfig {
        MatmulConfig {
            mode: MatmulMode::parse(&self.matmul_mode).unwrap_or(MatmulMode::Tf32),
            bf16_rounding: Bf16Rounding::parse(&self.bf16_rounding).unwrap_or(Bf16Rounding::Rne),
            bf16_min_ratio: self.bf16_min_ratio,
            bf16_cast_cache: !self.no_bf16_cast_cache,
            bf16_lt: self.bf16_lt,
            bf16_lt_workspace_mib: self.bf16_lt_workspace_mib,
            bf16_lt_tune: !self.no_bf16_lt_tune,
        }
        .with_env_fallback()
        .clamped()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// `MatmulArgs` is only ever reached through a subcommand, so give it a
    /// parseable shell for the tests.
    #[derive(Parser, Debug)]
    struct Harness {
        #[command(flatten)]
        matmul: MatmulArgs,
    }

    fn parse(extra: &[&str]) -> MatmulArgs {
        let mut argv = vec!["nsl"];
        argv.extend_from_slice(extra);
        Harness::try_parse_from(argv).expect("flags parse").matmul
    }

    /// THE STRUCTURAL GUARD.
    ///
    /// Every flag's clap default must describe the same arithmetic as
    /// `MatmulConfig::default()`. When they disagree the product silently runs
    /// something other than its documented default, and -- because
    /// `with_env_fallback` uses "still equals the default" as its sentinel for
    /// "the user did not pass this flag" -- the matching environment variable
    /// goes inert without warning.
    ///
    /// That is not hypothetical: `--bf16-cast-cache` was an affirmative
    /// `#[arg(long)] bool` (clap default FALSE) against a library default of
    /// TRUE. Every `nsl run` disabled the weight-cast cache, and
    /// `NSL_MATMUL_BF16_CAST_CACHE` did nothing at all. Both unit tests that
    /// were supposed to cover it passed, because both asked
    /// `MatmulConfig::default()` -- the layer that was correct -- instead of
    /// the layer that ships.
    ///
    /// Asserted field-by-field rather than with one `assert_eq!` so a failure
    /// names the flag.
    #[test]
    fn cli_defaults_match_the_library_defaults() {
        let got = parse(&[]).to_config();
        let want = MatmulConfig::default();

        assert_eq!(got.mode, want.mode, "--matmul-mode");
        assert_eq!(got.bf16_rounding, want.bf16_rounding, "--bf16-rounding");
        assert_eq!(got.bf16_min_ratio, want.bf16_min_ratio, "--bf16-min-ratio");
        assert_eq!(got.bf16_cast_cache, want.bf16_cast_cache, "--no-bf16-cast-cache");
        assert_eq!(got.bf16_lt, want.bf16_lt, "--bf16-lt");
        assert_eq!(
            got.bf16_lt_workspace_mib, want.bf16_lt_workspace_mib,
            "--bf16-lt-workspace-mib"
        );
        assert_eq!(got.bf16_lt_tune, want.bf16_lt_tune, "--no-bf16-lt-tune");
        assert_eq!(got, want, "the flag group as a whole");
    }

    /// The cache is on when nothing is passed, and the switch turns it off.
    /// The second half is the anti-vacuity side: a `to_config` that hard-coded
    /// `true` would satisfy the test above.
    #[test]
    fn the_cast_cache_is_on_by_default_and_the_switch_turns_it_off() {
        assert!(
            parse(&[]).to_config().bf16_cast_cache,
            "the weight-cast cache is ON unless asked otherwise"
        );
        assert!(
            !parse(&["--no-bf16-cast-cache"]).to_config().bf16_cast_cache,
            "--no-bf16-cast-cache must actually disable it"
        );
    }

    /// The affirmative flags still work in the direction they read.
    #[test]
    fn the_affirmative_flags_still_select_what_they_name() {
        let c = parse(&["--matmul-mode", "bf16", "--bf16-lt", "--no-bf16-lt-tune"]).to_config();
        assert_eq!(c.mode, MatmulMode::Bf16);
        assert!(c.bf16_lt);
        assert!(!c.bf16_lt_tune);
    }
}
