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
//! That also killed the environment fallback, which then took a field from
//! the environment only when it still equalled the library default. With
//! clap writing `false` over a `true` default the sentinel never held, so
//! `NSL_MATMUL_BF16_CAST_CACHE` was inert: no effect and no deprecation
//! notice, while its six siblings warned.
//!
//! ## Why the valued flags are `Option`s
//!
//! "Still equals the default" cannot tell `--matmul-mode tf32` from no flag
//! at all, so an inherited `NSL_MATMUL_BF16=1` beat an explicit tf32
//! (external review 2026-10-06). A valued flag is now `None` when omitted and
//! its default is applied in `to_config`; `MatmulExplicit` records which
//! fields the user set, and only the others take an environment fallback.
//!
//! So a default-ON option MUST be a `--no-*` flag, matching `--no-bf16-lt-tune`
//! below. `cli_defaults_match_the_library_defaults` pins the general rule.

use nsl_codegen::{Bf16Rounding, MatmulConfig, MatmulExplicit, MatmulMode};

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
    /// An explicit value also wins over the runtime's NSL_MATMUL_TF32 /
    /// NSL_MATMUL_PEDANTIC overrides.
    #[arg(long, value_name = "MODE", value_parser = ["tf32", "bf16", "f32"])]
    pub(crate) matmul_mode: Option<String>,

    /// Rounding for the bf16 operand cast: rne (default) or sr.
    /// Replaces NSL_MATMUL_BF16_ROUND. SR re-dithers per launch, which blocks
    /// CUDA graph capture and is incompatible with the bf16 cast cache.
    #[arg(long, value_name = "MODE", value_parser = ["rne", "sr"])]
    pub(crate) bf16_rounding: Option<String>,

    /// Minimum arithmetic intensity mnk/(a+b elements) for a GEMM to take the
    /// bf16 path. Replaces NSL_MATMUL_BF16_MIN_RATIO. ARITHMETIC: it decides
    /// WHICH matmuls are reduced precision. Default 512.
    #[arg(long, value_name = "RATIO")]
    pub(crate) bf16_min_ratio: Option<f64>,

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
    /// algorithms and changes the reduction order. Default 64.
    #[arg(long, value_name = "MIB")]
    pub(crate) bf16_lt_workspace_mib: Option<u32>,

    /// Disable cuBLASLt timed first-use plan selection. Replaces
    /// NSL_MATMUL_BF16_LT_TUNE=0. With tuning ON (the default) the winner
    /// depends on live machine state, so plan choice is NOT reproducible
    /// across processes.
    #[arg(long)]
    pub(crate) no_bf16_lt_tune: bool,
}

impl MatmulArgs {
    /// The `MatmulConfig` these flags describe, with the deprecated
    /// `NSL_MATMUL_BF16*` variables filling in any field the user did NOT set
    /// so an env-driven run still reaches the fingerprint, and `.clamped()`
    /// applying the runtime's bounds so the fingerprint records the EFFECTIVE
    /// value rather than the raw one.
    pub(crate) fn to_config(&self) -> MatmulConfig {
        let (config, explicit) = self.flags();
        config.with_env_fallback(explicit).clamped()
    }

    /// `to_config` over an injected environment lookup, for tests: the
    /// process environment is shared with every other test thread.
    #[cfg(test)]
    pub(crate) fn to_config_with(&self, env: impl Fn(&str) -> Option<String>) -> MatmulConfig {
        let (config, explicit) = self.flags();
        config.with_env_lookup(explicit, env).clamped()
    }

    /// The flags as a config (defaults for omitted ones) and which were set.
    fn flags(&self) -> (MatmulConfig, MatmulExplicit) {
        let d = MatmulConfig::default();
        let explicit = MatmulExplicit {
            mode: self.matmul_mode.is_some(),
            bf16_rounding: self.bf16_rounding.is_some(),
            bf16_min_ratio: self.bf16_min_ratio.is_some(),
            bf16_cast_cache: self.no_bf16_cast_cache,
            bf16_lt: self.bf16_lt,
            bf16_lt_workspace_mib: self.bf16_lt_workspace_mib.is_some(),
            bf16_lt_tune: self.no_bf16_lt_tune,
        };
        // clap's value_parser admits only names these parse, so the defaults
        // below are reached only for an omitted flag.
        let config = MatmulConfig {
            mode: self.matmul_mode.as_deref().and_then(MatmulMode::parse).unwrap_or(d.mode),
            bf16_rounding: self
                .bf16_rounding
                .as_deref()
                .and_then(Bf16Rounding::parse)
                .unwrap_or(d.bf16_rounding),
            bf16_min_ratio: self.bf16_min_ratio.unwrap_or(d.bf16_min_ratio),
            bf16_cast_cache: !self.no_bf16_cast_cache,
            bf16_lt: self.bf16_lt,
            bf16_lt_workspace_mib: self.bf16_lt_workspace_mib.unwrap_or(d.bf16_lt_workspace_mib),
            bf16_lt_tune: !self.no_bf16_lt_tune,
            mode_explicit: false,
        };
        (config, explicit)
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
    /// something other than its documented default (and, under the old
    /// "still equals the default" sentinel, the matching environment
    /// variable went inert without warning).
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
        let got = parse(&[]).to_config_with(|_| None);
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
            parse(&[]).to_config_with(|_| None).bf16_cast_cache,
            "the weight-cast cache is ON unless asked otherwise"
        );
        assert!(
            !parse(&["--no-bf16-cast-cache"]).to_config_with(|_| None).bf16_cast_cache,
            "--no-bf16-cast-cache must actually disable it"
        );
    }

    /// External review 2026-10-06: an explicit flag that EQUALS the default
    /// beats an inherited variable. Under the old "still equals the default"
    /// rule, `--matmul-mode tf32` with `NSL_MATMUL_BF16=1` ran bf16.
    #[test]
    fn an_explicit_default_beats_an_inherited_variable() {
        let env = |name: &str| match name {
            "NSL_MATMUL_BF16" => Some("1".to_string()),
            "NSL_MATMUL_BF16_ROUND" => Some("sr".to_string()),
            "NSL_MATMUL_BF16_MIN_RATIO" => Some("8".to_string()),
            "NSL_MATMUL_BF16_LT_WORKSPACE_MIB" => Some("16".to_string()),
            _ => None,
        };
        let explicit = parse(&[
            "--matmul-mode",
            "tf32",
            "--bf16-rounding",
            "rne",
            "--bf16-min-ratio",
            "512",
            "--bf16-lt-workspace-mib",
            "64",
        ])
        .to_config_with(env);
        assert_eq!(explicit.mode, MatmulMode::Tf32, "--matmul-mode tf32");
        assert!(explicit.mode_explicit, "the runtime must learn the mode was explicit");
        assert_eq!(explicit.bf16_rounding, Bf16Rounding::Rne, "--bf16-rounding rne");
        assert_eq!(explicit.bf16_min_ratio, 512.0, "--bf16-min-ratio 512");
        assert_eq!(explicit.bf16_lt_workspace_mib, 64, "--bf16-lt-workspace-mib 64");

        // The anti-vacuity half: with the flags omitted, the variables apply.
        let omitted = parse(&[]).to_config_with(env);
        assert_eq!(omitted.mode, MatmulMode::Bf16);
        assert!(!omitted.mode_explicit);
        assert_eq!(omitted.bf16_rounding, Bf16Rounding::Sr);
        assert_eq!(omitted.bf16_min_ratio, 8.0);
        assert_eq!(omitted.bf16_lt_workspace_mib, 16);
    }

    /// The explicit-mode bit is defined twice -- codegen emits it, the
    /// runtime masks it off -- and nothing else ties the copies together.
    #[test]
    fn the_explicit_mode_bit_agrees_across_codegen_and_runtime() {
        assert_eq!(nsl_codegen::MATMUL_MODE_EXPLICIT, nsl_runtime::matmul_config::MODE_EXPLICIT);
        // ...and it collides with no mode value.
        for mode in [
            nsl_runtime::matmul_config::MODE_TF32,
            nsl_runtime::matmul_config::MODE_BF16,
            nsl_runtime::matmul_config::MODE_F32,
        ] {
            assert_eq!(mode & nsl_codegen::MATMUL_MODE_EXPLICIT, 0);
        }
    }

    /// A typo is refused, not silently read as the default (`tf23` used to
    /// become tf32 -- and then pick up an inherited variable).
    #[test]
    fn a_misspelled_mode_is_refused() {
        let mut argv = vec!["nsl", "--matmul-mode", "tf23"];
        assert!(Harness::try_parse_from(argv.clone()).is_err(), "--matmul-mode tf23");
        argv[1] = "--bf16-rounding";
        argv[2] = "nearest";
        assert!(Harness::try_parse_from(argv).is_err(), "--bf16-rounding nearest");
    }

    /// The affirmative flags still work in the direction they read.
    #[test]
    fn the_affirmative_flags_still_select_what_they_name() {
        let c = parse(&["--matmul-mode", "bf16", "--bf16-lt", "--no-bf16-lt-tune"]).to_config_with(|_| None);
        assert_eq!(c.mode, MatmulMode::Bf16);
        assert!(c.bf16_lt);
        assert!(!c.bf16_lt_tune);
    }
}
