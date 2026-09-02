//! The matmul arithmetic configuration, set once at program start.
//!
//! # Why this exists
//!
//! Until 2026-09-02 this was five environment variables read lazily at first
//! GEMM. They were INVISIBLE to the execution fingerprint, with two consequences
//! that both cost real measurement time:
//!
//!  * A checkpoint could not tell you which arithmetic produced it, and a
//!    resume could not refuse a silent switch. Two legs of the 1B chain were
//!    written up as f32 while running bf16, because a copied driver carried a
//!    stale `NSL_MATMUL_BF16` and nothing read the banner back.
//!  * The fingerprint key that LOOKS like it covers this does not. `dtype` is
//!    the model dtype and defaults to `bf16` regardless of GEMM mode, so a
//!    sidecar reads `dtype=bf16` for a run whose matmuls were TF32. Two
//!    checkpoints taken on 2026-09-02 -- one banner-verified TF32, one
//!    banner-verified BF16 -- carry the identical `dtype=bf16`.
//!
//! Codegen now emits [`nsl_set_matmul_config`] before any user statement, from
//! compile options that are rendered into the fingerprint. The environment
//! variables remain as a deprecated fallback so existing scripts and gates keep
//! working; they warn once and are overridden by the compiled values.
//!
//! # What this can and cannot promise
//!
//! Matching `mmlt`/`mmltws`/`mmlttune` across a resume does NOT guarantee
//! bit-identical GEMMs, and the refusal message says so. Two independent
//! reasons, both live:
//!
//!  * cuBLASLt workspace allocation can fail and fall back to zero bytes
//!    (`lt_matmul.rs`, "continuing with zero workspace"), so a fuller card
//!    silently selects different kernels under an identical configuration.
//!  * Timed first-use plan selection (`mmlttune=1`, the default) picks a winner
//!    from live machine state, so plan choice is not reproducible across
//!    processes at all. `--deterministic` disables the tune, which is what makes
//!    the Lt path reproducible.
//!
//! A guard that overclaims is worse than no guard, so the configuration is
//! recorded as the INPUTS to kernel selection, not as a bit-identity promise.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

/// 0 = tf32, 1 = bf16, 2 = f32. Matches `nsl_codegen::MatmulMode`.
pub const MODE_TF32: i64 = 0;
pub const MODE_BF16: i64 = 1;
pub const MODE_F32: i64 = 2;

/// 0 = RNE, 1 = SR. Matches `nsl_codegen::Bf16Rounding`.
pub const ROUND_RNE: i64 = 0;
pub const ROUND_SR: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatmulConfig {
    pub mode: i64,
    pub rounding: i64,
    pub min_ratio: f64,
    pub cast_cache: bool,
    pub lt: bool,
    pub lt_workspace_mib: u32,
    pub lt_tune: bool,
}

impl Default for MatmulConfig {
    fn default() -> Self {
        Self {
            mode: MODE_TF32,
            rounding: ROUND_RNE,
            min_ratio: 512.0,
            // ON, as it was before #583. The pre-#583 read was
            // `var != Some("0")`, which is TRUE when the variable is unset.
            cast_cache: true,
            lt: false,
            lt_workspace_mib: 64,
            lt_tune: true,
        }
    }
}

static SET: AtomicBool = AtomicBool::new(false);
static MODE: AtomicU64 = AtomicU64::new(0);
static ROUNDING: AtomicU64 = AtomicU64::new(0);
static MIN_RATIO: AtomicU64 = AtomicU64::new(0);
static CAST_CACHE: AtomicBool = AtomicBool::new(false);
static LT: AtomicBool = AtomicBool::new(false);
static LT_WS: AtomicU32 = AtomicU32::new(64);
static LT_TUNE: AtomicBool = AtomicBool::new(true);

/// Called by codegen before any user statement runs.
///
/// Values arrive already CLAMPED by `CompileOptions::matmul.clamped()`, which
/// applies the same bounds the environment path applies -- a fingerprint that
/// recorded a raw value the runtime then clamped would let two runs with
/// identical arithmetic disagree.
///
/// # Safety
/// Plain scalars; no pointers are dereferenced.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_set_matmul_config(
    mode: i64,
    rounding: i64,
    min_ratio: f64,
    cast_cache: i64,
    lt: i64,
    lt_workspace_mib: i64,
    lt_tune: i64,
) -> i64 {
    MODE.store(mode as u64, Ordering::SeqCst);
    ROUNDING.store(rounding as u64, Ordering::SeqCst);
    MIN_RATIO.store(min_ratio.to_bits(), Ordering::SeqCst);
    CAST_CACHE.store(cast_cache != 0, Ordering::SeqCst);
    LT.store(lt != 0, Ordering::SeqCst);
    LT_WS.store(lt_workspace_mib.clamp(0, 4096) as u32, Ordering::SeqCst);
    LT_TUNE.store(lt_tune != 0, Ordering::SeqCst);
    SET.store(true, Ordering::SeqCst);
    0
}

/// The effective configuration.
///
/// # Why there is no environment fallback here any more
///
/// #583 put one here and it was wrong twice over.
///
///  * **It was unreachable.** Codegen emits `nsl_set_matmul_config`
///    unconditionally, so `SET` is always true for a compiled NSL program and
///    the fallback never ran. `NSL_MATMUL_BF16=1` silently produced TF32 --
///    the exact mislabeled-arm failure the feature exists to prevent.
///  * **Where it DID run** (a C host or a test driving the runtime directly) it
///    ran per call. `config()` is called inside math-mode selection, i.e. per
///    GEMM, and the fallback did seven `std::env::var` lookups and leaked a
///    `String` on every one of them.
///
/// The environment is now resolved ONCE, at CLI parse time, into
/// `CompileOptions::matmul` (see `nsl_codegen::MatmulConfig::with_env_fallback`).
/// That is strictly better than the #583 design: an env-driven run now reaches
/// the execution fingerprint too, instead of being invisible to it.
///
/// So this function is a pure read of process-global state with no I/O, which
/// is what a per-GEMM call has to be.
pub fn config() -> MatmulConfig {
    if SET.load(Ordering::SeqCst) {
        return MatmulConfig {
            mode: MODE.load(Ordering::SeqCst) as i64,
            rounding: ROUNDING.load(Ordering::SeqCst) as i64,
            min_ratio: f64::from_bits(MIN_RATIO.load(Ordering::SeqCst)),
            cast_cache: CAST_CACHE.load(Ordering::SeqCst),
            lt: LT.load(Ordering::SeqCst),
            lt_workspace_mib: LT_WS.load(Ordering::SeqCst),
            lt_tune: LT_TUNE.load(Ordering::SeqCst),
        };
    }
    MatmulConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiled_values_win_over_the_environment() {
        // Set the env to bf16 and then compile in tf32: the compiled value must
        // win, because the whole point is that the fingerprint and the runtime
        // agree. An env var that could override a compiled flag would put them
        // back out of sync.
        nsl_set_matmul_config(MODE_F32, ROUND_SR, 256.0, 1, 1, 128, 0);
        let c = config();
        assert_eq!(c.mode, MODE_F32);
        assert_eq!(c.rounding, ROUND_SR);
        assert_eq!(c.min_ratio, 256.0);
        assert!(c.cast_cache && c.lt && !c.lt_tune);
        assert_eq!(c.lt_workspace_mib, 128);
    }

    #[test]
    fn workspace_is_clamped_at_the_sink_too() {
        // The compile side clamps, but a hand-written caller (a C host, a test)
        // could pass anything. Clamp here as well so the runtime can never use
        // a value the fingerprint could not have recorded.
        nsl_set_matmul_config(MODE_BF16, ROUND_RNE, 512.0, 0, 1, 999_999, 1);
        assert_eq!(config().lt_workspace_mib, 4096);
        nsl_set_matmul_config(MODE_BF16, ROUND_RNE, 512.0, 0, 1, -5, 1);
        assert_eq!(config().lt_workspace_mib, 0);
    }

    #[test]
    fn defaults_are_the_historical_ones() {
        let d = MatmulConfig::default();
        assert_eq!(d.mode, MODE_TF32, "tf32 was the default matmul mode");
        assert_eq!(d.rounding, ROUND_RNE);
        assert_eq!(d.min_ratio, 512.0);
        assert!(d.lt_tune, "timed Lt selection defaulted ON");
        assert_eq!(d.lt_workspace_mib, 64);
    }
}
