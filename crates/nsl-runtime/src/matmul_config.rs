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
use std::sync::OnceLock;

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
            cast_cache: false,
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

/// One deprecation notice per variable, on the first read that falls back.
fn warn_env_once(var: &str, flag: &str) {
    static SEEN: OnceLock<std::sync::Mutex<Vec<&'static str>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| std::sync::Mutex::new(Vec::new()));
    let mut g = seen.lock().unwrap();
    let leaked: &'static str = Box::leak(var.to_string().into_boxed_str());
    if g.iter().any(|v| *v == var) {
        return;
    }
    g.push(leaked);
    eprintln!(
        "[nsl-matmul] DEPRECATED: {var} is read as a fallback because this \
         program was built without {flag}. Environment values do NOT reach the \
         execution fingerprint, so a checkpoint written under them cannot say \
         which arithmetic produced it and a resume cannot refuse a silent \
         switch. Rebuild with {flag}."
    );
}

fn env_flag(var: &str, flag: &str, on_is: &str) -> bool {
    match std::env::var(var) {
        Ok(v) => {
            warn_env_once(var, flag);
            v == on_is
        }
        Err(_) => false,
    }
}

/// The effective configuration: compiled values if codegen set them, otherwise
/// the deprecated environment fallback, otherwise the defaults.
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
    let d = MatmulConfig::default();
    MatmulConfig {
        mode: if env_flag("NSL_MATMUL_BF16", "--matmul-mode bf16", "1") {
            MODE_BF16
        } else {
            d.mode
        },
        rounding: match std::env::var("NSL_MATMUL_BF16_ROUND") {
            Ok(v) => {
                warn_env_once("NSL_MATMUL_BF16_ROUND", "--bf16-rounding");
                if v == "sr" { ROUND_SR } else { ROUND_RNE }
            }
            Err(_) => d.rounding,
        },
        min_ratio: match std::env::var("NSL_MATMUL_BF16_MIN_RATIO") {
            Ok(v) => {
                warn_env_once("NSL_MATMUL_BF16_MIN_RATIO", "--bf16-min-ratio");
                v.parse::<f64>().ok().filter(|r| r.is_finite() && *r >= 0.0).unwrap_or(d.min_ratio)
            }
            Err(_) => d.min_ratio,
        },
        // Historically ON unless explicitly "0"; preserved so a script that
        // never set it keeps its behaviour.
        cast_cache: match std::env::var("NSL_MATMUL_BF16_CAST_CACHE") {
            Ok(v) => {
                warn_env_once("NSL_MATMUL_BF16_CAST_CACHE", "--bf16-cast-cache");
                v != "0"
            }
            Err(_) => d.cast_cache,
        },
        lt: env_flag("NSL_MATMUL_BF16_LT", "--bf16-lt", "1"),
        lt_workspace_mib: match std::env::var("NSL_MATMUL_BF16_LT_WORKSPACE_MIB") {
            Ok(v) => {
                warn_env_once("NSL_MATMUL_BF16_LT_WORKSPACE_MIB", "--bf16-lt-workspace-mib");
                v.parse::<u32>().unwrap_or(d.lt_workspace_mib).min(4096)
            }
            Err(_) => d.lt_workspace_mib,
        },
        lt_tune: match std::env::var("NSL_MATMUL_BF16_LT_TUNE") {
            Ok(v) => {
                warn_env_once("NSL_MATMUL_BF16_LT_TUNE", "--bf16-lt-tune");
                v != "0"
            }
            Err(_) => d.lt_tune,
        },
    }
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
