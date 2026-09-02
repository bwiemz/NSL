//! @autotune: build-time kernel variant benchmarking with caching.
//!
//! Generates Cartesian products of tuning parameters, hashes kernel ASTs
//! for cache key generation, and provides read/write for the `.nsl-cache/autotune/`
//! directory.
//!
//! # What actually runs today
//!
//! The production COMPILE path (`compiler::kernel::autotune_select_best`)
//! calls [`find_best_variant_cost_model`], which **estimates** each variant
//! with the roofline model. A compile never launches anything — that
//! invariant is deliberate (no unbounded runtime autotuner) and is pinned by
//! `autotune_selection_method_is_honest` in `tests/autotune_cache_identity.rs`.
//!
//! [`find_best_variant`] — the real-measurement path — is wired to exactly
//! ONE production caller (roadmap item 10): [`measure_variants`], reached via
//! the offline `nsl autotune <file.nsl>` command. It times each variant with
//! CUDA events over synthesized buffers at a representative element count
//! and writes a [`SelectionMethod::Measured`] cache record. Because
//! [`load_cache_record`]'s cost-model-spec invalidation applies only to
//! `CostModel` records, a Measured record for this (device, key) then
//! short-circuits the cost model at every subsequent compile — tune once,
//! every later build uses the measured winner. The same honesty gate now
//! pins this wiring in both directions: the offline path is the ONLY caller,
//! and the compile path still never measures.
//!
//! When `NSL_AUTOTUNE_FALLBACK=1` is set, `select_middle_values()` is used
//! instead of either path.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

/// A list of tuning parameters, each with a name and a set of candidate values.
pub type TuningParams = Vec<(String, Vec<i64>)>;

/// A single variant: one concrete value chosen for each tuning parameter.
pub type Variant = Vec<(String, i64)>;

/// Result of an autotune benchmarking run.
pub struct AutotuneResult {
    pub winner: Variant,
    pub all_timings: Vec<(Variant, f64)>,
    /// Which machine this result describes.
    pub device: DeviceIdentity,
    /// How the winner was chosen — estimated or measured. Kept separate from
    /// `device` on purpose: the pre-item-10 record stuffed the literal string
    /// `"cost_model"` into its device-name field, so a reader could not tell an
    /// unmeasured estimate from a result for a machine named "cost_model".
    pub selection: SelectionMethod,
}

/// Version of the cache key + on-disk record format.
///
/// Bump on ANY change to what [`hash_kernel_ast`] absorbs or to
/// [`AutotuneCacheRecord`]'s shape. It is hashed into the key, so a bump
/// invalidates every existing entry rather than risking a stale entry being
/// read back under new semantics.
///
/// - v1: implicit; device identity came from `gpu_specs::default_gpu()`, which
///   returns A100-SXM unconditionally. Every machine produced the same key.
/// - v2: driver-reported device identity, length-prefixed hash fields,
///   serde record.
/// - v3: symbol-RESOLVED AST bytes (`resolve_symbols_for_hash`). v2 hashed
///   raw `Debug` output whose `SymbolU32` indices depend on everything the
///   process interned first, so `nsl autotune` and `nsl build` produced
///   different keys for the identical kernel and measured records were
///   never consumed.
pub const CACHE_SCHEMA_VERSION: u32 = 3;

/// How a winning variant was chosen.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SelectionMethod {
    /// Roofline estimate. Nothing was launched. `spec` names the `GpuSpec` the
    /// roofline was priced against — NOT necessarily the local device, because
    /// the GPU database may have no entry for it.
    CostModel { spec: String },
    /// Median of real CUDA-event timings on the device recorded alongside.
    Measured,
}

/// The machine a tuning result describes.
///
/// Every field comes from the CUDA driver, never from `gpu_specs`. See
/// [`DeviceIdentity::local`] for why that distinction is the whole point.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceIdentity {
    pub device_name: String,
    /// Compute capability major*10 + minor, e.g. 120 for sm_120.
    pub sm_version: u32,
    pub sm_count: u32,
    /// `cuDriverGetVersion`, e.g. 13030 for CUDA 13.3. A driver upgrade can
    /// change ptxas codegen and therefore which variant wins, so it is part of
    /// the identity rather than metadata.
    pub driver_version: u32,
}

/// Device name recorded when the compiler CAN see GPUs but found none.
///
/// A *named* sentinel, not a fallback to some real GPU's spec: a cache built on
/// a GPU-less machine must not be indistinguishable from one built on an A100.
pub const NO_DEVICE_NAME: &str = "no-cuda-device";

/// Device name recorded when the compiler was built without CUDA support.
///
/// Distinct from [`NO_DEVICE_NAME`] because the two are different claims and
/// only one of them is about the hardware. `cuda` is not a default feature, so
/// the stock `nsl` binary takes this path **even on a machine with a GPU in
/// it** — writing "no-cuda-device" there would be a false statement about the
/// box, and would let a no-CUDA build's entries and a real probe's entries be
/// confused if the probe ever failed.
pub const NO_CUDA_SUPPORT_NAME: &str = "cuda-unsupported-build";

impl DeviceIdentity {
    /// The local device's identity, or a sentinel naming why there isn't one.
    ///
    /// This is the function that fixes item 10's core defect. The cache key was
    /// previously built from `gpu_specs::default_gpu()`, whose contract is "the
    /// default when auto-detect is unavailable" — it returns A100-SXM
    /// unconditionally. Hashing that meant the key said "A100-SXM / sm_80 / 108
    /// SMs" on every machine, so an entry measured on one GPU was reused
    /// verbatim on any other.
    ///
    /// It must never consult `gpu_specs`' database for identity, in any branch.
    /// `local_identity_never_reads_the_gpu_database` pins that, because the
    /// natural-looking regression — "fall back to the default spec when the
    /// probe fails" — restores the original bug exactly.
    pub fn local() -> Self {
        match crate::gpu_specs::local_device_identity() {
            Some(d) => Self {
                device_name: d.name.clone(),
                sm_version: d.sm_version,
                sm_count: d.sm_count,
                driver_version: d.driver_version,
            },
            None if !nsl_runtime::CUDA_SUPPORT_COMPILED => Self::no_cuda_support(),
            None => Self::no_device(),
        }
    }

    /// The sentinel identity for "no CUDA device was found".
    pub fn no_device() -> Self {
        Self::sentinel(NO_DEVICE_NAME)
    }

    /// The sentinel identity for "this compiler has no CUDA support".
    pub fn no_cuda_support() -> Self {
        Self::sentinel(NO_CUDA_SUPPORT_NAME)
    }

    /// Whether this identity describes real probed silicon.
    pub fn is_real_device(&self) -> bool {
        self.device_name != NO_DEVICE_NAME && self.device_name != NO_CUDA_SUPPORT_NAME
    }

    fn sentinel(name: &str) -> Self {
        Self {
            device_name: name.to_string(),
            sm_version: 0,
            sm_count: 0,
            driver_version: 0,
        }
    }

    /// Human-readable one-liner for diagnostics.
    pub fn describe(&self) -> String {
        format!(
            "{} (sm_{}, {} SMs, driver {})",
            self.device_name, self.sm_version, self.sm_count, self.driver_version
        )
    }
}

/// One measured or estimated variant, as stored on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimingEntry {
    pub params: Vec<(String, i64)>,
    pub median_ms: f64,
}

/// The on-disk shape of a `.nsl-cache/autotune/<kernel>_<hash>.json` entry.
///
/// Serde-derived rather than built with `format!` and read back with substring
/// arithmetic: the previous writer emitted JSON by string concatenation and the
/// reader recovered the winner by locating `"winner":`, then the next `{`, then
/// the next `}`. Neither side validated anything — not the schema, not the
/// device, not even that the file's own recorded key matched the key that was
/// looked up.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AutotuneCacheRecord {
    pub schema_version: u32,
    pub kernel: String,
    pub cache_key: String,
    pub device: DeviceIdentity,
    pub selection: SelectionMethod,
    pub winner: Vec<(String, i64)>,
    pub winner_ms: f64,
    pub variants_tested: usize,
    pub all_timings: Vec<TimingEntry>,
    pub timestamp_secs: u64,
}

/// Why a cache entry on disk was not usable.
#[derive(Debug, Clone, PartialEq)]
pub enum CacheReject {
    /// Not valid JSON for the current record type — including every pre-item-10
    /// entry, which had no `schema_version` field at all.
    Unparseable(String),
    /// Written by a different schema version.
    SchemaMismatch { found: u32, expected: u32 },
    /// The record's own `cache_key` is not the key it was looked up under, so
    /// the file was renamed or copied from elsewhere.
    KeyMismatch { found: String, expected: String },
    /// Measured on different hardware. This is the failure item 10 exists to
    /// prevent, so it is the one that always warns.
    DeviceMismatch {
        found: Box<DeviceIdentity>,
        expected: Box<DeviceIdentity>,
    },
    /// The roofline was priced against a different `GpuSpec` than this build
    /// would use, so the winner it chose no longer follows from the model.
    CostModelSpecChanged { found: String, expected: String },
    /// Parsed and validated, but carries no winner to return.
    EmptyWinner,
}

/// First 16 characters of a cache key, for diagnostics.
///
/// Character-wise, not byte-wise. `found` comes from a JSON file on disk, whose
/// whole threat model is that it was "copied, committed, restored from a CI
/// artifact, or shared over a network mount" — so it is arbitrary text, and
/// `&s[..16]` panics when byte 16 lands mid-codepoint. That panic ran inside
/// the compiler, aborting a build over a malformed cache file.
fn abbreviate_key(key: &str) -> String {
    key.chars().take(16).collect()
}

impl CacheReject {
    /// Explain the rejection for a user-facing warning.
    pub fn describe(&self) -> String {
        match self {
            Self::Unparseable(e) => {
                format!("not readable as a v{CACHE_SCHEMA_VERSION} record ({e})")
            }
            Self::SchemaMismatch { found, expected } => {
                format!("written by schema v{found}, this build reads v{expected}")
            }
            Self::KeyMismatch { found, expected } => format!(
                "records cache key {} but was looked up under {} — copied or renamed file",
                abbreviate_key(found),
                abbreviate_key(expected)
            ),
            Self::CostModelSpecChanged { found, expected } => format!(
                "was priced against the {found} cost model, but this build prices against {expected}"
            ),
            Self::DeviceMismatch { found, expected } => format!(
                "describes {} but this machine is {}",
                found.describe(),
                expected.describe()
            ),
            Self::EmptyWinner => "contains no winning variant".to_string(),
        }
    }

    /// Whether this rejection deserves an unconditional warning.
    ///
    /// A schema bump or an unparseable pre-item-10 entry is the expected state
    /// of a cache directory right after an upgrade — warning per kernel there
    /// would be noise that trains users to ignore the channel. A device or key
    /// mismatch means an entry from another machine is sitting in this cache,
    /// which is exactly what nobody would otherwise notice.
    pub fn is_noteworthy(&self) -> bool {
        match self {
            // Expected right after an upgrade or a schema bump. Warning per
            // kernel here is noise that trains users to ignore the channel.
            Self::Unparseable(_) | Self::SchemaMismatch { .. } => false,
            // An entry from another machine, or a file that is not what its
            // name says: nobody would otherwise notice these.
            Self::DeviceMismatch { .. } | Self::KeyMismatch { .. } => true,
            // Recoverable but worth saying — the recorded winner no longer
            // follows from the model this build would use.
            Self::CostModelSpecChanged { .. } => true,
            // Should be unwritable (see `build_cache_record`); if one appears,
            // something upstream is producing empty winners.
            Self::EmptyWinner => true,
        }
    }
}

/// Generate the Cartesian product of all tuning parameter combinations.
pub fn cartesian_product(params: &TuningParams) -> Vec<Variant> {
    let mut result: Vec<Variant> = vec![vec![]];
    for (name, values) in params {
        let mut new_result = Vec::new();
        for existing in &result {
            for &val in values {
                let mut combo = existing.clone();
                combo.push((name.clone(), val));
                new_result.push(combo);
            }
        }
        result = new_result;
    }
    result
}

/// Select the middle value from each parameter range (`--no-autotune` fallback).
///
/// Parameters with an empty value list are skipped rather than asserted on. The
/// semantic checker now rejects `@autotune(x=[])` at the source level, which it
/// did NOT do when this function's doc first claimed it did — so the assertion
/// that used to be here was reachable from ordinary NSL and aborted the whole
/// compiler. Belt and braces: refusing in the checker gives a real diagnostic
/// with a span, and skipping here means a caller constructing `TuningParams`
/// programmatically cannot panic the compiler either.
pub fn select_middle_values(params: &TuningParams) -> Variant {
    params
        .iter()
        .filter(|(_, values)| !values.is_empty())
        .map(|(name, values)| {
            let mid_idx = values.len() / 2;
            (name.clone(), values[mid_idx])
        })
        .collect()
}

// ---------------------------------------------------------------------------
// GPU benchmarking protocol
// ---------------------------------------------------------------------------

/// Result of benchmarking a single autotuned variant.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// The parameter values for this variant.
    pub variant: Variant,
    /// Median kernel latency in milliseconds across measured runs.
    pub median_ms: f64,
    /// Minimum kernel latency.
    pub min_ms: f64,
    /// Maximum kernel latency.
    pub max_ms: f64,
}

/// Callback type for benchmarking a single variant on real hardware.
///
/// The production implementation is [`bench_one_variant`] (item 10): load
/// the PTX, synthesize buffers, launch with CUDA events, return a
/// `BenchmarkResult`. It is reached only through [`measure_variants`] —
/// the offline `nsl autotune` command — never from a compile; unit tests
/// pass synthetic closures.
///
/// Arguments: (ptx_source, kernel_name, variant_params) -> Result<BenchmarkResult>
pub type BenchmarkFn = dyn Fn(&str, &str, &Variant) -> Result<BenchmarkResult, String>;

/// Number of warmup launches (not timed) before measured runs — followed
/// by [`bench_one_variant`], the offline measured path (item 10).
pub const WARMUP_RUNS: usize = 5;
/// Number of measured launches for computing median latency
/// ([`bench_one_variant`]).
pub const MEASURED_RUNS: usize = 10;
/// Maximum time (ms) for a single variant before it's skipped.
pub const VARIANT_TIMEOUT_MS: f64 = 5000.0;

/// Find the best autotuned variant for a kernel by benchmarking all Cartesian-product
/// parameter combinations on real hardware.
///
/// - Checks cache first (cache hit skips benchmarking entirely).
/// - If `NSL_AUTOTUNE_FALLBACK=1` is set, uses `select_middle_values()` without GPU.
/// - Calls `ptx_generator` for each variant to produce PTX, then `benchmark_fn` to time it.
/// - Failed variants are skipped (e.g., too much shared memory, CUDA errors).
/// - Winner (lowest median latency) is written to cache.
/// - If `NSL_AUTOTUNE_VERBOSE=1` is set, prints a formatted report table.
pub fn find_best_variant(
    kernel_name: &str,
    tuning_params: &TuningParams,
    cache_hash: &str,
    device: &DeviceIdentity,
    ptx_generator: &dyn Fn(&Variant) -> Result<String, String>,
    benchmark_fn: &BenchmarkFn,
) -> Result<Variant, String> {
    // 1. Check cache — MEASURED records only. A CostModel record for the
    // same (device, key) is an ESTIMATE occupying the slot this path exists
    // to fill; accepting it made `nsl build` before `nsl autotune` a silent
    // no-op that could freeze the estimate into a pinned DB while printing
    // "measured winner" (review HIGH, 2026-08-06 — a collision the v3 key
    // unification created, since under v2's entry-point-dependent keys the
    // two paths never shared a slot). An existing Measured record still
    // short-circuits: re-tuning is idempotent by design.
    match load_cache_record(kernel_name, cache_hash, device, None) {
        Ok(Some(rec)) if matches!(rec.selection, SelectionMethod::Measured) => {
            if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                eprintln!("[autotune] cache hit for {kernel_name} (measured)");
            }
            return Ok(rec.winner);
        }
        Ok(Some(_)) if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() => {
            eprintln!(
                "[autotune] {kernel_name}: cost-model record present — measuring \
                 (measurement replaces estimates, never the reverse)"
            );
        }
        _ => {}
    }

    // 2. Fallback mode (no GPU)
    if std::env::var("NSL_AUTOTUNE_FALLBACK").is_ok() {
        if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
            eprintln!("[autotune] fallback mode: picking median values (no GPU benchmarking)");
        }
        return Ok(select_middle_values(tuning_params));
    }

    // 3. Generate all variants
    let all_variants = cartesian_product(tuning_params);
    let num_variants = all_variants.len();

    // 4. Benchmark each
    let mut results: Vec<BenchmarkResult> = Vec::new();
    for variant in &all_variants {
        let ptx = match ptx_generator(variant) {
            Ok(p) => p,
            Err(e) => {
                if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                    eprintln!("[autotune]   {:?} => compile FAILED: {e}", variant);
                }
                continue; // skip failed compilation
            }
        };

        match benchmark_fn(&ptx, kernel_name, variant) {
            Ok(result) => {
                // Timeout check
                if result.median_ms > VARIANT_TIMEOUT_MS {
                    if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                        eprintln!(
                            "[autotune]   {:?} => too slow ({:.1}ms), skipping",
                            variant, result.median_ms
                        );
                    }
                    continue;
                }
                results.push(result);
            }
            Err(e) => {
                if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                    eprintln!("[autotune]   {:?} => benchmark FAILED: {e}", variant);
                }
                continue;
            }
        }
    }

    // 5. All failed? Fall back to median
    if results.is_empty() {
        eprintln!(
            "[autotune] WARNING: all {num_variants} variants failed for {kernel_name}, using median fallback"
        );
        return Ok(select_middle_values(tuning_params));
    }

    // 6. Select winner
    results.sort_by(|a, b| a.median_ms.partial_cmp(&b.median_ms).unwrap());
    let winner = results[0].variant.clone();

    // 7. Verbose report
    if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
        print_benchmark_report(kernel_name, &results);
    }

    // 8. Write to cache
    let autotune_result = AutotuneResult {
        winner: winner.clone(),
        all_timings: results
            .iter()
            .map(|r| (r.variant.clone(), r.median_ms))
            .collect(),
        device: device.clone(),
        selection: SelectionMethod::Measured,
    };
    write_cache(cache_hash, kernel_name, &autotune_result);

    Ok(winner)
}

/// Parse an `@autotune(name=[v, ...], ...)` decorator into tuning params.
///
/// The single implementation behind both the compile path
/// (`Compiler::extract_autotune_params`) and the offline `nsl autotune`
/// command — the two must agree on decorator semantics or the offline
/// tuner would measure a different variant space than compiles select
/// from.
pub fn extract_autotune_params(
    decorators: &[nsl_ast::decl::Decorator],
    interner: &nsl_lexer::Interner,
) -> Result<TuningParams, String> {
    use nsl_ast::expr::ExprKind;
    let autotune_deco = decorators
        .iter()
        .find(|d| d.name.len() == 1 && interner.resolve(d.name[0].0).unwrap_or("") == "autotune")
        .ok_or_else(|| "@autotune decorator not found".to_string())?;

    let mut params = Vec::new();
    if let Some(ref args) = autotune_deco.args {
        for arg in args {
            let name = arg
                .name
                .as_ref()
                .and_then(|s| interner.resolve(s.0))
                .unwrap_or("unnamed")
                .to_string();
            let values = match &arg.value.kind {
                ExprKind::ListLiteral(items) => items
                    .iter()
                    .filter_map(|item| {
                        if let ExprKind::IntLiteral(v) = &item.kind {
                            Some(*v)
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => vec![],
            };
            params.push((name, values));
        }
    }
    Ok(params)
}

/// Rewrite every `Symbol(SymbolU32 { value: N })` in an AST `Debug`
/// rendering to the RESOLVED identifier text.
///
/// The cache key must not bake interner indices into the hashed bytes:
/// indices are a function of everything the process interned before the
/// kernel parsed, so two entry points compiling the identical kernel get
/// different keys (measured: `nsl autotune` vs `nsl build` disagreed on
/// every key until this existed). Resolving the symbols makes the bytes a
/// function of the SOURCE alone. `SymbolU32`'s Debug prints its internal
/// non-zero value — index + 1 — which the unit test pins.
pub fn resolve_symbols_for_hash(debug: &str, interner: &nsl_lexer::Interner) -> String {
    use string_interner::Symbol as _;
    const PAT: &str = "Symbol(SymbolU32 { value: ";
    let mut out = String::with_capacity(debug.len());
    let mut rest = debug;
    while let Some(i) = rest.find(PAT) {
        out.push_str(&rest[..i]);
        let tail = &rest[i + PAT.len()..];
        let Some(end) = tail.find(" })") else {
            // Malformed tail: keep the raw text rather than looping.
            out.push_str(&rest[i..]);
            return strip_node_ids(&out);
        };
        let name = tail[..end]
            .trim()
            .parse::<usize>()
            .ok()
            .and_then(|v| v.checked_sub(1)) // Debug value = index + 1
            .and_then(string_interner::DefaultSymbol::try_from_usize)
            .and_then(|s| interner.resolve(s))
            .unwrap_or("?unresolved?");
        out.push_str("Sym(\"");
        out.push_str(name);
        out.push_str("\")");
        rest = &tail[end + 3..];
    }
    out.push_str(rest);
    strip_node_ids(&out)
}

/// Drop `, id: NodeId(N)` fields: node ids are a PARSER COUNTER, so the
/// same kernel gets different ids depending on what the process parsed
/// first (measured: `nsl build` starts ~26 ids later than `nsl autotune`
/// on the identical file). Then neutralize `FileId(N)` inside spans: the
/// file id is the SOURCE-MAP REGISTRATION ORDER, so a kernel in an imported
/// module hashes differently under `nsl autotune lib.nsl` (FileId(0)) than
/// under `nsl build main.nsl` (FileId >= 1) — the import-structured layout
/// would silently never consume measured records (review HIGH, 2026-08-06).
/// Byte positions stay: they are file-content-relative and deterministic
/// (position sensitivity is conservative — an edit above the kernel
/// invalidates the tune, which is the safe direction).
///
/// Both rewrites assume their patterns cannot occur in user-controlled
/// text: the kernel DSL refuses string literals today, and if that ever
/// changes these delimiters must be escaped first or two kernels differing
/// only inside a literal could collide.
fn strip_node_ids(s: &str) -> String {
    let no_ids = strip_delimited(s, ", id: NodeId(", ')', "");
    strip_delimited(&no_ids, "file_id: FileId(", ')', "file_id: FileId(_")
}

/// Remove `pat…close` spans, emitting `replacement` in their place.
fn strip_delimited(s: &str, pat: &str, close: char, replacement: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find(pat) {
        out.push_str(&rest[..i]);
        let tail = &rest[i + pat.len()..];
        let Some(end) = tail.find(close) else {
            out.push_str(&rest[i..]);
            return out;
        };
        out.push_str(replacement);
        if !replacement.is_empty() {
            out.push(close);
        }
        rest = &tail[end + 1..];
    }
    out.push_str(rest);
    out
}

/// One synthesized kernel argument for the offline bencher.
///
/// Derived from the `KernelDef`'s parameter list by the `nsl autotune`
/// command: `Tensor` params become zero-filled device f32 buffers of the
/// representative element count, scalar params become fixed immediates.
/// Zero-filled inputs time faithfully for the data-independent kernels the
/// `@autotune` DSL expresses (elementwise bodies indexed by `thread_id()`);
/// a data-dependent kernel would need real inputs, which an offline tuner
/// does not have — the ranking caveat is documented on `measure_variants`.
#[derive(Debug, Clone, Copy)]
pub enum BenchArg {
    /// Device f32 buffer of `elems` elements, zero-filled.
    TensorF32,
    /// Scalar delivered as round-toward-zero(x): the kernel ABI declares
    /// every param `.param .u64` and converts numerically, so the bencher
    /// passes an 8-byte integer slot (see `bench_one_variant`).
    F32(f32),
    /// Immediate integer scalar (8-byte slot, matching the `.u64` ABI).
    I64(i64),
}

/// One `@autotune` kernel captured DURING a real compile, carrying
/// everything the offline bencher needs: the compile path's own cache key
/// (so a measured record lands exactly where later compiles look — key
/// reconstruction outside the pipeline provably drifts), the per-variant
/// PTX the compile path's own generator produced, and the synthesized
/// argument shapes.
#[derive(Debug, Clone)]
pub struct CapturedAutotuneKernel {
    pub name: String,
    pub params: TuningParams,
    pub hash: String,
    pub args: Vec<BenchArg>,
    pub variant_ptx: Vec<(Variant, String)>,
}

static CAPTURE: std::sync::Mutex<Option<Vec<CapturedAutotuneKernel>>> =
    std::sync::Mutex::new(None);

/// Arm capture mode: the next compile records every `@autotune` kernel it
/// selects for (name, params, key, per-variant PTX) instead of only
/// cost-model-selecting. Used by `nsl autotune`; normal compiles never arm.
pub fn arm_capture() {
    *CAPTURE.lock().unwrap() = Some(Vec::new());
}

pub fn capture_armed() -> bool {
    CAPTURE.lock().unwrap().is_some()
}

/// Take the kernels captured since [`arm_capture`], disarming.
pub fn take_capture() -> Vec<CapturedAutotuneKernel> {
    CAPTURE.lock().unwrap().take().unwrap_or_default()
}

pub(crate) fn push_capture(k: CapturedAutotuneKernel) {
    if let Some(v) = CAPTURE.lock().unwrap().as_mut() {
        v.push(k);
    }
}

/// Map one kernel AST parameter to a synthesized bench argument (shared by
/// the capture site so the arg model cannot drift from the kernel DSL).
pub fn bench_arg_for_param(
    p: &nsl_ast::decl::Param,
    interner: &nsl_lexer::Interner,
) -> BenchArg {
    use nsl_ast::types::TypeExprKind;
    match p.type_ann.as_ref().map(|t| &t.kind) {
        Some(TypeExprKind::Tensor { .. }) => BenchArg::TensorF32,
        Some(TypeExprKind::Named(sym)) => match interner.resolve(sym.0).unwrap_or("") {
            "Tensor" => BenchArg::TensorF32,
            "int" | "i64" => BenchArg::I64(1),
            _ => BenchArg::F32(1.0),
        },
        // Untyped params in the kernel DSL default to Tensor.
        _ => BenchArg::TensorF32,
    }
}

/// Item 10, the measured half: benchmark every captured variant of one
/// `@autotune` kernel with CUDA events and record the winner as a
/// [`SelectionMethod::Measured`] cache entry under the compile path's own
/// key.
///
/// This is [`find_best_variant`]'s production caller — reached only from
/// the offline `nsl autotune` command, never from a compile. The launch
/// geometry mirrors the elementwise convention the `@autotune` kernel
/// family uses: `blockDim.x` is the variant's block-like tuning parameter
/// (name contains "block"; falls back to 256),
/// `gridDim.x = ceil(elems / blockDim.x)`. Timings rank variants against
/// each other on THIS device at the representative `elems`; the ranking is
/// what the cache stores, and it is strictly more grounded than the cost
/// model's inverse-block-product proxy over the same assumed element count.
#[cfg(feature = "cuda")]
pub fn measure_variants(
    captured: &CapturedAutotuneKernel,
    device: &DeviceIdentity,
    elems: usize,
) -> Result<Variant, String> {
    let args = captured.args.clone();
    let bench = move |ptx: &str, name: &str, variant: &Variant| -> Result<BenchmarkResult, String> {
        bench_one_variant(ptx, name, variant, &args, elems)
    };
    let by_variant: std::collections::HashMap<Vec<(String, i64)>, &String> =
        captured.variant_ptx.iter().map(|(v, p)| (v.clone(), p)).collect();
    let ptx_generator = |variant: &Variant| -> Result<String, String> {
        by_variant
            .get(variant)
            .map(|p| (*p).clone())
            .ok_or_else(|| format!("variant {variant:?} was not captured"))
    };
    find_best_variant(
        &captured.name,
        &captured.params,
        &captured.hash,
        device,
        &ptx_generator,
        &bench,
    )
}

/// CUDA-event timing of one compiled variant over synthesized buffers.
///
/// Self-contained driver usage (cuInit + primary-context retain, mirroring
/// `bin/bench/launch.rs`); errors surface as `Err(String)` so
/// [`find_best_variant`] skips the variant instead of aborting the tune.
#[cfg(feature = "cuda")]
fn bench_one_variant(
    ptx: &str,
    kernel_name: &str,
    variant: &Variant,
    args: &[BenchArg],
    elems: usize,
) -> Result<BenchmarkResult, String> {
    use cudarc::driver::sys;
    use std::os::raw::c_void;

    unsafe {
        // One-time context (idempotent across variants and kernels).
        static CTX: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
        let ctx = CTX
            .get_or_init(|| {
                let r = sys::cuInit(0);
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cuInit: {r:?}"));
                }
                let mut dev: sys::CUdevice = 0;
                let r = sys::cuDeviceGet(&mut dev, 0);
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cuDeviceGet: {r:?}"));
                }
                let mut ctx: sys::CUcontext = std::ptr::null_mut();
                let r = sys::cuDevicePrimaryCtxRetain(&mut ctx, dev);
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cuDevicePrimaryCtxRetain: {r:?}"));
                }
                Ok(ctx as usize)
            })
            .clone()?;
        let r = sys::cuCtxSetCurrent(ctx as sys::CUcontext);
        if r != sys::CUresult::CUDA_SUCCESS {
            return Err(format!("cuCtxSetCurrent: {r:?}"));
        }

        // Load this variant's PTX with JIT log capture. The kernel compiler
        // emits null-terminated PTX (its consumers pass raw pointers);
        // CString must not see the terminator.
        let ptx_c = std::ffi::CString::new(ptx.trim_end_matches('\0'))
            .map_err(|_| "PTX contains interior NUL".to_string())?;
        let mut log = [0u8; 4096];
        let mut opts = [
            sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
            sys::CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut vals = [
            log.as_mut_ptr() as *mut c_void,
            log.len() as *mut c_void,
        ];
        let mut module: sys::CUmodule = std::ptr::null_mut();
        let r = sys::cuModuleLoadDataEx(
            &mut module,
            ptx_c.as_ptr() as *const c_void,
            opts.len() as u32,
            opts.as_mut_ptr(),
            vals.as_mut_ptr(),
        );
        if r != sys::CUresult::CUDA_SUCCESS {
            let msg = std::ffi::CStr::from_ptr(log.as_ptr() as *const std::ffi::c_char)
                .to_string_lossy()
                .into_owned();
            return Err(format!("cuModuleLoadDataEx: {r:?}: {msg}"));
        }
        // Everything below must unload the module on exit.
        let result = (|| -> Result<BenchmarkResult, String> {
            let name_c = std::ffi::CString::new(kernel_name).unwrap();
            let mut func: sys::CUfunction = std::ptr::null_mut();
            let r = sys::cuModuleGetFunction(&mut func, module, name_c.as_ptr());
            if r != sys::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleGetFunction({kernel_name}): {r:?}"));
            }

            // Synthesize arguments. Slot storage must outlive the launches.
            //
            // The kernel compiler declares EVERY parameter `.param .u64` and
            // consumes scalars via numeric conversion (`cvt.rn.f32.u64`), so
            // every slot here is 8 bytes — a 4-byte f32 slot would make
            // cuLaunchKernel read past the allocation AND deliver the f32's
            // bit pattern as a giant integer (review MEDIUM, 2026-08-06).
            // Scalars are therefore encoded as their integer value in a u64;
            // BenchArg::F32(x) reaches the kernel as round-toward-zero(x).
            let mut dev_bufs: Vec<sys::CUdeviceptr> = Vec::new();
            let mut scalar_slots: Vec<u64> = Vec::new();
            for a in args {
                match a {
                    BenchArg::TensorF32 => {
                        let mut p: sys::CUdeviceptr = 0;
                        let r = sys::cuMemAlloc_v2(&mut p, elems * 4);
                        if r != sys::CUresult::CUDA_SUCCESS {
                            for &b in &dev_bufs {
                                sys::cuMemFree_v2(b);
                            }
                            return Err(format!("cuMemAlloc({} B): {r:?}", elems * 4));
                        }
                        // Fill result ignored: a memset failure surfaces as
                        // a launch/timing error two calls later, on the same
                        // stream (same policy as the event calls below).
                        sys::cuMemsetD8_v2(p, 0, elems * 4);
                        dev_bufs.push(p);
                    }
                    BenchArg::F32(v) => scalar_slots.push(*v as u64),
                    BenchArg::I64(v) => scalar_slots.push(*v as u64),
                }
            }
            let free_all = |bufs: &[sys::CUdeviceptr]| {
                for &b in bufs {
                    sys::cuMemFree_v2(b);
                }
            };
            let (mut di, mut si) = (0usize, 0usize);
            let mut slots: Vec<*mut c_void> = Vec::with_capacity(args.len());
            for a in args {
                match a {
                    BenchArg::TensorF32 => {
                        slots.push(&mut dev_bufs[di] as *mut _ as *mut c_void);
                        di += 1;
                    }
                    BenchArg::F32(_) | BenchArg::I64(_) => {
                        slots.push(&mut scalar_slots[si] as *mut _ as *mut c_void);
                        si += 1;
                    }
                }
            }

            // Elementwise launch geometry; block from the variant. The
            // heuristic is name-based — say so out loud when it is guessing
            // (no block-named param, several of them, or a clamp), because a
            // clamped launch measures PTX compiled for one block size but
            // launched at another.
            let block_params: Vec<&(String, i64)> =
                variant.iter().filter(|(n, _)| n.contains("block")).collect();
            let block = match block_params.as_slice() {
                [] => {
                    eprintln!(
                        "[autotune] {kernel_name}: no block-named tuning param — \
                         launching at blockDim.x=256"
                    );
                    256
                }
                [one] => one.1,
                many => {
                    eprintln!(
                        "[autotune] {kernel_name}: {} block-named params — using '{}' \
                         for blockDim.x",
                        many.len(),
                        many[0].0
                    );
                    many[0].1
                }
            };
            let clamped = block.clamp(1, 1024);
            if clamped != block {
                eprintln!(
                    "[autotune] {kernel_name}: blockDim.x clamped {block} -> {clamped}; \
                     this variant's timing mixes PTX compiled for {block} with a \
                     {clamped}-thread launch"
                );
            }
            let block = clamped;
            let grid = ((elems as i64 + block - 1) / block).max(1);

            let launch = |func: sys::CUfunction, slots: &mut [*mut c_void]| -> Result<(), String> {
                let r = sys::cuLaunchKernel(
                    func,
                    grid as u32, 1, 1,
                    block as u32, 1, 1,
                    0,
                    std::ptr::null_mut(),
                    slots.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                if r != sys::CUresult::CUDA_SUCCESS {
                    return Err(format!("cuLaunchKernel: {r:?}"));
                }
                Ok(())
            };

            for _ in 0..WARMUP_RUNS {
                if let Err(e) = launch(func, &mut slots) {
                    free_all(&dev_bufs);
                    return Err(e);
                }
            }
            let r = sys::cuCtxSynchronize();
            if r != sys::CUresult::CUDA_SUCCESS {
                free_all(&dev_bufs);
                return Err(format!("warmup sync: {r:?}"));
            }

            let mut start: sys::CUevent = std::ptr::null_mut();
            let mut stop: sys::CUevent = std::ptr::null_mut();
            sys::cuEventCreate(&mut start, 0);
            sys::cuEventCreate(&mut stop, 0);
            let mut times = Vec::with_capacity(MEASURED_RUNS);
            for _ in 0..MEASURED_RUNS {
                sys::cuEventRecord(start, std::ptr::null_mut());
                if let Err(e) = launch(func, &mut slots) {
                    sys::cuEventDestroy_v2(start);
                    sys::cuEventDestroy_v2(stop);
                    free_all(&dev_bufs);
                    return Err(e);
                }
                sys::cuEventRecord(stop, std::ptr::null_mut());
                sys::cuEventSynchronize(stop);
                let mut ms = 0f32;
                let r = sys::cuEventElapsedTime_v2(&mut ms, start, stop);
                if r != sys::CUresult::CUDA_SUCCESS {
                    sys::cuEventDestroy_v2(start);
                    sys::cuEventDestroy_v2(stop);
                    free_all(&dev_bufs);
                    return Err(format!("cuEventElapsedTime: {r:?}"));
                }
                times.push(ms as f64);
            }
            sys::cuEventDestroy_v2(start);
            sys::cuEventDestroy_v2(stop);
            free_all(&dev_bufs);

            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(BenchmarkResult {
                variant: variant.clone(),
                median_ms: times[times.len() / 2],
                min_ms: times[0],
                max_ms: times[times.len() - 1],
            })
        })();
        sys::cuModuleUnload(module);
        result
    }
}

/// Find the best autotuned variant using the cost model as a proxy for GPU timing.
///
/// This is the compile-time path: generates PTX for each variant, estimates
/// execution time via the roofline cost model (from `cost_model.rs`), and picks
/// the variant with the lowest estimated latency. Much better than middle-value
/// fallback because the cost model considers the target GPU's bandwidth, compute,
/// and SM count.
///
/// When `fresh` is true, the cache is skipped entirely (--autotune-fresh).
pub fn find_best_variant_cost_model(
    kernel_name: &str,
    tuning_params: &TuningParams,
    cache_hash: &str,
    device: &DeviceIdentity,
    cost_model_spec: &str,
    fresh: bool,
    ptx_generator: &dyn Fn(&Variant) -> Result<String, String>,
    cost_estimator: &dyn Fn(&Variant) -> Result<f64, String>,
) -> Result<Variant, String> {
    // 1. Check cache (unless fresh)
    if !fresh {
        if let Some(cached) = check_cache(kernel_name, cache_hash, device, Some(cost_model_spec))
        {
            if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                eprintln!("[autotune] cache hit for {kernel_name}");
            }
            return Ok(cached);
        }
    } else if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
        eprintln!("[autotune] --autotune-fresh: skipping cache for {kernel_name}");
    }

    // 2. Fallback mode
    if std::env::var("NSL_AUTOTUNE_FALLBACK").is_ok() {
        if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
            eprintln!("[autotune] fallback mode: picking median values");
        }
        return Ok(select_middle_values(tuning_params));
    }

    // 3. Generate all variants
    let all_variants = cartesian_product(tuning_params);
    let num_variants = all_variants.len();

    // 4. For each variant: generate PTX (validates compilability) and estimate cost
    let mut results: Vec<BenchmarkResult> = Vec::new();
    for variant in &all_variants {
        // Verify PTX compiles (skip invalid variants, e.g. BLOCK_SIZE too large for shmem)
        match ptx_generator(variant) {
            Ok(_) => {}
            Err(e) => {
                if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                    eprintln!("[autotune]   {:?} => compile FAILED: {e}", variant);
                }
                continue;
            }
        }

        // Estimate cost via roofline model
        match cost_estimator(variant) {
            Ok(estimated_us) => {
                let estimated_ms = estimated_us / 1000.0;
                results.push(BenchmarkResult {
                    variant: variant.clone(),
                    median_ms: estimated_ms,
                    min_ms: estimated_ms,
                    max_ms: estimated_ms,
                });
            }
            Err(e) => {
                if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                    eprintln!("[autotune]   {:?} => cost estimate FAILED: {e}", variant);
                }
                continue;
            }
        }
    }

    // 5. All failed? Fall back to median
    if results.is_empty() {
        eprintln!(
            "[autotune] WARNING: all {num_variants} variants failed for {kernel_name}, using median fallback"
        );
        return Ok(select_middle_values(tuning_params));
    }

    // 6. Select winner (lowest estimated cost)
    results.sort_by(|a, b| a.median_ms.partial_cmp(&b.median_ms).unwrap());
    let winner = results[0].variant.clone();

    // 7. Verbose report
    if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
        eprintln!("\n=== Autotune Cost-Model Report: {kernel_name} (estimated, not measured) ===");
        print_benchmark_report(kernel_name, &results);
    }

    // 8. Write to cache
    let autotune_result = AutotuneResult {
        winner: winner.clone(),
        all_timings: results
            .iter()
            .map(|r| (r.variant.clone(), r.median_ms))
            .collect(),
        device: device.clone(),
        selection: SelectionMethod::CostModel {
            spec: cost_model_spec.to_string(),
        },
    };
    write_cache(cache_hash, kernel_name, &autotune_result);

    Ok(winner)
}

/// Print a formatted benchmark report table to stderr.
pub fn print_benchmark_report(kernel_name: &str, results: &[BenchmarkResult]) {
    eprintln!("\n=== Autotune Report: {kernel_name} ===");
    eprintln!(
        "{:<40} {:>10} {:>10} {:>10}",
        "Variant", "Median", "Min", "Max"
    );
    eprintln!("{:-<72}", "");
    for r in results {
        let params_str: String = r
            .variant
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ");
        let marker = if r.variant == results[0].variant {
            " <-- winner"
        } else {
            ""
        };
        eprintln!(
            "{:<40} {:>9.3}ms {:>9.3}ms {:>9.3}ms{}",
            params_str, r.median_ms, r.min_ms, r.max_ms, marker
        );
    }
    eprintln!();
}

/// Hash a kernel's AST body for cache key generation (SHA-256).
///
/// Absorbs the schema version, kernel name, serialised AST body, tuning
/// parameter definitions, input tensor shapes, and the **driver-reported**
/// device identity. The cache is invalidated whenever any of these change.
///
/// `input_shapes` is empty on the compile path — shapes are not known until
/// runtime. That is a real limit on how specific the key can be, not an
/// oversight; a variant tuned for one shape is reused for all of them. It is
/// hashed anyway so a future shape-specialised caller does not collide with
/// today's entries.
///
/// Every field is length-prefixed. Concatenating raw bytes let adjacent fields
/// slide into one another — kernel `"ab"` with body `"c"` hashed identically to
/// kernel `"a"` with body `"bc"`.
pub fn hash_kernel_ast(
    kernel_name: &str,
    ast_bytes: &[u8],
    tuning_params: &TuningParams,
    input_shapes: &[Vec<i64>],
    device: &DeviceIdentity,
) -> String {
    let mut hasher = Sha256::new();
    let mut field = |bytes: &[u8]| {
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(bytes);
    };
    field(&CACHE_SCHEMA_VERSION.to_le_bytes());
    field(kernel_name.as_bytes());
    field(ast_bytes);
    field(&(tuning_params.len() as u64).to_le_bytes());
    for (name, values) in tuning_params {
        field(name.as_bytes());
        field(&(values.len() as u64).to_le_bytes());
        for v in values {
            field(&v.to_le_bytes());
        }
    }
    field(&(input_shapes.len() as u64).to_le_bytes());
    for shape in input_shapes {
        field(&(shape.len() as u64).to_le_bytes());
        for &dim in shape {
            field(&dim.to_le_bytes());
        }
    }
    field(device.device_name.as_bytes());
    field(&device.sm_version.to_le_bytes());
    field(&device.sm_count.to_le_bytes());
    field(&device.driver_version.to_le_bytes());
    hex(&hasher.finalize())
}

/// Lower-case hex of a digest. sha2 0.11 hands back a `hybrid_array::Array`,
/// which — unlike the 0.10 `GenericArray` — does not implement `LowerHex`.
pub(crate) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Returns the autotune cache directory path.
pub fn cache_dir() -> PathBuf {
    PathBuf::from(".nsl-cache/autotune")
}

/// Read and validate the cache entry for a kernel + hash.
///
/// Returns `Ok(None)` when there is simply no entry, `Err(reason)` when one
/// exists but must not be trusted. Validation is the point: the key alone
/// cannot be relied on to keep foreign entries out, because a `.nsl-cache`
/// directory is an ordinary directory that gets copied, committed, restored
/// from a CI artifact, or shared over a network mount.
pub fn load_cache_record(
    kernel_name: &str,
    hash: &str,
    device: &DeviceIdentity,
    cost_model_spec: Option<&str>,
) -> Result<Option<AutotuneCacheRecord>, CacheReject> {
    // Frozen DB overlay first (item 10): a record shipped via
    // `--autotune-db` wins over the per-machine cache dir, so a pinned
    // build selects the pinned winners even on a machine that has tuned
    // locally since. Same validation as a disk record — a frozen entry for
    // another device or a drifted key is REJECTED at lookup, not trusted.
    if let Some(db) = FROZEN_DB.get() {
        if let Some(record) = db.get(&(kernel_name.to_string(), hash.to_string())) {
            return validate_record(record.clone(), hash, device, cost_model_spec).map(Some);
        }
        // Drift is otherwise SILENT: the pin validates content, not
        // applicability, so an edited kernel (or device swap) quietly
        // reverts to the cost model while the build looks pinned. Warn
        // once per kernel when the DB carries records for this kernel
        // that no longer match this source/device (review MEDIUM).
        if db.keys().any(|(k, _)| k == kernel_name) {
            static WARNED: std::sync::Mutex<Option<std::collections::HashSet<String>>> =
                std::sync::Mutex::new(None);
            let mut g = WARNED.lock().unwrap();
            if g.get_or_insert_with(Default::default).insert(kernel_name.to_string()) {
                eprintln!(
                    "[autotune] warning: the frozen tuning DB has record(s) for \
                     '{kernel_name}' but none match this source/device — the kernel \
                     or its file changed since the DB was frozen; falling back to \
                     the cost model. Re-run `nsl autotune --freeze` to re-pin."
                );
            }
        }
    }
    let path = cache_dir().join(format!("{}_{}.json", kernel_name, hash));
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Ok(None);
    };
    let record: AutotuneCacheRecord = serde_json::from_str(&content)
        .map_err(|e| CacheReject::Unparseable(e.to_string()))?;
    validate_record(record, hash, device, cost_model_spec).map(Some)
}

/// The validation every consumed record passes, wherever it came from
/// (disk cache or frozen DB): schema, device identity, key, cost-model
/// spec drift (CostModel records only), non-empty winner.
fn validate_record(
    record: AutotuneCacheRecord,
    hash: &str,
    device: &DeviceIdentity,
    cost_model_spec: Option<&str>,
) -> Result<AutotuneCacheRecord, CacheReject> {
    if record.schema_version != CACHE_SCHEMA_VERSION {
        return Err(CacheReject::SchemaMismatch {
            found: record.schema_version,
            expected: CACHE_SCHEMA_VERSION,
        });
    }
    // Device before key: both are disqualifying, but only one of them names the
    // machines involved, and a foreign entry that ALSO carries a foreign key
    // would otherwise be reported as a mere filename problem.
    if record.device != *device {
        return Err(CacheReject::DeviceMismatch {
            found: Box::new(record.device),
            expected: Box::new(device.clone()),
        });
    }
    if record.cache_key != hash {
        return Err(CacheReject::KeyMismatch {
            found: record.cache_key,
            expected: hash.to_string(),
        });
    }
    // A cost-model winner is only as good as the spec it was priced against,
    // and that spec is NOT part of the key — it is a model, not an identity.
    // The realistic drift: a card missing from GPU_DATABASE is priced against
    // the A100 default, someone later adds its real GpuSpec (which csha's
    // fallback message actively asks them to do), and the device identity is
    // unchanged — so without this check the A100-priced winner survives every
    // rebuild until someone passes --autotune-fresh.
    if let (SelectionMethod::CostModel { spec }, Some(expected)) = (&record.selection, cost_model_spec)
        && spec != expected
    {
        return Err(CacheReject::CostModelSpecChanged {
            found: spec.clone(),
            expected: expected.to_string(),
        });
    }
    if record.winner.is_empty() {
        return Err(CacheReject::EmptyWinner);
    }
    Ok(record)
}

// ---------------------------------------------------------------------------
// Item 10: frozen tuning database (`--autotune-db`, `nsl autotune --freeze`)
// ---------------------------------------------------------------------------

static FROZEN_DB: std::sync::OnceLock<
    std::collections::HashMap<(String, String), AutotuneCacheRecord>,
> = std::sync::OnceLock::new();

/// Serialize records into the frozen-DB wire form and its pin hash.
///
/// Deterministic: records sort by (kernel, cache_key), JSON is pretty with
/// stable field order (serde struct order), and the hash is the sha256 of
/// the exact bytes written — so `--autotune-db-sha256` pins content, not a
/// file identity.
pub fn freeze_records(records: &mut Vec<AutotuneCacheRecord>) -> (String, String) {
    records.sort_by(|a, b| (&a.kernel, &a.cache_key).cmp(&(&b.kernel, &b.cache_key)));
    let json = serde_json::to_string_pretty(&records).expect("records serialize");
    let mut h = Sha256::new();
    h.update(json.as_bytes());
    (json, hex(&h.finalize()))
}

/// Load a frozen tuning DB for this process. Consulted by every cache
/// lookup BEFORE the per-machine cache dir; per-record validation happens
/// at lookup (device identity, key, schema), so a DB tuned on another
/// machine simply never matches rather than poisoning selections.
///
/// `expected_sha256`: refuse a DB whose content hash differs — the
/// reproducible-build pin. Loading twice is an error (one DB per process;
/// the second load would silently shadow the first).
pub fn load_frozen_db(path: &std::path::Path, expected_sha256: Option<&str>) -> Result<usize, String> {
    let bytes = std::fs::read(path)
        .map_err(|e| format!("--autotune-db {}: {e}", path.display()))?;
    if let Some(expected) = expected_sha256 {
        let mut h = Sha256::new();
        h.update(&bytes);
        let got = hex(&h.finalize());
        if !got.eq_ignore_ascii_case(expected) {
            return Err(format!(
                "--autotune-db-sha256 mismatch for {}: expected {expected}, got {got} — \
                 the tuning database is not the one this build pinned",
                path.display()
            ));
        }
    }
    let records: Vec<AutotuneCacheRecord> = serde_json::from_slice(&bytes)
        .map_err(|e| format!("--autotune-db {}: unparseable: {e}", path.display()))?;
    let n = records.len();
    let mut map = std::collections::HashMap::with_capacity(n);
    for r in records {
        map.insert((r.kernel.clone(), r.cache_key.clone()), r);
    }
    FROZEN_DB
        .set(map)
        .map_err(|_| "--autotune-db loaded twice in one process".to_string())?;
    Ok(n)
}

/// Check whether a usable cached winner exists for the given kernel + hash.
///
/// A rejected entry is a miss, never an error: the worst outcome of ignoring a
/// cache file is recomputing the selection. Noteworthy rejections warn so the
/// entry does not silently do nothing forever.
pub fn check_cache(
    kernel_name: &str,
    hash: &str,
    device: &DeviceIdentity,
    cost_model_spec: Option<&str>,
) -> Option<Variant> {
    match load_cache_record(kernel_name, hash, device, cost_model_spec) {
        Ok(Some(record)) => Some(record.winner),
        Ok(None) => None,
        Err(reject) => {
            if reject.is_noteworthy() || std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
                eprintln!(
                    "[autotune] ignoring cache entry for '{kernel_name}': {}",
                    reject.describe()
                );
            }
            None
        }
    }
}

/// Build the on-disk record for an autotune result.
///
/// Split out from [`write_cache`] so a test can assert on the record without
/// touching the filesystem.
pub fn build_cache_record(hash: &str, kernel_name: &str, result: &AutotuneResult) -> AutotuneCacheRecord {
    let winner_ms = result
        .all_timings
        .iter()
        .find(|(v, _)| v == &result.winner)
        .map(|(_, ms)| *ms)
        .unwrap_or(0.0);
    let timestamp_secs = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    AutotuneCacheRecord {
        schema_version: CACHE_SCHEMA_VERSION,
        kernel: kernel_name.to_string(),
        cache_key: hash.to_string(),
        device: result.device.clone(),
        selection: result.selection.clone(),
        winner: result.winner.clone(),
        winner_ms,
        variants_tested: result.all_timings.len(),
        all_timings: result
            .all_timings
            .iter()
            .map(|(params, ms)| TimingEntry {
                params: params.clone(),
                median_ms: *ms,
            })
            .collect(),
        timestamp_secs,
    }
}

/// Write an autotune result (winner + all timings) to the cache directory.
pub fn write_cache(hash: &str, kernel_name: &str, result: &AutotuneResult) {
    // Never persist a winnerless entry. `load_cache_record` refuses one, so
    // writing it creates a file that is rejected on every subsequent read and
    // rewritten on every subsequent miss — a permanent warn-and-rewrite loop
    // that never converges. A kernel with no tuning parameters legitimately has
    // an empty winner; it simply has nothing worth caching.
    if result.winner.is_empty() {
        return;
    }
    let dir = cache_dir();
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let record = build_cache_record(hash, kernel_name, result);
    let Ok(json) = serde_json::to_string_pretty(&record) else {
        return;
    };
    let path = dir.join(format!("{}_{}.json", kernel_name, hash));
    std::fs::write(&path, json).ok();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A device identity that is not the local machine's, for hash and record
    /// tests that must not depend on what GPU the test host happens to have.
    fn dev(name: &str) -> DeviceIdentity {
        DeviceIdentity {
            device_name: name.to_string(),
            sm_version: 86,
            sm_count: 84,
            driver_version: 12_000,
        }
    }

    #[test]
    fn test_ast_hash_deterministic_and_sensitive() {
        let params = vec![("block_size".to_string(), vec![64, 128])];
        let d = dev("GPU0");

        let hash1 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![256]], &d);
        let hash2 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![256]], &d);
        assert_eq!(hash1, hash2, "identical inputs must produce the same hash");

        // Different AST body
        let hash3 = hash_kernel_ast("my_kernel", b"body_v2", &params, &[vec![256]], &d);
        assert_ne!(hash1, hash3, "different AST body must change hash");

        // Different input shapes
        let hash4 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![512]], &d);
        assert_ne!(hash1, hash4, "different input shapes must change hash");

        // Different device name
        let hash5 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![256]], &dev("GPU1"));
        assert_ne!(hash1, hash5, "different device must change hash");

        // Each identity field independently participates. Without this, a key
        // could name the right card and still describe the wrong silicon —
        // a rebadged part, a different SKU's SM count, a driver upgrade that
        // changes ptxas codegen.
        for (label, other) in [
            (
                "sm_version",
                DeviceIdentity {
                    sm_version: 89,
                    ..d.clone()
                },
            ),
            (
                "sm_count",
                DeviceIdentity {
                    sm_count: 128,
                    ..d.clone()
                },
            ),
            (
                "driver_version",
                DeviceIdentity {
                    driver_version: 13_030,
                    ..d.clone()
                },
            ),
        ] {
            assert_ne!(
                hash1,
                hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![256]], &other),
                "a different {label} must change the cache key"
            );
        }
    }

    #[test]
    fn hash_fields_do_not_slide_into_one_another() {
        // Length-prefixing check: without it, concatenating the raw bytes of
        // adjacent fields makes ("ab", "c") and ("a", "bc") hash identically,
        // so two genuinely different kernels share a cache entry.
        let params: TuningParams = vec![];
        let d = dev("GPU0");
        assert_ne!(
            hash_kernel_ast("ab", b"c", &params, &[], &d),
            hash_kernel_ast("a", b"bc", &params, &[], &d),
        );
    }

    #[test]
    fn test_cartesian_product() {
        let params = vec![
            ("block_size".to_string(), vec![64, 128, 256]),
            ("warps".to_string(), vec![2, 4]),
        ];
        let product = cartesian_product(&params);
        assert_eq!(product.len(), 6);
        assert!(product.contains(&vec![
            ("block_size".to_string(), 64),
            ("warps".to_string(), 2)
        ]));
        assert!(product.contains(&vec![
            ("block_size".to_string(), 256),
            ("warps".to_string(), 4)
        ]));
    }

    #[test]
    fn test_cartesian_product_single_param() {
        let params = vec![("threads".to_string(), vec![32, 64, 128, 256])];
        let product = cartesian_product(&params);
        assert_eq!(product.len(), 4);
    }

    #[test]
    fn test_cartesian_product_empty() {
        let params: TuningParams = vec![];
        let product = cartesian_product(&params);
        assert_eq!(product.len(), 1); // one empty variant
        assert_eq!(product[0].len(), 0);
    }

    #[test]
    fn test_middle_value_fallback() {
        let params = vec![
            ("block_size".to_string(), vec![64, 128, 256]),
            ("warps".to_string(), vec![2, 4, 8]),
        ];
        let middle = select_middle_values(&params);
        assert_eq!(
            middle,
            vec![("block_size".to_string(), 128), ("warps".to_string(), 4)]
        );
    }

    #[test]
    fn test_middle_value_even_count() {
        // With 4 values [1,2,3,4], index 2 => value 3
        let params = vec![("x".to_string(), vec![1, 2, 3, 4])];
        let middle = select_middle_values(&params);
        assert_eq!(middle, vec![("x".to_string(), 3)]);
    }

    #[test]
    fn test_cache_roundtrip() {
        let result = AutotuneResult {
            winner: vec![("block_size".to_string(), 256)],
            all_timings: vec![
                (vec![("block_size".to_string(), 128)], 1.5),
                (vec![("block_size".to_string(), 256)], 0.8),
            ],
            device: dev("TestGPU"),
            selection: SelectionMethod::Measured,
        };

        let hash = "test_hash_roundtrip_12345";
        write_cache(hash, "test_kernel", &result);

        let cached = check_cache("test_kernel", hash, &dev("TestGPU"), None);
        assert!(cached.is_some(), "cache file should be readable");
        let winner = cached.unwrap();
        assert_eq!(winner, vec![("block_size".to_string(), 256)]);

        // Cleanup
        let path = cache_dir().join(format!("test_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_check_cache_miss() {
        let cached = check_cache("nonexistent_kernel", "nonexistent_hash", &dev("TestGPU"), None);
        assert!(cached.is_none());
    }

    // ── GPU benchmarking tests ────────────────────────────────────────

    #[test]
    fn test_find_best_variant_with_mock_benchmark() {
        // Mock benchmark: pretend block_size=128 is fastest
        let params = vec![("block_size".to_string(), vec![64, 128, 256])];

        let ptx_gen = |variant: &Variant| -> Result<String, String> {
            Ok(format!("// PTX for {:?}", variant))
        };

        let benchmark =
            |_ptx: &str, _name: &str, variant: &Variant| -> Result<BenchmarkResult, String> {
                let block_size = variant[0].1;
                // Simulate: 128 is fastest, 64 medium, 256 slowest
                let median = match block_size {
                    64 => 1.5,
                    128 => 0.8,
                    256 => 2.1,
                    _ => 10.0,
                };
                Ok(BenchmarkResult {
                    variant: variant.clone(),
                    median_ms: median,
                    min_ms: median * 0.9,
                    max_ms: median * 1.1,
                })
            };

        // Use a unique cache hash to avoid collisions
        let hash = "test_mock_bench_001";
        // Clean up any previous cache entry
        let path = cache_dir().join(format!("test_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();

        let winner = find_best_variant(
            "test_kernel",
            &params,
            hash,
            &dev("MockGPU"),
            &ptx_gen,
            &benchmark,
        )
            .expect("should find a winner");

        assert_eq!(
            winner,
            vec![("block_size".to_string(), 128)],
            "block_size=128 should win (lowest median)"
        );

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_find_best_variant_cache_hit() {
        // Pre-populate cache, then verify find_best_variant returns cached result
        let hash = "test_cache_hit_002";
        let result = AutotuneResult {
            winner: vec![("tile_k".to_string(), 32)],
            all_timings: vec![(vec![("tile_k".to_string(), 32)], 0.5)],
            device: dev("MockGPU"),
            selection: SelectionMethod::Measured,
        };
        write_cache(hash, "cached_kernel", &result);

        let params = vec![("tile_k".to_string(), vec![16, 32, 64])];
        let ptx_gen = |_: &Variant| -> Result<String, String> {
            panic!("should not be called — cache hit");
        };
        let benchmark = |_: &str, _: &str, _: &Variant| -> Result<BenchmarkResult, String> {
            panic!("should not be called — cache hit");
        };

        let winner = find_best_variant(
            "cached_kernel",
            &params,
            hash,
            &dev("MockGPU"),
            &ptx_gen,
            &benchmark,
        )
            .expect("cache hit should succeed");

        assert_eq!(winner, vec![("tile_k".to_string(), 32)]);

        // Clean up
        let path = cache_dir().join(format!("cached_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_find_best_variant_all_fail_falls_back() {
        let params = vec![("x".to_string(), vec![1, 2, 3])];
        let hash = "test_all_fail_003";
        let path = cache_dir().join(format!("fail_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();

        let ptx_gen = |_: &Variant| -> Result<String, String> {
            Err("compile error".to_string()) // all variants fail
        };
        let benchmark = |_: &str, _: &str, _: &Variant| -> Result<BenchmarkResult, String> {
            panic!("should not be called — PTX gen fails");
        };

        let winner = find_best_variant(
            "fail_kernel",
            &params,
            hash,
            &dev("MockGPU"),
            &ptx_gen,
            &benchmark,
        )
            .expect("should fall back to median");

        // Median of [1,2,3] is index 1 => value 2
        assert_eq!(winner, vec![("x".to_string(), 2)]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_find_best_variant_skips_slow_variants() {
        let params = vec![("size".to_string(), vec![1, 2])];
        let hash = "test_timeout_004";
        let path = cache_dir().join(format!("timeout_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();

        let ptx_gen = |_: &Variant| -> Result<String, String> { Ok("// ptx".to_string()) };
        let benchmark = |_: &str, _: &str, variant: &Variant| -> Result<BenchmarkResult, String> {
            let size = variant[0].1;
            let median = if size == 1 { 10000.0 } else { 0.5 }; // size=1 is too slow
            Ok(BenchmarkResult {
                variant: variant.clone(),
                median_ms: median,
                min_ms: median,
                max_ms: median,
            })
        };

        let winner = find_best_variant(
            "timeout_kernel",
            &params,
            hash,
            &dev("MockGPU"),
            &ptx_gen,
            &benchmark,
        )
            .expect("should pick fast variant");

        assert_eq!(
            winner,
            vec![("size".to_string(), 2)],
            "should skip the 10-second variant and pick size=2"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_benchmark_report_format() {
        let results = vec![
            BenchmarkResult {
                variant: vec![("bs".to_string(), 128)],
                median_ms: 0.8,
                min_ms: 0.7,
                max_ms: 0.9,
            },
            BenchmarkResult {
                variant: vec![("bs".to_string(), 256)],
                median_ms: 1.2,
                min_ms: 1.0,
                max_ms: 1.5,
            },
        ];

        // Just verify it doesn't panic
        print_benchmark_report("test_report_kernel", &results);
    }

    // ── Cost-model autotune tests ────────────────────────────────────

    #[test]
    fn test_cost_model_picks_largest_block_factor() {
        // The cost estimator returns lower cost for larger block factors,
        // so block_size=256 should win.
        let params = vec![("block_size".to_string(), vec![64, 128, 256])];

        let ptx_gen = |variant: &Variant| -> Result<String, String> {
            Ok(format!("// PTX for {:?}", variant))
        };

        // Cost estimator: lower cost for larger block_size (inverse relationship)
        let cost_est = |variant: &Variant| -> Result<f64, String> {
            let bs = variant[0].1 as f64;
            // Simulate: larger block_size = faster (lower time)
            Ok(1000.0 / bs)
        };

        let hash = "test_cost_model_001";
        let path = cache_dir().join(format!("cost_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();

        let winner =
            find_best_variant_cost_model(
                "cost_kernel",
                &params,
                hash,
                &dev("MockGPU"),
                "MockSpec",
                false, &ptx_gen, &cost_est)
                .expect("should find a winner");

        assert_eq!(
            winner,
            vec![("block_size".to_string(), 256)],
            "block_size=256 should win (lowest cost estimate)"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_cost_model_fresh_skips_cache() {
        let params = vec![("x".to_string(), vec![1, 2, 3])];
        let hash = "test_cost_fresh_002";

        // Pre-populate cache with x=1 as winner
        let result = AutotuneResult {
            winner: vec![("x".to_string(), 1)],
            all_timings: vec![(vec![("x".to_string(), 1)], 0.1)],
            device: dev("MockGPU"),
            selection: SelectionMethod::CostModel {
                spec: "MockSpec".to_string(),
            },
        };
        write_cache(hash, "fresh_kernel", &result);

        let ptx_gen =
            |variant: &Variant| -> Result<String, String> { Ok(format!("// PTX {:?}", variant)) };

        // Cost model says x=3 is the best
        let cost_est = |variant: &Variant| -> Result<f64, String> {
            let x = variant[0].1 as f64;
            Ok(100.0 / x) // x=3 => 33.3, x=1 => 100.0
        };

        // Without fresh: should return cached winner (x=1)
        let cached_winner =
            find_best_variant_cost_model(
                "fresh_kernel",
                &params,
                hash,
                &dev("MockGPU"),
                "MockSpec",
                false, &ptx_gen, &cost_est)
                .expect("cache hit");
        assert_eq!(cached_winner, vec![("x".to_string(), 1)]);

        // With fresh=true: should re-evaluate and pick x=3
        let fresh_winner =
            find_best_variant_cost_model(
                "fresh_kernel",
                &params,
                hash,
                &dev("MockGPU"),
                "MockSpec",
                true, &ptx_gen, &cost_est)
                .expect("fresh evaluation");
        assert_eq!(fresh_winner, vec![("x".to_string(), 3)]);

        // Clean up
        let path = cache_dir().join(format!("fresh_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_cost_model_skips_failed_ptx() {
        // If PTX generation fails for some variants, they are skipped
        let params = vec![("size".to_string(), vec![32, 64, 128])];

        let ptx_gen = |variant: &Variant| -> Result<String, String> {
            let size = variant[0].1;
            if size == 32 {
                Err("too small for shared memory".to_string())
            } else {
                Ok(format!("// PTX for size={}", size))
            }
        };

        let cost_est = |variant: &Variant| -> Result<f64, String> {
            let size = variant[0].1 as f64;
            Ok(1000.0 / size)
        };

        let hash = "test_cost_skip_003";
        let path = cache_dir().join(format!("skip_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();

        let winner =
            find_best_variant_cost_model(
                "skip_kernel",
                &params,
                hash,
                &dev("MockGPU"),
                "MockSpec",
                false, &ptx_gen, &cost_est)
                .expect("should skip size=32 and pick from 64,128");

        // size=128 wins (1000/128 < 1000/64)
        assert_eq!(winner, vec![("size".to_string(), 128)]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_cost_model_all_fail_falls_back() {
        let params = vec![("y".to_string(), vec![10, 20, 30])];
        let hash = "test_cost_allfail_004";
        let path = cache_dir().join(format!("allfail_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();

        let ptx_gen = |_: &Variant| -> Result<String, String> { Err("all fail".to_string()) };
        let cost_est = |_: &Variant| -> Result<f64, String> {
            panic!("should not be called if PTX gen fails");
        };

        let winner = find_best_variant_cost_model(
                "allfail_kernel",
                &params,
                hash,
                &dev("MockGPU"),
                "MockSpec",
                false,
            &ptx_gen,
            &cost_est,
        )
        .expect("should fall back to median");

        // Median of [10,20,30] => index 1 => 20
        assert_eq!(winner, vec![("y".to_string(), 20)]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_cost_model_two_params_cartesian() {
        // Test with two tuning parameters to verify Cartesian product works
        let params = vec![
            ("block_m".to_string(), vec![32, 64]),
            ("block_n".to_string(), vec![32, 64]),
        ];

        let ptx_gen = |variant: &Variant| -> Result<String, String> {
            Ok(format!("// PTX for {:?}", variant))
        };

        // Cost: product of params. (64, 64) => 4096 => lowest cost
        let cost_est = |variant: &Variant| -> Result<f64, String> {
            let product: f64 = variant.iter().map(|(_, v)| *v as f64).product();
            Ok(100000.0 / product)
        };

        let hash = "test_cost_2d_005";
        let path = cache_dir().join(format!("twopar_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();

        let winner = find_best_variant_cost_model(
                "twopar_kernel",
                &params,
                hash,
                &dev("MockGPU"),
                "MockSpec",
                false,
            &ptx_gen,
            &cost_est,
        )
        .expect("should find winner");

        assert_eq!(
            winner,
            vec![("block_m".to_string(), 64), ("block_n".to_string(), 64)],
            "(64,64) should win"
        );

        std::fs::remove_file(&path).ok();
    }
}
