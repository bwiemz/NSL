//! @autotune: build-time kernel variant benchmarking with caching.
//!
//! Generates Cartesian products of tuning parameters, hashes kernel ASTs
//! for cache key generation, and provides read/write for the `.nsl-cache/autotune/`
//! directory.
//!
//! # What actually runs today
//!
//! The production compile path (`compiler::kernel::autotune_select_best`) calls
//! [`find_best_variant_cost_model`], which **estimates** each variant with the
//! roofline model. It never launches anything.
//!
//! [`find_best_variant`] — the real-measurement path — takes a [`BenchmarkFn`]
//! that the runtime would supply (CUDA-event timing). **No such callback exists
//! anywhere in the tree**: `find_best_variant` is reached only from this
//! module's own unit tests, which pass a synthetic closure. Wiring it is the
//! measured half of roadmap item 10. Until then a cache entry records
//! [`SelectionMethod::CostModel`] and says which `GpuSpec` it was priced
//! against, so nobody mistakes an estimate for a measurement.
//! `autotune_selection_method_is_honest` in
//! `tests/autotune_cache_identity.rs` fails if that stops being true.
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
pub const CACHE_SCHEMA_VERSION: u32 = 2;

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
/// **Nothing implements this.** The intended implementation belongs in the
/// runtime — load the PTX, allocate dummy data, launch with CUDA events, return
/// a `BenchmarkResult` — but no such function exists anywhere in the tree, so
/// the only callers are this module's unit tests passing synthetic closures.
/// Wiring it is the measured half of roadmap item 10; see the module header.
///
/// Arguments: (ptx_source, kernel_name, variant_params) -> Result<BenchmarkResult>
pub type BenchmarkFn = dyn Fn(&str, &str, &Variant) -> Result<BenchmarkResult, String>;

/// Number of warmup launches (not timed) before measured runs.
///
/// Advisory only — read by nothing, because nothing implements [`BenchmarkFn`].
/// It describes the protocol a future measured path should follow, not one this
/// compiler performs.
pub const WARMUP_RUNS: usize = 5;
/// Number of measured launches for computing median latency.
///
/// Advisory only, for the same reason as [`WARMUP_RUNS`].
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
    // 1. Check cache
    // `None`: this path measures, so there is no cost-model spec to invalidate
    // against. A Measured record stays valid until the device or key changes.
    if let Some(cached) = check_cache(kernel_name, cache_hash, device, None) {
        if std::env::var("NSL_AUTOTUNE_VERBOSE").is_ok() {
            eprintln!("[autotune] cache hit for {kernel_name}");
        }
        return Ok(cached);
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
    format!("{:x}", hasher.finalize())
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
    let path = cache_dir().join(format!("{}_{}.json", kernel_name, hash));
    let Ok(content) = std::fs::read_to_string(&path) else {
        return Ok(None);
    };
    let record: AutotuneCacheRecord = serde_json::from_str(&content)
        .map_err(|e| CacheReject::Unparseable(e.to_string()))?;
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
    {
        if spec != expected {
            return Err(CacheReject::CostModelSpecChanged {
                found: spec.clone(),
                expected: expected.to_string(),
            });
        }
    }
    if record.winner.is_empty() {
        return Err(CacheReject::EmptyWinner);
    }
    Ok(Some(record))
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
