//! @autotune: build-time kernel variant benchmarking with caching.
//!
//! Generates Cartesian products of tuning parameters, hashes kernel ASTs
//! for cache key generation, and provides read/write for the `.nsl-cache/autotune/`
//! directory.
//!
//! GPU benchmarking: `find_best_variant()` accepts a `BenchmarkFn` callback that
//! the runtime provides (actual CUDA event timing). When no GPU is available or
//! `NSL_AUTOTUNE_FALLBACK=1` is set, `select_middle_values()` is used instead.

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
    pub device_name: String,
    pub compute_capability: String,
    pub sm_count: u32,
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
/// Panics if any parameter has an empty values list (semantic checker prevents this).
pub fn select_middle_values(params: &TuningParams) -> Variant {
    params
        .iter()
        .map(|(name, values)| {
            assert!(!values.is_empty(), "autotune param '{}' has empty value list", name);
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
/// The runtime provides this — it loads the PTX, allocates dummy data, launches
/// with CUDA events, and returns the `BenchmarkResult`.
///
/// Arguments: (ptx_source, kernel_name, variant_params) -> Result<BenchmarkResult>
pub type BenchmarkFn = dyn Fn(&str, &str, &Variant) -> Result<BenchmarkResult, String>;

/// Number of warmup launches (not timed) before measured runs.
pub const WARMUP_RUNS: usize = 5;
/// Number of measured launches for computing median latency.
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
    ptx_generator: &dyn Fn(&Variant) -> Result<String, String>,
    benchmark_fn: &BenchmarkFn,
) -> Result<Variant, String> {
    // 1. Check cache
    if let Some(cached) = check_cache(kernel_name, cache_hash) {
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
        all_timings: results.iter().map(|r| (r.variant.clone(), r.median_ms)).collect(),
        device_name: String::new(), // filled by runtime if available
        compute_capability: String::new(),
        sm_count: 0,
    };
    write_cache(cache_hash, kernel_name, &autotune_result);

    Ok(winner)
}

/// Print a formatted benchmark report table to stderr.
pub fn print_benchmark_report(kernel_name: &str, results: &[BenchmarkResult]) {
    eprintln!("\n=== Autotune Report: {kernel_name} ===");
    eprintln!("{:<40} {:>10} {:>10} {:>10}", "Variant", "Median", "Min", "Max");
    eprintln!("{:-<72}", "");
    for r in results {
        let params_str: String = r.variant
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ");
        let marker = if r.variant == results[0].variant { " <-- winner" } else { "" };
        eprintln!(
            "{:<40} {:>9.3}ms {:>9.3}ms {:>9.3}ms{}",
            params_str, r.median_ms, r.min_ms, r.max_ms, marker
        );
    }
    eprintln!();
}

/// Hash a kernel's AST body for cache key generation (SHA-256).
///
/// The hash incorporates the kernel name, serialised AST body, tuning parameter
/// definitions, input tensor shapes, and the target GPU's device name, compute
/// capability, and SM count. This ensures the cache is invalidated whenever any
/// of these change.
pub fn hash_kernel_ast(
    kernel_name: &str,
    ast_bytes: &[u8],
    tuning_params: &TuningParams,
    input_shapes: &[Vec<i64>],
    device_name: &str,
    compute_capability: &str,
    sm_count: u32,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(kernel_name.as_bytes());
    hasher.update(ast_bytes);
    for (name, values) in tuning_params {
        hasher.update(name.as_bytes());
        for v in values {
            hasher.update(v.to_le_bytes());
        }
    }
    for shape in input_shapes {
        for &dim in shape {
            hasher.update(dim.to_le_bytes());
        }
        hasher.update(b"|");
    }
    hasher.update(device_name.as_bytes());
    hasher.update(compute_capability.as_bytes());
    hasher.update(sm_count.to_le_bytes());
    format!("{:x}", hasher.finalize())
}

/// Returns the autotune cache directory path.
pub fn cache_dir() -> PathBuf {
    PathBuf::from(".nsl-cache/autotune")
}

/// Check whether a cached winner exists for the given kernel + hash.
pub fn check_cache(kernel_name: &str, hash: &str) -> Option<Variant> {
    let path = cache_dir().join(format!("{}_{}.json", kernel_name, hash));
    if !path.exists() {
        return None;
    }
    let content = std::fs::read_to_string(&path).ok()?;
    parse_winner_from_cache(&content)
}

/// Write an autotune result (winner + all timings) to the cache directory.
pub fn write_cache(hash: &str, kernel_name: &str, result: &AutotuneResult) {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).ok();

    let timings_json: Vec<String> = result
        .all_timings
        .iter()
        .map(|(variant, ms)| {
            let params: Vec<String> = variant
                .iter()
                .map(|(k, v)| format!(r#""{}": {}"#, k, v))
                .collect();
            format!(
                r#"    {{"params": {{{}}}, "median_ms": {:.4}}}"#,
                params.join(", "),
                ms
            )
        })
        .collect();

    let winner_json: Vec<String> = result
        .winner
        .iter()
        .map(|(k, v)| format!(r#""{}": {}"#, k, v))
        .collect();

    let timestamp = {
        use std::time::SystemTime;
        let secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{}", secs)
    };

    let winner_ms = result
        .all_timings
        .iter()
        .find(|(v, _)| v == &result.winner)
        .map(|(_, ms)| *ms)
        .unwrap_or(0.0);

    let json = format!(
        r#"{{
  "kernel": "{}",
  "device": "{}",
  "compute_capability": "{}",
  "sm_count": {},
  "variants_tested": {},
  "winner": {{{}}},
  "median_time_ms": {:.4},
  "all_timings": [
{}
  ],
  "timestamp": "{}",
  "cache_key": "{}"
}}"#,
        kernel_name,
        result.device_name,
        result.compute_capability,
        result.sm_count,
        result.all_timings.len(),
        winner_json.join(", "),
        winner_ms,
        timings_json.join(",\n"),
        timestamp,
        hash,
    );

    let path = dir.join(format!("{}_{}.json", kernel_name, hash));
    std::fs::write(&path, json).ok();
}

/// Parse the winner variant from a cached JSON file.
fn parse_winner_from_cache(json: &str) -> Option<Variant> {
    let winner_start = json.find("\"winner\":")?;
    let brace_start = json[winner_start..].find('{')? + winner_start;
    let brace_end = json[brace_start..].find('}')? + brace_start;
    let winner_str = &json[brace_start + 1..brace_end];

    let mut variant = Vec::new();
    for pair in winner_str.split(',') {
        let parts: Vec<&str> = pair.split(':').collect();
        if parts.len() == 2 {
            let key = parts[0].trim().trim_matches('"').to_string();
            let val: i64 = parts[1].trim().parse().ok()?;
            variant.push((key, val));
        }
    }

    if variant.is_empty() {
        None
    } else {
        Some(variant)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_hash_deterministic_and_sensitive() {
        let params = vec![("block_size".to_string(), vec![64, 128])];

        let hash1 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![256]], "GPU0", "8.6", 84);
        let hash2 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![256]], "GPU0", "8.6", 84);
        assert_eq!(hash1, hash2, "identical inputs must produce the same hash");

        // Different AST body
        let hash3 = hash_kernel_ast("my_kernel", b"body_v2", &params, &[vec![256]], "GPU0", "8.6", 84);
        assert_ne!(hash1, hash3, "different AST body must change hash");

        // Different input shapes
        let hash4 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![512]], "GPU0", "8.6", 84);
        assert_ne!(hash1, hash4, "different input shapes must change hash");

        // Different device
        let hash5 = hash_kernel_ast("my_kernel", b"body_v1", &params, &[vec![256]], "GPU1", "8.9", 128);
        assert_ne!(hash1, hash5, "different device must change hash");
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
            device_name: "TestGPU".to_string(),
            compute_capability: "8.6".to_string(),
            sm_count: 84,
        };

        let hash = "test_hash_roundtrip_12345";
        write_cache(hash, "test_kernel", &result);

        let cached = check_cache("test_kernel", hash);
        assert!(cached.is_some(), "cache file should be readable");
        let winner = cached.unwrap();
        assert_eq!(winner, vec![("block_size".to_string(), 256)]);

        // Cleanup
        let path = cache_dir().join(format!("test_kernel_{}.json", hash));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_check_cache_miss() {
        let cached = check_cache("nonexistent_kernel", "nonexistent_hash");
        assert!(cached.is_none());
    }

    // ── GPU benchmarking tests ────────────────────────────────────────

    #[test]
    fn test_find_best_variant_with_mock_benchmark() {
        // Mock benchmark: pretend block_size=128 is fastest
        let params = vec![
            ("block_size".to_string(), vec![64, 128, 256]),
        ];

        let ptx_gen = |variant: &Variant| -> Result<String, String> {
            Ok(format!("// PTX for {:?}", variant))
        };

        let benchmark = |_ptx: &str, _name: &str, variant: &Variant| -> Result<BenchmarkResult, String> {
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
            &ptx_gen,
            &benchmark,
        ).expect("should find a winner");

        assert_eq!(winner, vec![("block_size".to_string(), 128)],
            "block_size=128 should win (lowest median)");

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
            device_name: "MockGPU".to_string(),
            compute_capability: "8.0".to_string(),
            sm_count: 108,
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
            "cached_kernel", &params, hash, &ptx_gen, &benchmark,
        ).expect("cache hit should succeed");

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
            "fail_kernel", &params, hash, &ptx_gen, &benchmark,
        ).expect("should fall back to median");

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

        let ptx_gen = |_: &Variant| -> Result<String, String> {
            Ok("// ptx".to_string())
        };
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
            "timeout_kernel", &params, hash, &ptx_gen, &benchmark,
        ).expect("should pick fast variant");

        assert_eq!(winner, vec![("size".to_string(), 2)],
            "should skip the 10-second variant and pick size=2");

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
}
