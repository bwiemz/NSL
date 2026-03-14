//! @autotune: build-time kernel variant benchmarking with caching.
//!
//! Generates Cartesian products of tuning parameters, hashes kernel ASTs
//! for cache key generation, and provides read/write for the `.nsl-cache/autotune/`
//! directory. Actual GPU benchmarking is deferred to M26 integration; for now,
//! the middle-value fallback is used.

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
pub fn select_middle_values(params: &TuningParams) -> Variant {
    params
        .iter()
        .map(|(name, values)| {
            let mid_idx = values.len() / 2;
            (name.clone(), values[mid_idx])
        })
        .collect()
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
}
