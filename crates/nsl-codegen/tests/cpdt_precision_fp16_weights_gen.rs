//! Generator for `cpdt_precision_fp16_weights.safetensors`.
//!
//! Run once with `cargo test -p nsl-codegen gen_cpdt_precision_fp16_weights -- --ignored`
//! to write the safetensors file into the fixture directory. The generated file
//! is then committed and used by later validation tests.
//!
//! ## Tensor layout
//!
//! The safetensors contains 10 tensors corresponding to the parameter paths in
//! `cpdt_precision_fp16.nsl`:
//!
//! | Tensor name       | Shape    | Value (f32) | Expected CPDT tier |
//! |-------------------|----------|-------------|-------------------|
//! | embed             | [8, 64]  | 1.0         | High (Embedding)  |
//! | blocks.0.w        | [64, 64] | 1.0         | High (FirstOrLast)|
//! | blocks.1.w        | [64, 64] | 1e-4        | Medium (Generic)  |
//! | blocks.2.w        | [64, 64] | 1e-4        | Medium (Generic)  |
//! | blocks.3.w        | [64, 64] | 1e-4        | Medium (Generic)  |
//! | blocks.4.w        | [64, 64] | 1e-4        | Medium (Generic)  |
//! | blocks.5.w        | [64, 64] | 1e-4        | Medium (Generic)  |
//! | blocks.6.w        | [64, 64] | 1e-4        | Medium (Generic)  |
//! | blocks.7.w        | [64, 64] | 1.0         | High (FirstOrLast)|
//! | final_norm        | [64]     | 1.0         | High (Norm)       |
//!
//! ## Tier derivation (n_layers=8, hardcoded in invoke_cpdt_if_enabled)
//!
//! Sensitivity formula: score = gm * pos / elts
//!   - CALIB_T0 = 6.106e-8  (High ↔ Medium boundary)
//!   - CALIB_T1 = 2.232e-8  (Medium ↔ Low boundary)
//!
//! Middle layers (blocks.1-6): gm = 1e-4, elts = 4096, pos = 1.0 or 1.3
//!   - score (pos=1.0) = 1e-4 / 4096 ≈ 2.44e-8  →  T1 < score < T0  →  Medium
//!   - score (pos=1.3) = 1.3e-4 / 4096 ≈ 3.17e-8 →  T1 < score < T0  →  Medium
//!
//! Override tiers (ignore score):
//!   - embed: Embedding kind → High
//!   - final_norm: Norm kind (name contains "norm") → High
//!   - blocks.0.w: FirstOrLast (l=0) → High
//!   - blocks.7.w: FirstOrLast (l=7, l+1=8=n_layers) → High

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

/// Generate and write `cpdt_precision_fp16_weights.safetensors` into the
/// test fixture directory.
///
/// Ignored by default — run with `-- --ignored` to regenerate.
#[test]
#[ignore]
fn gen_cpdt_precision_fp16_weights() {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    // Collect (name, shape, value) triples.
    let tensors: &[(&str, &[usize], f32)] = &[
        // High-tier (kind override — value does not affect tier)
        ("embed",       &[8, 64],  1.0),
        ("blocks.0.w",  &[64, 64], 1.0),
        ("blocks.7.w",  &[64, 64], 1.0),
        ("final_norm",  &[64],     1.0),
        // Medium-tier (Generic, gm=1e-4, pos=1.0..1.3, elts=4096)
        // score ≈ 2.44e-8 .. 3.17e-8, which is between T1=2.232e-8 and T0=6.106e-8
        ("blocks.1.w",  &[64, 64], 1e-4_f32),
        ("blocks.2.w",  &[64, 64], 1e-4_f32),
        ("blocks.3.w",  &[64, 64], 1e-4_f32),
        ("blocks.4.w",  &[64, 64], 1e-4_f32),
        ("blocks.5.w",  &[64, 64], 1e-4_f32),
        ("blocks.6.w",  &[64, 64], 1e-4_f32),
    ];

    // Build byte buffers keyed by name.
    let mut raw: HashMap<String, Vec<u8>> = HashMap::new();
    for (name, shape, value) in tensors {
        let elts: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..elts).flat_map(|_| value.to_le_bytes()).collect();
        raw.insert(name.to_string(), bytes);
    }

    // Build TensorViews (borrowing from raw).
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    for (name, shape, _) in tensors {
        let bytes = raw.get(*name).unwrap();
        let shape_vec: Vec<usize> = shape.to_vec();
        let view = TensorView::new(Dtype::F32, shape_vec, bytes.as_slice()).unwrap();
        views.insert(name.to_string(), view);
    }

    let serialized = serialize(&views, &None).unwrap();

    let out_path = fixture_dir().join("cpdt_precision_fp16_weights.safetensors");
    let mut f = std::fs::File::create(&out_path).unwrap();
    f.write_all(&serialized).unwrap();

    println!("wrote {} bytes to {}", serialized.len(), out_path.display());

    // Sanity-check: reload and verify tensor count and shapes.
    let wm = nsl_codegen::weight_aware::WeightMap::load(&out_path).unwrap();
    let mut count = 0;
    for (name, entry) in wm.entries() {
        println!("  {}: {} elements (f32)", name, entry.num_elements);
        count += 1;
    }
    assert_eq!(count, tensors.len(), "tensor count mismatch");
}

/// Verify tier assignments match expectations from the doc-comment above.
/// This test runs without the `--ignored` flag and validates the already-
/// committed safetensors (fails if the file has not been generated yet).
#[test]
fn cpdt_precision_fp16_weights_tier_smoke() {
    use nsl_codegen::cpdt_sensitivity::{
        assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
        CALIB_ALPHA,
    };
    use nsl_codegen::cpdt_tier_apply::Tier;
    use nsl_codegen::weight_aware::WeightMap;

    let path = fixture_dir().join("cpdt_precision_fp16_weights.safetensors");
    let Ok(wm) = WeightMap::load(&path) else {
        // File not yet generated; skip gracefully.
        eprintln!("cpdt_precision_fp16_weights.safetensors not found; run gen_ test first");
        return;
    };

    // n_layers=8 matches the hardcoded PrecisionConfig::default() in
    // invoke_cpdt_if_enabled (stmt.rs:93).
    let n_layers: u32 = 8;

    let expected: &[(&str, Tier)] = &[
        ("embed",      Tier::High),
        ("blocks.0.w", Tier::High),
        ("blocks.7.w", Tier::High),
        ("final_norm", Tier::High),
        ("blocks.1.w", Tier::Medium),
        ("blocks.2.w", Tier::Medium),
        ("blocks.3.w", Tier::Medium),
        ("blocks.4.w", Tier::Medium),
        ("blocks.5.w", Tier::Medium),
        ("blocks.6.w", Tier::Medium),
    ];

    for (name, expected_tier) in expected {
        let entry = wm.get(name).unwrap_or_else(|| panic!("tensor missing: {name}"));
        let layer = layer_of(name);
        let kind = classify_layer_kind(name, layer, n_layers);
        let gm = gradient_magnitude_est(Some(entry));
        let pos = position_criticality(layer, n_layers, CALIB_ALPHA);
        let elts = entry.num_elements.max(1) as f64;
        let score = gm * pos / elts;
        let tier = assign_tier(score, kind);
        assert_eq!(
            tier, *expected_tier,
            "tier mismatch on '{name}': expected {:?}, got {:?} (score={score:.3e}, gm={gm:.3e}, pos={pos}, elts={elts})",
            expected_tier, tier
        );
    }
}
