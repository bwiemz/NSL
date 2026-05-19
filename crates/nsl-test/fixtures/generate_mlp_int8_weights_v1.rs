//! Generates mlp_int8_weights_v1.bin + mlp_int8_weights_v1.toml.
//! Deterministic: running this binary multiple times produces bit-identical
//! output. Source of truth for the fixture; the .bin is a build artifact.
//!
//! Weight ranges per M57 v1 spec §1 + §4.6:
//!   W1, W2: i8 in [-64, 64]
//!   b1:     i32 in [-2^20, 2^20]
//!   b2:     i64 in [-2^30, 2^30]

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};

const GENERATOR_SEED: u64 = 0xC0DEFA11;
const NSL_VERSION: &str = "0.9.0";

// Block kind codes per fixture binary format spec §6.2
const KIND_WEIGHT: u32 = 0;
const KIND_BIAS: u32 = 1;

// Dtype codes per fixture binary format spec §6.2
const DTYPE_I8: u32 = 0;
const DTYPE_I32: u32 = 2;
const DTYPE_I64: u32 = 3;

fn write_file_header(bin: &mut Vec<u8>, n_blocks: u32) {
    bin.extend_from_slice(b"NSLF");             // magic
    bin.extend_from_slice(&1u32.to_le_bytes()); // format_version = 1
    bin.extend_from_slice(&n_blocks.to_le_bytes());
}

/// Write a 48-byte block header + raw LE data.
fn write_block(
    bin: &mut Vec<u8>,
    kind: u32,
    dtype: u32,
    dtype_size: usize,
    layer: u32,
    shape: &[u64],
    data: &[u8],
) {
    // Per spec: rank <= 4; shape is zero-padded to 4 dims.
    assert!(shape.len() <= 4);
    assert_eq!(data.len(), shape.iter().product::<u64>() as usize * dtype_size);

    bin.extend_from_slice(&kind.to_le_bytes());
    bin.extend_from_slice(&dtype.to_le_bytes());
    bin.extend_from_slice(&layer.to_le_bytes());
    bin.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    // 4 × u64 shape slots, zero-padded
    for i in 0..4 {
        let dim = if i < shape.len() { shape[i] } else { 0u64 };
        bin.extend_from_slice(&dim.to_le_bytes());
    }
    // Header is exactly 4+4+4+4 + 4*8 = 48 bytes (verified by the
    // per-block size assertions above; no runtime check needed here).
    bin.extend_from_slice(data);
}

fn i8_to_bytes(v: &[i8]) -> Vec<u8> {
    v.iter().map(|&x| x as u8).collect()
}

fn i32_to_bytes(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn i64_to_bytes(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn build_manifest_toml(hash_hex: &str) -> String {
    format!(
        r#"sha256                    = "{hash_hex}"
generated_by              = "generate_mlp_int8_weights_v1.rs"
generator_seed            = "0xC0DEFA11"
nsl_version_at_generation = "{NSL_VERSION}"

[meta]
description       = "v1 MLP fixture for M57 Tier-A Verilator + Yosys gates"
overflow_analysis = "see Section 4.6 of M57 v1 design spec"
spec_reference    = "docs/superpowers/specs/2026-05-18-m57-fpga-verilog-design.md"

[[block]]
name        = "W1"
kind        = "weight"
dtype       = "i8"
layer       = 1
shape       = [784, 128]
value_range = [-64, 64]

[[block]]
name        = "b1"
kind        = "bias"
dtype       = "i32"
layer       = 1
shape       = [128]
value_range = [-1048576, 1048576]

[[block]]
name        = "W2"
kind        = "weight"
dtype       = "i8"
layer       = 2
shape       = [128, 10]
value_range = [-64, 64]

[[block]]
name        = "b2"
kind        = "bias"
dtype       = "i64"
layer       = 2
shape       = [10]
value_range = [-1073741824, 1073741824]
"#
    )
}

fn main() {
    let mut rng = ChaCha20Rng::seed_from_u64(GENERATOR_SEED);

    // Layer 1: W1 ∈ [-64, 64] i8, b1 ∈ [-2^20, 2^20] i32
    let w1: Vec<i8> = (0..784 * 128)
        .map(|_| rng.random_range(-64i8..=64))
        .collect();
    let b1: Vec<i32> = (0..128)
        .map(|_| rng.random_range(-(1i32 << 20)..=(1i32 << 20)))
        .collect();
    // Layer 2: W2 ∈ [-64, 64] i8, b2 ∈ [-2^30, 2^30] i64
    let w2: Vec<i8> = (0..128 * 10)
        .map(|_| rng.random_range(-64i8..=64))
        .collect();
    let b2: Vec<i64> = (0..10)
        .map(|_| rng.random_range(-(1i64 << 30)..=(1i64 << 30)))
        .collect();

    let mut bin: Vec<u8> = Vec::with_capacity(110_000);
    write_file_header(&mut bin, 4);
    write_block(&mut bin, KIND_WEIGHT, DTYPE_I8,  1, 1, &[784, 128], &i8_to_bytes(&w1));
    write_block(&mut bin, KIND_BIAS,   DTYPE_I32, 4, 1, &[128],      &i32_to_bytes(&b1));
    write_block(&mut bin, KIND_WEIGHT, DTYPE_I8,  1, 2, &[128, 10],  &i8_to_bytes(&w2));
    write_block(&mut bin, KIND_BIAS,   DTYPE_I64, 8, 2, &[10],       &i64_to_bytes(&b2));

    let expected_size = 12 + 100_400 + 560 + 1_328 + 128;
    assert_eq!(
        bin.len(),
        expected_size,
        "fixture size mismatch: got {} bytes, expected {expected_size}",
        bin.len()
    );

    let hash_hex = sha256_hex(&bin);

    // Write to crates/nsl-test/fixtures/ relative to CARGO_MANIFEST_DIR
    // When invoked via `cargo run -p nsl-test --bin generate_mlp_int8_weights_v1`
    // from the workspace root, CARGO_MANIFEST_DIR points to crates/nsl-test.
    let out_dir = std::path::PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| ".".to_string()),
    )
    .join("fixtures");

    std::fs::create_dir_all(&out_dir).expect("create fixtures dir");

    let bin_path = out_dir.join("mlp_int8_weights_v1.bin");
    let toml_path = out_dir.join("mlp_int8_weights_v1.toml");

    std::fs::write(&bin_path, &bin).expect("write mlp_int8_weights_v1.bin");
    std::fs::write(&toml_path, build_manifest_toml(&hash_hex))
        .expect("write mlp_int8_weights_v1.toml");

    println!("Wrote {}", bin_path.display());
    println!("  size : {} bytes", bin.len());
    println!("  sha256: {hash_hex}");
    println!("Wrote {}", toml_path.display());
}
