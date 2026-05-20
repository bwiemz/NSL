//! Generator for WGGO Phase 2 merge-gate safetensors fixtures.
//!
//! Produces:
//!   tests/fixtures/wggo_calib_data.safetensors    — [8, 4, 32] f32 (key: "calibration")
//!   tests/fixtures/wggo_calib_weights.safetensors — four [32, 32] f32 weight tensors
//!
//! All values are deterministic (arithmetic sequences / lcg-style) so the files
//! are reproducible bit-for-bit on any platform.  Bytes are little-endian f32 as
//! mandated by the safetensors format specification.
//!
//! Usage (from repo root):
//!   cargo run --features calibrate --bin build_wggo_fixtures -- [--output-dir DIR]
//!
//! If --output-dir is omitted the files are written to tests/fixtures/ relative
//! to the repo root (the conventional location for committed fixture data).

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut output_dir: Option<String> = None;
    let mut positional: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --output-dir requires a value");
                    std::process::exit(2);
                }
                output_dir = Some(args[i].clone());
                i += 1;
            }
            other => {
                if other.starts_with("--") {
                    eprintln!("error: unknown flag: {other}");
                    std::process::exit(2);
                }
                positional.push(other.to_string());
                i += 1;
            }
        }
    }

    let out_dir_string = match (output_dir, positional.first()) {
        (Some(d), _) => d,
        (None, Some(d)) => d.clone(),
        (None, None) => {
            // Default: tests/fixtures/ relative to the manifest dir (repo root).
            // CARGO_MANIFEST_DIR points to the crate dir; go up two levels to reach
            // the workspace root, then into tests/fixtures/.
            let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
            PathBuf::from(&manifest)
                .join("../..")
                .join("tests/fixtures")
                .to_string_lossy()
                .into_owned()
        }
    };

    let out_dir = Path::new(&out_dir_string);
    std::fs::create_dir_all(out_dir).unwrap_or_else(|e| {
        eprintln!(
            "error: could not create output directory {}: {e}",
            out_dir.display()
        );
        std::process::exit(1);
    });

    write_calib_data(out_dir);
    write_calib_weights(out_dir);
}

// ── Calibration data ────────────────────────────────────────────────────────

/// Writes wggo_calib_data.safetensors — shape [8, 4, 32] f32.
///
/// Values: i / 1024.0 for i in 0..1024, deterministic, non-trivial variation
/// across the channel dimension to exercise AWQ scale estimation.
fn write_calib_data(out_dir: &Path) {
    let shape: Vec<usize> = vec![8, 4, 32];
    let numel: usize = 8 * 4 * 32; // 1024
    let data: Vec<f32> = (0..numel).map(|i| i as f32 / 1024.0).collect();

    // little-endian f32; required by the safetensors format specification
    let bytes: Vec<u8> = data.iter().flat_map(|x| x.to_le_bytes()).collect();

    let view = TensorView::new(Dtype::F32, shape, &bytes).unwrap();
    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    tensors.insert("calibration".into(), view);

    let serialized = serialize(&tensors, &None).unwrap();
    let path = out_dir.join("wggo_calib_data.safetensors");
    std::fs::write(&path, &serialized).unwrap();
    eprintln!("wrote {} ({} bytes)", path.display(), serialized.len());
}

// ── Calibration weights ─────────────────────────────────────────────────────

/// Writes wggo_calib_weights.safetensors — four [32, 32] f32 weight tensors.
///
/// Each projection uses a distinct arithmetic seed so the tensors are non-equal
/// but still fully deterministic:
///   AttentionMLP.q_proj  — seed 1
///   AttentionMLP.k_proj  — seed 2
///   AttentionMLP.v_proj  — seed 3
///   AttentionMLP.o_proj  — seed 4
///
/// Value formula: ((i * seed) % 1024) as f32 / 1024.0
fn write_calib_weights(out_dir: &Path) {
    let projections: &[(&str, u32)] = &[
        ("AttentionMLP.q_proj", 1),
        ("AttentionMLP.k_proj", 2),
        ("AttentionMLP.v_proj", 3),
        ("AttentionMLP.o_proj", 4),
    ];
    let shape: Vec<usize> = vec![32, 32];
    let numel: usize = 32 * 32;

    // Pre-compute bytes for each projection before building the TensorView map
    // (avoids dangling slice references across map insertion).
    let all_bytes: Vec<Vec<u8>> = projections
        .iter()
        .map(|(_, seed)| {
            let vals: Vec<f32> = (0..numel)
                .map(|i| ((i as u32).wrapping_mul(*seed) % 1024) as f32 / 1024.0)
                .collect();
            // little-endian f32; required by the safetensors format specification
            vals.iter().flat_map(|x| x.to_le_bytes()).collect()
        })
        .collect();

    let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
    for ((name, _), bytes) in projections.iter().zip(all_bytes.iter()) {
        let view = TensorView::new(Dtype::F32, shape.clone(), bytes.as_slice()).unwrap();
        tensors.insert((*name).to_string(), view);
    }

    let serialized = serialize(&tensors, &None).unwrap();
    let path = out_dir.join("wggo_calib_weights.safetensors");
    std::fs::write(&path, &serialized).unwrap();
    eprintln!("wrote {} ({} bytes)", path.display(), serialized.len());
}
