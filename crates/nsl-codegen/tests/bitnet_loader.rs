//! BitNet safetensors loader smoke test (cross-platform).
//!
//! The full end-to-end loader exercise requires the HF checkpoint
//! (~13 GB across 3 shards, Linux/macOS bash fetch script). This test
//! verifies basic plumbing only; merge-gate validation is in Task 10's
//! bitnet_logit_match.rs (#[ignore]'d, Linux CI only).

use nsl_codegen::bitnet::loader::{
    load_bitnet_b158_safetensors, read_pinned_revision, LoaderError,
};
use std::path::PathBuf;

#[test]
fn loader_handles_missing_file() {
    let path = PathBuf::from("/nonexistent/path/bitnet.safetensors");
    let result = load_bitnet_b158_safetensors(&path);
    assert!(result.is_err(), "missing file must return error");
    match result {
        Err(LoaderError::Io(_)) => {}
        Err(other) => panic!("expected Io error, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

#[test]
fn loader_handles_corrupt_safetensors() {
    use std::io::Write;
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("corrupt.safetensors");
    let mut file = std::fs::File::create(&path).unwrap();
    file.write_all(b"not a safetensors file").unwrap();
    drop(file);
    let result = load_bitnet_b158_safetensors(&path);
    assert!(result.is_err(), "corrupt safetensors must return error");
    match result {
        Err(LoaderError::Safetensors(_)) => {}
        Err(other) => panic!("expected Safetensors error, got {other:?}"),
        Ok(_) => panic!("expected error"),
    }
}

// ---------------------------------------------------------------------------
// Storage-dtype coverage
//
// REGRESSION GUARD. The loader originally hard-required `Dtype::F16` and
// returned `UnsupportedDtype` for anything else. The pinned checkpoint
// `1bitLLM/bitnet_b1_58-3B@af89e318` stores all 314 of its tensors as **F32**
// (its config.json says "torch_dtype": "float16", which is simply not what is
// on disk), so the loader could never read a single tensor from the checkpoint
// it was written for. Nothing caught it: the only loader tests covered missing
// and corrupt files, and the end-to-end gate was `#[ignore]`'d.
// ---------------------------------------------------------------------------

/// Build a minimal in-memory safetensors file containing one tensor.
///
/// Layout is `u64 header_len (LE) || header JSON || tensor data`.
fn synth_safetensors(name: &str, dtype: &str, shape: &[usize], data: &[u8]) -> Vec<u8> {
    let header = format!(
        r#"{{"{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[0,{}]}}}}"#,
        data.len()
    );
    let mut out = Vec::new();
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(data);
    out
}

fn write_temp(bytes: &[u8]) -> (tempfile::TempDir, PathBuf) {
    use std::io::Write;
    let dir = tempfile::tempdir().expect("create tempdir");
    let path = dir.path().join("weights.safetensors");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(bytes).unwrap();
    drop(f);
    (dir, path)
}

/// absmean = (2 + 0 + 0.2 + 1 + 1 + 2 + 0.2 + 1) / 8 = 0.925, so w/scale is
/// [-2.162, 0, 0.216, 1.081, -1.081, 2.162, -0.216, 1.081], clipped to [-1, 1]
/// and rounded to [-1, 0, 0, 1, -1, 1, 0, 1].
const VALUES: [f32; 8] = [-2.0, 0.0, 0.2, 1.0, -1.0, 2.0, -0.2, 1.0];
const EXPECTED_TRITS: [i8; 8] = [-1, 0, 0, 1, -1, 1, 0, 1];
const EXPECTED_SCALE: f32 = 0.925;

fn expected_packed() -> Vec<u8> {
    // HIGH-BITS-FIRST, code = trit + 1.
    EXPECTED_TRITS
        .chunks(4)
        .map(|g| {
            let c = |t: i8| (t + 1) as u8;
            (c(g[0]) << 6) | (c(g[1]) << 4) | (c(g[2]) << 2) | c(g[3])
        })
        .collect()
}

#[test]
fn loader_accepts_f32_storage_matching_the_pinned_checkpoint() {
    let data: Vec<u8> = VALUES.iter().flat_map(|v| v.to_le_bytes()).collect();
    let bytes = synth_safetensors(
        "model.layers.0.mlp.up_proj.weight",
        "F32",
        &[2, 4],
        &data,
    );
    let (_dir, path) = write_temp(&bytes);

    let loaded = load_bitnet_b158_safetensors(&path)
        .expect("F32 is the pinned checkpoint's actual storage dtype and must load");
    assert_eq!(loaded.len(), 1, "the one BitLinear tensor must be returned");
    assert_eq!(loaded[0].shape, vec![2, 4]);
    assert!(
        (loaded[0].scale - EXPECTED_SCALE).abs() < 1e-5,
        "absmean scale should be {EXPECTED_SCALE}, got {}",
        loaded[0].scale
    );
    assert_eq!(loaded[0].packed_bytes, expected_packed());
}

#[test]
fn loader_accepts_f16_storage() {
    let data: Vec<u8> = VALUES
        .iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect();
    let bytes = synth_safetensors(
        "model.layers.0.self_attn.q_proj.weight",
        "F16",
        &[2, 4],
        &data,
    );
    let (_dir, path) = write_temp(&bytes);

    let loaded = load_bitnet_b158_safetensors(&path).expect("F16 must still load");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].packed_bytes, expected_packed());
}

#[test]
fn loader_rejects_genuinely_unsupported_dtype() {
    let data = vec![0u8; 8 * 8]; // 8 x i64
    let bytes = synth_safetensors(
        "model.layers.0.mlp.down_proj.weight",
        "I64",
        &[2, 4],
        &data,
    );
    let (_dir, path) = write_temp(&bytes);

    match load_bitnet_b158_safetensors(&path) {
        Err(LoaderError::UnsupportedDtype { dtype, .. }) => {
            assert!(dtype.contains("I64"), "error must name the dtype, got {dtype}");
        }
        other => panic!("expected UnsupportedDtype for I64, got {other:?}"),
    }
}

#[test]
fn read_pinned_revision_returns_pi2_values() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let (model_id, revision) =
        read_pinned_revision(&repo_root).expect("read pinned revision");
    assert_eq!(model_id, "1bitLLM/bitnet_b1_58-3B",
               "model_id must match PI.2's pinned value");
    assert_eq!(revision.len(), 40,
               "revision SHA must be 40 hex chars");
    assert!(revision.chars().all(|c| c.is_ascii_hexdigit()),
            "revision SHA must be hex");
    assert_eq!(revision, "af89e318d78a70802061246bf037199d2fb97020",
               "revision SHA must match PI.2's pinned value");
}
