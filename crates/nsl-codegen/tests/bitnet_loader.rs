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
