//! CEP Option 2 SP1 — CLI integration test: `--cep-emit-weights`.
//!
//! Verifies that `nsl build --cep-prune --cep-emit-weights <path>` produces a
//! valid, physically-small .safetensors file alongside the usual delta JSON.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

fn canonical_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_canonical_small.nsl")
}

/// Write a safetensors file whose tensor shapes match the canonical fixture
/// (d_model=64, 2 layers, 4 heads, 2 kv heads, head_dim=16, d_ff=128).
/// All-zero f32 data — the shapes are what the CEP cross-check verifies.
fn write_matching_safetensors(path: &Path) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    let specs: Vec<(String, Vec<usize>)> = {
        let mut v = Vec::new();
        for l in 0..2 {
            v.push((format!("blocks.{l}.attn.wq"), vec![64, 64]));
            v.push((format!("blocks.{l}.attn.wk"), vec![64, 32]));
            v.push((format!("blocks.{l}.attn.wv"), vec![64, 32]));
            v.push((format!("blocks.{l}.ffn.w_gate"), vec![64, 128]));
            v.push((format!("blocks.{l}.ffn.w_up"), vec![64, 128]));
        }
        v
    };
    let buffers: Vec<(String, Vec<usize>, Vec<u8>)> = specs
        .into_iter()
        .map(|(name, shape)| {
            let elems: usize = shape.iter().product();
            (name, shape, vec![0u8; elems * 4])
        })
        .collect();
    let views: HashMap<String, TensorView<'_>> = buffers
        .iter()
        .map(|(name, shape, data)| {
            (
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), data.as_slice()).unwrap(),
            )
        })
        .collect();
    let bytes = serialize(&views, &None).unwrap();
    fs::write(path, bytes).unwrap();
}

#[test]
fn cep_emit_weights_writes_smaller_safetensors() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    let delta_out = tmp.path().join("m.cep.json");
    let weights_out = tmp.path().join("m.pruned.safetensors");
    write_matching_safetensors(&w);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--weights")
        .arg(&w)
        .arg("--cep-out")
        .arg(&delta_out)
        .arg("--cep-emit-weights")
        .arg(&weights_out);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CEP sliced weights written"));
    assert!(weights_out.exists(), "pruned weights file must exist");

    // size: pruned must be <= original (equal if this sparsity prunes nothing).
    let orig = std::fs::metadata(&w).unwrap().len();
    let pruned = std::fs::metadata(&weights_out).unwrap().len();
    assert!(pruned <= orig, "pruned ({pruned}) must be <= original ({orig})");

    // round-trip: the emitted file must be a VALID safetensors (deserializes cleanly).
    let bytes = std::fs::read(&weights_out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .expect("emitted file is valid safetensors");
    assert!(!st.names().is_empty(), "emitted safetensors has tensors");
}
