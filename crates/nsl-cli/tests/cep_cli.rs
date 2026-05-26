//! Task 9: end-to-end CEP CLI integration tests.
//!
//! Exercises the now-complete CEP frontend pipeline through the `nsl` binary:
//!
//!   1. `build --cep-prune` without `--weights` fails fast (the weights gate
//!      fires before any analysis, so the fixture need only be syntactically
//!      valid — but we use the canonical analyzable fixture anyway).
//!   2. `build --cep-prune --weights <w>` on the canonical analyzable fixture
//!      runs the full pipeline (frontend analysis → recognizer → cross-check →
//!      prune oracle), prints the Pruning Report, and writes a versioned delta
//!      JSON (`cep_version == 1`, `mode == "prune"`).
//!   3. `build --cep-prune` on `examples/gpt2.nsl` fails (gpt2 carries no
//!      `@cep_prune` decorator and its fused-QKV attention is unrecognized).
//!   4. `check --cep-search` on the searchable fixture runs the search oracle,
//!      prints the Architecture Search Report, and writes a delta JSON
//!      (`mode == "search"`).
//!
//! These spawn the real `nsl` binary, so they set `NSL_STDLIB_PATH` exactly as
//! `cpdt_cli.rs` does — the full frontend (`frontend_with_flags`) analyzes the
//! source and `process::exit(1)`s on any error-level diagnostic, so the happy
//! paths require fixtures that actually type-check.

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

fn searchable_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_searchable.nsl")
}

/// Write a safetensors file whose tensor shapes match the canonical fixture
/// (d_model=64, 2 layers, 4 heads, 2 kv heads, head_dim=16, d_ff=128). The
/// CEP cross-check inspects `blocks.{l}.attn.wq` (= n_heads·head_dim = 64),
/// `.wk`/`.wv` (= n_kv_heads·head_dim = 32), and `blocks.{l}.ffn.w_gate`/`.w_up`
/// (= d_ff = 128), each against d_model = 64. All-zero f32 data is fine — the
/// shapes are the only thing cross-checked.
fn write_matching_safetensors(path: &Path) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    // Keep the backing byte buffers alive for the lifetime of the views.
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
fn cep_prune_without_weights_errors() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build").arg(canonical_fixture()).arg("--cep-prune");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--weights"));
}

#[test]
fn cep_prune_end_to_end_writes_delta() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    let out_path = tmp.path().join("delta.cep.json");
    write_matching_safetensors(&weights_path);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--weights")
        .arg(&weights_path)
        .arg("--cep-out")
        .arg(&out_path);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("=== CEP Pruning Report ==="))
        .stdout(predicate::str::contains("CEP delta written"));

    let delta = fs::read_to_string(&out_path).expect("delta JSON written");
    let v: serde_json::Value = serde_json::from_str(&delta).expect("delta is valid JSON");
    assert_eq!(v["cep_version"], 1);
    assert_eq!(v["mode"], "prune");
}

#[test]
fn cep_prune_refuses_gpt2() {
    let tmp = TempDir::new().unwrap();
    let weights_path = tmp.path().join("weights.safetensors");
    write_matching_safetensors(&weights_path);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(workspace_root().join("examples/gpt2.nsl"))
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&weights_path);
    // gpt2 has no @cep_prune decorator and its fused-QKV attention is not the
    // canonical GQA structure — either way the prune path must refuse.
    cmd.assert().failure();
}

#[test]
fn cep_search_end_to_end_writes_delta() {
    let tmp = TempDir::new().unwrap();
    let out_path = tmp.path().join("search.cep.json");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("check")
        .arg(searchable_fixture())
        .arg("--cep-search")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--cep-out")
        .arg(&out_path);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains(
            "=== CEP Architecture Search Report ===",
        ))
        .stdout(predicate::str::contains("CEP delta written"));

    let delta = fs::read_to_string(&out_path).expect("delta JSON written");
    let v: serde_json::Value = serde_json::from_str(&delta).expect("delta is valid JSON");
    assert_eq!(v["mode"], "search");
}
