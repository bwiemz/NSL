//! CEP Mode 3 (paper §2.2) — CLI integration test: `--cep-joint`.
//!
//! Verifies that `nsl build --cep-joint --weights w` runs the joint prune-search
//! and emits the standard delta JSON alongside its report.

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

fn write_matching_safetensors(path: &Path) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    let specs: Vec<(String, Vec<usize>)> = {
        let mut v = Vec::new();
        for l in 0..2 {
            v.push((format!("blocks.{l}.attn.wq"), vec![64, 64]));
            v.push((format!("blocks.{l}.attn.wk"), vec![64, 32]));
            v.push((format!("blocks.{l}.attn.wv"), vec![64, 32]));
            v.push((format!("blocks.{l}.attn.wo"), vec![64, 64]));
            v.push((format!("blocks.{l}.ffn.w_gate"), vec![64, 128]));
            v.push((format!("blocks.{l}.ffn.w_up"), vec![64, 128]));
            v.push((format!("blocks.{l}.ffn.w_down"), vec![128, 64]));
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
fn cep_joint_writes_delta_with_joint_report_heading() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    let delta_out = tmp.path().join("m.cep.json");
    write_matching_safetensors(&w);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-joint")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--weights")
        .arg(&w)
        .arg("--cep-out")
        .arg(&delta_out);
    cmd.assert()
        .success()
        // The joint report has a distinct heading per `render_report` for CepMode::Joint.
        .stdout(predicate::str::contains("CEP Joint Prune-Search Report"))
        .stdout(predicate::str::contains("CEP joint delta written"));
    assert!(delta_out.exists(), "delta JSON must exist");
}

#[test]
fn cep_prune_and_cep_joint_are_mutually_exclusive() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    write_matching_safetensors(&w);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--cep-joint")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--weights")
        .arg(&w);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("mutually exclusive"));
}

#[test]
fn cep_joint_without_weights_refuses() {
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-joint")
        .arg("--cep-target")
        .arg("H100-SXM");
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("requires --weights"));
}
