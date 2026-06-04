//! CEP Option 2 SP2 — CLI integration test: `--cep-emit-source`.
//!
//! Verifies that `nsl build --cep-prune --cep-emit-source <path>` produces a
//! valid rewritten .nsl source alongside the usual delta JSON.

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

/// Same canonical-fixture-shaped safetensors used by SP1's CLI test.
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
fn cep_emit_source_writes_rewritten_nsl() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    let delta_out = tmp.path().join("m.cep.json");
    let source_out = tmp.path().join("m.pruned.nsl");
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
        .arg("--cep-emit-source")
        .arg(&source_out);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CEP rewritten source written"));
    assert!(source_out.exists(), "rewritten source must exist");

    // The emitted source must still be a valid NSL file containing the top-level model
    // and the blocks array decl. The exact rewrites depend on what the planner chooses
    // for the canonical fixture at default settings (may be a no-op if no prune fires);
    // unit tests `direct_literal_round_trip` + `empty_delta_is_identity` pin both branches.
    let bytes = fs::read_to_string(&source_out).unwrap();
    assert!(bytes.contains("model TinyCoder:"), "must contain the top model decl");
    assert!(
        bytes.contains("blocks: [TransformerBlock; 2]"),
        "must contain the blocks array decl"
    );
}
