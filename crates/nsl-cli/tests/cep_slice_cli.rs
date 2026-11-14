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
    let bytes = serialize(&views, None).unwrap();
    fs::write(path, bytes).unwrap();
}

/// The decorator asks for 30% of the fixture's 90,432 parameters. The
/// weights file here holds only the two blocks' matrices (73,728 values), so
/// the sliced file must hold at least 27,130 fewer, and the same heads and
/// neurons must be gone from every matrix of a layer.
#[test]
fn cep_emit_weights_writes_a_consistent_smaller_file() {
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

    let bytes = std::fs::read(&weights_out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .expect("emitted file is valid safetensors");
    let shape = |name: &str| st.tensor(name).unwrap_or_else(|_| panic!("missing {name}")).shape().to_vec();
    let mut kept = 0usize;
    for l in 0..2 {
        let wq = shape(&format!("blocks.{l}.attn.wq"));
        let wk = shape(&format!("blocks.{l}.attn.wk"));
        let wv = shape(&format!("blocks.{l}.attn.wv"));
        let wo = shape(&format!("blocks.{l}.attn.wo"));
        let gate = shape(&format!("blocks.{l}.ffn.w_gate"));
        let up = shape(&format!("blocks.{l}.ffn.w_up"));
        let down = shape(&format!("blocks.{l}.ffn.w_down"));
        // The residual width never changes.
        assert_eq!((wq[0], wk[0], wv[0], wo[1], gate[0], up[0], down[1]), (64, 64, 64, 64, 64, 64, 64));
        // Query heads: the same count in wq (columns) and wo (rows), in
        // whole heads of 16; key/value heads half as many (groups of 2).
        assert_eq!(wq[1], wo[0], "layer {l}: wq and wo disagree on the heads kept");
        assert_eq!(wq[1] % 32, 0, "layer {l}: heads go in key/value groups of 2x16 columns");
        assert_eq!(wk[1] * 2, wq[1], "layer {l}: wk must lose the group's key/value head");
        assert_eq!(wv[1], wk[1]);
        // FFN: one width across the three matrices.
        assert_eq!((gate[1], up[1]), (down[0], down[0]), "layer {l}: FFN widths disagree");
        kept += [&wq, &wk, &wv, &wo, &gate, &up, &down].iter().map(|s| s[0] * s[1]).sum::<usize>();
    }
    assert!(kept <= 73_728 - 27_130, "{kept} values kept; 30% of the model must be gone");
    assert_eq!(st.names().len(), 14);

    // The delta describes the same shapes.
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&delta_out).unwrap()).unwrap();
    for l in 0..2 {
        let heads = v["chosen"]["spec"]["n_heads"][l].as_u64().unwrap() as usize;
        let d_ff = v["chosen"]["spec"]["d_ff"][l].as_u64().unwrap() as usize;
        assert_eq!(shape(&format!("blocks.{l}.attn.wq"))[1], heads * 16);
        assert_eq!(shape(&format!("blocks.{l}.ffn.w_gate"))[1], d_ff);
    }
}

/// `.nslm` is read by position, and a `.safetensors` file does not record the
/// model's parameter order, so this combination is refused before any work.
#[test]
fn cep_emit_nslm_from_safetensors_is_refused() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    let delta_out = tmp.path().join("m.cep.json");
    let weights_out = tmp.path().join("m.pruned.nslm");
    write_matching_safetensors(&w);

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&w)
        .arg("--cep-out")
        .arg(&delta_out)
        .arg("--cep-emit-weights")
        .arg(&weights_out);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("does not record"))
        .stderr(predicate::str::contains("parameter order"));
    assert!(!weights_out.exists() && !delta_out.exists());
}
