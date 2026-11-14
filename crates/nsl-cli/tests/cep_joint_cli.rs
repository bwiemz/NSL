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
    let bytes = serialize(&views, None).unwrap();
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
    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&delta_out).unwrap()).unwrap();
    assert_eq!(v["mode"], "joint", "a joint plan's delta must say it is one");
    assert_eq!(v["feasible"], true);
}

/// A joint plan that drops a layer, written out as source and weights. 60% of
/// the fixture's 90,432 parameters cannot go without dropping one of its two
/// blocks (head and FFN pruning stop at 47.6%), so the emitted source must
/// declare one block and the emitted weights must hold one block's tensors,
/// at the shapes that block's constructor declares.
#[test]
fn cep_joint_emits_source_and_weights_for_a_dropped_layer() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    let delta_out = tmp.path().join("m.cep.json");
    let source_out = tmp.path().join("m.pruned.nsl");
    let weights_out = tmp.path().join("m.pruned.safetensors");
    write_matching_safetensors(&w);
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_fixture())
        .arg("--cep-joint")
        .arg("--cep-sparsity")
        .arg("0.6")
        .arg("--weights")
        .arg(&w)
        .arg("--cep-out")
        .arg(&delta_out)
        .arg("--cep-emit-source")
        .arg(&source_out)
        .arg("--cep-emit-weights")
        .arg(&weights_out);
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CEP rewritten source written"))
        .stdout(predicate::str::contains("dropped"));

    let v: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&delta_out).unwrap()).unwrap();
    let layers = v["delta"]["per_layer"].as_array().unwrap();
    assert_eq!(layers.iter().filter(|l| l["drop_layer"] == true).count(), 1);
    assert_eq!(v["chosen"]["spec"]["n_layers"], 1);

    let emitted = fs::read_to_string(&source_out).unwrap();
    let line = emitted.lines().find(|l| l.trim_start().starts_with("blocks:")).unwrap();
    assert!(line.contains("[TransformerBlock; 1]"), "one block survives: {line}");
    // TransformerBlock(d_model, n_heads, n_kv_heads, head_dim, d_ff, 0.1, max_seq, theta, n_layers)
    let args: Vec<u32> = line
        .rsplit_once("TransformerBlock(")
        .unwrap()
        .1
        .trim_end_matches(')')
        .split(',')
        .filter_map(|a| a.trim().parse().ok())
        .collect();
    let (n_heads, n_kv, d_ff, n_layers_arg) = (args[1], args[2], args[4], *args.last().unwrap());
    assert_eq!(n_layers_arg, 1, "the block's n_layers argument follows the array: {line}");

    let bytes = fs::read(&weights_out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut names = st.names();
    names.sort_unstable();
    assert_eq!(names.len(), 7, "one block's seven matrices: {names:?}");
    assert!(names.iter().all(|n| n.starts_with("blocks.0.")), "{names:?}");
    let shape = |rest: &str| st.tensor(&format!("blocks.0.{rest}")).unwrap().shape().to_vec();
    assert_eq!(shape("attn.wq"), vec![64, (n_heads * 16) as usize]);
    assert_eq!(shape("attn.wk"), vec![64, (n_kv * 16) as usize]);
    assert_eq!(shape("attn.wo"), vec![(n_heads * 16) as usize, 64]);
    assert_eq!(shape("ffn.w_down"), vec![d_ff as usize, 64]);

    let mut check = Command::cargo_bin("nsl").unwrap();
    check.env("NSL_STDLIB_PATH", stdlib_path());
    check.arg("check").arg(&source_out);
    check.assert().success();
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
