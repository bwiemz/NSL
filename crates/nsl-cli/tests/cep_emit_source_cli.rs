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
    let bytes = serialize(&views, None).unwrap();
    fs::write(path, bytes).unwrap();
}

/// Pull the integer arguments out of the emitted `TransformerBlock(...)` call.
fn block_ctor_args(source: &str) -> Vec<u32> {
    let line = source
        .lines()
        .find(|l| l.trim_start().starts_with("blocks:"))
        .expect("the blocks array decl");
    let args = line.rsplit_once("TransformerBlock(").expect("the block ctor").1;
    args.trim_end_matches(')').split(',').filter_map(|a| a.trim().parse().ok()).collect()
}

/// With `--cep-emit-source` the plan is kept uniform, so the source (which
/// has one block constructor) can express it: the emitted source and the
/// emitted weights must describe one and the same model, and the emitted
/// source must itself be a valid program.
#[test]
fn cep_emit_source_and_weights_describe_the_same_model() {
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
        .arg("--cep-prune")
        .arg("--cep-target")
        .arg("H100-SXM")
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
        .stdout(predicate::str::contains("Uniform plan"))
        .stdout(predicate::str::contains("CEP rewritten source written"));

    let emitted = fs::read_to_string(&source_out).unwrap();
    assert!(emitted.contains("model TinyCoder:"), "must contain the top model decl");
    assert!(emitted.contains("blocks: [TransformerBlock; 2]"), "no layer was dropped");
    // TransformerBlock(d_model, n_heads, n_kv_heads, head_dim, d_ff, 0.1, max_seq, theta, n_layers)
    let baseline = block_ctor_args(&fs::read_to_string(canonical_fixture()).unwrap());
    assert_eq!(baseline, vec![64, 4, 2, 16, 128, 1024, 2]);
    let args = block_ctor_args(&emitted);
    let (d_model, n_heads, n_kv, head_dim, d_ff) = (args[0], args[1], args[2], args[3], args[4]);
    assert_eq!((d_model, head_dim), (64, 16), "residual width and head_dim are never pruned");
    assert!(n_heads < 4 || d_ff < 128, "a 30% prune must change the block: {args:?}");
    assert_eq!(n_heads, 2 * n_kv, "heads go in whole key/value groups");

    // The weights have exactly the shapes that constructor declares.
    let bytes = fs::read(&weights_out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    for l in 0..2 {
        let shape = |rest: &str| st.tensor(&format!("blocks.{l}.{rest}")).unwrap().shape().to_vec();
        assert_eq!(shape("attn.wq"), vec![64, (n_heads * 16) as usize]);
        assert_eq!(shape("attn.wk"), vec![64, (n_kv * 16) as usize]);
        assert_eq!(shape("attn.wv"), vec![64, (n_kv * 16) as usize]);
        assert_eq!(shape("attn.wo"), vec![(n_heads * 16) as usize, 64]);
        assert_eq!(shape("ffn.w_gate"), vec![64, d_ff as usize]);
        assert_eq!(shape("ffn.w_up"), vec![64, d_ff as usize]);
        assert_eq!(shape("ffn.w_down"), vec![d_ff as usize, 64]);
    }

    // And the emitted source type-checks.
    let mut check = Command::cargo_bin("nsl").unwrap();
    check.env("NSL_STDLIB_PATH", stdlib_path());
    check.arg("check").arg(&source_out);
    check.assert().success();
}

/// The emitted source declares every layer with one constructor, so a plan
/// that must leave some layers alone cannot be written as source. Refused
/// before any work, with nothing written.
#[test]
fn cep_emit_source_with_preserve_is_refused() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    write_matching_safetensors(&w);
    let src = fs::read_to_string(canonical_fixture()).unwrap();
    let original = r#"@cep_prune(target = h100, sparsity = 0.3, granularity = "head+ffn")"#;
    assert!(src.contains(original));
    let model = tmp.path().join("model.nsl");
    fs::write(
        &model,
        src.replace(
            original,
            r#"@cep_prune(target = h100, sparsity = 0.1, granularity = "head+ffn", preserve = ["blocks.0.*"])"#,
        ),
    )
    .unwrap();
    let delta_out = tmp.path().join("m.cep.json");
    let source_out = tmp.path().join("m.pruned.nsl");
    let weights_out = tmp.path().join("m.pruned.safetensors");
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&model)
        .arg("--cep-prune")
        .arg("--weights")
        .arg(&w)
        .arg("--cep-out")
        .arg(&delta_out)
        .arg("--cep-emit-weights")
        .arg(&weights_out)
        .arg("--cep-emit-source")
        .arg(&source_out);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("--cep-emit-source cannot be combined with `preserve`"));
    assert!(!delta_out.exists() && !weights_out.exists() && !source_out.exists());
}

/// If one of the requested outputs cannot be written, none is: a weights file
/// without the source it belongs to is not a result.
#[test]
fn cep_outputs_are_all_or_nothing() {
    let tmp = TempDir::new().unwrap();
    let w = tmp.path().join("w.safetensors");
    write_matching_safetensors(&w);
    let delta_out = tmp.path().join("m.cep.json");
    let weights_out = tmp.path().join("m.pruned.safetensors");
    let source_out = tmp.path().join("no_such_dir").join("m.pruned.nsl");
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
        .arg(&weights_out)
        .arg("--cep-emit-source")
        .arg(&source_out);
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("no_such_dir"))
        .stderr(predicate::str::contains("does not exist"));
    assert!(!delta_out.exists(), "the delta must not be written when an output cannot be");
    assert!(!weights_out.exists(), "the weights must not be written when the source cannot be");
}
