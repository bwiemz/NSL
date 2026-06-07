//! CEP G6 — recovery fine-tuning CLI integration.
//!
//! Paper §3.2 (Dead Gradient Elimination for Pruned Training) promises that
//! the artifacts CEP emits — SP1's sliced safetensors + SP2's rewritten
//! .nsl source — feed back through NSL's existing training infrastructure
//! to support recovery fine-tuning. The user-facing recipe is:
//!
//!   1. `nsl build BASE.nsl --cep-prune --weights BASE.safetensors \
//!         --cep-emit-source PRUNED.nsl --cep-emit-weights PRUNED.safetensors`
//!   2. `nsl build PRUNED.nsl --emit-obj -o PRUNED.o --weights PRUNED.safetensors`
//!
//! Step 2 is the recovery-FT compile — the pruned source has a `train`
//! block that source-to-source AD lowers (or, today, tape-based AD picks
//! up). Either path produces a backward pass whose runtime tensor shapes
//! match the pruned topology.
//!
//! This test runs the full chain end-to-end against
//! `crates/nsl-codegen/tests/fixtures/cep_canonical_with_train.nsl` and
//! asserts:
//!
//!   - SP1+SP2 emission succeeds (already covered structurally by
//!     `cep_g18_e2e_binary_claim`, but smoke-pinned here in CLI form).
//!   - The SP2-emitted source compiles cleanly via `nsl build --emit-obj`
//!     using the SP1-sliced weights as `--weights`. Today this exercises
//!     the default AD path; the `--source-ad` path falls back to tape-AD
//!     on `self.blocks[i].*` nested-array field access (a documented
//!     codegen gap — the §3.2 "WRGA activates automatically" claim is
//!     satisfied at runtime by the tape, not at compile time by WRGA's
//!     ahead-of-time eliminator).
//!   - Both the optimizer object (`sgd_*.o`) and the model object
//!     (`*cep_canonical_with_train*.o` or similar) get written — proves
//!     the `train` block in the SP2-emitted source actually fired
//!     through the compiler and produced training-loop code.

use assert_cmd::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

fn canonical_with_train_fixture() -> PathBuf {
    workspace_root().join("crates/nsl-codegen/tests/fixtures/cep_canonical_with_train.nsl")
}

/// Write safetensors with shapes matching the canonical_with_train fixture:
/// d_model=64, 2 layers, 4 heads, 2 kv heads, head_dim=16, d_ff=128.
/// All-zero F32 data — the shapes are what CEP cross-check verifies.
fn write_canonical_safetensors(path: &Path) {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    let mut specs: Vec<(String, Vec<usize>)> = Vec::new();
    for l in 0..2 {
        specs.push((format!("blocks.{l}.attn.wq"), vec![64, 64]));
        specs.push((format!("blocks.{l}.attn.wk"), vec![64, 32]));
        specs.push((format!("blocks.{l}.attn.wv"), vec![64, 32]));
        specs.push((format!("blocks.{l}.attn.wo"), vec![64, 64]));
        specs.push((format!("blocks.{l}.ffn.w_gate"), vec![64, 128]));
        specs.push((format!("blocks.{l}.ffn.w_up"), vec![64, 128]));
        specs.push((format!("blocks.{l}.ffn.w_down"), vec![128, 64]));
    }
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

/// G6 / paper §3.2 — recovery fine-tuning closure. The SP1+SP2 outputs
/// feed back through `nsl build` with the `train` block intact, producing
/// the training-loop object code. This proves the artifact pair from
/// `--cep-prune --cep-emit-weights --cep-emit-source` is consumable by
/// the existing training pipeline without extra wiring.
#[test]
fn cep_pruned_artifacts_compile_recovery_ft() {
    let tmp = TempDir::new().unwrap();
    let baseline_weights = tmp.path().join("baseline.safetensors");
    let delta_out = tmp.path().join("delta.json");
    let pruned_source = tmp.path().join("pruned.nsl");
    let pruned_weights = tmp.path().join("pruned.safetensors");
    let pruned_obj = tmp.path().join("pruned_obj");
    write_canonical_safetensors(&baseline_weights);

    // --- Step 1: SP1 + SP2 emission.
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(canonical_with_train_fixture())
        .arg("--cep-prune")
        .arg("--cep-target")
        .arg("H100-SXM")
        .arg("--weights")
        .arg(&baseline_weights)
        .arg("--cep-out")
        .arg(&delta_out)
        .arg("--cep-emit-source")
        .arg(&pruned_source)
        .arg("--cep-emit-weights")
        .arg(&pruned_weights);
    cmd.assert().success();
    assert!(pruned_source.exists(), "SP2 emit must produce pruned source");
    assert!(pruned_weights.exists(), "SP1 emit must produce pruned weights");

    // Sanity-check the pruned artifacts before consuming them. Without
    // these the recovery-FT compile below could pass trivially on, e.g.,
    // an empty file emitted by an SP2 bug.
    let pruned_src_text = fs::read_to_string(&pruned_source).unwrap();
    assert!(
        pruned_src_text.contains("train(model"),
        "SP2-emitted source must preserve the `train(model = ...)` block verbatim — \
         recovery FT has nothing to compile otherwise.\nSource head:\n{}",
        &pruned_src_text[..pruned_src_text.len().min(400)]
    );
    // The fixture's baseline ctor is `TransformerBlock(64, 4, 2, 16, 128, 0.1)`.
    // SP2 must have rewritten at least one of (n_heads=4, n_kv_heads=2, d_ff=128)
    // — otherwise the round-trip is a no-op and the recovery-FT claim is
    // structurally meaningless.
    let baseline_ctor = "TransformerBlock(64, 4, 2, 16, 128, 0.1)";
    assert!(
        !pruned_src_text.contains(baseline_ctor),
        "SP2-emitted source still has the baseline ctor `{baseline_ctor}` — \
         pruning was a no-op, recovery-FT compile would trivially round-trip"
    );
    let baseline_weights_bytes = fs::metadata(&baseline_weights).unwrap().len();
    let pruned_weights_bytes = fs::metadata(&pruned_weights).unwrap().len();
    // Use `<=` not `<` here — if the planner ever picks "do nothing" at
    // this sparsity, the bytes will be equal and the ctor-literal check
    // above is the authoritative no-op detector (it asserts SP2 rewrote
    // at least one ctor arg, which can only happen if pruning actually
    // fired). Matches the discipline in `cep_slice_cli.rs`.
    assert!(
        pruned_weights_bytes <= baseline_weights_bytes,
        "SP1 sliced weights ({pruned_weights_bytes} bytes) must be no larger \
         than the baseline ({baseline_weights_bytes} bytes); SP1 inflated the artifact"
    );

    // --- Step 2: the recovery-FT build — compile the SP2-emitted source
    // against the SP1-sliced weights. Today this goes through the default
    // tape-AD path (source-AD's nested `self.blocks[i].*` access still
    // falls back to tape; see the test docstring). Either way, the
    // `train` block in the pruned source must compile cleanly and the
    // emitted objects must include the optimizer + the model.
    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&pruned_source)
        .arg("--weights")
        .arg(&pruned_weights)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&pruned_obj);
    let output = cmd.output().expect("nsl bin executes");
    assert!(
        output.status.success(),
        "recovery-FT build on pruned artifacts must succeed; stderr:\n{}\nstdout:\n{}",
        String::from_utf8_lossy(&output.stderr),
        String::from_utf8_lossy(&output.stdout),
    );

    // `nsl build --emit-obj` writes objects to a fresh `nsl_build_<pid>`
    // tempdir (see `crates/nsl-cli/src/main.rs` ~line 4620) and prints
    // `Wrote <abs-path>` for each. We extract every emitted PATH from
    // stdout, then verify each one EXISTS ON DISK before checking the
    // structural properties:
    //   (a) an object file named `sgd_*.o` — the optimizer was lowered,
    //       which only happens when the `train` block compiled;
    //   (b) a model object whose stem mentions `pruned` (the SP2-emitted
    //       basename), proving it was the PRUNED source that compiled.
    // The "verify exists" step defends against stdout-format drift,
    // accidental trailing whitespace, or a future hook that prints
    // a fake `Wrote ...` line.
    let stdout = String::from_utf8_lossy(&output.stdout);
    let written_paths: Vec<PathBuf> = stdout
        .lines()
        .map(|l| l.trim())
        .filter_map(|l| l.strip_prefix("Wrote "))
        .map(|p| PathBuf::from(p.trim()))
        .filter(|p| p.exists())
        .collect();
    let optimizer_emitted = written_paths.iter().any(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.starts_with("sgd_") && n.ends_with(".o"))
    });
    let pruned_model_emitted = written_paths.iter().any(|p| {
        p.file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.contains("pruned") && n.ends_with(".o"))
    });
    assert!(
        optimizer_emitted,
        "expected an `sgd_*.o` emit — `train` block did not lower an optimizer.\nWritten paths verified on disk: {written_paths:?}\nFull stdout:\n{stdout}"
    );
    assert!(
        pruned_model_emitted,
        "expected a `pruned*.o` model emit — SP2 source did not compile.\nWritten paths verified on disk: {written_paths:?}\nFull stdout:\n{stdout}"
    );
}
