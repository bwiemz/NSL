//! CPDT Part III v1 production-forward (S4): the M32 identity-skeleton
//! fallback (`nsl_moe_dispatch_full` / v1) must NOT be silently emitted
//! when `cpdt_mode == Full`. Pre-S4 the lowering at expr/calls.rs:1583
//! would fall through to v1 whenever `derive_v2_dims` returned None,
//! which silently routed CPDT-active builds through the identity-output
//! path with no diagnostic.
//!
//! S4 promotes that silent fallback to a hard CodegenError. This test
//! invokes `nsl build --cpdt full --cpdt-num-gpus 1` on a fixture that
//! contains `moe_dispatch` but no `--weights`, and asserts the build
//! fails with the documented diagnostic.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    std::path::Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Write a minimal safetensors file with a single tensor that does NOT
/// match the @moe block's expected router/experts entries. CPDT setup
/// succeeds (weights are present), but the moe_dispatch lowering's
/// `derive_v2_dims` returns None, triggering the S4 hard error.
fn write_unrelated_safetensors(path: &Path) {
    let bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    let view = TensorView::new(Dtype::F32, vec![1], bytes.as_slice()).unwrap();
    views.insert("unrelated.weight".to_string(), view);
    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

/// Minimal MoE fixture: a model with an `@moe` block whose forward path
/// calls `moe_dispatch(tokens, logits, experts)`. The `experts` arg is a
/// formal-parameter binding here, so even without a WeightMap entry under
/// the matching key, the codegen must surface a clear error rather than
/// silently fall through to v1.
const MOE_SRC: &str = r#"from nsl.nn.losses import mse_loss

@moe(num_experts=4, top_k=1, capacity_factor=2.0)
model Block:
    router: Tensor = ones([8, 4])
    experts: Tensor = ones([4, 8 * 16])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch(x, logits, self.experts)

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;

#[test]
fn cpdt_full_moe_dispatch_with_unrelated_weights_errors() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe.nsl");
    fs::write(&src, MOE_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_unrelated_safetensors(&weights);
    let out = tmp.path().join("moe.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out)
        .arg("--cpdt")
        .arg("full")
        .arg("--cpdt-num-gpus")
        .arg("1")
        .arg("--weights")
        .arg(&weights);
    // Weights present (CPDT setup succeeds) but no `<key>.router.weight` /
    // `<key>.experts.weight` — derive_v2_dims returns None, S4 hard-errs.

    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("moe_dispatch").and(
            predicate::str::contains("CPDT Full mode requires resolvable router + experts"),
        ));
}
