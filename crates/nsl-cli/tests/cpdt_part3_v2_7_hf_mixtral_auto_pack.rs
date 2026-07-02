//! CPDT Part III v2.7 end-to-end CLI gate:
//! `@moe(weight_prefix="…")` lets a user point the v3/v4 lowering at
//! an HF Mixtral safetensors prefix (which contains `.` characters
//! that NSL field names can't), and the v2.7 auto-pack at
//! `WeightMap::load` rewrites the per-expert HF tensors in place into
//! NSL packed convention so `derive_v4_dims` resolves them.
//!
//! Together these two pieces close the end-user flow that v2.6 left
//! incomplete: load a real-shaped HF Mixtral checkpoint, declare the
//! HF prefix in source, build with `nsl build --emit-obj` — no manual
//! repacking required.

use assert_cmd::prelude::*;
use predicates::prelude::*;
use safetensors::tensor::{serialize, TensorView};
use safetensors::Dtype;
use std::collections::BTreeMap;
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

/// Build a small synthetic HF Mixtral safetensors file with two MoE
/// experts and the router. Shapes follow PyTorch's `nn.Linear.weight`
/// `[out_features, in_features]` convention.
///
/// - hidden = 4, intermediate = 8, num_experts = 2.
/// - Tensors are zero-filled f32; numerical correctness is exercised
///   by the v4 FFI runtime tests, this CLI gate only confirms the
///   build pipeline accepts the HF layout end-to-end.
fn write_hf_mixtral_safetensors(path: &Path, hf_prefix: &str) {
    let hidden = 4_usize;
    let intermediate = 8_usize;
    let num_experts = 2_usize;

    // Allocate the buffers up front; safetensors borrows views into them.
    let router_bytes = vec![0u8; hidden * num_experts * 4];
    let w1_bytes = vec![0u8; intermediate * hidden * 4];
    let w3_bytes = vec![0u8; intermediate * hidden * 4];
    let w2_bytes = vec![0u8; hidden * intermediate * 4];

    // Per-expert allocations need their own owned vectors so the
    // TensorViews can hold non-overlapping slices.
    let mut per_expert_w1: Vec<Vec<u8>> = (0..num_experts).map(|_| w1_bytes.clone()).collect();
    let mut per_expert_w2: Vec<Vec<u8>> = (0..num_experts).map(|_| w2_bytes.clone()).collect();
    let mut per_expert_w3: Vec<Vec<u8>> = (0..num_experts).map(|_| w3_bytes.clone()).collect();
    let _ = (&mut per_expert_w1, &mut per_expert_w2, &mut per_expert_w3); // silence unused-mut

    let mut views: BTreeMap<String, TensorView<'_>> = BTreeMap::new();

    // Router under HF's `.gate.weight` naming.
    views.insert(
        format!("{}.gate.weight", hf_prefix),
        TensorView::new(Dtype::F32, vec![hidden, num_experts], &router_bytes).unwrap(),
    );

    // Per-expert w1/w2/w3.
    for e in 0..num_experts {
        views.insert(
            format!("{}.experts.{}.w1.weight", hf_prefix, e),
            TensorView::new(Dtype::F32, vec![intermediate, hidden], &per_expert_w1[e]).unwrap(),
        );
        views.insert(
            format!("{}.experts.{}.w3.weight", hf_prefix, e),
            TensorView::new(Dtype::F32, vec![intermediate, hidden], &per_expert_w3[e]).unwrap(),
        );
        views.insert(
            format!("{}.experts.{}.w2.weight", hf_prefix, e),
            TensorView::new(Dtype::F32, vec![hidden, intermediate], &per_expert_w2[e]).unwrap(),
        );
    }
    // Drop the borrow-of-_unused_ references (silence dead-code/borrow
    // checker; the safetensors crate's TensorView only borrows the
    // contents above for the lifetime of `views`).
    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

const MOE_SWIGLU_WEIGHT_PREFIX_SRC: &str = r#"model Block:
    @moe(num_experts=2, top_k=1, capacity_factor=2.0, weight_prefix="hf.layer")
    experts_dummy: int = 2
    router: Tensor = ones([4, 2])
    experts_gate: Tensor = ones([2, 4 * 8])
    experts_up: Tensor = ones([2, 4 * 8])
    experts_down: Tensor = ones([2, 8 * 4])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_swiglu(x, logits, self.experts_gate, self.experts_up, self.experts_down)

let m = Block()
let x = ones([2, 4])
let y = m.forward(x)
"#;

#[test]
fn weight_prefix_kwarg_plus_auto_pack_resolves_hf_mixtral_checkpoint() {
    // End-to-end: write an HF Mixtral-shaped safetensors file (with
    // dotted prefix segments), declare `@moe(weight_prefix="hf.layer")`
    // in source, and confirm that:
    //   1. The v2.7 auto-pack at WeightMap::load rewrites the per-
    //      expert HF tensors in place under the HF prefix (visible via
    //      the `[nsl] CPDT v2.7: auto-packed ...` diagnostic line).
    //   2. The v4 SwiGLU lowering's lookup under the weight_prefix
    //      resolves the packed entries (build succeeds with --emit-obj
    //      producing a non-empty object file).
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("hf_mixtral.nsl");
    fs::write(&src, MOE_SWIGLU_WEIGHT_PREFIX_SRC).unwrap();
    let weights = tmp.path().join("hf_mixtral.safetensors");
    write_hf_mixtral_safetensors(&weights, "hf.layer");
    // `--emit-obj` ALWAYS writes alongside the source file (the `-o`
    // flag only renames the final executable on a regular build).
    let expected_obj = tmp.path().join("hf_mixtral.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("--weights")
        .arg(&weights);

    let output = cmd.output().expect("nsl invocation");
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !output.status.success() {
        panic!(
            "build failed (exit {:?})\nstderr:\n{}\nstdout:\n{}",
            output.status.code(),
            stderr,
            stdout,
        );
    }

    // The auto-pack diagnostic must appear on stderr — confirms the
    // detector + packer ran. Pinning the prefix in the message protects
    // against a regression that detects but silently no-ops.
    assert!(
        stderr.contains("CPDT v2.7: auto-packed 1 HF Mixtral MoE block")
            && stderr.contains("hf.layer (num_experts=2)"),
        "auto-pack diagnostic missing — stderr was:\n{}",
        stderr,
    );

    assert!(
        expected_obj.exists(),
        "build should emit object file at {:?}\nstderr was:\n{}\nstdout was:\n{}",
        expected_obj,
        stderr,
        stdout,
    );
    let obj_bytes = fs::metadata(&expected_obj).unwrap().len();
    assert!(obj_bytes > 0, "emitted object file must not be empty");
}

#[test]
fn weight_prefix_rejects_empty_string() {
    // @moe(weight_prefix="") would look up under `.router.weight`
    // (leading dot), which is never a real safetensors key. Refuse
    // loudly at the codegen layer's extract_moe_decorator path —
    // pinned by an NSL fixture that exercises the build pipeline up
    // to the decorator parse.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("empty_prefix.nsl");
    fs::write(
        &src,
        r#"model Block:
    @moe(num_experts=2, top_k=1, capacity_factor=2.0, weight_prefix="")
    experts_dummy: int = 2
    router: Tensor = ones([4, 2])

    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = Block()
let x = ones([2, 4])
let y = m.forward(x)
"#,
    )
    .unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_hf_mixtral_safetensors(&weights, "hf.layer");
    let out = tmp.path().join("empty_prefix.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out)
        .arg("--weights")
        .arg(&weights);

    cmd.assert().failure().stderr(
        predicate::str::contains("weight_prefix must be a non-empty string"),
    );
}
