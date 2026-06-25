//! CPDT Part III v2.8 end-to-end CLI gate:
//! `@moe(activation="gelu") + moe_dispatch_swiglu(...)` now emits a
//! GeGLU FFN (v4 FFI with gate_activation_kind=2), and the same with
//! `activation="relu"` emits ReGLU (kind=3).
//!
//! v2.5/v2.7 hard-refused any `activation != silu` on the SwiGLU
//! intrinsic. v2.8 extends the v4 FFI signature with
//! `gate_activation_kind` and threads `info.activation as i64` through
//! the codegen v4 lowering arm — so these two source programs that
//! previously failed at build time now produce object files.
//!
//! Identity-refusal stays in v2.5's CLI gate
//! (`moe_dispatch_swiglu_with_identity_activation_errors_loudly`) —
//! a GLU with identity gate is structurally non-Mixtral.
//!
//! v2.8 adversarial-review (Refuter 3, HIGH) flagged that the two
//! happy-path tests above can't distinguish a regression where codegen
//! silently hard-codes gate_activation_kind=1 — both tests would still
//! pass because the build still succeeds and the .o is still non-empty.
//! `activation_kind_threads_through_codegen_byte_divergence` closes that
//! gap by building twice with identical source-modulo-activation and
//! asserting the emitted .o bytes DIFFER between activation="silu" vs
//! "gelu" vs "relu". A regression that hardcoded kind=1 would produce
//! byte-identical objects across all three activations (the rest of the
//! program is bit-identical), so the gate would fail loudly.

use assert_cmd::prelude::*;
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

/// Synthetic HF Mixtral safetensors: router + 2 experts × 3 projections.
/// The shapes follow PyTorch's `nn.Linear.weight` `[out, in]` convention
/// per expert. v2.7's auto-pack rewrites the per-expert HF tensors in
/// place under the HF prefix, so the codegen v4 lookup resolves
/// `<prefix>.experts.{gate,up,down}.weight` at build time.
fn write_hf_mixtral_safetensors(path: &Path, hf_prefix: &str) {
    let hidden = 4_usize;
    let intermediate = 8_usize;
    let num_experts = 2_usize;

    let router_bytes = vec![0u8; hidden * num_experts * 4];
    let w1_bytes = vec![0u8; intermediate * hidden * 4];
    let w3_bytes = vec![0u8; intermediate * hidden * 4];
    let w2_bytes = vec![0u8; hidden * intermediate * 4];

    let per_expert_w1: Vec<Vec<u8>> = (0..num_experts).map(|_| w1_bytes.clone()).collect();
    let per_expert_w2: Vec<Vec<u8>> = (0..num_experts).map(|_| w2_bytes.clone()).collect();
    let per_expert_w3: Vec<Vec<u8>> = (0..num_experts).map(|_| w3_bytes.clone()).collect();

    let mut views: BTreeMap<String, TensorView<'_>> = BTreeMap::new();
    views.insert(
        format!("{}.gate.weight", hf_prefix),
        TensorView::new(Dtype::F32, vec![hidden, num_experts], &router_bytes).unwrap(),
    );
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
    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

const MOE_GELU_SRC: &str = r#"model Block:
    @moe(num_experts=2, top_k=1, capacity_factor=2.0, activation="gelu", weight_prefix="hf.layer")
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

const MOE_RELU_SRC: &str = r#"model Block:
    @moe(num_experts=2, top_k=1, capacity_factor=2.0, activation="relu", weight_prefix="hf.layer")
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
fn gelu_activation_emits_geglu_object_via_moe_dispatch_swiglu() {
    // @moe(activation="gelu") used to be refused by the moe_dispatch_swiglu
    // lowering arm pre-v2.8. After v2.8 wires gate_activation_kind through
    // the v4 FFI, this is a happy path that emits a real GeGLU FFN.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("geglu.nsl");
    fs::write(&src, MOE_GELU_SRC).unwrap();
    let weights = tmp.path().join("geglu.safetensors");
    write_hf_mixtral_safetensors(&weights, "hf.layer");
    let expected_obj = tmp.path().join("geglu.o");

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
    assert!(
        expected_obj.exists(),
        "build should emit object file at {:?}\nstderr:\n{}",
        expected_obj, stderr,
    );
    assert!(
        fs::metadata(&expected_obj).unwrap().len() > 0,
        "emitted object file must not be empty",
    );
}

fn src_with_activation(activation: &str) -> String {
    format!(r#"model Block:
    @moe(num_experts=2, top_k=1, capacity_factor=2.0, activation="{activation}", weight_prefix="hf.layer")
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
"#)
}

/// Build `src_str` under `tmp/<label>/`, return bytes of emitted .o
/// so byte-divergence assertions can compare them. Each label gets
/// its own sub-temp so .o paths can't collide.
fn build_and_read_obj_bytes(tmp: &Path, label: &str, src_str: &str) -> Vec<u8> {
    let sub = tmp.join(label);
    fs::create_dir_all(&sub).unwrap();
    let src = sub.join("block.nsl");
    fs::write(&src, src_str).unwrap();
    let weights = sub.join("weights.safetensors");
    write_hf_mixtral_safetensors(&weights, "hf.layer");
    let expected_obj = sub.join("block.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("--weights")
        .arg(&weights);

    let output = cmd.output().expect("nsl invocation");
    if !output.status.success() {
        panic!(
            "build [{}] failed (exit {:?})\nstderr:\n{}\nstdout:\n{}",
            label,
            output.status.code(),
            String::from_utf8_lossy(&output.stderr),
            String::from_utf8_lossy(&output.stdout),
        );
    }
    assert!(expected_obj.exists(), "build [{label}] should emit object file");
    fs::read(&expected_obj).expect("read object file")
}

#[test]
fn activation_kind_threads_through_codegen_byte_divergence() {
    // v2.8 adversarial-review (Refuter 3) raised HIGH that the two
    // happy-path gates only check `obj.exists() && len > 0` — both
    // would pass if codegen silently hard-coded gate_activation_kind=1
    // (or reused top_k_val) and ignored info.activation.
    //
    // This gate closes that gap. The source-NSL fixture is generated
    // by a format!() so only the `activation="…"` substring differs
    // across the three builds; everything else (router shape, experts,
    // forward, hidden, intermediate, num_experts, top_k,
    // capacity_factor, weight_prefix) is bit-identical. The codegen
    // path differs only in one Cranelift iconst immediate (silu→1,
    // gelu→2, relu→3). A regression that hardcoded the immediate
    // would produce three byte-identical .o files, so this byte-
    // divergence check would fail loudly.
    //
    // NOT a numerical-correctness check — runtime output is pinned by
    // moe_dispatch_v4_swiglu.rs's hand-rolled-reference tests. This
    // test specifically pins the codegen→FFI plumbing.
    let tmp = TempDir::new().unwrap();
    let silu_bytes = build_and_read_obj_bytes(tmp.path(), "silu", &src_with_activation("silu"));
    let gelu_bytes = build_and_read_obj_bytes(tmp.path(), "gelu", &src_with_activation("gelu"));
    let relu_bytes = build_and_read_obj_bytes(tmp.path(), "relu", &src_with_activation("relu"));

    assert_ne!(
        silu_bytes, gelu_bytes,
        "silu (kind=1) and gelu (kind=2) builds produced byte-identical .o — codegen may be \
         ignoring info.activation. Check expr/calls.rs moe_dispatch_swiglu arm threads \
         `activation as i64` into the 11th nsl_moe_dispatch_full_v4 arg."
    );
    assert_ne!(
        silu_bytes, relu_bytes,
        "silu (kind=1) and relu (kind=3) builds produced byte-identical .o — codegen may be \
         ignoring info.activation."
    );
    assert_ne!(
        gelu_bytes, relu_bytes,
        "gelu (kind=2) and relu (kind=3) builds produced byte-identical .o — codegen may be \
         collapsing both non-silu kinds onto the same value."
    );
}

#[test]
fn relu_activation_emits_reglu_object_via_moe_dispatch_swiglu() {
    // @moe(activation="relu") was also refused pre-v2.8. v2.8 wires
    // gate_activation_kind=3 through to v4, emitting a ReGLU FFN.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("reglu.nsl");
    fs::write(&src, MOE_RELU_SRC).unwrap();
    let weights = tmp.path().join("reglu.safetensors");
    write_hf_mixtral_safetensors(&weights, "hf.layer");
    let expected_obj = tmp.path().join("reglu.o");

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
    assert!(
        expected_obj.exists(),
        "build should emit object file at {:?}\nstderr:\n{}",
        expected_obj, stderr,
    );
    assert!(
        fs::metadata(&expected_obj).unwrap().len() > 0,
        "emitted object file must not be empty",
    );
}
