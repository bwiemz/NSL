//! CPDT Part III v2.5 codegen lowering gate:
//! `moe_dispatch_swiglu(tokens, logits, experts_gate, experts_up,
//! experts_down)` is the opt-in 5-arg source intrinsic that emits
//! `nsl_moe_dispatch_full_v4` (Mixtral's SwiGLU FFN). This CLI test
//! pins the load-bearing safety gate: hard error when v4 dims can't
//! be derived (no silent fallback to v3 from this intrinsic).
//!
//! The success path (router + experts.gate + experts.up + experts.down
//! all resolvable → object emission) is covered by the codegen unit
//! tests (`derive_v4_dims_*` in nsl-codegen/src/moe.rs) plus the
//! runtime FFI tests (`moe_dispatch_v4_swiglu.rs`). A model-level
//! `@moe` doesn't populate `moe_configs` (the decorator is collected
//! only on fields — see compiler/collection.rs:548), so a CLI-level
//! success test would need test-only scaffolding that doesn't pay for
//! itself given the unit + FFI coverage already in place.

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

/// Valid-but-unrelated safetensors: CPDT setup succeeds (weights are
/// present), but the moe_dispatch_swiglu lowering's `derive_v4_dims`
/// returns None, triggering the v4-specific hard error.
fn write_unrelated_safetensors(path: &Path) {
    let bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    let view = TensorView::new(Dtype::F32, vec![1], bytes.as_slice()).unwrap();
    views.insert("unrelated.weight".to_string(), view);
    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

const MOE_SWIGLU_SRC: &str = r#"from nsl.nn.losses import mse_loss

@moe(num_experts=4, top_k=1, capacity_factor=2.0)
model Block:
    router: Tensor = ones([8, 4])
    experts_gate: Tensor = ones([4, 8 * 16])
    experts_up: Tensor = ones([4, 8 * 16])
    experts_down: Tensor = ones([4, 16 * 8])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_swiglu(x, logits, self.experts_gate, self.experts_up, self.experts_down)

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;

const MOE_SWIGLU_WITH_GELU_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Block:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, activation="gelu")
    experts_dummy: int = 4
    router: Tensor = ones([8, 4])
    experts_gate: Tensor = ones([4, 8 * 16])
    experts_up: Tensor = ones([4, 8 * 16])
    experts_down: Tensor = ones([4, 16 * 8])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_swiglu(x, logits, self.experts_gate, self.experts_up, self.experts_down)

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;

#[test]
fn moe_dispatch_swiglu_with_non_silu_activation_errors_loudly() {
    // SwiGLU is structurally `silu(gate) * up @ down` — the SiLU is
    // wired into the v4 per-expert kernel, not a selectable arg.
    // If a user writes `@moe(activation="gelu")` and calls
    // moe_dispatch_swiglu, the source-level intent (gelu-gated FFN)
    // does NOT match what v4 emits. Per the v2.4 INVALID-VALUE-FATAL
    // convention, that mismatch is a hard build failure rather than
    // silent silu emission. The @moe decorator is collected on FIELDS
    // (compiler/collection.rs:526), so the fixture places it on a
    // sentinel `experts_dummy` field to populate moe_configs.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu_gelu.nsl");
    fs::write(&src, MOE_SWIGLU_WITH_GELU_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_unrelated_safetensors(&weights);
    let out = tmp.path().join("moe_swiglu_gelu.o");

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
        predicate::str::contains("moe_dispatch_swiglu").and(
            predicate::str::contains("SwiGLU FFI hardcodes silu(gate)*up")
                .and(predicate::str::contains("Gelu")),
        ),
    );
}

#[test]
fn moe_dispatch_swiglu_with_unrelated_weights_errors_loudly() {
    // moe_dispatch_swiglu is an OPT-IN intrinsic — no silent v3 fallback.
    // If the WeightMap doesn't contain router + experts.gate +
    // experts.up + experts.down, the build must fail with the
    // v4-specific diagnostic. Pinning the v4 message specifically (NOT
    // the v3 or v2 message) catches a regression that would silently
    // route moe_dispatch_swiglu through v3.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_unrelated_safetensors(&weights);
    let out = tmp.path().join("moe_swiglu.o");

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
        predicate::str::contains("moe_dispatch_swiglu").and(predicate::str::contains(
            "v4 (SwiGLU) requires resolvable router + experts.gate + experts.up + experts.down",
        )),
    );
}
