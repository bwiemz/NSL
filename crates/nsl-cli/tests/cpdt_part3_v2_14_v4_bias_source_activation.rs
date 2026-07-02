//! CPDT Part III v2.14 — source-level activation of v4 (SwiGLU/GeGLU/
//! ReGLU) FFN biases.
//!
//! Symmetric to v2.12's v3 cycle: `moe_dispatch_swiglu` source intrinsic
//! gains an optional 8-arg form `(tokens, logits, experts_gate,
//! experts_up, experts_down, gate_bias, up_bias, down_bias)`. The
//! WeightMap detection refuses the 5-arg form when v4 biases are
//! present (all-or-nothing per layout family), partial-bias bundles,
//! and shape-mismatched biases.
//!
//! Tests at the CLI layer pin the DIAGNOSTIC paths a user actually
//! hits; the success-path 8-arg lowering + FFI bias-application is
//! covered by `moe_dispatch_v4_swiglu.rs` runtime tests + the
//! `detect_v4_biases_*` codegen unit tests.

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

/// Write a v4 safetensors bundle with optional gate/up/down biases
/// under `Block.*` prefix. Bias arguments are independent Option<usize>
/// element counts — `None` omits that bias entry.
#[allow(clippy::too_many_arguments)]
fn write_v4_safetensors(
    path: &Path,
    hidden: usize,
    intermediate: usize,
    num_experts: usize,
    gate_bias_elems: Option<usize>,
    up_bias_elems: Option<usize>,
    down_bias_elems: Option<usize>,
) {
    let router_bytes: Vec<u8> = vec![0u8; hidden * num_experts * 4];
    let gate_bytes: Vec<u8> = vec![0u8; num_experts * hidden * intermediate * 4];
    let up_bytes: Vec<u8> = vec![0u8; num_experts * hidden * intermediate * 4];
    let down_bytes: Vec<u8> = vec![0u8; num_experts * intermediate * hidden * 4];
    let gate_bias_bytes: Option<Vec<u8>> = gate_bias_elems.map(|n| vec![0u8; n * 4]);
    let up_bias_bytes: Option<Vec<u8>> = up_bias_elems.map(|n| vec![0u8; n * 4]);
    let down_bias_bytes: Option<Vec<u8>> = down_bias_elems.map(|n| vec![0u8; n * 4]);

    let router_view = TensorView::new(
        Dtype::F32,
        vec![hidden, num_experts],
        router_bytes.as_slice(),
    )
    .unwrap();
    let gate_view = TensorView::new(
        Dtype::F32,
        vec![num_experts, hidden * intermediate],
        gate_bytes.as_slice(),
    )
    .unwrap();
    let up_view = TensorView::new(
        Dtype::F32,
        vec![num_experts, hidden * intermediate],
        up_bytes.as_slice(),
    )
    .unwrap();
    let down_view = TensorView::new(
        Dtype::F32,
        vec![num_experts, intermediate * hidden],
        down_bytes.as_slice(),
    )
    .unwrap();

    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    views.insert("Block.router.weight".to_string(), router_view);
    views.insert("Block.experts.gate.weight".to_string(), gate_view);
    views.insert("Block.experts.up.weight".to_string(), up_view);
    views.insert("Block.experts.down.weight".to_string(), down_view);

    let gate_bias_view: Option<TensorView<'_>> = gate_bias_bytes.as_ref().map(|bytes| {
        TensorView::new(Dtype::F32, vec![bytes.len() / 4], bytes.as_slice()).unwrap()
    });
    let up_bias_view: Option<TensorView<'_>> = up_bias_bytes.as_ref().map(|bytes| {
        TensorView::new(Dtype::F32, vec![bytes.len() / 4], bytes.as_slice()).unwrap()
    });
    let down_bias_view: Option<TensorView<'_>> = down_bias_bytes.as_ref().map(|bytes| {
        TensorView::new(Dtype::F32, vec![bytes.len() / 4], bytes.as_slice()).unwrap()
    });
    if let Some(v) = gate_bias_view {
        views.insert("Block.experts.gate.bias".to_string(), v);
    }
    if let Some(v) = up_bias_view {
        views.insert("Block.experts.up.bias".to_string(), v);
    }
    if let Some(v) = down_bias_view {
        views.insert("Block.experts.down.bias".to_string(), v);
    }

    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

/// 5-arg `moe_dispatch_swiglu` source — the @moe field-decorator
/// convention from v2.7 (decorator must attach to a non-Tensor field)
/// with `weight_prefix="Block"` pointing at the v4 layout under that
/// prefix in the safetensors bundle.
///
/// v2.14 fix F2 (IMPORTANT adversarial review) — the `experts_dummy:
/// int = 4` field is the parse-required carrier for the `@moe`
/// decorator. The compiler-collection phase at `compiler/collection.rs:548`
/// only picks up field-level decorators (not class-level), AND the
/// field must be a non-Tensor type for the decorator to fire — a
/// tensor-typed field would change whether moe_configs is populated.
/// This is the v2.7 convention pinned across every CPDT-v2.x CLI gate
/// (v2.7, v2.10, v2.12, v2.13, v2.14). Documented here so future test
/// authors copying this pattern don't accidentally drop the dummy
/// anchor (which would silently break the gate).
const MOE_SWIGLU_5ARG_SRC: &str = r#"model Block:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block")
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

/// 8-arg form — success path with all 3 biases declared as model
/// fields and threaded through the 8-arg moe_dispatch_swiglu call.
const MOE_SWIGLU_8ARG_SRC: &str = r#"model Block:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block")
    experts_dummy: int = 4
    router: Tensor = ones([8, 4])
    experts_gate: Tensor = ones([4, 8 * 16])
    experts_up: Tensor = ones([4, 8 * 16])
    experts_down: Tensor = ones([4, 16 * 8])
    experts_gate_bias: Tensor = ones([4, 16])
    experts_up_bias: Tensor = ones([4, 16])
    experts_down_bias: Tensor = ones([4, 8])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_swiglu(
            x, logits,
            self.experts_gate, self.experts_up, self.experts_down,
            self.experts_gate_bias, self.experts_up_bias, self.experts_down_bias,
        )

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;

#[test]
fn moe_dispatch_swiglu_5arg_with_weight_map_biases_refuses_loudly() {
    // All 3 v4 biases in the WeightMap, 5-arg source call → must
    // refuse loudly so the loaded biases don't silently drop.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_5ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    // hidden=8, intermediate=16, num_experts=4.
    write_v4_safetensors(
        &weights,
        8, 16, 4,
        Some(4 * 16), Some(4 * 16), Some(4 * 8),
    );
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
        predicate::str::contains("moe_dispatch_swiglu")
            .and(predicate::str::contains("v4 FFN biases"))
            .and(predicate::str::contains("8-arg form")),
    );
}

#[test]
fn moe_dispatch_swiglu_5arg_with_partial_gate_only_bias_refuses_loudly() {
    // Only gate.bias present — partial bundle → loud refusal.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_5ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v4_safetensors(&weights, 8, 16, 4, Some(4 * 16), None, None);
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
        predicate::str::contains("partial v4 bias bundle")
            .and(predicate::str::contains("gate.bias")),
    );
}

#[test]
fn moe_dispatch_swiglu_5arg_with_two_of_three_biases_refuses_loudly() {
    // up.bias + down.bias present, gate.bias missing — partial bundle.
    // Distinguishes partial-but-2 from partial-but-1; v4 paper says
    // it's all 3 or none, and the diagnostic must call out gate as
    // missing.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_5ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v4_safetensors(&weights, 8, 16, 4, None, Some(4 * 16), Some(4 * 8));
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
        predicate::str::contains("partial v4 bias bundle")
            .and(predicate::str::contains("gate.bias"))
            .and(predicate::str::contains("missing")),
    );
}

#[test]
fn moe_dispatch_swiglu_5arg_with_mismatched_gate_bias_shape_refuses_loudly() {
    // gate.bias has 4 * 15 = 60 elems; expected 4 * 16 = 64. Off-by-
    // one in intermediate_dim would land as silent-corruption OOB
    // reads at runtime; codegen diagnostic is far more actionable.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_5ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v4_safetensors(
        &weights,
        8, 16, 4,
        Some(4 * 15), Some(4 * 16), Some(4 * 8),
    );
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
        predicate::str::contains("gate.bias")
            .and(predicate::str::contains("60 elements"))
            .and(predicate::str::contains("expected 64")),
    );
}

#[test]
fn moe_dispatch_swiglu_5arg_with_no_biases_still_succeeds() {
    // Bias-free v4 (Mixtral paper convention) — 5-arg path stays
    // byte-identical, no source-activation gate trips.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_5ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v4_safetensors(&weights, 8, 16, 4, None, None, None);
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

    cmd.assert().success();
}

#[test]
fn moe_dispatch_swiglu_8arg_form_with_weight_map_biases_succeeds() {
    // Success path: 8-arg source form + all 3 biases in WeightMap.
    // detect_v4_biases returns Ok(Some(())) but bias_vals=Some(_)
    // prevents the refusal; FFI receives the bias pointers at
    // positions 12+13+14.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_8ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v4_safetensors(
        &weights,
        8, 16, 4,
        Some(4 * 16), Some(4 * 16), Some(4 * 8),
    );
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

    cmd.assert().success();
    assert!(out.exists());
    let obj_bytes = fs::metadata(&out).unwrap().len();
    assert!(obj_bytes > 0, "emitted object file must not be empty");
}

#[test]
fn moe_dispatch_swiglu_wrong_arity_6_or_7_refuses_loudly() {
    // The arity check accepts {5, 8} only. 6 or 7 args is almost-
    // certainly a typo. The diagnostic must enumerate both supported
    // arities.
    let bad_src = r#"model Block:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block")
    experts_dummy: int = 4
    router: Tensor = ones([8, 4])
    experts_gate: Tensor = ones([4, 8 * 16])
    experts_up: Tensor = ones([4, 8 * 16])
    experts_down: Tensor = ones([4, 16 * 8])
    experts_gate_bias: Tensor = ones([4, 16])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_swiglu(x, logits, self.experts_gate, self.experts_up, self.experts_down, self.experts_gate_bias)

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, bad_src).unwrap();
    let out = tmp.path().join("moe_swiglu.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out);

    cmd.assert().failure().stderr(
        predicate::str::contains("moe_dispatch_swiglu")
            .and(predicate::str::contains("5 arguments"))
            .and(predicate::str::contains("8 arguments")),
    );
}
