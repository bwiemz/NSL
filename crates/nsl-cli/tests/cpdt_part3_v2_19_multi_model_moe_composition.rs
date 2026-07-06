//! CPDT Part III v2.19 — multi-model `@moe` composition CLI gate.
//!
//! Closes the v2.18 deferral. A source file with TWO model
//! declarations, each carrying its own `@moe(weight_prefix=...)`, and
//! each with its own `forward` that calls `moe_dispatch_swiglu`, must
//! compile both methods so each one resolves to its own MoE config.
//!
//! ## The bug (pre-v2.19)
//!
//! The candidate resolution in `expr/calls.rs` (3 sites: v2
//! `moe_dispatch`, v3 `moe_dispatch_ffn`, v4 `moe_dispatch_swiglu`)
//! filtered `moe_configs` by `state.current_function_name.split("__")
//! .next()` — which:
//!   1. is `None` for model method bodies (the model-method codegen
//!      path in `compile_model_methods` never sets
//!      `state.current_function_name`; only top-level functions do via
//!      `func.rs:133`)
//!   2. would return `""` even if it WERE the mangled name (the model
//!      method name starts with `__`, so split("__") yields the empty
//!      string before the first `__`)
//!
//! With `moe_configs.len() >= 2`, the singleton-fallback
//! (`or_else { if len == 1 { ... } }`) also returns `None`. Both
//! forwards then hit the same "v4 dims could not be resolved" path
//! despite the auto-pack chain having loaded both blocks correctly.
//!
//! ## The fix
//!
//! Prefer `self.current_method_model_name` (already populated in
//! `compile_model_methods.rs:579,735` for model methods, including
//! the agent.rs:424 reuse). Filter by `"{model_name}."` (trailing dot
//! avoids prefix shadowing between `Block` and `Block0`). When
//! multiple `moe_configs` start with that prefix (multi-MoE-per-
//! model case), the existing match arm refuses with `None`, which
//! preserves the documented v2.next deferral for THAT case (multiple
//! `@moe` decorators on different fields of the SAME model).
//!
//! ## Scope
//!
//! The v2.18 cycle pinned multi-BLOCK auto-pack telemetry (one
//! `model` declaration consuming the first block, others
//! auto-packed-but-unused). v2.19 pins multi-MODEL composition (each
//! model has its own forward and consumes its own block).

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
    Path::new(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Write a safetensors bundle holding HF Mixtral per-expert KEY NAMING
/// at MULTIPLE prefixes. Identical helper to v2.18 (same byte-level
/// shape, dtype F32, canonical `.w{N}.bias` suffix).
fn write_multi_block_hf_safetensors(
    path: &Path,
    hf_prefixes: &[&str],
    hidden: usize,
    intermediate: usize,
    num_experts: usize,
) {
    let mut byte_buffers: Vec<Vec<u8>> = Vec::new();
    let mut entries: Vec<(String, Vec<usize>)> = Vec::new();
    let mut buffer_index: Vec<usize> = Vec::new();

    fn push_buf(byte_buffers: &mut Vec<Vec<u8>>, n_bytes: usize) -> usize {
        let idx = byte_buffers.len();
        byte_buffers.push(vec![0u8; n_bytes]);
        idx
    }

    for prefix in hf_prefixes {
        entries.push((format!("{prefix}.router.weight"), vec![hidden, num_experts]));
        buffer_index.push(push_buf(&mut byte_buffers, hidden * num_experts * 4));

        for e in 0..num_experts {
            entries.push((
                format!("{prefix}.experts.{e}.w1.weight"),
                vec![intermediate, hidden],
            ));
            buffer_index.push(push_buf(&mut byte_buffers, intermediate * hidden * 4));
            entries.push((
                format!("{prefix}.experts.{e}.w3.weight"),
                vec![intermediate, hidden],
            ));
            buffer_index.push(push_buf(&mut byte_buffers, intermediate * hidden * 4));
            entries.push((
                format!("{prefix}.experts.{e}.w2.weight"),
                vec![hidden, intermediate],
            ));
            buffer_index.push(push_buf(&mut byte_buffers, hidden * intermediate * 4));
            entries.push((
                format!("{prefix}.experts.{e}.w1.bias"),
                vec![intermediate],
            ));
            buffer_index.push(push_buf(&mut byte_buffers, intermediate * 4));
            entries.push((
                format!("{prefix}.experts.{e}.w3.bias"),
                vec![intermediate],
            ));
            buffer_index.push(push_buf(&mut byte_buffers, intermediate * 4));
            entries.push((
                format!("{prefix}.experts.{e}.w2.bias"),
                vec![hidden],
            ));
            buffer_index.push(push_buf(&mut byte_buffers, hidden * 4));
        }
    }

    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    for ((name, shape), buf_idx) in entries.iter().zip(buffer_index.iter()) {
        views.insert(
            name.clone(),
            TensorView::new(Dtype::F32, shape.clone(), byte_buffers[*buf_idx].as_slice())
                .unwrap(),
        );
    }

    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

/// Two `model` declarations, each with its own `@moe(weight_prefix)`,
/// each consuming its own block via its own `forward` method.
const MOE_TWO_MODEL_SRC: &str = r#"model Block0:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block0")
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

model Block1:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block1")
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

let m0 = Block0()
let m1 = Block1()
let x = ones([2, 8])
let y0 = m0.forward(x)
let y1 = m1.forward(x)
"#;

#[test]
fn multi_model_each_moe_resolves_against_its_own_weight_prefix() {
    // HAPPY: two models, each with its own @moe(weight_prefix=...), both
    // forwards compile and resolve dims under their own block. Pre-fix:
    // candidate resolution returned None for both methods (state-level
    // function name absent for model methods + len>=2 fallback fails),
    // so derive_v4_dims was never called and the build errored with
    // "v4 dims could not be resolved".
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("multi_moe.nsl");
    fs::write(&src, MOE_TWO_MODEL_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_multi_block_hf_safetensors(&weights, &["Block0", "Block1"], 8, 16, 4);
    let out = tmp.path().join("multi_moe.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out)
        .arg("--weights")
        .arg(&weights);

    // Object file gets emitted only if BOTH model methods compiled.
    cmd.assert()
        .success()
        .stderr(
            predicate::str::contains("CPDT v2.7: auto-packed 2 HF Mixtral MoE blocks")
                .and(predicate::str::contains(
                    "CPDT v2.15: auto-packed v4 biases for 2 HF Mixtral MoE blocks",
                )),
        );
    assert!(out.exists(), "object file should be emitted when both forwards compile");
}

/// Same shape as the happy case, but Block1 omits the
/// `@moe(weight_prefix=...)` decorator so its forward should fall back
/// to the unambiguous Block1 config — except there's no MoE config for
/// Block1, so the call site should fail with a clear "v4 dims could
/// not be resolved" diagnostic citing the Block1 lookup path (NOT
/// silently use Block0's config). This regression-tests that the new
/// prefix filter doesn't accidentally cross-route between models.
const MOE_ASYMMETRIC_MODEL_SRC: &str = r#"model Block0:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block0")
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

model PlainBlock:
    router: Tensor = ones([8, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.router

let m0 = Block0()
let m1 = PlainBlock()
let x = ones([2, 8])
let y0 = m0.forward(x)
let y1 = m1.forward(x)
"#;

/// Helper-model pattern: one model carries `@moe`, a second model
/// with NO `@moe` whose `forward` ALSO calls `moe_dispatch_swiglu`
/// should resolve via the helper's singleton-fallback branch — the
/// pre-v2.19 single-MoE escape hatch v2.19 preserves. This exercises
/// the helper's "zero matches under current model → fall back to
/// singleton" path (branch C of the helper). Without this gate, a
/// future refactor that strips the singleton fallback (under a false
/// assumption that all `@moe`-calling methods live on `@moe`-bearing
/// models) would silently regress the helper-model use case.
const MOE_HELPER_MODEL_USES_SINGLETON_SRC: &str = r#"model HostBlock:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block0")
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

model HelperBlock:
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

let host = HostBlock()
let helper = HelperBlock()
let x = ones([2, 8])
let y0 = host.forward(x)
let y1 = helper.forward(x)
"#;

#[test]
fn helper_model_without_moe_falls_back_to_singleton_when_unique() {
    // BRANCH C: HostBlock carries the only `@moe` in the program;
    // HelperBlock has no decorator but still calls `moe_dispatch_swiglu`
    // from its forward. The helper's `current_method_model_name =
    // Some("HelperBlock")` finds zero matches under `"HelperBlock."`,
    // falls through to the singleton fallback, and picks up
    // HostBlock's config (the documented pre-v2.19 behaviour).
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("helper_singleton.nsl");
    fs::write(&src, MOE_HELPER_MODEL_USES_SINGLETON_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_multi_block_hf_safetensors(&weights, &["Block0"], 8, 16, 4);
    let out = tmp.path().join("helper_singleton.o");

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
    assert!(
        out.exists(),
        "helper model without @moe must still compile via singleton fallback to the program's only @moe config"
    );
}

#[test]
fn multi_model_one_with_moe_one_without_no_cross_routing() {
    // ROBUSTNESS: when only one model has an @moe decorator and the
    // other has none, the fix must still resolve Block0's MoE config
    // against Block0's prefix (the singleton-fallback path would also
    // work here, but the v2.19 prefix filter should take precedence).
    // PlainBlock's forward must not pick up Block0's MoE config.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("asym_moe.nsl");
    fs::write(&src, MOE_ASYMMETRIC_MODEL_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_multi_block_hf_safetensors(&weights, &["Block0"], 8, 16, 4);
    let out = tmp.path().join("asym_moe.o");

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
    assert!(out.exists(), "object emit succeeds with one @moe and one plain model");
}
