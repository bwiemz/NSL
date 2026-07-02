//! CPDT Part III v2.12 — source-level activation of v3 FFN biases.
//!
//! v2.11 extended `nsl_moe_dispatch_full_v3` with nullable bias args at
//! the FFI level but always emitted 0/0 from codegen. v2.12 adds:
//!
//!   1. `moe_dispatch_ffn` 6-arg form `(tokens, logits, experts_up,
//!      experts_down, bias_up, bias_down)` — bias args are now compiled
//!      and threaded through to the FFI instead of `iconst(0)`.
//!   2. Source-activation gate — when the WeightMap has v3 biases
//!      (`<prefix>.experts.up.bias` + `<prefix>.experts.down.bias`) but
//!      the call site uses the 4-arg form, codegen refuses LOUDLY (no
//!      silent drop of the loaded biases).
//!   3. Partial-bias refusal — only one of up/down bias present in the
//!      WeightMap is almost always a packing bug; codegen refuses with a
//!      diagnostic naming the missing direction.
//!   4. Bias shape mismatch refusal — total element count must match
//!      `num_experts * intermediate_dim` (up) or `num_experts *
//!      hidden_dim` (down). Off-by-one in those dims would otherwise
//!      land as silent-corruption OOB reads at runtime.
//!
//! Success-path coverage (6-arg lowering emitting the right FFI args)
//! lives in the codegen unit tests; the FFI's bias-application path is
//! covered by `moe_dispatch_v3_paper_faithful_ffn.rs`. These CLI gates
//! pin the *diagnostic* paths that the source-facing user actually hits.
//!
//! WHY these tests are at the CLI layer: the v3 lowering arm only fires
//! when a model-level `@moe(...)` decorator populates `moe_configs`,
//! which requires the full pipeline (parser → semantic → codegen). A
//! codegen-unit test would need test-only scaffolding that doesn't pay
//! for itself given the matching v2.3/v2.7/v2.8 CLI gates already in
//! place.

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

/// Write a safetensors bundle that exposes the v3 layout
/// (`Block.router.weight`, `Block.experts.up.weight`,
/// `Block.experts.down.weight`) with the requested optional biases. The
/// `Block.` prefix matches the model-class name used in MOE_BLOCK_SRC.
fn write_v3_safetensors(
    path: &Path,
    hidden: usize,
    intermediate: usize,
    num_experts: usize,
    up_bias_elems: Option<usize>,
    down_bias_elems: Option<usize>,
) {
    let router_bytes: Vec<u8> = vec![0u8; hidden * num_experts * 4];
    let up_bytes: Vec<u8> = vec![0u8; num_experts * hidden * intermediate * 4];
    let down_bytes: Vec<u8> = vec![0u8; num_experts * intermediate * hidden * 4];
    let up_bias_bytes: Option<Vec<u8>> = up_bias_elems.map(|n| vec![0u8; n * 4]);
    let down_bias_bytes: Option<Vec<u8>> = down_bias_elems.map(|n| vec![0u8; n * 4]);

    let router_view =
        TensorView::new(Dtype::F32, vec![hidden, num_experts], router_bytes.as_slice()).unwrap();
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
    views.insert("Block.experts.up.weight".to_string(), up_view);
    views.insert("Block.experts.down.weight".to_string(), down_view);

    // Bias views must live as long as their bytes Vecs — held in scope
    // by the Option<Vec<u8>> bindings above. The serialize call below
    // captures them via the views map.
    let up_bias_view: Option<TensorView<'_>> = up_bias_bytes.as_ref().map(|bytes| {
        TensorView::new(Dtype::F32, vec![bytes.len() / 4], bytes.as_slice()).unwrap()
    });
    let down_bias_view: Option<TensorView<'_>> = down_bias_bytes.as_ref().map(|bytes| {
        TensorView::new(Dtype::F32, vec![bytes.len() / 4], bytes.as_slice()).unwrap()
    });
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

/// 4-arg `moe_dispatch_ffn` source — must refuse when WeightMap has
/// biases (v2.12 source-activation gate). The `@moe` decorator lives on
/// a field (not the model class) so the collection phase populates
/// `moe_configs`; `weight_prefix="Block"` points the WeightMap lookup
/// at the safetensors prefix `Block.*` (matches `write_v3_safetensors`
/// below). This mirrors the v2.7 CLI-gate convention.
///
/// v2.12 fix F10 (IMPORTANT adversarial review): the `experts_dummy:
/// int = 4` field is the parse-required carrier for the `@moe`
/// decorator. The decorator must attach to a NON-TENSOR field (compiler
/// collection phase at `compiler/collection.rs:548` only picks up
/// field-level decorators); a tensor-typed field would change whether
/// the decorator fires AT ALL. This is the v2.7 convention pinned in
/// every CPDT-v2.x CLI gate.
const MOE_FFN_4ARG_SRC: &str = r#"model Block:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block")
    experts_dummy: int = 4
    router: Tensor = ones([8, 4])
    experts_up: Tensor = ones([4, 8 * 16])
    experts_down: Tensor = ones([4, 16 * 8])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_ffn(x, logits, self.experts_up, self.experts_down)

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;

/// 6-arg form — success path. v2.12 fix F4 (HIGH adversarial review):
/// pins that a 6-arg `moe_dispatch_ffn` call with model fields holding
/// bias tensors emits a successful build (and the v2.12 source-
/// activation gate does NOT mis-fire when the bias fields are present).
/// This is the happy path for GPT-2 / OPT FFN-bias families.
const MOE_FFN_6ARG_SRC: &str = r#"model Block:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block")
    experts_dummy: int = 4
    router: Tensor = ones([8, 4])
    experts_up: Tensor = ones([4, 8 * 16])
    experts_down: Tensor = ones([4, 16 * 8])
    experts_up_bias: Tensor = ones([4, 16])
    experts_down_bias: Tensor = ones([4, 8])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_ffn(
            x, logits,
            self.experts_up, self.experts_down,
            self.experts_up_bias, self.experts_down_bias,
        )

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;

#[test]
fn moe_dispatch_ffn_4arg_with_weight_map_biases_refuses_loudly() {
    // WeightMap has both biases; 4-arg call would silently drop them
    // and silently produce wrong output. v2.12 gate must fire.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_4ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    // hidden=8, intermediate=16, num_experts=4
    write_v3_safetensors(&weights, 8, 16, 4, Some(4 * 16), Some(4 * 8));
    let out = tmp.path().join("moe_ffn.o");

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
        predicate::str::contains("moe_dispatch_ffn")
            .and(predicate::str::contains("v3 FFN biases"))
            .and(predicate::str::contains("6-arg form")),
    );
}

#[test]
fn moe_dispatch_ffn_4arg_with_partial_up_bias_only_refuses_loudly() {
    // Up bias present, down bias missing. This is almost always a
    // packing bug (stale prune, mismatched bundle). v2.12 detect must
    // refuse with the partial-bias diagnostic — the user notices the
    // bug at build time, not at silent wrong output.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_4ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v3_safetensors(&weights, 8, 16, 4, Some(4 * 16), None);
    let out = tmp.path().join("moe_ffn.o");

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
        predicate::str::contains("partial-bias bundle")
            .and(predicate::str::contains("up.bias")),
    );
}

#[test]
fn moe_dispatch_ffn_4arg_with_partial_down_bias_only_refuses_loudly() {
    // Symmetric to the up-only case. Refusal must name `down.bias`
    // specifically so the diagnostic points at the actual missing
    // direction (not just a generic "partial bias" message that would
    // be ambiguous about which side is missing).
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_4ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v3_safetensors(&weights, 8, 16, 4, None, Some(4 * 8));
    let out = tmp.path().join("moe_ffn.o");

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
        predicate::str::contains("partial-bias bundle")
            .and(predicate::str::contains("down.bias")),
    );
}

#[test]
fn moe_dispatch_ffn_4arg_with_mismatched_bias_shape_refuses_loudly() {
    // Up bias has 4 * 15 = 60 elems; expected 4 * 16 = 64. Off-by-one
    // in intermediate_dim would otherwise land as silent-corruption
    // OOB reads at runtime (the FFI would refuse on len mismatch, but
    // the codegen diagnostic is far more actionable because it points
    // at the actual bundle source).
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_4ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v3_safetensors(&weights, 8, 16, 4, Some(4 * 15), Some(4 * 8));
    let out = tmp.path().join("moe_ffn.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out)
        .arg("--weights")
        .arg(&weights);

    // v2.12 fix F8 (IMPORTANT adversarial review): the predicate ties
    // the numeric values to the "elements" phrase so they cannot
    // false-match on incidental numbers in compiler output (e.g., line
    // numbers, byte counts, architecture strings).
    cmd.assert().failure().stderr(
        predicate::str::contains("up.bias")
            .and(predicate::str::contains("60 elements"))
            .and(predicate::str::contains("expected 64")),
    );
}

#[test]
fn moe_dispatch_ffn_4arg_with_no_biases_still_succeeds() {
    // Sanity: v2.5/v2.7 v3-emission path must remain byte-identical
    // when the WeightMap has NO bias entries. The 4-arg form is the
    // happy path for Llama-1+ / Mixtral-SwiGLU-without-bias bundles;
    // it must NOT trigger the source-activation gate.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_4ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_v3_safetensors(&weights, 8, 16, 4, None, None);
    let out = tmp.path().join("moe_ffn.o");

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
fn moe_dispatch_ffn_wrong_arity_5_refuses_loudly() {
    // v2.12 fix F9 (IMPORTANT adversarial review): renamed from
    // `wrong_arity_5_or_7_refuses_loudly`. The arity check refuses any
    // arity ∉ {4, 6}, but this test only exercises the 5-arg case
    // (most-common typo: one bias forgotten). 7-arg coverage would
    // require a duplicate source fixture for marginal value (the
    // `args.len() != 4 && args.len() != 6` predicate is single-
    // codepath — if it accepts 5, it'd accept 7 too).
    let bad_src = r#"model Block:
    @moe(num_experts=4, top_k=1, capacity_factor=2.0, weight_prefix="Block")
    experts_dummy: int = 4
    router: Tensor = ones([8, 4])
    experts_up: Tensor = ones([4, 8 * 16])
    experts_down: Tensor = ones([4, 16 * 8])
    experts_up_bias: Tensor = ones([4, 16])

    fn forward(self, x: Tensor) -> Tensor:
        let logits = x @ self.router
        return moe_dispatch_ffn(x, logits, self.experts_up, self.experts_down, self.experts_up_bias)

let m = Block()
let x = ones([2, 8])
let y = m.forward(x)
"#;
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, bad_src).unwrap();
    let out = tmp.path().join("moe_ffn.o");

    let mut cmd = Command::cargo_bin("nsl").unwrap();
    cmd.env("NSL_STDLIB_PATH", stdlib_path());
    cmd.arg("build")
        .arg(&src)
        .arg("--emit-obj")
        .arg("-o")
        .arg(&out);

    cmd.assert().failure().stderr(
        predicate::str::contains("moe_dispatch_ffn")
            .and(predicate::str::contains("4 arguments"))
            .and(predicate::str::contains("6 arguments")),
    );
}

#[test]
fn moe_dispatch_ffn_6arg_form_with_weight_map_biases_succeeds() {
    // v2.12 fix F4 (HIGH adversarial review): 6-arg success path. The
    // user has loaded a bundle with both biases AND uses the 6-arg
    // source form. Build must SUCCEED — the v2.12 detection gate
    // returned `Ok(Some(()))` but the source-form match prevents the
    // refusal from firing. Regressions that accidentally refuse all
    // bias-present bundles (or accidentally always emit iconst(0)
    // instead of threading bias_vals) would change the .success()
    // result here.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_6ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    // hidden=8, intermediate=16, num_experts=4 — matches MOE_FFN_6ARG_SRC.
    write_v3_safetensors(&weights, 8, 16, 4, Some(4 * 16), Some(4 * 8));
    let out = tmp.path().join("moe_ffn.o");

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
        "build should emit object file at {:?}",
        out,
    );
    let obj_bytes = fs::metadata(&out).unwrap().len();
    assert!(obj_bytes > 0, "emitted object file must not be empty");
}

#[test]
fn moe_dispatch_ffn_4arg_with_orphaned_biases_no_weights_refuses_loudly() {
    // v2.12 fix F2 (HIGH adversarial review): biases present but
    // weight dims fail to resolve (e.g., the user shipped a bundle
    // missing `experts.up.weight`). Pre-v2.12-F2, the diagnostic
    // would be the generic "v3 dims not resolvable" message, which
    // doesn't mention the orphaned bias entries. The v2.12 fix
    // surfaces the orphan condition specifically.
    //
    // To exercise this path: write a safetensors bundle that contains
    // the biases but NO weight tensors at all. `derive_v3_dims` returns
    // None, then the v2.12 pre-pass fires the orphan diagnostic.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_4ARG_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_orphaned_bias_only_safetensors(&weights);
    let out = tmp.path().join("moe_ffn.o");

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
        predicate::str::contains("orphaned-bias bundle")
            .and(predicate::str::contains("Block")),
    );
}

/// Write a bundle that has ONLY bias entries under the `Block.*` prefix
/// — no router, no `experts.up.weight`, no `experts.down.weight`. Used
/// to exercise the v2.12 fix F2 orphaned-bias path where dim derivation
/// fails but biases are still detected.
fn write_orphaned_bias_only_safetensors(path: &Path) {
    let up_bias_bytes = vec![0u8; 4 * 16 * 4];
    let down_bias_bytes = vec![0u8; 4 * 8 * 4];
    let up_view =
        TensorView::new(Dtype::F32, vec![4 * 16], up_bias_bytes.as_slice()).unwrap();
    let down_view =
        TensorView::new(Dtype::F32, vec![4 * 8], down_bias_bytes.as_slice()).unwrap();
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    views.insert("Block.experts.up.bias".to_string(), up_view);
    views.insert("Block.experts.down.bias".to_string(), down_view);
    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}
