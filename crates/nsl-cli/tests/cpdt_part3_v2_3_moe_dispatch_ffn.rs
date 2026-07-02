//! CPDT Part III v2.3 codegen lowering gate:
//! `moe_dispatch_ffn(tokens, logits, experts_up, experts_down)` is the
//! opt-in source intrinsic that emits `nsl_moe_dispatch_full_v3` (the
//! paper-faithful up→SiLU→down MoE FFN). v2.3 introduces the lowering;
//! this CLI test pins the load-bearing safety gate:
//!
//!   Hard error when v3 dims can't be derived (no silent fallback to v2
//!   from this intrinsic). The error message must be v3-specific so a
//!   future regression that silently routed moe_dispatch_ffn through v2
//!   would show up as a stderr-content mismatch.
//!
//! The success path (router + experts.up + experts.down all resolvable
//! → object file emission) is covered by the codegen unit tests
//! (`derive_v3_dims_*` in nsl-codegen/src/moe.rs) plus the runtime
//! FFI tests (`moe_dispatch_v3_paper_faithful_ffn.rs`). A model-level
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

/// Write a minimal safetensors file that is VALID for CPDT setup but
/// contains no router/experts.up/experts.down entries. moe_dispatch_ffn
/// lowering must surface its v3-specific error rather than silently
/// emitting v2 or v1.
fn write_unrelated_safetensors(path: &Path) {
    let bytes: Vec<u8> = 1.0f32.to_le_bytes().to_vec();
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    let view = TensorView::new(Dtype::F32, vec![1], bytes.as_slice()).unwrap();
    views.insert("unrelated.weight".to_string(), view);
    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

const MOE_FFN_SRC: &str = r#"from nsl.nn.losses import mse_loss

@moe(num_experts=4, top_k=1, capacity_factor=2.0)
model Block:
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

#[test]
fn moe_dispatch_ffn_with_unrelated_weights_errors_loudly() {
    // moe_dispatch_ffn is an OPT-IN intrinsic — there is no silent v2
    // fallback. If the WeightMap doesn't contain router + experts.up +
    // experts.down, the build must fail with the v3-specific diagnostic
    // (not the v2 diagnostic, since the user explicitly chose v3 by
    // calling moe_dispatch_ffn). Pinned for both CPDT modes — none of
    // the silent-fallback behavior should apply.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_ffn.nsl");
    fs::write(&src, MOE_FFN_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_unrelated_safetensors(&weights);
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
            .and(predicate::str::contains("v3 (paper-faithful FFN) requires resolvable router + experts.up + experts.down"))
    );
}

