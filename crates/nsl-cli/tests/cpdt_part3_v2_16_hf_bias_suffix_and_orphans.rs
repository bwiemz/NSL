//! CPDT Part III v2.16 — HF auto-pack hardening: `.b{1,2,3}` short-form
//! bias suffix support, bias-only-no-weights detection, enumerate-all-
//! orphans diagnostic.
//!
//! Closes the three v2.15 next-deferrals:
//!   - v2.16-A: `.b{1,2,3}` short form (early Mixtral converter variant)
//!     accepted alongside the canonical `.w{1,2,3}.bias` form. Per-block
//!     uniformity invariant — mixing forms is refused.
//!   - v2.16-B: bias-only-no-weights prefixes surface as a hard
//!     `BiasesWithoutWeights` build error at load time (was: silently
//!     dropped; user got a less obvious `derive_v4_dims` error later).
//!   - v2.16-C: `ExtraBiasesPresent` diagnostic enumerates ALL orphan
//!     indices (was: only the maximum, forcing rebuild-once-per-orphan).
//!
//! CLI gates pin the user-facing behavior end-to-end through the build:
//!   - HAPPY: short-form HF biases auto-pack alongside HF weights, the
//!     8-arg `moe_dispatch_swiglu` source compiles.
//!   - REFUSAL: bias-only-no-weights surfaces "biases-without-weights"
//!     hard error.
//!   - REFUSAL: mixed canonical+short suffix forms surfaces "mixes the
//!     canonical" hard error.

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

/// Bias suffix form to use in safetensors writing.
#[derive(Debug, Clone, Copy)]
enum SuffixForm {
    /// Canonical `.w{N}.bias`
    Canonical,
    /// Short `.b{N}`
    Short,
    /// Mixed — expert 0 uses canonical, expert 1+ uses short (refusal test)
    Mixed,
}

/// Write a safetensors bundle with HF Mixtral per-expert KEY NAMING.
/// `bias_form` controls which bias-suffix convention is written.
/// `write_weights` controls whether weight keys are written (false →
/// bias-only-no-weights scenario for v2.16-B).
fn write_hf_safetensors(
    path: &Path,
    hf_prefix: &str,
    hidden: usize,
    intermediate: usize,
    num_experts: usize,
    write_weights: bool,
    with_biases: bool,
    bias_form: SuffixForm,
) {
    let router_bytes: Vec<u8> = vec![0u8; hidden * num_experts * 4];
    let gate_bytes_per_expert: Vec<u8> = vec![0u8; intermediate * hidden * 4];
    let up_bytes_per_expert: Vec<u8> = vec![0u8; intermediate * hidden * 4];
    let down_bytes_per_expert: Vec<u8> = vec![0u8; hidden * intermediate * 4];
    let gate_bias_per_expert: Vec<u8> = vec![0u8; intermediate * 4];
    let up_bias_per_expert: Vec<u8> = vec![0u8; intermediate * 4];
    let down_bias_per_expert: Vec<u8> = vec![0u8; hidden * 4];

    let router_view = TensorView::new(
        Dtype::F32,
        vec![hidden, num_experts],
        router_bytes.as_slice(),
    )
    .unwrap();

    let mut weight_buffers: Vec<Vec<u8>> = Vec::with_capacity(3 * num_experts);
    let mut bias_buffers: Vec<Vec<u8>> = Vec::with_capacity(3 * num_experts);
    for _ in 0..num_experts {
        weight_buffers.push(gate_bytes_per_expert.clone());
        weight_buffers.push(up_bytes_per_expert.clone());
        weight_buffers.push(down_bytes_per_expert.clone());
        bias_buffers.push(gate_bias_per_expert.clone());
        bias_buffers.push(up_bias_per_expert.clone());
        bias_buffers.push(down_bias_per_expert.clone());
    }

    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    if write_weights {
        views.insert(format!("{hf_prefix}.router.weight"), router_view);
    }

    for e in 0..num_experts {
        let w1_idx = e * 3;
        let w3_idx = e * 3 + 1;
        let w2_idx = e * 3 + 2;
        if write_weights {
            views.insert(
                format!("{hf_prefix}.experts.{e}.w1.weight"),
                TensorView::new(
                    Dtype::F32,
                    vec![intermediate, hidden],
                    weight_buffers[w1_idx].as_slice(),
                )
                .unwrap(),
            );
            views.insert(
                format!("{hf_prefix}.experts.{e}.w3.weight"),
                TensorView::new(
                    Dtype::F32,
                    vec![intermediate, hidden],
                    weight_buffers[w3_idx].as_slice(),
                )
                .unwrap(),
            );
            views.insert(
                format!("{hf_prefix}.experts.{e}.w2.weight"),
                TensorView::new(
                    Dtype::F32,
                    vec![hidden, intermediate],
                    weight_buffers[w2_idx].as_slice(),
                )
                .unwrap(),
            );
        }

        if with_biases {
            let b1_idx = e * 3;
            let b3_idx = e * 3 + 1;
            let b2_idx = e * 3 + 2;
            // Per-expert form selection. SuffixForm::Mixed uses
            // canonical for expert 0, short for others.
            let use_short = match bias_form {
                SuffixForm::Canonical => false,
                SuffixForm::Short => true,
                SuffixForm::Mixed => e > 0,
            };
            let (g, u, d) = if use_short {
                ("b1", "b3", "b2")
            } else {
                ("w1.bias", "w3.bias", "w2.bias")
            };
            views.insert(
                format!("{hf_prefix}.experts.{e}.{g}"),
                TensorView::new(Dtype::F32, vec![intermediate], bias_buffers[b1_idx].as_slice())
                    .unwrap(),
            );
            views.insert(
                format!("{hf_prefix}.experts.{e}.{u}"),
                TensorView::new(Dtype::F32, vec![intermediate], bias_buffers[b3_idx].as_slice())
                    .unwrap(),
            );
            views.insert(
                format!("{hf_prefix}.experts.{e}.{d}"),
                TensorView::new(Dtype::F32, vec![hidden], bias_buffers[b2_idx].as_slice())
                    .unwrap(),
            );
        }
    }

    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

/// 8-arg source. Used for the v2.16-A happy path (HF weights + short-
/// form HF biases auto-pack into NSL convention, then 8-arg compiles).
const MOE_SWIGLU_8ARG_HF_SRC: &str = r#"model Block:
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
fn short_form_b_suffix_biases_auto_pack_and_8arg_compiles() {
    // v2.16-A HAPPY: HF weights + short-form `.b{1,2,3}` biases both
    // auto-pack; 8-arg moe_dispatch_swiglu compiles. Verifies the
    // v2.15 telemetry fires identically when biases use either
    // suffix convention.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_8ARG_HF_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_hf_safetensors(
        &weights, "Block", 8, 16, 4,
        true,                   // write weights
        true,                   // with biases
        SuffixForm::Short,      // short form .b{N}
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

    cmd.assert()
        .success()
        .stderr(
            predicate::str::contains("CPDT v2.7: auto-packed 1 HF Mixtral MoE block")
                .and(predicate::str::contains(
                    "CPDT v2.15: auto-packed v4 biases for 1 HF Mixtral MoE block",
                )),
        );
}

#[test]
fn mixed_canonical_and_short_form_suffixes_fails_build_loudly() {
    // v2.16-A REFUSAL: expert 0 uses `.w{N}.bias`, expert 1+ uses
    // `.b{N}`. Per-block uniformity invariant trips
    // MixedBiasSuffixForms.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_8ARG_HF_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_hf_safetensors(
        &weights, "Block", 8, 16, 4,
        true,
        true,
        SuffixForm::Mixed,
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
        predicate::str::contains("HF Mixtral auto-pack failed")
            .and(predicate::str::contains(
                "mixes the canonical",
            )),
    );
}

#[test]
fn bias_only_no_weights_orphan_prefix_fails_build_loudly() {
    // v2.16-B REFUSAL: the safetensors has biases at "Block.*" but
    // NO weight keys. find_bias_only_prefixes detects the orphan
    // prefix; BiasesWithoutWeights surfaces as a hard build error.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_8ARG_HF_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_hf_safetensors(
        &weights, "Block", 8, 16, 4,
        false,                  // NO weights
        true,                   // with biases
        SuffixForm::Canonical,
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
        predicate::str::contains("HF Mixtral auto-pack failed")
            .and(predicate::str::contains("biases-without-weights"))
            .and(predicate::str::contains(
                "per-expert bias keys but NO per-expert weight keys",
            )),
    );
}
