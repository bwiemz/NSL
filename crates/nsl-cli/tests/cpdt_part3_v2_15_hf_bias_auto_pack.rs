//! CPDT Part III v2.15 — HF Mixtral per-expert bias auto-pack CLI gates.
//!
//! Closes the v2.7+v2.12+v2.13+v2.14 one-sided composition gap: v2.7
//! auto-pack rewrote per-expert HF weight keys
//! (`<prefix>.experts.{e}.w{1,2,3}.weight`) into NSL packed convention
//! (`<prefix>.experts.{gate,up,down}.weight`) but did NOT touch the
//! matching per-expert HF bias keys. Users shipping HF Mixtral
//! checkpoints WITH biases had to pre-pack the biases via a one-shot
//! script before `detect_v4_biases` would resolve them at the v4
//! codegen lowering. v2.15 extends the auto-pack to also rewrite
//! `.w{1,2,3}.bias` into `.{gate,up,down}.bias`.
//!
//! These tests pin the user-facing behavior end-to-end through the
//! CLI:
//!   - HAPPY (A): HF weights + HF biases both auto-pack → 8-arg
//!     `moe_dispatch_swiglu` source compiles. Verify the v2.15
//!     telemetry on stderr.
//!   - REFUSAL (B): HF weights + PARTIAL bias bundle → build fails
//!     with the bias-pack failure surfaced through entry_points.rs.
//!   - HAPPY-NOOP (C): HF weights only (no biases) → 5-arg source
//!     compiles. v2.7 telemetry fires; v2.15 telemetry does NOT.

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

/// Write a safetensors bundle with HF Mixtral per-expert KEY NAMING for
/// the weights (and optionally biases). `with_biases` controls whether
/// per-expert `.w{1,2,3}.bias` entries are also written.
///
/// `bias_partial` (when `with_biases=true`) selects a partial-bundle
/// state for the refusal tests:
///   - None        → all 3 directions × all experts present (full)
///   - Some("gate_only")  → only `.w1.bias` for all experts present
fn write_hf_safetensors(
    path: &Path,
    hf_prefix: &str,
    hidden: usize,
    intermediate: usize,
    num_experts: usize,
    with_biases: bool,
    bias_partial: Option<&str>,
) {
    // Router stays under NSL convention — the auto-pack doesn't touch
    // it (router weights aren't per-expert).
    let router_bytes: Vec<u8> = vec![0u8; hidden * num_experts * 4];
    // Per-expert HF weights: shape [out, in] per nn.Linear convention.
    // w1 (gate) = [intermediate, hidden]; w3 (up) = same; w2 (down) =
    // [hidden, intermediate].
    let gate_bytes_per_expert: Vec<u8> = vec![0u8; intermediate * hidden * 4];
    let up_bytes_per_expert: Vec<u8> = vec![0u8; intermediate * hidden * 4];
    let down_bytes_per_expert: Vec<u8> = vec![0u8; hidden * intermediate * 4];
    // Per-expert biases (when present): w1.bias = [intermediate],
    // w3.bias = [intermediate], w2.bias = [hidden].
    let gate_bias_per_expert: Vec<u8> = vec![0u8; intermediate * 4];
    let up_bias_per_expert: Vec<u8> = vec![0u8; intermediate * 4];
    let down_bias_per_expert: Vec<u8> = vec![0u8; hidden * 4];

    // Build TensorViews holding references into the byte buffers above.
    // safetensors::serialize requires &TensorView with the buffer alive
    // for the duration of the serialize call.
    let router_view = TensorView::new(
        Dtype::F32,
        vec![hidden, num_experts],
        router_bytes.as_slice(),
    )
    .unwrap();

    // Per-expert weights need fresh buffers per (e, projection) so the
    // safetensors serializer doesn't see aliased slices.
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
    views.insert(format!("{hf_prefix}.router.weight"), router_view);

    for e in 0..num_experts {
        let w1_idx = e * 3;
        let w3_idx = e * 3 + 1;
        let w2_idx = e * 3 + 2;
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

        if with_biases {
            let b1_idx = e * 3;
            let b3_idx = e * 3 + 1;
            let b2_idx = e * 3 + 2;
            // For the partial-bundle test, only emit w1.bias.
            let emit_w3 = bias_partial != Some("gate_only");
            let emit_w2 = bias_partial != Some("gate_only");
            views.insert(
                format!("{hf_prefix}.experts.{e}.w1.bias"),
                TensorView::new(Dtype::F32, vec![intermediate], bias_buffers[b1_idx].as_slice())
                    .unwrap(),
            );
            if emit_w3 {
                views.insert(
                    format!("{hf_prefix}.experts.{e}.w3.bias"),
                    TensorView::new(
                        Dtype::F32,
                        vec![intermediate],
                        bias_buffers[b3_idx].as_slice(),
                    )
                    .unwrap(),
                );
            }
            if emit_w2 {
                views.insert(
                    format!("{hf_prefix}.experts.{e}.w2.bias"),
                    TensorView::new(Dtype::F32, vec![hidden], bias_buffers[b2_idx].as_slice())
                        .unwrap(),
                );
            }
        }
    }

    let bytes = serialize(&views, &None).unwrap();
    let mut f = fs::File::create(path).unwrap();
    f.write_all(&bytes).unwrap();
}

/// 8-arg `moe_dispatch_swiglu` source under `Block` prefix. Mirrors
/// the v2.14 8-arg source — the model fields use NSL convention names
/// (`router`, `experts_gate`, etc.) because the auto-pack rewrites the
/// HF keys into those before codegen. The `experts_dummy: int = 4`
/// field is the v2.7-pinned non-Tensor decorator anchor.
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

/// 5-arg source — no model-level bias fields. Used for the no-bias
/// happy-path verification.
const MOE_SWIGLU_5ARG_HF_SRC: &str = r#"model Block:
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

#[test]
fn hf_weights_and_biases_both_auto_pack_then_8arg_compiles() {
    // HAPPY (A): HF Mixtral per-expert weights AND biases present.
    // The v2.7 weight pack rewrites the .w{1,2,3}.weight keys; v2.15
    // bias pack rewrites the .w{1,2,3}.bias keys. The 8-arg
    // moe_dispatch_swiglu source then compiles cleanly.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_8ARG_HF_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_hf_safetensors(&weights, "Block", 8, 16, 4, true, None);
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

    // Build succeeds AND both auto-pack telemetry lines appear on
    // stderr — verifies the bias pack ran alongside the weight pack.
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
fn hf_weights_with_partial_bias_bundle_fails_build_loudly() {
    // REFUSAL (B): HF weights present + only .w1.bias per expert
    // present (no .w2.bias, no .w3.bias). The weight pack succeeds;
    // the bias pack refuses with PartialBiasBundle; entry_points.rs
    // surfaces it as a CodegenError. The build fails BEFORE codegen
    // gets to run the 8-arg lowering against half-loaded biases.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_8ARG_HF_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_hf_safetensors(&weights, "Block", 8, 16, 4, true, Some("gate_only"));
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
            .and(predicate::str::contains("biases"))
            .and(predicate::str::contains("partial HF bias bundle"))
            .and(predicate::str::contains("w2.bias")),
    );
}

#[test]
fn hf_weights_only_no_biases_compiles_with_5arg_and_v2_15_telemetry_silent() {
    // HAPPY-NOOP (C): HF weights present, NO bias keys. The v2.7
    // weight pack fires (telemetry on stderr); the v2.15 bias pack
    // hits Ok(None) and produces NO telemetry. The 5-arg
    // moe_dispatch_swiglu source compiles cleanly.
    //
    // Pins the no-bias-block contract: a single shared CLI run path
    // for both bias-bearing and bias-free HF checkpoints. The bias
    // pass must not produce false-positive telemetry for the common
    // bias-free case.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_swiglu.nsl");
    fs::write(&src, MOE_SWIGLU_5ARG_HF_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_hf_safetensors(&weights, "Block", 8, 16, 4, false, None);
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
                .and(predicate::str::contains("v2.15:").not()),
        );
}
