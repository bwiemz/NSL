//! CPDT Part III v2.18 — multi-block HF Mixtral auto-pack telemetry.
//!
//! Real production Mixtral checkpoints have 32+ MoE blocks (one per
//! transformer layer). Every prior CPDT v2.x CLI gate exercised a
//! SINGLE block. The lib tests
//! (`detect_hf_mixtral_blocks_finds_multiple_blocks_sorted_by_prefix`
//! and `pack_all_detected_hf_mixtral_blocks_independent_failure_does_not_block_other_blocks`)
//! cover multi-block detection + orchestration in isolation, but no
//! CLI gate has pinned the end-to-end behavior:
//!   - The orchestrator's telemetry reports the correct block count
//!     when the WeightMap holds >=2 HF blocks
//!   - The auto-pack runs ONCE per load (not once per block), so the
//!     telemetry count equals the TOTAL blocks in the file
//!   - Plural label formatting is correct (3 blocks → "blocks", not "block")
//!
//! Scope: this gate exercises the AUTO-PACK layer with multiple
//! blocks in the WeightMap. The source has a SINGLE `model`
//! declaration bound to the FIRST block; the other blocks auto-pack
//! into the WeightMap but aren't consumed by the source. That's a
//! valid pattern (auto-pack is WeightMap-level, not model-level —
//! the program can pick which blocks it uses).
//!
//! v2.next deferred:
//!   - Multi-model composition (two `model Block0:` + `model Block1:`
//!     declarations in one source file, each with its own
//!     `@moe(weight_prefix=...)`). When both models are bound to
//!     real blocks, `derive_v4_dims` for the second model fails to
//!     resolve the packed entries despite the auto-pack succeeding.
//!     This is an INDEPENDENT issue with multi-model `@moe`
//!     compilation (no prior CLI gate exercised it). Diagnosed +
//!     deferred to a focused multi-model cycle since the auto-pack
//!     itself is provably correct (telemetry confirms).

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
/// at MULTIPLE prefixes. Each prefix gets a full block (router +
/// weights + biases). Uses canonical `.w{N}.bias` suffix throughout.
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

/// Single-model source bound to `Block0`. The safetensors holds
/// multiple blocks but only Block0 is consumed by the program.
const MOE_SINGLE_MODEL_FIRST_BLOCK_SRC: &str = r#"model Block0:
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

let m = Block0()
let x = ones([2, 8])
let y = m.forward(x)
"#;

#[test]
fn hf_mixtral_two_block_auto_pack_telemetry_plural_correct() {
    // HAPPY: 2 HF blocks in one safetensors. Auto-pack telemetry
    // reports "2 HF Mixtral MoE blocks" (plural) for both v2.7
    // (weights) and v2.15 (biases). The source binds only Block0;
    // Block1's packed entries are unused but auto-packed cleanly.
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_two.nsl");
    fs::write(&src, MOE_SINGLE_MODEL_FIRST_BLOCK_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_multi_block_hf_safetensors(&weights, &["Block0", "Block1"], 8, 16, 4);
    let out = tmp.path().join("moe_two.o");

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
            predicate::str::contains("CPDT v2.7: auto-packed 2 HF Mixtral MoE blocks")
                .and(predicate::str::contains(
                    "CPDT v2.15: auto-packed v4 biases for 2 HF Mixtral MoE blocks",
                ))
                .and(predicate::str::contains("Block0"))
                .and(predicate::str::contains("Block1")),
        );
}

#[test]
fn hf_mixtral_three_block_auto_pack_telemetry_plural_correct() {
    // Three blocks — verify the plural label still fires for
    // arbitrary N>=2 (the orchestrator uses `if len == 1 { "" } else
    // { "s" }`).
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("moe_three.nsl");
    fs::write(&src, MOE_SINGLE_MODEL_FIRST_BLOCK_SRC).unwrap();
    let weights = tmp.path().join("weights.safetensors");
    write_multi_block_hf_safetensors(
        &weights,
        &["Block0", "Block1", "Block2"],
        8, 16, 4,
    );
    let out = tmp.path().join("moe_three.o");

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
            predicate::str::contains("CPDT v2.7: auto-packed 3 HF Mixtral MoE blocks")
                .and(predicate::str::contains(
                    "CPDT v2.15: auto-packed v4 biases for 3 HF Mixtral MoE blocks",
                ))
                .and(predicate::str::contains("Block2")),
        );
}
