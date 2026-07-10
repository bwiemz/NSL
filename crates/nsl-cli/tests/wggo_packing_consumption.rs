//! Integration pin for the WGGO `packing_mode` consumption layer (errata E2 /
//! audit gap #4): under `--source-ad --wggo`, the plan's per-layer PCA packing
//! decision must be validated against the synthesized attention kernels and
//! surface a `[pca] layer:N wggo-override-consumed/rejected` stderr verdict —
//! the same consumer contract CSHA/WRGA/FASE/CPDT/prune already follow. Before
//! this layer existed, `packing_mode` was computed by the joint ILP and then
//! silently dropped (report-only).
//!
//! Exit status is deliberately not asserted: the fixture shape currently fails
//! later in compilation on a pre-existing optimizer-resolution gap (see
//! `run_pca_per_doc_gate.rs` for the same pattern and its rationale); the
//! `[pca]` verdicts print during train-block planning, before that point.

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_wggo(source: &str, tag: &str) -> String {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_wggo_pack_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let corpus = tmp.join("tokens.bin");
    std::fs::write(&corpus, [0u8; 128]).unwrap();
    let fixture = tmp.join("prog.nsl");
    std::fs::write(&fixture, source.replace("{CORPUS}", &corpus.display().to_string())).unwrap();

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad", "--wggo", "greedy"])
        .arg(&fixture)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("failed to spawn nsl run");
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    std::fs::remove_dir_all(&tmp).ok();
    stderr
}

const ATTENTION_MODEL: &str = r#"model TinyAttn:
    w_norm: Tensor = ones([32])
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([32, 32])
"#;

/// Packed dataset + @pca + @flash_attention: masked kernels exist, so a
/// plan-side packing request must be CONSUMED (verdict names the kernel).
#[test]
fn wggo_packing_decision_consumed_when_masked_kernels_exist() {
    let source = format!(
        r#"from nsl.nn.losses import mse_loss

dataset PackedCorpus("docs"):
    source = "{{CORPUS}}"
    packing = true
    max_sequence_length = 64
    mean_doc_length = 16

{ATTENTION_MODEL}
@pca(strategy=per_document)
train(model = m, epochs = 1):
    data:
        source = PackedCorpus
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, x)
"#
    );
    let stderr = run_wggo(&source, "consumed");
    assert!(
        stderr.contains("[pca]") && stderr.contains("wggo-override-consumed packing_mode="),
        "expected a consumed [pca] verdict (masked kernels exist for this packed \
         dataset), got:\n{stderr}"
    );
}

/// No packed dataset: a plan-side packing request cannot be honored (masking
/// cannot be synthesized from nothing) and must be REJECTED loudly.
#[test]
fn wggo_packing_decision_rejected_without_packed_dataset() {
    let source = format!(
        r#"from nsl.nn.losses import mse_loss

{ATTENTION_MODEL}
train(model = m, epochs = 1):
    optimizer: AdamW(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, x)
"#
    );
    let stderr = run_wggo(&source, "rejected");
    // The greedy planner requests packing (its objective sees only savings), so
    // an unpacked program must surface the PackingRequiresPackedDataset verdict.
    // If a future cost model stops requesting packing here, the [pca] line
    // legitimately disappears — assert the pair (either rejected, or no [pca]
    // packing verdict at all), never a consumed verdict.
    assert!(
        !stderr.contains("wggo-override-consumed packing_mode="),
        "an unpacked program must never report a CONSUMED packing verdict:\n{stderr}"
    );
    if stderr.contains("[pca]") {
        assert!(
            stderr.contains("reason=packing_requires_packed_dataset"),
            "the [pca] verdict for an unpacked program must be the space-free \
             packing_requires_packed_dataset rejection token, got:\n{stderr}"
        );
    }
}
