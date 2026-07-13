//! WGGO-orchestrated PACKED pretraining, end to end (CPU).
//!
//! The campaign's integration moment: a GQA transformer trained on a
//! packed token stream under `--pretrain-optimized` (source AD + wggo
//! greedy + csha auto), where
//!   - the DataLoader builds block-diagonal packed batches at runtime
//!     (packing=true, pack_separator) with real document boundaries,
//!   - the model consumes the additive mask through the stdlib GQA
//!     `forward_masked` (PCA Stage B),
//!   - WGGO pre-plans per-layer decisions and the train block consumes
//!     the plan (fingerprint match),
//!   - the plan's packing decision is segment_id AND is reported as
//!     CONSUMED through the Stage-B source-masked channel (not rejected
//!     with packing_requires_packed_dataset noise),
//!   - and the loss actually decreases.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

#[test]
fn packed_pretrain_trains_and_consumes_the_wggo_plan() {
    let root = workspace_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/wggo_packed_pretrain.nsl");
    assert!(fixture.exists(), "fixture missing: {}", fixture.display());
    let tmp = tempfile::TempDir::new().unwrap();

    let out = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run"])
        .arg(&fixture)
        .arg("--pretrain-optimized")
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "run failed:\n{stderr}");

    // Loss stream: strictly-positive floats between the markers; the run
    // must improve substantially from the first optimizer step.
    let losses: Vec<f64> = stdout
        .lines()
        .skip_while(|l| !l.contains("LOSS_STREAM_BEGIN"))
        .take_while(|l| !l.contains("LOSS_STREAM_END"))
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert!(
        losses.len() >= 8,
        "expected a loss stream, got {} values:\n{stdout}",
        losses.len()
    );
    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first * 0.9,
        "loss did not decrease: first={first} last={last}"
    );
    assert!(last.is_finite() && last > 0.0, "non-finite loss: {last}");

    // Plan consumption + packing verdicts.
    assert!(
        stderr.contains("[wggo] consumed pre-solved plan"),
        "train block did not consume the pre-solved plan:\n{stderr}"
    );
    assert!(
        stderr.contains("wggo-override-consumed packing_mode=segment_id"),
        "plan's segment_id packing decision was not reported consumed:\n{stderr}"
    );
    assert!(
        stderr.contains("masked SDPA (Stage B decomposed path)"),
        "consumption channel should be the Stage-B source-masked path:\n{stderr}"
    );
    assert!(
        !stderr.contains("packing_requires_packed_dataset"),
        "packed module must not produce packing-rejection noise:\n{stderr}"
    );
}
