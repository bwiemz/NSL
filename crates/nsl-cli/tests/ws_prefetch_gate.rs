//! Item 11: `--stream-prefetch` double-buffers the backward weight stream —
//! each layer's pack is prefetched (async HtoD on the transfer stream) while
//! the previous layer computes, and the compute stream waits on a per-pack
//! CUDA event before reading it (the CADENCE assume/guarantee certificate).
//! WGGO's calibration activates the overlap only where compute hides the
//! transfer. The result must be BIT-EXACT with the synchronous arena path.
//!
//! CPU asserts the calibration decision + the discharged certificate. The GPU
//! test is the real proof: repeated prefetch runs each match the sync-arena
//! run bit-exactly (a race would surface as a nondeterministic diff), and the
//! counter shows real async prefetches happened.
//!
//! GPU: `cargo test -p nsl-cli --features cuda --test ws_prefetch_gate -- --ignored`

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_src(name: &str) -> String {
    let p = repo_root().join("crates/nsl-cli/tests/fixtures").join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

const BASE_FLAGS: &[&str] = &[
    "--source-ad",
    "--deterministic",
    "--checkpoint-blocks",
    "--layerwise-accum",
    "--weight-stream",
    "--stream-arena",
];

#[test]
fn calibration_activates_and_certificate_discharges() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_pf_cpu_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, fixture_src("csla_layerwise_ffn.nsl")).unwrap();
    let out = tmp.join("prog.out");

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "build"])
        .args(BASE_FLAGS)
        .arg("--stream-prefetch")
        .arg(&prog)
        .arg("-o")
        .arg(&out)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl build");
    assert!(output.status.success(), "compile failed");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("prefetch double-buffer: ACTIVE"),
        "calibration should activate for the FFN (avg_ops/range >> threshold); stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("transfer certificate: 2 prefetch obligations discharged"),
        "expected two discharged prefetch obligations; stderr:\n{stderr}"
    );
}

// ── GPU: repeated bit-exact parity + real async prefetch ──────────────────

fn run_gpu(fixture: &str, tag: &str, prefetch: bool) -> (String, Vec<u8>, i64) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_pf_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let ckpt = tmp.join("out.nslm");
    let src = fixture_src(fixture)
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        .replace("CSLA_SAVE_PATH", ckpt.to_str().unwrap());
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, &src).unwrap();

    let mut flags: Vec<&str> = BASE_FLAGS.to_vec();
    if prefetch {
        flags.push("--stream-prefetch");
    }
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run"])
        .args(&flags)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_WS_COUNTER", "1")
        .env("NSL_EMBEDDING_BWD_CPU", "1")
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "[{tag}] run failed:\n{stderr}");
    let loss = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default();
    let prefetches = stderr
        .lines()
        .find(|l| l.starts_with("[weight-stream] uploads: "))
        .and_then(|l| l.split("prefetches: ").nth(1))
        .and_then(|r| r.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(-1);
    (loss, std::fs::read(&ckpt).unwrap_or_default(), prefetches)
}

fn assert_prefetch_bit_exact(fixture: &str, tag: &str) {
    // Synchronous arena reference.
    let (sync_loss, sync_ckpt, sync_pf) = run_gpu(fixture, &format!("{tag}_sync"), false);
    assert!(!sync_loss.is_empty(), "[{tag}] empty sync loss");
    assert_eq!(sync_pf, 0, "[{tag}] sync path must not prefetch");

    // Repeat the async prefetch run several times: a missing event ordering
    // would show up as a nondeterministic diff across repeats.
    let mut saw_prefetch = false;
    for i in 0..6 {
        let (loss, ckpt, pf) = run_gpu(fixture, &format!("{tag}_pf{i}"), true);
        assert_eq!(loss, sync_loss, "[{tag}] run {i}: loss not bit-exact vs sync (race?)");
        assert_eq!(ckpt, sync_ckpt, "[{tag}] run {i}: checkpoint not bit-exact vs sync");
        assert!(pf >= 0, "[{tag}] missing prefetch counter");
        saw_prefetch |= pf > 0;
    }
    assert!(saw_prefetch, "[{tag}] expected real async prefetches to occur");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn prefetch_bit_exact_ffn_gpu() {
    assert_prefetch_bit_exact("csla_layerwise_ffn.nsl", "ffn");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn prefetch_bit_exact_packed_gqa_gpu() {
    assert_prefetch_bit_exact("csla_layerwise_packed_gqa.nsl", "gqa");
}
