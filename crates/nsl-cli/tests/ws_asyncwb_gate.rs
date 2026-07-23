//! Item 11 (writeback half): `--stream-async-writeback` issues each layer
//! pack's post-update DtoH on the transfer stream instead of blocking the
//! host/compute stream — the next range's compute overlaps the writeback,
//! and the mirror scatter is deferred to the runtime's drain points (queue
//! cap / affected re-upload / teardown). The result must be BIT-EXACT with
//! the synchronous arena path: same bytes into the same mirrors, only later.
//!
//! CPU asserts the compile-time diagnostic. The GPU test is the real proof:
//! repeated async runs each match the sync-arena run bit-exactly (a missed
//! drain would surface as stale-mirror training — a deterministic diff; a
//! missed event order as a nondeterministic one), and the counter shows real
//! async writebacks happened.
//!
//! GPU: `cargo test -p nsl-cli --features cuda --test ws_asyncwb_gate -- --ignored`

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
fn async_writeback_diagnostic_reports_active() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_awb_cpu_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, fixture_src("csla_layerwise_ffn.nsl")).unwrap();
    let out = tmp.join("prog.out");

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "build"])
        .args(BASE_FLAGS)
        .arg("--stream-async-writeback")
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
        stderr.contains("async writeback: ACTIVE"),
        "async-writeback diagnostic missing; stderr:\n{stderr}"
    );
}

#[test]
fn async_writeback_requires_stream_arena() {
    let root = repo_root();
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "build", "--stream-async-writeback", "x.nsl"])
        .output()
        .expect("spawn nsl build");
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("stream-arena") || stderr.contains("stream_arena"),
        "flag-dependency error should name --stream-arena; stderr:\n{stderr}"
    );
}

// ── GPU: repeated bit-exact parity + real async writebacks ────────────────

fn run_gpu(fixture: &str, tag: &str, extra: &[&str]) -> (String, Vec<u8>, i64) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_awb_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let ckpt = tmp.join("out.nslm");
    let src = fixture_src(fixture)
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        .replace("CSLA_SAVE_PATH", ckpt.to_str().unwrap());
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, &src).unwrap();

    let mut flags: Vec<&str> = BASE_FLAGS.to_vec();
    flags.extend_from_slice(extra);
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"])
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
    let async_wb = stderr
        .lines()
        .find(|l| l.starts_with("[weight-stream] uploads: "))
        .and_then(|l| l.split("async_wb: ").nth(1))
        .and_then(|r| r.split_whitespace().next())
        .and_then(|s| s.parse().ok())
        .unwrap_or(-1);
    (loss, std::fs::read(&ckpt).unwrap_or_default(), async_wb)
}

fn assert_asyncwb_bit_exact(fixture: &str, tag: &str, extra: &[&str]) {
    // Synchronous arena reference.
    let (sync_loss, sync_ckpt, sync_wb) = run_gpu(fixture, &format!("{tag}_sync"), &[]);
    assert!(!sync_loss.is_empty(), "[{tag}] empty sync loss");
    assert_eq!(sync_wb, 0, "[{tag}] sync path must not async-writeback");

    // Repeat the async runs: a missed drain or event order would show up as
    // a stale-mirror (deterministic) or racy (nondeterministic) diff.
    let mut saw_async = false;
    let mut async_flags = vec!["--stream-async-writeback"];
    async_flags.extend_from_slice(extra);
    for i in 0..4 {
        let (loss, ckpt, wb) = run_gpu(fixture, &format!("{tag}_awb{i}"), &async_flags);
        assert_eq!(loss, sync_loss, "[{tag}] run {i}: loss not bit-exact vs sync");
        assert_eq!(ckpt, sync_ckpt, "[{tag}] run {i}: checkpoint not bit-exact vs sync");
        assert!(wb >= 0, "[{tag}] missing async_wb counter");
        saw_async |= wb > 0;
    }
    assert!(saw_async, "[{tag}] expected real async writebacks to occur");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn asyncwb_bit_exact_ffn_gpu() {
    assert_asyncwb_bit_exact("csla_layerwise_ffn.nsl", "ffn", &[]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn asyncwb_bit_exact_packed_gqa_gpu() {
    assert_asyncwb_bit_exact("csla_layerwise_packed_gqa.nsl", "gqa", &[]);
}

/// The full Item-11 schedule: compute L, prefetch L+1, write back L-1 — all
/// three legs at once, still bit-exact with the fully synchronous arena.
#[test]
#[ignore = "requires CUDA GPU"]
fn asyncwb_plus_prefetch_bit_exact_ffn_gpu() {
    assert_asyncwb_bit_exact("csla_layerwise_ffn.nsl", "ffn_full", &["--stream-prefetch"]);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn asyncwb_plus_prefetch_bit_exact_packed_gqa_gpu() {
    assert_asyncwb_bit_exact(
        "csla_layerwise_packed_gqa.nsl",
        "gqa_full",
        &["--stream-prefetch"],
    );
}
