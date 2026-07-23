//! Item 10: `--stream-arena` batches a layer's per-param host<->device
//! transfers into ONE contiguous transfer through a stable device staging
//! slot. The transform must be BIT-EXACT with the per-param path (same mirror
//! bytes, same order) — only the number/shape of CUDA transfers changes.
//!
//! CPU: asserts the compile-time coarsening (one pack per streamed layer).
//! GPU: the real proof — arena vs per-param produce an identical loss stream
//! and a byte-identical checkpoint, while the counters show the batching win
//! (pack_uploads << uploads) and stable device addresses (ptr_moves == 0).
//!
//! GPU: `cargo test -p nsl-cli --features cuda --test ws_arena_gate -- --ignored`

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

fn build_only(source: &str, tag: &str, arena: bool) -> (bool, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_arena_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, source).unwrap();
    let out = tmp.join("prog.out");
    let mut args = vec![
        "run",
        "-q",
        "--manifest-path",
    ];
    let manifest = root.join("Cargo.toml");
    args.push(manifest.to_str().unwrap());
    args.extend([
        "-p",
        "nsl-cli",
        "--",
        "build",
        "--source-ad",
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
    ]);
    if arena {
        args.push("--stream-arena");
    }
    let output = Command::new(env!("CARGO"))
        .args(&args)
        .arg(&prog)
        .arg("-o")
        .arg(&out)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl build");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn arena_mode_coarsens_to_one_pack_per_layer() {
    // The 2-block FFN has two streamed layer groups (3 params each), so arena
    // mode must report exactly two contiguous packs.
    let (ok, stderr) = build_only(&fixture_src("csla_layerwise_ffn.nsl"), "ffn", true);
    assert!(ok, "compile failed:\n{stderr}");
    assert!(
        stderr.contains("arena mode: 2 contiguous layer packs (sizes [3,3])"),
        "expected two 3-param packs; stderr:\n{stderr}"
    );
}

#[test]
fn per_param_mode_reports_no_arena_packs() {
    // Without --stream-arena the arena diagnostic must not appear (the
    // per-param path is unchanged).
    let (ok, stderr) = build_only(&fixture_src("csla_layerwise_ffn.nsl"), "ffn_pp", false);
    assert!(ok, "compile failed:\n{stderr}");
    assert!(
        !stderr.contains("arena mode:"),
        "arena diagnostic leaked into per-param mode; stderr:\n{stderr}"
    );
}

// ── GPU: bit-exact parity + batching evidence ──────────────────────────────

struct GpuRun {
    loss: String,
    ckpt: Vec<u8>,
    uploads: i64,
    pack_uploads: i64,
    ptr_moves: i64,
}

fn run_gpu(fixture: &str, tag: &str, arena: bool) -> GpuRun {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_arena_gpu_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let ckpt = tmp.join("out.nslm");
    let src = fixture_src(fixture)
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        .replace("CSLA_SAVE_PATH", ckpt.to_str().unwrap());
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, &src).unwrap();

    let mut extra: Vec<&str> = vec![
        "--source-ad",
        "--deterministic",
        "--checkpoint-blocks",
        "--layerwise-accum",
        "--weight-stream",
    ];
    if arena {
        extra.push("--stream-arena");
    }
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"])
        .args(&extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_WS_COUNTER", "1")
        .env("NSL_EMBEDDING_BWD_CPU", "1")
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "[{tag}] run failed:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let loss = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| v.trim().to_string())
        .unwrap_or_default();
    let field = |label: &str| -> i64 {
        stderr
            .lines()
            .find(|l| l.starts_with("[weight-stream] uploads: "))
            .and_then(|l| l.split(label).nth(1))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(-1)
    };
    GpuRun {
        loss,
        ckpt: std::fs::read(&ckpt).unwrap_or_default(),
        uploads: field("uploads: "),
        pack_uploads: field("pack_uploads: "),
        ptr_moves: field("ptr_moves: "),
    }
}

fn assert_parity(fixture: &str, tag: &str) {
    let pp = run_gpu(fixture, &format!("{tag}_pp"), false);
    let ar = run_gpu(fixture, &format!("{tag}_ar"), true);

    assert!(!pp.loss.is_empty(), "[{tag}] empty per-param loss stream");
    assert_eq!(pp.loss, ar.loss, "[{tag}] loss stream differs (not bit-exact)");
    assert!(!pp.ckpt.is_empty() && &pp.ckpt[0..4] == b"NSLM", "[{tag}] bad checkpoint");
    assert_eq!(pp.ckpt, ar.ckpt, "[{tag}] checkpoint bytes differ (not bit-exact)");

    // Batching evidence: same logical uploads, delivered in far fewer packs.
    assert_eq!(ar.uploads, pp.uploads, "[{tag}] logical upload count changed");
    assert!(
        ar.pack_uploads > 0 && ar.pack_uploads < ar.uploads,
        "[{tag}] arena must batch: pack_uploads={} uploads={}",
        ar.pack_uploads,
        ar.uploads
    );
    assert_eq!(pp.pack_uploads, 0, "[{tag}] per-param path must not pack");
    // Stable arena addresses: no param's device pointer moved across steps.
    assert_eq!(ar.ptr_moves, 0, "[{tag}] arena addresses must be stable");
    assert!(pp.ptr_moves > 0, "[{tag}] per-param path reallocates each upload");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn arena_bit_exact_ffn_gpu() {
    assert_parity("csla_layerwise_ffn.nsl", "ffn");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn arena_bit_exact_packed_gqa_gpu() {
    assert_parity("csla_layerwise_packed_gqa.nsl", "gqa");
}
