//! P5 item 19 gates — opportunistic per-region CUDA graph capture/replay
//! (`--cuda-graphs`).
//!
//! The e2e gates prove the three load-bearing properties:
//!   1. CAPTURE HAPPENS (anti-vacuity): the teardown banner reports
//!      `captured>0` and `replays>0` on a readback-free fixture.
//!   2. BIT-EXACTNESS: the loss stream with graphs on is byte-identical to
//!      the same run with graphs off, and reruns reproduce bit-identically.
//!   3. SELF-HEALING: a fixture whose lowered regions perform per-step host
//!      readbacks (embedding/CE backward bounces) still trains bit-exactly —
//!      the affected regions taint and stay eager instead of corrupting.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    success: bool,
    stdout: String,
    stderr: String,
    losses: Vec<String>,
}

fn run_fixture(fixture: &str, tag: &str, rewrites: &[(&str, &str)], extra: &[&str]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_cudagraph_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src =
        std::fs::read_to_string(root.join("crates/nsl-cli/tests/fixtures").join(fixture)).unwrap();
    for (from, to) in rewrites {
        assert!(src.contains(from), "rewrite marker '{from}' missing in fixture");
        src = src.replace(from, to);
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--deterministic"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let losses = stdout
        .split_once("LOSS_STREAM_BEGIN")
        .and_then(|(_, r)| r.split_once("LOSS_STREAM_END"))
        .map(|(v, _)| {
            v.lines()
                .filter_map(|l| {
                    let l = l.trim();
                    if let Some(inner) =
                        l.strip_prefix("tensor([").and_then(|r| r.strip_suffix("])"))
                    {
                        Some(inner.to_string())
                    } else if l.parse::<f64>().is_ok() {
                        Some(l.to_string())
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();
    RunOut { success: out.status.success(), stdout, stderr, losses }
}

/// Parse `key=value` counters from the `[cuda-graph] regions=… captured=…`
/// teardown banner. Panics (with the stderr) when the banner is absent.
fn banner_counters(stderr: &str) -> std::collections::HashMap<String, u64> {
    let line = stderr
        .lines()
        .filter(|l| l.starts_with("[cuda-graph] regions="))
        .next_back()
        .unwrap_or_else(|| panic!("no [cuda-graph] banner in stderr:\n{stderr}"));
    line.trim_start_matches("[cuda-graph] ")
        .split_whitespace()
        .filter_map(|kv| {
            let (k, v) = kv.split_once('=')?;
            Some((k.to_string(), v.parse().ok()?))
        })
        .collect()
}

const GPU_MLP: (&str, &str) = (
    "# GPU_PLACEMENT",
    "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
);
const XG: (&str, &str) = ("m.forward_train(x)", "m.forward_train(xg)");
const YG: (&str, &str) = ("l1_loss(pred, y)", "l1_loss(pred, yg)");

/// Tape AD has no Wengert lowerings, so there is nothing to bracket — the
/// flag must refuse rather than silently train eager.
#[test]
fn cuda_graphs_refuses_tape_ad() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_cudagraph_tape_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--cuda-graphs"]) // no --source-ad
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(!out.status.success(), "tape-AD x --cuda-graphs ran");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--cuda-graphs requires --source-ad"),
        "wrong refusal:\n{stderr}"
    );
}

/// Eager per-launch sync is incompatible with stream capture — CLI refuses.
#[test]
fn cuda_graphs_refuses_cuda_sync() {
    let out = run_fixture(
        "cuda_graph_gate.nsl",
        "sync",
        &[],
        &["--cuda-graphs", "--cuda-sync"],
    );
    assert!(!out.success, "--cuda-sync x --cuda-graphs ran:\n{}", out.stdout);
    assert!(
        out.stderr
            .contains("--cuda-graphs does not compose with --cuda-sync"),
        "wrong refusal:\n{}",
        out.stderr
    );
}

/// Core e2e: on a readback-free fixture, regions capture and replay, and
/// the loss stream is bit-identical to the eager run and across reruns.
/// Covers both the monolithic (plain) and the CSLA per-layer-range shapes.
#[test]
#[ignore = "requires CUDA GPU"]
fn cuda_graph_capture_replay_bitexact_gpu() {
    for (shape, csla_flags) in [
        ("plain", &[][..]),
        ("csla", &["--checkpoint-blocks", "--layerwise-accum"][..]),
    ] {
        let mut on_args: Vec<&str> = vec!["--seed", "777", "--cuda-graphs"];
        on_args.extend_from_slice(csla_flags);
        let on = run_fixture(
            "cuda_graph_gate.nsl",
            &format!("on_{shape}"),
            &[GPU_MLP, XG, YG],
            &on_args,
        );
        assert!(
            on.success,
            "[{shape}] graphs-on run failed:\nstdout:\n{}\nstderr:\n{}",
            on.stdout, on.stderr
        );
        let c = banner_counters(&on.stderr);
        assert!(
            c["captured"] > 0,
            "[{shape}] no region ever captured (vacuous): {c:?}\n{}",
            on.stderr
        );
        assert!(
            c["replays"] > 0,
            "[{shape}] no captured region ever replayed: {c:?}\n{}",
            on.stderr
        );
        assert_eq!(c["mismatches"], 0, "[{shape}] replay verification broke: {c:?}");
        let lv: Vec<f64> = on.losses.iter().map(|s| s.parse().unwrap()).collect();
        assert!(lv.len() >= 8, "[{shape}] loss stream too short: {lv:?}");
        assert!(lv.iter().all(|v| v.is_finite()), "[{shape}] non-finite loss: {lv:?}");
        assert!(
            *lv.last().unwrap() < *lv.first().unwrap(),
            "[{shape}] did not train: {lv:?}"
        );

        // Bit-exactness vs the eager path: same everything minus the flag.
        let mut off_args: Vec<&str> = vec!["--seed", "777"];
        off_args.extend_from_slice(csla_flags);
        let off = run_fixture(
            "cuda_graph_gate.nsl",
            &format!("off_{shape}"),
            &[GPU_MLP, XG, YG],
            &off_args,
        );
        assert!(off.success, "[{shape}] graphs-off run failed:\n{}", off.stderr);
        assert_eq!(
            on.losses, off.losses,
            "[{shape}] graphs-on loss stream diverged from eager"
        );

        // Determinism: rerun with graphs on reproduces bit-identically.
        let on2 = run_fixture(
            "cuda_graph_gate.nsl",
            &format!("on2_{shape}"),
            &[GPU_MLP, XG, YG],
            &on_args,
        );
        assert!(on2.success, "[{shape}] rerun failed:\n{}", on2.stderr);
        assert_eq!(on.losses, on2.losses, "[{shape}] rerun not bit-identical");
    }
}

/// Self-healing e2e: the embedding+DataLoader fixture performs real host
/// readbacks inside its lowered regions every step (embedding/CE backward
/// device bounces). Those regions must taint and stay eager — and the run
/// must remain bit-identical to the eager path.
#[test]
#[ignore = "requires CUDA GPU"]
fn cuda_graph_self_heals_on_host_readbacks_gpu() {
    let gpu = ("# GPU_PLACEMENT", "m.to(cuda)");
    let base: &[&str] = &["--checkpoint-blocks", "--layerwise-accum", "--seed", "777"];

    let mut on_args: Vec<&str> = base.to_vec();
    on_args.push("--cuda-graphs");
    let on = run_fixture(
        "csla_layerwise_ffn.nsl",
        "heal_on",
        &[gpu, ("CSLA_SAVE_PATH", "h1.nslm")],
        &on_args,
    );
    assert!(
        on.success,
        "graphs-on run failed:\nstdout:\n{}\nstderr:\n{}",
        on.stdout, on.stderr
    );
    let c = banner_counters(&on.stderr);
    assert!(
        c["taints"] > 0,
        "expected host-readback regions to taint (fixture changed?): {c:?}"
    );

    let off = run_fixture(
        "csla_layerwise_ffn.nsl",
        "heal_off",
        &[gpu, ("CSLA_SAVE_PATH", "h2.nslm")],
        base,
    );
    assert!(off.success, "graphs-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "self-healed run diverged from the eager path"
    );
}
