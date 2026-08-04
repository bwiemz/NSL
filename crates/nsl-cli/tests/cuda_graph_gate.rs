//! P5 item 19 gates — opportunistic per-region CUDA graph capture/replay
//! (`--cuda-graphs`).
//!
//! The e2e gates prove the three load-bearing properties:
//!   1. CAPTURE HAPPENS (anti-vacuity): the teardown banner reports
//!      `captured>0` and `replays>0` on a readback-free fixture.
//!   2. BIT-EXACTNESS: the loss stream with graphs on is byte-identical to
//!      the same run with graphs off, and reruns reproduce bit-identically.
//!   3. SELF-HEALING: a region that performs a per-step host readback still
//!      trains bit-exactly — it taints and stays eager instead of corrupting.
//!      The readback is forced with a kill switch: the default path no longer
//!      has one, because the fixture's three original bounces were each
//!      device-driven away as perf work.

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
    run_fixture_env(fixture, tag, rewrites, extra, &[])
}

/// `run_fixture` plus environment overrides for the child `nsl run`.
///
/// Needed by the self-heal gate, which has to FORCE a host readback back into a
/// lowered region: the fixture's three original in-region bounces were all
/// device-driven away by later perf work (see that test's comment).
fn run_fixture_env(
    fixture: &str,
    tag: &str,
    rewrites: &[(&str, &str)],
    extra: &[&str],
    envs: &[(&str, &str)],
) -> RunOut {
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
        .envs(envs.iter().copied())
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

/// Capture is opportunistic and needs a WARM-UP: a region records, waits for
/// STABLE_STREAK identical digests, and only then captures. On this fixture
/// the sgemm output pointer moves between the first and second window, so the
/// streak restarts and capture completes on window 4 of 4 — with no fifth
/// occurrence, `replays` is necessarily 0 and this gate could never pass.
/// That is a fixture that is too short, not a broken feature: the shared
/// `epochs=8` belongs to three other gates, so lengthen it HERE only.
/// 20 epochs / grad_accumulation=2 = 10 windows, leaving ~6 replays per region.
const EPOCHS: (&str, &str) = ("epochs=8", "epochs=20");

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
            &[GPU_MLP, XG, YG, EPOCHS],
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
            &[GPU_MLP, XG, YG, EPOCHS],
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
            &[GPU_MLP, XG, YG, EPOCHS],
            &on_args,
        );
        assert!(on2.success, "[{shape}] rerun failed:\n{}", on2.stderr);
        assert_eq!(on.losses, on2.losses, "[{shape}] rerun not bit-identical");
    }
}

/// Perf-flagship numerics: under bf16-storage matmul mode every
/// high-intensity GEMM casts its operands to bf16 scratch through two hooked
/// kernel launches. Until 2026-08 that path TAINTED every region it touched —
/// the casts were recorded BEHIND the gemm pseudo-op they feed, and a
/// replay's early-return at the gemm hook could never re-issue them — so the
/// one matmul mode real pretrains run could never capture. This gate pins the
/// repaired ordering (casts first, then a gemm op carrying
/// `GemmPrecision::Bf16Storage`): regions capture, replay, and stay
/// bit-identical to the eager run under forced bf16 storage. A reintroduced
/// taint fails the `captured > 0` assert directly: with the cast path forced
/// onto every gemm, every capturable region on this fixture contains one.
///
/// `NSL_MATMUL_BF16_MIN_RATIO=1` forces the cast path onto this fixture's
/// small GEMMs — the default 512 intensity floor would decline them all and
/// this gate would silently re-test the plain-f32 path the core gate already
/// covers. The loss-stream INEQUALITY against a default-mode control is the
/// witness that bf16 actually engaged; without it a typo'd env var would
/// leave everything green while testing nothing new.
#[test]
#[ignore = "requires CUDA GPU"]
fn cuda_graph_bf16_storage_captures_gpu() {
    // NSL_CUDA_GRAPH_LOG so a failure's stderr names the taint/divergence
    // that broke capture instead of just showing captured=0.
    let bf16: &[(&str, &str)] = &[
        ("NSL_MATMUL_BF16", "1"),
        ("NSL_MATMUL_BF16_MIN_RATIO", "1"),
        ("NSL_CUDA_GRAPH_LOG", "1"),
    ];
    let on = run_fixture_env(
        "cuda_graph_gate.nsl",
        "bf16on",
        &[GPU_MLP, XG, YG, EPOCHS],
        &["--seed", "777", "--cuda-graphs"],
        bf16,
    );
    assert!(
        on.success,
        "bf16 graphs-on run failed:\nstdout:\n{}\nstderr:\n{}",
        on.stdout, on.stderr
    );
    let c = banner_counters(&on.stderr);
    assert!(
        c["captured"] > 0,
        "no region captured under bf16-storage mode — the cast-before-hook \
         ordering regressed (or a gemm taint is back): {c:?}\n{}",
        on.stderr
    );
    assert!(
        c["replays"] > 0,
        "no captured region ever replayed under bf16-storage mode: {c:?}\n{}",
        on.stderr
    );
    assert_eq!(c["mismatches"], 0, "replay verification broke under bf16: {c:?}");

    // Bit-exactness vs the eager path under the same mode.
    let off = run_fixture_env(
        "cuda_graph_gate.nsl",
        "bf16off",
        &[GPU_MLP, XG, YG, EPOCHS],
        &["--seed", "777"],
        bf16,
    );
    assert!(off.success, "bf16 graphs-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "bf16 graphs-on loss stream diverged from the eager bf16 run"
    );

    // Mode-engagement witness: bf16-storage products round differently from
    // the default mode, so the streams must NOT match.
    let control = run_fixture(
        "cuda_graph_gate.nsl",
        "bf16ctl",
        &[GPU_MLP, XG, YG, EPOCHS],
        &["--seed", "777", "--cuda-graphs"],
    );
    assert!(control.success, "control run failed:\n{}", control.stderr);
    assert_ne!(
        on.losses, control.losses,
        "bf16-storage mode did not engage (loss stream identical to the \
         default mode) — env plumbing dead, gate is vacuous"
    );
}

/// Self-healing e2e: a lowered region that performs a host readback must taint
/// and stay eager instead of capturing — and the run must stay bit-identical to
/// the fully eager path.
///
/// The readback is FORCED, via `NSL_GPU_CE_BACKWARD=0`, and that is the point of
/// this comment. This fixture used to bounce to the host three times per
/// micro-batch all by itself — the CE-composite gather, `nsl_cross_entropy_backward`
/// (which declined because the DataLoader emits host-resident i32 labels), and
/// `nsl_embedding_backward` (same shape of decline). `ea5ee585` and `4326ae7f`
/// device-drove all three; removing them was the explicit GOAL of that work,
/// since they were what kept loss-epilogue regions eager.
///
/// So by 2026-07-30 this gate was asserting `taints > 0` on a property the
/// fixture no longer had, and it sat red on main — invisibly, because the same
/// PR that removed the last bounce also broke the workspace test build that the
/// GPU certification lane runs before any gate. Restoring the property by
/// construction keeps the taint -> stay-eager -> bit-exact path covered; the
/// alternative (asserting `taints == 0`) would have retired the only e2e
/// coverage of a mechanism that exists to prevent silent corruption.
///
/// Both arms get the same env so the bit-exactness comparison stays meaningful.
#[test]
#[ignore = "requires CUDA GPU"]
fn cuda_graph_self_heals_on_host_readbacks_gpu() {
    let gpu = ("# GPU_PLACEMENT", "m.to(cuda)");
    let base: &[&str] = &["--checkpoint-blocks", "--layerwise-accum", "--seed", "777"];

    // Documented kill switch (crates/nsl-runtime/src/tensor/ad_ops.rs): routes the
    // cross-entropy backward through the host path, which DtoHs the [N, C] logits
    // inside the loss-epilogue region.
    let force_readback: &[(&str, &str)] = &[("NSL_GPU_CE_BACKWARD", "0")];

    let mut on_args: Vec<&str> = base.to_vec();
    on_args.push("--cuda-graphs");
    let on = run_fixture_env(
        "csla_layerwise_ffn.nsl",
        "heal_on",
        &[gpu, ("CSLA_SAVE_PATH", "h1.nslm")],
        &on_args,
        force_readback,
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

    let off = run_fixture_env(
        "csla_layerwise_ffn.nsl",
        "heal_off",
        &[gpu, ("CSLA_SAVE_PATH", "h2.nslm")],
        base,
        force_readback,
    );
    assert!(off.success, "graphs-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "self-healed run diverged from the eager path"
    );
}
