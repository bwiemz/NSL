//! Item 8: periodic-checkpointing gate (`--checkpoint-stride`).
//!
//! The stride coalesces CCR block anchors into super-segments — saving only
//! every k-th block boundary and recomputing the span. Two guarantees:
//!   1. BIT-EXACT across strides on CPU (recompute is deterministic), with and
//!      without CSLA (`--layerwise-accum`).
//!   2. On GPU, the Activations surface peak drops at the sweet-spot stride
//!      (the saved-boundary term, buffered per accumulation micro-batch, is the
//!      dominant surface). GPU test is `#[ignore]`.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct Run {
    ok: bool,
    stdout: String,
    stderr: String,
}

/// Run the fixture (optionally rewriting `# GPU_PLACEMENT`) with the given
/// extra args. `cuda` builds/runs the cuda feature.
fn run(rewrite_gpu: bool, cuda: bool, extra: &[&str]) -> Run {
    let root = repo_root();
    let src_path = root.join("crates/nsl-cli/tests/fixtures/ccr_periodic_stride.nsl");
    let mut src = std::fs::read_to_string(&src_path).expect("read fixture");
    if rewrite_gpu {
        assert!(src.contains("# GPU_PLACEMENT"));
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    let tmp = root.join(format!(
        "target/ccr_periodic_{}.nsl",
        if cuda { "gpu" } else { "cpu" }
    ));
    std::fs::write(&tmp, &src).expect("write temp fixture");

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q"]);
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.args(["-p", "nsl-cli", "--", "run", "--source-ad"]);
    cmd.args(extra);
    cmd.arg(&tmp)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    let out = cmd.output().expect("spawn nsl run");
    Run {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// The loss lines between the LOSS_STREAM markers, joined — the parity key.
fn loss_stream(stdout: &str) -> String {
    let after = stdout.split_once("LOSS_STREAM_BEGIN").map(|(_, r)| r).unwrap_or("");
    let inner = after.split_once("LOSS_STREAM_END").map(|(l, _)| l).unwrap_or("");
    inner
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn act_peak(stdout: &str) -> Option<u64> {
    let after = stdout.split_once("ACT_PEAK_BEGIN").map(|(_, r)| r)?;
    let inner = after.split_once("ACT_PEAK_END").map(|(l, _)| l)?;
    inner.lines().map(str::trim).find_map(|l| l.parse::<u64>().ok())
}

#[test]
fn periodic_stride_is_bit_exact_across_strides_cpu() {
    // CSLA active (the regime the stride targets): stride 1/2/4 must agree.
    let base = ["--checkpoint-blocks", "--layerwise-accum"];
    let mut variants = Vec::new();
    for k in ["1", "2", "4"] {
        let mut args = base.to_vec();
        args.push("--checkpoint-stride");
        args.push(k);
        let r = run(false, false, &args);
        assert!(r.ok, "stride {k} run failed:\n{}", r.stderr);
        variants.push((k, loss_stream(&r.stdout)));
    }
    let (_, ref reference) = variants[0];
    assert!(!reference.is_empty(), "no loss stream produced");
    for (k, ls) in &variants[1..] {
        assert_eq!(
            ls, reference,
            "stride {k} diverged from stride 1 — recompute must be bit-exact"
        );
    }
}

#[test]
fn periodic_stride_emits_coalescing_note() {
    let r = run(
        false,
        false,
        &["--checkpoint-blocks", "--checkpoint-stride", "2"],
    );
    assert!(r.ok, "run failed:\n{}", r.stderr);
    assert!(
        r.stderr.contains("periodic checkpointing: stride 2"),
        "expected the coalescing note on stderr:\n{}",
        r.stderr
    );
}

#[test]
fn auto_stride_reports_a_decision() {
    let r = run(
        false,
        false,
        &[
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--checkpoint-stride",
            "auto",
            "--checkpoint-budget-mib",
            "1",
        ],
    );
    assert!(r.ok, "auto run failed:\n{}", r.stderr);
    assert!(
        r.stderr.contains("--checkpoint-stride auto: chose stride"),
        "expected an auto decision line:\n{}",
        r.stderr
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn periodic_stride_reduces_activation_peak_gpu() {
    let base = ["--checkpoint-blocks", "--layerwise-accum"];
    let peak_at = |k: &str| -> u64 {
        let mut args = base.to_vec();
        args.push("--checkpoint-stride");
        args.push(k);
        let r = run(true, true, &args);
        assert!(r.ok, "gpu stride {k} failed:\n{}", r.stderr);
        act_peak(&r.stdout).unwrap_or_else(|| panic!("no ACT_PEAK for stride {k}:\n{}", r.stdout))
    };
    let (p1, p2) = (peak_at("1"), peak_at("2"));
    // Stride 2 is the sweet spot on this 8-block / G=4 model: fewer saved
    // boundaries (× the accumulation window) outweigh the larger recompute span.
    assert!(
        p2 < p1,
        "stride 2 must lower the activation peak: stride2={p2} vs stride1={p1}"
    );
}
