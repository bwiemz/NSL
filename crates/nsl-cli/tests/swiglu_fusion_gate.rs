//! P5 item 20 slice B gates — SwiGLU gate-backward fusion (always-on,
//! bit-exact peephole: Mul(dy, up) + silu_backward → swiglu_gate_backward).
//!
//! Trains a real SwiGLU MLP under source-AD (fused path) and tape-AD (the
//! independent reference, no peephole) and asserts the trained gate AND up
//! weights agree. GPU variant additionally asserts bit-determinism.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_fixture(tag: &str, gpu: bool, extra: &[&str]) -> String {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_swiglu_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/swiglu_parity.nsl"),
    )
    .unwrap();
    if gpu {
        src = src.replace(
            "# GPU_PLACEMENT",
            "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
        );
        src = src.replace("m.forward(x)", "m.forward(xg)");
        src = src.replace("l1_loss(pred, y)", "l1_loss(pred, yg)");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic"])
        .args(extra)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed (tag={tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn parse_between(stdout: &str, begin: &str, end: &str) -> Vec<f64> {
    let after = stdout.split_once(begin).map(|(_, r)| r).unwrap_or("");
    let inner = after.split_once(end).map(|(l, _)| l).unwrap_or("");
    inner
        .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
        .filter_map(|t| t.parse::<f64>().ok())
        .collect()
}

#[test]
fn swiglu_fused_backward_matches_tape_ad_cpu() {
    let fused = run_fixture("cpu_sa", false, &["--source-ad"]);
    let tape = run_fixture("cpu_tape", false, &[]);
    // The fixture's inits, to measure movement against: a weight that never
    // left its init would "match" trivially.
    let inits: [(&str, &str, &str, fn(usize) -> f64); 2] = [
        ("w_gate", "WG_BEGIN", "WG_END", |i| i as f64 * 0.02),
        ("w_up", "WU_BEGIN", "WU_END", |i| i as f64 * 0.01 + 0.05),
    ];
    for (name, b, e, init) in inits {
        let f = parse_between(&fused, b, e);
        let t = parse_between(&tape, b, e);
        assert_eq!(f.len(), 32, "{name}: fused produced {} values", f.len());
        assert_eq!(t.len(), 32, "{name}: tape produced {} values", t.len());
        let moved = t.iter().enumerate().map(|(i, v)| (v - init(i)).abs()).fold(0.0, f64::max);
        assert!(moved > 1e-2, "{name} barely moved off its init ({moved:.2e}): {t:?}");
        // Both paths run the same f64 formulas on the CPU; they differ by f32
        // rounding (~1e-8). Under SGD a wrong silu' moves a weight by a
        // fraction of its displacement, far past 1e-5.
        for (i, (a, b)) in f.iter().zip(&t).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "{name}[{i}] fused={a} vs tape={b} (|Δ|={})",
                (a - b).abs()
            );
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn swiglu_fused_backward_gpu_deterministic_and_matches_reference() {
    let a = run_fixture("gpu_a", true, &["--source-ad"]);
    let b = run_fixture("gpu_b", true, &["--source-ad"]);
    assert_eq!(a, b, "GPU fused SwiGLU backward not deterministic");

    let tape = run_fixture("cpu_ref", false, &[]);
    for (name, bg, en) in [("w_gate", "WG_BEGIN", "WG_END"), ("w_up", "WU_BEGIN", "WU_END")] {
        let f = parse_between(&a, bg, en);
        let t = parse_between(&tape, bg, en);
        assert_eq!(f.len(), 32, "{name}: GPU produced {} values", f.len());
        for (i, (x, y)) in f.iter().zip(&t).enumerate() {
            assert!(
                (x - y).abs() < 2e-3,
                "{name}[{i}] gpu-fused={x} vs cpu-tape={y} (|Δ|={})",
                (x - y).abs()
            );
        }
    }
}

/// The peephole must actually fire on the fixture, and must be bit-exact:
/// the fused source-AD run prints the same weights, to the last digit, as a
/// source-AD run with `NSL_FUSE_SWIGLU_GATE=0`. (It used to never fire —
/// `MulElementwise` wraps the product in `reduce_to_shape`, which the
/// peephole did not see through — so the gates above compared the unfused
/// path with itself.)
#[test]
fn swiglu_peephole_fires_and_is_bit_exact_cpu() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_swiglu_fires_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    std::fs::copy(root.join("crates/nsl-cli/tests/fixtures/swiglu_parity.nsl"), tmp.join("prog.nsl")).unwrap();
    let run = |fuse: bool| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
        cmd.args(["run", "--deterministic", "--source-ad", "prog.nsl"])
            .current_dir(&tmp)
            .env("NSL_STDLIB_PATH", root.join("stdlib"));
        if !fuse {
            cmd.env("NSL_FUSE_SWIGLU_GATE", "0");
        }
        let out = cmd.output().expect("spawn nsl run");
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "run failed (fuse={fuse}):\n{stderr}");
        (String::from_utf8_lossy(&out.stdout).into_owned(), stderr)
    };
    let (fused_out, fused_err) = run(true);
    let (plain_out, plain_err) = run(false);
    assert!(
        fused_err.contains("[fuse] swiglu gate-backward pairs: 1"),
        "the SwiGLU gate peephole did not fire on the fixture:\n{fused_err}"
    );
    assert!(
        !plain_err.contains("[fuse] swiglu gate-backward"),
        "NSL_FUSE_SWIGLU_GATE=0 did not disable the peephole:\n{plain_err}"
    );
    for (b, e) in [("WG_BEGIN", "WG_END"), ("WU_BEGIN", "WU_END")] {
        let section = |s: &str| s.split_once(b).and_then(|(_, r)| r.split_once(e)).map(|(v, _)| v.to_string());
        let (f, p) = (section(&fused_out), section(&plain_out));
        assert!(f.is_some(), "{b} missing from fused output:\n{fused_out}");
        assert_eq!(f, p, "{b}: fused and unfused weights differ");
    }
}
