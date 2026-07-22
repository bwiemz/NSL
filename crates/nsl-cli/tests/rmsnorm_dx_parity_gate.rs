//! Item 9 (correctness): source-AD RMSNorm INPUT gradient must match tape-AD.
//!
//! Source-AD previously computed the RMSNorm dx with the LayerNorm formula
//! (which subtracts the per-row mean) — wrong for RMSNorm, which does not
//! mean-subtract. An upstream weight whose gradient flows through the norm was
//! therefore trained on a wrong direction. This gate trains the same model both
//! ways and asserts the final weights agree to an f32 tolerance (tape-AD is the
//! independent reference; it uses the fused `rmsnorm_backward`).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run(source_ad: bool) -> String {
    run_args(if source_ad { &["--source-ad"] } else { &[] })
}

fn run_args(extra: &[&str]) -> String {
    let root = repo_root();
    let path = root.join("crates/nsl-cli/tests/fixtures/rmsnorm_dx_parity.nsl");
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli", "--", "run"]);
    cmd.args(extra);
    cmd.arg(&path)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    let out = cmd.output().expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed (args={extra:?}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Parse the floats printed between W_BEGIN / W_END from a `tensor([[...]])`.
fn parse_w(stdout: &str) -> Vec<f64> {
    parse_between(stdout, "W_BEGIN", "W_END")
}

/// Same for the trained gamma between G_BEGIN / G_END.
fn parse_g(stdout: &str) -> Vec<f64> {
    parse_between(stdout, "G_BEGIN", "G_END")
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
fn source_ad_rmsnorm_dx_matches_tape_ad() {
    let sa = parse_w(&run(true));
    let tape = parse_w(&run(false));
    assert_eq!(sa.len(), 16, "expected 16 weight values, got {}", sa.len());
    assert_eq!(tape.len(), 16, "tape produced {} values", tape.len());
    // The weight moved off its init (arange*0.05) — dx is a real, nonzero grad.
    assert!(
        (sa[0] - 0.0).abs() > 1e-3,
        "w[0] should have moved from 0.0; got {}",
        sa[0]
    );
    // source-AD (f32) vs tape-AD reference: agree to an f32 training tolerance.
    for (i, (a, b)) in sa.iter().zip(tape.iter()).enumerate() {
        assert!(
            (a - b).abs() < 2e-3,
            "w[{i}] source-AD={a} vs tape-AD={b} (|Δ|={})",
            (a - b).abs()
        );
    }
}

#[test]
fn fused_rmsnorm_dx_matches_decomposition_and_tape_ad() {
    // Item 9 fusion (+ P5 slice A): `--fuse-rmsnorm-backward` lowers BOTH the
    // RMSNorm dx and the gamma gradient to fused ops. On the CPU path they use
    // the same f64 formulas as tape-AD, so the trained w AND g must match
    // tape-AD (and the decomposition) to an f32 tolerance.
    let fused_out = run_args(&["--source-ad", "--fuse-rmsnorm-backward"]);
    let decomp_out = run(true);
    let tape_out = run(false);
    let fused = parse_w(&fused_out);
    let decomp = parse_w(&decomp_out);
    let tape = parse_w(&tape_out);
    assert_eq!(fused.len(), 16, "fused produced {} values", fused.len());
    for (i, ((f, d), t)) in fused.iter().zip(&decomp).zip(&tape).enumerate() {
        assert!(
            (f - t).abs() < 2e-3,
            "w[{i}] fused={f} vs tape-AD={t} (|Δ|={})",
            (f - t).abs()
        );
        assert!(
            (f - d).abs() < 2e-3,
            "w[{i}] fused={f} vs decomposition={d} (|Δ|={})",
            (f - d).abs()
        );
    }
    // P5 slice A: gamma is a trained param whose gradient now flows through
    // the fused dgamma op — it must move off init and agree across paths.
    let fg = parse_g(&fused_out);
    let dg = parse_g(&decomp_out);
    let tg = parse_g(&tape_out);
    assert_eq!(fg.len(), 4, "fused gamma produced {} values", fg.len());
    assert!(
        fg.iter().any(|v| (v - 1.0).abs() > 1e-3),
        "gamma never moved off ones-init (dgamma vacuous): {fg:?}"
    );
    for (i, ((f, d), t)) in fg.iter().zip(&dg).zip(&tg).enumerate() {
        assert!(
            (f - t).abs() < 2e-3,
            "g[{i}] fused={f} vs tape-AD={t} (|Δ|={})",
            (f - t).abs()
        );
        assert!(
            (f - d).abs() < 2e-3,
            "g[{i}] fused={f} vs decomposition={d} (|Δ|={})",
            (f - d).abs()
        );
    }
}

/// GPU: the fused dgamma kernels (per-row 1/rms + per-column row loop) must
/// train w and gamma to the same place as the CPU tape-AD reference, and be
/// bit-deterministic run-to-run.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_rmsnorm_gamma_backward_gpu_matches_reference() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_rmsg_gpu_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/rmsnorm_dx_parity.nsl"),
    )
    .unwrap();
    // Device placement: model + inputs on cuda.
    src = src.replace("let m = M()", "let m = M()
m.to(cuda)");
    src = src.replace(
        "let y = zeros([2, 4])",
        "let y = zeros([2, 4])
let xg = x.to(cuda)
let yg = y.to(cuda)",
    );
    src = src.replace("m.forward(x)", "m.forward(xg)");
    src = src.replace("mse_loss(pred, y)", "mse_loss(pred, yg)");
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let run_gpu = || {
        let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
            .args([
                "run",
                "--source-ad",
                "--deterministic",
                "--fuse-rmsnorm-backward",
            ])
            .arg(&prog)
            .current_dir(&tmp)
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .output()
            .expect("spawn nsl run");
        assert!(
            out.status.success(),
            "GPU fused run failed:
{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8_lossy(&out.stdout).into_owned()
    };
    let out1 = run_gpu();
    let out2 = run_gpu();
    assert_eq!(out1, out2, "GPU fused rmsnorm backward not deterministic");

    let tape = run(false); // CPU tape-AD reference
    let (wg, gg) = (parse_w(&out1), parse_g(&out1));
    let (wt, gt) = (parse_w(&tape), parse_g(&tape));
    assert_eq!(wg.len(), 16);
    assert_eq!(gg.len(), 4);
    assert!(
        gg.iter().any(|v| (v - 1.0).abs() > 1e-3),
        "GPU gamma never moved off ones-init: {gg:?}"
    );
    for (i, (a, b)) in wg.iter().zip(&wt).enumerate() {
        assert!((a - b).abs() < 2e-3, "w[{i}] gpu-fused={a} vs tape={b}");
    }
    for (i, (a, b)) in gg.iter().zip(&gt).enumerate() {
        assert!((a - b).abs() < 2e-3, "g[{i}] gpu-fused={a} vs tape={b}");
    }
}
