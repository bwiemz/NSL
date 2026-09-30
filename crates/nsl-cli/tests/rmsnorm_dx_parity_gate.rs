//! Item 9 (correctness): source-AD RMSNorm INPUT gradient must match tape-AD.
//!
//! Source-AD previously computed the RMSNorm dx with the LayerNorm formula
//! (which subtracts the per-row mean) — wrong for RMSNorm, which does not
//! mean-subtract. An upstream weight whose gradient flows through the norm was
//! therefore trained on a wrong direction. This gate trains the same model both
//! ways and asserts the final weights agree to an f32 tolerance.
//!
//! Audit slice 6: every lowering is also held to [`reference`], the fixture's
//! training run in f64 with gradients taken by central differences of the
//! forward. The lowerings used to be compared only with each other, under
//! AdamW (which steps by `lr * sign(grad)`), at 2e-3: a dx or dgamma off by
//! any positive factor, or wrong the same way in every lowering, passed.

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
    cmd.args(["run", "-q", "-p", "nsl-cli", "--features", if cfg!(feature = "cuda") { "cuda" } else { "" }, "--", "run"]);
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

/// The fixture's SGD run in f64: `(w, g)` after its six steps. Gradients are
/// central differences of the forward (`h = x @ w`, RMSNorm with gain `g` and
/// eps 1e-5, `mean((pred - y)^2)`), so no backward formula is shared with the
/// code under test.
fn reference() -> (Vec<f64>, Vec<f64>) {
    const LR: f64 = 0.1;
    const STEPS: usize = 6;
    let x = |r: usize, c: usize| (4 * r + c) as f64 * 0.1 + 0.1;
    let y = |r: usize, c: usize| (4 * r + c) as f64 * 0.3 - 1.0;
    // params[0..16] = w (row-major 4x4), params[16..20] = g.
    let loss = |p: &[f64]| -> f64 {
        let mut total = 0.0;
        for r in 0..2 {
            let h: Vec<f64> = (0..4).map(|j| (0..4).map(|k| x(r, k) * p[4 * k + j]).sum()).collect();
            let rms = (h.iter().map(|v| v * v).sum::<f64>() / 4.0 + 1e-5).sqrt();
            for j in 0..4 {
                let d = h[j] / rms * p[16 + j] - y(r, j);
                total += d * d;
            }
        }
        total / 8.0
    };
    let mut p: Vec<f64> = (0..16).map(|i| i as f64 * 0.05).chain([1.0; 4]).collect();
    for _ in 0..STEPS {
        let grad: Vec<f64> = (0..p.len())
            .map(|i| {
                let (mut up, mut down) = (p.clone(), p.clone());
                up[i] += 1e-6;
                down[i] -= 1e-6;
                (loss(&up) - loss(&down)) / 2e-6
            })
            .collect();
        for (v, g) in p.iter_mut().zip(&grad) {
            *v -= LR * g;
        }
    }
    (p[..16].to_vec(), p[16..].to_vec())
}

/// The trained values of one lowering against [`reference`]. f32 training
/// over six steps lands within about 1e-6; a gradient off by 10% lands at
/// least ~1e-3 away.
fn assert_matches_reference(label: &str, w: &[f64], g: &[f64]) {
    const TOL: f64 = 1e-5;
    let (rw, rg) = reference();
    assert_eq!((w.len(), g.len()), (16, 4), "{label}: expected 16 weights and 4 gains");
    let worst = w.iter().zip(&rw).chain(g.iter().zip(&rg)).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    eprintln!("{label}: max |Δ| from the f64 reference {worst:.2e} (tolerance {TOL:.0e})");
    for (i, (a, b)) in w.iter().zip(&rw).enumerate() {
        assert!((a - b).abs() < TOL, "{label}: w[{i}]={a} vs f64 reference {b} (|Δ|={:.2e})", (a - b).abs());
    }
    for (i, (a, b)) in g.iter().zip(&rg).enumerate() {
        assert!((a - b).abs() < TOL, "{label}: g[{i}]={a} vs f64 reference {b} (|Δ|={:.2e})", (a - b).abs());
    }
}

/// The reference is not vacuous: both parameter groups travel far beyond the
/// tolerance, so a wrong gradient cannot hide inside it.
#[test]
fn the_reference_trajectory_moves_both_parameter_groups() {
    let (w, g) = reference();
    let w_move = w.iter().enumerate().map(|(i, v)| (v - i as f64 * 0.05).abs()).fold(0.0, f64::max);
    let g_move = g.iter().map(|v| (v - 1.0).abs()).fold(0.0, f64::max);
    assert!(w_move > 0.05 && g_move > 0.1, "reference barely moves: |Δw| {w_move}, |Δg| {g_move}");
}

#[test]
fn source_ad_rmsnorm_dx_matches_tape_ad() {
    let (sa_out, tape_out) = (run(true), run(false));
    assert_matches_reference("source-AD", &parse_w(&sa_out), &parse_g(&sa_out));
    assert_matches_reference("tape-AD", &parse_w(&tape_out), &parse_g(&tape_out));
    let sa = parse_w(&sa_out);
    let tape = parse_w(&tape_out);
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
    assert_matches_reference("fused", &parse_w(&fused_out), &parse_g(&fused_out));
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
    // Device placement: model + inputs on cuda. Each rewrite must fire once:
    // a stale anchor would leave the inputs on the host and the run on CPU.
    let replace_once = |src: String, from: &str, to: &str| {
        assert_eq!(src.matches(from).count(), 1, "fixture rewrite anchor `{from}` must occur once");
        src.replacen(from, to, 1)
    };
    src = replace_once(src, "let m = M()", "let m = M()\nm.to(cuda)");
    src = replace_once(
        src,
        "let y = (arange(8).reshape([2, 4])) * 0.3 - full([2, 4], 1.0)",
        "let y = (arange(8).reshape([2, 4])) * 0.3 - full([2, 4], 1.0)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
    );
    src = replace_once(src, "m.forward(x)", "m.forward(xg)");
    src = replace_once(src, "mse_loss(pred, y)", "mse_loss(pred, yg)");
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
    assert_matches_reference("GPU fused", &wg, &gg);
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
