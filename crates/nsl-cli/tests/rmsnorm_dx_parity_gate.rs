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
    let after = stdout.split_once("W_BEGIN").map(|(_, r)| r).unwrap_or("");
    let inner = after.split_once("W_END").map(|(l, _)| l).unwrap_or("");
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
    // Item 9 fusion: `--fuse-rmsnorm-backward` lowers the RMSNorm dx to a single
    // fused op. On the CPU path it uses the same f64 formula as tape-AD, so it
    // must match tape-AD (and the decomposition) to an f32 tolerance.
    let fused = parse_w(&run_args(&["--source-ad", "--fuse-rmsnorm-backward"]));
    let decomp = parse_w(&run(true));
    let tape = parse_w(&run(false));
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
}
