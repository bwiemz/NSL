//! P1 Muon gates:
//! - item 6: parameter-ROLE routing (@param_role decorator + embedding_lookup
//!   usage inference) replacing the name-substring exclusion list.
//! - item 7: ns_steps must be a positive integer literal.
//! - items 8+10: the planned runtime orthogonalization primitive
//!   (muon_orthogonalize_fast) matches the NSL reference on CPU (f64 tight)
//!   and GPU (f32, norm computed device-resident).
//! - item 11: Muon x --layerwise-accum (separate-accumulator window mode) is
//!   BIT-IDENTICAL to the non-CSLA Muon run at the same config.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
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

fn run_source(tag: &str, src: &str, extra: &[&str]) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_muon_p1_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
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
                    if let Some(inner) = l
                        .strip_prefix("tensor([")
                        .and_then(|r| r.strip_suffix("])"))
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
    RunOut {
        success: out.status.success(),
        stdout,
        stderr,
        losses,
    }
}

fn run_fixture(fixture: &str, tag: &str, rewrites: &[(&str, &str)], extra: &[&str]) -> RunOut {
    let root = repo_root();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures").join(fixture),
    )
    .unwrap();
    for (from, to) in rewrites {
        assert!(src.contains(from), "rewrite marker '{from}' missing in {fixture}");
        src = src.replace(from, to);
    }
    run_source(tag, &src, extra)
}

// ── Item 6: parameter-role routing ──────────────────────────────────────

/// The three name-fragility cases the role system exists to fix, on one
/// model: an embedding named OUTSIDE the old list (usage-inferred), a
/// hidden weight whose name contains "embed" (default hidden), and an
/// untied head routed by @param_role.
#[test]
fn roles_route_by_role_not_name() {
    let tmp = std::env::temp_dir().join(format!("nsl_muon_p1_roles_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save = tmp.join("roles.nslm");
    let out = run_fixture(
        "muon_roles.nsl",
        "roles",
        &[("MUON_SAVE_PATH", &save.display().to_string().replace('\\', "/"))],
        &[],
    );
    assert!(out.success, "roles fixture failed:\n{}", out.stderr);
    let table: Vec<&str> = out
        .stderr
        .lines()
        .filter(|l| l.contains("[muon]"))
        .collect();
    let has = |frag: &[&str]| {
        table
            .iter()
            .any(|l| frag.iter().all(|f| l.contains(f)))
    };
    assert!(
        has(&["m.tok_table", "role=embedding", "embedding_lookup usage", "AdamW"]),
        "off-list embedding not usage-routed:\n{}",
        table.join("\n")
    );
    assert!(
        has(&["m.embed_proj", "role=hidden", "default", "Muon"]),
        "embed-NAMED hidden weight not Muon-routed (substring ghost?):\n{}",
        table.join("\n")
    );
    assert!(
        has(&["m.out_proj", "role=head", "@param_role", "AdamW"]),
        "decorated head not AdamW-routed:\n{}",
        table.join("\n")
    );
    assert!(
        has(&["m.norm_out", "role=vector", "rank"]),
        "rank-1 param not vector-classified:\n{}",
        table.join("\n")
    );
    assert!(!out.losses.is_empty(), "no losses:\n{}", out.stdout);
}

/// An invalid @param_role value is a hard compile error, not a silent
/// fallback to inference.
#[test]
fn param_role_invalid_value_is_fatal() {
    let src = r#"
model Bad:
    @param_role("emedding")
    w: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Bad()
print("ok")
"#;
    let out = run_source("bad_role", src, &[]);
    assert!(!out.success, "typo'd @param_role compiled:\n{}", out.stdout);
    assert!(
        out.stderr.contains("not a recognized parameter role"),
        "wrong error:\n{}",
        out.stderr
    );
}

// ── Item 7: ns_steps validation ─────────────────────────────────────────

#[test]
fn ns_steps_rejects_zero_negative_and_float() {
    let prog = |ns: &str| {
        format!(
            r#"from nsl.nn.losses import mse_loss

model T:
    w: Tensor = ones([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = T()
let x = ones([2, 4])
let y = zeros([2, 4])

train(model = m, epochs = 1):
    optimizer: Muon(lr = 0.01, ns_steps = {ns})
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)
"#
        )
    };
    for (ns, tag) in [("0", "zero"), ("-3", "neg"), ("3.5", "float")] {
        let out = run_source(&format!("ns_{tag}"), &prog(ns), &[]);
        assert!(!out.success, "ns_steps={ns} compiled:\n{}", out.stdout);
        assert!(
            out.stderr.contains("ns_steps must be a positive integer"),
            "ns_steps={ns}: wrong error:\n{}",
            out.stderr
        );
    }
}

// ── Items 8+10: planned primitive vs NSL reference ──────────────────────

const EQUIV_PROBE: &str = r#"
from nsl.optim.muon import muon_orthogonalize

# Wide (4x6), tall (6x4), square (5x5) — sqrt ramp is IEEE-exact and
# full-rank (see muon_orthogonalize_probe.nsl).
let gw = (sqrt(arange(24) * 1.7 + full([24], 0.3)) - full([24], 1.5)).reshape([4, 6])
let gt = (sqrt(arange(24) * 2.3 + full([24], 0.7)) - full([24], 1.2)).reshape([6, 4])
let gs = (sqrt(arange(25) * 1.3 + full([25], 0.5)) - full([25], 1.4)).reshape([5, 5])
# GPU_PLACEMENT_W
# GPU_PLACEMENT_T
# GPU_PLACEMENT_S
print("DIFF_BEGIN")
let ow = muon_orthogonalize(gw, 5.0)
let fw = muon_orthogonalize_fast(gw, 5.0)
print(sum((ow - fw) * (ow - fw)).item())
let ot = muon_orthogonalize(gt, 5.0)
let ft = muon_orthogonalize_fast(gt, 5.0)
print(sum((ot - ft) * (ot - ft)).item())
let os = muon_orthogonalize(gs, 5.0)
let fs = muon_orthogonalize_fast(gs, 5.0)
print(sum((os - fs) * (os - fs)).item())
print("DIFF_END")
"#;

fn parse_diffs(stdout: &str) -> Vec<f64> {
    stdout
        .split_once("DIFF_BEGIN")
        .and_then(|(_, r)| r.split_once("DIFF_END"))
        .map(|(v, _)| {
            v.lines()
                .filter_map(|l| l.trim().parse::<f64>().ok())
                .collect()
        })
        .unwrap_or_default()
}

#[test]
fn fast_primitive_matches_reference_cpu() {
    let out = run_source("equiv_cpu", EQUIV_PROBE, &[]);
    assert!(out.success, "equivalence probe failed:\n{}", out.stderr);
    let diffs = parse_diffs(&out.stdout);
    assert_eq!(diffs.len(), 3, "expected 3 diffs:\n{}", out.stdout);
    for (i, d) in diffs.iter().enumerate() {
        // CPU f64: identical op sequence except the pre-normalization sum
        // order (pre- vs post-transpose) — O(eps) only.
        assert!(
            *d < 1e-18,
            "shape-class {i}: sum-sq diff {d} vs reference (CPU f64)"
        );
    }
}

/// GPU: the fast path computes the norm DEVICE-RESIDENT in f32 (vs the
/// reference's f64 host round-trip) — small rounding drift is expected,
/// transpose/order bugs are O(1).
#[test]
#[ignore = "requires CUDA GPU"]
fn fast_primitive_matches_reference_gpu() {
    let src = EQUIV_PROBE
        .replace("# GPU_PLACEMENT_W", "gw = gw.to(cuda)")
        .replace("# GPU_PLACEMENT_T", "gt = gt.to(cuda)")
        .replace("# GPU_PLACEMENT_S", "gs = gs.to(cuda)");
    let out = run_source("equiv_gpu", &src, &[]);
    assert!(out.success, "GPU equivalence probe failed:\n{}", out.stderr);
    let diffs = parse_diffs(&out.stdout);
    assert_eq!(diffs.len(), 3, "expected 3 diffs:\n{}", out.stdout);
    for (i, d) in diffs.iter().enumerate() {
        assert!(
            *d < 1e-6,
            "shape-class {i}: sum-sq diff {d} vs reference (GPU f32)"
        );
    }
}

// ── Item 11: Muon x CSLA bit-exactness ──────────────────────────────────

#[test]
fn csla_muon_bit_exact_vs_baseline_cpu() {
    let muon_line = (
        "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
        "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
         weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
    );
    let tmp = std::env::temp_dir().join(format!("nsl_muon_p1_csla_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_base = tmp.join("base.nslm").display().to_string().replace('\\', "/");
    let save_csla = tmp.join("csla.nslm").display().to_string().replace('\\', "/");

    let base = run_fixture(
        "csla_layerwise_ffn.nsl",
        "csla_base",
        &[muon_line, ("CSLA_SAVE_PATH", &save_base)],
        &[],
    );
    assert!(base.success, "baseline muon failed:\n{}", base.stderr);
    assert!(!base.losses.is_empty(), "empty baseline stream");

    let csla = run_fixture(
        "csla_layerwise_ffn.nsl",
        "csla_lw",
        &[muon_line, ("CSLA_SAVE_PATH", &save_csla)],
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(csla.success, "csla muon failed:\n{}", csla.stderr);
    assert_eq!(
        base.losses, csla.losses,
        "muon x --layerwise-accum loss stream diverged from baseline\nstderr:\n{}",
        csla.stderr
    );
    let a = std::fs::read(tmp.join("base.nslm")).expect("baseline .nslm");
    let b = std::fs::read(tmp.join("csla.nslm")).expect("csla .nslm");
    assert_eq!(a, b, "model bytes diverged under --layerwise-accum");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn csla_muon_bit_exact_vs_baseline_gpu() {
    let muon_line = (
        "AdamW(lr=0.002, weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
        "Muon(lr=0.002, momentum=0.95, nesterov=true, ns_steps=5, \
         weight_decay=0.01, beta1=0.9, beta2=0.95, eps=1e-8)",
    );
    let gpu = ("# GPU_PLACEMENT", "m.to(cuda)");
    let tmp = std::env::temp_dir().join(format!("nsl_muon_p1_csla_gpu_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_base = tmp.join("gbase.nslm").display().to_string().replace('\\', "/");
    let save_csla = tmp.join("gcsla.nslm").display().to_string().replace('\\', "/");

    let base = run_fixture(
        "csla_layerwise_ffn.nsl",
        "csla_gbase",
        &[muon_line, gpu, ("CSLA_SAVE_PATH", &save_base)],
        &[],
    );
    assert!(base.success, "GPU baseline muon failed:\n{}", base.stderr);

    let csla = run_fixture(
        "csla_layerwise_ffn.nsl",
        "csla_glw",
        &[muon_line, gpu, ("CSLA_SAVE_PATH", &save_csla)],
        &["--checkpoint-blocks", "--layerwise-accum"],
    );
    assert!(csla.success, "GPU csla muon failed:\n{}", csla.stderr);
    assert_eq!(
        base.losses, csla.losses,
        "GPU muon x --layerwise-accum diverged\nstderr:\n{}",
        csla.stderr
    );
}
