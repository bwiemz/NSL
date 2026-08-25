//! MFU campaign C2 gate — the RoPE backward fold (`rotate_half_neg` emitted
//! instead of `rotate_half` + `Neg`; `NSL_FUSE_ROPE_NEG=0` kill-switch).
//!
//! A permutation with a sign flip has no rounding, so the fold is bit-exact
//! against the two-op pair by construction; the gate demands bit-identical
//! loss streams and the anti-vacuity marker. The packed-GQA fixture's
//! GroupedQueryAttention applies stdlib RoPE, so its adjoint contains the
//! RotateHalfBackward sites the fold rewrites. The fold is generation-time,
//! so it applies under every posture (unlike the chain fuser's CSLA skip).

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

fn run_fixture(tag: &str, gpu: bool, fold_off: bool) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_ropeneg_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/csla_layerwise_packed_gqa.nsl"),
    )
    .unwrap();
    let save = tmp.join("rope.nslm");
    src = src.replace("CSLA_SAVE_PATH", &save.to_string_lossy().replace('\\', "/"));
    if gpu {
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic", "--seed", "777"]);
    if fold_off {
        cmd.env("NSL_FUSE_ROPE_NEG", "0");
    }
    let out = cmd
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
    RunOut {
        success: out.status.success(),
        stdout,
        stderr,
        losses,
    }
}

const ROPE_MARKER: &str = "[fuse] rope backward folds:";

#[test]
fn rope_fold_bit_exact_cpu() {
    let on = run_fixture("cpu_on", false, false);
    assert!(on.success, "fold-on run failed:\n{}", on.stderr);
    assert!(
        on.stderr.contains(ROPE_MARKER),
        "rope fold never fired (vacuous — fixture or fold changed?):\n{}",
        on.stderr
    );
    let off = run_fixture("cpu_off", false, true);
    assert!(off.success, "fold-off run failed:\n{}", off.stderr);
    assert!(
        !off.stderr.contains(ROPE_MARKER),
        "kill-switch did not disable the fold:\n{}",
        off.stderr
    );
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "rope fold changed the loss stream — must be bit-exact"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn rope_fold_bit_exact_gpu() {
    let on = run_fixture("gpu_on", true, false);
    assert!(on.success, "fold-on run failed:\n{}", on.stderr);
    assert!(
        on.stderr.contains(ROPE_MARKER),
        "rope fold never fired:\n{}",
        on.stderr
    );
    let off = run_fixture("gpu_off", true, true);
    assert!(off.success, "fold-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "rope fold changed the GPU loss stream — must be bit-exact"
    );
    let on2 = run_fixture("gpu_on2", true, false);
    assert!(on2.success, "rerun failed:\n{}", on2.stderr);
    assert_eq!(on.losses, on2.losses, "folded path not deterministic");
}
