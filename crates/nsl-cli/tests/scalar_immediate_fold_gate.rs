//! MFU campaign C3 gate — the standalone scalar-immediate sweep
//! (`x OP const` with a RIGHT-operand constant -> one scalar-immediate
//! kernel; `NSL_FUSE_SCALAR_IMM=0` kill-switch).
//!
//! Bit-exact: the baseline narrows the f64 constant to f32 once at
//! `nsl_tensor_scalar(v, 1)` creation, uploads it, broadcast-materializes
//! full-size, then runs the same single `.rn` f32 kernel op; the scalar FFI
//! performs the identical narrowing at launch. So: bit-identical loss
//! streams on vs off, plus the anti-vacuity marker (the decomposed-rmsnorm
//! fixture's `+ eps` Adds are exactly the population — their consumer is a
//! Sqrt barrier, so the chain fuser leaves them for the sweep).

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

fn run_fixture(tag: &str, gpu: bool, sweep_off: bool) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_scalimm_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/ew_chain_fusion.nsl"),
    )
    .unwrap();
    if gpu {
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--source-ad", "--deterministic", "--seed", "777"]);
    if sweep_off {
        cmd.env("NSL_FUSE_SCALAR_IMM", "0");
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

const IMM_MARKER: &str = "[fuse] scalar immediates:";

#[test]
fn scalar_immediates_bit_exact_cpu() {
    let on = run_fixture("cpu_on", false, false);
    assert!(on.success, "sweep-on run failed:\n{}", on.stderr);
    assert!(
        on.stderr.contains(IMM_MARKER),
        "scalar sweep never fired (vacuous — fixture or sweep changed?):\n{}",
        on.stderr
    );
    let off = run_fixture("cpu_off", false, true);
    assert!(off.success, "sweep-off run failed:\n{}", off.stderr);
    assert!(
        !off.stderr.contains(IMM_MARKER),
        "kill-switch did not disable the sweep:\n{}",
        off.stderr
    );
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "scalar-immediate sweep changed the loss stream — must be bit-exact"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn scalar_immediates_bit_exact_gpu() {
    let on = run_fixture("gpu_on", true, false);
    assert!(on.success, "sweep-on run failed:\n{}", on.stderr);
    assert!(
        on.stderr.contains(IMM_MARKER),
        "scalar sweep never fired:\n{}",
        on.stderr
    );
    let off = run_fixture("gpu_off", true, true);
    assert!(off.success, "sweep-off run failed:\n{}", off.stderr);
    assert!(!on.losses.is_empty(), "no losses parsed:\n{}", on.stdout);
    assert_eq!(
        on.losses, off.losses,
        "scalar-immediate sweep changed the GPU loss stream — must be bit-exact"
    );
}
