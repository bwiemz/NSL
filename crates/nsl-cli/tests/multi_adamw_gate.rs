//! Fusion-queue item 1 gates: multi-tensor fused AdamW.
//!
//! The non-clip FASE-Deferred optimizer loop collapses into ONE
//! pointer-table launch (`nsl_fase_fused_adamw_step_multi`) whose
//! per-element arithmetic is byte-for-byte the single-step kernel's.
//! Contract: training output is BIT-IDENTICAL with the multi path on
//! (default) vs off (`NSL_FASE_MULTI_STEP=0`), on GPU and CPU alike, and
//! the batch launch actually fires (anti-vacuity marker).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct RunOut {
    stdout: String,
    stderr: String,
}

fn run_adamw_mlp(tag: &str, gpu: bool, multi_off: bool) -> RunOut {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_multiadamw_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap();
    if gpu {
        src = src.replace(
            "# GPU_PLACEMENT",
            "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)",
        );
        src = src.replace("m.forward_train(x)", "m.forward_train(xg)");
        src = src.replace("l1_loss(pred, y)", "l1_loss(pred, yg)");
    }
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.args(["run", "--deterministic", "--source-ad"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"));
    if multi_off {
        cmd.env("NSL_FASE_MULTI_STEP", "0");
    }
    let out = cmd.output().expect("spawn nsl run");
    assert!(
        out.status.success(),
        "run failed (tag={tag}):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    RunOut {
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn loss_stream(stdout: &str) -> Vec<String> {
    stdout
        .split("LOSS_STREAM_BEGIN")
        .nth(1)
        .and_then(|r| r.split("LOSS_STREAM_END").next())
        .unwrap_or("")
        .lines()
        .map(str::trim)
        // GPU losses print as `tensor([..])`, CPU losses as bare scalars.
        .filter(|l| l.starts_with("tensor([") || l.parse::<f64>().is_ok())
        .map(str::to_string)
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn multi_adamw_gpu_bit_identical_and_fires() {
    let on = run_adamw_mlp("gpu_on", true, false);
    let off = run_adamw_mlp("gpu_off", true, true);
    let (ls_on, ls_off) = (loss_stream(&on.stdout), loss_stream(&off.stdout));
    assert!(ls_on.len() >= 8, "expected >=8 losses, got {}", ls_on.len());
    assert_eq!(
        ls_on, ls_off,
        "multi-tensor AdamW diverged from the per-param loop (must be bit-identical)"
    );
    // Anti-vacuity: the batched launch actually fired, covering all 6
    // params of the fixture, and the kill-switch actually disabled it.
    assert!(
        on.stderr.contains("[fase-multi] batched 6 params"),
        "batch marker missing — multi path did not fire:\n{}",
        on.stderr
    );
    assert!(
        !off.stderr.contains("[fase-multi]"),
        "kill-switch NSL_FASE_MULTI_STEP=0 did not disable the multi path"
    );
}

#[test]
fn multi_adamw_cpu_bit_identical() {
    // CPU tensors take the multi FFI's per-param fallback arm — output must
    // stay bit-identical to the legacy loop.
    let on = run_adamw_mlp("cpu_on", false, false);
    let off = run_adamw_mlp("cpu_off", false, true);
    let (ls_on, ls_off) = (loss_stream(&on.stdout), loss_stream(&off.stdout));
    assert!(ls_on.len() >= 8, "expected >=8 losses, got {}", ls_on.len());
    assert_eq!(ls_on, ls_off, "CPU fallback arm diverged from the legacy loop");
}
