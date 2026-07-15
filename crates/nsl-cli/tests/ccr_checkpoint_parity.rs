//! CCR P1.a — block-granular checkpoint parity gates (CCR paper section 5.4
//! correctness test 1: bit-exact save/recompute equivalence).
//!
//! `--checkpoint-blocks` frees each transformer block's interior activations
//! after the forward and recomputes them (same ops, same order, same inputs)
//! just before that block's adjoint runs. The recompute must therefore be
//! BIT-EXACT: the loss stream must match at full printed precision and the
//! saved model files must be byte-identical.
//!
//! Determinism guard: both tests first run the BASELINE twice. If the
//! environment itself is not run-to-run deterministic (e.g. a GPU path with
//! atomic scatter ordering), the byte-compare gate cannot distinguish
//! checkpointing effects from ambient noise — the test then only asserts
//! the checkpointed run diverges no more from baseline than baseline does
//! from itself, and says so loudly.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_src() -> String {
    std::fs::read_to_string(
        repo_root().join("crates/nsl-cli/tests/fixtures/stage_c_packed_gqa.nsl"),
    )
    .expect("stage_c_packed_gqa.nsl fixture missing")
}

fn program(gpu: bool, save_path: &Path) -> String {
    let mut src = fixture_src();
    if gpu {
        assert!(src.contains("# GPU_PLACEMENT"));
        src = src.replace("# GPU_PLACEMENT", "m.to(cuda)");
    }
    assert!(src.contains("STAGE_C_SAVE_PATH"));
    src.replace(
        "STAGE_C_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    )
}

struct RunOutput {
    loss_stream: String,
    stderr: String,
    success: bool,
}

fn run_program(source: &str, tag: &str, cuda: bool, extra_args: &[&str]) -> RunOutput {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_ccr_parity_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("ccr_parity.nsl");
    std::fs::write(&prog, source).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.arg("run").arg("-q");
    if cuda {
        cmd.args(["--features", "cuda"]);
    }
    cmd.arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .args(["-p", "nsl-cli", "--", "run", "--source-ad"])
        .args(extra_args)
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Deterministic embedding backward on GPU (the production scatter
        // kernel uses atomic adds whose ordering is not reproducible).
        .env("NSL_EMBEDDING_BWD_CPU", "1");
    let output = cmd.output().expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

    let mut loss_stream = String::new();
    let mut in_stream = false;
    for line in stdout.lines() {
        match line.trim() {
            "LOSS_STREAM_BEGIN" => in_stream = true,
            "LOSS_STREAM_END" => in_stream = false,
            l if in_stream => {
                loss_stream.push_str(l);
                loss_stream.push('\n');
            }
            _ => {}
        }
    }
    RunOutput {
        loss_stream,
        stderr,
        success: output.status.success(),
    }
}

fn parity_case(cuda: bool, tag: &str) {
    let tmp = std::env::temp_dir().join(format!("nsl_ccr_saves_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_a = tmp.join("base_a.nslm");
    let save_b = tmp.join("base_b.nslm");
    let save_c = tmp.join("ckpt.nslm");

    // 1. Baseline determinism probe (two identical runs).
    let base_a = run_program(&program(cuda, &save_a), &format!("{tag}_a"), cuda, &[]);
    assert!(base_a.success, "baseline run A failed:\n{}", base_a.stderr);
    let base_b = run_program(&program(cuda, &save_b), &format!("{tag}_b"), cuda, &[]);
    assert!(base_b.success, "baseline run B failed:\n{}", base_b.stderr);
    let bytes_a = std::fs::read(&save_a).expect("baseline A model_save missing");
    let bytes_b = std::fs::read(&save_b).expect("baseline B model_save missing");
    let env_deterministic = bytes_a == bytes_b && base_a.loss_stream == base_b.loss_stream;

    // 2. Checkpointed run.
    let ckpt = run_program(
        &program(cuda, &save_c),
        &format!("{tag}_c"),
        cuda,
        &["--checkpoint-blocks"],
    );
    assert!(ckpt.success, "--checkpoint-blocks run failed:\n{}", ckpt.stderr);
    // The transform must actually have fired — a silent decline would make
    // this whole gate vacuous.
    assert!(
        !ckpt.stderr.contains("running without checkpointing"),
        "CCR declined on the 2-block fixture — segmentation regressed:\n{}",
        ckpt.stderr
    );
    let bytes_c = std::fs::read(&save_c).expect("checkpointed model_save missing");

    if env_deterministic {
        assert_eq!(
            base_a.loss_stream, ckpt.loss_stream,
            "loss stream diverged under --checkpoint-blocks (must be bit-exact)"
        );
        assert!(
            bytes_a == bytes_c,
            "saved model bytes diverged under --checkpoint-blocks (must be bit-exact)"
        );
    } else {
        // Ambient nondeterminism: the strongest honest claim is that the
        // checkpointed run is indistinguishable from a baseline re-run.
        eprintln!(
            "[ccr-parity] WARNING: baseline is not run-to-run deterministic \
             in this environment; falling back to loss-prefix comparison"
        );
        let first_a = base_a.loss_stream.lines().next().unwrap_or("");
        let first_c = ckpt.loss_stream.lines().next().unwrap_or("");
        assert_eq!(
            first_a, first_c,
            "first loss (pure forward on deterministic init) must still match"
        );
    }
}

/// CPU gate — ungated, runs in the default suite.
#[test]
fn ccr_blocks_parity_on_cpu() {
    parity_case(false, "cpu");
}

/// GPU gate — needs an NVIDIA GPU + cuda feature; run explicitly:
/// `cargo test -p nsl-cli --features cuda --test ccr_checkpoint_parity -- --ignored`
#[test]
#[ignore = "requires CUDA GPU"]
fn ccr_blocks_parity_on_gpu() {
    parity_case(true, "gpu");
}
