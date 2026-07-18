//! CCR (`--checkpoint-blocks`) x FBIP-capable activation regression.
//!
//! A recompute clone replays an original FORWARD op (e.g. `gelu(pre)`) but
//! is spliced into the ADJOINT list, which lowers with in-place suppression
//! OFF (adjoint temps are normally single-use, so FBIP is allowed there).
//! An FBIP-capable, input-reading activation clone (gelu/silu/abs) can then
//! mutate its own input in place before the real backward formula for that
//! same activation reads it — silently wrong gradients, with no crash and
//! no diagnostic. `ccr::apply_to_adjoint` now brackets each recompute-clone
//! splice with `nsl_set_inplace_suppressed` on/off so clones replay under
//! the same in-place semantics the original forward used.
//!
//! This is a dedicated, minimal fixture (not `stage_c_packed_gqa.nsl`,
//! which has no activation functions and is shared with several other
//! parity/differential tests) so this gate can't be muddied by unrelated
//! attention/embedding nondeterminism. `--checkpoint-blocks` recompute is
//! documented bit-exact (CCR section 5.4 test 1), so a plain --source-ad
//! run and a --source-ad --checkpoint-blocks run of the identical program
//! must save byte-identical models. The printed loss alone can't catch
//! this regression (it's the pre-step forward value, insensitive to a
//! corrupted gradient in a one-step run) — the saved model bytes are the
//! gate.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run(save_path: &std::path::Path, checkpoint_blocks: bool, tag: &str) -> (bool, String, String) {
    let root = repo_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/ccr_checkpoint_activation.nsl");
    let src = std::fs::read_to_string(&fixture).expect("ccr_checkpoint_activation.nsl missing");
    assert!(src.contains("CCR_ACT_SAVE_PATH"));
    let program = src.replace(
        "CCR_ACT_SAVE_PATH",
        &save_path.display().to_string().replace('\\', "/"),
    );
    let src_path = std::env::temp_dir().join(format!("nsl_ccr_act_{tag}_{}.nsl", std::process::id()));
    std::fs::write(&src_path, program).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "-p", "nsl-cli", "--", "run", "--source-ad"]);
    if checkpoint_blocks {
        cmd.arg("--checkpoint-blocks");
    }
    let output = cmd
        .arg(&src_path)
        .current_dir(&root)
        .output()
        .expect("spawn nsl run");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn ccr_activation_parity_on_cpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_ccr_act_saves_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_base = tmp.join("base.nslm");
    let save_ckpt = tmp.join("ckpt.nslm");

    let (ok_base, stdout_base, stderr_base) = run(&save_base, false, "base");
    assert!(ok_base, "baseline run failed:\nstdout:\n{stdout_base}\nstderr:\n{stderr_base}");

    let (ok_ckpt, stdout_ckpt, stderr_ckpt) = run(&save_ckpt, true, "ckpt");
    assert!(
        ok_ckpt,
        "--checkpoint-blocks run failed:\nstdout:\n{stdout_ckpt}\nstderr:\n{stderr_ckpt}"
    );
    // A silent CCR decline (e.g. the blocks.N segmentation regressed) would
    // make this gate vacuous.
    assert!(
        !stderr_ckpt.contains("running without checkpointing"),
        "CCR declined on the activation fixture — segmentation regressed:\n{stderr_ckpt}"
    );

    assert_eq!(
        stdout_base, stdout_ckpt,
        "forward loss diverged under --checkpoint-blocks (must be bit-exact)"
    );

    let bytes_base = std::fs::read(&save_base).expect("baseline model_save missing");
    let bytes_ckpt = std::fs::read(&save_ckpt).expect("checkpointed model_save missing");
    assert!(
        bytes_base == bytes_ckpt,
        "saved model diverged under --checkpoint-blocks with a gelu-activation \
         block interior — a recompute clone likely FBIP-mutated its own input \
         (e.g. gelu's pre-activation) before the real backward read it \
         (see ccr::apply_to_adjoint's inplace-suppress bracket)"
    );
}
