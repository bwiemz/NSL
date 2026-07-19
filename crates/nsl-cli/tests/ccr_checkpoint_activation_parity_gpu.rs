//! P0.1 GPU gate: CCR (`--checkpoint-blocks`) activation-recompute parity on
//! the GPU.
//!
//! The GPU twin of `ccr_checkpoint_activation_parity.rs`. A recompute clone
//! replays a FORWARD activation (gelu) but is spliced into the ADJOINT list,
//! which lowers with in-place suppression OFF — so an FBIP-capable activation
//! clone could mutate its own input before the real backward read it (#397).
//! On the GPU the fused activation-backward kernels make this hazard live in a
//! different code path than the CPU gate, so it needs its own coverage.
//!
//! Gate discipline: run the baseline (no checkpointing) TWICE first as a
//! run-to-run determinism probe. `--checkpoint-blocks` recompute is documented
//! bit-exact, so when the GPU baseline is itself deterministic the checkpointed
//! model must be BYTE-identical; if the baseline is not run-to-run stable (an
//! atomicAdd reduction somewhere), fall back to a tight loss + model tolerance,
//! which still catches the clobber (a mutated pre-activation diverges O(1e-1),
//! far outside tolerance).

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

struct Out {
    ok: bool,
    stdout: String,
    stderr: String,
}

fn run(save_path: &std::path::Path, checkpoint_blocks: bool, tag: &str) -> Out {
    let root = repo_root();
    let fixture = root.join("crates/nsl-cli/tests/fixtures/ccr_checkpoint_activation.nsl");
    let src = std::fs::read_to_string(&fixture).expect("ccr_checkpoint_activation.nsl missing");
    assert!(src.contains("CCR_ACT_SAVE_PATH"));
    // GPU placement: put the model on CUDA before the train block.
    assert!(src.contains("let m = Net()"));
    let program = src
        .replace("let m = Net()", "let m = Net()\nm.to(cuda)")
        .replace(
            "CCR_ACT_SAVE_PATH",
            &save_path.display().to_string().replace('\\', "/"),
        );
    let src_path =
        std::env::temp_dir().join(format!("nsl_ccr_act_gpu_{tag}_{}.nsl", std::process::id()));
    std::fs::write(&src_path, program).unwrap();

    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "-p", "nsl-cli", "--", "run", "--source-ad"]);
    if checkpoint_blocks {
        cmd.arg("--checkpoint-blocks");
    }
    // Deterministic embedding scatter is harmless here (no embedding) but keeps
    // parity with the CPU/GPU CCR gates.
    let out = cmd
        .arg(&src_path)
        .current_dir(&root)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_EMBEDDING_BWD_CPU", "1")
        .output()
        .expect("spawn nsl run");
    Out {
        ok: out.status.success(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

fn loss_stream(stdout: &str) -> Vec<f64> {
    let mut v = Vec::new();
    let mut in_stream = false;
    for line in stdout.lines() {
        match line.trim() {
            "LOSS_STREAM_BEGIN" => in_stream = true,
            "LOSS_STREAM_END" => in_stream = false,
            l if in_stream => {
                if let Ok(x) = l.parse::<f64>() {
                    v.push(x);
                }
            }
            _ => {}
        }
    }
    v
}

#[test]
#[ignore = "requires CUDA GPU"]
fn ccr_activation_parity_on_gpu() {
    let tmp = std::env::temp_dir().join(format!("nsl_ccr_act_gpu_saves_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let save_base_a = tmp.join("base_a.nslm");
    let save_base_b = tmp.join("base_b.nslm");
    let save_ckpt = tmp.join("ckpt.nslm");

    // Baseline twice — determinism probe.
    let base_a = run(&save_base_a, false, "base_a");
    assert!(base_a.ok, "baseline A failed:\n{}", base_a.stderr);
    let base_b = run(&save_base_b, false, "base_b");
    assert!(base_b.ok, "baseline B failed:\n{}", base_b.stderr);

    let ckpt = run(&save_ckpt, true, "ckpt");
    assert!(ckpt.ok, "--checkpoint-blocks run failed:\n{}", ckpt.stderr);
    assert!(
        !ckpt.stderr.contains("running without checkpointing"),
        "CCR declined on the GPU activation fixture — segmentation regressed:\n{}",
        ckpt.stderr
    );

    let bytes_base_a = std::fs::read(&save_base_a).expect("baseline A model missing");
    let bytes_base_b = std::fs::read(&save_base_b).expect("baseline B model missing");
    let bytes_ckpt = std::fs::read(&save_ckpt).expect("checkpointed model missing");

    if bytes_base_a == bytes_base_b {
        // GPU baseline is run-to-run deterministic → CCR recompute must be
        // BYTE-identical (the strong claim).
        assert!(
            bytes_base_a == bytes_ckpt,
            "GPU baseline is deterministic but the checkpointed model diverged — \
             a gelu recompute clone likely FBIP-mutated its pre-activation before \
             the real backward read it (ccr::apply_to_adjoint inplace-suppress bracket)"
        );
    } else {
        // Non-deterministic baseline → fall back to a tight tolerance on the
        // loss stream, which is still far tighter than a clobber's divergence.
        let la = loss_stream(&base_a.stdout);
        let lc = loss_stream(&ckpt.stdout);
        assert!(!la.is_empty() && la.len() == lc.len(), "loss streams mismatched length");
        for (a, c) in la.iter().zip(lc.iter()) {
            let rel = (a - c).abs() / a.abs().max(1e-6);
            assert!(
                rel <= 1e-4,
                "GPU loss diverged under --checkpoint-blocks beyond tolerance \
                 (baseline {a}, ckpt {c}, rel {rel:.2e}) — recompute clobber suspected"
            );
        }
    }
}
