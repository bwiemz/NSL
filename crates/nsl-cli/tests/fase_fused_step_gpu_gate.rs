//! Milestone C · p9 — fused FASE-Deferred optimizer-step gate.
//!
//! `fase_emit_final_step` interprets the AdamW `UpdateProgram` as ~15 kernel
//! launches + 3 DtoD copies + ~10 transient alloc/frees per parameter per
//! optimizer step. p9 replaces that with ONE fused kernel launch per parameter
//! (`nsl_fase_fused_adamw_step` / `FASE_FUSED_ADAMW_STEP_F32_PTX`), which
//! mirrors every interpreted op's rounding (`.rn`, `sqrt.rn`, `div.approx`)
//! and so must be **bit-identical**.
//!
//! The proof is the campaign-standard differential: the SAME FASE-Deferred
//! AdamW training run under
//!   * `NSL_FASE_FUSED_STEP=0` (interpreted per-op program — old behavior),
//!   * default                  (fused single-launch step)
//! must produce bit-identical trained-parameter sums under `--deterministic`.
//! Anti-vacuity: the fused arm runs with `NSL_FASE_FUSED_COUNTER=1` and must
//! report a nonzero `[fase-fused]` launch count (the fused path really fired;
//! grad_accumulation=2 over 4 epochs of 1 step ⇒ 2 optimizer windows × 2
//! params = 4 fused launches expected).
//!
//! GPU-only, `#[ignore]`. Run:
//!   cargo test -p nsl-cli --features cuda --test fase_fused_step_gpu_gate \
//!     -- --ignored --nocapture --test-threads=1

use std::path::PathBuf;
use std::process::Command;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// A small MLP trained on GPU with AdamW + grad_accumulation=2, which engages
/// FASE-Deferred mode (AdamW + accumulation ≥ 2 ⇒ `FaseMode::Deferred`) and
/// therefore the fused-vs-interpreted final step under test. weight_decay is
/// nonzero so the fused kernel's wd arm is exercised; silu keeps a nonlinear
/// backward in play. (No bias param: a broadcast-add bias hits a pre-existing
/// FASE-accumulate shape abort unrelated to p9 — fails identically both arms.)
const FASE_TRAIN_SRC: &str = r#"from nsl.nn.losses import mse_loss

model Mlp:
    w1: Tensor = ones([16, 16])
    w2: Tensor = ones([16, 16])

    fn forward(self, x: Tensor) -> Tensor:
        let h = silu(x @ self.w1)
        return h @ self.w2

let m = Mlp()
m.to(cuda)
let x = ones([4, 16])
let y = zeros([4, 16])

train(model = m, epochs = 4, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01)
    step(batch):
        let out = m.forward(x)
        let loss = mse_loss(out, y)

print("AFTER_w1")
print(sum(m.w1))
print("AFTER_w2")
print(sum(m.w2))
"#;

/// Run the fixture. `interpreted` sets `NSL_FASE_FUSED_STEP=0` (per-op program).
/// Returns (after_sums_block, success, stderr).
fn run(fixture: &std::path::Path, interpreted: bool) -> (String, bool, String) {
    let root = repo_root();
    let mut cmd = Command::new(env!("CARGO"));
    cmd.args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            // FASE-Deferred requires source-AD (the accumulate hook path).
            "--source-ad",
            // bit-reproducible GPU path, so fused vs interpreted is exact-comparable.
            "--deterministic",
            "--target",
            "sm_89",
        ])
        .arg(fixture)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        // Anti-vacuity counter report (atexit, stderr) — harmless in both arms.
        .env("NSL_FASE_FUSED_COUNTER", "1");
    if interpreted {
        cmd.env("NSL_FASE_FUSED_STEP", "0");
    }
    let out = cmd.output().expect("spawn nsl run");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let lines: Vec<&str> = stdout.lines().collect();
    let mut block = String::new();
    for (i, line) in lines.iter().enumerate() {
        let t = line.trim();
        if let Some(name) = t.strip_prefix("AFTER_") {
            for next in lines.iter().skip(i + 1) {
                let n = next.trim();
                if n.is_empty() || n.starts_with('[') {
                    continue;
                }
                // GPU scalar sums print as `tensor([N])`; CPU sums as a bare
                // number. Accept both; keep the exact digit string for the
                // bit-exact comparison.
                let num = n
                    .strip_prefix("tensor([")
                    .and_then(|r| r.strip_suffix("])"))
                    .unwrap_or(n);
                if num.parse::<f64>().is_ok() {
                    block.push_str(name);
                    block.push('=');
                    block.push_str(num);
                    block.push('\n');
                }
                break;
            }
        }
    }
    (
        block,
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Parse the `[fase-fused] optimizer fused-step launches: N` atexit report.
fn fused_count(stderr: &str) -> Option<u64> {
    stderr
        .lines()
        .find_map(|l| l.strip_prefix("[fase-fused] optimizer fused-step launches: "))
        .and_then(|n| n.trim().parse().ok())
}

#[test]
#[ignore = "requires a CUDA GPU (2 FASE training runs)"]
fn fused_step_matches_interpreted_bit_exact() {
    let tmp = std::env::temp_dir().join("nsl_fase_fused_step_gate.nsl");
    std::fs::write(&tmp, FASE_TRAIN_SRC).expect("write fixture");

    // Interpreted (kill-switch) first, so a cold compile doesn't skew things.
    let (interp_sums, ok_interp, err_interp) = run(&tmp, true);
    assert!(ok_interp, "interpreted (NSL_FASE_FUSED_STEP=0) run failed:\n{err_interp}");
    let (fused_sums, ok_fused, err_fused) = run(&tmp, false);
    assert!(ok_fused, "fused (default) run failed:\n{err_fused}");

    // Sanity: both params captured; w1 moved from its 256.0 init (a real
    // AdamW step happened, not a no-op that would trivially match).
    let n = interp_sums.lines().count();
    assert_eq!(n, 2, "expected 2 AFTER_ sums, got {n}:\ninterp=\n{interp_sums}");
    assert!(
        interp_sums.contains("w1=") && !interp_sums.contains("w1=256.0\n"),
        "expected w1 to move from its 256.0 init (a real optimizer step):\n{interp_sums}"
    );

    // Anti-vacuity: the fused arm actually took the fused path (and the
    // interpreted arm did NOT — kill-switch honored).
    let fused_n = fused_count(&err_fused).expect("missing [fase-fused] report in fused arm");
    assert!(
        fused_n > 0,
        "fused arm reported 0 fused-step launches — admission never fired:\n{err_fused}"
    );
    let interp_n = fused_count(&err_interp).unwrap_or(0);
    assert_eq!(
        interp_n, 0,
        "interpreted arm (NSL_FASE_FUSED_STEP=0) still took the fused path"
    );

    assert_eq!(
        fused_sums, interp_sums,
        "fused optimizer step diverged from the interpreted per-op program — \
         the kernel does not mirror the program's rounding. Re-run with \
         NSL_FASE_FUSED_STEP=0 to confirm, then audit \
         FASE_FUSED_ADAMW_STEP_F32_PTX against fase_emit_final_step's op order."
    );

    eprintln!(
        "[fase-fused-gate] OK: trained-param sums bit-exact (interpreted == fused, {fused_n} fused launches)\n{interp_sums}"
    );
}
