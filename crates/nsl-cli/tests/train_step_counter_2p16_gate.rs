//! A training run must survive micro-step 65,536.
//!
//! ## The defect this pins
//!
//! `on_step(step, loss)` bound its `step` parameter into `state.variables`
//! without recording a `variable_types` entry. The step-body cleanup sweep
//! (`compile_train_block_inner`) admits slots that are tensor-typed OR
//! indeterminate, and a missing entry reads as indeterminate — so every
//! iteration handed the STEP COUNTER VALUE to `nsl_tensor_free_if_valid`,
//! which probes its argument as a pointer and dereferences it to read the
//! tensor magic.
//!
//! That probe returns early for anything below `0x10000`, which is why the
//! bug was invisible: for the first 65,535 micro-steps the counter is small
//! enough to be skipped, and at exactly 65,536 the guard stops covering it.
//! Fault address `0x10000`, `SEGV_MAPERR`, in `nsl_tensor_free_if_valid`
//! called straight from `main`.
//!
//! It was a hard ceiling on every long run in the language — every
//! production recipe in this repo prints through `on_step`. On 2026-08-29 it
//! killed the 1B intermediate chain and all three arms of a bf16 matched
//! pair at the identical step, after ~600M tokens of shorter runs had never
//! reached it. The 22.6B production epoch (5,516,582 micro-steps) could
//! never have completed.
//!
//! ## Why the fixture looks like this
//!
//! 131,088 corpus elements / (batch 2 * seq 4) = 16,386 micro-steps per
//! epoch; 4 epochs = 65,544 — the CHEAPEST configuration that crosses
//! 65,536, and it runs in about five seconds. The model, the loss and the
//! data are deliberately trivial: the trigger is the COUNTER, so anything
//! else in the fixture would only add runtime and false failure modes.
//!
//! The callback is the whole point — a run without `on_step` never binds an
//! untyped scalar and survived 81,930 steps even before the fix, so a
//! fixture without one would pass vacuously.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const PAST_2P16: &str = r#"from nsl.nn.losses import mse_loss

model Tiny:
    w: Tensor = randn([8, 8])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let corpus = arange(131088)
let loader = DataLoader(corpus, batch_size = 2, seq_len = 4, shuffle = false, drop_last = true)
let x = full([4, 8], 0.5)
let target = zeros([4, 8])

train(model = m, epochs = 4, grad_accumulation = 2):
    optimizer: AdamW(lr = 0.0001)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, target)
    callbacks:
        on_step(step, loss):
            if step % 16384 == 0:
                print(step)
print("PAST_2P16")
"#;

#[test]
fn a_train_block_with_on_step_survives_micro_step_65536() {
    let root = repo_root();
    let dir = std::env::temp_dir().join(format!("nsl_2p16_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("past_2p16.nsl");
    std::fs::write(&prog, PAST_2P16).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--seed", "7"])
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let _ = std::fs::remove_dir_all(&dir);

    // Anti-vacuity: the run must actually REACH the boundary. A fixture that
    // silently stopped early (a loader change, a different accumulation
    // meaning) would otherwise "pass" without ever testing anything.
    assert!(
        stdout.contains("\n65536\n") || stdout.starts_with("65536\n"),
        "fixture never reached micro-step 65,536 — it cannot be testing the \
         boundary. stdout:\n{stdout}\nstderr tail:\n{}",
        stderr.chars().rev().take(600).collect::<String>().chars().rev().collect::<String>()
    );
    assert!(
        out.status.success() && stdout.contains("PAST_2P16"),
        "training died at or after micro-step 65,536 (status {:?}). This is \
         the untyped `on_step` step-counter slot being swept into \
         nsl_tensor_free_if_valid and dereferenced as a pointer at \
         0x10000.\nstdout:\n{stdout}\nstderr tail:\n{}",
        out.status.code(),
        stderr.chars().rev().take(1200).collect::<String>().chars().rev().collect::<String>()
    );
}
