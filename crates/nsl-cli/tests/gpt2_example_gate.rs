//! `examples/gpt2.nsl` compiles and trains on the CPU.
//!
//! The example used to be a sketch of a pipeline that never compiled: a
//! `train(precision = bf16)` key that does not exist, no `model =`, and
//! imports of stdlib modules that do not exist (`nsl.export`, `nsl.bench`,
//! `nsl.tokenize.BPETrainer`, ...). It is now a GPT-2 at a toy size on a
//! synthetic corpus whose next token is a function of the two before it.
//! The check is the training signal: the loss starts near the uniform
//! cross-entropy ln(64) = 4.16 and must end far below it.

use std::path::PathBuf;
use std::process::Command;

use assert_cmd::prelude::*;

fn workspace_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

#[test]
fn the_gpt2_example_trains_on_the_cpu() {
    let root = workspace_root();
    let out = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .current_dir(root.join("examples"))
        .arg("run")
        .arg(root.join("examples/gpt2.nsl"))
        .args(["--target", "cpu"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "the example did not run:\n{stderr}");
    let losses: Vec<(u64, f64)> = stdout
        .lines()
        .filter_map(|l| {
            let rest = l.strip_prefix("step ")?;
            let (step, loss) = rest.split_once(" | loss ")?;
            Some((step.trim().parse().ok()?, loss.trim().parse().ok()?))
        })
        .collect();
    assert_eq!(
        losses.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        vec![10, 20, 30, 40, 50, 60],
        "one loss line every 10 of the 60 optimizer steps:\n{stdout}"
    );
    let uniform = 64f64.ln();
    let (first, last) = (losses[0].1, losses[5].1);
    assert!(first.is_finite() && first < uniform, "step 10 loss {first} (uniform is {uniform})");
    assert!(last < 0.5 && last < first / 4.0, "the loss did not fall: {losses:?}");
    assert!(stdout.contains("GPT-2 training complete"), "{stdout}");
}
