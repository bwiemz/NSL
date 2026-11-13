//! `t op= e` on a tensor variable is `t = t op e`.
//!
//! The identifier arm of `compile_assign` is scalar arithmetic. For a tensor
//! it emitted `iadd` / `isub` / `imul` / `sdiv` on the two tensor POINTERS, so
//! `loss += aux` stored the sum of two addresses and the program exited 1
//! with no message at the next use. Source AD had the mirror image of the
//! same gap: its extractor skipped every compound assignment, so inside a
//! `grad` or `train` block the statement vanished from the forward graph and
//! the adjoint, and `y += sum(w * 10.0)` after `let y = sum(w * w)` reported
//! a loss of 16 and a gradient of 2w where the program computes 96 and
//! 2w + 10.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap().to_path_buf()
}

fn run(tag: &str, flags: &[&str], src: &str) -> String {
    let dir = std::env::temp_dir().join(format!("nsl_compound_{}_{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let prog = dir.join("p.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--target", "cpu"])
        .args(flags)
        .arg(&prog)
        .current_dir(&dir)
        .env("NSL_STDLIB_PATH", repo_root().join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let _ = std::fs::remove_dir_all(&dir);
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success() && stdout.contains("COMPOUND_DONE"),
        "[{tag}] the program failed (exit {:?}):\n{stdout}\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

#[test]
fn every_compound_operator_works_on_a_tensor() {
    let out = run(
        "plain",
        &[],
        "let w = full([2, 2], 2.0)\n\
         let y = sum(w * w)\n\
         y += sum(w * 10.0)\n\
         print(y.item())\n\
         let t = full([2, 2], 3.0)\n\
         t -= w\n\
         t *= w\n\
         t /= full([2, 2], 4.0)\n\
         print(sum(t).item())\n\
         let acc = zeros([2, 2])\n\
         for i in range(3):\n\
         \x20   acc += w\n\
         print(sum(acc).item())\n\
         let n = 1\n\
         n += 2\n\
         print(n)\n\
         print(\"COMPOUND_DONE\")\n",
    );
    let values: Vec<f64> = out.lines().filter_map(|l| l.trim().parse().ok()).collect();
    // 16 + 80; ((3 - 2) * 2 / 4) * 4 elements; 3 * 2 * 4 elements; the scalar arm.
    assert_eq!(values, [96.0, 2.0, 24.0, 3.0], "{out}");
}

/// Both AD modes differentiate the program as written: loss 96, gradient
/// 2w + 10 = 14 in every element.
#[test]
fn a_compound_assignment_inside_a_grad_block_is_differentiated() {
    let src = "let w = full([2, 2], 2.0)\n\
               let (loss, grads) = grad(w):\n\
               \x20   let y = sum(w * w)\n\
               \x20   y += sum(w * 10.0)\n\
               \x20   y\n\
               print(loss.item())\n\
               print(sum(grads).item())\n\
               print(\"COMPOUND_DONE\")\n";
    for (tag, flag) in [("tape", "--tape-ad"), ("source", "--source-ad")] {
        let out = run(tag, &[flag], src);
        let values: Vec<f64> = out.lines().filter_map(|l| l.trim().parse().ok()).collect();
        assert_eq!(values, [96.0, 56.0], "[{tag}] loss and summed gradient:\n{out}");
    }
}
