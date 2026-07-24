//! Regression gates for the INFERENCE-loop tensor leak (bugs.md 2026-07-23,
//! the "2.3GB/forward" class): a plain `for` loop at script scope doing
//! `let y = m.forward(x)` stranded every iteration's activation forever.
//!
//! Root causes (all in nsl-codegen):
//!   1. The ELTLS loop-let predeclare emitted its zero-def INSIDE the loop
//!      body, re-zeroing the slot every iteration — the rebind free saw 0
//!      instead of the previous iteration's tensor, at ALL five loop
//!      lowerings (for/while/while-let/model-array/dataloader). The free
//!      machinery had never fired at runtime since it shipped.
//!   2. compile_main had no return-local sweep, stranding the final
//!      iteration's value and every other script-scope tensor.
//!   3. `y.sum()` parses as Call{callee: MemberAccess}, which the owned-
//!      temporary predicate never matched — statement-position method temps
//!      (`print(y.sum())`) stranded one reduction block per call.
//!
//! Gate design: run the same forward-only GPU loop at two iteration counts
//! and assert the caching allocator's exit live_blocks are IDENTICAL — any
//! per-iteration strand scales with the count. A companion fixture rebinding
//! a model weight (`let w = m.w_in`) guards the other direction: the
//! activated rebind free must NOT fire on borrowed handles (the ownership
//! veto), which would corrupt weights instead of leaking.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

const MODEL_PREAMBLE: &str = r#"
model Blk:
    w_up: Tensor = randn([64, 128]) * 0.15
    w_down: Tensor = randn([128, 64]) * 0.15

    fn forward(self, h: Tensor) -> Tensor:
        let f = silu(h @ self.w_up) @ self.w_down
        return h + f

model TinyMLP:
    w_in: Tensor = randn([32, 64]) * 0.2
    blocks: [Blk; 2] = Blk()
    w_out: Tensor = randn([64, 32]) * 0.2

    fn forward_train(self, x: Tensor) -> Tensor:
        let h = x @ self.w_in
        for block in self.blocks:
            h = block.forward(h)
        return h @ self.w_out

let m = TinyMLP()
let x = randn([8, 32])
m.to(cuda)
let xg = x.to(cuda)
"#;

/// Weights legitimately live at exit: w_in + 2x(w_up, w_down) + w_out.
/// Everything else (xg, per-iteration activations, reduction temps) must be
/// freed by the loop rebind clears and the main-exit sweep.
const EXPECTED_LIVE_BLOCKS: i64 = 6;

fn run_loop_fixture(body: &str, iters: usize, tag: &str) -> (i64, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_infleak_{}_{tag}_{iters}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = format!(
        "{MODEL_PREAMBLE}for i in range({iters}):\n{body}print(\"DONE\")\n"
    );
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_GPU_MEM_REPORT", "1")
        .output()
        .expect("spawn nsl run");
    assert!(
        out.status.success(),
        "[{tag}/{iters}] run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stdout.contains("DONE"),
        "[{tag}/{iters}] fixture did not complete:\n{stdout}"
    );

    // Last [gpu-mem] report is the process-exit snapshot.
    let live_blocks = stderr
        .lines()
        .filter_map(|l| {
            l.split("live_blocks=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .last()
        .unwrap_or_else(|| panic!("[{tag}/{iters}] no [gpu-mem] live_blocks report:\n{stderr}"));
    (live_blocks, stdout.to_string())
}

/// The core leak gate: `let y = m.forward_train(xg)` in a script-scope loop.
/// Pre-fix this stranded one activation block per iteration (live_blocks
/// scaled 3 -> 9 with the count); post-fix both runs sit at the weight count.
#[test]
#[ignore = "requires CUDA GPU"]
fn inference_loop_method_call_live_blocks_iteration_independent() {
    let body = "    let y = m.forward_train(xg)\n";
    let (lb3, _) = run_loop_fixture(body, 3, "mcall");
    let (lb9, _) = run_loop_fixture(body, 9, "mcall");
    assert_eq!(
        lb3, lb9,
        "live_blocks scale with iteration count — per-iteration inference strand is back"
    );
    assert_eq!(
        lb3, EXPECTED_LIVE_BLOCKS,
        "live_blocks != weight count — a script-scope tensor (xg / final y) is stranding"
    );
}

/// Statement-position method temps: `print(y.sum())` stranded one reduction
/// block per call because Call{callee: MemberAccess} never matched the
/// owned-temporary predicate.
#[test]
#[ignore = "requires CUDA GPU"]
fn inference_loop_method_temp_live_blocks_iteration_independent() {
    let body = "    let y = m.forward_train(xg)\n    print(y.sum())\n";
    let (lb3, out3) = run_loop_fixture(body, 3, "mtemp");
    let (lb9, out9) = run_loop_fixture(body, 9, "mtemp");
    // Anti-vacuity: the sums actually printed (method temp path exercised).
    assert_eq!(out3.matches("tensor(").count(), 3, "expected 3 printed sums:\n{out3}");
    assert_eq!(out9.matches("tensor(").count(), 9, "expected 9 printed sums:\n{out9}");
    assert_eq!(
        lb3, lb9,
        "live_blocks scale with iteration count — method-call statement temps are stranding again"
    );
    assert_eq!(lb3, EXPECTED_LIVE_BLOCKS);
}

/// Ownership-veto guard: rebinding a BORROW (`let w = m.w_in`) in a loop must
/// not fire the rebind free — that would free the model weight itself and
/// corrupt every subsequent forward. The printed sums double as a liveness
/// check: identical values every iteration means the weight survived.
#[test]
#[ignore = "requires CUDA GPU"]
fn inference_loop_borrow_rebind_does_not_free_weights() {
    let body = "    let w = m.w_in\n    let y = xg @ w\n    print(y.sum())\n";
    let (lb, out) = run_loop_fixture(body, 6, "borrow");
    let sums: Vec<&str> = out
        .lines()
        .filter(|l| l.trim_start().starts_with("tensor("))
        .collect();
    assert_eq!(sums.len(), 6, "expected 6 printed sums:\n{out}");
    assert!(
        sums.iter().all(|s| *s == sums[0]),
        "per-iteration sums diverged — the borrowed weight was freed/corrupted: {sums:?}"
    );
    // w_in survives; y freed per-iteration; xg swept at exit.
    assert_eq!(lb, EXPECTED_LIVE_BLOCKS);
}
