//! Regression gate: a builtin call NESTED as another builtin's argument
//! must not strand its fresh result (2026-07-29).
//!
//! Most dispatch arms in `compile_call` compiled their arguments with
//! `compile_expr`, which never registers a nested call's result as a
//! statement temporary — `sum(cumsum(g, -1))` stranded the inner cumsum
//! output 1 block per call while `let t = cumsum(g, -1); sum(t)` was
//! clean (bound locals are swept). Measured 2026-07-28 during the
//! device-blind sampling work: device-independent, ~76 `compile_expr`
//! argument sites plus `matches!`-style arms like `sum`/`mean`. The fix
//! switches argument compilation to `compile_nested_expr`, whose
//! tracking predicate (`expr_call_returns_owning_ref` + tensor-typed
//! result) registers exactly the fresh-owning nested results and leaves
//! idents, literals, and non-tensor arguments untouched.
//!
//! The parity half of each fixture pins VALUES: the nested spelling must
//! compute the same result as the bound spelling — statement-end frees
//! must not touch anything a later op still reads.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run a whole program; return (exit live_blocks, stdout).
fn run_fixture(src: &str, tag: &str) -> (i64, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_nestarg_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
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
        "[{tag}] run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stdout.contains("DONE"), "[{tag}] incomplete:\n{stdout}");
    let lb = stderr
        .lines()
        .filter_map(|l| {
            l.split("live_blocks=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .next_back()
        .unwrap_or_else(|| panic!("[{tag}] no live_blocks report:\n{stderr}"));
    std::fs::remove_dir_all(&tmp).ok();
    (lb, stdout)
}

/// Nested spellings across the sampling + norm + reduction families, with
/// bound-spelling parity checks folded in (err must be exactly 0).
fn rounds_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
let ramp = arange(0.0, 16.0)
let gl = ((lt_scalar(ramp, 6.0) - lt_scalar(ramp, 5.0)) * 100.0).to(cuda)
let oh = (lt_scalar(ramp, 3.0) - lt_scalar(ramp, 2.0)).to(cuda)
let w = (ramp * 0.0 + 1.0).to(cuda)
let err = 0.0
"#,
    );
    for i in 0..calls {
        s.push_str(&format!(
            "let n1_{i} = sum(cumsum(gl, -1)).item()\n\
             let t1_{i} = cumsum(gl, -1)\n\
             let b1_{i} = sum(t1_{i}).item()\n\
             let n2_{i} = sum(lt_scalar(gl, 0.5)).item()\n\
             let t2_{i} = lt_scalar(gl, 0.5)\n\
             let b2_{i} = sum(t2_{i}).item()\n\
             let n3_{i} = sum(multinomial(oh, 4)).item()\n\
             let t3_{i} = multinomial(oh, 4)\n\
             let b3_{i} = sum(t3_{i}).item()\n\
             let n4_{i} = sum(rmsnorm(gl, w, 0.00001)).item()\n\
             let t4_{i} = rmsnorm(gl, w, 0.00001)\n\
             let b4_{i} = sum(t4_{i}).item()\n\
             let n5_{i} = sum(cumsum(cumsum(gl, -1), -1)).item()\n\
             let t5a_{i} = cumsum(gl, -1)\n\
             let t5b_{i} = cumsum(t5a_{i}, -1)\n\
             let b5_{i} = sum(t5b_{i}).item()\n\
             err = err + (n1_{i} - b1_{i}) * (n1_{i} - b1_{i}) \
                       + (n2_{i} - b2_{i}) * (n2_{i} - b2_{i}) \
                       + (n3_{i} - b3_{i}) * (n3_{i} - b3_{i}) \
                       + (n4_{i} - b4_{i}) * (n4_{i} - b4_{i}) \
                       + (n5_{i} - b5_{i}) * (n5_{i} - b5_{i})\n"
        ));
    }
    s.push_str(
        r#"print(err)
if err == 0.0:
    print("PARITY_OK")
print("DONE")
"#,
    );
    s
}

/// Nested-arg spellings inside a WHILE CONDITION and a FOR ITERABLE.
/// Pre-fix these were a COMPILE-TIME ICE ("range start index out of
/// range", review HIGH on 3b7f085f): condition temps registered before
/// the loop-scope mark were stolen by the body's whole-list drain,
/// leaving the exit cleanup's `[scope_start..]` slice out of bounds —
/// pre-existing on main via method-form conditions, made trivially
/// reachable by the arg conversion. `free_tensor_temporaries` now drains
/// only above the innermost mark.
///
/// The while-condition residual is now ZERO: `free_condition_temporaries`
/// frees each evaluation's temps inside the loop HEADER (after the
/// branch scalar is computed, before the brif) and drains them from the
/// list so the statement-end cleanup cannot double-free. The old
/// exact-6 pin (4 evaluations x 2 temps - 2 freed at statement end)
/// documented the strand this fix retires; the two iteration counts
/// below (4 and 7 evaluations) prove the frees are genuinely
/// per-evaluation rather than a constant offset. The for-iterable
/// evaluates once and was already clean.
#[test]
#[ignore = "requires CUDA GPU"]
fn loop_condition_nested_args_compile_and_run() {
    for (start, evals) in [("3.0", "4"), ("6.0", "7")] {
        let while_src = format!(
            r#"
let g = arange(0.0, 8.0).to(cuda)
let n = {start}
while sum(cumsum(g, -1)).item() * n > 0.0:
    n = n - 1.0
print(n)
print("DONE")
"#
        );
        let (lb, stdout) = run_fixture(&while_src, &format!("whilecond{evals}"));
        let val = stdout
            .lines()
            .take_while(|l| *l != "DONE")
            .last()
            .unwrap_or("");
        assert_eq!(val, "0", "while-condition loop computed wrong value:\n{stdout}");
        assert_eq!(
            lb, 0,
            "while-condition temps stranded at {evals} evaluations:\n{stdout}"
        );
    }

    // while-let goes through the same helper; an int-typed binding whose
    // expression carries nested tensor temps terminates naturally when
    // the sum reaches zero (3 evaluations here, 2 temps each — 4
    // stranded pre-fix).
    let while_let_src = r#"
let g = arange(0.0, 8.0).to(cuda)
let n = 2.0
let hits = 0.0
while let k = int(sum(lt_scalar(g, n)).item()):
    n = n - 1.0
    hits = hits + k
print(hits)
print("DONE")
"#;
    let (lb, stdout) = run_fixture(while_let_src, "whilelet");
    let val = stdout
        .lines()
        .take_while(|l| *l != "DONE")
        .last()
        .unwrap_or("");
    assert_eq!(val, "3", "while-let loop computed wrong value:\n{stdout}");
    assert_eq!(lb, 0, "while-let expression temps stranded:\n{stdout}");

    let for_src = r#"
let g = arange(0.0, 4.0).to(cuda)
let acc = 0.0
for i in range(int(sum(lt_scalar(g, 2.0)).item())):
    acc = acc + 1.0
print(acc)
print("DONE")
"#;
    let (lb, stdout) = run_fixture(for_src, "foriter");
    let val = stdout
        .lines()
        .take_while(|l| *l != "DONE")
        .last()
        .unwrap_or("");
    assert_eq!(val, "2", "for-iterable loop computed wrong value:\n{stdout}");
    assert_eq!(lb, 0, "for-iterable temps stranded:\n{stdout}");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn nested_builtin_args_do_not_strand_per_call() {
    let (lb1, s1) = run_fixture(&rounds_src(1), "n1");
    assert!(s1.contains("PARITY_OK"), "values diverged at N=1:\n{s1}");
    let (lb3, s3) = run_fixture(&rounds_src(3), "n3");
    assert!(s3.contains("PARITY_OK"), "values diverged at N=3:\n{s3}");
    assert_eq!(
        lb1, lb3,
        "nested-arg results strand {} block(s) per round",
        (lb3 - lb1) / 2
    );
}
