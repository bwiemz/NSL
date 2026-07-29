//! A user `fn` sharing a sampling builtin's name must dispatch to the
//! USER function (2026-07-29).
//!
//! The sampling dispatch arms in `compile_call` (`topk`, `multinomial`,
//! `manual_seed`, `argmax`, `cumsum`, `lt_scalar`) claimed the call by
//! NAME with no registry check, while the checker resolved the call to
//! the user fn — a miscompile: depending on signature shape it aborted at
//! runtime on a magic probe (the builtin returned a dict where the
//! checker promised a Tensor) or panicked cranelift-frontend outright
//! ("declared type of variable ... doesn't match"), both found by the
//! #427 review's whitelist-bypass probing. The arms now carry the same
//! `!registry.functions.contains_key(&func_name)` guard ~30 sibling arms
//! already had, so a user definition wins — matching the checker.
//!
//! CPU-only on purpose: dispatch is device-independent and this way the
//! gate runs in the plain workspace suite too.
//!
//! Scope: MODULE-LEVEL `fn`s only. A nested `fn` or a let-bound lambda
//! sharing a builtin name still loses to the builtin (registry.functions
//! never holds those names) — parity with the ~30 pre-guarded sibling
//! arms, not a regression; queued with the broader dispatch-arm audit.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn run_src(src: &str, tag: &str) -> String {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_shadow_{}_{tag}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--deterministic"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    assert!(
        out.status.success(),
        "[{tag}] run failed (the pre-guard failure mode is a runtime abort \
         or cranelift panic):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(stdout.contains("DONE"), "[{tag}] incomplete:\n{stdout}");
    std::fs::remove_dir_all(&tmp).ok();
    stdout
}

fn printed_value(stdout: &str) -> String {
    stdout
        .lines()
        .take_while(|l| *l != "DONE")
        .last()
        .unwrap_or("")
        .to_string()
}

/// The exact miscompile shape: user `topk` returns a Tensor; the builtin
/// returns a dict. Pre-guard this aborted on the magic probe in `sum`.
#[test]
fn user_fn_shadowing_topk_dispatches_to_the_user_fn() {
    let src = r#"
fn topk(x: Tensor, k: int) -> Tensor:
    return x

let x = arange(0.0, 8.0)
let r = topk(x, 3)
print(sum(r).item())
print("DONE")
"#;
    let stdout = run_src(src, "topk");
    assert_eq!(printed_value(&stdout), "28", "user topk did not win:\n{stdout}");
}

/// Scalar-returning shadow of `argmax` — the shape that reached the
/// cranelift verifier pre-guard (I64 fn result vs the arm's typing).
#[test]
fn user_fn_shadowing_argmax_dispatches_to_the_user_fn() {
    let src = r#"
fn argmax(x: Tensor, dim: int) -> float:
    return sum(x).item()

let x = arange(0.0, 4.0)
print(argmax(x, 0))
print("DONE")
"#;
    let stdout = run_src(src, "argmax");
    assert_eq!(printed_value(&stdout), "6", "user argmax did not win:\n{stdout}");
}

/// The guard must not disturb the unshadowed builtins.
#[test]
fn unshadowed_sampling_builtins_still_dispatch_to_the_runtime() {
    let src = r#"
let x = arange(0.0, 8.0)
let r = topk(x, 3)
let v = r["values"]
let c = cumsum(x, -1)
let a = argmax(x, -1)
print(sum(v).item() + sum(c).item() + sum(a).item())
print("DONE")
"#;
    let stdout = run_src(src, "builtin");
    // topk values 7+6+5 = 18; cumsum sum of prefix sums of 0..7 = 84;
    // argmax = 7. Total 109.
    assert_eq!(
        printed_value(&stdout),
        "109",
        "builtin dispatch changed:\n{stdout}"
    );
}
