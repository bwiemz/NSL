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
//! 2026-07-29: the nested-fn/lambda half is closed too. Nested `fn`s
//! are REMOVED from `registry.functions` right after their body
//! compiles (StmtKind::FnDef restores the previous entry) and lambdas
//! never enter it, so the registry guards could not see them — a nested
//! `fn topk` still hit the builtin arm. `compile_call` now routes any
//! LOCAL binding whose checker type is `Type::Function` straight to
//! `compile_indirect_call` before every builtin arm; non-Function
//! locals (a tensor named `sum`) keep builtin dispatch exactly as
//! before. Module-level arms WITHOUT a registry guard (e.g.
//! `reduce_max`) remain a documented gap for the broader dispatch-arm
//! audit — the new route only covers variable-bound function values.

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

/// The bugs.md 2026-07-28 ICE shape: a NESTED `fn topk` — invisible to
/// the registry guards because FnDef removes it after compiling. The
/// Function-typed-binding route dispatches it like any local fn value.
#[test]
fn nested_fn_shadowing_topk_dispatches_to_the_nested_fn() {
    let src = r#"
fn outer(x: Tensor) -> float:
    fn topk(t: Tensor, k: int) -> Tensor:
        return t
    let r = topk(x, 3)
    return sum(r).item()

let x = arange(0.0, 8.0)
print(outer(x))
print("DONE")
"#;
    let stdout = run_src(src, "nestedtopk");
    assert_eq!(
        printed_value(&stdout),
        "28",
        "nested topk did not win:\n{stdout}"
    );
}

/// A let-bound lambda sharing a builtin arm's name. Pre-route the
/// `cumsum` arm claimed the call and rejected it at compile time
/// (wrong arity for the builtin the user never meant to call).
///
/// Int-typed on purpose so this gate pins DISPATCH alone. The float
/// half (the lambda-signature ABI mismatch this fixture originally
/// dodged) is fixed and pinned separately in lambda_float_abi.rs.
#[test]
fn lambda_shadowing_cumsum_dispatches_to_the_lambda() {
    let src = r#"
let cumsum = |v: int| v * 2
print(cumsum(3))
print("DONE")
"#;
    let stdout = run_src(src, "lambdacumsum");
    assert_eq!(
        printed_value(&stdout),
        "6",
        "lambda cumsum did not win:\n{stdout}"
    );
}

/// A fn-typed PARAMETER sharing a builtin arm's name — same class, via
/// the parameter binding instead of a let. Int-typed for the same
/// reason as the lambda test above (float ABI pinned in
/// lambda_float_abi.rs).
#[test]
fn fn_typed_param_shadowing_lt_scalar_dispatches_indirect() {
    let src = r#"
fn apply(lt_scalar: (int) -> int, v: int) -> int:
    return lt_scalar(v)

let f = |z: int| z + 1
print(apply(f, 4))
print("DONE")
"#;
    let stdout = run_src(src, "paramshadow");
    assert_eq!(
        printed_value(&stdout),
        "5",
        "fn-typed param did not win:\n{stdout}"
    );
}

/// Review HIGH on 371ee02f: `state.variables` is function-flat while
/// the checker is block-scoped — a nested fn declared in an if-arm is
/// out of scope after the arm, and a later call to the SAME name
/// resolves to the builtin. The first cut of the guard trusted the flat
/// map and rerouted that call into a dead (undefined-on-path) function
/// pointer — a crash. `live_fn_bindings` now mirrors the checker's
/// scoping; these four fixtures are the reviewer's runtime-confirmed
/// repros, value-pinned.
#[test]
fn builtin_call_after_dead_arm_shadow_uses_the_builtin() {
    let src = r#"
let flag = 0
if flag > 0:
    fn sum(a: int) -> int:
        return a + 1
    print(sum(1))
let t = ones([4])
let s = sum(t)
print(s.item())
print("DONE")
"#;
    let stdout = run_src(src, "deadarm");
    assert_eq!(
        printed_value(&stdout),
        "4",
        "post-arm builtin sum broke:\n{stdout}"
    );
}

#[test]
fn builtin_call_after_live_arm_shadow_uses_the_builtin() {
    let src = r#"
let flag = 1
if flag > 0:
    fn sum(a: int) -> int:
        return a + 1
    print(sum(1))
let t = ones([4])
let s = sum(t)
print(s.item())
print("DONE")
"#;
    let stdout = run_src(src, "livearm");
    assert!(
        stdout.contains("2\n"),
        "in-arm nested sum did not win:\n{stdout}"
    );
    assert_eq!(
        printed_value(&stdout),
        "4",
        "post-arm builtin sum broke after a LIVE arm:\n{stdout}"
    );
}

#[test]
fn builtin_call_after_dead_arm_lambda_uses_the_builtin() {
    let src = r#"
let flag = 0
if flag > 0:
    let cumsum = |v: int| v * 2
    print(cumsum(3))
let x = arange(0.0, 4.0)
let c = cumsum(x, -1)
print(sum(c).item())
print("DONE")
"#;
    let stdout = run_src(src, "deadlambda");
    // cumsum of 0..3 = [0,1,3,6]; sum = 10.
    assert_eq!(
        printed_value(&stdout),
        "10",
        "post-arm builtin cumsum broke:\n{stdout}"
    );
}

#[test]
fn module_fn_call_after_dead_arm_shadow_uses_the_module_fn() {
    let src = r#"
fn helper(x: int) -> int:
    return x * 10

let flag = 0
if flag > 0:
    fn helper(x: int) -> int:
        return x + 1
    print(helper(1))
print(helper(5))
print("DONE")
"#;
    let stdout = run_src(src, "deadmodfn");
    assert_eq!(
        printed_value(&stdout),
        "50",
        "post-arm module fn broke:\n{stdout}"
    );
}

/// Grad-block bodies are checker-scoped but compile in the same
/// FuncState with no variables restore — the review's residual-gap
/// repro (MEDIUM on 682641ca): without a fn-binding scope around the
/// grad body, the post-block builtin `sum(w)` rerouted into the grad
/// body's dead nested fn (misaligned-deref abort).
#[test]
fn builtin_call_after_grad_block_shadow_uses_the_builtin() {
    let src = r#"
let w = ones([4])
let (loss, grads) = grad(w):
    fn sum(a: int) -> int:
        return a + 1
    mean(w * w)
print(loss.item())
let s = sum(w)
print(s.item())
print("DONE")
"#;
    let stdout = run_src(src, "gradshadow");
    assert!(
        stdout.contains("1\n"),
        "grad loss wrong (mean of ones = 1):\n{stdout}"
    );
    assert_eq!(
        printed_value(&stdout),
        "4",
        "post-grad-block builtin sum broke:\n{stdout}"
    );
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
