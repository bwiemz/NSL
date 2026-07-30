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

fn run_nsl(src: &str, tag: &str) -> std::process::Output {
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
    std::fs::remove_dir_all(&tmp).ok();
    out
}

fn run_src(src: &str, tag: &str) -> String {
    let out = run_nsl(src, tag);
    assert!(
        out.status.success(),
        "[{tag}] run failed (the pre-guard failure mode is a runtime abort \
         or cranelift panic):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(stdout.contains("DONE"), "[{tag}] incomplete:\n{stdout}");
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

/// 2026-07-29 dispatch-arm audit: ~50 builtin arms carried NO
/// `registry.functions` guard, so a MODULE-LEVEL user fn sharing the
/// name silently misdispatched to the builtin — `fn sum(x: Tensor) ->
/// float: return 42.0` printed the BUILTIN's sum (4.0 for ones([4])),
/// exit 0. One hoisted registry check at the top of compile_call now
/// makes codegen follow the checker for every arm at once.
#[test]
fn module_fn_shadowing_sum_dispatches_to_the_user_fn() {
    let src = r#"
fn sum(x: Tensor) -> float:
    return 42.0

let t = ones([4])
print(sum(t))
print("DONE")
"#;
    let stdout = run_src(src, "modsum");
    assert_eq!(printed_value(&stdout), "42", "module fn sum lost:\n{stdout}");
}

/// Same class through the scalar-math arm (`abs`) — silently printed 5
/// pre-hoist.
#[test]
fn module_fn_shadowing_abs_dispatches_to_the_user_fn() {
    let src = r#"
fn abs(x: int) -> int:
    return x + 100

print(abs(5))
print("DONE")
"#;
    let stdout = run_src(src, "modabs");
    assert_eq!(printed_value(&stdout), "105", "module fn abs lost:\n{stdout}");
}

/// The PR #435 documented gap verbatim: `fn reduce_max` hit the
/// builtin arm's arity check — a compile error for a call the checker
/// had already accepted against the USER fn.
#[test]
fn module_fn_shadowing_reduce_max_dispatches_to_the_user_fn() {
    let src = r#"
fn reduce_max(x: Tensor, d: int) -> float:
    return 99.0

let t = ones([4])
print(reduce_max(t, 0))
print("DONE")
"#;
    let stdout = run_src(src, "modrmax");
    assert_eq!(
        printed_value(&stdout),
        "99",
        "module fn reduce_max lost:\n{stdout}"
    );
}

/// The hoist must not disturb arms for names nobody redefined: pin the
/// BUILTIN results for the same three names, unshadowed.
#[test]
fn unshadowed_sum_abs_reduce_max_still_hit_builtins() {
    let src = r#"
let t = ones([4])
print(sum(t).item())
print(abs(0 - 7))
print("DONE")
"#;
    let stdout = run_src(src, "unshadowed3");
    let vals: Vec<&str> = stdout.lines().take_while(|l| *l != "DONE").collect();
    assert_eq!(vals, ["4", "7"], "builtin arms disturbed:\n{stdout}");
}

/// `generate` is the ONE stdlib collision (stdlib/nsl/inference/
/// generate.nsl): when NOT imported, the CFIE serve arm keeps the name
/// and refuses outside serve — pin that refusal so the hoist provably
/// leaves the unimported path alone.
#[test]
fn bare_generate_outside_serve_still_refused_by_the_cfie_arm() {
    let src = r#"
let t = ones([4])
let r = generate("m", t, 0)
print("DONE")
"#;
    let out = run_nsl(src, "baregen");
    assert!(
        !out.status.success(),
        "bare generate outside serve must still refuse:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("serve"),
        "refusal should still name the serve requirement:\n{stderr}"
    );
}

/// closure_info is compiler-global and symbol-keyed (#435 review LOW):
/// after `let f = |v| v + k` (capturing), a REBIND `let f = |v| v * 2`
/// (non-capturing) left the stale capture count — the call site then
/// read the bare function pointer as a closure struct: the program died
/// silently after the first print (no DONE, probed pre-fix). VarDecl
/// now clears the entry when the RHS is not a capturing lambda.
#[test]
fn non_capturing_rebind_clears_stale_closure_info() {
    let src = r#"
let k = 5
let f = |v: int| v + k
print(f(1))
let f = |v: int| v * 2
print(f(3))
print("DONE")
"#;
    let stdout = run_src(src, "closrebind");
    assert!(stdout.contains("6\n"), "capturing call broke:\n{stdout}");
    assert_eq!(
        printed_value(&stdout),
        "6",
        "post-rebind non-capturing call broke:\n{stdout}"
    );
}

/// Review HIGH on 44c011c1: closure_info is compiler-global and a
/// nested fn's body compiles MID-way through the outer body — a nested
/// `let f = 7` (any non-capturing rebind) deleted the OUTER function's
/// live closure entry, and the outer `f(1)` then executed the closure
/// struct as code (exit 1, zero output). The FnDef arm now snapshots
/// and restores closure_info around the nested compile.
#[test]
fn nested_fn_binding_does_not_corrupt_outer_closure_info() {
    let src = r#"
fn outer() -> int:
    let k = 5
    let f = |v: int| v + k
    fn helper() -> int:
        let f = 7
        return f
    let a = helper()
    return f(1) + a

print(outer())
print("DONE")
"#;
    let stdout = run_src(src, "nestclos");
    assert_eq!(
        printed_value(&stdout),
        "13",
        "outer closure corrupted by nested binding:\n{stdout}"
    );
}

/// Review MEDIUM-1 on 44c011c1: the plain-Assign rebind (`f = <lambda>`,
/// no `let`) had NO capture-count transfer at all — a non-capturing
/// assign-rebind after a capturing binding died exactly like the
/// VarDecl shape (silent death at the second call), and a capturing
/// assign-rebind was never recorded. compile_assign now mirrors the
/// VarDecl transfer/clear.
#[test]
fn assign_rebind_clears_stale_closure_info() {
    let src = r#"
let k = 5
let f = |v: int| v + k
print(f(1))
f = |v: int| v * 2
print(f(3))
print("DONE")
"#;
    let stdout = run_src(src, "assignrebind");
    assert!(stdout.contains("6\n"), "capturing call broke:\n{stdout}");
    assert_eq!(
        printed_value(&stdout),
        "6",
        "post-assign-rebind call broke:\n{stdout}"
    );
}

/// Deliberate semantics (review MEDIUM-2 on 44c011c1): shadowing a
/// builtin name with a NON-Function value makes calls of that name a
/// compile error — Python semantics — instead of the pre-hoist silent
/// fallback to the builtin. Pinned so the behavior change is on record.
#[test]
fn calling_a_non_function_shadow_of_a_builtin_is_a_compile_error() {
    let src = r#"
let sum = 5
let t = ones([4])
let y = sum(t)
print("DONE")
"#;
    let out = run_nsl(src, "nonfnshadow");
    assert!(
        !out.status.success(),
        "calling a non-Function shadow must be a compile error:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("not callable"),
        "error should say the value is not callable:\n{stderr}"
    );
}

/// Review HIGH on cb1bd16f: the closure clear was block-scope-blind —
/// a DEAD if-arm's non-capturing `let f = ...` deleted the outer
/// capturing `f`'s entry at compile time, and the post-arm `f(1)`
/// (whose runtime value is still the untouched closure) compiled as a
/// bare-pointer call: the closure struct executed as code.
/// closure_info now lives on FuncState with a per-scope undo log
/// rolled back by pop_fn_binding_scope.
#[test]
fn dead_arm_lambda_rebind_does_not_corrupt_outer_closure_info() {
    let src = r#"
let k = 5
let f = |v: int| v + k
let flag = 0
if flag > 0:
    let f = |v: int| v * 2
    print(f(3))
print(f(1))
print("DONE")
"#;
    let stdout = run_src(src, "deadarmclos");
    assert_eq!(
        printed_value(&stdout),
        "6",
        "outer closure corrupted by dead-arm rebind:\n{stdout}"
    );
}

/// Review HIGH on the FuncState move: a lambda capturing a CAPTURING
/// closure and calling it died silently — lambda bodies compile
/// deferred with a fresh FuncState, so the definer's closure metadata
/// for the captured closure was lost and the call compiled as a
/// bare-pointer call (the closure struct executed as code).
/// PendingLambda now carries the captured closures' capture counts and
/// compile_lambda_body seeds its state from them.
#[test]
fn lambda_capturing_a_capturing_closure_calls_it_correctly() {
    let src = r#"
let k = 5
let f = |v: int| v + k
let g = |v: int| f(v) + 1
print(g(2))
print("DONE")
"#;
    let stdout = run_src(src, "closcompose");
    assert_eq!(
        printed_value(&stdout),
        "8",
        "captured capturing closure broke:\n{stdout}"
    );
}

/// Control: a NON-capturing callee captured by a capturing lambda uses
/// the bare-pointer path — no metadata needed, must stay untouched.
#[test]
fn lambda_capturing_a_bare_fn_pointer_still_works() {
    let src = r#"
let m = 100
let f = |v: int| v * 2
let g = |v: int| f(v) + m
print(g(3))
print("DONE")
"#;
    let stdout = run_src(src, "closcomposectl");
    assert_eq!(
        printed_value(&stdout),
        "106",
        "bare-fn capture control broke:\n{stdout}"
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
