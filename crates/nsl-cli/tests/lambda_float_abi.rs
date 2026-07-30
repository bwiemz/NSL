//! Indirect calls must not miscompile FLOAT arguments, returns, or
//! captures (2026-07-29, bugs.md entry from the #435 session).
//!
//! Root cause: `compile_lambda` hardcoded every parameter to I64 in the
//! lambda's compiled signature, while `compile_indirect_call` builds the
//! call-site signature from the checker's `Type::Function` — F64 for
//! float. Under SysV the caller put the argument in an XMM register and
//! the callee read an integer register: `(|v: float| v * 2.0)(3.0)`
//! returned 4 (v arrived as leftover integer-register contents, then the
//! value-type-keyed int→float promotion in compile_binary_op laundered
//! it into a plausible float). Silent wrong values, exit 0. Int lambdas
//! agreed by accident (both sides I64).
//!
//! Captures had the mirror bug: the lambda's signature declared capture
//! params with the captured variable's REAL cl_type (F64 for a float
//! capture) while the closure call site declares and loads every capture
//! slot as I64. The fix standardizes capture slots to I64 bit-patterns
//! on both sides, converting at the closure-construction store and the
//! lambda-entry binding.
//!
//! map()/filter() are the one consumer with a FIXED runtime ABI —
//! `nsl_map`/`nsl_filter` invoke the pointer as `extern "C" fn(i64) ->
//! i64` (hof.rs). Float params/returns cannot survive that ABI, and
//! before this fix they produced garbage silently (the return was read
//! from RAX while the lambda wrote XMM0). They are now REFUSED at
//! compile time; int-typed functions keep working unchanged.
//!
//! CPU-only on purpose: the ABI is device-independent and this way the
//! gates run in the plain workspace suite.

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
    let tmp = std::env::temp_dir().join(format!("nsl_lfabi_{}_{tag}", std::process::id()));
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
        "[{tag}] run failed:\n{}",
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

/// The bugs.md repro verbatim: pre-fix this printed 4.
#[test]
fn float_lambda_argument_arrives_intact() {
    let src = r#"
let f = |v: float| v * 2.0
print(f(3.0))
print("DONE")
"#;
    let stdout = run_src(src, "mul");
    assert_eq!(printed_value(&stdout), "6", "float arg lost:\n{stdout}");
}

/// The second bugs.md shape: pre-fix this printed 3.
#[test]
fn float_lambda_addition_argument_arrives_intact() {
    let src = r#"
let f = |v: float| v + 1.0
print(f(4.0))
print("DONE")
"#;
    let stdout = run_src(src, "add");
    assert_eq!(printed_value(&stdout), "5", "float arg lost:\n{stdout}");
}

/// Same ABI through a fn-typed parameter — the callee compiles the
/// indirect call from the ANNOTATION, the lambda from its own inferred
/// type; both must lower float identically.
#[test]
fn float_lambda_via_fn_typed_param_arrives_intact() {
    let src = r#"
fn apply(g: (float) -> float, v: float) -> float:
    return g(v)

let f = |z: float| z * 3.0
print(apply(f, 2.0))
print("DONE")
"#;
    let stdout = run_src(src, "fnparam");
    assert_eq!(printed_value(&stdout), "6", "float via param lost:\n{stdout}");
}

/// Mixed float/int params: the signature must type each position
/// independently (F64, I64) — an all-I64 fallback would corrupt only
/// the float slot, and an all-F64 one only the int slot.
#[test]
fn mixed_float_int_params_arrive_intact() {
    let src = r#"
let f = |a: float, b: int| a * float(b)
print(f(2.5, 4))
print("DONE")
"#;
    let stdout = run_src(src, "mixed");
    assert_eq!(printed_value(&stdout), "10", "mixed params lost:\n{stdout}");
}

/// Float CAPTURE: the closure call site loads every capture slot as I64
/// while the lambda declared the capture param F64 pre-fix — the value
/// went through the wrong register class. Both sides now use I64 slots
/// with bitcasts at the boundaries.
#[test]
fn float_capture_arrives_intact() {
    let src = r#"
let k = 2.5
let f = |v: float| v * k
print(f(2.0))
print("DONE")
"#;
    let stdout = run_src(src, "fcap");
    assert_eq!(printed_value(&stdout), "5", "float capture lost:\n{stdout}");
}

/// Int captures worked pre-fix (both sides I64 by accident) and must
/// keep working through the new explicit slot convention.
#[test]
fn int_capture_still_arrives_intact() {
    let src = r#"
let n = 3
let f = |v: int| v * n
print(f(2))
print("DONE")
"#;
    let stdout = run_src(src, "icap");
    assert_eq!(printed_value(&stdout), "6", "int capture broke:\n{stdout}");
}

/// Bool capture: an I8-typed variable crossing the I64 slot — exercises
/// the narrow-int widen/narrow pair at the closure boundaries.
#[test]
fn bool_capture_arrives_intact() {
    let src = r#"
let flag = true
let f = |v: int| v + int(flag)
print(f(2))
print("DONE")
"#;
    let stdout = run_src(src, "bcap");
    assert_eq!(printed_value(&stdout), "3", "bool capture lost:\n{stdout}");
}

/// Nested `fn`s compile with their declared (real-typed) signatures and
/// reach the call through the #435 indirect route — this pins that the
/// two sides agree for floats there too.
#[test]
fn nested_fn_float_param_via_indirect_route() {
    let src = r#"
fn outer() -> float:
    fn scale(v: float) -> float:
        return v * 2.0
    return scale(3.0)

print(outer())
print("DONE")
"#;
    let stdout = run_src(src, "nested");
    assert_eq!(printed_value(&stdout), "6", "nested fn float lost:\n{stdout}");
}

/// map() with an int lambda is the supported hof shape — unchanged.
/// (Elements are consumed via `for` — the checker types map results as
/// List(Unknown), and subscript arithmetic on Unknown elements misroutes
/// into tensor ops; iteration is the established idiom, m5_features.nsl.)
#[test]
fn map_int_lambda_still_works() {
    let src = r#"
let r = map(|x: int| x * 2, [1, 2, 3])
for v in r:
    print(v)
print("DONE")
"#;
    let stdout = run_src(src, "mapint");
    let vals: Vec<&str> = stdout.lines().take_while(|l| *l != "DONE").collect();
    assert_eq!(vals, ["2", "4", "6"], "int map broke:\n{stdout}");
}

/// filter() with an int predicate — unchanged.
#[test]
fn filter_int_predicate_still_works() {
    let src = r#"
let r = filter(|x: int| x > 1, [1, 2, 3])
for v in r:
    print(v)
print("DONE")
"#;
    let stdout = run_src(src, "filtint");
    let vals: Vec<&str> = stdout.lines().take_while(|l| *l != "DONE").collect();
    assert_eq!(vals, ["2", "3"], "int filter broke:\n{stdout}");
}

/// A float PARAMETER cannot survive nsl_map's fn(i64)->i64 ABI — the
/// pre-fix behavior was silent garbage; it must now refuse to compile.
/// (The fixtures deliberately do NOT read elements back: with the guard
/// mutated away the program must run to a clean exit so the !success
/// assertion goes red, rather than crashing on an unrelated path.)
#[test]
fn map_float_param_lambda_is_refused_at_compile_time() {
    let src = r#"
let r = map(|x: float| x * 2.0, [1.0, 2.0])
print("DONE")
"#;
    let out = run_nsl(src, "mapfloat");
    assert!(
        !out.status.success(),
        "float-param map must be refused:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("int-typed"),
        "refusal should explain the int-only ABI:\n{stderr}"
    );
}

/// A float RETURN is equally unrepresentable in the hof ABI (the runtime
/// reads the integer return register) — refuse, don't corrupt.
#[test]
fn map_float_return_lambda_is_refused_at_compile_time() {
    let src = r#"
let r = map(|x: int| float(x), [1, 2])
print("DONE")
"#;
    let out = run_nsl(src, "mapfret");
    assert!(
        !out.status.success(),
        "float-return map must be refused:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("int-typed"),
        "refusal should explain the int-only ABI:\n{stderr}"
    );
}

/// filter() with a float-typed predicate parameter — same refusal.
#[test]
fn filter_float_param_is_refused_at_compile_time() {
    let src = r#"
let r = filter(|x: float| x > 1.0, [1.0, 2.0])
print("DONE")
"#;
    let out = run_nsl(src, "filtfloat");
    assert!(
        !out.status.success(),
        "float-param filter must be refused:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("int-typed"),
        "refusal should explain the int-only ABI:\n{stderr}"
    );
}
