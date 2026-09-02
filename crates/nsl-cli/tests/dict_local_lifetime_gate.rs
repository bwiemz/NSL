//! Regression gates for dict-local tensor lifetime (the aggregate-lifetime
//! gap, closed 2026-07-28).
//!
//! A `Dict<Str, Tensor>` local (today: `topk`'s result) owns its stored
//! tensors outright — every read CLONES (`compile_subscript`'s Dict arm) —
//! yet no codegen path outside DataLoader lowerings ever freed one. Every
//! `let r = topk(...)` stranded the dict plus both stored tensors: 2 VRAM
//! blocks per call on GPU inputs, ~79 MB/call resident heap in a CPU
//! `topk(x, 1M)` loop.
//!
//! The fix: `dict_lifetime::collect_sweepable_dict_locals` (a conservative
//! usage scan — single top-level `let` bound from a call, subscript READS
//! only, anything else vetoes) arms a `nsl_dict_free_tensor_values` pass in
//! `emit_return_local_sweep`. Two escape shapes turned out to be
//! structurally unreachable (probed 2026-07-28): returning a dict from a
//! user fn is a checker error ("wrong type", annotated or not), and passing
//! one to a user fn dies in codegen (Unknown-typed subscript lowers to list
//! access → verifier error). The veto still guards the reachable rest:
//! aliases, subscript writes, loop-local and conditional bindings.
//!
//! Safety direction matters here: a scan bug that ADMITS a bad shape is a
//! double-free/UAF, not a leak — which is why the alias/write gates assert
//! the run SUCCEEDS (a crash turns them red) on top of pinning the strand.

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
    let tmp = std::env::temp_dir().join(format!("nsl_dictlife_{}_{tag}", std::process::id()));
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
        "[{tag}] run failed. If stderr says 'CUDA support not compiled', \
         this suite was built without `--features cuda` — an environment \
         problem. Otherwise, for the veto gates, this is the DOUBLE-FREE \
         failure direction:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stdout.contains("DONE"),
        "[{tag}] fixture did not complete:\n{stdout}"
    );
    let live_blocks = stderr
        .lines()
        .filter_map(|l| {
            l.split("live_blocks=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .next_back()
        .unwrap_or_else(|| panic!("[{tag}] no [gpu-mem] live_blocks report:\n{stderr}"));
    std::fs::remove_dir_all(&tmp).ok();
    (live_blocks, stdout)
}

fn expect_value(stdout: &str, expected: &str, tag: &str) {
    let val = stdout
        .lines()
        .take_while(|l| *l != "DONE")
        .last()
        .unwrap_or("");
    assert_eq!(val, expected, "[{tag}] wrong computed value:\n{stdout}");
}

/// `load_safetensors` — the second fresh-dict builtin (whitelisted
/// 2026-07-29 with this gate; the #427 review verified its dict stores
/// fresh solely-owned tensors and reads clone, but the entry waited for
/// coverage). A weights dict loaded to GPU and only subscript-read must
/// be swept at exit; before the whitelist entry its dict + both tensors
/// stranded (red-proven at lb=2).
#[test]
#[ignore = "requires CUDA GPU"]
fn load_safetensors_dict_is_swept_at_exit() {
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;
    use std::collections::HashMap;

    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_dictlife_{}_sft", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let a_bytes: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let b_bytes: Vec<u8> = [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    views.insert(
        "a".to_string(),
        TensorView::new(Dtype::F32, vec![4], a_bytes.as_slice()).unwrap(),
    );
    views.insert(
        "b".to_string(),
        TensorView::new(Dtype::F32, vec![2], b_bytes.as_slice()).unwrap(),
    );
    std::fs::write(tmp.join("w.safetensors"), serialize(&views, None).unwrap()).unwrap();

    let src = r#"
let w = load_safetensors("w.safetensors", 1)
let a = w["a"]
let b = w["b"]
print(sum(a).item() + sum(b).item())
print("DONE")
"#;
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
        "[sft] run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(stdout.contains("DONE"), "[sft] incomplete:\n{stdout}");
    expect_value(&stdout, "40", "sft");
    let lb = stderr
        .lines()
        .filter_map(|l| {
            l.split("live_blocks=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .next_back()
        .unwrap_or_else(|| panic!("[sft] no live_blocks report:\n{stderr}"));
    std::fs::remove_dir_all(&tmp).ok();
    assert_eq!(lb, 0, "load_safetensors dict stranded:\n{stdout}");
}

/// Script-scope dict locals are swept at main's exit — the shape the
/// original 2-blocks-per-call measurement used.
#[test]
#[ignore = "requires CUDA GPU"]
fn top_level_dict_local_is_swept_at_main_exit() {
    for calls in [1_usize, 3] {
        let mut src = String::from("let x = arange(0.0, 8.0).to(cuda)\n");
        for i in 0..calls {
            src.push_str(&format!(
                "let r{i} = topk(x, 3)\nlet v{i} = r{i}[\"values\"]\nlet s{i} = sum(v{i}).item()\n"
            ));
        }
        src.push_str("print(\"DONE\")\n");
        let (lb, _) = run_fixture(&src, &format!("toplevel{calls}"));
        assert_eq!(lb, 0, "top-level dict locals stranded at calls={calls}");
    }
}

/// `return r["values"]` — the subscript read clones BEFORE the sweep frees
/// the dict, so the caller gets an independent tensor and nothing strands.
/// This is the allowed-shape/soundness pair for the sweep ordering.
#[test]
#[ignore = "requires CUDA GPU"]
fn returned_subscript_read_survives_the_dict_sweep() {
    let src = r#"
fn best(x: Tensor) -> Tensor:
    let r = topk(x, 3)
    return r["values"]

let x = arange(0.0, 8.0).to(cuda)
let v = best(x)
print(sum(v).item())
print("DONE")
"#;
    let (lb, stdout) = run_fixture(src, "retval");
    expect_value(&stdout, "18", "retval");
    assert_eq!(lb, 0, "returned-subscript shape stranded:\n{stdout}");
}

/// `let r2 = r` — the bare-identifier use vetoes BOTH the sweep's claim on
/// the symbol; the dict strands exactly as before the fix (2 blocks) and,
/// critically, the program must not double-free. Removing the scan's Ident
/// veto makes both locals swept — freeing the same dict twice — and this
/// gate's run-success assertion catches the crash.
#[test]
#[ignore = "requires CUDA GPU"]
fn alias_vetoed_dict_strands_but_never_double_frees() {
    let src = r#"
fn f(x: Tensor) -> float:
    let r = topk(x, 3)
    let r2 = r
    let v = r2["values"]
    return sum(v).item()

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let (lb, stdout) = run_fixture(src, "alias");
    expect_value(&stdout, "18", "alias");
    assert_eq!(
        lb, 2,
        "alias-vetoed dict strand changed (2 = the vetoed dict's values+indices):\n{stdout}"
    );
}

/// `r["values"] = t` — a subscript WRITE stores a handle the dict does not
/// own an extra reference to; sweeping would free the caller's tensor
/// through the dict. The write vetoes the sweep; the strand is the status
/// quo and the stored value must remain readable.
#[test]
#[ignore = "requires CUDA GPU"]
fn subscript_write_vetoes_the_sweep() {
    let src = r#"
fn f(x: Tensor) -> float:
    let r = topk(x, 3)
    let t = x * 2.0
    r["values"] = t
    let v = r["values"]
    return sum(v).item()

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let (lb, stdout) = run_fixture(src, "write");
    expect_value(&stdout, "56", "write");
    assert_eq!(lb, 2, "subscript-write strand changed:\n{stdout}");
}

/// Review HIGH-1 on d114b5d7 (was reproduced memory corruption): a
/// loop-body dict decl SHADOWING a function parameter is checker-legal
/// (the body is a child scope) and invisible to the scan, which only
/// walks the body. The predeclare skips the sym (the param's slot already
/// exists) — so a rebind clear keyed on the plan set alone freed the
/// CALLER'S TENSOR as a dict (`free_dict_impl` walked the tensor magic
/// bytes as a bucket pointer). The clear now keys off
/// `dict_loop_predeclared` — only slots the predeclare actually created —
/// and this shape reverts to the status-quo leak: correct value, run
/// success, dicts stranded.
#[test]
#[ignore = "requires CUDA GPU"]
fn param_shadowing_loop_dict_does_not_free_the_callers_tensor() {
    let src = r#"
fn f(w: Tensor, x: Tensor) -> float:
    let acc = 0.0
    for i in range(3):
        let w = topk(x, 3)
        let v = w["values"]
        acc = acc + sum(v).item()
    return acc

let a = arange(0.0, 4.0).to(cuda)
let x = arange(0.0, 8.0).to(cuda)
print(f(a, x))
print("DONE")
"#;
    let (lb, stdout) = run_fixture(src, "paramshadow");
    expect_value(&stdout, "54", "paramshadow");
    // 3 stranded per-iteration dicts x 2 tensors = the status-quo leak
    // this shape keeps (fail-closed direction).
    assert_eq!(lb, 6, "param-shadow strand changed:\n{stdout}");
}

/// Review HIGH-1 on 7a0f83d9 (was a confirmed abort): a list-comprehension
/// generator re-using a candidate's NAME re-points the symbol's variable
/// slot at the loop element — the sweep then handed an integer to
/// `nsl_dict_free_tensor_values`. The blanket pattern veto must keep the
/// candidate out; the dict strands (status quo) and the program completes.
#[test]
#[ignore = "requires CUDA GPU"]
fn comprehension_generator_shadowing_vetoes_the_sweep() {
    let src = r#"
fn f(x: Tensor) -> float:
    let r = topk(x, 3)
    let v = r["values"]
    let ys = [0 for r in [1, 2, 3]]
    return sum(v).item()

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let (lb, stdout) = run_fixture(src, "compshadow");
    expect_value(&stdout, "18", "compshadow");
    assert_eq!(lb, 2, "comprehension-shadow strand changed:\n{stdout}");
}

/// Review HIGH-2 on 7a0f83d9 (was a confirmed abort): a lambda returning a
/// captured dict types its call result Dict<Str, Tensor>, so admitting
/// arbitrary Dict-typed calls minted TWO owning candidates for ONE dict —
/// a double free. Pass 1's FRESH_DICT_BUILTINS callee whitelist keeps
/// lambda results out entirely; the dict (vetoed via the lambda capture)
/// strands as before and the program completes.
#[test]
#[ignore = "requires CUDA GPU"]
fn lambda_returned_dict_aliases_do_not_double_free() {
    let src = r#"
fn f(x: Tensor) -> float:
    let d = topk(x, 3)
    let g = |y| d
    let r1 = g(0)
    let r2 = g(0)
    let v = r1["values"]
    return sum(v).item()

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let (lb, stdout) = run_fixture(src, "lambdaalias");
    expect_value(&stdout, "18", "lambdaalias");
    assert_eq!(lb, 2, "lambda-alias strand changed:\n{stdout}");
}

/// Loop-local dict bindings, RATCHETED TO ZERO (the eltls clear twin for
/// dicts, 2026-07-29). A dict `let` in the DIRECT body of a top-level
/// `for`/`while` is scan-admitted into the loop-rebind pool: the loop
/// lowering zero-predeclares its slot, `compile_var_decl` frees the
/// previous iteration's dict before each rebind, and the return sweep
/// frees the last one. Was pinned at 2 blocks/iteration
/// (`loop_local_dicts_keep_status_quo`).
///
/// Covers the for-loop, while-loop, break (early exit leaves the last
/// dict to the return sweep), zero-iteration (the predeclared slot is 0 —
/// `free_dict_impl` must no-op), and script-scope (main) shapes. A
/// subscript read AFTER the loop is a checker error ("not found in
/// scope"), so the scan's outside-the-loop mention veto is unreachable
/// belt-and-braces today.
#[test]
#[ignore = "requires CUDA GPU"]
fn loop_local_dicts_are_cleared_per_rebind() {
    let for_src = r#"
fn f(x: Tensor) -> float:
    let acc = 0.0
    for i in range(3):
        let r = topk(x, 3)
        let v = r["values"]
        acc = acc + sum(v).item()
    return acc

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let while_src = r#"
fn f(x: Tensor) -> float:
    let acc = 0.0
    let i = 0
    while i < 3:
        let r = topk(x, 3)
        let v = r["values"]
        acc = acc + sum(v).item()
        i = i + 1
    return acc

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let break_src = r#"
fn f(x: Tensor) -> float:
    let acc = 0.0
    for i in range(5):
        let r = topk(x, 3)
        let v = r["values"]
        acc = acc + sum(v).item()
        if i == 1:
            break
    return acc

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let zero_src = r#"
fn f(x: Tensor) -> float:
    let acc = 0.0
    for i in range(0):
        let r = topk(x, 3)
        let v = r["values"]
        acc = acc + sum(v).item()
    return acc

let x = arange(0.0, 8.0).to(cuda)
print(f(x))
print("DONE")
"#;
    let main_src = r#"
let x = arange(0.0, 8.0).to(cuda)
let acc = 0.0
for i in range(3):
    let r = topk(x, 3)
    let v = r["values"]
    acc = acc + sum(v).item()
print(acc)
print("DONE")
"#;
    for (src, expected, tag) in [
        (for_src, "54", "loopfor"),
        (while_src, "54", "loopwhile"),
        (break_src, "36", "loopbreak"),
        (zero_src, "0", "loopzero"),
        (main_src, "54", "loopmain"),
    ] {
        let (lb, stdout) = run_fixture(src, tag);
        expect_value(&stdout, expected, tag);
        assert_eq!(lb, 0, "[{tag}] loop-local dict strand is back:\n{stdout}");
    }
}
