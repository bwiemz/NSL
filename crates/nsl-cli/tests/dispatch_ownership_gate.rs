//! ELTLS v2a gates: dispatch-boundary ownership registration
//! (2026-07-29).
//!
//! The recurring leak class of the ownership campaign was a dispatch arm
//! whose NSL-level name was missing from the hand-maintained owning-ref
//! allowlist (`expr_result_is_owned_temporary`): the arm's fresh result
//! was never tracked, so it stranded one block per call in nested,
//! receiver, and statement position — silently, because a missing entry
//! costs a leak, never a crash. Three separate cycles (#423 sdpa, #424
//! rmsnorm/dropout, #426 lt_scalar/multinomial) each added names by hand.
//!
//! v2a closes the class at its root for every arm whose terminal call is
//! a table-classified FFI: `compile_call_by_name` records its last
//! non-void emission, and the `compile_expr` Call arm registers the
//! dispatch result as Owned when it is PROVABLY that emission's fresh
//! output (counter-advanced-during-dispatch + value identity — see
//! `register_dispatch_result_ownership`). The nested tracker accepts the
//! per-statement `dispatch_fresh` set as an owning signal, so `reduce_max`
//! and `clamp` — table-classified terminals with NO allowlist entry —
//! stop stranding without anybody hand-listing them.
//!
//! What these gates pin:
//!   - `unallowlisted_dispatch_results_do_not_strand`: reduce_max/clamp
//!     in receiver, nested, and bare-statement position leave
//!     live_blocks == 0 at exit, at two different round counts (a
//!     constant-offset bug can't hide behind one count), with exact
//!     value parity against the bound spellings.
//!   - `identity_shaped_dispatch_does_not_steal_bindings`: the
//!     per-statement clearing of `dispatch_fresh` is a DEFENSIVE
//!     invariant. An identity-shaped arm (`copy_data` returns its first
//!     input) hands back the SAME Cranelift Value that a PREVIOUS
//!     statement's dispatch registered as fresh (SSA use_var in one
//!     block); if such an arm's result ever types as Tensor, tracking a
//!     stale entry would free the live binding. Today no arm has that
//!     full shape (`copy_data` types non-tensor, so the tracker's type
//!     filter rejects it and removing the clear does NOT go red — the
//!     mutation run on 2026-07-29 confirmed this). The fixture pins the
//!     binding's value anyway so the day an identity-shaped
//!     tensor-typed arm appears, a removed or broken clear aborts here
//!     instead of corrupting silently.

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
    let tmp = std::env::temp_dir().join(format!("nsl_dispown_{}_{tag}", std::process::id()));
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

/// Per round: a receiver-position reduce_max, a nested clamp, and a
/// bare-statement clamp — three strands per round pre-v2a — each with a
/// bound-spelling parity twin (err must be exactly 0).
fn rounds_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
let x = (arange(0.0, 64.0) * 0.01).to(cuda)
let err = 0.0
"#,
    );
    for i in 0..calls {
        s.push_str(&format!(
            "let n1_{i} = reduce_max(x, 0, 0).item()\n\
             let t1_{i} = reduce_max(x, 0, 0)\n\
             let b1_{i} = t1_{i}.item()\n\
             let n2_{i} = sum(clamp(x, 0.2, 0.5)).item()\n\
             let t2_{i} = clamp(x, 0.2, 0.5)\n\
             let b2_{i} = sum(t2_{i}).item()\n\
             clamp(x, 0.0, 1.0)\n\
             err = err + (n1_{i} - b1_{i}) * (n1_{i} - b1_{i}) \
                       + (n2_{i} - b2_{i}) * (n2_{i} - b2_{i})\n"
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

#[test]
#[ignore = "requires CUDA GPU"]
fn unallowlisted_dispatch_results_do_not_strand() {
    let (lb_small, out_small) = run_fixture(&rounds_src(2), "small");
    let (lb_big, out_big) = run_fixture(&rounds_src(6), "big");
    assert!(
        out_small.contains("PARITY_OK") && out_big.contains("PARITY_OK"),
        "value parity broke: statement-end frees touched something still read\n\
         small:\n{out_small}\nbig:\n{out_big}"
    );
    // Pre-v2a: 3 strands per round (receiver reduce_max, nested clamp,
    // bare-statement clamp) — 6 and 18. The two-count assert keeps a
    // constant-offset regression from hiding behind a single count.
    assert_eq!(
        (lb_small, lb_big),
        (0, 0),
        "unallowlisted dispatch results stranded (small={lb_small}, big={lb_big})"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn identity_shaped_dispatch_does_not_steal_bindings() {
    let src = r#"
let x = ones([16]).to(cuda)
let q = clamp(x, 0.25, 0.75)
let s = ones([16]).to(cuda)
copy_data(q, s)
let v = q.sum().item()
print(v)
if v == 16.0:
    print("BINDING_OK")
print("DONE")
"#;
    let (_lb, out) = run_fixture(src, "identity");
    assert!(
        out.contains("BINDING_OK"),
        "identity-shaped dispatch freed a live binding (dispatch_fresh \
         not cleared per statement?):\n{out}"
    );
}
