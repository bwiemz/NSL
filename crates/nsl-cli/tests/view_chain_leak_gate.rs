//! Regression gate for the SHAPE/VIEW-CHAIN tensor lifetime leak
//! (roadmap item 1 residual, 2026-07-27).
//!
//! `expr_result_is_owned_temporary`'s tensor-method allowlist carried its own
//! warning: *"this allowlist is hand-maintained and drifts silently — a missing
//! fresh-result builtin costs a leak, never a crash, so nothing fails loudly."*
//! That is precisely what had happened. The entire shape/view family —
//! `reshape`, `transpose`, `expand`, `contiguous`, `unsqueeze`, `select`,
//! `slice`, `cumsum` — was absent, so every such call in NESTED position
//! produced a handle nothing ever registered and nothing ever freed. A method
//! call bound to a `let` was fine (the named local is swept on return); only the
//! ANONYMOUS links of a chain stranded.
//!
//! A view is still an owning reference: `NslTensor::new_view_i64` allocates a
//! fresh handle and bumps the root owner's refcount. `contiguous` is the subtle
//! one — when the tensor is already contiguous it returns the receiver pointer,
//! but only after `refcount.fetch_add(1)`, so the caller owns a counted
//! reference either way.
//!
//! Measured on an RTX 5070 Ti / CUDA 13.3, `nsl.nn.gqa`'s
//! `GroupedQueryAttention::forward` at `[2,1024,512]`, which chains
//!
//! ```text
//! let q4    = q.reshape([...]).transpose(1, 2)
//! let k_exp = k5.expand([...]).contiguous().reshape([...])
//! ```
//!
//! went from **8 to 5 retained blocks per call** (24 MB -> 16 MB); a whole
//! Coder-50M `[2,1024]` forward went from **+89 to +65 blocks** and
//! **+292 MB to +228 MB** per call, with the N=3 peak 2.29 GB -> 2.10 GB.
//!
//! 2026-07-28: the remaining 4-block GQA residual was closed. Root cause was
//! two-layered: the semantic member table (`check_member_access`) typed
//! `expand`/`contiguous`/`unsqueeze`/`select`/`slice`/`cumsum` results as
//! Unknown, and the codegen ownership filters (`track_owned_tensor_expr_
//! result`, `expr_result_is_owned_temporary`) silently drop non-Tensor
//! types — so every chain link after the first `expand` stranded. GQA is now
//! at the 1-block/call floor (the bound result); Coder-50M went **+65 to +33
//! blocks** and **+228 MB to +132 MB** per call (the rest is the separate
//! nested-model-method-argument class, escape.rs's sound refusal).
//!
//! Gate design mirrors `fn_lifetime_leak_gate.rs`: run the same fixture at two
//! call counts and assert the caching allocator's exit `live_blocks` are
//! IDENTICAL. Anything that strands per call scales with the count.

use std::process::Command;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run a whole program and return (exit live_blocks, stdout, stderr).
fn run_fixture(src: &str, tag: &str, n: usize) -> (i64, String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_viewchain_{}_{tag}_{n}", std::process::id()));
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
        "[{tag}/{n}] run failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        stdout.contains("DONE"),
        "[{tag}/{n}] fixture did not complete:\n{stdout}"
    );
    let live_blocks = stderr
        .lines()
        .filter_map(|l| {
            l.split("live_blocks=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .last()
        .unwrap_or_else(|| panic!("[{tag}/{n}] no [gpu-mem] live_blocks report:\n{stderr}"));
    // Remove the scratch dir. `nsl run` links ~140 MB of objects per invocation
    // into the system temp dir, which here is a 31 GB tmpfs — leaving these
    // behind across a full test run fills it, and the resulting
    // "ld: final link failed: No space left on device" surfaces as failures in
    // whatever unrelated suite happens to run next.
    std::fs::remove_dir_all(&tmp).ok();
    (live_blocks, stdout, stderr)
}

/// The real `nsl.nn.gqa` forward, which is where this was found and where the
/// improvement is measured.
///
/// A hand-written chain fixture is NOT a substitute: the first version of this
/// gate used one and it showed byte-identical block counts with the fix applied
/// and reverted. The reason is NOT that it missed the affected path — review
/// showed view-only chains really are fixed there (`x.transpose(0,1).sum()`
/// goes 1 -> 0 blocks). It is that the fixture ended in a materialising
/// `.contiguous()` feeding `.sum()`, whose own separate residual is larger than
/// the improvement and masked it entirely. Either way the lesson stands: a gate
/// whose numbers do not move when you revert the fix is measuring something
/// else.
fn gqa_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
from nsl.nn.gqa import GroupedQueryAttention
let g = GroupedQueryAttention(512, 8, 4, 64, 0.1)
g.to(cuda)
let x = full([2, 1024, 512], 1.0).to(cuda)
"#,
    );
    for i in 0..calls {
        s.push_str(&format!("let r{i} = g.forward(x, false)\n"));
    }
    s.push_str("print(\"DONE\")\n");
    s
}

/// Per-call retained blocks, from two call counts.
fn blocks_per_call(tag: &str) -> f64 {
    let (lb1, _, _) = run_fixture(&gqa_src(1), tag, 1);
    let (lb3, _, _) = run_fixture(&gqa_src(3), tag, 3);
    assert!(
        lb1 > 0,
        "[{tag}] the fixture retained no GPU blocks at all — it is not \
         exercising the device, so any ratchet below would pass vacuously"
    );
    (lb3 - lb1) as f64 / 2.0
}

/// Measured on an RTX 5070 Ti / CUDA 13.3, `GroupedQueryAttention::forward` at
/// `[2,1024,512]`:
///
/// | | blocks/call | MB/call |
/// |---|---|---|
/// | before the view-family fix | 8 | 24 |
/// | after the view-family fix | 5 | 16 |
/// | after the member-table + tracking-filter fix (2026-07-28) | **1** | **4** |
///
/// One block per call is the FLOOR: the returned attention output, which the
/// caller binds and legitimately keeps. The former 4-block residual was two
/// stranded handles per `expand(..).contiguous().reshape(..)` chain — the
/// expand view pinning the RoPE output block, plus the contiguous
/// materialisation — both dropped from tracking because the semantic member
/// table typed `expand`/`contiguous` results Unknown and the codegen
/// ownership filters silently skip non-Tensor types.
const GQA_BLOCKS_PER_CALL_CEILING: f64 = 1.0;

#[test]
#[ignore = "requires CUDA GPU"]
fn anonymous_view_chain_links_do_not_strand_per_call() {
    let per_call = blocks_per_call("gqa_ratchet");
    assert!(
        per_call <= GQA_BLOCKS_PER_CALL_CEILING,
        "GroupedQueryAttention::forward now retains {per_call} blocks per call, \
         above the {GQA_BLOCKS_PER_CALL_CEILING} ceiling (the pre-fix values \
         were 8, then 5, then 1 = the live result only). Three independent \
         mechanisms feed this gate; check all of them: (1) the shape/view \
         family classified owning in `tensor_method_returns_owned_ref` \
         (nsl-codegen/src/expr/mod.rs); (2) the same methods typed Tensor in \
         `check_member_access` (nsl-semantic/src/checker/ops.rs — see \
         tensor_method_result_typing.rs, which fails first and names the \
         method); (3) the indeterminate-type acceptance in \
         `track_owned_tensor_expr_result` / `expr_result_is_owned_temporary`."
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_gqa_residual_is_closed_exactly() {
    // The predecessor of this test (`the_residual_is_still_present_and_bounded`)
    // deliberately asserted the 4-block residual was STILL PRESENT, so that
    // closing it would fail a gate and force the ceiling down instead of
    // letting the improvement pass silently. That fired on 2026-07-28: the
    // residual is now closed, the ceiling above is 1.0, and this replacement
    // pins EXACTITUDE — exactly one retained block per call (the bound
    // result), no more, no fewer.
    //
    // "No fewer" matters as much as "no more": per-call growth below 1 would
    // mean the caller-bound result itself was freed — that is the
    // use-after-free direction, and `view_chain_results_are_numerically_
    // unchanged` alone might miss it if the allocator recycles the block
    // late.
    let per_call = blocks_per_call("gqa_residual");
    assert!(
        (per_call - 1.0).abs() < f64::EPSILON,
        "GroupedQueryAttention::forward retains {per_call} blocks/call; \
         expected exactly 1.0 (the caller-bound result). More than 1 = a \
         chain link stranded again; less than 1 = something freed the bound \
         result (use-after-free hazard)."
    );
}

/// The FREE-FUNCTION spellings of the same family (`contiguous(t)`,
/// `unsqueeze(t, d)`, `slice(t, ...)`, `stack(l, d)`, `tensor_cat(l, d)`,
/// `cumsum(t, d)`) take dedicated early-return lowerings in expr/calls.rs and
/// are classified by the IDENT allowlist in `expr_result_is_owned_temporary`,
/// not the method table. Before 2026-07-28 that allowlist omitted all of
/// them: `x.transpose(0,1).contiguous().sum()` retained 1 block/call while
/// the semantically identical `contiguous(x.transpose(0,1)).sum()` retained
/// 2 — same runtime call, different books.
///
/// The fixture ends in `.item()` so nothing tensor-typed is bound per
/// iteration: with every link tracked, per-call growth must be ZERO.
#[test]
#[ignore = "requires CUDA GPU"]
fn free_function_chain_links_do_not_strand_per_call() {
    let src = |calls: usize| {
        let mut s = String::from("let x = full([256, 1024], 1.0).to(cuda)\n");
        for i in 0..calls {
            s.push_str(&format!(
                "let v{i} = contiguous(x.transpose(0, 1)).sum().item()\n"
            ));
        }
        s.push_str("print(\"DONE\")\n");
        s
    };
    let (lb1, _, _) = run_fixture(&src(1), "freefn", 1);
    let (lb3, _, _) = run_fixture(&src(3), "freefn", 3);
    assert!(
        lb1 > 0,
        "[freefn] fixture retained no GPU blocks at all — not exercising the \
         device, the growth assertion below would pass vacuously"
    );
    assert_eq!(
        lb3, lb1,
        "free-function chain strands {} blocks over 2 extra calls; the \
         free-function names (contiguous/unsqueeze/slice/stack/tensor_cat/\
         cumsum/argmax/causal_mask) must stay in the Ident allowlist of \
         `expr_result_is_owned_temporary` (nsl-codegen/src/expr/mod.rs)",
        lb3 - lb1
    );
}

/// Layer-2 backstop: a chain hanging off an UNANNOTATED fn parameter. The
/// parameter's type is Unknown, so the semantic member table cannot help —
/// tracking works only because (a) `expr_result_is_owned_temporary` accepts
/// indeterminate receivers (the dispatcher defaults them to tensor dispatch,
/// so the classification must follow) and (b) `track_owned_tensor_expr_result`
/// accepts an indeterminate RESULT type for table-owning tensor-method calls.
/// Reverting either re-strands these links.
///
/// Anti-vacuity: the run must actually take the Unknown-dispatch path, which
/// the compiler announces on stderr ("defaulting to tensor dispatch"). If
/// type inference later learns to type unannotated params, this fixture
/// silently stops testing the backstop — the stderr assertion turns that
/// into a visible failure so the fixture gets rewritten instead.
#[test]
#[ignore = "requires CUDA GPU"]
fn unknown_typed_receiver_chains_do_not_strand_per_call() {
    let src = |calls: usize| {
        let mut s = String::from(
            "fn probe(p) -> f64:\n    return p.transpose(0, 1).contiguous().sum().item()\n\nlet x = full([256, 1024], 1.0).to(cuda)\n",
        );
        for i in 0..calls {
            s.push_str(&format!("let v{i} = probe(x)\n"));
        }
        s.push_str("print(\"DONE\")\n");
        s
    };
    let (lb1, _, err1) = run_fixture(&src(1), "unkrecv", 1);
    let (lb3, _, _) = run_fixture(&src(3), "unkrecv", 3);
    assert!(
        err1.contains("defaulting to tensor dispatch"),
        "fixture no longer exercises the Unknown-receiver dispatch path (no \
         'defaulting to tensor dispatch' warning on stderr) — the backstop \
         assertion below is vacuous; rewrite the fixture so the receiver is \
         genuinely untyped:\n{err1}"
    );
    // Anti-vacuity: this fixture legitimately ends with ZERO live blocks
    // (every per-call result is a scalar, and the sweeps free everything
    // else), so "lb1 > 0" cannot be the device-exercise probe here. Driver
    // allocation counters can: they only move when the caching allocator
    // actually served device memory. Mutation-verified 2026-07-28: with the
    // indeterminate acceptance reverted, this fixture strands 2 blocks/call
    // (3 -> 7 across these two run lengths).
    let drv_allocs = err1
        .lines()
        .filter_map(|l| {
            l.split("drv_allocs=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .last()
        .unwrap_or(0);
    assert!(
        drv_allocs > 0,
        "[unkrecv] no driver allocations reported — the fixture is not \
         exercising the device:\n{err1}"
    );
    assert_eq!(
        lb3, lb1,
        "Unknown-receiver chain strands {} blocks over 2 extra calls; check \
         the indeterminate acceptance in `expr_result_is_owned_temporary` \
         and `track_owned_tensor_expr_result`/`expr_is_owning_tensor_method_\
         call` (nsl-codegen/src/expr/mod.rs)",
        lb3 - lb1
    );
}

/// The view ops must still compute the right thing. Freeing a handle that the
/// result actually aliases would corrupt values rather than leak them, so a
/// leak gate alone cannot tell a fix from a use-after-free.
#[test]
#[ignore = "requires CUDA GPU"]
fn view_chain_results_are_numerically_unchanged() {
    // Deterministic inputs: arange, not randn, so the expected sums are exact
    // integers computable by hand. sum(0..4095) = 4095*4096/2 = 8_386_560.
    let src = r#"
let x = arange(0.0, 4096.0).reshape([2, 4, 512]).to(cuda)
let a = x.reshape([2, 4, 8, 64]).transpose(1, 2)
let b = a.reshape([2, 8, 1, 4, 64]).expand([2, 8, 2, 4, 64]).contiguous().reshape([2, 16, 4, 64])
print(x.sum().item())
print(a.sum().item())
print(b.sum().item())
print("DONE")
"#;
    let (_, out, _) = run_fixture(src, "viewnum", 1);
    let nums: Vec<f64> = out
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    assert_eq!(nums.len(), 3, "expected three sums, got:\n{out}");
    // A permutation cannot change the sum.
    assert_eq!(nums[0], 8_386_560.0, "arange sum changed: {out}");
    assert_eq!(nums[1], 8_386_560.0, "transpose is not a permutation: {out}");
    // `expand` duplicates the kv axis 2x, so the sum doubles exactly.
    assert_eq!(nums[2], 16_773_120.0, "expand(2x) did not double the sum: {out}");
}
