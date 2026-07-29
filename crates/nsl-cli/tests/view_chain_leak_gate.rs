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
/// | after the member-table + tracking-filter fix (2026-07-28) | 1 | 4 |
/// | after the return/assign owning-ref upgrade (2026-07-28) | **0** | **0** |
///
/// ZERO is the true floor. The "1" of the previous row was believed to be
/// the caller-bound live result — it was not: top-level `let`s are swept at
/// main's return, so the bound result never survives to the exit report.
/// That block was GQA's `return dropout(out, ...)` being DOUBLE-owned: the
/// free-function builtins `rmsnorm`/`dropout` were missing from the
/// owning-ref allowlist, so the callee's Return arm conservatively retained
/// a result that was already an owning transfer, and the caller's single
/// free left one reference behind (the scaled_dot_product_attention bug
/// shape, recurring one allowlist gap later).
const GQA_BLOCKS_PER_CALL_CEILING: f64 = 0.0;

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
    // letting the improvement pass silently. That fired TWICE on 2026-07-28:
    // first when the chain links stopped stranding (5 -> 1), then when the
    // return-arm double-own of `dropout` was fixed (1 -> 0). Zero retained
    // blocks per call is exact: every per-call allocation is freed, and the
    // caller-bound results are swept at main's return before the exit
    // report. Use-after-free coverage is carried by
    // `view_chain_results_are_numerically_unchanged` and the full train/e2e
    // suites — a wrongly freed live tensor there produces wrong numbers, not
    // just block-count drift.
    let per_call = blocks_per_call("gqa_residual");
    assert!(
        per_call.abs() < f64::EPSILON,
        "GroupedQueryAttention::forward retains {per_call} blocks/call; \
         expected exactly 0. Any growth means a per-call allocation stopped \
         being freed — check the owning-ref allowlist (rmsnorm/dropout \
         entries), the Return/Assign Unknown->Owned upgrades, and the \
         ownership tracking filters in nsl-codegen/src/expr/mod.rs."
    );
}

/// End-to-end composition gate for the WHOLE item-1 mechanism, in the exact
/// shape of Coder-50M's `TransformerBlock` + `forward_core`: nested
/// model-method calls as arguments and operands (`x + self.inner.forward(
/// self.norm.forward(x))`), callees whose returns are free-function builtins
/// (`return rmsnorm(...)`, `return dropout(...)` — the Return-arm double-own
/// class), and model-method results ASSIGNED to an existing variable in a
/// loop and after it (`x = self.fwd(x)` — the Assign-arm double-own class).
/// Before 2026-07-28 this shape stranded 4 blocks per block-call plus 1 per
/// reassignment; a Coder-50M forward leaked +33 blocks / +132 MB. Now: ZERO
/// growth.
#[test]
#[ignore = "requires CUDA GPU"]
fn nested_model_composition_and_reassignment_do_not_strand_per_call() {
    let src = |calls: usize| {
        let mut s = String::from(
            r#"
model Norm(dim: int):
    weight: Tensor = ones([dim])
    fn forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, 0.00001)

model Inner(dim: int):
    w: Tensor = randn([dim, dim]) * full([1], 0.02)
    fn forward(self, x: Tensor) -> Tensor:
        let out = x @ self.w
        return dropout(out, 0.1, false)

model Block(dim: int):
    norm1: Norm = Norm(dim)
    inner: Inner = Inner(dim)
    norm2: Norm = Norm(dim)
    fn forward(self, x: Tensor) -> Tensor:
        let h = x + self.inner.forward(self.norm1.forward(x))
        return h + self.inner.forward(self.norm2.forward(h))
    fn run(self, x0: Tensor) -> Tensor:
        let x = x0 + full([1], 0.0)
        for i in range(0, 2):
            x = self.forward(x)
        x = self.norm1.forward(x)
        return x

let b = Block(512)
b.to(cuda)
let x = full([2, 256, 512], 1.0).to(cuda)
"#,
        );
        for i in 0..calls {
            s.push_str(&format!("let r{i} = b.run(x)\n"));
        }
        s.push_str("print(\"DONE\")\n");
        s
    };
    let (lb1, _, _) = run_fixture(&src(1), "composition", 1);
    let (lb3, _, _) = run_fixture(&src(3), "composition", 3);
    assert!(
        lb1 > 0,
        "[composition] fixture retained no GPU blocks at all — not \
         exercising the device (the weights alone should hold blocks)"
    );
    assert_eq!(
        lb3, lb1,
        "nested model composition strands {} blocks over 2 extra calls; \
         check (a) rmsnorm/dropout in the owning-ref allowlist, (b) the \
         Assign handler's Unknown->Owned upgrade in stmt.rs (the \
         `x = self.norm.forward(x)` twin of the Return-arm upgrade), and \
         (c) escape-analysis captivity for nested model-method args",
        lb3 - lb1
    );
}

/// The FREE-FUNCTION spellings of the same family (`contiguous(t)`,
/// `unsqueeze(t, d)`, `tensor_slice(t, ...)`, `stack(l, d)`,
/// `tensor_cat(l, d)`, `cumsum(t, d)`) take dedicated early-return lowerings
/// in expr/calls.rs and are classified by the IDENT allowlist in
/// `expr_result_is_owned_temporary`, not the method table. Before 2026-07-28
/// that allowlist omitted all of them: `x.transpose(0,1).contiguous().sum()`
/// retained 1 block/call while the semantically identical
/// `contiguous(x.transpose(0,1)).sum()` retained 2 — same runtime call,
/// different books.
///
/// The fixture exercises every allowlist entry whose result is a FRESH GPU
/// block: `contiguous` (of a transposed view), `tensor_slice`, `stack`,
/// `tensor_cat`. The other four entries cannot be observed by a GPU
/// block-count gate and are deliberately absent:
/// - `unsqueeze` returns a VIEW — a stranded view handle pins its root,
///   which here is the always-live `x`, so block counts never move;
/// - `cumsum` and `argmax` (sampling.rs) are device-blind CPU
///   implementations — handing them a GPU tensor would read device pointers
///   on host, so they can only run on CPU tensors, whose strands are host
///   allocations invisible to `[gpu-mem]`;
/// - `causal_mask` allocates its output on host for the same reason.
/// Their entries rest on the per-function runtime verification recorded in
/// the allowlist comment plus `tensor_method_result_typing.rs`.
///
/// The fixture ends in `.item()` so nothing tensor-typed is bound per
/// iteration: with every link tracked, per-call growth must be ZERO.
#[test]
#[ignore = "requires CUDA GPU"]
fn free_function_chain_links_do_not_strand_per_call() {
    let src = |calls: usize| {
        let mut s = String::from(
            "let x = full([256, 1024], 1.0).to(cuda)\nlet pair = [x, x]\n",
        );
        for i in 0..calls {
            s.push_str(&format!(
                "let a{i} = contiguous(x.transpose(0, 1)).sum().item()\n\
                 let b{i} = tensor_slice(x, 0, 0, 128).sum().item()\n\
                 let c{i} = stack(pair, 0).sum().item()\n\
                 let d{i} = tensor_cat(pair, 0).sum().item()\n"
            ));
        }
        s.push_str("print(\"DONE\")\n");
        s
    };
    let (lb1, _, stderr1) = run_fixture(&src(1), "freefn", 1);
    let (lb3, _, _) = run_fixture(&src(3), "freefn", 3);
    // Anti-vacuity: prove the fixture exercised the device via the driver
    // allocation counter, NOT via retained blocks. This gate originally
    // required lb1 > 0 — valid while the chains stranded a documented
    // handful of unobservable entries, but the nested-arg conversion
    // (2026-07-29) freed those too, taking lb to a legitimate ZERO. The
    // same probe swap was already made for the unknown-receiver gate
    // below when its fixture reached zero.
    let drv_allocs: i64 = stderr1
        .lines()
        .filter_map(|l| {
            l.split("drv_allocs=")
                .nth(1)
                .and_then(|s| s.split_whitespace().next())
                .and_then(|s| s.parse::<i64>().ok())
        })
        .next_back()
        .unwrap_or(0);
    assert!(
        drv_allocs > 0,
        "[freefn] no driver allocations reported — not exercising the \
         device, the growth assertion below would pass vacuously:\n{stderr1}"
    );
    assert_eq!(
        lb3, lb1,
        "free-function chains strand {} blocks over 2 extra calls; the \
         free-function names (contiguous/tensor_slice/stack/tensor_cat, plus \
         the host-side unsqueeze/cumsum/argmax/causal_mask) must stay in the \
         Ident allowlist of `expr_result_is_owned_temporary` \
         (nsl-codegen/src/expr/mod.rs)",
        lb3 - lb1
    );
    assert_eq!(
        (lb1, lb3),
        (0, 0),
        "the chains now free every link (nested-arg conversion) — a nonzero \
         count here is a strand returning"
    );
}

/// Regression gate for the review finding on 734c548e: the indeterminate-
/// receiver ownership arms must mirror the DISPATCHER's precedence. An
/// indeterminate-typed Ident that is a registered model-array loop variable
/// or agent variable dispatches to a compiled model/agent method
/// (`models.model_var_types` / `agent_var_types` lookups in expr/calls.rs),
/// NOT to tensor dispatch — so a method that happens to share a tensor-table
/// name (`fn mean(self) -> int`) returns a plain I64, and classifying it by
/// the tensor table would make statement cleanup call `nsl_tensor_free(5)`:
/// `from_ptr` dereferences the value as an `NslTensor` box — segfault or
/// silent heap corruption, not a leak.
///
/// REACHABILITY (verified by mutation, 2026-07-28): with the guard removed,
/// this fixture does NOT currently crash, because today's checker types
/// model-array loop vars concretely (the ownership arms additionally require
/// an indeterminate SEMANTIC type, and `for blk in ...` receivers here are
/// Model-typed). The live exposure of the unguarded arm is the M56
/// @pipeline_agent path, whose synthesised agent vars are Error-typed BY
/// DESIGN (see the `Type::Error` handling in expr/calls.rs), plus any future
/// inference change that de-types model vars. This gate therefore pins two
/// things that must BOTH hold for the hazard to stay closed: the value-level
/// behaviour of model-array method calls with table-colliding names, and —
/// should inference ever regress those receivers to Unknown — it becomes the
/// live crash reproducer for the unguarded classifier. CPU-only.
#[test]
fn model_array_methods_with_tensor_table_names_are_not_freed_as_tensors() {
    let src = r#"
model Blk(tag: int):
    _d: Tensor = full([1], float(tag))
    fn mean(self) -> int:
        return 5

model Holder(dummy: int):
    blocks: [Blk; 2] = Blk(0)

let h = Holder(0)
for blk in h.blocks:
    print(blk.mean())
print("DONE")
"#;
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_modelvar_{}", std::process::id()));
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
    let stdout = String::from_utf8_lossy(&out.stdout).to_string();
    std::fs::remove_dir_all(&tmp).ok();
    assert!(
        out.status.success(),
        "model-array method call crashed — the indeterminate-receiver \
         ownership arms are classifying a model method by the tensor table \
         again (check `indeterminate_receiver_takes_tensor_dispatch` in \
         nsl-codegen/src/expr/mod.rs):\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let fives = stdout.lines().filter(|l| l.trim() == "5").count();
    assert!(
        stdout.contains("DONE") && fives == 2,
        "expected two '5' lines and DONE, got:\n{stdout}"
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
