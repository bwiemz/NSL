//! Regression gates for the FUNCTION-LEVEL tensor lifetime leaks
//! (roadmap item 1, 2026-07-24).
//!
//! Baseline measured on `origin/main` @ 1e86f101 (RTX 5070 Ti, CUDA 13.3):
//! a plain Coder-50M `[2,1024]` forward retained **+1.81 GB per call** and
//! OOM'd by the 8th, and `@no_grad` changed not a single byte. Three distinct
//! root causes, one gate each:
//!
//!   1. `non_owning_symbols` is flow-INSENSITIVE while a loop body is
//!      generated exactly ONCE, so a loop-carried local seeded from a borrow
//!      (`let h = x` where `x` is a parameter) had its rebind-free site
//!      compiled under the first-iteration state and skipped forever.
//!      Fixed by `materialize_non_owning_aliases_before_loop`.
//!
//!   2. Builtin lowerings that emit multi-call chains never registered their
//!      intermediates. The naive `scaled_dot_product_attention` chain stranded
//!      its scores / scaled / softmax tensors — 3 x [B,H,S,S] per call.
//!      Fixed by `nsl_tensor_free_transient` (a no-op while the tape records).
//!
//!   3. `emit_return_local_sweep` was gated on `ret_ty.is_tensor()`, so any
//!      function returning a scalar never swept its tensor locals.
//!
//!   4. An owning call-result passed as an ARGUMENT stranded, because both
//!      call sites blanket-refused to track arguments on the grounds that the
//!      callee might escape a parameter. `nsl_codegen::escape` now decides
//!      that per parameter instead of assuming the worst.
//!
//! Gate design mirrors `inference_loop_leak_gate.rs`: run the same fixture at
//! two call counts and assert the caching allocator's exit `live_blocks` are
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

/// Run a whole program and return (exit live_blocks, stdout).
fn run_fixture(src: &str, tag: &str, n: usize) -> (i64, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_fnlife_{}_{tag}_{n}", std::process::id()));
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
    let stderr = String::from_utf8_lossy(&out.stderr);
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
    (live_blocks, stdout)
}

// ── Gate 1: loop-carried borrow-seeded local inside a method ──────────────

/// `let h = x` seeds the loop variable from a PARAMETER, so `h` lands in the
/// flow-insensitive `non_owning_symbols` veto set and the single generated
/// rebind-free never fired — one stranded activation per block iteration, per
/// call. This is the exact shape of `NSLCoder.forward_core`.
fn blocks_loop_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
model Blk:
    w: Tensor = randn([256, 256]) * 0.05

    fn forward(self, x: Tensor) -> Tensor:
        let a = x @ self.w
        return silu(a)

model Net:
    blocks: [Blk; 6] = Blk()

    fn run(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

let m = Net()
m.to(cuda)
let x = randn([128, 256]).to(cuda)
"#,
    );
    for i in 0..calls {
        s.push_str(&format!("let r{i} = m.run(x)\nprint(r{i}.ndim)\n"));
    }
    s.push_str("print(\"DONE\")\n");
    s
}

#[test]
#[ignore = "requires CUDA GPU"]
fn loop_carried_borrow_seed_does_not_strand_per_iteration() {
    let (lb2, _) = run_fixture(&blocks_loop_src(2), "blkloop", 2);
    let (lb6, _) = run_fixture(&blocks_loop_src(6), "blkloop", 6);
    assert_eq!(
        lb2, lb6,
        "live_blocks scale with call count — the loop-carried borrow-seeded \
         local is stranding one activation per block iteration again \
         (materialize_non_owning_aliases_before_loop regressed)"
    );
}

// ── Gate 2: naive SDPA chain transients ───────────────────────────────────

/// `scaled_dot_product_attention` without `@flash_attention` lowers to
/// transpose -> matmul -> mul_scalar -> (+mask) -> softmax -> matmul. Only the
/// last result is handed to the expression; every earlier link stranded.
fn sdpa_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
model Attn:
    wq: Tensor = randn([64, 64]) * 0.1

    fn forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        return scaled_dot_product_attention(q, k, v, 0.125, causal=true)

let m = Attn()
m.to(cuda)
let q = randn([1, 2, 64, 64]).to(cuda)
let k = randn([1, 2, 64, 64]).to(cuda)
let v = randn([1, 2, 64, 64]).to(cuda)
"#,
    );
    for i in 0..calls {
        s.push_str(&format!("let o{i} = m.forward(q, k, v)\nprint(o{i}.ndim)\n"));
    }
    s.push_str("print(\"DONE\")\n");
    s
}

#[test]
#[ignore = "requires CUDA GPU"]
fn naive_sdpa_chain_transients_are_released() {
    let (lb2, out2) = run_fixture(&sdpa_src(2), "sdpa", 2);
    let (lb6, out6) = run_fixture(&sdpa_src(6), "sdpa", 6);
    // Anti-vacuity: the attention actually ran the expected number of times.
    let rank_prints = |s: &str| s.lines().filter(|l| l.trim() == "4").count();
    assert_eq!(rank_prints(&out2), 2, "expected 2 rank prints:\n{out2}");
    assert_eq!(rank_prints(&out6), 6, "expected 6 rank prints:\n{out6}");
    assert_eq!(
        lb2, lb6,
        "live_blocks scale with call count — the naive SDPA chain is stranding \
         its scores/scaled/softmax intermediates again \
         (nsl_tensor_free_transient regressed)"
    );
}

// ── Gate 3: scalar-return local sweep ─────────────────────────────────────

/// `emit_return_local_sweep` used to require a tensor-typed return, so this
/// extremely common loss shape stranded every one of its tensor locals.
fn scalar_return_src(calls: usize) -> String {
    format!(
        r#"
model M:
    w: Tensor = randn([256, 256]) * 0.05

let m = M()
m.to(cuda)
let x = randn([128, 256]).to(cuda)

fn lossy(w: Tensor, x: Tensor) -> f64:
    let a = x @ w
    let b = silu(a)
    let d = b - a
    return sum(d * d).item()

let t = 0.0
for i in range(0, {calls}):
    t = t + lossy(m.w, x)
print(t)
print("DONE")
"#
    )
}

#[test]
#[ignore = "requires CUDA GPU"]
fn scalar_returning_fn_sweeps_its_tensor_locals() {
    let (lb3, _) = run_fixture(&scalar_return_src(3), "scalarret", 3);
    let (lb9, _) = run_fixture(&scalar_return_src(9), "scalarret", 9);
    assert_eq!(
        lb3, lb9,
        "live_blocks scale with call count — a scalar-returning function is \
         stranding its tensor locals again (the return sweep is gated on \
         ret_ty.is_tensor() once more)"
    );
}

// ── Gate 4: @no_grad must not be a byte-for-byte no-op ────────────────────

/// Pinned OBSERVATION, not an aspiration: before the item-1 work, an
/// `@no_grad` inference call allocated byte-identically to the plain call.
/// This gate asserts the leak-freedom the fixes deliver holds under
/// `@no_grad` too, so a future tape-scope change cannot silently regress it.
fn nograd_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
model Blk:
    w: Tensor = randn([256, 256]) * 0.05

    fn forward(self, x: Tensor) -> Tensor:
        return silu(x @ self.w)

model Net:
    blocks: [Blk; 4] = Blk()

    fn run(self, x: Tensor) -> Tensor:
        let h = x
        for block in self.blocks:
            h = block.forward(h)
        return h

@no_grad
fn infer(m: Net, x: Tensor) -> Tensor:
    return m.run(x)

let m = Net()
m.to(cuda)
let x = randn([128, 256]).to(cuda)
"#,
    );
    for i in 0..calls {
        s.push_str(&format!("let o{i} = infer(m, x)\nprint(o{i}.ndim)\n"));
    }
    s.push_str("print(\"DONE\")\n");
    s
}

#[test]
#[ignore = "requires CUDA GPU"]
fn no_grad_inference_is_iteration_independent() {
    let (lb2, _) = run_fixture(&nograd_src(2), "nograd", 2);
    let (lb6, _) = run_fixture(&nograd_src(6), "nograd", 6);
    assert_eq!(
        lb2, lb6,
        "live_blocks scale with call count under @no_grad — bounded inference \
         allocation regressed"
    );
}

// ── Gate 5: escape analysis releases owning call-result arguments ─────────

/// `self.inner.forward(self.n.forward(x))` passes a FRESH tensor straight into
/// a method that only reads it. Both call sites used to blanket-refuse to
/// track arguments — "the callee may ESCAPE a param" — so every such argument
/// stranded. `nsl_codegen::escape` now proves per-parameter captivity and the
/// argument is released at statement end.
///
/// LIMITATION pinned by this fixture's shape: all three models are declared in
/// ONE module. Imported model method bodies are not threaded into module
/// compiles (`compile_module_with_imports` receives only names + signatures),
/// so a call into an IMPORTED model still takes the conservative path.
fn escape_arg_src(calls: usize) -> String {
    let mut s = String::from(
        r#"
model Norm:
    g: Tensor = randn([256, 256]) * 0.05

    fn forward(self, x: Tensor) -> Tensor:
        return silu(x @ self.g)

model Inner:
    w: Tensor = randn([256, 256]) * 0.05

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

model Blk:
    n: Norm = Norm()
    inner: Inner = Inner()

    fn forward(self, x: Tensor) -> Tensor:
        return self.inner.forward(self.n.forward(x))

let m = Blk()
m.to(cuda)
let x = randn([128, 256]).to(cuda)
"#,
    );
    for i in 0..calls {
        s.push_str(&format!("let r{i} = m.forward(x)\nprint(r{i}.ndim)\n"));
    }
    s.push_str("print(\"DONE\")\n");
    s
}

#[test]
#[ignore = "requires CUDA GPU"]
fn owning_call_result_argument_is_released() {
    let (lb2, _) = run_fixture(&escape_arg_src(2), "escarg", 2);
    let (lb6, _) = run_fixture(&escape_arg_src(6), "escarg", 6);
    assert_eq!(
        lb2, lb6,
        "live_blocks scale with call count — the intermediate passed as an \
         argument is stranding again (escape analysis regressed or was \
         disabled)"
    );
    // Anti-vacuity: only the two model weights may survive.
    assert_eq!(lb2, 2, "expected exactly the 2 model weights live at exit");
}

/// The escape hatch must really restore the old behaviour, and the old
/// behaviour must really be worse — otherwise the gate above proves nothing.
#[test]
#[ignore = "requires CUDA GPU"]
fn disabling_escape_analysis_restores_the_strand() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_fnlife_{}_escoff", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");

    let mut lbs = Vec::new();
    for n in [2usize, 6usize] {
        std::fs::write(&prog, escape_arg_src(n)).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
            .args(["run", "--deterministic"])
            .arg(&prog)
            .current_dir(&tmp)
            .env("NSL_STDLIB_PATH", root.join("stdlib"))
            .env("NSL_GPU_MEM_REPORT", "1")
            .env("NSL_NO_ESCAPE_ANALYSIS", "1")
            .output()
            .expect("spawn nsl run");
        assert!(out.status.success());
        let stderr = String::from_utf8_lossy(&out.stderr);
        lbs.push(
            stderr
                .lines()
                .filter_map(|l| {
                    l.split("live_blocks=")
                        .nth(1)
                        .and_then(|s| s.split_whitespace().next())
                        .and_then(|s| s.parse::<i64>().ok())
                })
                .last()
                .expect("no live_blocks report"),
        );
    }
    assert!(
        lbs[1] > lbs[0],
        "with NSL_NO_ESCAPE_ANALYSIS=1 the argument should strand once per \
         call, so live_blocks must GROW ({} -> {}); if it does not, the \
         companion gate is vacuous",
        lbs[0],
        lbs[1]
    );
}
