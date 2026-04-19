# WRGA B.3.2 Option 3 -- Source-AD Wengert Handler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Teach source-AD to recognize `nsl_adapter_fused_gatedlora_matmul` calls so the fused GatedLoRA forward kernel actually fires under `--source-ad` inside `train` blocks. Backward decomposes to known primitive ops; no new kernel.

**Architecture:** Add one new `PrimalOp` variant (`FusedGatedLoraMatmul`), one extractor match arm for its callee name, and one AD rule that emits 5 input adjoints via existing primitive ops (`Matmul`, `Sigmoid`, `Mul`, `Sub`, `Transpose`, `BroadcastTo`, `ReduceSum`). Sequentially shipped: diagnostic warning first (separate commit, orthogonal), then option 3 wiring.

**Tech Stack:** Rust, NSL codegen internals (`wengert.rs`, `source_ad.rs`), existing integration test infrastructure (`wrga_adapter_runtime_equivalence.rs`).

**Spec:** [2026-04-19-wrga-b32-option3-source-ad-wiring-design.md](../specs/2026-04-19-wrga-b32-option3-source-ad-wiring-design.md)

---

## Task 0: Verification gate — confirm gate placement works

**Files:**
- No code changes. Diagnostic-only task.

- [ ] **Step 1: Write minimal NSL program to probe gate placement under train block**

Create `/tmp/probe_gate_placement.nsl`:

```text
from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = zeros([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)

train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
print("ok")
```

- [ ] **Step 2: Run with launch counter, check gate device**

```bash
cd .worktrees/wrga-b32-fused-backward && \
  NSL_STDLIB_PATH="$(pwd)/stdlib" \
  NSL_WRGA_FUSED_CUDA=1 \
  NSL_WRGA_GPU_LAUNCH_COUNTER=1 \
  target/debug/nsl.exe run --source-ad --target cuda_sm80 /tmp/probe_gate_placement.nsl 2>&1 | \
  grep -E "launch-count|fallback|inputs not on GPU"
```

Expected outcome **before** option 3 lands: `[nsl-gpu-launch-count] 0` (no fused kernel fires because source-AD doesn't handle the FFI). No "inputs not on GPU" warning — because the fused FFI is never called.

- [ ] **Step 3: Confirm the placement question with a non-train-block probe**

Create `/tmp/probe_gate_no_train.nsl` with a one-off `let y = m.forward(x)` AFTER an init train block (mirroring B.3.1 fixture A's pattern). Run the same way. Expected: `[nsl-gpu-launch-count] >= 1` (fused kernel fires in inference), AND no "inputs not on GPU" warning (gate placement works in this path).

- [ ] **Step 4: Record verdict in implementation-notes file**

If Step 3 succeeds (placement works in inference), the conclusion is: the placement infrastructure handles `gate` correctly in the inference path. The train-block path does not exercise the fused FFI yet (option 3 will). When option 3 lands, the train-block path will start calling the fused FFI with the same adapter-field side-table that the inference path used successfully. Placement is therefore in-scope-as-existing-code, not as a new fix.

If Step 3 fails with "inputs not on GPU" for any adapter field, the placement fix becomes a prerequisite to option 3 and the spec needs revision before proceeding.

Write the verdict (one paragraph) to `target/option3-placement-verdict.txt`. This file is the evidence that placement was verified; it stays on the branch as artifact.

- [ ] **Step 5: Commit the verdict artifact**

```bash
git add target/option3-placement-verdict.txt
git commit -m "chore(wrga-b32-opt3): verify gate placement precondition for option 3 tests"
```

*(if target/ is gitignored, use `git add -f`; the artifact is small and we want the record)*

---

## Task 1: Silent-fallback diagnostic warning (separate commit)

**Files:**
- Modify: `crates/nsl-codegen/src/source_ad.rs:1796-1800`

- [ ] **Step 1: Read the fallback site to confirm exact context**

Read `source_ad.rs` around line 1796 (the "unsupported callee expression" eprintln!) to understand surrounding code and match the existing diagnostic format.

- [ ] **Step 2: Write a failing test asserting the warning appears**

Add to a new file `crates/nsl-codegen/tests/source_ad_diagnostics.rs`:

```rust
//! Diagnostic coverage tests for source-AD fallback paths.

use std::process::Command;
use assert_cmd::prelude::*;
use tempfile::TempDir;

#[cfg(feature = "cuda")]
#[test]
fn source_ad_warns_on_unrecognized_ffi_callee() {
    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("probe.nsl");
    // Use a synthetic callee that will never be recognized. We trigger it via
    // the @adapter(gatedlora) path BEFORE option 3 lands — the fused FFI is
    // exactly the unrecognized callee case this warning is designed to catch.
    std::fs::write(&src_path, r#"from nsl.nn.losses import mse_loss

model Toy:
    w: Tensor = zeros([8, 8])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["Toy.w"], rank=2, alpha=2)
let m = Toy()
m.to(cuda)
let x = zeros([4, 8]).to(cuda)
let y_target = zeros([4, 8]).to(cuda)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#).unwrap();

    let root: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    let workspace_root = root.parent().unwrap().parent().unwrap();
    let stdlib = workspace_root.join("stdlib");

    let out = Command::cargo_bin("nsl").unwrap()
        .env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path)
        .output()
        .unwrap();

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("[source-ad] warning: unrecognized FFI callee"),
        "expected source-AD warning about unrecognized FFI callee.\nstderr:\n{stderr}",
    );
    assert!(
        stderr.contains("nsl_adapter_fused_gatedlora_matmul"),
        "warning should name the specific unrecognized FFI.\nstderr:\n{stderr}",
    );
}
```

- [ ] **Step 3: Run the test — expect FAIL**

```bash
cargo test --features cuda --test source_ad_diagnostics source_ad_warns_on_unrecognized_ffi_callee
```

Expected: FAIL because no warning is emitted yet.

- [ ] **Step 4: Add the warning at source_ad.rs:1796**

Read `source_ad.rs:1790-1802`, then edit. The existing code is:

```rust
                } else {
                    eprintln!(
                        "[source-ad] unsupported callee expression: {:?}",
                        std::mem::discriminant(&callee.kind)
                    );
                    return None; // Complex callee -- can't extract
                };
```

This is the OPAQUE callee case (not a direct function call). The specific case we care about is a `Call` whose callee is an `Ident` resolving to an unknown function name. Look for the path where `func_name` is known but no matching extractor arm fires.

Review `source_ad.rs:1998` which already has `eprintln!("[source-ad] unsupported function: '{}'", func_name);` — THAT is the site. Modify it to the new format:

```rust
eprintln!(
    "[source-ad] warning: unrecognized FFI callee '{}' in train block; \
     falling back to unfused AST evaluation. If you expected a fused kernel, \
     check that source-AD has a handler for this FFI.",
    func_name
);
```

- [ ] **Step 5: Run the test — expect PASS**

```bash
cargo test --features cuda --test source_ad_diagnostics source_ad_warns_on_unrecognized_ffi_callee
```

Expected: PASS.

- [ ] **Step 6: Run full source_ad test suite to catch regressions**

```bash
cargo test --features cuda -p nsl-codegen source_ad 2>&1 | tail -10
```

Expected: all pre-existing tests still green. The warning is strictly additive.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/source_ad.rs crates/nsl-codegen/tests/source_ad_diagnostics.rs
git commit -m "feat(source-ad): warn on unrecognized FFI callee instead of silent fallback

Prevents the class of silent-fallback bug discovered during B.3.2 trigger
measurement (2026-04-18), where NSL_WRGA_FUSED_CUDA=1 + @adapter(gatedlora)
in a train block produced [nsl-gpu-launch-count] 0 with no diagnostic
signal. The user had opted into the fused path but got unfused execution.

This commit does not add new fused-path handlers. It makes the failure
mode discoverable so future silent-fallback regressions surface at runtime.
Option 3 wiring (which adds the specific handler for
nsl_adapter_fused_gatedlora_matmul) lands separately.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Add PrimalOp::FusedGatedLoraMatmul variant

**Files:**
- Modify: `crates/nsl-codegen/src/wengert.rs` (PrimalOp enum + type_for_op)

- [ ] **Step 1: Read existing PrimalOp enum + type_for_op**

Read `wengert.rs` around lines 52-90 to confirm enum structure and patterns.

- [ ] **Step 2: Write a failing unit test**

Add to the existing `#[cfg(test)] mod tests { ... }` block in `wengert.rs`:

```rust
#[test]
fn fused_gatedlora_matmul_op_has_correct_type_and_inputs() {
    use super::*;
    // Shape: x[3, 8] @ W[8, 16] + sigmoid(gate[16]) * (x @ A[8, 4] @ B[4, 16]) * 1.5
    let op = make_op(
        0,
        0,
        PrimalOp::FusedGatedLoraMatmul { scale: 1.5 },
        vec![1, 2, 3, 4, 5], // x, W, A, B, gate VarIds
    );
    assert_eq!(op.inputs.len(), 5, "fused gatedlora takes 5 tensor inputs");
    assert_eq!(type_for_op(&op.op), WengertType::Tensor);
    // Round-trip through PartialEq
    match &op.op {
        PrimalOp::FusedGatedLoraMatmul { scale } => assert_eq!(*scale, 1.5),
        _ => panic!("expected FusedGatedLoraMatmul"),
    }
}
```

- [ ] **Step 3: Run the test — expect FAIL (compile error: variant not defined)**

```bash
cargo test -p nsl-codegen --lib wengert::tests::fused_gatedlora_matmul_op_has_correct_type_and_inputs
```

Expected: compile error.

- [ ] **Step 4: Add the PrimalOp variant**

Edit `wengert.rs` inside `pub enum PrimalOp { ... }`:

```rust
    /// Fused GatedLoRA forward matmul: a single FFI call that computes
    /// `y = x @ W + sigmoid(gate) ⊙ (x @ A @ B) * scale` in one kernel.
    ///
    /// Inputs (VarIds in order): [x, W, A, B, gate].
    /// Output: single tensor `y` with shape [B, N] where B=x.rows, N=W.cols.
    ///
    /// `scale` is compile-time known from the `@adapter(alpha=..., rank=...)`
    /// decorator (scale = alpha / rank). The kernel_handle argument of the
    /// underlying FFI is codegen-internal and does not participate in AD.
    ///
    /// Added 2026-04-19 for B.3.2 option 3 — see
    /// docs/superpowers/specs/2026-04-19-wrga-b32-option3-source-ad-wiring-design.md.
    FusedGatedLoraMatmul { scale: f32 },
```

- [ ] **Step 5: Add the type_for_op arm**

In `wengert.rs::type_for_op`, the existing fallback `_ => WengertType::Tensor` already covers this case (all tensor-producing ops fall through to the default). Verify by reading the function. If there's an explicit match for every variant (no `_` fallback), add:

```rust
        PrimalOp::FusedGatedLoraMatmul { .. } => WengertType::Tensor,
```

- [ ] **Step 6: Run the test — expect PASS**

```bash
cargo test -p nsl-codegen --lib wengert::tests::fused_gatedlora_matmul_op_has_correct_type_and_inputs
```

Expected: PASS.

- [ ] **Step 7: Run full wengert test suite**

```bash
cargo test -p nsl-codegen --lib wengert
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-codegen/src/wengert.rs
git commit -m "feat(wengert): add PrimalOp::FusedGatedLoraMatmul variant

Carries the scale compile-time parameter for B.3.2 option 3. The actual
extractor + AD rule land in subsequent commits; this commit only adds the
variant so downstream code has a concrete type to match against.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Extractor match arm in source_ad.rs

**Files:**
- Modify: `crates/nsl-codegen/src/source_ad.rs` (around line 1817 where `func_name.as_str()` is matched)

- [ ] **Step 1: Read the extractor's match block**

Read `source_ad.rs:1813-1830` to see the `match func_name.as_str()` where primitive op mapping happens (e.g., `"sigmoid" => PrimalOp::Sigmoid,`). The fused callee needs a different shape because it extracts a compile-time scale from its argument list.

- [ ] **Step 2: Write a failing unit test (extractor recognition)**

Add to a new file `crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs`:

```rust
//! Unit tests for source-AD's handling of the fused GatedLoRA forward FFI.
//! See docs/superpowers/specs/2026-04-19-wrga-b32-option3-source-ad-wiring-design.md.

use nsl_codegen::wengert::{PrimalOp, WengertList};

// Helper: build a minimal Wengert-extraction context and feed it a Call to
// nsl_adapter_fused_gatedlora_matmul. Assert that the resulting Wengert list
// contains a FusedGatedLoraMatmul op with scale=2.0.
//
// This test parses a tiny NSL program through the extractor directly rather
// than going through nsl run, so it's a unit test of extract_expr's behavior.

#[test]
fn extractor_recognizes_fused_gatedlora_matmul_callee() {
    // Build a synthetic NSL AST containing a Call to
    // nsl_adapter_fused_gatedlora_matmul(x, W, A, B, 2.0, gate, handle=-1)
    //
    // The extractor infrastructure requires a full compiler context. If
    // constructing one is too heavy for a unit test, fall back to running
    // the NSL program through the production path and inspecting the
    // Wengert list via a test-hook accessor. See the existing ad_csha_*
    // tests in crates/nsl-codegen/tests/ for the pattern.
    //
    // Minimum viable: use nsl_codegen::ast + nsl_codegen::source_ad test hooks.

    // IMPLEMENTER NOTE: if this test proves structurally hard because the
    // extractor requires full compiler state, convert it to an integration
    // test that inspects the Wengert list via a new debug-dump FFI (see
    // Task 3.5). That conversion is acceptable as long as the assertion
    // (variant + scale) is preserved.

    let wengert_list = extract_wengert_for_fused_gatedlora_call(
        /* x_shape */ &[3, 8],
        /* w_shape */ &[8, 16],
        /* a_shape */ &[8, 4],
        /* b_shape */ &[4, 16],
        /* gate_shape */ &[16],
        /* scale */ 2.0,
    );

    let fused_op = wengert_list
        .ops
        .iter()
        .find(|op| matches!(op.op, PrimalOp::FusedGatedLoraMatmul { .. }))
        .expect("Wengert list must contain FusedGatedLoraMatmul after extraction");

    match &fused_op.op {
        PrimalOp::FusedGatedLoraMatmul { scale } => assert_eq!(*scale, 2.0),
        _ => unreachable!(),
    }
    assert_eq!(
        fused_op.inputs.len(),
        5,
        "FusedGatedLoraMatmul takes 5 tensor inputs (x, W, A, B, gate); kernel_handle is not an AD input"
    );
}

fn extract_wengert_for_fused_gatedlora_call(
    _x_shape: &[usize],
    _w_shape: &[usize],
    _a_shape: &[usize],
    _b_shape: &[usize],
    _gate_shape: &[usize],
    _scale: f32,
) -> WengertList {
    // TODO: implementer fills in this helper. Approach options:
    //
    // A) Minimal: build an AST fragment directly, instantiate Wengert
    //    extractor, call extract_expr on the Call node.
    //
    // B) Integration: write a tiny .nsl source that constructs the fused
    //    call, run through compiler up to the Wengert-list stage, dump via
    //    test hook.
    //
    // Start with A. If A requires exposing too much extractor internals
    // (the public API is currently test-gated), fall back to B and add a
    // test-hook FFI at a stable place.
    todo!("implementer fills this based on extractor accessibility")
}
```

This test has a `todo!()` placeholder for the helper; Step 3 will decide which approach (A or B) is feasible and implement it. The assertion shape (variant + scale + input count) is pinned.

- [ ] **Step 3: Decide helper approach based on public-API accessibility**

Run `cargo doc -p nsl-codegen --open` (or grep for `pub fn` in source_ad.rs). If `extract_expr` or a `WengertExtractor::new()` + `extract_expr(...)` interface is publicly reachable from test code, pick approach A (build AST fragment directly). If not, approach B with a small test hook.

- [ ] **Step 4: Implement the helper (approach A or B)**

For approach A, the implementer fills in `extract_wengert_for_fused_gatedlora_call` using the public extractor API. For approach B, the implementer adds an annotated `#[cfg(test)]` or `#[cfg(feature = "test-hooks")]` accessor in source_ad.rs and constructs an NSL source string that would get compiled just through the Wengert stage.

- [ ] **Step 5: Run the test — expect FAIL (no extractor match arm yet)**

```bash
cargo test -p nsl-codegen --test source_ad_fused_gatedlora extractor_recognizes_fused_gatedlora_matmul_callee
```

Expected: FAIL with no `FusedGatedLoraMatmul` in the list (extractor falls through to unsupported-callee path).

- [ ] **Step 6: Add the extractor match arm**

In `source_ad.rs` around line 1817, add (inside the `match func_name.as_str()` block):

```rust
    "nsl_adapter_fused_gatedlora_matmul" => {
        // Args (from wrga_adapter_rewrite::synthesize_gatedlora_fused_call):
        //   args[0] = x       : Tensor, AD input
        //   args[1] = W       : Tensor, AD input
        //   args[2] = A       : Tensor, AD input
        //   args[3] = B       : Tensor, AD input
        //   args[4] = scale   : Float literal (compile-time)
        //   args[5] = gate    : Tensor, AD input
        //   args[6] = handle  : Int literal (codegen-internal, not an AD input)
        //
        // Reorder tensor inputs to [x, W, A, B, gate] and extract scale as f32.
        if args.len() != 7 {
            eprintln!(
                "[source-ad] unexpected arg count {} for nsl_adapter_fused_gatedlora_matmul; expected 7",
                args.len(),
            );
            return None;
        }
        let scale: f32 = match &args[4].value.kind {
            ExprKind::FloatLiteral(v) => *v as f32,
            ExprKind::IntLiteral(v) => *v as f32,
            _ => {
                eprintln!(
                    "[source-ad] nsl_adapter_fused_gatedlora_matmul arg[4] (scale) not a numeric literal; got {:?}",
                    std::mem::discriminant(&args[4].value.kind),
                );
                return None;
            }
        };
        // Re-extract tensor inputs in the order [x, W, A, B, gate].
        let mut tensor_inputs = Vec::with_capacity(5);
        for (idx, arg_idx) in [0_usize, 1, 2, 3, 5].iter().enumerate() {
            match self.extract_expr(&args[*arg_idx].value) {
                Some(v) => tensor_inputs.push(v),
                None => {
                    eprintln!(
                        "[source-ad] nsl_adapter_fused_gatedlora_matmul tensor arg {} (position {}) failed to extract",
                        idx, arg_idx,
                    );
                    return None;
                }
            }
        }
        let result = self.alloc_var();
        self.emit_op_with_inputs(
            PrimalOp::FusedGatedLoraMatmul { scale },
            tensor_inputs,
            result,
        );
        // NOTE: shape inference for `result` is handled by the tensor-type
        // system on a subsequent pass reading the input shapes. No explicit
        // var_type insert here; consistent with how Matmul is treated.
        return Some(result);
    }
```

(The exact helper method names `emit_op_with_inputs` / `alloc_var` may differ — use whatever the surrounding extractor arms call.)

- [ ] **Step 7: Run the test — expect PASS**

```bash
cargo test -p nsl-codegen --test source_ad_fused_gatedlora extractor_recognizes_fused_gatedlora_matmul_callee
```

Expected: PASS.

- [ ] **Step 8: Run full source_ad test suite for regressions**

```bash
cargo test -p nsl-codegen source_ad 2>&1 | tail -5
```

Expected: all existing tests green.

- [ ] **Step 9: Commit**

```bash
git add crates/nsl-codegen/src/source_ad.rs crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs
git commit -m "feat(source-ad): recognize nsl_adapter_fused_gatedlora_matmul in extractor

Adds a match arm in source_ad's extractor for the fused GatedLoRA forward
FFI. Emits a PrimalOp::FusedGatedLoraMatmul with the 5 tensor inputs (x,
W, A, B, gate; kernel_handle is codegen-internal) and scale pulled from
the 5th argument's float literal.

No AD rule yet — reverse walk of this PrimalOp will error until Task 4
lands. Covered by extractor_recognizes_fused_gatedlora_matmul_callee unit
test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: AD rule — emit 5 input adjoints

**Files:**
- Modify: `crates/nsl-codegen/src/source_ad.rs` (apply_ad_rule or reverse-walk dispatch)

- [ ] **Step 1: Identify where AD rules are applied**

Read `source_ad.rs:482-490` (the reverse walk):

```rust
let output_bar = self.get_or_create_adjoint(op.result);
let input_adjoints = apply_ad_rule(op, output_bar);

for InputAdjoint { input_var, expr } in input_adjoints {
    let adj_val = self.lower_adjoint_expr(expr);
    self.accumulate_adjoint(input_var, adj_val);
}
```

The `apply_ad_rule` function is imported from `wengert_lower.rs` (per source_ad.rs:5). Read that function to see how per-op rules dispatch.

- [ ] **Step 2: Write a failing unit test — Test 3 from spec §4 (shape assertions)**

Append to `crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs`:

```rust
#[test]
fn ad_rule_emits_all_five_input_adjoints_with_correct_shapes() {
    // Constants (B=3, K=8, R=4, N=16, scale=1.5 from spec §4 Test 3).
    let b_size = 3;
    let k = 8;
    let r = 4;
    let n = 16;
    let scale = 1.5_f32;

    let (wengert_list, input_vars) = build_wengert_with_fused_gatedlora_and_dy(
        &[b_size, k], &[k, n], &[k, r], &[r, n], &[n], scale,
    );

    let adjoints = run_ad_rule_and_collect_adjoints(&wengert_list);

    // Assert presence + shape per spec §4 Test 3:
    assert_eq!(adjoints.shape_of(input_vars.x), &[b_size, k]);
    assert_eq!(adjoints.shape_of(input_vars.w), &[k, n]);
    assert_eq!(adjoints.shape_of(input_vars.a), &[k, r]);
    assert_eq!(adjoints.shape_of(input_vars.b), &[r, n]);
    assert_eq!(adjoints.shape_of(input_vars.gate), &[n], "dgate must be reduced over batch axis to shape [N]");
}

struct FusedGatedLoraInputVars {
    x: usize, w: usize, a: usize, b: usize, gate: usize,
}

fn build_wengert_with_fused_gatedlora_and_dy(
    _x_shape: &[usize], _w_shape: &[usize], _a_shape: &[usize],
    _b_shape: &[usize], _gate_shape: &[usize], _scale: f32,
) -> (WengertList, FusedGatedLoraInputVars) {
    todo!("implementer fills based on Wengert construction API");
}

struct CollectedAdjoints { /* impl detail: maps VarId → emitted adjoint op shape */ }
impl CollectedAdjoints {
    fn shape_of(&self, _v: usize) -> &[usize] { todo!() }
}

fn run_ad_rule_and_collect_adjoints(_w: &WengertList) -> CollectedAdjoints {
    todo!("implementer runs the reverse walk + collects adjoint shapes");
}
```

- [ ] **Step 3: Implement helpers**

Implementer fills in `build_wengert_with_fused_gatedlora_and_dy` and `run_ad_rule_and_collect_adjoints`. Approach: construct a WengertList programmatically with `Input` primal ops for each of x/W/A/B/gate at the given shapes, then a `FusedGatedLoraMatmul` op consuming them. Feed through the existing reverse-walk test infrastructure.

If the reverse-walk machinery requires a full compiler context, instead run a tiny NSL program through `nsl run` with a debug-dump-adjoints test hook. Same fallback pattern as Task 3.

- [ ] **Step 4: Run the test — expect FAIL**

```bash
cargo test -p nsl-codegen --test source_ad_fused_gatedlora ad_rule_emits_all_five_input_adjoints_with_correct_shapes
```

Expected: FAIL — no AD rule for `FusedGatedLoraMatmul` means adjoints aren't emitted (probably panics with "no rule for PrimalOp::FusedGatedLoraMatmul").

- [ ] **Step 5: Write the AD rule**

Find `apply_ad_rule` (probably in `wengert_lower.rs`). Add:

```rust
        PrimalOp::FusedGatedLoraMatmul { scale } => {
            // Spec §2: per-shape recipe. Inputs = [x, W, A, B, gate]; output = y.
            // dy = output_bar (upstream adjoint).
            //
            // Forward: y = x @ W + sigmoid(gate) ⊙ (x @ A @ B) * scale
            //
            // Adjoints (see spec §2 for shape annotations):
            //   sig          = Sigmoid(gate)
            //   sig_prime    = Mul(sig, Sub(1, sig))
            //   xa           = Matmul(x, A)
            //   xab          = Matmul(xa, B)
            //   sig_bcast    = BroadcastTo(sig, target_shape=shape_of(y), broadcast_axes=[0])
            //   dy_sig       = Mul(dy, sig_bcast)
            //   dy_sig_sc    = Mul(dy_sig, scale)
            //   dx adjoint  = Matmul(dy, Transpose(W)) + Matmul(Matmul(dy_sig_sc, Transpose(B)), Transpose(A))
            //   dW adjoint  = Matmul(Transpose(x), dy)
            //   dA adjoint  = Matmul(Matmul(Transpose(x), dy_sig_sc), Transpose(B))
            //   dB adjoint  = Matmul(Transpose(xa), dy_sig_sc)
            //   dgate adjoint = Mul(ReduceSum(Mul(Mul(dy, xab), sig_prime), axis=0), scale)
            //
            // Implementation: express each line as an AdjointExpr or
            // directly-emitted intermediate PrimalOp and InputAdjoint
            // entries. The existing primitive AD rules handle each
            // sub-op's gradient flow if this rule re-uses them verbatim.

            let x = op.inputs[0];
            let w = op.inputs[1];
            let a = op.inputs[2];
            let b = op.inputs[3];
            let gate = op.inputs[4];
            let dy = output_bar;

            // Emit the intermediate primals (sig, sig_prime, xa, xab,
            // sig_bcast, dy_sig, dy_sig_sc) via adjoint-expression lowering
            // or direct WengertOp emission; use whichever surface
            // apply_ad_rule already uses for multi-op rules. See the
            // SigmoidBackward + LayerNormBackward precedents for patterns.

            // Return InputAdjoint entries for [x, W, A, B, gate] in that order.
            vec![
                InputAdjoint { input_var: x, expr: /* dx expr */ },
                InputAdjoint { input_var: w, expr: /* dW expr */ },
                InputAdjoint { input_var: a, expr: /* dA expr */ },
                InputAdjoint { input_var: b, expr: /* dB expr */ },
                InputAdjoint { input_var: gate, expr: /* dgate expr */ },
            ]
        }
```

The implementer completes each `/* dX expr */` by composing `AdjointExpr` values following the recipe. If the AdjointExpr surface doesn't support composite multi-step computations, introduce a new `AdjointExpr::FusedGatedLoraBackward { scale, x, w, a, b, gate, dy }` variant that lowers to the recipe in `lower_adjoint_expr`. Either approach is acceptable; choose based on what matches the existing backward idiom for multi-step rules.

- [ ] **Step 6: Run Test 3 — expect PASS**

```bash
cargo test -p nsl-codegen --test source_ad_fused_gatedlora ad_rule_emits_all_five_input_adjoints_with_correct_shapes
```

Expected: PASS with all 5 shape assertions satisfied.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/source_ad.rs crates/nsl-codegen/src/wengert_lower.rs crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs
git commit -m "feat(source-ad): AD rule for FusedGatedLoraMatmul via primitive ops

Emits 5 input adjoints (dx, dW, dA, dB, dgate) decomposed to existing
primitive PrimalOps (Matmul, Sigmoid, Mul, Sub, Transpose, BroadcastTo,
ReduceSum). No new kernel. Follows the shape-annotated recipe in spec §2.

Covered by ad_rule_emits_all_five_input_adjoints_with_correct_shapes
unit test which pins the expected shapes of all 5 adjoints.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Broadcast-axis unit test (Test 4 from spec §4)

**Files:**
- Modify: `crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs` (add one test fn)

- [ ] **Step 1: Write the test**

Append:

```rust
#[test]
fn broadcast_axis_is_correct_for_gate_dimension() {
    // Per spec §4 Test 4: feed sig=[1, 2, 4, 8] (shape [4]) and
    // dy=ones([3, 4]) (shape [3, 4]) through the Wengert graph the AD
    // rule emits. Assert dy_sig[i, j] == dy[i, j] * sig[j] for all i, j.
    //
    // Catches the axis-swap failure mode where sig is broadcast along
    // the wrong axis (would produce [[1,1,1,1],[2,2,2,2],...]).

    let sig_values = [1.0_f32, 2.0, 4.0, 8.0];
    let dy_values = [[1.0_f32; 4]; 3];

    let dy_sig = run_ad_rule_and_materialize_dy_sig(&sig_values, &dy_values);

    let expected = [
        [1.0_f32, 2.0, 4.0, 8.0],
        [1.0, 2.0, 4.0, 8.0],
        [1.0, 2.0, 4.0, 8.0],
    ];

    for i in 0..3 {
        for j in 0..4 {
            assert!(
                (dy_sig[i][j] - expected[i][j]).abs() < 1e-6,
                "dy_sig[{},{}] = {} but expected {}",
                i, j, dy_sig[i][j], expected[i][j],
            );
        }
    }
}

fn run_ad_rule_and_materialize_dy_sig(
    _sig: &[f32; 4],
    _dy: &[[f32; 4]; 3],
) -> [[f32; 4]; 3] {
    todo!("execute the Wengert graph for dy_sig and return the materialized tensor");
}
```

- [ ] **Step 2: Implement the helper**

Materialize the computation by running the Wengert-lowered graph through the CPU tensor runtime OR by hand-constructing the expected emitted sequence and verifying element-wise. Simpler approach: a direct simulation of what `Mul(dy, BroadcastTo(sig, shape=[3,4], broadcast_axes=[0]))` should produce, compared against the AD rule's output. Pick whichever is cleaner given the test scaffolding.

- [ ] **Step 3: Run the test — expect PASS (assuming Task 4 is correct) OR FAIL (if axis-swap bug)**

```bash
cargo test -p nsl-codegen --test source_ad_fused_gatedlora broadcast_axis_is_correct_for_gate_dimension
```

If PASS: AD rule correctly broadcasts `sig` along the batch axis, not the gate axis. If FAIL: AD rule has an axis-swap bug; fix it in Task 4's implementation before proceeding.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/tests/source_ad_fused_gatedlora.rs
git commit -m "test(source-ad): broadcast-axis unit test for fused GatedLoRA gate"
```

---

## Task 6: Launch-counter integration test (Test 1 from spec §4)

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs` (add one test fn)

- [ ] **Step 1: Write the test — parallels `build_4_fused_cuda_actually_fires`**

Add:

```rust
#[cfg(feature = "cuda")]
#[test]
fn gatedlora_fused_fires_in_train_block() {
    // B.3.2 option 3: the fused GatedLoRA kernel must fire when @adapter
    // is active inside a train block under --source-ad. Pre-option-3 this
    // produced [nsl-gpu-launch-count] 0 (see project_wrga_b32_measurement.md
    // 2026-04-19 addendum).
    let src = r#"from nsl.nn.losses import mse_loss

model LlamaProxy:
    w: Tensor = zeros([64, 64])
    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

@adapter(type=gatedlora, target=["LlamaProxy.w"], rank=2, alpha=2)
let m = LlamaProxy()
m.to(cuda)
let x = zeros([4, 64]).to(cuda)
let y_target = zeros([4, 64]).to(cuda)

train(model = m, epochs = 3):
    optimizer: SGD(lr = 0.0)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y_target)
"#;

    let tmp = TempDir::new().unwrap();
    let src_path = tmp.path().join("fused_in_train.nsl");
    fs::write(&src_path, src).unwrap();
    let root = workspace_root();
    let stdlib = root.join("stdlib");

    let out = Command::cargo_bin("nsl").unwrap()
        .env("NSL_STDLIB_PATH", &stdlib)
        .env("NSL_WRGA_FUSED_CUDA", "1")
        .env("NSL_WRGA_GPU_LAUNCH_COUNTER", "1")
        .arg("run")
        .args(["--source-ad", "--target", "cuda_sm80"])
        .arg(&src_path)
        .output()
        .expect("nsl run failed to spawn");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "nsl run failed:\n{stderr}");

    let launch_line = stderr
        .lines()
        .find(|l| l.contains("[nsl-gpu-launch-count]"))
        .expect("expected nsl-gpu-launch-count line in stderr");
    let count: u64 = launch_line
        .split_whitespace()
        .next_back()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    assert!(
        count >= 3,
        "expected >=3 fused CUDA launches (1 per train epoch, 3 epochs); got {count}.\n\
         0 means source-AD didn't dispatch the fused FFI — option 3 regression.\nstderr:\n{stderr}",
    );
}
```

- [ ] **Step 2: Run — expect PASS (if Tasks 2-4 landed correctly)**

```bash
cargo test --features cuda --test wrga_adapter_runtime_equivalence gatedlora_fused_fires_in_train_block
```

Expected: PASS with `count >= 3`.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs
git commit -m "test(wrga-b32-opt3): launch-counter assertion for fused forward in train block"
```

---

## Task 7: Numerical equivalence test (Test 2 from spec §4)

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

- [ ] **Step 1: Write the test**

Add a test that runs the same model twice (`cuda_sm80` with fused forward via option 3, then `cuda_sm70` with fully unfused) and compares weight tensors at 1e-4 tolerance after one training step with lr=1e-3, with the tolerance-rationale comment from spec §4 Test 2 pinned inline.

Detail: the two runs need to produce parseable weight tensors. Use the existing `print()` pattern from B.3.1 fixtures — after the train step, print `m.w`, `m.lora_A`, etc. as separate lines; parse via the existing `parse_tensor_2d` helper.

Inline the full tolerance-rationale comment from spec §4 Test 2 verbatim above the test function.

- [ ] **Step 2: Run — expect PASS**

```bash
cargo test --features cuda --test wrga_adapter_runtime_equivalence gatedlora_fused_backward_matches_unfused_reference
```

If FAIL: the AD rule from Task 4 produces gradients that differ from the unfused reference beyond the expected 2e-7 / 1e-4 tolerance. Debug via `dgate` shape assertion (common failure) and `sig` broadcast axis (next most common).

- [ ] **Step 3: Commit**

```bash
git commit -am "test(wrga-b32-opt3): fused-vs-unfused weight-level numerical equivalence"
```

---

## Task 8: Inference-unchanged regression test (Test 5 from spec §4)

**Files:**
- Modify: `crates/nsl-cli/tests/wrga_adapter_runtime_equivalence.rs`

- [ ] **Step 1: Write the test**

Run B.3.1 Fixture A's exact NSL program (inference mode — no train block around the final forward) with `--target cuda_sm80 --source-ad` both before option 3 and after, assert output tensor is bitwise-identical.

Since we can't easily run "before" inside the same test, assert instead that B.3.1's existing `gatedlora_fixture_a_baseline`, `_b`, `_c`, `_d` tests still pass unchanged (same expected values, same tolerances). This is already covered by re-running those tests; add an explicit `gatedlora_inference_fixtures_regression_under_source_ad` that invokes the existing fixture helpers and groups them as a single assertion, for reviewer-visibility.

- [ ] **Step 2: Run — expect PASS**

```bash
cargo test --features cuda --test wrga_adapter_runtime_equivalence gatedlora_
```

Expected: all 4 existing B.3.1 fixtures + the new 3 option-3 tests green.

- [ ] **Step 3: Commit**

```bash
git commit -am "test(wrga-b32-opt3): group inference-mode fixtures as regression gate"
```

---

## Task 9: Re-run trigger bench + apply decision tree

**Files:**
- No code changes. Measurement + memory file update.

- [ ] **Step 1: Run the fused trigger bench at full scale**

```bash
cd .worktrees/wrga-b32-fused-backward
cargo test --features cuda --test wrga_gatedlora_backward_trigger wrga_b32_fused_trigger_final -- --ignored --nocapture 2>&1 | tee /tmp/trigger_rerun.log
```

Expected runtime: ~20-25 min based on prior runs.

- [ ] **Step 2: Parse the ratio, apply the decision tree from spec §6**

Extract the `ratio: X.XXXx` lines from the output. For each config:

- ratio > 2.5× → branch: "proceed with B.3.2 kernel work"
- ratio in [1.5×, 2.5×] → branch: "profile matmul-bound vs allocator-bound"
- ratio < 1.5× → branch: "B.3.2 deferred; option 3 wiring justified by fused-forward-in-training value"

- [ ] **Step 3: Update the measurement memory file**

Edit `~/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wrga_b32_measurement.md`:

- In the "The B.3.2 trigger answer" section, add a new sub-section dated 2026-04-19+ titled "Post-option-3 re-measurement" with a table of the new ratios.
- Name the decision-tree branch that each config lands in.
- If one config says "proceed with kernel" and another says "defer," call out the heterogeneity and propose which branch dominates (usually the prescribed shape is authoritative).

- [ ] **Step 4: Commit measurement artifacts**

The bench output file (`target/wrga_b32_fused_trigger_report.md`) is regenerated; optionally commit it for evidence. At minimum commit the memory file update reference (memory is outside the repo so doesn't commit; noted in the repo via a commit message referencing the memory file path).

```bash
git commit --allow-empty -m "chore(wrga-b32-opt3): trigger re-measurement completed

Post-option-3 ratio recorded in memory file project_wrga_b32_measurement.md.
Decision-tree branch: <BRANCH from Step 2>.

See memory file for full verdict and per-config numbers."
```

---

## Task 10: Finishing

Follow superpowers:finishing-a-development-branch. Options at close-out: merge, PR, keep, discard. Default per prior milestones: push + PR.

---

## Self-review notes

Verified spec coverage:

- Silent-fallback warning (§5) → Task 1 ✓
- PrimalOp variant (§2 pt 1) → Task 2 ✓
- Extractor handler (§2 pt 2) → Task 3 ✓
- AD rule (§2 pt 3) → Task 4 ✓
- Test 1 (launch-counter) → Task 6 ✓
- Test 2 (numerical equivalence) → Task 7 ✓
- Test 3 (shape assertions) → Task 4 step 2-6 ✓
- Test 4 (broadcast-axis) → Task 5 ✓
- Test 5 (inference unchanged) → Task 8 ✓
- Verification gate (§7) → Task 0 ✓
- Decision tree + measurement (§6) → Task 9 ✓
- Finishing → Task 10 ✓

Placeholder scan: Tasks 3 and 4 have `todo!()` bodies in test helpers that must be filled by the implementer during Step 3-4 of each task. These are explicitly flagged as "implementer decides approach A or B" decisions, not deferred unknowns. Acceptable under plan discipline because the decision is bounded (API accessibility dictates the choice) and doesn't cascade.

Type consistency: `PrimalOp::FusedGatedLoraMatmul { scale: f32 }` used consistently across tasks. `FusedGatedLoraMatmul` named the same way everywhere. VarId argument order `[x, W, A, B, gate]` named consistently. Input count = 5 asserted twice (wengert test + extractor test) — passes both.

TDD cadence: each task follows write-failing-test → run → implement → run → commit. Tasks 0 and 9 are measurement tasks (no TDD applicable); Task 10 is the finish skill.

---

## Execution handoff

Plan complete. Save confirmed at `docs/superpowers/plans/2026-04-19-wrga-b32-option3-source-ad-wiring-plan.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
