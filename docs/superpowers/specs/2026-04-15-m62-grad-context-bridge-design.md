# M62 Grad-Context Bridge Fix — Design

**Date:** 2026-04-15
**Status:** Approved for implementation
**Branch (target):** `feat/m62-finish`
**Sibling spec:** [`@export` Decorator](2026-04-15-m62-export-decorator-design.md) — companion M62 finish work.
**Predecessor:** [M62 Legacy Interop spec](2026-03-19-m62-legacy-interop-design.md) §9 Gradient Context + §11 torch.autograd integration tests — partial implementation audit in the 2026-04-15 session identified three correctness bugs in the shipped bridge.

## 1. Goal

Make `nslpy.autograd.nsl_forward(model, *inputs)` + `loss.backward()` propagate correct gradients into an upstream PyTorch graph. Today this path silently returns all-None gradients because of three compounding bugs in the Rust-Python handoff. Fix the bugs; keep the shipped stateful FFI shape.

## 2. Non-Goals

- **Per-call context API** (spec §9 proposal with opaque handles). Stateful API is sufficient for every realistic PyTorch integration pattern; concurrent-NSL-models inside one autograd graph is niche enough to defer.
- **Second-order gradients** (grad-of-grad). The tape doesn't support this today.
- **Weight-gradient FFI export.** The shipped `nsl_model_backward` returns weight gradients (∂L/∂W); no consumer uses them correctly today. NSL's own training path compiles backward via stmt.rs/FASE and doesn't need FFI. Dead code path; removed.
- **Multi-output NSL functions returning tuples.** The scalar seed composition (§4.3) extends naturally, but the `NslTensorDesc` array plumbing on both sides needs more work; defer until an `@export` function actually returns a tuple.
- **Fixing `@export`.** Separate spec — the two fixes can land in any order and compose cleanly.

## 3. Three Bugs Being Fixed

### 3.1 Bug 1 — Rust returns weight gradients, Python treats them as input gradients

[c_api.rs:629-708](../../../crates/nsl-runtime/src/c_api.rs#L629) builds a `param_list` from `model.weight_ptrs` and calls `nsl_tape_backward(loss_ptr, param_list)`. The returned `grad_inputs_ptr` buffer is filled with **∂L/∂W** — one gradient per model weight.

[autograd.py:63-88](../../../python/nslpy/autograd.py#L63) receives that buffer and threads `grad_inputs[i]` into `NslFunction.backward`'s return tuple as if it were **∂L/∂input** — the gradient w.r.t. the i-th input tensor. These are different objects with different shapes (weights are model params, inputs are activation tensors passed to forward).

When `num_weights ≠ num_inputs` (typical), the Python `for i in range(ctx.num_inputs)` loop either returns garbage shapes or falls through to None. When `num_weights == num_inputs` by accident, upstream torch layers silently receive ∂L/∂W values that are semantic nonsense in their graph position.

### 3.2 Bug 2 — Python never enables grad recording

No occurrence of `enable_grad` anywhere in `python/nslpy/`. `NslFunction.forward` calls `model.forward()` directly without first toggling `model.grad_enabled`. When `NslFunction.backward` runs, [c_api.rs:642-645](../../../crates/nsl-runtime/src/c_api.rs#L642) hits `if !model.grad_enabled { return -1 }` and reports `grad not enabled` via thread-local error. Every call fails before tape replay.

### 3.3 Bug 3 — Swallowed exception masks Bugs 1 and 2

[autograd.py:76-80](../../../python/nslpy/autograd.py#L76):

```python
try:
    grad_inputs = model.backward(grad_output, inputs)
except Exception:
    return (None,) + (None,) * ctx.num_inputs
```

Any error (including Bug 2's `-1` return) becomes all-None gradients. `NslFunction.backward` returns cleanly; torch.autograd thinks the backward pass succeeded but no gradients propagate upstream. Users see `loss.backward()` complete without errors and optimizer steps silently do nothing useful.

### 3.4 Bug 4 (partial) — `grad_outputs_ptr` ignored

[c_api.rs:629-634](../../../crates/nsl-runtime/src/c_api.rs#L629) declares `_grad_outputs_ptr: i64` (underscore-prefix, marked "reserved") and seeds backward from the scalar forward output directly. Torch.autograd passes `grad_output` (the upstream gradient) expecting chain-rule composition; Rust ignores it. For the canonical `loss.backward()` on a scalar this is accidentally correct (`grad_output` is 1.0), but for any composition scenario (NSL sits in the middle of a larger torch graph), it produces wrong gradients.

## 4. Design

### 4.1 Rust — stash forward inputs on the model

Add to the `NslModel` struct (in the runtime module's model definition):

```rust
pub struct NslModel {
    // existing fields
    pub weight_ptrs: Vec<i64>,
    pub grad_enabled: bool,
    pub last_forward_outputs: Vec<i64>,

    // NEW
    pub last_forward_inputs: Vec<i64>,
}
```

In `nsl_model_forward` (existing FFI at [c_api.rs:255](../../../crates/nsl-runtime/src/c_api.rs#L255)), when `model.grad_enabled` is true:

1. At entry: clear BOTH `last_forward_outputs` AND `last_forward_inputs` (not end-of-previous-backward — see §4.5).
2. After converting each input `NslTensorDesc*` to an internal `NslTensor`, push the `NslTensor` pointer into `model.last_forward_inputs`.
3. Tape recording already starts inside the NSL-compiled forward function (via `nsl_tape_start` in generated code) — unchanged.

The NSL-side `NslTensor` wrappers created from input descriptors live until the next forward call clears them. Backing memory (DLPack / torch tensor data) is owned by Python and kept alive via `ctx.save_for_backward`.

### 4.2 Rust — `nsl_model_backward` semantic fix

Replace the body of `nsl_model_backward` at [c_api.rs:629-708](../../../crates/nsl-runtime/src/c_api.rs#L629):

```rust
#[no_mangle]
pub extern "C" fn nsl_model_backward(
    model_ptr: i64,
    grad_outputs_ptr: i64,         // NO LONGER underscore-prefixed
    num_grad_outputs: i64,
    grad_inputs_ptr: i64,
    num_grad_inputs_ptr: i64,
) -> i64 {
    if model_ptr == 0 { set_error("…: null model pointer\0".into()); return -1; }
    let model = unsafe { &*(model_ptr as *const NslModel) };

    if !model.grad_enabled {
        set_error("…: grad not enabled — call nsl_model_enable_grad first\0".into());
        return -1;
    }
    if model.last_forward_outputs.is_empty() {
        set_error("…: no forward pass recorded\0".into());
        return -1;
    }
    if model.last_forward_inputs.is_empty() {
        set_error("…: no inputs recorded — forward was called without grad enabled\0".into());
        return -1;
    }

    // Shape validation: one grad_output per forward output.
    if num_grad_outputs as usize != model.last_forward_outputs.len() {
        set_error(format!(
            "…: grad_outputs count {} does not match forward outputs count {}\0",
            num_grad_outputs, model.last_forward_outputs.len()
        ));
        return -1;
    }

    // 1. Convert grad_outputs from NslTensorDesc to NslTensor.
    let grad_output_ptrs: Vec<i64> = (0..num_grad_outputs as usize)
        .map(|i| {
            let desc_ptr = grad_outputs_ptr + (i * std::mem::size_of::<NslTensorDesc>()) as i64;
            nsl_tensor_from_desc(desc_ptr as *const NslTensorDesc)
        })
        .collect();

    // 2. Shape validation per pair (output_i vs grad_output_i).
    for (i, (&out_ptr, &grad_ptr)) in model.last_forward_outputs
        .iter()
        .zip(grad_output_ptrs.iter())
        .enumerate()
    {
        if !shapes_match(out_ptr, grad_ptr) {
            set_error(format!(
                "…: grad_output[{}] shape does not match forward output shape\0", i
            ));
            return -1;
        }
    }

    // 3. Compose scalar seed: L = sum_i (output_i * grad_output_i).sum()
    //    Mathematically equivalent to seeding tape-backward with the
    //    gradient vector. Reuses the existing scalar-seed tape API.
    let seed_ptr = compose_scalar_seed(&model.last_forward_outputs, &grad_output_ptrs);

    // 4. Run backward over INPUTS (not weights — Bug 1 fix).
    let input_list = crate::list::nsl_list_new();
    for &iptr in &model.last_forward_inputs {
        crate::list::nsl_list_push(input_list, iptr);
    }
    let grads_list = crate::autodiff::nsl_tape_backward(seed_ptr, input_list);
    crate::autodiff::nsl_tape_stop();

    let num_inputs = model.last_forward_inputs.len();

    // 5. Write gradient count + descriptors.
    if num_grad_inputs_ptr != 0 {
        unsafe { *(num_grad_inputs_ptr as *mut i64) = num_inputs as i64 };
    }
    if grad_inputs_ptr != 0 && num_inputs > 0 {
        let out_descs = unsafe {
            std::slice::from_raw_parts_mut(grad_inputs_ptr as *mut NslTensorDesc, num_inputs)
        };
        for (i, desc) in out_descs.iter_mut().enumerate() {
            let grad_ptr = crate::list::nsl_list_get(grads_list, i as i64);
            if grad_ptr != 0 {
                nsl_tensor_to_desc(grad_ptr, desc);
            }
        }
    }

    // 6. Cleanup — free the grad_output wrapper NslTensors + loss seed.
    for gptr in grad_output_ptrs { crate::tensor::nsl_tensor_free(gptr); }
    crate::tensor::nsl_tensor_free(seed_ptr);

    // last_forward_outputs cleared at NEXT forward, not here — see §4.5.
    crate::list::nsl_list_free(grads_list);
    crate::list::nsl_list_free(input_list);

    0
}
```

### 4.3 Scalar seed composition

Given forward outputs `y_1, ..., y_n` and user-provided gradient vectors `g_1, ..., g_n` (same shapes), the chain rule says backward from the vector-valued output with seed `g` computes:

    ∂(g · y) / ∂x = sum_i g_i · ∂y_i/∂x

Equivalently, define scalar `L = sum_i (y_i * g_i).sum()` and seed tape-backward from `L` with the default scalar seed (1.0). Then:

    ∂L/∂x = sum_i g_i · ∂y_i/∂x

which is exactly what `grad_output`-seeded backward produces. The existing tape API `nsl_tape_backward(scalar_loss_ptr, param_list)` handles this without modification — the implementation in §4.2 step 3 just needs a helper `compose_scalar_seed` that:

```rust
fn compose_scalar_seed(outputs: &[i64], grad_outputs: &[i64]) -> i64 {
    // For each (output, grad_output) pair:
    //   temp_i = nsl_tensor_elementwise_mul(output, grad_output)
    //   sum_i  = nsl_tensor_sum(temp_i)
    // Then accumulate: seed = sum_0 + sum_1 + ... + sum_n
    // Free temps; return seed pointer (single scalar tensor).
}
```

All operations here are already in the tape (elementwise_mul, sum, add) — recording continues during composition so the tape sees the full loss expression. No new tape ops needed.

### 4.4 Python — `autograd.py` fixes

Rewrite `NslFunction` to pair enable_grad with forward/backward and propagate errors:

```python
class NslFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, model, *inputs):
        ctx.model = model
        ctx.save_for_backward(*inputs)
        ctx.num_inputs = len(inputs)

        # Enable tape recording BEFORE forward. Pair with disable in backward
        # via try/finally to keep model state clean on error paths.
        model._lib.nsl_model_enable_grad(model._handle, 1)
        try:
            output = model.forward(*inputs)
        except Exception:
            model._lib.nsl_model_enable_grad(model._handle, 0)
            raise

        # Output normalization — unchanged from today except no longer
        # swallowing the int-handle case silently.
        if not isinstance(output, torch.Tensor):
            from nslpy._bridge import _dlpack_to_torch
            if isinstance(output, int):
                output = _dlpack_to_torch(output)
            elif isinstance(output, tuple):
                output = output[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs = ctx.saved_tensors
        model = ctx.model

        try:
            grad_inputs = model.backward(grad_output, inputs)
        finally:
            # Always disable grad, even if backward raised.
            model._lib.nsl_model_enable_grad(model._handle, 0)

        # Return (None for the `model` arg, *grad_inputs for each input).
        # Shape contract is enforced in model.backward via an assert — if
        # num_grads != num_inputs, an AssertionError bubbles up here.
        assert len(grad_inputs) == ctx.num_inputs, (
            f"NSL returned {len(grad_inputs)} input gradients but "
            f"expected {ctx.num_inputs}"
        )
        return (None,) + tuple(grad_inputs)
```

**Key changes vs today:**

1. `nsl_model_enable_grad(handle, 1)` called before every forward; `nsl_model_enable_grad(handle, 0)` called after every backward. Pairs via `try/finally`.
2. Exception handling: `except Exception: return None` REMOVED. Errors propagate. Forward-error path disables grad before re-raising. Backward-error path disables grad via `finally` before re-raising.
3. `len(grad_inputs) == ctx.num_inputs` is now an assertion, not a silent pad-with-None — Bug 1's root cause is gone, so the assertion serves as a pin against future regressions.

### 4.5 Input/output lifecycle

**Both `last_forward_inputs` AND `last_forward_outputs` drop at the START of the next `nsl_model_forward` call, not at the end of backward.**

Reasons:
- Torch may call `.backward()` zero times (output never reaches a `.backward()` call). End-of-backward cleanup would leak.
- Torch may call `.backward()` multiple times with `retain_graph=True`. End-of-backward cleanup would break the second call.
- Start-of-next-forward cleanup is both leak-free and re-backward-safe.

The existing `nsl_model_forward` already clears `last_forward_outputs` at entry; extend the clear to `last_forward_inputs` too. No end-of-backward cleanup needed.

### 4.6 `NslModel.backward` in `_core.py`

[_core.py:242-270](../../../python/nslpy/_core.py#L242) currently uses `max_grad_inputs = 8` (hardcoded). Tighten to use the actual input count:

```python
def backward(self, grad_output, inputs):
    from nslpy._bridge import prepare_inputs, convert_outputs
    dl_grads, cleanup = prepare_inputs((grad_output,), self._lib)

    n = len(inputs)
    grad_buf = (ctypes.c_int64 * n)()
    num_grads = ctypes.c_int64(0)

    result = self._lib.nsl_model_backward(
        self._handle,
        ctypes.cast(dl_grads, ctypes.c_int64),
        1,    # num_grad_outputs — one per forward output; assume single-output for now
        ctypes.cast(grad_buf, ctypes.c_int64),
        ctypes.addressof(num_grads),
    )
    _check_error(result, self._lib)

    assert num_grads.value == n, (
        f"NSL returned {num_grads.value} input gradients but expected {n}"
    )
    grads = convert_outputs(grad_buf, num_grads.value, self._lib)
    cleanup()
    return tuple(grads)
```

The `inputs` arg is now required (was defaulted `None` in today's signature). Callers must pass it — the one caller (`NslFunction.backward`) already does.

## 5. Testing

### 5.1 Rust unit tests (`crates/nsl-runtime/src/c_api.rs` test module)

1. **`backward_returns_input_gradients_not_weight_gradients`** — Forward a minimal graph `y = x * w` with scalar `x = 3.0`, `w = 2.0`. Backward with `grad_output = 1.0`. Assert: returned gradient count == 1 (one input); returned value equals `2.0` (= w). NOT 3.0 (= x, the weight gradient). This is the regression pin against Bug 1.

2. **`backward_honors_grad_outputs_seed`** — Same graph with `grad_output = 5.0`. Assert returned gradient value == `5.0 * 2.0 = 10.0`. Pins Bug 4 fix.

3. **`backward_errors_on_output_grad_output_count_mismatch`** — Forward produces 1 output; backward called with `num_grad_outputs = 2` → non-zero return code; error message contains "count".

4. **`backward_errors_on_grad_output_shape_mismatch`** — Forward produces `[4]` output; backward called with `[8]` grad_output → non-zero return code; error message contains "shape".

5. **`backward_errors_without_enable_grad`** — Existing test, kept verbatim.

6. **`backward_errors_without_inputs_recorded`** — Forward called without grad enabled; backward called with grad enabled → non-zero return; error mentions no inputs.

7. **`forward_clears_previous_inputs_at_entry`** — Two forwards in a row (both grad-enabled, different inputs). Backward after the second returns grads for the second's inputs only. Pins §4.5.

### 5.2 Python integration tests (`python/tests/test_autograd.py`, new file)

8. **`autograd_linear_matches_torch_reference`** — NSL model computing `y = W @ x` with fixed `W` and variable input `x`. Wrap in `nslpy.autograd.nsl_forward(model, x)`, compute `loss = y.sum()`, call `loss.backward()`. Compare `x.grad` to a pure-torch reference (same `W`, same `x`, same computation). Must match within `1e-5` absolute tolerance.

9. **`autograd_matmul_chain_matches_torch_reference`** — Same as 8 but the NSL model has two matmul layers. Catches chain-rule composition bugs inside NSL that a single-op test misses.

10. **`autograd_raises_on_rust_error`** — Force a shape mismatch (e.g., torch input shape does not match NSL model expectation; or deliberately mock a -1 return from `nsl_model_backward`). Assert `loss.backward()` raises a Python exception. Pins Bug 3 fix — silent None-tuple is gone.

11. **`nsl_function_forward_pairs_enable_grad_calls`** — Instrument `_lib.nsl_model_enable_grad` (monkeypatch the ctypes function). Run one `nsl_forward` + `backward` cycle. Assert `enable_grad(handle, 1)` was called once before forward AND `enable_grad(handle, 0)` was called once in the backward path (after backward or on forward error).

### 5.3 Integration

All four Python tests require a working `.so` with an NSL model containing at least one `@export`-able forward function. Depends on the sibling `@export` spec landing first — or, as a fallback, uses the existing `nsl_model_forward` path with a model compiled the old way. To avoid the dependency, tests 8-11 can be marked `@pytest.mark.skipif(not _has_export_compiled_model)` and run manually until `@export` ships. Test 7 doesn't need `@export` — it's Rust-only.

### 5.4 Regression

- All existing tests (in `python/tests/test_bridge.py`, `test_core.py`, `test_hub.py`) continue to pass. None exercise the autograd path, so nothing new should regress.
- Rust backward tests in `c_api.rs` that asserted null-handle errors: keep them; they're still valid.

## 6. Architecture Diagram

```
Python side
-----------
torch.autograd.Function.apply(NslFunction, model, x)
    │
    ▼
NslFunction.forward:
    nsl_model_enable_grad(handle, 1)         ◀─── NEW: Bug 2 fix
    try: output = model.forward(x)
    except: nsl_model_enable_grad(handle, 0); raise
    return output

[user: loss = output.sum(); loss.backward()]

NslFunction.backward(grad_output):
    try: grad_inputs = model.backward(grad_output, ctx.saved_tensors)
    finally: nsl_model_enable_grad(handle, 0)
    assert len(grad_inputs) == ctx.num_inputs   ◀─── NEW: Bug 1 pin
    return (None, *grad_inputs)                 ◀─── NEW: no None-pad, no try/except

Rust side
---------
nsl_model_forward(model, inputs, ...):
    model.last_forward_outputs.clear()
    model.last_forward_inputs.clear()           ◀─── NEW: §4.5
    <run forward, record tape>
    model.last_forward_outputs = outputs
    model.last_forward_inputs  = inputs         ◀─── NEW: Bug 1 fix

nsl_model_backward(model, grad_outputs, ..., grad_inputs_buf, ...):
    validate grad_outputs count + shapes        ◀─── NEW: Bug 4 fix
    seed = sum_i (output_i * grad_output_i).sum()
    input_list = model.last_forward_inputs      ◀─── NEW: Bug 1 fix (was weight_ptrs)
    grads = nsl_tape_backward(seed, input_list)
    write grads to grad_inputs_buf
```

## 7. Risks & Open Questions

- **Risk: `NslTensor` input wrappers outlive the Python-side DLPack tensors in rare paths.** If Python releases `ctx.save_for_backward` before `backward` runs (shouldn't happen in a single `loss.backward()` call, but edge cases exist with `retain_graph`), the NSL-side wrappers point at freed memory. Mitigation: document the invariant; the torch.autograd contract guarantees `saved_tensors` stays alive until backward returns.

- **Risk: `compose_scalar_seed`'s elementwise_mul + sum chain adds tape ops during backward-time seed construction.** The tape is in recording state when we add these ops (recording wasn't stopped before this). The chain-rule math is correct regardless; the seed IS part of the loss expression. Just note: seed composition must happen BEFORE `nsl_tape_backward` is called, and the seed tensor's creation is itself traced.

- **Risk: multi-output models will need additional shape-validation logic.** Current code assumes `num_grad_outputs == model.last_forward_outputs.len()`, which is 1 for single-output forwards. Multi-output `@export` functions (out of scope per §2) would exercise the loop over `zip(outputs, grad_outputs)` but need the Python side to pass multiple grad_outputs — requires plumbing that doesn't exist. Defer.

- **Open: should `shapes_match(out_ptr, grad_ptr)` compare dtypes in addition to shapes?** Torch may pass a different-dtype grad (e.g., fp32 grad for fp16 output) if mixed precision is active. For now, assert shape-only; let dtype mismatches produce a type error at the elementwise_mul step. Revisit if users hit it.

## 8. Success Criteria

1. `autograd_linear_matches_torch_reference` passes: NSL-computed gradients match torch reference within `1e-5`.
2. `autograd_raises_on_rust_error` passes: shape mismatch surfaces as a Python exception, not silent None gradients.
3. `backward_returns_input_gradients_not_weight_gradients` passes: Rust returns ∂L/∂input, not ∂L/∂W.
4. `nsl_function_forward_pairs_enable_grad_calls` passes: enable_grad is toggled correctly around every forward/backward cycle.
5. Existing FASE, CSHA, WRGA, CPDT tests continue to pass unchanged — no regressions in any unrelated path.
6. The `except Exception: return None` at `autograd.py:76-80` is deleted.

## 9. Files Touched

**Modify:**
- `crates/nsl-runtime/src/c_api.rs` — `nsl_model_forward` stashes `last_forward_inputs`; `nsl_model_backward` rewritten per §4.2; 6 new Rust unit tests.
- `crates/nsl-runtime/src/<model module>` — add `last_forward_inputs: Vec<i64>` field to `NslModel` struct (file TBD during implementation, likely wherever `NslModel` is defined — check with `grep "pub struct NslModel"`).
- `python/nslpy/autograd.py` — `NslFunction.forward` / `.backward` per §4.4; remove try/except swallow; add enable_grad pairing.
- `python/nslpy/_core.py` — `NslModel.backward` tightening per §4.6.

**Create:**
- `python/tests/test_autograd.py` — 4 Python integration tests (8, 9, 10, 11 from §5.2).

**Not touched:**
- `crates/nsl-runtime/src/autodiff/` — tape backward API is correct; only the param list passed to it changes (inputs vs weights).
- `crates/nsl-runtime/src/dlpack.rs` — DLPack bridge is correct; reused for grad_outputs.
- `python/nslpy/_bridge.py` — tensor conversion helpers are correct; reused.
