# Source-AD Forward In-Place Preservation (Milestone C · p4, slice 4)

Change: `crates/nsl-runtime/src/tensor/mod.rs` (guard + FFI + `can_mutate_inplace`),
`crates/nsl-runtime/src/tensor/activation.rs` (restore SiLU broadcast fallback),
`crates/nsl-codegen/src/{stmt.rs,builtins.rs}` (emit the guard, register the FFI).

## 1. The bug

`grad(x): sum(silu(x))` under `--source-ad` returned `silu'(silu(x))` instead of
`silu'(x)`, and left `x` mutated to `silu(x)`. Tape-AD (no `--source-ad`) was
correct. Discovered while E2E-testing p4 slice 3.

Root cause: FBIP (Functional-But-In-Place) lets an elementwise op reuse a
uniquely-owned input buffer as its output. `can_mutate_inplace()` already blocks
this while **tape-AD is recording** (`!autodiff::is_recording()`), so a forward op
never clobbers a value the backward still needs. **Source-AD builds no tape**, so
nothing set that guard: the forward `nsl_tensor_silu(x)` saw `x` with refcount 1
and mutated it in place, returning the same pointer. The compiled forward then
aliased both `x`'s VarId and the Silu-result VarId to that one (now `silu(x)`)
buffer, and the input-reading `SiluBackward` adjoint read the corrupted value.

Which activations were affected:

- **Input-reading backwards — silu, gelu, relu, abs** — read the saved *input*, so
  a mutated `x` gave `f'(f(x))`. (Relu is accidentally immune: `relu(x) > 0 ⇔ x >
  0`, so its mask is unchanged.)
- **Output-reading backwards — sigmoid, tanh** — read the saved *result*, which is
  exactly the value in the mutated buffer, so they were already correct. This is
  why slice 3's sigmoid/tanh E2E gave exact gradients.

The bug bites whenever the activation input is *uniquely owned* (refcount 1) at
the forward op. That is **not** rare: the canonical `silu(x @ W)` / SwiGLU-gate
pattern feeds silu a fresh matmul temporary with refcount 1, so a **`train`-block
step trained on `silu'(silu(h))`** — a silently wrong gradient in the primary
`--source-ad` pretraining path, not just the degenerate `grad(x): sum(silu(x))`
case (there the uniquely-owned input coincides with a scalar `sum`/`mean` seed).
Verified: one SGD(lr=1) step on `silu(x@W)` prints `sum(m.w)` = −4.59 under
`--source-ad` before the fix vs −4.73 for tape-AD (= `silu'(silu(h))` vs the
correct `silu'(h)`).

## 2. The fix

A dedicated in-place-suppression guard that mirrors what `is_recording()` does for
tape-AD, but without building a tape:

- Runtime (`tensor/mod.rs`): a thread-local depth counter, `inplace_suppressed()`,
  and FFI `nsl_set_inplace_suppressed(on)` (paired inc/dec so nested blocks
  compose). `can_mutate_inplace()` / `can_mutate_inplace_gpu()` gain
  `&& !inplace_suppressed()`.
- Codegen: `Compiler::emit_inplace_suppress(builder, on)` brackets **every**
  source-AD forward primal `compile_wengert_ops` with
  `nsl_set_inplace_suppressed(1)` … `(0)` — at all three source-AD forward sites:
  grad blocks (`stmt.rs::compile_source_ad_grad_block`), **train blocks**
  (`stmt.rs`, the `train(...)` step forward — the pretraining path), and model
  calibration (`calibration/binary_codegen.rs`). The **adjoint** pass runs with
  the guard clear, so backward FBIP still reclaims memory (backward FBIP is
  refcount-correct: a primal is only reused on its last adjoint read).

Scope: FBIP is suppressed only for the source-AD forward primal pass — the values
it produces are mostly the activations the backward needs anyway (which must be
kept regardless), so the memory cost is limited to genuinely-transient forward
intermediates. Correctness strictly improves. (The initial fix guarded only grad
blocks; adversarial review caught that the train-block and calibration forwards —
the important paths — were still corrupted, and they are now guarded too.)

## 3. SiLU broadcast-grad fallback (restored)

`nsl_tensor_silu_backward`'s scalar-seed broadcast fallback — reverted at the end
of slice 3 because it depended on this forward fix to be testable — is restored,
matching the sigmoid/tanh fallbacks. All three now build the `1.0` constant with
a **dtype-matched** `nsl_tensor_scalar` (f64 for f64 inputs, f32 otherwise) so the
decomposed chain stays single-dtype and the gradient is not silently downcast
f64→f32 (closes slice 3 review note L1's spirit at the source, not just the guard).

## 4. Validation

- **Mechanism unit test** (`inplace_suppress_preserves_forward_input`): with the
  guard raised, `nsl_tensor_silu` on a uniquely-owned input allocates a fresh
  output and leaves the input byte-intact; with it clear, the unique-owner FBIP
  path reuses the buffer (same pointer). Proves the guard toggles the behavior.
- **Broadcast fallback test** (`silu_backward_scalar_grad_broadcasts`): scalar
  grad vs x[n] → x-shaped output, values match `grad0·silu'(x[i])`.
- **Grad-block E2E** (`examples/silu_source_ad_grad.nsl` +
  `silu_source_ad_forward_preserve_e2e.rs`): `grad(x): sum(silu(x))` under
  `--source-ad` asserts `dx == silu'(x)` **and** `x` byte-identical afterward
  (0-tolerance) — source-AD now matches tape-AD (`[0.5, 0.92767, 0.07233]`).
- **Train-block E2E** (`examples/silu_train_source_ad.nsl`, same test file):
  one SGD step on `silu(x@W)` — the refcount-1 matmul-temp case — asserts
  `--source-ad` `sum(m.w)` matches the tape-AD run (no hard-coded value).
- **Regression**: 813 `nsl-runtime` (cuda) + 2905 `nsl-codegen` lib tests pass;
  conv2d source-AD gradient E2E, geglu/reglu activation E2E, and the
  `pretrain_loss_decreases_on_gpu` real training loop pass. Review-driven
  additions: a dtype-matched fallback constant, an f16/bf16 loud-reject guard on
  the three activation backwards, and a corrected mechanism-test free.

## 5. Deferred: GELU source-AD backward is independently wrong

`grad(x): sum(gelu(x))` under `--source-ad` returns `[0.625, 0.956, 0.174]` where
the sigmoid-approx `gelu'([0,1,-1])` is `[0.5, 1.068, -0.068]` — wrong even at
x=0, where *every* gelu derivative equals 0.5. This is **not** the forward-mutation
bug: with this fix `x` is preserved for gelu too (verified). It is a separate
defect in the source-AD `GeluBackward` lowering (or a forward/backward gelu-approx
mismatch). GELU backward fusion is deferred until this is understood — fusing a
wrong formula would only make it fast. Tape-AD gelu (tanh-approx derivative) is a
different formula again, so tape-vs-source is not a clean oracle for gelu.
