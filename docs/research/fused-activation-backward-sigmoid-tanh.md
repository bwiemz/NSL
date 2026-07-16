# Fused Activation Backward — Sigmoid & Tanh (Milestone C · p4, slice 3)

Change: `crates/nsl-runtime/src/{cuda/kernels.rs,tensor/activation.rs}`,
`crates/nsl-codegen/src/{source_ad.rs,wengert_lower.rs,builtins.rs}`.

## 1. Problem

Source-AD lowers each `AdjointExpr` into concrete PrimalOps, one runtime FFI =
one GPU launch + one device tensor per op. Two more elementwise activation
adjoints each expand into three:

```
SigmoidBackward(grad, y):   t1 = 1 - y;  t2 = y*t1;   out = grad*t2   => grad·y·(1-y)
TanhBackward(grad, y):      t1 = y*y;    t2 = 1 - t1;  out = grad*t2   => grad·(1-y²)
```

Both operands' second input `y` is the activation **output** (the adjoint
carries `op.result`), not the pre-activation. Three launches + two intermediates
each, once per activation per layer per backward. Purely elementwise → one
kernel, mirroring slice 2's SiLU.

## 2. The change

New FFIs `nsl_tensor_sigmoid_backward(grad, y)` / `nsl_tensor_tanh_backward(grad,
y)` compute the whole expression in one launch. `SigmoidBackward` / `TanhBackward`
now emit a single `Passthrough(...)`, lowered by `wengert_lower.rs` to the FFI
(same mechanism as `mse_backward`/`l1_backward`/`silu_backward`).

- **GPU f32**: `SIGMOID_BACKWARD_SRCAD_F32_PTX` / `TANH_BACKWARD_SRCAD_F32_PTX`
  reproduce source-AD's exact operation order.
- **CPU f32/f64**: the same order in a direct loop.

## 3. Bit-exactness

Byte-identical to the three-op decomposed path, so training output does not
shift. Two facts make it hold:

1. **`.rn` blocks fma-contraction.** In Tanh, `t1 = y*y` feeds `t2 = 1 - t1` — a
   mul-feeding-sub that ptxas would otherwise contract into a single-rounding
   `fma(-y, y, 1.0)` (~1 ULP drift, exactly as in slices 1–2). `mul.rn.f32`
   forces `y*y` to round to f32 first (matching the decomposed path, which stores
   `y_sq` to global memory before the subtract) and `sub.rn.f32` keeps them
   separate. Sigmoid has no contractible mul+add pair, but uses `.rn` too for
   consistency and zero cost.
2. **`reconcile_device` keeps the constant harmless.** The decomposed `Sub(1, y)`
   has the `1.0` constant (a CPU f32 scalar from `nsl_tensor_scalar` dt=1) as its
   *first* operand, so `reconcile_device` runs the subtract on CPU — but `1.0` is
   exactly representable, so `1.0 - y` is the correctly-rounded f32 result on
   either device. The `y*y`, `y*(1-y)` and final `grad*·` muls run on GPU. All
   match the fused kernel element-for-element.

   > This is precisely why GELU is **not** in this slice: its adjoint is
   > `Sigmoid(1.702·x)`, and the `1.702·x` mul (constant first) drags the sigmoid
   > onto CPU's libm path, which the GPU `ex2.approx` sigmoid cannot reproduce
   > bit-for-bit. GELU needs its own treatment — see slice 4.

## 4. Scalar-grad broadcast fallback

When the incoming adjoint is **not** the same shape/device/dtype as `y` — e.g. a
scalar `sum`/`mean` seed `[1]` broadcasting against `y[n]`, as in
`sum(sigmoid(x))` — the fused elementwise kernel can't apply (its final `grad··`
would need to broadcast, and the flat CPU loop assumes one dtype). Both FFIs
detect `!grad.shape_eq(y) || grad.device != y.device || grad.dtype != y.dtype`
and fall back to the decomposed op sequence (same f32 `1.0` constant), whose final
`nsl_tensor_mul` broadcasts grad and whose sub/mul promote across dtype —
numerically equivalent to the pre-fusion path. The common mid-network case (grad
shape/device/dtype == y) takes the single fused launch.

## 5. Validation

- **Bit-exact unit tests** (`{sigmoid,tanh}_backward_{cpu_f64,gpu_f32}_matches_3op`):
  fused FFI vs a hand-built three-op reference (`sub`/`mul`/`mul` etc., `flags=0`),
  `to_bits()` equality. GPU tests use n=257 (256-block tail).
- **Broadcast fallback tests** (`{sigmoid,tanh}_backward_scalar_grad_broadcasts`
  CPU + `sigmoid_backward_gpu_scalar_grad_broadcasts`): scalar grad vs y[n],
  y-shaped output, values match closed form.
- **E2E**: `grad(x): sum(sigmoid(x))` and `sum(tanh(x))` under `--source-ad`
  return the exact analytic gradients (`σ'`, `tanh'`) at x = [0, 1, -1],
  exercising the fused routing **and** the broadcast fallback.
- **Regression**: 811 `nsl-runtime` (cuda) + 2905 `nsl-codegen` lib tests pass.

## 6. Discovered (pre-existing, deferred): source-AD forward in-place mutation

While E2E-testing, `grad(x): sum(silu(x))` under `--source-ad` was found to leave
`x` mutated to `silu(x)` (verified: `print(x)` before/after). Tape-AD (no
`--source-ad`) preserves `x`. The source-AD **forward** FBIP-mutates the
activation input in place when it is uniquely owned. Consequences:

- **Output-saving activations (sigmoid, tanh) are immune** — their backward reads
  the saved output `y`, so a mutated `x` is irrelevant. Their gradients are
  correct (this slice's E2E confirms exact values).
- **Input-saving activations (silu, gelu) are corrupted** — their backward reads
  the saved input, which is now `f(x)`, so they compute `f'(f(x))`. In real MLPs
  the activation input is a matmul temporary with refcount > 1, so the forward
  FBIP is rejected and gradients are correct (slice 2's MLP matched tape-AD to
  1e-4); the bug only bites when the input is uniquely owned at the activation
  call.

This is orthogonal to backward fusion and pre-dates slice 2. It is the reason
**SiLU's broadcast-grad fallback and GELU are deferred to slice 4**, which fixes
the forward-mutation root cause first (guard the saved-for-backward input against
in-place FBIP) and only then fuses silu/gelu's broadcast/CPU-sigmoid cases.
