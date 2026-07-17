# Fused GELU Backward + FBIP-Clobber Fix (Milestone C · p4, GELU completion)

Change: `crates/nsl-runtime/src/{cuda/kernels.rs,tensor/activation.rs}`,
`crates/nsl-codegen/src/{source_ad.rs,wengert_lower.rs,builtins.rs}`.

## 1. The bug: source-AD GELU gradients were numerically wrong

`grad(x): sum(gelu(x))` under `--source-ad` returned `[0.625, 0.956, 0.174]` at
x = [0, 1, −1] — wrong even at x=0, where **every** gelu approximation's
derivative is exactly 0.5. Flagged during slice 4 as "a distinct GeluBackward
defect"; root-caused here.

The 7-op `GeluBackward` expansion emitted:

```
kx  = Mul(1.702, x)          # fresh adjoint-internal temp, refcount 1
s   = Sigmoid(kx)            # ← FBIP mutates kx IN PLACE (unique owner)
...
t   = Mul(kx, 1-s)           # ← reads σ(kx), not kx
```

The adjoint pass deliberately runs with slice 4's in-place-suppression guard
clear (so backward FBIP reclaims memory), and `nsl_tensor_sigmoid` reuses a
uniquely-owned input buffer. `kx` is the **only** adjoint-expansion temp in the
codebase that is consumed by an FBIP-capable unary op and then read again
(audited: every other unary site consumes a primal with refcount > 1, or a temp
that is dead afterward). The computed expression collapsed to `s·(1+s·(1−s))`,
which matches the observed wrong values to 7 decimal places at all three probe
points.

## 2. The fix: fuse — and match each device's actual forward

`AdjointExpr::GeluBackward` now emits a single `Passthrough("gelu_backward")` →
`nsl_tensor_gelu_backward(grad, x)`. Fusing eliminates the temp entirely; no
aliasing can exist inside one kernel. This also completes the p4 elementwise
activation-backward fusion set (silu, sigmoid, tanh, gelu): 7 launches + 6
parameter-shaped intermediates → 1 launch.

A pre-existing wrinkle discovered on the way: the **forward** gelu differs per
device — CPU `nsl_tensor_gelu` computes the **tanh approximation**
(`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`) while GPU `GELU_F32_PTX` computes the
**sigmoid approximation** (`x·σ(1.702x)`). The fused backward therefore computes
the derivative of the forward the device actually ran:

- **GPU** (`GELU_BACKWARD_SRCAD_F32_PTX`): `σ(1.702x)·(1+1.702x·(1−σ(1.702x)))`,
  with the sigmoid instruction sequence shared with `SIGMOID_F32_PTX` and `.rn`
  on the derivative ops (family convention; blocks fma-contraction).
  `0f3FD9DB23` = 1.702f, the same value `nsl_tensor_scalar(1.702, 1)` produced.
- **CPU**: the tanh-approx derivative
  `0.5(1+tanh k) + 0.5x·sech²k·√(2/π)(1+3·0.044715x²)` — the same formula the
  tape backward uses, so **source-AD now agrees with tape-AD on CPU** (exactly
  for f64, the CPU default; f32 CPU tensors may differ in the last ulp since
  tape computes the derivative in f64 and rounds once).

(The old expansion's sigmoid-approx formula never matched the CPU forward
anyway; per-device matching strictly improves both devices and makes finite
differences a tight oracle on each.)

Broadcast/mismatch fallback (scalar `sum`/`mean` seed, cross-device, mixed
dtype): materialize `deriv = gelu'(x)` as a fresh tensor (CPU loop / GPU kernel
with grad = ones), then `grad · deriv` via the broadcasting `nsl_tensor_mul`.
No emitted-op sequence — the fallback cannot re-create the FBIP hazard. f16/bf16
loud-reject as in the rest of the family.

## 3. Validation

- **Hypothesis check**: `s·(1+s·(1−s))` reproduces all three observed wrong
  values to 7 decimals — mechanism confirmed before fixing.
- **CPU f64 unit**: fused == `grad · gelu_tanh_deriv` bit-for-bit; x=0 gives
  exactly 0.5.
- **Finite differences, both devices** (gold standard — derivative of the
  *actual* forward): CPU f64 central diff (h=1e-6, tol 1e-7) and GPU f32 central
  diff against `GELU_F32_PTX` (h=1e-2, tol 5e-3) both pass.
- **GPU analytic**: kernel vs host-computed sigmoid-approx derivative over 257
  points (block-tail guard), tol 1e-5 (ex2/rcp.approx).
- **Broadcast fallback**: scalar grad → x-shaped output, values match closed form.
- **E2E** (`examples/gelu_source_ad_grad.nsl`): `grad(x): sum(gelu(x))` under
  `--source-ad` returns the exact tape-AD values `[0.5, 1.08296406, −0.08296406]`
  (was `[0.625, 0.956, 0.174]`) and leaves x byte-intact.
- **Regression**: 818 `nsl-runtime` (cuda) + 2905 `nsl-codegen` lib tests pass.

## 4. The general lesson (encoded as a comment at the AdjointExpr site)

Adjoint expansions must not create an internal temp that is consumed by an
FBIP-capable unary op (Relu/Sigmoid/Tanh/Gelu/Silu/Exp/Log/Sqrt/Abs/Neg) and
then read again — the unary op will alias-and-clobber it at refcount 1. Keep
temps single-use after a unary consumer, or fuse the expansion into one FFI.
Binary ops are safe (their FBIP is relinquish-flag-gated, and source-AD lowers
with flags=0).
