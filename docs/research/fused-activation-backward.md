# Fused Activation Backward — SiLU (Milestone C · p4, slice 2)

Change: `crates/nsl-runtime/src/{cuda/kernels.rs,tensor/activation.rs}`,
`crates/nsl-codegen/src/{source_ad.rs,wengert_lower.rs,builtins.rs}`.

## 1. Problem

Source-AD lowers each `AdjointExpr` into a sequence of concrete PrimalOps, and
every PrimalOp becomes one runtime FFI = one GPU kernel launch + one device
tensor. `AdjointExpr::SiluBackward` expanded to **six** ops —

```
s   = Sigmoid(x)          # kernel 1
t1  = Sub(1, s)           # kernel 2   (1 - s)
t2  = Mul(x, t1)          # kernel 3   (x·(1-s))
t3  = Add(1, t2)          # kernel 4   (1 + t2)
t4  = Mul(s, t3)          # kernel 5   (s·t3)
out = Mul(grad, t4)       # kernel 6   (grad·t4)  => grad·σ(x)·(1 + x·(1-σ(x)))
```

— six launches and five parameter-shaped intermediates, once per SiLU per layer
per backward. Purely elementwise, so it collapses to one kernel.

## 2. The change

A new FFI `nsl_tensor_silu_backward(grad, x)` computes the whole expression in
one launch. `AdjointExpr::SiluBackward` now emits a single
`Passthrough("silu_backward")` with inputs `[y_bar, x]`, lowered by
`wengert_lower.rs` to the FFI (the same mechanism as `mse_backward`/`l1_backward`).

- **GPU f32**: `SILU_BACKWARD_SRCAD_F32_PTX` reproduces source-AD's exact
  operation order, with the sigmoid computed by the *identical* instructions as
  `SIGMOID_F32_PTX` (`neg`, `mul` by log2(e), `ex2.approx`, `add 1.0`,
  `rcp.approx`).
- **CPU f32/f64**: the same order in a direct loop.

## 3. Bit-exactness

The result is **byte-identical** to the six-op path, so training output does not
shift. Two subtleties:

1. **Operation order matters.** The pre-existing `SILU_BACKWARD_F32_PTX` (used
   only by tape-AD) computes the algebraically-equal but differently-grouped
   `grad·(s + s·x·(1-s))`, which rounds differently. This PR deliberately does
   **not** reuse it; the new kernel matches source-AD's `grad·s·(1 + x·(1-s))`
   grouping element-for-element.
2. **`.rn` blocks fma-contraction.** Within one kernel, ptxas would by default
   contract the register-dependent `t2 = x·t1` mul feeding the `t3 = 1 + t2` add
   into a single-rounding `fma` (~1 ULP drift, exactly as in slice 1). The
   derivative ops use `sub.rn.f32`/`mul.rn.f32`/`add.rn.f32` — `.rn` is the
   default round-to-nearest but forbids contraction, so each op rounds
   independently, matching the six separate kernels (whose intermediates round to
   f32 through global memory). The sigmoid ops stay plain `.f32` to match
   `SIGMOID_F32_PTX` byte-for-byte (they contain no contractible mul+add pair).

## 4. Validation

- **Bit-exact unit tests** (`silu_backward_{cpu_f64,gpu_f32}_matches_6op`):
  compare the fused FFI against a hand-built six-op reference
  (`sigmoid`+`sub`+`mul`+`add`+`mul`+`mul`, `flags=0` to preserve operands) with
  `to_bits()` equality. GPU test uses n=257 (256-block tail).
- **E2E correctness**: a `silu` MLP trained 3 steps with `--source-ad` (fused
  path) agrees with the tape-AD path (an independent silu-backward impl) to
  ~1e-4 — gradients flow correctly through the fused backward (w1, upstream of
  the SiLU, moves 64→46.3).
- **Regression**: 804 `nsl-runtime` + 2905 `nsl-codegen` lib tests pass;
  geglu/reglu activation e2e passes. Independent review clean (PTX, ownership,
  wiring, CPU/GPU bit-exactness).

## 5. Follow-ups (same pattern)

`GeluBackward` (7 ops, `GELU_BACKWARD_F32_PTX` exists tape-AD-only, same
reordering caveat), `SigmoidBackward` (3), `TanhBackward` (3) fuse identically.
Norm backward (`RmsNormGammaBackward` ~8, `LayerNormBackward` ~17) is the larger
win but changes reduction order → tolerance-validated, a later p4 slice.
