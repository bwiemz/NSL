# FASE Fused Scaled-Add Accumulate (Milestone C · p4, slice 1)

Change: `crates/nsl-runtime/src/{cuda/kernels.rs,cuda/mod.rs,tensor/arithmetic.rs}`,
`crates/nsl-codegen/src/{builtins.rs,stmt_fase.rs}`.

## 1. Problem

FASE Deferred keeps a persistent first-moment accumulator `m_partial` per
parameter and folds each micro-batch's gradient into it:
`m_partial += accum_scale * grad` (`accum_scale = 1/N`). `fase_emit_accumulate`
emitted this as **two** GPU kernels plus a full parameter-sized temporary:

```
scaled = nsl_tensor_mul_scalar(grad, accum_scale)   // launch 1 + temp
nsl_tensor_add_inplace(m_partial, scaled)           // launch 2
free(scaled)
```

This runs once per parameter per micro-batch — ~217 calls/step for a 500M model
at grad_accumulation=8, i.e. ~434 elementwise launches and ~217 param-sized
scratch tensors per optimizer step, a dominant hidden cost of the backward.

The source-AD backward "unit of fusion" is one PrimalOp = one runtime FFI = one
GPU kernel launch + one output tensor, so collapsing this pair into one FFI is a
direct, mechanical win. It is the *conservative, bit-exact* half of p4's
"matmul + FASE-accum epilogue" target (the beta=1 GEMM fold is deferred — it is
not bit-exact and needs a flattened-batch dW restructuring, see §5).

## 2. The change

A new FFI `nsl_tensor_scalar_mul_add_inplace(m, g, s)` computes `m += s*g` in
place, leaving `g` intact, in ONE launch with no temporary. `fase_emit_accumulate`
calls it in both the resident and the P3-offload path, replacing the
mul_scalar + add_inplace pair (and dropping one of the two frees).

- **GPU f32**: `SCALAR_MUL_ADD_INPLACE_F32_PTX` — `mul.rn.f32` then `add.rn.f32`.
- **CPU f32 / f64**: the same two-rounding loop in Rust.
- **Everything else** (dtype/device/shape mismatch, non-contiguous, or fp16/bf16
  whose materialized-intermediate rounding a single kernel cannot reproduce):
  falls back to the exact decomposed `mul_scalar(flags=0)` + `add_inplace` + free.

## 3. Bit-exactness — the load-bearing `.rn`

The result must be byte-identical to the two-kernel path so FASE training output
does not shift. The two kernels **double-round**: `mul.f32` writes
`round_f32(g*s)` to global memory, which is reloaded and fed to `add.f32` →
`round_f32(m + round_f32(g*s))`.

A naive fused kernel with plain `mul.f32; add.f32` is **not** bit-exact: ptxas,
by default (`--fmad=true`), CONTRACTS a register-dependent multiply+add into a
single `fma.f32` (one rounding). Measured: the contracted result differed from
the two-op path by up to 1 ULP (e.g. `0.55075734` vs `0.55075729`).

The fix is the explicit `.rn` (round-to-nearest) modifier on
`mul.rn.f32`/`add.rn.f32`. `.rn` is numerically the default rounding, but writing
it explicitly forbids ptxas from contracting the pair — each op rounds
independently, reproducing the two-kernel double-rounding element-for-element.
(This is the PTX equivalent of C's `__fmul_rn`/`__fadd_rn`.) On the CPU side,
Rust never emits an FMA for `let t = g*s; m += t;` — it does not set LLVM's
`contract` fast-math flag (FMA requires an explicit `f32::mul_add`) — so the two
statements are guaranteed two roundings.

## 4. Validation

- **Bit-exact unit tests** (`arithmetic.rs`): `scalar_mul_add_inplace_*_matches_two_op`
  for CPU f64, CPU f32, and GPU f32 build random `m`, `g` and assert the fused
  FFI is bit-identical (`to_bits()` equality) to `mul_scalar` + `add_inplace`.
  The GPU test uses n=257 (not a multiple of the 256 block) to exercise the tail
  guard, and is what caught the FMA-contraction issue.
- **End-to-end**: the FASE deferred suite trains real fixtures through the fused
  accumulate — `fase_numerical_validation` (15, FASE≡AdamW/SGD numerical
  equivalence), `fase_mixed_clip_equivalence` (16), plus smoke/dispatch/phase2 —
  all pass. 802 `nsl-runtime` cuda lib tests pass.

## 5. Deferred (p4 phase 2+)

- **beta=1 GEMM fold** (`m_partial = accum_scale·(xᵀ@dY) + 1.0·m_partial` in one
  cuBLAS sgemm): NOT bit-exact (TF32 default + different intermediate rounding),
  and the raw dW is batched `[B,in,out]` that `reduce_to_shape` sums — a plain
  GEMM cannot both contract and match `m_partial`'s shape without restructuring
  dW into a flattened-batch contraction (valid only for non-broadcast params).
  A tolerance-validated follow-up.
- **Optimizer-step `ScalarMulAdd`** (`dst = a·src + b·bsrc`, e.g.
  `M = β₁·M + (1-β₁)·MPartial`): a two-scaled-term combine, once per step — a
  different fused kernel, belongs to p9 (fused multi-tensor optimizer).
- **Activation / norm backward fusion** (SiluBackward 6→1, RMSNorm/LayerNorm
  backward ~25→1): the larger structural wins; separate p4 slices (norm backward
  changes reduction order → tolerance-validated).
