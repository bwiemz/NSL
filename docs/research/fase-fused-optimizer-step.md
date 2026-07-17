# Fused FASE-Deferred Optimizer Step (Milestone C · p9)

Change: `crates/nsl-runtime/src/{cuda/kernels.rs,cuda/mod.rs,fase_step.rs,args.rs,lib.rs}`,
`crates/nsl-codegen/src/{stmt_fase.rs,builtins.rs}`,
gate `crates/nsl-cli/tests/fase_fused_step_gpu_gate.rs`.

## 1. Problem

`fase_emit_final_step` interprets the FASE-Deferred AdamW `UpdateProgram`
op-by-op through generic tensor FFIs: **~15 kernel launches + 3 DtoD copies +
~10 transient alloc/frees per parameter per optimizer step** (measured map in
the p8/p9 scout). For a real model (N ≈ 200–400 param tensors) that is
thousands of launches at every grad-accumulation boundary — the launch storm p8
(CUDA graphs) would otherwise have had to capture. Fusing it is both faster and
simpler than graphing it, and each element's state (m, v, θ, mp) is touched
ONCE instead of ~15 read-modify-write round-trips through global memory.

## 2. The change

One FFI — `nsl_fase_fused_adamw_step(theta, m, v, m_partial, lr, β₁, 1-β₁, β₂,
1-β₂, ε, wd, bc1_inv, bc2_inv)` — performs the whole update in a single kernel
launch (GPU) or single fused loop (CPU), **bit-exact** with the interpreted
program:

```
m  = rn(rn(β₁·m) + rn((1-β₁)·mp))          [ScalarMulAdd]
v  = rn(rn(β₂·v) + rn((1-β₂)·rn(mp·mp)))   [SquaredAccumulate, decay-then-add]
m̂  = rn(m·bc1_inv);  v̂ = rn(v·bc2_inv)     [ScalarMulByBc ×2]
t  = rn(sqrt(v̂) + ε)                       [SqrtPlusEps]
u  = m̂ / t                                 [Div]
θ  = rn(θ + rn(rn(-lr·u) + rn(-lr·wd·θ)))  [Update; wd arm skipped when wd==0]
```

Bit-exactness mechanics (the #383/#384/#385 doctrine):
- **GPU** (`FASE_FUSED_ADAMW_STEP_F32_PTX`): `.rn` on every mul/add (blocks
  ptxas fma-contraction so each op rounds like the decomposed kernel that
  stored to memory), `sqrt.rn.f32` and **`div.approx.f32`** — the exact
  instructions the decomposed SQRT/DIV kernels use. Scalars converted f64→f32
  at the FFI boundary with the same `as f32` every `nsl_tensor_*_scalar` op
  performs; `-lr` and `(-lr)·wd` computed in f64 first, exactly like codegen's
  `f64const` arguments.
- **CPU**: the decomposed CPU ops compute natively in the tensor dtype, so the
  FFI mirrors both uniform configurations — f64 loops for f64 tensors and
  f32-native loops (scalars `as f32`) for f32 tensors. (The FASE equivalence
  fixtures train **uniform-f32 CPU** models — a configuration the first cut
  loud-rejected; the interpreted ops accept it, so the fused path must too.)
- The `SqrtPlusEps` "+0.0 defensive copy" is an identity for v̂ ≥ +0 (v is a
  non-negative EMA of squares) and is skipped in registers.

Codegen (`stmt_fase.rs`): `match_adamw_program` **structurally matches** the
exact 7-op AdamW/Adam program shape and extracts its scalars; on a match (and
`bc_params` present), `fase_emit_final_step` emits the single FFI call instead
of interpreting, then falls through to the unchanged shared tail (zero
m_partial, envelope copy-backs). Any change to the program emitter fails the
match and falls back to the interpreted path — no silent drift. Because the
fused call runs on the **envelope-resolved working pointers**, the CPDT
reduced-precision and optimizer-state-offload envelopes compose unchanged, and
all three final-step call sites (monolithic ± two-phase clip, unified dispatch)
are covered by the one insertion. SGD/Lion/mismatched programs keep the
interpreted path.

Kill-switch: `NSL_FASE_FUSED_STEP=0`, read at COMPILE time (the
`NSL_PHASE_TIMING` doctrine — `nsl run` compiles and executes in one process).
Observability: an always-live launch counter with an
`NSL_FASE_FUSED_COUNTER=1` atexit report (`[fase-fused] optimizer fused-step
launches: N`) plus an in-process `nsl_fase_fused_step_count()` getter — the
WRGA launch-counter pattern, used by the gate's anti-vacuity check.

## 3. Validation

- **Unit differentials** (`fase_step.rs`): fused vs a reference reproducing the
  emitted IR's exact FFI choreography (flags, relinquish, copy_data), n=257
  (block-tail), wd≠0 and wd=0 — `to_bits()` equality on θ/m/v/mp for **CPU f64,
  CPU f32, and GPU f32** (RTX 5070 Ti).
- **Differential gate** (`fase_fused_step_gpu_gate.rs`, campaign pattern): the
  same FASE-Deferred AdamW GPU training run under `NSL_FASE_FUSED_STEP=0` vs
  default, `--source-ad --deterministic` — **bit-identical trained-parameter
  sums**, with anti-vacuity (fused arm reports 4 fused launches = 2 windows × 2
  params; kill-switch arm reports 0).
- **Semantics**: the full `fase_numerical_validation` suite (15 tests) passes
  with the fused path engaged — including the deferred ≡ full-AdamW equivalence
  pipelines and the v_t windowed-exactness discrimination test.
- **Regression**: 822 `nsl-runtime` (cuda) + full `nsl-codegen` suite green.
- **Performance** (`NSL_PHASE_TIMING`, 8×[256×256] MLP, 10 optimizer windows,
  5070 Ti): `[phase] opt=` mean **727µs → 82µs (~8.9×)**; the interpreted path
  costs ~15 launches + 3 DtoD + ~10 pool alloc/frees per param vs one launch
  and zero transients fused. (Informational, per repo perf-claim policy.)

## 4. Deferred / follow-ups

- **N→1 multi-tensor batching**: one launch for ALL params via a device
  pointer/offset table (binary-search indexing). The per-param fused kernel is
  the math core for it; the table infra also serves Milestone D's ZeRO
  sharding. With N launches already ~85µs for 8 params, the remaining win is
  modest until N grows large.
- **Grad-clip Phase A** (per-param `sum_sq` + DtoH) is a separate N-launch
  pre-pass a multi-tensor reduction could fold later.
- A broadcast-add **bias parameter** in a FASE-Deferred model aborts in
  `nsl_tensor_add_inplace` (len mismatch) on BOTH interpreted and fused arms —
  pre-existing, unrelated to p9 (the gate fixture documents it; needs its own
  fix).
- SGD/SgdMomentum recipes could get the same treatment (3–5 ops each).
