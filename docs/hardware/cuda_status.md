# CUDA / PTX Backend Status

NSL emits PTX for custom `kernel` ops and for the FlashAttention/CSHA codegen
paths, loaded at runtime through cudarc (`nsl_kernel_launch`, `nsl_test_cuda_*`).
This document records the **public contract** of that path: what is validated,
on what silicon, and what guarantees do (and do not) hold.

> Tier: **Beta** (see [`STATUS.md`](../../STATUS.md)). The PTX path works and is
> validated on specific hardware, but it is not yet a cross-vendor or
> cross-architecture guarantee.

## Supported SM versions

| Capability                          | Minimum SM | Notes |
|-------------------------------------|-----------|-------|
| FlashAttention-v2 D pre-pass        | sm_80     | Row-per-lane schedule; no SMEM; HBM-bandwidth bound |
| MMA / tensor-core matmul fragments  | sm_80     | `matmul_mma` fragment loads |
| General custom `kernel` ops         | sm_70+    | Scalar/elementwise PTX |

Validated silicon: **NVIDIA RTX 5070 Ti, sm_120** (2026-05-20).

## Numerical contract

- **Golden comparison against CPU reference is the correctness gate.** The
  FlashAttention-v2 D pre-pass matched the CPU reference **bit-exact
  (max_abs = 0.0)** at four configurations: `(b=1,h=1,s=32,hd=32)`,
  `(1,1,64,32)`, `(1,2,96,64)`, `(2,1,128,128)`.
- Where bit-exactness is not achievable (reductions, fast-math), kernels declare
  a tolerance; the staged tolerances for attention backward are 5e-3 / 2e-2 /
  4e-2 by stage.
- Determinism: NSL provides `nsl check --deterministic`. Kernels that use
  atomics or non-deterministic reduction order are flagged; the dQ backward
  kernel is explicitly designed **without `atomicAdd`** (register-resident
  accumulation) for reproducibility.

## Known gaps (do not assume these work)

These are tracked honestly so users don't over-trust the backend:

- **dQ / dK-dV backward kernels are structural scaffolds, not yet data-mobile.**
  As of the v0.9.0 line, the dQ-kernel emitter ships cp.async loads, HBM address
  derivation, SMEM scatter, and loop back-edges as PTX comments rather than
  emitted instructions. A launched dQ kernel would read uninitialized SMEM.
  GPU validation of dQ is gated on the "Phase 2.5" data-mobility work.
- sm_80/sm_90 (A100/H100-class) are **targeted and roofline-modeled** but the
  golden-validation rows are not yet recorded on this repo's hardware.
- Multi-GPU / NCCL collectives live in the **Experimental** distributed
  subsystem — not covered by this Beta contract.

## What to record when validating new silicon

When you validate a kernel on new hardware, add a row to
[`README.md`](README.md) and note:

- GPU model + SM (compute capability)
- CUDA toolkit + driver version
- Register usage / SMEM per kernel, target occupancy
- Stream/synchronization assumptions
- Golden tolerance vs CPU reference (bit-exact preferred)
