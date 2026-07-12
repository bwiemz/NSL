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

- **dK / dV backward numerics are not yet GPU-confirmed** (dQ is). The Tier B.2
  backward emitters (dQ, dK/dV, projection) are data-mobile as of the Phase 3
  work: they emit real `cp.async` loads, HBM address derivation, SMEM scatter,
  and loop back-edges — the earlier PTX-comment "scaffold" stage is done, so the
  old "a launched dQ kernel reads uninitialized SMEM" caveat no longer holds.
  Concretely: **dQ** is GPU-validated for full tiles at `head_dim ∈ {32,64,128}`
  against the CPU-naive reference (see `flash_attention_v2/tier_b2/backward/dq.rs`
  module doc). **dK/dV** are structurally + `ptxas`-validated and launchable, but
  their GPU-numerical parity tests are still `#[cfg(feature = "cuda")]` +
  `#[ignore]` (manual GPU-box runs) — treat dK/dV numerics as unconfirmed on
  silicon until those are lifted (`dkdv.rs` module doc).
- **The full 7-gradient hybrid backward** (`d_prepass → dq → dkdv → proj`,
  dispatched through `nsl_flash_attention_csha_backward`) is synthesized and
  wired; its all-gradient parity gate,
  `crates/nsl-codegen/tests/tier_b2_full_backward_cpu_reference.rs`, is a manual
  GPU gate (`cuda` + `#[ignore]`) that pins a narrow config (`head_dim ∈ {64,128}`,
  heads=1, seq=block_q, batch=1, causal, sm_80). No broader golden-on-silicon run
  is recorded in-repo beyond the D pre-pass (sm_120) and dQ.
- sm_80/sm_90 (A100/H100-class) are **targeted and roofline-modeled** but the
  golden-validation rows are not yet recorded on this repo's hardware.
- Multi-GPU / NCCL collectives live in the **Experimental** distributed
  subsystem — not covered by this Beta contract.

## Golden CPU-reference test coverage

GPU correctness is gated by **CPU-reference golden tests**: a kernel's output is
compared against an independent CPU implementation of the same math. The
pattern splits into two halves so CI stays GPU-free while the GPU path remains
checkable on real silicon:

- **CPU-reference half** — pure-CPU oracle computation, runs on every
  `cargo test` (no GPU). This locks the oracle so it cannot silently drift.
- **GPU half** — `#[cfg(feature = "cuda")]` + `#[ignore]`, launched manually on
  a GPU box, asserts the kernel matches the oracle within the declared
  tolerance (bit-exact where achievable).

Representative tests (search the tree for the current set):

| Test | Oracle | Status |
|------|--------|--------|
| `crates/nsl-codegen/tests/tier_b2_full_backward_cpu_reference.rs` | independent f64 backward, all 7 grads | Phase-3 hybrid gate; GPU-only (`cuda`+`#[ignore]`), narrow config |
| `crates/nsl-codegen/tests/tier_b2_dq_kernel_cpu_reference.rs` | D pre-pass + dQ reference | cuda-gated; GPU parity `#[ignore]` (dQ GPU-validated hd 32/64/128) |
| `crates/nsl-codegen/tests/tier_b2_dkdv_kernel_cpu_reference.rs` | dK/dV CPU reference | CPU analytic case in CI; GPU parity `#[ignore]`+cuda (numerics pending) |
| `crates/nsl-test/.../diagnostic_mode` (`compute_d_for_test`) | CSHA backward D | CPU oracle reusable as a bisection probe |
| FlashAttention-v2 D pre-pass (this doc, "Numerical contract") | rowsum(dO·O) | GPU-validated **bit-exact** on sm_120 |
| `crates/nsl-codegen/tests/ptx_metadata_public_api.rs` | declared regs / SMEM / target SM parsed from PTX text | CI (no GPU; static analysis) |

The static [`ptx_metadata`](../../crates/nsl-codegen/src/ptx_metadata.rs) report
(`nsl ptx-metadata <file.ptx>`) complements these: it surfaces per-kernel
register/shared-memory/target-SM figures from PTX text without a device, so a
reviewer can sanity-check occupancy assumptions against the golden runs.

## What to record when validating new silicon

When you validate a kernel on new hardware, add a row to
[`README.md`](README.md) and note:

- GPU model + SM (compute capability)
- CUDA toolkit + driver version
- Register usage / SMEM per kernel, target occupancy (cross-check with
  `nsl ptx-metadata`)
- Stream/synchronization assumptions
- Golden tolerance vs CPU reference (bit-exact preferred)
