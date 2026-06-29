# NSL Hardware Status

This directory makes NSL's hardware claims **traceable to what has actually
been validated**, versus what is analysis-only or built-but-untested. The goal
is that a hardware engineer can tell exactly which GPU/accelerator claims are
trustworthy today.

See also: [`STATUS.md`](../../STATUS.md) for the overall stable/beta/experimental
tiering, and [`cuda_status.md`](cuda_status.md) / [`fpga_status.md`](fpga_status.md)
for per-backend detail.

## Validation vocabulary

- **Validated** — kernel/codegen ran on the listed silicon with numerical
  results checked against a CPU reference (golden comparison). Commit/date noted.
- **Built** — codegen emits artifacts and structural/PTX-parse tests pass, but
  it has **not** been run end-to-end on real hardware.
- **Analysis-only** — the compiler reasons about the target (roofline, register
  budget, occupancy) but does not emit a validated kernel for it.

## Tested-on matrix

| Backend          | Target(s)                         | Status        | Evidence |
|------------------|-----------------------------------|---------------|----------|
| CPU (Cranelift)  | x86-64, aarch64                   | Validated     | Workspace unit/integration tests (every CI run) |
| CUDA / PTX       | NVIDIA RTX 5070 Ti (sm_120)       | Validated     | FlashAttention-v2 D pre-pass **bit-exact** vs CPU (CHANGELOG, 2026-05-20) |
| CUDA / PTX       | sm_80 / sm_90 (A100/H100 class)   | Built / analysis-only | Kernels target sm_80+; roofline modeled for H100-SXM. Not yet golden-validated here. |
| AMDGPU / ROCm    | —                                 | Built (untested) | KIR backend compiles; no hardware validation. |
| Metal            | —                                 | Built (untested) | KIR backend compiles; no hardware validation. |
| WGSL / WebGPU    | —                                 | Built (untested) | KIR backend compiles; no hardware validation. |
| FPGA / Verilog   | (lint/sim/synth only)             | Experimental  | Verilator + Yosys nightly job; see [`fpga_status.md`](fpga_status.md) |

> The single concrete GPU validation point today is **RTX 5070 Ti (sm_120)**,
> where the FlashAttention-v2 D pre-pass matched the CPU reference bit-exactly
> across four shape configurations. Treat every other GPU target as
> *built/analysis-only* until a golden comparison is recorded here.

## What "validated" requires before a row flips to Validated

A backend/target moves to **Validated** only when there is a committed test that:

1. Runs the kernel on the named silicon (driver/toolkit version recorded).
2. Compares output against a CPU reference within a stated tolerance (ideally
   bit-exact).
3. Records register/SMEM usage and target SM where relevant.

Until then it stays **Built** or **Analysis-only**, and `STATUS.md` keeps the
feature in Beta or Experimental.
