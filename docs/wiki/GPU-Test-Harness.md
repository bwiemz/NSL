<!-- owner: @bwiemz -->

# GPU Test Harness

CI has no CUDA device, so every GPU numerical test in the codebase is
`#[ignore]`d and gated on the `cuda` Cargo feature. A default `cargo test`
therefore skips them, and "it works on the GPU" has, historically, been an
untested claim. The **GPU test harness** (`tools/gpu-test.ps1`) is the
documented, reproducible manual gate that closes that hole: it runs the ignored
`--features cuda` suites on real silicon, stamps provenance, and reports a clear
PASS/FAIL with a non-zero exit on any failure.

This is Phase 0.1 of the "make pretraining actually work" plan — the
prerequisite for trusting any CSHA / PCA / FASE GPU correctness claim.

## Prerequisites

| Requirement | Check | Notes |
|---|---|---|
| NVIDIA GPU + driver | `nvidia-smi` | Must report a usable device. |
| CUDA Toolkit | `CUDA_PATH` set | Needed to link `cudarc` (dynamic-linking). |
| `cuda.lib` (Windows) | `%CUDA_PATH%\lib\x64\cuda.lib` | The MSVC import library the linker needs. |
| Rust toolchain | `rustc --version` | Workspace pins toolchain `1.95.0`. |
| `NSL_SKIP_CUDA_TESTS` **unset** | — | If set, every CUDA test early-returns as a "pass". The harness **refuses to run** while it is set, so a skip can never masquerade as green. |

The preflight verifies all of the above and fails loud with remediation before
building anything.

### Reference hardware

The canary set below is confirmed green on:

- **GPU:** NVIDIA GeForce RTX 5070 Ti (sm_120), 16 GB
- **CUDA:** Toolkit 13.2, driver 13.3
- **rustc:** 1.95.0

> CUDA 13.x removed `cuCtxCreate`; the runtime uses
> `cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent` with thread-local contexts.
> If you see `CUDA_ERROR_*` at launch, verify the driver/toolkit pair.

## Usage

```powershell
# The acceptance gate: run the curated known-green canary set.
pwsh tools/gpu-test.ps1 -Canary

# List the canary manifest without building.
pwsh tools/gpu-test.ps1 -ListCanary

# Run one test by exact name in a specific binary.
pwsh tools/gpu-test.ps1 -Filter x_prepass_matches_cpu_reference -TestBinary tier_b1_prepass_gpu

# Substring match instead of exact.
pwsh tools/gpu-test.ps1 -Filter prepass -TestBinary tier_b1_prepass_gpu -Loose

# Run every ignored CUDA test in a package (broad sweep; long).
pwsh tools/gpu-test.ps1 -All -Package nsl-codegen
```

Key switches: `-Package <name>` (default `nsl-runtime`; the other GPU-heavy
crate is `nsl-codegen`), `-TestBinary <name>`, `-TestThreads <n>` (default `1`
— GPU tests must serialize the device), `-Release`, `-NoPreflight`.

Each run tees full output to `target/gpu-test-logs/gpu-<timestamp>-<label>.log`.

### Shell-agnostic equivalent

The harness is a convenience wrapper. The underlying invocation is portable and
can be run from any shell (or a future self-hosted CI runner):

```bash
cargo test -p nsl-runtime --features cuda --test tier_b1_prepass_gpu \
    -- --ignored --nocapture --test-threads=1
```

> **PowerShell 5.1 note:** the script runs `cargo` under `cmd` redirection on
> purpose. Piping a native command through `2>&1` in Windows PowerShell 5.1
> wraps each stderr line (including cargo's own warnings) as a
> `NativeCommandError`, which — under `$ErrorActionPreference='Stop'` — aborts
> the run. Redirecting inside `cmd` keeps `$LASTEXITCODE` honest.

## Expected output (canary)

```text
======================================================================
GPU test harness - preflight & provenance
======================================================================
  timestamp : 2026-07-07 11:14:55 -04:00
  gpu       : NVIDIA GeForce RTX 5070 Ti, 610.62
  cuda      : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2
  nvcc      : Cuda compilation tools, release 13.2, V13.2.51
  rustc     : rustc 1.95.0
  ...
  preflight : OK
...
test w_prepass_matches_cpu_reference ... [w-prepass] all 65536 elements bit-exact match ... ok
test x_prepass_matches_cpu_reference ... [x-prepass] max_abs=9.7656e-4 mismatches(>1e-2)=0 ... ok
...
PASS: all 2 run(s) green on GPU.
```

## The canary set

The canary (`tools/gpu-canary.txt`) is the small set of GPU tests that make a
**real numerical claim against a CPU reference** and are confirmed green on the
reference hardware. It is the acceptance gate: if the canary passes, the harness
is working. Current members — the CSHA Tier B.1 "D pre-pass" kernels:

| Test | Claim |
|---|---|
| `tier_b1_prepass_gpu::w_prepass_matches_cpu_reference` | narrow + col-major chunkify, **bit-exact** vs CPU |
| `tier_b1_prepass_gpu::x_prepass_matches_cpu_reference` | RMSNorm + narrow + chunkify, tol `5e-2` (`rsqrt.approx`) |
| `csha_cuda_backward::t6_3_smoke_single_config` | CSHA fused backward, smoke scope (heads=1, seq=bq=32, hd=32): all 7 grads vs CPU ref within Tier-A tol. **Single-tile only** — see below. |

**Growing the set:** as later phases un-ignore and green their GPU numerical
gates (CSHA backward, PCA correctness, FASE optimizer step, …), add each
now-green test to `tools/gpu-canary.txt`. Membership rule: a real CPU-reference
numerical assertion, observed green — not a launch/`rc=0` structural check.

## A green CUDA test is not always full validation

A passing `--features cuda` test can still leave real shapes unvalidated. Two
patterns to watch for before treating "green" as "correct at production shapes":

1. **Structural-only tests.** Some tests assert only that the kernel launches
   (`rc=0`) and produces finite, correctly-shaped outputs — not equivalence to a
   CPU reference. The `csha_cuda_launch_fused` / `csha_cuda_launch_classic`
   launch tests are of this kind. Green means "it ran", not "it's numerically
   right". Do not add these to the canary.

2. **Asserted-refusal tests.** A test can pass by asserting the kernel
   *refuses* an unsupported configuration. `csha_cuda_backward` is the important
   example: its smoke case (`t6_3_smoke_single_config`, heads=1, seq=bq=32) is a
   **real numerical gate and is in the canary**, but its multi-tile cases
   (seq=128 > block_q=32) assert that the fused backward is *refused* —

   > multi-tile backward … is not implemented for fused-projection kernels:
   > cross-launch dW accumulation (f16 overwrite, needs f32 scratch RMW like
   > dk/dv) and the dx chain's partial dK/dV tiles would produce
   > silently-wrong gradients — refusing

   So a green `csha_cuda_backward` means **single-tile backward is correct and
   multi-tile backward is safely refused** — *not* that CSHA backward is
   validated at production shapes. Implementing the multi-tile accumulation to
   replace that refusal is **Phase 1.1** of the pretraining plan; once it lands,
   its numerical gate joins the canary.

When you un-ignore or enable a subsystem's numerical gate, confirm it green
through this harness at **real** shapes — not just the smoke config — before
treating the subsystem as validated.

## CI

There is deliberately no GPU in CI (see [Testing-Strategy](Testing-Strategy.md)).
Until a self-hosted CUDA runner exists, this harness is the enforced manual gate:
**run `pwsh tools/gpu-test.ps1 -Canary` (plus the specific suites your change
touches) before merging any kernel-level change, and paste the PASS summary into
the PR.**
