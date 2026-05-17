# Tier B SMEM Probe — Findings

**Probe spec:** §2 of `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`
**Sweep:** 3 static-sizes {256, 1024, 4096} × 3 dynamic-sizes {16 KB, 64 KB, 96 KB} × 2 architectures {sm_80, sm_120} = 18 configurations.

## 2026-05-13 — initial run

**Hardware / toolkit at run time:**

- Primary GPU: NVIDIA GeForce RTX 5070 Ti (sm_120, Blackwell)
- Secondary GPU: none
- CUDA toolkit: 13.2, V13.2.51 (built 2026-03-02)
- NVIDIA driver: 591.86
- Host OS: Windows 11 Home 10.0.26200

**Probe correctness fixes applied during this run** (before the final sweep produced a clean signal):

1. **`%tid` register-name collision.** The probe initially declared `.reg .u32 %tid` and then read `mov.u32 %tid, %tid.x` — ptxas parses the `.x` on the source operand as a video-selector against the user register `%tid` (not against the built-in special register). Renamed the local copy to `%t`.
2. **PTX `.version` was below sm_120 floor.** The probe initially declared `.version 7.0`. Standalone `ptxas -arch=sm_120` accepts this, but the driver's JIT (CUDA 13.2 / driver 591.86) rejects with `PTX .version 8.5 does not support .target sm_120`. sm_120 was introduced in PTX 8.7 (CUDA 12.8); bumped the probe to `.version 8.7` (backward compatible with sm_80).
3. **Dynamic-shmem opt-in missing.** Both sm_80 and sm_120 cap default dynamic shmem at 48 KB per CTA. Probe configs of 64 KB / 96 KB launched without `cuFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, M)` were rejected with `invalid argument`. Added the opt-in immediately before each launch.
4. **Generic vs `.shared` state-space mismatch.** After `cvta.shared.u64`, the address is in the generic state space; the probe initially used `st.shared.u8` / `ld.shared.u8` on those generic addresses. Switched to `st.u8` / `ld.u8` (generic).
5. **Verbose JIT-log capture.** Initial probe used `cuModuleLoadData`, which collapses every JIT failure to the generic `a PTX JIT compilation failed`. Switched to `cuModuleLoadDataEx` with `CU_JIT_ERROR_LOG_BUFFER` so the actual ptxas message is surfaced through the `LaunchError` row.

All five fixes are in the same probe file (`crates/nsl-codegen/tests/tier_b_smem_probe.rs`) and committed on the worktree branch in the same commit as this findings update.

**Sweep result:**

```text
sm_ 80  N=  256  M= 16384  Pass
sm_ 80  N=  256  M= 65536  Pass
sm_ 80  N=  256  M= 98304  Pass
sm_ 80  N= 1024  M= 16384  Pass
sm_ 80  N= 1024  M= 65536  Pass
sm_ 80  N= 1024  M= 98304  Pass
sm_ 80  N= 4096  M= 16384  Pass
sm_ 80  N= 4096  M= 65536  Pass
sm_ 80  N= 4096  M= 98304  LaunchError("cuFuncSetAttribute: invalid argument")
sm_120  N=  256  M= 16384  Pass
sm_120  N=  256  M= 65536  Pass
sm_120  N=  256  M= 98304  Pass
sm_120  N= 1024  M= 16384  Pass
sm_120  N= 1024  M= 65536  Pass
sm_120  N= 1024  M= 98304  Pass
sm_120  N= 4096  M= 16384  Pass
sm_120  N= 4096  M= 65536  Pass
sm_120  N= 4096  M= 98304  LaunchError("cuFuncSetAttribute: invalid argument")
```

**Outcome counts:** 16 Pass · 2 LaunchError · 0 ValueCorruption.

**Outcome row selected from spec §2.4 five-outcome decision matrix:** **"Passes all probe configs"** (Option 1).

The two `LaunchError` rows are not corruption and not a `.shared`-layout failure mode. Both fire on the same `N=4096, M=98304` cell on both architectures: 4096-byte static plus 98304-byte dynamic asks the device for ≥100 KB per-CTA dynamic SMEM via `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES)`. On consumer Blackwell (RTX 5070 Ti) and the sm_80-targeted kernel running on the same device, the per-CTA max-optin attribute is below the requested 96 KB-of-dynamic-plus-overhead, so the driver rejects the opt-in as "invalid argument" *before* launch. Sentinel bytes are never written or read in these rows. This is a device-budget ceiling, not a layout-corruption signal. Every config that stays inside the device budget passes; no `ValueCorruption` was observed on either architecture.

**Decision:** **Option 1 confirmed safe.** Proceed with the tail-embedded range tables in PCA Tier B as specified; leave forward's `.shared .align 4 .b8 seg_smem[N]` decl untouched and let Tier B ride the existing `.extern .shared shmem[]` via `smem_layout::tier_b_range_table_offset`. No pre-Tier-B forward-kernel SMEM refactor (option 2) is required on Blackwell sm_120 for this driver / toolkit / hardware combination.

**Rationale notes:**

- The "Blackwell `CUDA_ERROR_ILLEGAL_ADDRESS` on mixed static + extern `.shared`" framing from the 2026-05-02 spec's pause was driver-version-bound conjecture. On CUDA 13.2 / driver 591.86 with this RTX 5070 Ti the failure mode does not reproduce: every (N, M) combination that fits the device budget reads back both `0xAA` (from static) and `0xBB` (from extern) on every thread, on both sm_80 and sm_120.
- The `LaunchError` rows are documented in the design's decision matrix as not-corruption: spec §2.4 distinguishes `LaunchError` from `ValueCorruption`. The "fires only above a (N, M) threshold" row would apply if launches succeeded for low-shmem cases and corrupted at higher cases; that is not what we see. Here the failures happen at config time (`cuFuncSetAttribute`), not at launch and not at access.
- Per spec §2.3 the probe also tested the smallest static-array size (`N=256`) all the way through the largest dynamic-shmem cell that the device admits; no size-dependent corruption emerged.
- Tier B's actual production allocation is well below the failing 100 KB cell. Per the v1 forward kernel as shipped (PR #168), `ptxas -arch=sm_75 -v` reports **41984 bytes smem** for the canonical Tier B kernel — half the device's max optin, well below the failure cell.

## Re-run triggers

The probe is re-run when any of:

- **CUDA toolkit major version bump** (e.g., 13.x → 14.x). Driver behavior on mixed `.shared` allocations may shift.
- **New target architecture added to NSL's supported matrix** (e.g., sm_130 when it ships). Probe sweeps the new architecture.
- **Production deployment surface reports `CUDA_ERROR_ILLEGAL_ADDRESS`** on a Tier B kernel that previously passed CI. Indicates the probe's prior outcome no longer holds.

## Re-run log

(Future dated entries appended here. Each entry includes: date, trigger reason, hardware/toolkit, sweep result, outcome row, any decision changes.)
