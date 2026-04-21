# Matmul Primitive: cuBLAS Swap for `nsl_matmul_f32`

**Motivation:** WRGA B.3.2 trigger measurement (PR #93 aftermath) revealed the trigger's 106× ratio was inflated by an unoptimized matmul primitive. `nsl_matmul_f32` is a naive sm_52 scalar kernel achieving ~1-2 TFLOPs/s on a 5070 Ti (~15-20× below peak). Per `project_wrga_b32_measurement.md` 2026-04-20 addendum, B.3.2 is re-deferred until matmul-primitive optimization lands and the trigger is re-measured against a clean baseline.

**Goal:** replace the naive kernel with a cuBLAS `cublasSgemm_v2` call via `cudarc::cublas`. Preserve strict f32 semantics (modulo summation-order drift). Smallest intervention that closes the primitive-optimization gap.

**Spec discipline:** follows Appendix B.5 — direct probe via shape-diverse numerical equivalence, not just "a matmul test still passes." Explicit tolerance-relaxation enumeration for tests whose current tight tolerance depends on the naive kernel's summation order.

---

## 1. Verification before dispatch (complete)

- `cudarc` 0.19.4 declares `cublas = ["driver"]` in its Cargo.toml. cuBLAS is accessible via a feature flag addition to both `crates/nsl-codegen/Cargo.toml` and `crates/nsl-runtime/Cargo.toml` (both currently: `features = ["driver", "cuda-version-from-build-system", "dynamic-linking"]`). Add `"cublas"`.
- No new external dependencies. cuBLAS is a C library shipping with CUDA; already transitively present wherever cudarc's `driver` feature works. The architectural principle "no Python, no C++" is not violated — cuBLAS is a pure C API.
- Blast radius: `nsl_matmul_f32` is referenced at exactly two sites — the PTX constant at `crates/nsl-runtime/src/cuda/kernels.rs:306` and the launch site at `crates/nsl-runtime/src/cuda/mod.rs:1354 gpu_matmul_f32`. Every f32 GPU matmul in NSL goes through `gpu_matmul_f32`.

## 2. Intervention

Replace the PTX-launch path in `gpu_matmul_f32` with a `cudarc::cublas` call. Retain the PTX constant initially (gated by a compile-time feature flag or a runtime env var) as a fallback during the rollout, then delete it in a follow-up commit once the swap is stable.

Four implementation concerns the wrapper MUST handle. Each is a known-cuBLAS-integration bug class:

### 2.1 Row-major vs column-major

NSL tensors are **row-major** (the naive kernel indexes A as `[row*K + k]`, B as `[k*N + col]`, output as `[row*N + col]`). cuBLAS is **column-major** by default. Two correct ways to wrap:

- **Option (a):** call `cublasSgemm_v2(transa=N, transb=N, m=N, n=M, k=K, A=B_row, B=A_row, C=C_row)` — swap the operands and dimensions so cuBLAS's column-major interpretation produces row-major output. Standard cuBLAS-from-row-major idiom. No physical transpose of tensor data.
- **Option (b):** call with `transa=T, transb=T` to manually request transposes. Works but costs a kernel pass for each transpose. (a) is preferred.

**Pin option (a).** The wrapper computes `C = A @ B` in row-major by submitting the problem as `C^T = B^T @ A^T` in column-major, which cuBLAS naturally handles.

### 2.2 Alpha/beta scaling

cuBLAS computes `C = alpha * op(A) * op(B) + beta * C`. Pass `alpha = 1.0`, `beta = 0.0` to match the naive kernel's `C = A @ B`.

`alpha` and `beta` are `f32` pointers (cuBLAS expects `*const f32`). Per cuBLAS docs, the pointer mode defaults to `CUBLAS_POINTER_MODE_HOST`, so these are host pointers to f32 values. Allocate on the stack: `let alpha: f32 = 1.0; let beta: f32 = 0.0; cublasSgemm_v2(handle, ..., &alpha, ..., &beta, ...);`.

### 2.3 Handle lifecycle

`cublasHandle_t` is a stateful object that must be created once, reused across calls, and destroyed at program teardown. Three lifecycle patterns:

- **Per-call create/destroy:** simplest, but cublasCreate is ~ms-scale overhead. At 20 matmul calls per train iter, this dominates the speedup — bad.
- **Per-CUDA-context singleton:** one handle per device. NSL currently has one CUDA context globally (via cudarc's driver feature). A lazy-initialized global `OnceLock<CublasHandle>` is sufficient.
- **Per-stream singleton:** one handle per stream. NSL uses the default stream everywhere, so per-context and per-stream collapse to the same thing.

**Pin per-CUDA-context singleton.** Lazy-initialized at first `gpu_matmul_f32` call, destroyed via an `atexit` hook (NSL already has atexit infrastructure — see `crates/nsl-runtime/src/args.rs` for the pattern used by kernel-profiler / launch-counter).

### 2.4 Error translation

cuBLAS returns `cublasStatus_t`. On success: `CUBLAS_STATUS_SUCCESS`. On failure: several error codes (alloc failed, invalid value, internal error, etc). The wrapper translates into NSL's existing error model — for runtime FFI paths this means returning a null i64 tensor pointer (with an eprintln! diagnostic naming the cuBLAS status code), matching the pattern of the existing kernel-launch failure paths.

## 3. Direct-probe test (B.5-compliant)

The intervention's direct probe is shape-diverse numerical equivalence against the current naive kernel's output. Signals-that-only-the-claimed-path-can-produce:

- Output tensor's `device` field is 1 (GPU), same as before.
- Output tensor's element values match the naive kernel's output within 1e-5 relative tolerance (see §4 for tolerance rationale).
- `nsl_matmul_f32` PTX kernel is NOT launched during the test run (launch-counter assertion: the old kernel's name no longer appears in `NSL_PROFILE_KERNELS=1` output).

Test matrix (new unit test file `crates/nsl-runtime/tests/matmul_cublas_equivalence.rs`, `#[cfg(feature = "cuda")]`):

1. **Square small:** (32, 32, 32), (128, 128, 128), (256, 256, 256).
2. **Square mid:** (512, 512, 512), (1024, 1024, 1024).
3. **Rectangular:** (64, 128, 256), (256, 64, 128), (512, 2048, 128).
4. **Square Llama-scale:** (4096, 4096, 4096) — the shape that B.3.2's trigger measurement used.
5. **Batched-in-row m=1:** (1, 4096, 4096) — exercises the "single-row matmul" edge case that B.3.1's fused FFI registers.

For each: fill A and B with deterministic pseudo-random f32 values (seed the RNG once per shape; no shared state between shapes). Run matmul under both paths, compare element-wise at 1e-5 relative tolerance. Report per-shape max absolute and relative drift.

The "PTX kernel NOT launched" assertion uses the kernel-profile JSON — parse it, assert `nsl_matmul_f32` does not appear in the event list. If it does, the swap didn't take effect for that shape (routing bug).

## 4. Tolerance rationale

f32 floating-point addition is non-associative. The naive kernel sums sequentially: `c[m,n] = sum(a[m,k] * b[k,n] for k in 0..K)`. cuBLAS sums via a tiled/blocked order depending on its internal implementation — typically reduction trees with partial sums combined at the end.

For K=4096 f32 summation:
- Per-element fma introduces up to 1 ULP relative error per step (~1.2e-7 at f32).
- Naive sum: `K * ULP ≈ 4096 * 1.2e-7 ≈ 5e-4` worst-case relative error bound.
- Tiled tree sum (cuBLAS): `log2(K) * ULP ≈ 12 * 1.2e-7 ≈ 1.5e-6` worst-case relative error bound.

Both are well within 1e-5 for typical well-conditioned matmul inputs. The per-element drift between the two summation orders is typically in the 1e-7 to 1e-5 range for K=4096-scale problems.

**Acceptance criterion: 1e-5 relative tolerance on the new equivalence test matrix.** Tighter than that risks flakiness from summation-order drift; looser risks missing a real bug.

### Tolerance-relaxation enumeration for existing tests

Every test currently pinned at tolerances tighter than 1e-5 that compares against naive-kernel matmul output needs explicit review. The swap commit enumerates each relaxation with a reason.

Candidates to search (grep for tolerances near matmul usage):

- WRGA B.3.1 fixtures (Fixture A/B/C at 1e-4): the fused forward kernel's output tolerance is against a CPU reference (or mathematical ground truth), NOT the naive GPU kernel — so the swap shouldn't affect them directly. Verify by reading each fixture's comparison target.
- Fixture D (1e-3): same — ULP-derived from sigmoid, not from matmul. Verify.
- Any other matmul-adjacent test files under `crates/nsl-cli/tests/` and `crates/nsl-runtime/tests/`: enumerate during implementation.

**Rule:** if a test's tolerance is relaxed as part of the swap, the commit message lists the test name and the reason (all expected to be "summation-order drift from cuBLAS swap"). If any test's tight tolerance is there for a diagnostic reason unrelated to matmul (e.g., catching a specific regression), that test does NOT get relaxed — its failure under cuBLAS is a real bug in the wrapper and must be debugged. This discipline is the same B.3.1 Fixture D ULP-comment pattern applied to the swap.

## 5. Commit structure (single PR, two commits)

**Commit A — cuBLAS swap:**

- Add `"cublas"` to cudarc features in `nsl-codegen/Cargo.toml` and `nsl-runtime/Cargo.toml`.
- Add `gpu_matmul_f32_cublas` wrapper: cublasHandle lazy-init global, row-major-via-op-swap dispatch, alpha=1.0/beta=0.0, error translation.
- Route `gpu_matmul_f32` to call the cuBLAS wrapper. Retain the naive PTX kernel's launch code behind an env var `NSL_MATMUL_FORCE_NAIVE=1` as a rollout safety net.
- Add `crates/nsl-runtime/tests/matmul_cublas_equivalence.rs` per §3.
- Enumerate any tolerance relaxations in existing tests.

**Commit B — re-run B.3.2 trigger against clean baseline + record verdict:**

- Run `cargo test --features cuda --test wrga_gatedlora_backward_trigger wrga_b32_fused_trigger_final -- --ignored --nocapture` (75 min bench).
- Parse new ratios. Apply decision tree from `docs/superpowers/specs/2026-04-19-wrga-b32-option3-revised-design.md` §6:
  - ratio > 2.5x: B.3.2 proceeds; file a new B.3.2 kernel-work milestone.
  - ratio in [1.5x, 2.5x]: profile backward post-swap; decide per profile.
  - ratio < 1.5x: B.3.2 stays deferred; the cleaner primitive alone closed the gap.
- Update `project_wrga_b32_measurement.md` with the post-swap ratio + decision-tree branch.
- Empty commit or memory-file-reference commit to record the measurement.

## 6. Non-goals

- No TF32 tensor-core variant (option B). Separate future milestone with its own spec and `@use_tf32` opt-in.
- No pure-PTX SMEM-tiled kernel (option A). cuBLAS dominates the investment curve for f32 GEMM.
- No changes to `gpu_matmul_f64`, `gpu_sparse_matmul_csr_f32`, or other matmul variants. Scope is strictly `nsl_matmul_f32` → cublasSgemm.
- No changes to the WRGA fused kernels. They already use tensor-core MMA via hand-written PTX.
- No deletion of the naive PTX constant in this PR. Kept behind `NSL_MATMUL_FORCE_NAIVE=1` env var for rollout safety; removed in a follow-up after the swap has been live for at least one successful B.3.2-adjacent milestone.

## 7. Expected post-swap trigger ratio (analytical prediction)

Current state: `nsl_matmul_f32` ~1.5 TFLOPs/s; backward iter 12,800 ms; forward iter (fused) 1792 ms; ratio 106x.

Post-swap (cuBLAS on 5070 Ti, f32): expect ~25-30 TFLOPs/s peak at (4096, 4096, 4096). ~15-20x speedup on the naive kernel.

Forward iter time shrinks to ~1792 ms - small-delta (fused kernel already uses tensor cores via B.3.1's m16n8k16 MMA; cuBLAS doesn't help the fused path directly). Small-delta estimate: forward stays ~1700-1800 ms/iter.

Backward iter time shrinks to ~12,800 / 15 ≈ 850 ms/iter (6 matmuls × ~70 ms each + elementwise + loss overhead).

Expected post-swap ratio: **~850 / 1750 ≈ ~0.5x to ~2x.**

Wait — that's wrong because forward is ~1700 ms but backward dropped below that. Let me redo: fused forward (1792 ms) is dominated by the fused kernel itself, which already uses tensor cores. Its time doesn't meaningfully change. Backward drops ~15x to ~850 ms. Ratio = 850 / 1792 ≈ 0.47x. Backward would be FASTER than forward — counterintuitive but follows from: backward's 6 matmuls (plus primitives) in optimized cuBLAS land ≈ 850 ms of total kernel compute, while forward's one fused kernel (thread-0-only staging, per B.3.1 perf gap) takes ~1800 ms.

**If the post-swap ratio comes in < 1.0x, it means backward is actually faster than forward.** That inverts the trigger — B.3.2 fused backward wouldn't deliver further speedup; the forward kernel's thread-0-only staging is the next bottleneck (B.4 perf milestone).

The trigger decision-tree branches remain: > 2.5x → B.3.2; < 1.5x → deferred. An inverted ratio (< 1.0x) trivially lands in the "< 1.5x deferred" branch. **B.3.2 may stay deferred indefinitely once the matmul primitive is optimized.**

This is the measurement clarity option 2 was designed to produce. The analytical prediction doesn't bind the decision — the actual measurement does. But the prediction tells us what question the measurement is likely to answer, and the answer is plausibly "the real bottleneck has moved."

## 8. Success criteria

- Existing 15 WRGA integration fixtures + 27 ptxas + 3 source-AD structural tests green.
- New `matmul_cublas_equivalence.rs`: all 5 shape categories pass at 1e-5 relative tolerance; no `nsl_matmul_f32` launch observed in profile output.
- Tolerance relaxations in existing tests (if any) enumerated with reason in the commit message.
- B.3.2 trigger re-measurement completes; post-swap ratio recorded in memory file; decision-tree branch applied.
- `NSL_MATMUL_FORCE_NAIVE=1` env var still exercises the naive kernel path (rollout safety verified).
