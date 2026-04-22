# Matmul Primitive: cuBLAS Swap for `nsl_matmul_f32`

> **v3 UPDATE (2026-04-21 evening):** empirical post-swap measurement revealed cudarc's `CudaBlas::gemm` invokes `cublasSgemm_v2` under cuBLAS's default math mode, which on sm_80+ enables TF32 tensor cores. Observed throughput at (4096,4096,4096) is ~105 TFLOPs/s (above the 43 TFLOPs/s pure-f32 SIMT peak), and observed numerical drift is ~4.2e-4 at K=4096 (consistent with TF32 per-element ~1e-3 accuracy, NOT the 1.5e-6 summation-order bound §4 originally derived for strict f32).
>
> This section is a retrospective capture. §2 (alpha/beta), §3 (test presence/absence), §5 (commit structure), §8 (success criteria) carry over with the additions documented in new sections §9 (math-mode API) and §10 (split equivalence tests). §4's ULP analysis was correct for strict f32 but does not describe the implementation's runtime behavior under the default math mode — amended to describe both regimes.
>
> Two retrospective lessons (not yet codified as rules):
> 1. Pre-dispatch verification should include "what cuBLAS math mode will actually run at runtime" when the intervention is a cuBLAS swap. v1/v2 of this spec assumed strict f32; the implementation landed in a different math mode and the spec didn't account for the possibility.
> 2. **Layered bottleneck:** post-swap the B.3.2 trigger ratio dropped from 106.1x to 61.3x (fires, same decision-tree branch) but backward iter time INCREASED by ~28s despite ~400x matmul primitive speedup. The 188s backward was never matmul-dominated; the real bottleneck is allocator + source-AD tape overhead. B.3.2's expected speedup is correspondingly lower than originally framed. See memory file addendum.

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

**Worked example (square Llama-scale, A:[4096,4096] @ B:[4096,4096]):**

```text
cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               4096,        // m: rows of C^T (= cols of row-major C = N)
               4096,        // n: cols of C^T (= rows of row-major C = M)
               4096,        // k: contraction dim
               &SGEMM_ALPHA,
               B_ptr, 4096, // A^cublas := B^T (column-major), lda = N = 4096
               A_ptr, 4096, // B^cublas := A^T (column-major), ldb = K = 4096
               &SGEMM_BETA,
               C_ptr, 4096); // C^cublas := C^T (column-major), ldc = N = 4096
```

**Worked example (rectangular, A:[M,K] @ B:[K,N], all row-major):**

```text
cublasSgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               N,           // m: rows of C^T = N
               M,           // n: cols of C^T = M
               K,           // k: contraction
               &SGEMM_ALPHA,
               B_ptr, N,    // B has N columns row-major => N rows column-major, lda = N
               A_ptr, K,    // A has K columns row-major => K rows column-major, ldb = K
               &SGEMM_BETA,
               C_ptr, N);   // C has N columns row-major => N rows column-major, ldc = N
```

The leading-dimension argument for each operand equals "number of columns in row-major" (which becomes "number of rows in column-major"). For contiguous row-major tensors, that's just the tensor's last-dim extent. Non-contiguous strided tensors are out of scope; the wrapper asserts contiguity at entry and falls through to a clear error if not.

### 2.2 Alpha/beta scaling

cuBLAS computes `C = alpha * op(A) * op(B) + beta * C`. Pass `alpha = 1.0`, `beta = 0.0` to match the naive kernel's `C = A @ B`.

`alpha` and `beta` are `f32` pointers (cuBLAS expects `*const f32`). Per cuBLAS docs, the pointer mode defaults to `CUBLAS_POINTER_MODE_HOST`. **Pin the constants as `'static` at module scope**, not stack-allocated locals:

```rust
static SGEMM_ALPHA: f32 = 1.0;
static SGEMM_BETA: f32 = 0.0;
// later:
cublasSgemm_v2(handle, ..., &SGEMM_ALPHA, ..., &SGEMM_BETA, ...);
```

**Load-bearing:** cuBLAS with `CUBLAS_POINTER_MODE_HOST` reads `*alpha` and `*beta` at *kernel launch time*, not at the moment `cublasSgemm_v2` returns. Stack-allocated locals can go out of scope before the kernel executes (under any future concurrent use, or even today if the function returns before the launch completes on the stream). `'static` constants outlive every possible kernel launch by construction, eliminating the footgun at zero cost. NSL today runs matmuls synchronously on the default stream, so the footgun is latent — but it WILL bite the first time anyone adds concurrency, and `'static` prevents it forever.

### 2.3 Handle lifecycle

`cublasHandle_t` is a stateful object that must be created once and reused across calls. Three lifecycle patterns:

- **Per-call create/destroy:** simplest, but cublasCreate is ~ms-scale overhead. At 20 matmul calls per train iter, this dominates the speedup — bad.
- **Per-CUDA-context singleton:** one handle per device. NSL currently has one CUDA context globally (via cudarc's driver feature). A lazy-initialized global `OnceLock<CublasHandle>` is sufficient.
- **Per-stream singleton:** one handle per stream. NSL uses the default stream everywhere, so per-context and per-stream collapse to the same thing.

**Pin per-CUDA-context singleton via `OnceLock<CublasHandle>`. Intentionally leak the handle at process exit; do NOT register an atexit destructor.**

`atexit` callbacks fire during process teardown, at which point cudarc's CUDA context state is already being torn down. Calling `cublasDestroy` on a handle after its underlying CUDA context is gone produces spurious driver errors or crashes — the canonical "atexit ordering" hazard. The handle is a few KB; leaking one per process at exit is harmless. The OS reclaims the resource on process termination regardless. NSL has atexit infrastructure (kernel-profiler, launch-counter — see `crates/nsl-runtime/src/args.rs`) but those callbacks are specifically designed to be safe at teardown; cuBLAS handle destruction is not.

The `OnceLock` pattern guarantees thread-safe lazy init and shared access without explicit destruction:

```rust
use std::sync::OnceLock;
static CUBLAS_HANDLE: OnceLock<CublasHandle> = OnceLock::new();

fn cublas_handle() -> &'static CublasHandle {
    CUBLAS_HANDLE.get_or_init(|| {
        // cudarc::cublas::CudaBlas::new(stream) or equivalent;
        // panic on init failure (catastrophic — no recovery)
        ...
    })
}
```

### 2.4 Error translation

cuBLAS returns `cublasStatus_t`. On success: `CUBLAS_STATUS_SUCCESS`. On failure: several error codes (alloc failed, invalid value, internal error, etc). The wrapper translates into NSL's existing error model — for runtime FFI paths this means returning a null i64 tensor pointer (with an eprintln! diagnostic naming the cuBLAS status code), matching the pattern of the existing kernel-launch failure paths.

## 3. Direct-probe test (B.5-compliant)

The intervention's direct probe is shape-diverse numerical equivalence + presence/absence assertions on the kernel-profile output. Signals that only the claimed path can produce:

- Output tensor's `device` field is 1 (GPU), same as before.
- Output tensor's element values match a CPU/naive reference within 1e-5 relative tolerance (see §4 for tolerance rationale).
- **Presence:** a cuBLAS-pattern kernel name appears in the kernel-profile output. cuBLAS dispatches to architecture-specific kernels with names like `volta_sgemm_128x64_nn`, `ampere_sgemm_128x128_nn`, etc. The exact name varies by SM architecture, shape, and cuBLAS version, so the test pattern-matches on the substring `sgemm` or `gemm_` rather than pinning a specific name.
- **Absence:** `nsl_matmul_f32` does NOT appear in the kernel-profile output for the same run.

Both assertions are required. Presence-only would pass if cuBLAS kernels fire in *some* path (e.g., from a stdlib dep) while `nsl_matmul_f32` also fires for the workload under test. Absence-only would pass trivially if the profiler emits an empty event list (profiler broken, no GPU available, feature flag drift). Together they verify "the claimed path fires AND only the claimed path fires" per Appendix B.5.

```rust
let profile = parse_kernel_profile();
let names: Vec<&str> = profile.events.iter().map(|e| e.kernel_name.as_str()).collect();
let has_cublas = names.iter().any(|n| n.contains("sgemm") || n.contains("gemm_"));
let has_naive  = names.iter().any(|n| *n == "nsl_matmul_f32");
assert!(has_cublas, "expected cuBLAS sgemm kernel in profile; got: {:?}", names);
assert!(!has_naive, "naive nsl_matmul_f32 still firing despite cuBLAS swap; got: {:?}", names);
```

Test matrix (new unit test file `crates/nsl-runtime/tests/matmul_cublas_equivalence.rs`, `#[cfg(feature = "cuda")]`):

1. **Square small:** (32, 32, 32), (128, 128, 128), (256, 256, 256).
2. **Square mid:** (512, 512, 512), (1024, 1024, 1024).
3. **Rectangular:** (64, 128, 256), (256, 64, 128), (512, 2048, 128).
4. **Square Llama-scale:** (4096, 4096, 4096) — the shape that B.3.2's trigger measurement used.
5. **Batched-in-row m=1:** (1, 4096, 4096) — exercises the "single-row matmul" edge case that B.3.1's fused FFI registers.

For each: fill A and B with deterministic pseudo-random f32 values (seed the RNG once per shape; no shared state between shapes). Compare against a CPU reference (NSL's existing CPU `tiled_matmul_f32`, which has well-defined deterministic summation order). Run matmul through `gpu_matmul_f32`, compare element-wise at 1e-5 relative tolerance. Report per-shape max absolute and relative drift.

**Edge case — m=1 may need looser tolerance.** cuBLAS dispatches different algorithms for low-m shapes (vector-matrix product specialization). The summation-order analysis in §4 (`log2(K) * ULP ≈ 1.5e-6`) bounds the drift for typical paths, but a vector-matrix kernel may have a different reduction structure and slightly different drift. **Verify during test execution that the m=1 shape stays within 1e-5; if it exceeds, relax to 1e-4 for that shape with a documented rationale in the test, NOT a blanket relaxation.** The relaxation rule mirrors §4's general policy: any tolerance change has a reason next to it.

The presence/absence kernel-profile assertions from above run on every shape in the matrix. Routing bugs that affect only some shapes (e.g., "small shapes still hit the naive kernel") get caught per-shape rather than globally.

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
- Add `gpu_matmul_f32_cublas` wrapper: cublasHandle lazy-init `OnceLock` global (no atexit destruction; intentionally leaks at process exit per §2.3), row-major-via-op-swap dispatch (per the worked example in §2.1), `'static SGEMM_ALPHA/BETA` constants (per §2.2), error translation to NSL's null-i64 failure pattern.
- Route `gpu_matmul_f32` to call the cuBLAS wrapper. **Delete the naive PTX kernel and its launch code in the same commit.** No rollout safety net.
- Add `crates/nsl-runtime/tests/matmul_cublas_equivalence.rs` per §3.
- Enumerate any tolerance relaxations in existing tests in the commit message.

**No `NSL_MATMUL_FORCE_NAIVE=1` env var.** The earlier draft of this spec proposed retaining the naive kernel as a rollout safety net behind an env var. On reflection that's the wrong tradeoff: the safety net's value is "if cuBLAS misbehaves, fall back to slow but correct code" — but slow-but-correct silently masks the bug rather than surfacing it. The correct response to a cuBLAS regression is to fix the wrapper or revert the PR, not to silently route around the symptom. Plus: maintaining two code paths means they can drift over time (one gets a fix the other doesn't), which is the failure mode B.3.1's anti-drift discipline was created to prevent. Single path; trust the §3 direct-probe tests; revert if needed.

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
- The naive PTX constant IS deleted in the swap commit (per §5 reasoning). Single code path post-merge.

## 7. Pre-swap measurement state (corrected) and prediction caveats

**Source numbers from the post-Option-3 trigger bench (PR #93, recorded in `project_wrga_b32_measurement.md`):**

| Metric | Per-iter value |
|---|---|
| fused-forward-in-training (the fused kernel inside `train(step):`) | 1792 ms |
| fwd+bwd train iter (full forward + source-AD backward + loss) | 190,130 ms |
| backward-only (= fwd+bwd minus fused-forward) | 188,338 ms |
| Trigger ratio (fwd+bwd / fwd) | **106.1x** |

The 106x figure is `190,130 / 1792`, where both numerator and denominator are measured in the same train-iteration context. This is the train-forward vs train-backward comparison the trigger spec required; not a B.5 violation. (The pre-Option-3 14.7x figure WAS a B.5 violation — fused-inference-forward vs all-unfused-train; that's been corrected by Option 3 shipping.)

**Decomposition of 188,338 ms backward-only (per the kernel-name profile from 2026-04-20):**

- 6 `nsl_matmul_f32` calls per iter: `dW`, `dA`, `dB`, two `dx` contributions, plus a forward recompute in the AD rule.
- 13 elementwise/reduction calls per iter: scale, sigmoid, broadcast, mul, sub, add, reduce.
- 1 fused-GatedLoRA-forward call (in train iteration's forward portion, counted in the 1792 ms not the backward).

**The decomposition is by COUNT, not by TIME.** NSL's kernel profiler currently reports zero durations (event-creation-before-context bug, separate follow-up). We do NOT know how the 188 sec splits between matmul kernels, elementwise kernels, allocator overhead, memcpy, and source-AD tape machinery. This matters for the prediction.

### Naive prediction (over-optimistic)

If we assumed all 188 sec was matmul compute and cuBLAS gives a clean ~17x speedup (1.5 → ~25 TFLOPs/s on a 5070 Ti at this shape), backward would drop to 188 / 17 ≈ 11 sec/iter, ratio 11000 / 1792 ≈ 6.1x. Still well above 2.5x; B.3.2 fires.

### Realistic prediction (uncertain)

If matmul is only PART of the 188 sec (allocator thrash on 13 elementwise intermediates each at 65536x4096x4 = 1 GB tensor allocations, memcpy, source-AD tape lookups), the cuBLAS speedup applies to that part only. Plausible regimes:

- **Matmul-dominant:** matmul = 80% of 188 sec ≈ 150 sec. Post-swap: 150/17 + 38 ≈ 47 sec backward. Ratio 47000 / 1792 ≈ 26x. **B.3.2 fires hard.**
- **Mixed:** matmul = 50% of 188 sec ≈ 94 sec. Post-swap: 94/17 + 94 ≈ 100 sec backward. Ratio 100000 / 1792 ≈ 56x. **B.3.2 fires.**
- **Allocator-dominant:** matmul = 20% of 188 sec ≈ 38 sec. Post-swap: 38/17 + 150 ≈ 152 sec backward. Ratio 152000 / 1792 ≈ 85x. **B.3.2 fires, but kernel-fusion wins shrink because backward is allocator-bound.**

In all three regimes the ratio stays well above 2.5x. **B.3.2 is likely to remain scheduled even post-swap.** What changes is the *attribution* of the speedup B.3.2 would deliver: in the allocator-dominant regime, fusing matmuls into one backward kernel only addresses the matmul portion, and the allocator-bound elementwise overhead remains. The headline B.3.2 speedup quote depends on which regime we're in.

### Why the prediction can't be tightened analytically

The 188 sec / 17 ≈ 11 sec naive prediction earlier in this spec (and in earlier conversation framings) implicitly assumed matmul was 100% of backward. That's the same class of category-error the Option 3 retrospective flagged: assuming a measurement covers a regime it doesn't. Without per-kernel timing, we can't bound the matmul-vs-other split tighter than 20%-80%.

**Decision: don't tighten. Ship the swap, re-measure the trigger, apply the decision tree to the measured ratio.** The decision-tree branches (> 2.5x → B.3.2; [1.5, 2.5] → profile; < 1.5x → defer) all remain mechanical. The analytical prediction's value is "what range of ratios should we be prepared to see?" — the answer is roughly 6x-85x, all firing the trigger. If the measurement comes in below 6x, that's a new finding worth investigating.

The fused forward kernel itself does NOT use `nsl_matmul_f32` (it's a hand-written m16n8k16 MMA per B.3.1). cuBLAS swap doesn't change its time. Post-swap forward iter stays ~1700-1800 ms — though if backward sees the predicted 80% reduction in matmul time, the fused-forward thread-0-only staging (B.4 perf gap, separate milestone) becomes the next scaling bottleneck. That's a separate decision tree, not this one.

## 8. Success criteria

- Existing 15 WRGA integration fixtures + 27 ptxas + 3 source-AD structural tests green.
- New `matmul_cublas_equivalence.rs`: all 5 shape categories pass at 1e-5 relative tolerance; no `nsl_matmul_f32` launch observed in profile output.
- Tolerance relaxations in existing tests (if any) enumerated with reason in the commit message.
- B.3.2 trigger re-measurement completes; post-swap ratio recorded in memory file; decision-tree branch applied.
- Naive `nsl_matmul_f32` PTX constant + launch code deleted from `cuda/kernels.rs` and `cuda/mod.rs`. Single matmul code path post-merge.

## 9. Math-mode API (v3 addition — 2026-04-21 evening)

The v1/v2 swap landed with cuBLAS's default math mode. On sm_80+ that's TF32 tensor cores — ~400x faster than the naive kernel but with ~1e-3 per-element numerical drift. For most ML workloads this is the correct default (matches PyTorch, JAX, TensorFlow); for numerical-analysis workloads it's a silent regression vs the pre-swap strict-f32 behavior. v3 adds an explicit opt-out API.

### Three-level control (library / program / per-workload deferred)

- **Library level (compile-time):** new Cargo feature `strict-matmul` in `crates/nsl-runtime/Cargo.toml`. When enabled, the default math mode at runtime is `CUBLAS_PEDANTIC_MATH` (strict f32, no TF32). When disabled (the Cargo default), the runtime defaults to `CUBLAS_DEFAULT_MATH` (TF32 on sm_80+).
- **Program level (runtime):** env vars override the Cargo default.
  - `NSL_MATMUL_PEDANTIC=1` forces pedantic regardless of Cargo feature.
  - `NSL_MATMUL_TF32=1` forces TF32 regardless of Cargo feature.
  - If both are set, `NSL_MATMUL_PEDANTIC=1` wins (safer default).
- **Per-workload (future, Phase 2):** a `@pedantic_matmul` decorator is mentioned for forward-compat but NOT shipped in this PR. Out of scope.

### Resolution logic

```rust
fn resolve_math_mode() -> CublasMathMode {
    if std::env::var("NSL_MATMUL_PEDANTIC").ok().as_deref() == Some("1") {
        return CublasMathMode::Pedantic;
    }
    if std::env::var("NSL_MATMUL_TF32").ok().as_deref() == Some("1") {
        return CublasMathMode::Default;
    }
    if cfg!(feature = "strict-matmul") {
        CublasMathMode::Pedantic
    } else {
        CublasMathMode::Default
    }
}
```

Called once at `OnceLock<CudaBlas>` init; result baked into the handle via `cublasSetMathMode`. Changing env vars mid-process does NOT change the mode — handle's mode is set at first-use.

### Mode application

cudarc doesn't expose `cublasSetMathMode` in its safe API, but the raw FFI is accessible via `cudarc::cublas::sys::cublasSetMathMode`. After `CudaBlas::new(stream)` returns, call the raw FFI on the handle field. ~10 LOC addition.

### Discoverability

The resolved mode is logged at `OnceLock` init time:

```text
[nsl-matmul] cuBLAS math mode: TF32 (default — set NSL_MATMUL_PEDANTIC=1 or build with --features strict-matmul for strict f32)
```

or:

```text
[nsl-matmul] cuBLAS math mode: pedantic (strict f32, ~5-10x slower than TF32 default)
```

One-time log per process, at handle init. No spam.

## 10. Split equivalence tests (v3 addition)

The `matmul_cublas_equivalence.rs` test from v2 ran against cuBLAS default (TF32) with a K-scaled envelope. v3 replaces it with two explicit tests.

**Test A — `matmul_cublas_pedantic_equivalence`** (strict-f32 correctness gate):

- Runs with `NSL_MATMUL_PEDANTIC=1` set (via `std::env::set_var` before subprocess spawn OR a test-harness flag that forces pedantic at init).
- Uses the **1e-5 relative tolerance** from spec §4.
- Shape matrix: 10 shapes from v2 §3.
- Failure ⇒ wrapper bug (wrong lda/ldb/ldc, wrong operand order, missed `cublasSetMathMode`).

**Test B — `matmul_cublas_tf32_default_sanity`** (TF32 default bound):

- Runs with cuBLAS default math mode (no env var override).
- Uses a **5e-3 relative tolerance** per NVIDIA TF32 specs (10-bit mantissa vs f32's 24-bit).
- Shape matrix: same 10 shapes.
- Failure ⇒ TF32-specific (mode-resolution bug, cudarc version incompatibility).

Both tests assert the presence/absence kernel-profile invariant (cuBLAS `sgemm`/`gemm_` pattern appears; naive `nsl_matmul_f32` does not).

**Test isolation note:** NSL's `gpu_matmul_f32` is called from a subprocess (`nsl run`). Env vars set via `std::env::set_var` in the test body propagate to the subprocess via `Command::env`. Ensure Test A explicitly sets `NSL_MATMUL_PEDANTIC=1` on the `Command` builder; Test B explicitly clears both env vars (using `Command::env_remove`) to inherit the Cargo-feature default.

### CI recommendation

- Primary CI build: default features (TF32 default). Test A sets `NSL_MATMUL_PEDANTIC=1`; Test B clears env vars. Both pass.
- Secondary CI matrix entry: `--features strict-matmul`. Test A passes same way. Test B with cleared env vars now gets pedantic (the Cargo feature) — for Test B to exercise TF32 under this build, it must explicitly set `NSL_MATMUL_TF32=1`.

Running both catches bugs where one resolution path works but the other doesn't.

## 11. v3 commit structure

- **Commit A (already landed at f704b259):** cuBLAS swap with default math mode. Deleted naive PTX. 10-shape equivalence test (K-scaled envelope).
- **Commit B (v3 delta — pending):**
  - Add `strict-matmul` Cargo feature to `nsl-runtime/Cargo.toml`.
  - Add math-mode resolution + `cublasSetMathMode` call in `OnceLock` init.
  - Add discoverability `eprintln!` at handle init.
  - Replace K-scaled-envelope test with Test A (pedantic, 1e-5) + Test B (TF32, 5e-3).
  - Doc update: CI matrix recommendation.
- **Commit C (administrative — pending):** trigger re-measurement recorded (61.3x / 34.1x post-swap TF32) + backward-regression finding filed in memory file.

Single PR containing A+B+C. Merge gate: both Test A and Test B green on default Cargo build.
