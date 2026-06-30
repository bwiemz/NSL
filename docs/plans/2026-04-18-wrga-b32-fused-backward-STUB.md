# WRGA B.3.2 -- Fused GatedLoRA Backward (DEFERRED INDEFINITELY -- RESOLVED 2026-05-23)

**Status: RESOLVED 2026-05-23 -- trigger does NOT fire on a clean baseline; B.3.2 deferred indefinitely.**

The per-op profiling scope step (`crates/nsl-cli/tests/wrga_b32_per_op_breakdown.rs`)
ran on current `main` at the prescribed shape (b=32, seq=2048, dim=4096, r=16; RTX 5070 Ti,
CUDA 13.2; `--source-ad --target cuda_sm80`, `NSL_WRGA_FUSED_CUDA=1`). Results:

| metric | value |
|---|---|
| wall / iter | 6,558 ms |
| GPU / iter | 5,662 ms (86.3% of wall -- GPU-dominated) |
| host overhead | 896 ms (13.7% -- allocator/tape/CUDA-API/source-AD) |

GPU breakdown: **adapter_fused (forward kernel) 73.4%**, matmul 15.1%, reduction 8.1%,
elementwise_arith 2.7%, copy_layout 0.9%.

**Trigger evaluation:** forward (adapter_fused) = 4,156 ms; backward (matmul + reduction +
elementwise + copy) ≈ 1,519 ms. **backward / forward = 0.37x**, far below the 2.5x trigger
threshold below. The original 106x trigger reading (see `project_wrga_b32_measurement.md`
history) was taken on a broken substrate -- naive `nsl_matmul_f32` and/or CPU fallback and/or
the zero-duration profiler bug, all since fixed. On a clean baseline the trigger fails.

**Decision:** B.3.2 (fused GatedLoRA backward kernel) is **deferred indefinitely.** A fused
backward kernel's ceiling is the backward matmul = 15% of GPU / 13% of wall, and fusion only
removes HBM round-trips/launch overhead within that slice -- not worth the kernel.

**What to build instead:** the fused **forward** kernel is 73.4% of GPU time (a single ~4.2s
launch) -- the single-threaded thread-0-only staging pathology. See **WRGA B.4** scope doc
`docs/plans/2026-05-23-wrga-b4-fused-forward-staging-scope.md`. Allocator optimization is also
NOT warranted (host overhead is only 13.7%, not the ~99.5% an earlier hypothesis predicted).

The original measurement-gated framing is preserved below for the record.

---

**Status (original):** deferred pending measurement trigger. Do NOT schedule this milestone without satisfying the trigger below.

## Measurement-gated promotion trigger

Promote B.3.2 from deferred to scheduled **only when** a GatedLoRA training benchmark at seq=2048, rank=16 shows backward time materially larger than forward. Specifically: if `backward_time > 2.5 x forward_time` on a representative workload (Llama-3-8B-style, batch x seq = 65k tokens per microbatch, sm_80 or sm_90), schedule B.3.2.

If backward stays within 1.5x to 2x forward and remains matmul-bound (profiler confirms >= 80% of backward time in matmul kernels rather than elementwise ops), the unfused adapter-triple backward path is adequate; **B.3.2 remains deferred indefinitely**. Elementwise sigmoid backward (`sigma(x)*(1-sigma(x))` via the retained forward-tape value) is already cheap under source-AD -- there is no kernel-launch overhead to eliminate.

## Inherited institutional-memory rules

B.3.2, when scheduled, inherits the institutional lesson from the 2026-04-16 WRGA PTX scaffolding retrospective (see `project_wrga_ptx_scaffolding_discovered.md`): **ptxas validation MUST appear in B.3.2's first real-implementation commit, not deferred to a later test layer.** The commit that introduces the backward PTX emitter is the same commit that introduces its ptxas unit sweep. This rule is non-negotiable; it is the WRGA retrospective's load-bearing institutional correction. B.3.1 further promoted this to the WRGA paper's Appendix B.2 as a generalized rule covering all approximation-based PTX helpers.

## Four-layer test discipline inheritance

B.3.2 MUST ship with all four test layers green at milestone close:

1. **Skeleton snapshots** -- any new helpers gated by per-variant pinned snapshots (or structural assertions for simple helpers; see B.3.1's convention).
2. **Unit ptxas** -- backward kernel PTX validated across the same shape matrix as forward (6 reused LoRA shapes + any backward-distinctive configs), feeding each emission to `cuModuleLoadData` or `ptxas`.
3. **Integration numerical** -- gradient correctness verified against finite-difference reference at 1e-3 tolerance (looser than forward's 1e-4 because finite-difference step-size trades off against condition-number noise; set the step size and justify the tolerance inline at the fixture).
4. **E2E launch-counter** -- `backward_fused_cuda_actually_fires` parallel to the forward hardening test, asserting `[nsl-gpu-launch-count] >= 1` during a full forward+backward pass.

## Scope boundary

B.3.2 covers ONLY fused backward for GatedLoRA. Fused backward for LoRA and IA3 is a separate milestone (B.3.3 or similar) if demanded by the same measurement trigger applied to those adapter types. Do not conflate.
