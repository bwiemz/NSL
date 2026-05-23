# WRGA B.4 -- Fused-Forward Staging Parallelization (SCOPE)

**Status:** scoping document. Surfaced 2026-05-23 by the B.3.2 per-op breakdown
(`crates/nsl-cli/tests/wrga_b32_per_op_breakdown.rs`). Not yet a plan -- this
document frames the problem, options, and recommendation for review before any
implementation work begins.

**Supersedes the B.3.2 scheduling decision.** See
`docs/plans/2026-04-18-wrga-b32-fused-backward-STUB.md` (resolved: B.3.2 deferred)
and `project_wrga_b32_measurement.md` (⭐ BOTTOM LINE).

---

## 1. Problem

The fused GatedLoRA **forward** kernel is the dominant GPU cost in a training step,
by a wide margin. Per-op breakdown on current `main` (RTX 5070 Ti, CUDA 13.2,
prescribed shape b=32 / seq=2048 / dim=4096 / r=16, `--source-ad --target cuda_sm80`,
`NSL_WRGA_FUSED_CUDA=1`):

| category | ms/iter | % GPU | fwd/bwd |
|---|---|---|---|
| **adapter_fused** | **4,156** | **73.4%** | forward fused GatedLoRA kernel |
| matmul | 852 | 15.1% | backward (cuBLAS) |
| reduction | 460 | 8.1% | backward |
| elementwise_arith | 154 | 2.7% | backward |
| copy_layout | 53 | 0.9% | backward |

GPU is 86.3% of wall (5,662 / 6,558 ms). One fused-forward kernel launch takes
~4.2 s. For comparison, the entire backward (matmul+reduction+elementwise+copy) is
~1.5 s. **The forward kernel costs ~2.7x the whole backward.**

This is the known single-threaded-staging pathology, previously filed as B.4 /
follow-up #7 in `project_wrga_b32_measurement.md` and measured there as the fused
forward being ~9-10x slower than the unfused cuBLAS path (3.6 s vs 0.4 s).

## 2. Root cause (confirmed against current code)

The fused adapter kernel is **single-warp-per-tile** (module header,
`crates/nsl-codegen/src/wrga_kernel_helpers.rs:5`: "single-warp-per-tile,
m16n8k16 fixed, fp32 output"). All four tile-staging helpers gate their entire
global->SMEM load sequence on lane 0:

- `emit_lora_stage_x_tile` -- `wrga_kernel_helpers.rs:202`
- `emit_lora_stage_w_tile` -- `wrga_kernel_helpers.rs:284`
- `emit_lora_stage_a_tile` -- ~`wrga_kernel_helpers.rs:360`
- `emit_lora_stage_b_tile` -- ~`wrga_kernel_helpers.rs:420`

Each begins:

```ptx
setp.eq.u32 %p0, %tid_x, 0;
@!%p0 bra lora_<tile>_stage_done;
... 16 rows x 8 col_pairs of ld.global.f32 + cvt.rn.f16 + st.shared.b32 ...
lora_<tile>_stage_done:
```

So 31 of 32 warp lanes idle during staging. Each tile is 16 rows x 8 col_pairs =
128 `st.shared.b32` (≈256 `ld.global.f32`), all serialized on lane 0. At k=4096
the runtime K-loop runs `k_iters = k/16 = 256` times, staging 4 tiles per iter:

```
256 K-iters x 4 tiles x ~256 f32 loads ≈ 131,000 serialized global loads on lane 0
```

That is the ~4.2 s. The MMA itself (`matmul_mma` m16n8k16, invariants #1-#12) is
correct and fast; only the staging is the bottleneck.

## 3. The real question B.4 answers

The fused-forward kernel exists to beat the unfused path (x@W cuBLAS + adapter
triple) by eliminating HBM round-trips. It currently **loses by ~10x** because of
single-threaded staging. So B.4's goal is sharper than "make it faster":

> **Can the fused-forward kernel be made faster than the unfused cuBLAS path on the
> training-scale regime? If yes, it justifies its existence. If no even after
> parallel staging, the fused-forward kernel is the wrong tool for this regime and
> the dispatch should always pick unfused at scale.**

## 4. Options

### Option A -- Lane-distributed staging (within the existing single warp) [RECOMMENDED first real fix]

Remove the `tid_x == 0` guard. Distribute each tile's 128 b32 elements across the
32 warp lanes: lane `L` stages elements `L, L+32, L+64, L+96` (4 per lane instead
of lane 0 doing all 128). ~32x fewer serialized ops per tile.

- **Scope:** index-math change inside the 4 stagers + their register pool. No change
  to the MMA, the K-loop structure (invariant #19), the output store, or the
  single-warp assumption. The m-tail (`%m_param`) and k-tail predicates carry over
  per-lane.
- **Risk:** low-moderate. Preserves invariants #1-#20. Must keep the `bar.sync 0`
  discipline (#3): one barrier after staging completes (now across all lanes) before
  fragment loads.
- **Expected win:** large -- moves the 4.2 s staging toward memory-bandwidth-bound
  rather than single-lane-bound.

### Option B -- cp.async double-buffering [follow-on, only if A leaves staging dominant]

`cp.async.ca.shared.global` to overlap iter N+1 staging with iter N MMA.

- **Scope:** requires A first (cp.async is still per-lane). Adds a double-buffered
  SMEM allocation (2x tile SMEM), changes barrier discipline, sm_80+ only (already
  targeted).
- **Risk:** higher -- touches SMEM budget and barrier correctness (#3).
- **Expected win:** hides remaining HBM latency. Pursue only if a post-A
  re-measurement shows staging still dominates.

### Option C -- Multi-warp-per-tile + larger tiles [v2/v3, largest restructure]

Multiple warps per output tile: more lanes stage in parallel AND multiple MMAs run
concurrently.

- **Scope:** breaks the single-warp assumption baked throughout the module --
  register pool, output-tile coords, MMA lane init (#5, #10), kernel launch geometry.
- **Risk:** highest. A near-rewrite.
- **Verdict:** out of scope for B.4; revisit only if A+B prove insufficient.

### Option D -- m-aware dispatch stopgap [SHIP IMMEDIATELY, parallel track]

The fused forward is opt-in (`NSL_WRGA_FUSED_CUDA=1` + `--target cuda_sm80`). Make the
fused-dispatch gate **m-aware**: when runtime `m` (tokens) exceeds a threshold (e.g.
m > 256), dispatch the unfused cuBLAS path even when fused is requested -- because at
large m the fused kernel is a ~10x pessimization.

- **Scope:** a threshold check in the fused-dispatch decision
  (`try_cuda_launch_fused_gatedlora` in `crates/nsl-runtime/src/fused_adapter.rs`,
  and/or the AST-rewrite gate in `wrga_adapter_rewrite.rs`). Roughly one decision +
  one constant.
- **Risk:** near-zero. Falls back to the already-correct, already-faster unfused path.
- **Verdict:** ship first. Removes the footgun for anyone who opts into fused at
  training scale, independent of whether A ever lands. Lower urgency than if fused
  were default-on, but cheap insurance.

## 5. Recommendation

1. **D (stopgap) now** -- m-aware dispatch so opting into fused doesn't pessimize at
   scale.
2. **A (real fix)** -- lane-distributed staging. Then **re-measure** with the per-op
   breakdown bench (B.6 discipline: re-validate on the post-fix substrate).
3. Decide from the re-measurement:
   - fused-with-A < unfused -> the kernel justifies itself; keep it, relax D's
     threshold or remove D.
   - fused-with-A still >= unfused -> pursue **B** (cp.async) once, re-measure again.
   - fused-with-A+B still loses -> the fused-forward approach is wrong for this
     regime; make D permanent and close the fused-forward kernel as a dead end for
     large-m. (It may still win for small-m inference, where staging serialization is
     cheap -- preserve it there.)

## 6. Test discipline inheritance (mandatory at milestone close)

Per the WRGA paper Appendix B rules, B.4 inherits:

- **B.2 (numerical):** any change to staging must keep the existing integration
  fixtures green at 1e-4 (LoRA/IA3) / 1e-3 (GatedLoRA). Lane-distributed staging must
  produce bit-equivalent SMEM contents to the lane-0 version.
- **B.3 (scale-regime ptxas):** the `wrga_fused_ptx_ptxas.rs::scale__*` gates at
  n in {1024, 2048, 4096} must stay green. Add SASS no-spill checks if the per-lane
  register pressure changes.
- **B.5 (code-path verification):** verify the parallel-staging path actually fires
  at the prescribed shape (not a small-m proxy) via the launch-counter +
  `4/4 trainable params connected` probe.
- **B.6 (re-measure on clean substrate):** the post-A re-measurement IS the
  acceptance signal. Do not declare B.4 a win on PTX structure alone -- run the
  per-op breakdown bench and compare adapter_fused ms/iter before/after.
- **ptxas-in-first-commit:** the commit that changes a stager emits its ptxas sweep
  in the same commit (the non-negotiable WRGA retrospective rule).

## 7. Scope boundary

B.4 covers staging parallelization for the **GatedLoRA** fused forward (the measured
hot path). LoRA and IA3 share the same stagers (`emit_lora_stage_*` are
adapter-agnostic), so they benefit automatically -- but their numerical fixtures must
be re-confirmed, not assumed. Backward is explicitly out of scope (B.3.2 deferred).

## 8. Open questions for review

1. **Threshold for D:** m > 256? m > 512? Pick from a small m-sweep micro-bench, or
   start conservative (m > 64) and tune.
2. **Is fused-forward ever default-on planned?** If yes, D's urgency rises from
   "footgun guard" to "correctness-of-default." Confirm the intended dispatch policy.
3. **Per-lane register budget:** does distributing staging across lanes change the
   `.reg` pressure enough to risk SASS spills at n=4096? Verify with the existing
   no-spill SASS check pattern.
4. **Small-m inference regime:** confirm the lane-0 staging is actually fine (or even
   preferable) at m=1 inference before making D unconditional at large m -- the fused
   kernel may still be the right call for the decode path.
