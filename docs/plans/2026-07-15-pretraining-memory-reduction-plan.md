# Pretraining Memory Reduction — Action Plan (post-PR #370)

**Date:** 2026-07-15
**Goal:** Reduce GPU memory required to pretrain 500M–1B-param models on a 16 GiB
card (RTX 5070 Ti reference) **without regressing training speed, model quality,
or context length**. Concretely: (1) get the 1.037B config actually stepping on
16 GB, (2) restore headroom at 500M (currently 15.7 GiB peak = 95%+ of the card,
OOMs at seq 1024), (3) make context scaling memory-bounded, not memory-limited.

**Inputs:** open PR #370 (`feat/wggo-scaling-campaign`), the PR-head codebase, the
WGGO/CPDT/CCR/CFTP/CSHA research corpus in `docs/research/`, and the scaling
matrix in `models/benchmarks/WGGO_SCALING_RESULTS.md`.

---

## 1. Where the memory goes today

All numbers are f32 FASE-Deferred AdamW on the PR #370 head unless noted.

**Persistent, per-parameter (20 B/elem at peak):**

| Surface | Bytes/param | @1.037B | Notes |
|---|---|---|---|
| weights `w` | 4 (f32) | 4.15 GB | no bf16/fp16 weight training exists anywhere |
| moment `m` | 4 | 4.15 GB | host-offloadable (`--optim-state-offload`) |
| moment `v` | 4 | 4.15 GB | host-offloadable |
| `m_partial` window accumulator | 4 | 4.15 GB | **always device, always f32**, not covered by offload or fp16-moment budget |
| one live grad (Deferred epilogue) | 4 transient | — | full grad buffer only in FullBuffer mode |

**Non-persistent:** the source-AD backward **holds every backward-referenced
forward intermediate resident until end-of-backward** (`analyze_saved_tensors`,
`nsl-codegen/src/source_ad.rs:1567`). There is no general checkpointing —
recompute exists only for the attention prologue/KV under
`@checkpoint(policy="full")` (`flash_attention_v2/mod.rs:1270`,
`csha_hooks_backward.rs:1320`); `"selective"`/`"custom"` are refused. The GQA
GPU backward additionally materializes **fully-expanded K, V, dK, dV**
(`flash_attention.rs:550-602`) — 4 transient MHA-sized tensors per attention
backward, scaling linearly with seq.

**Measured walls (WGGO_SCALING_RESULTS.md):**
- 500M: 15.7 GiB peak at seq 512 (≈31 B/param all-in); OOM at seq 1024. Prior
  matrix showed 2.0 s IQR from allocator pressure at 95% VRAM.
- 1B without offload: OOM in `nsl_tensor_zeros_like` (state alloc, 16.6 GB > 16 GB).
- 1B **with** offload: passes state alloc, OOM in `nsl_tensor_matmul` at 15.8 GiB
  — the wall is now **live forward/backward intermediates**, not optimizer state.

**Conclusion:** optimizer-state placement is solved-in-principle; the binding
constraint is activation/adjoint liveness, then the remaining f32 device
surfaces (`m_partial`, weights). That ordering drives the plan.

---

## 2. Priority-ordered work items

### P0 — Instrumentation + offload hardening (small scope, do first)

These are prerequisites: P0.1 is how every later claim gets verified; P0.2/P0.3
remove the speed penalty that would otherwise make the 1B offload path
"fits but slow".

**P0.1 Per-surface VRAM accounting.** Neither benchmark doc can itemize the
500M 15.7 GiB peak (no activation or fused-workspace figure exists). Add a
peak-VRAM report that attributes bytes to {weights, m, v, m_partial, grads,
saved activations, attention workspace/expanded-KV, allocator slack} — the
caching allocator (`cuda/caching_allocator.rs`) already has pool separation
(Persistent/Transient) to hang tags on. Also: commit compressed raw per-run
benchmark JSON alongside the summary tables (review feedback from #370).
*Impact: zero memory by itself; makes every subsequent lever measurable and
feeds honest numbers to the WGGO budget planner.*

**P0.2 Offload quality-of-implementation: pinned + persistent + overlapped.**
Today `--optim-state-offload` stages m/v with pageable `malloc` host buffers,
blocking `cuMemcpyHtoD/DtoH` on the NULL stream, and re-allocates a fresh
transient device buffer every step (`stmt_fase.rs:113-135, 560-572`;
`tensor/mod.rs:1383-1413`). At 1B that is ~16.6 GB of pageable PCIe traffic per
optimizer step. Fix: (a) allocate host m/v with `cuMemAllocHost_v2` (pinned —
the primitive already exists at `cuda/mod.rs:566-597`, test-only today);
(b) keep one reusable device staging slot instead of alloc/free per step;
(c) overlap stage-out of layer *k* with the update of layer *k+1* on a copy
stream (per-layer double-buffer — FASE's fused per-layer epilogue already
iterates layers, so the seam exists). *Impact: no VRAM change; protects the
"no speed regression" constraint at 1B scale. The 16M measurement (opt step
14→31 ms, micro-batch unchanged) does not extrapolate to 1B without this.*

**P0.3 Make fp16 moments compose with offload.** They are hard-mutually-exclusive
today (`stmt.rs:4279-4287`) because the requant copy-back cannot cross devices.
Host-side fp16 m/v halves staging traffic (the dominant offload cost at 1B) and
halves host RAM; the dequant→f32-update→requant can run in the same on-device
epilogue that already exists for the #369 cast envelope. *Impact: speed
protection for offload, −8.3 GB host RAM; quality gated by the existing WGGO
sensitivity/role ceilings (embedding/LM-head stay f32).*

### P1 — Activation checkpointing on the decorator-free source-AD path (the big lever)

This is the wall at 1B (OOM now lands in fwd/bwd compute) **and** the wall on
context at 500M (OOM at seq 1024). It is also lever (a) in
WGGO_SCALING_RESULTS.md's own ordering, and the CCR paper
(`docs/research/CCR.pdf`) is the designed solution — its evaluation matrix
already targets a 400M model at seq 2048 on this exact 16 GB card.

Staged to keep the speed constraint honest (the backward is now FLOP-bound
after #370, so recompute overhead is real):

- **P1.a Block-granular recompute (CCR phases 1–2 analog).** Extend
  `apply_checkpoint_policy` (`source_ad.rs:1839`) from the attention prologue to
  whole transformer blocks: save only block-boundary residual-stream tensors;
  recompute block interiors during backward, per-segment, freeing each segment's
  intermediates before the next. This alone converts activation memory from
  O(layers × per-layer intermediates) to O(layers × d_model·s + one block's
  intermediates).
- **P1.b Selective policy (Megatron-style, ~5% overhead ceiling).** Never
  recompute matmuls; recompute the cheap/large stuff — norms, RoPE, elementwise
  activations, softmax internals. The attention side already has real machinery:
  tile-granular KV recompute (`synthesize_backward_with_recompute`) and the CSHA
  Tier A/B/C save-spectrum. Wire `"selective"` in the `@checkpoint` surface
  (currently refused) and make a sane default ON for `--pretrain-optimized`.
- **P1.c CCR MILP (phases 3–5) + the C-01 composition.** Per-tensor
  SAVE/RECOMPUTE decisions solved under a byte budget at WGGO time (one
  representative block, <100 ms solve), with FASE-Deferred's freed gradient
  buffer published to the activation budget (Cross-Paper C-01: reassign
  freed-pool pages to the activation pool). This is what keeps net overhead
  ≤~5% and can turn it *negative* on threshold models by flipping attention
  layers to cheaper tiers.

*Impact: unblocks 1B stepping; unblocks 500M at seq ≥1024 (context constraint);
2–5× activation reduction at 2–10% compute per CCR, offset by eliminating the
allocator-pressure stalls (2.0 s IQR) that come from running at 95% VRAM.
Quality: bit-exact save/recompute equivalence is testable (CCR §5.4) — no
numerics change in the P1.a/P1.b MVP.*

**Speed gate:** median micro-batch time at 500M/seq 512 must stay within 5% of
the #370 baseline (2819 ms) with checkpointing ON; the win is banked as context
and model scale, not paid as throughput.

### P2 — BF16 weights + f32 master weights

Halves the weight surface (4.15 → 2.07 GB at 1B) **and roughly halves live
activation bytes** (activations inherit weight dtype through the graph), which
compounds with P1 on the same wall. Prefer **bf16 over fp16**: no loss scaling
needed (none exists in the codebase — zero hits for
loss_scale/GradScaler/autocast), same exponent range as f32.

Design points grounded in the audit:
- The GPU path is canonically f32 today (`zero.rs:197-204, 454-462` refuse
  non-f32); f16-in-SMEM/f32-accumulate already exists in flash-attention v2 and
  bf16/fp16 activation paths exist for the fused LM-head CE — so kernels are
  closer than the optimizer plumbing is.
- Master f32 weights live **host-side with the offloaded m/v** (they ride the
  same pinned staging path from P0.2), so the device holds only bf16 weights.
  Update math stays f32 on-device (FASE≡AdamW exactness preserved); the bf16
  cast-back happens in the same fused epilogue.
- Keep f32 for norms/embedding/LM-head per the existing WGGO role ceilings.
- `wggo_cost.rs:418-423` already flags "a separate master_dtype_bytes is a
  future refinement" — implement it so the budget planner accounts this
  honestly.

*Impact at 1B: −2.1 GB weights, −~50% of whatever activation footprint remains
after P1, and likely speed-positive (Blackwell bf16 tensor-core matmul).
Quality gate: 500M convergence run, loss parity to the established 3–4 dp
standard vs f32 baseline.*

### P3 — Retire the device `m_partial` surface at scale

The last full param-sized f32 device surface (−4.15 GB at 1B). Two options, in
preference order:

1. **Offload `m_partial`** onto the P0.2 pinned/overlapped path. Caveat: it is
   touched **every micro-batch** (not once per window), so staging cost is
   per-micro-batch; only viable after P0.2's overlap, and should be
   layer-pipelined against the backward like the optimizer epilogue.
2. **Chunked window accumulation**: keep `m_partial` sharded into K device
   chunks with host backing, streaming chunks through a fixed-size device
   working set during the accumulate step.

**Hard constraint:** preserve exact-windowed FASE semantics. `v` needs
`(Σᵢgᵢ)²`; any partial-free scheme is Option-B `mean(g²)` which changes
numerics by `(1−β₂)·Var(g)` per window and is pinned against by
`fase_numerical_validation.rs`. Do not trade this for memory — that is a
quality regression by definition.

*Note on ordering: P3 is smaller scope than P2 and can be pulled ahead if P2
stalls; P2 ranks higher because it also cuts activations and is
speed-positive, while P3 only cuts a fixed 4 B/param and carries staging-cost
risk.*

### P4 — Native GQA flash backward (kill the expand-KV envelope)

The GPU GQA backward currently materializes expanded K, V, dK, dV at full head
count (`expand_kv_heads_device`, `flash_attention.rs:550-602`) — at the 1B
config (16Q/4KV) that is 4× KV inflation, four transient tensors, plus
canonical-copy staging, plus a reshape+`sum_dim` reduction, every backward.
Cost grows linearly with seq, so this is specifically a **context-length
protection** item. Write a grouped backward kernel that indexes KV by
`h / group_size` (the forward already does this) and accumulates dK/dV across
the group in-kernel. *Impact: removes O(b·h·s·d) transient memory and the DtoD
expand/reduce bandwidth from every step; strictly speed-positive.*

### P5 — Allocator near-ceiling behavior

- Soak and then default-on `NSL_ASYNC_ALLOC=1` (stream-ordered
  `cuMemAllocAsync`) for training; today it is opt-in and the OOM ladder only
  suggests it post-mortem.
- Transient-pool trim/defrag between micro-batches when peak > ~90% VRAM;
  the 500M 2.0 s IQR and the seq-1024 OOM at 15.8/16 GiB both smell of
  fragmentation slack, and P0.1 will confirm.
- Deferred from #370 and still correct to defer until after P1: the per-op
  sync/stream-ordered allocator rewrite interlock — do it here, once, with the
  allocator work, not piecemeal.

*Impact: recovers the gap between "allocatable" and "physically present" VRAM,
and removes tail-latency stalls — a speed and stability win that makes running
at high occupancy safe rather than lucky.*

### P6 — Later / opportunistic

- **CCR phases 6–7:** SAVE_CFIP compressed activation saves (IF4/IM3/FP8) and
  outlier-separated compression — turns some RECOMPUTE decisions back into
  cheap SAVEs; Cross-Paper C-02 (CFIP-in-HBM Tier C saves) claims ~3× smaller
  attention save-buffers **with** 5–10% throughput gain.
- **8-bit moments lowering.** The ILP already searches {32,16,8}
  (`wggo_ilp.rs:450`) but only fp16 is lowerable; CPDT Part II's
  sensitivity-scored mix (~1.3 B/param average optimizer state) plus
  stochastic-rounding on embeddings is the designed endgame. Matters less once
  m/v are host-resident; matters more for the no-offload single-device story.
- **CESH low-rank embedding decomposition** — opt-in only; real lever for
  big-vocab configs, quality-risky (F-23), keep behind an explicit annotation.
- **Multi-GPU ZeRO (CPDT Part I / errata E1 stage enum)** — real, but out of
  scope for the 16 GiB single-card goal; the offload work above is deliberately
  shaped as the single-GPU ZeRO-Offload analog so the partitioning story slots
  in later without rework.

---

## 3. Explicit non-levers (rejected for this goal)

- **CGAC** (gradient all-reduce compression): multi-node only, and it *adds*
  error-feedback memory (~1 GB at 1B). Skip until multi-node infra exists.
- **Option-B second-moment approximation** to drop `m_partial`: quality
  regression, pinned against by tests. No.
- **Structured sparsity (CSS) / pruning as a memory tool:** changes the model;
  violates the quality constraint.
- **fp16 weights without a master copy:** known convergence risk; bf16+f32
  master (P2) dominates it.

---

## 4. Projected budget at 1.037B (16 GiB card)

| After | Device-resident persistent | Activations/transients | Verdict |
|---|---|---|---|
| today + offload | 8.3 GB (w 4.15 + m_partial 4.15) | all intermediates live → OOM at 15.8 GiB | fails in matmul |
| + P1 checkpointing | 8.3 GB | ~1 block live + boundaries, est. 2–4 GB | **first 1B step lands** |
| + P2 bf16 weights | 6.2 GB (w 2.07 + m_partial 4.15) | halved again | headroom for seq/batch growth |
| + P3 m_partial retired | ~2.1 GB | — | 1B comfortable; context becomes the budget's own dial |

(500M inherits the same levers as headroom: seq 1024–4096 in reach after P1
alone, per the P0.1 accounting to be confirmed.)

---

## 5. Validation discipline (applies to every item)

Per WRGA Appendix B and the #370 precedent:
- **Loss parity 3–4 dp** vs baseline for every A/B pair (offload, fp16 moments,
  checkpointing, bf16); 500M convergence run for P2.
- **Probe the actual code path** (B.5): a failure-injection test per GPU path
  asserting no silent CPU fallback — the #369/#370 root causes
  (stride-guard CPU fallback, sum_sq std bug) were exactly silent-path bugs.
- **Measure at target scale on a clean substrate** (B.3/B.6): re-run the
  scaling matrix per phase; peak VRAM per P0.1 surface report, not just
  nvidia-smi polling.
- **Speed gate per phase:** median micro-batch within 5% of the #370 baseline
  at the reference shapes (16M base, 500M/seq 512, seq-sweep 512→8192).
- Commit raw benchmark JSON (compressed) with each matrix run.
