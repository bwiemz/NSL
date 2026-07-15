# WGGO scaling-campaign results (2026-07-15)

Follow-up to `WGGO_MATRIX_RESULTS.md` (PR #369). PR #369's matrix identified
the decomposed backward as host-glue-bound (9–50× the forward, 9–36% GPU
util) and named six levers. This campaign implemented all six, closed a
15-agent adversarial review, and re-ran the full validation matrix on the
fixed binary. RTX 5070 Ti (16 GB, sm_120), CUDA 13.x,
`nsl run --source-ad --pretrain-optimized`, one frozen post-review binary,
5 repeats per cell (3 for A/B pairs), first 2 micro-batches dropped as
JIT/alloc warmup, medians with IQR + stdev, `NSL_PHASE_TIMING=1` for the
fwd/bwd/opt split, `nvidia-smi` 0.5 s VRAM/util polls. Useful tok/s counts
only label ≠ −100 positions.

## Headline: the backward is no longer host-glue-bound

nsys root-cause on the 16M base config: **~50 % of ALL process CPU time was
`flash_attention_backward_cpu_gqa`** — the PCA Stage-C stride guard routed
every decorator-free attention backward to the CPU reference (dout always
arrives as a non-contiguous transpose view), on top of the classic MHA-only
kv gate. The optimizer step was a second host sink: 5 full-tensor CPU
round-trips per parameter (add_inplace / mul_scalar_inplace / sum_sq had no
GPU kernels). Fixed (commit `9a76bf6c`): flash backward materializes canonical
device copies + a GQA expand-KV envelope; GPU in-place optimizer ops; GPU
embedding-backward scatter; resolve-once CUfunction cache.

**Result — the backward is now bounded, and GPU util tripled:**

| seq (8192 tok/mb) | med ms/mb #369 → now | fwd ms | bwd ms | useful tok/s #369 → now | GPU util #369 → now |
|---|---|---|---|---|---|
| 512×16 | 3641 → **1113** | 325 | 760 | 2234 → **7305** | — → 61 % |
| 1024×8 | 4092 → 1137 | 340 | 772 | 1990 → 7161 | — → 64 % |
| 2048×4 | 4721 → 1195 | 370 | 799 | 1725 → 6819 | — → 64 % |
| 4096×2 | 5862 → 1309 | 427 | 856 | 1390 → 6223 | — → 67 % |
| 8192×1 | 7753 → **1156** | 414 | 712 | 1051 → **7046** | 9 % → 62 % |

Across a 16× sequence range the micro-batch time now grows only 1.04× (was
2.1×); the backward stays **712–856 ms** where #369 scaled it to 7258 ms. GPU
util is **57–80 %** across the whole matrix (was 9–36 %). The #369 report's
"next perf lever" is spent: the decomposed backward is FLOP-bound again.

## Full matrix (32 configs, 140 runs, 0 failures)

See `WGGO_MATRIX_TABLE.md` for the machine-generated table; excerpts below
(median ms/mb, then fwd/bwd/opt ms, then useful tok/s).

### Fused vs decomposed (NSL_SDPA_FUSED_DISABLE=1) — the win grew

| config | fused | decomposed | speedup | loss (f/d) |
|---|---|---|---|---|
| base (docs 192) | 633 | 3114 | **4.9×** | 6.4376 / 6.4362 |
| seq 4096 | 1309 | 8705 | **6.6×** | 6.3535 / 6.3470 |
| realistic docs | 688 | 6033 | **8.8×** | 7.2240 / 7.2119 |

The fused advantage jumped from #369's 1.41–1.51× because the fused/flash
backward is now GPU-resident while the decomposed fallback still runs its
slow adjoint chain (its backward is 2891–7862 ms, 9–15 % util). Losses match
to 3 decimals.

### Tier-B tile-skip (NSL_SDPA_TIER_B_DISABLE=1)

| config | fwd on/off (ms) | fwd gain | step gain | loss on/off |
|---|---|---|---|---|
| docs 192, seq 1024 | 172 / 246 | 1.43× | 1.11× | 6.4376 / 6.4377 |
| docs 32, seq 1024 | 159 / 245 | 1.54× | 1.14× | 5.4130 / 5.4130 |
| docs 192, seq 4096 | 427 / 1147 | **2.69×** | 1.56× | 6.3535 / 6.3535 |

Losses identical in every pair. The whole-step gain rose (1.11–1.56× vs
#369's 1.04–1.12×) now that the forward is a larger share of the (much
smaller) step.

### Optimizer-state offload (--optim-state-offload)

| config | med ms/mb | opt step ms | loss | peak VRAM |
|---|---|---|---|---|
| off | 633 | 14 | 6.4378 | 5737 MiB |
| on | 632 | 31 | 6.4377 | 5691 MiB |

At 16 M the offload is **free** on the micro-batch (the ~2× opt-step cost,
14→31 ms, amortizes over the accumulation window) and loss-identical — the
update runs on-device in f32. The VRAM saving is 2× the state, which at 16 M
is small but at 1 B is ~8.3 GB (below).

### Sequence, batch, accumulation, heads, boundaries

- **Batch** scales near-linearly (192/376/633/1157 ms for b=1/2/4/8); useful
  tok/s *improves* with batch (5297 → 7036).
- **Accumulation** is free per micro-batch (693/632/632 ms at accum 1/4/8);
  the opt step is 6 ms at accum 1, 14 ms at accum 4/8.
- **Head geometry**: hd128-MQA fastest (615 ms), hd64-MHA slowest (649 ms) —
  a 6 % spread; KV-head count barely matters. (In #369 hd32 was the slowest
  by 22 %; the backward fix flattened the geometry sensitivity.)
- **Boundary density**: useful tok/s ranges 6521 (docs 64) → 5577 (single
  doc) — a **1.17× swing, down from #369's 6×**. The swing collapsed because
  the backward host-glue that scaled with attention work is gone; document
  density now moves throughput only through the forward's real attention FLOPs.

### Long run (125 micro-batches, base geometry)

**632 ms/mb — 3.3× faster than the #369-era 2106 ms** on the same geometry.
Loss 8.445 → 0.0236.

## 500M model (503M: d1536, 18 blocks, 12Q/4KV hd128, FFN 6144, vocab 32k)

seq 512, batch 1, accum 8, realistic docs, 28 micro-batches:

* median **2819 ms/mb — 2.7× faster than #369's 7.69 s** on the same config.
* fwd **151 ms**, bwd **2634 ms** (was 7.41 s → **2.8×**), optimizer step
  **284 ms** (was 13.8 s → **48×** — the GPU in-place optimizer ops kill the
  per-param CPU round-trips that scaled with the 503M parameter count).
* loss 10.82 → 8.98; peak 15.7 GiB; util ~30 %. The optimizer 48× and the
  backward 2.8× both come from removing CPU work that scaled with parameter
  count, so the win is larger here than at 16 M.

## 1B model (1.037B: d2048, 22 blocks, 16Q/4KV hd128, FFN 8192): offload moves the wall

seq 512, batch 1, accum 8, realistic docs (measured, backtraces captured):

| config | result | peak | OOM site |
|---|---|---|---|
| AdamW, no offload | never trains | 15835 MiB | `nsl_tensor_zeros_like` (optimizer-state allocation, before the first forward) |
| AdamW, `--optim-state-offload` | passes state alloc, then OOM | 15805 MiB | `nsl_tensor_matmul` (inside the forward/backward) |

The offload version's backtrace lands in `nsl_tensor_matmul`, not
`nsl_tensor_zeros_like` — direct evidence that host-resident m/v cleared the
FIRST wall (state allocation) and pushed the failure into forward/backward
compute, exactly what a ZeRO-Offload analog should do.

Offloading m/v frees the largest single surface (~8.3 GB) and clears the
FIRST wall — state allocation — exactly what a ZeRO-Offload analog should do.
But 1B on 16 GB still needs a SECOND lever: f32 weights (4.15 GB) + one more
full param-sized surface (accum=1 gradient buffer, or the accum>1 FASE
m_partial window accumulator, ~4.15 GB each) + the source-AD backward's live
adjoint intermediates still exceed 16 GB. Characterized next levers, in
order: (a) activation checkpointing on the decorator-free path (the source-AD
backward holds all intermediates to end-of-backward); (b) extend offload to
m_partial (per-micro-batch staging traffic); (c) fp16 weights. The offload
envelope is the reusable infrastructure those build on. Note `--wggo-memory-
budget` (fp16 moments) saves less than offload (4.15 vs 8.3 GB) and the two
are mutually exclusive (the requant copy-back cannot cross devices), so on
this 16 GB card the budget path alone does not reach 1B either.

## Adversarial review (15 agents) — 3 real bugs found and fixed (commit ce9aa75b)

- **HIGH:** `nsl_tensor_sum_sq`'s GPU fast path returned the population STD,
  not Σx² (it read `gpu_tensor_stats_f32()[3]`, whose contract is
  `[min,max,mean,std]`), silently disabling GPU gradient clipping in FASE.
  Fixed with a dedicated raw-Σx² helper + a GPU parity gate. **The matrix was
  re-run on the fixed binary because this changed GPU clipping behavior.**
- **MEDIUM:** `--wggo-memory-budget` could pick 8-bit moments accounted at
  1 byte but stored as fp16 (2 bytes) → runtime OOM. Fixed: moment decision
  space restricted to {32, 16}.
- **MEDIUM:** the budget hard-refusal false-failed globally-fittable models
  (keyed on the even per-layer share, not the global role-aware floor). Fixed
  + boundary regression test.

## Honest notes
- Phase timing was on for all runs (device syncs; µs against multi-second /
  hundred-ms steps).
- Deferred: per-op `cuCtxSynchronize` removal (item 1e) — the caching
  allocator recycles freed blocks immediately, safe only because every op
  syncs; making it stream-ordered is a separate large change, not worth the
  corruption risk against the 3.3–48× already banked.
- Config generator + driver: `matrix_gen_model.py`, `matrix_configs.json`
  (this commit); raw per-run JSON is session-local.
