# WGGO all-six-subsystems pretraining benchmark — results (2026-07-14)

**Setup.** `models/benchmarks/wggo_all6_pretrain.nsl`: 16.3M-param GQA packed LM
(d_model 512, 8 Q / 2 KV heads, head_dim 64, 6 blocks, FFN 2048, vocab 4096,
weight-tied head), packed corpus of 192-token documents, seq_len 1024, batch 4
(~5.3 docs/sequence), grad_accumulation 4 + grad_clip 1.0 (FASE Deferred +
two-phase clip), AdamW(1e-3, wd 0.01, β 0.9/0.95). 524K tokens → 128
micro-batches → 32 optimizer steps. RTX 5070 Ti, CUDA 13.x, release build,
`nsl run --source-ad --pretrain-optimized` (+`--wggo-report`).

Three configurations, identical data/init:

| | **WGGO + fused packed (Stage C)** | WGGO + decomposed (`NSL_SDPA_FUSED_DISABLE=1`) | no WGGO (`--source-ad` only) |
|---|---|---|---|
| loss first → mid → last | 8.4452 → 0.856 → **0.0236** | 8.4452 → 0.886 → 0.0235 | 8.4452 → 0.856 → 0.0236 |
| micro-batch time mean | **2264 ms** | 3140 ms | 2256 ms |
| micro-batch time p50 / p95 | 2226 / 2396 ms | 3106 / 3276 ms | 2218 / 2386 ms |
| throughput (real tokens) | **1809 tok/s** | 1304 tok/s | 1815 tok/s |
| wall (128 micro-batches) | **291.1 s** | 403.2 s | 290.0 s |
| VRAM peak / median | 5566 / 4032 MiB | 5441 / 3950 MiB | 5617 / 4088 MiB |
| GPU util (0.5 s samples, mean) | 21.7 % | 14.1 % | 22.6 % |
| fused kernel fired | yes (`segmask=1, hd=64, 64x64`) | no (kill switch) | yes |

**Headlines.**

- **Stage C fused packed attention: 1.39× step-time speedup** (3140 → 2264 ms,
  −28%) over the Stage B decomposed path at identical numerics (first-step
  loss agrees to 3e-6; final loss 0.0236 vs 0.0235 — same trajectory).
- **WGGO planning is overhead-free at runtime**: with the plan consumed
  (per-layer report below) the step time equals the plan-free run (2264 vs
  2256 ms — noise). WGGO's value here is decisions + honesty, and it chose
  the configuration the numbers validate.
- **VRAM cost of fused: +125 MiB peak** (LSE saves + f16 staging), −11% of a
  GB — negligible against the 28% time win.

**Per-layer plan (all six subsystems, from `--wggo-report`):**
each of the 7 layers (`other`, `blocks.0..5`):
`8/8 heads (CEP evidence gate: random init ⇒ keep-all), FFN=2048,
CSHA-L3 requested → Level2 applied per layer (live smem negotiation:
`smem_208kb_exceeds_180kb`), LoRA r=0 [none] (WRGA: no adapters in
pretraining), m=32b v=32b (CPDT moment precision: opt-in not set),
FASE=fused (Deferred windowed AdamW under two-phase clip),
PCA=segment_id → consumed: "fused segment-masked flash kernel (Stage C
plain family; decomposed fallback when the runtime declines)"`.

**Honest findings (recorded, not hidden).**

1. GPU utilization is low (~22%) in all configs: per-op host glue +
   allocation churn dominate between kernels at this model size, and at
   least one adjoint chain (rmsnorm gamma) lowers on CPU (surfaced by the
   one-shot `add_inplace: reconciling` warning). Kernel-level wins are real
   but the end-to-end ceiling is host-bound at 16M params.
2. Tier-B tile-skip promotion is DISABLED (loud deferral in
   `ensure_sdpa_fwd_variant_table`): the tile-skip kernel deviates ~1e-2 at
   s=1024 while the base fused kernel matches the oracle at 3e-6. The
   measured speedup is from fusion alone; per-document tile skipping is
   additional headroom once Tier-B passes a packed-shape parity gate.
3. This benchmark's first production-scale run of the packed path surfaced
   and fixed three latent v2-segment bugs (Tier-B PTX entry name, Tier-B
   launch SMEM sizing, per-batch segment staging) — see commit history.

Raw metrics JSON + per-run stdout/stderr: benchmark driver at
`scratchpad/bench/run_bench.py` (session-local), `metrics_{fused,decomposed,nowggo}.json`.
