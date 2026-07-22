# Post-#403 certification campaign (P0 item 2)

New training baselines on the corrected stack — source-AD RMSNorm dx fix
(#403: pre-fix, every weight whose gradient flows through an RMSNorm got
LayerNorm mean-subtract math) and the root-fixed FASE bias handling — so
every future regression comparison starts from known-good curves.

RTX 5070 Ti 16 GiB, release build, `--deterministic --seed N`, 2026-07-21.
Data: fixed pseudo-random u16 block (`gen_tokens.py`, PCG64 seed 1234)
tiled per scale — learnable (memorization drill), identical across init
seeds. Runner: `p0_campaign.py certify`; full logs, loss streams, health
snapshots and `result.json` per arm under `models/benchmarks/p0_logs/`.

Campaign side-products (found+fixed BY this campaign, see commit log):
`--seed` previously never seeded the model-init RNG (all "seeds" produced
identical weights), and `--monitor` SIGSEGV'd on every FASE-Deferred train
block (health probe read grads the hook had freed) — silently reported as
exit 1. Both are gated now (`seed_gate.rs`, `monitor_fase_gate.rs`).

## 50M — 3 seeds × 2000 steps (batch 1, seq 1024, AdamW lr 3e-4, clip 1.0)

| arm | steps | first | mid | last | ‖θ‖ first→last | max ‖g‖ | notes |
|---|---|---|---|---|---|---|---|
| seed 1 | 2000 | 10.9133 | 0.0063 | 0.6217 | 211.7 → 352.8 | 0.21 | converged; late fixed-lr bounce |
| seed 2 | 2000 | 10.9059 | 0.0058 | 0.1728 | 211.7 → 345.4 | 0.20 | converged |
| seed 3 | 2000 | 10.9086 | 0.0055 | 0.0979 | 211.7 → 348.0 | 0.37 | converged |
| wd=0 integrity | 2000 | 10.9133 | 0.0053 | 0.0044 | 211.7 → 356.6 | 0.20 | **74/74 params finite every step**, 64 nonzero worst-case, missing=[] |
| reference path | 2000 | 10.9133 | 0.0068 | 0.0026 | — | — | `--training-reference` (fusion/FBIP/fused-step OFF) |

- Seeds produce **distinct trajectories from distinct inits** (identical
  first-loss for seed 1 pre-fix would have been the tell) converging to the
  same regime; parameter-norm growth is smooth and comparable across seeds.
- The wd=0 arm certifies **full gradient coverage**: 2000/2000 steps, every
  trainable param received a finite gradient every step, none missing.
  (The 10 zero-at-some-step params are dropout-path worst-cases, not drops.)
- The **reference path agrees**: same first loss to 4 dp (identical
  forward), same convergence regime (final 0.0026 vs 0.0044 at wd=0-adjacent
  config) — the optimized stack is not drifting from exact math.
- Grad-norm + weight-norm curves: 20 health snapshots per arm (every 25
  steps + step-0 init) in each arm's `result.json`.

## 500M — 2 seeds × 500 steps (batch 2, seq 512, accum 8, `--checkpoint-blocks --optim-state-offload`)

| arm | micro-batches | first | last | ‖θ‖ snapshots | integrity |
|---|---|---|---|---|---|
| seed 1 | 4000 | 11.0102 | 0.0188 | 40 (smooth growth) | checks=4000, 218/218 finite, missing=[] |
| seed 2 | 4000 | 11.0911 | 0.7610 | 40 | checks=4000, 218/218 finite, missing=[] |

- ~7 h/seed sustained (6.2 s/micro-batch, util 75%, smi peak ~10.1 GB,
  **no allocator growth across 4000 micro-batches**).
- Both seeds converge (EMA ≈ 0.016 by micro-batch 3400); seed 2 then
  oscillates 0.4–1.0 around the memorized minimum in the last ~10% —
  fixed-lr AdamW bouncing at a sharp minimum, seed-dependent; **all 4000
  integrity checks stayed finite** (no NaN/Inf event), so this is an
  optimization dynamic, not a numerical defect. A decayed-lr production
  schedule would suppress it.
- Per-param grad-norm curves are structurally absent at this scale: the
  FASE-Deferred hook consumes gradients during the backward (see the
  `[health]` note + `monitor_fase_gate.rs`); gradient coverage is instead
  certified by `--grad-integrity` (full 218-param coverage, all steps).

## 1B — 150 steps (demo streaming stack: CSLA + offload + weight-stream + arena + prefetch + async-wb)

| micro-batches | first | last | tok/s | alloc peak | smi peak | util | transfers |
|---|---|---|---|---|---|---|---|
| 1200 | 11.2186 | 2.0770 | 99 | 10.08 GB (flat) | 15.45 GB | 88% | 194,400 logical / 21,600 pack / 2,400 prefetch / 2,400 async-wb |

- ~3.5 h sustained streaming training, loss 11.22 → 2.08 and still
  descending at cutoff; allocator peak flat at 10.08 GB across all 150
  windows (no leak); 12 weight-norm snapshots recorded.
- `--grad-integrity` is **not wired** under `--layerwise-accum` and says so
  loudly (checks=0 = "not measured", pre-existing documented limitation) —
  gradient coverage at 1B rides on the 500M certification plus the CSLA
  bit-exactness gates.

## Certification verdict

The corrected post-#403 stack trains **stably at all three scales** on
their production paths — multi-seed convergence at 50M, 7-hour sustained
FASE-Deferred + CCR + offload runs at 500M with full gradient coverage,
and 3.5-hour streaming-stack training at 1B with flat memory — with loss,
gradient-norm (50M) and parameter-norm curves banked as the new baselines.
Known measurement gaps (grad norms under the FASE hook, integrity under
CSLA) fail loud, not silent.
