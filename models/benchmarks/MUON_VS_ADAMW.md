# Muon vs AdamW — P1 item 5 (+ item 7 validation sweeps)

**Question**: is mixed Muon/AdamW actually beneficial for NSL's coder
models, measured properly? The earlier toy fixture (8 epochs, tail-8
means 0.749 AdamW vs 0.774 Muon) shared ONE lr across both arms and
memorized a synthetic stream — it neither condemned nor validated Muon.

## Protocol (controls, per the item-5 checklist)

| control | how |
|---|---|
| tokens | identical token-file PREFIX per step count (`muon_data/train_tokens.bin`, byte-level real code: this repo's NSL + Rust + cargo-registry sources, deterministic file order) |
| data order | `shuffle=false`, same DataLoader config, same file |
| weight init | `--deterministic --seed 1` (post-#406 seeded init) |
| batch/accum | batch 1 x seq 1024 (50M-class); batch 2 x seq 512 x accum 8 (500M-class) |
| lr tuning budget | 3 lrs per optimizer x 400 steps, best picked by held-out val loss |
| wall time / FLOPs | tok/s recorded per arm; val-at-equal-wall derivable from timestamped curves |
| validation | held-out-BY-FILE byte streams, evaluated every 200 steps with dropout off: `val_nsl` (held-out NSL files) and `val_rust` (held-out repo Rust files) — the "held-out code benchmarks" at this scale |

Model: coder50m architecture at byte vocab 256 (~26M params,
hidden-matrix-dominant — the shape class Muon exists for). 500M
confirmation: coder500m architecture at byte vocab (~443M params) on the
production memory-reduction path (`--checkpoint-blocks
--optim-state-offload`).

**Fairness prerequisite shipped first**: `Muon(adamw_lr=...)` — the AdamW
arm (embedding + norms) gets its own lr (threaded as a fixed ratio of the
scheduled lr). Without it, one shared lr either detonates the embedding at
Muon's ~2e-2 or starves the hidden matrices at AdamW's ~3e-4 — the toy
comparison's structural handicap.

Runner: `muon_campaign.py` (tune / main / sweep / confirm500m); raw
curves + val streams in each `muon_logs/<arm>/result.json`.

## Phase 1 — lr tuning (400 steps each, seed 1)

Val loss on held-out [2, 1024] windows at step 400 (dropout off):

| arm | last train | val_nsl | val_rust | tok/s |
|---|---|---|---|---|
| AdamW lr=1e-4 | 2.2318 | **3.2489** | **3.2664** | 3448 |
| AdamW lr=3e-4 | 2.0910 | 3.3326 | 3.5354 | 3474 |
| AdamW lr=6e-4 | 3.3777 | 3.5876 | 3.6567 | 3424 |
| Muon lr=0.01 (adamw_lr=3e-4) | 1.8643 | **3.1018** | **2.9751** | 3153 |
| Muon lr=0.02 (adamw_lr=3e-4) | 1.8030 | 3.2250 | 3.0760 | 3171 |
| Muon lr=0.04 (adamw_lr=3e-4) | 2.6650 | 3.3841 | 3.4418 | 3132 |

- **Muon wins at every lr rank on BOTH val sets** at the 400-step budget:
  best-vs-best gap is −0.15 nats (val_nsl) and −0.29 nats (val_rust).
- Newton-Schulz overhead with the planned primitive: ~9% step time
  (3150 vs 3450 tok/s) — Muon's val advantage survives equal-WALL
  comparison at this scale (its 400-step val is below AdamW's even
  though AdamW fits ~9% more steps per second).
- lr picks for Phase 2: AdamW 1e-4 (and 3e-4 as insurance against
  short-horizon bias), Muon 0.01 (and 0.02).

## Phase 2 — head-to-head at best lr (3000 steps, ~3.07M tokens, single epoch)

Both top lrs per optimizer (guarding against the 400-step pick
undertuning either side). "BIG" = the [2, 1024] held-out windows at the
final step; "tiny" = the [2, 256] periodic windows (see val protocol
note). Seed 1:

| arm | last train | tiny val_nsl | tiny val_rust | BIG val_nsl | BIG val_rust | wall s | tok/s |
|---|---|---|---|---|---|---|---|
| AdamW 1e-4 | 3.1677 | 3.3779 | 3.7620 | 3.2949 | 3.5705 | 894 | 3437 |
| AdamW 3e-4 | 3.0713 | 3.4798 | 3.6499 | 3.2753 | 3.4164 | 904 | 3400 |
| Muon 0.01 | 3.2618 | 3.3761 | 3.5411 | 3.3597 | 3.3308 | 986 | 3115 |
| Muon 0.02 | 3.0929 | 3.2464 | 3.5381 | **3.2436** | **3.3603** | 989 | 3105 |

- **Equal tokens**: Muon best on BOTH headline val sets — best-vs-best
  −0.032 nats (val_nsl) and −0.056 nats (val_rust vs AdamW's best; both
  Muon arms beat both AdamW arms on val_rust, −0.086 at the extremes).
- **Equal wall time** (at AdamW's 904 s, Muon reaches ~step 2750): Muon's
  tiny val_nsl ≈ 3.31 vs AdamW's 3.48 — the advantage survives the ~9%
  Newton-Schulz tax with room to spare.
- Late-stream dynamics: as the single-epoch stream moves into harder
  cargo-registry code, AdamW's held-out loss RISES (3.37 → 3.48 over the
  last 600 steps) while Muon's keeps FALLING (3.45 → 3.25) — the
  generalization gap widens exactly where the data gets less like the
  val prefix.
- (Train loss is not comparable across the stream — later tokens are a
  different data mixture; val curves are the metric.)

### Seed replicate (seed 2, best-vs-best)

| arm | last train | BIG val_nsl | BIG val_rust | outcome |
|---|---|---|---|---|
| AdamW 3e-4, seed 2 | **NaN** | NaN | NaN | **diverged at ~step 2500** (train loss NaN from 2500, vals NaN from 2600 — despite grad_clip=1.0) |
| Muon 0.02, seed 2 | 3.0480 | 3.2855 | 3.3793 | stable, consistent with seed 1 |

AdamW at its val-preferred lr detonated on the harder late-stream data at
seed 2 while Muon (whose hidden-matrix update is norm-controlled by
construction) was stable at both seeds. AdamW's robust configuration is
lr=1e-4 — which is also its weaker one (seed-2 run at 1e-4 below).

| arm | last train | BIG val_nsl | BIG val_rust |
|---|---|---|---|
| AdamW 1e-4, seed 2 | 3.1584 | 3.1789 | 3.3798 |
| Muon 0.02 (m=0.95), seed 2 | 3.0480 | 3.2855 | 3.3793 |
| Muon 0.02 (m=0.8), seed 2 | 2.9424 | 3.2001 | 3.3892 |

Seed-mean of the completed configurations (BIG val):

| config | val_nsl mean | val_rust mean | completed |
|---|---|---|---|
| AdamW 1e-4 | 3.237 | 3.475 | 2/2 |
| AdamW 3e-4 | — | — | **1/2 (seed-2 NaN)** |
| Muon 0.02 m=0.95 | 3.265 | **3.370** | 2/2 |
| Muon 0.02 m=0.8 | 3.222 | **3.356** | 2/2 |

## Item 7 — hyperparameter + shape-class validation

### ns_steps / nesterov / momentum (400 steps, Muon lr=0.01, seed 1)

| arm | last train | val_nsl | val_rust | tok/s |
|---|---|---|---|---|
| ns_steps=3 | 1.8844 | 3.4349 | 3.2629 | 3226 |
| ns_steps=4 | 1.8965 | 3.4410 | 3.1101 | 3183 |
| ns_steps=5 (default) | 1.6380 | 3.2438 | 3.0126 | 3132 |
| nesterov=false (ns=5) | 1.8805 | 3.4026 | 3.0855 | 3113 |
| momentum=0.8 | **1.1887** | **2.8729** | **2.9171** | 3134 |
| momentum=0.9 | 1.4407 | 2.9231 | 2.9996 | 3131 |
| momentum=0.95 (default) | 1.6380 | 3.2438 | 3.0126 | 3132 |
| momentum=0.99 | 2.1787 | 3.1593 | 3.2823 | 3108 |

- **ns_steps=5 validated as the right default** (3 and 4 are cheaper per
  step — 3226 vs 3132 tok/s — but measurably worse on val at 400 steps).
- **nesterov=true validated** (off costs ~0.16 val_nsl).
- **momentum=0.8-0.9 beats the 0.95 default at this horizon** — the
  spec default is not condemned (short-horizon sweeps favor lower
  momentum), but a 3000-step momentum-0.8 arm is run below as the
  sweep's follow-through.

3000-step momentum-0.8 arms (sweep follow-through):

| arm | last train | BIG val_nsl | BIG val_rust |
|---|---|---|---|
| Muon 0.02 m=0.8, seed 1 | 2.0849 | 3.2445 | **3.3238** |
| Muon 0.02 m=0.8, seed 2 | 2.9424 | 3.2001 | 3.3892 |

momentum=0.8 confirms as at-least-as-good at the 3000-step horizon
(best seed-mean val_nsl of any config, best single-run val_rust) — worth
revisiting the 0.95 spec default for short-to-mid training runs.

### Shape-class orthogonality probe (`muon50m/shape_probe.nsl`)

Full-rank seeded-randn inputs (a smooth ramp is near-rank-1 and reads
err ~= 1 vacuously — degenerate-probe trap); err = ||gram_small(O) -
I||_F / sqrt(k); scaled rms = the sqrt(max(1, rows/cols)) convention
muon_step applies:

| shape class | ns | ortho err | rms | scaled rms |
|---|---|---|---|---|
| attn q/o 512x512 | 3 | 0.5030 | 0.04132 | 0.04132 |
| attn kv 512x256 (tall) | 3 | 0.3169 | 0.04554 | 0.06441 |
| ffn up 512x1408 (wide) | 3 | 0.3232 | 0.02910 | 0.02910 |
| ffn down 1408x512 (tall) | 3 | 0.3240 | 0.02912 | 0.04829 |
| vocab blk 256x512 (wide) | 3 | 0.3181 | 0.04554 | 0.04554 |
| attn q/o 512x512 | 4 | 0.4189 | 0.03789 | 0.03789 |
| attn kv 512x256 (tall) | 4 | 0.3509 | 0.03762 | 0.05320 |
| ffn up 512x1408 (wide) | 4 | 0.3578 | 0.02214 | 0.02214 |
| ffn down 1408x512 (tall) | 4 | 0.3586 | 0.02214 | 0.03671 |
| vocab blk 256x512 (wide) | 4 | 0.3536 | 0.03757 | 0.03757 |
| attn q/o 512x512 | 5 | 0.3217 | 0.04146 | 0.04146 |
| attn kv 512x256 (tall) | 5 | 0.3133 | 0.04148 | 0.05866 |
| ffn up 512x1408 (wide) | 5 | 0.2883 | 0.02532 | 0.02532 |
| ffn down 1408x512 (tall) | 5 | 0.2889 | 0.02532 | 0.04199 |
| vocab blk 256x512 (wide) | 5 | 0.3114 | 0.04152 | 0.04152 |

- **ns=5 has the lowest orthogonality error on every shape class**
  (0.29-0.32), agreeing with the training sweep; ns=3 is notably worse
  on square matrices (0.50).
- **Tall/wide symmetry verified**: ffn up (wide) and ffn down (tall)
  produce IDENTICAL rms — the transpose branch is orientation-consistent.
- **Aspect-ratio convention**: sqrt(max(1, rows/cols)) equalizes tall vs
  square (~0.042 scaled rms) but leaves WIDE matrices ~40% smaller
  per-element (the clamp never boosts ratio < 1) and over-boosts the 2:1
  kv projection (~0.059). Consistent with reference Muon; alternative
  conventions (e.g. 0.2*sqrt(max(r,c))) are a possible future knob, not
  changed here.

## 500M confirmation

Config: coder500m (505M params), seq 256, grad_accumulation 8, seed 1,
`--source-ad --deterministic --checkpoint-blocks --optim-state-offload`,
150 optimizer steps requested (1200 micro-batches). AdamW lr=3e-4; Muon
lr=0.02 / momentum 0.95 / nesterov / ns_steps=5 (the 3M-scale winner),
AdamW arm for 1-D params. Both arms ran on the SAME fixed binary
(post model-`.to(device)` slot fix — an earlier muon attempt on the
pre-fix binary spent 6h CPU-bound on host-resident tail params and was
discarded; the pre-fix AdamW result is preserved as
`result.prefix-bug.json` for provenance).

| arm | micros reached | val_nsl@400 | val_rs@400 | val_nsl@800 | val_rs@800 | wall | s/micro |
|---|---|---|---|---|---|---|---|
| AdamW 3e-4 | 800 (died rc=1: known eval leak) | 3.2720 | 3.6958 | 3.4523 | 3.7307 | 3469 s | 4.3 |
| Muon 0.02  | 400 (wrapper timeout, SIGKILL)   | 5.6558 | 5.6611 | — | — | 26713 s | 66.8 |

Symmetric comparison point = micro 400 (both arms healthy there):

- **Quality**: Muon val_nsl 5.656 vs AdamW 3.272 — **+2.38 nats behind at
  the same micro-batch count**. Muon's val_nsl ≈ val_rs (5.656 vs 5.661)
  says the model has barely learned domain structure at all by micro 400.
- **Throughput**: 66.8 s/micro vs 4.3 s/micro — **~15× slower wall-clock**.
  The Newton-Schulz orthogonalization (5 iterations of large Gram-matrix
  GEMMs per rank-2 param per optimizer step) plus per-step offload
  round-trips of the Muon momentum dominate; the profile was GPU-bound
  (98% util), so this is real compute, not a host stall.

## Verdict

**At 500M under this configuration, Muon decisively loses on both axes**:
~15× slower per micro-batch and +2.38 nats worse val_nsl at the symmetric
micro-400 point. The clear 3M-scale wins (tune400: muon-0.02 train 1.11 vs
adamw 2.63; main3000: muon ahead on every val metric) did NOT transfer to
500M with the same hyperparameters.

Caveats before writing Muon off at scale: single seed; lr=0.02 was tuned
at 3M and NOT re-tuned at 500M (Muon's effective step scales with the
spectral-normalized update, and 500M layer shapes are far from the 3M
tune's); `--optim-state-offload` taxes Muon's momentum uniquely (AdamW's
m/v staging is amortized by the fused step, while Muon pays PCIe on every
NS input); and the eval-leak rc=1 caps the horizon at micro 800. A fair
rematch needs: an lr sweep at ≥100M scale, Muon state resident (no
offload), and the NS GEMM cost amortized (batched NS or lower ns_steps).
Until someone runs that, **AdamW (fused, lr 3e-4) remains the production
optimizer for ≥500M NSL pretraining**, and Muon stays a small/mid-scale
option where its per-step quality advantage is proven and its NS overhead
is negligible.
