# Coder-500M production recipe — validation (roadmap item 9)

Machine: RTX PRO 4500 Blackwell, 32 GB, CUDA 13.x. `nsl` built `--release
--features cuda` at `2193b0cb` + this branch. Ambient desktop VRAM 1.4–1.6 GB
at launch (recorded because it has invalidated measurements before).

Program: `models/coder500m/pretrain_prod.nsl`.
Flags: `--source-ad --checkpoint-blocks`.
Corpus: `data/tokens/prod_train_slice.bin` (8,388,608 u16 tokens), held-out
`prod_val_slice.bin` (524,288), 13,020 tokens read by neither.

## EC1 — it fits, but only with activation recomputation

| | without `--checkpoint-blocks` | with |
|---|---|---|
| outcome | **OOM in the first backward** | runs to completion |
| failure | `VRAM free: 168.2 MB / 31.39 GB` at allocation #3819 (`matmul_f32`) | — |
| weights / optim m / optim v / m_partial | 2.07 / 2.06 / 2.06 / 2.06 GB | same |
| activations peak | exhausts the card | **8.34 GB** |
| driver-reported, step 5 | — | **10.9 GB** |

The persistent state is only 8.25 GB; it is the stored activations of 24
layers at batch 2 × seq 1024 that do not fit. The recipe header documents
`--checkpoint-blocks` as required on the strength of this measurement, the
same posture the 1B recipe takes toward its own flag set.

## EC2 — the schedule executes as derived

4096 micro-steps in one epoch; 5 checkpoints at micro-batch 800/1600/2400/
3200/4000 — exactly `floor(4096/8) / 100` optimizer-step boundaries. Each
save records the loader position, e.g.

    [checkpoint] saved: checkpoints/pretrain_prod_state.nslm (+.optim)
        at micro-batch step 4000 (218 params, epoch 0 loader slot 4000)

which is item 8 (PR #516) working at 500M, not just in the toy gates.
Throughput ≈ 2.25 micro-steps/s, ≈ 30 min/epoch. The held-out pass runs 256
batches forward-only without additional memory pressure.

## EC3 — THE FINDING: sequential reading made the loss unreadable

The first arm used `shuffle=false` (inherited from the 50M and 1B recipes).
Its training loss **rose** from 8.38 to 9.45 between micro-steps 1600 and
2400 and finished the epoch above where it started:

| region (micro-steps) | mean loss | corpus unigram entropy |
|---|---|---|
| 0–512 | 8.826 | 8.019 |
| 512–1024 | 8.434 | 8.013 |
| 1024–1536 | 8.694 | 7.990 |
| 1536–2048 | 9.176 | **8.406** |
| 2048–2560 | 9.306 | **8.569** |
| 2560–3072 | 9.010 | 7.942 |

That is not divergence. The corpus is a CONCATENATION — this repo's train
split, then 15 unrelated local projects — so reading it in order walks
through regions of different difficulty. Per-region unigram entropy against
per-region training loss: **Pearson r = 0.915, slope 1.21 nats of loss per nat
of entropy**. The decisive check is the 2560–3072 region: entropy drops back
to 7.94 and the loss follows it **down**, which a diverging optimizer does not
do.

A loss stream that measures where you are in the corpus rather than how well
you are learning is useless as the monitoring signal a production recipe
exists to emit — and it is actively misleading, since it reads as a failing
run.

**Fix: `shuffle=true`.** This is only affordable because of item 8: the
single-rank shuffle was seeded from ENTROPY until PR #516, so shuffling would
have made every run's data order irreproducible and every checkpoint's
recorded position meaningless. It is now keyed by (seed, epoch).

Both arms, per-region training loss over the same corpus positions:

| region | A `shuffle=false` | B `shuffle=true` |
|---|---|---|
| 0–512 | 8.826 | 9.093 |
| 512–1024 | 8.434 | 8.987 |
| 1024–1536 | 8.694 | 9.208 |
| 1536–2048 | 9.176 | 8.801 |
| 2048–2560 | 9.306 | 8.797 |
| 2560–3072 | 9.667 | 8.976 |
| 3072–3584 | **10.944** | 8.909 |
| 3584–4096 | 9.354 | 8.868 |

Arm A swings across 2.5 nats and spikes to 10.94; arm B stays inside 9.09–8.80
and its late-epoch band sits below every late arm-A region.

### …but shuffling did not improve the MODEL

| arm | VAL_LOSS (256 held-out batches) |
|---|---|
| A `shuffle=false` | **9.765** |
| B `shuffle=true` | **9.793** |

0.028 nats apart, with no replicate scatter measured — under the item-6
methodology that is not a difference, and the sign is against the shuffled arm
anyway. **The recipe shuffles for MONITORING interpretability, not for
quality.** Saying otherwise would be the same single-run-delta error the
SR-BF16 campaign was built to avoid. What shuffling buys is a loss stream an
operator can actually read: no false-divergence artifact, no 10.94 excursion,
and a curve whose movement means learning rather than location.

## EC5 — resume works at 500M, on a shuffled loader

Re-running the recipe UNCHANGED with `checkpoint_load=` added, from arm B's
step-4000 checkpoint:

    [checkpoint] resumed: checkpoints/pretrain_prod_state.nslm (+.optim)
        at micro-batch step 4000 (218 params, epoch 0 loader slot 4000)

It continued at micro-batches 4020/4040/4060/4080 — the remaining 96 of the
epoch, not a restart — and finished at **VAL_LOSS 9.799 against the
uninterrupted arm B's 9.793**, 0.006 nats apart. Bit-identity is not asserted
at this scale (TF32 GEMMs, fused-CE backward atomics; the same doctrine as the
1B endurance harness), so agreement to 0.006 nats on a held-out set is the
available evidence that the resumed run is the same run.

This is the case item 8's toy gates cannot reach: a shuffled loader, where
resuming at the wrong slot would silently retrain a different 96 batches.

## EC4 — the token budget, stated rather than buried

8.39M train tokens against ~505M parameters is ~0.017 tokens/param, roughly
two orders of magnitude below compute-optimal. Arm A's held-out loss (9.765)
sits above its own late-epoch training loss, which is what an under-trained
model on a heterogeneous corpus should look like. **This recipe validates the
workflow at 500M; it does not produce a good 500M model.** Closing that needs
a corpus, which is roadmap item 15 — and the held-out split exists precisely
so the claim is measured rather than assumed.

## Reproducing

```bash
python models/benchmarks/make_prod_split.py     # from the repo root
cd models/coder500m
nsl run --source-ad --checkpoint-blocks pretrain_prod.nsl
```

`crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs` pins the config↔recipe
agreement, the schedule derivation, the non-overlapping split, and the
shuffle choice for both 50M and 500M.
