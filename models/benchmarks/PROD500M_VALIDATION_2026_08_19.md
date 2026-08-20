# Coder-500M production recipe — validation (roadmap item 9)

Machine: RTX PRO 4500 Blackwell, 32 GB, CUDA 13.x. `nsl` built `--release
--features cuda` at `2193b0cb` + this branch. Ambient desktop VRAM 1.4–1.6 GB
at launch (recorded because it has invalidated measurements before).

Program: `models/coder500m/pretrain_prod.nsl`.
Flags: `--source-ad --checkpoint-blocks`.
Corpus: `data/tokens/prod_train_slice.bin` (8,388,608 u16 tokens), held-out
`prod_val_slice.bin` (524,288), 13,020 tokens read by neither. The slices are
separate files, so they are index-disjoint by construction; the gap exists to
keep the held-out set from beginning in the immediate continuation of the last
file trained on, which in a concatenated source corpus would flatter it.

> **Correction notice.** The first version of this record claimed the
> `shuffle=false` training loss tracked corpus difficulty at Pearson
> r = 0.915. That statistic was computed **mid-run, over the first 5 of the 8
> regions**, and then written up as if it described the epoch. Over all 8
> regions it is **r = +0.220**, and the conclusion it supported does not hold.
> Review caught it; §EC3 is the corrected analysis, and a third arm resolved
> the excursions to the learning rate instead. §EC1/EC2 were measured after
> the runs completed and are unaffected.

## EC1 — it fits, but only with activation recomputation

| | without `--checkpoint-blocks` | with |
|---|---|---|
| outcome | **OOM in the first backward** | runs to completion |
| failure | `VRAM free: 168.2 MB / 31.39 GB` at allocation #3819 (`matmul_f32`) | — |
| weights / optim m / optim v / m_partial | 2.07 / 2.06 / 2.06 / 2.06 GB | same |
| activations, allocator's *tracked* peak | exhausts the card | **8.34 GB** |
| whole process, `nvidia-smi` mid-run | — | **≈19.7 GiB** (incl. ≈1.4 GiB ambient) |

The persistent state is 8.25 GB; it is the stored activations of 24 layers at
batch 2 × seq 1024 that do not fit without recomputation.

**Do not read the allocator's `driver=` line as the peak.** Its 11 samples max
at 10.9 GB because `NSL_MEMSTATS` prints at step boundaries, where the
activation surface is 0 — the trap this repo's notes record as "a step-boundary
sample is not a peak", which the first draft of this document fell into. The
load-bearing numbers are the allocator's *tracked* activation peak (8.34 GB,
concurrent with the 8.25 GB of persistent state) and the external `nvidia-smi`
observation.

## EC2 — the schedule executes as derived

4096 micro-steps in one epoch; 5 checkpoints at micro-batch 800/1600/2400/
3200/4000 — exactly the `checkpoint_every(100) × accum(8)` boundaries, and
`floor(512/100) = 5`. Each save records the loader position:

    [checkpoint] saved: checkpoints/pretrain_prod_state.nslm (+.optim)
        at micro-batch step 4000 (218 params, epoch 0 loader slot 4000)

Throughput ≈ 2.25 micro-steps/s, ≈ 30 min/epoch. The held-out pass runs 256
batches forward-only without additional memory pressure.

## EC3 — the inherited learning rate was too high at 500M

Three complete arms, per-region mean training loss. Region entropy is the
corpus's own per-region unigram entropy, included because the first draft of
this record built its conclusion on it:

| region | corpus H | A `shuffle=false` lr 3e-4 | B `shuffle=true` lr 3e-4 | C `shuffle=true` lr 1.5e-4 |
|---|---|---|---|---|
| 0–512 | 8.019 | 8.826 | 9.093 | 9.115 |
| 512–1024 | 8.013 | 8.434 | 8.987 | 8.608 |
| 1024–1536 | 7.990 | 8.694 | 9.208 | 8.272 |
| 1536–2048 | 8.406 | 9.176 | 8.801 | 7.944 |
| 2048–2560 | 8.570 | 9.306 | 8.797 | 7.915 |
| 2560–3072 | 7.942 | 9.667 | 8.910 | 7.906 |
| 3072–3584 | 8.206 | **10.944** | 8.743 | 7.352 |
| 3584–4096 | 8.285 | 9.354 | **12.172** | **7.442** |
| max sampled | | 13.820 | 19.369 | 10.773 |
| **VAL_LOSS** | | **9.765** | **9.793** | **9.197** |

Arm C halves the learning rate (3e-4 → 1.5e-4) and doubles the warmup
(200 → 400 micro-steps, i.e. 25 → 50 optimizer steps). It is the only arm that
descends monotonically, and it takes **0.57 nats off the held-out loss** —
twenty times the 0.028 that separates the two shuffle arms.

**The cause was the learning rate.** This repo carries `lr = 3e-4` unchanged
across d_model 512 / 1280 / 2048; at 500M, on a schedule of only 512 optimizer
steps, it is too high. The recipe now uses the measured values.

### What the first draft got wrong

It attributed arms A/B's excursions to corpus ordering, on a Pearson
r = 0.915 between per-region entropy and per-region loss. That number was
computed **mid-run over the first 5 of 8 regions**; over all 8 it is +0.220.
Two checks that were available at the time and were not run:

- the region carrying arm A's 10.944 excursion has *mid-range* entropy
  (8.206) — lower than two regions that scored far better. A mechanism that
  fails on the most extreme point it was invoked to explain is not the
  mechanism.
- arm B is a free null control: shuffled, it cannot have a corpus-position
  artifact by construction. r(H, arm B) = +0.095, and it produced the
  *largest* excursion of any arm (12.172). Running the statistic against it
  takes seconds and would have killed the story immediately.

`shuffle=true` is kept, but on general grounds — a concatenated corpus should
not be fed in file order, and item 8 made a seeded shuffle reproducible and
resumable — **not** as a measured improvement: at the inherited LR it was
9.765 vs 9.793, which is nothing.

## EC4 — resume runs at 500M on a shuffled loader

Re-running the recipe with `checkpoint_load=` added, from arm B's step-4000
checkpoint:

    [checkpoint] resumed: checkpoints/pretrain_prod_state.nslm (+.optim)
        at micro-batch step 4000 (218 params, epoch 0 loader slot 4000)

It continued at micro-batches 4020/4040/4060/4080 — the remaining 96 of the
epoch, not a restart — and reached VAL_LOSS 9.799 against the uninterrupted
9.793.

**That 0.006-nat agreement is weak evidence and is not claimed as more.** 96
micro-steps is 2% of the epoch, so a resume that restored θ and the optimizer
state but landed on the *wrong* 96 batches would also finish within a few
hundredths. What this run establishes is that the machinery executes at 500M
scale: the sidecar loads, the position and RNG restore, the step counter
continues, and the epoch terminates where it should. The discriminating
evidence for *correct* position restore is the bit-exact CPU gate in
`crates/nsl-cli/tests/train_resume_dataloader_gate.rs`, not this run.

## EC5 — the token budget, stated rather than buried

8.39M train tokens against ~505M parameters is ~0.017 tokens/param, roughly
a factor of ~1,200 below a Chinchilla-ish ~20 tokens/param — item 10
corrected this line, which read "roughly two orders of magnitude" and
understated the gap by more than 10x. The best arm ends at 9.197
held-out against ln(49152) = 10.80 for a uniform predictor — the model has
learned something, and not much. **This recipe validates the workflow at 500M; it does
not produce a good 500M model.** Closing that needs a corpus, which is roadmap
item 15 — the held-out split exists so the claim is measured rather than
assumed.

## Reproducing

```bash
python models/benchmarks/make_prod_split.py     # resolves the repo itself
cd models/coder500m
nsl run --source-ad --checkpoint-blocks pretrain_prod.nsl
```

`crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs` pins the config↔recipe
agreement, the schedule derivation, the non-overlapping split, and the shuffle
choice for both 50M and 500M.
