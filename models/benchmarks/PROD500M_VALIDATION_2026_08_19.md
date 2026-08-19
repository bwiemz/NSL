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
> Review caught it; §EC3 is the corrected analysis. §EC1/EC2 were measured
> after the runs completed and are unaffected.

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

## EC3 — late-epoch loss excursions, cause NOT established

Per-region training loss, both arms, against each corpus region's unigram
entropy (all eight regions, completed runs):

| region | corpus unigram H | A `shuffle=false` | B `shuffle=true` |
|---|---|---|---|
| 0–512 | 8.019 | 8.826 | 9.093 |
| 512–1024 | 8.013 | 8.434 | 8.987 |
| 1024–1536 | 7.990 | 8.694 | 9.208 |
| 1536–2048 | 8.406 | 9.176 | 8.801 |
| 2048–2560 | 8.570 | 9.306 | 8.797 |
| 2560–3072 | 7.942 | 9.667 | 8.910 |
| 3072–3584 | 8.206 | **10.944** | 8.743 |
| 3584–4096 | 8.285 | 9.354 | **12.172** |

**The corpus-ordering hypothesis does not survive the full sample.**
r(H, arm A) = **+0.220** over all eight regions — weak. It is +0.915 over the
first five, which is the number the first draft reported after computing it
mid-run: a subset fit presented as an epoch-level result. And the region
carrying arm A's 10.944 excursion has *mid-range* entropy (8.206), lower than
two regions that scored far better — so unigram entropy does not explain the
excursion it was invoked to explain.

**Shuffling does not remove the excursions.** Arm B reads the corpus in a
seeded random order and therefore cannot have a corpus-position artifact at
all; it produces its own, larger excursion (12.172) in the final region.
r(H, arm B) = +0.095. That arm is the free null control, and running it against
the same entropies is what makes arm A's +0.220 read as noise.

What is established: **both arms destabilize late in the epoch, and the cause
is not corpus ordering.** The leading untested hypothesis is the learning
rate — every model size in this repo carries `lr = 3e-4` regardless of width
(d_model 512 / 1280 / 2048), and at 500M the schedule gives only 25 optimizer
steps of warmup out of 512. An arm at half the LR with double the warmup is
recorded below.

Neither shuffle choice moves held-out quality:

| arm | VAL_LOSS (256 held-out batches) |
|---|---|
| A `shuffle=false` | **9.765** |
| B `shuffle=true` | **9.793** |

0.028 nats apart with no replicate scatter measured — under the item-6
methodology that is not a difference, and its sign is against shuffling. The
recipe keeps `shuffle=true` on the general grounds that a concatenated corpus
should not be fed in file order, and because item 8 made a seeded shuffle
reproducible and resumable — **not** because it was measured to help.

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
two orders of magnitude below compute-optimal. Both arms end near 9.8 held-out
against ln(49152) = 10.80 for a uniform predictor — the model has learned
something, and not much. **This recipe validates the workflow at 500M; it does
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
