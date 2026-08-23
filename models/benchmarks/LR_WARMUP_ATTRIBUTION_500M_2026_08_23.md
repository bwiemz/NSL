# 500M LR × warmup — the 2×2, run on a compiler where schedules work

**Why this exists.** Item 9 changed two things at once (`lr 3e-4 → 1.5e-4`
AND `warmup 200 → 400`) and banked 0.57 nats. This closes that confound.

**Why it had to be re-run.** The first attempt at this matrix (2026-08-22)
was worthless: `scheduler:` was inert for `grad_accumulation > 1` (PR #520),
so the warmup axis did nothing and the 2×2 collapsed to a 1×2 in lr with two
replicates per level. **Every 500M and 1B result recorded before PR #520 was
produced at CONSTANT lr, whatever its recipe said.**

Hardware RTX PRO 4500 Blackwell. `--source-ad --checkpoint-blocks --seed 4242`,
one epoch (4096 micro-steps, accum 8), `min_lr = 3e-5` held fixed across all
four arms — which is what both of item 9's original arms used, so this
decomposes that comparison rather than adding a third variable.

## The matrix

| arm | lr | warmup | VAL (256 batches) |
| --- | --- | --- | --- |
| A | 3e-4 | 200 | 8.873 |
| B | 1.5e-4 | 200 | 8.747 |
| C | 3e-4 | 400 | 9.163 |
| **D** | **1.5e-4** | **400** | **8.716** |

## Effects (positive = the second config is better)

| contrast | Δ nats |
| --- | --- |
| lr 3e-4 → 1.5e-4, warmup 200 (A→B) | **+0.126** |
| lr 3e-4 → 1.5e-4, warmup 400 (C→D) | **+0.447** |
| warmup 200 → 400, lr 3e-4 (A→C) | **−0.290** |
| warmup 200 → 400, lr 1.5e-4 (B→D) | +0.031 |
| item 9's actual recipe move (A→D) | +0.157 |

## The noise floor, derived rather than assumed

The superseded sweep accidentally supplies it. Its arms B and D differed ONLY
in `warmup_steps` — which was inert — so they are **two replicates of the same
config** (1.5e-4, constant lr): **8.699 vs 8.672, spread 0.027 nats**. Its
arms A and C are likewise replicates of (3e-4, constant lr), and they came
back **10.114 and NaN**: one of two diverged.

So: ~0.03 nats at the low lr, and **unbounded at the high lr**. That asymmetry
governs every reading below.

## What this establishes

1. **The learning-rate finding SURVIVES.** 1.5e-4 beats 3e-4 at *both*
   warmups, same direction, +0.126 and +0.447 — 4.7× and 16× the low-lr
   floor. Item 9's central claim holds.

2. **Warmup 200 → 400 is not a lever at the shipped lr.** +0.031 nats sits
   exactly ON the noise floor. It is not resolvable, and it should never have
   been described as measured.

3. **Doubling warmup HURT at the high lr** (−0.290). Caveat this one: both
   cells are high-lr, where a replicate diverged outright, so the variance
   there is demonstrably large and unquantified. Direction only; do not quote
   the magnitude.

4. **Item 9's recipe move is worth +0.157 nats, not 0.57.** Its original
   comparison was between two runs that both trained at constant lr, so what
   it actually measured was the lr alone, in a regime where the high-lr arm
   had no warmup protecting it.

5. **What the broken scheduler was hiding is the biggest effect here.**
   Against the old constant-lr arms, giving 3e-4 a working warmup moves VAL
   from 10.114 (and one NaN) to 8.873 — **~1.24 nats**, larger than any
   contrast inside the matrix. Warmup matters enormously *versus none*; the
   200-vs-400 distinction does not.

## Recipe consequence

**`config.nsl` is unchanged.** (1.5e-4, 400) is still the best of the four,
so the shipped numbers stand — but for a partly different reason than
recorded, and its margin over (1.5e-4, 200) is 0.031 nats, i.e. none that this
experiment can see. The provenance comment is corrected in the same commit:
the lr is measured, the warmup is a default.

## Limits

- **n = 1 per cell.** The floor above comes from a different (constant-lr)
  configuration and is imported, not measured in this matrix.
- 8.39M train tokens against ~505M params is ~0.017 tokens/param. These are
  workflow measurements on an under-trained model, not a converged comparison.
- Nothing here is measured at 1B. Item 10's LR question remains open and its
  runs were constant-lr, so it needs re-running before any conclusion.
