# An interruption does not change the requested trajectory

**2026-09-02.** The 1B gate left one observation unexplained: a chain checkpoint
and a probe arm, at the same micro-step under nominally the same arithmetic,
differed by 0.272 nats — 1.9x the local adjacent-point swing. The only known
difference was that the chain replayed ~600 micro-steps after an OOM crash.
PR #548 left causality open. This experiment closes it.

Coder-50M, `shuffle=true`, one checkpoint, arms branched and rejoined:

    UNINT1, UNINT2   two uninterrupted runs           (THE CONTROL)
    RESUME           run to the mid-run cadence write,
                     kill, re-run with checkpoint_load= added

## Result 1 — under `--deterministic`, resume is EXACT

| | UNINT1 vs UNINT2 | UNINT1 vs RESUME |
|---|---|---|
| requested-trajectory fields | 12/12 exact | **12/12 exact** |
| loss stream (24 steps) | bit-identical | bit-identical when stitched |
| final theta sha256 | MATCH | **MATCH** |
| optimizer moments sha256 | MATCH | **MATCH** |

The interruption landed one step past the cadence, so **step 13 was computed
twice, in two different processes**, and produced the identical loss to the
last digit:

    step 13   phase1  10.925692558288574
              phase2  10.925692558288574

`phase1[1..12] + phase2[13..24]` is bit-identical to the uninterrupted run.

**Answer to the question as posed: an intentional interruption does not change
which mathematical trajectory is requested.** All twelve integer/string/seed
fields agree — `loader_slot`, `loader_id`, `loader_epoch`, `rng_seed`,
`rng_pos_hi/lo`, `gpu_dropout_ctr`, `train_epoch`, `step_count`, the execution
and train-config fingerprints, and the scheduler LR they imply. Under
determinism it does not change the numerical outcome either.

## Result 2 — under `det=0`, divergence needs NO interruption

The 1B chain ran `det=0` (read from its own exec fingerprint). Repeating the
experiment without `--deterministic`, 48 micro-steps:

| comparison | steps differing | max loss spread | first difference |
|---|---|---|---|
| **AMBIENT** — unint1 vs unint2 | 43/48 | 1.61e-04 | step 5 |
| **INTERRUPT** — unint1 vs resume | 43/48 | 1.08e-04 | step 5 |

Two *identical, uninterrupted* runs with the same seed already diverge by step
5 and end with different theta. **The interrupted arm diverges LESS than the
ambient baseline (0.67x)** — an interruption adds nothing measurable on top of
GPU reduction nondeterminism, and all 10 comparable resume-state fields remain
identical.

## What this means for the 0.272 nats

**The replay is exonerated.** Resume state is logically exact, and under the
`det=0` conditions the 1B chain actually ran, the chain and the probe arm would
have diverged *even if the OOM had never happened*. Nondeterminism seeds a
difference by step 5; it does not need a replay.

Combined with SCHEDULE_3ARM_262M_2026_09_02.md — a held LR makes the trajectory
4-5x noisier point-to-point than a decayed one — the anomaly has a mechanism:
a small ambient perturbation, amplified by a chaotic regime.

**Not claimed:** that 1.6e-04 over 48 steps at 50M *becomes* 0.272 nats over
24,000 steps at 1B. That is a 1000x extrapolation across two scales and is not
established here. What is established is the direction and the cause: resume is
not the mechanism, and nondeterminism plus chaos is sufficient to produce
divergence without one.

**Operationally:** an A/B whose arms must be comparable at this scale needs
`--deterministic`, or it is comparing two draws from a distribution. That cost
is real — the deterministic path is ~20x slower here, consistent with the ~36x
slope recorded in `det_train_check.nsl`.

## Method notes

- **`shuffle=true` on purpose.** `det_train_check.nsl` uses `shuffle=false`,
  where the loader slot alone fixes the batch and the RNG-restore path is never
  exercised. The 1B loader shuffles.
- **Distinct windows on purpose.** `det_train_check.nsl` builds its stream by
  `expand()`ing one row, so every window is identical and a shuffle is
  unobservable; `arange` gives windows that differ everywhere, so a wrong batch
  order changes the loss.
- **No direct token checksum.** `.sum().item()` on i32 ids hits `data_f64()`'s
  dtype assert and `astype` is not an implemented method, so batch identity is
  established from the loader state that GENERATES every future batch
  (`loader_slot` + `loader_id` + the three RNG fields) plus a bit-identical
  loss stream, rather than from a hash of one batch. Stated because it is a
  substitution, not the thing originally intended.
- Loss values are compared **as strings**, so "bit-identical" means that.

## Two failures worth recording

**A vacuous pass, caught.** The first comparison printed
`loss stream: 0 vs 0 steps -- BIT-IDENTICAL`. The parser expected a bare float;
NSL prints `tensor([10.906...])`, so it matched nothing and then declared two
EMPTY lists identical. The tool now REFUSES when either stream parses to zero
entries. See the both-sides-empty trap in PR #519.

**Three driver bugs, all the same category: logic written for a 5.5 h arm
applied to a 4 s run.** A 1-second poll on a 4-second run; then watching a
block-buffered log that stays empty until exit; then waiting up to 10 s for the
checkpoint mtime to settle BEFORE killing, during which the run simply
finished. The fix inverts the order — poll the checkpoint (a filesystem event
cannot be buffered away), kill immediately, then VERIFY integrity by header
arithmetic and retry if the kill tore the write.

Each of those three ended on #516's resume guard refusing to train zero steps
(*"is at epoch 1 but this train block declares epochs = 1"*). Without it, phase
2 would have exited 0 having trained nothing, and the comparison would have
found two completed runs equivalent — the exact false pass this experiment
exists to rule out.

## Reproduce

    campaign-resumeq/run_equiv.sh                    # deterministic, 24 steps
    SUB=nd48 EXTRA_FLAGS="" campaign-resumeq/run_equiv48.sh   # det=0, 48 steps
    campaign-resumeq/compare_equiv.py
