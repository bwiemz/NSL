# The 1B gate: the run got worse, and I said the opposite the day before

**2026-09-01.** The lr=1.5e-5 intermediate chain reached its 1B-token gate
(micro 248,000 = 62,000 optimizer steps = 1,015,808,000 tokens). Sixteen
checkpoints across the run are scored on held-out. **The model at the gate is
worse than the model at 262M tokens by +1.269 nats (STACK) and +1.101 (WEB),
and the degradation ACCELERATES over the final leg.**

The day before, this file's predecessor recorded the opposite conclusion —
"an excursion resolving, not divergence". That was wrong, and the two errors
behind it are the transferable part.

## The trajectory

Scored under TF32 by `run-out/score_trajectory.sh`, which rewrites
`models/coder1b/val_from_splice.nsl`'s `model_load` path per point; every point
is a spliced `.nslm` snapshot of theta taken at a cadence write. That scorer
reads `data/tokens/pilot_val_{stack,web}_2m.bin`, which are byte-identical
prefixes of the V2 mixture's `mix/{stack,web}_val.bin` (md5 `4e8ddd73…` /
`07764ae3…`, verified — PRECISION_1B_THREE_ARM_2026_08_31.md).
`val_from_splice_v2.nsl` reads the V2 slices under their own names and
reproduces the same numbers to the last digit; either tool gives this table.

| micro | tokens | VAL_STACK | VAL_WEB | |
|---|---|---|---|---|
| 8,000 | 33M | 8.132 | 8.430 | |
| 64,000 | 262M | **3.872** | **5.834** | **best model** |
| 88,000 | 360M | 4.304 | 6.031 | |
| 96,000 | 393M | 4.353 | 6.133 | |
| 160,000 | 655M | 4.569 | 6.291 | three-arm probe base |
| 168,000 | 688M | 4.547 | 6.400 | |
| 176,000 | 720M | 4.487 | 6.279 | |
| 184,000 | 753M | 4.635 | 6.285 | |
| 192,000 | 786M | 4.714 | 6.409 | |
| 200,000 | 819M | 4.594 | 6.329 | |
| 208,000 | 851M | 4.717 | 6.563 | |
| 216,000 | 884M | 4.586 | 6.332 | |
| 224,000 | 917M | 4.959 | 6.530 | |
| 232,000 | 950M | 5.178 | 6.716 | |
| 240,000 | 983M | 5.074 | 6.730 | |
| 248,000 | 1,016M | 5.141 | 6.935 | **1B gate** |

Least-squares slopes, nats per 100k micro-steps, positive = worse:

| window | n | STACK | WEB |
|---|---|---|---|
| 655M – 884M | 8 | +0.204 ± 0.143 | +0.205 ± 0.181 |
| 884M – 1,016M | 5 | +1.531 ± 0.642 | +1.759 ± 0.226 |
| **655M – 1,016M** | **12** | **+0.753 ± 0.137 (5.5σ)** | **+0.627 ± 0.126 (5.0σ)** |

**The degradation itself is unambiguous** — 5σ on both held-out sets over the
whole leg, and the gate-vs-best gap is 8.9x (STACK) / 7.9x (WEB) the mean
adjacent-point swing of 0.14 nats. It is not noise.

**The 7-8x acceleration in the final leg is solid on WEB and marginal on
STACK** — 5.4σ and 2.0σ respectively for the difference between the two
window slopes. Report it as "WEB says the degradation is accelerating, STACK
is consistent with it"; do not report a single accelerating trend as if both
sets carried it.

## Why the "excursion resolving" call was wrong

Two mistakes, and neither needed new data to catch.

**1. Decelerating rates read off WIDENING intervals.** The rates that looked
like deceleration — +0.43, +0.15, +0.08 per 98M tokens — were computed over
intervals of 98M, 33M and **262M** tokens. Averaging a rate over a longer
window mechanically flattens it. There was no deceleration in the data; there
was a spacing artifact in the arithmetic. **A sequence of rates is only a
trend if the intervals are equal — otherwise fit a slope.**

**2. A probe ARM's point was read as the CHAIN's.** The "−0.21 recovery" that
made STACK look like it had turned back down was the bf16+RNE arm of the
three-arm probe at micro 184,000 (4.363). The chain's own point at that same
micro-step is **4.635**. The arm was a branch off micro 160,000, not a
continuation of the run. **Points from a branch and points from the trunk do
not belong in one trajectory table**, and the previous version of that table
put them there with only a parenthetical to distinguish them.

Both errors pushed the same way, which is the shape of motivated reading: the
conclusion they supported was the one that let the campaign skip a 4.4-hour
learning-rate arm.

## The chain/arm divergence at micro 184,000

Those two numbers should have been close and were not.

| | VAL_STACK @ micro 184,000 |
|---|---|
| three-arm probe, bf16+RNE branch | 4.363 |
| chain trunk | 4.635 |
| difference | **+0.272 (1.9x the mean adjacent swing)** |

Both are bf16+RNE, both resumed from the micro-160,000 checkpoint, both under
`DROPOUT = 0.0`, both scored by the same tool under TF32. The checkpoint
restores the loader slot and every RNG stream, so the batch sequence should
match.

The one known difference: the chain crashed with a GPU OOM at micro 168,600
(a concurrent session's unguarded job took 14.4 GB; see PR #547) and resumed
from the micro-168,000 checkpoint, so it saw micro 168,000-168,600 **twice** —
about 2.5% of the interval as duplicated data.

That is not established as the cause, and it is not dismissed either. If a
2.5% replay really moves held-out loss by 0.272 nats, this regime is unstable
enough that **any** A/B at these sizes needs its arms checked for interruption
history, not just for flags. Anyone designing one should look here first.

## What this does NOT overturn

The three-arm precision result stands: bf16+RNE beat bf16+SR beat f32, the
same order on both held-out sets, from ONE checkpoint on identical batches
(PRECISION_1B_THREE_ARM_2026_08_31.md). That measurement compares arms
**against each other** under a shared perturbation and never depended on the
trajectory's shape. bf16 stays in the recipe; SR stays out.

The distinction is the useful one: a matched pair survived an environment that
defeated a trend read. Differential designs are robust to exactly the
instability that makes absolute trajectories hard to read here.

## The standing suspect: the learning rate never decayed

`scheduler: warmup_cosine(warmup_steps=8192, total_steps=5516582,
min_lr=0.0000045)` is sized for the full 22.6B-token production epoch. At the
1B gate the run is **4.35% into the cosine**:

| micro | tokens | lr | % of peak |
|---|---|---|---|
| 64,000 | 262M | 1.49973e-5 | 99.98% |
| 160,000 | 655M | 1.49803e-5 | 99.87% |
| 248,000 | 1,016M | 1.49510e-5 | 99.67% |

The model trained a billion tokens at what is, to three digits, a **held peak
learning rate**. Held learning rates have now failed twice in this campaign:
the intermediate chain's root cause was held lr=3e-5, and the 1e-4/3e-4 arms
at item 10 also failed to descend. `1.5e-5` is a fifth of 3e-5 and bought
230M tokens of descent before turning — consistent with "still too high,
just less so", though that is a hypothesis and not a result.

The experiment it implies is a matched pair from the micro-64,000 checkpoint:
held 1.5e-5 versus a cosine actually sized for the 1B horizon, identical
batches, both arms scored on held-out at the same cadence. That is the
measurement this run earned and did not make.

## Decision at the gate

The staged plan asked: *"At 1B, decide whether the extra 0.5-1B tokens would
actually answer another question; continue to 1.5-2B only if they will."*

**Answer: no.** More tokens on this schedule make the model monotonically
worse in expectation. Extending to 2B would buy another ~10 hours of GPU to
watch a slope that is already 5σ. **The run is stopped at the gate.**

State on disk under `models/coder1b/checkpoints_backup/` (141 GB total; a 1B
checkpoint pair is 4.30 GB of theta + 8.59 GB of `.optim` sidecar):

| what | where |
|---|---|
| **best theta (micro 64,000)** | `lr15_stage2_opt16000/` |
| gate state (micro 248,000) | `lr15_1bgate_bf16/` |
| dense trajectory, micro 168,000-248,000 | `traj/m*.nslm` (11 snapshots, theta only, 45 GB) |
| earlier stage points | `lr15_stage1_opt2000/`, `lr15_stage4_opt{24000,40000}/`, `lr15_pre_reboot_opt22000/` |

The first five trajectory rows come from those stage backups, not from
`traj/` — `traj_snapshot.sh` only started running at micro 168,000. Anyone
re-scoring the early leg has to go find them.

## Reproduce

    run-out/stage4_driver.sh bf16        # the chain leg
    run-out/traj_snapshot.sh             # theta at every cadence write
    run-out/score_trajectory.sh splice   # CPU, runs beside training
    run-out/score_trajectory.sh score    # GPU

`score_trajectory.sh` prints `FAILED` and names the stderr file for a point it
could not evaluate; an earlier draft printed `0.0000`, which is a plausible
loss and would have entered this table as a datum. Dry-run the renderer of any
results table before it runs on real checkpoints.
