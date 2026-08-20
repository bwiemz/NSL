# Coder-1B production recipe — validation (roadmap item 10)

What `models/coder1b/pretrain_prod.nsl` does on the machine it was written for:
an RTX PRO 4500 Blackwell (32 GB, 31.39 GiB usable), CUDA 13.x, source AD.

**Read this first, because the headline is a negative.** Item 9 measured a
learning rate at 500M and got a real result (0.57 nats of held-out loss for
halving it). Item 10 set out to repeat that at 1B and **did not get a result.**
Two full epochs, neither descending, an ordering that flips with the horizon,
one seed each, no null control. The recipe therefore ships `lr = 1e-4` as the
*width scaling* of item 9's number — `1.5e-4 × 1280/2048 = 9.4e-5` — explicitly
labelled a principled default, not a measurement. §EC3 is the whole story
including what an earlier draft of this record got wrong.

Everything else here — memory, schedule, resume, the per-flag table, three
diagnostics that came back negative — did resolve.

| | value |
|---|---|
| geometry | d_model 2048, 16 layers, 32 heads / 8 KV, d_ff 8192, vocab 49152, seq 2048 |
| parameters | ~1.07B |
| schedule | 1 epoch = 2048 micro-steps = 512 optimizer steps, 16,384 tokens/step |
| allocator peak | **22,073,520,640 B = 20.56 GiB** (byte-identical across arms) |
| driver peak | 25,186–26,734 MiB over a full epoch (varies ~1.5 GiB run to run — see §EC1) |
| held-out | 10.284 nats @ 3e-4, 10.525 @ 1.5e-4, against `ln(49152) = 10.803` |

---

## EC1 — it fits, and the stable number is the allocator's, not the driver's

Both full-epoch arms report an **identical** allocator peak:

```
PEAK_BYTES       22073520640     (20.56 GiB)
PEAK_WEIGHTS      4312096768     ( 4.02 GiB)
PEAK_ACTIVATIONS 17761423872     (16.54 GiB)
PEAK_OPTIM_M               0
PEAK_OPTIM_V               0
PEAK_M_PARTIAL             0
```

The three zeros are the witness that `--optim-state-offload` actually moved the
optimizer state off the device — not merely that the flag was accepted. With
f32 AdamW at 1.07B parameters those three surfaces are 4.00 GiB each.

**The driver-level peak is the noisy one.** Sampling `nvidia-smi` every 0.5 s
through both epochs:

| arm | ambient (min sample) | driver peak | net |
|---|---|---|---|
| 3e-4 | 1738 MiB | 25,186 MiB | 23,448 MiB |
| 1.5e-4 | 1652 MiB | 26,734 MiB | 25,082 MiB |

Same program, same allocator peak to the byte, **1548 MiB apart at the driver
level.** Size a card against the maximum (~26.1 GiB) and treat the allocator
number as the property of the program. Milestone B's rule applies here too: the
peak is at END-OF-FORWARD, not at a step boundary, so a step-boundary sample is
not a peak.

Units: the allocator's memstats say "MB" and its OOM dump says "GB", and both
divide by 1024. Everything above is GiB/MiB.

## EC2 — the schedule executes as derived

Both arms ran **2048 micro-steps** and scored **128 validation batches**, which
is the arithmetic in `config.nsl` executed rather than asserted:

```
floor(8,388,608 / (2048 * 2)) = 2048 micro-steps
floor(2048 / accum 4)         =  512 optimizer steps
2 * 2048 * 4                  = 16,384 tokens / optimizer step
512 / checkpoint_every 100    =    5 checkpoints
floor(524,288 / (2048 * 2))   =  128 validation batches
```

The last three lines are the 500M recipe's numbers exactly — deliberate, so the
two sizes share an effective batch, an optimizer-step count and a warmup
fraction, and are scored on identical held-out text.

## EC3 — the learning rate question did NOT resolve at 1B

Two full epochs, identical but for the LR:

| | armA `lr=3e-4` | armB `lr=1.5e-4` |
|---|---|---|
| first print | 10.278 | 11.528 |
| mean, first half | 9.166 | 9.162 |
| mean, last half | **9.408** | 9.101 |
| min | 8.217 @ print 40 | 8.228 @ print 76 |
| final 5 prints | 8.6 – 9.3 | **10.03 – 11.21** |
| **VAL_LOSS** | **10.284** | **10.525** |

**Neither arm descends.** armA's last half is *worse* than its first half;
armB's is flat. Both land just under the `ln(49152) = 10.803` uniform-predictor
bound — a model that has learned roughly the unigram distribution and little
else. armB additionally destabilizes over its final ~25 optimizer steps, while
the cosine schedule is at `min_lr`, which is what makes its held-out number the
worse of the two.

That is not "1.5e-4 is worse than 3e-4 at 1B" — and §EC4 makes that sharper
than "uncontrolled". **armB's late destabilization did not reproduce**: resumed
from its own step-1200 checkpoint, with the same weights, the same optimizer
state and the same data order, the rerun never entered the excursion (tail
9.38/9.34/8.72/9.05/9.13 against armB's 11.18/10.03/10.35/11.21/10.67). So the
number that makes 1.5e-4 look worse is partly a one-off chaotic event, and the
comparison rests on a value its own rerun contradicts.

It is also one seed per arm, no null
control, and an ordering that **flips with the horizon**: over the first 400
micro-steps the means order the other way (9.285 at 3e-4, 9.218 at 1.5e-4,
8.947 at 1e-4 — lower LR looks better), and the held-out numbers reverse it. An
effect whose sign depends on where you stop measuring is not an effect yet.

The shipped `lr = 1e-4` is therefore `1.5e-4 × 1280/2048 = 9.4e-5`, the width
scaling of the one LR result this repo owns. A principled default. Settling it
needs a corpus, i.e. roadmap item 15 — see §EC5 for why no LR can be resolved
on this budget.

### What an earlier draft of this record got wrong

It claimed "three LRs spanning 3× produce one curve (pearson +0.963/+0.964/
+0.969, mean |Δ| 0.13–0.24 nats)" and read that as *the model is barely
updating*. **Those numbers do not reproduce.** They were computed over an early
~10–20-print window while the runs were still going:

| window (prints) | A~B | A~C | B~C |
|---|---|---|---|
| 10 | +0.776 | +0.807 | +0.988 |
| 20 | +0.806 | +0.794 | +0.979 |
| 36 | +0.374 | +0.569 | +0.717 |
| **102 (full epoch)** | **+0.217** | — | — |

Over the finished epochs the correlation is +0.217, and the "one curve" reading
was never supported. This repo already carries that lesson from a prior campaign
(a banked r=0.915 over 5 of 8 regions whose true value was 0.220 — landing, as
it happens, within 0.003 of the number here). It was re-committed inside the
very campaign that recorded it. **No derived statistic gets banked until the run
feeding it has exited.**

The conclusion did not rest on the correlation and is unchanged.

## EC4 — mid-epoch resume across host-resident optimizer state

The first test anywhere of checkpoint/resume × `--optim-state-offload`, where
`m` and `v` live on the **host**. Resumed from armB's step-1200 checkpoint, so
**848 of the epoch's 2048 micro-steps (41%) are re-run** — item 9's resume
covered 2% and correctly refused to call a 0.006-nat agreement evidence.

Mechanically it is exact:

```
[checkpoint] resumed: ... at micro-batch step 1200 (146 params, epoch 0 loader slot 1200)
[checkpoint] saved:   ... at micro-batch step 1600 (146 params, epoch 0 loader slot 1600)
[checkpoint] saved:   ... at micro-batch step 2000 (146 params, epoch 0 loader slot 2000)
```

- **848 micro-steps executed**, not 2048 — `epochs` is the run TOTAL under
  resume, so the run finishes the epoch instead of restarting it.
- The restored position is the loader's **delivery slot**, not a batch count.
- The witness block is byte-identical to armB's (`PEAK_BYTES 22073520640`,
  optimizer surfaces `0/0/0`).
- **The first two prints after restore are bit-identical to armB's** (8.756,
  9.275, |Δ| = 0.000). Theta, `m`, `v` and the loader slot all come back
  exactly, host-resident moments included.

**But the held-out numbers differ by 0.673 nats** (11.198 resumed vs 10.525),
and that is worth being precise about rather than filing as noise:

| micro-step | armB | resumed | \|Δ\| |
|---|---|---|---|
| 1200 | 8.756 | 8.756 | **0.000** |
| 1220 | 9.275 | 9.275 | **0.000** |
| 1400 | 9.214 | 9.276 | 0.062 |
| 1460 | 8.453 | 8.511 | 0.057 |
| 1940 | 11.181 | 9.379 | **1.802** |
| 2000 | 11.211 | 9.053 | **2.158** |
| 2020 | 10.665 | 9.128 | **1.537** |

Mean |Δ| over the first five prints is **0.009**; over the last five, **1.561**.
The two runs track for ~260 micro-steps and then separate — and the separation
runs the *wrong way for a restore bug*: **armB destabilizes and the resumed run
does not.** armB's tail is 11.18 / 10.03 / 10.35 / 11.21 / 10.67 while the
resumed tail is 9.38 / 9.34 / 8.72 / 9.05 / 9.13.

So the divergence is chaotic amplification downstream of a **real training
instability**, not a defect in restoration — which the bit-identical restart
independently establishes. A float-noise difference decides whether the run
enters the excursion.

**This feeds back into §EC3.** armB's late destabilization is *not
reproducible*: given the same weights, the same optimizer state and the same
data order, the second run did not repeat it. So armB's VAL of 10.525 is partly
an artifact of a one-off excursion, and the arm comparison is weaker still than
§EC3 already says — not merely uncontrolled, but resting on a number that its
own rerun did not reproduce.

Note also that the resumed run's held-out loss, **11.198, is *above* the
`ln(49152) = 10.803` uniform bound** — on this budget the model is not reliably
better than a uniform predictor on unseen text. §EC5.

### The control: resume on a STABLE arm reproduces the held-out loss

A bit-identical restart proves *restoration*. It does not prove the practically
useful claim — that a resumed run ends up at the same model. armB could not
answer that, because its own tail was chaotic. So the same test was run against
**armA**, whose tail is smooth (8.6–9.3, no excursion):

| arm | tail | original VAL | resumed VAL | Δ |
|---|---|---|---|---|
| **armA** (stable) | 8.6–9.3 | 10.284266 | **10.278813** | **0.0055** |
| armB (unstable) | 11.18–10.67 | 10.525391 | 11.198396 | 0.673 |

**0.0055 nats over 41% of an epoch re-run**, against a run-to-run noise floor of
0.0356 measured in §EC7. That is the end-to-end claim: on a stable trajectory,
mid-epoch resume across host-resident optimizer state reproduces the final
held-out loss. The armB gap is the excursion, not the mechanism — which is what
the bit-identical restart implied and what this control confirms independently.

Both resumes executed exactly 848 micro-steps.

## EC5 — the token budget, stated rather than buried

8,388,608 training tokens against ~1.07B parameters is **0.008 tokens per
parameter**. Against a Chinchilla-ish ~20 tokens/param that is a factor of
~2,550 short, and it is **half** the ratio the 500M recipe gets from the same
corpus — on a fixed corpus the larger model sees *fewer* tokens per parameter,
so these numbers are not evidence that 1B beats 500M here.

This is why §EC3 could not resolve: **0.008 tokens/param is not an experiment.**
No learning rate rescues a budget that short, and a run that never leaves the
neighbourhood of the unigram bound cannot rank two learning rates. The recipe
validates the *workflow* at 1B — real corpus, real schedule, checkpoint/resume,
held-out scoring — and does not produce a well-trained 1B model. It is not
tooling that is missing; it is corpus (roadmap item 15).

## EC6 — per-flag necessity, measured

The header recommends four flags. Only some are load-bearing. Each probe drops
one flag, runs the recipe **unmodified** to 40 micro-steps (10 optimizer steps,
past the end-of-forward peak and the first offload round-trip), on a clean GPU
with ambient recorded per probe:

| dropped | outcome | driver peak | net of ambient |
|---|---|---|---|
| *(nothing — the recommended line)* | **survives** 40 steps | 25,054 MiB | 23,259 MiB |
| `--fuse-rmsnorm-backward` | **survives** 40 steps | 25,073 MiB | 23,256 MiB |
| `--checkpoint-blocks` | **OOM at step 0** | 31,935 MiB | 30,065 MiB |
| `--optim-state-offload` | **unusable** (see below) | 31,868 MiB | 30,212 MiB |

`--fuse-rmsnorm-backward` is worth **3 MiB** — it is in the recommended line
because it is a free speedup at this scale, *not* because the run needs it.
`nsl build --help` calls it an opt-in speedup matching the decomposition to an
f32 tolerance. Drop it if you want the reference backward.

`--checkpoint-blocks` OOMs outright: `Requested 67108864 (64.0 MB)`,
`VRAM free: 252.2 MB / 31.39 GB`, allocation #1296, `sum_dim_f32`.

`--optim-state-offload` fails in **two stages, and not the same way twice.**
The run first *degrades* — `[nsl] GPU OOM in <op> — falling back to CPU`, which
does not stop it, it just makes 1B unusably slow (0 micro-steps in 900 s) — and
in an earlier probe went on to abort at a later allocation
(`VRAM free: 182.8 MB / 31.39 GB`, allocation #1199, `nsl_mul_f32`). Treat that
allocation number and op name as **one instance, not a signature**. What
reproduces is the cause, visible in the `[gpu-mem] surfaces:` line at step 0 in
both probes:

```
weights=4112MB  optim_m=4096MB  optim_v=4096MB  m_partial=4096MB
```

16.02 GiB of *persistent* state on the device before a single activation is
stored, against a 16.54 GiB activation peak. That is the whole story of why
this recipe offloads.

Not tested here: `--layerwise-accum`, which Milestone B's endurance benchmark
uses. It **refuses** `grad_clip` ("two-phase clipping needs the GLOBAL L2 norm
over every parameter's completed `m_partial` before any update, which the
layerwise schedule never materializes"), and trading clipping away for memory
the offload already buys is the wrong trade for a production run.

## EC7 — three diagnostics that came back negative

The arms not descending (§EC3) looked like it could be a defect. This repo has
shipped exactly that failure before: item 5's materializing `contiguous`
recorded nothing, so GQA K/V trained with **zero** gradients while the loss
curve looked healthy — caught by checksums, not by curves, because AdamW is
scale-invariant per parameter. Three hypotheses, all refuted:

**D1 — is `--optim-state-offload` corrupting the update?** Tested at 500M,
which fits both with and without the flag, so the flag is the only difference.
**With a null control**, because this repo is *not* bit-reproducible run to run
and an uncalibrated agreement number means nothing:

| comparison | max |Δ| | mean |Δ| |
|---|---|---|
| **null control** — baseline vs baseline, same flags/seed | **0.0356** | 0.0098 |
| treatment — baseline vs `+--optim-state-offload` | **0.0161** | 0.0051 |

**The treatment difference is smaller than the run-to-run noise floor.** No
detectable effect. (This control aborted three times before it ran — every time
in `nsl_tensor_matmul` at 0 steps, because a previous run's orphaned child still
held ~24 GB. See "traps" below.)

**D2 — is `grad_clip=1.0` saturated, normalising the LR away?** Raised to a
non-binding `1000.0` at 1B, same code path. Does not unstick training; the
curve is as erratic as before. Not the cause.

**G1/G2 — are some parameters getting no gradient?** `--grad-integrity` over a
full tiny epoch at both sizes, run to natural exit:

| | 1B | 500M (positive control) |
|---|---|---|
| checks | 40 | 40 |
| expected / gradient params | 146 / 146 | 218 / 218 |
| finite | 146 | 218 |
| **nonzero** | **146** | **218** |
| missing | `[]` | `[]` |
| unjudged_checks | 0 | 0 |

Every trainable parameter gets a finite, nonzero gradient every step. The item-5
failure mode is not what is happening.

With offload correct, the clip not binding and every gradient live, what remains
is §EC5: the budget.

---

## Reproducing

```bash
python models/benchmarks/make_prod_split.py     # once — materializes both slices
cd models/coder1b
nsl run --source-ad --checkpoint-blocks --fuse-rmsnorm-backward \
        --optim-state-offload pretrain_prod.nsl
```

**69–72 minutes** end to end on the reference card (compile + 2048 micro-steps
+ 128 validation batches), measured from the 0.5 s VRAM sampler's own cadence
across the two arms. `checkpoints/` must exist — the
runtime does `File::create` and aborts rather than creating it, which is why a
`.gitkeep` is tracked. A checkpoint pair is 4.30 GB of theta plus an 8.59 GB
`.optim` sidecar, rewritten 5 times over the run: budget ~13 GB of disk churn.

The agreement between this file, `config.nsl` and the recipe is machine-checked
by `crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs`, which also asserts
that this record exists — a recipe citing a validation record is asserting the
measurement happened.

## Traps this campaign paid for

- **`nsl run` execs a child** at `/tmp/nsl_run_<pid>/<prog>`. Killing the `nsl`
  parent leaves the child holding ~22 GB, so the *next* run OOMs on a device
  that looks free in `ps` but not in `nvidia-smi`. Run under `setsid` and kill
  the process **group**. This killed four separate measurements.
- **A cooldown that warns but does not gate is not a guard.** The sweep printed
  `WARNING: GPU still at 24459 MiB` and ran anyway — that is how the null
  control died twice and how the 5e-5 / 2e-5 sweep arms were lost entirely.
- **`grep -c X f || echo 0` emits TWO lines** when there is no match (`grep -c`
  already prints 0 *and* exits 1), which is an arithmetic error inside `[[ ]]`.
- **`cargo test -p nsl-cli` rebuilds `nsl` without `--features cuda`**, so a GPU
  campaign must run against a frozen copy of the CUDA binary that no cargo
  invocation can reach. Preflight it with
  `nm -D <bin> | grep cuDevicePrimaryCtxRetain`.
- **A process-exit hook prints nothing if you kill the process.**
  `--grad-integrity` reports at exit; killing the run at step 40 produced
  silence, and silence looks exactly like "no problems found".
