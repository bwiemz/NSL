# bf16 vs f32 vs bf16+SR at 1B: a matched pair, and a retraction

**2026-08-31.** One checkpoint, identical batches, three arithmetics, held-out
scored on all three. The result reverses a conclusion this campaign had
previously banked, and the way it went wrong is more transferable than the
number itself.

## Result

Base: micro 160,000 of the lr=1.5e-5 chain (655M tokens, bf16-trained).
Each arm resumes that checkpoint and runs 24,000 micro = 98.3M tokens.
`DROPOUT = 0.0`, and the checkpoint restores the loader slot, so the arms are
deterministic apart from GEMM arithmetic. Held-out scored under TF32 for all
three, matching the rest of the trajectory record.

| arm | VAL_STACK | Δ base | VAL_WEB | Δ base |
|---|---|---|---|---|
| base @ micro 160,000 | 4.569 | — | 6.291 | — |
| **bf16 + RNE** | **4.363** | **−0.206** | **6.332** | +0.041 |
| bf16 + SR (`NSL_MATMUL_BF16_ROUND=sr`) | 4.477 | −0.092 | 6.385 | +0.094 |
| f32 (TF32 tensor cores) | 4.653 | +0.084 | 6.423 | +0.132 |

**Identical ranking on both held-out sets: bf16+RNE < bf16+SR < f32.**

Throughput on these same arms: bf16 **6,500 tok/s**, f32 **5,020 tok/s**
(+29.5% for bf16). bf16 wins on both axes simultaneously.

### 1. bf16 is acquitted, and the earlier conviction is REVERSED

The probe was designed so that if bf16 caused the chain's held-out rise, the
f32 arm would recover. It did the opposite: f32 is the WORST arm on both sets.
An earlier, badly-controlled read had convicted bf16 of "accumulating drift"
and the recipe carried that as fact. It was measured at held lr=3e-5, a regime
later shown to be diverging at any precision — the conviction was
LR-confounded. **bf16 stays in the production recipe.**

### 2. PR #540's stochastic rounding does not help convergence here

SR lands BETWEEN RNE and f32 on both sets while costing 5.9% throughput. Do
not enable it for the chain. This does not invalidate #540: its operand-level
gates are correct and the standing-bias mechanism it characterises is real.
It is simply not the cure at this learning rate and horizon, and "the mechanism
is real" was never the same claim as "the cure is needed".

### 3. The degradation is precision-INDEPENDENT

Every arm still worsens on WEB (+0.041 to +0.132 over 98M tokens). Whatever
moves the model survives all three arithmetics, so precision is not the
explanation for the trajectory — which is what sent the investigation to the
next section.

## RETRACTED 2026-09-01: the trajectory was NOT an excursion

This section previously concluded that the chain's held-out rise was "an
excursion resolving, not divergence". **That was wrong.** The chain ran on to
its 1B gate and the degradation accelerated; the gate model is +1.269 nats
(STACK) / +1.101 (WEB) worse than the 262M model, at 5σ over the leg. The full
sixteen-point trajectory, the fitted slopes and the two errors behind the bad
call are in **CHAIN_1B_GATE_2026_09_01.md**.

The errors in one line each, because they are cheap to repeat:

- **Rates were read off WIDENING intervals** (98M, 33M, then 262M tokens).
  Averaging a rate over a longer window mechanically flattens it. A sequence
  of rates is a trend only if the intervals are equal — otherwise fit a slope.
- **A probe ARM's point was tabled as the CHAIN's.** The "−0.21 recovery" at
  754M was this document's own bf16 arm (4.363), a branch off micro 160,000.
  The chain's own point at that micro-step is 4.635. Branch points and trunk
  points do not belong in one trajectory table.

**What is retracted is only the trajectory reading.** The three-arm result
above stands unchanged: it compares arms against each other from ONE
checkpoint on identical batches, and never depended on the trajectory's shape.
A matched pair survived the same data that defeated a trend read — which is
the argument for differential designs, stated more sharply than the original
draft managed.

The one change that section made which was worth keeping: the chain now
snapshots theta at EVERY cadence write (`run-out/traj_snapshot.sh`, .nslm
only, 4.3 GB each). That is why the gate could be read densely at all.

## Instruments, and why each was checked

Every measurement below was verified rather than trusted, and two of the checks
changed what could be claimed.

- **Held-out set provenance.** The inspection tool scores
  `pilot_val_{stack,web}_2m.bin` while the recipe scores `mix/{stack,web}_val.bin`
  — which looked like the whole campaign had measured against the wrong corpus.
  It had not: the pilot files are byte-identical prefixes of the V2 files
  (md5 `4e8ddd73…`, `07764ae3…`), and re-scoring against explicit V2 slices
  reproduced every number to the last digit. Per CORPUS_MANIFEST_v2.json,
  `mix/stack_val.bin` is one Stack shard off the training stride,
  repository-disjoint and verified by intersecting `repo_id`. The numbers
  measure real generalization.
- **Toolchain.** The trajectory record predates PR #541. Re-scoring the
  opt-16,000 splice with the current binary reproduced
  3.872422343150514 / 5.834139979276501 **bit-identically** — not a tooling
  artifact.
- **Scoring precision.** The first new score was taken under bf16 while the
  record was under TF32. Re-scored under TF32: 4.5687/6.2913 vs bf16's
  4.5686/6.2913, identical to 5 dp. Not confounded — **and a free result:
  bf16 GEMMs cost nothing at INFERENCE**, so the training-time effect lives in
  theta and the moments, not in the forward pass.
- **Arm identity.** bf16+RNE and bf16+SR print the SAME cuBLAS math-mode
  banner. Only `[nsl-matmul] bf16 operand cast: STOCHASTIC rounding`
  distinguishes them, and it prints for SR only — so the non-SR arms assert its
  ABSENCE. Checking only for what an arm should have is the half that missed a
  30-hour mislabel earlier in this campaign.

## Method: the paired delta is a usable proxy, the raw loss stream is not

Over the same 24,000 micro, on identical batches:

| | resolution |
|---|---|
| f32 arm's own loss slope | +0.50 ± 0.72 nats/100k micro (0.7σ) |
| bf16rne arm's own slope | −0.88 ± 0.71 (1.2σ) |
| **paired delta (bf16rne − f32)** | **+0.0216 ± 0.0061 (3.5σ)** |

Pairing tightens the error bar **8x** (0.0061 vs 0.0504) because identical
batches cancel composition noise. And it PREDICTED the held-out ordering: the
paired delta had bf16rne 0.182 nats ahead of f32 in the final window; held-out
then put it 0.290 (STACK) / 0.091 (WEB) ahead. Neither arm's individual slope
was significant.

**But the unpaired training-loss stream is worthless for this question.** Over
micro 80,000-160,000 it fitted −0.03 ± 0.13 nats/100k micro — flat to a tight
bound — while held-out rose 0.7 nats. Per-sample sd is 1.74; it cannot see a
generalization change of that size. Score held-out at every cadence
inspection; never infer health from the loss stream.

## Reproduce

    run-out/sr3arm_driver.sh          # three arms + held-out scoring
    tools/nslm_splice.py CKPT TEMPLATE OUT
    models/coder1b/val_from_splice.nsl

Arms write to `models/coder1b/checkpoints_sr3arm/<arm>_state.nslm`; the driver
asserts the rewritten recipe contains zero references to the chain's own
checkpoint before running.
