# The real-corpus SR-BF16 differential — F32 oracle vs `--param-dtype bf16-sr` (roadmap item 6)

**Date:** 2026-08-18 · **Hardware:** RTX PRO 4500 Blackwell 32 GB, CUDA 13.x ·
**Harness:** `models/benchmarks/sr_differential_50m.py` over
`models/coder50m/sr_differential.nsl` — one epoch over an 8.39M-token slice
of the production corpus (`train_new.bin`), held-out 512-batch tail slice
for validation, fixed-lr AdamW 3e-4 / wd 0.1 / accum 2, dropout 0.1 active,
per the banked cert conventions (no scheduler, no grad_clip, no
`@fused_lm_ce`). Flags: `--source-ad --deterministic --checkpoint-blocks
--layerwise-accum --weight-stream` (+ `--param-dtype bf16-sr` on SR arms).
The F32 oracle runs the IDENTICAL residency schedule, so the comparison
isolates authoritative bf16 storage + the fused stochastically-rounded
AdamW step — the dtype, not the stack. Every SR arm asserts the
`[sr-bf16] teardown` counters are non-vacuous; every F32 arm asserts the
marker is absent.

## Why this campaign exists

The 2026-07-30 banked trajectories (`SRBF16_TRAJECTORY_2026_07_30.md`)
established the mechanism at 500 steps on a **synthetic memorizable
stream** whose loss collapses to ~0.01 — a regime where a rounding-induced
quality delta has nothing to show up in. The roadmap item's criterion:
real corpus, an F32 oracle, and a quantified quality delta **before**
calling SR the default.

## Finding 1: this composition is not run-to-run deterministic — the
## paired-comparison design is invalid, and the banked doc never tested it

The first harness assumed `--deterministic` reproduces an arm
bit-for-bit and compared one F32/SR pair per seed. The determinism
control refuted it: same-seed same-arm replicates diverge **from the
first sampled step**, with dropout on or off (probe:
max step-gap 2.4e-3 with dropout, 2.9e-4 without, both diverging at
sample 0; VAL scatter across three same-seed F32 runs ≈ 0.1 over a
4096-update epoch). `--deterministic` covers RNG seeding and
compile-time nondeterminism detection — not the GPU backward's atomic
accumulation order under the CSLA + weight-stream composition.

The 2026-07-30 campaign asserted "loss stream is deterministic" from
**cross-arm first-loss identity**, which pins only the shared init — it
never reran the same arm twice. This campaign is the first same-arm
replicate test of the composition, and it fails. (Same lesson as the
item-5 floors: a property assumed from adjacent evidence is not a
property measured.)

Consequence: the verdict below is statistical — N replicates per
seed × arm, pooled per-arm means judged by Welch SE — not paired.

## Results

Per-arm pooled statistics (3 seeds × 2 replicates per arm; every run one
epoch = 4096 optimizer updates over 8.39M real-corpus tokens, VAL = mean
held-out CE over 512 never-trained batches):

| arm | n | VAL mean | VAL sd | VAL min..max | train tail64 mean | tok/s |
|-----|---|----------|--------|--------------|-------------------|-------|
| F32 oracle | 6 | 9.011521 | 0.114712 | 8.817903..9.136416 | 6.490777 | ~13.57k |
| SR-BF16 | 6 | 9.082296 | 0.139863 | 8.920083..9.254822 | 6.498234 | ~13.56k |

Raw VALs — F32: 8.8179, 8.9362, 9.0513, 9.0607, 9.0667, 9.1364 ·
SR: 8.9201, 8.9237, 9.0537, 9.1641, 9.1775, 9.2548.

Per-seed first losses pin the shared init (F32 vs SR first losses differ
by only ~3e-4 — the bf16 storage quantization of the first forward), and
the two watchdog-retried arms reproduce the same statistics as their
neighbors.

## Finding 2: the quality delta

**ΔVAL (SR − F32) = +0.0708, Welch SE 0.0738, 2SE bound ±0.1477 —
statistically indistinguishable from run noise.**

Reading it honestly in both directions: the 95%-ish interval is
[−0.077, +0.219] nats of held-out CE. The data is consistent with SR
being slightly better, identical, or up to ~0.22 nats worse; the point
estimate leans 0.07 worse but at less than 1 SE of separation. Train-loss
tails are practically identical (6.491 vs 6.498). Wall time is a wash at
50M (the banked 07-30 campaign's PCIe-traffic advantage for SR — 3.8 GB/s
→ 0.2-0.8 GB/s rx — is the real systems win; it reproduces here as equal
throughput with far less bus traffic).

## Recommendation on "SR as default"

**Not promoted to default.** No quality blocker was found — the delta is
bounded by run noise — but "indistinguishable at n=6, 50M, 1 epoch" is an
absence of evidence, not a demonstrated equivalence, and the point
estimate leans slightly against SR. The measurement power here is capped
by Finding 1 (the composition's ~0.11-0.14 VAL run scatter); each extra
replicate pair costs ~21 GPU-minutes and shrinks the bound slowly
(SE ∝ 1/√n).

Recommendation: keep `--param-dtype f32` the default; carry SR forward as
the certified memory/bandwidth option it already is; and make the
default call at the 500M campaign (roadmap item 9), where (a) the same
harness applies unchanged, (b) bf16 rounding error has more surface to
matter, and (c) the VRAM/PCIe savings become decisive rather than
convenient — so whichever way the quality bound lands there, the decision
is being made where the stakes are real.

## Environment notes

- Tape AD plays no role here (both arms are source-AD; the composition
  requires it). Item 5's differential separately certifies source-AD
  against the tape reference.
- Three transient `CUDA_ERROR_LAUNCH_TIMEOUT` trips occurred across the
  campaigns (display-GPU watchdog under desktop contention); every
  identical arm reran clean on the first retry. The harness retries that
  signature once and treats a second trip as a real failure.
- ~12.6-13.6k tok/s per arm at 50M under this composition (batch 1 ×
  seq 1024, accum 2); SR arms run within ~5% of F32 wall time.

## Reproduction

```bash
python models/benchmarks/sr_differential_50m.py \
    --nsl <cuda-nsl-binary> --seeds 3 --replicates 2
```
