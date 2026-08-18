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

## Finding 1: run-to-run determinism is LOST at 50M scale under this
## composition — filed as a bug, and it invalidated the paired design

The first harness assumed `--deterministic` reproduces an arm bit-for-bit
and compared one F32/SR pair per seed. The determinism control refuted
it: same-seed same-arm replicates diverge **from the first sampled
step**. The committed probe (`models/coder50m/sr_determinism_check.nsl`)
pins the numbers: max step-gap 2.9e-4 with TF32 on (the campaign
configuration), 1.1e-4 with `NSL_MATMUL_TF32=0` (3× smaller, not zero),
2.4e-3 with dropout active; VAL scatter across same-seed F32 epochs
≈ 0.09–0.11.

Two facts sharpen this from "flag scope" into a **bug**:

- M46's documented contract is device bit-reproducibility (kernel-variant
  routing exists for exactly this: the no-atomics embedding backward and
  the non-coalesced sum_dim route are both gated on `--deterministic`).
  The clap help's narrower phrasing ("compile-time non-determinism
  detection") undersells the repo's own docs.
- `srbf16_e2e_deterministic_and_sane_gpu` — a same-arm bit-identity
  rerun of this exact composition at toy scale (6 steps) — **passes on
  this machine today** (re-verified during this campaign). So the
  contract holds at gate scale and breaks at 50M: scale-dependent,
  consistent with shape-dependent kernel/algorithm selection rather than
  a single unguarded op. The 2026-07-30 doc cited that gate when calling
  its dice "bit-identical rerun asserted"; what had never been tested
  before this campaign is same-arm determinism **at trajectory length
  and model scale** — and there it fails.

Filed in `bugs.md` (memory) for a dedicated campaign; out of scope here.
Consequence for this campaign: the verdict below is statistical, and
seed-PAIRED — seeds are shared across arms, so init effects cancel in
the per-seed difference (the first version of this harness pooled
3 seeds × 2 replicates as 6 i.i.d. samples and printed SORTED VALs,
which both mis-specified the SE and destroyed the seed pairing in the
banked record — caught in review; the tables below preserve pairing).

## Results

Per-seed VALs (mean held-out CE over 512 never-trained batches; each cell
is [replicate-0, replicate-1]; every run one epoch = 4096 optimizer
updates over 8.39M real-corpus tokens):

| seed | F32 oracle VALs | SR-BF16 VALs | seed ΔVAL (mean SR − mean F32) |
|------|-----------------|--------------|--------------------------------|
| 1000 | [9.060671, 8.936207] | [9.177453, 9.053670] | +0.117123 |
| 1001 | [9.136416, 9.051269] | [9.164089, 9.254822] | +0.115613 |
| 1002 | [9.066659, 8.817903] | [8.920083, 8.923660] | −0.020410 |

Pooled per-arm (for scale): F32 9.0115 ± 0.1147 (n=6), SR 9.0823 ± 0.1399
(n=6). Within-seed replicate scatter (σ_rep ≈ 0.095) dominates the seed
effect (σ_seed ≈ 0.04): the nondeterminism noise, not the init, is the
limiting factor of this design's power. Train tail (last 64 samples =
final 512 micro-steps, i.e. the last 6.25% of the epoch): F32 6.4908,
SR 6.4982 — directionally consistent with the VAL delta and equally
inside noise. Throughput ~13.57k tok/s both arms.

## Finding 2: the quality delta

**Seed-paired ΔVAL (SR − F32) = +0.0708, paired SE 0.0456 (df 2,
t-crit 4.303): 95% CI [−0.125, +0.267] — covers zero; no detectable
delta.** The secondary pooled Welch view: SE 0.0738 at df 9.6
(t-crit 2.24 → CI [−0.095, +0.236]); it agrees, and it is reported for
reference only since it ignores the seed pairing.

What this design can and cannot say, stated plainly:

- Two of three seeds landed +0.12 against SR, one landed −0.02. The
  point estimate leans against SR at well under significance.
- Power is limited: against the paired criterion, a true 0.05-nat SR
  regression would be flagged only rarely; 80% power needs a true delta
  around ~0.2 nats. So this campaign RULES OUT a large regression
  (≳0.25 nats) and is silent below that — "no detectable delta" is
  absence of evidence at n_seeds=3, not demonstrated equivalence.
- **Coverage disclosure:** under `--param-dtype bf16-sr`, the tied
  embedding (25.2M of ~48.8M parameters, ~52% of the mass — it is
  view-rooted through the weight-tied LM head and therefore resident,
  not streamed) stays **f32-authoritative in BOTH arms**; SR rounds the
  streamed transformer-block parameters only. The measured delta is the
  cost of bf16-ing roughly half the model. That is the flag's genuine
  production behavior, so the arms are validly compared — but at 500M+
  the embedding fraction shrinks and SR coverage grows, which is one
  more reason the default call belongs there.
- Wall time is a wash at 50M; the banked 07-30 PCIe advantage (3.8 GB/s
  → 0.2–0.8 GB/s rx) is the systems win that motivates SR.

## Recommendation on "SR as default"

**Not promoted to default.** No large regression (the paired 95% CI caps
a true SR cost at ~+0.27 nats, and the evidence leans well under that),
but three seeds cannot certify equivalence, the point estimate leans
slightly against SR, and half the parameter mass at 50M isn't covered by
SR at all (the resident tied embedding). The measurement power is capped
by Finding 1's nondeterminism noise; each extra replicate pair costs
~21 GPU-minutes and shrinks the CI slowly.

Recommendation: keep `--param-dtype f32` the default; carry SR forward
as the certified memory/bandwidth option it already is; make the default
call at the 500M campaign (roadmap item 9), where (a) this harness
applies unchanged, (b) SR actually covers most of the parameter mass,
and (c) the VRAM/PCIe savings become decisive rather than convenient.

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
