# Resident-1B throughput + geometry sweep (next-roadmap item 1, 2026-08-25)

Items 11/12 (PR #524) made the fully-resident 1B configuration COMPLETE and
recorded its memory posture to the byte — but no throughput comparison of
the resident path against the old `--optim-state-offload` posture existed
anywhere. This measures it, plus the B×A geometry sweep at a constant
16,384 effective tokens per optimizer update, in the exact production
posture (`pretrain_prod.nsl`'s flags, grad_clip=1.0, AdamW + warmup_cosine,
`@fused_lm_ce`). Driver: `models/benchmarks/resident_bench.py`.

Method: every arm built once (`nsl build`; fused-CE route asserted in the
build stderr — the route is baked at compile time), 3 timing rounds
interleaved round-robin (best round reported; the PR #454 cold-clock
lesson; per-round spread ≤ 1.04% on every arm), 31 optimizer updates per
run (30 requested + one from the drop_last slack) with the first 10
excluded — 20 steady-state intervals per round — tok/s from `OPT_STEP`
arrival stamps (cadence-independent; stamps persisted per round),
NSL_EVENTS exact-byte memory series per step, and SEPARATE instrumented
builds/runs for the phase split, PCIe counters and kernel trace —
instrumentation is never timed. RTX PRO 4500 Blackwell 32 GB, tf32
roofline 87.6 TFLOPS (measured), `Scale("1b").flops_per_token(2048)`,
synthetic 507,904-token stream, binary + worktree commit recorded in
results.json.

## The headline: the resident path is ~30% faster

| arm | tok/s (best) | MFU | alloc peak | activations | optim on dev | PCIe h2d/d2h (per run) | fwd/bwd/opt (s) |
|---|---|---|---|---|---|---|---|
| **resident B2/A4** | **2,914** | **22.8%** | 19.47 GiB | 3.45 GiB | 12.0 GiB | 1 MiB / 0 | 24 / 147 / **2** |
| offload B2/A4 | 2,234 | 17.5% | 7.43 GiB | 3.42 GiB | 0 | 16,386 / 12,289 MiB | 24 / 166 / **36** |
| resident B4/A2 | 2,827 | 22.1% | 22.45 GiB | 6.43 GiB | 12.0 GiB | 1 MiB / 0 | 25 / 151 / 2 |
| offload B4/A2 | 2,246 | 17.6% | 10.41 GiB | 6.40 GiB | 0 | 16,386 / 12,289 MiB | 25 / 160 / 36 |
| resident B8/A1 | **OOM** | — | see below | — | — | — | — |
| offload B8/A1 | **OOM** | — | see below | — | — | — | — |

- **+30.4% tok/s at B2/A4** (2,914 vs 2,234) and +25.9% at B4/A2. Item 11's
  memory work bought this directly: before it, the resident configuration
  OOM'd in the forward and the 30% was structurally unreachable.
- **Where the 30% lives**: the phase split puts the offload optimizer phase
  at 36 s vs the resident 2 s over 30 updates (~1.13 s/update of moment
  staging), plus a slower backward (166 vs 147 s — gradient staging rides
  the same lanes). The kernel-trace breakdowns of the two arms are
  IDENTICAL — all 29 kernel names and launch counts match, times within
  ~1% (top: `flash_attn_bwd_main` 0.79 s, `sgemm_cublas` 0.67 s/574
  launches) — the compute stream does not differ; the delta is entirely
  host↔device optimizer traffic. PCIe counters agree: 16.0 GiB h2d +
  12.0 GiB d2h per run offloaded, ~1 MiB resident.
- **Steady-state MFU 22.8%** against the measured tf32 roofline. (The
  smoke's 23.1% at 4 opt steps is the same number with a noisier window.)

## Geometry: B2 is the sweet spot; B8 does not fit

- **B4/A2 is slightly SLOWER than B2/A4** on the resident path (2,827 vs
  2,914) at +3.0 GiB of peak (22.45 vs 19.47). The larger GEMMs do not pay
  for the larger activation surface; there is no free throughput in bigger
  micro-batches here.
- **B8/A1 OOMs in BOTH postures** on a same-shaped 512.0 MB request (the
  MLP intermediate at B8 rows: 16,384 × 8,192 × 4 B) with 237–261 MB free
  of 31.39 GB. The abort's own NSL_MEMSTATS dump names the mechanism:
  LIVE first-forward activation allocations — resident: activations
  17.05 GB current (+ 4.02 weights + 8.00 moments ≈ 29.1 GB allocated,
  `Pool drained: 0 B`, nothing reserved-but-reclaimable); offload:
  activations 24.49 GB (+ 4.02 weights), with the final abort raised in
  the CPU-fallback re-upload after a recoverable `nsl_mul_f32` OOM.
  (An earlier draft blamed "reservation growth" off the last
  STEP-BOUNDARY events sample, where activations are ~0 by construction —
  the exact step-boundary-is-not-a-peak mistake this record warns about;
  the review caught it against the abort dumps.) B8/A1@2048 is out of
  reach on 32 GB in this posture, offload included, and A1 cannot run any
  `--layerwise-accum` arm either (accumulation ≥ 2 required there).
- Allocation series: a +38.5 MiB rise across the first half (allocator
  warm-up) then byte-flat to the end on every completing arm (exact-byte
  NSL_EVENTS step-boundary series) — corroborating the slope-gate design
  that item 2 ships (its gate asserts from steady state on).

## What this recommends

The production 1B recipe's documented posture (`--optim-state-offload`,
required before item 11) now costs ~23% throughput for 12 GiB of headroom
the resident run does not need (allocator peak 19.47 GiB; the
card-sizing number is the DRIVER peak, 22.4 GiB of 31.39 usable —
~9 GiB of margin). **The resident configuration should become the 1B
default run line**, with offload kept as the escape hatch for smaller
cards — that change belongs to the recipe, not this benchmark, and should
land with the corpus-v2 pilot (next-roadmap item 6). For the PyTorch
comparison (item 7), the resident arm is the NSL number to bring.

## Traps hit while building the driver

- The generated program omitted `m.to(cuda)`: host-resident params made
  `nsl run` "hang" (the whole graph silently reconciled onto the
  single-threaded host — the defect-1 class item 2 now REFUSES) and the
  built binary die in fused-CE with CUDA_ERROR_ILLEGAL_ADDRESS. The
  committed `mem_probe_32step.nsl` was the discriminator (it completes;
  the diff was one line).
- `NSL_PHASE_TIMING` is a COMPILE-time knob (stmt.rs bakes sync+print into
  the binary): the phase-instrumented binary is a separate build artifact,
  and Pass A must never time it.
- `mem_probe_32step.nsl` line 19 had lost its `#` (a corrupted paste) and
  the probe's token slice (`mem_probe_32step.bin`) was never committed nor
  its generation documented — both fixed/regenerated here (the slice is
  `head -c 262144` of `prod_train_slice.bin`).
