# The pre-chain posture bench — 1B production model, five arms (2026-08-27)

The final benchmark before the staged intermediate run's checkpoint chain
begins. The chain's execution fingerprint (#519) treats the wgrad choice as
arithmetic identity, so the flag set had to be settled BEFORE the first
checkpoint existed. Driver `models/benchmarks/posture_bench_1b.py`
(resident_bench.py's discipline: generated step-limited program in the
recipe's exact B2/A4 posture, fused-CE build refusal, OPT_STEP arrival
stamps, 10-update warmup discard, interleaved 3-round round-robin,
gpu-guard). Binary: current main @ f0113a4d (#536-#539 merged; the #537
elementwise/RoPE/scalar fusions and #538 multi-warp FA backward are
default-on in every arm). RTX PRO 4500 Blackwell 32 GB. Raw:
run-out/posture-bench/results.json (worktree artifact).

## Results (best of 3 interleaved rounds; spread < 0.5% on every arm)

| arm | flags beyond --source-ad | tok/s | alloc peak |
|---|---|---|---|
| P0 | --checkpoint-blocks --fuse-rmsnorm-backward | 3,856 | 19.47 GiB |
| P1 | P0 + --fuse-wgrad-accum | 4,299 | 19.38 GiB |
| P2 | P1 + --cuda-graphs | 4,334 | 19.38 GiB |
| P3 | P1 + --checkpoint-selective | 5,053 | 25.23 GiB |
| **P4** | **P3, run under NSL_MATMUL_BF16=1** | **6,568** | **25.23 GiB** |

P0's 3,856 vs the item-1 bench's 2,914 on the same flags = +32% from the
merged #537/#538 stack alone. P4 vs the chain-eve baseline: **+70%**; vs
the original resident measurement: **+125%**. Full-epoch arithmetic moves
from ~89 to ~40 days; the intermediate run's 1B-token gate from ~6 days to
~42 h; the checkpoint cadence (2,000 updates) from ~3.1 h to ~83 min.

## Decisions

- **--fuse-wgrad-accum: IN.** +11.5% at zero memory cost, reproduced
  within 0.3% across rounds — and settled pre-chain, which was the point:
  adding it later would refuse every prior checkpoint as arithmetic drift.
- **--cuda-graphs: OUT.** Engagement is real and deterministic — byte-
  identical counters every round (`regions=140 captured=76 replays=2052
  taints=0 mismatches=0 repaired_ops=0 eager=64`, ~66 replays/update), so
  the #539 capture fixes hold at 1B and the acceptance criterion (counters,
  not tok/s) was MET. The payoff was not: +0.8% over P1. A multi-day chain
  does not carry extra replay machinery for under a percent, and the
  graphs × selective composition is unmeasured besides.
- **--checkpoint-selective (modifier of --checkpoint-blocks): IN.** +17.5%
  over full recompute for +5.85 GiB of activation peak — 73% of the
  no-recompute speedup for 18% of its memory, now confirmed AT 1B, and it
  fits: 25.23 GiB alloc beside 12.9 GiB resident optimizer state
  (driver margin ~3-4 GiB on the 32 GB card, desktop included).
- **NSL_MATMUL_BF16=1: CONVICTED AND REMOVED (2026-08-28 addendum).** The
  caveat this entry originally carried ("convergence at 1B is checked, not
  assumed") fired one stage later than expected: stage 1 (36M tokens)
  looked clean, but the stage-2 leg ROSE +1.5 nats after ~80M tokens and
  the opt-14,000 state scored a full nat WORSE on held-out than opt-2000
  (VAL_LOSS_STACK 8.730 vs 7.734). A write-free matched-pair discriminator
  — the same opt-2000 state resumed with identical loader-slot/RNG batches,
  bf16 unset — separated the causes cleanly: a shared hard-data-region bump
  that f32 RECOVERS from (trend +0.16 over 98M tokens, descending again at
  the probe's end) while bf16 degrades monotonically (+0.53, paired delta
  growing to −0.42 by micro 32k). bf16's rounding drift accumulates in the
  moments/θ at this horizon. The chain runs f32-selective at 5,053 tok/s
  (+31% over the chain-eve posture); bf16's +30% stays on the table only
  behind future work (stochastic rounding on the GEMM inputs, or periodic
  f32 re-anchoring), with THIS record as the bar any such scheme must
  clear on a matched pair before a chain carries it.

## The chain posture

    nsl run --source-ad --checkpoint-blocks --checkpoint-selective \
        --fuse-rmsnorm-backward --fuse-wgrad-accum pretrain_prod.nsl

(bf16 removed from the line 2026-08-28 — see its conviction above.)

Staged stops (full 5,516,582-micro scheduler preserved throughout; stops
only at checkpoint-cadence update boundaries): ~2,200 updates (36M tokens,
past the 2,048-update warmup, first cadence write + resume + memory
flatness + post-warmup behavior), then ~250M tokens, then the 1B-token
hard gate, then 1.5-2B only if it answers a question.


## 2026-08-28/29 addendum: the LR verdict — 3e-5 held was the root instability

The bf16 conviction above was correct but incomplete. The restarted f32
chain ALSO reversed upward post-warmup (trough 6.84 at micro 14-16k, then
monotone rise to 8.32 by 30k) — a different starting θ than the
discriminator, same stationarily-shuffled data whose statistics were
verified normal in every suspect window (unique-token counts, repeat
rates, 8-gram duplication). Three independent legs rising at held lr=3e-5,
plus a re-read of the pilot arm (its "3e-5 works" descent happened while
its SHORT cosine decayed lr below ~2.5e-5 — held-3e-5 was never tested),
made sustained lr the prime suspect.

The trajectory probe settled it causally: the f32 chain's opt-2000 state
resumed WRITE-FREE under a halved-peak schedule (1.5e-5, 0.3 min ratio,
NSL_RESUME_ALLOW_TRAJECTORY_DRIFT=1 — the #533 escape's first production
use), identical loader/RNG batches:

| micro window | 3e-5 held | 1.5e-5 | paired |
|---|---|---|---|
| 8-16k | 7.43 -> 6.85 | 7.30 -> 6.32 | -0.13 -> -0.53 |
| 16-24k | REVERSES 7.55 -> 7.45 | descends 6.59 -> 6.06 | -0.96 -> -1.40 |
| 24-30k | 7.87 -> 8.32 | 6.31 -> 6.04 | -1.56 -> **-2.28** |

First-to-last-quarter trends: 3e-5 **+0.86**, 1.5e-5 **-1.03**. Halving
the peak converts divergence into the strongest descent of the campaign.
The recipe pair carries lr=1.5e-5 / min_lr=4.5e-6 with the full
three-revision provenance chain (1e-4 flat -> 3e-5 pilot-ranked but
never held-tested -> 1.5e-5 probe-measured); the chain restarts fresh on
it. bf16 remains separately convicted as an amplifier (standing rounding
bias — the concurrent session's PR #540 SR mode is the candidate fix,
behind its own matched-pair bar).
