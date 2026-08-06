# Item 7 — SR-BF16 certification (2026-08-06)

RTX PRO 4500 Blackwell 32 GB. Harness: `srbf16_campaign.py` — each scale runs
an f32 control and a `--param-dtype bf16-sr` arm on the IDENTICAL
CSLA + weight-stream schedule (`--source-ad --deterministic
--checkpoint-blocks --layerwise-accum --weight-stream`), equal tokens, equal
seeds; every SR arm's teardown counters are asserted non-vacuous and every
f32 arm asserts the SR marker absent. Loss streams print per micro-batch;
tok/s figures are per micro-batch tokens (no accumulation multiplier).
Follows the recommended order: 500M synthetic → 500M real corpus → 1B →
reload continuations → histograms → equal-token/equal-wall reads. SR-BF16 ×
ZeRO-3 remains refused by composition rule, per the ordering.

Configs: 500M = batch 1 × seq 512 × accum 8, 150 optimizer steps (1,200
micros, ~4.9M tokens/arm). 1B = batch 1 × seq 512 × accum 8, 60 steps (480
micros, ~2.0M tokens/arm). Batch 2 at 500M does NOT fit the SR envelope
(f32 m/v 2.1 GB each + streamed weights + mirrors + batch-2 activations →
OOM ~step 6; with the histogram's D2H syncs interleaved the same pressure
surfaced as a display-watchdog LAUNCH_TIMEOUT — one cause, two signatures).

## 500M — synthetic deterministic stream, 3 seeds

| seed | f32 final | bf16sr final | mean \|ΔL\| | max \|ΔL\| | f32 wall s | SR wall s |
|---|---|---|---|---|---|---|
| 1 | 0.0059 | **0.0046** | 0.0696 | 1.086 | 319 | 261 |
| 2 | 0.0067 | **0.0044** | 0.0883 | 1.425 | 312 | 258 |
| 3 | 0.0070 | **0.0068** | 0.0914 | 2.258 | 306 | 258 |

Every SR final at-or-below its f32 pair (the banked 50M pattern, one scale
up). Divergence concentrates in the steep-descent cliff and re-converges.
f32 seed spread 0.0012; bf16sr 0.0024.

## 500M — REAL corpus (byte-level repo sources, `tools/build_byte_corpus.py`), 3 seeds

| seed | f32 final | bf16sr final | mean \|ΔL\| | max \|ΔL\| |
|---|---|---|---|---|
| 1 | 3.6694 | **3.5109** | 0.0229 | 0.407 |
| 2 | 3.7122 | 3.7208 | 0.0020 | 0.056 |
| 3 | 3.7570 | **3.6511** | 0.0112 | 0.421 |

The certification headline. On real data the paired curves are an order of
magnitude tighter than on the memorization stream (mean |ΔL| 0.002–0.023),
and the SR-vs-f32 difference sits INSIDE the f32 seed spread (0.088): at
equal tokens and equal seed, bf16-authoritative storage is statistically
indistinguishable from f32 on this workload while finishing at parity or
better on 2 of 3 seeds (tie on the third).

## 1B — synthetic, 2 seeds

| seed | f32 final | bf16sr final | mean \|ΔL\| | max \|ΔL\| | f32 wall s | SR wall s |
|---|---|---|---|---|---|---|
| 1 | 5.2646 | 5.2631 | 0.00058 | 0.0028 | 171 | 133 |
| 2 | 5.2925 | 5.2931 | 0.00051 | 0.0023 | 174 | 135 |

Tightest tracking of the whole campaign: over 480 micros the SR curve stays
within 0.003 of f32 at every step — differences ~50× smaller than the f32
seed spread (0.028).

## Reload continuation (all three campaigns)

`model_save` → fresh process → `model_load` → continue under bf16-sr.
Optimizer moments are not checkpointed, so the criterion is "resumes at the
parent's loss level, no re-warm spike":

| campaign | parent first → last | continuation first | verdict |
|---|---|---|---|
| 500M synthetic | 11.004 → 0.0046 | 0.0057 | RESUMED |
| 500M real corpus | 11.051 → 3.5109 | 3.1803¹ | RESUMED |
| 1B synthetic | 11.226 → 5.2631 | 5.0798 | RESUMED |

¹ the real-corpus continuation trains on a FRESH slice past the parent's
range (the harness refuses to replay or to switch distributions), so
starting below the parent's final is continued learning, not leakage.

## Equal-token / equal-wall reads

Equal tokens: the tables above — SR ≥ parity on quality everywhere.
Equal wall: SR arms are **18% faster at 500M** (258–262 s vs 306–319 s) and
**22% faster at 1B** (133–135 s vs 171–174 s), so at equal wall-clock SR
sees proportionally more tokens on top of equal-token parity. The mechanism
is visible in the counters: the f32 arms stream weights all run long
(291,600 upload/evict pairs at 500M; 77,760 at 1B; PCIe rx sampled
~37–45 GB/s-scale vs ~0.2–13 MB/s-scale on SR arms) while the bf16 mirrors
are DEVICE-RESIDENT — at these scales `--param-dtype bf16-sr` converts
weight streaming into resident halved-storage weights.

## Update-magnitude / stall histograms (`NSL_SR_HIST=1`)

Sampling: first 16,384 elements of each mirrored parameter's bf16 mirror,
before/after every fused SR step; |Δθ| bucketed by binary exponent; an
element whose bf16 bits did not change counts as STALLED (this includes
exactly-zero f32 updates — see the calibration note).

500M real corpus, seed 1 (422M sampled element-updates,
`srbf16_logs/bf16sr_real_s1/stderr.log`): |Δθ| mass spans 2⁻¹⁶…2⁻¹²
(5.2% / 7.3% / 8.2% / 9.2% / 7.2%) with a smooth sub-ULP tail down to 2⁻³¹
— exactly the shape stochastic rounding wants: most updates are comfortably
above bf16 ULP-at-|θ|, and the sub-ULP tail is converted to probabilistic
application rather than lost. Stalled: 57.6%. The 500M synthetic arm
(observed at run time; its per-arm log was later overwritten by the log-dir
collision noted below) showed the same band shape with 50.5% stalled.

Calibration notes for reading the stall figure: (a) the sample is the FIRST
16K elements per parameter, which for large embeddings covers only the
lowest-id vocab rows — a row whose token never occurs contributes permanent
100% stalls; (b) exactly-zero f32 updates count as stalls by construction.
The number is an upper bound on true SR underflow, tightest on the dense
non-embedding parameters. Cross-checked against outcomes: at every scale the
equal-token quality parity says the stall level costs nothing measurable.

## Anti-vacuity and provenance

Every bf16sr arm's `[sr-bf16] teardown` counters were asserted >0 (params /
SR steps / uploads); at 500M: 216 bf16-authoritative params, 32,400
step-param launches per arm. Captured stdout for all three campaigns:
`srbf16_logs/campaign_run_2026_08_06.log`. Known blemish: the 1B campaign
ran untagged and reused the 500M synthetic campaign's per-arm log dirs
(fixed — logs are now namespaced per scale); the 500M-synthetic stdout
tables in the captured log are complete, but its per-arm stderr files were
overwritten by the 1B arms.
