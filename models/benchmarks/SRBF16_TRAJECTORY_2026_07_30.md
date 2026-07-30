# SR-BF16 trajectory certification — 50M, 2026-07-30

Roadmap item 16's namesake work: the `--param-dtype bf16-sr` mechanism has
been unit/e2e-gated since P4, but the longest banked run was ~6 optimizer
steps on a toy FFN. This campaign banks full trajectories: **500 optimizer
steps × 3 seeds × {FP32, SR-BF16} at Coder-50M, equal tokens, identical
schedule**, plus a checkpoint-save/reload continuation arm.

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX PRO 4500 Blackwell, 32 GB, sm_120 |
| Driver / CUDA | 610.43.03 / 13.3 (ptxas V13.3.73) |
| Program | `models/coder50m/pretrain_srbf16_cert.nsl` (batch 1 × seq 1024, `grad_accumulation=2`, fixed-lr AdamW 3e-4, wd 0.1, **no grad_clip** — `--layerwise-accum` refuses two-phase clipping) |
| Flags | `--source-ad --deterministic --checkpoint-blocks --layerwise-accum --weight-stream` (+ `--param-dtype bf16-sr` on the SR arms) |
| Data | synthetic learnable stream (`gen_tokens.py`, fixed block, identical across arms/seeds) |
| Harness | `models/benchmarks/srbf16_campaign.py` (reuses the p0 runner; every bf16 arm hard-asserts the `[sr-bf16]` teardown counters are non-vacuous, every f32 arm asserts they are absent) |

The FP32 control runs the **identical CSLA + weight-stream residency
schedule** — the only delta on the SR arms is authoritative bf16 storage +
the fused SR step, so the comparison isolates the dtype, not the stack.

## Results (500 optimizer steps = 1000 micro-batches per arm)

| arm | first loss | last loss | train wall s | tok/s | GPU util % | PCIe rx peak MB/s |
|---|---|---|---|---|---|---|
| f32_s1 | 10.9283 | 0.0138 | 131 | 7,792 | 92 | 3,777 |
| bf16sr_s1 | 10.9282 | **0.0120** | 127 | 8,094 | 93 | 757 |
| f32_s2 | 10.9066 | 0.0136 | 132 | 7,747 | 92 | 3,775 |
| bf16sr_s2 | 10.9066 | **0.0134** | 129 | 7,956 | 93 | 208 |
| f32_s3 | 10.9089 | 0.0132 | 137 | 7,494 | 90 | 7,205 † |
| bf16sr_s3 | 10.9088 | **0.0129** | 128 | 8,001 | 93 | 174 |
| bf16sr_continue | 0.0115 | 0.0041 | 26 | 7,957 | 86 | 233 |

† f32_s3's resource columns (and its 17.0 GB smi peak, not shown) are
contaminated: an unrelated GPU test binary ran concurrently during that
arm. Its **loss stream is deterministic and unaffected** (identical first
loss to bf16sr_s3 pins the shared init); only the resource sampling for
that one arm is unreliable.

### Equal-token paired curves (same seed, same data)

| seed | mean \|ΔL\| | max \|ΔL\| (micro, f32 loss there) | first-100 mean \|ΔL\| | last-200 mean \|ΔL\| | f32 final | bf16sr final |
|---|---|---|---|---|---|---|
| 1 | 0.090 | 0.857 (484, at loss 3.74) | 0.0062 | 0.0078 | 0.0138 | 0.0120 |
| 2 | 0.098 | 1.397 (472, at loss 4.54) | 0.0135 | 0.0039 | 0.0136 | 0.0134 |
| 3 | 0.074 | 0.838 (306, at loss 6.55) | 0.0030 | 0.0040 | 0.0132 | 0.0129 |

The divergence signature is the benign one: curves agree tightly early
(≤0.014 mean over the first 100 micros), separate **only inside the steep
descent cliff** — where trajectory sensitivity to any parameter
perturbation is maximal and the SR dice legitimately pick different paths
— and **reconverge to the same floor** (≤0.008 mean over the last 200).
Nothing resembling divergence or a loss floor gap.

### Seed spread of final losses

- FP32: 0.0132–0.0138 (range 0.0006)
- SR-BF16: 0.0120–0.0134 (range 0.0014)

SR adds seed variance of the same order as init-seed variance, and every
SR final lands **at or below** its paired FP32 final.

### Checkpoint-save/reload continuation

`bf16sr_continue` loaded `bf16sr_s1`'s `.nslm` (written while the
authoritative weights lived in device bf16 mirrors) and resumed under the
same bf16-sr stack: first loss **0.0115 vs the parent's final 0.0120** —
resumption, not re-warmup — and kept improving to 0.0041 over 100 more
steps. (Optimizer moments are not checkpointed; this certifies weight
round-trip + fresh-moment continuation, the supported semantic.)

### Throughput / transfer profile

SR arms are consistently ~2–4% **faster** (7,956–8,094 vs 7,494–7,792
tok/s) with **5–20× lower peak PCIe rx** (174–757 vs 3,775+ MB/s): FP32
weight-streaming re-uploads every layer's f32 params from the host mirror
each window, while bf16-sr keeps 2-byte device mirrors and widens D2D.
At 50M the wall-clock effect is small; the transfer profile is the
scaling-relevant signal.

## What this certifies / what remains

Certified here (item 16, 50M leg): multi-seed full-trajectory parity at
equal tokens, non-vacuous SR schedule on every run, deterministic
seed-keyed dice (bit-identical rerun asserted by the existing e2e gate),
save/reload continuation.

Still open for full item-16 closure:
- 500M (several hundred steps) and a medium 1B run — the harness takes
  `500m` once `models/coder500m/pretrain_srbf16_cert.nsl` twins are added;
  the 32 GB card has headroom the 16 GB predecessor lacked.
- Real-corpus (non-synthetic) equal-token comparison.
- Parameter-update underflow histograms (needs new instrumentation; the
  SR unbiasedness unit gate covers the rounding math itself).
- bf16-sr × ZeRO-3 composition stays refused until the above bank exists
  at larger scale.

## Reproduce

```
CARGO_TARGET_DIR=... cargo build --release -p nsl-cli --features cuda
cd models/benchmarks
NSL_BIN=<that nsl> python3 srbf16_campaign.py 50m --steps 500 --seeds 1 2 3 --continue-steps 100
```

Raw logs: `models/benchmarks/srbf16_logs/` (gitignored; summary JSON at
`srbf16_logs/campaign_summary.json`).
