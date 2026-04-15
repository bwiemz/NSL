# NSL-Coder-1B — FASE load-bearing scale

Llama-3.2-1B-ish architecture scaled from `coder500m`. ~1.07B params.
This is the scale where **FASE Deferred is the difference between fit
and OOM** on a 16 GB GPU.

## Architecture

|  | coder500m | coder1b | coder7b |
|---|---|---|---|
| d_model | 1280 | **2048** | 4096 |
| n_layers | 24 | **16** | 32 |
| n_heads | 20 | **32** | 32 |
| n_kv_heads | 10 | **8** | 8 |
| head_dim | 64 | **64** | 128 |
| d_ff | 3520 | **8192** | 14336 |
| vocab | 49152 | 49152 | 49152 |
| max_seq_len | 1024 | **2048** | 2048 |
| params (f32) | ~505M | **~1.07B** | ~7.2B |

## Memory budget (f32, RTX 5070 Ti / 16 GB)

| | tape AD | source-AD + FASE |
|---|---|---|
| Parameters | 4.3 GB | 4.3 GB |
| AdamW state (m + v) | 8.6 GB | 8.6 GB |
| Gradients (peak) | **4.3 GB** (all live) | **0.4 GB** (one largest — the 49152×2048 embedding) |
| Activations + CUDA ctx | ~1 GB | ~1 GB |
| **Total peak** | **~18 GB → OOM on 16 GB** | **~14 GB → fits** |

## Files

- `config.nsl` — hyperparameters.
- `model.nsl` — `NSLCoder`, `TransformerBlock`, `SwiGLUFFN` definitions.
- `pretrain_fase.nsl` — runnable FASE demo with `grad_accumulation=8`,
  AdamW, `grad_clip=1.0`.

## Running

```bash
cargo build --release --bin nsl --features cuda

# Tape-AD baseline — EXPECTED TO OOM on 16 GB:
./target/release/nsl run --profile-memory models/coder1b/pretrain_fase.nsl

# Source-AD — FASE hook active, fits:
./target/release/nsl run --source-ad --profile-memory models/coder1b/pretrain_fase.nsl

# No-GPU planner check:
./target/release/nsl check --training-report models/coder1b/pretrain_fase.nsl
```

Expected training-report output:

```
=== Training Pipeline Report ===
[Block 1]
  FASE (Fused Accumulation-Step Elimination):
    grad_accumulation: 8
    optimizer:         AdamW
    mode:              Deferred
    backward_phases:   AccumulateOnly × 7, FinalTwoPhase × 1
    two_phase_clip:    true
```

## Expected result

Source-AD should run ~10 optimizer steps (30-60 seconds of GPU time
each at this size) with peak VRAM stable around 13-14 GB, identical
live-block count across steps (confirming the leak-free accumulate
shipped in `550e6f2`).

Tape-AD should OOM during the first backward pass — the
`allocated_bytes` in the memory profiler will exceed 16 GB before
step=0 completes.

This is the headline demo: **FASE turns a 1B-param training run from
an H100-class workload into a consumer-GPU workload.**
