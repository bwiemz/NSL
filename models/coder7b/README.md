# NSL-Coder-7B — FASE scale demo

Llama-3-8B-ish architecture scaled from `coder500m` by widening
`d_model` and deepening blocks. Same GQA + SwiGLU + RoPE +
weight-tied-LM-head structure — only hyperparams change.

## Architecture

|  | coder500m | coder7b |
|---|---|---|
| d_model | 1280 | **4096** |
| n_layers | 24 | **32** |
| n_heads | 20 | **32** |
| n_kv_heads | 10 | **8** (4:1 GQA) |
| head_dim | 64 | **128** |
| d_ff | 3520 | **14336** |
| vocab | 49152 | 49152 |
| max_seq_len | 1024 | **2048** |
| RoPE theta | 10000 | **500000** |
| params (f32) | ~505M | **~7.2B** |

## Files

- `config.nsl` — hyperparameters.
- `model.nsl` — `NSLCoder`, `TransformerBlock`, `SwiGLUFFN` definitions.
- `pretrain_fase.nsl` — runnable FASE demo with `grad_accumulation=8`,
  AdamW, `grad_clip=1.0`.

## Memory budget (f32)

| | size |
|---|---|
| Parameters | ~28.7 GB |
| AdamW state (m + v) | ~57.4 GB |
| FASE gradient peak (one largest param at a time, ~49152×4096 = 201M f32) | ~0.8 GB |
| Activations (batch=1, seq=2048) | ~2-3 GB |
| **Total peak VRAM** | **~89 GB** |

The 7B class does not fit on consumer GPUs at f32. FASE cuts
gradient peak to one parameter at a time, but the params + AdamW
state baseline is still 86 GB. Paths forward:

- **Full-fat (H100 80GB / A100 80GB + CPU spill)** — run as-is.
- **bf16 / fp16 params + state** — ~44 GB, fits on a single 48-80 GB card.
- **CPU offload of AdamW state** — leaves ~30 GB on GPU (params + activations + FASE grads). Requires roadmap item M36 memory planner.
- **Shrink** — drop to 16 layers or d_ff=8192 for a ~3.5B variant that fits on 24 GB.

## What you can do without full-fat hardware

```bash
# See the FASE planner's decision — no GPU execution, just compile:
nsl check --training-report models/coder7b/pretrain_fase.nsl
```

Expected report shape:

```
=== Training Pipeline Report ===
File: models/coder7b/pretrain_fase.nsl
Train blocks found: 1

[Block 1]
  Model: m

  FASE (Fused Accumulation-Step Elimination):
    grad_accumulation: 8
    optimizer:         AdamW
    mode:              Deferred
    rationale:         AdamW supports deferred first-moment accumulation with batch-variance v approximation
    backward_phases:   AccumulateOnly × 7, FinalTwoPhase × 1
    two_phase_clip:    true
```

Confirms the FASE plan scales identically from 500M to 7B — same
`backward_phases` shape, just wider tensors.

## Running at scale (when you have the hardware)

```bash
cargo build --release --bin nsl --features cuda

# Baseline (tape AD — all gradients materialise simultaneously):
./target/release/nsl run --profile-memory models/coder7b/pretrain_fase.nsl

# Source-AD (FASE Deferred fires, one grad at a time):
./target/release/nsl run --source-ad --profile-memory models/coder7b/pretrain_fase.nsl
```

Expected delta on ≥80 GB hardware:

| | tape AD | source-AD + FASE |
|---|---|---|
| Parameters | 28.7 GB | 28.7 GB |
| AdamW state | 57.4 GB | 57.4 GB |
| Gradients (peak) | 28.7 GB (all live) | 0.8 GB (single largest) |
| **Peak VRAM** | **~117 GB** | **~89 GB** |

FASE saves ~28 GB of peak gradient memory at this scale — enough to
turn an "H100 + CPU spill" configuration into a single H100 fit on
bf16.

## Scaling template — 13B, 30B, 70B

The `train(...)` block structure, FASE plan shape, and codegen paths
are identical across scales. To go bigger, edit `config.nsl`:

- **13B**: `d_model=5120, n_layers=40, n_heads=40, n_kv_heads=8, d_ff=13824`.
- **30B**: `d_model=6656, n_layers=60, n_heads=52, n_kv_heads=8, d_ff=17920`.
- **70B**: `d_model=8192, n_layers=80, n_heads=64, n_kv_heads=8, d_ff=28672`.

Every such variant inherits the validated FASE pipeline from
`coder500m` — the numerics, the hook, the two-phase clip, the
bias-correction scalars, the leak-free accumulate.
