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

## Reading the cert curves: this is a compute-limited budget

`config.nsl`'s pretrain schedule sizes the *memory* demo, not a
compute-optimal recipe.

| | value |
|---|---|
| tokens / optimizer step | `batch 1 × accum 8 × seq 2048` = **16,384** |
| total steps | 150,000 |
| **total token budget** | **~2.46B** |
| Chinchilla-ish compute-optimal for 7.2B params (~20 tok/param) | ~144B |
| **fraction of compute-optimal** | **~1.7%** (about 59x short) |

At this budget a 7B model is far into the under-trained regime — a 1B
model on the same token count would be strictly better. The schedule
exists so the FASE plan can be inspected and (on ≥80 GB hardware)
executed at 7B tensor widths; it is not a recipe for a usable 7B model.
Uniform-baseline loss over the 49,152-token vocab is
`ln(49152) = 10.80` nats; treat monotone descent and stable VRAM as the
signal, not the absolute value.

`PRETRAIN_LR = 3e-4` at a 16,384-token effective batch is aggressive for
the same reason as `coder1b` — see that README's section for the
reference points and the two levers.

## Weight decay: parameter groups

As in `coder1b`: `AdamW(weight_decay=...)` decays every trainable parameter by
default, RMSNorm gains and the tied embedding included. Exempt them with
`no_decay=[...]` over parameter roles — `no_decay=["vector"]` for the usual
norms-and-biases convention, plus `"embedding"` to spare the tied embedding.
`"vector"` is resolved from the tensor's real rank at step time, because real
model fields have no statically-derivable rank. See
`models/coder1b/README.md` for the role table, the measured evidence, and the
compositions that refuse.

## Packed corpora

`model.nsl` exposes `forward_train_packed(input_ids, segment_ids,
position_ids, training)` for `DataLoader(..., packing=true)` streams —
document-masked attention plus per-document RoPE position reset. The
unpacked `forward_train` on a packed stream attends across document
boundaries. `pretrain_fase.nsl` stays unpacked on purpose: its corpus is a
single repeated token with no document structure.
