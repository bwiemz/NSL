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
| RoPE theta | 10000 | **500000** | 500000 |
| params (f32) | ~505M | **~1.07B** | ~7.2B |

`max_seq_len` and `RoPE theta` are threaded from these constants into
`GroupedQueryAttention` -> `RotaryEmbedding`. They used to be hardcoded in
the stdlib attention layer as `(1024, 10000.0)`, which silently overrode
both — so this table's `500000` was aspirational, not what the model
trained with. `crates/nsl-cli/tests/model_config_drift.rs` now fails if
`config.nsl`, `model.nsl`'s const block, and the literals passed to
`TransformerBlock(...)` ever disagree again.

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

## Reading the cert curves: this is a compute-limited budget

`config.nsl`'s pretrain schedule is sized to one 16 GB consumer GPU, not
to a compute-optimal recipe. Read any loss curve from it with that in
mind.

| | value |
|---|---|
| tokens / optimizer step | `batch 2 × accum 8 × seq 2048` = **32,768** |
| total steps | 100,000 |
| **total token budget** | **~3.28B** |
| Chinchilla-ish compute-optimal for 1.07B params (~20 tok/param) | ~21.4B |
| **fraction of compute-optimal** | **~15%** (about 6.5x short) |

Consequences worth stating up front, so nobody reads a cert curve as a
quality claim:

- **The run does not converge.** Loss should still be falling when the
  schedule ends. A final loss from this budget is not comparable to a
  published 1B-model number — those are trained on 1-3T tokens, i.e.
  300-1000x this budget.
- **The reference points are the entropy bounds, not a target loss.**
  Uniform over the 49,152-token vocab is `ln(49152) = 10.80` nats. A
  healthy run leaves that within the first few hundred steps (the model
  learns the unigram distribution), then descends much more slowly. The
  useful signal from these demos is *monotone descent and stable VRAM*,
  not the absolute value.
- **`PRETRAIN_LR = 3e-4` is aggressive for a 32K-token effective batch.**
  For scale: GPT-3 1.3B used 2e-4 at ~1M-token batches, Llama-1 7B used
  3e-4 at ~4M-token batches. At 32K tokens per step this LR is roughly an
  order of magnitude high relative to those references. If a real run
  shows loss spikes, the two levers are raising `PRETRAIN_GRAD_ACCUM`
  (bigger effective batch, same VRAM) or lowering `PRETRAIN_LR` — prefer
  the former, since the budget is already batch-starved.

Note the `pretrain_fase.nsl` demo itself runs at `seq_len=512`, not 2048,
so its per-step token count is 8,192. It exists to exercise the FASE
memory path for ~10 steps, not to train a model.

## Weight decay applies to every parameter

`AdamW(weight_decay=0.1)` in a `train` block is a single scalar applied to
**all** trainable parameters — including RMSNorm gains and the tied
embedding / LM head. NSL has no parameter-group mechanism in the train-block
optimizer DSL today, so the usual convention (decay 2-D weights only,
exempt norms/biases/embeddings) cannot be expressed.

This is nonstandard and worth knowing when comparing against a PyTorch
baseline that does exempt them. The classification needed to fix it already
exists — `crates/nsl-codegen/src/muon_roles.rs` labels every parameter
`embedding` / `head` / `vector` (rank != 2) / `hidden`, via the
`@param_role(...)` decorator and `embedding_lookup`-usage inference, and the
mixed Muon/AdamW router already consumes it. Wiring a decay exemption to
those same roles is a contained change; it is **not** implemented.

Until then, `weight_decay=0.0` is the option that avoids decaying norms,
at the cost of not regularizing the projections either.

## Packed corpora

`model.nsl` exposes `forward_train_packed(input_ids, segment_ids,
position_ids, training)` alongside `forward_train`. Use it whenever the
DataLoader is packing:

```nsl
let loader = DataLoader(tokens, batch_size=2, seq_len=2048, shuffle=false,
                        drop_last=true, packing=true, pack_separator=0)
...
    step(batch):
        let logits = m.forward_train_packed(batch.input_ids, batch.segment_ids,
                                            batch.position_ids, true)
```

That routes attention through the stdlib GQA's `forward_packed`, which masks
to document boundaries and resets RoPE positions at each document start. The
unpacked `forward_train` on a packed stream would let every document attend
to the ones before it in the same row.

`pretrain_fase.nsl` deliberately stays on the unpacked path: its corpus is
`full([131072], 1.0)` — one token repeated, with no document boundaries at
all — so packing would be a no-op there. It is a memory demo, not a
pretraining recipe.
