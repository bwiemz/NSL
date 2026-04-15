# NSL-Coder-500M — FASE peak-memory demo

Scaled-up sibling of `coder50m`. Same GQA + SwiGLU + RoPE transformer
architecture, ~10× the parameter count, designed as a realistic stress
test for the FASE roadmap shipped in items #1-#6 of
`feat/fase-deferred`.

## Architecture

| | coder50m | coder500m |
|---|---|---|
| d_model | 512 | **1280** |
| n_layers | 8 | **24** |
| n_heads | 8 | **20** |
| n_kv_heads | 4 | **10** |
| head_dim | 64 | 64 |
| d_ff | 1408 | **3520** |
| vocab | 49152 | 49152 |
| params (f32) | 48.8M | **~505M** |

## Files

- `config.nsl` — hyperparameters (architecture + pretrain + finetune).
- `model.nsl` — `NSLCoder`, `TransformerBlock`, `SwiGLUFFN` definitions.
- `pretrain_fase.nsl` — runnable FASE demo using a `train(...)` block with
  `grad_accumulation=8`, AdamW, `grad_clip=1.0`. Synthetic tokens (no
  data pipeline needed).

## Running the FASE demo

Build `nsl` once:

```bash
cargo build --release --bin nsl
```

Baseline (tape AD — no FASE hook, all gradients materialize simultaneously
during backward):

```bash
./target/release/nsl run models/coder500m/pretrain_fase.nsl
```

Source-AD (FASE Deferred mode activates, the consume-per-param hook frees
each gradient immediately after accumulation — one gradient live at a
time during backward):

```bash
./target/release/nsl run --source-ad models/coder500m/pretrain_fase.nsl
```

## Measuring peak VRAM

Option A — `nsl run --profile-memory` (existing GPU memory profiler from
M36 groundwork, writes `memory_profile.json` on exit):

```bash
./target/release/nsl run --profile-memory --source-ad models/coder500m/pretrain_fase.nsl
```

Option B — external `nvidia-smi` watcher in a second terminal:

```bash
# terminal 2: poll peak usage during the run
nvidia-smi dmon -s m -c 120
```

Option C — `nsl check --training-report` to see the compiler's FASE plan
before running anything:

```bash
./target/release/nsl check --training-report models/coder500m/pretrain_fase.nsl
```

Expected report shape:

```
=== Training Pipeline Report ===
File: models/coder500m/pretrain_fase.nsl
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
    ...
```

## Expected results

With source-AD and FASE Deferred active, the hook frees each parameter
gradient immediately after `m_partial += (1/N)·g` runs. Peak gradient
memory drops from `N_params × param_size` (all gradients simultaneously)
to `~1 × largest_param_size` (one gradient at a time).

Rough numerical expectations (f32, RTX 5070 Ti):

| | tape AD | source-AD + FASE |
|---|---|---|
| Parameters | ~2 GB | ~2 GB |
| Optimizer state (m, v, m_partial) | ~6 GB | ~6 GB |
| Gradients (peak) | ~2 GB (all live) | ~250 MB (largest single grad: `49152×1280 f32` tied LM head) |
| Activations | ~0.5 GB | ~0.5 GB |
| **Total peak VRAM** | ~10.5 GB | ~8.75 GB |

Actual numbers depend on batch/seq sizes, CUDA context overhead, and
cuDNN workspace allocations. The **delta** is the load-bearing
measurement, not absolute values.

## Notes

- FASE Deferred's peak-memory win requires `--source-ad`. Tape-AD path
  still produces correct numerics but materializes all gradients at once
  (documented limitation from item #4's design).
- `grad_clip=1.0` in the fixture exercises item #3's two-phase clip
  codegen on the final micro-batch of each accumulation window.
- Item #5's `test-hooks`-gated CPU peak-tracking validated this
  mechanism on a 4-parameter fixture (794 KB delta). This 500M demo
  scales it to a real model on GPU.

## Building bigger variants

This fixture is the template for larger scales. For 7B:

- d_model ≈ 4096, n_layers ≈ 32, n_heads ≈ 32, d_ff ≈ 11008, vocab ≈ 32000
- Same `train(...)` block structure; adjust batch / grad_accum so a
  single gradient fits in VRAM with optimizer state + activations.
