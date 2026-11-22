# NSL-Coder-500M — FASE peak-memory demo

> **Read this first (2026-11-12).** Every measurement in this file and in the
> `models/benchmarks/` records it cites was taken before the attention
> backward's gradient underflow was fixed (see `CHANGELOG.md`, Unreleased →
> Fixed). That defect made the query/key projection gradients 12% wrong at
> Coder-50M and 80% wrong at Coder-1B with the production flags, so the
> learning-rate conclusions here — including the shipped `PRETRAIN_LR` —
> describe the broken toolchain, not the model.
>
> Measured after the fix, same weights, corpus and batch order
> (`tools/refcheck/README.md` has the method):
>
> | Coder-500M, lr 3e-4, one epoch of 8.2M tokens | held-out loss |
> | --- | --- |
> | toolchain before the fix | 8.27 (the corpus's unigram entropy is 8.18) |
> | toolchain after the fix | 6.64 |
> | PyTorch twin | 6.63 |
>
> The recipe's numbers are unchanged; they need re-deriving with a pilot run
> on the fixed toolchain before a long run is committed to them.

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
- **`pretrain_prod.nsl` — the production recipe (roadmap item 9).** Real
  corpus, derived schedule, checkpoint/resume, and a held-out validation
  pass. Start here for actual training.
- `pretrain_fase.nsl` — runnable FASE demo using a `train(...)` block with
  `grad_accumulation=8`, AdamW, `grad_clip=1.0`. Synthetic tokens (no
  data pipeline needed).
- `pretrain_cert.nsl`, `pretrain_srbf16_cert.nsl` — certification arms driven
  by harnesses, not standalone recipes.

## Production pretraining

**Toolchain.** This directory is pinned to the `0.11` toolchain channel
(`nsl-toolchain.toml`, NSL V2 plan 0.1): production runs stay on a release
toolchain while `main` moves. `nsl run` / `nsl build` on a file here hands the
invocation over to `~/.nsl/toolchains/0.11/bin/nsl` (install it once with
`scripts/install-toolchain.sh 0.11 origin/release/0.11`); a toolchain of
another channel refuses when that is not installed. `0.11` is the first
channel with bf16 activation storage, which `pretrain_prod.nsl` declares
(`precision = "bf16"`: 1.17x tok/s over the bf16 GEMM mode and 0.94 GiB
less peak at this size; held-out loss over 5 seeds no worse than TF32's,
`models/benchmarks/BF16_STORAGE_CONVERGENCE_2026_11.md`).
`--ignore-toolchain-pin` runs on the invoked toolchain instead — for gates
and experiments, not production runs.

```bash
python models/benchmarks/make_prod_split.py          # once, from the repo root
cd models/coder500m
nsl run --source-ad --checkpoint-blocks --checkpoint-selective \
    --fuse-rmsnorm-backward --fuse-wgrad-accum pretrain_prod.nsl
```

**The run line is the 1B recipe's** (since 2026-11-22). Until then it was
`--source-ad --checkpoint-blocks`, which runs at half the speed. Measured on
this directory's `0.11` toolchain (`nsl build` here handed over to
`~/.nsl/toolchains/0.11`, release/0.11 at 6445d66a) with the recipe itself —
batch 2 × 1024, accumulation 8, dropout 0.1; only its corpus paths and its
`on_step` callback changed, to stop at micro-step 480 (and, in the runs the
losses below come from, to print every step) — 3 interleaved rounds, 40
updates timed after the first 20:

| line | tok/s | allocator peak | reserved | driver peak |
| --- | --- | --- | --- | --- |
| `--source-ad --checkpoint-blocks` (until 2026-11-22) | 11,340 | 11.28 GiB | 12.11 GiB | 12.50 GiB |
| this line | 22,526 (1.986x paired, 1.975-1.989) | 12.88 GiB | 13.84 GiB | 14.22 GiB |
| this line without `--checkpoint-blocks --checkpoint-selective` | 24,287 (1.077x this line) | 19.99 GiB | 20.78 GiB | 21.16 GiB |

The corpus was a same-size slice of another corpus with the same tokenizer
(`data/tokens/train_new.bin` was not on the measuring machine); the step does
the same work whatever the ids are. The persistent state is 8.26 GiB of every
row (weights 2.07 + optim m, v and m_partial 2.06 each).

- **`--checkpoint-blocks` is no longer required on a 32 GB card.** It was in
  2026-08, when activations were stored f32: the run died in the first
  backward with 168 MB free of 31.39 GB. With bf16 activation storage
  (`precision = "bf16"`, 0.11) the run fits without any checkpointing (last
  row). It stays in the line because the 7.1 GiB that dropping it costs buy
  only 1.077x.
- **Why selective checkpointing fits.** Full block recompute replays every
  block interior in the backward, GEMMs included; `--checkpoint-selective`
  keeps the matmul-class outputs and replays only norms, RoPE and elementwise
  ops. Stored bf16, what it keeps puts the activation surface at the peak at
  4.62 GiB, against 3.50 GiB for full recompute with the same fusions and
  8.34 GB for full recompute with f32 storage in 2026-08.
- **Losses.** `--fuse-wgrad-accum` and `--fuse-rmsnorm-backward` change the
  summation order of the weight and RMSNorm-gain gradients, so the losses are
  not bit-equal to the old line's. Over the first 480 micro-steps (warmup and
  80 more) the two lines' per-step losses differ by at most 0.71% (mean 0.03%),
  while two runs of the old line differ by up to 0.68% and two of the new one
  by up to 0.38%: neither line is bit-reproducible run to run (timed GEMM
  selection, the atomic embedding backward). Both descend from 11.05 (mean of
  the first 8 micro-steps) to 7.84 (mean of the last 8), with no non-finite
  loss.
- **Dropout is not re-drawn.** Replaying a dropout in the backward would draw a
  different mask. Neither recompute policy replays a draw: on 0.11 CCR
  force-saves every dropout mask and output (`crates/nsl-codegen/src/ccr.rs`;
  on `main` the unit test `dropout_is_not_recomputed_under_the_selective_policy`
  holds the selective policy to it). On 0.11, with the run-to-run
  nondeterminism pinned (`NSL_EMBEDDING_BWD_CPU=1`, `--no-bf16-lt-tune`), the
  first 48 micro-step losses of this line are bit-identical to the same flags
  with full block recompute and with no checkpointing at all, a backward with
  nothing to recompute. The step makes 48 dropout draws (24 blocks × attention
  and MLP outputs, p = 0.1), so the comparison is not vacuous.
- **Resume.** The two fusions are arithmetic keys of the execution
  fingerprint: a checkpoint written under the old line is refused under this
  one. Finish such a run with the line it started on.

The learning rate is **measured, not inherited**: `lr = 1.5e-4` with 400
micro-steps of warmup, against the `3e-4 / 200` this repo carries at every
model size. At 500M the inherited pair produced an erratic epoch with
excursions to 10.94; the measured pair descends monotonically and lands 0.57
nats lower on held-out loss.

Two things this recipe deliberately does not hide:

- **Token budget.** 8.39M train tokens against ~505M parameters is ~0.017
  tokens/param, a factor of ~1,200 under a Chinchilla-ish ~20 tokens/param
  (item 10 corrected "roughly two orders of magnitude", which understated it
  by more than 10x). This
  validates the *workflow* at 500M; it does not produce a good 500M model.
  A bigger corpus is roadmap item 15.
- **The held-out tail.** Training reads a prefix slice; `VAL_LOSS` is scored
  on the last 524,288 tokens, with 13,020 tokens between them that neither
  side reads. They are separate files, so disjointness is automatic; the gap
  keeps the held-out set from starting in the immediate continuation of the
  last file trained on. At this token budget the train/val gap is the
  measurement that matters, which is why a training loss alone would be
  misleading.

`crates/nsl-cli/tests/pretrain_prod_agreement_gate.rs` re-derives the schedule
from the corpus arithmetic for both this model and coder50m, so the two files
cannot drift — or agree on a shared fiction, which is how the previous
`total_steps = 305000` survived.

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
Training blocks found: 1

[Block 1]
  Kind: train
  Model: m

  FASE (Fused Accumulation-Step Elimination):
    grad_accumulation: 8
    optimizer:         AdamW
    mode:              Deferred
    rationale:         AdamW supports deferred first-moment accumulation with batch-variance v approximation
    backward_phases:   AccumulateOnly x 7, FinalTwoPhase
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
- `MAX_SEQ_LEN` / `ROPE_THETA` are threaded from `config.nsl` into
  `GroupedQueryAttention` -> `RotaryEmbedding` (they were hardcoded in the
  stdlib attention layer before). This model's values, 1024 / 10000, match
  what it effectively used, so its numerics are unchanged by that fix —
  unlike `coder1b` / `coder7b`, whose declared `ROPE_THETA = 500000` was
  dead. `crates/nsl-cli/tests/model_config_drift.rs` gates the agreement.
- `AdamW(weight_decay=...)` decays **every** trainable parameter by default,
  RMSNorm gains and the tied embedding included. Pass
  `no_decay=["vector"]` to exempt norms/biases (add `"embedding"` for the
  tied embedding); `"vector"` is resolved from the runtime tensor rank. See
  `models/coder1b/README.md` for the role table and the refused
  compositions.
- `model.nsl` also exposes `forward_train_packed(input_ids, segment_ids,
  position_ids, training)` for `DataLoader(..., packing=true)` streams;
  `pretrain_fase.nsl` uses the unpacked path because its synthetic corpus
  has no document boundaries.

## Building bigger variants

This fixture is the template for larger scales. For 7B:

- d_model ≈ 4096, n_layers ≈ 32, n_heads ≈ 32, d_ff ≈ 11008, vocab ≈ 32000
- Same `train(...)` block structure; adjust batch / grad_accum so a
  single gradient fits in VRAM with optimizer state + activations.
