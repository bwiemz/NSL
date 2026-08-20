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

That table is the `pretrain_fase.nsl` demo, which runs at **seq 512**. The
production recipe runs at seq 2048, where the activation surface is the
dominant term rather than a rounding error, and the numbers are measured
rather than budgeted (item 10, RTX PRO 4500 Blackwell / 32 GB):

All figures in **GiB**, because that is what both the allocator's memstats
lines (which say "MB" but divide by 1024) and its OOM dump (`format_bytes`,
likewise) actually print. Mixing the two conventions makes the same 4112 MiB
weight surface look like two different numbers:

| | bare `--source-ad --checkpoint-blocks --fuse-rmsnorm-backward` | `+ --optim-state-offload` |
|---|---|---|
| weights | 4.02 GiB | 4.02 GiB (same surface) |
| optim m / v / m_partial, on device | 4.00 / 4.00 / 4.00 GiB | **0 / 0 / 0** (host-resident) |
| persistent subtotal | **16.02 GiB** | **4.02 GiB** |
| activations, at the allocator peak | — (never reached: VRAM is gone first) | 16.54 GiB |
| **outcome** | **unusable** — degrades to CPU (`GPU OOM in <op> — falling back to CPU`, 0 steps in 900 s), then aborts at a later allocation | allocator peak **20.56 GiB**, driver **24.60–26.11 GiB** over a full epoch |

The allocator peak is the *stable* number — two full epochs of the same
program reported it byte-identically (22,073,520,640 B) while their
driver-level peaks differed by 1548 MiB. **Size a card against the driver
maximum (~26.1 GiB), not the allocator figure.** Per-flag necessity — which of
the four flags above is actually load-bearing, measured rather than asserted —
is `PROD1B_VALIDATION_2026_08_19.md` §EC6; the short version is that
`--checkpoint-blocks` is required and `--fuse-rmsnorm-backward` is worth 3 MiB.

**`--optim-state-offload` is what makes 1B@2048 fit here**, and it is not
interchangeable with the endurance benchmark's `--layerwise-accum`: that
flag refuses `grad_clip` outright. The recipe prints `PEAK_OPTIM_M` /
`PEAK_OPTIM_V` / `PEAK_M_PARTIAL` at the end of every run so a regression
to device-resident moments shows up as a number rather than as someone
else's OOM.

## Files

- `config.nsl` — hyperparameters.
- `model.nsl` — `NSLCoder`, `TransformerBlock`, `SwiGLUFFN` definitions.
- **`pretrain_prod.nsl` — the production recipe** (roadmap item 10): real
  corpus, derived schedule, checkpoint/resume, held-out validation. This is
  the one to run if you want to train the model.
- `pretrain_1b2048.nsl` — Milestone B's **certification workload**, not a
  recipe. It carries `B2048_TOKENS_PATH` / `B2048_CKPT_ARGS` marker strings
  that only `models/benchmarks/endurance_1b.py` rewrites, prints `WITNESS_*`
  blocks for that harness to assert on, runs no scheduler and reports no
  held-out loss. It cannot be run by hand. The two files are kept separate on
  purpose — a benchmark that certifies the memory stack and a recipe that
  trains a model want different things —
  and `pretrain_prod_agreement_gate.rs` holds that separation in both
  directions.
- `pretrain_fase.nsl` — runnable FASE demo with `grad_accumulation=8`,
  AdamW, `grad_clip=1.0`.

## Running

**The production recipe.** It must run from `models/coder1b/` — its corpus
paths and `from model import ...` are relative to that directory — so the `cd`
is inside a subshell, and everything after it stays repo-root-relative:

```bash
cargo build --release --bin nsl --features cuda
python models/benchmarks/make_prod_split.py      # materializes the corpus split, once

( cd models/coder1b && ../../target/release/nsl run \
    --source-ad --checkpoint-blocks --fuse-rmsnorm-backward \
    --optim-state-offload pretrain_prod.nsl )
```

The demos below are a different thing — a memory demo at seq 512, not a
recipe — and run from the repo root:

```bash
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

## Reading the curves: this is a corpus-limited budget

Until item 10 this section described a 100,000-step / ~3.28B-token
schedule. **That corpus does not exist in this repo** — the same fiction
item 4 removed at 50M and item 9 at 500M. The real budget is one epoch
over an 8.39M-token slice, and it is a good deal smaller than the old
table implied:

| | value |
|---|---|
| tokens / optimizer step | `batch 2 × accum 4 × seq 2048` = **16,384** |
| optimizer steps | 512 (2048 micro-steps ÷ accum 4) |
| **total token budget** | **8.39M** |
| Chinchilla-ish compute-optimal for 1.07B params (~20 tok/param) | ~21.4B |
| **fraction of compute-optimal** | **~0.04%** |

That 16,384-token effective batch and 512-step count are the 500M
recipe's exactly, and both sizes read the same corpus slice and the same
held-out tail. That makes the two VAL_LOSS numbers worth putting side by
side — but it is **not** a controlled experiment: the models also differ
in `DROPOUT` (0.1 at 500M, 0.0 here) and `ROPE_THETA` (10000 vs 500000),
and dropout is precisely the knob that moves a train-to-held-out gap.

Consequences worth stating up front, so nobody reads a curve as a
quality claim:

- **The run does not converge, and it is not close.** A final loss from
  this budget is not comparable to a published 1B-model number — those
  are trained on 1-3T tokens.
- **1B here is not "better than 500M".** On a fixed corpus the larger
  model gets *fewer* tokens per parameter: 0.008 against the 500M
  recipe's 0.017. Expect the held-out numbers to reflect that.
- **The reference points are the entropy bounds, not a target loss.**
  Uniform over the 49,152-token vocab is `ln(49152) = 10.80` nats. A
  healthy run leaves that within the first few hundred steps (the model
  learns the unigram distribution), then descends much more slowly. The
  useful signal is *monotone descent and stable VRAM*, not the absolute
  value.
- **`PRETRAIN_LR` has a provenance now — and it is *not* "measured at 1B".**
  This README used to argue from published references that 3e-4 was
  aggressive here, and to recommend raising `PRETRAIN_GRAD_ACCUM` in
  preference to lowering the LR. Item 9 measured the LR at 500M — halving
  it was worth 0.57 nats of held-out loss — and item 10 tried to repeat
  that at 1B rather than assume it transfers. **The 1B arms did not
  separate:** two full epochs at 3e-4 and 1.5e-4, neither descending,
  both landing just under the `ln(49152) = 10.80` uniform bound (10.284
  vs 10.525), with the ordering flipping between the early and the
  held-out horizon and no null control behind either. The shipped
  `1e-4` is therefore the *width scaling* of item 9's result
  (`1.5e-4 × 1280/2048 = 9.4e-5`), a principled default rather than a
  measurement — the budget above cannot resolve a learning rate. Arms and
  corrected statistics: `models/benchmarks/PROD1B_VALIDATION_2026_08_19.md`.
  The old recommendation to prefer a bigger batch does not survive a
  corpus this small, where accumulation buys effective batch by *spending
  optimizer steps* there are only 512 of.

Note the `pretrain_fase.nsl` demo itself runs at `seq_len=512`, not 2048,
so its per-step token count is 8,192. It exists to exercise the FASE
memory path for ~10 steps, not to train a model.

## Weight decay: parameter groups

`AdamW(weight_decay=0.1)` applies to **every** trainable parameter by default,
RMSNorm gains and the tied embedding / LM head included. The conventional
recipe decays only the weight matrices. Express that with `no_decay=[...]`,
which names parameter ROLES to exempt:

```nsl
train(model=m, epochs=1, grad_accumulation=8, grad_clip=1.0):
    # decay the projections; exempt norms and biases (the usual convention)
    optimizer: AdamW(lr=0.0001, weight_decay=0.1, beta1=0.9, beta2=0.95,
                     eps=1e-8, no_decay=["vector"])
```

| role | what it covers | resolved |
|---|---|---|
| `vector` | anything not rank-2 — RMSNorm gains, biases, scalars | at step time, from the tensor's real rank |
| `embedding` | tables used by `embedding_lookup` (so the tied LM head too) | compile time |
| `head` | an UNTIED lm_head, via `@param_role("head")` | compile time |
| `hidden` | everything else — the projections | compile time |

Add `"embedding"` to also exempt the tied embedding:
`no_decay=["vector", "embedding"]`.

**`vector` is decided at step time, not compile time, and that is
load-bearing.** A model field only gets a statically-known rank when its
initializer is a direct `zeros/ones/randn/...` call over integer literals.
Real models fail that: `RMSNorm.weight = ones([dim])` passes an identifier,
and `wq = randn([...]) * sqrt(...)` is a multiply. Measured on `coder50m` — 74
parameters classify as 1 `embedding` + 73 `hidden`, **zero** `vector`, with
both RMSNorm gains in the `hidden` bucket. A static-only "exempt rank < 2"
would have compiled, printed a plausible table, and decayed every norm anyway.

Every run prints what it exempted, and a scope that matches nothing is a
compile error rather than a silent no-op:

```
[wd-groups] weight_decay=0.1 exempting roles [vector (runtime rank != 2)] over
            74 params: 0 exempt by role at compile time, plus every param that
            is not rank-2 at step time
```

Defaults and exactness: omitting `no_decay` decays everything, exactly as
before — the emitted optimizer IR is unchanged. Exempt parameters are given
λ = 0, which routes them down each optimizer arm's existing, already-gated
no-decay branch, so a decayed parameter's arithmetic is bit-identical to a run
without the feature. Both directions are pinned by
`crates/nsl-cli/tests/weight_decay_groups_gate.rs`.

Not supported with `--muon-batch-ns`, `--layerwise-accum`,
`--optim-state-offload`, or the `@pipeline` train path — each hoists a single
weight-decay scalar out of the per-parameter loop. Those combinations refuse
loudly instead of silently decaying the parameters you asked to exempt.

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
