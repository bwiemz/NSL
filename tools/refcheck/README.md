# refcheck — NSL against an independent implementation

Every other gate in this repository compares NSL with NSL: tape AD against
source AD, the PTX interpreter against silicon, a kernel against an oracle that
rounds the way the kernel rounds. Those catch a path that disagrees with its
twin. They cannot catch what both sides share.

`refcheck` compares a training step with PyTorch instead. `ref.py` is a PyTorch
twin of the `NSLCoder` decoder that loads an `.nslm` file, so both sides start
from the same weights and read the same batches; what differs afterwards is
NSL's doing.

It exists because of what it found the first time it was run (2026-11-12):

| measurement | before | after |
| --- | --- | --- |
| gradient of the query/key projections, Coder-1B, production flags | 80% wrong in the top block | 0.2% |
| gradient of every other parameter, Coder-1B | 6-12% wrong in the lower blocks | 0.1% |
| learning rate applied at steps 0, 1, 2 of an 8-step warmup to 1e-3 | 1e-3, 0, 1.25e-4 | 0, 1.25e-4, 2.5e-4 |
| three accumulated AdamW steps with clip, decay and warmup | 31% from the reference | 0.6% |

The first two were the flash-attention backward converting gradients to f16
without scaling them; the last two were the schedule being evaluated after the
optimizer step. The 500M and 1B runs that would not descend were the first.

## Setup

A CUDA build of `nsl`, and a Python environment with `torch`, `numpy` and
`tokenizers`:

```bash
cargo build --release --bin nsl --features cuda
python -m venv .venv && .venv/bin/pip install torch numpy tokenizers
.venv/bin/python tools/refcheck/make_corpus.py refcheck-work/data
```

`make_corpus.py` encodes the repository's own tracked text with the shipped v1
tokenizer, so the checks have tokens with real structure on a fresh checkout.
Any u16 token stream works in its place.

## The checks

Each takes `--nsl <binary>` and passes everything after `--` to `nsl run`, so a
recipe's flag set is checked as the recipe runs it. Each exits non-zero on
failure.

**Gradients.** One SGD step at lr 1 leaves `init - after` equal to the gradient.

```bash
tools/refcheck/grad_check.py --size 1b --tokens refcheck-work/data/train.bin \
    --nsl target/release/nsl -- --source-ad --checkpoint-blocks --fuse-rmsnorm-backward
```

Expect a worst relative error near 2e-3 on the default TF32 matmuls, 1e-3 with
`--matmul-mode f32`, and 1e-6 on `--tape-ad --matmul-mode f32`. The floor on the
source-AD path is the attention kernels' f16 operands. Coder-1B at batch 1 needs
about 21 GB.

**Optimizer stack.** AdamW with any of accumulation, clipping, weight decay and
a warmup schedule, over three optimizer steps:

```bash
tools/refcheck/optimizer_check.py --tokens refcheck-work/data/train.bin \
    --nsl target/release/nsl --accum 4 --clip 1.0 --wd 0.1 --schedule -- --source-ad
```

Expect the update to agree to about 1e-2 on source AD and 1e-4 on tape AD.

**Applied learning rate.** SGD's update is linear in the per-micro-batch
gradients, so a least-squares fit against the reference gradients recovers the
rate NSL applied to each micro-batch:

```bash
tools/refcheck/optimizer_check.py --tokens refcheck-work/data/train.bin \
    --nsl target/release/nsl --applied-lr --accum 4 -- --source-ad
```

**Loss curves.** `train_ref.py` is the PyTorch side of a full run, for when a
recipe's curve is in question: same weights, same batch order, same optimizer
arithmetic. Run the recipe with `model_save(m, "init.nslm")` before the train
block and `shuffle=false`, then:

```bash
tools/refcheck/train_ref.py --size 500m --init init.nslm \
    --tokens train.bin --val val.bin --batch 2 --accum 8 --micro-steps 4096 \
    --lr 3e-4 --warmup 200 --total 4096 --min-lr 3e-5 --wd 0.1 --clip 1.0 --tf32
```

The two curves track each other for the first few hundred steps and then
separate the way two seeds do; compare held-out loss, not individual points.

## What it does not cover

Dropout (the two frameworks cannot share a mask; every check runs with it off),
packed batches, optimizers other than AdamW and SGD, and any architecture other
than `NSLCoder`. A model that changes shape needs its twin in `ref.py`.
