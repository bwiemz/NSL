# NSL-Coder-50M

A 50M-parameter code language model defined, trained, and served entirely in NSL.
LLaMA-style architecture: GQA, RoPE, SwiGLU, RMSNorm.

## Quick Start

### 1. Prepare NSL training data

```bash
cd models/coder50m/data
python prepare_nsl.py path/to/codeforge.json path/to/tokens.bin
```

### 2. Pretrain (Stage 1: 10B StarCoder tokens)

```bash
nsl run models/coder50m/pretrain.nsl --source-ad
```

`--source-ad` is required, not optional: without it the train block fails to
compile with "train step body must assign to a variable named 'loss'". It is
also the only path with flash attention and the fused cross-entropy that
`pretrain.nsl`'s `@fused_lm_ce` decorator asks for. `--pretrain-optimized`
implies `--source-ad` and adds the WGGO/CSHA planning stack on top.

Edit the `load_mmap` path at the top of the script to point at the
`tokens.bin` step 1 produced.

### 3. Finetune (Stage 2: NSL + general code mix)

```bash
nsl run models/coder50m/finetune.nsl --source-ad
```

### 4. Generate code interactively

```bash
nsl run models/coder50m/generate.nsl
```

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | ~48.8M |
| Vocab | 49,152 (BPE) |
| Hidden dim | 512 |
| Layers | 8 |
| Q heads | 8 |
| KV heads | 4 (GQA) |
| FFN dim | 1408 (SwiGLU) |
| Context | 1024 |
| Precision | f32 |

## Files

- `config.nsl` -- All hyperparameters
- `model.nsl` -- LLaMA-style transformer architecture
- `pretrain.nsl` -- Stage 1 training on StarCoder data
- `finetune.nsl` -- Stage 2 finetuning on NSL code
- `generate.nsl` -- Interactive code generation
- `data/prepare_nsl.py` -- Tokenize NSL source files
