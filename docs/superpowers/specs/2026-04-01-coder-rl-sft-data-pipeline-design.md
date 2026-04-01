# Coder-RL SFT Data Pipeline — Phase A Design

**Date**: 2026-04-01
**Status**: Draft
**Scope**: Build the tokenizer and pre-tokenized SFT corpus for coder-rl Phase 1 (Supervised Fine-Tuning)

## 1. Goal

Produce two artifacts that enable SFT training of the NSLCoderRL model (10.1M params, 6-layer decoder-only transformer):

1. **`nsl_coder_4k.json`** — A 4096-vocab NSL-aware BPE tokenizer that efficiently encodes both NSL code (for model output) and English prose (for GRPO prompt encoding in Phase 3).
2. **`sft_tokens.bin`** — Pre-tokenized pure NSL code corpus in u16 binary format, containing only compiler-ready NeuralScript with zero English contamination.

The key design principle is **input/output asymmetry**: the tokenizer vocabulary must handle both sides of the prompt-completion boundary (English instructions in, NSL code out), but the SFT training corpus must contain only code output. This prevents the model from learning to generate English prose while ensuring it can efficiently parse English instructions during the later GRPO phase.

## 2. Tokenizer Design

### 2.1 Vocabulary Structure (4096 tokens)

| Range | Count | Category | Description |
|-------|-------|----------|-------------|
| 0-9 | 10 | Special | PAD, BOS, EOS, CODE_START, CODE_END, ERROR_START, ERROR_END, FIX_START, TASK_START, TASK_END |
| 10-59 | 50 | Keywords | All NSL language keywords as single tokens: `fn`, `let`, `const`, `model`, `train`, `grad`, `if`, `else`, `for`, `while`, `return`, `from`, `import`, `as`, `in`, `not`, `and`, `or`, `true`, `false`, `none`, `class`, `struct`, `enum`, `match`, `break`, `continue`, `pass`, `with`, `assert`, `raise`, `try`, `except`, `finally`, `yield`, `async`, `await`, `type`, `where`, `pub`, `mut`, `ref`, `self`, `super`, `is`, `lambda`, `del`, `global`, `nonlocal` |
| 60-139 | 80 | Builtins | NSL built-in functions and types: `Tensor`, `matmul`, `softmax`, `cross_entropy`, `relu`, `silu`, `gelu`, `sigmoid`, `tanh`, `dropout`, `layernorm`, `rmsnorm`, `embedding_lookup`, `AdamW`, `SGD`, `DataLoader`, `load_mmap`, `model_save`, `model_load`, `print`, `randn`, `zeros`, `ones`, `full`, `arange`, `reshape`, `transpose`, `sum`, `mean`, `abs`, `sqrt`, `exp`, `log`, `clamp`, `gather`, `scatter_add`, `tensor_cat`, `tensor_cos`, `tensor_sin`, `rotate_half`, `scaled_dot_product_attention`, `RMSNorm`, `LayerNorm`, `GroupedQueryAttention`, `RotaryEmbedding`, `Linear`, `Embedding`, `MLP`, `warmup_cosine`, `int`, `float`, `bool`, `str`, `len`, `range`, `enumerate`, `zip`, `map`, `filter`, `min`, `max`, `sorted`, `reversed`, `list`, `dict`, `set`, `tuple`, `type`, `isinstance`, `hasattr`, `getattr`, `setattr` |
| 140-169 | 30 | Operators | `+`, `-`, `*`, `/`, `//`, `%`, `**`, `@`, `\|>`, `=`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `+=`, `-=`, `*=`, `/=`, `->`, `:`, `.`, `,`, `(`, `)`, `[`, `]`, `{`, `}` |
| 170-179 | 10 | Structure | INDENT (4-space), DEDENT, NEWLINE, TAB, DOUBLE_NEWLINE, COLON_NEWLINE, COMMENT_START (`#`), ELLIPSIS (`...`), UNDERSCORE (`_`), HASH_BANG |
| 180-211 | 32 | Numbers | Single digits `0`-`9`, common literals: `0.0`, `1.0`, `0.1`, `0.01`, `0.001`, `0.0001`, `1e-5`, `1e-6`, `2.0`, `-1`, `0.5`, `0.9`, `0.95`, `0.99`, `16`, `32`, `64`, `128`, `256`, `512`, `1024`, `2048` |
| 212-399 | 188 | Identifiers | Common variable names: `x`, `y`, `z`, `w`, `b`, `h`, `q`, `k`, `v`, `m`, `n`, `i`, `j`, `self`, `loss`, `logits`, `output`, `input`, `hidden`, `weight`, `bias`, `grad`, `param`, `batch`, `seq`, `dim`, `embed`, `heads`, `layers`, `tokens`, `labels`, `mask`, `scale`, `eps`, `lr`, `beta1`, `beta2`, `step`, `epoch`, `batch_size`, `seq_len`, `d_model`, `d_ff`, `n_heads`, `n_kv_heads`, `n_layers`, `vocab_size`, `max_seq_len`, `dropout_p`, `weight_decay`, `head_dim`, `training`, `forward`, `backward`, `shape`, `ndim`, `item`, `contiguous`, `clone`, `to`, `cuda`, `cpu`, etc. |
| 400-4095 | 3696 | BPE | Learned byte-pair merges from the full corpus (NSL code + English spec prose) |

### 2.2 Training Corpus

The BPE trainer processes **all four sources** to learn merges for both NSL and English:

- `stdlib/nsl/` — all `.nsl` files
- `examples/` — all `.nsl` files
- `models/` — all `.nsl` files
- `spec/` — all `.nsl.md` files (raw markdown including English prose)

This ensures the 3696 BPE slots learn common English programming words ("function", "tensor", "parameter", "compile", etc.) alongside NSL-specific subword patterns. Without this, English GRPO prompts would be shredded into individual characters, wasting context window on the 12M model.

### 2.3 Pre-Tokenizer: ByteLevel (Critical)

The tokenizer **must** use a `ByteLevel` pre-tokenizer (GPT-2/RoBERTa style), not a whitespace splitter. This maps all raw bytes — including spaces, tabs, and newlines — to visible unicode characters (e.g., `Ġ` for space, `Ċ` for newline).

**Why this is critical**: NSL uses indentation-based syntax (like Python). The default HuggingFace whitespace pre-tokenizer would strip or mangle consecutive spaces and newlines before they reach the BPE logic, destroying the indentation structure that defines NSL's block semantics. ByteLevel encoding preserves every whitespace byte as a visible character that BPE can learn to merge.

The reserved structure tokens (170-179) serve as post-processing markers for canonical patterns. For example, the ByteLevel encoding of 4 consecutive spaces can be mapped to the INDENT token during a normalization pass after BPE training.

### 2.4 Implementation

**File**: `models/coder-rl/data/build_tokenizer.py`

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from pathlib import Path

# 1. Define fixed vocabulary (tokens 0-399)
special_tokens = [...]  # PAD, BOS, EOS, keywords, builtins, operators, etc.

# 2. Initialize BPE model with ByteLevel pre-tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# 3. Collect training corpus (all 4 sources)
corpus_files = collect_all_files(["stdlib/", "examples/", "models/", "spec/"])

# 4. Train BPE on remaining slots (400-4095)
trainer = trainers.BpeTrainer(
    vocab_size=4096,
    special_tokens=special_tokens,
    min_frequency=2,
)
tokenizer.train(corpus_files, trainer)

# 5. Save
tokenizer.save("tokenizer/nsl_coder_4k.json")
```

### 2.5 Validation

- **Round-trip test**: Encode then decode every `.nsl` file in the corpus. Verify exact byte-level reconstruction.
- **Keyword atomicity**: Verify every token in ranges 10-169 encodes as exactly one token ID.
- **Indentation preservation**: Encode a multi-level indented NSL file, decode it, diff against original. Zero differences.
- **English efficiency**: Encode a sample English prompt ("Write a SwiGLU feedforward block with dropout"). Verify it uses fewer than 2x the tokens a dedicated English tokenizer would use.

## 3. SFT Corpus Design

### 3.1 Sources (Code Only)

| Source | Method | Description |
|--------|--------|-------------|
| `stdlib/nsl/` | Direct inclusion | All `.nsl` files, unmodified |
| `examples/` | Direct inclusion | All `.nsl` files, unmodified |
| `models/` | Direct inclusion | All `.nsl` files, unmodified |
| `spec/` | Regex extraction | NSL code blocks only, no English prose |

### 3.2 Spec Code Extraction

Extract NSL code blocks from markdown spec files using a forgiving regex:

```python
import re

def extract_nsl_blocks(markdown_text: str) -> str:
    # Primary: fenced ```nsl blocks (forgiving whitespace boundaries)
    blocks = re.findall(r'```nsl\s*(.*?)\s*```', markdown_text, re.DOTALL)
    
    # Fallback: unfenced ``` blocks containing NSL keywords
    if not blocks:
        generic = re.findall(r'```\s*(.*?)\s*```', markdown_text, re.DOTALL)
        nsl_keywords = {'fn ', 'let ', 'const ', 'model ', 'train(', 'from '}
        blocks = [b for b in generic if any(kw in b for kw in nsl_keywords)]
    
    return "\n\n".join(blocks)
```

The forgiving `\s*` boundaries (instead of strict `\n`) handle inconsistent markdown formatting — missing trailing newlines before closing backticks, extra whitespace, etc. This prevents silently skipping code blocks from the spec.

### 3.3 Document Packing Strategy

**Phase A (immediate)**: Pad each document to `seq_len` boundary with PAD tokens. Each document is an independent training sequence with no cross-document attention leakage.

At ~40-48K tokens across ~130 files, with seq_len=2048, padding waste is negligible (~20-25 sequences total). This requires zero changes to NSL's existing `DataLoader`.

**Phase B (future, when corpus grows to 238K+ tokens)**: Output `sft_offsets.bin` alongside `sft_tokens.bin` — a u32 array marking the token index where each document starts. Pass this offset array into NSL's FlashAttention kernel (M27) to isolate documents in SRAM with zero padding and zero attention leakage. This becomes important when packing many short documents into long sequences.

### 3.4 Padding Loss Isolation (Critical)

When padding documents to `seq_len` boundaries, the model must **not** compute gradients for padding positions. Without this, the model wastes capacity learning that "the most likely token after PAD is another PAD", diluting the gradients from actual NSL syntax.

**Solution**: `prepare_sft.py` outputs **two** parallel arrays of equal length:

- **`sft_tokens.bin`** (inputs): Token IDs with PAD_ID=0 in padding positions.
- **`sft_labels.bin`** (targets): Token IDs shifted by 1, with **-100** in padding positions (not 0).

Using -100 as the ignore label matches PyTorch convention and works with NSL's existing `cross_entropy` implementation in `losses.nsl`, which already masks targets via `valid_mask = clamp(targets + 1.0, 0.0, 1.0)` — this produces 0.0 for targets of -100, zeroing out their loss and gradient contribution.

Example for a 10-token document with seq_len=16:

```text
inputs: [BOS  t1  t2  t3  t4  t5  t6  t7  t8  EOS  PAD  PAD  PAD  PAD  PAD  PAD]
labels: [ t1  t2  t3  t4  t5  t6  t7  t8  EOS -100 -100 -100 -100 -100 -100 -100]
```

Note: u16 cannot represent -100, so `sft_labels.bin` is stored as **i16** (signed). NSL's `load_mmap` with dtype=2 (i32) or a new i16 dtype handles this. Alternatively, store labels as i32 for simplicity and use dtype=2.

### 3.5 Binary Format

**File**: `sft_tokens.bin`
**Format**: Flat array of u16 token IDs (inputs). Each document padded to `seq_len` boundary with PAD_ID=0. Total file = `num_sequences * seq_len * 2` bytes.

**File**: `sft_labels.bin`
**Format**: Flat array of i32 token IDs (labels). Each document's labels are the input shifted by 1, with -100 in padding positions. Total file = `num_sequences * seq_len * 4` bytes.

**Loaded in NSL as**:
```
let inputs = load_mmap("data/sft_tokens.bin", 3)   # dtype=3 = u16
let labels = load_mmap("data/sft_labels.bin", 2)    # dtype=2 = i32
```

### 3.5 Implementation

**File**: `models/coder-rl/data/prepare_sft.py`

```python
from tokenizers import Tokenizer
from pathlib import Path
import numpy as np

SEQ_LEN = 2048
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

# 1. Load trained tokenizer
tok = Tokenizer.from_file("tokenizer/nsl_coder_4k.json")

# 2. Collect code-only corpus
nsl_files = collect_nsl_files(["stdlib/", "examples/", "models/"])
spec_code = extract_spec_code_blocks("spec/")

# 3. Encode each document, build input/label pairs with padding
all_inputs = []
all_labels = []
LABEL_IGNORE = -100

for text in all_documents:
    ids = [BOS_ID] + tok.encode(text).ids + [EOS_ID]
    # Labels are inputs shifted by 1
    inp = ids[:-1]
    lbl = ids[1:]
    # Pad to next seq_len boundary
    padded_len = ((len(inp) + SEQ_LEN - 1) // SEQ_LEN) * SEQ_LEN
    pad_count = padded_len - len(inp)
    inp += [PAD_ID] * pad_count
    lbl += [LABEL_IGNORE] * pad_count
    all_inputs.extend(inp)
    all_labels.extend(lbl)

# 4. Write as binary
np.array(all_inputs, dtype=np.uint16).tofile("sft_tokens.bin")
np.array(all_labels, dtype=np.int32).tofile("sft_labels.bin")

n_seq = len(all_inputs) // SEQ_LEN
print(f"Documents: {len(all_documents)}")
print(f"Total tokens: {len(all_inputs)} ({n_seq} sequences of {SEQ_LEN})")
print(f"sft_tokens.bin: {len(all_inputs) * 2} bytes")
print(f"sft_labels.bin: {len(all_labels) * 4} bytes")
```

### 3.6 Validation

- Verify `sft_tokens.bin` contains zero English prose (spot-check by decoding random windows)
- Verify BOS/EOS framing: every document starts with BOS (token 1) and ends with EOS (token 2)
- Verify PAD tokens only appear between EOS and next BOS (never mid-document)
- Load in NSL via `load_mmap` and verify shape/dtype
- Run 1-epoch SFT training with `--source-ad` flag to validate end-to-end pipeline

## 4. File Layout

```
models/coder-rl/
├── data/
│   ├── build_tokenizer.py      # Step 1: Train BPE tokenizer
│   ├── prepare_sft.py          # Step 2: Build SFT corpus
│   ├── tokenizer/
│   │   └── nsl_coder_4k.json   # Trained tokenizer (generated)
│   ├── sft_tokens.bin          # Pre-tokenized SFT inputs, u16 (generated)
│   ├── sft_labels.bin          # Shifted labels with -100 padding, i32 (generated)
│   └── sft_offsets.bin         # Document boundaries (Phase B, future)
├── model.nsl                   # NSLCoderRL (VOCAB_SIZE=4096, unchanged)
├── train_grpo.nsl              # Training script (update data path)
└── ...
```

## 5. Training Script Change

One line change in `train_grpo.nsl`:
```
# Before:
let sft_tokens = load_mmap("data/sft_tokens.bin", 3)

# After (path is already correct, just need the file to exist):
let sft_tokens = load_mmap("data/sft_tokens.bin", 3)
```

No change needed — the path in `train_grpo.nsl` already points to `data/sft_tokens.bin`.

## 6. Model Compatibility

`NSLCoderRL` already uses `VOCAB_SIZE = 4096`. The embedding matrix is `[4096, 384]` = 1.57M parameters, which is ~15% of the total 10.1M model — a healthy ratio for a model this size. No model changes required.

## 7. Success Criteria

1. `nsl_coder_4k.json` passes all validation checks (round-trip, keyword atomicity, indentation preservation)
2. `sft_tokens.bin` loads via `load_mmap` and contains only valid NSL code
3. `nsl run --source-ad models/coder-rl/train_grpo.nsl` completes at least 1 epoch without errors
4. Loss decreases over the first epoch (model is learning, not diverging)

## 8. Phase B Preview (Not in Scope)

Phase B adds synthetic task-code pairs to the corpus:
- English task descriptions paired with NSL code completions
- Augmentation: variable renaming, dimension shuffling, optimizer swapping
- `sft_offsets.bin` for document-aware attention masking
- Estimated corpus growth: 40K tokens -> 238K+ tokens

Phase B is a separate spec after Phase A validates end-to-end.
