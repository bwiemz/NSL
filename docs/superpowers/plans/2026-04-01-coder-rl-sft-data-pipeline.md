# Coder-RL SFT Data Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 4096-vocab NSL-aware BPE tokenizer and pre-tokenized SFT corpus for coder-rl Phase 1 training.

**Architecture:** Two Python scripts under `models/coder-rl/data/`. `build_tokenizer.py` trains a BPE tokenizer with hand-crafted NSL tokens (0-399) and learned BPE merges (400-4095) on the full corpus. `prepare_sft.py` uses that tokenizer to encode code-only files into `sft_tokens.bin` (u16 inputs) and `sft_labels.bin` (i32 labels with -100 padding).

**Tech Stack:** Python 3.14, HuggingFace `tokenizers` 0.22.2, `numpy`. Invoke via `py` on Windows.

**Spec:** `docs/superpowers/specs/2026-04-01-coder-rl-sft-data-pipeline-design.md`

---

### Task 1: Create directory structure and vocabulary definition

**Files:**
- Create: `models/coder-rl/data/vocab.py`
- Create: `models/coder-rl/data/tokenizer/` (empty dir, output target)

This file defines all fixed tokens (0-399) as Python lists. Both `build_tokenizer.py` and `prepare_sft.py` import from here. Single source of truth for the vocabulary.

- [ ] **Step 1: Create `vocab.py` with all fixed token definitions**

```python
"""Fixed vocabulary for the NSL-Coder-RL 4096-token BPE tokenizer.

Tokens 0-399 are hand-assigned. Tokens 400-4095 are learned via BPE.
Both build_tokenizer.py and prepare_sft.py import from this file.
"""

# --- Special tokens (0-9) ---
SPECIAL_TOKENS = [
    "<pad>",          # 0
    "<bos>",          # 1
    "<eos>",          # 2
    "<code_start>",   # 3
    "<code_end>",     # 4
    "<error_start>",  # 5
    "<error_end>",    # 6
    "<fix_start>",    # 7
    "<task_start>",   # 8
    "<task_end>",     # 9
]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# --- NSL keywords (10-59) ---
NSL_KEYWORDS = [
    "fn", "let", "const", "model", "train", "return", "if", "else", "elif",
    "for", "while", "break", "continue", "in", "from", "import", "as",
    "true", "false", "and", "or", "not", "none", "pass",
    "grad", "kernel", "quant", "serve", "match", "case",
    "Tensor", "int", "float", "bool", "str", "list", "dict", "tuple",
    "self", "super", "class", "def", "with", "try", "except", "raise",
    "pub", "struct", "impl", "use", "mut", "unsafe",
]

# --- NSL builtins (60-139) ---
NSL_BUILTINS = [
    "randn", "zeros", "ones", "full", "arange", "linspace",
    "matmul", "softmax", "relu", "gelu", "silu", "sigmoid", "tanh",
    "exp", "log", "sqrt", "abs", "neg", "sign", "clamp",
    "sum", "mean", "reduce_max", "argmax",
    "reshape", "transpose", "squeeze", "unsqueeze", "contiguous", "expand",
    "embedding_lookup", "layernorm", "rmsnorm", "batchnorm",
    "cross_entropy", "mse_loss", "l1_loss", "bce_loss",
    "dropout", "bias_add", "gather", "scatter_add",
    "conv2d", "maxpool2d", "avgpool2d",
    "to", "cuda", "cpu", "shape", "ndim", "item",
    "print", "clock",
    "model_save", "model_load",
    "load_mmap", "DataLoader",
    "AdamW", "Adam", "SGD", "Lion",
    "warmup_cosine", "cosine_anneal", "linear_decay",
    "RMSNorm", "LayerNorm", "Linear", "Embedding",
    "GroupedQueryAttention", "SwiGLUFFN", "TransformerBlock",
    "byte_tokenizer_new", "tokenizer_encode", "tokenizer_decode",
    "tensor_slice", "causal_mask",
    "scaled_dot_product_attention", "rotate_half",
    "tensor_cos", "tensor_sin", "tensor_cat",
    "zeros_like", "ones_like", "full_like",
]

# --- Operators (140-169) ---
NSL_OPERATORS = [
    "@", "|>", "**", "+=", "-=", "*=", "/=",
    "->", ":", "=", "==", "!=", "<", ">", "<=", ">=",
    "+", "-", "*", "/", "%", ".", ",", ";",
    "(", ")", "[", "]", "{", "}",
]

# --- Structure tokens (170-179) ---
STRUCTURE_TOKENS = [
    "<indent>", "<dedent>", "<newline>",
    "<tab>", "<double_newline>", "<colon_newline>",
    "#", "...", "_", "<hashbang>",
]

# --- Number tokens (180-211) ---
NUMBER_TOKENS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "0.0", "1.0", "0.1", "0.01", "0.001", "0.0001",
    "0.5", "2.0", "0.02", "0.9", "0.95", "0.99",
    "32", "64", "128", "256", "512", "1024", "2048", "4096",
    "49152", "384",
]

# --- Common identifiers (212-399) ---
COMMON_IDENTS = [
    "x", "y", "z", "h", "w", "b", "m", "n", "k", "v", "q",
    "loss", "logits", "pred", "target", "labels", "input_ids",
    "hidden", "output", "result", "out",
    "batch", "seq", "dim", "size", "len", "num",
    "batch_size", "seq_len", "d_model", "d_ff", "n_heads", "n_kv_heads",
    "head_dim", "vocab_size", "num_layers", "dropout_p",
    "weight", "bias", "scale", "gamma", "beta", "eps",
    "lr", "epoch", "step", "param",
    "gate", "up", "down", "proj", "norm",
    "embed", "blocks", "attn", "ffn",
    "w_gate", "w_up", "w_down", "w_q", "w_k", "w_v", "w_o",
    "attn_norm", "ffn_norm", "final_norm",
    "training", "forward", "forward_train",
    "tokens", "loader",
    "running_loss", "total_tokens",
    "optimizer", "scheduler",
    "data", "callbacks",
    "on_step", "on_epoch",
]

# Pad each category to its range boundary
def _pad_to(tokens: list[str], target_len: int, prefix: str) -> list[str]:
    while len(tokens) < target_len:
        tokens.append(f"<{prefix}_reserved_{len(tokens)}>")
    return tokens[:target_len]

def build_fixed_vocab() -> list[str]:
    """Return the complete fixed vocabulary (tokens 0-399) as a list."""
    vocab: list[str] = []
    vocab.extend(_pad_to(list(SPECIAL_TOKENS), 10, "special"))    # 0-9
    vocab.extend(_pad_to(list(NSL_KEYWORDS), 50, "kw"))           # 10-59
    vocab.extend(_pad_to(list(NSL_BUILTINS), 80, "builtin"))      # 60-139
    vocab.extend(_pad_to(list(NSL_OPERATORS), 30, "op"))          # 140-169
    vocab.extend(_pad_to(list(STRUCTURE_TOKENS), 10, "struct"))   # 170-179
    vocab.extend(_pad_to(list(NUMBER_TOKENS), 32, "num"))         # 180-211
    vocab.extend(_pad_to(list(COMMON_IDENTS), 188, "ident"))      # 212-399
    assert len(vocab) == 400, f"Fixed vocab should be 400, got {len(vocab)}"
    return vocab

VOCAB_SIZE = 4096
FIXED_VOCAB_SIZE = 400
BPE_VOCAB_SIZE = VOCAB_SIZE - FIXED_VOCAB_SIZE  # 3696
```

- [ ] **Step 2: Create the output directory**

```bash
mkdir -p models/coder-rl/data/tokenizer
```

- [ ] **Step 3: Verify vocab.py loads and produces 400 tokens**

```bash
cd c:/Users/bwiem/projects/NSL
py -c "import sys; sys.path.insert(0, 'models/coder-rl/data'); from vocab import build_fixed_vocab; v = build_fixed_vocab(); print(f'Fixed vocab: {len(v)} tokens'); assert len(v) == 400; print('OK')"
```

Expected: `Fixed vocab: 400 tokens` then `OK`.

- [ ] **Step 4: Commit**

```bash
git add models/coder-rl/data/vocab.py
git commit -m "feat(coder-rl): add fixed vocabulary definition for 4096-token BPE"
```

---

### Task 2: Build the BPE tokenizer trainer

**Files:**
- Create: `models/coder-rl/data/build_tokenizer.py`

Trains a ByteLevel BPE tokenizer on the full corpus (all 4 sources including spec prose). Fixed tokens 0-399 are pre-assigned; BPE learns merges for slots 400-4095.

- [ ] **Step 1: Create `build_tokenizer.py`**

```python
#!/usr/bin/env python3
"""Train a 4096-vocab NSL-aware BPE tokenizer.

Trains on all NSL sources (stdlib, examples, models, spec) so the BPE
merges learn both NSL code patterns and English prose patterns. This
ensures efficient encoding of English prompts during GRPO Phase 3.

Usage:
    py build_tokenizer.py
"""

import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
except ImportError:
    print("Install: py -m pip install tokenizers")
    sys.exit(1)

from vocab import build_fixed_vocab, VOCAB_SIZE


def collect_corpus_files(repo_root: Path) -> list[str]:
    """Collect all training corpus files (all 4 sources)."""
    paths: list[str] = []

    # stdlib .nsl files
    for f in sorted((repo_root / "stdlib").rglob("*.nsl")):
        paths.append(str(f))

    # examples .nsl files
    for f in sorted((repo_root / "examples").rglob("*.nsl")):
        paths.append(str(f))

    # models .nsl files
    for f in sorted((repo_root / "models").rglob("*.nsl")):
        paths.append(str(f))

    # spec .md files (raw markdown including English prose)
    for f in sorted((repo_root / "spec").rglob("*.md")):
        paths.append(str(f))

    return paths


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_path = Path(__file__).resolve().parent / "tokenizer" / "nsl_coder_4k.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Build the fixed vocabulary (tokens 0-399)
    fixed_vocab = build_fixed_vocab()
    print(f"Fixed vocabulary: {len(fixed_vocab)} tokens (0-{len(fixed_vocab)-1})")

    # 2. Collect training corpus (all 4 sources)
    corpus_files = collect_corpus_files(repo_root)
    print(f"Training corpus: {len(corpus_files)} files")
    for f in corpus_files[:5]:
        print(f"  {f}")
    if len(corpus_files) > 5:
        print(f"  ... and {len(corpus_files) - 5} more")

    # 3. Create BPE tokenizer with ByteLevel pre-tokenizer
    #    ByteLevel maps all raw bytes (including spaces, tabs, newlines)
    #    to visible unicode characters, preserving NSL's indentation syntax.
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 4. Train BPE with fixed special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=fixed_vocab,
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train(corpus_files, trainer)

    # 5. Save
    tokenizer.save(str(out_path))
    print(f"\nSaved tokenizer to {out_path}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")

    # 6. Quick validation
    test_code = 'fn forward(self, x: Tensor) -> Tensor:\n    return relu(x @ self.w)'
    encoded = tokenizer.encode(test_code)
    decoded = tokenizer.decode(encoded.ids)
    print(f"\nTest encode/decode:")
    print(f"  Input:   {test_code!r}")
    print(f"  Tokens:  {len(encoded.ids)} ids")
    print(f"  Decoded: {decoded!r}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the tokenizer trainer**

```bash
cd c:/Users/bwiem/projects/NSL/models/coder-rl/data
py build_tokenizer.py
```

Expected output:
```
Fixed vocabulary: 400 tokens (0-399)
Training corpus: ~148 files
...
Saved tokenizer to .../tokenizer/nsl_coder_4k.json
Vocab size: 4096
Test encode/decode:
  Input:   'fn forward(self, x: Tensor) -> Tensor:\n    return relu(x @ self.w)'
  Tokens:  ~15-25 ids
  Decoded: 'fn forward(self, x: Tensor) -> Tensor:\n    return relu(x @ self.w)'
```

- [ ] **Step 3: Verify keyword atomicity**

```bash
cd c:/Users/bwiem/projects/NSL/models/coder-rl/data
py -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('tokenizer/nsl_coder_4k.json')
keywords = ['fn', 'let', 'const', 'model', 'train', 'return', 'grad', 'Tensor']
for kw in keywords:
    ids = tok.encode(kw).ids
    print(f'  {kw:12s} -> {ids}')
    assert len(ids) == 1, f'{kw} should be 1 token, got {len(ids)}'
print('All keywords are single tokens. OK')
"
```

Expected: each keyword maps to exactly 1 token ID.

- [ ] **Step 4: Verify indentation round-trip**

```bash
cd c:/Users/bwiem/projects/NSL/models/coder-rl/data
py -c "
from tokenizers import Tokenizer
from pathlib import Path
tok = Tokenizer.from_file('tokenizer/nsl_coder_4k.json')
repo = Path('.').resolve().parents[2]
test_file = repo / 'stdlib' / 'nsl' / 'nn' / 'gqa.nsl'
original = test_file.read_text(encoding='utf-8')
encoded = tok.encode(original)
decoded = tok.decode(encoded.ids)
if original == decoded:
    print(f'Round-trip OK ({len(encoded.ids)} tokens)')
else:
    # ByteLevel may normalize some whitespace; check structural equivalence
    orig_lines = original.strip().splitlines()
    dec_lines = decoded.strip().splitlines()
    diffs = sum(1 for a, b in zip(orig_lines, dec_lines) if a != b)
    print(f'Round-trip: {diffs} line differences out of {len(orig_lines)}')
    if diffs > 0:
        for i, (a, b) in enumerate(zip(orig_lines, dec_lines)):
            if a != b:
                print(f'  Line {i}: {a!r} vs {b!r}')
                break
"
```

Expected: zero differences or only trivial whitespace normalization.

- [ ] **Step 5: Verify English prompt efficiency**

```bash
cd c:/Users/bwiem/projects/NSL/models/coder-rl/data
py -c "
from tokenizers import Tokenizer
tok = Tokenizer.from_file('tokenizer/nsl_coder_4k.json')
prompt = 'Write a SwiGLU feedforward block in NSL with dropout and Xavier initialization'
ids = tok.encode(prompt).ids
words = len(prompt.split())
print(f'Prompt: {words} words -> {len(ids)} tokens (ratio: {len(ids)/words:.1f}x)')
assert len(ids) < words * 3, f'English too fragmented: {len(ids)} tokens for {words} words'
print('English encoding efficiency OK')
"
```

Expected: ratio below 3x (ideally 1.5-2x).

- [ ] **Step 6: Commit**

```bash
cd c:/Users/bwiem/projects/NSL
git add models/coder-rl/data/build_tokenizer.py models/coder-rl/data/tokenizer/
git commit -m "feat(coder-rl): train 4096-vocab NSL-aware BPE tokenizer"
```

---

### Task 3: Build the SFT corpus preparer

**Files:**
- Create: `models/coder-rl/data/prepare_sft.py`

Encodes code-only files (stdlib + examples + models + extracted spec code blocks) into `sft_tokens.bin` (u16 inputs) and `sft_labels.bin` (i32 labels with -100 for padding).

- [ ] **Step 1: Create `prepare_sft.py`**

```python
#!/usr/bin/env python3
"""Build pre-tokenized SFT corpus for coder-rl Phase 1 training.

Encodes pure NSL code (no English prose) into binary format:
  - sft_tokens.bin: u16 input token IDs, padded with PAD_ID (0)
  - sft_labels.bin: i32 target labels shifted by 1, padded with -100

Usage:
    py prepare_sft.py
"""

import re
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer
except ImportError:
    print("Install: py -m pip install tokenizers")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Install: py -m pip install numpy")
    sys.exit(1)

from vocab import BOS_ID, EOS_ID, PAD_ID

SEQ_LEN = 2048
LABEL_IGNORE = -100


def collect_nsl_files(repo_root: Path) -> list[Path]:
    """Collect all .nsl files from stdlib, examples, models."""
    files: list[Path] = []
    for d in ["stdlib", "examples", "models"]:
        files.extend(sorted((repo_root / d).rglob("*.nsl")))
    return files


def extract_nsl_blocks(markdown_text: str) -> str:
    """Extract NSL code blocks from markdown spec files.

    Uses forgiving whitespace boundaries to handle inconsistent formatting.
    Falls back to unfenced code blocks that contain NSL keywords.
    """
    # Primary: fenced ```nsl blocks
    blocks = re.findall(r"```nsl\s*(.*?)\s*```", markdown_text, re.DOTALL)

    # Fallback: unfenced ``` blocks containing NSL keywords
    if not blocks:
        generic = re.findall(r"```\s*(.*?)\s*```", markdown_text, re.DOTALL)
        nsl_keywords = {"fn ", "let ", "const ", "model ", "train(", "from "}
        blocks = [b for b in generic if any(kw in b for kw in nsl_keywords)]

    return "\n\n".join(blocks)


def collect_spec_code(repo_root: Path) -> list[tuple[str, str]]:
    """Extract NSL code blocks from spec markdown files.

    Returns list of (source_name, code_text) tuples.
    """
    results: list[tuple[str, str]] = []
    spec_dir = repo_root / "spec"
    if not spec_dir.exists():
        return results

    for f in sorted(spec_dir.glob("*.md")):
        text = f.read_text(encoding="utf-8", errors="replace")
        code = extract_nsl_blocks(text)
        if code.strip():
            results.append((f"spec/{f.name}", code))

    return results


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(__file__).resolve().parent
    tok_path = out_dir / "tokenizer" / "nsl_coder_4k.json"

    if not tok_path.exists():
        print(f"Tokenizer not found at {tok_path}")
        print("Run build_tokenizer.py first.")
        sys.exit(1)

    tok = Tokenizer.from_file(str(tok_path))

    # 1. Collect code-only documents
    documents: list[tuple[str, str]] = []

    # .nsl files from stdlib, examples, models
    for f in collect_nsl_files(repo_root):
        text = f.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            rel = f.relative_to(repo_root)
            documents.append((str(rel), text))

    # Extracted code blocks from spec
    spec_docs = collect_spec_code(repo_root)
    documents.extend(spec_docs)

    print(f"Collected {len(documents)} documents:")
    nsl_count = sum(1 for name, _ in documents if not name.startswith("spec/"))
    spec_count = sum(1 for name, _ in documents if name.startswith("spec/"))
    print(f"  .nsl files: {nsl_count}")
    print(f"  spec code blocks: {spec_count}")

    # 2. Encode each document, build input/label pairs with padding
    all_inputs: list[int] = []
    all_labels: list[int] = []
    total_real_tokens = 0
    total_pad_tokens = 0

    for name, text in documents:
        ids = [BOS_ID] + tok.encode(text).ids + [EOS_ID]
        total_real_tokens += len(ids)

        # Inputs: all tokens except last. Labels: all tokens except first.
        inp = ids[:-1]
        lbl = ids[1:]

        # Pad to next seq_len boundary
        padded_len = ((len(inp) + SEQ_LEN - 1) // SEQ_LEN) * SEQ_LEN
        pad_count = padded_len - len(inp)
        total_pad_tokens += pad_count

        inp += [PAD_ID] * pad_count
        lbl += [LABEL_IGNORE] * pad_count

        all_inputs.extend(inp)
        all_labels.extend(lbl)

    # 3. Write binary files
    inputs_arr = np.array(all_inputs, dtype=np.uint16)
    labels_arr = np.array(all_labels, dtype=np.int32)

    inputs_path = out_dir / "sft_tokens.bin"
    labels_path = out_dir / "sft_labels.bin"

    inputs_arr.tofile(str(inputs_path))
    labels_arr.tofile(str(labels_path))

    n_seq = len(all_inputs) // SEQ_LEN
    pad_pct = total_pad_tokens / len(all_inputs) * 100

    print(f"\nCorpus statistics:")
    print(f"  Real tokens:     {total_real_tokens:,}")
    print(f"  Pad tokens:      {total_pad_tokens:,} ({pad_pct:.1f}%)")
    print(f"  Total tokens:    {len(all_inputs):,}")
    print(f"  Sequences:       {n_seq} (seq_len={SEQ_LEN})")
    print(f"  sft_tokens.bin:  {inputs_path.stat().st_size:,} bytes")
    print(f"  sft_labels.bin:  {labels_path.stat().st_size:,} bytes")
    print(f"\nAt 10 epochs: ~{n_seq * 10} forward+backward passes")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the corpus preparer**

```bash
cd c:/Users/bwiem/projects/NSL/models/coder-rl/data
py prepare_sft.py
```

Expected output:
```
Collected ~135+ documents:
  .nsl files: ~135
  spec code blocks: ~12
Corpus statistics:
  Real tokens:     ~40,000-50,000
  Pad tokens:      ~... (...)
  Total tokens:    ~...
  Sequences:       ~20-30 (seq_len=2048)
  sft_tokens.bin:  ~... bytes
  sft_labels.bin:  ~... bytes
```

- [ ] **Step 3: Validate no English contamination**

```bash
cd c:/Users/bwiem/projects/NSL/models/coder-rl/data
py -c "
import numpy as np
from tokenizers import Tokenizer

tok = Tokenizer.from_file('tokenizer/nsl_coder_4k.json')
tokens = np.fromfile('sft_tokens.bin', dtype=np.uint16)

# Decode 5 random 128-token windows and print them
import random
random.seed(42)
for _ in range(5):
    start = random.randint(0, len(tokens) - 128)
    window = tokens[start:start+128].tolist()
    # Skip if all padding
    if all(t == 0 for t in window):
        continue
    text = tok.decode(window)
    print(f'--- offset {start} ---')
    print(text[:200])
    print()
"
```

Expected: all windows show NSL code (keywords, operators, indentation), no English prose paragraphs.

- [ ] **Step 4: Validate label masking**

```bash
cd c:/Users/bwiem/projects/NSL/models/coder-rl/data
py -c "
import numpy as np

inputs = np.fromfile('sft_tokens.bin', dtype=np.uint16)
labels = np.fromfile('sft_labels.bin', dtype=np.int32)

assert len(inputs) == len(labels), f'Length mismatch: {len(inputs)} vs {len(labels)}'

# Check: wherever input is PAD (0) after EOS, label should be -100
pad_positions = (inputs == 0)
label_at_pad = labels[pad_positions]
non_ignore = label_at_pad[label_at_pad != -100]
# Some PAD positions at start of sequences may have valid labels (BOS token)
# but post-EOS padding should always be -100
print(f'Total tokens: {len(inputs):,}')
print(f'PAD positions: {pad_positions.sum():,}')
print(f'Labels at PAD that are not -100: {len(non_ignore)} (should be ~0)')
print(f'Labels that are -100: {(labels == -100).sum():,}')

# Check: no -100 in non-pad regions (real code)
real_mask = labels != -100
real_labels = labels[real_mask]
assert (real_labels >= 0).all(), 'Found negative labels in non-pad region'
print(f'Real label range: [{real_labels.min()}, {real_labels.max()}]')
print('Label masking OK')
"
```

Expected: `Label masking OK` with minimal non-ignore values at PAD positions.

- [ ] **Step 5: Commit**

```bash
cd c:/Users/bwiem/projects/NSL
git add models/coder-rl/data/prepare_sft.py models/coder-rl/data/sft_tokens.bin models/coder-rl/data/sft_labels.bin
git commit -m "feat(coder-rl): build pre-tokenized SFT corpus (Phase A)"
```

---

### Task 4: End-to-end training validation

**Files:**
- No new files. Uses existing `models/coder-rl/train_grpo.nsl` and `models/coder-rl/model.nsl`.

Verify the full pipeline works: tokenizer -> corpus -> DataLoader -> model forward -> loss -> backward.

- [ ] **Step 1: Create a minimal SFT test script**

Create `models/coder-rl/test_sft.nsl`:

```nsl
# models/coder-rl/test_sft.nsl
# Quick validation: 1 epoch SFT on the tiny corpus

from model import NSLCoderRL
from nsl.nn.losses import cross_entropy

print("Loading model...")
let m = NSLCoderRL()
print("Model created (10.1M params)")

let tokens = load_mmap("data/sft_tokens.bin", 3)
let loader = DataLoader(tokens, batch_size=1, seq_len=2048, shuffle=false, drop_last=true)

print("Starting 1-epoch SFT test...")

train(model=m, epochs=1, grad_clip=1.0):
    optimizer: AdamW(lr=0.0003, weight_decay=0.01)
    step(batch):
        let logits = m.forward_train(batch.input_ids, true)
        let ls = logits.shape
        let flat_logits = logits.reshape([ls[0] * ls[1], ls[2]])
        let flat_labels = batch.labels.reshape([ls[0] * ls[1]])
        let loss = cross_entropy(flat_logits, flat_labels)
    callbacks:
        on_step(step, loss):
            print(step)
            print(loss)

print("SFT test complete.")
```

- [ ] **Step 2: Run the end-to-end test with source AD**

```bash
cd c:/Users/bwiem/projects/NSL
cargo run -- run --source-ad models/coder-rl/test_sft.nsl
```

Expected: model initializes, processes at least 1 batch, prints loss values, loss is a reasonable number (not NaN, not 0). If loss decreases over steps, training is working.

- [ ] **Step 3: Commit the test script**

```bash
cd c:/Users/bwiem/projects/NSL
git add models/coder-rl/test_sft.nsl
git commit -m "test(coder-rl): add end-to-end SFT validation script"
```

---

### Task 5: Final cleanup and push

- [ ] **Step 1: Verify all generated files exist**

```bash
ls -la models/coder-rl/data/tokenizer/nsl_coder_4k.json
ls -la models/coder-rl/data/sft_tokens.bin
ls -la models/coder-rl/data/sft_labels.bin
ls -la models/coder-rl/data/vocab.py
ls -la models/coder-rl/data/build_tokenizer.py
ls -la models/coder-rl/data/prepare_sft.py
ls -la models/coder-rl/test_sft.nsl
```

- [ ] **Step 2: Add generated binaries to .gitignore**

Add to the project `.gitignore`:

```
# Coder-RL generated data (rebuild with build_tokenizer.py + prepare_sft.py)
models/coder-rl/data/sft_tokens.bin
models/coder-rl/data/sft_labels.bin
models/coder-rl/data/tokenizer/
```

- [ ] **Step 3: Final commit and push**

```bash
cd c:/Users/bwiem/projects/NSL
git add .gitignore models/coder-rl/data/vocab.py models/coder-rl/data/build_tokenizer.py models/coder-rl/data/prepare_sft.py models/coder-rl/test_sft.nsl
git commit -m "feat(coder-rl): complete Phase A SFT data pipeline

Adds the full tokenizer + corpus pipeline for coder-rl SFT training:
- vocab.py: 400 fixed tokens (keywords, builtins, operators, identifiers)
- build_tokenizer.py: trains 4096-vocab ByteLevel BPE on full corpus
- prepare_sft.py: encodes code-only corpus with -100 label masking
- test_sft.nsl: end-to-end training validation script"

git push origin main
```
