# NSL-Coder-50M: Proof-of-Concept Language Model

**Date:** 2026-03-19
**Status:** Draft
**Goal:** Train a 50M-parameter LLaMA-style code LM defined, trained, and served
entirely in NSL — proving the language can replace Python+PyTorch for real AI workloads.

## 1. Motivation

NSL has a complete compiler pipeline (lexer → parser → semantic → Cranelift codegen),
a Rust runtime with 77 tensor ops, 15 PTX kernels, FlashAttention-2, and a training DSL.
What it lacks is a **trained model** that demonstrates the full loop:

1. Define a modern architecture in NSL
2. Train it on real data
3. Interactively generate code from it

NSL-Coder-50M is that proof-of-concept. It is simultaneously:
- **A showcase artifact** — a real, usable code-generating LLM in the repo
- **A compiler stress test** — exercising imports, model codegen, training DSL,
  GPU forward pass, dynamic shapes, and FlashAttention in a single workload

## 2. Architecture

Decoder-only transformer following the LLaMA-2/3 architecture family.

### 2.1 Hyperparameters

| Parameter       | Value   | Rationale                                      |
|-----------------|---------|-------------------------------------------------|
| vocab_size      | 49,152  | Matches existing CodeForge BPE tokenizer        |
| d_model         | 512     | Hidden dimension                                |
| n_layers        | 8       | Transformer blocks                              |
| n_heads         | 8       | Query attention heads                           |
| n_kv_heads      | 4       | KV heads (GQA 2:1 ratio)                       |
| head_dim        | 64      | d_model / n_heads                               |
| d_ff            | 1408    | SwiGLU FFN (≈ 8/3 × d_model, 128-aligned)      |
| max_seq_len     | 1024    | Context window                                  |
| rope_theta      | 10000.0 | RoPE base frequency                             |
| precision       | f32     | Full runtime support; BF16 deferred (see 2.4)   |
| weight_tying    | yes     | lm_head shares embedding weights                |
| norm            | RMSNorm | Pre-norm, already in stdlib                     |
| activation      | SwiGLU  | gate × silu(up), proven in CodeForge Nano       |

### 2.2 Parameter Count

| Component                | Parameters  |
|--------------------------|-------------|
| Embedding (tied w/ head) | 25,165,824  |
| Per-layer attention      | 786,432     |
| Per-layer SwiGLU FFN     | 2,162,688   |
| Per-layer RMSNorm × 2    | 1,024       |
| **Per-layer total**      | **2,950,144** |
| **8 layers**             | **23,601,152** |
| Final RMSNorm            | 512         |
| **Total**                | **~48.8M**  |

### 2.3 Architecture Diagram

```
Input IDs [B, S]
    │
    ▼
┌─────────────────────┐
│  Embedding (49152×512) │  ← shared with LM Head
└─────────┬───────────┘
          │ + RoPE positional encoding
          ▼
┌─────────────────────────────────────────┐
│  Transformer Block × 8                  │
│  ┌───────────────────────────────────┐  │
│  │ RMSNorm → GQA Attention          │  │
│  │   Q: [B,S,512] → [B,8,S,64]     │  │
│  │   K: [B,S,512] → [B,4,S,64]     │  │
│  │   V: [B,S,512] → [B,4,S,64]     │  │
│  │   K,V expand → [B,8,S,64]       │  │
│  │   Attention (f32, naive O(N²))   │  │
│  │   O: [B,8,S,64] → [B,S,512]     │  │
│  │ + residual                        │  │
│  ├───────────────────────────────────┤  │
│  │ RMSNorm → SwiGLU FFN             │  │
│  │   gate = W_gate @ x    [512→1408]│  │
│  │   up   = W_up   @ x    [512→1408]│  │
│  │   out  = W_down @ (silu(gate)*up) │  │
│  │                         [1408→512]│  │
│  │ + residual                        │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
          │
          ▼
┌─────────────────────┐
│  RMSNorm → LM Head  │  ← tied weights (transpose of embedding)
└─────────┬───────────┘
          ▼
Logits [B, S, 49152]
```

### 2.4 Key Architectural Decisions

**GQA (Grouped Query Attention):** 4 KV heads serve 8 query heads via `expand`
on the head dimension. The current `expand` implementation performs a physical copy;
it must be rewritten to use stride=0 metadata (zero-copy view) so the attention
matmul reads the same KV head data twice without allocating duplicate memory.
This is a must-have compiler fix (see Section 6.1). Once zero-copy, GQA saves
~30% memory bandwidth compared to full MHA.

**RoPE (Rotary Position Embeddings):** Uses the **half-split** variant (LLaMA-2/3
style). Precomputed sin/cos frequency table of shape `[max_seq_len, head_dim/2]`.
Applied as complex rotation to Q and K before attention:

```
x_rotated = x * cos_cached + rotate_half(x) * sin_cached
```

Where `rotate_half` splits the last dimension in half and swaps with negation:
`rotate_half([x1, x2, x3, x4]) = [-x3, -x4, x1, x2]`
(i.e., `cat(-x[..., D/2:], x[..., :D/2], dim=-1)`).
Implemented as a fused runtime intrinsic to avoid 3 discrete memory passes
(slice, negate, cat).

**SwiGLU:** Gated linear unit with SiLU activation. Uses 3 weight matrices (gate, up,
down) instead of the standard 2 (up, down). The 8/3 scaling factor on d_ff compensates
for the extra parameters, keeping the total FLOPs per layer comparable to standard FFN.
d_ff=1408 is the nearest 128-aligned value (for Tensor Core tile alignment).

**f32 Precision:** The NSL tensor runtime currently supports f64 and f32 code paths
for all operations. BF16 support (dtype=3) is not yet wired through — `element_size()`
panics on BF16, and no activation functions dispatch on it. f32 is the pragmatic
choice: all tensor ops work, training is stable for 50M params, and there are no
loss scaling concerns. This means FlashAttention-2 Tensor Core path (which requires
16-bit inputs) is unavailable — attention falls back to the standard O(N^2) path.
At seq_len=1024 this is acceptable (~1M elements per attention computation).
BF16 support across the runtime is a follow-up optimization that would unlock
FlashAttention for this model.

**Weight Tying:** The embedding matrix (25.1M params) is over 50% of the model. Tying
it with the LM head avoids doubling this cost. The `@tie_weights` decorator is not yet
implemented in the compiler, so weight tying is done manually in the forward method:
`logits = x @ self.embed.weight.transpose(0, 1)` — directly reusing the embedding
weight matrix as the LM head projection. No compiler changes required.

## 3. Training Pipeline

### 3.1 Stage 1: Pretrain (General Code)

| Parameter        | Value                                             |
|------------------|---------------------------------------------------|
| Data             | First 10B tokens from StarCoder (pretokenized u16)|
| Data path        | `tokens.bin` via `load_mmap(path, 3)` (dtype 3=u16) |
| Tokenizer        | CodeForge BPE (49,152 vocab)                      |
| Batch size       | 32                                                |
| Sequence length  | 1024                                              |
| Tokens per step  | 32,768                                            |
| Total steps      | ~305,000                                          |
| Optimizer        | AdamW (lr=3e-4, β1=0.9, β2=0.95, wd=0.1)        |
| Scheduler        | warmup_cosine (warmup=3000, total=305K, min=3e-5) |
| Gradient clip    | Global norm 1.0                                   |
| Checkpoints      | Every 10,000 steps (.nslm format)                 |
| Device           | Single CUDA GPU                                   |

**Data source:** The existing pretokenized StarCoder dataset at
`codeforge/data/pretokenized/150m/tokens.bin` (78 GB total, 30B tokens).
We consume only the first 10B tokens (≈26 GB). The dataset covers 8 languages:
Python (35%), JavaScript (12%), TypeScript (12%), Java (10%), C++ (8%),
C (5%), Go (7%), Rust (7%).

**Why 10B tokens for 50M params:** This gives a 200× token-to-parameter ratio,
well into the "Chinchilla-overtrained" regime (optimal is ~20×). This is intentional —
small models benefit disproportionately from overtraining, and the data is already
tokenized so there is no added cost.

### 3.2 Stage 2: Finetune (NSL Code)

| Parameter        | Value                                             |
|------------------|---------------------------------------------------|
| Data             | 10M token mixture: 1M NSL + 9M general code       |
| NSL source       | All .nsl files: stdlib/, examples/, tests/, spec/  |
| General source   | Sampled from Stage 1 StarCoder data                |
| Estimated NSL    | ~500K–1M tokens after BPE tokenization             |
| Epochs           | 1–2 over the 10M token mixture                     |
| Batch size       | 16                                                |
| Sequence length  | 1024                                              |
| Tokens per step  | 16,384                                            |
| Total steps      | ~610 (1 epoch) to ~1,220 (2 epochs)               |
| Optimizer        | AdamW (lr=1e-5, β1=0.9, β2=0.95, wd=0.1)        |
| Scheduler        | warmup_cosine (warmup=100, min=1e-6)              |
| Checkpoints      | Every 1,000 steps                                 |

**Why data mixing:** Training exclusively on ~1M NSL tokens for many epochs causes
catastrophic memorization — the model regurgitates exact files instead of generating
novel code. It also causes catastrophic forgetting of the general coding skills
learned in Stage 1. The 1:9 mix ratio and low epoch count preserve general ability
while teaching NSL-specific syntax (`model`, `train`, `grad`, `|>`, etc.).

### 3.3 Gradient Clipping

Global norm clipping is **mandatory** for training stability. Without it, early-stage
gradient spikes in the attention logits corrupt AdamW momentum buffers, sending loss
to NaN.

Implementation: compute the global L2 norm across all parameter gradients. If the
norm exceeds the threshold (1.0), scale all gradients by `threshold / global_norm`.

If the runtime cannot support global norm computation at launch, fall back to
element-wise `clamp(grad, -1.0, 1.0)` as a stopgap. Element-wise clamping is
mathematically inferior (it distorts gradient direction) but infinitely better than
no clipping at all.

### 3.4 Data Preparation

**`data/prepare_nsl.py`** — Python helper script that:
1. Globs all `.nsl` files from the repo (stdlib/, examples/, tests/)
2. Reads each file, concatenates with `<|file_sep|>` separator tokens
3. Tokenizes with the CodeForge BPE tokenizer (`codeforge.json`)
4. Writes output as a flat u16 binary file for `load_mmap()`
5. Prints token count statistics

This is deliberately in Python — NSL is a math/execution engine, not a file-parsing
utility. PyTorch itself relies on Python for data preparation.

### 3.5 Data Loading & Batching

The pretokenized data is a flat 1D array of u16 token IDs loaded via
`load_mmap(path, 3)`. The `DataLoader` handles reshaping into training batches:

```nsl
let tokens = load_mmap(DATA_PATH, 3)
let loader = DataLoader(tokens, batch_size=32, seq_len=1024, shuffle=false, drop_last=true)
for batch in loader:
    # batch.input_ids: [32, 1024]
    # batch.labels:    [32, 1024] (shifted by 1)
```

The DataLoader slices the flat array into contiguous chunks of `batch_size * seq_len`
tokens, reshapes to `[batch_size, seq_len]`, and constructs input/label pairs with
the standard causal shift (labels = input shifted right by 1 position).

For Stage 1 (10B tokens, single epoch), shuffling is skipped — sequential consumption
of the pretokenized data is standard practice. For Stage 2 (10M tokens, 1-2 epochs),
shuffling is enabled to avoid order effects.

**Note on `load_mmap` dtype numbering:** `load_mmap` uses its own dtype enum
(0=f64, 1=f32, 2=i32, 3=u16), which is separate from the tensor dtype constants
(0=f64, 1=f32, 2=fp16, 3=bf16). The u16 values from the file are converted to the
tensor's native dtype (f64 or i32) on load. This namespace collision is a known
inconsistency that should be resolved in a future cleanup.

### 3.6 Validation & Metrics

Training progress is monitored via:

1. **Training loss** — logged every 100 steps via `on_step` callback
2. **Validation loss** — computed every 5,000 steps on a held-out 1% of the data
   (first 100M tokens reserved for validation, remaining 9.9B for training)
3. **Perplexity** — `exp(val_loss)`, logged alongside validation loss

Validation prevents silent overfitting and confirms the training run is healthy
without waiting for all 305K steps to complete.

## 4. Inference & Interactive Demo

### 4.1 Generation Loop

```
$ nsl run models/coder50m/generate.nsl

NSL-Coder-50M loaded (48.8M params, f32)
Type a code prompt, then press Enter twice to generate.

>>> fn fibonacci(n: int) -> int:
────────────────────────────────────
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

>>> _
```

### 4.2 Sampling Parameters

| Parameter   | Value | Rationale                                      |
|-------------|-------|-------------------------------------------------|
| top_k       | 40    | Industry standard for code generation            |
| temperature | 0.8   | Balances syntactic strictness and creativity     |
| max_tokens  | 512   | Configurable, sensible default                   |
| stop tokens | EOF, `\n\n`, max_tokens | Function-level generation |

### 4.3 Prefill vs. Decode

The interactive loop executes two distinct computational phases:

1. **Prefill:** The prompt tokens are processed in parallel. Forward pass shape:
   `[1, prompt_len, 512]`. KV-cache is populated for all prompt positions.

2. **Decode:** Tokens are generated one at a time. Forward pass shape:
   `[1, 1, 512]`. Each step appends one KV entry and produces one logit vector.

This requires M28 Dynamic Shapes to handle the variable sequence length dimension.
The Cranelift codegen must compile the model's forward function with the sequence
length as a bounded dynamic dimension (`Dim::Bounded(max_seq_len)`).

**KV-cache:** Listed as nice-to-have (Section 6.3). Without it, each decode step
re-processes all previous tokens from scratch, giving O(n^2) total compute for
n generated tokens. For a 512-token generation at seq_len ~1024, this is noticeably
slow but acceptable for a proof-of-concept demo. KV-cache append is the primary
follow-up optimization for interactive latency.

### 4.4 Sampling Path

**Initial implementation:** CPU-side sampling. The full logit vector `[49152]`
(~192 KB in f32) is transferred from GPU to CPU. `sample_top_k()` runs in Rust:
sort, truncate, softmax, multinomial. The selected token ID is a single `u32`.

**Follow-up optimization:** Fused GPU `sample_top_k` PTX kernel using warp-level
bitonic sort. Only the sampled `u32` token ID crosses PCIe. Eliminates 2-5ms
per-token latency from the logit transfer.

### 4.5 stdin Input

A new `read_line()` runtime intrinsic reads a line from stdin and returns a string.
Implemented as a thin FFI wrapper around Rust's `std::io::stdin().read_line()`.
Exposed in the `nsl.io` stdlib module.

## 5. File Structure

```
models/coder50m/
├── config.nsl           # All hyperparameters as constants
├── model.nsl            # LLaMA architecture: RoPE, GQA, SwiGLU, RMSNorm
├── pretrain.nsl         # Stage 1: 10B StarCoder tokens
├── finetune.nsl         # Stage 2: 1M NSL + 9M general code mix
├── generate.nsl         # Interactive inference loop with stdin
├── data/
│   └── prepare_nsl.py   # Tokenize NSL source files with BPE
└── README.md            # Reproduction instructions
```

**config.nsl** — Single source of truth for all model/training hyperparameters.
Imported by model.nsl, pretrain.nsl, finetune.nsl, and generate.nsl.

**model.nsl** — Defines `NSLCoder` model using the `model` keyword. Imports from
stdlib (RMSNorm, Linear, Embedding) and new modules (RotaryEmbedding, GQAttention).

**pretrain.nsl** — Contains the Stage 1 `train()` block. Loads data via `load_mmap()`,
creates a `DataLoader`, runs training with callbacks for logging and checkpointing.

**finetune.nsl** — Contains the Stage 2 `train()` block. Loads the pretrained
checkpoint, trains on the mixed NSL+general dataset with lower learning rate.

**generate.nsl** — Interactive loop: `read_line()` → `tokenizer_encode()` →
prefill → decode loop → `tokenizer_decode()` → print. Loads the finetuned
checkpoint. The decode step converts each sampled `u32` token ID back to a
string fragment via the BPE tokenizer (already in verified deps, Section 6.4).

## 6. Compiler & Runtime Work

### 6.1 Must-Have Primitives

These block the model and must be implemented before training can begin.

| #  | Primitive                  | Purpose                              | Scope          |
|----|----------------------------|--------------------------------------|----------------|
| 1  | `expand(tensor, shape)`    | Zero-copy stride view (stride=0)     | Runtime tensor  |
| 2  | `rotate_half(x)`           | Fused element swap for RoPE          | Runtime intrinsic |
| 3  | `sin(x)` / `cos(x)`       | Elementwise trig for RoPE freqs      | Runtime intrinsic |
| 4  | `arange(start, stop, dtype)` | Sequential tensor creation          | Runtime intrinsic |
| 5  | Global norm gradient clip  | Training stability                   | Optimizer/runtime |
| 6  | `read_line()`              | stdin input for interactive demo     | Runtime intrinsic |
| 7  | M28 dynamic shapes verify  | Prefill/decode shape transition      | Codegen verification |

**expand:** The current `expand` implementation (`shape_ops.rs:621`) performs a
physical copy — it allocates a new buffer and copies element-by-element. This must
be rewritten to modify tensor stride metadata only (set stride=0 on the expanded
dimension) without copying data. For GQA, the KV head dimension stride is set to 0
so that 4 physical KV heads appear as 8 logical heads to the attention matmul.
All downstream consumers (matmul, elementwise ops) must handle non-contiguous
tensors correctly. This is PyTorch `Tensor.expand()` semantics.

**rotate_half:** Takes a tensor of shape `[..., D]` and returns
`cat(-x[..., D/2:], x[..., :D/2], dim=-1)` — the half-split variant used by
LLaMA-2/3. Splits the last dimension in half, negates the second half, and
concatenates in swapped order. Fused into a single memory pass to avoid 3
discrete ops (slice, negate, cat).

**sin/cos:** Tensor-level elementwise trigonometric functions. Only scalar versions
exist today (`nsl_sin(f64) -> f64`, `nsl_cos(f64) -> f64` in `math.rs`). New
tensor-level `nsl_tensor_sin` and `nsl_tensor_cos` functions must be added to the
runtime, operating element-wise over tensors of arbitrary shape. Needed for
precomputing `sin(pos * freq)` and `cos(pos * freq)` tables of shape
`[max_seq_len, head_dim/2]`.

**arange:** Creates a 1D tensor with sequential values `[start, start+1, ..., stop-1]`.
Essential for RoPE position indices `t = arange(0, seq_len)` and dimension indices
`i = arange(0, head_dim, 2)`. Without this, the only alternative is a slow scalar
loop.

**Global norm clip:** Computes `sqrt(sum(grad**2))` across all model parameters.
If the result exceeds the threshold, scales all gradients by `threshold / norm`.
Can be implemented as a new optimizer intrinsic or a stdlib function.

**read_line:** FFI wrapper around `std::io::stdin().read_line()`. Returns a string
(trimmed). Trivial to implement.

**M28 verify:** The model's forward function must accept variable sequence lengths
within `[1, max_seq_len]`. This is a verification task, not a new feature — M28
(Dynamic Shapes) is already implemented. Need to confirm it works for this specific
use case (prefill → decode transition).

### 6.2 New Stdlib Modules

| Module           | Contents                                              |
|------------------|-------------------------------------------------------|
| `nsl.nn.rope`    | `RotaryEmbedding(dim, max_seq, theta)` model          |
|                  | Precomputes freq table, applies rotation to Q/K       |
| `nsl.nn.gqa`     | `GroupedQueryAttention(d_model, n_heads, n_kv_heads)` |
|                  | Q/K/V projections, KV head expansion, attention       |
| `nsl.io`         | `read_line() -> str` stdin wrapper                    |

### 6.3 Nice-to-Have (Defer if Needed)

| Primitive                | Purpose                                    |
|--------------------------|--------------------------------------------|
| Fused GPU `sample_top_k` | On-device sampling, avoids PCIe bottleneck |
| KV-cache append           | Efficient single-token KV append in decode |
| Checkpoint resume         | Load .nslm + continue training from step N |

### 6.4 Verified Runtime Dependencies

These primitives already exist in the runtime and are confirmed working. Listed
here to avoid surprises during implementation.

| Primitive            | Location               | Used For             |
|----------------------|------------------------|----------------------|
| `softmax(x, dim)`    | `tensor/reduction.rs`  | Attention scores     |
| `rmsnorm(x, w, eps)` | `tensor/norm.rs`       | Layer normalization  |
| `silu(x)`            | `tensor/activation.rs` | SwiGLU gate          |
| `embedding_lookup`   | `tensor/mod.rs`        | Token embedding      |
| `matmul` / `@`       | `tensor/matmul.rs`     | Linear projections   |
| `cross_entropy`      | `loss.rs`              | Training loss        |
| `tokenizer_load`     | `tokenizer.rs`         | Load BPE tokenizer   |
| `tokenizer_encode`   | `tokenizer.rs`         | Encode text → tokens |
| `tokenizer_decode`   | `tokenizer.rs`         | Decode tokens → text |
| `sample_top_k`       | `sampling.rs`          | Token sampling       |
| `topk`               | `sampling.rs`          | Top-k selection      |
| `multinomial`        | `sampling.rs`          | Categorical sampling |
| `load_mmap`          | `dataloader.rs`        | Memory-mapped data   |
| `DataLoader`         | `dataloader.rs`        | Batch construction   |
| `reshape`            | `tensor/shape_ops.rs`  | Head dim reshaping   |
| `transpose`          | `tensor/shape_ops.rs`  | Q/K/V reordering     |
| `causal_mask`        | `tensor/mod.rs`        | Attention masking    |

### 6.5 Explicitly Out of Scope

- **No distributed training.** Single GPU suffices for 50M params.
- **No custom CUDA kernels** beyond existing 15 + the sampling follow-up.
- **No serving infrastructure.** The interactive CLI loop is the demo.
- **No mixed-precision training.** f32 throughout (BF16 runtime support deferred).
- **No new tokenizer.** Reuse existing CodeForge BPE (49,152 vocab).
- **No FlashAttention.** Requires BF16 inputs; deferred with BF16 support.

## 7. Success Criteria

1. **`nsl run pretrain.nsl`** completes 305K steps with monotonically decreasing loss
2. **`nsl run finetune.nsl`** runs 1-2 epochs on the NSL+code mixture without diverging
3. **`nsl run generate.nsl`** produces syntactically plausible code completions
   interactively from stdin prompts
4. The model generates recognizable NSL syntax when prompted with NSL-specific
   constructs (`model`, `train`, `fn forward`, `|>`)
5. All model code (.nsl files) compiles without errors on the NSL compiler
6. No Python or C++ is required at any point in the train→inference pipeline
   (only the data prep script is Python)

## 8. Risks & Mitigations

| Risk                                  | Impact | Mitigation                          |
|---------------------------------------|--------|--------------------------------------|
| Missing compiler primitives block model definition | High | Fix-as-we-go approach; primitives are scoped and small |
| RoPE implementation has numerical issues | Medium | Test against PyTorch reference values |
| GQA expand interacts badly with FlashAttention | Medium | Test with naive attention first, then switch |
| Dynamic shapes break during decode phase | Medium | Verify M28 with a minimal test case first |
| Training loss diverges or goes NaN | Medium | Gradient clipping is mandatory; tune LR if needed |
| f32 attention is too slow at seq_len=1024 | Low | O(N^2) is ~1M elements — trivial; BF16+FlashAttn is follow-up |
| NSL corpus too small for meaningful finetune | Low | Data mixing mitigates; eval is qualitative not quantitative |
