# Section 13 — Full Worked Example

## Design Rationale

This section demonstrates all NSL language features working together in a single, complete,
idiomatic program. It implements the full ML pipeline from tokenization through training,
quantization, export, and benchmarking — the kind of program that would require 500+ lines
of Python/PyTorch spread across multiple files, here expressed in ~200 lines of NSL.

```nsl
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
## GPT-2 (124M) — Complete Training, Quantization, and Deployment Pipeline
## NeuralScript v0.1
## ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import nsl.nn.{Linear, Embedding, LayerNorm, Dropout, MLP, softmax, gelu}
import nsl.nn.{cross_entropy, causal_mask}
import nsl.optim.{AdamW, WarmupCosineScheduler}
import nsl.data.{MemoryMapped, DataLoader, SequencePacker}
import nsl.tokenize.{BPETrainer, Tokenizer}
import nsl.quant.{quantize, gptq}
import nsl.export.{to_onnx}
import nsl.bench.{benchmark, count_params, summary}
import nsl.compat.{save_safetensors}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

const CONFIG = {
    vocab_size:    50257,
    d_model:       768,
    n_layers:      12,
    n_heads:       12,
    d_ff:          3072,
    max_seq:       1024,
    dropout:       0.1,
    batch_size:    8,
    accumulate:    4,        # effective batch = 8 * 4 = 32
    lr:            6e-4,
    warmup_steps:  2000,
    total_steps:   100000,
    weight_decay:  0.1,
}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 1: TOKENIZER — BPE with 50,257 vocabulary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

tokenizer gpt_tokenizer(algorithm=bpe, vocab_size=CONFIG.vocab_size):
    special_tokens:
        eos = "<|endoftext|>"
        pad = "<|pad|>"

    normalize: nfkc
    pre_tokenize: whitespace, byte_fallback

    padding:
        side = left
        pad_to = longest

    truncation:
        max_length = CONFIG.max_seq

# Train or load the tokenizer
let tokenizer = if file_exists("tokenizers/gpt2_bpe.json"):
    gpt_tokenizer.load("tokenizers/gpt2_bpe.json")
else:
    let trainer = BPETrainer(
        vocab_size=CONFIG.vocab_size,
        min_frequency=2,
        special_tokens=gpt_tokenizer.special_tokens
    )
    trainer.train_from_files(["data/openwebtext/*.txt"], output="tokenizers/gpt2_bpe.json")
    gpt_tokenizer.load("tokenizers/gpt2_bpe.json")

print(f"Tokenizer loaded: {tokenizer.vocab_size} tokens")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 2: MODEL — GPT-2 (124M) Transformer Decoder
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Single transformer block with pre-norm, causal self-attention, and GELU MLP
model GPT2Block(d_model: int, n_heads: int, d_ff: int, dropout: f32):
    ln1: LayerNorm = LayerNorm(d_model)
    ln2: LayerNorm = LayerNorm(d_model)

    # Attention projections (combined QKV for efficiency)
    c_attn: Linear = Linear(d_model, 3 * d_model)
    c_proj: Linear = Linear(d_model, d_model)
    attn_drop: Dropout = Dropout(dropout)
    resid_drop: Dropout = Dropout(dropout)

    # Feedforward
    c_fc: Linear = Linear(d_model, d_ff)
    c_fc_proj: Linear = Linear(d_ff, d_model)
    mlp_drop: Dropout = Dropout(dropout)

    n_heads: int = n_heads
    head_dim: int = d_model // n_heads

    fn forward(x: Tensor<[batch, seq, d_model], bf16>) -> Tensor<[batch, seq, d_model], bf16>:
        let B, S, D = x.shape

        # --- Multi-Head Causal Self-Attention ---
        let h = self.ln1(x)
        let qkv = self.c_attn(h)                                    # [B, S, 3*D]
        let (q, k, v) = qkv.chunk(3, dim=-1)                        # 3x [B, S, D]

        # Reshape for multi-head: [B, S, D] -> [B, H, S, dk]
        let q = q.reshape([B, S, self.n_heads, self.head_dim]).transpose(1, 2)
        let k = k.reshape([B, S, self.n_heads, self.head_dim]).transpose(1, 2)
        let v = v.reshape([B, S, self.n_heads, self.head_dim]).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        let scores = (q @ k.transpose(-2, -1)) / (self.head_dim as f32).sqrt()
        let mask = causal_mask(S, device=x.device)
        let scores = scores.masked_fill(mask == false, -1e9)
        let attn = softmax(scores, dim=-1) |> self.attn_drop

        let out = (attn @ v).transpose(1, 2).reshape([B, S, D])     # [B, S, D]
        let out = self.c_proj(out) |> self.resid_drop
        let x = x + out                                              # residual

        # --- Feedforward ---
        let h = self.ln2(x)
        let ff = self.c_fc(h) |> gelu |> self.c_fc_proj |> self.mlp_drop
        return x + ff                                                 # residual


## Full GPT-2 model stacking 12 transformer blocks
model GPT2(vocab_size: int, d_model: int, n_layers: int, n_heads: int,
           d_ff: int, max_seq: int, dropout: f32):
    wte: Embedding = Embedding(vocab_size, d_model)                  # token embeddings
    wpe: Embedding = Embedding(max_seq, d_model)                     # position embeddings
    drop: Dropout = Dropout(dropout)

    blocks: list<GPT2Block> = [
        GPT2Block(d_model, n_heads, d_ff, dropout)
        for _ in 0..n_layers
    ]

    ln_f: LayerNorm = LayerNorm(d_model)
    lm_head: Linear = Linear(d_model, vocab_size, bias=false)
        @tie_weights(self.wte.weight)                                # weight tying

    fn forward(input_ids: Tensor<[batch, seq], int32>) -> Tensor<[batch, seq, vocab_size], bf16>:
        let B, S = input_ids.shape
        let positions = arange(S, device=input_ids.device)           # [S]

        let h = self.wte(input_ids) + self.wpe(positions) @broadcast # [B, S, D]
        let h = self.drop(h)

        for block in self.blocks:
            h = block.forward(h)

        let h = self.ln_f(h)
        return self.lm_head(h)                                       # [B, S, vocab_size]

# Instantiate model on GPU
let model = GPT2(
    vocab_size  = CONFIG.vocab_size,
    d_model     = CONFIG.d_model,
    n_layers    = CONFIG.n_layers,
    n_heads     = CONFIG.n_heads,
    d_ff        = CONFIG.d_ff,
    max_seq     = CONFIG.max_seq,
    dropout     = CONFIG.dropout
).to(cuda)

print(f"GPT-2 initialized: {count_params(model):,} parameters")
print(summary(model, input_shape=[1, CONFIG.max_seq]))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 3: DATASET — Memory-mapped text with sequence packing
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

dataset train_data("openwebtext"):
    source = MemoryMapped("data/openwebtext_tokens.bin", dtype=int32)
    sequence_length = CONFIG.max_seq
    packing = true
    pack_separator = tokenizer.eos_token_id

dataset val_data("validation"):
    source = MemoryMapped("data/val_tokens.bin", dtype=int32)
    sequence_length = CONFIG.max_seq
    packing = true
    pack_separator = tokenizer.eos_token_id

print(f"Training data: {train_data.total_tokens():,} tokens")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 4: TRAINING — 1 epoch, AdamW, cosine LR, bf16, gradient accumulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train(
    model       = model,
    epochs      = 1,
    precision   = bf16,
    grad_scaler = auto,
    accumulate  = CONFIG.accumulate,
    clip_grad_norm = 1.0
):
    data:
        source      = train_data
        batch_size  = CONFIG.batch_size
        num_workers = 8
        prefetch    = 4
        pin_memory  = true

    optimizer: AdamW(
        lr           = CONFIG.lr,
        betas        = (0.9, 0.95),
        weight_decay = CONFIG.weight_decay,
        groups = [
            {params: model.params(filter="*.weight"), weight_decay: CONFIG.weight_decay},
            {params: model.params(filter="*.bias|*ln*"), weight_decay: 0.0}
        ]
    )

    scheduler: WarmupCosine(
        warmup_steps = CONFIG.warmup_steps,
        total_steps  = CONFIG.total_steps,
        min_lr       = CONFIG.lr / 10
    )

    step(batch):
        let logits = model.forward(batch.input_ids)                  # [B, S, V]
        # Next-token prediction: shift logits and labels
        loss = cross_entropy(
            logits[:, :-1, :].reshape([-1, CONFIG.vocab_size]),
            batch.input_ids[:, 1:].reshape([-1])
        )

    callbacks:
        on_step(step, loss):
            if step % 200 == 0:
                let ppl = loss.exp().item()
                print(f"step {step:>6} | loss {loss.item():.4f} | ppl {ppl:.1f} | lr {scheduler.get_lr():.2e}")

        on_step(step, _):
            if step % 10000 == 0 and step > 0:
                # Periodic validation
                @no_grad
                let val_loss = evaluate_loss(model, val_data, max_batches=100)
                print(f"  val_loss: {val_loss:.4f} | val_ppl: {val_loss.exp():.1f}")
                model.save(f"checkpoints/gpt2_step{step}.nslm")

print("Training complete!")
model.save("checkpoints/gpt2_final.nslm")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 5: QUANTIZATION — QAT with INT8 weights, FP8 activations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Starting quantization-aware training...")

let qat_model = quant(
    scheme       = int8,
    mode         = aware,
    granularity  = per_channel
):
    model = model

# Short QAT fine-tuning to recover quality
train(model=qat_model, epochs=1, precision=bf16):
    data:
        source     = train_data
        batch_size = CONFIG.batch_size

    optimizer: AdamW(lr=1e-5, weight_decay=0.0)
    scheduler: WarmupCosine(warmup_steps=100, total_steps=2000)

    step(batch):
        let logits = model.forward(batch.input_ids)
        loss = cross_entropy(
            logits[:, :-1, :].reshape([-1, CONFIG.vocab_size]),
            batch.input_ids[:, 1:].reshape([-1])
        )

# Convert to actual quantized model
let quantized_model = qat_model.convert()

# Validate quality
@no_grad
let fp_loss = evaluate_loss(model, val_data, max_batches=200)
@no_grad
let q_loss = evaluate_loss(quantized_model, val_data, max_batches=200)
print(f"FP32 val_loss: {fp_loss:.4f} | Quantized val_loss: {q_loss:.4f}")
print(f"Quality degradation: {q_loss - fp_loss:+.4f}")

quantized_model.save("checkpoints/gpt2_int8.nslm")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 6: EXPORT — ONNX and QNF (Quadric Native Format)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("Exporting models...")

# ONNX export (for broad deployment: TensorRT, ONNX Runtime, etc.)
to_onnx(
    quantized_model,
    input_shape  = [1, CONFIG.max_seq],
    output       = "exports/gpt2_int8.onnx",
    opset_version = 17
)
print("Exported to ONNX: exports/gpt2_int8.onnx")

# QNF export (for Quadric Chimera NPU deployment)
nsl.export.to_qnf(
    quantized_model,
    input_shape     = [1, CONFIG.max_seq],
    output          = "exports/gpt2_chimera.qnf",
    target          = npu<QuadricChimera>,
    optimize_tiling = true
)
print("Exported to QNF: exports/gpt2_chimera.qnf")

# Also save as safetensors for HuggingFace compatibility
save_safetensors(quantized_model.state_dict(), "exports/model.safetensors")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# STEP 7: BENCHMARK — Inference throughput measurement
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

print("\n=== Inference Benchmark ===\n")

# Benchmark FP32 model
let fp_perf = benchmark(model, input_shape=[1, CONFIG.max_seq], num_iters=100, warmup=10)
print(f"FP32 Model:")
print(f"  Throughput:  {fp_perf.tokens_per_second:,.0f} tok/s")
print(f"  Latency:     p50={fp_perf.latency_p50_ms:.1f}ms  p99={fp_perf.latency_p99_ms:.1f}ms")
print(f"  Memory peak: {fp_perf.memory_peak_mb:.0f} MB")

# Benchmark INT8 quantized model
let q_perf = benchmark(quantized_model, input_shape=[1, CONFIG.max_seq], num_iters=100, warmup=10)
print(f"\nINT8 Quantized Model:")
print(f"  Throughput:  {q_perf.tokens_per_second:,.0f} tok/s")
print(f"  Latency:     p50={q_perf.latency_p50_ms:.1f}ms  p99={q_perf.latency_p99_ms:.1f}ms")
print(f"  Memory peak: {q_perf.memory_peak_mb:.0f} MB")

# Speedup summary
let speedup = q_perf.tokens_per_second / fp_perf.tokens_per_second
let compression = fp_perf.memory_peak_mb / q_perf.memory_peak_mb
print(f"\nSpeedup: {speedup:.2f}x | Compression: {compression:.2f}x")
print(f"Model size: {model.size_mb():.0f}MB -> {quantized_model.size_mb():.0f}MB")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Helper function for evaluation
@no_grad
fn evaluate_loss(model: GPT2, data: Dataset, max_batches: int = -1) -> f32:
    let loader = DataLoader(dataset=data, batch_size=CONFIG.batch_size)
    let total_loss = 0.0
    let count = 0
    for (i, batch) in loader.enumerate():
        if max_batches > 0 and i >= max_batches:
            break
        let logits = model.forward(batch.input_ids)
        let loss = cross_entropy(
            logits[:, :-1, :].reshape([-1, CONFIG.vocab_size]),
            batch.input_ids[:, 1:].reshape([-1])
        )
        total_loss += loss.item()
        count += 1
    return total_loss / count
```

## What This Example Demonstrates

| Language Feature              | Where It Appears                                    |
|-------------------------------|-----------------------------------------------------|
| Type inference                | Throughout — most variables have inferred types      |
| Tensor type system            | Shape annotations on forward() signatures            |
| Named constants               | CONFIG struct with training hyperparameters           |
| Model keyword                 | GPT2Block and GPT2 model definitions                 |
| Nested models                 | GPT2 contains list<GPT2Block>                        |
| Weight tying                  | `@tie_weights(self.wte.weight)` on lm_head           |
| Pipe operator                 | `gelu \|> self.c_fc_proj \|> self.mlp_drop`           |
| Train block DSL               | Declarative training with data/optimizer/scheduler    |
| Gradient accumulation         | `accumulate = 4` in train config                     |
| Mixed precision               | `precision = bf16` in train config                   |
| Gradient clipping             | `clip_grad_norm = 1.0` in train config               |
| Parameter groups              | Weight decay exclusion for biases and norms           |
| @no_grad                      | Evaluation function and inline validation             |
| Quantization-aware training   | `quant(scheme=int8, mode=aware)` block               |
| Quantized model conversion    | `qat_model.convert()`                                |
| ONNX export                   | `to_onnx()` one-liner                                |
| QNF export                    | `to_qnf()` for Quadric Chimera NPU                  |
| Safetensors export            | `save_safetensors()` for HuggingFace compat          |
| Benchmarking                  | `benchmark()` for throughput/latency/memory           |
| Tokenizer definition          | `tokenizer gpt_tokenizer(...)` block                 |
| Dataset definition            | `dataset train_data(...)` block                      |
| Memory-mapped data            | `MemoryMapped()` for zero-copy data loading          |
| Sequence packing              | `packing = true` in dataset config                   |
| Closures                      | Callbacks in train block                             |
| Destructuring                 | `let B, S, D = x.shape`                             |
| Native serialization          | `model.save()` / `model.load()`                      |
| f-strings                     | Throughout for formatted output                      |
| Conditional expressions       | Tokenizer load-or-train logic                        |
