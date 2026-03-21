# NeuralScript: Inference & Serving Features

## Overview

NSL treats inference as a first-class deployment target with features spanning attention optimization, memory management, model parallelism, and continuous batching. The design philosophy is "inference-first" — NSL's compile-time advantages matter most for production inference, where Python's dynamic overhead is most costly.

---

## PagedAttention & KV-Cache (M25)

**Runtime**: `paged_kv/` module

Implements vLLM-style paged KV-cache management:

- **Block allocation**: KV-cache stored in fixed-size blocks (not contiguous per-sequence)
- **Page table**: Maps (sequence, layer, position) → physical block
- **Copy-on-Write (CoW)**: Shared prefixes between sequences share KV blocks; copied on divergence
- **Memory watermark**: Configurable high/low watermarks for preemption decisions
- **Per-layer management**: Each transformer layer has independent KV-cache pages

This eliminates memory fragmentation from variable-length sequences and enables efficient beam search / parallel sampling.

---

## FlashAttention-2 (M27)

Production-quality tiled attention (~2,600 lines):

- **Online softmax**: Numerically stable attention without materializing the full S×S attention matrix
- **Tiled computation**: Processes attention in tiles that fit in GPU shared memory / registers
- **Hopper wgmma.mma_async** (sm_90+): Asynchronous matrix multiply with TMA loads, warp specialization (producer/consumer), and pingpong scheduling to overlap softmax with tensor core compute
- **Ampere mma.sync** (sm_80): `mma.sync.aligned.m16n8k16` fallback for A100-class GPUs
- **Paged KV integration**: Reads KV from paged blocks via block table lookup
- **RoPE fusion**: Rotary positional embedding applied in-register (both half-split and adjacent layouts)
- **GQA**: Grouped query attention with configurable head mapping (multiple Q heads per KV head)
- **Tree causal mask**: Supports non-contiguous causal masks for speculative decoding token trees
- **21+ kernel variants**: Parameterized by head dim, block size, causal mode
- **Logsumexp saving**: For correct backward pass gradient computation
- **Fallback path**: Scalar FMA for GPUs without tensor cores

Activated via decorator:
```
@flash_attention
fn attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    return scaled_dot_product_attention(q, k, v)
```

---

## Continuous Batching & Serving Engine (M29)

### Serve Block
```
serve inference_server:
    model = MyModel()
    max_batch_size = 64
    max_seq_len = 4096
    kv_cache_blocks = 2048

    fn handle(request: Request) -> Response:
        return model.generate(request.tokens, max_new_tokens=256)
```

The `serve` block is a declarative DSL for building inference servers:

- **Request scheduler**: Manages incoming request queue with priority
- **Chunked prefill**: Processes long prompts in chunks interleaved with decode steps
- **Iteration-level batching**: Adds/removes sequences from the batch at each decode step
- **Preemption**: Evicts low-priority sequences when memory pressure exceeds watermark
- **Ragged tensors**: Handles variable-length sequences within a single batch (no padding waste)

### Serving Runtime
Located in `nsl-runtime/src/serving/`:
- Request lifecycle management (queued → running → completed/preempted)
- Scheduling policies (FCFS with priority)
- Batch assembly from heterogeneous sequence lengths

---

## Speculative Decoding (M33)

**Codegen**: `speculative.rs`

Accelerates autoregressive generation by using a small "draft" model to propose multiple tokens, then verifying them in parallel with the full model:

```
@speculative(draft_model=DraftModel, num_tokens=5, temperature=0.8)
fn generate(model: MainModel, prompt: Tensor) -> Tensor:
    ...
```

Configuration fields:
- `draft_model`: Smaller model for fast token proposals
- `num_tokens`: Number of speculative tokens per step
- `temperature`: Sampling temperature for draft
- `tree_width`: Width of token tree (Medusa-style)
- `@medusa` variant for multi-head prediction

The FlashAttention implementation includes tree-structured causal masks to support verification of speculative token trees.

---

## Mixture of Experts (M32)

**Codegen**: `moe.rs`, `moe_kernels.rs`
**Runtime**: `moe/` module

```
@moe(num_experts=8, top_k=2, capacity_factor=1.25, aux_loss_coeff=0.01)
model MoELayer:
    experts: [FFN; 8]
    gate: Linear

    fn forward(self, x: Tensor) -> Tensor:
        ...
```

Implementation:
- **Router**: Softmax-based top-k routing with capacity tracking
- **Load balancing**: Auxiliary loss (importance + load) to prevent expert collapse
- **Token dispatch**: Scatter/gather operations for routing tokens to selected experts
- **Capacity factor**: Controls maximum tokens per expert (overflow → dropped or routed to fallback)

---

## Ring Attention / Context Parallelism (M34)

**Codegen**: `context_parallel.rs`
**Runtime**: `context_parallel/`

Distributes long sequences across multiple GPUs using ring communication:

```
@context_parallel(ring_size=4)
fn long_context_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    ...
```

- **Ring topology**: Each GPU holds a chunk of the sequence; KV blocks rotate around the ring
- **Double buffering**: Overlaps communication with computation (send current KV while computing with received KV)
- **Causal masking**: Handles causal attention across sequence chunks
- **CPU fallback**: Simulates ring communication for testing without multi-GPU

---

## Tensor Parallelism (M30)

**Codegen**: `tensor_parallel.rs`
**Runtime**: `tensor_parallel/`

Splits model weights across GPUs:

```
@shard(dim=0, devices=4)
let weight: Tensor<[hidden, ffn], f16>
```

- **@shard decorator**: Specifies which dimension to split and across how many devices
- **All-reduce**: Gradient synchronization across shards (NCCL stubs)
- **Worker-per-GPU threads**: Each GPU runs in its own OS thread with dedicated CUDA context
- **Multi-GPU KV-cache**: Paged KV-cache distributed across tensor-parallel ranks

---

## Pipeline Parallelism (M43)

**Runtime**: `pipeline/`

Splits model layers across GPUs in a pipeline:

- **1F1B schedule**: One-forward-one-backward scheduling with correct warmup/steady/cooldown phases
- **Stage staggering**: Each pipeline stage offset by one micro-batch
- **Activation send/recv**: Forward activations sent to next stage, gradients sent backward
- **GPipe variant**: Available as alternative scheduling strategy
- **3D parallelism**: Composable with tensor parallelism and data parallelism

---

## Disaggregated Inference (M41)

**Runtime**: `disaggregated/`

Separates prefill and decode phases onto different hardware:

- **Router**: Dispatches requests to prefill or decode worker pools
- **Prefill workers**: Optimized for compute-bound prompt processing (high FLOP/s GPUs)
- **Decode workers**: Optimized for memory-bound token generation (high bandwidth GPUs)
- **KV transfer**: Mechanism for sending computed KV-cache from prefill to decode workers
- **Scheduling policies**: LeastLoaded, RoundRobin (prefill); LeastLoaded, MemoryAware (decode)

---

## KV-Cache Compression (M42)

**Runtime**: `kv_compress/`

Reduces KV-cache memory footprint:

| Scheme | Description |
|--------|-------------|
| INT8 per-head | Quantize KV to INT8 with per-head scale factors |
| INT8 per-token | Quantize with per-token scale factors |
| INT4 per-group | 4-bit quantization with group scaling |
| FP8 | 8-bit floating point (E4M3) |

Additional compression strategies:
- **H2O (Heavy Hitter Oracle)**: Evicts KV entries with lowest cumulative attention scores, preserving "sink tokens" (first few tokens that receive disproportionate attention)
- **Sliding window**: Only retains recent KV entries within a fixed window size

---

## Constrained Decoding / Structured Generation (M44)

**Codegen**: `grammar_compiler.rs`
**Runtime**: `grammar.rs`

Compiles grammar specifications into finite state machines for constrained token generation:

**Compilation pipeline**:
1. Grammar (BNF or regex) → NFA (Thompson construction)
2. NFA → DFA (subset construction with epsilon closure)
3. DFA → Minimized DFA (Hopcroft's algorithm)
4. Minimized DFA → Token-level FSM (per-state bitmask for logit masking)

**Runtime execution**:
- CSR-compressed DFA stored in `GrammarFSM`
- Binary search on sorted token transitions for O(log n) stepping
- Per-state bitmasks for efficient logit masking during sampling
- `is_valid_token()` and `step()` for integration with generation loop

This enables guaranteed-valid JSON, SQL, code, or any grammar-conformant output at near-zero overhead (compiled FSM, not interpreted regex).

---

## Dynamic Shapes (M28)

**Codegen**: `dynamic_shapes.rs`

Handles variable-length inputs without recompilation:

- **Symbolic dimensions**: Named dimensions that resolve at runtime (e.g., `seq_len`)
- **Bounded dimensions**: Known maximum with runtime actual (enables static memory allocation)
- **Stride-based codegen**: Tensor metadata includes strides, enabling reshape/transpose without copy
- **Ragged tensors**: Variable-length sequences packed contiguously (no padding)

---

## Standalone Deployment (M24)

```bash
nsl build --standalone model.nsl --weights model.safetensors -o server
./server  # zero dependencies, no Python, no CUDA toolkit
```

The standalone binary contains:
- Compiled model code (native x86-64 or ARM)
- Embedded or sidecar weight files
- The nsl-runtime static library
- Optional: paged KV-cache, continuous batching, FlashAttention kernels

Only requirement: NVIDIA GPU driver (if using CUDA).

---

## Text Generation

### Generation Loop (stdlib)
```
import nsl.inference.generate
import nsl.inference.sampling

let tokens = generate(model, prompt_tokens,
    max_new_tokens=256,
    temperature=0.8,
    top_k=50,
    top_p=0.95)
```

### Sampling Strategies
- **Temperature scaling**: Divide logits by temperature before softmax
- **Top-k**: Keep only top k highest-probability tokens
- **Top-p (nucleus)**: Keep smallest set of tokens whose cumulative probability ≥ p
- **Greedy**: Always select highest-probability token (temperature=0)

### Tokenization
BPE tokenizer via HuggingFace `tokenizers` crate (Rust, not Python):
```
import nsl.tokenize.tokenizer

let tok = Tokenizer("path/to/tokenizer.json")
let ids = tok.encode("Hello, world!")
let text = tok.decode(ids)
```
