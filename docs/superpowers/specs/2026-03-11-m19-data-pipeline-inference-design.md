# M19: Data Pipeline + Inference Sampling — Design Spec

**Goal:** Load a dataset, train a model, generate text with sampling — the complete ML workflow.

**Scope:** Core production data pipeline (JSONL, CSV, memory-mapped binary), multi-threaded DataLoader with batching/shuffling/sequence packing, and inference sampling primitives (topk, multinomial, argmax) with stdlib generate() function.

**Deferred to later milestones:** Parquet, Arrow, HuggingFace streaming data sources, image transforms, `nsl pkg`, `nsl doc`, KV-cache optimization, distributed sampler.

---

## 1. Runtime Tensor Primitives

### 1.1 topk

**FFI:** `nsl_tensor_topk(tensor_ptr: i64, k: i64, dim: i64) -> i64`

- Returns `NslDict` with keys `"values"` (same dtype as input) and `"indices"` (i64 dtype)
- Algorithm: min-heap of size k, O(n log k) per row
- Supports negative dim (dim=-1 = last axis)
- Operates per-row for 2D input

### 1.2 multinomial

**FFI:** `nsl_tensor_multinomial(tensor_ptr: i64, num_samples: i64) -> i64`

- Input: 1D or 2D probability tensor (does NOT need to sum to exactly 1.0)
- Algorithm: cumulative sum + binary search (inverse transform sampling)
- **CDF self-normalization:** divides sampled uniform random value by actual CDF total (last element), eliminating floating-point drift issues from temperature scaling
- Returns i64-dtype index tensor
- Uses deterministic thread-local RNG (see 1.4)

### 1.3 argmax

**FFI:** `nsl_tensor_argmax(tensor_ptr: i64, dim: i64) -> i64`

- Returns i64-dtype index tensor of maximum values along dimension
- Linear scan, O(n)
- Used for greedy decoding

### 1.4 Deterministic RNG

**FFI:** `nsl_manual_seed(seed: i64)`

- Sets a thread-local `StdRng` (ChaCha-based) stored in `RefCell`
- Same pattern as autodiff `TAPE` thread-local
- Default seed = 0 for reproducible tests
- All sampling operations (`multinomial`) draw from this RNG

### 1.5 cumsum

**FFI:** `nsl_tensor_cumsum(tensor_ptr: i64, dim: i64) -> i64`

- Prefix sum along dimension
- Returns new tensor with same shape/dtype
- Required for nucleus (top-p) sampling

### 1.6 lt_scalar (float comparison mask)

**FFI:** `nsl_tensor_lt_scalar(tensor_ptr: i64, scalar: f64) -> i64`

- Returns f32/f64 tensor with 1.0 where element < scalar, 0.0 otherwise
- Avoids need for boolean tensor dtype
- Used in top-p sampling to mask beyond nucleus threshold

### 1.7 Slice Assignment

**FFI:** `nsl_tensor_slice_assign(target_ptr: i64, src_ptr: i64, dims_ptr: i64, num_dims: i64)`

```rust
#[repr(C)]
pub struct NslSliceDim {
    pub is_scalar: bool,  // true = single index, false = range
    pub start: i64,
    pub end: i64,         // ignored if is_scalar
}
```

- N-dimensional slice mutation: `tensor[0, a:b] = other`
- Compiler builds array of `NslSliceDim` from index expression AST
- Required for pre-allocated generate() buffer pattern

### 1.8 NslTensor `owns_data` Flag

**Modification to existing struct:**

```rust
#[repr(C)]
pub struct NslTensor {
    pub data: *mut c_void,
    pub shape: *mut i64,
    pub ndim: i64,
    pub len: i64,
    pub refcount: i64,
    pub device: u8,
    pub dtype: u8,
    pub owns_data: u8,  // NEW: 0 = borrowed (mmap), 1 = owned (heap)
}
```

- `nsl_tensor_free` / `nsl_tensor_release` check `owns_data` before deallocating `.data`
- Prevents segfault when freeing mmap'd tensor data
- Default: `owns_data = 1` for all existing tensor creation paths

---

## 2. Data Source Runtime

**New file:** `crates/nsl-runtime/src/data_source.rs`

### 2.1 JSONL Loader

**FFI:** `nsl_load_jsonl(path_ptr: i64, path_len: i64, field_ptr: i64, field_len: i64) -> i64`

- Reads `.jsonl` file, extracts named field from each JSON object
- Returns `NslList` of string values
- Uses `serde_json` for parsing
- Skips malformed lines with warning
- **Role: Small-data utility** — for eval sets, prompt lists (thousands of rows). Not for multi-GB training data.

### 2.2 CSV Loader

**FFI:** `nsl_load_csv(path_ptr: i64, path_len: i64, col_idx: i64, has_header: i64) -> i64`

- Reads CSV file, extracts column by index
- Returns `NslList` of string values
- Hand-rolled parser (split on commas, handle quoted fields)
- **Role: Small-data utility** — same as JSONL

### 2.3 Memory-Mapped Binary Loader

**FFI:** `nsl_load_mmap(path_ptr: i64, path_len: i64, dtype: i64) -> i64`

- Memory-maps binary file as flat 1D tensor using `memmap2` crate
- `dtype`: 0=f64, 1=f32, 2=i32, 3=u16
- Returns `NslTensor` with `owns_data = 0` (data points into mmap region)
- Mmap handle leaked intentionally (lives for process lifetime)
- **u16 dtype (3):** Primary pathway for pre-tokenized LLM datasets. Vocabulary sizes (e.g., GPT-2 = 50,257) fit in u16. Halves disk size and PCIe bandwidth vs i32.
- **Role: Heavy-duty training pathway** — zero-copy, O(1) random access

---

## 3. DataLoader Runtime

**New file:** `crates/nsl-runtime/src/dataloader.rs`

### 3.1 Core Struct

```rust
struct DataLoader {
    data: *const u8,              // source data (mmap pointer)
    data_len: usize,              // total elements
    element_size: usize,          // bytes per element (2 for u16, 4 for i32, etc.)
    batch_size: usize,
    seq_len: usize,
    shuffle: bool,
    num_workers: usize,
    prefetch: usize,
    pin_memory: bool,
    drop_last: bool,
    packing: bool,
    pack_separator: i64,          // EOS token id
    // internal state
    cursor: Arc<AtomicUsize>,     // shared cursor into token stream
    reorder_buffer: Arc<Mutex<HashMap<usize, BatchData>>>,
    expected_batch_id: AtomicUsize,
    condvar: Arc<Condvar>,
    worker_handles: Vec<JoinHandle<()>>,
    epoch: usize,
    total_batches: usize,         // data_len / (batch_size * seq_len)
}
```

### 3.2 FFI Surface

| Function | Purpose |
|----------|---------|
| `nsl_dataloader_create(data_ptr, data_len, config_ptr, config_len) -> dl_ptr` | Parse JSON config, allocate struct |
| `nsl_dataloader_start(dl_ptr)` | Spawn N worker threads, begin filling reorder buffer |
| `nsl_dataloader_next_batch(dl_ptr) -> dict_ptr` | Block until expected batch ready. Returns `NslDict` with `input_ids` [B,S], `labels` [B,S], `attention_mask` [B,S,S]. Returns 0 (null) at epoch end. |
| `nsl_dataloader_reset(dl_ptr)` | Reset cursor to 0, shuffle if enabled, increment epoch |
| `nsl_dataloader_stop(dl_ptr)` | Signal workers to stop, join threads |
| `nsl_dataloader_free(dl_ptr)` | Free all resources |

### 3.3 Threading Model

- N worker threads share `Arc<AtomicUsize>` cursor + `Arc<Mutex<HashMap>>` reorder buffer
- Each worker: atomically claim `batch_id`, compute cursor offset, slice data, build batch, insert into reorder buffer keyed by `batch_id`, notify condvar
- Main thread `next_batch()`: lock reorder buffer, check if `expected_batch_id` present. If yes, remove and return. If no, wait on condvar. Increment `expected_batch_id` after return.
- **Deterministic ordering guaranteed** by reorder buffer pattern

### 3.4 Batch Construction

**Without packing:** Workers slice `batch_size` consecutive sequences of `seq_len` tokens. Labels = input_ids shifted right by 1, last position = -100. Attention mask = standard lower-triangular causal mask.

**With packing (continuous stream chunking):** Workers slice `batch_size * seq_len` tokens from stream. Scan for `pack_separator` to assign `doc_id` per position. Build block-diagonal mask: `mask[i][j] = 0.0` if `doc_id[i] == doc_id[j] AND j <= i`, else `-1e9`. Labels at document boundaries (where `input_ids[i] == pack_separator`) set to `-100`.

### 3.5 Pin Memory

- If `pin_memory=true` and CUDA available: allocate batch tensors via `cuMemAllocHost` (driver API)
- Feature-gated: `#[cfg(feature = "cuda")]`
- Falls back to regular heap allocation when CUDA unavailable
- Enables async DMA transfer to GPU via `cuMemcpyHtoDAsync`

### 3.6 Shuffle

- Fisher-Yates shuffle on sequence-level indices at each `reset()`
- For packing mode: shuffle changes the cursor start position using a permuted offset table
- Not token-level shuffle — preserves document integrity within sequences

### 3.7 Memory Footprint

At B=32, S=1024: attention mask = 32 * 1024 * 1024 * 4 bytes = ~134 MB per batch. With prefetch=2, total mask overhead ~268 MB. Acceptable for production training.

---

## 4. Sequence Packing

**New file:** `crates/nsl-runtime/src/packing.rs`

### 4.1 Continuous Stream Chunking

**NOT bin-packing.** The entire dataset is treated as one massive 1D token stream joined by EOS separators. To create a batch:

1. Slice `B * S` tokens from the stream starting at cursor
2. Scan for separator tokens to assign `doc_id` per position
3. Build block-diagonal attention mask from doc_ids
4. Set `labels[i] = -100` where `input_ids[i] == separator` (prevent cross-document prediction)
5. Advance cursor by `B * S`

**Properties:**
- Zero data waste — no truncation, no padding, 100% packing efficiency
- Long documents naturally span multiple chunks — model trains on all content
- O(B*S) per batch — no pre-scanning or bin-packing heuristics
- Cursor wraps to 0 at epoch boundary

### 4.2 Block-Diagonal Attention Mask

For each sequence in the batch:
```
mask[i][j] = 0.0   if doc_id[i] == doc_id[j] AND j <= i   (same doc, causal)
mask[i][j] = -1e9   otherwise                                (cross-doc or future)
```

This is an **additive mask** applied before softmax: `attn_weights = attn_scores + mask`

### 4.3 Packing Efficiency

**FFI:** `nsl_packing_efficiency(dl_ptr: i64) -> f64`

With continuous stream chunking, this always returns 1.0 (no padding tokens ever allocated). Provided as API for compatibility with the spec.

---

## 5. Compiler Intrinsics & Codegen

### 5.1 Semantic Builtins (`builtins.rs`)

9 new entries in `register_builtins()`:

| Name | Param Types | Return Type |
|------|------------|-------------|
| `load_jsonl` | `Unknown, Unknown` | `List<Str>` |
| `load_csv` | `Unknown, Unknown` | `List<Str>` |
| `load_mmap` | `Unknown, Unknown` | `Tensor` |
| `DataLoader` | `Unknown` (variadic) | `Unknown` (opaque) |
| `topk` | `Unknown, Unknown` | `Dict<Str, Tensor>` |
| `multinomial` | `Unknown, Unknown` | `Tensor` |
| `argmax` | `Unknown, Unknown` | `Tensor` |
| `manual_seed` | `Unknown` | `Void` |
| `cumsum` | `Unknown, Unknown` | `Tensor` |

### 5.2 Codegen Intrinsic Handlers (`expr.rs`)

Each follows the M18b pattern:
1. Match `func_name`
2. Compile argument expressions
3. Extract string literals where needed (compile-time path extraction)
4. `compile_call_by_name(builder, "nsl_ffi_name", &[args])`

### 5.3 DataLoader Codegen

**Construction:** `DataLoader(data, batch_size=32, ...)` compiles to:
1. Serialize keyword args to JSON config string at compile time
2. `nsl_dataloader_create(data_ptr, data_len, config_ptr, config_len)`
3. `nsl_dataloader_start(dl_ptr)`
4. Store `dl_ptr` as local variable

**Loop:** `for batch in loader:` compiles to:
```
loop_start:
    batch_ptr = nsl_dataloader_next_batch(dl_ptr)
    if batch_ptr == 0: jump loop_end
    ... loop body ...
    nsl_dict_free(batch_ptr)    // CRITICAL: prevent per-step memory leak
    jump loop_start
loop_end:
    nsl_dataloader_reset(dl_ptr)  // prepare for next epoch
```

**Scope teardown:** When DataLoader-typed variables go out of scope:
```
nsl_dataloader_stop(dl_ptr)
nsl_dataloader_free(dl_ptr)
```

### 5.4 Slice Assignment Codegen

`tensor[i, a:b] = value` compiles to:
1. Build `NslSliceDim` array on stack from index expression AST
2. `nsl_tensor_slice_assign(target_ptr, src_ptr, dims_ptr, num_dims)`

### 5.5 New Method Dispatch

- `.cumsum(dim)` on tensor -> `nsl_tensor_cumsum`
- `.shape(dim)` on tensor -> `nsl_tensor_shape_dim` (returns single i64 dim size)

---

## 6. Standard Library

### 6.1 `stdlib/nsl/data/loader.nsl`

Documentation module. All data functions are compiler intrinsics available without import:
- `load_jsonl(path, field)`, `load_csv(path, col_idx)`, `load_mmap(path, dtype)`
- `DataLoader(data, batch_size=..., shuffle=..., num_workers=..., ...)`

### 6.2 `stdlib/nsl/inference/sampling.nsl`

```nsl
from nsl.inference.sampling import sample_top_k, sample_top_p, sample_greedy

fn sample_top_k(logits: Tensor, k: int, temperature: float) -> Tensor:
    let scaled = logits / temperature
    let result = topk(scaled, k)
    let probs = softmax(result["values"], dim=-1)
    let sampled = multinomial(probs, 1)
    return result["indices"].gather(-1, sampled)

fn sample_top_p(logits: Tensor, p: float, temperature: float) -> Tensor:
    let scaled = logits / temperature
    let probs = softmax(scaled, dim=-1)
    let result = topk(probs, probs.shape(-1))
    let cumulative = result["values"].cumsum(dim=-1)
    let mask = lt_scalar(cumulative - result["values"], p)
    let filtered = result["values"] * mask
    let sampled = multinomial(filtered, 1)
    return result["indices"].gather(-1, sampled)

fn sample_greedy(logits: Tensor) -> Tensor:
    return argmax(logits, dim=-1)
```

### 6.3 `stdlib/nsl/inference/generate.nsl`

```nsl
from nsl.inference.sampling import sample_top_k

fn generate(model, prompt: Tensor, max_tokens: int,
            temperature: float, top_k: int) -> Tensor:
    let prompt_len = prompt.shape(-1)
    let total_len = prompt_len + max_tokens
    let tokens = zeros([1, total_len])
    tokens[0, 0:prompt_len] = prompt

    no_grad:
        for i in range(prompt_len, total_len):
            let current_view = tokens[0, 0:i]
            let logits = model.forward(current_view)
            let next_logit = logits.select(-2, -1)
            let next_token = sample_top_k(next_logit, top_k, temperature)
            tokens[0, i] = next_token

    return tokens
```

**Key:** `no_grad:` block prevents autodiff tape from recording 1000+ forward passes during generation, avoiding VRAM OOM.

---

## 7. Dependencies

### 7.1 New Cargo Dependencies (`nsl-runtime/Cargo.toml`)

| Crate | Purpose | Feature Gate |
|-------|---------|-------------|
| `serde_json` | JSONL parsing + DataLoader config deserialization | Always |
| `memmap2` | Cross-platform memory-mapped file access | Always |
| `rand` | Deterministic RNG for multinomial sampling | Always |

### 7.2 Linker

No changes to `linker.rs` — all new dependencies compile into `libnsl_runtime.a` via Cargo.

---

## 8. Deliverable

An NSL program that:
1. Loads a pre-tokenized binary dataset via `load_mmap`
2. Creates a `DataLoader` with batching, shuffling, multi-threaded workers, and sequence packing
3. Trains a model using the `train` block (M14) with packed batches
4. Generates text using `generate()` with top-k sampling
5. Demonstrates deterministic output via `manual_seed()`
