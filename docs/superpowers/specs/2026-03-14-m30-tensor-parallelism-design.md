# M30: Tensor Parallelism & NCCL — Design Specification

**Status:** Approved
**Date:** 2026-03-14
**Dependencies:** M17 (GPU/CUDA), M25 (PagedAttention), M27 (FlashAttention), M29 (Continuous Batching)

## Goal

Split large models across 2-8 consumer GPUs using Megatron-LM-style tensor parallelism, enabling inference of models that exceed single-GPU VRAM. A 70B FP8 model (~75GB) should run across 4x 24GB GPUs with automatic weight sharding, per-GPU KV-cache, and NCCL collective communication.

## Architecture Overview

**Execution model:** Process-based SPMD (Single Program, Multiple Data). Each GPU gets its own OS process running the same Cranelift-compiled binary. Processes synchronize only at collective communication points (AllReduce, AllGather, Broadcast). No master-worker channel dispatch — threads run at full speed independently.

**Collective backend:** Trait-based abstraction (`CollectiveBackend`) with two implementations:
- `NcclBackend` — real NCCL via dynamic linking (Linux, feature-gated `#[cfg(feature = "nccl")]`)
- `SimulatedBackend` — CPU-based shared-memory collectives for Windows/testing

**Language surface:** `@shard(dim=D)` decorator on model layer declarations. Hardware-agnostic — device count comes from `--devices N` CLI flag, not from source code.

---

## Section 1: Runtime Module Structure

### Module Layout

New module: `crates/nsl-runtime/src/tensor_parallel/`

| File | Responsibility |
|---|---|
| `mod.rs` | Module declarations, re-exports |
| `collective.rs` | `CollectiveBackend` trait + `SimulatedBackend` impl |
| `nccl.rs` | `NcclBackend` impl (feature-gated `#[cfg(feature = "nccl")]`) |
| `sharding.rs` | Mmap-based weight loading — returns normal `NslTensor` for current rank |
| `ffi.rs` | `#[no_mangle] pub extern "C"` functions for codegen |

No `executor.rs` — process spawning lives in the CLI crate.

### CollectiveBackend Trait

```rust
use std::ffi::c_void;

/// Dtype uses the existing runtime u16 ID space (see tensor.rs):
///   DTYPE_F64 = 0, DTYPE_F32 = 1
/// Passed as i64 over FFI boundaries. The CollectiveBackend uses
/// the raw u16 value to determine element byte width and to map
/// to ncclDataType_t in the NcclBackend.
pub type DtypeId = u16;

pub const DTYPE_F64: DtypeId = 0;
pub const DTYPE_F32: DtypeId = 1;
pub const DTYPE_F16: DtypeId = 2;
pub const DTYPE_BF16: DtypeId = 3;
pub const DTYPE_I8: DtypeId = 4;
pub const DTYPE_FP8: DtypeId = 5;
// Custom BYOD dtypes start at 256

/// Stream handle — `*mut c_void` unconditionally (opaque pointer).
/// Resolves to CUstream inside #[cfg(feature = "cuda")] code.
/// SimulatedBackend ignores this parameter.
pub type StreamHandle = *mut c_void;

pub trait CollectiveBackend: Send + Sync {
    fn all_reduce_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32;

    fn all_gather(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        send_count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32;

    fn broadcast(
        &self,
        buf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        root_rank: i32,
        stream: StreamHandle,
    ) -> i32;

    fn barrier(&self) -> i32;
    fn rank(&self) -> i32;
    fn world_size(&self) -> i32;
}
```

Key design decisions:
- **Dtype uses existing `u16` ID space** from `tensor.rs` (`DTYPE_F64=0`, `DTYPE_F32=1`, etc.) — no new enum, avoids ID mismatch with existing runtime. Includes F64, FP8, and BYOD custom dtypes (256+).
- **`StreamHandle = *mut c_void`** unconditionally — avoids compile failure on Windows where `CUstream` type doesn't exist. Cast to real `CUstream` only inside `#[cfg(feature = "cuda")]` blocks.
- `*const c_void` for sendbuf (read-only), `*mut c_void` for recvbuf — correct mutability semantics
- `all_reduce_sum` supports in-place (same pointer for send/recv, same output size)
- `all_gather` is **out-of-place only** — recvbuf must be `world_size * send_count` elements. In-place would corrupt adjacent GPU memory.
- `reduce_scatter` intentionally omitted — M30 uses `all_reduce` for row-parallel and `all_gather` for LM head. `reduce_scatter + all_gather` is an optimization for compute/communication overlap, deferred to M34.

### SimulatedBackend

For Windows/testing. All ranks run as separate OS processes sharing the same physical GPU(s).

- Uses shared-memory (file-backed mmap) for cross-process communication
- **Shared memory path convention:** `{TEMP}/nsl_tp_{world_size}_{pid_of_parent}.shm` — parent PID ensures isolation between concurrent runs. Parent creates the file before spawning children; children open it by reading `NSL_TP_SHM_PATH` env var. Cleaned up by parent after all children exit.
- **Shared memory layout:** Header (generation counter, arrival counter, world_size) + per-rank data slots. Total size: `header(64B) + world_size * max_collective_bytes`. `max_collective_bytes` is the largest tensor that will pass through a collective — conservatively set to 64MB for M30 (sufficient for LM head logits at batch=256, vocab=128K, f32).
- `all_reduce_sum`: each process writes its data to its rank slot, increments arrival counter. Last to arrive performs the reduction across all slots and writes result to slot 0, then increments generation counter. All processes spin on generation counter, then copy result from slot 0 to their recvbuf. Double-buffering via generation counter (even/odd) prevents use-after-overwrite between consecutive collectives.
- `all_gather`: each process writes its shard to its rank slot, barrier, then all read the full concatenated result from slots 0..world_size
- `broadcast`: root writes to slot 0, barrier, others read from slot 0
- `barrier`: generation-based — processes increment arrival counter, spin until count == world_size, then increment generation

### SPMD Execution Model

Each process:
1. Reads `NSL_LOCAL_RANK` and `NSL_WORLD_SIZE` from environment
2. Binds to GPU via modulo mapping: `physical_id = rank % cuDeviceGetCount()`
3. Creates its own `CollectiveBackend` instance
4. Runs the same Cranelift-compiled `nsl_main()` function
5. Synchronizes only at collective calls (implicit barriers via NCCL stream semantics)

Thread-local rank/world_size exposed via FFI: `nsl_tp_rank() -> i64`, `nsl_tp_world_size() -> i64`.

---

## Section 2: Weight Sharding

### Mmap-Based Loading

```rust
/// The Rust-side API. rank and world_size are read from env vars
/// internally (NSL_LOCAL_RANK, NSL_WORLD_SIZE), matching the SPMD model.
pub fn load_sharded_weight(
    mmap: &Mmap,           // memory-mapped safetensors file
    tensor_name: &str,     // e.g. "layers.0.attn.q_proj.weight"
    shape: &[usize],       // full tensor shape
    shard_dim: usize,      // which dim to split
    dtype: DtypeId,
) -> NslTensor  // normal tensor, just smaller
```

**Mmap lifecycle:**
- `nsl_tp_mmap_open(path_ptr, path_len) -> i64` — opens and mmaps the safetensors file, returns opaque handle. Called once per process at model load time.
- `nsl_tp_load_sharded_weight(mmap_handle, name_ptr, name_len, shape_ptr, ndims, shard_dim, dtype) -> i64` — loads this rank's slice from the mmap. Rank/world_size read from env internally.
- `nsl_tp_mmap_close(mmap_handle) -> i64` — unmaps the file after all weights are loaded to GPU.
- The mmap handle is kept alive between `open` and `close`. After `close`, GPU memory holds the weight data independently.

**How it works:**
1. Process mmaps the safetensors file (virtual memory, near-zero RSS)
2. Calculates byte offset for its rank's slice along `shard_dim`
3. For `shard_dim == 0`: contiguous slice — single `cudaMemcpyHostToDevice` from mmap pointer
4. For `shard_dim > 0`: strided copy — row-by-row `cudaMemcpyHostToDevice`
5. Returns a normal `NslTensor` with reduced shape — compiled code doesn't know it's a shard

**Why mmap:** If 4 processes each fully load a 20GB safetensors file to RAM, that's 80GB RSS to extract 5GB per process. With mmap, each process maps the file (near-zero RSS), seeks to its offset, and copies only its slice to GPU. Total CPU RAM: ~0.

### Compile-Time Validation

During `compile_model_constructor()`, when `@shard(dim=D)` is found on a layer:
- Check `shape[D] % world_size == 0` — compile error if not: `"Cannot shard dim 0 of size 32001 across 4 devices"`
- Store `ShardInfo { dim }` in `shard_configs` for codegen phase

### Collective Insertion Rules

Static analysis during model forward pass codegen. The compiler determines which collective to insert based on `@shard` args and layer position:

| Decorator | Layer type | Post-op collective | Rationale |
|---|---|---|---|
| `@shard(dim=0)` | Embedding | `all_reduce_sum` | Out-of-range GPUs produce zeros; sum recovers |
| `@shard(dim=0)` | Linear (column-parallel) | None | Shards flow to next row-parallel layer |
| `@shard(dim=1)` | Linear (row-parallel) | `all_reduce_sum` | Partial sums combined |
| `@shard(dim=0)` | Linear (LM head / terminal) | `all_gather` (out-of-place) | Full logits needed for sampling |
| (none) | LayerNorm, RMSNorm | None | Replicated, no collective |

---

## Section 3: Codegen — `@shard` Extraction, Activation State, Collective Emission

### Compiler Context Extension

```rust
pub struct ShardInfo {
    pub dim: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DistState {
    Replicated,
    Sharded { dim: usize },
}

// Added to Compiler struct:
pub shard_configs: HashMap<String, ShardInfo>,
pub activation_states: HashMap<String, DistState>,
```

### Activation State Tracking

The compiler tracks the distributed state of intermediate tensors throughout the forward pass, not just the static weights. This prevents mathematically invalid TP graphs from compiling silently.

**State transitions:**

1. At model entry, all inputs are `Replicated`
2. Column-parallel Linear (`@shard(dim=0)`):
   - Input must be `Replicated` — compile error if `Sharded`
   - Output becomes `Sharded { dim: 1 }` (columns split across GPUs)
   - No collective emitted
3. Row-parallel Linear (`@shard(dim=1)`):
   - Input must be `Sharded` — compile error if `Replicated`: `"Row-parallel Linear expected sharded input, found replicated tensor"`
   - Emit `nsl_tp_all_reduce_sum` after matmul
   - Output becomes `Replicated`
4. Vocab-parallel Embedding (`@shard(dim=0)`):
   - Input: token IDs (`Replicated`)
   - Emit `nsl_tp_all_reduce_sum` after lookup
   - Output: `Replicated`
5. Terminal layer / LM head (`@shard(dim=0)`, last sharded layer):
   - Allocate full-size output tensor: `batch * vocab_size * sizeof(dtype)`
   - Emit out-of-place `nsl_tp_all_gather`
   - Output: `Replicated`
6. Unsharded layers (LayerNorm, RMSNorm): pass through, must be `Replicated`

### FFI Functions

| Function | Signature | Notes |
|---|---|---|
| `nsl_tp_init` | `() -> i64` | Reads `NSL_LOCAL_RANK`, `NSL_WORLD_SIZE`, `NSL_SIMULATED_TP` from env. Inits collective backend. |
| `nsl_tp_rank` | `() -> i64` | Returns this process's rank (cached from env at init) |
| `nsl_tp_world_size` | `() -> i64` | Returns total device count (cached from env at init) |
| `nsl_tp_mmap_open` | `(path_ptr, path_len) -> i64` | Mmap a safetensors file, return opaque handle |
| `nsl_tp_load_sharded_weight` | `(mmap_handle, name_ptr, name_len, shape_ptr, ndims, shard_dim, dtype) -> i64` | Load this rank's slice. Rank/world_size read internally. |
| `nsl_tp_mmap_close` | `(mmap_handle) -> i64` | Unmap safetensors file after weights loaded to GPU |
| `nsl_tp_all_reduce_sum` | `(sendbuf, recvbuf, count, dtype, stream) -> i64` | In-place OK (same ptr for send/recv) |
| `nsl_tp_all_gather` | `(sendbuf, recvbuf, send_count, dtype, stream) -> i64` | Out-of-place — recvbuf = `world_size * send_count` |
| `nsl_tp_broadcast` | `(buf, count, dtype, root_rank, stream) -> i64` | Root sends, others receive |
| `nsl_tp_barrier` | `() -> i64` | Synchronize all ranks |
| `nsl_tp_destroy` | `() -> i64` | Tear down collective backend + shared memory |

All collective FFI functions accept a `stream` parameter (`StreamHandle = *mut c_void`). M30 passes `null_mut()` (default stream); the parameter exists for M34 (Ring Attention) compute/communication overlap.

**`nsl_tp_init` reads env vars internally** — the compiled code does not need to extract rank/world_size. This matches the SPMD model: each process is identical, differing only by environment.

### Error Handling

All FFI functions return `i64`: 0 = success, non-zero = error code. On non-zero return from any collective:
- The process calls `nsl_panic` with the error message, aborting immediately
- The parent CLI detects the non-zero exit code and kills all remaining child processes via `SIGTERM` (Unix) or `TerminateProcess` (Windows)
- Parent reports which rank failed and the error code

This prevents rank divergence (one rank fails while others hang waiting at the next collective).

### Activation State Across Model Boundaries

Activation states are tracked interprocedurally across model method calls. When the compiler compiles a nested model's `forward()` as a separate Cranelift function, the caller's `DistState` for the input is passed through the call boundary. The return value's `DistState` is determined by the last operation in the callee.

For `@shard` on nested model members (e.g., `layers: [TransformerBlock; N]`): `@shard` is only valid on leaf layer declarations (Linear, Embedding). To shard all sub-layers of a `TransformerBlock`, each sub-layer must have its own `@shard` decorator. This keeps the decorator extraction logic simple and the shard semantics explicit.

**Element-wise ops between identically-sharded tensors** (e.g., SwiGLU gate * up in MLP): valid — both operands have the same `Sharded{dim:1}` state, and the result preserves that state. The activation state rules apply: element-wise ops between `Sharded{dim:X}` and `Sharded{dim:X}` produce `Sharded{dim:X}`. Mismatched shard dims are a compile error.

### Codegen Flow

```
nsl_tp_init()                                   // reads env vars internally

// mmap weight file once
mmap = nsl_tp_mmap_open("model.safetensors")

// weight loading — each process loads only its shard
nsl_tp_load_sharded_weight(mmap, "embed", [32000,8192], 0, DTYPE_F32)
nsl_tp_load_sharded_weight(mmap, "q_proj", [8192,8192], 0, DTYPE_F32)
nsl_tp_load_sharded_weight(mmap, "o_proj", [8192,8192], 1, DTYPE_F32)

nsl_tp_mmap_close(mmap)                        // weights on GPU, release mmap

// forward pass:
embed_out = lookup(tokens, embed_shard)        // vocab-parallel
nsl_tp_all_reduce_sum(embed_out, embed_out, n, DTYPE_F32, null)  // in-place
                                                // state: Replicated

q = matmul(x, q_proj_shard)                    // column-parallel
                                                // state: Sharded{dim:1}

attn_out = attention(q, k, v)                  // per-GPU heads
out = matmul(attn_out, o_proj_shard)           // row-parallel (input Sharded ✓)
nsl_tp_all_reduce_sum(out, out, n, DTYPE_F32, null)  // in-place reduce
                                                // state: Replicated

norm_out = layer_norm(out)                     // replicated

// LM head — out-of-place gather
logit_shard = matmul(norm_out, lm_head_shard)  // column-parallel
full_logits = alloc(batch * vocab * sizeof(f32))
nsl_tp_all_gather(logit_shard, full_logits, shard_count, DTYPE_F32, null)
                                                // state: Replicated

nsl_tp_destroy()
```

---

## Section 4: Per-GPU KV-Cache & M29 Serving Integration

### GQA-Aware KV Head Assignment

Modern models use Grouped-Query Attention (GQA) where Q heads != KV heads. The compiler must handle this:

```rust
let my_q_heads = total_q_heads / world_size;   // always sharded
let my_kv_heads = if total_kv_heads >= world_size {
    total_kv_heads / world_size                 // shard KV heads
} else {
    total_kv_heads                              // replicate all KV heads
};
```

**Compile-time validation:**
- `total_q_heads % world_size == 0` — hard error if not
- `total_kv_heads < world_size` — warning, KV replicated (no shard, no collective on KV projections)
- `total_kv_heads >= world_size && total_kv_heads % world_size != 0` — hard error

Each process inits its KV-cache with `my_kv_heads`:
```rust
nsl_kv_cache_init_gpu(num_blocks, block_size, my_kv_heads, head_dim, num_layers)
```

No cross-GPU KV-cache communication during attention — PagedFlashAttention runs independently per process on its head subset.

**No changes to M25 `KvCacheManager` or `BlockAllocator` are needed.** Each process is a regular single-GPU process with fewer heads — the existing API handles this naturally since `num_heads` is already a parameter.

### StepContext Broadcast (Page Table Synchronization)

Rank 0 is the single source of truth for memory management. It broadcasts a packed `StepContext` — not bare token IDs — to ensure all ranks have identical page table state.

```rust
// Fixed-size constants for StepContext. The struct is broadcast via shared
// memory so its byte size must be deterministic and identical across processes.
pub const MAX_BATCH: usize = 256;           // max concurrent sequences
pub const MAX_TOTAL_TOKENS: usize = 32768;  // max tokens per step (prefill chunks + decodes)
pub const MAX_BLOCK_UPDATES: usize = 4096;  // max new block allocations per step
// Total struct size: ~330 KB (fits comfortably in shared memory)

#[repr(C)]
pub struct StepContext {
    pub num_sequences: i32,
    pub total_tokens: i32,
    // Per-sequence metadata
    pub seq_ids: [i64; MAX_BATCH],
    pub seq_lens: [i32; MAX_BATCH],
    pub num_tokens_per_seq: [i32; MAX_BATCH],  // prefill chunk size or 1 for decode
    // Flattened token IDs for the batch
    pub token_ids: [i64; MAX_TOTAL_TOKENS],
    // Block table updates from Rank 0's allocator
    pub num_block_updates: i32,
    pub block_updates: [BlockUpdate; MAX_BLOCK_UPDATES],
}

#[repr(C)]
pub struct BlockUpdate {
    pub seq_id: i64,
    pub logical_block: i32,
    pub physical_block: i32,  // assigned by Rank 0's BlockAllocator
}
```

**Why:** If Rank 0 allocates Physical Block 14 for a sequence but Rank 1 independently allocates Physical Block 8, their block table pointers diverge. Rank 1's FlashAttention kernel reads garbage memory. All ranks must mirror Rank 0's physical block assignments exactly, bypassing their local BlockAllocator.

### Request Lifecycle in Multi-GPU Serve

```
Rank 0 (Control Plane):            Ranks 1-N (Data Plane):
  accept HTTP request
  scheduler.step() -> batch
  allocate new blocks if needed
  pack StepContext
  broadcast(step_ctx, root=0)       receive(step_ctx)
                                    apply block_updates to PageTable
  forward_pass(tokens)              forward_pass(tokens)    // SPMD
    [all_reduce per layer]            [all_reduce per layer]
    [all_gather at LM head]           [all_gather at LM head]
  sample(full_logits) -> token
  broadcast(token, root=0)          receive(token)
  kv_cache.append(token)            kv_cache.append(token)  // same phys block
  if EOS: respond to client
```

### Codegen for Multi-GPU Serve Block

When `--devices > 1` with a `serve` block:
- Wraps existing `compile_serve_block` output with TP init/destroy
- Injects `nsl_tp_broadcast` calls for `StepContext` and sampled tokens
- Only rank 0 runs the scheduler loop and HTTP handling
- Other ranks enter a `recv → compute → send` loop

---

## Section 5: CLI Changes & E2E Testing

### CLI `--devices` Flag

```
nsl run model.nsl --devices 4
```

**Build-then-spawn flow** (avoids compilation race condition):

1. **Parent process compiles once:** Parse → Semantic → Cranelift codegen → write binary to `.nsl-cache/`
2. **Parent spawns N child processes** with the pre-built binary:
   ```
   NSL_LOCAL_RANK=0 NSL_WORLD_SIZE=4 NSL_SIMULATED_TP=1 .nsl-cache/model
   NSL_LOCAL_RANK=1 NSL_WORLD_SIZE=4 NSL_SIMULATED_TP=1 .nsl-cache/model
   NSL_LOCAL_RANK=2 NSL_WORLD_SIZE=4 NSL_SIMULATED_TP=1 .nsl-cache/model
   NSL_LOCAL_RANK=3 NSL_WORLD_SIZE=4 NSL_SIMULATED_TP=1 .nsl-cache/model
   ```
3. Children skip compilation — execute the binary directly
4. Parent waits for all children, reports aggregate status. Non-zero exit from any rank is reported with the failing rank number.

**Why build-then-spawn:** If N children all compile simultaneously, they race on `.nsl-cache/` file locks, corrupt object files, and waste CPU. Compilation is single-process; execution is multi-process.

### Modulo Device Binding

```rust
let physical_count = cuDeviceGetCount();
let physical_id = rank as u32 % physical_count;
let device = cuDeviceGet(physical_id);
```

On a single-GPU machine with `--devices 4`, all 4 processes share GPU 0. Mathematically correct for `SimulatedBackend` tests. Real multi-GPU uses 1:1 mapping.

### Hardware-Agnostic `@shard` Decorator

```python
# Portable — works on any GPU count
@shard(dim=0)
embed: Embedding(vocab=32000, dim=8192)

@shard(dim=1)
o_proj: Linear(8192, 8192)
```

- `devices` parameter removed from decorator — compiler reads `world_size` from `--devices N`
- Same `.nsl` script runs on 2, 4, or 8 GPUs without source changes
- Validation: `shape[dim] % N == 0` using CLI-provided N
- `@shard` without `--devices`: compile error `"@shard requires --devices flag"`

**Edge cases:**
- `--devices 1` with `@shard` decorators: shard with world_size=1 (each shard = full tensor). No collectives emitted. Zero overhead — code is equivalent to non-sharded. This makes `@shard`-annotated code portable.
- `--devices N` without any `@shard` decorators: silently ignored. All N processes run the same replicated computation. No error — allows testing the SPMD launcher independently.
- `--devices 0` or negative: CLI error before compilation.

### E2E Test Strategy

Tests use `SimulatedBackend` (set via `NSL_SIMULATED_TP=1`):

| Test | What it verifies |
|---|---|
| `m30_tp_basic.nsl` | `serve` block with `@shard` layers, `--devices 2`. Two simulated processes, collective calls via `SimulatedBackend`. Output matches single-device reference. |
| `m30_shard_validation.nsl` | Compile-time error: `@shard(dim=0)` on layer with size not divisible by `--devices`. Expects compiler error message. |
| `m30_activation_state_error.nsl` | Compile-time error: row-parallel Linear fed a replicated tensor. Expects `"expected sharded input"` error. |
| `m30_gqa_replication.nsl` | 32 Q heads, 4 KV heads, `--devices 8`. KV replicated, Q sharded. No division-by-zero crash. |
| `m30_serve_multi_step.nsl` | Multi-step serve loop with 2+ requests across `--devices 2`. Verifies StepContext broadcast and page table synchronization. |

**Test harness** (`e2e.rs`):
- New helper `run_nsl_multi(file, devices)` — parent compiles, spawns N children with `NSL_SIMULATED_TP=1`
- Collects stdout from rank 0 only (rank 0 is the output process)
- Compares against expected output in `tests/expected/`

**Test fixture:** Small `tests/fixtures/m30_test_weights.safetensors` — tiny 2-layer model with known values. Tests verify each rank loads the correct slice.

---

## Out of Scope

- Multi-node (network/InfiniBand) — single-node only
- Pipeline parallelism — no stage splitting
- Expert parallelism (MoE) — no router/expert dispatch
- Data parallelism for training — inference only
- Custom NCCL reduction ops for BYOD dtypes
- Logit gathering optimization (local top-K then gather scalars) — future profiling-driven work
- Compute/communication overlap via multiple streams — M34 (stream parameter already in FFI)

## Success Criteria

- 4-GPU tensor-parallel forward pass: output matches single-GPU reference to fp16 precision
- Automatic weight sharding from single safetensors file via mmap (near-zero CPU RAM)
- KV-cache 1/N per GPU (verified via M25 profiler), with GQA replication when kv_heads < world_size
- M29 serving integration: continuous batching across N GPUs with synchronized page tables
- Compile-time validation catches divisibility errors and invalid activation state transitions
- SimulatedBackend E2E tests pass on single-GPU Windows CI
