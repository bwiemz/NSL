# M27: FlashAttention-2 + Fused Attention Kernels — Design Spec

**Date:** 2026-03-14
**Status:** Draft
**Prerequisites:** M25 (PagedAttention), M26 (@autotune + fusion + kernel profiling), M17 (GPU/CUDA)
**Architecture:** Hand-written PTX templates in nsl-codegen (AOT), launch wrappers in nsl-runtime

---

## 1. Goal

Implement FlashAttention-2 as hand-optimized tiled attention PTX kernels that never materialize the full `[seq, seq]` attention matrix, reducing memory from O(N²) to O(N) and delivering 2-5x speedup over naive attention. Integrate with M25's paged KV-cache, support RoPE positional encoding and grouped-query attention (GQA), and expose the optimization through a stdlib intrinsic with decorator-driven lowering.

## 2. Not in Scope

- FlashAttention backward pass (training-time)
- Sliding-window attention (Mistral-style)
- Flash-Decoding (optimized single-token decode)
- Fusing the K/V projection matmul into `rope_cache_write`

---

## 3. Architecture: AOT PTX Templates

### 3.1 The AOT Boundary

NeuralScript is an ahead-of-time compiler. PTX synthesis happens at `nsl build` time, not at inference time.

- **`crates/nsl-codegen/src/flash_attention.rs`** — PTX string synthesis. Rust functions parameterized by `(block_q, block_kv, head_dim, causal, paged, rope_q, rope_style, gqa_group_size)` that return `Vec<u8>` (null-terminated PTX). Called during compilation. The `@autotune` harness generates PTX for each tile-size variant, benchmarks them (when GPU benchmarking is activated), and the winning PTX is embedded as a `static` byte array in the binary's `.rodata` segment. Until GPU benchmarking is activated, `select_middle_values()` picks the fallback.

- **`crates/nsl-runtime/src/flash_attention.rs`** — Launch only. `extern "C"` FFI wrappers that compute grid/block dimensions from tensor shapes, marshal arguments, and call `kernel_launch()` with the pre-baked `.rodata` PTX pointer. No string synthesis, no PTX generation at runtime.

### 3.2 Kernel Table

| Kernel | Purpose | Shared Memory | SRAM Budget |
|--------|---------|---------------|-------------|
| `flash_attention_2` | Standard tiled attention, contiguous KV | Q tile + K/V tile (shared) | `(block_q + block_kv) * head_dim * sizeof(f16)` |
| `paged_flash_attention` | Same algorithm, paged KV via block table | Q tile + K tile | Same |
| `rope_cache_write` | RoPE rotation + paged cache write | None (elementwise) | N/A |
| `flash_attention_rope` | Attention with Q-only RoPE in registers | Q tile + K tile | Same (cos/sin in registers, not SRAM) |
| `flash_attention_gqa` | GQA head broadcasting | Q tile + K tile | Same |

### 3.3 Variant Composition

The five kernels are generated from a single parameterized PTX template with orthogonal flags:

| Flag | Type | Effect on PTX |
|------|------|---------------|
| `paged` | `bool` | K/V load uses block table indirection |
| `rope_q` | `bool` | Q load includes register-level RoPE rotation before SRAM write |
| `rope_style` | `enum` | `half_split` (LLaMA default) or `adjacent` (GPT-NeoX) |
| `gqa_group_size` | `u32` | KV head index = Q head / group_size (1 = standard MHA) |
| `causal` | `bool` | Loop bound clamped to diagonal |
| `scale` | `f32` | `1/sqrt(head_dim)`, applied to S before online softmax |

A model uses one specific combination — the compiler generates exactly that variant. `@autotune` explores tile sizes within the single variant.

### 3.4 `kernel_launch()` Signature Change

`kernel_launch()` gains a `shared_mem_bytes: u32` parameter, passed as the 7th argument to `cuLaunchKernel` (currently hardcoded to 0):

```rust
// Internal Rust function (NOT extern "C" — called by other Rust code in runtime)
fn kernel_launch(
    ptx_ptr: *const u8,
    name_ptr: *const u8,
    grid: [i64; 3],
    block: [i64; 3],
    args: &[*mut c_void],
    shared_mem_bytes: u32,    // NEW
) -> CUresult
```

All existing callers pass `shared_mem_bytes: 0` (backward compatible). The FlashAttention launch wrappers pass `(block_q + block_kv) * head_dim * sizeof(f16)`.

**Propagation:** The `extern "C" fn nsl_kernel_launch()` FFI wrapper (called from Cranelift-generated code for user-defined `kernel` blocks) must also gain a `shared_mem_bytes` parameter and forward it to `inner::kernel_launch()`. All ~8 internal call sites (`gpu_elementwise_binary`, `gpu_elementwise_unary`, `gpu_matmul_f32`, `gpu_scalar_op`, backward helpers, clamp_backward, `nsl_kernel_launch`, tests) are updated to pass `0` for backward compatibility.

---

## 4. FlashAttention-2 Algorithm

### 4.1 Core Algorithm — Online Softmax with Tiled Accumulation

Each thread block produces one tile of the output matrix (`block_q` rows).

**Per thread block:**

1. **Load Q tile** into shared memory: `shmem_Q[block_q][head_dim]`

2. **Allocate register accumulators** per thread:
   - `O_acc` — running output accumulator (initialized to 0)
   - `row_max` — running max per Q row (initialized to `-inf`)
   - `row_sum` — running denominator per Q row (initialized to 0)

3. **Compute causal loop bound** (zero-divergence):
   ```
   k_max = causal ? min(seq_len, q_start + block_q) : seq_len
   ```
   The inner loop naturally terminates at the diagonal. No per-tile branch needed. ~50% compute reduction for causal attention with zero thread-divergence overhead.

4. **Inner loop** — iterate K/V tiles from `k_start = 0` to `k_max`, stepping by `block_kv`. Three `bar.sync` calls per iteration to prevent SRAM read-after-write hazards:

   ```
   // Phase 1: Load K tile into SRAM
   shmem_K[block_kv][head_dim] = K[k_start : k_start+block_kv][:]
   bar.sync 0    // FENCE 1: ensure K tile fully in SRAM before any warp reads it

   // Phase 2: Compute S = Q_tile @ K_tile^T (result in registers)
   // S[block_q][block_kv] lives entirely in registers
   // Apply scale: S[i][j] = S[i][j] * scale   (mul.f32)
   // Partial causal mask on final diagonal tile:
   //   set S[i][j] = -inf where k_start + j > q_start + i
   bar.sync 0    // FENCE 2: ensure ALL warps finished reading K before SRAM is overwritten

   // Phase 3: Online softmax — S transforms to P in-place in registers
   //
   // Warp-level reductions: mma.sync distributes row fragments across threads
   // within a warp. row_max and row_sum require cross-thread reduction via
   // shfl.sync.bfly (butterfly shuffle) — log2(warp_size) steps per reduction.
   // This is the ONLY correct reduction pattern; shared memory reduction would
   // require an extra bar.sync and waste SRAM.
   //
   new_max = max(row_max, warp_reduce_max(row_max_of_S))  // shfl.sync.bfly
   correction = exp(old_max - new_max)       // always <= 1.0, no overflow
   row_sum = row_sum * correction + warp_reduce_sum(exp(S - new_max))  // shfl.sync.bfly
   O_acc = O_acc * correction                // rescale previous accumulation
   P = exp(S - new_max)                      // overwrites S registers in-place
   // P never touches SRAM — stays in registers as the A operand for Phase 5

   // Phase 4: Load V tile (reuses shmem_K address space)
   shmem_V[block_kv][head_dim] = V[k_start : k_start+block_kv][:]
   bar.sync 0    // FENCE 3: ensure V tile fully in SRAM

   // Phase 5: Accumulate O_acc += P @ V_tile
   // P is in registers (from Phase 3), V is in SRAM
   ```

5. **Finalize:** `O = O_acc / row_sum` — element-wise division in registers

6. **Store** output tile to global memory

### 4.2 Three bar.sync Per Iteration — Critical Synchronization

| Fence | After | Before | Prevents |
|-------|-------|--------|----------|
| 1 | K tile loaded into SRAM | Any warp reads K | Partial-load race |
| 2 | All warps finish S = Q @ K^T | Any warp overwrites K SRAM with V | Read-after-write hazard: Warp 0 clobbers K data Warp 1 is still reading |
| 3 | V tile loaded into SRAM | Any warp reads V for P @ V | Partial-load race |

### 4.3 In-Register S→P Transformation

The `exp(S - new_max)` computation overwrites the S register pool in-place. P is immediately available as the left operand for the `P @ V_tile` tile matmul. No SRAM round-trip for softmax weights. This is critical — materializing P to shared memory would halve the available SRAM for Q/K tiles.

### 4.4 Numerical Precision

- All intermediate accumulation in f32, even when inputs are f16
- Inputs widened on load (`cvt.f32.f16`), output narrowed on store (`cvt.f16.f32`)
- Online softmax correction factor `exp(old_max - new_max)` is always `<= 1.0`, preventing overflow
- Mathematically identical results to full materialized softmax (online softmax is exact, not approximate)
- Validation target: `atol=1e-3` vs naive attention at fp16 precision

---

## 5. Paged FlashAttention — M25 Integration

### 5.1 Block-Aligned Tiling Constraint

`block_kv` must be a multiple of M25's `block_size`. The semantic checker enforces this at compile time when `@autotune(block_kv=[...])` is used with a `@paged_kv` model.

Example: if `block_size=16`, valid `block_kv` values are `[16, 32, 64]`. This guarantees each K/V tile load aligns exactly to physical block boundaries — no cross-block scatter.

### 5.2 Paged K/V Tile Loading

M25's `BlockAllocator` maintains **separate K and V pools** (`k_pool` and `v_pool`), each with layout `[num_heads, block_size, head_dim]` per physical block. `block_stride = num_heads * block_size * head_dim * sizeof(f32)` covers all heads within one block. The FlashAttention kernel receives `k_pool_ptr` and `v_pool_ptr` as separate arguments, consistent with M25's `get_kv_ptrs()` API.

The inner loop Phases 1 and 4 replace contiguous loads with block-table-driven loads:

```
num_blocks_per_tile = block_kv / block_size
for b in 0..num_blocks_per_tile:
    logical_idx = (k_start / block_size) + b
    physical_id = block_table[seq_id * max_blocks + logical_idx]

    // ALL threads cooperatively load from SAME physical block
    // Coalesced 128-byte transactions — consecutive elements
    // block_stride includes all heads; index by specific head within block
    src = k_pool + physical_id * block_stride + head * block_size * head_dim
    shmem_K[b * block_size : (b+1) * block_size][:] = src[0 : block_size][:]
```

One page table lookup per physical block (every `block_size` tokens), not per token.

### 5.3 KV Pool Element Type

M25's paged pools are f32-backed (`block_stride` uses `sizeof(f32)`). FlashAttention kernels operate on f16 for SRAM efficiency. The paged load path widens f32→f16 is unnecessary — the pools already store f32, and the kernel widens to f32 for accumulation anyway. The PTX loads with `ld.global.f32`, accumulates in f32, and narrows to f16 only on final output store (`cvt.f16.f32`). No dtype conversion needed at the pool boundary.

### 5.4 Block Table GPU Sync

The block table is small — for a 4096-token sequence with `block_size=16`: 256 entries × 4 bytes = 1KB. Synced to GPU once per sequence via pinned host memory + `cuMemcpyHtoD`.

### 5.4 Safe Garbage in Partial Blocks

Because M25's allocator always allocates full `block_size` physical blocks, reading "past" valid tokens inside a physical block is memory-safe (no segfault). The PTX kernel loads the whole block into SRAM without bounds-checking; the causal/length mask on S nullifies stale data before softmax. This saves ALU cycles vs per-element validity checks.

### 5.5 Q Remains Contiguous

Only K/V are paged. Q is ephemeral (recomputed each step during inference) and small (single token during decode, prompt-length during prefill). No paging benefit.

### 5.6 M29-Ready Kernel Signature

The PTX kernel accepts `*const u32 seq_ids` and `*const u32 seq_lens` arrays. During batched inference, `seq_id = seq_ids[blockIdx.y]` and `seq_len = seq_lens[blockIdx.y]`. This avoids a kernel rewrite when M29 (continuous batching) introduces ragged batches.

- **Prefill:** single sequence, `blockIdx.y = 0`
- **Decode:** `blockIdx.y` maps to batch index, each with potentially different `seq_len`

---

## 6. Kernel Variants

### 6.1 `rope_cache_write` — Two-Phase K Preparation

Runs **once per new token batch, before attention**. Two steps:

1. **K/V projection** — `K = x @ Wk`, `V = x @ Wv` via existing matmul path (not fused into this kernel)
2. **`rope_cache_write` kernel** — elementwise, no shared memory:
   - Each thread handles one `(head, dim_pair)` element for one token
   - Loads projected K elements from global memory into registers
   - Loads `cos[pos]`, `sin[pos]` from precomputed frequency table (global memory → registers)
   - Applies RoPE rotation in registers
   - Looks up physical block via block table
   - Writes rotated K directly into paged K pool
   - Same scatter-write for V (no rotation, just paged placement)

**Grid:** `(num_tokens, num_heads, ceil(head_dim/2))` — 3D grid where `blockIdx.x` is the token index. Token dimension is on `blockIdx.x` (supports up to 2^31-1) rather than `blockIdx.z` (limited to 65535) to handle large prefill batches (e.g., batch=128 × seq_len=2048 = 262144 tokens). Handles both decode (`num_tokens = batch_size`) and prefill (`num_tokens = batch_size * seq_len`) in a single launch. No host-side token loop.

### 6.2 RoPE Interleaving Styles

Two industry standards, selectable via `rope_style` flag:

| Style | Pairing | Stride | Used By |
|-------|---------|--------|---------|
| `half_split` (default) | `(x[i], x[i + head_dim/2])` | `head_dim / 2` | LLaMA, Qwen, Mistral |
| `adjacent` | `(x[2i], x[2i+1])` | `1` | GPT-NeoX, GPT-J |

The stride difference is a single constant in the PTX address calculation — no algorithmic change. Default to `half_split` for compatibility with the most common open-weight models.

### 6.3 `flash_attention_rope` — Q-Only RoPE in Registers

Modifies Phase 1 of the base algorithm (Q tile load). Instead of loading Q directly into SRAM, each thread:

1. Loads Q element pair from global memory into registers
2. Loads corresponding `cos[q_pos]`, `sin[q_pos]` from global memory into registers
3. Applies RoPE rotation entirely in registers:
   ```
   q_rot_a = q_a * cos - q_b * sin
   q_rot_b = q_a * sin + q_b * cos
   ```
4. Writes rotated Q into SRAM for the tile matmul

cos/sin never touch SRAM. The shared memory budget remains `(block_q + block_kv) * head_dim * sizeof(f16)` — identical to the base kernel. RoPE is effectively "free" from a memory-bandwidth perspective.

K in the KV-cache is already pre-rotated by `rope_cache_write`, so the inner loop (Phases 2-5) is unchanged.

### 6.4 `flash_attention_gqa` — Grouped-Query Attention

GQA uses fewer KV heads than Q heads. For example, 32 Q heads with 8 KV heads: `group_size = 4`.

**The change is minimal — only the K/V address calculation differs:**

```
kv_head = q_head / group_size   // integer division, compile-time constant
src_K = K[batch][kv_head][k_start : k_start+block_kv][:]
```

Q tile, SRAM layout, online softmax, and accumulation are completely unchanged. `group_size` is baked into the PTX template as a literal.

---

## 7. Codegen Integration

### 7.1 Surface Syntax — Stdlib Intrinsic

Users trigger FlashAttention through a stdlib intrinsic with decorator-driven lowering:

```
from nsl.nn import scaled_dot_product_attention

model Transformer:
    @flash_attention(causal=true)
    @gqa(groups=4)
    @rope(style="half_split")
    @paged_kv(block_size=16)
    fn attention(self, Q, K, V) -> Tensor:
        let scale = 1.0 / sqrt(float(Q.shape[-1]))
        return scaled_dot_product_attention(Q, K, V, scale, causal=true)
```

**Lowering rules:**
- When the compiler sees `scaled_dot_product_attention()` inside a function decorated with `@flash_attention`, it lowers directly to the FlashAttention PTX dispatch.
- Without `@flash_attention`, `scaled_dot_product_attention` compiles to naive attention. When `causal=true`, the naive path injects a causal mask (upper triangle set to `-inf`) before softmax: `softmax(apply_causal_mask((Q @ K.T) * scale)) @ V`. When `causal=false`, the mask is omitted: `softmax((Q @ K.T) * scale) @ V`. This ensures the naive fallback is numerically equivalent to the FlashAttention path in both modes.
- The decorators determine which variant flags are set: `@paged_kv` → `paged=true`, `@rope` → `rope_q=true` + `rope_style`, `@gqa(groups=N)` → `gqa_group_size=N`.

### 7.2 Compile-Time Flow (`nsl build`)

1. **Semantic checker** validates decorator placement and arguments (extends existing M26 validation)
2. **Codegen** calls `flash_attention.rs` to determine variant flags from decorators
3. **If `@autotune` present:** generate PTX for each `(block_q, block_kv)` variant in the Cartesian product. Each is a complete kernel. The harness benchmarks (when activated) and selects winner. Cache via existing SHA-256 infrastructure in `.nsl-cache/autotune/`.
4. **Winning PTX** (or middle-value fallback) embedded as `static` byte array in `.rodata`, referenced by symbol (e.g., `_nsl_flash_attention_ptx`)
5. **Cranelift IR** emitted for the call site:
   - Compute grid dimensions from tensor shapes
   - Compute `shared_mem_bytes`
   - Call the `extern "C"` launch wrapper with tensor pointers + metadata

### 7.3 Runtime FFI Wrapper

The FFI boundary uses C-ABI-safe types (no Rust fat pointers):

```rust
#[no_mangle]
pub extern "C" fn nsl_flash_attention(
    q_ptr: *mut c_void, k_ptr: *mut c_void, v_ptr: *mut c_void,
    out_ptr: *mut c_void, scale: f32,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    // Paged KV params (zero/null if not paged variant)
    block_table_ptr: *mut c_void,
    k_pool_ptr: *mut c_void, v_pool_ptr: *mut c_void,
    block_size: i64,
    // RoPE params (zero/null if not rope variant)
    cos_ptr: *mut c_void, sin_ptr: *mut c_void,
    // Ragged batch params (M29-ready)
    seq_ids_ptr: *mut c_void, seq_lens_ptr: *mut c_void,
    shared_mem_bytes: u32,
) -> i64
```

This wrapper internally builds the args array and calls `kernel_launch()`. Note: `gqa_group_size` is absent from this signature because it is baked into the PTX template as a compile-time literal (Section 6.4). The Cranelift IR passes the pre-selected PTX `.rodata` pointer directly — the runtime does not need to choose between PTX variants. All variant flags (`paged`, `rope_q`, `gqa_group_size`, `causal`, `rope_style`) are resolved at compile time and encoded into the specific PTX string that gets embedded.

Similarly for `rope_cache_write`:

```rust
#[no_mangle]
pub extern "C" fn nsl_rope_cache_write(
    k_projected_ptr: *mut c_void, v_projected_ptr: *mut c_void,
    cos_ptr: *mut c_void, sin_ptr: *mut c_void,
    positions_ptr: *mut c_void,
    k_pool_ptr: *mut c_void, v_pool_ptr: *mut c_void,
    block_table_ptr: *mut c_void,
    // Ragged batch params (M29-ready) — thread needs seq_id for block table lookup
    seq_ids_ptr: *mut c_void, seq_lens_ptr: *mut c_void,
    num_tokens: i64, num_heads: i64, head_dim: i64, block_size: i64,
) -> i64
```

### 7.4 Builtin Registration

New FFI functions registered in:
- `crates/nsl-codegen/src/builtins.rs` — `nsl_flash_attention`, `nsl_rope_cache_write`
- `crates/nsl-semantic/src/builtins.rs` — type signatures for `scaled_dot_product_attention`

### 7.5 Decorator Validation (Semantic Checker)

Extensions to `crates/nsl-semantic/src/checker.rs`:

- `@flash_attention`: valid on `fn` inside `model` only. Args: `causal: bool` (default true).
- `@rope`: valid on `fn` decorated with `@flash_attention`. Args: `style: str` (default "half_split", valid values "half_split" | "adjacent").
- `@gqa`: valid on `fn` decorated with `@flash_attention`. Args: `groups: int` (required, must be > 0).
- `@paged_kv` + `@autotune` interaction: if both present, validate `block_kv` values are multiples of `block_size`.
- **Decorator order:** The semantic checker accepts decorators in any order. It collects all decorators on a function and validates them as a set, not a sequence.

---

## 8. @autotune Integration

### 8.1 Tuning Parameters

```
@autotune(block_q=[32, 64, 128], block_kv=[32, 64])
@flash_attention(causal=true)
fn attention(self, Q, K, V) -> Tensor:
    ...
```

The `@autotune` decorator on a `@flash_attention` function generates `|block_q| × |block_kv|` PTX variants (e.g., 3 × 2 = 6). Each variant has different shared memory requirements and parallelism characteristics.

**Semantic checker change:** The existing `@autotune` validation (M26) restricts it to `kernel` blocks only. This must be relaxed to also accept `fn` declarations decorated with `@flash_attention`. The error message for other `fn` targets remains: `"@autotune can only be applied to kernel blocks or @flash_attention functions"`.

### 8.2 Shared Memory Validation

At PTX generation time, validate that `(block_q + block_kv) * head_dim * sizeof(f16)` does not exceed the GPU's shared memory limit (typically 48KB for sm_52, up to 164KB for sm_90 with opt-in). Emit a compile-time error if any variant exceeds the limit.

### 8.3 Cache Key

The existing SHA-256 cache key from M26 incorporates the kernel AST hash, device info, and tuning parameters. For FlashAttention, the "AST hash" is derived from the variant flags `(paged, rope_q, rope_style, gqa_group_size, causal, head_dim)` plus the tile sizes.

---

## 9. Kernel Profiler Integration

FlashAttention kernels appear as single entries in the M26 kernel profiler timeline. The `kernel_launch()` call records cuEvent pairs around the launch (existing M26 infrastructure), so each attention call shows as one span in Chrome tracing JSON.

The kernel name encodes the variant: `flash_attn_p{paged}_r{rope}_g{gqa}_c{causal}_q{block_q}_kv{block_kv}`.

---

## 10. Testing Strategy

### 10.1 Numerical Correctness

For each variant, compare FlashAttention output against naive `softmax((Q @ K.T) * scale) @ V`:
- Random Q, K, V tensors at fp16
- `atol=1e-3` (fp16 precision bound)
- Test with `causal=true` and `causal=false`
- Test at multiple sequence lengths: 128, 512, 2048
- Test edge cases: `seq_len < block_kv` (single tile), `seq_len` not divisible by `block_kv`

### 10.2 Paged Correctness

Verify paged variant produces identical output to contiguous variant:
- Same Q, K, V data, one stored contiguously, one stored in paged blocks
- Bit-identical output expected (same algorithm, same data, different memory layout)

### 10.3 RoPE Correctness

- Verify `rope_cache_write` + `flash_attention_rope` produces same result as applying RoPE to both Q and K before naive attention
- Test both `half_split` and `adjacent` styles

### 10.4 GQA Correctness

- 32 Q heads, 8 KV heads (`group_size=4`)
- Verify each group of 4 Q heads produces the same result as if K/V were explicitly replicated

### 10.5 E2E Tests

- `examples/m27_flash_attention.nsl` — basic FlashAttention with correctness check
- `examples/m27_paged_attention.nsl` — paged variant
- `examples/m27_rope_gqa.nsl` — composed RoPE + GQA variant

### 10.6 Stubbed GPU Path

Same pattern as M26. What is fully implemented vs stubbed:

- **Fully implemented:** PTX string synthesis (`flash_attention.rs` in codegen), all 5 kernel variant templates, online softmax math, shared memory layout, `.rodata` embedding, semantic checker validation, decorator lowering, `scaled_dot_product_attention` naive fallback path.
- **Stubbed (TODO markers):** The `cuLaunchKernel` invocation inside the `extern "C"` launch wrappers in the runtime. These wrappers compute grid/block/args correctly but gate the actual launch behind `#[cfg(feature = "cuda")]` with a fallback to the naive attention path.

E2E tests validate the naive fallback path and compile-time validation (decorator errors, `@autotune` shared memory limits, `@paged_kv` block_kv alignment).

---

## 11. Deliverables Checklist

1. `crates/nsl-codegen/src/flash_attention.rs` — PTX template generator for all 5 kernel variants
2. `crates/nsl-runtime/src/flash_attention.rs` — `extern "C"` launch wrappers (FFI boundary)
3. `kernel_launch()` updated with `shared_mem_bytes` parameter
4. Semantic checker validation for `@flash_attention`, `@rope`, `@gqa` decorators
5. `@autotune` integration — tile-size Cartesian product, shared memory validation
6. `scaled_dot_product_attention` stdlib intrinsic registered in builtins
7. Kernel profiler shows FlashAttention as single timeline entry
8. Decorator interaction validation (`@paged_kv` block_kv alignment check)
9. 3 E2E tests + unit tests for PTX generation and online softmax math
10. All existing tests continue to pass

---

## 12. File Changes Summary

**New files:**
- `crates/nsl-codegen/src/flash_attention.rs` — PTX synthesis (largest file, ~800-1200 lines)
- `crates/nsl-runtime/src/flash_attention.rs` — Launch wrappers (~150 lines)
- `examples/m27_flash_attention.nsl`, `examples/m27_paged_attention.nsl`, `examples/m27_rope_gqa.nsl`
- `tests/expected/m27_*.txt`

**Modified files:**
- `crates/nsl-runtime/src/cuda/mod.rs` — `kernel_launch()` gains `shared_mem_bytes` param
- `crates/nsl-runtime/src/lib.rs` — `pub mod flash_attention;`
- `crates/nsl-codegen/src/lib.rs` — `pub mod flash_attention;`
- `crates/nsl-codegen/src/builtins.rs` — register FFI functions
- `crates/nsl-codegen/src/compiler.rs` — `@flash_attention` lowering in decorated fn handler
- `crates/nsl-semantic/src/builtins.rs` — `scaled_dot_product_attention` type signature
- `crates/nsl-semantic/src/checker.rs` — decorator validation for `@flash_attention`, `@rope`, `@gqa`
- `crates/nsl-cli/tests/e2e.rs` — 3 new E2E tests
