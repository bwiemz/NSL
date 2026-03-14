# M27: FlashAttention-2 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement FlashAttention-2 as hand-optimized PTX kernel templates with paged KV, RoPE, and GQA variants, exposed via `scaled_dot_product_attention()` stdlib intrinsic with decorator-driven lowering.

**Architecture:** PTX string synthesis in `nsl-codegen` (AOT, compile-time), launch wrappers in `nsl-runtime` (runtime). Five kernel variants from one parameterized template with orthogonal flags (`paged`, `rope_q`, `rope_style`, `gqa_group_size`, `causal`). `@autotune` explores tile sizes; winning PTX baked into `.rodata`. Naive fallback path for non-`@flash_attention` uses.

**Tech Stack:** Rust, Cranelift, PTX ISA 7.0, cudarc 0.19

**Spec:** `docs/superpowers/specs/2026-03-14-m27-flashattention-design.md`

---

## File Structure

**New files (4 source + 6 test/example):**

| File | Responsibility | Est. Lines |
|------|---------------|------------|
| `crates/nsl-codegen/src/flash_attention.rs` | PTX string synthesis for all 5 kernel variants. Parameterized by `(block_q, block_kv, head_dim, causal, paged, rope_q, rope_style, gqa_group_size)`. Returns `Vec<u8>` (null-terminated PTX). | 800–1200 |
| `crates/nsl-runtime/src/flash_attention.rs` | `extern "C"` FFI launch wrappers: `nsl_flash_attention()`, `nsl_rope_cache_write()`. Compute grid/block/shared_mem, marshal args, call `kernel_launch()`. No PTX generation. | 150–200 |
| `examples/m27_flash_attention.nsl` | Basic FlashAttention E2E — `scaled_dot_product_attention` with `@flash_attention(causal=true)` | 30–40 |
| `examples/m27_paged_attention.nsl` | Paged variant E2E — `@flash_attention` + `@paged_kv` | 30–40 |
| `examples/m27_rope_gqa.nsl` | Composed variant E2E — `@flash_attention` + `@rope` + `@gqa` | 30–40 |
| `tests/expected/m27_flash_attention.txt` | Expected output for basic E2E | 5–10 |
| `tests/expected/m27_paged_attention.txt` | Expected output for paged E2E | 5–10 |
| `tests/expected/m27_rope_gqa.txt` | Expected output for composed E2E | 5–10 |

**Modified files (9):**

| File | Change | Lines Affected |
|------|--------|---------------|
| `crates/nsl-runtime/src/cuda/mod.rs` | Add `shared_mem_bytes: u32` to `kernel_launch()` and `nsl_kernel_launch()`. Update all 8 internal call sites to pass `0`. Update `cuLaunchKernel` call. | ~266–338, ~379–824 |
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod flash_attention;` | +1 line |
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod flash_attention;` | +1 line |
| `crates/nsl-codegen/src/builtins.rs` | Add `nsl_flash_attention` (22 params) and `nsl_rope_cache_write` (16 params) FFI signatures. Update `nsl_kernel_launch` from 10 to 11 params (add `shared_mem_bytes`). All params `I64` for Cranelift ABI. | ~216, +6 lines |
| `crates/nsl-codegen/src/expr.rs` | Add `shared_mem_bytes=0` to `compile_kernel_call()` args for `nsl_kernel_launch`. Add `scaled_dot_product_attention()` lowering (naive path + flash path). | ~2988–3016, +80 lines |
| `crates/nsl-codegen/src/compiler.rs` | Add `compile_flash_attention_kernels()` pass in main pipeline. Track flash_attention config per model function. | ~2026, +60 lines |
| `crates/nsl-semantic/src/builtins.rs` | Register `scaled_dot_product_attention` type signature. | +8 lines |
| `crates/nsl-semantic/src/checker.rs` | Validate `@flash_attention`, `@rope`, `@gqa` decorators. Relax `@autotune` to accept `@flash_attention`-decorated `fn`. | ~377–420, +60 lines |
| `crates/nsl-cli/tests/e2e.rs` | Add 3 new E2E tests. | +15 lines |

---

## Chunk 1: Foundation — `shared_mem_bytes` Plumbing + Runtime Launch Wrappers

This chunk adds the `shared_mem_bytes` parameter throughout the kernel launch pipeline and creates the FlashAttention runtime launch wrappers (no PTX generation yet).

### Task 1: Add `shared_mem_bytes` to `kernel_launch()` Internal Function

**Files:**
- Modify: `crates/nsl-runtime/src/cuda/mod.rs:266-338`

**Context:** The internal `kernel_launch()` function in `mod inner` currently hardcodes `shared_mem_bytes = 0` at line 317 in the `cuLaunchKernel` call. FlashAttention needs to pass `(block_q + block_kv) * head_dim * sizeof(f16)` bytes of shared memory. This task adds the parameter and forwards it to `cuLaunchKernel`.

- [ ] **Step 1: Read the current `kernel_launch` function**

Read `crates/nsl-runtime/src/cuda/mod.rs` lines 264–338 to understand the full function.

- [ ] **Step 2: Add `shared_mem_bytes: u32` parameter**

Change the function signature from:
```rust
pub(crate) fn kernel_launch(
    ptx_ptr: *const u8,
    name_ptr: *const u8,
    grid: [i64; 3],
    block: [i64; 3],
    args: &[*mut c_void],
) -> CUresult {
```
to:
```rust
pub(crate) fn kernel_launch(
    ptx_ptr: *const u8,
    name_ptr: *const u8,
    grid: [i64; 3],
    block: [i64; 3],
    args: &[*mut c_void],
    shared_mem_bytes: u32,
) -> CUresult {
```

- [ ] **Step 3: Forward `shared_mem_bytes` to `cuLaunchKernel`**

Change line 317 from:
```rust
0, std::ptr::null_mut(),
```
to:
```rust
shared_mem_bytes, std::ptr::null_mut(),
```

- [ ] **Step 4: Update all 8 internal call sites to pass `0`**

Each of these call sites currently calls `inner::kernel_launch(ptx, name, grid, block, &args)`. Add `, 0` as the final argument:

1. Line 379 (`gpu_elementwise_binary`): add `, 0` after `&args`
2. Line 416 (`gpu_elementwise_unary`): add `, 0` after `&args`
3. Line 483 (`gpu_matmul_f32`): add `, 0` after `&args`
4. Line 526 (`gpu_scalar_op`): add `, 0` after `&args`
5. Line 570 (`gpu_backward_binary`): add `, 0` after `&args`
6. Line 661 (`gpu_clamp_backward`): add `, 0` after `&args`
7. Line 718 (`nsl_kernel_launch` FFI): add `, 0` after `args_slice`
8. Line 818 (`test_vec_add_kernel_launch`): add `, 0` after `&args.map(|p| p)`

- [ ] **Step 5: Verify compilation**

Run: `cargo build -p nsl-runtime`
Expected: Compiles successfully (all call sites updated)

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/cuda/mod.rs
git commit -m "feat(m27): add shared_mem_bytes parameter to kernel_launch()"
```

---

### Task 2: Add `shared_mem_bytes` to `nsl_kernel_launch` FFI Export

**Files:**
- Modify: `crates/nsl-runtime/src/cuda/mod.rs:700-734`
- Modify: `crates/nsl-codegen/src/builtins.rs:216`
- Modify: `crates/nsl-codegen/src/expr.rs:2988-3016`

**Context:** The `nsl_kernel_launch` FFI function is called from Cranelift-generated code for user-defined `kernel` blocks. It currently takes 10 i64 params. This task adds `shared_mem_bytes` as an 11th param (i64 for Cranelift ABI, cast to u32 internally). The codegen side must also emit this new argument.

- [ ] **Step 1: Update `nsl_kernel_launch` FFI signature**

In `crates/nsl-runtime/src/cuda/mod.rs`, change `nsl_kernel_launch` (line 701) to add `shared_mem_bytes: i64` as the 11th parameter:

```rust
#[no_mangle]
pub extern "C" fn nsl_kernel_launch(
    ptx_ptr: i64,
    name_ptr: i64,
    grid_x: i64,
    grid_y: i64,
    grid_z: i64,
    block_x: i64,
    block_y: i64,
    block_z: i64,
    args_ptr: i64,
    num_args: i64,
    shared_mem_bytes: i64,
) -> i64 {
```

Forward to inner: change `args_slice,` (line 723) to `args_slice, shared_mem_bytes as u32,`

Update the `#[cfg(not(feature = "cuda"))]` block to also suppress the new param:
```rust
let _ = (block_x, block_y, block_z, args_ptr, num_args, shared_mem_bytes);
```

- [ ] **Step 2: Update codegen builtin registration**

In `crates/nsl-codegen/src/builtins.rs` line 216, change from 10 params to 11:

```rust
("nsl_kernel_launch", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
```

- [ ] **Step 3: Update `compile_kernel_call` in expr.rs to pass `shared_mem_bytes = 0`**

In `crates/nsl-codegen/src/expr.rs`, in the `compile_kernel_call` method:

At the zero-args path (~line 2988), add a shared_mem_bytes=0 argument:
```rust
let shared_mem = builder.ins().iconst(cl_types::I64, 0);
return self.compile_call_by_name(
    builder,
    "nsl_kernel_launch",
    &[ptx_ptr, name_ptr, grid_x, grid_y, grid_z, block_x, block_y, block_z, null_ptr, num_args_val, shared_mem],
);
```

At the normal path (~line 3012), add the same:
```rust
let shared_mem = builder.ins().iconst(cl_types::I64, 0);
self.compile_call_by_name(
    builder,
    "nsl_kernel_launch",
    &[ptx_ptr, name_ptr, grid_x, grid_y, grid_z, block_x, block_y, block_z, args_ptr, num_args_val, shared_mem],
)
```

- [ ] **Step 4: Verify full build**

Run: `cargo build -p nsl-codegen -p nsl-runtime`
Expected: Compiles. All existing kernel blocks now pass `shared_mem_bytes=0`, preserving behavior.

- [ ] **Step 5: Run existing tests**

Run: `cargo test -p nsl-cli --test e2e`
Expected: All existing E2E tests pass (no behavioral change).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/cuda/mod.rs crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(m27): add shared_mem_bytes to nsl_kernel_launch FFI and codegen"
```

---

### Task 3: Create Runtime FlashAttention Launch Wrappers

**Files:**
- Create: `crates/nsl-runtime/src/flash_attention.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

**Context:** These are `extern "C"` FFI functions that compute grid/block dimensions from tensor shapes, marshal arguments, and call `kernel_launch()` with the pre-baked PTX pointer. On non-CUDA builds, they fall back to the naive attention path (matmul + softmax + masking). The PTX pointer is passed as a parameter from codegen — the runtime does NOT choose variants.

- [ ] **Step 1: Create `flash_attention.rs` with `nsl_flash_attention` wrapper**

Create `crates/nsl-runtime/src/flash_attention.rs`:

```rust
//! FlashAttention-2 runtime launch wrappers.
//!
//! These functions compute grid/block dimensions from tensor shapes, marshal
//! arguments, and call `kernel_launch()` with pre-baked PTX from .rodata.
//! No PTX generation happens at runtime. On non-CUDA builds, falls back to
//! naive matmul+softmax attention path.

use std::ffi::c_void;

/// FlashAttention-2 kernel launch wrapper.
///
/// All params are i64 for Cranelift ABI compatibility (same pattern as nsl_kernel_launch).
/// f32 scale is passed as i64 and reconstructed via f32::from_bits(scale as u32).
///
/// Returns 0 on success, non-zero CUDA error code on failure.
#[no_mangle]
pub extern "C" fn nsl_flash_attention(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, scale_bits: i64,  // f32 bits as i64
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    // Paged KV params (zero/null if not paged variant)
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    // RoPE params (zero/null if not rope variant)
    cos_ptr: i64, sin_ptr: i64,
    // Ragged batch params (M29-ready)
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    // PTX and kernel name from .rodata
    ptx_ptr: i64, name_ptr: i64,
    // Tile sizes (needed for grid computation)
    block_q: i64, _block_kv: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let scale = f32::from_bits(scale_bits as u32);

        // Grid: (ceil(seq_len / block_q), batch * heads, 1)
        let grid_x = (seq_len + block_q - 1) / block_q;
        let grid_y = batch * heads;
        let grid_z = 1i64;

        // Block: (128, 1, 1) — 4 warps per thread block
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // Marshal all kernel arguments as u64 values
        let mut q = q_ptr as u64;
        let mut k = k_ptr as u64;
        let mut v = v_ptr as u64;
        let mut out = out_ptr as u64;
        let mut s = scale;
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;

        let args: [*mut c_void; 17] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
        ];

        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv);
        // Fallback: naive attention path handled by codegen (not runtime)
        eprintln!("[nsl] FlashAttention requires CUDA. Use naive path (no @flash_attention decorator).");
        -1
    }
}

/// RoPE + paged cache write kernel launch wrapper.
///
/// All params i64 for Cranelift ABI compatibility.
/// Grid: (num_tokens, num_heads, ceil(head_dim/2))
#[no_mangle]
pub extern "C" fn nsl_rope_cache_write(
    k_projected_ptr: i64, v_projected_ptr: i64,
    cos_ptr: i64, sin_ptr: i64,
    positions_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_table_ptr: i64,
    // Ragged batch params (M29-ready)
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    num_tokens: i64, num_heads: i64, head_dim: i64, block_size: i64,
    // PTX and kernel name from .rodata
    ptx_ptr: i64, name_ptr: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        // Grid: (num_tokens, num_heads, ceil(head_dim/2))
        // Token dim on blockIdx.x for 2^31 limit (not blockIdx.z which is 65535)
        let grid_x = num_tokens;
        let grid_y = num_heads;
        let grid_z = (head_dim + 1) / 2;

        // Block: (1, 1, 1) — elementwise, one thread per element
        let block_x = 1i64;
        let block_y = 1i64;
        let block_z = 1i64;

        let mut kp = k_projected_ptr as u64;
        let mut vp = v_projected_ptr as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut pos = positions_ptr as u64;
        let mut k_pool = k_pool_ptr as u64;
        let mut v_pool = v_pool_ptr as u64;
        let mut bt = block_table_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut nt = num_tokens as u64;
        let mut nh = num_heads as u64;
        let mut hd = head_dim as u64;
        let mut bs = block_size as u64;

        let args: [*mut c_void; 14] = [
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut k_pool as *mut _ as *mut c_void,
            &mut v_pool as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];

        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            0, // elementwise — no shared memory
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (k_projected_ptr, v_projected_ptr, cos_ptr, sin_ptr, positions_ptr);
        let _ = (k_pool_ptr, v_pool_ptr, block_table_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (num_tokens, num_heads, head_dim, block_size, ptx_ptr, name_ptr);
        eprintln!("[nsl] rope_cache_write requires CUDA.");
        -1
    }
}
```

- [ ] **Step 2: Register module in lib.rs**

In `crates/nsl-runtime/src/lib.rs`, add after line 54 (`pub mod kernel_profiler;`):
```rust
pub mod flash_attention;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p nsl-runtime`
Expected: Compiles. The `#[cfg(feature = "cuda")]` path references `crate::cuda::inner::kernel_launch` which exists. The non-CUDA path prints a warning.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/flash_attention.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(m27): add FlashAttention and rope_cache_write runtime launch wrappers"
```

---

### Task 4: Register FFI Functions in Codegen Builtins

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`

**Context:** The codegen `RUNTIME_FUNCTIONS` array tells Cranelift about every FFI function the runtime exports. Each entry is `(name, &[param_types], return_type)`. All params are `I64` for Cranelift ABI compatibility (even `f32` is passed as i64 bits and reconstructed via `f32::from_bits(val as u32)` in the wrapper). `nsl_flash_attention` has 22 params (18 tensor/metadata + ptx_ptr + name_ptr + block_q + block_kv). `nsl_rope_cache_write` has 16 params.

- [ ] **Step 1: Add `nsl_flash_attention` and `nsl_rope_cache_write` entries**

In `crates/nsl-codegen/src/builtins.rs`, add after the `nsl_kernel_launch` line (~216):

```rust
    // FlashAttention-2 launch wrappers (M27)
    ("nsl_flash_attention", &[
        types::I64, types::I64, types::I64, types::I64, types::I64,  // q, k, v, out, scale
        types::I64, types::I64, types::I64, types::I64,              // batch, heads, seq_len, head_dim
        types::I64, types::I64, types::I64, types::I64,              // block_table, k_pool, v_pool, block_size
        types::I64, types::I64,                                       // cos, sin (RoPE)
        types::I64, types::I64,                                       // seq_ids, seq_lens (M29-ready)
        types::I64,                                                   // shared_mem_bytes
        types::I64, types::I64,                                       // ptx_ptr, name_ptr
        types::I64, types::I64,                                       // block_q, block_kv
    ], Some(types::I64)),
    ("nsl_rope_cache_write", &[
        types::I64, types::I64,                                       // k_projected, v_projected
        types::I64, types::I64, types::I64,                          // cos, sin, positions
        types::I64, types::I64, types::I64,                          // k_pool, v_pool, block_table
        types::I64, types::I64,                                       // seq_ids, seq_lens (M29-ready)
        types::I64, types::I64, types::I64, types::I64,              // num_tokens, num_heads, head_dim, block_size
        types::I64, types::I64,                                       // ptx_ptr, name_ptr
    ], Some(types::I64)),
```

- [ ] **Step 2: Verify compilation**

Run: `cargo build -p nsl-codegen`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/builtins.rs
git commit -m "feat(m27): register nsl_flash_attention and nsl_rope_cache_write FFI builtins"
```

---

## Chunk 2: PTX Templates — FlashAttention-2 Kernel Synthesis

This chunk implements the core PTX string synthesis module in codegen. All 5 kernel variants are generated from a single parameterized template.

### Task 5: Create PTX Template Foundation — Base FlashAttention-2

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

**Context:** This is the largest file in M27. It synthesizes PTX strings at compile time (AOT). The base `flash_attention_2` kernel implements online softmax with tiled accumulation: Q tile in SRAM, iterate K/V tiles with 3× `bar.sync` per iteration, accumulate O in registers. All intermediate math in f32, inputs/outputs f16.

The PTX template is parameterized by a `FlashAttentionConfig` struct:

```rust
pub struct FlashAttentionConfig {
    pub block_q: i64,       // Q tile rows (e.g., 64)
    pub block_kv: i64,      // K/V tile rows (e.g., 64)
    pub head_dim: i64,      // head dimension (e.g., 128)
    pub causal: bool,       // clamp loop bound to diagonal
    pub paged: bool,        // K/V loads use block table indirection
    pub rope_q: bool,       // Q load includes RoPE rotation
    pub rope_style: RopeStyle,  // half_split or adjacent
    pub gqa_group_size: u32,    // 1 = standard MHA
}
// Note: `scale` (1/sqrt(head_dim)) is a RUNTIME parameter passed via PTX .param,
// not a compile-time template constant. It does not affect PTX structure.
// The spec lists it in the variant composition table (Section 3.3) for completeness.

pub enum RopeStyle {
    HalfSplit,  // LLaMA, Qwen, Mistral — stride = head_dim/2
    Adjacent,   // GPT-NeoX, GPT-J — stride = 1
}
```

- [ ] **Step 1: Create `flash_attention.rs` with config structs and module skeleton**

Create `crates/nsl-codegen/src/flash_attention.rs`:

```rust
//! FlashAttention-2 PTX template synthesis.
//!
//! Generates PTX kernel strings at compile time (AOT). Each variant is parameterized
//! by orthogonal flags (paged, rope_q, rope_style, gqa_group_size, causal) and tile
//! sizes (block_q, block_kv). The generated PTX is embedded in .rodata and launched
//! by the runtime wrappers in nsl-runtime/src/flash_attention.rs.

/// RoPE interleaving style.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RopeStyle {
    /// (x[i], x[i + head_dim/2]) — LLaMA, Qwen, Mistral
    HalfSplit,
    /// (x[2i], x[2i+1]) — GPT-NeoX, GPT-J
    Adjacent,
}

/// Configuration for a FlashAttention PTX kernel variant.
#[derive(Clone, Debug)]
pub struct FlashAttentionConfig {
    pub block_q: i64,
    pub block_kv: i64,
    pub head_dim: i64,
    pub causal: bool,
    pub paged: bool,
    pub rope_q: bool,
    pub rope_style: RopeStyle,
    pub gqa_group_size: u32,
}

/// Generate PTX for the FlashAttention-2 kernel with the given configuration.
///
/// Returns null-terminated PTX bytes ready for .rodata embedding.
pub fn synthesize_flash_attention_ptx(config: &FlashAttentionConfig) -> Vec<u8> {
    let mut ptx = String::with_capacity(8192);
    let kernel_name = flash_attention_kernel_name(config);

    // PTX header
    emit_ptx_header(&mut ptx);

    // Kernel entry point
    emit_flash_attention_entry(&mut ptx, &kernel_name, config);

    // Null-terminate
    ptx.push('\0');
    ptx.into_bytes()
}

/// Generate PTX for the rope_cache_write elementwise kernel.
///
/// Returns null-terminated PTX bytes.
pub fn synthesize_rope_cache_write_ptx(
    head_dim: i64,
    rope_style: RopeStyle,
) -> Vec<u8> {
    let mut ptx = String::with_capacity(4096);

    emit_ptx_header(&mut ptx);
    emit_rope_cache_write_entry(&mut ptx, head_dim, rope_style);

    ptx.push('\0');
    ptx.into_bytes()
}

/// Compute the kernel name encoding variant flags and tile sizes.
///
/// Format: `flash_attn_p{paged}_r{rope}_g{gqa}_c{causal}_q{block_q}_kv{block_kv}`
pub fn flash_attention_kernel_name(config: &FlashAttentionConfig) -> String {
    format!(
        "flash_attn_p{}_r{}_{}_g{}_c{}_q{}_kv{}",
        config.paged as u8,
        config.rope_q as u8,
        match config.rope_style {
            RopeStyle::HalfSplit => "hs",
            RopeStyle::Adjacent => "adj",
        },
        config.gqa_group_size,
        config.causal as u8,
        config.block_q,
        config.block_kv,
    )
}

/// Compute shared memory bytes for a given config.
///
/// Formula: (block_q + block_kv) * head_dim * sizeof(f16)
/// where sizeof(f16) = 2.
pub fn shared_mem_bytes(config: &FlashAttentionConfig) -> u32 {
    ((config.block_q + config.block_kv) * config.head_dim * 2) as u32
}

// ── PTX emission helpers ──────────────────────────────────────────

fn emit_ptx_header(ptx: &mut String) {
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_52\n");
    ptx.push_str(".address_size 64\n\n");
}

fn emit_flash_attention_entry(
    ptx: &mut String,
    kernel_name: &str,
    config: &FlashAttentionConfig,
) {
    // Parameter declarations — ALWAYS declare ALL params regardless of variant flags.
    // The runtime wrapper always passes the full 17-arg set (unused params are null/zero).
    // Conditional params are simply ignored in the kernel body when the flag is off.
    // This ensures cuLaunchKernel arg alignment is always correct.
    ptx.push_str(&format!(".visible .entry {} (\n", kernel_name));
    ptx.push_str("    .param .u64 q_ptr,\n");
    ptx.push_str("    .param .u64 k_ptr,\n");
    ptx.push_str("    .param .u64 v_ptr,\n");
    ptx.push_str("    .param .u64 out_ptr,\n");
    ptx.push_str("    .param .f32 scale,\n");
    ptx.push_str("    .param .u64 batch,\n");
    ptx.push_str("    .param .u64 heads,\n");
    ptx.push_str("    .param .u64 seq_len,\n");
    ptx.push_str("    .param .u64 head_dim,\n");
    // Paged KV params (null/zero when paged=false, but always declared)
    ptx.push_str("    .param .u64 block_table_ptr,\n");
    ptx.push_str("    .param .u64 k_pool_ptr,\n");
    ptx.push_str("    .param .u64 v_pool_ptr,\n");
    ptx.push_str("    .param .u64 block_size,\n");
    // RoPE params (null when rope_q=false, but always declared)
    ptx.push_str("    .param .u64 cos_ptr,\n");
    ptx.push_str("    .param .u64 sin_ptr,\n");
    // M29-ready ragged batch params
    ptx.push_str("    .param .u64 seq_ids_ptr,\n");
    ptx.push_str("    .param .u64 seq_lens_ptr\n");

    ptx.push_str(")\n");
    ptx.push_str("{\n");

    // Shared memory declaration (must be inside kernel body for ptxas)
    let shmem_bytes = shared_mem_bytes(config);
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 shmem[{}];\n",
        shmem_bytes
    ));

    // Register declarations
    emit_register_declarations(ptx, config);

    // Load parameters
    emit_param_loads(ptx, config);

    // Compute thread/block indices
    emit_index_computation(ptx, config);

    // Load Q tile into shared memory
    emit_q_tile_load(ptx, config);

    // Initialize accumulators (O_acc=0, row_max=-inf, row_sum=0)
    emit_accumulator_init(ptx, config);

    // Main K/V tile loop with online softmax
    emit_kv_tile_loop(ptx, config);

    // Finalize: O = O_acc / row_sum
    emit_finalize(ptx, config);

    // Store output tile to global memory
    emit_output_store(ptx, config);

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

fn emit_register_declarations(ptx: &mut String, config: &FlashAttentionConfig) {
    // Thread indexing registers
    ptx.push_str("    .reg .u32 %tid_x, %bid_x, %bid_y;\n");
    ptx.push_str("    .reg .u64 %rd<64>;\n");
    ptx.push_str("    .reg .f32 %f<128>;\n");
    ptx.push_str("    .reg .b16 %h<32>;\n");  // f16 registers for output conversion
    ptx.push_str("    .reg .pred %p<16>;\n");
    ptx.push_str("    .reg .u32 %r<32>;\n");

    // Scale register
    ptx.push_str("    .reg .f32 %scale;\n");

    // LOG2E constant for exp() via ex2.approx
    ptx.push_str("    .reg .f32 %log2e;\n");
    ptx.push_str("    mov.f32 %log2e, 0x3FB8AA3B;  // 1.4426950408 (log2(e))\n");

    // Loop counter for K/V tile iteration
    ptx.push_str("    .reg .u64 %k_start, %k_max;\n");

    // Accumulator registers for online softmax
    // row_max, row_sum, correction per Q row handled by this thread
    ptx.push_str("    .reg .f32 %row_max, %row_sum, %correction;\n");
    ptx.push_str("    .reg .f32 %new_max, %old_max;\n");

    // Warp reduction temporaries (shfl.sync.bfly)
    ptx.push_str("    .reg .f32 %shfl_tmp;\n");

    if config.rope_q {
        ptx.push_str("    .reg .f32 %cos_val, %sin_val;\n");
        ptx.push_str("    .reg .f32 %q_a, %q_b, %q_rot_a, %q_rot_b;\n");
    }

    // O_acc registers: each thread accumulates a subset of the [block_q, head_dim] output tile
    // Total O_acc regs per thread = (block_q * head_dim) / blockDim.x
    // Example: block_q=64, head_dim=128, blockDim.x=128 → 64 registers
    // Declared dynamically based on config in emit_accumulator_init
    let _ = config;
}

fn emit_param_loads(ptx: &mut String, config: &FlashAttentionConfig) {
    // Always load ALL params — PTX entry declares them all regardless of variant.
    // Unused params (null/zero) are simply never referenced in the kernel body.
    ptx.push_str("    ld.param.u64 %rd0, [q_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd1, [k_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd2, [v_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd3, [out_ptr];\n");
    ptx.push_str("    ld.param.f32 %scale, [scale];\n");
    ptx.push_str("    ld.param.u64 %rd4, [batch];\n");
    ptx.push_str("    ld.param.u64 %rd5, [heads];\n");
    ptx.push_str("    ld.param.u64 %rd6, [seq_len];\n");
    ptx.push_str("    ld.param.u64 %rd7, [head_dim];\n");
    // Paged params (always loaded; only used when paged=true)
    ptx.push_str("    ld.param.u64 %rd8, [block_table_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd9, [k_pool_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd10, [v_pool_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd11, [block_size];\n");
    // RoPE params (always loaded; only used when rope_q=true)
    ptx.push_str("    ld.param.u64 %rd12, [cos_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd13, [sin_ptr];\n");
    // Ragged batch params
    ptx.push_str("    ld.param.u64 %rd14, [seq_ids_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd15, [seq_lens_ptr];\n");

    let _ = config; // variant flags control which registers are USED, not loaded
}

fn emit_index_computation(ptx: &mut String, _config: &FlashAttentionConfig) {
    // threadIdx.x, blockIdx.x (Q tile index), blockIdx.y (batch*head index)
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");

    // q_start = blockIdx.x * block_q
    // batch_head_idx = blockIdx.y
    // head_idx = batch_head_idx % heads
    // batch_idx = batch_head_idx / heads
    ptx.push_str("    // q_start = blockIdx.x * block_q\n");
    ptx.push_str("    // batch_head routing computed from blockIdx.y\n");
}

fn emit_q_tile_load(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // Load Q tile into shared memory\n");
    if config.rope_q {
        ptx.push_str("    // RoPE: load Q from global, rotate in registers, store to SRAM\n");
        ptx.push_str("    // cos/sin loaded from global memory into registers (NOT SRAM)\n");
        let stride_comment = match config.rope_style {
            RopeStyle::HalfSplit => "stride = head_dim/2 (half_split)",
            RopeStyle::Adjacent => "stride = 1 (adjacent)",
        };
        ptx.push_str(&format!("    // RoPE style: {}\n", stride_comment));
    }
    ptx.push_str("    bar.sync 0;\n");
    ptx.push_str("    // Q tile now in shmem[0 .. block_q * head_dim * 2]\n");
}

fn emit_accumulator_init(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("    // Initialize accumulators\n");
    ptx.push_str("    // O_acc = 0, row_max = -inf, row_sum = 0\n");
    ptx.push_str("    mov.f32 %row_max, 0xFF800000;  // -inf as IEEE 754\n");
    ptx.push_str("    mov.f32 %row_sum, 0x00000000;  // 0.0\n");
}

fn emit_kv_tile_loop(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    // === Main K/V tile loop ===\n");

    if config.causal {
        ptx.push_str("    // Causal: k_max = min(seq_len, q_start + block_q)\n");
        ptx.push_str("    // Zero-divergence — loop naturally terminates at diagonal\n");
    } else {
        ptx.push_str("    // Non-causal: k_max = seq_len\n");
    }

    ptx.push_str("LOOP_KV_START:\n");

    // Phase 1: Load K tile
    ptx.push_str("    // Phase 1: Load K tile into SRAM\n");
    if config.paged {
        ptx.push_str("    // Paged: block table indirection per physical block\n");
        ptx.push_str("    // One page table lookup per block_size tokens\n");
    }
    if config.gqa_group_size > 1 {
        ptx.push_str(&format!(
            "    // GQA: kv_head = q_head / {} (compile-time literal)\n",
            config.gqa_group_size
        ));
    }
    ptx.push_str("    bar.sync 0;  // FENCE 1: K tile fully in SRAM\n");

    // Phase 2: Compute S = Q @ K^T, apply scale
    ptx.push_str("    // Phase 2: S = Q_tile @ K_tile^T (registers)\n");
    ptx.push_str("    // S[i][j] *= scale  (mul.f32)\n");
    if config.causal {
        ptx.push_str("    // Partial causal mask on diagonal tile: S[i][j] = -inf where k_start+j > q_start+i\n");
    }
    ptx.push_str("    bar.sync 0;  // FENCE 2: all warps done reading K before SRAM overwrite\n");

    // Phase 3: Online softmax
    ptx.push_str("    // Phase 3: Online softmax — S→P in-place in registers\n");
    ptx.push_str("    // Warp-level reductions via shfl.sync.bfly for row_max, row_sum\n");
    ptx.push_str("    // new_max = max(row_max, warp_reduce_max(row_max_of_S))\n");
    ptx.push_str("    // correction = exp(old_max - new_max)  // <= 1.0, no overflow\n");
    ptx.push_str("    // row_sum = row_sum * correction + warp_reduce_sum(exp(S - new_max))\n");
    ptx.push_str("    // O_acc *= correction\n");
    ptx.push_str("    // P = exp(S - new_max)  // overwrites S registers in-place\n");

    // Phase 4: Load V tile (reuses K SRAM)
    ptx.push_str("    // Phase 4: Load V tile (reuses shmem_K address)\n");
    if config.paged {
        ptx.push_str("    // Paged V load: same block table indirection as K\n");
    }
    ptx.push_str("    bar.sync 0;  // FENCE 3: V tile fully in SRAM\n");

    // Phase 5: Accumulate O
    ptx.push_str("    // Phase 5: O_acc += P @ V_tile\n");
    ptx.push_str("    // P in registers (Phase 3), V in SRAM\n");

    // Loop back
    ptx.push_str("    // Increment k_start, check loop bound\n");
    ptx.push_str("    bra LOOP_KV_START;\n");
    ptx.push_str("LOOP_KV_END:\n");
}

fn emit_finalize(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("    // Finalize: O = O_acc / row_sum\n");
}

fn emit_output_store(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("    // Store output tile to global memory\n");
}

fn emit_rope_cache_write_entry(
    ptx: &mut String,
    head_dim: i64,
    rope_style: RopeStyle,
) {
    ptx.push_str(".visible .entry nsl_rope_cache_write (\n");
    ptx.push_str("    .param .u64 k_projected_ptr,\n");
    ptx.push_str("    .param .u64 v_projected_ptr,\n");
    ptx.push_str("    .param .u64 cos_ptr,\n");
    ptx.push_str("    .param .u64 sin_ptr,\n");
    ptx.push_str("    .param .u64 positions_ptr,\n");
    ptx.push_str("    .param .u64 k_pool_ptr,\n");
    ptx.push_str("    .param .u64 v_pool_ptr,\n");
    ptx.push_str("    .param .u64 block_table_ptr,\n");
    ptx.push_str("    .param .u64 seq_ids_ptr,\n");
    ptx.push_str("    .param .u64 seq_lens_ptr,\n");
    ptx.push_str("    .param .u64 num_tokens,\n");
    ptx.push_str("    .param .u64 num_heads,\n");
    ptx.push_str("    .param .u64 head_dim,\n");
    ptx.push_str("    .param .u64 block_size\n");
    ptx.push_str(")\n");
    ptx.push_str("{\n");

    ptx.push_str("    .reg .u32 %tid_x, %bid_x, %bid_y, %bid_z;\n");
    ptx.push_str("    .reg .u64 %rd<32>;\n");
    ptx.push_str("    .reg .f32 %f<16>;\n");
    ptx.push_str("    .reg .f32 %cos_val, %sin_val, %k_a, %k_b, %k_rot_a, %k_rot_b;\n");

    ptx.push_str("    // Grid: (num_tokens, num_heads, ceil(head_dim/2))\n");
    ptx.push_str("    // token_idx = blockIdx.x (up to 2^31-1)\n");
    ptx.push_str("    // head_idx = blockIdx.y\n");
    ptx.push_str("    // dim_pair = blockIdx.z\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");
    ptx.push_str("    mov.u32 %bid_z, %ctaid.z;\n");

    let stride_comment = match rope_style {
        RopeStyle::HalfSplit => format!("stride = {} (half_split)", head_dim / 2),
        RopeStyle::Adjacent => "stride = 1 (adjacent)".to_string(),
    };
    ptx.push_str(&format!("    // RoPE style: {}\n", stride_comment));

    ptx.push_str("    // 1. Load K element pair from k_projected into registers\n");
    ptx.push_str("    // 2. Load cos[pos], sin[pos] from frequency table → registers\n");
    ptx.push_str("    // 3. Apply RoPE: k_rot_a = k_a*cos - k_b*sin; k_rot_b = k_a*sin + k_b*cos\n");
    ptx.push_str("    // 4. Look up physical block via block_table[seq_id * max_blocks + logical_idx]\n");
    ptx.push_str("    // 5. Write rotated K into paged K pool\n");
    ptx.push_str("    // 6. Write V directly into paged V pool (no rotation)\n");

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let _ = head_dim; // used in stride computation
}
```

**Implementation note:** The PTX body above is a **structural skeleton** — the `emit_*` helper functions contain commented pseudocode that maps 1:1 to the spec's algorithm (Section 4.1). The actual register-level PTX instructions (ld.global, mul.f32, ex2.approx, bar.sync, shfl.sync.bfly, etc.) will be filled in during implementation. Each helper is a self-contained function that can be implemented and tested independently.

- [ ] **Step 2: Register module in codegen lib.rs**

In `crates/nsl-codegen/src/lib.rs`, add after line 8 (`pub mod fusion;`):
```rust
pub mod flash_attention;
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p nsl-codegen`
Expected: Compiles. The module is pure Rust string synthesis — no external dependencies.

- [ ] **Step 4: Write unit test for PTX synthesis**

Add to the bottom of `crates/nsl-codegen/src/flash_attention.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_name_encoding() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
        };
        assert_eq!(
            flash_attention_kernel_name(&config),
            "flash_attn_p0_r0_hs_g1_c1_q64_kv64"
        );
    }

    #[test]
    fn test_kernel_name_full_variant() {
        let config = FlashAttentionConfig {
            block_q: 128,
            block_kv: 32,
            head_dim: 64,
            causal: true,
            paged: true,
            rope_q: true,
            rope_style: RopeStyle::Adjacent,
            gqa_group_size: 4,
        };
        assert_eq!(
            flash_attention_kernel_name(&config),
            "flash_attn_p1_r1_adj_g4_c1_q128_kv32"
        );
    }

    #[test]
    fn test_shared_mem_bytes_computation() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
        };
        // (64 + 64) * 128 * 2 = 32768 bytes (32 KB)
        assert_eq!(shared_mem_bytes(&config), 32768);
    }

    #[test]
    fn test_shared_mem_within_48kb_limit() {
        // block_q=128, block_kv=64, head_dim=128
        // (128 + 64) * 128 * 2 = 49152 > 48KB → would exceed sm_52 default
        let config = FlashAttentionConfig {
            block_q: 128,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
        };
        assert_eq!(shared_mem_bytes(&config), 49152);
        // This exceeds 48KB — the semantic checker should reject this combination
    }

    #[test]
    fn test_ptx_synthesis_produces_valid_header() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap(); // strip null
        assert!(ptx_str.starts_with(".version 7.0\n"));
        assert!(ptx_str.contains(".target sm_52"));
        assert!(ptx_str.contains(&flash_attention_kernel_name(&config)));
        assert!(ptx_str.contains("bar.sync 0"));
        assert!(ptx_str.contains("FENCE 1"));
        assert!(ptx_str.contains("FENCE 2"));
        assert!(ptx_str.contains("FENCE 3"));
    }

    #[test]
    fn test_ptx_causal_flag() {
        let mut config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
        };
        let ptx_no_causal = synthesize_flash_attention_ptx(&config);
        let str_no = std::str::from_utf8(&ptx_no_causal[..ptx_no_causal.len()-1]).unwrap();
        assert!(str_no.contains("Non-causal"));
        assert!(!str_no.contains("Zero-divergence"));

        config.causal = true;
        let ptx_causal = synthesize_flash_attention_ptx(&config);
        let str_c = std::str::from_utf8(&ptx_causal[..ptx_causal.len()-1]).unwrap();
        assert!(str_c.contains("Zero-divergence"));
    }

    #[test]
    fn test_ptx_paged_variant() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: true,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("block_table_ptr"));
        assert!(ptx_str.contains("k_pool_ptr"));
        assert!(ptx_str.contains("v_pool_ptr"));
    }

    #[test]
    fn test_ptx_rope_variant() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: true,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("cos_ptr"));
        assert!(ptx_str.contains("sin_ptr"));
        assert!(ptx_str.contains("half_split"));
    }

    #[test]
    fn test_ptx_gqa_variant() {
        let config = FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 4,
        };
        let ptx = synthesize_flash_attention_ptx(&config);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("kv_head = q_head / 4"));
    }

    #[test]
    fn test_rope_cache_write_ptx() {
        let ptx = synthesize_rope_cache_write_ptx(128, RopeStyle::HalfSplit);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("nsl_rope_cache_write"));
        assert!(ptx_str.contains("seq_ids_ptr"));
        assert!(ptx_str.contains("seq_lens_ptr"));
        assert!(ptx_str.contains("half_split"));
    }

    #[test]
    fn test_rope_cache_write_adjacent() {
        let ptx = synthesize_rope_cache_write_ptx(128, RopeStyle::Adjacent);
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap();
        assert!(ptx_str.contains("adjacent"));
    }
}
```

- [ ] **Step 5: Run unit tests**

Run: `cargo test -p nsl-codegen -- flash_attention`
Expected: All 10 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(m27): FlashAttention-2 PTX template synthesis with 5 kernel variants"
```

---

### Task 6: Fill In PTX Register-Level Instructions

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention.rs`

**Context:** Task 5 created the structural skeleton with commented pseudocode in each `emit_*` helper. This task fills in the actual PTX instructions. The implementation follows the spec Section 4.1 algorithm exactly. Each helper function is independently verifiable.

This task is intentionally large because the PTX instructions are tightly interdependent within each helper. Breaking it into sub-helpers would create artificial boundaries in the middle of register allocation chains.

- [ ] **Step 1: Implement `emit_register_declarations`**

Declare all registers needed for the kernel:
- Thread indexing: `%tid_x`, `%bid_x`, `%bid_y`, block dims
- Address computation: `%rd0`–`%rd63` (u64 for pointers/offsets)
- Computation: `%f0`–`%f127` (f32 for accumulation)
- Predicates: `%p0`–`%p15`
- Integer work: `%r0`–`%r31` (u32)
- Named accumulators: `%row_max`, `%row_sum`, `%correction`, `%new_max`, `%old_max`, `%scale`

- [ ] **Step 2: Implement `emit_param_loads`**

Load all `.param` values into registers using `ld.param.u64` / `ld.param.f32`. Register allocation must be deterministic (based on config flags).

- [ ] **Step 3: Implement `emit_index_computation`**

```ptx
mov.u32 %tid_x, %tid.x;
mov.u32 %bid_x, %ctaid.x;
mov.u32 %bid_y, %ctaid.y;

// q_start = blockIdx.x * block_q
cvt.u64.u32 %rd_qstart, %bid_x;
mul.lo.u64 %rd_qstart, %rd_qstart, {block_q};

// batch_head routing
cvt.u64.u32 %rd_bh, %bid_y;
rem.u64 %rd_head, %rd_bh, %rd5;  // head_idx = bid_y % heads
div.u64 %rd_batch, %rd_bh, %rd5; // batch_idx = bid_y / heads
```

If GQA: `div.u64 %rd_kvhead, %rd_head, {gqa_group_size};`

- [ ] **Step 4: Implement `emit_q_tile_load`**

Cooperative Q tile load into shared memory. Each thread loads `block_q * head_dim / blockDim.x` elements. If `rope_q` is set, apply RoPE rotation in registers before writing to SRAM:

```ptx
// Load Q element pair from global → registers
ld.global.f32 %q_a, [q_addr];
ld.global.f32 %q_b, [q_addr + stride_bytes];
// Load cos/sin from global → registers
ld.global.f32 %cos_val, [cos_addr];
ld.global.f32 %sin_val, [sin_addr];
// Rotate in registers
mul.f32 %q_rot_a, %q_a, %cos_val;
mul.f32 %f_tmp, %q_b, %sin_val;
sub.f32 %q_rot_a, %q_rot_a, %f_tmp;   // q_rot_a = q_a*cos - q_b*sin
mul.f32 %q_rot_b, %q_a, %sin_val;
mul.f32 %f_tmp, %q_b, %cos_val;
add.f32 %q_rot_b, %q_rot_b, %f_tmp;   // q_rot_b = q_a*sin + q_b*cos
// Write to SRAM
st.shared.f32 [shmem + offset_a], %q_rot_a;
st.shared.f32 [shmem + offset_b], %q_rot_b;
```

- [ ] **Step 5: Implement `emit_accumulator_init`**

Each thread computes a subset of the output tile. With 128 threads (4 warps) and `block_q` Q rows, each thread handles `block_q / num_warps` rows, and within each row, accumulates across `head_dim` output elements. The register count per thread is `(block_q / 4) * head_dim / 32`. For block_q=64, head_dim=128: each thread accumulates 64 f32 O_acc registers.

```ptx
mov.f32 %row_max, 0xFF800000;  // -inf (IEEE 754)
mov.f32 %row_sum, 0x00000000;  // 0.0
// Zero O_acc registers — dynamically emitted based on config
// num_oacc = (block_q * head_dim) / blockDim.x
// For block_q=64, head_dim=128, blockDim.x=128: num_oacc = 64
mov.f32 %f_oacc_0, 0x00000000;
mov.f32 %f_oacc_1, 0x00000000;
// ... emit num_oacc mov.f32 instructions via Rust loop
// Compute k_max for loop bound
```

The emit function uses a Rust `for i in 0..num_oacc` loop to generate all the `mov.f32` instructions. No manual unrolling needed — the PTX synthesizer generates the exact count.

- [ ] **Step 6: Implement `emit_kv_tile_loop` — the core algorithm**

This is the most critical function. Implements the 4-phase inner loop:

**Phase 1 — K tile load:**
- If `paged`: block table indirection per physical block
  ```ptx
  // logical_idx = (k_start / block_size) + b
  // physical_id = block_table[seq_id * max_blocks + logical_idx]
  ld.global.u32 %r_phys, [block_table + logical_idx * 4];
  // src = k_pool + physical_id * block_stride + head * block_size * head_dim * 4
  ```
- If not paged: contiguous load `ld.global.f32` from K base pointer
- `bar.sync 0;  // FENCE 1`

**Phase 2 — S = Q @ K^T, apply scale:**

Tile matrix multiply: each thread computes a subset of S[block_q][block_kv]. The inner loop iterates over `head_dim` with sequential dot-product accumulation:

```ptx
// Thread computes S[q_row][k_col] for its assigned (q_row, k_col) pairs
// q_row: assigned rows = threadIdx.x / (blockDim.x / block_q)
// k_col: iterated in inner loop
mov.f32 %f_s, 0x00000000;  // accumulator for one S element
LOOP_HD:
    ld.shared.f32 %f_q, [shmem_Q + q_row * head_dim + d];  // Q from SRAM
    ld.shared.f32 %f_k, [shmem_K + k_col * head_dim + d];  // K from SRAM
    fma.rn.f32 %f_s, %f_q, %f_k, %f_s;                     // dot product
    add.u32 %d, %d, 1;
    setp.lt.u32 %p_hd, %d, head_dim;
    @%p_hd bra LOOP_HD;

// Apply scale
mul.f32 %f_s, %f_s, %scale;
```

Causal masking on diagonal tile:
```ptx
// setp: is this element in the upper triangle?
setp.gt.u64 %p_mask, %k_col_abs, %q_row_abs;  // k_start+j > q_start+i
@%p_mask mov.f32 %f_s, 0xFF800000;              // -inf
```

`bar.sync 0;  // FENCE 2`

**Phase 3 — Online softmax (S→P in registers):**

Warp-level butterfly reduction — exactly 5 steps for warp_size=32. Each step halves the reduction distance. The `shfl.sync.bfly` instruction exchanges values between lanes at the given offset:

```ptx
// Full butterfly reduction for row_max across warp (5 steps)
shfl.sync.bfly.b32 %shfl_tmp, %f_local_max, 16, 31, 0xFFFFFFFF;
max.f32 %f_local_max, %f_local_max, %shfl_tmp;
shfl.sync.bfly.b32 %shfl_tmp, %f_local_max, 8, 31, 0xFFFFFFFF;
max.f32 %f_local_max, %f_local_max, %shfl_tmp;
shfl.sync.bfly.b32 %shfl_tmp, %f_local_max, 4, 31, 0xFFFFFFFF;
max.f32 %f_local_max, %f_local_max, %shfl_tmp;
shfl.sync.bfly.b32 %shfl_tmp, %f_local_max, 2, 31, 0xFFFFFFFF;
max.f32 %f_local_max, %f_local_max, %shfl_tmp;
shfl.sync.bfly.b32 %shfl_tmp, %f_local_max, 1, 31, 0xFFFFFFFF;
max.f32 %f_local_max, %f_local_max, %shfl_tmp;
// %f_local_max now holds row max across all 32 lanes
```

Same 5-step pattern for `row_sum` reduction (replace `max.f32` with `add.f32`).

Correction and P computation:
```ptx
// exp() via ex2.approx: exp(x) = 2^(x * log2(e))
mov.f32 %old_max, %row_max;
mov.f32 %row_max, %f_local_max;       // new_max
sub.f32 %f_tmp, %old_max, %row_max;   // old_max - new_max (<= 0)
mul.f32 %f_tmp, %f_tmp, %log2e;       // * log2(e)
ex2.approx.f32 %correction, %f_tmp;   // exp(old_max - new_max), always <= 1.0

// Rescale running accumulators
mul.f32 %row_sum, %row_sum, %correction;
// Rescale each O_acc register (emitted via Rust loop)
mul.f32 %f_oacc_0, %f_oacc_0, %correction;
// ... (one per O_acc register)

// Compute P = exp(S - new_max) — overwrites S registers in-place
sub.f32 %f_p, %f_s, %row_max;
mul.f32 %f_p, %f_p, %log2e;
ex2.approx.f32 %f_p, %f_p;
// Add to row_sum
add.f32 %row_sum, %row_sum, %f_p;
```

**Phase 4 — V tile load (reuses K SRAM):**

Same cooperative load pattern as Phase 1, but reads from V pointer (or V pool if paged). The SRAM region previously holding K is overwritten with V — this is why FENCE 2 is critical.

`bar.sync 0;  // FENCE 3`

**Phase 5 — O_acc += P @ V_tile:**

P is in registers (from Phase 3), V is in SRAM. Same inner-loop structure as Phase 2 but accumulates into O_acc:

```ptx
// For each V column assigned to this thread:
fma.rn.f32 %f_oacc_j, %f_p, %f_v, %f_oacc_j;  // O_acc[j] += P[k] * V[k][j]
```

The inner loop iterates over `block_kv` (the P/V shared dimension). Each thread accumulates `head_dim / threads_per_kv_col` output elements.

- [ ] **Step 6b: Implement loop counter and termination**

The K/V tile loop needs explicit loop control. Add to `emit_kv_tile_loop`:

```ptx
// Initialize loop counter
mov.u64 %k_start, 0;

// Compute loop bound
// k_max = causal ? min(seq_len, q_start + block_q) : seq_len
```
If causal:
```ptx
add.u64 %k_max, %rd_qstart, {block_q};
min.u64 %k_max, %k_max, %rd6;  // %rd6 = seq_len
```
If not causal:
```ptx
mov.u64 %k_max, %rd6;  // k_max = seq_len
```

At end of loop body (after Phase 5):
```ptx
// Increment k_start by block_kv
add.u64 %k_start, %k_start, {block_kv};
setp.lt.u64 %p_loop, %k_start, %k_max;
@%p_loop bra LOOP_KV_START;
LOOP_KV_END:
```

- [ ] **Step 7: Implement `emit_finalize`**

```ptx
// O = O_acc / row_sum
rcp.approx.f32 %f_inv_sum, %row_sum;   // 1/row_sum
mul.f32 %f_oacc_0, %f_oacc_0, %f_inv_sum;
// ... for each O_acc register
```

- [ ] **Step 8: Implement `emit_output_store`**

```ptx
// Convert f32 → f16 for output (uses .b16 registers declared in emit_register_declarations)
cvt.rn.f16.f32 %h0, %f_oacc_0;
st.global.b16 [out_addr], %h0;
// Emitted via Rust loop — one cvt+store per O_acc register
cvt.rn.f16.f32 %h1, %f_oacc_1;
st.global.b16 [out_addr + 2], %h1;  // +2 bytes per f16
// ...
```

- [ ] **Step 9: Run unit tests**

Run: `cargo test -p nsl-codegen -- flash_attention`
Expected: All tests pass. PTX output now contains actual instructions.

- [ ] **Step 10: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention.rs
git commit -m "feat(m27): fill in register-level PTX instructions for FlashAttention-2"
```

---

## Chunk 3: Codegen Integration, Semantic Validation, and E2E Tests

This chunk wires the PTX templates into the compiler pipeline: decorator validation, `scaled_dot_product_attention` lowering, `@autotune` integration, and E2E tests.

### Task 7: Semantic Checker — Decorator Validation

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs:377-420`
- Modify: `crates/nsl-semantic/src/builtins.rs`

**Context:** The semantic checker validates decorator placement and arguments. M27 adds three new decorators (`@flash_attention`, `@rope`, `@gqa`) and relaxes `@autotune` to accept `@flash_attention`-decorated functions (not just `kernel` blocks).

- [ ] **Step 1: Register `scaled_dot_product_attention` in semantic builtins**

In `crates/nsl-semantic/src/builtins.rs`, add after the `causal_mask` definition (~line 358):

```rust
    // M27: scaled_dot_product_attention(Q, K, V, scale, causal=true) -> tensor
    def(
        "scaled_dot_product_attention",
        Type::Function {
            params: vec![
                tensor_ret.clone(),  // Q
                tensor_ret.clone(),  // K
                tensor_ret.clone(),  // V
                Type::Float,         // scale
            ],
            ret: Box::new(tensor_ret.clone()),
        },
    );
```

- [ ] **Step 2: Add `@flash_attention` decorator validation**

In `crates/nsl-semantic/src/checker.rs`, after the `@autotune` block (~line 420), add:

```rust
if dname == "flash_attention" {
    // Valid target: fn inside model only
    match &stmt.kind {
        StmtKind::FnDef(_) => {
            // Validate args: causal: bool (optional, default true)
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(ref name_sym) = arg.name {
                        let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                        if aname != "causal" {
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "@flash_attention: unknown argument '{}' (expected 'causal')", aname
                                ))
                                .with_label(arg.span, "unknown argument")
                            );
                        }
                    }
                }
            }
        }
        _ => {
            self.diagnostics.push(
                Diagnostic::error("@flash_attention can only be applied to fn declarations inside model blocks")
                    .with_label(deco.span, "invalid @flash_attention target")
            );
        }
    }
}
```

- [ ] **Step 3: Add `@rope` decorator validation**

After the `@flash_attention` block:

```rust
if dname == "rope" {
    // Valid only when @flash_attention is also present
    let has_flash = decorators.iter().any(|d| {
        d.name.len() == 1
            && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
    });
    if !has_flash {
        self.diagnostics.push(
            Diagnostic::error("@rope requires @flash_attention on the same function")
                .with_label(deco.span, "@rope without @flash_attention")
        );
    }
    // Validate args: style: str (optional, default "half_split")
    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                if aname == "style" {
                    if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                        let val = self.interner.resolve(s.0).unwrap_or("");
                        if val != "half_split" && val != "adjacent" {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "@rope style must be \"half_split\" or \"adjacent\""
                                )
                                .with_label(arg.span, "invalid style")
                            );
                        }
                    }
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "@rope: unknown argument '{}' (expected 'style')", aname
                        ))
                        .with_label(arg.span, "unknown argument")
                    );
                }
            }
        }
    }
}
```

- [ ] **Step 4: Add `@gqa` decorator validation**

```rust
if dname == "gqa" {
    let has_flash = decorators.iter().any(|d| {
        d.name.len() == 1
            && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
    });
    if !has_flash {
        self.diagnostics.push(
            Diagnostic::error("@gqa requires @flash_attention on the same function")
                .with_label(deco.span, "@gqa without @flash_attention")
        );
    }
    // Validate args: groups: int (required, > 0)
    let mut has_groups = false;
    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                if aname == "groups" {
                    has_groups = true;
                    if let ExprKind::IntLiteral(n) = arg.value.kind {
                        if n <= 0 {
                            self.diagnostics.push(
                                Diagnostic::error("@gqa groups must be a positive integer")
                                    .with_label(arg.span, "must be > 0")
                            );
                        }
                    } else {
                        self.diagnostics.push(
                            Diagnostic::error("@gqa groups must be an integer literal")
                                .with_label(arg.span, "expected integer")
                        );
                    }
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "@gqa: unknown argument '{}' (expected 'groups')", aname
                        ))
                        .with_label(arg.span, "unknown argument")
                    );
                }
            }
        }
    }
    if !has_groups {
        self.diagnostics.push(
            Diagnostic::error("@gqa requires a 'groups' argument (e.g., @gqa(groups=4))")
                .with_label(deco.span, "missing 'groups'")
        );
    }
}
```

- [ ] **Step 5: Relax `@autotune` to accept `@flash_attention`-decorated functions**

In the `@autotune` validation block (~line 377-420), change the error case from:

```rust
_ => {
    self.diagnostics.push(
        Diagnostic::error("@autotune can only be applied to kernel blocks")
            .with_label(deco.span, "invalid @autotune target")
    );
}
```

to check for `@flash_attention` on `FnDef`:

```rust
StmtKind::FnDef(_) => {
    // Valid if also decorated with @flash_attention
    let has_flash = decorators.iter().any(|d| {
        d.name.len() == 1
            && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
    });
    if !has_flash {
        self.diagnostics.push(
            Diagnostic::error("@autotune on fn requires @flash_attention decorator")
                .with_label(deco.span, "invalid @autotune target")
        );
    }
    // Same list-of-ints validation as kernel blocks
    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref _name_sym) = arg.name {
                match &arg.value.kind {
                    ExprKind::ListLiteral(items) => {
                        for item in items {
                            if !matches!(item.kind, ExprKind::IntLiteral(_)) {
                                self.diagnostics.push(
                                    Diagnostic::error("@autotune parameter values must be integer literals")
                                        .with_label(item.span, "expected integer")
                                );
                            }
                        }
                    }
                    _ => {
                        self.diagnostics.push(
                            Diagnostic::error("@autotune parameters must be lists of integers")
                                .with_label(arg.span, "expected list")
                        );
                    }
                }
            }
        }
    }
}
_ => {
    self.diagnostics.push(
        Diagnostic::error("@autotune can only be applied to kernel blocks or @flash_attention functions")
            .with_label(deco.span, "invalid @autotune target")
    );
}
```

- [ ] **Step 6: Verify compilation**

Run: `cargo build -p nsl-semantic -p nsl-codegen`
Expected: Compiles.

- [ ] **Step 7: Run existing tests**

Run: `cargo test -p nsl-cli --test e2e`
Expected: All existing E2E tests pass (no regressions).

- [ ] **Step 8: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs crates/nsl-semantic/src/builtins.rs
git commit -m "feat(m27): semantic validation for @flash_attention, @rope, @gqa decorators"
```

---

### Task 8: Codegen — `scaled_dot_product_attention` Lowering

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

**Context:** When the compiler sees `scaled_dot_product_attention(Q, K, V, scale, causal=true)`:
- **With `@flash_attention`:** Lower to PTX dispatch via `nsl_flash_attention()` FFI.
- **Without `@flash_attention`:** Lower to naive `softmax(apply_causal_mask((Q @ K.T) * scale)) @ V`.

The naive path reuses existing builtins: `nsl_tensor_matmul`, `nsl_tensor_transpose`, `nsl_tensor_mul_scalar`, `nsl_tensor_causal_mask`, `nsl_tensor_add` (for mask application), `nsl_tensor_softmax`, `nsl_tensor_matmul` (final).

- [ ] **Step 1: Add `scaled_dot_product_attention` lowering in expr.rs**

In `crates/nsl-codegen/src/expr.rs`, after the `causal_mask` handler (~line 1091), add:

```rust
// M27: scaled_dot_product_attention(Q, K, V, scale, causal=...)
if func_name == "scaled_dot_product_attention" && !self.functions.contains_key(&func_name) {
    if args.len() < 4 {
        return Err(CodegenError::new(
            "scaled_dot_product_attention() requires at least 4 arguments (Q, K, V, scale)"
        ));
    }

    let q_val = self.compile_expr(builder, state, &args[0].value)?;
    let k_val = self.compile_expr(builder, state, &args[1].value)?;
    let v_val = self.compile_expr(builder, state, &args[2].value)?;
    let scale_val = self.compile_expr(builder, state, &args[3].value)?;

    // Check for causal=true named arg
    let mut causal = false;
    for arg in &args[4..] {
        if let Some(name_sym) = arg.name {
            let name = self.resolve_sym(name_sym).to_string();
            if name == "causal" {
                if let ExprKind::BoolLiteral(b) = arg.value.kind {
                    causal = b;
                }
            }
        }
    }

    // Check if the enclosing function has @flash_attention decorator
    // (tracked in self.flash_attention_context set during compile_user_functions)
    if self.flash_attention_context.is_some() {
        // Flash path: emit call to nsl_flash_attention with PTX from .rodata
        return self.compile_flash_attention_call(builder, state, q_val, k_val, v_val, scale_val);
    }

    // Naive path: softmax(apply_causal_mask((Q @ K.T) * scale)) @ V
    // Step 1: K_T = transpose(K)
    let k_t = self.compile_call_by_name(builder, "nsl_tensor_transpose", &[k_val])?;
    // Step 2: scores = Q @ K_T
    let scores = self.compile_call_by_name(builder, "nsl_tensor_matmul", &[q_val, k_t])?;
    // Step 3: scores = scores * scale (broadcast scalar mul)
    let scaled = self.compile_call_by_name(builder, "nsl_tensor_mul_scalar", &[scores, scale_val])?;

    // Step 4: Apply causal mask if causal=true
    let masked = if causal {
        // Get seq_len from Q shape (dim 0 or -2)
        let seq_len = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[
            q_val,
            builder.ins().iconst(cl_types::I64, -2i64),
        ])?;
        let mask = self.compile_call_by_name(builder, "nsl_tensor_causal_mask", &[seq_len])?;
        // scores + mask (mask has 0 for valid, -1e9 for masked positions)
        self.compile_call_by_name(builder, "nsl_tensor_add", &[scaled, mask])?
    } else {
        scaled
    };

    // Step 5: attn_weights = softmax(masked, dim=-1)
    let dim_neg1 = builder.ins().iconst(cl_types::I64, -1i64);
    let attn_weights = self.compile_call_by_name(builder, "nsl_tensor_softmax", &[masked, dim_neg1])?;

    // Step 6: output = attn_weights @ V
    return self.compile_call_by_name(builder, "nsl_tensor_matmul", &[attn_weights, v_val]);
}
```

- [ ] **Step 2: Add `flash_attention_context` field to Compiler**

In `crates/nsl-codegen/src/compiler.rs`, add to the Compiler struct fields:

```rust
/// When compiling a function with @flash_attention, this holds the variant config
/// and PTX .rodata references. None when compiling non-flash functions.
flash_attention_context: Option<FlashAttentionCompileContext>,
```

```rust
pub struct FlashAttentionCompileContext {
    pub ptx_data_id: DataId,
    pub name_data_id: DataId,
    pub config: crate::flash_attention::FlashAttentionConfig,
}
```

Initialize to `None` in the constructor.

- [ ] **Step 3: Add `compile_flash_attention_call` method to expr.rs**

```rust
fn compile_flash_attention_call(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FunctionState,
    q_val: Value,
    k_val: Value,
    v_val: Value,
    scale_val: Value,
) -> Result<Value, CodegenError> {
    let ctx = self.flash_attention_context.as_ref()
        .ok_or_else(|| CodegenError::new("flash_attention_context not set"))?;

    // Allocate output tensor (same shape as Q)
    let out_val = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[q_val])?;

    // Get PTX and name pointers from .rodata
    let ptx_gv = self.module.declare_data_in_func(ctx.ptx_data_id, builder.func);
    let ptx_ptr = builder.ins().symbol_value(cl_types::I64, ptx_gv);
    let name_gv = self.module.declare_data_in_func(ctx.name_data_id, builder.func);
    let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);

    // Constants from config
    let block_q = builder.ins().iconst(cl_types::I64, ctx.config.block_q);
    let block_kv = builder.ins().iconst(cl_types::I64, ctx.config.block_kv);
    let shmem = builder.ins().iconst(cl_types::I64,
        crate::flash_attention::shared_mem_bytes(&ctx.config) as i64);

    // Null pointers for unused variant params
    let null = builder.ins().iconst(cl_types::I64, 0);

    // Build call: nsl_flash_attention(q, k, v, out, scale, batch, heads, seq_len, head_dim,
    //     block_table, k_pool, v_pool, block_size, cos, sin, seq_ids, seq_lens,
    //     shared_mem_bytes, ptx_ptr, name_ptr, block_q, block_kv)
    // For now, batch/heads/seq_len/head_dim are extracted from tensor shapes at runtime
    let batch = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, builder.ins().iconst(cl_types::I64, 0)])?;
    let heads = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, builder.ins().iconst(cl_types::I64, 1)])?;
    let seq_len = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, builder.ins().iconst(cl_types::I64, 2)])?;
    let head_dim = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, builder.ins().iconst(cl_types::I64, 3)])?;

    // Paged params — pass null for now. The paged variant's runtime launch
    // wrapper is fully implemented; what is stubbed is the codegen wiring that
    // extracts block_table/k_pool/v_pool from the model's @paged_kv member and
    // passes them through the Cranelift IR. This will be wired when the first
    // paged E2E test runs on actual GPU hardware (requires CUDA feature).
    let (block_table, k_pool, v_pool, block_size_val) = (null, null, null, null);

    // RoPE params — pass null for now. Same reasoning: the rope_cache_write
    // PTX and launch wrapper are implemented, but the codegen wiring that
    // extracts cos/sin frequency tables from model state is deferred until
    // GPU E2E testing.
    let (cos_p, sin_p) = (null, null);

    self.compile_call_by_name(builder, "nsl_flash_attention", &[
        q_val, k_val, v_val, out_val, scale_val,
        batch, heads, seq_len, head_dim,
        block_table, k_pool, v_pool, block_size_val,
        cos_p, sin_p,
        null, null,  // seq_ids, seq_lens (M29-ready)
        shmem,
        ptx_ptr, name_ptr,
        block_q, block_kv,
    ])
}
```

- [ ] **Step 4: Wire flash_attention into compiler pipeline**

In `crates/nsl-codegen/src/compiler.rs`, add a `compile_flash_attention_kernels()` method that:

1. Walks function definitions for `@flash_attention` decorator
2. Extracts variant flags from other decorators (`@rope`, `@gqa`, `@paged_kv`)
3. Extracts `@autotune` params if present (calls `extract_autotune_params`)
4. Calls `synthesize_flash_attention_ptx()` to generate PTX
5. Embeds PTX in `.rodata` (same pattern as `compile_single_kernel`)
6. Stores `FlashAttentionCompileContext` for the function name

Add this call to the main compilation pipeline (~line 2026), after `compile_kernels()`:

```rust
self.compile_flash_attention_kernels(stmts)?;
```

- [ ] **Step 5: Verify compilation**

Run: `cargo build -p nsl-codegen`
Expected: Compiles.

- [ ] **Step 6: Run existing tests**

Run: `cargo test -p nsl-cli --test e2e`
Expected: All existing E2E tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/expr.rs crates/nsl-codegen/src/compiler.rs
git commit -m "feat(m27): scaled_dot_product_attention lowering with naive and flash paths"
```

---

### Task 9: @autotune Integration for FlashAttention

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

**Context:** When `@autotune(block_q=[32, 64, 128], block_kv=[32, 64])` is present on a `@flash_attention` function, the compiler generates `|block_q| × |block_kv|` PTX variants (e.g., 3 × 2 = 6). The Cartesian product comes from the existing `autotune::cartesian_product()`. Each variant's shared memory is validated against the 48KB sm_52 limit. `select_middle_values()` picks the fallback. Cache key includes variant flags.

- [ ] **Step 1: Add shared memory validation to `compile_flash_attention_kernels`**

In the autotune path of `compile_flash_attention_kernels()`:

```rust
// Validate shared memory for each variant
for variant in &variants {
    let mut test_config = base_config.clone();
    for (name, val) in variant {
        match name.as_str() {
            "block_q" => test_config.block_q = *val,
            "block_kv" => test_config.block_kv = *val,
            _ => {}
        }
    }
    let shmem = crate::flash_attention::shared_mem_bytes(&test_config);
    if shmem > 49152 { // 48 KB = 49152 bytes (sm_52 default)
        return Err(CodegenError::new(format!(
            "@autotune variant (block_q={}, block_kv={}) requires {}KB shared memory, exceeds 48KB limit for sm_52",
            test_config.block_q, test_config.block_kv, shmem / 1024
        )));
    }
}
```

- [ ] **Step 2: Add `@paged_kv` block_kv alignment validation**

When both `@paged_kv` and `@autotune` are present:

```rust
if config.paged {
    if let Some(paged_config) = self.paged_kv_configs.get(&model_name) {
        let block_size = paged_config.1; // block_size is the 2nd element
        for variant in &variants {
            for (name, val) in variant {
                if name == "block_kv" && val % block_size != 0 {
                    return Err(CodegenError::new(format!(
                        "@autotune block_kv={} is not a multiple of @paged_kv block_size={}",
                        val, block_size
                    )));
                }
            }
        }
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p nsl-codegen`
Expected: Compiles.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs
git commit -m "feat(m27): @autotune integration with shared memory and block_kv validation"
```

---

### Task 10: E2E Tests

**Files:**
- Create: `examples/m27_flash_attention.nsl`
- Create: `examples/m27_paged_attention.nsl`
- Create: `examples/m27_rope_gqa.nsl`
- Create: `tests/expected/m27_flash_attention.txt`
- Create: `tests/expected/m27_paged_attention.txt`
- Create: `tests/expected/m27_rope_gqa.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

**Context:** E2E tests validate the naive fallback path (no CUDA required) and compile-time validation. The pattern matches existing tests: `.nsl` source in `examples/`, expected output in `tests/expected/`, test function in `e2e.rs`.

- [ ] **Step 1: Create `examples/m27_flash_attention.nsl`**

```nsl
# M27: FlashAttention-2 — basic test
# Without @flash_attention decorator, uses naive softmax path

from nsl.nn import scaled_dot_product_attention

let Q = Tensor.randn([1, 4, 8, 16])  # [batch, heads, seq_len, head_dim]
let K = Tensor.randn([1, 4, 8, 16])
let V = Tensor.randn([1, 4, 8, 16])

let scale = 1.0 / sqrt(16.0)

# Naive path: softmax((Q @ K.T) * scale) @ V
let out = scaled_dot_product_attention(Q, K, V, scale)
print("Shape:", out.shape)
print("FlashAttention naive path OK")

# Causal variant
let out_causal = scaled_dot_product_attention(Q, K, V, scale, causal=true)
print("Causal shape:", out_causal.shape)
print("FlashAttention causal naive path OK")
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m27_flash_attention.txt`:
```
Shape: [1, 4, 8, 16]
FlashAttention naive path OK
Causal shape: [1, 4, 8, 16]
FlashAttention causal naive path OK
```

- [ ] **Step 3: Create `examples/m27_paged_attention.nsl`**

```nsl
# M27: Paged FlashAttention — decorator validation test
# Tests that @flash_attention and @paged_kv decorators parse correctly
# (actual GPU execution gated behind cuda feature)

from nsl.nn import scaled_dot_product_attention

model TestTransformer:
    @paged_kv(num_blocks=32, block_size=16, num_heads=4, head_dim=16, num_layers=1)
    k_cache: str

    @flash_attention(causal=true)
    @paged_kv(block_size=16)
    fn attention(self, Q, K, V) -> Tensor:
        let scale = 1.0 / sqrt(float(Q.shape[-1]))
        return scaled_dot_product_attention(Q, K, V, scale, causal=true)

let m = TestTransformer()
print("Paged FlashAttention model created OK")
```

- [ ] **Step 4: Create expected output**

Create `tests/expected/m27_paged_attention.txt`:
```
Paged FlashAttention model created OK
```

- [ ] **Step 5: Create `examples/m27_rope_gqa.nsl`**

```nsl
# M27: RoPE + GQA variant — decorator validation test

from nsl.nn import scaled_dot_product_attention

model GQATransformer:
    @flash_attention(causal=true)
    @gqa(groups=4)
    @rope(style="half_split")
    fn attention(self, Q, K, V) -> Tensor:
        let scale = 1.0 / sqrt(float(Q.shape[-1]))
        return scaled_dot_product_attention(Q, K, V, scale, causal=true)

let m = GQATransformer()
print("RoPE + GQA model created OK")
```

- [ ] **Step 6: Create expected output**

Create `tests/expected/m27_rope_gqa.txt`:
```
RoPE + GQA model created OK
```

- [ ] **Step 7: Add E2E test functions**

In `crates/nsl-cli/tests/e2e.rs`, add after the M26 tests (~line 514):

```rust
// ---------------------------------------------------------------------------
// M27: FlashAttention-2
// ---------------------------------------------------------------------------

#[test]
fn e2e_m27_flash_attention() {
    assert_output_matches("m27_flash_attention");
}

#[test]
fn e2e_m27_paged_attention() {
    assert_output_matches("m27_paged_attention");
}

#[test]
fn e2e_m27_rope_gqa() {
    assert_output_matches("m27_rope_gqa");
}
```

- [ ] **Step 8: Run E2E tests**

Run: `cargo test -p nsl-cli --test e2e -- m27`
Expected: All 3 new tests pass. The basic test validates the naive softmax path. The paged and RoPE/GQA tests validate decorator parsing and model creation.

- [ ] **Step 9: Run full test suite**

Run: `cargo test -p nsl-cli --test e2e`
Expected: All tests pass (no regressions).

- [ ] **Step 10: Run clippy**

Run: `cargo clippy -p nsl-codegen -p nsl-runtime -p nsl-semantic -- -D warnings`
Expected: No warnings.

- [ ] **Step 11: Commit**

```bash
git add examples/m27_*.nsl tests/expected/m27_*.txt crates/nsl-cli/tests/e2e.rs
git commit -m "feat(m27): E2E tests for FlashAttention naive path and decorator validation"
```

---

### Task 11: Final Integration and Cleanup

**Files:**
- All M27 files (verification only)

**Context:** Final verification that everything is wired up, all tests pass, no clippy warnings, and the deliverables checklist from the spec is satisfied.

- [ ] **Step 1: Run full build**

Run: `cargo build`
Expected: Clean build.

- [ ] **Step 2: Run full test suite**

Run: `cargo test`
Expected: All tests pass (unit + E2E).

- [ ] **Step 3: Run clippy on all crates**

Run: `cargo clippy -- -D warnings`
Expected: No warnings.

- [ ] **Step 4: Verify deliverables checklist**

Cross-reference against spec Section 11:
1. `crates/nsl-codegen/src/flash_attention.rs` — PTX template generator
2. `crates/nsl-runtime/src/flash_attention.rs` — launch wrappers
3. `kernel_launch()` updated with `shared_mem_bytes`
4. Semantic checker validation for decorators
5. `@autotune` integration
6. `scaled_dot_product_attention` registered in builtins
7. Kernel profiler shows FlashAttention (inherits from `kernel_launch` profiler)
8. `@paged_kv` block_kv alignment check
9. 3 E2E tests + unit tests
10. All existing tests pass

- [ ] **Step 5: Commit final state**

```bash
# Only stage M27-related files — avoid untracked files like roundtrip.safetensors
git add crates/nsl-codegen/src/flash_attention.rs crates/nsl-runtime/src/flash_attention.rs \
  crates/nsl-runtime/src/cuda/mod.rs crates/nsl-runtime/src/lib.rs \
  crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/builtins.rs \
  crates/nsl-codegen/src/expr.rs crates/nsl-codegen/src/compiler.rs \
  crates/nsl-semantic/src/builtins.rs crates/nsl-semantic/src/checker.rs \
  crates/nsl-cli/tests/e2e.rs \
  examples/m27_*.nsl tests/expected/m27_*.txt
git commit -m "feat(m27): FlashAttention-2 milestone complete — all deliverables verified"
```
