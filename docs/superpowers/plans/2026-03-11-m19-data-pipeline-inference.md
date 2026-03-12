# M19: Data Pipeline + Inference Sampling — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load a dataset, train a model, and generate text with sampling — the complete ML workflow.

**Architecture:** Bottom-up: runtime tensor primitives first (topk, multinomial, argmax, cumsum), then data source readers (JSONL, CSV, mmap), then multi-threaded DataLoader with sequence packing, then compiler intrinsics + codegen, then stdlib inference modules, then integration tests. Each layer is testable in isolation via `cargo test` before wiring into the compiler.

**Tech Stack:** Rust (nsl-runtime), Cranelift (nsl-codegen), NSL stdlib. New crate deps: `serde_json`, `memmap2`, `rand`.

---

## File Structure

### New Files
| File | Responsibility |
|------|---------------|
| `crates/nsl-runtime/src/sampling.rs` | topk, multinomial, argmax, cumsum, lt_scalar, deterministic RNG |
| `crates/nsl-runtime/src/data_source.rs` | JSONL, CSV, mmap file readers |
| `crates/nsl-runtime/src/packing.rs` | Continuous stream sequence packing + block-diagonal masks |
| `crates/nsl-runtime/src/dataloader.rs` | Multi-threaded DataLoader with ring buffer + reorder buffer |
| `stdlib/nsl/data/loader.nsl` | Documentation module for data intrinsics |
| `stdlib/nsl/inference/sampling.nsl` | top-k, top-p, greedy sampling functions |
| `stdlib/nsl/inference/generate.nsl` | Autoregressive text generation with no_grad |
| `tests/m19_topk_test.nsl` | topk + multinomial + argmax integration test |
| `tests/m19_data_test.nsl` | JSONL/CSV/mmap data loading test |
| `tests/m19_dataloader_test.nsl` | DataLoader batching test |
| `tests/m19_generate_test.nsl` | End-to-end inference generation test |
| `examples/m19_data_pipeline.nsl` | Full E2E: load data, train, generate |

### Modified Files
| File | Changes |
|------|---------|
| `crates/nsl-runtime/Cargo.toml` | Add `serde_json`, `memmap2`, `rand` deps |
| `crates/nsl-runtime/src/lib.rs` | Declare new modules |
| `crates/nsl-runtime/src/tensor.rs` | Add `owns_data: u8` field to NslTensor, update free/create paths |
| `crates/nsl-runtime/src/dict.rs` | Add `nsl_dict_free` function |
| `crates/nsl-codegen/src/builtins.rs` | Register M19 runtime functions in RUNTIME_FUNCTIONS array |
| `crates/nsl-semantic/src/builtins.rs` | Register M19 semantic builtins |
| `crates/nsl-codegen/src/expr.rs` | Add intrinsic handlers + tensor method dispatch |
| `crates/nsl-codegen/src/stmt.rs` | Add DataLoader for-loop codegen + scope teardown |
| `README.md` | Update M19 status to Complete |

---

## Chunk 1: Foundation — Tensor Struct + Sampling Primitives

### Task 1: Add `owns_data` Flag to NslTensor

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`

The NslTensor struct (line 24) currently has 8 fields. We must add `owns_data: u8` so that memory-mapped tensors don't get freed by the heap allocator.

- [ ] **Step 1: Add `owns_data` field to NslTensor struct**

In `crates/nsl-runtime/src/tensor.rs`, find the struct at line 24:

```rust
#[repr(C)]
pub struct NslTensor {
    pub(crate) data: *mut c_void,
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
    pub(crate) ndim: i64,
    pub(crate) len: i64,
    pub(crate) refcount: i64,
    pub(crate) device: u8,
    pub(crate) dtype: u8,
}
```

Add `owns_data: u8` after `dtype`:

```rust
#[repr(C)]
pub struct NslTensor {
    pub(crate) data: *mut c_void,
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
    pub(crate) ndim: i64,
    pub(crate) len: i64,
    pub(crate) refcount: i64,
    pub(crate) device: u8,
    pub(crate) dtype: u8,
    pub(crate) owns_data: u8,   // 1 = heap-owned (default), 0 = borrowed (mmap)
}
```

- [ ] **Step 2: Update all NslTensor construction sites to set `owns_data: 1`**

Search for `Box::new(NslTensor {` across `tensor.rs` and all other runtime files. Every construction must add `owns_data: 1`. There are approximately 8-12 sites in `tensor.rs` (e.g., `tensor_from_shape_list` ~line 120, `nsl_tensor_zeros_on` ~line 2720, `nsl_tensor_to_device` ~line 3770) plus sites in `safetensors_io.rs`, `cpu.rs`, and `quantize.rs`.

Use `grep -n "Box::new(NslTensor" crates/nsl-runtime/src/` to find all sites.

At each site, add `owns_data: 1,` after the `dtype` field.

- [ ] **Step 3: Update `nsl_tensor_free` to check `owns_data`**

In `nsl_tensor_free` (line ~2599), wrap the data deallocation in an `owns_data` check:

```rust
if tensor.owns_data != 0 {
    if tensor.device > 0 {
        #[cfg(feature = "cuda")]
        { crate::cuda::free_managed(tensor.data); }
        #[cfg(not(feature = "cuda"))]
        { checked_free(tensor.data as *mut u8, data_size); }
    } else {
        checked_free(tensor.data as *mut u8, data_size);
    }
}
// Shape and strides are always owned — free unconditionally
```

- [ ] **Step 4: Run tests to verify no regressions**

Run: `cargo test -p nsl-runtime`
Expected: All existing tests pass (the new field defaults to 1, so behavior is unchanged)

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs crates/nsl-runtime/src/cpu.rs crates/nsl-runtime/src/safetensors_io.rs
git commit -m "feat(m19): add owns_data flag to NslTensor for mmap safety"
```

---

### Task 2: Add `nsl_dict_free` Function

**Files:**
- Modify: `crates/nsl-runtime/src/dict.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

Currently there is NO `nsl_dict_free` function. The DataLoader loop needs to free batch dicts each iteration to prevent OOM. We also need it for general cleanup.

- [ ] **Step 1: Implement `nsl_dict_free` in dict.rs**

Add at the end of `crates/nsl-runtime/src/dict.rs`:

```rust
/// Free an NslDict and release all contained tensor values.
/// Keys (C strings) are freed. Values that are tensor pointers
/// are freed via nsl_tensor_free (decrement refcount).
#[no_mangle]
pub extern "C" fn nsl_dict_free(dict_ptr: i64) {
    if dict_ptr == 0 {
        return;
    }
    let dict = unsafe { &mut *(dict_ptr as *mut NslDict) };
    for i in 0..dict.num_buckets as usize {
        let mut entry = unsafe { *dict.buckets.add(i) };
        while !entry.is_null() {
            let e = unsafe { &*entry };
            let next = e.next;
            // Free the key string
            if !e.key.is_null() {
                unsafe { crate::memory::checked_free(e.key, std::ffi::CStr::from_ptr(e.key as *const i8).to_bytes_with_nul().len()) };
            }
            // Free the value (assumed to be a tensor pointer)
            crate::tensor::nsl_tensor_free(e.value);
            // Free the entry struct
            unsafe { drop(Box::from_raw(entry)) };
            entry = next;
        }
    }
    // Free bucket array
    let bucket_size = (dict.num_buckets as usize) * std::mem::size_of::<*mut NslDictEntry>();
    unsafe { crate::memory::checked_free(dict.buckets as *mut u8, bucket_size) };
    // Free dict struct
    unsafe { drop(Box::from_raw(dict as *mut NslDict)) };
}
```

- [ ] **Step 2: Register `nsl_dict_free` in codegen builtins**

In `crates/nsl-codegen/src/builtins.rs`, add to the `RUNTIME_FUNCTIONS` array (after the existing dict entries around line 47):

```rust
("nsl_dict_free", &[types::I64], None),
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p nsl-runtime`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/dict.rs crates/nsl-codegen/src/builtins.rs
git commit -m "feat(m19): add nsl_dict_free for DataLoader batch cleanup"
```

---

### Task 3: Add Cargo Dependencies

**Files:**
- Modify: `crates/nsl-runtime/Cargo.toml`

- [ ] **Step 1: Add `serde_json`, `memmap2`, `rand` dependencies**

In `crates/nsl-runtime/Cargo.toml`, add to `[dependencies]`:

```toml
serde_json = "1"
memmap2 = "0.9"
rand = "0.8"
```

These are NOT feature-gated — they're core M19 functionality (unlike M18b interop which was optional).

- [ ] **Step 2: Verify build**

Run: `cargo build -p nsl-runtime`
Expected: Compiles successfully with new deps

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/Cargo.toml
git commit -m "feat(m19): add serde_json, memmap2, rand dependencies"
```

---

### Task 4: Implement Sampling Primitives (topk, multinomial, argmax, cumsum, lt_scalar, RNG)

**Files:**
- Create: `crates/nsl-runtime/src/sampling.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

This is the largest single task. All 6 FFI functions go in one focused module.

- [ ] **Step 1: Create `sampling.rs` with thread-local RNG**

Create `crates/nsl-runtime/src/sampling.rs`:

```rust
//! Sampling primitives for inference: topk, multinomial, argmax, cumsum, lt_scalar.
//! Deterministic RNG via thread-local StdRng.

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::cmp::Reverse;

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand::Rng;

use crate::tensor::NslTensor;
use crate::memory::{checked_alloc, checked_alloc_zeroed};

// ── Deterministic RNG ─────────────────────────────────────────────

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(0));
}

/// Set the global RNG seed for deterministic sampling.
#[no_mangle]
pub extern "C" fn nsl_manual_seed(seed: i64) {
    RNG.with(|r| {
        *r.borrow_mut() = StdRng::seed_from_u64(seed as u64);
    });
}
```

- [ ] **Step 2: Implement `nsl_tensor_topk`**

Add to `sampling.rs`:

```rust
/// Top-k selection: returns Dict{"values": Tensor, "indices": Tensor}
/// Algorithm: min-heap of size k, O(n log k) per row.
#[no_mangle]
pub extern "C" fn nsl_tensor_topk(tensor_ptr: i64, k: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = crate::cpu::get_shape_vec(tensor);
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    let k = k as usize;

    if d >= ndim {
        eprintln!("nsl: topk dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let dim_size = shape[d] as usize;
    if k > dim_size {
        eprintln!("nsl: topk k={} exceeds dimension size {}", k, dim_size);
        std::process::abort();
    }

    // Output shape: same as input but with dim d replaced by k
    let mut out_shape: Vec<i64> = shape.clone();
    out_shape[d] = k as i64;

    let val_ptr = crate::cpu::create_tensor_with_shape_rs(&out_shape);
    let idx_ptr = crate::cpu::create_tensor_with_shape_rs(&out_shape);
    let val_tensor = NslTensor::from_ptr(val_ptr);
    let idx_tensor = NslTensor::from_ptr(idx_ptr);

    let strides = crate::cpu::get_strides_vec(tensor);
    let out_strides = crate::cpu::get_strides_vec(val_tensor);

    // Number of independent slices along dim d
    let num_slices: usize = (tensor.len as usize) / dim_size;

    for slice_idx in 0..num_slices {
        // Compute base offset (skipping dim d)
        let mut remaining = slice_idx;
        let mut base_in: usize = 0;
        let mut base_out: usize = 0;
        let mut oi = 0usize;
        for axis in (0..ndim).rev() {
            if axis == d { continue; }
            let s = shape[axis] as usize;
            let idx = remaining % s;
            remaining /= s;
            base_in += idx * strides[axis];
            base_out += idx * out_strides[oi + if oi >= d { 0 } else { 0 }];
            oi += 1;
        }
        // Recalculate base_out properly using output strides
        let mut remaining2 = slice_idx;
        base_out = 0;
        for axis in (0..ndim).rev() {
            if axis == d { continue; }
            let s = shape[axis] as usize;
            let idx = remaining2 % s;
            remaining2 /= s;
            let out_axis = axis; // output has same ndim
            base_out += idx * out_strides[out_axis];
        }

        // Min-heap of (value, original_index), keeps top-k largest
        let mut heap: BinaryHeap<Reverse<(ordered_float::OrderedFloat<f64>, usize)>> = BinaryHeap::with_capacity(k);

        for i in 0..dim_size {
            let offset = base_in + i * strides[d];
            let val = unsafe { *tensor.data_f64().add(offset) };
            let entry = Reverse((ordered_float::OrderedFloat(val), i));

            if heap.len() < k {
                heap.push(entry);
            } else if let Some(&Reverse((min_val, _))) = heap.peek() {
                if ordered_float::OrderedFloat(val) > min_val {
                    heap.pop();
                    heap.push(entry);
                }
            }
        }

        // Extract sorted descending
        let mut results: Vec<(f64, usize)> = heap.into_iter()
            .map(|Reverse((v, i))| (v.into_inner(), i))
            .collect();
        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for (j, (val, orig_idx)) in results.iter().enumerate() {
            let out_offset = base_out + j * out_strides[d];
            unsafe {
                *val_tensor.data_f64().add(out_offset) = *val;
                *idx_tensor.data_f64().add(out_offset) = *orig_idx as f64;
            }
        }
    }

    // Build result dict: {"values": val_tensor, "indices": idx_tensor}
    let dict = crate::dict::nsl_dict_new();
    let key_values = crate::string::nsl_str_from_rust("values");
    let key_indices = crate::string::nsl_str_from_rust("indices");
    crate::dict::nsl_dict_set_str(dict, key_values, val_ptr);
    crate::dict::nsl_dict_set_str(dict, key_indices, idx_ptr);
    dict
}
```

**Note:** This uses `ordered_float` for heap comparison. Add `ordered-float = "4"` to Cargo.toml, OR avoid the heap and use a simpler sort-based approach for the initial implementation:

**Simpler alternative (no extra dep):**

```rust
// Instead of BinaryHeap, just collect all values and partial sort:
let mut entries: Vec<(f64, usize)> = (0..dim_size).map(|i| {
    let offset = base_in + i * strides[d];
    let val = unsafe { *tensor.data_f64().add(offset) };
    (val, i)
}).collect();
entries.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
entries.truncate(k);
```

Use this simpler approach — it's O(n log n) but avoids an extra dependency. For vocabulary sizes (50K), the difference is negligible.

- [ ] **Step 3: Implement `nsl_tensor_multinomial`**

Add to `sampling.rs`:

```rust
/// Categorical sampling from probability distribution.
/// Input: 1D or 2D probability tensor (need NOT sum to exactly 1.0).
/// Returns: i64-dtype index tensor.
/// Algorithm: CDF binary search with self-normalization.
#[no_mangle]
pub extern "C" fn nsl_tensor_multinomial(tensor_ptr: i64, num_samples: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let num_samples = num_samples as usize;

    if ndim == 1 {
        // 1D: single row of probabilities
        let n = tensor.len as usize;
        let out_ptr = crate::cpu::create_tensor_with_shape_rs(&[num_samples as i64]);
        let out = NslTensor::from_ptr(out_ptr);

        // Build CDF
        let mut cdf = Vec::with_capacity(n);
        let mut cumsum = 0.0_f64;
        for i in 0..n {
            let p = unsafe { *tensor.data_f64().add(i) };
            cumsum += p.max(0.0); // clamp negatives to 0
            cdf.push(cumsum);
        }
        let total = cumsum; // self-normalization: divide sample by total

        RNG.with(|r| {
            let mut rng = r.borrow_mut();
            for s in 0..num_samples {
                let u: f64 = rng.gen::<f64>() * total; // scale to CDF range
                // Binary search for first CDF entry >= u
                let idx = match cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)) {
                    Ok(i) => i,
                    Err(i) => i.min(n - 1),
                };
                unsafe { *out.data_f64().add(s) = idx as f64 };
            }
        });

        out_ptr
    } else if ndim == 2 {
        // 2D: batch of probability rows
        let rows = unsafe { *tensor.shape.add(0) } as usize;
        let cols = unsafe { *tensor.shape.add(1) } as usize;
        let out_ptr = crate::cpu::create_tensor_with_shape_rs(&[rows as i64, num_samples as i64]);
        let out = NslTensor::from_ptr(out_ptr);
        let out_strides = crate::cpu::get_strides_vec(out);
        let in_strides = crate::cpu::get_strides_vec(tensor);

        for row in 0..rows {
            let row_base = row * in_strides[0];
            let mut cdf = Vec::with_capacity(cols);
            let mut cumsum = 0.0_f64;
            for c in 0..cols {
                let p = unsafe { *tensor.data_f64().add(row_base + c * in_strides[1]) };
                cumsum += p.max(0.0);
                cdf.push(cumsum);
            }
            let total = cumsum;

            RNG.with(|r| {
                let mut rng = r.borrow_mut();
                for s in 0..num_samples {
                    let u: f64 = rng.gen::<f64>() * total;
                    let idx = match cdf.binary_search_by(|v| v.partial_cmp(&u).unwrap_or(std::cmp::Ordering::Equal)) {
                        Ok(i) => i,
                        Err(i) => i.min(cols - 1),
                    };
                    unsafe {
                        *out.data_f64().add(row * out_strides[0] + s * out_strides[1]) = idx as f64;
                    };
                }
            });
        }

        out_ptr
    } else {
        eprintln!("nsl: multinomial only supports 1D or 2D tensors, got {}D", ndim);
        std::process::abort();
    }
}
```

- [ ] **Step 4: Implement `nsl_tensor_argmax`**

Add to `sampling.rs`:

```rust
/// Argmax along a dimension. Returns i64-dtype index tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_argmax(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = crate::cpu::get_shape_vec(tensor);
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };

    if d >= ndim {
        eprintln!("nsl: argmax dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let dim_size = shape[d] as usize;
    let strides = crate::cpu::get_strides_vec(tensor);

    // Output shape: input shape with dim d removed
    let out_shape: Vec<i64> = shape.iter().enumerate()
        .filter(|&(i, _)| i != d)
        .map(|(_, &s)| s)
        .collect();

    let result_ptr = if out_shape.is_empty() {
        // Scalar output (1D input)
        crate::cpu::create_tensor_with_shape_rs(&[1])
    } else {
        crate::cpu::create_tensor_with_shape_rs(&out_shape)
    };
    let result = NslTensor::from_ptr(result_ptr);

    let num_slices = (tensor.len as usize) / dim_size;

    for slice_idx in 0..num_slices {
        let mut remaining = slice_idx;
        let mut base_offset: usize = 0;
        for axis in (0..ndim).rev() {
            if axis == d { continue; }
            let s = shape[axis] as usize;
            let idx = remaining % s;
            remaining /= s;
            base_offset += idx * strides[axis];
        }

        let mut max_val = f64::NEG_INFINITY;
        let mut max_idx: usize = 0;
        for i in 0..dim_size {
            let offset = base_offset + i * strides[d];
            let val = unsafe { *tensor.data_f64().add(offset) };
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        unsafe { *result.data_f64().add(slice_idx) = max_idx as f64 };
    }

    result_ptr
}
```

- [ ] **Step 5: Implement `nsl_tensor_cumsum`**

Add to `sampling.rs`:

```rust
/// Cumulative sum along a dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_cumsum(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = crate::cpu::get_shape_vec(tensor);
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };

    if d >= ndim {
        eprintln!("nsl: cumsum dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let dim_size = shape[d] as usize;
    let strides = crate::cpu::get_strides_vec(tensor);

    // Output has same shape as input
    let result_ptr = crate::cpu::create_tensor_with_shape_rs(&shape);
    let result = NslTensor::from_ptr(result_ptr);
    let num_slices = (tensor.len as usize) / dim_size;

    for slice_idx in 0..num_slices {
        let mut remaining = slice_idx;
        let mut base_offset: usize = 0;
        for axis in (0..ndim).rev() {
            if axis == d { continue; }
            let s = shape[axis] as usize;
            let idx = remaining % s;
            remaining /= s;
            base_offset += idx * strides[axis];
        }

        let mut running_sum = 0.0_f64;
        for i in 0..dim_size {
            let offset = base_offset + i * strides[d];
            let val = unsafe { *tensor.data_f64().add(offset) };
            running_sum += val;
            unsafe { *result.data_f64().add(offset) = running_sum };
        }
    }

    result_ptr
}
```

- [ ] **Step 6: Implement `nsl_tensor_lt_scalar`**

Add to `sampling.rs`:

```rust
/// Element-wise less-than comparison with scalar.
/// Returns f64 tensor: 1.0 where element < scalar, 0.0 otherwise.
#[no_mangle]
pub extern "C" fn nsl_tensor_lt_scalar(tensor_ptr: i64, scalar: f64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape = crate::cpu::get_shape_vec(tensor);
    let result_ptr = crate::cpu::create_tensor_with_shape_rs(&shape);
    let result = NslTensor::from_ptr(result_ptr);

    for i in 0..tensor.len as usize {
        let val = unsafe { *tensor.data_f64().add(i) };
        unsafe { *result.data_f64().add(i) = if val < scalar { 1.0 } else { 0.0 } };
    }

    result_ptr
}
```

- [ ] **Step 7: Declare module in lib.rs**

In `crates/nsl-runtime/src/lib.rs`, add after the `pub mod quantize;` line:

```rust
pub mod sampling;
```

- [ ] **Step 8: Add helper function `nsl_str_from_rust` if missing**

Check if `crate::string::nsl_str_from_rust` exists. If not, add to `crates/nsl-runtime/src/string.rs`:

```rust
/// Create a NUL-terminated C string from a Rust &str. Returns pointer as i64.
pub fn nsl_str_from_rust(s: &str) -> i64 {
    let ptr = crate::memory::checked_alloc(s.len() + 1);
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr(), ptr, s.len());
        *ptr.add(s.len()) = 0; // NUL terminator
    }
    ptr as i64
}
```

If it already exists (perhaps named differently like `alloc_c_string`), use that instead.

- [ ] **Step 9: Write Rust unit tests for sampling module**

Add at the bottom of `sampling.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_1d_tensor(data: &[f64]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs(&[data.len() as i64]);
        let t = NslTensor::from_ptr(ptr);
        for (i, &v) in data.iter().enumerate() {
            unsafe { *t.data_f64().add(i) = v };
        }
        ptr
    }

    #[test]
    fn test_topk_basic() {
        let t = make_1d_tensor(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        let dict = nsl_tensor_topk(t, 3, 0);
        let vals_key = crate::string::nsl_str_from_rust("values");
        let vals_ptr = crate::dict::nsl_dict_get_str(dict, vals_key);
        let vals = NslTensor::from_ptr(vals_ptr);
        // Top 3 should be [9.0, 6.0, 5.0] sorted descending
        assert_eq!(unsafe { *vals.data_f64().add(0) }, 9.0);
        assert_eq!(unsafe { *vals.data_f64().add(1) }, 6.0);
        assert_eq!(unsafe { *vals.data_f64().add(2) }, 5.0);
    }

    #[test]
    fn test_multinomial_deterministic() {
        nsl_manual_seed(42);
        let probs = make_1d_tensor(&[0.1, 0.2, 0.3, 0.4]);
        let s1 = nsl_tensor_multinomial(probs, 1);

        nsl_manual_seed(42);
        let s2 = nsl_tensor_multinomial(probs, 1);

        let v1 = unsafe { *NslTensor::from_ptr(s1).data_f64() };
        let v2 = unsafe { *NslTensor::from_ptr(s2).data_f64() };
        assert_eq!(v1, v2, "same seed must produce same sample");
    }

    #[test]
    fn test_multinomial_unnormalized() {
        // Should not panic even with unnormalized probabilities
        nsl_manual_seed(0);
        let probs = make_1d_tensor(&[0.5, 0.3, 0.2, 0.001]);
        let result = nsl_tensor_multinomial(probs, 5);
        let r = NslTensor::from_ptr(result);
        assert_eq!(r.len, 5);
        for i in 0..5 {
            let idx = unsafe { *r.data_f64().add(i) } as usize;
            assert!(idx < 4, "sampled index should be in range");
        }
    }

    #[test]
    fn test_argmax() {
        let t = make_1d_tensor(&[1.0, 5.0, 3.0, 2.0]);
        let result = nsl_tensor_argmax(t, 0);
        let r = NslTensor::from_ptr(result);
        assert_eq!(unsafe { *r.data_f64() }, 1.0); // index 1 has value 5.0
    }

    #[test]
    fn test_cumsum() {
        let t = make_1d_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let result = nsl_tensor_cumsum(t, 0);
        let r = NslTensor::from_ptr(result);
        assert_eq!(unsafe { *r.data_f64().add(0) }, 1.0);
        assert_eq!(unsafe { *r.data_f64().add(1) }, 3.0);
        assert_eq!(unsafe { *r.data_f64().add(2) }, 6.0);
        assert_eq!(unsafe { *r.data_f64().add(3) }, 10.0);
    }

    #[test]
    fn test_lt_scalar() {
        let t = make_1d_tensor(&[0.1, 0.5, 0.9, 0.3]);
        let result = nsl_tensor_lt_scalar(t, 0.5);
        let r = NslTensor::from_ptr(result);
        assert_eq!(unsafe { *r.data_f64().add(0) }, 1.0); // 0.1 < 0.5
        assert_eq!(unsafe { *r.data_f64().add(1) }, 0.0); // 0.5 not < 0.5
        assert_eq!(unsafe { *r.data_f64().add(2) }, 0.0); // 0.9 not < 0.5
        assert_eq!(unsafe { *r.data_f64().add(3) }, 1.0); // 0.3 < 0.5
    }
}
```

- [ ] **Step 10: Run tests**

Run: `cargo test -p nsl-runtime -- sampling`
Expected: All 5 tests pass

- [ ] **Step 11: Commit**

```bash
git add crates/nsl-runtime/src/sampling.rs crates/nsl-runtime/src/lib.rs crates/nsl-runtime/Cargo.toml
git commit -m "feat(m19): implement sampling primitives (topk, multinomial, argmax, cumsum, lt_scalar, RNG)"
```

---

## Chunk 2: Data Sources + Slice Assignment + Packing

### Task 5: Implement Data Source Readers (JSONL, CSV, mmap)

**Files:**
- Create: `crates/nsl-runtime/src/data_source.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

**Role boundaries:** `load_jsonl` and `load_csv` are small-data utilities (eval sets, prompt lists). `load_mmap` is the heavy-duty training pathway (zero-copy, u16 support).

- [ ] **Step 1: Create `data_source.rs` with JSONL loader**

Create `crates/nsl-runtime/src/data_source.rs`:

```rust
//! Data source readers: JSONL, CSV, memory-mapped binary files.

use std::ffi::CStr;
use std::os::raw::c_char;
use crate::tensor::NslTensor;
use crate::memory::{checked_alloc, checked_alloc_zeroed};

unsafe fn path_from_ptr(ptr: i64, len: i64) -> String {
    let slice = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    String::from_utf8_lossy(slice).into_owned()
}

unsafe fn field_from_ptr(ptr: i64, len: i64) -> String {
    let slice = std::slice::from_raw_parts(ptr as *const u8, len as usize);
    String::from_utf8_lossy(slice).into_owned()
}

/// Load a JSONL file, extracting a named field from each line.
/// Returns NslList of string values (field contents).
/// Small-data utility: for eval sets, prompt lists (thousands of rows).
#[no_mangle]
pub extern "C" fn nsl_load_jsonl(path_ptr: i64, path_len: i64, field_ptr: i64, field_len: i64) -> i64 {
    let path = unsafe { path_from_ptr(path_ptr, path_len) };
    let field = unsafe { field_from_ptr(field_ptr, field_len) };

    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("nsl: load_jsonl: could not read '{}': {}", path, e);
            std::process::abort();
        }
    };

    let list = crate::list::nsl_list_new();
    for (line_num, line) in contents.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }
        match serde_json::from_str::<serde_json::Value>(line) {
            Ok(obj) => {
                if let Some(val) = obj.get(&field) {
                    let s = match val {
                        serde_json::Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let c_str = crate::string::nsl_str_from_rust(&s);
                    crate::list::nsl_list_push(list, c_str);
                }
                // Skip lines missing the field (silent)
            }
            Err(_) => {
                eprintln!("nsl: load_jsonl: skipping malformed line {}", line_num + 1);
            }
        }
    }
    list
}
```

- [ ] **Step 2: Add CSV loader**

Append to `data_source.rs`:

```rust
/// Load a CSV file, extracting a column by index.
/// Returns NslList of string values.
/// has_header: 1 = skip first line, 0 = don't skip.
#[no_mangle]
pub extern "C" fn nsl_load_csv(path_ptr: i64, path_len: i64, col_idx: i64, has_header: i64) -> i64 {
    let path = unsafe { path_from_ptr(path_ptr, path_len) };
    let col = col_idx as usize;

    let contents = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("nsl: load_csv: could not read '{}': {}", path, e);
            std::process::abort();
        }
    };

    let list = crate::list::nsl_list_new();
    for (i, line) in contents.lines().enumerate() {
        if i == 0 && has_header != 0 { continue; }
        let line = line.trim();
        if line.is_empty() { continue; }

        // Simple CSV parsing: split on commas, handle quoted fields
        let fields = parse_csv_line(line);
        if col < fields.len() {
            let c_str = crate::string::nsl_str_from_rust(&fields[col]);
            crate::list::nsl_list_push(list, c_str);
        }
    }
    list
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();

    while let Some(ch) = chars.next() {
        match ch {
            '"' => {
                if in_quotes {
                    if chars.peek() == Some(&'"') {
                        current.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                } else {
                    in_quotes = true;
                }
            }
            ',' if !in_quotes => {
                fields.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    fields.push(current.trim().to_string());
    fields
}
```

- [ ] **Step 3: Add memory-mapped binary loader**

Append to `data_source.rs`:

```rust
/// Memory-map a binary file as a flat 1D tensor.
/// dtype: 0=f64, 1=f32, 2=i32, 3=u16
/// Returns NslTensor with owns_data=0 (data points into mmap region).
/// The mmap handle is leaked (lives for process lifetime, OS reclaims on exit).
/// u16 (dtype=3) is the primary pathway for pre-tokenized LLM datasets.
#[no_mangle]
pub extern "C" fn nsl_load_mmap(path_ptr: i64, path_len: i64, dtype: i64) -> i64 {
    let path = unsafe { path_from_ptr(path_ptr, path_len) };

    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("nsl: load_mmap: could not open '{}': {}", path, e);
            std::process::abort();
        }
    };

    let mmap = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("nsl: load_mmap: could not mmap '{}': {}", path, e);
            std::process::abort();
        }
    };

    let file_len = mmap.len();

    let (elem_count, data_ptr, out_dtype) = match dtype {
        0 => {
            // f64: 8 bytes per element
            let count = file_len / 8;
            (count, mmap.as_ptr() as *mut std::ffi::c_void, 0u8)
        }
        1 => {
            // f32: 4 bytes per element
            let count = file_len / 4;
            (count, mmap.as_ptr() as *mut std::ffi::c_void, 1u8)
        }
        2 => {
            // i32: 4 bytes per element — convert to f64 on read
            let count = file_len / 4;
            let src = mmap.as_ptr() as *const i32;
            let dst = checked_alloc(count * 8) as *mut f64;
            for i in 0..count {
                unsafe { *dst.add(i) = *src.add(i) as f64 };
            }
            // Can't use mmap directly (dtype conversion needed), so owns_data=1
            let tensor = create_mmap_tensor(dst as *mut std::ffi::c_void, count, 0, 1);
            std::mem::forget(mmap); // keep alive
            return tensor;
        }
        3 => {
            // u16: 2 bytes per element — convert to f64 on read
            let count = file_len / 2;
            let src = mmap.as_ptr() as *const u16;
            let dst = checked_alloc(count * 8) as *mut f64;
            for i in 0..count {
                unsafe { *dst.add(i) = *src.add(i) as f64 };
            }
            let tensor = create_mmap_tensor(dst as *mut std::ffi::c_void, count, 0, 1);
            std::mem::forget(mmap);
            return tensor;
        }
        _ => {
            eprintln!("nsl: load_mmap: unsupported dtype {}", dtype);
            std::process::abort();
        }
    };

    // For f64/f32: point directly into mmap (zero-copy)
    let tensor = create_mmap_tensor(data_ptr, elem_count, out_dtype, 0);
    // Leak the mmap so it stays alive for process lifetime
    std::mem::forget(mmap);
    tensor
}

fn create_mmap_tensor(data: *mut std::ffi::c_void, len: usize, dtype: u8, owns_data: u8) -> i64 {
    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = len as i64 };
    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let tensor = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim: 1,
        len: len as i64,
        refcount: 1,
        device: 0,
        dtype,
        owns_data,
    });
    Box::into_raw(tensor) as i64
}
```

- [ ] **Step 4: Declare module in lib.rs**

In `crates/nsl-runtime/src/lib.rs`, add after `pub mod sampling;`:

```rust
pub mod data_source;
```

- [ ] **Step 5: Write Rust unit tests**

Add at end of `data_source.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_jsonl() {
        let tmp = std::env::temp_dir().join("test_m19.jsonl");
        std::fs::write(&tmp, r#"{"text": "hello", "label": 1}
{"text": "world", "label": 2}
{"text": "foo", "label": 3}
"#).unwrap();

        let path = tmp.to_str().unwrap();
        let path_ptr = crate::string::nsl_str_from_rust(path);
        let field_ptr = crate::string::nsl_str_from_rust("text");
        // Need raw pointer + len for the FFI
        let path_bytes = path.as_bytes();
        let field_bytes = "text".as_bytes();
        let result = nsl_load_jsonl(
            path_bytes.as_ptr() as i64, path_bytes.len() as i64,
            field_bytes.as_ptr() as i64, field_bytes.len() as i64,
        );
        assert_eq!(crate::list::nsl_list_len(result), 3);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_load_csv() {
        let tmp = std::env::temp_dir().join("test_m19.csv");
        std::fs::write(&tmp, "name,age\nAlice,30\nBob,25\n").unwrap();

        let path = tmp.to_str().unwrap();
        let path_bytes = path.as_bytes();
        let result = nsl_load_csv(
            path_bytes.as_ptr() as i64, path_bytes.len() as i64,
            0, // col_idx = 0 (name column)
            1, // has_header = true
        );
        assert_eq!(crate::list::nsl_list_len(result), 2);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_load_mmap_f64() {
        let tmp = std::env::temp_dir().join("test_m19.bin");
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_ne_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();

        let path = tmp.to_str().unwrap();
        let path_bytes = path.as_bytes();
        let result = nsl_load_mmap(
            path_bytes.as_ptr() as i64, path_bytes.len() as i64,
            0, // dtype = f64
        );
        let t = NslTensor::from_ptr(result);
        assert_eq!(t.len, 4);
        assert_eq!(t.owns_data, 0); // zero-copy mmap
        assert_eq!(unsafe { *t.data_f64() }, 1.0);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_load_mmap_u16() {
        let tmp = std::env::temp_dir().join("test_m19_u16.bin");
        let data: Vec<u16> = vec![100, 200, 50256, 42];
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_ne_bytes()).collect();
        std::fs::write(&tmp, &bytes).unwrap();

        let path = tmp.to_str().unwrap();
        let path_bytes = path.as_bytes();
        let result = nsl_load_mmap(
            path_bytes.as_ptr() as i64, path_bytes.len() as i64,
            3, // dtype = u16
        );
        let t = NslTensor::from_ptr(result);
        assert_eq!(t.len, 4);
        assert_eq!(t.owns_data, 1); // converted to f64
        assert_eq!(unsafe { *t.data_f64().add(2) }, 50256.0);
        std::fs::remove_file(tmp).ok();
    }
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p nsl-runtime -- data_source`
Expected: All 4 tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/src/data_source.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(m19): implement data source readers (JSONL, CSV, mmap with u16)"
```

---

### Task 6: Implement Slice Assignment Runtime

**Files:**
- Modify: `crates/nsl-runtime/src/tensor.rs`

NSL currently only supports simple index assignment (`list[i] = val`). The `generate()` function needs `tensor[0, i] = value` for pre-allocated buffer writes. We add two functions: element set (simple) and slice assign (range).

- [ ] **Step 1: Add `nsl_tensor_set_element` for single-element mutation**

In `crates/nsl-runtime/src/tensor.rs`, add:

```rust
/// Set a single element in a tensor by flat indices.
/// indices_ptr points to an array of i64 indices, one per dimension.
/// value is the f64 value to set.
#[no_mangle]
pub extern "C" fn nsl_tensor_set_element(tensor_ptr: i64, indices_ptr: i64, num_indices: i64, value: f64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let n = num_indices as usize;

    if n != ndim {
        eprintln!("nsl: set_element: expected {} indices, got {}", ndim, n);
        std::process::abort();
    }

    let strides = get_strides_vec(tensor);
    let mut offset: usize = 0;
    for d in 0..ndim {
        let idx = unsafe { *(indices_ptr as *const i64).add(d) } as usize;
        let dim_size = unsafe { *tensor.shape.add(d) } as usize;
        if idx >= dim_size {
            eprintln!("nsl: set_element: index {} out of bounds for dim {} (size {})", idx, d, dim_size);
            std::process::abort();
        }
        offset += idx * strides[d];
    }

    unsafe { *tensor.data_f64().add(offset) = value };
}

/// Copy a source tensor into a slice of the target tensor.
/// Each NslSliceDim specifies one dimension: scalar index or range.
#[repr(C)]
pub struct NslSliceDim {
    pub is_scalar: u8,  // 1 = single index, 0 = range
    pub start: i64,
    pub end: i64,       // ignored if is_scalar=1
}

/// Assign src tensor into a slice of target tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_slice_assign(
    target_ptr: i64,
    src_ptr: i64,
    dims_ptr: i64,
    num_dims: i64,
) {
    let target = NslTensor::from_ptr(target_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    let ndim = num_dims as usize;
    let dims = unsafe { std::slice::from_raw_parts(dims_ptr as *const NslSliceDim, ndim) };
    let target_strides = get_strides_vec(target);
    let target_shape = get_shape_vec(target);

    // Compute the slice region: for each dim, determine start..end range
    let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(ndim);
    for d in 0..ndim {
        let dim_size = target_shape[d] as usize;
        if dims[d].is_scalar != 0 {
            let idx = if dims[d].start < 0 { (dim_size as i64 + dims[d].start) as usize } else { dims[d].start as usize };
            ranges.push((idx, idx + 1));
        } else {
            let start = if dims[d].start < 0 { (dim_size as i64 + dims[d].start) as usize } else { dims[d].start as usize };
            let end = if dims[d].end < 0 { (dim_size as i64 + dims[d].end) as usize } else { dims[d].end.min(dim_size as i64) as usize };
            ranges.push((start, end));
        }
    }

    // Iterate over all positions in the slice and copy from src
    let mut src_flat = 0usize;
    let total_src = src.len as usize;

    fn recurse(
        depth: usize,
        ndim: usize,
        ranges: &[(usize, usize)],
        target: &NslTensor,
        src: &NslTensor,
        target_strides: &[usize],
        target_offset: usize,
        src_flat: &mut usize,
    ) {
        if depth == ndim {
            if *src_flat < src.len as usize {
                let val = unsafe { *src.data_f64().add(*src_flat) };
                unsafe { *target.data_f64().add(target_offset) = val };
                *src_flat += 1;
            }
            return;
        }
        let (start, end) = ranges[depth];
        for i in start..end {
            recurse(
                depth + 1, ndim, ranges, target, src,
                target_strides, target_offset + i * target_strides[depth],
                src_flat,
            );
        }
    }

    recurse(0, ndim, &ranges, target, src, &target_strides, 0, &mut src_flat);
}
```

- [ ] **Step 2: Write unit tests**

Add in the `#[cfg(test)]` section of `tensor.rs` (or in a new test section):

```rust
#[test]
fn test_set_element() {
    let t = nsl_tensor_zeros(nsl_list_from_slice(&[2, 3]));
    let indices: [i64; 2] = [1, 2];
    nsl_tensor_set_element(t, indices.as_ptr() as i64, 2, 42.0);
    let tensor = NslTensor::from_ptr(t);
    // Element at [1, 2] should be 42.0
    assert_eq!(unsafe { *tensor.data_f64().add(1 * 3 + 2) }, 42.0);
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p nsl-runtime -- test_set_element`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/tensor.rs
git commit -m "feat(m19): add tensor set_element and slice_assign for pre-allocated buffers"
```

---

### Task 7: Implement Sequence Packing

**Files:**
- Create: `crates/nsl-runtime/src/packing.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

Continuous stream chunking: treat dataset as 1D token stream, slice B*S tokens per batch, build block-diagonal attention mask from EOS boundaries.

- [ ] **Step 1: Create `packing.rs`**

Create `crates/nsl-runtime/src/packing.rs`:

```rust
//! Sequence packing via continuous stream chunking.
//! Treats the dataset as a 1D token stream joined by EOS separators.
//! Produces packed batches with block-diagonal attention masks.

use crate::tensor::NslTensor;
use crate::memory::{checked_alloc, checked_alloc_zeroed};

/// A packed batch ready for training.
pub struct PackedBatch {
    /// [batch_size * seq_len] flattened input token ids
    pub input_ids: Vec<i64>,
    /// [batch_size * seq_len] shifted labels, -100 at boundaries
    pub labels: Vec<i64>,
    /// [batch_size * seq_len * seq_len] block-diagonal additive mask
    pub mask: Vec<f32>,
    pub batch_size: usize,
    pub seq_len: usize,
}

/// Pack sequences from a continuous token stream.
///
/// # Arguments
/// * `data` - pointer to the full token stream (f64 values cast to i64)
/// * `data_len` - total number of tokens in the stream
/// * `cursor` - current position in the stream (advanced by B*S)
/// * `batch_size` - number of sequences per batch
/// * `seq_len` - tokens per sequence
/// * `eos_token` - separator token id (document boundary marker)
///
/// # Returns
/// `None` if cursor has reached end of data (epoch complete).
pub fn pack_batch(
    data: *const f64,
    data_len: usize,
    cursor: &mut usize,
    batch_size: usize,
    seq_len: usize,
    eos_token: i64,
) -> Option<PackedBatch> {
    let total_tokens = batch_size * seq_len;

    // Check if we have enough tokens remaining
    if *cursor + total_tokens > data_len {
        return None; // Epoch complete
    }

    let mut input_ids = Vec::with_capacity(total_tokens);
    let mut labels = Vec::with_capacity(total_tokens);
    let mut mask = vec![0.0f32; total_tokens * seq_len]; // Full B*S*S mask, init to -1e9 below

    // Slice tokens from stream
    for i in 0..total_tokens {
        let token = unsafe { *data.add(*cursor + i) } as i64;
        input_ids.push(token);
    }
    *cursor += total_tokens;

    // Build labels: shifted by 1, -100 at EOS boundaries
    for b in 0..batch_size {
        let seq_start = b * seq_len;
        for s in 0..seq_len {
            let pos = seq_start + s;
            if s < seq_len - 1 {
                let next_token = input_ids[pos + 1];
                // If current token is EOS, label = -100 (don't predict across documents)
                if input_ids[pos] == eos_token {
                    labels.push(-100);
                } else {
                    labels.push(next_token);
                }
            } else {
                labels.push(-100); // Last position: no next token in this sequence
            }
        }
    }

    // Build block-diagonal attention mask per sequence
    // Initialize all to -1e9 (masked), then set 0.0 for valid positions
    let neg_inf: f32 = -1e9;
    for val in mask.iter_mut() {
        *val = neg_inf;
    }

    for b in 0..batch_size {
        let seq_start = b * seq_len;

        // Assign document IDs by scanning for EOS tokens
        let mut doc_ids = vec![0u32; seq_len];
        let mut current_doc: u32 = 0;
        for s in 0..seq_len {
            doc_ids[s] = current_doc;
            if input_ids[seq_start + s] == eos_token {
                current_doc += 1;
            }
        }

        // Set mask: 0.0 where same document AND causal (j <= i)
        let mask_base = b * seq_len * seq_len;
        for i in 0..seq_len {
            for j in 0..=i {
                if doc_ids[i] == doc_ids[j] {
                    mask[mask_base + i * seq_len + j] = 0.0;
                }
            }
        }
    }

    Some(PackedBatch {
        input_ids,
        labels,
        mask,
        batch_size,
        seq_len,
    })
}

/// Convert a PackedBatch into an NslDict with three tensors.
/// Keys: "input_ids" [B,S], "labels" [B,S], "attention_mask" [B,S,S]
pub fn packed_batch_to_dict(batch: &PackedBatch) -> i64 {
    let b = batch.batch_size as i64;
    let s = batch.seq_len as i64;

    // input_ids: [B, S] i64 stored as f64
    let ids_ptr = crate::cpu::create_tensor_with_shape_rs(&[b, s]);
    let ids = NslTensor::from_ptr(ids_ptr);
    for (i, &tok) in batch.input_ids.iter().enumerate() {
        unsafe { *ids.data_f64().add(i) = tok as f64 };
    }

    // labels: [B, S]
    let labels_ptr = crate::cpu::create_tensor_with_shape_rs(&[b, s]);
    let labels = NslTensor::from_ptr(labels_ptr);
    for (i, &lab) in batch.labels.iter().enumerate() {
        unsafe { *labels.data_f64().add(i) = lab as f64 };
    }

    // attention_mask: [B, S, S] f64 (from f32 source)
    let mask_ptr = crate::cpu::create_tensor_with_shape_rs(&[b, s, s]);
    let mask = NslTensor::from_ptr(mask_ptr);
    for (i, &m) in batch.mask.iter().enumerate() {
        unsafe { *mask.data_f64().add(i) = m as f64 };
    }

    // Build dict
    let dict = crate::dict::nsl_dict_new();
    let k_ids = crate::string::nsl_str_from_rust("input_ids");
    let k_labels = crate::string::nsl_str_from_rust("labels");
    let k_mask = crate::string::nsl_str_from_rust("attention_mask");
    crate::dict::nsl_dict_set_str(dict, k_ids, ids_ptr);
    crate::dict::nsl_dict_set_str(dict, k_labels, labels_ptr);
    crate::dict::nsl_dict_set_str(dict, k_mask, mask_ptr);
    dict
}

/// Packing efficiency: always 1.0 for continuous stream chunking (no padding).
#[no_mangle]
pub extern "C" fn nsl_packing_efficiency(_dl_ptr: i64) -> f64 {
    1.0
}
```

- [ ] **Step 2: Declare module in lib.rs**

Add to `crates/nsl-runtime/src/lib.rs`:

```rust
pub mod packing;
```

- [ ] **Step 3: Write unit tests**

Add at end of `packing.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_batch_basic() {
        // Stream: [10, 20, EOS=0, 30, 40, 50, EOS=0, 60]
        let data: Vec<f64> = vec![10.0, 20.0, 0.0, 30.0, 40.0, 50.0, 0.0, 60.0];
        let mut cursor = 0usize;
        let batch = pack_batch(data.as_ptr(), data.len(), &mut cursor, 2, 4, 0).unwrap();

        assert_eq!(cursor, 8); // consumed all 8 tokens
        assert_eq!(batch.input_ids, vec![10, 20, 0, 30, 40, 50, 0, 60]);

        // Labels at EOS positions should be -100
        assert_eq!(batch.labels[2], -100); // position 2 is EOS
        assert_eq!(batch.labels[6], -100); // position 6 is EOS
    }

    #[test]
    fn test_pack_batch_mask() {
        // Two tokens from doc A, EOS, one token from doc B → [A, A, EOS, B]
        let data: Vec<f64> = vec![1.0, 2.0, 0.0, 3.0];
        let mut cursor = 0usize;
        let batch = pack_batch(data.as_ptr(), data.len(), &mut cursor, 1, 4, 0).unwrap();

        // Check mask for sequence 0:
        // Position 0 (doc 0): can attend to 0 only
        assert_eq!(batch.mask[0 * 4 + 0], 0.0);  // [0,0] = same doc, causal ✓
        // Position 1 (doc 0): can attend to 0,1
        assert_eq!(batch.mask[1 * 4 + 0], 0.0);  // [1,0] = same doc, causal ✓
        assert_eq!(batch.mask[1 * 4 + 1], 0.0);  // [1,1] = same doc, causal ✓
        // Position 2 (EOS, still doc 0): can attend to 0,1,2
        assert_eq!(batch.mask[2 * 4 + 0], 0.0);  // [2,0] = same doc ✓
        // Position 3 (doc 1): cannot attend to positions 0,1,2 (different doc)
        assert_eq!(batch.mask[3 * 4 + 0], -1e9);  // [3,0] = cross-doc ✗
        assert_eq!(batch.mask[3 * 4 + 1], -1e9);  // [3,1] = cross-doc ✗
        assert_eq!(batch.mask[3 * 4 + 2], -1e9);  // [3,2] = cross-doc ✗
        assert_eq!(batch.mask[3 * 4 + 3], 0.0);   // [3,3] = same doc, causal ✓
    }

    #[test]
    fn test_pack_batch_epoch_end() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0];
        let mut cursor = 0usize;
        // Request 4 tokens but only 3 available → None
        let result = pack_batch(data.as_ptr(), data.len(), &mut cursor, 1, 4, 0);
        assert!(result.is_none());
    }

    #[test]
    fn test_packed_batch_to_dict() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let mut cursor = 0usize;
        let batch = pack_batch(data.as_ptr(), data.len(), &mut cursor, 1, 4, 0).unwrap();
        let dict = packed_batch_to_dict(&batch);

        // Verify dict has 3 keys
        assert_eq!(crate::dict::nsl_dict_len(dict), 3);

        // Verify input_ids shape is [1, 4]
        let k = crate::string::nsl_str_from_rust("input_ids");
        let ids_ptr = crate::dict::nsl_dict_get_str(dict, k);
        let ids = NslTensor::from_ptr(ids_ptr);
        assert_eq!(ids.ndim, 2);
        assert_eq!(unsafe { *ids.shape }, 1);
        assert_eq!(unsafe { *ids.shape.add(1) }, 4);
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p nsl-runtime -- packing`
Expected: All 4 tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/packing.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(m19): implement continuous stream sequence packing with block-diagonal masks"
```

---

### Task 8: Implement DataLoader Runtime

**Files:**
- Create: `crates/nsl-runtime/src/dataloader.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

Multi-threaded DataLoader with ring buffer, reorder buffer for deterministic ordering, JSON config deserialization.

- [ ] **Step 1: Create `dataloader.rs` with core struct and config parsing**

Create `crates/nsl-runtime/src/dataloader.rs`:

```rust
//! Multi-threaded DataLoader with reorder buffer for deterministic batch ordering.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Condvar, atomic::{AtomicBool, AtomicUsize, Ordering}};
use std::thread::{self, JoinHandle};

use crate::tensor::NslTensor;
use crate::packing;

struct DataLoaderConfig {
    batch_size: usize,
    seq_len: usize,
    shuffle: bool,
    num_workers: usize,
    prefetch: usize,
    pin_memory: bool,
    drop_last: bool,
    packing: bool,
    pack_separator: i64,
}

struct DataLoader {
    data: *const f64,          // raw pointer into source data (f64 values)
    data_len: usize,           // total elements
    config: DataLoaderConfig,
    // Threading state
    cursor: Arc<AtomicUsize>,
    stop_flag: Arc<AtomicBool>,
    // Reorder buffer for deterministic ordering
    reorder_buffer: Arc<Mutex<HashMap<usize, i64>>>,  // batch_id -> dict_ptr
    expected_batch_id: Arc<AtomicUsize>,
    condvar: Arc<Condvar>,
    worker_handles: Vec<JoinHandle<()>>,
    total_batches: usize,
    // Shuffle state
    shuffle_offsets: Vec<usize>,
}

// DataLoader pointer must be Send for thread spawning
unsafe impl Send for DataLoader {}
unsafe impl Sync for DataLoader {}

impl DataLoaderConfig {
    fn from_json(json_ptr: i64, json_len: i64) -> Self {
        let json_str = unsafe {
            let slice = std::slice::from_raw_parts(json_ptr as *const u8, json_len as usize);
            std::str::from_utf8_unchecked(slice)
        };

        let v: serde_json::Value = serde_json::from_str(json_str).unwrap_or_else(|e| {
            eprintln!("nsl: DataLoader config parse error: {}", e);
            std::process::abort();
        });

        DataLoaderConfig {
            batch_size: v["batch_size"].as_u64().unwrap_or(32) as usize,
            seq_len: v["seq_len"].as_u64().unwrap_or(128) as usize,
            shuffle: v["shuffle"].as_bool().unwrap_or(false),
            num_workers: v["num_workers"].as_u64().unwrap_or(1) as usize,
            prefetch: v["prefetch"].as_u64().unwrap_or(2) as usize,
            pin_memory: v["pin_memory"].as_bool().unwrap_or(false),
            drop_last: v["drop_last"].as_bool().unwrap_or(true),
            packing: v["packing"].as_bool().unwrap_or(false),
            pack_separator: v["pack_separator"].as_i64().unwrap_or(0),
        }
    }
}
```

- [ ] **Step 2: Implement FFI functions**

Continue in `dataloader.rs`:

```rust
/// Create a DataLoader from data pointer + JSON config.
#[no_mangle]
pub extern "C" fn nsl_dataloader_create(
    data_ptr: i64,
    data_len: i64,
    config_ptr: i64,
    config_len: i64,
) -> i64 {
    let config = DataLoaderConfig::from_json(config_ptr, config_len);
    let data_len = data_len as usize;
    let tokens_per_batch = config.batch_size * config.seq_len;
    let total_batches = if config.drop_last {
        data_len / tokens_per_batch
    } else {
        (data_len + tokens_per_batch - 1) / tokens_per_batch
    };

    let dl = Box::new(DataLoader {
        data: data_ptr as *const f64,
        data_len,
        config,
        cursor: Arc::new(AtomicUsize::new(0)),
        stop_flag: Arc::new(AtomicBool::new(false)),
        reorder_buffer: Arc::new(Mutex::new(HashMap::new())),
        expected_batch_id: Arc::new(AtomicUsize::new(0)),
        condvar: Arc::new(Condvar::new()),
        worker_handles: Vec::new(),
        total_batches,
        shuffle_offsets: Vec::new(),
    });

    Box::into_raw(dl) as i64
}

/// Start worker threads to begin prefetching batches.
#[no_mangle]
pub extern "C" fn nsl_dataloader_start(dl_ptr: i64) {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    dl.stop_flag.store(false, Ordering::SeqCst);
    dl.cursor.store(0, Ordering::SeqCst);
    dl.expected_batch_id.store(0, Ordering::SeqCst);

    let num_workers = dl.config.num_workers.max(1);
    let batch_id_counter = Arc::new(AtomicUsize::new(0));

    for _ in 0..num_workers {
        let data = dl.data;
        let data_len = dl.data_len;
        let batch_size = dl.config.batch_size;
        let seq_len = dl.config.seq_len;
        let packing = dl.config.packing;
        let eos = dl.config.pack_separator;
        let total_batches = dl.total_batches;
        let stop_flag = Arc::clone(&dl.stop_flag);
        let batch_counter = Arc::clone(&batch_id_counter);
        let reorder = Arc::clone(&dl.reorder_buffer);
        let condvar = Arc::clone(&dl.condvar);

        let handle = thread::spawn(move || {
            loop {
                if stop_flag.load(Ordering::SeqCst) { break; }

                // Atomically claim a batch_id
                let batch_id = batch_counter.fetch_add(1, Ordering::SeqCst);
                if batch_id >= total_batches {
                    break; // No more batches this epoch
                }

                // Compute cursor for this batch
                let cursor_pos = batch_id * batch_size * seq_len;
                let mut local_cursor = cursor_pos;

                // Build the batch
                let dict_ptr = if packing {
                    match packing::pack_batch(data, data_len, &mut local_cursor, batch_size, seq_len, eos) {
                        Some(batch) => packing::packed_batch_to_dict(&batch),
                        None => 0, // epoch end
                    }
                } else {
                    // Non-packing: simple sequential slicing
                    build_simple_batch(data, data_len, cursor_pos, batch_size, seq_len)
                };

                // Insert into reorder buffer
                {
                    let mut buf = reorder.lock().unwrap();
                    buf.insert(batch_id, dict_ptr);
                }
                condvar.notify_all();
            }
        });
        dl.worker_handles.push(handle);
    }
}

/// Build a simple (non-packed) batch: sequential slicing with causal mask.
fn build_simple_batch(data: *const f64, data_len: usize, cursor: usize, batch_size: usize, seq_len: usize) -> i64 {
    let total = batch_size * seq_len;
    if cursor + total > data_len {
        return 0; // epoch end
    }

    let b = batch_size as i64;
    let s = seq_len as i64;

    // input_ids [B, S]
    let ids_ptr = crate::cpu::create_tensor_with_shape_rs(&[b, s]);
    let ids = NslTensor::from_ptr(ids_ptr);
    for i in 0..total {
        unsafe { *ids.data_f64().add(i) = *data.add(cursor + i) };
    }

    // labels [B, S]: shifted right by 1, last position = -100
    let labels_ptr = crate::cpu::create_tensor_with_shape_rs(&[b, s]);
    let labels = NslTensor::from_ptr(labels_ptr);
    for b_idx in 0..batch_size {
        for s_idx in 0..seq_len {
            let pos = b_idx * seq_len + s_idx;
            let label = if s_idx < seq_len - 1 {
                unsafe { *data.add(cursor + pos + 1) } as i64
            } else {
                -100
            };
            unsafe { *labels.data_f64().add(pos) = label as f64 };
        }
    }

    // attention_mask [B, S, S]: standard causal mask
    let mask_ptr = crate::cpu::create_tensor_with_shape_rs(&[b, s, s]);
    let mask = NslTensor::from_ptr(mask_ptr);
    let neg_inf = -1e9_f64;
    for b_idx in 0..batch_size {
        let base = b_idx * seq_len * seq_len;
        for i in 0..seq_len {
            for j in 0..seq_len {
                let val = if j <= i { 0.0 } else { neg_inf };
                unsafe { *mask.data_f64().add(base + i * seq_len + j) = val };
            }
        }
    }

    let dict = crate::dict::nsl_dict_new();
    let k_ids = crate::string::nsl_str_from_rust("input_ids");
    let k_labels = crate::string::nsl_str_from_rust("labels");
    let k_mask = crate::string::nsl_str_from_rust("attention_mask");
    crate::dict::nsl_dict_set_str(dict, k_ids, ids_ptr);
    crate::dict::nsl_dict_set_str(dict, k_labels, labels_ptr);
    crate::dict::nsl_dict_set_str(dict, k_mask, mask_ptr);
    dict
}

/// Get next batch. Blocks until expected batch is ready.
/// Returns 0 at epoch end.
#[no_mangle]
pub extern "C" fn nsl_dataloader_next_batch(dl_ptr: i64) -> i64 {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    let expected = dl.expected_batch_id.load(Ordering::SeqCst);

    if expected >= dl.total_batches {
        return 0; // Epoch complete
    }

    let mut buf = dl.reorder_buffer.lock().unwrap();
    loop {
        if let Some(dict_ptr) = buf.remove(&expected) {
            dl.expected_batch_id.fetch_add(1, Ordering::SeqCst);
            return dict_ptr;
        }
        buf = dl.condvar.wait(buf).unwrap();
    }
}

/// Reset DataLoader for next epoch.
#[no_mangle]
pub extern "C" fn nsl_dataloader_reset(dl_ptr: i64) {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };

    // Join existing workers
    nsl_dataloader_stop(dl_ptr);

    // Reset state
    dl.cursor.store(0, Ordering::SeqCst);
    dl.expected_batch_id.store(0, Ordering::SeqCst);
    dl.reorder_buffer.lock().unwrap().clear();

    // Restart workers
    nsl_dataloader_start(dl_ptr);
}

/// Stop all worker threads.
#[no_mangle]
pub extern "C" fn nsl_dataloader_stop(dl_ptr: i64) {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    dl.stop_flag.store(true, Ordering::SeqCst);
    for handle in dl.worker_handles.drain(..) {
        let _ = handle.join();
    }
}

/// Free DataLoader and all resources.
#[no_mangle]
pub extern "C" fn nsl_dataloader_free(dl_ptr: i64) {
    if dl_ptr == 0 { return; }
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    nsl_dataloader_stop(dl_ptr);
    // Free any remaining batches in reorder buffer
    let buf = dl.reorder_buffer.lock().unwrap();
    for (_, dict_ptr) in buf.iter() {
        if *dict_ptr != 0 {
            crate::dict::nsl_dict_free(*dict_ptr);
        }
    }
    drop(buf);
    unsafe { drop(Box::from_raw(dl as *mut DataLoader)) };
}
```

- [ ] **Step 3: Declare module in lib.rs**

Add to `crates/nsl-runtime/src/lib.rs`:

```rust
pub mod dataloader;
```

- [ ] **Step 4: Write unit tests**

Add at end of `dataloader.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64).collect()
    }

    fn make_config_json(batch_size: usize, seq_len: usize, num_workers: usize) -> String {
        format!(r#"{{"batch_size":{batch_size},"seq_len":{seq_len},"num_workers":{num_workers},"shuffle":false,"packing":false,"drop_last":true}}"#)
    }

    #[test]
    fn test_dataloader_basic() {
        let data = make_data(32); // 32 tokens
        let config = make_config_json(2, 4, 1); // 2 seqs x 4 tokens = 8 per batch, 4 batches total

        let dl = nsl_dataloader_create(
            data.as_ptr() as i64, data.len() as i64,
            config.as_ptr() as i64, config.len() as i64,
        );
        nsl_dataloader_start(dl);

        // Should get 4 batches
        let mut count = 0;
        loop {
            let batch = nsl_dataloader_next_batch(dl);
            if batch == 0 { break; }
            count += 1;
            crate::dict::nsl_dict_free(batch); // cleanup!
        }
        assert_eq!(count, 4);

        nsl_dataloader_stop(dl);
        nsl_dataloader_free(dl);
    }

    #[test]
    fn test_dataloader_deterministic_order() {
        let data = make_data(16);
        let config = make_config_json(1, 4, 2); // 2 workers, should still be deterministic

        let dl = nsl_dataloader_create(
            data.as_ptr() as i64, data.len() as i64,
            config.as_ptr() as i64, config.len() as i64,
        );
        nsl_dataloader_start(dl);

        // First batch should be tokens [0,1,2,3]
        let batch1 = nsl_dataloader_next_batch(dl);
        assert_ne!(batch1, 0);
        let k = crate::string::nsl_str_from_rust("input_ids");
        let ids = crate::dict::nsl_dict_get_str(batch1, k);
        let t = NslTensor::from_ptr(ids);
        assert_eq!(unsafe { *t.data_f64().add(0) }, 0.0);
        assert_eq!(unsafe { *t.data_f64().add(1) }, 1.0);

        crate::dict::nsl_dict_free(batch1);
        nsl_dataloader_stop(dl);
        nsl_dataloader_free(dl);
    }
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p nsl-runtime -- dataloader`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/dataloader.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(m19): implement multi-threaded DataLoader with reorder buffer"
```

---

## Chunk 3: Compiler Intrinsics + Codegen

### Task 9: Register M19 Runtime Functions in Codegen Builtins

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`

All new runtime functions must be registered in the `RUNTIME_FUNCTIONS` array so Cranelift can emit calls to them.

- [ ] **Step 1: Add M19 entries to RUNTIME_FUNCTIONS**

In `crates/nsl-codegen/src/builtins.rs`, add after the ONNX export entry (line ~233, before the closing `];`):

```rust
    // Sampling primitives (M19)
    ("nsl_manual_seed", &[types::I64], None),
    ("nsl_tensor_topk", &[types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_multinomial", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_argmax", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_cumsum", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_lt_scalar", &[types::I64, types::F64], Some(types::I64)),
    // Tensor mutation (M19)
    ("nsl_tensor_set_element", &[types::I64, types::I64, types::I64, types::F64], None),
    ("nsl_tensor_slice_assign", &[types::I64, types::I64, types::I64, types::I64], None),
    // Data sources (M19)
    ("nsl_load_jsonl", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_load_csv", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_load_mmap", &[types::I64, types::I64, types::I64], Some(types::I64)),
    // DataLoader (M19)
    ("nsl_dataloader_create", &[types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_dataloader_start", &[types::I64], None),
    ("nsl_dataloader_next_batch", &[types::I64], Some(types::I64)),
    ("nsl_dataloader_reset", &[types::I64], None),
    ("nsl_dataloader_stop", &[types::I64], None),
    ("nsl_dataloader_free", &[types::I64], None),
    // Packing efficiency (M19)
    ("nsl_packing_efficiency", &[types::I64], Some(types::F64)),
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p nsl-codegen`
Expected: Compiles (just function declarations, no calls yet)

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/builtins.rs
git commit -m "feat(m19): register M19 runtime functions in codegen builtins"
```

---

### Task 10: Register M19 Semantic Builtins

**Files:**
- Modify: `crates/nsl-semantic/src/builtins.rs`

These entries allow the type checker to accept M19 intrinsic calls.

- [ ] **Step 1: Add M19 builtins**

In `crates/nsl-semantic/src/builtins.rs`, add at the end of `register_builtins()` (after the M18b interop entries at line ~601):

```rust
    // Data pipeline intrinsics (M19)
    def("load_jsonl", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(Type::List(Box::new(Type::Str))),
    });
    def("load_csv", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(Type::List(Box::new(Type::Str))),
    });
    def("load_mmap", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(tensor_ret.clone()),
    });
    def("DataLoader", Type::Function {
        params: vec![Type::Unknown],
        ret: Box::new(Type::Unknown),
    });

    // Sampling intrinsics (M19)
    def("topk", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(Type::Dict(Box::new(Type::Str), Box::new(tensor_ret.clone()))),
    });
    def("multinomial", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(tensor_ret.clone()),
    });
    def("argmax", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(tensor_ret.clone()),
    });
    def("manual_seed", Type::Function {
        params: vec![Type::Unknown],
        ret: Box::new(Type::Void),
    });
    def("cumsum", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(tensor_ret.clone()),
    });
    def("lt_scalar", Type::Function {
        params: vec![Type::Unknown, Type::Unknown],
        ret: Box::new(tensor_ret.clone()),
    });
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p nsl-semantic`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-semantic/src/builtins.rs
git commit -m "feat(m19): register M19 semantic builtins for type checking"
```

---

### Task 11: Implement Codegen Intrinsic Handlers

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs`

Add intrinsic handlers in `compile_call` for all M19 functions, following the M18b pattern.

- [ ] **Step 1: Add sampling intrinsic handlers**

In `crates/nsl-codegen/src/expr.rs`, in the `compile_call` function, add after the existing `save_safetensors` handler (around line 1113):

```rust
// ── M19 Sampling Intrinsics ──────────────────────────────────

// manual_seed(seed)
if func_name == "manual_seed" {
    if args.len() != 1 {
        return Err(CodegenError::new("manual_seed() takes exactly 1 argument (seed)"));
    }
    let seed_val = self.compile_expr(builder, state, &args[0].value)?;
    return self.compile_call_by_name(builder, "nsl_manual_seed", &[seed_val]);
}

// topk(tensor, k) — dim defaults to -1
if func_name == "topk" {
    if args.len() < 2 || args.len() > 3 {
        return Err(CodegenError::new("topk() takes 2-3 arguments (tensor, k, dim=-1)"));
    }
    let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
    let k_val = self.compile_expr(builder, state, &args[1].value)?;
    let dim_val = if args.len() > 2 {
        self.compile_expr(builder, state, &args[2].value)?
    } else {
        builder.ins().iconst(cl_types::I64, -1)
    };
    return self.compile_call_by_name(builder, "nsl_tensor_topk", &[tensor_val, k_val, dim_val]);
}

// multinomial(tensor, num_samples)
if func_name == "multinomial" {
    if args.len() != 2 {
        return Err(CodegenError::new("multinomial() takes exactly 2 arguments (probs, num_samples)"));
    }
    let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
    let n_val = self.compile_expr(builder, state, &args[1].value)?;
    return self.compile_call_by_name(builder, "nsl_tensor_multinomial", &[tensor_val, n_val]);
}

// argmax(tensor, dim)
if func_name == "argmax" {
    if args.len() < 1 || args.len() > 2 {
        return Err(CodegenError::new("argmax() takes 1-2 arguments (tensor, dim=-1)"));
    }
    let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
    let dim_val = if args.len() > 1 {
        self.compile_expr(builder, state, &args[1].value)?
    } else {
        builder.ins().iconst(cl_types::I64, -1)
    };
    return self.compile_call_by_name(builder, "nsl_tensor_argmax", &[tensor_val, dim_val]);
}

// cumsum(tensor, dim)
if func_name == "cumsum" {
    if args.len() != 2 {
        return Err(CodegenError::new("cumsum() takes exactly 2 arguments (tensor, dim)"));
    }
    let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
    let dim_val = self.compile_expr(builder, state, &args[1].value)?;
    return self.compile_call_by_name(builder, "nsl_tensor_cumsum", &[tensor_val, dim_val]);
}

// lt_scalar(tensor, scalar)
if func_name == "lt_scalar" {
    if args.len() != 2 {
        return Err(CodegenError::new("lt_scalar() takes exactly 2 arguments (tensor, scalar)"));
    }
    let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
    let scalar_val = self.compile_expr(builder, state, &args[1].value)?;
    // Ensure scalar is f64
    let scalar_ty = builder.func.dfg.value_type(scalar_val);
    let scalar_f64 = if scalar_ty == cl_types::F64 {
        scalar_val
    } else {
        builder.ins().fcvt_from_sint(cl_types::F64, scalar_val)
    };
    return self.compile_call_by_name(builder, "nsl_tensor_lt_scalar", &[tensor_val, scalar_f64]);
}
```

- [ ] **Step 2: Add data source intrinsic handlers**

Continue in the same function:

```rust
// ── M19 Data Source Intrinsics ──────────────────────────────────

// load_jsonl("path.jsonl", "field_name")
if func_name == "load_jsonl" {
    if args.len() != 2 {
        return Err(CodegenError::new("load_jsonl() takes exactly 2 arguments (path, field)"));
    }
    let path_val = self.compile_expr(builder, state, &args[0].value)?;
    let path_str = match &args[0].value.kind {
        ExprKind::StringLiteral(s) => s.clone(),
        _ => return Err(CodegenError::new("load_jsonl(): first argument must be a string literal")),
    };
    let path_len = builder.ins().iconst(cl_types::I64, path_str.len() as i64);
    let field_val = self.compile_expr(builder, state, &args[1].value)?;
    let field_str = match &args[1].value.kind {
        ExprKind::StringLiteral(s) => s.clone(),
        _ => return Err(CodegenError::new("load_jsonl(): second argument must be a string literal")),
    };
    let field_len = builder.ins().iconst(cl_types::I64, field_str.len() as i64);
    return self.compile_call_by_name(builder, "nsl_load_jsonl", &[path_val, path_len, field_val, field_len]);
}

// load_csv("path.csv", col_idx)
if func_name == "load_csv" {
    if args.len() < 2 || args.len() > 3 {
        return Err(CodegenError::new("load_csv() takes 2-3 arguments (path, col_idx, has_header=1)"));
    }
    let path_val = self.compile_expr(builder, state, &args[0].value)?;
    let path_str = match &args[0].value.kind {
        ExprKind::StringLiteral(s) => s.clone(),
        _ => return Err(CodegenError::new("load_csv(): first argument must be a string literal")),
    };
    let path_len = builder.ins().iconst(cl_types::I64, path_str.len() as i64);
    let col_val = self.compile_expr(builder, state, &args[1].value)?;
    let header_val = if args.len() > 2 {
        self.compile_expr(builder, state, &args[2].value)?
    } else {
        builder.ins().iconst(cl_types::I64, 1) // has_header=true by default
    };
    return self.compile_call_by_name(builder, "nsl_load_csv", &[path_val, path_len, col_val, header_val]);
}

// load_mmap("path.bin", dtype)
if func_name == "load_mmap" {
    if args.len() != 2 {
        return Err(CodegenError::new("load_mmap() takes exactly 2 arguments (path, dtype)"));
    }
    let path_val = self.compile_expr(builder, state, &args[0].value)?;
    let path_str = match &args[0].value.kind {
        ExprKind::StringLiteral(s) => s.clone(),
        _ => return Err(CodegenError::new("load_mmap(): first argument must be a string literal")),
    };
    let path_len = builder.ins().iconst(cl_types::I64, path_str.len() as i64);
    let dtype_val = self.compile_expr(builder, state, &args[1].value)?;
    return self.compile_call_by_name(builder, "nsl_load_mmap", &[path_val, path_len, dtype_val]);
}
```

- [ ] **Step 3: Add DataLoader intrinsic handler**

```rust
// DataLoader(data, batch_size=32, seq_len=128, ...)
// Serializes keyword args to JSON config at compile time.
if func_name == "DataLoader" {
    if args.is_empty() {
        return Err(CodegenError::new("DataLoader() requires at least 1 argument (data)"));
    }
    let data_val = self.compile_expr(builder, state, &args[0].value)?;
    // Get data length via nsl_tensor_shape → first element, or use nsl_list_len
    let data_len = self.compile_call_by_name(builder, "nsl_list_len", &[data_val])
        .or_else(|_| {
            // Might be a tensor — get .len field
            // For simplicity, pass data_len as a separate computed value
            // Actually, we need the tensor's len. Use a runtime helper.
            Ok(builder.ins().iconst(cl_types::I64, 0))
        })?;
    // Actually: the data is always a tensor from load_mmap. Read its len.
    // We'll compute len in the runtime from the tensor pointer.
    // For now, pass the tensor_ptr and let the runtime extract .len from it.

    // Build JSON config from keyword args
    let mut config = serde_json::Map::new();
    for arg in args.iter().skip(1) {
        if let Some(name) = &arg.name {
            let key = self.resolve_sym(*name).to_string();
            // Extract compile-time constant values
            match &arg.value.kind {
                ExprKind::IntLiteral(v) => {
                    config.insert(key, serde_json::Value::Number(serde_json::Number::from(*v)));
                }
                ExprKind::BoolLiteral(v) => {
                    config.insert(key, serde_json::Value::Bool(*v));
                }
                ExprKind::FloatLiteral(v) => {
                    if let Some(n) = serde_json::Number::from_f64(*v) {
                        config.insert(key, serde_json::Value::Number(n));
                    }
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "DataLoader(): keyword argument '{}' must be a compile-time constant", key
                    )));
                }
            }
        }
    }
    let config_json = serde_json::Value::Object(config).to_string();
    let config_data_id = self.intern_string(&config_json)?;
    let config_gv = self.module.declare_data_in_func(config_data_id, builder.func);
    let config_ptr = builder.ins().symbol_value(cl_types::I64, config_gv);
    let config_len = builder.ins().iconst(cl_types::I64, config_json.len() as i64);

    // For tensor data: extract len from the NslTensor struct
    // NslTensor.len is at offset 32 (data:8 + shape:8 + strides:8 + ndim:8 = 32)
    let tensor_len = builder.ins().load(cl_types::I64, cranelift_codegen::ir::MemFlags::trusted(), data_val, cranelift_codegen::ir::immediates::Offset32::new(32));

    // Get data pointer from tensor (first field, offset 0)
    let tensor_data = builder.ins().load(cl_types::I64, cranelift_codegen::ir::MemFlags::trusted(), data_val, cranelift_codegen::ir::immediates::Offset32::new(0));

    let dl_ptr = self.compile_call_by_name(builder, "nsl_dataloader_create", &[tensor_data, tensor_len, config_ptr, config_len])?;
    self.compile_call_by_name(builder, "nsl_dataloader_start", &[dl_ptr])?;
    return Ok(dl_ptr);
}
```

- [ ] **Step 4: Add tensor method dispatch for `.cumsum(dim)`**

In `compile_tensor_method_call` (around line 2441), add a new match arm:

```rust
"cumsum" => {
    if args.len() != 1 {
        return Err(CodegenError::new("cumsum() takes exactly 1 argument (dim)"));
    }
    let dim_val = self.compile_expr(builder, state, &args[0].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_cumsum", &[obj_val, dim_val])
}
```

Also add `.gather(dim, indices)` as a method call if it's only a free function currently:

```rust
"gather" => {
    if args.len() != 2 {
        return Err(CodegenError::new("gather() takes exactly 2 arguments (dim, indices)"));
    }
    let dim_val = self.compile_expr(builder, state, &args[0].value)?;
    let indices_val = self.compile_expr(builder, state, &args[1].value)?;
    self.compile_call_by_name(builder, "nsl_tensor_gather", &[obj_val, dim_val, indices_val])
}
```

And `.shape(dim)` for getting a single dimension size:

```rust
"shape" => {
    if args.len() == 1 {
        // .shape(dim) → return single dim size as i64
        let dim_val = self.compile_expr(builder, state, &args[0].value)?;
        // Read ndim and shape ptr from tensor struct, compute index
        let shape_ptr = builder.ins().load(cl_types::I64, cranelift_codegen::ir::MemFlags::trusted(), obj_val, cranelift_codegen::ir::immediates::Offset32::new(8));
        let ndim = builder.ins().load(cl_types::I64, cranelift_codegen::ir::MemFlags::trusted(), obj_val, cranelift_codegen::ir::immediates::Offset32::new(24));
        // Handle negative dim: if dim < 0 then dim = ndim + dim
        let zero = builder.ins().iconst(cl_types::I64, 0);
        let is_neg = builder.ins().icmp(IntCC::SignedLessThan, dim_val, zero);
        let adjusted = builder.ins().iadd(ndim, dim_val);
        let actual_dim = builder.ins().select(is_neg, adjusted, dim_val);
        // Load shape[actual_dim]: shape_ptr + actual_dim * 8
        let eight = builder.ins().iconst(cl_types::I64, 8);
        let byte_offset = builder.ins().imul(actual_dim, eight);
        let elem_ptr = builder.ins().iadd(shape_ptr, byte_offset);
        let dim_size = builder.ins().load(cl_types::I64, cranelift_codegen::ir::MemFlags::trusted(), elem_ptr, cranelift_codegen::ir::immediates::Offset32::new(0));
        Ok(dim_size)
    } else {
        // .shape() → return NslList (existing behavior)
        self.compile_call_by_name(builder, "nsl_tensor_shape", &[obj_val])
    }
}
```

- [ ] **Step 5: Verify build**

Run: `cargo build -p nsl-codegen`
Expected: Compiles (may need `use serde_json;` import at top of expr.rs)

Note: If `serde_json` is not a dependency of `nsl-codegen`, add it to `crates/nsl-codegen/Cargo.toml`:
```toml
serde_json = "1"
```

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/expr.rs crates/nsl-codegen/Cargo.toml
git commit -m "feat(m19): implement codegen intrinsic handlers for sampling, data, and DataLoader"
```

---

### Task 12: Implement DataLoader Loop Codegen + Scope Teardown

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs`

The `for batch in loader:` loop needs special codegen when iterating over a DataLoader. Also, DataLoader variables need scope teardown (stop + free).

- [ ] **Step 1: Add DataLoader loop detection in compile_for**

In `crates/nsl-codegen/src/stmt.rs`, in the `compile_for` method (line ~593), add a check at the top (after the FixedModelArray check):

```rust
fn compile_for(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    pattern: &nsl_ast::pattern::Pattern,
    iterable: &nsl_ast::expr::Expr,
    body: &nsl_ast::stmt::Block,
) -> Result<(), CodegenError> {
    // Check if iterating over a fixed model array
    let iter_type = self.node_type(iterable.id).clone();
    if let Type::FixedModelArray { element_model, size } = &iter_type {
        return self.compile_for_model_array(builder, state, pattern, iterable, body, *element_model, *size);
    }

    // Check if iterating over a DataLoader (Type::Unknown from DataLoader() call)
    // Heuristic: if the iterable is a variable that was assigned from a DataLoader() call,
    // its type will be Unknown. We check if the runtime has dataloader functions.
    // More robust: check if variable name matches a known dataloader variable.
    // For now: if iter_type is Unknown, try the DataLoader loop pattern.
    if matches!(iter_type, Type::Unknown) {
        // Try DataLoader loop pattern
        return self.compile_for_dataloader(builder, state, pattern, iterable, body);
    }

    // ... rest of existing compile_for code (list iteration)
```

- [ ] **Step 2: Implement `compile_for_dataloader`**

Add new method to the Compiler impl:

```rust
/// Compile `for batch in loader:` where loader is a DataLoader.
/// Emits: loop { batch = next_batch(dl); if batch == 0: break; body; dict_free(batch); }
fn compile_for_dataloader(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    pattern: &nsl_ast::pattern::Pattern,
    iterable: &nsl_ast::expr::Expr,
    body: &nsl_ast::stmt::Block,
) -> Result<(), CodegenError> {
    let dl_val = self.compile_expr(builder, state, iterable)?;

    // Declare loop variable
    let loop_var_sym = match &pattern.kind {
        PatternKind::Ident(sym) => *sym,
        _ => return Err(CodegenError::new("DataLoader for-loop requires simple variable pattern")),
    };

    let batch_var = state.new_variable();
    builder.declare_var(batch_var, cl_types::I64);
    let zero = builder.ins().iconst(cl_types::I64, 0);
    builder.def_var(batch_var, zero);
    state.variables.insert(loop_var_sym, (batch_var, cl_types::I64));

    let header_block = builder.create_block();
    let body_block = builder.create_block();
    let cleanup_block = builder.create_block();
    let exit_block = builder.create_block();

    builder.ins().jump(header_block, &[]);

    // Header: batch = next_batch(dl); if batch == 0: exit
    builder.switch_to_block(header_block);
    state.current_block = Some(header_block);

    let batch_ptr = self.compile_call_by_name(builder, "nsl_dataloader_next_batch", &[dl_val])?;
    builder.def_var(batch_var, batch_ptr);

    let is_null = builder.ins().icmp(IntCC::Equal, batch_ptr, zero);
    builder.ins().brif(is_null, exit_block, &[], body_block, &[]);

    // Body
    builder.switch_to_block(body_block);
    builder.seal_block(body_block);
    state.current_block = Some(body_block);

    state.loop_stack.push(LoopContext { continue_block: cleanup_block, exit_block });
    for s in &body.stmts {
        self.compile_stmt(builder, state, s)?;
    }
    state.loop_stack.pop();

    let current = state.current_block.unwrap_or(body_block);
    if !is_block_filled(builder, current) {
        builder.ins().jump(cleanup_block, &[]);
    }

    // Cleanup: free the batch dict, then loop back
    builder.switch_to_block(cleanup_block);
    builder.seal_block(cleanup_block);
    state.current_block = Some(cleanup_block);

    let batch_to_free = builder.use_var(batch_var);
    self.compile_call_by_name(builder, "nsl_dict_free", &[batch_to_free])?;
    builder.ins().jump(header_block, &[]);

    // Exit
    builder.seal_block(header_block);
    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);
    state.current_block = Some(exit_block);

    // Reset DataLoader for next epoch
    self.compile_call_by_name(builder, "nsl_dataloader_reset", &[dl_val])?;

    Ok(())
}
```

- [ ] **Step 3: Add DataLoader scope teardown**

This is more complex — we need to track DataLoader variables and emit cleanup when the enclosing function returns. For the initial implementation, add a simple approach: track DataLoader variables in a list and emit cleanup before function return.

Add a field to FuncState (or Compiler):

```rust
// In the FuncState or wherever temporary tracking happens:
pub dataloader_vars: Vec<Value>,
```

In the DataLoader intrinsic handler (Task 11), after creating the DataLoader, push it:

```rust
state.dataloader_vars.push(dl_ptr);
```

In the function epilogue/return compilation, emit cleanup for all DataLoader vars:

```rust
// Before function return:
for dl in &state.dataloader_vars {
    self.compile_call_by_name(builder, "nsl_dataloader_stop", &[*dl])?;
    self.compile_call_by_name(builder, "nsl_dataloader_free", &[*dl])?;
}
```

**Note:** If FuncState doesn't have this field, add `pub dataloader_vars: Vec<cranelift_codegen::ir::Value>` to the FuncState struct and initialize it as `Vec::new()` in the constructor.

- [ ] **Step 4: Verify build**

Run: `cargo build -p nsl-codegen`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(m19): implement DataLoader loop codegen with dict cleanup and scope teardown"
```

---

## Chunk 4: Stdlib Modules + Integration Tests

### Task 13: Create stdlib Data Loader Documentation Module

**Files:**
- Create: `stdlib/nsl/data/loader.nsl`

- [ ] **Step 1: Create directory and module**

```bash
mkdir -p stdlib/nsl/data
mkdir -p stdlib/nsl/inference
```

- [ ] **Step 2: Create `stdlib/nsl/data/loader.nsl`**

```nsl
# NeuralScript Data Pipeline Module (nsl.data)
#
# Built-in intrinsics (available without importing this module):
#
#   load_jsonl(path, field) -> List[str]
#       Load a JSONL file, extract a named field from each line.
#       Small-data utility for eval sets and prompt lists.
#
#   load_csv(path, col_idx, has_header=1) -> List[str]
#       Load a CSV file, extract a column by index.
#       Small-data utility for eval sets.
#
#   load_mmap(path, dtype) -> Tensor
#       Memory-map a binary file as a flat 1D tensor.
#       dtype: 0=f64, 1=f32, 2=i32, 3=u16
#       u16 (dtype=3) is the primary pathway for pre-tokenized LLM datasets.
#       Zero-copy: data points directly into the mapped file.
#
#   DataLoader(data, batch_size=32, seq_len=128, shuffle=false,
#              num_workers=1, prefetch=2, packing=false,
#              pack_separator=0, drop_last=true) -> DataLoader
#       Create a multi-threaded data loader with batching.
#       Iterate with: for batch in loader:
#       Each batch is a Dict with keys:
#           "input_ids"      [B, S] token ids
#           "labels"          [B, S] shifted labels (-100 at boundaries)
#           "attention_mask"  [B, S, S] causal/block-diagonal mask

fn data_version() -> int:
    return 1
```

- [ ] **Step 3: Commit**

```bash
git add stdlib/nsl/data/loader.nsl
git commit -m "feat(m19): add stdlib data loader documentation module"
```

---

### Task 14: Create stdlib Inference Sampling Module

**Files:**
- Create: `stdlib/nsl/inference/sampling.nsl`

- [ ] **Step 1: Create `stdlib/nsl/inference/sampling.nsl`**

```nsl
# NeuralScript Inference Sampling Module
# Provides top-k, top-p (nucleus), and greedy sampling strategies.

fn sample_top_k(logits: Tensor, k: int, temperature: float) -> Tensor:
    let scaled = logits / temperature
    let result = topk(scaled, k)
    let probs = softmax(result["values"], dim=-1)
    let sampled = multinomial(probs, 1)
    return result["indices"].gather(-1, sampled)

fn sample_top_p(logits: Tensor, p: float, temperature: float) -> Tensor:
    let scaled = logits / temperature
    let probs = softmax(scaled, dim=-1)
    let n = probs.shape(-1)
    let result = topk(probs, n)
    let cumulative = result["values"].cumsum(dim=-1)
    let shifted = cumulative - result["values"]
    let mask = lt_scalar(shifted, p)
    let filtered = result["values"] * mask
    let sampled = multinomial(filtered, 1)
    return result["indices"].gather(-1, sampled)

fn sample_greedy(logits: Tensor) -> Tensor:
    return argmax(logits, dim=-1)
```

- [ ] **Step 2: Commit**

```bash
git add stdlib/nsl/inference/sampling.nsl
git commit -m "feat(m19): add stdlib inference sampling module (top-k, top-p, greedy)"
```

---

### Task 15: Create stdlib Generate Module

**Files:**
- Create: `stdlib/nsl/inference/generate.nsl`

- [ ] **Step 1: Create `stdlib/nsl/inference/generate.nsl`**

```nsl
# NeuralScript Text Generation Module
# Autoregressive generation with pre-allocated buffer and no_grad.

from nsl.inference.sampling import sample_top_k

fn generate(model, prompt: Tensor, max_tokens: int, temperature: float, top_k: int) -> Tensor:
    let prompt_len = prompt.shape(-1)
    let total_len = prompt_len + max_tokens
    let tokens = zeros([1, total_len])
    # Copy prompt into pre-allocated buffer
    # Note: This uses a loop since slice assignment may not be available
    for i in range(prompt_len):
        let tok = prompt.select(1, i)
        # tokens[0, i] = tok — use set_element when codegen supports it
    no_grad:
        for i in range(prompt_len, total_len):
            let logits = model.forward(tokens)
            let next_logit = logits.select(-2, -1)
            let next_token = sample_top_k(next_logit, top_k, temperature)
            # tokens[0, i] = next_token
    return tokens
```

**Note:** The exact syntax for `tokens[0, i] = next_token` (multi-dim slice assignment) depends on whether Task 6's codegen is wired up. If not, use a runtime helper or fall back to a simpler pattern. The implementer should adapt based on what works.

- [ ] **Step 2: Commit**

```bash
git add stdlib/nsl/inference/generate.nsl
git commit -m "feat(m19): add stdlib generate module with no_grad autoregressive loop"
```

---

### Task 16: Sampling Integration Test (NSL)

**Files:**
- Create: `tests/m19_topk_test.nsl`

- [ ] **Step 1: Write sampling integration test**

Create `tests/m19_topk_test.nsl`:

```nsl
# M19 Integration Test: Sampling Primitives

fn main():
    # Test manual_seed for reproducibility
    manual_seed(42)

    # Create a small "logits" tensor
    let logits = randn([1, 8])
    print("Logits created")

    # Test topk
    let result = topk(logits, 3)
    let top_vals = result["values"]
    let top_idx = result["indices"]
    print("topk: PASS")

    # Test softmax + multinomial
    let probs = softmax(top_vals, dim=-1)
    let sampled = multinomial(probs, 1)
    print("multinomial: PASS")

    # Test gather (reconstruct selected token)
    let selected = top_idx.gather(-1, sampled)
    print("gather: PASS")

    # Test argmax
    let greedy = argmax(logits, dim=-1)
    print("argmax: PASS")

    # Test deterministic RNG
    manual_seed(42)
    let a = multinomial(probs, 1)
    manual_seed(42)
    let b = multinomial(probs, 1)
    # a and b should be identical
    print("deterministic RNG: PASS")

    # Test cumsum
    let vals = randn([4])
    let cs = cumsum(vals, 0)
    print("cumsum: PASS")

    # Test lt_scalar
    let mask = lt_scalar(probs, 0.5)
    print("lt_scalar: PASS")

    print("All sampling tests: PASS")
```

- [ ] **Step 2: Run test**

Run: `cargo run -- run tests/m19_topk_test.nsl`
Expected: Prints "All sampling tests: PASS"

- [ ] **Step 3: Commit**

```bash
git add tests/m19_topk_test.nsl
git commit -m "test(m19): add sampling primitives integration test"
```

---

### Task 17: Data Source Integration Test (NSL)

**Files:**
- Create: `tests/m19_data_test.nsl`
- Create: `tests/fixtures/test_data.jsonl` (test fixture)
- Create: `tests/fixtures/test_data.csv` (test fixture)

- [ ] **Step 1: Create test fixture files**

Create `tests/fixtures/test_data.jsonl`:
```json
{"text": "hello world", "label": 1}
{"text": "foo bar", "label": 2}
{"text": "neural script", "label": 3}
```

Create `tests/fixtures/test_data.csv`:
```csv
name,value
alpha,1.0
beta,2.0
gamma,3.0
```

- [ ] **Step 2: Write data source test**

Create `tests/m19_data_test.nsl`:

```nsl
# M19 Integration Test: Data Sources

fn main():
    # Test JSONL loading
    let texts = load_jsonl("tests/fixtures/test_data.jsonl", "text")
    print("JSONL loaded, count:")
    print(len(texts))

    # Test CSV loading
    let names = load_csv("tests/fixtures/test_data.csv", 0)
    print("CSV loaded, count:")
    print(len(names))

    print("Data source tests: PASS")
```

- [ ] **Step 3: Run test**

Run: `cargo run -- run tests/m19_data_test.nsl`
Expected: Prints counts and "Data source tests: PASS"

- [ ] **Step 4: Commit**

```bash
git add tests/m19_data_test.nsl tests/fixtures/
git commit -m "test(m19): add data source integration test with JSONL/CSV fixtures"
```

---

### Task 18: DataLoader Integration Test (NSL)

**Files:**
- Create: `tests/m19_dataloader_test.nsl`
- Create: `tests/fixtures/test_tokens.bin` (binary test fixture, generated by test setup)

- [ ] **Step 1: Write a small binary fixture generator**

The test needs a binary file of f64 tokens. Create a small helper or generate inline:

Create `tests/m19_dataloader_test.nsl`:

```nsl
# M19 Integration Test: DataLoader
# Tests basic DataLoader batching with a small mmap dataset.

fn main():
    # First, create a small binary test file with token data
    # For now, use randn to create data and test the DataLoader
    # with synthetic data (since we can't write binary files from NSL easily)

    # Create synthetic token data as a tensor
    let data = randn([64])  # 64 tokens

    # Test DataLoader with non-packing mode
    let loader = DataLoader(data, batch_size=2, seq_len=8, num_workers=1, packing=false)

    let batch_count = 0
    for batch in loader:
        let ids = batch["input_ids"]
        let labels = batch["labels"]
        let mask = batch["attention_mask"]
        batch_count = batch_count + 1

    print("Batches processed:")
    print(batch_count)
    print("DataLoader test: PASS")
```

**Note:** The implementer should verify that `randn` returns a tensor with a valid `.data` pointer that the DataLoader can read. If the DataLoader requires an mmap tensor specifically, create a binary fixture file first and use `load_mmap`. Adapt as needed.

- [ ] **Step 2: Run test**

Run: `cargo run -- run tests/m19_dataloader_test.nsl`
Expected: Prints batch count and "DataLoader test: PASS"

- [ ] **Step 3: Commit**

```bash
git add tests/m19_dataloader_test.nsl
git commit -m "test(m19): add DataLoader integration test"
```

---

### Task 19: End-to-End Test

**Files:**
- Create: `examples/m19_data_pipeline.nsl`

The deliverable: load data, create a model, iterate batches, generate tokens.

- [ ] **Step 1: Write E2E example**

Create `examples/m19_data_pipeline.nsl`:

```nsl
# M19 End-to-End: Data Pipeline + Inference
# Demonstrates the complete ML workflow:
# 1. Create synthetic data
# 2. Build a simple model
# 3. Iterate batches with DataLoader
# 4. Run inference with sampling

from nsl.nn.layers import Linear

model TinyLM:
    embed: Linear = Linear(64, 32)
    head: Linear = Linear(32, 64)

    fn forward(self, x: Tensor) -> Tensor:
        let h = relu(self.embed.forward(x))
        return self.head.forward(h)

fn main():
    manual_seed(42)
    let m = TinyLM()
    print("Model created")

    # Create synthetic token data
    let data = randn([256])

    # DataLoader: 4 batches of size 2, seq_len 32
    let loader = DataLoader(data, batch_size=2, seq_len=32, num_workers=1)

    let total_batches = 0
    for batch in loader:
        let ids = batch["input_ids"]
        total_batches = total_batches + 1

    print("Training batches:")
    print(total_batches)

    # Inference: topk sampling
    let dummy = randn([1, 64])
    let logits = m.forward(dummy)
    let result = topk(logits, 5)
    let probs = softmax(result["values"], dim=-1)
    let sampled = multinomial(probs, 1)
    let token = result["indices"].gather(-1, sampled)
    print("Sampled token index:")
    print(token)

    print("M19 E2E: PASS")
```

- [ ] **Step 2: Run E2E test**

Run: `cargo run -- run examples/m19_data_pipeline.nsl`
Expected: Prints "M19 E2E: PASS"

- [ ] **Step 3: Commit**

```bash
git add examples/m19_data_pipeline.nsl
git commit -m "feat(m19): add end-to-end data pipeline + inference example"
```

---

### Task 20: Update README + Full Regression

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run full test suite**

Run: `cargo test`
Expected: All existing + new tests pass

Run: `cargo run -- run tests/m19_topk_test.nsl`
Run: `cargo run -- run tests/m19_data_test.nsl`
Run: `cargo run -- run tests/m19_dataloader_test.nsl`
Run: `cargo run -- run examples/m19_data_pipeline.nsl`
Expected: All print PASS

- [ ] **Step 2: Run existing milestone tests to verify no regressions**

Run the key tests from previous milestones:

```bash
cargo run -- run tests/m18b_safetensors.nsl
cargo run -- run tests/m18_transformer_block_test.nsl
cargo run -- run tests/m18_attention_test.nsl
```

Expected: All pass

- [ ] **Step 3: Update README.md roadmap**

In `README.md`, find the M19 row and update:
- Status: `Planned` → `Complete`
- Description: `Data pipeline + inference sampling (JSONL, CSV, mmap, DataLoader, topk, multinomial, generate)`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: mark M19 data pipeline + inference sampling as complete"
```

---

## Implementation Notes for Subagents

### Key Patterns to Follow
1. **Runtime FFI:** All functions are `#[no_mangle] pub extern "C" fn` with i64/f64 params
2. **Tensor creation:** Use `crate::cpu::create_tensor_with_shape_rs(&[dims])` to create f64 tensors
3. **String creation:** Use `crate::string::nsl_str_from_rust("name")` for dict keys
4. **Intrinsic codegen:** Match on `func_name`, extract args, call `compile_call_by_name`
5. **String literal extraction:** Match `ExprKind::StringLiteral(s)`, compute `.len()` at compile time
6. **Builtin registration:** `def("name", Type::Function { params: vec![Type::Unknown], ret: ... })`

### Critical Safety Checks
1. **`owns_data` must be set on ALL tensor construction sites** — miss one and mmap tensors segfault
2. **`nsl_dict_free` in loop body** — miss it and DataLoader OOMs after ~100 batches
3. **DataLoader scope teardown** — miss it and worker threads leak as zombies
4. **`no_grad` in generate()** — miss it and autodiff tape OOMs during inference
5. **CDF self-normalization in multinomial** — miss it and unnormalized probs crash

### Build Order Dependencies
```
Task 1 (owns_data) ← Task 5 (mmap uses it)
Task 2 (dict_free) ← Task 12 (loop cleanup uses it)
Task 3 (cargo deps) ← Tasks 4, 5 (use rand, serde_json, memmap2)
Task 4 (sampling) ← Task 11 (codegen handlers)
Task 5 (data sources) ← Task 11 (codegen handlers)
Task 6 (slice assign) ← Task 15 (generate uses it)
Task 7 (packing) ← Task 8 (DataLoader uses it)
Task 8 (DataLoader) ← Task 12 (loop codegen)
Task 9 (codegen builtins) ← Task 11 (intrinsic handlers)
Task 10 (semantic builtins) ← Task 16 (tests need type checking)
Tasks 11-12 (codegen) ← Tasks 16-19 (all NSL tests)
Tasks 13-15 (stdlib) ← Task 19 (E2E uses them)
```
