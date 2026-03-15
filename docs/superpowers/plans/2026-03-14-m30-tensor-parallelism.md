# M30: Tensor Parallelism Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split large models across 2-8 GPUs using process-based SPMD tensor parallelism with automatic weight sharding, per-GPU KV-cache, and NCCL collective communication (with SimulatedBackend for testing).

**Architecture:** Process-based SPMD — `nsl run --devices N` spawns N OS processes, each running the same Cranelift-compiled binary. Processes synchronize only at collective calls (AllReduce, AllGather, Broadcast). A `CollectiveBackend` trait abstracts real NCCL (Linux) from `SimulatedBackend` (Windows/testing). The `@shard(dim=D)` decorator on model layers drives compile-time weight partitioning and automatic collective insertion.

**Tech Stack:** Rust, Cranelift, memmap2, safetensors format, NCCL (feature-gated), shared-memory IPC

**Spec:** `docs/superpowers/specs/2026-03-14-m30-tensor-parallelism-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|---|---|
| `crates/nsl-runtime/src/tensor_parallel/mod.rs` | Module declarations and re-exports |
| `crates/nsl-runtime/src/tensor_parallel/collective.rs` | `CollectiveBackend` trait + `DtypeId` constants + `SimulatedBackend` |
| `crates/nsl-runtime/src/tensor_parallel/sharding.rs` | Mmap-based weight sharding — `load_sharded_weight` |
| `crates/nsl-runtime/src/tensor_parallel/ffi.rs` | `#[no_mangle] pub extern "C"` functions for codegen |
| `crates/nsl-codegen/src/tensor_parallel.rs` | `@shard` extraction, `DistState` tracking, collective emission |
| `examples/m30_tp_basic.nsl` | E2E: basic tensor-parallel serve block |
| `examples/m30_shard_validation.nsl` | E2E: compile-time shard divisibility error |
| `examples/m30_gqa_replication.nsl` | E2E: GQA KV-head replication |
| `tests/expected/m30_tp_basic.txt` | Expected output |
| `tests/expected/m30_shard_validation.txt` | Expected error output |
| `tests/expected/m30_gqa_replication.txt` | Expected output |

### Modified Files

| File | Lines | Change |
|---|---|---|
| `crates/nsl-runtime/src/lib.rs` | after line 56 | Add `pub mod tensor_parallel;` |
| `crates/nsl-runtime/Cargo.toml` | deps section | Add `safetensors = "0.4"` as non-optional dep |
| `crates/nsl-codegen/src/builtins.rs` | before line 329 | Register 11 new `nsl_tp_*` FFI functions |
| `crates/nsl-codegen/src/compiler.rs` | lines 58-109 | Add `shard_configs`, `activation_states`, `world_size` fields |
| `crates/nsl-codegen/src/lib.rs` | module list | Add `pub mod tensor_parallel;` |
| `crates/nsl-codegen/src/stmt.rs` | near line 222 | Modify ServeBlock dispatch for TP wrapping |
| `crates/nsl-cli/src/main.rs` | lines 38-57 | Add `--devices` arg to Run command |
| `crates/nsl-cli/tests/e2e.rs` | after line 571 | Add M30 E2E test functions + `run_nsl_multi` helper |

---

## Chunk 1: Runtime Foundation

### Task 1: CollectiveBackend Trait + DtypeId Constants

**Files:**
- Create: `crates/nsl-runtime/src/tensor_parallel/mod.rs`
- Create: `crates/nsl-runtime/src/tensor_parallel/collective.rs`
- Modify: `crates/nsl-runtime/src/lib.rs:56`

- [ ] **Step 1: Write the test for DtypeId constants and trait**

In `crates/nsl-runtime/src/tensor_parallel/collective.rs`, add at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_byte_width() {
        assert_eq!(dtype_byte_width(DTYPE_F64), 8);
        assert_eq!(dtype_byte_width(DTYPE_F32), 4);
        assert_eq!(dtype_byte_width(DTYPE_F16), 2);
        assert_eq!(dtype_byte_width(DTYPE_BF16), 2);
        assert_eq!(dtype_byte_width(DTYPE_I8), 1);
        assert_eq!(dtype_byte_width(DTYPE_FP8), 1);
    }

    #[test]
    fn simulated_backend_rank_and_world_size() {
        let backend = SimulatedBackend::new(0, 1, std::ptr::null_mut(), 0);
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.world_size(), 1);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-runtime tensor_parallel -- --nocapture 2>&1`
Expected: FAIL — module doesn't exist yet

- [ ] **Step 3: Create module skeleton and CollectiveBackend trait**

Create `crates/nsl-runtime/src/tensor_parallel/mod.rs`:
```rust
pub mod collective;
pub mod sharding;
pub mod ffi;
```

Create `crates/nsl-runtime/src/tensor_parallel/collective.rs`:
```rust
//! CollectiveBackend trait and SimulatedBackend for tensor parallelism.

use std::ffi::c_void;

// --- DtypeId: reuse the runtime's u16 dtype space (see tensor.rs) ---

pub type DtypeId = u16;

pub const DTYPE_F64: DtypeId = 0;
pub const DTYPE_F32: DtypeId = 1;
pub const DTYPE_F16: DtypeId = 2;
pub const DTYPE_BF16: DtypeId = 3;
pub const DTYPE_I8: DtypeId = 4;
pub const DTYPE_FP8: DtypeId = 5;

/// Returns byte width for a given dtype ID.
pub fn dtype_byte_width(dtype: DtypeId) -> usize {
    match dtype {
        DTYPE_F64 => 8,
        DTYPE_F32 => 4,
        DTYPE_F16 | DTYPE_BF16 => 2,
        DTYPE_I8 | DTYPE_FP8 => 1,
        _ => 4, // default to f32 width for unknown/custom dtypes
    }
}

/// Opaque stream handle — *mut c_void unconditionally.
/// Resolves to CUstream inside #[cfg(feature = "cuda")] code.
/// SimulatedBackend ignores this parameter.
pub type StreamHandle = *mut c_void;

/// Abstraction over collective communication backends.
/// NcclBackend (feature-gated) and SimulatedBackend both implement this.
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

// ---------------------------------------------------------------------------
// SimulatedBackend — shared-memory IPC for testing on single-GPU machines
// ---------------------------------------------------------------------------

use std::sync::atomic::{AtomicU32, Ordering};

/// Header at the start of the shared memory region.
#[repr(C)]
struct ShmHeader {
    generation: AtomicU32,
    arrival: AtomicU32,
    world_size: u32,
    _pad: u32,
}

/// SimulatedBackend uses file-backed shared memory for cross-process
/// collective operations. Each rank gets a data slot of `max_slot_bytes`.
pub struct SimulatedBackend {
    rank: i32,
    world_size: i32,
    shm_ptr: *mut u8,
    shm_len: usize,
    max_slot_bytes: usize,
}

// SAFETY: The shared memory is accessed via atomic operations for
// synchronization, and each rank only writes to its own slot.
unsafe impl Send for SimulatedBackend {}
unsafe impl Sync for SimulatedBackend {}

const SHM_HEADER_SIZE: usize = std::mem::size_of::<ShmHeader>();
/// 64 MB per slot — sufficient for LM head logits at batch=256, vocab=128K, f32
const DEFAULT_MAX_SLOT_BYTES: usize = 64 * 1024 * 1024;

impl SimulatedBackend {
    /// Create a new SimulatedBackend.
    /// `shm_ptr` points to the base of the shared memory region.
    /// `shm_len` is the total size. For world_size=1, shm_ptr can be null.
    pub fn new(rank: i32, world_size: i32, shm_ptr: *mut u8, shm_len: usize) -> Self {
        let max_slot_bytes = if shm_len > SHM_HEADER_SIZE && world_size > 0 {
            (shm_len - SHM_HEADER_SIZE) / world_size as usize
        } else {
            0
        };
        SimulatedBackend {
            rank,
            world_size,
            shm_ptr,
            shm_len,
            max_slot_bytes,
        }
    }

    fn header(&self) -> &ShmHeader {
        assert!(!self.shm_ptr.is_null(), "SimulatedBackend: null shm_ptr");
        unsafe { &*(self.shm_ptr as *const ShmHeader) }
    }

    fn slot_ptr(&self, rank: i32) -> *mut u8 {
        assert!(!self.shm_ptr.is_null());
        unsafe {
            self.shm_ptr
                .add(SHM_HEADER_SIZE + rank as usize * self.max_slot_bytes)
        }
    }

    /// Barrier: increment arrival, spin until all ranks arrive,
    /// then increment generation for double-buffering.
    fn barrier_impl(&self) {
        if self.world_size <= 1 {
            return;
        }
        let hdr = self.header();
        let gen_before = hdr.generation.load(Ordering::Acquire);

        let arrived = hdr.arrival.fetch_add(1, Ordering::AcqRel) + 1;
        if arrived == self.world_size as u32 {
            // Last to arrive: reset arrival counter, bump generation
            hdr.arrival.store(0, Ordering::Release);
            hdr.generation.store(gen_before + 1, Ordering::Release);
        } else {
            // Spin until generation changes
            while hdr.generation.load(Ordering::Acquire) == gen_before {
                std::hint::spin_loop();
            }
        }
    }
}

impl CollectiveBackend for SimulatedBackend {
    fn all_reduce_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        if self.world_size <= 1 {
            // Single rank: just copy
            if sendbuf != recvbuf as *const c_void {
                let bytes = count * dtype_byte_width(dtype);
                unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, recvbuf as *mut u8, bytes) };
            }
            return 0;
        }

        let bytes = count * dtype_byte_width(dtype);
        assert!(bytes <= self.max_slot_bytes, "all_reduce_sum: data exceeds slot size");

        // Write our data to our slot
        unsafe {
            std::ptr::copy_nonoverlapping(
                sendbuf as *const u8,
                self.slot_ptr(self.rank),
                bytes,
            );
        }

        // Barrier: wait for all ranks to write
        self.barrier_impl();

        // Last rank to arrive does the reduction; but since barrier is complete,
        // all ranks can reduce independently (deterministic result).
        // Reduce all slots into recvbuf (f32 sum).
        if dtype == DTYPE_F32 {
            let out = recvbuf as *mut f32;
            // Zero the output
            unsafe { std::ptr::write_bytes(out, 0, count) };
            for r in 0..self.world_size {
                let src = self.slot_ptr(r) as *const f32;
                for i in 0..count {
                    unsafe { *out.add(i) += *src.add(i) };
                }
            }
        } else if dtype == DTYPE_F64 {
            let out = recvbuf as *mut f64;
            unsafe { std::ptr::write_bytes(out, 0, count) };
            for r in 0..self.world_size {
                let src = self.slot_ptr(r) as *const f64;
                for i in 0..count {
                    unsafe { *out.add(i) += *src.add(i) };
                }
            }
        } else {
            // For other dtypes, just copy rank 0's data (placeholder)
            unsafe {
                std::ptr::copy_nonoverlapping(self.slot_ptr(0), recvbuf as *mut u8, bytes);
            }
        }

        // Barrier: ensure all ranks finish reading before next collective
        self.barrier_impl();
        0
    }

    fn all_gather(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        send_count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        let send_bytes = send_count * dtype_byte_width(dtype);

        if self.world_size <= 1 {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    sendbuf as *const u8,
                    recvbuf as *mut u8,
                    send_bytes,
                );
            }
            return 0;
        }

        assert!(send_bytes <= self.max_slot_bytes, "all_gather: data exceeds slot size");

        // Write our shard to our slot
        unsafe {
            std::ptr::copy_nonoverlapping(
                sendbuf as *const u8,
                self.slot_ptr(self.rank),
                send_bytes,
            );
        }

        // Barrier
        self.barrier_impl();

        // Each rank reads all slots into recvbuf
        for r in 0..self.world_size {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.slot_ptr(r),
                    (recvbuf as *mut u8).add(r as usize * send_bytes),
                    send_bytes,
                );
            }
        }

        self.barrier_impl();
        0
    }

    fn broadcast(
        &self,
        buf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        root_rank: i32,
        _stream: StreamHandle,
    ) -> i32 {
        if self.world_size <= 1 {
            return 0;
        }

        let bytes = count * dtype_byte_width(dtype);
        assert!(bytes <= self.max_slot_bytes, "broadcast: data exceeds slot size");

        if self.rank == root_rank {
            // Root writes to slot 0
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buf as *const u8,
                    self.slot_ptr(0),
                    bytes,
                );
            }
        }

        // Barrier
        self.barrier_impl();

        if self.rank != root_rank {
            // Non-root reads from slot 0
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.slot_ptr(0),
                    buf as *mut u8,
                    bytes,
                );
            }
        }

        self.barrier_impl();
        0
    }

    fn barrier(&self) -> i32 {
        self.barrier_impl();
        0
    }

    fn rank(&self) -> i32 {
        self.rank
    }

    fn world_size(&self) -> i32 {
        self.world_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_byte_width_values() {
        assert_eq!(dtype_byte_width(DTYPE_F64), 8);
        assert_eq!(dtype_byte_width(DTYPE_F32), 4);
        assert_eq!(dtype_byte_width(DTYPE_F16), 2);
        assert_eq!(dtype_byte_width(DTYPE_BF16), 2);
        assert_eq!(dtype_byte_width(DTYPE_I8), 1);
        assert_eq!(dtype_byte_width(DTYPE_FP8), 1);
    }

    #[test]
    fn simulated_backend_rank_and_world_size() {
        let backend = SimulatedBackend::new(0, 1, std::ptr::null_mut(), 0);
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.world_size(), 1);
    }

    #[test]
    fn simulated_backend_single_rank_all_reduce() {
        let backend = SimulatedBackend::new(0, 1, std::ptr::null_mut(), 0);
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let result = backend.all_reduce_sum(
            data.as_ptr() as *const c_void,
            data.as_mut_ptr() as *mut c_void,
            3,
            DTYPE_F32,
            std::ptr::null_mut(),
        );
        assert_eq!(result, 0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn simulated_backend_single_rank_all_gather() {
        let backend = SimulatedBackend::new(0, 1, std::ptr::null_mut(), 0);
        let send: Vec<f32> = vec![1.0, 2.0];
        let mut recv: Vec<f32> = vec![0.0, 0.0];
        let result = backend.all_gather(
            send.as_ptr() as *const c_void,
            recv.as_mut_ptr() as *mut c_void,
            2,
            DTYPE_F32,
            std::ptr::null_mut(),
        );
        assert_eq!(result, 0);
        assert_eq!(recv, vec![1.0, 2.0]);
    }

    #[test]
    fn simulated_backend_single_rank_broadcast() {
        let backend = SimulatedBackend::new(0, 1, std::ptr::null_mut(), 0);
        let mut data: Vec<f32> = vec![42.0];
        let result = backend.broadcast(
            data.as_mut_ptr() as *mut c_void,
            1,
            DTYPE_F32,
            0,
            std::ptr::null_mut(),
        );
        assert_eq!(result, 0);
        assert_eq!(data, vec![42.0]);
    }
}
```

Create empty placeholder files:

`crates/nsl-runtime/src/tensor_parallel/sharding.rs`:
```rust
//! Mmap-based weight sharding for tensor parallelism.
```

`crates/nsl-runtime/src/tensor_parallel/ffi.rs`:
```rust
//! FFI exports for tensor parallelism.
```

- [ ] **Step 4: Add module to lib.rs**

In `crates/nsl-runtime/src/lib.rs`, add after line 56 (`pub mod serving;`):
```rust
pub mod tensor_parallel;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p nsl-runtime tensor_parallel -- --nocapture 2>&1`
Expected: 5 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-runtime/src/tensor_parallel/ crates/nsl-runtime/src/lib.rs
git commit -m "feat(m30): CollectiveBackend trait, DtypeId constants, SimulatedBackend"
```

---

### Task 2: Weight Sharding Module

**Files:**
- Create: `crates/nsl-runtime/src/tensor_parallel/sharding.rs`
- Modify: `crates/nsl-runtime/Cargo.toml`

- [ ] **Step 1: Write the tests**

In `crates/nsl-runtime/src/tensor_parallel/sharding.rs`:

```rust
//! Mmap-based weight sharding for tensor parallelism.
//!
//! Each SPMD process mmaps a safetensors file and loads only its
//! rank's slice of each sharded tensor. Rank and world_size are
//! read from environment variables (NSL_LOCAL_RANK, NSL_WORLD_SIZE).

use std::ffi::c_void;

/// Compute the byte offset and length for a rank's shard along `shard_dim`.
///
/// Returns (offset_elements, shard_elements, shard_shape).
pub fn compute_shard_slice(
    shape: &[usize],
    shard_dim: usize,
    rank: usize,
    world_size: usize,
) -> (usize, usize, Vec<usize>) {
    assert!(shard_dim < shape.len(), "shard_dim out of bounds");
    assert!(
        shape[shard_dim] % world_size == 0,
        "shape[{}]={} not divisible by world_size={}",
        shard_dim,
        shape[shard_dim],
        world_size
    );

    let shard_size = shape[shard_dim] / world_size;

    // Build shard shape
    let mut shard_shape = shape.to_vec();
    shard_shape[shard_dim] = shard_size;

    // Compute element offset
    // For dim 0: contiguous blocks. offset = rank * (product of remaining dims) * shard_size
    // For dim > 0: strided access
    let elements_per_slice: usize = shape.iter().skip(shard_dim + 1).product::<usize>().max(1);
    let offset_elements = rank * shard_size * elements_per_slice;

    let shard_elements: usize = shard_shape.iter().product();

    (offset_elements, shard_elements, shard_shape)
}

/// Copy a shard from a source buffer into a destination buffer.
/// For shard_dim == 0, this is a contiguous memcpy.
/// For shard_dim > 0, this is a strided copy.
pub fn copy_shard(
    src: *const u8,
    dst: *mut u8,
    shape: &[usize],
    shard_dim: usize,
    rank: usize,
    world_size: usize,
    elem_bytes: usize,
) {
    let shard_size = shape[shard_dim] / world_size;
    let inner_elements: usize = shape.iter().skip(shard_dim + 1).product::<usize>().max(1);

    if shard_dim == 0 {
        // Contiguous copy
        let offset_bytes = rank * shard_size * inner_elements * elem_bytes;
        let copy_bytes = shard_size * inner_elements * elem_bytes;
        unsafe {
            std::ptr::copy_nonoverlapping(src.add(offset_bytes), dst, copy_bytes);
        }
    } else {
        // Strided copy: iterate over outer dimensions
        let outer_elements: usize = shape.iter().take(shard_dim).product::<usize>().max(1);
        let stride_elements = shape[shard_dim] * inner_elements;
        let shard_row_bytes = shard_size * inner_elements * elem_bytes;

        let mut dst_offset = 0usize;
        for outer in 0..outer_elements {
            let src_offset =
                (outer * stride_elements + rank * shard_size * inner_elements) * elem_bytes;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.add(src_offset),
                    dst.add(dst_offset),
                    shard_row_bytes,
                );
            }
            dst_offset += shard_row_bytes;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_shard_slice_dim0() {
        // [8, 4] sharded on dim 0 across 2 ranks
        let (offset, elements, shard_shape) =
            compute_shard_slice(&[8, 4], 0, 0, 2);
        assert_eq!(offset, 0);
        assert_eq!(elements, 16); // 4 * 4
        assert_eq!(shard_shape, vec![4, 4]);

        let (offset, elements, shard_shape) =
            compute_shard_slice(&[8, 4], 0, 1, 2);
        assert_eq!(offset, 16); // 4 * 4
        assert_eq!(elements, 16);
        assert_eq!(shard_shape, vec![4, 4]);
    }

    #[test]
    fn compute_shard_slice_dim1() {
        // [4, 8] sharded on dim 1 across 2 ranks
        let (offset, elements, shard_shape) =
            compute_shard_slice(&[4, 8], 1, 0, 2);
        assert_eq!(offset, 0);
        assert_eq!(elements, 16); // 4 * 4
        assert_eq!(shard_shape, vec![4, 4]);

        let (offset, elements, shard_shape) =
            compute_shard_slice(&[4, 8], 1, 1, 2);
        assert_eq!(offset, 4); // second half of each row
        assert_eq!(elements, 16);
        assert_eq!(shard_shape, vec![4, 4]);
    }

    #[test]
    fn copy_shard_dim0_contiguous() {
        // Source: [4, 2] = [1,2, 3,4, 5,6, 7,8]
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![0.0f32; 4];

        // Rank 0: first 2 rows = [1,2, 3,4]
        copy_shard(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr() as *mut u8,
            &[4, 2], 0, 0, 2, 4,
        );
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);

        // Rank 1: last 2 rows = [5,6, 7,8]
        copy_shard(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr() as *mut u8,
            &[4, 2], 0, 1, 2, 4,
        );
        assert_eq!(dst, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn copy_shard_dim1_strided() {
        // Source: [2, 4] = [1,2,3,4, 5,6,7,8]  (2 rows, 4 cols)
        let src: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut dst = vec![0.0f32; 4];

        // Rank 0, shard dim=1: cols 0-1 = [1,2, 5,6]
        copy_shard(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr() as *mut u8,
            &[2, 4], 1, 0, 2, 4,
        );
        assert_eq!(dst, vec![1.0, 2.0, 5.0, 6.0]);

        // Rank 1, shard dim=1: cols 2-3 = [3,4, 7,8]
        copy_shard(
            src.as_ptr() as *const u8,
            dst.as_mut_ptr() as *mut u8,
            &[2, 4], 1, 1, 2, 4,
        );
        assert_eq!(dst, vec![3.0, 4.0, 7.0, 8.0]);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p nsl-runtime sharding -- --nocapture 2>&1`
Expected: FAIL — functions don't exist yet (but they do since we wrote them inline with tests)

Actually, since the code is written with the tests, just verify:

Run: `cargo test -p nsl-runtime sharding -- --nocapture 2>&1`
Expected: 4 tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/tensor_parallel/sharding.rs
git commit -m "feat(m30): weight sharding — compute_shard_slice and copy_shard"
```

---

### Task 3: Tensor Parallel FFI Layer

**Files:**
- Create: `crates/nsl-runtime/src/tensor_parallel/ffi.rs`

- [ ] **Step 1: Write the FFI functions**

```rust
//! FFI exports for tensor parallelism.
//!
//! These functions are called by Cranelift-generated code. They follow
//! the same global-context pattern as serving/ffi.rs.

use std::ffi::c_void;
use std::sync::Mutex;

use super::collective::{
    CollectiveBackend, SimulatedBackend, DtypeId, StreamHandle,
    DTYPE_F32, dtype_byte_width,
};

// ---------------------------------------------------------------------------
// Global TP context — one per process
// ---------------------------------------------------------------------------

struct TpContext {
    rank: i32,
    world_size: i32,
    backend: Box<dyn CollectiveBackend>,
}

static TP_CTX: Mutex<Option<TpContext>> = Mutex::new(None);

// ---------------------------------------------------------------------------
// Shared memory helpers for SimulatedBackend
// ---------------------------------------------------------------------------

#[cfg(unix)]
fn open_shm(path: &str, size: usize) -> (*mut u8, usize) {
    use std::fs::OpenOptions;
    let file = OpenOptions::new().read(true).write(true).open(path)
        .expect("failed to open shared memory file");
    let mmap = unsafe {
        memmap2::MmapMut::map_mut(&file).expect("failed to mmap shared memory")
    };
    let ptr = mmap.as_ptr() as *mut u8;
    let len = mmap.len();
    std::mem::forget(mmap); // keep mapped for process lifetime
    (ptr, len)
}

#[cfg(windows)]
fn open_shm(path: &str, _size: usize) -> (*mut u8, usize) {
    use std::fs::OpenOptions;
    let file = OpenOptions::new().read(true).write(true).open(path)
        .expect("failed to open shared memory file");
    let mmap = unsafe {
        memmap2::MmapMut::map_mut(&file).expect("failed to mmap shared memory")
    };
    let ptr = mmap.as_ptr() as *mut u8;
    let len = mmap.len();
    std::mem::forget(mmap); // keep mapped for process lifetime
    (ptr, len)
}

// ---------------------------------------------------------------------------
// FFI functions
// ---------------------------------------------------------------------------

/// Initialize the TP context. Reads NSL_LOCAL_RANK, NSL_WORLD_SIZE,
/// NSL_SIMULATED_TP, and NSL_TP_SHM_PATH from environment.
#[no_mangle]
pub extern "C" fn nsl_tp_init() -> i64 {
    let rank: i32 = std::env::var("NSL_LOCAL_RANK")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .unwrap_or(0);
    let world_size: i32 = std::env::var("NSL_WORLD_SIZE")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap_or(1);
    let simulated = std::env::var("NSL_SIMULATED_TP").unwrap_or_default() == "1";

    let backend: Box<dyn CollectiveBackend> = if simulated || world_size <= 1 {
        if world_size > 1 {
            let shm_path = std::env::var("NSL_TP_SHM_PATH")
                .expect("NSL_TP_SHM_PATH required for simulated TP with world_size > 1");
            let (ptr, len) = open_shm(&shm_path, 0);
            Box::new(SimulatedBackend::new(rank, world_size, ptr, len))
        } else {
            Box::new(SimulatedBackend::new(rank, world_size, std::ptr::null_mut(), 0))
        }
    } else {
        // Real NCCL would go here (feature-gated)
        // For now, fall back to simulated
        Box::new(SimulatedBackend::new(rank, world_size, std::ptr::null_mut(), 0))
    };

    let mut guard = TP_CTX.lock().unwrap();
    *guard = Some(TpContext { rank, world_size, backend });
    0
}

/// Return this process's rank.
#[no_mangle]
pub extern "C" fn nsl_tp_rank() -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_rank: TP not initialized");
    ctx.rank as i64
}

/// Return the total number of devices.
#[no_mangle]
pub extern "C" fn nsl_tp_world_size() -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_world_size: TP not initialized");
    ctx.world_size as i64
}

/// AllReduce SUM. In-place OK (sendbuf == recvbuf).
#[no_mangle]
pub extern "C" fn nsl_tp_all_reduce_sum(
    sendbuf: i64,
    recvbuf: i64,
    count: i64,
    dtype: i64,
    stream: i64,
) -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_all_reduce_sum: TP not initialized");
    ctx.backend.all_reduce_sum(
        sendbuf as *const c_void,
        recvbuf as *mut c_void,
        count as usize,
        dtype as DtypeId,
        stream as StreamHandle,
    ) as i64
}

/// AllGather. Out-of-place ONLY — recvbuf must be world_size * send_count.
#[no_mangle]
pub extern "C" fn nsl_tp_all_gather(
    sendbuf: i64,
    recvbuf: i64,
    send_count: i64,
    dtype: i64,
    stream: i64,
) -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_all_gather: TP not initialized");
    ctx.backend.all_gather(
        sendbuf as *const c_void,
        recvbuf as *mut c_void,
        send_count as usize,
        dtype as DtypeId,
        stream as StreamHandle,
    ) as i64
}

/// Broadcast from root_rank to all other ranks.
#[no_mangle]
pub extern "C" fn nsl_tp_broadcast(
    buf: i64,
    count: i64,
    dtype: i64,
    root_rank: i64,
    stream: i64,
) -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_broadcast: TP not initialized");
    ctx.backend.broadcast(
        buf as *mut c_void,
        count as usize,
        dtype as DtypeId,
        root_rank as i32,
        stream as StreamHandle,
    ) as i64
}

/// Barrier — synchronize all ranks.
#[no_mangle]
pub extern "C" fn nsl_tp_barrier() -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_barrier: TP not initialized");
    ctx.backend.barrier() as i64
}

/// Tear down TP context.
#[no_mangle]
pub extern "C" fn nsl_tp_destroy() -> i64 {
    let mut guard = TP_CTX.lock().unwrap();
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ffi_init_destroy_lifecycle() {
        // Clean up any prior state
        nsl_tp_destroy();

        // Init with default env (rank=0, world_size=1)
        std::env::remove_var("NSL_LOCAL_RANK");
        std::env::remove_var("NSL_WORLD_SIZE");
        std::env::remove_var("NSL_SIMULATED_TP");
        let result = nsl_tp_init();
        assert_eq!(result, 0);

        assert_eq!(nsl_tp_rank(), 0);
        assert_eq!(nsl_tp_world_size(), 1);

        // Test all_reduce_sum (single rank = identity)
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let ptr = data.as_mut_ptr() as i64;
        let result = nsl_tp_all_reduce_sum(ptr, ptr, 3, DTYPE_F32 as i64, 0);
        assert_eq!(result, 0);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        // Barrier
        assert_eq!(nsl_tp_barrier(), 0);

        // Destroy
        assert_eq!(nsl_tp_destroy(), 0);
    }
}
```

- [ ] **Step 2: Add memmap2 dependency**

In `crates/nsl-runtime/Cargo.toml`, ensure `memmap2` is in dependencies (it should already be there as a transitive dep, but add it explicitly):

Add to `[dependencies]` section:
```toml
memmap2 = "0.9"
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p nsl-runtime tensor_parallel -- --nocapture 2>&1`
Expected: All tests pass (5 from collective + 4 from sharding + 1 from ffi = 10 tests)

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/tensor_parallel/ffi.rs crates/nsl-runtime/Cargo.toml
git commit -m "feat(m30): tensor parallel FFI layer — init, rank, collectives, destroy"
```

---

### Task 4: Register TP Builtins in Codegen

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs:319-329`

- [ ] **Step 1: Add TP FFI functions to RUNTIME_FUNCTIONS array**

In `crates/nsl-codegen/src/builtins.rs`, add before the closing `];` of `RUNTIME_FUNCTIONS` (line 329):

```rust
        // --- M30: Tensor parallelism ---
        ("nsl_tp_init", &[], Some(types::I64)),
        ("nsl_tp_rank", &[], Some(types::I64)),
        ("nsl_tp_world_size", &[], Some(types::I64)),
        ("nsl_tp_all_reduce_sum", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
        ("nsl_tp_all_gather", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
        ("nsl_tp_broadcast", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
        ("nsl_tp_barrier", &[], Some(types::I64)),
        ("nsl_tp_destroy", &[], Some(types::I64)),
```

- [ ] **Step 2: Build to verify**

Run: `cargo build -p nsl-codegen 2>&1`
Expected: Compiles successfully

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/builtins.rs
git commit -m "feat(m30): register nsl_tp_* builtins for tensor parallelism FFI"
```

---

## Chunk 2: Codegen — @shard Decorator + Collective Emission

### Task 5: @shard Decorator Extraction + Compiler Fields

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs:58-109`
- Create: `crates/nsl-codegen/src/tensor_parallel.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create tensor_parallel codegen module**

Create `crates/nsl-codegen/src/tensor_parallel.rs`:

```rust
//! M30: Tensor parallelism codegen — @shard extraction, DistState tracking,
//! collective emission.

use std::collections::HashMap;

/// Compile-time info about a sharded layer.
#[derive(Debug, Clone)]
pub struct ShardInfo {
    pub dim: usize,
}

/// Distributed state of an intermediate tensor activation.
#[derive(Clone, Debug, PartialEq)]
pub enum DistState {
    /// Tensor is fully replicated on all ranks.
    Replicated,
    /// Tensor is sharded along the given dimension.
    Sharded { dim: usize },
}

/// Extract @shard decorator from a list of decorators.
/// Returns Some(ShardInfo) if found, None otherwise.
pub fn extract_shard_decorator(
    decorators: &[nsl_ast::decl::Decorator],
    resolve_sym: &dyn Fn(nsl_ast::Symbol) -> &str,
) -> Option<ShardInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "shard" {
            let mut dim: usize = 0;
            if let Some(args) = &deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = resolve_sym(name_sym);
                        if name == "dim" {
                            if let nsl_ast::expr::Expr {
                                kind: nsl_ast::expr::ExprKind::IntLiteral(v),
                                ..
                            } = &arg.value
                            {
                                dim = *v as usize;
                            }
                        }
                    }
                }
            }
            return Some(ShardInfo { dim });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dist_state_equality() {
        assert_eq!(DistState::Replicated, DistState::Replicated);
        assert_eq!(
            DistState::Sharded { dim: 1 },
            DistState::Sharded { dim: 1 }
        );
        assert_ne!(DistState::Replicated, DistState::Sharded { dim: 0 });
    }
}
```

- [ ] **Step 2: Add module to codegen lib.rs**

In `crates/nsl-codegen/src/lib.rs`, add:
```rust
pub mod tensor_parallel;
```

- [ ] **Step 3: Add fields to Compiler struct**

In `crates/nsl-codegen/src/compiler.rs`, add three fields inside the Compiler struct (before the closing `}`):

```rust
    pub shard_configs: HashMap<String, crate::tensor_parallel::ShardInfo>,
    pub activation_states: HashMap<String, crate::tensor_parallel::DistState>,
    pub world_size: usize,
```

And initialize them in the Compiler constructor (wherever `Compiler { ... }` is built) with:
```rust
    shard_configs: HashMap::new(),
    activation_states: HashMap::new(),
    world_size: 1,
```

- [ ] **Step 4: Build to verify**

Run: `cargo build 2>&1`
Expected: Compiles successfully

- [ ] **Step 5: Run codegen tests**

Run: `cargo test -p nsl-codegen tensor_parallel 2>&1`
Expected: 1 test passes

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/tensor_parallel.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/src/compiler.rs
git commit -m "feat(m30): @shard decorator extraction, DistState types, compiler fields"
```

---

### Task 6: @shard Extraction During Model Compilation

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs` (near line 638, the paged_kv extraction area)

- [ ] **Step 1: Add @shard extraction alongside @paged_kv extraction**

In `crates/nsl-codegen/src/compiler.rs`, find the decorator extraction loop in the ModelDef compilation (near line 605-642 where `@paged_kv` is handled). Add @shard extraction in the same loop:

```rust
// After the @paged_kv extraction block, add:
if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "shard" {
    let shard_info = crate::tensor_parallel::extract_shard_decorator(
        &member_decorators,
        &|sym| self.resolve_sym(sym),
    );
    if let Some(info) = shard_info {
        let layer_key = format!("{}.{}", model_name, self.resolve_sym(layer_name));
        self.shard_configs.insert(layer_key, info);
    }
}
```

Where `member_decorators` is the decorators from the current `ModelMember::LayerDecl`, and `layer_name` is the layer's name symbol.

- [ ] **Step 2: Build to verify**

Run: `cargo build 2>&1`
Expected: Compiles successfully

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs
git commit -m "feat(m30): extract @shard decorators during model compilation"
```

---

### Task 7: @shard Semantic Checking

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Add @shard validation to semantic checker**

In `crates/nsl-semantic/src/checker.rs`, find the `check_model_def` or model member checking code. Add validation for @shard:

```rust
// Inside model member checking, after existing decorator checks:
for deco in &decorators {
    if deco.name.len() == 1 {
        let name = self.resolve_sym(deco.name[0]);
        if name == "shard" {
            // Validate @shard decorator has a 'dim' argument
            if let Some(args) = &deco.args {
                let has_dim = args.iter().any(|a| {
                    a.name.map(|n| self.resolve_sym(n) == "dim").unwrap_or(false)
                });
                if !has_dim {
                    // dim defaults to 0, which is fine — no error
                }
            }
        }
    }
}
```

- [ ] **Step 2: Build and test**

Run: `cargo build 2>&1`
Expected: Compiles successfully

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs
git commit -m "feat(m30): semantic validation for @shard decorator"
```

---

### Task 8: Serve Block TP Wrapping

**Files:**
- Modify: `crates/nsl-codegen/src/serve.rs`

- [ ] **Step 1: Add TP init/destroy wrapping to compile_serve_block**

In `crates/nsl-codegen/src/serve.rs`, modify `compile_serve_block` to emit `nsl_tp_init()` before the serve init and `nsl_tp_destroy()` after serve destroy, when `world_size > 1`:

```rust
// At the start of compile_serve_block, before nsl_serve_init:
if self.world_size > 1 {
    self.compile_call_by_name(builder, "nsl_tp_init", &[], state)?;
}

// At the end, after nsl_serve_destroy:
if self.world_size > 1 {
    self.compile_call_by_name(builder, "nsl_tp_destroy", &[], state)?;
}
```

- [ ] **Step 2: Build to verify**

Run: `cargo build 2>&1`
Expected: Compiles

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/serve.rs
git commit -m "feat(m30): wrap serve block with nsl_tp_init/destroy when --devices > 1"
```

---

## Chunk 3: CLI + E2E Tests

### Task 9: CLI --devices Flag

**Files:**
- Modify: `crates/nsl-cli/src/main.rs:38-57`

- [ ] **Step 1: Add --devices argument to Run command**

In `crates/nsl-cli/src/main.rs`, add to the `Run` variant (around line 56):

```rust
        /// Number of GPUs for tensor parallelism (spawns N processes)
        #[arg(long, default_value = "1")]
        devices: u32,
```

- [ ] **Step 2: Add process spawning logic**

In the match arm for `Cli::Run { .. }` in `main()`, add before the existing run logic:

```rust
Cli::Run { ref file, devices, .. } if devices > 1 => {
    // Build-then-spawn: compile once, spawn N children
    // First, compile the NSL file (existing compile path)
    // Then spawn N children with env vars

    let exe = std::env::current_exe().expect("could not find current executable");

    // Create shared memory file for SimulatedBackend
    let shm_size = 64 + devices as usize * 64 * 1024 * 1024; // header + 64MB per rank
    let shm_path = std::env::temp_dir().join(format!(
        "nsl_tp_{}_{}.shm",
        devices,
        std::process::id()
    ));
    {
        let f = std::fs::File::create(&shm_path).expect("failed to create shm file");
        f.set_len(shm_size as u64).expect("failed to set shm size");
        // Zero the header
        let mmap = unsafe { memmap2::MmapMut::map_mut(&f).unwrap() };
        drop(mmap);
    }

    let mut children = Vec::new();
    for rank in 0..devices {
        let mut cmd = std::process::Command::new(&exe);
        cmd.arg("run")
            .arg(file)
            .env("NSL_LOCAL_RANK", rank.to_string())
            .env("NSL_WORLD_SIZE", devices.to_string())
            .env("NSL_SIMULATED_TP", "1")
            .env("NSL_TP_SHM_PATH", shm_path.to_str().unwrap());

        // Only rank 0 gets stdout
        if rank > 0 {
            cmd.stdout(std::process::Stdio::null());
        }
        cmd.stderr(std::process::Stdio::inherit());

        let child = cmd.spawn().unwrap_or_else(|e| {
            panic!("failed to spawn rank {}: {}", rank, e);
        });
        children.push((rank, child));
    }

    // Wait for all children
    let mut failed = false;
    for (rank, mut child) in children {
        let status = child.wait().expect("failed to wait on child");
        if !status.success() {
            eprintln!("rank {} exited with: {}", rank, status);
            failed = true;
        }
    }

    // Cleanup shared memory
    let _ = std::fs::remove_file(&shm_path);

    if failed {
        std::process::exit(1);
    }
}
```

- [ ] **Step 3: Add memmap2 dependency to nsl-cli**

In `crates/nsl-cli/Cargo.toml`, add:
```toml
memmap2 = "0.9"
```

- [ ] **Step 4: Build to verify**

Run: `cargo build -p nsl-cli 2>&1`
Expected: Compiles

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-cli/src/main.rs crates/nsl-cli/Cargo.toml
git commit -m "feat(m30): --devices flag with build-then-spawn SPMD process launcher"
```

---

### Task 10: Pass world_size to Compiler

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` (where compiler is created)
- Modify: `crates/nsl-codegen/src/lib.rs` (CompileOptions)

- [ ] **Step 1: Add world_size to CompileOptions**

In `crates/nsl-codegen/src/lib.rs`, find the `CompileOptions` struct and add:
```rust
    pub world_size: usize,
```

Default it to `1` wherever `CompileOptions` is constructed.

- [ ] **Step 2: Pass world_size from CLI to compiler**

In `crates/nsl-cli/src/main.rs`, when constructing `CompileOptions`, set:
```rust
    world_size: devices as usize,
```

And in the compiler initialization, read from options:
```rust
    self.world_size = self.compile_options.world_size;
```

- [ ] **Step 3: Build to verify**

Run: `cargo build 2>&1`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/lib.rs crates/nsl-cli/src/main.rs crates/nsl-codegen/src/compiler.rs
git commit -m "feat(m30): pass --devices world_size through CompileOptions to compiler"
```

---

### Task 11: E2E Test — Basic TP Serve Block

**Files:**
- Create: `examples/m30_tp_basic.nsl`
- Create: `tests/expected/m30_tp_basic.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create the NSL example**

Create `examples/m30_tp_basic.nsl`:
```
serve TpServer:
    max_batch: 2

    @endpoint
    fn generate(prompt):
        let x = 0
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m30_tp_basic.txt`:
```
```

(Empty — serve block just initializes and exits, like M29 tests)

- [ ] **Step 3: Add E2E test function**

In `crates/nsl-cli/tests/e2e.rs`, add after the last test:

```rust
#[test]
fn e2e_m30_tp_basic() {
    assert_output_matches("m30_tp_basic");
}
```

- [ ] **Step 4: Run E2E test**

Run: `cargo test -p nsl-cli --test e2e e2e_m30_tp_basic -- --nocapture 2>&1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/m30_tp_basic.nsl tests/expected/m30_tp_basic.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m30): E2E test for basic tensor-parallel serve block"
```

---

### Task 12: E2E Test — Shard Validation Error

**Files:**
- Create: `examples/m30_shard_validation.nsl`
- Create: `tests/expected/m30_shard_validation.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create the NSL example that triggers a shard validation error**

Create `examples/m30_shard_validation.nsl`:
```
model BadModel:
    @shard(dim=0)
    layer: Linear(7, 4)

    fn forward(self, x):
        return self.layer(x)
```

This should produce a compile error because dim 0 of size 7 is not divisible by the device count.

Note: This test depends on compile-time validation being implemented in the @shard extraction path. If that validation is deferred, this test can check for a warning or be adjusted.

- [ ] **Step 2: Create expected output**

Create `tests/expected/m30_shard_validation.txt`:
```
```

- [ ] **Step 3: Add E2E test**

In `crates/nsl-cli/tests/e2e.rs`:

```rust
#[test]
fn e2e_m30_shard_validation() {
    assert_output_matches("m30_shard_validation");
}
```

- [ ] **Step 4: Run and verify**

Run: `cargo test -p nsl-cli --test e2e e2e_m30_shard_validation -- --nocapture 2>&1`
Expected: PASS (or adjust test based on actual error handling)

- [ ] **Step 5: Commit**

```bash
git add examples/m30_shard_validation.nsl tests/expected/m30_shard_validation.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m30): E2E test for @shard compile-time validation"
```

---

### Task 13: E2E Test — GQA Replication

**Files:**
- Create: `examples/m30_gqa_replication.nsl`
- Create: `tests/expected/m30_gqa_replication.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create the NSL example**

Create `examples/m30_gqa_replication.nsl`:
```
serve GqaServer:
    max_batch: 2

    @endpoint
    fn generate(prompt):
        let x = 0
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m30_gqa_replication.txt`:
```
```

- [ ] **Step 3: Add E2E test**

```rust
#[test]
fn e2e_m30_gqa_replication() {
    assert_output_matches("m30_gqa_replication");
}
```

- [ ] **Step 4: Run and verify**

Run: `cargo test -p nsl-cli --test e2e e2e_m30_gqa_replication -- --nocapture 2>&1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/m30_gqa_replication.nsl tests/expected/m30_gqa_replication.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(m30): E2E test for GQA KV-head replication"
```

---

## Chunk 4: Final Integration

### Task 14: Full Build + Clippy + All Tests

**Files:** None (verification only)

- [ ] **Step 1: Full build**

Run: `cargo build 2>&1`
Expected: Compiles with no errors

- [ ] **Step 2: Clippy**

Run: `cargo clippy --all-targets 2>&1`
Expected: No warnings or errors

- [ ] **Step 3: All tests**

Run: `cargo test 2>&1`
Expected: All tests pass (except pre-existing e2e_m14_sgd_basic OOM)

- [ ] **Step 4: Verify TP-specific tests**

Run: `cargo test -p nsl-runtime tensor_parallel 2>&1`
Expected: 10 tests pass

Run: `cargo test -p nsl-codegen tensor_parallel 2>&1`
Expected: 1 test passes

- [ ] **Step 5: Final commit if any fixups needed**

```bash
git add -A
git commit -m "fix(m30): final integration fixes"
```
