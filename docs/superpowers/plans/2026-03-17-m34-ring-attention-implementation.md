# M34: Context Parallelism (Ring Attention) — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ring attention for splitting long sequences across multiple GPUs, enabling million-token context windows via ring-based K/V exchange with communication-computation overlap.

**Architecture:** Runtime context_parallel module handles sequence partitioning, ring communication (extending M30's CollectiveBackend with send/recv), online softmax correction across ring passes, and causal early termination. Codegen extracts `@context_parallel` decorator and lowers attention to ring attention calls. FlashAttention gains an `accumulate` mode for partial result accumulation across ring passes.

**Tech Stack:** Rust (runtime + codegen), Cranelift (codegen), NCCL via M30's CollectiveBackend (GPU path), SimulatedBackend (CPU test path)

**Spec:** `docs/superpowers/specs/2026-03-15-m34-ring-attention-design.md`

---

## Scope Note

The spec covers 7 subsystems. This plan orders them so each produces independently testable code:

1. **Tasks 1-2**: Sequence partitioning (pure Rust, zero dependencies)
2. **Tasks 3-4**: Online softmax correction (pure math, key algorithm)
3. **Tasks 5-6**: CollectiveBackend send/recv + ring communication
4. **Task 7**: Ring attention orchestration loop
5. **Tasks 8-9**: Semantic validation + codegen integration
6. **Task 10**: FFI layer
7. **Tasks 11-12**: E2E tests + final verification

**Deferred to M34b:** FlashAttention `accumulate` mode PTX (requires modifying the inner PTX kernel), async CUDA stream overlap, gradient checkpointing at ring boundaries, `--context-parallel` CLI flag.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/context_parallel/mod.rs` | Module declarations, re-exports | 15 |
| `crates/nsl-runtime/src/context_parallel/types.rs` | `RingAttentionContext`, `RingPassInfo`, `CpCommunicator` structs | 80 |
| `crates/nsl-runtime/src/context_parallel/partition.rs` | Sequence split/gather (chunk extraction + all-gather) | 120 |
| `crates/nsl-runtime/src/context_parallel/softmax.rs` | Online softmax correction across ring passes | 100 |
| `crates/nsl-runtime/src/context_parallel/ring.rs` | Ring send/recv using CollectiveBackend, double-buffer management | 180 |
| `crates/nsl-runtime/src/context_parallel/attention.rs` | Ring attention outer loop orchestrating FlashAttention + ring comm | 250 |
| `crates/nsl-runtime/src/context_parallel/ffi.rs` | `#[no_mangle] pub extern "C"` FFI wrappers | 250 |
| `crates/nsl-semantic/src/context_parallel.rs` | `@context_parallel` decorator validation | 100 |
| `crates/nsl-codegen/src/context_parallel.rs` | Decorator extraction, `ContextParallelInfo`, compiler integration | 200 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod context_parallel;` |
| `crates/nsl-runtime/src/tensor_parallel/collective.rs` | Add `send`/`recv` to `CollectiveBackend` trait + `SimulatedBackend` impl |
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod context_parallel;` |
| `crates/nsl-codegen/src/builtins.rs` | Register 6 CP FFI functions |
| `crates/nsl-codegen/src/compiler.rs` | Add `context_parallel_configs` + `cp_ring_size` fields |
| `crates/nsl-codegen/src/tensor_parallel.rs` | Add `SeqPartitioned` to `DistState` |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod context_parallel;` |
| `crates/nsl-semantic/src/checker.rs` | Add `@context_parallel` decorator validation |
| `crates/nsl-cli/tests/e2e.rs` | Add M34 E2E tests |

---

## Task 1: Runtime Module Skeleton + Types

**Files:**
- Create: `crates/nsl-runtime/src/context_parallel/mod.rs`
- Create: `crates/nsl-runtime/src/context_parallel/types.rs`
- Create: stub files for `partition.rs`, `softmax.rs`, `ring.rs`, `attention.rs`, `ffi.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Add `pub mod context_parallel;` to runtime lib.rs**

- [ ] **Step 2: Create `context_parallel/mod.rs`**

```rust
pub mod types;
pub mod partition;
pub mod softmax;
pub mod ring;
pub mod attention;
pub mod ffi;
```

- [ ] **Step 3: Create `context_parallel/types.rs`**

```rust
/// Configuration for ring attention context parallelism.
#[derive(Debug, Clone)]
pub struct RingConfig {
    /// Number of GPUs in the ring.
    pub ring_size: usize,
    /// This GPU's rank in the ring (0..ring_size-1).
    pub ring_rank: usize,
    /// Tokens per GPU chunk: total_seq_len / ring_size.
    pub local_seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of KV heads (for GQA).
    pub num_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Whether attention is causal (enables early termination).
    pub causal: bool,
}

impl RingConfig {
    /// Number of ring passes this GPU needs.
    /// Causal: ring_rank + 1 (GPU 0 needs only local, GPU N-1 needs all).
    /// Non-causal: ring_size (all GPUs need all K/V).
    pub fn num_passes(&self) -> usize {
        if self.causal {
            self.ring_rank + 1
        } else {
            self.ring_size
        }
    }

    /// Source rank for a given ring pass.
    /// Pass 0 = local (ring_rank). Pass p = (ring_rank - p + ring_size) % ring_size.
    pub fn source_rank(&self, pass: usize) -> usize {
        (self.ring_rank + self.ring_size - pass) % self.ring_size
    }
}

/// Per-ring-pass metadata.
#[derive(Debug, Clone)]
pub struct RingPassInfo {
    /// Which chunk of the global sequence is in the current K/V buffer.
    pub chunk_start: usize,
    /// Length of the K/V chunk.
    pub chunk_len: usize,
    /// Source rank that produced this K/V chunk.
    pub source_rank: usize,
    /// Whether this is the last pass.
    pub is_final: bool,
}
```

- [ ] **Step 4: Create empty stubs**
- `partition.rs` — `// Sequence partitioning`
- `softmax.rs` — `// Online softmax correction`
- `ring.rs` — `// Ring send/recv`
- `attention.rs` — `// Ring attention loop`
- `ffi.rs` — `// Context parallel FFI`

- [ ] **Step 5: Verify compilation, commit**

```bash
cargo check -p nsl-runtime
git commit -m "feat(m34): add context_parallel runtime module skeleton with types"
```

---

## Task 2: Sequence Partitioning

**Files:**
- Modify: `crates/nsl-runtime/src/context_parallel/partition.rs`

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_partition_4way() {
        // 12 tokens, 4 ranks, hidden_dim=2
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let chunk = partition_sequence(&data, 12, 2, 4, 1); // rank 1
        // rank 1 gets tokens 3..6 = indices 6..12
        assert_eq!(chunk, vec![6.0, 7.0, 8.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn test_partition_gather_roundtrip() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let hidden_dim = 2;
        let ring_size = 2;
        let seq_len = 8;

        let chunk0 = partition_sequence(&data, seq_len, hidden_dim, ring_size, 0);
        let chunk1 = partition_sequence(&data, seq_len, hidden_dim, ring_size, 1);

        let gathered = gather_sequence(&[chunk0, chunk1], seq_len, hidden_dim);
        assert_eq!(gathered, data);
    }

    #[test]
    fn test_chunk_boundaries() {
        let (start, end) = chunk_range(1000, 4, 2); // rank 2 of 4
        assert_eq!(start, 500);
        assert_eq!(end, 750);
    }
}
```

- [ ] **Step 2: Implement**

```rust
/// Compute the token range for a given rank.
pub fn chunk_range(seq_len: usize, ring_size: usize, rank: usize) -> (usize, usize) {
    let chunk_len = seq_len / ring_size;
    let start = rank * chunk_len;
    let end = start + chunk_len;
    (start, end)
}

/// Extract this rank's chunk from a full sequence.
/// `data`: flat [seq_len, hidden_dim]
pub fn partition_sequence(
    data: &[f32],
    seq_len: usize,
    hidden_dim: usize,
    ring_size: usize,
    rank: usize,
) -> Vec<f32> {
    let (start, end) = chunk_range(seq_len, ring_size, rank);
    let chunk_len = end - start;
    let mut chunk = vec![0.0f32; chunk_len * hidden_dim];
    for t in 0..chunk_len {
        let src_off = (start + t) * hidden_dim;
        let dst_off = t * hidden_dim;
        chunk[dst_off..dst_off + hidden_dim]
            .copy_from_slice(&data[src_off..src_off + hidden_dim]);
    }
    chunk
}

/// Gather chunks from all ranks into a full sequence.
/// `chunks`: one Vec per rank, each [chunk_len, hidden_dim]
pub fn gather_sequence(
    chunks: &[Vec<f32>],
    seq_len: usize,
    hidden_dim: usize,
) -> Vec<f32> {
    let mut full = vec![0.0f32; seq_len * hidden_dim];
    let chunk_len = seq_len / chunks.len();
    for (rank, chunk) in chunks.iter().enumerate() {
        let start = rank * chunk_len * hidden_dim;
        full[start..start + chunk.len()].copy_from_slice(chunk);
    }
    full
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime context_parallel::partition -- --nocapture
git commit -m "feat(m34): implement sequence partitioning for ring attention"
```

---

## Task 3: Online Softmax Correction

**Files:**
- Modify: `crates/nsl-runtime/src/context_parallel/softmax.rs`

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_pass_matches_single_pass() {
        // Single-pass softmax over [1.0, 2.0, 3.0, 4.0]
        let full = vec![1.0f32, 2.0, 3.0, 4.0];
        let full_softmax = softmax(&full);

        // Two-pass: first [1.0, 2.0], then correct with [3.0, 4.0]
        let (max1, sum1, weighted1) = partial_softmax(&[1.0, 2.0]);
        let (max2, sum2, weighted2) = partial_softmax(&[3.0, 4.0]);

        let (final_max, final_sum, corrected) =
            merge_partial_softmax(max1, sum1, &weighted1, max2, sum2, &weighted2);

        // Normalize
        let result: Vec<f32> = corrected.iter().map(|&w| w / final_sum).collect();

        for (a, b) in full_softmax.iter().zip(result.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_correction_with_larger_new_max() {
        // Old pass had small values, new pass has large values
        let (max1, sum1, w1) = partial_softmax(&[0.1, 0.2]);
        let (max2, sum2, w2) = partial_softmax(&[10.0, 20.0]);

        let (final_max, final_sum, _) =
            merge_partial_softmax(max1, sum1, &w1, max2, sum2, &w2);

        assert!((final_max - 20.0).abs() < 1e-5);
        assert!(final_sum > 0.0);
    }
}
```

- [ ] **Step 2: Implement**

```rust
/// Compute partial softmax for a chunk: returns (max, sum_exp, weighted_values).
/// weighted_values[i] = exp(values[i] - max)
pub fn partial_softmax(values: &[f32]) -> (f32, f32, Vec<f32>) {
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    (max_val, sum, exps)
}

/// Merge two partial softmax results using online correction.
///
/// Returns (new_max, new_sum, corrected_weights) where:
/// - corrected_weights contains both old (rescaled) and new weights
/// - new_sum = old_sum * correction + new_sum_local
pub fn merge_partial_softmax(
    old_max: f32,
    old_sum: f32,
    old_weights: &[f32],
    new_max: f32,
    new_sum: f32,
    new_weights: &[f32],
) -> (f32, f32, Vec<f32>) {
    let global_max = old_max.max(new_max);
    let old_correction = (old_max - global_max).exp();
    let new_correction = (new_max - global_max).exp();

    let corrected_old_sum = old_sum * old_correction;
    let corrected_new_sum = new_sum * new_correction;
    let total_sum = corrected_old_sum + corrected_new_sum;

    let mut corrected = Vec::with_capacity(old_weights.len() + new_weights.len());
    for &w in old_weights {
        corrected.push(w * old_correction);
    }
    for &w in new_weights {
        corrected.push(w * new_correction);
    }

    (global_max, total_sum, corrected)
}

/// Standard softmax for reference testing.
pub fn softmax(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = values.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime context_parallel::softmax -- --nocapture
git commit -m "feat(m34): implement online softmax correction for ring attention"
```

---

## Task 4: RingConfig Tests

**Files:**
- Modify: `crates/nsl-runtime/src/context_parallel/types.rs`

- [ ] **Step 1: Add tests for causal early termination and source rank calculation**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_causal_num_passes() {
        let cfg = RingConfig {
            ring_size: 4, ring_rank: 0, local_seq_len: 250,
            num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: true,
        };
        assert_eq!(cfg.num_passes(), 1); // GPU 0: local only

        let cfg3 = RingConfig { ring_rank: 3, ..cfg };
        assert_eq!(cfg3.num_passes(), 4); // GPU 3: all passes
    }

    #[test]
    fn test_noncausal_num_passes() {
        let cfg = RingConfig {
            ring_size: 4, ring_rank: 0, local_seq_len: 250,
            num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: false,
        };
        assert_eq!(cfg.num_passes(), 4); // all GPUs need all passes
    }

    #[test]
    fn test_source_rank_ring_order() {
        let cfg = RingConfig {
            ring_size: 4, ring_rank: 2, local_seq_len: 250,
            num_heads: 32, num_kv_heads: 8, head_dim: 128, causal: false,
        };
        // Pass 0: local (rank 2)
        assert_eq!(cfg.source_rank(0), 2);
        // Pass 1: from rank 1
        assert_eq!(cfg.source_rank(1), 1);
        // Pass 2: from rank 0
        assert_eq!(cfg.source_rank(2), 0);
        // Pass 3: from rank 3
        assert_eq!(cfg.source_rank(3), 3);
    }
}
```

- [ ] **Step 2: Run tests, commit**

```bash
cargo test -p nsl-runtime context_parallel::types -- --nocapture
git commit -m "test(m34): add ring config tests for causal early termination and source rank"
```

---

## Task 5: CollectiveBackend send/recv Extension

**Files:**
- Modify: `crates/nsl-runtime/src/tensor_parallel/collective.rs`

- [ ] **Step 1: Add `send` and `recv` to the `CollectiveBackend` trait**

Add after the existing `barrier` method:

```rust
    /// Point-to-point send to a specific rank.
    fn send(
        &self,
        sendbuf: *const std::os::raw::c_void,
        count: usize,
        dtype_bytes: usize,
        dst_rank: i32,
    ) -> i32;

    /// Point-to-point receive from a specific rank.
    fn recv(
        &self,
        recvbuf: *mut std::os::raw::c_void,
        count: usize,
        dtype_bytes: usize,
        src_rank: i32,
    ) -> i32;
```

- [ ] **Step 2: Implement send/recv on SimulatedBackend**

Follow the same pattern as `all_reduce_sum`: write to shared memory, barrier, read from partner's slot.

```rust
    fn send(&self, sendbuf: *const c_void, count: usize, dtype_bytes: usize, dst_rank: i32) -> i32 {
        let bytes = count * dtype_bytes;
        let slot = self.rank_data_slot(self.rank as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(sendbuf as *const u8, slot, bytes);
        }
        self.spin_barrier();
        0
    }

    fn recv(&self, recvbuf: *mut c_void, count: usize, dtype_bytes: usize, src_rank: i32) -> i32 {
        self.spin_barrier();
        let bytes = count * dtype_bytes;
        let slot = self.rank_data_slot(src_rank as usize);
        unsafe {
            std::ptr::copy_nonoverlapping(slot, recvbuf as *mut u8, bytes);
        }
        self.spin_barrier();
        0
    }
```

NOTE: The SimulatedBackend uses barriers for synchronization. The send writes to the sender's slot, then both ranks barrier. The recv reads from the sender's slot after the barrier. This is correct for the simulated case but won't overlap with compute (that's fine — async overlap is CUDA-only and deferred).

- [ ] **Step 3: Add test**

```rust
    #[test]
    fn simulated_backend_send_recv() {
        // Single-rank send/recv to self (degenerate case)
        let backend = SimulatedBackend::new(0, 1);
        let send_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut recv_data = vec![0.0f32; 4];

        let rc = backend.send(
            send_data.as_ptr() as *const c_void,
            4, std::mem::size_of::<f32>(), 0,
        );
        assert_eq!(rc, 0);

        let rc = backend.recv(
            recv_data.as_mut_ptr() as *mut c_void,
            4, std::mem::size_of::<f32>(), 0,
        );
        assert_eq!(rc, 0);
        assert_eq!(recv_data, send_data);
    }
```

- [ ] **Step 4: Verify all existing collective tests still pass, commit**

```bash
cargo test -p nsl-runtime tensor_parallel::collective -- --nocapture
git commit -m "feat(m34): add send/recv to CollectiveBackend trait with SimulatedBackend impl"
```

---

## Task 6: Ring Send/Recv Wrapper

**Files:**
- Modify: `crates/nsl-runtime/src/context_parallel/ring.rs`

- [ ] **Step 1: Write tests for ring communication**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_next_prev() {
        assert_eq!(ring_next(2, 4), 3);
        assert_eq!(ring_next(3, 4), 0); // wraps
        assert_eq!(ring_prev(0, 4), 3); // wraps
        assert_eq!(ring_prev(2, 4), 1);
    }

    #[test]
    fn test_double_buffer_index() {
        assert_eq!(buffer_index(0), 0);
        assert_eq!(buffer_index(1), 1);
        assert_eq!(buffer_index(2), 0); // wraps
        assert_eq!(buffer_index(3), 1);
    }
}
```

- [ ] **Step 2: Implement ring topology helpers**

```rust
/// Next rank in the ring (send destination).
pub fn ring_next(rank: usize, ring_size: usize) -> usize {
    (rank + 1) % ring_size
}

/// Previous rank in the ring (recv source).
pub fn ring_prev(rank: usize, ring_size: usize) -> usize {
    (rank + ring_size - 1) % ring_size
}

/// Double-buffer index for a ring pass.
pub fn buffer_index(pass: usize) -> usize {
    pass % 2
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime context_parallel::ring -- --nocapture
git commit -m "feat(m34): add ring topology helpers and double-buffer indexing"
```

---

## Task 7: Ring Attention Orchestration (CPU Fallback)

**Files:**
- Modify: `crates/nsl-runtime/src/context_parallel/attention.rs`

- [ ] **Step 1: Write test for ring attention matching standard attention**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::context_parallel::softmax;
    use crate::context_parallel::partition;

    #[test]
    fn test_ring_attention_matches_standard_2gpu() {
        // Simple dot-product attention: softmax(Q @ K^T) @ V
        // Q = K = V = [[1,0],[0,1],[1,1],[0,0]] (4 tokens, dim=2)
        let q: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0];
        let k = q.clone();
        let v = q.clone();
        let seq_len = 4;
        let dim = 2;

        // Standard: full attention
        let standard_out = naive_attention(&q, &k, &v, seq_len, dim, false);

        // Ring with 2 GPUs: partition, compute local + remote, merge
        let q0 = partition::partition_sequence(&q, seq_len, dim, 2, 0);
        let k0 = partition::partition_sequence(&k, seq_len, dim, 2, 0);
        let v0 = partition::partition_sequence(&v, seq_len, dim, 2, 0);
        let k1 = partition::partition_sequence(&k, seq_len, dim, 2, 1);
        let v1 = partition::partition_sequence(&v, seq_len, dim, 2, 1);

        // GPU 0: pass 0 (local k0,v0), pass 1 (remote k1,v1)
        let out0 = ring_attention_cpu(&q0, &[&k0, &k1], &[&v0, &v1], 2, dim, false);

        let q1 = partition::partition_sequence(&q, seq_len, dim, 2, 1);
        let out1 = ring_attention_cpu(&q1, &[&k1, &k0], &[&v1, &v0], 2, dim, false);

        // Gather and compare
        let ring_out = partition::gather_sequence(&[out0, out1], seq_len, dim);

        for (i, (a, b)) in standard_out.iter().zip(ring_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch at index {}: standard={}, ring={}", i, a, b
            );
        }
    }
}
```

- [ ] **Step 2: Implement naive attention + ring attention CPU path**

```rust
use crate::context_parallel::softmax::{partial_softmax, merge_partial_softmax};

/// Naive full attention: softmax(Q @ K^T / sqrt(d)) @ V
/// Q,K,V: flat [seq_len, dim]
pub fn naive_attention(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, dim: usize, causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; seq_len * dim];

    for i in 0..seq_len {
        // Compute attention scores for query i
        let mut scores = vec![f32::NEG_INFINITY; seq_len];
        let k_max = if causal { i + 1 } else { seq_len };
        for j in 0..k_max {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q[i * dim + d] * k[j * dim + d];
            }
            scores[j] = dot * scale;
        }

        // Softmax
        let probs = crate::context_parallel::softmax::softmax(&scores[..k_max]);

        // Weighted sum of V
        for d in 0..dim {
            let mut sum = 0.0f32;
            for j in 0..k_max {
                sum += probs[j] * v[j * dim + d];
            }
            out[i * dim + d] = sum;
        }
    }
    out
}

/// Ring attention CPU fallback: computes attention across multiple K/V chunks.
///
/// `q_local`: [local_seq_len, dim] — this GPU's queries
/// `k_chunks`: ordered K chunks from ring passes (local first, then remote)
/// `v_chunks`: corresponding V chunks
/// Returns: [local_seq_len, dim] attention output
pub fn ring_attention_cpu(
    q_local: &[f32],
    k_chunks: &[&[f32]],
    v_chunks: &[&[f32]],
    local_seq_len: usize,
    dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (dim as f32).sqrt();
    let mut out = vec![0.0f32; local_seq_len * dim];

    for qi in 0..local_seq_len {
        let mut global_max = f32::NEG_INFINITY;
        let mut global_sum = 0.0f32;
        let mut o_acc = vec![0.0f32; dim];

        for (pass, (k_chunk, v_chunk)) in k_chunks.iter().zip(v_chunks.iter()).enumerate() {
            let chunk_len = k_chunk.len() / dim;

            // Compute scores Q[qi] @ K_chunk^T
            let mut scores = Vec::with_capacity(chunk_len);
            for kj in 0..chunk_len {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q_local[qi * dim + d] * k_chunk[kj * dim + d];
                }
                scores.push(dot * scale);
            }

            // Partial softmax for this chunk
            let (chunk_max, chunk_sum, chunk_weights) = partial_softmax(&scores);

            // Online correction: rescale accumulated output
            let new_max = global_max.max(chunk_max);
            let old_correction = (global_max - new_max).exp();
            let new_correction = (chunk_max - new_max).exp();

            // Rescale accumulated output
            for d in 0..dim {
                o_acc[d] *= old_correction;
            }
            global_sum = global_sum * old_correction + chunk_sum * new_correction;
            global_max = new_max;

            // Accumulate: O_acc += softmax_weights @ V_chunk
            for kj in 0..chunk_len {
                let w = chunk_weights[kj] * new_correction;
                for d in 0..dim {
                    o_acc[d] += w * v_chunk[kj * dim + d];
                }
            }
        }

        // Normalize
        for d in 0..dim {
            out[qi * dim + d] = o_acc[d] / global_sum;
        }
    }
    out
}
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-runtime context_parallel::attention -- --nocapture
git commit -m "feat(m34): implement ring attention CPU path with online softmax correction"
```

---

## Task 8: Semantic Validation

**Files:**
- Create: `crates/nsl-semantic/src/context_parallel.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Create validation function** (same pattern as `moe.rs`)

```rust
use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

pub fn validate_context_parallel_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<usize> {
    let mut ring_size: Option<usize> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "ring_size" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 1 {
                                diagnostics.push(
                                    Diagnostic::error("@context_parallel: ring_size must be >= 1".to_string())
                                        .with_label(arg.span, "must be >= 1"),
                                );
                            } else {
                                ring_size = Some(*n as usize);
                            }
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!("@context_parallel: unknown argument '{}'", aname))
                                .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if ring_size.is_none() {
        diagnostics.push(
            Diagnostic::error("@context_parallel: ring_size is required".to_string())
                .with_label(deco.span, "missing ring_size"),
        );
    }

    ring_size
}
```

- [ ] **Step 2: Add `pub mod context_parallel;` to lib.rs, wire into checker.rs**

- [ ] **Step 3: Verify, commit**

```bash
cargo check -p nsl-semantic
git commit -m "feat(m34): add @context_parallel semantic validation"
```

---

## Task 9: Codegen Integration

**Files:**
- Create: `crates/nsl-codegen/src/context_parallel.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`, `builtins.rs`, `compiler.rs`
- Modify: `crates/nsl-codegen/src/tensor_parallel.rs` (add `SeqPartitioned`)

- [ ] **Step 1: Create `context_parallel.rs` with decorator extraction**

```rust
//! M34: Context parallelism codegen — @context_parallel extraction.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

#[derive(Debug, Clone)]
pub struct ContextParallelInfo {
    pub ring_size: usize,
}

pub fn extract_context_parallel_decorator<'a>(
    decorators: &[Decorator],
    resolve_sym: &dyn Fn(Symbol) -> &'a str,
) -> Option<ContextParallelInfo> {
    for deco in decorators {
        if deco.name.len() == 1 && resolve_sym(deco.name[0]) == "context_parallel" {
            let mut ring_size: usize = 0;
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        if resolve_sym(name_sym) == "ring_size" {
                            if let ExprKind::IntLiteral(v) = &arg.value.kind {
                                ring_size = *v as usize;
                            }
                        }
                    }
                }
            }
            if ring_size > 0 {
                return Some(ContextParallelInfo { ring_size });
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_empty() {
        assert!(extract_context_parallel_decorator(&[], &|_| "").is_none());
    }
}
```

- [ ] **Step 2: Add `SeqPartitioned` to DistState**

In `crates/nsl-codegen/src/tensor_parallel.rs`:
```rust
pub enum DistState {
    Replicated,
    Sharded { dim: usize },
    SeqPartitioned { ring_size: usize },  // M34
}
```

- [ ] **Step 3: Register 6 FFI functions in builtins.rs**

```rust
    // --- M34: Context parallelism (ring attention) ---
    ("nsl_cp_init", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_sequence_partition", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_ring_attention", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_ring_send_recv", &[types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_sequence_gather", &[types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64, types::I64], Some(types::I64)),
    ("nsl_cp_destroy", &[types::I64], Some(types::I64)),
```

- [ ] **Step 4: Add compiler fields**

In `compiler.rs`:
```rust
pub context_parallel_configs: HashMap<String, crate::context_parallel::ContextParallelInfo>,
pub cp_ring_size: usize,
```
Initialize as `context_parallel_configs: HashMap::new(), cp_ring_size: 1`.

- [ ] **Step 5: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m34): add context_parallel codegen module, builtins, DistState extension"
```

---

## Task 10: FFI Layer (Stubs)

**Files:**
- Modify: `crates/nsl-runtime/src/context_parallel/ffi.rs`

- [ ] **Step 1: Implement FFI stubs**

```rust
//! M34: Context parallelism FFI.

#[no_mangle]
pub extern "C" fn nsl_cp_init(
    _ring_size: i64, _local_seq_len: i64, _num_heads: i64,
    _num_kv_heads: i64, _head_dim: i64, _dtype: i64,
) -> i64 { 0 }

#[no_mangle]
pub extern "C" fn nsl_sequence_partition(
    _input_ptr: i64, _batch: i64, _seq_len: i64, _hidden_dim: i64,
    _ring_size: i64, _rank: i64, _output_ptr: i64,
) -> i64 { 0 }

#[no_mangle]
pub extern "C" fn nsl_ring_attention(
    _ctx_handle: i64, _scale_bits: i64, _causal: i64,
    _block_table_ptr: i64, _k_pool_ptr: i64, _v_pool_ptr: i64, _block_size: i64,
    _output_ptr: i64,
    _ptx_ptr: i64, _name_ptr: i64, _block_q: i64, _block_kv: i64,
    _shared_mem_bytes: i64,
) -> i64 { 0 }

#[no_mangle]
pub extern "C" fn nsl_ring_send_recv(
    _send_buf_ptr: i64, _recv_buf_ptr: i64,
    _count: i64, _dtype: i64, _stream: i64,
) -> i64 { 0 }

#[no_mangle]
pub extern "C" fn nsl_sequence_gather(
    _local_output_ptr: i64, _full_output_ptr: i64,
    _batch: i64, _local_seq_len: i64, _hidden_dim: i64,
    _ring_size: i64, _dtype: i64, _stream: i64,
) -> i64 { 0 }

#[no_mangle]
pub extern "C" fn nsl_cp_destroy(_ctx_handle: i64) -> i64 { 0 }
```

- [ ] **Step 2: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m34): add context parallelism FFI stubs"
```

---

## Task 11: E2E Tests

**Files:**
- Create: `examples/m34_cp_basic.nsl`, `examples/m34_cp_validation_error.nsl`
- Create: `tests/expected/m34_cp_basic.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create basic CP example** (compiles with @context_parallel decorator)

```nsl
# M34: Context parallelism — decorator validation test

model Attention:
    @context_parallel(ring_size=2)
    layer: int = 0

    fn forward(self, x: Tensor) -> Tensor:
        return x

let m = Attention()
let x = ones([4, 8])
let y = m.forward(x)
print(y)
```

- [ ] **Step 2: Create validation error example** (missing ring_size)

```nsl
# M34: @context_parallel validation error — missing ring_size

model Bad:
    @context_parallel()
    layer: int = 0

    fn forward(self, x: Tensor) -> Tensor:
        return x
```

- [ ] **Step 3: Add E2E tests, verify, commit**

---

## Task 12: Full Test Suite + Clippy

- [ ] **Step 1: Run `cargo test --workspace --lib`** — all tests pass
- [ ] **Step 2: Run `cargo test -p nsl-cli e2e_m34 -- --nocapture`** — E2E tests pass
- [ ] **Step 3: Run `cargo clippy --workspace -- -D warnings`** — clean
- [ ] **Step 4: Commit if fixes needed**

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | Module skeleton + types | compile check |
| 2 | Sequence partitioning | 3 unit |
| 3 | Online softmax correction | 2 unit |
| 4 | RingConfig tests | 3 unit |
| 5 | CollectiveBackend send/recv | 1 unit |
| 6 | Ring topology helpers | 2 unit |
| 7 | Ring attention CPU path | 1 unit |
| 8 | Semantic @context_parallel | compile check |
| 9 | Codegen decorator + builtins | 1 unit |
| 10 | FFI stubs | compile check |
| 11 | E2E tests | 2 E2E |
| 12 | Full verification | all tests |

**Total: 12 tasks, ~13 unit tests + 2 E2E tests**

### Deferred to M34b
- FlashAttention `accumulate` mode PTX (partial result accumulation)
- Async CUDA stream overlap (double-buffered communication)
- Gradient checkpointing at ring boundaries
- `--context-parallel` CLI flag
- TP+CP communicator group assignment
- Causal ring attention with actual token masking in ring passes
