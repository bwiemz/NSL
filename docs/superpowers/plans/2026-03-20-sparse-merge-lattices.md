# Sparse Tensor Merge Lattices (TACO-style) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement TACO-style merge lattices for sparse tensor co-iteration so NSL can efficiently compute operations on pairs of sparse tensors (SpMM, sparse add, sparse mul) without converting to dense. Currently NSL has COO creation and format-based kernel selection but no actual sparse co-iteration algorithm.

**Architecture:** Three components: (1) iteration graphs that determine traversal order based on tensor formats, (2) merge lattices that generate the correct while-loop structure for co-iterating sparse dimensions (intersection for mul, union for add), (3) workspace transformation for efficient output assembly into compressed formats. The merge lattice drives code generation of the actual sparse kernel loops.

**Tech Stack:** Rust, existing `nsl-codegen/src/sparse.rs`, `nsl-runtime/src/sparse.rs`

**Research Basis:** Frontier Features notebook describes TACO compiler's three core mechanisms: iteration graphs, merge lattices, and workspace transformation. The merge lattice approach generates asymptotically optimal co-iteration code for any combination of sparse formats.

---

## Background

### TACO Merge Lattice Concept

For operation `C = A * B` where A is CSR and B is CSC:

**Intersection (multiply):** Only compute where both A and B have non-zeros:
```
while a_idx < A.nnz && b_idx < B.nnz:
    if A.col[a_idx] == B.row[b_idx]:
        C[i,j] += A.val[a_idx] * B.val[b_idx]
        a_idx++; b_idx++
    elif A.col[a_idx] < B.row[b_idx]:
        a_idx++    // skip A (no matching B)
    else:
        b_idx++    // skip B (no matching A)
```

**Union (add):** Compute where either A or B has non-zeros:
```
while a_idx < A.nnz || b_idx < B.nnz:
    if both valid && A.col[a_idx] == B.row[b_idx]:
        C[i,j] = A.val[a_idx] + B.val[b_idx]
        a_idx++; b_idx++
    elif a_only || A.col[a_idx] < B.row[b_idx]:
        C[i,j] = A.val[a_idx]
        a_idx++
    else:
        C[i,j] = B.val[b_idx]
        b_idx++
```

### Current State

- `nsl-codegen/src/sparse.rs`: `SparseOp` enum, format-based kernel selection, but no actual loop generation
- `nsl-runtime/src/sparse.rs`: `NslSparseTensor` struct (COO creation only), no SpMM/SpMV
- No iteration graph, no merge lattice, no workspace

---

## Files to Modify

| File | Change |
|------|--------|
| `crates/nsl-codegen/src/sparse.rs` | Add `MergeLattice`, `IterationGraph`, sparse loop codegen |
| `crates/nsl-runtime/src/sparse.rs` | Add SpMV, SpMM, sparse add/mul, CSR conversion, workspace |
| `crates/nsl-semantic/src/sparse.rs` | Add sparsity propagation for output format inference |
| Tests | Co-iteration correctness, format combinations, workspace assembly |

---

## Tasks

### Task 1: CSR/CSC Conversion

- [ ] **1.1** Implement COO → CSR conversion in runtime:
```rust
pub extern "C" fn nsl_sparse_coo_to_csr(coo_ptr: i64) -> i64 {
    // Sort by row, then build row_ptr array
    // row_ptr[i] = index of first non-zero in row i
}
```

- [ ] **1.2** Implement COO → CSC conversion (sort by column).

- [ ] **1.3** Test: create COO sparse matrix, convert to CSR, verify row_ptr and col_ind arrays.

### Task 2: Iteration Graph

- [ ] **2.1** Define iteration graph structure:
```rust
pub struct IterationGraph {
    pub levels: Vec<IterLevel>,  // one per tensor dimension
}

pub struct IterLevel {
    pub dimension: usize,
    pub format: LevelFormat,     // Dense, Compressed (CSR/CSC), Singleton (COO)
    pub merge_with: Vec<usize>,  // which other tensors co-iterate at this level
}

pub enum LevelFormat {
    Dense,       // iterate 0..N
    Compressed,  // iterate via ptr/idx arrays (CSR row_ptr + col_ind)
    Singleton,   // iterate via coordinate array (COO)
}
```

- [ ] **2.2** Build iteration graph from operation + format combination:
  - CSR × Dense vector → level 0: Compressed (A rows), level 1: Dense (B)
  - CSR × CSC → level 0: Compressed (A rows), level 1: merge Compressed (A cols) ∩ Compressed (B rows)

- [ ] **2.3** Test: `SpMV(CSR, Dense)` produces correct 2-level iteration graph.

### Task 3: Merge Lattice Construction

- [ ] **3.1** Define merge lattice:
```rust
pub enum MergeOp {
    Intersection,  // multiply: both must have non-zeros
    Union,         // add: either has non-zeros
}

pub struct MergeLatticePoint {
    pub iterators: Vec<bool>,  // which tensors participate at this point
    pub expression: MergeExpr, // what to compute
}

pub struct MergeLattice {
    pub points: Vec<MergeLatticePoint>,
    pub merge_op: MergeOp,
}
```

- [ ] **3.2** Build lattice from operation type:
  - `A * B` → Intersection lattice: one point where both A and B are present
  - `A + B` → Union lattice: three points (A only, B only, both)
  - `A * B + C` → nested: intersection(A,B) then union with C

- [ ] **3.3** Test: `sparse_mul` generates intersection lattice; `sparse_add` generates union lattice.

### Task 4: Loop Code Generation (Runtime)

- [ ] **4.1** Implement `nsl_sparse_spmv()` (SpMV: sparse matrix × dense vector):
```rust
pub extern "C" fn nsl_sparse_spmv(
    a_ptr: i64,    // CSR sparse matrix [M, K]
    x_ptr: i64,    // Dense vector [K]
) -> i64           // Dense vector [M]
{
    // For each row i:
    //   sum = 0
    //   for j in row_ptr[i]..row_ptr[i+1]:
    //     sum += val[j] * x[col_ind[j]]
    //   result[i] = sum
}
```

- [ ] **4.2** Implement `nsl_sparse_spmm()` (SpMM: sparse matrix × dense matrix):
```rust
pub extern "C" fn nsl_sparse_spmm(
    a_ptr: i64,    // CSR sparse matrix [M, K]
    b_ptr: i64,    // Dense matrix [K, N]
) -> i64           // Dense matrix [M, N]
```

- [ ] **4.3** Implement `nsl_sparse_add()` (sparse + sparse → sparse):
  - Uses union merge lattice
  - Output in COO format (convert to CSR afterward if needed)

- [ ] **4.4** Implement `nsl_sparse_mul()` (sparse * sparse → sparse):
  - Uses intersection merge lattice
  - Output in COO format

### Task 5: Workspace Transformation

- [ ] **5.1** For operations that produce sparse output (sparse + sparse):
  - Allocate dense temporary workspace for one row/column
  - Compute into workspace (O(1) random access)
  - Compact workspace into output compressed format
  - This avoids O(n) insertion cost into CSR

```rust
fn sparse_add_row_workspace(
    a_row: &[f64], a_cols: &[i64],
    b_row: &[f64], b_cols: &[i64],
    workspace: &mut [f64],       // dense, size = num_cols
    workspace_flag: &mut [bool], // tracks non-zeros
) -> (Vec<i64>, Vec<f64>)       // output cols and vals
```

- [ ] **5.2** Test: sparse add of two CSR matrices produces correct result.

### Task 6: Codegen Integration

- [ ] **6.1** Update `select_sparse_kernel()` in codegen to use the new runtime operations.

- [ ] **6.2** Wire `nsl_sparse_spmv`, `nsl_sparse_spmm`, `nsl_sparse_add`, `nsl_sparse_mul` as FFI builtins.

- [ ] **6.3** Update semantic checker to infer output format from operation + input formats.

### Task 7: Cost Model Integration

- [ ] **7.1** Add sparse FLOP estimation to cost model:
  - SpMV: `2 * nnz` FLOPs (one mul + one add per non-zero)
  - SpMM: `2 * nnz * N` FLOPs
  - Sparse add: `nnz_a + nnz_b` (merge cost)

- [ ] **7.2** This connects to the existing `sparse-cost-model.md` plan.

---

## Effort Estimate

- Task 1 (CSR/CSC conversion): 1 day
- Task 2 (iteration graph): 1 day
- Task 3 (merge lattice): 1.5 days
- Task 4 (runtime kernels): 2 days
- Task 5 (workspace): 1 day
- Task 6-7 (integration): 1 day
- Total: 7-8 days
