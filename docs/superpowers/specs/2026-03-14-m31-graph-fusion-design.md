# M31: Graph-Level Operator Fusion — Design Spec

## Overview

M31 adds graph-level operator fusion to the NSL compiler, automatically combining multi-operator subgraphs into single fused kernels. This eliminates global memory round-trips between operations, reducing kernel launch count by 50-70% in transformer models.

Three fusion strategies, applied in priority order (most specific first):
1. **Reduction fusion** — replaces softmax/layernorm/rmsnorm subgraphs with pre-optimized kernels
2. **Epilogue fusion** — appends elementwise ops (bias, activation) into matmul kernels
3. **Elementwise fusion** — generic pointwise chain cleanup (M26, sweeps leftovers)

The system uses a hybrid approach: a lightweight consumer-counting DAG for correctness analysis, combined with pattern matching for fusion detection.

**Dependencies:** M26 (elementwise fusion infrastructure), M28 (dynamic shapes)

---

## Section 1: Expression DAG & Consumer Counting

### Module: `crates/nsl-codegen/src/fusion_graph.rs`

### Data Structures

```rust
type NodeId = u32;
type FusedKernelId = u32;

struct FusionGraph {
    nodes: Vec<FusionNode>,
    name_to_node: HashMap<String, NodeId>,
}

struct FusionNode {
    id: NodeId,
    name: Option<String>,          // user-given let-binding name, or None for synthetic
    op: FusionOp,
    inputs: Vec<NodeId>,
    consumers: Vec<NodeId>,
    is_graph_output: bool,         // referenced by return — must materialize
    fused_into: Option<FusedKernelId>,  // None = unclaimed, Some = already fused
    shape: Option<Vec<usize>>,     // resolved from semantic type_map
    dtype: Option<DtypeId>,        // resolved from semantic type_map
}

enum FusionOp {
    Input,                         // function parameter
    Matmul,                        // matmul(a, b)
    Elementwise(String),           // "add", "relu", "gelu", "sigmoid", etc.
    Reduction(String),             // "sum", "mean", "reduce_max", "reduce_min", "var"
    View(String),                  // "transpose", "reshape", "broadcast", "expand"
    FlashAttention,                // unfusible barrier — never fuse into or out of
    Other,                         // anything not categorized above
}
```

### ANF Linearization

The DAG builder walks the AST **recursively**, not just let-bindings. Every subexpression gets its own `NodeId`. This is standard A-Normal Form (ANF) — nested expressions are flattened so every operation is a node with explicit inputs.

Example:
```
# User writes:
let out = relu(matmul(A, B) + C)
return out

# DAG builder produces:
Node 0: Input("A")
Node 1: Input("B")
Node 2: Input("C")
Node 3: Matmul(0, 1)              // synthetic — no user name
Node 4: Elementwise("add")(3, 2)  // synthetic
Node 5: Elementwise("relu")(4)    // name = "out", is_graph_output = true
```

The builder recurses into `ExprKind::Call`, `ExprKind::BinaryOp`, `ExprKind::UnaryOp`, etc. Each compound subexpression emits a node before its parent can reference it. Named let-bindings attach the user name to the final node of their RHS expression.

Without ANF linearization, a DAG builder that only looks at named let-bindings would see `relu(matmul(A, B) + C)` as a single node, making it impossible to detect the matmul for epilogue fusion.

### Graph Output Protection

Any node referenced by a `return` statement (or the implicit last-expression return) gets `is_graph_output = true`. The fusion eligibility gate:

```rust
fn is_fusible_into_producer(&self, node: &FusionNode) -> bool {
    node.consumers.len() == 1
        && !node.is_graph_output
        && !matches!(node.op, FusionOp::FlashAttention)
        && node.fused_into.is_none()
}
```

Graph outputs must always materialize to global memory. A fused kernel that includes a graph output must write the final result — it can eliminate intermediate writes but not the output write. Without this gate, the compiler could fuse a return value into a prior kernel without materializing it, leaving the caller with a pointer to unwritten memory.

### View Tracking

`View` nodes are pass-through for data but carry layout information. During epilogue fusion analysis, when we see `Elementwise("add")(matmul_out, bias)`, we check whether any `View` nodes intervene. A broadcast-compatible view is safe for epilogue fusion — the PTX loads the operand with stride 0 along the broadcast dimension. A transpose or reshape between the matmul and the elementwise op is a fusion barrier (the memory layout no longer matches the matmul's output tiling).

### Shape Caching

Each `FusionNode` caches resolved tensor shape and dtype from the semantic analysis phase (via `type_map` lookup during DAG construction). This enables the fusion report to compute bytes saved without re-deriving shapes:

```rust
fn bytes_saved(node: &FusionNode) -> u64 {
    match (&node.shape, &node.dtype) {
        (Some(shape), Some(dtype)) => {
            let elements: u64 = shape.iter().map(|&d| d as u64).product();
            elements * dtype_byte_width(*dtype) as u64 * 2  // one write + one read eliminated
        }
        _ => 0,
    }
}
```

### Construction

Two-pass build:
1. **Forward pass**: Walk function body recursively. For each operation, allocate `NodeId`, record `op` and `inputs`, resolve shape/dtype from type_map. Register name→NodeId for let-bindings.
2. **Back-link pass**: For each node, iterate its `inputs` and append `self.id` to each input's `consumers` list. Mark return-referenced nodes as `is_graph_output = true`.

Scope is function-local. `@fuse_graph`-decorated functions always get DAG analysis. Other functions get it if they contain matmul or reduction ops.

---

## Section 2: Epilogue Fusion (Matmul + Elementwise Tail)

### Module: `crates/nsl-codegen/src/epilogue_fusion.rs`

Epilogue fusion eliminates global memory round-trips between a matmul and its elementwise consumers (bias add, activation, etc.). This is the highest-value optimization — for `relu(matmul(A, B) + bias)`, it reduces 3 global writes + 2 global reads of the full output tensor down to 1 global write.

### Pattern Detection

Walk the DAG starting from each `FusionOp::Matmul` node. Follow the consumer chain forward collecting elementwise ops as long as:
- The current node has exactly 1 consumer (or is the chain tail)
- The consumer is `FusionOp::Elementwise`
- No `View` nodes with layout-changing semantics (transpose, reshape) intervene — broadcast-only views are allowed
- The consumer is not `FlashAttention`
- The consumer's `fused_into` is `None` (not already claimed by reduction pass)

```rust
struct EpilogueChain {
    matmul_node: NodeId,
    epilogue_ops: Vec<EpilogueOp>,
    output_node: NodeId,           // last node in chain (what gets materialized)
}

enum EpilogueOp {
    BiasAdd { bias_node: NodeId, broadcast_dim: usize },
    Activation(String),            // "relu", "gelu", "sigmoid", "tanh", "silu"
    ScalarMul { scalar_node: NodeId },
    Clamp { min_node: NodeId, max_node: NodeId },
}
```

### Broadcast Dimension Resolution

The semantic analyzer determines `broadcast_dim` during shape checking. When the DAG builder encounters `Elementwise("add")(matmul_out, bias)`, it compares shapes:
- Matmul output `[M, N]`, bias `[N]` or `[1, N]` → `broadcast_dim = 0` (broadcast across rows, index by col)
- Matmul output `[M, N]`, bias `[M, 1]` → `broadcast_dim = 1` (broadcast across cols, index by row)
- Shape mismatch that isn't broadcast-compatible → fusion barrier, add stays as separate kernel

### PTX Synthesis: MMA Lane-to-Coordinate Mapping

When the matmul uses Tensor Cores (`mma.sync.aligned.m16n8k16`), the 32 warp lanes hold output fragments in an interleaved pattern — thread `t` does NOT hold `Out[t/N, t%N]`. Before any epilogue op, each thread must compute its logical `(row, col)` from `lane_id = threadIdx.x % 32` using the MMA fragment layout:

```
// m16n8k16 fragment layout (per thread holds 4 accumulator regs)
row = (lane_id / 4) + (i / 2) * 8
col = (lane_id % 4) * 2 + (i % 2)
```

For `BiasAdd`, the thread uses its fragment's `col` (if `broadcast_dim == 0`) or `row` (if `broadcast_dim == 1`) to index into the bias vector. Without this mapping, every bias element gets added to the wrong output position.

### PTX Synthesis: F32 Epilogue Mandate

Mixed-precision matmuls accumulate in f32 regardless of input dtype. **All epilogue operations execute in f32** on the existing accumulator registers. The dtype downcast happens exactly once, as the very last instruction before `st.global`:

```
mma.sync.aligned.m16n8k16.f32.f16.f16.f32  // accumulate in f32
// -- epilogue begins (all f32) --
add.f32 %acc, %acc, %bias                   // BiasAdd
mul.f32 %t, %acc, 0.7978845608              // gelu: sqrt(2/pi)
fma.rn.f32 %t, %t, %acc, ...               // gelu: tanh approx
// -- epilogue ends --
cvt.rn.f16.f32 %out, %acc                   // downcast ONCE
st.global.b16 [addr], %out                  // single global write
```

If output dtype matches accumulation (f32→f32), the `cvt` is skipped.

Computing the epilogue in f16 before the final store would cause severe precision loss — the f32 accumulator precision must be preserved through the entire epilogue chain.

### Fallback Path

If the matmul does NOT use Tensor Cores (small matrices below MMA threshold, non-aligned dimensions), the epilogue still applies but uses standard linear thread-to-element mapping (`tid = blockIdx.x * blockDim.x + threadIdx.x`). The f32 mandate and broadcast dimension logic are the same regardless.

### Fusion Barriers

The epilogue chain stops when:
- Consumer is a `Reduction` (can't do reductions in epilogue)
- Consumer is `Matmul` (next matmul — that's a new chain)
- Consumer has multiple consumers (must materialize for the other path)
- Consumer is a `View("transpose")` or `View("reshape")` (layout mismatch)
- Consumer is `FlashAttention`
- Consumer has `@no_fuse` annotation
- Consumer is already claimed by a prior fusion pass (`fused_into.is_some()`)

---

## Section 3: Reduction Fusion (Softmax, LayerNorm, RMSNorm)

### Module: `crates/nsl-codegen/src/reduction_fusion.rs`

Reduction fusion replaces multi-kernel reduction patterns with single pre-optimized kernels. Unlike epilogue fusion (which extends an existing kernel), reduction fusion substitutes an entire subgraph with a purpose-built kernel.

### Pattern Library

Three patterns, detected by walking the DAG and matching subgraph shapes:

**1. Softmax**: `exp(x - reduce_max(x)) / reduce_sum(exp(x - reduce_max(x)))`

The compiler detects BOTH the numerically stable form and the naive form (`exp(x) / sum(exp(x))`). The naive form is silently replaced with the stable version — this is a correctness fix, not just an optimization. Naive softmax overflows to `inf` for large values (common in attention logits).

DAG pattern (stable form):
```
Node A: Reduction("reduce_max")(x)           // optional — absent in naive form
Node B: Elementwise("sub")(x, A)             // optional — absent in naive form
Node C: Elementwise("exp")(B or x)
Node D: Reduction("reduce_sum")(C)
Node E: Elementwise("div")(C, D)
```

**2. LayerNorm**: `(x - mean(x)) / sqrt(var(x) + eps) * gamma + beta`

DAG pattern:
```
Node A: Reduction("mean")(x)
Node B: Elementwise("sub")(x, A)
Node C: Reduction("var")(x)                  // or manual: mean((x - mean)^2)
Node D: Elementwise("add")(C, eps)
Node E: Elementwise("sqrt")(D)
Node F: Elementwise("div")(B, E)
Node G: Elementwise("mul")(F, gamma)         // optional affine
Node H: Elementwise("add")(G, beta)          // optional affine
```

**3. RMSNorm**: `x / sqrt(mean(x^2) + eps) * gamma`

DAG pattern:
```
Node A: Elementwise("mul")(x, x)             // x^2 (or pow(x, 2))
Node B: Reduction("mean")(A)
Node C: Elementwise("add")(B, eps)
Node D: Elementwise("sqrt")(C)
Node E: Elementwise("div")(x, D)
Node F: Elementwise("mul")(E, gamma)         // optional affine
```

### Pattern Matching Infrastructure

```rust
struct ReductionPattern {
    name: &'static str,
    match_fn: fn(&FusionGraph, NodeId) -> Option<ReductionMatch>,
}

struct ReductionMatch {
    pattern: &'static str,
    root_node: NodeId,
    input_nodes: Vec<NodeId>,              // x, gamma, beta, eps as applicable
    all_matched_nodes: Vec<NodeId>,
    reduction_dim: i64,                    // must be consistent across all reductions
    is_naive: bool,                        // softmax only: was it the unsafe form?
}
```

### Internal vs. External Consumer Rule

The single-consumer constraint applies to consumers **outside** the matched subgraph, not within it. This is critical for LayerNorm where `mean(x)` is consumed by both `sub(x, mean)` and `var(x)` — both are internal to the subgraph.

```rust
fn is_valid_reduction_match(graph: &FusionGraph, matched: &ReductionMatch) -> bool {
    let matched_set: HashSet<NodeId> = matched.all_matched_nodes.iter().copied().collect();

    for &node_id in &matched.all_matched_nodes {
        let node = &graph.nodes[node_id as usize];
        for &consumer in &node.consumers {
            if !matched_set.contains(&consumer) {
                // External consumer — only the subgraph's output node may have them
                if node_id != matched.root_node {
                    return false;
                }
            }
        }
    }
    true
}
```

### Reduction Dimension Verification

Each `match_fn` extracts the reduction dimension from every reduction node in the candidate subgraph and asserts they are identical. If two reduction nodes operate on different dimensions, the match aborts immediately.

Additionally, fused kernels require the reduction to operate on the **last contiguous dimension** (`dim == -1` or `dim == rank - 1`). Non-contiguous dimension reductions are rejected — the ops stay as separate kernels.

This prevents the corruption case where `reduce_max(x, dim=0)` and `reduce_sum(x, dim=1)` would be incorrectly fused into a single softmax.

### Hand-Written PTX Templates

Following the M27 FlashAttention playbook, reduction kernels use **hand-written parameterized PTX string templates** — not the general-purpose `KernelCompiler` from `kernel.rs`. The `KernelCompiler` was built for 1D/2D elementwise ops and cannot synthesize two-pass loop structures, block synchronization (`bar.sync`), or warp-level butterfly shuffles (`shfl.sync.down.b32`).

```rust
fn synthesize_fused_softmax_ptx(hidden_dim: usize, dtype: DtypeId) -> Vec<u8> { ... }
fn synthesize_fused_layernorm_ptx(hidden_dim: usize, has_affine: bool, eps: f32, dtype: DtypeId) -> Vec<u8> { ... }
fn synthesize_fused_rmsnorm_ptx(hidden_dim: usize, has_affine: bool, eps: f32, dtype: DtypeId) -> Vec<u8> { ... }
```

Each template is parameterized by:
- `hidden_dim` — determines strategy: warp-only (dim <= 32), single-block (dim <= 1024), multi-block (dim > 1024)
- `has_affine` — whether gamma/beta scaling is applied
- `dtype` — output dtype for final `cvt` + `st.global`

Template structure:
- Warp-level reductions via `shfl.sync.down.b32` for partial sums within a warp
- `bar.sync 0` for cross-warp synchronization within a block
- Dynamic shared memory for inter-warp communication
- **Softmax**: Two-pass — pass 1 computes max and sum, pass 2 applies `exp(x - max) / sum`
- **LayerNorm**: Welford's online algorithm for numerically stable mean+variance, then normalize + affine
- **RMSNorm**: Single pass computes mean of squares, second pass normalizes and scales

### Naive Softmax Diagnostic

When the compiler detects and replaces naive softmax, it emits a warning on stderr:

```
warning: numerically unstable softmax replaced with safe max-subtraction form
  --> model.nsl:42
   |
42 |     let attn = exp(scores) / sum(exp(scores))
   |                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   = note: naive exp(x)/sum(exp(x)) overflows for large values
   = help: use softmax(x) or exp(x - max(x)) / sum(exp(x - max(x)))
```

---

## Section 4: `@fuse_graph` Decorator & Fusion Report

### `@fuse_graph` Decorator

User-decorated functions get all fusion strategies applied to their body:

```python
@fuse_graph
fn swiglu(x, W_gate, W_up):
    let gate = sigmoid(matmul(x, W_gate))
    let up = matmul(x, W_up)
    return gate * up
```

Processing pipeline:
1. Semantic checker validates `@fuse_graph` is on a function (not model, not kernel)
2. During codegen, compiler builds a `FusionGraph` for the function body
3. All three fusion passes run in priority order over the graph:
   - **Reduction fusion** (largest subgraphs — softmax, layernorm, rmsnorm)
   - **Epilogue fusion** (matmul + single-consumer elementwise tail)
   - **Elementwise fusion** (generic pointwise chain cleanup — sweeps leftovers)
4. Best-effort: whatever fuses, fuses. Whatever hits a barrier stays as separate kernels.

**Pass ordering is critical.** The passes run from most specific to most generic. Each pass marks consumed nodes via `fused_into`. If elementwise fusion ran first, it would greedily swallow `exp` and `div` nodes before the reduction pass could recognize them as part of a softmax pattern. Running reduction first ensures the largest patterns are claimed before the generic cleanup pass.

### `@no_fuse` Annotation

Applied to individual let-bindings to force materialization:

```python
@fuse_graph
fn debug_forward(x, W):
    @no_fuse
    let intermediate = matmul(x, W)    # force materialization
    let out = relu(intermediate)
    return out
```

The DAG builder marks `@no_fuse` nodes as fusion barriers. The node's output must materialize to global memory regardless of consumer count.

### `@fuse_graph` as Logging Gate

All functions with matmul/reduction ops get fusion optimization regardless of `@fuse_graph`. The decorator controls reporting verbosity:

- **`@fuse_graph` present**: Always collects `FusionEvent`/`FusionBarrierEvent` entries, even without `--fusion-report`. The function always appears in the fusion report.
- **`@fuse_graph` absent, `--fusion-report` enabled**: Collects and reports events for all functions (global verbose mode).
- **`@fuse_graph` absent, `--fusion-report` disabled**: Fusion passes run for optimization but event structs are not allocated. Zero overhead for the common case.

```rust
fn should_collect_events(&self, has_fuse_graph_decorator: bool) -> bool {
    has_fuse_graph_decorator || self.fusion_report_enabled
}
```

### Fusion Report (`--fusion-report`)

New CLI flag on `nsl build`:

```rust
/// Show fusion optimization report
#[arg(long)]
fusion_report: bool,
```

Data structures:

```rust
struct FusionEvent {
    function_name: String,
    strategy: FusionStrategy,
    matched_ops: Vec<String>,
    eliminated_launches: u32,
    estimated_bytes_saved: u64,
    location: String,
}

struct FusionBarrierEvent {
    function_name: String,
    reason: BarrierReason,
    node_description: String,
    location: String,
}

enum FusionStrategy { Epilogue, Reduction, Elementwise }

enum BarrierReason {
    MultiConsumer,
    LayoutChange,
    FlashAttention,
    NoFuseAnnotation,
    DimensionMismatch,
    UnsupportedOp,
}
```

Report output (stderr):

```
Fusion Report:
  swiglu (model.nsl:15):
    matmul(x, W_gate) + sigmoid -> FUSED (epilogue)
      Savings: 1 eliminated launch, ~32MB eliminated traffic
    matmul(x, W_up) -> not fused (multi-consumer barrier)

  attention (model.nsl:28):
    exp(scores) / sum(exp(scores)) -> FUSED (reduction: softmax)
      Warning: naive softmax replaced with numerically stable form
      Savings: 4 eliminated launches, ~16MB eliminated traffic
    matmul(attn_weights, V) + bias -> FUSED (epilogue)
      Savings: 1 eliminated launch, ~64MB eliminated traffic

  Summary: 8 opportunities found, 6 applied, 2 barriers (multi-consumer)
```

Compiler integration:

```rust
// On Compiler struct:
pub fusion_events: Vec<FusionEvent>,
pub fusion_barriers: Vec<FusionBarrierEvent>,
pub fusion_report_enabled: bool,
```

`CompileOptions` gets `pub fusion_report: bool` (default `false`).

---

## Section 5: Testing, Dynamic Shapes, and Integration

### Testing Strategy

**Unit tests per module:**

- **`fusion_graph.rs`**:
  - Nested expression ANF linearization (`relu(matmul(A, B) + C)` → 6 nodes)
  - Consumer counting across let-bindings
  - `is_graph_output` correctly set for return-referenced nodes
  - View nodes tracked for transpose/reshape/broadcast

- **`epilogue_fusion.rs`**:
  - `matmul + bias_add + relu` → single `EpilogueChain` with 2 epilogue ops
  - `matmul + bias_add` where bias consumed elsewhere → chain stops (multi-consumer)
  - `matmul + transpose + relu` → no fusion (layout change barrier)
  - Broadcast dimension correctly resolved for `[M,N] + [N]` vs `[M,N] + [M,1]`
  - `@no_fuse` on intermediate → chain breaks

- **`reduction_fusion.rs`**:
  - Naive softmax detected, `is_naive = true`
  - Stable softmax detected, `is_naive = false`
  - LayerNorm with internal multi-consumer (mean used by sub and var) → valid match
  - LayerNorm where mean also consumed outside subgraph → match rejected
  - Mismatched reduction dims (max on dim=0, sum on dim=1) → match rejected
  - Non-last-dimension reduction → match rejected
  - RMSNorm with and without affine

- **Pass ordering tests**:
  - Softmax pattern survives when all three passes run (reduction claims before elementwise)
  - Nodes with `fused_into.is_some()` are skipped by later passes

### E2E Tests

- **`examples/m31_epilogue_fusion.nsl`** — matmul + bias + activation, verifies correct output
- **`examples/m31_reduction_fusion.nsl`** — softmax and layernorm patterns, verifies naive softmax warning on stderr
- **`examples/m31_fuse_graph.nsl`** — `@fuse_graph` decorated function with `@no_fuse` escape hatch

### Dynamic Shapes (M28 Compatibility)

- **Epilogue fusion**: Matmul kernel already handles dynamic M/N/K via runtime grid calculation. Epilogue ops are per-element and shape-agnostic. No additional work needed.

- **Reduction fusion**: PTX templates are parameterized by `hidden_dim`. For static shapes, this is a compile-time constant. For dynamic shapes (symbolic dimensions), the compiler emits a **runtime dispatch table** — pre-compiled PTX variants for common sizes (128, 256, 512, 1024, 2048, 4096, 8192) plus a fallback unfused path for unmatched sizes.

```rust
struct DynamicReductionDispatch {
    variants: Vec<(usize, Vec<u8>)>,   // (hidden_dim, ptx_bytes)
    fallback_unfused: bool,
}
```

### File Structure

**New files:**

| File | Responsibility |
|---|---|
| `crates/nsl-codegen/src/fusion_graph.rs` | FusionGraph, FusionNode, ANF linearization, consumer counting |
| `crates/nsl-codegen/src/epilogue_fusion.rs` | EpilogueChain detection, MMA-aware PTX synthesis |
| `crates/nsl-codegen/src/reduction_fusion.rs` | Pattern library, hand-written PTX templates, naive softmax warning |
| `crates/nsl-codegen/src/fusion_report.rs` | FusionEvent collection, `--fusion-report` formatting |

**Modified files:**

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Module declarations, `fusion_report` in `CompileOptions` |
| `crates/nsl-codegen/src/compiler.rs` | `fusion_events`, `fusion_barriers`, `fusion_report_enabled` fields; 3-pass fusion pipeline |
| `crates/nsl-codegen/src/expr.rs` | `try_auto_fuse()` emits fused kernel launches from DAG |
| `crates/nsl-codegen/src/fusion.rs` | Respect `fused_into` so M26 pass skips claimed nodes |
| `crates/nsl-semantic/src/checker.rs` | Validate `@fuse_graph` and `@no_fuse` decorators |
| `crates/nsl-cli/src/main.rs` | `--fusion-report` flag, wire to `CompileOptions` |
| `crates/nsl-cli/tests/e2e.rs` | M31 E2E test functions |

### Out of Scope

- Cross-function fusion (each function is an independent DAG)
- Cross-layer fusion (model forward passes are separate function calls)
- Training-time backward pass fusion
- Fusing into FlashAttention kernels (always a barrier)
- Automatic discovery of novel reduction patterns beyond the three in the library
- Fusing across control flow (`if`/`else` branches)
