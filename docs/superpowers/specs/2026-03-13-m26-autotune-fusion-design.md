# M26 Design Spec: @autotune + Elementwise Operator Fusion + Kernel Profiling

**Date:** 2026-03-13
**Status:** Approved
**Milestone:** M26
**Prerequisite:** M25 (PagedAttention + Memory Profiling)
**Dependency:** M27 (FlashAttention-2) uses autotune; M31 (Graph Fusion) generalizes fusion infrastructure
**Build Order:** Kernel Profiler -> @fuse/Auto-Fusion -> @autotune

## Overview

Three tightly coupled compiler+runtime features that make GPU code fast without hand-tuning:

1. **Kernel Profiler** — async GPU event recording with zero-sync overhead, Chrome tracing output, unified CPU+GPU timeline
2. **@fuse + Auto-Fusion** — merge elementwise op chains into single PTX kernels, eliminating global memory round-trips
3. **@autotune** — build-time kernel variant benchmarking with caching, selects fastest variant for the target hardware

These are sequenced in dependency order: @autotune's benchmarking harness consumes the kernel profiler infrastructure, and @fuse produces the fused kernels that @autotune can optimize.

---

## Part 1: Kernel Profiler (`kernel_profiler.rs`)

### Purpose

Async, zero-sync GPU kernel timing with pre-allocated event pools. Serves two roles:
1. User-facing `--profile-kernels` CLI flag for performance debugging
2. Internal infrastructure for @autotune's build-time benchmarking harness

### Architecture

```
KernelProfiler (global singleton, behind Mutex)
+-- enabled: AtomicBool
+-- cpu_start_time: Instant              // recorded at profiler_start()
+-- gpu_base_event: CUevent              // enqueued on default stream at start
+-- event_pool: Vec<(CUevent, CUevent)>  // pre-allocated 4096 start/stop pairs
+-- pool_cursor: usize                   // next free pair index
+-- traces: Vec<KernelTrace>             // (pool_idx, name, grid, block)
+-- flush() -> Vec<ChromeTraceEvent>     // single cuCtxSynchronize, resolve all deltas
```

### Event Lifecycle

1. **`nsl_kernel_profiler_start()`** — allocate 4096 event pairs via `cuEventCreate`, record `cpu_start_time = Instant::now()`, enqueue `cuEventRecord(gpu_base_event, stream_0)`
2. **`nsl_kernel_profiler_record(name, grid, block, stream)`** — called from `kernel_launch()` in `cuda/mod.rs`:
   - **Lock** mutex, pop next `(start, stop)` pair, bump `pool_cursor`, **unlock**
   - `cuEventRecord(start, stream)` — no lock held
   - `cuLaunchKernel(...)` — no lock held (caller does this)
   - `cuEventRecord(stop, stream)` — no lock held
   - **Lock** mutex, push `KernelTrace`, **unlock**
   - Critical: two-phase lock prevents mutex poisoning if CUDA driver panics
3. **`nsl_kernel_profiler_flush(path)`** — consuming operation:
   - `cuCtxSynchronize()` — single sync, drains GPU pipeline
   - For each trace: `cuEventElapsedTime(start, stop)` for duration, `cuEventElapsedTime(gpu_base_event, start)` for offset
   - Compute unified timestamp: `ts_us = cpu_start_time_us + gpu_offset_ms * 1000.0`
   - Write Chrome tracing JSON
   - `cuEventDestroy` on all `4096 * 2` events + `gpu_base_event` — full resource cleanup
   - Reset pool to empty state
4. **Pool exhaustion:** if `pool_cursor >= 4096`, flush and recycle (single sync penalty, warning to stderr)

### Stream Safety

The `record()` function accepts `CUstream` as an argument. All events for a kernel launch are recorded on the same stream as the kernel itself. The `gpu_base_event` is recorded on stream 0 at startup. For M26's single-stream-per-thread model, this guarantees monotonic timestamp alignment. Multi-stream scheduling (M30) would require `cuStreamWaitEvent` barriers.

### Clock Synchronization

The memory profiler (M25) uses CPU timestamps (`std::time::Instant`). The kernel profiler uses GPU event elapsed times. To merge both into a single Chrome tracing timeline:

```
unified_ts_us = cpu_start_time_us + cuEventElapsedTime(gpu_base_event, kernel_start_event) * 1000.0
```

This anchors the GPU timeline to the CPU timeline via a single reference point recorded at startup.

### FFI Exports

| Function | Signature | Purpose |
|----------|-----------|---------|
| `nsl_kernel_profiler_start` | `() -> ()` | Init event pool + base event |
| `nsl_kernel_profiler_stop` | `() -> ()` | Disable recording (no cleanup) |
| `nsl_kernel_profiler_record` | `(name_ptr, name_len, grid, block, stream) -> ()` | Two-phase lock event recording |
| `nsl_kernel_profiler_flush` | `(path_ptr, path_len) -> ()` | Sync + dump JSON + cuEventDestroy all |

### Integration with `cuda/mod.rs`

The profiler is integrated *inside* `kernel_launch()`, not wrapping it externally. The existing `kernel_launch()` holds a `Mutex<CudaState>` for module cache lookup, function lookup, launch, and synchronize. The profiler event recording is injected between the lock-protected phases:

```rust
fn kernel_launch(...) {
    let state = CUDA_STATE.lock();
    let module = state.get_or_load_module(ptx);
    let func = state.get_function(module, name);
    drop(state);  // release lock before CUDA calls

    // Profiler: record start event (no lock held)
    let events = if kernel_profiler_enabled() {
        Some(kernel_profiler_pop_events())  // lock-pop-unlock on profiler mutex
    } else { None };

    if let Some((start, _)) = &events {
        cuEventRecord(*start, stream);
    }

    cuLaunchKernel(func, ...);  // no lock held

    if let Some((_, stop)) = &events {
        cuEventRecord(*stop, stream);
        kernel_profiler_push_trace(name, grid, block);  // lock-push-unlock
    }

    // NOTE: existing cuCtxSynchronize removed — see "Per-Launch Sync Removal" below
}
```

**Per-Launch Sync Removal:** The existing `kernel_launch()` calls `cuCtxSynchronize()` after every launch. This was added in M17 for correctness during initial GPU bring-up. M26 removes this per-launch sync as a prerequisite change — it is safe because unified memory (`cuMemAllocManaged`) provides coherence guarantees, and the profiler's `flush()` performs a single sync at program end. This is a significant performance improvement independent of profiling. Without removing it, the "zero-sync overhead" profiling design is moot since every launch already syncs.

**Stream parameter:** In M26, `stream` is always `null_mut()` (CUDA default stream / stream 0). The parameter exists in the profiler API for forward-compatibility with M30's multi-stream scheduling. All events (including `gpu_base_event`) are recorded on the default stream, guaranteeing monotonic timestamp alignment.

### Output Format

Merges with M25's `profile.json`. Memory events on `tid: 0`, kernel events on `tid: 1` — separate tracks in Chrome tracing:

```json
{
  "traceEvents": [
    {"name": "block_alloc", "ph": "X", "ts": 100, "dur": 0, "pid": 0, "tid": 0},
    {"name": "matmul_256", "ph": "X", "ts": 150, "dur": 500, "pid": 0, "tid": 1},
    {"name": "fused_relu_add", "ph": "X", "ts": 700, "dur": 200, "pid": 0, "tid": 1}
  ],
  "metadata": {
    "peak_blocks": 42,
    "total_kernel_launches": 156,
    "total_kernel_time_ms": 82.5
  }
}
```

Chrome tracing visualization:
```
Track 0: [Memory]  == alloc_block  == alloc_block      == free_seq
Track 1: [Kernels] === matmul(256) = fused_relu_add ===== paged_attn
```

### Implementation File

`crates/nsl-runtime/src/kernel_profiler.rs` — isolated from `profiling.rs` (M25). Different data lifecycles: M25 uses CPU atomic counters, M26 uses async GPU event pairs. Merged only at JSON dump time.

---

## Part 2: Elementwise Fusion (`@fuse` + Lexical Auto-Fusion)

### Purpose

Reduce kernel launch overhead and global memory traffic by fusing chains of elementwise operations into single PTX kernels. Two modes: explicit `@fuse` decorator and opportunistic lexical auto-fusion.

### 2a: Explicit `@fuse` Decorator

```nsl
@fuse
fn gelu_approx(x: Tensor<[N], f32>) -> Tensor<[N], f32>:
    let t = sqrt(2.0 / pi) * (x + 0.044715 * x ** 3)
    return 0.5 * x * (1.0 + tanh(t))
```

**Compiler behavior:**
- Semantic checker validates `@fuse` placement: allowed on `fn` declarations only
- `@fuse` on a `kernel` block → compile error: `"@fuse cannot be applied to kernel blocks (kernel blocks are already single PTX kernels)"`
- `@fuse` on a `model` method → compile error: `"@fuse cannot be applied to model methods; extract the fusible logic into a standalone fn"`
- `@fuse` on any other target → compile error: `"@fuse can only be applied to fn declarations"`
- Verifies all ops in the function body are in the fusible set (see below)
- If a non-fusible op is found, emit compile error: `"@fuse function contains non-fusible operation: matmul"`
- Codegen synthesizes a single PTX kernel for the entire function body
- `let` bindings within `@fuse` map to PTX registers (no global memory round-trip)
- Input tensors: one global memory read each. Output: one global memory write.

**Fusible ops:** `+`, `-`, `*`, `/`, `**`, `neg`, `abs`, `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`, `sign`, `clamp`, scalar broadcast, dtype cast.

**Not fusible (compile error inside @fuse):** `matmul`, `sum`, `mean`, reductions, `reshape`/`transpose`, `conv`.

### 2b: Lexical Auto-Fusion (No DAG Infrastructure)

**Architectural boundary:** Post-order AST traversal during codegen. `let` binding = hard fusion barrier. No dataflow graph, no liveness analysis — that's M31's scope.

**Rule:** If a node is an elementwise op and its operand is also an inline elementwise op (not assigned to a `let` binding), accumulate into the same fused kernel.

**What auto-fuses:**
```nsl
let y = relu(x @ w + b)
# matmul is non-fusible -> emits kernel, writes to global mem
# relu(matmul_result + b) -> both elementwise, inline -> fused into one PTX kernel
```

**What bails out:**
```nsl
let temp = x @ w + b    # Add fused with bias broadcast, writes result to global mem
let y = relu(temp)       # Separate kernel launch — 'let' is a fusion barrier
```

**Why this is the right boundary for M26:**
- Captures the 80/20 win: `matmul + bias + activation` is the most common pattern in ML
- Zero DAG infrastructure cost — just AST node inspection
- M31 inherits full graph analysis with liveness tracking and multi-consumer handling

### Broadcast Indexing (Multi-Dimensional)

When fusing ops with different-shaped inputs, the PTX synthesizer must handle broadcasting correctly:

1. Decompose flat `tid` into N-dimensional coordinates: `coord[i] = (tid / stride[i]) % dim[i]`
2. Apply broadcast rule: if input dimension is 1, coordinate becomes 0
3. Re-flatten into load index using input's actual strides: `load_idx = sum(coord[i] * input_stride[i])`

Same-shape inputs use direct flat load (zero overhead). Only broadcast inputs pay the decomposition cost.

**No naive 1D modulo.** The flat `tid % dim_size` shortcut only works for 1D-to-1D broadcasting and silently produces wrong results for multi-dimensional cases like broadcasting `[4, 1]` into `[4, 8]`.

### Autodiff Safety (Critical)

**The problem:** NSL's M12 autodiff engine records operations to a tape at runtime. If auto-fusion merges `relu(add(x, b))` into a single kernel, the individual `add` and `relu` calls never execute, nothing gets pushed to the tape, and `.backward()` produces incorrect gradients.

**Fix for auto-fusion — runtime branch:**
```
if nsl_is_training():
    # Unfused path: standard runtime calls, tape populated, intermediates saved
    let temp = nsl_tensor_add(matmul_res, b)
    let out = nsl_tensor_relu(temp)
else:
    # Fused path: single kernel, registers only, zero tape overhead
    let out = launch_fused_add_relu(matmul_res, b)
```

The `try_fuse_expr` codegen pass emits this branch as a standard Cranelift `brif`/merge-block pattern — the same infrastructure already used for `match` expressions and `if/else` in the existing codegen. Both branches produce a tensor pointer that merges at the join block. No novel control-flow infrastructure required.

Training correctness is preserved; inference gets full fusion speed.

**Fix for explicit @fuse — backward annotation required:**
- If `@fuse fn foo()` is called during training and no `@backward(foo)` exists → runtime panic: `"@fuse function 'foo' called during training without @backward annotation"`
- If `@backward(foo)` is provided, training path uses the user's custom backward function
- Inference path always uses the fused kernel regardless

### PTX Synthesis for Fused Kernels

**Dtype:** Fused kernels are GPU-only and operate exclusively on f32 (dtype=1). Tensors that arrive via `.to(device)` are already converted to f32 by the M17 device transfer logic. The PTX synthesizer emits `.f32` instructions unconditionally. If a future milestone adds GPU f16/bf16 support, the fusion PTX generator would need dtype dispatch, but for M26 f32-only is correct.

Thread ID = flat element index: `tid = blockIdx.x * blockDim.x + threadIdx.x`

For each input tensor:
- Compute load index with stride-aware broadcast (see above)
- `ld.global.f32 %input_reg, [input_ptr + load_idx * 4]`

All intermediate ops: register-to-register PTX arithmetic (no global memory)

Final result: `st.global.f32 [output_ptr + tid * 4], %result_reg`

### Implementation

- **Semantic validation:** `crates/nsl-semantic/src/checker.rs` — validate `@fuse` decorator placement and fusible-op constraint
- **Fusion pass:** `crates/nsl-codegen/src/fusion.rs` — `try_fuse_expr(expr) -> Option<FusedKernel>` walks expr tree, collects fusible chain, returns `None` if chain length is 1
- **Integration:** `crates/nsl-codegen/src/expr.rs` — when compiling tensor expressions, call `try_fuse_expr` first; if fusion succeeds, emit fused kernel instead of per-op dispatch
- **Fused kernel naming:** Auto-fused kernels are named by concatenating the fused ops: `"fused_add_relu"`, `"fused_mul_add_sigmoid"`. Explicit `@fuse` kernels use the function name: `"fused_gelu_approx"`. These names appear in profiler traces and error messages.

---

## Part 3: @autotune Build-Time Kernel Benchmarking

### Purpose

At `nsl build` time, generate all kernel variants from tuning parameter ranges, benchmark each on real hardware, cache the winner, and compile only the fastest variant into the final binary.

### Syntax

```nsl
@autotune(
    block_size=[64, 128, 256],
    warps=[2, 4, 8],
    stages=[1, 2]
)
kernel matmul(a: Tensor<[M, K], f32>, b: Tensor<[K, N], f32>) -> Tensor<[M, N], f32>:
    # block_size, warps, stages are compile-time constants inside the kernel body
    ...
```

### Compilation Pipeline

```
Parse @autotune decorator
    |
    v
Extract tuning params -> Cartesian product (e.g., 3x3x2 = 18 variants)
    |
    v
Hash kernel AST body (NOT PTX) -> check .nsl-cache/autotune/
    |
    +-- Cache HIT:  compile ONLY winning variant's PTX -> embed in .rodata
    |
    +-- Cache MISS:
            |
            v
        [If GPU available at build time]
            For each variant:
                Substitute constants into kernel AST
                KernelCompiler::compile() -> PTX string
            |
            v
            Benchmarking harness:
                1. Select device: env NSL_AUTOTUNE_DEVICE (default 0)
                2. Allocate synthetic input tensors (random f32)
                3. For each variant's PTX:
                    - cuModuleLoadData (catch errors -> skip variant)
                    - 3 warmup launches (discarded -- GPU P-state ramp)
                    - 10 measured launches (cuEventRecord on same stream)
                    - Compute median elapsed time
                    - Catch CUDA errors -> record f64::INFINITY, continue
                4. Select variant with lowest median time
                5. Write autotune_report.json to .nsl-cache/autotune/
            |
            v
            Embed ONLY winning variant's PTX in .rodata
        |
        [If no GPU at build time OR --no-autotune]
            Fallback: select middle value of each range
            (block_size=128, warps=4, stages=1)
            Embed single variant's PTX in .rodata
```

### Cache Strategy

**Location:** `.nsl-cache/autotune/`

**Cache key:** `SHA-256(kernel_name + hash(kernel_AST_body) + tuning_ranges + input_shapes + device_name + compute_capability + sm_count)`

Critical: the cache key hashes the **AST**, not the PTX. This allows the compiler to check the cache *before* any PTX compilation. On cache hit, only the winning variant's PTX is generated — 17 out of 18 compilations are skipped entirely.

**AST hashing strategy:** The hash must be computed over the *semantic content* of the kernel body, not its syntactic representation:
- `Span` fields (source locations) are **excluded** — reformatting source code must not invalidate the cache
- `Symbol` fields (string interner indices) are **resolved to string values** before hashing — interner indices are assigned fresh each compilation session and are not stable across runs
- The hash walks the AST in canonical order, serializing node types, resolved symbol names, literal values, and operator kinds into a deterministic byte sequence, then SHA-256s the result

The SM count is queried via `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)` and included in the cache key to differentiate laptop vs desktop silicon with identical device names.

**Cache value** (`.nsl-cache/autotune/<kernel_name>_<hash>.json`):
```json
{
  "kernel": "matmul",
  "device": "NVIDIA RTX 4090",
  "compute_capability": "8.9",
  "sm_count": 128,
  "variants_tested": 18,
  "winner": {"block_size": 256, "warps": 8, "stages": 2},
  "median_time_ms": 0.42,
  "all_timings": [
    {"params": {"block_size": 64, "warps": 2, "stages": 1}, "median_ms": 1.87},
    {"params": {"block_size": 64, "warps": 2, "stages": 2}, "median_ms": 1.54}
  ],
  "timestamp": "2026-03-13T14:30:00Z",
  "cache_key": "a1b2c3d4..."
}
```

**Cache management flags:**
- `--autotune-fresh` — force re-benchmark (ignore cache)
- `--autotune-clean` — delete `.nsl-cache/autotune/` directory

### Warmup Strategy (Mandatory)

**3 warmup iterations (discarded) + 10 measured iterations.**

Modern GPUs aggressively downclock when idle (P8 state). The first kernel launch triggers a clock ramp to P0/P2, taking several milliseconds. Without warmup, the first variant tested pays an artificial "wake-up tax" and appears slower, causing the compiler to select the wrong kernel.

The warmup cost is paid once per cache miss. Subsequent builds with unchanged kernels skip benchmarking entirely.

### Device Selection

`NSL_AUTOTUNE_DEVICE` environment variable, default `0`.

This avoids the "display GPU trap": many ML researchers run a cheap GPU in slot 0 (driving monitors) and a compute GPU in slot 1. Hardcoding device 0 would autotune for the wrong GPU.

Note: cudarc's `dynamic-linking` feature automatically respects `CUDA_VISIBLE_DEVICES`, so `CUDA_VISIBLE_DEVICES=1 nsl build` also works (device 1 becomes logical device 0).

**CUDA context isolation:** The autotune benchmarking harness creates its own CUDA context on the selected device, separate from the runtime's global `CUDA_STATE` (which hardcodes device 0 in `state()`). The harness calls `cuDeviceGet(&mut dev, autotune_device_index)` and `cuDevicePrimaryCtxRetain` directly, without touching the runtime singleton. This ensures autotune can benchmark on a different device than the runtime default without modifying `CUDA_STATE`.

### Graceful Variant Failure

The Cartesian product may produce variants that exceed hardware limits (e.g., `block_size=256, stages=4` requiring 120KB shared memory on a GPU with 96KB per SM). The benchmarking harness handles this:

- `cuModuleLoadData` failure → log warning: `"Variant 14 skipped: module load failed"`, record `f64::INFINITY`
- `cuLaunchKernel` failure → log warning: `"Variant 14 skipped: exceeds hardware resources"`, record `f64::INFINITY`
- Continue to next variant. No panics. No build crashes.

If ALL variants fail, emit a compile error with diagnostic information.

### Synthetic Tensor Allocation

- Parse kernel signature for tensor parameter shapes
- Concrete shapes: allocate exactly that size
- Symbolic dimensions (e.g., `M`, `K`, `N`): use representative size 1024. Autotune results are shape-dependent; document this in the report
- Fill with random f32 data — numerical values don't affect timing, but random avoids degenerate memory access patterns and cache-line aliasing

### Benchmarking Statistics

Use **median** (not mean) of the 10 measured iterations. Median rejects outlier iterations caused by:
- GPU thermal throttling
- OS scheduling jitter
- Background DMA transfers

### Implementation

- `crates/nsl-codegen/src/autotune.rs` — variant generation, AST hashing, cache logic, report writing
- Benchmarking harness calls into `nsl-runtime` kernel profiler FFI + CUDA driver API
- `crates/nsl-cli/src/main.rs` — CLI flags

---

## Part 4: CLI Flags

### Profiling Flags

| Flag | Env Var | Effect |
|------|---------|--------|
| `--profile-memory` | `NSL_PROFILE_MEMORY=1` | Memory watermark profiler only (M25) |
| `--profile-kernels` | `NSL_PROFILE_KERNELS=1` | Kernel event profiler only (M26) |
| `--profile` | Both set | Both profilers, merged into single `profile.json` |

Fine-grained control is essential: the memory profiler generates ~70 events per run, but the kernel profiler generates hundreds of thousands (one per kernel launch). A user debugging a memory leak doesn't want a 500MB trace file.

**Profile merge strategy:** When `--profile` (both) is used, the CLI layer is responsible for merging:
1. Memory profiler dumps its events to a temp buffer via `nsl_profiler_dump`
2. Kernel profiler dumps its events to a temp buffer via `nsl_kernel_profiler_flush`
3. CLI merges both `traceEvents` arrays into a single `profile.json` file
When only one profiler is active, it writes directly to `profile.json`. This keeps both profiler modules independent — neither knows about the other.

### Autotune Flags

| Flag | Effect |
|------|--------|
| `--no-autotune` | Skip benchmarking, use middle values (no GPU required) |
| `--autotune-fresh` | Force re-benchmark, ignore cache |
| `--autotune-clean` | Delete `.nsl-cache/autotune/` directory |

---

## Key Files

| New File | Purpose |
|----------|---------|
| `crates/nsl-runtime/src/kernel_profiler.rs` | Async GPU event profiler with pool management |
| `crates/nsl-codegen/src/fusion.rs` | @fuse validation + auto-fusion PTX synthesis |
| `crates/nsl-codegen/src/autotune.rs` | Variant generation, AST hashing, benchmarking, caching |

| Modified File | Changes |
|----------------|---------|
| `crates/nsl-runtime/src/cuda/mod.rs` | `kernel_launch()` gains profiler record hook |
| `crates/nsl-runtime/src/lib.rs` | `pub mod kernel_profiler` |
| `crates/nsl-semantic/src/checker.rs` | Validate @fuse and @autotune decorators |
| `crates/nsl-semantic/src/builtins.rs` | Register kernel profiler FFI functions |
| `crates/nsl-codegen/src/builtins.rs` | Declare kernel profiler Cranelift imports |
| `crates/nsl-codegen/src/expr.rs` | Call `try_fuse_expr` before per-op dispatch |
| `crates/nsl-codegen/src/compiler.rs` | @autotune variant generation in `compile_kernels()` — must unwrap `StmtKind::Decorated` nodes to find `@autotune` on `KernelDef` |
| `crates/nsl-cli/src/main.rs` | New CLI flags |

---

## Deliverables

1. **@autotune matmul:** 18 variants benchmarked, fastest selected, JSON report with all timings
2. **@fuse GELU:** single kernel, 3x bandwidth improvement over unfused
3. **Auto-fusion:** `relu(x * w + b)` fused without annotation (inference path)
4. **Training safety:** auto-fused code emits `nsl_is_training()` branch; @fuse requires @backward
5. **Kernel profiler:** loads in Chrome tracing alongside M25 memory watermarks, unified timeline
6. **`--no-autotune`:** succeeds on machines without GPUs (middle value fallback)
7. **`--profile-kernels`:** independent of `--profile-memory`, avoidable trace bloat

## Not in Scope

- Graph-level fusion across matmuls/reductions (M31)
- Runtime autotuning / JIT recompilation
- Cross-device tuning / multi-GPU autotune (M30)
- Full dataflow DAG with liveness analysis for auto-fusion (M31)
- Multi-stream profiling with `cuStreamWaitEvent` barriers (M30)
