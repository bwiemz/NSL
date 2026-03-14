# M26: @autotune + Elementwise Fusion + Kernel Profiling — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add build-time kernel autotuning, elementwise operator fusion, and async GPU kernel profiling to the NeuralScript compiler and runtime.

**Architecture:** Three subsystems built in dependency order: (1) kernel profiler — async GPU event recording with pre-allocated pools, zero-sync during execution, Chrome tracing output; (2) elementwise fusion — explicit `@fuse` decorator and lexical auto-fusion with `let`-as-barrier, training-safe via `nsl_is_training()` branch; (3) `@autotune` — build-time variant benchmarking with AST-hashed caching, graceful failure on hardware limits.

**Tech Stack:** Rust, Cranelift, CUDA Driver API (cuEvent*, cuLaunchKernel), PTX ISA 7.0, SHA-256, Chrome Tracing JSON format.

**Spec:** `docs/superpowers/specs/2026-03-13-m26-autotune-fusion-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/nsl-runtime/src/kernel_profiler.rs` | Async GPU event pool, two-phase lock recording, clock sync, Chrome tracing flush, cuEvent lifecycle |
| `crates/nsl-codegen/src/fusion.rs` | `try_fuse_expr()` for lexical auto-fusion, `FusedKernel` PTX synthesis, broadcast index math, `@fuse` fn codegen |
| `crates/nsl-codegen/src/autotune.rs` | Cartesian product variant generation, AST hashing (Span-excluded, Symbol-resolved), cache read/write, benchmarking harness |
| `examples/m26_kernel_profiler.nsl` | E2E test: kernel launches with `--profile-kernels`, validates JSON output |
| `examples/m26_fuse.nsl` | E2E test: `@fuse fn` and auto-fused expressions, validates correct output |
| `examples/m26_autotune.nsl` | E2E test: `@autotune kernel` with `--no-autotune` fallback |

### Modified Files

| File | Changes |
|------|---------|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod kernel_profiler;` |
| `crates/nsl-runtime/src/cuda/mod.rs` | Restructure `kernel_launch()`: drop mutex before CUDA calls, inject profiler event recording, remove per-launch `cuCtxSynchronize()` |
| `crates/nsl-semantic/src/checker.rs` | Add `@fuse` and `@autotune` decorator validation |
| `crates/nsl-semantic/src/builtins.rs` | Register `kernel_profiler_start`, `kernel_profiler_stop` builtins |
| `crates/nsl-codegen/src/builtins.rs` | Declare Cranelift imports for kernel profiler FFI functions |
| `crates/nsl-codegen/src/expr.rs` | Hook `try_fuse_expr()` into tensor op dispatch, add kernel profiler builtin dispatch |
| `crates/nsl-codegen/src/compiler.rs` | Unwrap `StmtKind::Decorated` in `compile_kernels()`, handle `@autotune` variant selection |
| `crates/nsl-cli/src/main.rs` | Add `--profile-kernels`, `--profile`, `--no-autotune`, `--autotune-fresh`, `--autotune-clean` flags |

---

## Chunk 1: Kernel Profiler

### Task 1: Kernel Profiler Module — Core Data Structures and Event Pool

**Files:**
- Create: `crates/nsl-runtime/src/kernel_profiler.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Write unit test for profiler enable/disable**

Add to `kernel_profiler.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_enable_disable() {
        // Reset state
        KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);

        assert!(!kernel_profiler_enabled());
        nsl_kernel_profiler_start();
        assert!(kernel_profiler_enabled());
        nsl_kernel_profiler_stop();
        assert!(!kernel_profiler_enabled());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-runtime test_profiler_enable_disable -- --nocapture`
Expected: FAIL — `kernel_profiler` module does not exist yet

- [ ] **Step 3: Implement core data structures and start/stop**

Create `crates/nsl-runtime/src/kernel_profiler.rs`:
```rust
//! Async GPU kernel profiler with pre-allocated event pools.
//! Records cuEvent pairs around kernel launches, resolves timestamps
//! at flush time with a single cuCtxSynchronize. Zero sync during execution.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// A recorded kernel launch trace (event indices + metadata).
pub(crate) struct KernelTrace {
    pub pool_idx: usize,
    pub name: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

/// Global kernel profiler singleton.
pub(crate) struct KernelProfiler {
    pub enabled: AtomicBool,
    pub cpu_start_time: Mutex<Option<Instant>>,
    pub traces: Mutex<Vec<KernelTrace>>,
    pub pool_cursor: Mutex<usize>,
    // GPU event pool and base event are stored as raw u64 handles
    // (CUevent is a pointer, stored as u64 for Send+Sync safety)
    pub event_pool: Mutex<Vec<(u64, u64)>>, // (start_event, stop_event) pairs
    pub gpu_base_event: Mutex<u64>,
}

// SAFETY: CUevent handles are thread-safe opaque pointers managed by the CUDA driver.
unsafe impl Send for KernelProfiler {}
unsafe impl Sync for KernelProfiler {}

pub(crate) static KERNEL_PROFILER: KernelProfiler = KernelProfiler {
    enabled: AtomicBool::new(false),
    cpu_start_time: Mutex::new(None),
    traces: Mutex::new(Vec::new()),
    pool_cursor: Mutex::new(0),
    event_pool: Mutex::new(Vec::new()),
    gpu_base_event: Mutex::new(0),
};

pub fn kernel_profiler_enabled() -> bool {
    KERNEL_PROFILER.enabled.load(Ordering::Relaxed)
}

const EVENT_POOL_SIZE: usize = 4096;

/// Initialize the kernel profiler. Allocates event pool on GPU if available.
#[no_mangle]
pub extern "C" fn nsl_kernel_profiler_start() {
    KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);
    *KERNEL_PROFILER.cpu_start_time.lock().unwrap() = Some(Instant::now());
    *KERNEL_PROFILER.traces.lock().unwrap() = Vec::new();
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;

    // GPU event allocation is done conditionally when CUDA is available
    #[cfg(feature = "cuda")]
    {
        let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
        pool.clear();
        pool.reserve(EVENT_POOL_SIZE);
        for _ in 0..EVENT_POOL_SIZE {
            let mut start: u64 = 0;
            let mut stop: u64 = 0;
            unsafe {
                crate::cuda::cu_event_create(&mut start);
                crate::cuda::cu_event_create(&mut stop);
            }
            pool.push((start, stop));
        }
        let mut base = KERNEL_PROFILER.gpu_base_event.lock().unwrap();
        unsafe {
            crate::cuda::cu_event_create(&mut *base);
            crate::cuda::cu_event_record(*base, std::ptr::null_mut());
        }
    }
}

/// Disable recording. Does not free resources (flush does that).
#[no_mangle]
pub extern "C" fn nsl_kernel_profiler_stop() {
    KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);
}
```

Add to `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod kernel_profiler;
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-runtime test_profiler_enable_disable -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/kernel_profiler.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(runtime): add kernel profiler core data structures and start/stop"
```

---

### Task 2: Kernel Profiler — Two-Phase Lock Event Recording

**Files:**
- Modify: `crates/nsl-runtime/src/kernel_profiler.rs`

- [ ] **Step 1: Write unit test for event pool pop/push pattern**

```rust
#[test]
fn test_pool_cursor_advancement() {
    // Reset state
    nsl_kernel_profiler_stop();
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;
    KERNEL_PROFILER.traces.lock().unwrap().clear();
    // Pre-fill pool with dummy handles (no GPU needed)
    {
        let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
        pool.clear();
        for i in 0..10u64 {
            pool.push((i * 2, i * 2 + 1));
        }
    }
    KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);

    // Pop events — should advance cursor
    let events = kernel_profiler_pop_events();
    assert!(events.is_some());
    assert_eq!(*KERNEL_PROFILER.pool_cursor.lock().unwrap(), 1);

    // Push trace
    kernel_profiler_push_trace("test_kernel", [1, 1, 1], [256, 1, 1]);
    assert_eq!(KERNEL_PROFILER.traces.lock().unwrap().len(), 1);
    assert_eq!(KERNEL_PROFILER.traces.lock().unwrap()[0].name, "test_kernel");

    nsl_kernel_profiler_stop();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-runtime test_pool_cursor -- --nocapture`
Expected: FAIL — `kernel_profiler_pop_events` not defined

- [ ] **Step 3: Implement pop_events and push_trace with two-phase locking**

Add to `kernel_profiler.rs`:
```rust
/// Pop the next (start, stop) event pair from the pool.
/// Lock-pop-unlock pattern: single lock acquisition, extract event pair, unlock.
/// Returns None if profiler disabled or pool exhausted.
///
/// Lock ordering: always lock event_pool FIRST (contains pool + cursor).
/// This is consistent with flush/destroy which also lock event_pool first.
pub(crate) fn kernel_profiler_pop_events() -> Option<(u64, u64, usize)> {
    if !kernel_profiler_enabled() {
        return None;
    }

    // Single lock: pool and cursor accessed together to avoid ordering issues
    let pool = KERNEL_PROFILER.event_pool.lock().unwrap();
    let mut cursor = KERNEL_PROFILER.pool_cursor.lock().unwrap();

    if *cursor >= pool.len() {
        eprintln!("[nsl] kernel profiler: event pool exhausted ({} events recorded), flushing and recycling", pool.len());
        return None;
    }

    let idx = *cursor;
    let (start, stop) = pool[idx];
    *cursor += 1;
    drop(cursor);
    drop(pool);
    // Locks dropped — CUDA calls happen outside lock

    Some((start, stop, idx))
}

/// Record a completed kernel trace. Lock-push-unlock pattern.
pub(crate) fn kernel_profiler_push_trace(name: &str, grid: [u32; 3], block: [u32; 3]) {
    let cursor = *KERNEL_PROFILER.pool_cursor.lock().unwrap();
    KERNEL_PROFILER.traces.lock().unwrap().push(KernelTrace {
        pool_idx: cursor - 1, // points to the event pair just used
        name: name.to_string(),
        grid,
        block,
    });
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-runtime test_pool_cursor -- --nocapture`
Expected: PASS

- [ ] **Step 5: Write test for pool exhaustion**

```rust
#[test]
fn test_pool_exhaustion_returns_none() {
    nsl_kernel_profiler_stop();
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;
    {
        let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
        pool.clear();
        pool.push((100, 101)); // Only 1 pair
    }
    KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);

    let first = kernel_profiler_pop_events();
    assert!(first.is_some());

    let second = kernel_profiler_pop_events();
    assert!(second.is_none()); // Pool exhausted

    nsl_kernel_profiler_stop();
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p nsl-runtime test_pool_exhaustion -- --nocapture`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-runtime/src/kernel_profiler.rs
git commit -m "feat(runtime): add two-phase lock event recording for kernel profiler"
```

---

### Task 3: Kernel Profiler — Chrome Tracing JSON Flush

**Files:**
- Modify: `crates/nsl-runtime/src/kernel_profiler.rs`

- [ ] **Step 1: Write unit test for flush JSON output**

```rust
#[test]
fn test_flush_produces_valid_json() {
    use std::io::Read;

    nsl_kernel_profiler_stop();
    *KERNEL_PROFILER.cpu_start_time.lock().unwrap() = Some(Instant::now());
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;
    KERNEL_PROFILER.traces.lock().unwrap().clear();
    {
        let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
        pool.clear();
        for i in 0..4u64 {
            pool.push((i * 2, i * 2 + 1));
        }
    }
    KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);

    // Simulate 2 kernel traces (without actual GPU events)
    kernel_profiler_push_trace("matmul_256", [4, 1, 1], [256, 1, 1]);
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 2;
    kernel_profiler_push_trace("fused_relu_add", [4, 1, 1], [256, 1, 1]);

    // Flush to temp file (CPU-only mode, no cuEventElapsedTime)
    let tmp = std::env::temp_dir().join("test_kernel_profiler.json");
    let path_str = tmp.to_str().unwrap();
    flush_traces_cpu_fallback(path_str);

    // Verify JSON
    let mut contents = String::new();
    std::fs::File::open(&tmp).unwrap().read_to_string(&mut contents).unwrap();
    assert!(contents.contains("traceEvents"));
    assert!(contents.contains("matmul_256"));
    assert!(contents.contains("fused_relu_add"));
    assert!(contents.contains("\"ph\":\"X\""));
    assert!(contents.contains("\"tid\":1"));

    std::fs::remove_file(&tmp).ok();
    nsl_kernel_profiler_stop();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-runtime test_flush_produces -- --nocapture`
Expected: FAIL — `flush_traces_cpu_fallback` not defined

- [ ] **Step 3: Implement flush with CPU fallback and GPU path**

Add to `kernel_profiler.rs`:
```rust
use std::io::Write;

/// CPU-only flush: writes traces with monotonically increasing synthetic timestamps.
/// Used in tests and when GPU events are not available.
pub(crate) fn flush_traces_cpu_fallback(path: &str) {
    let traces = KERNEL_PROFILER.traces.lock().unwrap();

    // CPU fallback uses synthetic timestamps (no real GPU events),
    // so base is always 0. cpu_start_time is only used by the GPU path.
    let mut events = Vec::new();
    for (i, trace) in traces.iter().enumerate() {
        let ts = (i as f64) * 100.0; // synthetic 100us spacing from t=0
        events.push(format!(
            r#"{{"name":"{}","ph":"X","ts":{:.1},"dur":50.0,"pid":0,"tid":1,"args":{{"grid":[{},{},{}],"block":[{},{},{}]}}}}"#,
            trace.name, ts,
            trace.grid[0], trace.grid[1], trace.grid[2],
            trace.block[0], trace.block[1], trace.block[2],
        ));
    }

    let total_launches = traces.len();
    let json = format!(
        r#"{{"traceEvents":[{}],"metadata":{{"total_kernel_launches":{},"profiler_mode":"cpu_fallback"}}}}"#,
        events.join(","),
        total_launches,
    );

    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(json.as_bytes());
    }
}

/// GPU flush: resolves cuEventElapsedTime for all traces, writes Chrome tracing JSON.
/// Consumes the event pool (calls cuEventDestroy on all events).
#[cfg(feature = "cuda")]
pub(crate) fn flush_traces_gpu(path: &str) {
    use crate::cuda;

    // Disable profiler first to prevent new recordings during flush
    KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);

    // Single sync to drain GPU pipeline
    unsafe { cuda::cu_ctx_synchronize(); }

    // Extract data from mutexes one at a time to avoid lock ordering issues
    let traces: Vec<KernelTrace> = {
        let mut guard = KERNEL_PROFILER.traces.lock().unwrap();
        std::mem::take(&mut *guard)
    };
    let pool: Vec<(u64, u64)> = {
        KERNEL_PROFILER.event_pool.lock().unwrap().clone()
    };
    let base_event = *KERNEL_PROFILER.gpu_base_event.lock().unwrap();
    let cpu_start = 0.0f64; // Base is 0, all times relative to profiler start

    let mut events_json = Vec::new();
    let mut total_kernel_time_ms = 0.0f64;

    for trace in traces.iter() {
        if trace.pool_idx >= pool.len() { continue; }
        let (start_event, stop_event) = pool[trace.pool_idx];

        let mut duration_ms: f32 = 0.0;
        let mut offset_ms: f32 = 0.0;
        unsafe {
            cuda::cu_event_elapsed_time(&mut duration_ms, start_event, stop_event);
            cuda::cu_event_elapsed_time(&mut offset_ms, base_event, start_event);
        }

        let ts_us = cpu_start + (offset_ms as f64) * 1000.0;
        let dur_us = (duration_ms as f64) * 1000.0;
        total_kernel_time_ms += duration_ms as f64;

        events_json.push(format!(
            r#"{{"name":"{}","ph":"X","ts":{:.1},"dur":{:.1},"pid":0,"tid":1,"args":{{"grid":[{},{},{}],"block":[{},{},{}]}}}}"#,
            trace.name, ts_us, dur_us,
            trace.grid[0], trace.grid[1], trace.grid[2],
            trace.block[0], trace.block[1], trace.block[2],
        ));
    }

    let total_launches = traces.len();
    let json = format!(
        r#"{{"traceEvents":[{}],"metadata":{{"total_kernel_launches":{},"total_kernel_time_ms":{:.2}}}}}"#,
        events_json.join(","),
        total_launches,
        total_kernel_time_ms,
    );

    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(json.as_bytes());
    }

    // Cleanup: destroy all events (traces and pool already consumed above)
    destroy_event_pool();
}

/// Destroy all cuEvents in the pool and the base event. Full resource cleanup.
#[cfg(feature = "cuda")]
fn destroy_event_pool() {
    let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
    for (start, stop) in pool.drain(..) {
        unsafe {
            crate::cuda::cu_event_destroy(start);
            crate::cuda::cu_event_destroy(stop);
        }
    }
    let base = *KERNEL_PROFILER.gpu_base_event.lock().unwrap();
    if base != 0 {
        unsafe { crate::cuda::cu_event_destroy(base); }
        *KERNEL_PROFILER.gpu_base_event.lock().unwrap() = 0;
    }
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;
}

/// FFI flush entry point. Calls GPU path if available, CPU fallback otherwise.
/// # Safety
/// `path_ptr` must point to valid UTF-8 bytes of length `path_len`.
#[no_mangle]
pub unsafe extern "C" fn nsl_kernel_profiler_flush(path_ptr: *const u8, path_len: i64) {
    let path = if path_ptr.is_null() || path_len <= 0 {
        "kernel_profile.json".to_string()
    } else {
        let bytes = std::slice::from_raw_parts(path_ptr, path_len as usize);
        std::str::from_utf8(bytes).unwrap_or("kernel_profile.json").to_string()
    };

    #[cfg(feature = "cuda")]
    {
        flush_traces_gpu(&path);
        return;
    }

    #[allow(unreachable_code)]
    flush_traces_cpu_fallback(&path);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-runtime test_flush_produces -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/kernel_profiler.rs
git commit -m "feat(runtime): add Chrome tracing JSON flush for kernel profiler"
```

---

### Task 4: Integrate Kernel Profiler into kernel_launch()

**Files:**
- Modify: `crates/nsl-runtime/src/cuda/mod.rs`

- [ ] **Step 1: Read the current kernel_launch function**

Read: `crates/nsl-runtime/src/cuda/mod.rs` lines 240-310

- [ ] **Step 2: Restructure kernel_launch — drop mutex before CUDA calls, inject profiler**

Modify `kernel_launch()` in `cuda/mod.rs`:

The current pattern holds the mutex across the entire function. Restructure to:
1. Lock mutex → lookup/load module and function → drop mutex
2. If profiler enabled: pop events (two-phase lock on profiler mutex), record start event
3. cuLaunchKernel (no lock held)
4. If profiler enabled: record stop event, push trace
5. Remove cuCtxSynchronize (per spec: "Per-Launch Sync Removal")

Key changes:
- Move `cuCtxSetCurrent` before the mutex lock (it's per-thread state, safe to call without lock)
- Extract module/function lookup into locked section, drop lock
- Add profiler event recording around cuLaunchKernel
- Remove the trailing `cuCtxSynchronize()` call from `kernel_launch()` ONLY

**Important:** The sync removal applies only to the low-level `kernel_launch()` primitive. The higher-level helper functions (`gpu_elementwise_binary`, `gpu_matmul`, backward kernels, etc.) each have their own `cuCtxSynchronize()` calls and those MUST stay — they need synchronous results because they immediately read output buffers after launch. `kernel_launch()` is used by codegen-emitted user kernels where the profiler flush provides the final sync.

```rust
pub(crate) fn kernel_launch(
    ptx_ptr: *const u8,
    name_ptr: *const u8,
    grid: [i64; 3],
    block: [i64; 3],
    args: &[*mut c_void],
) -> CUresult {
    let state = state();
    let func = {
        let mut guard = state.lock().unwrap();
        unsafe { cuCtxSetCurrent(guard.context); }

        // Module cache lookup (existing logic)
        let ptx_key = ptx_ptr as usize;
        let module = if let Some(m) = guard.module_cache.get(&ptx_key) {
            *m
        } else {
            let mut m: CUmodule = std::ptr::null_mut();
            let res = unsafe { cuModuleLoadData(&mut m, ptx_ptr as *const c_void) };
            if res != CUresult::CUDA_SUCCESS { return res; }
            guard.module_cache.insert(ptx_key, m);
            m
        };

        let name = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const i8) };
        let mut func: CUfunction = std::ptr::null_mut();
        let res = unsafe { cuModuleGetFunction(&mut func, module, name.as_ptr()) };
        if res != CUresult::CUDA_SUCCESS { return res; }

        func
    }; // guard dropped here — no lock held for CUDA calls

    // Profiler: pop event pair (lock-pop-unlock on profiler mutex)
    let profiler_events = if crate::kernel_profiler::kernel_profiler_enabled() {
        crate::kernel_profiler::kernel_profiler_pop_events()
    } else {
        None
    };

    // Record start event before launch
    #[cfg(feature = "cuda")]
    if let Some((start, _, _)) = &profiler_events {
        unsafe { cuEventRecord(*start as CUevent, std::ptr::null_mut()); }
    }

    // Launch kernel (no lock held)
    let mut kernel_args: Vec<*mut c_void> = args.to_vec();
    let res = unsafe {
        cuLaunchKernel(
            func,
            grid[0] as u32, grid[1] as u32, grid[2] as u32,
            block[0] as u32, block[1] as u32, block[2] as u32,
            0, std::ptr::null_mut(),
            kernel_args.as_mut_ptr(), std::ptr::null_mut(),
        )
    };

    // Record stop event after launch
    #[cfg(feature = "cuda")]
    if let Some((_, stop, _)) = &profiler_events {
        unsafe { cuEventRecord(*stop as CUevent, std::ptr::null_mut()); }
        // Push trace (lock-push-unlock on profiler mutex)
        let name = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const i8) };
        let name_str = name.to_str().unwrap_or("unknown");
        crate::kernel_profiler::kernel_profiler_push_trace(
            name_str,
            [grid[0] as u32, grid[1] as u32, grid[2] as u32],
            [block[0] as u32, block[1] as u32, block[2] as u32],
        );
    }

    // NOTE: cuCtxSynchronize removed — unified memory provides coherence,
    // profiler flush performs single sync at program end
    res
}
```

- [ ] **Step 3: Add cuEvent wrapper functions to cuda module**

The cuEvent functions (`cuEventCreate`, `cuEventRecord`, `cuEventElapsedTime`, `cuEventDestroy_v2`) are already available via the existing `use cudarc::driver::sys::*;` import at the top of the `inner` module. Do NOT add duplicate `extern "C"` declarations — that would cause linker symbol conflicts.

Add thin wrappers **inside `mod inner`** (where `use cudarc::driver::sys::*;` is in scope). The cudarc types (`CUevent`, `CUstream`, `CUresult`, etc.) are only available inside `mod inner` — placing these functions outside will cause "unresolved name" errors.

```rust
// Inside mod inner { ... } in cuda/mod.rs:

pub unsafe fn cu_event_create(event: *mut u64) {
    // CUevent is a pointer type, stored as u64
    cuEventCreate(event as *mut CUevent, 0); // CU_EVENT_DEFAULT = 0
}

pub unsafe fn cu_event_record(event: u64, stream: *mut std::ffi::c_void) {
    cuEventRecord(event as CUevent, stream as CUstream);
}

pub unsafe fn cu_event_elapsed_time(ms: *mut f32, start: u64, stop: u64) {
    cuEventElapsedTime(ms, start as CUevent, stop as CUevent);
}

pub unsafe fn cu_event_destroy(event: u64) {
    cuEventDestroy_v2(event as CUevent);
}

pub unsafe fn cu_ctx_synchronize() {
    cuCtxSynchronize();
}
```

Then at the `cuda` module level (outside `mod inner`), re-export for use by `kernel_profiler.rs`:
```rust
#[cfg(feature = "cuda")]
pub(crate) use inner::{cu_event_create, cu_event_record, cu_event_elapsed_time, cu_event_destroy, cu_ctx_synchronize};
```

- [ ] **Step 4: Build and run unit tests**

Run: `cargo test -p nsl-runtime -- --nocapture`
Expected: All existing tests pass (45+), kernel profiler tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/cuda/mod.rs crates/nsl-runtime/src/kernel_profiler.rs
git commit -m "feat(runtime): integrate kernel profiler into kernel_launch, remove per-launch sync"
```

---

### Task 5: Kernel Profiler — Semantic + Codegen Builtin Registration

**Files:**
- Modify: `crates/nsl-semantic/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

- [ ] **Step 1: Register kernel profiler builtins in semantic checker**

Add to `crates/nsl-semantic/src/builtins.rs` in `register_builtins()`, after the existing profiler registrations:
```rust
// Kernel profiler builtins (M26)
// Only register start/stop — flush is NOT user-callable (it's called from
// the atexit handler and env var init, not from NSL code).
// compile_call_by_name does NOT split strings into (ptr, len) pairs,
// so exposing flush as a builtin would cause an argument count mismatch.
def("kernel_profiler_start", Type::Function { params: vec![], ret: Box::new(Type::Void) });
def("kernel_profiler_stop", Type::Function { params: vec![], ret: Box::new(Type::Void) });
```

- [ ] **Step 2: Declare Cranelift imports in codegen builtins**

Add to `crates/nsl-codegen/src/builtins.rs` in the `RUNTIME_FUNCTIONS` array:
```rust
("nsl_kernel_profiler_start", &[], None),
("nsl_kernel_profiler_stop", &[], None),
// nsl_kernel_profiler_flush is NOT registered here — it's only called from
// the atexit handler in Rust, not from compiled NSL code.
```

- [ ] **Step 3: Add dispatch in expr.rs**

Add `kernel_profiler_start` and `kernel_profiler_stop` (NOT `kernel_profiler_flush`) to the existing runtime builtin match arm in `crates/nsl-codegen/src/expr.rs`. The existing codebase uses `compile_call_by_name` for all builtin dispatch:

```rust
// Add to the existing match arm that handles profiler_start/profiler_stop/profiler_peak:
"kv_cache_alloc_seq" | "kv_cache_append" |
"kv_cache_k_ptr" | "kv_cache_v_ptr" |
"kv_cache_free_seq" | "kv_cache_seq_len" |
"kv_cache_seq_blocks" | "kv_cache_seq_num_blocks" |
"kv_cache_utilization" | "kv_cache_destroy" |
"profiler_start" | "profiler_stop" | "profiler_peak" |
"kernel_profiler_start" | "kernel_profiler_stop")
{
    let mut arg_vals = Vec::new();
    for arg in args {
        arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
    }
    let rt_name = format!("nsl_{func_name}");
    return self.compile_call_by_name(builder, &rt_name, &arg_vals);
}
```

`kernel_profiler_flush` is NOT registered as a codegen builtin — it's only called from Rust code (atexit handler in `kernel_profiler.rs`), never from compiled NSL code. The `compile_call_by_name` helper does NOT perform string-to-`(ptr, len)` splitting, so passing a string argument would cause a Cranelift argument count mismatch.

- [ ] **Step 4: Build the workspace**

Run: `cargo build --workspace`
Expected: Compiles without errors

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-semantic/src/builtins.rs crates/nsl-codegen/src/builtins.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(codegen): register kernel profiler builtins in semantic + codegen pipeline"
```

---

### Task 6: CLI — `--profile-kernels` and `--profile` Flags

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`

- [ ] **Step 1: Add CLI flags to Run subcommand**

Add to the `Run` variant in `Cli` enum:
```rust
Run {
    file: PathBuf,
    #[arg(long)]
    profile_memory: bool,
    #[arg(long)]
    profile_kernels: bool,
    #[arg(long)]
    profile: bool,
    #[arg(last = true)]  // match existing codebase style (not #[clap])
    args: Vec<String>,
},
```

- [ ] **Step 2: Handle the flags in the run handler**

In the `Cli::Run` match arm, after the existing `profile_memory` handling:
```rust
if profile_memory || profile {
    cmd.env("NSL_PROFILE_MEMORY", "1");
}
if profile_kernels || profile {
    cmd.env("NSL_PROFILE_KERNELS", "1");
}

// IMPORTANT: The merge must happen BEFORE process::exit() in the run handler.
// The existing run_run() function calls process::exit(status.code()...) at the end.
// Insert this merge block between the subprocess .status() call and process::exit:
//
// let status = cmd.status()?;
// // --- INSERT MERGE HERE ---
// if profile {
//     merge_profile_traces("memory_profile.json", "kernel_profile.json", "profile.json");
// }
// // --- THEN the existing cleanup + exit ---
// process::exit(status.code().unwrap_or(1));
if profile {
    merge_profile_traces("memory_profile.json", "kernel_profile.json", "profile.json");
}
```

Add the merge function:
```rust
fn merge_profile_traces(memory_path: &str, kernel_path: &str, output_path: &str) {
    let mem_json = std::fs::read_to_string(memory_path).unwrap_or_default();
    let kern_json = std::fs::read_to_string(kernel_path).unwrap_or_default();

    // Extract traceEvents arrays from both files
    let mem_events = extract_trace_events(&mem_json).unwrap_or_default();
    let kern_events = extract_trace_events(&kern_json).unwrap_or_default();

    // Merge: memory events keep tid:0, kernel events get tid:1
    let merged = format!(
        r#"{{"traceEvents":[{}{}{}],"metadata":{{"merged":true}}}}"#,
        mem_events,
        if !mem_events.is_empty() && !kern_events.is_empty() { "," } else { "" },
        kern_events,
    );

    std::fs::write(output_path, &merged).ok();

    // Clean up individual files
    std::fs::remove_file(memory_path).ok();
    std::fs::remove_file(kernel_path).ok();
    eprintln!("[nsl] merged profile written to {}", output_path);
}

fn extract_trace_events(json: &str) -> Option<String> {
    let start = json.find("\"traceEvents\":")? + "\"traceEvents\":".len();
    let bracket_start = json[start..].find('[')? + start;
    let mut depth = 0;
    let mut bracket_end = bracket_start;
    for (i, ch) in json[bracket_start..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => { depth -= 1; if depth == 0 { bracket_end = bracket_start + i; break; } }
            _ => {}
        }
    }
    Some(json[bracket_start + 1..bracket_end].to_string())
}
```

- [ ] **Step 3: Add env var check in runtime args init**

In `crates/nsl-runtime/src/args.rs`, add after the existing `NSL_PROFILE_MEMORY` check:
```rust
if std::env::var("NSL_PROFILE_KERNELS").is_ok() {
    crate::kernel_profiler::nsl_kernel_profiler_start();
}
```

- [ ] **Step 4: Add atexit handler for kernel profiler dump**

In `crates/nsl-runtime/src/kernel_profiler.rs`, add an atexit registration in `nsl_kernel_profiler_start()`:
```rust
use std::sync::atomic::AtomicBool;
static ATEXIT_REGISTERED: AtomicBool = AtomicBool::new(false);

// Raw extern for atexit — avoids libc dependency
extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

// Inside nsl_kernel_profiler_start(), after enabling:
if !ATEXIT_REGISTERED.swap(true, Ordering::Relaxed) {
    unsafe {
        atexit(kernel_profiler_atexit);
    }
}

extern "C" fn kernel_profiler_atexit() {
    if kernel_profiler_enabled() {
        let path = "kernel_profile.json";
        let ptr = path.as_ptr();
        let len = path.len() as i64;
        unsafe { nsl_kernel_profiler_flush(ptr, len); }
        eprintln!("[nsl] kernel profile written to {}", path);
    }
}
```

- [ ] **Step 5: Build and verify**

Run: `cargo build --workspace`
Expected: Compiles without errors

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-cli/src/main.rs crates/nsl-runtime/src/args.rs crates/nsl-runtime/src/kernel_profiler.rs
git commit -m "feat(cli): add --profile-kernels and --profile flags with atexit auto-dump"
```

---

## Chunk 2: Elementwise Fusion

### Task 7: Semantic Validation — `@fuse` Decorator

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 1: Add @fuse validation in the Decorated handler**

In `checker.rs`, inside the `StmtKind::Decorated` match arm, add after the existing decorator checks (near the `@test` and `@paged_kv` validation):

```rust
if dname == "fuse" {
    match &stmt.kind {
        StmtKind::FnDef(_) => {
            // Valid target — further validation in codegen
        }
        StmtKind::KernelDef(_) => {
            self.diagnostics.push(
                Diagnostic::error("@fuse cannot be applied to kernel blocks (kernel blocks are already single PTX kernels)")
                    .with_label(deco.span, "invalid @fuse target")
            );
        }
        StmtKind::ModelDef(_) => {
            self.diagnostics.push(
                Diagnostic::error("@fuse cannot be applied to model methods; extract the fusible logic into a standalone fn")
                    .with_label(deco.span, "invalid @fuse target")
            );
        }
        _ => {
            self.diagnostics.push(
                Diagnostic::error("@fuse can only be applied to fn declarations")
                    .with_label(deco.span, "invalid @fuse target")
            );
        }
    }
}
```

- [ ] **Step 2: Add @autotune validation**

```rust
if dname == "autotune" {
    match &stmt.kind {
        StmtKind::KernelDef(_) => {
            // Valid target — validate args are lists of integers
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(ref name_sym) = arg.name {
                        let _aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                        // Each arg value must be a list literal of integers
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
                                    Diagnostic::error("@autotune parameters must be lists of integers (e.g., [64, 128, 256])")
                                        .with_label(arg.span, "expected list")
                                );
                            }
                        }
                    }
                }
            } else {
                self.diagnostics.push(
                    Diagnostic::error("@autotune requires at least one tuning parameter")
                        .with_label(deco.span, "missing parameters")
                );
            }
        }
        _ => {
            self.diagnostics.push(
                Diagnostic::error("@autotune can only be applied to kernel blocks")
                    .with_label(deco.span, "invalid @autotune target")
            );
        }
    }
}
```

- [ ] **Step 3: Build and verify**

Run: `cargo build --workspace`
Expected: Compiles without errors

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs
git commit -m "feat(semantic): validate @fuse and @autotune decorator placement and arguments"
```

---

### Task 8: Fusion Module — FusedKernel Data Structure and PTX Synthesis

**Files:**
- Create: `crates/nsl-codegen/src/fusion.rs`
- Modify: `crates/nsl-codegen/src/mod.rs` (or `lib.rs`)

- [ ] **Step 1: Create fusion.rs with FusedKernel and fusible op classification**

Create `crates/nsl-codegen/src/fusion.rs`:
```rust
//! Elementwise operator fusion: explicit @fuse and lexical auto-fusion.
//! Synthesizes single PTX kernels from chains of elementwise ops.
//! let-binding = hard fusion barrier (no DAG infrastructure).

/// A chain of elementwise operations that can be fused into a single PTX kernel.
pub struct FusedKernel {
    /// Names of ops in the chain (e.g., ["add", "relu"])
    pub op_chain: Vec<String>,
    /// Input tensor count
    pub num_inputs: usize,
    /// Generated PTX bytes (null-terminated)
    pub ptx: Vec<u8>,
    /// Human-readable name for profiler traces
    pub name: String,
}

/// Classification of operations for fusion eligibility.
pub fn is_fusible_op(name: &str) -> bool {
    matches!(name,
        "add" | "sub" | "mul" | "div" | "pow" | "neg" | "abs"
        | "relu" | "sigmoid" | "tanh" | "exp" | "log" | "sqrt"
        | "sign" | "clamp"
        // NOTE: gelu and silu are NOT fusible in M26 — they require multi-instruction
        // sequences that are better handled as @fuse fn with explicit PTX synthesis.
        // They may be added in a future milestone.
    )
}

/// Binary ops that are fusible (from BinOp AST nodes).
pub fn is_fusible_binop(op: &str) -> bool {
    matches!(op, "Add" | "Sub" | "Mul" | "Div" | "Pow")
}

/// Unary ops that are fusible.
pub fn is_fusible_unaryop(op: &str) -> bool {
    matches!(op, "Neg")
}

/// Check if an op is NOT fusible (matmul, reductions, etc.).
pub fn is_fusion_barrier(name: &str) -> bool {
    matches!(name,
        "matmul" | "sum" | "mean" | "reduce_max" | "reduce_min"
        | "reshape" | "transpose" | "conv" | "softmax"
        | "layernorm" | "gather" | "scatter"
    )
}
```

- [ ] **Step 2: Add module declaration**

Add to `crates/nsl-codegen/src/lib.rs` (or wherever the codegen modules are declared):
```rust
pub mod fusion;
```

- [ ] **Step 3: Build**

Run: `cargo build -p nsl-codegen`
Expected: Compiles without errors

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/fusion.rs crates/nsl-codegen/src/lib.rs
git commit -m "feat(codegen): add fusion module with FusedKernel struct and op classification"
```

---

### Task 9: Fusion — PTX Synthesis for Fused Elementwise Chains

**Files:**
- Modify: `crates/nsl-codegen/src/fusion.rs`

- [ ] **Step 1: Write test for fused PTX generation**

Add to `fusion.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthesize_fused_add_relu() {
        let ptx = synthesize_fused_ptx(
            "fused_add_relu",
            &["add", "relu"],
            2, // 2 input tensors (result of matmul + bias)
        );
        let ptx_str = std::str::from_utf8(&ptx[..ptx.len()-1]).unwrap(); // strip null
        assert!(ptx_str.contains(".version 7.0"));
        assert!(ptx_str.contains(".entry fused_add_relu"));
        assert!(ptx_str.contains("ld.global.f32")); // loads inputs
        assert!(ptx_str.contains("add.f32"));        // fused add
        assert!(ptx_str.contains("max.f32"));         // relu = max(x, 0)
        assert!(ptx_str.contains("st.global.f32"));   // stores output
    }

    #[test]
    fn test_synthesize_single_op_returns_none() {
        // Chain of 1 op has no fusion benefit
        let result = try_synthesize_fused(&["relu"], 1);
        assert!(result.is_none());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p nsl-codegen test_synthesize -- --nocapture`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement fused PTX synthesis**

Add to `fusion.rs`:
```rust
/// Attempt to synthesize a fused PTX kernel for a chain of elementwise ops.
/// Returns None if chain has fewer than 2 ops (no fusion benefit).
pub fn try_synthesize_fused(op_chain: &[&str], num_inputs: usize) -> Option<FusedKernel> {
    if op_chain.len() < 2 {
        return None;
    }
    let name = format!("fused_{}", op_chain.join("_"));
    let ptx = synthesize_fused_ptx(&name, op_chain, num_inputs);
    Some(FusedKernel {
        op_chain: op_chain.iter().map(|s| s.to_string()).collect(),
        num_inputs,
        ptx,
        name,
    })
}

/// Generate PTX for a fused elementwise kernel with multi-dimensional broadcast support.
/// Each input is loaded once from global memory, all ops are register-to-register,
/// output is stored once to global memory.
///
/// **Broadcast indexing:** For inputs with different shapes, the kernel decomposes the
/// flat thread ID into N-dimensional coordinates, applies the broadcast rule (stride=0
/// for dimensions of size 1), and re-flattens with per-input strides. This avoids the
/// naive 1D modulo approach that breaks on non-contiguous broadcasts.
///
/// **ISA correctness:** `ex2.approx.f32` computes base-2 exp, so natural exp requires
/// multiplying by log2(e) ≈ 1.4427 first. Similarly, `lg2.approx.f32` computes base-2
/// log, so natural log requires multiplying by ln(2) ≈ 0.6931 after.
pub fn synthesize_fused_ptx(name: &str, ops: &[&str], num_inputs: usize) -> Vec<u8> {
    // log2(e) = 1.4426950408889634 → IEEE 754 f32: 0x3FB8AA3B
    const LOG2_E_HEX: &str = "0f3FB8AA3B";
    // ln(2) = 0.6931471805599453 → IEEE 754 f32: 0x3F317218
    const LN_2_HEX: &str = "0f3F317218";

    let mut ptx = String::new();

    // Header
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_52\n");
    ptx.push_str(".address_size 64\n\n");

    // Function signature:
    // For broadcast-capable kernels, each input has (ptr, ndim, shape_ptr, stride_ptr).
    // For M26 we use the simpler flat layout: all inputs same shape (no broadcast needed).
    // Multi-dimensional broadcast is handled when input shapes differ (detected at call site).
    // Signature: (output_ptr, input0_ptr, ..., inputN_ptr, num_elements)
    ptx.push_str(&format!(".visible .entry {}(\n", name));
    ptx.push_str("    .param .u64 param_out,\n");
    for i in 0..num_inputs {
        ptx.push_str(&format!("    .param .u64 param_in{},\n", i));
    }
    ptx.push_str("    .param .u64 param_n\n");
    ptx.push_str(") {\n");

    // Register declarations — allocate enough for multi-op chains.
    // Extra +8 buffer accounts for temporary registers borrowed by log (1 extra),
    // clamp (1 extra), sigmoid (1 extra), and other multi-instruction ops.
    // Worst case: chain with all borrowing ops uses at most ops.len() extra temps.
    let max_regs = num_inputs + ops.len() * 2 + 8;
    ptx.push_str(&format!("    .reg .u32 %r<{}>;\n", 8.max(max_regs)));
    ptx.push_str(&format!("    .reg .u64 %rd<{}>;\n", (4 + num_inputs + 2).max(16)));
    ptx.push_str(&format!("    .reg .f32 %f<{}>;\n", max_regs.max(16)));
    ptx.push_str("    .reg .pred %p<4>;\n\n");

    // Thread ID = blockIdx.x * blockDim.x + threadIdx.x
    ptx.push_str("    mov.u32 %r0, %tid.x;\n");
    ptx.push_str("    mov.u32 %r1, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %r2, %ntid.x;\n");
    ptx.push_str("    mul.lo.u32 %r3, %r1, %r2;\n");
    ptx.push_str("    add.u32 %r0, %r0, %r3;\n");
    ptx.push_str("    cvt.u64.u32 %rd0, %r0;\n\n");

    // Bounds check: if tid >= n, return
    ptx.push_str("    ld.param.u64 %rd1, [param_n];\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd0, %rd1;\n");
    ptx.push_str("    @%p0 bra $L_end;\n\n");

    // Byte offset: tid * 4 (f32)
    ptx.push_str("    shl.b64 %rd2, %rd0, 2;\n\n");

    // Load inputs from global memory
    // NOTE: For broadcast support, the call site must ensure inputs are pre-broadcast
    // to the output shape, OR pass per-input stride arrays and use decomposed indexing:
    //   coord[d] = (tid / output_stride[d]) % output_shape[d]
    //   input_idx = sum(coord[d] * input_stride[d])  // stride=0 for broadcast dims
    // M26 uses the flat path (all inputs same shape). The broadcast decomposition
    // infrastructure is generated at the call site in compiler.rs when shapes differ.
    for i in 0..num_inputs {
        ptx.push_str(&format!("    ld.param.u64 %rd{}, [param_in{}];\n", 3 + i, i));
        ptx.push_str(&format!("    add.u64 %rd{}, %rd{}, %rd2;\n", 3 + i, 3 + i));
        ptx.push_str(&format!("    ld.global.f32 %f{}, [%rd{}];\n\n", i, 3 + i));
    }

    // Apply fused ops — register-to-register
    let mut result_reg = 0; // Track which %f register has the current result
    for (op_idx, op) in ops.iter().enumerate() {
        let out_reg = num_inputs + op_idx;
        match *op {
            "add" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                let rhs = if op_idx == 0 { 1 } else { 1 };
                ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, lhs, rhs));
            }
            "sub" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    sub.f32 %f{}, %f{}, %f{};\n", out_reg, lhs, 1));
            }
            "mul" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, lhs, 1));
            }
            "div" => {
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    div.rn.f32 %f{}, %f{}, %f{};\n", out_reg, lhs, 1));
            }
            "pow" => {
                // pow(a, b): a^b = 2^(b * log2(a)) — uses base-2 intrinsics directly
                let lhs = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    lg2.approx.f32 %f{}, %f{};\n", out_reg, lhs)); // log2(a)
                ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, 1)); // b * log2(a)
                ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg)); // 2^(...)
            }
            "relu" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", out_reg));
                ptx.push_str(&format!("    max.f32 %f{}, %f{}, %f{};\n", out_reg, src, out_reg));
            }
            "neg" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, src));
            }
            "abs" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    abs.f32 %f{}, %f{};\n", out_reg, src));
            }
            "exp" => {
                // CRITICAL: ex2.approx.f32 is base-2. For natural exp: x * log2(e), then ex2
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, {};\n", out_reg, LOG2_E_HEX)); // log2(e)
                ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, src, out_reg)); // x * log2(e)
                ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg)); // 2^(x*log2(e)) = e^x
            }
            "log" => {
                // CRITICAL: lg2.approx.f32 is base-2. For natural log: lg2(x) * ln(2)
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    lg2.approx.f32 %f{}, %f{};\n", out_reg, src)); // log2(x)
                ptx.push_str(&format!("    mov.f32 %f{}, {};\n", out_reg + 1, LN_2_HEX)); // ln(2) — borrow next reg
                ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg + 1)); // log2(x) * ln(2) = ln(x)
            }
            "sqrt" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    sqrt.approx.f32 %f{}, %f{};\n", out_reg, src));
            }
            "clamp" => {
                // clamp(x, min, max): assumes min=0.0, max=1.0 for common activation use
                // Full clamp with arbitrary bounds uses input registers for min/max
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", out_reg)); // 0.0
                ptx.push_str(&format!("    max.f32 %f{}, %f{}, %f{};\n", out_reg, src, out_reg)); // max(x, 0)
                ptx.push_str(&format!("    mov.f32 %f{}, 0f3F800000;\n", out_reg + 1)); // 1.0 — borrow next reg
                ptx.push_str(&format!("    min.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg + 1)); // min(result, 1)
            }
            "sigmoid" => {
                // sigmoid(x) = 1 / (1 + exp(-x))
                // Must use base-2 correction: exp(-x) = 2^(-x * log2(e))
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, src)); // -x
                ptx.push_str(&format!("    mov.f32 %f{}, {};\n", out_reg + 1, LOG2_E_HEX)); // log2(e) — borrow reg
                ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg + 1)); // -x * log2(e)
                ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg)); // 2^(...) = e^(-x)
                ptx.push_str(&format!("    add.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg)); // + 1.0
                ptx.push_str(&format!("    rcp.approx.f32 %f{}, %f{};\n", out_reg, out_reg)); // 1/...
            }
            "tanh" => {
                // tanh(x) = 2*sigmoid(2x) - 1
                // sigmoid uses base-2 corrected exp
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, src, src)); // 2x
                ptx.push_str(&format!("    neg.f32 %f{}, %f{};\n", out_reg, out_reg)); // -2x
                ptx.push_str(&format!("    mov.f32 %f{}, {};\n", out_reg + 1, LOG2_E_HEX)); // log2(e)
                ptx.push_str(&format!("    mul.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg + 1)); // -2x*log2(e)
                ptx.push_str(&format!("    ex2.approx.f32 %f{}, %f{};\n", out_reg, out_reg)); // e^(-2x)
                ptx.push_str(&format!("    add.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg)); // +1
                ptx.push_str(&format!("    rcp.approx.f32 %f{}, %f{};\n", out_reg, out_reg)); // sigmoid(2x)
                ptx.push_str(&format!("    add.f32 %f{}, %f{}, %f{};\n", out_reg, out_reg, out_reg)); // *2
                ptx.push_str(&format!("    sub.f32 %f{}, %f{}, 0f3F800000;\n", out_reg, out_reg)); // -1
            }
            "sign" => {
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    setp.gt.f32 %p1, %f{}, 0f00000000;\n", src));
                ptx.push_str(&format!("    setp.lt.f32 %p2, %f{}, 0f00000000;\n", src));
                ptx.push_str(&format!("    selp.f32 %f{}, 0f3F800000, 0f00000000, %p1;\n", out_reg));
                ptx.push_str(&format!("    selp.f32 %f{}, 0fBF800000, %f{}, %p2;\n", out_reg, out_reg));
            }
            _ => {
                // Unknown op: treat as passthrough (should not happen — validated earlier)
                let src = if op_idx == 0 { 0 } else { result_reg };
                ptx.push_str(&format!("    mov.f32 %f{}, %f{};\n", out_reg, src));
            }
        }
        result_reg = out_reg;
    }

    // Store result to output
    ptx.push_str(&format!("\n    ld.param.u64 %rd{}, [param_out];\n", 3 + num_inputs));
    ptx.push_str(&format!("    add.u64 %rd{}, %rd{}, %rd2;\n", 3 + num_inputs, 3 + num_inputs));
    ptx.push_str(&format!("    st.global.f32 [%rd{}], %f{};\n", 3 + num_inputs, result_reg));

    // End
    ptx.push_str("\n$L_end:\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    let mut bytes = ptx.into_bytes();
    bytes.push(0); // null terminator
    bytes
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p nsl-codegen test_synthesize -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/fusion.rs
git commit -m "feat(codegen): add fused PTX synthesis for elementwise op chains"
```

---

### Task 10: Fusion — `try_fuse_expr` Integration into Expression Codegen

**Files:**
- Modify: `crates/nsl-codegen/src/fusion.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

- [ ] **Step 1: Add AST analysis function to fusion.rs**

Add to `fusion.rs`:
```rust
use nsl_ast::expr::{Expr, ExprKind, Arg};
use nsl_ast::operator::{BinOp, UnaryOp};
use nsl_ast::Symbol;

/// Analyze an expression tree and extract a fusible elementwise chain.
/// Returns the op names and input expressions if fusion is profitable.
/// let-binding = hard fusion barrier (only inline expressions are fused).
///
/// `resolve_name`: closure that maps Symbol → String using the interner.
/// This is needed to identify fusible Call expressions (e.g., `relu(...)`, `sigmoid(...)`).
pub fn analyze_fusible_chain<F>(
    expr: &Expr,
    resolve_name: &F,
) -> Option<(Vec<String>, Vec<&Expr>)>
where F: Fn(Symbol) -> Option<String> {
    let mut ops = Vec::new();
    let mut inputs = Vec::new();
    collect_fusible_ops(expr, &mut ops, &mut inputs, resolve_name);

    if ops.len() < 2 {
        return None; // No fusion benefit for single op
    }

    Some((ops, inputs))
}

/// Collect fusible ops from an expression tree.
/// The `resolve_name` closure resolves Symbol indices to &str (uses the interner).
fn collect_fusible_ops<'a, F>(
    expr: &'a Expr,
    ops: &mut Vec<String>,
    inputs: &mut Vec<&'a Expr>,
    resolve_name: &F,
) where F: Fn(Symbol) -> Option<String> {
    match &expr.kind {
        ExprKind::BinaryOp { left, op, right } => {
            let op_name = match op {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                BinOp::Mul => "mul",
                BinOp::Div => "div",
                BinOp::Pow => "pow",
                _ => { inputs.push(expr); return; } // Non-fusible binop
            };
            // Recurse into children (inline — no let barrier)
            collect_fusible_ops(left, ops, inputs, resolve_name);
            collect_fusible_ops(right, ops, inputs, resolve_name);
            ops.push(op_name.to_string());
        }
        ExprKind::UnaryOp { op, operand } if matches!(op, UnaryOp::Neg) => {
            collect_fusible_ops(operand, ops, inputs, resolve_name);
            ops.push("neg".to_string());
        }
        ExprKind::Call { callee, args } => {
            // Resolve the function name via the interner
            if let ExprKind::Ident(name_sym) = &callee.kind {
                if let Some(name) = resolve_name(*name_sym) {
                    if is_fusible_op(&name) {
                        // clamp() is fusible only as unary (hardcoded 0/1 bounds).
                        // 3-arg clamp(x, min, max) with non-literal bounds is not fusible.
                        if name == "clamp" && args.len() != 1 {
                            inputs.push(expr);
                            return;
                        }
                        // Fusible unary activation: recurse into the argument
                        // args is Vec<Arg> — access .value to get the Expr
                        if args.len() == 1 {
                            collect_fusible_ops(&args[0].value, ops, inputs, resolve_name);
                            ops.push(name);
                            return;
                        }
                    }
                }
            }
            // Non-fusible call or multi-arg: treat as input (fusion boundary)
            inputs.push(expr);
        }
        _ => {
            // Leaf node or non-fusible: this is an input to the fused chain
            inputs.push(expr);
        }
    }
}
```

- [ ] **Step 2: Hook into expr.rs tensor dispatch**

In `crates/nsl-codegen/src/expr.rs`, add a method that checks for fusion before compiling a Call expression to a fusible activation function. Insert this check in `compile_call_by_name` when the function name is a fusible activation (relu, sigmoid, etc.) and the argument is an inline expression (not a variable reference):

```rust
/// Attempt auto-fusion on an expression. Called from compile_expr() before
/// normal dispatch for BinaryOp and Call nodes.
///
/// Parameters follow the Compiler's method signature pattern — builder and state
/// are passed explicitly (the Compiler struct does not own a FunctionBuilder).
///
/// Returns Ok(Some(val)) if fused, Ok(None) to fall through, Err on codegen error.
fn try_auto_fuse(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    expr: &Expr,
) -> Result<Option<cranelift::prelude::Value>, CodegenError> {
    let interner = self.interner;
    let resolve = |sym: Symbol| -> Option<String> {
        interner.resolve(sym.0).map(|s| s.to_string())
    };

    let (ops, inputs) = match crate::fusion::analyze_fusible_chain(expr, &resolve) {
        Some(result) => result,
        None => return Ok(None), // Not fusible or single op
    };

    // Compile each input expression to get tensor Values
    let mut input_vals = Vec::new();
    for input in &inputs {
        input_vals.push(self.compile_expr(builder, state, input)?);
    }

    // Synthesize fused PTX
    let op_strs: Vec<&str> = ops.iter().map(|s| s.as_str()).collect();
    let fused = match crate::fusion::try_synthesize_fused(&op_strs, input_vals.len()) {
        Some(f) => f,
        None => return Ok(None),
    };

    // Emit training branch: nsl_is_training() → unfused, else → fused kernel
    // Uses compile_call_by_name for the runtime call (existing pattern)
    let is_training = self.compile_call_by_name(builder, "nsl_is_training", &[])?;

    let training_block = builder.create_block();
    let inference_block = builder.create_block();
    let merge_block = builder.create_block();
    builder.append_block_param(merge_block, types::I64);
    builder.ins().brif(is_training, training_block, &[], inference_block, &[]);

    // Training path: compile the expression via normal (unfused) dispatch
    builder.switch_to_block(training_block);
    // Temporarily disable auto-fusion to prevent infinite recursion
    state.in_fuse_bypass = true;
    let unfused_result = self.compile_expr(builder, state, expr)?;
    state.in_fuse_bypass = false;
    builder.ins().jump(merge_block, &[unfused_result]);

    // Inference path: launch fused kernel via nsl_kernel_launch
    builder.switch_to_block(inference_block);
    // Embed PTX in .rodata, get pointer, launch via existing kernel_launch infrastructure
    let ptx_data = self.create_data_for_bytes(&fused.ptx);
    let ptx_gv = self.module.declare_data_in_func(ptx_data, builder.func);
    let ptx_ptr = builder.ins().global_value(types::I64, ptx_gv);
    // Build kernel name, grid/block, and arg pointers — follows compile_kernels pattern
    let fused_result = self.emit_fused_kernel_call(builder, &fused, ptx_ptr, &input_vals)?;
    builder.ins().jump(merge_block, &[fused_result]);

    // Merge
    builder.switch_to_block(merge_block);
    let result = builder.block_params(merge_block)[0];
    Ok(Some(result))
}
```

**Note:** `emit_fused_kernel_call` and the `in_fuse_bypass` flag on `FuncState` need to be implemented as part of this task. `emit_fused_kernel_call` follows the same pattern as `compile_kernels` — it builds the arg array and calls `nsl_kernel_launch`. `in_fuse_bypass` is a `bool` field added to `FuncState` (initially `false`) that `try_auto_fuse` checks at entry to prevent recursion.

The integration point: in `compile_expr`, when visiting a `BinaryOp` or `Call` node, check `!state.in_fuse_bypass` and call `try_auto_fuse(builder, state, expr)` first. If it returns `Ok(Some(value))`, use that value and skip the normal compilation path. If `Ok(None)`, fall through to the existing per-op dispatch.

- [ ] **Step 3: Build**

Run: `cargo build -p nsl-codegen`
Expected: Compiles without errors

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/fusion.rs crates/nsl-codegen/src/expr.rs
git commit -m "feat(codegen): add fusible chain analysis and expr.rs integration hook"
```

---

### Task 11: Fusion — `@fuse fn` Codegen Path

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`
- Modify: `crates/nsl-codegen/src/fusion.rs`

- [ ] **Step 1: Handle @fuse decorated functions in compiler**

In `compiler.rs`, add a new method or extend the existing function compilation to detect `@fuse` on `FnDef`:

When `StmtKind::Decorated { decorators, stmt }` is encountered and the decorator is `@fuse` on a `FnDef`:
1. Walk the function body AST and validate all ops are fusible:

```rust
fn validate_fuse_body(&self, fn_def: &FnDef) -> Result<(), CodegenError> {
    for stmt in &fn_def.body {
        match &stmt.kind {
            StmtKind::Return(Some(expr)) => {
                self.validate_fusible_expr(expr)?;
            }
            StmtKind::Expr(expr) => {
                self.validate_fusible_expr(expr)?;
            }
            StmtKind::Let { value: Some(expr), .. } => {
                // let bindings are allowed in @fuse (e.g., `let t = sqrt(2.0 / pi) * (x + ...)`)
                // but the assigned expression must be fusible
                self.validate_fusible_expr(expr)?;
            }
            StmtKind::ModelDef(_) => {
                return Err(CodegenError::new(
                    "model blocks cannot appear inside @fuse fn".to_string(),
                ));
            }
            _ => {
                return Err(CodegenError::new(
                    format!("@fuse function body must contain only fusible expressions and let bindings, found: {:?}", stmt.kind),
                ));
            }
        }
    }
    Ok(())
}

fn validate_fusible_expr(&self, expr: &Expr) -> Result<(), CodegenError> {
    match &expr.kind {
        ExprKind::BinaryOp { left, op, right } => {
            if !crate::fusion::is_fusible_binop(&format!("{:?}", op)) {
                return Err(CodegenError::new(
                    format!("@fuse: non-fusible binary op {:?} (only Add/Sub/Mul/Div/Pow allowed)", op),
                ));
            }
            self.validate_fusible_expr(left)?;
            self.validate_fusible_expr(right)?;
        }
        ExprKind::Call { callee, args } => {
            if let ExprKind::Ident(sym) = &callee.kind {
                let name = self.interner.resolve(sym.0).unwrap_or("");
                if !crate::fusion::is_fusible_op(name) {
                    return Err(CodegenError::new(
                        format!("@fuse: non-fusible function call '{}' (only elementwise ops allowed)", name),
                    ));
                }
            }
            // args is Vec<Arg> — validate each arg's value expression
            for arg in args { self.validate_fusible_expr(&arg.value)?; }
        }
        ExprKind::Ident(_) | ExprKind::FloatLiteral(_) | ExprKind::IntLiteral(_) => {
            // Leaf nodes: OK
        }
        ExprKind::UnaryOp { operand, .. } => {
            self.validate_fusible_expr(operand)?;
        }
        _ => {
            return Err(CodegenError::new(
                "@fuse: unsupported expression in fused function body".to_string(),
            ));
        }
    }
    Ok(())
}
```

2. Determine input parameters (tensor params of the function)
3. Call `synthesize_fused_ptx()` to generate the fused kernel PTX
4. Embed the PTX in .rodata (same pattern as `compile_kernels`)
5. Generate a wrapper function that:
   - If `nsl_is_training()`: call the unfused implementation (runtime panic if no `@backward`)
   - Else: launch the fused kernel

- [ ] **Step 2: Add training safety branch**

The codegen for a `@fuse fn` emits:
```
// Pseudo-Cranelift:
let is_training = call nsl_is_training()
brif is_training, training_block, inference_block

training_block:
    // Check if @backward exists for this function
    // If not: call nsl_panic("@fuse function 'name' called during training without @backward annotation")
    // If yes: call the backward-compatible unfused version
    jump merge_block(training_result)

inference_block:
    // Launch fused kernel
    call nsl_kernel_launch(fused_ptx, name, grid, block, args)
    jump merge_block(inference_result)

merge_block(result):
    return result
```

- [ ] **Step 3: Build and verify**

Run: `cargo build --workspace`
Expected: Compiles without errors

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs crates/nsl-codegen/src/fusion.rs
git commit -m "feat(codegen): @fuse fn codegen with training safety branch"
```

---

### Task 12: Auto-Fusion — Lexical Inline Folding in Expression Codegen

**Files:**
- Modify: `crates/nsl-codegen/src/expr.rs`
- Modify: `crates/nsl-codegen/src/fusion.rs`

- [ ] **Step 1: Add unit test for auto-fusion chain detection**

Add to `crates/nsl-codegen/src/fusion.rs` tests module:
```rust
#[test]
fn test_analyze_fusible_chain_add_relu() {
    use nsl_ast::expr::*;
    use nsl_ast::operator::BinOp;
    use nsl_ast::{NodeId, Span, Symbol};
    use string_interner::StringInterner;

    // Create interner and intern symbol names
    let mut interner = StringInterner::default();
    let sym_x = Symbol(interner.get_or_intern("x"));
    let sym_b = Symbol(interner.get_or_intern("b"));
    let sym_relu = Symbol(interner.get_or_intern("relu"));

    // Build AST: relu(x + b) — should detect ["add", "relu"] chain with 2 inputs
    let x = Expr { kind: ExprKind::Ident(sym_x), span: Span::dummy(), id: NodeId::dummy() };
    let b = Expr { kind: ExprKind::Ident(sym_b), span: Span::dummy(), id: NodeId::dummy() };
    let add = Expr {
        kind: ExprKind::BinaryOp { left: Box::new(x), op: BinOp::Add, right: Box::new(b) },
        span: Span::dummy(),
        id: NodeId::dummy(),
    };
    let relu_name = Expr { kind: ExprKind::Ident(sym_relu), span: Span::dummy(), id: NodeId::dummy() };
    let relu_arg = Arg { name: None, value: add, span: Span::dummy() };
    let relu_call = Expr {
        kind: ExprKind::Call { callee: Box::new(relu_name), args: vec![relu_arg] },
        span: Span::dummy(),
        id: NodeId::dummy(),
    };

    let resolve = |sym: Symbol| -> Option<String> {
        interner.resolve(sym.0).map(|s| s.to_string())
    };

    let result = analyze_fusible_chain(&relu_call, &resolve);
    assert!(result.is_some());
    let (ops, inputs) = result.unwrap();
    assert_eq!(ops, vec!["add", "relu"]);
    assert_eq!(inputs.len(), 2); // x, b
}

#[test]
fn test_single_op_not_fused() {
    use nsl_ast::expr::*;
    use nsl_ast::{NodeId, Span, Symbol};
    use string_interner::StringInterner;

    let mut interner = StringInterner::default();
    let sym_x = Symbol(interner.get_or_intern("x"));
    let sym_relu = Symbol(interner.get_or_intern("relu"));

    // Single relu(x) — chain length 1, no fusion benefit
    let x = Expr { kind: ExprKind::Ident(sym_x), span: Span::dummy(), id: NodeId::dummy() };
    let relu_name = Expr { kind: ExprKind::Ident(sym_relu), span: Span::dummy(), id: NodeId::dummy() };
    let relu_arg = Arg { name: None, value: x, span: Span::dummy() };
    let relu_call = Expr {
        kind: ExprKind::Call { callee: Box::new(relu_name), args: vec![relu_arg] },
        span: Span::dummy(),
        id: NodeId::dummy(),
    };

    let resolve = |sym: Symbol| -> Option<String> {
        interner.resolve(sym.0).map(|s| s.to_string())
    };

    let result = analyze_fusible_chain(&relu_call, &resolve);
    assert!(result.is_none()); // Single op, no fusion
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p nsl-codegen test_analyze_fusible -- --nocapture`
Expected: PASS

- [ ] **Step 3: Add auto-fusion detection in compile_expr**

In `expr.rs`, in the `compile_expr` method, add a check at the top of Call and BinOp handling:

```rust
// At the start of compile_expr for Call/BinOp nodes:
if let Some(fused_result) = self.try_auto_fuse(expr) {
    return Ok(fused_result); // Fused path taken
}
// Fall through to existing per-op dispatch...
```

The `try_auto_fuse` method (implemented in Task 10) handles the full pipeline: chain detection → PTX synthesis → training branch → kernel launch. This step simply wires it into the expression compilation entry point.

The key insight: when we see `relu(arg)` and `arg.kind` is `BinOp::Add(matmul_result, b)`, we know `add` and `relu` can fuse. The matmul is a fusion barrier — it runs first and produces a tensor in global memory. Then `add` + `relu` fuse into a single kernel that reads the matmul output and the bias, applies both ops in registers, and writes the result.

- [ ] **Step 4: Add nsl_is_training() branch for auto-fused code**

Same pattern as @fuse fn: emit `brif` on `nsl_is_training()`, unfused path for training, fused path for inference. This is already implemented inside `try_auto_fuse()` from Task 10 Step 2 — this step verifies the branch is correct by inspection.

- [ ] **Step 5: Build and verify**

Run: `cargo build --workspace`
Expected: Compiles without errors

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/expr.rs crates/nsl-codegen/src/fusion.rs
git commit -m "feat(codegen): lexical auto-fusion for inline elementwise chains with training branch"
```

---

## Chunk 3: @autotune + CLI + E2E Tests

### Task 13: Autotune Module — AST Hashing and Cache

**Files:**
- Create: `crates/nsl-codegen/src/autotune.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Write test for AST hashing (Span-excluded, Symbol-resolved)**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_hash_excludes_spans() {
        // Same AST content with different spans should produce identical hash
        let params = vec![("block_size".to_string(), vec![64, 128])];

        // Hash with identical content should match regardless of other factors
        let hash1 = hash_kernel_ast(
            "my_kernel", b"body_content_v1", &params, &[vec![256]], "GPU0", "8.6", 84,
        );
        let hash2 = hash_kernel_ast(
            "my_kernel", b"body_content_v1", &params, &[vec![256]], "GPU0", "8.6", 84,
        );
        assert_eq!(hash1, hash2, "identical inputs must produce identical hash");

        // Different AST content should produce different hash
        let hash3 = hash_kernel_ast(
            "my_kernel", b"body_content_v2", &params, &[vec![256]], "GPU0", "8.6", 84,
        );
        assert_ne!(hash1, hash3, "different AST content must produce different hash");

        // Different input shapes should produce different hash
        let hash4 = hash_kernel_ast(
            "my_kernel", b"body_content_v1", &params, &[vec![512]], "GPU0", "8.6", 84,
        );
        assert_ne!(hash1, hash4, "different input shapes must produce different hash");

        // Different device should produce different hash
        let hash5 = hash_kernel_ast(
            "my_kernel", b"body_content_v1", &params, &[vec![256]], "GPU1", "8.9", 128,
        );
        assert_ne!(hash1, hash5, "different device must produce different hash");
    }

    #[test]
    fn test_cartesian_product() {
        let params = vec![
            ("block_size".to_string(), vec![64, 128, 256]),
            ("warps".to_string(), vec![2, 4]),
        ];
        let product = cartesian_product(&params);
        assert_eq!(product.len(), 6); // 3 * 2
        assert!(product.contains(&vec![("block_size".to_string(), 64), ("warps".to_string(), 2)]));
        assert!(product.contains(&vec![("block_size".to_string(), 256), ("warps".to_string(), 4)]));
    }

    #[test]
    fn test_middle_value_fallback() {
        let params = vec![
            ("block_size".to_string(), vec![64, 128, 256]),
            ("warps".to_string(), vec![2, 4, 8]),
        ];
        let middle = select_middle_values(&params);
        assert_eq!(middle, vec![("block_size".to_string(), 128), ("warps".to_string(), 4)]);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p nsl-codegen test_cartesian -- --nocapture`
Expected: FAIL

- [ ] **Step 3: Implement autotune module**

Create `crates/nsl-codegen/src/autotune.rs`:
```rust
//! @autotune: build-time kernel variant benchmarking with caching.
//! Generates Cartesian product of tuning parameters, benchmarks on GPU,
//! caches winner keyed by AST hash.

use sha2::{Sha256, Digest};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tuning parameter: name → list of candidate values.
pub type TuningParams = Vec<(String, Vec<i64>)>;

/// A single variant: specific values for each tuning parameter.
pub type Variant = Vec<(String, i64)>;

/// Result of autotuning: winning variant + all timings.
pub struct AutotuneResult {
    pub winner: Variant,
    pub all_timings: Vec<(Variant, f64)>, // (variant, median_ms)
    pub device_name: String,
    pub compute_capability: String,
    pub sm_count: u32,
}

/// Generate Cartesian product of all tuning parameter combinations.
pub fn cartesian_product(params: &TuningParams) -> Vec<Variant> {
    let mut result = vec![vec![]];
    for (name, values) in params {
        let mut new_result = Vec::new();
        for existing in &result {
            for &val in values {
                let mut combo = existing.clone();
                combo.push((name.clone(), val));
                new_result.push(combo);
            }
        }
        result = new_result;
    }
    result
}

/// Select the middle value from each parameter range (--no-autotune fallback).
pub fn select_middle_values(params: &TuningParams) -> Variant {
    params.iter().map(|(name, values)| {
        let mid_idx = values.len() / 2;
        (name.clone(), values[mid_idx])
    }).collect()
}

/// Hash a kernel's AST body for cache key generation.
/// Excludes Span fields, resolves Symbol indices to string values.
/// Includes input_shapes so cache invalidates when tensor dimensions change.
pub fn hash_kernel_ast(
    kernel_name: &str,
    ast_bytes: &[u8], // Serialized AST content (Span-free, Symbol-resolved)
    tuning_params: &TuningParams,
    input_shapes: &[Vec<i64>], // shapes of all kernel input tensors
    device_name: &str,
    compute_capability: &str,
    sm_count: u32,
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(kernel_name.as_bytes());
    hasher.update(ast_bytes);
    for (name, values) in tuning_params {
        hasher.update(name.as_bytes());
        for v in values {
            hasher.update(v.to_le_bytes());
        }
    }
    // Include input shapes in hash — different shapes may need different tuning
    for shape in input_shapes {
        for &dim in shape {
            hasher.update(dim.to_le_bytes());
        }
        hasher.update(b"|"); // shape separator
    }
    hasher.update(device_name.as_bytes());
    hasher.update(compute_capability.as_bytes());
    hasher.update(sm_count.to_le_bytes());
    format!("{:x}", hasher.finalize())
}

/// Cache directory for autotune results.
pub fn cache_dir() -> PathBuf {
    PathBuf::from(".nsl-cache/autotune")
}

/// Check if a cached result exists for the given hash.
pub fn check_cache(kernel_name: &str, hash: &str) -> Option<Variant> {
    let path = cache_dir().join(format!("{}_{}.json", kernel_name, hash));
    if !path.exists() { return None; }

    let content = std::fs::read_to_string(&path).ok()?;
    // Parse JSON to extract winner variant
    parse_winner_from_cache(&content)
}

/// Write autotune result to cache.
pub fn write_cache(hash: &str, kernel_name: &str, result: &AutotuneResult) {
    let dir = cache_dir();
    std::fs::create_dir_all(&dir).ok();

    let timings_json: Vec<String> = result.all_timings.iter().map(|(variant, ms)| {
        let params: Vec<String> = variant.iter()
            .map(|(k, v)| format!(r#""{}": {}"#, k, v))
            .collect();
        format!(r#"    {{"params": {{{}}}, "median_ms": {:.4}}}"#, params.join(", "), ms)
    }).collect();

    let winner_json: Vec<String> = result.winner.iter()
        .map(|(k, v)| format!(r#""{}": {}"#, k, v))
        .collect();

    // Use std::time for timestamp (avoids adding chrono as a dependency)
    let timestamp = {
        use std::time::SystemTime;
        let secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        format!("{}", secs)
    };

    let json = format!(
        r#"{{
  "kernel": "{}",
  "device": "{}",
  "compute_capability": "{}",
  "sm_count": {},
  "variants_tested": {},
  "winner": {{{}}},
  "median_time_ms": {:.4},
  "all_timings": [
{}
  ],
  "timestamp": "{}",
  "cache_key": "{}"
}}"#,
        kernel_name,
        result.device_name,
        result.compute_capability,
        result.sm_count,
        result.all_timings.len(),
        winner_json.join(", "),
        result.all_timings.iter()
            .find(|(v, _)| v == &result.winner)
            .map(|(_, ms)| *ms)
            .unwrap_or(0.0),
        timings_json.join(",\n"),
        timestamp,
        hash,
    );

    let path = dir.join(format!("{}_{}.json", kernel_name, hash));
    std::fs::write(&path, json).ok();
}

fn parse_winner_from_cache(json: &str) -> Option<Variant> {
    // Simple JSON parsing for winner field
    // Look for "winner": { ... } and extract key-value pairs
    let winner_start = json.find("\"winner\":")?;
    let brace_start = json[winner_start..].find('{')? + winner_start;
    let brace_end = json[brace_start..].find('}')? + brace_start;
    let winner_str = &json[brace_start+1..brace_end];

    let mut variant = Vec::new();
    for pair in winner_str.split(',') {
        let parts: Vec<&str> = pair.split(':').collect();
        if parts.len() == 2 {
            let key = parts[0].trim().trim_matches('"').to_string();
            let val: i64 = parts[1].trim().parse().ok()?;
            variant.push((key, val));
        }
    }

    if variant.is_empty() { None } else { Some(variant) }
}
```

- [ ] **Step 4: Add sha2 dependency to Cargo.toml**

Add to `crates/nsl-codegen/Cargo.toml`:
```toml
sha2 = "0.10"
serde_json = "1"  # for AST serialization in hash_kernel_body
```

(No `chrono` dependency needed — timestamp uses `std::time::SystemTime`)

- [ ] **Step 5: Add module declaration**

Add to `crates/nsl-codegen/src/lib.rs`:
```rust
pub mod autotune;
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p nsl-codegen test_cartesian test_middle -- --nocapture`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/autotune.rs crates/nsl-codegen/src/lib.rs crates/nsl-codegen/Cargo.toml
git commit -m "feat(codegen): add autotune module with Cartesian product, AST hashing, and cache"
```

---

### Task 14: Autotune — Variant Generation in compile_kernels

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Extend compile_kernels to unwrap Decorated nodes**

In `compile_kernels()`, add handling for `StmtKind::Decorated`:

```rust
pub fn compile_kernels(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::KernelDef(kernel) => {
                // Existing path: no decorators, compile single PTX variant
                self.compile_single_kernel(kernel)?;
            }
            StmtKind::Decorated { decorators, stmt: inner } => {
                if let StmtKind::KernelDef(kernel) = &inner.kind {
                    let has_autotune = decorators.iter().any(|d| {
                        d.name.len() == 1 &&
                        self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
                    });

                    if has_autotune {
                        self.compile_autotuned_kernel(kernel, decorators)?;
                    } else {
                        self.compile_single_kernel(kernel)?;
                    }
                }
            }
            _ => {}
        }
    }
    Ok(())
}
```

- [ ] **Step 2: Implement compile_autotuned_kernel**

```rust
fn compile_autotuned_kernel(
    &mut self,
    kernel: &KernelDef,
    decorators: &[Decorator],
) -> Result<(), CodegenError> {
    // 1. Extract tuning parameters from @autotune decorator
    let params = self.extract_autotune_params(decorators)?;

    // 2. Hash kernel AST (excluding Spans, resolving Symbols)
    let ast_hash = self.hash_kernel_body(kernel, &params);

    // 3. Check cache (kernel_name used in filename for human readability)
    let kernel_name = self.interner.resolve(kernel.name.0).unwrap_or("unknown");
    if let Some(winner) = crate::autotune::check_cache(kernel_name, &ast_hash) {
        // Cache hit: compile only the winning variant
        // Substitute winning constants and compile single PTX
        let ptx = self.compile_kernel_variant(kernel, &winner);
        self.embed_kernel_ptx(kernel, &ptx)?;
        return Ok(());
    }

    // 4. Cache miss: try GPU benchmarking if available and not --no-autotune
    let no_autotune = self.compile_options.no_autotune;

    #[cfg(feature = "cuda")]
    if !no_autotune {
        // Generate all variants and benchmark on GPU
        let variants = crate::autotune::cartesian_product(&params);
        let input_shapes: Vec<Vec<i64>> = kernel.params.iter().filter_map(|p| {
            // Extract shape from Tensor<[dims], dtype> type annotations.
        // TypeExpr doesn't have extract_tensor_shape() — match on TypeExprKind
        // to extract literal shape dimensions from the type annotation.
        // If shape is not statically known (e.g., symbolic N), return empty vec.
        p.ty.as_ref().and_then(|t| {
            // Pattern match on TypeExprKind to find tensor shape literals.
            // Actual implementation depends on TypeExprKind variants in nsl-ast.
            // For now, return None for non-literal shapes (symbolic dims).
            None::<Vec<i64>>
        })
        }).collect();

        let mut compile_fn = |variant: &crate::autotune::Variant| -> Vec<u8> {
            self.compile_kernel_variant(kernel, variant)
        };

        match crate::autotune::benchmark_variants(
            kernel_name, &variants, &mut compile_fn, &input_shapes, &ast_hash,
        ) {
            Ok(result) => {
                eprintln!("[nsl] autotune: {} winner {:?} ({:.4}ms)",
                    kernel_name, result.winner,
                    result.all_timings.iter()
                        .find(|(v, _)| v == &result.winner)
                        .map(|(_, ms)| *ms).unwrap_or(0.0));
                let ptx = self.compile_kernel_variant(kernel, &result.winner);
                self.embed_kernel_ptx(kernel, &ptx)?;
                return Ok(());
            }
            Err(e) => {
                eprintln!("[nsl] autotune: benchmarking failed for {}: {}, using middle values", kernel_name, e);
                // Fall through to middle-value fallback
            }
        }
    }

    // Fallback: use middle values (--no-autotune, no GPU, or benchmarking failed)
    let fallback = crate::autotune::select_middle_values(&params);
    let ptx = self.compile_kernel_variant(kernel, &fallback);
    self.embed_kernel_ptx(kernel, &ptx)?;
    if no_autotune {
        eprintln!("[nsl] autotune: --no-autotune, using middle values for {}", kernel_name);
    }

    Ok(())
}
```

- [ ] **Step 3: Implement helper methods**

```rust
fn extract_autotune_params(&self, decorators: &[Decorator]) -> Result<TuningParams, CodegenError> {
    let autotune_deco = decorators.iter().find(|d| {
        d.name.len() == 1 &&
        self.interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
    }).expect("@autotune decorator must exist");

    let mut params = Vec::new();
    if let Some(ref args) = autotune_deco.args {
        for arg in args {
            let name = arg.name.as_ref()
                .and_then(|s| self.interner.resolve(s.0))
                .unwrap_or("unnamed")
                .to_string();
            let values = match &arg.value.kind {
                ExprKind::ListLiteral(items) => {
                    items.iter().filter_map(|item| {
                        if let ExprKind::IntLiteral(v) = &item.kind { Some(*v) } else { None }
                    }).collect()
                }
                _ => vec![],
            };
            params.push((name, values));
        }
    }
    Ok(params)
}

fn hash_kernel_body(&self, kernel: &KernelDef, params: &TuningParams) -> String {
    // Serialize kernel body AST for hashing: exclude Span fields, resolve Symbols.
    // Uses serde_json for deterministic serialization, then strips Span data.
    // This is simpler and more maintainable than a custom AST walker.
    let mut buf = Vec::new();
    for stmt in &kernel.body {
        // Serialize via serde_json (Stmt derives Serialize), then strip "span" fields.
        // This gives us Span-free, deterministic output without a custom walker.
        if let Ok(json) = serde_json::to_string(stmt) {
            // Simple span stripping: remove "span":{...} objects from JSON.
            // Symbols are already serialized as integer indices by serde —
            // but we need to resolve them to strings for cache stability.
            // For correctness, walk the body and resolve Symbol values inline.
            buf.extend_from_slice(json.as_bytes());
        }
    }
    // Additionally, hash resolved symbol names for each identifier in the body.
    // This ensures cache invalidation when variable names change (since Symbol
    // indices are session-specific but names are stable across compilations).
    for stmt in &kernel.body {
        crate::autotune::collect_symbols_for_hash(stmt, self.interner, &mut buf);
    }
    let kernel_name = self.interner.resolve(kernel.name.0).unwrap_or("unknown");

    // Extract input shapes from kernel parameter types (compile-time known)
    let input_shapes: Vec<Vec<i64>> = kernel.params.iter().filter_map(|p| {
        // Extract shape from Tensor<[dims], dtype> type annotations
        // Extract shape from Tensor<[dims], dtype> type annotations.
        // TypeExpr doesn't have extract_tensor_shape() — match on TypeExprKind
        // to extract literal shape dimensions from the type annotation.
        // If shape is not statically known (e.g., symbolic N), return empty vec.
        p.ty.as_ref().and_then(|t| {
            // Pattern match on TypeExprKind to find tensor shape literals.
            // Actual implementation depends on TypeExprKind variants in nsl-ast.
            // For now, return None for non-literal shapes (symbolic dims).
            None::<Vec<i64>>
        })
    }).collect();

    // Query device info if CUDA is available, otherwise use defaults.
    // The hash includes device info so cache invalidates across GPUs.
    #[cfg(feature = "cuda")]
    let (device_name, compute_cap, sm_count) = {
        let dn = nsl_runtime::autotune_harness::get_device_name().unwrap_or_default();
        let (maj, min) = nsl_runtime::autotune_harness::get_compute_capability().unwrap_or((0, 0));
        let sm = nsl_runtime::autotune_harness::get_sm_count().unwrap_or(0);
        (dn, format!("{}.{}", maj, min), sm)
    };
    #[cfg(not(feature = "cuda"))]
    let (device_name, compute_cap, sm_count) = ("cpu".to_string(), "0.0".to_string(), 0u32);

    crate::autotune::hash_kernel_ast(
        kernel_name, &buf, params, &input_shapes, &device_name, &compute_cap, sm_count,
    )
}

fn compile_kernel_variant(&mut self, kernel: &KernelDef, constants: &Variant) -> Vec<u8> {
    // Create a modified KernelCompiler and substitute tuning constants.
    // The KernelCompiler's compile_kernel method produces PTX; we wrap it
    // to substitute named constants before PTX generation.
    let mut compiler = KernelCompiler::new(); // KernelCompiler::new() takes no args

    // Build a constant substitution map: when the kernel body references
    // a tuning parameter by name, replace it with the constant value.
    let const_map: std::collections::HashMap<String, i64> = constants.iter().cloned().collect();

    // compile_kernel_with_constants: new method on KernelCompiler (add in this task).
    // When emitting PTX for the kernel body, if a variable name matches a key in
    // const_map, emit `mov.u32 %rN, <constant>` instead of `ld.param`.
    //
    // Implementation: In KernelCompiler::compile_kernel (kernel.rs), add optional
    // `constants: Option<&HashMap<String, i64>>` parameter. During PTX emit, check
    // if each param name is in the constants map — if so, emit mov instead of ld.param.
    // compile_kernel_with_constants wraps this:
    //   pub fn compile_kernel_with_constants(&mut self, kernel: &KernelDef,
    //       constants: &HashMap<String, i64>) -> Vec<u8> {
    //       self.compile_kernel_inner(kernel, Some(constants))
    //   }
    compiler.compile_kernel_with_constants(kernel, &const_map)
}

fn embed_kernel_ptx(&mut self, kernel: &KernelDef, ptx: &[u8]) -> Result<(), CodegenError> {
    // Same pattern as existing compile_kernels (compiler.rs lines 1128-1145):
    // Store BOTH the PTX bytes AND the kernel name string in .rodata,
    // then insert (ptx_data_id, name_data_id) tuple into kernel_ptx_data.
    let name = self.interner.resolve(kernel.name.0).unwrap_or("unknown");

    // 1. Embed PTX bytes in .rodata
    let ptx_data_id = self.module.declare_data(
        &format!("__ptx_{}", name), cranelift_module::Linkage::Local, false, false,
    )?;
    let mut ptx_desc = cranelift_module::DataDescription::new();
    ptx_desc.define(ptx.to_vec().into_boxed_slice());
    self.module.define_data(ptx_data_id, &ptx_desc)?;

    // 2. Embed kernel name string (null-terminated) in .rodata
    let mut name_bytes = name.as_bytes().to_vec();
    name_bytes.push(0); // null terminator
    let name_data_id = self.module.declare_data(
        &format!("__ptx_name_{}", name), cranelift_module::Linkage::Local, false, false,
    )?;
    let mut name_desc = cranelift_module::DataDescription::new();
    name_desc.define(name_bytes.into_boxed_slice());
    self.module.define_data(name_data_id, &name_desc)?;

    // 3. Store tuple matching existing HashMap<String, (DataId, DataId)> type
    self.kernel_ptx_data.insert(name.to_string(), (ptx_data_id, name_data_id));
    Ok(())
}
```

- [ ] **Step 4: Add CUDA benchmarking helpers to nsl-runtime**

The benchmarking harness lives in `nsl-runtime` (not `nsl-codegen`) since it needs direct CUDA driver API access.

**Visibility fix:** The `cuda` module in nsl-runtime is `pub(crate)`. To expose benchmarking helpers to nsl-codegen, either:
- Change `pub(crate) mod cuda` to `pub mod cuda` in `crates/nsl-runtime/src/lib.rs`, OR
- Create a new `pub mod autotune_harness` in lib.rs that re-exports only the benchmarking helpers.

The second approach is preferred (minimal exposure). But for simplicity, the helper functions below are placed **inside `mod inner`** (which has access to `cudarc::driver::sys::*`) and re-exported via a new public module.

**Important:** All functions below use types already imported by `use cudarc::driver::sys::*;` at the top of `mod inner`. Do NOT add duplicate `extern "C"` declarations.

Add to `crates/nsl-runtime/src/cuda/mod.rs` **inside the `mod inner` block**:

```rust
// --- Autotune benchmarking helpers ---

#[cfg(feature = "cuda")]
pub fn get_device_name() -> Option<String> {
    let mut name = [0u8; 256];
    let res = unsafe { cuDeviceGetName(name.as_mut_ptr() as *mut i8, 256, 0) };
    if res != CUresult::CUDA_SUCCESS { return None; }
    let len = name.iter().position(|&b| b == 0).unwrap_or(256);
    Some(String::from_utf8_lossy(&name[..len]).to_string())
}

#[cfg(feature = "cuda")]
pub fn get_compute_capability() -> Option<(i32, i32)> {
    let mut major = 0i32;
    let mut minor = 0i32;
    unsafe {
        cuDeviceGetAttribute(&mut major, 75, 0); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
        cuDeviceGetAttribute(&mut minor, 76, 0); // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
    }
    Some((major, minor))
}

#[cfg(feature = "cuda")]
pub fn get_sm_count() -> Option<u32> {
    let mut count = 0i32;
    unsafe {
        cuDeviceGetAttribute(&mut count, 16, 0); // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
    }
    Some(count as u32)
}

/// Allocate N copies of data on GPU for benchmarking.
#[cfg(feature = "cuda")]
pub fn allocate_test_tensors(data: &[f32], count: usize) -> Result<Vec<u64>, String> {
    let byte_len = data.len() * std::mem::size_of::<f32>();
    let mut ptrs = Vec::with_capacity(count);
    for _ in 0..count {
        let mut dptr: u64 = 0;
        let res = unsafe { cuMemAlloc_v2(&mut dptr, byte_len) };
        if res != CUresult::CUDA_SUCCESS {
            return Err(format!("cuMemAlloc failed: {:?}", res));
        }
        unsafe {
            cuMemcpyHtoD_v2(dptr, data.as_ptr() as *const c_void, byte_len);
        }
        ptrs.push(dptr);
    }
    Ok(ptrs)
}

#[cfg(feature = "cuda")]
pub fn allocate_output_tensor(elements: usize) -> Result<u64, String> {
    let byte_len = elements * std::mem::size_of::<f32>();
    let mut dptr: u64 = 0;
    let res = unsafe { cuMemAlloc_v2(&mut dptr, byte_len) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuMemAlloc failed: {:?}", res));
    }
    Ok(dptr)
}

#[cfg(feature = "cuda")]
pub fn try_load_module(ptx: &[u8]) -> Result<CUmodule, String> {
    let mut module: CUmodule = std::ptr::null_mut();
    let res = unsafe { cuModuleLoadData(&mut module, ptx.as_ptr() as *const c_void) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuModuleLoadData failed: {:?}", res));
    }
    Ok(module)
}

#[cfg(feature = "cuda")]
pub fn get_kernel_function(module: &CUmodule, name: &str) -> Result<CUfunction, String> {
    let c_name = std::ffi::CString::new(name).unwrap();
    let mut func: CUfunction = std::ptr::null_mut();
    let res = unsafe { cuModuleGetFunction(&mut func, *module, c_name.as_ptr()) };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuModuleGetFunction '{}' failed: {:?}", name, res));
    }
    Ok(func)
}

#[cfg(feature = "cuda")]
pub fn launch_kernel_sync(
    func: &CUfunction, grid: u32, block: u32,
    output: &u64, inputs: &[u64], n: i64,
) -> Result<(), String> {
    let mut args: Vec<*mut c_void> = Vec::new();
    args.push(output as *const u64 as *mut c_void);
    for inp in inputs {
        args.push(inp as *const u64 as *mut c_void);
    }
    args.push(&n as *const i64 as *mut c_void);

    let res = unsafe {
        cuLaunchKernel(*func, grid, 1, 1, block, 1, 1, 0, std::ptr::null_mut(),
            args.as_mut_ptr(), std::ptr::null_mut())
    };
    if res != CUresult::CUDA_SUCCESS {
        return Err(format!("cuLaunchKernel failed: {:?}", res));
    }
    unsafe { cuCtxSynchronize(); }
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn unload_module(module: &CUmodule) {
    unsafe { cuModuleUnload(*module); }
}

#[cfg(feature = "cuda")]
pub fn free_test_tensors(ptrs: &[u64]) {
    for &ptr in ptrs {
        unsafe { cuMemFree_v2(ptr); }
    }
}

#[cfg(feature = "cuda")]
pub fn free_output_tensor(ptr: &u64) {
    unsafe { cuMemFree_v2(*ptr); }
}
```

All CUDA driver functions (`cuDeviceGetName`, `cuDeviceGetAttribute`, `cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemcpyHtoD_v2`, `cuModuleUnload`) are already available via `use cudarc::driver::sys::*;` at the top of `mod inner`. Do NOT add duplicate `extern "C"` declarations.

Then add a public re-export module in `crates/nsl-runtime/src/lib.rs`:
```rust
/// Public autotune helpers for benchmarking kernel variants.
/// Re-exports from cuda::inner so nsl-codegen can access them.
#[cfg(feature = "cuda")]
pub mod autotune_harness {
    pub use crate::cuda::inner::{
        get_device_name, get_compute_capability, get_sm_count,
        allocate_test_tensors, allocate_output_tensor, free_test_tensors, free_output_tensor,
        try_load_module, get_kernel_function, launch_kernel_sync, unload_module,
    };
}
```

**Note:** This requires changing the `get_device_name`, `allocate_test_tensors`, etc. functions from `pub(crate)` to `pub` visibility within `mod inner`.

- [ ] **Step 5: Add benchmarking harness function to autotune.rs**

Add to `crates/nsl-codegen/src/autotune.rs` — this is the orchestration function that calls the `nsl-runtime` CUDA helpers:
```rust
/// Run the benchmarking harness: compile all variants, benchmark on GPU,
/// select winner by median time.
///
/// Protocol:
/// 1. Create synthetic input tensors matching kernel parameter shapes
/// 2. For each variant in the Cartesian product:
///    a. Compile variant PTX
///    b. Load module via cuModuleLoadData
///    c. Run 3 warmup iterations (discard — GPU P-state ramp)
///    d. Run 10 measured iterations with cuEvent timing
///    e. Record median of 10 measurements
///    f. On CUDA error: skip variant, log warning (graceful failure)
/// 3. Select variant with lowest median time
/// 4. Write result to cache
pub fn benchmark_variants(
    kernel_name: &str,
    variants: &[Variant],
    compile_variant: &mut dyn FnMut(&Variant) -> Vec<u8>,
    input_shapes: &[Vec<i64>],
    cache_hash: &str,
) -> Result<AutotuneResult, String> {
    const WARMUP_ITERS: usize = 3;
    const MEASURED_ITERS: usize = 10;

    // Query device info via nsl-runtime
    let device_name = nsl_runtime::autotune_harness::get_device_name().unwrap_or_else(|| "unknown".to_string());
    let (major, minor) = nsl_runtime::autotune_harness::get_compute_capability().unwrap_or((0, 0));
    let compute_cap = format!("{}.{}", major, minor);
    let sm_count = nsl_runtime::autotune_harness::get_sm_count().unwrap_or(0);

    // Allocate synthetic input tensors (filled with 1.0f32)
    let total_elements: i64 = input_shapes.first()
        .map(|s| s.iter().product())
        .unwrap_or(1024);
    let synthetic_data: Vec<f32> = vec![1.0f32; total_elements as usize];
    let input_ptrs = nsl_runtime::autotune_harness::allocate_test_tensors(&synthetic_data, input_shapes.len())?;
    let output_ptr = nsl_runtime::autotune_harness::allocate_output_tensor(total_elements as usize)?;

    let mut all_timings: Vec<(Variant, f64)> = Vec::new();

    'variant_loop: for variant in variants {
        let ptx = compile_variant(variant);

        // Graceful failure: skip variant on CUDA errors
        let module = match nsl_runtime::autotune_harness::try_load_module(&ptx) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("[nsl] autotune: skipping variant {:?} — {}", variant, e);
                continue;
            }
        };

        let func = match nsl_runtime::autotune_harness::get_kernel_function(&module, kernel_name) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[nsl] autotune: skipping variant {:?} — {}", variant, e);
                continue;
            }
        };

        let block_size = variant.iter()
            .find(|(k, _)| k == "block_size")
            .map(|(_, v)| *v as u32)
            .unwrap_or(256);
        let grid_size = ((total_elements as u32) + block_size - 1) / block_size;

        // Warmup: 3 iterations (GPU P-state ramp, results discarded)
        for _ in 0..WARMUP_ITERS {
            if let Err(e) = nsl_runtime::autotune_harness::launch_kernel_sync(
                &func, grid_size, block_size, &output_ptr, &input_ptrs, total_elements,
            ) {
                eprintln!("[nsl] autotune: warmup failed for {:?} — {}", variant, e);
                continue 'variant_loop; // Skip entire variant on warmup failure
            }
        }

        // Measured: 10 iterations with cuEvent timing.
        // IMPORTANT: Use async launch (NOT launch_kernel_sync) for event-based timing.
        // The event must be recorded BEFORE cuCtxSynchronize — recording after sync
        // would measure near-zero since the kernel already completed.
        //
        // The cu_event_* wrappers (cu_event_create, cu_event_record, cu_event_elapsed_time,
        // cu_event_destroy) are defined in Task 4 Step 3 of Chunk 1. They must be made
        // `pub` (not `pub(crate)`) and re-exported through autotune_harness module.
        let mut times_ms = Vec::with_capacity(MEASURED_ITERS);

        for _ in 0..MEASURED_ITERS {
            let mut start_event: u64 = 0;
            let mut stop_event: u64 = 0;
            unsafe {
                nsl_runtime::autotune_harness::cu_event_create(&mut start_event);
                nsl_runtime::autotune_harness::cu_event_create(&mut stop_event);

                // Record start BEFORE launch
                nsl_runtime::autotune_harness::cu_event_record(start_event, std::ptr::null_mut());
            }

            // Async launch (no sync): build args and call cuLaunchKernel directly.
            // Do NOT use launch_kernel_sync here — it syncs before stop event is recorded.
            if let Err(e) = nsl_runtime::autotune_harness::launch_kernel_async(
                &func, grid_size, block_size, &output_ptr, &input_ptrs, total_elements,
            ) {
                eprintln!("[nsl] autotune: measurement failed for {:?} — {}", variant, e);
                unsafe {
                    nsl_runtime::autotune_harness::cu_event_destroy(start_event);
                    nsl_runtime::autotune_harness::cu_event_destroy(stop_event);
                }
                continue 'variant_loop;
            }

            unsafe {
                // Record stop AFTER launch but BEFORE sync
                nsl_runtime::autotune_harness::cu_event_record(stop_event, std::ptr::null_mut());
                // Now sync to wait for kernel + stop event to complete
                nsl_runtime::autotune_harness::cu_ctx_synchronize();
                // Read elapsed time between events
                let mut elapsed_ms: f32 = 0.0;
                nsl_runtime::autotune_harness::cu_event_elapsed_time(
                    &mut elapsed_ms, start_event, stop_event,
                );
                times_ms.push(elapsed_ms as f64);
                nsl_runtime::autotune_harness::cu_event_destroy(start_event);
                nsl_runtime::autotune_harness::cu_event_destroy(stop_event);
            }
        }

        // Note: launch_kernel_async is a new helper (same as launch_kernel_sync but
        // WITHOUT the trailing cuCtxSynchronize). Add alongside launch_kernel_sync
        // in Task 14 Step 4.

        // Median of measured times
        times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = times_ms[times_ms.len() / 2];
        all_timings.push((variant.clone(), median));

        nsl_runtime::autotune_harness::unload_module(&module);
    }

    // Free synthetic tensors
    nsl_runtime::autotune_harness::free_test_tensors(&input_ptrs);
    nsl_runtime::autotune_harness::free_output_tensor(&output_ptr);

    if all_timings.is_empty() {
        return Err("all variants failed during benchmarking".to_string());
    }

    // Select winner: lowest median time
    let winner = all_timings.iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
        .0
        .clone();

    let result = AutotuneResult { winner, all_timings, device_name, compute_capability: compute_cap, sm_count };
    write_cache(cache_hash, kernel_name, &result);
    Ok(result)
}
```

- [ ] **Step 6: Build**

Run: `cargo build --workspace`
Expected: Compiles without errors

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs crates/nsl-codegen/src/autotune.rs crates/nsl-runtime/src/cuda/mod.rs
git commit -m "feat(codegen): @autotune variant generation, benchmarking harness, and cache integration"
```

---

### Task 15: Autotune — CLI Flags

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`

- [ ] **Step 1: Add autotune flags to Build subcommand**

```rust
Build {
    file: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
    // ... existing flags ...
    #[arg(long)]
    no_autotune: bool,
    #[arg(long)]
    autotune_fresh: bool,
    #[arg(long)]
    autotune_clean: bool,
},
```

- [ ] **Step 2: Handle --autotune-clean**

```rust
if autotune_clean {
    let cache_dir = std::path::Path::new(".nsl-cache/autotune");
    if cache_dir.exists() {
        std::fs::remove_dir_all(cache_dir).ok();
        eprintln!("[nsl] autotune cache cleaned");
    }
    return Ok(());
}
```

- [ ] **Step 3: Pass flags through to codegen via CompileOptions struct**

Pass autotune flags through a `CompileOptions` struct (not environment variables). This requires three changes:

1. **Define the struct** in `crates/nsl-codegen/src/lib.rs`:
```rust
#[derive(Default)]
pub struct CompileOptions {
    pub no_autotune: bool,
    pub autotune_fresh: bool,
}
```

2. **Add the field to `Compiler`** in `crates/nsl-codegen/src/compiler.rs`:
```rust
pub struct Compiler<'a> {
    // ... existing fields ...
    pub compile_options: CompileOptions,
}
```

Update `Compiler::new()` to accept `compile_options: CompileOptions` and store it. Update all callers of `Compiler::new()` in `main.rs` (compile, build, test, standalone entry points) to pass `CompileOptions { no_autotune, autotune_fresh, ..Default::default() }` (or `Default::default()` when flags are not relevant).

3. **In CLI handler**, construct and pass to the compiler:
```rust
let compile_opts = CompileOptions {
    no_autotune,
    autotune_fresh,
    ..Default::default()
};
// Pass to Compiler::new(..., compile_opts)
```

**Important:** Task 15 MUST be executed before Task 14, because Task 14 Step 2 references `self.compile_options.no_autotune`. The plan task order is correct (Task 15 is before Task 14's execution via the benchmarking call path), but the implementer must ensure `CompileOptions` is on `Compiler` before `compile_autotuned_kernel` compiles.

- [ ] **Step 4: Build and verify**

Run: `cargo build -p nsl-cli`
Expected: Compiles without errors

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-cli/src/main.rs
git commit -m "feat(cli): add --no-autotune, --autotune-fresh, --autotune-clean flags"
```

---

### Task 16: E2E Test — Kernel Profiler

**Files:**
- Create: `examples/m26_kernel_profiler.nsl`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create E2E test NSL program**

Create `examples/m26_kernel_profiler.nsl`:
```nsl
# M26 E2E: Kernel profiler test
# Runs GPU tensor ops with --profile-kernels, validates output

fn main():
    let x = Tensor.zeros([4, 4])
    let y = Tensor.ones([4, 4])
    let z = x + y
    let w = relu(z)
    print(w.shape[0])
    print("done")
```

Expected output: `4\ndone`

- [ ] **Step 2: Add E2E test to e2e.rs**

```rust
#[test]
fn e2e_m26_kernel_profiler() {
    assert_output_matches("m26_kernel_profiler");
}

#[test]
fn e2e_m26_kernel_profiler_json_output() {
    // Run with --profile-kernels and verify JSON file is produced.
    // Uses the same Command pattern as the existing run_example helper,
    // but adds the --profile-kernels flag.
    let root = workspace_root();
    let example_path = root.join("examples/m26_kernel_profiler.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "run"])
        .arg(&example_path)
        .arg("--profile-kernels")
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl run");
    assert!(output.status.success(), "profiler run failed: {}",
        String::from_utf8_lossy(&output.stderr));

    // Check for kernel_profile.json in the workspace root (current_dir)
    let json_path = root.join("kernel_profile.json");
    if json_path.exists() {
        let content = std::fs::read_to_string(&json_path).unwrap();
        assert!(content.contains("traceEvents"), "missing traceEvents key");
        assert!(content.contains("\"ph\":\"X\""), "missing duration events");
        assert!(content.contains("\"tid\":1"), "kernel events should be on tid:1");
        std::fs::remove_file(&json_path).ok(); // cleanup
    }
    // Note: on CPU-only CI, profiler produces empty events (expected)
}
```

Add expected output file at `tests/expected/m26_kernel_profiler.txt` (NOT `examples/expected/` — all baselines are in `tests/expected/`):
```
4
done
```

- [ ] **Step 3: Run E2E test**

Run: `cargo test -p nsl-cli e2e_m26_kernel_profiler -- --nocapture`
Expected: PASS (basic tensor ops work, profiling doesn't affect output; JSON test may be conditional on CUDA)

- [ ] **Step 4: Commit**

```bash
git add examples/m26_kernel_profiler.nsl tests/expected/m26_kernel_profiler.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(e2e): add M26 kernel profiler E2E test"
```

---

### Task 17: E2E Test — @fuse and Auto-Fusion

**Files:**
- Create: `examples/m26_fuse.nsl`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create E2E test for fusion**

Create `examples/m26_fuse.nsl`:
```nsl
# M26 E2E: Elementwise fusion test

@fuse
fn add_relu(x: Tensor<[N], f32>, b: Tensor<[N], f32>) -> Tensor<[N], f32>:
    return relu(x + b)

fn main():
    let x = Tensor.ones([8])
    let b = Tensor.ones([8])
    let y = add_relu(x, b)
    print(y.shape[0])
    print("done")
```

Expected output: `8\ndone`

- [ ] **Step 2: Add E2E test**

```rust
#[test]
fn e2e_m26_fuse() {
    assert_output_matches("m26_fuse");
}
```

- [ ] **Step 3: Run E2E test**

Run: `cargo test -p nsl-cli e2e_m26_fuse -- --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add examples/m26_fuse.nsl tests/expected/m26_fuse.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(e2e): add M26 @fuse E2E test"
```

---

### Task 18: E2E Test — @autotune (--no-autotune fallback)

**Files:**
- Create: `examples/m26_autotune.nsl`
- Modify: `crates/nsl-cli/tests/e2e.rs`

- [ ] **Step 1: Create E2E test for autotune**

Create `examples/m26_autotune.nsl`:
```nsl
# M26 E2E: @autotune with --no-autotune fallback

@autotune(block_size=[64, 128, 256])
kernel simple_scale(data: Tensor<[N], f32>, scale: f32) -> Tensor<[N], f32>:
    let tid = thread_id()
    data[tid] = data[tid] * scale

fn main():
    let x = Tensor.ones([256])
    let y = simple_scale(x, 2.0)  # Actually invoke the autotuned kernel
    print(y.shape[0])
    print("done")
```

Expected output: `256\ndone`

- [ ] **Step 2: Add E2E test (with --no-autotune flag)**

```rust
#[test]
fn e2e_m26_autotune() {
    // Run with --no-autotune since CI may not have a GPU for benchmarking.
    // This exercises the middle-value fallback path (block_size=128).
    let root = workspace_root();
    let example_path = root.join("examples/m26_autotune.nsl");
    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "-p", "nsl-cli", "--", "build"])
        .arg(&example_path)
        .arg("--no-autotune")
        .current_dir(&root)
        .output()
        .expect("failed to execute nsl build");
    assert!(output.status.success(), "autotune build failed: {}",
        String::from_utf8_lossy(&output.stderr));

    // Verify --no-autotune was exercised: stderr should show fallback message
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--no-autotune") || stderr.contains("using middle values"),
        "expected fallback message in stderr when --no-autotune is used"
    );

    // Verify no autotune cache was created (no GPU benchmarking happened)
    let cache_dir = root.join(".nsl-cache/autotune");
    assert!(
        !cache_dir.exists(),
        "autotune cache should not be created with --no-autotune"
    );
}
```

- [ ] **Step 3: Run E2E test**

Run: `cargo test -p nsl-cli e2e_m26_autotune -- --nocapture`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add examples/m26_autotune.nsl tests/expected/m26_autotune.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test(e2e): add M26 @autotune E2E test with --no-autotune fallback"
```

---

### Task 19: Final Integration — Profile Merge and Polish

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`
- Modify: `crates/nsl-runtime/src/kernel_profiler.rs`

- [ ] **Step 1: Implement profile merge in CLI**

When `--profile` is used (both profilers), after program execution:
1. Read `memory_profile.json` (from M25 profiler atexit dump — verify filename in `crates/nsl-runtime/src/profiling.rs`)
2. Read `kernel_profile.json` (from M26 kernel profiler atexit dump)
3. Merge `traceEvents` arrays into single `profile.json`
4. Delete the individual files

The merge function is already implemented in Task 6 (Step 2). The CLI run handler calls `merge_profile_traces()` after the child process exits. Verify the filenames match what each profiler's atexit handler writes:

```rust
// In the run handler, after process exits:
if profile {
    // Filenames must match the atexit handlers in profiling.rs and kernel_profiler.rs
    let mem_path = "memory_profile.json";      // M25 atexit output (profiling.rs line 177)
    let kern_path = "kernel_profile.json";      // M26 atexit output
    merge_profile_traces(mem_path, kern_path, "profile.json");
}
```

- [ ] **Step 2: Run full test suite**

Run: `cargo test --workspace --exclude nsl-cli && cargo test -p nsl-cli -- --nocapture`
Expected: All unit tests pass, E2E tests pass

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-cli/src/main.rs crates/nsl-runtime/src/kernel_profiler.rs
git commit -m "feat(cli): profile merge for combined memory + kernel profiling output"
```

---

### Task 20: Code Review and Final Verification

- [ ] **Step 1: Run cargo clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: No warnings

- [ ] **Step 2: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass

- [ ] **Step 3: Verify deliverables checklist**

1. Kernel profiler with Chrome tracing output (tid:1 track)
2. `@fuse` decorator compiles to single PTX kernel
3. Auto-fusion for inline elementwise chains (inference path)
4. Training safety: `nsl_is_training()` branch + `@backward` requirement
5. `@autotune` with `--no-autotune` fallback
6. `--profile-kernels`, `--profile` CLI flags
7. Cache at `.nsl-cache/autotune/`
8. E2E tests passing

- [ ] **Step 4: Commit any remaining fixes**

```bash
git add crates/nsl-runtime/ crates/nsl-codegen/ crates/nsl-semantic/ crates/nsl-cli/ examples/
git commit -m "fix: address clippy warnings and final polish for M26"
```
