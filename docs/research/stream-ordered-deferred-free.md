# Stream-Ordered Deferred Free — remove the sync before raw `cuMemFree` (Milestone C · p3-remainder)

Change: `crates/nsl-runtime/src/cuda/mod.rs` (`defer_free_device` +
`drain_completed_frees` + `drain_all_deferred_frees`) and the two raw-free hot
sites in `crates/nsl-runtime/src/flash_attention.rs`.
Gate: `crates/nsl-cli/tests/deferred_free_gpu_gate.rs`.

## 1. Problem

p3 (stream-ordered execution) gated away every per-op `cuCtxSynchronize`, but
kept the syncs in front of two **raw** `cuMemFree` sites on the CSHA hot path:

- `PrepassScratch::drop` — the Tier B.1 per-call x-scratch buffer.
- `nsl_csha_free_backward_activations` — the six HBM save buffers the CSHA
  forward fills for the fused backward (`q_proj`, `k_proj`, `v_proj`, `row_max`,
  `row_sum`, `x_raw`).

Both allocate with `alloc_device` (raw `cuMemAlloc_v2`) and free with
`free_device` (raw `cuMemFree_v2`), gated by a blocking `cuCtxSynchronize`. The
sync is genuinely required: a raw `cuMemFree` is **not** stream-ordered — it
returns the pages to the driver immediately, even if a NULL-stream kernel is
still reading the buffer. The sync guaranteed the consuming kernel had finished.

Why raw free and not the caching allocator (which is already stream-safe)?
Because these buffers are large and per-step, and the memory-reduction campaign
(1B on 16 GB) depends on them being **physically returned to the driver** rather
than pooled. Routing them through `free_managed` would keep the VRAM resident in
the caching pool and defeat that profile — so the raw-free semantics must be
preserved. What we want to drop is only the *host stall*, not the physical free.

## 2. The change — event-based deferred free

Replace the host barrier with a CUDA event:

```rust
pub(crate) fn defer_free_device_batch(ptrs: &[*mut c_void]) {
    // ... filter nulls, ensure_context ...
    if sync_mode_enabled() {                 // NSL_CUDA_SYNC=1 kill-switch
        cuCtxSynchronize();                  // old sync-then-free
        for p in live { free_device(p); }
        return;
    }
    let event = acquire_free_event();        // disable-timing, recycled
    cuEventRecord(event, current_stream());  // NULL stream, AFTER consumers
    DEFERRED_FREES.push_back((live, event));
    drain_completed_frees();                 // opportunistic
}
```

`drain_completed_frees` polls each queued event with `cuEventQuery` and physically
`free_device`s the entries whose event has completed. `drain_all_deferred_frees`
synchronizes once and frees the whole queue; it runs from `pool_drain` (the
"reclaim everything" path, invoked on OOM recovery and explicit cache-empty).

### Why it is safe

Every kernel and every event is issued on the **single NULL stream**, which is
strictly in-order. The event is recorded *after* the buffer's consuming kernels,
so `cuEventQuery` cannot return `CUDA_SUCCESS` until those kernels have retired.
We free only on `CUDA_SUCCESS` — therefore the raw `cuMemFree` provably runs
after the last read. No host stall; identical physical-free semantics.

The RAII ordering is load-bearing: `PrepassScratch` and the six CSHA buffers must
be handed to `defer_free_device*` only *after* the consuming `kernel_launch`
returns (i.e. after the launch is enqueued on the NULL stream). Dropping the
handle earlier would record the completion event before the kernel and reopen the
race. The call sites already satisfy this (`_prepass_handle` outlives the launch;
`nsl_csha_free_backward_activations` is invoked right after the backward launch).

### Bounded, leak-free

The queue is drained on every `defer_free_device` call, on OOM recovery, and in
`pool_drain`, so it stays small and never holds VRAM indefinitely. Events are
recycled through a pool (`CU_EVENT_DISABLE_TIMING`), so steady state creates no
new events. The `nsl_debug_gpu_mem` report prints the pending count so a deferred
hold-over is never mistaken for a leak.

## 3. Validation

- **Correctness — differential bit-exactness.** `deferred_free_gpu_gate.rs`
  trains a `@flash_attention` model with `--source-ad --deterministic` (which
  dispatches the CSHA fused backward and frees the six save buffers every step)
  under `NSL_CUDA_SYNC=1` (eager sync-then-free) and default (deferred). The four
  trained-parameter sums are **bit-identical** across the two modes over three
  SGD steps — with parameters genuinely moving (e.g. `wv` 1024 → 1000.55), so the
  deferred-free path is actually exercised. A free-before-read race would corrupt
  the gradients and diverge.
- **No leak.** `test_deferred_free_device_reclaims_vram` loops
  `alloc_device`→`defer_free_device` 50× on a 4 MB buffer and asserts, after
  `drain_all_deferred_frees`, that driver-reported free VRAM dropped by < 128 MB
  (a broken deferred free would leak ~200 MB). The CSHA-specific
  `csha_free_backward_activations_returns_memory_to_driver` regression asserts the
  same through the real free FFI.
- **Guards.** `test_deferred_free_handles_null_and_empty` covers null pointers,
  all-null batches, and draining an empty queue.
- **No op-level regression.** The 799 `nsl-runtime` cuda unit tests (which
  directly exercise the flash/CSHA functions) pass, and the p3
  `stream_ordering_gpu_gate` stays bit-exact.

## 4. Out of scope

The Phase-2 error-path free in `flash_attention_backward_gpu` frees via
`free_managed` (stream-safe caching allocator, not raw `cuMemFree`) on a cold
launch-failure / async-fault path that returns 0 to trigger the CPU fallback —
no hot-path cost, so its defensive drain is left as-is.
