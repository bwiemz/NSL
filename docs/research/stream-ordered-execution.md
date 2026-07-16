# Stream-Ordered Execution — remove per-op device sync (Milestone C · p3)

Change: `crates/nsl-runtime/src/cuda/mod.rs` (`sync_after_kernel` + 37 gated sites).
Gate: `crates/nsl-cli/tests/stream_ordering_gpu_gate.rs`.

## 1. Problem

The runtime launched a CUDA kernel per tensor op and then called
`cuCtxSynchronize()` before returning — a fully synchronous execution model. The
CPU blocked on every op until the GPU drained, so launch overhead serialized
with execution and there was zero CPU/GPU overlap. On a small model, where each
kernel is short, that per-op stall dominates wall-clock. #370 deferred removing
these syncs.

## 2. Why the per-op syncs are redundant

Two facts make a `cuCtxSynchronize` after a pure-GPU kernel unnecessary for
correctness:

1. **Every kernel launches on the NULL stream.** `inner::current_stream()`
   returns `null_mut()`; every `cuLaunchKernel` passes it, and cuBLAS uses its
   default (NULL) stream. NULL-stream work is issued and completes strictly
   in-order, so a kernel that consumes another kernel's output is already ordered
   after it. GPU→GPU dataflow needs no host-side barrier.
2. **Host reads go through the *synchronous* `cuMemcpyDtoH_v2`.** `memcpy_dtoh`
   (and therefore `.item()`, `.to(cpu)`, stats/`sum_sq`) is a blocking copy that
   is itself a NULL-stream barrier — it waits for all prior NULL-stream work
   before copying. So any host read is correctly ordered after the kernels that
   produced its data, with or without the per-op sync.

Block reuse is safe for the same reason: the caching allocator only marks a
freed block available (it does not `cuMemFree`), and a NULL-stream kernel that
reuses a block cannot start until the prior kernel using it has finished. The
single-stream serialization is what makes reuse race-free without per-op syncs.

The central launcher (`kernel_launch`, `launch_function_raw`) already dropped its
unconditional sync — its post-launch `cuCtxSynchronize` is gated on
`sync_mode_enabled()` (the `NSL_CUDA_SYNC` / `--cuda-sync` toggle), and flash
attention was already converted. The remaining syncs were *separate*
unconditional calls inside the per-op wrapper functions.

## 3. The change

A new policy helper:

```rust
#[inline]
pub(crate) fn sync_after_kernel() {
    if sync_mode_enabled() { unsafe { cuCtxSynchronize(); } }
}
```

Default: no-op (stream ordering carries correctness). With `NSL_CUDA_SYNC=1` it
restores an eager sync — a **bisection kill-switch**: if a result changes with it
off but matches with it on, a genuine host-read site was mis-gated.

This PR converts the **37 clean redundant (R) sites** in `cuda/mod.rs` — the core
tensor-op wrappers that launch a kernel and return a device tensor with no host
read: `gpu_elementwise_binary/unary` (+ inplace), `gpu_scalar_op`,
`gpu_matmul_f32`, `gpu_softmax`/`log_softmax`, `gpu_layernorm`/`rmsnorm`,
`gpu_bias_add`, `gpu_rotate_half` (RoPE), `gpu_embedding_lookup`/`backward`,
`gpu_gather`, `gpu_backward_binary`, the reductions (`sum_dim`, `max_dim`,
`global_sum`, deterministic variants), `gpu_scatter_add`, `gpu_conv2d`,
`gpu_dropout`, `gpu_clamp*`, `gpu_slice`, `gpu_strided_copy`, and the sparse
SpMM/SpMV family — from `cuCtxSynchronize()` to `inner::sync_after_kernel()`.

### What is deliberately kept

- **(H) host-read barriers** — `nsl_tensor_item`, `gpu_tensor_stats_f32`,
  `gpu_tensor_sum_sq_f32`, `gpu_maxpool2d_f32` (argmax D→H). These read device
  memory to host and must wait.
- **(T) transfers** — offload-stream drain, `.to(cpu)`, model-move error surface.
- **(A) allocator** — OOM-retry / drain syncs.
- **(P) profiling / timing / health** — NSL_PHASE_TIMING, kernel profiler,
  tape-start health check, and the already-gated launcher error-surfacing.
- **Free-safety and flash/fused fault-trap sites** (flash forward/backward,
  fused CE/adapter, raw-`cuMemFree` guards in flash prepass / CSHA activation
  free / precision-cast / KV-compress) — deferred to a follow-up. Some double as
  a GPU-error → CPU-fallback trap, and the raw-free ones need stream-ordered
  frees first. They are on the hot path but were left synced here to keep this
  PR's blast radius to the clean, review-verified core.

## 4. Validation

- **Correctness — differential bit-exactness.** `stream_ordering_gpu_gate.rs`
  runs the same `--deterministic` training fixture under `NSL_CUDA_SYNC=1`
  (eager, old behavior) and default (stream-ordered), and asserts the loss stream
  is **bit-identical at every one of 96 steps**. A removed sync that exposed a
  read-before-complete race would diverge. Result: identical.
- **No op-level regression.** 797 `nsl-runtime` cuda unit tests pass; CCR
  checkpoint parity (a different fixture + path) passes 3/3.
- **Speed.** On the packed-GQA fixture, eager **8.47 s → stream-ordered 2.62 s
  (3.24×)** end-to-end (including the fixed NSL-program compile); the training
  loop itself speeds up more.

## 5. Follow-up

- Convert the flash-attention and fused-CE/adapter (R) sites, folding their
  GPU-error → CPU-fallback trap into the `NSL_CUDA_SYNC` path so the default
  stays async.
- Make the raw-`cuMemFree` free-safety sites stream-ordered (event-based deferred
  free) so their guarding syncs can go too — the "stream-ordered allocator" half
  of p3's title. This also composes with p8 (CUDA graphs), which need stable
  device pointers and no per-op host syncs across the captured region.
