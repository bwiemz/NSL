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

## 5. Flash-attention + fused-kernel sites (done, follow-up PR)

The 14 remaining (R) syncs in the op files are now gated too:

- **Simple tails** (flash FA3-Hopper, SDPA-backward entry+tail, `fused_kl_ce`
  fwd/bwd, `fused_linear_ce` fwd/large/bwd, `fused_adapter` IA³, CSR-SpMM) →
  `sync_after_kernel()`. The SDPA-backward-entry sync only preceded
  `NslTensor::from_ptr`, which reads the host-side struct (the `.data` address),
  never device memory — so it was pure defensiveness.
- **Fault-trap sites** (flash SDPA fwd `nsl_sdpa_fused_forward`, backward
  `flash_attention_backward_gpu`, `fused_adapter` LoRA + GatedLoRA) capture the
  sync result to drive a GPU-error → CPU-fallback. Gated as
  `let rc = if sync_mode_enabled() { cuCtxSynchronize() } else { SUCCESS };` — the
  trap stays under `NSL_CUDA_SYNC=1`; by default `rc` is `SUCCESS`, so the check
  is a no-op and a real async fault surfaces later. Every free on these paths is
  `free_managed` (caching allocator), which is stream-safe.

**Conscious tradeoff (fault-trap sites).** Those four syncs previously caught an
*asynchronous* GPU execution fault (illegal address mid-kernel) and routed to the
graceful CPU fallback — the `#324` "never return silent zeros / corrupt
gradients" guard. Gated off by default, a genuine async fault is no longer caught
*here*; because a CUDA fault is sticky, it surfaces loudly at the next
synchronous `memcpy_dtoh` as a panic rather than a clean CPU fallback. This is
**not** a silent-wrong-results risk (the `#324` guarantee holds — it is a loud
crash, never quiet corruption), and `NSL_CUDA_SYNC=1` fully restores the eager
fallback. The accepted net effect: on a real kernel fault (a bug in an
admission-guarded, validated kernel — rare) these four paths hard-crash instead
of degrading to CPU, which also stops such a bug from being silently masked.

Validated: the 797 `nsl-runtime` cuda unit tests (which directly exercise the
flash/fused functions) pass, and the differential gate stays bit-exact at 96/96
steps. The packed-GQA fixture's *default*-path speedup is unchanged (3.3×) — it
is elementwise/matmul-bound, so the flash/CE syncs were a small part of its
budget; the win is proportionally larger on attention-heavy / long-sequence
models where the fused SDPA fwd/bwd dominate.

## 6. Still remaining

- The raw-`cuMemFree` free-safety syncs (flash `PrepassScratch::drop`,
  `nsl_csha_free_backward_activations`, and the error-path free in
  `flash_attention_backward_gpu`) are KEPT: they guard a physical device free, so
  they need event-based deferred free (the "stream-ordered allocator" half of
  p3's title) before their syncs can go. This also composes with p8 (CUDA
  graphs), which need stable device pointers and no per-op host syncs across the
  captured region.
