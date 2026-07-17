# Compute-Stream Migration (Milestone C · p8, PR-A)

Change: `crates/nsl-runtime/src/{cuda/mod.rs,kernel_profiler.rs}`,
gate `crates/nsl-cli/tests/stream_migration_gpu_gate.rs`.

## 1. Why

Every kernel launch went to the **legacy NULL stream**
(`current_stream() == null`), and CUDA cannot capture the legacy stream —
`cuStreamBeginCapture` on it returns `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`.
CUDA-graph replay of the training step (p8 PR-B) therefore requires all GPU
work to ride a *real* stream first. This PR is that migration, and nothing
else: no graphs, no new overlap, no behavior change.

## 2. The ordering argument (why this is semantically a no-op)

The new per-thread compute stream is created with **`CU_STREAM_DEFAULT`
(flags=0), a BLOCKING stream**. Blocking-stream semantics: the legacy NULL
stream is an implicit **two-way barrier** against every blocking stream — an
op issued to the NULL stream waits for all prior work on blocking streams, and
an op issued to a blocking stream waits for all prior NULL-stream work. Since
the runtime issues from one thread, every pair of operations that lands on
(NULL, compute) in either order is serialized exactly as when both were on
NULL: the **total order of GPU work is unchanged**.

That argument covers all four NULL-stream-ordering assumptions the scout
flagged as load-bearing:
- **sync-DtoH-as-barrier policy** (the p3 "no per-op sync" design): sync
  `cuMemcpyDtoH_v2` runs NULL-stream-ordered → still waits for all prior
  compute-stream kernels.
- **Sync HtoD uploads / DtoD copies**: same two-way barrier.
- **Event-based deferred frees** (#382): `defer_free_device` already records
  its completion event on `current_stream()` — it now records on the compute
  stream, where the consuming kernels actually run.
- **Offload transfer stream**: its ordering event is now recorded on
  `current_stream()` (the compute stream) — the state it copies out is written
  by compute-stream kernels; interleaved NULL-stream writes are ordered before
  that record by the same blocking semantics.

## 3. What changed

- `inner::current_stream()`: lazily-created per-thread blocking stream
  (mirrors the transfer-stream machinery); `NSL_LEGACY_NULL_STREAM=1` restores
  NULL-stream launches (read once, cached).
- Both `cuLaunchKernel` sites (`kernel_launch`, CFIE `launch_function_raw`)
  now pass `current_stream()`.
- **cuBLAS**: `sgemm_row_major` calls `cublasSetStream_v2(handle,
  current_stream())` per call (the handle is process-global, the stream
  per-thread; SetStream is a cheap handle-field write). Previously cuBLAS
  deliberately ran on the default stream to match `current_stream()` — the
  same invariant, maintained under the new answer.
- Profiler begin/end events (kernel_launch wrapper, `gpu_matmul_f32` wrapper,
  and the kernel-profiler base epoch) record on `current_stream()` so timings
  bracket the kernel on its actual stream. (`profiler/ffi.rs` already used
  `cu_event_record_on_current_stream` — its comment anticipated this exact
  migration.)
- NOT changed (correct via the blocking-semantics argument): sync memcpys,
  `cuMemPrefetchAsync` hint, the opt-in `NSL_ASYNC_ALLOC`
  `cuMemAllocAsync/cuMemFreeAsync` on the NULL stream, all `cuCtxSynchronize`
  sites.

## 4. Validation

- **Dedicated differential gate** (`stream_migration_gpu_gate`): the same CSHA
  training run (kernels + cuBLAS + deferred frees + sync memcpys interleaved)
  under `NSL_LEGACY_NULL_STREAM=1` vs default — **bit-identical trained-param
  sums** under `--deterministic`.
- **Existing gate battery re-run on the new stream**, all bit-exact:
  `deferred_free_gpu_gate` (event-based frees), `fase_fused_step_gpu_gate`
  (p9), `stream_ordering_gpu_gate` (48-step packed-GQA LM loss stream), plus
  `pretrain_loss_decreases_on_gpu` (real training loop).
- 822 `nsl-runtime` (cuda) lib tests pass (every GPU bit-exact differential
  from p4/p9 runs on the compute stream now).

## 5. What PR-B builds on this

With all kernels + gemms on one capturable stream, PR-B can attempt
`cuStreamBeginCapture_v2` capture-replay of the fwd+bwd region under strict
admission (no `.item()` readbacks, no fused-LCE `num_valid`, dropout-free,
digest-verified launch sequences, pointer-stability checks). The remaining
prerequisites are pointer stability for transients (arena Stage-2 or
`NSL_SKIP_GPU_DRAIN`-style pool pinning) and capture-legal handling of the
allocator — documented in the p8/p9 scout notes.
