# Compiler state and thread-local globals

This document maps where NSL keeps mutable state during compilation and at
runtime, classifies the thread-local globals, and records the migration
direction toward explicit context objects. It exists because the
architecture-hardening review flagged "hidden thread-local state" as a risk:
state that flows through globals instead of through an explicit context is hard
to reason about, hard to test in isolation, and unsafe under parallelism.

**TL;DR:** `Compiler` is already the de-facto compile *session* object — most
state lives there as fields. A small number of thread-local globals bypass it.
Most are *legitimate* (test instrumentation, FFI error slots, RNG seeding,
runtime resource caches). A few carry *real compile/runtime behavior* and are
the migration targets. We are **not** doing the deep call-chain refactor now
(it would change public signatures and risk regressions for little immediate
gain); instead this doc fixes the direction so new code does not deepen the
debt.

---

## The session object: `Compiler`

`crates/nsl-codegen/src/compiler/mod.rs` defines `Compiler<'a>`, which already
holds the bulk of per-compilation state as ordinary fields: the module being
built, the interner/type-map, `compile_options: CompileOptions`, emission
caches (flash-attention, fused-CE), the inter-pass channels in `bus`, and
diagnostic/profiling state. Passes receive `&mut Compiler` and thread state
explicitly. This is the right shape — when state needs to be added, **add a
`Compiler` field (or a `CompileOptions` field), not a global.**

**Not every field is the same kind of thing, and the distinction is now
enforced.** Eight of them were the *pass bus*: values one pass produces for
another to consume — `last_wrga_plan`, `cpdt_plan`, `last_csha_bridge` and
friends. Among 74 fields nothing marked them out from a scratch counter or a
`Value`-keyed emission cache, and nobody could ask whether a published value
was ever read. They now live in
[`crates/nsl-codegen/src/pass_bus.rs`](../../crates/nsl-codegen/src/pass_bus.rs)
as `PassBus`, reached through `compiler.bus`. Its fields are private **to that
module** — not to `compiler/mod.rs`, which defines `Compiler` and could
otherwise still reach them — so an accessor is the only way in, and every
access is counted. See `NSL_PASS_TRACE` in
[Optimization-Passes.md](../wiki/Optimization-Passes.md) for what the counts
report.

When adding state, prefer the narrowest home that fits: a value flowing from
one pass to another belongs on the bus (add a `ChannelDescriptor`, or the drift
gate fails); a forward-to-backward buffer keyed by `cranelift::ir::Value`
belongs in the emission caches; configuration belongs on `CompileOptions`.

`CompileOptions` (in `crates/nsl-codegen/src/lib.rs`) is the configuration
half of the session. It is being decomposed from a flat "god-config" into
cohesive sub-structs (`WcetOptions`, `ZkOptions`, `WggoOptions`, `CshaOptions`,
`CpdtOptions`, …) as part of the same hardening effort.

A future `CompileSession { options, diagnostics, … }` wrapper could formalize
this, but it would be redundant churn unless it also absorbs the globals below.
The pragmatic path is: shrink the global set first, then decide whether a
wrapper still earns its keep.

---

## Thread-local inventory

Classification key:

- **TEST** — instrumentation read only by tests; compiled out or inert in
  production. Acceptable; leave as-is.
- **FFI/RUNTIME-OK** — thread-local is the *correct* design for a C-ABI or
  runtime concern (per-thread error slot, per-thread RNG, lazy resource cache).
  Acceptable; leave as-is.
- **MIGRATE** — carries real compile/runtime behavior through a global. Should
  eventually move into an explicit context. Do not add more of these.

The table is MACHINE-CHECKED: `thread_local_inventory_drift.rs` (nsl-runtime
integration tests) scans EVERY workspace crate for `thread_local!` statics and diffs
them against the rows between the markers below. A new thread-local that is
not added here — with a class and a reason — fails CPU CI. (The 2026-08-06
refresh found the previous hand-maintained table had drifted by ~20 statics,
including the autodiff `TAPE` itself.)

<!-- TL-INVENTORY-BEGIN -->
| Location | State | Class | Notes |
|----------|-------|-------|-------|
| `nsl-codegen/src/lib.rs` | `ADJOINT_OPS_DROPPED`, `ALLOC_SLOTS_PRE_HINT`, `ALLOC_SLOTS_POST_HINT`, `CONSUME_HINTS_CALLS` | TEST | Source-AD / allocator instrumentation counters, read via `debug_*` accessors by tests only. |
| `nsl-codegen/src/hir/ids.rs` | `WIRE_ID_COUNTER`, `REGISTER_ID_COUNTER`, `GENVAR_ID_COUNTER` | TEST | FPGA HIR id generation (macro-declared), reset by `KirToHirPass::lower`. Per-thread BY DESIGN: a process-global atomic would break snapshot tests under parallel sharding. Note: these do shape emitted HIR id values, so they are the one TEST entry with artifact influence — a lowering-context field is the eventual home. |
| `nsl-codegen/src/pass_trace.rs` | `CURRENT_PHASE` | FFI/RUNTIME-OK | Active driver phase for trace attribution — written during every production compile and read by `NSL_PASS_TRACE` reporting, so not test-only; diagnostic-only either way. |
| `nsl-codegen/src/pass_trace.rs` | `CURRENT_EPOCH` | MIGRATE (compile) | Per-compile attribution scope, added with the pass manager's ordering decision. Thread-local strictly BEATS the process-global counter it replaced (a compile never spans threads, and a process-global let one thread's compile claim another's pass invocations) — but unlike `CURRENT_PHASE` it is not diagnostic-only: `per_compile_view` is what the manager ENFORCES ordering from, so a wrong epoch is a wrong refusal. The only compile-side MIGRATE row; last in priority, not first, because its explicit home (an epoch owned by `PassManager` and handed to `record`) means threading a parameter through 32 ambient call sites in unrelated passes. |
| `nsl-codegen/src/pass_manager.rs` | `SCANNED_TAPES` | TEST | The scheduler's before/after tape digests, keyed `(epoch, pass)`. Written only when a pass is scheduled WITH a tape and read only by `assert_tape_unchanged_since`, whose sole caller is the WRGA arm — so it steers a refusal, not an artifact, and the digest is never consulted on a passing compile. Retired in `PassManager::drop`, so it cannot outlive its epoch. Thread-local for the same reason as `CURRENT_EPOCH`: a compile never spans threads, and a process-global map would let one thread's compile match another's digest. Added by #499 and MISSING from this table until the drift gate said so — the gate works. |
| `nsl-runtime/src/pca_rope_runtime.rs` | `PACKING_METADATA` | **MIGRATE** | Device pointers for `segment_ids`/`doc_starts` set per training step. Phase 3a (done): the fused-AD @flash_attention reads inside a train block are explicit Cranelift dataflow (`Compiler::packing_meta_vars`, per-micro-batch `def_var` at the stash) — no global read on that path. What remains (3b): the model-METHOD readers in `expr/advanced.rs`, a different Cranelift function, which need the hidden-param-vs-handle ABI decision; until then the setter and both getters stay. |
| `nsl-runtime/src/autodiff/mod.rs` | `TAPE` | MIGRATE (runtime) | THE autodiff tape — the heaviest behavioral thread-local in the codebase (27 access sites). Missing from every previous version of this table. |
| `nsl-runtime/src/tensor/mod.rs` | `TENSOR_SCOPE`, `TRAINING_MODE` | MIGRATE (runtime) | Scope-stack pointer and the global train/eval flag that gates tape recording. |
| `nsl-runtime/src/tensor/mod.rs` | `INPLACE_SUPPRESS_DEPTH` | MIGRATE (runtime) | FBIP in-place-suppression depth. Its only production readers were the (permanently disabled) refcount-elision predicates — writers are still emitted; the live reads are one test. A migration should first decide whether it still earns its writes. |
| `nsl-runtime/src/cuda/caching_allocator.rs` | `CURRENT_POOL` | MIGRATE (runtime) | Persistent-vs-transient pool selector; has RAII (`PoolGuard`, 2026-08-06) but is still a global channel. |
| `nsl-runtime/src/cuda/graph_capture.rs` | `ACTIVE`, `REGIONS`, `OCCURRENCE`, `NESTED_SKIP` | MIGRATE (runtime) | The CUDA-graph capture state machine (record vs replay vs skip). Self-contained cluster — a capture-context struct is the natural shape. |
| `nsl-runtime/src/transient_arena.rs` | `PIN`, `PLACED_AT` | MIGRATE (runtime) | The placed transient arena's single-shot placement channel: `PIN` arms `(payload_ptr, bytes, slot)` for the NEXT arena allocation, `PLACED_AT` records the payload pointer it was actually consumed at so the verify/unbind step can prove the binding rather than assume it. Thread-local because the allocation it steers happens inside an `extern "C"` whose signature has no room for an out-parameter — the same shape as `CURRENT_POOL`, and the same eventual fix (an explicit placement argument on the allocation path). |
| `nsl-runtime/src/tensor/mod.rs` | `OFFLOAD_DRAIN_TENSORS`, `OFFLOAD_DRAIN_DEVICE_BUFS` | RUNTIME-OK | In-flight async DtoH copy-back queues, deliberately paired with the per-thread `TRANSFER_STREAM` — converting to shared state would be a regression. |
| `nsl-runtime/src/memory.rs` | `ALLOC_REGISTRY` | RUNTIME-OK | ptr→size map that reconstructs the `Layout` for `nsl_free`. CORRECTNESS-CRITICAL and thread-affine (a cross-thread free cannot reconstruct the layout) — previously misfiled as test-only accounting. |
| `nsl-runtime/src/cuda/caching_allocator.rs` | `CURRENT_SURFACE`, `CURRENT_ALLOC_IDENTITY` | RUNTIME-OK | VRAM-surface tag and (op, tensor) attribution for allocation reports; RAII via `SurfaceGuard`. Metadata only. |
| `nsl-runtime/src/cuda/mod.rs` | `TRANSFER_STREAM`, `COMPUTE_STREAM`, `WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF` | RUNTIME-OK | `!Send` CUDA stream handles and persistent kernel workspaces — thread-local is the correct ownership model. |
| `nsl-runtime/src/muon_batch.rs` | `WS_CACHE` | RUNTIME-OK | Batched Newton-Schulz workspace cache, same model as `WS`. |
| `nsl-runtime/src/c_api/mod.rs` | `LAST_ERROR` | FFI/RUNTIME-OK | Per-thread C-ABI error slot (`nsl_get_last_error` / `nsl_clear_error`). |
| `nsl-runtime/src/c_api/mod.rs` | `DISPATCH_MODE` | FFI/RUNTIME-OK | Item-7 dispatch ownership mode (Into/Alloc), armed only for the synchronous span of one `nsl_model_call_into`/`_alloc` dispatch on the calling thread; the generated wrapper ABI is frozen (`ExportRegistry` transmutes it), so a thread-local side channel is the design, and it is armed via same-image `nsl_dispatch_ownership_*` FFIs because each artifact statically links its own runtime copy. |
| `nsl-runtime/src/cuda/mod.rs` | `OOM_CONTEXT` | FFI/RUNTIME-OK | Current-operation description used only to enrich OOM diagnostics. |
| `nsl-runtime/src/inspect/stream.rs` | `INSPECT_STREAM` | FFI/RUNTIME-OK | Lazily-initialized CUDA stream for inspection ops. |
| `nsl-runtime/src/sampling.rs` | `RNG` | FFI/RUNTIME-OK | Per-thread RNG seeded by `nsl_manual_seed`; per-thread seeding is the determinism model. |
| `nsl-runtime/src/tensor/mod.rs` | `STAGING_REGISTRY` | FFI/RUNTIME-OK | Write-once-at-init custom-dtype registry. |
| `nsl-runtime/src/tensor_trace.rs` | `RECORDER` | TEST | Tensor-op trace recorder; records only when armed. |
| `nsl-runtime/src/muon_prof.rs` | `OPEN` | FFI/RUNTIME-OK | Muon profiling scope stack — armed in production by `NSL_MUON_PROF` via the extern-C begin/end pair, so not test-only. |
| `nsl-runtime/src/trace.rs` | `TRACE_TEST_RECORDING` | TEST | `#[cfg(test)]`; exists precisely because a process-global flag let parallel test threads inject stray ops (the bug class this whole doc is about). |
| `nsl-runtime/src/memory.rs` | `ALLOC_COUNT`, `FREE_COUNT`, `ALLOC_BYTES`, `FREE_BYTES`, `CUDA_ALLOC_COUNT`, `CUDA_FREE_COUNT`, `CUDA_ALLOC_BYTES`, `CUDA_FREE_BYTES` | TEST | `#[cfg(test)]` fuzz counters (NOT feature-gated, as previously claimed). |
<!-- TL-INVENTORY-END -->

Retired: the `nsl-cli` build-path globals (`WRGA_TARGET_OVERRIDE` /
`WRGA_ABLATION_OVERRIDE` / `WRGA_PLAN_CAPTURE`) now live on
`CompileOptions::wrga_check` — Phase 2 below.

---

## Migration plan

Staged so each step is independently shippable and behavior-preserving.

**Phase 1 — document + contain (this doc).** Establish the inventory and the
rule: *new compile/runtime behavior threads through `Compiler` /
`CompileOptions` / an explicit runtime context, never a new thread-local.* A
reviewer should push back on any new `thread_local!` that is not TEST or
FFI/RUNTIME-OK.

**Phase 2 — retire the WRGA build-side globals (non-breaking). ✅ DONE.**
All three (`WRGA_TARGET_OVERRIDE`, `WRGA_ABLATION_OVERRIDE`, `WRGA_PLAN_CAPTURE`)
plus their RAII guards now live on `CompileOptions::wrga_check`
(`nsl_codegen::WrgaCheckContext`): the two overrides are CLI-applied onto
`WrgaInputs` by the WRGA bridge, and `plan_capture` is an
`Arc<Mutex<Option<WrgaPlan>>>` slot mirroring `cpdt.plan_out`. State now lives on
a stack-local `CompileOptions` that drops on return/panic, so it can no longer
leak across in-process CLI calls. Internal to the CLI→codegen boundary, so no
public C-ABI changes.

**Phase 3 — runtime contexts (larger).** The MIGRATE set is almost entirely
runtime-side: Phase 2 drained the compile-side well, and `CURRENT_EPOCH`
refilled exactly one row of it (see its note — it is deliberately last, and it
replaced something worse). The runtime side splits into three independent
clusters — do NOT build one mega-context:

1. **Per-train-step inputs**: `PACKING_METADATA`. Phase 3a landed: the
   in-function half (the wengert-lowered forward claim + csha backward,
   which share the train block's Cranelift function with the stash) reads
   function-local variables the stash `def_var`s per micro-batch — the
   CSLA window-backward's re-stash now feeds dataflow rather than
   compensating for a global surviving loop iterations. The residual 3b
   half is the model-method readers (`expr/advanced.rs`), a DIFFERENT
   Cranelift function: reaching them means either a hidden parameter in
   emitted method signatures or an opaque handle — an ABI change
   (`declaration.rs` signature build, `c_wrapper.rs`, FPGA lowering's
   params[0] assertion), which is why 3b needs its own design rather than
   a mechanical sweep.
2. **Autodiff/execution session**: `TAPE`, `TRAINING_MODE`, `TENSOR_SCOPE`,
   `INPLACE_SUPPRESS_DEPTH` (decide the last one's fate first — its
   production reader is dead).
3. **Capture/allocator state machines**: `graph_capture`'s four cells;
   `CURRENT_POOL` (RAII-guarded already); the transient arena's `PIN` /
   `PLACED_AT` pair, which is the same "steer the next allocation through a
   global because the FFI has no out-parameter" shape and should be solved
   once for all three.

Not a prerequisite for the compile-side cleanup, and the RUNTIME-OK rows are
not targets at all — several (`ALLOC_REGISTRY`, `OFFLOAD_DRAIN_*`, the CUDA
stream/workspace cells) are thread-affine for CORRECTNESS, and moving them
to shared state would be a regression, not a cleanup.

## What *not* to do

- Don't cfg-out or delete the TEST / FFI-OK thread-locals — they are correct.
- Don't refactor the deep `stmt → expr → kernel` call chains to thread a new
  context object in one shot; that is high-risk for low immediate value.
- Don't merge everything into one mega-context; split by concern (compile-time
  vs. runtime; session vs. per-train-step).
