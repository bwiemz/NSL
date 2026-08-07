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
integration tests) scans both crates for `thread_local!` statics and diffs
them against the rows between the markers below. A new thread-local that is
not added here — with a class and a reason — fails CPU CI. (The 2026-08-06
refresh found the previous hand-maintained table had drifted by ~20 statics,
including the autodiff `TAPE` itself.)

<!-- TL-INVENTORY-BEGIN -->
| Location | State | Class | Notes |
|----------|-------|-------|-------|
| `nsl-codegen/src/lib.rs` | `ADJOINT_OPS_DROPPED`, `ALLOC_SLOTS_PRE_HINT`, `ALLOC_SLOTS_POST_HINT`, `CONSUME_HINTS_CALLS` | TEST | Source-AD / allocator instrumentation counters, read via `debug_*` accessors by tests only. |
| `nsl-codegen/src/hir/ids.rs` | `WIRE_ID_COUNTER`, `REGISTER_ID_COUNTER`, `GENVAR_ID_COUNTER` | TEST | FPGA HIR id generation (macro-declared), reset by `KirToHirPass::lower`. Per-thread BY DESIGN: a process-global atomic would break snapshot tests under parallel sharding. Note: these do shape emitted HIR id values, so they are the one TEST entry with artifact influence — a lowering-context field is the eventual home. |
| `nsl-codegen/src/pass_trace.rs` | `CURRENT_PHASE` | TEST | Active driver phase for trace attribution. "A compile never spans threads" (its own doc); diagnostic only. |
| `nsl-runtime/src/pca_rope_runtime.rs` | `PACKING_METADATA` | **MIGRATE** | Device pointers for `segment_ids`/`doc_starts` set per training step — the "explicit step input" migration target (Phase 3). Real runtime behavior via a global; races if a step's data prep and its FA call ever land on different threads. |
| `nsl-runtime/src/autodiff/mod.rs` | `TAPE` | MIGRATE (runtime) | THE autodiff tape — the heaviest behavioral thread-local in the codebase (27 access sites). Missing from every previous version of this table. |
| `nsl-runtime/src/tensor/mod.rs` | `TENSOR_SCOPE`, `TRAINING_MODE` | MIGRATE (runtime) | Scope-stack pointer and the global train/eval flag that gates tape recording. |
| `nsl-runtime/src/tensor/mod.rs` | `INPLACE_SUPPRESS_DEPTH` | MIGRATE (runtime) | FBIP in-place-suppression depth. Its only production readers were the (permanently disabled) refcount-elision predicates — writers are still emitted; the live reads are one test. A migration should first decide whether it still earns its writes. |
| `nsl-runtime/src/cuda/caching_allocator.rs` | `CURRENT_POOL` | MIGRATE (runtime) | Persistent-vs-transient pool selector; has RAII (`PoolGuard`, 2026-08-06) but is still a global channel. |
| `nsl-runtime/src/cuda/graph_capture.rs` | `ACTIVE`, `REGIONS`, `OCCURRENCE`, `NESTED_SKIP` | MIGRATE (runtime) | The CUDA-graph capture state machine (record vs replay vs skip). Self-contained cluster — a capture-context struct is the natural shape. |
| `nsl-runtime/src/tensor/mod.rs` | `OFFLOAD_DRAIN_TENSORS`, `OFFLOAD_DRAIN_DEVICE_BUFS` | RUNTIME-OK | In-flight async DtoH copy-back queues, deliberately paired with the per-thread `TRANSFER_STREAM` — converting to shared state would be a regression. |
| `nsl-runtime/src/memory.rs` | `ALLOC_REGISTRY` | RUNTIME-OK | ptr→size map that reconstructs the `Layout` for `nsl_free`. CORRECTNESS-CRITICAL and thread-affine (a cross-thread free cannot reconstruct the layout) — previously misfiled as test-only accounting. |
| `nsl-runtime/src/cuda/caching_allocator.rs` | `CURRENT_SURFACE`, `CURRENT_ALLOC_IDENTITY` | RUNTIME-OK | VRAM-surface tag and (op, tensor) attribution for allocation reports; RAII via `SurfaceGuard`. Metadata only. |
| `nsl-runtime/src/cuda/mod.rs` | `TRANSFER_STREAM`, `COMPUTE_STREAM`, `WS`, `SR_WS`, `CE_SCRATCH`, `MUON_STATS_BUF` | RUNTIME-OK | `!Send` CUDA stream handles and persistent kernel workspaces — thread-local is the correct ownership model. |
| `nsl-runtime/src/muon_batch.rs` | `WS_CACHE` | RUNTIME-OK | Batched Newton-Schulz workspace cache, same model as `WS`. |
| `nsl-runtime/src/c_api/mod.rs` | `LAST_ERROR` | FFI/RUNTIME-OK | Per-thread C-ABI error slot (`nsl_get_last_error` / `nsl_clear_error`). |
| `nsl-runtime/src/cuda/mod.rs` | `OOM_CONTEXT` | FFI/RUNTIME-OK | Current-operation description used only to enrich OOM diagnostics. |
| `nsl-runtime/src/inspect/stream.rs` | `INSPECT_STREAM` | FFI/RUNTIME-OK | Lazily-initialized CUDA stream for inspection ops. |
| `nsl-runtime/src/sampling.rs` | `RNG` | FFI/RUNTIME-OK | Per-thread RNG seeded by `nsl_manual_seed`; per-thread seeding is the determinism model. |
| `nsl-runtime/src/tensor/mod.rs` | `STAGING_REGISTRY` | FFI/RUNTIME-OK | Write-once-at-init custom-dtype registry. |
| `nsl-runtime/src/tensor_trace.rs` | `RECORDER` | TEST | Tensor-op trace recorder; records only when armed. |
| `nsl-runtime/src/muon_prof.rs` | `OPEN` | TEST | Muon profiling scope stack. |
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

**Phase 3 — runtime contexts (larger).** The MIGRATE set is entirely
runtime-side now (the compile-side well ran dry with Phase 2), and it splits
into three independent clusters — do NOT build one mega-context:

1. **Per-train-step inputs**: `PACKING_METADATA`. Smallest and cleanest —
   the first target. The obstacle every cluster shares: these are reached
   across `extern "C"` boundaries from Cranelift-emitted code, so an
   explicit context means either a hidden parameter in emitted signatures
   or an opaque handle — an ABI change, which is why this phase needs its
   own design rather than a mechanical sweep.
2. **Autodiff/execution session**: `TAPE`, `TRAINING_MODE`, `TENSOR_SCOPE`,
   `INPLACE_SUPPRESS_DEPTH` (decide the last one's fate first — its
   production reader is dead).
3. **Capture/allocator state machines**: `graph_capture`'s four cells;
   `CURRENT_POOL` (RAII-guarded already).

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
