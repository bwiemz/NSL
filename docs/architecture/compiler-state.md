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
built, the interner/type-map, `compile_options: CompileOptions`, pass-result
caches (`last_wrga_plan`, `cpdt_plan`, flash-attention caches, fused-CE
caches), and diagnostic/profiling state. Passes receive `&mut Compiler` and
thread state explicitly. This is the right shape — when state needs to be
added, **add a `Compiler` field (or a `CompileOptions` field), not a global.**

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

| Location | State | Class | Notes |
|----------|-------|-------|-------|
| `nsl-codegen/src/lib.rs` (`debug_hooks`) | `ADJOINT_OPS_DROPPED`, `ALLOC_SLOTS_*_HINT`, `CONSUME_HINTS_CALLS` | TEST | Source-AD / allocator instrumentation counters, read via `debug_*` accessors by tests only. |
| `nsl-codegen/src/hir/ids.rs` | `WIRE_ID_COUNTER`, `REGISTER_ID_COUNTER`, `GENVAR_ID_COUNTER` | TEST | FPGA HIR id generation with reset hooks so snapshot tests are deterministic. |
| ~~`nsl-cli` build path~~ | ~~`WRGA_TARGET_OVERRIDE`, `WRGA_ABLATION_OVERRIDE`, `WRGA_PLAN_CAPTURE`~~ | **DONE** | Retired (Phase 2 below). Now carried explicitly on `CompileOptions::wrga_check` (`nsl_codegen::WrgaCheckContext`), mirroring `cpdt.plan_out`. |
| `nsl-runtime/src/pca_rope_runtime.rs` | `PACKING_METADATA` | **MIGRATE** | Device pointers for `segment_ids`/`doc_starts` set per training step. Real runtime behavior via a global; would race across training threads. Should be an explicit per-step context. |
| `nsl-runtime/src/tensor/mod.rs` | `TENSOR_SCOPE`, `TRAINING_MODE` | MIGRATE (runtime) | Scope-stack pointer and the global train/eval flag that gates tape recording. Core runtime behavior; documented here for completeness but lower priority than the compile-side globals. |
| `nsl-runtime/src/cuda/caching_allocator.rs` | `CURRENT_POOL` | MIGRATE (runtime) | Persistent-vs-transient allocation pool selector; not re-entrant. An explicit allocator context would be safer. |
| `nsl-runtime/src/c_api/mod.rs` | `LAST_ERROR` | FFI/RUNTIME-OK | Per-thread C-ABI error slot (`nsl_get_last_error` / `nsl_clear_error`). Thread-local is the correct C-ABI pattern. |
| `nsl-runtime/src/cuda/mod.rs` | `OOM_CONTEXT` | FFI/RUNTIME-OK | Current-operation description used only to enrich OOM diagnostics. |
| `nsl-runtime/src/inspect/stream.rs` | `INSPECT_STREAM` | FFI/RUNTIME-OK | Lazily-initialized CUDA stream for inspection ops. |
| `nsl-runtime/src/sampling.rs` | `RNG` | FFI/RUNTIME-OK | Per-thread RNG seeded by `nsl_manual_seed`; per-thread seeding is the correct determinism model. |
| `nsl-runtime/src/tensor/mod.rs` | `STAGING_REGISTRY` | FFI/RUNTIME-OK | Write-once-at-init custom-dtype registry; read-heavy and stable. |
| `nsl-runtime/src/memory.rs` | `ALLOC_REGISTRY`, alloc/free counters | TEST / RUNTIME-OK | Allocation accounting; the counter half is gated behind `feature = "test-helpers"`. |

> Line numbers intentionally omitted — search for the symbol name; these
> drift. Re-run `rg "thread_local!" crates/` to refresh this table when the set
> changes, and update it in the same PR (same discipline as `STATUS.md`).

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

**Phase 3 — runtime contexts (larger).** `PACKING_METADATA`, `CURRENT_POOL`,
`TENSOR_SCOPE`, and `TRAINING_MODE` are runtime concerns that touch hot paths
and FFI; moving them needs care and its own design. Track separately; not a
prerequisite for the compile-side cleanup.

## What *not* to do

- Don't cfg-out or delete the TEST / FFI-OK thread-locals — they are correct.
- Don't refactor the deep `stmt → expr → kernel` call chains to thread a new
  context object in one shot; that is high-risk for low immediate value.
- Don't merge everything into one mega-context; split by concern (compile-time
  vs. runtime; session vs. per-train-step).
