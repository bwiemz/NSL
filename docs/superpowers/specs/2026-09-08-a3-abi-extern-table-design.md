# Roadmap A3 — `nsl-abi` as the typed source of the runtime C-ABI

**Roadmap criterion:** *Make `nsl-abi` the single typed source of truth: a
declarative `extern_table!` macro (or a `build.rs` from a TOML manifest) that
generates (a) the Rust `extern "C"` signatures in runtime, (b) the Cranelift
`Signature` declarations in codegen, (c) the C header, and (d) the Python
ctypes mirror. Move the shared constants into `nsl-abi`. Then drop the
codegen → runtime edge. The text-parsing cross-checker becomes a one-line
`assert_eq!(generated, checked_in)`.*

This is a design spec, not a plan: it says what the one table is, how each of
today's copies becomes a rendering of it, what the codegen still takes from
the runtime once the table exists, and the order of steps that get to "no
edge" under gates that already exist. No step changes a symbol's signature or
the emitted CLIF; every step is gated on `signature_agreement`,
`c_header_agreement` and the 28 train-block CLIF snapshots staying green.

## Where it stands

Two of the criterion's four clauses are done:

- **Shared constants.** `nsl_abi::wire` holds the dtype tags, the tensor
  header's data offset and magic, the plan bits, the allocation-surface tags
  and the ABI version. Both crates read them from there; the compiler no
  longer imports a constant from the runtime.
- **Cross-check.** `nsl-abi` parses the three copies as *text* and compares
  them: `signature_agreement` (every `RUNTIME_FUNCTIONS*` table in the codegen
  against the runtime's `extern "C"` items) and `c_header_agreement` (the
  C-API header against `c_api/mod.rs`). `check_workspace` reads every `.rs`
  under both crates. The gate works — it is what made the constants move
  safe — but it is a validator of three hand-written copies, not a source.

What the three copies are today:

| Copy | Where | Size |
|---|---|---|
| Runtime implementations | `#[unsafe(no_mangle)] extern "C" fn` across `nsl-runtime/src` | 813 functions |
| Codegen declarations | 15 `RUNTIME_FUNCTIONS*` tables (`builtins/{tensor,collections,scalar,io,memory}.rs`, `runtime_abi/{training,diagnostics,tensor,interop,inference,distributed,quantization,memory,optimizer}.rs`, one in `expr/literals.rs`), each row `(name, &[cranelift Type], Option<Type>)` | ~370 rows |
| C header | emitted by `nsl_codegen::c_wrapper::emit` (the `--emit-header` path) from a fixed-surface prototype list hand-written in `c_wrapper.rs`; `c_header_agreement` parses the emitted text against `c_api/mod.rs` | the `nsl_model_*` / `nsl_tensor_*` / `nsl_get_last_error` surface, ~20 prototypes |
| Python mirror | `python/nslpy/_bridge.py` sets `argtypes`/`restype` by hand for the functions it calls | 6 assignments |

The codegen declares only the 370 it emits calls to; the other ~440 runtime
externs are the C API, the Python bridge, kernels called by other runtime
code, and test hooks. They are outside the table and stay outside it; the
header generator (below) is what covers the C-API subset.

**The edge.** `nsl-codegen` depends on `nsl-runtime` (with `interop`), so a
compiler build pulls the runtime's whole tree: `cargo tree -e normal` lists
248 crates for `nsl-codegen`, 113 of them the runtime's own closure. What the
codegen actually takes from the runtime is small and sorts into three kinds:

| Kind | Symbols (file) | Count |
|---|---|---|
| Logging | `nsl_runtime::nsl_log!` (every codegen diagnostic since roadmap C3) | 371 sites |
| Data layouts and records | `c_api::NslTensorDesc` (`c_wrapper.rs`), `c_api::nsl_abi_version`, `awq::AwqScales::from_blob` (`calibration/binary_codegen.rs`), `calibration_data::peek_batch_seq`, `param_plan::PLAN_*`, `train_config_record`, `env_record` (`lib.rs`, `stmt.rs`, `runtime_abi/optimizer.rs`) | 9 |
| Compile-time device probes | `CUDA_SUPPORT_COMPILED`, `cuda_device_identity`, `cuda_device_name`, `CudaDeviceIdentity` (`stmt_train/identity.rs`, `gpu_specs.rs`), `flash_attention::{select_backward_blocks, nsl_flash_attention, nsl_flash_attention_csha}` (`flash_attention.rs`, `autotune.rs`), `pca_tier_b_runtime` | 8 |

The first kind is the C3 work: it moved every diagnostic onto one macro, and
the macro lives in the runtime because the subscriber that keeps stderr
byte-identical does. The second kind is *format* knowledge (a blob layout, a
`repr(C)` struct, a record schema) that both sides must agree on — exactly
what `wire` is for. The third kind is real: the compiler runs kernels at
compile time to autotune and to identify the device, and that needs the
runtime to be linked *when that feature is on*.

## The table

One declaration per runtime function the codegen calls, written once, in
`nsl-abi`, in a form every consumer can expand without parsing text. The form
is an X-macro — `nsl-abi` is dependency-free and must stay so, and a
`macro_rules!` table costs nothing at build time and needs no `build.rs`:

```rust
// crates/nsl-abi/src/table.rs
#[macro_export]
macro_rules! for_each_runtime_fn {
    ($m:ident) => { $m! {
        // group    name                    params            ret   runtime path                         feature
        [tensor]    nsl_tensor_add          (i64, i64)     -> i64  tensor::arithmetic::nsl_tensor_add   ;
        [tensor]    nsl_tensor_free         (i64)          -> ()   tensor::nsl_tensor_free              ;
        [interop]   nsl_safetensors_load    (i64, i64, i64)-> i64  safetensors_io::nsl_safetensors_load [interop];
        [training]  nsl_adamw_step          (i64, i64, i64, f64, f64, f64, f64, i64) -> ()  optim::nsl_adamw_step ;
        // …
    } };
}
```

Each row carries what the four renderings need and nothing else:

- **group** — the codegen table it belongs to today (`tensor`, `training`,
  …), so the codegen's `all_runtime_functions()` keeps its grouping and the
  `feature_composition_gate` needle sets keep matching.
- **name** — the link symbol.
- **params / ret** — in `AbiScalar` spelling (`i64`, `f64`, `f32`, `i32`,
  `()`): the register class and width, which is what the ABI is. A row cannot
  say `I64` on one side and `f64` on the other because there is one side.
- **runtime path** — the Rust path of the implementation, so the runtime's
  rendering can name it (below). This is the one column the text checker
  could not have: it knows the symbol exists, not which item it is.
- **feature** — the cargo feature the implementation is behind (`interop`,
  `cuda`, `nccl`), so each rendering can `cfg` it the same way.

`nsl-abi` invokes the macro once itself to build `pub static RUNTIME_ABI:
&[FnDecl]` (`FnDecl { group, name, params: &[AbiScalar], ret:
Option<AbiScalar>, feature: Option<&str> }`), which is what the header and
Python generators and the existing `cross_check` consume.

## The four renderings

**(b) Codegen — Cranelift declarations.** `runtime_abi/mod.rs` invokes
`for_each_runtime_fn!` with a macro that emits one `const
RUNTIME_FUNCTIONS_<GROUP>: &[(&str, &[types::Type], Option<types::Type>)]`
per group, mapping `i64 → types::I64`, `f64 → types::F64`, `f32 → types::F32`,
`i32 → types::I32`, `() → None`. The 15 hand-written tables are deleted; the
`all_runtime_functions()` iterator, the builtin dispatch and every
`declare_runtime_fn(name)` caller are unchanged because the constants keep
their names and shape. A row behind a feature is emitted under
`#[cfg(feature = …)]` exactly where the hand-written table was.

**(a) Runtime — the signatures are checked, not generated.** The 813
implementations are hand-written functions with bodies; nothing generates
them. What the table gives the runtime is a *typed* check with no text
parsing: `nsl-runtime/src/abi_check.rs` invokes the macro with

```rust
macro_rules! assert_abi_impl {
    ($([$g:ident] $name:ident ($($p:ty),*) -> $r:ty $path:path $([$feat:literal])? ;)*) => { $(
        $(#[cfg(feature = $feat)])?
        const _: unsafe extern "C" fn($($p),*) -> $r = $crate::$path;
    )* };
}
```

A row whose params, return type or path disagree with the implementation is
a compile error in the runtime crate, with the row's name in it. This is the
"one-line assert" the criterion asks for, stronger than an
`assert_eq!(generated, checked_in)`: it is checked by `rustc`, so an `i64`
where the implementation takes `f64` cannot compile, and a renamed or removed
function cannot either. (`nsl_tensor_free` and the other `-> ()` rows render
as `-> ()`; the `unsafe` on the fn-pointer type matches the
`#[unsafe(no_mangle)] pub extern "C"` items, which are safe to name but
unsafe to call.)

**(c) C header.** The emitter exists (`c_wrapper::emit`, behind
`--emit-header`); what changes is where its fixed surface comes from. The
~20 prototypes hand-written in `c_wrapper.rs` become rows tagged `[capi]` (a
sixth column, or a second X-macro for the C-API surface — the C API is a
*different* set from the codegen's 370, overlapping on `nsl_tensor_*`), and
the emitter renders them from `RUNTIME_ABI`. `c_header_agreement` then has
nothing left to parse on the header side: the `[capi]` rows are covered by
the runtime's typed assertions (rendering (a)), and the test keeps only its
two structural checks (the ABI-version typedef and
`header_inlines_do_not_shadow_runtime_symbols`, whose inlines stay a
hand-written prologue the emitter writes verbatim).

**(d) Python mirror.** `nsl abi python` (a `nsl doc`-style subcommand, like
`nsl doc cli` / `nsl env list --markdown`) renders `python/nslpy/_abi.py`:
one `argtypes`/`restype` assignment per `[capi]` row, `c_int64`/`c_double`
from `AbiScalar`. `_bridge.py` imports it instead of writing the six by
hand, and the checked-in file is pinned by `assert_eq!(render(),
fs::read(path))`, the arrangement the generated wiki pages already use.

## Dropping the edge

With the table in place the codegen no longer needs the runtime for
*declarations*; the three kinds of remaining use each have a home:

1. **Logging → `nsl-log`.** A new dependency-free crate (well, `tracing`
   only) holding `nsl_log!` and an install hook: `nsl_log::set_installer(fn())`
   is called once by the runtime's `log::ensure_installed` path (the runtime
   is always linked into any process that runs compiled code or the CLI), and
   the macro calls `nsl_log::ensure_installed()`, which runs the hook if one
   is set. The runtime keeps the subscriber, the events-stream mirror and the
   allocation-free stderr path unchanged — the C3 byte-identity tests
   (`log_stderr_identity`) stay as they are. `nsl-codegen` and `nsl-cli`
   depend on `nsl-log`; `nsl_runtime::nsl_log` becomes a re-export so no
   call site changes.
2. **Data layouts and records → `nsl_abi::wire`.** `NslTensorDesc` is
   `repr(C)` with a documented 48-byte layout: it moves next to the dtype
   tags (the runtime re-exports it). `AwqScales`'s blob format, the
   `peek_batch_seq` record, `train_config_record` and `env_record` schemas
   are the same shape of thing — the *format* moves to `wire` (a struct and
   its `from_bytes`/`to_bytes`), the code that *produces* the data stays in
   the runtime. `nsl_abi_version` is already `wire::version::packed()`.
3. **Device probes → a feature.** `CUDA_SUPPORT_COMPILED`,
   `cuda_device_identity`, the FlashAttention autotune calls and
   `pca_tier_b_runtime` are the compiler executing runtime code at compile
   time. They stay, behind `nsl-codegen`'s existing `cuda` feature (which
   already implies `nsl-runtime/cuda`): the `nsl-runtime` dependency becomes
   `optional = true`, enabled by `cuda`, and the eight sites are
   `cfg(feature = "cuda")` with the CPU-build fallbacks they already have
   (`CUDA_SUPPORT_COMPILED == false` today is the CPU path). A CPU-only
   compiler build then has no runtime in its tree at all; a GPU build links
   it, as it must.

The measured outcome is the dependency closure: `cargo tree -e normal -p
nsl-codegen | sort -u | wc -l` goes from 248 to the Cranelift-plus-frontend
set (the runtime's 113 leave, minus the few both share), and a compiler-only
change stops rebuilding the runtime.

## Steps

Each step is one PR, mechanical, and gated on the existing checks plus the
CLIF snapshots.

1. **The table and the two Rust renderings.** Add `nsl_abi::table` with the
   370 rows (generated once from the hand-written tables by a script, then
   checked in), the codegen rendering replacing the 15 tables, and the
   runtime `abi_check.rs`. `signature_agreement` keeps running for this
   release as belt-and-braces — it must report zero mismatches, which is
   also the proof the transcription was exact — and is deleted in the next.
2. **Header from the table, Python generator.** `c_wrapper::emit` reads the
   `[capi]` rows; `nsl abi python` and its gate; delete the prototype parser
   (`parse_c_prototypes`) and the hand-written `argtypes`.
3. **`nsl-log`.** The macro and the install hook; `nsl_runtime::nsl_log`
   re-exports. Gated by the C3 byte-identity tests.
   *As landed:* the subscriber moved with the macro, and the hook is an
   `EventsMirror` the runtime registers (consulted at event time) rather
   than an installer that swaps in a runtime subscriber — an install hook
   loses the `NSL_EVENTS` mirror for good whenever the compiler's first
   line precedes the runtime's, since `tracing`'s global subscriber is set
   once. `nsl_runtime::nsl_log!` stays a thin wrapper (register the
   mirror, then `nsl_log::nsl_log!`), and the codegen and CLI call
   `nsl_log::nsl_log!` directly, so step 5 has nothing left to rename.
4. **`wire` structs and formats.** One PR per format, each with a
   round-trip test in `nsl-abi`.
   *As landed (device identity):* `wire::device_identity::CudaDeviceIdentity`,
   the four-field record the runtime's compile-time probe returns and the
   compiler keys its autotune cache on, so step 5 can `cfg` the probe
   itself without the compiler losing the type; the runtime re-exports it
   at `nsl_runtime::CudaDeviceIdentity`.
   *As landed, first PR:* `wire::tensor_desc::NslTensorDesc` (layout
   constant-asserted; the compiler's descriptor stride is its `sizeof`)
   and `wire::train_config::{MOMENT_KEYS, TRAJECTORY_KEYS}` (the record's
   schema, read by the renderer and the resume checker alike); the runtime
   re-exports both at their historical paths. *Second PR:*
   `wire::awq_scales` — the AWQ activation-scales blob's layout, `encode`,
   `AwqScales::from_blob` / `to_blob` and `AWQ_SIDECAR_KEY`; it replaced
   three hand-matched copies (the codegen's `calibration/awq_sidecar.rs`
   encoder + decoder, the runtime's decoder, and the inline encoder in
   `nsl_calib_write_sidecar`), and the runtime keeps only the JSON +
   base64 sidecar reader around it. Still in the runtime: `peek_batch_seq`
   (the calibration-data readers, which need `safetensors`) and the
   `env_record` renderer.
5. **Optional runtime dependency.** `cfg` the eight probe sites; CI's
   Ubuntu lane builds `nsl-codegen` with `--no-default-features` (no
   runtime) as well as the default, and the `cuda` job as today.
6. **Delete the text parser** (`parse_runtime_functions_table*`,
   `parse_externs_in_file`, `cross_check`) once nothing reads it; `nsl-abi`
   is then the table, the wire constants and the two generators.

## Non-goals

- No symbol's name, arity or types change. A row that does not match its
  implementation fails the build; fixing the implementation is a separate
  change with its own review.
- The ~440 runtime externs the codegen does not call are not tabled; the
  C-API subset is covered by the header rows, the rest (kernels, test hooks)
  by the runtime's own tests.
- No `build.rs`, no TOML: the X-macro is the manifest, readable in one file,
  and it keeps `nsl-abi` dependency-free.
- Step 3 does not change what any diagnostic prints; step 5 does not change
  what a GPU build links.
