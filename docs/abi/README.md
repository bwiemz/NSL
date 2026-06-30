# NSL Runtime C-ABI Contract

This document is the public contract for the boundary between NSL-generated
native code (and external C/C++/Python hosts) and `libnsl_runtime`. It
complements `crates/nsl-runtime/ARCHITECTURE.md` (which describes the runtime
internals) by stating the **stable, host-facing** ABI rules in one place.

> Tier: **Beta** (see [`STATUS.md`](../../STATUS.md)). The C ABI is the highest-
> risk surface in the project — see [`SECURITY.md`](../../SECURITY.md).

## Versioning

The ABI carries an explicit, checkable version:

- The runtime exports `int64_t nsl_abi_version(void)`, returning the version
  packed as `(major << 16) | minor`.
- Generated C headers `#define NSL_ABI_VERSION_MAJOR` / `NSL_ABI_VERSION_MINOR`
  / `NSL_ABI_VERSION`, pinned at generation time to the runtime's
  `nsl_runtime::c_api::NSL_ABI_VERSION_*` constants (single source of truth — a
  header can never claim a version the runtime didn't define).

**Compatibility rule for hosts:** after loading `libnsl_runtime`, call
`nsl_abi_version()` and compare:

- **major mismatch** → incompatible. Refuse to run.
- runtime **minor ≥** header minor (same major) → safe.
- runtime **minor <** header minor → the host may rely on additions the runtime
  lacks; refuse or degrade.

**When to bump (maintainers):**

- **Major** — any breaking change: an exported symbol's signature/semantics
  changes, a symbol is removed, or the `NslTensorDesc` layout changes.
- **Minor** — backward-compatible additions (new exported symbols, new optional
  trailing behavior).

The `NslTensorDesc` layout is pinned by **two** independent tests, so a drift
on either the Rust or the generated-header side is caught:

- A Rust golden test (`nsl_tensor_desc_abi_layout_is_pinned` in
  `crates/nsl-runtime/src/c_api/mod.rs`): size 48 bytes, 8-byte aligned, fixed
  field offsets.
- A C-side test (`generated_c_header_compiles_and_pins_abi_layout` in
  `crates/nsl-codegen/tests/c_header_compiles.rs`) that **compiles the generated
  C header with a real C compiler** and re-asserts the same `sizeof`/`offsetof`
  layout plus the `NSL_ABI_VERSION_*` macros via `_Static_assert`. This also
  guarantees `c_header::emit` keeps producing valid C.

If either test fails, you are making a **major** ABI change.

## FFI safety contract (every exported symbol)

These rules hold for **all** `extern "C"` functions in `nsl-runtime::c_api`.
They mirror the invariants documented in `crates/nsl-runtime/ARCHITECTURE.md`.

| Aspect          | Contract |
|-----------------|----------|
| **Ownership**   | Documented per function. Unless stated otherwise, pointer args are *borrowed* for the duration of the call; returned heap pointers are *owned* by the caller and freed via the matching `nsl_*_free` / `nsl_free_cstr`. |
| **Nullability** | A function must document whether each pointer arg may be NULL. Passing NULL where it is not allowed is undefined behavior. Lifecycle entry points (e.g. `nsl_model_*`) return 0/NULL on failure rather than trapping. |
| **Alignment / layout** | Structs crossing the boundary are `#[repr(C)]` and must match the generated header exactly (`NslTensorDesc` is the canonical example). |
| **Length**      | Array/pointer + count pairs must be consistent; the runtime trusts the provided count. |
| **Errors**      | Status is returned by value (sentinel/return code). Detail is retrievable via `nsl_get_last_error()`; clear with `nsl_clear_error()`. |
| **Unwinding**   | **No unwinding across the ABI.** Panics are treated as aborts; they must never propagate into foreign frames. |
| **Threading**   | Error state is thread-local. Functions document any shared/global state they touch. |

When adding a new exported symbol, document each of the above in its doc comment,
add it to the generated header if host-facing, and decide whether it is a
**minor** (additive) ABI bump.

## See also

- `crates/nsl-runtime/ARCHITECTURE.md` — runtime internals and the FFI safety
  contract at the source level.
- `crates/nsl-codegen/src/c_header.rs` — the generated-header emitter.
- [`STATUS.md`](../../STATUS.md), [`SECURITY.md`](../../SECURITY.md).
