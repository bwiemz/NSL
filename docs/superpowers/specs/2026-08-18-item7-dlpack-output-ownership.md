# Item 7 — DLPack output ownership: `call_into`, `call_alloc`, export signatures

**Goal (roadmap item 7 / Milestone G):** `NslModel.forward` returns *owned*
PyTorch tensors. Today the DLPack output path is unimplemented by construction
(`nsl_model_call_dlpack` passes zero-initialised output descs and every
tensor-returning export refuses), the working desc path both leaks the impl
result tensor every call *and* depends on that leak (callers read mirrored
shape/strides pointers into it), output buffers are sized by guessing
(`max(input_elems, 4096)` f32 — silent heap overrun for anything larger), and
scalar-returning exports reachable via `nsl_model_call` wild-read the scalar
value as an address.

## Design

### Ownership models (two new runtime FFIs)

**`nsl_model_call_into(model, name, inputs, n_in, outputs, n_out,
out_capacities: *const u64) -> rc`** — caller-allocates. Same dispatch as
`nsl_model_call` plus an explicit capacity contract:

- `out_capacities[i]` = allocated byte size of `outputs[i].data`. An
  undersized buffer **refuses** and the error text reports the required byte
  count (this is also the sizing answer for symbolic-dim outputs: probe, read
  the requirement, retry).
- Output desc contract: `outputs[i].shape` / `.strides` point at
  caller-owned arrays; `outputs[i].ndim` **on entry** is their slot capacity.
  The result's dims/strides are **deep-copied** into them (never pointer-
  mirrored into runtime-owned memory) and `ndim` is set to the result rank.
- The impl result tensor is **freed** after the copy — no per-call leak on
  this path.

**`nsl_model_call_alloc(model, name, inputs, n_in, out_dl: *mut *mut
DLManagedTensor, n_out) -> rc`** — NSL allocates, ownership transfers. Each
output slot receives a `DLManagedTensor*` whose **deleter releases the
underlying `NslTensor` exactly once** via `nsl_tensor_free` (the only correct
release primitive: refcount/owns_data/data_owner/slab-aware). Consuming the
capsule in `torch.from_dlpack` hands the memory to torch's GC — the returned
torch tensor is *owned*.

- Idempotence: the deleter nulls `(*managed).deleter` before releasing, so
  `nsl_dlpack_free` (the host-facing entry) is a no-op on a second call.
- Weight aliasing: if the captured result **is** a registered model weight
  pointer, its refcount is incremented before transfer, making
  deleter-vs-`nsl_model_destroy` order-independent. Views of weights are
  already order-safe through the `data_owner` refcount chain.
- Refusals (deferral-must-refuse): GPU-resident outputs (deleter runs on the
  consumer's GC thread; CUDA context discipline deferred), scalar-returning
  exports (no DLPack representation; use `call_into`), dtypes with no DLPack
  mapping (fp8/u16-token/u16-segment/custom — previously **mislabeled as
  f64** on export; now refused in both the owned and borrowed exporters).

`nsl_model_call_dlpack` / `nsl_model_forward_dlpack` are rewritten on top of
the alloc core — the "cannot allocate DLPack outputs" note and the dead
`rc==0` export loop are deleted in the same change, per the comment contract
at their site.

### How the runtime obtains the impl result (no new generated symbols)

The typed wrapper already passes the impl result `NslTensor*` to the runtime
via `nsl_tensor_to_desc_ffi(result_tensor, scratch_desc)`. The new entry
points arm a **thread-local dispatch mode** (RAII-disarmed) before invoking
the registry fn-ptr; the mode:

- captures the impl tensor pointer in `nsl_tensor_to_desc_ffi`,
- switches `nsl_dispatch_apply_result` to capacity-checked deep-copy (`Into`)
  or metadata-only pass-through (`Alloc`),
- lets the entry point free (Into) or wrap-and-transfer (Alloc) the captured
  tensor afterwards, including on every refusal path (leak-free refusal).

Dispatch is synchronous on the calling thread, so thread-local context is
sound; `nsl_model_call` with the mode unarmed behaves **bit-identically to
today** (alias-and-leak, pinned by existing tests). No `<name>__nsl_dispatch`
signature change → no registry-transmute hazard, no new MSVC generated-symbol
plumbing.

### Scalar returns — fixing the wild read

The dispatch wrapper stores scalar bits at scratch offset 0 (the `data`
field), then `nsl_dispatch_apply_result` dereferences them as an address.
Fix in codegen: scalar-returning exports now call a new runtime helper
`nsl_dispatch_apply_scalar_result(scratch, dst)` that stores the 8 bytes
**into** `dst.data` (capacity ≥ 8 under `Into`), `ndim=0`, dtype `f64`.
Dispatch support narrows to `Scalar(F64)` — other scalar dtypes get the
existing clean refusal stub instead of the wild read (typed symbols
unaffected).

### Introspection

**`nsl_model_get_export_signature(model, name) -> *const c_char`** — returns
per-export JSON (serialized `ExportInfo`: params + return with stringified
shapes incl. symbolic dims like `"B"`, dtypes, devices), valid until
`nsl_model_destroy`. Codegen embeds a `__nsl_export_sigs` blob (same
offset-table layout as `__nsl_export_names`) plus accessor
`nsl_get_export_signature_json(idx)`; `ExportRegistry` binds it at create
time (absent in older artifacts → clear refusal). nslpy uses it to size
`call_into` buffers exactly and to learn output arity.

### DLPack hygiene (prerequisites for trusting foreign structs)

- `DLDeviceType` becomes a plain `c_int` newtype with named constants —
  dereferencing a foreign `DLManagedTensor` carrying `kDLCUDAHost=3` (torch
  pinned memory) is UB with the current fieldless `repr(C)` enum. Unknown
  device types **refuse** on import instead of mapping to CPU.
- `PyCapsule_GetPointer` gets `restype=c_void_p` in nslpy (today the
  `DLManagedTensor*` is truncated to 32 bits above 4 GB).
- nslpy consumes capsules exactly once and no longer returns raw int
  pointers on conversion failure.

### ABI

`NSL_ABI_VERSION_MINOR` 0 → 1 (additive symbols; `NslTensorDesc` stays
48 bytes — capacity travels as a function parameter, never a struct field).
`nsl_model_call_dlpack` changing from always-refusing to functioning is
treated as minor: no working caller could depend on the refusal, and the
refusal-note text was explicitly contracted to be deleted with this change.

## Out of scope (recorded, deliberate)

- GPU-resident DLPack outputs (refused with a clear message).
- Multi-output (tuple) exports through the packed dispatch (still the
  existing refusal; typed symbols keep tuple support; the signature API
  reports true arity so hosts can tell).
- `nsl_model_backward` grad-desc free story (pre-existing leak, separate
  campaign).
- DLPack v1.0 versioned structs / read-only flag (v0.8 unversioned, as today).
