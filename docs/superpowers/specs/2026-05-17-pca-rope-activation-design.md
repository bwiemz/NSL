# PCA RoPE-Reset + Tier A segment_ids Activation (Design)

**Date:** 2026-05-17
**Status:** Design — small follow-on PR to #179
**Parent specs:**

- [CFTP §4.3 RoPE position reset](2026-05-16-pca-rope-position-reset-design.md) (PR #179, merged at `0125f7ef`)
- [PCA Tier A](2026-04-18-pca-tier-a-design.md) (PRs #78/#105/#109, merged)

---

## §1 — Scope

Both PCA Tier A (`segment_ids_ptr`) and §4.3 RoPE-reset (`doc_starts_ptr`) shipped the codegen, producer, and FFI infrastructure for kernel-side support. Neither feature is *active* at runtime because all four Cranelift CSHA call sites pass `null` for `segment_ids_ptr` and `doc_starts_disabled_sentinel(0)` for `doc_starts_ptr`. The packer ([`packed_batch_to_dict`](../../../crates/nsl-runtime/src/packing.rs)) emits both tensors into the batch dict per step, but the consumer-side propagation from batch dict to FA-2 call site doesn't exist.

This spec wires it up. Single activation PR closes both features' "Task 11" debt simultaneously.

### Out of scope

- The fused linear+CE separator-skip kernel (CFTP §4.4) — deferred per [2026-05-16-cftp-section-4-4-deferral.md](2026-05-16-cftp-section-4-4-deferral.md).
- Multi-Sequence Per-CTA (CFTP §4.2 Strategy 3) — deferred per [2026-05-16-pca-strategy-3-per-cta-design.md](2026-05-16-pca-strategy-3-per-cta-design.md).
- New @flash_attention configurations (rope_q, segment_masked) — already handled by existing codegen.
- Inference-side activation — no batch-dict source at inference; this PR only activates the **training path** (train block step body).

---

## §2 — Constraint: forward() doesn't see batch

The Cranelift FA call sites live inside the model's compiled `forward` function. The batch dict lives in the train block's `step(batch)` scope. The user's source code looks like:

```nsl
model M:
    @flash_attention(...)
    fn forward(self, x: Tensor) -> Tensor:
        return scaled_dot_product_attention(q, k, v, scale)   # ← FA call site

train(model=m, dataset=ds):
    step(batch):                                                # ← batch in scope here
        let out = m.forward(batch.input_ids)                    # ← but m.forward() doesn't take batch
```

`forward` only sees `x` (passed by the user). `batch.segment_ids` and `batch.doc_starts` aren't reachable through `forward`'s parameters. The compiler can't add hidden parameters to `forward` without invasively changing the model's compiled signature.

## §3 — Mechanism: thread-local registry

**Decision (per brainstorm):** introduce a thread-local registry that the train block sets per step and the FA call sites read.

```rust
// crates/nsl-runtime/src/pca_rope_runtime.rs (new file)

use std::cell::Cell;

thread_local! {
    static PACKING_METADATA: Cell<(i64, i64)> = const { Cell::new((0, 0)) };
}

#[no_mangle]
pub extern "C" fn nsl_packing_metadata_set(segment_ids_ptr: i64, doc_starts_ptr: i64) {
    PACKING_METADATA.with(|c| c.set((segment_ids_ptr, doc_starts_ptr)));
}

#[no_mangle]
pub extern "C" fn nsl_packing_metadata_get_segment_ids() -> i64 {
    PACKING_METADATA.with(|c| c.get().0)
}

#[no_mangle]
pub extern "C" fn nsl_packing_metadata_get_doc_starts() -> i64 {
    PACKING_METADATA.with(|c| c.get().1)
}
```

Thread-local because:

- **Test isolation:** parallel tests don't interfere.
- **Multi-worker inference:** if multiple threads call into the same model, each has its own metadata (uninitialized → `(0, 0)` → identity path).
- **No locking overhead:** `Cell::set`/`Cell::get` are non-atomic single-thread reads/writes.

### Why this is safe

When the registry is uninitialized (no `set` has been called), `Cell::get()` returns `(0, 0)` — the spec-defined sentinel for "disabled / identity path." All existing inference call sites, calibration subprocesses, snapshot tests, and `@flash_attention` usages outside a train block see byte-stable PTX behavior.

### Two-FFI vs one-FFI getter

Two getters (`get_segment_ids` + `get_doc_starts`) rather than one returning a tuple because Cranelift FFI calls return a single value cleanly. Two separate i64 returns are simpler than tuple unpacking at the call site.

---

## §4 — Train block: set the registry per step

In [`stmt.rs`](../../../crates/nsl-codegen/src/stmt.rs), the train block's step body around line 3767-3779 already extracts `batch["input_ids"]` and `batch["labels"]` and prefetches them. Extend this block to also probe for `batch["segment_ids"]` and `batch["doc_starts"]` and set the registry accordingly:

```rust
// Probe whether the batch has packing metadata. The packer
// (packing.rs::packed_batch_to_dict) emits both tensors when
// DataLoaderConfig.packing=true; absent otherwise.
let k_seg = self.compile_string_literal(builder, "segment_ids")?;
let has_seg = self.compile_call_by_name(builder, "nsl_dict_contains", &[batch_val, k_seg])?;

let has_seg_block = builder.create_block();
let no_seg_block = builder.create_block();
let after_block = builder.create_block();
let has_seg_cond = builder.ins().icmp_imm(IntCC::NotEqual, has_seg, 0);
builder.ins().brif(has_seg_cond, has_seg_block, &[], no_seg_block, &[]);

builder.switch_to_block(has_seg_block);
builder.seal_block(has_seg_block);
let seg_tensor = self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_seg])?;
let k_doc = self.compile_string_literal(builder, "doc_starts")?;
let doc_tensor = self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_doc])?;
let seg_data_ptr = self.compile_call_by_name(builder, "nsl_tensor_data_ptr", &[seg_tensor])?;
let doc_data_ptr = self.compile_call_by_name(builder, "nsl_tensor_data_ptr", &[doc_tensor])?;
self.compile_call_by_name(builder, "nsl_packing_metadata_set", &[seg_data_ptr, doc_data_ptr])?;
builder.ins().jump(after_block, &[]);

builder.switch_to_block(no_seg_block);
builder.seal_block(no_seg_block);
let zero = builder.ins().iconst(cl_types::I64, 0);
self.compile_call_by_name(builder, "nsl_packing_metadata_set", &[zero, zero])?;
builder.ins().jump(after_block, &[]);

builder.switch_to_block(after_block);
builder.seal_block(after_block);
```

### Why probe at runtime, not codegen time

The dataset block's `packing = true/false` flag is known at compile time, but **the batch dict's actual contents are runtime data**. Probing handles:

- Mixed-batch workloads (rare but possible)
- DataLoader implementations that conditionally emit segment_ids based on actual document structure
- Test fixtures that compose multiple DataLoaders into a single train block

The runtime probe is `nsl_dict_contains` — one CStr comparison per step. Negligible cost.

## §5 — FA call sites: read the registry

Four call sites need updating. All currently pass `null` (for `segment_ids_ptr`) and `doc_starts_disabled_sentinel(builder)` (for `doc_starts_ptr`):

1. [`expr/advanced.rs:1689`](../../../crates/nsl-codegen/src/expr/advanced.rs#L1689) — `nsl_flash_attention_csha_with_saves` (with-saves inference path)
2. [`expr/advanced.rs:~1790`](../../../crates/nsl-codegen/src/expr/advanced.rs#L1790) — `nsl_flash_attention_csha` (plain inference path)
3. [`wengert_lower.rs:~580`](../../../crates/nsl-codegen/src/wengert_lower.rs#L580) — `nsl_flash_attention_csha_with_saves` (@train fused path)
4. [`wengert_lower.rs:~1907`](../../../crates/nsl-codegen/src/wengert_lower.rs#L1907) — `nsl_flash_attention_csha_backward` (@train fused backward)

At each site, **before** the `compile_call_by_name`, emit registry reads:

```rust
let seg_ptr = self.compile_call_by_name(
    builder, "nsl_packing_metadata_get_segment_ids", &[],
)?;
let doc_ptr = self.compile_call_by_name(
    builder, "nsl_packing_metadata_get_doc_starts", &[],
)?;
```

Then in the call's args array, replace `null` (segment_ids slot) and `doc_starts_v` (doc_starts slot) with `seg_ptr` and `doc_ptr`. Remove the now-unused `doc_starts_disabled_sentinel(builder)` hoist.

### Byte-stable identity path preserved

When the registry is uninitialized (inference, calibration, snapshot tests, non-train @flash_attention), both getters return `0`. The FA-2 entry points already null-check `segment_ids_ptr == 0` and `doc_starts_ptr == 0` to take the identity path. So existing test snapshots remain byte-stable at the PTX/Cranelift level.

### New `nsl_tensor_data_ptr` FFI

The registry stores raw device pointers (not `NslTensor*`). The train block needs a way to extract the `data` field from an `NslTensor*`. Add one new FFI:

```rust
// crates/nsl-runtime/src/tensor/mod.rs

#[no_mangle]
pub extern "C" fn nsl_tensor_data_ptr(tensor_ptr: i64) -> i64 {
    if tensor_ptr == 0 {
        return 0;
    }
    unsafe { (*(tensor_ptr as *const NslTensor)).data as i64 }
}
```

Self-contained, no CUDA dependency, no allocation. Used only by the train block's per-step registry update.

---

## §6 — FFI signature declarations

Add four new entries to [`builtins.rs::RUNTIME_FUNCTIONS`](../../../crates/nsl-codegen/src/builtins.rs):

| FFI | Params | Returns |
|-----|--------|---------|
| `nsl_tensor_data_ptr` | `[i64]` | `i64` |
| `nsl_packing_metadata_set` | `[i64, i64]` | none |
| `nsl_packing_metadata_get_segment_ids` | `[]` | `i64` |
| `nsl_packing_metadata_get_doc_starts` | `[]` | `i64` |

---

## §7 — Validation

### Layer 1 — Runtime registry roundtrip

In `crates/nsl-runtime/src/pca_rope_runtime.rs` `#[cfg(test)] mod tests`:

- Initial state returns `(0, 0)` (each getter returns 0).
- After `nsl_packing_metadata_set(0xAAAA, 0xBBBB)`, both getters return the corresponding values.
- Cross-thread isolation: spawn a thread, set values in it, parent thread still sees `(0, 0)`.

### Layer 2 — Codegen byte-stability for non-packing

Existing snapshot tests (`pca_forward_kernel_snapshot`, `pca_backward_kernel_snapshot`, `fa_v2_snapshots`, `pca_segment_mask_snapshot`) must remain green with **no `.snap.new` changes**, because:

- Snapshot configs use `segment_masked=false` and/or `rope_q=false` — gates don't fire.
- The new codegen path (registry getters) emits AT call sites that compile against the new builtins. Existing snapshots compile against the same builtins; the IR-level change is one extra `call` per site, which doesn't affect PTX snapshots (those are kernel-side).
- Cranelift IR snapshots (if any exist for the FA call site) would differ; check those explicitly.

### Layer 3 — End-to-end training smoke

New test in `crates/nsl-codegen/tests/pca_rope_activation_e2e.rs` (cfg-gated on `cuda` feature):

1. Build a 2-doc packed batch via `build_segment_ids_and_doc_starts`.
2. Compile a small CSHA model with `@flash_attention(segment_masked=true, rope_q=true)`.
3. Run one train step with packing-enabled DataLoader.
4. Verify forward output != identity-positions reference (proves the activation actually flipped the gate).
5. Verify second train step with packing-DISABLED batch sees identity behavior (proves registry resets correctly).

Without `--features cuda`, the test is skipped.

### Layer 4 — Cranelift call-site verification (cargo-test, no CUDA)

In `crates/nsl-codegen/tests/pca_rope_activation_call_sites.rs`:

- Compile a small train block with @flash_attention.
- Inspect the emitted Cranelift IR for the FA call site.
- Assert that `nsl_packing_metadata_get_segment_ids` and `nsl_packing_metadata_get_doc_starts` are called BEFORE the FA call.
- Assert the args passed to FA are the values returned by those getters (not iconst 0 or null).

This guards against regressions where future codegen changes accidentally drop the registry-read.

---

## §8 — Risks

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | Stale registry state leaks across steps. | Train block calls `nsl_packing_metadata_set` **every step** (both has-seg and no-seg branches). No path skips the call. |
| R2 | Registry set in train block, FA call site in a different thread (theoretical multi-worker training). | Thread-local registry isolates — if the FA call site runs on a different thread, it sees the uninitialized state and falls back to identity. Documented limitation; current training is single-threaded. |
| R3 | `nsl_tensor_data_ptr` returns garbage if tensor is freed. | Existing `NslTensor::magic` poison check guards `*tensor_ptr` deref invalidity at NSL's level. FFI returns whatever's in the data field — same risk as q_ptr/k_ptr through `csha_tensor_data_ptr`. Acceptable. |
| R4 | Calibration subprocess (AWQ) accidentally calls a CSHA kernel that reads the registry. | The registry getters return 0 for uninitialized state. Calibration subprocesses never call `nsl_packing_metadata_set`. Identity path; byte-stable. Already a concern dimension for #179, no change. |
| R5 | Per-step probe adds runtime overhead. | `nsl_dict_contains` is a single CStr lookup ~10-20 ns. Negligible vs. CUDA kernel launches (~10-100 μs). |
| R6 | Inference path (no train block) wants RoPE-reset. | Out of scope (§1). Future inference work can set the registry from a different entry point. |

---

## §9 — Test-fixture impact

Existing tests that depend on `null`/`sentinel-0` being passed at the FA call site:

- **AWQ calibration tests:** call CSHA via the inference path inside a calibration subprocess. Subprocess never runs a train block → registry uninitialized → 0 → identity path. **Unchanged.**
- **PCA forward/backward snapshot tests:** snapshot the kernel PTX, not the Cranelift IR. Kernel-side identity path unchanged. **Snapshots stay byte-stable.**
- **CSHA correctness tests** (`pca_tier_a_forward_correctness`, etc.): use `null` directly when constructing args. They call the FFI directly, bypassing the registry. **Unchanged.**
- **FFI sentinel tests** (`pca_rope_ffi_sentinel.rs`, `pca_rope_ffi_decls.rs`): test the FFI signature shape. **Unchanged.**

No existing test should fail. The activation only fires when the call sites are reached *through* a train block with a packing-enabled DataLoader.

---

## §10 — Institutional references

- **IR-001 — API-shape-enforced invariants:** registry getters can't return wrong-typed values (i64 by signature).
- **IR-002 — external references as one-time anchors:** §4 train-block context cites PR #179's deferral note (T11) once.
- **IR-003 — pre-implementation verification:** verified in this design via exploration of stmt.rs:3767-3779 (existing batch-prefetch pattern), `nsl_dict_contains` existence, `NslTensor::data` layout, and the four current call-site sentinel paths.
- **IR-009 — bounded dormancy:** PR #179's dormancy clock for RoPE-reset stops on this activation PR's merge.
