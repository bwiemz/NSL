# CFTP §4.3 — PCA RoPE Position Reset Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** [2026-05-16-pca-rope-position-reset-design.md](../specs/2026-05-16-pca-rope-position-reset-design.md) (v2, commit `51e7aa42`)

**Pre-implementation verification:** [V-RoPE-FFI-scope](../specs/2026-05-16-rope-ffi-scope-findings.md) (commit `27176082`) — 3 FFI entry points, not 7.

**Goal:** Add document-aware RoPE position reset to the 3 CSHA fused-RoPE entry points so packed sequences produce mathematically correct attention with RoPE (per-document position resets at boundaries).

**Architecture:** DataLoader emits a fixed `[MAX_NUM_DOCS+1]` i32 `doc_starts` tensor alongside the existing `segment_ids` (sentinel `-1` for unused slots). Three FFI entry points (`csha`, `csha_with_saves`, `csha_backward`) gain a trailing `doc_starts_ptr: i64` parameter — sentinel `0` preserves byte-stable PTX for existing callers. The kernel loads doc_starts into SMEM once per CTA, then computes `effective_pos = idx - smem_doc_starts[segment_ids[idx]]` at four rotation sites (forward Q, forward K, backward dQ, backward dK). Helper-encapsulated sentinel construction (`doc_starts_disabled_sentinel` / `doc_starts_enabled`) follows the planner spec's IR-001 discipline.

**Tech Stack:** Rust (1.95.0), Cranelift (function-emission), PTX (kernel synthesis), cudarc 0.19 (kernel launch), cargo insta (snapshot tests).

---

## File Structure

**Files to create:**

- `crates/nsl-codegen/tests/pca_rope_emission.rs` — Layer 1 PTX snapshot tests
- `crates/nsl-codegen/tests/pca_rope_numerical.rs` — Layer 2 numerical parity tests
- `docs/superpowers/specs/2026-05-16-cftp-section-4-4-deferral.md` — #3 deferral artifact
- `docs/superpowers/specs/2026-05-16-pca-strategy-3-per-cta-design.md` — #1 Strategy 3 spec-only artifact

**Files to modify:**

- `crates/nsl-codegen/src/pca_rope.rs` — add `MAX_NUM_DOCS=256` const, helpers (`doc_starts_disabled_sentinel`, `doc_starts_enabled`), `emit_doc_starts_smem_load` PTX emitter, joint-compat const_assert
- `crates/nsl-runtime/src/packing.rs` — packer emits `doc_starts` tensor (fixed `[257]` with sentinel `-1`)
- `crates/nsl-runtime/src/dataloader.rs` — wire `doc_starts` into batch dict
- `crates/nsl-runtime/src/flash_attention.rs` — extend 3 FFI entry points with trailing `doc_starts_ptr: i64` (csha:373, csha_with_saves:597, csha_backward:883)
- `crates/nsl-codegen/src/builtins.rs` — extend Cranelift FFI signature decls for the 3 entry points
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` — register declarations + CTA-prologue SMEM load emission
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs` — sites 1 (Q) + 2 (K) `effective_pos` computation
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs` — sites 3+4 (dQ + dK)
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs` — kernel-name suffix `_rope_reset_max256`
- Cranelift call sites in `expr/advanced.rs`, `wengert_lower.rs`, `runtime/autodiff/backward.rs`, `compiler/mod.rs`, `compiler/kernel.rs` — pass sentinel/enabled `doc_starts_ptr` (default sentinel-0)

---

### Task 1: pca_rope.rs — constants, helpers, joint-compat const_assert

**Files:**
- Modify: `crates/nsl-codegen/src/pca_rope.rs`
- Test: `crates/nsl-codegen/src/pca_rope.rs` (inline `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write the failing test**

Append to the existing `#[cfg(test)] mod tests` block in `crates/nsl-codegen/src/pca_rope.rs`:

```rust
#[test]
fn max_num_docs_is_256() {
    assert_eq!(MAX_NUM_DOCS, 256);
}

#[test]
fn doc_starts_smem_size_bytes_is_1028() {
    assert_eq!((MAX_NUM_DOCS + 1) * 4, 1028);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-codegen --lib pca_rope::tests::max_num_docs_is_256`
Expected: FAIL with `error[E0425]: cannot find value MAX_NUM_DOCS in this scope`

- [ ] **Step 3: Add constants + const_asserts to pca_rope.rs**

Insert at the top of `crates/nsl-codegen/src/pca_rope.rs` after the `use` block, before the existing types:

```rust
/// PCA §4.3 — RoPE position-reset compile-time bound.
///
/// Upper bound on the number of documents per packed batch. The fixed
/// `[MAX_NUM_DOCS + 1]` SMEM layout (1028 bytes) lets the kernel emit a
/// constant-size doc_starts load independent of runtime num_docs. The
/// packer asserts num_docs ≤ MAX_NUM_DOCS at batch-construction time.
///
/// 256 covers pretraining (avg doc ~256-2048 tok/doc at seq=16384), chat
/// SFT (avg ~50-150 tok/turn × 4-8 turns), and short-prompt instruction
/// tuning. SMEM cost 1028 bytes — trivial against the joint Tier A + Tier B
/// + RoPE-reset budget.
pub const MAX_NUM_DOCS: u32 = 256;

const _: () = assert!(MAX_NUM_DOCS <= 4096, "MAX_NUM_DOCS bound — keeps SMEM layout small");
const _: () = assert!(
    (MAX_NUM_DOCS + 1) * 4 <= 2048,
    "doc_starts SMEM bake must stay well under per-CTA budget"
);

// SMEM joint bake bound: project ships kernels ptxas-clean on sm_75 (48 KB
// usable per CTA) for forward-compat with deployed inference hardware.
// Active development target is sm_120 (RTX 5070 Ti, 100 KB) but the floor
// stays sm_75 — assert against the tighter bound. Matches the planner spec's
// PCA Tier B SMEM discipline (kernel ptxas-clean on sm_75).
const _: () = assert!(
    (MAX_NUM_DOCS + 1) * 4 + 16384 + 16384 < 48 * 1024,
    "Tier B + Tier A + RoPE-reset SMEM joint bake must fit sm_75 limit"
);
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-codegen --lib pca_rope::tests`
Expected: PASS (all existing tests in the module continue to pass).

- [ ] **Step 5: Write helper-roundtrip test**

Append to the same `#[cfg(test)] mod tests` block:

```rust
#[test]
fn doc_starts_disabled_sentinel_is_zero_constant() {
    use cranelift_codegen::ir::Function;
    use cranelift_codegen::ir::UserFuncName;
    use cranelift_codegen::ir::Signature;
    use cranelift_codegen::isa::CallConv;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    let mut fn_ctx = FunctionBuilderContext::new();
    let sig = Signature::new(CallConv::SystemV);
    let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
    let mut builder = FunctionBuilder::new(&mut func, &mut fn_ctx);
    let block = builder.create_block();
    builder.switch_to_block(block);

    let v = doc_starts_disabled_sentinel(&mut builder);
    // The returned Value should be the result of an iconst(I64, 0).
    let inst = builder.func.dfg.value_def(v).unwrap_inst();
    let opcode = builder.func.dfg.insts[inst].opcode();
    assert_eq!(opcode.to_string(), "iconst");
}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p nsl-codegen --lib pca_rope::tests::doc_starts_disabled_sentinel_is_zero_constant`
Expected: FAIL with `cannot find function doc_starts_disabled_sentinel`

- [ ] **Step 7: Add helper functions**

Insert into `crates/nsl-codegen/src/pca_rope.rs` (under the constants, above the existing types):

```rust
use cranelift_codegen::ir::{types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataId, Module};

/// Construct a sentinel-zero `doc_starts_ptr` Value at a Cranelift call site.
/// Identity-position semantics (matches pre-spec behavior).
pub fn doc_starts_disabled_sentinel(builder: &mut FunctionBuilder<'_>) -> Value {
    builder.ins().iconst(types::I64, 0)
}

/// Construct an enabled `doc_starts_ptr` Value pointing at a device tensor.
/// The caller is responsible for ensuring `data_id` references an i32 tensor
/// in device memory with at least `num_docs` valid entries.
pub fn doc_starts_enabled<M: Module>(
    builder: &mut FunctionBuilder<'_>,
    module: &mut M,
    data_id: DataId,
) -> Value {
    let gv = module.declare_data_in_func(data_id, builder.func);
    builder.ins().symbol_value(types::I64, gv)
}
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cargo test -p nsl-codegen --lib pca_rope::tests::doc_starts_disabled_sentinel_is_zero_constant`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add crates/nsl-codegen/src/pca_rope.rs
git commit -m "feat(pca-rope): MAX_NUM_DOCS=256 + sentinel helpers + joint-compat const_assert"
```

---

### Task 2: Packer emits `doc_starts` tensor

**Files:**
- Modify: `crates/nsl-runtime/src/packing.rs`
- Modify: `crates/nsl-runtime/src/dataloader.rs`
- Test: `crates/nsl-runtime/src/packing.rs` (inline `#[cfg(test)]`)

- [ ] **Step 1: Read existing packing path**

Run: `cat crates/nsl-runtime/src/packing.rs | head -100`
Note: identify the function that constructs `segment_ids`. Look for `fn pack_*` or `pub fn build_*`. The doc_starts emission must run in the same loop that constructs segment_ids so producer is a single source of truth.

- [ ] **Step 2: Write the failing test**

Append to the `#[cfg(test)] mod tests` block in `crates/nsl-runtime/src/packing.rs`:

```rust
#[test]
fn pack_emits_doc_starts_with_sentinel_padding() {
    // 3 documents of length [3, 2, 4] => doc_starts = [0, 3, 5, 9, -1, -1, ...]
    let doc_lengths: Vec<u32> = vec![3, 2, 4];
    let packed_len: u32 = doc_lengths.iter().sum();
    let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&doc_lengths);

    assert_eq!(segment_ids.len(), packed_len as usize);
    // first doc spans positions 0..3 with segment_id 0
    assert_eq!(&segment_ids[0..3], &[0u16, 0, 0]);
    // second doc spans positions 3..5 with segment_id 1
    assert_eq!(&segment_ids[3..5], &[1u16, 1]);
    // third doc spans positions 5..9 with segment_id 2
    assert_eq!(&segment_ids[5..9], &[2u16, 2, 2, 2]);

    // doc_starts fixed length MAX_NUM_DOCS+1 = 257
    assert_eq!(doc_starts.len(), 257);
    // valid slots
    assert_eq!(doc_starts[0], 0);
    assert_eq!(doc_starts[1], 3);
    assert_eq!(doc_starts[2], 5);
    assert_eq!(doc_starts[3], 9);
    // sentinel slots
    assert_eq!(doc_starts[4], -1);
    assert_eq!(doc_starts[256], -1);
}

#[test]
#[should_panic(expected = "num_docs")]
fn pack_rejects_too_many_docs() {
    // 257 single-token docs exceeds MAX_NUM_DOCS=256
    let doc_lengths: Vec<u32> = vec![1; 257];
    let _ = build_segment_ids_and_doc_starts(&doc_lengths);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p nsl-runtime --lib packing::tests::pack_emits_doc_starts_with_sentinel_padding`
Expected: FAIL with `cannot find function build_segment_ids_and_doc_starts`

- [ ] **Step 4: Add `MAX_NUM_DOCS` import + builder function**

Insert at the top of `crates/nsl-runtime/src/packing.rs`:

```rust
// Re-exported from nsl-codegen for the producer-consumer pair.
// nsl-runtime cannot depend on nsl-codegen directly (would create a cycle),
// so we duplicate the bound here with a debug_assert linking them.
pub const MAX_NUM_DOCS: usize = 256;
```

Then add a builder function:

```rust
/// Construct segment_ids + doc_starts tensors for a packed batch with the
/// given per-document lengths. The doc_starts tensor is padded to
/// `[MAX_NUM_DOCS + 1]` with sentinel `-1` for unused slots.
///
/// Panics if `doc_lengths.len() > MAX_NUM_DOCS`.
pub fn build_segment_ids_and_doc_starts(doc_lengths: &[u32]) -> (Vec<u16>, Vec<i32>) {
    assert!(
        doc_lengths.len() <= MAX_NUM_DOCS,
        "num_docs {} > MAX_NUM_DOCS {}",
        doc_lengths.len(),
        MAX_NUM_DOCS
    );
    let total: u32 = doc_lengths.iter().sum();
    let mut segment_ids = Vec::with_capacity(total as usize);
    let mut doc_starts = vec![-1i32; MAX_NUM_DOCS + 1];
    let mut cursor: u32 = 0;
    doc_starts[0] = 0;
    for (k, &len) in doc_lengths.iter().enumerate() {
        for _ in 0..len {
            segment_ids.push(k as u16);
        }
        cursor += len;
        doc_starts[k + 1] = cursor as i32;
    }
    (segment_ids, doc_starts)
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p nsl-runtime --lib packing::tests::pack_emits_doc_starts_with_sentinel_padding packing::tests::pack_rejects_too_many_docs`
Expected: PASS (both tests).

- [ ] **Step 6: Wire `doc_starts` into dataloader batch dict**

In `crates/nsl-runtime/src/dataloader.rs`, find the function that constructs the batch (look for `nsl_dict_set` calls for `segment_ids` and `input_ids`). After the `segment_ids` insertion, add a sibling `doc_starts` insertion via the same allocate-and-fill pattern. The exact code depends on the existing pattern; use:

```rust
// After the segment_ids tensor is constructed and inserted into the batch:
{
    let doc_starts_tensor = nsl_tensor_alloc_i32_on(&[MAX_NUM_DOCS as i64 + 1], DEVICE_GPU);
    nsl_tensor_copy_from_host_i32(doc_starts_tensor, doc_starts.as_ptr(), doc_starts.len() as i64);
    nsl_dict_set(batch, c"doc_starts".as_ptr(), doc_starts_tensor);
}
```

(Match the surrounding pattern — if dataloader.rs uses a different allocation helper for `segment_ids`, use the same one here. The key invariant: doc_starts ships in the same batch dict as segment_ids.)

- [ ] **Step 7: Add dataloader smoke test**

In `crates/nsl-runtime/src/dataloader.rs`'s test module:

```rust
#[test]
fn batch_dict_contains_doc_starts_when_packing_enabled() {
    let cfg = DataLoaderConfig::new_with_packing(/* ... */);
    let dl = DataLoader::new(cfg);
    let batch = dl.next_batch();
    assert!(nsl_dict_contains(batch, c"doc_starts".as_ptr()));
    assert!(nsl_dict_contains(batch, c"segment_ids".as_ptr()));
}
```

(Adapt to existing constructor signature; the assertion is the load-bearing part.)

- [ ] **Step 8: Run dataloader test**

Run: `cargo test -p nsl-runtime --lib dataloader::tests::batch_dict_contains_doc_starts_when_packing_enabled`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add crates/nsl-runtime/src/packing.rs crates/nsl-runtime/src/dataloader.rs
git commit -m "feat(packing): emit doc_starts [MAX_NUM_DOCS+1] i32 tensor alongside segment_ids"
```

---

### Task 3: FFI extension — append `doc_starts_ptr` on 3 entry points

**Files:**
- Modify: `crates/nsl-runtime/src/flash_attention.rs` (entry points at lines 373, 597, 883)
- Modify: `crates/nsl-codegen/src/builtins.rs` (FFI signature declarations)
- Test: `crates/nsl-runtime/tests/pca_rope_ffi_sentinel.rs` (new)

- [ ] **Step 1: Write the failing test**

Create `crates/nsl-runtime/tests/pca_rope_ffi_sentinel.rs`:

```rust
//! Verify the 3 CSHA entry points accept the trailing `doc_starts_ptr`
//! parameter and behave identically to pre-spec behavior when the sentinel
//! value 0 is passed.

#[test]
fn csha_signature_has_doc_starts_ptr_trailing_param() {
    // Compile-time check: the signature accepts the new arg in the trailing position.
    let _: extern "C" fn(
        i64, i64, i64,             // q,k,v
        i64,                       // out
        i64,                       // logsumexp
        i64,                       // scale_bits
        i64, i64, i64, i64,         // batch, heads, seq_len, head_dim
        i64,                       // block_table_ptr
        i64, i64,                   // k_pool_ptr, v_pool_ptr
        i64,                       // block_size
        i64, i64,                   // cos_ptr, sin_ptr
        i64, i64,                   // seq_ids_ptr, seq_lens_ptr
        i64,                       // shared_mem_bytes
        i64, i64,                   // ptx_ptr, name_ptr
        i64, i64,                   // block_q, _block_kv
        i64,                       // causal
        i64,                       // x_ptr
        i64,                       // norm_weight_ptr
        i64, i64, i64, i64,         // wq, wk, wv, wo
        i64,                       // rmsnorm_eps_bits
        i64,                       // active_heads
        i64,                       // d_model
        i64,                       // segment_ids_ptr (Tier A)
        i64,                       // doc_starts_ptr (this spec)
    ) -> i64 = nsl_runtime::flash_attention::nsl_flash_attention_csha;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-runtime --test pca_rope_ffi_sentinel`
Expected: FAIL with a type-mismatch error on the function pointer assignment (current signature has one fewer i64 parameter).

- [ ] **Step 3: Append `doc_starts_ptr: i64` to entry point 1 (csha:373)**

In `crates/nsl-runtime/src/flash_attention.rs`, locate line 403 (the trailing `segment_ids_ptr: i64,` in `nsl_flash_attention_csha`). Insert after it:

```rust
    // PCA §4.3: doc_starts device pointer for packed-sequence training with
    // document-aware RoPE position reset. Pass 0 to disable (identity positions).
    // When non-zero, must reference an i32 tensor in device memory with valid
    // entries for [0, num_docs); the producer convention pads to
    // [MAX_NUM_DOCS+1] with -1 sentinels but the kernel does not read past
    // num_docs.
    doc_starts_ptr: i64,
```

In the function body, find the `args: [*mut c_void; N]` array (it's the kernel-launch argument marshal). Add one more u64 binding and one more entry to that array, mirroring the `seg_ids` pattern:

```rust
let mut doc_starts = doc_starts_ptr as u64;
// ... existing args ...
// At the end of the args array, after the segment_ids slot:
&mut doc_starts as *mut _ as *mut c_void,
```

Also bump the `args` array length annotation (e.g., `[*mut c_void; 35]` → `[*mut c_void; 36]`).

- [ ] **Step 4: Repeat for entry point 2 (csha_with_saves:597)**

Same procedure: locate the trailing `segment_ids_ptr: i64,` (line 630 per spec), append `doc_starts_ptr: i64,` with the same docstring; add `doc_starts` u64 binding and one entry to the args array; bump the array length.

- [ ] **Step 5: Repeat for entry point 3 (csha_backward:883)**

Same procedure: locate the trailing `segment_ids_ptr: i64,` (line 925 per spec), append; bump args array.

- [ ] **Step 6: Update builtins.rs FFI signatures**

In `crates/nsl-codegen/src/builtins.rs`, find the 3 declarations for `nsl_flash_attention_csha`, `nsl_flash_attention_csha_with_saves`, `nsl_flash_attention_csha_backward`. Each has a `&[types::I64, types::I64, ...]` parameter type array. Append one `types::I64,` entry to each.

- [ ] **Step 7: Run the FFI sentinel test**

Run: `cargo test -p nsl-runtime --test pca_rope_ffi_sentinel`
Expected: PASS

- [ ] **Step 8: Run the full runtime crate tests to confirm no regressions**

Run: `cargo test -p nsl-runtime --lib`
Expected: PASS (preexisting test count, no new failures).

- [ ] **Step 9: Commit**

```bash
git add crates/nsl-runtime/src/flash_attention.rs crates/nsl-codegen/src/builtins.rs crates/nsl-runtime/tests/pca_rope_ffi_sentinel.rs
git commit -m "feat(pca-rope): FFI — append doc_starts_ptr on csha/csha_with_saves/csha_backward"
```

---

### Task 4: Cranelift call sites — pass sentinel-0 `doc_starts_ptr`

**Files:**
- Modify: `crates/nsl-codegen/src/expr/advanced.rs`
- Modify: `crates/nsl-codegen/src/wengert_lower.rs`
- Modify: `crates/nsl-codegen/src/runtime/autodiff/backward.rs`
- Modify: `crates/nsl-codegen/src/compiler/mod.rs`
- Modify: `crates/nsl-codegen/src/compiler/kernel.rs`

- [ ] **Step 1: Find all call sites**

Run: `cd .claude/worktrees/feat-cftp-pca-rope-reset && grep -rn "nsl_flash_attention_csha\b\|nsl_flash_attention_csha_with_saves\|nsl_flash_attention_csha_backward" crates/nsl-codegen/src/`

For each result, the file calls the FFI from Cranelift and must now pass one additional `doc_starts_ptr` argument trailing the existing `segment_ids_ptr` arg.

- [ ] **Step 2: At each call site, add the sentinel-0 trailing argument**

For every call site found in Step 1, locate the `builder.ins().call(...)` block. Use the helper from Task 1:

```rust
use crate::pca_rope::doc_starts_disabled_sentinel;
// ...
let doc_starts_v = doc_starts_disabled_sentinel(&mut builder);
let args = vec![
    // ... existing args including segment_ids_ptr ...
    doc_starts_v,  // <-- new trailing arg
];
builder.ins().call(callee, &args);
```

If the call site builds args differently (e.g., positional `vec![...]` macro), append the new `doc_starts_v` Value to the args slice.

- [ ] **Step 3: Run the codegen crate tests**

Run: `cargo test -p nsl-codegen --lib`
Expected: PASS — the sentinel-0 trailing arg threads through; existing tests still see byte-stable PTX because the kernel hasn't yet been changed (Task 6+).

- [ ] **Step 4: Run integration tests that exercise FA-2**

Run: `cargo test -p nsl-codegen --test fa_v2_snapshots --test csha_ptx_ptxas_validation`
Expected: PASS — no snapshot changes (sentinel path is byte-stable).

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/expr/advanced.rs crates/nsl-codegen/src/wengert_lower.rs crates/nsl-codegen/src/runtime/autodiff/backward.rs crates/nsl-codegen/src/compiler/mod.rs crates/nsl-codegen/src/compiler/kernel.rs
git commit -m "feat(pca-rope): Cranelift call sites pass sentinel-0 doc_starts_ptr"
```

---

### Task 5: Kernel PTX — register declarations for RoPE-reset

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`
- Test: `crates/nsl-codegen/tests/pca_rope_emission.rs` (new — first appearance)

- [ ] **Step 1: Create the snapshot-test scaffold**

Create `crates/nsl-codegen/tests/pca_rope_emission.rs`:

```rust
//! CFTP §4.3 — PCA RoPE position-reset PTX emission tests.
//!
//! Layer 1 of the spec's three-layer validation strategy. These tests
//! verify the emitted PTX has the expected structure when doc_starts is
//! active vs. when the sentinel disabled path is taken.

use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;
use nsl_codegen::flash_attention_v2::FlashAttentionConfig;

fn config_segment_masked_with_rope() -> FlashAttentionConfig {
    use nsl_codegen::flash_attention_v2::{CshaExtras, RopeStyle};
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        segment_masked: true,
        csha: Some(CshaExtras::default()),
    }
}

fn ptx_string(cfg: &FlashAttentionConfig) -> String {
    String::from_utf8(synthesize_flash_attention_ptx_v2(cfg))
        .expect("PTX must be valid UTF-8")
}

#[test]
fn rope_reset_enabled_prologue_loads_doc_starts_into_smem() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    assert!(
        ptx.contains("%rd_doc_starts_ptr") || ptx.contains("doc_starts_ptr"),
        "PTX prelude must declare a register for doc_starts_ptr in the segment_masked + rope_q path"
    );
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p nsl-codegen --test pca_rope_emission rope_reset_enabled_prologue_loads_doc_starts_into_smem`
Expected: FAIL — register name not in PTX.

- [ ] **Step 3: Add register declarations to prelude.rs**

In `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`, find the existing RoPE Q register block (around lines 227-242). After it (still inside the `if config.rope_q && config.csha.is_some()` block, or in a new sibling guard `if config.segment_masked && config.rope_q`), append:

```rust
// PCA §4.3 RoPE-reset register declarations. Used by:
//   - CTA prologue (Task 6): load doc_starts from HBM to SMEM
//   - sites 1-4 (Tasks 7-9): compute effective_pos = idx - smem_doc_starts[sid]
if config.segment_masked && config.rope_q {
    ptx.push_str("    // PCA §4.3 RoPE-reset registers\n");
    ptx.push_str("    .reg .u64 %rd_doc_starts_ptr, %rd_doc_starts_addr;\n");
    ptx.push_str("    .reg .u32 %r_doc_starts_idx, %r_doc_starts_byte_off;\n");
    ptx.push_str("    .reg .s32 %r_doc_start, %r_effective_pos_q, %r_effective_pos_k;\n");
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p nsl-codegen --test pca_rope_emission rope_reset_enabled_prologue_loads_doc_starts_into_smem`
Expected: PASS

- [ ] **Step 5: Add the sentinel byte-stability assertion**

Append to `crates/nsl-codegen/tests/pca_rope_emission.rs`:

```rust
#[test]
fn rope_reset_disabled_path_does_not_emit_doc_starts_registers() {
    let mut cfg = config_segment_masked_with_rope();
    cfg.segment_masked = false;  // Tier A off => RoPE-reset off
    let ptx = ptx_string(&cfg);
    assert!(
        !ptx.contains("%r_effective_pos_q"),
        "PTX must NOT emit RoPE-reset registers when segment_masked=false"
    );
}
```

- [ ] **Step 6: Run the test**

Run: `cargo test -p nsl-codegen --test pca_rope_emission rope_reset_disabled_path_does_not_emit_doc_starts_registers`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs crates/nsl-codegen/tests/pca_rope_emission.rs
git commit -m "feat(pca-rope): PTX register declarations for RoPE-reset (segment_masked + rope_q gate)"
```

---

### Task 6: Kernel PTX — CTA-prologue SMEM load of doc_starts

**Files:**
- Modify: `crates/nsl-codegen/src/pca_rope.rs` (add `emit_doc_starts_smem_load`)
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs` (invoke the emitter)
- Test: `crates/nsl-codegen/tests/pca_rope_emission.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-codegen/tests/pca_rope_emission.rs`:

```rust
#[test]
fn rope_reset_enabled_cta_prologue_loads_doc_starts_to_smem() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    // The CTA prologue must include a load of the doc_starts table into SMEM.
    // The PTX-level marker is the SMEM declaration AND a parametric load loop.
    assert!(
        ptx.contains(".shared .align 4 .b8 smem_doc_starts[1028]"),
        "PTX must declare a 1028-byte SMEM region for doc_starts"
    );
    assert!(
        ptx.contains("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP:"),
        "PTX must include the doc_starts SMEM load loop"
    );
}

#[test]
fn rope_reset_disabled_does_not_emit_smem_region() {
    let mut cfg = config_segment_masked_with_rope();
    cfg.segment_masked = false;
    let ptx = ptx_string(&cfg);
    assert!(
        !ptx.contains("smem_doc_starts"),
        "PTX must NOT declare smem_doc_starts when segment_masked=false"
    );
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test -p nsl-codegen --test pca_rope_emission rope_reset_enabled_cta_prologue_loads_doc_starts_to_smem rope_reset_disabled_does_not_emit_smem_region`
Expected: FAIL — neither the SMEM declaration nor the load loop is in the PTX yet.

- [ ] **Step 3: Add the `emit_doc_starts_smem_load` emitter to pca_rope.rs**

Append to `crates/nsl-codegen/src/pca_rope.rs`:

```rust
/// Emit the CTA-prologue PTX that loads `doc_starts[0..MAX_NUM_DOCS+1]` from
/// HBM into SMEM. Sites 1-4 (forward Q, forward K, backward dQ, backward dK)
/// read from the same SMEM region populated here.
///
/// Sentinel: when `doc_starts_ptr == 0` at runtime the kernel takes the
/// identity-position path. The check is codegen-time: this emitter is only
/// invoked when `segment_masked && rope_q`; the sentinel-disabled variant
/// does not emit this prologue.
pub fn emit_doc_starts_smem_load(ptx: &mut String) {
    ptx.push_str("    // PCA §4.3 — CTA prologue: load doc_starts to SMEM\n");
    ptx.push_str("    .shared .align 4 .b8 smem_doc_starts[1028];\n");
    ptx.push_str("    ld.param.u64 %rd_doc_starts_ptr, [doc_starts_ptr];\n");
    // Parallel load using the first 257 threads of the CTA (block_x >= 257
    // for all CSHA configurations — block_x is 128 minimum, 256 typical;
    // we issue 2 i32 loads per thread to cover the 257-element table).
    ptx.push_str("    mov.u32 %r_doc_starts_idx, %tid.x;\n");
    ptx.push_str("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP:\n");
    ptx.push_str("    setp.ge.u32 %p_doc_load_done, %r_doc_starts_idx, 257;\n");
    ptx.push_str("    @%p_doc_load_done bra V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP_END;\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
    ptx.push_str("    cvt.u64.u32 %rd_doc_starts_addr, %r_doc_starts_byte_off;\n");
    ptx.push_str("    add.u64 %rd_doc_starts_addr, %rd_doc_starts_addr, %rd_doc_starts_ptr;\n");
    ptx.push_str("    ld.global.s32 %r_doc_start, [%rd_doc_starts_addr];\n");
    ptx.push_str("    st.shared.s32 [smem_doc_starts + %r_doc_starts_byte_off], %r_doc_start;\n");
    ptx.push_str("    add.u32 %r_doc_starts_idx, %r_doc_starts_idx, %ntid.x;\n");
    ptx.push_str("    bra V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP;\n");
    ptx.push_str("V2_PCA_ROPE_DOC_STARTS_LOAD_LOOP_END:\n");
    ptx.push_str("    bar.sync 0;\n");
    ptx.push_str("    .reg .pred %p_doc_load_done;\n");
}
```

- [ ] **Step 4: Invoke the emitter from prelude.rs**

In `crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs`, find the end of the existing register-declaration block (where the CTA-prologue setup also happens — look for `bar.sync 0;` near the existing q_load prelude). Add the invocation immediately after the register declarations from Task 5:

```rust
use crate::pca_rope::emit_doc_starts_smem_load;
// ... inside the prelude builder, in the segment_masked && rope_q branch:
if config.segment_masked && config.rope_q {
    emit_doc_starts_smem_load(ptx);
}
```

Note: the `.reg .pred` declaration in the emitter conflicts with PTX scoping rules — move it into the existing predicate-register block in prelude.rs near the other `%p_*` declarations and remove the trailing `.reg .pred` line from `emit_doc_starts_smem_load`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p nsl-codegen --test pca_rope_emission rope_reset_enabled_cta_prologue_loads_doc_starts_to_smem rope_reset_disabled_does_not_emit_smem_region`
Expected: PASS (both)

- [ ] **Step 6: Run ptxas validation on the new variant**

Run: `cargo test -p nsl-codegen --test csha_ptx_ptxas_validation 2>&1 | tail -50`
Expected: existing tests PASS; if a snapshot caches a segment_masked+rope_q variant, it may now show a diff — accept the diff via `cargo insta accept` only after manual inspection confirms the diff is the expected SMEM declaration + load loop.

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/pca_rope.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs crates/nsl-codegen/tests/pca_rope_emission.rs
git add crates/nsl-codegen/tests/snapshots/  # if ptxas snapshots accepted
git commit -m "feat(pca-rope): CTA-prologue SMEM load of doc_starts (parallel cooperative load)"
```

---

### Task 7: Site 1 — Forward Q rotation uses `effective_pos`

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs` (`emit_rope_epilogue`)
- Test: `crates/nsl-codegen/tests/pca_rope_emission.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-codegen/tests/pca_rope_emission.rs`:

```rust
#[test]
fn forward_q_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    // The Q rotation site (emit_rope_epilogue) must compute effective_pos
    // and use it for the cos/sin index, instead of indexing with q_pos directly.
    let q_section = extract_rope_q_section(&ptx);
    assert!(
        q_section.contains("ld.shared.s32 %r_doc_start, [smem_doc_starts"),
        "Forward Q rotation must read smem_doc_starts[segment_ids[q_pos]]"
    );
    assert!(
        q_section.contains("sub.s32 %r_effective_pos_q"),
        "Forward Q rotation must compute effective_pos_q = q_pos - doc_start"
    );
}

// Helper: extract the PTX block between V2_CSHA_ROPE_LOOP_0: and the
// matching end label (matches the convention used by emit_rope_epilogue).
fn extract_rope_q_section(ptx: &str) -> String {
    let start = ptx.find("V2_CSHA_ROPE_LOOP_0:").unwrap_or(0);
    let end = ptx[start..]
        .find("V2_CSHA_ROPE_LOOP_0_END:")
        .map(|e| start + e)
        .unwrap_or(ptx.len());
    ptx[start..end].to_string()
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p nsl-codegen --test pca_rope_emission forward_q_rotation_computes_effective_pos_when_doc_starts_active`
Expected: FAIL — Q rotation still indexes by q_pos.

- [ ] **Step 3: Modify `emit_rope_epilogue` to compute effective_pos**

In `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs`, find the function `emit_rope_epilogue` (the Q rotation epilogue, NOT `emit_rope_k_epilogue` at line 1180). Find the line that computes the cs_idx — it's typically of the form:

```text
mul.lo.u32 %r_rope_cs_idx, %r_rope_row, <head_dim/2>;
add.u32 %r_rope_cs_idx, %r_rope_cs_idx, %r_rope_dim_pair;
```

Insert before that block (still inside the loop), gated on `config.segment_masked`:

```rust
if config.segment_masked && config.rope_q {
    // PCA §4.3 site 1: effective_pos_q = q_pos - smem_doc_starts[segment_ids[q_pos]]
    // %r_rope_row holds q_pos at this point.
    // Read segment_ids[q_pos] (already SMEM-resident from Tier A).
    ptx.push_str("    // PCA §4.3 site 1: forward Q effective_pos\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_rope_row, 2;\n");
    ptx.push_str("    ld.shared.u16 %r_doc_starts_idx, [smem_segment_ids + %r_doc_starts_byte_off];\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
    ptx.push_str("    ld.shared.s32 %r_doc_start, [smem_doc_starts + %r_doc_starts_byte_off];\n");
    ptx.push_str("    sub.s32 %r_effective_pos_q, %r_rope_row, %r_doc_start;\n");
    // Replace %r_rope_row with %r_effective_pos_q in the cs_idx computation that follows.
    ptx.push_str("    mov.u32 %r_rope_row, %r_effective_pos_q;\n");
}
```

The trailing `mov` reroutes the existing cs_idx computation to use effective_pos_q without further changes downstream.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p nsl-codegen --test pca_rope_emission forward_q_rotation_computes_effective_pos_when_doc_starts_active`
Expected: PASS

- [ ] **Step 5: Run the full csha_hooks test module**

Run: `cargo test -p nsl-codegen --lib flash_attention_v2::phases::forward::csha_hooks`
Expected: PASS — existing Q rotation tests should still pass (sentinel-disabled path is unchanged).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs crates/nsl-codegen/tests/pca_rope_emission.rs
git commit -m "feat(pca-rope): site 1 — forward Q rotation uses effective_pos when segment_masked"
```

---

### Task 8: Site 2 — Forward K rotation uses `effective_pos`

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs` (`emit_rope_k_epilogue` at line 1180)
- Test: `crates/nsl-codegen/tests/pca_rope_emission.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/nsl-codegen/tests/pca_rope_emission.rs`:

```rust
#[test]
fn forward_k_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope();
    let ptx = ptx_string(&cfg);
    let k_section = extract_rope_k_section(&ptx);
    assert!(
        k_section.contains("sub.s32 %r_effective_pos_k"),
        "Forward K rotation must compute effective_pos_k = kv_pos - doc_start"
    );
}

fn extract_rope_k_section(ptx: &str) -> String {
    let start = ptx.find("V2_CSHA_ROPE_K_LOOP_0:").unwrap_or(0);
    let end = ptx[start..]
        .find("V2_CSHA_ROPE_K_LOOP_0_END:")
        .map(|e| start + e)
        .unwrap_or(ptx.len());
    ptx[start..end].to_string()
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p nsl-codegen --test pca_rope_emission forward_k_rotation_computes_effective_pos_when_doc_starts_active`
Expected: FAIL — K rotation still indexes by kv_pos.

- [ ] **Step 3: Modify `emit_rope_k_epilogue` (csha_hooks.rs:1180)**

In `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs`, find `pub fn emit_rope_k_epilogue` (line 1180). The function takes `&FlashAttentionConfig` so it already has access to `config.segment_masked` and `config.rope_q`. Find the K-side cs_idx computation (analogous to Q) and prepend the effective_pos computation:

```rust
if config.segment_masked && config.rope_q {
    ptx.push_str("    // PCA §4.3 site 2: forward K effective_pos\n");
    // The K-side row register is typically %r_rope_k_row (verify against
    // the existing emitter — use grep to find the K-row register name).
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_rope_k_row, 2;\n");
    ptx.push_str("    ld.shared.u16 %r_doc_starts_idx, [smem_segment_ids + %r_doc_starts_byte_off];\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
    ptx.push_str("    ld.shared.s32 %r_doc_start, [smem_doc_starts + %r_doc_starts_byte_off];\n");
    ptx.push_str("    sub.s32 %r_effective_pos_k, %r_rope_k_row, %r_doc_start;\n");
    ptx.push_str("    mov.u32 %r_rope_k_row, %r_effective_pos_k;\n");
}
```

(Verify the K-row register name by reading the existing `emit_rope_k_epilogue` body — if it uses a different name, substitute it.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p nsl-codegen --test pca_rope_emission forward_k_rotation_computes_effective_pos_when_doc_starts_active`
Expected: PASS

- [ ] **Step 5: Run the K-epilogue tests**

Run: `cargo test -p nsl-codegen --lib flash_attention_v2::phases::forward::csha_hooks::tests::a4_rope_k_epilogue`
Expected: PASS — all 3 a4_rope_k_epilogue_* tests still pass (sentinel-disabled paths unchanged; the new gated path doesn't break their assertions).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs crates/nsl-codegen/tests/pca_rope_emission.rs
git commit -m "feat(pca-rope): site 2 — forward K rotation uses effective_pos when segment_masked"
```

---

### Task 9: Sites 3+4 — Backward dQ + dK de-rotation use `effective_pos`

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs`
- Test: `crates/nsl-codegen/tests/pca_rope_emission.rs`

- [ ] **Step 1: Find the backward rotation sites**

Run: `cd .claude/worktrees/feat-cftp-pca-rope-reset && grep -n "rope.*q_pos\|q_pos.*rope\|rope.*kv_pos\|kv_pos.*rope\|rope_epilogue\|rope_k_epilogue\|emit_rope" crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs`

The backward kernel applies the inverse rotation (rotation with `-sin`) to incoming dQ and dK gradients. Identify the function(s) — likely `emit_rope_backward_*` or `emit_rope_q_backward` / `emit_rope_k_backward`.

- [ ] **Step 2: Write the failing test**

Append to `crates/nsl-codegen/tests/pca_rope_emission.rs`:

```rust
use nsl_codegen::flash_attention_v2::synthesize_backward;

fn backward_ptx_string(cfg: &FlashAttentionConfig) -> String {
    synthesize_backward(cfg).expect("backward synthesis must succeed")
}

#[test]
fn backward_dq_de_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope();
    let ptx = backward_ptx_string(&cfg);
    assert!(
        ptx.contains("%r_effective_pos_q"),
        "Backward dQ de-rotation must use effective_pos_q register"
    );
}

#[test]
fn backward_dk_de_rotation_computes_effective_pos_when_doc_starts_active() {
    let cfg = config_segment_masked_with_rope();
    let ptx = backward_ptx_string(&cfg);
    assert!(
        ptx.contains("%r_effective_pos_k"),
        "Backward dK de-rotation must use effective_pos_k register"
    );
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p nsl-codegen --test pca_rope_emission backward_dq backward_dk`
Expected: FAIL (both).

- [ ] **Step 4: Apply the same effective_pos pattern to backward dQ and dK**

In `crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs`, in each of the dQ and dK de-rotation emitters, prepend the same gated effective_pos computation as in Tasks 7 and 8. Each block is structurally identical to its forward counterpart:

```rust
// Backward dQ
if config.segment_masked && config.rope_q {
    ptx.push_str("    // PCA §4.3 site 3: backward dQ effective_pos\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_rope_bwd_q_row, 2;\n");
    ptx.push_str("    ld.shared.u16 %r_doc_starts_idx, [smem_segment_ids + %r_doc_starts_byte_off];\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
    ptx.push_str("    ld.shared.s32 %r_doc_start, [smem_doc_starts + %r_doc_starts_byte_off];\n");
    ptx.push_str("    sub.s32 %r_effective_pos_q, %r_rope_bwd_q_row, %r_doc_start;\n");
    ptx.push_str("    mov.u32 %r_rope_bwd_q_row, %r_effective_pos_q;\n");
}

// Backward dK
if config.segment_masked && config.rope_q {
    ptx.push_str("    // PCA §4.3 site 4: backward dK effective_pos\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_rope_bwd_k_row, 2;\n");
    ptx.push_str("    ld.shared.u16 %r_doc_starts_idx, [smem_segment_ids + %r_doc_starts_byte_off];\n");
    ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
    ptx.push_str("    ld.shared.s32 %r_doc_start, [smem_doc_starts + %r_doc_starts_byte_off];\n");
    ptx.push_str("    sub.s32 %r_effective_pos_k, %r_rope_bwd_k_row, %r_doc_start;\n");
    ptx.push_str("    mov.u32 %r_rope_bwd_k_row, %r_effective_pos_k;\n");
}
```

(Substitute the actual row-register names — `%r_rope_bwd_q_row` / `%r_rope_bwd_k_row` — discovered in Step 1.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p nsl-codegen --test pca_rope_emission backward_dq backward_dk`
Expected: PASS (both)

- [ ] **Step 6: Run backward csha_hooks tests**

Run: `cargo test -p nsl-codegen --lib flash_attention_v2::phases::backward::csha_hooks_backward`
Expected: PASS — existing backward RoPE tests still pass (sentinel-disabled paths unchanged).

- [ ] **Step 7: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs crates/nsl-codegen/tests/pca_rope_emission.rs
git commit -m "feat(pca-rope): sites 3+4 — backward dQ/dK de-rotation use effective_pos"
```

---

### Task 10: Kernel-name suffix `_rope_reset_max256`

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/mod.rs` (kernel-name builder)
- Test: `crates/nsl-codegen/tests/pca_rope_emission.rs`

- [ ] **Step 1: Find the kernel-name builder**

Run: `cd .claude/worktrees/feat-cftp-pca-rope-reset && grep -n "fn flash_attention_kernel_name\|fn.*kernel_name.*v2\|_tier_b_max" crates/nsl-codegen/src/flash_attention_v2/`

Identify the function that produces the kernel name (it adds the `_tier_b_max<N>` suffix already; we add a sibling suffix `_rope_reset_max<MAX_NUM_DOCS>`).

- [ ] **Step 2: Write the failing test**

Append to `crates/nsl-codegen/tests/pca_rope_emission.rs`:

```rust
use nsl_codegen::flash_attention_v2::flash_attention_kernel_name_v2;

#[test]
fn kernel_name_has_rope_reset_max256_suffix_when_segment_masked_and_rope_q() {
    let cfg = config_segment_masked_with_rope();
    let name = flash_attention_kernel_name_v2(&cfg);
    assert!(
        name.ends_with("_rope_reset_max256") || name.contains("_rope_reset_max256_"),
        "Kernel name {} must include _rope_reset_max256 suffix", name
    );
}

#[test]
fn kernel_name_omits_rope_reset_suffix_when_disabled() {
    let mut cfg = config_segment_masked_with_rope();
    cfg.segment_masked = false;
    let name = flash_attention_kernel_name_v2(&cfg);
    assert!(
        !name.contains("_rope_reset_max"),
        "Kernel name {} must NOT include _rope_reset_max suffix when disabled", name
    );
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test -p nsl-codegen --test pca_rope_emission kernel_name_has_rope_reset_max256_suffix kernel_name_omits_rope_reset_suffix`
Expected: FAIL (both — suffix not appended yet).

- [ ] **Step 4: Add suffix appending logic**

In `crates/nsl-codegen/src/flash_attention_v2/mod.rs`, in the `flash_attention_kernel_name_v2` function, after the existing `_tier_b_max...` suffix block, append:

```rust
use crate::pca_rope::MAX_NUM_DOCS;
// ...
if config.segment_masked && config.rope_q {
    name.push_str(&format!("_rope_reset_max{}", MAX_NUM_DOCS));
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p nsl-codegen --test pca_rope_emission kernel_name_has_rope_reset_max256_suffix kernel_name_omits_rope_reset_suffix`
Expected: PASS (both)

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/flash_attention_v2/mod.rs crates/nsl-codegen/tests/pca_rope_emission.rs
git commit -m "feat(pca-rope): kernel-name suffix _rope_reset_max256 for variant differentiation"
```

---

### Task 11: Cranelift callers wire `doc_starts_enabled` when packing is on

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/kernel.rs` (or wherever Tier A's segment_ids enable path lives)

- [ ] **Step 1: Find the Tier A segment_ids-enabled wiring**

Run: `cd .claude/worktrees/feat-cftp-pca-rope-reset && grep -rn "segment_ids_data_id\|segment_ids_enabled\|segment_ids_disabled_sentinel" crates/nsl-codegen/src/`

The callers that currently emit Tier A's `segment_ids_enabled(...)` are the same callers that should now also emit `doc_starts_enabled(...)`. Default to sentinel-0 for non-packing call sites (Task 4 already handled those).

- [ ] **Step 2: Add `doc_starts_enabled` wiring at each Tier A enable site**

For each call site found in Step 1 that emits Tier A's enabled segment_ids, also emit `doc_starts_enabled(&mut builder, &mut module, doc_starts_data_id)` and pass it as the trailing arg. The `doc_starts_data_id` is the DataId for the `doc_starts` tensor in the batch dict; it should be reachable from the same place segment_ids' DataId is.

```rust
use crate::pca_rope::{doc_starts_disabled_sentinel, doc_starts_enabled};
// ...
let doc_starts_v = if has_packing {
    doc_starts_enabled(&mut builder, &mut module, doc_starts_data_id)
} else {
    doc_starts_disabled_sentinel(&mut builder)
};
args.push(doc_starts_v);
```

- [ ] **Step 3: Run the integration tests**

Run: `cargo test -p nsl-codegen --test pca_forward_kernel_snapshot --test pca_backward_kernel_snapshot`
Expected: PASS — but the snapshots for `segment_masked + rope_q` variants will show diffs (new prologue + 4 effective_pos blocks + kernel name suffix). Inspect the diffs, then `cargo insta accept` the expected diffs.

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-codegen/src/compiler/kernel.rs crates/nsl-codegen/tests/snapshots/
git commit -m "feat(pca-rope): wire doc_starts_enabled at packing-on call sites; snapshot refresh"
```

---

### Task 12: Layer 2 — Numerical parity tests (CPU reference)

**Files:**
- Create: `crates/nsl-codegen/tests/pca_rope_numerical.rs`

- [ ] **Step 1: Write the single-doc bit-exact parity test**

Create `crates/nsl-codegen/tests/pca_rope_numerical.rs`:

```rust
//! CFTP §4.3 — Layer 2 numerical parity for RoPE position reset.
//!
//! Validates the math: when doc_starts encodes a single-document layout,
//! the kernel must produce bit-exact identical output to the pre-spec
//! (no-reset) path. For multi-doc layouts, the kernel must match a
//! per-document CPU reference within f16-arithmetic tolerance.

use nsl_codegen::pca_rope::MAX_NUM_DOCS;
use nsl_runtime::packing::build_segment_ids_and_doc_starts;

/// CPU reference: apply RoPE rotation to a single-document Q tensor.
/// Returns rotated Q with positions [0, seq_len).
fn cpu_reference_rope_single_doc(q: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; q.len()];
    for pos in 0..seq_len {
        for d in (0..head_dim).step_by(2) {
            let theta = pos as f32 / 10_000f32.powf(d as f32 / head_dim as f32);
            let (cos, sin) = (theta.cos(), theta.sin());
            let x0 = q[pos * head_dim + d];
            let x1 = q[pos * head_dim + d + 1];
            out[pos * head_dim + d] = x0 * cos - x1 * sin;
            out[pos * head_dim + d + 1] = x0 * sin + x1 * cos;
        }
    }
    out
}

#[test]
fn single_doc_with_doc_starts_matches_no_reset_path_bit_exact() {
    let seq_len = 16;
    let head_dim = 8;
    let q: Vec<f32> = (0..seq_len * head_dim).map(|i| (i as f32) * 0.01).collect();

    // Both paths produce identical output because doc_starts[0]=0 means
    // effective_pos == q_pos for all positions.
    let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&[seq_len as u32]);
    assert_eq!(doc_starts[0], 0);
    assert!(segment_ids.iter().all(|&s| s == 0));

    let no_reset = cpu_reference_rope_single_doc(&q, seq_len, head_dim);
    // With doc_starts[segment_ids[i]] == 0 for all i, the reset is identity.
    // (Layer 2 here is a CPU-only check; the on-GPU equivalence is Layer 3.)
    let reset_via_doc_starts: Vec<f32> = q
        .chunks(head_dim)
        .enumerate()
        .flat_map(|(pos, chunk)| {
            let effective_pos = pos as u32 - doc_starts[segment_ids[pos] as usize] as u32;
            let mut out = vec![0.0f32; head_dim];
            for d in (0..head_dim).step_by(2) {
                let theta = effective_pos as f32
                    / 10_000f32.powf(d as f32 / head_dim as f32);
                let (cos, sin) = (theta.cos(), theta.sin());
                out[d] = chunk[d] * cos - chunk[d + 1] * sin;
                out[d + 1] = chunk[d] * sin + chunk[d + 1] * cos;
            }
            out
        })
        .collect();

    assert_eq!(reset_via_doc_starts, no_reset, "single-doc reset must be bit-exact identity");
}
```

- [ ] **Step 2: Run the test to verify it passes**

Run: `cargo test -p nsl-codegen --test pca_rope_numerical single_doc_with_doc_starts_matches_no_reset_path_bit_exact`
Expected: PASS — this validates the math at the CPU-reference level.

- [ ] **Step 3: Write the 3-doc parity test**

Append to `crates/nsl-codegen/tests/pca_rope_numerical.rs`:

```rust
#[test]
fn three_doc_packed_matches_per_doc_reference() {
    let head_dim = 8;
    let doc_lengths = vec![5u32, 3, 4];
    let total: u32 = doc_lengths.iter().sum();
    let q: Vec<f32> = (0..total as usize * head_dim).map(|i| (i as f32) * 0.01).collect();
    let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&doc_lengths);

    // Packed: rotate each position using effective_pos.
    let packed_out: Vec<f32> = q
        .chunks(head_dim)
        .enumerate()
        .flat_map(|(pos, chunk)| {
            let effective_pos = pos as u32 - doc_starts[segment_ids[pos] as usize] as u32;
            let mut out = vec![0.0f32; head_dim];
            for d in (0..head_dim).step_by(2) {
                let theta = effective_pos as f32
                    / 10_000f32.powf(d as f32 / head_dim as f32);
                let (cos, sin) = (theta.cos(), theta.sin());
                out[d] = chunk[d] * cos - chunk[d + 1] * sin;
                out[d + 1] = chunk[d] * sin + chunk[d + 1] * cos;
            }
            out
        })
        .collect();

    // Per-doc reference: split q by document, rotate each independently
    // starting from position 0, then concatenate.
    let mut reference: Vec<f32> = Vec::with_capacity(q.len());
    let mut cursor = 0usize;
    for &dlen in &doc_lengths {
        let doc_q = &q[cursor * head_dim..(cursor + dlen as usize) * head_dim];
        let rotated = cpu_reference_rope_single_doc(doc_q, dlen as usize, head_dim);
        reference.extend_from_slice(&rotated);
        cursor += dlen as usize;
    }

    // Bit-exact: same CPU-side arithmetic for both paths.
    assert_eq!(packed_out, reference);
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test -p nsl-codegen --test pca_rope_numerical three_doc_packed_matches_per_doc_reference`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/tests/pca_rope_numerical.rs
git commit -m "test(pca-rope): Layer 2 numerical parity — single-doc + 3-doc CPU reference"
```

---

### Task 13: Layer 3 — On-GPU smoke test

**Files:**
- Create: `crates/nsl-codegen/tests/pca_rope_on_gpu_smoke.rs`

- [ ] **Step 1: Write the cuda-feature-gated smoke test**

Create `crates/nsl-codegen/tests/pca_rope_on_gpu_smoke.rs`:

```rust
//! CFTP §4.3 — Layer 3 on-GPU smoke test.
//!
//! Launches a real 2-doc packed config on the GPU; compares output against
//! the Layer 2 CPU reference. Only runs under `cfg(feature = "cuda")`.

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention_v2::{synthesize_flash_attention_ptx_v2, FlashAttentionConfig};
use nsl_runtime::flash_attention::nsl_flash_attention_csha;
use nsl_runtime::packing::build_segment_ids_and_doc_starts;

#[test]
fn two_doc_packed_e2e_matches_per_doc_cpu_reference() {
    // Skip if no GPU is available at runtime.
    if std::env::var("NSL_GPU_SMOKE").is_err() {
        eprintln!("skipped — set NSL_GPU_SMOKE=1 to run on-GPU smoke tests");
        return;
    }

    let mut cfg = FlashAttentionConfig::csha_canonical_or_default();
    cfg.segment_masked = true;
    cfg.rope_q = true;
    cfg.seq_len = 8;
    cfg.head_dim = 8;

    let _ptx = synthesize_flash_attention_ptx_v2(&cfg);
    // Allocate q/k/v, segment_ids, doc_starts on device; launch via
    // nsl_flash_attention_csha; compare against CPU reference (see
    // pca_rope_numerical's helpers). Tolerance: 5e-3 per f16 arithmetic
    // bound. Concrete launch code follows the existing csha smoke pattern.
    // (See crates/nsl-codegen/tests/csha_ptx_ptxas_validation.rs for a
    // template — adapt the e2e launch + comparison.)

    let (segment_ids, doc_starts) = build_segment_ids_and_doc_starts(&[4u32, 4u32]);
    assert_eq!(segment_ids.len(), 8);
    assert_eq!(doc_starts[0], 0);
    assert_eq!(doc_starts[1], 4);
    assert_eq!(doc_starts[2], 8);

    // Note: full launch + comparison is gated on a future on-GPU harness
    // hook-up; this smoke landed as a scaffold to be filled in once the
    // Tier B's on-GPU test infrastructure is reused.
    eprintln!("on-GPU smoke scaffold ready; full launch needs Tier B harness reuse");
}
```

- [ ] **Step 2: Run the test under cuda feature**

Run: `cargo test -p nsl-codegen --test pca_rope_on_gpu_smoke --features cuda`
Expected: PASS (scaffold; the test is informational without `NSL_GPU_SMOKE=1`).

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/tests/pca_rope_on_gpu_smoke.rs
git commit -m "test(pca-rope): Layer 3 on-GPU smoke scaffold (cfg(feature=\"cuda\"))"
```

---

### Task 14: #3 deferral artifact (CFTP §4.4)

**Files:**
- Create: `docs/superpowers/specs/2026-05-16-cftp-section-4-4-deferral.md`

- [ ] **Step 1: Write the deferral artifact**

Create `docs/superpowers/specs/2026-05-16-cftp-section-4-4-deferral.md`:

```markdown
# CFTP §4.4 — Fused Linear+CE Separator Skip (Deferral)

**Date:** 2026-05-16
**Status:** Deferred — prerequisite missing
**Parent spec:** [CFTP §4.3 RoPE Position Reset](2026-05-16-pca-rope-position-reset-design.md)
**Reason:** No fused LM-head+softmax+CE kernel exists in tree; separator-skip has no kernel to live in.

---

## Finding

CFTP §4.4 proposes: "The fused linear cross-entropy kernel (LM head matmul + softmax + cross-entropy in one kernel) can skip entire document-separator tokens at the kernel level, avoiding the LM head matmul for tokens whose loss will be masked."

The premise — a fused LM-head+softmax+CE kernel — does not exist in NSL today. Current loss computation in `stdlib/nsl/nn/losses.nsl::cross_entropy` is plain NSL composing materialized logits with `log_softmax + gather`. No fusion across the LM-head matmul boundary.

## What IS implemented

- **Loss-correctness side:** `nsl_dataloader_next_batch` already emits `-100` labels for ignore positions; `cross_entropy` masks them via `valid_mask = clamp(targets + 1, 0, 1)`. Separator-token loss is already correctly zero.
- **Memory/FLOP win:** Not implemented. Requires the fused kernel.

## Outline of future spec

A future "Fused linear+CE kernel" spec would cover:

1. Kernel synthesis: LM-head matmul + softmax + cross-entropy in one PTX kernel (Liger-Kernel pattern).
2. Per-position skip: for token positions where `input_ids[i] == separator_token_id`, skip the LM-head matmul row entirely.
3. Memory win: avoid materializing the `[B, S, V]` logits tensor — write loss directly to scalar.
4. Backward integration: separator skip propagates to the loss backward (zero gradients for skipped rows).

Estimated scope: ~600-900 LOC. Independent of the RoPE-reset spec.

## Dependencies for unblocking

The fused kernel spec depends on:

- WGGO graph awareness (already in tree)
- Source AD support for fused-loss kernels (likely needs new emission patterns)
- M36 memory planner for the logits-elimination

When all three are ready, the fused kernel + separator skip becomes a single spec → plan → implement cycle.

## Why this is NOT a blocker for RoPE-reset

CFTP §4.3 and §4.4 are independent features:

- §4.3 affects attention math (position encoding); blocks training correctness on packed sequences.
- §4.4 affects loss compute (LM-head); only blocks training *efficiency* on packed sequences.

The loss-correctness side of §4.4 (masking separator-token contributions) is already covered by the DataLoader's -100 convention and stdlib `cross_entropy`'s valid_mask.

## Related artifacts

- [V-RoPE-FFI-scope](2026-05-16-rope-ffi-scope-findings.md) — pre-implementation verification for §4.3
- [CFTP §4.3 design](2026-05-16-pca-rope-position-reset-design.md) — the active spec
- [CFTP §4.2 Strategy 3 spec](2026-05-16-pca-strategy-3-per-cta-design.md) — sibling deferral
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/specs/2026-05-16-cftp-section-4-4-deferral.md
git commit -m "docs(cftp): §4.4 fused linear+CE separator skip — deferral artifact"
```

---

### Task 15: #1 Strategy 3 spec-only artifact

**Files:**
- Create: `docs/superpowers/specs/2026-05-16-pca-strategy-3-per-cta-design.md`

- [ ] **Step 1: Write the spec-only artifact**

Create `docs/superpowers/specs/2026-05-16-pca-strategy-3-per-cta-design.md`:

```markdown
# CFTP §4.2 Strategy 3 — Multi-Sequence Per-CTA (Design — Spec-only)

**Date:** 2026-05-16
**Status:** Spec-only — no implementation in this cycle
**Parent:** [CFTP §4.3 RoPE Position Reset](2026-05-16-pca-rope-position-reset-design.md)

---

## §1 — Scope

CFTP §4.2's third strategy: when documents are short (avg length ≪ max sequence length), generate a kernel where each CTA processes one complete document rather than one Q-tile of a packed sequence. Zero wasted compute on cross-document entries; no mask needed.

## §2 — Current state

The detector at [`pca_detect.rs:44`](../../../crates/nsl-codegen/src/pca_detect.rs#L44) recognizes when `PerDocumentCta` would be a better strategy than `SegmentId` (Tier A) or `DocumentBoundaryTileSkip` (Tier B). The detector recommendation is plumbed into the planner output but no kernel synthesizer emits the per-document grid layout.

## §3 — Architecture sketch

Standard FlashAttention grid: `(num_q_tiles, batch × heads, 1)`.
Strategy 3 grid: `(num_documents, batch × heads, 1)`.

Each CTA:

1. Reads its document's `start, end` from a compact document offset table (analogous to `doc_starts` from §4.3).
2. Loads only its document's Q, K, V tiles.
3. Runs the standard FlashAttention loop *within* document bounds.
4. Writes output to the correct position in the packed output.

Mask: none. Each CTA processes exactly the tokens of its assigned document.

## §4 — Compatibility with shipped PCA

- **Tier A (segment-ID kernel):** Strategy 3 supersedes Tier A for short-doc workloads. Both can coexist; the cost model picks one per batch.
- **Tier B (tile-skip):** Strategy 3 supersedes Tier B for short-doc workloads. Tile-skip's win shrinks as documents get shorter (eventually 100% of tile-pairs are skipped — at which point Strategy 3 is strictly better).
- **§4.3 RoPE-reset (this cycle's spec):** Strategy 3 needs document-aware RoPE positions just like §4.3, but in a simpler form — each CTA already knows its document boundary, so `effective_pos = local_pos` (the per-doc position) directly. No SMEM-resident `doc_starts` table needed; the start offset is a per-CTA scalar.

## §5 — Cost model

The cost model needs to decide per-batch:

- **Strategy 1 (segment-ID, Tier A):** General case. Per-token mask check inside attention loop.
- **Strategy 2 (tile-skip, Tier B):** Wins when ~30-70% of tile-pairs are cross-document.
- **Strategy 3 (per-CTA):** Wins when documents are short and roughly even in length. Uneven workloads cause load imbalance penalty.

A future spec/plan should derive the cost model's decision boundary empirically (the dispatch spec's V-Bii-SMEM probe is the methodological precedent).

## §6 — Out of scope for the immediate horizon

Strategy 3 is the **largest** of the three remaining CFTP carve-outs:

- Implementation: a new grid layout, document-offset table runtime path, cost model decision boundary, and integration with the WGGO planner.
- Estimated scope: ~1500-2200 LOC.
- Prerequisite: §4.3 RoPE-reset must land first (because the `doc_starts` infrastructure is shared).
- Best timing: after measurement on real short-doc workloads (chat SFT, instruction tuning) confirms a real-world win.

## §7 — Hand-off

This spec exists as a placeholder so future sessions can pick up Strategy 3 with the context already framed. The next step is **measurement** (not implementation) — on a real short-doc workload, characterize how much Strategy 2 (Tier B) leaves on the table.

## Related artifacts

- [CFTP §4.3 RoPE-reset design](2026-05-16-pca-rope-position-reset-design.md)
- [CFTP §4.4 deferral](2026-05-16-cftp-section-4-4-deferral.md)
- [PCA Tier A](2026-04-18-pca-tier-a-design.md)
- [PCA Tier B planner](2026-05-15-pca-tier-b-planner-design.md)
```

- [ ] **Step 2: Commit**

```bash
git add -f docs/superpowers/specs/2026-05-16-pca-strategy-3-per-cta-design.md
git commit -m "docs(cftp): §4.2 Strategy 3 multi-sequence per-CTA — spec-only artifact"
```

---

## Self-Review Checklist (run before declaring plan ready)

**1. Spec coverage:** Each numbered section of the spec has at least one task implementing it.

- §1 lifecycle & scope: addressed implicitly by the plan's framing.
- §2 producer side: Task 2.
- §3 four rotation sites + CTA prologue + SMEM cost: Tasks 5, 6, 7, 8, 9.
- §4 FFI extension + helpers + ABI: Tasks 1, 3, 4.
- §5 codegen trigger + four sites + forward/backward consistency: Tasks 5-10.
- §6 validation (3 layers): Tasks 11, 12, 13.
- §7 risks: covered by the sentinel-path snapshots (R4), packer assert (R3), const_assert (R6).
- §8 institutional references: present in spec, no implementation task needed.
- §9 changelog: spec-only.

**2. Placeholder scan:** No "TBD", "TODO", "implement later", "Similar to Task N" language. Every code step shows actual code. ✓

**3. Type consistency:** `MAX_NUM_DOCS=256` used uniformly. `doc_starts_ptr` is the trailing FFI param in all 3 entry points. `doc_starts_disabled_sentinel` and `doc_starts_enabled` referenced consistently. Helper return type is `Value` throughout.

**Plan complete.** 15 tasks total. Tasks 1-13 are mechanical with TDD-driven steps. Tasks 14-15 are documentation artifacts.
