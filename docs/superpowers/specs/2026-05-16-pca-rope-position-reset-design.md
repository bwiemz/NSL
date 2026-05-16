# CFTP §4.3 — PCA RoPE Position Reset (Design)

**Date:** 2026-05-16
**Status:** Design — awaiting user review before implementation plan
**Pre-implementation verification:** [V-RoPE-FFI-scope](2026-05-16-rope-ffi-scope-findings.md) (committed 27176082) — 3 entry points, not 7

---

## §1 — Scope and lifecycle pinning

This spec implements **CFTP §4.3 — Position ID Fusion**, the third of three PCA sub-features the CFTP paper proposes. Status of PCA features prior to this spec:

| CFTP § | Feature | Status |
|--------|---------|--------|
| 4.2 Strategy 1 | Segment-ID kernel (Tier A) | Shipped — PRs #78/#105/#109 |
| 4.2 Strategy 2 | Tile-skip map (Tier B) | Shipped — PRs #168/#169 (kernel), #175 (runtime dispatch) |
| 4.2 Strategy 3 | Multi-Sequence Per-CTA | **Spec-only deferral**, see [companion artifact](2026-05-16-pca-strategy-3-per-cta-design.md) |
| **4.3** | **RoPE position reset (this spec)** | **Design — awaiting review** |
| 4.4 | Fused linear+CE separator skip | **Deferred — prerequisite missing**, see [companion artifact](2026-05-16-cftp-section-4-4-deferral.md) |

### Relationship to existing PCA work

- **Tier A** added `segment_ids_ptr: i64` at the end of each CSHA-fused entry point's signature (sentinel-0 path preserves the pre-Tier-A signature byte-stably). This spec extends that pattern with a sibling `doc_starts_ptr: i64` parameter inserted immediately after `segment_ids_ptr`.
- **Tier B's planner spec** established the helper-encapsulated sentinel pattern (`tier_b_disabled_sentinel` / `tier_b_enabled`) per IR-001. This spec reuses the pattern: `doc_starts_disabled_sentinel` / `doc_starts_enabled` in [`pca_rope.rs`](../../../crates/nsl-codegen/src/pca_rope.rs).
- **V-RoPE-FFI-scope** narrowed the apparent 7-entry-point surface to **3 entry points** (csha / csha_with_saves / csha_backward). `nsl_rope_cache_write` already takes `positions_ptr` and is out of scope; the other 3 entry points have no RoPE in their PTX bodies.

### Out of scope

- The fused linear+softmax+CE kernel needed for CFTP §4.4 (separate spec).
- Multi-Sequence Per-CTA grid layout for short-doc packing (separate spec).
- Inference-side RoPE-reset — paged-KV inference uses `nsl_rope_cache_write` with caller-supplied `positions_ptr`; callers compute reset positions externally.

---

## §2 — Producer side: DataLoader emits `doc_starts`

### Data shape

A new per-batch tensor:

```text
doc_starts: Tensor<[MAX_NUM_DOCS + 1], i32>
```

with **`MAX_NUM_DOCS = 64`** (compile-time constant) and **sentinel `-1`** for unused slots. The tensor's logical content for a batch with `K` documents:

```text
doc_starts[0]     = 0
doc_starts[1]     = len(doc_0)
doc_starts[2]     = len(doc_0) + len(doc_1)
...
doc_starts[K]     = packed_length
doc_starts[K+1..] = -1  (sentinel — unused slots)
```

The kernel reads `doc_starts[segment_ids[q_pos]]` directly; segment_ids' value range is `[0, K)`, so the kernel never indexes the sentinel slots. The sentinel exists only so kernel codegen can use a fixed `[65]` SMEM layout independent of runtime `num_docs`.

### Ownership and lifecycle

- **Producer:** `nsl_dataloader_next_batch` in [`crates/nsl-runtime/src/packing.rs`](../../../crates/nsl-runtime/src/packing.rs). The packer already knows document boundaries (the packer constructs them); emitting `doc_starts` is metadata at packing time with no Python overhead.
- **Owner:** the batch dict — `doc_starts` is reachable as `batch["doc_starts"]` alongside `batch["segment_ids"]` and `batch["input_ids"]`. Freed when the batch is freed (existing batch-lifecycle machinery).
- **Device:** lives in device memory (same as `segment_ids`) — the kernel reads from device, not host.

### Compile-time bound assertion

```rust
// crates/nsl-codegen/src/pca_rope.rs
pub const MAX_NUM_DOCS: u32 = 64;

const _: () = assert!(MAX_NUM_DOCS <= 256, "MAX_NUM_DOCS must fit in u16 for SMEM-bounded reads");
const _: () = assert!(
    (MAX_NUM_DOCS + 1) * 4 <= 1024,
    "doc_starts SMEM bake must stay well under per-CTA budget"
);
```

(Mirrors the planner spec's `TIER_B_SEQ_LEN_FLOOR ≤ TIER_B_MAX_BAKED_SEQ_LEN` joint-compat assert.)

### Producer-side runtime invariant

```rust
// inside nsl_dataloader_next_batch packing path
debug_assert!(num_docs <= MAX_NUM_DOCS, "packed batch has {num_docs} > MAX_NUM_DOCS={MAX_NUM_DOCS}");
```

Runtime failure mode: the packer must enforce `num_docs ≤ MAX_NUM_DOCS` at packing time. For seq_len=16384 (the planner spec's MAX_BAKED), `MAX_NUM_DOCS=64` allows avg doc ~256 tokens — comfortable for typical pretraining. If a workload violates this (very-short-doc packing), the packer asserts at batch-construction time, surfacing the constraint before any kernel launches.

---

## §3 — Consumer side: RoPE epilogue uses `effective_pos`

### Current kernel behavior (CSHA Level 1)

The fused RoPE epilogue lives in [`flash_attention_v2/phases/forward/prelude.rs`](../../../crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs) lines ~228-242. The current PTX path:

```text
%r_rope_cs_idx = q_pos * (head_dim / 2) + dim_pair
cos = cos_ptr[%r_rope_cs_idx]
sin = sin_ptr[%r_rope_cs_idx]
(x0', x1') = rotate(x0, x1, cos, sin)
```

`q_pos` is grid-derived (the q-tile row index inside the packed sequence). When the sequence is packed and documents are concatenated, `q_pos` is the **packed** index — wrong for RoPE inside any document past the first.

### New kernel behavior with doc_starts

```text
[CTA prologue, executed once per CTA]
if doc_starts_ptr != 0:
    parallel-load doc_starts[0..MAX_NUM_DOCS+1] into SMEM (one warp, 65 i32 reads from HBM)
    smem_doc_starts[0..MAX_NUM_DOCS+1] = doc_starts

[Per-q_pos in the q-tile loop]
sid = segment_ids[q_pos]                       # already SMEM-resident from Tier A
if doc_starts_ptr != 0:
    effective_pos = q_pos - smem_doc_starts[sid]
else:
    effective_pos = q_pos                      # identity / pre-existing behavior
%r_rope_cs_idx = effective_pos * (head_dim / 2) + dim_pair
cos = cos_ptr[%r_rope_cs_idx]
sin = sin_ptr[%r_rope_cs_idx]
(x0', x1') = rotate(x0, x1, cos, sin)
```

### SMEM cost

- `doc_starts` table: `(MAX_NUM_DOCS + 1) * 4 = 260 bytes` per CTA.
- Loaded once at CTA prologue, reused across all q_pos in the CTA.
- Lookup is a single SMEM read per RoPE-rotated dim_pair — same latency class as the segment_ids lookup Tier A already does.

Joint-compat with the planner spec's V-Bii-SMEM budget (MAX_BAKED_SEQ_LEN=16384):
- Tier B SMEM (when active): ~16384 bytes for the q-tile range table.
- Tier A SMEM: ~seq_len bytes for segment_ids.
- doc_starts SMEM (this spec): 260 bytes.
- Joint total at MAX configuration: comfortably under sm_75's 48 KB and sm_120's 100 KB usable SMEM.

Compile-time assert lands in [`pca_rope.rs`](../../../crates/nsl-codegen/src/pca_rope.rs):

```rust
const _: () = assert!(
    (MAX_NUM_DOCS + 1) * 4 + 16384 + 16384 < 48 * 1024,
    "Tier B + Tier A + RoPE-reset SMEM joint bake must fit sm_75 limit"
);
```

(Conservative: 16384 for Tier B range table + 16384 for segment_ids ceiling + 260 for doc_starts ≈ 33 KB.)

### Sentinel path byte-stability

When `doc_starts_ptr == 0`:
- The kernel's RoPE epilogue PTX is **identical** to the pre-spec PTX (no extra ld.shared, no extra sub).
- Existing callers (Tier A only, no packing-aware RoPE) pass `0` and observe byte-stable PTX for the segment_masked=true / doc_starts_disabled configuration.

The sentinel check happens at PTX-emit time (codegen decides whether to emit the new prologue + per-q_pos sub), not at runtime — so the kernel-name suffix encodes the variant (analogous to `_tier_b_max16384`). Variant kernel name: `..._rope_reset_max64` for the doc_starts-active variant; unchanged name for the sentinel/identity variant.

---

## §4 — FFI extension

### Entry points (3 — per V-RoPE-FFI-scope)

1. `nsl_flash_attention_csha` (line 373)
2. `nsl_flash_attention_csha_with_saves` (line 597)
3. `nsl_flash_attention_csha_backward` (line 883)

### Signature change

Each gets one new `i64` parameter inserted **immediately after `segment_ids_ptr`** (which is currently the last param for `csha`; for the other two the insertion site is between segment_ids_ptr and any subsequent trailing params — see implementation plan for exact byte positions):

```rust
pub extern "C" fn nsl_flash_attention_csha(
    ...,
    segment_ids_ptr: i64,
    doc_starts_ptr: i64,   // <-- new — sentinel 0 = identity-position path
) -> i64
```

### Helper functions (IR-001)

```rust
// crates/nsl-codegen/src/pca_rope.rs (extends existing module)

/// Construct a sentinel-zero `doc_starts_ptr` value at a Cranelift call site.
/// Identity-position semantics (matches pre-spec behavior).
pub fn doc_starts_disabled_sentinel<M: Module>(
    builder: &mut FunctionBuilder<'_>,
) -> Value {
    builder.ins().iconst(types::I64, 0)
}

/// Construct an enabled `doc_starts_ptr` value pointing at a runtime tensor.
/// The caller is responsible for ensuring `data_id` references a valid
/// `[MAX_NUM_DOCS+1]` i32 tensor in device memory.
pub fn doc_starts_enabled<M: Module>(
    builder: &mut FunctionBuilder<'_>,
    module: &mut M,
    data_id: DataId,
) -> Value {
    let gv = module.declare_data_in_func(data_id, builder.func);
    builder.ins().symbol_value(types::I64, gv)
}
```

Same shape and discipline as the planner spec's `tier_b_disabled_sentinel` / `tier_b_enabled` helpers.

### Header documentation

The 3 entry points get matching docstring extensions:

```rust
/// PCA §4.3: doc_starts device pointer for packed-sequence training with
/// document-aware RoPE position reset. Pass 0 to disable (identity positions).
/// When non-zero, must reference a `[MAX_NUM_DOCS+1]` i32 tensor in device
/// memory with sentinel `-1` for unused slots.
doc_starts_ptr: i64,
```

### ABI preservation

Existing call sites that pass `0` for `doc_starts_ptr` see byte-stable PTX. The compiled binary's call instruction gains one trailing 8-byte argument; no other layout changes.

---

## §5 — Codegen trigger

### Activation condition

RoPE-reset is **auto-on** when all three of the following hold at codegen time:

1. `config.segment_masked == true` — the dataset has `packing=true`, so Tier A is also active.
2. `config.rope_q == true` — the model uses RoPE Q/K projection (existing CSHA Level 1 flag).
3. The producer (DataLoader) emits `doc_starts` — guaranteed by the same packing path that emits `segment_ids`.

No user-facing decorator (`@pca(rope_reset=true)` etc.) is introduced. Rationale: RoPE without document-aware reset gives *wrong positions* for any token past the first document boundary — there is no realistic workload that wants packed sequences + RoPE *without* reset. Auto-on prevents the silent-correctness footgun.

### Codegen integration site

[`flash_attention_v2/phases/forward/prelude.rs`](../../../crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs):
- New `if config.segment_masked && config.rope_q && config.doc_starts_active` guard.
- When true, emit the CTA-prologue `doc_starts` SMEM load and the per-q_pos `effective_pos` computation.
- When false, emit the existing identity-position path unchanged.

### Forward + backward consistency

`nsl_flash_attention_csha_backward` must reapply the *same* `effective_pos` to compute gradients w.r.t. RoPE-rotated Q/K. Implementation:
- The backward kernel receives the same `doc_starts_ptr` from the runtime.
- Its RoPE de-rotation step uses identical `effective_pos = q_pos - doc_starts[segment_ids[q_pos]]` to undo the forward rotation.
- Source AD differentiates the forward's `effective_pos` computation as an integer-domain operation with no learnable parameters — backward's effective_pos is the same data-dependent function, not a learned mapping.

---

## §6 — Validation

### Layer 1 — PTX emission snapshots

New snapshot tests in `crates/nsl-codegen/tests/pca_rope_emission.rs`:

1. **`rope_reset_disabled_byte_stable`** — assert the emitted PTX for `segment_masked=true, doc_starts_active=false` is byte-identical to the pre-spec PTX. Verifies the sentinel path doesn't drift.
2. **`rope_reset_enabled_prologue_present`** — assert the PTX for `segment_masked=true, doc_starts_active=true` contains the `doc_starts` SMEM prologue and `effective_pos` sub.
3. **`rope_reset_enabled_max_docs_64`** — assert the SMEM declaration includes `260 + segment_ids_size` bytes.
4. **Kernel-name suffix:** `..._rope_reset_max64` present for the enabled variant.

### Layer 2 — Numerical parity (CPU reference)

New tests in `crates/nsl-codegen/tests/pca_rope_numerical.rs`:

1. **Single-doc parity:** A packed sequence with `num_docs=1, doc_lengths=[N]` must produce RoPE-rotated Q/K identical to an unpacked sequence of length N. (effective_pos = q_pos when doc_starts[0]=0.) Tolerance: bit-exact.
2. **Three-doc parity:** A 3-doc packed sequence with `doc_lengths=[a, b, c]` must produce RoPE-rotated Q/K that, when split by document boundary, equals 3 separate per-document forwards. Tolerance: `5e-3` (f16-arithmetic bound).
3. **Sentinel parity:** `doc_starts_ptr=0` must produce the same output as the pre-spec kernel (identity positions).

### Layer 3 — On-GPU smoke

Gated on `cfg(feature = "cuda")`:

1. **End-to-end emission + launch:** Compile a 2-doc packed config, launch on a real GPU, compare against the Layer 2 CPU reference. Tolerance: same as Layer 2.

### Test gating

- Layers 1 and 2 run in CI on every commit (no GPU required).
- Layer 3 runs in CI under the `cuda` feature; on machines without a CUDA GPU it's skipped.
- All three layers must pass before the spec's implementation plan is considered done.

---

## §7 — Risks

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | doc_starts must match segment_ids exactly (producer-side invariant). | Single producer (packer) constructs both tensors in the same loop. Debug-assert in packer that segment_ids[i] < num_docs for all i. |
| R2 | Backward must use identical effective_pos to forward (cross-kernel invariant). | Backward consumes the same doc_starts_ptr from runtime — not recomputed; cross-checked in Layer 3 on-GPU smoke. |
| R3 | num_docs > MAX_NUM_DOCS at runtime. | Packer asserts `num_docs <= MAX_NUM_DOCS` at packing time. The packer is the source of truth; failure surfaces at batch-construction, before any kernel launch. |
| R4 | Sentinel path drift (doc_starts_disabled becomes byte-non-stable over time). | Layer 1 snapshot 1 explicitly asserts byte-stability vs pre-spec PTX. |
| R5 | Inference-side RoPE-reset uncovered. | Documented limitation: `nsl_rope_cache_write` callers compute reset-aware positions externally. Pre-V-RoPE-FFI-scope this looked like a code gap; verification showed it's the cleaner separation. |
| R6 | Joint SMEM with Tier A + Tier B exceeds budget. | Compile-time const_assert in `pca_rope.rs` bounds joint cost; tested against sm_75 (48 KB) and sm_120 (100 KB). 260 bytes is trivially OK. |
| R7 | `@pca(off)` decorator's interaction with auto-on. | When user sets `@pca(off)`, segment_masked is forced false; the RoPE-reset path is therefore also off (gated on segment_masked). No new interaction surface. |

---

## §8 — Institutional references

- **IR-001 — API-shape-enforced invariants:** sentinel construction via `doc_starts_disabled_sentinel` / `doc_starts_enabled` helpers.
- **IR-002 — external references as one-time anchors:** Tier A's `segment_ids_ptr` extension is the precedent reference (this spec re-cites once, then proceeds).
- **IR-003 — pre-implementation verification:** V-RoPE-FFI-scope findings doc anchors the FFI-extension surface.
- **IR-009 — bounded-dormancy clock:** N/A — this spec activates on landing.
- **IR-013 — external-caller-context assumptions warrant pre-implementation verification:** the rule V-RoPE-FFI-scope tests.
- **Planner spec §4.6 / §4.8 — Tier B FFI extension precedent:** structural twin to this spec's FFI extension; same helper-discipline pattern.

---

## §9 — Revision changelog

- **2026-05-16 v1** — initial design. Decisions captured: fixed-size [65] doc_starts with sentinel -1; MAX_NUM_DOCS=64; auto-on activation; 3-entry-point FFI extension; full validation strategy (snapshots + parity + on-GPU smoke).
