# V-RoPE-FFI-scope — Pre-Implementation Verification

**Date:** 2026-05-16
**Purpose:** Pin the FFI extension scope for CFTP §4.3 RoPE position-reset before writing the design spec. Mirrors the planner spec's V-Bii-SMEM / V-planner-options / V-dispatch-integration pre-implementation verification pattern.
**Invariant under test:** "Which FFI entry points need a `doc_starts_ptr` extension to support document-aware RoPE position reset?"
**Method:** Grep `crates/nsl-runtime/src/flash_attention.rs` for `pub extern "C" fn nsl_`, inspect each signature for RoPE involvement, classify against the RoPE-reset feature's requirements.

---

## Why this verification matters

A pre-design refinement proposed extending `doc_starts_ptr` to 7 entry points by analogy with the planner spec's Tier B FFI extension (which targeted 6 FA-2 entry points, with `nsl_rope_cache_write` explicitly excluded). The refinement asserted that RoPE-reset should *include* `nsl_rope_cache_write` because "RoPE-reset's document-boundary information lives here."

This is exactly the class of cross-module assumption IR-003 was promoted to catch — a load-bearing claim about external caller behavior whose verification cost is a 15-minute grep but whose mis-bake cost is 4 surplus FFI extension touches + cascade snapshot churn + reviewer confusion.

The original M35.2a tally was 2-for-2 on catching real mismatches via pre-implementation verification (the BitNet STE finding + the dispatch spec's case-(β) finding). This grep is the third anchor.

---

## Evidence — entry-point inventory

`grep "pub extern \"C\" fn nsl_" crates/nsl-runtime/src/flash_attention.rs` returns 11 entries; the 7 substantive ones (excluding alloc/free pairs) are:

| Line | Entry point                                | RoPE involvement                                                       | Position input                              |
|------|--------------------------------------------|-------------------------------------------------------------------------|---------------------------------------------|
| 109  | `nsl_flash_attention`                      | None — plain attention, no fused RoPE                                  | N/A                                          |
| 373  | `nsl_flash_attention_csha`                 | **Fused RoPE** in Q/K projection epilogue                              | Synthesized internally from `q_pos` index    |
| 597  | `nsl_flash_attention_csha_with_saves`      | **Fused RoPE** + Tier C activation saves                               | Synthesized internally from `q_pos` index    |
| 883  | `nsl_flash_attention_csha_backward`        | **Fused RoPE re-application** (must match forward's effective_pos)     | Synthesized internally from `q_pos` index    |
| 1459 | `nsl_flash_attention_quantized`            | None — quantized KV cache path, no RoPE in its PTX body                | N/A                                          |
| 1556 | `nsl_rope_cache_write`                     | RoPE rotation factor write for paged-KV inference                      | **Runtime `positions_ptr` already in FFI**   |
| 2107 | `nsl_flash_attention_backward`             | None — non-CSHA backward                                               | N/A                                          |

### Signature evidence for the load-bearing cases

**`nsl_flash_attention_csha` (line 373):**
```rust
pub extern "C" fn nsl_flash_attention_csha(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,           // <-- precomputed cos/sin tables
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    ...
    segment_ids_ptr: i64,                  // <-- Tier A extension (PCA Strategy 1)
) -> i64
```

Position is **not** an explicit input — the kernel uses its grid-derived `q_pos` (the q-tile's row index inside the packed sequence) to index `cos_ptr` and `sin_ptr` directly. To make position-reset work, the kernel needs `doc_starts_ptr` so it can compute `effective_pos = q_pos - doc_starts[segment_ids[q_pos]]` and re-index cos/sin with `effective_pos`.

**`nsl_rope_cache_write` (line 1556):**
```rust
pub extern "C" fn nsl_rope_cache_write(
    k_projected_ptr: i64, v_projected_ptr: i64,
    cos_ptr: i64, sin_ptr: i64,
    positions_ptr: i64,                    // <-- caller-supplied positions tensor
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_table_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    num_tokens: i64, num_heads: i64, head_dim: i64, block_size: i64,
    ptx_ptr: i64, name_ptr: i64,
) -> i64
```

The caller passes a precomputed `positions_ptr`. To get document-reset behavior, the caller computes `effective_pos = q_pos - doc_starts[segment_ids[q_pos]]` Python-side (or via a small NSL helper) before invocation. **No FFI extension required at this entry point.** The cleanest separation: paged-KV inference path already abstracts positions as caller-supplied data; document-reset is just a different way of populating that tensor.

---

## Conclusion — corrected scope

**3 entry points** receive the `doc_starts_ptr: i64` FFI extension:

1. `nsl_flash_attention_csha`
2. `nsl_flash_attention_csha_with_saves`
3. `nsl_flash_attention_csha_backward`

**4 entry points are out of scope:**

- `nsl_flash_attention`, `nsl_flash_attention_quantized`, `nsl_flash_attention_backward` — no fused RoPE in their PTX bodies.
- `nsl_rope_cache_write` — already accepts an arbitrary `positions_ptr`; caller computes reset-aware positions externally.

### Tally vs original refinement

| Source                     | Count | Notes                                                            |
|----------------------------|-------|------------------------------------------------------------------|
| Original refinement asserts | 7    | Tier B planner-spec analogy (6 FA-2 + `nsl_rope_cache_write`)    |
| Code verifies              | 3    | Only CSHA-fused-RoPE entry points need the extension             |
| Net delta                  | -4   | 4 entry points avoided                                            |

### Sentinel encoding

Existing call sites of the 3 in-scope entry points pass `0` as `doc_starts_ptr`. The kernel's RoPE epilogue checks `doc_starts_ptr != 0` and falls through to the identity-position path (`effective_pos = q_pos`) on sentinel. This preserves byte-stable PTX for the `segment_masked=false, doc_starts_disabled` configuration — same discipline as the planner spec's `tier_b_disabled_sentinel`.

Helper functions per IR-001 (API-shape-enforced invariants):

```rust
// crates/nsl-codegen/src/pca_rope.rs (extends existing module)
pub fn doc_starts_disabled_sentinel<M: Module>(builder: &mut FunctionBuilder, ...) -> Value { /* zero */ }
pub fn doc_starts_enabled<M: Module>(builder: &mut FunctionBuilder, module: &mut M, data_id: DataId) -> Value { /* data-section addr */ }
```

Matches the planner spec's `tier_b_disabled_sentinel` / `tier_b_enabled` pattern.

---

## Institutional references

- **IR-003 — pre-implementation verification:** the discipline this artifact implements.
- **IR-013 — external-caller-context assumptions warrant pre-implementation verification:** promoted concurrently with the planner spec (`docs/wiki/institutional-rules.md`). RoPE-reset's FFI scope is precisely this class of assumption.
- **Planner spec §4.6 / §4.8:** the precedent for "Tier B FFI extension targets 6 entry points; `nsl_rope_cache_write` is explicitly excluded." Re-cited here to make the scope difference explicit — Tier B excluded `nsl_rope_cache_write` because Tier B's tile-skip logic doesn't apply to RoPE cache writes; RoPE-reset excludes it for a different reason (already runtime-driven via `positions_ptr`).

---

## Re-bound brainstorm cycle tally

Pre-implementation verification has now caught 3 load-bearing assumption mismatches in 3 consecutive specs:

1. M35.2a STE finding — vanilla STE, not clipped (PR #164)
2. Dispatch spec case-(β) finding — production callers lack seq_len at codegen time (PR #169 amendment)
3. **RoPE-FFI-scope finding (this artifact)** — 3 entry points need extension, not 7

The discipline's cost-per-finding is roughly 15-30 minutes of grep work. Each finding prevents at least one round of cascade snapshot churn + reviewer surface mismatch. The cumulative ROI argues for keeping IR-003 as a load-bearing institutional rule.
