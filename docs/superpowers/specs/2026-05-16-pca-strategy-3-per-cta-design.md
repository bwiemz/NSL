# CFTP §4.2 Strategy 3 — Multi-Sequence Per-CTA (Design — Spec-only)

**Date:** 2026-05-16
**Status:** IMPLEMENTED (superseding the original "Spec-only" status). The
per-document CTA forward and backward kernels are emitted by
`flash_attention_v2::per_doc_cta::synthesize_per_doc_cta_forward`/`_backward`,
admission is wired via `pca_per_doc::admit`, and the path is enabled end to end
by `@pca(strategy=per_document)` (see `pca_per_doc.rs` module docs). The design
sketch below is retained for historical context.
**Parent:** [CFTP §4.3 RoPE Position Reset](2026-05-16-pca-rope-position-reset-design.md)

---

## §1 — Scope

CFTP §4.2's third strategy: when documents are short (avg length ≪ max sequence length), generate a kernel where each CTA processes one complete document rather than one Q-tile of a packed sequence. Zero wasted compute on cross-document entries; no mask needed.

## §2 — Current state

The detector at [`pca_detect.rs:44`](../../../crates/nsl-codegen/src/pca_detect.rs#L44) recognizes when `PerDocumentCta` would be a better strategy than `SegmentId` (Tier A) or `DocumentBoundaryTileSkip` (Tier B). **Update:** the per-document grid-layout synthesizers now exist (`per_doc_cta::synthesize_per_doc_cta_forward`/`_backward`) and are reached in production when the user requests `@pca(strategy=per_document)`; the detector's recommendation alone does not auto-enable it.

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
