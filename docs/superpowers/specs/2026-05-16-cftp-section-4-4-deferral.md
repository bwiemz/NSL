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
