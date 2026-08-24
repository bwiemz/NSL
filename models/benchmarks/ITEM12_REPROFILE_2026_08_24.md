# Item 12 — re-profile after per-segment early-free, and the fusion verdict

The roadmap's ordering rule for this item: *re-profile the new peak, and only
then add targeted fusion or lifetime transformations.* This is the re-profile.
The verdict is that **no further activation transformation is justified at
1B**, with the measurements that force that conclusion.

Probe: `models/coder1b/mem_probe_32step.nsl` (reproduces the full-epoch
allocator peak to the byte), leak-fixed build of PR #524, fully-resident
configuration (no offload — the workflow item 11 unlocked).

## The new peak, decomposed

**19.47 GiB total** (20,906,110,976 B), 11.9 GiB of margin on the 31.39
usable. At-peak context table:

| what | bytes | share |
| --- | --- | --- |
| persistent (weights 4.02 + optim m/v/m_partial 12.0) | 16.02 GB | **82.3%** |
| boundary saves (`nsl_add_f32`, the escaping residuals) | 1.66 GB | 8.5% |
| matmul workspace | 656 MB | 3.3% |
| elementwise working set (`div`/`mul`/`silu`/`rmsnorm`) | ~977 MB | 4.9% |
| `sum_dim` | 64 MB | 0.3% |

Allocator current-bytes is FLAT across steps (16,439 MB steady state).

## Why no fusion or further lifetime work is justified

1. **The peak is optimizer-state-dominated.** 61.6% of it is f32 AdamW
   m/v/m_partial. No activation transformation touches that. If headroom is
   ever needed, the lever that exists TODAY is optimizer-state precision
   (CPDT reduced-precision moments) or `--optim-state-offload` — not fusion.
2. **Half the remaining activation surface is the checkpointing contract.**
   The 1.66 GB of boundary saves is what `--checkpoint-blocks` KEEPS so the
   backward can recompute; no lifetime pass may free them, and fusing
   elementwise ops does not shrink them.
3. **The fusable lines total under 1 GiB.** Perfect fusion of every
   elementwise context (div + mul + silu + rmsnorm ≈ 0.98 GB) buys less than
   1 GiB against an 11.9 GiB margin. Nothing currently blocks on that
   gigabyte.
4. **Stride coalescing re-measured: still worse.** The pre-item-11 record
   said "stride-2 made the peak WORSE" — measured when the forward peak was
   interior-dominated, so per-segment freeing could have flipped it. It did
   not: stride-2 resident = 20.25 GiB peak / 4.54 GiB activations (vs 19.47 /
   3.42 at stride 1). Boundary saves barely change (1.64 GB — kept anchors,
   not segment count, decide them) while the coalesced double-segment's
   working set grows (matmul 656 MB → 1.00 GB, mul 337 → 545 MB): a
   double-segment holds two blocks' interiors until ITS end. The conclusion
   survives the discipline change; it is not stale.

## What this closes and what it defers

Closed: items 11 + 12 of the detour plan. The fully-resident 1B workflow
runs with grad clipping and f32 optimizer state, 13.12 GiB came off the
activation peak, and the measured composition says further activation work
buys, at most, tenths of the margin already available.

Deferred, with the trigger stated: revisit activation fusion only if a
future configuration (longer sequence, bigger batch, wider model on this
card) pushes the peak within ~2 GiB of the usable ceiling AND the at-peak
table still shows ≥1 GiB in fusable elementwise contexts. Until both hold,
fusion here is work without a customer.
