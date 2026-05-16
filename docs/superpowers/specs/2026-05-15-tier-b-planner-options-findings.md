# Tier B Planner — V-planner-options Findings

**Date:** 2026-05-15
**Branch:** `worktree-feat-pca-tier-b-dispatch`
**Builds on:** [`2026-05-14-pca-tier-b-dispatch-design.md`](2026-05-14-pca-tier-b-dispatch-design.md) §14.3 — pivot to follow-on planner spec with three scoped options.
**Status:** verification only — no source edits.

## Why this verification exists

The dispatch spec's §14.3 amendment proposed three options for the follow-on planner spec:

- **Option A** — Make `seq_len` available at codegen time (substantial upstream change).
- **Option B** — Move dispatch past codegen; runtime branch in kernel-launch wrapper (~50 LOC; **recommended**).
- **Option C** — Defer to NSL's shape-inference maturity.

The amendment's recommendation for Option B was made under time pressure during the pivot. Inheriting that recommendation without verification would repeat the dispatch spec's case-(α) failure mode at one layer up. **V-planner-options runs an analogous 30-45 minute pre-implementation verification before the planner spec's §3 commits to A / B / C.**

Same institutional pattern as V-dispatch-integration (IR-013, first instance): pre-implementation verification before the dependent design commits.

## Verification protocol

Three sub-investigations, each producing a single-paragraph answer:

1. **(A) AST threading audit:** what does FlashAttentionConfig currently carry, and which phases consume it? Specifically: would adding `seq_len: u32` break existing invariants?
2. **(B) Launch wrapper inspection:** what's the shape of NSL's kernel-launch path? Does it already do runtime variant selection? Does it have `seq_len` in scope?
3. **(C) Roadmap check:** is shape-inference improvement planned in `docs/superpowers/specs/`? If not, Option C reduces to indefinite deferral.

## Sub-investigation (A) — AST threading audit

`FlashAttentionConfig` (definition: `crates/nsl-codegen/src/flash_attention.rs:142`) carries 13 fields. Every dimensional field is a **PTX-layout** dimension, not a runtime tensor dimension:

| Field | Kind | Why baked into config |
|-------|------|------------------------|
| `block_q: i64` | PTX-layout | Tile size; CTA shape depends on it. |
| `block_kv: i64` | PTX-layout | Tile size; cp.async stride depends on it. |
| `head_dim: i64` | PTX-layout | Per-head MMA dimensions specialize the kernel. |
| `causal: bool` | PTX-layout | Selects masked vs unmasked S-compute path. |
| `paged: bool` | PTX-layout | Selects paged-KV indexing. |
| `rope_q: bool` | PTX-layout | RoPE prelude emission. |
| `rope_style: RopeStyle` | PTX-layout | RoPE variant. |
| `gqa_group_size: u32` | PTX-layout | Per-K-head replication factor. |
| `tree_mask: bool` | PTX-layout | Speculative-decoding mask emission. |
| `gpu_sm: u32` | PTX-layout | Target SM version (cp.async availability). |
| `segment_masked: bool` | PTX-layout | Segment-mask kernel variant. |
| `csha: Option<CshaExtras>` | PTX-layout | CSHA fusion variant. |

**No runtime tensor dimensions are in the config.** `batch`, `seq_len`, and `head_count` are all passed through the FFI dispatcher at launch time (see §B below). The pattern is deliberate: the config IS the kernel-naming surface — `flash_attention_kernel_name_v2(config)` mangles every field into the symbol name, which `cuModuleGetFunction` then resolves at launch. Adding `seq_len: u32` would either (i) bloat per-length kernel-name variants (one PTX blob per length bucket — multiplicative explosion), or (ii) violate the invariant that every config field is part of the kernel's name.

**Option A's cost is therefore higher than the amendment estimated** — it's not just "thread seq_len through codegen"; it's an architectural revision of what FlashAttentionConfig represents. Either:

- (A-i) seq_len becomes a kernel-name-affecting field → per-length kernel explosion.
- (A-ii) seq_len is added as a config field deliberately excluded from name mangling → the "type-system honesty" failure mode (§5.3 of the dispatch spec).
- (A-iii) seq_len is threaded NOT via FlashAttentionConfig but via a separate `KernelLaunchContext` parameter to `synthesize_flash_attention_ptx_v2` — but then we're back to case (α) of V-dispatch-integration, which production callers don't have.

None of A-i / A-ii / A-iii is mechanically cheap. **Option A's true cost is ~architectural revision of the kernel-config surface, not ~50 LOC of threading.**

### Verdict for (A)

Option A is feasible only as architectural revision, not as additive threading. Cost likely 5-10× the amendment's "substantial" framing. **Not the cheapest path.**

## Sub-investigation (B) — Launch wrapper inspection

`nsl_flash_attention_csha` (definition: `crates/nsl-runtime/src/flash_attention.rs:373`) is the host-side FFI launch entry point for CSHA-flavored Flash Attention kernels. Inspection reveals three load-bearing facts:

1. **`seq_len: i64` is already a parameter** at line 378. The host-side dispatcher has runtime seq_len available without any architectural change.
2. **Runtime variant selection is an established pattern** at lines 414-418: `effective_heads` is computed from `active_heads` (a config-baked compile-time hint set by `CshaExtras.active_heads`) and `heads` (the actual runtime count). The dispatcher picks `effective_heads = if active_heads > 0 && active_heads < heads { active_heads } else { heads }` and shrinks `grid_y` accordingly. This is the **exact pattern Option B would use** — a runtime branch keyed on a compile-time hint and a runtime value.
3. **`grid_x = (seq_len + block_q - 1) / block_q`** is computed at line 419 — the dispatcher is already doing runtime arithmetic on seq_len to size the launch grid. Not a thin pass-through.

Additionally, the dispatcher receives:

- `ptx_ptr: i64, name_ptr: i64` (line 385) — the PTX bytes and kernel name come from the AOT-compiled Cranelift module's `.rodata`. For dual-emission Option B, the codegen layer would emit BOTH Tier-B-on and Tier-B-off PTX blobs and pass BOTH to the runtime via extended FFI; the dispatcher picks which to `cuModuleLoadData`.
- `segment_ids_ptr: i64` (line 403) — segment-mask presence is already a runtime FFI argument. The runtime KNOWS whether segments are in use without needing additional codegen-time information.
- `seq_lens_ptr: i64` (line 383) — per-segment lengths already plumbed through. Tier B's range table could potentially be sized from this at runtime (see (B-iii) below).

### Option B's sub-question — single-emission vs dual-emission

V-planner-options surfaced a sub-question the amendment didn't capture: Tier B's SMEM range tables are sized at codegen time. `pca_tilerange.rs::compute_range_table_bytes(seq_len, block_q, block_kv)` returns a byte count that determines the `.shared` declaration in the emitted PTX. This means Option B has three sub-variants:

- **(B-i) Dual-emission, both blobs co-resident.** Codegen emits Tier-B-on PTX (sized for a specific seq_len) AND Tier-B-off PTX (the existing one). FFI signature extends to carry both PTX blobs + names. Runtime dispatcher picks based on `segment_ids_ptr != 0 && seq_len >= floor`. Binary-size cost: ~2× PTX bytes for `segment_masked` configs; 1× for the rest. ~50-100 LOC at launch sites + extended FFI signature.

- **(B-ii) Single-emission with conservative max seq_len.** Codegen emits Tier-B-on PTX only when `config.segment_masked`, sized for a conservative compile-time max (e.g., 16384). FFI signature unchanged. Runtime trusts that `seq_len <= MAX_BAKED_SEQ_LEN` and dispatcher reads only the first `(seq_len / block_*) + 1` range-table entries; remaining SMEM region is reserved-but-unread. Binary-size cost: 1× PTX per config, but SMEM footprint is sized for the max. ~30-50 LOC at launch sites. **Risk**: SMEM cap concern (the 99 KB Blackwell cap from B.1.5-2 returns at MAX=16384 + block=32). Needs measurement-gated verification that the max-seq_len SMEM fits.

- **(B-iii) Single-emission with HBM-resident range tables.** Codegen emits Tier-B-on PTX that reads the range table from HBM via a runtime-bound pointer (analogous to how `seq_lens_ptr` already works). FFI signature extends to carry a range-table device pointer; runtime computes the table from `seq_lens` and uploads it via cudaMemcpy. **Cost**: real kernel architecture change — moves the range-table data from SMEM to HBM with all the latency cost that implies. Original Tier B design (2026-05-02 §11) explicitly listed "HBM-resident tables for seq_len > 16 K" as a deferred extension; B-iii pulls that work forward.

### Verdict for (B)

Option B is feasible. The launch wrapper already has the shape needed (seq_len in scope, runtime variant selection precedent). The sub-question is which (B-i / B-ii / B-iii) to commit to:

- **(B-i) preferred for v1**: dual-emission has the cleanest blast radius (no kernel changes, just FFI extension + runtime branch). Binary-size cost is bounded (~2× only for segment_masked kernels, which are a minority of configs in production today).
- **(B-ii) viable but needs SMEM verification**: a measurement-gated step is required before committing.
- **(B-iii) deferred**: kernel architecture change beyond v1 scope; matches the original 2026-05-02 spec §11 framing.

The planner spec's §3 should commit to (B-i) with explicit reasoning, deferring (B-ii) to a future optimization and (B-iii) to the deferred-extension queue.

## Sub-investigation (C) — Shape-inference roadmap check

Grep across `docs/superpowers/specs/*.md` for `shape inference`, `shape-inference`, `shape_inference` (case-insensitive): **zero matches.** No planned shape-inference work exists as a dated spec.

Cross-check: MEMORY.md's roadmap section mentions "M49 shape algebra" under v0.3-v0.8 moat features. However:

- M49 is in the "v0.3-v0.8 (Phases 4-9, M32-M51)" range — currently no dated spec exists.
- The MEMORY.md roadmap framing for M49 is "Shape algebra" (a moat feature), not specifically "shape inference improvements that thread seq_len through codegen."
- The codebase is at v0.9.0 per `project_version_status`; M49 is upstream of v0.9.0 in the milestone numbering, suggesting it's either already-done-in-some-form or skipped.

**Without a dated, planned shape-inference improvement, Option C reduces to indefinite deferral.** "Defer dispatch activation until shape inference matures" with no concrete shape-inference work scheduled is functionally equivalent to "defer indefinitely" — the IR-009 anti-pattern.

### Verdict for (C)

Option C is **not viable in v1** absent a concrete dated shape-inference spec to depend on. If M49 surfaces with a dated spec and includes seq_len-at-codegen as part of its scope, the planner spec can revisit C as a future migration trigger. Until then, C is excluded.

## Tally

| Option | Cost (verified) | Viability for v1 |
|--------|----------------|-------------------|
| **A** | ~5-10× the amendment's "substantial" framing (architectural revision of FlashAttentionConfig surface) | Not viable; reserves to future architectural revision |
| **B** | ~50-100 LOC for sub-variant (B-i); existing launch-wrapper patterns reusable | **Viable; sub-variant (B-i) recommended** |
| **C** | N/A (no shape-inference roadmap exists) | Not viable; collapses to indefinite deferral |

## Recommendation for the planner spec

**Option B sub-variant (B-i)** — dual-emission with both PTX blobs co-resident; FFI extension to carry both; runtime dispatcher picks based on `segment_ids_ptr != 0 && seq_len >= TIER_B_SEQ_LEN_FLOOR`.

### Why (B-i) over (B-ii)

- (B-i) has the **cleanest blast radius**: no kernel changes, no SMEM-cap risk, no architectural extension.
- (B-i)'s binary-size cost (~2× for segment_masked configs only) is bounded and measurable.
- (B-ii) requires a fresh SMEM-budget measurement at the conservative-max seq_len, similar to the SMEM probe Phase 0 of the revision spec. The measurement infrastructure exists, but the work is non-trivial and 2026-05-13's findings already covered the gate-fixture dimensions.
- (B-iii) is a deferred-extension item per the original 2026-05-02 spec §11.

### What the planner spec must additionally pin

- **Floor value source**: the floor is computed by the dispatch spec's D-2 milestone (still valid). The planner spec inherits this; D-2's measurement runs as planned (despite being decoupled from the dispatch case-(β) finding) because the floor is now consumed by the runtime dispatcher, not the codegen-time toggle.
- **`should_emit_tier_b` at codegen layer**: the codegen-side toggle becomes `config.segment_masked` (planning signal — emit the Tier-B-on PTX if `segment_masked`, regardless of seq_len). The runtime dispatcher applies the seq_len floor at launch time. This rehabilitates the dispatch spec's case-(β-ii) by relocating the seq_len check from codegen to launch.
- **FFI signature extension**: `nsl_flash_attention_csha` gains `tier_b_ptx_ptr: i64, tier_b_name_ptr: i64` (and equivalent for backward + the 6 other launch entry points). Existing call sites pass `0, 0` to preserve no-op behavior.
- **PR #169 kernel surface remains unchanged.** Only the launch path and the codegen wrapper that emits both PTX blobs are touched.

## Cross-references

- Dispatch design spec: [`2026-05-14-pca-tier-b-dispatch-design.md`](2026-05-14-pca-tier-b-dispatch-design.md) §14 amendment (the V-dispatch-integration outcome that this verification answers a sub-question of).
- D-1 findings: [`2026-05-14-tier-b-dispatch-integration-findings.md`](2026-05-14-tier-b-dispatch-integration-findings.md) — caller-context verification that produced the case-(β) finding.
- Original Tier B design: `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` §11 — deferred-extension queue.
- Code touched (read-only): `crates/nsl-codegen/src/flash_attention.rs:142` (FlashAttentionConfig); `crates/nsl-runtime/src/flash_attention.rs:373` (nsl_flash_attention_csha dispatcher); `crates/nsl-codegen/src/pca_tilerange.rs:39` (compute_range_table_bytes).
- IR-013 (institutional-rules.md) — V-planner-options is the **second instance** of external-caller-context verification (first: V-dispatch-integration). Reinforces IR-013's narrow framing as a load-bearing institutional rule.

## Budget

~35 minutes (3 sub-investigations × ~10 min each + ~5 min findings doc structure). Matches V-dispatch-integration's 30-45 min budget.
