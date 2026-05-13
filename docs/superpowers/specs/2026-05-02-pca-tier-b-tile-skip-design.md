# PCA Tier B — Tile-Skip (runtime per-tile segment-range predicate)

## Revision Changelog

- **2026-05-12** — §3.1, §3.4, §6.3 revised after re-brainstorm; cites probe findings doc + plan v2; see `2026-05-12-pca-tier-b-revision-design.md` for the brainstorm output and the institutional-rules citations (IR-004, IR-005, IR-006, IR-007). Original §3.1/§3.4/§6.3 preserved in git history.
- **2026-05-02** — Initial spec; execution paused after 6 prerequisite commits (PR #138) due to §3.1 inline `.shared` decl incompatibility with Blackwell sm_120.

**Status:** Design (spec) — execution paused
**Date:** 2026-05-02
**Owner:** bwiemz
**Paper:** `docs/research/CFTP.pdf` §2 ("PCA — Packed-Context Attention"), specifically the "Tile-skip optimisation" sub-section
**Companion milestone:** PCA Tier A — shipped 2026-04-21 (PRs #78, #105, #109)
**Worktree / branch (TBD):** `feat/pca-tier-b`
**Precedent:** PCA Tier A (segment-mask kernel integration, ptxas hex-verification, fixture matrix); CSHA Tier C (forward+backward bundled landing); WRGA B.3.1 (shared-helper discipline); AWQ pre-scan ↔ in-compile parity test (estimator/runtime hard-equality discipline)

---

## 1. Problem

PCA Tier A eliminated the `S × S` block-diagonal attention mask by replacing it with a compact `segment_ids: [B, S] u16` tensor and a per-element predicate that masks `S[i, j]` when `segment_ids[i] != segment_ids[j]`. The mask tensor is gone; the HBM cost is gone. **The wasted FLOPs are not.**

For a packed sequence of three roughly-equal documents, ~67% of `(q_pos, k_pos)` cells across the `S × S` matrix are cross-document. Tier A's predicate fires on every cell, masks the cross-document ones to `-inf`, and contributes 0 to the softmax. **The QK^T matmul still computes those scores.** The masmul work that produces a value subsequently masked to `-inf` is wasted.

PCA Tier B addresses this at tile granularity: when an entire `(q_tile, kv_tile)` pair is fully cross-document, the kernel skips the matmul, the softmax accumulator update, and the PV reduction for that pair entirely. For typical 3-document packing with 64-token tiles, ~55% of tile pairs are fully cross-document and can be skipped — the figure cited in the CFTP paper.

The mechanism: a runtime per-tile segment-range predicate. The kernel preamble computes `(min_segment, max_segment)` for each Q tile and each KV tile from segment_ids already in SMEM. The KV-tile loop tests range disjointness before launching matmul. Disjoint ranges `→` skip; overlapping ranges `→` keep, with Tier A's per-cell mask handling cross-segment cells inside.

## 2. Scope and dependency

### 2.1 What Tier B ships

- New shared module `crates/nsl-codegen/src/pca_tilerange.rs` (~250 LOC). Emits the range-table preamble (parallel reduction over segment_ids) and the predicate snippet. Consumed by both the forward and backward FA2 phase emitters.
- Forward path integration: `flash_attention_v2/phases/forward/s_compute.rs` calls `pca_tilerange::emit_predicate` at the top of the KV-tile loop. Skipped tiles bypass the QK^T matmul, the softmax stat update, and the PV reduction.
- Backward path integration: `flash_attention_v2/phases/backward/ds_compute.rs` calls the same predicate. Skipped tiles bypass the dS computation, the dQ/dK accumulators, and the dV accumulator.
- Compile-time estimator parity: `pca_tileskip.rs::build` is repurposed as the *reference* skip-mask producer for `nsl check --training-report` and the parity test. Hard 0-tile equality enforced via kernel-side debug instrumentation.
- Test infrastructure: kernel-side per-tile skip-decision array (gated behind a debug build configuration) for M3 skip-mask parity.
- Composition with Tier A: Tier B *filters* tile pairs; Tier A masks cells inside surviving tiles. Verified end-to-end on the standard 3-document fixture and one skewed-packing fixture.

### 2.2 What Tier B does not ship

The CFTP paper's "PCA" section lists four features. Tier A shipped one (segment-ID kernel). Tier B ships the second (tile-skip). The remaining two are deferred:

| Feature | Deferred to | Trigger / consumer |
| --- | --- | --- |
| **RoPE position-reset fusion** | Tier C | Long-context packed-sequence eval shows position-encoding drift, OR fused RoPE (CSHA Level 1) needs document-aware position offsets. `pca_rope.rs` exists today as orphaned-pending. |
| **Per-document CTA scheduling** | Tier D | Profile shows attention kernel launch overhead exceeds Y% of attention wall time at small batches. `PerDocumentCta` strategy in `pca_detect.rs` exists today as orphaned-pending. |
| **CE separator skip** | Orthogonal milestone | Fused linear cross-entropy with separator-token skip — touches LM head, not attention. Out of PCA tier scope. |

### 2.3 Hard dependency on Tier A

Tier A must be merged before Tier B. The dependencies:

- **G1's gate** (always-on with `segment_masked = true`) piggybacks on Tier A's `FlashAttentionConfig::segment_masked` flag.
- **M1's baseline** (bit-exact match between Tier-B-on and Tier-A-only) requires Tier A to be the comparison baseline.
- **SMEM allocation** for segment_ids is set up by Tier A's kernel preamble (`pca_segment::SegmentResidency::Shared`); Tier B reads from the same SMEM region.

If Tier A is delayed, Tier B's gates need reformulation against the pre-Tier-A baseline (additional work, not in v1).

### 2.4 Tier B does not replace Tier A

This is load-bearing and worth pinning explicitly to forestall future "drop Tier A now we have Tier B" optimization passes:

> Tier B's range-check filters which tile pairs the kernel processes. Tier A's per-element segment_mask determines per-cell validity inside surviving tiles. **For typical packed batches with multiple documents per sequence and tiles smaller than docs, most surviving tiles are boundary tiles where the per-cell mask determines validity.** The per-element mask remains the load-bearing per-cell correctness mechanism; Tier B reduces the tile-pair count it processes. Removing Tier A would silently produce wrong outputs on boundary tiles.

## 3. Architecture

### 3.1 Runtime per-tile segment-range table

Revised 2026-05-12 per IR-005, IR-007; see `2026-05-12-pca-tier-b-revision-design.md` §4 for the brainstorm rationale.

#### 3.1.1 Emission shape — per-phase runtime loop

Two phases per kernel: one for q-tiles, one for kv-tiles. Both phases have identical structure parameterized by phase tag.

```ptx
    // ─── PCA Tier B: q-phase range-table reduction ───
    .reg .u32 %r_tile_q_TILERANGE;
    .reg .u16 %rs_min_q_TILERANGE, %rs_max_q_TILERANGE;
    .reg .pred %p_done_q_TILERANGE, %p_lane_zero;

    mov.u32 %r_tile_q_TILERANGE, 0;          // unconditional init — uniform across warp
LOOP_Q_TILERANGE:
    // [body: per-tile reduction parameterized by %r_tile_q_TILERANGE]
    //   1. Lane L loads segment_ids[%r_tile_q * block_q + L] (and +L+32 if block_q > warp_size)
    //   2. Lane-local min/max into %rs_min_q, %rs_max_q
    //   3. Five butterfly steps: shfl.sync.bfly + min.u16/max.u16 (warp-shuffle reduction)
    //   4. Lane-0 predicated store to range_table_addrs.qtile_{min,max} + 2 * %r_tile_q

    add.u32 %r_tile_q_TILERANGE, %r_tile_q_TILERANGE, 1;     // unconditional increment
    setp.lt.u32 %p_done_q_TILERANGE, %r_tile_q_TILERANGE, NUM_Q_TILES;
    @%p_done_q_TILERANGE bra LOOP_Q_TILERANGE;               // compiles to BRA.U
```

Followed by an identical block for kv-phase. Final `bar.sync 0` so all warps see the completed range table before the main kv-tile loop reads from it.

**Total PTX size: ~120 bytes per phase regardless of `num_*_tiles`.** Compile-time unroll would have produced `~12-14 instructions × num_tiles` linear growth (~6.7 KB at 16 K seq, ~27 KB at 64 K). Runtime loop scales to long-context Tier B-extended without re-emission work.

#### 3.1.2 Warp-uniformity discipline (load-bearing)

For ptxas to compile the loop branch as `BRA.U` rather than `BRA` (per-thread predicated):

1. `%r_tile_*` is `.reg .u32` — not `.u64`, not signed, not derived from `%tid.x` or any thread-divergent source.
2. `mov.u32 %r_tile_*, 0` is the only initializer; no per-thread initialization paths.
3. `add.u32 %r_tile_*, %r_tile_*, 1` is unconditional (not predicated on `%tid` or any thread-divergent predicate).
4. `setp.lt.u32 %p_done_*, %r_tile_*, NUM_*_TILES` compares two uniform values (uniform register vs. compile-time literal).
5. The branch `@%p_done_* bra LOOP_*` consumes a uniform predicate.

**Anti-pattern: uniformity-through-load.** `%r_tile_*` must remain register-resident throughout the loop. Loading from `.local` (thread-local) or `.shared` (warp-or-CTA-local) into `%r_tile_*` breaks ptxas's uniformity tracking even if the loaded value happens to be uniform across threads. The only initialization is `mov.u32` from a literal; the only update is `add.u32 %r_tile_*, %r_tile_*, 1`. If a future optimization wants to vary the tile-count source, the SASS verification gate catches the regression (BRA replaces BRA.U).

This is IR-007 applied: PTX emission discipline is pinned at the instruction-pattern level (specific register class, specific operand shape, specific predicate construction) because the ptxas → SASS pipeline is pattern-recognition-driven.

#### 3.1.3 Lane-0 store pattern (predicated execution, not divergent branch)

The single-lane store uses PTX predicated execution, not a thread-divergent branch:

```ptx
.reg .pred %p_lane_zero;
setp.eq.u32 %p_lane_zero, %lane_id, 0;
@%p_lane_zero st.shared.u16 [%addr_min], %rs_min_q_TILERANGE;
@%p_lane_zero st.shared.u16 [%addr_max], %rs_max_q_TILERANGE;
```

The predicate is thread-divergent (only lane 0 evaluates to true), but the predicated store is a single instruction that ptxas handles without affecting the surrounding warp-uniform loop structure. A thread-divergent branch (`@%p_lane_zero bra STORE_BLOCK; ...; STORE_BLOCK:`) would also be correct but emits divergent SASS — measurably worse and a different code shape than this spec assumes.

#### 3.1.4 Inner reduction body

Fixed-instruction-count warp-shuffle butterfly (no nested loop):

1. Lane L loads `segment_ids[%r_tile_q * block_q + L]` into a u16 register.
2. If `block_q > warp_size` (e.g., block_q = 64 = 2 × warp_size on most configs), also load `+L+warp_size` and merge into lane-local min/max.
3. Five butterfly steps with offsets `{16, 8, 4, 2, 1}` using `shfl.sync.bfly` + `min.u16` / `max.u16` ops.
4. Lane-0 predicated store (per §3.1.3) to `qtile_seg_min[%r_tile_q]` / `qtile_seg_max[%r_tile_q]`.

Total inner-body instructions: ~12-14 per iteration. Only the outer tile-iteration is the runtime loop; the inner butterfly is unrolled (5 steps × 2 ops + load + store).

#### 3.1.5 Address arithmetic into the range table

Per §3.4's alignment guarantee, `range_table_addrs.qtile_{min,max}` are 2-byte aligned. The per-iteration address computation:

```text
%addr_min = range_table_addrs.qtile_min + 2 * %r_tile_q_TILERANGE
%addr_max = range_table_addrs.qtile_max + 2 * %r_tile_q_TILERANGE
```

Both offsets are uniform across the warp (same on every thread); only lane 0 actually performs the store. The shift-and-add is two instructions per iteration — negligible overhead.

#### 3.1.6 SASS verification gate

`pca_sass_byte_identity` extracts the SASS for each phase's loop branch and asserts:

- The q-phase loop branch is `BRA.U` (or per-architecture equivalent).
- The kv-phase loop branch is `BRA.U`.
- On sm_90 / sm_120: range-table min/max stores use `UR<n>` operands (direct uniform-class verification).
- On sm_80 / sm_86: BRA.U at the loop branch is the available proxy — necessary for uniformity, but not sufficient to verify the inner store operands are uniform-resident. Accepted v1 limitation.

If the SASS check fails, the fallback is to investigate ptxas's uniformity-recognition heuristic and adjust the emission to match — not paper over with a register-allocator hint until the root cause is understood.

#### 3.1.7 Hybrid-rejection paragraph (cite IR-005)

> The hybrid emission split (compile-time-unroll at small `num_tiles`, runtime loop above a threshold) was considered and rejected. Per IR-005, bifurcation of emission paths for the same logical operation requires measurement-driven justification, not architectural preference. At v1 scale, the slight runtime cost (~4 instructions per tile: branch + counter increment + setp) is dwarfed by the matmul work being skipped or computed; the hybrid's "small-N performance win" is not a measurable concern. Uniform emission across the config matrix is a v1 architectural property. Bifurcation deferred to Phase 1.5 with explicit measurement-driven justification if it becomes warranted. Same discipline as FA-2 v2 Tier B.1's "no producer/consumer split in v1."

### 3.2 Conservative-skip semantic — load-bearing

The KV-tile loop predicate, emitted at the top of `s_compute.rs::kv_tile_iter` and `ds_compute.rs::kv_tile_iter`:

```c
if (qtile_seg_max[qt] < kvtile_seg_min[kvt] ||
    qtile_seg_min[qt] > kvtile_seg_max[kvt]) {
    continue;  // tile pair fully cross-document — skip matmul + softmax + PV
}
// fall through: tile pair has at least one within-segment cell.
// Tier A's per-element segment_mask masks the cross-segment cells inside.
```

**Range check is conservative, not exact.** The predicate skips a tile pair only when the segment ranges are strictly disjoint. Tile pairs whose ranges *overlap* are kept, even if the majority of cells inside are cross-segment. This is necessary because *some* cells inside are within-segment and need to contribute to attention; the per-element segment_mask masks the rest.

Future optimizations that "tighten" this check (e.g., "skip if not all cells within-segment") would silently drop valid attention computations and produce wrong outputs. The conservative semantic is the correct one and must be preserved.

Worked example. Q tile spans tokens 64–127. Token 64–99 is in document A (segment 0); token 100–127 is in document B (segment 1). So `qtile_seg_min = 0, qtile_seg_max = 1`.

| KV tile | range | predicate | outcome |
| --- | --- | --- | --- |
| `(min=0, max=0)` (entirely in doc A) | `1 < 0 \|\| 0 > 0` = false | keep | correct — doc-A KV cells contribute |
| `(min=1, max=1)` (entirely in doc B) | `1 < 1 \|\| 0 > 1` = false | keep | correct — doc-B KV cells contribute |
| `(min=2, max=2)` (entirely in doc C) | `1 < 2 \|\| 0 > 2` = true | skip | correct — no overlap |

In the kept cases, Tier A's per-cell mask zeroes out the cross-segment cells inside the tile.

### 3.3 SMEM cost — formula and v1 bound

Range table size formula:

```
range_table_bytes = 2 × (num_q_tiles + num_kv_tiles) × sizeof(seg_id_t)
where:
    num_q_tiles  = ceil(seq_len / block_q)
    num_kv_tiles = ceil(seq_len / block_kv)
```

`block_q` and `block_kv` are FA2 variant-dependent (head_dim drives them; the variant name encodes both — `q{block_q}_kv{block_kv}`). The current NSL convention is `seg_id_t = u16` (2 bytes), supporting up to 65 535 segments per batch.

**Examples (illustrative — assume `block_q = block_kv = 64`):**

| seq_len | num_q_tiles | num_kv_tiles | range table |
| --- | --- | --- | --- |
| 2 K | 32 | 32 | 256 B |
| 4 K | 64 | 64 | 512 B |
| 16 K | 256 | 256 | 2 KB |
| 64 K | 1 024 | 1 024 | 8 KB |
| 1 M | 16 384 | 16 384 | 128 KB |

These numbers shift for other block configurations. With `block_q = 128, block_kv = 128` (head_dim = 64): same sequence lengths give half the table size. With `block_q = 32, block_kv = 64` (head_dim = 256, asymmetric): a 16K sequence gives `2×512×2 + 2×256×2 = 3 KB`.

**v1 bound.** Tier B v1 supports the full FA2 variant matrix **for which the computed range-table size is ≤ 8 KB**. The codegen emitter computes this size per-variant. For variants exceeding the threshold, the emitter **falls back to Tier A only** (segment-mask predicate, no tile-skip preamble, no range-check) and emits a compile-time warning naming the variant and the computed table size. The kernel still emits and runs correctly; it just doesn't get Tier B's FLOP savings.

Beyond 8 KB, the range table needs to move out of SMEM. Two options for a future "Tier B-extended":

- *(ii) Per-warp uniform registers via warp-shuffle communication.* No SMEM cost; harder to implement; constrained by register pressure.
- *(iii) Global memory, computed once per batch.* Constant SMEM cost; one extra HBM read per tile pair (well-cached but not free).

Both are out of v1. The 8 KB threshold is set conservatively against typical SMEM budgets (96 KB on H100, 100 KB on A100, 100 KB on RTX 5070 Ti) leaving ample room for Q/K/V tiles, softmax stats, and segment_ids.

### 3.4 Tier A residency interaction — Shared required, with explicit budget bump

Tier A's `pca_segment::plan_kernel` selects between `SegmentResidency::Shared` (segment_ids fit in SMEM) and `SegmentResidency::Streamed` (segment_ids loaded per-tile from HBM). The range-table preamble in §3.1 needs the **full** segment_ids array in SMEM at preamble time to compute per-tile (min, max) reductions.

**Tier B v1 requires `SegmentResidency::Shared`.** For variants where Tier A would select Streamed residency (sequences too long for the segment-ID SMEM budget), Tier B falls back to Tier A only with a compile-time warning, the same fallback behavior as the > 8 KB range-table case in §3.3.

#### 3.4.1 Tier A current state and prerequisite

Tier A defines `DEFAULT_SMEM_SEGMENT_BUDGET = 4096` in [pca_segment.rs:53](../../crates/nsl-codegen/src/pca_segment.rs#L53) as a **module-level const with no per-variant override mechanism**. With u16 segment_ids, this caps Shared residency at seq_len ≤ 2048.

Every fixture in §4.4's M3 parity matrix is ≥ 4 K. Without a budget bump, every fixture would fall back to the Tier-A-only path and the acceptance gates would silently grade the wrong thing.

**Resolution.** Tier B v1's **Commit 0** raises `DEFAULT_SMEM_SEGMENT_BUDGET` to **32 KB** (one-line const change in `pca_segment.rs`). This allows Shared residency for seq_len ≤ 16 K with u16 segment_ids — covers the entire §4.4 fixture matrix. The bump is a Tier A behavioral change (segment_ids residency boundary moves outward) and is the first commit in Tier B's PR, justified by:

- The bumped budget is read only in the segment_masked path (non-segment_masked variants don't allocate the buffer).
- For segment_masked workloads at seq_len > 2 K, the per-CTA SMEM cost was already paid as HBM streaming traffic; making it resident is a tradeoff (less HBM traffic, more SMEM).
- The bump's safety on existing Tier A fixtures is verified before subsequent Tier B commits land (M1's bit-exact match against pre-bump baseline must pass for the bump alone).

A future "Tier A v1.1 multi-budget API" — adding per-variant override to `plan_kernel`'s signature — is out of v1 scope. The simple const bump suffices for the §4.4 fixture range.

Revised 2026-05-12 per IR-004; see `2026-05-12-pca-tier-b-revision-design.md` §3 for the brainstorm rationale.

#### 3.4.2 Surface in `smem_layout.rs` — one new public symbol

```rust
/// PCA Tier B range table — single tail offset into extern shmem.
///
/// Returns the byte offset where Tier B's four sub-tables (qtile_min,
/// qtile_max, kvtile_min, kvtile_max) start. Caller is responsible for
/// adding `range_table_bytes(...)` to its dynamic-SMEM launch parameter.
///
/// Returned offset is `align_up(2)` for u16 range-table slot alignment.
pub fn tier_b_range_table_offset(config: &FlashAttentionConfig) -> u64 {
    let base = backward_total_bytes(config) + seg_overhead(config);
    align_up(base, 2)
}
```

Sits after the existing `kv_offset`, `sp_offset`, etc. accessors. Returned offset is the tail of the extern shmem allocation, past everything else, aligned to 2 bytes for u16 slot loads.

**Alignment guarantee is owned upstream.** `pca_tilerange::range_table_addrs` assumes its `base` parameter is 2-byte aligned; it does not re-check. Same discipline as `kv_offset` / `sp_offset` — offset-computation site owns alignment, consumers assume ready-to-use values. (IR-004.)

#### 3.4.3 Internal sub-layout in `pca_tilerange.rs`

```rust
#[derive(Debug, Clone, Copy)]
pub struct RangeTableAddrs {
    pub qtile_min:  u64,
    pub qtile_max:  u64,
    pub kvtile_min: u64,
    pub kvtile_max: u64,
}

/// Constructor for the four sub-table offsets within Tier B's
/// range-table region. Caller obtains `base` from
/// `smem_layout::tier_b_range_table_offset(config)`; `num_q_tiles` and
/// `num_kv_tiles` from `pca_tile_config::num_tiles(seq_len, block_*)`.
pub fn range_table_addrs(
    base: u64,
    num_q_tiles: u32,
    num_kv_tiles: u32,
) -> RangeTableAddrs {
    let q_bytes  = num_q_tiles  as u64 * 2;
    let kv_bytes = num_kv_tiles as u64 * 2;
    RangeTableAddrs {
        qtile_min:  base,
        qtile_max:  base + q_bytes,
        kvtile_min: base + 2 * q_bytes,
        kvtile_max: base + 2 * q_bytes + kv_bytes,
    }
}
```

**Visibility.** `RangeTableAddrs` and `range_table_addrs` are `pub` at the module level. Three consumers in the same crate: forward preamble emitter (`emit_range_table_preamble` for forward kernel), backward preamble emitter (Tier B.2, when it lands), and parity test bit-equivalence comparison against `pca_tileskip::build`.

"Tier B owns its sub-structure" applies at the crate-boundary level (`smem_layout.rs` knows about one Tier B region), not at the file-boundary level (within Tier B's modules, the sub-layout is shared infrastructure).

#### 3.4.4 Shared tile-count source of truth

New module `crates/nsl-codegen/src/pca_tile_config.rs`:

```rust
/// Compute the number of tiles for a given (seq_len, block_size).
/// Ceiling division — last tile may be partial.
///
/// Single source of truth for tile counts across the codebase:
///   - pca_tilerange::range_table_addrs callers.
///   - FA2 kernel tile-loop emission (forward + backward).
///   - M3 parity test fixture construction.
pub fn num_tiles(seq_len: u32, block_size: u32) -> u32 {
    seq_len.div_ceil(block_size)
}
```

#### 3.4.5 `num_tiles` identity unit test

The "asserted identical" claim is verifiable, not aspirational. Test structure:

For every `(seq_len, block_size)` tuple in the supported config matrix:

1. Call `num_tiles(seq_len, block_size)` from Rust.
2. Generate the FA2 kernel PTX for that configuration; parse the tile-loop bound from the emitted `setp.lt.u32 %p, %r_tile, NUM_TILES;` instruction.
3. Compute the same value at the `range_table_addrs` consumer site (via direct Rust call to `pca_tile_config::num_tiles(seq_len, config.block_q)`).

All three values must equal. Test failure surface identifies which-of-three site disagrees with which other: Rust-formula vs emitted-PTX disagreement (kernel emission added a `+1` or `-1` adjustment), Rust-formula vs consumer-computation disagreement (consumer using the wrong block-size operand), or emitted-PTX vs consumer-computation disagreement (kernel and range-table allocator use different tile-count paths).

Failure localizable without bisecting kernel emission code. Same diagnostic-precision discipline as IR-006 applied at the unit-test level.

#### 3.4.6 Launch-side accounting (no-op guarantee)

`shared_mem_bytes_v2_backward(config)` and the corresponding forward `shared_mem_bytes_v2(config)` gain a conditional Tier B contribution:

```rust
let tier_b_bytes = if should_emit_tier_b(config, seq_len, residency) {
    tier_b_range_table_bytes(config)
} else {
    0
};
let shared_mem_total = existing_terms + tier_b_bytes;
```

**No-op guarantee.** When Tier B is not emitted (config below threshold, sm < 80, etc.), the range-table-bytes contribution is exactly zero. Pre-Tier-B configurations have byte-identical SMEM layout to post-Tier-B. Snapshot tests for non-Tier-B kernels are unaffected; only the Tier B kernel's snapshot is new.

#### 3.4.7 Findings-doc dependency citation

The revised §3.4's correctness depends on the SMEM probe's outcome. The struct-based sub-layout assumes tail-embed is safe.

> Probe finding `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md` selected an outcome row from §2.4's decision matrix. The §3.4 sub-layout above is correct for that outcome. Probe re-runs that shift the outcome may require §3.4 revisiting; the findings doc's "Re-run log" is the audit trail.

### 3.5 Phase integration

Forward path. The KV-tile loop currently sits in `flash_attention_v2/phases/forward/s_compute.rs::emit`, called once per `q_tile_iter` from the outer loop in `mod.rs`. Tier B inserts the range-check predicate at the top of the KV-tile body, before the existing K-tile load and QK matmul:

```
for kv_tile in 0..num_kv_tiles:
    [Tier B] if range_disjoint(qt, kv_tile): continue
    [Tier A + Tier B] load K tile from HBM to SMEM
    [Tier A + Tier B] QK matmul → S[warp_id, k]
    [Tier A] segment_mask + causal mask: S[i, j] = -inf if cross-seg or j > i
    softmax stats update
    load V tile, PV accumulator update
```

Backward path. The KV-tile loop in `flash_attention_v2/phases/backward/ds_compute.rs` mirrors the forward path. Tier B's predicate fires identically; skipped tiles bypass the dS computation, dQ/dK accumulators, and dV accumulator.

The mathematical justification for symmetric forward/backward skip: backward gradients are zero on cells the forward set to `-inf` (since `dL/dS = P × (something)` and `P = 0` for masked cells). Skipping the same tiles in backward is correct.

### 3.6 Module placement — shared-helper discipline

Following the WRGA B.3.1 / Tier A pattern:

- New module: `crates/nsl-codegen/src/pca_tilerange.rs`. Exposes:
  - `emit_range_table_preamble(ptx: &mut String, config: &FlashAttentionConfig, seq_len: u64)` — emits the parallel-reduction code that computes `qtile_seg_{min,max}` and `kvtile_seg_{min,max}` from segment_ids in SMEM.
  - `emit_skip_predicate(ptx: &mut String, qt: &str, kvt: &str, on_skip: &str)` — emits the disjoint-range check; `on_skip` is the PTX label or directive to branch to (e.g., `continue` equivalent).
  - `emit_skip_decision_writeback(ptx: &mut String, qt: &str, kvt: &str, decision: &str)` — gated behind `cfg!(debug_kernel_instrumentation)`; writes the skip decision to a per-launch device buffer for M3 parity verification (see §4.3).
- Repurposed module: `crates/nsl-codegen/src/pca_tileskip.rs`. The existing `TileSkipMap` struct stays in place but is no longer consumed by the kernel emitter. Its sole consumers become:
  - `nsl-codegen/src/training_report.rs` — for `nsl check --training-report` skip-rate estimation.
  - The M3 parity test — as the reference skip-mask producer.
- Untouched: `pca_segment.rs`, `pca_detect.rs`, `pca_rope.rs`, `flash_attention_v2/phases/segment_mask.rs`. Tier B does not modify Tier A's segment-mask emission.

## 4. Estimator/runtime parity

### 4.1 The discipline

`pca_tileskip.rs::build` and the runtime `pca_tilerange` predicate **must produce bitwise-identical skip masks on identical layouts**. Skip-count agreement without skip-mask agreement is insufficient — same count of skipped tiles in different positions is exactly the failure mode parity catches.

This pattern is reused from the AWQ pre-scan ↔ in-compile differential test: the compile-time estimator and the runtime mechanism are two implementations of the same predicate, and drift between them silently makes `nsl check --training-report` numbers untrustworthy.

### 4.2 Layout-equivalence definition — pinned

Hard 0-tile equality holds when the estimator and kernel are given the same inputs at the same fidelity. The estimator must accept the same packing description the kernel sees:

```
LayoutDescription = (
    num_docs,
    doc_lengths,
    doc_offsets_within_sequence,
    padding_token_locations,
)
```

Estimator paths that approximate from a less complete description (e.g., just `num_docs` and `doc_lengths`, assuming contiguous packing) are out of scope for parity. Such paths must be flagged in `--training-report` output as "estimated assuming contiguous packing" so users know the number is not a hard contract.

### 4.3 Kernel-side instrumentation for M3

M3 verifies skip-mask equality, not just skip-count equality. This requires the kernel to emit per-tile skip decisions to an observable buffer.

#### 4.3.1 Buffer shape and allocation

The buffer is allocated by the kernel launcher when `debug_kernel_instrumentation` is set, with full 4-D shape and explicit memory layout:

```
skip_decisions: [batch, head, num_q_tiles, num_kv_tiles] : u8
flat layout (row-major):
    skip_decisions[b][h][qt][kvt]
        = decisions_buf[((b × num_heads + h) × num_q_tiles + qt) × num_kv_tiles + kvt]
slot value:
    1 → range-disjoint (skipped),  0 → range-overlap (kept)
```

Total size: `batch × num_heads × num_q_tiles × num_kv_tiles` bytes.

Worked size estimate for the §4.4 fixtures (assuming `block_q = block_kv = 64`, `batch = 2`, `num_heads = 16`):

| Fixture | seq_len | num_tiles | buffer size |
| --- | --- | --- | --- |
| `standard_3doc` | 4 K | 64 × 64 | 128 KB |
| `long_seq_5doc` | 16 K | 256 × 256 | 2 MB |
| `single_doc` | 4 K | 64 × 64 | 128 KB |

Buffer is HBM-resident (not SMEM), allocated by the kernel launcher pre-launch and freed post-readback. Test-only memory; never allocated in production builds because the writeback code path is `cfg!`-gated out at codegen time.

#### 4.3.2 Write semantics — lane 0 of the owning warp

FA2's tile-warp mapping assigns each `(qt, kvt)` tile pair to exactly one warp within the CTA. The **lane-0 of that owning warp** writes the skip decision exactly once, before the predicate's `continue`. No CTA-level synchronization; no race between warps because each (qt, kvt) slot has exactly one owning warp.

The CTA's `(batch, head)` context is already threaded through FA2's existing kernel parameters (`%batch_idx`, `%head_idx` in the PTX). The writeback uses these to compute the slot's flat offset:

```
flat_offset = ((%batch_idx × num_heads + %head_idx) × num_q_tiles + qt) × num_kv_tiles + kvt;
if (lane == 0 && warp == owning_warp(qt, kvt)):
    decisions_buf[flat_offset] = is_disjoint ? 1 : 0;
```

`owning_warp(qt, kvt)` is the existing FA2 warp-tile assignment function — Tier B's writeback piggybacks on it rather than introducing a separate mapping.

#### 4.3.3 Cost and threading

The instrumentation is a real piece of work — allocating the buffer per launch, threading the gate through the emitter, reading back without serializing GPU work, and confirming the warp-owning-pair mapping matches FA2's actual tile distribution. Budget ~0.5–1 day for its own commit in the implementation plan.

The cost is justified by the failure mode it catches: a "right count, wrong tiles" bug in the kernel-side range-table reduction would pass M1's bit-exact check on small fixtures (where the wrong tiles happen to be sparse) and fail catastrophically on production workloads where attention magnitudes differ across positions.

A misaligned buffer layout (e.g., omitting the `(batch, head)` dimensions) would cause different CTAs to overwrite the same slots — the parity test would report "skip mask matches!" while actually misverifying. Pinning the full 4-D shape forecloses this failure mode.

### 4.4 Test fixture matrix

M3 parity tests run across **all** packing fixtures, not just the standard 3-document case. The minimum required matrix:

| Fixture | seq_len | docs | skew | purpose |
| --- | --- | --- | --- | --- |
| `standard_3doc` | 4 K | 3 × ~1.3 K | uniform | paper's reference case |
| `long_seq_5doc` | 16 K | 5 × ~3.2 K | uniform | tests range-table size near v1 bound |
| `skewed_packing` | 4 K | 1 × 3 K + 3 × ~333 | skewed | unequal documents |
| `boundary_dense` | 4 K | 16 × 256 | uniform | many small docs (lots of boundary tiles) |
| `single_doc` | 4 K | 1 × 4 K | n/a | M6 worst case (no skips possible) |
| `tail_padding` | 4 K | 2 × 1 K + 2 K padding | uniform | exercises padding-token handling |

Failure diagnostics must identify the diverging fixture. Adding a fixture variant requires extending both the estimator and the runtime test harness.

## 5. Gating — G1 always-on

Tier B's range-check fires whenever Tier A's `segment_masked = true`. No decorator opt-in, no statistic-based gate, no profile-driven activation threshold.

### 5.1 Rationale

The range-check is correctness-preserving for any layout (it only skips strictly disjoint tile pairs; the per-element mask handles everything else). The cost is bounded (≤ 8 KB SMEM at v1 bound, one warp-uniform branch per tile pair). There is no workload class where Tier B is wrong, and no workload class where its cost meaningfully exceeds its benefit. With both axes answered "no," gating is pure overhead.

### 5.2 Anti-treadmill principle

Gates accumulate maintenance debt. Once an opt-in surface ships you commit to:

- Documenting the opt-in.
- Maintaining the heuristic that decides when to enable.
- Eventually graduating from opt-in to default (its own migration).
- Absorbing the "should I enable Tier B?" support questions.

For correctness-preserving optimizations with bounded cost, this overhead has no offsetting benefit. **Institutional rule (pinned for future PCA tiers and other compiler optimizations):** the bar for "should this be gated?" is "is there a real workload class where this is wrong, OR a workload class where the cost meaningfully exceeds the benefit?" If the answer is no to both, ship always-on. Reserve gates for optimizations that *aren't* correctness-preserving or whose cost is unbounded.

## 6. ptxas / SASS verification

Following the WRGA B.3.1 / Tier A precedent: every Tier B PTX emission goes through `ptxas` and `cuobjdump` checks before the spec considers the variant verified.

### 6.1 Range-table preamble

- Compiles to ≤ N SASS instructions per warp (target N established during implementation; typically 8–16 for a min/max reduction over a 64-token tile).
- Zero spills from the reduction.
- Final (min, max) values land in uniform registers, not per-thread registers.

### 6.2 Skip predicate

- Compiles to **one** comparison + **one** branch.
- The branch is warp-uniform — emitted as `BRA.U` (or equivalent uniform-branch SASS instruction on the target architecture), **not** a per-thread predicated branch (`@P0 BRA`).
- Requirement: the range tables are loaded into uniform registers before the predicate. If they're loaded per-thread, the branch becomes per-thread predicated and warps diverge on tile-pair boundaries — a silent perf regression that defeats the optimization.
- ptxas verification grep target: confirm `BRA.U` appears at the predicted offset for every Tier B variant.

### 6.3 Branch-uniformity verification — pinned

Revised 2026-05-12 per IR-006; see `2026-05-12-pca-tier-b-revision-design.md` §5 for the brainstorm rationale.

#### 6.3.1 Isolation-level test (catches PTX-string drift)

`crates/nsl-codegen/tests/pca_tier_b_preamble_isolation.rs`, ~20 LOC:

```rust
use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_tilerange::emit_range_table_preamble;

fn fa_4k_block64_seg_masked() -> FlashAttentionConfig { /* same fixture as skeleton tests */ }

#[test]
fn preamble_4k_block64_isolation_snapshot() {
    let mut ptx = String::new();
    let base_offset = 0xDEAD_BEEF; // sentinel — appears verbatim in emitted PTX
    emit_range_table_preamble(&mut ptx, &fa_4k_block64_seg_masked(), 4096, base_offset);
    insta::assert_snapshot!("preamble_4k_block64", ptx);
}
```

The sentinel `0xDEAD_BEEF` makes offset-consumption regressions loud: the snapshot shows `0xDEAD_BEEF` verbatim in the emitted arithmetic, so a regression where the emitter accidentally consumes the offset (e.g., `mov.u64 base, 0` instead of `mov.u64 base, BASE_LITERAL`) is immediately visible.

**Sentinel detection assumes register-loaded literal emission.** The detection works for two PTX emission shapes: (α) `mov.u64 %base, 0xDEADBEEF` — register-loaded literal; sentinel visible verbatim. (γ) `add.u64 %addr, %something, 0xDEADBEEF` — arithmetic operand; sentinel visible as instruction operand. The detection does **not** work for (β) `ld.param.u64 %base, [param_slot]` — `base_offset` passed as launch-time `.param`. **Required emission discipline:** the preamble emitter materializes `base_offset` as a register-loaded literal or arithmetic operand at PTX-emission time, not as a launch-time `.param`.

**Failure surface (alpha: PTX-string drift):** ~5-line PTX diff localized to Tier B's emission. Reviewer sees the exact instruction(s) that changed and decides accept (legitimate refactor) or reject (bug) immediately.

#### 6.3.2 Integration-level checks (catches SASS regressions)

Extended in three existing test surfaces:

**`pca_forward_kernel_snapshot` + `pca_backward_kernel_snapshot`** — gain Tier B-enabled config variants (e.g., `forward_kernel_segment_masked_tier_b_causal_64_64_64`). These already run real FA2 kernel PTX through ptxas + cuobjdump in CI; Tier B emission is part of the full PTX/SASS dump. Catches integration regressions where Tier B's preamble interacts incorrectly with surrounding FA2 code (scratch-register collision, SMEM offset overlap, etc.).

**`pca_sass_byte_identity`** — extended with assertions at Tier B's expected SASS offset:

```rust
let q_loop_branch = locate_sass_at_label(&sass, "LOOP_Q_TILERANGE");
assert!(
    q_loop_branch.contains("BRA.U") || q_loop_branch.contains("BRA.UNI"),
    "q-phase loop branch is divergent (BRA), not uniform (BRA.U):\n{q_loop_branch}"
);
// Same for kv-phase.
// Per-architecture: on sm_90/sm_120, additionally grep for UR<n> in min/max store operands.
let on_sm_90_or_120 = matches!(config.gpu_sm, 90 | 120);
if on_sm_90_or_120 {
    let store_lines = extract_sass_block(&sass, "qtile_seg_min_store", "qtile_seg_max_store");
    assert!(
        store_lines.contains("UR"),
        "range-table stores not using uniform-class registers on sm_{}:\n{store_lines}",
        config.gpu_sm
    );
}
// sm_80/sm_86 fall back to BRA.U-as-proxy per §3.1.6's accepted v1 limitation.
```

**Per-variant SASS baselines** at `tests/sass_baselines/<variant>_tier_b.txt` record cuobjdump instruction count + zero-spill-count for each Tier B-affected FA2 variant. Drift > ±2 fails CI. New variant ships its own baseline file in the same PR.

**Failure surface (beta: SASS regression):** test name identifies the specific regression class (`BRA.U`, uniform-register class, instruction count, spill count); SASS diff at the offending offset is the diagnostic.

#### 6.3.3 PR-coordination discipline

Any PR modifying Tier B's preamble emission updates all three test surfaces in the same commit. The PR description includes a checklist:

- [ ] `pca_tier_b_preamble_isolation` snapshot re-baselined; diff included in PR description.
- [ ] `pca_{forward,backward}_kernel_snapshot` snapshots re-baselined for each affected Tier B variant; per-variant diffs summarized.
- [ ] `pca_sass_byte_identity` assertions verified; any assertion-offset changes are explicit code edits with rationale.
- [ ] Per-variant SASS baseline files updated; instruction count + spill count match the new SASS.
- [ ] PR description summarizes the cascade: "PTX-text change at X → kernel SASS shifted at Y → uniform-register class verified at Z."

The cascade-summary line is the load-bearing audit point. If the three layers' changes don't form a coherent narrative (e.g., PTX-text change at the q-phase loop counter but SASS diff at an unrelated offset), one of the updates is wrong. Same discipline as WGGO Phase 2's "snapshot re-baseline review discipline" — snapshot acceptance is the load-bearing review moment, not a rubber-stamp.

#### 6.3.4 Deletion of `emit-pca-tier-b-preamble-harness` bin

The original plan's Task 3 Step 4 proposed a standalone `bin` target that emits stub kernel PTX containing only the Tier B preamble for isolated ptxas verification. This is deleted.

**Load-bearing reason: maintenance cost over time.** The harness would have required ongoing maintenance as FA2's register environment evolves. Every change to scratch register usage, every new CSHA hook, every shift in SMEM layout would require mirroring the change into the harness's mock environment. Either the mock drifts from production (silent test passes for wrong reasons) or the maintenance cost is paid continuously.

The hybrid approach has zero mock-environment maintenance: the isolation test uses the real preamble emitter; the kernel snapshot tests use the real FA2 emitter. Both verification surfaces are coupled to actual production code, not to a parallel mock.

#### 6.3.5 §6.3 satisfaction table

All four v1 ptxas-pass criteria satisfied:

| Criterion | Satisfied by |
| --- | --- |
| ptxas accepts on sm_120 + sm_80 | `pca_{forward,backward}_kernel_snapshot` tests already run ptxas; Tier B-enabled variants run alongside |
| Zero spills (`ptxas -v` output) | Per-variant baseline files at `tests/sass_baselines/<variant>_tier_b.txt` record + grep |
| `BRA.U` at skip predicate + loop branch | Extended `pca_sass_byte_identity` assertions at q-phase + kv-phase loop branches |
| Uniform-register placement | Direct on sm_90/sm_120 (`UR<n>` grep); BRA.U-as-proxy on sm_80/sm_86 |

Plus the isolation-level snapshot from §6.3.1 — strictly additive to §6.3; not required by spec text but justified by IR-006 (distinct failure modes warrant distinct test surfaces).

## 7. Acceptance gates

Tier B v1 cannot merge without all five gates passing. Variants that don't apply to a gate (e.g., M2 doesn't apply on a non-packed fixture) are explicitly enumerated as N/A in the merge PR.

### M1 — Numerical correctness

Bit-exact match within fp16 tolerance between Tier-B-on and Tier-A-only on every segment_masked test fixture (forward and backward). Tier B is a FLOP optimization, not a numerical change — output should be identical to within rounding noise.

Pass criterion: `max_abs_diff(out_tier_b, out_tier_a) < tier_a_baseline_tolerance` for forward outputs (`O`) and backward gradients (`dQ`, `dK`, `dV`) on each fixture in §4.4.

### M2 — FLOP reduction

≥ **30%** reduction in attention-kernel arithmetic operations on the standard packed-3-doc fixture, measured via Nsight Compute SASS-instruction-executed counters.

**Pinned formula.** FLOP count =

```
1 × smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
+ 2 × smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
+ 1 × smsp__sass_thread_inst_executed_op_fmul_pred_on.sum
```

Excluded: `op_fexp_pred_on.sum`, `op_flog_pred_on.sum`, and other special-function-unit (SFU) ops. Tier B reduces the matmul work done in QK^T and PV; SFU ops are softmax-internal and don't change with tile-skip. Including them would dilute the metric.

**Both sides of the comparison use the same formula.** Tier-B-on FLOP count and Tier-A-only FLOP count are measured with identical Nsight Compute invocations on identical fixtures; the threshold compares the ratio.

```
flop_reduction = 1 - (flops_tier_b / flops_tier_a)
pass criterion: flop_reduction ≥ 0.30
```

Wall-time win is **reported informationally but not gated on a specific number**. Rationale: FLOP reduction is a true property of the kernel; wall-time conflates "did Tier B work" with "is this workload FLOP-bound." A workload where attention is memory-bound rather than FLOP-bound will show smaller wall-time wins despite the FLOP reduction landing exactly as predicted; gating merge on wall-time conflates two different questions.

The 30% threshold corresponds to the lower bound of paper-claimed savings (paper claims ~50% on typical 3-doc packing; 30% leaves margin for the conservative-skip semantic and warp-uniform overhead).

### M3 — Estimator/runtime skip-mask parity

Hard 0-tile equality between `pca_tileskip::build`'s output and the kernel's actual skip-decision array (read back from device memory via the §4.3 instrumentation), across **every fixture in §4.4**.

This is a skip-*mask* check, not a skip-*count* check — matching positions, not just matching totals.

Failure diagnostics must identify which fixture diverged and at which tile-pair coordinates.

### M5 — ptxas SASS verification

All four checks in §6.3 pass for every Tier B-affected FA2 variant. Compile-time emission warns when the range-table size exceeds 8 KB; variants exceeding the threshold do not block merge but are documented as out of v1 scope.

### M6 — Regression bound on unpacked-but-segment_masked-on workloads

≤ **1%** wall-time regression on the `single_doc` fixture (one document filling the whole sequence; segment_masked still on). This is Tier B's worst case: every tile pair is within-segment, range check always passes, range check is pure overhead.

Pass criterion: `(wall_time_tier_b / wall_time_tier_a) ≤ 1.01` on the `single_doc` fixture, measured as the median of 5 runs.

If this gate fails, Tier B's overhead is bigger than expected; either the range-table preamble or the predicate is more expensive than designed and needs revisiting before merge.

### Implied (M4 — variant matrix audit)

Tier B adds no new dimensions to `FlashAttentionConfig`. The variant count is unchanged. M4 is implied — no explicit audit needed beyond confirming `flash_attention.rs::FlashAttentionConfig` is unchanged in the merge PR.

## 8. Alternatives considered

### 8.1 Option A — compile-time bitmap baked into PTX `.const` (paper's literal text)

**Rejected.** The cited paper assumes packing is fixed at kernel-build time, which holds for inference servers serving fixed prompt patterns. NSL's training case packs different documents per batch, breaking the assumption. The compile-time bitmap is correct under the paper's assumptions and incorrect under NSL's.

The design difference reflects a context difference, not a correction to the paper. The paper's authors weren't wrong; their context was different. Tier B's contribution is recognizing that static-layout assumptions don't hold in training and that the runtime range check is the correct adaptation.

### 8.2 Option C — hybrid: compile-time bitmap with runtime layout-match guard

**Rejected.** The layout-match predicate is itself correctness-bearing. Getting it wrong (e.g., accepting layouts that are "close enough" but not identical in skip-decision space) produces silent wrong outputs.

The matrix of "what makes two layouts skip-equivalent" is non-obvious — same `num_docs` but different `doc_length` distributions can produce different skip patterns; same lengths but different offsets can shift tile boundaries; padding distribution affects tail tiles. The complexity of correctly implementing the layout-match predicate exceeds the cost of just doing the runtime range check on every batch.

**Option B's runtime cost is bounded and known; option C's correctness depends on getting an unbounded predicate right.** That's the deciding asymmetry, not the surface complexity comparison.

### 8.3 Option D — per-cell check only (status quo, no Tier B)

This is what Tier A ships today: the per-element segment_mask predicate fires on every cell, masks cross-segment cells to `-inf`, and lets the matmul compute work that gets zeroed. It's correct and shipped. Tier B's value over option D is the ~30% FLOP reduction on packed workloads.

Option D remains the fallback if Tier B is later discovered to have a regression. Reverting Tier B is straightforward — one phase emitter call removed from forward and backward, plus the range-table preamble removed from the kernel preamble.

## 9. Module orphan policy — pinned

Two modules in `nsl-codegen/src/` are currently orphaned (no consumers). After Tier B lands, both remain orphaned. Their status:

- **`pca_rope.rs`** — *orphaned-pending Tier C* (RoPE position-reset fusion). Concrete future consumer planned. Tier B does not modify `pca_rope.rs`.
- **`PerDocumentCta` strategy in `pca_detect.rs`** — *orphaned-pending Tier D* (per-document CTA scheduling). Concrete future consumer planned. Tier B does not modify `PerDocumentCta`.

**Six-month deprecation-review trigger.** If either module is still orphaned six months after Tier B merges with no spec progress on its consumer tier, the maintainer marks it for deprecation review. Indefinitely-orphaned modules accumulate stale assumptions about NSL internals and become "rotting code" that's harder to revive than to rewrite.

This policy is general — applies to any orphan-pending module, not just Tier B's. Pin in `docs/wiki/` if the team finds it useful as a project-wide rule.

## 10. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Range-table reduction has a bug → wrong skip mask → wrong outputs | medium | high | M3 skip-mask parity test (§4.3) catches this on the fixture matrix before merge. The instrumentation cost (~0.5–1 day) is the price. |
| Branch is per-thread predicated, not warp-uniform → warps diverge → no perf win | medium | medium | M5's `BRA.U` SASS check (§6.3) catches this. If the check fails, the range-table emitter put values in per-thread registers — fix is to mark them uniform. |
| SMEM cost exceeds 8 KB on long-context variants → kernel doesn't fit | low | medium | Compile-time emitter warning when the formula exceeds 8 KB. Out-of-bound variants emit but with a documented diagnostic; long-context support deferred to "Tier B-extended." |
| Estimator and runtime drift over time (one refactored without the other) | medium | high | M3 parity test runs in CI, not just at merge. Same discipline as the AWQ pre-scan ↔ in-compile differential test. |
| Tier A relies on conservative-skip and someone tightens the predicate | low | high | §3.2 pins the conservative-skip semantic with worked example and explicit "do not tighten" warning. §2.4 pins "Tier B does not replace Tier A." |
| Performance regresses on a non-packed-but-segment_masked-on workload | low | medium | M6 catches this at merge time. ≤ 1% on `single_doc` fixture is the bound. |
| §4.4 fixtures fall back to Tier A only because segment-ID SMEM budget caps Shared residency at 2 K — M2/M3/M6 silently grade the wrong path | medium pre-mitigation, near-zero post-Commit-0 | high | §3.4 Commit 0 raises `DEFAULT_SMEM_SEGMENT_BUDGET` to 32 KB before any Tier B code lands. Verified by running M1 against the pre-bump baseline on existing Tier A fixtures (the bump's standalone safety) before subsequent Tier B commits. |
| Skip-decision instrumentation buffer layout is wrong (e.g., omits `(batch, head)` dims) — different CTAs overwrite the same slots and M3 silently passes despite misverification | medium | high | §4.3.1's full 4-D shape `[batch, head, num_q_tiles, num_kv_tiles]` is pinned. §4.3.2's lane-0-of-owning-warp write rule pins the threading model. The plan's instrumentation commit includes a dedicated test that fills the buffer with sentinel values and confirms only the expected `(b, h, qt, kvt)` slots are overwritten. |
| New Tier B-affected variant lands without a `tests/sass_baselines/` file → SASS drift goes undetected | low | medium | §6.3.1's per-variant baseline file mechanism is enforced in CI: a Tier B emission for a variant without a recorded baseline fails the build with a diagnostic pointing at the missing file. The merging PR establishes the baseline. |
| Probe outcome shifts under future CUDA toolkit / driver / new architecture; spec §3 needs revisit | medium long-term | high | Dated probe re-run triggers in `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`. Each re-run produces a dated entry; if outcome changes, §3.4 sub-layout choice revisits. |

## 11. Out-of-scope items captured for future work

- **Long-context Tier B (> 8 KB range tables)** — needs per-warp register or HBM-resident range tables. Out of v1; deferred until long-context training becomes a priority workload.
- **CTA-uniform vs warp-uniform predicate** — Tier B v1 uses warp-uniform. CTA-uniform would let multiple warps in a CTA share one decision but requires `__syncthreads()` overhead per tile-pair. Out of scope.
- **Tile-skip-aware backward checkpointing** — when skipping a tile in forward, we don't need to recompute it in checkpointed backward either. Currently the checkpoint mechanism doesn't know about tile-skips. Marginal win; out of scope.
- **Dynamic block size selection based on packing** — short docs benefit from smaller blocks; long docs from larger blocks. Out of scope; orthogonal to Tier B.

## 12. References

- CFTP paper, `docs/research/CFTP.pdf`, §2 ("PCA — Packed-Context Attention"), Tile-Skip subsection
- PCA Tier A spec, `docs/superpowers/specs/2026-04-18-pca-tier-a-design.md` (precedent for shared-helper discipline, ptxas verification, fixture matrix)
- AWQ pre-scan ↔ in-compile differential test, `docs/superpowers/specs/2026-04-22-awq-real-subprocess-completion-design.md` (precedent for estimator/runtime parity)
- WRGA B.3.1 spec, `docs/superpowers/specs/2026-04-13-wrga-milestone-b3-design.md` (precedent for ptxas hex-verification, shared-helper discipline)
- NVIDIA PTX ISA 7.0+, `BRA.U` warp-uniform branch instruction
- NVIDIA Nsight Compute, `smsp__sass_thread_inst_executed_op_*` FLOP counters (M2 measurement)
- Probe findings doc — `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`
- Revision design (brainstorm output) — `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md`
- Plan v2 — `docs/superpowers/plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md`
- Institutional-rules registry — `docs/wiki/institutional-rules.md`
