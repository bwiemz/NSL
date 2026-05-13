# PCA Tier B — Revision Design (re-brainstorm of §3.1, §3.4, §6.3)

**Status:** Design (spec) — revision of `2026-05-02-pca-tier-b-tile-skip-design.md`
**Date:** 2026-05-12
**Owner:** bwiemz
**Supersedes (in place, after probe runs):** `2026-05-02-pca-tier-b-tile-skip-design.md` §3.1, §3.4, §6.3
**Drives:** in-place edits to the 2026-05-02 spec + new plan v2 + new institutional-rules registry
**Precedent for the revision discipline:** WGGO Phase 2 #134 spec (per-commit milestone matrix), CSHA Tier B.1 V1/V2/V3 findings (pre-implementation verification), BitNet Phase 1 spec (institutional-rule codification at revision time)

---

## 1. Why this revision exists

The 2026-05-02 PCA Tier B spec shipped 6 prerequisite commits to main (PR #138, merged 2026-05-03) and then paused. The spec's header explicitly invited a re-brainstorm of §3.1 and §3.4:

> Plan's PTX-emission tasks (3–7) need rework: this spec's §3.1 and §3.4 call for inline `.shared` decls for the range tables, but the codebase's actual pattern (`forward/prelude.rs:270` + `backward/prelude.rs:249-258`) uses `smem_layout::*_offset(config)` with the range tables embedded in the existing `.extern .shared shmem[]` allocation. Inline `.shared` decls trigger `CUDA_ERROR_ILLEGAL_ADDRESS` on Blackwell sm_120 (the user's primary GPU). Re-brainstorm §3.1 and §3.4 before resuming.

Three follow-on SMEM fixes (PRs #147, #150, #152, merged 2026-05-11) landed Tier A-level sizing corrections (`shared_mem_bytes_v2_backward(config)` as single source of truth; `DEFAULT_SMEM_SEGMENT_BUDGET` sourcing in both preludes; `seg_overhead` accounting in `validate_scalar_v2_config`) but did **not** restructure the asymmetric seg_smem allocation pattern: forward `prelude.rs:285` still uses `.shared .align 4 .b8 seg_smem[N]`, backward `prelude.rs:270` still tail-embeds in extern shmem. The documented Blackwell warning at `backward/prelude.rs:270` is verbatim and confirmed load-bearing.

This revision design captures the re-brainstorm outcome:

- **§3.1** — replace compile-time-unrolled preamble with runtime PTX loop emission.
- **§3.4** — replace inline `.shared` decls with single tail offset (`smem_layout::tier_b_range_table_offset`) feeding a struct-based sub-layout in `pca_tilerange.rs`.
- **§6.3** — replace standalone-harness verification with hybrid (isolation-level insta snapshot + extended FA2 kernel snapshots + extended `pca_sass_byte_identity`).
- **New §1.5 (Phase 0)** — empirical forced-access SMEM probe gates the §3 work; produces a separate findings doc.

The revision is in-place at the 2026-05-02 spec (per Section 5.1 below). The original §3.1 / §3.4 / §6.3 text is preserved in git history; the revision changelog header cites this document by path.

## 2. Phase 0 — SMEM probe (gates §3 work)

### 2.1 What the probe answers

The 2026-05-02 spec's pause rationale presumes the Blackwell `CUDA_ERROR_ILLEGAL_ADDRESS` failure applies to mixed static-`.shared` + extern-shmem coexistence at runtime. Before §3 commits to tail-embedding Tier B's range tables (which would add extern shmem usage to forward's existing static `.shared seg_smem` decl), the assumption needs empirical verification on the hardware in use.

The probe is a B1.0-style pre-implementation verification deliverable: small, dated, produces a permanent findings doc that downstream work cites by path.

### 2.2 Probe specification (forced-access)

Probe kernel structure at `crates/nsl-codegen/tests/tier_b_smem_probe.rs` (test-only, not production code):

```ptx
.shared .align 4 .b8 seg_smem[N];        // Mimics Tier A forward's existing static decl
.extern .shared .align 16 .b8 shmem[];   // Mimics Tier B's planned extern usage

kernel_entry:
    mov.u32 tid, %tid.x;
    st.shared.u8 [seg_smem + tid], 0xAA;     // Force static-shared access
    st.shared.u8 [shmem + tid], 0xBB;        // Force dynamic-shared access
    bar.sync 0;
    ld.shared.u8 r1, [seg_smem + tid];
    ld.shared.u8 r2, [shmem + tid];
    // Write r1, r2 to global memory for verification.
```

**Forced-access discipline.** Every thread writes-and-reads both arrays. The historical bug fires at access time, not declaration time; declaration-only probe risks false-negative (probe reports "safe" when actual Tier B kernel would crash because mock didn't exercise the same access path).

### 2.3 Cross-config sweep

Configurations: static-array sizes `N ∈ {256, 1024, 4096}` × extern shmem sizes `M ∈ {16 KB, 64 KB, 96 KB}` × architectures `{sm_80, sm_120}`. 18 configurations total. The historical bug may be size-dependent; the sweep catches "passes at small sizes, fires at production scale" failure modes.

For each `(N, M, sm)` tuple:
1. Build the probe kernel PTX with the static-array size literal `N`.
2. Launch with dynamic shmem set to `M` bytes.
3. Check `cuLaunchKernel` returns `CUDA_SUCCESS` (no ILLEGAL_ADDRESS).
4. Read back sentinel bytes (`0xAA` from seg_smem, `0xBB` from shmem) on every thread.
5. Record outcome row: pass / launch-error / value-corruption.

### 2.4 Decision matrix (five outcomes)

| Probe result | Decision |
|---|---|
| Passes all probe configs | Option 1 confirmed safe. Proceed with tail-embedded range tables; leave forward's `.shared seg_smem` decl untouched. |
| Fires on all probe configs | Option 2 forced. Pre-Tier-B PR refactors forward to tail-embed seg_smem (snapshot refresh + hardware verification). |
| Fires only above a (N, M) threshold | Option 1 with planner gating. Tier B admitted only for configs whose total SMEM stays under the discovered threshold; update supported matrix CSV. |
| Fires only on sm_120 | **Conditional on deployment context at probe time.** If sm_120 is a primary deployment target (current state: RTX 5070 Ti is the user's primary GPU per project memory), option 2 forced — tail-embed both directions, accept snapshot-refresh scope. If sm_80/sm_90 is the primary target, option 1 with sm-aware dispatch; sm_120 deferred to Phase 1.5. The findings doc records which case applies. |
| Fires only on sm_80 | Architecturally unusual; investigate before proceeding (probe bug suspected). |
| Fires unpredictably across runs | Option 2 forced regardless of structural pattern. Unpredictable failures suggest memory-corruption pattern; option 1 risks intermittent production crashes. |

The "sm_120 is the live failure case" framing is concrete, not hypothetical: this user's primary GPU is RTX 5070 Ti (sm_120). The "fires only on sm_120 with sm_120 as primary target" branch is the most likely live outcome.

### 2.5 Findings doc

Lives at `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`. Initial entry records the 18-config sweep outcome + CUDA toolkit version + driver version + hardware list. Append-only "Re-run log" at the bottom.

**Re-run triggers** (pinned in the findings doc):

- CUDA toolkit major version bump (e.g., 13.x → 14.x). Driver behavior on mixed `.shared` allocations may shift.
- New target architecture added to NSL's supported matrix (e.g., sm_130 when it ships).
- Any production deployment surface reports ILLEGAL_ADDRESS on a Tier B kernel that previously passed CI.

Each re-run produces a dated entry: `2026-05-12 initial`, `2026-XX-XX CUDA 14 retest`, etc. The revised PCA Tier B spec's §3.4 cites this findings doc by path; the most recent finding determines the active §3.4 layout choice.

### 2.6 Budget

~3 hours probe + cross-config sweep + Blackwell/Ampere validation. ~1 hour findings doc. Total ~4 hours. Recoverable cost if option 1 is confirmed; load-bearing prerequisite if option 2 is forced (snapshot refresh adds ~half-day).

## 3. §3.4 revision — SMEM layout API

Replaces the original §3.4's inline `.shared .align 2 .b16` decls.

### 3.4.1 Surface in `smem_layout.rs` — one new public symbol

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

### 3.4.2 Internal sub-layout in `pca_tilerange.rs`

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

**Visibility.** `RangeTableAddrs` and `range_table_addrs` are `pub` at the module level. Three consumers in the same crate:
- Forward preamble emitter (`emit_range_table_preamble` for forward kernel).
- Backward preamble emitter (Tier B.2, when it lands).
- Parity test bit-equivalence comparison against `pca_tileskip::build`.

"Tier B owns its sub-structure" applies at the crate-boundary level (`smem_layout.rs` knows about one Tier B region), not at the file-boundary level (within Tier B's modules, the sub-layout is shared infrastructure).

### 3.4.3 Shared tile-count source of truth

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

### 3.4.4 num_tiles identity unit test

The "asserted identical" claim is verifiable, not aspirational. Test structure:

For every `(seq_len, block_size)` tuple in the supported config matrix:

1. Call `num_tiles(seq_len, block_size)` from Rust.
2. Generate the FA2 kernel PTX for that configuration; parse the tile-loop bound from the emitted `setp.lt.u32 %p, %r_tile, NUM_TILES;` instruction.
3. Compute the same value at the `range_table_addrs` consumer site (via direct Rust call to `pca_tile_config::num_tiles(seq_len, config.block_q)`).

All three values must equal. Test failure surface identifies which-of-three site disagrees with which other:
- Rust-formula vs emitted-PTX disagreement: kernel emission added a `+1` or `-1` adjustment somewhere.
- Rust-formula vs consumer-computation disagreement: consumer is using the wrong block-size operand (e.g., `block_q` for kv-tile count).
- Emitted-PTX vs consumer-computation disagreement: kernel and range-table allocator use different tile-count paths.

Failure localizable without bisecting kernel emission code. Same diagnostic-precision discipline as IR-006 applied at the unit-test level.

### 3.4.5 Launch-side accounting (no-op guarantee)

`shared_mem_bytes_v2_backward(config)` (added in commit `9ca2ea07`, post-merge fix) and the corresponding forward `shared_mem_bytes_v2(config)` gain a conditional Tier B contribution:

```rust
let tier_b_bytes = if should_emit_tier_b(config, seq_len, residency) {
    tier_b_range_table_bytes(config)
} else {
    0
};
let shared_mem_total = existing_terms + tier_b_bytes;
```

**No-op guarantee.** When Tier B is not emitted (config below threshold, sm < 80, etc.), the range-table-bytes contribution is exactly zero. Pre-Tier-B configurations have byte-identical SMEM layout to post-Tier-B. Snapshot tests for non-Tier-B kernels are unaffected; only the Tier B kernel's snapshot is new.

Same "what's unchanged" framing as #134's AWQ regression discipline — when a change affects an existing structure, explicitly characterize what's unchanged for existing consumers.

### 3.4.6 Findings-doc dependency citation

The revised §3.4's correctness depends on the SMEM probe's outcome. The struct-based sub-layout assumes tail-embed is safe.

> Probe finding `2026-05-12-tier-b-smem-probe-findings.md` selected outcome row `<row>` from §2.4's decision matrix. The §3.4 sub-layout above is correct for that outcome. Probe re-runs that shift the outcome may require §3.4 revisiting; the findings doc's "Re-run log" is the audit trail.

## 4. §3.1 revision — preamble shape (runtime PTX loop)

Replaces the original §3.1's compile-time-unrolled reduction sketch.

### 4.1 Emission shape — per-phase runtime loop

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

### 4.2 Warp-uniformity discipline (load-bearing)

For ptxas to compile the loop branch as `BRA.U` rather than `BRA` (per-thread predicated):

1. `%r_tile_*` is `.reg .u32` — not `.u64`, not signed, not derived from `%tid.x` or any thread-divergent source.
2. `mov.u32 %r_tile_*, 0` is the only initializer; no per-thread initialization paths.
3. `add.u32 %r_tile_*, %r_tile_*, 1` is unconditional (not predicated on `%tid` or any thread-divergent predicate).
4. `setp.lt.u32 %p_done_*, %r_tile_*, NUM_*_TILES` compares two uniform values (uniform register vs. compile-time literal).
5. The branch `@%p_done_* bra LOOP_*` consumes a uniform predicate.

**Anti-pattern: uniformity-through-load.** `%r_tile_*` must remain register-resident throughout the loop. Loading from `.local` (thread-local) or `.shared` (warp-or-CTA-local) into `%r_tile_*` breaks ptxas's uniformity tracking even if the loaded value happens to be uniform across threads. The only initialization is `mov.u32` from a literal; the only update is `add.u32 %r_tile_*, %r_tile_*, 1`. If a future optimization wants to vary the tile-count source, the SASS verification gate catches the regression (BRA replaces BRA.U).

This is IR-007 applied: PTX emission discipline is pinned at the instruction-pattern level (specific register class, specific operand shape, specific predicate construction) because the ptxas → SASS pipeline is pattern-recognition-driven.

### 4.3 Lane-0 store pattern (predicated execution, not divergent branch)

The single-lane store uses PTX predicated execution, not a thread-divergent branch:

```ptx
.reg .pred %p_lane_zero;
setp.eq.u32 %p_lane_zero, %lane_id, 0;
@%p_lane_zero st.shared.u16 [%addr_min], %rs_min_q_TILERANGE;
@%p_lane_zero st.shared.u16 [%addr_max], %rs_max_q_TILERANGE;
```

The predicate is thread-divergent (only lane 0 evaluates to true), but the predicated store is a single instruction that ptxas handles without affecting the surrounding warp-uniform loop structure. A thread-divergent branch (`@%p_lane_zero bra STORE_BLOCK; ...; STORE_BLOCK:`) would also be correct but emits divergent SASS — measurably worse and a different code shape than this spec assumes.

### 4.4 Inner reduction body

Fixed-instruction-count warp-shuffle butterfly (no nested loop):

1. Lane L loads `segment_ids[%r_tile_q * block_q + L]` into a u16 register.
2. If `block_q > warp_size` (e.g., block_q = 64 = 2 × warp_size on most configs), also load `+L+warp_size` and merge into lane-local min/max.
3. Five butterfly steps with offsets `{16, 8, 4, 2, 1}` using `shfl.sync.bfly` + `min.u16` / `max.u16` ops.
4. Lane-0 predicated store (per 4.3) to `qtile_seg_min[%r_tile_q]` / `qtile_seg_max[%r_tile_q]`.

Total inner-body instructions: ~12-14 per iteration. Only the outer tile-iteration is the runtime loop; the inner butterfly is unrolled (5 steps × 2 ops + load + store).

### 4.5 Address arithmetic into the range table

Per §3.4's alignment guarantee, `range_table_addrs.qtile_{min,max}` are 2-byte aligned. The per-iteration address computation:

```text
%addr_min = range_table_addrs.qtile_min + 2 * %r_tile_q_TILERANGE
%addr_max = range_table_addrs.qtile_max + 2 * %r_tile_q_TILERANGE
```

Both offsets are uniform across the warp (same on every thread); only lane 0 actually performs the store. The shift-and-add is two instructions per iteration — negligible overhead.

### 4.6 SASS verification gate

`pca_sass_byte_identity` extracts the SASS for each phase's loop branch and asserts:

- The q-phase loop branch is `BRA.U` (or per-architecture equivalent).
- The kv-phase loop branch is `BRA.U`.
- On sm_90 / sm_120: range-table min/max stores use `UR<n>` operands (direct uniform-class verification).
- On sm_80 / sm_86: BRA.U at the loop branch is the available proxy — necessary for uniformity, but not sufficient to verify the inner store operands are uniform-resident. Accepted v1 limitation.

If the SASS check fails, the fallback is to investigate ptxas's uniformity-recognition heuristic and adjust the emission to match — not paper over with a register-allocator hint until the root cause is understood.

### 4.7 Hybrid-rejection paragraph (cite IR-005)

> The hybrid emission split (compile-time-unroll at small `num_tiles`, runtime loop above a threshold) was considered and rejected. Per IR-005, bifurcation of emission paths for the same logical operation requires measurement-driven justification, not architectural preference. At v1 scale, the slight runtime cost (~4 instructions per tile: branch + counter increment + setp) is dwarfed by the matmul work being skipped or computed; the hybrid's "small-N performance win" is not a measurable concern. Uniform emission across the config matrix is a v1 architectural property. Bifurcation deferred to Phase 1.5 with explicit measurement-driven justification if it becomes warranted. Same discipline as FA-2 v2 Tier B.1's "no producer/consumer split in v1."

## 5. §6.3 revision — verification hybrid

Replaces the original §6.3's standalone `emit-pca-tier-b-preamble-harness` binary approach entirely.

### 5.1 Isolation-level test (catches PTX-string drift)

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

**Sentinel detection assumes register-loaded literal emission.** The detection works for two PTX emission shapes:
- (α) `mov.u64 %base, 0xDEADBEEF` — register-loaded literal; sentinel visible verbatim.
- (γ) `add.u64 %addr, %something, 0xDEADBEEF` — arithmetic operand; sentinel visible as instruction operand.

The detection does **not** work for:
- (β) `ld.param.u64 %base, [param_slot]` — `base_offset` passed as launch-time `.param`; sentinel value lives in launch parameter list, not in PTX text.

**Required emission discipline:** the preamble emitter materializes `base_offset` as a register-loaded literal or arithmetic operand at PTX-emission time, not as a launch-time `.param`. If a future optimization wants to pass `base_offset` via `.param`, the isolation test needs a corresponding adaptation (e.g., a second sentinel for the `.param`-slot index, or a separate test for param-slot resolution).

**Failure surface (α: PTX-string drift):** ~5-line PTX diff localized to Tier B's emission. Reviewer sees the exact instruction(s) that changed and decides accept (legitimate refactor) or reject (bug) immediately.

### 5.2 Integration-level checks (catches SASS regressions)

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
// sm_80/sm_86 fall back to BRA.U-as-proxy per §4.6's accepted v1 limitation.
```

**Per-variant SASS baselines** at `tests/sass_baselines/<variant>_tier_b.txt` record cuobjdump instruction count + zero-spill-count for each Tier B-affected FA2 variant. Drift > ±2 fails CI. New variant ships its own baseline file in the same PR.

**Failure surface (β: SASS regression):** test name identifies the specific regression class (`BRA.U`, uniform-register class, instruction count, spill count); SASS diff at the offending offset is the diagnostic.

### 5.3 PR-coordination discipline

Any PR modifying Tier B's preamble emission updates all three test surfaces in the same commit. The PR description includes a checklist:

- [ ] `pca_tier_b_preamble_isolation` snapshot re-baselined; diff included in PR description.
- [ ] `pca_{forward,backward}_kernel_snapshot` snapshots re-baselined for each affected Tier B variant; per-variant diffs summarized.
- [ ] `pca_sass_byte_identity` assertions verified; any assertion-offset changes are explicit code edits with rationale.
- [ ] Per-variant SASS baseline files updated; instruction count + spill count match the new SASS.
- [ ] PR description summarizes the cascade: "PTX-text change at X → kernel SASS shifted at Y → uniform-register class verified at Z."

The cascade-summary line is the load-bearing audit point. If the three layers' changes don't form a coherent narrative (e.g., PTX-text change at the q-phase loop counter but SASS diff at an unrelated offset), one of the updates is wrong.

Same discipline as WGGO Phase 2's "snapshot re-baseline review discipline" — snapshot acceptance is the load-bearing review moment, not a rubber-stamp.

### 5.4 Deletion of `emit-pca-tier-b-preamble-harness` bin

The original plan's Task 3 Step 4 proposed a standalone `bin` target that emits stub kernel PTX containing only the Tier B preamble for isolated ptxas verification. This is deleted.

**Load-bearing reason: maintenance cost over time.** The harness would have required ongoing maintenance as FA2's register environment evolves. Every change to scratch register usage, every new CSHA hook, every shift in SMEM layout would require mirroring the change into the harness's mock environment. Either the mock drifts from production (silent test passes for wrong reasons) or the maintenance cost is paid continuously.

The hybrid approach has zero mock-environment maintenance: the isolation test uses the real preamble emitter; the kernel snapshot tests use the real FA2 emitter. Both verification surfaces are coupled to actual production code, not to a parallel mock.

### 5.5 §6.3 satisfaction table

All four v1 ptxas-pass criteria satisfied:

| Criterion | Satisfied by |
|---|---|
| ptxas accepts on sm_120 + sm_80 | `pca_{forward,backward}_kernel_snapshot` tests already run ptxas; Tier B-enabled variants run alongside |
| Zero spills (`ptxas -v` output) | Per-variant baseline files at `tests/sass_baselines/<variant>_tier_b.txt` record + grep |
| `BRA.U` at skip predicate + loop branch | Extended `pca_sass_byte_identity` assertions at q-phase + kv-phase loop branches |
| Uniform-register placement | Direct on sm_90/sm_120 (`UR<n>` grep); BRA.U-as-proxy on sm_80/sm_86 |

Plus the isolation-level snapshot from §5.1 — strictly additive to §6.3; not required by spec text but justified by IR-006 (distinct failure modes warrant distinct test surfaces).

## 6. Packaging and delivery

### 6.1 In-place spec revision

`docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` — revised in place, original preserved via git history. Specific changes:

| Section | Change |
|---|---|
| Header | Remove "EXECUTION PAUSED 2026-05-02" warning. Add **Revision Changelog** entry: `2026-05-12 — §3.1, §3.4, §6.3 revised after re-brainstorm; cites probe findings doc + plan v2; see 2026-05-12-pca-tier-b-revision-design.md for the brainstorm output.` |
| **§3.1** | Replace compile-time-unrolled sketch with §4 of this document. |
| **§3.4** | Replace inline `.shared .align 2 .b16` decls with §3 of this document. Cite probe findings doc as a dependency. |
| **§6.3** | Replace standalone-harness approach with §5 of this document. |
| §10 (risks) | Add row: "Probe outcome shifts under future CUDA toolkit / driver / new architecture; spec §3 needs revisit. Mitigation: dated probe re-run triggers listed in findings doc." |
| §12 (references) | Add: probe findings doc path; plan v2 path; this revision design path. |

### 6.2 Probe findings doc

`docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md` — separate document. Initial entry recorded post-probe-run. Append-only "Re-run log" maintained per §2.5.

### 6.3 New plan v2

`docs/superpowers/plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md` — fresh document, not a diff of plan v1. v1 stays in git history; v2 is the active plan.

**v2 task graph (13 main tasks with 3 verification sub-tasks for 16 entries; 8 new/reshaped from v1, 8 unchanged in shape):**

| # | Task | Δ from v1 |
|---|---|---|
| 0 | Worktree setup + prereq verification | Unchanged |
| 1 | **SMEM probe — Phase 0 gate** | NEW (per §2 here) |
| 2 | **Apply probe outcome to spec** | NEW |
| 3 | **`pca_tile_config::num_tiles` shared helper + identity test** | NEW (per §3.4.3 + §3.4.4 here) |
| 4 | **`smem_layout::tier_b_range_table_offset` + `RangeTableAddrs` struct + constructor** | RESHAPED |
| 5 | **Implement `emit_range_table_preamble` (runtime loop)** | RESHAPED (per §4 here) |
| 5a | **Isolation insta snapshot for preamble** | NEW (per §5.1 here) |
| 5b | **Extend `pca_sass_byte_identity` with `BRA.U` + `UR<n>` assertions at preamble offset** | NEW (per §5.2 here, replaces v1 standalone harness) |
| 6 | Implement `emit_skip_predicate` + wire into `s_compute.rs` kv-tile loop | Mostly unchanged |
| 6a | **Isolation snapshot for predicate + SASS assertion (BRA.U at branch)** | NEW (mirrors 5a/5b) |
| 7 | Kernel-side debug instrumentation for M3 skip-decision writeback | Unchanged |
| 8 | Wire Tier B forward into FA2 kernel end-to-end + extended kernel snapshot | Unchanged in shape; snapshot extension per §5.3 cascade discipline |
| 9 | M3 estimator/runtime parity test | Unchanged |
| 10 | Per-variant SASS baseline files + CI gating | Unchanged in shape |
| 11 | M2 + M6 measurement infrastructure | Unchanged |
| 12 | Close-out + memory updates | Unchanged |

Reshape ratio: 8 of 16 entries new/reshaped (3 NEW main tasks: 1, 2, 3; 2 RESHAPED main tasks: 4, 5; 3 NEW sub-tasks: 5a, 5b, 6a). Substantial enough to justify a fresh plan document over an in-place revision.

### 6.4 Institutional-rules registry

`docs/wiki/institutional-rules.md` — new canonical surface. Each rule has a stable identifier (IR-NNN), a one-paragraph statement, and a "Cited from" list of specs.

**Initial entries from this revision:**

- **IR-001 — Preconditions enforced by API shape, not docstrings.** Where a function has an unstated precondition (a callee must be invoked first, an input must be in a specific state), the API should make violation structurally impossible — keep the unsafe-without-prep function module-internal, expose only the safe composition. Distinct from type-system enforcement; this is about visibility and composition shape. Cited from BitNet Phase 1 `quantized_ternary_gemm` fusion, AWQ calibration `weight_index_map`, this revision's §3.4 `RangeTableAddrs` constructor.

- **IR-002 — External references as one-time anchors.** External measurements / hardware findings recorded in dated findings docs and cited by path from specs that depend on them. Cited from BitNet HF checkpoint pinning, AWQ calibration, this revision's §3.4 (probe findings doc dependency).

- **IR-003 — Pre-implementation verification of load-bearing assumptions.** When a spec relies on cross-module behavior, verify the assumption via grep / probe / measurement before writing the code that depends on it. Cited from WGGO Phase 2 NodeId space, CSHA Tier B.1 V1/V2/V3, this revision's §2 SMEM probe.

- **IR-004 — APIs return aligned, ready-to-use values so consumers don't have to.** Offset-computation site owns alignment; consumer assumes ready-to-use. Cited from `kv_offset` / `sp_offset` discipline, BitNet `packed_load.rs`, this revision's §3.4.1 `tier_b_range_table_offset` alignment guarantee.

- **IR-005 — Bifurcation of emission paths for the same logical operation requires measurement-driven justification, not architectural preference.** v1 prefers uniform emission across the config matrix; bifurcation deferred to Phase 1.5 with explicit measurement justification. Cited from FA-2 v2 Tier B.1 (no producer/consumer split), this revision's §4.7 (no compile-time-unroll / runtime-loop hybrid).

- **IR-006 — Distinct failure modes warrant distinct test surfaces.** Bundle test concerns only when their failure modes have identical diagnostic shape; otherwise split them. Cited from CSHA Tier B.1 cost-model correction (standalone PR), WGGO Phase 2 #134 (per-commit milestone matrix), this revision's §5 (isolation snapshot + integration SASS split).

- **IR-007 — PTX emission discipline is pinned at the instruction-pattern level, not the algorithmic level, because the ptxas → SASS pipeline is pattern-recognition-driven.** Cited from FA-2 v2 cp.async commit-group cadence, BitNet `BFE.U32` for trit unpack, this revision's §4.2 (warp-uniformity through specific register-class + operand patterns).

- **IR-008 — For long-lived kernels, verification surface investment is the load-bearing cost.** Observation across multiple kernel specs: kernel emission is a few hundred lines; verification (snapshots, SASS assertions, parity tests, reference impls) is comparable or larger and is what catches regressions over the kernel's lifetime. Not a rule that prescribes a ratio — a framing that justifies budget allocation when the verification surface seems disproportionate to the emission. Cited from FA-2 v2 Tier B.1, BitNet Phase 1, this revision's overall §3.1 / §3.4 / §6.3 balance.

**Registry intro structure:**

```markdown
# NSL Institutional Rules

This document catalogs project-level design principles surfaced across NSL's
spec and brainstorm work. Each rule has a stable identifier (IR-NNN) used in
spec citations.

## How to read this document

Each rule is one paragraph stating the principle and citing the specs where
it was surfaced or applied. Specs cite rules by identifier
(e.g., "per IR-003, the verification gate runs before implementation").

## How to add a rule (entry criteria)

A pattern becomes an IR when it satisfies all three:

- Surfaced across at least two distinct specs/brainstorms.
- The pattern's violation produced or would have produced a real failure mode in retrospect.
- The pattern is small enough to cite by identifier and explain in one paragraph.

Patterns that don't satisfy these criteria are documented in the rejecting
spec's text as "considered but not codified," not added to the registry.

The criteria prevent two failure modes: registry inflation (every spec adding
"lessons learned" entries) and registry stagnation (real patterns never
codified because nobody knows when to add).

## Rules

[IR-001 through IR-008 here.]
```

Per-rule "Cited from" list is the audit trail. Future readers seeing `IR-003` cited in a new spec navigate to the registry and find the precedent.

### 6.5 Net delivery summary

After Phase 0 probe + revision + v2 plan + registry land:

- **1 revised spec** in place (§3.1 / §3.4 / §6.3 rewritten + §10 / §12 extended + changelog header).
- **1 new findings doc** (probe results + dated re-run log).
- **1 new plan v2** (13 main tasks + 3 verification sub-tasks; 8 new/reshaped from v1, 8 unchanged in shape).
- **1 new institutional-rules registry** (`docs/wiki/institutional-rules.md` with IR-001..IR-008).
- **0 deleted artifacts.** v1 spec preserved via git history; v1 plan preserved via git history; existing memory file `project_pca_tier_b_paused.md` updated to point at the new revision (done earlier this session).

The "0 deleted artifacts" property preserves all prior work — git history holds the original spec, plan v1, and the EXECUTION PAUSED state. The revision is additive at the file-system level even when it's substitutive at the code-content level. Same discipline as the calibration spec's deprecated-shim lifecycle and the WGGO Phase 2 spec's snapshot-re-baseline review.

## 7. Implementation sequence

After this revision design commits, the implementation sequence is:

1. **Probe runs** (§2 deliverable) → findings doc commits to `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`.
2. **Spec revised in place** per §6.1, citing the findings doc by path. Outcome row from the five-outcome decision matrix instantiates §3.4's struct-based sub-layout.
3. **Institutional-rules registry created** per §6.4 at `docs/wiki/institutional-rules.md`. Can land in parallel with step 2; not blocked by probe.
4. **Plan v2 written** per §6.3, referencing both the revised spec and the registry.
5. **Implementation proceeds** per plan v2's task graph.

Step 1 is gating: §3.4's sub-layout choice depends on the probe outcome. Steps 2 and 3 are parallelizable. Step 4 follows step 2. Step 5 follows step 4.

## 8. References

- 2026-05-02 PCA Tier B spec (paused) — `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md`
- 2026-05-02 PCA Tier B plan v1 (preserved via git history; superseded by v2 per §6.3) — `docs/superpowers/plans/2026-05-02-pca-tier-b-tile-skip-implementation.md`
- Probe findings doc (created post-probe) — `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md`
- Plan v2 (created post-revision) — `docs/superpowers/plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md`
- Institutional-rules registry (created alongside revision) — `docs/wiki/institutional-rules.md`
- FA-2 v2 Tier B.1 (CSHA pipelined MMA forward) — `docs/superpowers/specs/2026-05-11-csha-tier-b1-pipelined-attention-design.md` (precedent for uniform-emission discipline, V1/V2/V3 verification pattern)
- WGGO Phase 2 #134 spec — precedent for per-commit milestone matrix and snapshot-re-baseline review discipline
- BitNet Phase 1 design — precedent for institutional-rule codification at revision time, API-shape-enforced invariants
- Existing post-merge fix `9ca2ea07` (`shared_mem_bytes_v2_backward` single source of truth) — load-bearing for §3.4.5's launch-side accounting
- Existing post-merge fix `7c99854c` (seg_smem sizing from `DEFAULT_SMEM_SEGMENT_BUDGET`) — pinned the asymmetric forward (separate `.shared`) vs backward (tail-embed) seg_smem allocation that §2's probe verifies safe for Tier B coexistence
