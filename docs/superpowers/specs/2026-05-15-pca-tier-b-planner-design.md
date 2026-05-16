# PCA Tier B — Planner Design (Option B: Runtime Dispatch)

**Status:** Design (spec) — follow-on to the dispatch spec's §14 amendment with V-planner-options grounding.
**Date:** 2026-05-15
**Owner:** bwiemz
**Builds on:**

- `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` — dispatch spec with §14 amendment (Option B recommendation; case-(α) path invalidated).
- `docs/superpowers/specs/2026-05-14-tier-b-dispatch-integration-findings.md` — V-dispatch-integration findings (α=0, β=3, γ=0).
- `docs/superpowers/specs/2026-05-15-tier-b-planner-options-findings.md` — V-planner-options findings (Option B viable; sub-variant question deferred to V-Bii-SMEM).
- `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` — KEEP UNCONDITIONALLY outcome (73.33% wall-time win, saturating-curve finding).
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` — original SMEM probe Phase 0 (V-Bii-SMEM inherits this discipline).
- `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` — original spec §11 deferred-extension queue (B-iii citation).

**Codifies (or extends):** IR-008 + IR-011 (new "Cited from" entries per §12.1).

---

## 1. Why this spec exists

The dispatch spec (2026-05-14) pivoted at amendment §14 after V-dispatch-integration found α=0, β=3, γ=0 across all production callers of `synthesize_flash_attention_ptx_v2`. The amendment recommended Option B (move dispatch past codegen) but the recommendation was made under time pressure; V-planner-options (2026-05-15) ran the empirical re-derivation and confirmed B as the only viable v1 path:

- **Option A** is architectural revision of FlashAttentionConfig (not threading) — ~5-10× the amendment's cost framing. Reserves to a future architectural revision.
- **Option C** has no dated shape-inference roadmap to depend on — collapses to indefinite deferral (IR-009 anti-pattern).
- **Option B** is viable. The launch wrapper (`nsl_flash_attention_csha:373`) already has `seq_len` in scope and already does runtime variant selection (`effective_heads` from `active_heads`, lines 414-418). The dispatch decision moves to where the data lives.

V-planner-options additionally surfaced a sub-question the amendment didn't capture: Tier B SMEM range tables are sized at codegen time, so Option B has three sub-variants (B-i dual-emission, B-ii single-emission with conservative max, B-iii HBM-resident tables). The planner spec runs a Phase 0 V-Bii-SMEM probe with a five-outcome decision matrix (mirroring the original SMEM probe's discipline from the revision spec §1) before committing to a sub-variant.

The spec relates to the dispatch spec as follows: **dispatch spec defines the heuristic shape; planner spec defines where the decision lives.** After the planner spec lands with its A/B/C resolution and sub-variant choice, the dispatch spec's D-3 milestone proceeds per the resolution; D-2's floor derivation is rehabilitated as a runtime input.

### 1.1 Lifecycle pinning

After this planner spec lands with V-Bii-SMEM's outcome and the sub-variant choice, the dispatch spec's §14 amendment gains a closing citation: *"Resolved by `docs/superpowers/specs/2026-05-15-pca-tier-b-planner-design.md` §N (Option B-i / B-ii / B-ii-restricted per V-Bii-SMEM findings)."* The dispatch spec then proceeds through D-2 (floor derivation, rehabilitated as runtime input per §7 of this planner spec), D-3 (implementation per the planner spec's chosen sub-variant), and D-4 (snapshot re-baselines + activation PR).

The dispatch spec's D-1 milestone (V-dispatch-integration) is complete. D-2 onwards proceeds after this planner spec lands. The two specs' milestones interlock: planner spec's V-Bii-SMEM precedes dispatch spec's D-2 onwards; planner spec's outcome determines dispatch spec's D-3 implementation path.

## 2. Scope and out-of-scope

### 2.1 In scope (this spec)

- **Option B commitment** as the architectural answer to the dispatch spec's §14 gap. Dispatch decision moves from codegen to launch wrapper; codegen emits Tier-B-on PTX for `segment_masked` configs; runtime dispatcher applies the seq_len floor.
- **V-Bii-SMEM probe** as Phase 0 gate. Five-outcome decision matrix mirroring revision spec §1's discipline. Sweep `MAX ∈ {4096, 8192, 16384}` × `block ∈ {32, 64}` × architecture `{sm_80, sm_120}` (12 configs). ~4 hour budget.
- **Sub-variant resolution conditional on probe outcome** — maps deterministically to one of three sub-variants:

  | Probe outcome | Sub-variant | Supported seq_len |
  |---|---|---|
  | MAX=16384/block=32 fits across all configs | B-ii unrestricted | Up to 16384 |
  | MAX=8192 fits but 16384 doesn't | B-ii-restricted (MAX=8192) | Up to 8192; configs above fall back to Tier-B-off |
  | Only MAX=4096 fits | B-i (single-emission at MAX=4096) | Up to 4096; matches PR #169 baseline |
  | MAX=4096 doesn't fit either | Investigation; B-iii becomes default | — |
  | Unpredictable / non-monotonic | Investigation before proceeding | — |

- **FFI signature extension** of `nsl_flash_attention_csha` (and 5 sibling launch entry points) — 2-param shape (`tier_b_ptx_ptr`, `tier_b_name_ptr`) in all sub-variants. In-place extension with sentinel defaults. Helper functions encapsulate sentinel construction (IR-001 discipline).
- **Codegen emission policy**: for `segment_masked` configs, codegen emits a Tier-B-on PTX blob alongside the existing base (Tier-B-off) PTX. Non-`segment_masked` configs are unchanged.
- **Runtime dispatcher branch logic**: launch wrapper picks the Tier-B-on variant based on `segment_ids_ptr != 0 && seq_len >= TIER_B_SEQ_LEN_FLOOR && seq_len <= TIER_B_MAX_BAKED_SEQ_LEN`. Sentinel-pair-mismatch detection panics with a diagnostic.
- **Floor derivation rehabilitation**: dispatch spec's D-2 milestone is repurposed as the source for `TIER_B_SEQ_LEN_FLOOR`. Same measurement protocol; outcome consumed at runtime instead of codegen.
- **Test surface composition**:
  - **B-i:** snapshot strategy uses a single Tier-B-on variant snapshot per affected fixture (MAX=4096 matches PR #169's existing baseline; minimal re-baselining).
  - **B-ii / B-ii-restricted:** snapshot strategy uses a single Tier-B-on variant snapshot per affected fixture (at the resolved MAX value); the Tier-B-off path goes through `_with_tier_b(config, None)` which the codegen heuristic also emits as the base PTX.
  - The 2 isolation tests (`pca_tier_b_preamble_isolation`, `pca_tier_b_predicate_isolation`) call emission functions directly per PR #168; they stay byte-stable.
- **Migration triggers to v2** for both the dispatch spec's existing deferrals and new triggers introduced by this spec (§9 below).

### 2.2 Out of scope (deferred)

- **Option A architectural revision of FlashAttentionConfig** — future architectural revision trigger only.
- **Option C shape-inference-dependent dispatch** — needs a dated shape-inference spec on the roadmap before becoming viable.
- **B-iii HBM-resident range tables** — kernel architecture change; matches the original 2026-05-02 spec §11 deferred-extension queue.
- **Dispatch spec's existing v2/v3 deferrals** — TierBPolicy enum, planner module above codegen, per-config residency selection. Inherited unchanged from dispatch spec §9.
- **Tier B-extended for seq_len > 16K** — two distinct v2 triggers depending on V-Bii-SMEM's measured SMEM headroom:
  - **If MAX=16384 fits with significant headroom** (<60% utilization): v2 trigger is "extend baked max to MAX=32768" with a new V-Bii-SMEM probe at the extended dimensions. ~30 min probe + 2 LOC change.
  - **If MAX=16384 fits tightly** (>85% utilization): v2 trigger is "migrate to B-iii (HBM-resident range tables)" — ~500 LOC kernel architecture change.

  The V-Bii-SMEM findings doc records the SMEM-headroom percentage at MAX=16384; this number determines which v2 path is appropriate when the trigger fires.

### 2.3 Why this scope is right-sized

The spec resolves exactly one concern: **where does the dispatch decision live?** The dispatch spec answered "what is the heuristic, what's the default residency, how do snapshots get re-baselined?" Those decisions are inherited unchanged. The planner spec's surface is the dispatch decision's location — codegen vs launch — plus the consequential FFI / emission / runtime-branch design decisions that follow from picking launch.

The institutional pattern: **smallest viable v1 surface that closes the gap V-dispatch-integration surfaced.** Adding scope beyond this (e.g., revisiting the heuristic shape, or expanding to multi-variant Tier B) would conflate the planner spec's purpose with the dispatch spec's.

## 3. V-Bii-SMEM probe (Phase 0 gate)

### 3.1 What the probe establishes

The probe characterizes the **SMEM feasibility envelope** for single-emission Option B variants. Tier B's range tables are sized at codegen via `compute_range_table_bytes(seq_len, block_q, block_kv)` (in `crates/nsl-codegen/src/pca_tilerange.rs:39`). Single-emission requires baking a conservative-max `seq_len` into the kernel's SMEM allocation. The 99 KB Blackwell `.shared` cap (per revision spec §1's findings) is the binding constraint. If the conservative-max binding fits across the sub-variants the binary needs to cover, single-emission (B-ii / B-ii-restricted) is viable; if it doesn't, single-emission at the gate-fixture seq_len (B-i) is the only path.

This is structurally the same question as the original SMEM probe (revision spec §1) — does a forced-access allocation pattern fit the cap across configurations — but with a different free variable (conservative-max `seq_len` instead of segment-tile residency).

### 3.2 Probe structure

Forced-access SMEM probe at `crates/nsl-codegen/tests/tier_b_bii_smem_probe.rs` (test-only; mirrors `tier_b_smem_probe.rs` from revision spec §1):

```ptx
.shared .align 4 .b8 seg_smem[N];        // Forward's static SMEM region (sized from FlashAttentionConfig).
.extern .shared .align 16 .b8 shmem[];   // Tier B's extern SMEM region (sized to MAX × block-derived range table bytes).
// Every thread writes a sentinel to both regions; bar.sync; readback verification confirms allocation succeeded.
```

Probe kernel computes `range_table_bytes = compute_range_table_bytes(MAX, block_q, block_kv)` at codegen, declares the SMEM region of that size, and the host code attempts `cuModuleLoadData` + `cuLaunchKernel`. The cap is enforced by ptxas + the driver; failures surface as `CUDA_ERROR_INVALID_PTX` (ptxas) or `CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES` (driver). Success = the probe runs and the sentinel readback verifies.

### 3.2.1 Co-measurement — Tier-B-off baseline

The probe kernel naturally has access to both Tier-B-off SMEM (`seg_smem[N]` only) and Tier-B-on SMEM (`seg_smem[N]` + `shmem[]`). The probe records both numbers per config:

- **Tier-B-off baseline:** `seg_smem[N]` bytes. SMEM allocation Tier B's range tables ADD to.
- **Tier-B-on total:** `seg_smem[N]` + `shmem[]` bytes. Value the cap binds.
- **Tier B's incremental contribution:** difference between the two.

Co-measurement is essentially free (the probe kernel allocates both regions; recording sizes is metadata-only). Findings doc records all three per config for diagnostic completeness — the probe doubles as a SMEM-allocation regression checkpoint for the entire FA-2 v2 kernel.

### 3.3 Sweep dimensions

12 configurations total:

| Dimension | Values | Why |
|---|---|---|
| `MAX` (conservative-max `seq_len`) | {4096, 8192, 16384} | 4096 = gate-fixture seq_len (lower bound); 16384 = long-context aspirational (upper bound); 8192 = midpoint for B-ii-restricted possibility |
| `block` (`block_q = block_kv` for the probe) | {32, 64} | 32 = smallest gate-fixture block (worst case for range-table footprint); 64 = canonical gate-fixture block |
| Architecture | {sm_80, sm_120} | sm_120 = Blackwell with 99 KB cap (binding constraint); sm_80 = Ampere with 100 KB cap (sanity-check ceiling) |

**Architecture-dimension rationale:** `{sm_80, sm_120}` matches the original SMEM probe (revision spec §1) and NSL's current supported architecture matrix. sm_75 (Turing) is not currently in NSL's supported set; adding it would produce data NSL can't act on. If sm_75 enters NSL's supported matrix in a future architectural revision, V-Bii-SMEM is re-run on the extended set (per IR-002 — external references as one-time anchors).

### 3.4 Five-outcome decision matrix

| Probe outcome | Sub-variant | Supported seq_len | Implementation impact |
|---|---|---|---|
| MAX=16384/block=32 fits across all configs | **B-ii unrestricted** | Up to 16384 | Tier-B-on PTX sized at MAX=16384 |
| MAX=8192 fits but 16384 doesn't | **B-ii-restricted (MAX=8192)** | Up to 8192; configs above fall back to Tier-B-off via runtime gate | Tier-B-on PTX sized at MAX=8192 |
| Only MAX=4096 fits | **B-i (single-emission at MAX=4096)** | Up to 4096; matches PR #169's existing snapshot baseline | Tier-B-on PTX sized at MAX=4096 |
| **MAX=4096 doesn't fit either** | **Investigation before proceeding; B-iii becomes default** | — | **P-2 blocks; single-emission infeasible at any conservative-max; HBM-resident range tables (B-iii) become the only path.** Revisit architecture per §3.4.1. |
| Unpredictable / non-monotonic | **Investigation before proceeding** | — | P-2 blocks indefinitely until resolution |

Same `pass thresholds + fail semantics + protocol pinned before measurement` discipline as IR-010.

### 3.4.1 Investigation outcome — operational meaning

If the probe surfaces unpredictable / non-monotonic results (e.g., MAX=4096 fits, MAX=8192 fails, MAX=16384 fits — a non-monotonic pattern with respect to MAX), the implementer:

1. **Suspects probe-implementation bug.** Re-verify the probe kernel's PTX emission, sentinel-write logic, and readback verification. The probe's correctness is the first hypothesis when results are non-monotonic.
2. **Cross-checks against the original SMEM probe** (revision spec §1) on the same hardware/toolkit/driver. Any V-Bii-SMEM non-monotonic result should reconcile with the original probe's findings.
3. **Documents the anomaly in the findings doc.** Specific configs that produced unexpected results; suspected cause; resolution path.
4. **Does NOT commit to a sub-variant** while investigation is open. The planner spec's downstream work (FFI extension, codegen emission, runtime dispatcher) is blocked until the probe produces a clean five-outcome-matrix row.
5. **Budget for investigation:** ~2-4 hours if probe-implementation bug; ~1 day if hardware/driver anomaly. If unresolvable, escalate to a separate spec or revisit the architecture (B-iii becomes the default).

### 3.5 SMEM-headroom recording

The probe records SMEM utilization percentages for two distinct v2-trigger ladders.

**At MAX=16384/block=32** (always recorded if it fits): primary headroom for the MAX=16384 case's v2 triggers.

- **<60% utilization:** significant headroom; v2 trigger is "extend baked max to MAX=32768" with a follow-up probe.
- **60–85% utilization:** bounded headroom; v2 trigger evaluates both extend-max and B-iii migration based on workload urgency.
- **>85% utilization:** tight fit; v2 trigger is "migrate to B-iii" (HBM-resident).

**At MAX=8192/block=32** (recorded when MAX=16384 doesn't fit but MAX=8192 does): headroom for B-ii-restricted's v2 trigger ladder.

- **<60% utilization at MAX=8192:** significant headroom; v2 trigger is "re-run probe at MAX=16384 with current driver/toolkit" — the gap may have closed.
- **60–85% utilization at MAX=8192:** bounded headroom; v2 trigger evaluates both re-probe-at-16384 and B-iii migration based on workload urgency.
- **>85% utilization at MAX=8192:** tight fit; v2 trigger is "migrate to B-iii" directly.

### 3.6 Deliverable

Findings doc at `docs/superpowers/specs/2026-05-XX-tier-b-bii-smem-probe-findings.md`. Records per-config pass/fail + SMEM byte counts (Tier-B-off baseline + Tier-B-on total + incremental contribution) + cap utilization %; decision-matrix row that fires; sub-variant resolution; FFI extension scope; SMEM-headroom percentages for v2 trigger reference; cross-reference to V-planner-options findings + dispatch spec §14 amendment.

**Re-run triggers** (append-only re-run log per IR-012):

- New architecture added to NSL's supported matrix.
- CUDA toolkit / driver update where SMEM-cap or compilation behavior may shift.
- Probe-implementation suspected bug surfaces; re-run with fixed probe.

### 3.7 Budget

~3 hours probe authoring + cross-config sweep + cross-architecture verification + ~1 hour findings doc structure. Total ~4 hours, matching the original SMEM probe's institutional discipline (IR-008 — verification investment comparable to emission).

### 3.8 Sequencing

V-Bii-SMEM is **Phase 0**: runs before any FFI extension, codegen emission policy, or runtime dispatcher work commits. Spec §§4–8 all depend on §3's outcome. Implementation plan's P-0 milestone runs the probe; remaining milestones (P-1 onward) proceed conditionally per the resolved sub-variant.

## 4. FFI signature extension

### 4.1 Extension shape — 2-param in all sub-variants

The FFI extension extends `nsl_flash_attention_csha` (and 5 sibling launch entry points: `_csha_with_saves`, `_csha_backward`, `_flash_attention`, `_flash_attention_quantized`, `_flash_attention_backward`) by appending **2 Tier-B-variant pointer parameters** at the end of the signature, **in all sub-variants**:

```rust
pub extern "C" fn nsl_flash_attention_csha(
    /* ... existing 36 params ... */
    tier_b_ptx_ptr: i64,   // sentinel (0, 0) = no Tier-B-on variant available
    tier_b_name_ptr: i64,  // populated when codegen emits Tier-B-on for this config
) -> i64 { ... }
```

The base `ptx_ptr, name_ptr` parameters continue to carry the **Tier-B-off PTX** (unchanged from PR #168/#169 behavior, equivalent to `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)`). The appended Tier-B params carry the **Tier-B-on variant** for `segment_masked` configs.

**Why 2-param in all sub-variants** (and not 4-param as a brainstorm intermediate considered): the Tier-B-off path is already carried by the base `ptx_ptr, name_ptr`. The Tier B extension only needs to add the Tier-B-on variant. The sub-variant choice (B-i / B-ii / B-ii-restricted) affects only the `TIER_B_MAX_BAKED_SEQ_LEN` constant the Tier-B-on PTX is sized for, not the FFI signature shape.

### 4.2 Sentinel semantics — structurally enforced via helpers (IR-001 discipline)

Sentinel construction is encapsulated in helper functions at the Cranelift-emitter boundary, NOT inlined as `0, 0` literals at call sites:

```rust
/// Sentinel pair indicating "no Tier B variant available."
/// Emitted by call sites whose codegen path doesn't produce a Tier B PTX blob
/// (non-segment_masked configs, or any path that pre-dates this spec).
fn tier_b_disabled_sentinel() -> [Value; 2] { /* emit two zero constants */ }

/// Populated pair carrying the Tier B variant pointer + kernel-name pointer.
/// Emitted by call sites whose codegen path produced the Tier B PTX blob.
fn tier_b_enabled(
    tier_b_ptx_data: DataId,
    tier_b_name_data: DataId,
) -> [Value; 2] { /* emit data-section address references */ }
```

The helpers are the **single edit point** for sentinel encoding. If the encoding ever changes (e.g., to a tagged-pointer scheme), only the helper bodies change; call sites are unaffected.

### 4.3 Dispatcher-side invariant (shared assertion helper)

The runtime dispatcher asserts both sentinels agree at function entry via a shared helper (not duplicated across 6 entry points — IR-001 single-edit-point property for assertion behavior):

```rust
/// Asserts the Tier B sentinel pair has both values either zero or both non-zero.
/// Panics if mismatched.
#[inline(always)]
fn assert_tier_b_sentinels(
    entry_point: &'static str,
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) {
    if (tier_b_ptx_ptr == 0) != (tier_b_name_ptr == 0) {
        eprintln!(
            "FATAL [{entry_point}]: tier_b_ptx_ptr={tier_b_ptx_ptr:#x} but \
             tier_b_name_ptr={tier_b_name_ptr:#x}; sentinel pair must agree \
             (both zero = disabled, both non-zero = enabled). Call site emitted \
             via inline literals instead of tier_b_disabled_sentinel() / \
             tier_b_enabled()?"
        );
        std::process::abort();
    }
}
```

Each entry point calls `assert_tier_b_sentinels("nsl_flash_attention_csha", ...)` at function entry. The assertion catches the helper-bypass case loudly; ~5 LOC × 6 entry points → 6 lines of helper calls + 15 lines of helper definition = 21 LOC total (vs 30 LOC of duplicated assertion code).

### 4.4 No metadata field — deferred per IR-009

The B-ii-restricted runtime range check uses a Rust-side compile-time constant (`TIER_B_MAX_BAKED_SEQ_LEN`) in the dispatch code path, NOT a runtime FFI field. The dispatcher knows the conservative-max value at compile time because the value was baked into the kernel's SMEM allocation at codegen time. Adding a runtime metadata field would replicate compile-time information at runtime — overhead without benefit.

Future extensions that need true runtime metadata (e.g., B-iii's HBM pointer to range tables, or telemetry fields for per-launch diagnostics) add the appropriate fields at extension time with specific semantics. The current FFI extension stays at the minimum the v1 sub-variant needs.

### 4.5 Header documentation discipline

The `nsl_flash_attention_csha` signature gains a header comment block documenting:

1. Which params are the Tier B extension (line range pinned).
2. The sentinel encoding (`(0, 0)`).
3. The dispatcher-side precondition (the §4.3 assertion).
4. The Cranelift-side construction discipline (use the `tier_b_*_sentinel()` / `tier_b_enabled()` helpers, never inline literals).
5. Citation to this planner spec + the V-Bii-SMEM findings doc.

Same audit-trail discipline as PR A.2.5's CSHA-extras documentation pattern. The Tier B extension explicitly inherits four properties from PR A.2.5's precedent:

1. **Sentinel default preserves no-op behavior** for callers that don't pass Tier B variant pointers.
2. **Parameters appended at the end** of `nsl_flash_attention_csha`'s signature; binary ABI preserved.
3. **Helper functions encapsulate sentinel construction** at call sites (strengthens PR A.2.5's pattern).
4. **FFI signature's header comment records the extension** — which params, what their sentinel encodes, what the dispatcher does with them.

### 4.6 Apply to 6 launch entry points

The extension applies uniformly to every `nsl_flash_attention*` extern function except `nsl_rope_cache_write`:

- `nsl_flash_attention:109`
- `nsl_flash_attention_csha:373`
- `nsl_flash_attention_csha_with_saves:597`
- `nsl_flash_attention_csha_backward:883`
- `nsl_flash_attention_quantized:1459`
- `nsl_flash_attention_backward:2107`

**`nsl_rope_cache_write` exclusion — structural basis:** Tier B's skip optimization is correctness-preserving for kernels that compute attention contributions affected by segment masking. The skip predicate's symmetric-correctness property (forward S[qt,kvt]=0 iff backward dS[qt,kvt]=0 — per dispatch spec §7.1 ↔ revision spec §3.2) applies to attention computation. RoPE cache writing computes rotation factors per token position; it has no `(qt, kvt)` pair structure, no segment masking interaction, and no skip opportunities. The exclusion is structural, not stylistic.

### 4.7 Backward-compatibility

- **Binary ABI preserved:** existing positional arguments don't shift; appended params have sentinel defaults; existing callers that aren't recompiled continue to work as Tier-B-disabled.
- **Source-level compatibility:** callers using `tier_b_disabled_sentinel()` helper have a one-line change at each call site; new callers using `tier_b_enabled()` opt in explicitly.
- **Test harness:** bench binary's `--tier-b={on,off}` flag is reachable by constructing the appropriate sentinel/enabled pair via the helpers.

### 4.8 LOC budget

| Component | LOC |
|---|---|
| FFI signature extension (6 entry points × ~4 lines each) | ~24 |
| Helper functions (2 helpers in Cranelift-emitter) | ~15 |
| Shared dispatcher-side sentinel assertion + 6 callers | ~21 |
| Header documentation (6 entry points) | ~80 |
| Cranelift-side call-site migrations (3 production sites) | ~6 |
| **Total FFI extension surface** | **~146 LOC** |

## 5. Codegen emission policy

### 5.1 Trigger — case (β-ii) collapse rehabilitated

The codegen-side dispatch toggle:

```rust
/// Codegen-time gate: should this config's emission include a Tier-B-on PTX variant?
/// Returns true iff config.segment_masked. Called from emit_tier_b_variants_for_config.
pub fn should_emit_tier_b_at_codegen(config: &FlashAttentionConfig) -> bool {
    config.segment_masked
}
```

The case-(β-ii) collapse from the dispatch spec's §14 amendment: at codegen, the heuristic reduces to `config.segment_masked`. The seq_len floor gate is applied at runtime by the launch wrapper, not here.

### 5.2 What codegen emits per config

| Config | Base PTX (existing `ptx_ptr, name_ptr`) | Appended Tier-B-on PTX |
|---|---|---|
| `segment_masked = true` | `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)` — Tier-B-off path; PR #168/#169 baseline | `synthesize_flash_attention_ptx_v2_with_tier_b(config, Some((MAX_BAKED, Tiled)))` — Tier-B-on path; sized for MAX_BAKED |
| `segment_masked = false` | Existing baseline | **Sentinel `(0, 0)`** — no Tier-B-on variant emitted |

The base PTX path is **unchanged from PR #168/#169 for all configs.** Only the new emission is conditional.

### 5.3 `TIER_B_MAX_BAKED_SEQ_LEN` constant

```rust
// In nsl-tier-b-constants (per §6.2 (β) commitment) OR per-crate fallback (α):

/// Conservative-max seq_len baked into Tier-B-on PTX SMEM allocation.
/// Resolved by the V-Bii-SMEM probe per §3.4. See findings doc at
/// docs/superpowers/specs/2026-05-XX-tier-b-bii-smem-probe-findings.md.
pub const TIER_B_MAX_BAKED_SEQ_LEN: u32 = <probe-resolved value>;

// Compile-time assertion: value must be probe-validated. Catches probe-investigation-
// outcome bypass at compile time.
const _: () = assert!(
    TIER_B_MAX_BAKED_SEQ_LEN == 4096
        || TIER_B_MAX_BAKED_SEQ_LEN == 8192
        || TIER_B_MAX_BAKED_SEQ_LEN == 16384,
    "TIER_B_MAX_BAKED_SEQ_LEN must be one of {4096, 8192, 16384} per V-Bii-SMEM \
     probe's five-outcome matrix; investigation-row outcomes require resolving \
     the probe anomaly before this constant is set."
);
```

Single-edit-point property (consistent with dispatch spec §7.2's `SegmentResidency::Tiled` literal).

### 5.4 Emission helper at the Cranelift-emitter boundary

Production callers (`kernel.rs:750, 1050, 1173` per V-dispatch-integration's tally) migrate from direct `synthesize_flash_attention_ptx_v2(&config)` calls to a new helper in `crates/nsl-codegen/src/pca_tier_b.rs` (new file):

```rust
pub struct TierBEmissionResult {
    pub base_ptx: Vec<u8>,
    pub base_kernel_name: String,
    /// Some iff config.segment_masked.
    pub tier_b_on_ptx: Option<Vec<u8>>,
    pub tier_b_on_kernel_name: Option<String>,
}

pub fn emit_tier_b_variants_for_config(
    config: &FlashAttentionConfig,
) -> TierBEmissionResult {
    let base_ptx = synthesize_flash_attention_ptx_v2_with_tier_b(config, None);
    let base_kernel_name = flash_attention_kernel_name_v2(config);

    let (tier_b_on_ptx, tier_b_on_kernel_name) = if should_emit_tier_b_at_codegen(config) {
        let tier_b_args = Some((TIER_B_MAX_BAKED_SEQ_LEN, SegmentResidency::Tiled));
        let on_ptx = synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b_args);
        let on_name = flash_attention_kernel_name_v2_tier_b_on(config);
        (Some(on_ptx), Some(on_name))
    } else {
        (None, None)
    };

    TierBEmissionResult { base_ptx, base_kernel_name, tier_b_on_ptx, tier_b_on_kernel_name }
}
```

Single edit point for the codegen emission policy.

### 5.5 Kernel-name distinctness — encode `MAX_BAKED` in suffix

The Tier-B-on PTX needs a distinct kernel-name from the base PTX so `cuModuleGetFunction` resolves each correctly. The suffix encodes `MAX_BAKED` to eliminate cross-PR / cross-architecture / debug-vs-release collision class:

```rust
pub fn flash_attention_kernel_name_v2_tier_b_on(config: &FlashAttentionConfig) -> String {
    format!("{}_tier_b_max{}", flash_attention_kernel_name_v2(config), TIER_B_MAX_BAKED_SEQ_LEN)
}
```

Example: `fa_v2_csha_train_block64_head_dim64_tier_b_max16384`.

**Three failure modes this prevents:**

- Cross-PR collision: v1's `_tier_b_max8192` and v2's `_tier_b_max16384` are distinguishable.
- Cross-architecture collision: if `MAX_BAKED` becomes per-architecture in future work, kernel names self-disambiguate.
- Debug-vs-release confusion: if a debug build uses different `MAX_BAKED`, kernel names reflect it.

Same IR-012 (measurement-infrastructure-contract) discipline applied to kernel-naming.

### 5.6 Cranelift-side data-section embedding

Each call site at `kernel.rs:750, 1050, 1173` calls `emit_tier_b_variants_for_config(&config)`, then:

1. Embeds `base_ptx` as a Cranelift data section under symbol `base_kernel_name` (existing pattern).
2. If `tier_b_on_ptx.is_some()`: embeds it as a second data section under `tier_b_on_kernel_name`.
3. At the call site that emits the FFI invocation, constructs the Tier B sentinel pair via §4.2 helpers:
   - `tier_b_disabled_sentinel()` if `tier_b_on_ptx.is_none()`
   - `tier_b_enabled(tier_b_on_ptx_data_id, tier_b_on_name_data_id)` otherwise

### 5.7 Binary-size impact

For each `segment_masked` config: ~2× PTX bytes (base + Tier-B-on co-resident). Non-`segment_masked` configs: 1× PTX (unchanged from PR #168/#169).

Estimating ~5-10 `segment_masked` kernels per binary × ~10-15 KB per Tier B PTX → **~100-300 KB total binary-size increase.** Bounded; non-trivial; documented.

### 5.8 Codegen-side test surface

Tests at `crates/nsl-codegen/tests/pca_tier_b_emission.rs`:

1. `emission_helper_returns_two_blobs_for_segment_masked` — assert both `base_ptx` and `tier_b_on_ptx` are `Some` non-empty.
2. `emission_helper_returns_one_blob_for_non_segment_masked` — assert `tier_b_on_ptx` is `None`.
3. `kernel_name_distinctness` — assert `name.ends_with("_tier_b_max{N}")` where N is `TIER_B_MAX_BAKED_SEQ_LEN`.
4. **Compile-time const assertion** on `TIER_B_MAX_BAKED_SEQ_LEN ∈ {4096, 8192, 16384}` (per §5.3) — catches probe-investigation-outcome bypass at compile time. Build-time guard.

## 6. Runtime dispatcher branch

### 6.1 The runtime gate

`should_dispatch_tier_b_at_runtime` lives in `crates/nsl-runtime/src/pca_tier_b_runtime.rs` (new file):

```rust
/// Runtime gate: should this kernel launch dispatch to the Tier-B-on variant?
///
/// Returns true iff ALL FOUR conditions hold:
///   1. Codegen emitted a Tier-B-on variant for this config (tier_b_ptx_ptr != 0).
///   2. The caller passed a non-null segment_ids pointer (segment_ids_ptr != 0).
///   3. seq_len is at or above the empirical profitability floor (seq_len >= TIER_B_SEQ_LEN_FLOOR).
///   4. seq_len fits the conservative-max baked into the Tier-B-on PTX
///      (seq_len <= TIER_B_MAX_BAKED_SEQ_LEN).
pub fn should_dispatch_tier_b_at_runtime(
    tier_b_ptx_ptr: i64,
    segment_ids_ptr: i64,
    seq_len: u32,
) -> bool {
    tier_b_ptx_ptr != 0
        && segment_ids_ptr != 0
        && seq_len >= TIER_B_SEQ_LEN_FLOOR
        && seq_len <= TIER_B_MAX_BAKED_SEQ_LEN
}
```

The `tier_b_name_ptr` parameter isn't read by this gate (the sentinel-agreement assertion from §4.3 has already verified both are zero or both non-zero before this gate runs).

### 6.2 Constant-source location

Codegen-side (`pca_tier_b.rs`) and runtime-side (`pca_tier_b_runtime.rs`) both consume `TIER_B_MAX_BAKED_SEQ_LEN` and `TIER_B_SEQ_LEN_FLOOR`. These MUST agree.

**Commitment:** **(β) shared crate `nsl-tier-b-constants`** is the v1 default. Both `nsl-codegen` and `nsl-runtime` depend on it; single source of truth at compile time.

**Fallback condition:** if pre-implementation dependency-graph inspection (P-2's 5-minute check: `cargo tree -p nsl-codegen | grep nsl-runtime` and vice versa) reveals that `nsl-codegen` already depends on `nsl-runtime` (or vice versa), option **(α) runtime-as-source** is acceptable as the dependency direction is already established. Document the inspection's finding in the implementation PR's description.

**Option (γ) (both declare; cross-crate test enforces) is rejected** on IR-001 grounds — convention-only enforcement decays.

### 6.3 Integration into `nsl_flash_attention_csha`

At the dispatcher's entry (after the §4.3 sentinel-agreement assertion):

```rust
pub extern "C" fn nsl_flash_attention_csha(
    /* ... existing 36 params ... */
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) -> i64 {
    // Step 1: assert sentinel agreement (§4.3 shared helper).
    assert_tier_b_sentinels("nsl_flash_attention_csha", tier_b_ptx_ptr, tier_b_name_ptr);

    // Step 2: pick PTX / kernel-name based on runtime gate.
    let (effective_ptx_ptr, effective_name_ptr) =
        if should_dispatch_tier_b_at_runtime(tier_b_ptx_ptr, segment_ids_ptr, seq_len as u32) {
            (tier_b_ptx_ptr, tier_b_name_ptr)  // Tier-B-on.
        } else {
            (ptx_ptr, name_ptr)  // Existing Tier-B-off baseline.
        };

    // Step 3: load module + dispatch as before, but using effective_ptx_ptr/name_ptr.
    #[cfg(feature = "cuda")]
    {
        let module = cuModuleLoadData(effective_ptx_ptr as *const _);
        let kernel = cuModuleGetFunction(module, effective_name_ptr as *const _);
        // ... rest unchanged ...
    }
}
```

The runtime branch is **2 lines at function entry**. Mirrors the existing `effective_heads` pattern at lines 414-418.

### 6.4 Apply to 6 launch entry points

Same integration pattern repeats at each of the 6 entry points from §4.6. Pattern is mechanically uniform; no entry-point-specific branching logic.

### 6.5 Module-cache implications

The runtime maintains a `cuModule` cache keyed on kernel name. Both Tier-B-off and Tier-B-on PTX blobs are cached separately under their distinct names (per §5.5's `_tier_b_max<N>` suffix).

**Realistic working set:** the runtime gate's decision depends on `seq_len` and `segment_ids_ptr`, both workload-stable. Production workloads pick Tier-B-on OR Tier-B-off consistently, not alternately. Module-cache working set is therefore **~1 module per kernel-config**, not 2.

The 2-modules-cached-per-config framing is an upper bound (both potentially in cache); the realistic working set is half that. **Edge case:** autotuning sweeps may temporarily hold both; sweeps are bounded in scope (typically <100 trials). **v2 trigger** (per §9): cache-eviction-related performance issues reported.

### 6.6 Telemetry — deferred per IR-009

v1 does NOT ship telemetry, even feature-gated. The atomic-counter machinery, counter naming, and exposure mechanism become public commitments once shipped.

**v2 triggers** (specific):

- **A production workload report requests dispatch observability.** Issue or feature request citing a specific debugging scenario where bench-binary-side measurement is insufficient.
- **A correctness investigation requires per-launch dispatch trace.** Bug report says "Tier B dispatched when it shouldn't have" or vice versa.
- **A performance investigation requires dispatch-rate measurement.** Workload's wall-time regresses with suspected dispatch-rate root cause.

**Not v2 triggers:** curiosity / general observability; future-spec-round-trip avoidance.

### 6.7 Runtime-side test surface

Tests at `crates/nsl-runtime/tests/pca_tier_b_dispatch.rs` cover the four-condition truth table:

| Test | tier_b_ptx_ptr | segment_ids_ptr | seq_len | Expected |
|---|---|---|---|---|
| `dispatch_tier_b_on_happy_path` | non-zero | non-zero | FLOOR | `true` |
| `dispatch_tier_b_on_at_max_baked` | non-zero | non-zero | MAX_BAKED | `true` |
| `no_dispatch_when_no_tier_b_emitted` | 0 | non-zero | FLOOR | `false` |
| `no_dispatch_when_no_segment_ids` | non-zero | 0 | FLOOR | `false` |
| `no_dispatch_below_floor` | non-zero | non-zero | FLOOR - 1 | `false` |
| `no_dispatch_above_max_baked` | non-zero | non-zero | MAX_BAKED + 1 | `false` |

Plus one integration test catching inverted-branch failure mode:

- `dispatch_branch_picks_base_ptx_when_runtime_gate_false` — constructs scenario with consistent sentinel pair, gate fires OFF for orthogonal reason (seq_len < FLOOR); verifies dispatcher selects base PTX, not Tier-B-on.

### 6.8 Cross-crate constant-agreement test

Under §6.2's (β) shared crate commitment, the test is moot (one source of truth). Under (α) fallback, a test at `crates/nsl-codegen/tests/tier_b_constants_agree_with_runtime.rs`:

```rust
use nsl_codegen::pca_tier_b::TIER_B_MAX_BAKED_SEQ_LEN as CODEGEN;
use nsl_runtime::pca_tier_b_runtime::TIER_B_MAX_BAKED_SEQ_LEN as RUNTIME;

#[test]
fn max_baked_constants_agree_across_crates() {
    assert_eq!(CODEGEN, RUNTIME, "Codegen bakes PTX for MAX={CODEGEN}; runtime checks seq_len <= {RUNTIME}.");
}
```

Either way, the divergence failure mode is structurally caught.

## 7. Floor derivation rehabilitation

### 7.1 Inheritance from the dispatch spec's D-2

The dispatch spec's D-2 milestone defines the floor-derivation measurement: a 7-point `seq_len ∈ {128, 256, 512, 1024, 2048, 4096, 8192}` sweep at `head_dim=64, batch=4, sparsity=50%, block_q=64, block_kv=64, segment-masked causal, sm_120`. Median-of-5 wall-time, Tier-B-on vs Tier-B-off; floor = smallest seq_len with `wall-time win ≥ 10%`. Findings doc + CSV per IR-012.

**Measurement protocol inherited unchanged from dispatch spec §6.** Only the binding-point of D-2's output moves; the measurement isn't redone.

### 7.2 Rehabilitated consumption layer

Original framing (dispatch spec, pre-amendment): the floor was consumed at codegen by `should_emit_tier_b(config, seq_len) -> Option<TierBArgs>`. After this planner spec's Option B commitment, the floor is consumed **at runtime** by `should_dispatch_tier_b_at_runtime` (§6.1).

The codegen-side `should_emit_tier_b_at_codegen(config) -> bool = config.segment_masked` deliberately does NOT consult the floor — emission is unconditional for `segment_masked` configs; the seq_len profitability gate fires at launch time.

### 7.3 Why the rehabilitation preserves measurement validity — evidence basis explicit

Two sub-claims compose the "wrong-ON case is benign" framing:

**(a) Tier B's overhead is bounded even at low skip ratio.** PR #169's measurement at sparsity=10% (the §4.3 sensitivity tier's lowest data point) recorded positive wall-time win. Tier B's per-tile skip-check overhead is ~4 instructions plus the predicate evaluation — small relative to the matmul work. **Directly measured per the §4.3 sensitivity tier** (NOT derived from saturating-curve interpolation).

**(b) The sparsity=50%-derived floor generalizes across workload-realistic sparsities.** This is an interpolation claim grounded in (a): if Tier B's wall-time win is monotonic across sparsities ∈ {10%, 50%, 90%}, then the seq_len-only profitability axis captures the relevant sensitivity. Workloads at other sparsities should see at least the floor's measured win at the same seq_len.

**(b) is an interpolation, not a measurement.** If a future workload reports the D-2-derived floor underperforming at a specific sparsity value, that surfaces a measurement gap, not a generalization failure. The v2 trigger (§9 #7) fires to re-derive the floor at the workload's sparsity, not to invalidate Tier B.

**No additional measurement is required for v1.** Adding a per-sparsity floor sweep at D-2 time would expand to 21+ points without measured workload demand.

### 7.4 Sequencing within the implementation plan

| Phase | Milestone | What runs |
|---|---|---|
| **P-0** | V-Bii-SMEM probe | Resolves `TIER_B_MAX_BAKED_SEQ_LEN` per §3 |
| **P-1** | D-2 floor derivation (inherited) | Resolves `TIER_B_SEQ_LEN_FLOOR` per dispatch spec §6 |
| **P-2** | Shared-crate creation + dependency-graph inspection | Pins (β) vs (α) per §6.2 |
| **P-3** | FFI extension (§4) + codegen emission (§5) + runtime dispatcher (§6) | Implementation per probe + floor resolution |
| **P-4** | Test surface + snapshot re-baseline + activation PR | Per §8 |

**P-0 and P-1 are parallel-independent** — V-Bii-SMEM doesn't read the floor; D-2 doesn't read MAX_BAKED.

**Explicit P-1 → P-2 synchronization:** Both findings docs must commit with non-investigation outcomes before P-2 begins. If either constant is in "investigation" state, P-2 cannot proceed.

### 7.5 Findings-doc citation discipline

Per §6.2's (β) commitment, the constants live in the shared crate `nsl-tier-b-constants`. The findings-doc citations are source comments **at the definition site** (in the shared crate), NOT at the re-export sites in `pca_tier_b.rs` and `pca_tier_b_runtime.rs`.

```rust
// In nsl-tier-b-constants/src/lib.rs:
/// Empirical seq_len floor (wall-time win ≥ 10% per dispatch spec §6).
/// Derived from docs/superpowers/specs/2026-05-XX-tier-b-floor-derivation-findings.md.
pub const TIER_B_SEQ_LEN_FLOOR: u32 = <D-2-resolved value>;

// In pca_tier_b.rs (codegen) and pca_tier_b_runtime.rs (runtime):
/// Re-exported from nsl-tier-b-constants. See that crate for findings-doc citations.
pub use nsl_tier_b_constants::{TIER_B_MAX_BAKED_SEQ_LEN, TIER_B_SEQ_LEN_FLOOR};
```

Same IR-002 (external references as one-time anchors) discipline.

### 7.6 D-2 worst-case-outcome handling — two sub-cases

**Case (a): wins are positive but sub-threshold (e.g., 5-8% across seq_lens).**

Tier B provides some benefit but doesn't clear the 10% acceptance bar. Response:

- **Investigation (~1 day):** re-run with stricter measurement protocol (more inner iterations, isolate JIT-amortization, control for power-state variance). Profile per-tile-skip-decision overhead.
- **If post-investigation wins still sub-threshold:** lower the acceptance bar in a separate spec amendment, OR re-defer per IR-009 (§3.4 revert semantics from B.1.5+B.2 spec applying — feature flag + 6-month decay timer).
- **`TIER_B_SEQ_LEN_FLOOR` is NOT set during investigation;** const assertion keeps the value unresolved.

**Case (b): wins are zero or negative across seq_lens.**

Tier B provides no benefit or actively hurts. **This contradicts PR #169's measurement outcome** (73% wall-time win at the gate fixture). The measurement disagreement is itself the load-bearing finding — something is architecturally different.

- **Stop planner spec implementation immediately.** Option B commitment depends on Tier B being measurably profitable at runtime, which D-2 has disproven.
- **Re-derive Option B viability** in a follow-on amendment. Possible causes: PR #169's measurement was context-specific (gate-fixture / sm_120 / CUDA 13.2); the FFI extension's runtime overhead eats the benefit; conservative-max baking eats per-tile efficiency.
- **Option B fallback:** retreat to Option A (substantial architectural revision cost) or C if shape-inference work has materialized.

### 7.7 Test surface contribution

D-2's findings doc is the input; no new test for the floor value itself. The §6.7 truth table verifies the floor's consumption at runtime. The const-agreement test from §6.8 ensures codegen-side and runtime-side floor values match.

## 8. Test surface composition

### 8.1 Test-surface inventory (organized by IR-006 distinct failure modes)

Eight distinct surfaces, each catching a distinct failure mode:

| # | Surface | Failure mode caught | Source |
|---|---------|---------------------|--------|
| 1 | **Kernel-output parity** (16 tests: 10 forward + 6 backward × dQ+dK+dV) | Tier-B-on emits incorrect kernel output | PR #169 (inherited unchanged) |
| 2 | **PTX snapshot tests** (re-baselined per §8.3) | Tier-B-on PTX content regressions | Dispatch spec §8 (re-baselined for Option B) |
| 3 | **Isolation snapshot stability** | Tier B emission machinery regresses | Dispatch spec §8.1 (inherited, expected-stable) |
| 4 | **Runtime gate unit tests** (6 tests from §6.7 truth table) | Runtime gate's four-condition logic regresses | New (this spec) |
| 5 | **Runtime dispatcher integration test** | Dispatcher's branch wiring inverts or short-circuits | New (this spec) |
| 6 | **FFI sentinel-discipline tests** (sentinel-pair-mismatch + helper-roundtrip) | Helper-bypass produces silent wrong-variant dispatch | New (this spec) |
| 7 | **Codegen emission-helper tests** (3 tests from §5.8 + const assertion) | Emission policy regresses; `MAX_BAKED` set to non-probe-validated value | New (this spec) |
| 8 | **Cross-crate constant-agreement test** | Codegen and runtime constants diverge | New (this spec, conditional on §6.2 (α) fallback) |

### 8.2 Surfaces 1, 3 — inherited unchanged

The PR #169 kernel-output parity tests (#1) compare kernel outputs at the bit-identical level — agnostic to PTX emission count, FFI parameter count, or runtime dispatch path. Under Option B, when the runtime gate fires Tier-B-on, the kernel output is bit-identical to Tier-B-off by construction (skipped tiles contribute zero).

The isolation tests (#3) call emission functions directly. They stay byte-stable under all sub-variants. If a `.snap.new` is generated, investigation precedes acceptance.

### 8.3 Surface 2 — PTX snapshot re-baselining (per IR-006 cascade discipline)

**Base PTX snapshot** stays unchanged from PR #168/#169. **New Tier-B-on PTX snapshots** per affected fixture (filename includes `_tier_b_max<N>` suffix per §5.5).

**Affected fixture set:**

- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32.snap`
- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_64_64_64.snap`
- `pca_backward_kernel_snapshot__backward_kernel_segment_masked_tier_b_on_causal_32_32_32.snap`

(Exact count varies by `MAX_BAKED` resolution. If MAX=4096 (B-i), PR #169's snapshots stay byte-identical; if 8192 or 16384, snapshots re-baseline with new SMEM allocation size.)

**Cascade narrative — explicit "should-stay-stable" enumeration:**

When `MAX_BAKED` changes from PR #169's baseline (MAX=4096), the PTX diff is localized to exactly three regions:

**Regions that CAN change (cascade target):**

1. `.shared seg_smem[N]` declaration's `[N]` byte count.
2. Range-table indexing offsets in the preamble (immediate operands).
3. `compute_range_table_bytes` constant in any SMEM size computation.

**Regions that MUST stay byte-identical (cascade-breakage indicators):**

1. S-compute path (matmul instructions, register allocations, MMA operand encodings).
2. Skip predicate evaluation (`setp.lt.u16`, `setp.gt.u16`, `or.pred` sequences).
3. Tile-skip branch (`@%p_skip bra ...` instruction sequence).
4. Post-skip body (matmul continuation for non-skipped tiles).
5. Softmax (exp/sum/divide instructions).
6. Writeback path (`st.global` instructions, predicate guards).

If snapshot diff bleeds into any should-stay-stable region, the re-baseline is structurally wrong. The implementer's PR description includes a region-by-region diff summary.

### 8.4 Surfaces 4, 5 — runtime tests

6 truth-table tests + 1 integration test from §6.7. Deliberately separate per IR-006 — gate-logic verification vs gate-into-dispatcher wiring.

### 8.5 Surface 6 — FFI sentinel-discipline tests

Two tests gated behind `bench-internal` Cargo feature:

1. **Sentinel-pair-mismatch test:** constructs an invalid sentinel pair, asserts the §4.3 assertion fires.
2. **Helper-roundtrip test:** verifies `tier_b_disabled_sentinel()` returns `(0, 0)`; verifies `tier_b_enabled(ptx, name)` returns `(ptx_addr, name_addr)`.

The bench binary's CLI gains `--verify-ffi-sentinels` subcommand.

### 8.6 Surface 7 — codegen emission-helper tests

Per §5.8: 4 tests (3 positive + 1 build-time const assertion).

### 8.7 Surface 8 — cross-crate constant-agreement test

Per §6.8 (conditional on (α) fallback).

### 8.8 Test surface — what's NOT covered

Four failure modes are deliberately NOT covered by v1's test surface:

- **Performance regression at runtime-dispatch overhead level.** A regression where Tier-B-on dispatch adds ~10 ns to launch overhead would still pass all 8 surfaces (correctness preserved). v2 trigger: workload report of regression.
- **Production telemetry correctness.** Per §6.6, telemetry isn't in v1.
- **Module-cache thrashing under autotuning.** Per §6.5. v2 trigger: cache-eviction-related performance issues reported.
- **Constant-value-vs-findings-doc divergence.** The const assertion (surface 7) verifies type-validity; the cross-crate test (surface 8) verifies cross-crate agreement. Neither verifies the source-code value matches what the V-Bii-SMEM and D-2 findings docs recorded. **PR-review discipline is the load-bearing v1 enforcement.** v2 trigger: a real bug surfaces from constant-vs-findings-doc divergence; build-time markdown-parsing assertion (~50 LOC) added then.

### 8.9 Activation PR test plan

The PR description's opening summary pins the three context decisions resolved during P-0 / P-1:

```markdown
## Tier B activation — context decisions

- **V-Bii-SMEM outcome:** [B-ii unrestricted / B-ii-restricted / B-i] per
  docs/superpowers/specs/2026-05-XX-tier-b-bii-smem-probe-findings.md.
  - TIER_B_MAX_BAKED_SEQ_LEN = <value>
  - FFI extension shape: 2-param (uniform across sub-variants)
- **D-2 outcome:** [clear pass / case-(a) sub-threshold / case-(b) Option B re-evaluation]
  per docs/superpowers/specs/2026-05-XX-tier-b-floor-derivation-findings.md.
  - TIER_B_SEQ_LEN_FLOOR = <value>
- **Constant-source location:** [β shared crate / α runtime-as-source] per §6.2 fallback decision.

## Test plan

- [ ] Build clean (`cargo build`)
- [ ] All 16 runtime parity tests pass (PR #169 inherited)
- [ ] 2 isolation snapshot tests stay byte-identical (no re-baseline)
- [ ] Affected PTX snapshots re-baselined per §8.3 cascade narrative
      (N snapshots, hand-verified diffs; region-by-region summary in PR description)
- [ ] 6 runtime gate unit tests pass (truth table)
- [ ] 1 runtime dispatcher integration test passes (branch wiring)
- [ ] 2 FFI sentinel tests pass under `bench-internal` feature
- [ ] 4 codegen emission-helper tests pass (including const assertion at build time)
- [ ] Cross-crate constant-agreement test passes (or N/A if (β) shared crate used)
- [ ] Constants in `nsl-tier-b-constants` match findings-doc values; both findings docs
      cited by path in PR description
```

### 8.10 LOC budget

| Surface | LOC |
|---------|-----|
| #1 (inherited PR #169 parity) | 0 (existing) |
| #2 (PTX snapshot re-baselines) | ~50 (snapshot diffs only) |
| #3 (inherited isolation) | 0 (existing) |
| #4 (6 runtime gate tests) | ~80 |
| #5 (1 integration test + helper) | ~30 |
| #6 (2 FFI sentinel tests + CLI flag) | ~60 |
| #7 (4 codegen helper tests + const assertion) | ~50 |
| #8 (1 cross-crate test, conditional on (α)) | ~15 |
| **Total new test LOC** | **~285** |

Test LOC is ~1.7× implementation LOC (~170 from §4.8). Reported observationally per IR-008 — **not a budget ceiling.** Symptom-based v2 triggers (duplicate coverage, speculative tests, over-specification of settled surfaces) are the right shape for future test-surface review, not ratio thresholds.

## 9. Migration triggers to v2

### 9.1 Inherited from dispatch spec §9

| Trigger | Source | v2 cost |
|---------|--------|---------|
| **TierBPolicy enum** (user-overrideable `ForceOn`/`ForceOff`) | Dispatch spec §9.1 | ~80 LOC |
| **Planner module above codegen** | Dispatch spec §9.2 | ~200 LOC |
| **Per-config `SegmentResidency` selection** | Dispatch spec §9.3 | ~80 LOC |

### 9.2 New triggers introduced by this planner spec

| # | Trigger | Source section | v2 cost |
|---|---------|----------------|---------|
| 1 | Tier B-extended for seq_len > 16K (path A: extend baked max) | §2.2 + §3.5 | ~30 min probe + 2 LOC |
| 2 | Tier B-extended for seq_len > 16K (path B: migrate to B-iii HBM-resident) | §2.2 + §3.5 + original 2026-05-02 spec §11 | ~500 LOC kernel architecture change |
| 3 | B-ii-restricted → MAX extension (re-probe at higher MAX) | §3.5 (B-ii-restricted ladder) | ~4 hr re-probe + 2 LOC |
| 4 | B-ii-restricted → B-iii migration (workload above MAX=8192) | §3.5 (B-ii-restricted ladder) | ~500 LOC kernel architecture change |
| 5 | Per-launch telemetry | §6.6 | ~30 LOC + atomic counter overhead |
| 6 | Module-cache strategy re-evaluation | §6.5 | TBD per workload symptom |
| 7 | Per-sparsity floor re-derivation | §7.3 | ~3 hr extended D-2 sweep + 1 LOC |
| 8 | D-2 worst-case case (b) — Option B re-evaluation | §7.6 | full Option B vs A vs C re-derivation |
| 9 | Markdown-parsing assertion infrastructure for constant-vs-findings-doc | §8.8 | ~50 LOC |
| 10 | Test-surface symptom-based review | §8.10 | varies per symptom |

### 9.3 Trigger entry points — what specifically fires each

Each trigger has a concrete entry point; curiosity / "wouldn't it be nice" doesn't qualify:

- **#1** (extend baked max): V-Bii-SMEM-resolved MAX provides <60% utilization at MAX=16384 AND workload requires seq_len > 16K.
- **#2** (B-iii migration from B-ii unrestricted): MAX=16384 utilization >85% AND workload requires seq_len > 16K.
- **#3** (B-ii-restricted MAX extension): probe-time headroom at MAX=8192 was <60% AND workload requires seq_len in (8192, 16384].
- **#4** (B-ii-restricted B-iii migration): probe-time headroom at MAX=8192 was >85% AND workload requires seq_len > 8192.
- **#5** (telemetry): workload report requesting dispatch observability, OR correctness/performance investigation requiring per-launch trace.
- **#6** (cache): measured cache-eviction-related kernel-load-latency dominating launch overhead.
- **#7** (per-sparsity floor): workload at sparsity ≠ 50% reports performance regression at the D-2-derived floor.
- **#8** (Option B re-evaluation): D-2 surfaces case-(b) outcome (zero or negative wins).
- **#9** (markdown-parsing assertion): real bug surfaces from constant-vs-findings-doc divergence.
- **#10** (test-surface review): specific symptom — duplicate failure-mode coverage, speculative test without recorded scenario, or test-LOC growth without corresponding implementation growth.

### 9.4 Expected demand sequence (not enforced ordering)

Triggers are independent per §9.3; any can fire at any time when its entry point is met. The sequencing below reflects expected demand patterns based on typical ML-research workload evolution, NOT spec-enforced ordering:

- **#1 / #3** (extend baked max) are most likely earliest — workloads moving to longer contexts hit these first.
- **#2 / #4** (B-iii migration) follow if extend-baked-max isn't feasible per the probe headroom.
- **#5** (telemetry) fires when production debugging needs surface.
- **#7** (per-sparsity floor) fires when a workload's actual sparsity differs significantly from 50%.
- **#8** (Option B re-evaluation) ideally never fires — load-bearing case-(b) safety net.
- **#9** (markdown-parsing assertion) fires only after a real bug materializes.

**Why include this section if triggers are independent?** Two institutional purposes: implementer mental model (which triggers are likely earliest); reviewer context for v2 PRs (whether a trigger's expected demand pattern matches reality).

Triggers #2 and #4 are kept distinct despite shared implementation cost — their upstream conditions (pre-migration baseline, workload-demand threshold, SMEM-headroom evidence) differ structurally. A v2 PR could legitimately address both in one PR; distinct triggers in §9 ensure reviewer context is unambiguous.

### 9.5 What's explicitly NOT a v2 trigger

Per IR-009 dead-code-lifecycle discipline:

- **Curiosity / "wouldn't it be useful"** for telemetry, additional probes, or expanded test surface.
- **Future-spec-round-trip avoidance** — speculative shipping of metadata fields, hook points, or extensibility surfaces.
- **Aggregate metrics crossing thresholds** without specific symptoms — test-LOC ratio compounding, cache size growing, kernel name length increasing.
- **"Best practice" generic improvements** — adding logging, adding documentation, adding tests for already-covered failure modes — without specific instance demonstrating the gap.
- **Performance-optimization desire without measured regression.** Wanting to reduce binary size, dispatch overhead, or module-cache pressure as a general optimization is NOT a v2 trigger. Triggers require measured workload regression against the current implementation. Distinct from curiosity because engineering-virtuous intuition has stronger pull; same institutional answer — measurement-anchored triggers only.

### 9.6 Cross-reference to institutional rules

- **IR-009 (dead-code lifecycle):** every trigger has specific entry point; bounded-dormancy clock from dispatch spec §14.5 (2026-11-13 decay cap) applies to the planner spec's deliverable.
- **IR-010 (measurement-gated decisions):** triggers #1 / #3 / #7 require new measurements before v2 work begins.
- **IR-011 (distinct test surfaces roll up into richer decision space):** triggers #1-#4 split the "Tier B can't handle longer contexts" v2 work into four distinct paths based on V-Bii-SMEM's outcome row and SMEM-headroom recording. Third instance of IR-011's institutional value (first: dispatch spec §13.3 sensitivity tier; second: this spec's §9 trigger-space split).
- **IR-013 (external-caller-context assumptions):** trigger #8's Option B re-evaluation may surface additional caller-context findings.

## 10. Implementation milestones

Inherited from §7.4 with concrete deliverables and gates:

| Phase | Milestone | Deliverable | Gate to next phase |
|-------|-----------|-------------|---------------------|
| **P-0** | V-Bii-SMEM probe (§3) | Probe kernel + 12-config sweep + findings doc with five-outcome-matrix result + SMEM-headroom %s | Non-investigation outcome row resolved; sub-variant pinned |
| **P-1** | D-2 floor derivation (inherited from dispatch spec) | 7-point seq_len sweep findings doc + CSV; `TIER_B_SEQ_LEN_FLOOR` value pinned | Non-case-(b) outcome (per §7.6); case-(a) gets investigation; case-(b) escalates |
| **P-2** | Shared-crate `nsl-tier-b-constants` creation | 5-min dependency-graph inspection; new crate per (β) or runtime-as-source per (α); both constants declared | Both constants compile-time validated by const assertions |
| **P-3** | Implementation (§§4–6) | FFI extension (6 entry points) + emission helper + runtime dispatcher + helper functions | See **P-3 gate criterion** below |
| **P-4** | Test surface + snapshots + activation PR (§8) | PTX snapshot re-baselines per §8.3 cascade; 2 isolation tests stay byte-stable; FFI sentinel tests; activation PR with context-decision pinning per §8.9 | PR description complete; reviewer approval |

### 10.1 P-3 gate criterion — runtime parity vs snapshot parity

P-3's gate criterion distinguishes:

- **Runtime parity tests** (16 tests inherited from PR #169): MUST pass at P-3's gate. Compare kernel outputs Tier-B-on vs Tier-B-off at byte level via execution; snapshot-independent.
- **Runtime gate unit tests** (6 from §6.7 truth table): MUST pass at P-3's gate.
- **Runtime dispatcher integration test** (1 from §6.7): MUST pass at P-3's gate.
- **Snapshot-based tests** (PTX snapshots, isolation snapshots): NOT in P-3's gate scope. P-4's responsibility — re-baselined per §8.3 cascade narrative after P-3's implementation lands.

P-3 gates on runtime correctness; P-4 gates on snapshot acceptance. Same IR-006 distinct-test-surfaces discipline as the dispatch spec's D-3/D-4 split.

### 10.2 Sequencing

**P-0 and P-1 are parallel-independent** with synchronization at P-2 entry. P-3 and P-4 strictly sequential.

### 10.3 Wall-clock budget

~4-5 hours measurement (P-0 + P-1, parallel) + ~1-2 days implementation (P-3) + ~0.5 day snapshot + PR review (P-4). Total ~3-4 days assuming parallel execution where indicated.

## 11. Risks

| # | Risk | Mitigation | v2 trigger if it fires |
|---|------|------------|------------------------|
| 1 | V-Bii-SMEM probe surfaces investigation row (non-monotonic) | §3.4.1 investigation procedure (5 steps); P-2 blocks indefinitely | Trigger #8 if investigation reveals architectural issue |
| 2 | D-2 worst-case case (b) — zero or negative wins | §7.6 case (b) procedure; stop and re-evaluate Option B | Same trigger #8 |
| 3 | Sentinel helper bypass | §4.3 dispatcher-side assertion catches at runtime; bench test verifies | None — assertion is structural |
| 4 | Constant-vs-findings-doc divergence (§8.8) | PR-review discipline; activation PR checklist requires findings-doc citation | Trigger #9 (markdown-parsing assertion) |
| 5 | Module-cache thrashing under autotuning (§6.5) | Realistic working set ~1 module per kernel-config | Trigger #6 |
| 6 | FFI ABI compatibility break | Sentinel default `(0, 0)` preserves no-op; binary ABI extended | None — backward-compatible by construction |
| 7 | Saturating-curve generalization across sparsities (§7.3 sub-claim (b)) | Floor at sparsity=50% conservative; wrong-ON bounded per PR #169 sensitivity tier | Trigger #7 |
| 8 | Snapshot cascade narrative breakage (§8.3) | 6 should-stay-stable regions enumerated; PR-description per-region diff summary | None — structural verification failure |
| 9 | **V-Bii-SMEM-vs-D-2 outcome mismatch** (joint-outcome where FLOOR > MAX_BAKED) | Const assertion in shared crate: `assert!(TIER_B_SEQ_LEN_FLOOR <= TIER_B_MAX_BAKED_SEQ_LEN, ...)`. Build-time guard catches at P-2's shared-crate creation. | Joint-outcome investigation may surface that probes need shared input (e.g., D-2's sweep dimensions bounded by V-Bii-SMEM's MAX). ~1 hour re-run. |

Risks #1, #2, #4, and #9 are the load-bearing ones. Risk #9 is the silent-failure mode where the FFI extension + codegen emission ships correctly but the runtime gate is **never satisfied** because `FLOOR > MAX_BAKED`. The const assertion makes this structurally impossible — same IR-001 discipline applied to joint-outcome validity.

## 12. References

- **Dispatch spec:** `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` — heuristic shape, residency default, snapshot discipline; §14 amendment is this spec's direct predecessor.
- **D-1 findings:** `docs/superpowers/specs/2026-05-14-tier-b-dispatch-integration-findings.md` — V-dispatch-integration outcome (α=0, β=3, γ=0).
- **V-planner-options findings:** `docs/superpowers/specs/2026-05-15-tier-b-planner-options-findings.md` — Option B viability + sub-variant question + V-Bii-SMEM-needed framing.
- **Original Tier B design:** `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` — §11 deferred-extension queue (cited for B-iii migration).
- **Revision spec:** `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` — original SMEM probe Phase 0 (V-Bii-SMEM inherits the discipline).
- **B.1.5+B.2 spec:** `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` — KEEP UNCONDITIONALLY measurement + sensitivity tier finding.
- **M2/M6 findings:** `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` — wall-time win 73.33% at gate fixture; saturating-curve finding.
- **Institutional rules:** `docs/wiki/institutional-rules.md` — IR-001 through IR-013.
- **V-Bii-SMEM findings doc** (to be created at P-0): `docs/superpowers/specs/2026-05-XX-tier-b-bii-smem-probe-findings.md`.
- **D-2 findings doc** (to be created at P-1): `docs/superpowers/specs/2026-05-XX-tier-b-floor-derivation-findings.md`.

**Code references:**

- `crates/nsl-codegen/src/pca_tier_b.rs` — new file; codegen-side toggle + emission helper + kernel-name helper.
- `crates/nsl-runtime/src/pca_tier_b_runtime.rs` — new file; runtime gate.
- `nsl-tier-b-constants/src/lib.rs` (or fallback per §6.2) — shared constants with findings-doc citations.
- `crates/nsl-runtime/src/flash_attention.rs:373` — `nsl_flash_attention_csha` and 5 sibling entry points; FFI extension target.
- `crates/nsl-codegen/src/pca_tilerange.rs:39` — existing `compute_range_table_bytes`; consumed by V-Bii-SMEM probe (unchanged by this spec).
- `crates/nsl-codegen/src/flash_attention.rs:142` — `FlashAttentionConfig` definition (unchanged; V-planner-options confirmed the no-seq_len invariant).
- `crates/nsl-codegen/src/compiler/kernel.rs:750, 1050, 1173` — three production callers migrating to `emit_tier_b_variants_for_config`.

### 12.1 Institutional-rules registry maintenance

As part of this spec's deliverable, `docs/wiki/institutional-rules.md` gains two new "Cited from" entries:

- **IR-008** (verification surface investment): this spec §8.10 as a citation showing the 1.7× test-LOC ratio as a concrete instance of IR-008's observational framing.
- **IR-011** (distinct test surfaces roll up into richer decision space): this spec §9.6 as the third instance (first: dispatch spec §13.3 sensitivity tier; second: this spec's §9 trigger-space split via V-Bii-SMEM outcome + headroom recording).

These additions happen in the same PR that lands this spec. Activation PR checklist (§8.9) verifies both citations are added correctly.

---

## Revision changelog

- **2026-05-15** — initial. Follow-on to dispatch spec's §14 amendment, grounded in V-planner-options findings (Option B viable; A architectural revision; C no roadmap). Commits to 2-param FFI extension shape (uniform across sub-variants); sub-variant resolved by V-Bii-SMEM probe with five-outcome decision matrix. Adds risk #9 (joint-outcome const assertion). Extends IR-008 and IR-011 "Cited from" lists. Implementation budget ~3-4 days across P-0 through P-4.
- **2026-05-15 (post-review)** — reconciled §2.1 and §3.4 outcome matrices: "MAX=4096 doesn't fit either" is now its own investigation row (B-iii becomes default), distinct from "Only MAX=4096 fits" (B-i at MAX=4096, valid). Prior framing conflated the two as "B-i" — structurally contradictory because B-i at MAX=4096 requires MAX=4096 to fit.
