# PCA Tier B — Planner-Side Dispatch Design

**Status:** Design (spec) — post-merge follow-up to PR #169 (Tier B forward + backward; KEEP UNCONDITIONALLY measurement outcome).
**Date:** 2026-05-14
**Owner:** bwiemz
**Builds on:**

- `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` (original Tier B design).
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` (§3.1 / §3.4 / §6.3 revision).
- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` (B.1.5 + B.2 with the measurement-validated outcome).
- `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` (the KEEP UNCONDITIONALLY measurement: 73.33% wall-time win, 0.875 skip ratio at gate fixture; saturating sensitivity curve).

**Codifies (or extends):** IR-003 specialization for external-system-state assumptions (candidate IR-013 if a second instance materializes).

---

## 1. Why this spec exists

PR #169 (merged 2026-05-14) shipped Tier B forward + backward kernel emission with measurement-validated correctness and performance:

- **§3.1 acceptance bar cleared** at the gate fixture: 73.33% wall-time win (7.3× margin over the ≥10% threshold), 0.875 skip ratio (2.2× margin over the ≥40% threshold).
- **§4.3 sensitivity curve is saturating** (50→90 step ¼ the magnitude of 10→50) with positive wall-time win across all tested sparsities → **no sub-threshold-hurt regime** → **no sparsity gate needed** for downstream planner-dispatch.
- **§7 backward parity passes** with bit-identical dQ + dK + dV across 6 M3 fixtures.

But the central toggle `should_emit_tier_b(config) -> bool` still returns `false` unconditionally. Nothing in production calls the Tier B path. PR #169's ~5K lines of code sit dormant — **bounded dormant** during the measurement-pending phase, per IR-009.

The B.1.5+B.2 spec §2.2 explicitly identified this as the downstream follow-up: *"Production callers activating Tier B. Today nothing calls `synthesize_flash_attention_ptx_v2_with_tier_b` with `Some(...)`. The planner-side dispatch decision (when to opt in to Tier B for a given config) is its own spec, downstream of the M2/M6 keep/revert decision. If keep, the dispatch spec follows."*

The keep decision landed. This spec is the dispatch.

The institutional discipline: **the no-op guarantee that PR #168/#169 established was load-bearing during the bounded-dormant state but is exhausted after the keep decision.** Continuing it indefinitely converts bounded dormancy into permanent dormancy — exactly the dead-code-without-removal-trigger anti-pattern IR-009 was codified to prevent. The right invariant evolution is to **activate** the heuristic, not to **preserve** dormancy. Redefining the no-op guarantee around the heuristic IS the activation.

## 2. Scope and out-of-scope

### 2.1 In scope (this spec)

- **Minimal heuristic** in `should_emit_tier_b`: `config.segment_masked && seq_len >= floor`. No user-facing knob; no planner module above codegen. The heuristic IS the policy.
- **V-dispatch-integration verification gate** (~30-45 min) before implementation, with α/β/γ failure-mode enumeration. Findings doc determines which downstream option (synthesizer-arg / FlashAttentionConfig-field / sparsity-only collapse) is correct.
- **Floor derivation measurement** (~1 hour, 7-point seq_len sweep at fixed sparsity=50%); findings doc pins the floor value.
- **No-op invariant redefinition:** 1-arg synthesizer == 2-arg synthesizer with heuristic's choice. Replaces the bounded-dormant invariant from PR #168/#169.
- **Snapshot re-baseline discipline:** 8 affected snapshots (4 forward + 4 backward) with per-snapshot diff + cascade narrative verification per IR-006.
- **`SegmentResidency::Tiled` as the heuristic's residency default** — what the M2/M6 measurements + 16 parity tests verified.
- **Off-path API contract:** 2-arg form's `Option<...>` remains the public override surface with three operationally distinct uses (parity tests, bench binary measurement, planner-level explicit override).
- **v2 migration triggers** pinned explicitly for both heuristic policy and residency default.

### 2.2 Out of scope (deferred)

- **TierBPolicy enum (heuristic + user override).** v2 migration trigger: workload-driven need with measured perf regression at the heuristic-determined floor, OR a specific user reports `ForceOn`/`ForceOff` requirement. Migration cost: ~80 LOC. Debugging needs are covered by the bench binary's `--tier-b={on,off}` flag, NOT a migration trigger.
- **Workload-aware sparsity estimation (planner module above codegen).** v3 if ever. Reserved for when Tier B has multiple variants (extended, checkpointed) that warrant central planning.
- **Per-config `SegmentResidency` selection.** Default stays at `Tiled` until measured >10% wall-time regression at a new workload triggers v2. Caller preference for debugging is NOT a trigger.
- **Tier B-extended for seq_len > 16 K.** Range tables grow past 8 KB; needs per-warp register-resident or HBM-resident tables. Per the original 2026-05-02 spec §11.
- **CTA-uniform predicate trade-off.** Currently per-warp. Per the original 2026-05-02 spec §11.
- **Tile-skip-aware backward checkpointing.** Per the original 2026-05-02 spec §11.

### 2.3 Why this scope is right-sized

The brainstorm's pattern across this decision cycle (heuristic shape, no-op invariant evolution, seq_len source, residency default) settled on **measurement-grounded defaults + explicit triggers for v2** at every junction. This is the smallest viable v1 surface that activates the Tier B feature.

## 3. Policy — minimal heuristic

### 3.1 The heuristic

```rust
fn should_emit_tier_b(config: &FlashAttentionConfig, seq_len: u32) -> Option<TierBArgs> {
    if config.segment_masked && seq_len >= TIER_B_SEQ_LEN_FLOOR {
        Some(TierBArgs {
            seq_len,
            residency: SegmentResidency::Tiled,
        })
    } else {
        None
    }
}
```

The `TIER_B_SEQ_LEN_FLOOR` constant is derived from measurement (see §6). The `SegmentResidency::Tiled` literal appears in exactly one location (see §7's single-edit-point property).

### 3.2 Why these signals — and why no others

The minimal heuristic uses exactly two signals: `config.segment_masked` (correctness precondition — without a segment mask, no skip opportunities exist) and `seq_len` (the only known sensitivity dimension where the floor matters).

**Signals deliberately NOT in the heuristic:**

- **Estimated sparsity.** §4.3.3 of the previous spec found the sensitivity curve is saturating with no sub-threshold-hurt regime. The heuristic doesn't need a sparsity gate because wrong-ON is benign.
- **`block_q × block_kv` (num_tiles).** The preamble cost is amortized per the floor sweep; baking num_tiles into the heuristic would prematurely optimize for the few-tile regime where the floor sweep already gates correctly.
- **`head_dim`, `gqa_group_size`, `causal`.** Orthogonal to skip mechanics. Tier B's correctness-preservation property (§7.1 of the prior spec) holds regardless of these.
- **SMEM-budget check.** PR #169's §11.4 fix (Direction-aware offset) keeps forward-only kernels at ~50 KB at the gate fixture dims — well under the 99 KB Blackwell cap. Adding a SMEM-budget guard pre-empts a failure mode that the architectural fix already prevents.

The principle: **measurement-grounded defaults, not plausibility-driven enrichment.** Each additional signal requires its own measurement to verify it helps; adding signals without measurement is the failure mode IR-003 prevents. v2 expansions add signals only when a measured workload regression justifies them.

### 3.3 Why minimal heuristic over user-overrideable or planner-module-above-codegen

The `§4.3.3` saturating-curve finding IS the load-bearing input that makes minimal heuristic the right choice. Without that finding, the heuristic would carry real wrong-ON risk and a user-overrideable shape (TierBPolicy enum with `ForceOn`/`ForceOff`) would be more defensible. The sensitivity tier paid for itself by enabling the lower-complexity v1 dispatch policy.

**This is the second time the sensitivity tier produced load-bearing institutional value.** First instance: enabling the "keep with sparsity gate" outcome in §10 of the previous spec. Second instance: enabling the minimal-heuristic dispatch here. Worth citing in IR-011's "Cited from" list when this spec lands — distinct test surfaces (the sensitivity tier specifically) rolled up into a richer decision space than single-surface evaluation would have permitted.

The planner-module shape (option 3 in the brainstorm — separate planner above codegen, consuming upstream workload context) is correctly deferred as v3. It would be architecturally pure but adds a public surface that v1 doesn't need. Reserved for when Tier B has multiple variants that warrant centralized planning.

## 4. No-op invariant evolution

### 4.1 The previous invariant (bounded by the measurement-pending state)

PR #168/#169's invariant: *"`synthesize_flash_attention_ptx_v2(config)` (1-arg form) produces byte-identical PTX to `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)` (2-arg form passed None)."* This was true unconditionally because `should_emit_tier_b` returned `false` unconditionally.

The invariant was load-bearing during the bounded-dormant state — it prevented production behavior changes while the keep/revert measurement was pending. The keep decision lands; the invariant's purpose is exhausted.

### 4.2 The new invariant (after this spec activates)

*"`synthesize_flash_attention_ptx_v2(config)` (1-arg form) produces byte-identical PTX to `synthesize_flash_attention_ptx_v2_with_tier_b(config, should_emit_tier_b_args(config, seq_len))` (2-arg form passed the heuristic's choice)."*

Concretely, after activation:

```rust
pub fn synthesize_flash_attention_ptx_v2(config: &FlashAttentionConfig, seq_len: u32) -> Vec<u8> {
    synthesize_flash_attention_ptx_v2_with_tier_b(config, should_emit_tier_b(config, seq_len))
}
```

(The signature changes from `(&config)` to `(&config, seq_len: u32)` if V-dispatch-integration's α-case fires; see §5 for the alternatives.)

### 4.3 What this means for production behavior

For `segment_masked` configs with `seq_len >= floor`:

- Old behavior: kernel emits Tier-B-off PTX.
- New behavior: kernel emits Tier-B-on PTX with the predicate-skip path active.

For ALL other configs (non-segment-masked, OR seq_len below floor):

- Old behavior: kernel emits Tier-B-off PTX.
- New behavior: kernel emits Tier-B-off PTX (unchanged).

**Kernel output bit-stability:** the PTX changes for segment_masked configs, but the kernel outputs (O for forward; dQ/dK/dV for backward) remain byte-identical to pre-activation. The PR #169 parity tier verified this: Tier-B-on outputs are byte-identical to Tier-B-off outputs by construction (skipped tiles contribute exactly zero).

The cascade narrative for snapshot re-baselining (per IR-006):

1. Heuristic returns `true` at fixture X (because `config.segment_masked = true` AND `seq_len >= floor`).
2. PTX synthesis gains the predicate-skip path (snapshot diff localized to the predicate block).
3. Kernel output stays bit-identical (parity tier assertion).

If any link breaks, the re-baseline is wrong.

### 4.4 Off-path API contract

The 2-arg form `synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b: Option<...>)` remains the public API for explicit dispatch control. The `Option<...>` parameter has three operationally distinct uses, ALL of which must remain callable from external code:

- `Some(args)`: force Tier B on with explicit arguments. Used by future planner module if migrated to v2's TierBPolicy.
- `None`: force Tier B off. Used by:
  - **Parity tests** — one side of the bit-identical assertion (`tier_b_on_output == tier_b_off_output`).
  - **Bench binary's `--tier-b=off` flag** — calls `_with_tier_b(config, None)` to measure Tier-B-off wall-time as the baseline for the §3.1 acceptance bar.
  - **Production debugging / measurement workloads** — explicit Tier-B-off behavior on a config where the heuristic would otherwise return `Some(...)`.

The heuristic returns one of `Some(args)` or `None` based on `config.segment_masked && seq_len >= floor`. The 1-arg form's behavior is **the heuristic's default**; the 2-arg form's `Option<...>` is **the explicit override**.

This pins the 2-arg form as a first-class public API rather than a test-only escape hatch. Future v2 migration to TierBPolicy has a clean integration point.

## 5. V-dispatch-integration verification gate

### 5.1 What this verifies

The minimal heuristic needs `seq_len` at codegen time. FlashAttentionConfig doesn't carry seq_len today (it's runtime-launched). Before pinning the seq_len source, verify the existing FA-2 codegen call-graph: who calls `synthesize_flash_attention_ptx_v2(config)` in production code paths, and what do they know about seq_len at that point?

This is a pre-implementation verification (IR-003) specialized to **external-system-state assumptions** — the caller behavior is "external" to this spec's design decisions. Picking the seq_len-source option without verifying caller behavior would substitute plausibility for evidence.

### 5.2 Failure-mode enumeration

Three sub-cases:

- **(α) All callers have seq_len handy at call time.** Caller invokes the synthesizer after parsing input shape; seq_len is known. **Outcome:** option 4 (add `seq_len` arg to `synthesize_flash_attention_ptx_v2`). Migration cost: ~50 call sites; mechanical.

- **(β) Some callers don't have seq_len at call time (early-compilation context).** Caller invokes synthesizer in a type-level context where runtime dimensions aren't bound. **Outcome:** either (i) those callers don't activate Tier B (callers in α do; callers in β stay no-op), or (ii) the heuristic collapses to sparsity-only (`config.segment_masked`). Decide based on what fraction of callers are α vs β.

- **(γ) Callers know seq_len structurally as a type parameter, not a value.** **Outcome:** option 3 (add `seq_len: u32` to FlashAttentionConfig). Requires also updating ~50 fixture construction sites.

### 5.3 Choice criteria — option 4 vs option 3 when both implementable

If V-dispatch-integration shows case (α) only, option 4 (synthesizer arg) is preferred over option 3 (FlashAttentionConfig field) on three grounds:

- **Type-system honesty:** seq_len is runtime-variable; adding it to a compile-time-constant config struct misrepresents the property. NSL's existing dynamic batch/seq handling does NOT encode these dimensions into config structs; option 3 would create an inconsistency.
- **Call-site visibility:** option 4 makes seq_len explicit at the call site. Option 3 hides it in the config; callers might construct configs with stale seq_len values.
- **Migration locality:** both require ~50 updates, but option 4's updates concentrate at synthesizer call sites (code that explicitly invokes the kernel). Option 3's updates distribute across fixture setup, test code, debugging utilities.

Option 3 is reserved for an architectural revision that wants to encode kernel dimensions more uniformly. v1 picks option 4 when case (α) fires.

### 5.4 Deliverable

Findings doc at `docs/superpowers/specs/2026-05-XX-tier-b-dispatch-integration-findings.md`. Records:

- Caller classification (α/β/γ counts + file:line evidence per caller).
- Outcome decision (option 2, 3, or 4 per §5.2).
- Conditional path for the floor-derivation measurement's scope (see §6.5).

The dispatch-policy implementation cites this findings doc by path.

### 5.5 Budget

30-45 minutes. Caller-classification is grep-and-read work; outcome mapping is mechanical.

## 6. Floor derivation measurement protocol

### 6.1 What the measurement establishes

The `TIER_B_SEQ_LEN_FLOOR` constant — the minimum `seq_len` at which Tier B's wall-time advantage is positive. Below the floor, the skip-check overhead exceeds the skip-payoff; above the floor, Tier B helps (per the sensitivity curve's saturating behavior at seq_len=4096).

The previous spec's M2/M6 measurements verified Tier B at the gate fixture (seq_len=4096); they didn't sweep seq_len. The floor's value isn't in existing measurements — it needs a new sweep.

### 6.2 Sweep dimensions

- **Sweep dimension:** `seq_len ∈ {128, 256, 512, 1024, 2048, 4096, 8192}` (7 points, powers-of-two from below to above the gate fixture's seq_len).
- **Fixed dimensions:** `head_dim=64, batch=4, sparsity=50%, block_q=64, block_kv=64`, segment-masked causal, sm_120. (Matches the gate fixture's dimensions except for varying seq_len.)

### 6.3 Measurement protocol

- **Median-of-5 wall-time, Tier-B-on vs Tier-B-off,** same protocol as §8 of the previous spec.
- **100 inner iterations per outer run.**
- **CUDA events bracketing the kernel** (consistent with §8.2 of the previous spec).
- **±10% reproducibility tolerance** (consistent with §8.5 of the previous spec).

### 6.4 Output

Per-seq_len wall-time win %. Then:

- **Floor = smallest seq_len in the sweep where wall-time win ≥ 10%** (matches the §3.1 acceptance bar threshold of the previous spec).
- If the sweep shows positive wall-time win at all tested seq_lens, **floor = 128** (admit Tier B for all `segment_masked` configs at seq_len ≥ 128).
- If the sweep shows a threshold in the middle, floor = the smallest seq_len above the threshold.
- If the sweep shows non-monotonic behavior, the heuristic may need refinement (e.g., `seq_len ∈ [low_floor, high_floor]` rather than `seq_len >= floor`). **Investigation rather than acceptance** — same discipline as §4.3.1 of the previous spec.

### 6.5 Scope conditional on V-dispatch-integration

The floor's empirical scope depends on V-dispatch-integration's outcome (§5):

- **If V-dispatch-integration surfaces case (α) only:** floor applies to all production callers. The 10% threshold at the floor is the universal dispatch criterion.
- **If V-dispatch-integration surfaces (α) + (β):** floor applies only to (α) callers. The (β) callers operate under sparsity-only collapsed heuristic (`config.segment_masked` alone).
- **If V-dispatch-integration surfaces (γ):** the seq_len mechanism shifts to FlashAttentionConfig; floor's scope is configs whose `seq_len` field is set.

**Sequencing:** V-dispatch-integration runs FIRST (30-45 min); floor derivation runs SECOND (~1 hour). The verification's outcome determines the floor's scope.

### 6.6 Deliverable

Findings doc at `docs/superpowers/specs/2026-05-XX-tier-b-floor-derivation.md`. Records:

- Per-seq_len wall-time win %.
- Derived floor value.
- Curve-shape inference (monotonic / threshold / non-monotonic).
- Cross-reference to V-dispatch-integration findings doc for scope.

The heuristic's source code references this findings doc as a comment beside `TIER_B_SEQ_LEN_FLOOR`.

### 6.7 Budget

~1 hour (7 measurements × ~5 minutes each at iterations=100 + ~30 min findings doc).

## 7. `SegmentResidency::Tiled` as the default — single-edit-point property

### 7.1 Why `Tiled`

What the bench binary used. All M2/M6 measurements + 16 parity tests (10 forward + 6 backward × dQ+dK+dV) verified Tier B works correctly with `Tiled`. The other variants are unverified at this scale. Picking `Tiled` is the measurement-grounded answer — **the heuristic emits what was measured.**

### 7.2 Single-edit-point property

The literal `SegmentResidency::Tiled` appears in exactly ONE location: the heuristic's `Some(...)` constructor in `should_emit_tier_b`. No branching on config; no fallback path to alternative variants.

```rust
Some(TierBArgs {
    seq_len,
    residency: SegmentResidency::Tiled,  // Default per §6.5 of M2/M6 findings doc.
    //         ^^^^^^^^^^^^^^^^^^^^^^^^
    // Single edit point. v2 migration: change this literal and add
    // per-config branching. ~80 LOC ceiling (mirrors TierBPolicy enum
    // migration estimate).
})
```

The source comment at the literal cites the M2/M6 measurement findings doc (`docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md`) as the empirical basis. Future readers see why `Tiled` is unconditional, not why this configuration was chosen at-random.

### 7.3 v2 trigger — specificity

The default stays at `Tiled` until a measured performance regression triggers v2 migration. Specifically:

- **Trigger:** new workload's M2/M6 measurements show `Tiled` underperforms an alternative variant (`Inline`, `Hbm`, or similar) by **>10% wall-time** at the same fixture. Matches the §3.1 acceptance bar threshold (consistent with IR-010 — measurement thresholds are pinned, not negotiated).
- **NOT a trigger:** caller preference for debugging or architectural comparison. The bench binary's `--tier-b={on,off}` covers this; if residency-variant comparison becomes a debugging need, add `--residency=<variant>` to the bench binary without touching the heuristic.
- **NOT a trigger:** correctness regression in `Tiled` at some configurations. This triggers a `Tiled` bugfix, not v2 architecture migration.

v2 migration cost: ~80 LOC for the residency parameter threading + per-variant testing. Same backward-compatible migration shape as the TierBPolicy enum trigger — adding a parameter doesn't break the default.

## 8. Snapshot re-baseline discipline

### 8.1 What changes

The 8 snapshots affected by activation (4 forward + 4 backward, for `segment_masked` configs at gate fixture dimensions and other test fixtures where seq_len >= floor) require re-baselining. The exact list:

- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32`
- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_64_64_64`
- `pca_tier_b_preamble_isolation__preamble_4k_block64` (range-table-base immediate; possibly stable since the preamble emit is unchanged)
- `pca_tier_b_predicate_isolation__predicate_4k_block64` (range-table-base immediate; possibly stable since the predicate emit is unchanged)
- The 4 backward equivalents (`pca_backward_kernel_snapshot__backward_kernel_segment_masked_tier_b_*`)

(The exact set may differ slightly once the heuristic is wired — the implementation phase determines the final list.)

### 8.2 PR-description discipline

Per IR-006 (distinct failure modes warrant distinct test surfaces), the re-baseline PR's description must include:

- **Per-snapshot summary:** old PTX section X bytes vs new PTX section Y bytes, with the dispatch-related additions highlighted. The new snapshot's added content is the predicate path; the rest of the PTX should be unchanged at the byte level.
- **Hand-derivation per fixture:** for each affected fixture, state the heuristic's expected output (`should_emit_tier_b` returns `Some(args)` or `None` at this fixture's `seq_len` vs floor). Verify the snapshot matches that expectation.
- **Cross-validation against parity tier:** for the 4 forward snapshots, the parity test must show bit-identical kernel outputs Tier-B-on vs Tier-B-off at the same fixture. This proves the PTX change is byte-altering but the output is byte-stable — the dispatch is correctness-preserving as the previous spec's §4.2 promised.
- **Reviewer verification:** the cascade narrative — *"heuristic returns true at fixture X"* → *"PTX gains predicate path"* → *"kernel output stays bit-identical"* — must be explicit. If any link breaks (e.g., snapshot diff shows changes outside the predicate path), the cascade isn't coherent and the re-baseline is wrong.

Same snapshot-re-baseline-review discipline as WGGO Phase 2 #134 and CSHA Tier B.1 (cited from IR-006).

### 8.3 What stays byte-identical

- All non-`segment_masked` snapshots (the heuristic returns `None`; old + new behavior identical).
- All snapshots with seq_len < floor (the heuristic returns `None`; old + new behavior identical).
- The Tier-B-off side of every parity assertion (it calls `_with_tier_b(config, None)` explicitly; the heuristic is bypassed).

## 9. Migration triggers to v2

The minimal heuristic is v1. Migration to a more elaborate dispatch policy (TierBPolicy enum, planner module, per-variant residency) happens when explicitly triggered.

### 9.1 TierBPolicy enum (user-overrideable heuristic)

**Triggers:**

- A specific user workload requires `ForceOn` or `ForceOff` beyond what the heuristic provides. Issue/feature request citing the workload's needs.
- The heuristic's measured floor proves wrong for a real workload (Tier B hurts at `seq_len > floor`). Performance regression report.

**NOT a trigger:** debugging/measurement use cases. The bench binary's `--tier-b={on,off}` flag covers these.

**Migration cost:** ~80 LOC (enum definition + field threading + three branch points in `should_emit_tier_b`). Backward-compatible — adding the field doesn't break existing call sites.

### 9.2 Planner module above codegen

**Triggers:**

- Tier B has multiple variants (extended for seq_len > 16K; checkpointed for tile-skip-aware backward) that benefit from centralized planning.
- Workload-aware sparsity estimation becomes load-bearing (e.g., a future workload's sparsity varies by run, and the planner needs runtime sparsity hints).

**Migration cost:** ~200 LOC (new module + integration). Architecturally pure; reserves for the multi-variant future.

### 9.3 Per-config `SegmentResidency` selection

**Trigger:** measured >10% wall-time regression at a new workload using `Tiled` (per §7.3).

**Migration cost:** ~80 LOC.

### 9.4 Dead-code lifecycle reminder

Per IR-009, each deferred option has an explicit trigger. None of them sit indefinitely without a removal/promotion event. The bench binary's `--tier-b` flag is the operational escape hatch for the debugging use cases that DON'T qualify as v2 triggers.

## 10. Outcomes matrix (after V-dispatch-integration)

After V-dispatch-integration runs and produces its findings doc, exactly one of these outcomes applies:

| V-dispatch-integration outcome | seq_len source | Heuristic shape | Implementation cost |
|---|---|---|---|
| **Case (α): all callers have seq_len** | Synthesizer arg (option 4) | `config.segment_masked && seq_len >= floor` | ~50 call site updates + heuristic body |
| **Case (β): some callers don't (mostly α)** | Synthesizer arg for α; no-op for β | Same as (α); β callers stay dormant | Same as (α) + β callers documented as no-Tier-B paths |
| **Case (β): some callers don't (mostly β)** | Sparsity-only collapse | `config.segment_masked` (no seq_len gate) | Heuristic body simpler; α callers also use sparsity-only |
| **Case (γ): seq_len is type-level, not value** | FlashAttentionConfig field (option 3) | Same as (α) but reads config.seq_len | ~50 fixture construction site updates + heuristic body |

The findings doc determines the row. The implementation phase follows the chosen row's path.

## 11. Risks and out-of-scope

### 11.1 Risks tracked

- **Saturating-curve generalization across seq_lens.** §4.3.3 of the previous spec verified saturating behavior at the gate fixture's seq_len=4096. The "wrong-ON is benign" property depends on this generalizing across the seq_len sweep range. **Mitigation:** §6's floor derivation explicitly verifies. If the sweep shows non-monotonic behavior, the heuristic may need refinement.

- **Case (β) split between callers.** If V-dispatch-integration surfaces (α) + (β), the (β) callers' dormancy means Tier B doesn't activate everywhere. Mitigation: §10's outcomes-matrix row makes this explicit; future planner work could un-dormify the (β) callers.

- **Snapshot re-baseline scope.** §8.1's list of 8 affected snapshots may differ from what implementation reveals. Mitigation: the implementation phase's first task is `cargo test --tests | grep FAILED` to surface the exact list; the PR description captures it.

- **Performance regression in case (γ).** Adding `seq_len: u32` to FlashAttentionConfig requires updating ~50 fixture construction sites; any mistakes could subtly break test parity. Mitigation: re-run all 16 parity tests (10 forward + 6 backward × dQ+dK+dV) after the migration; same byte-identical assertion catches drift.

### 11.2 Out-of-scope items (deferred to later work)

Per §2.2 — TierBPolicy enum, planner module, per-config residency selection, Tier B-extended for seq_len > 16K, CTA-uniform predicate trade-off, tile-skip-aware backward checkpointing. None of these are required to activate Tier B for the workloads PR #169 measured.

## 12. Implementation milestones

The work breaks into 4 sequential milestones.

### 12.1 Milestone table

| ID | Phase | Scope | ~LOC | Gate criterion | Gated by |
|---|---|---|---|---|---|
| **D-1** | Verification | V-dispatch-integration: classify callers (α/β/γ); findings doc. | ~0 code, doc only | Findings doc committed with caller classification + outcome decision. Budget 30-45 min. | — |
| **D-2** | Measurement | Floor derivation: 7-point seq_len sweep at sparsity=50%; findings doc. | ~0 code; ~50 LOC of measurement script reuse from B.1.5-5. | Findings doc committed with per-seq_len wall-time win %, derived floor value, curve-shape inference. Budget ~1 hour. | D-1 |
| **D-3** | Implementation | Implement minimal heuristic per V-dispatch-integration outcome (option 2/3/4 from §5.2). `should_emit_tier_b` body; signature changes; 1-arg wrapper. | ~50-100 LOC implementation + ~50 LOC call-site updates (if α or γ). | `cargo build` clean; all 16 parity tests pass; new heuristic returns expected values at gate fixture + parity fixtures. | D-1 + D-2 |
| **D-4** | Snapshots + PR | Re-baseline 8 snapshots per §8 PR-description discipline; ship activation PR. | ~0 code; snapshot diffs. | PR description includes per-snapshot diff, hand-derivation per fixture, cascade-narrative verification, cross-validation against parity. | D-3 |

### 12.2 Sequencing

D-1 → D-2 → D-3 → D-4 strictly sequential. The verification gate determines the implementation path; the measurement determines the floor value; the implementation produces the snapshot diffs; the PR ships them.

### 12.3 PR shape

Single PR for the activation. Title: `feat(pca-tier-b): activate planner-side dispatch (minimal heuristic; floor=N)`. The N is the empirical floor from D-2.

### 12.4 LOC budget (rough)

- D-1: 0 code; 1 findings doc (~1 page).
- D-2: 0 code beyond shell-script reuse; 1 findings doc (~1 page) + 7 CSV rows.
- D-3: ~50-150 LOC depending on outcome.
- D-4: 0 LOC; 8 snapshot updates.

Total new code: ~150 LOC (small relative to the verification + measurement infrastructure).

## 13. Institutional rules added or extended

### 13.1 IR-003 extension — external-system-state assumptions

This spec extends IR-003 (pre-implementation verification of load-bearing assumptions) with the **external-system-state specialization** surfaced during the V-dispatch-integration discussion. Most of IR-003's existing citations are about internal code-graph assumptions (NodeId space, ABI signatures, predicate operand symmetry). V-dispatch-integration is about caller behavior — structurally similar but external to the spec's own design surface.

**New "Cited from" entry on IR-003:**

> `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` §5 — V-dispatch-integration verification of caller behavior before pinning seq_len source. First instance of the external-system-state specialization (caller behavior is external to this spec's design surface; verification establishes facts about callers before the dispatch policy commits to a specific shape).

### 13.2 IR-013 candidate (deferred per IR-NNN entry criteria)

If a second instance of external-system-state verification materializes — e.g., during a future spec where hardware capability or runtime-value assumptions need pre-implementation verification — promote to **IR-013: External-system-state assumptions warrant the same pre-implementation verification discipline as internal-code-graph assumptions.** Until a second instance lands, fold this specialization into IR-003's "Cited from" list (per the registry's ≥2-instance entry criterion from Section 5.4 of `docs/wiki/institutional-rules.md`).

### 13.3 IR-011 extension — sensitivity tier's institutional value

The previous spec's §4.3 sensitivity tier produced load-bearing institutional value TWICE in this brainstorm cycle:

- First: enabled the "keep with sparsity gate" outcome in §10 of the previous spec.
- Second: enabled the minimal-heuristic dispatch policy here (without saturating-curve finding, user-overrideable shape would have been more defensible).

Worth a new "Cited from" entry on IR-011 (distinct test surfaces roll up into a richer decision space than single-surface evaluation):

> `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` §3.3 — sensitivity tier's saturating-curve finding (from previous spec) enabled minimal-heuristic dispatch policy (which would otherwise have required user-overrideable shape to manage wrong-ON risk).

## 14. References

- `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` — original Tier B design.
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` — revision design (§3.1 / §3.4 / §6.3 in-place).
- `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md` — B.1.5 + B.2 design with §4.3 sensitivity tier and §11.4 + §11.6 discoveries.
- `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` — KEEP UNCONDITIONALLY outcome (73.33% wall-time win, 0.875 skip ratio, saturating curve).
- `docs/superpowers/specs/2026-05-13-tier-b-b2-predicate-verification-findings.md` — V-B.2-predicate's case (β) findings.
- `docs/superpowers/specs/2026-05-13-tier-b-b15-3-skip-ratio-investigation.md` — case (α) investigation pattern that V-dispatch-integration mirrors.
- `docs/wiki/institutional-rules.md` — IR-001 through IR-012; this spec extends IR-003 and IR-011, candidate IR-013.
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs::should_emit_tier_b` — central toggle point (currently returns false unconditionally; this spec's D-3 milestone makes it heuristic-driven).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs::synthesize_flash_attention_ptx_v2_with_tier_b` — 2-arg public API (remains the override surface per §4.4).
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs::synthesize_flash_attention_ptx_v2` — 1-arg wrapper (signature may change in D-3 per V-dispatch-integration outcome).

---

## Revision changelog

- **2026-05-14** — initial. Post-merge follow-up to PR #169. Four brainstorm decisions resolved: policy shape (minimal heuristic), no-op invariant evolution (1-arg form == heuristic's choice), seq_len source (V-dispatch-integration gates options 2/3/4), SegmentResidency default (Tiled). IR-003 extension + IR-011 extension + candidate IR-013 deferred per registry entry criterion.
