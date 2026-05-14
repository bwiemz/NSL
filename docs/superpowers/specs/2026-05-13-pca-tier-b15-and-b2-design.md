# PCA Tier B.1.5 + B.2 — Design

**Status:** Design (spec) — post-merge follow-up to PR #168 (Tier B v1 forward)
**Date:** 2026-05-13
**Owner:** bwiemz
**Builds on:** `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` (original); `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` (§3.1 / §3.4 / §6.3 revision); `docs/superpowers/plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md` (v2 plan).
**SMEM probe finding referenced:** `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md` — Option 1 confirmed safe on RTX 5070 Ti / CUDA 13.2 / driver 591.86 on 2026-05-13.
**Codifies:** IR-009 (dead-code lifecycle), IR-010 (measurement-gated decision discipline), IR-011 (multi-fixture composition unlocks nuanced outcomes), IR-012 (measurement-infrastructure contracts).

---

## 1. Why this spec exists

PR #168 (merged 2026-05-13) shipped PCA Tier B v1 forward as opt-in dormant infrastructure: `synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b)` plumbs the predicate path end-to-end, but the existing 1-arg form `synthesize_flash_attention_ptx_v2(config)` passes `None`, so no production code path activates Tier B today. The non-Tier-B paths are byte-identical (no-op guarantee). The Phase 0 SMEM probe ran on RTX 5070 Ti and disproved the Blackwell ILLEGAL_ADDRESS concern that originally paused implementation.

Two phases were explicitly deferred at v1:

- **Tier B.1.5** — bench harness + M3 launch + M2/M6 measurement. The 6 `#[ignore]`'d parity tests in `pca_tier_b_m3_parity.rs` require a launch harness that doesn't yet exist; the measurement scripts `scripts/measure_tier_b_m{2,6}.sh` reference a `nsl-codegen-bench` binary that doesn't yet exist.
- **Tier B.2** — backward kernel integration. Mirror the forward's skip predicate into `ds_compute.rs`; replace the v1 single-warp approximation in M3 instrumentation with a real `owning_warp(qt, kvt)` mapping; ship backward kernel snapshot variant; preserve the no-op guarantee for non-Tier-B configs.

This spec covers both phases. It pins **what "Tier B is worth keeping" looks like** as a measurement-driven keep/revert decision with non-negotiable thresholds and explicit revert semantics, codifies the verification protocols, and resolves five brainstorm decisions surfaced during the 2026-05-13 re-brainstorm session (acceptance bar, fixture matrix, bench binary location, B.2 backward inheritance, M3 parity owning_warp).

The institutional discipline: **measurement-gated decisions specify the decision rules — pass thresholds, fail semantics, measurement protocol — before the measurement runs.** Post-measurement, only the data is examined; the decision rules are settled. This is what makes pre-implementation verification trustworthy. Codified as IR-010.

## 2. Scope and out-of-scope

### 2.1 In scope (this spec)

- **Acceptance bar** for the Tier B keep/revert decision: ≥10% wall-time win AND ≥40% kv-tile skip ratio, with pinned rationale and non-negotiability policy.
- **Three-tier fixture matrix:** ONE gate fixture (segment-masked causal, seq_len=4096, head_dim=64, batch=4, sparsity=50%) drives keep/revert; SIX existing M3-parity fixtures verify bit-identical correctness; THREE sensitivity fixtures at sparsities {10%, 50%, 90%} characterize the curve shape.
- **`nsl-codegen-bench` binary** at `crates/nsl-codegen/src/bin/bench.rs` with pinned invocation contract (output format, exit codes, `--seed`) and explicit migration triggers to a dedicated crate if/when v1.5 cost becomes justified.
- **M3 parity tier:** byte-identical numerics (skipped tiles contribute exactly zero, so Tier-B-on vs Tier-B-off outputs are bit-identical); round-robin `owning_warp(qt, kvt) = (qt * num_kv_tiles + kvt) % num_warps` for the parity-trace HBM writeback.
- **B.2 backward inheritance:** reuse forward's `emit_skip_predicate` and `emit_range_table_preamble` wholesale; pre-implementation V-B.2-predicate verification gates the reuse (Q-outer vs KV-outer iteration order check); partial-tile masking inherited from `csha_hooks::emit_segment_mask`.
- **M2/M6 measurement protocol:** median-of-5 identical runs at the gate fixture, CUDA events bracketing the FA2 kernel summed over 100 inner iterations, Tier-B-off via flag-flip on same binary, ±10% reproducibility tolerance.
- **Outcomes matrix:** three pass/fail outcomes (keep unconditionally / keep with sparsity gate / revert), each with pinned downstream actions.
- **Dead-code revert semantics:** if the keep/revert decision returns "revert," it means `should_emit_tier_b` returns false unconditionally + tests `#[ignore]`'d + SASS baselines retained, NOT git-revert PR #168. With a 6-month decay timer and explicit re-evaluation triggers.

### 2.2 Out of scope (deferred to later specs / PRs)

- **Production callers activating Tier B.** Today nothing calls `synthesize_flash_attention_ptx_v2_with_tier_b` with `Some(...)`. The planner-side dispatch decision (when to opt in to Tier B for a given config) is its own spec, downstream of the M2/M6 keep/revert decision. If keep, the dispatch spec follows; if revert, it doesn't.
- **Tier B-extended for seq_len > 16 K** (range tables > 8 KB; needs per-warp register-resident or HBM-resident tables). Out of scope per the original 2026-05-02 spec §11.
- **CTA-uniform vs warp-uniform predicate trade-off.** Out of scope per 2026-05-02 spec §11.
- **Tile-skip-aware backward checkpointing.** Out of scope per 2026-05-02 spec §11.
- **External nsl-codegen-bench consumers.** Today only NSL's own measurement scripts call the bench binary. If outside consumers materialize, migration to `crates/nsl-codegen-bench` (option 2 in §5.3) becomes justified.

### 2.3 Why the scope splits this way

The B.1.5 ↔ B.2 boundary is set by the keep/revert decision. B.1.5 produces the measurements that drive the decision; B.2 ships the symmetric backward only if B.1.5's measurements clear the thresholds. Running them as ONE PR rather than two has a real benefit (a single landing finalizes the feature's complete shape) but a real cost (if B.1.5 says "revert," the B.2 work was wasted). The pragmatic choice is to **execute B.1.5 first, gate B.2 on B.1.5's measurement outcome, ship both in one PR if both clear**, ship just B.1.5's findings + revert-state if not.

The single-PR-when-both-pass shape preserves reviewer context (the whole Tier B story in one place) without committing to B.2 effort under uncertainty. The PR description distinguishes the two phases for review clarity.

## 3. Acceptance bar (the load-bearing decision criterion)

### 3.1 Strict AND-conjunction with pinned thresholds

The Tier B feature is "worth keeping" if and only if:

- **M6 (median-of-5 wall-time):** the gate-fixture forward pass with Tier B enabled is **≥10% faster** than the same fixture with Tier B disabled (flag flip, same binary).
- **M2 (skip ratio from M3 instrumentation):** the kv-tile skip ratio (skipped tiles / total tiles) averaged over the same 100 inner iterations is **≥40%**.

The two conditions are AND-conjoined. Either alone is insufficient.

### 3.2 Why these thresholds, specifically (defense against post-measurement erosion)

The thresholds are not picked by feel; they're derived from the cost structure of carrying Tier B forward.

**Wall-time ≥10%:** SMEM-budget cost (~1.5 KB additional per-CTA when Tier B is admitted), kernel-complexity cost (additional verification surface that lives forever), and per-call-site PTX scope discipline (IR-007) combined exceed the maintenance threshold below which the win doesn't recover the investment. Below 10% wall-time win, Tier B is a marginal feature that costs more to maintain than it saves. The 10% threshold is the minimum win that justifies the ongoing maintenance burden of two kernel emission paths.

**Skip ratio ≥40%:** below 40% skip ratio, the kernel performs the skip-check overhead (~4 instructions per tile in the runtime preamble loop plus the per-tile setp / or.pred / @%p_skip_TB bra block) without proportional payoff. The skip-check itself has cost; 40% is the threshold where the skipped work amortizes the skip-check overhead.

**AND, not OR:** wall-time alone can hide structural issues — Tier B might happen to be faster for unrelated reasons (compiler-side spill differences in the additional preamble registers, etc.) without actually skipping tiles. Skip ratio alone can hide kernel issues — the skip logic might fire as expected but the kernel still be slow due to overhead the skip doesn't address. The AND-conjunction requires both **the structural property (skip ratio shows tiles are actually being skipped) AND the observable benefit (wall-time win)** to hold simultaneously. Either alone is insufficient evidence that Tier B is paying for itself.

### 3.3 Post-measurement adjustment policy (the load-bearing discipline)

The thresholds are **NOT negotiable in the keep/revert decision.** If measurements come in below threshold, the decision is REVERT (per §3.4), with a separate follow-on issue to investigate whether the threshold was wrong or the kernel was. Lowering thresholds to pass the measurement is the failure mode this gate exists to prevent — it would convert the gate from "decision support" to "post-hoc rationalization."

The three natural post-measurement temptations and what the spec rules out:

- **(α) Skip-ratio passes (≥40%) but wall-time falls short (e.g., 7%).** The temptation: "Tier B is doing its structural job; wall-time measurement noise / kernel overhead masks the win; lower the wall-time bar to 5%." **Ruled out.** Wall-time below 10% means the win doesn't recover maintenance cost regardless of whether the skip mechanism functions structurally. The threshold isn't measurement-noise-bounded; it's maintenance-cost-bounded.
- **(β) Wall-time passes (≥10%) but skip-ratio falls short (e.g., 30%).** The temptation: "Tier B delivers wall-time win; who cares about skip ratio; skip ratio is just a proxy." **Ruled out.** Wall-time win without skip ratio is suspicious — it suggests the win comes from something other than skipping (e.g., compiler-side register allocation incidentally improved). The AND-conjunction exists to require the structural property as evidence the win is from the mechanism Tier B claims.
- **(γ) Both fall short by small amounts (e.g., 8% wall-time, 35% skip ratio).** The temptation: "Both are close; failing the gate over small misses seems pedantic; relax 2-5 points on each." **Ruled out.** The thresholds are calibrated to the cost structure; "close to passing" still doesn't recover the maintenance cost. The decision is binary: either the thresholds are met or they aren't.

If post-measurement analysis suggests the threshold itself was wrong (e.g., production workloads have characteristics that make 8% wall-time genuinely sufficient because deployment scale amplifies the absolute time savings), that analysis lands as a **separate spec revision PR with explicit justification**, not as a measurement-time adjustment. The revision PR's review evaluates the new threshold; the original measurement remains valid against the original threshold.

This converts the AND-conjunction from a numeric gate into a discipline-protected gate. The numbers (10%, 40%) become defensible against negotiation because the justification is in-spec, not in measurement-author memory. Codified as **IR-010** (measurement-gated decision discipline).

### 3.4 "Revert" semantics — dead-code via feature flag, NOT git-revert

If the keep/revert decision returns "revert," the action is:

1. **`should_emit_tier_b(config, ...)` is modified to return false unconditionally.** The dispatcher continues to exist; the dispatch outcome is hardwired off.
2. **Tier B tests gain `#[ignore = "Tier B disabled per M2/M6 results 2026-XX-XX"]`** markers. Tests remain in tree but don't run.
3. **Per-variant SASS baselines retained** at their current paths. The baselines represent verified-working PTX/SASS; deleting them loses replication ability for future revival.
4. **M2/M6 findings doc updated** with the measurement values, the keep/revert outcome, and the disable decision.
5. **PR title:** `feat(pca-tier-b): disable Tier B dispatch; M2/M6 thresholds not met` (or `pass and B.2 ships`, conditional on outcome).

This is **NOT** git-revert PR #168. Git-revert removes code from history; the dead-code state retains the code and disables it. The distinction matters because:

- The skip-predicate emission code is well-tested and verified; future revival shouldn't pay re-derivation cost.
- The supported-matrix CSV, fixtures, and verification harness represent material work; deletion is permanent loss.
- The "future workload changes the calculus" recovery path is concrete: a new workload with different sparsity profile might re-justify Tier B, and the recovery is "flip the flag" rather than "re-implement from scratch."

#### 3.4.1 Dead-code decay policy (the 6-month timer)

Dead-code without an explicit removal trigger becomes permanent (codified as **IR-009**). Tier B dead-code remains until one of three triggers fires:

- **6 months elapsed without a revival proposal** → spec-level review evaluating whether maintenance cost exceeded option-value. If yes, git-revert PR #168 (move from dead-code to history-removed). The review is the explicit trigger; absent the review, dead-code persists.
- **A new workload appears that re-justifies Tier B** (e.g., long-context training with different sparsity profile, or an external benchmark requiring tile-skip optimization) → revival PR re-enables `should_emit_tier_b`; tests un-ignore; re-measurement runs against the new workload's characteristics. If the new workload's measurements clear the thresholds (possibly with adjustment via §3.3's separate revision PR if the original 10%/40% no longer applies), Tier B becomes active.
- **A CUDA toolkit / driver change breaks Tier B** (CI failure, ptxas rejection, runtime kernel error) → choice between fixing or git-reverting. If fixing cost exceeds the option value (6 months elapsed without revival proposal, no foreseeable workload), git-revert; if a revival proposal is in progress or imminent, fixing is justified.

The 6-month timer is calibrated to be long enough to revisit when a workload emerges (training research cycles are ~3-6 months for new model architectures) but not so long that dead-code rots beyond reasonable maintenance.

## 4. Three-tier fixture matrix

The brainstorm surfaced three distinct decision contexts that fixtures serve, each warranting a distinct fixture tier. Codified as **IR-011** (distinct test surfaces *roll up* into a richer decision space than single-surface evaluation would have permitted).

### 4.1 Gate tier — ONE fixture, drives keep/revert

**Fixture:** segment-masked causal attention, `seq_len=4096`, `head_dim=64`, `batch=4`, `sparsity=50%`. Block sizes `block_q=64`, `block_kv=64` (gate-fixture-specific; gate fixture is fully pinned).

**Purpose:** drives the binary keep/revert decision per §3.

**Acceptance:** ≥10% wall-time win AND ≥40% skip ratio per §3.1.

This is the only fixture whose pass/fail status decides the binary keep/revert outcome. Sensitivity tier (§4.3) can refine the outcome into "keep with sparsity gate," but absent that refinement the gate fixture is sufficient and sole.

### 4.2 Parity tier — SIX existing M3 fixtures, byte-identical correctness

**Fixtures:** the existing six `PackingFixture` entries in `crates/nsl-codegen/tests/fixtures/mod.rs`. These span the block_q / block_kv / head_dim matrix (32×32×32 to 64×64×64) that Tier B currently supports.

**Purpose:** verify Tier B is correctness-preserving across the supported configuration matrix. For each fixture, run the forward kernel with Tier B enabled and with Tier B disabled (flag flip, same binary, same input); compare outputs.

**Acceptance: bit-identical outputs.**

The bit-identical assertion (not tolerance-bounded) is correct on the load-bearing reasoning:

> Tier B's skip logic is correctness-preserving by construction — a skipped tile is one whose contribution is identically zero (segment mask says no token in the tile is in the relevant segment). Computing a zero-contribution tile vs skipping it produces identical floating-point results because the contribution is **exactly zero, not approximately zero.** The MMA accumulator merge with zero is a no-op at every floating-point precision.

**Tolerance-bounded equivalence (`max_abs_diff < ulp_tolerance`) is rejected** because it would silently accept reduction-order regressions. If a future change to Tier B re-orders MMA accumulator merges or introduces a different precision path, the byte-identical assertion fails — that's a real correctness signal worth investigating, not a tolerance bound to relax.

**Future variant note:** if a future Tier B variant is genuinely non-byte-identical by design (e.g., a different MMA tile shape that changes accumulation order), that variant's parity tier needs an explicit design-time decision to switch to tolerance-bounded, with explicit ULP justification. This is design-time, not measurement-time.

### 4.3 Sensitivity tier — THREE fixtures, characterizes the sparsity → benefit curve

**Fixtures:** the gate fixture's dimensions held constant, varying sparsity across `{10%, 50%, 90%}`. The 50% point is identical to the gate fixture (redundancy / cross-check).

**Purpose:** characterize Tier B's behavior across regimes, enabling the **nuanced "keep with sparsity gate" outcome** that single-fixture-gate would have foreclosed.

#### 4.3.1 Per-point purpose

- **10% sparsity (sub-threshold candidate):** few tiles are skippable; the skip-check overhead (~4 instructions per tile in the preamble loop + per-tile predicate evaluation) may exceed the skip-payoff. The 10% measurement answers: **"does Tier B actively hurt at low sparsity, or merely under-help?"** This is the diagnostic-rich point.
- **50% sparsity (gate condition):** identical to the keep/revert gate fixture by definition. Provides redundancy with the gate measurement; expected to match within the ±10% reproducibility tolerance (per §8). If the two 50% measurements diverge beyond tolerance, **investigation rather than acceptance** — drift indicates either hardware variation, driver variation, fixture randomness from different `--seed` values, or protocol drift.
- **90% sparsity (saturation regime):** most tiles are skippable; expected to show maximum Tier B benefit. Establishes the upper bound of "how good can this get."

#### 4.3.2 The sub-question the three points collectively answer

Is the sparsity → benefit curve **monotonic, linear, thresholded, or saturating?**

- **Monotonic-linear:** benefit scales smoothly from 10% to 90%; the gate fixture's 50% is on the curve; the result generalizes to intermediate sparsities by interpolation.
- **Thresholded:** benefit is zero or negative below some sparsity threshold (somewhere in (10%, 50%)), then rises sharply above. The gate fixture's 50% is above the threshold; future workloads below the threshold should NOT use Tier B regardless of the gate decision.
- **Saturating:** benefit plateaus above some sparsity (somewhere in (50%, 90%)); diminishing returns. Suggests the kernel has a maximum-helpfulness regime that future workloads can target.

The findings doc's "where does Tier B help most" characterization figure plots these three points, **names the inferred curve shape**, and documents whether the curve shape is consistent with the architectural reasoning (skip-check amortization theory in §3.2). If the curve is unexpected (e.g., non-monotonic, or hurt at 10%), that's a real finding worth investigating before keep/revert decision finalizes.

#### 4.3.3 Nuanced keep/revert outcomes unlocked by sensitivity tier

If the sensitivity curve shows Tier B helps at high sparsity but hurts or no-ops at low sparsity, the keep/revert decision has a third option beyond binary keep/revert:

- **Keep with sparsity gate:** `should_emit_tier_b(config)` returns true only when `estimated_sparsity(config) >= threshold`, where threshold is derived from the sensitivity curve (the sparsity above which Tier B's benefit is positive). Configurations below threshold dispatch to Tier A (or whatever the no-Tier-B path is).

This option exists **only if the sensitivity tier shows a clear threshold.** Without the sensitivity tier (single-fixture-gate option), the third option isn't available — the binary keep/revert decision absorbs whatever the gate fixture shows.

The sparsity-gate-threshold's specific value comes from the sensitivity curve, not from the spec. Spec pins the protocol; data pins the threshold.

**Note:** the planner's mechanism for computing `estimated_sparsity(config)` for an arbitrary config is out of scope here. It would land as part of the planner-side dispatch spec (§2.2's first bullet); if M2/M6 returns "keep with sparsity gate," that spec becomes necessary; if it returns "keep unconditionally" or "revert," it doesn't.

## 5. `nsl-codegen-bench` binary contract

### 5.1 Location: `crates/nsl-codegen/src/bin/bench.rs`

Verified precondition (`crates/nsl-codegen-cli` does not exist as of 2026-05-13). The single-file binary inside the existing codegen crate is the smallest viable v1 surface: picks up codegen's existing deps (cudarc, FlashAttentionConfig, synthesize_flash_attention_ptx_v2_with_tier_b) for free; no new Cargo workspace registration.

### 5.2 Invocation contract — output format, exit codes, reproducibility seed

**Output format (parseable, stable):** the bench binary emits ONE line to stdout per measurement, in this format:

```
tier_b_bench_result:fixture=<name>:tier_b=<on|off>:median_us=<float>:n=<int>:skip_ratio=<float>
```

Single line, key=value pairs, machine-parseable. Shell scripts grep for `tier_b_bench_result:` prefix. Adding new key=value pairs to the right of the existing pairs is backward-compatible; removing or renaming existing pairs is a breaking change requiring a `tier_b_bench_result:` prefix change to `tier_b_bench_result_v2:` etc.

**Exit codes:**

- `0`: measurement succeeded. Output line emitted to stdout.
- `1`: fixture not found / invalid CLI args. Error message to stderr. No output line.
- `2`: kernel launch error / CUDA error / ptxas error. Error message + stack trace to stderr.
- `3`: measurement framework error (timer setup failed, CUDA events failed to create, etc.). Error message to stderr.

Shell scripts distinguish exit codes: `0` is the only "use the data" outcome; `1-3` are abort-with-message.

**Reproducibility flag:** `--seed <u64>` sets the PRNG seed for any fixture-input randomness (random Q / K / V matrices, segment mask generation). Default seed: `42` for reproducibility across runs. The seed is emitted as part of the output line for traceability:

```
tier_b_bench_result:fixture=gate_4096:tier_b=on:median_us=234.567:n=100:skip_ratio=0.487:seed=42
```

**CLI surface (v1):**

```
nsl-codegen-bench --fixture <name> --tier-b={on,off} [--emit-time-only] [--seed <u64>] [--iterations <n>]
```

- `--fixture <name>`: required. One of the registered fixtures (gate, parity-1..6, sensitivity-10, sensitivity-50, sensitivity-90).
- `--tier-b={on,off}`: required. Sets `should_emit_tier_b` return value for this invocation.
- `--emit-time-only`: optional. Suppresses everything except the output line. Default: also emits human-readable progress to stderr.
- `--seed <u64>`: optional. Default 42.
- `--iterations <n>`: optional. Default 100 inner iterations (matches §8's measurement protocol). Override only for sensitivity analysis or debugging.

### 5.3 Migration triggers to `crates/nsl-codegen-bench`

The single-file v1 approach is right for now. Migrate to a dedicated crate when any of:

- **More than one binary needed.** If M2 and M6 measurement scripts each grow distinct binary entry points (vs. invoking the same binary with different flags), the lib+bin split is the natural Cargo shape.
- **Bench-specific deps that don't belong in `nsl-codegen`.** If the bench wants `criterion` for statistical analysis, `plotters` for findings-doc figures, or `serde_json` for richer output formats, adding these to `nsl-codegen`'s Cargo.toml pollutes the kernel-emission crate's deps.
- **External consumers want the bench infrastructure.** If outside contributors or workflow tools want to reuse the bench harness for their own kernels, the crate boundary makes packaging cleaner.

Migration is ~20 lines of Cargo.toml + workspace registration; the `bench.rs` file moves with minimal changes. Easier to migrate later than to start with a dedicated crate and discover the bench remains single-file forever.

### 5.4 Contract discipline

The output format / exit code / seed protocol are spec-pinned, not implementation-pinned. Shell scripts (`measure_tier_b_m{2,6}.sh`) and any future CI consumer encode these as integration assumptions. Convention-only enforcement decays; explicit contracts persist. Codified as **IR-012** (measurement-infrastructure contracts are explicit in the spec, not implicit in the implementation).

## 6. M3 parity tier — bit-identical numerics + round-robin owning_warp

### 6.1 Bit-identical parity assertion (per §4.2)

For each of the six existing `PackingFixture` entries, the bench binary computes outputs with Tier B on and off (flag flip, same binary, same input via `--seed`), and asserts:

```rust
assert_eq!(
    tier_b_on_output.as_bytes(),
    tier_b_off_output.as_bytes(),
    "Tier B output differs from Tier B-off at fixture {name} — \
     skip logic is no longer correctness-preserving"
);
```

Byte-equality, not float-equality with tolerance. Justification per §4.2 (skipped tiles contribute exactly zero).

### 6.2 Round-robin owning_warp mapping (replaces v1 warp-0 approximation)

**Formula:** `owning_warp(qt, kvt) = (qt * num_kv_tiles + kvt) % num_warps`

**PTX-level writeback predicate** (in `emit_skip_decision_writeback`, gated behind `debug_kernel_instrumentation` Cargo feature):

```ptx
// Derive warp_id from tid: warp_id = tid >> 5
mov.u32 %tid, %tid.x;
shr.u32 %warp_id, %tid, 5;

// Compute owning_warp for this (qt, kvt)
mul.lo.u32 %owning_warp, %qt, %num_kv_tiles_const;
add.u32 %owning_warp, %owning_warp, %kvt;
rem.u32 %owning_warp, %owning_warp, %num_warps_const;

// Writeback predicate: warp_id == owning_warp && (tid & 0x1F) == 0
setp.eq.u32 %p_warp_owner, %warp_id, %owning_warp;
and.b32 %lane, %tid, 0x1F;
setp.eq.u32 %p_lane_owner, %lane, 0;
and.pred %p_writeback, %p_warp_owner, %p_lane_owner;

@%p_writeback st.global.u8 [...], %skip_decision;
```

### 6.3 Round-robin vs warp-0 — calibrated comparison

The round-robin choice over the v1 warp-0 approximation is the more principled architectural shape, but the framing should not overstate the case.

**Round-robin's distinct benefit:** surfaces hypothetical bugs where warp partitioning is incorrect in a way warp-0 happens to mask. If warp 0 were incidentally the warp that correctly handles all `(qt, kvt)` pairs (e.g., a bug where other warps' partitioning is off-by-one), warp-0-always parity would pass while other warps silently produce wrong values. Round-robin tests would catch this because they attempt to write from a misconfigured warp.

**However:** this is a hypothetical bug class that hasn't been documented as a real instance in NSL's history. The benefit is speculative, not historical.

**Warp-0's distinct properties:**

- Simpler code (no warp-id derivation, no rem.u32).
- Identical correctness value for M3 instrumentation's purpose (debug-only parity verification, debug scale, single-fixture).
- If a future need for multi-warp instrumentation appears, migration from warp-0 → round-robin is non-trivial but not large (~20 lines of PTX).

**The decision stays at round-robin** because:

- "Speculative future benefit" plus "principled architectural property" plus "cheap implementation" is a defensible case for the more-principled option.
- The cost difference is ~10 additional PTX instructions inside a `debug_kernel_instrumentation`-gated section; negligible for both code volume and debug-build performance.
- Pre-empting the warp-partitioning-bug failure mode is cheap insurance for a debug-only feature.

The framing is documented in spec, not just in the decision. Future readers wondering "why round-robin?" see both the principled reason and the cheaper alternative considered.

### 6.4 Round-robin edge cases

The formula has degenerate cases worth pinning:

- **`num_warps == 1`:** `(...) % 1 == 0` for every tile; `warp_id == 0` is always true. Every tile is written by warp 0 lane 0 — identical behavior to the v1 warp-0 approximation. **Acceptable** — the round-robin's "distributed writes" benefit is moot at single-warp scale; the formula degrades gracefully.
- **`num_warps == 0`:** invalid configuration; should not reach this code path. If it does, `(...) % 0` is undefined behavior in most architectures. Add `debug_assert!(num_warps > 0, "Tier B M3 instrumentation requires num_warps > 0")` at the formula's emission site to make the invariant explicit. PTX-level: the `num_warps_const` value passed from the launcher is presumed non-zero (validated launcher-side).
- **`num_kv_tiles == 0`:** also invalid; no `(qt, kvt)` pairs to enumerate. The outer iteration in the bench binary's parity driver wouldn't execute, so the formula never evaluates. No additional guard needed.

### 6.5 Verifiability of round-robin distribution

The distribution property is **provable by construction from the formula**, not verified by a runtime test. `(qt * num_kv_tiles + kvt) % num_warps` produces a deterministic, even-modulo distribution. The formula's correctness is a static property documented in this spec.

If a future failure mode shows the distribution isn't actually even in observed behavior (e.g., warp scheduling collapses round-robin into effective serial writes due to a hardware-side optimization), a runtime check could be added — but absent that signal, the formula-level argument is sufficient.

## 7. B.2 backward inheritance

### 7.1 Symmetric correctness justification (the load-bearing reason reuse works)

A `(qt, kvt)` pair is **empty** if no query in `qt` attends to any key in `kvt` — a property of the segment mask alone, not of the kernel's direction.

For forward: `S[qt, kvt] = Q[qt] @ K[kvt]^T` is zero in the empty case (because every element is zero after segment masking). Therefore `O = softmax(S) @ V` contributions from this `(qt, kvt)` are zero.

For backward:
- `dS[qt, kvt]` is zero in the empty case (forward's `S[qt, kvt]` was zero; the chain rule's contribution from this `(qt, kvt)` is zero).
- `dQ` contribution via `dS[qt, kvt] @ K[kvt]` is zero.
- `dK[kvt]` contribution via `P[qt, kvt]^T @ dO[qt]` is zero (because `P[qt, kvt]` is zero).
- `dV[kvt]` contribution via `P[qt, kvt]^T @ dO[qt]` is zero (same reason).

Therefore **all** gradient contributions from a skipped `(qt, kvt)` are exactly zero. Skipping is correctness-preserving for forward AND backward by the same mathematical property.

This is what justifies reusing `emit_skip_predicate` wholesale in `ds_compute.rs` — the predicate's PTX-level emission is unchanged; only the call site moves from forward's `s_compute.rs` to backward's `ds_compute.rs`.

### 7.2 Partial-tile masking (the edge-case closure)

The skip predicate operates at **tile granularity.** A partial tile — a `(qt, kvt)` pair where SOME elements are in-segment and others are not — is **NOT skipped.** Partial tiles are admitted by the skip predicate and handled by **intra-tile segment masking** in the inner kernel.

For forward, intra-tile masking is in `csha_hooks::emit_segment_mask` and zeros out cross-segment attention contributions before the softmax accumulator. The masked-out elements contribute exactly zero to `S` (because of the mask multiplication), so subsequent operations propagate the zero through `softmax(S)`, `O = softmax(S) @ V`, etc.

For backward, the same masking must apply to `dS`, `dP`, `dQ`, `dK`, `dV` accumulations: cross-segment attention contributions are zero in forward, so their gradients are also zero in backward. The intra-tile masking is therefore **reused wholesale** from `csha_hooks::emit_segment_mask`, same as the skip predicate.

The "symmetric correctness" property is two-layered:

1. **Tile-level skip** (this spec): `emit_skip_predicate` reused, justified by all-zero tile contribution.
2. **Intra-tile mask** (existing CSHA hooks): `emit_segment_mask` reused, justified by partial-tile zero-contribution element-wise.

Both layers inherit the symmetric correctness property. An implementer reading `ds_compute.rs` sees both: the skip predicate at the tile-loop boundary (via reused `emit_skip_predicate`) and the intra-tile mask inside the admitted-tile body (via reused `emit_segment_mask` from CSHA).

### 7.3 V-B.2-predicate — pre-implementation verification (the load-bearing gate before B.2 code begins)

The reuse-wholesale recommendation depends on backward's iteration shape matching forward's at the predicate-evaluation point. If backward iterates a different outer/inner loop order, `emit_skip_predicate`'s operand order or SMEM addressing might need a transposed variant.

#### 7.3.1 What V-B.2-predicate verifies

Before B.2 implementation begins, verify backward's planned iteration order by reading `ds_compute.rs` and `flash_attention_v2/phases/backward/` modules. Pin one of two cases:

- **Case (α) — Q-outer, KV-inner (matches forward).** Same iteration order as forward. Same `(qt, kvt)` evaluation points. `emit_skip_predicate` reuses wholesale; no signature change.
- **Case (β) — KV-outer, Q-inner (FA-2 backward convention for dK/dV residency).** Iteration order swapped relative to forward. The predicate is still `(qt, kvt)`, but it's evaluated at different points in the kernel's control flow.

  - **Sub-question:** does the predicate function's operand order produce a uniform predicate at the evaluation point? The four range tables (`qtile_min/max`, `kvtile_min/max`) are symmetric in their addressing, but ptxas's uniformity tracking depends on the order of operands in the `setp.lt.u16` / `setp.gt.u16` instructions.

  - **Outcome if case (β):** `emit_skip_predicate` gains an `IterationOrder` parameter (enum `IterationOrder { QOuter, KVOuter }`), and the operand order branches at emission time. This is the documented reuse-with-parameter form, distinct from "duplicated function" or "wholesale reuse."

#### 7.3.2 Deliverable

Findings doc at `docs/superpowers/specs/2026-05-XX-tier-b-b2-predicate-verification-findings.md` (date filled at verification time). Records:

- Backward's actual iteration order (case α or case β) as observed in `ds_compute.rs` and related modules.
- The PTX-level predicate-emission shape (operand order + uniform-class register usage) that the iteration order implies.
- Whether `emit_skip_predicate` needs an `IterationOrder` parameter or reuses wholesale.

#### 7.3.3 Budget

~30 minutes (read FA-2 backward's existing loop structure to determine the planned iteration order; document findings). Cheap by intent — the verification's value is gate function, not deep analysis.

#### 7.3.4 Gate semantics

If V-B.2-predicate surfaces case (β), B.2 implementation **proceeds with the parameterized form** (not the wholesale-reuse form). The parameterization is a real code change but small; it's expected to add ~50 lines split between `pca_tilerange.rs` (predicate emission) and `ds_compute.rs` (call site).

If V-B.2-predicate surfaces case (α), B.2 implementation **proceeds with wholesale reuse** as the spec's primary path. No new public surface.

The gate is **not** a block on B.2 — both outcomes have planned paths. The gate's purpose is to **pre-empt the Q-outer-vs-KV-outer assumption mismatch** that would otherwise surface during implementation as "wait, the predicate doesn't fit here." Codified as instance of IR-003 (pre-implementation verification of load-bearing assumptions).

### 7.4 What B.2 ships (regardless of α/β outcome)

- `ds_compute.rs` gains the skip predicate at its tile-loop boundary, with the appropriate operand order per V-B.2-predicate's findings.
- `backward/prelude.rs` gains the range-table preamble (calling `emit_range_table_preamble` with backward's SMEM context); same shape as forward's prelude.
- Backward kernel snapshot variant: refresh `flash_attention_v2_backward_*` snapshots, including new Tier B-on variants.
- No-op guarantee: `synthesize_flash_attention_ptx_v2_backward(config)` (no `_with_tier_b` arg) produces byte-identical PTX as today, via the same `tier_b: Option<...>` parameter passing `None` from the 1-arg form. Identical pattern to forward's no-op guarantee from PR #168.
- Real `owning_warp(qt, kvt)` mapping in backward's M3 instrumentation (per §6.2; same formula as forward's).

### 7.5 What B.2 does NOT ship

- **Tile-skip-aware backward checkpointing.** Out of scope per 2026-05-02 spec §11.
- **A different `IterationOrder` parameter beyond α/β.** If a third iteration shape emerges, that's a separate design discussion.
- **Backward bench-binary fixtures.** The M2/M6 measurements are forward-only per §4 and §8 (forward is where the wall-time win matters most; backward's perf characteristics are downstream of correctness verification). If backward perf measurement becomes load-bearing later, add as a separate bench scope.

## 8. M2/M6 measurement protocol

### 8.1 Median-of-5 identical runs, gate fixture

For the M6 wall-time measurement:

- **Hardware:** RTX 5070 Ti (sm_120, Blackwell). Same hardware as the SMEM probe and the v1 Tier B verification.
- **Fixture:** the gate fixture per §4.1 (segment-masked causal, seq_len=4096, head_dim=64, batch=4, sparsity=50%, block_q=64, block_kv=64).
- **5 outer runs** of identical configuration on the same hardware.
- **100 inner iterations per outer run.** Each inner iteration is one forward kernel launch with CUDA events bracketing it. The outer run reports the sum of 100 inner iteration times.
- **Median of the 5 outer runs** is the reported `median_us` value.
- **Both Tier B-on and Tier B-off** runs follow this protocol. The 10% threshold compares Tier B-on median vs Tier B-off median.

### 8.2 Wall-time methodology — CUDA events

Wall-time is measured using `cudaEventRecord` + `cudaEventSynchronize` + `cudaEventElapsedTime` bracketing the FA2 kernel launch:

```rust
let start = unsafe { cuEventCreate_v2(...) };
let stop  = unsafe { cuEventCreate_v2(...) };
for _ in 0..100 {
    unsafe { cuEventRecord(start, stream); }
    launch_forward(...);  // includes Tier B preamble + tile loop
    unsafe { cuEventRecord(stop, stream); }
    unsafe { cuEventSynchronize(stop); }
    let ms = unsafe { cuEventElapsedTime(start, stop) };
    total_us += ms * 1000.0;
}
median_us = total_us / 100.0;  // for this outer run
```

The 100 inner iterations reduce launch-overhead noise; the median-of-5 outer runs reduce wall-clock-variance noise. CUDA events measure GPU-side time, not CPU-side time, which excludes Rust-side overhead like the bench binary's PRNG calls.

### 8.3 Tier-B-off comparison — flag-flip on same binary

The Tier B-off comparison is **the same binary** with `should_emit_tier_b` returning false, not a separately-compiled binary without the Tier B PTX-emission code. This isolates the kernel-emission difference from any unrelated binary differences (different optimization decisions, different Rust monomorphization, etc.).

Implementation: bench binary inspects `--tier-b={on,off}` argument and calls either `synthesize_flash_attention_ptx_v2_with_tier_b(config, Some(...))` or `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)`. Both code paths exercise the same Rust binary; only the PTX synthesis call differs.

### 8.4 Skip ratio measurement

For M2 (skip ratio), the bench binary enables `debug_kernel_instrumentation` (compiled-in unconditionally for the bench binary; not the default for the production codegen crate's other consumers). The kernel writes per-tile skip decisions to an HBM buffer per §6.2's `emit_skip_decision_writeback`. The bench binary reads the buffer back at the end of each inner iteration:

- `total_tiles = num_q_tiles * num_kv_tiles` per inner iteration.
- `skipped_tiles = sum(skip_decisions == 1)` from the HBM buffer.
- `skip_ratio = skipped_tiles / total_tiles` per inner iteration.
- `skip_ratio` averaged over the same 100 inner iterations as the wall-time measurement, then reported as `skip_ratio=<float>` in the output line.

The skip ratio's 100-inner-iteration averaging is **not** independent of the wall-time's 100 inner iterations — they're the same 100 iterations of the same fixture. This co-measurement ensures the two metrics describe the same kernel runs, not independent statistical samples.

### 8.5 Reproducibility tolerance ±10%

A second reviewer (or future re-measurement) running the same protocol on equivalent hardware should produce values within ±10% of the original measurement. Drift beyond that range indicates either:

- Hardware variation (different RTX 5070 Ti units; different ambient conditions; thermal throttling).
- Driver / toolkit variation (CUDA 13.2 → 13.3 ptxas heuristics).
- Protocol drift (different inner-iteration count; different fixture seed; different binary).

Drift beyond ±10% triggers **investigation rather than acceptance.** The investigation's outcome determines whether the original measurement was outlier-anomalous or whether the second measurement reveals a real environmental issue.

### 8.6 Reproducibility seed

The bench binary's `--seed <u64>` flag (default 42) controls all fixture-input randomness. Two invocations with the same `--seed` produce byte-identical input matrices. A reviewer attempting to reproduce a measurement uses the same seed value as the original; the output line emits the seed for traceability.

## 9. Institutional rules added by this spec

Four rules are codified during this spec; full text in `docs/wiki/institutional-rules.md`:

- **IR-009 — Dead-code lifecycle requires explicit removal triggers.** Dead-code (feature-flag-disabled but not removed) without an explicit removal trigger becomes permanent; specs that deprecate features via feature flag pin a decay timer + revival triggers + breakage-trigger semantics so dead-code has a finite lifetime.

- **IR-010 — Measurement-gated decisions specify pass thresholds, fail semantics, and measurement protocol before measurement runs.** Decisions deferred until after measurement risk post-hoc threshold negotiation; pinning all three at design time converts the gate from "decision support" to "data-driven decision." Post-measurement only the data is examined.

- **IR-011 — Distinct test surfaces roll up into a richer decision space than single-surface evaluation.** Multi-fixture compositions (gate / parity / sensitivity tiers in this spec; V1/V2/V3 in CSHA Tier B.1) enable nuanced outcomes (e.g., "keep with sparsity gate" emerging from sensitivity tier) that single-surface evaluation forecloses. The composition is the load-bearing property, not the multiple surfaces themselves.

- **IR-012 — Measurement-infrastructure contracts are explicit in spec, not implicit in implementation.** Shell scripts and CI configurations encode measurement-binary contracts (output format, exit codes, fixture names) as integration assumptions. Pinning the contract in spec — output format with stable prefix, exit code semantics, reproducibility seed — prevents future refactors from silently breaking downstream consumers.

## 10. Outcomes matrix (three possible keep/revert results)

After M2/M6 measurements run on the three-tier fixture matrix, exactly one of these outcomes applies:

| Outcome | Trigger | Action |
|---|---|---|
| **Keep unconditionally** | Gate fixture clears both thresholds (≥10% wall-time AND ≥40% skip) AND sensitivity curve is monotonic-linear or saturating (no sub-threshold hurt) | Planner-side dispatch spec follows. `should_emit_tier_b` is unconditional-true when Tier B's preconditions match. Ship B.2 in same PR. |
| **Keep with sparsity gate** | Gate fixture clears both thresholds AND sensitivity curve shows a thresholded shape (10% sparsity hurts or no-ops) | Planner-side dispatch spec adds `estimated_sparsity(config) >= threshold` predicate, threshold derived from sensitivity curve. Ship B.2 in same PR with sparsity-gated dispatch. |
| **Revert** | Gate fixture fails either threshold (regardless of sensitivity outcome) | Dead-code per §3.4. `should_emit_tier_b` returns false unconditionally. B.2 does not ship. PR title: "disable Tier B dispatch; M2/M6 thresholds not met." 6-month decay timer begins. |

The three outcomes are mutually exclusive and exhaustive. Each outcome has a pinned action; no outcome leaves work in an undefined state.

The "keep with sparsity gate" outcome is the **distinct value of the sensitivity tier** — single-fixture-gate would have collapsed the first two outcomes into one (lose the sparsity-curve information).

## 11. Implementation milestones

The work breaks into nine milestones across two phases. B1.5 milestones (1–6) execute sequentially; B2 milestones (1–3) are conditional on B1.5-6's outcome being keep or keep-with-sparsity-gate per §10.

### 11.1 Milestone table

| ID | Phase | Scope | ~LOC | Gate criterion | Gated by |
|---|---|---|---|---|---|
| **B1.5-1** | B.1.5 | `nsl-codegen-bench` binary scaffolding: CLI surface (`--fixture`, `--tier-b`, `--seed`, `--iterations`, `--emit-time-only`), output line format with `tier_b_bench_result:` prefix, exit codes (0/1/2/3) per §5.2. | ~150 (Rust) | Binary builds + runs against a hand-picked fixture; emits a well-formed output line on stdout matching the §5.2 contract; honors `--seed` reproducibly across two runs. | — |
| **B1.5-2** | B.1.5 | Gate fixture wiring: pin the gate fixture (segment-masked causal, seq_len=4096, head_dim=64, batch=4, block_q=64, block_kv=64, sparsity=50%) in a registered fixtures module; bench binary loads it by name; both `--tier-b=on` and `--tier-b=off` paths execute end-to-end. | ~50 (Rust) | Bench binary `--fixture gate_4096 --tier-b=on` and `--tier-b=off` both produce output lines; the two outputs differ only on the `tier_b=` field and `median_us` / `skip_ratio`. | B1.5-1 |
| **B1.5-3** | B.1.5 | Parity tier: un-`#[ignore]` the six `pca_tier_b_m3_parity.rs` tests; route them through the bench binary's launch harness; assert bit-identical outputs (per §6.1) for each fixture. Implement `emit_skip_decision_writeback` HBM readback in bench harness for skip_ratio extraction. | ~100 (Rust) | All six parity tests pass with `assert_eq!` byte-equality on Tier-B-on vs Tier-B-off outputs. | B1.5-2 |
| **B1.5-4** | B.1.5 | Sensitivity tier: three fixtures at sparsities {10%, 50%, 90%} sharing the gate fixture's other dims; sensitivity-50 is structurally identical to the gate fixture (cross-check redundancy per §4.3.1). | ~50 (Rust) | Bench binary produces output lines for `sensitivity-10`, `sensitivity-50`, `sensitivity-90`; sensitivity-50's `skip_ratio` matches gate fixture's within ±10% per §4.3.1's tolerance. | B1.5-2 |
| **B1.5-5** | B.1.5 | Implement `scripts/measure_tier_b_m2.sh` and `scripts/measure_tier_b_m6.sh` per §8: invoke bench binary 5× outer × 100 inner per measurement; parse output lines; emit findings-doc-ready CSV with median_us + skip_ratio per fixture. | ~100 (shell) | Scripts run end-to-end against the gate + sensitivity fixtures; produce CSV; exit cleanly. Re-running with same `--seed` produces values within ±10% (§8.5). | B1.5-1..4 |
| **B1.5-6** | B.1.5 | Run M2 + M6 measurements on RTX 5070 Ti; write findings doc; apply the §10 outcomes matrix decision. Outcome is one of: **keep unconditionally**, **keep with sparsity gate**, **revert**. Per-decision actions: keep → planner-dispatch spec deferred but B2 milestones unblocked; keep-with-sparsity-gate → planner-dispatch spec deferred (with sparsity-threshold value from sensitivity curve) and B2 milestones unblocked; revert → `should_emit_tier_b` returns false unconditionally per §3.4, B2 milestones skipped, dead-code 6-month timer begins (IR-009). | ~30 (Rust + docs) | Findings doc at `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` committed with raw measurement values + outcome row + decision rationale. | B1.5-5 |
| **B2-1** | B.2 | V-B.2-predicate pre-implementation verification per §7.3. Read `flash_attention_v2/phases/backward/` modules; pin Q-outer/KV-inner (case α) or KV-outer/Q-inner (case β); record findings. | ~0 code, doc only | Findings doc at `docs/superpowers/specs/2026-05-XX-tier-b-b2-predicate-verification-findings.md` committed with case label + iteration order evidence + reuse-shape decision (wholesale vs `IterationOrder` parameter). ~30 min budget. | B1.5-6 ∈ {keep, keep-with-sparsity-gate} |
| **B2-2** | B.2 | Backward kernel integration per §7.4: `ds_compute.rs` gains the skip predicate at its tile-loop boundary; `backward/prelude.rs` gains the range-table preamble; refresh `flash_attention_v2_backward_*` snapshots including Tier B-on variants; preserve no-op guarantee for non-Tier-B configs. If B2-1 surfaced case (β), `emit_skip_predicate` gains an `IterationOrder` parameter (~50 additional LOC). Real `owning_warp(qt, kvt)` mapping in backward's M3 instrumentation per §6.2. | ~200 (Rust + PTX) | Backward kernel snapshot stable; SASS baselines passing (analog of forward's `tier_b_causal_*`); no-op guarantee preserved (non-Tier-B backward kernel byte-identical to today). | B2-1 |
| **B2-3** | B.2 | Backward kernel parity tier extension: apply the same byte-identical assertion (per §6.1) to backward outputs (dQ, dK, dV) across the six `PackingFixture` entries. | ~50 (Rust) | Bit-identical backward outputs Tier-B-on vs Tier-B-off across all six fixtures. | B2-2 |

### 11.2 Conditional execution logic

The B1.5-6 outcome gates the B.2 phase. The matrix:

- **Outcome = keep unconditionally** → B2-1, B2-2, B2-3 execute in sequence. PR ships with the complete Tier B forward + backward feature.
- **Outcome = keep with sparsity gate** → B2-1, B2-2, B2-3 execute in sequence. PR ships with the complete feature; planner-side dispatch (a separate downstream spec) will add the sparsity-threshold predicate.
- **Outcome = revert** → B2-1, B2-2, B2-3 are skipped. PR ships with §3.4's revert semantics applied: `should_emit_tier_b` returns false unconditionally; Tier B tests `#[ignore]`'d; SASS baselines retained; findings doc records the revert decision and the 6-month decay timer's start date.

### 11.3 PR shape

- **Single PR.** All milestones land in one PR for reviewer-context preservation, regardless of B1.5-6's outcome.
- **PR title varies by outcome:**
  - keep unconditionally → `feat(pca-tier-b): forward + backward (B.1.5 + B.2 measurement-validated)`
  - keep with sparsity gate → `feat(pca-tier-b): forward + backward (B.1.5 + B.2; sparsity-gated dispatch follows)`
  - revert → `feat(pca-tier-b): disable Tier B dispatch (B.1.5 measurements below thresholds)`
- **Commit sequence reflects milestone IDs** (B1.5-1, B1.5-2, etc.) so reviewers can navigate the PR by phase.

### 11.4 Discovery during B1.5-2: spec-pinned fixture SMEM exceeds device cap

**What surfaced.** During B1.5-2 (gate fixture wiring + launch harness), the spec-pinned gate fixture (block_q=64, block_kv=64, head_dim=64, segment-masked causal) produced a kernel requiring **141,056 bytes** of dynamic SMEM. RTX 5070 Ti's per-CTA `MAX_DYNAMIC_SHARED_SIZE_BYTES` opt-in cap is **99 KB** (101,376 bytes). `cuFuncSetAttribute(MAX_DYNAMIC_SHARED_SIZE_BYTES, 141056)` returns `CUDA_ERROR_INVALID_VALUE`; `cuLaunchKernel` then rejects with the same code. The `--tier-b=on` smoke test failed; the `--tier-b=off` path ran cleanly (38.4 ms median over 100 iter).

**Root cause.** The original Tier B revision spec (§3.4 of `2026-05-12-pca-tier-b-revision-design.md`) derived `tier_b_range_table_offset(config) = backward_total_bytes(config) + seg_overhead(config)` — the range table rides at the tail of the **larger** of the two direction-specific SMEM totals so forward+backward can share. The "rides at the tail of the larger" choice was correct for cross-direction sharing; it was incorrect for the forward-only kernel case, which is the only case in production today (B.2 backward is gated on B1.5-6). Forward-only kernels inherit the backward-sized SMEM allocation without needing the backward-tile space (~90 KB of waste at the gate fixture's dims).

**What would have caught it.** A pre-implementation SMEM-budget calculation **instantiated at the spec-pinned gate fixture dimensions**, run during spec review and recorded in the spec text. The CSHA Tier B.1 spec did this in its §3.5 ("canonical Tier A config does NOT fit Tier B on Blackwell-class SMEM" — the section that surfaced the cross-config-doesn't-fit problem pre-implementation). The PCA Tier B revision did not do this for the gate fixture specifically; the SMEM-budget arithmetic in §3.4 was generic over `(bq, bkv, hd)`, not instantiated at the gate fixture's `(64, 64, 64)`.

**General lesson.** When a spec pins specific fixture dimensions for measurement (e.g., the gate fixture in §4.1), the spec's resource-budget arithmetic must be **instantiated at those specific values**, not left generic-over-config. Otherwise the spec passes review based on plausible-generic numbers while the spec-pinned fixture has an unverified-specific failure mode. This is an application of IR-003 (pre-implementation verification of load-bearing assumptions) specialized to resource-budget arithmetic.

**Architectural fix (applied during B1.5-2 scope expansion).** Parameterize `tier_b_range_table_offset` by `Direction`:

```rust
pub enum Direction { Forward, Backward }

pub fn tier_b_range_table_offset(config: &FlashAttentionConfig, direction: Direction) -> u32 {
    let base = match direction {
        Direction::Forward  => total_bytes(config),
        Direction::Backward => backward_total_bytes(config),
    } + seg_overhead(config);
    align_up_u32(base, 2)
}
```

Forward-only kernels pass `Direction::Forward`; the offset becomes `total_bytes(config) + seg_overhead(config)`. At the gate fixture's dims this drops the offset from ~140 KB to ~50 KB (forward total_bytes ~17 KB + seg_overhead 32 KB + range tables 0.5 KB = ~50 KB, well under the 99 KB cap). Backward kernels (when B.2 ships) pass `Direction::Backward` and get the original `backward_total_bytes(config) + seg_overhead(config)` semantics — no change for the not-yet-shipped backward direction.

**Snapshot impact.** Four forward kernel snapshots are invalidated by the offset change:

- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_32_32_32`
- `pca_forward_kernel_snapshot__forward_kernel_segment_masked_tier_b_causal_64_64_64`
- `pca_tier_b_preamble_isolation` (range-table-base immediate changes)
- `pca_tier_b_predicate_isolation` (range-table-base immediate changes)

Re-baseline via `cargo insta accept`. PR description shows the per-snapshot offset diff for reviewer verification (e.g., "offset 141008 → 50688" — the literal in `add.u64 %addr_min_TILERANGE, %addr_min_TILERANGE, <NUM>;`). Backward kernel snapshots (not yet shipped) remain stable because backward inherits `Direction::Backward` and gets the original value. Same snapshot-re-baseline review discipline as WGGO Phase 2 #134 (per-snapshot diffs with rationale in PR body).

**Institutional rule citation.** This finding is recorded as a new "Cited from" entry on IR-003 in `docs/wiki/institutional-rules.md`, NOT promoted to a new IR-013. Per the IR-NNN entry criteria (Section 5.4 of the institutional-rules registry: pattern must surface across ≥2 distinct specs to warrant a new rule), this is the first instance of fixture-pinned resource-budget verification failing as a load-bearing oversight. If a second instance materializes (e.g., during M35.2a backward SMEM verification at spec-pinned training fixture dims), promote to IR-013 then with both instances cited.

The brainstorm's pre-implementation-verification discipline is the system that catches this class of issue. The discipline operating correctly means: each instance of "verification missed something" produces a tightening of the discipline. After the first instance (the SMEM probe added during the 2026-05-12 revision after the compile-time-unroll concern surfaced), the gates tightened to include the Phase 0 probe. After this second instance, §11.4 documents the gap and IR-003 gains a new "Cited from" entry making the fixture-pinned-resource-budget pattern visible for future spec reviews.

### 11.5 LOC budget (rough)

- B.1.5 total: ~480 LOC (mostly Rust; ~100 shell).
- B.2 (if executed) total: ~250 LOC.
- B.2 (if skipped, revert outcome) total: ~10 LOC (`should_emit_tier_b` modification + `#[ignore]` markers + findings doc).
- Verification surface (snapshots, SASS baselines, parity assertions, findings docs) is ~30–40% of the total per IR-008.

These estimates are bench-shaped, not commit-shaped — the actual commit count may be larger (each milestone may split into multiple commits for review granularity).

## 12. Risks and out-of-scope

### 12.1 Risks tracked

- **Hardware-specific measurement.** M2/M6 measurement runs on a single RTX 5070 Ti. The keep/revert decision applies to this hardware specifically. Production deployment on different hardware (different Blackwell variants, sm_80 GPUs, etc.) may have different characteristics. Mitigation: spec pins the hardware in §8.1; future deployment surfaces re-run the protocol or accept the RTX 5070 Ti measurement as a proxy.

- **Driver / toolkit drift.** CUDA 13.2 / driver 591.86 specifically. Future toolkit updates may change ptxas optimization decisions that affect wall-time. Mitigation: re-run triggers in IR-002 / §3.4.1's "CUDA toolkit / driver change" trigger; ±10% reproducibility tolerance per §8.5.

- **V-B.2-predicate uncovers case (β).** B.2 implementation cost grows from "reuse wholesale" to "parameterize emit_skip_predicate." Mitigation: §7.3 pins the case-(β) implementation path explicitly; ~50 additional lines is small absolute cost. Risk is low.

- **Sensitivity tier surfaces non-monotonic curve.** If 10% sparsity shows Tier B helps and 50% shows hurt (or any non-monotonic pattern), §4.3.2's three-shape model doesn't capture the actual behavior. Mitigation: the findings doc records the actual curve; investigation precedes acceptance per §4.3.1's "investigation rather than acceptance" discipline.

### 12.2 Out-of-scope items (deferred to later work)

Per §2.2 — production-caller activation, Tier B-extended for seq_len > 16K, CTA-uniform predicate trade-off, tile-skip-aware checkpointing, external bench consumers. None of these are required to make the M2/M6 keep/revert decision; all of them follow downstream of "keep" outcomes.

## 13. References

- `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md` — original Tier B design (revised in place via §3.1 / §3.4 / §6.3 in the 2026-05-12 revision spec).
- `docs/superpowers/specs/2026-05-12-pca-tier-b-revision-design.md` — revision design that drove PR #168.
- `docs/superpowers/specs/2026-05-12-tier-b-smem-probe-findings.md` — Phase 0 SMEM probe findings (Option 1 confirmed safe).
- `docs/superpowers/plans/2026-05-12-pca-tier-b-tile-skip-implementation-v2.md` — v2 implementation plan (Tasks 1-16; B.1.5 + B.2 reference selected tasks).
- `docs/wiki/institutional-rules.md` — IR-001 through IR-008 (existing); IR-009 through IR-012 codified by this spec.
- `crates/nsl-codegen/tests/pca_tier_b_m3_parity.rs` — 6 `#[ignore]`'d parity tests this spec un-ignores via the bench binary's launch harness.
- `crates/nsl-codegen/tests/fixtures/mod.rs` — `PackingFixture` matrix for the parity tier.
- `crates/nsl-codegen/src/pca_tilerange.rs` — forward's `emit_skip_predicate` + `emit_range_table_preamble` + `emit_skip_decision_writeback` (reused by B.2 per §7).
- `crates/nsl-codegen/src/flash_attention_v2/csha_hooks.rs` — `emit_segment_mask` (reused by B.2 per §7.2).
- `crates/nsl-codegen/src/flash_attention_v2/phases/backward/` — backward kernel modules touched by B.2.
- `scripts/measure_tier_b_m2.sh`, `scripts/measure_tier_b_m6.sh` — measurement scripts that invoke the bench binary per §5.2's CLI contract.

---

## Revision changelog

- **2026-05-13** — initial. Post-merge follow-up to PR #168. Five brainstorm decisions resolved (acceptance bar, fixture matrix, bench binary location, B.2 inheritance, owning_warp). Four institutional rules codified (IR-009 through IR-012).
