# PCA Tier B Dispatch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Activate PCA Tier B emission for `segment_masked` configs at `seq_len >= floor` via a minimal heuristic in `flash_attention_v2::should_emit_tier_b`, replacing the unconditional-false stub from PR #168/#169.

**Architecture:** Four sequential milestones — D-1 (verification, doc only) → D-2 (measurement, doc + CSV) → D-3 (heuristic implementation; signature change conditional on D-1 outcome) → D-4 (snapshot re-baseline + activation PR). The D-3 gate is **runtime parity** (kernel outputs byte-identical); the D-4 scope is **snapshot parity** (PTX content re-baseline). These distinct test surfaces must NOT be conflated — see design spec §12.2.

**Tech Stack:** Rust 1.95.0, Cranelift + PTX synthesis, `cargo test --tests`, `cargo run -p nsl-codegen --bin nsl-codegen-bench`, insta snapshot framework.

**Design spec:** `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` — read §3 (heuristic), §5 (V-dispatch-integration), §6 (floor protocol), §8 (snapshot discipline), §12 (milestones) before starting.

---

## File Structure

**Read-only references (no edits, just consumed):**
- `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md` — design spec.
- `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md` — KEEP UNCONDITIONALLY findings; cited from heuristic source.
- `crates/nsl-codegen/src/pca_tilerange/mod.rs::should_emit_tier_b` — fine-grained PTX-budget check (already wired through prelude/s_compute); the planner-side toggle gates this one. **Do not modify.**

**Created in D-1 (verification, doc only):**
- `docs/superpowers/specs/2026-05-14-tier-b-dispatch-integration-findings.md` — caller classification (α/β/γ) + outcome decision.

**Created in D-2 (measurement, doc + CSV):**
- `docs/superpowers/specs/2026-05-14-tier-b-floor-derivation-findings.md` — per-seq_len wall-time win %, derived floor.
- `docs/superpowers/specs/2026-05-14-tier-b-floor-derivation.csv` — raw measurement data (7 rows).

**Modified in D-3 (implementation — case (α) baseline):**
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs` — `should_emit_tier_b` body becomes the heuristic; signature gains `seq_len: u32`; `synthesize_flash_attention_ptx_v2` 1-arg wrapper gains `seq_len: u32`.
- `crates/nsl-codegen/src/flash_attention_selector.rs` — `synthesize_flash_attention_ptx_selected*` threads `seq_len` through.
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs:2045, 2072` — pass `seq_len` to the 1-arg call.
- `crates/nsl-codegen/tests/csha_gap_b_ptx_context.rs`, `csha_forward_saves_diag.rs`, `csha_dx_norm_readback_diag.rs`, `csha_cuda_backward.rs`, `csha_orchestrator_softmax_save_gate.rs`, `csha_ptx_ptxas_validation.rs`, `fa_v2_snapshots.rs`, `pca_forward_kernel_snapshot.rs` — update 1-arg test call sites (~20 sites).

**Modified in D-3 (case (β) variant — sparsity-only collapse, if D-1 outcome dictates):**
- `crates/nsl-codegen/src/flash_attention_v2/mod.rs` — `should_emit_tier_b` body uses `config.segment_masked` only; signature unchanged. No call-site updates needed.

**Modified in D-3 (case (γ) variant — FlashAttentionConfig field, if D-1 outcome dictates):**
- `crates/nsl-codegen/src/flash_attention.rs` — `FlashAttentionConfig` gains `seq_len: u32` field.
- ~50 fixture construction sites (test code, debugging utilities) gain the new field.

**Re-baselined in D-4:**
- `crates/nsl-codegen/tests/snapshots/pca_forward_kernel_snapshot__forward_kernel_segment_masked_*.snap` — 2 forward snapshots (existing `.snap.new` files at gate fixture dims confirm 2 are pending; verify exact count at D-3 start).
- `crates/nsl-codegen/tests/snapshots/pca_backward_kernel_snapshot__backward_kernel_segment_masked_*.snap` — 2 backward snapshots.
- ≤2 additional snapshots if cargo surfaces them at D-3.

**Expected-stable in D-4 (verify byte-identical, do NOT re-baseline):**
- `crates/nsl-codegen/tests/snapshots/pca_tier_b_preamble_isolation__*.snap`
- `crates/nsl-codegen/tests/snapshots/pca_tier_b_predicate_isolation__*.snap`

---

## Branch and Worktree Setup

The worktree is already at `c:\Users\bwiem\projects\NSL\.claude\worktrees\feat-pca-tier-b-dispatch` on branch `feat/m35-2a-design-only-landing`. Before D-1, create a fresh branch off `origin/main` for the dispatch work — the existing branch carries unrelated M35.2a design-only commits.

- [ ] **Step S-0: Fetch and branch**

```bash
git fetch origin
git checkout -b feat/pca-tier-b-dispatch origin/main
git status
```

Expected: clean working tree on `feat/pca-tier-b-dispatch` tracking `origin/main`.

---

## Task D-1: V-dispatch-integration verification

**Scope:** Classify every production caller of `synthesize_flash_attention_ptx_v2` as α (has seq_len at call time), β (doesn't have seq_len — early-compilation context), or γ (knows seq_len structurally as type parameter, not value). Decide the seq_len source per §5.2 + §5.3 of the design spec.

**Budget:** 30-45 minutes. No code edits — findings doc only.

**Files:**
- Create: `docs/superpowers/specs/2026-05-14-tier-b-dispatch-integration-findings.md`
- Read-only: `crates/nsl-codegen/src/flash_attention_selector.rs:52` (the production wrapper); `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs:2045,2072` (CSHA hooks).

- [ ] **Step D1-1: Enumerate every production caller of `synthesize_flash_attention_ptx_v2`**

Run:
```bash
git grep -n 'synthesize_flash_attention_ptx_v2[^_]' crates/nsl-codegen/src
```

Expected: enumeration of 3-5 production call sites (excluding tests):
- `crates/nsl-codegen/src/flash_attention_selector.rs:52`
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs:2045`
- `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs:2072`
- (possibly more — record what grep returns)

Record each in the findings doc with file:line + 5-line surrounding context.

- [ ] **Step D1-2: For each caller, read upward to find seq_len availability**

For every production caller, trace the call chain upward (the caller of the caller, etc.) until either:
- A `seq_len` value is in scope (case α at that caller).
- A point where seq_len is provably not known (case β — typically type-level compilation context).
- A point where seq_len exists as a type parameter, not a runtime value (case γ).

Document the trace in the findings doc — file:line of each link in the chain, and the classification verdict.

**Pattern from prior verifications** (V-B.2-predicate, V-M35.2a-STE): record the grep commands you ran so a reader can reproduce. The findings doc is the audit trail per IR-002.

- [ ] **Step D1-3: Tally and decide outcome**

Tally α/β/γ counts. Apply §5.2 + §5.3 of the spec:

- **All α (most likely outcome):** option 4 (synthesizer arg). Plan defaults to this path.
- **Mostly α + small β:** option 4 for α callers; β callers stay no-op. Note β callers as known-dormant in the findings doc.
- **Mostly β:** option 2 (sparsity-only collapse — drop seq_len gate). D-3 implementation simplifies; floor derivation in D-2 still useful as future v2 input but doesn't gate dispatch.
- **All γ:** option 3 (FlashAttentionConfig field). D-3 implementation paths shifts to fixture construction.

- [ ] **Step D1-4: Write findings doc**

The doc must include:
1. **Caller classification table** — file:line + α/β/γ + 1-line justification per row.
2. **Outcome decision** — which of options 2 / 3 / 4 (per §5.2 of the spec).
3. **Conditional path for D-2 floor scope** (per §6.5 of the spec).
4. **Cross-reference** to design spec §5.

Commit the findings doc:

```bash
git add docs/superpowers/specs/2026-05-14-tier-b-dispatch-integration-findings.md
git commit -m "docs(pca-tier-b-dispatch): D-1 — V-dispatch-integration findings (option <N>)"
```

**Gate to D-2:** findings doc committed; option N chosen.

---

## Task D-2: Floor derivation measurement

**Scope:** Sweep `seq_len ∈ {128, 256, 512, 1024, 2048, 4096, 8192}` at `head_dim=64, batch=4, sparsity=50%, block_q=64, block_kv=64, segment-masked causal, sm_120`. Median-of-5 wall-time, Tier-B-on vs Tier-B-off; derive `TIER_B_SEQ_LEN_FLOOR = smallest seq_len with wall-time win ≥ 10%`.

**Budget:** ~1 hour (7 measurements × ~5 min each at iterations=100 + ~30 min findings doc).

**Files:**
- Create: `docs/superpowers/specs/2026-05-14-tier-b-floor-derivation-findings.md`
- Create: `docs/superpowers/specs/2026-05-14-tier-b-floor-derivation.csv`
- Read-only: `crates/nsl-codegen/src/bin/bench/launch.rs` (existing bench harness from B.1.5-5).

- [ ] **Step D2-1: Verify bench binary reproduces M2/M6 gate-fixture measurement**

Run the gate-fixture measurement (seq_len=4096, sparsity=50%) and verify it reproduces the §3.1 acceptance bar from the 2026-05-13 findings doc (wall-time win 73.33% ±10% tolerance). This is a self-check that the measurement harness is sound before we sweep.

```bash
cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
  --fixture m2 --seq-len 4096 --sparsity 50 --iterations 100 --tier-b on \
  --seed 0xDEADBEEF
cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
  --fixture m2 --seq-len 4096 --sparsity 50 --iterations 100 --tier-b off \
  --seed 0xDEADBEEF
```

Expected: wall-time win at seq_len=4096 within ±10% of the 73.33% recorded in `docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md`.

**If the reproducer drifts > 10%:** stop and investigate before continuing the sweep. The drift breaks measurement-protocol reproducibility (IR-012); the sweep would inherit the drift.

- [ ] **Step D2-2: Run 7-point sweep**

For each `seq_len ∈ {128, 256, 512, 1024, 2048, 4096, 8192}`:

```bash
cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
  --fixture m2 --seq-len <S> --sparsity 50 --iterations 100 --tier-b on \
  --seed 0xDEADBEEF | tee -a /tmp/tier_b_floor_on.txt
cargo run -p nsl-codegen --release --bin nsl-codegen-bench -- \
  --fixture m2 --seq-len <S> --sparsity 50 --iterations 100 --tier-b off \
  --seed 0xDEADBEEF | tee -a /tmp/tier_b_floor_off.txt
```

Record median-of-5 wall-time per (seq_len, tier-b-mode). Compute wall-time win % = `(off - on) / off * 100` per seq_len.

- [ ] **Step D2-3: Build CSV**

Write `docs/superpowers/specs/2026-05-14-tier-b-floor-derivation.csv` with header `seq_len,wall_time_on_ns,wall_time_off_ns,win_pct` and 7 rows.

- [ ] **Step D2-4: Derive floor**

Per §6.4 of the spec:
- **Floor = smallest seq_len with win_pct ≥ 10%.**
- If win_pct ≥ 10% at all 7 points → floor = 128.
- If a threshold exists in the middle → floor = smallest seq_len above the threshold.
- If non-monotonic → **investigation rather than acceptance** (per §6.4). Stop and document; do NOT pick a floor mechanically — refer back to the spec author.

- [ ] **Step D2-5: Write findings doc**

The doc must include:
1. **Per-seq_len win % table** (cite the CSV).
2. **Derived `TIER_B_SEQ_LEN_FLOOR = <value>`** with explicit justification.
3. **Curve-shape inference** — monotonic / threshold / non-monotonic.
4. **Cross-reference** to D-1 findings (scope conditional per §6.5) and design spec §6.

Commit both files:

```bash
git add docs/superpowers/specs/2026-05-14-tier-b-floor-derivation-findings.md \
        docs/superpowers/specs/2026-05-14-tier-b-floor-derivation.csv
git commit -m "docs(pca-tier-b-dispatch): D-2 — floor derivation (floor=<N>, <shape>)"
```

**Gate to D-3:** findings doc committed; `TIER_B_SEQ_LEN_FLOOR` value pinned.

---

## Task D-3 (case α): Heuristic implementation — synthesizer arg

**Branch decision:** This task assumes D-1 produced **option 4 (synthesizer arg)**. If D-1 produced option 2 (sparsity-only) or option 3 (FlashAttentionConfig field), skip to **Task D-3 (case β)** or **Task D-3 (case γ)** below.

**Scope:** Implement the heuristic in `should_emit_tier_b(&config, seq_len)`. Change `synthesize_flash_attention_ptx_v2(config)` signature to `synthesize_flash_attention_ptx_v2(config, seq_len)`. Update all production call sites and ~20 test call sites. Verify runtime parity at D-3 gate. Snapshot parity (D-4 scope) WILL break — that's expected per §12.2 of the spec.

**Files:** see "File Structure" → Modified in D-3 (case (α) baseline).

- [ ] **Step D3α-1: Add `TIER_B_SEQ_LEN_FLOOR` constant**

In `crates/nsl-codegen/src/flash_attention_v2/mod.rs`, near the existing `should_emit_tier_b`:

```rust
/// Empirical floor below which Tier B's skip-check overhead exceeds the
/// skip-payoff. Derived from `docs/superpowers/specs/2026-05-14-tier-b-floor-derivation-findings.md`
/// at sparsity=50%, head_dim=64, batch=4, block_q=64, block_kv=64,
/// segment-masked causal, sm_120 (gate-fixture-aligned dimensions).
pub const TIER_B_SEQ_LEN_FLOOR: u32 = <VALUE_FROM_D2>;
```

The `<VALUE_FROM_D2>` is the floor pinned in D-2 step 4.

- [ ] **Step D3α-2: Write the failing heuristic test**

Create `crates/nsl-codegen/tests/pca_tier_b_dispatch_heuristic.rs`:

```rust
//! Unit tests for the minimal-heuristic dispatch toggle.
//!
//! Verifies that `should_emit_tier_b` returns `true` exactly when
//! `config.segment_masked` AND `seq_len >= TIER_B_SEQ_LEN_FLOOR`.

use nsl_codegen::flash_attention::FlashAttentionConfig;
use nsl_codegen::flash_attention_v2::{
    should_emit_tier_b, TIER_B_SEQ_LEN_FLOOR,
};

fn segment_masked_config() -> FlashAttentionConfig {
    let mut cfg = FlashAttentionConfig::csha_canonical();
    cfg.segment_masked = true;
    cfg.causal = true;
    cfg
}

fn non_segment_masked_config() -> FlashAttentionConfig {
    let mut cfg = FlashAttentionConfig::csha_canonical();
    cfg.segment_masked = false;
    cfg
}

#[test]
fn heuristic_emits_for_segment_masked_at_floor() {
    let cfg = segment_masked_config();
    assert!(should_emit_tier_b(&cfg, TIER_B_SEQ_LEN_FLOOR));
}

#[test]
fn heuristic_emits_for_segment_masked_above_floor() {
    let cfg = segment_masked_config();
    assert!(should_emit_tier_b(&cfg, TIER_B_SEQ_LEN_FLOOR + 1));
    assert!(should_emit_tier_b(&cfg, TIER_B_SEQ_LEN_FLOOR * 2));
}

#[test]
fn heuristic_rejects_below_floor() {
    let cfg = segment_masked_config();
    if TIER_B_SEQ_LEN_FLOOR > 0 {
        assert!(!should_emit_tier_b(&cfg, TIER_B_SEQ_LEN_FLOOR - 1));
    }
    assert!(!should_emit_tier_b(&cfg, 0));
}

#[test]
fn heuristic_rejects_non_segment_masked() {
    let cfg = non_segment_masked_config();
    assert!(!should_emit_tier_b(&cfg, TIER_B_SEQ_LEN_FLOOR));
    assert!(!should_emit_tier_b(&cfg, TIER_B_SEQ_LEN_FLOOR * 4));
}
```

- [ ] **Step D3α-3: Run the test to verify it fails**

Run:
```bash
cargo test -p nsl-codegen --test pca_tier_b_dispatch_heuristic
```

Expected: compile error — `should_emit_tier_b` signature is `(&FlashAttentionConfig)` not `(&FlashAttentionConfig, u32)`. This is the right failure mode for D3α-2.

- [ ] **Step D3α-4: Change `should_emit_tier_b` signature and body**

In `crates/nsl-codegen/src/flash_attention_v2/mod.rs`, replace the existing `should_emit_tier_b` with:

```rust
/// Central dispatch toggle for PCA Tier B PTX emission.
///
/// Returns `true` when `config.segment_masked` AND `seq_len >= TIER_B_SEQ_LEN_FLOOR`.
/// The floor is measurement-derived (see `TIER_B_SEQ_LEN_FLOOR` doc comment).
///
/// Distinct from the fine-grained PTX-budget check in
/// [`crate::pca_tilerange::should_emit_tier_b`] — the planner-side toggle
/// gates the budget check.
pub fn should_emit_tier_b(config: &FlashAttentionConfig, seq_len: u32) -> bool {
    config.segment_masked && seq_len >= TIER_B_SEQ_LEN_FLOOR
}
```

- [ ] **Step D3α-5: Change `synthesize_flash_attention_ptx_v2` signature and wire heuristic**

In `crates/nsl-codegen/src/flash_attention_v2/mod.rs`, replace the existing 1-arg wrapper:

```rust
/// 1-arg synthesizer — applies the planner-side dispatch heuristic.
///
/// New invariant (post-dispatch activation):
/// `synthesize_flash_attention_ptx_v2(config, seq_len)` produces byte-identical
/// PTX to `synthesize_flash_attention_ptx_v2_with_tier_b(config, args)` where
/// `args = if should_emit_tier_b(config, seq_len) { Some(TierBArgs { seq_len,
/// residency: SegmentResidency::Tiled }) } else { None }`.
///
/// The 2-arg `_with_tier_b` form remains the public override surface (per
/// design spec §4.4): explicit `Some(args)` or `None` bypasses the heuristic.
pub fn synthesize_flash_attention_ptx_v2(
    config: &FlashAttentionConfig,
    seq_len: u32,
) -> Vec<u8> {
    let tier_b = if should_emit_tier_b(config, seq_len) {
        Some((seq_len, crate::pca_segment::SegmentResidency::Tiled))
        //                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        // Single edit point per design spec §7.2. Measurement-grounded default
        // (see docs/superpowers/specs/2026-05-13-tier-b-m2-m6-findings.md).
    } else {
        None
    };
    synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b)
}
```

(The exact `Option` type — `Option<(u32, SegmentResidency)>` vs `Option<TierBArgs>` — depends on the current `_with_tier_b` signature. Match what exists.)

- [ ] **Step D3α-6: Run heuristic unit tests to verify they pass**

```bash
cargo test -p nsl-codegen --test pca_tier_b_dispatch_heuristic
```

Expected: 4/4 tests pass.

- [ ] **Step D3α-7: Update production call sites in `flash_attention_selector.rs`**

In `crates/nsl-codegen/src/flash_attention_selector.rs`, change every signature that calls `synthesize_flash_attention_ptx_v2` to thread `seq_len: u32`:

- `synthesize_flash_attention_ptx_selected_with_diag(config, diagnostics)` → `synthesize_flash_attention_ptx_selected_with_diag(config, seq_len, diagnostics)`.
- `synthesize_flash_attention_ptx_selected(config)` → `synthesize_flash_attention_ptx_selected(config, seq_len)`.
- The `synthesize_flash_attention_ptx_v2(config)` call at line 52 becomes `synthesize_flash_attention_ptx_v2(config, seq_len)`.

- [ ] **Step D3α-8: Update production call sites in `phases/forward/csha_hooks.rs`**

Both calls at lines 2045 and 2072 — trace upward to find `seq_len` in scope per the D-1 findings (case α verified seq_len is available). Pass `seq_len` to the new 1-arg call.

- [ ] **Step D3α-9: Update all selector callers**

Run:
```bash
git grep -n 'synthesize_flash_attention_ptx_selected\b' crates/
```

Update every caller to pass `seq_len`. Same trace-upward pattern as D3α-8.

- [ ] **Step D3α-10: Update test call sites**

Run:
```bash
git grep -n 'synthesize_flash_attention_ptx_v2(' crates/nsl-codegen/tests
```

For each test, decide on a seq_len value:
- Tests that target a specific kernel-snapshot fixture: use the fixture's intended seq_len (4096 for gate-fixture-aligned snapshots; smaller for cheaper test fixtures).
- Tests that don't care about Tier B activation: use `seq_len = 0` (forces heuristic to return false; preserves pre-dispatch PTX).
- Tests that explicitly verify the no-op invariant (e.g., `pca_forward_kernel_snapshot.rs:139`): see D3α-11.

- [ ] **Step D3α-11: Update the no-op invariant assertion**

In `crates/nsl-codegen/tests/pca_forward_kernel_snapshot.rs:139-144`, update the assertion from:

```rust
// Old: 1-arg == 2-arg-with-None
let via_1arg = synthesize_flash_attention_ptx_v2(&cfg);
let via_2arg = synthesize_flash_attention_ptx_v2_with_tier_b(&cfg, None);
assert_eq!(via_1arg, via_2arg, "...");
```

To the new invariant per design spec §4.2:

```rust
// New: 1-arg == 2-arg-with-heuristic's-choice
let seq_len = <FIXTURE_SEQ_LEN>;
let via_1arg = synthesize_flash_attention_ptx_v2(&cfg, seq_len);
let heuristic_choice = if should_emit_tier_b(&cfg, seq_len) {
    Some((seq_len, crate::pca_segment::SegmentResidency::Tiled))
} else {
    None
};
let via_2arg = synthesize_flash_attention_ptx_v2_with_tier_b(&cfg, heuristic_choice);
assert_eq!(via_1arg, via_2arg,
    "no-op guarantee violated: 1-arg form must equal 2-arg form with heuristic's choice");
```

- [ ] **Step D3α-12: Verify build**

Run:
```bash
cargo build -p nsl-codegen
```

Expected: clean build. Any remaining call site is a missed update.

- [ ] **Step D3α-13: Run runtime-parity tests (the D-3 gate)**

The runtime parity tests from PR #169 are the gate. Find them:

```bash
git grep -ln 'tier_b.*parity' crates/nsl-codegen/tests
```

Expected hits include `pca_tier_b_m3_parity.rs` and any forward parity equivalents. Run them:

```bash
cargo test -p nsl-codegen --test pca_tier_b_m3_parity
# plus any forward parity test files surfaced by the grep
```

Expected: all 16 runtime parity assertions pass (per design spec §12.2). These compare kernel outputs (dQ, dK, dV for backward; O for forward) byte-identically — they pass regardless of PTX text changes because skipped tiles contribute exactly zero by construction.

**Snapshot tests will fail at this step — that is expected.** Snapshot re-baseline is D-4 scope. Conflating runtime + snapshot parity in this gate would create the deadlock §12.2 of the spec pre-empts.

- [ ] **Step D3α-14: Verify the 2 isolation tests stay byte-identical**

Run:
```bash
cargo test -p nsl-codegen --test pca_tier_b_preamble_isolation
cargo test -p nsl-codegen --test pca_tier_b_predicate_isolation
```

Expected: both pass with no `.snap.new` files generated. The isolation tests call emission functions directly (per PR #168's design); they don't route through `synthesize_flash_attention_ptx_v2`, so dispatch activation MUST NOT change their output.

**If a `.snap.new` is generated:** STOP. Per design spec §8.1, isolation-test drift indicates an unexpected call-graph dependency. Investigate before re-baselining; do not mechanically accept the diff.

- [ ] **Step D3α-15: Commit D-3 work**

```bash
git add -A crates/nsl-codegen/src crates/nsl-codegen/tests
git commit -m "feat(pca-tier-b): D-3 — minimal-heuristic dispatch (case α, floor=<N>)"
```

**Gate to D-4:** runtime parity passes; isolation tests stay byte-identical; cargo build clean.

---

## Task D-3 (case β): Heuristic implementation — sparsity-only collapse

**Branch:** If D-1's outcome was option 2 (sparsity-only collapse), use this task in place of D-3 (case α).

**Scope:** `should_emit_tier_b(&config) -> bool` returns `config.segment_masked`. No seq_len gate. `TIER_B_SEQ_LEN_FLOOR` constant is not introduced (D-2's findings are useful as future v2 input but not load-bearing here). The `synthesize_flash_attention_ptx_v2` 1-arg form keeps its `(&FlashAttentionConfig)` signature.

- [ ] **Step D3β-1: Write the failing heuristic test**

Create `crates/nsl-codegen/tests/pca_tier_b_dispatch_heuristic.rs`:

```rust
use nsl_codegen::flash_attention::FlashAttentionConfig;
use nsl_codegen::flash_attention_v2::should_emit_tier_b;

fn segment_masked_config() -> FlashAttentionConfig {
    let mut cfg = FlashAttentionConfig::csha_canonical();
    cfg.segment_masked = true;
    cfg
}

#[test]
fn heuristic_emits_for_segment_masked() {
    assert!(should_emit_tier_b(&segment_masked_config()));
}

#[test]
fn heuristic_rejects_non_segment_masked() {
    let mut cfg = segment_masked_config();
    cfg.segment_masked = false;
    assert!(!should_emit_tier_b(&cfg));
}
```

- [ ] **Step D3β-2: Run the test to verify it fails**

```bash
cargo test -p nsl-codegen --test pca_tier_b_dispatch_heuristic
```

Expected: test fails because `should_emit_tier_b` returns `false` unconditionally.

- [ ] **Step D3β-3: Change `should_emit_tier_b` body**

In `crates/nsl-codegen/src/flash_attention_v2/mod.rs`:

```rust
pub fn should_emit_tier_b(config: &FlashAttentionConfig) -> bool {
    config.segment_masked
}
```

(No signature change — `seq_len` is not threaded.)

- [ ] **Step D3β-4: Update `synthesize_flash_attention_ptx_v2` to call the heuristic**

```rust
pub fn synthesize_flash_attention_ptx_v2(config: &FlashAttentionConfig) -> Vec<u8> {
    let tier_b = if should_emit_tier_b(config) {
        // seq_len is unavailable at this call site (case β justification);
        // pass a placeholder that flows through pca_tilerange's fine-grained
        // budget check. See D-1 findings doc for the caller-classification
        // evidence that seq_len cannot be threaded.
        Some((/* placeholder seq_len from pca_tilerange convention */, crate::pca_segment::SegmentResidency::Tiled))
    } else {
        None
    };
    synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b)
}
```

**Note:** the placeholder seq_len source is case-β-specific. D-1 findings doc MUST specify the convention (e.g., "use config.block_kv * N" or "extract from FlashAttentionConfig.max_seq_len if present") — if D-1 didn't specify, return to the spec author before continuing.

- [ ] **Step D3β-5: Run heuristic unit tests + runtime parity tests**

```bash
cargo test -p nsl-codegen --test pca_tier_b_dispatch_heuristic
cargo test -p nsl-codegen --test pca_tier_b_m3_parity
# plus forward parity tests
```

Expected: heuristic tests 2/2 pass; runtime parity tests 16/16 pass.

- [ ] **Step D3β-6: Verify isolation tests stay byte-identical**

Same as D3α-14.

- [ ] **Step D3β-7: Commit D-3 work**

```bash
git add -A crates/nsl-codegen/src crates/nsl-codegen/tests
git commit -m "feat(pca-tier-b): D-3 — sparsity-only dispatch (case β)"
```

---

## Task D-3 (case γ): Heuristic implementation — FlashAttentionConfig field

**Branch:** If D-1's outcome was option 3 (add `seq_len: u32` to FlashAttentionConfig), use this task in place of D-3 (case α).

**Scope:** Add `seq_len: u32` field to FlashAttentionConfig. Heuristic reads `config.seq_len`. ~50 fixture construction sites gain the field. `synthesize_flash_attention_ptx_v2` signature unchanged.

- [ ] **Step D3γ-1: Add `seq_len` field to FlashAttentionConfig**

In `crates/nsl-codegen/src/flash_attention.rs`, add `pub seq_len: u32` field with a sentinel default (e.g., 0) for existing constructors.

- [ ] **Step D3γ-2: Update every fixture construction site**

Run:
```bash
git grep -ln 'FlashAttentionConfig {' crates/
```

For each construction site, add `seq_len: <appropriate_value>`. Use 4096 for gate-fixture-aligned configs; 0 for configs that explicitly should NOT activate Tier B.

- [ ] **Step D3γ-3: Write heuristic test**

```rust
#[test]
fn heuristic_emits_for_segment_masked_at_floor() {
    let mut cfg = FlashAttentionConfig::csha_canonical();
    cfg.segment_masked = true;
    cfg.seq_len = TIER_B_SEQ_LEN_FLOOR;
    assert!(should_emit_tier_b(&cfg));
}
// + below-floor, non-segment-masked variants as in D3α-2.
```

- [ ] **Step D3γ-4: Implement heuristic body**

```rust
pub fn should_emit_tier_b(config: &FlashAttentionConfig) -> bool {
    config.segment_masked && config.seq_len >= TIER_B_SEQ_LEN_FLOOR
}
```

- [ ] **Step D3γ-5: Update `synthesize_flash_attention_ptx_v2`**

```rust
pub fn synthesize_flash_attention_ptx_v2(config: &FlashAttentionConfig) -> Vec<u8> {
    let tier_b = if should_emit_tier_b(config) {
        Some((config.seq_len, crate::pca_segment::SegmentResidency::Tiled))
    } else {
        None
    };
    synthesize_flash_attention_ptx_v2_with_tier_b(config, tier_b)
}
```

- [ ] **Step D3γ-6: Run heuristic + runtime parity tests**

Same as D3α-13.

- [ ] **Step D3γ-7: Verify isolation tests stay byte-identical**

Same as D3α-14.

- [ ] **Step D3γ-8: Commit D-3 work**

```bash
git add -A
git commit -m "feat(pca-tier-b): D-3 — FlashAttentionConfig.seq_len dispatch (case γ, floor=<N>)"
```

---

## Task D-4: Snapshot re-baseline and activation PR

**Scope:** Re-baseline 6 affected kernel snapshots per §8.1; verify 2 isolation snapshots stay byte-identical; write the activation PR with the cascade-narrative discipline per §8.2.

**Files:** the `.snap` files listed under "File Structure" → Re-baselined in D-4.

- [ ] **Step D4-1: Surface the exact affected-snapshot set**

```bash
cargo test -p nsl-codegen --tests 2>&1 | grep -E '^test .* FAILED' | head -20
ls crates/nsl-codegen/tests/snapshots/*.snap.new
```

Expected: ≤8 `.snap.new` files. Cross-reference against the spec's §8.1 list (4 forward + 4 backward, may differ slightly).

- [ ] **Step D4-2: Per-snapshot diff inspection**

For each `.snap.new`, run:

```bash
git diff crates/nsl-codegen/tests/snapshots/<file>.snap
diff crates/nsl-codegen/tests/snapshots/<file>.snap crates/nsl-codegen/tests/snapshots/<file>.snap.new
```

Per design spec §8.2, verify each diff is **localized to the predicate block** — the rest of the PTX should be byte-identical between old and new snapshot. If a diff bleeds outside the predicate block, STOP — that indicates an unintended emission change, not a heuristic activation.

- [ ] **Step D4-3: Hand-derive expected dispatch per fixture**

For each affected snapshot fixture, derive what the heuristic should return:
- Read the fixture's `seq_len` and `segment_masked` values.
- Compute `should_emit_tier_b(&cfg, seq_len)` by hand.
- Verify the snapshot diff matches: `true` → predicate path added; `false` → snapshot unchanged (snapshot should NOT have been in the affected set).

If a snapshot is in the affected set but the hand-derivation says `false`, STOP — the heuristic is mis-emitting somewhere.

- [ ] **Step D4-4: Accept the snapshot updates**

```bash
cargo insta accept --workspace-root crates/nsl-codegen
```

Or per-file:
```bash
for f in crates/nsl-codegen/tests/snapshots/*.snap.new; do
  mv "$f" "${f%.new}"
done
```

- [ ] **Step D4-5: Re-run full test suite**

```bash
cargo test -p nsl-codegen
```

Expected: green across the board. Runtime parity from D-3 + accepted snapshots from D-4 + isolation tests unchanged.

- [ ] **Step D4-6: Commit snapshot updates**

```bash
git add crates/nsl-codegen/tests/snapshots/
git commit -m "test(pca-tier-b): D-4 — re-baseline 6 affected kernel snapshots"
```

- [ ] **Step D4-7: Push and open activation PR**

```bash
git push -u origin feat/pca-tier-b-dispatch
gh pr create --title "feat(pca-tier-b): activate planner-side dispatch (minimal heuristic; floor=<N>)" \
  --body "$(cat <<'EOF'
## Summary

Activates PCA Tier B emission for `segment_masked` configs at `seq_len >= TIER_B_SEQ_LEN_FLOOR` via a minimal heuristic in `flash_attention_v2::should_emit_tier_b`. Replaces the unconditional-false stub from PR #168/#169.

Per design spec `docs/superpowers/specs/2026-05-14-pca-tier-b-dispatch-design.md`:
- **§3.1 heuristic:** `config.segment_masked && seq_len >= floor`.
- **§4.2 new no-op invariant:** 1-arg form == 2-arg form with heuristic's choice.
- **§5 V-dispatch-integration:** option <N> per `docs/superpowers/specs/2026-05-14-tier-b-dispatch-integration-findings.md`.
- **§6 floor derivation:** `TIER_B_SEQ_LEN_FLOOR = <N>` per `docs/superpowers/specs/2026-05-14-tier-b-floor-derivation-findings.md` (<shape> curve).
- **§7 SegmentResidency::Tiled** as default — single-edit-point.

## Snapshot re-baseline (per IR-006)

**6 re-baselined:**
- [list each affected `.snap` file + 1-line description of the predicate-block addition]

**2 expected-stable (verified byte-identical, NOT re-baselined):**
- `pca_tier_b_preamble_isolation__*`
- `pca_tier_b_predicate_isolation__*`

These isolation tests call emission functions directly per PR #168's design; they do not route through `synthesize_flash_attention_ptx_v2`. Their byte-stability under dispatch activation verifies the call-graph property.

## Cascade narrative

Per design spec §8.2:
1. Heuristic returns `true` at fixture X (because `config.segment_masked = true` AND `seq_len >= floor`).
2. PTX gains the predicate-skip path (snapshot diff localized to predicate block).
3. Kernel output stays bit-identical (the 16 runtime-parity tests from PR #169 pass).

Each affected snapshot's diff is byte-localized to the predicate path — verified by hand per §8.2.

## Test plan

- [x] `cargo build -p nsl-codegen` clean
- [x] `cargo test -p nsl-codegen --test pca_tier_b_dispatch_heuristic` 4/4 pass
- [x] `cargo test -p nsl-codegen --test pca_tier_b_m3_parity` 6/6 backward parity pass
- [x] Forward parity tests 10/10 pass
- [x] `cargo test -p nsl-codegen` overall green (snapshots updated)
- [x] Isolation snapshots byte-identical (no `.snap.new` from preamble/predicate isolation tests)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

**Final gate:** PR opened with cascade-narrative discipline; runtime + snapshot tests green; 2 isolation snapshots unchanged.

---

## Out-of-scope reminders (per design spec §2.2)

- **TierBPolicy enum** (user-overrideable ForceOn/ForceOff). v2 — needs workload-driven trigger.
- **Planner module above codegen.** v3 — needs multi-variant Tier B.
- **Per-config SegmentResidency selection.** v2 — needs >10% wall-time regression measurement at a new workload.
- **Tier B-extended for seq_len > 16K**; **CTA-uniform predicate trade-off**; **tile-skip-aware backward checkpointing**. Per the original 2026-05-02 spec §11.

If any of these surface as load-bearing during D-3 or D-4, STOP — that's a v2 trigger, not a v1 scope expansion. Document the trigger in a new spec; do not in-line it here.

---

## Institutional rule extensions (per design spec §13)

After the activation PR merges, extend `docs/wiki/institutional-rules.md`:

- **IR-003 "Cited from" addition:** this spec §5 — V-dispatch-integration verification of caller behavior (first instance of external-system-state specialization).
- **IR-011 "Cited from" addition:** this spec §3.3 — sensitivity tier's saturating-curve finding enabled minimal-heuristic dispatch (second instance of sensitivity-tier institutional value, first being the "keep with sparsity gate" outcome).
- **IR-013 candidate (deferred):** caller-behavior verification specialization of IR-003. Narrow framing per design spec §13.2 — DO NOT promote without a second caller-behavior-specific instance materializing.

This is the registry-hygiene tail. Do not skip it — the registry is load-bearing for future spec authors per IR-NNN entry criterion.
