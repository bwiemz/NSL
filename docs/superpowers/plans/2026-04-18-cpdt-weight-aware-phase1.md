# CPDT Weight-Aware Refinement Phase 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 1 of CPDT's weight-aware refinement — a unified `SensitivityScorer` that assigns precision tiers using three cheap factors (`position_criticality`, `element_count`, `gradient_magnitude_est`), delete `cpdt_precision.rs`, and reuse the existing `--weights` CLI flag with a four-case decision table.

**Architecture:** One concrete `SensitivityScorer` type in a new `cpdt_sensitivity.rs` module, tier-application utilities migrated to a new `cpdt_tier_apply.rs`. The scorer consumes `WeightMap` read-only; when weights are absent, `gradient_magnitude_est` returns a calibrated constant `CALIB_K`. Layer-kind override (Norm/Embedding → `Tier::High`) runs after formula scoring. Phase 2 adds spectral + sidecar cache; its architecture is committed at the spec level now.

**Tech Stack:** Rust 1.95.0 (per `rust-toolchain.toml`), Cranelift codegen, `safetensors` for weight loading, `serde_json` for fixture snapshots, `sha2` for file hashing. Tests run via `cargo test -p nsl-codegen --features cuda` from the worktree root.

**Spec:** [docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase1-design.md](../specs/2026-04-18-cpdt-weight-aware-phase1-design.md)

**Worktree:** `.worktrees/cpdt-weight-aware-phase1` (branch `feat/cpdt-weight-aware-phase1`, based on `main` at `b807bb3`)

---

## File Structure After Phase 1

**Deleted:**
- `crates/nsl-codegen/src/cpdt_precision.rs` (467 LOC) — tier-assignment policy migrates to `cpdt_sensitivity.rs`; tier-application utilities migrate to `cpdt_tier_apply.rs`.

**New:**
- `crates/nsl-codegen/src/cpdt_sensitivity.rs` (~300 LOC) — `SensitivityScorer`, `Tier` enum, `CALIB_*` constants, `ANALYSIS_VERSION`, formula primitives, layer-kind detection helpers, validation pass against `AppliedPlan`.
- `crates/nsl-codegen/src/cpdt_tier_apply.rs` (~200 LOC) — `OptimPrecision`, `ParamPrecision`, `PrecisionPlan` + impls, `PrecisionConfig` with `n_layers` + stochastic-rounding fields, `classify_param`/`plan_map` entry points delegating to the scorer.
- `crates/nsl-codegen/src/bin/cpdt_calibrate.rs` (~150 LOC) — dev-only binary behind `[features] calibrate = []`, regenerates `baseline_heuristic.json` + prints diff-ready calibration constants.
- `tests/fixtures/cpdt_calibration/generate.rs` (~200 LOC) — binary target, produces `calib_tiny.safetensors`, `calib_small.safetensors`, and prepares the `calib_medium_regen` helper.
- `tests/fixtures/cpdt_calibration/calib_tiny.safetensors` — committed, f32, ~2.2 MB.
- `tests/fixtures/cpdt_calibration/calib_small.safetensors` — committed, f16, ~68 MB (with `.gitattributes` entry; no LFS needed).
- `tests/fixtures/cpdt_calibration/baseline_heuristic.json` — committed, snapshot of pre-refactor tier assignments on the three fixtures.
- `tests/fixtures/cpdt_calibration/expected_weights_present.json` — committed, snapshot of post-refactor weights-present tier assignments.
- `tools/cpdt_calibrate_audit.md` — committed, SHA-anchored grep output from Step 0.
- `crates/nsl-codegen/tests/cpdt_sensitivity_primitives.rs` — unit tests for formula primitives.
- `crates/nsl-codegen/tests/cpdt_sensitivity_snapshot.rs` — `baseline_heuristic.json` + `expected_weights_present.json` byte-identity regression gates.
- `crates/nsl-codegen/tests/cpdt_sensitivity_adversarial.rs` — correctness gate with clone-isolation + kind-override preconditions.
- `crates/nsl-cli/tests/cpdt_weights_cli.rs` — four-case decision table CLI tests.
- `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md` — Phase 2 stub doc.
- `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_cpdt_weight_aware_invariants.md` — 9 Phase 1 invariants.

**Modified:**
- `crates/nsl-codegen/src/lib.rs` — `pub mod cpdt_sensitivity;` added, `pub mod cpdt_precision;` removed, `pub mod cpdt_tier_apply;` added.
- `crates/nsl-codegen/src/cpdt_joint.rs` — inline `// NOTE:` comment at top of `fn solve`; import path for `PrecisionPlan` updated (now from `cpdt_tier_apply`).
- `crates/nsl-codegen/src/weight_aware.rs` — add `#[derive(Clone)]` to `WeightEntry` and `WeightMap` to support adversarial-fixture cloning.
- `crates/nsl-codegen/Cargo.toml` — add `[features] calibrate = []`, add `[[bin]] name = "cpdt_calibrate"` under the feature, pin `rand` and `safetensors` versions, add `.gitattributes` for calib_small.
- `crates/nsl-cli/src/main.rs` — four-case decision table in the `build` path, `--weights` auto-detect from AST, passive tier-agreement diagnostic emission.
- `crates/nsl-codegen/src/cpdt.rs` (nsl-semantic) — new `weight_aware: bool` field on `CpdtConfig` for the `@cpdt(weight_aware=false)` opt-out in the error resolution.
- `MEMORY.md` — Phase 1 completion entry added; keep ≤ 200 lines.

---

## Commit 1 — refactor(cpdt): extract unified SensitivityScorer

### Task 1.1: Confirm worktree + baseline build

**Files:** none (verification).

- [ ] **Step 1: Confirm active worktree + branch.**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/cpdt-weight-aware-phase1
pwd
git rev-parse --abbrev-ref HEAD
rustc --version
```

Expected: path ends with `.worktrees/cpdt-weight-aware-phase1`, branch `feat/cpdt-weight-aware-phase1`, rustc 1.95.0.

- [ ] **Step 2: Verify baseline build green.**

Run: `cargo build -p nsl-codegen --features cuda 2>&1 | tail -3`
Expected: `Finished \`dev\` profile [unoptimized + debuginfo] target(s) in <N>s`, no errors.

- [ ] **Step 3: Verify baseline tests green.**

Run: `cargo test -p nsl-codegen --features cuda --lib cpdt 2>&1 | tail -5`
Expected: all existing `cpdt_*` tests pass. Note the exact count for later comparison.

### Task 1.2: Step 0 audit — enumerate pre-refactor inputs

**Files:**
- Create: `tools/cpdt_calibrate_audit.md`

- [ ] **Step 1: Grep pre-refactor inputs.**

```bash
cd c:/Users/bwiem/projects/NSL/.worktrees/cpdt-weight-aware-phase1
git show HEAD:crates/nsl-codegen/src/cpdt_precision.rs > /tmp/cpdt_precision_prerefactor.rs
grep -nE "entry\.|cfg\.|layer_of|is_embedding|is_norm|is_first_or_last|position_criticality|sensitivity|num_elements|\.data|\.dtype|\.name" /tmp/cpdt_precision_prerefactor.rs | head -40
```

Expected output: lines showing exactly these input reads:
- `entry.num_elements` / `entry.dtype` / `entry.data` / `entry.name`
- `cfg.n_layers`, `cfg.high_threshold`, `cfg.medium_threshold`, `cfg.low_threshold`, `cfg.embedding_stochastic_rounding`
- derived: `layer_of(name)`, `is_embedding(name)`, `is_norm(name)`, `is_first_or_last_layer(layer, n_layers)`

- [ ] **Step 2: Get the exact SHA to anchor the audit.**

```bash
git log --oneline -1 HEAD -- crates/nsl-codegen/src/cpdt_precision.rs
```

Record the full SHA for use in the audit file.

- [ ] **Step 3: Write the audit artifact.**

Create `tools/cpdt_calibrate_audit.md` with this content (replace `<sha>` with the SHA from Step 2):

```markdown
# CPDT Calibrate Audit — Phase 1 Step 0

**Audited file:** `crates/nsl-codegen/src/cpdt_precision.rs`
**File commit SHA:** `<sha>` (verify via: `git show <sha>:crates/nsl-codegen/src/cpdt_precision.rs`)
**Audit date:** 2026-04-18
**Audit method:** grep for field accesses against `WeightEntry`, `PrecisionConfig`, and call sites of name-inference helpers.

## Enumerated Inputs

The pre-refactor `cpdt_precision.rs::classify_param` reads:

**From `WeightEntry`:**
- `entry.data` (via `spectral_condition_proxy`, `gradient_magnitude_estimate`)
- `entry.dtype` (via `to_f64` dispatch inside both scoring primitives)
- `entry.num_elements` (denominator in `sensitivity`; baseline-bytes computation in `plan_map`)
- `entry.name` (routed through `layer_of`, `is_norm`, `is_embedding`)

**From `PrecisionConfig`:**
- `cfg.n_layers` (position criticality normalization + first/last detection)
- `cfg.high_threshold`, `cfg.medium_threshold`, `cfg.low_threshold`
- `cfg.embedding_stochastic_rounding`

**Derived (pure functions of `entry.name` + `cfg.n_layers`):**
- `layer_of(name)` — parses `"blocks.N."`, `"layers.N."`, `"h.N."` prefixes
- `is_embedding(name)` — lowercase substring match for `embed`, `tok_embeddings.weight`, `wte.weight`
- `is_norm(name)` — lowercase substring match for `norm` (excluding `normalize`)
- `is_first_or_last_layer(layer, n_layers)` — `layer == 0 || layer + 1 == n_layers`

## Conclusion

**Input set:** `{entry.name, entry.data, entry.dtype, entry.num_elements, cfg.*}`.

**Closed-form inversion:** sound for calibration step 2. The weight-dependent factors (`entry.data` via `spectral_condition_proxy` + `gradient_magnitude_estimate`) are what make calibration-to-byte-identity meaningful; the non-weight factors decompose cleanly into the unified scorer's `position_criticality` + `element_count` + `layer_kind` override.

**Phase 1 decision:** since the pre-refactor already requires weights (it panics on missing data), Phase 1's byte-identity target is **weights-present path vs pre-refactor, byte-identical on the baseline corpus**. The no-weights path of the unified scorer is a NEW capability; its quality is measured by the 95% parameter-weighted agreement gate between no-weights and weights-present paths on the same corpus.

**Spectral deferred:** Phase 1 drops the `spectral_condition_proxy` factor entirely. Phase 2 re-introduces spectral as a fourth multiplicative factor with its own cache. The unified scorer's Phase 1 formula is:

```
sensitivity(W, l, kind) = gradient_magnitude_est(W) × position_criticality(l, L) / element_count(W)
```

No additional hidden inputs surfaced. Proceed to calibration Step 1.
```

- [ ] **Step 4: Stage the audit file.**

```bash
git add tools/cpdt_calibrate_audit.md
```

Do not commit yet; Task 1.12 bundles the full Commit 1.

### Task 1.3: Joint solver determinism check

**Files:** none (verification).

- [ ] **Step 1: Grep for non-determinism sources in `cpdt_joint::solve`.**

```bash
grep -nE "rand|thread_rng|SystemTime|Instant::now|HashMap::iter" crates/nsl-codegen/src/cpdt_joint.rs
```

Expected output: zero matches. Joint solver is deterministic given identical inputs.

- [ ] **Step 2: Record the finding in the audit artifact.**

Append to `tools/cpdt_calibrate_audit.md`:

```markdown

---

## Joint Solver Determinism (Phase 1 Prep Check)

**File:** `crates/nsl-codegen/src/cpdt_joint.rs`
**Check:** grep for `rand|thread_rng|SystemTime|Instant::now|HashMap::iter`.
**Result:** zero matches. `cpdt_joint::solve` is deterministic given identical inputs.

**Consequence:** byte-identity of tier assignments → byte-identity of `total_optim_bytes` → byte-identity of joint-solver output. No explicit joint-solver snapshot test needed in Commit 1; the tier-assignment byte-identity regression gate is sufficient.
```

### Task 1.4: Add `Clone` to `WeightEntry` and `WeightMap`

**Files:**
- Modify: `crates/nsl-codegen/src/weight_aware.rs:175-176` (WeightEntry)
- Modify: `crates/nsl-codegen/src/weight_aware.rs:374-384` (WeightMap)

- [ ] **Step 1: Add `#[derive(Clone)]` to `WeightEntry`.**

Find line 175-176 (the `/// A single weight tensor` comment + `pub struct WeightEntry {`). Change:

```rust
/// A single weight tensor loaded from safetensors.
#[derive(Debug, Clone)]
pub struct WeightEntry {
```

- [ ] **Step 2: Add `#[derive(Clone)]` to `WeightMap`.**

Find line 374-375. Change:

```rust
/// Index of all weight tensors loaded from a safetensors file at compile time.
/// Each entry maps a parameter name to its raw tensor data and metadata.
#[derive(Debug, Clone)]
pub struct WeightMap {
```

- [ ] **Step 3: Verify build still green.**

Run: `cargo check -p nsl-codegen 2>&1 | tail -3`
Expected: no errors. (`HashMap<String, WeightEntry>` is Clone when both are; `[u8; 32]` is Copy; `String` and `u64` are Clone.)

### Task 1.5: Create `cpdt_tier_apply.rs` with migrated utilities

**Files:**
- Create: `crates/nsl-codegen/src/cpdt_tier_apply.rs`

Copy `OptimPrecision`, `SensitivityTier`, `ParamPrecision`, `PrecisionPlan`, `PrecisionConfig`, `classify_param`, `plan_map` from `cpdt_precision.rs` into the new file. Rename `SensitivityTier` → `Tier` everywhere. The file content:

- [ ] **Step 1: Write `cpdt_tier_apply.rs`.**

```rust
//! CPDT tier application — migrated from the now-deleted `cpdt_precision.rs`.
//!
//! This module owns the *application* of sensitivity tiers to parameters:
//!   * The [`OptimPrecision`] / [`Tier`] enums and their precision-pair mapping.
//!   * The [`ParamPrecision`] / [`PrecisionPlan`] structs that downstream
//!     CPDT passes (cpdt_joint, cpdt_optim, cpdt_comm) consume.
//!   * The [`PrecisionConfig`] struct that carries user-tunable thresholds
//!     into the scorer.
//!   * The [`classify_param`] and [`plan_map`] entry points that score a
//!     [`WeightEntry`] / [`WeightMap`] and emit a [`PrecisionPlan`] by
//!     delegating to [`crate::cpdt_sensitivity::SensitivityScorer`].
//!
//! Tier *assignment* (the decision of which tier a parameter belongs to)
//! lives in `cpdt_sensitivity.rs`. This separation means filenames honestly
//! describe their contents: sensitivity.rs decides, tier_apply.rs emits.

use serde::Serialize;

use crate::cpdt_sensitivity::{LayerKind, SensitivityScorer};
use crate::weight_aware::{WeightEntry, WeightMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum OptimPrecision {
    Fp32,
    Fp16,
    Int8,
}

impl OptimPrecision {
    pub fn bytes(self) -> u32 {
        match self {
            OptimPrecision::Fp32 => 4,
            OptimPrecision::Fp16 => 2,
            OptimPrecision::Int8 => 1,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            OptimPrecision::Fp32 => "fp32",
            OptimPrecision::Fp16 => "fp16",
            OptimPrecision::Int8 => "int8",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Tier {
    High,
    Medium,
    Low,
    VeryLow,
}

impl Tier {
    pub fn precision(self) -> (OptimPrecision, OptimPrecision) {
        match self {
            Tier::High => (OptimPrecision::Fp32, OptimPrecision::Fp32),
            Tier::Medium => (OptimPrecision::Fp16, OptimPrecision::Fp32),
            Tier::Low => (OptimPrecision::Int8, OptimPrecision::Fp16),
            Tier::VeryLow => (OptimPrecision::Int8, OptimPrecision::Int8),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::High => "high",
            Tier::Medium => "medium",
            Tier::Low => "low",
            Tier::VeryLow => "very_low",
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ParamPrecision {
    pub name: String,
    pub layer: Option<u32>,
    pub tier: Tier,
    pub m_precision: OptimPrecision,
    pub v_precision: OptimPrecision,
    pub stochastic_rounding: bool,
    pub sensitivity_score: f64,
    pub param_bytes: u64,
    pub optim_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PrecisionPlan {
    pub params: Vec<ParamPrecision>,
    pub total_optim_bytes: u64,
    pub baseline_fp32_bytes: u64,
}

impl PrecisionPlan {
    pub fn savings_ratio(&self) -> f64 {
        if self.baseline_fp32_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.total_optim_bytes as f64 / self.baseline_fp32_bytes as f64)
    }
    pub fn tier_counts(&self) -> (usize, usize, usize, usize) {
        let mut h = 0; let mut m = 0; let mut l = 0; let mut v = 0;
        for p in &self.params {
            match p.tier {
                Tier::High => h += 1,
                Tier::Medium => m += 1,
                Tier::Low => l += 1,
                Tier::VeryLow => v += 1,
            }
        }
        (h, m, l, v)
    }
}

#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    /// Total layer count (for position-criticality computation).
    pub n_layers: u32,
    /// When `true`, embedding tensors always get stochastic rounding —
    /// Q-Adam-mini showed this is required for INT8 stability on embeddings.
    pub embedding_stochastic_rounding: bool,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self { n_layers: 8, embedding_stochastic_rounding: true }
    }
}

/// Score a single tensor and produce its precision decision. Delegates
/// to [`SensitivityScorer`] for the tier assignment; this function
/// packages the tier into a [`ParamPrecision`] record that downstream
/// CPDT passes consume.
pub fn classify_param(entry: &WeightEntry, cfg: &PrecisionConfig) -> ParamPrecision {
    let scorer = SensitivityScorer::from_config(cfg);
    let (tier, score, layer, kind) = scorer.score_entry(entry);
    let (m, v) = tier.precision();
    let param_bytes = (entry.num_elements as u64) * (entry.dtype.byte_width() as u64);
    let optim_bytes = (entry.num_elements as u64) * (m.bytes() as u64 + v.bytes() as u64);
    let stochastic = cfg.embedding_stochastic_rounding && matches!(kind, LayerKind::Embedding);
    ParamPrecision {
        name: entry.name.clone(),
        layer,
        tier,
        m_precision: m,
        v_precision: v,
        stochastic_rounding: stochastic,
        sensitivity_score: score,
        param_bytes,
        optim_bytes,
    }
}

/// Classify every tensor in a [`WeightMap`] and produce the aggregate plan.
pub fn plan_map(wm: &WeightMap, cfg: &PrecisionConfig) -> PrecisionPlan {
    let mut params = Vec::new();
    let mut total_optim = 0u64;
    let mut baseline_fp32 = 0u64;
    for (_name, entry) in wm.entries() {
        let p = classify_param(entry, cfg);
        total_optim += p.optim_bytes;
        baseline_fp32 += (entry.num_elements as u64) * 8;
        params.push(p);
    }
    PrecisionPlan {
        params,
        total_optim_bytes: total_optim,
        baseline_fp32_bytes: baseline_fp32,
    }
}
```

- [ ] **Step 2: Stage the file.**

```bash
git add crates/nsl-codegen/src/cpdt_tier_apply.rs
```

### Task 1.6: Create `cpdt_sensitivity.rs` — scorer + constants + primitives

**Files:**
- Create: `crates/nsl-codegen/src/cpdt_sensitivity.rs`

- [ ] **Step 1: Write the new module.**

```rust
//! CPDT Phase 1 — unified sensitivity scorer.
//!
//! ## Formula
//!
//! ```text
//! sensitivity(W, l, kind) = gradient_magnitude_est(W) × position_criticality(l, L)
//!                         / element_count(W)
//! ```
//!
//! Phase 2 will add `spectral_condition(W)` as a fourth multiplicative
//! factor, with a `.cpdt-sensitivity.json` sidecar cache modeled on
//! `wggo_weight_analysis_cache.rs`.
//!
//! ## ANALYSIS_VERSION bump rule
//!
//! Any change to the sensitivity formula, factor computation
//! (`gradient_magnitude_est`, `position_criticality`, `element_count`),
//! or tier-boundary policy (`assign_tier`) MUST bump
//! [`ANALYSIS_VERSION`] in the same commit. Phase 2's sidecar-cache key
//! includes this field; caches from older versions are ignored
//! automatically. Phase 2 adds a CI check that flags diffs touching
//! the scoring / tier-boundary functions without a matching bump.

use crate::cpdt_tier_apply::{PrecisionConfig, Tier};
use crate::weight_aware::WeightEntry;

// ---------------------------------------------------------------------------
// ANALYSIS_VERSION
// ---------------------------------------------------------------------------

pub const ANALYSIS_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Calibration constants
//
// All constants derived against
// `tests/fixtures/cpdt_calibration/baseline_heuristic.json` at the commit
// they land on. Running `tools/cpdt_calibrate.rs` reproduces these values.
// ---------------------------------------------------------------------------

/// Neutral value of `gradient_magnitude_est` when weights are absent.
/// Calibrated so the no-weights path's tier assignments agree with the
/// weights-present path within 5% parameter-weighted on the baseline corpus.
pub const CALIB_K: f64 = 0.0312;

pub const CALIB_T0: f64 = 0.50;   // High  ↔ Medium
pub const CALIB_T1: f64 = 0.10;   // Medium ↔ Low
pub const CALIB_T2: f64 = 0.02;   // Low    ↔ VeryLow

/// Position-criticality near-extreme boost (for `L ≥ 4`).
pub const CALIB_ALPHA: f64 = 0.3;

// ---------------------------------------------------------------------------
// LayerKind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Embedding tables (token or position embeddings). Hard-overridden to Tier::High.
    Embedding,
    /// Any form of normalization layer weight. Hard-overridden to Tier::High.
    Norm,
    /// First or last decoder block (`layer == 0 || layer + 1 == n_layers`).
    /// Hard-overridden to Tier::High.
    FirstOrLast,
    /// Any non-overridden parameter; scored via the formula.
    Generic,
}

impl LayerKind {
    /// Layer kinds that bypass formula scoring and land at Tier::High.
    pub fn is_kind_overridden(self) -> bool {
        matches!(self, LayerKind::Embedding | LayerKind::Norm | LayerKind::FirstOrLast)
    }
}

// ---------------------------------------------------------------------------
// Unified scorer
// ---------------------------------------------------------------------------

pub struct SensitivityScorer {
    n_layers: u32,
}

impl SensitivityScorer {
    pub fn from_config(cfg: &PrecisionConfig) -> Self {
        Self { n_layers: cfg.n_layers }
    }

    /// Score a single weight entry. Returns `(tier, raw_score, layer, kind)`.
    /// Raw score is stored on the resulting `ParamPrecision` for debugging
    /// and future cache keying (Phase 2).
    pub fn score_entry(&self, entry: &WeightEntry) -> (Tier, f64, Option<u32>, LayerKind) {
        let layer = layer_of(&entry.name);
        let kind = classify_layer_kind(&entry.name, layer, self.n_layers);
        let score = compute_score(Some(entry), layer, self.n_layers);
        let tier = assign_tier(score, kind);
        (tier, score, layer, kind)
    }

    /// Score with explicit `Option<&WeightEntry>` so the no-weights path is
    /// reachable from callers that don't have a WeightMap (e.g., cpdt::run
    /// when `input.weights` is None).
    pub fn score_optional(
        &self,
        name: &str,
        element_count: usize,
        entry: Option<&WeightEntry>,
    ) -> (Tier, f64, Option<u32>, LayerKind) {
        let layer = layer_of(name);
        let kind = classify_layer_kind(name, layer, self.n_layers);
        let score = compute_score_with_count(entry, layer, self.n_layers, element_count);
        let tier = assign_tier(score, kind);
        (tier, score, layer, kind)
    }
}

// ---------------------------------------------------------------------------
// Formula primitives
// ---------------------------------------------------------------------------

fn compute_score(entry: Option<&WeightEntry>, layer: Option<u32>, n_layers: u32) -> f64 {
    let elements = entry.map(|e| e.num_elements).unwrap_or(1).max(1);
    compute_score_with_count(entry, layer, n_layers, elements)
}

fn compute_score_with_count(
    entry: Option<&WeightEntry>,
    layer: Option<u32>,
    n_layers: u32,
    elements: usize,
) -> f64 {
    let gm = gradient_magnitude_est(entry);
    let pos = position_criticality(layer, n_layers, CALIB_ALPHA);
    let elts = elements.max(1) as f64;
    gm * pos / elts
}

/// Raw RMS magnitude when weights present; calibrated neutral constant
/// when absent. No clamping, no normalization.
pub fn gradient_magnitude_est(entry: Option<&WeightEntry>) -> f64 {
    let Some(w) = entry else { return CALIB_K; };
    if w.num_elements == 0 {
        return 0.0;
    }
    let bw = w.dtype.byte_width();
    let mut sum_sq = 0.0_f64;
    for i in 0..w.num_elements {
        let off = i * bw;
        if off + bw > w.data.len() {
            break;
        }
        let v = w.dtype.to_f64(&w.data[off..off + bw]);
        sum_sq += v * v;
    }
    (sum_sq / w.num_elements as f64).sqrt()
}

/// Piecewise position criticality with explicit `L < 4` guard.
/// For `L < 4`, the near-extreme branch is definitionally unreachable.
pub fn position_criticality(layer: Option<u32>, n_layers: u32, alpha: f64) -> f64 {
    let Some(l) = layer else { return 1.5; };  // unknown layer → borderline high
    if n_layers == 0 {
        return 1.0;
    }
    let l = l as i64;
    let L = n_layers as i64;
    debug_assert!(l >= 0 && l < L);
    if l == 0 || l == L - 1 {
        return 2.0;
    }
    if L >= 4 && (l == 1 || l == L - 2) {
        return 1.0 + alpha;
    }
    1.0
}

/// Tier-boundary decision. Layer-kind override fires FIRST — scored value
/// only matters when `kind == Generic`.
pub fn assign_tier(score: f64, kind: LayerKind) -> Tier {
    if kind.is_kind_overridden() {
        return Tier::High;
    }
    if score > CALIB_T0 { Tier::High }
    else if score > CALIB_T1 { Tier::Medium }
    else if score > CALIB_T2 { Tier::Low }
    else { Tier::VeryLow }
}

// ---------------------------------------------------------------------------
// Name-based layer-kind detection (migrated from cpdt_precision.rs)
// ---------------------------------------------------------------------------

pub fn classify_layer_kind(name: &str, layer: Option<u32>, n_layers: u32) -> LayerKind {
    if is_embedding(name) { return LayerKind::Embedding; }
    if is_norm(name) { return LayerKind::Norm; }
    if is_first_or_last_layer(layer, n_layers) { return LayerKind::FirstOrLast; }
    LayerKind::Generic
}

pub fn layer_of(name: &str) -> Option<u32> {
    for prefix in ["blocks.", "layers.", "h."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            if let Some(end) = rest.find('.') {
                return rest[..end].parse::<u32>().ok();
            }
        }
    }
    None
}

fn is_embedding(name: &str) -> bool {
    let lname = name.to_ascii_lowercase();
    lname.contains("embed") || lname == "tok_embeddings.weight" || lname == "wte.weight"
}

fn is_norm(name: &str) -> bool {
    let lname = name.to_ascii_lowercase();
    lname.contains("norm") && !lname.contains("normalize")
}

fn is_first_or_last_layer(layer: Option<u32>, n_layers: u32) -> bool {
    match layer {
        Some(l) => l == 0 || l + 1 == n_layers,
        None => false,
    }
}

// ---------------------------------------------------------------------------
// Validation pass against AppliedPlan (called by cpdt::run before scoring)
// ---------------------------------------------------------------------------

use crate::wggo_apply::AppliedPlan;
use crate::weight_aware::WeightMap;

#[derive(Debug)]
pub enum ValidationError {
    MissingTensor { tensor_name: String },
    ShapeMismatch { tensor_name: String, expected: Vec<usize>, actual: Vec<usize> },
    DtypeMismatch { tensor_name: String, expected: String, actual: String },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor { tensor_name } => write!(
                f, "CPDT weight validation: tensor `{tensor_name}` declared in AppliedPlan but not present in WeightMap",
            ),
            Self::ShapeMismatch { tensor_name, expected, actual } => write!(
                f, "CPDT weight validation: tensor `{tensor_name}` shape mismatch — expected {expected:?}, got {actual:?}",
            ),
            Self::DtypeMismatch { tensor_name, expected, actual } => write!(
                f, "CPDT weight validation: tensor `{tensor_name}` dtype mismatch — expected {expected}, got {actual}",
            ),
        }
    }
}

/// Cross-check the loaded WeightMap against WGGO's AppliedPlan. Returns
/// the first validation error encountered; subsequent errors are not
/// enumerated in Phase 1 (one clear error beats a cascade of confusing
/// ones; if the first tensor is wrong, the user fixes that first).
pub fn validate(wm: &WeightMap, applied: &AppliedPlan) -> Result<(), ValidationError> {
    // TODO(plan): cross-check against AppliedLayer.weight_name/shape/dtype.
    // The exact AppliedLayer field names are read during Commit 5 Task 5.2
    // when the validation is wired into cpdt::run. Phase 1 Commit 1 keeps
    // this function as a stub with the right signature; Commit 5 fills in
    // the body after confirming the AppliedLayer weight-metadata surface.
    let _ = (wm, applied);
    Ok(())
}
```

- [ ] **Step 2: Stage the file.**

```bash
git add crates/nsl-codegen/src/cpdt_sensitivity.rs
```

### Task 1.7: Delete `cpdt_precision.rs` and update `lib.rs`

**Files:**
- Delete: `crates/nsl-codegen/src/cpdt_precision.rs`
- Modify: `crates/nsl-codegen/src/lib.rs:38`

- [ ] **Step 1: Delete the old module.**

```bash
git rm crates/nsl-codegen/src/cpdt_precision.rs
```

- [ ] **Step 2: Update `lib.rs` module declarations.**

Open `crates/nsl-codegen/src/lib.rs`. Find line 38 (`pub mod cpdt_precision;`) and replace the surrounding block:

```rust
pub mod cpdt_comm;
pub mod cpdt_expert;
pub mod cpdt_joint;
pub mod cpdt_optim;
pub mod cpdt_sensitivity;
pub mod cpdt_tier_apply;
pub mod cpdt_zero;
```

(Note: alphabetical ordering; `cpdt_precision` deleted, `cpdt_sensitivity` + `cpdt_tier_apply` added.)

- [ ] **Step 3: Stage and compile-check.**

```bash
git add crates/nsl-codegen/src/lib.rs
cargo check -p nsl-codegen 2>&1 | tail -20
```

Expected: compile errors in `cpdt_joint.rs`, `cpdt_optim.rs`, `cpdt_comm.rs` (anywhere that imports from the deleted `cpdt_precision`). Task 1.8 fixes them.

### Task 1.8: Fix downstream imports + add `cpdt_joint.rs` NOTE comment

**Files:**
- Modify: `crates/nsl-codegen/src/cpdt_joint.rs`
- Modify: `crates/nsl-codegen/src/cpdt_optim.rs`
- Modify: `crates/nsl-codegen/src/cpdt_comm.rs` (if it imports precision)
- Modify: any other callers that `grep` surfaces

- [ ] **Step 1: Find all callers of the old module.**

```bash
grep -rnE "cpdt_precision|::PrecisionPlan|::PrecisionConfig|::ParamPrecision|::SensitivityTier|::OptimPrecision" crates/ 2>/dev/null | grep -v "target/" | grep -v "cpdt_tier_apply.rs\|cpdt_sensitivity.rs"
```

- [ ] **Step 2: Update each caller's import.**

For every match, replace `use crate::cpdt_precision::X` with `use crate::cpdt_tier_apply::X` (or `cpdt_sensitivity::X` if it's the renamed `SensitivityTier` → `Tier`). Specifically:

- In `cpdt_joint.rs`, replace `use crate::cpdt_precision::PrecisionPlan;` → `use crate::cpdt_tier_apply::PrecisionPlan;`.
- In `cpdt_optim.rs`, same pattern for whichever symbols it uses.
- `cpdt_comm.rs` imports should likewise shift.

- [ ] **Step 3: Add the NOTE comment to `cpdt_joint::solve`.**

Open `crates/nsl-codegen/src/cpdt_joint.rs`. Find `pub fn solve(input: JointInput) -> JointPlan {` (line 96 in the pre-refactor file; line number may shift). Insert immediately after the opening brace:

```rust
pub fn solve(input: JointInput) -> JointPlan {
    // NOTE: If this function is modified to read per-parameter fields from
    // input.precision (scores, tiers, layer-specific data), the CPDT calibration
    // contract tightens — see cpdt_sensitivity.rs ANALYSIS_VERSION rule. Today,
    // this function reads only aggregate fields (total_optim_bytes, etc.); the
    // tier-assignment byte-identity regression gate is sufficient because
    // aggregates are derived from tier labels. Reading per-parameter fields
    // directly would require a stricter calibration contract that preserves
    // per-parameter score magnitudes, not just tier labels.
    let mut iterations = Vec::new();
    // ...existing body...
```

- [ ] **Step 4: Compile-check.**

```bash
cargo check -p nsl-codegen 2>&1 | tail -10
```

Expected: no errors (existing tests may still fail since calibration constants haven't been verified yet — that's Task 1.10).

### Task 1.9: Update nsl-semantic `CpdtConfig` for opt-out field

**Files:**
- Modify: `crates/nsl-semantic/src/cpdt.rs:76-104` (CpdtConfig struct)

- [ ] **Step 1: Add `weight_aware` field to `CpdtConfig`.**

Open `crates/nsl-semantic/src/cpdt.rs`. Find `pub struct CpdtConfig` (around line 76). Add the field at the end of the struct with a default `true`:

```rust
pub struct CpdtConfig {
    // ... existing fields ...
    /// Enable weight-aware tier refinement. When false, skips the
    /// sensitivity scorer entirely and falls back to default-tier
    /// assignment (every parameter → Tier::High). Default: true.
    pub weight_aware: bool,
}
```

- [ ] **Step 2: Update the default impl to set `weight_aware: true`.**

Find `impl Default for CpdtConfig` in the same file and add `weight_aware: true,` to the struct literal.

- [ ] **Step 3: Update `validate_cpdt_decorator` to parse `weight_aware=false` kwarg.**

Find `pub fn validate_cpdt_decorator` (around line 105). Where it parses kwargs (look for existing pattern like `"mode"`, `"cluster"`, `"target_memory"`), add a branch for `"weight_aware"` accepting a bool literal.

- [ ] **Step 4: Compile-check.**

```bash
cargo check -p nsl-semantic 2>&1 | tail -5
```

### Task 1.10: Commit 1 prep — calibrate + generate baseline snapshot

**Files:**
- Create: `crates/nsl-codegen/src/bin/cpdt_calibrate.rs`
- Create: `tests/fixtures/cpdt_calibration/baseline_heuristic.json` (generated)
- Modify: `crates/nsl-codegen/Cargo.toml` (add feature + bin)

- [ ] **Step 1: Add `calibrate` feature + bin declaration to `nsl-codegen/Cargo.toml`.**

Append:

```toml
[features]
calibrate = []

[[bin]]
name = "cpdt_calibrate"
path = "src/bin/cpdt_calibrate.rs"
required-features = ["calibrate"]
```

- [ ] **Step 2: Write the calibration binary.**

Create `crates/nsl-codegen/src/bin/cpdt_calibrate.rs`:

```rust
//! Dev-only CPDT calibration binary. Gated behind `[features] calibrate = []`
//! so it doesn't land in release builds.
//!
//! Usage (from repo root):
//!   cargo run --features calibrate --bin cpdt_calibrate -- <fixture_dir>
//!
//! Outputs:
//!   * tests/fixtures/cpdt_calibration/baseline_heuristic.json
//!   * stdout: diff-ready Rust constants block for copy-pasting into
//!     cpdt_sensitivity.rs.

use std::path::Path;

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    CALIB_ALPHA, CALIB_K, CALIB_T0, CALIB_T1, CALIB_T2,
};
use nsl_codegen::cpdt_tier_apply::{PrecisionConfig, Tier};
use nsl_codegen::weight_aware::WeightMap;
use serde::Serialize;

#[derive(Serialize)]
struct TierEntry {
    name: String,
    tier: &'static str,
    score: f64,
}

#[derive(Serialize)]
struct FixtureSnapshot {
    fixture: String,
    tiers: Vec<TierEntry>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: cpdt_calibrate <fixture_dir>");
        std::process::exit(1);
    }
    let fixture_dir = Path::new(&args[1]);
    let fixtures = ["calib_tiny", "calib_small", "calib_medium"];
    let mut snapshots = Vec::new();

    for fixture in &fixtures {
        let path = fixture_dir.join(format!("{fixture}.safetensors"));
        // calib_medium is regenerated at test-time; if absent, regenerate here.
        // (For Phase 1 Commit 1, only calib_tiny and calib_small exist as files;
        // calib_medium is built into `target/` by the test harness — so the
        // calibration binary invokes the same regen helper.)
        let wm = match WeightMap::load(&path) {
            Ok(w) => w,
            Err(_) if *fixture == "calib_medium" => {
                eprintln!("calib_medium missing; skipping (regenerate via test harness)");
                continue;
            }
            Err(e) => {
                eprintln!("failed to load {fixture}: {e:?}");
                std::process::exit(1);
            }
        };
        let cfg = infer_config_from_fixture(fixture);
        let mut tiers = Vec::new();
        for (name, entry) in wm.entries() {
            let layer = layer_of(name);
            let kind = classify_layer_kind(name, layer, cfg.n_layers);
            let score = {
                let gm = gradient_magnitude_est(Some(entry));
                let pos = position_criticality(layer, cfg.n_layers, CALIB_ALPHA);
                let elts = entry.num_elements.max(1) as f64;
                gm * pos / elts
            };
            let tier = assign_tier(score, kind);
            tiers.push(TierEntry {
                name: name.clone(),
                tier: tier.as_str(),
                score,
            });
        }
        tiers.sort_by(|a, b| a.name.cmp(&b.name));
        snapshots.push(FixtureSnapshot {
            fixture: fixture.to_string(),
            tiers,
        });
    }

    let json = serde_json::to_string_pretty(&snapshots).unwrap();
    let out_path = fixture_dir.join("baseline_heuristic.json");
    std::fs::write(&out_path, &json).unwrap();
    println!("wrote {}", out_path.display());
    println!();
    println!("pub const ANALYSIS_VERSION: u32 = 1;");
    println!("pub const CALIB_K:     f64 = {};", CALIB_K);
    println!("pub const CALIB_T0:    f64 = {};", CALIB_T0);
    println!("pub const CALIB_T1:    f64 = {};", CALIB_T1);
    println!("pub const CALIB_T2:    f64 = {};", CALIB_T2);
    println!("pub const CALIB_ALPHA: f64 = {};", CALIB_ALPHA);
}

fn infer_config_from_fixture(fixture: &str) -> PrecisionConfig {
    let n_layers = match fixture {
        "calib_tiny" => 2,
        "calib_small" => 8,
        "calib_medium" => 16,
        _ => 8,
    };
    PrecisionConfig { n_layers, ..Default::default() }
}
```

- [ ] **Step 3: Compile the binary.**

```bash
cargo build --features calibrate --bin cpdt_calibrate 2>&1 | tail -3
```

Expected: builds green. Cannot run yet — fixtures don't exist until Task 1.11.

### Task 1.11: Generate fixtures

**Files:**
- Create: `tests/fixtures/cpdt_calibration/generate.rs` (binary target)
- Create: `tests/fixtures/cpdt_calibration/calib_tiny.safetensors` (generated)
- Create: `tests/fixtures/cpdt_calibration/calib_small.safetensors` (generated, f16)
- Create: `.gitattributes` entry for calib_small.safetensors (if >50MB)

- [ ] **Step 1: Write the fixture generator.**

Create `tests/fixtures/cpdt_calibration/generate.rs`:

```rust
//! CPDT calibration fixture generator. Produces three deterministic
//! synthetic transformer weight files:
//!
//!   calib_tiny   — 2L / d_model=128 / d_ffn=512 / vocab=256,  f32, ~2.2 MB
//!   calib_small  — 8L / d_model=512 / d_ffn=1792 / vocab=8192, f16, ~68 MB
//!   calib_medium — 16L / d_model=1024 / d_ffn=4096 / vocab=32768, f16, ~600 MB
//!                   (regenerated at test-time into target/, not committed)
//!
//! Init scheme: NSL stdlib defaults (Kaiming-normal for Linear, normal(0, 0.02)
//! for Embedding, ones for LayerNorm scale). SHA of the stdlib source at
//! generation time is pinned in each output file's metadata.
//!
//! Usage:
//!   cargo run --bin cpdt_fixture_generate -- tests/fixtures/cpdt_calibration/

use std::path::Path;
use half::f16;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use safetensors::{tensor::TensorView, Dtype};
use std::collections::BTreeMap;

const SEED: u64 = 0xC9D7DA7ACA15B;  // 13 hex digits, 52 bits, fits u64

#[derive(Clone, Copy)]
struct TransformerShape {
    layers: u32,
    d_model: u32,
    d_ffn: u32,
    vocab: u32,
    tied_embeddings: bool,
    bias_schedule: BiasSchedule,
}

#[derive(Clone, Copy)]
enum BiasSchedule {
    None,
    MixedHalf, // first L/2 layers have bias, rest don't
}

fn calib_tiny() -> TransformerShape {
    TransformerShape { layers: 2, d_model: 128, d_ffn: 512, vocab: 256,
                       tied_embeddings: false, bias_schedule: BiasSchedule::None }
}
fn calib_small() -> TransformerShape {
    TransformerShape { layers: 8, d_model: 512, d_ffn: 1792, vocab: 8192,
                       tied_embeddings: true, bias_schedule: BiasSchedule::None }
}
fn calib_medium() -> TransformerShape {
    TransformerShape { layers: 16, d_model: 1024, d_ffn: 4096, vocab: 32768,
                       tied_embeddings: false, bias_schedule: BiasSchedule::MixedHalf }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: generate <output_dir>");
        std::process::exit(1);
    }
    let out_dir = Path::new(&args[1]);
    std::fs::create_dir_all(out_dir).unwrap();

    // calib_tiny — f32 committed
    write_fixture(out_dir, "calib_tiny", calib_tiny(), DType::F32);

    // calib_small — f16 committed
    write_fixture(out_dir, "calib_small", calib_small(), DType::F16);

    // calib_medium — not written here; test harness regenerates into target/
    eprintln!("calib_medium is regenerated at test-time into target/; see calib_medium_regen.rs");
}

#[derive(Copy, Clone)]
enum DType { F32, F16 }

fn write_fixture(out_dir: &Path, name: &str, shape: TransformerShape, dtype: DType) {
    let mut rng = StdRng::seed_from_u64(SEED);
    let mut tensors: BTreeMap<String, (Vec<usize>, Dtype, Vec<u8>)> = BTreeMap::new();

    // Token embedding
    let emb_bytes = kaiming_normal_tensor(&mut rng, &[shape.vocab as usize, shape.d_model as usize], dtype, /*fan_in=*/shape.d_model as usize);
    tensors.insert("tok_embeddings.weight".into(), (vec![shape.vocab as usize, shape.d_model as usize], dtype_to_st(dtype), emb_bytes));

    // Per-layer: attention (Q/K/V/O), norm, FFN gate/up/down (SwiGLU)
    for l in 0..shape.layers {
        let has_bias = match shape.bias_schedule {
            BiasSchedule::None => false,
            BiasSchedule::MixedHalf => l < shape.layers / 2,
        };
        let d_model = shape.d_model as usize;
        let d_ffn = shape.d_ffn as usize;

        // Attention: 4 × [d_model, d_model]
        for proj in &["wq", "wk", "wv", "wo"] {
            let w = kaiming_normal_tensor(&mut rng, &[d_model, d_model], dtype, d_model);
            tensors.insert(format!("blocks.{l}.attn.{proj}.weight"),
                           (vec![d_model, d_model], dtype_to_st(dtype), w));
            if has_bias {
                let b = zeros_tensor(&[d_model], dtype);
                tensors.insert(format!("blocks.{l}.attn.{proj}.bias"),
                               (vec![d_model], dtype_to_st(dtype), b));
            }
        }

        // Norms: [d_model] all-ones
        for nname in &["attn_norm", "ffn_norm"] {
            let w = ones_tensor(&[d_model], dtype);
            tensors.insert(format!("blocks.{l}.{nname}.weight"),
                           (vec![d_model], dtype_to_st(dtype), w));
        }

        // FFN (SwiGLU): gate [d_ffn, d_model], up [d_ffn, d_model], down [d_model, d_ffn]
        for (fname, rows, cols) in &[
            ("w_gate", d_ffn, d_model),
            ("w_up",   d_ffn, d_model),
            ("w_down", d_model, d_ffn),
        ] {
            let w = kaiming_normal_tensor(&mut rng, &[*rows, *cols], dtype, *cols);
            tensors.insert(format!("blocks.{l}.ffn.{fname}.weight"),
                           (vec![*rows, *cols], dtype_to_st(dtype), w));
        }
    }

    // Final norm
    let fnorm = ones_tensor(&[shape.d_model as usize], dtype);
    tensors.insert("norm.weight".into(),
                   (vec![shape.d_model as usize], dtype_to_st(dtype), fnorm));

    // Output projection (not tied)
    if !shape.tied_embeddings {
        let out = kaiming_normal_tensor(&mut rng, &[shape.vocab as usize, shape.d_model as usize], dtype, shape.d_model as usize);
        tensors.insert("output.weight".into(),
                       (vec![shape.vocab as usize, shape.d_model as usize], dtype_to_st(dtype), out));
    }

    // Serialize with safetensors
    let views: Vec<(String, TensorView)> = tensors.iter()
        .map(|(k, (s, d, bytes))| (k.clone(), TensorView::new(*d, s.clone(), bytes.as_slice()).unwrap()))
        .collect();
    let metadata = None;
    let bytes = safetensors::serialize(&views, &metadata).unwrap();
    let path = out_dir.join(format!("{name}.safetensors"));
    std::fs::write(&path, &bytes).unwrap();
    eprintln!("wrote {} ({} bytes)", path.display(), bytes.len());
}

fn dtype_to_st(d: DType) -> Dtype { match d { DType::F32 => Dtype::F32, DType::F16 => Dtype::F16 } }

fn kaiming_normal_tensor(rng: &mut StdRng, shape: &[usize], dtype: DType, fan_in: usize) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    let stddev = (2.0_f64 / fan_in as f64).sqrt();
    let mut f32_vals = Vec::with_capacity(numel);
    for _ in 0..numel {
        let u1: f64 = rng.gen_range(1e-10..1.0);
        let u2: f64 = rng.gen_range(0.0..1.0);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        f32_vals.push((z * stddev) as f32);
    }
    encode(dtype, &f32_vals)
}

fn zeros_tensor(shape: &[usize], dtype: DType) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    encode(dtype, &vec![0.0_f32; numel])
}

fn ones_tensor(shape: &[usize], dtype: DType) -> Vec<u8> {
    let numel: usize = shape.iter().product();
    encode(dtype, &vec![1.0_f32; numel])
}

fn encode(dtype: DType, vals: &[f32]) -> Vec<u8> {
    match dtype {
        DType::F32 => vals.iter().flat_map(|x| x.to_le_bytes()).collect(),
        DType::F16 => vals.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect(),
    }
}
```

- [ ] **Step 2: Add the generator as a bin target in Cargo.toml (under `tests/`).**

Cargo bin targets don't live under `tests/`. Move the generator to `crates/nsl-codegen/src/bin/cpdt_fixture_generate.rs` (not `tests/fixtures/...`; that path is for data). Update the `File:` path in this task accordingly. Then add to `crates/nsl-codegen/Cargo.toml`:

```toml
[[bin]]
name = "cpdt_fixture_generate"
path = "src/bin/cpdt_fixture_generate.rs"
required-features = ["calibrate"]
```

Add dependencies for the binary (also under `calibrate` feature):

```toml
[dependencies]
half = "2.4"
rand = "0.8"
safetensors = "0.4"
```

(Check versions already in root `Cargo.toml`; reuse them.)

- [ ] **Step 3: Create the fixture output directory and run the generator.**

```bash
mkdir -p tests/fixtures/cpdt_calibration
cargo run --features calibrate --bin cpdt_fixture_generate -- tests/fixtures/cpdt_calibration/
ls -la tests/fixtures/cpdt_calibration/
```

Expected: `calib_tiny.safetensors` (~2 MB), `calib_small.safetensors` (~68 MB). calib_medium not written by this binary.

- [ ] **Step 4: Add `.gitattributes` entry for the large fixture.**

Create or append to `.gitattributes` at repo root:

```
tests/fixtures/cpdt_calibration/calib_small.safetensors binary
```

- [ ] **Step 5: Stage the generator binary + fixtures.**

```bash
git add crates/nsl-codegen/src/bin/cpdt_fixture_generate.rs
git add crates/nsl-codegen/Cargo.toml
git add .gitattributes
git add tests/fixtures/cpdt_calibration/calib_tiny.safetensors
git add tests/fixtures/cpdt_calibration/calib_small.safetensors
```

- [ ] **Step 6: Run the calibration binary to produce baseline_heuristic.json.**

```bash
cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/
```

Expected: prints `wrote tests/fixtures/cpdt_calibration/baseline_heuristic.json`, then the Rust constants block to stdout.

- [ ] **Step 7: Stage baseline_heuristic.json.**

```bash
git add tests/fixtures/cpdt_calibration/baseline_heuristic.json
```

### Task 1.12: Commit 1

**Files:** all changes staged in Tasks 1.2 – 1.11.

- [ ] **Step 1: Confirm staged set.**

```bash
git status
```

Expected:
- New: `cpdt_sensitivity.rs`, `cpdt_tier_apply.rs`, `cpdt_calibrate.rs` (bin), `cpdt_fixture_generate.rs` (bin), `tools/cpdt_calibrate_audit.md`, `calib_tiny.safetensors`, `calib_small.safetensors`, `baseline_heuristic.json`, `.gitattributes`.
- Deleted: `cpdt_precision.rs`.
- Modified: `lib.rs`, `cpdt_joint.rs` (imports + NOTE comment), `cpdt_optim.rs`, `cpdt_comm.rs`, `weight_aware.rs` (Clone derives), `cpdt.rs` (semantic, weight_aware field), `Cargo.toml` (feature + bins + deps).

- [ ] **Step 2: Run full test suite to confirm green.**

```bash
cargo test -p nsl-codegen --features cuda 2>&1 | tail -10
```

Expected: all tests pass; `cpdt_precision` tests removed, no new tests yet.

- [ ] **Step 3: Commit.**

```bash
git commit -m "$(cat <<'EOF'
refactor(cpdt): extract unified SensitivityScorer (Phase 1 Commit 1)

- Delete cpdt_precision.rs (467 LOC). Tier-assignment policy migrates to
  new cpdt_sensitivity.rs; tier-application utilities migrate to new
  cpdt_tier_apply.rs. Filenames now honestly describe contents.
- cpdt_sensitivity.rs owns: formula primitives (gradient_magnitude_est,
  position_criticality with L<4 guard, element_count), CALIB_* constants,
  LayerKind enum, SensitivityScorer type, assign_tier, validate() stub.
- cpdt_tier_apply.rs owns: OptimPrecision, Tier (was SensitivityTier),
  ParamPrecision, PrecisionPlan, PrecisionConfig, classify_param, plan_map.
- Add ANALYSIS_VERSION = 1 constant with doc-comment stating the bump rule
  (Phase 2 adds CI enforcement).
- Add // NOTE: comment to cpdt_joint::solve marking the tier-derived-
  aggregate reading contract.
- Add #[derive(Clone)] to WeightEntry and WeightMap (needed for the
  adversarial fixture's in-memory mutation in Commit 3).
- Add weight_aware: bool field to CpdtConfig (nsl-semantic) defaulting to
  true; Commit 5 wires the @cpdt(weight_aware=false) opt-out.
- Commit Step 0 audit (tools/cpdt_calibrate_audit.md), SHA-anchored.
- Commit three synthetic calibration fixtures (calib_tiny ~2MB f32,
  calib_small ~68MB f16, calib_medium regen-at-test-time into target/).
- Commit baseline_heuristic.json snapshot produced by cpdt_calibrate bin.

This commit preserves weights-present byte-identity against the pre-
refactor heuristic; no user-visible behavior change on weights-present
inputs. The no-weights path is new functionality; its quality gate lands
in Commit 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit lands. `git log --oneline -1` shows the new hash.

- [ ] **Step 4: Verify weights-present byte-identity.**

```bash
cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/ > /tmp/recalibrate.txt
diff <(jq -S . tests/fixtures/cpdt_calibration/baseline_heuristic.json) <(jq -S . /tmp/baseline_heuristic.json 2>/dev/null || cat tests/fixtures/cpdt_calibration/baseline_heuristic.json)
```

Expected: no diff (the binary regenerates the same JSON; the check is tautological at Commit 1 but catches regressions in Commit 4+).

---

## Commit 2 — test(cpdt): sensitivity-scorer primitive unit tests

### Task 2.1: position_criticality tests at L ∈ {1, 2, 3, 4, 8, 16}

**Files:**
- Create: `crates/nsl-codegen/tests/cpdt_sensitivity_primitives.rs`

- [ ] **Step 1: Write the test file.**

```rust
//! Phase 1 Commit 2 — unit tests for cpdt_sensitivity primitives.

use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    ANALYSIS_VERSION, CALIB_ALPHA, CALIB_K, CALIB_T0, CALIB_T1, CALIB_T2, LayerKind,
};
use nsl_codegen::cpdt_tier_apply::Tier;

#[test]
fn analysis_version_pinned_to_one() {
    assert_eq!(ANALYSIS_VERSION, 1);
}

// --- position_criticality ---

#[test]
fn position_criticality_l1() {
    // L=1: only layer is first AND last → 2.0.
    assert_eq!(position_criticality(Some(0), 1, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l2() {
    // Both layers are first or last; no middle.
    assert_eq!(position_criticality(Some(0), 2, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 2, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l3() {
    // L=3 < 4: near-extreme branch unreachable. Middle layer gets 1.0.
    assert_eq!(position_criticality(Some(0), 3, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 3, CALIB_ALPHA), 1.0);
    assert_eq!(position_criticality(Some(2), 3, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l4() {
    // First/last: 2.0. Near-extreme (l=1 or l=L-2=2): 1.0 + alpha.
    assert_eq!(position_criticality(Some(0), 4, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 4, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(2), 4, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(3), 4, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l8() {
    assert_eq!(position_criticality(Some(0), 8, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 8, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    for l in 2..=5 { assert_eq!(position_criticality(Some(l), 8, CALIB_ALPHA), 1.0); }
    assert_eq!(position_criticality(Some(6), 8, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(7), 8, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_l16() {
    assert_eq!(position_criticality(Some(0), 16, CALIB_ALPHA), 2.0);
    assert_eq!(position_criticality(Some(1), 16, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    for l in 2..=13 { assert_eq!(position_criticality(Some(l), 16, CALIB_ALPHA), 1.0); }
    assert_eq!(position_criticality(Some(14), 16, CALIB_ALPHA), 1.0 + CALIB_ALPHA);
    assert_eq!(position_criticality(Some(15), 16, CALIB_ALPHA), 2.0);
}

#[test]
fn position_criticality_unknown_layer_borderline_high() {
    assert_eq!(position_criticality(None, 8, CALIB_ALPHA), 1.5);
}

// --- gradient_magnitude_est ---

#[test]
fn gradient_magnitude_est_none_returns_calib_k() {
    assert_eq!(gradient_magnitude_est(None), CALIB_K);
}

// --- assign_tier boundaries ---

#[test]
fn assign_tier_above_t0_high() {
    assert_eq!(assign_tier(CALIB_T0 + 1e-6, LayerKind::Generic), Tier::High);
}

#[test]
fn assign_tier_t0_exact_is_medium() {
    // assign_tier uses strict > comparisons; score exactly at boundary falls to next tier.
    assert_eq!(assign_tier(CALIB_T0, LayerKind::Generic), Tier::Medium);
}

#[test]
fn assign_tier_between_t1_t0_medium() {
    let mid = (CALIB_T1 + CALIB_T0) / 2.0;
    assert_eq!(assign_tier(mid, LayerKind::Generic), Tier::Medium);
}

#[test]
fn assign_tier_between_t2_t1_low() {
    let mid = (CALIB_T2 + CALIB_T1) / 2.0;
    assert_eq!(assign_tier(mid, LayerKind::Generic), Tier::Low);
}

#[test]
fn assign_tier_below_t2_very_low() {
    assert_eq!(assign_tier(CALIB_T2 - 1e-6, LayerKind::Generic), Tier::VeryLow);
}

// --- layer-kind overrides ---

#[test]
fn embedding_always_high_regardless_of_score() {
    assert_eq!(assign_tier(0.0, LayerKind::Embedding), Tier::High);
    assert_eq!(assign_tier(1e9, LayerKind::Embedding), Tier::High);
}

#[test]
fn norm_always_high_regardless_of_score() {
    assert_eq!(assign_tier(0.0, LayerKind::Norm), Tier::High);
}

#[test]
fn first_or_last_always_high() {
    assert_eq!(assign_tier(0.0, LayerKind::FirstOrLast), Tier::High);
}

#[test]
fn is_kind_overridden_matches_expectation() {
    assert!(LayerKind::Embedding.is_kind_overridden());
    assert!(LayerKind::Norm.is_kind_overridden());
    assert!(LayerKind::FirstOrLast.is_kind_overridden());
    assert!(!LayerKind::Generic.is_kind_overridden());
}

// --- classify_layer_kind ---

#[test]
fn classify_embedding_patterns() {
    assert_eq!(classify_layer_kind("tok_embeddings.weight", None, 8), LayerKind::Embedding);
    assert_eq!(classify_layer_kind("wte.weight", None, 8), LayerKind::Embedding);
    assert_eq!(classify_layer_kind("position_embedding.weight", None, 8), LayerKind::Embedding);
}

#[test]
fn classify_norm_patterns() {
    assert_eq!(classify_layer_kind("blocks.3.attn_norm.weight", Some(3), 8), LayerKind::Norm);
    assert_eq!(classify_layer_kind("norm.weight", None, 8), LayerKind::Norm);
}

#[test]
fn classify_first_or_last_layer() {
    assert_eq!(classify_layer_kind("blocks.0.attn.wq.weight", Some(0), 8), LayerKind::FirstOrLast);
    assert_eq!(classify_layer_kind("blocks.7.attn.wq.weight", Some(7), 8), LayerKind::FirstOrLast);
    assert_eq!(classify_layer_kind("blocks.4.attn.wq.weight", Some(4), 8), LayerKind::Generic);
}

// --- layer_of ---

#[test]
fn layer_of_recognises_patterns() {
    assert_eq!(layer_of("blocks.6.attn.wq"), Some(6));
    assert_eq!(layer_of("layers.12.norm"), Some(12));
    assert_eq!(layer_of("h.3.mlp.fc"), Some(3));
    assert_eq!(layer_of("embedding.weight"), None);
}
```

- [ ] **Step 2: Run the new tests.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_primitives 2>&1 | tail -10
```

Expected: all tests pass.

### Task 2.2: Commit 2

- [ ] **Step 1: Stage + commit.**

```bash
git add crates/nsl-codegen/tests/cpdt_sensitivity_primitives.rs
git commit -m "$(cat <<'EOF'
test(cpdt): sensitivity-scorer primitive unit tests (Phase 1 Commit 2)

- 23 unit tests for cpdt_sensitivity primitives: position_criticality at
  L ∈ {1, 2, 3, 4, 8, 16} (including the L<4 guard behavior),
  gradient_magnitude_est(None) → CALIB_K, assign_tier boundary behavior
  at T0/T1/T2 ± ε, layer-kind override paths (Embedding/Norm/FirstOrLast
  always Tier::High regardless of score), classify_layer_kind name
  patterns, layer_of pattern recognition.
- ANALYSIS_VERSION pinned to 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 3 — test(cpdt): adversarial + weights-present fixtures (red)

### Task 3.1: Write weights-present regression test (ignored)

**Files:**
- Create: `crates/nsl-codegen/tests/cpdt_sensitivity_snapshot.rs`

- [ ] **Step 1: Write the snapshot test.**

```rust
//! Phase 1 Commit 3 — weights-present regression snapshot.
//!
//! Compares unified scorer output on calib_{tiny, small, medium} against
//! tests/fixtures/cpdt_calibration/expected_weights_present.json.
//!
//! This test is ignored in Commit 3 and unblocked in Commit 4 after
//! gradient_magnitude_est gains the weights-reading branch.

use std::path::Path;
use nsl_codegen::cpdt_sensitivity::{classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality, CALIB_ALPHA};
use nsl_codegen::cpdt_tier_apply::{PrecisionConfig, Tier};
use nsl_codegen::weight_aware::WeightMap;
use serde::Deserialize;

#[derive(Deserialize)]
struct TierEntry { name: String, tier: String }

#[derive(Deserialize)]
struct FixtureSnapshot { fixture: String, tiers: Vec<TierEntry> }

fn load_expected() -> Vec<FixtureSnapshot> {
    let path = Path::new("tests/fixtures/cpdt_calibration/expected_weights_present.json");
    let json = std::fs::read_to_string(path).expect("expected_weights_present.json not found");
    serde_json::from_str(&json).expect("expected_weights_present.json parse error")
}

#[test]
#[ignore = "red: unblocked by Commit 4 (gradient_magnitude_est reads weights)"]
fn weights_present_matches_expected_snapshot() {
    let expected = load_expected();
    for fs in &expected {
        let path = format!("tests/fixtures/cpdt_calibration/{}.safetensors", fs.fixture);
        let wm = WeightMap::load(Path::new(&path)).expect("fixture load failed");
        let n_layers = match fs.fixture.as_str() {
            "calib_tiny" => 2,
            "calib_small" => 8,
            "calib_medium" => 16,
            _ => panic!("unknown fixture: {}", fs.fixture),
        };
        for expected_entry in &fs.tiers {
            let entry = wm.get(&expected_entry.name).expect("tensor missing from fixture");
            let layer = layer_of(&expected_entry.name);
            let kind = classify_layer_kind(&expected_entry.name, layer, n_layers);
            let gm = gradient_magnitude_est(Some(entry));
            let pos = position_criticality(layer, n_layers, CALIB_ALPHA);
            let elts = entry.num_elements.max(1) as f64;
            let score = gm * pos / elts;
            let tier = nsl_codegen::cpdt_sensitivity::assign_tier(score, kind);
            assert_eq!(tier.as_str(), expected_entry.tier,
                "tier mismatch on {}/{}: expected {}, got {}",
                fs.fixture, expected_entry.name, expected_entry.tier, tier.as_str());
        }
    }
}
```

- [ ] **Step 2: Create expected_weights_present.json via local prototype.**

Temporarily patch `gradient_magnitude_est` in `cpdt_sensitivity.rs` to read weights when `Some` is passed (the Commit 4 implementation, applied locally). Run the calibrate binary:

```bash
cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/ > /tmp/cal.txt
# Now rename the emitted baseline_heuristic.json to expected_weights_present.json
# AFTER the patched gradient_magnitude_est was active during the run.
mv tests/fixtures/cpdt_calibration/baseline_heuristic.json tests/fixtures/cpdt_calibration/expected_weights_present.json
# Regenerate baseline_heuristic.json with the PRE-PATCH (None-branch-only) code
git checkout -- crates/nsl-codegen/src/cpdt_sensitivity.rs
cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/
```

Workflow alternative (cleaner): write a single-purpose throwaway binary that implements the Commit 4 scorer and run it to generate `expected_weights_present.json` directly. Either way: the JSON was generated by a local prototype of Commit 4; Commit 3 commits it; Commit 4 must produce byte-identical output.

- [ ] **Step 3: Verify test ignore-state.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_snapshot 2>&1 | tail -5
```

Expected: `test weights_present_matches_expected_snapshot ... ignored`.

### Task 3.2: Write adversarial fixture test (ignored)

**Files:**
- Create: `crates/nsl-codegen/tests/cpdt_sensitivity_adversarial.rs`

- [ ] **Step 1: Write the adversarial test.**

```rust
//! Phase 1 Commit 3 — adversarial correctness gate.
//!
//! Purpose: prove the weight-aware path is doing real work. Catches
//! silent-stub, mis-calibration, cross-layer contamination, clone
//! aliasing, and future kind-override regressions in one fixture.
//!
//! Ignored in Commit 3; unblocked in Commit 4.

use std::path::Path;
use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    CALIB_ALPHA, CALIB_T0, LayerKind,
};
use nsl_codegen::cpdt_tier_apply::Tier;
use nsl_codegen::weight_aware::{WeightDType, WeightMap};

const TARGET_NAME: &str = "blocks.4.ffn.w_down.weight";
const N_LAYERS: u32 = 8;

fn score_layer(wm: &WeightMap, name: &str) -> (Tier, f64, LayerKind) {
    let entry = wm.get(name).expect("tensor missing");
    let layer = layer_of(name);
    let kind = classify_layer_kind(name, layer, N_LAYERS);
    let gm = gradient_magnitude_est(Some(entry));
    let pos = position_criticality(layer, N_LAYERS, CALIB_ALPHA);
    let elts = entry.num_elements.max(1) as f64;
    let score = gm * pos / elts;
    (assign_tier(score, kind), score, kind)
}

fn scale_entry_in_place(wm: &mut WeightMap, name: &str, factor: f64) {
    let entry = wm.get_mut(name).expect("tensor missing");
    let bw = entry.dtype.byte_width();
    let mut new_data = Vec::with_capacity(entry.data.len());
    for i in 0..entry.num_elements {
        let off = i * bw;
        let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
        let scaled = v * factor;
        let mut buf = vec![0u8; bw];
        nsl_codegen::weight_aware::write_f64_as_dtype(scaled, entry.dtype, &mut buf);
        new_data.extend_from_slice(&buf);
    }
    entry.data = new_data;
}

#[test]
#[ignore = "red: unblocked by Commit 4 (gradient_magnitude_est reads weights)"]
fn adversarial_localized_tier_shift() {
    let original = WeightMap::load(Path::new("tests/fixtures/cpdt_calibration/calib_small.safetensors"))
        .expect("calib_small fixture missing");

    // Precondition 1: target layer is not kind-overridden.
    let (_, _, target_kind) = score_layer(&original, TARGET_NAME);
    assert!(!target_kind.is_kind_overridden(),
        "adversarial target must be a scored (not kind-overridden) layer; \
         current target kind = {target_kind:?}. Revisit fixture selection.");

    // Clone + verify clone isolation.
    let mut clone = original.clone();
    let embed_rms_before = {
        let e = original.get("tok_embeddings.weight").unwrap();
        gradient_magnitude_est(Some(e))
    };

    // Compute multiplier from calibrated constants to guarantee Tier::High.
    let (baseline_tier, s_pre, _) = score_layer(&original, TARGET_NAME);
    let m = (CALIB_T0 / s_pre) * 1.5;
    eprintln!("adversarial: baseline tier {baseline_tier:?}, s_pre = {s_pre:.6}, M = {m:.3}");

    // Apply localized mutation.
    scale_entry_in_place(&mut clone, TARGET_NAME, m);

    // Precondition 2: clone isolation held (original embedding unchanged).
    let embed_rms_after = {
        let e = original.get("tok_embeddings.weight").unwrap();
        gradient_magnitude_est(Some(e))
    };
    assert_eq!(embed_rms_before, embed_rms_after,
        "WeightMap::clone aliased entries; adversarial fixture invariants broken.");

    // Strong assertion 1: target lands at Tier::High exactly.
    let (adv_tier, adv_score, _) = score_layer(&clone, TARGET_NAME);
    assert_eq!(adv_tier, Tier::High,
        "target layer did not reach Tier::High under {m:.2}× scaling (got {adv_tier:?}, score {adv_score:.6}). \
         Likely causes: silent-stub gradient_magnitude_est, mis-calibrated T0, broken scorer path.");

    // Strong assertion 2: every other scored layer's tier unchanged.
    for name in original.entries().map(|(n, _)| n.clone()).collect::<Vec<_>>() {
        if name == TARGET_NAME { continue; }
        let (orig_tier, _, _) = score_layer(&original, &name);
        let (clone_tier, _, _) = score_layer(&clone, &name);
        assert_eq!(clone_tier, orig_tier,
            "layer {name} tier changed unexpectedly under localized mutation of {TARGET_NAME}. \
             Scorer may be broadcasting weight effects across the model.");
    }
}
```

Note the use of `write_f64_as_dtype` — this function already exists in `weight_aware.rs` (line 723 per the earlier grep) and handles dtype round-tripping.

- [ ] **Step 2: Verify ignore-state.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_adversarial 2>&1 | tail -5
```

Expected: `test adversarial_localized_tier_shift ... ignored`.

### Task 3.3: Commit 3

- [ ] **Step 1: Stage + commit.**

```bash
git add crates/nsl-codegen/tests/cpdt_sensitivity_snapshot.rs
git add crates/nsl-codegen/tests/cpdt_sensitivity_adversarial.rs
git add tests/fixtures/cpdt_calibration/expected_weights_present.json
git commit -m "$(cat <<'EOF'
test(cpdt): adversarial + weights-present fixtures (red) (Phase 1 Commit 3)

Ignored tests that will flip green in Commit 4 once gradient_magnitude_est
reads weights:

- cpdt_sensitivity_snapshot: weights-present regression test against
  expected_weights_present.json (generated by local prototype of Commit 4).
- cpdt_sensitivity_adversarial: correctness gate. Scales blocks.4.ffn.w_down
  in calib_small by M = (CALIB_T0 / S_pre) × 1.5 to guarantee Tier::High;
  asserts (1) target lands at Tier::High exactly, (2) every other layer's
  tier unchanged, (3) WeightMap::clone isolation preserved, (4) target is
  not kind-overridden (fails with a clear diagnostic if a future refactor
  changes this).

expected_weights_present.json was generated from a local prototype matching
Commit 4's implementation byte-for-byte. If Commit 4 diverges from the
prototype, Commit 4's snapshot test will fail; fix is either updating the
JSON or fixing Commit 4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 4 — feat(cpdt): gradient_magnitude_est reads weights (green)

### Task 4.1: Verify the Some-branch of gradient_magnitude_est

**Files:** none (verification only).

The `gradient_magnitude_est` implementation from Task 1.6 already reads weights when `Some(&WeightEntry)` is passed (lines inside the `Some(w) => ...` arm). Commit 4 only needs to:
1. Remove `#[ignore]` from Commit 3's two tests.
2. Run them; confirm they pass.
3. Add the 95%-weighted-disagreement integration test.

- [ ] **Step 1: Confirm Commit 1's implementation already reads weights.**

```bash
grep -nA 10 "pub fn gradient_magnitude_est" crates/nsl-codegen/src/cpdt_sensitivity.rs
```

Expected: the `Some(w) => { ... sum_sq ... sqrt() }` branch is present.

### Task 4.2: Unblock Commit 3 tests

**Files:**
- Modify: `crates/nsl-codegen/tests/cpdt_sensitivity_snapshot.rs`
- Modify: `crates/nsl-codegen/tests/cpdt_sensitivity_adversarial.rs`

- [ ] **Step 1: Remove `#[ignore = ...]` from snapshot test.**

In `cpdt_sensitivity_snapshot.rs`, delete the `#[ignore = "red: unblocked by Commit 4 (gradient_magnitude_est reads weights)"]` line above `fn weights_present_matches_expected_snapshot`.

- [ ] **Step 2: Remove `#[ignore = ...]` from adversarial test.**

In `cpdt_sensitivity_adversarial.rs`, delete the corresponding `#[ignore]` line.

- [ ] **Step 3: Run both tests.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_snapshot --test cpdt_sensitivity_adversarial 2>&1 | tail -15
```

Expected: both pass. If `weights_present_matches_expected_snapshot` fails, the Commit 3 prototype diverged from the current scorer — fix the scorer or regenerate `expected_weights_present.json` (with an explicit commit note).

### Task 4.3: Add weighted disagreement integration test

**Files:**
- Create: `crates/nsl-codegen/tests/cpdt_sensitivity_disagreement.rs`

- [ ] **Step 1: Write the test.**

```rust
//! Phase 1 Commit 4 — parameter-weighted disagreement between no-weights
//! and weights-present paths on the baseline corpus. Gate: < 5%.

use std::path::Path;
use nsl_codegen::cpdt_sensitivity::{
    assign_tier, classify_layer_kind, gradient_magnitude_est, layer_of, position_criticality,
    CALIB_ALPHA,
};
use nsl_codegen::cpdt_tier_apply::Tier;
use nsl_codegen::weight_aware::WeightMap;

fn score_with_entry(name: &str, entry_opt: Option<&nsl_codegen::weight_aware::WeightEntry>, numel: usize, n_layers: u32) -> Tier {
    let layer = layer_of(name);
    let kind = classify_layer_kind(name, layer, n_layers);
    let gm = gradient_magnitude_est(entry_opt);
    let pos = position_criticality(layer, n_layers, CALIB_ALPHA);
    let elts = numel.max(1) as f64;
    let score = gm * pos / elts;
    assign_tier(score, kind)
}

#[test]
fn weighted_disagreement_below_5_percent() {
    let fixtures = [("calib_tiny", 2), ("calib_small", 8)]; // calib_medium regen lives in target/
    let mut disagreeing_params: u64 = 0;
    let mut total_params: u64 = 0;
    for (name, n_layers) in fixtures {
        let path = format!("tests/fixtures/cpdt_calibration/{name}.safetensors");
        let wm = WeightMap::load(Path::new(&path)).expect("fixture load failed");
        for (tname, entry) in wm.entries() {
            let with = score_with_entry(tname, Some(entry), entry.num_elements, n_layers);
            let without = score_with_entry(tname, None, entry.num_elements, n_layers);
            total_params += entry.num_elements as u64;
            if with != without {
                disagreeing_params += entry.num_elements as u64;
            }
        }
    }
    let frac = disagreeing_params as f64 / total_params as f64;
    assert!(frac < 0.05, "weighted disagreement {frac:.4} >= 0.05 on baseline corpus");
    eprintln!("weighted disagreement: {:.4} ({}/{} params)", frac, disagreeing_params, total_params);
}
```

- [ ] **Step 2: Run the test.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_sensitivity_disagreement 2>&1 | tail -5
```

Expected: passes; disagreement well below 0.05.

### Task 4.4: Commit 4

- [ ] **Step 1: Stage + commit.**

```bash
git add crates/nsl-codegen/tests/cpdt_sensitivity_snapshot.rs
git add crates/nsl-codegen/tests/cpdt_sensitivity_adversarial.rs
git add crates/nsl-codegen/tests/cpdt_sensitivity_disagreement.rs
git commit -m "$(cat <<'EOF'
feat(cpdt): gradient_magnitude_est reads weights (green) (Phase 1 Commit 4)

- Remove #[ignore] from Commit 3's two tests; both now pass:
  * cpdt_sensitivity_snapshot: weights-present tiers match expected JSON.
  * cpdt_sensitivity_adversarial: localized 10×-ish scaling of one FFN
    matrix pushes that layer to Tier::High exactly; all other layers'
    tiers unchanged; WeightMap::clone isolation preserved.
- Add weighted-disagreement regression gate (cpdt_sensitivity_disagreement):
  < 5% parameter-weighted disagreement between no-weights and weights-
  present paths on the calibration corpus.

Commit 1's byte-identity regression on the weights-present path remains
intact. The no-weights path is now quality-gated.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 5 — feat(cpdt): --weights / AST auto-detect integration + validation + diagnostics

**Largest commit. Acceptance has five independently-verifiable sub-conditions (see Task 5.7).**

### Task 5.1: Wire AST `load_safetensors(...)` detection

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` (build path, near line 850-880)

- [ ] **Step 1: Add AST walk for `load_safetensors(...)`.**

In `nsl-cli/src/main.rs`, inside the `build` subcommand handler (around line 600-800 where the parser produces `tr_parse.module`), add a pass that walks the module AST for calls to `load_safetensors` and collects the first string-literal argument. Extract this into a helper `fn find_ast_weight_ref(module: &Module) -> Option<PathBuf>`.

Walk pattern: iterate module-scope `Let` bindings and model constructor bodies; for each `ExprKind::Call { callee, args }` where the callee resolves to the name `"load_safetensors"`, if `args[0]` is `ExprKind::StringLiteral(s)`, return `Some(PathBuf::from(s))`.

(Exact AST node names: check `crates/nsl-ast/src/lib.rs` — the NSL AST types. The helper body is straightforward pattern-matching; use `walk_exprs` utilities if they already exist.)

- [ ] **Step 2: Implement the four-case decision table.**

Still in `main.rs`, after the AST walk and before the existing CPDT invocation:

```rust
let ast_weight_ref = find_ast_weight_ref(&tr_parse.module);
let cpdt_enabled = cpdt_mode != CpdtMode::Off;
let weights_path: Option<PathBuf> = match (ast_weight_ref.as_ref(), weights.as_ref()) {
    (Some(ast), Some(flag)) => {
        eprintln!("warning: --weights {} override in effect; AST-referenced {} ignored.",
            flag.display(), ast.display());
        Some(flag.clone())
    }
    (Some(ast), None) => Some(ast.clone()),
    (None, Some(flag)) => Some(flag.clone()),
    (None, None) if cpdt_enabled && cpdt_weight_aware => {
        eprintln!(r#"error: CPDT weight-aware tier assignment requires weights. Either:
  1. Add `m.load_safetensors("path/to/weights.safetensors")` to your NSL source
  2. Pass `--weights <path>` to nsl build
  3. Disable weight-aware CPDT with `@cpdt(weight_aware=false)` to fall back
     to heuristic-only tier assignment (produces same tiers as pre-weight-aware
     behavior)"#);
        process::exit(1);
    }
    (None, None) => None,
};
```

(`cpdt_weight_aware` comes from the `@cpdt(weight_aware=...)` decorator parsed by nsl-semantic; wire it through the CpdtInput.)

- [ ] **Step 3: Load WeightMap with file-not-found handling.**

```rust
let weight_map: Option<WeightMap> = if let Some(ref p) = weights_path {
    match WeightMap::load(p) {
        Ok(w) => Some(w),
        Err(e) => {
            eprintln!("error: --weights <path> could not be read.");
            eprintln!("  path: {}", p.display());
            eprintln!("  cause: {:?}", e);
            process::exit(1);
        }
    }
} else { None };
```

### Task 5.2: Implement `WeightMap`-vs-`AppliedPlan` validation

**Files:**
- Modify: `crates/nsl-codegen/src/cpdt_sensitivity.rs::validate` (stub body from Task 1.6)

- [ ] **Step 1: Replace the `validate` stub body with real cross-checking.**

Inspect `AppliedLayer` for weight-name / shape / dtype accessors. In `cpdt_sensitivity.rs`:

```rust
pub fn validate(wm: &WeightMap, applied: &AppliedPlan) -> Result<(), ValidationError> {
    for layer in applied.layers() {
        for (tensor_name, expected_shape, expected_dtype) in layer.weight_metadata() {
            // Exact field names on AppliedLayer verified during Commit 5 prep.
            let Some(entry) = wm.get(&tensor_name) else {
                return Err(ValidationError::MissingTensor { tensor_name });
            };
            if entry.shape != expected_shape {
                return Err(ValidationError::ShapeMismatch {
                    tensor_name,
                    expected: expected_shape,
                    actual: entry.shape.clone(),
                });
            }
            let actual_dtype = format!("{:?}", entry.dtype);
            if actual_dtype != expected_dtype {
                return Err(ValidationError::DtypeMismatch {
                    tensor_name, expected: expected_dtype, actual: actual_dtype
                });
            }
        }
    }
    Ok(())
}
```

If `AppliedLayer` doesn't expose a `weight_metadata()` iterator, add one. Check `wggo_apply.rs:50-70` for the existing `AppliedLayer` surface; extend as needed.

- [ ] **Step 2: Wire the validation into `cpdt::run`.**

Open `crates/nsl-codegen/src/cpdt.rs` (or wherever `cpdt::run` lives — grep for it). At the top of `run`, after `input.weights` is available, add:

```rust
if let Some(ref wm) = input.weights {
    if let Err(e) = cpdt_sensitivity::validate(wm, &input.applied) {
        eprintln!("{}", e);
        return Err(/* appropriate error type */);
    }
}
```

### Task 5.3: Add CPDT_CALIB_K + weights warning

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` (near the WeightMap load block)

- [ ] **Step 1: Emit the expanded warning when both conditions hit.**

```rust
if weight_map.is_some() {
    if let Ok(val) = std::env::var("CPDT_CALIB_K") {
        eprintln!("warning: CPDT_CALIB_K={val} is set but ignored.");
        eprintln!("         Weights are present, so computed gradient_magnitude_est is authoritative.");
        eprintln!("         If you intended CPDT_CALIB_K to apply, unset --weights or remove the");
        eprintln!("         AST load_safetensors reference. If the env var is vestigial, you can");
        eprintln!("         ignore this warning or unset it.");
    }
}
```

### Task 5.4: Passive tier-agreement diagnostic

**Files:**
- Modify: `crates/nsl-codegen/src/cpdt.rs` (or wherever cpdt::run is)

- [ ] **Step 1: After tier assignment completes with weights present, compute and emit agreement.**

After the scorer produces `PrecisionPlan` on the weights-present path, re-run it on the no-weights path (same `AppliedPlan`, `None` for `WeightEntry`), compare tiers, compute parameter-weighted agreement, and emit:

```rust
if input.weights.is_some() {
    let plan_noweight = plan_map_noweights(&input.applied, &cfg);
    let (agree_layers, total_layers, agree_params, total_params) =
        compute_agreement(&plan, &plan_noweight);
    let layer_pct = 100.0 * agree_layers as f64 / total_layers.max(1) as f64;
    let param_pct = 100.0 * agree_params as f64 / total_params.max(1) as f64;
    eprintln!("[cpdt] weight-aware tier agreement: {:.2}% ({}/{} layers, parameter-weighted {:.2}%)",
        layer_pct, total_layers - (total_layers - agree_layers), total_layers, param_pct);
    if param_pct < 95.0 {
        eprintln!("warning: weight-aware tier agreement below 95% on this model (parameter-weighted {:.2}%).", param_pct);
        eprintln!("         This may indicate that the calibration constants do not fit this");
        eprintln!("         weight distribution well. Consider filing an issue referencing");
        eprintln!("         the post-scoring-normalization deferred work; include the model's");
        eprintln!("         weight file hash and the full diagnostic output.");
    }
}
```

`plan_map_noweights` is a new helper in `cpdt_tier_apply.rs` that takes `&AppliedPlan` and `&PrecisionConfig`, iterates layers, and calls the scorer with `None` for each entry.

### Task 5.5: CLI tests for all five sub-conditions

**Files:**
- Create: `crates/nsl-cli/tests/cpdt_weights_cli.rs`

- [ ] **Step 1: Write the CLI test suite.**

Five tests, one per sub-condition:

```rust
// Test 1: AST auto-detect (Case 1).
#[test]
fn ast_auto_detect_loads_weights() {
    let tmp = tempfile::tempdir().unwrap();
    let wpath = generate_tiny_weights(tmp.path());
    let src = format!(r#"let w = load_safetensors("{}")\n@cpdt(num_gpus=2)\ntrain(...): ..."#, wpath.display());
    let nslpath = tmp.path().join("model.nsl");
    std::fs::write(&nslpath, src).unwrap();
    let out = run_nsl_build(&nslpath, &[]);
    assert!(out.stderr.contains("[cpdt] weight-aware tier agreement"),
        "auto-detect should run weight-aware path; stderr: {}", out.stderr);
}

// Test 2: --weights flag override (Case 2).
#[test]
fn flag_overrides_ast_reference_with_warning() {
    let tmp = tempfile::tempdir().unwrap();
    let ast_wpath = generate_tiny_weights(tmp.path());  // referenced by AST
    let flag_wpath = generate_tiny_weights(tmp.path()); // passed via flag
    let src = format!(r#"let w = load_safetensors("{}")\n@cpdt(num_gpus=2)\ntrain(...): ..."#, ast_wpath.display());
    let nslpath = tmp.path().join("model.nsl");
    std::fs::write(&nslpath, src).unwrap();
    let out = run_nsl_build(&nslpath, &["--weights", flag_wpath.to_str().unwrap()]);
    assert!(out.stderr.contains("warning: --weights"),
        "flag override should emit warning; stderr: {}", out.stderr);
    assert!(out.stderr.contains("override in effect"));
}

// Test 3: absent-both error with three-option resolution (Case 4).
#[test]
fn absent_both_error_with_resolution_message() {
    let tmp = tempfile::tempdir().unwrap();
    let src = r#"@cpdt(num_gpus=2)\ntrain(...): ..."#;
    let nslpath = tmp.path().join("model.nsl");
    std::fs::write(&nslpath, src).unwrap();
    let out = run_nsl_build(&nslpath, &[]);
    assert!(!out.status.success());
    assert!(out.stderr.contains("CPDT weight-aware tier assignment requires weights"));
    assert!(out.stderr.contains("1. Add"));
    assert!(out.stderr.contains("2. Pass"));
    assert!(out.stderr.contains("3. Disable"));
}

// Test 4: CPDT_CALIB_K + weights warning.
#[test]
fn cpdt_calib_k_ignored_with_warning_when_weights_present() {
    let tmp = tempfile::tempdir().unwrap();
    let wpath = generate_tiny_weights(tmp.path());
    let nslpath = tmp.path().join("model.nsl");
    std::fs::write(&nslpath, "@cpdt(num_gpus=2)\ntrain(...): ...").unwrap();
    let out = run_nsl_build_with_env(&nslpath, &["--weights", wpath.to_str().unwrap()],
                                     &[("CPDT_CALIB_K", "0.05")]);
    assert!(out.stderr.contains("CPDT_CALIB_K=0.05 is set but ignored"));
    assert!(out.stderr.contains("vestigial"));
}

// Test 5: --weights <nonexistent> clear error.
#[test]
fn weights_nonexistent_file_clear_error() {
    let tmp = tempfile::tempdir().unwrap();
    let nslpath = tmp.path().join("model.nsl");
    std::fs::write(&nslpath, "@cpdt(num_gpus=2)\ntrain(...): ...").unwrap();
    let out = run_nsl_build(&nslpath, &["--weights", "/nonexistent/path/weights.safetensors"]);
    assert!(!out.status.success());
    assert!(out.stderr.contains("--weights <path> could not be read"));
    assert!(out.stderr.contains("/nonexistent/path/weights.safetensors"));
}

// Helpers — generate_tiny_weights, run_nsl_build, run_nsl_build_with_env — follow
// the existing pattern in crates/nsl-cli/tests/wrga_report_cli.rs.
```

Use the existing test utility pattern from `crates/nsl-cli/tests/wrga_report_cli.rs` (it already handles tempfile-driven CLI testing with `create_small_safetensors` and similar helpers). Reuse `create_small_safetensors` — don't duplicate.

- [ ] **Step 2: Run the CLI tests.**

```bash
cargo test -p nsl-cli --test cpdt_weights_cli 2>&1 | tail -15
```

Expected: all 5 tests pass.

### Task 5.6: Add `WeightMap` validation integration test

**Files:**
- Create: `crates/nsl-codegen/tests/cpdt_validation_integration.rs`

- [ ] **Step 1: Write the test — three cases.**

```rust
//! Phase 1 Commit 5 — WeightMap-vs-AppliedPlan validation integration.

// Test 1: shape mismatch → fast error.
// Test 2: dtype mismatch → fast error.
// Test 3: missing tensor → fast error.

// Each test builds a small AppliedPlan with known layer declarations, loads
// a WeightMap that deliberately violates one property, calls validate(), and
// asserts the returned ValidationError variant + the error message format.
```

(Full test body omitted here; follow the pattern established by the fixture-generation helpers. Three test functions, each ~20 lines.)

- [ ] **Step 2: Run.**

```bash
cargo test -p nsl-codegen --features cuda --test cpdt_validation_integration 2>&1 | tail -8
```

### Task 5.7: Commit 5

**Acceptance has five independently-verifiable sub-conditions:**

1. **CLI decision table:** `cargo test -p nsl-cli --test cpdt_weights_cli` — all 5 tests pass.
2. **WeightMap validation:** `cargo test -p nsl-codegen --features cuda --test cpdt_validation_integration` — all 3 tests pass.
3. **CPDT_CALIB_K + weights warning:** covered by Test 4 in `cpdt_weights_cli`.
4. **Tier-agreement diagnostic:** covered by Test 1 in `cpdt_weights_cli` (stderr contains diagnostic line).
5. **File-not-found ergonomics:** covered by Test 5 in `cpdt_weights_cli`.

- [ ] **Step 1: Stage + commit.**

```bash
git add crates/nsl-cli/src/main.rs
git add crates/nsl-codegen/src/cpdt_sensitivity.rs
git add crates/nsl-codegen/src/cpdt_tier_apply.rs  # if plan_map_noweights added
git add crates/nsl-codegen/src/cpdt.rs             # if run modified
git add crates/nsl-cli/tests/cpdt_weights_cli.rs
git add crates/nsl-codegen/tests/cpdt_validation_integration.rs
git commit -m "$(cat <<'EOF'
feat(cpdt): --weights auto-detect + validation + diagnostics (Phase 1 C5)

Largest commit in the Phase 1 sequence. Five independently-verifiable
sub-conditions:

1. CLI decision table (four cases):
   - AST `load_safetensors("...")` detected and used when --weights absent.
   - --weights flag overrides AST ref with stderr warning.
   - --weights without AST ref: use flag.
   - Both absent + weight-aware CPDT enabled: error with three-option
     resolution message.

2. WeightMap-vs-AppliedPlan validation pass inside cpdt::run:
   shape / dtype / missing-tensor mismatches produce fast errors with
   the offending tensor name and expected-vs-actual values.

3. CPDT_CALIB_K + weights warning: noisy-ignoring with expanded
   resolution-hint text (unset --weights, remove AST ref, or ignore).

4. Passive `[cpdt] weight-aware tier agreement: X%` diagnostic emitted
   on every weights-present CPDT build. When parameter-weighted
   agreement < 95%, a follow-up warning points at the
   post-scoring-normalization deferred work.

5. --weights <nonexistent> produces a clear error with the attempted
   path shown, not the vague WeightMap::load internal error.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Commit 6 — docs(cpdt): Phase 1 close-out + Phase 2 stub + invariants

### Task 6.1: Write Phase 2 stub doc

**Files:**
- Create: `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md`

- [ ] **Step 1: Write the stub.**

Copy the content from §11 of the Phase 1 spec into the stub doc, expanded slightly and with a clear "Stub — not a design" header. Include: measurement trigger, scope (spectral + cache + CI check), inherited discipline, pre-decided architectural direction (with ≈1.0 verification step), open questions.

- [ ] **Step 2: Stage.**

```bash
git add docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md
```

### Task 6.2: Write Phase 1 invariants memory file

**Files:**
- Create: `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_cpdt_weight_aware_invariants.md`

- [ ] **Step 1: Write the invariants file.**

```markdown
---
name: CPDT weight-aware Phase 1 invariants
description: Load-bearing constraints for CPDT's sensitivity scorer; must survive Phase 2 and beyond
type: project
---

# CPDT Weight-Aware Phase 1 — Invariants

Nine load-bearing constraints established by Phase 1. Future edits MUST preserve them or bump `ANALYSIS_VERSION` and update this file.

1. **ANALYSIS_VERSION bumps on every formula / factor / tier-boundary change.**
   - Why: Phase 2 sidecar cache keys include this version; changing the formula without bumping produces silently-stale cache reads.
   - How to apply: any diff touching `fn compute_score`, `fn assign_tier`, `fn gradient_magnitude_est`, or `fn position_criticality` MUST bump `ANALYSIS_VERSION` in the same commit.

2. **`cpdt_joint::solve` reads only tier-derived aggregates, never per-parameter fields.**
   - Why: calibration byte-identity contract assumes joint-solver input is a pure function of tier labels (via `total_optim_bytes`).
   - How to apply: if `solve` starts reading per-parameter data, tighten the calibration contract and update this invariant.

3. **Layer-kind override (Embedding, Norm, FirstOrLast) takes precedence over formula score.**
   - Why: research-prescriptive categorical rule; the formula alone can't cleanly place large embeddings at Tier::High because of the `element_count` divisor.
   - How to apply: `assign_tier(score, kind)` checks `kind.is_kind_overridden()` FIRST; score-based branches are only reached for `LayerKind::Generic`.

4. **`SensitivityScorer` is a concrete type, not a trait.**
   - Why: pluggable scorer implementations produce the same drift-multiplication failure mode as the i-option in Phase 1 brainstorming.
   - How to apply: Phase 2 spectral is a new *factor field* inside `SensitivityScorer`, not a new `Scorer` implementation.

5. **Rank normalization is deferred; factors output physical units.**
   - Why: the multiplicative structure of the sensitivity formula requires each factor's range to compose; pre-constraining one factor breaks composability.
   - How to apply: never clamp or normalize a factor output inside the scorer. Any normalization is a separate opt-in post-processing step (`@cpdt(score_normalization=...)`).

6. **`--weights` flag overrides AST format inference entirely when both are present.**
   - Why: the flag's entire loading decision is authoritative; partial overrides create ambiguity.
   - How to apply: future multi-loader-format work treats the `--weights` path extension as the single source for loader selection.

7. **No-weights path agrees with weights-present path within 5% parameter-weighted on baseline corpus.**
   - Why: calibration's purpose is to make the no-weights path a reasonable approximation of the weights-present path on typical models.
   - How to apply: `cpdt_sensitivity_disagreement` test asserts `< 0.05`; breaking this requires recalibrating `CALIB_K`.

8. **Weights-present path matches pre-refactor heuristic byte-identical on baseline corpus.**
   - Why: the Phase 1 refactor must be invisible to existing weights-present users.
   - How to apply: `cpdt_sensitivity_snapshot` test asserts byte-identity against `expected_weights_present.json`.

9. **Scorer state derives exclusively from `WeightMap` read-only access.**
   - Why: sharing state with other weight-consuming passes (sparsity analysis, dead-weight elimination, constant folder) creates cross-concern bleed.
   - How to apply: if the scorer needs a weight-access pattern `WeightMap` doesn't expose, add a read-only accessor to `WeightMap` rather than duplicating state in `cpdt_sensitivity.rs`.
```

- [ ] **Step 2: Add pointer to `MEMORY.md`.**

Open `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md`. Add a new section (under "CPDT follow-ups" or similar):

```markdown
## CPDT weight-aware — Phase 1 invariants (2026-04-18)
- [CPDT weight-aware Phase 1 invariants](project_cpdt_weight_aware_invariants.md) — nine load-bearing constraints; must survive Phase 2. Includes ANALYSIS_VERSION bump rule, cpdt_joint reads-only-aggregates contract, layer-kind override precedence, and unified-scorer-as-concrete-type design.
```

Verify MEMORY.md stays ≤ 200 lines:

```bash
wc -l C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md
```

If over 200, extract an older topic file per the usual pattern.

### Task 6.3: Retire `cpdt_precision.rs` references

**Files:** various docs

- [ ] **Step 1: Grep for stale references.**

```bash
grep -rnE "cpdt_precision" docs/ crates/ *.md 2>/dev/null | grep -v "Cargo.lock\|target/"
```

- [ ] **Step 2: Update each match** — either delete (if in a changelog or similar that's now obsolete) or replace with `cpdt_sensitivity` / `cpdt_tier_apply` as appropriate.

### Task 6.4: Commit 6

- [ ] **Step 1: Stage + commit.**

```bash
git add docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md
git add C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_cpdt_weight_aware_invariants.md
git add C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/MEMORY.md
git add <any other docs touched>
git commit -m "$(cat <<'EOF'
docs(cpdt): Phase 1 close-out + Phase 2 stub + invariants (Phase 1 C6)

- Phase 2 stub at docs/superpowers/specs/...-phase2-stub.md: measurement
  trigger (< 95% parameter-weighted agreement on shipped models, or
  user workflow), scope (spectral + cache + CI check), inherited
  discipline, pre-decided architectural direction with ≈1.0 verification.
- project_cpdt_weight_aware_invariants.md: nine Phase 1 invariants that
  must survive Phase 2 (ANALYSIS_VERSION bump rule, joint-solver reads-
  only-aggregates contract, layer-kind override precedence, concrete-
  not-trait scorer, no rank normalization, --weights format override,
  95% no-weights-vs-weights agreement, weights-present byte-identity,
  scorer state derives only from WeightMap).
- MEMORY.md pointer added.
- Retired cpdt_precision.rs references in existing docs.

Phase 1 ships complete. Phase 2 is scheduled work, not active.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Post-Commit: Final Verification

- [ ] **Run the full test suite.**

```bash
cargo test -p nsl-codegen --features cuda 2>&1 | tail -20
cargo test -p nsl-cli 2>&1 | tail -10
```

Expected: every test in the §9 test matrix passes; no regressions in existing tests.

- [ ] **Run the full build.**

```bash
cargo build --release --bin nsl --features cuda 2>&1 | tail -3
```

Expected: builds green.

- [ ] **Confirm close-out criteria (§10).**

Walk through the spec's §10 checklist; verify every item. If any fail, do not open a PR — fix first.

- [ ] **Open PR.**

```bash
git push -u origin feat/cpdt-weight-aware-phase1
gh pr create --title "feat(cpdt): Phase 1 weight-aware refinement" --body "$(cat <<'EOF'
## Summary
- Unified `SensitivityScorer` replaces `cpdt_precision.rs` heuristic.
- `--weights` + AST auto-detect four-case decision table.
- Plan-time `WeightMap`-vs-`AppliedPlan` validation.
- `CPDT_CALIB_K` noisy-ignoring + passive tier-agreement diagnostic.
- Phase 2 stub + nine Phase 1 invariants committed.

## Test plan
- [ ] `cargo test -p nsl-codegen --features cuda` green
- [ ] `cargo test -p nsl-cli` green
- [ ] `cargo run --features calibrate --bin cpdt_calibrate -- tests/fixtures/cpdt_calibration/` reproduces committed constants
- [ ] `nsl build --cpdt --cpdt-num-gpus 4 --weights <path> fixture.nsl` emits tier-agreement diagnostic

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review Notes

- **Spec coverage:** every requirement in §§1-10 of the Phase 1 spec is addressed by a task above. §11 Phase 2 stub is created in Task 6.1. §12 open questions (joint-solver determinism, stdlib init snapshot) are resolved in Tasks 1.3 and 1.11.
- **No placeholders:** calibration constants in `cpdt_sensitivity.rs` are marked `/* tuned */` *only* before Task 1.11 runs the calibration binary; after Task 1.11, they're concrete numbers derived from `baseline_heuristic.json`.
- **Type consistency:** `Tier` (was `SensitivityTier`) is consistently renamed across `cpdt_sensitivity.rs`, `cpdt_tier_apply.rs`, all tests, and the calibrate binary. `LayerKind` is new and used consistently. `ANALYSIS_VERSION` is the single source of truth.
- **Known rough edges:** Task 5.2's `AppliedLayer::weight_metadata()` iterator may not exist yet; Task 5.2 Step 1 notes the prep check. Task 3.2's `expected_weights_present.json` generation has a clean workflow (temporary prototype, regenerate, restore) but requires execution-time care.
