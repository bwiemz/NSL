# CPDT Pipeline Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `cpdt::run` into `compile_quant_block` end-to-end so WGGO's `min_shard_factor` recommendation reaches CPDT's planner. Pure planner integration — no IR mutation.

**Architecture:** Mirror WRGA Milestone B.1's "invoke + capture + report" shape. New CLI flags (`--cpdt`, `--cpdt-num-gpus`, `--cpdt-intra-bw`, `--cpdt-inter-bw`, `--cpdt-report`) feed `CompileOptions`; `Compiler` gains `cpdt_mode`/`cluster`/`cpdt_plan` fields; `compile_quant_block` calls a new `invoke_cpdt_if_enabled` after WGGO/CSHA/WRGA; CLI post-compile renders stderr override diagnostics (always) and a `--cpdt-report` stdout summary (on demand).

**Tech Stack:** Rust, existing `cpdt::*` modules, `clap`-style CLI in `crates/nsl-cli/src/main.rs`.

**Spec:** [docs/superpowers/specs/2026-04-15-cpdt-pipeline-integration-design.md](../specs/2026-04-15-cpdt-pipeline-integration-design.md)

**Branch:** `feat/cpdt-pipeline-integration` (fresh from synced `main`).

---

## File Inventory

**Create:**
- `crates/nsl-cli/tests/cpdt_cli.rs` — 3 CLI flag tests
- `crates/nsl-codegen/tests/cpdt_pipeline_integration.rs` — 2 end-to-end tests

**Modify:**
- `crates/nsl-codegen/src/wggo_apply.rs` — extend `AppliedLayer` with `param_bytes` + `activation_bytes` (Task 0, only if discovery confirms)
- `crates/nsl-codegen/src/cpdt_zero.rs` — add `ModelSize::from_applied_plan` constructor
- `crates/nsl-codegen/src/compiler/mod.rs` — add `cpdt_mode`, `cpdt_cluster`, `cpdt_plan`, `cpdt_report_requested` fields; initialize defaults
- `crates/nsl-codegen/src/stmt.rs` — add `invoke_cpdt_if_enabled` helper, call after WRGA invocation in `compile_quant_block`; add AdamW-derivation helper
- `crates/nsl-cli/src/main.rs` — add 5 CLI flags, plumb to `CompileOptions`/`Compiler`, render stderr diagnostics and stdout report after compilation
- `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wggo_consumers.md` — mark Consumer 5 fully shipped

---

## Task 0: Discovery — verify ModelSize derivability

**Files:** Read-only. No edits unless discovery says so.

- [ ] **Step 1: Read `AppliedLayer` and `LayerIlpSolution` to inventory available fields**

Run:
```bash
grep -n "pub struct AppliedLayer\|pub struct LayerIlpSolution\|pub struct LayerIlpInput" crates/nsl-codegen/src/wggo_apply.rs crates/nsl-codegen/src/wggo_ilp.rs
```

- [ ] **Step 2: Determine source of param_bytes and activation_bytes**

Required by `ModelSize`: `per_layer_param_bytes: Vec<u64>`, `per_layer_activation_bytes: Vec<u64>`, `optim_state_multiplier: f64`, `per_layer_compute_us: Vec<f64>`.

`AppliedLayer.estimated_us` already exists (compute proxy). For param/activation bytes, search for the upstream source:
```bash
grep -rn "param_bytes\|activation_bytes\|per_layer_param" crates/nsl-codegen/src/wggo*.rs
```

- [ ] **Step 3: Decide path forward**

Two valid outcomes:
- **(A)** Upstream WGGO inputs already carry the bytes per layer → propagate them through `apply()` into `AppliedLayer`. Task 0a (below) becomes required.
- **(B)** Bytes are NOT computed upstream → derive a conservative estimate from `LayerIlpSolution` shape data (e.g. `ffn_width × hidden_size × dtype_bytes`). Task 0a uses the estimator instead.

Document the decision inline in this checklist as a comment before proceeding.

- [ ] **Step 4: Commit decision note**

If any code is touched (e.g. exploratory grep result captured in a comment), commit. Otherwise skip — Task 0 may produce no commit.

---

## Task 0a: Extend `AppliedLayer` with param/activation bytes

**Files:**
- Modify: `crates/nsl-codegen/src/wggo_apply.rs:21-42` (struct), `wggo_apply.rs:68+` (`apply()` body)
- Test: `crates/nsl-codegen/src/wggo_apply.rs` (existing test module)

- [ ] **Step 1: Write the failing test**

Add to the existing test module in `wggo_apply.rs`:

```rust
#[test]
fn applied_layer_carries_param_and_activation_bytes() {
    let inter = crate::wggo_dp::InterLayerPlan {
        layers: vec![crate::wggo_dp::Layer {
            decision: crate::wggo_dp::LayerDecision::Train,
            param_bytes: 6_000_000,
            activation_bytes: 2_000_000,
            ..Default::default()
        }],
        peak_memory_bytes: 0,
    };
    let ilp = vec![crate::wggo_ilp::LayerIlpSolution::default()];
    let plan = apply(&inter, &ilp);
    assert_eq!(plan.layers[0].param_bytes, 6_000_000);
    assert_eq!(plan.layers[0].activation_bytes, 2_000_000);
}
```

(If discovery in Task 0 selected path **B**, replace `param_bytes`/`activation_bytes` source with the chosen derivation. Adjust this test to assert the derived values.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-codegen wggo_apply::applied_layer_carries_param_and_activation_bytes`
Expected: FAIL with "no field `param_bytes` on type `AppliedLayer`".

- [ ] **Step 3: Add the fields**

In `wggo_apply.rs:21-42` add to `AppliedLayer`:

```rust
/// Parameter bytes for this layer (sum of all weights).
pub param_bytes: u64,
/// Peak activation bytes during forward pass for this layer.
pub activation_bytes: u64,
```

In `apply()` (after the existing field assignments):

```rust
param_bytes: coarse.param_bytes,            // path A
activation_bytes: coarse.activation_bytes,  // path A
// OR for path B:
// param_bytes: derive_param_bytes(ilp_sol),
// activation_bytes: derive_activation_bytes(ilp_sol),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-codegen wggo_apply`
Expected: all `wggo_apply` tests PASS.

- [ ] **Step 5: Run full crate test to detect regressions**

Run: `cargo test -p nsl-codegen --lib`
Expected: PASS (no other tests construct `AppliedLayer` literals; if any do, fix them with the new fields set to `0`).

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/wggo_apply.rs
git commit -m "feat(wggo): expose param_bytes + activation_bytes on AppliedLayer

Required by upcoming CPDT pipeline integration: ModelSize::from_applied_plan
needs per-layer parameter and activation byte counts to populate CpdtInput.model.
Path <A|B> per discovery task."
```

---

## Task 1: `ModelSize::from_applied_plan` constructor

**Files:**
- Modify: `crates/nsl-codegen/src/cpdt_zero.rs` (existing `impl ModelSize` block)
- Test: same file, existing test module

- [ ] **Step 1: Write the failing test**

Add to `cpdt_zero.rs` test module:

```rust
#[test]
fn model_size_from_applied_plan_sums_per_layer_bytes() {
    use crate::wggo_apply::{AppliedLayer, AppliedPlan};
    use crate::wggo_dp::LayerDecision as CoarseDecision;

    let plan = AppliedPlan {
        layers: vec![
            AppliedLayer {
                layer_index: 0,
                layer_name: "blocks.0".into(),
                coarse: CoarseDecision::Train,
                pipeline_stage: 0,
                shard_factor: 1,
                active_heads: 8,
                ffn_width: 4096,
                csha_level: 0,
                adapter_rank: 0,
                optim_m_bits: 32,
                optim_v_bits: 32,
                fase_fused: false,
                packing_mode: 0,
                estimated_us: 12.5,
                param_bytes: 6_000_000,
                activation_bytes: 2_000_000,
            },
            AppliedLayer {
                layer_index: 1,
                layer_name: "blocks.1".into(),
                coarse: CoarseDecision::Train,
                pipeline_stage: 0,
                shard_factor: 1,
                active_heads: 8,
                ffn_width: 4096,
                csha_level: 0,
                adapter_rank: 0,
                optim_m_bits: 32,
                optim_v_bits: 32,
                fase_fused: false,
                packing_mode: 0,
                estimated_us: 14.0,
                param_bytes: 8_000_000,
                activation_bytes: 3_000_000,
            },
        ],
        total_us: 26.5,
        peak_memory_bytes: 0,
    };

    let ms = ModelSize::from_applied_plan(&plan);
    assert_eq!(ms.per_layer_param_bytes, vec![6_000_000, 8_000_000]);
    assert_eq!(ms.per_layer_activation_bytes, vec![2_000_000, 3_000_000]);
    assert_eq!(ms.per_layer_compute_us, vec![12.5, 14.0]);
    assert_eq!(ms.optim_state_multiplier, 8.0); // AdamW default: m+v at fp32 = 8x param fp32
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-codegen cpdt_zero::model_size_from_applied_plan_sums_per_layer_bytes`
Expected: FAIL with "no associated function `from_applied_plan` on `ModelSize`".

- [ ] **Step 3: Add the constructor**

In `cpdt_zero.rs` `impl ModelSize` block:

```rust
/// Build a `ModelSize` from an `AppliedPlan`. Sums per-layer bytes
/// directly from `AppliedLayer.param_bytes` / `activation_bytes` /
/// `estimated_us`. Defaults `optim_state_multiplier` to 8.0 (AdamW
/// FP32 m+v state = 2 × 4 bytes = 8 × param-fp32).
pub fn from_applied_plan(plan: &crate::wggo_apply::AppliedPlan) -> Self {
    Self {
        per_layer_param_bytes: plan.layers.iter().map(|l| l.param_bytes).collect(),
        per_layer_activation_bytes: plan.layers.iter().map(|l| l.activation_bytes).collect(),
        per_layer_compute_us: plan.layers.iter().map(|l| l.estimated_us).collect(),
        optim_state_multiplier: 8.0,
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-codegen cpdt_zero::model_size_from_applied_plan`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/cpdt_zero.rs
git commit -m "feat(cpdt): ModelSize::from_applied_plan constructor

Bridges WGGO's AppliedPlan to CPDT's ModelSize input shape. Sums per-layer
param/activation bytes directly from AppliedLayer; optim_state_multiplier
defaults to 8.0 (AdamW m+v FP32)."
```

---

## Task 2: AdamW hyperparam derivation from `@train` block

**Files:**
- Modify: `crates/nsl-codegen/src/stmt.rs` (add helper near `compile_quant_block`)
- Test: `crates/nsl-codegen/src/stmt.rs` test module

- [ ] **Step 1: Write the failing test**

Add to `stmt.rs` test module (find existing `mod tests` near the bottom):

```rust
#[test]
fn adamw_hyperparams_derived_from_train_block() {
    use crate::cpdt_optim::AdamWHyperparams;
    use nsl_parser::ast::{Expr, ExprKind, TrainBlock, TrainConfig};

    // Synthesize a TrainBlock with optimizer = AdamW(beta1=0.85, beta2=0.99)
    let train = TrainBlock {
        config: TrainConfig {
            optimizer: Some(Expr {
                kind: ExprKind::Call {
                    callee: Box::new(Expr::ident("AdamW")),
                    args: vec![
                        ("beta1".to_string(), Expr::float(0.85)),
                        ("beta2".to_string(), Expr::float(0.99)),
                    ],
                },
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    let adamw = adamw_from_train_block(Some(&train));
    assert!((adamw.beta1 - 0.85).abs() < 1e-9);
    assert!((adamw.beta2 - 0.99).abs() < 1e-9);
    // eps unspecified → default
    let default = AdamWHyperparams::default();
    assert!((adamw.eps - default.eps).abs() < 1e-12);
}

#[test]
fn adamw_hyperparams_default_when_no_train_block() {
    let adamw = adamw_from_train_block(None);
    let default = crate::cpdt_optim::AdamWHyperparams::default();
    assert_eq!(adamw.beta1, default.beta1);
    assert_eq!(adamw.beta2, default.beta2);
}
```

(If the actual `TrainBlock` / `TrainConfig` shape differs from the names above, adjust to match the real types found at `stmt.rs:3012` — that's the FASE site that already reads optimizer config.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p nsl-codegen stmt::tests::adamw_hyperparams`
Expected: FAIL with "function `adamw_from_train_block` not found".

- [ ] **Step 3: Implement the helper**

In `stmt.rs` near `compile_quant_block`:

```rust
/// Derive `AdamWHyperparams` from a `@train` block's optimizer config.
/// Returns library defaults when `train` is `None`, when the optimizer
/// is not `AdamW`, or for any β1/β2/ε field not explicitly set.
fn adamw_from_train_block(
    train: Option<&nsl_parser::ast::TrainBlock>,
) -> crate::cpdt_optim::AdamWHyperparams {
    use crate::cpdt_optim::AdamWHyperparams;
    use nsl_parser::ast::ExprKind;

    let mut hp = AdamWHyperparams::default();
    let Some(train) = train else { return hp; };
    let Some(opt_expr) = train.config.optimizer.as_ref() else { return hp; };

    let ExprKind::Call { callee, args } = &opt_expr.kind else { return hp; };
    let ExprKind::Ident(name) = &callee.kind else { return hp; };
    if name != "AdamW" { return hp; }

    for (kw, expr) in args {
        let Some(v) = expr.as_f64_literal() else { continue; };
        match kw.as_str() {
            "beta1" => hp.beta1 = v,
            "beta2" => hp.beta2 = v,
            "eps"   => hp.eps = v,
            _ => {}
        }
    }
    hp
}
```

(If `Expr` lacks `as_f64_literal`, inline the match on `ExprKind::FloatLit`/`IntLit`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p nsl-codegen stmt::tests::adamw_hyperparams`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-codegen/src/stmt.rs
git commit -m "feat(cpdt): derive AdamW hyperparams from @train block

Mirrors stmt.rs:3012's FASE optimizer-name lookup; extends to extract
beta1/beta2/eps when AdamW is the configured optimizer. Falls back to
library defaults for any unspecified field."
```

---

## Task 3: `Compiler` fields + `invoke_cpdt_if_enabled` plumbing

**Files:**
- Modify: `crates/nsl-codegen/src/compiler/mod.rs` (add fields + initializers)
- Modify: `crates/nsl-codegen/src/stmt.rs` (add helper + call site in `compile_quant_block`)

- [ ] **Step 1: Add Compiler fields**

In `crates/nsl-codegen/src/compiler/mod.rs`, find the `pub struct Compiler<'_>` definition (it already has `wrga_plan: Option<WrgaPlan>` and `wggo_overrides: Option<WggoOverrides>` — add adjacent):

```rust
/// CPDT mode requested via CLI. `Off` → planner does not run.
pub cpdt_mode: crate::cpdt::CpdtMode,
/// Cluster topology supplied via `--cpdt-num-gpus` / `--cpdt-intra-bw` / `--cpdt-inter-bw`.
/// `None` when CPDT is off.
pub cpdt_cluster: Option<crate::cpdt_zero::ClusterSpec>,
/// Resulting plan after `compile_quant_block` runs CPDT.
pub cpdt_plan: Option<crate::cpdt::CpdtPlan>,
/// Whether `--cpdt-report` was requested.
pub cpdt_report_requested: bool,
```

In every `Compiler::new*` initializer (search the file for existing `wrga_plan: None,` lines — add CPDT fields next to each):

```rust
cpdt_mode: crate::cpdt::CpdtMode::Off,
cpdt_cluster: None,
cpdt_plan: None,
cpdt_report_requested: false,
```

- [ ] **Step 2: Verify it builds**

Run: `cargo build -p nsl-codegen`
Expected: clean build (no E0063 missing-field errors — those are listed as the most common mistake in this project).

- [ ] **Step 3: Add `invoke_cpdt_if_enabled` to stmt.rs**

In `crates/nsl-codegen/src/stmt.rs`, add an `impl Compiler` method (near `invoke_wrga_if_enabled`):

```rust
fn invoke_cpdt_if_enabled(
    &mut self,
    applied_plan: &crate::wggo_apply::AppliedPlan,
    train_block: Option<&nsl_parser::ast::TrainBlock>,
) {
    use crate::cpdt::{CpdtInput, CpdtMode, run as cpdt_run};
    use crate::cpdt_expert::ExpertConfig;
    use crate::cpdt_joint::JointConfig;
    use crate::cpdt_precision::PrecisionConfig;
    use crate::cpdt_zero::ModelSize;
    use crate::wggo_overrides::WggoOverrides;

    if self.cpdt_mode == CpdtMode::Off {
        return;
    }
    let Some(cluster) = self.cpdt_cluster.clone() else { return; };

    let overrides = WggoOverrides::from_applied(applied_plan);
    let model = ModelSize::from_applied_plan(applied_plan);
    let adamw = adamw_from_train_block(train_block);

    let input = CpdtInput {
        mode: self.cpdt_mode,
        model,
        cluster,
        weights: None,
        precision_cfg: PrecisionConfig::default(),
        adamw,
        moe_shape: None,
        moe_router: None,
        moe_roofline_slack: 0.0,
        expert_cfg: ExpertConfig::default(),
        joint_cfg: JointConfig::default(),
        wggo_recommended_shard: overrides.min_shard_factor(),
    };

    self.cpdt_plan = Some(cpdt_run(input));
}
```

- [ ] **Step 4: Wire the call site in `compile_quant_block`**

Find the existing `invoke_wrga_if_enabled` call in `compile_quant_block`. Immediately after it, add:

```rust
self.invoke_cpdt_if_enabled(&applied_plan, train_block_opt.as_deref());
```

(`train_block_opt` is whatever local already references the active `@train` block; `applied_plan` is WGGO's output already in scope. If those names differ in the actual source, use the actual locals.)

- [ ] **Step 5: Build + lint**

Run: `cargo build -p nsl-codegen && cargo clippy -p nsl-codegen -- -D warnings`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/compiler/mod.rs crates/nsl-codegen/src/stmt.rs
git commit -m "feat(cpdt): Compiler fields + invoke_cpdt_if_enabled wiring

Adds cpdt_mode, cpdt_cluster, cpdt_plan, cpdt_report_requested to Compiler
and an invoke_cpdt_if_enabled helper called immediately after WRGA in
compile_quant_block. Mirrors WRGA's plumbing pattern. CLI defaults leave
cpdt_mode = Off, so existing builds are byte-identical."
```

---

## Task 4: Pipeline integration tests (TDD on the wiring)

**Files:**
- Create: `crates/nsl-codegen/tests/cpdt_pipeline_integration.rs`

- [ ] **Step 1: Write the two failing tests**

Create `crates/nsl-codegen/tests/cpdt_pipeline_integration.rs`:

```rust
//! End-to-end: WGGO recommendation → CPDT plan via Compiler.

use nsl_codegen::compiler::Compiler;
use nsl_codegen::cpdt::CpdtMode;
use nsl_codegen::cpdt_zero::ClusterSpec;
use nsl_codegen::wggo_overrides::OverrideRejectReason;

const FIXTURE_NSL: &str = include_str!("fixtures/cpdt_min_shard.nsl");

fn compile_with_cpdt(num_gpus: u32) -> Compiler<'static> {
    let mut compiler = Compiler::new_for_test();
    compiler.cpdt_mode = CpdtMode::Full;
    compiler.cpdt_cluster = Some(ClusterSpec {
        num_gpus,
        memory_budget_bytes: 80 * 1024 * 1024 * 1024,
        intra_bw_bps: 9e11,
        inter_bw_bps: 1e11,
        gpus_per_node: num_gpus.min(8),
    });
    compiler.compile_source(FIXTURE_NSL).expect("compile");
    compiler
}

#[test]
fn wggo_recommendation_flows_to_cpdt_plan() {
    // Fixture is constructed so WGGO produces shard_factor=4 on at least
    // one layer. With num_gpus=8, the planner should select s_p=4.
    let c = compile_with_cpdt(8);
    let plan = c.cpdt_plan.expect("CPDT must run when mode != Off");
    let zero = plan.zero.expect("ZeRO planner must produce a tuple");
    assert_eq!(zero.config.s_p, 4, "WGGO recommendation should propagate");
    assert!(plan.override_diagnostics.is_empty(),
            "no diagnostics expected on the happy path");
}

#[test]
fn world_size_mismatch_emits_diagnostic() {
    // num_gpus=5 doesn't divide WGGO's recommendation of 4 → Gate 1 reject.
    let c = compile_with_cpdt(5);
    let plan = c.cpdt_plan.expect("CPDT must run");
    let diag = plan.override_diagnostics.first()
        .expect("expected one ShardFactorIncompatibleWithWorldSize diagnostic");
    assert!(matches!(
        diag.reason,
        OverrideRejectReason::ShardFactorIncompatibleWithWorldSize { .. }
    ));
}
```

- [ ] **Step 2: Create the fixture**

Create `crates/nsl-codegen/tests/fixtures/cpdt_min_shard.nsl` with the smallest NSL program that:
- Compiles cleanly today.
- Includes a `@quant` block (so `compile_quant_block` runs).
- Triggers WGGO to produce a `shard_factor=4` recommendation on at least one layer.

If you don't already know a minimal fixture, copy from an existing test fixture in `crates/nsl-codegen/tests/fixtures/` that already exercises WGGO (search: `ls crates/nsl-codegen/tests/fixtures/ | grep -i wggo`) and add a `@quant` block if missing.

- [ ] **Step 3: Add `Compiler::new_for_test` if it doesn't exist**

Run: `grep -n "fn new_for_test" crates/nsl-codegen/src/compiler/mod.rs`

If absent, add a `#[cfg(test)]` (or `pub` if integration tests need it — this is an integration test so it needs `pub`) constructor on `Compiler`:

```rust
/// Construct a Compiler suitable for integration tests with all
/// optional features off and no source loaded yet.
pub fn new_for_test() -> Self {
    // ...mirror the simplest existing constructor, defaulting all
    // fields including the new CPDT fields.
}
```

If a similar helper already exists under a different name, use it and adjust the test imports.

- [ ] **Step 4: Run tests to verify they fail in the right way**

Run: `cargo test -p nsl-codegen --test cpdt_pipeline_integration`
Expected: tests compile and FAIL — the `cpdt_plan` field is `Some`, but either `s_p` mismatches or no diagnostic appears, depending on whether the fixture actually triggers the desired WGGO recommendation.

If the fixture doesn't produce `shard_factor=4`, iterate on the fixture (not the test) until WGGO emits the recommendation. Use `--cpdt-report` mentally: if needed, add a temporary `eprintln!("{:#?}", applied_plan)` in `invoke_cpdt_if_enabled` to inspect.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p nsl-codegen --test cpdt_pipeline_integration`
Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/tests/cpdt_pipeline_integration.rs crates/nsl-codegen/tests/fixtures/cpdt_min_shard.nsl crates/nsl-codegen/src/compiler/mod.rs
git commit -m "test(cpdt): pipeline integration — WGGO recommendation propagates

Two end-to-end tests exercise compile_quant_block → invoke_cpdt_if_enabled
→ cpdt::run with a real WGGO-produced AppliedPlan. Verifies shard_factor
propagation on the happy path and stderr-bound diagnostic emission on
world-size mismatch."
```

---

## Task 5: CLI flags + fail-fast validation

**Files:**
- Modify: `crates/nsl-cli/src/main.rs` (add 5 flags, plumb to `CompileOptions`/`Compiler`, add validation)

- [ ] **Step 1: Add the 5 CLI flags to the existing `clap` definitions**

Find the existing `--csha` and `--wrga-report` flag definitions in `main.rs`. Add adjacent:

```rust
/// Enable Compile-time Parallelism & Distributed Training planner.
/// Optional value: `full` (default), `zero_only`, or `off`.
#[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "full")]
cpdt: Option<String>,

/// Number of GPUs in the target cluster. Required when `--cpdt` is set.
#[arg(long, value_name = "N")]
cpdt_num_gpus: Option<u32>,

/// Intra-node bandwidth in bytes per second. Default: 9e11 (900 GB/s).
#[arg(long, value_name = "BPS", default_value_t = 9e11)]
cpdt_intra_bw: f64,

/// Inter-node bandwidth in bytes per second. Default: 1e11 (100 GB/s).
#[arg(long, value_name = "BPS", default_value_t = 1e11)]
cpdt_inter_bw: f64,

/// Emit the full CPDT plan to stdout. Implies `--cpdt` (Full mode unless
/// `--cpdt=...` is also specified).
#[arg(long, default_value_t = false)]
cpdt_report: bool,
```

(Add to **every** subcommand that accepts compilation flags — find them by grepping for `wrga_report` near args; the pattern repeats per subcommand.)

- [ ] **Step 2: Add the implication + validation logic**

In each subcommand handler that just parsed these flags, immediately after parsing add:

```rust
// --cpdt-report implies --cpdt (Full mode if not otherwise specified).
let cpdt_mode_str = match (cpdt.as_deref(), cpdt_report) {
    (Some(s), _) => Some(s.to_string()),
    (None, true) => Some("full".to_string()),
    (None, false) => None,
};

let cpdt_mode = match cpdt_mode_str.as_deref() {
    None => nsl_codegen::cpdt::CpdtMode::Off,
    Some(s) => match nsl_codegen::cpdt::CpdtMode::parse(s) {
        Some(m) => m,
        None => {
            eprintln!("error: --cpdt value '{}' is not one of full|zero_only|off", s);
            std::process::exit(2);
        }
    },
};

if cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off {
    let n = match cpdt_num_gpus {
        Some(n) if n >= 1 => n,
        Some(_) => {
            eprintln!("nsl: --cpdt-num-gpus must be ≥ 1");
            std::process::exit(2);
        }
        None => {
            eprintln!("nsl: --cpdt requires --cpdt-num-gpus N");
            std::process::exit(2);
        }
    };
    compiler.cpdt_mode = cpdt_mode;
    compiler.cpdt_cluster = Some(nsl_codegen::cpdt_zero::ClusterSpec {
        num_gpus: n,
        memory_budget_bytes: 80 * 1024 * 1024 * 1024, // generous default; future flag
        intra_bw_bps: cpdt_intra_bw,
        inter_bw_bps: cpdt_inter_bw,
        gpus_per_node: n.min(8),
    });
    compiler.cpdt_report_requested = cpdt_report;
}
```

(Wrap in a helper if it's repeated >2 times across subcommand handlers.)

- [ ] **Step 3: Add stderr diagnostic + stdout report rendering after `compile()` returns**

In each subcommand handler, after the existing post-compile WRGA report block:

```rust
// CPDT: stderr diagnostics always; stdout report on demand.
if let Some(plan) = compiler.cpdt_plan.as_ref() {
    for diag in &plan.override_diagnostics {
        eprintln!(
            "[cpdt] scope:global wggo-override-rejected requested={} applied={} reason={:?}",
            diag.requested, diag.applied, diag.reason
        );
    }
    if compiler.cpdt_report_requested {
        print!("{}", plan.render_report());
        println!();
        println!("=== Defaults Assumed ===");
        println!("precision_cfg: BF16-mixed (override: --cpdt-precision, future)");
        println!("joint_cfg:     {} iters tolerance={:.0e} (override: --cpdt-budget, future)",
            nsl_codegen::cpdt_joint::JointConfig::default().max_iterations,
            nsl_codegen::cpdt_joint::JointConfig::default().tolerance);
        println!("expert_cfg:    none (no MoE block detected)");
        println!("weights:       none (weight-aware refinement deferred)");
    }
}
```

(Field names on `JointConfig::default()` may differ — adjust to actual names; if the type doesn't have `max_iterations` / `tolerance`, print whatever fields it does have or hardcode the descriptive labels.)

- [ ] **Step 4: Build**

Run: `cargo build -p nsl-cli`
Expected: clean.

- [ ] **Step 5: Smoke-check by hand**

Run a known-compiling fixture with `--cpdt --cpdt-num-gpus 4`:
```bash
cargo run -p nsl-cli -- build crates/nsl-codegen/tests/fixtures/cpdt_min_shard.nsl --cpdt --cpdt-num-gpus 4
```
Expected: succeeds, no stderr `[cpdt]` line (no override conflict on this combo).

Then with `--cpdt-report`:
```bash
cargo run -p nsl-cli -- build crates/nsl-codegen/tests/fixtures/cpdt_min_shard.nsl --cpdt-report --cpdt-num-gpus 4
```
Expected: stdout contains `=== CPDT Training Plan ===` and `=== Defaults Assumed ===`.

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-cli/src/main.rs
git commit -m "feat(cli): --cpdt + --cpdt-report flags wire CPDT into pipeline

Adds --cpdt[=mode], --cpdt-num-gpus (required when CPDT enabled),
--cpdt-intra-bw, --cpdt-inter-bw, --cpdt-report. --cpdt-report implies
--cpdt. Stderr diagnostics fire whenever CPDT runs; full plan + Defaults
Assumed footer goes to stdout only with --cpdt-report."
```

---

## Task 6: CLI flag tests

**Files:**
- Create: `crates/nsl-cli/tests/cpdt_cli.rs`

- [ ] **Step 1: Write the three CLI tests**

Create `crates/nsl-cli/tests/cpdt_cli.rs`:

```rust
//! CPDT CLI flag behaviour: --cpdt-report implies --cpdt; --cpdt requires
//! --cpdt-num-gpus; --cpdt-num-gpus must be ≥ 1.

use assert_cmd::Command;
use predicates::prelude::*;

const FIXTURE: &str = "crates/nsl-codegen/tests/fixtures/cpdt_min_shard.nsl";

#[test]
fn bare_cpdt_report_enables_full_mode() {
    Command::cargo_bin("nsl").unwrap()
        .arg("build").arg(FIXTURE)
        .arg("--cpdt-report").arg("--cpdt-num-gpus").arg("4")
        .assert()
        .success()
        .stdout(predicate::str::contains("=== CPDT Training Plan ==="))
        .stdout(predicate::str::contains("Mode: full"))
        .stdout(predicate::str::contains("=== Defaults Assumed ==="));
}

#[test]
fn missing_num_gpus_fast_fails() {
    Command::cargo_bin("nsl").unwrap()
        .arg("build").arg(FIXTURE)
        .arg("--cpdt")
        .assert()
        .failure()
        .stderr(predicate::str::contains("--cpdt requires --cpdt-num-gpus"));
}

#[test]
fn zero_num_gpus_fast_fails() {
    Command::cargo_bin("nsl").unwrap()
        .arg("build").arg(FIXTURE)
        .arg("--cpdt").arg("--cpdt-num-gpus").arg("0")
        .assert()
        .failure()
        .stderr(predicate::str::contains("must be ≥ 1"));
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p nsl-cli --test cpdt_cli`
Expected: all three PASS.

- [ ] **Step 3: Run full test suite to detect regressions**

Run: `cargo test --workspace`
Expected: PASS. If any pre-existing test now fails, investigate — likely cause is a missed `Compiler` initializer site (E0063 missing-fields, the most common project mistake).

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-cli/tests/cpdt_cli.rs
git commit -m "test(cli): three CPDT CLI flag behaviour tests

Covers --cpdt-report implication, --cpdt-num-gpus required, and zero-gpu
fast-fail. Uses the same fixture as the codegen-side integration tests."
```

---

## Task 7: Update memory file

**Files:**
- Modify: `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/project_wggo_consumers.md`

- [ ] **Step 1: Edit the Consumer 5 section**

Replace the `## Consumer 5: CPDT/Sharding (consumer API shipped 2026-04-14; pipeline glue deferred)` heading and "Pipeline integration deferred" subsection with:

```markdown
## Consumer 5: CPDT/Sharding (shipped 2026-04-15)

Consumer-side API merged 2026-04-14 (PR `feat/wggo-cpdt-wiring`); pipeline
integration merged 2026-04-15 (PR `feat/cpdt-pipeline-integration`).
```

Then add a sub-section documenting the integration shape:

```markdown
**Pipeline integration:**
- 5 CLI flags: `--cpdt[=full|zero_only|off]`, `--cpdt-num-gpus N` (required),
  `--cpdt-intra-bw` (default 9e11), `--cpdt-inter-bw` (default 1e11),
  `--cpdt-report` (implies `--cpdt`).
- `Compiler` carries `cpdt_mode`, `cpdt_cluster`, `cpdt_plan`, `cpdt_report_requested`.
- `compile_quant_block` calls `invoke_cpdt_if_enabled` after WRGA.
- `ModelSize::from_applied_plan` bridges WGGO's `AppliedPlan` to CPDT input;
  required adding `param_bytes` + `activation_bytes` to `AppliedLayer`.
- AdamW hyperparams derived from `@train` block via `adamw_from_train_block`;
  defaults when no `@train` block present or non-AdamW optimizer.
- Stderr `[cpdt] scope:global ...` diagnostic format matches CSHA/WRGA precedent.
- Stdout `--cpdt-report` ends with `=== Defaults Assumed ===` footer documenting
  hardcoded values (`precision_cfg`, `joint_cfg`, `expert_cfg`, `weights`).

**Out of scope (still deferred):** IR mutation (sharding codegen), MoE detection,
`--cpdt-precision` / `--cpdt-budget` / `--cpdt-config` flags, multi-host topology,
weight-aware refinement.

**Spec:** `docs/superpowers/specs/2026-04-15-cpdt-pipeline-integration-design.md`
**Plan:** `docs/superpowers/plans/2026-04-15-cpdt-pipeline-integration-implementation.md`
```

- [ ] **Step 2: Update top-of-file rollout status**

Find the line near the top that summarizes consumer status. Update from:
> Consumer 1 (CSHA) merged 2026-04-14. WRGA, FASE, Prune, Sharding pending follow-up plans...

to:
> Consumers 1 (CSHA), 2 (WRGA), and 5 (CPDT) shipped. FASE and Prune still pending their internal refactors.

- [ ] **Step 3: No commit needed (memory file is outside the repo)**

Memory files live in `C:/Users/bwiem/.claude/projects/c--Users-bwiem-projects-NSL/memory/` — not under git. Skip the commit step.

---

## Task 8: PR

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/cpdt-pipeline-integration
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "feat(cpdt): pipeline integration (WGGO consumer 5)" --body "$(cat <<'EOF'
## Summary
- Wires `cpdt::run` into `compile_quant_block` end-to-end via `invoke_cpdt_if_enabled`.
- Adds 5 CLI flags: `--cpdt`, `--cpdt-num-gpus`, `--cpdt-intra-bw`, `--cpdt-inter-bw`, `--cpdt-report`.
- Closes out the deferred glue from the 2026-04-14 wggo-cpdt-wiring spec; WGGO consumer 5 fully shipped.

## Test plan
- [ ] `cargo test -p nsl-codegen --test cpdt_pipeline_integration` — 2 tests pass
- [ ] `cargo test -p nsl-cli --test cpdt_cli` — 3 tests pass
- [ ] `cargo test --workspace` — no regressions
- [ ] Manual: `nsl build fixture.nsl --cpdt-report --cpdt-num-gpus 4` emits report + Defaults footer
- [ ] Manual: `nsl build fixture.nsl --cpdt` exits non-zero with `--cpdt-num-gpus` required message

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist (run before claiming done)

- [ ] Every spec section (§3 CLI, §4 outputs, §5 integration site, §6 inputs, §7 testing) has at least one task implementing it.
- [ ] No `TBD` / `TODO` / `implement later` / "fill in" anywhere in the plan.
- [ ] Method names consistent across tasks: `invoke_cpdt_if_enabled`, `adamw_from_train_block`, `ModelSize::from_applied_plan`, fields `cpdt_mode`/`cpdt_cluster`/`cpdt_plan`/`cpdt_report_requested`.
- [ ] Every code-touching step shows the actual code.
- [ ] Test commands include expected pass/fail.
- [ ] Project's most-common mistake (E0063 missing-field initializers) is explicitly called out at Task 3 Step 2.
