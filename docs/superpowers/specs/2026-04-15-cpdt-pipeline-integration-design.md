# CPDT Pipeline Integration — Design

**Date:** 2026-04-15
**Status:** Approved for implementation
**Branch (target):** `feat/cpdt-pipeline-integration`
**Predecessor:** [WGGO → CPDT consumer wiring](2026-04-14-wggo-cpdt-wiring-design.md) (consumer-side API shipped; pipeline glue deferred to this spec)

## 1. Goal

Wire `cpdt::run` into the compile pipeline so WGGO's `min_shard_factor` recommendation reaches CPDT's planner end-to-end. Phase 1 is a **pure planner integration** — no IR mutation, no codegen rewrites — mirroring WRGA Milestone B.1's "invoke + capture + report" shape.

After this lands, the WGGO 5-consumer rollout has all five consumer surfaces wired (CSHA + WRGA shipped; FASE/Prune still pending their own internal refactors).

## 2. Non-Goals (Explicitly Deferred)

- IR mutation: sharding codegen, communication-op insertion, optimizer-state partitioning.
- MoE detection / `expert_cfg` derivation from the AST.
- `--cpdt-precision`, `--cpdt-budget`, `--cpdt-config` flags.
- Multi-host or heterogeneous topology (config-file approach).
- Weight-aware refinement (`CpdtInput.weights = Some(...)`).

## 3. CLI Surface

| Flag | Required? | Default | Notes |
|---|---|---|---|
| `--cpdt[=full\|zero_only\|off]` | no | `off` | Bare `--cpdt` ⇒ `Full`. |
| `--cpdt-num-gpus N` | **yes when `--cpdt` enabled** | — | `N ≥ 1`. |
| `--cpdt-intra-bw F` | no | `9e11` (900 GB/s) | Bytes/sec. Matches existing CPDT test fixtures. |
| `--cpdt-inter-bw F` | no | `1e11` (100 GB/s) | Bytes/sec. Matches existing CPDT test fixtures. |
| `--cpdt-report` | no | off | Emits full plan to stdout. **Implies `--cpdt`** (Full mode unless `--cpdt=...` already specified). |

### 3.1 Failure Modes
1. `--cpdt` enabled without `--cpdt-num-gpus` → fast error: `nsl: --cpdt requires --cpdt-num-gpus N`.
2. `--cpdt-num-gpus 0` → fast error: `nsl: --cpdt-num-gpus must be ≥ 1`.
3. `--cpdt-report` set but no `@train` / `@quant` block compiled → warning to stderr, no stdout report. Mirrors `--wrga-report`'s behavior at `main.rs:2009`.
4. `--cpdt-num-gpus N` where `N % wggo_recommended_shard != 0` → already handled by `cpdt::run` Gate 1; emits the standard `[cpdt] scope:global wggo-override-rejected ...` stderr diagnostic.

## 4. Outputs

### 4.1 Stderr (always when CPDT runs)
For each `OverrideDiagnostic` in `CpdtPlan.override_diagnostics`:
```
[cpdt] scope:global wggo-override-rejected requested=R applied=A reason=<reason_string>
```
Format matches the CSHA/WRGA precedent. Silent when WGGO and CPDT agree (or when WGGO produced no recommendation).

### 4.2 Stdout (only with `--cpdt-report`)
Full plan via `CpdtPlan::render_report()` — already implemented; emits ZeRO tuple, comm schedule, precision tiers, optim bytes, MoE placement (when present), and joint solver convergence.

**New addition:** A `=== Defaults Assumed ===` footer documenting every hardcoded input the integration supplied so users see what's available to override later. Format:
```
=== Defaults Assumed ===
precision_cfg: BF16-mixed (override: --cpdt-precision, future)
joint_cfg:     budget=<N> tolerance=<T> (override: --cpdt-budget, future)
expert_cfg:    none (no MoE block detected)
weights:       none (weight-aware refinement deferred)
```

## 5. Integration Site

**File:** `crates/nsl-codegen/src/stmt.rs::compile_quant_block`
**Position:** After existing CSHA and WRGA invocations (CSHA → WRGA → CPDT). The existing reorder from the WGGO-CSHA wiring already establishes Calibration → WGGO → consumers ordering; CPDT slots in last among the consumers.

**New helper:** `Compiler::invoke_cpdt_if_enabled(&mut self, applied_plan: &AppliedPlan, train_block: Option<&TrainBlock>)`
- Signature mirrors `invoke_wrga_if_enabled`.
- Early-returns when `self.cpdt_mode == CpdtMode::Off`.
- Builds `CpdtInput` per §6, calls `cpdt::run`, stores result on `self.cpdt_plan`.

**New `Compiler` field:** `cpdt_plan: Option<CpdtPlan>` — paralleling `wrga_plan`. The CLI layer reads this after `compile()` returns and renders the report (§4.2) and stderr diagnostics (§4.1).

## 6. Input Derivation

| `CpdtInput` field | Source |
|---|---|
| `mode` | `--cpdt` value (parsed via `CpdtMode::parse`). |
| `cluster` | `ClusterSpec { num_gpus: --cpdt-num-gpus, intra_bw_bps: --cpdt-intra-bw, inter_bw_bps: --cpdt-inter-bw }`. |
| `model: ModelSize` | New helper `ModelSize::from_applied_plan(&AppliedPlan) -> ModelSize`. Sums per-layer param counts already present in `AppliedPlan`. |
| `wggo_recommended_shard` | `WggoOverrides::from_applied(&applied).min_shard_factor()` — already implemented (consumer-API spec). |
| `adamw` | Walk `@train` block AST for the optimizer config; mirror `stmt.rs:3012` (FASE's optimizer-name lookup) and extend to read β1/β2/ε from the `optimizer = AdamW(...)` arg list. Fall back to `AdamWHyperparams::default()` when no `@train` block or non-AdamW optimizer. |
| `precision_cfg` | Hardcoded `PrecisionConfig::default()` (BF16-mixed). Documented in report footer. |
| `joint_cfg` | Hardcoded `JointConfig::default()` (matches CPDT test defaults). Documented in report footer. |
| `expert_cfg` | `ExpertConfig::default()`. |
| `moe_shape` / `moe_router` | `None`. |
| `moe_roofline_slack` | `0.0`. |
| `weights` | `None`. |

## 7. Testing

### 7.1 CLI flag tests
**File (new):** `crates/nsl-cli/tests/cpdt_cli.rs`
1. **`bare_cpdt_report_enables_full_mode`** — `nsl build --cpdt-report --cpdt-num-gpus 4 ...` produces stdout containing `Mode: full` and the `=== Defaults Assumed ===` footer.
2. **`missing_num_gpus_fast_fails`** — `nsl build --cpdt ...` (no `--cpdt-num-gpus`) exits non-zero with stderr containing `--cpdt requires --cpdt-num-gpus`.
3. **`zero_num_gpus_fast_fails`** — `--cpdt --cpdt-num-gpus 0` exits non-zero with stderr containing `must be ≥ 1`.

### 7.2 Integration tests
**File (new):** `crates/nsl-codegen/tests/cpdt_pipeline_integration.rs`
1. **`wggo_recommendation_flows_to_cpdt_plan`** — Compile a fixture with WGGO active and a per-layer `shard_factor` recommendation. Assert `compiler.cpdt_plan.unwrap().zero.unwrap().config.s_p` matches the recommendation when divisible.
2. **`world_size_mismatch_emits_diagnostic`** — Set `--cpdt-num-gpus 5` with WGGO recommending `shard_factor=4`. Assert `cpdt_plan.override_diagnostics` contains a `ShardFactorIncompatibleWithWorldSize` entry.

### 7.3 AdamW derivation unit test
**File:** `crates/nsl-codegen/src/stmt.rs` test module (extend existing).
1. **`adamw_hyperparams_derived_from_train_block`** — Parse a tiny `@train` block with `optimizer = AdamW(beta1=0.85, beta2=0.99)`. Assert the helper that builds `CpdtInput.adamw` returns those values rather than the library defaults.

**Total: 6 new tests** (3 CLI + 2 integration + 1 unit).

## 8. Architecture Diagram

```
┌──────────────────────┐
│  CLI: --cpdt* flags  │
└─────────┬────────────┘
          ▼
┌──────────────────────┐    ┌────────────────────┐
│  CompileOptions      │───▶│  Compiler {        │
│   .cpdt_mode         │    │    cpdt_mode,      │
│   .cluster_spec      │    │    cluster,        │
│   .cpdt_report       │    │    cpdt_plan,      │
└──────────────────────┘    │    wggo_overrides  │
                            │  }                 │
                            └─────────┬──────────┘
                                      │
                                      ▼
                  ┌─────────────────────────────────────┐
                  │  compile_quant_block                │
                  │    ├─ Calibration                   │
                  │    ├─ WGGO   ──▶ AppliedPlan        │
                  │    ├─ CSHA   (reads overrides)      │
                  │    ├─ WRGA   (reads overrides)      │
                  │    └─ CPDT   ◀── NEW                │
                  │         invoke_cpdt_if_enabled()    │
                  │         ↓                           │
                  │      cpdt::run(input)               │
                  │         ↓                           │
                  │      self.cpdt_plan = Some(plan)    │
                  └────────────────┬────────────────────┘
                                   │
                                   ▼
                  ┌─────────────────────────────────────┐
                  │  CLI post-compile:                  │
                  │    stderr ← override_diagnostics    │
                  │    stdout ← render_report()         │
                  │              + Defaults footer      │
                  │              (only if --cpdt-report)│
                  └─────────────────────────────────────┘
```

## 9. Risks & Open Questions

- **Risk:** `ModelSize::from_applied_plan` may need fields not currently exposed on `AppliedLayer`. Mitigation: pre-implementation discovery task to verify the param-count fields are accessible; if not, extend `AppliedLayer` (small) before the integration task.
- **Risk:** AdamW hyperparam parsing in the `@train` block AST may turn out non-trivial. Mitigation: scope the unit test as the first task and accept defaults-only for Phase 1 if the AST shape is hostile — derivation is an enhancement, not a blocker for the WGGO contract.
- **Open:** Where exactly should the CPDT report print relative to other CLI output (before/after `--wrga-report`, etc.)? Defer to implementation: place after `--wrga-report` in source order, document in the implementation plan.

## 10. Success Criteria

1. `nsl build --cpdt --cpdt-num-gpus 4 fixture.nsl` runs CPDT's planner and stores the plan on the `Compiler`.
2. `--cpdt-report` emits the full plan to stdout, terminating with the `Defaults Assumed` footer.
3. WGGO's `min_shard_factor` reaches `cpdt::run` and is honored when feasible.
4. WGGO/CPDT disagreements emit the standard `[cpdt] scope:global ...` stderr diagnostic regardless of `--cpdt-report`.
5. All 6 new tests pass; no existing tests regress.
6. Memory file `project_wggo_consumers.md` updated to mark Consumer 5 fully shipped.
