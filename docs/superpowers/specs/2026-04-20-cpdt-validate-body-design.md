# CPDT `validate(wm, applied)` Body — Layer-Prefix Validation Design

> **Framing:** Fills in the `cpdt_sensitivity::validate` stub shipped with Phase 1 (PR #80). Layer-prefix validation only — catches the "wrong-checkpoint-entirely" class of mismatch at plan-time. Per-tensor shape/dtype validation is deferred to Phase 2 where the spectral-factor wiring requires per-tensor metadata anyway.

**Parent specs:**
- [2026-04-18-cpdt-weight-aware-phase1-design.md](2026-04-18-cpdt-weight-aware-phase1-design.md)
- [2026-04-19-cpdt-calibration-correction-design.md](2026-04-19-cpdt-calibration-correction-design.md)

**Worktree:** `.worktrees/cpdt-validate-body` (branch `feat/cpdt-validate-body`, based on `main`)

---

## 1. Motivation

`cpdt_sensitivity::validate(wm, applied)` shipped as a `TODO(plan)` stub in Phase 1 Commit 1. Today when a user passes a checkpoint that doesn't match their model declaration — e.g. a HuggingFace-format Llama checkpoint (tensors prefixed `transformer.h.0.`, `model.layers.0.`) passed to an NSL-native GPT-2 model (tensors prefixed `blocks.0.`) — the build succeeds with a `WeightMap` that contains zero matching tensors for the model's declared layers, and the CPDT scorer produces empty / nonsense tier assignments that surface downstream as corrupt codegen, wrong-gradient training, or deep pipeline errors. The silent-failure mode costs significant debugging time.

Layer-prefix validation catches this class of mismatch at plan-time, before CPDT runs, with an error message specific enough to diagnose the root cause.

## 2. Scope & Non-Goals

**In scope:**

- Per-layer existence check: for every `AppliedLayer` whose `layer_name` follows the `blocks.N` / `layers.N` / `h.N` pattern, require at least one tensor in the `WeightMap` whose name starts with `layer_name + "."`.
- Aggregation: collect ALL missing layers into a single error rather than fail-fast on the first. Include a summary of the weight map's tensor-name prefixes so the user can diagnose format mismatches (e.g. HuggingFace vs NSL-native).
- Wire validation into `invoke_cpdt_if_enabled` in `stmt.rs` — fail before `cpdt_run` if the check fails.

**Out of scope (deferred to Phase 2):**

- Per-tensor shape validation.
- Per-tensor dtype validation.
- Validation of the `"other"` catch-all layer (embeddings, norms, LM head) — its contents are heterogeneous and per-tensor metadata is unavailable today.
- Heuristic "you passed a HuggingFace checkpoint" hints. Deferred; can be added as a follow-up if the prefix-summary isn't dense enough in practice.

**Pre-dispatch finding** (noted in the retrospective addendum of this session): `AppliedPlan`'s `layer_name` vocabulary is `{blocks.N, layers.N, h.N, "other"}` — not the broader set (`tok_embeddings`, `output`, `final_norm`, etc.) the initial Option-A pitch anticipated. This simplifies the matching rule: only the three hierarchical prefixes need handling; the `"other"` bucket is skipped with an explicit reason.

## 3. Design

### 3.1 Matching rule

For each `AppliedLayer`:

```text
if layer_name starts with "blocks.", "layers.", or "h.":
    require >=1 tensor t in WeightMap where t.name starts with layer_name + "."
else if layer_name == "other":
    skip  (heterogeneous catch-all; Phase 2 extends per-tensor)
else:
    unknown layer form; log a warning to stderr (tolerant — don't fail for
    future expansion of the vocabulary)
```

The matching rule is simpler than the original Option-A pitch anticipated because `AppliedPlan`'s vocabulary is constrained by `wggo_graph::layer_prefix`.

### 3.2 Error aggregation

Collect every missing layer into a `Vec<String>`. If the vec is non-empty after iteration, emit a single aggregated error:

```text
error: weight map does not match model declaration.
  Missing layers (<N>): blocks.0, blocks.1, blocks.2, ...
  WeightMap contains <M> tensors with top-level prefixes:
    transformer.h.0, transformer.h.1, transformer.h.2, ...
  (first 8 unique prefixes shown; run `nsl show-weights <path>` for full list)
```

The "top-level prefix" of a tensor name is computed by taking the substring up to the **second** dot (or the first dot if only one dot exists). For `transformer.h.0.attn.wq.weight`, the top-level prefix is `transformer.h`. For `tok_embeddings.weight`, it's `tok_embeddings`. This gives enough signal for the user to spot format mismatches without listing every tensor.

Prefix set is deduplicated and sorted; if the set exceeds 8 entries, truncate with `"..."` and note the count.

### 3.3 Wiring

`cpdt_sensitivity::validate` returns `Result<(), ValidationError>` (existing signature). New variants needed:

```rust
pub enum ValidationError {
    // Existing variants (kept for Phase 2):
    MissingTensor { tensor_name: String },
    ShapeMismatch { ... },
    DtypeMismatch { ... },
    // New Phase 1 variant:
    LayersMissing {
        missing: Vec<String>,            // e.g. ["blocks.0", "blocks.1"]
        total_layers_checked: usize,
        weight_map_prefix_summary: Vec<String>,  // deduplicated, sorted, capped
        weight_map_total_tensors: usize,
    },
}
```

`Display for ValidationError::LayersMissing` produces the aggregated error message above.

Call site: `stmt.rs::invoke_cpdt_if_enabled`. Before the `let plan = cpdt_run(input);` line, if `weight_map_ref` is `Some(wm)` AND `compiler.cpdt_mode == CpdtMode::Full`:

```rust
if let Err(e) = cpdt_sensitivity::validate(wm, applied_plan) {
    eprintln!("{}", e);
    return;  // skip CPDT entirely; compilation aborts through downstream errors
}
```

The `return` means CPDT is skipped when validation fails; the compilation may proceed to produce a binary without CPDT's optimizer-precision plan, which is the correct behavior (the user's model has a weight-mismatch problem that CPDT can't fix — other passes may still be meaningful). Downstream errors will surface the bad weights eventually via mismatched tensor operations.

Alternative: convert to a hard compiler error by calling `std::process::exit(1)` after the `eprintln!`. This prevents any further work on a broken checkpoint. Decision: **hard exit** — a weight-map that can't be validated is a user-visible bug that should stop the build with a clear signal rather than proceed and produce downstream symptoms. The `return` approach is for deferrable errors; this is not deferrable.

## 4. Commit Sequencing

Single commit on `feat/cpdt-validate-body`:

```text
feat(cpdt): layer-prefix validation in cpdt_sensitivity::validate
```

Files modified:

- `crates/nsl-codegen/src/cpdt_sensitivity.rs` — fill in `validate` body + add `LayersMissing` variant + Display impl.
- `crates/nsl-codegen/src/stmt.rs` — wire the validate call into `invoke_cpdt_if_enabled` before `cpdt_run`.
- `crates/nsl-codegen/tests/cpdt_validate_body.rs` — new test file, four unit tests.
- `docs/superpowers/specs/2026-04-18-cpdt-weight-aware-phase2-stub.md` — add per-tensor-validation entry to the Phase 2 scope section.

## 5. Test Plan

Four unit tests in `crates/nsl-codegen/tests/cpdt_validate_body.rs`:

1. **`all_layers_matched_green`**: Build a synthetic `AppliedPlan` with 3 `blocks.N` layers; build a synthetic `WeightMap` containing tensors for all three (e.g. `blocks.0.attn.wq.weight`, `blocks.1.attn.wq.weight`, `blocks.2.attn.wq.weight`). `validate` returns `Ok(())`.

2. **`single_missing_layer_red`**: `AppliedPlan` with 3 `blocks.N`; WeightMap missing `blocks.1`. `validate` returns `Err(LayersMissing)` listing only `blocks.1`.

3. **`multiple_missing_layers_red_aggregates`**: `AppliedPlan` with 12 `blocks.N`; WeightMap has `transformer.h.N`-prefixed tensors (HuggingFace-style, all 12 missing). `validate` returns `Err(LayersMissing)` listing all 12 missing layers. Error message contains `blocks.0` through `blocks.11` AND `transformer.h` in the prefix summary.

4. **`empty_weightmap_red`**: `AppliedPlan` with 3 `blocks.N`; WeightMap has 0 tensors. `validate` returns `Err(LayersMissing)` with all 3 layers listed and an empty prefix summary.

All tests use `PrecisionConfig::default()` and a small synthetic `AppliedPlan` constructed via the public `apply()` function or a direct struct literal. No fixture-file access required.

## 6. Phase 2 Integration

Phase 2's spectral-factor wiring requires per-tensor metadata (shape, dtype, RMS) to compute spectral conditioning. When that per-tensor metadata pipeline lands, `cpdt_sensitivity::validate` extends to use it: for every declared tensor, cross-check name, shape, and dtype against the WeightMap. At that point the `MissingTensor` / `ShapeMismatch` / `DtypeMismatch` variants (already in the enum, presently unused) become active.

The Phase 1 layer-prefix validation stays in place as the fast-path first check; per-tensor validation runs only if layer-prefix passes. This preserves the aggregation-based diagnostic message for the common case (wrong-checkpoint-entirely) while adding fine-grained detail for the rarer case (one-tensor-mismatched).

## 7. Close-Out Criteria

- `cpdt_sensitivity::validate` body fills in layer-prefix check with `LayersMissing` variant.
- Four unit tests in `cpdt_validate_body.rs` all pass.
- `invoke_cpdt_if_enabled` calls validate before cpdt_run; hard-exit on failure.
- All existing tests still pass (61 lib + 32 integration).
- Phase 2 stub's scope section documents per-tensor validation as the extension path.
- Single commit; PR opened.
