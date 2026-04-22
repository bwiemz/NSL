# WGGO Prune v1 — Real IR Rewrite Design

**Status:** design approved 2026-04-22; implementation plan pending.
**Branch:** `feat/wggo-prune-ir-rewrite`.
**Predecessor:** PR #102 landed the diagnostic stub (stderr `[prune] … reason=ir_rewrite_not_implemented`).
**Scope:** v1 (sub-block residual pruning). Whole-block chain-collapse deferred to v2.

---

## Summary

v1 implements the real WGGO Prune IR rewrite: WGGO's DP planner emits `CoarseDecision::Prune` for layers below the importance floor; this consumer turns each `Attention` / `Ffn` sub-block prune decision into a Wengert-level rewrite that (a) deletes the sub-block's compute ops and (b) aliases the sub-block's residual output to its residual input (identity). The rewrite runs before `wrga_prune::run()` in `stmt.rs`, so WRGA Prune and source-AD both observe the already-reduced forward.

Every precondition the rewrite depends on is a first-class check with a three-part error message. The rewrite never falls back to a weaker transformation when a precondition fails — it refuses. This spec inherits the *transformation-precondition-refusal* rule (see `memory/feedback_transformation_precondition_refusal.md`).

---

## Scope

**v1 supports** pre-norm transformer sub-blocks with residual structure, specifically `LayerRole::{Attention, Ffn}`. The planner's current `CoarseDecision::Prune` emission criterion is unchanged; v1 faithfully executes those decisions for sub-block roles.

**Out of scope (v2):**

- Whole-block prune (`LayerRole::Block`). Requires chain-collapse across multiple residual Adds; its preconditions are a separate design exercise.
- Non-residual architectures (SSM, Mamba, non-standard MoE). These naturally hit v1's refusal cases §3.2 and §3.5 at the sub-block level; no special handling is added.
- Post-norm transformers and scaled-residual architectures. These pattern-match as non-standard and may hit §3.3 (parallel residuals) or §3.4 (ambiguous pattern). A future version can relax the pattern-match; v1 rejects them loudly.
- WGGO Phase 2 integration. Phase 2 (gradient-based importance scoring) is blocked on AWQ retention debug. v1 is independent: the rewrite consumes any `AppliedPlan` with `CoarseDecision::Prune` entries regardless of how the planner produced them.

---

## §1. Semantic contract: residual-identity alias

### §1.1 What "prune sub-block N" means

In the standard pre-norm transformer, a sub-block has residual structure:

```text
h_before                                   # input to sub-block
x_norm  = RMSNorm(h_before)                # sub-block prelude
out     = SubBlock(x_norm)                 # sub-block body (Attn or FFN)
h_after = h_before + out                   # residual Add — the sub-block output
```

Pruning means `h_after := h_before`. Concretely:

1. Remove every op that computes `x_norm`, `out`, and any internal intermediates.
2. Remove the residual `Add` op itself.
3. Repoint every consumer of the Add's output VarId to use `h_before` directly.

The net effect on downstream computation is that `SubBlock` contributes nothing — the residual stream flows through untouched. This is the classical "skip this block" semantic of residual networks.

### §1.2 Precondition discipline: refuse, do not fall back

If the sub-block does not have residual structure, the rewrite **refuses** and emits a three-part error. It does not substitute "zero-contribution" (`h_after = h_before + 0`), does not fall back to "attempt identity and emit a diagnostic," and does not silently skip the decision. Rationale: every plausible fallback produces different semantics in the edge case the fallback exists for, and diagnostic-gated fallbacks train users to ignore the diagnostic over time.

This discipline generalizes: compiler transformations refuse when preconditions aren't met; they do not fall back to weaker transformations with different semantics. Two adjacent prior instances: CPDT `one-@cpdt-per-program` enforcement, PCA Tier A convention-match rule.

### §1.3 Pattern-match contract

The rewrite searches for exactly one op in the sub-block's closure (see §2) whose shape is:

```text
Add(h_before, block_output)
  where block_output ∈ closure ∧ consumers(block_output) == {this_Add}
```

Any deviation fails refusal. The specific deviations and their refusal variants are enumerated in §3.

---

## §2. Layer identification: parameter-anchored data-flow closure

### §2.1 Rationale

`AppliedLayer.layer_name` is a string like `"blocks.7.attn"`. The Wengert list has no layer-boundary markers; only VarIds and (for parameters/named intermediates) `var_names` entries. A workable "op belongs to layer N" rule must be computable from existing IR state without schema changes.

A field-based approach (tagging each `WengertOp` with its block/layer at extraction time) was considered and rejected. Field-based representations invite drift: any future pass that merges, splits, or inlines ops across layers must correctly maintain the field, and silent divergence is invisible until it causes wrong output. A computed-property approach cannot drift — it is recomputed from current IR state and always reflects reality.

### §2.2 Closure rule

```text
closure(layer_name) = transitive forward-closure of ops whose input set includes
                     any VarId whose var_name matches prefix "{layer_name}."
                     UNION the ops that directly produce any such parameter VarId,
                     computed up to but NOT INCLUDING the residual Add at the layer boundary.
```

Precise three-category treatment of VarIds during rewrite:

| Category | Treatment |
| --- | --- |
| Ops in the closure | **deleted** |
| The residual Add at the boundary | **rewritten** (consumers of `h_after` repointed to `h_before`), then **deleted** |
| `h_before` (the residual input) | **untouched** (belongs to the prior stream / prior layer) |

The distinction matters: if the implementer includes the Add in the closure and deletes it before the repointing step, consumers of `h_after` dangle and the rewrite is incorrect. The spec fixes these three categories as separate implementation concerns.

**Parameter producer ops:** ops that directly materialize a `{layer_name}.*` parameter VarId (e.g., weight-load ops reading from the WeightMap) are included in the closure and deleted alongside their consumers. The cross-layer-param check in §2.3 (precondition #2) subsumes the correctness precondition: if every consumer of a parameter VarId is inside the closure (no §3.1 violation), then the producer has no remaining consumers after deletion and is safe to delete. If a §3.1 violation exists, the refusal fires before the closure commits — parameter-producer deletion is never attempted on a parameter with external consumers.

### §2.3 Preconditions for correctness (all checked, all refuse on violation)

1. **Non-empty parameters** — at least one VarId in `var_names` has the prefix `{layer_name}.`. If zero match, the closure is empty and the prune target doesn't exist (refusal §3.5).
2. **No cross-layer parameter consumption** — every layer-N parameter is consumed only by ops inside the closure. If any layer-N parameter has a consumer outside the closure, the layer participates in cross-layer parameter sharing (tied embeddings, shared ALiBi biases, etc.) and v1 refuses (refusal §3.1). This check is load-bearing for correctness regardless of how rare the case is in practice; the rewrite's contract is independent of the planner's current policy.
3. **Single residual boundary** — exactly one op in the closure pattern-matches `Add(h_before, block_output)` with `block_output` single-use. Zero, multiple-parallel, or ambiguous cases each refuse with distinct error messages (refusals §3.2, §3.3, §3.4).

---

## §3. Refusal surface — seven cases, three-part errors

Every refusal emits a three-part error naming (i) what was requested, (ii) what structure was expected, (iii) what structure was found. Error messages are fixed by this spec, not left to implementer discretion.

### §3.1 Cross-layer parameter consumption

Trigger: a VarId with name `{layer_name}.<suffix>` has a consumer op that is NOT in the closure.

```text
prune: layer has cross-layer parameter sharing (not supported in v1).
  requested:  prune {layer_name}  (role={role})
  expected:   all parameters matching `{layer_name}.*` consumed only within
              the layer's computational closure
  found:      parameter `{param_name}` (VarId {param_var_id}) is consumed by
              op_id={external_consumer_op_id} ({external_op_kind}), which is
              outside the closure for {layer_name}
```

Variant: `PruneRefusal::CrossLayerParam`.

### §3.2 No residual Add in closure

Trigger: the closure is non-empty but contains zero ops matching the residual pattern.

```text
prune: layer is not residual-structured (no boundary Add found).
  requested:  prune {layer_name}  (role={role})
  expected:   exactly one op in the closure matching Add(h_before, block_output)
              with block_output ∈ closure ∧ block_output single-consumer
  found:      closure has {closure_size} ops but zero ops match the residual
              pattern; the layer appears to be non-residual (SSM / Mamba /
              non-standard architecture)
```

Variant: `PruneRefusal::NoResidualAdd`.

### §3.3 Parallel residual branches

Trigger: the closure contains ≥ 2 Add ops each pattern-matching `Add(h_before_i, block_output_i)` against distinct `h_before_i` (parallel residuals, e.g., Parallel Transformers).

```text
prune: layer has parallel residual branches (not supported in v1).
  requested:  prune {layer_name}  (role={role})
  expected:   exactly one residual boundary Add
  found:      {k} residual Adds detected at ops {add_op_ids}; each appears to
              be a separate residual branch (distinct h_before values). Parallel
              residual pruning requires branch-by-branch semantics not yet
              specified.
```

Variant: `PruneRefusal::ParallelResidualBranches`.

### §3.4 Ambiguous pattern match

Trigger: the closure contains ≥ 2 Add ops each pattern-matching `Add(h_before, block_output)` against the SAME `h_before` (architecturally ambiguous boundary; e.g., pre-norm and post-norm residuals both present).

```text
prune: layer has multiple candidate residual boundaries (pattern-match ambiguous).
  requested:  prune {layer_name}  (role={role})
  expected:   exactly one op matching the residual pattern
  found:      {k} candidate Adds match the residual pattern against the same
              h_before (VarId {h_before_var_id}): ops {candidate_add_op_ids}.
              Boundary disambiguation requires architecture-specific rules not
              yet specified.
```

Variant: `PruneRefusal::AmbiguousPatternMatch`.

### §3.5 Empty closure

Trigger: zero VarIds match the `{layer_name}.` prefix in `var_names`.

```text
prune: no parameters match the requested layer prefix.
  requested:  prune {layer_name}  (role={role})
  expected:   at least one parameter VarId with var_name starting
              with `{layer_name}.`
  found:      zero matching parameters in the WeightMap. Check layer name /
              index; the requested layer does not exist in the compiled model.
```

Variant: `PruneRefusal::EmptyClosure`.

### §3.6 Whole-block prune unsupported (v1)

Trigger: `AppliedLayer.coarse == CoarseDecision::Prune` with `layer_role == LayerRole::Block`.

```text
prune: whole-block pruning (LayerRole::Block) is not supported in v1.
  requested:  prune {layer_name}  (role=Block)
  supported:  prune {layer_name}.attn  (role=Attention)
              prune {layer_name}.ffn   (role=Ffn)
  workaround: emit two sub-block prune decisions for layer {N}; their combined
              effect is semantically equivalent to whole-block prune in standard
              pre-norm transformer architectures (NOT equivalent for post-norm,
              parallel, or scaled-residual architectures).
  planned:    whole-block prune tracked for v2 (chain-collapse transformation).
```

Variant: `PruneRefusal::WholeBlockUnsupported`.

### §3.7 Conflicting prune decisions (defensive, v1)

Trigger: two decisions in the same plan produce conflicting VarId aliasing (both try to alias the same `h_after` to different sources, or two closures overlap in a way that breaks single-consumer invariants).

```text
prune: two prune decisions in the same plan conflict.
  requested:  prune {layer_a} AND prune {layer_b} in the same plan
  expected:   each rewrite's closure and VarId aliasing is disjoint from every
              other rewrite's
  found:      {conflict_reason}
```

Variant: `PruneRefusal::ConflictingPruneDecisions`. Defensive — v1's supported scope (disjoint sub-block prefixes) cannot produce conflicts, but the variant exists for robustness against future plan types.

---

## §4. Pipeline position

The prune rewrite runs **before `wrga_prune::run()`** in `stmt.rs` (site: ~line 4520, just before the existing call). Rationale:

- `wrga_prune` computes `backward_live` by walking the forward Wengert list. If forward ops are deleted after `backward_live` is computed, the set has stale entries referring to non-existent ops.
- Running WGGO Prune first reduces the forward. WRGA then computes `backward_live` on the already-reduced forward, naturally correct, no post-hoc patching of `backward_live` needed.
- Source-AD's adjoint generation runs after `wrga_prune`, so it also sees the reduced forward.
- The rewrite runs AFTER any WRGA adapter-injection pass; adapter ops on a pruned layer are included in the closure and deleted together with the layer's base-weight ops.

Ordering invariant: WGGO Prune's effect on the Wengert list is strictly subtractive (ops deleted). WRGA Prune's effect is strictly additive to `backward_live` filtering (VarIds excluded from adjoint computation). Neither pass mutates the other's concerns — WRGA doesn't re-add forward ops; WGGO doesn't touch `backward_live` directly.

---

## §5. API + module layout

### §5.1 New module: `crates/nsl-codegen/src/wggo_prune.rs`

Module doc-comment opens with disambiguation + pipeline context:

```rust
//! Layer-level Wengert rewriting driven by WGGO `CoarseDecision::Prune`.
//!
//! Distinct from `wrga_prune.rs`, which handles parameter-level `backward_live`
//! filtering for frozen adapter weights. This module removes whole layer
//! computations from the forward; `wrga_prune` then computes `backward_live`
//! on the already-reduced forward.
//!
//! Pipeline position: runs before `wrga_prune::run()` in `stmt.rs`, and
//! therefore before source-AD's adjoint generation. The rewrite produces
//! the final forward Wengert that both WRGA Prune and source-AD will consume.
//!
//! Design principle: this module refuses transformations when preconditions
//! aren't met; it does not fall back to weaker transformations with different
//! semantics. See memory/feedback_transformation_precondition_refusal.md for
//! the generalized rule.
```

### §5.2 Public API

```rust
pub struct PruneRewriteResult {
    pub rewrites: Vec<PruneRewrite>,
    pub refusals: Vec<PruneRefusal>,
    pub pruned_forward_var_ids: BTreeSet<VarId>,
    pub ops_deleted: usize,
}

pub struct PruneRewrite {
    pub layer_name: String,
    pub layer_role: LayerRole,
    pub h_before_var: VarId,
    pub h_after_var: VarId,
    pub residual_add_op: OpId,
    pub closure_ops: Vec<OpId>,
}

pub enum PruneRefusal {
    CrossLayerParam {
        layer_name: String,
        layer_role: LayerRole,
        param_name: String,
        param_var: VarId,
        external_consumer: OpId,
        external_op_kind: String,
    },
    NoResidualAdd {
        layer_name: String,
        layer_role: LayerRole,
        closure_size: usize,
    },
    ParallelResidualBranches {
        layer_name: String,
        layer_role: LayerRole,
        add_ops: Vec<OpId>,
    },
    AmbiguousPatternMatch {
        layer_name: String,
        layer_role: LayerRole,
        h_before_var: VarId,
        candidate_adds: Vec<OpId>,
    },
    EmptyClosure {
        layer_name: String,
        layer_role: LayerRole,
        prefix: String,
    },
    WholeBlockUnsupported {
        layer_name: String,
    },
    ConflictingPruneDecisions {
        decision_a: String,
        decision_b: String,
        reason: String,
    },
}

/// Entry point. Dry-run-then-commit: validates all decisions first; applies
/// mutations only if all pass. On refusal, `wengert` is unchanged.
pub fn run(
    wengert: &mut WengertList,
    applied_plan: &AppliedPlan,
    weight_map: &WeightMap,
) -> PruneRewriteResult;
```

### §5.3 Dry-run-then-commit semantics

`run()` has three internal phases:

1. **Phase 1 — Validate.** For each `CoarseDecision::Prune` in the plan, compute closure + pattern-match without mutating `wengert`. Produce either a `PruneRewritePlan` (internal, opaque) or a `PruneRefusal`.
2. **Phase 2 — Early-return on refusal.** If any refusal was produced, return immediately with `rewrites: Vec::new()` and all refusals. `wengert` is guaranteed unchanged.
3. **Phase 3 — Commit.** Only reached if all Phase 1 results were plans. Apply every plan's mutation sequentially: delete closure ops, repoint residual Add consumers, delete the Add. Emit one `PruneRewrite` per plan.

Contract: the caller sees either "all rewrites applied, no refusals" or "some refusals, wengert untouched." Never "some rewrites applied, some refusals, partial state." This avoids the "fix one refusal, recompile, see the next" whack-a-mole UX and gives `wengert` a clean transactional semantics.

Phase 2 may also surface cross-plan conflicts (via `ConflictingPruneDecisions`) before mutation happens.

### §5.4 Caller integration (`stmt.rs`)

Insertion site is just before the existing `wrga_prune::run()` call. Caller responsibility:

1. Call `wggo_prune::run(&mut wengert, &applied_plan, &weight_map)`.
2. If `result.refusals.is_empty()` is false, emit each refusal's three-part error via the diagnostics infrastructure and fail compilation. All refusals emitted in one pass — no "fix-one-and-recompile" loop.
3. Otherwise, for each `rewrite` in `result.rewrites`, emit the success-path stderr line (§6.1).
4. Pass `result.pruned_forward_var_ids` to `wrga_prune::run()` via whatever channel is appropriate (add to the existing plan struct or propagate through a new parameter; implementation-plan-level decision).

### §5.5 Modifications to `wggo_overrides.rs`

- Rename `OverrideRejectReason::PruneNotImplemented` → `OverrideRejectReason::WholeBlockPruneNotImplemented`. (Grep confirmed internal-only: 5 hits, all inside `nsl-codegen`; no serialized fixtures.)
- Narrow `collect_prune_diagnostics` to iterate only `LayerRole::Block` prune decisions. Sub-block prune decisions are now handled by `wggo_prune.rs` and do not flow through `collect_prune_diagnostics`.
- Rename `prune_not_implemented_reason()` → `whole_block_prune_not_implemented_reason()`. The returned string stays `"ir_rewrite_not_implemented"` to preserve the PR #102 diagnostic contract for `LayerRole::Block`.

**Stderr-contract preservation (load-bearing):** the `reason=ir_rewrite_not_implemented` stderr substring is emitted via an explicit string (the return value of `whole_block_prune_not_implemented_reason()` or equivalent), NOT via a debug-formatted enum variant name. Any `format!("reason={:?}", variant)` pattern in the existing PR #102 emission site would break the contract on rename by printing `WholeBlockPruneNotImplemented` instead of `ir_rewrite_not_implemented`. The rename changes the variant's Rust name but not the stderr-contract string. Layer 4 tests (§7.6) verify this by snapshotting the stderr text for the `LayerRole::Block` refusal path.

### §5.6 Mutation model

`run()` mutates `wengert` in place internally (Phase 3). The external contract is transactional: either all mutations land or none do. Internal in-place mutation is chosen over functional return for performance (the list is large, clone cost dominates) and for consistency with the existing `eliminate_by_backward_live` pattern.

---

## §6. Diagnostic contract

### §6.1 Success-path stderr format

One line per successful rewrite, emitted by `stmt.rs` after `wggo_prune::run()` returns with no refusals:

```text
[prune] layer={layer_index} name={layer_name} role={role} applied=true closure_size={K} ops_deleted={K} residual_add_op={op_id}
```

Separator convention: `key=value` throughout (no colons). `closure_size` and `ops_deleted` are conceptually distinct (closure size is "ops identified as belonging to the layer"; ops_deleted is "ops actually removed from wengert"); in v1 they are equal because the rewrite deletes every closure op. Recording both as separate fields makes the diagnostic self-documenting for future versions that may do partial prune.

Example:

```text
[prune] layer=7 name=blocks.7.attn role=Attention applied=true closure_size=14 ops_deleted=14 residual_add_op=483
```

### §6.2 Refusal-path stderr format

Multi-line three-part error per refusal, format fixed in §3. Each refusal emission is terminated by a blank line so multiple refusals are visually separable.

PR #102's `reason=ir_rewrite_not_implemented` string survives ONLY for refusal §3.6 (`WholeBlockUnsupported`). All other refusals use the new three-part format.

### §6.3 Structured diagnostic envelope

Alongside the stderr text, each refusal produces a structured diagnostic via the existing `DiagnosticCode` enum (see existing uses in `stmt.rs`). New codes:

```rust
DiagnosticCode::PruneCrossLayerParam
DiagnosticCode::PruneNoResidualAdd
DiagnosticCode::PruneParallelResidualBranches
DiagnosticCode::PruneAmbiguousPatternMatch
DiagnosticCode::PruneEmptyClosure
DiagnosticCode::PruneWholeBlockUnsupported
DiagnosticCode::PruneConflictingDecisions
```

Tests (§7, Layer 4) assert both the stderr text snapshot AND the `DiagnosticCode` — disjoint drift detection. Text-only tests break on diagnostic-infrastructure refactors; code-only tests miss text regressions; both together catch either.

---

## §7. Test discipline

Four-layer pyramid inheriting the WRGA B.3.2 test-discipline convention (skeleton-snapshot / unit / integration-numerical / e2e-launch-counter, adapted to this pass's surface). All tests are CPU-eligible — no CUDA required — so CI cost is bounded.

### §7.1 Test fixture: `prune_rewrite_toy.nsl`

Location: `crates/nsl-codegen/tests/fixtures/prune_rewrite_toy.nsl`.

Structure: 4-block pre-norm transformer with distinct `attn` and `ffn` sub-blocks per block. Dimensions kept tiny: `d_model=32, d_ffn=64, 4 heads × head_dim=8, vocab=64, seq=16, batch=1`. Purpose-built for structural-rewrite testing — not numerical-accuracy at scale.

Four-block count (not two) enables first / middle / last / multi-prune test cases:

- Prune first block's sub-blocks (`blocks.0.attn`, `blocks.0.ffn`)
- Prune last block's sub-blocks (`blocks.3.attn`, `blocks.3.ffn`)
- Prune middle block's sub-blocks (`blocks.1.attn`, `blocks.1.ffn`, `blocks.2.attn`, `blocks.2.ffn`)
- Prune both sub-blocks of one layer (`blocks.1.attn` AND `blocks.1.ffn`)
- Prune across layers (`blocks.1.ffn` AND `blocks.2.attn`)

### §7.2 Reference fixtures (reference-as-separate-source)

For each integration test case, a matching reference `.nsl` source lives alongside the fixture:

```text
prune_rewrite_toy.nsl                              # 4-block baseline
prune_rewrite_toy_ref_blocks_0_attn.nsl            # blocks.0.attn replaced by identity
prune_rewrite_toy_ref_blocks_0_ffn.nsl
prune_rewrite_toy_ref_blocks_3_attn.nsl
... (one per integration test case)
```

Each reference hand-codes the pruned sub-block as `h = h` (identity) instead of the residual sub-block. Both baseline and reference compile through the same codegen; equivalence tests compare their outputs bit-exactly.

Drift risk: if `prune_rewrite_toy.nsl` changes (e.g., dimension tweak), every reference variant needs a matching edit. Both files are snapshot-tested at Layer 1, so drift is caught in review via adjacent snapshot diffs.

### §7.3 Layer 1 — IR skeleton snapshots (insta)

One snapshot per (block_position × sub_block_role) combination × baseline: 4 blocks × 2 roles × (before + after) = **16 snapshots**. Additional snapshots for multi-prune cases (~4 more). Total: ~20 snapshots.

**Snapshot pretty-print format (VarId-number-stable):** snapshots use symbolic names where available (e.g., `%param:blocks.7.attn.wq`) and topologically-ordered per-snapshot placeholders (`%t1`, `%t2`, …) for anonymous intermediate VarIds. Global VarId numbers from the allocator are NOT printed. Rationale: snapshots that embed global VarId numbers become brittle against any refactor that perturbs VarId allocation (adding ops to the extractor, renumbering, etc.) — every mechanical renumbering invalidates all 20 snapshots and hides real behavioral drift in the noise. Symbolic + topologically-ordered placeholders stay stable through renumbering refactors; snapshots change only when the structural rewrite's behavior changes.

Each snapshot captures:

- The Wengert list after `wggo_prune::run()` (or baseline, pre-rewrite) in the stable format above
- `PruneRewriteResult` contents serialized as pretty-print (`PruneRewrite` entries printed by symbolic `layer_name` + `layer_role` + `closure_ops` count; raw OpId/VarId numbers elided or replaced with placeholders)
- VarId aliasing evidence (the `h_after_var`'s absence from the post-rewrite list, or its presence with a distinct predecessor, named symbolically)

### §7.4 Layer 2 — Closure + pattern-match unit tests

Unit tests operate on hand-constructed `WengertList` fixtures — no NSL source compilation. Target: the `wggo_prune::plan_rewrite` function (internal Phase-1 validator) directly.

Positive tests:

- `closure_captures_transitive_compute_ops` — synthetic 3-op block; closure includes all 3.
- `closure_stops_at_residual_add` — residual Add is NOT in the closure; its inputs (h_before, block_output) are correctly identified.
- `closure_includes_parameter_producer_ops` — parameter loader ops are in the closure alongside consumers.

Negative tests (one per refusal case, using synthetic Wengert lists designed to trigger each):

- `cross_layer_param_refusal` — layer-7 param consumed by an op outside the closure → `PruneRefusal::CrossLayerParam` with correct fields.
- `no_residual_add_refusal` — closure of length 3 with zero Adds → `PruneRefusal::NoResidualAdd`.
- `parallel_residuals_refusal` — closure contains two Adds with distinct h_before → `PruneRefusal::ParallelResidualBranches`.
- `ambiguous_pattern_match_refusal` — closure contains two Adds with same h_before → `PruneRefusal::AmbiguousPatternMatch`.
- `empty_closure_refusal` — no VarIds match prefix → `PruneRefusal::EmptyClosure`.
- `whole_block_refusal_from_planner` — `LayerRole::Block` in the plan → `PruneRefusal::WholeBlockUnsupported`.
- `conflicting_decisions_refusal` — two decisions whose closures would alias conflicting VarIds → `PruneRefusal::ConflictingPruneDecisions`.

Refusal-message format tests — one per refusal variant — assert the exact stderr text matches §3's template, using string comparison (not snapshot; these are short and format-critical).

### §7.5 Layer 3 — Numerical equivalence (integration)

**Scope:** Layer 3 tests cover supported prune decisions end-to-end. Refusal cases that require non-standard architectures (§3.1 cross-layer param, §3.2 no residual Add, §3.3 parallel residuals, §3.4 ambiguous pattern) are covered exhaustively at Layer 2 via hand-constructed synthetic Wengert lists; Layer 3 only verifies the integration-path refusals reachable from the supported fixture — specifically §3.5 empty-closure (via a bogus `blocks.99.attn` layer name in the plan) and §3.6 whole-block (via planner decision manipulation forcing `LayerRole::Block`). This explicitly scopes Layer 3 and prevents the implementer from attempting to construct non-standard-architecture `.nsl` fixtures that don't readily express in NSL source.

Compile `prune_rewrite_toy.nsl` with and without each prune decision; compile the matching reference `.nsl`; run forward at a fixed seed (deterministic init per existing NSL conventions); compare outputs.

**Equivalence contract: bit-exact.** Not tolerance-based. Rationale: the rewrite is purely structural. Ops that remain execute identical arithmetic on identical inputs; any non-bit-exact result is a real bug (missed consumer, wrong ops deleted, VarId collision, accidental operand swap in a commutative op, accidental broadcast). Tolerance-based testing would hide exactly the bug classes this test exists to catch.

Assertion shape:

```rust
let out_pruned = run_forward(&compile(&toy, &plan_prune_blocks_1_attn));
let out_ref    = run_forward(&compile(&toy_ref_blocks_1_attn, &plan_no_prune));
assert_eq!(out_pruned.bytes(), out_ref.bytes(),
    "prune rewrite should produce bit-exact output vs identity-aliased reference");
```

Backward-equivalence assertion: compile a full train-block with the prune decision, run one SGD step, collect gradients, compare.

**Strong adjoint assertion:** the test also verifies that pruned VarIds are absent from the adjoint tape as both producers and consumers, and that the backward op count is strictly reduced versus the no-prune baseline:

```rust
let pruned_fwd_var_ids = result.pruned_forward_var_ids;
let adjoint_tape_pruned   = compile_adjoint(&toy, &plan_prune);
let adjoint_tape_baseline = compile_adjoint(&toy, &plan_no_prune);

assert!(adjoint_tape_pruned.ops.len() < adjoint_tape_baseline.ops.len(),
    "prune should reduce backward op count; got {} vs {}",
    adjoint_tape_pruned.ops.len(), adjoint_tape_baseline.ops.len());

for var_id in pruned_fwd_var_ids {
    assert!(!adjoint_tape_pruned.references_var(var_id),
        "pruned VarId {:?} still referenced in backward tape", var_id);
}
```

Catches the "dead backward ops referencing pruned forward VarIds" bug class — defeats the memory win that's a major motivation for prune.

### §7.6 Layer 4 — E2E diagnostic contract (stderr snapshot + structural code assertion)

Each test captures compilation stderr and asserts:

1. **Primary — text snapshot:** the captured stderr matches an insta snapshot. One snapshot per success case and per refusal case. Reviewer sees the user-facing message at review time.
2. **Secondary — `DiagnosticCode` assertion:** the compilation's structured diagnostics include the expected `DiagnosticCode` variant. Robust to text refactors (adds a source-location prefix, reorders lines, etc.); catches behavioral-vs-textual drift.

Two assertions per refusal case means disjoint failure modes — text drift and behavioral drift — are distinguishable during review.

Success-case snapshots also assert the structured absence of any refusal diagnostic code, guarding against "rewrite silently refused but stderr didn't show it" bugs.

### §7.7 CI cost

All tests run on CPU. No CUDA, no GPU. Approximate timing per the 4-block toy: each forward is sub-millisecond; each full compile is < 1s. Total test-pass time estimate: < 10s across all four layers. Safe to run on every PR.

---

## §8. Design principles cited

This spec's discipline inherits from three prior institutional rules:

1. **Transformation-precondition-refusal** (this spec's third instance; memory: `feedback_transformation_precondition_refusal.md`). Compiler transformations refuse when preconditions aren't met; they do not fall back to weaker transformations with different semantics. Prior instances: CPDT one-`@cpdt`-per-program enforcement, PCA Tier A convention-match rule.

2. **Positive scope framing (B.5 family)** — spec claims are direct ("v1 supports pre-norm transformer sub-blocks with residual structure") rather than implied through negative space ("works on anything that doesn't hit refusals"). Users with non-standard architectures see "this isn't the supported scope" up front.

3. **Computed property over field-based state** — closure ownership is computed from current IR state, not stored on `WengertOp` as a field. Computed properties can't drift; field-based properties can. Analogous to the CPDT scorer unification decision (iii over i/ii).

The module doc-comment cites (1) explicitly. The spec's scope section (§ Scope + §1.2) embodies (2). §2.1's rejection of field-based approaches embodies (3).

---

## §9. Out-of-scope follow-ups (v2 and beyond)

Tracked here so future work can plan against v1's contract.

- **Whole-block prune (`LayerRole::Block`) via chain-collapse.** Requires new preconditions (single h-stream thread, no op consumes intermediate h_i except the next residual Add, all residuals pattern-match consistently). Each needs its own refusal case with its own error message. A chain-collapse design deserves its own spec. Tracked in refusal §3.6's `planned` line.

- **Scaled residuals (DeepNorm etc.).** Pattern `h + alpha * block(h)`. v1 refuses via §3.4 (pattern doesn't match single-factor Add); v2 may relax the pattern-match to recognize the scaled form and delete the multiplier alongside the block ops.

- **Fused Add / AddNorm.** If a downstream pass fuses `h + block(h)` with a subsequent norm into a single op before WGGO Prune runs, the pattern-match fails. v1 runs early enough to avoid this; v2 may need to either (a) recognize fused forms or (b) enforce an ordering constraint in the pass pipeline.

- **Partial prune.** v1 deletes every op in the closure. A future partial-prune mode might delete only some (e.g., reducing FFN width while keeping structure). That mode would have `closure_size > ops_deleted`; the diagnostic format already separates the two fields for this future evolution.

- **Non-residual architectures.** SSM, Mamba, non-standard MoE variants. Any future support would be a new transformation (not an extension of v1's residual-identity semantic). Currently the planner can emit Prune for these, and v1 refuses via §3.2 (no residual Add) or §3.5 (no matching params).

---

## §10. Commit structure (informative)

Implementation PR is expected to land in a small number of commits:

- **Commit A — Module scaffolding.** New `wggo_prune.rs` with public API types (`PruneRewriteResult`, `PruneRewrite`, `PruneRefusal`) and `run()` stubbed to return an empty result. `stmt.rs` wired to call it. `wggo_overrides.rs` renamed as per §5.5. Tests at Layer 2 for the seven refusal variants (hand-constructed fixtures, verifying enum-variant dispatch). Compiles, snapshots empty, baseline preserved.
- **Commit B — Closure + pattern-match.** Phase 1 validator: closure computation, pattern-match, refusal-enum population. All Layer 2 tests (positive + negative) green.
- **Commit C — Mutation commit.** Phase 3: op deletion, VarId repointing, residual Add removal. Layer 1 snapshots generated. Layer 3 integration numerical tests (bit-exact) green. `pruned_forward_var_ids` wired through to `wrga_prune::run()`.
- **Commit D — Diagnostics.** Success-path stderr + structured `DiagnosticCode` emission. Layer 4 tests (text snapshot + code assertion) green. `WholeBlockPruneNotImplemented` path preserved for `LayerRole::Block`.

Informative only — implementer has latitude to split differently; what matters is that each commit passes its corresponding test layer.

---

## §11. Success criteria (merge gate)

1. All four test layers green under `cargo test -p nsl-codegen`.
2. Layer 3 integration tests pass **bit-exactly** (no tolerance). Any drift is a bug.
3. Each of the seven refusal variants produces its three-part error, verified by both text snapshot and `DiagnosticCode` assertion.
4. `wggo_prune::run()` returns `rewrites: Vec::new()` and untouched `wengert` whenever ANY refusal occurs (dry-run-then-commit invariant).
5. Pipeline position verified: WGGO Prune runs before `wrga_prune::run()`; `backward_live` is computed on the reduced forward; source-AD sees the reduced forward.
6. PR #102's `[prune] … reason=ir_rewrite_not_implemented` stderr format survives for `LayerRole::Block` decisions only.
7. Manual smoke check: a plan with mixed supported + unsupported prune decisions fails compilation with ALL refusals emitted in one pass — no fix-one-recompile loop.
