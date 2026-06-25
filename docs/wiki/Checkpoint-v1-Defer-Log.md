# Checkpoint v1 (@checkpoint policy="full") — Honest Deferral Log

Paper §5.3 Checkpointing-Aware Backward, cycle-10 v1 landing.

This document enumerates what cycle 10 ships, what it does NOT ship, and the
acceptance criteria for the v2 follow-on. Read this before claiming `@checkpoint`
is "done" — v1 is intentionally a structural-and-refusal landing, not a
GPU-validated functional landing.

## What v1 ships (cycle 10)

- `CheckpointPolicy::Full` enum + `CheckpointExtras` carrier on
  `FlashAttentionConfig` (Task 1).
- `PrimalOp::PrologueRecompute { subgraph_id }` + `SubgraphId(u32)` newtype
  (Task 2).
- Semantic `@checkpoint(policy="full")` kwarg parsing + R0 deprecation warning
  for bare `@checkpoint` (Task 3).
- `EffectChecker::paged_kv_models` cross-scope tracker for R9 (Task 4).
- `WengertExtractor::with_checkpoint_policies` + transitive stamping +
  `PrologueRecompute` marker emission (Task 5).
- Wire-up: semantic `analyze_with_imports` -> `ModuleData` -> `CompileOptions`
  -> `WengertExtractor::with_checkpoint_policies` (Task 6).
- SMEM validator extension via `recompute_extra_bytes` + R5 refusal (Task 7).
- `emit_prologue_namespaced` + `emit_prologue_recompute` namespace-suffix
  refactor (Task 8).
- Backward dispatch fork at `synthesize_backward_with_tier` that routes
  `config.checkpoint.is_some()` calls to `synthesize_backward_with_recompute`
  (Task 9).
- Five refusals (R3 / R7 / R9 / R10 / R8.1) with the documented substrings
  (Task 9).
- Four test gates (G1 / G2 / G3 / G6) all green (Task 9).

## What v1 explicitly defers

### G4: GPU numerical equivalence

The cycle-9 spec required a Gate 4 sweep comparing recompute-backward
gradients against the materialized-saves baseline for a config sweep
on a live CUDA device. We have **no GPU access in the v1 cycle**;
this is documented honestly here rather than fabricated.

**v1 substitute**: the G3 structural probe asserts the diagnostic comment
plus the namespace-suffix evidence appears in the emitted PTX. This proves
the dispatch fork lands and the Task-8 wiring is exercised. It does NOT
prove the recompute output is bit-identical to the materialized-saves
backward.

**Implementation gap that ALSO contributes to deferral**: cycle 10
ships the recompute injection as PTX comments + per-Q-iter label
markers. True functional substitution (skip the `kv_load::emit_k_suffixed`
/ `emit_v_suffixed` calls, route the recompute helper's output into
`%k_smem_base` / `%v_smem_base` so downstream `ds_compute` /
`dqdk_accum` / `dv_accum` see the recomputed projections) is a
follow-on refactor that depends on the recompute helper itself being
extended to write SMEM in the same layout the kv_load emitters do.

**v2 acceptance criteria**:

1. Replace the comment-only injection in
   `synthesize_backward_with_recompute` with a true emission path that
   suppresses the kv_load calls under `checkpoint.is_some()` and
   substitutes `emit_prologue_recompute` calls that write into the
   same SMEM bases.
2. Land a numerical-equivalence test on a CUDA-equipped host that
   compares dQ / dK / dV against the materialized-saves baseline at
   `head_dim in {64, 128}` and `seq_len in {2048, 4096}`. Tolerances:
   max abs err < 1e-4 for fp32, < 5e-3 for fp16.
3. Add a CI label that gates v2 promotion on the GPU job passing.

### G5: paper §6.3 exact diagnostic string

Cycle 9 specified an exact diagnostic comment string for the
documentation-tooling pipeline. Cycle 10 replaces this with the G3
substring assertion. Same machinery, same correctness signal, lower
maintenance cost; promote to exact-string match when the diagnostic
format stabilizes.

### R6: per-head mixed precision

The §3.3 per-head dtype field on `FlashAttentionConfig` is not landed.
R6 is INERT in v1: there is no field for it to read from. Forward-compat
refusal hook is left as a `TODO` comment at the dispatch fork.

### R8 (original): sinks-v0 refusal

Sinks-v0 was already closed by cycles 5 / 7 / 8 refusals. R8 in v1 is a
no-op; the only sinks-related refusal that fires today is R8.1 (sinks-v2
+ checkpoint composition).

### R8.1 verified smoke-set carve-out

The cycle-9 spec specified an "unless (hd, S) is in verified smoke-set"
carve-out for R8.1. The cycle-7 backward sinks refusal at
`synthesize_backward_with_tier_b` structurally blocks any
`num_sink_tokens > 0` config from reaching the recompute path. v1
therefore refuses R8.1 unconditionally with the checkpoint-specific
substring; the smoke-set carve-out becomes meaningful only when cycle
11 lifts the lower-level sinks refusal.

### R9 full call-graph propagation

Cycle 10 ships R9 as a **wire-up-time** flag on `CheckpointExtras`
(`paged_kv_collision: bool`). When the wire-up site (`loader.rs`)
sees a `@checkpoint` fn whose enclosing model is in
`EffectChecker::paged_kv_models`, it sets the flag. Multi-call-graph
resolution (e.g., a `@checkpoint` fn called transitively from a
`@paged_kv` model layer) is deferred to v4.

**v4 acceptance criteria**:

1. Plumb call-graph reachability from `@paged_kv` models into the
   semantic layer.
2. Set `paged_kv_collision = true` on the carrier for any
   `@checkpoint` fn reachable from a `@paged_kv` model.
3. Add a multi-call-graph test that catches the transitive case.

## Operational consequences

Until the v2 GPU validation lands:

- DO NOT promote `policy="full"` to user-facing docs as a "supported"
  feature. Cycle 10 ships the API surface + refusals; we do not yet
  ship verified gradients.
- DO promote it as "experimental / unstable" if downstream agents need
  to exercise the path for stub-level testing.
- DO read R3 / R7 / R9 / R10 / R8.1 refusal substrings before opening
  a bug — most "checkpoint doesn't work" reports will be a refusal
  firing correctly.

## Gate inventory at v1 landing

| Gate | Status | Notes |
|---|---|---|
| G1 byte-identity (no-decorator) | GREEN | 25/25 fa_v2_snapshots |
| G2 refusal substring coverage   | GREEN | 5/5 in checkpoint_v1_integration |
| G3 structural recompute probe   | GREEN | diagnostic + namespace-suffix evidence |
| G4 GPU numerical equivalence    | DEFERRED | requires CUDA host; v2 |
| G5 paper §6.3 exact string      | REPLACED | by G3 substring |
| G6 sibling-leak (EffectChecker) | GREEN | empty map for non-@checkpoint module |

Cross-reference: see cycle-10 spec corrections appendix at
`docs/superpowers/specs/2026-06-24-csha-checkpointing-aware-backward-design.md`
for the W1-W17 fabrication audit and the authoritative cut sequence.
