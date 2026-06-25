# Checkpoint v1 (@checkpoint policy="full") — Honest Deferral Log

Paper §5.3 Checkpointing-Aware Backward, cycle-10 v1 landing.

This document enumerates what cycle 10 ships, what it does NOT ship, and the
acceptance criteria for the v2 follow-on. Read this before claiming `@checkpoint`
is "done" — v1 is intentionally a structural-and-refusal landing, not a
GPU-validated functional landing.

## Cycle 10 first-pass functional gap closure (R0)

Cycle 10's first pass landed the §5.3 API surface (carriers, semantic
plumbing, dispatch fork, SMEM validator extension, the namespace_suffix
refactor, and a R3/R7/R9/R10/R8.1 refusal cascade) and shipped the
recompute path as a **PTX-comment-only post-injection** on top of the
existing tier-B backward emitter. Reviewer 1 (correctness) found three
silent-correctness gaps in that approach:

1. **`kv_load` was not skipped**: the recompute path still ran the
   materialized-saves `emit_k_suffixed` / `emit_v_suffixed` loads, so
   `policy="full"` PTX was structurally identical to the no-decorator
   baseline plus comments.
2. **`WengertOp.checkpointed` had zero downstream consumers**: the
   per-op stamping from Task 5 landed but never reached codegen.
3. **G3 was tautological**: it asserted PTX-comment substrings that
   were synthesized inside the same comments, so the gate passed
   trivially without proving any functional behavior.

Together these violated cycle-5 invariant `feedback_deferral_must_refuse`:
API-surface sprints that defer downstream emission MUST add a hard refusal
at codegen time, not log disclosure plus a silently-no-op emission path.

**Phase F closure** (this commit):

- Adds **R0** — a codegen-time refusal that fires at the top of
  `synthesize_backward_with_recompute` for any `policy="full"` config.
  Substring: `@checkpoint(policy="full") functional recompute not yet
  wired in v1: ships API surface + refusal cascade only. kv_load
  substitution + SMEM-base routing deferred to follow-on cycle behind
  GPU validation gate.`
- Removes the comment-only post-injection block entirely. With R0 at
  the top of the function, the post-injection was dead code.
- Drops the structurally-unreachable R8.1 verified smoke-set carve-out
  (`static_seq_len.unwrap_or(0)` always evaluated to 0 in production
  because `CshaExtras::level1` does not set `static_seq_len`). R8.1 is
  now honest about being unconditional in v1. The smoke-set carve-out
  is deferred to v2.
- Replaces G3 with `g3_policy_full_refuses_with_r0_substring`, a honest
  refusal-reachability gate that fails if R0 is removed without a
  functional replacement.
- Adds R0 to the G2 substring sweep (6 refusal tests instead of 5; in
  v1 the original R3/R7/R9/R10/R8.1 tests assert the R0 shadow
  substring because R0 fires before them).

R0 is lifted in cycle 11 when the functional substitution lands. See
"v2 acceptance criteria" under G4 below for the exact lift contract.

**Honesty note on `WengertOp.checkpointed` stamping**: the Task 5
plumbing remains — it stamps Wengert ops as `checkpointed: true` when
they fall inside a `@checkpoint`-decorated boundary chain. In v1 this
stamping is **decorative**: no downstream consumer reads it. Cycle 11
wires the consumer (codegen reads the stamp to decide whether to emit
the recompute path vs. the materialized-saves path on a per-op basis).

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

**v2 acceptance criteria** (cycle 11 — lifts R0):

1. Lift R0 + wire `kv_load` skip + SMEM-base routing + GPU numerical
   validation (atol=5e-4 rtol=5e-3 at hd in {64,128}, S in {512, 2048,
   4096}). Specifically: suppress `kv_load::emit_k_suffixed` /
   `emit_v_suffixed` when `config.checkpoint.is_some()`, and route
   `emit_prologue_recompute` output into `%k_smem_base` / `%v_smem_base`
   so downstream `ds_compute` / `dqdk_accum` / `dv_accum` see the
   recomputed projections.
2. Land the numerical-equivalence test on a CUDA-equipped host that
   compares dQ / dK / dV against the materialized-saves baseline at the
   tolerances above.
3. Replace G3 (currently `g3_policy_full_refuses_with_r0_substring`)
   with the numerical-equivalence assertion driven from the GPU job.
4. Add a CI label that gates v2 promotion on the GPU job passing.

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

## Cycle 11 functional substitution landing (R0 stays)

Cycle 11 wires the kv_load -> emit_kv_recompute substitution behind R0:

- Test-only bypass via bypass_r0_for_testing() on CheckpointExtras (cfg-gated, no production attack surface)
- kv_load::emit_k_suffixed/emit_v_suffixed skipped when config.checkpoint.is_some()
- emit_kv_recompute writes K_proj/V_proj into %k_smem_base/%v_smem_base via x_raw_ptr + W_k/W_v + RoPE saves
- Structural probes G3a (kv_load skip evidence), G3b (recompute label), G3c (SMEM ordering), G3d (no-decorator byte-identity)

Cycle 12 will lift R0 + run G4 GPU numerical validation at (hd, S) in {(64, 512), (64, 2048), (128, 2048), (128, 4096)} via csha_cuda_backward.rs harness.

