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

## Cycle 13 K/V save-suppression structural landing

Cycle 13 wires the forward-side gate for K_proj + V_proj save-suppression
under `@checkpoint(policy="full")`. This is the first cycle that
materially reduces forward HBM write traffic when the checkpoint
decorator is active.

- Mechanism shipped: `should_emit_kv_save(cfg)` helper in
  `flash_attention_v2/phases/forward/csha_hooks.rs` + per-tensor gate
  inside `emit_save_activations_subset`. Single source of truth; called
  exactly once per K and V emit-site.
- Suppresses K_proj + V_proj saves (each ~8.39 MB at hd=128, S=2048,
  B=1, h=16 reference shape from Phase A facet 1 Table D). Total
  cycle-13 structural reduction: **16.78 MB of 50.60 MB = 33% of total**.
- Paper §6.3 49% headline awaits Q_proj suppression (cycle 15+) or
  x_raw suppression (cycle 14+ rewire).
- G9 structural probe (`tests/csha_checkpoint_full_save_suppression.rs`):
  4 falsifiable assertions covering suppression, baseline byte-identity,
  refusal-comment presence, and the cycle-12 REACHABILITY corollary
  fallback path.
- `validate_checkpoint_eligibility` visibility widened from private to
  `pub(crate)` so the forward emit site can consult the same cascade
  the backward dispatch site does — guaranteeing forward-emit and
  backward-route stay in lockstep (no uninitialized HBM reads).
- `save_activations_for_backward` and `checkpoint` stay ORTHOGONAL.
  The gate's doc-comment explains why and forbids collapsing them.

### Deferred from cycle 13

- **Real HBM measurement**: cycle 14 — requires Blackwell. The codegen-
  site refusal comment carries the marker
  `"HBM-byte savings claim unvalidated until cycle 14"`, asserted by
  G9-c.
- **Q_proj save-suppression**: cycle 15+ — requires `emit_q_recompute`
  orchestrator + second backward dispatch fork (backward `q_load::emit`
  currently reads `q_proj_ptr` unconditionally per Phase A facet 2).
- **x_raw save-suppression**: cycle 14+ — Phase A facet 1 §B.1 confirms
  cycle-11/12 `emit_kv_recompute` loads x from `x_raw_ptr`, so
  suppressing x_raw save breaks recompute. Requires forward-emitter
  rewire to expose `x_input` as a live SMEM/register tile to the
  backward path.
- **HBM allocation savings**: `nsl_csha_alloc_backward_activations`
  still allocates K_proj/V_proj HBM even though writes are now
  suppressed. v1.3 accepted: runtime null-skip handles unused pointers
  safely. Cycle-14 threads the gate into the alloc site as part of the
  empirical-measurement plumbing.
- **WengertOp.checkpointed first-consumer wiring**: deferred. Cycle-10's
  decorative stamp stays decorative for one more cycle. Production
  wire-up at `compiler/kernel.rs:711-715` (cycle-12) already populates
  `FlashAttentionConfig.checkpoint` at function granularity, which
  `should_emit_kv_save` consumes directly. Per-CSHA-call granularity
  within a single function (the only thing the WengertOp stamp would
  enable) is not needed until multiple CSHA calls per function carry
  distinct checkpoint policies.
- **R11 / R12 cascade lifts**: separate cycles; cycle 13 consumes the
  existing cycle-12 cascade unchanged.
- **Selective per-tensor checkpointing**: `policy="full"` only.

### Gate inventory updates

| Gate | Status | Notes |
|---|---|---|
| G9 K/V structural save-suppression | GREEN | 4/4 in csha_checkpoint_full_save_suppression |
| G1 byte-identity (no-decorator) | GREEN | 25/25 fa_v2_snapshots preserved post-cycle-13 |

## Cycle 14 GPU harness activation — R0 STAYS

Cycle 14 wired the cycle-13 skeleton harness end-to-end on RTX 5070 Ti
sm_120 with the goal of letting numerics decide R0 disposition. Both
paths landed RED on hardware; R0 remains in production refusal cascade.

### What landed (6 commits, all GREEN at structural level)

- **Task 1** — `cuda_available()` activation via `nsl_cuda_init() == 0`
  honoring `NSL_SKIP_CUDA_TESTS`; full FFI declaration block mirroring
  the sister `csha_cuda_backward.rs:57-65` template. G14-E default-run
  gate pins the env-var honor invariant.
- **Task 2** — Config downsized to `block=32`, `gpu_sm=80`. Backward
  prelude target selector now mirrors forward (`Sm80` when
  `gpu_sm >= 80`, else `Sm75`) — required for G14-C `.target sm_80`
  pin. fa_v2_snapshots 25/25 still byte-identical. G14-B/C/D default-run
  gates all GREEN.
- **Task 3** — Forward `nsl_flash_attention_csha_with_saves` launches
  cleanly (rc=0) at hd=64 S=512 on RTX 5070 Ti; first 4 output values
  reach back to host as finite f16; 1940/32768 elements non-zero (sparse
  is plausible under causal+rope_q+f16-flush).
- **Task 4** — Path A (`checkpoint=None` baseline) launches successfully
  after R-C14-4 dynamic-SMEM grant fix (90496 bytes for hd=64 bq=32
  L1-fused-proj), but produces gradients that diverge from
  `csha_reference_backward` by orders of magnitude on all 7 tensors.
  See "R0 disposition" below.
- **Task 5** — Path B (`checkpoint=Some(Full)`) diagnostic surfaced
  FOUR pre-existing cycle-11/12 bugs:
    1. `feedback_ptx_comment_ascii_only` violation (`§` in six PTX
       comments) — FIXED in `csha_hooks_backward.rs`.
    2. `emit_kv_recompute` undeclared predicate/u64 regs — FIXED via
       function-scoped sub-block.
    3. `emit_prologue_recompute_from_raw` ~13 undeclared regs — FIXED.
    4. `emit_one_recompute_matmul` ~18 undeclared regs — FIXED.
  After these 4 fixes Path B reaches `emit_rope_k_epilogue` (step 5 of
  emit_kv_recompute) which calls into the forward's RoPE emitter
  expecting RoPE registers in scope — but the backward prelude doesn't
  declare them. Path B still RED with ptxas rc=218. Closing this is
  cycle-15 work.

### R0 disposition: STAYS

R0 must reaffirm because:

- **Path A (kv_load baseline) RED vs cpu_reference** on RTX 5070 Ti
  at hd=64 S=512 bq=32 causal+rope_q+L1-fused-proj (the cycle-14
  config flavor). Verbatim per-tensor results:

      dq:  max_abs=4.420e0  max_rel=1.000e0
      dk:  max_abs=4.356e0  max_rel=2.040e1
      dv:  max_abs=8.571e0  max_rel=3.979e3
      dwq: max_abs=3.403e1  max_rel=1.000e0
      dwk: max_abs=3.664e1  max_rel=1.020e0
      dwv: max_abs=1.733e1  max_rel=1.032e0
      dx:  max_abs=4.691e1  max_rel=1.118e0

  This is independent of @checkpoint — the dispatch fork at
  `mod.rs:1496` takes the standard `kv_load` branch when
  `checkpoint=None`. So the bug is upstream of cycle-11/12, in the
  shared Level-1 fused-projections scalar backward emission. The
  closest GREEN sister (`t6_3_smoke_single_config` at causal=false
  rope_q=false hd=32 level=2) does not exercise this config flavor.

- **Path B doesn't even reach the launch state.** The cycle-11/12
  `emit_kv_recompute` and its callees emit PTX that ptxas rejects
  (Bugs 2+3+4 fixed in cycle 14; Bug 5 remains for cycle 15). With
  Path B uncompilable, the §5.3 mechanism's numerical correctness
  remains untested on Blackwell. The structural fixes that DID land
  (ASCII + 3 register-decl gaps) tighten the cycle-15 surface area
  without unlocking the validation.

### Paper §6.3 49%-headline status (post-cycle-14)

Cycle 14 was scoped to deliver "partially validated K/V suppression
mechanism end-to-end on Blackwell." The mechanism remains **UNVERIFIED
numerically** on hardware. The cycle-13 STRUCTURAL gate (G9: K and V
save sites suppressed) holds; the cycle-14 NUMERICAL gate stays RED.

### Cycle 15 prerequisites (what unblocks the §5.3 evidence)

1. Add the RoPE register block to `phases/backward/prelude.rs` (or
   factor into a shared helper called from both preludes). This is
   what makes Path B compilable — without it, no Path B comparison is
   possible at all.
2. Root-cause the Path A numerical divergence. The config flavor
   (causal=true + rope_q=true + L1 fused_projections) has never had a
   full backward GPU oracle test; `t6_3_matrix_sweep_numerical` covers
   level=2 not level=1.
3. Once both are closed, re-run the cycle-14 harness. With a clean
   Path A baseline + a launchable Path B, the cycle-14 spec §2 R0
   retirement criterion ("Path B vs cpu_reference GREEN") becomes
   testable on Blackwell.

### Gate inventory updates (cycle 14)

| Gate | Status | Notes |
|---|---|---|
| G14-B R5 refusal pin (hd=128 bq=64 over-cap) | GREEN | `g14_b_recompute_hd128_s4096_bq64_refuses_r5` |
| G14-C `.target sm_80` PTX pin | GREEN | After backward prelude target-sm fix |
| G14-D SMEM budget accounting | GREEN | `g14_d_recompute_extra_bytes_accounted` |
| G14-E cuda_available NSL_SKIP_CUDA_TESTS honor | GREEN | `g14_e_cuda_available_honors_skip_env` |
| G14-F Path A baseline correctness | RED | causal+rope_q+L1-fused-proj diverges on RTX 5070 Ti |
| G14-G Path B compilability | RED | ptxas rejects emit_rope_k_epilogue undeclared regs |
| G14-H Path B numerical correctness (§5.3) | BLOCKED | depends on G14-F + G14-G |

## CYCLE 15 - BUG 1 (Tier B scalar backward): DEFERRED TO CYCLE 16

**Triage protocol completed.** Ablation test infrastructure committed at
`crates/nsl-codegen/tests/csha_cycle15_bug1_ablations.rs`. All four ablations
(A1-A4) compile cleanly under `cargo test --no-run` but GPU execution requires
Blackwell hardware not available in this task session. Results below are
STATIC-ANALYSIS-ONLY, not empirical GPU runs.

Ablation results (STATIC-ANALYSIS):
- A1 (rope_q off): NOT-RUN-NO-GPU
- A2 (causal off): NOT-RUN-NO-GPU
- A3 (fused_proj off): NOT-RUN-NO-GPU
- A4 (hd=128): NOT-RUN-NO-GPU

**Pattern:** STATIC-ANALYSIS-DEFERRED (no GPU in task scope)

**Root cause (static analysis, HIGH confidence):** Two compounding structural defects
in the Tier C backward KV-loop / Phase 3 hook split:

1. **dK SMEM staleness in Phase 3 (Candidate C, HIGH confidence):**
   `emit_store_kv_only` (in `finalize.rs:68`) is called inside the KV outer loop
   and writes the current-tile dK to f32 HBM scratch via RMW, then the SMEM tile
   is re-zeroed for the next KV tile. After the KV loop completes, the dK SMEM
   tile holds ONLY the last-KV-tile contribution (k_start = seq-block_kv). Phase 3
   then calls `emit_drope` and `emit_dproj` on this stale SMEM tile, so dWk
   receives only a fraction of the correct dWk accumulation (1/num_kv_tiles ≈ 1/16
   for seq=512, block_kv=32). This corrupts dwk and cascades into dx.

2. **dK inverse-RoPE k_start offset missing (Candidate B, HIGH confidence):**
   In `csha_hooks_backward.rs:315`, the K tile's cos/sin index uses
   `mov.u64 %rd35, %rd33` (tile_local_row only, 0..block_kv-1), ignoring k_start.
   After the KV loop, the dK SMEM holds positions from the last KV tile
   (k_start = 480 for seq=512, block_kv=32). `emit_drope` applies cos[0..31]
   to those positions instead of cos[480..511], corrupting the dK SMEM tile that
   `emit_dproj` reads for dWk.

3. **dK HBM output is pre-inverse-RoPE:** The f32 scratch populated by
   `emit_store_kv_only` inside the KV loop is converted to f16 dK output by the
   runtime after the kernel exits. This dK is the attention-backward dK BEFORE
   inverse RoPE, while the CPU reference returns dK AFTER inverse RoPE. This
   explains the dK tensor failure independently of defects 1 and 2.

These three defects interact: the dV failure and large dQ errors likely stem from
cascading numerical errors across the softmax Jacobian path once the D-correction
strip or S-recompute uses stale state. The 4.4e0 dQ error and 3.979e3 dV relative
error magnitude match what is expected from a missing-RoPE + partial-accumulation
double fault.

**Why deferred:** Fixing these three defects requires:
(a) Restructuring the backward to accumulate dK across KV tiles using SMEM RMW
    (or carrying k_start into Phase 3 hooks so the inverse RoPE uses the correct
    cos/sin slice), AND
(b) Emitting a Phase 4 cooperative store for dK (post-inverse-RoPE) to HBM f16
    in the same manner as dQ.
The combined fix surface is >60 LOC across finalize.rs, mod.rs, and
csha_hooks_backward.rs. This exceeds the Task 2 bounded-scope limit (30 LOC)
and requires careful regression testing against the full fa_v2_snapshots suite.

**Cycle 16 prerequisite:** Ship a revised backward orchestration that:
- Carries k_start into `emit_drope` K-tile cs_row computation OR restructures
  Phase 3 to process each KV tile's dK contribution in-loop before re-zeroing.
- Stores post-inverse-RoPE dK to the f16 HBM output in Phase 4 (symmetric to dQ).
- All three structural defects must be fixed together because they interact.

**Cycle-5 invariant:** §5.3 numerical validation NOT claimed for Path A baseline.
Paper §6.3 49% headline remains STRUCTURAL PARTIAL, NUMERICAL UNVERIFIED on Blackwell.

