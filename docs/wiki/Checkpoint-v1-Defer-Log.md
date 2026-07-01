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
| G14-G Path B compilability | GREEN (cycle-15) | Resolved by Bug 2 fix at commit `525a2fb5`; verified empirically by cycle-15 Task 3 GPU re-run on RTX 5070 Ti (no ptxas rc=218) |
| G14-H Path B numerical correctness (§5.3) | BLOCKED | depends on G14-F + new G15-K (Path B runtime CUDA_ERROR_ILLEGAL_ADDRESS) |

## CYCLE 15 - BUG 2 (RoPE register gap, cycle-14 Bug 5 equivalent): RESOLVED

**Status:** RESOLVED in commit `525a2fb5` (cycle-15 Task 1).

**Identity:** Cycle-15 Bug 2 == cycle-14 Bug 5 (`emit_rope_k_epilogue` missing
RoPE register block from backward prelude). Same defect, finally closed.

**Root cause:** Forward prelude declared the RoPE pair-sweep register block
under gate `cfg.rope_q && cfg.csha.is_some()`, but backward prelude did NOT.
`emit_kv_recompute` (cycle-12 functional substitution path) invokes
`emit_rope_k_epilogue` which emits PTX referencing `%rd_rope_cos`,
`%rd_rope_sin`, `%f_rope_cos`, etc. - undeclared in backward prelude scope.
ptxas rejected at PTX line ~18541 with rc=218 "Arguments mismatch for
instruction 'ld'".

**Fix:** Lifted the verbatim register block (11 .reg lines) into
`crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs`
under the same gate as forward. Placement: after base register pools,
before the PCA §4.3 RoPE-reset block (which declares disjoint registers
under the narrower `segment_masked && rope_q` gate).

**Structural witnesses:** G15-2a (positive: `rope_q=true && csha.is_some()`
emits `%rd_rope_cos` in prelude window) and G15-2b (negative: `rope_q=false`
skips block) in `crates/nsl-codegen/tests/cycle15_backward_prelude_rope.rs`.
Both GREEN.

**Empirical verification:** Cycle-15 Task 3 GPU re-run on RTX 5070 Ti
(Blackwell sm_120, CUDA 13.2, cudarc 0.19.4): Path B PTX compiles cleanly
(no ptxas rc=218). Independently reproduced by Reviewer 2 byte-for-byte.

**Gate update:** G14-G transitions RED -> GREEN. G15-Bug2 closed.

**Cycle-5 invariant:** Bug 2 closure unblocks Path B PTXAS COMPILE only.
It does NOT unlock §5.3 numerical validation. Path B still RED at runtime
with CUDA_ERROR_ILLEGAL_ADDRESS (new defect, see Cycle 16 work items).
R0 retirement HOLDS unconditionally.

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

### Cycle-15 GPU re-run on RTX 5070 Ti (CUDA 13.2, cudarc 0.19.4)

**t_recompute_hd64_s512_bq32:**

- Path A (baseline, checkpoint=None): RED -- all 7 gradients FAIL.
  dq: max_abs=4.420e0 max_rel=1.000e0 | dk: max_abs=4.356e0 max_rel=2.040e1 |
  dv: max_abs=8.571e0 max_rel=3.979e3 | dwq: max_abs=3.403e1 max_rel=1.000e0 |
  dwk: max_abs=3.664e1 max_rel=1.020e0 | dwv: max_abs=1.733e1 max_rel=1.032e0 |
  dx: max_abs=4.691e1 max_rel=1.118e0
- Path B (checkpoint=Some(Full)): CUDA_ERROR_ILLEGAL_ADDRESS crash in
  nsl_flash_attention_csha_backward after backward launch. PTX compiled (no
  ptxas rc=218). No numerical values produced -- process aborted with
  STATUS_STACK_BUFFER_OVERRUN (exit code 0xc0000409). Bug 2 fix prevented the
  ptxas compile failure but Path B still crashes at runtime with an illegal
  address access in the backward kernel.

**Cycle-15 ablations (all run sequentially --test-threads=1):**
- A1 (rope_q=FALSE): INFRA-FAIL -- panicked at csha_reference.rs:85:31 "index
  out of bounds: the len is 0 but the index is 0". The CPU reference
  implementation does not handle rope_q=false; this is a test infrastructure
  bug, not a GPU backward bug. No GPU numerical output produced.
- A2 (causal=FALSE): RED -- all 7 gradients FAIL. PTX compiled, backward
  launched (rc=0). dq: max_abs=inf max_rel=inf | dk: max_abs=inf max_rel=inf |
  dv: max_abs=inf max_rel=inf | dwq: max_abs=inf max_rel=inf |
  dwk: max_abs=3.463e1 max_rel=5.188e1 | dwv: max_abs=1.533e1 max_rel=4.015e1 |
  dx: max_abs=9.269e6 max_rel=1.916e9. inf values on dq/dk/dv/dwq indicate
  catastrophic divergence (NaN/inf in output); dx magnitude 9.269e6 vs Path A
  4.691e1 indicates causal=false exposes additional kernel defects beyond Bug 1.
- A3 (fused_proj=FALSE): CUDA_ERROR_MISALIGNED_ADDRESS crash on cuMemFree_v2
  after backward launch. PTX compiled (SMEM=45696 bytes dyn=0). No numerical
  output produced. The misaligned-address error on free indicates GPU memory
  corruption during or after the backward kernel execution with fused_proj=false.
- A4 (hd=128): SKIPPED -- backward validator refused: 217472 bytes > 101376 byte
  SMEM cap at (block_q=32, head_dim=128). Config not executable on this device.

**Classification:** Spec §5 matrix cell = **Path A RED + Path B RED**.
Path B's failure mode CHANGED (ptxas rc=218 in cycle 14 -> CUDA_ERROR_ILLEGAL_ADDRESS
at runtime in cycle 15). Bug 2 fix (Task 1) confirmed compilable at PTX-emit but
revealed a downstream runtime defect (G15-K, see Cycle 16 work items). Path A
remains numerically RED (Bug 1 deferred). Cycle-5 disposition: §5.3 NUMERICAL
EVIDENCE NOT CLAIMABLE. Cycle-14 spec's "CASE C (partially)" working label is
NOT in spec §5 - the honest matrix mapping is "both bugs persist, scope reduced
to Bug 2 closure + Bug 1 triage report + new runtime defect surfacing."

**Task-2 static analysis empirical status:** PARTIAL -- Path A numbers are
byte-for-byte consistent with cycle-14 results (same magnitudes: dq 4.420e0,
dv 3.979e3 max_rel, dx 4.691e1), confirming Tasks 1-2 did not perturb the
forward/backward emission for the baseline path. The three static-analysis defects
(dK SMEM staleness, k_start offset, pre-RoPE HBM output) remain the leading
hypothesis for Bug 1. However, A1 infra-failure and A3/Path-B runtime crashes
prevent empirical confirmation of which specific defect drives which gradient error.
A2 (causal=false) shows inf values not present in Path A, suggesting causal masking
suppresses a secondary divergence path that is distinct from Bug 1.

**Cycle-5 invariant disposition:** §5.3 numerical evidence NOT CLAIMED.
Path B compiles PTX (Bug 2 fix confirmed) but crashes at runtime; no
checkpoint-recompute backward numerics were produced.

**Paper §6.3 49% headline:** STRUCTURAL PARTIAL, NUMERICAL UNVERIFIED on Blackwell.

### Cycle 16 work items (explicitly labeled)

The following four surfaces are cycle-15 OUTPUTS - all carried forward to cycle 16:

1. **G16-1 (carries from Bug 1):** Tier B scalar backward fix (3 structural
   defects). Highest priority. See "CYCLE 15 - BUG 1" section above for full
   triage. Estimated >60 LOC across `finalize.rs`, `mod.rs`,
   `csha_hooks_backward.rs`. Must fix all three together (dK SMEM staleness,
   k_start offset, pre-RoPE HBM output). Required to GREEN Path A baseline.

2. **G16-2 (NEW from cycle-15 Task 3):** Path B runtime
   `CUDA_ERROR_ILLEGAL_ADDRESS` inside `nsl_flash_attention_csha_backward`
   after Bug 2 (PTX compile) closure. Suspected memory/pointer defect in
   `emit_kv_recompute` path or its load addressing. Must be triaged with
   `cuda-memcheck` or printf instrumentation. Required to GREEN Path B
   compilability-to-launch.

3. **G16-3 (NEW from cycle-15 Task 3):** Ablation A3 (`fused_proj=FALSE`)
   `CUDA_ERROR_MISALIGNED_ADDRESS` on cuMemFree_v2/cuMemsetD8_v2 inside or
   after backward kernel. SMEM=45696 bytes dyn=0. Separate from G16-2 because
   it surfaces on a different config flavor (no fused projections), but
   likely related to memory layout/alignment in the non-fused backward path.
   Cycle 16 should triage these two together if they share a root cause.

4. **G16-4 (NEW from cycle-15 Task 3, INFRASTRUCTURE):** Test infrastructure
   bug in `csha_reference.rs:85:31` - panics with "index out of bounds:
   the len is 0 but the index is 0" when `rope_q=false`. The CPU reference
   does not handle empty cos/sin slices. Must be fixed BEFORE A1 ablation
   can produce useful narrowing data. NOT a GPU backward bug.

The following surface is **NOT a cycle-16 work item** - it is a configuration
constraint, not a defect:

- **A4 SMEM cap refusal** (217472 > 101376 byte cap at `block_q=32, head_dim=128`
  on sm_120, Blackwell consumer). The backward validator correctly refuses
  this config because the SMEM requirement exceeds the device cap. To
  exercise `head_dim=128` on cycle-16 ablations, EITHER (a) reduce
  `block_q` to 16, OR (b) restructure the backward to consume less SMEM
  per tile. Neither requires defect-level work; this is a config-not-bug
  classification per cycle-5 honesty discipline.

---

## CYCLE 16 - TASK 3: G16-2 + G16-3 Joint Triage

**Date:** 2026-06-27
**Branch:** `feat/csha-cycle16-multi-bug-closure`
**Tests added:** `crates/nsl-codegen/tests/csha_cycle16_g16_2_g16_3_triage.rs` (6 structural tests)

### G16-2: Path B CUDA_ERROR_ILLEGAL_ADDRESS -- RESOLVED (commit `2f62090e`)

**Root cause identified and fixed in cycle-16 Task 3.**

The `emit_kv_recompute` path (activated only when `checkpoint=Some(Full)`) writes a
recomputed x_norm scratch tile to SMEM starting at `recompute_xnorm_offset`. The
`smem_layout::recompute_xnorm_offset(config)` returns:

    total_bytes(config) + backward_extra_bytes(config)

This is exactly the boundary of what `shared_mem_bytes_v2_backward` was granting
before the fix. The kernel wrote to SMEM addresses beyond the allocation boundary
immediately after launch, corrupting adjacent memory. The crash manifested at the
subsequent `cuMemFree_v2(dk_scratch_raw)` call inside
`nsl_flash_attention_csha_backward` because the kernel had overwritten allocator
metadata for the scratch buffer.

**Fix (< 10 LOC):**
`shared_mem_bytes_v2_backward` in `crates/nsl-codegen/src/flash_attention_v2/mod.rs`
now adds `recompute_extra_bytes(config) as u32` when `config.checkpoint.is_some()`.
The test harness `launch_backward_path()` in `csha_checkpoint_recompute_gpu.rs` was
also updated to derive SMEM from `shared_mem_bytes_v2_backward` rather than computing
it manually.

**Verification:**

- Path B (checkpoint=Full, hd=64, S=512) now runs to completion: SMEM 90496 -> 94592 bytes.
- Path B is numerically RED (inherits Bug 1 / G16-1 divergence), but no CUDA error.
- lib tests: 2049/2049 GREEN.
- fa_v2_snapshots: 25/25 GREEN (byte-identical; no non-checkpoint path affected).

**Structural witnesses (pass by default, no GPU needed):**

- `g16_2_t1_smem_includes_recompute_extra_after_fix`: delta == `recompute_extra_bytes`.
- `g16_2_t2_smem_no_change_without_checkpoint`: Path A SMEM unchanged.
- `g16_2_t3_recompute_xnorm_offset_equals_old_boundary`: confirms write started at old boundary.

---

### G16-3: A3 CUDA_ERROR_MISALIGNED_ADDRESS -- DEFERRED to cycle 17

**Root cause: structurally inconclusive in cycle 16.**

Config: `causal=true, rope_q=true, fused_projections=false, d_model=0, hd=64, S=512, bq=32, checkpoint=None` (Path A).

**Static analysis findings:**

- Kernel synthesizes OK (PTX emission step is not the failure).
- Kernel launches successfully (launch rc=0, kernel runs to completion).
- Crash fires during `cuMemFree_v2(dk_scratch_raw)` inside
  `nsl_flash_attention_csha_backward` -- GPU memory corruption during or after kernel.
- SMEM=45696 bytes, dyn=0 (static) -- this is NOT a SMEM-grant issue (unlike G16-2).
- `emit_xnorm_recompute`, `emit_dproj`, `emit_drmsnorm`: all return early when `d_model=0`; no SMEM writes from these paths.
- The "Phase 1b drains garbage" hypothesis from cycle-15 task description is
  structurally REFUTED: `emit_drmsnorm` exits at lines 551-553 BEFORE emitting
  any phase 1, 1b, or 2 code when `d_model == 0`.
- Basic dQ/dK/dV backward (emit_store_kv_only, emit_store_dk_only) still runs and
  is the candidate corruption site.

**SMEM aliasing finding (documented in g16_3_t1_zero_d_model_smem_aliasing):**
When `d_model=0`, `backward_x_norm_offset == backward_dx_norm_offset == backward_rms_strip_offset`.
The three conceptual tiles collapse to the same SMEM address. The rms_strip tile
(bq*4 = 128 bytes) lives at this offset and could alias with Phase-4 SMEM read/write
ranges from the dQ/dK/dV cooperative store.

**Working hypothesis for cycle 17:**
A Phase-4 SMEM cooperative store (dQ or dK tile) overshoots into the rms_strip
region (which is zero-sized due to d_model=0 aliasing). With rms_strip co-located
at the same offset as the V_in tile or another adjacent tile, the write corrupts
the allocator metadata for the dk_scratch buffer that was placed immediately after
the kernel's SMEM shadow region on the heap. Needs `cuda-memcheck` to confirm
the exact address of the stray write.

**Cycle 16 disposition:** DEFERRED-TO-CYCLE-17 per cycle-5 deferral-must-refuse
invariant. This is a judgment call -- the crash is real, observed, and reproducible,
but the fix scope is inconclusive (could be SMEM layout re-ordering, a missing
guard, or allocator interaction). Forcing a fix without cuda-memcheck evidence
risks silent mis-correction.

**Repro:**

    cargo test --release --features cuda --test csha_cycle15_bug1_ablations -- \
      --ignored a3_fused_proj_off_causal_true_rope_q_true_hd64 --nocapture

**Structural witnesses (pass by default, no GPU needed):**

- `g16_3_t1_zero_d_model_smem_aliasing`: confirms 3-way aliasing when d_model=0.
- `g16_3_t2_a3_backward_ptx_synthesizes_ok`: PTX synthesis succeeds; crash is runtime-only.
- `g16_3_t3_a3_smem_is_static`: SMEM < 48*1024; dyn=0 (unlike G16-2 which needed dynamic SMEM).

---

### Cycle-16 Task 4 - GPU re-run on RTX 5070 Ti (CUDA 13.1, cudarc 0.19.4)

**Hardware:** NVIDIA GeForce RTX 5070 Ti (WDDM), Driver 591.86, CUDA header 13.1.

**Build status:** BLOCKED on C drive (0 bytes free). Linker temp files write to
C:\Users\bwiem\AppData\Local\Temp which is exhausted. Fresh E-drive build
(CARGO_TARGET_DIR=E:\cargo-target) failed at nsl-codegen lib due to pre-existing
compile errors (CheckpointPolicy import path divergence in source_ad.rs +
entry_points.rs; AnalysisResult struct mismatch). Best available binary:
`csha_checkpoint_recompute_gpu-ee148af4d4ea1f41.exe` (16:34 local time, 2700288
bytes). This binary includes Task 1 (G16-4) + Task 2 runtime Defect-3 (dK f32->f16
conversion removed from nsl-runtime, compiled at 16:28). It does NOT include Task 2
PTX Defects 1+2 (csha_hooks_backward.rs k_start fix + finalize.rs Phase-4 dK store)
because the nsl-codegen rlib linked at 16:34 predates the 16:49 rlib that contains
those changes. Evidence: Path A magnitudes are byte-identical to cycle-15 baseline,
which is impossible if the PTX K-tile RoPE fix (Defect 2) landed.

**t_recompute_hd64_s512_bq32:**

- Path A (baseline, checkpoint=None):
  - dq: max_abs=4.420e0 max_rel=1.000e0 (atol=5e-4 rtol=5e-3) FAIL
  - dk: max_abs=4.356e0 max_rel=1.000e0 (atol=5e-4 rtol=5e-3) FAIL
  - dv: max_abs=8.571e0 max_rel=3.979e3 (atol=5e-4 rtol=5e-3) FAIL
  - dwq: max_abs=3.403e1 max_rel=1.000e0 (atol=1e-3 rtol=1e-2) FAIL
  - dwk: max_abs=3.664e1 max_rel=1.181e0 (atol=1e-3 rtol=1e-2) FAIL
  - dwv: max_abs=1.733e1 max_rel=1.032e0 (atol=1e-3 rtol=1e-2) FAIL
  - dx: max_abs=4.691e1 max_rel=1.029e0 (atol=1e-2 rtol=2e-2) FAIL
  - bwd SMEM=90496 bytes, dyn_request=90496
  - Status: RED (byte-identical to cycle-15; PTX Defects 1+2 not in binary)

- Path B (checkpoint=Some(Full)):
  - dq: max_abs=1.341e1 max_rel=3.566e6 (atol=5e-4 rtol=5e-3) FAIL
  - dk: max_abs=4.356e0 max_rel=1.000e0 (atol=5e-4 rtol=5e-3) FAIL
  - dv: max_abs=2.022e0 max_rel=3.702e4 (atol=5e-4 rtol=5e-3) FAIL
  - dwq: max_abs=4.534e1 max_rel=3.315e3 (atol=1e-3 rtol=1e-2) FAIL
  - dwk: max_abs=3.658e1 max_rel=5.124e2 (atol=1e-3 rtol=1e-2) FAIL
  - dwv: max_abs=1.728e1 max_rel=2.570e1 (atol=1e-3 rtol=1e-2) FAIL
  - dx: max_abs=9.497e1 max_rel=3.116e4 (atol=1e-2 rtol=2e-2) FAIL
  - bwd SMEM=94592 bytes, dyn_request=94592
  - Path B vs Path A: dq FAIL, dk PASS, dv FAIL, dwq FAIL, dwk FAIL, dwv FAIL, dx FAIL
  - Status: RED (numerically wrong but no crash; changed from cycle-15's CUDA_ERROR_ILLEGAL_ADDRESS)

**Cycle-15 ablations (re-run after G16-4, using binary at 16:37 which has G16-4 + runtime Defect-3):**

- A1 (rope_q=FALSE, causal=true, fused_proj=true, hd=64, S=512):
  - RUNS TO COMPLETION -- no panic (G16-4 fix confirmed)
  - dq: max_abs=6.198e4 max_rel=3.406e8 FAIL
  - dk: max_abs=5.415e0 max_rel=1.000e0 FAIL
  - dv: max_abs=7.779e3 max_rel=1.588e6 FAIL
  - dwq: max_abs=3.049e4 max_rel=6.807e6 FAIL
  - dwk: max_abs=4.010e1 max_rel=1.852e1 FAIL
  - dwv: max_abs=1.851e1 max_rel=4.804e1 FAIL
  - dx: max_abs=5.478e5 max_rel=3.703e7 FAIL
  - Status: RED numerically (expected -- same PTX bugs; G16-4 only fixed CPU reference panic)

- A2 (causal=FALSE, rope_q=true, fused_proj=true, hd=64, S=512):
  - dq: max_abs=inf max_rel=inf FAIL
  - dk: max_abs=3.740e0 max_rel=1.000e0 FAIL
  - dv: max_abs=inf max_rel=inf FAIL
  - dwq: max_abs=inf max_rel=inf FAIL
  - dwk: max_abs=3.460e1 max_rel=1.532e1 FAIL
  - dwv: max_abs=1.533e1 max_rel=4.015e1 FAIL
  - dx: max_abs=9.269e6 max_rel=1.916e9 FAIL
  - Status: RED (unchanged from cycle-15 in character; dk/dwk/dwv finite matches cycle-15)

- A3 (fused_proj=FALSE, causal=true, rope_q=true, hd=64, S=512):
  - CUDA_ERROR_MISALIGNED_ADDRESS in nsl_flash_attention_csha_backward (unchanged)
  - Process aborted (non-unwinding panic); A4 not reached.

- A4 (hd=128, causal=true, rope_q=true, fused_proj=true): SKIPPED (config-not-bug; A3 abort prevents A4)

**Comparison to cycle-15:**

- Path A: BYTE-IDENTICAL (dq 4.420e0, dk 4.356e0, dv 8.571e0/3.979e3, dwq 3.403e1, dwk 3.664e1, dwv 1.733e1, dx 4.691e1) -- confirms PTX Defects 1+2 NOT compiled into tested binary
- Path B: CHANGED from CUDA_ERROR_ILLEGAL_ADDRESS to numerical RED -- confirms Defect-3 (runtime dK f32->f16 conversion removal) partially effective; PTX bugs remain
- A1: CHANGED from csha_reference.rs:85 OOB panic to numerical RED -- G16-4 fix confirmed empirically
- A2: UNCHANGED in character (inf dq/dv/dwq, finite dk/dwk/dwv/dx)
- A3: UNCHANGED (CUDA_ERROR_MISALIGNED_ADDRESS abort)
- A4: UNCHANGED (skipped)

**Classification:** Spec section 7 cell = Path A RED + Path B RED

Cell text: "Both still RED. Cycle scope shrinks to: ship whatever closed cleanly (likely G16-4), defer the rest. R0 stays retired. Cycle 17 carries G16-1/G16-2/G16-3. Paper section 6.3 stays STRUCTURAL PARTIAL."

**Task-1 (G16-4) empirical disposition:** EMPIRICALLY CONFIRMED. A1 no longer panics; runs to numerical comparison. CPU reference rope_q=false gate verified working on RTX 5070 Ti.

**Task-2 (G16-1) empirical disposition:** PARTIALLY EMPIRICALLY CONFIRMED, FULL VERIFICATION BLOCKED. Defect-3 (runtime dK f32->f16 conversion removal) confirmed via Path B no longer crashing. PTX Defects 1+2 (K-tile k_start + Phase-4 dK cooperative store) NOT testable in this session -- nsl-codegen PTX changes could not be compiled due to disk-full blocker on build host. Deferred to Cycle 17 as first action: fresh build + GPU re-run on clean disk.

**Task-3 (G16-2 SMEM grant) empirical disposition:** SMEM grant structural witnesses all pass (g16_2_t1 through g16_2_t3, 3/3 ok). GPU execution of Path B runs to completion (94592 bytes SMEM, no CUDA_ERROR_ILLEGAL_ADDRESS) -- this is the primary empirical evidence that G16-2 landed. However the binary tested (16:34) predates the G16-2 SMEM grant change (16:52 commit), so the SMEM=94592 result is from the pre-G16-2 binary running at hd=64 where SMEM naturally fits. Full G16-2 validation requires fresh build at hd=128 or a config that previously hit the SMEM cap.

**Cycle-5 invariant disposition:** section 5.3 numerical evidence NOT CLAIMABLE. Both Path A and Path B RED. Independent reviewer re-run on clean build host required before any claim.

**Paper section 6.3 49% headline:** STRUCTURAL PARTIAL, NUMERICAL UNVERIFIED on Blackwell.

**Build blocker for Cycle 17:** C drive must be freed before next GPU re-run. Pre-existing nsl-codegen compile errors (CheckpointPolicy import path + AnalysisResult field mismatch) prevent fresh E-drive build and must be resolved or the incremental C-drive cache preserved. Recommend: (1) free C drive space, (2) verify `cargo build --release --features cuda --tests` succeeds cleanly on C drive, (3) re-run t_recompute_hd64_s512_bq32 as first Cycle 17 action.

---

### Cycle-16 Task 4 SUPPLEMENTARY VERIFICATION (post-implementer, fresh E-drive build)

**Critical correction to Task-2 (G16-1) empirical disposition.** Task 4 implementer
assumed PTX Defects 1+2 were NOT in the tested binary because Path A was byte-identical
to cycle 15. The orchestrator re-ran with a fresh E-drive build (TEMP also redirected
to E:\\tmp to bypass linker disk-full) at branch tip post-Tasks 1+2+3:

1. `cargo build --release --features cuda --tests` from cycle-16 worktree with E:
   TEMP + target succeeds in 1m 33s, exit 0. The claimed "pre-existing nsl-codegen
   compile errors" do NOT exist on cycle-16 source -- they were a stale E-drive
   incremental-cache artifact from a partial earlier build.

2. `cargo test --features cuda --test csha_checkpoint_recompute_gpu -- --ignored
   t_recompute_hd64_s512_bq32 --nocapture` produces a kernel that DOES contain all
   cycle-16 PTX fixes. Direct PTX dump verification at
   `E:\\tmp\\cycle14_path_A_hd64_S512.ptx` (2352198 bytes):
   - `add.u64 %rd35, %rd33, %k_start` -- PRESENT (defect-2 K-tile RoPE fix landed)
   - `Phase 4 dK store` / `emit_store_dk_only` marker -- PRESENT (Phase-4 dK store added)
   - Old buggy `mov.u64 %rd35, %rd33;` line (K-tile branch) -- ABSENT (replaced)

3. Path A magnitudes on this fresh binary: BYTE-IDENTICAL to cycle 15 baseline
   (dq 4.420e0, dk 4.356e0, dv 8.571e0 / max_rel 3.979e3, dwq 3.403e1, dwk 3.664e1,
   dwv 1.733e1, dx 4.691e1).

**Empirical conclusion:** G16-1 Option A IS in the kernel PTX but is INSUFFICIENT
to GREEN Path A baseline. The cycle-15 static analysis identifying 3 defects was
directionally correct (Option A landed cleanly per structural witnesses 3/3 GREEN)
but does NOT close the actual numerical bug. The true root cause of Bug 1 is
upstream of or orthogonal to the 3 defects fixed in Option A.

**Cycle 17 implications:**

- DO NOT revert Task 2 commit `d232b615`. The K-tile k_start RoPE fix is provably
  correct (Q-tile symmetric pattern was always intended); the Phase-4 dK cooperative
  store mirrors the dQ pattern which is architecturally sound. Both changes are
  sound directions; they just don't fully close the bug.
- Re-investigate Bug 1 root cause. Working hypotheses for cycle 17:
  1. dK SMEM may be re-zeroed elsewhere (not just inside emit_store_kv_only loop),
     so removing the per-KV-tile dK store didn't actually accumulate dK across KV
     tiles.
  2. The Phase-4 dK store reads from SMEM that was correctly accumulated, but the
     SMEM accumulation itself never happened (dqdk_accum may not be RMW for dK).
  3. Cycle-15 static analysis missed a 4th defect that is the actual numerical-
     divergence root cause.
  4. Bug 1 is in dq/dv/dx path (not just dK/dwk) and the dK/dwk magnitudes match
     because they share an upstream divergence. dq max_rel 1.000e0 (i.e. dq is
     essentially zero where it shouldn't be) and dv max_rel 3.979e3 strongly
     suggest an upstream-of-dK divergence.

**Task-2 corrected disposition:** STRUCTURAL FIX LANDED. EMPIRICAL VERIFICATION
SHOWS OPTION A IS INSUFFICIENT. Bug 1 root cause requires Cycle 17 re-investigation.

**Cycle-16 net disposition:**

| Work item | Cycle 16 outcome (commit SHA) | Cycle 17 carryover |
|---|---|---|
| G16-1 | Option A LANDED `d232b615` (correct direction, insufficient empirically; refuted by `c2c6879b`) | Re-investigate Bug 1 root cause |
| G16-2 | RESOLVED `2f62090e` (SMEM grant fix; Path B compiles + runs; verified) | None (closed) |
| G16-3 | DEFERRED with refined hypothesis (Phase-4 SMEM overshoot into rms_strip at d_model=0) | Triage with cuda-memcheck |
| G16-4 | RESOLVED `a8309e72` (A1 no longer panics; empirically confirmed) | None (closed) |
| C17-cleanup | (R1 MEDIUM finding) `%rd_dk_*` register names in `emit_store_kv_only` (V-only path) should be renamed to `%rd_dv_*` for clarity. Non-blocking; cosmetic only. | Rename in cycle 17 |

**Cycle-5 invariant disposition (final):** Section 5.3 numerical evidence NOT
CLAIMABLE. Paper section 6.3 49% headline: STRUCTURAL PARTIAL, NUMERICAL UNVERIFIED
on Blackwell. R0 retirement HOLDS unconditionally.

**Build blocker addendum:** Both blockers (disk-full and "pre-existing compile
errors") were addressable in-session by (1) freeing C: drive (cycle-16 worktree
target dir cleanup ~14 GB), and (2) redirecting both `TEMP` and `CARGO_TARGET_DIR`
to E:. No actual lib compile errors exist on cycle-16 source. Cycle 17 can
proceed with the same workaround.

---

## Cycle 17 (2026-06-30): STRUCTURAL FIX LANDED + NUMERICAL CLOSURE PARTIAL

### G16-1: Tier B scalar backward Bug 1 — STRUCTURAL FIX LANDED, NUMERICAL PARTIAL (commit `89bb7f05`)

**Phase A** (4 parallel investigators + PTX dump comparator + G16-3 prep + decorator scope →
synthesizer): 4/4 hypothesis investigators converged on the post-loop `%k_start` site.
Synthesizer leading verdict H2+H3 COMBINED at confidence 82: post-loop `%k_start` staleness
AND per-KV-iter dK SMEM zero-init defeating cycle-16's post-loop `emit_store_dk_only`.

**T1 GATE (diagnostic probe; reverted post-measurement):** Injected `mov.u64 %k_start, 0;`
post-loop. Empirically confirmed `%k_start` IS load-bearing for Path A:
- dk.max_rel: 1.000 → 1.002 (small but reproducible)
- dwk.max_rel: 1.181 → 1.020 (decreased; wrong-but-different-wrong as Phase A predicted)
- dx.max_rel: 1.029 → 1.118
- dq, dv, dwq, dwv unchanged (separate code paths from dK)

GATE PASS verdict. H1 dispatch-elsewhere concern empirically refuted (Path A reaches the
`synthesize_backward_with_tier_b` emitter as Phase A predicted).

**T3 PRE (compute-sanitizer baseline):** 0 memcheck errors across 5 variants + 0 racecheck
hazards. This initially appeared to REFUTE the OOB hypothesis. Closer inspection of the
subagent's own PTX evidence at `finalize.rs:115-116`:

```
setp.lt.u64 %p0, %rd43, %rd6;     // %rd43 = row + %k_start, %rd6 = seq_len
and.pred %p_dk, %p_dk, %p0;        // mask used to predicate all stores
```

With `%k_start = seq_len` post-loop, the predicate is FALSE for all rows ≥ 0. **The Phase-4
dK store is dead-code-masked.** No OOB writes occur because no writes occur at all.
compute-sanitizer's 0-error result is consistent with dead-code masking.

**Refined root cause:** Cycle-16 Option A's post-loop `emit_store_dk_only` is dead code. dK
HBM ends up as whatever was there from allocation (likely zero). Cycle 15 had no Phase-4
store either, so both produce byte-identical output. The T3 PRE subagent's STOP
recommendation was based on a misreading (assumed memory-safety bug; actual is
correctness bug — store doesn't fire at all). Orchestrator override was sound; PTX-level
dead-code-mask analysis was concrete.

**T2 architectural fix (commit `89bb7f05`):** Introduced `DropeBranch::{Q, K, Both}` enum +
`emit_drope_branch` function (backward-compat `emit_drope` wrapper preserved for 12 legacy
callers). Moved `emit_drope_branch(K)` + `emit_store_dk_only` INSIDE V2_BWD_LOOP_KV body —
between `emit_store_kv_only` (V-only) and the `%k_start` increment. Each iter rotates+stores
its own tile's dK using the in-range loop-variable `%k_start`. Post-loop now ONLY emits
Q-related Phase-3+ work.

**T3 POST (compute-sanitizer validation on T2):** 0 memcheck errors + 0 racecheck hazards.
T2 is memory-safe + race-free. The in-loop store fires correctly per-iter.

**Path A numerical outcome (`t_recompute_hd64_s512_bq32`):**

| grad | pre-T2 max_abs | pre-T2 max_rel | post-T2 max_abs | post-T2 max_rel | disposition |
|------|----------------|----------------|------------------|------------------|-------------|
| dq   | 4.420e0        | 1.000e0        | 4.420e0          | 1.000e0          | UNCHANGED (separate path) |
| dk   | 4.356e0        | 1.000e0        | 4.356e0          | **2.146e1**      | store now firing; values wrong |
| dv   | 8.571e0        | 3.979e3        | 8.571e0          | 3.979e3          | UNCHANGED (separate path) |
| dwq  | 3.403e1        | 1.000e0        | 3.403e1          | 1.000e0          | UNCHANGED |
| dwk  | 3.664e1        | 1.181e0        | 3.664e1          | **1.711e0**      | drifted (worse) |
| dwv  | 1.733e1        | 1.032e0        | 1.733e1          | 1.032e0          | UNCHANGED |
| dx   | 4.691e1        | 1.029e0        | 4.691e1          | **1.043e0**      | drifted (slight) |

**Disposition: STRUCTURAL FIX LANDED, NUMERICAL CLOSURE PARTIAL.** The dead-code mask is
empirically closed (dk HBM transitioned from all-zero to per-iter writes). However, dk
max_rel = 21.46 is NOT within Tier B tolerance — Bug 1 has additional defects beyond
dead-code-masking. **G16-1 NOT closed.**

### New defect surfaced by T2 — cycle 18 carryover

`emit_dproj` reads post-loop dK SMEM. Pre-T2 with dead-code-masked Phase-4 store, post-loop
dK SMEM contained whatever stale data the per-iter zero-init + dqdk_accum + (skipped)
emit_drope(K) had left. T2's per-iter zero-init makes post-loop dK SMEM contain ONLY the
LAST KV iter's rotated tile. emit_dproj now reads this LAST-iter-only tile and produces
drifted dwk (1.181→1.711) and dx (1.029→1.043).

Likely c18 fix candidates:
- (a) `dqdk_accum` formula or scale — verify the in-kernel dK accumulation matches the dKdV math
- (b) `emit_dproj` should compute dwk per-iter (not post-loop), mirroring the dV approach
- (c) Or: dK SMEM accumulation needs an HBM-scratch flush like dV (in-loop f32 RMW), so
  post-loop emit_dproj reads a properly-accumulated tile
- (d) Investigate dq + dv paths separately — both stayed RED through T2; they have their own
  defects that share no code with the dK chain T2 fixed

### G16-3 — SMEM aliasing at d_model=0 — STILL DEFERRED to c18

Did not investigate further in c17 (compute-sanitizer ran only Path A, not A3 ablation).
Refined hypothesis from c16 (Phase-4 SMEM overshoot into rms_strip at d_model=0) remains
untested.

### T4 — CLI decorator regression gates LANDED (commit `be68fd9c`, separate branch)

`crates/nsl-cli/tests/csha_checkpoint_decorator_cli_e2e.rs` (180 LOC):

- **Test A:** `@csha(disable=true)` round-trips through `nsl build --csha-report`; asserts
  CSHA compilation report header is ABSENT in stderr. Empirically verified as LIVE
  REGRESSION GATE — reverting `loader.rs:433` causes Test A to fail with the exact
  expected regression message.
- **Test B:** `@checkpoint(policy=full)` survives `nsl check`; asserts exit 0 + no
  "unknown decorator" stderr. Limitation honestly documented: `nsl check` bypasses the
  multi-file loader, so Test B catches parse/semantic regressions but NOT loader.rs:439.
  The in-process `pipeline.rs::checkpoint_full_policy_survives_pipeline_handoff` unit test
  (from the phase2.6 ← main merge restoration) still holds that gate.

Deviations: Test A uses `disable=true` instead of `level=2` because `--csha-report`'s level
value is the SMEM-clamped post-planning value, not the decorator value (binary present/absent
is a cleaner observable). Test B uses `nsl check` instead of `nsl check --csha-report`
because the `--csha-report` flag was removed from `CheckArgs` during refactor `5049c2c2` —
**existing `csha_check_report_cli.rs` is BROKEN FOR THE SAME REASON** (unrelated to cycle 17;
carryover to c18 or arch-hardening campaign).

### FFI fix (commit `b65f9d77`)

Post-merge `csha_checkpoint_recompute_gpu.rs` FFI signature drift. The phase2.6 ← origin/main
merge updated production FFI signatures to 54 params with trailing pair
`[tier_b2_active, num_docs_or_zero]`, but the merge subagent missed the test file. T1 GATE
subagent surfaced this and applied minimal trailing `num_docs_or_zero=0` placeholders at
both call sites (44-arg `nsl_flash_attention_csha_with_saves` and 54-arg
`nsl_flash_attention_csha_backward`).

### Cycle 17 net disposition

| Work item | Cycle 17 outcome (commit SHA) | Cycle 18 carryover |
|---|---|---|
| G16-1 | STRUCTURAL FIX LANDED `89bb7f05` (in-loop dK store + dRoPE-K). NUMERICAL CLOSURE PARTIAL — dk max_rel 21.46 not in tolerance. | (a) `dqdk_accum` formula/scale audit / (b) `emit_dproj` per-iter dwk compute / (c) dK HBM-scratch RMW like dV / (d) dq + dv path investigation |
| G16-3 | DEFERRED to c18 (untouched in c17) | cuda-memcheck triage with refined hypothesis (Phase-4 SMEM overshoot at d_model=0) |
| T4 decorator CLI | LANDED `be68fd9c` (Test A live gate; Test B parse-only with documented limitation) | Restore `--csha-report` flag in `CheckArgs` (broken from refactor `5049c2c2`); fix existing `csha_check_report_cli.rs` |
| FFI test-file fix | LANDED `b65f9d77` | None (closed) |
| C17-cleanup `%rd_dk_*` rename | DEFERRED (T2 touched the same surface but architectural change took priority) | Cosmetic rename in c18 |

### Cycle-5 invariant disposition (final)

**Section 5.3 numerical evidence NOT CLAIMABLE.** Paper §6.3 49% headline: **STRUCTURAL
PARTIAL, NUMERICAL UNVERIFIED on Blackwell** (unchanged from c16). R0 retirement HOLDS
unconditionally.

The cycle-17 net advance is:
- DEAD-CODE MASK EMPIRICALLY CLOSED (verified by T1 + T2)
- T2 ARCHITECTURALLY SOUND (verified by T3 POST memcheck + racecheck clean)
- NEW DEFECT(S) STRUCTURALLY IDENTIFIED (emit_dproj post-loop dK SMEM dependency) FOR c18

### Meta-lessons codified for c18

1. **"Structural witness passing does NOT empirically validate" — REVERIFIED.** Phase A
   converged on H2+H3 at confidence 82. T2 implemented the predicted fix. T3 POST sanitizer
   clean. Yet dk max_rel = 21.46 ≠ tolerance. The architectural fix was real and correct;
   the bug was deeper than the convergent hypothesis. C18 must independently re-verify the
   dK chain end-to-end.

2. **Pre-existing CI-broken tests are technical debt that masks future regressions.** The
   `csha_check_report_cli.rs` test broke when refactor `5049c2c2` removed `--csha-report`
   from `CheckArgs`. No one noticed because the test was already not exercised. T4
   surfaced this. The arch-hardening campaign should add a "broken test census" to prevent
   this class of silent debt.

3. **Independent reviewer second-guessing IS valuable — but reviewers can be wrong too.**
   T3 PRE subagent's STOP recommendation was based on a misreading (assumed memory-safety
   bug; actual is correctness bug — dead-code mask). Orchestrator override was sound
   because the PTX-level dead-code-mask analysis was concrete (line numbers, predicate
   logic). Convergent multi-investigator signal is the right bar for confidence, not
   single-reviewer veto.

4. **Defect surfacing during incremental fixes is GOOD.** T2's per-iter zero-init made the
   `emit_dproj` post-loop dK SMEM dependency observable. Cycle 16's dead-code mask had
   hidden it. This is forward progress even though Bug 1 isn't closed yet — each cycle
   narrows the suspect surface area.

---

## Cycle 18 (2026-06-30): VERIFICATION-GAIN + T4 CLI restoration + DEGENERATE-PROBE meta-lesson

### Disposition: NO NUMERICAL CLOSURE; meta-lesson value preserved via probe-gated discipline

Cycle 18 ships ONE production-code-changing commit (T4 `--csha-report` CLI restoration)
and a substantial set of EMPIRICAL FINDINGS from Phase A.2 probe gating. **Zero numerical
closure** on dq/dk/dv. Paper §6.3 stays STRUCTURAL PARTIAL NUMERICAL UNVERIFIED (unchanged
from c17). The cycle's net advance is verification-gain: c17's untested hypothesis chain
collapsed under empirical probing, refining the c19 target with measurable rigor.

### Phase A — 3-chain investigation (5 parallel investigators)

| Chain | Verdict | Confidence | Recommendation |
|---|---|---|---|
| dK post-T2 (CHAIN 1) | WEAKLY_SUPPORTED | 68 | Option (c) dk_preRoPE_scratch HBM RMW |
| dq collapse (CHAIN 2) | WEAKLY_REFUTED stated + 82 alternative | 82 (on alternative) | %scale uninit OR dS=0 upstream |
| dV catastrophic (CHAIN 3) | WEAKLY_REFUTED stated | 55 | Side-channel oracle BEFORE codegen |
| G16-3 sanitizer | sm_120 sanitizer REFUTED c16 hypothesis | — | New site at forward kernel +0xB30 |
| `--csha-report` flag | mechanical restoration from `6bc2ffa1` | — | Ship pure restore (no new logic) |

Synthesizer's NET LEADING CONFIDENCE was 62 — **correctly applying cycle 17 meta-lesson**.
All chains gated through Phase A.2 empirical probes BEFORE production codegen. Strategy
C-prime ("no ship and hope") explicitly chosen over Strategy A (3-in-one) per cycle 17
honesty.

### T1 dq debug-store probe — outcome (b) b_dS_zero (REVERTED before commit)

**Probe scratch dump:**
- `%scale = 1.250000000e-1` (bits `0x3e000000`) = 1/sqrt(64) — CORRECT
- `%f_dS = 1.508969477e-10` (bits `0x2f25e9b7`) — f32 noise floor (~0)
- `%f_dq_0 = 0.000000000e0` — exactly zero

**Phase A 82-confidence hypothesis (`%scale` uninitialized) EMPIRICALLY REFUTED.** This is
the value of probe-gating: the c18 PRELUDE fix path that would have been taken under
Strategy A is empirically wrong; a placebo commit was avoided.

**HOWEVER, R11 review refined the interpretation:** The probe at `(row=0, col=0,
causal=true)` is a **STRUCTURALLY DEGENERATE ZERO BY ARITHMETIC IDENTITY**. At position
(0,0) with causal masking, the causal window has exactly 1 valid key, so softmax gives
`P[0,0]=1.0` exactly. Then `D = sum_k P*dP = 1*dP[0,0] = dP[0,0]`, and
`dS = P*(dP - D) = 1*(dP - dP) = 0` — by arithmetic identity, NOT a ds_compute bug.

**R11's verdict:** the probe machinery is correctly instrumented (post-init, post-compute,
right lane) — but the SAMPLED COORDINATES are at a structurally degenerate point. The
`dS≈0 → ds_compute is broken` interpretation is NOT supported by this measurement.

**R1+R3 review additionally BLOCKED the probe ship** on 5 compilation-breaking FFI
integrity issues (Confidence 100):
1. `builtins.rs` missing trailing `dbg_scratch_ptr` Cranelift sig entry
2. `wengert_lower.rs` AD-path call missing trailing null
3. `pca_rope_ffi_sentinel.rs` both coercions still 54-param (won't compile)
4. `csha_cycle15_bug1_ablations.rs` direct call missing 6th trailing zero
5. `pca_backward_kernel_snapshot` insta snapshots stale (probe emitted PTX unconditionally)

**Decision: REVERT T1 probe.** Both reviewers converge against shipping — R1+R3 on compile
break, R11 on degenerate-position semantics. Probe + T5 PTX dump example stashed in c18
worktree (`git stash`) for c19 reproducibility. C19 must re-implement with proper
feature-gating AND probe at a non-degenerate position (row ≥ 1 or `causal=false`).

### T3 dV GPU side-channel oracle — REAL_BUG classification CONFIRMED by R11

**Worst-cell evidence:**
- j=284 d=37: kernel=-9.105e+02 vs ref=+1.870e-05 (`rel_err=4.869e+07`)
- j=284 d=56: kernel=+1.025e+03 vs ref=+4.355e-05 (`rel_err=2.354e+07`)
- j=202 d=34: kernel=+6.980e+03 vs ref=+1.489e-03 (`rel_err=4.687e+06`)
- 7-order-of-magnitude discrepancy at finite (non-saturated) values

**R11's verdict:** REAL_BUG VALID. The reference oracle (`csha_reference.rs:340-349`)
correctly computes `d_v[j,d] += sum_i P[i,j] * dO[i,d]` using f32 arithmetic on properly
f16-rounded inputs. The worst-cell pattern at multiple rows (j=284, 202, 158) with finite
wrong values cannot be explained by tolerance drift or oracle mismatch.

The `causal=false` ±Inf saturation at j=299 is a SEPARATE phenomenon (known f16 storage
limit during final scratch→dv_ptr conversion) and is filtered from the worst-cell
analysis. The causal=true finite-but-wrong cells are independent solid evidence.

**Recommendation:** ESCALATE to c19 with HIGH confidence. C19 dV investigation has solid
empirical basis.

### T5 G16-3 forward-PTX disassembly — REFACTOR_NEEDED (DEFER_C19)

c16 hypothesis (Phase-4 backward SMEM overshoot into rms_strip at d_model=0) **REFUTED**
by sm_120 compute-sanitizer triage. New site surfaced: **1024 violations at FORWARD
kernel offset +0xB30, pattern `0x5 + 0xA*lane` (all ODD = misaligned for f16 reads)**.
dq/dk/dv came back zero; dwq/dwk/dwv/dx all FAIL — forward never wrote valid Q-proj
outputs.

T5 disassembly localized:
- Primary candidate: `ld.shared.b16 %h_save_v, [%rd_save_smem]` at PTX line 358 (and
  parallel sites 370, 394, 406, 626, 638) — Q/K/V save SMEM read in
  `emit_save_activations_subset`
- **Secondary correctness bug surfaced:** `q_load.rs:201` — `fma.rn.f32 %f{reg}, %f{reg},
  %f0, %f1` IGNORES the shuffled RoPE partner `%f2`; non-fused-projections inline RoPE
  computes `q*cos+sin` instead of `q*cos+partner*sin`. Independent c19 finding.

**Fix scope: REFACTOR_NEEDED.** Orchestration self-admission at `mod.rs:374-384`:
"Backward numerical correctness for non-fused path is NOT the goal of this edit."
Recommend DEFER_C19 with refined hypothesis.

### T4 — CLI `--csha-report` restoration LANDED (commit `a89432f4`)

Pure mechanical restoration from commit `6bc2ffa1`. Refactor commit `5049c2c2` had removed
`--csha-report` and `--csha` from `CheckArgs`, breaking
`crates/nsl-cli/tests/csha_check_report_cli.rs` (4 failing tests). Restoration:

- Added `csha: Option<String>` + `csha_report: bool` to `args::CheckArgs` (mirrors
  `BuildArgs:622/626`)
- Restored Sprint-3 dispatch body in `commands/check.rs` (~95 LOC after wrga_compare,
  before cep_search)
- Uses `crate::pipeline::analysis_to_csha_configs` (the cycle-17 merge `pub(crate)`
  helper)
- 143 LOC total

**Test results:** 4/4 csha_check_report_cli tests GREEN; 2286/2286 nsl-codegen lib GREEN;
25/25 fa_v2_snapshots GREEN; 34/35 nsl-cli test files GREEN. The 1 failing file is the
pre-existing `calibration_flag_validation.rs` hardcoded-path issue (unrelated, predates
c18).

**R1+R3 verdict: APPROVE_SHIP.** No high-severity findings.

### Cycle 18 net disposition

| Work item | Cycle 18 outcome (commit SHA) | Cycle 19 carryover |
|---|---|---|
| T4 `--csha-report` restore | LANDED `a89432f4` (pure mechanical) | None (closed) |
| T1 dq probe | REVERTED (compile-block + degenerate-site) | Re-probe at row≥1 or causal=false; feature-gate the emission |
| Phase A 82-confidence `%scale` hypothesis | EMPIRICALLY REFUTED | `%scale = 0.125` correct; do NOT pursue prelude scale fix |
| ds_compute pivot (presumed from T1 dS=0) | NOT YET CONFIRMED (R11 flagged degenerate position) | Conditional on c19 re-probe showing dS≠0 at non-degenerate site |
| T3 dV catastrophic | REAL_BUG CONFIRMED (diagnostic only; no codegen change) | HIGH-confidence dV investigation in c19 |
| T5 G16-3 forward +0xB30 | REFACTOR_NEEDED (DEFER_C19) | Forward-kernel emit_save_activations_subset audit + q_load.rs:201 RoPE partner bug |
| q_load.rs:201 RoPE partner bug | SURFACED as side-finding by T5 | Independent c19 fix candidate |
| `--csha-report` carryover from c17 | LANDED (T4) | None (closed) |
| `csha_check_report_cli.rs` broken test | RESOLVED via T4 | None (closed) |
| %rd_dk_* cosmetic rename | STILL DEFERRED | c19+ |

### Cycle-5 invariant disposition (final)

**Section 5.3 numerical evidence NOT CLAIMABLE.** Paper §6.3 49% headline: **STRUCTURAL
PARTIAL, NUMERICAL UNVERIFIED on Blackwell** (UNCHANGED from c17). R0 retirement HOLDS
unconditionally.

C18 net advance is honest verification-gain:
- Phase A 82-confidence `%scale` hypothesis EMPIRICALLY REFUTED before any production
  change (probe-gated discipline worked)
- T3 dV REAL_BUG confirmed by independent R11 oracle audit
- T5 G16-3 hypothesis REFINED (forward kernel, not backward SMEM)
- T4 mechanical fix LANDED with R1+R3 APPROVE_SHIP
- ZERO over-claim; ZERO placebo commits; ZERO Phase A.2 outcome-blind production
  shipments

### Meta-lessons codified for c19

1. **Probe-gating WORKS exactly as cycle 17 specified.** Phase A predicted `%scale` uninit
   at confidence 82. T1 probe REFUTED in <2 hours. Production code unchanged. Compare to
   cycle 17 T2 which spent ~4 hours implementing a fix that was later empirically refuted.
   **Codify: Phase A.2 probe-gating is now MANDATORY for any future c19 hypothesis at
   confidence ≥70.**

2. **Probe semantics matter as much as probe instrumentation.** R11 found the T1 probe
   was correctly instrumented but sampled a structurally degenerate cell. The dS≈0
   measurement was VALID but UNINFORMATIVE — degenerate by arithmetic identity at
   `(row=0, col=0, causal=true)`. **Codify: every probe must explicitly justify its
   sampled coordinates as non-degenerate. R11-style oracle review is the gate.**

3. **Implementer "tests green" claims must be verified against the FULL test surface.**
   T1 subagent reported "probe_landed: true; tests green" but had only run the specific
   GPU test it cared about. R1+R3 found 5 compile-breaking issues across
   `builtins.rs`/`wengert_lower.rs`/`pca_rope_ffi_sentinel.rs`/`csha_cycle15_bug1_ablations.rs`/snapshot
   tests. **Codify: implementer reports MUST include `cargo check -p <crate> --tests`
   on ALL impacted crates + snapshot tests. Verifier-side independent test runs are the
   gate.**

4. **Multi-reviewer adversarial review catches what single-reviewer reviews miss.**
   R1+R3 caught compile-breaks (technical correctness). R11 caught degenerate-position
   sampling (semantic correctness). Either reviewer alone would have either rubber-stamped
   the technical fix and pivoted c19 to ds_compute (wasted weeks) OR blocked on compile
   issues but missed the semantic problem (compile-fix and ship a probe that doesn't
   actually measure what it claims). Both perspectives were necessary.

5. **Production-code FFI sig changes must include ALL 5 sites (cycle-17 carryover
   reverified).** The c17 phase2.6 ← main merge required ABI consistency across builtins/
   wengert_lower/runtime/pca_rope_ffi_decls/pca_rope_ffi_sentinel. The c18 T1 probe
   subagent attempted to widen the sig and missed 4 of the 5 sites. **Codify: any FFI sig
   widening checklist enforcement.**

### Cycle 19 prerequisites (carryover from c18)

1. **dS re-probe at non-degenerate coordinates** (R11 PRECONDITION): row ≥ 1 OR causal=false.
   Feature-gate the probe emission (`#[cfg(feature = "csha_cycle19_probe")]`) so snapshot
   tests are not affected. ALL 5 FFI sites updated atomically per c17 meta-lesson #5.
   Probe scratch values dumped, classified into 4-outcome decision tree.
2. **dV catastrophic investigation** (HIGH confidence): Trace the dV accumulation path
   from `dv_accum.rs` through finalize.rs emit_store_kv_only V-only RMW. Inspect for
   thread-id-specific or KV-iter-specific code paths that affect rows j=284/202/158.
   Compare to dwv path (which is mostly correct).
3. **G16-3 forward kernel investigation:**
   - emit_save_activations_subset (`csha_hooks_forward.rs:805-959`) SMEM read ordering
     under `(save_activations_for_backward && !fused_projections && d_model=0)` gate
   - q_load.rs:201 RoPE partner shuffle bug — independent fix candidate, single-line
     surgical likely
4. **T1 probe re-implementation** with proper feature-flag gating (preserves snapshot
   byte-identity)
5. **csha_check_report_cli.rs broken-test class:** consider arch-hardening campaign
   "broken-test census" sub-task
6. **%rd_dk_* cosmetic rename:** STILL deferred

## Cycle 19 (2026-07-01): T1 FFI variant-B scaffolding LANDED (PARTIAL) + T3 q_load RoPE partner surgical LANDED

### Scope

Cycle 19 delivered T1 dS-probe FFI scaffolding (PARTIAL — ABI variant-B + hygiene
tests; PTX emission deferred to c20) and T3 q_load.rs RoPE partner-shuffle surgical
fix (LANDED with RED-then-GREEN structural witness). T2 dV catastrophic trace and
T4 conditional codegen fix DEFERRED to c20 (blocked on T1 PTX emission).

Phase-A synthesizer (`wbeju206r`) ranked c18 carryover items and refined the FFI
site count from 5 (c18 belief) to 12 (R4 evidence). Recommended variant-B FFI
extension via a NEW `_probe` symbol rather than widening the existing 54-param
signature — leaves the 12 pre-c19 call sites byte-identical.

### T1 — dS probe FFI scaffolding (PARTIAL, ea8e0157 + 3 fixups → aa9402dc)

Variant-B: NEW `nsl_flash_attention_csha_backward_probe` FFI symbol behind the
`csha_cycle19_probe` Cargo feature (default OFF).

**Landed:**
- Cargo feature `csha_cycle19_probe` on both nsl-runtime and nsl-codegen
- Runtime `pub extern "C" fn nsl_flash_attention_csha_backward_probe` — 56 i64
  params (54 orig + `probe_ds_out_ptr` + `probe_dv_out_ptr`); delegates to the
  existing 54-param body verbatim
- Cranelift extern-decl added to `builtins.rs::declare_runtime_functions` under
  identical cfg; the RUNTIME_FUNCTIONS const is UNTOUCHED
- Additive 56-arity assertion + per-param `types::I64` type-lock in
  `pca_rope_ffi_decls.rs` — existing 54-arity assertion on the non-probe symbol
  is unchanged
- Additive typed-coercion sentinel in `pca_rope_ffi_sentinel.rs`
- Grep-hygiene test `csha_backward_ffi_hygiene.rs` locking down the 12 known
  pre-c19 call sites (would catch a hypothetical 13th site)
- `#[ignore]`d integration test `csha_cycle19_ds_probe.rs` with honest XFAIL
  rationale ("PTX-side probe emission deferred to c20")
- Fixups: stale allowlist entry deleted; sentinel-block comment rewritten;
  0.25 coefficient doc caveat added; per-param i64 lock

**Deferred (honest PARTIAL per c18 DEGENERATE-PROBE meta-lesson):**
- PTX-side probe emission: predicated `st.global.f32` writes populating the 8-slot
  layout `{row_max, row_sum, S_pre_mask, P, dP, rowsum_dP_P, dS, scale*dS}` at
  `(warp_id==1 && lane==0 && q_tile_iter==0 && batch_idx==0 && head_idx==0)`
- `PrimalOp::FusedCshaBackwardProbe` variant + wengert_lower dispatch
- Integration test body (currently `unimplemented!()` under `#[ignore]`)

**Gates green:** default features + `csha_cycle19_probe` feature both compile;
2286/2286 nsl-codegen lib tests; 25/25 fa_v2_snapshots byte-identical; W13
CshaSavePointers UNTOUCHED; all 12 pre-c19 FFI sites UNTOUCHED.

### T3 — q_load.rs RoPE partner-shuffle fix (LANDED, 899cbbe0 + d98508d6 + 84bded4a)

Non-CSHA inline forward path bug at `phases/forward/q_load.rs`. The RoPE emitter
loaded shuffled partner into `%f2` but the follow-up FMA read self (pre-shuffle)
and treated sin as additive bias, ignoring `%f2` entirely. HalfSplit `@%p0` and
`@!%p0` bodies were byte-identical (no sign flip).

**Fix:** Two-FMA-per-output encoding per csha_hooks.rs:1584-1595 reference math.
Adjacent even lane (holds x0) computes `x0*cos + (-partner)*sin = x0*cos - x1*sin`;
Adjacent odd lane (holds x1) computes `x1*cos + partner*sin = x1*cos + x0*sin`.
HalfSplit lane<16 mirrors even Adjacent; lane>=16 mirrors odd. Signs correctly
DIFFER between branches.

**Discipline:** RED-then-GREEN via structural PTX-content test (RED commit
first, then fix). R11 fixups tightened tests to symmetric branch coverage —
regression in EITHER branch of EITHER style now fails. Re-verified RED-then-GREEN
after tightening.

**Gates green:** 2286/2286 lib tests; 25/25 fa_v2_snapshots byte-identical (no
snapshot touches production `fused_projections=true` path); q_load.rs signature
unchanged (bypasses 12-site FFI atomicity concern); csha_hooks.rs UNTOUCHED.

**Deferred to c20:** Numerical GPU-vs-CPU witness (`max_rel_err < 1e-3`) —
requires wiring the non-CSHA v2 rope_q forward launcher at the Rust FFI level.
CPU oracle is in place; test is `#[cfg(feature = "cuda")]` + `#[ignore]`d.

Orthogonal defect discovered but NOT touched: inline cos/sin indexing may read
`cos[d]/sin[d]` where reference indexes by `d/2` (pair index). Preserving pre-c19
behavior; audit deferred to c20.

### Adversarial review (3-lens, wqm5uiw24)

- R1 correctness: APPROVE_SHIP both
- R3 codegen: APPROVE_SHIP both
- R11 semantic: APPROVE_WITH_FIXUPS T1, APPROVE_SHIP T3 (fixups applied)

Zero HIGH-severity findings. All medium findings addressed in Phase E fixups.

### §6.3 status

**UNCHANGED:** STRUCTURAL PARTIAL NUMERICAL UNVERIFIED. Cycle 19 did not advance
numerical closure of dq/dk/dv (T1 PTX emission and T2 dV trace both deferred).
Cycle 19 DID advance FFI hygiene, feature-flag isolation pattern, and non-CSHA
forward path structural correctness.

### Meta-lessons codified (extend from c18)

1. **Variant-B FFI extension** (new `_probe` symbol) is the safe pattern for adding
   probe outputs — leaves the pre-c19 call sites byte-identical, sidesteps R3
   Cranelift sig-mismatch panic risk.
2. **Grep-hygiene test** locks the FFI call-site landscape; catches hypothetical
   13th caller before merge.
3. **Honest PARTIAL** (scaffolding + XFAIL body) is superior to shipping a green
   test with trivially-satisfied assertions — matches c18 DEGENERATE-PROBE lesson.
4. **Symmetric branch coverage** in RED-then-GREEN tests: assert BOTH predicate
   branches of a split emitter carry the fix, not just one. R11 caught the
   asymmetric-coverage risk in T3 review; fixups tightened via
   `lines_under_pred` + `line_consumes_partner` helpers.
5. **Register-pool discipline:** new registers must fall within pre-declared
   pool ranges (`%f<48+head_dim/32>`, `%p<8>`, `%r<16>`) — verified downstream
   emitters do not clobber before use.

### Carryover to c20

1. **T1 PTX emission** — extend variant-B FFI with actual `st.global.f32` probe
   writes at the 8 sample sites in `ds_compute.rs` + `dqdk_accum.rs`; populate
   integration test body
2. **T2 dV catastrophic trace** — use T1 probe scaffold once populated; audit
   `emit_drope_branch(K)` SMEM footprint; classify into 4-branch outcome tree
3. **T4 conditional codegen fix** — evidence-driven fix based on T1+T2 classifications
4. **T3 numerical witness** — wire non-CSHA v2 rope_q forward launcher at FFI
   level; flip Adjacent/HalfSplit GPU-vs-CPU tests from `#[ignore]` to enforced
5. **T3 orthogonal:** audit inline cos/sin indexing (`d` vs `d/2`)
6. **G16-3 forward:** `emit_save_activations_subset` SMEM ordering (deep triage)
7. **csha_check_report_cli.rs broken-test census** (arch-hardening scoping)
8. **%rd_dk_* cosmetic rename** — deferred cycles 16+17+18+19

