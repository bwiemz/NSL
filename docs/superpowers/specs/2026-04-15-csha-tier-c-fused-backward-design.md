# CSHA Tier C — Fused Source-AD Backward (Numerical Verification) — Design

**Date:** 2026-04-15
**Branch:** `feat/csha-tier-c` (new worktree to be created)
**Prerequisite:** Tier A (PR #36 merged) + Track B (head_dim=128 launch fix, PR pending)
**Owner:** single-session cycles, ~4-5 sessions total

## 1. Purpose

Deliver a CSHA-fused backward kernel that produces
`{dQ, dK, dV, dWq, dWk, dWv, dx}` in a single CUDA launch when the AD
dispatcher encounters a CSHA-claimed forward chain. Matches both the
existing unfused GPU backward and a new CPU reference within the
tiered-tolerance envelope established in Tier A.

The fused backward replaces 6+ separate kernel launches (one per
gradient) with a single kernel that reuses SMEM activation tiles and
eliminates the HBM round-trips between adjoint operations. This closes
the training-throughput gap: today every training step on CSHA-enabled
configs does forward on GPU but backward on a non-fused path that is
10-100× slower.

## 2. Background

### State on `origin/main` at design time (post Tier A + Track B)

- Tier A forward fully fused and numerically verified across the §4
  matrix including `head_dim=128` via Track B's dynamic SMEM opt-in.
- Backward today: `crates/nsl-runtime/src/flash_attention.rs:917`
  hosts `flash_attention_backward_gpu`, which is non-fused — each
  gradient (dQ, dK, dV, dWq, dWk, dWv) goes through separate matmul
  and softmax-Jacobian launches.
- Source-AD pipeline (`source_ad.rs`, `ad_rules.rs`, `wengert.rs`)
  walks primal Wengert ops in reverse topological order and emits
  one adjoint rule per op.
- `csha::FusionMark` records which ops belong to a CSHA-fused forward
  chain; today only used to suppress double-emission of forward ops
  that CSHA's fused path already handles.

### Why fuse backward

Training step wall-clock is dominated by the backward pass on CSHA
configs. Non-fused backward re-reads Q_proj, K_proj, V_proj, and
intermediate activations from HBM for each of the six gradient
computations — the exact HBM-round-trip overhead CSHA's forward pass
was designed to eliminate. Fusing backward reuses the same SMEM
tiling contract forward already established.

## 3. Scope

### In scope

- Fused backward PTX emission for the CSHA chain
  `RMSNorm → Wq/Wk/Wv → RoPE(Q,K) → attention`.
- Forward-side activation save path (Q_proj, K_proj, V_proj, row_max,
  row_sum to HBM, gated on `@train` mode).
- AD dispatcher integration: route CSHA-claimed chains to the fused
  backward FFI, fall back to per-op adjoint rules when the backward
  validator rejects the config.
- CPU reference for backward gradients + three-way numerical gate
  (fused GPU, unfused GPU, CPU).
- SMEM-budget validator extended with `Direction::Backward`.

### Out of scope (deferred to follow-ups)

- Wo + residual backward — deferred with forward (§9a of Tier A).
  Backward kernel does not compute dWo or d(residual).
- Dead-head gradient elimination (Tier C description mentions this;
  pushed to a post-milestone optimization once base fused backward
  is proven correct).
- Tile-granular checkpointing via `@checkpoint`. Separable by
  definition — changes when activations are recomputed, not what
  gradients are produced.
- Performance benchmarking and speedup measurement. Separate
  validation axis; runs against the same kernel after correctness
  lands.
- `block_q != block_kv` asymmetric configs (blocked for forward per
  the existing follow-up; backward inherits the same constraint).

## 4. Supported configuration matrix

Base matrix (same as Tier A forward post-Track B):

| Dim         | Values                              |
|-------------|-------------------------------------|
| head_dim    | {32, 64}                            |
| head_dim=128| only with d_model=32 (Track B cap)  |
| block_q     | {32, 64}                            |
| block_kv    | {32, 64} (symmetric only)           |
| causal      | {true, false}                       |
| heads       | {4, 8}                              |

Per-config admission: `validate_scalar_v2_config(config, Direction::Backward)`
computes backward's SMEM total (forward tiles + dQ_tile + dK_tile +
dV_tile + recomputed P_tile). Configs that fit forward but exceed the
99 KB hardware cap on backward fall through to AD's per-op adjoint
path with a diagnostic naming the exact byte excess.

## 5. Architecture

### 5.1 Module structure

**Phase 0 refactor** (single commit, zero logic change):
```
flash_attention_v2/phases/<file>.rs  →  flash_attention_v2/phases/forward/<file>.rs
```
Plus new `phases/backward/` added in later phases:
```
flash_attention_v2/
  mod.rs                          # orchestrator: direction branch
  smem_layout.rs                  # shared, Direction-parameterised validator
  register_budget.rs              # shared
  phases/
    mod.rs                        # re-exports forward::* + backward::*
    forward/                      # moved from phases/ in Phase 0
      prelude.rs q_load.rs s_compute.rs softmax.rs pv_accum.rs
      finalize.rs csha_hooks.rs
    backward/                     # new in Phase 3
      prelude.rs q_load.rs ds_compute.rs dv_accum.rs dqdk_accum.rs
      csha_hooks_backward.rs finalize.rs
```

### 5.2 Save-activations contract (Phase 1)

Forward writes five tensors to HBM when `csha.save_activations_for_backward=true`:

| Tensor      | Shape                                | Dtype | Purpose                          |
|-------------|--------------------------------------|-------|----------------------------------|
| Q_proj      | [batch, heads, seq, head_dim]        | f16   | Q tile post-RMSNorm+proj+RoPE    |
| K_proj      | [batch, heads, seq, head_dim]        | f16   | K tile post-RMSNorm+proj+RoPE    |
| V_proj      | [batch, heads, seq, head_dim]        | f16   | V tile post-RMSNorm+proj         |
| row_max     | [batch, heads, seq]                  | f32   | softmax row maximum              |
| row_sum     | [batch, heads, seq]                  | f32   | softmax normalizer               |

`row_max` and `row_sum` stored in f32 to guarantee backward's
`P = exp(S - row_max) / row_sum` recomputation produces P numerically
identical to forward's P (f16 storage would drift). Gated on a
compile-time flag set by the compiler when a `@train` block is active;
inference builds pay zero HBM cost.

### 5.3 Backward computation (Phase 3)

Per-tile backward pipeline (mirrors forward's split-loop structure):

```
[PV-style loop over KV tiles]
  load Q_proj tile, dO tile from HBM
  recompute P = exp(S - row_max) / row_sum using saved stats
    (with causal mask applied before recomputation, same helper as forward)
  dP = dO @ V^T
  dS = P * (dP - rowsum(dP * P))
  dV += P^T @ dO                (in SMEM accumulator)
  dQ += dS @ K                  (in SMEM accumulator)
  dK += dS^T @ Q                (in SMEM accumulator)
[end tile loop]

[output-projection-style final pass per q_tile_iter]
  dRoPE: rotate dQ_tile and dK_tile in registers
  dproj: dWq += x_norm^T @ dQ_proj, dWk += x_norm^T @ dK_proj,
         dWv += x_norm^T @ dV_proj (per-lane accumulation via
         Tier A's warp-per-row contract, run in reverse)
  dRMSNorm: chain-rule back through the RMSNorm to produce dx
  store dQ, dK, dV, dWq, dWk, dWv, dx to global memory
```

### 5.4 AD dispatcher integration (Phase 5)

`FusionMark` extended with `backward_emitted: bool` (defaults false).
`ad_rules::dispatch_adjoint` match arm checks whether the current op
belongs to a claimed CSHA chain:

```rust
if let Some(chain) = csha_claimed_chain_for(op) {
    if chain.backward_emitted { return Ok(()); }  // already emitted
    match validate_scalar_v2_config(chain.config, Direction::Backward) {
        Ok(()) => {
            emit_fused_backward_call(chain);
            chain.backward_emitted = true;
            register_output_varids(chain);  // dx, dWq, ... now in adjoint map
        }
        Err(e) => {
            diagnostics.push(format!(
                "CSHA fused backward rejected: {} bytes > {} byte cap at (block_q={}, head_dim={}); falling back to unfused backward",
                e.actual, e.limit, chain.config.block_q, chain.config.head_dim
            ));
            // fall through to per-op adjoint rules
        }
    }
} else {
    // fall through to per-op adjoint rules
}
```

The reverse-walk invariant (confirmed by design Q3): "first claimed
op encountered in reverse walk = chain's output op" means the fused
backward call naturally fires before any upstream adjoint needs the
chain's gradient outputs. Debug-asserted in the dispatcher itself.

### 5.5 Three-way numerical gate (Phase 6)

Per matrix row:
```
forward(fused, save_activations=true)
  → O, Q_proj, K_proj, V_proj, row_max, row_sum

dO = det_seq(seed)

grads_fused   = fused_backward(dO, saved_activations, weights, x)
grads_unfused = flash_attention_backward_gpu(dO, Q, K, V, weights, x)
grads_cpu     = csha_reference_backward(inputs, dO)

for each g in {dQ, dK, dV, dWq, dWk, dWv, dx}:
    assert max_abs(grads_fused[g]   - grads_cpu[g]) < tol_tier(head_dim)
    assert max_abs(grads_unfused[g] - grads_cpu[g]) < tol_tier(head_dim)
    report max_abs(grads_fused[g]   - grads_unfused[g]) as diagnostic
```

Tier C adds a P-recomputation assertion for one causal config:
`max_abs(P_recomputed_in_backward - P_forward) < 1e-6`. The backward
kernel writes P to a debug buffer gated on a test-only flag.

### 5.6 Fallback discipline (Phase 7)

One regression test selects a config known to pass forward validation
but fail backward validation. Compiles a training snippet, asserts
the compiled Wengert adjoint count equals the sum of per-op adjoint
rules (not 1, which would indicate the fused call fired despite
rejection). Asserts the exact SmemBudgetExceeded diagnostic fires in
the compiler warning channel.

## 6. Work decomposition and dependencies

```
P0 (refactor) → P1 (save activations) → P2 (backward validator)
    ↓
P3 (backward phases) → P4 (orchestrator) → P5 (AD dispatcher)
    ↑
P6.1 (CPU reference) ──→ P6.2 (golden tests) ──→ P6.3 (3-way gate after P5)
    ↓
P7 (fallback test after P5)
```

P6.1 starts in parallel with P3 (per design feedback), serving as
ground truth for phase-level debugging during P3's softmax Jacobian
and gradient accumulation work.

## 7. Testing strategy

### Unit tests
- P1.1: `save_activations_for_backward` default false; forward PTX
  gains save emissions only when true.
- P2.2: backward validator produces diagnostic with exact byte
  numbers; accepts at least `(block_q=32, block_kv=32, head_dim=32)`.
- P3.*: snapshot + label-uniqueness tests per backward phase.
- P5: `FusionMark::backward_emitted` toggles correctly; dispatcher
  emits fused call exactly once per chain on first reverse-walk hit;
  fallback path fires on validator rejection.
- P6.1/P6.2: CPU backward golden-value test, 4×4 identity weights,
  tolerance 1e-6.

### ptxas validation
- New `direction=Backward` configs clean on sm_75/sm_90/sm_120.
- Register-usage assertion per config (worst case: head_dim=64,
  heads=8).

### Activation-save smoke
- Forward with `save_activations=true` vs `save_activations=false`
  on identical input produces byte-identical O and LSE (rules out
  corruption from the save path).

### GPU three-way gate (parametric sweep)
- Every matrix row: fused-GPU, unfused-GPU, CPU all agree within
  tolerance tier.
- Causal P-recomputation assertion on one dedicated row.

### AD fallback test (P7)
- Forward-accepts-backward-rejects config compiles to per-op adjoints
  with the exact diagnostic string in warnings.

### Regression coverage
- Full Tier A forward matrix stays green at the same max_abs values.
- Part 1 classic path stays green.
- `cargo test -p nsl-codegen --lib` count ≥ 1520 at milestone end
  (baseline 1515 after Track B).

## 8. Risks

**R1. Softmax Jacobian sign/transpose bugs.**
`dS = P * (dP - rowsum(dP * P))` is easy to get wrong. Mitigation:
CPU reference written before P3.3 (per dependency graph); golden
values pin exact expected dS shape on 2×2 attention; per-phase
snapshot tests catch PTX-level defects before launch.

**R2. Saved activation corruption masking forward numerics.**
Post-RoPE HBM writes must not race with attention body's SMEM
consumption. Mitigation: `bar.sync 0` fence after RoPE epilogue,
before any cooperative HBM write. P1.5 smoke asserts forward output
unchanged between `save_activations={true,false}`.

**R3. AD dispatcher emits fused call on non-output op.**
If FusionMark lists ops in non-topological order, the fused call
fires mid-chain and upstream VarIds aren't populated. Mitigation:
P5.1 unit test with handcrafted FusionMark in both orders;
`debug_assert!` in dispatcher verifying the processed op is the
chain's topological tail.

**R4. Validator accepts configs ptxas then rejects.**
`validate_scalar_v2_config` computes SMEM bytes; ptxas may reject
for register pressure unrelated to SMEM. Mitigation: ptxas validation
as first-class P3 gate; validator wraps ptxas as a "second opinion"
pass.

**R5. `dO` memory-layout mismatch.**
Training pipeline's `dO` layout may differ from what the fused
backward expects. Mitigation: P5.3 registers VarIds with explicit
shape/dtype metadata; three-way gate uses identical `dO` across all
paths — layout mismatch surfaces as order-of-magnitude divergence.

**R6. Causal mask inconsistency.**
Backward's `S` recomputation must apply the causal mask before
comparison to `row_max`. Mitigation: dedicated P-recomputation
assertion on a causal config; causal-mask emission lifted to a shared
helper used by both forward s_compute and backward ds_compute.

**R7. Phase 0 refactor breaks downstream imports.**
Cross-crate tests may reach into specific submodule paths.
Mitigation: refactor commit is a pure rename; `cargo check --workspace`
+ `cargo test --workspace --lib` gates every import; `phases/mod.rs`
re-exports forward via `pub use forward::*` to preserve paths.

## 9. Exit criteria

1. Fused backward matches both unfused GPU and CPU references within
   tolerance tiers across every accepted matrix config.
2. AD dispatcher routes CSHA-claimed chains to fused backward on
   validator accept; falls back to per-op adjoint rules on validator
   reject with a diagnostic containing the exact `actual` vs `limit`
   byte numbers and the `(block_q, head_dim)` triple.
3. `save_activations_for_backward` flag gated on `@train` — inference
   builds pay zero extra HBM cost (verified by a snapshot test
   asserting no additional `st.global.*` emissions in forward PTX
   when flag is false).
4. Tier A forward matrix stays green at the same max_abs values post
   Phase 1 and Phase 4 changes.
5. Three-way agreement smoke: at least one shared config reports all
   three paths within tolerance of each other.

## 10. Follow-ups (documented now, separate milestones later)

- Dead-head gradient elimination — conditional zero-out of gradient
  accumulators for heads marked inactive by weight-aware analysis.
- Tile-granular `@checkpoint` integration — recomputation discipline
  for activation save memory vs compute trade.
- Performance benchmarking — end-to-end training step wall-clock on
  coder500m / coder1b to quantify the fusion speedup.
- `block_q ≠ block_kv` asymmetric tile support (forward follow-up
  unblocks backward too).
- Wo + residual backward when forward Wo fusion ships (streamed
  weight tiles or dynamic SMEM variant).
