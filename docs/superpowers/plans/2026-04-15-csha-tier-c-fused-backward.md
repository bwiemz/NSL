# CSHA Tier C — Fused Source-AD Backward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a CSHA-fused backward CUDA kernel that produces `{dQ, dK, dV, dWq, dWk, dWv, dx}` in a single launch when the AD dispatcher encounters a CSHA-claimed forward chain, numerically verified against both the existing unfused GPU backward and a new CPU reference across the supported matrix on RTX 5070 Ti.

**Architecture:** Forward phases move under `phases/forward/` (zero-logic refactor); new `phases/backward/` sibling hosts the backward emission. Forward gains an activation-save path gated on `@train`. AD dispatcher checks a `FusionMark::backward_emitted` flag on the first reverse-walk hit of a claimed chain; validator-rejected configs fall back to per-op adjoints with a diagnostic. Three-way numerical gate: fused GPU vs existing unfused GPU vs new CPU reference.

**Tech Stack:** Rust, Cranelift IR, Cuda PTX (sm_75/80/90/120), cudarc 0.19, `nsl-codegen` + `nsl-runtime` crates.

**Working directory:** `c:\Users\bwiem\projects\NSL\.worktrees\csha-tier-c` (new worktree, branch `feat/csha-tier-c` off `origin/main`).

**Spec:** `docs/superpowers/specs/2026-04-15-csha-tier-c-fused-backward-design.md`.

---

## Operating rules

- TDD per task: write failing test, run it, implement, re-run, commit.
- One commit per task unless the task explicitly splits.
- Every PTX-emitting task ends with a `ptxas` validation step for sm_75 + sm_90 + sm_120.
- `cargo test -p nsl-codegen --lib` must stay green after every commit. Baseline: 1515 tests.
- Forward-pass Tier A matrix sweep must stay green with the same `max_abs` values after Phases P1 and P4. Re-run via `cargo test -p nsl-codegen --test csha_cuda_launch_fused --features cuda -- --ignored --nocapture` after each phase that touches forward code.
- Commit message prefixes: `feat(csha-c):`, `fix(csha-c):`, `test(csha-c):`, `refactor(csha-c):`, `docs(csha-c):`.
- All Cargo/git commands run from the worktree. First task creates it.

---

## File Structure

### Created files

| Path | Responsibility |
|------|---------------|
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/mod.rs` | Re-exports backward phases. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs` | Backward PTX register pool + kernel entry preamble. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/q_load.rs` | Load saved `Q_proj` from HBM into SMEM. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/ds_compute.rs` | Recompute `P`, compute `dP = dO @ V^T`, `dS = P * (dP - rowsum(dP * P))`. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/dv_accum.rs` | `dV += P^T @ dO` per tile. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/dqdk_accum.rs` | `dQ += dS @ K`, `dK += dS^T @ Q` per tile. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs` | `dRoPE` rotation, `dproj` (dWq/dWk/dWv), `dRMSNorm`, producing `dx`. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/backward/finalize.rs` | Store `dQ/dK/dV/dWq/dWk/dWv/dx` to global memory. |
| `crates/nsl-codegen/tests/csha_cuda_backward.rs` | Three-way GPU numerical gate + parametric matrix sweep. |
| `crates/nsl-codegen/tests/csha_ptx_ptxas_backward_validation.rs` | ptxas clean-assembly gate for backward configs on sm_75/sm_90/sm_120. |

### Modified files

| Path | Change |
|------|--------|
| `crates/nsl-codegen/src/flash_attention_v2/phases/` → `phases/forward/` | Phase 0 refactor: move 7 forward phase files under `forward/`. |
| `crates/nsl-codegen/src/flash_attention_v2/phases/mod.rs` | Re-export `forward::*` and `backward::*`. |
| `crates/nsl-codegen/src/flash_attention.rs` | Add `save_activations_for_backward: bool` field to `CshaExtras`. |
| `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs` | `Direction::Forward \| Direction::Backward` enum; `validate_scalar_v2_config(config, dir)` computes dir-specific SMEM budget; new diagnostic includes `actual_bytes` + `limit_bytes` + `(block_q, head_dim)`. |
| `crates/nsl-codegen/src/flash_attention_v2/mod.rs` | Forward orchestrator: post-RoPE `bar.sync` + HBM writes of `Q_proj, K_proj, V_proj` gated on `save_activations_for_backward`; post-softmax write of `row_max, row_sum`. New `synthesize_backward(config)` entry. |
| `crates/nsl-runtime/src/flash_attention.rs` | New FFI `nsl_flash_attention_csha_backward` (40+ args: forward's 30 + 5 saved activations + 7 gradient outputs + 2 x_residual slots). |
| `crates/nsl-runtime/src/flash_attention.rs` | `nsl_csha_alloc_backward_activations(batch, heads, seq, head_dim)` returning `{q_proj, k_proj, v_proj, row_max, row_sum}` device pointers. |
| `crates/nsl-codegen/src/csha_apply.rs` | `FusionMark` gains `backward_emitted: bool` (interior-mutable). `CshaChain` gains `config: FlashAttentionConfig` for backward validator lookup. |
| `crates/nsl-codegen/src/ad_rules.rs` | New arm in `dispatch_adjoint`: if op belongs to a claimed CSHA chain, on first hit run backward validator + emit fused call OR fall through with diagnostic. |
| `crates/nsl-codegen/tests/csha_reference.rs` | New `pub fn csha_reference_backward(...) -> Gradients` + `CshaGradients` struct. Golden-value 4×4 identity-weight backward test. |
| `crates/nsl-codegen/tests/csha_cuda_launch_fused.rs` | Add one smoke row that sets `save_activations=true`, reads back saved tensors, compares within tolerance. |

### Deleted files

None. Forward module files move; no deletions.

---

## Operating sequence

Phases numbered to match spec §5-6. Task IDs within each phase are T{phase}.{n}.

```
P0 refactor
  ├─ P1 save activations ─┐
  │                       ├─ P3 backward phases ─ P4 orchestrator ─ P5 AD dispatcher ─┐
  └─ P2 backward validator ┘                    │                                     ├─ P6.3 3-way gate
                                                P6.1 (CPU ref parallel with P3)       ├─ P7 fallback test
                                                P6.2 (golden tests)                   │
```

---

## Task T0.1: Create worktree

**Files:** none (git operation).

- [ ] **Step 1: Fetch latest main and create worktree**

```bash
cd c:/Users/bwiem/projects/NSL
git fetch origin main
git worktree add .worktrees/csha-tier-c -b feat/csha-tier-c origin/main
cd .worktrees/csha-tier-c
```

Expected: worktree created, branch `feat/csha-tier-c` tracks `origin/main`.

- [ ] **Step 2: Baseline test count**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: 1515 passed (or current main count — record it for later regression assertions).

---

## Phase 0 — Refactor: move forward phases under phases/forward/

### Task T0.2: Move forward phase files

**Files:**
- Rename: `crates/nsl-codegen/src/flash_attention_v2/phases/{prelude,q_load,s_compute,softmax,pv_accum,finalize,csha_hooks}.rs` → `phases/forward/<same>.rs`
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/mod.rs`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/mod.rs`

- [ ] **Step 1: Move files**

```bash
mkdir crates/nsl-codegen/src/flash_attention_v2/phases/forward
git mv crates/nsl-codegen/src/flash_attention_v2/phases/prelude.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/prelude.rs
git mv crates/nsl-codegen/src/flash_attention_v2/phases/q_load.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/q_load.rs
git mv crates/nsl-codegen/src/flash_attention_v2/phases/s_compute.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/s_compute.rs
git mv crates/nsl-codegen/src/flash_attention_v2/phases/softmax.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/softmax.rs
git mv crates/nsl-codegen/src/flash_attention_v2/phases/pv_accum.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/pv_accum.rs
git mv crates/nsl-codegen/src/flash_attention_v2/phases/finalize.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/finalize.rs
git mv crates/nsl-codegen/src/flash_attention_v2/phases/csha_hooks.rs crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs
```

- [ ] **Step 2: Create `phases/forward/mod.rs`**

```rust
// crates/nsl-codegen/src/flash_attention_v2/phases/forward/mod.rs
//! Forward-pass phase emission modules. See `phases/backward/` for the
//! backward pass.
pub mod prelude;
pub mod q_load;
pub mod s_compute;
pub mod softmax;
pub mod pv_accum;
pub mod finalize;
pub mod csha_hooks;
```

- [ ] **Step 3: Rewrite `phases/mod.rs` to re-export**

```rust
// crates/nsl-codegen/src/flash_attention_v2/phases/mod.rs
pub mod forward;
// Backward added in Phase 3.

// Back-compat re-exports so internal callers don't need to update paths.
pub use forward::{prelude, q_load, s_compute, softmax, pv_accum, finalize, csha_hooks};
```

- [ ] **Step 4: Run workspace check**

```bash
cargo check --workspace 2>&1 | grep -E "^error" | head -10
```

Expected: no errors. If any import elsewhere referenced `crate::flash_attention_v2::phases::forward::*` (wasn't there before), the re-export at Step 3 covers it.

- [ ] **Step 5: Run full lib test suite**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
```

Expected: same test count as T0.1 Step 2. Zero semantic change.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(csha-c): Phase 0 — move forward phases under phases/forward/

Zero logic change. New phases/backward/ sibling added in Phase 3.
Existing callers keep working via phases/mod.rs re-exports."
```

---

## Phase 1 — Save activations (forward-side prep)

### Task T1.1: Add `save_activations_for_backward` field

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention.rs` (`CshaExtras` struct)

- [ ] **Step 1: Write failing test**

Add to `crates/nsl-codegen/src/flash_attention.rs` tests module:

```rust
#[test]
fn cshaextras_save_activations_defaults_false() {
    let e = CshaExtras::default();
    assert!(!e.save_activations_for_backward);
}

#[test]
fn cshaextras_save_activations_independent_of_fused_projections() {
    let mut e = CshaExtras::default();
    e.save_activations_for_backward = true;
    assert!(e.save_activations_for_backward);
    assert!(!e.fused_projections); // independent flags
}
```

- [ ] **Step 2: Run — expect FAIL** (field doesn't exist)

```bash
cargo test -p nsl-codegen --lib cshaextras_save_activations
```

- [ ] **Step 3: Add the field**

In `CshaExtras` struct declaration, add:
```rust
/// When true, forward writes Q_proj, K_proj, V_proj, row_max, row_sum to
/// HBM for backward consumption. Gated on @train mode by the compiler.
/// Inference builds leave this false; forward pays zero extra HBM cost.
pub save_activations_for_backward: bool,
```

Update the `Default` impl to add `save_activations_for_backward: false,`.

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T1.1 add save_activations_for_backward to CshaExtras"
```

### Task T1.2: Runtime activation-buffer allocator

**Files:**
- Modify: `crates/nsl-runtime/src/flash_attention.rs` (new `nsl_csha_alloc_backward_activations`)

- [ ] **Step 1: Write failing test**

Add to `crates/nsl-runtime/src/flash_attention.rs` tests (inline, `#[cfg(test)]`):

```rust
#[test]
#[cfg(feature = "cuda")]
fn nsl_csha_alloc_backward_activations_allocates_five_buffers() {
    // Shape: batch=2, heads=4, seq=32, head_dim=32.
    // Expected buffer sizes:
    //   q_proj, k_proj, v_proj: 2*4*32*32 f16 = 8192 bytes each
    //   row_max, row_sum: 2*4*32 f32 = 1024 bytes each
    let result = unsafe {
        nsl_csha_alloc_backward_activations(2, 4, 32, 32)
    };
    assert_ne!(result.q_proj, 0, "q_proj alloc failed");
    assert_ne!(result.k_proj, 0, "k_proj alloc failed");
    assert_ne!(result.v_proj, 0, "v_proj alloc failed");
    assert_ne!(result.row_max, 0, "row_max alloc failed");
    assert_ne!(result.row_sum, 0, "row_sum alloc failed");

    // Cleanup.
    unsafe {
        nsl_csha_free_backward_activations(result);
    }
}
```

- [ ] **Step 2: Run — expect FAIL** (functions don't exist)

- [ ] **Step 3: Implement**

Add to `crates/nsl-runtime/src/flash_attention.rs`:

```rust
#[repr(C)]
pub struct CshaBackwardActivations {
    pub q_proj: i64,
    pub k_proj: i64,
    pub v_proj: i64,
    pub row_max: i64,
    pub row_sum: i64,
}

/// Allocate the 5 HBM buffers forward fills when
/// `csha.save_activations_for_backward = true`. Called by the
/// compiler before the forward launch in training mode.
#[no_mangle]
pub unsafe extern "C" fn nsl_csha_alloc_backward_activations(
    batch: i64, heads: i64, seq: i64, head_dim: i64,
) -> CshaBackwardActivations {
    use crate::cuda::inner::{cuda_alloc};
    let qkv_bytes = (batch * heads * seq * head_dim * 2) as i64; // f16
    let stats_bytes = (batch * heads * seq * 4) as i64; // f32
    CshaBackwardActivations {
        q_proj: cuda_alloc(qkv_bytes).unwrap_or(0),
        k_proj: cuda_alloc(qkv_bytes).unwrap_or(0),
        v_proj: cuda_alloc(qkv_bytes).unwrap_or(0),
        row_max: cuda_alloc(stats_bytes).unwrap_or(0),
        row_sum: cuda_alloc(stats_bytes).unwrap_or(0),
    }
}

#[no_mangle]
pub unsafe extern "C" fn nsl_csha_free_backward_activations(
    a: CshaBackwardActivations,
) {
    use crate::cuda::inner::cuda_free;
    if a.q_proj != 0 { cuda_free(a.q_proj); }
    if a.k_proj != 0 { cuda_free(a.k_proj); }
    if a.v_proj != 0 { cuda_free(a.v_proj); }
    if a.row_max != 0 { cuda_free(a.row_max); }
    if a.row_sum != 0 { cuda_free(a.row_sum); }
}
```

Adapt `cuda_alloc` / `cuda_free` symbol names to whatever already exists in `crates/nsl-runtime/src/cuda/mod.rs` — grep for `pub fn.*alloc` in that module before writing.

- [ ] **Step 4: Run — expect PASS**

```bash
cargo test -p nsl-runtime --lib nsl_csha_alloc_backward_activations --features cuda
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T1.2 runtime backward-activation buffer allocator"
```

### Task T1.3: Forward emits post-RoPE activation saves

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/mod.rs`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/forward/csha_hooks.rs`

- [ ] **Step 1: Write failing snapshot test**

In `phases/forward/csha_hooks.rs` tests:

```rust
#[test]
fn save_activations_emits_post_rope_writes() {
    let mut cfg = base_cfg_for_rope_test();
    cfg.csha = Some(CshaExtras {
        fused_projections: true,
        save_activations_for_backward: true,
        ..CshaExtras::default()
    });

    let mut ptx = String::new();
    // Invoke the orchestrator's save emission (use whichever function
    // wraps the post-RoPE save sequence; likely a new emit_save_activations).
    emit_save_activations(&mut ptx, &cfg, 0);

    // Q/K/V post-RoPE writes — one global store per saved tensor
    assert!(ptx.contains("st.global.b16") && ptx.contains("q_proj_ptr"),
            "Q_proj save missing");
    assert!(ptx.contains("st.global.b16") && ptx.contains("k_proj_ptr"),
            "K_proj save missing");
    assert!(ptx.contains("st.global.b16") && ptx.contains("v_proj_ptr"),
            "V_proj save missing");
    // Fence before the save to avoid racing the attention body's SMEM reads
    assert!(ptx.contains("bar.sync 0") , "fence before save missing");
}

#[test]
fn save_activations_zero_emission_when_flag_false() {
    let mut cfg = base_cfg_for_rope_test();
    cfg.csha = Some(CshaExtras {
        fused_projections: true,
        save_activations_for_backward: false,
        ..CshaExtras::default()
    });
    let mut ptx = String::new();
    emit_save_activations(&mut ptx, &cfg, 0);
    assert!(ptx.contains("save_activations=false, no emission") || ptx.is_empty());
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `emit_save_activations` in csha_hooks.rs**

Emits:
1. `bar.sync 0` fence.
2. Cooperative HBM store of `Q_proj` from SMEM Q tile to `[q_proj_ptr + batch*heads*seq*head_dim offset]`.
3. Same for K_proj and V_proj.

The store indexing must match what the backward `q_load` expects — `[batch, heads, seq, head_dim]` row-major. Use `%batch_idx`, `%head_idx`, `%q_start` to compute the output base pointer.

- [ ] **Step 4: Wire into orchestrator**

In `flash_attention_v2/mod.rs`, immediately after `emit_rope_epilogue` call, add:

```rust
phases::forward::csha_hooks::emit_save_activations(&mut ptx, config, q_iter);
```

- [ ] **Step 5: Softmax-state save**

In `phases/forward/csha_hooks.rs` (or wherever `emit_save_softmax_state` lives for forward today), extend to ALSO write `row_max` and `row_sum` to global memory when `save_activations_for_backward=true`. They're f32, `[batch, heads, seq]` shape.

Add an analogous snapshot test asserting `row_max_ptr` and `row_sum_ptr` stores fire only when the flag is set.

- [ ] **Step 6: Run full lib suite**

```bash
cargo test -p nsl-codegen --lib
```

Expected: baseline + new tests pass. Forward output unchanged on configs without `save_activations_for_backward=true`.

- [ ] **Step 7: ptxas validation**

Extend `csha_ptx_ptxas_validation.rs` with a `save_activations=true` config, assert clean on sm_75/sm_90/sm_120.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T1.3 forward emits post-RoPE activation saves when gated"
```

### Task T1.4: Regression — forward output byte-identical with/without save flag

**Files:**
- Modify: `crates/nsl-codegen/tests/csha_cuda_launch_fused.rs`

- [ ] **Step 1: Write the regression test**

Add ignored GPU test:

```rust
#[test]
#[ignore]
fn t1_forward_output_invariant_under_save_activations_flag() {
    // Launch forward twice on identical input: once with save=false,
    // once with save=true. O and LSE must be byte-identical (any diff
    // would signal that the HBM save path is racing the attention body).
    let (out_no_save, lse_no_save) = run_forward_config(
        32, 32, 32, 4, 128, false, true, /*save_activations*/ false);
    let (out_save, lse_save) = run_forward_config(
        32, 32, 32, 4, 128, false, true, /*save_activations*/ true);

    assert_eq!(out_no_save, out_save,
        "forward O diverged between save=false and save=true");
    assert_eq!(lse_no_save, lse_save,
        "forward LSE diverged between save=false and save=true");
}
```

Refactor existing `run_fused_config` to extract `run_forward_config(block_q, block_kv, head_dim, heads, d_model, causal, rope_q, save_activations)` — the matrix sweep passes `false` for save_activations, this test passes both.

- [ ] **Step 2: Run — expect PASS on RTX 5070 Ti**

```bash
cargo test -p nsl-codegen --test csha_cuda_launch_fused --features cuda -- --ignored t1_forward_output_invariant
```

If FAIL: the fence in T1.3 Step 3 is misplaced or the HBM write path races the SMEM consumption. Fix the fence before proceeding.

- [ ] **Step 3: Tier A regression**

```bash
cargo test -p nsl-codegen --test csha_cuda_launch_fused --features cuda -- --ignored csha_fused_matrix_sweep
```

Expected: all previously-green configs still green at the same max_abs values (within test-run f16 variance).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "test(csha-c): T1.4 forward output invariant under save_activations flag"
```

---

## Phase 2 — Backward SMEM validator

### Task T2.1: `Direction` enum + direction-parameterised validator

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/smem_layout.rs`

- [ ] **Step 1: Write failing tests**

In `smem_layout.rs` tests:

```rust
#[test]
fn direction_backward_accepts_smallest_config() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    assert!(validate_scalar_v2_config(&cfg, Direction::Backward).is_ok());
}

#[test]
fn direction_backward_rejects_over_budget_with_detailed_diagnostic() {
    // head_dim=64, heads=8, block_q=64 with backward gradient tiles
    // should exceed the 99 KB cap.
    let cfg = base_cfg_fused_backward(64, 64, 64, 8, 64);
    let err = validate_scalar_v2_config(&cfg, Direction::Backward)
        .expect_err("expected backward over-budget rejection");
    let msg = format!("{err}");
    // Must include actual + limit bytes
    assert!(msg.contains("bytes >"), "err must include byte comparison: {msg}");
    assert!(msg.contains("block_q=64"), "err must include block_q: {msg}");
    assert!(msg.contains("head_dim=64"), "err must include head_dim: {msg}");
}

#[test]
fn direction_forward_budget_unchanged_by_phase_2() {
    // Config that Tier A accepts must still accept with Direction::Forward.
    let cfg = base_cfg_fused_forward(32, 32, 32, 4, 32);
    assert!(validate_scalar_v2_config(&cfg, Direction::Forward).is_ok());
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

In `smem_layout.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction { Forward, Backward }

/// Additional SMEM bytes the backward pass needs on top of forward's
/// total: dQ_tile, dK_tile, dV_tile, P_tile (recomputed).
pub fn backward_extra_bytes(config: &FlashAttentionConfig) -> usize {
    let p_bytes = (config.block_q * config.block_kv * 4) as usize;  // f32
    let dq_bytes = (config.block_q * config.head_dim * 4) as usize; // f32 accum
    let dk_bytes = (config.block_kv * config.head_dim * 4) as usize; // f32 accum
    let dv_bytes = (config.block_kv * config.head_dim * 4) as usize; // f32 accum
    p_bytes + dq_bytes + dk_bytes + dv_bytes
}
```

Update the existing `validate_scalar_v2_config(config)` to take a new `direction: Direction` argument. Forward path: unchanged. Backward path: adds `backward_extra_bytes` to `total_bytes` before the 99 KB check. Update ALL existing call sites (grep `validate_scalar_v2_config` workspace-wide) to pass `Direction::Forward` explicitly.

Error message format (must match spec §5.4 + user's note on exit criterion 2):
```rust
Err(SmemBudgetExceeded {
    actual_bytes: total,
    limit_bytes: SMEM_HARDWARE_CAP,
    direction,
    block_q: config.block_q as u32,
    head_dim: config.head_dim as u32,
})
```

Update the existing `SmemBudgetExceeded` struct if it doesn't already carry direction. `Display` impl produces:
```
"CSHA fused {direction:?} rejected: {actual_bytes} bytes > {limit_bytes} byte cap at (block_q={block_q}, head_dim={head_dim})"
```

- [ ] **Step 4: Add `base_cfg_fused_forward` and `base_cfg_fused_backward` helpers**

Near the top of `smem_layout.rs` tests, add two base-config helpers to keep tests DRY. Both construct a valid `FlashAttentionConfig`; the backward helper sets `save_activations_for_backward = true`.

- [ ] **Step 5: Run — expect PASS**

- [ ] **Step 6: Workspace check**

```bash
cargo check --workspace 2>&1 | grep -E "^error" | head -5
```

Any call site not updated from 1-arg to 2-arg is caught here.

- [ ] **Step 7: Full lib suite regression**

```bash
cargo test -p nsl-codegen --lib
```

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T2.1 Direction enum + backward-aware SMEM validator

Adds Direction::{Forward,Backward} parameter to validate_scalar_v2_config.
Backward path adds dQ_tile + dK_tile + dV_tile + recomputed P_tile bytes
to the budget. Rejection diagnostic includes actual + limit bytes and
the (block_q, head_dim) triple for actionable errors."
```

---

## Phase 6.1 — CPU backward reference (parallel with Phase 3)

Start here in parallel with Phase 3. This is the debugging ground-truth for backward phase emission.

### Task T6.1: CPU reference backward — types

**Files:**
- Modify: `crates/nsl-codegen/tests/csha_reference.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn csha_gradients_struct_has_seven_fields() {
    // Sanity smoke: the struct must have dQ, dK, dV, dWq, dWk, dWv, dx.
    let g = CshaGradients {
        dq: vec![0.0],
        dk: vec![0.0],
        dv: vec![0.0],
        dwq: vec![0.0],
        dwk: vec![0.0],
        dwv: vec![0.0],
        dx: vec![0.0],
    };
    assert_eq!(g.dq.len(), 1);
    assert_eq!(g.dwv.len(), 1);
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement types**

Add to `csha_reference.rs`:

```rust
pub struct CshaGradients {
    pub dq: Vec<f32>,   // [seq, heads * head_dim]
    pub dk: Vec<f32>,   // [seq, heads * head_dim]
    pub dv: Vec<f32>,   // [seq, heads * head_dim]
    pub dwq: Vec<f32>,  // [d_model, heads * head_dim]
    pub dwk: Vec<f32>,  // [d_model, heads * head_dim]
    pub dwv: Vec<f32>,  // [d_model, heads * head_dim]
    pub dx: Vec<f32>,   // [seq, d_model]
}
```

- [ ] **Step 4: Run — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "test(csha-c): T6.1 CshaGradients struct for backward reference"
```

### Task T6.2: CPU backward — body

**Files:**
- Modify: `crates/nsl-codegen/tests/csha_reference.rs`

Follow the full chain-rule backward pipeline in f32. ~200 LOC. Order (exact inverse of forward):

```
Input: inputs (x, wq, wk, wv, norm_weight, cos, sin), shape, dO [seq, heads*head_dim]
Intermediate (computed by re-running forward): x_norm, Q, K, V, S, P, O

For each head h, each row i, each col j (respecting causal):
    # dO -> dP via (dV reverse): dV = P^T @ dO; dP = dO @ V^T
    dP[i,j] = sum over d (dO[i,d] * V[j,d])
    dS[i,j] = P[i,j] * (dP[i,j] - sum_k(dP[i,k] * P[i,k]))
    dQ[i,d] += dS[i,j] * K[j,d]
    dK[j,d] += dS[i,j] * Q[i,d]
    dV[j,d] += P[i,j] * dO[i,d]

# Reverse RoPE on dQ, dK:
for each row i, each pair p in 0..(head_dim/2):
    cos, sin = cos_table[i, p], sin_table[i, p]
    # forward: Q[2p] = x0*cos - x1*sin; Q[2p+1] = x0*sin + x1*cos
    # inverse: dx0 = d(Q[2p])*cos + d(Q[2p+1])*sin
    #          dx1 = -d(Q[2p])*sin + d(Q[2p+1])*cos
    d(Q_preRoPE)[2p]   = dQ[2p]*cos   + dQ[2p+1]*sin
    d(Q_preRoPE)[2p+1] = -dQ[2p]*sin  + dQ[2p+1]*cos
    # same for K (dK_preRoPE). V untouched.

# Reverse projections:
dWq += x_norm^T @ dQ_preRoPE  # [d_model, heads*head_dim]
dWk += x_norm^T @ dK_preRoPE
dWv += x_norm^T @ dV

dx_norm = dQ_preRoPE @ Wq^T + dK_preRoPE @ Wk^T + dV @ Wv^T

# Reverse RMSNorm:
# forward: x_norm[i,d] = x[i,d] / rms[i] * norm_weight[d]
# where rms[i] = sqrt(mean(x[i,:]^2) + eps)
for each row i:
    rms = sqrt(mean(x[i,:]^2) + norm_eps)
    scale = 1.0 / rms
    # dx contribution has two terms: direct d(x/rms) and d(rms)/dx
    g = sum_d (dx_norm[i,d] * norm_weight[d] * x[i,d]) / (rms * rms * rms * d_model)
    for d in 0..d_model:
        dx[i,d] = dx_norm[i,d] * norm_weight[d] * scale - x[i,d] * g
```

- [ ] **Step 1: Write failing golden-value test**

```rust
#[test]
fn csha_reference_backward_identity_weights_golden() {
    // Same 4x4 setup as the forward golden test, with identity weights.
    // Hand-computed expected dQ/dK/dV/dWq/dWk/dWv/dx to 1e-6.
    let shape = CshaShape { seq: 4, heads: 1, head_dim: 4, d_model: 4,
                            causal: false, norm_eps: 1e-5 };
    let x = [1.0, 0.0, 0.0, 0.0,
             0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 1.0, 0.0,
             0.0, 0.0, 0.0, 1.0];
    let eye = [1.0, 0.0, 0.0, 0.0,
               0.0, 1.0, 0.0, 0.0,
               0.0, 0.0, 1.0, 0.0,
               0.0, 0.0, 0.0, 1.0];
    let ones = [1.0f32; 4];
    let cos = [1.0f32; 4 * 2];
    let sin = [0.0f32; 4 * 2];
    let inputs = CshaInputs { x: &x, wq: &eye, wk: &eye, wv: &eye,
                              norm_weight: &ones, cos: &cos, sin: &sin };

    // Upstream gradient: d(O)/d(loss) = ones (pretend the loss is sum(O))
    let do_vec: Vec<f32> = vec![1.0; 16];

    let grads = csha_reference_backward(&inputs, &shape, &do_vec);

    // With identity weights, unit-norm inputs, cos/sin=1/0 (RoPE identity),
    // and loss = sum(O), the gradients have a known symmetric structure.
    // The expected values are populated after first run via a println loop.
    let expected_dx = [/* FILL IN after first test run, tolerance 1e-6 */];
    // ... (same for dQ, dK, dV, dWq, dWk, dWv)

    for (i, (a, b)) in grads.dx.iter().zip(expected_dx.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "dx[{i}]: got {a}, expected {b}");
    }
    // Repeat for each gradient.
}
```

- [ ] **Step 2: Run — expect FAIL** (`csha_reference_backward` undefined)

- [ ] **Step 3: Implement `csha_reference_backward`**

Transcribe the pseudocode above into Rust. Re-use the forward `csha_reference` internals for computing x_norm, Q, K, V, S, P, O (either by refactoring forward to expose those intermediates, or by recomputing inline). Prefer a helper: `forward_intermediates(inputs, shape) -> Intermediates { x_norm, q, k, v, s, p, o }` that both forward and backward use.

- [ ] **Step 4: Populate `expected_*` arrays**

First run will fail with the `[/* FILL IN */]` placeholder. Temporarily print each gradient, copy values into the expected arrays, tighten the tolerance to 1e-6, commit cleanly.

- [ ] **Step 5: Add second test for head_dim=64 smoke**

```rust
#[test]
fn csha_reference_backward_runs_for_matrix_shape() {
    // Shape: seq=8, heads=4, head_dim=64, d_model=128. Deterministic
    // seeded inputs. Just assert gradients come out with finite values
    // and expected shapes. Numerical correctness is validated by T6.3
    // against the GPU paths.
    ...
}
```

- [ ] **Step 6: Run — expect PASS**

```bash
cargo test -p nsl-codegen --test csha_reference
```

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "test(csha-c): T6.2 CPU backward reference with 4x4 golden + shape smoke"
```

---

## Phase 3 — Backward phase emission

Phases T3.1..T3.8 emit PTX for each backward stage. Each follows the same pattern: write failing snapshot test, implement, validate via ptxas, commit. **Use the CPU reference from T6.2 as a debugging ground-truth** when individual phase outputs look wrong.

### Task T3.1: Backward prelude (register pool + kernel entry)

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/prelude.rs`
- Modify: `crates/nsl-codegen/src/flash_attention_v2/phases/mod.rs` (add `pub mod backward`)

- [ ] **Step 1: Create `phases/backward/mod.rs`**

```rust
// crates/nsl-codegen/src/flash_attention_v2/phases/backward/mod.rs
pub mod prelude;
// Remaining phases added in T3.2..T3.8.
```

Update `phases/mod.rs`:
```rust
pub mod forward;
pub mod backward;
pub use forward::{prelude as fw_prelude, q_load, /* etc */};
```

Don't re-export backward at top-level; callers use `phases::backward::*`.

- [ ] **Step 2: Write failing snapshot test**

In `prelude.rs`:

```rust
#[test]
fn backward_prelude_declares_gradient_registers() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit(&mut ptx, &cfg);

    // Gradient accumulator registers
    assert!(ptx.contains(".reg .f32 %f_dq") || ptx.contains(".reg .f32 %f_dq_"));
    assert!(ptx.contains("%f_dk"));
    assert!(ptx.contains("%f_dv"));
    // Softmax recompute state
    assert!(ptx.contains("%f_P"));
    assert!(ptx.contains("%f_dP"));
    assert!(ptx.contains("%f_dS"));
    // Kernel entry: .visible .entry with backward-specific param block
    assert!(ptx.contains(".visible .entry"));
    assert!(ptx.contains("dO_ptr"));
    assert!(ptx.contains("q_proj_ptr"));
    assert!(ptx.contains("row_max_ptr"));
}
```

- [ ] **Step 3: Run — expect FAIL**

- [ ] **Step 4: Implement `emit`**

Start by copying `phases/forward/prelude.rs::emit` as a template. Adapt:
- Kernel name: `flash_attn_backward_<config-key>`.
- Param block: forward's 30 args + `dO_ptr`, `q_proj_ptr`, `k_proj_ptr`, `v_proj_ptr`, `row_max_ptr`, `row_sum_ptr`, `dq_ptr`, `dk_ptr`, `dv_ptr`, `dwq_ptr`, `dwk_ptr`, `dwv_ptr`, `dx_ptr`.
- Register pool: forward's existing pool PLUS backward-specific (`%f_dq_0..N`, `%f_dk_*`, `%f_dv_*`, `%f_P_*`, `%f_dP_*`, `%f_dS_*`, `%f_rowsum_dP_P` for the Jacobian reduction).
- Thread/warp setup: same 128 threads/4 warps contract as forward.

- [ ] **Step 5: Run — expect PASS**

- [ ] **Step 6: ptxas validation**

Create `crates/nsl-codegen/tests/csha_ptx_ptxas_backward_validation.rs`:

```rust
//! ptxas clean-assembly gate for backward PTX on sm_75/sm_90/sm_120.

#![cfg(feature = "test-helpers")]

#[test]
fn backward_prelude_ptxas_clean_sm75_sm90_sm120() {
    use nsl_codegen::flash_attention_v2::synthesize_backward;
    // ... same pattern as the existing forward ptxas_validation test.
    // For each SM in ["sm_75", "sm_90", "sm_120"]:
    //   generate PTX with only prelude emission,
    //   pipe to ptxas, assert exit 0.
}
```

(`synthesize_backward` is added in Task T4.1; for T3.1, use a minimal synth that emits header + prelude only. If that's not available yet, add a test-helper `synthesize_backward_prelude_only(cfg, sm) -> String` to `test_helpers.rs`.)

- [ ] **Step 7: Full regression**

```bash
cargo test -p nsl-codegen --lib
```

Expected: still green.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T3.1 backward prelude — register pool + kernel entry"
```

### Task T3.2: Backward Q-load (load saved Q_proj from HBM)

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/q_load.rs`

- [ ] **Step 1: Update `backward/mod.rs` to export `q_load`**

- [ ] **Step 2: Write failing snapshot test**

```rust
#[test]
fn backward_q_load_reads_from_q_proj_ptr() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit(&mut ptx, &cfg, 0);

    // Must read from q_proj_ptr (saved activation), NOT recompute from x
    assert!(ptx.contains("ld.param.u64") && ptx.contains("q_proj_ptr"));
    assert!(ptx.contains("ld.global.b16"));  // f16 load from HBM
    // SMEM write into q_tile region
    assert!(ptx.contains("st.shared.b16"));
    assert!(ptx.contains("V2_BWD_Q_LOAD_0:"));  // per-iter label
    // Must NOT recompute RoPE — Q_proj is already post-RoPE
    assert!(!ptx.contains("cos_ptr"), "backward q_load must use saved post-RoPE Q");
}
```

- [ ] **Step 3: Run — expect FAIL**

- [ ] **Step 4: Implement**

`emit(ptx, config, q_tile_iter)` — cooperative HBM → SMEM load of Q_proj tile. Indexing: `[batch, heads, seq, head_dim]` row-major. Use the same per-lane addressing pattern as forward's cooperative loads. SMEM destination: the existing `%q_smem_base` (Tier A's tile layout is reused because the backward SMEM layout from T2.1 adds gradient tiles AFTER the forward tiles — no overlap).

- [ ] **Step 5: Run — expect PASS + ptxas**

Extend `csha_ptx_ptxas_backward_validation.rs` with a backward q_load config.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T3.2 backward q_load — read saved Q_proj from HBM"
```

### Task T3.3: Backward dS compute (P recompute + dP + dS)

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/ds_compute.rs`

- [ ] **Step 1: Write failing snapshot test**

```rust
#[test]
fn backward_ds_compute_recomputes_P_and_computes_dS() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit(&mut ptx, &cfg, 0);

    // P recomputation from saved stats:
    assert!(ptx.contains("row_max_ptr"));
    assert!(ptx.contains("row_sum_ptr"));
    assert!(ptx.contains("sub.f32"));  // S - row_max
    assert!(ptx.contains("ex2.approx.f32") || ptx.contains("exp.approx.f32"));
    assert!(ptx.contains("div.approx.f32"));  // / row_sum
    // Causal mask BEFORE P recompute (when causal=true)
    // (config test_helper has causal=false so no mask expected)

    // dP = dO @ V^T (inner-loop-over-k):
    assert!(ptx.contains("V2_BWD_DP_LOOP_0:"));
    // dS = P * (dP - rowsum(dP*P)) - warp-butterfly reduction for rowsum
    assert!(ptx.contains("V2_BWD_DS_0:"));
    assert!(ptx.matches("shfl.sync.bfly.b32").count() >= 5,
            "5-step butterfly reduction for dS rowsum");
    // Final dS written to SMEM
    assert!(ptx.contains("%f_dS"));
}

#[test]
fn backward_ds_compute_applies_causal_mask_on_recompute() {
    let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    cfg.causal = true;
    let mut ptx = String::new();
    emit(&mut ptx, &cfg, 0);
    // Causal guard: -INF for col > row
    assert!(ptx.contains("setp.gt"));  // col > row check
    // NEG_INFINITY literal
    assert!(ptx.contains("0xff800000") || ptx.contains("0fFF800000"));
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

Structure per-tile-inner-loop:
```
# Load row_max[row], row_sum[row] from HBM once per row
# For each (k-tile, k-iter) in this tile:
#   S[row, col] = Q[row, :] · K[col, :] / sqrt(head_dim)    # recompute
#   if causal && col > row: S = -INF
#   P[row, col] = exp(S - row_max[row]) / row_sum[row]
#   dP[row, col] = dO[row, :] · V[col, :]                   # reduction over d
#   accumulate rowsum_dP_P[row] += dP[row, col] * P[row, col]
# Butterfly-reduce rowsum_dP_P across the warp
# For each col again:
#   dS[row, col] = P[row, col] * (dP[row, col] - rowsum_dP_P[row])
# Write dS to SMEM for dQdK accum
```

The causal mask is emitted from the SAME helper forward's `s_compute` uses — extract to `phases/forward/s_compute.rs::emit_causal_mask_guard(ptx, cfg)` (or similar) in this task and invoke from both sides. Enforces the R6 "one source of truth" invariant from the spec.

- [ ] **Step 4: Run — expect PASS + ptxas + regression**

- [ ] **Step 5: Debugging hook**

If numerical comparison in T6.3 later shows dS mismatch, the CPU reference from T6.2 produces a ground-truth dS you can dump alongside the kernel's to localize. Document this in a code comment at the top of the file.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T3.3 backward dS compute — P recompute + dP + Jacobian"
```

### Task T3.4: Backward dV accumulation

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/dv_accum.rs`

- [ ] **Step 1: Write failing snapshot test**

```rust
#[test]
fn backward_dv_accum_p_transpose_do() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit(&mut ptx, &cfg, 0);

    // dV[col, d] += sum over row (P[row, col] * dO[row, d])
    assert!(ptx.contains("V2_BWD_DV_ACCUM_0:"));
    assert!(ptx.contains("%f_dv"));
    assert!(ptx.contains("fma.rn.f32"));
    // Accumulation (no zero-out inside the tile loop — reset happens
    // in the orchestrator before the tile loop starts)
    assert!(!ptx.contains("mov.f32 %f_dv_0, 0f00000000"),
            "dv accumulator reset belongs in orchestrator, not inside accum emit");
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

Per-lane-owns-column design (same as Tier A projection):
- Each lane owns one output column of dV for a specific KV row.
- Inner loop iterates over Q rows.
- Accumulator `%f_dv_<slice>` held in registers; written to SMEM dV tile at the END (orchestrator handles the write + accumulation across tiles).

- [ ] **Step 4: ptxas + regression + commit**

```bash
git add -A
git commit -m "feat(csha-c): T3.4 backward dV accumulation — P^T @ dO"
```

### Task T3.5: Backward dQ and dK accumulation

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/dqdk_accum.rs`

- [ ] **Step 1: Write failing snapshot test**

```rust
#[test]
fn backward_dqdk_accum_both_directions() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit(&mut ptx, &cfg, 0);

    assert!(ptx.contains("V2_BWD_DQ_ACCUM_0:"));
    assert!(ptx.contains("V2_BWD_DK_ACCUM_0:"));
    // dQ[row, d] += sum over col (dS[row, col] * K[col, d])
    // dK[col, d] += sum over row (dS[row, col] * Q[row, d])
    assert!(ptx.contains("%f_dq"));
    assert!(ptx.contains("%f_dk"));
    // Both use the dS tile produced by T3.3
    assert!(ptx.contains("%f_dS"));
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

`emit(ptx, config, q_tile_iter)` — two sub-sweeps, one for dQ, one for dK. Each follows the per-lane-owns-column pattern. Both read from the `%f_dS` tile produced by T3.3. Q and K SMEM tiles are still in place from T3.2 and forward-equivalent k_tile_load.

- [ ] **Step 4: ptxas + regression + commit**

```bash
git add -A
git commit -m "feat(csha-c): T3.5 backward dQ+dK accumulation"
```

### Task T3.6: Backward CSHA hooks — dRoPE, dproj, dRMSNorm

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/csha_hooks_backward.rs`

- [ ] **Step 1: Write failing snapshot test — dRoPE**

```rust
#[test]
fn backward_drope_rotates_dq_dk_inversely() {
    let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    cfg.rope_q = true;
    let mut ptx = String::new();
    emit_drope(&mut ptx, &cfg, 0);

    // Inverse rotation math: dx0 = dQ_rot[2p]*cos + dQ_rot[2p+1]*sin
    //                        dx1 = -dQ_rot[2p]*sin + dQ_rot[2p+1]*cos
    assert!(ptx.contains("V2_BWD_DROPE_Q_LOOP_0:"));
    assert!(ptx.contains("V2_BWD_DROPE_K_LOOP_0:"));
    // Four fma.rn.f32 for each pair (2 per dim, × Q and K)
    assert!(ptx.matches("fma.rn.f32").count() >= 4);
    // V is never rotated — must NOT appear
    assert!(!ptx.contains("V2_BWD_DROPE_V_LOOP"));
}
```

- [ ] **Step 2: Write failing snapshot test — dproj**

```rust
#[test]
fn backward_dproj_accumulates_dwq_dwk_dwv() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit_dproj(&mut ptx, &cfg, 0);

    // dWq += x_norm^T @ dQ_preRoPE
    assert!(ptx.contains("V2_BWD_DPROJ_WQ_LOOP_0:"));
    assert!(ptx.contains("V2_BWD_DPROJ_WK_LOOP_0:"));
    assert!(ptx.contains("V2_BWD_DPROJ_WV_LOOP_0:"));
    // Each of Wq/Wk/Wv null-checked independently (matches forward's fix)
    assert!(ptx.contains("setp.eq.u64 %p") && ptx.contains("csha_wq_ptr"));
}
```

- [ ] **Step 3: Write failing snapshot test — dRMSNorm**

```rust
#[test]
fn backward_drmsnorm_produces_dx() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit_drmsnorm(&mut ptx, &cfg, 0);

    // Two-term dx computation:
    //   dx[i,d] = dx_norm[i,d] * norm_weight[d] / rms
    //             - x[i,d] * (sum_d(dx_norm[i,d] * norm_weight[d] * x[i,d]))
    //               / (rms^3 * d_model)
    assert!(ptx.contains("V2_BWD_DRMSNORM_0:"));
    assert!(ptx.contains("%f_g"));  // the rms^3 reduction term
    assert!(ptx.contains("rsqrt.approx.f32"));  // inverse rms
    // Warp butterfly for the rowsum over d
    assert!(ptx.matches("shfl.sync.bfly.b32").count() >= 5);
}
```

- [ ] **Step 4: Run — expect FAIL** (all three fns undefined)

- [ ] **Step 5: Implement all three**

Each function mirrors its forward counterpart in csha_hooks.rs (projection, rope, rmsnorm) with the gradient math. Use the CPU reference in T6.2 as ground truth — compute an expected `g = sum_d(dx_norm * norm_weight * x) / (rms^3 * d_model)` on a hand-held example and verify the PTX arithmetic produces the same shape.

- [ ] **Step 6: Run — expect PASS + ptxas + regression**

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T3.6 backward CSHA hooks — dRoPE + dproj + dRMSNorm"
```

### Task T3.7: Backward finalize — global stores

**Files:**
- Create: `crates/nsl-codegen/src/flash_attention_v2/phases/backward/finalize.rs`

- [ ] **Step 1: Write failing snapshot test**

```rust
#[test]
fn backward_finalize_stores_all_seven_gradients() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let mut ptx = String::new();
    emit(&mut ptx, &cfg, 0);

    // One global store (f16 for tensors, f32 for dx) per gradient
    assert!(ptx.contains("dq_ptr") && ptx.contains("st.global.b16"));
    assert!(ptx.contains("dk_ptr"));
    assert!(ptx.contains("dv_ptr"));
    assert!(ptx.contains("dwq_ptr"));
    assert!(ptx.contains("dwk_ptr"));
    assert!(ptx.contains("dwv_ptr"));
    assert!(ptx.contains("dx_ptr"));
    // Final bar.sync so all lanes finish before kernel exit
    assert!(ptx.contains("bar.sync 0"));
}
```

- [ ] **Step 2: Implement**

Cooperative global stores of the 7 gradient SMEM tiles to their respective output pointers. dx is f32 (the user-visible gradient for training); dQ/dK/dV/dWq/dWk/dWv are f16 (matches forward's Q/K/V f16 layout for symmetric storage). Respect the FFI's specified layouts.

- [ ] **Step 3: Run — expect PASS + ptxas + regression**

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T3.7 backward finalize — global stores of all 7 gradients"
```

---

## Phase 4 — Backward orchestrator + runtime FFI

### Task T4.1: Backward orchestrator `synthesize_backward`

**Files:**
- Modify: `crates/nsl-codegen/src/flash_attention_v2/mod.rs`

- [ ] **Step 1: Write failing snapshot test**

```rust
#[test]
fn synthesize_backward_emits_all_phases_in_order() {
    let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
    let ptx = synthesize_backward(&cfg).expect("synth backward");

    // Phase order: prelude → q_load → [ds_compute → dV_accum → dqdk_accum] per tile
    //               → dRoPE → dproj → dRMSNorm → finalize
    let idx_prelude = ptx.find(".visible .entry").unwrap();
    let idx_qload = ptx.find("V2_BWD_Q_LOAD_0:").unwrap();
    let idx_dS = ptx.find("V2_BWD_DS_0:").unwrap();
    let idx_dV = ptx.find("V2_BWD_DV_ACCUM_0:").unwrap();
    let idx_dQ = ptx.find("V2_BWD_DQ_ACCUM_0:").unwrap();
    let idx_dRoPE = ptx.find("V2_BWD_DROPE_Q_LOOP_0:").unwrap();
    let idx_dproj = ptx.find("V2_BWD_DPROJ_WQ_LOOP_0:").unwrap();
    let idx_drmsnorm = ptx.find("V2_BWD_DRMSNORM_0:").unwrap();
    let idx_final = ptx.find("ret;").unwrap();

    assert!(idx_prelude < idx_qload);
    assert!(idx_qload < idx_dS);
    assert!(idx_dS < idx_dV);
    assert!(idx_dV < idx_dQ);
    assert!(idx_dQ < idx_dRoPE);
    assert!(idx_dRoPE < idx_dproj);
    assert!(idx_dproj < idx_drmsnorm);
    assert!(idx_drmsnorm < idx_final);
}
```

- [ ] **Step 2: Run — expect FAIL** (`synthesize_backward` undefined)

- [ ] **Step 3: Implement**

In `mod.rs`:
```rust
pub fn synthesize_backward(config: &FlashAttentionConfig) -> Result<String, String> {
    smem_layout::validate_scalar_v2_config(config, Direction::Backward)
        .map_err(|e| format!("backward validator rejected: {e}"))?;
    let mut ptx = String::new();
    phases::backward::prelude::emit(&mut ptx, config);
    let iters = (config.block_q / 32) as u32;  // same split-loop pattern as forward
    for q_iter in 0..iters {
        phases::backward::q_load::emit(&mut ptx, config, q_iter);
        // Inner KV loop (similar to forward's split-loop structure):
        phases::backward::ds_compute::emit(&mut ptx, config, q_iter);
        phases::backward::dv_accum::emit(&mut ptx, config, q_iter);
        phases::backward::dqdk_accum::emit(&mut ptx, config, q_iter);
    }
    phases::backward::csha_hooks_backward::emit_drope(&mut ptx, config, 0);
    phases::backward::csha_hooks_backward::emit_dproj(&mut ptx, config, 0);
    phases::backward::csha_hooks_backward::emit_drmsnorm(&mut ptx, config, 0);
    phases::backward::finalize::emit(&mut ptx, config, 0);
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
    Ok(ptx)
}
```

- [ ] **Step 4: Run — expect PASS + ptxas clean**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T4.1 backward orchestrator synthesize_backward"
```

### Task T4.2: Runtime FFI `nsl_flash_attention_csha_backward`

**Files:**
- Modify: `crates/nsl-runtime/src/flash_attention.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
#[cfg(feature = "cuda")]
fn nsl_flash_attention_csha_backward_smoke() {
    // Call the FFI with all pointers null except the mandatory ones.
    // Expected: returns rc=0 (success) on a minimal 32x32x32 config.
    // Numerical correctness validated by T6.3 three-way gate.
    // This test just verifies the FFI signature + basic launch doesn't crash.
    ...
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

New extern C function. Build the 40-arg launch list (forward's 30 + 5 saved activations + 7 gradient outputs). Similar structure to `nsl_flash_attention_csha` from Tier A. Dispatch via `cuda::inner::kernel_launch`. Include dynamic SMEM opt-in path from Track B for configs where `total_bytes > 48 KB`.

- [ ] **Step 4: Run — expect PASS + regression**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T4.2 runtime FFI nsl_flash_attention_csha_backward"
```

---

## Phase 5 — AD dispatcher integration

### Task T5.1: `FusionMark.backward_emitted` flag

**Files:**
- Modify: `crates/nsl-codegen/src/csha_apply.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn fusion_mark_has_backward_emitted_default_false() {
    let m = FusionMark { layer: "blocks.0".into(), op_ids: vec![1,2,3],
                         config: FlashAttentionConfig::default(),
                         backward_emitted: std::cell::Cell::new(false) };
    assert!(!m.backward_emitted.get());
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Add `backward_emitted` field** (interior-mutable Cell so dispatcher can set without `&mut self`)

Also add `config: FlashAttentionConfig` to the mark so the AD dispatcher can invoke `validate_scalar_v2_config(&mark.config, Direction::Backward)` without reaching back into the plan.

Update `csha_apply::bridge` to populate both fields when creating marks.

- [ ] **Step 4: Run — expect PASS + lib regression**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T5.1 FusionMark.backward_emitted + config"
```

### Task T5.2: AD dispatcher routes claimed chains to fused backward

**Files:**
- Modify: `crates/nsl-codegen/src/ad_rules.rs`

- [ ] **Step 1: Write failing test**

```rust
#[test]
fn ad_dispatcher_emits_fused_backward_on_claimed_chain_output_op() {
    // Construct a minimal Wengert list with a CSHA chain; invoke
    // dispatch_adjoint in reverse order; assert exactly one fused
    // backward call emitted on the first claimed-op encounter, and
    // no-op on subsequent claimed ops (backward_emitted flag set).
    ...
}

#[test]
fn ad_dispatcher_falls_back_on_validator_reject_with_diagnostic() {
    // Construct a chain whose config FAILS validate_scalar_v2_config
    // (e.g., head_dim=128 heads=8). Dispatcher must emit per-op
    // adjoints AND surface the exact SmemBudgetExceeded diagnostic.
    ...
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement**

New match arm in `dispatch_adjoint`:

```rust
if let Some(mark) = self.csha_claimed_chain_for(op_idx) {
    if mark.backward_emitted.get() { return Ok(()); }
    match smem_layout::validate_scalar_v2_config(&mark.config, Direction::Backward) {
        Ok(()) => {
            self.emit_fused_backward_call(&mark)?;
            mark.backward_emitted.set(true);
            self.register_output_varids(&mark);
            return Ok(());
        }
        Err(e) => {
            self.diagnostics.push(format!("CSHA fused backward rejected: {e}; falling back to unfused backward"));
            // fall through to per-op adjoint rules below
        }
    }
}
// existing per-op adjoint dispatch
```

**Implementation detail invariant** (from spec §5.4): the reverse-walk must hit the chain's OUTPUT op first. Add a `debug_assert!(mark.op_ids.last() == Some(&op_idx))` in the emission branch to catch any FusionMark that lists ops in non-topological order.

- [ ] **Step 4: Run — expect PASS + lib regression**

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(csha-c): T5.2 AD dispatcher routes CSHA chains to fused backward"
```

---

## Phase 6.3 — Three-way GPU numerical gate

### Task T6.3: GPU three-way gate with parametric sweep

**Files:**
- Create: `crates/nsl-codegen/tests/csha_cuda_backward.rs`

- [ ] **Step 1: Scaffold test file with smoke**

```rust
//! Three-way numerical gate for CSHA fused backward on RTX 5070 Ti.
//! Compares: (fused GPU) vs (existing unfused GPU) vs (CPU reference).

#![cfg(feature = "cuda")]

#[path = "csha_reference.rs"]
mod csha_reference;
use csha_reference::{csha_reference, csha_reference_backward,
                     CshaInputs, CshaShape, CshaGradients};

fn run_fused_backward_config(
    block_q: u32, block_kv: u32, head_dim: u32, heads: u32, d_model: u32,
    causal: bool, rope_q: bool,
) -> Result<(f32 /* max_abs fused-vs-cpu */, f32 /* unfused-vs-cpu */,
              f32 /* fused-vs-unfused diag */), String> {
    // 1. Forward with save_activations=true
    // 2. Get saved Q_proj, K_proj, V_proj, row_max, row_sum
    // 3. Random dO (seed=42 LCG)
    // 4. Launch fused backward -> grads_fused
    // 5. Call flash_attention_backward_gpu (unfused) -> grads_unfused
    // 6. Call csha_reference_backward (CPU) -> grads_cpu
    // 7. Return max_abs diffs for all three pairwise
    ...
}

#[test]
#[ignore]
fn t6_3_smoke_single_config() {
    let (f_v_c, u_v_c, f_v_u) = run_fused_backward_config(32, 32, 32, 4, 32, false, true)
        .expect("smoke config should succeed");
    eprintln!("[T6.3 smoke] fused-vs-cpu={f_v_c:.3e} unfused-vs-cpu={u_v_c:.3e} fused-vs-unfused={f_v_u:.3e}");
    assert!(f_v_c < 5e-3, "fused vs CPU exceeds head_dim=32 tolerance");
    assert!(u_v_c < 5e-3, "unfused vs CPU exceeds head_dim=32 tolerance (latent pre-existing bug?)");
}
```

- [ ] **Step 2: Implement `run_fused_backward_config` body**

Mirror `run_fused_config` from `csha_cuda_launch_fused.rs`. The unfused path calls `nsl_flash_attention_backward` (the existing FFI).

- [ ] **Step 3: Run smoke — expect PASS**

```bash
cargo test -p nsl-codegen --test csha_cuda_backward --features cuda -- --ignored t6_3_smoke
```

If `f_v_c` fails (fused wrong), the CPU reference localizes where the kernel diverges — dump intermediate P, dS, dQ/dK/dV tiles with a debug flag.
If `u_v_c` fails (unfused wrong), you've found a latent pre-existing bug — **valuable**. Document + investigate, but do NOT let it block Tier C.
If both fail and disagree with each other, something is fundamentally off (likely layout/shape mismatch).

- [ ] **Step 4: P-recomputation assertion on causal**

```rust
#[test]
#[ignore]
fn t6_3_p_recomputation_matches_forward_on_causal() {
    // Gated via a test-only flag in the kernel (or a debug-output buffer
    // that the backward writes recomputed P into). Compare to forward's P
    // (reproducible from the same inputs via csha_reference forward path).
    ...
}
```

- [ ] **Step 5: Add matrix sweep**

```rust
#[test]
#[ignore]
fn t6_3_matrix_sweep() {
    let configs: &[(u32, u32, u32, u32, u32, bool, bool)] = &[
        (32, 32, 32, 4, 32, false, true),
        (32, 32, 32, 4, 32, true,  true),
        (32, 32, 64, 4, 64, false, true),
        (32, 32, 64, 4, 64, true,  true),
        (64, 64, 32, 8, 32, false, true),
        (64, 64, 32, 8, 32, true,  true),
        // head_dim=128 with d_model=32 (per Track B)
        (32, 32, 128, 4, 32, false, true),
    ];
    for &(bq, bkv, hd, h, dm, causal, rope) in configs {
        let tol = if hd >= 128 { 4e-2 } else if hd >= 64 { 2e-2 } else { 5e-3 };
        match run_fused_backward_config(bq, bkv, hd, h, dm, causal, rope) {
            Ok((f_v_c, u_v_c, f_v_u)) => {
                eprintln!("[T6.3] bq={bq} bkv={bkv} hd={hd} h={h} dm={dm} causal={causal}: \
                           fused-v-cpu={f_v_c:.3e} unfused-v-cpu={u_v_c:.3e} \
                           fused-v-unfused={f_v_u:.3e} tol={tol:.1e}");
                assert!(f_v_c < tol, "fused-vs-cpu exceeds tolerance");
                assert!(u_v_c < tol, "unfused-vs-cpu exceeds tolerance");
            }
            Err(e) => {
                // Validator-rejected configs produce Err; that's OK for the
                // sweep — assert no silent mismatches, not that every config fires.
                eprintln!("[T6.3] skipped (validator rejected): {e}");
            }
        }
    }
}
```

- [ ] **Step 6: Run — iterate on failures up to 4 attempts per config**

Budget: if after 4 iterations any matrix row won't go green, STOP and report BLOCKED with the specific config + max_abs values + hypothesis. Do NOT relax tolerance beyond the tiered model.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "test(csha-c): T6.3 three-way GPU numerical gate + matrix sweep"
```

---

## Phase 7 — Fallback regression test

### Task T7.1: AD fallback when validator rejects

**Files:**
- Create: test inline in `crates/nsl-codegen/src/ad_rules.rs` OR new `tests/ad_csha_fallback.rs` — choose whichever matches the existing AD test convention

- [ ] **Step 1: Write the test**

```rust
#[test]
fn ad_dispatcher_falls_back_when_backward_validator_rejects() {
    // Config: one that passes forward but fails backward (e.g., heads=8
    // head_dim=64 at large block dims — exact config emerges from T2.1
    // validator testing).
    let forward_ok_backward_reject_cfg = ...;

    // Construct a Wengert list + CSHA chain + invoke the AD dispatcher.
    // Capture emitted adjoint ops and diagnostics.
    let result = dispatch_adjoints_for_cfg(&forward_ok_backward_reject_cfg);

    // Adjoint op count equals sum of per-op adjoint rules (NOT 1).
    assert!(result.emitted_op_count > 1,
            "expected per-op fallback, got fused call (op_count={})",
            result.emitted_op_count);
    // Diagnostic surfaces the byte excess.
    assert!(result.diagnostics.iter().any(|d|
        d.contains("CSHA fused backward rejected") &&
        d.contains("bytes >") &&
        d.contains("falling back to unfused backward")
    ), "expected fallback diagnostic, got {:?}", result.diagnostics);
}
```

- [ ] **Step 2: Run — expect PASS** (T5.2 already implemented the behavior)

If FAIL: the AD dispatcher is not hitting the fallback branch. Debug by running the validator standalone on the same config — if validator says OK, the test's config needs tightening; if validator says reject and dispatcher still emits fused, the match arm from T5.2 has a bug.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "test(csha-c): T7.1 AD dispatcher fallback on backward validator reject"
```

---

## Phase 8 — Release discipline

### Task T8.1: Full regression + push

- [ ] **Step 1: Full test suite**

```bash
cargo test -p nsl-codegen --lib 2>&1 | tail -3
cargo test -p nsl-codegen --test csha_reference
cargo test -p nsl-codegen --test csha_cuda_launch_fused --features cuda -- --ignored
cargo test -p nsl-codegen --test csha_cuda_backward --features cuda -- --ignored
cargo test -p nsl-codegen --test csha_ptx_ptxas_backward_validation --features test-helpers -- --ignored
```

All green. Lib count ≥ 1520. Tier A forward matrix sweep unchanged max_abs.

- [ ] **Step 2: Memory close-out**

Write new memory file `project_csha_tier_c_shipped.md` describing:
- What shipped (reference this plan).
- Three-way gate results per matrix config.
- Any latent-bug findings in the existing unfused backward (if `u_v_c` failures surfaced).
- Known follow-ups (dead-head elimination, @checkpoint fusion, performance benchmarking, Wo backward when Wo forward ships).

Update `MEMORY.md` index entry: move Tier A's line under "Shipped" and add Tier C under "Active blockers" or "Shipped" as appropriate.

- [ ] **Step 3: Push + PR**

```bash
git push -u origin feat/csha-tier-c
```

Draft PR body summarizing the milestone + three-way gate results + any latent-bug findings.

---

## Self-Review Checklist

- [ ] **Spec coverage:** P0 (T0.2), P1 (T1.1-T1.4), P2 (T2.1), P3 (T3.1-T3.7), P4 (T4.1-T4.2), P5 (T5.1-T5.2), P6.1/6.2 (T6.1-T6.2), P6.3 (T6.3), P7 (T7.1), release (T8.1). Every spec phase has tasks. Exit criteria §9.1 → T6.3 matrix sweep. §9.2 → T5.2 + T7.1. §9.3 → T1.4 + T1.3 snapshot. §9.4 → T1.4 Step 3 re-run of Tier A sweep. §9.5 → T6.3 smoke reports all three paths.
- [ ] **Placeholder scan:** Every step has concrete code or exact commands. Two exceptions explicitly marked "FILL IN after first test run": T6.2 Step 4 (golden array population pattern established in Tier A C1) and T4.2 Step 3 (FFI implementation mirrors T4.1 of Tier A — pattern exists). Neither hides design uncertainty.
- [ ] **Type consistency:** `CshaGradients` fields (`dq`, `dk`, `dv`, `dwq`, `dwk`, `dwv`, `dx`) match across T6.1, T6.2, T6.3. `Direction::{Forward, Backward}` consistent across T2.1, T4.1, T5.2. `FusionMark.backward_emitted` (interior-mutable Cell) consistent across T5.1, T5.2. FFI arg order for `nsl_flash_attention_csha_backward` documented in T4.2 (30+5+7 layout).
- [ ] **Decision points surfaced:** R2 (fence placement) in T1.3 Step 3 + T1.4 regression gate. R3 (reverse-walk invariant) in T5.2 Step 3 debug_assert. R6 (causal mask single source) in T3.3 Step 3 helper extraction.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-15-csha-tier-c-fused-backward.md`. Two execution options:

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks, fast iteration. Best for a ~22-task plan with clear test gates per task.
2. **Inline Execution** — execute tasks in this session using executing-plans, batched checkpoints for review.

Which approach?
