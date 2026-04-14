# FASE Deferred-Mode Codegen Integration — Design

**Date:** 2026-04-14
**Status:** Design approved, ready for implementation plan
**Scope:** Item #1 of the FASE remaining-work ordering (see research/CFTP.pdf)

## Context

FASE (Fused Accumulation-Step Elimination, CFTP paper Part I) has a complete
*planner* and *recipe* layer under `crates/nsl-codegen/src/fase*.rs` (~1218 LOC),
but no consumer. The train-block backward emitter in `stmt.rs` still uses the
pre-FASE `grad_accumulation` path: allocate a per-parameter accumulation buffer,
sum raw gradients into it across micro-batches, then run a separate optimizer
step after the loop.

This spec wires the planner in and emits the Deferred-mode rewrite described in
CFTP §2.3 when the planner selects it. `Full` and `Passthrough` fall through to
the existing code unchanged.

## Goals

1. Call `fase::plan()` from the train-block backward emitter.
2. On `FaseMode::Deferred`, emit:
   - Per-micro-batch: `m_partial[k] += (1/N) * grad[k]` (pre-scaled).
   - Final micro-batch: per-parameter fused optimizer step using the recipe
     from `fase_optimizer.rs`, followed by `m_partial[k] = 0`.
3. Fix the `b_scale: 0.0` placeholder in `fase_optimizer.rs` by adding a
   `Register::MPartial` variant so AdamW can read `m` and `m_partial` as
   distinct operands.

## Non-Goals (explicit deferrals)

- **Per-parameter interleaving inside the backward pass** (CFTP §2.4 peak
  memory scheduling). Pass #1 uses two-stage emission: run backward into a
  gradient list, then loop over parameters to emit fused steps. The
  interleaving is tracked as item #1b and lands after numerical-equivalence
  tests pass.
- **Two-phase gradient clipping.** If `grad_clip` is set and the planner would
  otherwise return `Deferred`, the planner instead returns `Full` for this
  pass. The two-phase clip codegen is item #3.
- **M36 memory-planner integration.** Item #4.
- **`nsl check --training-report` CLI.** Item #6.
- **Routing `Full`/`Passthrough` through the planner's match arm.** Item #6
  will refactor this when the CLI gains a consumer; not needed now.

## Design Decisions

### D1. Dual path with planner dispatch

The planner already distinguishes Deferred (fusable) from Full (e.g. Lion,
which needs the raw accumulated gradient for its sign-based update). The
codegen dispatches:

```
match plan.mode {
    Passthrough | Full => <existing code, untouched>,
    Deferred           => emit_fase_deferred(..., &plan),
}
```

### D2. Reuse the existing `accum_list` allocation as `m_partial`

The allocation at `stmt.rs:3329` has identical shape, lifetime, and reset
pattern to the `m_partial` buffer required by Deferred mode. No new
allocation; only the *contents* and the *consumer* change. Net memory delta
vs today's code: zero.

### D3. Pre-scaled accumulation

Emit `m_partial += (1/N) * grad[k]` per micro-batch rather than
`accum += grad[k]` followed by a post-loop divide. Produces identical
final values in f32, but keeps intermediate magnitudes smaller — better
numerical behavior under f16/bf16 and a direct match for the recipe's
expected input semantics (`m_partial` is the *mean* gradient at the final
step, not the sum).

### D4. Two-stage emission on final micro-batch (approach B)

On micro-batch `N-1`, emit:

1. Final `m_partial += (1/N) * grad[k]` for all parameters (completes
   backward as today, but writing into `m_partial`, not `accum`).
2. A per-parameter loop that invokes `fase_optimizer::emit_adamw` (or the
   appropriate recipe) and lowers each op to Cranelift IR, followed by
   `m_partial[k] = 0`.

This keeps the backward lowering untouched. Item #1b will merge stage 2
into stage 1 per-parameter once tests confirm the math.

### D5. Add `Register::MPartial` to the update IR

Current `Register::M` is documented as dual-purpose (`m_partial` in Deferred,
AdamW `m` in standard). AdamW's Deferred final step needs both registers
simultaneously:

```
m_new = β₁ * m_old + (1 - β₁) * m_partial
```

A single register cannot hold both operands. Add:

```rust
pub enum Register {
    Theta,
    M,          // first moment (AdamW/Adam/Lion)
    MPartial,   // Deferred-mode accumulated mean gradient
    V,
    G,
    Tmp,
}
```

Update `emit_adamw` so the `ScalarMulAdd` at
`fase_optimizer.rs:165-171` reads:

```rust
UpdateOp::ScalarMulAdd {
    dst:   Register::M,
    src:   Register::M,
    a:     recipe.beta1,
    b_src: Some(Register::MPartial),
    b_scale: recipe.one_minus_beta1,   // replaces the 0.0 placeholder
}
```

No additional `1/N` factor — `m_partial` is already the mean gradient by D3.

`v` update uses `m_partial` as its operand (already correct in the existing
code, just needs the register rename):

```rust
UpdateOp::SquaredAccumulate {
    dst: Register::V,
    src: Register::V,
    operand: Register::MPartial,
    scale: recipe.one_minus_beta2,
}
```

## Components Touched

| File | Change |
|---|---|
| `crates/nsl-codegen/src/fase_optimizer.rs` | Add `Register::MPartial`; fix AdamW/Adam emitters to use it; set `b_scale` correctly. |
| `crates/nsl-codegen/src/fase.rs` | Planner returns `Full` instead of `Deferred` when `grad_clip` is set (temporary; reverted by item #3). |
| `crates/nsl-codegen/src/stmt.rs` | Call `fase::plan()` after parsing train config; dispatch to new `emit_fase_deferred` on Deferred; existing code unchanged otherwise. |

New function `emit_fase_deferred` lives in `stmt.rs` (or a new
`stmt_fase.rs` sibling if it grows beyond ~150 LOC) and handles:
- emitting `m_partial += (1/N) * grad[k]` in place of `accum += grad[k]`
- emitting the per-parameter fused step on the final micro-batch
- lowering each `UpdateOp` from the recipe to Cranelift IR
- zeroing `m_partial[k]` after each accumulation window

## Risks

- **Recipe-consumer interface mismatch.** The recipes in `fase_optimizer.rs`
  haven't been exercised by real codegen. First-integration work will
  surface any missing data (hyperparameter hookup, register-to-buffer-slot
  mapping). Budget for minor recipe adjustments during implementation.
- **Numerical drift vs standard path.** Pre-scaled accumulation (D3) is
  mathematically identical in exact arithmetic but can diverge in f32 at
  the last-bit level. Item #2's equivalence test tolerance must account
  for this (suggest `1e-5` relative for f32 AdamW after 4 micro-batches).

## Success Criteria for This Pass

- `cargo build -p nsl-codegen` succeeds.
- Existing train-block tests (Full path, Passthrough path) pass unchanged.
- A hand-compiled train block with `grad_accumulation = 4` and AdamW emits
  the Deferred IR (verified by inspection or a new unit test on the IR
  shape — not a numerical test; that's item #2).
- No behavior change for `grad_accumulation = 1` or Lion.

## Follow-Ups

- **Item #1b:** per-parameter interleaving (approach A) for the peak-memory
  win.
- **Item #2:** numerical equivalence test (standard vs Deferred, same seed,
  f32 AdamW, N=4).
- **Item #3:** two-phase gradient clipping codegen, removing the planner's
  temporary `grad_clip → Full` fallback.
