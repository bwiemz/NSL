# M38b: Linear Types — Codegen & Safety Proofs Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the ownership checker's decisions into actual Cranelift IR emission — emit `nsl_tensor_free` at consumption points for linear tensors, skip refcount ops (`nsl_tensor_incref`/`nsl_tensor_decref`) for proven single-owner bindings, and add debug-mode poison values for use-after-move detection. This is the payoff from M38a: the semantic analysis is done, now the codegen exploits it.

**Architecture:** Extend `crates/nsl-codegen/src/ownership.rs` with methods that integrate into the compilation pipeline. The `OwnershipLowering` struct (already created in M38a) gains `emit_free_at_consumption`, `emit_poison_after_move`, and `should_skip_refcount` methods. The compiler's `compile_fn_def` hooks into ownership lowering to decide per-binding whether to emit refcount ops or consumption-point frees. The existing `BackwardAccess` classification from `ownership_autodiff.rs` is used to determine which consumed tensors can be freed early (ShapeOnly) vs must stay alive (DataRequired).

**Tech Stack:** Rust (codegen Cranelift IR emission + ownership integration)

**Spec:** `docs/superpowers/specs/2026-03-15-m38-linear-types-design.md` (Sections 3, 4, 6)

**Prerequisites:** M38a (OwnershipChecker, OwnershipState, OwnershipLowering types, BackwardAccess classification — all complete)

---

## Important: Scope of This Plan

**This plan wires ownership decisions into codegen.** It delivers:
- `OwnershipLowering` methods: `should_skip_refcount`, `emit_free_at_consumption`, `emit_poison_after_move`
- Compiler integration: populate `OwnershipLowering` from `OwnershipChecker` results
- Refcount elision: linear bindings skip `nsl_tensor_incref`/`nsl_tensor_decref`
- Consumption-point free: `nsl_tensor_free` emitted where linear tensor is consumed, not at scope exit
- Debug poison: zero pointer slot after move (gated by debug flag)
- `ownership_info` field on Compiler struct (HashMap of FunctionOwnership per function)
- BackwardAccess-aware free decisions: ShapeOnly ops → free input early; DataRequired → tape holds reference
- 10+ unit tests for ownership lowering decisions

**Deferred to M38c:** M36 memory planner integration (buffer reuse for consumed tensors), `--warn-shared` audit mode, performance benchmarks (measuring refcount ops eliminated), E2E tests with actual model forward/backward passes.

---

## File Structure

### Modified Files

| File | Change | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/ownership.rs` | Add codegen emission methods to OwnershipLowering | +150 |
| `crates/nsl-codegen/src/compiler.rs` | Add `ownership_info` field, populate from checker, hook into compile_fn_def | +40 |

---

## Phase 1: OwnershipLowering Emission Methods

### Task 1: Extend OwnershipLowering with Codegen Methods

**Files:**
- Modify: `crates/nsl-codegen/src/ownership.rs`

- [ ] **Step 1: Add emission decision methods and tests to ownership.rs**

Add these methods to the existing `OwnershipLowering` impl block, and add the `OwnershipDecision` enum:

```rust
/// Decision for how to handle a tensor at a usage site.
#[derive(Debug, Clone, PartialEq)]
pub enum OwnershipDecision {
    /// Tensor is linear and being consumed — emit nsl_tensor_free at this point.
    /// No refcount inc/dec needed.
    FreeAtConsumption,
    /// Tensor is linear but used in a DataRequired autodiff op — tape holds reference,
    /// do NOT free early. The tape's saved_* field keeps it alive.
    TapeHoldsReference,
    /// Tensor is @shared — use normal refcount inc/dec (current behavior).
    RefcountManaged,
    /// Tensor is borrowed — no ownership action needed (caller retains ownership).
    BorrowedNoAction,
    /// Debug mode: after consumption, zero the source pointer slot for null-on-reuse detection.
    PoisonAfterMove,
}

impl OwnershipLowering {
    /// Determine the ownership action for a tensor binding at a specific usage site.
    ///
    /// `sym`: the binding being used
    /// `is_consuming`: true if this use consumes (moves) the tensor
    /// `backward_access`: if inside a `grad` block, the tape op's backward classification
    /// `debug_mode`: whether to emit poison values after moves
    pub fn decide(
        &self,
        sym: &Symbol,
        is_consuming: bool,
        backward_access: Option<&str>,
        debug_mode: bool,
    ) -> Vec<OwnershipDecision> {
        let mut decisions = Vec::new();

        // @shared bindings always use refcount management
        if self.shared_bindings.contains(sym) {
            decisions.push(OwnershipDecision::RefcountManaged);
            return decisions;
        }

        // Borrowed bindings need no ownership action
        if self.active_borrows.contains_key(sym) {
            decisions.push(OwnershipDecision::BorrowedNoAction);
            return decisions;
        }

        // Linear binding being consumed
        if self.linear_bindings.contains(sym) && is_consuming {
            // Check if the autodiff tape needs this tensor's data
            if let Some(op_name) = backward_access {
                use nsl_semantic::ownership_autodiff::{classify_backward_access, BackwardAccess};
                let access = classify_backward_access(op_name);
                match access {
                    BackwardAccess::ShapeOnly => {
                        // Safe to free immediately — backward only needs shape
                        decisions.push(OwnershipDecision::FreeAtConsumption);
                    }
                    BackwardAccess::DataRequired => {
                        // Tape saved_* holds a reference — don't free
                        decisions.push(OwnershipDecision::TapeHoldsReference);
                    }
                    BackwardAccess::AuxDataRequired => {
                        // Aux data is owned by tape, input tensor can be freed
                        decisions.push(OwnershipDecision::FreeAtConsumption);
                    }
                }
            } else {
                // Not in grad block — always free at consumption
                decisions.push(OwnershipDecision::FreeAtConsumption);
            }

            // Debug mode: poison the source slot
            if debug_mode {
                decisions.push(OwnershipDecision::PoisonAfterMove);
            }

            return decisions;
        }

        // Linear binding NOT being consumed (e.g., borrowed use)
        if self.linear_bindings.contains(sym) {
            decisions.push(OwnershipDecision::BorrowedNoAction);
            return decisions;
        }

        // Default: refcount managed (no ownership info available)
        decisions.push(OwnershipDecision::RefcountManaged);
        decisions
    }

    // NOTE: Use existing M38a methods for refcount queries:
    //   should_elide_refcount(sym) — true if linear (ignores shared status)
    //   should_free_at_consumption(sym) — true if linear AND not shared

    /// Register a binding from the ownership checker results.
    // NOTE: Use existing M38a methods for registration:
    //   mark_linear(sym)  — adds to linear_bindings
    //   mark_shared(sym)  — adds to shared_bindings
    // And existing query methods:
    //   should_elide_refcount(sym)  — true if linear (existing, does NOT check shared)
    //   should_free_at_consumption(sym) — true if linear AND not shared
}

// NOTE: Do NOT duplicate the backward access classifier. Reuse the semantic layer:
//   use nsl_semantic::ownership_autodiff::{classify_backward_access, BackwardAccess};
// nsl-codegen already depends on nsl-semantic in Cargo.toml.
// The `decide()` method calls classify_backward_access(op_name) directly.
//
// Cat/Stack/BiasAdd are classified as DataRequired (tape holds refcount-bumped pointers
// for gradient routing). This deviates from the spec's Section 6.1 which puts Cat in
// ShapeOnly — the implementation is correct, the spec is wrong.
// Conv2d, Unsqueeze, Expand are additions not in the original spec match arms.
```

Add tests to the existing `#[cfg(test)]` module (or create one if absent):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::Symbol;

    type Interner = string_interner::StringInterner<
        string_interner::backend::BucketBackend<string_interner::DefaultSymbol>,
    >;

    fn make_sym(interner: &mut Interner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    #[test]
    fn linear_consuming_no_grad_frees() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::FreeAtConsumption]);
    }

    #[test]
    fn linear_consuming_shape_only_op_frees() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, Some("Add"), false);
        assert_eq!(decisions, vec![OwnershipDecision::FreeAtConsumption]);
    }

    #[test]
    fn linear_consuming_data_required_op_tape_holds() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, Some("MatMul"), false);
        assert_eq!(decisions, vec![OwnershipDecision::TapeHoldsReference]);
    }

    #[test]
    fn linear_consuming_debug_adds_poison() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, true, None, true);
        assert_eq!(decisions, vec![
            OwnershipDecision::FreeAtConsumption,
            OwnershipDecision::PoisonAfterMove,
        ]);
    }

    #[test]
    fn shared_always_refcounted() {
        let mut interner = Interner::new();
        let w = make_sym(&mut interner, "W");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_shared(w);

        let decisions = lowering.decide(&w, true, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::RefcountManaged]);
    }

    #[test]
    fn borrowed_no_action() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let b = make_sym(&mut interner, "ref_x");
        let mut lowering = OwnershipLowering::new();
        lowering.active_borrows.insert(x, BorrowKind::Immutable { borrower: b });

        let decisions = lowering.decide(&x, false, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::BorrowedNoAction]);
    }

    #[test]
    fn elide_refcount_for_linear() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let w = make_sym(&mut interner, "W");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);
        lowering.mark_shared(w);

        assert!(lowering.should_elide_refcount(&x));
        assert!(!lowering.should_elide_refcount(&w));
        assert!(lowering.should_free_at_consumption(&x));
        assert!(!lowering.should_free_at_consumption(&w));
    }

    #[test]
    fn mark_linear_and_shared() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let w = make_sym(&mut interner, "W");
        let mut lowering = OwnershipLowering::new();

        lowering.mark_linear(x);
        lowering.mark_shared(w);

        assert!(lowering.linear_bindings.contains(&x));
        assert!(lowering.shared_bindings.contains(&w));
    }

    #[test]
    fn autodiff_classify_coverage() {
        use nsl_semantic::ownership_autodiff::{classify_backward_access, BackwardAccess};
        assert_eq!(classify_backward_access("Add"), BackwardAccess::ShapeOnly);
        assert_eq!(classify_backward_access("MatMul"), BackwardAccess::DataRequired);
        assert_eq!(classify_backward_access("Dropout"), BackwardAccess::AuxDataRequired);
        assert_eq!(classify_backward_access("UnknownOp"), BackwardAccess::DataRequired);
    }

    #[test]
    fn linear_non_consuming_is_borrow() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut lowering = OwnershipLowering::new();
        lowering.mark_linear(x);

        let decisions = lowering.decide(&x, false, None, false);
        assert_eq!(decisions, vec![OwnershipDecision::BorrowedNoAction]);
    }
}
```

---

## Phase 2: Compiler Integration

### Task 2: Wire into Compiler

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 2: Add `ownership_info` field and populate from checker results**

Add to the Compiler struct:
```rust
/// M38b: Per-function ownership metadata from the ownership checker.
/// Populated when --linear-types is active.
pub ownership_info: HashMap<String, crate::ownership::FunctionOwnership>,
```
Initialize: `ownership_info: HashMap::new()`.

The `ownership_info` is populated after the ownership checker runs (in the CLI or compile entry point). For M38b, the codegen reads this info during `compile_fn_def` in **`func.rs`** (NOT compiler.rs — `compile_fn_def` is at `func.rs:14`) to create an `OwnershipLowering` for each function:

```rust
// In func.rs, at the start of compile_fn_def, after getting function name:
let ownership_lowering = if self.linear_types_enabled {
    if let Some(fn_ownership) = self.ownership_info.get(&name) {
        let mut lowering = crate::ownership::OwnershipLowering::new();
        for sym in &fn_ownership.linear_params {
            lowering.mark_linear(*sym);
        }
        for sym in &fn_ownership.shared_params {
            lowering.mark_shared(*sym);
        }
        // NOTE: borrowed_params registration deferred — borrows tracked at usage site
        Some(lowering)
    } else {
        None
    }
} else {
    None
};
// Store in FuncState or pass to compilation functions
```

---

## Phase 3: Build Verification

- [ ] **Step 3: `cargo build` — verify no compile errors**

- [ ] **Step 4: `cargo test` — run all tests, expect 10+ new tests passing**

Expected new tests in `ownership.rs`:
- `linear_consuming_no_grad_frees`
- `linear_consuming_shape_only_op_frees`
- `linear_consuming_data_required_op_tape_holds`
- `linear_consuming_debug_adds_poison`
- `shared_always_refcounted`
- `borrowed_no_action`
- `skip_refcount_for_linear`
- `register_binding`
- `autodiff_classify_coverage`
- `linear_non_consuming_is_borrow`

- [ ] **Step 5: `cargo clippy` — no warnings**

---

## Verification Checklist

1. **Linear consuming free**: Linear tensor consumed outside grad block → FreeAtConsumption
2. **ShapeOnly ops**: Add/Sub consumed tensors → FreeAtConsumption (safe, backward only needs shape)
3. **DataRequired ops**: MatMul/ReLU consumed tensors → TapeHoldsReference (tape has saved_*)
4. **AuxData ops**: Dropout consumed tensors → FreeAtConsumption (aux data owned by tape)
5. **Debug poison**: consumption + debug mode → FreeAtConsumption + PoisonAfterMove
6. **Shared always refcounted**: @shared bindings → RefcountManaged regardless of context
7. **Borrow no-op**: Borrowed bindings → BorrowedNoAction
8. **Refcount skip**: `should_skip_refcount` true for linear, false for shared
9. **Compiler field**: `ownership_info` populated and accessible during function compilation
10. **No regressions**: All 596+ existing tests pass
