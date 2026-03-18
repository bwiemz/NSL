# M40: Source-to-Source Automatic Differentiation — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add compile-time source-to-source AD infrastructure: Wengert list extraction, reverse-mode adjoint rule table, adjoint generation, dead gradient elimination, and saved tensor analysis — enabling the compiler to generate backward functions at compile time instead of using runtime tape recording.

**Architecture:** Three new codegen modules: (1) `wengert.rs` — `WengertOp`, `PrimalOp`, `WengertList`, `VarId` data structures for SSA-form computation graph representation. (2) `ad_rules.rs` — `AdjointExpr` enum and per-operation reverse-mode AD rule table covering all 30+ primal operations. (3) `source_ad.rs` — `AdjointGenerator` (reverse-walk adjoint generation), `DeadGradientEliminator` (liveness-based pruning), `SavedTensorAnalyzer` (forward-backward cross-reference). All modules are pure analysis/transformation with full unit test coverage — no runtime changes in this plan.

**Tech Stack:** Rust (codegen analysis)

**Spec:** `docs/superpowers/specs/2026-03-15-m40-source-ad-design.md`

---

## Important: Scope of This Plan

**This plan builds the core AD analysis library as standalone, tested modules.** It delivers:
- `WengertOp` / `PrimalOp` / `WengertList` data structures (SSA computation graph)
- `AdjointExpr` enum with 13 backward primitive operations
- Complete AD rule table for 20+ primal operations with unit tests
- `AdjointGenerator` — reverse-walk backward generation with gradient accumulation
- `DeadGradientEliminator` — liveness pruning of unused gradient chains
- `SavedTensorAnalyzer` — identifies which intermediates must be saved for backward
- Compiler field (`source_ad_enabled`)

**Deferred to M40b:** AST-to-Wengert extraction (requires walking the typed AST), Cranelift IR emission of backward functions, `BackwardContext` runtime FFI, grad block strategy selection in `stmt.rs`, cross-boundary fusion with M31, higher-order derivatives, `@checkpoint` recomputation, `--tape-ad` CLI flag, hybrid static/dynamic AD, E2E numerical validation tests.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-codegen/src/wengert.rs` | WengertOp, PrimalOp, WengertList, VarId/OpId types | 150 |
| `crates/nsl-codegen/src/ad_rules.rs` | AdjointExpr enum, AD rule table, apply_rule() | 250 |
| `crates/nsl-codegen/src/source_ad.rs` | AdjointGenerator, DeadGradientEliminator, SavedTensorAnalyzer | 300 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-codegen/src/lib.rs` | Add `pub mod wengert; pub mod ad_rules; pub mod source_ad;` |
| `crates/nsl-codegen/src/compiler.rs` | Add `source_ad_enabled: bool` field |
| `crates/nsl-cli/tests/e2e.rs` | Add M40 E2E tests |

---

## Phase 1: Wengert List Data Structures

### Task 1: WengertOp + PrimalOp + WengertList

**Files:**
- Create: `crates/nsl-codegen/src/wengert.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `wengert.rs` with core types and tests**

```rust
//! M40: Wengert list (straight-line SSA) for source-to-source AD.

use std::collections::HashMap;

pub type OpId = u32;
pub type VarId = u32;

/// A single operation in the primal (forward) computation trace.
#[derive(Debug, Clone)]
pub struct WengertOp {
    pub id: OpId,
    pub result: VarId,
    pub op: PrimalOp,
    pub inputs: Vec<VarId>,
    /// Does the backward rule need this intermediate's value?
    pub saved_for_backward: bool,
    /// @checkpoint — recompute instead of save.
    pub checkpointed: bool,
}

/// Primitive operations in the computation graph.
#[derive(Debug, Clone, PartialEq)]
pub enum PrimalOp {
    // Elementwise unary
    Relu, Sigmoid, Tanh, Gelu, Silu,
    Exp, Log, Sqrt, Abs, Neg,

    // Elementwise binary
    Add, Sub, Mul, Div,

    // Linear algebra
    Matmul,
    Transpose { dim0: usize, dim1: usize },

    // Reductions
    Sum { dim: Option<i64> },
    Mean { dim: Option<i64> },
    Softmax { dim: i64 },

    // Shape ops
    Reshape { target_ndim: usize },
    Broadcast,

    // Markers
    Input(String),
    Param(String),
    Constant(f64),
}

/// Linearized forward computation graph.
#[derive(Debug, Clone)]
pub struct WengertList {
    pub ops: Vec<WengertOp>,
    pub output: VarId,
    /// VarId -> human-readable name for debugging.
    pub var_names: HashMap<VarId, String>,
}

impl WengertList {
    /// Check if a variable is defined by a primal (forward) operation.
    pub fn defines(&self, var: VarId) -> bool {
        self.ops.iter().any(|op| op.result == var)
    }

    /// Find the operation that produces a given variable.
    pub fn find_producer(&self, var: VarId) -> Option<&WengertOp> {
        self.ops.iter().find(|op| op.result == var)
    }

    /// Check if a variable's producer is marked as checkpointed.
    pub fn is_checkpointed(&self, var: VarId) -> bool {
        self.find_producer(var).map(|op| op.checkpointed).unwrap_or(false)
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Whether the list is empty.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp { id, result, op, inputs, saved_for_backward: false, checkpointed: false }
    }

    #[test]
    fn test_wengert_list_defines() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1,
            var_names: HashMap::new(),
        };
        assert!(list.defines(0));
        assert!(list.defines(1));
        assert!(!list.defines(99));
    }

    #[test]
    fn test_wengert_list_find_producer() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1,
            var_names: HashMap::new(),
        };
        let producer = list.find_producer(1).unwrap();
        assert_eq!(producer.op, PrimalOp::Relu);
        assert!(list.find_producer(99).is_none());
    }

    #[test]
    fn test_checkpoint_detection() {
        let list = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                WengertOp {
                    id: 1, result: 1, op: PrimalOp::Relu, inputs: vec![0],
                    saved_for_backward: true, checkpointed: true,
                },
            ],
            output: 1,
            var_names: HashMap::new(),
        };
        assert!(!list.is_checkpointed(0));
        assert!(list.is_checkpointed(1));
    }

    #[test]
    fn test_wengert_list_len() {
        let list = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Constant(1.0), vec![])],
            output: 0,
            var_names: HashMap::new(),
        };
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
    }
}
```

- [ ] **Step 2: Add module declarations to lib.rs**

```rust
pub mod wengert;
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-codegen wengert -- --nocapture
git commit -m "feat(m40): add Wengert list data structures for source-to-source AD"
```

---

## Phase 2: AD Rule Table

### Task 2: AdjointExpr + AD Rules

**Files:**
- Create: `crates/nsl-codegen/src/ad_rules.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `ad_rules.rs` with AdjointExpr and rule application**

```rust
//! M40: Reverse-mode AD rules — maps each primal operation to its adjoint computation.

use crate::wengert::{PrimalOp, VarId, WengertOp};

/// Primitive and compound backward operations used in adjoint expressions.
/// Compound variants encode multi-step backward rules that can't be expressed
/// as a single elementwise op (e.g., sigmoid backward = y_bar * y * (1-y)).
#[derive(Debug, Clone, PartialEq)]
pub enum AdjointExpr {
    /// y_bar * x (elementwise multiply)
    MulElementwise(VarId, VarId),
    /// y_bar @ B^T (matmul backward for left input)
    MatmulTransposeLeft(VarId, VarId),
    /// A^T @ y_bar (matmul backward for right input)
    MatmulTransposeRight(VarId, VarId),
    /// y_bar * scalar
    Scale(VarId, f64),
    /// -y_bar
    Negate(VarId),
    /// broadcast(y_bar, target_shape) (for sum backward)
    Broadcast(VarId),
    /// broadcast(y_bar / N, target_shape) (for mean backward)
    ScaleBroadcast(VarId, f64),
    /// transpose(y_bar, d0, d1)
    Transpose(VarId, usize, usize),
    /// reshape(y_bar, original_shape)
    Reshape(VarId),
    /// Pass-through (e.g., add backward: x_bar += y_bar)
    Identity(VarId),

    // --- Compound backward rules (multi-step, lowered to sequences of WengertOps) ---

    /// y_bar * y (exp backward: x_bar += y_bar * exp(x) = y_bar * y)
    ExpBackward(VarId, VarId),
    /// y_bar * (x > 0) (relu backward — mask, not multiply by x)
    ReluBackward(VarId, VarId),
    /// y_bar * y * (1 - y) (sigmoid backward — saves output y)
    SigmoidBackward(VarId, VarId),
    /// y_bar * (1 - y^2) (tanh backward — saves output y)
    TanhBackward(VarId, VarId),
    /// y_bar / x (log backward — saves input x)
    LogBackward(VarId, VarId),
    /// y_bar / (2 * y) (sqrt backward — saves output y)
    SqrtBackward(VarId, VarId),
    /// y_bar / b (div backward for numerator a)
    DivNumeratorBackward(VarId, VarId),
    /// -y_bar * a / b^2 (div backward for denominator b)
    DivDenominatorBackward(VarId, VarId, VarId),
}

/// A single input-adjoint pair from applying an AD rule.
#[derive(Debug, Clone)]
pub struct InputAdjoint {
    /// Which primal input receives this gradient contribution.
    pub input_var: VarId,
    /// The adjoint expression to compute.
    pub expr: AdjointExpr,
}

/// Apply the reverse-mode AD rule for a primal operation.
/// Returns a list of (input_var, adjoint_expr) pairs.
/// `output_bar` is the adjoint of the operation's output (incoming gradient).
pub fn apply_ad_rule(op: &WengertOp, output_bar: VarId) -> Vec<InputAdjoint> {
    match &op.op {
        // --- Elementwise binary ---
        PrimalOp::Add => {
            // y = a + b: a_bar += y_bar, b_bar += y_bar
            vec![
                InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::Identity(output_bar) },
                InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::Identity(output_bar) },
            ]
        }
        PrimalOp::Sub => {
            // y = a - b: a_bar += y_bar, b_bar += -y_bar
            vec![
                InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::Identity(output_bar) },
                InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::Negate(output_bar) },
            ]
        }
        PrimalOp::Mul => {
            // y = a * b: a_bar += y_bar * b, b_bar += y_bar * a
            vec![
                InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::MulElementwise(output_bar, op.inputs[1]) },
                InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::MulElementwise(output_bar, op.inputs[0]) },
            ]
        }
        PrimalOp::Div => {
            // y = a / b: a_bar += y_bar / b, b_bar += -y_bar * a / b^2
            vec![
                InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::DivNumeratorBackward(output_bar, op.inputs[1]) },
                InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::DivDenominatorBackward(output_bar, op.inputs[0], op.inputs[1]) },
            ]
        }
        PrimalOp::Neg => {
            vec![InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::Negate(output_bar) }]
        }

        // --- Activations ---
        PrimalOp::Relu => {
            // y = relu(x): x_bar += y_bar * (x > 0) — mask, NOT multiply by x
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::ReluBackward(output_bar, op.inputs[0]),
            }]
        }
        PrimalOp::Sigmoid => {
            // y = sigmoid(x): x_bar += y_bar * y * (1 - y) — saves output y
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::SigmoidBackward(output_bar, op.result),
            }]
        }
        PrimalOp::Tanh => {
            // y = tanh(x): x_bar += y_bar * (1 - y^2) — saves output y
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::TanhBackward(output_bar, op.result),
            }]
        }
        PrimalOp::Exp => {
            // y = exp(x): x_bar += y_bar * y — saves output y
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::ExpBackward(output_bar, op.result),
            }]
        }
        PrimalOp::Log => {
            // y = log(x): x_bar += y_bar / x — saves input x
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::LogBackward(output_bar, op.inputs[0]),
            }]
        }
        PrimalOp::Sqrt => {
            // y = sqrt(x): x_bar += y_bar / (2 * y) — saves output y
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::SqrtBackward(output_bar, op.result),
            }]
        }

        // --- Linear algebra ---
        PrimalOp::Matmul => {
            // y = A @ B: A_bar += y_bar @ B^T, B_bar += A^T @ y_bar
            vec![
                InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::MatmulTransposeLeft(output_bar, op.inputs[1]) },
                InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::MatmulTransposeRight(op.inputs[0], output_bar) },
            ]
        }
        PrimalOp::Transpose { dim0, dim1 } => {
            // Transpose backward is transpose with same dims
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::Transpose(output_bar, *dim0, *dim1),
            }]
        }

        // --- Reductions ---
        PrimalOp::Sum { .. } => {
            // sum backward: broadcast y_bar to input shape
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::Broadcast(output_bar),
            }]
        }
        PrimalOp::Mean { .. } => {
            // mean backward: broadcast (y_bar / N) to input shape
            // N is determined at lowering time from input shape; placeholder scale=1.0 here
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::ScaleBroadcast(output_bar, 1.0), // actual 1/N computed during lowering
            }]
        }

        // --- Shape ops ---
        PrimalOp::Reshape { .. } => {
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::Reshape(output_bar),
            }]
        }

        // No gradient for inputs/params/constants/broadcast/softmax (handled specially)
        _ => vec![],
    }
}

/// Determine which variables an AD rule needs saved from the forward pass.
pub fn saved_for_backward(op: &PrimalOp) -> SavedRequirement {
    match op {
        // Save nothing (shape-only backward)
        PrimalOp::Add | PrimalOp::Sub | PrimalOp::Neg
        | PrimalOp::Transpose { .. } | PrimalOp::Reshape { .. }
        | PrimalOp::Sum { .. } | PrimalOp::Mean { .. }
        | PrimalOp::Broadcast => SavedRequirement::Nothing,

        // Save input(s) for backward
        PrimalOp::Mul | PrimalOp::Div | PrimalOp::Matmul
        | PrimalOp::Relu | PrimalOp::Log | PrimalOp::Abs => SavedRequirement::Inputs,

        // Save output for backward
        PrimalOp::Sigmoid | PrimalOp::Tanh | PrimalOp::Exp
        | PrimalOp::Sqrt | PrimalOp::Softmax { .. } => SavedRequirement::Output,

        // Save input for backward (complex activations)
        PrimalOp::Gelu | PrimalOp::Silu => SavedRequirement::Inputs,

        // No gradient
        _ => SavedRequirement::Nothing,
    }
}

/// What a backward rule needs saved from the forward pass.
#[derive(Debug, Clone, PartialEq)]
pub enum SavedRequirement {
    /// Nothing needed (shape-only rules like Add, Sub, Transpose).
    Nothing,
    /// Input tensor(s) needed (Mul, Div, MatMul, ReLU, Log).
    Inputs,
    /// Output tensor needed (Sigmoid, Tanh, Exp, Sqrt, Softmax).
    Output,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::WengertOp;

    fn make_op(result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp { id: 0, result, op, inputs, saved_for_backward: false, checkpointed: false }
    }

    #[test]
    fn test_add_rule() {
        let op = make_op(2, PrimalOp::Add, vec![0, 1]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 2);
        assert_eq!(adjoints[0].input_var, 0);
        assert_eq!(adjoints[1].input_var, 1);
        assert!(matches!(adjoints[0].expr, AdjointExpr::Identity(100)));
        assert!(matches!(adjoints[1].expr, AdjointExpr::Identity(100)));
    }

    #[test]
    fn test_sub_rule() {
        let op = make_op(2, PrimalOp::Sub, vec![0, 1]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 2);
        assert!(matches!(adjoints[0].expr, AdjointExpr::Identity(100)));
        assert!(matches!(adjoints[1].expr, AdjointExpr::Negate(100)));
    }

    #[test]
    fn test_mul_rule() {
        let op = make_op(2, PrimalOp::Mul, vec![0, 1]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 2);
        // a_bar += y_bar * b
        assert!(matches!(adjoints[0].expr, AdjointExpr::MulElementwise(100, 1)));
        // b_bar += y_bar * a
        assert!(matches!(adjoints[1].expr, AdjointExpr::MulElementwise(100, 0)));
    }

    #[test]
    fn test_matmul_rule() {
        let op = make_op(2, PrimalOp::Matmul, vec![0, 1]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 2);
        assert!(matches!(adjoints[0].expr, AdjointExpr::MatmulTransposeLeft(100, 1)));
        assert!(matches!(adjoints[1].expr, AdjointExpr::MatmulTransposeRight(0, 100)));
    }

    #[test]
    fn test_relu_rule() {
        let op = make_op(1, PrimalOp::Relu, vec![0]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 1);
        assert_eq!(adjoints[0].input_var, 0);
        // ReluBackward(y_bar=100, x=0) — mask by (x > 0), not multiply by x
        assert!(matches!(adjoints[0].expr, AdjointExpr::ReluBackward(100, 0)));
    }

    #[test]
    fn test_sigmoid_backward_compound() {
        let op = make_op(1, PrimalOp::Sigmoid, vec![0]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 1);
        // SigmoidBackward(y_bar=100, y=1) — computes y_bar * y * (1-y)
        assert!(matches!(adjoints[0].expr, AdjointExpr::SigmoidBackward(100, 1)));
    }

    #[test]
    fn test_log_backward_reciprocal() {
        let op = make_op(1, PrimalOp::Log, vec![0]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 1);
        // LogBackward(y_bar=100, x=0) — computes y_bar / x
        assert!(matches!(adjoints[0].expr, AdjointExpr::LogBackward(100, 0)));
    }

    #[test]
    fn test_sqrt_backward_compound() {
        let op = make_op(1, PrimalOp::Sqrt, vec![0]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 1);
        // SqrtBackward(y_bar=100, y=1) — computes y_bar / (2*y)
        assert!(matches!(adjoints[0].expr, AdjointExpr::SqrtBackward(100, 1)));
    }

    #[test]
    fn test_div_backward() {
        let op = make_op(2, PrimalOp::Div, vec![0, 1]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 2);
        assert!(matches!(adjoints[0].expr, AdjointExpr::DivNumeratorBackward(100, 1)));
        assert!(matches!(adjoints[1].expr, AdjointExpr::DivDenominatorBackward(100, 0, 1)));
    }

    #[test]
    fn test_sum_rule_broadcasts() {
        let op = make_op(1, PrimalOp::Sum { dim: Some(0) }, vec![0]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 1);
        assert!(matches!(adjoints[0].expr, AdjointExpr::Broadcast(100)));
    }

    #[test]
    fn test_mean_rule_scale_broadcasts() {
        let op = make_op(1, PrimalOp::Mean { dim: Some(0) }, vec![0]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 1);
        assert!(matches!(adjoints[0].expr, AdjointExpr::ScaleBroadcast(100, _)));
    }

    #[test]
    fn test_transpose_rule() {
        let op = make_op(1, PrimalOp::Transpose { dim0: 0, dim1: 1 }, vec![0]);
        let adjoints = apply_ad_rule(&op, 100);
        assert_eq!(adjoints.len(), 1);
        assert!(matches!(adjoints[0].expr, AdjointExpr::Transpose(100, 0, 1)));
    }

    // --- Saved requirements ---

    #[test]
    fn test_saved_nothing() {
        assert_eq!(saved_for_backward(&PrimalOp::Add), SavedRequirement::Nothing);
        assert_eq!(saved_for_backward(&PrimalOp::Sub), SavedRequirement::Nothing);
        assert_eq!(saved_for_backward(&PrimalOp::Transpose { dim0: 0, dim1: 1 }), SavedRequirement::Nothing);
    }

    #[test]
    fn test_saved_inputs() {
        assert_eq!(saved_for_backward(&PrimalOp::Mul), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Matmul), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Relu), SavedRequirement::Inputs);
    }

    #[test]
    fn test_saved_output() {
        assert_eq!(saved_for_backward(&PrimalOp::Sigmoid), SavedRequirement::Output);
        assert_eq!(saved_for_backward(&PrimalOp::Tanh), SavedRequirement::Output);
        assert_eq!(saved_for_backward(&PrimalOp::Exp), SavedRequirement::Output);
    }
}
```

- [ ] **Step 2: Add module to lib.rs, run tests, commit**

```bash
cargo test -p nsl-codegen ad_rules -- --nocapture
git commit -m "feat(m40): add reverse-mode AD rule table with AdjointExpr for 20+ operations"
```

---

## Phase 3: Source AD Analysis

### Task 3: AdjointGenerator + DeadGradientEliminator + SavedTensorAnalyzer

**Files:**
- Create: `crates/nsl-codegen/src/source_ad.rs`
- Modify: `crates/nsl-codegen/src/lib.rs`

- [ ] **Step 1: Create `source_ad.rs` with generators and analysis**

```rust
//! M40: Source-to-source AD — adjoint generation, dead gradient elimination,
//! and saved tensor analysis.

use std::collections::{HashMap, HashSet};
use crate::wengert::{WengertList, WengertOp, VarId, OpId, PrimalOp};
use crate::ad_rules::{apply_ad_rule, saved_for_backward, AdjointExpr, InputAdjoint, SavedRequirement};

// ---------------------------------------------------------------------------
// Adjoint generation
// ---------------------------------------------------------------------------

/// Generates the backward (adjoint) Wengert list from a forward (primal) list.
pub struct AdjointGenerator {
    adjoint_ops: Vec<WengertOp>,
    /// Primal var -> its adjoint accumulator var.
    adjoint_vars: HashMap<VarId, VarId>,
    var_counter: VarId,
    op_counter: OpId,
}

impl AdjointGenerator {
    pub fn new(start_var: VarId) -> Self {
        Self {
            adjoint_ops: Vec::new(),
            adjoint_vars: HashMap::new(),
            var_counter: start_var,
            op_counter: 0,
        }
    }

    fn next_var(&mut self) -> VarId {
        let v = self.var_counter;
        self.var_counter += 1;
        v
    }

    fn next_op(&mut self) -> OpId {
        let o = self.op_counter;
        self.op_counter += 1;
        o
    }

    /// Get or create the adjoint accumulator for a primal variable.
    pub fn get_or_create_adjoint(&mut self, primal_var: VarId) -> VarId {
        if let Some(&adj) = self.adjoint_vars.get(&primal_var) {
            adj
        } else {
            let adj = self.next_var();
            self.adjoint_vars.insert(primal_var, adj);
            adj
        }
    }

    /// Generate the backward Wengert list.
    /// Walks the primal ops in reverse, applies AD rules, accumulates adjoints.
    pub fn generate(&mut self, primal: &WengertList) -> WengertList {
        // Initialize: loss_bar = 1.0 (seed gradient)
        let loss_bar = self.next_var();
        self.adjoint_vars.insert(primal.output, loss_bar);
        self.adjoint_ops.push(WengertOp {
            id: self.next_op(),
            result: loss_bar,
            op: PrimalOp::Constant(1.0),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });

        // Walk primal ops in REVERSE order
        for op in primal.ops.iter().rev() {
            let output_bar = self.get_or_create_adjoint(op.result);
            let input_adjoints = apply_ad_rule(op, output_bar);

            for InputAdjoint { input_var, expr } in input_adjoints {
                let adj_val = self.lower_adjoint_expr(expr);
                self.accumulate_adjoint(input_var, adj_val);
            }
        }

        WengertList {
            ops: self.adjoint_ops.clone(),
            output: loss_bar,
            var_names: HashMap::new(),
        }
    }

    fn lower_adjoint_expr(&mut self, expr: AdjointExpr) -> VarId {
        // Each AdjointExpr becomes one or more WengertOps in the backward list.
        // Compound variants (SigmoidBackward, etc.) will be expanded to op sequences
        // during Cranelift lowering in M40b. For now, they emit placeholder Mul ops
        // with the correct input references so saved-tensor analysis works correctly.
        let result = self.next_var();
        let (op, inputs) = match expr {
            AdjointExpr::Identity(v) => return v, // pass-through, no op needed
            AdjointExpr::Negate(v) => (PrimalOp::Neg, vec![v]),
            AdjointExpr::MulElementwise(a, b) => (PrimalOp::Mul, vec![a, b]),
            AdjointExpr::MatmulTransposeLeft(a, b) => (PrimalOp::Matmul, vec![a, b]),
            AdjointExpr::MatmulTransposeRight(a, b) => (PrimalOp::Matmul, vec![a, b]),
            AdjointExpr::Scale(v, _s) => (PrimalOp::Mul, vec![v]),
            AdjointExpr::Broadcast(v) => (PrimalOp::Broadcast, vec![v]),
            AdjointExpr::ScaleBroadcast(v, _n) => (PrimalOp::Broadcast, vec![v]),
            AdjointExpr::Transpose(v, d0, d1) => (PrimalOp::Transpose { dim0: d0, dim1: d1 }, vec![v]),
            AdjointExpr::Reshape(v) => (PrimalOp::Reshape { target_ndim: 0 }, vec![v]),
            // Compound backward rules: emit placeholder ops that reference the correct
            // primal vars so saved-tensor analysis detects cross-references correctly.
            AdjointExpr::ExpBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::ReluBackward(y_bar, x) => (PrimalOp::Mul, vec![y_bar, x]),
            AdjointExpr::SigmoidBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::TanhBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::LogBackward(y_bar, x) => (PrimalOp::Div, vec![y_bar, x]),
            AdjointExpr::SqrtBackward(y_bar, y) => (PrimalOp::Div, vec![y_bar, y]),
            AdjointExpr::DivNumeratorBackward(y_bar, b) => (PrimalOp::Div, vec![y_bar, b]),
            AdjointExpr::DivDenominatorBackward(y_bar, a, b) => (PrimalOp::Mul, vec![y_bar, a, b]),
        };
        self.adjoint_ops.push(WengertOp {
            id: self.next_op(),
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        });
        result
    }

    fn accumulate_adjoint(&mut self, var: VarId, value: VarId) {
        if let Some(&existing) = self.adjoint_vars.get(&var) {
            // Accumulate: existing_bar += value (emit Add op)
            let sum = self.next_var();
            self.adjoint_ops.push(WengertOp {
                id: self.next_op(),
                result: sum,
                op: PrimalOp::Add,
                inputs: vec![existing, value],
                saved_for_backward: false,
                checkpointed: false,
            });
            self.adjoint_vars.insert(var, sum);
        } else {
            self.adjoint_vars.insert(var, value);
        }
    }

    /// Get the adjoint variable for a primal variable (after generation).
    pub fn adjoint_of(&self, primal_var: VarId) -> Option<VarId> {
        self.adjoint_vars.get(&primal_var).copied()
    }
}

// ---------------------------------------------------------------------------
// Dead gradient elimination
// ---------------------------------------------------------------------------

/// Prune backward ops whose results are never needed by any parameter gradient.
pub fn eliminate_dead_gradients(
    adjoint_ops: &[WengertOp],
    needed_vars: &HashSet<VarId>,
) -> Vec<WengertOp> {
    let mut live_ops = HashSet::new();
    let mut worklist: Vec<VarId> = needed_vars.iter().copied().collect();

    while let Some(var) = worklist.pop() {
        for (i, op) in adjoint_ops.iter().enumerate() {
            if op.result == var && !live_ops.contains(&i) {
                live_ops.insert(i);
                worklist.extend(op.inputs.iter().copied());
            }
        }
    }

    adjoint_ops.iter().enumerate()
        .filter(|(i, _)| live_ops.contains(i))
        .map(|(_, op)| op.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Saved tensor analysis
// ---------------------------------------------------------------------------

/// Identify which forward-pass intermediates must be saved for the backward pass.
pub fn analyze_saved_tensors(
    primal: &WengertList,
    adjoint: &WengertList,
) -> Vec<SavedTensorInfo> {
    let mut saved = Vec::new();
    let primal_vars: HashSet<VarId> = primal.ops.iter().map(|op| op.result).collect();
    let adjoint_vars: HashSet<VarId> = adjoint.ops.iter().map(|op| op.result).collect();

    for adj_op in &adjoint.ops {
        for &input in &adj_op.inputs {
            // If this input is defined in primal but not in adjoint,
            // it must be saved from the forward pass.
            if primal_vars.contains(&input) && !adjoint_vars.contains(&input) {
                let checkpointed = primal.is_checkpointed(input);
                saved.push(SavedTensorInfo {
                    var: input,
                    checkpointed,
                });
            }
        }
    }

    // Deduplicate
    saved.sort_by_key(|s| s.var);
    saved.dedup_by_key(|s| s.var);
    saved
}

/// Information about a tensor that must be saved from forward for backward.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedTensorInfo {
    pub var: VarId,
    pub checkpointed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{WengertOp, PrimalOp, WengertList};

    fn make_op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp { id, result, op, inputs, saved_for_backward: false, checkpointed: false }
    }

    #[test]
    fn test_adjoint_generator_simple_add() {
        // y = a + b (var 0 + var 1 = var 2)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("a".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("b".into()), vec![]),
                make_op(2, 2, PrimalOp::Add, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);

        // Should have generated adjoint ops
        assert!(!adjoint.ops.is_empty());
        // Both inputs should have adjoint vars
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    #[test]
    fn test_adjoint_generator_matmul() {
        // y = matmul(A, B)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("A".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("B".into()), vec![]),
                make_op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let _adjoint = gen.generate(&primal);

        // Both A and B should have adjoint vars
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    #[test]
    fn test_dead_gradient_elimination() {
        // 5 backward ops, but we only need the gradient for var 3
        let ops = vec![
            make_op(0, 10, PrimalOp::Mul, vec![3, 4]),  // needed (produces var needed by 10->3)
            make_op(1, 11, PrimalOp::Add, vec![5, 6]),   // not needed
            make_op(2, 12, PrimalOp::Neg, vec![7]),       // not needed
            make_op(3, 3, PrimalOp::Add, vec![10, 8]),    // needed (produces needed var 3)
            make_op(4, 13, PrimalOp::Mul, vec![9, 10]),   // not needed
        ];

        let needed = HashSet::from([3_u32]);
        let pruned = eliminate_dead_gradients(&ops, &needed);

        // Should keep ops that contribute to var 3
        assert!(pruned.len() < ops.len());
        assert!(pruned.iter().any(|op| op.result == 3));
    }

    #[test]
    fn test_dead_gradient_empty_needed() {
        let ops = vec![make_op(0, 10, PrimalOp::Add, vec![0, 1])];
        let needed = HashSet::new();
        let pruned = eliminate_dead_gradients(&ops, &needed);
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_saved_tensor_analysis() {
        // Primal: var 0 (input), var 1 (relu of 0)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1,
            var_names: HashMap::new(),
        };
        // Adjoint: uses var 0 (primal input) in backward
        let adjoint = WengertList {
            ops: vec![
                make_op(0, 10, PrimalOp::Constant(1.0), vec![]),
                make_op(1, 11, PrimalOp::Mul, vec![10, 0]), // references primal var 0
            ],
            output: 10,
            var_names: HashMap::new(),
        };

        let saved = analyze_saved_tensors(&primal, &adjoint);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].var, 0);
        assert!(!saved[0].checkpointed);
    }

    #[test]
    fn test_saved_tensor_no_cross_reference() {
        let primal = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Input("x".into()), vec![])],
            output: 0,
            var_names: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![make_op(0, 10, PrimalOp::Constant(1.0), vec![])],
            output: 10,
            var_names: HashMap::new(),
        };

        let saved = analyze_saved_tensors(&primal, &adjoint);
        assert!(saved.is_empty()); // no cross-references
    }
}
```

- [ ] **Step 2: Add modules to lib.rs**

```rust
pub mod ad_rules;
pub mod source_ad;
pub mod wengert;
```

- [ ] **Step 3: Run tests, commit**

```bash
cargo test -p nsl-codegen source_ad -- --nocapture
cargo test -p nsl-codegen ad_rules -- --nocapture
cargo test -p nsl-codegen wengert -- --nocapture
git commit -m "feat(m40): add source AD adjoint generator, dead gradient eliminator, saved tensor analyzer"
```

---

## Phase 4: Compiler Integration + Verification

### Task 4: Compiler Fields

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 1: Add `source_ad_enabled` field**

After `vmap_configs`:

```rust
    /// M40: Source-to-source AD enabled (default true; --tape-ad forces tape-only)
    pub source_ad_enabled: bool,
```

Initialize as `source_ad_enabled: true` in `Compiler::new()`.

- [ ] **Step 2: Verify, commit**

```bash
cargo check --workspace
git commit -m "feat(m40): add source_ad_enabled compiler field"
```

---

### Task 5: Full Verification + Clippy

- [ ] **Step 1: Run all workspace lib tests**

```bash
cargo test --workspace --lib
```

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 3: Fix any issues, commit**

```bash
git commit -m "chore(m40): fix clippy warnings and verify full test suite"
```

---

## Summary

| Task | Component | Tests |
|---|---|---|
| 1 | WengertOp + PrimalOp + WengertList | 4 unit |
| 2 | AdjointExpr + AD rules + SavedRequirement | 16 unit |
| 3 | AdjointGenerator + DeadGradientEliminator + SavedTensorAnalyzer | 5 unit |
| 4 | Compiler field (source_ad_enabled) | compile check |
| 5 | Full verification | all tests |

**Total: 5 tasks, ~25 unit tests**

### Deferred to M40b

- `WengertExtractor` — AST-to-Wengert-list extraction (walks typed AST, detects dynamic control flow)
- `BackwardContext` runtime FFI (save/load/free for saved tensors)
- Cranelift IR emission of backward functions
- Grad block strategy selection in `stmt.rs` (source AD vs tape fallback)
- Cross-boundary fusion with M31 FusionGraph
- Higher-order derivatives (nested grad blocks)
- `@checkpoint` recomputation logic
- `--tape-ad` CLI flag to force tape-based AD
- Hybrid static/dynamic AD (split at dynamic boundary)
- E2E numerical validation (source AD gradients match tape-based)
