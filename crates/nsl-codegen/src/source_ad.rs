//! M40: Source-to-source AD — adjoint generation, dead gradient elimination,
//! and saved tensor analysis.

use crate::ad_rules::{apply_ad_rule, AdjointExpr, InputAdjoint};
use crate::wengert::{
    type_for_op, CompareKind, OpId, PrimalOp, VarId, WengertList, WengertOp, WengertType,
};
use std::collections::{HashMap, HashSet};

// M40b: Wengert extraction from typed AST
use nsl_ast::expr::ExprKind;
use nsl_ast::operator::{BinOp, UnaryOp as AstUnaryOp};
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;

// ---------------------------------------------------------------------------
// Adjoint generation
// ---------------------------------------------------------------------------

/// Generates the backward (adjoint) Wengert list from a forward (primal) list.
pub struct AdjointGenerator {
    adjoint_ops: Vec<WengertOp>,
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

    pub fn get_or_create_adjoint(&mut self, primal_var: VarId) -> VarId {
        if let Some(&adj) = self.adjoint_vars.get(&primal_var) {
            adj
        } else {
            // Allocate a new adjoint VarId. No op is emitted yet —
            // the VarId is a "ghost" that gets populated by accumulate_adjoint
            // when a gradient arrives. If no gradient arrives, the VarId
            // stays unmapped and the lowerer should produce zeros_like(primal).
            let adj = self.next_var();
            self.adjoint_vars.insert(primal_var, adj);
            adj
        }
    }

    /// Generate the backward Wengert list by walking primal ops in reverse.
    pub fn generate(&mut self, primal: &WengertList) -> WengertList {
        let loss_bar = self.next_var();
        let seed_op_id = self.next_op();
        self.adjoint_vars.insert(primal.output, loss_bar);
        self.adjoint_ops.push(WengertOp {
            id: seed_op_id,
            result: loss_bar,
            op: PrimalOp::Constant(1.0),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });

        for op in primal.ops.iter().rev() {
            let output_bar = self.get_or_create_adjoint(op.result);
            let input_adjoints = apply_ad_rule(op, output_bar);

            for InputAdjoint { input_var, expr } in input_adjoints {
                let adj_val = self.lower_adjoint_expr(expr);
                self.accumulate_adjoint(input_var, adj_val);
            }
        }

        // Build var_types for the adjoint graph from its ops.
        let mut adjoint_var_types = HashMap::new();
        for op in &self.adjoint_ops {
            adjoint_var_types.insert(op.result, type_for_op(&op.op));
        }
        WengertList {
            ops: self.adjoint_ops.clone(),
            output: loss_bar,
            var_names: HashMap::new(),
            var_types: adjoint_var_types,
        }
    }

    /// Emit a single intermediate op and return its result VarId.
    fn emit_op(&mut self, op: PrimalOp, inputs: Vec<VarId>) -> VarId {
        let result = self.next_var();
        let op_id = self.next_op();
        self.adjoint_ops.push(WengertOp {
            id: op_id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        });
        result
    }

    /// Emit a constant value and return its VarId.
    fn emit_constant(&mut self, value: f64) -> VarId {
        self.emit_op(PrimalOp::Constant(value), vec![])
    }

    fn lower_adjoint_expr(&mut self, expr: AdjointExpr) -> VarId {
        // Identity: pass-through, no op needed
        if let AdjointExpr::Identity(v) = expr {
            return v;
        }

        match expr {
            AdjointExpr::Identity(v) => {
                // Gradient passes through unchanged — emit a "copy" op
                // so the VarId gets registered in the var_map during lowering.
                self.emit_op(PrimalOp::Broadcast, vec![v])
            }

            // --- Simple pass-through ops (single instruction, mathematically correct) ---
            AdjointExpr::Negate(v) => self.emit_op(PrimalOp::Neg, vec![v]),
            AdjointExpr::MulElementwise(a, b) => self.emit_op(PrimalOp::Mul, vec![a, b]),
            AdjointExpr::MatmulTransposeLeft(grad, b) => {
                // d_loss/d_A = grad @ B^T (transpose last two dims for N-D support)
                let b_t = self.emit_op(
                    PrimalOp::Transpose {
                        dim0: usize::MAX - 1,
                        dim1: usize::MAX,
                    },
                    vec![b],
                );
                self.emit_op(PrimalOp::Matmul, vec![grad, b_t])
            }
            AdjointExpr::MatmulTransposeRight(a, grad, b) => {
                // d_loss/d_B = A^T @ grad (transpose last two dims for N-D support)
                // When the forward matmul broadcasts (A has more dims than B),
                // the gradient must be summed over the extra batch dimensions
                // so it matches B's shape.
                let a_t = self.emit_op(
                    PrimalOp::Transpose {
                        dim0: usize::MAX - 1,
                        dim1: usize::MAX,
                    },
                    vec![a],
                );
                let raw_grad = self.emit_op(PrimalOp::Matmul, vec![a_t, grad]);
                // Reduce to match B's shape: nsl_tensor_reduce_to_shape(raw_grad, B)
                self.emit_op(
                    PrimalOp::Passthrough("reduce_to_shape".into()),
                    vec![raw_grad, b],
                )
            }
            AdjointExpr::Scale(v, s) => {
                let scale = self.emit_constant(s);
                self.emit_op(PrimalOp::Mul, vec![v, scale])
            }
            AdjointExpr::Broadcast(v) => self.emit_op(PrimalOp::Broadcast, vec![v]),
            AdjointExpr::ScaleBroadcast(v, n) => {
                let scale = self.emit_constant(n);
                let scaled = self.emit_op(PrimalOp::Mul, vec![v, scale]);
                self.emit_op(PrimalOp::Broadcast, vec![scaled])
            }
            AdjointExpr::Transpose(v, d0, d1) => {
                self.emit_op(PrimalOp::Transpose { dim0: d0, dim1: d1 }, vec![v])
            }
            AdjointExpr::ReshapeLike(v, target) => {
                let shape = self.emit_op(PrimalOp::Passthrough("shape".into()), vec![target]);
                self.emit_op(PrimalOp::Passthrough("reshape".into()), vec![v, shape])
            }
            // --- Expand backward: sum-reduce gradient over broadcast-expanded dims ---
            AdjointExpr::ReduceToShape(grad, target) => self.emit_op(
                PrimalOp::Passthrough("reduce_to_shape".into()),
                vec![grad, target],
            ),

            // --- Exp backward: d(exp(x))/dx = exp(x) = y. grad * y (correct as-is) ---
            AdjointExpr::ExpBackward(y_bar, y) => self.emit_op(PrimalOp::Mul, vec![y_bar, y]),

            // --- ReLU backward: grad * (x > 0), NOT grad * x ---
            AdjointExpr::ReluBackward(y_bar, x) => {
                let zero = self.emit_constant(0.0);
                let cond = self.emit_op(PrimalOp::Condition(CompareKind::Gt), vec![x, zero]);
                self.emit_op(PrimalOp::Select, vec![cond, y_bar, zero])
            }

            // --- Sigmoid backward: grad * σ(x) * (1 - σ(x)), where y = σ(x) ---
            AdjointExpr::SigmoidBackward(y_bar, y) => {
                let one = self.emit_constant(1.0);
                let one_minus_y = self.emit_op(PrimalOp::Sub, vec![one, y]);
                let sig_deriv = self.emit_op(PrimalOp::Mul, vec![y, one_minus_y]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, sig_deriv])
            }

            // --- Tanh backward: grad * (1 - tanh²(x)), where y = tanh(x) ---
            AdjointExpr::TanhBackward(y_bar, y) => {
                let y_sq = self.emit_op(PrimalOp::Mul, vec![y, y]);
                let one = self.emit_constant(1.0);
                let one_minus_y_sq = self.emit_op(PrimalOp::Sub, vec![one, y_sq]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, one_minus_y_sq])
            }

            // --- Log backward: grad / x (correct as-is) ---
            AdjointExpr::LogBackward(y_bar, x) => self.emit_op(PrimalOp::Div, vec![y_bar, x]),

            // --- Sqrt backward: grad / (2 * sqrt(x)), where y = sqrt(x) ---
            AdjointExpr::SqrtBackward(y_bar, y) => {
                let two = self.emit_constant(2.0);
                let two_y = self.emit_op(PrimalOp::Mul, vec![two, y]);
                self.emit_op(PrimalOp::Div, vec![y_bar, two_y])
            }

            // --- Div numerator backward: d(a/b)/da = 1/b, so grad / b ---
            AdjointExpr::DivNumeratorBackward(y_bar, b) => {
                self.emit_op(PrimalOp::Div, vec![y_bar, b])
            }

            // --- Div denominator backward: d(a/b)/db = -a/b², so grad * (-a/b²) ---
            AdjointExpr::DivDenominatorBackward(y_bar, a, b) => {
                let b_sq = self.emit_op(PrimalOp::Mul, vec![b, b]);
                let a_over_b_sq = self.emit_op(PrimalOp::Div, vec![a, b_sq]);
                let neg = self.emit_op(PrimalOp::Neg, vec![a_over_b_sq]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, neg])
            }

            // --- Select backward (control flow) ---
            AdjointExpr::SelectTrue(y_bar, cond) => {
                let zero = self.emit_constant(0.0);
                self.emit_op(PrimalOp::Select, vec![cond, y_bar, zero])
            }
            AdjointExpr::SelectFalse(y_bar, cond) => {
                let zero = self.emit_constant(0.0);
                self.emit_op(PrimalOp::Select, vec![cond, zero, y_bar])
            }

            // =================================================================
            // Compound backward rules — multi-instruction lowerings
            // =================================================================

            // --- GELU backward: d/dx[x * Φ(x)] = Φ(x) + x * φ(x) ---
            // Approximation: GELU'(x) ≈ 0.5*(1+tanh(√(2/π)*(x+0.044715*x³)))
            //   + 0.5*x*sech²(√(2/π)*(x+0.044715*x³))*√(2/π)*(1+3*0.044715*x²)
            // Simplified: we use the identity d/dx[GELU(x)] = σ(1.702*x)*(1 + 1.702*x*(1-σ(1.702*x)))
            // which is the exact derivative of the sigmoid-linear approximation GELU(x) ≈ x*σ(1.702*x)
            AdjointExpr::GeluBackward(y_bar, x) => {
                let k = self.emit_constant(1.702);
                let kx = self.emit_op(PrimalOp::Mul, vec![k, x]);
                let sig_kx = self.emit_op(PrimalOp::Sigmoid, vec![kx]);
                let one = self.emit_constant(1.0);
                let one_minus_sig = self.emit_op(PrimalOp::Sub, vec![one, sig_kx]);
                let kx_term = self.emit_op(PrimalOp::Mul, vec![kx, one_minus_sig]);
                let one_plus_kx_term = self.emit_op(PrimalOp::Add, vec![one, kx_term]);
                let gelu_deriv = self.emit_op(PrimalOp::Mul, vec![sig_kx, one_plus_kx_term]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, gelu_deriv])
            }

            // --- SiLU backward: d/dx[x * σ(x)] = σ(x) * (1 + x * (1 - σ(x))) ---
            AdjointExpr::SiluBackward(y_bar, x) => {
                let sig_x = self.emit_op(PrimalOp::Sigmoid, vec![x]);
                let one = self.emit_constant(1.0);
                let one_minus_sig = self.emit_op(PrimalOp::Sub, vec![one, sig_x]);
                let x_term = self.emit_op(PrimalOp::Mul, vec![x, one_minus_sig]);
                let one_plus_x_term = self.emit_op(PrimalOp::Add, vec![one, x_term]);
                let silu_deriv = self.emit_op(PrimalOp::Mul, vec![sig_x, one_plus_x_term]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, silu_deriv])
            }

            // --- Abs backward: grad * sign(x) ---
            AdjointExpr::SignMul(y_bar, x) => {
                let zero = self.emit_constant(0.0);
                let pos_cond = self.emit_op(PrimalOp::Condition(CompareKind::Gt), vec![x, zero]);
                let one = self.emit_constant(1.0);
                let neg_one = self.emit_constant(-1.0);
                let sign = self.emit_op(PrimalOp::Select, vec![pos_cond, one, neg_one]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, sign])
            }

            // --- Clamp backward: grad * (min <= x <= max) ---
            // Derivative is 1 when x is in [min, max], 0 outside.
            // Implemented as chained Selects: pass grad only if x >= min AND x <= max.
            AdjointExpr::ClampBackward(y_bar, x, min, max) => {
                let zero = self.emit_constant(0.0);
                let min_val = self.emit_constant(min);
                let max_val = self.emit_constant(max);
                let ge_min = self.emit_op(PrimalOp::Condition(CompareKind::GtEq), vec![x, min_val]);
                let le_max = self.emit_op(PrimalOp::Condition(CompareKind::LtEq), vec![x, max_val]);
                let pass_if_le_max = self.emit_op(PrimalOp::Select, vec![le_max, y_bar, zero]);
                self.emit_op(PrimalOp::Select, vec![ge_min, pass_if_le_max, zero])
            }

            // --- Softmax backward: s * (grad - dot(grad, s)) where s = softmax output ---
            // The Sum{dim:-1} reduces the last dimension, so we must Broadcast
            // back to the full shape before subtracting from the un-reduced grad.
            AdjointExpr::SoftmaxBackward(y_bar, y) => {
                let dot = self.emit_op(PrimalOp::Mul, vec![y_bar, y]);
                let dot_sum = self.emit_op(PrimalOp::Sum { dim: Some(-1) }, vec![dot]);
                let dot_sum_bc = self.emit_op(PrimalOp::Broadcast, vec![dot_sum]);
                let diff = self.emit_op(PrimalOp::Sub, vec![y_bar, dot_sum_bc]);
                self.emit_op(PrimalOp::Mul, vec![y, diff])
            }

            // --- LogSoftmax backward: grad - exp(y) * sum(grad) ---
            // The Sum{dim:-1} reduces the last dimension, so we must Broadcast
            // back to the full shape before multiplying with exp(y).
            AdjointExpr::LogSoftmaxBackward(y_bar, y) => {
                let exp_y = self.emit_op(PrimalOp::Exp, vec![y]);
                let grad_sum = self.emit_op(PrimalOp::Sum { dim: Some(-1) }, vec![y_bar]);
                let grad_sum_bc = self.emit_op(PrimalOp::Broadcast, vec![grad_sum]);
                let correction = self.emit_op(PrimalOp::Mul, vec![exp_y, grad_sum_bc]);
                self.emit_op(PrimalOp::Sub, vec![y_bar, correction])
            }

            // --- LayerNorm backward: dx = rstd * (grad - mean(grad) - x_hat * mean(grad * x_hat)) ---
            // Recomputes mean/rstd from input (not op.result) for correctness.
            // Every Mean{dim} reduction MUST be followed by Broadcast before use in
            // Sub/Mul with full-shape tensors to avoid shape mismatch.
            AdjointExpr::LayerNormBackward(y_bar, x, _mean_unused, _rstd_unused, eps_val) => {
                // Recompute mean and rstd from input (standard approach, matches PyTorch)
                let mean = self.emit_op(PrimalOp::Mean { dim: Some(-1) }, vec![x]);
                let mean_bc = self.emit_op(PrimalOp::Broadcast, vec![mean]);
                let x_centered = self.emit_op(PrimalOp::Sub, vec![x, mean_bc]);
                let x_sq = self.emit_op(PrimalOp::Mul, vec![x_centered, x_centered]);
                let var = self.emit_op(PrimalOp::Mean { dim: Some(-1) }, vec![x_sq]);
                let eps = self.emit_constant(eps_val);
                let var_eps = self.emit_op(PrimalOp::Add, vec![var, eps]);
                let std = self.emit_op(PrimalOp::Sqrt, vec![var_eps]);
                let one = self.emit_constant(1.0);
                let rstd = self.emit_op(PrimalOp::Div, vec![one, std]);
                let rstd_bc = self.emit_op(PrimalOp::Broadcast, vec![rstd]);
                let x_hat = self.emit_op(PrimalOp::Mul, vec![x_centered, rstd_bc]);
                // Compute gradient corrections
                let mean_grad = self.emit_op(PrimalOp::Mean { dim: Some(-1) }, vec![y_bar]);
                let mean_grad_bc = self.emit_op(PrimalOp::Broadcast, vec![mean_grad]);
                let grad_x_hat = self.emit_op(PrimalOp::Mul, vec![y_bar, x_hat]);
                let mean_gxh = self.emit_op(PrimalOp::Mean { dim: Some(-1) }, vec![grad_x_hat]);
                let mean_gxh_bc = self.emit_op(PrimalOp::Broadcast, vec![mean_gxh]);
                let correction = self.emit_op(PrimalOp::Mul, vec![x_hat, mean_gxh_bc]);
                let t1 = self.emit_op(PrimalOp::Sub, vec![y_bar, mean_grad_bc]);
                let t2 = self.emit_op(PrimalOp::Sub, vec![t1, correction]);
                self.emit_op(PrimalOp::Mul, vec![t2, rstd_bc])
            }

            // --- BatchNorm backward: same structure as LayerNorm but over batch dim (dim=0) ---
            // Recomputes mean/rstd from input for correctness.
            // Every Mean{dim} reduction MUST be followed by Broadcast before use in
            // Sub/Mul with full-shape tensors to avoid shape mismatch.
            AdjointExpr::BatchNormBackward(y_bar, x, _mean_unused, _rstd_unused, eps_val) => {
                // Recompute mean and rstd from input over batch dimension
                let mean = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![x]);
                let mean_bc = self.emit_op(PrimalOp::Broadcast, vec![mean]);
                let x_centered = self.emit_op(PrimalOp::Sub, vec![x, mean_bc]);
                let x_sq = self.emit_op(PrimalOp::Mul, vec![x_centered, x_centered]);
                let var = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![x_sq]);
                let eps = self.emit_constant(eps_val);
                let var_eps = self.emit_op(PrimalOp::Add, vec![var, eps]);
                let std = self.emit_op(PrimalOp::Sqrt, vec![var_eps]);
                let one = self.emit_constant(1.0);
                let rstd = self.emit_op(PrimalOp::Div, vec![one, std]);
                let rstd_bc = self.emit_op(PrimalOp::Broadcast, vec![rstd]);
                let x_hat = self.emit_op(PrimalOp::Mul, vec![x_centered, rstd_bc]);
                // Compute gradient corrections
                let mean_grad = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![y_bar]);
                let mean_grad_bc = self.emit_op(PrimalOp::Broadcast, vec![mean_grad]);
                let grad_x_hat = self.emit_op(PrimalOp::Mul, vec![y_bar, x_hat]);
                let mean_gxh = self.emit_op(PrimalOp::Mean { dim: Some(0) }, vec![grad_x_hat]);
                let mean_gxh_bc = self.emit_op(PrimalOp::Broadcast, vec![mean_gxh]);
                let correction = self.emit_op(PrimalOp::Mul, vec![x_hat, mean_gxh_bc]);
                let t1 = self.emit_op(PrimalOp::Sub, vec![y_bar, mean_grad_bc]);
                let t2 = self.emit_op(PrimalOp::Sub, vec![t1, correction]);
                self.emit_op(PrimalOp::Mul, vec![t2, rstd_bc])
            }

            // --- Normalization gamma backward: grad * x_hat ---
            // Recomputes x_hat = (x - mean) / std from input to get the correct
            // normalized values (NOT the output, which is gamma * x_hat + beta).
            AdjointExpr::NormGammaBackward(y_bar, x, eps_val, dim, weight) => {
                let mean = self.emit_op(PrimalOp::Mean { dim: Some(dim) }, vec![x]);
                let mean_bc = self.emit_op(PrimalOp::Broadcast, vec![mean]);
                let x_centered = self.emit_op(PrimalOp::Sub, vec![x, mean_bc]);
                let x_sq = self.emit_op(PrimalOp::Mul, vec![x_centered, x_centered]);
                let var = self.emit_op(PrimalOp::Mean { dim: Some(dim) }, vec![x_sq]);
                let eps = self.emit_constant(eps_val);
                let var_eps = self.emit_op(PrimalOp::Add, vec![var, eps]);
                let std = self.emit_op(PrimalOp::Sqrt, vec![var_eps]);
                let one = self.emit_constant(1.0);
                let rstd = self.emit_op(PrimalOp::Div, vec![one, std]);
                let rstd_bc = self.emit_op(PrimalOp::Broadcast, vec![rstd]);
                let x_hat = self.emit_op(PrimalOp::Mul, vec![x_centered, rstd_bc]);
                // dgamma = reduce_to_shape(grad * x_hat, weight)
                // The weight has shape [last_dim], so we sum over all dims except the last.
                // Using reduce_to_shape handles arbitrary input ndim.
                let grad_x_hat = self.emit_op(PrimalOp::Mul, vec![y_bar, x_hat]);
                self.emit_op(
                    PrimalOp::Passthrough("reduce_to_shape".into()),
                    vec![grad_x_hat, weight],
                )
            }

            // --- Dropout backward: grad * mask * (1/(1-p)) ---
            AdjointExpr::DropoutBackward(y_bar, mask, scale) => {
                let masked = self.emit_op(PrimalOp::Mul, vec![y_bar, mask]);
                let inv_keep = self.emit_constant(scale);
                self.emit_op(PrimalOp::Mul, vec![masked, inv_keep])
            }

            // --- Embedding backward: scatter_add(grad, indices) into weight-shaped gradient ---
            // Uses dedicated runtime function that creates a zeros tensor matching
            // the weight's shape and scatter-adds gradient rows at index positions.
            AdjointExpr::EmbeddingBackward(y_bar, indices, weight) => self.emit_op(
                PrimalOp::Passthrough("embedding_backward".into()),
                vec![y_bar, indices, weight],
            ),

            // --- Gather backward: scatter_add(grad, indices, dim) ---
            AdjointExpr::GatherBackward(y_bar, indices, dim) => {
                self.emit_op(PrimalOp::ScatterAdd { dim }, vec![y_bar, indices])
            }

            // --- ScatterAdd src backward: gather(grad, indices, dim) ---
            AdjointExpr::ScatterAddSrcBackward(y_bar, indices, dim) => {
                self.emit_op(PrimalOp::Gather { dim }, vec![y_bar, indices])
            }

            // --- Shape backward ops ---
            AdjointExpr::ConcatSplit(y_bar, dim, offset, size) => self.emit_op(
                PrimalOp::Slice {
                    dim,
                    start: offset as i64,
                    end: (offset + size) as i64,
                    orig_dim_size: 0,
                },
                vec![y_bar],
            ),
            AdjointExpr::SplitConcat(y_bar, _dim) => {
                // Split backward = concat the gradient pieces (handled by accumulate_adjoint)
                self.emit_op(PrimalOp::Reshape { target_ndim: 0 }, vec![y_bar])
            }
            AdjointExpr::SliceBackward(y_bar, dim, start, end, orig_dim_size) => {
                // Slice backward = zero-pad grad into original shape along the sliced dim.
                // pad_before = start (zeros before the slice region)
                // pad_after = orig_dim_size - end (zeros after the slice region)
                let pad_after = orig_dim_size - end;
                self.emit_op(
                    PrimalOp::PadZero {
                        dim,
                        pad_before: start,
                        pad_after,
                    },
                    vec![y_bar],
                )
            }

            // --- Conv/Pool backward ---
            AdjointExpr::ConvTransposeInput(y_bar, weight, stride, padding) => self.emit_op(
                PrimalOp::ConvTranspose2d { stride, padding },
                vec![y_bar, weight],
            ),
            AdjointExpr::ConvTransposeWeight(input, y_bar) => {
                // Conv weight gradient requires im2col + matmul (cross-correlation),
                // not a plain matmul on 4D tensors. The tape-based backward in
                // backward.rs implements the correct nested loop. For source-AD,
                // we transpose the input and compute the correlation via matmul
                // on the flattened spatial dimensions.
                // TODO(M40c): implement proper im2col + matmul for 4D conv grad.
                // For now, transpose input and matmul (correct for 2D FC-like convs).
                let input_t = self.emit_op(PrimalOp::Transpose { dim0: 0, dim1: 1 }, vec![input]);
                self.emit_op(PrimalOp::Matmul, vec![input_t, y_bar])
            }
            AdjointExpr::MaxPoolBackward(y_bar, indices) => {
                // MaxPool backward: scatter grad to argmax positions
                self.emit_op(PrimalOp::ScatterAdd { dim: 0 }, vec![y_bar, indices])
            }
            AdjointExpr::AvgPoolBackward(y_bar, pool_size) => {
                // Repeat (upsample) each element to cover the pool region,
                // then scale by 1/pool_size to distribute the gradient evenly.
                let repeated = self.emit_op(PrimalOp::Repeat { kernel: pool_size }, vec![y_bar]);
                let scale = self.emit_constant(1.0 / pool_size as f64);
                self.emit_op(PrimalOp::Mul, vec![repeated, scale])
            }

            // --- CrossEntropy backward: (softmax(logits) - one_hot(targets)) * y_bar / N ---
            // Uses dedicated runtime function that handles class-index targets
            // (not one-hot), computes softmax internally, and fuses the /N scaling.
            AdjointExpr::CrossEntropyBackward(y_bar, logits, targets) => self.emit_op(
                PrimalOp::Passthrough("cross_entropy_backward".into()),
                vec![y_bar, logits, targets],
            ),

            // --- MSE backward for reduction='sum': d/d(pred) = 2*(pred - target) * grad_output ---
            // For reduction='mean', the user's NSL code divides by N before loss,
            // which propagates 1/N through grad_output automatically.
            AdjointExpr::MSEBackward(y_bar, pred, target) => {
                let diff = self.emit_op(PrimalOp::Sub, vec![pred, target]);
                let two = self.emit_constant(2.0);
                let scaled_diff = self.emit_op(PrimalOp::Mul, vec![two, diff]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, scaled_diff])
            }

            // --- L1 backward for reduction='sum': d/d(pred) = sign(pred - target) * grad_output ---
            // For reduction='mean', the user's NSL code divides by N before loss,
            // which propagates 1/N through grad_output automatically.
            AdjointExpr::L1Backward(y_bar, pred, target) => {
                let diff = self.emit_op(PrimalOp::Sub, vec![pred, target]);
                let zero = self.emit_constant(0.0);
                let pos_cond = self.emit_op(PrimalOp::Condition(CompareKind::Gt), vec![diff, zero]);
                let one = self.emit_constant(1.0);
                let neg_one = self.emit_constant(-1.0);
                let sign = self.emit_op(PrimalOp::Select, vec![pos_cond, one, neg_one]);
                self.emit_op(PrimalOp::Mul, vec![y_bar, sign])
            }

            // --- Attention backward: per-component extraction from fused kernel ---
            // Each component (dQ=0, dK=1, dV=2) is extracted via a dedicated op
            // that carries the causal flag so the runtime can apply the correct mask.
            AdjointExpr::AttentionBackwardQ(y_bar, q, k, v, causal) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtract {
                    causal,
                    component: 0,
                },
                vec![y_bar, q, k, v],
            ),
            AdjointExpr::AttentionBackwardK(y_bar, q, k, v, causal) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtract {
                    causal,
                    component: 1,
                },
                vec![y_bar, q, k, v],
            ),
            AdjointExpr::AttentionBackwardV(y_bar, q, k, v, causal) => self.emit_op(
                PrimalOp::FlashAttentionBackwardExtract {
                    causal,
                    component: 2,
                },
                vec![y_bar, q, k, v],
            ),

            // --- RoPE backward: apply inverse rotation (rotate by -θ) ---
            // RoPE(x, θ) = R(θ)·x, so d/dx = R(θ)^T = R(-θ).
            // Negating the input is WRONG (that's reflection, not inverse rotation).
            // Emit as RoPE with negated sin component.
            AdjointExpr::RoPEBackward(y_bar, dim) => {
                self.emit_op(PrimalOp::RoPEInverse { dim }, vec![y_bar])
            }

            // --- rotate_half backward: -rotate_half(grad) ---
            // forward: y[..h] = -x[h..], y[h..] = x[..h]
            // backward: dx[..h] = dy[h..], dx[h..] = -dy[..h]
            // which equals -rotate_half(dy). Lower as a Passthrough rotate_half
            // followed by Neg.
            AdjointExpr::RotateHalfBackward(y_bar) => {
                let rotated = self.emit_op(
                    PrimalOp::Passthrough("rotate_half".into()),
                    vec![y_bar],
                );
                self.emit_op(PrimalOp::Neg, vec![rotated])
            }
        }
    }

    fn accumulate_adjoint(&mut self, var: VarId, value: VarId) {
        if let Some(&existing) = self.adjoint_vars.get(&var) {
            let sum = self.next_var();
            let op_id = self.next_op();
            self.adjoint_ops.push(WengertOp {
                id: op_id,
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

    adjoint_ops
        .iter()
        .enumerate()
        .filter(|(i, _)| live_ops.contains(i))
        .map(|(_, op)| op.clone())
        .collect()
}

// ---------------------------------------------------------------------------
// Saved tensor analysis
// ---------------------------------------------------------------------------

/// Information about a tensor that must be saved from forward for backward.
#[derive(Debug, Clone, PartialEq)]
pub struct SavedTensorInfo {
    pub var: VarId,
    pub checkpointed: bool,
}

/// Identify which forward-pass intermediates must be saved for the backward pass.
pub fn analyze_saved_tensors(primal: &WengertList, adjoint: &WengertList) -> Vec<SavedTensorInfo> {
    let primal_vars: HashSet<VarId> = primal.ops.iter().map(|op| op.result).collect();
    let adjoint_vars: HashSet<VarId> = adjoint.ops.iter().map(|op| op.result).collect();

    let mut saved = Vec::new();
    for adj_op in &adjoint.ops {
        for &input in &adj_op.inputs {
            if primal_vars.contains(&input) && !adjoint_vars.contains(&input) {
                saved.push(SavedTensorInfo {
                    var: input,
                    checkpointed: primal.is_checkpointed(input),
                });
            }
        }
    }

    saved.sort_by_key(|s| s.var);
    saved.dedup_by_key(|s| s.var);
    saved
}

// ---------------------------------------------------------------------------
// M40b: Wengert Extraction from AST
// ---------------------------------------------------------------------------

/// Extracts a WengertList from a sequence of AST statements.
///
/// Returns `Some(WengertList)` if the computation is fully static (no
/// data-dependent control flow). Returns `None` if dynamic control flow
/// is detected, signaling fallback to tape-based AD.
pub struct WengertExtractor<'a> {
    interner: &'a Interner,
    list: WengertList,
    /// Maps AST symbol -> WengertList VarId.
    symbol_to_var: HashMap<nsl_ast::Symbol, VarId>,
    /// Next VarId to allocate.
    next_var: VarId,
    /// Whether this computation graph is fully static.
    is_static: bool,
    /// Symbols that are model parameters (need gradients).
    param_symbols: HashSet<nsl_ast::Symbol>,
    /// Current "self" context prefix for model method inlining (e.g., "m", "m.blocks.0").
    self_context: Option<String>,
    /// Model method bodies: model_type_name -> method_name -> FnDef
    model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>>,
    /// Model field type info: model_type -> field_name -> type_string
    model_field_types: HashMap<String, HashMap<String, String>>,
    /// Context prefix -> model type name mapping (e.g., "m" -> "NSLCoder", "m.blocks.0" -> "TransformerBlock")
    context_to_model_type: HashMap<String, String>,
    /// Override resolved names for symbols (loop var -> "m.blocks.0")
    symbol_name_overrides: HashMap<nsl_ast::Symbol, String>,
    /// Model instance type for symbols (loop var -> "TransformerBlock")
    model_instance_types: HashMap<nsl_ast::Symbol, String>,
    /// Named parameter VarIds with their compound names (e.g., "m.blocks.0.attn.w_q" -> VarId).
    /// Used for gradient collection since compound params don't have unique AST Symbols.
    named_param_vars: Vec<(String, VarId)>,
}

impl<'a> WengertExtractor<'a> {
    pub fn new(interner: &'a Interner) -> Self {
        WengertExtractor {
            interner,
            list: WengertList {
                ops: Vec::new(),
                output: 0,
                var_names: HashMap::new(),
                var_types: HashMap::new(),
            },
            symbol_to_var: HashMap::new(),
            next_var: 0,
            is_static: true,
            param_symbols: HashSet::new(),
            self_context: None,
            model_method_bodies: HashMap::new(),
            model_field_types: HashMap::new(),
            context_to_model_type: HashMap::new(),
            symbol_name_overrides: HashMap::new(),
            model_instance_types: HashMap::new(),
            named_param_vars: Vec::new(),
        }
    }

    fn alloc_var(&mut self) -> VarId {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    /// Push a WengertOp and auto-tag its result VarId with the correct WengertType.
    fn push_op(&mut self, op: WengertOp) {
        let ty = type_for_op(&op.op);
        self.list.var_types.insert(op.result, ty);
        self.list.ops.push(op);
    }

    fn extract_int_literal(expr: &nsl_ast::expr::Expr) -> Option<i64> {
        match &expr.kind {
            ExprKind::IntLiteral(v) => Some(*v),
            ExprKind::UnaryOp {
                op: AstUnaryOp::Neg,
                operand,
            } => match &operand.kind {
                ExprKind::IntLiteral(v) => Some(-*v),
                _ => None,
            },
            _ => None,
        }
    }

    fn encode_transpose_dim(dim: i64) -> Option<usize> {
        match dim {
            -2 => Some(usize::MAX - 1),
            -1 => Some(usize::MAX),
            value if value >= 0 => Some(value as usize),
            _ => None,
        }
    }

    fn extract_transpose_dims(args: &[nsl_ast::expr::Arg]) -> Option<(usize, usize)> {
        let dim0 = args
            .first()
            .and_then(|arg| Self::extract_int_literal(&arg.value))?;
        let dim1 = args
            .get(1)
            .and_then(|arg| Self::extract_int_literal(&arg.value))?;
        Some((
            Self::encode_transpose_dim(dim0)?,
            Self::encode_transpose_dim(dim1)?,
        ))
    }

    /// Register a parameter symbol (needs gradient).
    pub fn register_param(&mut self, sym: nsl_ast::Symbol) {
        let var = self.alloc_var();
        self.symbol_to_var.insert(sym, var);
        self.param_symbols.insert(sym);

        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.list.var_names.insert(var, name.clone());
        self.push_op(WengertOp {
            id: self.list.ops.len() as u32,
            result: var,
            op: PrimalOp::Param(name),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
    }

    /// Register an input symbol (data, no gradient).
    pub fn register_input(&mut self, sym: nsl_ast::Symbol) {
        let var = self.alloc_var();
        self.symbol_to_var.insert(sym, var);

        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.list.var_names.insert(var, name.clone());
        self.push_op(WengertOp {
            id: self.list.ops.len() as u32,
            result: var,
            op: PrimalOp::Input(name),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
    }

    /// Set model method bodies for inline expansion during extraction.
    /// Maps model_type_name -> method_name -> FnDef.
    pub fn set_model_method_bodies(
        &mut self,
        bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>>,
    ) {
        self.model_method_bodies = bodies;
    }

    /// Set model field type info for for-loop unrolling.
    /// Maps model_type -> field_name -> type_string (e.g., "[TransformerBlock;8]").
    pub fn set_model_field_types(&mut self, types: HashMap<String, HashMap<String, String>>) {
        self.model_field_types = types;
    }

    /// Register a model instance: maps a variable symbol to a model type name
    /// and sets up the context-to-type mapping for method inlining.
    pub fn register_model_instance(&mut self, sym: nsl_ast::Symbol, model_type: &str) {
        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.model_instance_types
            .insert(sym, model_type.to_string());
        self.context_to_model_type
            .insert(name, model_type.to_string());
    }

    /// Get named parameter VarIds with their compound names.
    /// Returns (compound_name, VarId) pairs for gradient collection.
    pub fn named_param_var_ids(&self) -> &[(String, VarId)] {
        &self.named_param_vars
    }

    /// Extract statements into the Wengert list.
    /// Returns false if dynamic control flow is detected.
    pub fn extract_stmts(&mut self, stmts: &[nsl_ast::stmt::Stmt]) -> bool {
        for stmt in stmts {
            if !self.extract_stmt(stmt) {
                self.is_static = false;
                eprintln!(
                    "[source-ad] extraction failed at {:?} (line {:?})",
                    std::mem::discriminant(&stmt.kind),
                    stmt.span
                );
                return false;
            }
        }
        true
    }

    fn extract_stmt(&mut self, stmt: &nsl_ast::stmt::Stmt) -> bool {
        match &stmt.kind {
            StmtKind::VarDecl {
                pattern,
                value: Some(val),
                ..
            } => {
                let _var_name = if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    self.interner.resolve(sym.0).unwrap_or("?").to_string()
                } else {
                    "?".to_string()
                };
                if let Some(var) = self.extract_expr(val) {
                    if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                        self.symbol_to_var.insert(*sym, var);
                        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
                        self.list.var_names.insert(var, name);
                    }
                    true
                } else {
                    eprintln!(
                        "[source-ad] VarDecl '{}' extraction failed at expr {:?}",
                        _var_name,
                        std::mem::discriminant(&val.kind)
                    );
                    false
                }
            }

            StmtKind::Return(Some(expr)) => {
                if let Some(var) = self.extract_expr(expr) {
                    self.list.output = var;
                    true
                } else {
                    false
                }
            }

            StmtKind::Expr(expr) => self.extract_expr(expr).is_some(),

            // Dynamic control flow -> not static
            StmtKind::While { .. } => {
                self.is_static = false;
                false
            }

            // For loops: try to unroll if iterating over a model's FixedArray field
            StmtKind::For {
                pattern,
                iterable,
                body,
            } => self.try_unroll_for(pattern, iterable, body),

            // If/else: extract both branches, emit Select ops for variables
            // that differ between branches. The condition is saved for backward.
            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => self.extract_if(condition, then_block, elif_clauses, else_block.as_ref()),

            // Assignment: x = expr (rebind variable to new value)
            StmtKind::Assign { target, op, value } => {
                if let nsl_ast::operator::AssignOp::Assign = op {
                    if let nsl_ast::expr::ExprKind::Ident(sym) = &target.kind {
                        if let Some(var) = self.extract_expr(value) {
                            self.symbol_to_var.insert(*sym, var);
                            return true;
                        }
                        return false;
                    }
                }
                true
            }

            // Other statements pass through (decorated, etc.)
            _ => true,
        }
    }

    /// Extract an expression into a WengertOp, returning its VarId.
    /// Returns None if the expression contains dynamic control flow.
    fn extract_expr(&mut self, expr: &nsl_ast::expr::Expr) -> Option<VarId> {
        match &expr.kind {
            ExprKind::Ident(sym) => self.symbol_to_var.get(sym).copied(),

            // Field access: self.w, m.w, block.attn → treat as a parameter reference
            ExprKind::MemberAccess { object, member } => {
                // Early check: .shape and .ndim are always non-differentiable metadata
                let member_name = self.interner.resolve(member.0).unwrap_or("?");
                if member_name == "shape" || member_name == "ndim" {
                    if let Some(obj_var) = self.extract_expr(object) {
                        let result = self.alloc_var();
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result,
                            op: PrimalOp::Passthrough(member_name.to_string()),
                            inputs: vec![obj_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                        return Some(result);
                    }
                }

                // Resolve the object prefix to a string
                let obj_prefix = match &object.kind {
                    ExprKind::SelfRef => self.self_context.clone(),
                    ExprKind::Ident(obj_sym) => self
                        .symbol_name_overrides
                        .get(obj_sym)
                        .cloned()
                        .or_else(|| {
                            Some(self.interner.resolve(obj_sym.0).unwrap_or("?").to_string())
                        }),
                    // Nested member access: self.attn.w_q -> resolve recursively
                    ExprKind::MemberAccess { .. } => {
                        // Try to extract the inner object as a compound name
                        // by recursively resolving the member access chain
                        self.resolve_member_access_prefix(object)
                    }
                    _ => None,
                };

                // shape/ndim already handled by early check above

                if let Some(prefix) = obj_prefix {
                    let field_name = self.interner.resolve(member.0).unwrap_or("?").to_string();
                    let compound = format!("{}.{}", prefix, field_name);

                    // Check if we already have this compound name registered
                    // (search by name in var_names to handle cross-iteration reuse)
                    for (vid, name) in &self.list.var_names {
                        if name == &compound {
                            return Some(*vid);
                        }
                    }

                    // Determine if this is a model parameter (prefix is a known model context)
                    // or a data access (e.g., batch.input_ids). Only model params get gradients.
                    let is_model_param = self.context_to_model_type.contains_key(&prefix);

                    let var = self.alloc_var();
                    self.symbol_to_var.insert(*member, var);
                    self.list.var_names.insert(var, compound.clone());

                    if is_model_param {
                        self.param_symbols.insert(*member);
                        self.named_param_vars.push((compound.clone(), var));
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: var,
                            op: PrimalOp::Param(compound),
                            inputs: vec![],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                    } else {
                        // Data access (e.g., batch.input_ids) — emit a dict_get op
                        // that depends on the object value, not a disconnected leaf.
                        // This preserves the computation edge: batch -> dict_get -> tensor.
                        let obj_var = self.extract_expr(object)?;
                        self.push_op(WengertOp {
                            id: self.list.ops.len() as u32,
                            result: var,
                            op: PrimalOp::Passthrough(format!("dict_get:{}", field_name)),
                            inputs: vec![obj_var],
                            saved_for_backward: false,
                            checkpointed: false,
                        });
                    }
                    Some(var)
                } else {
                    None // Can't resolve object prefix
                }
            }

            ExprKind::BinaryOp { left, op, right } => {
                let l = self.extract_expr(left)?;
                let r = self.extract_expr(right)?;
                let result = self.alloc_var();
                let primal_op = match op {
                    BinOp::Add => PrimalOp::Add,
                    BinOp::Sub => PrimalOp::Sub,
                    BinOp::Mul => PrimalOp::Mul,
                    BinOp::Div => PrimalOp::Div,
                    // Comparison operators (non-differentiable, used for branch conditions)
                    BinOp::Gt => PrimalOp::Condition(CompareKind::Gt),
                    BinOp::Lt => PrimalOp::Condition(CompareKind::Lt),
                    BinOp::GtEq => PrimalOp::Condition(CompareKind::GtEq),
                    BinOp::LtEq => PrimalOp::Condition(CompareKind::LtEq),
                    BinOp::Eq => PrimalOp::Condition(CompareKind::Eq),
                    BinOp::NotEq => PrimalOp::Condition(CompareKind::NotEq),
                    BinOp::MatMul => PrimalOp::Matmul,
                    _ => return None, // Unsupported op for AD
                };
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: vec![l, r],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::UnaryOp { op, operand } => {
                let input = self.extract_expr(operand)?;
                let result = self.alloc_var();
                let primal_op = match op {
                    AstUnaryOp::Neg => PrimalOp::Neg,
                    _ => return None,
                };
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: vec![input],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::Call { callee, args } => {
                // Check for model method calls: obj.method(args) → inline the method body
                if let ExprKind::MemberAccess { object, member } = &callee.kind {
                    let method_name = self.interner.resolve(member.0).unwrap_or("?").to_string();

                    // Case 1: obj.method(args) where obj is an Ident (e.g., m.forward_train(...))
                    if let ExprKind::Ident(obj_sym) = &object.kind {
                        let obj_type = self.model_instance_types.get(obj_sym).cloned();
                        if let Some(model_type) = obj_type {
                            let fn_def = self
                                .model_method_bodies
                                .get(&model_type)
                                .and_then(|methods| methods.get(&method_name))
                                .cloned();
                            if let Some(fn_def) = fn_def {
                                return self.inline_method_call(*obj_sym, &fn_def, args);
                            } else {
                                eprintln!("[source-ad] method '{}' not found in model type '{}' (available: {:?})",
                                    method_name, model_type,
                                    self.model_method_bodies.get(&model_type)
                                        .map(|m| m.keys().collect::<Vec<_>>())
                                        .unwrap_or_default());
                            }
                        }
                    }

                    // Case 2: self.sub_model.method(args) — nested model field
                    if let ExprKind::MemberAccess {
                        object: inner_obj,
                        member: field_sym,
                    } = &object.kind
                    {
                        let obj_prefix = match &inner_obj.kind {
                            ExprKind::SelfRef => self.self_context.clone(),
                            ExprKind::Ident(sym) => {
                                self.symbol_name_overrides.get(sym).cloned().or_else(|| {
                                    Some(self.interner.resolve(sym.0).unwrap_or("?").to_string())
                                })
                            }
                            _ => None,
                        };
                        if let Some(prefix) = obj_prefix {
                            let sub_field_name = self
                                .interner
                                .resolve(field_sym.0)
                                .unwrap_or("?")
                                .to_string();
                            let compound_prefix = format!("{}.{}", prefix, sub_field_name);

                            // Find the sub-model's type
                            let parent_type = self.context_to_model_type.get(&prefix).cloned();
                            let sub_type = parent_type
                                .as_ref()
                                .and_then(|pt| self.model_field_types.get(pt))
                                .and_then(|fields| fields.get(&sub_field_name))
                                .cloned();

                            if let Some(ref sub_type) = sub_type {
                                // Don't try to inline array types
                                if !sub_type.starts_with('[') {
                                    self.context_to_model_type
                                        .insert(compound_prefix.clone(), sub_type.clone());
                                    let fn_def = self
                                        .model_method_bodies
                                        .get(sub_type)
                                        .and_then(|methods| methods.get(&method_name))
                                        .cloned();
                                    if let Some(fn_def) = fn_def {
                                        // Pre-extract args in CALLER's context before switching to callee's context.
                                        // This is critical: args like `self.attn_norm.forward(x)` use `self` referring
                                        // to the caller's model (TransformerBlock), not the callee (GQA).
                                        let extracted = self.pre_extract_args(&fn_def, args)?;
                                        let saved_self = self.self_context.clone();
                                        self.self_context = Some(compound_prefix);
                                        let result = self.inline_method_body(&fn_def, extracted);
                                        self.self_context = saved_self;
                                        return result;
                                    }
                                }
                            }
                        }
                    }

                    // Case 3: self.method(args) — method call on self
                    if let ExprKind::SelfRef = &object.kind {
                        if let Some(ctx) = self.self_context.clone() {
                            let model_type = self.context_to_model_type.get(&ctx).cloned();
                            if let Some(model_type) = model_type {
                                let fn_def = self
                                    .model_method_bodies
                                    .get(&model_type)
                                    .and_then(|methods| methods.get(&method_name))
                                    .cloned();
                                if let Some(fn_def) = fn_def {
                                    // Pre-extract args in caller's context
                                    let extracted = self.pre_extract_args(&fn_def, args)?;
                                    let saved_self = self.self_context.clone();
                                    self.self_context = Some(ctx);
                                    let result = self.inline_method_body(&fn_def, extracted);
                                    self.self_context = saved_self;
                                    return result;
                                }
                            }
                        }
                    }
                }

                // Tensor method calls: tensor.reshape(arg), tensor.transpose(d0, d1), etc.
                if let ExprKind::MemberAccess { object, member } = &callee.kind {
                    let method_name = self.interner.resolve(member.0).unwrap_or("?").to_string();
                    match method_name.as_str() {
                        "transpose" => {
                            let obj = self.extract_expr(object)?;
                            if let Some((dim0, dim1)) = Self::extract_transpose_dims(args) {
                                let result = self.alloc_var();
                                self.push_op(WengertOp {
                                    id: self.list.ops.len() as u32,
                                    result,
                                    op: PrimalOp::Transpose { dim0, dim1 },
                                    inputs: vec![obj],
                                    saved_for_backward: false,
                                    checkpointed: false,
                                });
                                return Some(result);
                            }

                            let mut inputs = vec![obj];
                            for arg in args {
                                inputs.push(self.extract_expr(&arg.value)?);
                            }
                            let result = self.alloc_var();
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result,
                                op: PrimalOp::Passthrough(method_name),
                                inputs,
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                            return Some(result);
                        }
                        "reshape" | "contiguous" | "item" | "expand" | "squeeze" | "unsqueeze" => {
                            let obj = self.extract_expr(object)?;
                            let mut inputs = vec![obj];
                            for arg in args {
                                inputs.push(self.extract_expr(&arg.value)?);
                            }
                            let result = self.alloc_var();
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result,
                                op: PrimalOp::Passthrough(method_name),
                                inputs,
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                            return Some(result);
                        }
                        "shape" | "ndim" => {
                            // Property access compiled as method — non-diff metadata
                            let obj = self.extract_expr(object)?;
                            let result = self.alloc_var();
                            self.push_op(WengertOp {
                                id: self.list.ops.len() as u32,
                                result,
                                op: PrimalOp::Passthrough(method_name),
                                inputs: vec![obj],
                                saved_for_backward: false,
                                checkpointed: false,
                            });
                            return Some(result);
                        }
                        _ => {} // fall through to model method handling
                    }
                }

                // Regular function call: extract function name
                let func_name = if let ExprKind::Ident(sym) = &callee.kind {
                    self.interner.resolve(sym.0).unwrap_or("").to_string()
                } else if let ExprKind::MemberAccess { object, member } = &callee.kind {
                    let method = self.interner.resolve(member.0).unwrap_or("?");
                    let obj_desc = match &object.kind {
                        ExprKind::Ident(s) => {
                            format!("Ident({})", self.interner.resolve(s.0).unwrap_or("?"))
                        }
                        ExprKind::SelfRef => "self".to_string(),
                        ExprKind::MemberAccess {
                            object: inner,
                            member: m,
                        } => {
                            let m_name = self.interner.resolve(m.0).unwrap_or("?");
                            match &inner.kind {
                                ExprKind::SelfRef => format!("self.{}", m_name),
                                ExprKind::Ident(s) => format!(
                                    "{}.{}",
                                    self.interner.resolve(s.0).unwrap_or("?"),
                                    m_name
                                ),
                                _ => format!("?.{}", m_name),
                            }
                        }
                        _ => format!("{:?}", std::mem::discriminant(&object.kind)),
                    };
                    eprintln!("[source-ad] unresolved method call: {}.{}() — model type not found in method bodies", obj_desc, method);
                    return None;
                } else {
                    eprintln!(
                        "[source-ad] unsupported callee expression: {:?}",
                        std::mem::discriminant(&callee.kind)
                    );
                    return None; // Complex callee -- can't extract
                };

                // Extract arguments
                let mut input_vars = Vec::new();
                for arg in args {
                    if let Some(var) = self.extract_expr(&arg.value) {
                        input_vars.push(var);
                    } else {
                        return None;
                    }
                }

                let result = self.alloc_var();
                let primal_op = match func_name.as_str() {
                    // Elementwise unary
                    "relu" => PrimalOp::Relu,
                    "sigmoid" => PrimalOp::Sigmoid,
                    "tanh" => PrimalOp::Tanh,
                    "gelu" => PrimalOp::Gelu,
                    "silu" | "swish" => PrimalOp::Silu,
                    "exp" => PrimalOp::Exp,
                    "log" => PrimalOp::Log,
                    "sqrt" => PrimalOp::Sqrt,
                    "abs" => PrimalOp::Abs,
                    // Linear algebra
                    "matmul" => PrimalOp::Matmul,
                    // Reductions — extract dim from second arg if present
                    "softmax" => {
                        let dim = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(-1);
                        PrimalOp::Softmax { dim }
                    }
                    "log_softmax" => {
                        let dim = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(-1);
                        PrimalOp::LogSoftmax { dim }
                    }
                    // Normalization
                    "layer_norm" | "layernorm" => PrimalOp::LayerNorm { eps: 1e-5 },
                    "batch_norm" | "batchnorm" => PrimalOp::BatchNorm {
                        eps: 1e-5,
                        training: true,
                    },
                    "rmsnorm" | "rms_norm" => PrimalOp::RMSNorm { eps: 1e-5 },
                    // Loss functions
                    "cross_entropy" | "cross_entropy_loss" => PrimalOp::CrossEntropyLoss,
                    "mse_loss" => PrimalOp::MSELoss,
                    "l1_loss" => PrimalOp::L1Loss,
                    // Reductions
                    "sum" => PrimalOp::Sum { dim: None },
                    "mean" => PrimalOp::Mean { dim: None },
                    // Regularization — extract p from second arg if literal
                    "dropout" => {
                        let p = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::FloatLiteral(v) => Some(*v),
                                _ => None,
                            })
                            .unwrap_or(0.1);
                        PrimalOp::Dropout { p }
                    }
                    // Indexing
                    "embedding" | "embedding_lookup" => PrimalOp::Embedding,
                    "gather" => {
                        // gather(tensor, dim, indices) — extract dim from second arg
                        let dim = args
                            .get(1)
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(0);
                        PrimalOp::Gather { dim }
                    }
                    // Attention
                    "scaled_dot_product_attention" => {
                        PrimalOp::ScaledDotProductAttention { causal: true }
                    }
                    // Transpose (default: swap last two dims)
                    "transpose" => {
                        let dim_args = args.get(1..).unwrap_or(&[]);
                        let (dim0, dim1) = Self::extract_transpose_dims(dim_args).unwrap_or((0, 1));
                        PrimalOp::Transpose { dim0, dim1 }
                    }
                    // Shape ops (non-differentiable passthrough for AD)
                    "reshape" | "contiguous" | "expand" | "squeeze" | "unsqueeze" => {
                        PrimalOp::Passthrough(func_name.clone())
                    }
                    // Trigonometric (for RoPE)
                    "tensor_cos" | "cos" => PrimalOp::Passthrough("cos".into()),
                    "tensor_sin" | "sin" => PrimalOp::Passthrough("sin".into()),
                    "rotate_half" => PrimalOp::Passthrough("rotate_half".into()),
                    // Tensor construction (non-differentiable)
                    "arange" | "zeros" | "ones" | "full" | "randn" | "zeros_like" | "ones_like" => {
                        PrimalOp::Passthrough(func_name.clone())
                    }
                    // Concatenation
                    "tensor_cat" | "cat" => {
                        // tensor_cat(tensors, dim) — extract dim from last arg
                        let dim = args
                            .last()
                            .and_then(|a| match &a.value.kind {
                                ExprKind::IntLiteral(v) => Some(*v),
                                ExprKind::UnaryOp {
                                    op: AstUnaryOp::Neg,
                                    operand,
                                } => match &operand.kind {
                                    ExprKind::IntLiteral(v) => Some(-*v),
                                    _ => None,
                                },
                                _ => None,
                            })
                            .unwrap_or(-1);
                        PrimalOp::Concat { dim }
                    }
                    // Negative
                    "neg" => PrimalOp::Neg,
                    // Clamp
                    "clamp" => PrimalOp::Clamp {
                        min: f64::NEG_INFINITY,
                        max: f64::INFINITY,
                    },
                    // Scalar extraction (non-differentiable)
                    "int" | "float" => PrimalOp::Passthrough(func_name.clone()),
                    _ => {
                        eprintln!("[source-ad] unsupported function: '{}'", func_name);
                        return None;
                    }
                };

                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: primal_op,
                    inputs: input_vars,
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::IntLiteral(v) => {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(*v as f64),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                // Override type: integer literals are raw i64, not tensor pointers.
                // push_op defaults to Tensor (for adjoint seed constants), but
                // IntLiterals are used for subscript indices and shape dimensions.
                self.list.var_types.insert(result, WengertType::Integer);
                Some(result)
            }

            ExprKind::FloatLiteral(v) => {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(*v),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::BoolLiteral(v) => {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(if *v { 1.0 } else { 0.0 }),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::Paren(inner) => self.extract_expr(inner),

            // Subscript: list[index] — non-differentiable metadata access
            ExprKind::Subscript { object, index } => {
                let obj = self.extract_expr(object)?;
                if let nsl_ast::expr::SubscriptKind::Index(idx_expr) = index.as_ref() {
                    let idx = self.extract_expr(idx_expr)?;
                    // Note: idx type is NOT overridden here — the subscript lowerer
                    // handles type conversion (extracts i64 from tensor/scalar).
                    let result = self.alloc_var();
                    self.push_op(WengertOp {
                        id: self.list.ops.len() as u32,
                        result,
                        op: PrimalOp::Passthrough("subscript".into()),
                        inputs: vec![obj, idx],
                        saved_for_backward: false,
                        checkpointed: false,
                    });
                    Some(result)
                } else {
                    None
                }
            }

            // List literal: [a, b, c] — non-differentiable
            ExprKind::ListLiteral(elements) => {
                let mut inputs = Vec::new();
                for elem in elements {
                    inputs.push(self.extract_expr(elem)?);
                }
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Passthrough("list".into()),
                    inputs,
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            // Anything else we can't extract -> fallback
            _ => None,
        }
    }

    /// Extract an if/else statement into the Wengert list.
    ///
    /// Strategy: flatten branches into linear SSA by extracting both branches
    /// independently, then emitting `Select(cond, true_val, false_val)` ops
    /// for any variables that differ between branches.
    ///
    /// For `if cond { return x*x } else { return -x }`, this produces:
    ///   cond_var = Condition(Gt, x, 0)
    ///   true_val = Mul(x, x)        // from then-branch
    ///   false_val = Neg(x)           // from else-branch
    ///   result = Select(cond_var, true_val, false_val)
    ///
    /// # Speculative (Eager) Evaluation of Both Branches
    ///
    /// **Both branches are eagerly evaluated**: all ops from both the then-branch
    /// and the else-branch appear unconditionally in the Wengert list. Only the
    /// `Select` op at the end chooses which result to use based on the condition.
    ///
    /// This design is **intentional** for two reasons:
    /// 1. **SSA flattening**: the Wengert list is a linear DAG with no control flow;
    ///    both paths must be materialized so that reverse-mode AD can propagate
    ///    gradients through the chosen branch.
    /// 2. **AD correctness**: the `SelectTrue`/`SelectFalse` adjoint rules mask
    ///    gradient flow to the non-taken branch (multiplying by 0), which is the
    ///    correct sub-gradient for a piecewise-linear selection.
    ///
    /// **Known limitation — side effects and expensive computations**: Because both
    /// branches execute unconditionally, any side-effectful code (I/O, mutations,
    /// random sampling) inside a branch will run on *every* evaluation regardless of
    /// the condition. Similarly, expensive computations in the unused branch still
    /// pay their full cost. This is a deliberate trade-off for AD correctness; users
    /// with side-effectful branches should use the dynamic tape fallback (`@no_static`
    /// or constructs that prevent static extraction) instead.
    fn extract_if(
        &mut self,
        condition: &nsl_ast::expr::Expr,
        then_block: &nsl_ast::stmt::Block,
        elif_clauses: &[(nsl_ast::expr::Expr, nsl_ast::stmt::Block)],
        else_block: Option<&nsl_ast::stmt::Block>,
    ) -> bool {
        // Extract the condition expression
        let cond_var = match self.extract_expr(condition) {
            Some(v) => v,
            None => return false,
        };

        // Snapshot the current symbol -> var mapping before entering branches
        let snapshot = self.symbol_to_var.clone();
        let output_before = self.list.output;

        // --- Extract then-branch ---
        if !self.extract_stmts(&then_block.stmts) {
            return false;
        }
        let then_symbols = self.symbol_to_var.clone();
        let then_output = self.list.output;

        // Restore snapshot for else-branch extraction
        self.symbol_to_var = snapshot.clone();
        self.list.output = output_before;

        // --- Extract else-branch ---
        // Handle elif as nested if/else: `elif c2: B2 else: B3` becomes
        // `else: if c2: B2 else: B3` — we support at most simple else for now.
        // If there are elif clauses, desugar the first one as a nested if.
        if !elif_clauses.is_empty() {
            // Desugar: elif chain becomes nested if/else.
            // `elif c2: B2 elif c3: B3 else: B4` is treated as
            // `else: if c2: B2 elif c3: B3 else: B4`, recursively.
            let (elif_cond, elif_block) = &elif_clauses[0];
            let remaining_elifs = &elif_clauses[1..];
            // Pass the remaining elif clauses into the recursive call so the
            // entire chain is desugared, not just the first level.
            if !self.extract_if(elif_cond, elif_block, remaining_elifs, else_block) {
                return false;
            }
        } else if let Some(else_blk) = else_block {
            if !self.extract_stmts(&else_blk.stmts) {
                return false;
            }
        }
        // If no else block: variables retain their pre-branch values (identity)

        let else_symbols = self.symbol_to_var.clone();
        let else_output = self.list.output;

        // --- Merge branches with Select ops ---
        // Collect all symbols that exist in either branch
        let all_symbols: HashSet<nsl_ast::Symbol> = then_symbols
            .keys()
            .chain(else_symbols.keys())
            .copied()
            .collect();

        for sym in &all_symbols {
            let then_var = then_symbols.get(sym).copied();
            let else_var = else_symbols.get(sym).copied();
            let before_var = snapshot.get(sym).copied();

            let tv = then_var.or(before_var);
            let ev = else_var.or(before_var);

            if let (Some(t), Some(e)) = (tv, ev) {
                if t != e {
                    // Different values in each branch — emit Select
                    let result = self.alloc_var();
                    self.push_op(WengertOp {
                        id: self.list.ops.len() as u32,
                        result,
                        op: PrimalOp::Select,
                        inputs: vec![cond_var, t, e],
                        saved_for_backward: false,
                        checkpointed: false,
                    });
                    self.symbol_to_var.insert(*sym, result);
                    // Propagate variable name
                    if let Some(name) = self
                        .list
                        .var_names
                        .get(&t)
                        .cloned()
                        .or_else(|| self.list.var_names.get(&e).cloned())
                    {
                        self.list.var_names.insert(result, name);
                    }
                } else {
                    // Same value in both branches — no Select needed
                    self.symbol_to_var.insert(*sym, t);
                }
            }
        }

        // Handle output (return value) merging
        if then_output != output_before || else_output != output_before {
            let t_out = if then_output != output_before {
                then_output
            } else {
                output_before
            };
            let e_out = if else_output != output_before {
                else_output
            } else {
                output_before
            };
            if t_out != e_out {
                let result = self.alloc_var();
                self.push_op(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Select,
                    inputs: vec![cond_var, t_out, e_out],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                self.list.output = result;
            } else {
                self.list.output = t_out;
            }
        }

        true
    }

    /// Try to unroll a for-loop over a model's FixedArray field.
    ///
    /// Handles patterns like `for block in self.blocks: x = block.forward(x, training)`
    /// by resolving the iterable to a model field with a known array type (e.g.,
    /// `[TransformerBlock; 8]`), then expanding the loop body N times with the loop
    /// variable bound to each element's context prefix (e.g., "m.blocks.0", "m.blocks.1", ...).
    fn try_unroll_for(
        &mut self,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> bool {
        // 1. Get the loop variable symbol
        let loop_var_sym = match &pattern.kind {
            nsl_ast::pattern::PatternKind::Ident(sym) => *sym,
            _ => {
                self.is_static = false;
                return false;
            }
        };

        // 2. Resolve the iterable to get the context prefix and field name.
        // The iterable is `self.blocks` (MemberAccess { SelfRef, "blocks" })
        // or `m.blocks` (MemberAccess { Ident(m), "blocks" }).
        let (context_prefix, field_name) = match &iterable.kind {
            ExprKind::MemberAccess { object, member } => {
                let ctx = match &object.kind {
                    ExprKind::SelfRef => self.self_context.clone(),
                    ExprKind::Ident(sym) => {
                        self.symbol_name_overrides.get(sym).cloned().or_else(|| {
                            Some(self.interner.resolve(sym.0).unwrap_or("?").to_string())
                        })
                    }
                    _ => None,
                };
                let fname = self.interner.resolve(member.0).unwrap_or("?").to_string();
                match ctx {
                    Some(c) => (c, fname),
                    None => {
                        self.is_static = false;
                        return false;
                    }
                }
            }
            _ => {
                self.is_static = false;
                return false;
            }
        };

        // 3. Look up the model type for this context, then the field type
        let model_type = self.context_to_model_type.get(&context_prefix).cloned();
        let field_info = model_type
            .as_ref()
            .and_then(|mt| self.model_field_types.get(mt))
            .and_then(|fields| fields.get(&field_name))
            .cloned();

        // Parse field type — look for "[ModelType;N]" pattern
        let (element_type, size) = match field_info {
            Some(ref ft) if ft.starts_with('[') && ft.contains(';') => {
                // Parse "[TransformerBlock;8]"
                let inner = ft.trim_start_matches('[').trim_end_matches(']');
                let parts: Vec<&str> = inner.split(';').collect();
                if parts.len() == 2 {
                    let elem = parts[0].trim().to_string();
                    let n = parts[1].trim().parse::<usize>().unwrap_or(0);
                    if n > 0 {
                        (elem, n)
                    } else {
                        self.is_static = false;
                        return false;
                    }
                } else {
                    self.is_static = false;
                    return false;
                }
            }
            _ => {
                self.is_static = false;
                return false;
            }
        };

        // 4. Unroll: for i in 0..size, expand loop body with adjusted context
        for i in 0..size {
            let iter_prefix = format!("{}.{}.{}", context_prefix, field_name, i);

            // Register this iteration's context -> model type
            self.context_to_model_type
                .insert(iter_prefix.clone(), element_type.clone());

            // Map loop variable to this iteration's context
            self.symbol_name_overrides
                .insert(loop_var_sym, iter_prefix.clone());
            self.model_instance_types
                .insert(loop_var_sym, element_type.clone());

            // Save and set self_context for nested self.field resolution
            let saved_self = self.self_context.clone();

            // Extract loop body
            if !self.extract_stmts(&body.stmts) {
                self.self_context = saved_self;
                self.is_static = false;
                return false;
            }

            self.self_context = saved_self;
        }

        // Clean up overrides
        self.symbol_name_overrides.remove(&loop_var_sym);
        true
    }

    /// Inline a model method call, given the model instance symbol and FnDef.
    /// Sets up self_context from the symbol name (or override), pre-extracts
    /// args in the caller's context, then inlines the method body.
    fn inline_method_call(
        &mut self,
        model_sym: nsl_ast::Symbol,
        fn_def: &nsl_ast::decl::FnDef,
        call_args: &[nsl_ast::expr::Arg],
    ) -> Option<VarId> {
        // Pre-extract args in caller's context before switching to callee's
        let extracted = self.pre_extract_args(fn_def, call_args)?;
        let name = self
            .symbol_name_overrides
            .get(&model_sym)
            .cloned()
            .unwrap_or_else(|| {
                self.interner
                    .resolve(model_sym.0)
                    .unwrap_or("?")
                    .to_string()
            });
        let saved_self = self.self_context.clone();
        self.self_context = Some(name);
        let result = self.inline_method_body(fn_def, extracted);
        self.self_context = saved_self;
        result
    }

    /// Inline a model method call's body into the current Wengert list.
    /// Assumes `self.self_context` is already set to the correct prefix.
    ///
    /// Binds call arguments to the method's parameter names (skipping `self`),
    /// then extracts the method body statements. The return value of the method
    /// becomes the result of this expression.
    /// Pre-extract call arguments in the current (caller's) context.
    /// Returns (param_symbol, param_name, VarId) triples for binding.
    fn pre_extract_args(
        &mut self,
        fn_def: &nsl_ast::decl::FnDef,
        call_args: &[nsl_ast::expr::Arg],
    ) -> Option<Vec<(nsl_ast::Symbol, String, VarId)>> {
        let mut result = Vec::new();
        let mut arg_idx = 0;
        for param in &fn_def.params {
            let param_name = self
                .interner
                .resolve(param.name.0)
                .unwrap_or("?")
                .to_string();
            if param_name == "self" {
                continue;
            }
            if arg_idx < call_args.len() {
                if let Some(var) = self.extract_expr(&call_args[arg_idx].value) {
                    result.push((param.name, param_name, var));
                } else {
                    return None;
                }
                arg_idx += 1;
            }
        }
        Some(result)
    }

    /// Inline a method body with pre-extracted argument bindings.
    fn inline_method_body(
        &mut self,
        fn_def: &nsl_ast::decl::FnDef,
        extracted_args: Vec<(nsl_ast::Symbol, String, VarId)>,
    ) -> Option<VarId> {
        // Inline callee locals in an isolated symbol scope so names like
        // `batch` inside a model method do not overwrite caller bindings such
        // as the train-step batch dict.
        let saved_symbols = self.symbol_to_var.clone();

        // Bind the pre-extracted argument VarIds to parameter symbols
        for (sym, name, var) in extracted_args {
            self.symbol_to_var.insert(sym, var);
            self.list.var_names.insert(var, name);
        }

        // Save the output before extraction
        let saved_output = self.list.output;

        // Extract the method body
        if !self.extract_stmts(&fn_def.body.stmts) {
            self.symbol_to_var = saved_symbols;
            self.list.output = saved_output;
            return None;
        }

        // The method's return value is the list output (set by Return stmt extraction)
        let result = self.list.output;

        // Restore caller scope now that the callee result has been captured.
        self.symbol_to_var = saved_symbols;
        self.list.output = saved_output;

        // If the method didn't set output (no return statement), check if a variable
        // was assigned that should be the result
        if result == saved_output {
            // No return found — method might use implicit return via last expression
            return None;
        }

        Some(result)
    }

    /// Resolve a nested MemberAccess chain to a compound prefix string.
    /// E.g., `self.attn` -> "m.blocks.0.attn", `block.ffn` -> "m.blocks.0.ffn"
    fn resolve_member_access_prefix(&self, expr: &nsl_ast::expr::Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::MemberAccess { object, member } => {
                let obj_prefix = match &object.kind {
                    ExprKind::SelfRef => self.self_context.clone(),
                    ExprKind::Ident(sym) => {
                        self.symbol_name_overrides.get(sym).cloned().or_else(|| {
                            Some(self.interner.resolve(sym.0).unwrap_or("?").to_string())
                        })
                    }
                    ExprKind::MemberAccess { .. } => self.resolve_member_access_prefix(object),
                    _ => None,
                };
                let field_name = self.interner.resolve(member.0).unwrap_or("?").to_string();
                obj_prefix.map(|p| format!("{}.{}", p, field_name))
            }
            _ => None,
        }
    }

    /// Finalize extraction. Returns the WengertList if the graph is static.
    pub fn finalize(self) -> Option<WengertList> {
        if self.is_static && !self.list.ops.is_empty() {
            Some(self.list)
        } else {
            None
        }
    }

    /// Check if the computation graph is static (no dynamic control flow).
    pub fn is_static_graph(&self) -> bool {
        self.is_static
    }

    /// Get the parameter VarIds (for gradient computation targets).
    pub fn param_vars(&self) -> Vec<VarId> {
        self.param_symbols
            .iter()
            .filter_map(|sym| self.symbol_to_var.get(sym).copied())
            .collect()
    }

    /// Get the symbol-to-VarId mapping for resolving primal vars to Cranelift Values.
    pub fn symbol_var_map(&self) -> &HashMap<nsl_ast::Symbol, VarId> {
        &self.symbol_to_var
    }

    /// Get VarIds for model parameters (unordered).
    pub fn param_var_ids(&self) -> Vec<(nsl_ast::Symbol, VarId)> {
        self.param_symbols
            .iter()
            .filter_map(|sym| self.symbol_to_var.get(sym).map(|vid| (*sym, *vid)))
            .collect()
    }

    /// Get the next available VarId (for AdjointGenerator::new start_var).
    pub fn next_var_id(&self) -> VarId {
        self.next_var
    }

    /// Access the extracted WengertList.
    pub fn wengert_list(&self) -> &WengertList {
        &self.list
    }

    /// Set the output VarId of the extracted WengertList.
    /// This is needed when the step body uses `let loss = ...` (VarDecl) rather
    /// than `return loss`, since VarDecl extraction does not set list.output.
    pub fn set_output(&mut self, var: VarId) {
        self.list.output = var;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertList, WengertOp};

    fn make_op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    #[test]
    fn test_adjoint_generator_simple_add() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("a".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("b".into()), vec![]),
                make_op(2, 2, PrimalOp::Add, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);
        assert!(!adjoint.ops.is_empty());
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    #[test]
    fn test_adjoint_generator_matmul() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("A".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("B".into()), vec![]),
                make_op(2, 2, PrimalOp::Matmul, vec![0, 1]),
            ],
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let mut gen = AdjointGenerator::new(10);
        let _adjoint = gen.generate(&primal);
        assert!(gen.adjoint_of(0).is_some());
        assert!(gen.adjoint_of(1).is_some());
    }

    #[test]
    fn test_dead_gradient_elimination() {
        let ops = vec![
            make_op(0, 10, PrimalOp::Mul, vec![3, 4]),
            make_op(1, 11, PrimalOp::Add, vec![5, 6]),
            make_op(2, 12, PrimalOp::Neg, vec![7]),
            make_op(3, 3, PrimalOp::Add, vec![10, 8]),
            make_op(4, 13, PrimalOp::Mul, vec![9, 10]),
        ];
        let needed = HashSet::from([3_u32]);
        let pruned = eliminate_dead_gradients(&ops, &needed);
        assert!(pruned.len() < ops.len());
        assert!(pruned.iter().any(|op| op.result == 3));
    }

    #[test]
    fn test_dead_gradient_empty_needed() {
        let ops = vec![make_op(0, 10, PrimalOp::Add, vec![0, 1])];
        let pruned = eliminate_dead_gradients(&ops, &HashSet::new());
        assert!(pruned.is_empty());
    }

    #[test]
    fn test_saved_tensor_analysis() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Relu, vec![0]),
            ],
            output: 1,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![
                make_op(0, 10, PrimalOp::Constant(1.0), vec![]),
                make_op(1, 11, PrimalOp::Mul, vec![10, 0]),
            ],
            output: 10,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let saved = analyze_saved_tensors(&primal, &adjoint);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].var, 0);
    }

    #[test]
    fn test_saved_tensor_no_cross_reference() {
        let primal = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Input("x".into()), vec![])],
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![make_op(0, 10, PrimalOp::Constant(1.0), vec![])],
            output: 10,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        assert!(analyze_saved_tensors(&primal, &adjoint).is_empty());
    }

    // --- M40b: WengertExtractor tests ---

    use nsl_ast::expr::ExprKind;
    use nsl_ast::stmt::StmtKind;
    use nsl_lexer::Interner;

    fn test_expr(kind: ExprKind) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind,
            span: nsl_errors::Span::dummy(),
            id: nsl_ast::NodeId::dummy(),
        }
    }

    fn test_arg(value: nsl_ast::expr::Expr) -> nsl_ast::expr::Arg {
        nsl_ast::expr::Arg {
            name: None,
            value,
            span: nsl_errors::Span::dummy(),
        }
    }

    #[test]
    fn extract_simple_input() {
        let mut interner = Interner::new();
        // Pre-intern before borrowing into extractor
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let extractor = {
            let mut ext = WengertExtractor::new(&interner);
            ext.register_input(x_sym);
            ext
        };

        assert!(extractor.is_static_graph());
        let list = extractor.finalize();
        assert!(list.is_some());
        assert_eq!(list.unwrap().ops.len(), 1);
    }

    #[test]
    fn extract_registers_params_and_inputs() {
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let w_sym = nsl_ast::Symbol(interner.get_or_intern("W"));

        let extractor = {
            let mut ext = WengertExtractor::new(&interner);
            ext.register_input(x_sym);
            ext.register_param(w_sym);
            ext
        };

        let params = extractor.param_vars();
        assert_eq!(params.len(), 1);
        assert!(extractor.is_static_graph());
    }

    #[test]
    fn extract_method_transpose_as_primal_transpose() {
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let transpose_sym = nsl_ast::Symbol(interner.get_or_intern("transpose"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_input(x_sym);

        let call = test_expr(ExprKind::Call {
            callee: Box::new(test_expr(ExprKind::MemberAccess {
                object: Box::new(test_expr(ExprKind::Ident(x_sym))),
                member: transpose_sym,
            })),
            args: vec![
                test_arg(test_expr(ExprKind::IntLiteral(1))),
                test_arg(test_expr(ExprKind::IntLiteral(2))),
            ],
        });

        let result = extractor.extract_expr(&call);
        assert!(result.is_some());

        let list = extractor
            .finalize()
            .expect("static transpose extraction should succeed");
        let transpose_op = list.ops.last().expect("transpose op should be emitted");
        assert_eq!(transpose_op.op, PrimalOp::Transpose { dim0: 1, dim1: 2 });
        assert_eq!(transpose_op.inputs.len(), 1);
    }

    #[test]
    fn extract_function_transpose_preserves_literal_dims() {
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let transpose_sym = nsl_ast::Symbol(interner.get_or_intern("transpose"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_input(x_sym);

        let call = test_expr(ExprKind::Call {
            callee: Box::new(test_expr(ExprKind::Ident(transpose_sym))),
            args: vec![
                test_arg(test_expr(ExprKind::Ident(x_sym))),
                test_arg(test_expr(ExprKind::IntLiteral(1))),
                test_arg(test_expr(ExprKind::IntLiteral(2))),
            ],
        });

        let result = extractor.extract_expr(&call);
        assert!(result.is_some());

        let list = extractor
            .finalize()
            .expect("static transpose extraction should succeed");
        let transpose_op = list.ops.last().expect("transpose op should be emitted");
        assert_eq!(transpose_op.op, PrimalOp::Transpose { dim0: 1, dim1: 2 });
    }

    #[test]
    fn extract_detects_dynamic_while() {
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let while_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::While {
                condition: nsl_ast::expr::Expr {
                    kind: ExprKind::BoolLiteral(true),
                    span: nsl_errors::Span::dummy(),
                    id: nsl_ast::NodeId(0),
                },
                body: nsl_ast::stmt::Block {
                    stmts: vec![],
                    span: nsl_errors::Span::dummy(),
                },
            },
            span: nsl_errors::Span::dummy(),
            id: nsl_ast::NodeId(1),
        };

        let result = extractor.extract_stmts(&[while_stmt]);
        assert!(!result);
        assert!(!extractor.is_static_graph());
    }

    #[test]
    fn extract_detects_dynamic_for() {
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let for_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::For {
                pattern: nsl_ast::pattern::Pattern {
                    kind: nsl_ast::pattern::PatternKind::Wildcard,
                    span: nsl_errors::Span::dummy(),
                    id: nsl_ast::NodeId(0),
                },
                iterable: nsl_ast::expr::Expr {
                    kind: ExprKind::IntLiteral(10),
                    span: nsl_errors::Span::dummy(),
                    id: nsl_ast::NodeId(0),
                },
                body: nsl_ast::stmt::Block {
                    stmts: vec![],
                    span: nsl_errors::Span::dummy(),
                },
            },
            span: nsl_errors::Span::dummy(),
            id: nsl_ast::NodeId(1),
        };

        let result = extractor.extract_stmts(&[for_stmt]);
        assert!(!result);
    }

    #[test]
    fn extractor_finalize_none_when_empty() {
        let interner = Interner::new();
        let extractor = WengertExtractor::new(&interner);
        assert!(extractor.finalize().is_none());
    }

    // --- Helper: build AST nodes for if/else tests ---

    fn dummy_span() -> nsl_errors::Span {
        nsl_errors::Span::dummy()
    }

    fn make_ident_expr(sym: nsl_ast::Symbol) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::Ident(sym),
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_float_expr(v: f64) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::FloatLiteral(v),
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_binop_expr(
        left: nsl_ast::expr::Expr,
        op: nsl_ast::operator::BinOp,
        right: nsl_ast::expr::Expr,
    ) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::BinaryOp {
                left: Box::new(left),
                op,
                right: Box::new(right),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_neg_expr(operand: nsl_ast::expr::Expr) -> nsl_ast::expr::Expr {
        nsl_ast::expr::Expr {
            kind: ExprKind::UnaryOp {
                op: nsl_ast::operator::UnaryOp::Neg,
                operand: Box::new(operand),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_return_stmt(expr: nsl_ast::expr::Expr) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::Return(Some(expr)),
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_let_stmt(sym: nsl_ast::Symbol, value: nsl_ast::expr::Expr) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::VarDecl {
                is_const: false,
                pattern: nsl_ast::pattern::Pattern {
                    kind: nsl_ast::pattern::PatternKind::Ident(sym),
                    span: dummy_span(),
                    id: nsl_ast::NodeId::next(),
                },
                type_ann: None,
                value: Some(value),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_block(stmts: Vec<nsl_ast::stmt::Stmt>) -> nsl_ast::stmt::Block {
        nsl_ast::stmt::Block {
            stmts,
            span: dummy_span(),
        }
    }

    fn make_if_stmt(
        condition: nsl_ast::expr::Expr,
        then_stmts: Vec<nsl_ast::stmt::Stmt>,
        else_stmts: Option<Vec<nsl_ast::stmt::Stmt>>,
    ) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::If {
                condition,
                then_block: make_block(then_stmts),
                elif_clauses: vec![],
                else_block: else_stmts.map(make_block),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    fn make_if_elif_stmt(
        condition: nsl_ast::expr::Expr,
        then_stmts: Vec<nsl_ast::stmt::Stmt>,
        elif_cond: nsl_ast::expr::Expr,
        elif_stmts: Vec<nsl_ast::stmt::Stmt>,
        else_stmts: Option<Vec<nsl_ast::stmt::Stmt>>,
    ) -> nsl_ast::stmt::Stmt {
        nsl_ast::stmt::Stmt {
            kind: StmtKind::If {
                condition,
                then_block: make_block(then_stmts),
                elif_clauses: vec![(elif_cond, make_block(elif_stmts))],
                else_block: else_stmts.map(make_block),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId::next(),
        }
    }

    // ---------------------------------------------------------------
    // Task 1: Source AD accepts simple if/else (no longer marks dynamic)
    // ---------------------------------------------------------------

    #[test]
    fn extract_accepts_simple_if_else() {
        // f(x) = if x > 0: return x else: return 0.0
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let then_stmts = vec![make_return_stmt(make_ident_expr(x_sym))];
        let else_stmts = vec![make_return_stmt(make_float_expr(0.0))];
        let if_stmt = make_if_stmt(cond, then_stmts, Some(else_stmts));

        let result = extractor.extract_stmts(&[if_stmt]);
        assert!(result, "Source AD should accept simple if/else");
        assert!(
            extractor.is_static_graph(),
            "If/else should not make graph dynamic"
        );

        let list = extractor.finalize();
        assert!(list.is_some(), "Should produce a valid WengertList");
    }

    #[test]
    fn extract_still_rejects_while_loops() {
        // Ensure while loops are still dynamic
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let while_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::While {
                condition: nsl_ast::expr::Expr {
                    kind: ExprKind::BoolLiteral(true),
                    span: dummy_span(),
                    id: nsl_ast::NodeId(0),
                },
                body: make_block(vec![]),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId(1),
        };

        assert!(!extractor.extract_stmts(&[while_stmt]));
        assert!(!extractor.is_static_graph());
    }

    #[test]
    fn extract_still_rejects_for_loops() {
        let interner = Interner::new();
        let mut extractor = WengertExtractor::new(&interner);

        let for_stmt = nsl_ast::stmt::Stmt {
            kind: StmtKind::For {
                pattern: nsl_ast::pattern::Pattern {
                    kind: nsl_ast::pattern::PatternKind::Wildcard,
                    span: dummy_span(),
                    id: nsl_ast::NodeId(0),
                },
                iterable: nsl_ast::expr::Expr {
                    kind: ExprKind::IntLiteral(10),
                    span: dummy_span(),
                    id: nsl_ast::NodeId(0),
                },
                body: make_block(vec![]),
            },
            span: dummy_span(),
            id: nsl_ast::NodeId(1),
        };

        assert!(!extractor.extract_stmts(&[for_stmt]));
    }

    // ---------------------------------------------------------------
    // Task 2: Forward pass saves branch condition + emits Select
    // ---------------------------------------------------------------

    #[test]
    fn extract_if_else_produces_select_op() {
        // f(x) = if x > 0: return x*x else: return -x
        // Should produce: cond = Condition(Gt, x, 0); t = Mul(x,x); e = Neg(x); out = Select(cond, t, e)
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let then_stmts = vec![make_return_stmt(make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        ))];
        let else_stmts = vec![make_return_stmt(make_neg_expr(make_ident_expr(x_sym)))];
        let if_stmt = make_if_stmt(cond, then_stmts, Some(else_stmts));

        let result = extractor.extract_stmts(&[if_stmt]);
        assert!(result, "Should extract if/else successfully");

        let list = extractor.finalize().expect("Should produce WengertList");

        // Verify a Condition op was emitted
        let has_condition = list
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Condition(CompareKind::Gt)));
        assert!(
            has_condition,
            "Should contain a Condition(Gt) op for `x > 0`"
        );

        // Verify a Select op was emitted
        let select_ops: Vec<_> = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .collect();
        assert!(
            !select_ops.is_empty(),
            "Should contain at least one Select op"
        );

        // The Select op should have 3 inputs: [cond, true_val, false_val]
        for sel in &select_ops {
            assert_eq!(
                sel.inputs.len(),
                3,
                "Select should have 3 inputs (cond, true, false)"
            );
        }

        // The output should be the Select result
        let last_select = select_ops.last().unwrap();
        assert_eq!(
            list.output, last_select.result,
            "Output should be the Select result"
        );
    }

    #[test]
    fn extract_if_else_condition_saved_as_var() {
        // Verify the condition comparison result is captured in the Wengert list
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let then_stmts = vec![make_return_stmt(make_ident_expr(x_sym))];
        let else_stmts = vec![make_return_stmt(make_float_expr(0.0))];
        let if_stmt = make_if_stmt(cond, then_stmts, Some(else_stmts));

        extractor.extract_stmts(&[if_stmt]);
        let list = extractor.finalize().unwrap();

        // Find the Condition op
        let cond_op = list
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .unwrap();
        let cond_var = cond_op.result;

        // Find the Select op that uses this condition
        let select_op = list
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Select) && op.inputs[0] == cond_var);
        assert!(
            select_op.is_some(),
            "Select should reference the saved condition variable"
        );
    }

    // ---------------------------------------------------------------
    // Task 3: Backward pass produces conditional adjoint selection
    // ---------------------------------------------------------------

    #[test]
    fn adjoint_of_select_produces_conditional_gradient() {
        // Primal: x (param), 0.0 (const), cond = x > 0, t = x, e = 0.0, out = Select(cond, t, e)
        // This is relu: f(x) = max(x, 0)
        // Backward: d(out)/dx = cond ? 1 : 0
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Select, vec![2, 0, 1]),
            ],
            output: 3,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);

        // The output (var 3) should have an adjoint
        assert!(gen.adjoint_of(3).is_some(), "Output should have adjoint");

        // The param x (var 0) should have an adjoint (from Select true-branch)
        assert!(
            gen.adjoint_of(0).is_some(),
            "Param x should have adjoint through Select"
        );

        // Verify Select ops appear in adjoint list (for conditional gradient selection)
        let has_select_in_adjoint = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Select));
        assert!(
            has_select_in_adjoint,
            "Backward pass should contain Select ops for conditional adjoint selection"
        );
    }

    #[test]
    fn adjoint_select_has_zero_for_inactive_branch() {
        // For Select(cond, a, b), backward should create:
        //   adj_a = Select(cond, adj_out, 0)
        //   adj_b = Select(cond, 0, adj_out)
        // Verify zero constants are generated
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Select, vec![2, 0, 1]),
            ],
            output: 3,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);

        // Should have Constant(0.0) ops in the adjoint for the zero branches
        let zero_consts: Vec<_> = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Constant(v) if *v == 0.0))
            .collect();
        assert!(
            !zero_consts.is_empty(),
            "Backward pass should contain Constant(0.0) for inactive branch adjoints"
        );
    }

    #[test]
    fn adjoint_quadratic_linear_branch() {
        // f(x) = if x > 0 { x * x } else { -x }
        // Primal ops: x(0), zero(1), cond(2)=x>0, t(3)=x*x, e(4)=-x, out(5)=Select(cond, t, e)
        // d/dx for true branch (x*x): 2x
        // d/dx for false branch (-x): -1
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Mul, vec![0, 0]), // x * x (then-branch)
                make_op(4, 4, PrimalOp::Neg, vec![0]),    // -x (else-branch)
                make_op(5, 5, PrimalOp::Select, vec![2, 3, 4]),
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(10);
        let adjoint = gen.generate(&primal);

        // x should have adjoint (from both Mul and Neg, gated by Select)
        assert!(gen.adjoint_of(0).is_some(), "x should have an adjoint");
        // The true result and false result should have adjoints
        assert!(
            gen.adjoint_of(3).is_some(),
            "true-branch result (x*x) should have adjoint"
        );
        assert!(
            gen.adjoint_of(4).is_some(),
            "false-branch result (-x) should have adjoint"
        );

        // Both branch results' adjoints should be Select ops (conditional on saved cond)
        let select_count = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(select_count >= 2,
            "Should have at least 2 Select ops in backward (one per branch input), got {select_count}");
    }

    // ---------------------------------------------------------------
    // Task 4: Nested if/else
    // ---------------------------------------------------------------

    #[test]
    fn extract_nested_if_else() {
        // f(x) = if x > 0 { if x > 1 { x*x*x } else { x*x } } else { 0 }
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        // Inner if: if x > 1 { x*x*x } else { x*x }
        let inner_cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(1.0),
        );
        let x_cubed = make_binop_expr(
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Mul,
                make_ident_expr(x_sym),
            ),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let x_squared = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );

        let inner_if = make_if_stmt(
            inner_cond,
            vec![make_return_stmt(x_cubed)],
            Some(vec![make_return_stmt(x_squared)]),
        );

        // Outer if: if x > 0 { <inner_if> } else { 0 }
        let outer_cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let outer_if = make_if_stmt(
            outer_cond,
            vec![inner_if],
            Some(vec![make_return_stmt(make_float_expr(0.0))]),
        );

        let result = extractor.extract_stmts(&[outer_if]);
        assert!(result, "Nested if/else should be extractable");
        assert!(extractor.is_static_graph());

        let list = extractor.finalize().expect("Should produce WengertList");

        // Should have 2 Condition ops (one per if)
        let cond_count = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .count();
        assert_eq!(
            cond_count, 2,
            "Should have 2 Condition ops for nested if/else, got {cond_count}"
        );

        // Should have at least 2 Select ops (one per merge point)
        let select_count = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            select_count >= 2,
            "Should have at least 2 Select ops, got {select_count}"
        );
    }

    #[test]
    fn adjoint_nested_branches_propagates() {
        // Nested: if x > 0 { if x > 1 { x*x } else { x } } else { 0 }
        // We build the flat Wengert for this manually:
        //   x(0), zero(1), one(2)
        //   cond_outer(3) = x > 0
        //   cond_inner(4) = x > 1
        //   t_inner(5) = x * x
        //   inner_select(6) = Select(cond_inner, t_inner, x)  -- inner merge
        //   outer_select(7) = Select(cond_outer, inner_select, zero) -- outer merge
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Constant(1.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                make_op(5, 5, PrimalOp::Mul, vec![0, 0]),
                make_op(6, 6, PrimalOp::Select, vec![4, 5, 0]),
                make_op(7, 7, PrimalOp::Select, vec![3, 6, 1]),
            ],
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // x (var 0) should have an adjoint — gradient propagates through both Select ops
        assert!(
            gen.adjoint_of(0).is_some(),
            "x should have adjoint through nested Select"
        );

        // The adjoint list should contain multiple Select ops (from both nesting levels)
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Adjoint should have >= 2 Select ops for nested branching, got {adj_selects}"
        );
    }

    // ---------------------------------------------------------------
    // Task 5: If-without-else (identity for missing branch)
    // ---------------------------------------------------------------

    #[test]
    fn extract_if_without_else() {
        // f(x):
        //   let y = x
        //   if x > 5: y = x * 2
        //   return y
        // For x > 5: f(x) = 2x, f'(x) = 2
        // For x <= 5: f(x) = x, f'(x) = 1
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let y_sym = nsl_ast::Symbol(interner.get_or_intern("y"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        // let y = x
        let let_y = make_let_stmt(y_sym, make_ident_expr(x_sym));

        // if x > 5: let y = x * 2.0
        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(5.0),
        );
        let x_times_2 = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_float_expr(2.0),
        );
        let reassign_y = make_let_stmt(y_sym, x_times_2);
        let if_stmt = make_if_stmt(cond, vec![reassign_y], None); // No else!

        // return y
        let return_y = make_return_stmt(make_ident_expr(y_sym));

        let result = extractor.extract_stmts(&[let_y, if_stmt, return_y]);
        assert!(result, "If-without-else should be extractable");
        assert!(extractor.is_static_graph());

        let list = extractor.finalize().expect("Should produce WengertList");

        // Should have a Select op that chooses between modified y (x*2) and original y (x)
        let select_ops: Vec<_> = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .collect();
        assert!(
            !select_ops.is_empty(),
            "If-without-else should produce Select op (modified vs identity)"
        );

        // The Select should use the pre-branch value as the false (else) input
        // This is the identity case — when condition is false, y retains its original value
        let sel = select_ops.last().unwrap();
        assert_eq!(sel.inputs.len(), 3);
    }

    #[test]
    fn adjoint_if_without_else_identity_passthrough() {
        // Manual Wengert for: y = x; if x > 5: y = x * 2; return y
        //   x(0), five(1), two(2)
        //   cond(3) = x > 5
        //   modified_y(4) = x * 2
        //   y_merged(5) = Select(cond, modified_y, x)  // else branch is identity (x)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(5.0), vec![]),
                make_op(2, 2, PrimalOp::Constant(2.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(4, 4, PrimalOp::Mul, vec![0, 2]), // x * 2
                make_op(5, 5, PrimalOp::Select, vec![3, 4, 0]), // cond ? x*2 : x
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // x should have an adjoint
        assert!(gen.adjoint_of(0).is_some(), "x should have adjoint");

        // The backward pass should produce Select ops for conditional adjoint
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 1,
            "Backward should use Select for conditional adjoint, got {adj_selects}"
        );
    }

    #[test]
    fn extract_elif_desugars_to_nested() {
        // f(x) = if x > 1: x*x*x  elif x > 0: x*x  else: 0
        // This tests elif desugaring (single elif clause + else)
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let x_cubed = make_binop_expr(
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Mul,
                make_ident_expr(x_sym),
            ),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let x_squared = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );

        let if_stmt = make_if_elif_stmt(
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Gt,
                make_float_expr(1.0),
            ),
            vec![make_return_stmt(x_cubed)],
            make_binop_expr(
                make_ident_expr(x_sym),
                nsl_ast::operator::BinOp::Gt,
                make_float_expr(0.0),
            ),
            vec![make_return_stmt(x_squared)],
            Some(vec![make_return_stmt(make_float_expr(0.0))]),
        );

        let result = extractor.extract_stmts(&[if_stmt]);
        assert!(result, "elif should be desugared and extractable");
        assert!(extractor.is_static_graph());

        let list = extractor.finalize().expect("Should produce WengertList");

        // Should have 2 Condition ops (outer and elif)
        let cond_count = list
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .count();
        assert_eq!(
            cond_count, 2,
            "Should have 2 Condition ops for if/elif/else, got {cond_count}"
        );
    }

    // ---------------------------------------------------------------
    // Integration: end-to-end extraction + adjoint generation
    // ---------------------------------------------------------------

    #[test]
    fn end_to_end_relu_extraction_and_adjoint() {
        // Build AST for: if x > 0 { return x } else { return 0 }
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let if_stmt = make_if_stmt(
            cond,
            vec![make_return_stmt(make_ident_expr(x_sym))],
            Some(vec![make_return_stmt(make_float_expr(0.0))]),
        );

        assert!(extractor.extract_stmts(&[if_stmt]));
        let primal = extractor.finalize().unwrap();

        // Generate adjoint
        let max_primal_var = primal.ops.iter().map(|op| op.result).max().unwrap_or(0);
        let mut gen = AdjointGenerator::new(max_primal_var + 1);
        let adjoint = gen.generate(&primal);

        // Verify the full pipeline works
        assert!(!adjoint.ops.is_empty(), "Adjoint list should not be empty");
        // x should have an adjoint
        let x_var = primal
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Param(ref n) if n == "x"))
            .map(|op| op.result)
            .unwrap();
        assert!(
            gen.adjoint_of(x_var).is_some(),
            "Param x should have adjoint in relu"
        );
    }

    #[test]
    fn end_to_end_quadratic_branch_extraction_and_adjoint() {
        // Build AST for: if x > 0 { return x*x } else { return -x }
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);

        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        let x_squared = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let neg_x = make_neg_expr(make_ident_expr(x_sym));
        let if_stmt = make_if_stmt(
            cond,
            vec![make_return_stmt(x_squared)],
            Some(vec![make_return_stmt(neg_x)]),
        );

        assert!(extractor.extract_stmts(&[if_stmt]));
        let primal = extractor.finalize().unwrap();

        let max_var = primal.ops.iter().map(|op| op.result).max().unwrap_or(0);
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        assert!(!adjoint.ops.is_empty());
        // Both branches contribute to the adjoint of x
        let x_var = primal
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Param(ref n) if n == "x"))
            .map(|op| op.result)
            .unwrap();
        assert!(gen.adjoint_of(x_var).is_some());
    }

    // ---------------------------------------------------------------
    // Task 6: E2E-style tests for branching gradients
    //
    // These tests exercise the full pipeline (extractor → primal list →
    // adjoint generator) for the three canonical patterns from the plan.
    // Full NSL-file E2E tests are not practical without a live runtime,
    // so we use the AST-level API which is exactly what the compiler uses.
    // ---------------------------------------------------------------

    // ---------------------------------------------------------------
    // Task 6.1: Leaky ReLU
    //   f(x) = if x > 0 { x } else { 0.01 * x }
    //   f'(x) = 1.0  when x > 0
    //   f'(x) = 0.01 when x <= 0
    //
    // Primal Wengert:
    //   x(0), zero(1), alpha(2)=0.01
    //   cond(3) = x > 0          Condition(Gt, x, zero)
    //   scaled(4) = alpha * x    Mul(alpha, x)
    //   out(5) = Select(cond, x, scaled)
    //
    // Backward:
    //   adj(x) should come from Select(cond, 1·adj_out, 0.01·adj_out)
    // ---------------------------------------------------------------
    #[test]
    fn leaky_relu_gradient_structure() {
        // Build using the AST extractor to stay close to the real compiler path.
        let mut interner = Interner::new();
        let x_sym = nsl_ast::Symbol(interner.get_or_intern("x"));
        let alpha_sym = nsl_ast::Symbol(interner.get_or_intern("alpha"));

        let mut extractor = WengertExtractor::new(&interner);
        extractor.register_param(x_sym);
        extractor.register_input(alpha_sym);

        // cond: x > 0
        let cond = make_binop_expr(
            make_ident_expr(x_sym),
            nsl_ast::operator::BinOp::Gt,
            make_float_expr(0.0),
        );
        // then branch: return x
        let then_branch = vec![make_return_stmt(make_ident_expr(x_sym))];
        // else branch: return alpha * x
        let alpha_times_x = make_binop_expr(
            make_ident_expr(alpha_sym),
            nsl_ast::operator::BinOp::Mul,
            make_ident_expr(x_sym),
        );
        let else_branch = vec![make_return_stmt(alpha_times_x)];

        let if_stmt = make_if_stmt(cond, then_branch, Some(else_branch));

        let ok = extractor.extract_stmts(&[if_stmt]);
        assert!(ok, "Leaky ReLU if/else should be extractable by source AD");
        assert!(extractor.is_static_graph(), "Should remain static graph");

        let primal = extractor.finalize().expect("Should produce WengertList");

        // Structural checks on the primal
        let cond_count = primal
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Condition(_)))
            .count();
        assert_eq!(cond_count, 1, "Leaky ReLU needs exactly 1 Condition op");

        let select_count = primal
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert_eq!(select_count, 1, "Leaky ReLU needs exactly 1 Select op");

        // Generate the adjoint
        let max_var = primal.ops.iter().map(|op| op.result).max().unwrap_or(0);
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        // x (the param) must have an adjoint
        let x_var = primal
            .ops
            .iter()
            .find(|op| matches!(&op.op, PrimalOp::Param(ref n) if n == "x"))
            .map(|op| op.result)
            .expect("x must be in primal");
        assert!(
            gen.adjoint_of(x_var).is_some(),
            "Param x should have an adjoint in leaky ReLU"
        );

        // The adjoint must use Select to conditionally route gradient
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 1,
            "Leaky ReLU backward should use at least 1 Select op for conditional gradient, \
             got {adj_selects}"
        );

        // There must be no tape-style Input ops in the adjoint (source AD is tape-free)
        let tape_ops = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Input(_)))
            .count();
        assert_eq!(
            tape_ops, 0,
            "Source AD adjoint must not contain Input (tape-push) ops, got {tape_ops}"
        );
    }

    // ---------------------------------------------------------------
    // Task 6.2: Nested clamp gradient
    //   f(x, lo, hi) = if x < lo { lo } else { if x > hi { hi } else { x } }
    //   f'(x) = 0  when x < lo  (gradient killed — lo is constant)
    //   f'(x) = 0  when x > hi  (gradient killed — hi is constant)
    //   f'(x) = 1  when lo <= x <= hi (pass-through)
    //
    // We build the flat Wengert directly (as the adjoint generator sees it):
    //   x(0), lo(1), hi(2)
    //   cond_low(3)  = x < lo       Condition(Lt)
    //   cond_high(4) = x > hi       Condition(Gt)
    //   inner(5) = Select(cond_high, hi, x)   -- inner if: x>hi ? hi : x
    //   out(6)   = Select(cond_low, lo, inner) -- outer if: x<lo ? lo : inner
    //
    // In the true branch of cond_low the result is lo (Input, not Param of x),
    // so x gets gradient 0 from that branch.  In the false branch x's gradient
    // flows through inner_select, again gated by cond_high.
    // ---------------------------------------------------------------
    #[test]
    fn nested_clamp_gradient_structure() {
        // x is the param we differentiate w.r.t.; lo and hi are inputs (constants).
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("lo".into()), vec![]),
                make_op(2, 2, PrimalOp::Input("hi".into()), vec![]),
                // cond_low = x < lo
                make_op(3, 3, PrimalOp::Condition(CompareKind::Lt), vec![0, 1]),
                // cond_high = x > hi
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                // inner = cond_high ? hi : x
                make_op(5, 5, PrimalOp::Select, vec![4, 2, 0]),
                // out = cond_low ? lo : inner
                make_op(6, 6, PrimalOp::Select, vec![3, 1, 5]),
            ],
            output: 6,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let max_var = 6;
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        // x must have an adjoint
        assert!(
            gen.adjoint_of(0).is_some(),
            "x should have an adjoint through the nested clamp Select chain"
        );

        // The adjoint should use multiple Select ops to gate the gradient
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Nested clamp backward needs >= 2 Select ops (one per nesting level), \
             got {adj_selects}"
        );

        // lo (var 1) and hi (var 2) are Inputs — no gradient expected
        // (they are not Params so AdjointGenerator will not produce a path to them
        //  that matters for x's gradient, but the adjoint list should still be sound)
        assert!(!adjoint.ops.is_empty(), "Adjoint list must not be empty");
    }

    // ---------------------------------------------------------------
    // Task 6.3: Multiple differentiable variables in branches
    //   f(cond_flag, a, b) = if cond_flag > 0.5 { a * b } else { a + b }
    //
    // Gradient w.r.t. a:
    //   true  branch: d(a*b)/da = b  → adjoint contributes b * adj_out
    //   false branch: d(a+b)/da = 1  → adjoint contributes 1 * adj_out
    //   Combined via Select: adj_a = Select(cond, b·adj_out, adj_out)
    //
    // Gradient w.r.t. b:
    //   true  branch: d(a*b)/db = a  → a * adj_out
    //   false branch: d(a+b)/db = 1  → adj_out
    //   Combined via Select: adj_b = Select(cond, a·adj_out, adj_out)
    //
    // Primal Wengert (flat):
    //   cond_raw(0), a(1), b(2)  — all Params
    //   half(3) = Constant(0.5)
    //   cond(4)  = cond_raw > 0.5
    //   t_mul(5) = a * b
    //   f_add(6) = a + b
    //   out(7)   = Select(cond, t_mul, f_add)
    // ---------------------------------------------------------------
    #[test]
    fn multi_variable_branch_adjoint_structure() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("cond_flag".into()), vec![]),
                make_op(1, 1, PrimalOp::Param("a".into()), vec![]),
                make_op(2, 2, PrimalOp::Param("b".into()), vec![]),
                make_op(3, 3, PrimalOp::Constant(0.5), vec![]),
                // cond = cond_flag > 0.5
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 3]),
                // true branch: a * b
                make_op(5, 5, PrimalOp::Mul, vec![1, 2]),
                // false branch: a + b
                make_op(6, 6, PrimalOp::Add, vec![1, 2]),
                // output: Select(cond, t_mul, f_add)
                make_op(7, 7, PrimalOp::Select, vec![4, 5, 6]),
            ],
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let max_var = 7;
        let mut gen = AdjointGenerator::new(max_var + 1);
        let adjoint = gen.generate(&primal);

        // Both a and b must have adjoints
        assert!(
            gen.adjoint_of(1).is_some(),
            "Param a should have an adjoint"
        );
        assert!(
            gen.adjoint_of(2).is_some(),
            "Param b should have an adjoint"
        );

        // The true-branch result and false-branch result should have adjoints
        assert!(
            gen.adjoint_of(5).is_some(),
            "True-branch Mul result should have adjoint"
        );
        assert!(
            gen.adjoint_of(6).is_some(),
            "False-branch Add result should have adjoint"
        );

        // The backward pass must contain Select ops (one per branch input of the primal Select)
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Multi-variable branch backward needs >= 2 Select ops, got {adj_selects}"
        );

        // No tape-style markers in source AD adjoint
        let tape_inputs = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Input(_)))
            .count();
        assert_eq!(
            tape_inputs, 0,
            "Source AD adjoint must be tape-free (no Input ops), got {tape_inputs}"
        );
    }

    // ---------------------------------------------------------------
    // Task 7: Source AD produces no tape push/pop for branching code.
    //
    // Tape-based AD would introduce Input("__tape_push__") / Param markers
    // to record intermediate values at runtime.  Source AD encodes all
    // branching information structurally via Condition + Select ops, so
    // the adjoint Wengert list must be entirely free of tape markers.
    //
    // We verify this for three representative branching functions.
    // ---------------------------------------------------------------
    #[test]
    fn source_ad_no_tape_ops_for_simple_branch() {
        // f(x) = if x > 0 { x * x } else { -x }
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(3, 3, PrimalOp::Mul, vec![0, 0]),
                make_op(4, 4, PrimalOp::Neg, vec![0]),
                make_op(5, 5, PrimalOp::Select, vec![2, 3, 4]),
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // Structural: must not contain Input ops (tape-push markers)
        let has_tape_input = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Input(_)));
        assert!(
            !has_tape_input,
            "Source AD adjoint for branching code must not contain tape Input ops"
        );

        // Must not contain Param ops (tape-checkpoint markers) beyond what primal already has
        // (AdjointGenerator never introduces Param ops; it only uses Constant/Select/arithmetic)
        let has_new_param = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Param(_)));
        assert!(
            !has_new_param,
            "Source AD adjoint must not introduce new Param ops"
        );

        // Must contain Select ops to encode the branch gradient
        let has_select = adjoint
            .ops
            .iter()
            .any(|op| matches!(&op.op, PrimalOp::Select));
        assert!(
            has_select,
            "Source AD adjoint must use Select (not tape push/pop) for branch gradients"
        );

        // x must have an adjoint — the whole point of AD
        assert!(gen.adjoint_of(0).is_some(), "x must have adjoint");
    }

    #[test]
    fn source_ad_no_tape_ops_for_nested_branch() {
        // Nested: if x > 0 { if x > 1 { x*x } else { x } } else { 0 }
        // Flat Wengert (same as adjoint_nested_branches_propagates)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.0), vec![]),
                make_op(2, 2, PrimalOp::Constant(1.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 1]),
                make_op(4, 4, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                make_op(5, 5, PrimalOp::Mul, vec![0, 0]),
                make_op(6, 6, PrimalOp::Select, vec![4, 5, 0]),
                make_op(7, 7, PrimalOp::Select, vec![3, 6, 1]),
            ],
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // No tape-style ops in the backward pass
        assert!(
            !adjoint
                .ops
                .iter()
                .any(|op| matches!(&op.op, PrimalOp::Input(_))),
            "Nested-branch adjoint must be tape-free"
        );
        assert!(
            !adjoint
                .ops
                .iter()
                .any(|op| matches!(&op.op, PrimalOp::Param(_))),
            "Nested-branch adjoint must not introduce Param ops"
        );

        // Must be richer in Select ops than a straight-line function would be
        let adj_selects = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Select))
            .count();
        assert!(
            adj_selects >= 2,
            "Nested-branch adjoint should have >= 2 Select ops (source AD encodes branches \
             structurally, not via tape), got {adj_selects}"
        );
    }

    #[test]
    fn source_ad_adjoint_op_count_dominated_by_arithmetic_not_tape() {
        // For a branching function the adjoint ops should be entirely arithmetic /
        // Select / Constant — zero tape-style bookkeeping.
        //
        // Tape AD would add O(branch_depth) Input ops for each checkpoint save.
        // Source AD adds exactly 0.
        //
        // f(x) = if x > 0 { x } else { 0.01 * x }  (leaky ReLU)
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Param("x".into()), vec![]),
                make_op(1, 1, PrimalOp::Constant(0.01), vec![]),
                make_op(2, 2, PrimalOp::Constant(0.0), vec![]),
                make_op(3, 3, PrimalOp::Condition(CompareKind::Gt), vec![0, 2]),
                make_op(4, 4, PrimalOp::Mul, vec![1, 0]), // 0.01 * x
                make_op(5, 5, PrimalOp::Select, vec![3, 0, 4]), // cond ? x : 0.01*x
            ],
            output: 5,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };

        let mut gen = AdjointGenerator::new(20);
        let adjoint = gen.generate(&primal);

        // Count op types
        let tape_ops = adjoint
            .ops
            .iter()
            .filter(|op| matches!(&op.op, PrimalOp::Input(_) | PrimalOp::Param(_)))
            .count();
        let structural_ops = adjoint
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.op,
                    PrimalOp::Select | PrimalOp::Condition(_) | PrimalOp::Constant(_)
                )
            })
            .count();
        let arithmetic_ops = adjoint
            .ops
            .iter()
            .filter(|op| {
                matches!(
                    &op.op,
                    PrimalOp::Add | PrimalOp::Mul | PrimalOp::Neg | PrimalOp::Sub
                )
            })
            .count();

        assert_eq!(
            tape_ops, 0,
            "Source AD for leaky ReLU must have 0 tape ops, got {tape_ops}"
        );
        assert!(
            structural_ops + arithmetic_ops > 0,
            "Adjoint must have actual gradient computation ops"
        );

        // x must have an adjoint
        assert!(gen.adjoint_of(0).is_some(), "x must have an adjoint");
    }
}
