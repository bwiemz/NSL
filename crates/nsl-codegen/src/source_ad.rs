//! M40: Source-to-source AD — adjoint generation, dead gradient elimination,
//! and saved tensor analysis.

use std::collections::{HashMap, HashSet};
use crate::ad_rules::{apply_ad_rule, AdjointExpr, InputAdjoint};
use crate::wengert::{CompareKind, OpId, PrimalOp, VarId, WengertList, WengertOp};

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
            id: seed_op_id, result: loss_bar, op: PrimalOp::Constant(1.0),
            inputs: vec![], saved_for_backward: false, checkpointed: false,
        });

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
        // Identity: pass-through, no op needed
        if let AdjointExpr::Identity(v) = expr {
            return v;
        }

        let result = self.next_var();
        let op_id = self.next_op();
        let (op, inputs) = match expr {
            AdjointExpr::Identity(_) => unreachable!(),
            AdjointExpr::Negate(v) => (PrimalOp::Neg, vec![v]),
            AdjointExpr::MulElementwise(a, b) => (PrimalOp::Mul, vec![a, b]),
            AdjointExpr::MatmulTransposeLeft(a, b) => (PrimalOp::Matmul, vec![a, b]),
            AdjointExpr::MatmulTransposeRight(a, b) => (PrimalOp::Matmul, vec![a, b]),
            AdjointExpr::Scale(v, _s) => (PrimalOp::Mul, vec![v]),
            AdjointExpr::Broadcast(v) => (PrimalOp::Broadcast, vec![v]),
            AdjointExpr::ScaleBroadcast(v, _n) => (PrimalOp::Broadcast, vec![v]),
            AdjointExpr::Transpose(v, d0, d1) => (PrimalOp::Transpose { dim0: d0, dim1: d1 }, vec![v]),
            AdjointExpr::Reshape(v) => (PrimalOp::Reshape { target_ndim: 0 }, vec![v]),
            AdjointExpr::ExpBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::ReluBackward(y_bar, x) => (PrimalOp::Mul, vec![y_bar, x]),
            AdjointExpr::SigmoidBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::TanhBackward(y_bar, y) => (PrimalOp::Mul, vec![y_bar, y]),
            AdjointExpr::LogBackward(y_bar, x) => (PrimalOp::Div, vec![y_bar, x]),
            AdjointExpr::SqrtBackward(y_bar, y) => (PrimalOp::Div, vec![y_bar, y]),
            AdjointExpr::DivNumeratorBackward(y_bar, b) => (PrimalOp::Div, vec![y_bar, b]),
            AdjointExpr::DivDenominatorBackward(y_bar, a, b) => (PrimalOp::Mul, vec![y_bar, a, b]),
            // SelectTrue: cond ? adj_out : 0
            AdjointExpr::SelectTrue(y_bar, cond) => {
                // Emit a zero constant first
                let zero = self.next_var();
                let zero_op = self.next_op();
                self.adjoint_ops.push(WengertOp {
                    id: zero_op, result: zero, op: PrimalOp::Constant(0.0),
                    inputs: vec![], saved_for_backward: false, checkpointed: false,
                });
                (PrimalOp::Select, vec![cond, y_bar, zero])
            }
            // SelectFalse: !cond ? adj_out : 0, i.e. cond ? 0 : adj_out
            AdjointExpr::SelectFalse(y_bar, cond) => {
                let zero = self.next_var();
                let zero_op = self.next_op();
                self.adjoint_ops.push(WengertOp {
                    id: zero_op, result: zero, op: PrimalOp::Constant(0.0),
                    inputs: vec![], saved_for_backward: false, checkpointed: false,
                });
                (PrimalOp::Select, vec![cond, zero, y_bar])
            }
        };
        self.adjoint_ops.push(WengertOp {
            id: op_id, result, op, inputs,
            saved_for_backward: false, checkpointed: false,
        });
        result
    }

    fn accumulate_adjoint(&mut self, var: VarId, value: VarId) {
        if let Some(&existing) = self.adjoint_vars.get(&var) {
            let sum = self.next_var();
            let op_id = self.next_op();
            self.adjoint_ops.push(WengertOp {
                id: op_id, result: sum, op: PrimalOp::Add,
                inputs: vec![existing, value],
                saved_for_backward: false, checkpointed: false,
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
pub fn analyze_saved_tensors(
    primal: &WengertList,
    adjoint: &WengertList,
) -> Vec<SavedTensorInfo> {
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
}

impl<'a> WengertExtractor<'a> {
    pub fn new(interner: &'a Interner) -> Self {
        WengertExtractor {
            interner,
            list: WengertList {
                ops: Vec::new(),
                output: 0,
                var_names: HashMap::new(),
            },
            symbol_to_var: HashMap::new(),
            next_var: 0,
            is_static: true,
            param_symbols: HashSet::new(),
        }
    }

    fn alloc_var(&mut self) -> VarId {
        let id = self.next_var;
        self.next_var += 1;
        id
    }

    /// Register a parameter symbol (needs gradient).
    pub fn register_param(&mut self, sym: nsl_ast::Symbol) {
        let var = self.alloc_var();
        self.symbol_to_var.insert(sym, var);
        self.param_symbols.insert(sym);

        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
        self.list.var_names.insert(var, name.clone());
        self.list.ops.push(WengertOp {
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
        self.list.ops.push(WengertOp {
            id: self.list.ops.len() as u32,
            result: var,
            op: PrimalOp::Input(name),
            inputs: vec![],
            saved_for_backward: false,
            checkpointed: false,
        });
    }

    /// Extract statements into the Wengert list.
    /// Returns false if dynamic control flow is detected.
    pub fn extract_stmts(&mut self, stmts: &[nsl_ast::stmt::Stmt]) -> bool {
        for stmt in stmts {
            if !self.extract_stmt(stmt) {
                self.is_static = false;
                return false;
            }
        }
        true
    }

    fn extract_stmt(&mut self, stmt: &nsl_ast::stmt::Stmt) -> bool {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value: Some(val), .. } => {
                if let Some(var) = self.extract_expr(val) {
                    if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                        self.symbol_to_var.insert(*sym, var);
                        let name = self.interner.resolve(sym.0).unwrap_or("?").to_string();
                        self.list.var_names.insert(var, name);
                    }
                    true
                } else {
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

            StmtKind::Expr(expr) => {
                self.extract_expr(expr).is_some()
            }

            // Dynamic control flow -> not static
            StmtKind::While { .. } => {
                self.is_static = false;
                false
            }

            // For loops: could be static if range is compile-time known,
            // but conservatively mark as dynamic for M40b
            StmtKind::For { .. } => {
                self.is_static = false;
                false
            }

            // If/else: extract both branches, emit Select ops for variables
            // that differ between branches. The condition is saved for backward.
            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                self.extract_if(condition, then_block, elif_clauses, else_block.as_ref())
            }

            // Other statements pass through (assign, decorated, etc.)
            _ => true,
        }
    }

    /// Extract an expression into a WengertOp, returning its VarId.
    /// Returns None if the expression contains dynamic control flow.
    fn extract_expr(&mut self, expr: &nsl_ast::expr::Expr) -> Option<VarId> {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                self.symbol_to_var.get(sym).copied()
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
                    _ => return None, // Unsupported op for AD
                };
                self.list.ops.push(WengertOp {
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
                self.list.ops.push(WengertOp {
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
                // Extract function name
                let func_name = if let ExprKind::Ident(sym) = &callee.kind {
                    self.interner.resolve(sym.0).unwrap_or("").to_string()
                } else {
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
                    "relu" => PrimalOp::Relu,
                    "sigmoid" => PrimalOp::Sigmoid,
                    "tanh" => PrimalOp::Tanh,
                    "exp" => PrimalOp::Exp,
                    "log" => PrimalOp::Log,
                    "sqrt" => PrimalOp::Sqrt,
                    "matmul" => PrimalOp::Matmul,
                    // Transpose and Softmax are struct variants requiring dim args --
                    // fall back to tape for M40b; proper arg extraction in M40c
                    "transpose" | "softmax" => return None,
                    _ => return None, // Unknown function -- can't differentiate
                };

                self.list.ops.push(WengertOp {
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
                self.list.ops.push(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(*v as f64),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::FloatLiteral(v) => {
                let result = self.alloc_var();
                self.list.ops.push(WengertOp {
                    id: self.list.ops.len() as u32,
                    result,
                    op: PrimalOp::Constant(*v),
                    inputs: vec![],
                    saved_for_backward: false,
                    checkpointed: false,
                });
                Some(result)
            }

            ExprKind::Paren(inner) => self.extract_expr(inner),

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
            // Desugar: elif chain becomes nested if/else
            let (elif_cond, elif_block) = &elif_clauses[0];
            let remaining_elifs = &elif_clauses[1..];
            // The "else" of this elif is either the remaining elifs or the final else
            let nested_else = if remaining_elifs.is_empty() {
                else_block
            } else {
                // For simplicity, only support single elif + optional else
                // More complex chains fall back to dynamic
                self.is_static = false;
                return false;
            };
            if !self.extract_if(elif_cond, elif_block, &[], nested_else) {
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
        let all_symbols: HashSet<nsl_ast::Symbol> = then_symbols.keys()
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
                    self.list.ops.push(WengertOp {
                        id: self.list.ops.len() as u32,
                        result,
                        op: PrimalOp::Select,
                        inputs: vec![cond_var, t, e],
                        saved_for_backward: false,
                        checkpointed: false,
                    });
                    self.symbol_to_var.insert(*sym, result);
                    // Propagate variable name
                    if let Some(name) = self.list.var_names.get(&t).cloned()
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
            let t_out = if then_output != output_before { then_output } else { output_before };
            let e_out = if else_output != output_before { else_output } else { output_before };
            if t_out != e_out {
                let result = self.alloc_var();
                self.list.ops.push(WengertOp {
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
        self.param_symbols.iter()
            .filter_map(|sym| self.symbol_to_var.get(sym).copied())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertList, WengertOp};

    fn make_op(id: OpId, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp { id, result, op, inputs, saved_for_backward: false, checkpointed: false }
    }

    #[test]
    fn test_adjoint_generator_simple_add() {
        let primal = WengertList {
            ops: vec![
                make_op(0, 0, PrimalOp::Input("a".into()), vec![]),
                make_op(1, 1, PrimalOp::Input("b".into()), vec![]),
                make_op(2, 2, PrimalOp::Add, vec![0, 1]),
            ],
            output: 2, var_names: HashMap::new(),
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
            output: 2, var_names: HashMap::new(),
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
            output: 1, var_names: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![
                make_op(0, 10, PrimalOp::Constant(1.0), vec![]),
                make_op(1, 11, PrimalOp::Mul, vec![10, 0]),
            ],
            output: 10, var_names: HashMap::new(),
        };
        let saved = analyze_saved_tensors(&primal, &adjoint);
        assert_eq!(saved.len(), 1);
        assert_eq!(saved[0].var, 0);
    }

    #[test]
    fn test_saved_tensor_no_cross_reference() {
        let primal = WengertList {
            ops: vec![make_op(0, 0, PrimalOp::Input("x".into()), vec![])],
            output: 0, var_names: HashMap::new(),
        };
        let adjoint = WengertList {
            ops: vec![make_op(0, 10, PrimalOp::Constant(1.0), vec![])],
            output: 10, var_names: HashMap::new(),
        };
        assert!(analyze_saved_tensors(&primal, &adjoint).is_empty());
    }

    // --- M40b: WengertExtractor tests ---

    use nsl_ast::expr::ExprKind;
    use nsl_ast::stmt::StmtKind;
    use nsl_lexer::Interner;

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
                body: nsl_ast::stmt::Block { stmts: vec![], span: nsl_errors::Span::dummy() },
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
                body: nsl_ast::stmt::Block { stmts: vec![], span: nsl_errors::Span::dummy() },
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

    fn make_let_stmt(
        sym: nsl_ast::Symbol,
        value: nsl_ast::expr::Expr,
    ) -> nsl_ast::stmt::Stmt {
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
        nsl_ast::stmt::Block { stmts, span: dummy_span() }
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
        assert!(extractor.is_static_graph(), "If/else should not make graph dynamic");

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
        let has_condition = list.ops.iter().any(|op| matches!(&op.op, PrimalOp::Condition(CompareKind::Gt)));
        assert!(has_condition, "Should contain a Condition(Gt) op for `x > 0`");

        // Verify a Select op was emitted
        let select_ops: Vec<_> = list.ops.iter().filter(|op| matches!(&op.op, PrimalOp::Select)).collect();
        assert!(!select_ops.is_empty(), "Should contain at least one Select op");

        // The Select op should have 3 inputs: [cond, true_val, false_val]
        for sel in &select_ops {
            assert_eq!(sel.inputs.len(), 3, "Select should have 3 inputs (cond, true, false)");
        }

        // The output should be the Select result
        let last_select = select_ops.last().unwrap();
        assert_eq!(list.output, last_select.result, "Output should be the Select result");
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
        let cond_op = list.ops.iter().find(|op| matches!(&op.op, PrimalOp::Condition(_))).unwrap();
        let cond_var = cond_op.result;

        // Find the Select op that uses this condition
        let select_op = list.ops.iter().find(|op| {
            matches!(&op.op, PrimalOp::Select) && op.inputs[0] == cond_var
        });
        assert!(select_op.is_some(), "Select should reference the saved condition variable");
    }
}
