//! M45: Compile-time NaN risk analysis.
//!
//! Provides both a constraint-based API (`check_call_risk`, `infer_constraint`)
//! and a whole-module AST walker (`analyze_module`) that scans all function
//! bodies for NaN/Inf risk patterns: `log(x)` on unconstrained args,
//! `sqrt(x)` on possibly-negative args, and division by potentially-zero tensors.

use std::collections::HashMap;
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::stmt::{StmtKind, Block};
use nsl_ast::{Module, Symbol};
use nsl_errors::Diagnostic;
use nsl_lexer::Interner;

/// Known value constraint for a tensor binding.
#[derive(Clone, Debug, PartialEq)]
pub enum ValueConstraint {
    Unconstrained,
    NonNegative,        // >= 0 (from relu, abs, x*x)
    StrictlyPositive,   // > 0  (from relu(x) + eps, softplus, exp)
}

/// Analyzes function bodies for NaN/Inf risk patterns.
pub struct NanAnalyzer {
    constraints: HashMap<Symbol, ValueConstraint>,
    pub diagnostics: Vec<Diagnostic>,
}

impl Default for NanAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl NanAnalyzer {
    pub fn new() -> Self {
        NanAnalyzer { constraints: HashMap::new(), diagnostics: Vec::new() }
    }

    /// Mark a binding as having a known constraint.
    pub fn set_constraint(&mut self, sym: Symbol, constraint: ValueConstraint) {
        self.constraints.insert(sym, constraint);
    }

    /// Get the constraint for a binding.
    pub fn get_constraint(&self, sym: &Symbol) -> ValueConstraint {
        self.constraints.get(sym).cloned().unwrap_or(ValueConstraint::Unconstrained)
    }

    /// Check if a function call is a NaN risk given its argument constraints.
    ///
    /// Returns a warning diagnostic if a risk is detected, None otherwise.
    pub fn check_call_risk(
        &self,
        func_name: &str,
        arg_syms: &[Symbol],
        span: nsl_errors::Span,
    ) -> Option<Diagnostic> {
        match func_name {
            "log" => {
                if let Some(arg) = arg_syms.first() {
                    let c = self.get_constraint(arg);
                    if c != ValueConstraint::StrictlyPositive {
                        return Some(
                            Diagnostic::warning("log() argument may be zero or negative — NaN risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            "sqrt" => {
                if let Some(arg) = arg_syms.first() {
                    let c = self.get_constraint(arg);
                    if c == ValueConstraint::Unconstrained {
                        return Some(
                            Diagnostic::warning("sqrt() argument may be negative — NaN risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            // Division: second argument could be zero -> Inf
            "div" => {
                if arg_syms.len() >= 2 {
                    let c = self.get_constraint(&arg_syms[1]);
                    if c != ValueConstraint::StrictlyPositive {
                        return Some(
                            Diagnostic::warning("division by tensor that may contain zeros — Inf risk")
                                .with_label(span, "here")
                        );
                    }
                }
                None
            }
            // pow and log(softmax(...)) detection deferred to M45b
            _ => None,
        }
    }

    /// Infer constraints from known function results.
    pub fn infer_constraint(&self, func_name: &str) -> ValueConstraint {
        match func_name {
            "relu" | "abs" => ValueConstraint::NonNegative,
            "exp" | "softplus" => ValueConstraint::StrictlyPositive,
            "sigmoid" => ValueConstraint::StrictlyPositive, // (0, 1)
            _ => ValueConstraint::Unconstrained,
        }
    }

    /// Walk an entire module's AST, scanning all function bodies for NaN risk
    /// patterns. Collects warnings into `self.diagnostics`.
    ///
    /// This is a single-pass forward walk: when a let-binding assigns the result
    /// of a constraint-producing function (relu, exp, sigmoid, …) to a variable,
    /// that variable is marked with the inferred constraint. Subsequent calls to
    /// log/sqrt/div are checked against those constraints.
    pub fn analyze_module(&mut self, module: &Module, interner: &Interner) {
        for stmt in &module.stmts {
            self.walk_stmt(stmt, interner);
        }
    }

    /// Walk a single statement, descending into function bodies and blocks.
    fn walk_stmt(&mut self, stmt: &nsl_ast::stmt::Stmt, interner: &Interner) {
        match &stmt.kind {
            StmtKind::FnDef(fndef) => {
                // Reset constraints per function body (local scope)
                let saved = self.constraints.clone();
                self.walk_block(&fndef.body, interner);
                self.constraints = saved;
            }
            StmtKind::ModelDef(modeldef) => {
                for member in &modeldef.members {
                    match member {
                        nsl_ast::decl::ModelMember::Method(fndef, _) => {
                            let saved = self.constraints.clone();
                            self.walk_block(&fndef.body, interner);
                            self.constraints = saved;
                        }
                        nsl_ast::decl::ModelMember::LayerDecl { init: Some(expr), .. } => {
                            self.walk_expr(expr, interner);
                        }
                        _ => {}
                    }
                }
            }
            StmtKind::VarDecl { pattern, value: Some(val_expr), .. } => {
                // If the RHS is a function call that produces a known constraint,
                // propagate it to the LHS binding.
                if let Some(constraint) = self.expr_constraint(val_expr, interner) {
                    if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                        if constraint != ValueConstraint::Unconstrained {
                            self.set_constraint(*sym, constraint);
                        }
                    }
                }
                // Also check the RHS expression itself for risky calls
                self.walk_expr(val_expr, interner);
            }
            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                self.walk_expr(condition, interner);
                self.walk_block(then_block, interner);
                for (cond, block) in elif_clauses {
                    self.walk_expr(cond, interner);
                    self.walk_block(block, interner);
                }
                if let Some(eb) = else_block {
                    self.walk_block(eb, interner);
                }
            }
            StmtKind::For { iterable, body, .. } => {
                self.walk_expr(iterable, interner);
                self.walk_block(body, interner);
            }
            StmtKind::While { condition, body } => {
                self.walk_expr(condition, interner);
                self.walk_block(body, interner);
            }
            StmtKind::WhileLet { expr, body, .. } => {
                self.walk_expr(expr, interner);
                self.walk_block(body, interner);
            }
            StmtKind::Return(Some(expr)) | StmtKind::Expr(expr) => {
                self.walk_expr(expr, interner);
            }
            _ => {}
        }
    }

    fn walk_block(&mut self, block: &Block, interner: &Interner) {
        for stmt in &block.stmts {
            self.walk_stmt(stmt, interner);
        }
    }

    /// Recursively walk an expression, checking calls for NaN risks.
    fn walk_expr(&mut self, expr: &Expr, interner: &Interner) {
        match &expr.kind {
            ExprKind::Call { callee, args, .. } => {
                // Resolve callee name
                if let Some(func_name) = self.resolve_call_name(callee, interner) {
                    let arg_syms: Vec<Symbol> = args.iter().filter_map(|a| {
                        if let ExprKind::Ident(sym) = &a.value.kind {
                            Some(*sym)
                        } else {
                            None
                        }
                    }).collect();

                    if let Some(diag) = self.check_call_risk(&func_name, &arg_syms, expr.span) {
                        self.diagnostics.push(diag);
                    }
                }
                // Walk into argument sub-expressions
                for arg in args {
                    self.walk_expr(&arg.value, interner);
                }
            }
            ExprKind::BinaryOp { op: _, left, right } => {
                self.walk_expr(left, interner);
                self.walk_expr(right, interner);
            }
            ExprKind::UnaryOp { op: _, operand } => {
                self.walk_expr(operand, interner);
            }
            ExprKind::Pipe { left, right } => {
                self.walk_expr(left, interner);
                self.walk_expr(right, interner);
            }
            ExprKind::MemberAccess { object, .. } => {
                self.walk_expr(object, interner);
            }
            ExprKind::ListLiteral(exprs) | ExprKind::TupleLiteral(exprs) => {
                for e in exprs {
                    self.walk_expr(e, interner);
                }
            }
            ExprKind::Subscript { object, .. } => {
                self.walk_expr(object, interner);
            }
            ExprKind::IfExpr { condition, then_expr, else_expr } => {
                self.walk_expr(condition, interner);
                self.walk_expr(then_expr, interner);
                self.walk_expr(else_expr, interner);
            }
            ExprKind::Lambda { body, .. } => {
                self.walk_expr(body, interner);
            }
            _ => {}
        }
    }

    /// Try to extract a simple function name from a callee expression.
    fn resolve_call_name(&self, callee: &Expr, interner: &Interner) -> Option<String> {
        match &callee.kind {
            ExprKind::Ident(sym) => {
                interner.resolve(sym.0).map(|s| s.to_string())
            }
            _ => None,
        }
    }

    /// Infer the constraint of an expression — used for let-binding propagation.
    fn expr_constraint(&self, expr: &Expr, interner: &Interner) -> Option<ValueConstraint> {
        match &expr.kind {
            ExprKind::Call { callee, .. } => {
                self.resolve_call_name(callee, interner)
                    .map(|name| self.infer_constraint(&name))
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::Span;
    use nsl_lexer::Interner;

    /// Helper to create a Symbol from a string via a string interner.
    fn make_sym(interner: &mut Interner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    #[test]
    fn log_unconstrained_warns() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let analyzer = NanAnalyzer::new();
        let result = analyzer.check_call_risk("log", &[x], Span::dummy());
        assert!(result.is_some(), "log() on unconstrained arg should warn");
    }

    #[test]
    fn log_strictly_positive_no_warn() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut analyzer = NanAnalyzer::new();
        analyzer.set_constraint(x, ValueConstraint::StrictlyPositive);
        let result = analyzer.check_call_risk("log", &[x], Span::dummy());
        assert!(result.is_none(), "log() on StrictlyPositive arg should not warn");
    }

    #[test]
    fn sqrt_unconstrained_warns() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let analyzer = NanAnalyzer::new();
        let result = analyzer.check_call_risk("sqrt", &[x], Span::dummy());
        assert!(result.is_some(), "sqrt() on unconstrained arg should warn");
    }

    #[test]
    fn sqrt_non_negative_no_warn() {
        let mut interner = Interner::new();
        let x = make_sym(&mut interner, "x");
        let mut analyzer = NanAnalyzer::new();
        analyzer.set_constraint(x, ValueConstraint::NonNegative);
        let result = analyzer.check_call_risk("sqrt", &[x], Span::dummy());
        assert!(result.is_none(), "sqrt() on NonNegative arg should not warn");
    }

    #[test]
    fn div_unconstrained_warns() {
        let mut interner = Interner::new();
        let a = make_sym(&mut interner, "a");
        let b = make_sym(&mut interner, "b");
        let analyzer = NanAnalyzer::new();
        let result = analyzer.check_call_risk("div", &[a, b], Span::dummy());
        assert!(result.is_some(), "div() with unconstrained divisor should warn");
    }

    #[test]
    fn relu_produces_non_negative() {
        let analyzer = NanAnalyzer::new();
        assert_eq!(analyzer.infer_constraint("relu"), ValueConstraint::NonNegative);
        assert_eq!(analyzer.infer_constraint("abs"), ValueConstraint::NonNegative);
        assert_eq!(analyzer.infer_constraint("exp"), ValueConstraint::StrictlyPositive);
        assert_eq!(analyzer.infer_constraint("sigmoid"), ValueConstraint::StrictlyPositive);
        assert_eq!(analyzer.infer_constraint("softplus"), ValueConstraint::StrictlyPositive);
        assert_eq!(analyzer.infer_constraint("unknown_fn"), ValueConstraint::Unconstrained);
    }

    /// Helper: parse NSL source and run the NaN analyzer walker.
    fn analyze_source(source: &str) -> Vec<Diagnostic> {
        let mut interner = Interner::new();
        let file_id = nsl_errors::FileId(0);
        let (tokens, _) = nsl_lexer::tokenize(source, file_id, &mut interner);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        let mut analyzer = NanAnalyzer::new();
        analyzer.analyze_module(&parse_result.module, &interner);
        analyzer.diagnostics
    }

    #[test]
    fn walker_detects_log_risk() {
        let diags = analyze_source("fn foo(x: Tensor):\n    let y = log(x)\n");
        assert!(!diags.is_empty(), "should warn about log() on unconstrained arg");
        assert!(
            diags[0].message.contains("log()"),
            "warning should mention log(): {}",
            diags[0].message
        );
    }

    #[test]
    fn walker_detects_sqrt_risk() {
        let diags = analyze_source("fn foo(x: Tensor):\n    let y = sqrt(x)\n");
        assert!(!diags.is_empty(), "should warn about sqrt() on unconstrained arg");
    }

    #[test]
    fn walker_no_warn_after_relu() {
        // relu produces NonNegative, so sqrt(relu(x)) should NOT warn.
        // But since we propagate constraints through let-bindings only, we need:
        //   let safe = relu(x)
        //   let y = sqrt(safe)
        let diags = analyze_source(
            "fn foo(x: Tensor):\n    let safe = relu(x)\n    let y = sqrt(safe)\n"
        );
        // The sqrt(safe) call should NOT warn because safe has NonNegative constraint.
        let sqrt_warns: Vec<_> = diags.iter().filter(|d| d.message.contains("sqrt()")).collect();
        assert!(sqrt_warns.is_empty(), "sqrt(relu(x)) should not warn, but got: {:?}", sqrt_warns);
    }

    #[test]
    fn walker_no_false_positive_on_safe_code() {
        let diags = analyze_source("fn foo(x: Tensor):\n    let y = relu(x)\n");
        assert!(diags.is_empty(), "relu(x) alone should produce no warnings");
    }
}
