//! M46: Compile-time determinism checking.
//!
//! Classifies operations as deterministic or non-deterministic,
//! detects implicit RNG usage, and tracks explicit seed state.

use std::collections::{HashMap, HashSet};
use nsl_ast::Symbol;
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::stmt::{Block, StmtKind};
use nsl_errors::Diagnostic;
use nsl_lexer::Interner;

/// How strictly determinism is enforced.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeterminismMode {
    /// No checking (default).
    Off,
    /// Only @deterministic-decorated functions are checked.
    FunctionLevel,
    /// All functions are checked (--deterministic flag).
    Global,
}

/// Classification of a function's determinism.
#[derive(Clone, Debug, PartialEq)]
pub enum DeterminismClass {
    /// No non-deterministic operations.
    Deterministic,
    /// Uses non-deterministic GPU ops (auto-fixable with deterministic kernel variants).
    NonDeterministicGpu { ops: Vec<String> },
    /// Uses implicit RNG without explicit seed (compile error).
    NonDeterministicRng { calls: Vec<String> },
    /// Contains external/unknown calls.
    Unknown,
}

/// RNG variable state tracking.
#[derive(Clone, Debug, PartialEq)]
pub enum RngState {
    /// Created with explicit seed: Rng(seed=42)
    ExplicitSeed(i64),
    /// Derived from another explicit Rng (.fork())
    Derived,
    /// Implicit/global RNG — error under --deterministic
    Implicit,
}

/// Non-deterministic operation categories.
#[derive(Clone, Debug, PartialEq)]
pub enum NonDetCategory {
    /// GPU atomic-based reduction — auto-fixable with sort-based kernel
    GpuAtomic,
    /// Implicit RNG — requires explicit seed argument
    ImplicitRng,
    /// Algorithm selection (cuBLAS heuristics) — auto-fixable
    AlgorithmSelection,
    /// External source (float non-associativity, thread scheduling) — warning only
    External,
}

/// Checks tensor operations for non-determinism.
pub struct DeterminismChecker {
    mode: DeterminismMode,
    deterministic_fns: HashSet<String>,
    rng_variables: HashMap<Symbol, RngState>,
    pub diagnostics: Vec<Diagnostic>,
}

impl DeterminismChecker {
    pub fn new(mode: DeterminismMode) -> Self {
        DeterminismChecker {
            mode,
            deterministic_fns: HashSet::new(),
            rng_variables: HashMap::new(),
            diagnostics: Vec::new(),
        }
    }

    /// Mark a function as @deterministic.
    pub fn mark_deterministic(&mut self, name: &str) {
        self.deterministic_fns.insert(name.to_string());
    }

    /// Check if a function is marked @deterministic.
    pub fn is_deterministic_fn(&self, name: &str) -> bool {
        self.deterministic_fns.contains(name)
    }

    /// Register an RNG variable with its seed state.
    pub fn register_rng(&mut self, sym: Symbol, state: RngState) {
        self.rng_variables.insert(sym, state);
    }

    /// Classify a tensor operation by its determinism properties.
    pub fn classify_op(&self, op_name: &str) -> NonDetCategory {
        match op_name {
            // Category 1: GPU atomic reductions — auto-fixable
            "reduce_sum" | "reduce_mean" | "scatter_add" | "embedding_backward" =>
                NonDetCategory::GpuAtomic,

            // Category 2: Algorithm selection — auto-fixable
            "matmul" | "conv2d" =>
                NonDetCategory::AlgorithmSelection,

            // Category 3: Implicit RNG — compile error
            "rand" | "randn" | "dropout" | "random_normal" | "random_uniform" =>
                NonDetCategory::ImplicitRng,

            // Everything else is deterministic or external
            _ => NonDetCategory::External,
        }
    }

    /// Check if a function call is allowed under the current determinism mode.
    ///
    /// Returns a diagnostic if the call violates determinism requirements.
    pub fn check_call(
        &self,
        func_name: &str,
        has_rng_arg: bool,
        span: nsl_errors::Span,
    ) -> Option<Diagnostic> {
        if self.mode == DeterminismMode::Off {
            return None;
        }

        let category = self.classify_op(func_name);
        match category {
            NonDetCategory::ImplicitRng if !has_rng_arg => {
                Some(Diagnostic::error(format!(
                    "'{func_name}' uses implicit RNG — pass explicit 'rng: Rng' argument in deterministic mode"
                )).with_label(span, "non-deterministic call"))
            }
            NonDetCategory::GpuAtomic => {
                // Auto-fixable — emit info diagnostic
                Some(Diagnostic::warning(format!(
                    "'{func_name}' uses GPU atomics — will auto-select deterministic kernel variant"
                )).with_label(span, "auto-fixed"))
            }
            _ => None,
        }
    }

    /// Get the deterministic kernel variant name for an op.
    pub fn deterministic_variant(&self, op_name: &str) -> Option<&'static str> {
        match op_name {
            "reduce_sum" => Some("nsl_tensor_reduce_sum_deterministic"),
            "reduce_mean" => Some("nsl_tensor_reduce_mean_deterministic"),
            "scatter_add" => Some("nsl_tensor_scatter_add_deterministic"),
            _ => None,
        }
    }

    /// Walk a module's AST and emit diagnostics for all non-deterministic ops.
    ///
    /// Used by `nsl check --deterministic` to surface warnings/errors before codegen.
    pub fn scan_module(&mut self, module: &nsl_ast::Module, interner: &Interner) {
        for stmt in &module.stmts {
            self.scan_stmt(stmt, interner);
        }
    }

    fn scan_stmt(&mut self, stmt: &nsl_ast::stmt::Stmt, interner: &Interner) {
        match &stmt.kind {
            StmtKind::FnDef(fndef) => {
                self.scan_block(&fndef.body, interner);
            }
            StmtKind::ModelDef(modeldef) => {
                for member in &modeldef.members {
                    if let nsl_ast::decl::ModelMember::Method(fndef, _) = member {
                        self.scan_block(&fndef.body, interner);
                    }
                }
            }
            StmtKind::VarDecl { value: Some(expr), .. }
            | StmtKind::Return(Some(expr))
            | StmtKind::Expr(expr) => {
                self.scan_expr(expr, interner);
            }
            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                self.scan_expr(condition, interner);
                self.scan_block(then_block, interner);
                for (cond, block) in elif_clauses {
                    self.scan_expr(cond, interner);
                    self.scan_block(block, interner);
                }
                if let Some(eb) = else_block {
                    self.scan_block(eb, interner);
                }
            }
            StmtKind::For { iterable, body, .. } => {
                self.scan_expr(iterable, interner);
                self.scan_block(body, interner);
            }
            StmtKind::While { condition, body } => {
                self.scan_expr(condition, interner);
                self.scan_block(body, interner);
            }
            _ => {}
        }
    }

    fn scan_block(&mut self, block: &Block, interner: &Interner) {
        for stmt in &block.stmts {
            self.scan_stmt(stmt, interner);
        }
    }

    fn scan_expr(&mut self, expr: &Expr, interner: &Interner) {
        match &expr.kind {
            ExprKind::Call { callee, args, .. } => {
                if let ExprKind::Ident(sym) = &callee.kind {
                    if let Some(name) = interner.resolve(sym.0) {
                        // Check for explicit RNG keyword arg named "rng"
                        let has_rng_arg = args.iter().any(|a| {
                            a.name.as_ref().and_then(|sym| interner.resolve(sym.0))
                                .map_or(false, |s| s == "rng")
                        });
                        if let Some(diag) = self.check_call(name, has_rng_arg, expr.span) {
                            self.diagnostics.push(diag);
                        }
                    }
                }
                for arg in args {
                    self.scan_expr(&arg.value, interner);
                }
            }
            ExprKind::BinaryOp { left, right, .. } => {
                self.scan_expr(left, interner);
                self.scan_expr(right, interner);
            }
            ExprKind::UnaryOp { operand, .. } => {
                self.scan_expr(operand, interner);
            }
            ExprKind::Pipe { left, right } => {
                self.scan_expr(left, interner);
                self.scan_expr(right, interner);
            }
            ExprKind::MemberAccess { object, .. } => {
                self.scan_expr(object, interner);
            }
            ExprKind::ListLiteral(exprs) | ExprKind::TupleLiteral(exprs) => {
                for e in exprs {
                    self.scan_expr(e, interner);
                }
            }
            ExprKind::Subscript { object, .. } => {
                self.scan_expr(object, interner);
            }
            ExprKind::IfExpr { condition, then_expr, else_expr } => {
                self.scan_expr(condition, interner);
                self.scan_expr(then_expr, interner);
                self.scan_expr(else_expr, interner);
            }
            ExprKind::Lambda { body, .. } => {
                self.scan_expr(body, interner);
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_gpu_atomic_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("reduce_sum"), NonDetCategory::GpuAtomic);
        assert_eq!(checker.classify_op("scatter_add"), NonDetCategory::GpuAtomic);
    }

    #[test]
    fn classify_implicit_rng_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("rand"), NonDetCategory::ImplicitRng);
        assert_eq!(checker.classify_op("dropout"), NonDetCategory::ImplicitRng);
    }

    #[test]
    fn classify_algorithm_selection_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("matmul"), NonDetCategory::AlgorithmSelection);
        assert_eq!(checker.classify_op("conv2d"), NonDetCategory::AlgorithmSelection);
    }

    #[test]
    fn classify_deterministic_ops() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.classify_op("relu"), NonDetCategory::External);
        assert_eq!(checker.classify_op("add"), NonDetCategory::External);
    }

    #[test]
    fn implicit_rng_errors_in_global_mode() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        let diag = checker.check_call("rand", false, nsl_errors::Span::dummy());
        assert!(diag.is_some());
        // With explicit rng arg → no error
        let diag = checker.check_call("rand", true, nsl_errors::Span::dummy());
        assert!(diag.is_none());
    }

    #[test]
    fn gpu_atomic_warns_in_global_mode() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        let diag = checker.check_call("reduce_sum", false, nsl_errors::Span::dummy());
        assert!(diag.is_some()); // warning, not error
    }

    #[test]
    fn no_checks_in_off_mode() {
        let checker = DeterminismChecker::new(DeterminismMode::Off);
        let diag = checker.check_call("rand", false, nsl_errors::Span::dummy());
        assert!(diag.is_none());
    }

    #[test]
    fn deterministic_variant_selection() {
        let checker = DeterminismChecker::new(DeterminismMode::Global);
        assert_eq!(checker.deterministic_variant("reduce_sum"), Some("nsl_tensor_reduce_sum_deterministic"));
        assert_eq!(checker.deterministic_variant("relu"), None);
    }

    #[test]
    fn mark_and_query_deterministic_fn() {
        let mut checker = DeterminismChecker::new(DeterminismMode::FunctionLevel);
        assert!(!checker.is_deterministic_fn("attention"));
        checker.mark_deterministic("attention");
        assert!(checker.is_deterministic_fn("attention"));
    }

    #[test]
    fn rng_state_tracking() {
        let mut checker = DeterminismChecker::new(DeterminismMode::Global);
        let mut interner = nsl_lexer::Interner::new();
        let sym = nsl_ast::Symbol(interner.get_or_intern("rng"));
        checker.register_rng(sym, RngState::ExplicitSeed(42));
        assert_eq!(checker.rng_variables.get(&sym), Some(&RngState::ExplicitSeed(42)));
    }
}
