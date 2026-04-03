//! FBIP Phase 2: Use-count pre-pass for codegen.
//!
//! Walks a function's AST before codegen to count how many times each
//! binding (Symbol) is referenced in expressions. Bindings referenced
//! exactly once after definition are "single-use" — the codegen can
//! emit in-place op variants and skip clones for these.

use std::collections::HashMap;
use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::Symbol;

/// Per-function variable use counts, computed before codegen.
#[derive(Debug, Default)]
pub struct UseCountMap {
    counts: HashMap<Symbol, u32>,
}

impl UseCountMap {
    /// Returns true if `sym` is referenced exactly once in the function body.
    /// Single-use bindings are safe for in-place mutation and clone elision.
    #[inline]
    pub fn is_single_use(&self, sym: &Symbol) -> bool {
        self.counts.get(sym).copied() == Some(1)
    }

    /// Returns the use count for a binding (0 if never referenced).
    #[inline]
    pub fn use_count(&self, sym: &Symbol) -> u32 {
        self.counts.get(sym).copied().unwrap_or(0)
    }
}

/// Count all identifier references in a function body.
pub fn analyze_use_counts(body: &Block) -> UseCountMap {
    let mut map = UseCountMap::default();
    for stmt in &body.stmts {
        count_stmt(&mut map.counts, stmt);
    }
    map
}

fn count_stmt(counts: &mut HashMap<Symbol, u32>, stmt: &Stmt) {
    match &stmt.kind {
        StmtKind::VarDecl { value: Some(expr), .. } => {
            count_expr(counts, expr);
        }
        StmtKind::VarDecl { value: None, .. } => {}
        StmtKind::Assign { target, value, .. } => {
            count_expr(counts, target);
            count_expr(counts, value);
        }
        StmtKind::Expr(expr) => count_expr(counts, expr),
        StmtKind::Return(Some(expr)) | StmtKind::Yield(Some(expr)) => {
            count_expr(counts, expr);
        }
        StmtKind::Return(None) | StmtKind::Yield(None) => {}
        StmtKind::If { condition, then_block, elif_clauses, else_block } => {
            count_expr(counts, condition);
            count_block(counts, then_block);
            for (cond, block) in elif_clauses {
                count_expr(counts, cond);
                count_block(counts, block);
            }
            if let Some(block) = else_block {
                count_block(counts, block);
            }
        }
        StmtKind::While { condition, body, .. } => {
            count_expr(counts, condition);
            count_block(counts, body);
        }
        StmtKind::WhileLet { expr, body, .. } => {
            count_expr(counts, expr);
            count_block(counts, body);
        }
        StmtKind::For { iterable, body, .. } => {
            count_expr(counts, iterable);
            count_block(counts, body);
        }
        StmtKind::Match { subject, arms } => {
            count_expr(counts, subject);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    count_expr(counts, guard);
                }
                count_block(counts, &arm.body);
            }
        }
        StmtKind::Break | StmtKind::Continue => {}
        // Skip nested defs — they have their own use-count analysis
        StmtKind::FnDef(_) | StmtKind::ModelDef(_) | StmtKind::StructDef(_)
        | StmtKind::EnumDef(_) | StmtKind::TraitDef(_) => {}
        // Skip imports
        StmtKind::Import(_) | StmtKind::FromImport(_) => {}
        // Decorated: count uses in the inner statement
        StmtKind::Decorated { stmt, .. } => count_stmt(counts, stmt),
        // Train/grad/quant/kernel/tokenizer/dataset/datatype/serve blocks:
        // conservatively skip — FBIP optimizations won't apply inside these.
        _ => {}
    }
}

fn count_block(counts: &mut HashMap<Symbol, u32>, block: &Block) {
    for stmt in &block.stmts {
        count_stmt(counts, stmt);
    }
}

fn count_expr(counts: &mut HashMap<Symbol, u32>, expr: &Expr) {
    match &expr.kind {
        ExprKind::Ident(sym) => {
            *counts.entry(*sym).or_insert(0) += 1;
        }
        ExprKind::BinaryOp { left, right, .. } => {
            count_expr(counts, left);
            count_expr(counts, right);
        }
        ExprKind::UnaryOp { operand, .. } => count_expr(counts, operand),
        ExprKind::Pipe { left, right } => {
            count_expr(counts, left);
            count_expr(counts, right);
        }
        ExprKind::MemberAccess { object, .. } => count_expr(counts, object),
        ExprKind::Subscript { object, index } => {
            count_expr(counts, object);
            match index.as_ref() {
                SubscriptKind::Index(e) => count_expr(counts, e),
                SubscriptKind::Slice { lower, upper, step } => {
                    if let Some(e) = lower { count_expr(counts, e); }
                    if let Some(e) = upper { count_expr(counts, e); }
                    if let Some(e) = step { count_expr(counts, e); }
                }
                SubscriptKind::MultiDim(dims) => {
                    for d in dims {
                        match d {
                            SubscriptKind::Index(e) => count_expr(counts, e),
                            SubscriptKind::Slice { lower, upper, step } => {
                                if let Some(e) = lower { count_expr(counts, e); }
                                if let Some(e) = upper { count_expr(counts, e); }
                                if let Some(e) = step { count_expr(counts, e); }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
        ExprKind::Call { callee, args } => {
            count_expr(counts, callee);
            for arg in args { count_expr(counts, &arg.value); }
        }
        ExprKind::Lambda { body, .. } => count_expr(counts, body),
        ExprKind::ListComp { element, generators } => {
            count_expr(counts, element);
            for gen in generators {
                count_expr(counts, &gen.iterable);
                for cond in &gen.conditions {
                    count_expr(counts, cond);
                }
            }
        }
        ExprKind::IfExpr { condition, then_expr, else_expr } => {
            count_expr(counts, condition);
            count_expr(counts, then_expr);
            count_expr(counts, else_expr);
        }
        ExprKind::BlockExpr(block) => {
            count_block(counts, block);
        }
        ExprKind::ListLiteral(elems) | ExprKind::TupleLiteral(elems) => {
            for e in elems { count_expr(counts, e); }
        }
        ExprKind::DictLiteral(pairs) => {
            for (k, v) in pairs {
                count_expr(counts, k);
                count_expr(counts, v);
            }
        }
        ExprKind::FString(parts) => {
            for part in parts {
                if let nsl_ast::expr::FStringPart::Expr(e) = part {
                    count_expr(counts, e);
                }
            }
        }
        // Literals and SelfRef don't reference bindings
        ExprKind::IntLiteral(_) | ExprKind::FloatLiteral(_) | ExprKind::StringLiteral(_)
        | ExprKind::BoolLiteral(_) | ExprKind::NoneLiteral | ExprKind::SelfRef => {}
        ExprKind::Range { start, end, .. } => {
            if let Some(e) = start { count_expr(counts, e); }
            if let Some(e) = end { count_expr(counts, e); }
        }
        ExprKind::Paren(e) | ExprKind::Await(e) => count_expr(counts, e),
        ExprKind::MatchExpr { subject, arms } => {
            count_expr(counts, subject);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    count_expr(counts, guard);
                }
                for stmt in &arm.body.stmts {
                    count_stmt(counts, stmt);
                }
            }
        }
        ExprKind::Error => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::Span;
    fn make_sym(interner: &mut nsl_lexer::Interner, name: &str) -> Symbol {
        Symbol(interner.get_or_intern(name))
    }

    fn ident_expr(s: Symbol) -> Expr {
        Expr { id: nsl_ast::NodeId(0), kind: ExprKind::Ident(s), span: Span::DUMMY }
    }

    fn expr_stmt(e: Expr) -> Stmt {
        Stmt { id: nsl_ast::NodeId(0), kind: StmtKind::Expr(e), span: Span::DUMMY }
    }

    fn block(stmts: Vec<Stmt>) -> Block {
        Block { span: Span::DUMMY, stmts }
    }

    #[test]
    fn test_single_use() {
        let mut interner = nsl_lexer::Interner::new();
        let x = make_sym(&mut interner, "x");
        let body = block(vec![expr_stmt(ident_expr(x))]);
        let map = analyze_use_counts(&body);
        assert!(map.is_single_use(&x));
        assert_eq!(map.use_count(&x), 1);
    }

    #[test]
    fn test_multi_use() {
        let mut interner = nsl_lexer::Interner::new();
        let x = make_sym(&mut interner, "x");
        let body = block(vec![expr_stmt(ident_expr(x)), expr_stmt(ident_expr(x))]);
        let map = analyze_use_counts(&body);
        assert!(!map.is_single_use(&x));
        assert_eq!(map.use_count(&x), 2);
    }

    #[test]
    fn test_unused() {
        let mut interner = nsl_lexer::Interner::new();
        let x = make_sym(&mut interner, "x");
        let body = block(vec![]);
        let map = analyze_use_counts(&body);
        assert!(!map.is_single_use(&x));
        assert_eq!(map.use_count(&x), 0);
    }
}
