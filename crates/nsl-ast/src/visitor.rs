use crate::expr::Expr;
use crate::pattern::Pattern;
use crate::stmt::{Block, Stmt};
use crate::types::TypeExpr;
use crate::Module;

/// Trait for visiting AST nodes. Default implementations walk children.
pub trait Visitor: Sized {
    fn visit_module(&mut self, module: &Module) {
        walk_module(self, module);
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        walk_stmt(self, stmt);
    }

    fn visit_expr(&mut self, expr: &Expr) {
        walk_expr(self, expr);
    }

    fn visit_type_expr(&mut self, _ty: &TypeExpr) {
        // Default: no children to walk for types (could be extended)
    }

    fn visit_pattern(&mut self, _pat: &Pattern) {
        // Default: no children to walk
    }

    fn visit_block(&mut self, block: &Block) {
        walk_block(self, block);
    }
}

pub fn walk_module(v: &mut impl Visitor, module: &Module) {
    for stmt in &module.stmts {
        v.visit_stmt(stmt);
    }
}

pub fn walk_block(v: &mut impl Visitor, block: &Block) {
    for stmt in &block.stmts {
        v.visit_stmt(stmt);
    }
}

pub fn walk_stmt(v: &mut impl Visitor, stmt: &Stmt) {
    use crate::stmt::StmtKind;
    match &stmt.kind {
        StmtKind::VarDecl { value, .. } => {
            if let Some(val) = value {
                v.visit_expr(val);
            }
        }
        StmtKind::FnDef(f) => {
            v.visit_block(&f.body);
        }
        StmtKind::ModelDef(m) => {
            for member in &m.members {
                match member {
                    crate::decl::ModelMember::Method(f) => v.visit_block(&f.body),
                    crate::decl::ModelMember::LayerDecl { init, .. } => {
                        if let Some(init) = init {
                            v.visit_expr(init);
                        }
                    }
                }
            }
        }
        StmtKind::If {
            condition,
            then_block,
            elif_clauses,
            else_block,
        } => {
            v.visit_expr(condition);
            v.visit_block(then_block);
            for (cond, block) in elif_clauses {
                v.visit_expr(cond);
                v.visit_block(block);
            }
            if let Some(block) = else_block {
                v.visit_block(block);
            }
        }
        StmtKind::For {
            iterable, body, ..
        } => {
            v.visit_expr(iterable);
            v.visit_block(body);
        }
        StmtKind::While { condition, body } => {
            v.visit_expr(condition);
            v.visit_block(body);
        }
        StmtKind::Return(Some(e)) | StmtKind::Yield(Some(e)) => {
            v.visit_expr(e);
        }
        StmtKind::Assign { target, value, .. } => {
            v.visit_expr(target);
            v.visit_expr(value);
        }
        StmtKind::Expr(e) => {
            v.visit_expr(e);
        }
        StmtKind::Decorated { stmt, .. } => {
            v.visit_stmt(stmt);
        }
        _ => {}
    }
}

pub fn walk_expr(v: &mut impl Visitor, expr: &Expr) {
    use crate::expr::ExprKind;
    match &expr.kind {
        ExprKind::BinaryOp { left, right, .. } => {
            v.visit_expr(left);
            v.visit_expr(right);
        }
        ExprKind::UnaryOp { operand, .. } => {
            v.visit_expr(operand);
        }
        ExprKind::Pipe { left, right } => {
            v.visit_expr(left);
            v.visit_expr(right);
        }
        ExprKind::MemberAccess { object, .. } => {
            v.visit_expr(object);
        }
        ExprKind::Subscript { object, .. } => {
            v.visit_expr(object);
        }
        ExprKind::Call { callee, args, .. } => {
            v.visit_expr(callee);
            for arg in args {
                v.visit_expr(&arg.value);
            }
        }
        ExprKind::Lambda { body, .. } => {
            v.visit_expr(body);
        }
        ExprKind::ListLiteral(items) | ExprKind::TupleLiteral(items) => {
            for item in items {
                v.visit_expr(item);
            }
        }
        ExprKind::DictLiteral(entries) => {
            for (k, val) in entries {
                v.visit_expr(k);
                v.visit_expr(val);
            }
        }
        ExprKind::IfExpr {
            condition,
            then_expr,
            else_expr,
        } => {
            v.visit_expr(condition);
            v.visit_expr(then_expr);
            v.visit_expr(else_expr);
        }
        ExprKind::Paren(inner) => {
            v.visit_expr(inner);
        }
        ExprKind::Await(inner) => {
            v.visit_expr(inner);
        }
        _ => {}
    }
}
