use crate::expr::{Expr, ExprKind, FStringPart, SubscriptKind};
use crate::pattern::{Pattern, PatternKind};
use crate::stmt::{Block, Stmt, StmtKind};
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

    fn visit_pattern(&mut self, pat: &Pattern) {
        walk_pattern(self, pat);
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

pub fn walk_pattern(v: &mut impl Visitor, pat: &Pattern) {
    match &pat.kind {
        PatternKind::Tuple(pats) | PatternKind::List(pats) | PatternKind::Or(pats) => {
            for p in pats {
                v.visit_pattern(p);
            }
        }
        PatternKind::Constructor { args, .. } => {
            for p in args {
                v.visit_pattern(p);
            }
        }
        PatternKind::Struct { fields, .. } => {
            for f in fields {
                if let Some(p) = &f.pattern {
                    v.visit_pattern(p);
                }
            }
        }
        PatternKind::Guarded { pattern, guard } => {
            v.visit_pattern(pattern);
            v.visit_expr(guard);
        }
        PatternKind::Typed { pattern, .. } => {
            v.visit_pattern(pattern);
        }
        PatternKind::Literal(e) => {
            v.visit_expr(e);
        }
        PatternKind::Ident(_) | PatternKind::Wildcard | PatternKind::Rest(_) => {}
    }
}

pub fn walk_stmt(v: &mut impl Visitor, stmt: &Stmt) {
    match &stmt.kind {
        StmtKind::VarDecl { pattern, value, .. } => {
            v.visit_pattern(pattern);
            if let Some(val) = value {
                v.visit_expr(val);
            }
        }
        StmtKind::FnDef(f) => {
            for param in &f.params {
                if let Some(default) = &param.default {
                    v.visit_expr(default);
                }
            }
            v.visit_block(&f.body);
        }
        StmtKind::ModelDef(m) => {
            for param in &m.params {
                if let Some(default) = &param.default {
                    v.visit_expr(default);
                }
            }
            for member in &m.members {
                match member {
                    crate::decl::ModelMember::Method(f) => {
                        for param in &f.params {
                            if let Some(default) = &param.default {
                                v.visit_expr(default);
                            }
                        }
                        v.visit_block(&f.body);
                    }
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
        StmtKind::WhileLet { expr, body, pattern, .. } => {
            v.visit_pattern(pattern);
            v.visit_expr(expr);
            v.visit_block(body);
        }
        StmtKind::Match { subject, arms } => {
            v.visit_expr(subject);
            for arm in arms {
                v.visit_pattern(&arm.pattern);
                if let Some(guard) = &arm.guard {
                    v.visit_expr(guard);
                }
                v.visit_block(&arm.body);
            }
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
        StmtKind::Decorated { decorators, stmt } => {
            for dec in decorators {
                if let Some(args) = &dec.args {
                    for arg in args {
                        v.visit_expr(&arg.value);
                    }
                }
            }
            v.visit_stmt(stmt);
        }
        StmtKind::TrainBlock(train) => {
            use crate::block::TrainSection;
            for section in &train.sections {
                match section {
                    TrainSection::Data(stmts) => {
                        for s in stmts {
                            v.visit_stmt(s);
                        }
                    }
                    TrainSection::Optimizer(e)
                    | TrainSection::Scheduler(e)
                    | TrainSection::Distribute(e) => {
                        v.visit_expr(e);
                    }
                    TrainSection::Step { body, .. } | TrainSection::Eval { body, .. } => {
                        v.visit_block(body);
                    }
                    TrainSection::Callbacks(cbs) => {
                        for cb in cbs {
                            v.visit_block(&cb.body);
                        }
                    }
                    TrainSection::Stmt(s) => {
                        v.visit_stmt(s);
                    }
                }
            }
        }
        StmtKind::GradBlock(g) => {
            if let Some(pat) = &g.outputs {
                v.visit_pattern(pat);
            }
            v.visit_expr(&g.targets);
            v.visit_block(&g.body);
        }
        StmtKind::KernelDef(k) => {
            for param in &k.params {
                if let Some(default) = &param.default {
                    v.visit_expr(default);
                }
            }
            v.visit_block(&k.body);
        }
        StmtKind::TokenizerDef(t) => {
            for s in &t.body {
                v.visit_stmt(s);
            }
        }
        StmtKind::DatasetDef(d) => {
            v.visit_expr(&d.source);
            for s in &d.body {
                v.visit_stmt(s);
            }
        }
        StmtKind::StructDef(s) => {
            for field in &s.fields {
                if let Some(default) = &field.default {
                    v.visit_expr(default);
                }
            }
        }
        StmtKind::EnumDef(e) => {
            for variant in &e.variants {
                if let Some(val) = &variant.value {
                    v.visit_expr(val);
                }
            }
        }
        StmtKind::TraitDef(t) => {
            for method in &t.methods {
                for param in &method.params {
                    if let Some(default) = &param.default {
                        v.visit_expr(default);
                    }
                }
                v.visit_block(&method.body);
            }
        }
        _ => {}
    }
}

pub fn walk_expr(v: &mut impl Visitor, expr: &Expr) {
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
        ExprKind::Subscript { object, index } => {
            v.visit_expr(object);
            walk_subscript(v, index);
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
        ExprKind::MatchExpr { subject, arms } => {
            v.visit_expr(subject);
            for arm in arms {
                v.visit_pattern(&arm.pattern);
                if let Some(guard) = &arm.guard {
                    v.visit_expr(guard);
                }
                v.visit_block(&arm.body);
            }
        }
        ExprKind::ListComp { element, generators } => {
            v.visit_expr(element);
            for gen in generators {
                v.visit_pattern(&gen.pattern);
                v.visit_expr(&gen.iterable);
                for cond in &gen.conditions {
                    v.visit_expr(cond);
                }
            }
        }
        ExprKind::Range { start, end, .. } => {
            if let Some(s) = start {
                v.visit_expr(s);
            }
            if let Some(e) = end {
                v.visit_expr(e);
            }
        }
        ExprKind::FString(parts) => {
            for part in parts {
                if let FStringPart::Expr(e) = part {
                    v.visit_expr(e);
                }
            }
        }
        _ => {}
    }
}

fn walk_subscript(v: &mut impl Visitor, kind: &SubscriptKind) {
    match kind {
        SubscriptKind::Index(e) => v.visit_expr(e),
        SubscriptKind::Slice { lower, upper, step } => {
            if let Some(e) = lower {
                v.visit_expr(e);
            }
            if let Some(e) = upper {
                v.visit_expr(e);
            }
            if let Some(e) = step {
                v.visit_expr(e);
            }
        }
        SubscriptKind::MultiDim(dims) => {
            for d in dims {
                walk_subscript(v, d);
            }
        }
    }
}
