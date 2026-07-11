//! AST-scan helpers for the `nsl build` command.
//!
//! Walks a parsed [`Module`] to detect:
//!
//!   * `load_safetensors(<string_literal>, ...)` calls — for CPDT's
//!     Phase 1 AST auto-detect of the weight-file path.
//!   * `@cpdt(..., weight_aware=<bool>)` decorators — for CPDT's Phase 1
//!     weight-aware opt-out.
//!
//! Both walkers are shallow structural recursion, not full type-aware
//! analysis: they don't resolve scopes, imports, or types. Non-literal
//! arguments (e.g. `load_safetensors(PATH_CONSTANT)`) return `None`.
//! Only the first match is returned — multiple `load_safetensors` calls
//! or multiple `@cpdt` decorators are treated as the caller's concern.
//!
//! Design: `docs/superpowers/specs/2026-04-21-cpdt-ast-autodetect-design.md`.

use std::path::PathBuf;

use nsl_ast::block::{TrainBlock, TrainSection};
use nsl_ast::decl::{Decorator, FnDef, ModelDef, ModelMember};
use nsl_ast::expr::{Arg, Expr, ExprKind, SubscriptKind};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::Module;
use nsl_lexer::Interner;

/// Find the first `load_safetensors(<string_literal>, ...)` call anywhere
/// in `module` and return the string-literal's value as a `PathBuf`.
/// Returns `None` when no such call exists, when the first argument
/// isn't a string literal, or when the callee isn't a bare identifier.
pub fn find_ast_weight_ref(module: &Module, interner: &Interner) -> Option<PathBuf> {
    for stmt in &module.stmts {
        if let Some(p) = scan_stmt_for_weight_ref(stmt, interner) {
            return Some(p);
        }
    }
    None
}

/// Find `@cpdt(..., weight_aware=<bool>)` in `module` and return the bool.
/// Returns `None` when no `@cpdt` decorator exists or when the `weight_aware`
/// kwarg is absent. Callers treat `None` as "default" (which is `true` per
/// the semantic pass' `CpdtConfig::weight_aware` default).
pub fn find_ast_cpdt_weight_aware(module: &Module, interner: &Interner) -> Option<bool> {
    for stmt in &module.stmts {
        if let Some(b) = scan_stmt_for_cpdt_weight_aware(stmt, interner) {
            return Some(b);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// load_safetensors scanner
// ---------------------------------------------------------------------------

fn scan_stmt_for_weight_ref(stmt: &Stmt, interner: &Interner) -> Option<PathBuf> {
    match &stmt.kind {
        StmtKind::VarDecl { value, .. } => value
            .as_ref()
            .and_then(|e| scan_expr_for_weight_ref(e, interner)),
        StmtKind::Assign { target, value, .. } => scan_expr_for_weight_ref(target, interner)
            .or_else(|| scan_expr_for_weight_ref(value, interner)),
        StmtKind::Return(Some(e)) => scan_expr_for_weight_ref(e, interner),
        StmtKind::Return(None) => None,
        StmtKind::Expr(e) => scan_expr_for_weight_ref(e, interner),
        StmtKind::If {
            condition,
            then_block,
            elif_clauses,
            else_block,
        } => {
            if let Some(p) = scan_expr_for_weight_ref(condition, interner) {
                return Some(p);
            }
            if let Some(p) = scan_block_for_weight_ref(then_block, interner) {
                return Some(p);
            }
            for (cond, body) in elif_clauses {
                if let Some(p) = scan_expr_for_weight_ref(cond, interner) {
                    return Some(p);
                }
                if let Some(p) = scan_block_for_weight_ref(body, interner) {
                    return Some(p);
                }
            }
            else_block
                .as_ref()
                .and_then(|b| scan_block_for_weight_ref(b, interner))
        }
        StmtKind::While { condition, body, .. } => scan_expr_for_weight_ref(condition, interner)
            .or_else(|| scan_block_for_weight_ref(body, interner)),
        StmtKind::For { iterable, body, .. } => scan_expr_for_weight_ref(iterable, interner)
            .or_else(|| scan_block_for_weight_ref(body, interner)),
        StmtKind::WhileLet { expr, body, .. } => scan_expr_for_weight_ref(expr, interner)
            .or_else(|| scan_block_for_weight_ref(body, interner)),
        StmtKind::Match { subject, arms } => {
            if let Some(p) = scan_expr_for_weight_ref(subject, interner) {
                return Some(p);
            }
            for arm in arms {
                if let Some(p) = scan_block_for_weight_ref(&arm.body, interner) {
                    return Some(p);
                }
            }
            None
        }
        StmtKind::FnDef(f) => scan_fn_def_for_weight_ref(f, interner),
        StmtKind::ModelDef(m) => scan_model_def_for_weight_ref(m, interner),
        StmtKind::TrainBlock(tb) => scan_train_block_for_weight_ref(tb, interner),
        // CPKD: distill blocks can also reference load_safetensors in their
        // config/sections (e.g. teacher weight loading in data: or step()).
        StmtKind::DistillBlock(db) => {
            for arg in db.config.iter().chain(db.loss.iter()) {
                if let Some(p) = scan_expr_for_weight_ref(&arg.value, interner) {
                    return Some(p);
                }
            }
            scan_train_sections_for_weight_ref(&db.sections, interner)
        }
        StmtKind::Decorated { decorators, stmt } => {
            for d in decorators {
                if let Some(args) = &d.args {
                    for arg in args {
                        if let Some(p) = scan_expr_for_weight_ref(&arg.value, interner) {
                            return Some(p);
                        }
                    }
                }
            }
            scan_stmt_for_weight_ref(stmt, interner)
        }
        _ => None,
    }
}

fn scan_block_for_weight_ref(block: &Block, interner: &Interner) -> Option<PathBuf> {
    for s in &block.stmts {
        if let Some(p) = scan_stmt_for_weight_ref(s, interner) {
            return Some(p);
        }
    }
    None
}

fn scan_fn_def_for_weight_ref(f: &FnDef, interner: &Interner) -> Option<PathBuf> {
    scan_block_for_weight_ref(&f.body, interner)
}

fn scan_model_def_for_weight_ref(m: &ModelDef, interner: &Interner) -> Option<PathBuf> {
    for member in &m.members {
        match member {
            ModelMember::LayerDecl { init, .. } => {
                if let Some(e) = init {
                    if let Some(p) = scan_expr_for_weight_ref(e, interner) {
                        return Some(p);
                    }
                }
            }
            ModelMember::Method(f, _decorators) => {
                if let Some(p) = scan_fn_def_for_weight_ref(f, interner) {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn scan_train_block_for_weight_ref(tb: &TrainBlock, interner: &Interner) -> Option<PathBuf> {
    for arg in &tb.config {
        if let Some(p) = scan_expr_for_weight_ref(&arg.value, interner) {
            return Some(p);
        }
    }
    scan_train_sections_for_weight_ref(&tb.sections, interner)
}

fn scan_train_sections_for_weight_ref(
    sections: &[TrainSection],
    interner: &Interner,
) -> Option<PathBuf> {
    for section in sections {
        match section {
            TrainSection::Data(stmts) => {
                for s in stmts {
                    if let Some(p) = scan_stmt_for_weight_ref(s, interner) {
                        return Some(p);
                    }
                }
            }
            TrainSection::Optimizer(e)
            | TrainSection::Scheduler(e)
            | TrainSection::Distribute(e) => {
                if let Some(p) = scan_expr_for_weight_ref(e, interner) {
                    return Some(p);
                }
            }
            TrainSection::Step { body, .. } | TrainSection::Eval { body, .. } => {
                if let Some(p) = scan_block_for_weight_ref(body, interner) {
                    return Some(p);
                }
            }
            TrainSection::Callbacks(callbacks) => {
                for cb in callbacks {
                    if let Some(p) = scan_block_for_weight_ref(&cb.body, interner) {
                        return Some(p);
                    }
                }
            }
            TrainSection::Stmt(s) => {
                if let Some(p) = scan_stmt_for_weight_ref(s, interner) {
                    return Some(p);
                }
            }
        }
    }
    None
}

fn scan_expr_for_weight_ref(expr: &Expr, interner: &Interner) -> Option<PathBuf> {
    if let Some(p) = match_load_safetensors_call(expr, interner) {
        return Some(p);
    }
    // Recurse into sub-expressions for nested calls.
    match &expr.kind {
        ExprKind::Call { callee, args } => {
            if let Some(p) = scan_expr_for_weight_ref(callee, interner) {
                return Some(p);
            }
            for a in args {
                if let Some(p) = scan_expr_for_weight_ref(&a.value, interner) {
                    return Some(p);
                }
            }
            None
        }
        ExprKind::BinaryOp { left, right, .. } => scan_expr_for_weight_ref(left, interner)
            .or_else(|| scan_expr_for_weight_ref(right, interner)),
        ExprKind::UnaryOp { operand, .. } => scan_expr_for_weight_ref(operand, interner),
        ExprKind::Subscript { object, index } => {
            if let Some(p) = scan_expr_for_weight_ref(object, interner) {
                return Some(p);
            }
            scan_subscript_for_weight_ref(index, interner)
        }
        ExprKind::MemberAccess { object, .. } => scan_expr_for_weight_ref(object, interner),
        ExprKind::TupleLiteral(items) | ExprKind::ListLiteral(items) => {
            for e in items {
                if let Some(p) = scan_expr_for_weight_ref(e, interner) {
                    return Some(p);
                }
            }
            None
        }
        ExprKind::DictLiteral(pairs) => {
            for (k, v) in pairs {
                if let Some(p) = scan_expr_for_weight_ref(k, interner) {
                    return Some(p);
                }
                if let Some(p) = scan_expr_for_weight_ref(v, interner) {
                    return Some(p);
                }
            }
            None
        }
        ExprKind::IfExpr {
            condition,
            then_expr,
            else_expr,
        } => scan_expr_for_weight_ref(condition, interner)
            .or_else(|| scan_expr_for_weight_ref(then_expr, interner))
            .or_else(|| scan_expr_for_weight_ref(else_expr, interner)),
        ExprKind::MatchExpr { subject, arms } => {
            if let Some(p) = scan_expr_for_weight_ref(subject, interner) {
                return Some(p);
            }
            for arm in arms {
                if let Some(p) = scan_block_for_weight_ref(&arm.body, interner) {
                    return Some(p);
                }
            }
            None
        }
        ExprKind::Pipe { left, right } => scan_expr_for_weight_ref(left, interner)
            .or_else(|| scan_expr_for_weight_ref(right, interner)),
        ExprKind::Paren(inner) | ExprKind::Await(inner) => {
            scan_expr_for_weight_ref(inner, interner)
        }
        ExprKind::Lambda { body, .. } => scan_expr_for_weight_ref(body, interner),
        ExprKind::BlockExpr(b) => scan_block_for_weight_ref(b, interner),
        _ => None,
    }
}

fn scan_subscript_for_weight_ref(
    index: &SubscriptKind,
    interner: &Interner,
) -> Option<PathBuf> {
    match index {
        SubscriptKind::Index(e) => scan_expr_for_weight_ref(e, interner),
        SubscriptKind::Slice { lower, upper, step } => lower
            .as_ref()
            .and_then(|e| scan_expr_for_weight_ref(e, interner))
            .or_else(|| {
                upper
                    .as_ref()
                    .and_then(|e| scan_expr_for_weight_ref(e, interner))
            })
            .or_else(|| {
                step.as_ref()
                    .and_then(|e| scan_expr_for_weight_ref(e, interner))
            }),
        SubscriptKind::MultiDim(kinds) => {
            for k in kinds {
                if let Some(p) = scan_subscript_for_weight_ref(k, interner) {
                    return Some(p);
                }
            }
            None
        }
    }
}

fn match_load_safetensors_call(expr: &Expr, interner: &Interner) -> Option<PathBuf> {
    let ExprKind::Call { callee, args } = &expr.kind else {
        return None;
    };
    let ExprKind::Ident(sym) = &callee.kind else {
        return None;
    };
    let name = interner.resolve(sym.0)?;
    if name != "load_safetensors" {
        return None;
    }
    first_string_literal_arg(args).map(PathBuf::from)
}

fn first_string_literal_arg(args: &[Arg]) -> Option<String> {
    args.iter().find_map(|a| match &a.value.kind {
        ExprKind::StringLiteral(s) => Some(s.clone()),
        _ => None,
    })
}

// ---------------------------------------------------------------------------
// @cpdt(weight_aware=...) decorator scanner
// ---------------------------------------------------------------------------

fn scan_stmt_for_cpdt_weight_aware(stmt: &Stmt, interner: &Interner) -> Option<bool> {
    match &stmt.kind {
        StmtKind::Decorated { decorators, stmt: inner } => {
            for d in decorators {
                if let Some(b) = extract_cpdt_weight_aware(d, interner) {
                    return Some(b);
                }
            }
            scan_stmt_for_cpdt_weight_aware(inner, interner)
        }
        // Decorators don't appear deep inside blocks in common CPDT use,
        // but recursing into nested containers keeps the walker honest.
        StmtKind::FnDef(f) => scan_block_for_cpdt_weight_aware(&f.body, interner),
        StmtKind::ModelDef(m) => {
            for member in &m.members {
                if let ModelMember::Method(f, _decorators) = member {
                    if let Some(b) = scan_block_for_cpdt_weight_aware(&f.body, interner) {
                        return Some(b);
                    }
                }
            }
            None
        }
        StmtKind::TrainBlock(tb) => {
            for section in &tb.sections {
                if let TrainSection::Stmt(s) = section {
                    if let Some(b) = scan_stmt_for_cpdt_weight_aware(s, interner) {
                        return Some(b);
                    }
                }
            }
            None
        }
        _ => None,
    }
}

fn scan_block_for_cpdt_weight_aware(block: &Block, interner: &Interner) -> Option<bool> {
    for s in &block.stmts {
        if let Some(b) = scan_stmt_for_cpdt_weight_aware(s, interner) {
            return Some(b);
        }
    }
    None
}

fn extract_cpdt_weight_aware(d: &Decorator, interner: &Interner) -> Option<bool> {
    if d.name.len() != 1 {
        return None;
    }
    let dname = interner.resolve(d.name[0].0)?;
    if dname != "cpdt" {
        return None;
    }
    let args = d.args.as_ref()?;
    for arg in args {
        let Some(name_sym) = arg.name else {
            continue;
        };
        let aname = interner.resolve(name_sym.0)?;
        if aname == "weight_aware" {
            if let ExprKind::BoolLiteral(b) = arg.value.kind {
                return Some(b);
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::FileId;

    fn parse(src: &str) -> (Module, Interner) {
        let mut interner = Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
        let parse = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parse.diagnostics.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
            "parse errors in test source: {:?}",
            parse.diagnostics
        );
        (parse.module, interner)
    }

    #[test]
    fn find_weight_ref_in_top_level_let() {
        let (m, ip) = parse(r#"let w = load_safetensors("weights.safetensors")"#);
        assert_eq!(
            find_ast_weight_ref(&m, &ip),
            Some(PathBuf::from("weights.safetensors"))
        );
    }

    #[test]
    fn find_weight_ref_ignores_non_string_literal() {
        let (m, ip) = parse(r#"
let path = "weights.safetensors"
let w = load_safetensors(path)
"#);
        assert_eq!(find_ast_weight_ref(&m, &ip), None);
    }

    #[test]
    fn find_weight_ref_takes_first_match() {
        let (m, ip) = parse(r#"
let w1 = load_safetensors("first.safetensors")
let w2 = load_safetensors("second.safetensors")
"#);
        assert_eq!(
            find_ast_weight_ref(&m, &ip),
            Some(PathBuf::from("first.safetensors"))
        );
    }

    #[test]
    fn find_weight_ref_returns_none_when_absent() {
        let (m, ip) = parse(r#"let x = 1 + 2"#);
        assert_eq!(find_ast_weight_ref(&m, &ip), None);
    }

    /// Minimal train-block source for decorator scanning tests. Mirrors
    /// the shape used in `crates/nsl-cli/tests/cpdt_cli.rs`: a `train(...)`
    /// block on a trivial Linear model.
    const TRAIN_SRC_WRAPPED: &str = r#"from nsl.nn.losses import mse_loss

model Linear:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Linear()
let x = ones([4, 2])
let y = zeros([4, 1])

{DECORATORS}
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
        let loss = mse_loss(pred, y)
"#;

    fn train_src_with_decorators(decos: &str) -> String {
        TRAIN_SRC_WRAPPED.replace("{DECORATORS}", decos)
    }

    #[test]
    fn find_cpdt_weight_aware_false() {
        let src = train_src_with_decorators("@cpdt(weight_aware=false)");
        let (m, ip) = parse(&src);
        assert_eq!(find_ast_cpdt_weight_aware(&m, &ip), Some(false));
    }

    #[test]
    fn find_cpdt_weight_aware_true_when_present() {
        let src = train_src_with_decorators("@cpdt(weight_aware=true)");
        let (m, ip) = parse(&src);
        assert_eq!(find_ast_cpdt_weight_aware(&m, &ip), Some(true));
    }

    #[test]
    fn find_cpdt_weight_aware_none_when_kwarg_absent() {
        // @cpdt with no weight_aware kwarg; walker returns None. The CLI's
        // decision table treats None as "default" (true), matching the
        // semantic pass' CpdtConfig::weight_aware default.
        let src = train_src_with_decorators("@cpdt(mode=full)");
        let (m, ip) = parse(&src);
        assert_eq!(find_ast_cpdt_weight_aware(&m, &ip), None);
    }

    #[test]
    fn find_cpdt_weight_aware_none_when_decorator_absent() {
        let src = train_src_with_decorators("");
        let (m, ip) = parse(&src);
        assert_eq!(find_ast_cpdt_weight_aware(&m, &ip), None);
    }
}
