//! A compact S-expression rendering of the AST, for the table tests
//! (`tests/*_table.rs`). The goldens in `tests/parse/` pin every field of
//! a whole file; these tests pin one construct per case, so the rendering
//! keeps only what says how the source was grouped: the tree shape,
//! operators, names and literal values. Spans and node ids are left out.
//!
//! A node kind the printer does not spell out (the training, quantization
//! and serving blocks) prints as its kind name alone, so a table case can
//! still say which statement a line became.

use nsl_ast::Symbol;
use nsl_ast::decl::{Decorator, EffectExpr, FnDef, ImportItems, ModelMember, Param, TypeParam};
use nsl_ast::expr::{Arg, Expr, ExprKind, FStringPart, MatchArm, SubscriptKind};
use nsl_ast::operator::{AssignOp, BinOp, UnaryOp};
use nsl_ast::pattern::{Pattern, PatternKind};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::types::{DeviceExpr, DimExpr, DimValue, TypeExpr, TypeExprKind};
use nsl_errors::{Diagnostic, FileId, Level};
use nsl_lexer::Interner;

/// The outcome of lexing and parsing one source text.
pub struct Parsed {
    /// Every statement of the module, rendered, one per element.
    pub stmts: Vec<String>,
    /// Lexer then parser diagnostics, as `error: <message>` lines.
    pub diagnostics: Vec<String>,
}

pub fn parse(source: &str) -> Parsed {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, FileId(1), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let p = Printer {
        interner: &interner,
    };
    Parsed {
        stmts: parsed.module.stmts.iter().map(|s| p.stmt(s)).collect(),
        diagnostics: lex_diags
            .iter()
            .chain(&parsed.diagnostics)
            .map(render_diagnostic)
            .collect(),
    }
}

fn render_diagnostic(d: &Diagnostic) -> String {
    let level = match d.level {
        Level::Error => "error",
        Level::Warning => "warning",
        Level::Info => "info",
    };
    format!("{level}: {}", d.message)
}

/// Parse `source`, require that it produced no diagnostics, and return the
/// rendering of its statements joined by newlines.
pub fn parse_clean(source: &str) -> String {
    let parsed = parse(source);
    assert!(
        parsed.diagnostics.is_empty(),
        "expected a clean parse of {source:?}, got {:#?}",
        parsed.diagnostics
    );
    parsed.stmts.join("\n")
}

/// Parse `source` as the single expression statement it must be, and
/// return the rendering of that expression.
pub fn expr(source: &str) -> String {
    let rendered = parse_clean(source);
    let inner = rendered
        .strip_prefix("(expr ")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or_else(|| panic!("{source:?} is not one expression statement: {rendered}"));
    inner.to_string()
}

/// Parse `source`, require at least one error diagnostic, and return the
/// diagnostics.
pub fn parse_errors(source: &str) -> Parsed {
    let parsed = parse(source);
    assert!(
        parsed.diagnostics.iter().any(|d| d.starts_with("error: ")),
        "expected {source:?} to be refused, but it parsed as {:#?}",
        parsed.stmts
    );
    parsed
}

struct Printer<'a> {
    interner: &'a Interner,
}

fn join<T>(items: &[T], f: impl Fn(&T) -> String) -> String {
    items.iter().map(f).collect::<Vec<_>>().join(" ")
}

/// `(head a b c)`, or `(head)` when there are no parts.
fn node(head: &str, parts: &[String]) -> String {
    let parts: Vec<&str> = parts
        .iter()
        .map(String::as_str)
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        format!("({head})")
    } else {
        format!("({head} {})", parts.join(" "))
    }
}

impl Printer<'_> {
    fn sym(&self, s: Symbol) -> String {
        self.interner
            .resolve(s.0)
            .unwrap_or("<unresolved>")
            .to_string()
    }

    fn path(&self, path: &[Symbol]) -> String {
        path.iter()
            .map(|s| self.sym(*s))
            .collect::<Vec<_>>()
            .join(".")
    }

    // ---- expressions ----------------------------------------------------

    fn expr(&self, e: &Expr) -> String {
        match &e.kind {
            ExprKind::IntLiteral(v) => v.to_string(),
            ExprKind::FloatLiteral(v) => format!("{v:?}"),
            ExprKind::StringLiteral(s) => format!("{s:?}"),
            ExprKind::FString(parts) => node(
                "f",
                &parts
                    .iter()
                    .map(|p| match p {
                        FStringPart::Text(t) => format!("{t:?}"),
                        FStringPart::Expr(e) => format!("{{{}}}", self.expr(e)),
                    })
                    .collect::<Vec<_>>(),
            ),
            ExprKind::BoolLiteral(b) => b.to_string(),
            ExprKind::NoneLiteral => "none".to_string(),
            ExprKind::ListLiteral(items) => format!("[{}]", join(items, |e| self.expr(e))),
            ExprKind::TupleLiteral(items) => node("tuple", &[join(items, |e| self.expr(e))]),
            ExprKind::DictLiteral(entries) => node(
                "dict",
                &[join(entries, |(k, v)| {
                    format!("{}:{}", self.expr(k), self.expr(v))
                })],
            ),
            ExprKind::Ident(s) => self.sym(*s),
            ExprKind::SelfRef => "self".to_string(),
            ExprKind::BinaryOp { left, op, right } => {
                node(binop(*op), &[self.expr(left), self.expr(right)])
            }
            ExprKind::UnaryOp { op, operand } => {
                let head = match op {
                    UnaryOp::Neg => "neg",
                    UnaryOp::Not => "not",
                };
                node(head, &[self.expr(operand)])
            }
            ExprKind::Pipe { left, right } => node("|>", &[self.expr(left), self.expr(right)]),
            ExprKind::MemberAccess { object, member } => {
                node(".", &[self.expr(object), self.sym(*member)])
            }
            ExprKind::Subscript { object, index } => {
                node("index", &[self.expr(object), self.subscript(index)])
            }
            ExprKind::Call { callee, args } => {
                let mut parts = vec![self.expr(callee)];
                parts.extend(args.iter().map(|a| self.arg(a)));
                node("call", &parts)
            }
            ExprKind::Lambda { params, body } => {
                let ps = params
                    .iter()
                    .map(|p| match &p.type_ann {
                        Some(t) => format!("{}:{}", self.sym(p.name), self.ty(t)),
                        None => self.sym(p.name),
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                node("lambda", &[format!("({ps})"), self.expr(body)])
            }
            ExprKind::BlockExpr(b) => node("block", &[self.block(b)]),
            ExprKind::ListComp {
                element,
                generators,
            } => {
                let mut parts = vec![self.expr(element)];
                for g in generators {
                    let mut generator = vec![self.pat(&g.pattern), self.expr(&g.iterable)];
                    generator.extend(g.conditions.iter().map(|c| node("if", &[self.expr(c)])));
                    parts.push(node("for", &generator));
                }
                node("listcomp", &parts)
            }
            ExprKind::IfExpr {
                condition,
                then_expr,
                else_expr,
            } => node(
                "ifexpr",
                &[
                    self.expr(condition),
                    self.expr(then_expr),
                    self.expr(else_expr),
                ],
            ),
            ExprKind::MatchExpr { subject, arms } => {
                let mut parts = vec![self.expr(subject)];
                parts.extend(arms.iter().map(|a| self.arm(a)));
                node("matchexpr", &parts)
            }
            ExprKind::Range {
                start,
                end,
                inclusive,
            } => node(
                if *inclusive { "..=" } else { ".." },
                &[
                    start.as_ref().map_or("_".to_string(), |e| self.expr(e)),
                    end.as_ref().map_or("_".to_string(), |e| self.expr(e)),
                ],
            ),
            ExprKind::Paren(inner) => node("paren", &[self.expr(inner)]),
            ExprKind::Await(inner) => node("await", &[self.expr(inner)]),
            ExprKind::Error => "<error>".to_string(),
        }
    }

    fn arg(&self, a: &Arg) -> String {
        match a.name {
            Some(name) => format!("{}={}", self.sym(name), self.expr(&a.value)),
            None => self.expr(&a.value),
        }
    }

    fn subscript(&self, s: &SubscriptKind) -> String {
        let opt = |e: &Option<Expr>| e.as_ref().map_or("_".to_string(), |e| self.expr(e));
        match s {
            SubscriptKind::Index(e) => self.expr(e),
            SubscriptKind::Slice { lower, upper, step } => {
                node("slice", &[opt(lower), opt(upper), opt(step)])
            }
            SubscriptKind::MultiDim(dims) => node("dims", &[join(dims, |d| self.subscript(d))]),
        }
    }

    fn arm(&self, a: &MatchArm) -> String {
        let mut parts = vec![self.pat(&a.pattern)];
        if let Some(g) = &a.guard {
            parts.push(node("if", &[self.expr(g)]));
        }
        parts.push(self.block(&a.body));
        node("case", &parts)
    }

    // ---- patterns -------------------------------------------------------

    fn pat(&self, p: &Pattern) -> String {
        match &p.kind {
            PatternKind::Ident(s) => self.sym(*s),
            PatternKind::Wildcard => "_".to_string(),
            PatternKind::Literal(e) => node("lit", &[self.expr(e)]),
            PatternKind::Tuple(ps) => node("ptuple", &[join(ps, |p| self.pat(p))]),
            PatternKind::List(ps) => format!("[{}]", join(ps, |p| self.pat(p))),
            PatternKind::Struct { fields, rest } => {
                let mut parts: Vec<String> = fields
                    .iter()
                    .map(|f| match &f.pattern {
                        Some(p) => format!("{}:{}", self.sym(f.name), self.pat(p)),
                        None => self.sym(f.name),
                    })
                    .collect();
                if let Some(r) = rest {
                    parts.push(format!("..{}", self.sym(*r)));
                }
                node("pstruct", &parts)
            }
            PatternKind::Constructor { path, args } => {
                let mut parts = vec![self.path(path)];
                parts.extend(args.iter().map(|p| self.pat(p)));
                node("ctor", &parts)
            }
            PatternKind::Or(ps) => node("or", &[join(ps, |p| self.pat(p))]),
            PatternKind::Guarded { pattern, guard } => {
                node("guard", &[self.pat(pattern), self.expr(guard)])
            }
            PatternKind::Rest(name) => match name {
                Some(n) => format!("..{}", self.sym(*n)),
                None => "..".to_string(),
            },
            PatternKind::Typed { pattern, type_ann } => {
                node("typed", &[self.pat(pattern), self.ty(type_ann)])
            }
        }
    }

    // ---- types ----------------------------------------------------------

    fn ty(&self, t: &TypeExpr) -> String {
        match &t.kind {
            TypeExprKind::Named(s) => self.sym(*s),
            TypeExprKind::Generic { name, args } => {
                format!("{}<{}>", self.sym(*name), join(args, |t| self.ty(t)))
            }
            TypeExprKind::Tensor {
                shape,
                dtype,
                device,
            } => {
                let mut parts = vec![self.dims(shape), self.sym(*dtype)];
                if let Some(d) = device {
                    parts.push(self.device(d));
                }
                node("Tensor", &parts)
            }
            TypeExprKind::Param { shape, dtype } => {
                node("Param", &[self.dims(shape), self.sym(*dtype)])
            }
            TypeExprKind::Buffer { shape, dtype } => {
                node("Buffer", &[self.dims(shape), self.sym(*dtype)])
            }
            TypeExprKind::Sparse {
                shape,
                dtype,
                format,
            } => node(
                "Sparse",
                &[self.dims(shape), self.sym(*dtype), self.sym(*format)],
            ),
            TypeExprKind::Function {
                params,
                ret,
                effect,
            } => {
                let mut parts = vec![format!("({})", join(params, |t| self.ty(t))), self.ty(ret)];
                if let Some(e) = effect {
                    parts.push(node("effect", &[self.effect(e)]));
                }
                node("fn", &parts)
            }
            TypeExprKind::Union(ts) => node("union", &[join(ts, |t| self.ty(t))]),
            TypeExprKind::Tuple(ts) => node("ttuple", &[join(ts, |t| self.ty(t))]),
            TypeExprKind::Wildcard => "_".to_string(),
            TypeExprKind::FixedArray { element_type, size } => {
                format!("[{}; {size}]", self.ty(element_type))
            }
            TypeExprKind::Borrow(inner) => format!("&{}", self.ty(inner)),
        }
    }

    fn dims(&self, dims: &[DimExpr]) -> String {
        let one = |d: &DimExpr| match d {
            DimExpr::Concrete(n) => n.to_string(),
            DimExpr::Symbolic(s) => self.sym(*s),
            DimExpr::Named { name, value } => match value {
                DimValue::String(v) => format!("{}={v:?}", self.sym(*name)),
                DimValue::Int(v) => format!("{}={v}", self.sym(*name)),
            },
            DimExpr::Bounded { name, upper_bound } => {
                format!("{}<{upper_bound}", self.sym(*name))
            }
            DimExpr::Wildcard => "_".to_string(),
        };
        format!("[{}]", join(dims, one))
    }

    fn device(&self, d: &DeviceExpr) -> String {
        let indexed = |name: &str, i: &Option<Box<Expr>>| match i {
            Some(e) => format!("{name}({})", self.expr(e)),
            None => name.to_string(),
        };
        match d {
            DeviceExpr::Cpu => "cpu".to_string(),
            DeviceExpr::Cuda(i) => indexed("cuda", i),
            DeviceExpr::Metal => "metal".to_string(),
            DeviceExpr::Rocm(i) => indexed("rocm", i),
            DeviceExpr::Npu(s) => format!("npu<{}>", self.sym(*s)),
        }
    }

    fn effect(&self, e: &EffectExpr) -> String {
        match e {
            EffectExpr::Var(s) => format!("var:{}", self.sym(*s)),
            EffectExpr::Named(s) => self.sym(*s),
            EffectExpr::Union(es) => node("union", &[join(es, |e| self.effect(e))]),
        }
    }

    // ---- statements -----------------------------------------------------

    fn block(&self, b: &Block) -> String {
        node("do", &[join(&b.stmts, |s| self.stmt(s))])
    }

    fn type_params(&self, tps: &[TypeParam]) -> String {
        if tps.is_empty() {
            return String::new();
        }
        let one = |tp: &TypeParam| {
            if tp.bounds.is_empty() {
                self.sym(tp.name)
            } else {
                format!("{}:{}", self.sym(tp.name), join(&tp.bounds, |t| self.ty(t)))
            }
        };
        format!("<{}>", join(tps, one))
    }

    fn param(&self, p: &Param) -> String {
        let mut s = String::new();
        if p.is_variadic {
            s.push('*');
        }
        s.push_str(&self.sym(p.name));
        if let Some(t) = &p.type_ann {
            s.push(':');
            s.push_str(&self.ty(t));
        }
        if let Some(d) = &p.default {
            s.push('=');
            s.push_str(&self.expr(d));
        }
        s
    }

    fn fn_def(&self, f: &FnDef) -> String {
        let mut parts = vec![format!(
            "{}{}",
            self.sym(f.name),
            self.type_params(&f.type_params)
        )];
        if f.is_async {
            parts.push("async".to_string());
        }
        parts.push(format!("({})", join(&f.params, |p| self.param(p))));
        if let Some(r) = &f.return_type {
            parts.push(format!("-> {}", self.ty(r)));
        }
        if let Some(e) = &f.return_effect {
            parts.push(node("effect", &[self.effect(e)]));
        }
        parts.push(self.block(&f.body));
        node("fn", &parts)
    }

    fn decorator(&self, d: &Decorator) -> String {
        let name = format!("@{}", self.path(&d.name));
        match &d.args {
            Some(args) => node(&name, &[join(args, |a| self.arg(a))]),
            None => name,
        }
    }

    fn stmt(&self, s: &Stmt) -> String {
        match &s.kind {
            StmtKind::VarDecl {
                is_const,
                pattern,
                type_ann,
                value,
            } => {
                let mut parts = vec![self.pat(pattern)];
                if let Some(t) = type_ann {
                    parts.push(format!(":{}", self.ty(t)));
                }
                if let Some(v) = value {
                    parts.push(self.expr(v));
                }
                node(if *is_const { "const" } else { "let" }, &parts)
            }
            StmtKind::FnDef(f) => self.fn_def(f),
            StmtKind::ModelDef(m) => {
                let mut parts = vec![
                    format!("{}{}", self.sym(m.name), self.type_params(&m.type_params)),
                    format!("({})", join(&m.params, |p| self.param(p))),
                ];
                for member in &m.members {
                    parts.push(match member {
                        ModelMember::LayerDecl {
                            name,
                            type_ann,
                            init,
                            decorators,
                            ..
                        } => {
                            let mut layer = decorators
                                .iter()
                                .map(|d| self.decorator(d))
                                .collect::<Vec<_>>();
                            layer.push(format!("{}:{}", self.sym(*name), self.ty(type_ann)));
                            if let Some(i) = init {
                                layer.push(self.expr(i));
                            }
                            node("layer", &layer)
                        }
                        ModelMember::Method(f, decorators) => {
                            let mut method = decorators
                                .iter()
                                .map(|d| self.decorator(d))
                                .collect::<Vec<_>>();
                            method.push(self.fn_def(f));
                            node("method", &method)
                        }
                    });
                }
                node("model", &parts)
            }
            StmtKind::AgentDef(_) => "(agent)".to_string(),
            StmtKind::StructDef(d) => {
                let mut parts = vec![format!(
                    "{}{}",
                    self.sym(d.name),
                    self.type_params(&d.type_params)
                )];
                parts.extend(d.fields.iter().map(|f| {
                    let mut s = format!("{}:{}", self.sym(f.name), self.ty(&f.type_ann));
                    if let Some(v) = &f.default {
                        s.push('=');
                        s.push_str(&self.expr(v));
                    }
                    s
                }));
                node("struct", &parts)
            }
            StmtKind::EnumDef(d) => {
                let mut parts = vec![format!(
                    "{}{}",
                    self.sym(d.name),
                    self.type_params(&d.type_params)
                )];
                parts.extend(d.variants.iter().map(|v| {
                    let mut s = self.sym(v.name);
                    if !v.fields.is_empty() {
                        s.push_str(&format!("({})", join(&v.fields, |t| self.ty(t))));
                    }
                    if let Some(val) = &v.value {
                        s.push('=');
                        s.push_str(&self.expr(val));
                    }
                    s
                }));
                node("enum", &parts)
            }
            StmtKind::TraitDef(d) => {
                let mut parts = vec![format!(
                    "{}{}",
                    self.sym(d.name),
                    self.type_params(&d.type_params)
                )];
                parts.extend(d.methods.iter().map(|m| self.fn_def(m)));
                node("trait", &parts)
            }
            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                let mut parts = vec![self.expr(condition), self.block(then_block)];
                for (c, b) in elif_clauses {
                    parts.push(node("elif", &[self.expr(c), self.block(b)]));
                }
                if let Some(b) = else_block {
                    parts.push(node("else", &[self.block(b)]));
                }
                node("if", &parts)
            }
            StmtKind::For {
                pattern,
                iterable,
                body,
            } => node(
                "for",
                &[self.pat(pattern), self.expr(iterable), self.block(body)],
            ),
            StmtKind::While { condition, body } => {
                node("while", &[self.expr(condition), self.block(body)])
            }
            StmtKind::WhileLet {
                pattern,
                expr,
                body,
            } => node(
                "whilelet",
                &[self.pat(pattern), self.expr(expr), self.block(body)],
            ),
            StmtKind::Match { subject, arms } => {
                let mut parts = vec![self.expr(subject)];
                parts.extend(arms.iter().map(|a| self.arm(a)));
                node("match", &parts)
            }
            StmtKind::Break => "(break)".to_string(),
            StmtKind::Continue => "(continue)".to_string(),
            StmtKind::Return(v) => node(
                "return",
                &[v.as_ref().map_or(String::new(), |e| self.expr(e))],
            ),
            StmtKind::Yield(v) => node(
                "yield",
                &[v.as_ref().map_or(String::new(), |e| self.expr(e))],
            ),
            StmtKind::Assign { target, op, value } => {
                let head = match op {
                    AssignOp::Assign => "=",
                    AssignOp::AddAssign => "+=",
                    AssignOp::SubAssign => "-=",
                    AssignOp::MulAssign => "*=",
                    AssignOp::DivAssign => "/=",
                };
                node(head, &[self.expr(target), self.expr(value)])
            }
            StmtKind::Import(i) => {
                let mut parts = vec![self.path(&i.path), self.import_items(&i.items)];
                if let Some(a) = i.alias {
                    parts.push(format!("as {}", self.sym(a)));
                }
                node("import", &parts)
            }
            StmtKind::FromImport(i) => node(
                "from",
                &[self.path(&i.module_path), self.import_items(&i.items)],
            ),
            StmtKind::TrainBlock(_) => "(train)".to_string(),
            StmtKind::DistillBlock(_) => "(distill)".to_string(),
            StmtKind::GradBlock(_) => "(grad)".to_string(),
            StmtKind::QuantBlock(_) => "(quant)".to_string(),
            StmtKind::KernelDef(_) => "(kernel)".to_string(),
            StmtKind::TokenizerDef(_) => "(tokenizer)".to_string(),
            StmtKind::DatasetDef(_) => "(dataset)".to_string(),
            StmtKind::DatatypeDef(_) => "(datatype)".to_string(),
            StmtKind::ServeBlock(_) => "(serve)".to_string(),
            StmtKind::Decorated { decorators, stmt } => {
                let mut parts: Vec<String> = decorators.iter().map(|d| self.decorator(d)).collect();
                parts.push(self.stmt(stmt));
                node("decorated", &parts)
            }
            StmtKind::Expr(e) => node("expr", &[self.expr(e)]),
        }
    }

    fn import_items(&self, items: &ImportItems) -> String {
        match items {
            ImportItems::Module => String::new(),
            ImportItems::Glob => "*".to_string(),
            ImportItems::Named(named) => format!(
                "{{{}}}",
                join(named, |i| match i.alias {
                    Some(a) => format!("{} as {}", self.sym(i.name), self.sym(a)),
                    None => self.sym(i.name),
                })
            ),
        }
    }
}

fn binop(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "+",
        BinOp::Sub => "-",
        BinOp::Mul => "*",
        BinOp::Div => "/",
        BinOp::FloorDiv => "//",
        BinOp::Mod => "%",
        BinOp::Pow => "**",
        BinOp::MatMul => "@",
        BinOp::Eq => "==",
        BinOp::NotEq => "!=",
        BinOp::Lt => "<",
        BinOp::Gt => ">",
        BinOp::LtEq => "<=",
        BinOp::GtEq => ">=",
        BinOp::And => "and",
        BinOp::Or => "or",
        BinOp::Is => "is",
        BinOp::In => "in",
        BinOp::BitOr => "|",
        BinOp::BitAnd => "&",
    }
}

/// Parse `source` as the type annotation of `let x: <source> = 0` and
/// return the rendering of the type.
pub fn ty(source: &str) -> String {
    let rendered = parse_clean(&format!("let x: {source} = 0"));
    rendered
        .strip_prefix("(let x :")
        .and_then(|s| s.strip_suffix(" 0)"))
        .unwrap_or_else(|| panic!("{source:?} did not parse as a type: {rendered}"))
        .to_string()
}

/// Parse `source` as the pattern of a one-arm `match` and return the
/// rendering of the pattern (with its guard, if it has one).
pub fn pat(source: &str) -> String {
    let rendered = parse_clean(&format!("match v:\n    case {source}:\n        pass"));
    rendered
        .strip_prefix("(match v (case ")
        .and_then(|s| s.strip_suffix(" (do (expr pass))))"))
        .unwrap_or_else(|| panic!("{source:?} did not parse as a pattern: {rendered}"))
        .to_string()
}

/// One `#[test]` per case: `name: source => expected;`, where the expected
/// value is what `$render(source)` must return.
macro_rules! cases {
    ($render:path; $($name:ident: $src:expr => $want:expr;)*) => {$(
        #[test]
        fn $name() {
            assert_eq!($render($src), $want, "source: {:?}", $src);
        }
    )*};
}

/// Parse `source` + `"\nlet after = 0"`, require at least one error, and
/// return the first diagnostic together with whether `let after = 0` still
/// reached the module as its own top-level statement — i.e. whether the
/// parser recovered before the next line.
pub fn refusal(source: &str) -> (String, bool) {
    let parsed = parse_errors(&format!("{source}\nlet after = 0"));
    let survived = parsed.stmts.iter().any(|s| s == "(let after 0)");
    (parsed.diagnostics[0].clone(), survived)
}
