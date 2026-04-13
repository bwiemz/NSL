//! Tiny embedded predicate compiler for @inspect(condition="...").
//! Grammar:
//!   expr      = or
//!   or        = and ("or" and)*
//!   and       = not ("and" not)*
//!   not       = "not" not | cmp
//!   cmp       = atom (("<"|">"|"<="|">="|"=="|"!=") atom)?
//!   atom      = ident | int | float | "(" expr ")"

#[derive(Debug, Clone, PartialEq)]
pub enum PredicateExpr {
    IntLit(i64),
    FloatLit(f64),
    Ident(String),
    Cmp(Box<PredicateExpr>, CmpOp, Box<PredicateExpr>),
    And(Box<PredicateExpr>, Box<PredicateExpr>),
    Or(Box<PredicateExpr>, Box<PredicateExpr>),
    Not(Box<PredicateExpr>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CmpOp { Gt, Lt, Ge, Le, Eq, Ne }

const ALLOWED_IDENTS: &[&str] = &[
    "step", "loss", "loss_ema", "loss_ema_slope",
    "grad_norm_total", "nan_inf_count_window",
];

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Ident(String),
    Int(i64),
    Float(f64),
    Op(String),
    LParen,
    RParen,
    KwAnd,
    KwOr,
    KwNot,
}

fn tokenize(src: &str) -> Result<Vec<Tok>, String> {
    let mut toks = Vec::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() { i += 1; continue; }
        if b == b'(' { toks.push(Tok::LParen); i += 1; continue; }
        if b == b')' { toks.push(Tok::RParen); i += 1; continue; }
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') { i += 1; }
            let s = std::str::from_utf8(&bytes[start..i]).unwrap();
            match s {
                "and" => toks.push(Tok::KwAnd),
                "or"  => toks.push(Tok::KwOr),
                "not" => toks.push(Tok::KwNot),
                other => toks.push(Tok::Ident(other.to_string())),
            }
            continue;
        }
        if b.is_ascii_digit() || (b == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            if b == b'-' { i += 1; }
            while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') { i += 1; }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') { i += 1; }
                while i < bytes.len() && bytes[i].is_ascii_digit() { i += 1; }
            }
            let s = std::str::from_utf8(&bytes[start..i]).unwrap();
            if s.contains('.') || s.contains('e') || s.contains('E') {
                let v: f64 = s.parse().map_err(|e| format!("bad float {:?}: {}", s, e))?;
                toks.push(Tok::Float(v));
            } else {
                let v: i64 = s.parse().map_err(|e| format!("bad int {:?}: {}", s, e))?;
                toks.push(Tok::Int(v));
            }
            continue;
        }
        if matches!(b, b'<' | b'>' | b'=' | b'!') {
            let start = i;
            i += 1;
            if i < bytes.len() && bytes[i] == b'=' { i += 1; }
            let s = std::str::from_utf8(&bytes[start..i]).unwrap();
            toks.push(Tok::Op(s.to_string()));
            continue;
        }
        return Err(format!("unexpected byte {:?} at offset {}", b as char, i));
    }
    Ok(toks)
}

struct Parser { toks: Vec<Tok>, pos: usize }
impl Parser {
    fn peek(&self) -> Option<&Tok> { self.toks.get(self.pos) }
    fn eat(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() { self.pos += 1; }
        t
    }

    fn parse_expr(&mut self) -> Result<PredicateExpr, String> { self.parse_or() }

    fn parse_or(&mut self) -> Result<PredicateExpr, String> {
        let mut left = self.parse_and()?;
        while matches!(self.peek(), Some(Tok::KwOr)) {
            self.eat();
            let right = self.parse_and()?;
            left = PredicateExpr::Or(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_and(&mut self) -> Result<PredicateExpr, String> {
        let mut left = self.parse_not()?;
        while matches!(self.peek(), Some(Tok::KwAnd)) {
            self.eat();
            let right = self.parse_not()?;
            left = PredicateExpr::And(Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn parse_not(&mut self) -> Result<PredicateExpr, String> {
        if matches!(self.peek(), Some(Tok::KwNot)) {
            self.eat();
            let inner = self.parse_not()?;
            return Ok(PredicateExpr::Not(Box::new(inner)));
        }
        self.parse_cmp()
    }

    fn parse_cmp(&mut self) -> Result<PredicateExpr, String> {
        let left = self.parse_atom()?;
        if let Some(Tok::Op(op_str)) = self.peek().cloned() {
            self.eat();
            let right = self.parse_atom()?;
            let op = match op_str.as_str() {
                ">" => CmpOp::Gt, "<" => CmpOp::Lt,
                ">=" => CmpOp::Ge, "<=" => CmpOp::Le,
                "==" => CmpOp::Eq, "!=" => CmpOp::Ne,
                other => return Err(format!("unknown comparator {:?}", other)),
            };
            return Ok(PredicateExpr::Cmp(Box::new(left), op, Box::new(right)));
        }
        Ok(left)
    }

    fn parse_atom(&mut self) -> Result<PredicateExpr, String> {
        match self.eat() {
            Some(Tok::Int(v)) => Ok(PredicateExpr::IntLit(v)),
            Some(Tok::Float(v)) => Ok(PredicateExpr::FloatLit(v)),
            Some(Tok::Ident(s)) => {
                if !ALLOWED_IDENTS.contains(&s.as_str()) {
                    return Err(format!("unknown identifier {:?} (allowed: {:?})", s, ALLOWED_IDENTS));
                }
                Ok(PredicateExpr::Ident(s))
            }
            Some(Tok::LParen) => {
                let e = self.parse_expr()?;
                match self.eat() {
                    Some(Tok::RParen) => Ok(e),
                    other => Err(format!("expected ), got {:?}", other)),
                }
            }
            Some(other) => Err(format!("unexpected token {:?}", other)),
            None => Err("unexpected end of input".to_string()),
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────
// Predicate lowering (Phase 5 Task 7)
//
// Compiles a parsed `PredicateExpr` into Cranelift IR.  The resulting `Value`
// is a boolean (`I8`) where `1` means "predicate satisfied, take dump branch"
// and `0` means "skip dump".
// ──────────────────────────────────────────────────────────────────────────

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types as cl_types, FuncRef, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

/// Shared context threaded through predicate lowering.  Callers construct
/// this once per `@inspect` site and pass it through the recursion.
pub struct PredicateLowerCtx {
    /// Current train-loop step counter loaded as an I64 value.
    pub step_val: Value,
    /// Current scalar loss (F64).  When the compiler cannot resolve `loss`
    /// in scope (e.g. `@inspect` fires before the loss is computed), this is
    /// a compile-time zero constant — predicates referencing `loss` degrade
    /// to `false` / `0.0` rather than erroring.
    pub loss_val: Value,
    pub get_loss_ema_ref: FuncRef,
    pub get_loss_ema_slope_ref: FuncRef,
    pub get_grad_norm_total_ref: FuncRef,
    pub get_nan_inf_count_window_ref: FuncRef,
}

/// Lower a predicate AST into an I8 boolean value (0 or 1).
pub fn lower_predicate(
    pred: &PredicateExpr,
    builder: &mut FunctionBuilder,
    ctx: &PredicateLowerCtx,
) -> Value {
    match pred {
        PredicateExpr::And(l, r) => {
            let lv = lower_predicate(l, builder, ctx);
            let rv = lower_predicate(r, builder, ctx);
            builder.ins().band(lv, rv)
        }
        PredicateExpr::Or(l, r) => {
            let lv = lower_predicate(l, builder, ctx);
            let rv = lower_predicate(r, builder, ctx);
            builder.ins().bor(lv, rv)
        }
        PredicateExpr::Not(inner) => {
            let v = lower_predicate(inner, builder, ctx);
            let one = builder.ins().iconst(cl_types::I8, 1);
            builder.ins().bxor(v, one)
        }
        PredicateExpr::Cmp(l, op, r) => {
            let (lv, rv, is_float) = lower_cmp_operands(l, r, builder, ctx);
            if is_float {
                let cc = match op {
                    CmpOp::Gt => FloatCC::GreaterThan,
                    CmpOp::Lt => FloatCC::LessThan,
                    CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    CmpOp::Le => FloatCC::LessThanOrEqual,
                    CmpOp::Eq => FloatCC::Equal,
                    CmpOp::Ne => FloatCC::NotEqual,
                };
                builder.ins().fcmp(cc, lv, rv)
            } else {
                let cc = match op {
                    CmpOp::Gt => IntCC::SignedGreaterThan,
                    CmpOp::Lt => IntCC::SignedLessThan,
                    CmpOp::Ge => IntCC::SignedGreaterThanOrEqual,
                    CmpOp::Le => IntCC::SignedLessThanOrEqual,
                    CmpOp::Eq => IntCC::Equal,
                    CmpOp::Ne => IntCC::NotEqual,
                };
                builder.ins().icmp(cc, lv, rv)
            }
        }
        // Bare literals / idents as predicates — "truthy" test.
        PredicateExpr::IntLit(n) => {
            let v = builder.ins().iconst(cl_types::I64, *n);
            let zero = builder.ins().iconst(cl_types::I64, 0);
            builder.ins().icmp(IntCC::NotEqual, v, zero)
        }
        PredicateExpr::FloatLit(f) => {
            let v = builder.ins().f64const(*f);
            let zero = builder.ins().f64const(0.0);
            builder.ins().fcmp(FloatCC::NotEqual, v, zero)
        }
        PredicateExpr::Ident(_) => {
            let (v, is_float) = lower_ident_to_value(pred, builder, ctx);
            if is_float {
                let zero = builder.ins().f64const(0.0);
                builder.ins().fcmp(FloatCC::NotEqual, v, zero)
            } else {
                let zero = builder.ins().iconst(cl_types::I64, 0);
                builder.ins().icmp(IntCC::NotEqual, v, zero)
            }
        }
    }
}

/// Resolve an ident/literal atom to its runtime `Value`.  Returns `(value,
/// is_float)` — the caller may need to coerce to a common type.
fn lower_ident_to_value(
    expr: &PredicateExpr,
    builder: &mut FunctionBuilder,
    ctx: &PredicateLowerCtx,
) -> (Value, bool) {
    match expr {
        PredicateExpr::IntLit(n) => (builder.ins().iconst(cl_types::I64, *n), false),
        PredicateExpr::FloatLit(f) => (builder.ins().f64const(*f), true),
        PredicateExpr::Ident(s) => match s.as_str() {
            "step" => (ctx.step_val, false),
            "loss" => (ctx.loss_val, true),
            "loss_ema" => {
                let inst = builder.ins().call(ctx.get_loss_ema_ref, &[]);
                (builder.inst_results(inst)[0], true)
            }
            "loss_ema_slope" => {
                let inst = builder.ins().call(ctx.get_loss_ema_slope_ref, &[]);
                (builder.inst_results(inst)[0], true)
            }
            "grad_norm_total" => {
                let inst = builder.ins().call(ctx.get_grad_norm_total_ref, &[]);
                (builder.inst_results(inst)[0], true)
            }
            "nan_inf_count_window" => {
                let inst = builder.ins().call(ctx.get_nan_inf_count_window_ref, &[]);
                (builder.inst_results(inst)[0], false)
            }
            other => panic!(
                "predicate ident {:?} should have been rejected by parse_predicate",
                other
            ),
        },
        _ => panic!("lower_ident_to_value called on non-atom predicate expr"),
    }
}

/// Lower both operands of a `Cmp`, coercing to a common numeric type.
/// Mixed int/float comparisons widen the int side to f64.
fn lower_cmp_operands(
    l: &PredicateExpr,
    r: &PredicateExpr,
    builder: &mut FunctionBuilder,
    ctx: &PredicateLowerCtx,
) -> (Value, Value, bool) {
    let (lv, lf) = lower_ident_to_value(l, builder, ctx);
    let (rv, rf) = lower_ident_to_value(r, builder, ctx);
    let is_float = lf || rf;
    if !is_float {
        return (lv, rv, false);
    }
    let lv2 = if lf {
        lv
    } else {
        builder.ins().fcvt_from_sint(cl_types::F64, lv)
    };
    let rv2 = if rf {
        rv
    } else {
        builder.ins().fcvt_from_sint(cl_types::F64, rv)
    };
    (lv2, rv2, true)
}

pub fn parse_predicate(src: &str) -> Result<PredicateExpr, String> {
    if src.trim().is_empty() { return Err("empty predicate".to_string()); }
    let toks = tokenize(src)?;
    if toks.is_empty() { return Err("empty token stream".to_string()); }
    let mut p = Parser { toks, pos: 0 };
    let e = p.parse_expr()?;
    if p.pos != p.toks.len() {
        return Err(format!("unexpected trailing tokens at position {}", p.pos));
    }
    Ok(e)
}
