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
