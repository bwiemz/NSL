# NSL Dev Tools — Phase 5 Tensor Inspector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `@inspect(tensor, every=N | condition="...")` per `nsl dev tools.pdf` Tool 4 — compile-time decorator emitting on-device stats reduction (cheap) plus optional async full-tensor dump (gated by user-authored predicate), all routed through a dedicated `inspect_stream` so the compute path is uninterrupted.

**Architecture:** Decorator parse/AST already exists. Codegen handler emits two parallel branches: stats path (every=N → `nsl_tensor_stats` reduction kernel + 48-byte async D2H + `.stats.bin` write) and dump path (condition → predicate IR → full async D2H + `.tensor.bin`). Memory planner extension pins inspect-ed tensors past the inspect stream's completion so allocations aren't reused before the memcpy lands. Tiny embedded predicate compiler resolves `step`, `loss`, and four Phase 4 health symbols.

**Tech Stack:** Rust workspace, Cranelift IR for instrumentation, `cudarc` for stream/event APIs, `serde_json` for NSLI headers.

**Spec:** `docs/superpowers/specs/2026-04-13-nsl-dev-tools-phase5-design.md`

**Branch / worktree:** Continue on `feat/dev-tools-phase1` in `c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1`. No new worktree.

---

## Task 1: NSLI binary format

**Files:**
- Create: `crates/nsl-runtime/src/inspect/mod.rs`
- Create: `crates/nsl-runtime/src/inspect/format.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` — add `pub mod inspect;`
- Test: `crates/nsl-runtime/tests/inspect_format.rs` (new)

Foundation task; no other component depends on this except writes. Pure data-format work.

- [ ] **Step 1: Write failing tests**

Create `crates/nsl-runtime/tests/inspect_format.rs`:

```rust
use nsl_runtime::inspect::format::{StatsHeader, FullHeader, write_stats, write_full};

#[test]
fn write_stats_round_trips_header() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let stats: [f64; 6] = [1.5, 0.3, -2.0, 4.5, 0.0, 0.0];
    write_stats(tmp.path(), 100, "h0", &stats).unwrap();
    let bytes = std::fs::read(tmp.path()).unwrap();

    assert_eq!(&bytes[0..4], b"NSLI", "magic");
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    assert_eq!(version, 1);
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    let json_bytes = &bytes[16..16 + header_len];
    let header: StatsHeader = serde_json::from_slice(json_bytes).unwrap();
    assert_eq!(header.step, 100);
    assert_eq!(header.tensor_name, "h0");
    assert!((header.mean - 1.5).abs() < 1e-9);
    assert!((header.std - 0.3).abs() < 1e-9);
    assert_eq!(header.nan_count, 0);
}

#[test]
fn write_full_aligns_data_to_64_bytes() {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let header = FullHeader {
        step: 5,
        tensor_name: "qkv".into(),
        kind: "full".into(),
        dtype: "bf16".into(),
        shape: vec![1, 8, 16],
        stats: StatsHeader {
            step: 5, tensor_name: "qkv".into(), kind: "full".into(),
            mean: 0.0, std: 1.0, min: -1.0, max: 1.0,
            nan_count: 0, inf_count: 0,
        },
    };
    let data: Vec<u8> = (0..256u16).flat_map(|n| n.to_le_bytes()).collect();
    write_full(tmp.path(), &header, &data).unwrap();
    let bytes = std::fs::read(tmp.path()).unwrap();

    assert_eq!(&bytes[0..4], b"NSLI");
    let header_len = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    // Data start = ceil((16 + header_len) / 64) * 64
    let expected_data_start = ((16 + header_len + 63) / 64) * 64;
    assert_eq!(expected_data_start % 64, 0);
    assert_eq!(&bytes[expected_data_start..expected_data_start + data.len()], &data[..]);
}

#[test]
fn write_stats_creates_parent_dirs_implicitly_no_we_dont() {
    // Creating parent dirs is the FFI's responsibility, not format's.
    // This test just confirms write_stats fails cleanly on a missing dir,
    // so the FFI knows to mkdir first.
    let path = std::path::Path::new("/nonexistent/path/foo.bin");
    assert!(write_stats(path, 1, "x", &[0.0; 6]).is_err());
}
```

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-runtime --test inspect_format 2>&1 | tail -10
```

- [ ] **Step 3: Implement format**

Create `crates/nsl-runtime/src/inspect/mod.rs`:

```rust
pub mod format;
```

Create `crates/nsl-runtime/src/inspect/format.rs`:

```rust
//! NSLI binary log format (mirror of NSLM checkpoint).
//!
//! Layout:
//!   [0..4]   magic = b"NSLI"
//!   [4..8]   version: u32 (1)
//!   [8..16]  header_len: u64 (LE)
//!   [16..16+header_len]  JSON header (UTF-8)
//!   [aligned to 64-byte boundary]  raw tensor bytes (full dumps only)

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

const MAGIC: &[u8; 4] = b"NSLI";
const VERSION: u32 = 1;
const ALIGN: u64 = 64;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatsHeader {
    pub step: u64,
    pub tensor_name: String,
    pub kind: String,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub nan_count: u64,
    pub inf_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullHeader {
    pub step: u64,
    pub tensor_name: String,
    pub kind: String,
    pub dtype: String,
    pub shape: Vec<i64>,
    pub stats: StatsHeader,
}

pub fn write_stats(path: &Path, step: u64, name: &str, stats: &[f64]) -> std::io::Result<()> {
    if stats.len() != 6 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput,
            "stats slice must contain 6 values: mean, std, min, max, nan_count, inf_count"));
    }
    let header = StatsHeader {
        step, tensor_name: name.into(), kind: "stats".into(),
        mean: stats[0], std: stats[1], min: stats[2], max: stats[3],
        nan_count: stats[4] as u64, inf_count: stats[5] as u64,
    };
    let json = serde_json::to_vec(&header)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&(json.len() as u64).to_le_bytes())?;
    f.write_all(&json)?;
    Ok(())
}

pub fn write_full(path: &Path, header: &FullHeader, data: &[u8]) -> std::io::Result<()> {
    let json = serde_json::to_vec(header)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let header_len = json.len() as u64;
    let aligned = ((16 + header_len + ALIGN - 1) / ALIGN) * ALIGN;
    let pad = aligned - (16 + header_len);
    let mut f = std::fs::File::create(path)?;
    f.write_all(MAGIC)?;
    f.write_all(&VERSION.to_le_bytes())?;
    f.write_all(&header_len.to_le_bytes())?;
    f.write_all(&json)?;
    f.write_all(&vec![0u8; pad as usize])?;
    f.write_all(data)?;
    Ok(())
}
```

Add `pub mod inspect;` to `crates/nsl-runtime/src/lib.rs`.

- [ ] **Step 4: Run — expect 3 passed**

```
cargo test -p nsl-runtime --test inspect_format 2>&1 | tail -10
cargo build -p nsl-runtime 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```
git add crates/nsl-runtime/src/inspect/ \
        crates/nsl-runtime/src/lib.rs \
        crates/nsl-runtime/tests/inspect_format.rs
git commit -m "feat(runtime): NSLI binary format for inspector dumps"
```

---

## Task 2: Predicate compiler (parser + AST + lowering)

**Files:**
- Create: `crates/nsl-codegen/src/inspect/mod.rs`
- Create: `crates/nsl-codegen/src/inspect/predicate.rs`
- Modify: `crates/nsl-codegen/src/lib.rs` — add `pub mod inspect;`
- Test: `crates/nsl-codegen/tests/inspect_predicate.rs` (new)

Pure parser + AST task. The Cranelift lowering function is sketched but tested separately in Task 7 because it needs a `FunctionBuilder` and registered FuncRefs.

- [ ] **Step 1: Write failing tests for the parser**

Create `crates/nsl-codegen/tests/inspect_predicate.rs`:

```rust
use nsl_codegen::inspect::predicate::{parse_predicate, CmpOp, PredicateExpr};

#[test]
fn parses_simple_int_comparison() {
    let p = parse_predicate("step > 500").unwrap();
    match p {
        PredicateExpr::Cmp(l, op, r) => {
            assert!(matches!(*l, PredicateExpr::Ident(ref s) if s == "step"));
            assert_eq!(op, CmpOp::Gt);
            assert!(matches!(*r, PredicateExpr::IntLit(500)));
        }
        _ => panic!("expected Cmp, got {:?}", p),
    }
}

#[test]
fn parses_float_comparison() {
    let p = parse_predicate("loss > 5.0").unwrap();
    match p {
        PredicateExpr::Cmp(_, op, r) => {
            assert_eq!(op, CmpOp::Gt);
            assert!(matches!(*r, PredicateExpr::FloatLit(v) if (v - 5.0).abs() < 1e-9));
        }
        _ => panic!(),
    }
}

#[test]
fn parses_and() {
    let p = parse_predicate("step > 500 and loss > 5.0").unwrap();
    assert!(matches!(p, PredicateExpr::And(_, _)));
}

#[test]
fn parses_or_with_lower_precedence() {
    // a and b or c => (a and b) or c
    let p = parse_predicate("step > 1 and step > 2 or loss > 0.0").unwrap();
    match p {
        PredicateExpr::Or(l, _) => {
            assert!(matches!(*l, PredicateExpr::And(_, _)));
        }
        _ => panic!("expected Or at root, got {:?}", p),
    }
}

#[test]
fn parses_not_with_highest_precedence() {
    // not a and b => (not a) and b
    let p = parse_predicate("not step > 1 and loss > 0.0").unwrap();
    match p {
        PredicateExpr::And(l, _) => {
            assert!(matches!(*l, PredicateExpr::Not(_)));
        }
        _ => panic!("expected And at root, got {:?}", p),
    }
}

#[test]
fn parses_parens() {
    let p = parse_predicate("not (step > 1 or loss > 0.0)").unwrap();
    assert!(matches!(p, PredicateExpr::Not(_)));
}

#[test]
fn rejects_unknown_identifier() {
    let err = parse_predicate("foo > 1").unwrap_err();
    assert!(err.contains("unknown identifier"), "got {:?}", err);
    assert!(err.contains("foo"), "should mention the bad ident: {:?}", err);
}

#[test]
fn rejects_truncated_expression() {
    assert!(parse_predicate("step >").is_err());
    assert!(parse_predicate("").is_err());
    assert!(parse_predicate("(step > 1").is_err());
}

#[test]
fn accepts_all_comparators() {
    for op in &[">", "<", ">=", "<=", "==", "!="] {
        let src = format!("step {} 1", op);
        assert!(parse_predicate(&src).is_ok(), "should parse {:?}", src);
    }
}

#[test]
fn accepts_all_health_idents() {
    for ident in &["step", "loss", "loss_ema", "loss_ema_slope", "grad_norm_total", "nan_inf_count_window"] {
        let src = format!("{} > 0", ident);
        assert!(parse_predicate(&src).is_ok(), "should parse {:?}", src);
    }
}
```

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-codegen --test inspect_predicate 2>&1 | tail -10
```

- [ ] **Step 3: Implement parser**

Create `crates/nsl-codegen/src/inspect/mod.rs`:

```rust
pub mod predicate;
```

Create `crates/nsl-codegen/src/inspect/predicate.rs`:

```rust
//! Tiny embedded predicate compiler for @inspect(condition="...").
//! Grammar:
//!   expr      = or
//!   or        = and ("or" and)*
//!   and       = not ("and" not)*
//!   not       = "not" not | cmp
//!   cmp       = atom (("<"|">"|"<="|">="|"=="|"!=") atom)?
//!   atom      = ident | int | float | "(" expr ")"
//!   ident     = "step" | "loss" | "loss_ema" | "loss_ema_slope"
//!             | "grad_norm_total" | "nan_inf_count_window"
//!
//! Cranelift lowering: lower_predicate produces an I8 boolean Value.

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
    Op(String),     // ">", "<", ">=", "<=", "==", "!="
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
            // Allow scientific notation: e.g. "1e-3"
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
    fn eat(&mut self) -> Option<Tok> { let t = self.toks.get(self.pos).cloned(); if t.is_some() { self.pos += 1; } t }

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

// Lowering is exercised by Task 7's integration. Stub here:
//
// pub struct PredicateLowerCtx<'a> { ... }
// pub fn lower_predicate(...) -> Value { ... }
//
// The lowering builds a chain of icmp / fcmp instructions and combines them
// with band/bor/bxor for short-circuiting. Returns an I8 (0 or 1).
// Each Ident maps to either a direct Value (step, loss) or a FuncRef call
// (loss_ema, etc.) — see PredicateLowerCtx in the spec §4.2.
```

Add `pub mod inspect;` to `crates/nsl-codegen/src/lib.rs`.

- [ ] **Step 4: Run — expect 9 passed**

```
cargo test -p nsl-codegen --test inspect_predicate 2>&1 | tail -10
cargo build -p nsl-codegen 2>&1 | tail -5
```

- [ ] **Step 5: Commit**

```
git add crates/nsl-codegen/src/inspect/ \
        crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/tests/inspect_predicate.rs
git commit -m "feat(codegen): predicate parser for @inspect condition strings"
```

---

## Task 3: Health getter FFI for predicate idents

**Files:**
- Modify: `crates/nsl-runtime/src/health/ffi.rs` — add four getters.
- Test: `crates/nsl-runtime/tests/health_ffi.rs` (extend).

Phase 4's `HealthCollector` is locked behind a `Mutex`; the predicate compiler will emit FFI calls to read individual fields. Add four small getters that snapshot the fields under one lock.

- [ ] **Step 1: Write failing tests**

Append to `crates/nsl-runtime/tests/health_ffi.rs`:

```rust
use nsl_runtime::health::ffi::{
    nsl_health_get_loss_ema, nsl_health_get_loss_ema_slope,
    nsl_health_get_grad_norm_total, nsl_health_get_nan_inf_count_window,
};

#[test]
fn getters_return_zero_default_then_recorded_values() {
    // After the prior tests pollute global state, just verify shapes.
    nsl_health_record_loss(2.0, 1);
    nsl_health_record_loss(2.5, 2);
    let ema = nsl_health_get_loss_ema();
    assert!(ema > 0.0, "ema should be > 0 after recording, got {}", ema);

    // grad_norm_total: zero when no grad-norm calls happened in this run yet
    // (state is sticky across tests, so we only assert non-negative)
    let g = nsl_health_get_grad_norm_total();
    assert!(g >= 0.0, "grad_norm_total should be >= 0, got {}", g);

    let nan = nsl_health_get_nan_inf_count_window();
    assert!(nan >= 0, "nan count should be >= 0, got {}", nan);
}
```

(All Phase 4 health_ffi tests use `--test-threads=1` because of the global Mutex; new tests follow the same convention.)

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-runtime --test health_ffi getters -- --test-threads=1 2>&1 | tail -10
```

- [ ] **Step 3: Implement getters**

Append to `crates/nsl-runtime/src/health/ffi.rs`:

```rust
#[no_mangle]
pub extern "C" fn nsl_health_get_loss_ema() -> f64 {
    COLLECTOR.lock().unwrap().snapshot().loss_ema.unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_loss_ema_slope() -> f64 {
    COLLECTOR.lock().unwrap().snapshot().loss_ema_slope.unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_grad_norm_total() -> f64 {
    COLLECTOR.lock().unwrap().snapshot().grad_norm_total.unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_nan_inf_count_window() -> i64 {
    COLLECTOR.lock().unwrap().snapshot().nan_inf_count_window as i64
}
```

(Calling `snapshot()` per getter is slightly wasteful — one snapshot copies HashMaps. If profiling later shows this matters, refactor to a `get_field` method that doesn't allocate. For now: snapshot per call, simple and correct.)

- [ ] **Step 4: Run — expect green**

```
cargo test -p nsl-runtime --test health_ffi -- --test-threads=1 2>&1 | tail -10
cargo build -p nsl-runtime 2>&1 | tail -5
```

Expected: 7 passed (6 prior + getters).

- [ ] **Step 5: Commit**

```
git add crates/nsl-runtime/src/health/ffi.rs \
        crates/nsl-runtime/tests/health_ffi.rs
git commit -m "feat(runtime): health getter FFIs for predicate identifier resolution"
```

---

## Task 4: Inspect runtime — stream + stats kernel + record/dump FFIs

**Files:**
- Create: `crates/nsl-runtime/src/inspect/stream.rs` (cuda-gated)
- Create: `crates/nsl-runtime/src/inspect/stats_kernel.rs`
- Create: `crates/nsl-runtime/src/inspect/ffi.rs`
- Modify: `crates/nsl-runtime/src/inspect/mod.rs` — add `pub mod stream; pub mod stats_kernel; pub mod ffi;`
- Test: `crates/nsl-runtime/tests/inspect_ffi.rs` (new)

This is the largest runtime task. Stats kernel is the trickiest piece: implementer can either fuse all six stats in a custom kernel or sequence existing reductions + add a NaN/Inf counter.

- [ ] **Step 1: Write failing tests**

Create `crates/nsl-runtime/tests/inspect_ffi.rs`:

```rust
use nsl_runtime::inspect::ffi::{nsl_inspect_record_stats, nsl_inspect_set_dir};

fn set_tmp_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let bytes = path.as_bytes();
    unsafe { nsl_inspect_set_dir(bytes.as_ptr(), bytes.len()); }
    tmp
}

#[test]
fn record_stats_writes_expected_file_layout() {
    let dir = set_tmp_dir();
    let stats: [f64; 6] = [1.5, 0.3, -2.0, 4.5, 0.0, 0.0];
    let name = "h0";
    let nb = name.as_bytes();
    let rc = unsafe {
        nsl_inspect_record_stats(stats.as_ptr(), 100, nb.as_ptr(), nb.len())
    };
    assert_eq!(rc, 0, "expected 0, got {}", rc);

    let path = dir.path().join("step_100_h0.stats.bin");
    assert!(path.exists(), "expected stats file at {}", path.display());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"NSLI");
}

#[test]
fn record_stats_null_pointer_returns_error() {
    let _dir = set_tmp_dir();
    let rc = unsafe {
        nsl_inspect_record_stats(std::ptr::null(), 1, std::ptr::null(), 0)
    };
    assert_ne!(rc, 0);
}

#[test]
fn record_stats_creates_dir_if_missing() {
    let outer = tempfile::tempdir().unwrap();
    let nested = outer.path().join("a").join("b");
    let pstr = nested.to_str().unwrap();
    let pb = pstr.as_bytes();
    unsafe { nsl_inspect_set_dir(pb.as_ptr(), pb.len()); }
    let stats: [f64; 6] = [0.0; 6];
    let name = "x";
    let nb = name.as_bytes();
    let rc = unsafe { nsl_inspect_record_stats(stats.as_ptr(), 0, nb.as_ptr(), nb.len()) };
    assert_eq!(rc, 0);
    assert!(nested.join("step_0_x.stats.bin").exists());
}
```

The `nsl_tensor_stats` and `nsl_inspect_dump_full` are CUDA-feature tests; defer their integration to Task 9's smoke. For now, write a trivial linkage test:

```rust
#[test]
fn nsl_tensor_stats_symbol_exists() {
    let _f: extern "C" fn(i64, *mut f64) -> i32 = nsl_runtime::inspect::stats_kernel::nsl_tensor_stats;
    // We can't construct a valid tensor handle without scope setup — just confirm linkage.
}

#[test]
fn nsl_inspect_dump_full_symbol_exists() {
    let _f: unsafe extern "C" fn(i64, u64, *const u8, usize) -> i32 = nsl_runtime::inspect::ffi::nsl_inspect_dump_full;
}
```

- [ ] **Step 2: Run — expect fail**

```
cargo test -p nsl-runtime --test inspect_ffi -- --test-threads=1 2>&1 | tail -15
```

- [ ] **Step 3: Implement `stream.rs`**

Create `crates/nsl-runtime/src/inspect/stream.rs`:

```rust
//! Lazy-init dedicated CUstream for inspect copies. Inspect-side memcpys
//! route through this stream so they don't serialize on the compute path.
//!
//! Sync model: the codegen-emitted hook is responsible for calling
//! cuEventRecord on the compute stream (after the producing kernel) and
//! cuStreamWaitEvent on this inspect stream BEFORE issuing the memcpy.

#![cfg(feature = "cuda")]

use cudarc::driver::sys;
use std::cell::RefCell;

thread_local! {
    static INSPECT_STREAM: RefCell<Option<sys::CUstream>> = RefCell::new(None);
}

pub fn current_inspect_stream() -> sys::CUstream {
    INSPECT_STREAM.with(|s| {
        let mut g = s.borrow_mut();
        if g.is_none() {
            let mut stream: sys::CUstream = std::ptr::null_mut();
            unsafe {
                let res = sys::lib().cuStreamCreate(&mut stream, 0);
                if res.0 != 0 { panic!("cuStreamCreate failed: {:?}", res); }
            }
            *g = Some(stream);
        }
        g.unwrap()
    })
}
```

Add `#[cfg(feature = "cuda")] pub mod stream;` to `crates/nsl-runtime/src/inspect/mod.rs`.

- [ ] **Step 4: Implement `stats_kernel.rs`**

Create `crates/nsl-runtime/src/inspect/stats_kernel.rs`:

```rust
//! On-device reduction computing six tensor statistics in one pass.
//! Writes [mean, std, min, max, nan_count, inf_count] as f64 into out_buf.
//!
//! Implementation: For Phase 5 ship-fast policy, this is implemented as a
//! sequence of existing reductions + a new NaN/Inf counter. A future
//! optimization can fuse them into a single PTX kernel.

use crate::tensor::NslTensor;

#[no_mangle]
pub extern "C" fn nsl_tensor_stats(t: i64, out_buf: *mut f64) -> i32 {
    if out_buf.is_null() { return 1; }
    if t == 0 { return 2; }
    let tensor = NslTensor::from_ptr(t);

    // mean
    let mean = match tensor_mean_f64(&tensor) { Ok(v) => v, Err(_) => return 3 };
    // std
    let std = match tensor_std_f64(&tensor, mean) { Ok(v) => v, Err(_) => return 4 };
    // min, max
    let (mn, mx) = match tensor_min_max_f64(&tensor) { Ok(p) => p, Err(_) => return 5 };
    // NaN/Inf counts
    let (nan, inf) = match tensor_nan_inf_counts(&tensor) { Ok(p) => p, Err(_) => return 6 };

    unsafe {
        out_buf.add(0).write(mean);
        out_buf.add(1).write(std);
        out_buf.add(2).write(mn);
        out_buf.add(3).write(mx);
        out_buf.add(4).write(nan as f64);
        out_buf.add(5).write(inf as f64);
    }
    0
}

// Helper functions: implementer wires these against existing tensor reduction
// infrastructure in `crates/nsl-runtime/src/tensor/reduction.rs`.
//
// nsl_tensor_mean and nsl_tensor_min/max already exist (see Phase 5 spec §4.5
// notes from the survey). std and nan/inf counts are new — std can use a
// second reduction pass over (x - mean)^2 then sqrt; nan/inf can use either
// a CPU loop after a host copy (slow but simple) or a new GPU reduction kernel.
//
// Ship the simpler path first: if implementer can reuse reduction.rs to fuse
// passes, do so; otherwise sequence existing primitives. Document choice
// in the task report.

fn tensor_mean_f64(t: &NslTensor) -> Result<f64, &'static str> {
    // Call into crate::tensor::reduction::nsl_tensor_mean or compute on host
    // after copying. For ship-fast: route through whichever existing API
    // returns an f64 mean.
    crate::tensor::reduction::tensor_mean_scalar(t)
        .ok_or("tensor_mean failed")
}

fn tensor_std_f64(t: &NslTensor, mean: f64) -> Result<f64, &'static str> {
    crate::tensor::reduction::tensor_std_scalar(t, mean)
        .ok_or("tensor_std failed")
}

fn tensor_min_max_f64(t: &NslTensor) -> Result<(f64, f64), &'static str> {
    crate::tensor::reduction::tensor_min_max_scalar(t)
        .ok_or("tensor_min_max failed")
}

fn tensor_nan_inf_counts(t: &NslTensor) -> Result<(u64, u64), &'static str> {
    crate::tensor::reduction::tensor_nan_inf_counts(t)
        .ok_or("tensor_nan_inf_counts failed")
}
```

The four helper functions assume the existing reduction module exposes (or can grow) the right scalar accessors. **Implementer must:**
1. Read `crates/nsl-runtime/src/tensor/reduction.rs` to find what's already there.
2. Add any missing scalar reduction functions (likely `tensor_std_scalar`, `tensor_min_max_scalar`, `tensor_nan_inf_counts` are new; `tensor_mean_scalar` may already exist).
3. For NaN/Inf: simplest implementation is a CPU loop after copying tensor data via `tensor.to_host()` or equivalent. GPU reduction is a future optimization.

Add `pub mod stats_kernel;` to `crates/nsl-runtime/src/inspect/mod.rs`.

- [ ] **Step 5: Implement `ffi.rs`**

Create `crates/nsl-runtime/src/inspect/ffi.rs`:

```rust
//! Inspector FFI hooks emitted by codegen at @inspect sites.

use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::path::PathBuf;
use super::format;

static INSPECT_DIR: Lazy<Mutex<PathBuf>> = Lazy::new(|| Mutex::new(PathBuf::from(".nsl-inspect")));

/// # Safety
/// Caller guarantees (path_ptr, path_len) is valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_set_dir(path_ptr: *const u8, path_len: usize) {
    if path_ptr.is_null() { return; }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    if let Ok(s) = std::str::from_utf8(bytes) {
        *INSPECT_DIR.lock().unwrap() = PathBuf::from(s);
    }
}

/// Stats already computed on-device into stats_buf_ptr (6 f64 values).
/// Caller has cuStreamWaitEvent'd the inspect stream onto the kernel
/// completion event before issuing the (logical) async D2H of stats_buf.
///
/// In Phase 5 ship-first, the helper functions in stats_kernel.rs are
/// host-side, so stats_buf is already host-resident. The "async memcpy"
/// is a future optimization once the on-device stats kernel lands.
///
/// # Safety
/// Caller guarantees (stats_buf_ptr, 6 * 8 bytes) is readable and
/// (name_ptr, name_len) is valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_record_stats(
    stats_buf_ptr: *const f64,
    step: u64,
    name_ptr: *const u8, name_len: usize,
) -> i32 {
    if stats_buf_ptr.is_null() || name_ptr.is_null() { return 1; }
    let stats = std::slice::from_raw_parts(stats_buf_ptr, 6);
    let name = match std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len)) {
        Ok(s) => s, Err(_) => return 2,
    };
    let dir = INSPECT_DIR.lock().unwrap().clone();
    if let Err(_) = std::fs::create_dir_all(&dir) { return 3; }
    let path = dir.join(format!("step_{}_{}.stats.bin", step, name));
    match format::write_stats(&path, step, name, stats) {
        Ok(_) => 0, Err(_) => 4,
    }
}

/// Full tensor dump — copies tensor to host and writes raw bytes + JSON header.
///
/// # Safety
/// Caller guarantees tensor_handle is a valid NslTensor pointer and
/// (name_ptr, name_len) is valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_dump_full(
    tensor_handle: i64,
    step: u64,
    name_ptr: *const u8, name_len: usize,
) -> i32 {
    if tensor_handle == 0 || name_ptr.is_null() { return 1; }
    let name = match std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len)) {
        Ok(s) => s, Err(_) => return 2,
    };
    let dir = INSPECT_DIR.lock().unwrap().clone();
    if let Err(_) = std::fs::create_dir_all(&dir) { return 3; }

    let tensor = crate::tensor::NslTensor::from_ptr(tensor_handle);

    // Compute stats inline (small cost compared to full bytes write).
    let mut stats_buf = [0.0f64; 6];
    if super::stats_kernel::nsl_tensor_stats(tensor_handle, stats_buf.as_mut_ptr()) != 0 {
        return 4;
    }

    let dtype_str = format!("{:?}", tensor.dtype);     // adapt if NslTensor.dtype is named differently
    let shape: Vec<i64> = tensor.shape.iter().map(|&n| n as i64).collect();

    // Move tensor to host (sync — async via inspect stream is a future
    // optimization once Phase 5 stats path is on the GPU reduction kernel).
    let host_bytes: Vec<u8> = match tensor.to_host_bytes() {
        Some(b) => b, None => return 5,
    };

    let header = format::FullHeader {
        step, tensor_name: name.into(), kind: "full".into(),
        dtype: dtype_str, shape,
        stats: format::StatsHeader {
            step, tensor_name: name.into(), kind: "full".into(),
            mean: stats_buf[0], std: stats_buf[1],
            min: stats_buf[2], max: stats_buf[3],
            nan_count: stats_buf[4] as u64, inf_count: stats_buf[5] as u64,
        },
    };
    let path = dir.join(format!("step_{}_{}.tensor.bin", step, name));
    match format::write_full(&path, &header, &host_bytes) {
        Ok(_) => 0, Err(_) => 6,
    }
}
```

`tensor.dtype`, `tensor.shape`, `tensor.to_host_bytes()` are placeholders. Implementer adapts to the actual `NslTensor` API — likely:
- `tensor.dtype` is an enum/u8; format with `Debug` or a small `match`.
- `tensor.shape` is a slice/Vec of usize.
- For `to_host_bytes`: there's an existing `nsl_tensor_to_device(t, 0)` host-copy in the runtime. Wrap it to return the raw byte vector.

Add `pub mod ffi;` to `crates/nsl-runtime/src/inspect/mod.rs`.

- [ ] **Step 6: Run tests**

```
cargo test -p nsl-runtime --test inspect_ffi -- --test-threads=1 2>&1 | tail -10
cargo build -p nsl-runtime 2>&1 | tail -5
cargo build -p nsl-runtime --features cuda 2>&1 | tail -5
```

Expected: 5 passed; both feature configs build clean.

- [ ] **Step 7: Commit**

```
git add crates/nsl-runtime/src/inspect/ \
        crates/nsl-runtime/tests/inspect_ffi.rs
git commit -m "feat(runtime): inspector FFIs — stream, stats kernel, record/dump"
```

**Scope guardrails for Task 4:**
- If `tensor_mean_scalar` / `tensor_std_scalar` / `tensor_min_max_scalar` / `tensor_nan_inf_counts` don't exist, **add them** to `crates/nsl-runtime/src/tensor/reduction.rs` with simple host-side implementations (CPU loop). Document in the task report.
- The "async D2H" path is described in the spec but the first-cut implementation can be sync (data is already host-resident after the helper calls).Future PR can add the on-device GPU reduction + true async memcpy.
- DON'T modify any other Phase 1-4 code beyond `health/ffi.rs` (Task 3).

---

## Task 5: Memory planner extension — `pinned_until_inspect_sync`

**Files:**
- Modify: `crates/nsl-codegen/src/wrga_memory.rs` (or wherever `MemoryPlan` lifetimes live)
- Test: `crates/nsl-codegen/tests/inspect_memory_pin.rs` (new)

Without this, FASE memory planner can free an inspect-ed tensor's slot before the inspect stream's memcpy completes, corrupting the dump non-deterministically.

- [ ] **Step 1: Locate planner**

```
grep -n "fn plan_memory\|pub struct MemoryPlanInputs\|pinned\|binary_retention" crates/nsl-codegen/src/wrga_memory.rs crates/nsl-codegen/src/binary_retention.rs 2>/dev/null | head -20
```

Identify:
- The struct that holds inputs to the planner (call it `MemoryPlanInputs` here; adapt to the real name).
- How `binary_retention.rs` extends a tensor's death point (the existing precedent the spec calls out).
- Where `plan_memory` consumes the inputs to compute slot lifetimes.

- [ ] **Step 2: Write failing test**

Create `crates/nsl-codegen/tests/inspect_memory_pin.rs`:

```rust
use std::collections::{BTreeSet, HashSet};
use nsl_codegen::wrga_memory::{plan_memory, MemoryPlan};
// Adapt imports to actual locations and names.

#[test]
fn inspect_pinned_tensor_has_extended_death() {
    // Build a minimal Wengert list with three vars:
    //   v0 produced at pp 0
    //   v1 produced at pp 1, last-used at pp 2
    //   v2 produced at pp 2
    // Without pinning: v1 dies at pp 2.
    // With v1 ∈ pinned_until_inspect_sync: v1 should die at the latest pp + 1.
    let (list, sizes, last_pp) = build_three_var_list();   // implementer writes
    let mut pinned: BTreeSet<_> = BTreeSet::new();
    let v1 = list.var_at(1);
    pinned.insert(v1);

    let plan_unpinned = plan_memory(&list, &Default::default(), &sizes, &Default::default());
    let plan_pinned = nsl_codegen::wrga_memory::plan_memory_with_pin(
        &list, &Default::default(), &sizes, &Default::default(), &pinned,
    );

    let unpinned_v1 = plan_unpinned.assignments.iter().find(|s| s.var == v1).unwrap();
    let pinned_v1 = plan_pinned.assignments.iter().find(|s| s.var == v1).unwrap();

    assert!(pinned_v1.death > unpinned_v1.death,
        "pinned death {} should exceed unpinned death {}", pinned_v1.death, unpinned_v1.death);
    assert!(pinned_v1.death >= last_pp,
        "pinned death {} should reach the end of the program (>= {})", pinned_v1.death, last_pp);
}

// Helper: the smallest valid Wengert+SizeHints inputs we can synthesize.
// Adapt to the actual `WengertList` constructor — check existing tests in
// crates/nsl-codegen/src/ for a fixture pattern.
fn build_three_var_list() -> (/* WengertList */, /* SizeHints */, u32) {
    todo!("build a 3-op Wengert + SizeHints fixture mirroring an existing test")
}
```

If `plan_memory_with_pin` doesn't exist, this test asserts it gets added. Alternative: extend `MemoryPlanInputs` with `pinned: HashSet<VarId>` and the test passes a struct.

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-codegen --test inspect_memory_pin 2>&1 | tail -10
```

- [ ] **Step 4: Implement extension**

Two paths — implementer picks based on existing API shape:

**A. New parallel function `plan_memory_with_pin`:** lowest-disruption add. Existing `plan_memory` callers stay unchanged; new function takes the extra `&BTreeSet<VarId>` and overrides `death` for those vars to `max_pp + 1` (or whatever the planner's "end of step" sentinel is).

**B. Extend `plan_memory` signature:** add `pinned: &BTreeSet<VarId>` as a new parameter, default to empty set in all existing callers. Cleaner long-term but touches every caller.

Pick **A** for ship-fast. If the codebase is small enough that updating callers is trivial, **B** is fine.

In `crates/nsl-codegen/src/wrga_memory.rs`:

```rust
pub fn plan_memory_with_pin(
    list: &WengertList,
    activation_live: &BTreeSet<VarId>,
    sizes: &SizeHints,
    extra_live_at_end: &BTreeSet<VarId>,
    pinned_until_inspect_sync: &BTreeSet<VarId>,
) -> MemoryPlan {
    let mut plan = plan_memory(list, activation_live, sizes, extra_live_at_end);
    let max_pp = plan.assignments.iter().map(|s| s.death).max().unwrap_or(0);
    let pin_target = max_pp.saturating_add(1);
    for slot in &mut plan.assignments {
        if pinned_until_inspect_sync.contains(&slot.var) {
            slot.death = slot.death.max(pin_target);
        }
    }
    plan
}
```

If the implementer prefers reusing `binary_retention.rs`'s pattern instead of a fresh helper, that's also fine — the goal is "extend death past inspect memcpy".

- [ ] **Step 5: Run tests**

```
cargo test -p nsl-codegen --test inspect_memory_pin 2>&1 | tail -10
cargo test -p nsl-codegen --tests 2>&1 | tail -10
```

Expected: green.

- [ ] **Step 6: Commit**

```
git add crates/nsl-codegen/src/wrga_memory.rs \
        crates/nsl-codegen/tests/inspect_memory_pin.rs
git commit -m "feat(codegen): pinned_until_inspect_sync extends tensor death past inspect memcpy"
```

---

## Task 6: Decorator semantic check

**Files:**
- Modify: `crates/nsl-semantic/src/checker/stmt.rs` — add `@inspect` validation arm.
- Test: `crates/nsl-semantic/tests/inspect_decorator.rs` (new)

- [ ] **Step 1: Locate decorator validation**

```
grep -n "fn check_decor\|pub fn .*decorator\|StmtKind::Decor" crates/nsl-semantic/src/checker/stmt.rs | head -10
```

Read the existing `@no_grad` / `@fuse` validation around `stmt.rs:207-235` to mirror the pattern.

- [ ] **Step 2: Write failing tests**

Create `crates/nsl-semantic/tests/inspect_decorator.rs`:

```rust
use nsl_semantic::analyze;
// Adapt to actual semantic-test helpers — see Phase 2/3 inspect tests for the
// parse_and_analyze pattern.

fn analyze_src(src: &str) -> Vec<String> {
    // Lex + parse + analyze; collect diagnostic messages.
    // Use existing test helpers if available, else inline lex+parse.
    todo!("construct an analyzer for the test src; collect diagnostic strings")
}

#[test]
fn at_inspect_with_no_args_is_error() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect()
    return h
"#;
    let errs = analyze_src(src);
    assert!(errs.iter().any(|e| e.contains("inspect") && e.contains("argument")),
        "expected error mentioning inspect args, got {:?}", errs);
}

#[test]
fn at_inspect_with_only_tensor_is_error() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h)
    return h
"#;
    let errs = analyze_src(src);
    assert!(errs.iter().any(|e| e.contains("every") || e.contains("condition")),
        "must specify every= or condition=, got {:?}", errs);
}

#[test]
fn at_inspect_with_every_succeeds() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h, every=10)
    return h
"#;
    let errs = analyze_src(src);
    assert!(errs.is_empty(), "should be clean, got {:?}", errs);
}

#[test]
fn at_inspect_with_condition_succeeds() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h, condition="step > 100")
    return h
"#;
    let errs = analyze_src(src);
    assert!(errs.is_empty(), "should be clean, got {:?}", errs);
}

#[test]
fn at_inspect_with_both_succeeds() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    let h = x
    @inspect(h, every=10, condition="loss > 5.0")
    return h
"#;
    let errs = analyze_src(src);
    assert!(errs.is_empty());
}

#[test]
fn at_inspect_non_tensor_first_arg_errors() {
    let src = r#"
fn forward(x: Tensor<[1, 8, 16], bf16>) -> Tensor:
    @inspect(42, every=10)
    return x
"#;
    let errs = analyze_src(src);
    assert!(errs.iter().any(|e| e.to_lowercase().contains("tensor")),
        "expected error mentioning tensor type, got {:?}", errs);
}
```

If `analyze_src` is hard to write standalone, the implementer can skip these and rely on Task 9's smoke + existing semantic-test infra — semantic checks are quick to verify by hand.

- [ ] **Step 3: Run — expect fail**

```
cargo test -p nsl-semantic --test inspect_decorator 2>&1 | tail -15
```

- [ ] **Step 4: Implement check**

In `crates/nsl-semantic/src/checker/stmt.rs`, in the decorator-handling block, add an arm for `inspect`. Mirror the `@no_grad` validation but check args:

```rust
if dname == "inspect" {
    let args = decor.args.as_ref().ok_or(/* error: missing args */)?;
    if args.is_empty() {
        report_error(decor.span, "@inspect requires at least one argument (the tensor)");
        continue;
    }
    // First positional arg must resolve to a Tensor type.
    let first = &args[0];
    if first.name.is_some() {
        report_error(first.span, "@inspect first argument must be positional (the tensor)");
        continue;
    }
    let ty = self.type_of(&first.value);
    if !matches!(ty, Type::Tensor(_)) {
        report_error(first.span,
            &format!("@inspect first argument must be a Tensor, got {:?}", ty));
        continue;
    }
    // Validate keyword args: at least one of every= / condition=
    let mut has_every = false;
    let mut has_cond = false;
    for arg in &args[1..] {
        let kw = arg.name.as_ref().map(|s| self.resolve_sym(*s));
        match kw.as_deref() {
            Some("every") => {
                has_every = true;
                // Must be a positive integer literal.
                if !matches!(&arg.value.kind, ExprKind::IntLit(n) if *n > 0) {
                    report_error(arg.value.span, "@inspect every= must be a positive integer literal");
                }
            }
            Some("condition") => {
                has_cond = true;
                if !matches!(&arg.value.kind, ExprKind::StringLit(_)) {
                    report_error(arg.value.span, "@inspect condition= must be a string literal");
                }
            }
            Some(other) => {
                report_error(arg.span, &format!("@inspect: unknown keyword {:?}", other));
            }
            None => {
                report_error(arg.span, "@inspect: extra positional argument; only the tensor is positional");
            }
        }
    }
    if !has_every && !has_cond {
        report_error(decor.span, "@inspect: must specify every=N or condition=\"...\" (or both)");
    }
}
```

Adapt `report_error`, `self.type_of`, `self.resolve_sym`, `ExprKind::IntLit`, `ExprKind::StringLit` to the existing semantic API names.

- [ ] **Step 5: Run tests**

```
cargo test -p nsl-semantic --test inspect_decorator 2>&1 | tail -10
cargo test -p nsl-semantic --tests 2>&1 | tail -10
```

Expected: green; no regressions.

- [ ] **Step 6: Commit**

```
git add crates/nsl-semantic/src/checker/stmt.rs \
        crates/nsl-semantic/tests/inspect_decorator.rs
git commit -m "feat(semantic): @inspect arg validation"
```

---

## Task 7: Codegen handler — emit hooks for `@inspect`

**Files:**
- Modify: `crates/nsl-codegen/src/lib.rs` `CompileOptions` — add `inspect_enabled: bool`.
- Modify: `crates/nsl-codegen/src/stmt.rs` (and any sibling let-lowering site) — emit hooks.
- Modify: `crates/nsl-codegen/src/builtins.rs` — register all new FFI symbols.
- Modify: every `CompileOptions { ... }` literal site.
- Modify: `crates/nsl-codegen/src/inspect/predicate.rs` — add `lower_predicate` + `PredicateLowerCtx`.
- Test: `crates/nsl-codegen/tests/inspect_codegen.rs` (new)

Heaviest task. Touches stmt.rs, builtins, CompileOptions literals, and adds the Cranelift lowering for predicates.

- [ ] **Step 1: Add CompileOptions field**

In `crates/nsl-codegen/src/lib.rs::CompileOptions`:

```rust
pub inspect_enabled: bool,
```

Default to `false` in `impl Default`. Update all `CompileOptions { ... }` literals. From the Phase 4 plan we know: 2 in `crates/nsl-cli/src/main.rs` (lines ~852 and ~1042). Find any others:
```
grep -rn "CompileOptions {" crates/
```

- [ ] **Step 2: Register FFI symbols in `builtins.rs`**

Add to `crates/nsl-codegen/src/builtins.rs::RUNTIME_FUNCTIONS` (tuple shape `(&str, &[Type], Option<Type>)` per Phase 3 Task 3 finding):

```rust
("nsl_tensor_stats",                    &[I64, I64], Some(I32)),  // (handle, *mut f64) -> i32
("nsl_inspect_record_stats",            &[I64, I64, I64, I64], Some(I32)), // (stats_ptr, step, name_ptr, name_len) -> i32
("nsl_inspect_dump_full",               &[I64, I64, I64, I64], Some(I32)), // (handle, step, name_ptr, name_len) -> i32
("nsl_inspect_set_dir",                 &[I64, I64], None),
("nsl_health_get_loss_ema",             &[], Some(F64)),
("nsl_health_get_loss_ema_slope",       &[], Some(F64)),
("nsl_health_get_grad_norm_total",      &[], Some(F64)),
("nsl_health_get_nan_inf_count_window", &[], Some(I64)),
```

(Pointers are `I64` per workspace convention.)

- [ ] **Step 3: Add predicate lowering**

In `crates/nsl-codegen/src/inspect/predicate.rs`, append:

```rust
use cranelift_codegen::ir::{types, InstBuilder, IntCC, FloatCC, Value, FuncRef};
use cranelift_frontend::FunctionBuilder;

pub struct PredicateLowerCtx {
    pub step_val: Value,                                // I64
    pub loss_val: Value,                                // F64
    pub get_loss_ema_ref: FuncRef,                       // -> F64
    pub get_loss_ema_slope_ref: FuncRef,                 // -> F64
    pub get_grad_norm_total_ref: FuncRef,                // -> F64
    pub get_nan_inf_count_window_ref: FuncRef,           // -> I64
}

/// Lower a predicate to an I8 boolean Value.
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
            let one = builder.ins().iconst(types::I8, 1);
            builder.ins().bxor(v, one)
        }
        PredicateExpr::Cmp(l, op, r) => {
            // Evaluate both sides, promoting integers to f64 if either side is a float.
            let (lv, rv, is_float) = lower_cmp_operands(l, r, builder, ctx);
            let cc = if is_float {
                let cc = match op {
                    CmpOp::Gt => FloatCC::GreaterThan,
                    CmpOp::Lt => FloatCC::LessThan,
                    CmpOp::Ge => FloatCC::GreaterThanOrEqual,
                    CmpOp::Le => FloatCC::LessThanOrEqual,
                    CmpOp::Eq => FloatCC::Equal,
                    CmpOp::Ne => FloatCC::NotEqual,
                };
                let r = builder.ins().fcmp(cc, lv, rv);
                // Cranelift fcmp returns I8 already.
                r
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
            };
            cc
        }
        // Top-level lits aren't really meaningful — but support them for completeness.
        PredicateExpr::IntLit(n) => {
            // Treat as boolean (0 = false, else true).
            let v = builder.ins().iconst(types::I64, *n);
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().icmp(IntCC::NotEqual, v, zero)
        }
        PredicateExpr::FloatLit(f) => {
            let v = builder.ins().f64const(*f);
            let zero = builder.ins().f64const(0.0);
            builder.ins().fcmp(FloatCC::NotEqual, v, zero)
        }
        PredicateExpr::Ident(_) => {
            // Bare ident at top level — treat as "value != 0".
            let v = lower_ident_to_value(pred, builder, ctx).0;
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().icmp(IntCC::NotEqual, v, zero)
        }
    }
}

/// Lower an Ident or literal to its (Value, is_float) tuple.
fn lower_ident_to_value(
    expr: &PredicateExpr,
    builder: &mut FunctionBuilder,
    ctx: &PredicateLowerCtx,
) -> (Value, bool) {
    match expr {
        PredicateExpr::IntLit(n) => (builder.ins().iconst(types::I64, *n), false),
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
            other => panic!("predicate ident {:?} should have been rejected by parser", other),
        },
        _ => panic!("lower_ident_to_value called on non-ident expr"),
    }
}

fn lower_cmp_operands(
    l: &PredicateExpr,
    r: &PredicateExpr,
    builder: &mut FunctionBuilder,
    ctx: &PredicateLowerCtx,
) -> (Value, Value, bool) {
    let (lv, lf) = lower_ident_to_value(l, builder, ctx);
    let (rv, rf) = lower_ident_to_value(r, builder, ctx);
    let is_float = lf || rf;
    if !is_float { return (lv, rv, false); }
    // Promote any integer side to f64.
    let lv2 = if lf { lv } else { builder.ins().fcvt_from_sint(types::F64, lv) };
    let rv2 = if rf { rv } else { builder.ins().fcvt_from_sint(types::F64, rv) };
    (lv2, rv2, true)
}
```

(Cranelift API names like `band`/`bor`/`bxor`/`fcmp`/`icmp`/`iconst`/`f64const`/`fcvt_from_sint` are correct as of cranelift-codegen 0.x. If the workspace's cranelift version names them differently, adapt — error messages will point at the right names.)

- [ ] **Step 4: Write integration test for predicate lowering**

Create `crates/nsl-codegen/tests/inspect_codegen.rs`:

```rust
//! Integration test: a hand-built FunctionBuilder + lowered predicate
//! produces the expected I8 result. Confirms Cranelift API works end-to-end.

#[test]
fn predicate_lower_step_gt_500_returns_int_when_step_is_1000() {
    // Implementer constructs a small Cranelift function that:
    //   1. Returns an I8 from lower_predicate against a constant step=1000, loss=0.0.
    //   2. JIT-compiles using cranelift-jit (already a workspace dep — check).
    //   3. Invokes and asserts result == 1.
    //
    // If cranelift-jit isn't a dep, skip the live test — verify by inspection
    // that the IR emitted contains the expected icmp instruction:
    //   use cranelift_reader's pretty-printer or just count call instructions.
    //
    // Either way, this test gates the lowering's correctness end-to-end.
    todo!("write a small JIT or IR-inspection test for lower_predicate")
}
```

If JIT or IR inspection is too involved to set up cleanly, document and rely on Task 9's manual smoke. The unit tests for `parse_predicate` (Task 2) catch parser bugs; lowering correctness can be verified by running a real `nsl run --inspect` invocation.

- [ ] **Step 5: Implement codegen handler in `compile_train_block`**

In `crates/nsl-codegen/src/stmt.rs`, when processing a `Stmt::Decorated` for `@inspect`, gated on `self.compile_options.inspect_enabled`:

For each `@inspect(<tensor_ident>, every=N?, condition="..."?)`:

```rust
// Resolve the tensor identifier to its current Cranelift Value.
let tensor_val = self.resolve_tensor_handle(&tensor_ident_arg);

// Mark for memory pinning (Task 5 — pass to MemoryPlanInputs.pinned_until_inspect_sync).
self.inspect_pinned_vars.insert(tensor_var_id);

// Intern the tensor name as static UTF-8.
let name = self.resolve_sym(tensor_ident_sym);
let name_data = self.intern_string(&name);
let name_ptr = builder.ins().symbol_value(I64, name_data);
let name_len = builder.ins().iconst(I64, name.len() as i64);

// Stats branch (every=N).
if let Some(n) = every_n {
    let zero = builder.ins().iconst(I64, 0);
    let step_val = builder.ins().load(I64, MemFlags::trusted(), step_count_var, 0);
    let mod_val = builder.ins().urem_imm(step_val, n as i64);
    let due = builder.ins().icmp(IntCC::Equal, mod_val, zero);
    let do_block = builder.create_block();
    let after_block = builder.create_block();
    builder.ins().brif(due, do_block, &[], after_block, &[]);

    builder.switch_to_block(do_block);
    // Allocate a static 48-byte buffer per inspect site (BSS or .data).
    let stats_buf_data = self.declare_inspect_stats_buf(&name);
    let stats_buf_ptr = builder.ins().symbol_value(I64, stats_buf_data);
    // Compute on-device stats (Phase 5 ship-fast: this is a sync helper that
    // also handles the D2H internally).
    builder.ins().call(self.func_ref("nsl_tensor_stats"), &[tensor_val, stats_buf_ptr]);
    builder.ins().call(
        self.func_ref("nsl_inspect_record_stats"),
        &[stats_buf_ptr, step_val, name_ptr, name_len],
    );
    builder.ins().jump(after_block, &[]);
    builder.seal_block(do_block);
    builder.switch_to_block(after_block);
}

// Dump branch (condition="...").
if let Some(cond_str) = condition {
    let pred_ast = nsl_codegen::inspect::predicate::parse_predicate(&cond_str)
        .map_err(|e| codegen_error(format!("@inspect predicate: {}", e)))?;
    let step_val = builder.ins().load(I64, MemFlags::trusted(), step_count_var, 0);
    let ctx = PredicateLowerCtx {
        step_val,
        loss_val: self.current_loss_val(),  // captured in train block scope
        get_loss_ema_ref: self.func_ref("nsl_health_get_loss_ema"),
        get_loss_ema_slope_ref: self.func_ref("nsl_health_get_loss_ema_slope"),
        get_grad_norm_total_ref: self.func_ref("nsl_health_get_grad_norm_total"),
        get_nan_inf_count_window_ref: self.func_ref("nsl_health_get_nan_inf_count_window"),
    };
    let pred_val = lower_predicate(&pred_ast, builder, &ctx);

    let zero8 = builder.ins().iconst(I8, 0);
    let due = builder.ins().icmp(IntCC::NotEqual, pred_val, zero8);
    let do_block = builder.create_block();
    let after_block = builder.create_block();
    builder.ins().brif(due, do_block, &[], after_block, &[]);

    builder.switch_to_block(do_block);
    builder.ins().call(
        self.func_ref("nsl_inspect_dump_full"),
        &[tensor_val, step_val, name_ptr, name_len],
    );
    builder.ins().jump(after_block, &[]);
    builder.seal_block(do_block);
    builder.switch_to_block(after_block);
}
```

Helpers to add (or locate equivalents):
- `resolve_tensor_handle(arg)` — looks up the variable in the current Cranelift scope.
- `intern_string(s)` — Phase 4 Task 4 used `self.intern_string` from `compiler/mod.rs:706`.
- `declare_inspect_stats_buf(name)` — declares a static 48-byte buffer per inspect site. If codegen has no helper for `.bss` allocations, host-side allocation via a runtime FFI is fine: call `nsl_inspect_alloc_stats_buf(name_ptr, name_len) -> i64` once at fn entry and stash the pointer in a Cranelift slot.
- `inspect_pinned_vars: BTreeSet<VarId>` — new field on Compiler. Pass into `plan_memory_with_pin` from Task 5.
- `current_loss_val()` — Phase 4 already knows where this lives.

When `inspect_enabled == false`, skip the entire emission. Identical IR to no-decorator case.

- [ ] **Step 6: Run codegen tests + smoke**

```
cargo test -p nsl-codegen --tests 2>&1 | tail -10
cargo build -p nsl-codegen -p nsl-cli 2>&1 | tail -5
```

Expected: green.

- [ ] **Step 7: Commit**

```
git add crates/nsl-codegen/src/lib.rs \
        crates/nsl-codegen/src/stmt.rs \
        crates/nsl-codegen/src/builtins.rs \
        crates/nsl-codegen/src/inspect/predicate.rs \
        crates/nsl-codegen/tests/inspect_codegen.rs \
        crates/nsl-cli/src/main.rs
git commit -m "feat(codegen): @inspect decorator emits hooks for stats + dump branches"
```

**Scope guardrails for Task 7:**
- All emission gated on `inspect_enabled` — when false, zero IR difference.
- DON'T touch the runtime helpers (Task 4 done).
- DON'T wire CLI flag (Task 8).
- The "static stats buffer" allocation is the trickiest piece — if codegen has no clean way to declare a per-inspect-site buffer, fall back to a runtime-allocated buffer (one `malloc`-equivalent per inspect site at function entry, freed at function exit). Functionality is identical; just slightly more code.

---

## Task 8: CLI integration — `--inspect` flag + summary

**Files:**
- Modify: `crates/nsl-cli/src/main.rs::Cli::Run` — add `inspect: bool` flag, set CompileOptions, print summary.
- Test: `crates/nsl-cli/tests/inspect_cli.rs` (new, smoke test only)

- [ ] **Step 1: Add the flag**

In `Cli::Run` clap variant:

```rust
#[arg(long)]
inspect: bool,
```

In the dispatch arm, set:

```rust
compile_opts.inspect_enabled = inspect;
```

(Both in the Build- and Run-path `CompileOptions { ... }` literals — Build path stays `false`, Run path uses the flag value.)

- [ ] **Step 2: Print summary at end**

After child process exits, scan the inspect output dir:

```rust
if inspect {
    let dir = std::path::PathBuf::from(".nsl-inspect");
    if let Ok(entries) = std::fs::read_dir(&dir) {
        let mut stats_count = 0;
        let mut full_count = 0;
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.ends_with(".stats.bin") { stats_count += 1; }
            else if name_str.ends_with(".tensor.bin") { full_count += 1; }
        }
        eprintln!("Wrote {} stats records, {} full dumps to {}/",
            stats_count, full_count, dir.display());
    }
}
```

- [ ] **Step 3: Smoke test**

Create `crates/nsl-cli/tests/inspect_cli.rs`:

```rust
use std::process::Command;

#[test]
fn inspect_flag_compiles_clean() {
    // We can't run a real train fixture in this env, but we can confirm the
    // binary accepts --inspect without erroring on unknown flag.
    let bin = env!("CARGO_BIN_EXE_nsl");
    let out = Command::new(bin).args(["run", "--help"]).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("--inspect"), "--inspect flag not in --help output:\n{}", stdout);
}
```

- [ ] **Step 4: Run + commit**

```
cargo test -p nsl-cli --test inspect_cli 2>&1 | tail -10
cargo build -p nsl-cli 2>&1 | tail -5

git add crates/nsl-cli/src/main.rs \
        crates/nsl-cli/tests/inspect_cli.rs
git commit -m "feat(cli): nsl run --inspect activates @inspect hooks + prints summary"
```

---

## Task 9: Final verification

- [ ] **Step 1: Workspace test sweep**

```
cargo test --workspace --tests -- --test-threads=1 2>&1 | grep -E "^test result:|FAILED"
```

Expected: all green. Known Windows e2e file-lock issue avoided by `--test-threads=1`.

- [ ] **Step 2: Counts**

```
cargo test --workspace --tests -- --test-threads=1 2>&1 | grep -E "^test result:" | awk '{sum+=$4} END {print "Total passing: "sum}'
```

Expected: ~2150+ (Phase 4 ended at 2122; Phase 5 adds ~30).

- [ ] **Step 3: Manual acceptance**

If a fixture with `@inspect` exists, run it. Otherwise, write a tiny one:

```
mkdir -p /tmp/p5
cat > /tmp/p5/inspect_smoke.nsl <<'EOF'
fn forward(x: Tensor<[1, 8, 16], bf16>, w: Tensor<[16, 16], bf16>) -> Tensor:
    let h = x @ w
    @inspect(h, every=1)
    return h
EOF

cargo run -p nsl-cli -- check /tmp/p5/inspect_smoke.nsl
cargo run -p nsl-cli -- run --inspect /tmp/p5/inspect_smoke.nsl
ls .nsl-inspect/
```

Expected: at minimum, `nsl check` reports no semantic errors. `nsl run --inspect` may not fully execute without a train block (fixture has none), but it should compile cleanly. If it does run and the kernel fires, expect at least one `step_*_h.stats.bin` file.

- [ ] **Step 4: Branch state**

```
cd c:/Users/bwiem/projects/NSL/.worktrees/dev-tools-phase1
git log --oneline main..HEAD | head -10
```

Expected: 8 new Phase 5 commits on top of Phase 4.

- [ ] **Step 5: Don't push or open PR**

Per user instruction (2026-04-12), held local until all milestones ship. Phase 5 closes the research-document feature set — but the user's hold-local instruction stands until they say otherwise.

---

## Self-Review

**Spec coverage:**

- §4.1 Decorator semantic check → Task 6. ✅
- §4.2 Predicate compiler → Tasks 2 (parser) + 7 (lowering). ✅
- §4.3 Codegen handler → Task 7. ✅
- §4.4 Memory planner extension → Task 5. ✅
- §4.5 Runtime FFI (stream, stats kernel, record/dump, getters) → Tasks 3 + 4. ✅
- §4.6 NSLI binary format → Task 1. ✅
- §4.7 CLI integration → Task 8. ✅
- §6 error handling — bad predicate (Task 2 error tests), null pointers (Task 4 tests), missing tensor type (Task 6 tests), free-then-reuse race (Task 5 prevention). ✅
- §7 testing — every section has at least one task with concrete tests. ✅
- §8 non-goals respected — no live viewer, no abs(), no replay tooling. ✅

**Placeholder scan:**

Three intentional `todo!()` stubs:
- Task 5 Step 2 `build_three_var_list` — implementer mirrors existing Wengert test fixture.
- Task 6 Step 2 `analyze_src` — implementer wires existing semantic-test infra; spec allows skipping these tests if too heavy.
- Task 7 Step 4 `predicate_lower_*` integration test — JIT setup may be too heavy; spec allows fallback to manual smoke.

Each `todo!()` is annotated with the fallback path. No silent placeholders elsewhere.

**Type consistency:**

- `StatsHeader`, `FullHeader`, `write_stats`, `write_full` defined in Task 1, used in Task 4.
- `PredicateExpr`, `CmpOp`, `parse_predicate`, `lower_predicate`, `PredicateLowerCtx` defined in Tasks 2 + 7, used in Task 7.
- `nsl_inspect_record_stats`, `nsl_inspect_dump_full`, `nsl_inspect_set_dir`, `nsl_tensor_stats` defined in Task 4, registered in Task 7 builtins, called from Task 7 emission.
- `nsl_health_get_*` defined in Task 3, registered in Task 7 builtins, called from Task 7 lowering.
- `pinned_until_inspect_sync` / `inspect_pinned_vars` introduced in Task 5, populated in Task 7.
- `CompileOptions.inspect_enabled` introduced in Task 7 Step 1, used by Task 7 emission gates and by Task 8 CLI.

**Scope:**

9 tasks (8 impl + 1 verification). Each produces testable software. Final smoke matches PDF Tool 4 output expectations.
