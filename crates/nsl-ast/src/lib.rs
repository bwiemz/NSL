pub mod block;
pub mod decl;
pub mod expr;
pub mod operator;
pub mod pattern;
pub mod stmt;
pub mod types;
pub mod visitor;

use serde::Serialize;
use std::sync::atomic::{AtomicU32, Ordering};

pub use nsl_errors::Span;

/// Interned string symbol. Wraps string_interner::DefaultSymbol with Serialize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(pub string_interner::DefaultSymbol);

impl Serialize for Symbol {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // Serialize as the raw usize index
        serializer.serialize_u32(string_interner::Symbol::to_usize(self.0) as u32)
    }
}

impl From<string_interner::DefaultSymbol> for Symbol {
    fn from(s: string_interner::DefaultSymbol) -> Self {
        Symbol(s)
    }
}

/// Unique identifier for each AST node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct NodeId(pub u32);

static NEXT_NODE_ID: AtomicU32 = AtomicU32::new(0);

impl NodeId {
    pub fn next() -> Self {
        NodeId(NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed))
    }

    pub fn dummy() -> Self {
        NodeId(u32::MAX)
    }
}

/// The root of an NSL source file.
#[derive(Debug, Clone, Serialize)]
pub struct Module {
    pub stmts: Vec<stmt::Stmt>,
    pub span: Span,
}
