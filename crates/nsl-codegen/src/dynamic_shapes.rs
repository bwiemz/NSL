//! Dynamic shape codegen helpers for M28.
//!
//! When a function has parameters with symbolic dimensions (e.g. `Tensor<[Batch, SeqLen], f32>`),
//! codegen must:
//! 1. Query the actual dimension values from the first tensor that uses each symbol
//! 2. Assert subsequent uses of the same symbol have matching dimensions
//! 3. For Bounded dims, assert the value is within the upper bound
//!
//! This module tracks which symbolic dims have been "resolved" (first seen) vs need assertion.

use std::collections::HashMap;

use nsl_ast::Symbol;
use nsl_semantic::types::{Dim, Shape};

/// Information about a symbolic dimension found in a function parameter.
pub enum DimInfo {
    /// First occurrence of this symbol — query and record
    Symbolic(Symbol),
    /// Bounded — query, record, and assert within bound
    Bounded { name: Symbol, upper_bound: i64 },
}

/// Tracks symbolic dimension resolution within a function's codegen.
/// Each symbolic name maps to the Cranelift Value holding its resolved runtime i64 dimension value.
pub struct SymbolicDimTracker {
    /// symbol → cranelift Value holding the resolved runtime i64 dimension value
    resolved: HashMap<Symbol, cranelift_codegen::ir::Value>,
}

impl SymbolicDimTracker {
    pub fn new() -> Self {
        Self {
            resolved: HashMap::new(),
        }
    }

    /// Check if a symbolic dim has been resolved (first tensor with this dim was seen).
    pub fn is_resolved(&self, sym: &Symbol) -> bool {
        self.resolved.contains_key(sym)
    }

    /// Record that a symbolic dim has been resolved to the given Cranelift value.
    pub fn resolve(&mut self, sym: Symbol, value: cranelift_codegen::ir::Value) {
        self.resolved.insert(sym, value);
    }

    /// Get the resolved runtime value for a symbolic dim, if available.
    pub fn get(&self, sym: &Symbol) -> Option<cranelift_codegen::ir::Value> {
        self.resolved.get(sym).copied()
    }
}

impl Default for SymbolicDimTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Extract symbolic dims from a shape, yielding (dim_index, DimInfo) pairs.
pub fn extract_symbolic_dims(shape: &Shape) -> Vec<(usize, DimInfo)> {
    let mut result = Vec::new();
    for (i, dim) in shape.dims.iter().enumerate() {
        match dim {
            Dim::Symbolic(sym) => {
                result.push((i, DimInfo::Symbolic(*sym)));
            }
            Dim::Bounded { name, upper_bound } => {
                result.push((
                    i,
                    DimInfo::Bounded {
                        name: *name,
                        upper_bound: *upper_bound,
                    },
                ));
            }
            _ => {}
        }
    }
    result
}
