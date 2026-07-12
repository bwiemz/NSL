//! Profile-capture side channel: real compiler artifacts snapshotted during
//! a full compile for consumption by `nsl profile` (dev-tools paper
//! follow-up: replace the synthetic two-block Wengert and the fixed-lifetime
//! memory timeline with the real train-block extraction).
//!
//! The capture slot is an `Rc<RefCell<..>>` handed to the `Compiler` by
//! `compile_with_profile_captures` so the snapshot SURVIVES downstream
//! codegen errors: `nsl check`-grade minimal compiles routinely fail AFTER
//! train-block extraction (e.g. on unresolved optimizer stdlib symbols —
//! see the note in `commands/check.rs`), and the whole point of the capture
//! is to keep the already-extracted Wengert in exactly that case.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::wengert::{VarId, WengertList};
use crate::wrga_fusion::FusionPlan;

/// Real artifacts captured from the most recent train-block compilation.
#[derive(Debug, Clone, Default)]
pub struct ProfileCaptures {
    /// The real primal WengertList extracted from the train block's step
    /// body (source-AD path). `None` when the module has no train block or
    /// extraction did not run (tape-AD fallback).
    pub train_wengert: Option<WengertList>,
    /// Best-effort per-var payload sizes in bytes, resolved from the typed
    /// AST (`VarId -> NodeId -> Type::Tensor/Param` with all-concrete
    /// dims). Vars with symbolic or unknown shapes are absent — consumers
    /// must treat missing entries as "live but unsized", not zero-cost.
    pub var_size_hints: HashMap<VarId, u64>,
    /// The fusion plan the compile actually produced (WRGA), when any.
    pub fusion: Option<FusionPlan>,
}

/// Shared out-slot: the entry point keeps one `Rc` clone and reads it after
/// the pipeline finishes or errors; the train-block stash site writes it.
pub type ProfileCaptureSlot = Rc<RefCell<Option<ProfileCaptures>>>;

/// Resolve per-var byte sizes from the extractor's `VarId -> NodeId` map and
/// the semantic `TypeMap`.
///
/// Only fully-concrete tensor shapes produce an entry: `Dim::Concrete`,
/// `Dim::Named` wrapping a concrete size, and `Dim::Bounded` (its
/// compile-time upper bound — an honest upper estimate). Symbolic, computed,
/// and wildcard dims make the var unsized. Byte width comes from the
/// DECLARED dtype (the compile-time model), not the runtime CPU-f64/GPU-f32
/// convention.
pub fn size_hints_from_var_nodes(
    var_nodes: &HashMap<VarId, nsl_ast::NodeId>,
    type_map: &nsl_semantic::checker::TypeMap,
) -> HashMap<VarId, u64> {
    use nsl_semantic::types::Type;
    let mut out = HashMap::new();
    for (&var, node) in var_nodes {
        let ty = match type_map.get(node) {
            Some(t) => t,
            None => continue,
        };
        let (shape, dtype) = match ty {
            Type::Tensor { shape, dtype, .. } => (shape, dtype),
            Type::Param { shape, dtype } => (shape, dtype),
            Type::Buffer { shape, dtype } => (shape, dtype),
            _ => continue,
        };
        let mut elems: u64 = 1;
        let mut concrete = !shape.dims.is_empty();
        for dim in &shape.dims {
            match concrete_dim(dim) {
                Some(n) if n > 0 => elems = elems.saturating_mul(n as u64),
                _ => {
                    concrete = false;
                    break;
                }
            }
        }
        if concrete {
            out.insert(var, elems.saturating_mul(dtype_bytes(dtype)));
        }
    }
    out
}

fn concrete_dim(dim: &nsl_semantic::types::Dim) -> Option<i64> {
    use nsl_semantic::types::Dim;
    match dim {
        Dim::Concrete(n) => Some(*n),
        Dim::Named { size, .. } => concrete_dim(size),
        Dim::Bounded { upper_bound, .. } => Some(*upper_bound),
        Dim::Symbolic(_) | Dim::Computed(_) | Dim::Wildcard => None,
    }
}

fn dtype_bytes(dtype: &nsl_semantic::types::DType) -> u64 {
    use nsl_semantic::types::DType;
    match dtype {
        DType::F64 | DType::Int64 => 8,
        DType::F32 | DType::Int32 => 4,
        DType::Fp16 | DType::Bf16 | DType::Int16 => 2,
        DType::Fp8E4m3 | DType::Fp8E5m2 | DType::Int8 | DType::Uint8 | DType::Bool => 1,
        // Sub-byte / packed storage types: bill one byte per element — an
        // over-estimate is safer than a silent zero for a memory timeline.
        _ => 1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_semantic::types::{DType, Device, Dim, Shape, Type};

    fn tensor_ty(dims: Vec<Dim>, dtype: DType) -> Type {
        Type::Tensor {
            shape: Shape { dims },
            dtype,
            device: Device::Cpu,
        }
    }

    #[test]
    fn concrete_shape_produces_byte_size() {
        let mut type_map = nsl_semantic::checker::TypeMap::new();
        let node = nsl_ast::NodeId(7);
        type_map.insert(
            node,
            tensor_ty(vec![Dim::Concrete(4), Dim::Concrete(8)], DType::F32),
        );
        let mut var_nodes = HashMap::new();
        var_nodes.insert(3u32, node);
        let hints = size_hints_from_var_nodes(&var_nodes, &type_map);
        assert_eq!(hints.get(&3), Some(&(4 * 8 * 4)));
    }

    #[test]
    fn non_concrete_dim_is_unsized() {
        let mut type_map = nsl_semantic::checker::TypeMap::new();
        let node = nsl_ast::NodeId(9);
        type_map.insert(
            node,
            tensor_ty(vec![Dim::Wildcard, Dim::Concrete(8)], DType::F64),
        );
        let mut var_nodes = HashMap::new();
        var_nodes.insert(1u32, node);
        let hints = size_hints_from_var_nodes(&var_nodes, &type_map);
        assert!(hints.is_empty());
    }

    #[test]
    fn non_tensor_type_is_skipped() {
        let mut type_map = nsl_semantic::checker::TypeMap::new();
        let node = nsl_ast::NodeId(11);
        type_map.insert(node, Type::Int);
        let mut var_nodes = HashMap::new();
        var_nodes.insert(2u32, node);
        assert!(size_hints_from_var_nodes(&var_nodes, &type_map).is_empty());
    }
}
