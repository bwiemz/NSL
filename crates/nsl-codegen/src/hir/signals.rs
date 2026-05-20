//! SignalRef — names a signal being *referenced* (RHS of an assignment).

use crate::hir::ids::{RegisterId, WireId};
use crate::hir::nodes::IndexExpr;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalRef {
    Wire(WireId),
    Register(RegisterId),    // reads the .q output
    Port(String),            // reads an input port by name
    LocalParam(String),      // reads a named constant
    /// M57.1 wire-array realization: reads one element of a `WireArray`
    /// declared at module scope. `array_name` matches a `HirNode::WireArray`'s
    /// `name`; `indices` must be `dims.len()` long.
    WireArrayElement {
        array_name: String,
        indices: Vec<IndexExpr>,
    },
}

impl SignalRef {
    pub fn wire(id: WireId) -> Self { Self::Wire(id) }
    pub fn register(id: RegisterId) -> Self { Self::Register(id) }
    pub fn port(name: impl Into<String>) -> Self { Self::Port(name.into()) }
    pub fn local_param(name: impl Into<String>) -> Self { Self::LocalParam(name.into()) }
    pub fn wire_array_element(name: impl Into<String>, indices: Vec<IndexExpr>) -> Self {
        Self::WireArrayElement { array_name: name.into(), indices }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signal_ref_variants_compare_correctly() {
        let w = WireId::fresh();
        assert_eq!(SignalRef::wire(w), SignalRef::Wire(w));
        assert_ne!(SignalRef::port("clk"), SignalRef::port("rst"));
    }

    #[test]
    fn signal_ref_wire_array_element_constructs() {
        let s = SignalRef::wire_array_element("acc_l1", vec![
            IndexExpr::Genvar("_gv_o".to_string()),
            IndexExpr::GenvarPlus("_gv_k".to_string(), 1),
        ]);
        match s {
            SignalRef::WireArrayElement { array_name, indices } => {
                assert_eq!(array_name, "acc_l1");
                assert_eq!(indices.len(), 2);
            }
            _ => panic!("expected WireArrayElement"),
        }
    }
}
