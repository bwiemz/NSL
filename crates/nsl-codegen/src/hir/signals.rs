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
    /// M57.1 wire-array realization (Task W5): reads one element of a
    /// `LocalParamArray` declared at module scope. Identical shape to
    /// `WireArrayElement` but lowers as a Verilog localparam read.
    /// `array_name` matches a `LocalParamArray`'s `name`; `indices` must
    /// be `dims.len()` long.
    IndexedLocalParam {
        array_name: String,
        indices: Vec<IndexExpr>,
    },
    /// M57.1 wire-array realization (Task W5): Verilog indexed part select on
    /// a Port (or wire) — emits `{name}[{base_bit} +: {width}]`. Used to
    /// slice the layer-1 `Port::Input("x_l1")` flat bus into per-element
    /// `x_l1_a[k]` WireArray cells.
    PortBitSlice {
        name: String,
        base_bit: usize,
        width: usize,
    },
    /// M57.1 wire-array realization (Task W5): concatenation across an
    /// array's elements — emits Verilog `{name[n-1], ..., name[0]}` (1D) or
    /// `{name[n-1][fix], ..., name[0][fix]}` (2D with a fixed inner index).
    /// Used to drive tap ports + the final-layer `out` port from a WireArray.
    ///
    /// `n` is the number of elements (the outer dim of the array). When
    /// `fixed_index` is `Some(fix)`, the inner index is held at `fix` for
    /// every concat element (used for `acc[*][k_dim]`-style fixed-column
    /// reads).
    WireArrayConcat {
        array_name: String,
        n: usize,
        fixed_index: Option<usize>,
    },
    /// M57.2: combinational read of one element of a `RegArray` (the clocked
    /// sibling of `WireArrayElement`). Used for MAC inputs `x_buf[k]` /
    /// `h_buf[o]`. Renders identically to `WireArrayElement`: `name[idx...]`.
    RegArrayElement {
        array_name: String,
        indices: Vec<IndexExpr>,
    },
    /// M57.2: concatenation across a `RegArray`'s elements — the clocked
    /// sibling of `WireArrayConcat` (no `fixed_index`; 1D arrays only). Emits
    /// `{name[n-1], …, name[0]}` (high-order first per packed-bus convention).
    /// Used to drive the final-layer `out` port from `h_buf`.
    RegArrayConcat {
        array_name: String,
        n: usize,
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
    pub fn indexed_local_param(name: impl Into<String>, indices: Vec<IndexExpr>) -> Self {
        Self::IndexedLocalParam { array_name: name.into(), indices }
    }
    pub fn port_bit_slice(name: impl Into<String>, base_bit: usize, width: usize) -> Self {
        Self::PortBitSlice { name: name.into(), base_bit, width }
    }
    pub fn wire_array_concat(
        name: impl Into<String>, n: usize, fixed_index: Option<usize>,
    ) -> Self {
        Self::WireArrayConcat { array_name: name.into(), n, fixed_index }
    }
    pub fn reg_array_element(name: impl Into<String>, indices: Vec<IndexExpr>) -> Self {
        Self::RegArrayElement { array_name: name.into(), indices }
    }
    pub fn reg_array_concat(name: impl Into<String>, n: usize) -> Self {
        Self::RegArrayConcat { array_name: name.into(), n }
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

    #[test]
    fn signal_ref_reg_array_element_constructs() {
        use crate::hir::ids::RegisterId;
        let s = SignalRef::reg_array_element("x_buf", vec![IndexExpr::Reg(RegisterId(3))]);
        match s {
            SignalRef::RegArrayElement { array_name, indices } => {
                assert_eq!(array_name, "x_buf");
                assert_eq!(indices.len(), 1);
            }
            _ => panic!("expected RegArrayElement"),
        }
    }
}
