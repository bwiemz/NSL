//! SignalRef — names a signal being *referenced* (RHS of an assignment).

use crate::hir::ids::{RegisterId, WireId};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SignalRef {
    Wire(WireId),
    Register(RegisterId),    // reads the .q output
    Port(String),            // reads an input port by name
    LocalParam(String),      // reads a named constant
}

impl SignalRef {
    pub fn wire(id: WireId) -> Self { Self::Wire(id) }
    pub fn register(id: RegisterId) -> Self { Self::Register(id) }
    pub fn port(name: impl Into<String>) -> Self { Self::Port(name.into()) }
    pub fn local_param(name: impl Into<String>) -> Self { Self::LocalParam(name.into()) }
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
}
