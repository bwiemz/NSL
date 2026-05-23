//! HirModule — root container for HIR. Carries clock/reset domain name tables
//! (BTreeMap for deterministic iteration), test_taps flag, and the SSA body
//! of nodes. Builder enforces §3.4 invariant 1 (single-driver per signal) at
//! the point of `add_node`.

use std::collections::{BTreeMap, HashSet};

use crate::hir::control::{GenerateFor, GenerateIf};
use crate::hir::ids::{ClockDomainId, ResetSignalId, WireId};
use crate::hir::nodes::*;

#[derive(Debug, Clone, PartialEq)]
pub enum HirNode {
    Wire(Wire),
    Register(Register),
    Mul(Mul),
    Add(Add),
    Max0(Max0),
    SignExtend(SignExtend),
    GenerateFor(GenerateFor),
    GenerateIf(GenerateIf),
    /// M57.1 wire-array realization: module-scope multi-dimensional wire
    /// array. Declared once per array; referenced by element via
    /// `SignalRef::WireArrayElement`. Does not produce a single WireId
    /// (it declares an N-dimensional family of wires whose drivers are
    /// `assign`s inside generate blocks).
    WireArray(WireArray),
    /// M57.1 wire-array realization (Task W4): driver for one element of a
    /// WireArray. Inside a GenerateFor body, indices typically include Genvar
    /// refs; at module top level, indices are typically Literals. Does NOT
    /// produce a single WireId — it drives a single element of an
    /// N-dimensional wire family declared by `WireArray`.
    AssignWireArrayElement(AssignWireArrayElement),
    /// M57.2: combinational equality against a compile-time constant (Task 3).
    /// Produces a 1-bit wire: `assign _w{out} = ($signed({lhs}) == {rhs});`.
    CmpEq(CmpEq),
}

impl HirNode {
    /// Returns the WireId this node produces, if any. Used for single-driver
    /// invariant enforcement.
    fn produces_wire(&self) -> Option<WireId> {
        match self {
            HirNode::Wire(w) => Some(w.id),
            HirNode::Mul(m) => Some(m.out),
            HirNode::Add(a) => Some(a.out),
            HirNode::Max0(m) => Some(m.out),
            HirNode::SignExtend(s) => Some(s.dst),
            // M57.2 (Task 3): combinational equality drives one wire.
            HirNode::CmpEq(c) => Some(c.out),
            // Register, GenerateFor, GenerateIf, WireArray, AssignWireArrayElement
            // don't produce a single WireId. (WireArray declares multi-element
            // wire family; AssignWireArrayElement drives ONE element of one;
            // single-driver invariant for each element is the caller's
            // responsibility, same as GenerateFor body uniqueness.)
            HirNode::Register(_) => None,
            HirNode::GenerateFor(_) => None,
            HirNode::GenerateIf(_) => None,
            HirNode::WireArray(_) => None,
            HirNode::AssignWireArrayElement(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HirModule {
    pub name: String,
    pub ports: Vec<Port>,
    /// pub(crate) — readers access via `nodes()`; writers MUST use `add_node`
    /// so the single-driver invariant (spec §3.4 invariant 1) is enforced.
    pub(crate) bodies: Vec<HirNode>,
    /// pub(crate) — readers access via `local_params()`; writers go through
    /// `add_local_param`. Task 4.1's CLI-time bake_fixture_into_localparams
    /// will need mutable access — added then via a dedicated accessor.
    /// (M57.1 Task 3.2 review follow-up.)
    pub(crate) local_params: Vec<LocalParam>,
    /// M57.1 wire-array mini §3.2 / Task W5: module-scope multi-dimensional
    /// const arrays (`W<i>`, `b<i>`). Accessed via `local_param_arrays()`
    /// (read) / `local_param_arrays_mut()` (Task W6 CLI baking) /
    /// `add_local_param_array` (construction).
    pub(crate) local_param_arrays: Vec<LocalParamArray>,
    /// BTreeMap (not HashMap) for deterministic iteration order across Rust
    /// versions and platforms — Layer 1 Verilog emission snapshots depend on
    /// reproducible output (spec §3.5 / §3 issue 5).
    pub clock_domains: BTreeMap<ClockDomainId, String>,
    pub reset_signals: BTreeMap<ResetSignalId, String>,
    pub test_taps: bool,

    // Builder state — used by add_node to enforce single-driver invariant.
    driven_wires: HashSet<WireId>,
}

#[derive(Debug, thiserror::Error)]
pub enum HirBuilderError {
    #[error("WireId {0:?} already has a driver in module '{1}'; single-driver \
             invariant (spec §3.4 invariant 1) violated")]
    DuplicateDriver(WireId, String),
}

impl HirModule {
    pub fn new(name: impl Into<String>) -> Self {
        let mut clock_domains = BTreeMap::new();
        clock_domains.insert(ClockDomainId::DEFAULT, "clk".to_string());
        let mut reset_signals = BTreeMap::new();
        reset_signals.insert(ResetSignalId::DEFAULT, "rst".to_string());

        Self {
            name: name.into(),
            ports: Vec::new(),
            bodies: Vec::new(),
            local_params: Vec::new(),
            local_param_arrays: Vec::new(),
            clock_domains,
            reset_signals,
            test_taps: true,    // v1 default
            driven_wires: HashSet::new(),
        }
    }

    pub fn add_port(&mut self, port: Port) {
        self.ports.push(port);
    }

    pub fn add_local_param(&mut self, lp: LocalParam) {
        self.local_params.push(lp);
    }

    /// Add a node to the module body. Enforces single-driver invariant: if
    /// the node produces a WireId already driven by another node, returns
    /// DuplicateDriver error. (Generate-loop bodies are NOT recursively
    /// checked at top-level — the builder treats GenerateFor as opaque; its
    /// internal driver-uniqueness is the caller's responsibility, per the
    /// generate-loop "fresh genvar each iteration" Verilog semantics.)
    pub fn add_node(&mut self, node: HirNode) -> Result<(), HirBuilderError> {
        if let Some(wid) = node.produces_wire() {
            if !self.driven_wires.insert(wid) {
                return Err(HirBuilderError::DuplicateDriver(wid, self.name.clone()));
            }
        }
        self.bodies.push(node);
        Ok(())
    }

    /// Read-only access to the SSA node list. Construction must go through
    /// `add_node` so the single-driver invariant holds.
    pub fn nodes(&self) -> &[HirNode] {
        &self.bodies
    }

    /// Read-only access to declared LocalParams. Construction must go through
    /// `add_local_param`. (M57.1 Task 3.2 review follow-up: field tightened
    /// to `pub(crate)`; this is the public reader.)
    pub fn local_params(&self) -> &[LocalParam] {
        &self.local_params
    }

    /// Mutable access to declared LocalParams for CLI-time value baking.
    ///
    /// Task 4.1 (`nsl fpga-compile`) walks the FixtureFile's blocks and writes
    /// the parsed weight/bias values into the matching `W<i>_<k>_<o>` and
    /// `b<i>_<o>` LocalParams by name. This is the only sanctioned mutator —
    /// `add_local_param` is the construction-time entry point and is preserved
    /// for the KirToHirPass; this accessor is for post-construction value
    /// substitution only. Callers must not push/remove entries through it.
    pub fn local_params_mut(&mut self) -> &mut Vec<LocalParam> {
        &mut self.local_params
    }

    /// M57.1 wire-array mini §3.2 / Task W5: register a LocalParamArray
    /// (the fused-emitter shape for `W<i>` / `b<i>`). The fused matmul
    /// ripple body reads it via `SignalRef::IndexedLocalParam`.
    pub fn add_local_param_array(&mut self, lpa: LocalParamArray) {
        self.local_param_arrays.push(lpa);
    }

    /// Read-only access to declared LocalParamArrays. Construction via
    /// `add_local_param_array`. Renders in the LocalParam preamble block of
    /// the emitted Verilog.
    pub fn local_param_arrays(&self) -> &[LocalParamArray] {
        &self.local_param_arrays
    }

    /// Mutable access to declared LocalParamArrays for CLI-time value baking
    /// (Task W6). The fixture loader writes parsed weight/bias values into
    /// the `values` field by `[k][o]` (or `[o]`) index. Callers must not
    /// push/remove entries through it.
    pub fn local_param_arrays_mut(&mut self) -> &mut Vec<LocalParamArray> {
        &mut self.local_param_arrays
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hir::signals::SignalRef;

    #[test]
    fn empty_module_has_default_clock_and_reset_names() {
        let m = HirModule::new("test");
        assert_eq!(m.clock_domains.get(&ClockDomainId::DEFAULT), Some(&"clk".to_string()));
        assert_eq!(m.reset_signals.get(&ResetSignalId::DEFAULT), Some(&"rst".to_string()));
    }

    #[test]
    fn add_node_accepts_unique_wire_drivers() {
        let mut m = HirModule::new("test");
        let mul = Mul::new(SignalRef::port("a"), SignalRef::port("b"), 8, 8, 16);
        assert!(m.add_node(HirNode::Mul(mul)).is_ok());
    }

    #[test]
    fn add_node_rejects_duplicate_wire_driver() {
        let mut m = HirModule::new("test");
        let wid = WireId::fresh();
        let m1 = Mul {
            a: SignalRef::port("a"), b: SignalRef::port("b"),
            out: wid, a_width: 8, b_width: 8, out_width: 16,
        };
        let m2 = Mul {
            a: SignalRef::port("c"), b: SignalRef::port("d"),
            out: wid, a_width: 8, b_width: 8, out_width: 16,
        };
        assert!(m.add_node(HirNode::Mul(m1)).is_ok());
        let err = m.add_node(HirNode::Mul(m2)).unwrap_err();
        match err {
            HirBuilderError::DuplicateDriver(w, _) => assert_eq!(w, wid),
        }
    }

    #[test]
    fn test_taps_defaults_to_true_in_v1() {
        let m = HirModule::new("test");
        assert!(m.test_taps);
    }

    #[test]
    fn clock_domains_iterate_in_btreemap_order() {
        let mut m = HirModule::new("test");
        m.clock_domains.insert(ClockDomainId(5), "clk_io".to_string());
        m.clock_domains.insert(ClockDomainId(2), "clk_compute".to_string());
        let keys: Vec<_> = m.clock_domains.keys().collect();
        // BTreeMap iterates in sorted order: 0 (DEFAULT), 2, 5
        assert_eq!(keys, vec![&ClockDomainId(0), &ClockDomainId(2), &ClockDomainId(5)]);
    }
}
