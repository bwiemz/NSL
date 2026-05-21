//! HIR node definitions per M57 v1 spec §3.2.

use crate::hir::clock_reset::{ClkRef, ResetRef};
use crate::hir::ids::{RegisterId, WireId};
use crate::hir::signals::SignalRef;

// --- Wire-array primitives (M57.1 wire-array realization §2) ---

/// Index expression for `SignalRef::WireArrayElement` — resolves to a Verilog
/// expression at emit time.
///
/// M57.1 wire-array realization: the fused MLP emitter declares wire arrays
/// like `acc_l1 [0:127][0:784]` and references their elements with mixed
/// literal / genvar / genvar-plus-constant indices (e.g. `acc_l1[_gv_o][_gv_k]`
/// for reads, `acc_l1[_gv_o][_gv_k + 1]` for ripple writes).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IndexExpr {
    /// Compile-time literal. Emits `n`.
    Literal(usize),
    /// Reference to a genvar declared by an enclosing `GenerateFor`.
    /// Emits the literal genvar name (e.g. `_gv_o`).
    Genvar(String),
    /// Genvar plus a non-negative constant offset. Emits `(name + k)`.
    /// Used for ripple-write targets like `acc[o][k + 1]`.
    GenvarPlus(String, i64),
}

/// Module-scope multi-dimensional wire array declaration.
///
/// Emits `wire signed [width-1:0] {name} [0:{dims[0]-1}][0:{dims[1]-1}]...;`
/// at the module level (before any generate blocks). Construction-time
/// invariant: every `SignalRef::WireArrayElement { array_name: <name>, indices }`
/// reference must have `indices.len() == dims.len()`. The emitter assumes
/// well-formed HIR and does not validate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireArray {
    pub name: String,
    pub dims: Vec<usize>,
    pub width: usize,
}

/// Drives one element of a `WireArray` from a `SignalRef`.
///
/// Emits `assign {array_name}[{idx0}]...[{idxN}] = {src};` at the position
/// where it appears in the HIR node list. Inside a GenerateFor body, indices
/// typically include `Genvar` / `GenvarPlus` refs; at module top level, indices
/// are typically `Literal`.
///
/// Construction-time invariant: `indices.len()` matches the corresponding
/// `WireArray.dims.len()` for the array being driven. The emitter does not
/// validate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignWireArrayElement {
    pub array_name: String,
    pub indices: Vec<IndexExpr>,
    pub src: crate::hir::signals::SignalRef,
}

/// Module-scope multi-dimensional const array per M57.1 wire-array mini §3.2 +
/// §4 (Task W5).
///
/// Emits
/// `localparam signed [width-1:0] {name} [0:dims[0]-1][0:dims[1]-1] = '{...values...};`
/// at the module level (in the LocalParam preamble block). For `dims.len() == 2`
/// the literal renders as `'{ '{row0}, '{row1}, ... }`; for `dims.len() == 1`
/// as `'{ cell, cell, ... }`. Higher ranks are not yet supported by the v1
/// emitter (panics).
///
/// Used to lower `W<i>` (`[k_dim, n_outputs]`) and `b<i>` (`[n_outputs]`)
/// into Verilog arrays indexable by genvar inside the matmul ripple body.
/// Values are baked at CLI time (Task W6) by the fixture loader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalParamArray {
    pub name: String,
    pub dims: Vec<usize>,
    pub width: usize,
    /// Flat values in row-major order. For 2D arrays:
    /// `values[r * dims[1] + c]` is the cell at `[r][c]`. For 1D, indexed
    /// linearly. Length must equal `dims.iter().product()`.
    pub values: Vec<i128>,
}

// --- Module-boundary nodes ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Port {
    Input  { name: String, width: usize },
    Output { name: String, width: usize, driver: SignalRef },
}

impl Port {
    pub fn input(name: impl Into<String>, width: usize) -> Self {
        Self::Input { name: name.into(), width }
    }
    pub fn output(name: impl Into<String>, width: usize, driver: SignalRef) -> Self {
        Self::Output { name: name.into(), width, driver }
    }
}

// --- Storage nodes ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wire {
    pub id: WireId,
    pub width: usize,
}

impl Wire {
    pub fn new(width: usize) -> Self {
        Self { id: WireId::fresh(), width }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalParam {
    pub name: String,
    pub width: usize,
    /// Wide enough for any v1 dtype (i64 max). Headroom over i64 supports
    /// future-milestone dtype extensions (e.g., i128 accumulator types in M58)
    /// without HIR shape change. NOT a sign bit — i64 is already signed.
    pub value: i128,
}

impl LocalParam {
    pub fn new(name: impl Into<String>, width: usize, value: i128) -> Self {
        assert!(width <= 64, "v1 LocalParam width capped at 64 bits (§3.1 spec)");
        Self { name: name.into(), width, value }
    }
}

// --- Combinational nodes ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mul {
    pub a: SignalRef,
    pub b: SignalRef,
    pub out: WireId,
    pub a_width: usize,
    pub b_width: usize,
    pub out_width: usize,
    // Invariant: out_width >= a_width + b_width. Same formula for signed and
    // unsigned — signed two's-complement has an asymmetric range whose corner
    // product (min × min = +2^(N+M-2)) is positive and would overflow N+M-1
    // bits. A static-range analysis could prove operand magnitudes stay below
    // the corner case and reduce width by one (v1's fixture admits this:
    // 64 × 128 = 8192 fits in i14), but the type-level invariant is conservative.
    //
    // Invariant covers product width only. Accumulator headroom is the KIR→HIR
    // pass's responsibility — see spec §4 for layer-by-layer sizing.
}

impl Mul {
    pub fn new(
        a: SignalRef, b: SignalRef,
        a_width: usize, b_width: usize, out_width: usize,
    ) -> Self {
        assert!(out_width >= a_width + b_width,
                "Mul out_width={} violates >= a+b={}+{}", out_width, a_width, b_width);
        Self { a, b, out: WireId::fresh(), a_width, b_width, out_width }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Add {
    pub a: SignalRef,
    pub b: SignalRef,
    pub out: WireId,
    pub width: usize,
    // Invariant: a, b, out all have width `width`. No implicit extension.
    // Width-mismatched accumulator chains (e.g., i16 product + i32 acc) are
    // resolved by inserting explicit SignExtend nodes upstream of Add.
}

impl Add {
    pub fn new(a: SignalRef, b: SignalRef, width: usize) -> Self {
        Self { a, b, out: WireId::fresh(), width }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Max0 {
    pub a: SignalRef,
    pub out: WireId,
    pub width: usize,
    // ReLU preserves width. Width changes across ReLU are unrepresentable —
    // insert explicit narrowing nodes upstream/downstream if needed (v1
    // doesn't exercise narrowing because there's no requantization).
}

impl Max0 {
    pub fn new(a: SignalRef, width: usize) -> Self {
        Self { a, out: WireId::fresh(), width }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignExtend {
    pub src: SignalRef,
    pub dst: WireId,
    pub src_width: usize,
    pub dst_width: usize,
    // Invariant: dst_width > src_width.
}

impl SignExtend {
    pub fn new(src: SignalRef, src_width: usize, dst_width: usize) -> Self {
        assert!(dst_width > src_width,
                "SignExtend dst_width={} must exceed src_width={}", dst_width, src_width);
        Self { src, dst: WireId::fresh(), src_width, dst_width }
    }
}

// --- Sequential node ---

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Register {
    pub id: RegisterId,
    pub width: usize,
    pub clock: ClkRef,
    pub reset: ResetRef,
    pub d: SignalRef,
    // .q is implicit: SignalRef::Register(self.id)
}

impl Register {
    /// PUBLIC — v1's only construction path. Constructs a register with the
    /// codegen default dialect (sync active-high, single clock domain). Per
    /// spec §3.3 Pin 1, alternate dialect Registers are representable in HIR
    /// but unconstructable via this public API in v1.
    pub fn new_v1_default(width: usize, d: SignalRef) -> Self {
        use crate::hir::clock_reset::*;
        use crate::hir::ids::*;
        Self {
            id: RegisterId::fresh(),
            width,
            clock: ClkRef { domain_id: ClockDomainId::DEFAULT },
            reset: ResetRef {
                signal_id: ResetSignalId::DEFAULT,
                polarity: ResetPolarity::High,
                sync: ResetSync::Sync,
            },
            d,
        }
    }

    /// pub(crate) — internal tests only. Exercises sync active-low for
    /// lowering-completeness tests (spec §3.3 Pin 2). Pattern 1 / IR-009:
    /// HIR can represent alternate dialects; v1 public API only constructs
    /// the codegen default. This constructor documents the extension point.
    #[allow(dead_code)]
    pub(crate) fn new_active_low_for_test(width: usize, d: SignalRef) -> Self {
        use crate::hir::clock_reset::*;
        use crate::hir::ids::*;
        Self {
            id: RegisterId::fresh(), width,
            clock: ClkRef { domain_id: ClockDomainId::DEFAULT },
            reset: ResetRef {
                signal_id: ResetSignalId::DEFAULT,
                polarity: ResetPolarity::Low,
                sync: ResetSync::Sync,
            },
            d,
        }
    }

    /// pub(crate) — async active-high test-only constructor.
    #[allow(dead_code)]
    pub(crate) fn new_async_for_test(width: usize, d: SignalRef) -> Self {
        use crate::hir::clock_reset::*;
        use crate::hir::ids::*;
        Self {
            id: RegisterId::fresh(), width,
            clock: ClkRef { domain_id: ClockDomainId::DEFAULT },
            reset: ResetRef {
                signal_id: ResetSignalId::DEFAULT,
                polarity: ResetPolarity::High,
                sync: ResetSync::Async,
            },
            d,
        }
    }

    /// pub(crate) — async active-low test-only constructor.
    #[allow(dead_code)]
    pub(crate) fn new_async_active_low_for_test(width: usize, d: SignalRef) -> Self {
        use crate::hir::clock_reset::*;
        use crate::hir::ids::*;
        Self {
            id: RegisterId::fresh(), width,
            clock: ClkRef { domain_id: ClockDomainId::DEFAULT },
            reset: ResetRef {
                signal_id: ResetSignalId::DEFAULT,
                polarity: ResetPolarity::Low,
                sync: ResetSync::Async,
            },
            d,
        }
    }

    /// pub(crate) — alternate clock domain test-only constructor.
    #[allow(dead_code)]
    pub(crate) fn new_with_clock_domain_for_test(
        width: usize, d: SignalRef, domain: crate::hir::ids::ClockDomainId,
    ) -> Self {
        use crate::hir::clock_reset::*;
        use crate::hir::ids::*;
        Self {
            id: RegisterId::fresh(), width,
            clock: ClkRef { domain_id: domain },
            reset: ResetRef {
                signal_id: ResetSignalId::DEFAULT,
                polarity: ResetPolarity::High,
                sync: ResetSync::Sync,
            },
            d,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_has_fresh_id() {
        let a = Wire::new(8);
        let b = Wire::new(8);
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn local_param_width_capped_at_64() {
        let _ = LocalParam::new("ok", 64, 0);   // exactly at cap, OK
    }

    #[test]
    #[should_panic(expected = "width capped at 64")]
    fn local_param_width_above_64_panics() {
        let _ = LocalParam::new("too_wide", 128, 0);
    }

    #[test]
    fn port_input_has_name_and_width() {
        let p = Port::input("clk", 1);
        match p {
            Port::Input { name, width } => {
                assert_eq!(name, "clk");
                assert_eq!(width, 1);
            },
            _ => panic!("expected Input variant"),
        }
    }

    #[test]
    fn mul_invariant_holds_at_equality() {
        let _ = Mul::new(SignalRef::port("a"), SignalRef::port("b"), 8, 8, 16);
    }

    #[test]
    #[should_panic(expected = "violates >= a+b")]
    fn mul_invariant_panics_when_out_too_narrow() {
        let _ = Mul::new(SignalRef::port("a"), SignalRef::port("b"), 8, 8, 15);
    }

    #[test]
    #[should_panic(expected = "must exceed src_width")]
    fn sign_extend_rejects_same_width() {
        let _ = SignExtend::new(SignalRef::port("x"), 32, 32);
    }

    #[test]
    fn add_constructs_with_matched_widths() {
        let a = Add::new(SignalRef::port("a"), SignalRef::port("b"), 32);
        assert_eq!(a.width, 32);
    }

    #[test]
    fn v1_default_register_is_sync_active_high() {
        use crate::hir::clock_reset::*;
        let r = Register::new_v1_default(32, SignalRef::port("x"));
        assert_eq!(r.reset.polarity, ResetPolarity::High);
        assert_eq!(r.reset.sync, ResetSync::Sync);
        assert_eq!(r.clock.domain_id, crate::hir::ids::ClockDomainId::DEFAULT);
    }

    #[test]
    fn test_only_constructors_produce_distinct_dialects() {
        use crate::hir::clock_reset::*;
        let sync_hi = Register::new_v1_default(8, SignalRef::port("x"));
        let sync_lo = Register::new_active_low_for_test(8, SignalRef::port("x"));
        let async_hi = Register::new_async_for_test(8, SignalRef::port("x"));
        let async_lo = Register::new_async_active_low_for_test(8, SignalRef::port("x"));

        assert_eq!(sync_hi.reset.sync, ResetSync::Sync);
        assert_eq!(sync_hi.reset.polarity, ResetPolarity::High);
        assert_eq!(sync_lo.reset.polarity, ResetPolarity::Low);
        assert_eq!(async_hi.reset.sync, ResetSync::Async);
        assert_eq!(async_lo.reset.sync, ResetSync::Async);
        assert_eq!(async_lo.reset.polarity, ResetPolarity::Low);
    }

    #[test]
    fn index_expr_literal_constructs() {
        let e = IndexExpr::Literal(5);
        assert!(matches!(e, IndexExpr::Literal(5)));
    }

    #[test]
    fn index_expr_genvar_constructs() {
        let e = IndexExpr::Genvar("_gv_o".to_string());
        match e {
            IndexExpr::Genvar(name) => assert_eq!(name, "_gv_o"),
            _ => panic!("expected Genvar"),
        }
    }

    #[test]
    fn index_expr_genvar_plus_constructs() {
        let e = IndexExpr::GenvarPlus("_gv_k".to_string(), 1);
        match e {
            IndexExpr::GenvarPlus(name, k) => {
                assert_eq!(name, "_gv_k");
                assert_eq!(k, 1);
            }
            _ => panic!("expected GenvarPlus"),
        }
    }

    #[test]
    fn wire_array_constructs_with_dims_and_width() {
        let wa = WireArray {
            name: "acc_l1".to_string(),
            dims: vec![128, 785],
            width: 32,
        };
        assert_eq!(wa.name, "acc_l1");
        assert_eq!(wa.dims, vec![128, 785]);
        assert_eq!(wa.width, 32);
    }
}
