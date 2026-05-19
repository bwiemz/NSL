//! HIR — Hardware Intermediate Representation.
//!
//! Sits between KIR (SIMT-shaped, for GPU codegen) and Verilog. The KIR → HIR
//! pass is load-bearing (§2.3 of M57 v1 spec); HIR → Verilog is mechanical
//! (§2.4). HIR is SSA + synthesizable-by-construction.

pub mod clock_reset;
pub mod control;
pub mod ids;
pub mod module;
pub mod nodes;
pub mod signals;

#[cfg(test)]
mod tests;

pub use clock_reset::{ClkRef, ResetPolarity, ResetRef, ResetSync};
pub use control::{ConstExpr, GenerateFor, GenerateIf};
pub use ids::{ClockDomainId, GenvarId, RegisterId, ResetSignalId, WireId};
pub use module::{HirModule, HirNode};
pub use nodes::{Add, LocalParam, Max0, Mul, Port, Register, SignExtend, Wire};
pub use signals::SignalRef;
