//! HIR → Verilog mechanical lowering pass per M57 v1 spec §2.4 and §3.
//! One fixed Verilog template per HIR node kind; no optimization.

pub mod filter;
pub mod lower;
pub mod templates;
pub mod yosys;

#[cfg(test)]
mod tests;

pub use lower::VerilogEmitter;
pub use yosys::{YosysGate, YosysGateError};
