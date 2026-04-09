//! M55: ZK Intermediate Representation (ZK-IR).
//!
//! Defines the instruction set and program structure used to represent
//! NSL inference graphs as arithmetic circuits over the BN254 scalar field.
//!
//! ## Design
//!
//! A [`ZkIR`] is a flat, SSA-like list of [`ZkInstruction`]s operating on
//! [`Wire`]s.  Each wire is a unique variable (field element) in the circuit.
//! Wires are allocated sequentially via [`ZkIR::alloc_wire`], giving them
//! deterministic IDs that correspond directly to columns in the PLONKish
//! arithmetization.
//!
//! Non-linear operations (activations) are handled via pre-computed
//! [`LookupTable`]s registered in the [`ZkIR`] and referenced by name in
//! [`ZkInstruction::Lookup`].

use super::field::FieldElement;
use super::lookup::LookupTable;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Wire
// ---------------------------------------------------------------------------

/// A wire (variable) in the ZK circuit.
///
/// Each wire represents a single field element value. Wire IDs are assigned
/// sequentially starting from 0 by [`ZkIR::alloc_wire`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wire(pub u64);

// ---------------------------------------------------------------------------
// ZkInstruction
// ---------------------------------------------------------------------------

/// ZK-IR instruction set.
///
/// Instructions are pure (no side effects) and form a static single-assignment
/// (SSA) graph. Each instruction writes to exactly one output wire (or a vector
/// of output wires for [`ZkInstruction::Remap`]).
#[derive(Debug, Clone)]
pub enum ZkInstruction {
    /// Field multiplication: `out = a * b`
    ///
    /// Costs one multiplication gate in PLONKish arithmetization.
    Mul { out: Wire, a: Wire, b: Wire },

    /// Field addition: `out = a + b`
    ///
    /// Free in PLONKish (absorbed into the linear combination of the
    /// adjacent multiplication gate). Emitted as a distinct instruction
    /// for clarity and backend flexibility.
    Add { out: Wire, a: Wire, b: Wire },

    /// Embed a constant field element.
    ///
    /// The constant is fixed at circuit-build time and does not require
    /// a private witness column.
    Const { out: Wire, value: FieldElement },

    /// Equality assertion: `a == b`
    ///
    /// Compiles to a constraint `a - b = 0`. Used to bind public inputs
    /// and outputs to their claimed values.
    AssertEq { a: Wire, b: Wire },

    /// Custom gate: dot product of two wire vectors.
    ///
    /// `out = sum_i(a[i] * b[i])`
    ///
    /// Backends that support custom gates (e.g. Halo2) can lower this
    /// directly; others expand it into a sequence of Mul + Add instructions.
    /// `a` and `b` must have the same length.
    DotProduct {
        out: Wire,
        a: Vec<Wire>,
        b: Vec<Wire>,
    },

    /// Fixed-point multiplication with rescaling.
    ///
    /// Computes `out = (a * b) >> frac_bits` in the field, modelling
    /// quantized integer matrix-multiply semantics. The right-shift is
    /// implemented as a field inverse of `2^frac_bits`.
    FixedMul {
        out: Wire,
        a: Wire,
        b: Wire,
        frac_bits: u32,
    },

    /// Lookup table evaluation.
    ///
    /// Looks up `input` in the pre-registered table named `table` and
    /// writes the result to `out`. The table must have been registered
    /// via [`ZkIR::register_table`] with the matching `input_bits`.
    Lookup {
        out: Wire,
        table: String,
        input: Wire,
        input_bits: u32,
    },

    /// Requantize a wide accumulator to a narrow bit-width before lookup.
    ///
    /// Applies the affine transformation:
    /// `out = clamp(round((input - zero_point) * scale), 0, 2^target_bits - 1)`
    ///
    /// Used to bring the output of a fixed-point matmul back into the range
    /// expected by the subsequent activation lookup table.
    Requantize {
        out: Wire,
        input: Wire,
        scale: FieldElement,
        zero_point: FieldElement,
        target_bits: u32,
    },

    /// Wire remapping (reshape, transpose) — zero cost.
    ///
    /// Reorders `input` wires into `out` according to `permutation`, where
    /// `out[i] = input[permutation[i]]`. No arithmetic constraints are
    /// generated; this is a pure bookkeeping operation for the compiler.
    ///
    /// `out`, `input`, and `permutation` must all have the same length.
    Remap {
        out: Vec<Wire>,
        input: Vec<Wire>,
        permutation: Vec<usize>,
    },
}

// ---------------------------------------------------------------------------
// ZkIR
// ---------------------------------------------------------------------------

/// Complete ZK-IR program for a model.
///
/// A `ZkIR` captures the full arithmetic circuit derived from an NSL model's
/// forward pass. It is the bridge between the NSL compiler IR and the
/// backend-specific circuit representations (Halo2, Plonky3, …).
///
/// ## Workflow
///
/// 1. Allocate wires for all model inputs via [`alloc_wire`].
/// 2. Append instructions via [`push`].
/// 3. Mark public inputs/outputs via [`set_public_inputs`] / [`set_public_outputs`].
/// 4. Pass the completed `ZkIR` to a [`ZkBackend::compile`] implementation.
///
/// [`ZkBackend::compile`]: super::backend::ZkBackend::compile
/// [`alloc_wire`]: ZkIR::alloc_wire
#[derive(Debug)]
pub struct ZkIR {
    /// Human-readable name of the circuit (usually the model name).
    pub name: String,
    /// Ordered list of instructions in evaluation order.
    pub instructions: Vec<ZkInstruction>,
    /// Wires whose values are supplied publicly by the verifier.
    pub public_inputs: Vec<Wire>,
    /// Wires whose values are revealed publicly after proof generation.
    pub public_outputs: Vec<Wire>,
    /// Wires whose values are known only to the prover (model weights, etc.).
    pub private_inputs: Vec<Wire>,
    /// Total number of allocated wires (next free wire ID).
    pub num_wires: u64,
    /// Pre-computed lookup tables, keyed by `(name, input_bits)`.
    ///
    /// Multiple tables with the same name but different bit-widths are stored
    /// separately, enabling mixed-precision circuits.
    pub lookup_tables: HashMap<(String, u32), LookupTable>,
    /// Optional debug names for wires, keyed by wire ID.
    ///
    /// Not every wire needs a name; this is populated by [`alloc_wire`] when
    /// a non-empty name is given.
    pub wire_names: HashMap<Wire, String>,
}

impl ZkIR {
    /// Create a new, empty `ZkIR` with the given circuit name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            instructions: Vec::new(),
            public_inputs: Vec::new(),
            public_outputs: Vec::new(),
            private_inputs: Vec::new(),
            num_wires: 0,
            lookup_tables: HashMap::new(),
            wire_names: HashMap::new(),
        }
    }

    /// Allocate the next wire ID and optionally record a debug name for it.
    ///
    /// Wire IDs are assigned sequentially: the first call returns `Wire(0)`,
    /// the second `Wire(1)`, and so on.
    ///
    /// If `name` is non-empty, the name is stored in [`wire_names`] for use
    /// in debug output and error messages.
    ///
    /// [`wire_names`]: ZkIR::wire_names
    pub fn alloc_wire(&mut self, name: &str) -> Wire {
        let wire = Wire(self.num_wires);
        self.num_wires += 1;
        if !name.is_empty() {
            self.wire_names.insert(wire, name.to_string());
        }
        wire
    }

    /// Append an instruction to the end of the instruction list.
    pub fn push(&mut self, instruction: ZkInstruction) {
        self.instructions.push(instruction);
    }

    /// Set the list of public input wires.
    ///
    /// Public inputs are field elements provided by the verifier; they appear
    /// in the proof's public statement alongside public outputs.
    pub fn set_public_inputs(&mut self, wires: Vec<Wire>) {
        self.public_inputs = wires;
    }

    /// Set the list of public output wires.
    ///
    /// Public outputs are field elements revealed after proof generation;
    /// they form the verifiable claim about the model's inference result.
    pub fn set_public_outputs(&mut self, wires: Vec<Wire>) {
        self.public_outputs = wires;
    }

    /// Register a pre-computed lookup table for use in [`ZkInstruction::Lookup`].
    ///
    /// The table is keyed by `(table.name.clone(), table.input_bits)`.  If a
    /// table with the same key already exists it is silently replaced.
    pub fn register_table(&mut self, table: LookupTable) {
        self.lookup_tables
            .insert((table.name.clone(), table.input_bits), table);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::lookup::LookupTable;

    #[test]
    fn zkir_can_represent_simple_mul() {
        let mut ir = ZkIR::new("test_mul");
        let w0 = ir.alloc_wire("input_a");
        let w1 = ir.alloc_wire("input_b");
        let w2 = ir.alloc_wire("output");
        ir.push(ZkInstruction::Mul {
            out: w2,
            a: w0,
            b: w1,
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2]);
        assert_eq!(ir.num_wires, 3);
        assert_eq!(ir.instructions.len(), 1);
    }

    #[test]
    fn zkir_lookup_table_keyed_by_name_and_bits() {
        let mut ir = ZkIR::new("test_lookup");
        ir.register_table(LookupTable {
            name: "gelu".into(),
            input_bits: 8,
            output_bits: 8,
            entries: vec![],
        });
        ir.register_table(LookupTable {
            name: "gelu".into(),
            input_bits: 16,
            output_bits: 16,
            entries: vec![],
        });
        assert_eq!(ir.lookup_tables.len(), 2);
    }

    #[test]
    fn wire_ids_are_sequential() {
        let mut ir = ZkIR::new("test_seq");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        assert_eq!(w0.0, 0);
        assert_eq!(w1.0, 1);
        assert_eq!(w2.0, 2);
    }

    #[test]
    fn alloc_wire_stores_debug_names() {
        let mut ir = ZkIR::new("test_names");
        let w = ir.alloc_wire("my_weight");
        assert_eq!(ir.wire_names.get(&w).map(String::as_str), Some("my_weight"));
    }

    #[test]
    fn alloc_wire_empty_name_not_stored() {
        let mut ir = ZkIR::new("test_anonymous");
        let w = ir.alloc_wire("");
        assert!(
            !ir.wire_names.contains_key(&w),
            "empty name should not be stored"
        );
    }

    #[test]
    fn zkir_push_appends_in_order() {
        let mut ir = ZkIR::new("test_order");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        let w3 = ir.alloc_wire("d");
        ir.push(ZkInstruction::Add {
            out: w2,
            a: w0,
            b: w1,
        });
        ir.push(ZkInstruction::Mul {
            out: w3,
            a: w0,
            b: w2,
        });
        assert_eq!(ir.instructions.len(), 2);
        // Verify ordering: first instruction is Add, second is Mul.
        assert!(matches!(ir.instructions[0], ZkInstruction::Add { .. }));
        assert!(matches!(ir.instructions[1], ZkInstruction::Mul { .. }));
    }

    #[test]
    fn zkir_public_io_roundtrip() {
        let mut ir = ZkIR::new("test_io");
        let inp = ir.alloc_wire("x");
        let out = ir.alloc_wire("y");
        ir.set_public_inputs(vec![inp]);
        ir.set_public_outputs(vec![out]);
        assert_eq!(ir.public_inputs, vec![inp]);
        assert_eq!(ir.public_outputs, vec![out]);
    }

    #[test]
    fn register_table_replaces_same_key() {
        let mut ir = ZkIR::new("test_replace");
        ir.register_table(LookupTable {
            name: "relu".into(),
            input_bits: 8,
            output_bits: 8,
            entries: vec![],
        });
        ir.register_table(LookupTable {
            name: "relu".into(),
            input_bits: 8,
            output_bits: 16, // different output_bits, same key
            entries: vec![],
        });
        // Key is (name, input_bits), so there should still be only 1 entry.
        assert_eq!(ir.lookup_tables.len(), 1);
        let tbl = ir.lookup_tables.get(&("relu".into(), 8)).unwrap();
        assert_eq!(
            tbl.output_bits, 16,
            "table should be replaced by second registration"
        );
    }

    #[test]
    fn zkir_new_has_zero_wires_and_no_instructions() {
        let ir = ZkIR::new("empty");
        assert_eq!(ir.num_wires, 0);
        assert!(ir.instructions.is_empty());
        assert!(ir.public_inputs.is_empty());
        assert!(ir.public_outputs.is_empty());
        assert!(ir.private_inputs.is_empty());
        assert!(ir.lookup_tables.is_empty());
        assert!(ir.wire_names.is_empty());
    }
}
