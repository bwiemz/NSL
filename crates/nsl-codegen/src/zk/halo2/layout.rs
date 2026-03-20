//! Halo2 PLONKish grid layout planner.
//!
//! Plans how ZK-IR instructions map to the Halo2 PLONKish grid (rows x columns).
//! For each [`ZkInstruction`] the planner computes:
//!
//! - How many rows it occupies in the PLONKish table.
//! - Which gate type handles the constraint.
//! - Where in the grid it is placed (start row).
//!
//! The output [`CircuitLayout`] is consumed by the circuit compiler to configure
//! Halo2's column assignments and to determine the circuit size parameter `k`.
//!
//! ## Row cost model
//!
//! | Instruction   | Rows                           |
//! |---------------|--------------------------------|
//! | `DotProduct`  | `len(a)` (one per MAC)         |
//! | `Mul`         | 1                              |
//! | `FixedMul`    | 1 (mul + range check fused)    |
//! | `Add`         | 0 (folded into linear comb.)   |
//! | `Const`       | 0 (embedded in fixed column)   |
//! | `AssertEq`    | 1 (linear constraint)          |
//! | `Lookup`      | 1 (per element)                |
//! | `Requantize`  | 1 (affine + range check fused) |
//! | `Remap`       | 0 (wire permutation only)      |

use super::super::ir::{ZkIR, ZkInstruction};

// ---------------------------------------------------------------------------
// CircuitLayout
// ---------------------------------------------------------------------------

/// Complete layout plan for a Halo2 PLONKish circuit.
///
/// Describes the grid dimensions (columns x rows), the `k` parameter (where
/// the table has `2^k` rows), and the assignment of each ZK-IR instruction
/// to a region of the grid.
#[derive(Debug, Clone)]
pub struct CircuitLayout {
    /// Number of advice (witness) columns. Max 4 for dot-product gates
    /// (a, b, acc_curr, acc_next), fewer for simpler circuits.
    pub num_advice_columns: usize,
    /// Number of fixed columns. 1 for constants, plus 1 per lookup table.
    pub num_fixed_columns: usize,
    /// Number of instance (public I/O) columns. 1 for public inputs,
    /// 1 for public outputs. Minimum 1 if any public wires exist.
    pub num_instance_columns: usize,
    /// Total number of constraint rows (before padding to `2^k`).
    pub num_rows: usize,
    /// Circuit size parameter: `2^k >= num_rows`.
    pub k: u32,
    /// Per-instruction assignment: which rows each instruction occupies.
    pub gate_assignments: Vec<GateAssignment>,
}

// ---------------------------------------------------------------------------
// GateAssignment
// ---------------------------------------------------------------------------

/// Describes how a single ZK-IR instruction is placed in the PLONKish grid.
#[derive(Debug, Clone)]
pub struct GateAssignment {
    /// Index of the instruction in `ZkIR::instructions`.
    pub instruction_idx: usize,
    /// First row occupied by this gate.
    pub start_row: usize,
    /// Number of rows consumed.
    pub num_rows: usize,
    /// Human-readable gate type name (e.g. `"DotProduct"`, `"Mul"`, `"Lookup"`).
    pub gate_type: String,
}

// ---------------------------------------------------------------------------
// plan_layout
// ---------------------------------------------------------------------------

/// Plan the PLONKish grid layout for a given ZK-IR program.
///
/// Walks the instruction list, assigns each instruction a contiguous range of
/// rows, and computes the required column counts and the `k` parameter.
pub fn plan_layout(ir: &ZkIR) -> CircuitLayout {
    let mut current_row: usize = 0;
    let mut gate_assignments = Vec::with_capacity(ir.instructions.len());
    let mut max_advice_columns: usize = 0;
    let mut needs_dot_product = false;
    let mut needs_fixed_mul = false;
    let mut needs_requantize = false;

    // Collect unique lookup table names — each table needs a fixed column.
    let mut lookup_table_names = std::collections::BTreeSet::new();

    for (idx, instr) in ir.instructions.iter().enumerate() {
        let (rows, gate_type) = match instr {
            ZkInstruction::DotProduct { a, .. } => {
                needs_dot_product = true;
                (a.len(), "DotProduct")
            }
            ZkInstruction::Mul { .. } => (1, "Mul"),
            ZkInstruction::FixedMul { .. } => {
                needs_fixed_mul = true;
                (1, "FixedMul")
            }
            ZkInstruction::Add { .. } => (0, "Add"),
            ZkInstruction::Const { .. } => (0, "Const"),
            ZkInstruction::AssertEq { .. } => (1, "AssertEq"),
            ZkInstruction::Lookup { table, input_bits, .. } => {
                lookup_table_names.insert((table.clone(), *input_bits));
                (1, "Lookup")
            }
            ZkInstruction::Requantize { .. } => {
                needs_requantize = true;
                (1, "Requantize")
            }
            ZkInstruction::Remap { .. } => (0, "Remap"),
        };

        gate_assignments.push(GateAssignment {
            instruction_idx: idx,
            start_row: current_row,
            num_rows: rows,
            gate_type: gate_type.to_string(),
        });

        current_row += rows;
    }

    // -- Compute advice columns needed -----------------------------------------
    // DotProduct needs 4 columns (a, b, acc_curr, acc_next).
    // FixedMul needs 4 (a, b, result, remainder).
    // Requantize needs 4 (input, shifted, scaled, out).
    // Simple Mul needs 3 (a, b, out).
    // Minimum 1 for any circuit.
    if needs_dot_product || needs_fixed_mul || needs_requantize {
        max_advice_columns = 4;
    } else if ir.instructions.iter().any(|i| matches!(i, ZkInstruction::Mul { .. })) {
        max_advice_columns = 3;
    }
    // AssertEq needs 2 columns.
    if max_advice_columns < 2
        && ir
            .instructions
            .iter()
            .any(|i| matches!(i, ZkInstruction::AssertEq { .. }))
    {
        max_advice_columns = 2;
    }
    // Lookup needs at least 2 (input, output).
    if max_advice_columns < 2 && !lookup_table_names.is_empty() {
        max_advice_columns = 2;
    }
    // Ensure at least 1 advice column for non-empty circuits.
    if max_advice_columns == 0 && !ir.instructions.is_empty() {
        max_advice_columns = 1;
    }

    // -- Fixed columns: 1 for constants + 1 per lookup table -------------------
    let num_fixed_columns = if lookup_table_names.is_empty() {
        1
    } else {
        1 + lookup_table_names.len()
    };

    // -- Instance columns: 1 for public inputs, 1 for public outputs -----------
    let has_public_inputs = !ir.public_inputs.is_empty();
    let has_public_outputs = !ir.public_outputs.is_empty();
    let num_instance_columns = match (has_public_inputs, has_public_outputs) {
        (false, false) => 0,
        (true, false) | (false, true) => 1,
        (true, true) => 2,
    };

    // -- Compute total rows including public I/O + blinding --------------------
    // Public inputs/outputs need their own rows in Halo2.
    let public_io_rows = ir.public_inputs.len() + ir.public_outputs.len();
    // Halo2 typically needs 5 blinding rows for zero-knowledge.
    const BLINDING_ROWS: usize = 5;
    let total_rows = current_row + public_io_rows + BLINDING_ROWS;

    // k = ceil(log2(total_rows)), minimum 1
    let k = compute_k(total_rows);

    CircuitLayout {
        num_advice_columns: max_advice_columns,
        num_fixed_columns,
        num_instance_columns,
        num_rows: current_row,
        k,
        gate_assignments,
    }
}

/// Compute the minimum `k` such that `2^k >= rows`, with `k >= 1`.
fn compute_k(rows: usize) -> u32 {
    let mut k: u32 = 1;
    while (1usize << k) < rows {
        k += 1;
        if k >= 28 {
            break; // safety cap: 2^28 = 268M rows
        }
    }
    k
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::field::FieldElement;
    use crate::zk::ir::{Wire, ZkIR, ZkInstruction};

    #[test]
    fn layout_simple_add_circuit() {
        let mut ir = ZkIR::new("test");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("out");
        ir.push(ZkInstruction::Add {
            out: w2,
            a: w0,
            b: w1,
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2]);

        let layout = plan_layout(&ir);

        // Add is free: 0 constraint rows.
        assert_eq!(layout.num_rows, 0);
        // k must be >= 1.
        assert!(layout.k >= 1);
        // Instance columns: 1 for inputs + 1 for outputs = 2.
        assert_eq!(layout.num_instance_columns, 2);
        // 1 gate assignment for the single Add instruction.
        assert_eq!(layout.gate_assignments.len(), 1);
        assert_eq!(layout.gate_assignments[0].gate_type, "Add");
        assert_eq!(layout.gate_assignments[0].num_rows, 0);
    }

    #[test]
    fn layout_mul_circuit() {
        let mut ir = ZkIR::new("test_mul");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("out");
        ir.push(ZkInstruction::Mul {
            out: w2,
            a: w0,
            b: w1,
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2]);

        let layout = plan_layout(&ir);

        assert_eq!(layout.num_rows, 1);
        assert_eq!(layout.num_advice_columns, 3);
        assert!(layout.k >= 1);
        assert_eq!(layout.gate_assignments[0].gate_type, "Mul");
        assert_eq!(layout.gate_assignments[0].num_rows, 1);
    }

    #[test]
    fn layout_dot_product() {
        let mut ir = ZkIR::new("test_dot");
        let a: Vec<Wire> = (0..64).map(|i| ir.alloc_wire(&format!("a{i}"))).collect();
        let b: Vec<Wire> = (0..64).map(|i| ir.alloc_wire(&format!("b{i}"))).collect();
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::DotProduct {
            out,
            a: a.clone(),
            b: b.clone(),
        });
        ir.set_public_inputs(vec![a[0]]);
        ir.set_public_outputs(vec![out]);

        let layout = plan_layout(&ir);

        // DotProduct of length 64 = 64 rows.
        assert_eq!(layout.num_rows, 64);
        // Needs 4 advice columns (a, b, acc_curr, acc_next).
        assert_eq!(layout.num_advice_columns, 4);
        assert_eq!(layout.gate_assignments[0].gate_type, "DotProduct");
        assert_eq!(layout.gate_assignments[0].num_rows, 64);
    }

    #[test]
    fn layout_mixed_instructions() {
        let mut ir = ZkIR::new("mixed");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("mul_out");
        let w3 = ir.alloc_wire("add_out");
        let w4 = ir.alloc_wire("fixed_mul_out");
        let w5 = ir.alloc_wire("lookup_out");

        ir.push(ZkInstruction::Mul {
            out: w2,
            a: w0,
            b: w1,
        });
        ir.push(ZkInstruction::Add {
            out: w3,
            a: w0,
            b: w2,
        });
        ir.push(ZkInstruction::FixedMul {
            out: w4,
            a: w2,
            b: w3,
            frac_bits: 8,
        });
        ir.push(ZkInstruction::Lookup {
            out: w5,
            table: "relu".into(),
            input: w4,
            input_bits: 8,
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w5]);

        let layout = plan_layout(&ir);

        // Mul: 1 row, Add: 0, FixedMul: 1, Lookup: 1 = 3 rows total.
        assert_eq!(layout.num_rows, 3);
        // FixedMul requires 4 advice columns.
        assert_eq!(layout.num_advice_columns, 4);
        // 1 constant column + 1 for relu table = 2 fixed columns.
        assert_eq!(layout.num_fixed_columns, 2);
        // 4 gate assignments.
        assert_eq!(layout.gate_assignments.len(), 4);
    }

    #[test]
    fn layout_requantize() {
        let mut ir = ZkIR::new("requant");
        let w0 = ir.alloc_wire("input");
        let w1 = ir.alloc_wire("output");
        ir.push(ZkInstruction::Requantize {
            out: w1,
            input: w0,
            scale: FieldElement::from_u64(1),
            zero_point: FieldElement::zero(),
            target_bits: 8,
        });
        ir.set_public_inputs(vec![w0]);
        ir.set_public_outputs(vec![w1]);

        let layout = plan_layout(&ir);

        assert_eq!(layout.num_rows, 1);
        assert_eq!(layout.num_advice_columns, 4);
        assert_eq!(layout.gate_assignments[0].gate_type, "Requantize");
    }

    #[test]
    fn layout_remap_and_const_are_zero_cost() {
        let mut ir = ZkIR::new("zero_cost");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let wc = ir.alloc_wire("const");

        ir.push(ZkInstruction::Const {
            out: wc,
            value: FieldElement::from_u64(42),
        });
        ir.push(ZkInstruction::Remap {
            out: vec![w1],
            input: vec![w0],
            permutation: vec![0],
        });

        let layout = plan_layout(&ir);

        assert_eq!(layout.num_rows, 0);
        assert_eq!(layout.gate_assignments.len(), 2);
        assert_eq!(layout.gate_assignments[0].gate_type, "Const");
        assert_eq!(layout.gate_assignments[0].num_rows, 0);
        assert_eq!(layout.gate_assignments[1].gate_type, "Remap");
        assert_eq!(layout.gate_assignments[1].num_rows, 0);
    }

    #[test]
    fn layout_empty_circuit() {
        let ir = ZkIR::new("empty");
        let layout = plan_layout(&ir);

        assert_eq!(layout.num_rows, 0);
        assert_eq!(layout.num_advice_columns, 0);
        assert_eq!(layout.num_instance_columns, 0);
        assert!(layout.k >= 1);
        assert!(layout.gate_assignments.is_empty());
    }

    #[test]
    fn layout_gate_assignments_sequential() {
        let mut ir = ZkIR::new("sequential");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        let w3 = ir.alloc_wire("d");
        let w4 = ir.alloc_wire("e");

        ir.push(ZkInstruction::Mul {
            out: w2,
            a: w0,
            b: w1,
        }); // row 0
        ir.push(ZkInstruction::AssertEq { a: w2, b: w3 }); // row 1
        ir.push(ZkInstruction::Lookup {
            out: w4,
            table: "relu".into(),
            input: w2,
            input_bits: 8,
        }); // row 2

        let layout = plan_layout(&ir);

        assert_eq!(layout.gate_assignments[0].start_row, 0);
        assert_eq!(layout.gate_assignments[1].start_row, 1);
        assert_eq!(layout.gate_assignments[2].start_row, 2);
        assert_eq!(layout.num_rows, 3);
    }

    #[test]
    fn layout_no_public_io_gives_zero_instance_columns() {
        let mut ir = ZkIR::new("no_io");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        ir.push(ZkInstruction::Mul {
            out: w2,
            a: w0,
            b: w1,
        });
        // No public inputs or outputs set.

        let layout = plan_layout(&ir);
        assert_eq!(layout.num_instance_columns, 0);
    }

    #[test]
    fn layout_k_accounts_for_blinding_and_public_io() {
        let mut ir = ZkIR::new("k_test");
        // Create a circuit with exactly 1 constraint row.
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("out");
        ir.push(ZkInstruction::Mul {
            out: w2,
            a: w0,
            b: w1,
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2]);

        let layout = plan_layout(&ir);

        // Total rows = 1 (constraint) + 3 (public IO) + 5 (blinding) = 9
        // 2^4 = 16 >= 9, so k should be 4.
        assert_eq!(layout.k, 4);
    }
}
