//! AIR (Algebraic Intermediate Representation) constraint format.
//!
//! Replaces PLONKish gates with a trace-based representation suitable for
//! folding provers and FRI-based backends. Each constraint operates on
//! adjacent rows of a trace matrix.
//!
//! Three constraint types:
//!   - **Transition**: polynomial relation between row[i] and row[i+1]
//!   - **Boundary**: row[0] or row[last] equals a known public value
//!   - **LookupNative**: (input, output) pair must exist in a precomputed table

use super::field::Field;

// ---------------------------------------------------------------------------
// AIR Trace
// ---------------------------------------------------------------------------

/// An execution trace: a matrix of field elements with `width` columns.
///
/// Each row represents one step of the computation. The prover fills the trace
/// by executing the model layer, then commits to the trace columns.
#[derive(Debug, Clone)]
pub struct AIRTrace<F: Field> {
    /// Number of columns per row.
    pub width: usize,
    /// Trace rows. Each inner `Vec` has exactly `width` elements.
    pub rows: Vec<Vec<F>>,
}

impl<F: Field> AIRTrace<F> {
    /// Create a new empty trace with the given column width.
    pub fn new(width: usize) -> Self {
        Self {
            width,
            rows: Vec::new(),
        }
    }

    /// Append a row to the trace.
    ///
    /// # Panics
    /// Panics if `row.len() != self.width`.
    pub fn push_row(&mut self, row: Vec<F>) {
        assert_eq!(row.len(), self.width, "row width mismatch: expected {}, got {}", self.width, row.len());
        self.rows.push(row);
    }

    /// Number of rows in the trace.
    pub fn num_rows(&self) -> usize {
        self.rows.len()
    }

    /// Access element at (row, col).
    pub fn get(&self, row: usize, col: usize) -> &F {
        &self.rows[row][col]
    }
}

// ---------------------------------------------------------------------------
// AIR Constraints
// ---------------------------------------------------------------------------

/// A single constraint in an AIR program.
#[derive(Debug, Clone)]
pub enum AIRConstraint {
    /// Transition constraint: relates row[i] columns to row[i+1] columns.
    ///
    /// The `column_indices` list the columns involved. The constraint verifies
    /// that the specified relation holds between adjacent rows.
    Transition {
        /// Polynomial degree of the constraint.
        degree: usize,
        /// Columns read from the current row.
        current_cols: Vec<usize>,
        /// Columns read from the next row.
        next_cols: Vec<usize>,
        /// Human-readable description (for debugging/reporting).
        description: String,
    },

    /// Boundary constraint: a specific cell equals a known public value.
    Boundary {
        /// Row index (0 for initial, trace.len()-1 for final).
        row: usize,
        /// Column index.
        col: usize,
        /// Description of what this boundary represents.
        description: String,
    },

    /// Lookup-native constraint: (input_col, output_col) must exist in a
    /// precomputed table. This is the core of Jolt-style arithmetization —
    /// activation functions are verified via table lookups rather than
    /// polynomial constraints, yielding 4-7x speedup.
    LookupNative {
        /// Column containing lookup input values.
        input_col: usize,
        /// Column containing lookup output values.
        output_col: usize,
        /// Name of the precomputed table (e.g., "relu", "gelu").
        table_name: String,
    },
}

// ---------------------------------------------------------------------------
// AIR Program
// ---------------------------------------------------------------------------

/// A complete AIR program: constraints + trace metadata.
///
/// The prover generates a trace satisfying all constraints, commits to it,
/// and the verifier checks constraint satisfaction probabilistically.
#[derive(Debug, Clone)]
pub struct AIRProgram {
    /// All constraints that the trace must satisfy.
    pub constraints: Vec<AIRConstraint>,
    /// Number of columns in the trace.
    pub trace_width: usize,
    /// Columns whose values are public inputs (known to verifier).
    pub public_input_cols: Vec<usize>,
    /// Columns whose values are public outputs (revealed by proof).
    pub public_output_cols: Vec<usize>,
}

impl AIRProgram {
    /// Create a new empty AIR program with the given trace width.
    pub fn new(trace_width: usize) -> Self {
        Self {
            constraints: Vec::new(),
            trace_width,
            public_input_cols: Vec::new(),
            public_output_cols: Vec::new(),
        }
    }

    /// Add a transition constraint.
    pub fn add_transition(
        &mut self,
        degree: usize,
        current_cols: Vec<usize>,
        next_cols: Vec<usize>,
        description: &str,
    ) {
        self.constraints.push(AIRConstraint::Transition {
            degree,
            current_cols,
            next_cols,
            description: description.to_string(),
        });
    }

    /// Add a boundary constraint.
    pub fn add_boundary(&mut self, row: usize, col: usize, description: &str) {
        self.constraints.push(AIRConstraint::Boundary {
            row,
            col,
            description: description.to_string(),
        });
    }

    /// Add a lookup-native constraint.
    pub fn add_lookup(&mut self, input_col: usize, output_col: usize, table_name: &str) {
        self.constraints.push(AIRConstraint::LookupNative {
            input_col,
            output_col,
            table_name: table_name.to_string(),
        });
    }

    /// Count constraints by type.
    pub fn constraint_counts(&self) -> (usize, usize, usize) {
        let mut transitions = 0;
        let mut boundaries = 0;
        let mut lookups = 0;
        for c in &self.constraints {
            match c {
                AIRConstraint::Transition { .. } => transitions += 1,
                AIRConstraint::Boundary { .. } => boundaries += 1,
                AIRConstraint::LookupNative { .. } => lookups += 1,
            }
        }
        (transitions, boundaries, lookups)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::field_m31::Mersenne31Field;

    #[test]
    fn test_air_trace_basic() {
        let mut trace = AIRTrace::<Mersenne31Field>::new(3);
        trace.push_row(vec![
            Mersenne31Field::from_u64(1),
            Mersenne31Field::from_u64(2),
            Mersenne31Field::from_u64(3),
        ]);
        trace.push_row(vec![
            Mersenne31Field::from_u64(4),
            Mersenne31Field::from_u64(5),
            Mersenne31Field::from_u64(6),
        ]);
        assert_eq!(trace.num_rows(), 2);
        assert_eq!(trace.get(0, 1), &Mersenne31Field::from_u64(2));
        assert_eq!(trace.get(1, 2), &Mersenne31Field::from_u64(6));
    }

    #[test]
    fn test_air_program_constraints() {
        let mut prog = AIRProgram::new(4);
        prog.add_transition(2, vec![0, 1], vec![2], "matmul accumulation");
        prog.add_boundary(0, 0, "input");
        prog.add_lookup(1, 2, "relu");
        prog.add_lookup(1, 3, "gelu");

        let (t, b, l) = prog.constraint_counts();
        assert_eq!(t, 1);
        assert_eq!(b, 1);
        assert_eq!(l, 2);
        assert_eq!(prog.constraints.len(), 4);
    }

    #[test]
    #[should_panic(expected = "row width mismatch")]
    fn test_air_trace_width_mismatch() {
        let mut trace = AIRTrace::<Mersenne31Field>::new(3);
        trace.push_row(vec![Mersenne31Field::zero(), Mersenne31Field::one()]); // wrong width
    }
}
