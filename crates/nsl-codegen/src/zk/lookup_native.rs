//! Jolt-style lookup-native gates for ZK inference circuits.
//!
//! In traditional ZK, activation functions (ReLU, GELU, etc.) are expressed as
//! high-degree polynomial constraints — expensive to prove. Lookup-native
//! arithmetization replaces these with table lookups: the prover demonstrates
//! that (input, output) exists in a precomputed table.
//!
//! This approach (used by Jolt, zkLLM, Lasso) is 4-7x faster for activation-heavy
//! inference circuits because:
//!   1. No degree-3+ constraints for non-linear functions
//!   2. Multiplicity tracking via logarithmic derivatives is cheap
//!   3. Tables are precomputed once and reused across all layers

use super::field::Field;

// ---------------------------------------------------------------------------
// Lookup-native gate
// ---------------------------------------------------------------------------

/// A lookup-native gate: activation f(x) is a table lookup, not a polynomial constraint.
///
/// The prover demonstrates that every (input, output) pair in the trace exists
/// in the precomputed table. Verification uses multiplicity tracking (counting
/// how many times each table entry is accessed).
#[derive(Debug, Clone)]
pub struct LookupNativeGate {
    /// Name of the activation/operation (e.g., "relu", "gelu", "sigmoid").
    pub table_name: String,
    /// Bit width of input values.
    pub input_bits: u32,
    /// Bit width of output values.
    pub output_bits: u32,
    /// Precomputed (input, output) pairs as raw u64 values.
    /// These are the quantized fixed-point representations.
    pub entries: Vec<(u64, u64)>,
}

impl LookupNativeGate {
    /// Create a lookup-native gate from an existing lookup table.
    ///
    /// Integrates with `lookup.rs` precomputed tables by extracting the raw
    /// u64 input/output pairs from the field-element table entries.
    pub fn from_lookup_table(table: &super::lookup::LookupTable) -> Self {
        let entries: Vec<(u64, u64)> = table
            .entries
            .iter()
            .map(|(inp, out)| {
                // Extract the least-significant limb as raw u64
                (inp.limbs[0], out.limbs[0])
            })
            .collect();

        Self {
            table_name: table.name.clone(),
            input_bits: table.input_bits,
            output_bits: table.output_bits,
            entries,
        }
    }

    /// Number of entries in the table.
    pub fn table_size(&self) -> usize {
        self.entries.len()
    }

    /// Look up an output value for a given input.
    /// Returns None if the input is not in the table.
    pub fn lookup(&self, input: u64) -> Option<u64> {
        self.entries
            .iter()
            .find(|(i, _)| *i == input)
            .map(|(_, o)| *o)
    }
}

// ---------------------------------------------------------------------------
// Multiplicity tracking
// ---------------------------------------------------------------------------

/// Multiplicity tracker: counts how many times each table entry is accessed.
///
/// This is the core data structure for lookup-native verification. The prover
/// commits to the multiplicity vector alongside the trace, and the verifier
/// checks that the multiplicities are consistent with the trace's lookup columns.
///
/// Logarithmic derivative approach (Lasso/Jolt):
///   sum_i 1/(X - trace_i) == sum_j multiplicity_j / (X - table_j)
///
/// This identity is checked at a random evaluation point, making lookup
/// verification a single inner-product test.
#[derive(Debug, Clone)]
pub struct LookupMultiplicities {
    /// Per-entry access counts. `counts[i]` = number of trace rows that
    /// accessed table entry `i`.
    pub counts: Vec<u64>,
    /// Total number of lookups performed.
    pub total_lookups: u64,
}

impl LookupMultiplicities {
    /// Create a new multiplicity tracker for a table of the given size.
    pub fn new(table_size: usize) -> Self {
        Self {
            counts: vec![0; table_size],
            total_lookups: 0,
        }
    }

    /// Record an access to table entry at the given index.
    pub fn record(&mut self, entry_index: usize) {
        self.counts[entry_index] += 1;
        self.total_lookups += 1;
    }

    /// Check that all multiplicities are non-negative (always true for u64)
    /// and that they sum to the expected total.
    pub fn verify_consistency(&self) -> bool {
        let sum: u64 = self.counts.iter().sum();
        sum == self.total_lookups
    }

    /// Return the number of distinct entries accessed (entries with count > 0).
    pub fn distinct_entries(&self) -> usize {
        self.counts.iter().filter(|&&c| c > 0).count()
    }
}

// ---------------------------------------------------------------------------
// Lookup argument (logarithmic derivative)
// ---------------------------------------------------------------------------

/// Compute the logarithmic derivative sum for trace lookups.
///
/// For a random challenge `beta`, computes:
///   sum_i 1 / (beta - trace_value_i)
///
/// This is half of the Lasso/Jolt lookup argument. The verifier checks this
/// against the table-side sum (weighted by multiplicities).
pub fn log_derivative_trace_sum<F: Field>(trace_values: &[F], beta: &F) -> F {
    let mut sum = F::zero();
    for val in trace_values {
        // 1 / (beta - val)
        let diff = beta.field_sub(val);
        let inv = diff.field_inv();
        sum = sum.field_add(&inv);
    }
    sum
}

/// Compute the logarithmic derivative sum for the table side.
///
/// For a random challenge `beta`, computes:
///   sum_j multiplicity_j / (beta - table_value_j)
///
/// If the lookup is valid, this equals `log_derivative_trace_sum`.
pub fn log_derivative_table_sum<F: Field>(
    table_values: &[F],
    multiplicities: &LookupMultiplicities,
    beta: &F,
) -> F {
    let mut sum = F::zero();
    for (j, val) in table_values.iter().enumerate() {
        if multiplicities.counts[j] == 0 {
            continue; // Skip unused entries
        }
        let diff = beta.field_sub(val);
        let inv = diff.field_inv();
        let mult = F::from_u64(multiplicities.counts[j]);
        sum = sum.field_add(&mult.field_mul(&inv));
    }
    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::field_m31::Mersenne31Field;

    #[test]
    fn test_multiplicity_tracking() {
        let mut mult = LookupMultiplicities::new(256);
        mult.record(0);
        mult.record(0);
        mult.record(42);
        mult.record(255);

        assert_eq!(mult.total_lookups, 4);
        assert_eq!(mult.counts[0], 2);
        assert_eq!(mult.counts[42], 1);
        assert_eq!(mult.counts[255], 1);
        assert_eq!(mult.distinct_entries(), 3);
        assert!(mult.verify_consistency());
    }

    #[test]
    fn test_log_derivative_consistency() {
        // Simple test: 3-element table, trace uses entries 0 and 2
        let table: Vec<Mersenne31Field> = vec![
            Mersenne31Field::from_u64(10),
            Mersenne31Field::from_u64(20),
            Mersenne31Field::from_u64(30),
        ];
        let trace: Vec<Mersenne31Field> = vec![
            Mersenne31Field::from_u64(10), // entry 0
            Mersenne31Field::from_u64(30), // entry 2
            Mersenne31Field::from_u64(10), // entry 0 again
        ];

        let mut mult = LookupMultiplicities::new(3);
        mult.record(0); // 10
        mult.record(2); // 30
        mult.record(0); // 10

        let beta = Mersenne31Field::from_u64(42);
        let trace_sum = log_derivative_trace_sum(&trace, &beta);
        let table_sum = log_derivative_table_sum(&table, &mult, &beta);

        assert_eq!(
            trace_sum, table_sum,
            "log-derivative sums must match for valid lookup"
        );
    }

    #[test]
    fn test_lookup_native_gate() {
        let gate = LookupNativeGate {
            table_name: "relu".to_string(),
            input_bits: 8,
            output_bits: 8,
            entries: vec![(0, 0), (1, 1), (128, 0), (255, 0)], // simplified relu
        };

        assert_eq!(gate.table_size(), 4);
        assert_eq!(gate.lookup(1), Some(1));
        assert_eq!(gate.lookup(128), Some(0));
        assert_eq!(gate.lookup(99), None);
    }
}
