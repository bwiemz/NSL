//! M55: Circuit size estimation and reporting for ZK inference circuits.
//!
//! Analyses a compiled [`ZkIR`] and produces human-readable statistics about
//! constraint counts, gate breakdown, estimated proof size, proving time, and
//! witness memory requirements.
//!
//! ## Cost model
//!
//! The estimates are based on a Halo2/KZG backend:
//!
//! - **Constraints**: every [`ZkInstruction::Mul`] and inner product in
//!   [`ZkInstruction::DotProduct`] costs 1 constraint. [`ZkInstruction::FixedMul`]
//!   costs 2 constraints (mul + range check, matching the Halo2 `FixedMulGate`).
//!   [`ZkInstruction::AssertEq`] costs 1 constraint (linear equality).
//!   [`ZkInstruction::Requantize`] costs 1 constraint. Lookup-table constraints
//!   (1 per [`ZkInstruction::Lookup`]) are counted separately.
//! - **Add gates**: [`ZkInstruction::Add`] is absorbed into linear combinations
//!   in PLONKish arithmetization; zero constraints, counted for informational
//!   purposes only.
//! - **Proof size**: ~32 KiB baseline for KZG commitments + linear term.
//! - **Prove time**: ~0.3 ms per constraint (single-threaded CPU, BN254).
//! - **Verify time**: ~20 ms, essentially constant for Halo2.
//! - **Witness memory**: `num_wires × 32 bytes` (one BN254 field element per wire).

use super::ir::{ZkIR, ZkInstruction};

// ---------------------------------------------------------------------------
// CircuitStats
// ---------------------------------------------------------------------------

/// Summary statistics for a compiled ZK-IR circuit.
#[derive(Debug, Clone)]
pub struct CircuitStats {
    /// Total number of arithmetic constraints (mul gates + lookup gates +
    /// requantize gates). This is the primary cost driver for proof generation.
    pub num_constraints: u64,
    /// Number of pure multiplication gates (`Mul` + inner products from
    /// `DotProduct` + `FixedMul`). `Mul` costs 1 constraint; `FixedMul` costs
    /// 2 (mul + range check, matching Halo2 backend gates.rs cost model).
    pub num_mul_gates: u64,
    /// Number of addition gates (`Add`). Free in PLONKish — zero constraints.
    pub num_add_gates: u64,
    /// Number of `DotProduct` instructions. The mul gate count already includes
    /// the `len(a)` inner-product multiplications from each dot product.
    pub num_dot_product_gates: u64,
    /// Number of lookup-table evaluations (`Lookup`). Each contributes 1
    /// constraint via the lookup argument.
    pub num_lookup_gates: u64,
    /// Number of requantize operations. Each costs 1 constraint.
    pub num_requantize_gates: u64,
    /// Number of wires that appear in `public_inputs` or `public_outputs`.
    pub num_public_wires: u64,
    /// Number of wires that are not public.
    pub num_private_wires: u64,
    /// Summary of all registered lookup tables: `(name, input_bits, entry_count)`.
    pub lookup_table_summary: Vec<(String, u32, usize)>,
    /// Estimated proof size in bytes. Baseline 32 KiB for Halo2/KZG.
    pub estimated_proof_size_bytes: u64,
    /// Estimated single-threaded proving time in milliseconds.
    pub estimated_prove_time_ms: u64,
    /// Estimated verifier time in milliseconds. Near-constant for Halo2.
    pub estimated_verify_time_ms: u64,
    /// Memory required to store the full witness: `num_wires × 32 bytes`.
    pub estimated_witness_memory_bytes: u64,
}

// ---------------------------------------------------------------------------
// compute_stats
// ---------------------------------------------------------------------------

/// Walk a [`ZkIR`] and compute [`CircuitStats`].
///
/// The function is a single linear scan over the instruction list and the
/// registered lookup tables; it is O(|instructions| + |tables|).
pub fn compute_stats(ir: &ZkIR) -> CircuitStats {
    // Constraint counts match Halo2 backend (gates.rs) cost model
    let mut num_mul_gates: u64 = 0;
    let mut num_add_gates: u64 = 0;
    let mut num_dot_product_gates: u64 = 0;
    let mut num_lookup_gates: u64 = 0;
    let mut num_requantize_gates: u64 = 0;
    let mut num_assert_eq_constraints: u64 = 0;

    for instr in &ir.instructions {
        match instr {
            ZkInstruction::Mul { .. } => {
                num_mul_gates += 1;
            }
            ZkInstruction::Add { .. } => {
                num_add_gates += 1;
            }
            ZkInstruction::Const { .. } => {
                // No constraints generated.
            }
            ZkInstruction::AssertEq { .. } => {
                // Compiles to a linear constraint (a - b = 0); counts as 1 constraint.
                // Constraint counts match Halo2 backend (gates.rs) cost model
                num_assert_eq_constraints += 1;
            }
            ZkInstruction::DotProduct { a, .. } => {
                // Each inner product is `len` multiplications.
                num_mul_gates += a.len() as u64;
                num_dot_product_gates += 1;
            }
            ZkInstruction::FixedMul { .. } => {
                // 2 constraints: mul + range check, matching Halo2 backend (gates.rs) cost model
                num_mul_gates += 2;
            }
            ZkInstruction::Lookup { .. } => {
                num_lookup_gates += 1;
            }
            ZkInstruction::Requantize { .. } => {
                num_requantize_gates += 1;
            }
            ZkInstruction::Remap { .. } => {
                // Zero-cost bookkeeping — no constraints.
            }
        }
    }

    // Total constraints: multiplication gates + lookup argument gates + requantize gates
    // + linear equality constraints. Constraint counts match Halo2 backend (gates.rs) cost model.
    let num_constraints =
        num_mul_gates + num_lookup_gates + num_requantize_gates + num_assert_eq_constraints;

    // Public wire count: union of public_inputs and public_outputs.
    // Use a set to avoid double-counting wires that appear in both lists.
    let mut public_wire_ids = std::collections::HashSet::new();
    for w in &ir.public_inputs {
        public_wire_ids.insert(w.0);
    }
    for w in &ir.public_outputs {
        public_wire_ids.insert(w.0);
    }
    let num_public_wires = public_wire_ids.len() as u64;
    let num_private_wires = ir.num_wires.saturating_sub(num_public_wires);

    // Lookup table summary: collect (name, input_bits, entry_count) for each
    // registered table, sorted by name then input_bits for deterministic output.
    let mut lookup_table_summary: Vec<(String, u32, usize)> = ir
        .lookup_tables
        .values()
        .map(|tbl| (tbl.name.clone(), tbl.input_bits, tbl.entries.len()))
        .collect();
    lookup_table_summary.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));

    // Proof size estimate: 32 KiB baseline for Halo2/KZG group elements + a
    // small per-constraint term. The baseline dominates in practice.
    const PROOF_SIZE_BASE_BYTES: u64 = 32 * 1024; // 32 KiB
    const PROOF_SIZE_PER_CONSTRAINT_BYTES: u64 = 0; // negligible for KZG
    let estimated_proof_size_bytes =
        PROOF_SIZE_BASE_BYTES + num_constraints * PROOF_SIZE_PER_CONSTRAINT_BYTES;

    // Prove time: ~0.3 ms per constraint single-threaded.
    // Use integer arithmetic: constraints * 3 / 10 ms (rounds down).
    let estimated_prove_time_ms = num_constraints * 3 / 10;

    // Verify time: essentially constant at ~20 ms for Halo2/KZG.
    let estimated_verify_time_ms = 20;

    // Witness memory: one BN254 scalar (32 bytes) per wire.
    let estimated_witness_memory_bytes = ir.num_wires * 32;

    CircuitStats {
        num_constraints,
        num_mul_gates,
        num_add_gates,
        num_dot_product_gates,
        num_lookup_gates,
        num_requantize_gates,
        num_public_wires,
        num_private_wires,
        lookup_table_summary,
        estimated_proof_size_bytes,
        estimated_prove_time_ms,
        estimated_verify_time_ms,
        estimated_witness_memory_bytes,
    }
}

// ---------------------------------------------------------------------------
// format_stats
// ---------------------------------------------------------------------------

/// Format [`CircuitStats`] as a human-readable multi-line string.
///
/// Produces output in the following style:
///
/// ```text
/// Circuit: MyModel
///   Constraints:         1024
///   Mul gates:            640
///   Add gates:            128
///   Dot product gates:      8
///   Lookup gates:          32
///   Requantize gates:       0
///   Public wires:           5
///   Private wires:        995
///
///   Lookup tables:
///     relu    8-bit   256 entries
///     gelu    8-bit   256 entries
///
///   Estimated proof size:  32.0 KB
///   Estimated prove time:  307 ms
///   Estimated verify time:  20 ms
///   Estimated witness mem:  32.0 KB
/// ```
pub fn format_stats(stats: &CircuitStats, name: &str) -> String {
    let mut out = String::new();

    out.push_str(&format!("Circuit: {name}\n"));
    out.push_str(&format!(
        "  Constraints:     {:>12}\n",
        stats.num_constraints
    ));
    out.push_str(&format!(
        "  Mul gates:       {:>12}\n",
        stats.num_mul_gates
    ));
    out.push_str(&format!(
        "  Add gates:       {:>12}\n",
        stats.num_add_gates
    ));
    out.push_str(&format!(
        "  Dot product:     {:>12}\n",
        stats.num_dot_product_gates
    ));
    out.push_str(&format!(
        "  Lookup gates:    {:>12}\n",
        stats.num_lookup_gates
    ));
    out.push_str(&format!(
        "  Requantize:      {:>12}\n",
        stats.num_requantize_gates
    ));
    out.push_str(&format!(
        "  Public wires:    {:>12}\n",
        stats.num_public_wires
    ));
    out.push_str(&format!(
        "  Private wires:   {:>12}\n",
        stats.num_private_wires
    ));

    if !stats.lookup_table_summary.is_empty() {
        out.push('\n');
        out.push_str("  Lookup tables:\n");
        for (name, bits, count) in &stats.lookup_table_summary {
            out.push_str(&format!(
                "    {name:<12} {bits:>2}-bit  {count:>6} entries\n"
            ));
        }
    }

    out.push('\n');
    out.push_str(&format!(
        "  Est. proof size: {:>10}\n",
        format_bytes(stats.estimated_proof_size_bytes)
    ));
    out.push_str(&format!(
        "  Est. prove time: {:>10}\n",
        format_ms(stats.estimated_prove_time_ms)
    ));
    out.push_str(&format!(
        "  Est. verify time:{:>10}\n",
        format_ms(stats.estimated_verify_time_ms)
    ));
    out.push_str(&format!(
        "  Est. witness mem:{:>10}\n",
        format_bytes(stats.estimated_witness_memory_bytes)
    ));

    out
}

/// Format a byte count as a human-readable string (B / KB / MB).
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Format a millisecond duration as a human-readable string.
fn format_ms(ms: u64) -> String {
    if ms >= 1000 {
        format!("{:.2} s", ms as f64 / 1000.0)
    } else {
        format!("{ms} ms")
    }
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
    fn stats_counts_instructions() {
        let mut ir = ZkIR::new("test");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        let w3 = ir.alloc_wire("d");
        let w4 = ir.alloc_wire("e");
        ir.push(ZkInstruction::Mul { out: w2, a: w0, b: w1 });
        ir.push(ZkInstruction::Add { out: w3, a: w0, b: w2 });
        ir.push(ZkInstruction::Lookup {
            out: w4,
            table: "relu".into(),
            input: w3,
            input_bits: 8,
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w4]);

        let stats = compute_stats(&ir);
        assert_eq!(stats.num_mul_gates, 1);
        assert_eq!(stats.num_add_gates, 1);
        assert_eq!(stats.num_lookup_gates, 1);
        assert_eq!(stats.num_public_wires, 3); // 2 inputs + 1 output
        assert_eq!(stats.estimated_witness_memory_bytes, 5 * 32); // 5 wires * 32 bytes
    }

    #[test]
    fn format_stats_produces_readable_output() {
        let stats = CircuitStats {
            num_constraints: 100,
            num_mul_gates: 60,
            num_add_gates: 20,
            num_dot_product_gates: 10,
            num_lookup_gates: 30,
            num_requantize_gates: 0,
            num_public_wires: 5,
            num_private_wires: 95,
            lookup_table_summary: vec![("relu".into(), 8, 256)],
            estimated_proof_size_bytes: 32768,
            estimated_prove_time_ms: 30,
            estimated_verify_time_ms: 20,
            estimated_witness_memory_bytes: 3200,
        };
        let output = format_stats(&stats, "TestCircuit");
        assert!(output.contains("TestCircuit"));
        assert!(output.contains("100"));
    }

    #[test]
    fn constraints_equal_mul_plus_lookup_plus_requantize() {
        let mut ir = ZkIR::new("constraint_count");
        let w0 = ir.alloc_wire("x");
        let w1 = ir.alloc_wire("y");
        let w2 = ir.alloc_wire("z");
        let w3 = ir.alloc_wire("a");
        let w4 = ir.alloc_wire("b");
        let scale = FieldElement::from_u64(1);
        let zp = FieldElement::from_u64(0);
        ir.push(ZkInstruction::Mul { out: w2, a: w0, b: w1 });
        ir.push(ZkInstruction::Lookup {
            out: w3,
            table: "relu".into(),
            input: w2,
            input_bits: 8,
        });
        ir.push(ZkInstruction::Requantize {
            out: w4,
            input: w3,
            scale,
            zero_point: zp,
            target_bits: 8,
        });

        let stats = compute_stats(&ir);
        // 1 mul + 1 lookup + 1 requantize = 3 constraints
        assert_eq!(stats.num_constraints, 3);
        assert_eq!(stats.num_mul_gates, 1);
        assert_eq!(stats.num_lookup_gates, 1);
        assert_eq!(stats.num_requantize_gates, 1);
    }

    #[test]
    fn dot_product_contributes_inner_product_mul_count() {
        let mut ir = ZkIR::new("dot_test");
        let a_wires: Vec<Wire> = (0..4).map(|_| ir.alloc_wire("a")).collect();
        let b_wires: Vec<Wire> = (0..4).map(|_| ir.alloc_wire("b")).collect();
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::DotProduct {
            out,
            a: a_wires.clone(),
            b: b_wires,
        });

        let stats = compute_stats(&ir);
        // DotProduct of length 4 => 4 mul gates
        assert_eq!(stats.num_mul_gates, 4);
        assert_eq!(stats.num_dot_product_gates, 1);
        assert_eq!(stats.num_constraints, 4);
    }

    #[test]
    fn add_gates_are_not_counted_in_constraints() {
        let mut ir = ZkIR::new("add_test");
        let w0 = ir.alloc_wire("x");
        let w1 = ir.alloc_wire("y");
        let w2 = ir.alloc_wire("z");
        ir.push(ZkInstruction::Add { out: w2, a: w0, b: w1 });

        let stats = compute_stats(&ir);
        assert_eq!(stats.num_add_gates, 1);
        assert_eq!(stats.num_constraints, 0, "Add gates must not add constraints");
    }

    #[test]
    fn witness_memory_is_num_wires_times_32() {
        let mut ir = ZkIR::new("mem_test");
        for i in 0..10 {
            ir.alloc_wire(&format!("w{i}"));
        }

        let stats = compute_stats(&ir);
        assert_eq!(stats.estimated_witness_memory_bytes, 10 * 32);
    }

    #[test]
    fn public_wires_deduplicated_when_in_both_inputs_and_outputs() {
        let mut ir = ZkIR::new("dedup_test");
        let w0 = ir.alloc_wire("shared");
        let w1 = ir.alloc_wire("input_only");
        // w0 appears in both public_inputs and public_outputs
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w0]);

        let stats = compute_stats(&ir);
        // w0 + w1 = 2 unique public wires (w0 is not double-counted)
        assert_eq!(stats.num_public_wires, 2);
    }

    #[test]
    fn private_wires_are_total_minus_public() {
        let mut ir = ZkIR::new("private_test");
        let w0 = ir.alloc_wire("pub_in");
        let _w1 = ir.alloc_wire("private_weight");
        let w2 = ir.alloc_wire("pub_out");
        ir.set_public_inputs(vec![w0]);
        ir.set_public_outputs(vec![w2]);

        let stats = compute_stats(&ir);
        assert_eq!(stats.num_public_wires, 2); // w0, w2
        assert_eq!(stats.num_private_wires, 1); // w1
    }

    #[test]
    fn lookup_table_summary_is_sorted() {
        use crate::zk::lookup::LookupTable;
        let mut ir = ZkIR::new("sorted_tables");
        ir.register_table(LookupTable {
            name: "relu".into(),
            input_bits: 8,
            output_bits: 8,
            entries: vec![],
        });
        ir.register_table(LookupTable {
            name: "gelu".into(),
            input_bits: 8,
            output_bits: 8,
            entries: vec![],
        });

        let stats = compute_stats(&ir);
        // Expect alphabetical order: gelu, relu
        assert_eq!(stats.lookup_table_summary[0].0, "gelu");
        assert_eq!(stats.lookup_table_summary[1].0, "relu");
    }

    #[test]
    fn empty_ir_produces_zero_stats() {
        let ir = ZkIR::new("empty");
        let stats = compute_stats(&ir);
        assert_eq!(stats.num_constraints, 0);
        assert_eq!(stats.num_mul_gates, 0);
        assert_eq!(stats.num_add_gates, 0);
        assert_eq!(stats.num_lookup_gates, 0);
        assert_eq!(stats.num_public_wires, 0);
        assert_eq!(stats.num_private_wires, 0);
        assert_eq!(stats.estimated_witness_memory_bytes, 0);
    }

    #[test]
    fn format_bytes_helper() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.0 KB");
        assert_eq!(format_bytes(32 * 1024), "32.0 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.0 MB");
    }

    #[test]
    fn format_ms_helper() {
        assert_eq!(format_ms(0), "0 ms");
        assert_eq!(format_ms(20), "20 ms");
        assert_eq!(format_ms(1000), "1.00 s");
        assert_eq!(format_ms(2500), "2.50 s");
    }
}
