//! M55: Witness generator for ZK inference circuits.
//!
//! Executes a [`ZkIR`] program with concrete field-element inputs and records
//! every wire value.  The resulting [`Witness`] is passed to a [`ZkBackend`]
//! to generate a zero-knowledge proof.
//!
//! ## Design
//!
//! [`WitnessGenerator`] maintains an arena-allocated `wire_values` vector sized
//! to [`ZkIR::num_wires`] — a single allocation per `generate` call.  It then
//! interprets each [`ZkInstruction`] in order, writing output wire values
//! directly into the arena.  This is cache-friendly and avoids per-instruction
//! heap allocations.
//!
//! Lookup inputs are raw unsigned indices (matching how [`precompute_table`]
//! stores entries as `FieldElement::from_u64(i)`).  Callers must ensure that
//! the wire value fed into a [`ZkInstruction::Lookup`] is the raw table index,
//! not the signed interpretation.
//!
//! [`precompute_table`]: crate::zk::lookup::precompute_table

use super::backend::{Witness, ZkError};
use super::field::FieldElement;
use super::ir::{Wire, ZkIR, ZkInstruction};

// ---------------------------------------------------------------------------
// WitnessGenerator
// ---------------------------------------------------------------------------

/// Interprets a [`ZkIR`] program, assigning concrete field-element values to
/// every wire.
///
/// # Usage
///
/// ```rust,ignore
/// let mut gen = WitnessGenerator::new();
/// let witness = gen.generate(&ir, &public_inputs, &private_weights)?;
/// let output = witness.get(output_wire);
/// ```
pub struct WitnessGenerator {
    /// Arena: one slot per wire, indexed by [`Wire::0`].
    wire_values: Vec<FieldElement>,
}

impl WitnessGenerator {
    /// Create a new, empty witness generator.
    pub fn new() -> Self {
        WitnessGenerator {
            wire_values: Vec::new(),
        }
    }

    /// Execute `ir` with the given `inputs` (public) and `weights` (private),
    /// returning a fully-populated [`Witness`].
    ///
    /// # Arguments
    /// - `ir`:      the ZK-IR program to evaluate.
    /// - `inputs`:  public input values, assigned to `ir.public_inputs` in order.
    /// - `weights`: private weight values, assigned to `ir.private_inputs` in order.
    ///
    /// # Errors
    /// - [`ZkError::ProvingError`] if an [`ZkInstruction::AssertEq`] constraint
    ///   is violated, or if a [`ZkInstruction::Lookup`] input is not found in
    ///   the registered table.
    pub fn generate(
        &mut self,
        ir: &ZkIR,
        inputs: &[FieldElement],
        weights: &[FieldElement],
    ) -> Result<Witness, ZkError> {
        // 1. Pre-allocate wire arena (single allocation).
        self.wire_values = vec![FieldElement::zero(); ir.num_wires as usize];

        // 2. Assign public input wires.
        for (i, wire) in ir.public_inputs.iter().enumerate() {
            if i < inputs.len() {
                self.wire_values[wire.0 as usize] = inputs[i];
            }
        }

        // 3. Assign private weight wires.
        for (i, wire) in ir.private_inputs.iter().enumerate() {
            if i < weights.len() {
                self.wire_values[wire.0 as usize] = weights[i];
            }
        }

        // 4. Execute each instruction in evaluation order.
        for inst in &ir.instructions {
            self.execute_instruction(inst, ir)?;
        }

        // 5. Build the Witness from the populated arena.
        let public_inputs = ir
            .public_inputs
            .iter()
            .map(|w| (*w, self.wire_values[w.0 as usize]))
            .collect();
        let public_outputs = ir
            .public_outputs
            .iter()
            .map(|w| (*w, self.wire_values[w.0 as usize]))
            .collect();

        Ok(Witness {
            values: self.wire_values.clone(),
            public_inputs,
            public_outputs,
        })
    }

    /// Execute a single instruction, updating `self.wire_values`.
    fn execute_instruction(&mut self, inst: &ZkInstruction, ir: &ZkIR) -> Result<(), ZkError> {
        match inst {
            ZkInstruction::Mul { out, a, b } => {
                let va = self.wire_values[a.0 as usize];
                let vb = self.wire_values[b.0 as usize];
                self.wire_values[out.0 as usize] = va.mul(&vb);
            }

            ZkInstruction::Add { out, a, b } => {
                let va = self.wire_values[a.0 as usize];
                let vb = self.wire_values[b.0 as usize];
                self.wire_values[out.0 as usize] = va.add(&vb);
            }

            ZkInstruction::Const { out, value } => {
                self.wire_values[out.0 as usize] = *value;
            }

            ZkInstruction::AssertEq { a, b } => {
                let va = self.wire_values[a.0 as usize];
                let vb = self.wire_values[b.0 as usize];
                if va != vb {
                    return Err(ZkError::ProvingError(format!(
                        "AssertEq failed: wire {} ({:?}) != wire {} ({:?})",
                        a.0, va, b.0, vb
                    )));
                }
            }

            ZkInstruction::DotProduct { out, a, b } => {
                let mut acc = FieldElement::zero();
                for (wa, wb) in a.iter().zip(b.iter()) {
                    let va = self.wire_values[wa.0 as usize];
                    let vb = self.wire_values[wb.0 as usize];
                    acc = acc.add(&va.mul(&vb));
                }
                self.wire_values[out.0 as usize] = acc;
            }

            ZkInstruction::FixedMul {
                out,
                a,
                b,
                frac_bits,
            } => {
                let va = self.wire_values[a.0 as usize];
                let vb = self.wire_values[b.0 as usize];
                let product = va.mul(&vb);
                // Right-shift by frac_bits: multiply by the field inverse of 2^frac_bits.
                // This models quantised integer matmul rescaling in the field.
                let shift = FieldElement::from_u64(1u64 << frac_bits);
                let shift_inv = shift.inv();
                self.wire_values[out.0 as usize] = product.mul(&shift_inv);
            }

            ZkInstruction::Lookup {
                out,
                table,
                input,
                input_bits,
            } => {
                let input_val = self.wire_values[input.0 as usize];
                let key = (table.clone(), *input_bits);
                if let Some(lut) = ir.lookup_tables.get(&key) {
                    // Linear scan: tables are small (≤ 65 536 entries for 16-bit inputs).
                    let found = lut.entries.iter().find(|(k, _)| *k == input_val);
                    if let Some((_, v)) = found {
                        self.wire_values[out.0 as usize] = *v;
                    } else {
                        return Err(ZkError::ProvingError(format!(
                            "Lookup failed: input {:?} not found in table '{}' ({}bit)",
                            input_val, table, input_bits
                        )));
                    }
                } else {
                    return Err(ZkError::ProvingError(format!(
                        "Lookup table '{}' ({}bit) not registered in ZkIR",
                        table, input_bits
                    )));
                }
            }

            ZkInstruction::Requantize {
                out,
                input,
                scale,
                zero_point,
                target_bits: _,
            } => {
                // Affine transform: (input - zero_point) * scale
                // Clamping to [0, 2^target_bits) is a range constraint enforced
                // separately by the circuit backend; witness gen only stores the value.
                let vi = self.wire_values[input.0 as usize];
                let centered = vi.sub(zero_point);
                let scaled = centered.mul(scale);
                self.wire_values[out.0 as usize] = scaled;
            }

            ZkInstruction::Remap {
                out,
                input,
                permutation,
            } => {
                // Zero-cost bookkeeping: out[i] = input[permutation[i]].
                // Read all source values first to handle in-place aliasing safely.
                let src: Vec<FieldElement> = permutation
                    .iter()
                    .map(|&idx| {
                        if idx < input.len() {
                            self.wire_values[input[idx].0 as usize]
                        } else {
                            FieldElement::zero()
                        }
                    })
                    .collect();
                for (i, val) in src.into_iter().enumerate() {
                    if i < out.len() {
                        self.wire_values[out[i].0 as usize] = val;
                    }
                }
            }
        }
        Ok(())
    }

    /// Return the value of a single wire from the most recent `generate` call.
    ///
    /// Useful for diagnostics; prefer [`Witness::get`] in production code.
    ///
    /// # Panics
    /// Panics if `wire.0` is out of bounds or if `generate` has not been called.
    pub fn wire_value(&self, wire: Wire) -> FieldElement {
        self.wire_values[wire.0 as usize]
    }
}

impl Default for WitnessGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::field::FieldElement;
    use crate::zk::ir::{ZkIR, ZkInstruction};
    use crate::zk::lookup::precompute_table;

    // -----------------------------------------------------------------------
    // Basic arithmetic instructions
    // -----------------------------------------------------------------------

    #[test]
    fn witness_add() {
        let mut ir = ZkIR::new("test_add");
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

        let mut gen = WitnessGenerator::new();
        let witness = gen
            .generate(
                &ir,
                &[FieldElement::from_u64(3), FieldElement::from_u64(7)],
                &[],
            )
            .unwrap();
        assert_eq!(witness.get(w2), FieldElement::from_u64(10));
    }

    #[test]
    fn witness_mul() {
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

        let mut gen = WitnessGenerator::new();
        let witness = gen
            .generate(
                &ir,
                &[FieldElement::from_u64(6), FieldElement::from_u64(7)],
                &[],
            )
            .unwrap();
        assert_eq!(witness.get(w2), FieldElement::from_u64(42));
    }

    #[test]
    fn witness_const() {
        let mut ir = ZkIR::new("test_const");
        let w0 = ir.alloc_wire("input");
        let wc = ir.alloc_wire("const_42");
        let out = ir.alloc_wire("output");
        ir.push(ZkInstruction::Const {
            out: wc,
            value: FieldElement::from_u64(42),
        });
        ir.push(ZkInstruction::Add { out, a: w0, b: wc });
        ir.set_public_inputs(vec![w0]);
        ir.set_public_outputs(vec![out]);

        let mut gen = WitnessGenerator::new();
        let witness = gen
            .generate(&ir, &[FieldElement::from_u64(8)], &[])
            .unwrap();
        assert_eq!(witness.get(out), FieldElement::from_u64(50));
    }

    // -----------------------------------------------------------------------
    // DotProduct
    // -----------------------------------------------------------------------

    #[test]
    fn witness_dot_product() {
        let mut ir = ZkIR::new("test_dot");
        let a0 = ir.alloc_wire("a0");
        let a1 = ir.alloc_wire("a1");
        let b0 = ir.alloc_wire("b0");
        let b1 = ir.alloc_wire("b1");
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::DotProduct {
            out,
            a: vec![a0, a1],
            b: vec![b0, b1],
        });
        ir.set_public_inputs(vec![a0, a1, b0, b1]);
        ir.set_public_outputs(vec![out]);

        let mut gen = WitnessGenerator::new();
        // dot([2, 3], [4, 5]) = 2*4 + 3*5 = 8 + 15 = 23
        let inputs = [2u64, 3, 4, 5].map(FieldElement::from_u64);
        let witness = gen.generate(&ir, &inputs, &[]).unwrap();
        assert_eq!(witness.get(out), FieldElement::from_u64(23));
    }

    // -----------------------------------------------------------------------
    // Lookup — relu
    // -----------------------------------------------------------------------

    #[test]
    fn witness_relu_lookup() {
        // Lookup table keys are raw unsigned indices (from_u64(i)).
        // For an 8-bit signed table: -5 is stored at index 251 (= 256 - 5).
        //                              5 is stored at index 5.

        let mut ir = ZkIR::new("test_relu");
        let w0 = ir.alloc_wire("input");
        let w1 = ir.alloc_wire("output");
        ir.push(ZkInstruction::Lookup {
            out: w1,
            table: "relu".into(),
            input: w0,
            input_bits: 8,
        });
        ir.register_table(precompute_table("relu", 8, 8));
        ir.set_public_inputs(vec![w0]);
        ir.set_public_outputs(vec![w1]);

        let mut gen = WitnessGenerator::new();

        // relu(-5) = 0:  raw index for signed -5 in 8-bit two's complement is 251
        let neg5_idx = FieldElement::from_u64(251);
        let witness = gen.generate(&ir, &[neg5_idx], &[]).unwrap();
        assert_eq!(witness.get(w1), FieldElement::zero());

        // relu(5) = 5:  raw index for 5 is 5
        let pos5_idx = FieldElement::from_u64(5);
        let witness2 = gen.generate(&ir, &[pos5_idx], &[]).unwrap();
        assert_eq!(witness2.get(w1), FieldElement::from_u64(5));
    }

    // -----------------------------------------------------------------------
    // AssertEq — satisfied and violated
    // -----------------------------------------------------------------------

    #[test]
    fn witness_assert_eq_satisfied() {
        let mut ir = ZkIR::new("test_assert_ok");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        ir.push(ZkInstruction::AssertEq { a: w0, b: w1 });
        ir.set_public_inputs(vec![w0, w1]);

        let mut gen = WitnessGenerator::new();
        let v = FieldElement::from_u64(99);
        // Both wires get the same value — constraint satisfied.
        gen.generate(&ir, &[v, v], &[]).unwrap();
    }

    #[test]
    fn witness_assert_eq_violated() {
        let mut ir = ZkIR::new("test_assert_fail");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        ir.push(ZkInstruction::AssertEq { a: w0, b: w1 });
        ir.set_public_inputs(vec![w0, w1]);

        let mut gen = WitnessGenerator::new();
        let result = gen.generate(
            &ir,
            &[FieldElement::from_u64(1), FieldElement::from_u64(2)],
            &[],
        );
        assert!(
            matches!(result, Err(ZkError::ProvingError(_))),
            "AssertEq with different wires must return ProvingError"
        );
    }

    // -----------------------------------------------------------------------
    // FixedMul
    // -----------------------------------------------------------------------

    #[test]
    fn witness_fixed_mul_identity() {
        // (4 * 8) >> 0  = 32  (frac_bits = 0, shift = 1, inv(1) = 1)
        let mut ir = ZkIR::new("test_fixed_mul");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::FixedMul {
            out,
            a: w0,
            b: w1,
            frac_bits: 0,
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![out]);

        let mut gen = WitnessGenerator::new();
        let witness = gen
            .generate(
                &ir,
                &[FieldElement::from_u64(4), FieldElement::from_u64(8)],
                &[],
            )
            .unwrap();
        assert_eq!(witness.get(out), FieldElement::from_u64(32));
    }

    // -----------------------------------------------------------------------
    // Private inputs (weights)
    // -----------------------------------------------------------------------

    #[test]
    fn witness_private_weight_mul() {
        let mut ir = ZkIR::new("test_weight_mul");
        let input = ir.alloc_wire("input");
        let weight = ir.alloc_wire("weight");
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::Mul {
            out,
            a: input,
            b: weight,
        });
        ir.set_public_inputs(vec![input]);
        ir.private_inputs.push(weight);
        ir.set_public_outputs(vec![out]);

        let mut gen = WitnessGenerator::new();
        let witness = gen
            .generate(
                &ir,
                &[FieldElement::from_u64(5)], // public input
                &[FieldElement::from_u64(7)], // private weight
            )
            .unwrap();
        assert_eq!(witness.get(out), FieldElement::from_u64(35));
    }

    // -----------------------------------------------------------------------
    // Remap
    // -----------------------------------------------------------------------

    #[test]
    fn witness_remap_reverses() {
        let mut ir = ZkIR::new("test_remap");
        let w0 = ir.alloc_wire("w0");
        let w1 = ir.alloc_wire("w1");
        let w2 = ir.alloc_wire("w2");
        let w3 = ir.alloc_wire("w3");
        // Remap [w0, w1] -> [w2, w3] with permutation [1, 0] (reverse)
        ir.push(ZkInstruction::Remap {
            out: vec![w2, w3],
            input: vec![w0, w1],
            permutation: vec![1, 0],
        });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2, w3]);

        let mut gen = WitnessGenerator::new();
        let witness = gen
            .generate(
                &ir,
                &[FieldElement::from_u64(10), FieldElement::from_u64(20)],
                &[],
            )
            .unwrap();
        assert_eq!(witness.get(w2), FieldElement::from_u64(20));
        assert_eq!(witness.get(w3), FieldElement::from_u64(10));
    }

    // -----------------------------------------------------------------------
    // Witness accessors
    // -----------------------------------------------------------------------

    #[test]
    fn witness_public_output_values() {
        let mut ir = ZkIR::new("test_outputs");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::Add { out, a: w0, b: w1 });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![out]);

        let mut gen = WitnessGenerator::new();
        let witness = gen
            .generate(
                &ir,
                &[FieldElement::from_u64(100), FieldElement::from_u64(200)],
                &[],
            )
            .unwrap();
        assert_eq!(
            witness.public_output_values(),
            vec![FieldElement::from_u64(300)]
        );
    }

    #[test]
    fn wire_value_accessor() {
        let mut ir = ZkIR::new("test_wire_value");
        let w0 = ir.alloc_wire("x");
        let w1 = ir.alloc_wire("y");
        ir.push(ZkInstruction::Add {
            out: w1,
            a: w0,
            b: w0,
        });
        ir.set_public_inputs(vec![w0]);

        let mut gen = WitnessGenerator::new();
        gen.generate(&ir, &[FieldElement::from_u64(7)], &[])
            .unwrap();
        assert_eq!(gen.wire_value(w1), FieldElement::from_u64(14));
    }
}
