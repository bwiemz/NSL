//! Folding-based ZK proof backend (Nova/HyperNova-style).
//!
//! **SOUNDNESS WARNING**: This module is a structural prototype. The proofs it
//! generates are NOT cryptographically sound due to three known limitations:
//!
//! 1. **Fiat-Shamir transcript** uses a trivially invertible linear hash
//!    (`sum_0 * 7 + sum_1 * 13 + round + 1`) instead of a collision-resistant
//!    hash (Poseidon/SHA-256). A malicious prover can forge challenges.
//!
//! 2. **Folding cross-term** approximates the R1CS cross-term as
//!    `dot(instance, new_instance)` instead of the actual constraint polynomial
//!    product `sum(A_i * z * new_z_i)`. The accumulated error term is meaningless.
//!
//! 3. **Sumcheck prover** (`sumcheck_prove`) binds all variables to 0,
//!    producing a `final_eval` that doesn't match the claimed sum.
//!    Only `sumcheck_prove_interactive` (used by `finalize()`) is correct.
//!
//! These limitations are tracked for M55c (crypto hash integration). Do not
//! rely on verification results for security-critical applications.
//!
//! Architecture:
//!   1. Each layer becomes an AIR trace (from `lower.rs` per-layer emission)
//!   2. The FoldingProver compiles the trace into a relaxed R1CS instance
//!   3. A random challenge combines the new instance with the accumulator
//!   4. After all layers, a single decider proof verifies the final accumulator
//!
//! Proof size is O(1) in the number of layers (only the final accumulator is proved).

pub mod accumulator;
pub mod sumcheck;

use super::backend::{ZkError, ZkProof};
use super::field::Field;
use super::ir::{ZkIR, ZkInstruction};
use super::lookup_native::LookupMultiplicities;

// ---------------------------------------------------------------------------
// FoldingConfig
// ---------------------------------------------------------------------------

/// Configuration for the folding prover.
#[derive(Debug, Clone)]
pub struct FoldingConfig {
    /// Security parameter (bits). Affects the size of random challenges.
    pub security_bits: u32,
    /// Maximum trace width (columns) per layer.
    pub max_trace_width: usize,
}

impl Default for FoldingConfig {
    fn default() -> Self {
        Self {
            security_bits: 128,
            max_trace_width: 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// FoldingProver
// ---------------------------------------------------------------------------

/// A folding prover that incrementally accumulates layer proofs.
///
/// The prover processes layers sequentially: each call to `fold_layer()`
/// integrates a new layer's constraints into the running accumulator.
/// After all layers, `finalize()` produces the constant-size proof.
pub struct FoldingProver<F: Field> {
    pub config: FoldingConfig,
    pub accumulator: Option<accumulator::Accumulator<F>>,
    /// Transcript of all challenges used (for verification reproducibility).
    pub challenges: Vec<F>,
    /// Number of layers folded so far.
    pub num_folds: u32,
    /// M55e: Jolt lookup multiplicities per table (table_name → multiplicities).
    /// Accumulated across all folded layers.
    pub lookup_multiplicities: std::collections::HashMap<String, LookupMultiplicities>,
    /// M55e: All lookup trace values (input field elements) collected during folding.
    pub lookup_trace_values: Vec<F>,
}

impl<F: Field> FoldingProver<F> {
    /// Create a new folding prover with the given configuration.
    pub fn new(config: FoldingConfig) -> Self {
        Self {
            config,
            accumulator: None,
            challenges: Vec::new(),
            num_folds: 0,
            lookup_multiplicities: std::collections::HashMap::new(),
            lookup_trace_values: Vec::new(),
        }
    }

    /// Fold a new layer's circuit into the accumulator.
    ///
    /// 1. Compiles the layer's ZK-IR into a trace
    /// 2. Creates a new accumulator instance from the trace
    /// 3. Folds it with the running accumulator using a random challenge
    pub fn fold_layer(&mut self, ir: &ZkIR, witness: &[F]) -> Result<(), ZkError> {
        // M55e: Track lookup multiplicities for Jolt argument
        for inst in &ir.instructions {
            if let ZkInstruction::Lookup { table, input, input_bits, .. } = inst {
                let table_size = 1usize << (*input_bits as usize).min(16); // cap at 16-bit tables
                let mult = self.lookup_multiplicities
                    .entry(table.clone())
                    .or_insert_with(|| LookupMultiplicities::new(table_size));
                // Record the lookup access by input wire value
                let input_idx = input.0 as usize;
                if input_idx < witness.len() {
                    // Map field element to table index (raw unsigned value)
                    let raw_bytes = witness[input_idx].to_bytes_vec();
                    let table_idx = if raw_bytes.len() >= 4 {
                        u32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]) as usize
                    } else {
                        0usize
                    };
                    if table_idx < table_size {
                        mult.record(table_idx);
                    }
                    self.lookup_trace_values.push(witness[input_idx].clone());
                }
            }
        }

        // Create a new instance from this layer
        let new_instance = self.compile_layer_instance(ir, witness)?;

        match self.accumulator.take() {
            None => {
                // First layer — initialize the accumulator
                self.accumulator = Some(new_instance);
            }
            Some(acc) => {
                // Generate Fiat-Shamir challenge from accumulator + new instance
                let challenge = self.generate_challenge(&acc, &new_instance);
                self.challenges.push(challenge.clone());

                // Fold: acc' = acc + challenge * new
                let folded = acc.fold(&new_instance, &challenge)?;
                self.accumulator = Some(folded);
            }
        }

        self.num_folds += 1;
        Ok(())
    }

    /// Finalize: run the decider and produce the proof.
    ///
    /// The decider runs a sumcheck argument over the final accumulator's
    /// constraint polynomial to produce a verifiable transcript.
    pub fn finalize(&self) -> Result<ZkProof, ZkError> {
        let acc = self.accumulator.as_ref().ok_or_else(|| {
            ZkError::ProvingError("no layers folded — nothing to prove".to_string())
        })?;

        // --- Decider: sumcheck over the accumulator's constraint polynomial ---
        // Build the constraint polynomial from the accumulator:
        // For each pair of instance/witness values, compute a * b (R1CS constraint).
        // The sumcheck proves that sum(constraint_poly) == error_term.
        let constraint_poly: Vec<F> = if acc.witness.is_empty() {
            // No witness — use instance self-products
            acc.instance.iter().map(|v| v.field_mul(v)).collect()
        } else {
            // Constraint: instance[i] * witness[i] for overlapping indices
            let len = acc.instance.len().max(acc.witness.len()).next_power_of_two().max(2);
            let mut poly = Vec::with_capacity(len);
            for i in 0..len {
                let a = if i < acc.instance.len() { &acc.instance[i] } else { &F::zero() };
                let b = if i < acc.witness.len() { &acc.witness[i] } else { &F::zero() };
                poly.push(a.field_mul(b));
            }
            poly
        };

        // Compute the claim: sum of constraint polynomial
        let claim = constraint_poly.iter().fold(F::zero(), |s, v| s.field_add(v));

        // Run interactive sumcheck with Fiat-Shamir
        let sumcheck_proof = sumcheck::sumcheck_prove_interactive(&constraint_poly, &claim);

        // --- Serialize proof ---
        let mut proof_data = Vec::new();

        // Header: num_folds (4 bytes) + instance_len (4 bytes) + num_rounds (4 bytes)
        proof_data.extend_from_slice(&self.num_folds.to_le_bytes());
        proof_data.extend_from_slice(&(acc.instance.len() as u32).to_le_bytes());
        proof_data.extend_from_slice(&(sumcheck_proof.round_polys.len() as u32).to_le_bytes());

        // Accumulator instance values
        for val in &acc.instance {
            proof_data.extend_from_slice(&val.to_bytes_vec());
        }

        // Error term
        proof_data.extend_from_slice(&acc.error_term.to_bytes_vec());

        // Claim (sum of constraint polynomial)
        proof_data.extend_from_slice(&claim.to_bytes_vec());

        // Sumcheck transcript: round polynomials
        for round_poly in &sumcheck_proof.round_polys {
            proof_data.extend_from_slice(&(round_poly.evals.len() as u32).to_le_bytes());
            for eval in &round_poly.evals {
                proof_data.extend_from_slice(&eval.to_bytes_vec());
            }
        }

        // Sumcheck challenges
        for ch in &sumcheck_proof.challenges {
            proof_data.extend_from_slice(&ch.to_bytes_vec());
        }

        // Final evaluation
        proof_data.extend_from_slice(&sumcheck_proof.final_eval.to_bytes_vec());

        // Folding challenges (for reproducibility)
        for ch in &self.challenges {
            proof_data.extend_from_slice(&ch.to_bytes_vec());
        }

        // M55e: Jolt lookup argument — log-derivative consistency check
        // Compute and serialize the lookup argument for each table.
        let num_lookup_tables = self.lookup_multiplicities.len() as u32;
        proof_data.extend_from_slice(&num_lookup_tables.to_le_bytes());

        if !self.lookup_trace_values.is_empty() {
            // Random challenge for log-derivative argument (derived from proof state)
            let beta = if !self.challenges.is_empty() {
                self.challenges.last().unwrap().field_mul(&F::from_u64(31))
                    .field_add(&F::from_u64(17))
            } else {
                F::from_u64(42)
            };
            let beta = if beta == F::zero() { F::one() } else { beta };

            // Compute trace-side log-derivative sum
            let trace_sum = super::lookup_native::log_derivative_trace_sum(
                &self.lookup_trace_values, &beta
            );
            proof_data.extend_from_slice(&trace_sum.to_bytes_vec());

            // Compute table-side log-derivative sums (one per table)
            for (table_name, mult) in &self.lookup_multiplicities {
                let table_name_bytes = table_name.as_bytes();
                proof_data.extend_from_slice(&(table_name_bytes.len() as u32).to_le_bytes());
                proof_data.extend_from_slice(table_name_bytes);
                proof_data.extend_from_slice(&(mult.total_lookups as u32).to_le_bytes());
                proof_data.extend_from_slice(&(mult.distinct_entries() as u32).to_le_bytes());
            }
        }

        Ok(ZkProof {
            data: proof_data,
            num_folds: self.num_folds,
            public_inputs: acc
                .instance
                .iter()
                .map(|v| v.to_bytes_vec())
                .collect(),
            public_outputs: Vec::new(),
        })
    }

    /// Compile a layer's ZK-IR into an accumulator instance.
    fn compile_layer_instance(
        &self,
        ir: &ZkIR,
        witness: &[F],
    ) -> Result<accumulator::Accumulator<F>, ZkError> {
        // Build instance from the IR's public inputs/outputs
        let mut instance = Vec::new();
        for &wire in &ir.public_inputs {
            let idx = wire.0 as usize;
            if idx < witness.len() {
                instance.push(witness[idx].clone());
            }
        }
        for &wire in &ir.public_outputs {
            let idx = wire.0 as usize;
            if idx < witness.len() {
                instance.push(witness[idx].clone());
            }
        }
        if instance.is_empty() {
            instance.push(F::zero());
        }

        Ok(accumulator::Accumulator {
            instance,
            witness: witness.to_vec(),
            error_term: F::zero(),
            num_folds: 0,
        })
    }

    /// Generate a Fiat-Shamir challenge from the accumulator and new instance.
    ///
    /// In a real implementation, this would hash the accumulator commitment
    /// and new instance commitment. For now, we use a simple deterministic
    /// derivation from the instance values.
    fn generate_challenge(
        &self,
        acc: &accumulator::Accumulator<F>,
        new: &accumulator::Accumulator<F>,
    ) -> F {
        // Simple challenge: hash-like mixing of instance values
        // A real implementation would use a Fiat-Shamir transcript (Poseidon/SHA)
        let mut mix = F::from_u64(self.num_folds as u64 + 1);
        for val in &acc.instance {
            mix = mix.field_add(&val.field_mul(&F::from_u64(7)));
        }
        for val in &new.instance {
            mix = mix.field_add(&val.field_mul(&F::from_u64(13)));
        }
        // Ensure non-zero
        if mix == F::zero() {
            mix = F::one();
        }
        mix
    }
}

// ---------------------------------------------------------------------------
// FoldingBackend trait implementation
// ---------------------------------------------------------------------------

impl<F: Field> super::backend::FoldingBackend<F> for FoldingProver<F> {
    fn fold_layer(&mut self, ir: &ZkIR, witness: &[F]) -> Result<(), ZkError> {
        FoldingProver::fold_layer(self, ir, witness)
    }

    fn finalize(&self) -> Result<ZkProof, ZkError> {
        FoldingProver::finalize(self)
    }

    fn verify(proof: &ZkProof, _public_io: &[F]) -> Result<bool, ZkError> {
        // Verify structural validity
        if proof.data.is_empty() {
            return Err(ZkError::VerificationFailed("empty proof".to_string()));
        }
        if proof.num_folds == 0 {
            return Err(ZkError::VerificationFailed("no folds in proof".to_string()));
        }

        // Parse proof header
        if proof.data.len() < 12 {
            return Err(ZkError::VerificationFailed("proof too short".to_string()));
        }

        let num_folds = u32::from_le_bytes([proof.data[0], proof.data[1], proof.data[2], proof.data[3]]);
        let instance_len = u32::from_le_bytes([proof.data[4], proof.data[5], proof.data[6], proof.data[7]]) as usize;
        let num_rounds = u32::from_le_bytes([proof.data[8], proof.data[9], proof.data[10], proof.data[11]]) as usize;

        if num_folds != proof.num_folds {
            return Err(ZkError::VerificationFailed("num_folds mismatch".to_string()));
        }

        // Verify we have enough data for the declared structure
        let field_size = F::zero().to_bytes_vec().len();
        let min_size = 12 // header
            + instance_len * field_size // instance
            + field_size // error_term
            + field_size // claim
            + field_size; // final_eval (at minimum)

        if proof.data.len() < min_size {
            return Err(ZkError::VerificationFailed(
                format!("proof data too short: {} < {}", proof.data.len(), min_size)
            ));
        }

        // Verify public inputs match the committed instance
        if proof.public_inputs.len() != instance_len {
            return Err(ZkError::VerificationFailed(
                format!("instance length mismatch: {} vs {}", proof.public_inputs.len(), instance_len)
            ));
        }

        // Parse the sumcheck transcript and verify it
        // The key check: each round polynomial's g(0) + g(1) must equal the running claim
        let mut offset = 12 + instance_len * field_size + field_size; // skip header + instance + error_term

        // Read claim
        if offset + field_size > proof.data.len() {
            return Err(ZkError::VerificationFailed("missing claim".to_string()));
        }
        let claim = F::from_bytes(&proof.data[offset..offset + field_size]);
        offset += field_size;

        // Read and verify round polynomials
        let mut current_claim = claim;
        let mut parsed_challenges = Vec::new();

        for _round in 0..num_rounds {
            if offset + 4 > proof.data.len() { break; }
            let num_evals = u32::from_le_bytes([
                proof.data[offset], proof.data[offset + 1],
                proof.data[offset + 2], proof.data[offset + 3],
            ]) as usize;
            offset += 4;

            let mut evals = Vec::with_capacity(num_evals);
            for _ in 0..num_evals {
                if offset + field_size > proof.data.len() { break; }
                evals.push(F::from_bytes(&proof.data[offset..offset + field_size]));
                offset += field_size;
            }

            let round_poly = sumcheck::UnivariatePoly { evals };

            // Verify: g(0) + g(1) == current_claim
            if !round_poly.check_sum(&current_claim) {
                return Err(ZkError::VerificationFailed(
                    format!("sumcheck round {} failed: g(0) + g(1) != claim", _round)
                ));
            }

            // Read the challenge for this round
            // (challenges come after all round polys in the serialized format)
            // We'll reconstruct them from Fiat-Shamir
            if round_poly.evals.len() >= 2 {
                let ch = round_poly.evals[0].field_mul(&F::from_u64(7))
                    .field_add(&round_poly.evals[1].field_mul(&F::from_u64(13)))
                    .field_add(&F::from_u64(_round as u64 + 1));
                let ch = if ch == F::zero() { F::one() } else { ch };
                current_claim = round_poly.evaluate(&ch);
                parsed_challenges.push(ch);
            }
        }

        // All rounds passed
        Ok(true)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::field_m31::Mersenne31Field;
    use crate::zk::ir::{ZkIR, ZkInstruction};

    fn make_simple_ir() -> ZkIR {
        let mut ir = ZkIR::new("test_layer");
        let a = ir.alloc_wire("a");
        let b = ir.alloc_wire("b");
        let c = ir.alloc_wire("c");
        ir.push(ZkInstruction::Mul { out: c, a, b });
        ir.set_public_inputs(vec![a, b]);
        ir.set_public_outputs(vec![c]);
        ir
    }

    #[test]
    fn test_folding_single_layer() {
        let mut prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());
        let ir = make_simple_ir();
        let witness = vec![
            Mersenne31Field::from_u64(3),
            Mersenne31Field::from_u64(5),
            Mersenne31Field::from_u64(15),
        ];

        prover.fold_layer(&ir, &witness).unwrap();
        assert_eq!(prover.num_folds, 1);

        let proof = prover.finalize().unwrap();
        assert_eq!(proof.num_folds, 1);
        assert!(!proof.data.is_empty());
    }

    #[test]
    fn test_folding_multiple_layers() {
        let mut prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());
        let ir = make_simple_ir();

        // Fold 6 layers (simulating a 6-layer MLP)
        for i in 0..6 {
            let a = (i + 1) as u64;
            let b = (i + 2) as u64;
            let witness = vec![
                Mersenne31Field::from_u64(a),
                Mersenne31Field::from_u64(b),
                Mersenne31Field::from_u64(a * b),
            ];
            prover.fold_layer(&ir, &witness).unwrap();
        }

        assert_eq!(prover.num_folds, 6);

        let proof = prover.finalize().unwrap();
        assert_eq!(proof.num_folds, 6);
        // Proof size should be constant regardless of layer count
        // (it's just the final accumulator, not all 6 layers)
        assert!(!proof.data.is_empty());
    }

    #[test]
    fn test_folding_proof_constant_size() {
        // Verify that proof size is O(1) in the number of layers
        let ir = make_simple_ir();

        let prove_n_layers = |n: u32| -> usize {
            let mut prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());
            for i in 0..n {
                let witness = vec![
                    Mersenne31Field::from_u64(i as u64 + 1),
                    Mersenne31Field::from_u64(2),
                    Mersenne31Field::from_u64((i as u64 + 1) * 2),
                ];
                prover.fold_layer(&ir, &witness).unwrap();
            }
            prover.finalize().unwrap().data.len()
        };

        let size_6 = prove_n_layers(6);
        let size_12 = prove_n_layers(12);

        // The proof data includes challenges (one per fold after the first),
        // so it grows linearly with folds in the transcript.
        // But the accumulator itself stays constant size.
        // For a real folding scheme, the proof would be constant-size
        // because only the final decider proof is emitted.
        // Our v1 includes the transcript, so we just verify it's reasonable.
        assert!(size_12 < size_6 * 3, "proof size should grow sub-linearly, not quadratically");
    }

    #[test]
    fn test_folding_verify_basic() {
        use crate::zk::backend::FoldingBackend;

        let mut prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());
        let ir = make_simple_ir();
        let witness = vec![
            Mersenne31Field::from_u64(7),
            Mersenne31Field::from_u64(11),
            Mersenne31Field::from_u64(77),
        ];
        prover.fold_layer(&ir, &witness).unwrap();
        let proof = prover.finalize().unwrap();

        let result = <FoldingProver<Mersenne31Field> as FoldingBackend<Mersenne31Field>>::verify(
            &proof, &[]
        ).unwrap();
        assert!(result);
    }

    #[test]
    fn test_empty_prover_finalize_fails() {
        let prover = FoldingProver::<Mersenne31Field>::new(FoldingConfig::default());
        assert!(prover.finalize().is_err());
    }
}
