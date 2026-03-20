//! M55: Halo2 ZK backend for NSL inference circuits.
//!
//! Implements the [`ZkBackend`] trait using the Halo2 proof system (PSE fork).
//! The Halo2 backend targets PLONKish arithmetization with KZG polynomial
//! commitments over the BN254 curve, producing proofs that are efficiently
//! verifiable on-chain (EVM-compatible).
//!
//! ## Architecture
//!
//! - **Custom gates** ([`gates`]): Define the constraint structure for ML-specific
//!   operations (dot product, fixed-point multiply, requantize, layer norm).
//! - **Circuit compiler**: Translates [`ZkIR`] instructions into a
//!   [`CompiledCircuit`] with the correct `k` sizing parameter.
//! - **Proof generation/verification**: Stubbed pending full Halo2 integration
//!   (returns informative errors).
//!
//! ## Halo2 dependency
//!
//! The `halo2_proofs` crate is an optional dependency (feature-gated). The
//! circuit compilation logic (gate analysis, constraint counting, `k` selection)
//! works without it. Full proof generation requires the feature to be enabled.

pub mod gates;

use super::backend::{CompiledCircuit, Witness, ZkBackend, ZkConfig, ZkCurve, ZkError, ZkIR};
use super::field::FieldElement;
use super::ir::ZkInstruction;

// ---------------------------------------------------------------------------
// Halo2Backend
// ---------------------------------------------------------------------------

/// Halo2 PLONKish proof backend.
///
/// Compiles ZK-IR into arithmetic circuits sized for the Halo2 proof system.
/// The `curve` field selects the elliptic curve for polynomial commitments;
/// BN254 is recommended for EVM verification, BLS12-381 for higher security.
#[derive(Debug)]
pub struct Halo2Backend {
    /// Elliptic curve used for polynomial commitments.
    pub curve: ZkCurve,
}

impl Halo2Backend {
    /// Create a new Halo2 backend targeting the given elliptic curve.
    pub fn new(curve: ZkCurve) -> Self {
        Self { curve }
    }

    /// Count the total number of PLONKish constraints for the given IR.
    ///
    /// This mirrors the logic in `stats::compute_stats` but is used here to
    /// determine the circuit size parameter `k`.
    fn count_constraints(ir: &ZkIR) -> u64 {
        let mut total: u64 = 0;
        for instr in &ir.instructions {
            match instr {
                ZkInstruction::Mul { .. } => total += 1,
                ZkInstruction::Add { .. } => {
                    // Free in PLONKish — absorbed into linear combination.
                }
                ZkInstruction::Const { .. } => {
                    // Fixed column, no constraint row.
                }
                ZkInstruction::AssertEq { .. } => {
                    // Linear constraint (a - b = 0), costs 1 row.
                    total += 1;
                }
                ZkInstruction::DotProduct { a, .. } => {
                    // One multiplication constraint per element pair.
                    total += a.len() as u64;
                }
                ZkInstruction::FixedMul { .. } => {
                    // Mul + range check = 2 constraints (via FixedMulGate).
                    total += 2;
                }
                ZkInstruction::Lookup { .. } => {
                    // 1 lookup argument constraint.
                    total += 1;
                }
                ZkInstruction::Requantize { .. } => {
                    // 3 constraints (via RequantizeGate).
                    total += 3;
                }
                ZkInstruction::Remap { .. } => {
                    // Zero cost — pure bookkeeping.
                }
            }
        }
        total
    }

    /// Compute the minimum `k` such that `2^k >= constraints + blinding_rows`.
    ///
    /// Halo2 requires the circuit table to have `2^k` rows. We add a small
    /// margin for blinding rows (used for zero-knowledge) and public input
    /// rows. The minimum `k` is 1 (2 rows) even for trivial circuits.
    fn compute_k(constraints: u64, num_public_io: u64) -> u32 {
        // Blinding rows: Halo2 typically needs 5 blinding rows for ZK.
        const BLINDING_ROWS: u64 = 5;
        // Public inputs need their own rows in Halo2.
        let total_rows = constraints + num_public_io + BLINDING_ROWS;

        // Find minimum k where 2^k >= total_rows, with k >= 1.
        let mut k: u32 = 1;
        while (1u64 << k) < total_rows {
            k += 1;
            if k >= 28 {
                // Safety cap: 2^28 = 268M rows is the practical limit.
                break;
            }
        }
        k
    }
}

impl ZkBackend for Halo2Backend {
    type ProvingKey = Halo2ProvingKey;
    type VerificationKey = Halo2VerificationKey;
    type Proof = Halo2Proof;

    fn compile(&self, ir: &ZkIR, config: &ZkConfig) -> Result<CompiledCircuit, ZkError> {
        let constraints = Self::count_constraints(ir);

        // Check constraint limit if configured.
        if let Some(max) = config.max_constraints {
            if constraints > max {
                return Err(ZkError::InvalidConfig(format!(
                    "circuit has {constraints} constraints, exceeds limit of {max}"
                )));
            }
        }

        let num_public_io =
            ir.public_inputs.len() as u64 + ir.public_outputs.len() as u64;
        let k = Self::compute_k(constraints, num_public_io);

        Ok(CompiledCircuit {
            ir: ZkIR {
                name: ir.name.clone(),
                instructions: ir.instructions.clone(),
                public_inputs: ir.public_inputs.clone(),
                public_outputs: ir.public_outputs.clone(),
                private_inputs: ir.private_inputs.clone(),
                num_wires: ir.num_wires,
                lookup_tables: ir.lookup_tables.clone(),
                wire_names: ir.wire_names.clone(),
            },
            k,
        })
    }

    fn setup(
        &self,
        _circuit: &CompiledCircuit,
    ) -> Result<(Self::ProvingKey, Self::VerificationKey), ZkError> {
        // Halo2 key generation requires the full circuit layout, which is
        // completed in a follow-up task (real halo2 synthesis).
        Err(ZkError::BackendNotImplemented {
            backend: "halo2",
            message: "Halo2 key generation requires full circuit synthesis (M55 follow-up)",
        })
    }

    fn prove(
        &self,
        _pk: &Self::ProvingKey,
        _witness: &Witness,
    ) -> Result<Self::Proof, ZkError> {
        Err(ZkError::BackendNotImplemented {
            backend: "halo2",
            message: "Halo2 proof generation requires full circuit synthesis (M55 follow-up)",
        })
    }

    fn verify(
        &self,
        _vk: &Self::VerificationKey,
        _proof: &Self::Proof,
        _public_inputs: &[FieldElement],
    ) -> Result<bool, ZkError> {
        Err(ZkError::BackendNotImplemented {
            backend: "halo2",
            message: "Halo2 verification requires full circuit synthesis (M55 follow-up)",
        })
    }

    fn estimate_proof_size(&self, circuit: &CompiledCircuit) -> u64 {
        // Halo2/KZG proof size is roughly constant at ~32 KiB (dominated by
        // group element commitments). A small linear term accounts for the
        // number of advice columns.
        const BASE_BYTES: u64 = 32 * 1024; // 32 KiB
        // Each advice commitment is ~64 bytes (2 curve points).
        // Estimate 4-6 advice columns for a typical ML circuit.
        let advice_commitment_bytes = 6 * 64;
        // Opening proof: ~log2(2^k) * 64 bytes for the IPA/KZG opening.
        let opening_bytes = (circuit.k as u64) * 64;
        BASE_BYTES + advice_commitment_bytes + opening_bytes
    }

    fn emit_solidity(&self, _vk: &Self::VerificationKey) -> Option<String> {
        // Solidity verifier generation is planned but not yet implemented.
        None
    }
}

// ---------------------------------------------------------------------------
// Key and Proof types (stubs for associated types)
// ---------------------------------------------------------------------------

/// Halo2 proving key (stub).
///
/// Will hold the actual `halo2_proofs::plonk::ProvingKey` when full
/// circuit synthesis is implemented.
#[derive(Debug)]
pub struct Halo2ProvingKey {
    /// Circuit size parameter for this key.
    pub k: u32,
}

/// Halo2 verification key (stub).
///
/// Will hold the actual `halo2_proofs::plonk::VerifyingKey` when full
/// circuit synthesis is implemented.
#[derive(Debug)]
pub struct Halo2VerificationKey {
    /// Circuit size parameter for this key.
    pub k: u32,
}

/// Halo2 proof (stub).
///
/// Will hold the serialized proof bytes when full proving is implemented.
#[derive(Debug)]
pub struct Halo2Proof {
    /// Raw proof bytes.
    pub bytes: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::backend::ZkConfig;
    use crate::zk::ir::{Wire, ZkIR, ZkInstruction};

    #[test]
    fn halo2_backend_compiles_simple_add() {
        let mut ir = ZkIR::new("test_add");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("out");
        ir.push(ZkInstruction::Add { out: w2, a: w0, b: w1 });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2]);

        let backend = Halo2Backend::new(ZkCurve::BN254);
        let result = backend.compile(&ir, &ZkConfig::default());
        assert!(result.is_ok());
        let circuit = result.unwrap();
        // Add is free, but we still need rows for public IO + blinding.
        // 0 constraints + 3 public wires + 5 blinding = 8 => k=3 (2^3=8)
        assert!(circuit.k >= 1);
        assert_eq!(circuit.ir.name, "test_add");
    }

    #[test]
    fn halo2_backend_compiles_mul() {
        let mut ir = ZkIR::new("test_mul");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("out");
        ir.push(ZkInstruction::Mul { out: w2, a: w0, b: w1 });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2]);

        let backend = Halo2Backend::new(ZkCurve::BN254);
        let result = backend.compile(&ir, &ZkConfig::default());
        assert!(result.is_ok());
        let circuit = result.unwrap();
        // 1 mul constraint + 3 public IO + 5 blinding = 9 => k=4 (2^4=16)
        assert!(circuit.k >= 1);
    }

    #[test]
    fn halo2_backend_compiles_dot_product() {
        let mut ir = ZkIR::new("test_dot");
        let a: Vec<Wire> = (0..128).map(|i| ir.alloc_wire(&format!("a{i}"))).collect();
        let b: Vec<Wire> = (0..128).map(|i| ir.alloc_wire(&format!("b{i}"))).collect();
        let out = ir.alloc_wire("out");
        ir.push(ZkInstruction::DotProduct {
            out,
            a: a.clone(),
            b: b.clone(),
        });
        ir.set_public_inputs(a[..2].to_vec());
        ir.set_public_outputs(vec![out]);

        let backend = Halo2Backend::new(ZkCurve::BN254);
        let result = backend.compile(&ir, &ZkConfig::default());
        assert!(result.is_ok());
        let circuit = result.unwrap();
        // 128 constraints + 3 public + 5 blinding = 136 => k=8 (2^8=256)
        assert!(circuit.k >= 8);
    }

    #[test]
    fn halo2_backend_respects_max_constraints() {
        let mut ir = ZkIR::new("test_limit");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("out");
        ir.push(ZkInstruction::Mul { out: w2, a: w0, b: w1 });

        let mut config = ZkConfig::default();
        config.max_constraints = Some(0); // Limit to 0 constraints

        let backend = Halo2Backend::new(ZkCurve::BN254);
        let result = backend.compile(&ir, &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("exceeds limit"), "error: {msg}");
    }

    #[test]
    fn halo2_backend_k_is_at_least_1() {
        // Even an empty circuit should have k >= 1.
        let ir = ZkIR::new("empty");
        let backend = Halo2Backend::new(ZkCurve::BN254);
        let result = backend.compile(&ir, &ZkConfig::default());
        assert!(result.is_ok());
        let circuit = result.unwrap();
        assert!(circuit.k >= 1);
    }

    #[test]
    fn halo2_backend_compute_k_correctness() {
        // 0 constraints, 0 public IO: 0 + 0 + 5 = 5 rows => k=3 (2^3=8)
        assert_eq!(Halo2Backend::compute_k(0, 0), 3);
        // 1 constraint, 0 public: 1 + 0 + 5 = 6 => k=3
        assert_eq!(Halo2Backend::compute_k(1, 0), 3);
        // 4 constraints, 0 public: 4 + 0 + 5 = 9 => k=4 (2^4=16)
        assert_eq!(Halo2Backend::compute_k(4, 0), 4);
        // 128 constraints, 3 public: 128 + 3 + 5 = 136 => k=8 (2^8=256)
        assert_eq!(Halo2Backend::compute_k(128, 3), 8);
    }

    #[test]
    fn halo2_backend_count_constraints_mixed() {
        let mut ir = ZkIR::new("mixed");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("c");
        let w3 = ir.alloc_wire("d");
        let w4 = ir.alloc_wire("e");
        let w5 = ir.alloc_wire("f");

        // 1 mul = 1
        ir.push(ZkInstruction::Mul { out: w2, a: w0, b: w1 });
        // 1 add = 0 (free)
        ir.push(ZkInstruction::Add { out: w3, a: w0, b: w2 });
        // 1 fixed_mul = 2
        ir.push(ZkInstruction::FixedMul {
            out: w4,
            a: w2,
            b: w3,
            frac_bits: 8,
        });
        // 1 lookup = 1
        ir.push(ZkInstruction::Lookup {
            out: w5,
            table: "relu".into(),
            input: w4,
            input_bits: 8,
        });

        // Total: 1 + 0 + 2 + 1 = 4
        assert_eq!(Halo2Backend::count_constraints(&ir), 4);
    }

    #[test]
    fn halo2_backend_count_constraints_with_requantize() {
        use crate::zk::field::FieldElement;

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

        // Requantize = 3 constraints
        assert_eq!(Halo2Backend::count_constraints(&ir), 3);
    }

    #[test]
    fn halo2_backend_estimate_proof_size() {
        let mut ir = ZkIR::new("proof_size");
        let w0 = ir.alloc_wire("a");
        let w1 = ir.alloc_wire("b");
        let w2 = ir.alloc_wire("out");
        ir.push(ZkInstruction::Mul { out: w2, a: w0, b: w1 });
        ir.set_public_inputs(vec![w0, w1]);
        ir.set_public_outputs(vec![w2]);

        let backend = Halo2Backend::new(ZkCurve::BN254);
        let circuit = backend.compile(&ir, &ZkConfig::default()).unwrap();
        let proof_size = backend.estimate_proof_size(&circuit);
        // Should be at least 32 KiB baseline.
        assert!(proof_size >= 32 * 1024);
    }

    #[test]
    fn halo2_backend_setup_returns_not_implemented() {
        let ir = ZkIR::new("test");
        let backend = Halo2Backend::new(ZkCurve::BN254);
        let circuit = backend.compile(&ir, &ZkConfig::default()).unwrap();
        let result = backend.setup(&circuit);
        assert!(result.is_err());
    }

    #[test]
    fn halo2_backend_bn254_and_bls12_381() {
        let ir = ZkIR::new("curve_test");
        let config = ZkConfig::default();

        let bn254 = Halo2Backend::new(ZkCurve::BN254);
        let bls = Halo2Backend::new(ZkCurve::BLS12_381);

        // Both should compile the same IR successfully.
        assert!(bn254.compile(&ir, &config).is_ok());
        assert!(bls.compile(&ir, &config).is_ok());
    }

    #[test]
    fn halo2_backend_preserves_ir_in_compiled_circuit() {
        let mut ir = ZkIR::new("preserve_test");
        let w0 = ir.alloc_wire("x");
        let w1 = ir.alloc_wire("y");
        ir.push(ZkInstruction::Add { out: w1, a: w0, b: w0 });
        ir.set_public_inputs(vec![w0]);
        ir.set_public_outputs(vec![w1]);

        let backend = Halo2Backend::new(ZkCurve::BN254);
        let circuit = backend.compile(&ir, &ZkConfig::default()).unwrap();

        assert_eq!(circuit.ir.name, "preserve_test");
        assert_eq!(circuit.ir.num_wires, 2);
        assert_eq!(circuit.ir.instructions.len(), 1);
        assert_eq!(circuit.ir.public_inputs.len(), 1);
        assert_eq!(circuit.ir.public_outputs.len(), 1);
    }
}
