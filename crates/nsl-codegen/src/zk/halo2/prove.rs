//! Halo2 proof generation — v1 placeholder with correct data flow.
//!
//! This module implements the proof generation pipeline for the Halo2 backend.
//! In v1 the proof is a serialized encoding of the public outputs that exercises
//! the full data flow (IR -> witness -> proof bytes). Real Halo2 proof generation
//! (implementing the `Circuit` trait, calling `create_proof()`) is deferred to a
//! follow-up task.
//!
//! ## Proof format (v1)
//!
//! ```text
//! [0..16]   "NSL_ZK_PROOF_V1\0"   magic header
//! [16..20]  num_outputs (u32 LE)   number of public output field elements
//! [20..]    outputs × 32 bytes     each public output as 32-byte LE limbs
//! ```

use super::super::backend::{Witness, ZkConfig, ZkError};
use super::super::field::FieldElement;
use super::super::ir::ZkIR;

/// Magic header for v1 placeholder proofs.
pub const PROOF_MAGIC: &[u8; 16] = b"NSL_ZK_PROOF_V1\0";

/// Size of the serialized header: 16 bytes magic + 4 bytes output count.
const HEADER_SIZE: usize = 20;

/// Generate a v1 placeholder proof from a witness and its ZK-IR.
///
/// The proof encodes the public output field elements so that `verify_proof`
/// can check them. This exercises the full pipeline (compile -> witness -> prove
/// -> verify) even before real Halo2 circuit synthesis is wired up.
///
/// # Arguments
/// - `_ir`: the ZK-IR (unused in v1, needed for the real prover).
/// - `witness`: the fully-populated witness from [`WitnessGenerator`].
/// - `_config`: compilation config (unused in v1).
///
/// # Errors
/// Returns [`ZkError::ProvingError`] if the witness has no public outputs.
pub fn generate_proof(
    _ir: &ZkIR,
    witness: &Witness,
    _config: &ZkConfig,
) -> Result<Vec<u8>, ZkError> {
    // WARNING: v1 placeholder — does NOT provide zero-knowledge security.
    // Proofs can be trivially forged. Replace with real Halo2 circuit synthesis.
    let outputs = &witness.public_outputs;
    if outputs.is_empty() {
        return Err(ZkError::ProvingError(
            "cannot generate proof: witness has no public outputs".into(),
        ));
    }

    let num_outputs = outputs.len() as u32;
    let mut proof = Vec::with_capacity(HEADER_SIZE + outputs.len() * 32);

    // Write magic header.
    proof.extend_from_slice(PROOF_MAGIC);

    // Write output count (little-endian u32).
    proof.extend_from_slice(&num_outputs.to_le_bytes());

    // Write each public output as 32 bytes (4 × u64 little-endian limbs).
    for (_, val) in outputs {
        proof.extend_from_slice(&val.to_bytes());
    }

    Ok(proof)
}

/// Extract the public output field elements from a v1 proof.
///
/// Returns the outputs in the order they were written by `generate_proof`.
///
/// # Errors
/// Returns [`ZkError::VerificationFailed`] if the proof is too short or has
/// an invalid header.
pub fn extract_outputs(proof: &[u8]) -> Result<Vec<FieldElement>, ZkError> {
    if proof.len() < HEADER_SIZE {
        return Err(ZkError::VerificationFailed(
            "proof too short for header".into(),
        ));
    }
    if &proof[..16] != PROOF_MAGIC.as_slice() {
        return Err(ZkError::VerificationFailed(
            "invalid proof magic header".into(),
        ));
    }

    let num_outputs =
        u32::from_le_bytes(proof[16..20].try_into().unwrap()) as usize;

    let payload = &proof[HEADER_SIZE..];
    if payload.len() < num_outputs * 32 {
        return Err(ZkError::VerificationFailed(
            "proof too short for declared outputs".into(),
        ));
    }

    let mut outputs = Vec::with_capacity(num_outputs);
    for i in 0..num_outputs {
        let start = i * 32;
        let fe = FieldElement::from_bytes(&payload[start..start + 32]);
        outputs.push(fe);
    }
    Ok(outputs)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zk::backend::{Witness, ZkConfig};
    use crate::zk::field::FieldElement;
    use crate::zk::ir::{Wire, ZkIR};

    fn make_witness(outputs: Vec<(Wire, FieldElement)>) -> Witness {
        Witness {
            values: vec![FieldElement::zero(); 4],
            public_inputs: vec![],
            public_outputs: outputs,
        }
    }

    #[test]
    fn generate_proof_encodes_magic_header() {
        let ir = ZkIR::new("test");
        let witness = make_witness(vec![(Wire(0), FieldElement::from_u64(42))]);
        let proof = generate_proof(&ir, &witness, &ZkConfig::default()).unwrap();
        assert!(proof.starts_with(PROOF_MAGIC));
    }

    #[test]
    fn generate_proof_encodes_output_count() {
        let ir = ZkIR::new("test");
        let witness = make_witness(vec![
            (Wire(0), FieldElement::from_u64(1)),
            (Wire(1), FieldElement::from_u64(2)),
        ]);
        let proof = generate_proof(&ir, &witness, &ZkConfig::default()).unwrap();
        let count = u32::from_le_bytes(proof[16..20].try_into().unwrap());
        assert_eq!(count, 2);
    }

    #[test]
    fn generate_proof_roundtrips_field_elements() {
        let ir = ZkIR::new("test");
        let val = FieldElement::from_u64(12345);
        let witness = make_witness(vec![(Wire(0), val)]);
        let proof = generate_proof(&ir, &witness, &ZkConfig::default()).unwrap();

        let outputs = extract_outputs(&proof).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], val);
    }

    #[test]
    fn generate_proof_fails_with_no_outputs() {
        let ir = ZkIR::new("test");
        let witness = make_witness(vec![]);
        let result = generate_proof(&ir, &witness, &ZkConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn extract_outputs_rejects_bad_magic() {
        let bad_proof = b"NOT_A_VALID_PROOF_HEADER_PLUS_PAD";
        let result = extract_outputs(bad_proof);
        assert!(result.is_err());
    }

    #[test]
    fn extract_outputs_rejects_truncated_proof() {
        // Valid header but claims 1 output with no payload.
        let mut proof = Vec::new();
        proof.extend_from_slice(PROOF_MAGIC);
        proof.extend_from_slice(&1u32.to_le_bytes());
        // No output bytes follow.
        let result = extract_outputs(&proof);
        assert!(result.is_err());
    }
}
