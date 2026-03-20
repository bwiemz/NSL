//! Halo2 proof verification — v1 placeholder with correct data flow.
//!
//! Verifies a v1 placeholder proof by extracting the encoded public outputs and
//! comparing them against the expected values. Real Halo2 verification (calling
//! `verify_proof()` with the KZG verifying key) is deferred to a follow-up task.
//!
//! ## Verification steps (v1)
//!
//! 1. Check the proof magic header (`NSL_ZK_PROOF_V1\0`).
//! 2. Extract encoded public outputs from the proof payload.
//! 3. Compare each extracted output against the expected values.
//! 4. Return `Ok(true)` if all match, `Ok(false)` if any differs.

use super::super::backend::ZkError;
use super::super::field::FieldElement;
use super::prove;

/// Verify a v1 placeholder proof against expected public outputs.
///
/// # Arguments
/// - `proof`: the raw proof bytes produced by [`prove::generate_proof`].
/// - `public_outputs`: the expected public output values, in order.
/// - `_config`: unused in v1 (needed for real Halo2 verification).
///
/// # Returns
/// - `Ok(true)` if the proof is valid and all public outputs match.
/// - `Ok(false)` if the proof is structurally valid but outputs differ.
/// - `Err(ZkError::VerificationFailed)` if the proof is malformed.
pub fn verify_proof(
    proof: &[u8],
    public_outputs: &[FieldElement],
    _config: &super::super::backend::ZkConfig,
) -> Result<bool, ZkError> {
    // WARNING: v1 placeholder — does NOT provide zero-knowledge security.
    // Proofs can be trivially forged. Replace with real Halo2 circuit synthesis.

    // Extract the outputs encoded in the proof.
    let proof_outputs = prove::extract_outputs(proof)?;

    // Length mismatch means the proof and expected outputs are incompatible.
    if proof_outputs.len() != public_outputs.len() {
        return Ok(false);
    }

    // Element-wise comparison.
    for (got, expected) in proof_outputs.iter().zip(public_outputs.iter()) {
        if got != expected {
            return Ok(false);
        }
    }

    Ok(true)
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

    fn make_proof(outputs: &[FieldElement]) -> Vec<u8> {
        let pairs: Vec<(Wire, FieldElement)> = outputs
            .iter()
            .enumerate()
            .map(|(i, &v)| (Wire(i as u64), v))
            .collect();
        let witness = Witness {
            values: vec![FieldElement::zero(); outputs.len()],
            public_inputs: vec![],
            public_outputs: pairs,
        };
        let ir = ZkIR::new("test");
        prove::generate_proof(&ir, &witness, &ZkConfig::default()).unwrap()
    }

    #[test]
    fn verify_matching_outputs() {
        let val = FieldElement::from_u64(42);
        let proof = make_proof(&[val]);
        let result = verify_proof(&proof, &[val], &ZkConfig::default()).unwrap();
        assert!(result, "matching outputs should verify");
    }

    #[test]
    fn verify_mismatched_outputs_returns_false() {
        let proof = make_proof(&[FieldElement::from_u64(42)]);
        let result = verify_proof(
            &proof,
            &[FieldElement::from_u64(99)],
            &ZkConfig::default(),
        )
        .unwrap();
        assert!(!result, "mismatched outputs should not verify");
    }

    #[test]
    fn verify_wrong_length_returns_false() {
        let proof = make_proof(&[FieldElement::from_u64(1)]);
        // Expect 2 outputs but proof has 1.
        let result = verify_proof(
            &proof,
            &[FieldElement::from_u64(1), FieldElement::from_u64(2)],
            &ZkConfig::default(),
        )
        .unwrap();
        assert!(!result, "length mismatch should not verify");
    }

    #[test]
    fn verify_empty_proof_returns_error() {
        let result = verify_proof(&[], &[], &ZkConfig::default());
        assert!(result.is_err(), "empty proof should fail");
    }

    #[test]
    fn verify_bad_magic_returns_error() {
        let bad = b"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX";
        let result = verify_proof(bad, &[], &ZkConfig::default());
        assert!(result.is_err(), "bad magic should fail");
    }

    #[test]
    fn verify_multiple_outputs() {
        let vals = vec![
            FieldElement::from_u64(10),
            FieldElement::from_u64(20),
            FieldElement::from_u64(30),
        ];
        let proof = make_proof(&vals);
        let result = verify_proof(&proof, &vals, &ZkConfig::default()).unwrap();
        assert!(result, "multiple matching outputs should verify");
    }
}
