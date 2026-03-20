//! M55: Plonky3 ZK backend stub.
//!
//! Plonky3 is a FRI-based proof system with a fast prover and no trusted setup.
//! This stub returns [`ZkError::BackendNotImplemented`] for all operations.
//! Full implementation is tracked in a follow-up task.

use crate::zk::backend::{
    CompiledCircuit, Witness, ZkBackend, ZkConfig, ZkError, ZkIR,
};
use crate::zk::field::FieldElement;

/// Plonky3 FRI-based proof backend (stub).
///
/// All methods currently return [`ZkError::BackendNotImplemented`].
/// This struct exists so that the `plonky3` variant can be wired into the
/// CLI option parser and compiler pipeline without a runtime panic.
pub struct Plonky3Backend;

impl ZkBackend for Plonky3Backend {
    type ProvingKey = ();
    type VerificationKey = ();
    type Proof = ();

    fn compile(&self, _ir: &ZkIR, _config: &ZkConfig) -> Result<CompiledCircuit, ZkError> {
        Err(ZkError::BackendNotImplemented {
            backend: "plonky3",
            message: "Plonky3 circuit compilation is not yet implemented (M55 Task 4)",
        })
    }

    fn setup(
        &self,
        _circuit: &CompiledCircuit,
    ) -> Result<(Self::ProvingKey, Self::VerificationKey), ZkError> {
        Err(ZkError::BackendNotImplemented {
            backend: "plonky3",
            message: "Plonky3 key setup is not yet implemented (M55 Task 4)",
        })
    }

    fn prove(
        &self,
        _pk: &Self::ProvingKey,
        _witness: &Witness,
    ) -> Result<Self::Proof, ZkError> {
        Err(ZkError::BackendNotImplemented {
            backend: "plonky3",
            message: "Plonky3 proof generation is not yet implemented (M55 Task 4)",
        })
    }

    fn verify(
        &self,
        _vk: &Self::VerificationKey,
        _proof: &Self::Proof,
        _public_inputs: &[FieldElement],
    ) -> Result<bool, ZkError> {
        Err(ZkError::BackendNotImplemented {
            backend: "plonky3",
            message: "Plonky3 verification is not yet implemented (M55 Task 4)",
        })
    }

    fn estimate_proof_size(&self, _circuit: &CompiledCircuit) -> u64 {
        0
    }

    fn emit_solidity(&self, _vk: &Self::VerificationKey) -> Option<String> {
        None
    }
}
