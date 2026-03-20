//! M55: ZkBackend trait and configuration types.
//!
//! Defines the abstract interface that all ZK proof backends (Halo2, Plonky3, …)
//! must implement, plus the shared configuration and error types.

use super::field::FieldElement;

// ---------------------------------------------------------------------------
// Placeholder types — will be replaced by real `ir.rs` types in Task 2
// ---------------------------------------------------------------------------

/// Placeholder for the ZK intermediate representation (Task 2).
pub struct ZkIR;

/// Placeholder for a compiled arithmetic circuit (Task 2).
pub struct CompiledCircuit;

/// Placeholder for the prover witness (Task 2).
pub struct Witness;

// ---------------------------------------------------------------------------
// ZkBackend trait
// ---------------------------------------------------------------------------

/// Abstract interface for zero-knowledge proof backends.
///
/// Implementors compile NSL inference graphs into arithmetic circuits, generate
/// proving/verification keys, create proofs, and verify them.  The associated
/// types allow each backend to use its own key and proof representations without
/// boxing or type-erasing.
pub trait ZkBackend {
    /// The proving key type produced by [`ZkBackend::setup`].
    type ProvingKey;
    /// The verification key type produced by [`ZkBackend::setup`].
    type VerificationKey;
    /// The proof type produced by [`ZkBackend::prove`].
    type Proof;

    /// Compile a [`ZkIR`] graph into a backend-specific [`CompiledCircuit`].
    fn compile(&self, ir: &ZkIR, config: &ZkConfig) -> Result<CompiledCircuit, ZkError>;

    /// Perform trusted setup / key generation for the compiled circuit.
    ///
    /// Returns `(proving_key, verification_key)`.
    fn setup(
        &self,
        circuit: &CompiledCircuit,
    ) -> Result<(Self::ProvingKey, Self::VerificationKey), ZkError>;

    /// Generate a proof for the given witness under `pk`.
    fn prove(
        &self,
        pk: &Self::ProvingKey,
        witness: &Witness,
    ) -> Result<Self::Proof, ZkError>;

    /// Verify a proof against public inputs.
    ///
    /// Returns `Ok(true)` if the proof is valid, `Ok(false)` if it is not.
    fn verify(
        &self,
        vk: &Self::VerificationKey,
        proof: &Self::Proof,
        public_inputs: &[FieldElement],
    ) -> Result<bool, ZkError>;

    /// Estimate the serialised proof size in bytes for the given circuit.
    fn estimate_proof_size(&self, circuit: &CompiledCircuit) -> u64;

    /// Emit a Solidity verifier contract for `vk`, if the backend supports it.
    ///
    /// Returns `None` if the backend does not support Solidity output.
    fn emit_solidity(&self, vk: &Self::VerificationKey) -> Option<String>;
}

// ---------------------------------------------------------------------------
// ZkConfig
// ---------------------------------------------------------------------------

/// Configuration passed to [`ZkBackend::compile`].
#[derive(Debug, Clone)]
pub struct ZkConfig {
    /// Which ZK backend to use.
    pub backend: ZkBackendType,
    /// Privacy mode for the inference circuit.
    pub mode: ZkMode,
    /// Whether to also emit a Solidity verifier contract.
    pub emit_solidity: bool,
    /// Elliptic curve to use for commitments / pairing.
    pub curve: ZkCurve,
    /// Optional hard cap on the number of arithmetic constraints.
    ///
    /// Compilation fails with [`ZkError::InvalidConfig`] if the circuit
    /// exceeds this limit.
    pub max_constraints: Option<u64>,
}

impl Default for ZkConfig {
    fn default() -> Self {
        Self {
            backend: ZkBackendType::Halo2,
            mode: ZkMode::ArchitectureAttestation,
            emit_solidity: false,
            curve: ZkCurve::BN254,
            max_constraints: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ZkBackendType
// ---------------------------------------------------------------------------

/// Selects which proof system backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkBackendType {
    /// Halo2 (PSE fork) — plonkish arithmetization, no trusted setup.
    Halo2,
    /// Plonky3 — FRI-based, fast prover, no trusted setup.
    Plonky3,
}

// ---------------------------------------------------------------------------
// ZkMode
// ---------------------------------------------------------------------------

/// Controls which values are kept private in the inference circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkMode {
    /// Model weights are private; inputs are public.
    WeightPrivate,
    /// Model inputs are private; weights are public.
    InputPrivate,
    /// Both weights and inputs are private.
    FullPrivate,
    /// Proves the architecture (layer shapes, activation functions) without
    /// revealing weights or inputs.  Useful for model attestation / auditing.
    ArchitectureAttestation,
}

// ---------------------------------------------------------------------------
// ZkCurve
// ---------------------------------------------------------------------------

/// Elliptic curve used for polynomial commitments / pairing operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkCurve {
    /// BN254 (alt_bn128) — efficient EVM pairing, 128-bit security.
    BN254,
    /// BLS12-381 — higher security margin, used in Ethereum consensus.
    BLS12_381,
}

// ---------------------------------------------------------------------------
// ZkError
// ---------------------------------------------------------------------------

/// Errors returned by [`ZkBackend`] operations.
#[derive(Debug)]
pub enum ZkError {
    /// The requested backend is not yet implemented.
    BackendNotImplemented {
        backend: &'static str,
        message: &'static str,
    },
    /// A compilation error occurred while lowering the IR to a circuit.
    CompilationError(String),
    /// An error occurred during proof generation.
    ProvingError(String),
    /// Proof verification failed (invalid proof or mismatched public inputs).
    VerificationFailed(String),
    /// The supplied [`ZkConfig`] is invalid or inconsistent.
    InvalidConfig(String),
}

impl std::fmt::Display for ZkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ZkError::BackendNotImplemented { backend, message } => {
                write!(f, "ZK backend '{backend}' is not implemented: {message}")
            }
            ZkError::CompilationError(msg) => write!(f, "ZK compilation error: {msg}"),
            ZkError::ProvingError(msg) => write!(f, "ZK proving error: {msg}"),
            ZkError::VerificationFailed(msg) => write!(f, "ZK verification failed: {msg}"),
            ZkError::InvalidConfig(msg) => write!(f, "invalid ZK configuration: {msg}"),
        }
    }
}

impl std::error::Error for ZkError {}
