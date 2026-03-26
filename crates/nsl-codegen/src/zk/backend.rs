//! M55: ZkBackend trait and configuration types.
//!
//! Defines the abstract interface that all ZK proof backends (Halo2, Plonky3, …)
//! must implement, plus the shared configuration and error types.

use super::field::FieldElement;

// Re-export the IR types from `ir.rs`. Downstream code that imports
// `ZkIR`, `Wire`, etc. from `crate::zk::backend` (e.g. plonky3/mod.rs)
// continues to compile without modification.
pub use super::ir::{Wire, ZkIR};

// ---------------------------------------------------------------------------
// CompiledCircuit
// ---------------------------------------------------------------------------

/// A backend-agnostic compiled arithmetic circuit.
///
/// Produced by [`ZkBackend::compile`] from a [`ZkIR`]. It retains the
/// original IR alongside the circuit sizing metadata computed during
/// compilation (e.g. the number of rows `2^k`).
#[derive(Debug)]
pub struct CompiledCircuit {
    /// The ZK-IR that was compiled into this circuit.
    pub ir: ZkIR,
    /// Circuit size parameter: the PLONKish table has `2^k` rows.
    pub k: u32,
}

// ---------------------------------------------------------------------------
// Witness
// ---------------------------------------------------------------------------

/// The prover's witness: concrete field-element values for all wires.
///
/// A `Witness` is produced by the NSL runtime during inference and passed to
/// [`ZkBackend::prove`]. It assigns a [`FieldElement`] value to each wire in
/// the circuit, together with the public inputs and outputs that appear in the
/// proof's public statement.
#[derive(Debug)]
pub struct Witness {
    /// One field element per wire, indexed by `wire.0`.
    ///
    /// `values.len()` must equal `ZkIR::num_wires` for the compiled circuit.
    pub values: Vec<FieldElement>,
    /// Public input wires paired with their concrete values.
    pub public_inputs: Vec<(Wire, FieldElement)>,
    /// Public output wires paired with their concrete values.
    pub public_outputs: Vec<(Wire, FieldElement)>,
}

impl Witness {
    /// Look up the concrete value of a wire by ID.
    ///
    /// # Panics
    /// Panics if `wire.0` is out of bounds (i.e. the witness is malformed).
    pub fn get(&self, wire: Wire) -> FieldElement {
        self.values[wire.0 as usize]
    }

    /// Return the concrete values of all public output wires, in order.
    pub fn public_output_values(&self) -> Vec<FieldElement> {
        self.public_outputs.iter().map(|(_, v)| *v).collect()
    }
}

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
    /// Which finite field to use for circuit arithmetic.
    pub field: ZkField,
    /// Optional hard cap on the number of arithmetic constraints.
    ///
    /// Compilation fails with [`ZkError::InvalidConfig`] if the circuit
    /// exceeds this limit.
    pub max_constraints: Option<u64>,
    /// Number of fractional bits used in fixed-point arithmetic.
    ///
    /// Controls the precision of [`ZkInstruction::FixedMul`] instructions:
    /// `out = (a * b) >> frac_bits`. Defaults to 8.
    pub frac_bits: u32,
    /// Optional path to a `.safetensors` weight file used to populate the ZK witness.
    ///
    /// When set, weight values are loaded at compile time and embedded into the
    /// circuit witness instead of using dummy zeros.
    pub weights_path: Option<std::path::PathBuf>,
}

impl Default for ZkConfig {
    fn default() -> Self {
        Self {
            backend: ZkBackendType::Folding,
            mode: ZkMode::ArchitectureAttestation,
            emit_solidity: false,
            curve: ZkCurve::BN254,
            field: ZkField::Mersenne31,
            max_constraints: None,
            frac_bits: 8,
            weights_path: None,
        }
    }
}

// ---------------------------------------------------------------------------
// ZkBackendType
// ---------------------------------------------------------------------------

/// Selects which proof system backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkBackendType {
    /// Halo2 (PSE fork) — plonkish arithmetization, no trusted setup. DEPRECATED.
    Halo2,
    /// Plonky3 — FRI-based, fast prover, no trusted setup.
    Plonky3,
    /// Folding — Nova/HyperNova-style incremental verifiable computation.
    /// Lookup-native with AIR constraints. Production backend for large models.
    Folding,
}

/// Selects which finite field to use for ZK arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZkField {
    /// BN254 scalar field (256-bit). EVM-compatible for Solidity verifiers.
    BN254,
    /// Mersenne-31 (p = 2^31 - 1, 32-bit). ~10x faster, used by folding prover.
    Mersenne31,
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

// ---------------------------------------------------------------------------
// FoldingBackend trait — incremental proving for large models
// ---------------------------------------------------------------------------

/// Abstract interface for folding-based ZK proof backends.
///
/// Unlike `ZkBackend` (which compiles the entire circuit at once), a
/// `FoldingBackend` processes the model layer-by-layer, folding each layer's
/// constraints into a running accumulator. This produces constant-size proofs
/// regardless of model depth.
pub trait FoldingBackend<F: super::field::Field> {
    /// Fold a new layer's circuit into the running accumulator.
    fn fold_layer(&mut self, ir: &ZkIR, witness: &[F]) -> Result<(), ZkError>;

    /// Finalize: produce the proof from the accumulated state.
    fn finalize(&self) -> Result<ZkProof, ZkError>;

    /// Verify a folding proof against public inputs.
    fn verify(proof: &ZkProof, public_io: &[F]) -> Result<bool, ZkError>;
}

/// A ZK proof produced by any backend (folding or one-shot).
///
/// Opaque bytes that can be serialized, transmitted, and verified.
#[derive(Debug, Clone)]
pub struct ZkProof {
    /// Serialized proof bytes.
    pub data: Vec<u8>,
    /// Number of layers folded (for folding proofs).
    pub num_folds: u32,
    /// Public inputs committed in the proof.
    pub public_inputs: Vec<Vec<u8>>,
    /// Public outputs committed in the proof.
    pub public_outputs: Vec<Vec<u8>>,
}
