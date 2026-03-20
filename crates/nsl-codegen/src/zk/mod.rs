//! M55: Zero-Knowledge Inference Circuits for NeuralScript.
//!
//! This module implements ZK proof generation for NSL inference graphs.
//! It compiles NSL model forward passes into arithmetic circuits over the
//! BN254 scalar field, enabling verifiable ML inference without revealing weights.

pub mod field;
pub mod backend;
pub mod plonky3;
