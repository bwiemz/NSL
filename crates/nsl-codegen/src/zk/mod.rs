//! M55: Zero-Knowledge Inference Circuits for NeuralScript.
//!
//! This module implements ZK proof generation for NSL inference graphs.
//! It compiles NSL model forward passes into arithmetic circuits over the
//! BN254 scalar field, enabling verifiable ML inference without revealing weights.

pub mod field;
pub mod lookup;
pub mod ir;
pub mod backend;
pub mod halo2;
pub mod lower;
pub mod plonky3;
pub mod stats;
pub mod witness;

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

// ---------------------------------------------------------------------------
// Result type for top-level ZK compilation
// ---------------------------------------------------------------------------

/// Result of compiling a function or model decorated with `@zk_proof`.
pub struct ZkCompileResult {
    pub zkir: ir::ZkIR,
    pub stats: stats::CircuitStats,
}

// ---------------------------------------------------------------------------
// extract_zk_proof_decorator
// ---------------------------------------------------------------------------

/// Extract `@zk_proof` or `@zk_proof(mode="weight_private")` from a decorator list.
///
/// Returns the [`backend::ZkMode`] if a `@zk_proof` decorator is found;
/// defaults to [`backend::ZkMode::ArchitectureAttestation`] when no `mode=` arg
/// is present.
pub fn extract_zk_proof_decorator<'a>(
    decos: &[Decorator],
    resolve: &dyn Fn(Symbol) -> &'a str,
) -> Option<backend::ZkMode> {
    for d in decos {
        if d.name.len() == 1 && resolve(d.name[0]) == "zk_proof" {
            let mode = d
                .args
                .as_ref()
                .and_then(|args| {
                    args.iter().find_map(|arg| {
                        let name_sym = arg.name?;
                        if resolve(name_sym) != "mode" {
                            return None;
                        }
                        if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                            match s.as_str() {
                                "weight_private" => Some(backend::ZkMode::WeightPrivate),
                                "input_private" => Some(backend::ZkMode::InputPrivate),
                                "full_private" => Some(backend::ZkMode::FullPrivate),
                                "architecture_attestation" => {
                                    Some(backend::ZkMode::ArchitectureAttestation)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or(backend::ZkMode::ArchitectureAttestation);
            return Some(mode);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// extract_zk_lookup_decorator
// ---------------------------------------------------------------------------

/// Extract `@zk_lookup(input_bits=N, output_bits=M)` from a decorator list.
///
/// Returns `Some((input_bits, output_bits))` if a `@zk_lookup` decorator is found.
/// Both `input_bits` and `output_bits` are required; the decorator is ignored
/// if either argument is missing or not an integer literal.
pub fn extract_zk_lookup_decorator<'a>(
    decos: &[Decorator],
    resolve: &dyn Fn(Symbol) -> &'a str,
) -> Option<(u32, u32)> {
    for d in decos {
        if d.name.len() == 1 && resolve(d.name[0]) == "zk_lookup" {
            let args = d.args.as_ref()?;
            let mut input_bits: Option<u32> = None;
            let mut output_bits: Option<u32> = None;

            for arg in args {
                let name_sym = match arg.name {
                    Some(s) => s,
                    None => continue,
                };
                let name = resolve(name_sym);
                if let ExprKind::IntLiteral(val) = &arg.value.kind {
                    match name {
                        "input_bits" => input_bits = Some(*val as u32),
                        "output_bits" => output_bits = Some(*val as u32),
                        _ => {}
                    }
                }
            }

            if let (Some(ib), Some(ob)) = (input_bits, output_bits) {
                return Some((ib, ob));
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// compile_zk — top-level orchestrator (placeholder until Task 14)
// ---------------------------------------------------------------------------

/// Top-level ZK compilation orchestrator.
///
/// Compiles a function or model decorated with `@zk_proof` into a [`ZkCompileResult`].
/// Full end-to-end wiring is completed in Task 14; this stub confirms the function
/// was detected and returns an informative error so callers can iterate.
pub fn compile_zk(
    fn_name: &str,
    _mode: backend::ZkMode,
    _config: &backend::ZkConfig,
) -> Result<ZkCompileResult, backend::ZkError> {
    // Placeholder — full AST/type lowering is wired in Task 14.
    Err(backend::ZkError::CompilationError(format!(
        "ZK compilation of '{}' not yet fully wired (Task 14)",
        fn_name
    )))
}
