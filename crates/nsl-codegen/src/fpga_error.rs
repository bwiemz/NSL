//! FPGA backend error types per M57 v1 spec §2.5.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum FpgaLoweringError {
    #[error(
        "KIR op `{op_kind}` is not supported by the FPGA target in v1. \
         Supported ops: {supported:?}. \
         See M57 v1 deferred-roadmap (§1.5 of design spec) for the future \
         milestone that adds this op."
    )]
    UnsupportedKirOp {
        op_kind: &'static str,
        supported: &'static [&'static str],
        // Source span omitted in v1's first implementation; future PR adds
        // it once the AST → KIR pass propagates spans through structured ops.
    },

    #[error("HIR builder error: {0}")]
    HirBuilder(#[from] crate::hir::module::HirBuilderError),

    #[error(
        "Unsupported v1 MLP shape: {found}\n\
         Expected: {expected}\n\
         v1's kernel_lower_fpga recognizes the v1 MLP pattern only; arbitrary \
         model shapes are deferred to a future milestone (M57.1 §3.3, M57 §2.5)."
    )]
    UnsupportedV1Shape {
        found: String,
        expected: &'static str,
    },
}

/// v1 supported KIR ops for the UnsupportedKirOp::supported field.
pub const V1_SUPPORTED_OPS: &[&str] = &["Matmul", "ElementwiseAdd", "Relu"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_op_error_message_names_supported_list() {
        let err = FpgaLoweringError::UnsupportedKirOp {
            op_kind: "Tanh",
            supported: V1_SUPPORTED_OPS,
        };
        let msg = err.to_string();
        assert!(msg.contains("Tanh"));
        assert!(msg.contains("Matmul"));
        assert!(msg.contains("v1 deferred-roadmap"));
    }
}
