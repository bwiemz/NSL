//! M57.1 §3.3 — FPGA-specific AST → structured-KIR lowerer.
//!
//! Recognizes the v1 MLP shape:
//!
//!     model TinyMlp:
//!         W<i>: Tensor<[K, N], dtype>
//!         b<i>: Tensor<[N], acc_dtype>
//!         ...
//!         fn forward(self, x: Tensor<[1, K], dtype>) -> Tensor<[1, N], acc_dtype>:
//!             let h<i> = relu(matmul(<prev>, self.W<i>) + self.b<i>)
//!             ...
//!             return relu(matmul(<prev>, self.W<N>) + self.b<N>)
//!
//! Emits 3 structured KIR ops per layer: `KirOp::Matmul`, `KirOp::ElementwiseAdd`,
//! `KirOp::Relu`. Errors on unrecognized shapes per [M57 §2.5]'s op-validated-
//! shape-accepting discipline.
//!
//! Blast-radius isolation (M57.1 spec Q5): dispatched at the AST→KIR entry point
//! via a single additive `if target == Fpga` branch; GPU codegen path untouched.
//!
//! Task 2.1 lands the skeleton; Task 2.2 fills in the actual v1 MLP recognizer.

use crate::fpga_error::FpgaLoweringError;

// IMPORTANT: imports below are placeholder shapes; adapt to actual AST + KIR
// types in Task 2.2 (when the recognizer is implemented). For the skeleton,
// keep imports minimal — only what's needed for the entry-point stub.

/// Entry point — invoked when the AST→KIR pipeline dispatches on
/// `target == GpuTarget::Fpga`.
///
/// Walks the parsed NSL `model` block's `fn forward` body, recognizes the
/// per-layer `relu(matmul(x, W) + b)` pattern, and emits the corresponding
/// structured KIR ops.
///
/// Task 2.1: skeleton returns UnsupportedV1Shape with "<not yet implemented>".
/// Task 2.2: actual recognizer implementation.
pub fn lower(/* ast: &Module, interner: &StringInterner */) -> Result<(), FpgaLoweringError> {
    Err(FpgaLoweringError::UnsupportedV1Shape {
        found: "<not yet implemented>".to_string(),
        expected: "model TinyMlp with fn forward(self, x) returning relu(matmul(...) + bias) per layer",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lower_skeleton_returns_unsupported() {
        // Task 2.1 only ships the entry-point stub; Task 2.2 implements the
        // recognizer. This test documents the current skeleton state and will
        // be tightened in Task 2.4 with the real shape-recognition cases.
        let err = lower().expect_err("skeleton always returns UnsupportedV1Shape");
        assert!(matches!(err, FpgaLoweringError::UnsupportedV1Shape { .. }));
    }
}
