//! M38b: Autodiff tape ownership safety — classifies each TapeOp by its backward
//! data requirements to determine which tensors must stay alive during backward pass.

/// How a tape op accesses input data during the backward pass.
#[derive(Clone, Debug, PartialEq)]
pub enum BackwardAccess {
    /// Only needs input shape and/or scalar constants stored on the tape, not input
    /// tensor data. The input tensor's data buffer can be freed after the forward op.
    ShapeOnly,
    /// Needs saved input/output data for backward (Mul, Div, MatMul, ReLU, etc.),
    /// OR the tape holds refcount-bumped input pointers for gradient routing
    /// (BiasAdd, Cat, Stack). The tape holds a live reference — do not free early.
    DataRequired,
    /// Needs auxiliary data structures for backward (ReduceMax/argmax, Dropout/mask).
    /// Aux data is owned by the tape — no input tensor ownership concerns.
    AuxDataRequired,
}

/// Classify a tape operation name by its backward data requirements.
/// Names match the TapeOp variant names in `crates/nsl-runtime/src/autodiff.rs`.
pub fn classify_backward_access(op_name: &str) -> BackwardAccess {
    match op_name {
        // Shape-only: backward only needs input shape and/or scalar constants,
        // not input tensor data. Input buffer can be freed after forward op.
        "Add" | "Sub" | "Neg" | "AddScalar" | "MulScalar"
        | "SumReduce" | "MeanReduce" | "Transpose" | "Slice"
        | "Unsqueeze" | "Expand" => BackwardAccess::ShapeOnly,

        // Data-required: backward needs saved input/output tensor data, OR the tape
        // holds refcount-bumped input pointers for gradient routing (BiasAdd, Cat, Stack).
        "Mul" | "Div" | "MatMul"
        | "ReLU" | "GELU" | "SiLU" | "Log" | "Abs" | "Clamp"
        | "Exp" | "Sqrt" | "Sigmoid" | "Tanh" | "Softmax"
        | "EmbeddingLookup" | "LayerNorm" | "RMSNorm" | "Conv2d"
        | "BiasAdd" | "Cat" | "Stack" => BackwardAccess::DataRequired,

        // Auxiliary data: backward uses saved indices/masks/argmax, not input tensors
        "ReduceMax" | "MaxPool2d" | "Dropout" | "Gather" => BackwardAccess::AuxDataRequired,

        // Unknown ops conservatively require data
        _ => BackwardAccess::DataRequired,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- ShapeOnly ops ---

    #[test]
    fn test_add_is_shape_only() {
        assert_eq!(classify_backward_access("Add"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_sub_is_shape_only() {
        assert_eq!(classify_backward_access("Sub"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_neg_is_shape_only() {
        assert_eq!(classify_backward_access("Neg"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_sum_reduce_is_shape_only() {
        assert_eq!(classify_backward_access("SumReduce"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_mean_reduce_is_shape_only() {
        assert_eq!(classify_backward_access("MeanReduce"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_transpose_is_shape_only() {
        assert_eq!(classify_backward_access("Transpose"), BackwardAccess::ShapeOnly);
    }

    // --- DataRequired ops ---

    #[test]
    fn test_mul_is_data_required() {
        assert_eq!(classify_backward_access("Mul"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_matmul_is_data_required() {
        assert_eq!(classify_backward_access("MatMul"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_relu_is_data_required() {
        assert_eq!(classify_backward_access("ReLU"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_softmax_is_data_required() {
        assert_eq!(classify_backward_access("Softmax"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_bias_add_is_data_required() {
        assert_eq!(classify_backward_access("BiasAdd"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_cat_is_data_required() {
        assert_eq!(classify_backward_access("Cat"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_stack_is_data_required() {
        assert_eq!(classify_backward_access("Stack"), BackwardAccess::DataRequired);
    }

    // --- AuxDataRequired ops ---

    #[test]
    fn test_reduce_max_is_aux() {
        assert_eq!(classify_backward_access("ReduceMax"), BackwardAccess::AuxDataRequired);
    }

    #[test]
    fn test_dropout_is_aux() {
        assert_eq!(classify_backward_access("Dropout"), BackwardAccess::AuxDataRequired);
    }

    #[test]
    fn test_gather_is_aux() {
        assert_eq!(classify_backward_access("Gather"), BackwardAccess::AuxDataRequired);
    }

    #[test]
    fn test_maxpool_is_aux() {
        assert_eq!(classify_backward_access("MaxPool2d"), BackwardAccess::AuxDataRequired);
    }

    // --- Unknown fallback ---

    #[test]
    fn test_unknown_op_conservative() {
        assert_eq!(classify_backward_access("FutureOp"), BackwardAccess::DataRequired);
    }

    // --- Completeness: verify all 36 TapeOp variants are classified ---

    #[test]
    fn test_all_tape_ops_classified() {
        let shape_only = vec![
            "Add", "Sub", "Neg", "AddScalar", "MulScalar",
            "SumReduce", "MeanReduce", "Transpose", "Slice",
            "Unsqueeze", "Expand",
        ];
        let data_required = vec![
            "Mul", "Div", "MatMul",
            "ReLU", "GELU", "SiLU", "Log", "Abs", "Clamp",
            "Exp", "Sqrt", "Sigmoid", "Tanh", "Softmax",
            "EmbeddingLookup", "LayerNorm", "RMSNorm", "Conv2d",
            "BiasAdd", "Cat", "Stack",
        ];
        let aux_required = vec!["ReduceMax", "MaxPool2d", "Dropout", "Gather"];

        for op in &shape_only {
            assert_eq!(
                classify_backward_access(op), BackwardAccess::ShapeOnly,
                "Expected ShapeOnly for {op}"
            );
        }
        for op in &data_required {
            assert_eq!(
                classify_backward_access(op), BackwardAccess::DataRequired,
                "Expected DataRequired for {op}"
            );
        }
        for op in &aux_required {
            assert_eq!(
                classify_backward_access(op), BackwardAccess::AuxDataRequired,
                "Expected AuxDataRequired for {op}"
            );
        }
        let total = shape_only.len() + data_required.len() + aux_required.len();
        assert_eq!(total, 36, "Should cover all 36 TapeOp variants");
    }
}
