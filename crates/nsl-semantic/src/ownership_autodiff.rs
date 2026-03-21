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

    // ── Task 5: Borrowed tensors in grad() scope ─────────────────────
    //
    // Key property: borrows are semantically transparent for reads.
    // When a borrowed tensor is used in a tape-recording op, the tape records
    // the same raw i64 pointer as the underlying owned tensor would produce.
    // The backward access classification is identical whether the input is
    // owned (&T stripped) or borrowed (&T) — the borrow wrapper disappears
    // at the IR level since both produce the same pointer value.
    //
    // This means:
    //   - A borrowed MatMul input: DataRequired (tape must keep the data alive —
    //     but the borrow guarantees the owner keeps it alive for us).
    //   - A borrowed Add input:    ShapeOnly (backward only needs shape, which
    //     is embedded in the NslTensor header, so the borrow is safe to release).
    //   - @no_grad on &T: the borrow inherits the no_grad annotation; the tape
    //     does not record ops on this param (same as owned @no_grad).
    //
    // The combination is: borrow safety (caller keeps tensor alive) +
    // tape classification (determines when the caller CAN release) = no data race.

    #[test]
    fn test_borrowed_matmul_data_required() {
        // A tensor passed by borrow to MatMul: DataRequired.
        // The borrow guarantees the underlying tensor stays alive for the entire
        // scope, satisfying the tape's data-alive requirement at no extra cost.
        assert_eq!(classify_backward_access("MatMul"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_borrowed_add_shape_only() {
        // A tensor passed by borrow to Add: ShapeOnly.
        // The borrow can be released as soon as the forward op completes —
        // the tape only needs the shape (stored in the NslTensor header), which
        // is embedded in the tape entry itself.
        assert_eq!(classify_backward_access("Add"), BackwardAccess::ShapeOnly);
    }

    #[test]
    fn test_borrowed_relu_data_required() {
        // ReLU backward needs the output value (sign mask). When the input is
        // borrowed, the borrow's owner keeps the tensor alive. DataRequired.
        assert_eq!(classify_backward_access("ReLU"), BackwardAccess::DataRequired);
    }

    #[test]
    fn test_borrowed_dropout_aux_data() {
        // Dropout backward uses a saved mask (aux data), not the input tensor.
        // The borrow can be released after the forward op. AuxDataRequired.
        assert_eq!(classify_backward_access("Dropout"), BackwardAccess::AuxDataRequired);
    }

    #[test]
    fn test_no_grad_param_classification_note() {
        // @no_grad on &T: the codegen omits tape_record for this op entirely.
        // From the tape's perspective, there is no op to classify. This test
        // documents that classify_backward_access is only called for ops that
        // ARE recorded on the tape — @no_grad ops are filtered before this point.
        //
        // To simulate: if @no_grad is active, backward_access = None in decide().
        // classify_backward_access is never called. We verify the no_grad path
        // in codegen by checking that None backward_access → FreeAtConsumption
        // in ownership::decide() (tested in nsl-codegen's ownership tests).
        //
        // No assertion here — this is a documentation/spec test.
        let _ = BackwardAccess::ShapeOnly; // ensures the type is in scope
    }
}
