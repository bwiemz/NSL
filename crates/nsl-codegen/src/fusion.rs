//! Elementwise operator fusion: explicit @fuse and lexical auto-fusion.
//! Synthesizes single PTX kernels from chains of elementwise ops.
//! let-binding = hard fusion barrier (no DAG infrastructure).

/// A chain of elementwise operations that can be fused into a single PTX kernel.
pub struct FusedKernel {
    /// Names of ops in the chain (e.g., ["add", "relu"])
    pub op_chain: Vec<String>,
    /// Input tensor count
    pub num_inputs: usize,
    /// Generated PTX bytes (null-terminated)
    pub ptx: Vec<u8>,
    /// Human-readable name for profiler traces
    pub name: String,
}

/// Classification of operations for fusion eligibility.
pub fn is_fusible_op(name: &str) -> bool {
    matches!(
        name,
        "add" | "sub"
            | "mul"
            | "div"
            | "pow"
            | "neg"
            | "abs"
            | "relu"
            | "sigmoid"
            | "tanh"
            | "exp"
            | "log"
            | "sqrt"
            | "sign"
            | "clamp"
    )
}

/// Binary ops that are fusible (from BinOp AST nodes).
pub fn is_fusible_binop(op: &str) -> bool {
    matches!(op, "Add" | "Sub" | "Mul" | "Div" | "Pow")
}

/// Unary ops that are fusible.
pub fn is_fusible_unaryop(op: &str) -> bool {
    matches!(op, "Neg")
}

/// Check if an op is NOT fusible (matmul, reductions, etc.).
pub fn is_fusion_barrier(name: &str) -> bool {
    matches!(
        name,
        "matmul"
            | "sum"
            | "mean"
            | "reduce_max"
            | "reduce_min"
            | "reshape"
            | "transpose"
            | "conv"
            | "softmax"
            | "layernorm"
            | "gather"
            | "scatter"
    )
}
