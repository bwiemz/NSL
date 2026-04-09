//! M46: Deterministic kernel variant selection and graph fingerprinting.

/// Kernel variant for deterministic mode.
#[derive(Debug, Clone, PartialEq)]
pub enum KernelVariant {
    Default,
    DeterministicSortReduce,
    DeterministicSortAccumulate,
    DeterministicCublas,
}

/// Select the appropriate kernel variant based on determinism mode.
pub fn select_kernel(op_name: &str, deterministic: bool) -> KernelVariant {
    if !deterministic {
        return KernelVariant::Default;
    }
    match op_name {
        "reduce_sum" | "reduce_mean" => KernelVariant::DeterministicSortReduce,
        "scatter_add" | "embedding_backward" => KernelVariant::DeterministicSortAccumulate,
        "matmul" | "conv2d" => KernelVariant::DeterministicCublas,
        _ => KernelVariant::Default,
    }
}

/// Compute a deterministic hash of a computation graph for checkpoint fingerprinting.
///
/// Uses a simple string-based hash of operation names and shapes.
/// Full structural hashing (BLAKE3) deferred to M46b.
pub fn compute_graph_hash(op_sequence: &[(&str, &[usize])]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (op_name, shape) in op_sequence {
        op_name.hash(&mut hasher);
        shape.hash(&mut hasher);
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_variant_when_not_deterministic() {
        assert_eq!(select_kernel("reduce_sum", false), KernelVariant::Default);
        assert_eq!(select_kernel("matmul", false), KernelVariant::Default);
    }

    #[test]
    fn deterministic_variant_for_atomic_ops() {
        assert_eq!(
            select_kernel("reduce_sum", true),
            KernelVariant::DeterministicSortReduce
        );
        assert_eq!(
            select_kernel("scatter_add", true),
            KernelVariant::DeterministicSortAccumulate
        );
    }

    #[test]
    fn deterministic_variant_for_cublas() {
        assert_eq!(
            select_kernel("matmul", true),
            KernelVariant::DeterministicCublas
        );
    }

    #[test]
    fn default_variant_for_deterministic_ops() {
        assert_eq!(select_kernel("relu", true), KernelVariant::Default);
        assert_eq!(select_kernel("add", true), KernelVariant::Default);
    }

    #[test]
    fn graph_hash_deterministic() {
        let ops1 = vec![("matmul", &[32, 128][..]), ("relu", &[32, 128][..])];
        let ops2 = vec![("matmul", &[32, 128][..]), ("relu", &[32, 128][..])];
        assert_eq!(compute_graph_hash(&ops1), compute_graph_hash(&ops2));
    }

    #[test]
    fn graph_hash_changes_with_ops() {
        let ops1 = vec![("matmul", &[32, 128][..])];
        let ops2 = vec![("relu", &[32, 128][..])];
        assert_ne!(compute_graph_hash(&ops1), compute_graph_hash(&ops2));
    }
}
