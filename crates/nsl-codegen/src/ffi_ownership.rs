//! ELTLS tensor FFI ownership classification table (spec §6.2).
//!
//! Every tensor-returning FFI exported by nsl-runtime should have an entry
//! here. compile_call consults this table after emitting the FFI call to
//! register the result's ownership automatically.

use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy)]
pub enum FfiOwnershipKind {
    /// Result is a freshly-allocated tensor. Default for ~95% of FFIs.
    OwnedNewResult,
    /// Result aliases input N (view/reshape/squeeze).
    BorrowedFromInput(usize),
    /// Result is not a tensor (scalar-returning).
    NotATensor,
}

pub fn ffi_ownership_kind(name: &str) -> Option<FfiOwnershipKind> {
    static TABLE: OnceLock<HashMap<&'static str, FfiOwnershipKind>> = OnceLock::new();
    TABLE.get_or_init(build_table).get(name).copied()
}

fn build_table() -> HashMap<&'static str, FfiOwnershipKind> {
    use FfiOwnershipKind::*;
    let mut m = HashMap::new();
    // Binary ops
    m.insert("nsl_tensor_add", OwnedNewResult);
    m.insert("nsl_tensor_sub", OwnedNewResult);
    m.insert("nsl_tensor_mul", OwnedNewResult);
    m.insert("nsl_tensor_div", OwnedNewResult);
    m.insert("nsl_tensor_matmul", OwnedNewResult);
    m.insert("nsl_tensor_add_scalar", OwnedNewResult);
    m.insert("nsl_tensor_mul_scalar", OwnedNewResult);
    m.insert("nsl_fp8_matmul_training", OwnedNewResult);
    // Unary ops
    m.insert("nsl_tensor_neg", OwnedNewResult);
    m.insert("nsl_tensor_exp", OwnedNewResult);
    m.insert("nsl_tensor_log", OwnedNewResult);
    m.insert("nsl_tensor_sqrt", OwnedNewResult);
    m.insert("nsl_tensor_abs", OwnedNewResult);
    m.insert("nsl_tensor_sign", OwnedNewResult);
    m.insert("nsl_tensor_relu", OwnedNewResult);
    m.insert("nsl_tensor_gelu", OwnedNewResult);
    m.insert("nsl_tensor_silu", OwnedNewResult);
    m.insert("nsl_tensor_sigmoid", OwnedNewResult);
    m.insert("nsl_tensor_tanh", OwnedNewResult);
    m.insert("nsl_tensor_softmax", OwnedNewResult);
    m.insert("nsl_tensor_log_softmax", OwnedNewResult);
    m.insert("nsl_tensor_sin", OwnedNewResult);
    m.insert("nsl_tensor_cos", OwnedNewResult);
    // Creation / conversion
    m.insert("nsl_tensor_clone", OwnedNewResult);
    m.insert("nsl_tensor_contiguous", OwnedNewResult);
    m.insert("nsl_tensor_to_device", OwnedNewResult);
    m.insert("nsl_tensor_cast", OwnedNewResult);
    m.insert("nsl_tensor_zeros_like", OwnedNewResult);
    m.insert("nsl_tensor_ones_like", OwnedNewResult);
    // Shape ops that produce new storage
    m.insert("nsl_tensor_reshape", OwnedNewResult);
    m.insert("nsl_tensor_transpose", OwnedNewResult);
    m.insert("nsl_tensor_permute", OwnedNewResult);
    m.insert("nsl_tensor_concat", OwnedNewResult);
    m.insert("nsl_tensor_stack", OwnedNewResult);
    m.insert("nsl_tensor_gather", OwnedNewResult);
    m.insert("nsl_tensor_scatter", OwnedNewResult);
    m.insert("nsl_tensor_expand", OwnedNewResult);
    m.insert("nsl_tensor_broadcast_to", OwnedNewResult);
    // Views (alias input storage)
    m.insert("nsl_tensor_view", BorrowedFromInput(0));
    m.insert("nsl_tensor_slice", BorrowedFromInput(0));
    // Reductions
    m.insert("nsl_tensor_sum", OwnedNewResult);
    m.insert("nsl_tensor_mean", OwnedNewResult);
    m.insert("nsl_tensor_max", OwnedNewResult);
    m.insert("nsl_tensor_min", OwnedNewResult);
    m.insert("nsl_tensor_sum_dim", OwnedNewResult);
    m.insert("nsl_tensor_mean_dim", OwnedNewResult);
    m.insert("nsl_tensor_argmax", OwnedNewResult);
    m.insert("nsl_tensor_argmin", OwnedNewResult);
    // Scalar-returning
    m.insert("nsl_tensor_sum_to_scalar", NotATensor);
    m.insert("nsl_tensor_mean_to_scalar", NotATensor);
    m.insert("nsl_tensor_item", NotATensor);
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_ffi_returns_some() {
        assert!(ffi_ownership_kind("nsl_tensor_add").is_some());
        assert!(matches!(
            ffi_ownership_kind("nsl_tensor_add"),
            Some(FfiOwnershipKind::OwnedNewResult)
        ));
    }

    #[test]
    fn unknown_ffi_returns_none() {
        assert!(ffi_ownership_kind("nsl_totally_fake_function").is_none());
    }

    #[test]
    fn view_is_borrowed_from_input_zero() {
        assert!(matches!(
            ffi_ownership_kind("nsl_tensor_view"),
            Some(FfiOwnershipKind::BorrowedFromInput(0))
        ));
    }
}
