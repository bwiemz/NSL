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

/// Every FFI name the table classifies. Exists for the
/// `ffi_ownership_drift` gate, which binds the table's keys to the
/// runtime's actual `pub extern "C"` inventory (a key naming a
/// nonexistent FFI is a dead entry — `nsl_tensor_concat` sat here for
/// months while the real extern is `nsl_tensor_cat`) and to the
/// owning-ref allowlist in `expr::expr_result_is_owned_temporary`.
pub fn ffi_ownership_table_keys() -> Vec<&'static str> {
    build_table().into_keys().collect()
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
    m.insert("nsl_tensor_tanh_act", OwnedNewResult); // extern spelling: _act suffix
    m.insert("nsl_tensor_softmax", OwnedNewResult);
    m.insert("nsl_tensor_logsoftmax", OwnedNewResult); // extern spelling: no underscore
    m.insert("nsl_tensor_sin", OwnedNewResult);
    m.insert("nsl_tensor_cos", OwnedNewResult);
    // Creation / conversion
    m.insert("nsl_tensor_clone", OwnedNewResult);
    m.insert("nsl_tensor_contiguous", OwnedNewResult);
    m.insert("nsl_tensor_to_device", OwnedNewResult);
    m.insert("nsl_tensor_cast", OwnedNewResult);
    m.insert("nsl_tensor_zeros_like", OwnedNewResult);
    m.insert("nsl_tensor_zeros_like_dtype", OwnedNewResult);
    // CPDT §3.2: INT8 blockwise quant/dequant (both produce freshly-allocated tensors).
    m.insert("nsl_tensor_quant_int8_blockwise", OwnedNewResult);
    m.insert("nsl_tensor_dequant_int8_blockwise", OwnedNewResult);
    m.insert("nsl_tensor_ones_like", OwnedNewResult);
    // Shape ops that produce new storage
    m.insert("nsl_tensor_reshape", OwnedNewResult);
    m.insert("nsl_tensor_transpose", OwnedNewResult);
    // NOTE the spelling: the runtime extern is `nsl_tensor_cat` — a
    // `nsl_tensor_concat` entry sat here dead for months (no such FFI;
    // found by the ffi_ownership_drift runtime-extern binding,
    // 2026-07-29). Same disease as the dead `"slice"` allowlist entry
    // PR #423's review caught.
    m.insert("nsl_tensor_cat", OwnedNewResult);
    m.insert("nsl_tensor_stack", OwnedNewResult);
    m.insert("nsl_tensor_gather", OwnedNewResult);
    m.insert("nsl_tensor_expand", OwnedNewResult);
    // Corrected 2026-07-29 (was BorrowedFromInput(0), documented
    // KNOWN-WRONG since 2026-07-27): the runtime allocates FRESH storage
    // on both paths — `shape_ops.rs` slice CPU and
    // `cuda::gpu_slice_f32_with_shape` GPU — re-verified three times
    // since (the PR #423 allowlist audit, the #426 sampling review's
    // per-runtime walk, and the #427 cycle). The `.slice()` method and
    // the `tensor_slice` free fn have been classified owning (and
    // statement-cleanup freed) through all of those cycles with the leak
    // gates green — the borrow misfile governed only the two
    // `register_ffi_result_ownership` call sites, where it under-freed.
    m.insert("nsl_tensor_slice", OwnedNewResult);
    // Creation intrinsics (all publish fresh storage).
    m.insert("nsl_tensor_zeros", OwnedNewResult);
    m.insert("nsl_tensor_ones", OwnedNewResult);
    m.insert("nsl_tensor_full", OwnedNewResult);
    m.insert("nsl_tensor_rand", OwnedNewResult);
    m.insert("nsl_tensor_randn", OwnedNewResult);
    m.insert("nsl_tensor_arange", OwnedNewResult);
    // The remaining Ident-allowlist family (`expr_result_is_owned_
    // temporary`), each per-runtime verified in that allowlist's own
    // comment blocks across PRs #423/#424/#426: fresh output on every
    // path (contiguous/unsqueeze return counted references either way;
    // eval dropout clones; the sampling ops' GPU arms return fresh
    // uploads).
    m.insert("nsl_tensor_rotate_half", OwnedNewResult);
    m.insert("nsl_tensor_cumsum", OwnedNewResult);
    m.insert("nsl_tensor_unsqueeze", OwnedNewResult);
    m.insert("nsl_tensor_causal_mask", OwnedNewResult);
    m.insert("nsl_tensor_rmsnorm", OwnedNewResult);
    m.insert("nsl_tensor_dropout", OwnedNewResult);
    // Source-AD dropout forward: returns an NslList handle wrapping
    // [out, mask] — a list, not a tensor (same class as topk's dict). The
    // two element tensors become the DropoutMask/Dropout op results and are
    // freed by the wengert owned-values cleanup.
    m.insert("nsl_tensor_dropout_fwd_mask", NotATensor);
    m.insert("nsl_tensor_layernorm", OwnedNewResult);
    m.insert("nsl_tensor_bias_add", OwnedNewResult);
    m.insert("nsl_tensor_embedding_lookup", OwnedNewResult);
    m.insert("nsl_tensor_lt_scalar", OwnedNewResult);
    m.insert("nsl_tensor_multinomial", OwnedNewResult);
    // topk returns a DICT handle, not a tensor — "no tensor tracking" is
    // exactly right here; dict_lifetime.rs owns the dict's lifecycle.
    m.insert("nsl_tensor_topk", NotATensor);
    // Reductions
    m.insert("nsl_tensor_sum", OwnedNewResult);
    m.insert("nsl_tensor_mean", OwnedNewResult);
    m.insert("nsl_tensor_reduce_max", OwnedNewResult); // extern spelling: reduce_ prefix
    m.insert("nsl_tensor_sum_dim", OwnedNewResult);
    m.insert("nsl_tensor_mean_dim", OwnedNewResult);
    m.insert("nsl_tensor_argmax", OwnedNewResult);
    // Fresh on both paths (shape_ops.rs:355; #7b8483f2-review-verified) —
    // was one of the documented absences the old KNOWN-WRONG comment
    // carried.
    m.insert("nsl_tensor_select", OwnedNewResult);
    // The two formerly-unclassified raw-callable externs, each classified
    // from a per-runtime freshness read (2026-07-29), closing the gap the
    // old STILL-UNCLASSIFIED note tracked:
    // - nsl_tensor_reduce_to_shape (ad_ops.rs): every non-degenerate path
    //   hands back a COUNTED reference the caller must free — the
    //   same-shape identity path retains grad before returning it (the
    //   contiguous convention), the reduce paths return fresh
    //   clone/sum_dim outputs, and the migrate path frees its own
    //   to_device temp. The null-ARG guard returns grad_ptr unretained,
    //   but grad_ptr is 0 on the only reachable arm of that guard unless
    //   target is null, which is a broken upstream invariant already
    //   reported loudly.
    // - nsl_tensor_gelu_backward (activation.rs): fresh output on all
    //   three paths (broadcast-fallback mul result, GPU
    //   gpu_backward_binary output, CPU elementwise fresh); frees its own
    //   ones/contiguous/deriv temps.
    m.insert("nsl_tensor_reduce_to_shape", OwnedNewResult);
    m.insert("nsl_tensor_gelu_backward", OwnedNewResult);
    // Dispatch-arm terminals with no allowlist entry (their free-function
    // results stranded in nested/receiver/statement position until the
    // v2a dispatch-boundary registration; each read per-runtime
    // 2026-07-29):
    // - nsl_tensor_clamp (activation.rs): counted reference on every
    //   path — the FBIP in-place arms bump the receiver's refcount
    //   BEFORE returning it (the contiguous convention; both
    //   can_mutate_inplace predicates are permanently false since
    //   ad59b929 anyway), the GPU and CPU fallbacks are fresh.
    // - nsl_tensor_conv2d / nsl_tensor_maxpool2d (tensor/mod.rs): fresh
    //   on the GPU arm (gpu_conv2d_f32 / gpu_maxpool2d_f32; each frees
    //   its own contiguous temp) and NslTensor::publish-fresh on CPU.
    m.insert("nsl_tensor_clamp", OwnedNewResult);
    m.insert("nsl_tensor_conv2d", OwnedNewResult);
    m.insert("nsl_tensor_maxpool2d", OwnedNewResult);
    // Scalar-returning
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
    fn removed_dead_keys_are_gone() {
        // 2026-07-29: `nsl_tensor_view` (with a BorrowedFromInput entry)
        // named no runtime extern at all — real views are minted inside
        // method dispatch via `new_view_i64`, never through an FFI of
        // that name. The `ffi_ownership_drift` gate now binds every key
        // to the extern inventory; the dead keys were removed rather
        // than guessed at. BorrowedFromInput currently has no live
        // entries — it stays for the day a true aliasing FFI appears.
        assert!(ffi_ownership_kind("nsl_tensor_view").is_none());
        assert!(ffi_ownership_kind("nsl_tensor_concat").is_none());
        assert!(ffi_ownership_kind("nsl_tensor_tanh").is_none());
    }
}
