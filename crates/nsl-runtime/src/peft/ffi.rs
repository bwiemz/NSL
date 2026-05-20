//! C ABI entry points for WRGA's runtime-visible surface.
//!
//! Design: WRGA folds the LoRA contribution into the host matmul's epilogue,
//! so the hot path is a *fused* PTX/CPU kernel.  But some LoRA instances can't
//! be epilogue-fused (e.g. LoRA on a norm host), and for those cases we still
//! want a single entry point that computes `out = main + α · B · (A · x)`.
//! That's what [`nsl_wrga_epilogue_fused_lora`] provides.
//!
//! Every tensor pointer arriving here is a `*NslTensor` encoded as `i64` —
//! the same calling convention the rest of the NSL runtime uses.  The routine
//! returns a freshly-allocated `*NslTensor` that the caller owns and must
//! free with `nsl_tensor_free`.

use crate::tensor::arithmetic::{nsl_tensor_add, nsl_tensor_matmul, nsl_tensor_mul_scalar};
use crate::tensor::fbip_flags::{RELINQUISH_A, RELINQUISH_B};
use crate::tensor::nsl_tensor_free;

/// Compute `main_result + alpha · (B · (A · x))` in a single call.
///
/// * `main_result`: output of the base matmul / normalisation, owned by caller
/// * `x`: the original input activation
/// * `lora_a`: adapter down-projection, shape `[d_in, rank]`
/// * `lora_b`: adapter up-projection, shape `[rank, d_out]`
/// * `alpha`: scaling factor (typically `α/r`)
///
/// Returns a newly allocated tensor; the four operand pointers remain owned by
/// the caller.  When `RELINQUISH` semantics are desired, the caller can free
/// the operands themselves — this entry point never takes ownership of them.
///
/// On any null input pointer, returns `0` (the runtime's null-tensor
/// sentinel).
///
/// # Safety
/// All tensor pointers must be valid `*NslTensor` handles allocated by the
/// NSL runtime, or `0`.
#[no_mangle]
pub extern "C" fn nsl_wrga_epilogue_fused_lora(
    main_result: i64,
    x: i64,
    lora_a: i64,
    lora_b: i64,
    alpha: f64,
) -> i64 {
    if main_result == 0 || x == 0 || lora_a == 0 || lora_b == 0 {
        return 0;
    }

    //   down = x @ A        (shape: [..., rank])
    //   up   = down @ B     (shape: [..., d_out])
    //   scaled = up * alpha
    //   out    = main + scaled
    //
    // We intentionally operate out-of-place here: the caller still needs
    // `main_result`, `x`, `lora_a`, and `lora_b` alive on return.  Only the
    // intermediate `down`, `up`, and `scaled` tensors are consumed via the
    // RELINQUISH flags so the runtime can fuse them and avoid heap churn.

    let down = nsl_tensor_matmul(x, lora_a, 0);
    if down == 0 {
        return 0;
    }
    let up = nsl_tensor_matmul(down, lora_b, RELINQUISH_A);
    if up == 0 {
        // `down` already freed by RELINQUISH_A; nothing else to clean up.
        return 0;
    }
    let scaled = nsl_tensor_mul_scalar(up, alpha, RELINQUISH_A);
    if scaled == 0 {
        return 0;
    }
    let out = nsl_tensor_add(main_result, scaled, RELINQUISH_B);
    if out == 0 {
        // `scaled` freed by RELINQUISH_B.  Don't leak `main_result`.
        return 0;
    }
    out
}

/// Fused IA³ scaling: `out = gamma ⊙ f(x)` where `gamma` is a learned
/// per-output scaling vector.  The host kernel (e.g. RMSNorm) typically fuses
/// this into its epilogue at compile time; this entry point exists for cases
/// where a runtime fallback is needed (dynamic shapes, mixed-precision edge
/// cases, etc.).
///
/// * `activation`: the host op's output, owned by caller
/// * `gamma`: IA³ scaling vector
///
/// Returns a freshly allocated tensor.  On failure or null inputs, returns 0.
///
/// # Safety
/// See [`nsl_wrga_epilogue_fused_lora`].
#[no_mangle]
pub extern "C" fn nsl_wrga_epilogue_fused_ia3(activation: i64, gamma: i64) -> i64 {
    if activation == 0 || gamma == 0 {
        return 0;
    }
    // IA³ is broadcast-elementwise multiplication.  `nsl_tensor_mul` applies
    // broadcasting when shapes differ, so this works for 1-D gamma applied to
    // any rank-k activation.
    crate::tensor::arithmetic::nsl_tensor_mul(activation, gamma, 0)
}

/// Free a tensor produced by a WRGA fused call.  Wrapper over
/// [`nsl_tensor_free`] for symbolic parity with the other FFI entry points.
///
/// # Safety
/// The pointer must either be `0` or have been returned by one of the
/// `nsl_wrga_epilogue_*` functions.
#[no_mangle]
pub extern "C" fn nsl_wrga_result_free(tensor_ptr: i64) {
    if tensor_ptr != 0 {
        nsl_tensor_free(tensor_ptr);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_inputs_return_null() {
        assert_eq!(nsl_wrga_epilogue_fused_lora(0, 0, 0, 0, 1.0), 0);
        assert_eq!(nsl_wrga_epilogue_fused_ia3(0, 0), 0);
    }

    #[test]
    fn free_null_is_safe() {
        // Must be callable with zero without crashing.
        nsl_wrga_result_free(0);
    }
}
