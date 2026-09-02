//! Per-parameter weight decay — AdamW "parameter groups", NSL-style.
//!
//! `AdamW(weight_decay=λ)` in a `train` block used to apply λ to EVERY
//! trainable parameter, RMSNorm gains and the tied embedding included. The
//! conventional recipe decays only the 2-D weight matrices and exempts
//! norms / biases (and often embeddings); there was no way to say that.
//!
//! `no_decay=[...]` in the optimizer call now names PARAMETER ROLES to
//! exempt, reusing the role vocabulary `@param_role` and the mixed
//! Muon/AdamW router already share (`nsl-codegen/src/param_roles.rs`):
//! `"vector"`, `"embedding"`, `"head"`, `"hidden"`.
//!
//! ## Why the decision lives here and not in codegen
//!
//! Roles come from a COMPILE-TIME table, but `"vector"` cannot. A model
//! field only gets a statically-known rank when its initializer is a direct
//! `zeros/ones/randn/rand/full/arange` call whose shape list is all integer
//! LITERALS (`extract_shape_from_tensor_init`). Real models fail that on
//! both counts: `RMSNorm.weight = ones([dim])` passes an identifier, and
//! `wq = randn([...]) * sqrt(...)` is a multiply, not a call. Measured on
//! `models/coder50m` — 74 parameters classify as 1 `embedding` + 73
//! `hidden`, and **zero** `vector`. Both RMSNorm gains are in that
//! `hidden` bucket.
//!
//! So a static-only "exempt rank < 2" would compile, print a plausible
//! table, and silently decay every norm in the model — the failure mode
//! this repo keeps relearning. `"vector"` is therefore resolved from the
//! tensor's ACTUAL rank at step time, which is the same backstop
//! `muon_step` uses (`adamw_route > 0.5 or len(s) != 2`).
//!
//! ## Why exemption is expressed as λ = 0
//!
//! Every optimizer arm already branches on the decay being zero — the
//! stdlib `adamw_step` / `muon_step` guard `if weight_decay > 0.0`, the
//! FASE update program skips its `wd·θ` term when `wd == 0.0`, and the
//! fused kernels take `has_wd` as a launch parameter. Handing an exempt
//! parameter λ = 0 therefore routes it down each arm's EXISTING, already
//! gated no-decay path: no new arithmetic, and a decayed parameter's
//! numerics are bit-identical to before this feature existed.
//!
//! This is the one place the rule is implemented. Codegen calls it per
//! parameter inside the optimizer loop; `nsl_fase_fused_adamw_step_multi`
//! calls it per parameter while bucketing launches. Two copies of an
//! eligibility predicate that must agree exactly is a bug pattern this
//! file exists to avoid.

use crate::tensor::NslTensor;

/// Resolve the weight decay for ONE parameter.
///
/// * `param` — the parameter tensor (may be 0; treated as "no rank info",
///   which decays, matching the pre-feature default).
/// * `static_exempt` — 1 when the compile-time role table exempted this
///   parameter (its role appeared in `no_decay`).
/// * `exempt_non_rank2` — 1 when `no_decay` includes `"vector"`. Resolved
///   here against the runtime rank, for the reasons in the module docs.
/// * `wd` — the configured `weight_decay`.
///
/// Returns `wd` to decay, or `0.0` to exempt.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_optim_param_wd(
    param: i64,
    static_exempt: i64,
    exempt_non_rank2: i64,
    wd: f64,
) -> f64 {
    if static_exempt != 0 {
        return 0.0;
    }
    if exempt_non_rank2 != 0 && param != 0 {
        // Raw read + magic conjunct, deliberately NOT `from_ptr_ref`: the
        // fallthrough here is "if this is not a live tensor, do not exempt
        // it — decay it", which the accessor would turn into an abort on
        // a path that runs once per parameter per step.
        //
        // SAFETY: `param` is a live NslTensor pointer from the optimizer's
        // parameter list; the caller holds it for the duration of the step.
        let t = unsafe { &*(param as *const NslTensor) };
        if t.magic == crate::tensor::TENSOR_MAGIC && t.ndim != 2 {
            return 0.0;
        }
    }
    wd
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::nsl_tensor_free;

    fn tensor_with_shape(shape: &[i64]) -> i64 {
        let total: i64 = shape.iter().product();
        let data = crate::memory::checked_alloc(total as usize * 4);
        let sh = crate::memory::checked_alloc(shape.len() * 8) as *mut i64;
        for (i, &s) in shape.iter().enumerate() {
            unsafe { *sh.add(i) = s };
        }
        let strides = NslTensor::compute_strides(sh, shape.len() as i64);
        let t = Box::new(NslTensor::new(
            data as *mut std::ffi::c_void,
            sh,
            strides,
            shape.len() as i64,
            total,
            0,
            1,
            1,
            0,
        ));
        NslTensor::publish(t)
    }

    #[test]
    fn no_exemption_returns_configured_wd() {
        let t = tensor_with_shape(&[4, 4]);
        assert_eq!(nsl_optim_param_wd(t, 0, 0, 0.1), 0.1);
        // Even a rank-1 tensor decays when "vector" was not listed.
        let v = tensor_with_shape(&[4]);
        assert_eq!(nsl_optim_param_wd(v, 0, 0, 0.1), 0.1);
        nsl_tensor_free(t);
        nsl_tensor_free(v);
    }

    #[test]
    fn static_exemption_wins_regardless_of_rank() {
        let t = tensor_with_shape(&[4, 4]);
        assert_eq!(nsl_optim_param_wd(t, 1, 0, 0.1), 0.0);
        assert_eq!(nsl_optim_param_wd(t, 1, 1, 0.1), 0.0);
        nsl_tensor_free(t);
    }

    /// The load-bearing case: an RMSNorm gain is role `hidden` (no static
    /// rank is derivable from `ones([dim])`), so ONLY the runtime rank check
    /// can exempt it.
    #[test]
    fn vector_exemption_is_decided_at_runtime() {
        let gain = tensor_with_shape(&[512]);
        let matrix = tensor_with_shape(&[512, 512]);
        assert_eq!(
            nsl_optim_param_wd(gain, 0, 1, 0.1),
            0.0,
            "a rank-1 gain must be exempt when no_decay includes \"vector\""
        );
        assert_eq!(
            nsl_optim_param_wd(matrix, 0, 1, 0.1),
            0.1,
            "a rank-2 matrix must still be decayed"
        );
        // Rank 3+ is also not a matrix.
        let cube = tensor_with_shape(&[2, 3, 4]);
        assert_eq!(nsl_optim_param_wd(cube, 0, 1, 0.1), 0.0);
        nsl_tensor_free(gain);
        nsl_tensor_free(matrix);
        nsl_tensor_free(cube);
    }

    #[test]
    fn null_param_decays_rather_than_guessing() {
        // ZeRO / conditional-v configurations can hand a null slot to the
        // per-param loop. Defaulting to "decay" keeps the pre-feature
        // behaviour instead of silently exempting a real weight.
        assert_eq!(nsl_optim_param_wd(0, 0, 1, 0.1), 0.1);
    }

    #[test]
    fn zero_wd_stays_zero() {
        let t = tensor_with_shape(&[4, 4]);
        assert_eq!(nsl_optim_param_wd(t, 0, 0, 0.0), 0.0);
        nsl_tensor_free(t);
    }
}
