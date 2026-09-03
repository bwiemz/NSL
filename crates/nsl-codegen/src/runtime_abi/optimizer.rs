//! Optimizer steps and their fused/batched variants.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_OPTIMIZER: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // P1 Muon items 8+10: planned Newton-Schulz orthogonalization primitive
    // (no .item() sync; device-resident norm; materialized tall/wide).
    ("nsl_tensor_muon_orthogonalize", &[types::I64, types::F64], Some(types::I64)),
    // Muon perf campaign: batched momentum + Newton-Schulz + param update
    // over (params, grads, m, routes) lists; f64 lr/momentum/wd, i64
    // nesterov flag, f64 ns_steps.
    (
        "nsl_muon_step_batch",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::F64,
            types::F64,
            types::F64,
            types::I64,
            types::F64,
        ],
        None,
    ),
    // Muon internal profiler (perf-campaign item 2): explicit begin/end
    // region markers + on-demand report, no-ops unless NSL_MUON_PROF is set.
    ("nsl_muon_prof_begin", &[types::I64], None),
    ("nsl_muon_prof_end", &[types::I64], None),
    ("nsl_muon_prof_report", &[], None),
    // Fusion item 1: multi-tensor fused AdamW final step over the whole
    // param/m/v/m_partial lists (one pointer-table launch, bit-identical).
    //
    // The two trailing I64s are the AdamW parameter groups (0 = no no_decay);
    // the final F64 is mp_scale, the two-phase-clip factor folded into the
    // m_partial read, 1.0 = unclipped (exact legacy behaviour, branched around
    // in-kernel). The two features are independent and BOTH are load-bearing —
    // this list, the emission order in `stmt.rs`, and the Rust signature in
    // `fase_step.rs` must stay in lockstep or arguments silently shift.
    (
        "nsl_fase_fused_adamw_step_multi",
        &[
            types::I64, // params_list
            types::I64, // m_list
            types::I64, // v_list
            types::I64, // mp_list
            types::F64, // lr
            types::F64, // beta1
            types::F64, // one_minus_beta1
            types::F64, // beta2
            types::F64, // one_minus_beta2
            types::F64, // eps
            types::F64, // wd
            types::F64, // bc1_inv
            types::F64, // bc2_inv
            types::I64, // wd_exempt_list (0 = no parameter groups)
            types::I64, // wd_exempt_non_rank2
            types::F64, // mp_scale (1.0 = unclipped)
        ],
        None,
    ),
    // Item 8, CSLA half: subset variant — same contract with an extra NslList
    // of i64 param indices (arg 5) selecting which parameters to step. The
    // CSLA layerwise window emits one call per layer group. Same lockstep
    // warning as above: this list, the `stmt.rs` emission order, and the Rust
    // signature in `fase_step.rs` must not drift.
    (
        "nsl_fase_fused_adamw_step_multi_idx",
        &[
            types::I64, // params_list
            types::I64, // m_list
            types::I64, // v_list
            types::I64, // mp_list
            types::I64, // idx_list (NslList of i64 param indices)
            types::F64, // lr
            types::F64, // beta1
            types::F64, // one_minus_beta1
            types::F64, // beta2
            types::F64, // one_minus_beta2
            types::F64, // eps
            types::F64, // wd
            types::F64, // bc1_inv
            types::F64, // bc2_inv
            types::I64, // wd_exempt_list (0 = flat wd; CSLA passes 0)
            types::I64, // wd_exempt_non_rank2
            types::F64, // mp_scale (1.0 = unclipped; CSLA refuses grad_clip)
        ],
        None,
    ),
    // Item 8, bf16-SR arm: the SR twin of the idx entry above. Same contract
    // minus mp_scale (the SR path has no clip fold; the layerwise schedule
    // refuses grad_clip) plus the trailing `step` the SR counter stream is
    // keyed on. No full-list twin exists: bf16-sr requires --weight-stream,
    // which requires --layerwise-accum, so SR only ever batches from the
    // CSLA group update. Same lockstep warning: this list, the `stmt.rs`
    // emission order, and the Rust signature in `sr_bf16.rs` must not drift.
    (
        "nsl_fase_fused_adamw_step_bf16sr_multi_idx",
        &[
            types::I64, // params_list
            types::I64, // m_list
            types::I64, // v_list
            types::I64, // mp_list
            types::I64, // idx_list (NslList of i64 param indices)
            types::F64, // lr
            types::F64, // beta1
            types::F64, // one_minus_beta1
            types::F64, // beta2
            types::F64, // one_minus_beta2
            types::F64, // eps
            types::F64, // wd
            types::F64, // bc1_inv
            types::F64, // bc2_inv
            types::I64, // wd_exempt_list (0 = flat wd; CSLA passes 0)
            types::I64, // wd_exempt_non_rank2
            types::I64, // step (SR counter key)
        ],
        None,
    ),
    // FASE two-phase-clip Phase A: global sum-of-squares over an NslList of
    // m_partial tensors with ONE pipeline drain (batched device reduction).
    ("nsl_fase_sum_sq_list", &[types::I64], Some(types::F64)),
    // FASE Deferred bias correction: 1/(1 - base^step).  Scalar, no tensor args.
    (
        "nsl_bias_correction_inv",
        &[types::F64, types::I64],
        Some(types::F64),
    ),
    // AdamW parameter groups: resolve ONE param's weight decay from its
    // compile-time role flag + its runtime rank.
    // (param, static_exempt, exempt_non_rank2, wd) -> wd_for_this_param
    (
        "nsl_optim_param_wd",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::F64),
    ),
    // FASE Deferred two-phase grad clip: sum of squared elements, in-place scale.
    (
        "nsl_tensor_sum_sq",
        &[types::I64],
        Some(types::F64),
    ),
    (
        "nsl_tensor_mul_scalar_inplace",
        &[types::I64, types::F64],
        None,
    ),
    // Item 3: the compiled ParameterPlan, cross-checked against the three
    // residency tables at run time. LOCKSTEP: the flag bits are
    // `nsl_runtime::param_plan::PLAN_*`, re-exported by
    // `crate::parameter_plan` — there is no second copy to keep in sync.
    ("nsl_param_plan_declare", &[types::I64, types::I64, types::I64], Some(types::I64)), // (tensor, idx, flags)
    ("nsl_param_plan_verify", &[], Some(types::I64)),
    ("nsl_param_plan_teardown", &[], Some(types::I64)),
    // P4 item 17: SR-BF16 authoritative weights
    ("nsl_sr_bf16_enable", &[], None),
    ("nsl_sr_bf16_note_param", &[types::I64, types::I64], None), // (tensor, idx)
    // P4 item 18 rung 2: (src_f32, dst_bf16, step, param_idx)
    ("nsl_muon_state_sr_store", &[types::I64, types::I64, types::I64, types::I64], None),
    (
        "nsl_sr_bf16_step_adamw",
        &[
            types::I64, types::I64, types::I64, types::I64, // theta, m, v, mp
            types::F64, types::F64, types::F64, types::F64, types::F64, // lr, b1, omb1, b2, omb2
            types::F64, types::F64, types::F64, types::F64, // eps, wd, bc1_inv, bc2_inv
            types::I64, // step
        ],
        None,
    ),
    ("nsl_zero3_reduce_grad_slot", &[types::I64, types::I64], Some(types::I64)), // (list, idx)
    // p9: fused per-param FASE-Deferred AdamW step — one launch for the whole
    // m/v/θ update. (theta, m, v, m_partial, lr, β₁, 1-β₁, β₂, 1-β₂, ε, wd,
    // bc1_inv, bc2_inv) → void.
    (
        "nsl_fase_fused_adamw_step",
        &[
            types::I64, types::I64, types::I64, types::I64,
            types::F64, types::F64, types::F64, types::F64, types::F64,
            types::F64, types::F64, types::F64, types::F64,
        ],
        None,
    ),
];
