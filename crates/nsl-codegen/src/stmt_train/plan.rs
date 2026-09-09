//! The `TrainPlan` carrier — the planning-time facts of one train block as
//! plain data (roadmap A1; step 1 of the design in
//! `docs/superpowers/specs/2026-09-08-a1-train-plan-ir-design.md`).
//!
//! The driver builds one after the setup phases have resolved the header,
//! the contract, the parameter list and the optimizer state, and the late
//! emitters (`optimizer_step`, `scheduler_step`, `csla_window`,
//! `health_hooks`, `identity`) take `&plan` in place of the copies of these
//! values their `Inputs` structs used to carry. Nothing here is a Cranelift
//! handle: `Value`s and `Variable`s stay in the `Inputs` structs (the
//! `EmitState` of the design's step 3).
//!
//! The carrier holds, for now, exactly the facts the late emitters read; it
//! grows as the remaining emitters and the planning passes migrate onto it
//! (`Tapes`, `TechniquePlans`, the `data:` section, the callbacks). A field
//! is added when an emitter reads it, so `dead_code` keeps the struct
//! honest about what is actually consumed through the plan.

use crate::fase::FasePlan;
use crate::param_roles::NoDecayScope;
use nsl_semantic::optim_config::ResolvedScheduler;

/// One train block's resolved plan.
pub(crate) struct TrainPlan {
    /// The header and contract: optimizer, scheduler, hyper-parameters,
    /// the FASE plan and the admissions that were decided at planning time.
    pub(crate) spec: TrainSpec,
    /// The model's parameters, as facts.
    pub(crate) params: ParamPlan,
    /// Accumulation and checkpointing.
    pub(crate) schedule: TrainSchedule,
}

/// The resolved optimizer / scheduler contract and the header's knobs.
pub(crate) struct TrainSpec {
    pub(crate) optimizer_name: String,
    pub(crate) lr_value: f64,
    pub(crate) momentum_value: f64,
    pub(crate) dampening_value: f64,
    pub(crate) weight_decay_value: f64,
    pub(crate) no_decay_scope: NoDecayScope,
    pub(crate) nesterov_value: bool,
    pub(crate) beta1_value: f64,
    pub(crate) beta2_value: f64,
    pub(crate) eps_value: f64,
    pub(crate) ns_steps_value: f64,
    pub(crate) adamw_lr_value: Option<f64>,
    pub(crate) scheduler: Option<ResolvedScheduler>,
    /// `f64::MAX` = no clipping.
    pub(crate) grad_clip: f64,
    pub(crate) fase_plan: FasePlan,
    pub(crate) fase_deferred: bool,
    /// The `--layerwise-accum` admission decided at planning time.
    pub(crate) csla_active: bool,
}

/// The model's parameters, as facts (no handles).
pub(crate) struct ParamPlan {
    /// Parameter paths; list order == runtime list index.
    pub(crate) paths: Vec<String>,
    /// Optimizer state buffers per parameter (0, 1 or 2).
    pub(crate) num_state_buffers: usize,
}

/// Accumulation and checkpointing.
pub(crate) struct TrainSchedule {
    pub(crate) grad_accumulation_steps: i64,
    pub(crate) checkpoint_save_path: Option<String>,
    pub(crate) checkpoint_every: i64,
}
