//! `EmitState`: the Cranelift handles one train block's setup emitters
//! produce and its late emitters consume (roadmap A1, `TrainPlan` step 3 of
//! `docs/superpowers/specs/2026-09-08-a1-train-plan-ir-design.md`).
//!
//! `TrainPlan` (`plan.rs`) carries the block's facts and no handles. This is
//! the other half of that split. The runtime lists the setup phase
//! allocates, the optimizer state, the accumulation buffers, the resume
//! handles and the loop variables that outlive one step are one value,
//! built once the setup has run. The late emitters (`optimizer_step`,
//! `csla_window`, `scheduler_step`, `health_hooks`) take `&EmitState` next
//! to `&TrainPlan`, where each used to list the same dozen handles in its
//! own `Inputs` struct. What stays in an `Inputs` struct is only what one
//! micro-batch produces: its gradient list, its loss, its `should_step`
//! flag, and the CSLA window's pending save.
//!
//! Every field is a `Value` or a `Variable`, so the struct is `Copy`. An
//! emitter destructures it the way it destructured its `Inputs`, and its
//! body, and so the CLIF it emits, does not change. The CLIF snapshots are
//! the proof, as for every earlier step.

use cranelift_codegen::ir::Value;
use cranelift_frontend::Variable;

/// The setup handles of one train block.
#[derive(Clone, Copy)]
pub(crate) struct EmitState {
    // ── Parameter lists (`model_params`, `param_lists`) ──────────────────
    /// The runtime list of parameter tensors; list order is the plan's
    /// `params.paths` order.
    pub(crate) param_list: Value,
    /// `params.paths.len()`, as an `i64`.
    pub(crate) num_params_val: Value,
    /// FASE's per-parameter mode byte table, when the plan has one.
    pub(crate) mode_table_base: Option<Value>,
    /// CPDT's per-parameter moment dtype codes (m, v).
    pub(crate) cpdt_precision_dtypes: Option<(Value, Value)>,
    /// The parameter names a checkpoint records.
    pub(crate) checkpoint_names_list: Option<Value>,
    /// Muon's per-parameter route flags.
    pub(crate) muon_route_list: Option<Value>,
    /// AdamW's per-parameter decay-exempt flags.
    pub(crate) decay_exempt_list: Option<Value>,

    // ── Optimizer state (`optimizer_state`) ──────────────────────────────
    pub(crate) state_list_1: Value,
    pub(crate) state_list_2: Value,
    /// Muon's momentum dtype codes.
    pub(crate) muon_state_m_codes: Option<Value>,
    /// The one-shot moment-fill latch.
    pub(crate) moment_fill_latch: Option<Value>,

    // ── Accumulation and the CSLA window ─────────────────────────────────
    /// The gradient accumulation buffers, when `grad_accumulation_steps > 1`.
    pub(crate) accum_list: Option<Value>,
    /// The CSLA window's `(saves_outer, dicts)` lists.
    pub(crate) csla_buffers: Option<(Variable, Variable)>,

    // ── Identity and resume (`identity`) ─────────────────────────────────
    /// Whether the block reads a DataLoader, when that is a runtime value.
    pub(crate) has_dataloader: Option<Value>,
    /// The DataLoader handle a checkpoint records.
    pub(crate) checkpoint_dl_handle: Value,

    // ── Loop variables ───────────────────────────────────────────────────
    pub(crate) lr_var: Variable,
    pub(crate) step_count_var: Variable,
    pub(crate) epoch_counter_var: Variable,
}
