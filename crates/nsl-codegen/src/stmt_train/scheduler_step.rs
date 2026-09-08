//! The tail of the train block's step, after the optimizer step (sections
//! 7g2–7h of the driver): the scheduler call that redefines the learning
//! rate (emitted BEFORE the step-count increment so step 0 produces the
//! step-0 rate; the resolved scheduler's constants are passed in the
//! stdlib signature order after the auto-injected base rate and step),
//! the step-count increment, and the Milestone B periodic full-train-state
//! checkpoint, which fires when the post-increment step count is a
//! multiple of `checkpoint_every` optimizer steps — an optimizer-step
//! boundary, where the accumulation buffers were just zeroed and the
//! offload drain has completed — and records the loader handle and the
//! epoch to resume at.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 132 lines,
//! 14 inputs ([`SchedulerStepInputs`]). The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the scheduler call and the
//! increment on every fixture.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use nsl_semantic::optim_config::ResolvedScheduler;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

/// Every binding of `compile_train_block_inner` the step tail reads;
/// names are the driver's.
pub(crate) struct SchedulerStepInputs<'a> {
    /// The DataLoader handle recorded in a checkpoint (null without a loader).
    pub(crate) checkpoint_dl_handle: Value,
    /// Optimizer steps between periodic checkpoints.
    pub(crate) checkpoint_every: i64,
    /// The parameter-name list built at setup whenever `checkpoint_save` is set.
    pub(crate) checkpoint_names_list: Option<Value>,
    /// The `checkpoint_save` path; `None` disables the periodic checkpoint.
    pub(crate) checkpoint_save_path: &'a Option<String>,
    /// The epoch counter variable.
    pub(crate) epoch_counter_var: Variable,
    pub(crate) grad_accumulation_steps: i64,
    /// The DataLoader value when the `data:` section declares one.
    pub(crate) has_dataloader: Option<Value>,
    /// The base learning rate.
    pub(crate) lr_value: f64,
    /// The learning-rate variable the scheduler redefines.
    pub(crate) lr_var: Variable,
    /// The model parameter list.
    pub(crate) param_list: Value,
    /// The resolved scheduler, if any.
    pub(crate) scheduler: &'a Option<ResolvedScheduler>,
    /// The first optimizer-state list.
    pub(crate) state_list_1: Value,
    /// The second optimizer-state list.
    pub(crate) state_list_2: Value,
    /// The step counter variable (incremented here, after the scheduler call).
    pub(crate) step_count_var: Variable,
}

impl Compiler<'_> {
    /// Emit the scheduler call, the step-count increment and the periodic
    /// checkpoint (see the module header).
    pub(crate) fn emit_scheduler_step(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: SchedulerStepInputs<'_>,
    ) -> Result<(), CodegenError> {
        let SchedulerStepInputs {
            checkpoint_dl_handle,
            checkpoint_every,
            checkpoint_names_list,
            checkpoint_save_path,
            epoch_counter_var,
            grad_accumulation_steps,
            has_dataloader,
            lr_value,
            lr_var,
            param_list,
            scheduler,
            state_list_1,
            state_list_2,
            step_count_var,
        } = inputs;

        // 7g2. Scheduler: update learning rate if a scheduler is configured.
        // NOTE: step_count is incremented AFTER the scheduler call so that
        // step 0 produces the step-0 learning rate (e.g. warmup starts
        // correctly). All names/kwargs/defaults were resolved by
        // resolve_optim_config — this just emits the constants in the
        // stdlib signature order after the auto-injected (base_lr, step).
        if let Some(sched) = scheduler {
            use nsl_semantic::optim_config::ResolvedScheduler;

            let mangled = format!("nsl__optim__schedulers__{}", sched.fn_name());

            // Find the actual function name (check functions/runtime_fns
            // with fallback).
            let sched_fn = if self.registry.functions.contains_key(mangled.as_str()) {
                mangled.clone()
            } else {
                let simple = sched.fn_name().to_string();
                if self.registry.functions.contains_key(simple.as_str()) {
                    simple
                } else if self.registry.runtime_fns.contains_key(mangled.as_str()) {
                    mangled.clone()
                } else if self.registry.runtime_fns.contains_key(simple.as_str()) {
                    simple
                } else {
                    mangled.clone()
                }
            };

            let base_lr_val = builder.ins().f64const(lr_value);
            let step_count_val = builder.use_var(step_count_var);
            let step_float = builder.ins().fcvt_from_sint(cl_types::F64, step_count_val);

            let extra: Vec<f64> = match sched {
                ResolvedScheduler::ConstantLr => vec![],
                ResolvedScheduler::StepLr { step_size, gamma } => vec![*step_size, *gamma],
                ResolvedScheduler::ExponentialLr { gamma } => vec![*gamma],
                ResolvedScheduler::LinearDecay {
                    total_steps,
                    end_factor,
                } => vec![*total_steps, *end_factor],
                ResolvedScheduler::CosineAnneal { t_max, eta_min } => vec![*t_max, *eta_min],
                ResolvedScheduler::WarmupCosine {
                    warmup_steps,
                    total_steps,
                    min_lr,
                } => vec![*warmup_steps, *total_steps, *min_lr],
                ResolvedScheduler::OneCycle {
                    max_lr,
                    total_steps,
                    pct_start,
                } => vec![*max_lr, *total_steps, *pct_start],
            };

            let mut call_args = vec![base_lr_val, step_float];
            for v in extra {
                call_args.push(builder.ins().f64const(v));
            }
            let new_lr = self.compile_call_by_name(builder, &sched_fn, &call_args)?;

            builder.def_var(lr_var, new_lr);
        }

        // 7h. Increment step count (after scheduler so step 0 uses the initial LR)
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iadd_imm_s(sc, 1);
        builder.def_var(step_count_var, one_i64);

        // Milestone B: periodic full-train-state checkpoint. Post-increment
        // step_count is a multiple of grad_accumulation exactly at optimizer-
        // step boundaries — where 7g just zeroed the accum buffers and the
        // offload drain has completed — so the state on disk is always a
        // clean boundary state. Fires every `checkpoint_every` optimizer
        // steps; the runtime writes tmp files and renames, so a crash mid-
        // save leaves the previous checkpoint intact.
        if let Some(save_path) = checkpoint_save_path.clone() {
            let names_list = checkpoint_names_list
                .expect("names list is built at setup whenever checkpoint_save is set");
            let interval = builder.ins().iconst(
                cl_types::I64,
                checkpoint_every.saturating_mul(grad_accumulation_steps.max(1)),
            );
            let sc_now = builder.use_var(step_count_var);
            let rem = builder.ins().srem(sc_now, interval);
            let z = builder.ins().iconst(cl_types::I64, 0);
            let fire = builder.ins().icmp(IntCC::Equal, rem, z);
            let save_block = builder.create_block();
            let cont_block = builder.create_block();
            builder.ins().brif(fire, save_block, &[], cont_block, &[]);
            builder.switch_to_block(save_block);
            builder.seal_block(save_block);
            state.current_block = Some(save_block);
            self.intern_string(&save_path)?;
            let path_val = self.compile_string_literal(builder, &save_path)?;
            let path_len = builder.ins().iconst(cl_types::I64, save_path.len() as i64);
            // Item 8: the loader handle and the epoch make the saved state a
            // position in the data stream, not just a step number. Read here
            // — inside the fire block, at an optimizer-step boundary — so the
            // recorded slot is the one the next batch would come from.
            //
            // What is recorded is the epoch to RESUME AT, which is not always
            // the epoch in progress. With a DataLoader the epoch is a loop
            // over batches and the checkpoint lands part-way through it, so
            // the resume point is (this epoch, next slot). WITHOUT a loader
            // the epoch body runs exactly once and has just finished, so the
            // resume point is the NEXT epoch — recording the current one
            // would re-run a completed epoch on every restart.
            let epoch_ctr = builder.use_var(epoch_counter_var);
            let epoch_now = if has_dataloader.is_some() {
                epoch_ctr
            } else {
                builder.ins().iadd_imm_s(epoch_ctr, 1)
            };
            self.compile_call_by_name(
                builder,
                "nsl_train_checkpoint_save",
                &[
                    path_val,
                    path_len,
                    names_list,
                    param_list,
                    state_list_1,
                    state_list_2,
                    sc_now,
                    checkpoint_dl_handle,
                    epoch_now,
                ],
            )?;
            builder.ins().jump(cont_block, &[]);
            builder.switch_to_block(cont_block);
            builder.seal_block(cont_block);
            state.current_block = Some(cont_block);
        }

        Ok(())
    }
}
