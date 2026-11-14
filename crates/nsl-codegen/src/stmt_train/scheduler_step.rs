//! The learning-rate schedule and the tail of the train block's step.
//!
//! Two emitters, at two points of the step:
//!
//! * [`Compiler::emit_scheduler_lr`] runs at the TOP of every micro-step,
//!   before the forward: it calls the resolved scheduler with the current
//!   0-based step and redefines the learning rate. The step is the
//!   micro-batch count by default, so under `grad_accumulation = N` the
//!   schedule's `warmup_steps` / `total_steps` are in micro-batches;
//!   `schedule_unit = "optimizer_step"` in the train header makes it the
//!   optimizer-step index, `step_count / N`. Either way
//!   the optimizer update taken at step `k` uses `schedule(k)` — on the
//!   standard step, on the CSLA window schedule that updates parameters
//!   inside the backward, and on the first step after a checkpoint resume.
//!   The resolved scheduler's constants are passed in the stdlib signature
//!   order after the auto-injected base rate and step.
//!
//!   It used to run AFTER the optimizer step. The rate computed at step
//!   `k` was therefore first consumed at step `k + 1`, and the first
//!   optimizer step of a run (and of every resume without accumulation)
//!   ran at the optimizer's constructor rate — a `warmup_cosine` run took
//!   its first AdamW step, the one where every parameter moves by a full
//!   `lr`, at the peak rate the warmup exists to defer. Measured by fitting
//!   the applied rate per micro-batch against reference gradients:
//!   `1.0e-3, 0, 1.25e-4, 2.5e-4, ...` for `warmup_steps=8` at base 1e-3.
//!
//! * [`Compiler::emit_scheduler_step`] is the tail after the optimizer step
//!   (sections 7h onward of the driver): the step-count increment, and the
//!   Milestone B periodic full-train-state checkpoint, which fires when the
//!   post-increment step count is a multiple of `checkpoint_every`
//!   optimizer steps — an optimizer-step boundary, where the accumulation
//!   buffers were just zeroed and the offload drain has completed — and
//!   records the loader handle and the epoch to resume at.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1). The train-block
//! CLIF snapshots (`tests/train_clif_snapshots.rs`) pin the scheduler call
//! and the increment on every fixture; `tests/scheduler_applied_lr_gate.rs`
//! (nsl-cli) pins the rate each step actually trains at.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt_train::emit_state::EmitState;
use crate::stmt_train::plan::TrainPlan;

/// Every binding of `compile_train_block_inner` the step tail reads;
/// names are the driver's.
pub(crate) struct SchedulerStepInputs<'a> {
    /// The block's planning-time facts (roadmap A1, TrainPlan step 1).
    pub(crate) plan: &'a TrainPlan,
    /// The setup handles (roadmap A1, TrainPlan step 3).
    pub(crate) emit: &'a EmitState,
}

impl Compiler<'_> {
    /// Emit the scheduler call that sets this micro-step's learning rate
    /// (see the module header). A no-op without a `scheduler:` section: the
    /// rate stays the optimizer's constructor value.
    pub(crate) fn emit_scheduler_lr(
        &mut self,
        builder: &mut FunctionBuilder,
        inputs: SchedulerStepInputs<'_>,
    ) -> Result<(), CodegenError> {
        let SchedulerStepInputs { plan, emit } = inputs;
        let EmitState { lr_var, step_count_var, .. } = *emit;
        let lr_value = plan.spec.lr_value;
        let scheduler = &plan.spec.scheduler;

        // Scheduler: this step's learning rate, from the pre-increment step
        // count — step 0 trains at the step-0 rate. All names/kwargs/defaults
        // were resolved by resolve_optim_config — this just emits the
        // constants in the stdlib signature order after the auto-injected
        // (base_lr, step).
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
            // `schedule_unit = "optimizer_step"`: the schedule's step is the
            // index of the optimizer step this micro-batch belongs to, so
            // every micro-batch of a window runs at one rate and
            // `warmup_steps` / `total_steps` count optimizer steps. The
            // default (micro-batches) emits nothing here.
            let window = plan.schedule.grad_accumulation_steps;
            let schedule_step = if plan.schedule.schedule_in_optimizer_steps && window > 1 {
                let n = builder.ins().iconst(cl_types::I64, window);
                builder.ins().sdiv(step_count_val, n)
            } else {
                step_count_val
            };
            let step_float = builder.ins().fcvt_from_sint(cl_types::F64, schedule_step);

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

        Ok(())
    }

    /// Emit the step-count increment and the periodic checkpoint (see the
    /// module header).
    pub(crate) fn emit_scheduler_step(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: SchedulerStepInputs<'_>,
    ) -> Result<(), CodegenError> {
        let SchedulerStepInputs {
            plan,
            emit,
        } = inputs;
        let EmitState {
            checkpoint_dl_handle,
            checkpoint_names_list,
            epoch_counter_var,
            has_dataloader,
            param_list,
            state_list_1,
            state_list_2,
            step_count_var,
            ..
        } = *emit;
        // TrainPlan step 1 (roadmap A1): the facts this phase used to receive
        // as copied fields, read from the carrier under their old names so
        // the body below is unchanged.
        let checkpoint_every = plan.schedule.checkpoint_every;
        let checkpoint_save_path = &plan.schedule.checkpoint_save_path;
        let grad_accumulation_steps = plan.schedule.grad_accumulation_steps;

        // 7h. Increment step count
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
