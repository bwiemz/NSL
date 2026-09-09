//! The train block's per-step diagnostics, emitted after the backward and
//! before gradient clipping (sections 7e1b–7e1c of the driver): the
//! `--debug-training` gradient checksum, the P0.3 grad-integrity scan of
//! the materialized gradient list, and the Dev Tools health-monitor hooks
//! (the per-step loss record — also under `--inspect`, whose predicates
//! read the loss back — the per-parameter gradient norms, the per-parameter
//! weight norms at step 0 and every 100 steps, and the snapshot flush).
//! With none of those options on, the emission is empty.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 195 lines,
//! 7 inputs ([`HealthHooksInputs`]). The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the emitted calls on every fixture
//! that turns a diagnostic on.

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::Module;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt_train::plan::TrainPlan;
use crate::stmt::parse_layer_idx_for_health;

/// Every binding of `compile_train_block_inner` the per-step diagnostics
/// read; names are the driver's.
pub(crate) struct HealthHooksInputs<'a> {
    /// Whether the FASE-Deferred hook consumed the gradients (then `grads_list` is a null sentinel and the per-gradient hooks are skipped).
    pub(crate) fase_hook_active: bool,
    /// The per-batch gradient list.
    pub(crate) grads_list: Value,
    /// The loss tensor of this step.
    pub(crate) loss_val: Value,
    /// `param_paths.len()` as an `iconst`.
    pub(crate) num_params_val: Value,
    /// The model parameter list.
    pub(crate) param_list: Value,
    /// The step counter variable.
    pub(crate) step_count_var: Variable,
    /// The block's planning-time facts (roadmap A1, TrainPlan step 1).
    pub(crate) plan: &'a TrainPlan,
}

impl Compiler<'_> {
    /// Emit the per-step diagnostics (see the module header).
    pub(crate) fn emit_train_health_hooks(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: HealthHooksInputs<'_>,
    ) -> Result<(), CodegenError> {
        let HealthHooksInputs {
            plan,
            fase_hook_active,
            grads_list,
            loss_val,
            num_params_val,
            param_list,
            step_count_var,
        } = inputs;
        // TrainPlan step 1 (roadmap A1): the facts this phase used to receive
        // as copied fields, read from the carrier under their old names so
        // the body below is unchanged.
        let param_paths: &[String] = &plan.params.paths;


        // 7e1b. Debug training: emit gradient checksum to catch silent corruption.
        // Prints sum(abs(grad)) per parameter — detects NaN, zero, and misrouted gradients.
        // Skip when hook active — grads_list is a null sentinel.
        if self.compile_options.diagnostics.debug_training && !fase_hook_active {
            self.compile_call_by_name(
                builder,
                "nsl_debug_grad_checksum",
                &[grads_list, num_params_val],
            )?;
        }

        // 7e1b'. P0.3 gradient-integrity gate (FullBuffer / composite path):
        // scan the materialized grads list once per step. Skipped when the
        // FASE hook is active (grads_list is a null sentinel) — that path is
        // instrumented per-parameter inside the hook (step_begin/note/step_end).
        if self.compile_options.diagnostics.grad_integrity && !fase_hook_active {
            self.compile_call_by_name(
                builder,
                "nsl_grad_integrity_check",
                &[grads_list, num_params_val],
            )?;
        }

        // 7e1c. Dev Tools Phase 4 Task 4: health-monitor hooks.
        // Emits per-step loss, per-parameter gradient norm, per-parameter
        // weight norm (step 0 + every 100 steps), and a snapshot flush every
        // 100 steps.  Gated on `health_monitor`; when neither it nor
        // `inspect_enabled` is on the IR is byte-identical to pre-phase-4.
        //
        // TODO(phase4-fase): splice grad-norm emission into the FASE per-layer
        // loop when FASE is active.  Phase 4 Task 4 ships the standard-
        // backward-only path — grads_list is indexed the same whether the
        // primary backward was tape-AD or source-AD.

        // (a) Record loss: scalarize loss_val and call
        //     nsl_health_record_loss(loss_scalar, step).
        // Also emitted when only `--inspect` is on: `@inspect` predicates read
        // `loss` back through nsl_health_get_last_loss, so without this call
        // the collector would stay empty and `loss` would read 0.0 (the exact
        // silent-wrong the getter replaced).
        if self.compile_options.dev_tools.health_monitor || self.compile_options.dev_tools.inspect_enabled {
            let loss_scalar = self.compile_call_by_name(
                builder,
                "nsl_tensor_item",
                &[loss_val],
            )?;
            let step_now = builder.use_var(step_count_var);
            self.compile_call_by_name(
                builder,
                "nsl_health_record_loss",
                &[loss_scalar, step_now],
            )?;
        }

        if self.compile_options.dev_tools.health_monitor {
            use cranelift_codegen::ir::{types as cl_types, MemFlagsData};
            let _ = MemFlagsData::trusted(); // keep import valid across cfgs

            // Precompute step-gating flags shared by grad/weight/flush hooks.
            let zero_i64_h = builder.ins().iconst(cl_types::I64, 0);
            let step_h = builder.use_var(step_count_var);
            let hundred = builder.ins().iconst(cl_types::I64, 100);
            let step_mod = builder.ins().srem(step_h, hundred);
            let is_flush_due = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                step_mod,
                zero_i64_h,
            );
            // Init == step 0.  We reuse `is_flush_due` for the weight-norm
            // periodic check (step 0 also satisfies step % 100 == 0), so a
            // single gate covers both "init" and "every 100".

            // (b) Per-parameter gradient norms (unrolled over compile-time
            //     param_paths; grads_list / param_list are indexed by idx).
            //
            // P0 cert campaign FIX: under the FASE-Deferred source-AD hook
            // (AdamW/Adam + grad_accumulation >= 2) the hook accumulates
            // each gradient into m_partial and FREES it during the
            // backward — grads_list entries DANGLE here, and the l2_norm
            // read segfaulted every `--monitor` run at 500M/1B scale
            // (SIGSEGV in emitted code, silently reported as exit 1). Skip
            // per-micro-batch grad norms on that path and say so loudly;
            // loss, weight-norm and flush recording still run.
            if !fase_hook_active {
                for (i, path) in param_paths.iter().enumerate() {
                    let path_data_id = self.intern_string(path)?;
                    let gv = self.module.declare_data_in_func(path_data_id, builder.func);
                    let path_ptr = builder.ins().symbol_value(cl_types::I64, gv);
                    let path_len = builder
                        .ins()
                        .iconst(cl_types::I64, path.len() as i64);
                    let layer_idx = parse_layer_idx_for_health(path);
                    let layer_idx_val = builder
                        .ins()
                        .iconst(cl_types::I32, layer_idx as i64);

                    let idx_val = builder.ins().iconst(cl_types::I64, i as i64);
                    let grad = self.compile_call_by_name(
                        builder,
                        "nsl_list_get",
                        &[grads_list, idx_val],
                    )?;
                    let gnorm = self.compile_call_by_name(
                        builder,
                        "nsl_tensor_l2_norm",
                        &[grad],
                    )?;
                    self.compile_call_by_name(
                        builder,
                        "nsl_health_record_grad_norm",
                        &[path_ptr, path_len, layer_idx_val, gnorm],
                    )?;
                }
            } else {
                nsl_log::nsl_log!(INFO, "health", 
                    "[health] note: per-parameter gradient norms are not \
                     recorded under the FASE-Deferred hook (per-batch grads \
                     are consumed into m_partial during the backward) — the \
                     health snapshot's grad_norm fields will be absent. \
                     Loss and weight-norm recording are unaffected."
                );
            }

            // (c) Per-parameter weight norms — gated by step % 100 == 0
            //     (which includes step 0 as the initial weight snapshot).
            let wnorm_block = builder.create_block();
            let after_wnorm = builder.create_block();
            builder
                .ins()
                .brif(is_flush_due, wnorm_block, &[], after_wnorm, &[]);

            builder.switch_to_block(wnorm_block);
            builder.seal_block(wnorm_block);
            state.current_block = Some(wnorm_block);

            for (i, path) in param_paths.iter().enumerate() {
                let path_data_id = self.intern_string(path)?;
                let gv = self.module.declare_data_in_func(path_data_id, builder.func);
                let path_ptr = builder.ins().symbol_value(cl_types::I64, gv);
                let path_len = builder
                    .ins()
                    .iconst(cl_types::I64, path.len() as i64);
                let idx_val = builder.ins().iconst(cl_types::I64, i as i64);
                let param = self.compile_call_by_name(
                    builder,
                    "nsl_list_get",
                    &[param_list, idx_val],
                )?;
                let wnorm = self.compile_call_by_name(
                    builder,
                    "nsl_tensor_l2_norm",
                    &[param],
                )?;
                // is_init flag: 1 iff step == 0. icmp already yields I8 in
                // current Cranelift (the old b1 type is gone) — a further
                // uextend to I8 is a same-width extend and fails verification,
                // which broke every `nsl run --monitor` on a train program.
                let step_cmp = builder.use_var(step_count_var);
                let zero_cmp = builder.ins().iconst(cl_types::I64, 0);
                let is_init_i8 = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    step_cmp,
                    zero_cmp,
                );
                self.compile_call_by_name(
                    builder,
                    "nsl_health_record_weight_norm",
                    &[path_ptr, path_len, wnorm, is_init_i8],
                )?;
            }

            // (d) Snapshot flush (reuses is_flush_due — we're still in wnorm_block).
            let snap_path = self
                .compile_options
                .dev_tools.profile_source_file_name
                .as_ref()
                .map(|p| format!("{}.nsl-health.json", p))
                .unwrap_or_else(|| "nsl-health.json".to_string());
            let snap_data_id = self.intern_string(&snap_path)?;
            let snap_gv = self.module.declare_data_in_func(snap_data_id, builder.func);
            let snap_ptr = builder.ins().symbol_value(cl_types::I64, snap_gv);
            let snap_len = builder
                .ins()
                .iconst(cl_types::I64, snap_path.len() as i64);
            self.compile_call_by_name(
                builder,
                "nsl_health_flush_snapshot",
                &[snap_ptr, snap_len],
            )?;

            builder.ins().jump(after_wnorm, &[]);
            builder.switch_to_block(after_wnorm);
            builder.seal_block(after_wnorm);
            state.current_block = Some(after_wnorm);
        }

        Ok(())
    }
}
