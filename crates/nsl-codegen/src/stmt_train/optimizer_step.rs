//! Sections 7e4–7g of the train block: the optimizer step. The
//! gradient-accumulation gate (`should_step`), the three step arms — the
//! WGGO mode-table dispatch, the FASE-deferred fused step, and the stdlib
//! per-parameter loop (with Muon's batched Newton-Schulz call ahead of it)
//! — the ZeRO gradient reduce before and the parameter sync after, and the
//! post-optimizer cleanup that zeroes the accumulation buffers or frees the
//! direct gradients.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1):
//! 986 lines, thirty-two inputs ([`OptimizerStepInputs`]), no escaping
//! binding. The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`)
//! pin the emitted step on every fixture (the accumulation gate on the
//! `*_accum` variants, the FASE arm on the `*_fase` ones, the stdlib loop
//! everywhere else); the refusal texts are found by the CLI composition
//! gate's wholesale sweep of `crates/nsl-codegen/src`.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::param_roles::NoDecayScope;

/// Every binding of `compile_train_block_inner` the optimizer step reads;
/// names are the driver's.
pub(crate) struct OptimizerStepInputs {
    /// The gradient-accumulation buffer list (`None` = step on the direct grads).
    pub(crate) accum_list: Option<Value>,
    /// Muon's AdamW-routed learning rate, as a ratio of `lr_value` at the step.
    pub(crate) adamw_lr_value: Option<f64>,
    /// The base learning rate the ratio above divides by.
    pub(crate) lr_value: f64,
    pub(crate) beta1_value: f64,
    pub(crate) beta2_value: f64,
    pub(crate) dampening_value: f64,
    pub(crate) eps_value: f64,
    pub(crate) momentum_value: f64,
    pub(crate) weight_decay_value: f64,
    /// Muon's Newton-Schulz iteration count.
    pub(crate) ns_steps_value: f64,
    pub(crate) nesterov_value: bool,
    pub(crate) grad_clip: f64,
    /// The CPDT per-parameter dtype-code lists (m, v).
    pub(crate) cpdt_precision_dtypes: Option<(Value, Value)>,
    pub(crate) csla_active: bool,
    pub(crate) fase_deferred: bool,
    pub(crate) fase_hook_active: bool,
    /// AdamW parameter groups: the per-parameter decay-exempt flags.
    pub(crate) decay_exempt_list: Option<Value>,
    /// The FASE plan (consumed here; nothing after the step reads it).
    pub(crate) fase_plan: crate::fase::FasePlan,
    pub(crate) grad_accumulation_steps: i64,
    /// The direct gradient list (a null sentinel when the FASE hook consumed them).
    pub(crate) grads_list: Value,
    pub(crate) lr_var: Variable,
    /// Computed once before the accumulation loop; read at the gate.
    pub(crate) should_step_var: Variable,
    pub(crate) step_count_var: Variable,
    /// WGGO's per-parameter mode table (the unified dispatch arm).
    pub(crate) mode_table_base: Option<Value>,
    /// Muon's per-parameter route flags (`--muon-batch-ns`).
    pub(crate) muon_route_list: Option<Value>,
    pub(crate) no_decay_scope: NoDecayScope,
    pub(crate) num_params_val: Value,
    pub(crate) param_list: Value,
    pub(crate) state_list_1: Value,
    pub(crate) state_list_2: Value,
    pub(crate) num_state_buffers: usize,
    pub(crate) optimizer_name: String,
}

impl Compiler<'_> {
    /// Emit the optimizer step (see the module header).
    pub(crate) fn emit_optimizer_step(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: OptimizerStepInputs,
    ) -> Result<(), CodegenError> {
        let OptimizerStepInputs {
            accum_list,
            adamw_lr_value,
            lr_value,
            beta1_value,
            beta2_value,
            dampening_value,
            eps_value,
            momentum_value,
            weight_decay_value,
            ns_steps_value,
            nesterov_value,
            grad_clip,
            cpdt_precision_dtypes,
            csla_active,
            fase_deferred,
            fase_hook_active,
            decay_exempt_list,
            fase_plan,
            grad_accumulation_steps,
            grads_list,
            lr_var,
            should_step_var,
            step_count_var,
            mode_table_base,
            muon_route_list,
            no_decay_scope,
            num_params_val,
            param_list,
            state_list_1,
            state_list_2,
            num_state_buffers,
            optimizer_name,
        } = inputs;

        // 7e4. Gradient accumulation gate: only step optimizer every N batches
        let optimizer_block = builder.create_block();
        let post_optimizer_block = builder.create_block();

        if grad_accumulation_steps > 1 {
            // Reuse the already-computed should_step value (defined above).
            let should_step = builder.use_var(should_step_var);
            builder
                .ins()
                .brif(should_step, optimizer_block, &[], post_optimizer_block, &[]);

            builder.switch_to_block(optimizer_block);
            builder.seal_block(optimizer_block);
            state.current_block = Some(optimizer_block);
        } else {
            // No accumulation — always step
            builder.ins().jump(optimizer_block, &[]);
            builder.switch_to_block(optimizer_block);
            builder.seal_block(optimizer_block);
            state.current_block = Some(optimizer_block);
        }

        // NSL_PHASE_TIMING: optimizer-phase start (fires only on accumulation
        // boundaries — this block is skipped on non-step micro-batches). The
        // matching report is emitted before each of the three
        // post_optimizer_block jumps below.
        let phase_timing_opt =
            std::env::var("NSL_PHASE_TIMING").ok().as_deref() == Some("1");
        let phase_opt_t0 = if phase_timing_opt {
            self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
            Some(self.compile_call_by_name(builder, "nsl_clock", &[])?)
        } else {
            None
        };

        // 7f. Optimizer step: for each param, call optimizer step function
        // FASE Deferred: emit fused per-parameter step; otherwise use existing path.
        //
        // FASE Codegen Phase 2: Optimizer step dispatch is at outer scope here
        // (the `if fase_deferred { Deferred loops } else { per-optimizer stdlib
        // step }` shape can't be split per-param without hoisting a runtime
        // branch through two structurally different optimizer-step emission
        // paths — the Deferred side uses `fase_emit_final_step` on m_partial,
        // the FullBuffer side dispatches to optimizer-specific stdlib functions
        // (adam_step/sgd_step/...) with different calling conventions.
        //
        // Deferring per-param optimizer dispatch to a follow-up plan. WGGO's
        // per-layer signal still influences accumulation behavior via the
        // mode-table dispatch in ga_body (Task 4b), which is the dominant
        // memory-cost decision. The optimizer step's compute cost is identical
        // between modes — only accumulation buffer shape differs.
        //
        // FASE Optim-Step Dispatch Task 5: shared locals hoisted from the
        // fallback `else` arm so the new `if let Some(mtb)` branch can
        // reference them. Cranelift DCE ensures constants unused by the
        // Deferred arm don't affect emitted code for that path.
        let opt_grads = if let Some(accum) = accum_list {
            accum
        } else {
            grads_list
        };

        // H.2: mangling convention is `{module_prefix}__{fn_name}` where
        // `module_prefix` is the dotted stdlib path with `.` -> `_`
        // (single underscore per path separator). So `nsl.optim.sgd` ->
        // `nsl_optim_sgd`, and the step fn becomes
        // `nsl_optim_sgd__sgd_step`. Pre-H.2 this site emitted
        // `nsl__optim__sgd__sgd_step` (double underscores between every
        // path part) which did not match `stdlib_loader::module_prefix_for`
        // output — `compile_entry` failed with "undefined function".
        let optimizer_fn_name = match optimizer_name.as_str() {
            "sgd" => "nsl_optim_sgd__sgd_step",
            "adam" => "nsl_optim_adam__adam_step",
            "adamw" => "nsl_optim_adamw__adamw_step",
            "lion" => "nsl_optim_lion__lion_step",
            "muon" => "nsl_optim_muon__muon_step",
            "soap" => "nsl_optim_soap__soap_step",
            _ => {
                return Err(CodegenError::new(format!(
                    "unsupported optimizer '{}' in train block",
                    optimizer_name
                )));
            }
        };

        // Check if optimizer function exists, try fallback name patterns
        let opt_fn = if self.registry.functions.contains_key(optimizer_fn_name) {
            optimizer_fn_name.to_string()
        } else {
            // Try simpler name: e.g. "sgd_step"
            let simple = format!("{}_step", optimizer_name);
            if self.registry.functions.contains_key(&simple) {
                simple
            } else if self.registry.runtime_fns.contains_key(optimizer_fn_name) {
                optimizer_fn_name.to_string()
            } else if self.registry.runtime_fns.contains_key(&simple) {
                simple
            } else {
                // Register as runtime function so it can be resolved at link time
                optimizer_fn_name.to_string()
            }
        };

        let lr = builder.use_var(lr_var);
        let momentum_const = builder.ins().f64const(momentum_value);
        let dampening_const = builder.ins().f64const(dampening_value);
        let weight_decay_const = builder.ins().f64const(weight_decay_value);
        let nesterov_const = builder
            .ins()
            .iconst(cl_types::I8, if nesterov_value { 1 } else { 0 });
        let beta1_const = builder.ins().f64const(beta1_value);
        let beta2_const = builder.ins().f64const(beta2_value);
        let eps_const = builder.ins().f64const(eps_value);

        // M43b: ZeRO Stage 1+ — all-reduce gradients before optimizer step.
        // The rc is asserted: a mid-run collective failure (capacity -4,
        // GPU-placement refusal -5, dtype -1) must abort, not continue with
        // un-reduced gradients (review: discarded rc -> silent wrong train).
        // P3 ZeRO-3: stage 3 reduces PER LAYER GROUP inside the (csla)
        // window backward — the monolithic all-param reduce here would run
        // on the window's already-nulled accum slots, so it is stages 1/2
        // only.
        let zero_enabled = self.features.zero_stage.filter(|&s| s >= 1).is_some();
        let zero_monolithic =
            self.features.zero_stage.filter(|&s| (1..=2).contains(&s)).is_some();
        if zero_monolithic {
            let rc = self.compile_call_by_name(
                builder,
                "nsl_zero_reduce_grads",
                &[opt_grads, num_params_val],
            )?;
            let z = builder.ins().iconst(cl_types::I64, 0);
            let ok = builder.ins().icmp(IntCC::Equal, rc, z);
            let m = "nsl: ZeRO gradient reduction failed (see message above) — aborting";
            self.intern_string(m)?;
            let mp = self.compile_string_literal(builder, m)?;
            self.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
        }

        if let Some(mtb) = mode_table_base {
            // D3 v1: the unified mode-table dispatch has its own per-param
            // update kernel selection — owner-gating inside it is untested
            // territory. Refuse the composition loudly.
            if zero_enabled {
                return Err(CodegenError::new(
                    "--zero-stage is not supported with a WGGO per-param mode \
                     table yet: the sharded update gate is lowered only for \
                     the monolithic Deferred and FullBuffer optimizer arms. \
                     Drop --wggo mode overrides or --zero-stage",
                ));
            }
            self.emit_unified_optim_step_dispatch(
                builder,
                state,
                mtb,
                num_params_val,
                param_list,
                state_list_1,
                state_list_2,
                num_state_buffers,
                accum_list,
                opt_grads,
                fase_hook_active,
                step_count_var,
                &fase_plan,
                optimizer_name.as_str(),
                &opt_fn,
                lr,
                momentum_const,
                dampening_const,
                weight_decay_const,
                nesterov_const,
                beta1_const,
                beta2_const,
                eps_const,
                grad_accumulation_steps,
                grad_clip,
                cpdt_precision_dtypes,
                decay_exempt_list.map(|l| (l, no_decay_scope.exempt_non_rank2)),
            )?;

            // M43b: ZeRO Stage 1+ — all-gather updated params after optimizer step
            if zero_enabled {
                self.compile_call_by_name(builder, "nsl_zero_step", &[])?;
            }

            // Jump to post-optimizer block (merges optimizer and skip paths)
            if let Some(t0) = phase_opt_t0 {
                self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                let opt = builder.ins().fsub(t3, t0);
                self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
            }
            builder.ins().jump(post_optimizer_block, &[]);
            builder.switch_to_block(post_optimizer_block);
            builder.seal_block(post_optimizer_block);
            state.current_block = Some(post_optimizer_block);
        } else if fase_deferred {
            // CSLA (D1b): every parameter was already updated inside the
            // window backward region (per-layer groups + the epilogue group,
            // with the same bias-correction pair this site would compute) —
            // the monolithic loop below must not run on the NULL accumulator
            // slots. Emit only the pass-through to post_optimizer_block.
            if csla_active {
                if let Some(t0) = phase_opt_t0 {
                    self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                    let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                    let opt = builder.ins().fsub(t3, t0);
                    self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
                }
                builder.ins().jump(post_optimizer_block, &[]);
                builder.switch_to_block(post_optimizer_block);
                builder.seal_block(post_optimizer_block);
                state.current_block = Some(post_optimizer_block);
            } else if let Some(accum) = accum_list {
                // ── FASE Deferred: compute bias-correction scalars once per step ──
                // opt_step = (step_count + 1) / grad_accumulation_steps
                // bc_inv = nsl_bias_correction_inv(β, opt_step)
                let sc_val = builder.use_var(step_count_var);
                let one_i64 = builder.ins().iconst(cl_types::I64, 1);
                let sc_plus_one = builder.ins().iadd(sc_val, one_i64);
                let grad_accum_const = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
                let opt_step = builder.ins().sdiv(sc_plus_one, grad_accum_const);

                let beta1_const = builder.ins().f64const(fase_plan.recipe.beta1);
                let beta2_const = builder.ins().f64const(fase_plan.recipe.beta2);
                let bc1_inv = self.compile_call_by_name(
                    builder,
                    "nsl_bias_correction_inv",
                    &[beta1_const, opt_step],
                )?;
                let bc2_inv = self.compile_call_by_name(
                    builder,
                    "nsl_bias_correction_inv",
                    &[beta2_const, opt_step],
                )?;

                // Fusion item 1 admission, shared by BOTH arms below: when
                // every param would take the fused AdamW path (exact
                // structural match, no envelopes, no ZeRO gating, no
                // offload staging), the per-param final-step loop collapses
                // into ONE multi-tensor pointer-table launch. The clip arm
                // additionally folds its Phase B pre-scale into the same
                // launch via the kernel's mp_scale argument (bit-identical
                // to scale-then-step; see FASE_FUSED_ADAMW_MULTI_F32_PTX).
                // Kill-switches NSL_FASE_FUSED_STEP=0 / NSL_FASE_MULTI_STEP=0
                // (compile-time).
                let multi_scalars = if cpdt_precision_dtypes.is_none()
                    && !self.compile_options.diagnostics.training_reference
                    && std::env::var("NSL_FASE_FUSED_STEP").ok().as_deref() != Some("0")
                    && std::env::var("NSL_FASE_MULTI_STEP").ok().as_deref() != Some("0")
                    // ZeRO 1/2 no longer excludes: the launch below goes
                    // through the OWNER SUBSET (see
                    // `emit_fused_multi_launch`). It used to, which meant
                    // every `--zero-stage` run silently reverted the item-8
                    // batching to the per-param loop with no diagnostic
                    // anywhere — a shipped optimization that was 100% off in
                    // a supported configuration.
                    && !self.compile_options.optim_state_offload
                    // Belt only: bf16-sr structurally cannot reach this
                    // FullBuffer path (it requires --weight-stream, which
                    // clap-requires --layerwise-accum, so SR always takes the
                    // CSLA schedule and its SR multi_idx arm — item 8). Kept
                    // so a future envelope change fails safe into the
                    // per-param loop instead of silently mis-batching.
                    && !self.features.param_dtype_bf16sr
                    && num_state_buffers >= 2
                {
                    Self::match_adamw_program(&crate::fase_optimizer::emit_final_step(
                        &fase_plan.recipe,
                    ))
                } else {
                    None
                };

                if fase_plan.two_phase_clip {
                    // Phase A batching: ONE `nsl_fase_sum_sq_list` call (a
                    // single pipeline drain) instead of one synchronously
                    // read-back `nsl_tensor_sum_sq` per parameter — 74
                    // drains/step at Coder-50M. The f64-vs-f32 accumulation
                    // note lives on the FFI. Kill-switch
                    // NSL_FASE_BATCH_SUMSQ=0 (compile-time, like
                    // NSL_FASE_FUSED_STEP).
                    let batch_sumsq =
                        std::env::var("NSL_FASE_BATCH_SUMSQ").ok().as_deref() != Some("0");
                    // ── Phase A: fused accumulation + sum_sq loop ──
                    // Accumulate the final micro-batch's gradients into m_partial
                    // (the standard accumulation loop was skipped for this batch),
                    // and simultaneously accumulate sum(||g_i||^2) for the global
                    // L2 norm. When the hook already accumulated during adjoint
                    // lowering AND the norm is batched, there is nothing left
                    // for the loop to do — skip emitting it entirely.
                    let pa_tot_var = builder.declare_var(cl_types::F64);
                    let pa_zero_f = builder.ins().f64const(0.0);
                    builder.def_var(pa_tot_var, pa_zero_f);

                    if !(fase_hook_active && batch_sumsq) {
                    let pa_i_var = builder.declare_var(cl_types::I64);
                    let pa_i_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(pa_i_var, pa_i_zero);

                    let pa_hdr = builder.create_block();
                    let pa_body = builder.create_block();
                    let pa_exit = builder.create_block();
                    builder.ins().jump(pa_hdr, &[]);
                    builder.switch_to_block(pa_hdr);
                    let pa_i = builder.use_var(pa_i_var);
                    let pa_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, pa_i, num_params_val);
                    builder.ins().brif(pa_cont, pa_body, &[], pa_exit, &[]);
                    builder.switch_to_block(pa_body);
                    builder.seal_block(pa_body);

                    let pa_mpart =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, pa_i])?;
                    // Monolithic Phase A: this branch only runs when NO mode
                    // table exists (no WGGO overrides), so every param is
                    // globally Deferred — no mode dispatch needed. Mixed
                    // tables take the emit_unified_optim_step_dispatch path,
                    // whose Phase A mirrors this fused accumulate.
                    //
                    // When FASE hook is active, accumulation already happened
                    // during adjoint lowering — skip the grads_list read + accumulate.
                    if !fase_hook_active {
                        let pa_grad = self.compile_call_by_name(
                            builder,
                            "nsl_list_get",
                            &[grads_list, pa_i],
                        )?;
                        let off = self.compile_options.optim_state_offload;
                        self.fase_emit_accumulate(
                            builder,
                            pa_mpart,
                            pa_grad,
                            fase_plan.recipe.accum_scale,
                            off,
                        )?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[pa_grad])?;
                    }
                    if !batch_sumsq {
                        let pa_sq = self.compile_call_by_name(
                            builder,
                            "nsl_tensor_sum_sq",
                            &[pa_mpart],
                        )?;
                        let pa_tot_cur = builder.use_var(pa_tot_var);
                        let pa_tot_new = builder.ins().fadd(pa_tot_cur, pa_sq);
                        builder.def_var(pa_tot_var, pa_tot_new);
                    }
                    let pa_i_next = builder.ins().iadd_imm_s(pa_i, 1);
                    builder.def_var(pa_i_var, pa_i_next);
                    builder.ins().jump(pa_hdr, &[]);

                    builder.switch_to_block(pa_exit);
                    builder.seal_block(pa_hdr);
                    builder.seal_block(pa_exit);
                    }

                    // Free grads_list wrapper — skip when hook active (null sentinel).
                    if !fase_hook_active {
                        self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
                    }

                    // ── Scalar: clip_factor = min(1, grad_clip / (sqrt(total_sq) + 1e-6)) ──
                    // `grad_clip` is the raw f64 from the train-block config
                    // (f64::MAX when not set, but two_phase_clip is only true when it IS set).
                    let grad_clip_threshold = grad_clip;
                    let total_sq = if batch_sumsq {
                        self.compile_call_by_name(builder, "nsl_fase_sum_sq_list", &[accum])?
                    } else {
                        builder.use_var(pa_tot_var)
                    };
                    let norm = builder.ins().sqrt(total_sq);
                    let eps_v = builder.ins().f64const(1e-6_f64);
                    let denom = builder.ins().fadd(norm, eps_v);
                    let tau_v = builder.ins().f64const(grad_clip_threshold);
                    let ratio = builder.ins().fdiv(tau_v, denom);
                    let one_f = builder.ins().f64const(1.0_f64);
                    let clip_factor = builder.ins().fmin(one_f, ratio);

                    // ── Phase B, multi arm: ONE pointer-table launch with the
                    // clip factor folded into the m_partial read. Replaces
                    // 74 nsl_tensor_mul_scalar_inplace + 74 fused-step
                    // launches (+ the per-param list traffic) at Coder-50M.
                    if let Some(sc) = multi_scalars {
                        // The parameter-group arguments are as load-bearing here
                        // as in the unclipped arm below. Omitting them does not
                        // fail to compile — it shifts `clip_factor` into the
                        // `wd_exempt_list` slot, so the runtime would read a
                        // float bit-pattern as an NslList pointer while the clip
                        // silently stopped being applied. (Passing them through
                        // one shared emitter is now what keeps that true.)
                        let exempt_list_v = decay_exempt_list
                            .unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 0));
                        let exempt_nr2_v = builder.ins().iconst(
                            cl_types::I64,
                            i64::from(no_decay_scope.exempt_non_rank2),
                        );
                        self.emit_fused_multi_launch(
                            builder,
                            zero_monolithic,
                            num_params_val,
                            (param_list, state_list_1, state_list_2, accum),
                            &sc,
                            lr,
                            (bc1_inv, bc2_inv),
                            (exempt_list_v, exempt_nr2_v),
                            // mp_scale: the clip factor, folded into the
                            // m_partial read in-kernel.
                            clip_factor,
                        )?;
                    } else {
                    // ── Phase B: scale m_partial in place, then fused optimizer step ──
                    let pb_i_var = builder.declare_var(cl_types::I64);
                    let pb_i_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(pb_i_var, pb_i_zero);

                    let pb_hdr = builder.create_block();
                    let pb_body = builder.create_block();
                    let pb_exit = builder.create_block();
                    builder.ins().jump(pb_hdr, &[]);
                    builder.switch_to_block(pb_hdr);
                    let pb_i = builder.use_var(pb_i_var);
                    let pb_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, pb_i, num_params_val);
                    builder.ins().brif(pb_cont, pb_body, &[], pb_exit, &[]);
                    builder.switch_to_block(pb_body);
                    builder.seal_block(pb_body);

                    let pb_mpart =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, pb_i])?;
                    // D3 (ZeRO-1): only the owner rank updates this param.
                    // Non-owners must still ZERO m_partial (the update
                    // normally does it) or the next window accumulates onto
                    // stale gradients.
                    let pb_zero_blocks = if zero_enabled {
                        let owns = self
                            .compile_call_by_name(builder, "nsl_zero_owns_param", &[pb_i])?;
                        let one_i = builder.ins().iconst(cl_types::I64, 1);
                        let owned = builder.ins().icmp(IntCC::Equal, owns, one_i);
                        let pb_do = builder.create_block();
                        let pb_skip = builder.create_block();
                        let pb_join = builder.create_block();
                        builder.ins().brif(owned, pb_do, &[], pb_skip, &[]);
                        builder.switch_to_block(pb_skip);
                        builder.seal_block(pb_skip);
                        // L8 hygiene: a true zero-fill, NOT mul-by-0.0 — if
                        // a diverging window left Inf/NaN in m_partial,
                        // x*0.0 is NaN and the buffer never recovers.
                        self.compile_call_by_name(
                            builder,
                            "nsl_tensor_zero_inplace",
                            &[pb_mpart],
                        )?;
                        builder.ins().jump(pb_join, &[]);
                        builder.switch_to_block(pb_do);
                        builder.seal_block(pb_do);
                        Some(pb_join)
                    } else {
                        None
                    };
                    self.compile_call_by_name(
                        builder,
                        "nsl_tensor_mul_scalar_inplace",
                        &[pb_mpart, clip_factor],
                    )?;
                    let pb_theta = self.compile_call_by_name(
                        builder,
                        "nsl_list_get",
                        &[param_list, pb_i],
                    )?;
                    let pb_m = self.compile_call_by_name(
                        builder,
                        "nsl_list_get",
                        &[state_list_1, pb_i],
                    )?;
                    let pb_v = if num_state_buffers >= 2 {
                        self.compile_call_by_name(
                            builder,
                            "nsl_list_get",
                            &[state_list_2, pb_i],
                        )?
                    } else {
                        pb_m
                    };
                    let wrap_precision = cpdt_precision_dtypes.is_some();
                    self.fase_emit_final_step(
                        builder,
                        pb_theta,
                        pb_m,
                        pb_mpart,
                        pb_v,
                        &fase_plan.recipe,
                        lr,
                        Some((bc1_inv, bc2_inv)),
                        wrap_precision,
                        self.compile_options.optim_state_offload,
                        Some(opt_step),
                    )?;
                    if let Some(pb_join) = pb_zero_blocks {
                        builder.ins().jump(pb_join, &[]);
                        builder.switch_to_block(pb_join);
                        builder.seal_block(pb_join);
                    }
                    let pb_i_next = builder.ins().iadd_imm_s(pb_i, 1);
                    builder.def_var(pb_i_var, pb_i_next);
                    builder.ins().jump(pb_hdr, &[]);

                    builder.switch_to_block(pb_exit);
                    builder.seal_block(pb_hdr);
                    builder.seal_block(pb_exit);
                    state.current_block = Some(pb_exit);
                    }
                    // Offload P0.2: one drain per optimizer step (transfer-
                    // stream sync + deferred frees of the staged tensors).
                    // (Unreachable from the multi arm: its admission requires
                    // optim_state_offload == false.)
                    if self.compile_options.optim_state_offload {
                        self.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
                    }
                } else {
                    // ── Non-clip Deferred path: per-parameter fused final step ──
                    // accum_list is m_partial.  state_list_1 = m, state_list_2 = v.
                    //
                    // Fusion item 1: when EVERY param would take the fused
                    // AdamW path, the whole loop collapses into ONE
                    // multi-tensor pointer-table launch — bit-identical per
                    // element (the multi kernel is the single kernel's body
                    // with table addressing, and it folds the shared tail's
                    // m_partial zero). Admission (`multi_scalars`) is hoisted
                    // above the clip fork and shared with the clip arm.
                    if let Some(sc) = multi_scalars {
                        // The recipe constants are materialized inside
                        // `emit_fused_multi_launch` now — emitting them here
                        // too would leave four dead Cranelift values and a
                        // `-D warnings` clippy failure.
                        let exempt_list_v = decay_exempt_list
                            .unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 0));
                        let exempt_nr2_v = builder.ins().iconst(
                            cl_types::I64,
                            i64::from(no_decay_scope.exempt_non_rank2),
                        );
                        // mp_scale = 1.0: unclipped path (the kernel branches
                        // around the multiply, keeping exact bit-identity).
                        let one_scale = builder.ins().f64const(1.0);
                        self.emit_fused_multi_launch(
                            builder,
                            zero_monolithic,
                            num_params_val,
                            (param_list, state_list_1, state_list_2, accum),
                            &sc,
                            lr,
                            (bc1_inv, bc2_inv),
                            // AdamW parameter groups. 0 = no no_decay, in
                            // which case the runtime takes `wd` for every
                            // param exactly as before.
                            (exempt_list_v, exempt_nr2_v),
                            one_scale,
                        )?;
                    } else {
                    let fs_i_var = builder.declare_var(cl_types::I64);
                    let fs_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(fs_i_var, fs_zero);
                    let fs_hdr = builder.create_block();
                    let fs_body = builder.create_block();
                    let fs_exit = builder.create_block();
                    builder.ins().jump(fs_hdr, &[]);
                    builder.switch_to_block(fs_hdr);
                    let fs_i = builder.use_var(fs_i_var);
                    let fs_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, fs_i, num_params_val);
                    builder.ins().brif(fs_cont, fs_body, &[], fs_exit, &[]);
                    builder.switch_to_block(fs_body);
                    builder.seal_block(fs_body);
                    // D3 (ZeRO-1): owner-gated update; non-owners zero
                    // m_partial (see the clip-path comment).
                    let fs_zero_blocks = if zero_enabled {
                        let mp = self
                            .compile_call_by_name(builder, "nsl_list_get", &[accum, fs_i])?;
                        let owns = self
                            .compile_call_by_name(builder, "nsl_zero_owns_param", &[fs_i])?;
                        let one_i = builder.ins().iconst(cl_types::I64, 1);
                        let owned = builder.ins().icmp(IntCC::Equal, owns, one_i);
                        let fs_do = builder.create_block();
                        let fs_skip = builder.create_block();
                        let fs_join = builder.create_block();
                        builder.ins().brif(owned, fs_do, &[], fs_skip, &[]);
                        builder.switch_to_block(fs_skip);
                        builder.seal_block(fs_skip);
                        // L8 hygiene: true zero-fill (see the clip-path
                        // comment — mul-by-0.0 keeps NaN/Inf alive).
                        self.compile_call_by_name(
                            builder,
                            "nsl_tensor_zero_inplace",
                            &[mp],
                        )?;
                        builder.ins().jump(fs_join, &[]);
                        builder.switch_to_block(fs_do);
                        builder.seal_block(fs_do);
                        Some(fs_join)
                    } else {
                        None
                    };
                    let theta =
                        self.compile_call_by_name(builder, "nsl_list_get", &[param_list, fs_i])?;
                    let m =
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, fs_i])?;
                    let m_partial =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, fs_i])?;
                    let v = if num_state_buffers >= 2 {
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, fs_i])?
                    } else {
                        // SGD has no v state — pass m as a placeholder (not used by SgdUpdate recipe)
                        m
                    };
                    let wrap_precision = cpdt_precision_dtypes.is_some();
                    // AdamW parameter groups. `fase_emit_final_step` bakes λ
                    // into the emitted update program at COMPILE time (it
                    // elides the `wd·θ` term entirely when λ == 0), so a
                    // runtime λ cannot be handed to it. Branch between two
                    // compile-time recipes instead: each arm is then
                    // byte-identical to what this site emitted before the
                    // feature, at its own λ. Multiplying θ by a runtime 0
                    // would have been the smaller diff and is NOT used —
                    // this file already documents why mul-by-0.0 is not a
                    // zeroing idiom here (it keeps NaN/Inf alive).
                    let fs_wd_join = self.emit_deferred_step_with_groups(
                        builder,
                        decay_exempt_list.map(|l| (l, no_decay_scope.exempt_non_rank2)),
                        fs_i,
                        theta,
                        m,
                        m_partial,
                        v,
                        &fase_plan.recipe,
                        lr,
                        (bc1_inv, bc2_inv),
                        wrap_precision,
                        Some(opt_step),
                    )?;
                    if let Some(join) = fs_wd_join {
                        builder.ins().jump(join, &[]);
                        builder.switch_to_block(join);
                        builder.seal_block(join);
                    }
                    // fase_emit_final_step zeroed m_partial already — no Site E needed.
                    if let Some(fs_join) = fs_zero_blocks {
                        builder.ins().jump(fs_join, &[]);
                        builder.switch_to_block(fs_join);
                        builder.seal_block(fs_join);
                    }
                    let fs_one = builder.ins().iconst(cl_types::I64, 1);
                    let fs_next = builder.ins().iadd(fs_i, fs_one);
                    builder.def_var(fs_i_var, fs_next);
                    builder.ins().jump(fs_hdr, &[]);
                    builder.seal_block(fs_hdr);
                    builder.switch_to_block(fs_exit);
                    builder.seal_block(fs_exit);
                    state.current_block = Some(fs_exit);
                    }
                    // Offload P0.2: one drain per optimizer step (transfer-
                    // stream sync + deferred frees of the staged tensors).
                    // (Structurally unreachable on the multi path — offload
                    // is excluded from its admission — kept outside the
                    // else for byte-stability of the legacy arm.)
                    if self.compile_options.optim_state_offload {
                        self.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
                    }
                }

                // D3 (ZeRO-1): broadcast every param from its owner so
                // all ranks hold the full updated model — the post-step
                // sync both Deferred sub-arms converge on. Fixes the
                // pre-existing gap where this arm emitted no post-step
                // ZeRO call at all.
                if zero_enabled {
                    let rc = self.compile_call_by_name(
                        builder,
                        "nsl_zero_sync_params",
                        &[param_list, num_params_val],
                    )?;
                    let z = builder.ins().iconst(cl_types::I64, 0);
                    let ok = builder.ins().icmp(IntCC::Equal, rc, z);
                    let m = "nsl: ZeRO param sync failed (see message above) — aborting";
                    self.intern_string(m)?;
                    let mp = self.compile_string_literal(builder, m)?;
                    self.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
                }
                // Jump to post-optimizer block (merges optimizer and skip paths)
                if let Some(t0) = phase_opt_t0 {
                self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                let opt = builder.ins().fsub(t3, t0);
                self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
            }
            builder.ins().jump(post_optimizer_block, &[]);
                builder.switch_to_block(post_optimizer_block);
                builder.seal_block(post_optimizer_block);
                state.current_block = Some(post_optimizer_block);
            }
        } else {

        // 7f. Optimizer step loop: for i in 0..num_params (runtime loop)
        {
            // Muon perf campaign (`--muon-batch-ns`): ONE batched call
            // handles every Muon-routed rank-2 param (momentum update +
            // shape-grouped Newton-Schulz + parameter update, all on
            // device); the per-param loop below then SKIPS those params and
            // keeps the stdlib call — bit-for-bit — for the AdamW-routed
            // and non-rank-2 remainder. Combos that would break this split
            // (CSLA, offload, ZeRO, bf16 momentum, non-muon) were refused
            // at parse time above.
            let muon_batch_active =
                self.compile_options.muon.batch_ns && optimizer_name == "muon";
            if muon_batch_active {
                let route_list = muon_route_list.ok_or_else(|| {
                    CodegenError::new(
                        "--muon-batch-ns: muon route list missing (internal)",
                    )
                })?;
                let mom_c = builder.ins().f64const(momentum_value);
                let wd_c = builder.ins().f64const(weight_decay_value);
                let nest_c = builder
                    .ins()
                    .iconst(cl_types::I64, i64::from(nesterov_value));
                let ns_c = builder.ins().f64const(ns_steps_value);
                self.compile_call_by_name(
                    builder,
                    "nsl_muon_step_batch",
                    &[
                        param_list,
                        opt_grads,
                        state_list_1,
                        route_list,
                        lr,
                        mom_c,
                        wd_c,
                        nest_c,
                        ns_c,
                    ],
                )?;
            }

            let opt_i_var = builder.declare_var(cl_types::I64);
            let opt_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(opt_i_var, opt_zero);

            let opt_header = builder.create_block();
            let opt_body = builder.create_block();
            let opt_exit = builder.create_block();

            builder.ins().jump(opt_header, &[]);
            builder.switch_to_block(opt_header);
            state.current_block = Some(opt_header);

            let idx = builder.use_var(opt_i_var);
            let opt_cond = builder
                .ins()
                .icmp(IntCC::SignedLessThan, idx, num_params_val);
            builder.ins().brif(opt_cond, opt_body, &[], opt_exit, &[]);

            builder.switch_to_block(opt_body);
            builder.seal_block(opt_body);
            state.current_block = Some(opt_body);

            // D3 (ZeRO-1): owner-gated update. FullBuffer accum cleanup
            // (7g below) zeroes every accum slot unconditionally, so the
            // skip path needs no manual m_partial zero here.
            let opt_zero_blocks = if zero_enabled {
                let owns =
                    self.compile_call_by_name(builder, "nsl_zero_owns_param", &[idx])?;
                let one_i = builder.ins().iconst(cl_types::I64, 1);
                let owned = builder.ins().icmp(IntCC::Equal, owns, one_i);
                let z_do = builder.create_block();
                let z_join = builder.create_block();
                builder.ins().brif(owned, z_do, &[], z_join, &[]);
                builder.switch_to_block(z_do);
                builder.seal_block(z_do);
                Some(z_join)
            } else {
                None
            };

            // Get param, gradient, state buffers via runtime list indexing
            let param_val =
                self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;
            let grad_val = self.compile_call_by_name(builder, "nsl_list_get", &[opt_grads, idx])?;
            let s1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, idx])?;

            let s2 = if num_state_buffers >= 2 {
                self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, idx])?
            } else {
                s1 // placeholder for non-Adam/SOAP optimizers (ignored by helper)
            };
            // P5 Muon: per-param routing flag (parameter-role classification,
            // see 4a / param_roles.rs) + Newton-Schulz depth + the AdamW-arm
            // lr as a fixed ratio of the (possibly scheduled) lr.
            let muon_extra = if let Some(route_list) = muon_route_list {
                let flag_i =
                    self.compile_call_by_name(builder, "nsl_list_get", &[route_list, idx])?;
                let flag_f = builder.ins().fcvt_from_sint(cl_types::F64, flag_i);
                let ns_steps_const = builder.ins().f64const(ns_steps_value);
                let ratio = adamw_lr_value.map(|a| a / lr_value).unwrap_or(1.0);
                let ratio_const = builder.ins().f64const(ratio);
                let adamw_lr = builder.ins().fmul(lr, ratio_const);
                Some((flag_f, ns_steps_const, adamw_lr))
            } else {
                None
            };
            // --muon-batch-ns: params the pre-loop batch call already
            // updated are skipped here. Eligibility must mirror
            // nsl_muon_step_batch's filter EXACTLY (route flag 0 AND
            // runtime rank 2) or a param would be double-stepped/dropped —
            // the shared predicate IS that mirror (see its doc for why the
            // runtime side must abort, never skip, past those two tests).
            let batch_skip_join = if muon_batch_active {
                let route_list = muon_route_list.expect("refused above when None");
                let both =
                    self.emit_muon_route_predicate(builder, route_list, idx, param_val)?;
                let do_block = builder.create_block();
                let join = builder.create_block();
                builder.ins().brif(both, join, &[], do_block, &[]);
                builder.switch_to_block(do_block);
                builder.seal_block(do_block);
                state.current_block = Some(do_block);
                Some(join)
            } else {
                None
            };
            // FullBuffer-global path (no mode table, no WGGO). The CPDT
            // PrecisionPlan gate (`precision_active`'s 4th condition
            // requires `fase_deferred=true`) suppresses cpdt_precision_dtypes
            // construction here, so wrap_precision is always structurally
            // false at this call site. Threaded through for signature
            // uniformity with the unified-dispatch site.
            // AdamW parameter groups: per-param decay. Identity (no emitted
            // IR) when no_decay was not configured.
            let param_wd = self.emit_param_wd(
                builder,
                decay_exempt_list.map(|l| (l, no_decay_scope.exempt_non_rank2)),
                idx,
                param_val,
                weight_decay_const,
            )?;

            self.emit_stdlib_optim_call(
                builder,
                optimizer_name.as_str(),
                &opt_fn,
                param_val,
                grad_val,
                s1,
                s2,
                lr,
                momentum_const,
                dampening_const,
                param_wd,
                nesterov_const,
                beta1_const,
                beta2_const,
                eps_const,
                step_count_var,
                false,
                None,
                self.compile_options.optim_state_offload,
                muon_extra,
            )?;

            if let Some(join) = batch_skip_join {
                builder.ins().jump(join, &[]);
                builder.switch_to_block(join);
                builder.seal_block(join);
                state.current_block = Some(join);
            }

            if let Some(z_join) = opt_zero_blocks {
                builder.ins().jump(z_join, &[]);
                builder.switch_to_block(z_join);
                builder.seal_block(z_join);
            }
            let one_opt = builder.ins().iconst(cl_types::I64, 1);
            let next_opt = builder.ins().iadd(idx, one_opt);
            builder.def_var(opt_i_var, next_opt);
            builder.ins().jump(opt_header, &[]);
            builder.seal_block(opt_header);

            builder.switch_to_block(opt_exit);
            builder.seal_block(opt_exit);
            state.current_block = Some(opt_exit);

            // Offload P0.2: one drain per optimizer step (transfer-stream
            // sync + deferred frees of the staged tensors).
            if self.compile_options.optim_state_offload {
                self.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
            }
        }

        // D3 (ZeRO-1): broadcast every param from its owner — the real
        // post-step sync (nsl_zero_step retained as a no-op for ABI compat).
        if zero_enabled {
            self.compile_call_by_name(builder, "nsl_zero_step", &[])?;
            let rc = self.compile_call_by_name(
                builder,
                "nsl_zero_sync_params",
                &[param_list, num_params_val],
            )?;
            let z = builder.ins().iconst(cl_types::I64, 0);
            let ok = builder.ins().icmp(IntCC::Equal, rc, z);
            let m = "nsl: ZeRO param sync failed (see message above) — aborting";
            self.intern_string(m)?;
            let mp = self.compile_string_literal(builder, m)?;
            self.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
        }

        // 7g. Post-optimizer cleanup: zero accum buffers or free direct grads
        // Runtime loop over num_params_val
        if let Some(accum) = accum_list {
            let cleanup_i_var = builder.declare_var(cl_types::I64);
            let c_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(cleanup_i_var, c_zero);
            let c_header = builder.create_block();
            let c_body = builder.create_block();
            let c_exit = builder.create_block();
            builder.ins().jump(c_header, &[]);
            builder.switch_to_block(c_header);
            let ci = builder.use_var(cleanup_i_var);
            let cc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ci, num_params_val);
            builder.ins().brif(cc, c_body, &[], c_exit, &[]);
            builder.switch_to_block(c_body);
            builder.seal_block(c_body);
            let buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, ci])?;
            let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[buf])?;
            self.compile_call_by_name(builder, "nsl_grad_zero", &[buf, n_elems])?;
            let c_one = builder.ins().iconst(cl_types::I64, 1);
            let c_next = builder.ins().iadd(ci, c_one);
            builder.def_var(cleanup_i_var, c_next);
            builder.ins().jump(c_header, &[]);
            builder.seal_block(c_header);
            builder.switch_to_block(c_exit);
            builder.seal_block(c_exit);
            state.current_block = Some(c_exit);
        } else if !fase_hook_active {
            // No accumulation — free gradient tensors and grads_list every batch.
            // Skip when FASE hook is active: grads are already freed during
            // adjoint lowering, and grads_list is a null sentinel.
            let cleanup_i_var = builder.declare_var(cl_types::I64);
            let c_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(cleanup_i_var, c_zero);
            let c_header = builder.create_block();
            let c_body = builder.create_block();
            let c_exit = builder.create_block();
            builder.ins().jump(c_header, &[]);
            builder.switch_to_block(c_header);
            let ci = builder.use_var(cleanup_i_var);
            let cc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ci, num_params_val);
            builder.ins().brif(cc, c_body, &[], c_exit, &[]);
            builder.switch_to_block(c_body);
            builder.seal_block(c_body);
            let grad_val = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, ci])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[grad_val])?;
            let c_one = builder.ins().iconst(cl_types::I64, 1);
            let c_next = builder.ins().iadd(ci, c_one);
            builder.def_var(cleanup_i_var, c_next);
            builder.ins().jump(c_header, &[]);
            builder.seal_block(c_header);
            builder.switch_to_block(c_exit);
            builder.seal_block(c_exit);
            state.current_block = Some(c_exit);
            self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        }

        // Jump to post-optimizer block (merges optimizer and skip paths)
        if let Some(t0) = phase_opt_t0 {
                self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                let opt = builder.ins().fsub(t3, t0);
                self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
            }
            builder.ins().jump(post_optimizer_block, &[]);
        builder.switch_to_block(post_optimizer_block);
        builder.seal_block(post_optimizer_block);
        state.current_block = Some(post_optimizer_block);
        } // end else (non-FASE-Deferred optimizer path)

        Ok(())
    }
}
