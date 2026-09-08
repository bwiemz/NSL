//! The pipelined train block: the `pipeline(...)` sibling of
//! `driver::compile_train_block_inner`, moved out of `stmt.rs` byte-for-byte
//! (roadmap A1). `compile_train_block_pipelined` is the entry point the
//! train-block dispatch calls (it installs the fused-CE decorator config
//! around the lowering, like `compile_train_block` does);
//! `compile_train_block_pipelined_inner` is the lowering itself: the stage
//! loop over the model's layers with logical stage-to-stage communication
//! in a single process (model partitioning is deferred to M43c).

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;
use nsl_ast::block::TrainSection;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt::is_data_section_config_pair;

impl Compiler<'_> {
    /// Model partitioning (which layers run on which stage) is deferred to
    /// M43c; the initial implementation runs the full model in a single
    /// process with logical stage-to-stage communication.
    pub(crate) fn compile_train_block_pipelined(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
        // CFTP v10 (item 3): matches `compile_train_block`; installs the
        // fused-CE decorator config for THIS train block before the
        // pipelined lowering runs and restores it before returning.
        train_block_stmt_id: nsl_ast::NodeId,
    ) -> Result<(), CodegenError> {
        let saved_active_fused_ce =
            self.set_active_fused_ce_config_for_train_block(train_block_stmt_id);
        let result = self.compile_train_block_pipelined_inner(builder, state, train);
        self.restore_active_fused_ce_config(saved_active_fused_ce);
        result
    }

    /// CFTP v10 (item 3): pipelined-body analogue of
    /// [`compile_train_block_inner`] so the `active_fused_ce_config`
    /// prologue/epilogue can wrap the pipelined path uniformly.
    fn compile_train_block_pipelined_inner(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
    ) -> Result<(), CodegenError> {
        let saved_variables = state.variables.clone();
        let saved_variable_types = state.variable_types.clone();
        let saved_dataloader_symbols = state.dataloader_symbols.clone();
        let saved_borrowed_batch_symbols = state.borrowed_batch_symbols.clone();

        let config = self.features.pipeline_config.clone().unwrap();
        let num_stages = config.num_stages;

        // ── 1. Pipeline init ────────────────────────────────────────────
        let v_stages = builder.ins().iconst(cl_types::I64, num_stages as i64);
        let v_schedule = builder.ins().iconst(
            cl_types::I64,
            match config.schedule_type {
                crate::pipeline::ScheduleType::OneF1B => 0i64,
                crate::pipeline::ScheduleType::GPipe => 1i64,
            },
        );
        let v_micro = builder.ins().iconst(cl_types::I64, 8); // default micro-batches
        self.compile_call_by_name(
            builder,
            "nsl_pipeline_init",
            &[v_stages, v_schedule, v_micro],
        )?;

        // ── 2. Extract config from train(...) args ──────────────────────
        // Same Training Configuration Contract as the standard path: the
        // old scan here read ONLY `model=` and accepted everything else
        // unvalidated — under @pipeline a typo'd key (or a duplicate)
        // was doubly invisible. NOTE: epochs/grad_accumulation/grad_clip
        // are still accepted-but-inert on this lowering path (no epoch
        // loop exists; micro-batches are hardcoded below) — a known
        // contract gap tracked for the pipelined path's own change; the
        // resolver at least guarantees the keys are well-formed and the
        // namespace closed.
        let pipe_cfg = nsl_semantic::train_config::resolve_train_config(
            train,
            &|sym| self.resolve_sym(sym).to_string(),
            nsl_semantic::train_config::TrainConfigPurpose::UserTrainBlock,
        )
        .map_err(|diags| {
            let msgs: Vec<String> = diags.into_iter().map(|d| d.message).collect();
            CodegenError::new(format!("train config refused: {}", msgs.join("; ")))
        })?;
        let model_sym: Option<nsl_ast::Symbol> = pipe_cfg.model;

        // Item 8 note: checkpointing is not lowered on this path, and does
        // not need a refusal HERE — `compile_train_block` already refuses
        // checkpoint_save/load/every for every program that reaches this
        // dispatch (the Milestone B arm above the pipelined branch). A
        // second copy here would be dead code that reads like the only
        // guard. Pinned by `pipelined_train_path_refuses_checkpoint_config`
        // in crates/nsl-cli/tests/train_resume_dataloader_gate.rs.

        // Same optimizer/scheduler contract as the standard path — ONE
        // resolver. This also closes the pipelined path's own historical
        // gaps: its private kwarg copy lacked adamw_lr/ns_steps arms and
        // had no Muon spec-default backfill, so the same source trained
        // differently under @pipeline.
        let optim_cfg = nsl_semantic::optim_config::resolve_optim_config(
            &train.sections,
            train.span,
            &|sym| self.resolve_sym(sym).to_string(),
            nsl_semantic::train_config::TrainConfigPurpose::UserTrainBlock,
        )
        .map_err(|diags| {
            let msgs: Vec<String> = diags.into_iter().map(|d| d.message).collect();
            CodegenError::new(format!(
                "optimizer config refused: {}",
                msgs.join("; ")
            ))
        })?;

        // Deferral-must-refuse: the pipelined optimizer step passes only
        // the shared hyperparameters (no adamw_route/ns_steps/adamw_lr
        // slots, and its state allocation gives Muon one moment buffer
        // where muon_step needs two) — lowering Muon here would emit a
        // call that cannot match muon_step's signature.
        if optim_cfg.optimizer.kind == nsl_semantic::optim_config::OptimizerKind::Muon {
            return Err(CodegenError::new(
                "the Muon optimizer is not supported on the @pipeline train \
                 path yet: the per-stage optimizer step does not thread the \
                 route/ns_steps/adamw_lr arguments muon_step requires. Use \
                 AdamW here, or drop @pipeline",
            ));
        }

        let optimizer_name = optim_cfg.optimizer.kind.as_str().to_string();
        let lr_value: f64 = optim_cfg.optimizer.lr;
        let momentum_value: f64 = optim_cfg.optimizer.momentum;
        let dampening_value: f64 = optim_cfg.optimizer.dampening;
        let weight_decay_value: f64 = optim_cfg.optimizer.weight_decay;
        // AdamW parameter groups (`no_decay=[...]`). Non-empty is refused
        // below at the optimizer-step emitter, where the comment explains
        // why the role flags have no list to be parallel to.
        let no_decay_scope = crate::param_roles::NoDecayScope {
            static_roles: optim_cfg.optimizer.no_decay_static_roles.clone(),
            exempt_non_rank2: optim_cfg.optimizer.no_decay_exempt_non_rank2,
        };
        let nesterov_value: bool = optim_cfg.optimizer.nesterov;
        let beta1_value: f64 = optim_cfg.optimizer.beta1;
        let beta2_value: f64 = optim_cfg.optimizer.beta2;
        let eps_value: f64 = optim_cfg.optimizer.eps;
        let mut step_body: Option<(&nsl_ast::stmt::Block, nsl_ast::Symbol)> = None;

        for section in &train.sections {
            match section {
                TrainSection::Optimizer(_) => {
                    // Fully consumed by resolve_optim_config above.
                }
                TrainSection::Step { param, body } => {
                    step_body = Some((body, *param));
                }
                TrainSection::Data(stmts) => {
                    // See `compile_train_block::TrainSection::Data` for why
                    // the allowlisted config pairs are skipped.
                    for stmt in stmts {
                        if is_data_section_config_pair(stmt, self.interner) {
                            continue;
                        }
                        self.compile_stmt(builder, state, stmt)?;
                    }
                }
                // Same treatment as the standard train path: bare statements
                // run once pre-training; eval:/distribute: refuse loudly
                // instead of being silently dropped.
                TrainSection::Stmt(s) => {
                    self.compile_stmt(builder, state, s)?;
                }
                TrainSection::Eval { .. } => {
                    return Err(CodegenError::new(
                        "train block `eval:` sections are not yet executed; move \
                         evaluation logic into an `on_epoch` callback (which \
                         receives the epoch and loss) so it actually runs",
                    ));
                }
                TrainSection::Distribute(_) => {
                    return Err(CodegenError::new(
                        "train block `distribute:` sections are not supported; \
                         configure distribution via the @pipeline decorator / \
                         CLI options instead",
                    ));
                }
                // Deferral-must-refuse: these previously fell into a
                // `_ => {}` wildcard and were silently dropped — a
                // scheduler: section compiled clean under @pipeline and
                // trained at constant lr; callbacks: never fired.
                TrainSection::Scheduler(_) => {
                    return Err(CodegenError::new(
                        "scheduler: sections are not supported on the \
                         @pipeline train path yet — its per-stage loop \
                         never updates the learning rate, so the schedule \
                         would be silently ignored. Remove the section or \
                         drop @pipeline",
                    ));
                }
                TrainSection::Callbacks(_) => {
                    return Err(CodegenError::new(
                        "callbacks: sections are not supported on the \
                         @pipeline train path yet — the per-stage loop \
                         never invokes them, so on_step/on_epoch logic \
                         would silently not run. Remove the section or \
                         drop @pipeline",
                    ));
                }
            }
        }

        let model_sym = model_sym.ok_or_else(|| {
            CodegenError::new("pipelined train block requires 'model=<ident>' config argument")
        })?;

        // (Missing-optimizer refusal moved into resolve_optim_config.)

        let (step_body, step_param_sym) = step_body
            .ok_or_else(|| CodegenError::new("pipelined train block requires a step section"))?;

        // ── 3. Resolve model and build param_list ───────────────────────
        let (model_var, _) = *state.variables.get(&model_sym).ok_or_else(|| {
            CodegenError::new(format!(
                "undefined model variable '{}' in pipelined train block",
                self.resolve_sym(model_sym)
            ))
        })?;
        let model_ptr = builder.use_var(model_var);

        let model_var_name = self.resolve_sym(model_sym).to_string();
        let model_type_name = {
            let mut found_name = None;
            for (_node_id, ty) in self.type_map.iter() {
                match ty {
                    nsl_semantic::types::Type::Model { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.types.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                            break;
                        }
                    }
                    nsl_semantic::types::Type::Struct { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.types.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                        }
                    }
                    _ => {}
                }
            }
            found_name.unwrap_or_else(|| model_var_name.clone())
        };

        let layout = self
            .types
            .struct_layouts
            .get(&model_type_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "no struct layout found for model '{}' in pipelined train block",
                    model_type_name
                ))
            })?;

        // Build param_list by recursively collecting tensor fields (same as
        // non-pipelined path — handles nested sub-models and FixedArray fields).
        let num_slots = builder
            .ins()
            .iconst(cl_types::I64, (layout.total_size / 8) as i64);
        let param_list = self.compile_call_by_name(
            builder,
            "nsl_collect_model_params",
            &[model_ptr, num_slots],
        )?;
        let num_params_val = self.compile_call_by_name(builder, "nsl_list_len", &[param_list])?;

        // ── 4. Create optimizer state buffers (runtime NslLists) ────────
        // Optimizer-state offload is NOT wired on the pipelined path (its
        // optimizer emission does not run through the shared envelope
        // helpers) — refuse rather than silently keeping state on-device.
        if self.compile_options.train.optim_state_offload {
            return Err(CodegenError::new(
                "--optim-state-offload is not supported for pipelined train \
                 blocks yet; remove the flag or use the non-pipelined path.",
            ));
        }
        // P5 Muon: the mixed Muon/AdamW step (routing flags + Newton-Schulz
        // args) is wired into the non-pipelined emitter only. Refuse loudly
        // rather than emit a stale-arity call into the upgraded stdlib fn.
        if optimizer_name == "muon" {
            return Err(CodegenError::new(
                "the mixed Muon/AdamW optimizer is not wired into @pipeline \
                 train blocks yet — drop @pipeline or use adamw/sgd/lion/soap",
            ));
        }
        let num_state_buffers = match optimizer_name.as_str() {
            "adam" | "adamw" | "soap" => 2,
            _ => 1,
        };

        let state_list_1 = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        let state_list_2 = if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_new", &[])?
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };

        // Runtime loop: for i in 0..num_params, create zeros_like(param_list[i])
        {
            let init_i = builder.declare_var(cl_types::I64);
            let init_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(init_i, init_zero);
            let hdr = builder.create_block();
            let body = builder.create_block();
            let exit = builder.create_block();
            builder.ins().jump(hdr, &[]);
            builder.switch_to_block(hdr);
            builder.seal_block(hdr);
            let i = builder.use_var(init_i);
            let c = builder.ins().icmp(IntCC::SignedLessThan, i, num_params_val);
            builder.ins().brif(c, body, &[], exit, &[]);
            builder.switch_to_block(body);
            builder.seal_block(body);
            state.current_block = Some(body);
            let p = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, i])?;
            let b1 = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[state_list_1, b1])?;
            if num_state_buffers >= 2 {
                let b2 = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
                self.compile_call_by_name(builder, "nsl_list_push", &[state_list_2, b2])?;
            }
            let one = builder.ins().iconst(cl_types::I64, 1);
            let next = builder.ins().iadd(i, one);
            builder.def_var(init_i, next);
            builder.ins().jump(hdr, &[]);
            builder.switch_to_block(exit);
            builder.seal_block(exit);
            state.current_block = Some(exit);
        }

        // ── 5. Declare step parameter and step counter ──────────────────
        let step_param_var = builder.declare_var(cl_types::I64);
        let init_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_param_var, init_null);
        state
            .variables
            .insert(step_param_sym, (step_param_var, cl_types::I64));

        let step_count_var = builder.declare_var(cl_types::I64);
        let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_count_var, zero_i64);

        // Phase 5 Task 7: publish step counter for @inspect in pipelined train.
        self.inspect_train_step_var = Some(step_count_var);

        let lr_var = builder.declare_var(cl_types::F64);
        let lr_const = builder.ins().f64const(lr_value);
        builder.def_var(lr_var, lr_const);

        // ── 6. Forward pass under tape recording ────────────────────────
        let true_val = builder.ins().iconst(cl_types::I8, 1);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        state.flags.in_tape_region = true;
        for stmt in &step_body.stmts {
            self.compile_stmt(builder, state, stmt)?;
        }
        // ELTLS: free tape-held tensors before clearing the tape flag.
        self.free_tape_held_tensors(builder, state);
        state.flags.in_tape_region = false;

        // Find loss variable
        let loss_val = {
            let mut found = None;
            for (sym, (var, _)) in &state.variables {
                if self.resolve_sym(*sym) == "loss" {
                    found = Some(builder.use_var(*var));
                    break;
                }
            }
            found.ok_or_else(|| {
                CodegenError::new(
                    "pipelined train step body must assign to a variable named 'loss'",
                )
            })?
        };

        // ── 7. Activation send — send loss to next stage ────────────────
        // In single-process pipeline, stage 0 sends activations to logical
        // stage 1. The runtime's shared-memory backend serializes the tensor
        // into a mailbox keyed by (dst_rank, tag).
        let zero_tag = builder.ins().iconst(cl_types::I64, 0);
        let zero_stream = builder.ins().iconst(cl_types::I64, 0);
        let next_stage = builder.ins().iconst(cl_types::I64, 1);
        self.compile_call_by_name(
            builder,
            "nsl_pipeline_send",
            &[loss_val, next_stage, zero_tag, zero_stream],
        )?;

        // ── 8. Backward pass — tape backward + stop ─────────────────────
        let grads_list =
            self.compile_call_by_name(builder, "nsl_tape_backward", &[loss_val, param_list])?;
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;

        let false_val = builder.ins().iconst(cl_types::I8, 0);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

        // ── 9. Gradient send — serialize each param gradient ────────────
        // Send gradients to the previous stage (stage 0 receives gradients
        // from stage 1 in the backward direction). Each gradient is tagged
        // with its parameter index for correct matching.
        let prev_stage = builder.ins().iconst(cl_types::I64, 0);
        {
            let gs_i = builder.declare_var(cl_types::I64);
            let gs_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(gs_i, gs_zero);
            let gs_hdr = builder.create_block();
            let gs_body = builder.create_block();
            let gs_exit = builder.create_block();
            builder.ins().jump(gs_hdr, &[]);
            builder.switch_to_block(gs_hdr);
            builder.seal_block(gs_hdr);
            let gi = builder.use_var(gs_i);
            let gc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, gi, num_params_val);
            builder.ins().brif(gc, gs_body, &[], gs_exit, &[]);
            builder.switch_to_block(gs_body);
            builder.seal_block(gs_body);
            state.current_block = Some(gs_body);
            let grad_val = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, gi])?;
            self.compile_call_by_name(
                builder,
                "nsl_pipeline_send_grad",
                &[grad_val, prev_stage, gi, zero_stream],
            )?;
            let g_one = builder.ins().iconst(cl_types::I64, 1);
            let g_next = builder.ins().iadd(gi, g_one);
            builder.def_var(gs_i, g_next);
            builder.ins().jump(gs_hdr, &[]);
            builder.switch_to_block(gs_exit);
            builder.seal_block(gs_exit);
            state.current_block = Some(gs_exit);
        }

        // ── 10. Optimizer step ──────────────────────────────────────────
        // H.2: see comment in the non-pipelined emitter — `stdlib_loader`
        // produces `nsl_optim_sgd__sgd_step` (single underscore between
        // path parts), so this site must match that convention.
        let optimizer_fn_name = match optimizer_name.as_str() {
            "sgd" => "nsl_optim_sgd__sgd_step",
            "adam" => "nsl_optim_adam__adam_step",
            "adamw" => "nsl_optim_adamw__adamw_step",
            "lion" => "nsl_optim_lion__lion_step",
            "muon" => "nsl_optim_muon__muon_step",
            "soap" => "nsl_optim_soap__soap_step",
            _ => {
                return Err(CodegenError::new(format!(
                    "unsupported optimizer '{}' in pipelined train block",
                    optimizer_name
                )));
            }
        };

        let opt_fn = if self.registry.functions.contains_key(optimizer_fn_name) {
            optimizer_fn_name.to_string()
        } else {
            let simple = format!("{}_step", optimizer_name);
            if self.registry.functions.contains_key(&simple) {
                simple
            } else if self.registry.runtime_fns.contains_key(optimizer_fn_name) {
                optimizer_fn_name.to_string()
            } else if self.registry.runtime_fns.contains_key(&simple) {
                simple
            } else {
                optimizer_fn_name.to_string()
            }
        };

        // AdamW parameter groups are not wired through the @pipeline path:
        // its optimizer step is emitted per pipeline stage rather than over
        // the model's flat param list, so the role table's positional flags
        // have no list to be parallel to here. Refuse rather than decay the
        // parameters the user asked to exempt.
        if !no_decay_scope.is_empty() {
            return Err(CodegenError::new(
                "no_decay=[...] is not supported on the @pipeline train path \
                 yet: its optimizer step is emitted per stage rather than over \
                 the flat parameter list the role flags index. Drop one",
            ));
        }

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

        {
            let opt_i = builder.declare_var(cl_types::I64);
            let opt_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(opt_i, opt_zero);
            let opt_hdr = builder.create_block();
            let opt_body = builder.create_block();
            let opt_exit = builder.create_block();
            builder.ins().jump(opt_hdr, &[]);
            builder.switch_to_block(opt_hdr);
            builder.seal_block(opt_hdr);
            let idx = builder.use_var(opt_i);
            let oc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, idx, num_params_val);
            builder.ins().brif(oc, opt_body, &[], opt_exit, &[]);
            builder.switch_to_block(opt_body);
            builder.seal_block(opt_body);
            state.current_block = Some(opt_body);

            let param_val =
                self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;
            let grad_val =
                self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, idx])?;
            let s1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, idx])?;

            match optimizer_name.as_str() {
                "sgd" => {
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            lr,
                            momentum_const,
                            dampening_const,
                            weight_decay_const,
                            nesterov_const,
                        ],
                    )?;
                }
                "adam" | "adamw" => {
                    let s2 =
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, idx])?;
                    let t_val = builder.use_var(step_count_var);
                    let one = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_one = builder.ins().iadd(t_val, one);
                    let t_float = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_one);
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            s2,
                            lr,
                            beta1_const,
                            beta2_const,
                            eps_const,
                            weight_decay_const,
                            t_float,
                        ],
                    )?;
                }
                "lion" => {
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            lr,
                            beta1_const,
                            beta2_const,
                            weight_decay_const,
                        ],
                    )?;
                }
                "muon" => {
                    // Muon refuses at the top of this emitter (mixed
                    // Muon/AdamW is not wired for @pipeline). Keep this arm
                    // an ERROR, not a call: the old 7-arg call shape no
                    // longer matches the 14-param mixed stdlib fn, and a
                    // silently re-enabled arm would pass garbage into
                    // adamw_route/betas/t.
                    return Err(CodegenError::new(
                        "internal: muon reached the pipelined optimizer arm \
                         despite the @pipeline refusal — mixed Muon/AdamW is \
                         not wired for pipelined train blocks",
                    ));
                }
                "soap" => {
                    let s2 =
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, idx])?;
                    let t_val_p = builder.use_var(step_count_var);
                    let one_p = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_p = builder.ins().iadd(t_val_p, one_p);
                    let t_float_p = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_p);
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            s2,
                            lr,
                            beta1_const,
                            beta2_const,
                            eps_const,
                            t_float_p,
                        ],
                    )?;
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "unsupported optimizer '{}' in pipelined train block",
                        optimizer_name
                    )));
                }
            }

            let o_one = builder.ins().iconst(cl_types::I64, 1);
            let o_next = builder.ins().iadd(idx, o_one);
            builder.def_var(opt_i, o_next);
            builder.ins().jump(opt_hdr, &[]);
            builder.switch_to_block(opt_exit);
            builder.seal_block(opt_exit);
            state.current_block = Some(opt_exit);
        }

        // ── 11. Increment step count ────────────────────────────────────
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iconst(cl_types::I64, 1);
        let sc_next = builder.ins().iadd(sc, one_i64);
        builder.def_var(step_count_var, sc_next);

        // ── 12. Barrier — synchronize all pipeline stages ───────────────
        self.compile_call_by_name(builder, "nsl_pipeline_barrier", &[])?;

        // ── 13. Cleanup — free gradients, param_list, optimizer buffers ─
        // Runtime loop for gradient + state buffer cleanup
        {
            let cl_i = builder.declare_var(cl_types::I64);
            let cl_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(cl_i, cl_zero);
            let cl_hdr = builder.create_block();
            let cl_body = builder.create_block();
            let cl_exit = builder.create_block();
            builder.ins().jump(cl_hdr, &[]);
            builder.switch_to_block(cl_hdr);
            builder.seal_block(cl_hdr);
            let ci = builder.use_var(cl_i);
            let cc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ci, num_params_val);
            builder.ins().brif(cc, cl_body, &[], cl_exit, &[]);
            builder.switch_to_block(cl_body);
            builder.seal_block(cl_body);
            state.current_block = Some(cl_body);
            // Free gradient
            let gv = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, ci])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[gv])?;
            // Free state buffers
            let sb1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, ci])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[sb1])?;
            if num_state_buffers >= 2 {
                let sb2 =
                    self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, ci])?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[sb2])?;
            }
            let cl_one = builder.ins().iconst(cl_types::I64, 1);
            let cl_next = builder.ins().iadd(ci, cl_one);
            builder.def_var(cl_i, cl_next);
            builder.ins().jump(cl_hdr, &[]);
            builder.switch_to_block(cl_exit);
            builder.seal_block(cl_exit);
            state.current_block = Some(cl_exit);
        }
        self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[state_list_1])?;
        if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_free", &[state_list_2])?;
        }

        // ── 14. Pipeline destroy ────────────────────────────────────────
        self.compile_call_by_name(builder, "nsl_pipeline_destroy", &[])?;

        state.variables = saved_variables;
        state.variable_types = saved_variable_types;
        state.dataloader_symbols = saved_dataloader_symbols;
        state.borrowed_batch_symbols = saved_borrowed_batch_symbols;

        // Phase 5 Task 7: clear train-scope @inspect context on exit.
        self.inspect_train_step_var = None;

        // Gap I.B: drop stale CSHA per-function cache entries so a
        // subsequent train/grad block in the same module gets a clean
        // slate (Cranelift `Value` IDs reset per function and would
        // otherwise alias against leftover keys).
        self.clear_csha_per_function_caches();

        Ok(())
    }
}
