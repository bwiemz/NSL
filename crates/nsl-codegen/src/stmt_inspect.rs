//! The `@inspect` hook emission (Dev Tools): `emit_inspect_hook` lowers the
//! decorator's predicate and the runtime probe call at a let-binding site,
//! gated on `compile_options.dev_tools.inspect_enabled`.
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1); the
//! let-binding lowering in `stmt.rs` still calls it.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::expr::ExprKind;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    /// Dev Tools Phase 5 Task 7: emit IR for one `@inspect(target, every=?, condition=?)`
    /// decorator attached to a `let` binding.
    ///
    /// Ship-first scope:
    ///   * Only fires inside a train block (requires `inspect_train_step_var`).
    ///     Outside train scope, emits nothing.
    ///   * `every=N` → step-gated call to `nsl_tensor_stats` +
    ///     `nsl_inspect_record_stats`.
    ///   * `condition="..."` → predicate-gated `nsl_inspect_dump_full`.
    ///     Predicate AST is lowered via `inspect::predicate::lower_predicate`.
    ///   * The `loss` identifier reads the most recent recorded loss at
    ///     runtime via `nsl_health_get_last_loss` (recorded per step whenever
    ///     `--inspect` or the health monitor is on). Because @inspect fires at
    ///     the let-binding site — before the current step's loss compute — the
    ///     predicate sees the previous completed step's loss (0.0 on step 0).
    ///
    /// All emission gated on `compile_options.dev_tools.inspect_enabled`.  When that
    /// flag is off, this method is never called.
    pub(crate) fn emit_inspect_hook(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        decorator: &nsl_ast::decl::Decorator,
        target_sym: nsl_ast::Symbol,
    ) -> Result<(), CodegenError> {
        // Outside a train block we skip entirely for Phase 5 ship-first.
        let step_count_var = match self.inspect_train_step_var {
            Some(v) => v,
            None => return Ok(()),
        };

        // Resolve target tensor: prefer decorator arg[0] (which semantic
        // guarantees is a positional Ident), fall back to the let binding's
        // own LHS symbol when argument extraction fails.
        let (resolved_sym, tensor_name) = {
            let mut s = target_sym;
            if let Some(args) = &decorator.args
                && let Some(first) = args.first()
                && first.name.is_none()
                && let ExprKind::Ident(sym) = &first.value.kind
            {
                s = *sym;
            }
            let name = self.resolve_sym(s).to_string();
            (s, name)
        };
        let tensor_val = match state.variables.get(&resolved_sym) {
            Some((var, _)) => builder.use_var(*var),
            None => return Ok(()),
        };

        // Extract every=N and condition="..." from decorator args.
        let mut every_n: Option<i64> = None;
        let mut cond_str: Option<String> = None;
        if let Some(args) = &decorator.args {
            // args[0] is the positional tensor target — resolved above.
            for arg in args.iter().skip(1) {
                let kw = arg.name.map(|s| self.resolve_sym(s).to_string());
                match kw.as_deref() {
                    Some("every") => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind
                            && *n > 0
                        {
                            every_n = Some(*n);
                        }
                    }
                    Some("condition") => {
                        if let ExprKind::StringLiteral(s) = &arg.value.kind {
                            cond_str = Some(s.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        // Intern the tensor name once — shared by stats + dump branches.
        let name_data_id = self.intern_string(&tensor_name)?;
        let name_gv = self
            .module
            .declare_data_in_func(name_data_id, builder.func);

        // ── (a) Stats branch: every=N ────────────────────────────────────
        if let Some(n) = every_n {
            let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
            let step_loaded = builder.use_var(step_count_var);
            let n_val = builder.ins().iconst(cl_types::I64, n);
            let rem = builder.ins().srem(step_loaded, n_val);
            let due = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                rem,
                zero_i64,
            );

            let do_block = builder.create_block();
            let after_block = builder.create_block();
            builder.ins().brif(due, do_block, &[], after_block, &[]);

            builder.switch_to_block(do_block);
            builder.seal_block(do_block);
            state.current_block = Some(do_block);

            // Allocate a 48-byte 8-aligned stack slot for the stats struct
            // (matches the runtime's NslTensorStats layout — 6 × f64).
            let slot = builder.create_sized_stack_slot(
                cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    48,
                    3,
                ),
            );
            let stats_ptr = builder.ins().stack_addr(cl_types::I64, slot, 0);

            self.compile_call_by_name(
                builder,
                "nsl_tensor_stats",
                &[tensor_val, stats_ptr],
            )?;

            let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
            let name_len = builder
                .ins()
                .iconst(cl_types::I64, tensor_name.len() as i64);
            let step_now = builder.use_var(step_count_var);
            self.compile_call_by_name(
                builder,
                "nsl_inspect_record_stats",
                &[stats_ptr, step_now, name_ptr, name_len],
            )?;

            builder.ins().jump(after_block, &[]);
            builder.switch_to_block(after_block);
            builder.seal_block(after_block);
            state.current_block = Some(after_block);
        }

        // ── (b) Dump branch: condition="..." ──────────────────────────────
        if let Some(cond_src) = cond_str {
            let ast = match crate::inspect::predicate::parse_predicate(&cond_src) {
                Ok(p) => p,
                Err(e) => {
                    nsl_runtime::nsl_log!(ERROR, "codegen", 
                        "[@inspect] predicate parse failed for {:?}: {}",
                        cond_src, e
                    );
                    return Ok(());
                }
            };

            // Resolve FuncRefs for all health getters.  Any missing symbol
            // means builtins.rs / Phase 4+5 runtime didn't register — bail.
            let (lema_id, _) = match self.registry.runtime_fns.get("nsl_health_get_loss_ema") {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_loss_ema_ref =
                self.module.declare_func_in_func(lema_id, builder.func);
            let (lslope_id, _) = match self
                .registry
                .runtime_fns
                .get("nsl_health_get_loss_ema_slope")
            {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_loss_ema_slope_ref =
                self.module.declare_func_in_func(lslope_id, builder.func);
            let (gnt_id, _) = match self
                .registry
                .runtime_fns
                .get("nsl_health_get_grad_norm_total")
            {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_grad_norm_total_ref =
                self.module.declare_func_in_func(gnt_id, builder.func);
            let (nic_id, _) = match self
                .registry
                .runtime_fns
                .get("nsl_health_get_nan_inf_count_window")
            {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_nan_inf_count_window_ref =
                self.module.declare_func_in_func(nic_id, builder.func);

            let (lloss_id, _) = match self.registry.runtime_fns.get("nsl_health_get_last_loss") {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_last_loss_ref =
                self.module.declare_func_in_func(lloss_id, builder.func);

            let step_loaded = builder.use_var(step_count_var);

            let ctx = crate::inspect::predicate::PredicateLowerCtx {
                step_val: step_loaded,
                get_last_loss_ref,
                get_loss_ema_ref,
                get_loss_ema_slope_ref,
                get_grad_norm_total_ref,
                get_nan_inf_count_window_ref,
            };
            let pred_val =
                crate::inspect::predicate::lower_predicate(&ast, builder, &ctx);

            let do_block = builder.create_block();
            let after_block = builder.create_block();
            builder.ins().brif(pred_val, do_block, &[], after_block, &[]);

            builder.switch_to_block(do_block);
            builder.seal_block(do_block);
            state.current_block = Some(do_block);

            let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
            let name_len = builder
                .ins()
                .iconst(cl_types::I64, tensor_name.len() as i64);
            let step_now = builder.use_var(step_count_var);
            self.compile_call_by_name(
                builder,
                "nsl_inspect_dump_full",
                &[tensor_val, step_now, name_ptr, name_len],
            )?;

            builder.ins().jump(after_block, &[]);
            builder.switch_to_block(after_block);
            builder.seal_block(after_block);
            state.current_block = Some(after_block);
        }

        Ok(())
    }
}
