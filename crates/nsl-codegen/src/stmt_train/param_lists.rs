//! The train block's per-parameter runtime lists: the Muon/AdamW route
//! flags, the weight-decay exemption flags and the gradient-accumulation
//! buffers. Each is an `NslList` parallel to `param_list`, built once at
//! setup between the parameter list and the optimizer-state buffers, and
//! each is one value the driver binds and hands on.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1).
//! The three blocks measured the same way #598 chose its seam: 53 / 80 /
//! 66 lines, one escaping local each (`muon_route_list`,
//! `decay_exempt_list`, `accum_list`), no `self` writes. The train-block
//! CLIF snapshots (`tests/train_clif_snapshots.rs`) pin the emitted
//! `nsl_list_new` / `nsl_list_push` calls and the accumulator loop on the
//! `grad_accumulation` fixtures; the Muon routing table and the
//! `[wd-groups]` report are stderr, pinned by `muon_state_gate.rs` and the
//! weight-decay-groups gates in `nsl-cli`.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::param_roles::NoDecayScope;
use crate::stmt::SURFACE_M_PARTIAL;

impl Compiler<'_> {
    /// P1 Muon item 6: the parameter-ROLE routing flags (`1` = force
    /// AdamW), as an i64 `NslList` parallel to `param_list`, for the
    /// `muon` optimizer only. Prints the routing table.
    pub(crate) fn muon_route_flags(
        &mut self,
        builder: &mut FunctionBuilder,
        optimizer_name: &str,
        model_type_name: &str,
        param_paths: &[String],
    ) -> Result<Option<Value>, CodegenError> {
        // Mixed Muon/AdamW routes by parameter ROLE — explicit @param_role
        // decorator > embedding_lookup-usage inference > declared rank >
        // default hidden (see param_roles.rs). The name-substring exclusion
        // list is gone. Flags ride in an i64 NslList parallel to param_list
        // (1 = force-AdamW); the rank-2 structural check remains the runtime
        // backstop inside muon_step. Built BEFORE the optimizer state
        // buffers so the v (second-moment) allocation can skip Muon-routed
        // params (item 9). The routing table prints loudly so a misrouted
        // param is never silent.
        let muon_route_list = if optimizer_name == "muon" {
            let table = self.classify_param_roles(model_type_name, param_paths);
            let list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            for e in &table.entries {
                let flag = builder.ins().iconst(cl_types::I64, e.adamw as i64);
                self.compile_call_by_name(builder, "nsl_list_push", &[list, flag])?;
            }
            let adamw_count = table.entries.iter().filter(|e| e.adamw).count();
            nsl_runtime::nsl_log!(INFO, "muon", 
                "[muon] role-based Muon/AdamW routing over {} params ({} \
                 AdamW-routed, {} Muon):",
                table.entries.len(),
                adamw_count,
                table.entries.len() - adamw_count,
            );
            for e in &table.entries {
                nsl_runtime::nsl_log!(INFO, "muon", 
                    "[muon]   {} role={} ({}) -> {}",
                    e.path,
                    e.role,
                    e.source,
                    if e.adamw {
                        "AdamW"
                    } else {
                        "Muon (if rank-2 at runtime)"
                    },
                );
            }
            if !table.entries.iter().any(|e| e.role == "head") {
                nsl_runtime::nsl_log!(INFO, "muon", 
                    "[muon] note: no param has role 'head' — correct for \
                     weight-tied models (the tied embedding covers it); if \
                     this model has an UNTIED lm_head, annotate it with \
                     @param_role(\"head\")."
                );
            }
            for w in &table.warnings {
                nsl_runtime::nsl_log!(WARN, "muon", "[muon] warning: {w}");
            }
            Some(list)
        } else {
            None
        };
        Ok(muon_route_list)
    }

    /// AdamW parameter groups: the per-parameter decay-exempt flags (`1` =
    /// exempt) for `no_decay=[...]`, as an i64 `NslList` parallel to
    /// `param_list`; `None` when the scope is empty. Prints the
    /// `[wd-groups]` table and refuses a scope that can never fire.
    pub(crate) fn decay_exempt_flags(
        &mut self,
        builder: &mut FunctionBuilder,
        no_decay_scope: &NoDecayScope,
        weight_decay_value: f64,
        model_type_name: &str,
        param_paths: &[String],
    ) -> Result<Option<Value>, CodegenError> {
        // `no_decay=[...]` names ROLES to exempt from weight decay. The
        // static half of the decision (embedding / head / hidden) is decided
        // here from the same role table Muon routes on; the `"vector"` half
        // is NOT, because a model field only has a statically-derivable rank
        // when its initializer is a direct zeros/ones/randn/... call over
        // integer literals — which real models are not. Both halves meet in
        // `nsl_optim_param_wd`, the single runtime rule.
        //
        // Flags ride in an i64 NslList parallel to param_list (1 = exempt),
        // exactly like the Muon route flags. The table prints loudly: a
        // parameter group that silently exempted nothing (or everything) is
        // the failure this feature is supposed to remove, not add.
        let decay_exempt_list = if no_decay_scope.is_empty() {
            None
        } else {
            if weight_decay_value == 0.0 {
                nsl_runtime::nsl_log!(WARN, "wd-groups", 
                    "[wd-groups] warning: no_decay=[...] was given but \
                     weight_decay is 0.0 — nothing is being decayed, so the \
                     exemption has no effect."
                );
            }
            let table = self.classify_param_roles(model_type_name, param_paths);
            let list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            let mut static_exempt = 0usize;
            for e in &table.entries {
                let exempt = no_decay_scope.exempts_role(e.role);
                static_exempt += usize::from(exempt);
                let flag = builder.ins().iconst(cl_types::I64, i64::from(exempt));
                self.compile_call_by_name(builder, "nsl_list_push", &[list, flag])?;
            }
            let mut scope_desc = no_decay_scope.static_roles.clone();
            if no_decay_scope.exempt_non_rank2 {
                scope_desc.push("vector (runtime rank != 2)".to_string());
            }
            nsl_runtime::nsl_log!(INFO, "wd-groups", 
                "[wd-groups] weight_decay={} exempting roles [{}] over {} params: \
                 {} exempt by role at compile time{}",
                weight_decay_value,
                scope_desc.join(", "),
                table.entries.len(),
                static_exempt,
                if no_decay_scope.exempt_non_rank2 {
                    ", plus every param that is not rank-2 at step time"
                } else {
                    ""
                },
            );
            for e in &table.entries {
                if no_decay_scope.exempts_role(e.role) {
                    nsl_runtime::nsl_log!(INFO, "wd-groups", 
                        "[wd-groups]   {} role={} ({}) -> NO decay",
                        e.path, e.role, e.source
                    );
                }
            }
            // Anti-vacuity: a scope that exempts nothing statically AND does
            // not ask for the runtime rank check cannot ever fire. That is a
            // configuration error, not a no-op to shrug at.
            if static_exempt == 0 && !no_decay_scope.exempt_non_rank2 {
                return Err(CodegenError::new(format!(
                    "no_decay=[{}] matches no parameter in this model — every \
                     param classified as one of the roles it does NOT name. \
                     Roles present: {}. Note that norms and biases usually \
                     classify as role `hidden` (their rank is not derivable \
                     from the field initializer), so use no_decay=[\"vector\"] \
                     to exempt them by runtime rank",
                    no_decay_scope.static_roles.join(", "),
                    {
                        let mut roles: Vec<&str> =
                            table.entries.iter().map(|e| e.role).collect();
                        roles.sort_unstable();
                        roles.dedup();
                        roles.join(", ")
                    },
                )));
            }
            Some(list)
        };
        Ok(decay_exempt_list)
    }

    /// The gradient-accumulation buffers (`grad_accumulation > 1`): one
    /// `zeros_like(param)` per parameter under the MPartial surface, or
    /// NULL slots under `--layerwise-accum`. Emits a runtime loop over
    /// `param_list` and leaves `state.current_block` on its exit block.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn alloc_grad_accum_buffers(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad_accumulation_steps: i64,
        csla_active: bool,
        num_params_val: Value,
        param_list: Value,
        surface_prev: Value,
    ) -> Result<Option<Value>, CodegenError> {
        // These persist across batches within each accumulation window. Each buffer
        // is zeros_like(param) and gets += each batch's grads, then zeroed after
        // the optimizer step every N batches.
        let accum_list = if grad_accumulation_steps > 1 {
            let list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            // P0.1: grad-accumulation buffers under the MPartial surface.
            let surface_m_partial = builder.ins().iconst(cl_types::I8, SURFACE_M_PARTIAL);
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_m_partial])?;
            // Runtime loop over param_list (not layout.fields — which may include sub-models)
            let accum_i_var = builder.declare_var(cl_types::I64);
            let accum_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(accum_i_var, accum_zero);
            let accum_hdr = builder.create_block();
            let accum_body = builder.create_block();
            let accum_exit = builder.create_block();
            builder.ins().jump(accum_hdr, &[]);
            builder.switch_to_block(accum_hdr);
            // Don't seal accum_hdr yet — back-edge not added
            let ai = builder.use_var(accum_i_var);
            let ac = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ai, num_params_val);
            builder.ins().brif(ac, accum_body, &[], accum_exit, &[]);
            builder.switch_to_block(accum_body);
            builder.seal_block(accum_body);
            let p = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, ai])?;
            // P3: under --optim-state-offload the FASE window accumulator
            // (m_partial) is the last device-resident param-sized f32 surface
            // (~4.15 GB at 1B). Allocate it HOST-resident (pinned) too; the
            // accumulate hook stages it to the grad's device per micro-batch
            // and the final step stages it in for the m/v update. m_partial
            // is ALWAYS f32 (exact-windowed semantics — the reduced-precision
            // moment path never touches it), so f32 host regardless of the
            // CPDT precision plan.
            //
            // CSLA (D1b): the whole point — do NOT allocate the full-model
            // window here. Slots start NULL; the window backward allocates
            // each layer's accumulators just before that layer's replay and
            // frees them right after its per-layer update, so the live
            // accumulator surface is max(one layer) + the epilogue globals
            // instead of 4·P bytes. (All accumulation happens inside the
            // window region under csla — the per-micro-batch ga_body loop is
            // hook-skipped — so a NULL slot is never read between windows.)
            let zeros = if csla_active {
                builder.ins().iconst(cl_types::I64, 0)
            } else if self.compile_options.train.optim_state_offload {
                self.compile_call_by_name(builder, "nsl_tensor_zeros_like_host_f32", &[p])?
            } else {
                self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?
            };
            self.compile_call_by_name(builder, "nsl_list_push", &[list, zeros])?;
            let a_one = builder.ins().iconst(cl_types::I64, 1);
            let a_next = builder.ins().iadd(ai, a_one);
            builder.def_var(accum_i_var, a_next);
            builder.ins().jump(accum_hdr, &[]);
            builder.seal_block(accum_hdr);
            builder.switch_to_block(accum_exit);
            builder.seal_block(accum_exit);
            state.current_block = Some(accum_exit);
            // End of the MPartial bracket — restore the caller's surface.
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;
            Some(list)
        } else {
            None
        };
        Ok(accum_list)
    }
}
