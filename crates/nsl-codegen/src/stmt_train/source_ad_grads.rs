//! Section 8 of the train block's source-AD arm: the parameter-gradient
//! list. Under the FASE hook every gradient was already consumed during
//! adjoint lowering, so the list is a null sentinel; otherwise one zero
//! gradient per runtime parameter is allocated and the lowered adjoints
//! are swapped in by name, with the gradient-summary diagnostics. Then the
//! ownership sweep: the primal/adjoint intermediates the lowering owned
//! are released (minus what CCR, the hook and the CSLA save phase already
//! freed), the loss is retained for the epilogue, and training mode is
//! switched off.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 332 lines,
//! 16 inputs ([`SourceAdGradsInputs`]); the arm's value —
//! `(grads, loss, source_ad, wengert_freed)` — is the function's result.
//! The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`) pin the
//! emitted list on every source-AD fixture.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt::{classify_source_ad_param_name, SourceAdParamDiagnosticKind, SURFACE_GRADS};
use crate::stmt_train::csla_window::CslaPending;

/// Every binding of `compile_train_block_inner` the gradient collection
/// reads; names are the driver's.
pub(crate) struct SourceAdGradsInputs<'a> {
    /// Primal values CCR already freed during the forward.
    pub(crate) ccr_freed_primal: &'a std::collections::HashSet<crate::wengert::VarId>,
    /// The CSLA carrier, when the window save phase deferred the backward.
    pub(crate) csla_pending: &'a Option<CslaPending>,
    pub(crate) fase_hook_active: bool,
    /// Adjoint values the hook / replay already freed.
    pub(crate) freed_adjoint_vars: &'a std::collections::HashSet<crate::wengert::VarId>,
    /// The lowered primal (owned values, explicit frees).
    pub(crate) full_lowered: &'a crate::wengert_lower::LoweredWengert,
    pub(crate) full_vars: &'a crate::wengert_lower::VarMap,
    pub(crate) generator: &'a crate::source_ad::AdjointGenerator,
    /// The forward extractor (its named parameter VarIds).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// The inline-lowered adjoint (`None` under the FASE hook and CSLA).
    pub(crate) grad_lowered: &'a Option<crate::wengert_lower::LoweredWengert>,
    pub(crate) grad_vars: Option<&'a crate::wengert_lower::VarMap>,
    pub(crate) loss_val: Value,
    pub(crate) loss_var_id: crate::wengert::VarId,
    pub(crate) model_type_name: &'a str,
    pub(crate) model_var_name: &'a str,
    pub(crate) num_params_val: Value,
    pub(crate) param_list: Value,
}

impl Compiler<'_> {
    /// Emit the parameter-gradient list and the ownership sweep (see the
    /// module header); returns the source-AD arm's value
    /// `(grads, loss, source_ad, wengert_freed)`.
    pub(crate) fn emit_source_ad_grads(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: SourceAdGradsInputs<'_>,
    ) -> Result<(Value, Value, bool, std::collections::HashSet<Value>), CodegenError> {
        let SourceAdGradsInputs {
            ccr_freed_primal,
            csla_pending,
            fase_hook_active,
            freed_adjoint_vars,
            full_lowered,
            full_vars,
            generator,
            extractor,
            grad_lowered,
            grad_vars,
            loss_val,
            loss_var_id,
            model_type_name,
            model_var_name,
            num_params_val,
            param_list,
        } = inputs;

        // 8. Collect parameter gradients into grads_list (NslList)
        //
        // When FASE hook is active, every parameter gradient was already
        // consumed (accumulated into m_partial + freed) during adjoint
        // lowering.  Skip the grads_list construction entirely and emit
        // a null sentinel so downstream code that is also guarded by
        // `!fase_hook_active` never dereferences it.
        //
        // When hook is inactive: seed grads_list from the runtime
        // param_list so it always has exactly one slot per collected
        // parameter, then overwrite the matching slots using source-AD
        // gradients keyed by parameter pointer identity.  This avoids
        // relying on a compile-time DFS path enumeration matching the
        // runtime collector's traversal.
        let grads = if !fase_hook_active {
        let grads_inner = self.compile_call_by_name(builder, "nsl_list_new", &[])?;

        // P0.1: gradient seed buffers under the Grads surface. The
        // ambient surface here is Activations (set at the transient-
        // pool flip); get/set keeps the bracket nesting-safe.
        let surface_grads_prev =
            self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
        let surface_grads = builder.ins().iconst(cl_types::I8, SURFACE_GRADS);
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_grads])?;

        // 8a. Initialize one zero gradient per runtime parameter.
        let fill_i_var = builder.declare_var(cl_types::I64);
        let fill_zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(fill_i_var, fill_zero);
        let fill_hdr = builder.create_block();
        let fill_body = builder.create_block();
        let fill_exit = builder.create_block();
        builder.ins().jump(fill_hdr, &[]);
        builder.switch_to_block(fill_hdr);
        let fi = builder.use_var(fill_i_var);
        let fc = builder
            .ins()
            .icmp(IntCC::SignedLessThan, fi, num_params_val);
        builder.ins().brif(fc, fill_body, &[], fill_exit, &[]);
        builder.switch_to_block(fill_body);
        builder.seal_block(fill_body);
        let p = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, fi])?;
        let z = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
        self.compile_call_by_name(builder, "nsl_list_push", &[grads_inner, z])?;
        let fill_one = builder.ins().iconst(cl_types::I64, 1);
        let fill_next = builder.ins().iadd(fi, fill_one);
        builder.def_var(fill_i_var, fill_next);
        builder.ins().jump(fill_hdr, &[]);
        builder.seal_block(fill_hdr);
        builder.switch_to_block(fill_exit);
        builder.seal_block(fill_exit);
        state.current_block = Some(fill_exit);

        // End of the Grads bracket — restore the ambient surface.
        self.compile_call_by_name(
            builder,
            "nsl_gpu_set_alloc_surface",
            &[surface_grads_prev],
        )?;

        // 8b. Replace zero slots with actual source-AD gradients by
        // scanning the runtime param_list for each resolved parameter
        // leaf from the extracted forward graph.
        let tensor_param_paths: std::collections::HashSet<String> = self
            .enumerate_all_model_tensor_paths(model_var_name, model_type_name)
            .into_iter()
            .collect();
        let trainable_tensor_param_paths: std::collections::HashSet<String> =
            tensor_param_paths
                .iter()
                .filter(|path| self.is_trainable_param_name(path))
                .cloned()
                .collect();
        let mut seen_trainable_tensor_params: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        let mut grad_connected = 0usize;
        let mut grad_ignored_config_tensor = 0usize;
        let mut grad_ignored_non_tensor = 0usize;
        let mut grad_skipped_no_primal = 0usize;
        let mut grad_skipped_no_adjoint = 0usize;
        let mut grad_skipped_no_lowered = 0usize;
        for (param_name, vid) in extractor.named_param_var_ids() {
            match classify_source_ad_param_name(param_name, &tensor_param_paths) {
                SourceAdParamDiagnosticKind::Trainable => {
                    seen_trainable_tensor_params.insert(param_name.clone());
                }
                SourceAdParamDiagnosticKind::IgnoredConfig => {
                    grad_ignored_config_tensor += 1;
                    continue;
                }
                SourceAdParamDiagnosticKind::IgnoredNonTensor => {
                    grad_ignored_non_tensor += 1;
                    continue;
                }
            }
            let Some(param_ptr) = full_vars.get(vid).copied() else {
                nsl_log::nsl_log!(INFO, "nsl", "[nsl] source AD: param '{}' has no primal value (VarId {:?} not in full_vars)", param_name, vid);
                grad_skipped_no_primal += 1;
                continue;
            };
            let Some(adj_vid) = generator.adjoint_of(*vid) else {
                nsl_log::nsl_log!(INFO, "nsl", "[nsl] source AD: param '{}' has no adjoint (VarId {:?} — no gradient generated)", param_name, vid);
                grad_skipped_no_adjoint += 1;
                continue;
            };
            let Some(grad_val) = grad_vars
                .expect("non-hook path always lowers the adjoint inline")
                .get(&adj_vid)
                .copied()
            else {
                nsl_log::nsl_log!(INFO, "nsl", "[nsl] source AD: param '{}' adjoint VarId {:?} not in lowered grad vars (cascade skip)", param_name, adj_vid);
                grad_skipped_no_lowered += 1;
                continue;
            };
            grad_connected += 1;

            // WRGA B.3.2 Option 3: adapter-injected params (lora_A_*,
            // lora_B_*, ia3_scale_*, gate_*) are NOT in the runtime
            // param_list (the runtime list is built from
            // `enumerate_model_tensor_paths`, which excludes side-table
            // entries). Skip the align-with-runtime scan for them —
            // we count them as connected for the gradient-summary
            // diagnostic but don't emit the `nsl_assert` that would
            // always fire "missing param" for these paths.
            let leaf = param_name
                .rsplit('.')
                .next()
                .unwrap_or(param_name.as_str());
            if crate::expr::access::is_synthesized_adapter_field_name(leaf) {
                continue;
            }

            let scan_i_var = builder.declare_var(cl_types::I64);
            let scan_match_count_var = builder.declare_var(cl_types::I64);
            let scan_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(scan_i_var, scan_zero);
            builder.def_var(scan_match_count_var, scan_zero);

            let scan_hdr = builder.create_block();
            let scan_body = builder.create_block();
            let scan_match = builder.create_block();
            let scan_next = builder.create_block();
            let scan_exit = builder.create_block();

            builder.ins().jump(scan_hdr, &[]);
            builder.switch_to_block(scan_hdr);
            let si = builder.use_var(scan_i_var);
            let scan_cond = builder
                .ins()
                .icmp(IntCC::SignedLessThan, si, num_params_val);
            builder
                .ins()
                .brif(scan_cond, scan_body, &[], scan_exit, &[]);

            builder.switch_to_block(scan_body);
            builder.seal_block(scan_body);
            let runtime_param =
                self.compile_call_by_name(builder, "nsl_list_get", &[param_list, si])?;
            let is_match = builder.ins().icmp(IntCC::Equal, runtime_param, param_ptr);
            builder
                .ins()
                .brif(is_match, scan_match, &[], scan_next, &[]);

            builder.switch_to_block(scan_match);
            builder.seal_block(scan_match);
            let old_grad =
                self.compile_call_by_name(builder, "nsl_list_get", &[grads_inner, si])?;
            // ELTLS (FBIP-3): nsl_tensor_add takes a flags byte (flags=0 here).
            let flags_zero = builder.ins().iconst(cl_types::I8, 0);
            let summed_grad = self.compile_call_by_name(
                builder,
                "nsl_tensor_add",
                &[old_grad, grad_val, flags_zero],
            )?;
            let match_count = builder.use_var(scan_match_count_var);
            let match_one = builder.ins().iconst(cl_types::I64, 1);
            let next_match_count = builder.ins().iadd(match_count, match_one);
            builder.def_var(scan_match_count_var, next_match_count);
            self.compile_call_by_name(builder, "nsl_tensor_free", &[old_grad])?;
            self.compile_call_by_name(builder, "nsl_list_set", &[grads_inner, si, summed_grad])?;
            builder.ins().jump(scan_next, &[]);

            builder.switch_to_block(scan_next);
            builder.seal_block(scan_next);
            let scan_one = builder.ins().iconst(cl_types::I64, 1);
            let scan_inc = builder.ins().iadd(si, scan_one);
            builder.def_var(scan_i_var, scan_inc);
            builder.ins().jump(scan_hdr, &[]);

            builder.seal_block(scan_hdr);
            builder.switch_to_block(scan_exit);
            builder.seal_block(scan_exit);
            let match_count = builder.use_var(scan_match_count_var);
            let matched = builder.ins().icmp_imm_s(IntCC::NotEqual, match_count, 0);
            let missing_msg = format!(
                "source AD gradient could not be aligned with runtime param list: {}",
                param_name,
            );
            self.intern_string(&missing_msg)?;
            let missing_msg_ptr = self.compile_string_literal(builder, &missing_msg)?;
            self.compile_call_by_name(builder, "nsl_assert", &[matched, missing_msg_ptr])?;
            state.current_block = Some(scan_exit);
        }

        let grad_missing_trainable = trainable_tensor_param_paths
            .len()
            .saturating_sub(seen_trainable_tensor_params.len());
        nsl_log::nsl_log!(WARN, "nsl", 
            "[nsl] source AD gradient summary: {}/{} trainable tensor params connected, {} missing-from-forward, {} no-primal, {} no-adjoint, {} cascade-skip, {} ignored config-tensor, {} ignored non-tensor",
            grad_connected,
            trainable_tensor_param_paths.len(),
            grad_missing_trainable,
            grad_skipped_no_primal,
            grad_skipped_no_adjoint,
            grad_skipped_no_lowered,
            grad_ignored_config_tensor,
            grad_ignored_non_tensor,
        );
        grads_inner // value returned from the `if !fase_hook_active` arm
        } else {
            // Hook consumed all param grads during adjoint lowering.
            // Emit a null sentinel — downstream grads_list consumers are
            // all guarded by `!fase_hook_active`.
            builder.ins().iconst(cl_types::I64, 0)
        };

        let mut retained_full_vars = std::collections::HashSet::new();
        retained_full_vars.insert(loss_var_id);
        // CCR: block interiors were freed right after the forward
        // (the early-free mini list) — the bulk free must skip them.
        retained_full_vars.extend(ccr_freed_primal.iter().copied());
        // CCR compressed saves: originals were freed by the primal
        // TAIL (recorded on the main primal lowering), and the half
        // tensors were freed by the adjoint's decompress splice
        // (recorded on the adjoint lowering, but owned by the
        // PRIMAL's owned_values since the tail produced them).
        retained_full_vars.extend(full_lowered.explicit_freed_vars.iter().copied());
        if let Some(gl) = &grad_lowered {
            retained_full_vars.extend(gl.explicit_freed_vars.iter().copied());
        }
        // CSLA: the window-buffered imports must survive this
        // iteration — their frees happen after each buffered
        // micro-batch's replay in the window backward (or at the
        // window/teardown sweeps for shells and the partial tail).
        if let Some(p) = &csla_pending {
            retained_full_vars.extend(p.slots.iter().map(|(v, _)| *v));
            // LSE tape-carry: the PUSHED logsumexps ride owned_values
            // under the u32::MAX sentinel — buffered per micro-batch,
            // freed by the window backward (or the teardown sweep),
            // never by this per-iteration bulk free. Retention is
            // per-VALUE (review F2): sentinel entries whose aux out
            // mapped to no buffered slot were NOT pushed — free those
            // right here, per iteration, exactly as the baseline
            // would have.
            if !p.lse_slots.is_empty() {
                retained_full_vars.insert(u32::MAX);
                for (vid, val, _) in &full_lowered.owned_values {
                    if *vid == u32::MAX && !p.lse_pushed.contains(val) {
                        self.compile_call_by_name(
                            builder,
                            "nsl_tensor_free_if_valid",
                            &[*val],
                        )?;
                    }
                }
            }
        }
        if let Some(gl) = &grad_lowered {
            self.free_wengert_owned_values(
                builder,
                &gl.owned_values,
                freed_adjoint_vars,
            )?;
        }
        self.free_wengert_owned_values(
            builder,
            &full_lowered.owned_values,
            &retained_full_vars,
        )?;

        // Collect all Cranelift Values freed by the Wengert cleanup so the
        // step-variable sweep (below) doesn't double-free them.
        let mut wengert_freed: std::collections::HashSet<Value> = std::collections::HashSet::new();
        if let Some(gl) = &grad_lowered {
            for (vid, val, _) in &gl.owned_values {
                if !freed_adjoint_vars.contains(vid) {
                    wengert_freed.insert(*val);
                }
            }
        }
        for (vid, val, _) in &full_lowered.owned_values {
            if !retained_full_vars.contains(vid) {
                wengert_freed.insert(*val);
            }
            // CCR early-freed interiors (mini list), compressed
            // originals (primal tail) and halves (adjoint splice)
            // were freed outside the bulk cleanup — still "already
            // freed" as far as the step-variable sweep is concerned.
            if ccr_freed_primal.contains(vid)
                || full_lowered.explicit_freed_vars.contains(vid)
                || grad_lowered
                    .as_ref()
                    .is_some_and(|gl| gl.explicit_freed_vars.contains(vid))
            {
                wengert_freed.insert(*val);
            }
        }
        // CSLA: buffered imports are handled by the window backward's
        // per-b frees — the step-variable sweep must treat them as
        // already handled or it would free a buffered value that a
        // later micro-batch's replay still reads.
        if let Some(p) = &csla_pending {
            for (vid, _) in &p.slots {
                if let Some(v) = full_vars.get(vid) {
                    wengert_freed.insert(*v);
                }
            }
            // LSE tape-carry: the sentinel-owned logsumexp values are
            // buffer-managed too.
            if !p.lse_slots.is_empty() {
                for (vid, val, _) in &full_lowered.owned_values {
                    if *vid == u32::MAX {
                        wengert_freed.insert(*val);
                    }
                }
            }
        }

        let false_val = builder.ins().iconst(cl_types::I8, 0);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;


        Ok((grads, loss_val, true, wengert_freed))
    }
}
