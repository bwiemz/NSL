//! The FASE-hook arm of the adjoint lowering (section 7 of the train
//! block's source-AD arm). Under FASE Deferred every parameter gradient
//! is consumed the moment its adjoint op is lowered: the per-parameter
//! callback accumulates it into the `m_partial` slot (or emits the fused
//! `--fuse-wgrad-accum` GEMM), notes it for the grad-integrity gate and
//! frees it unless a later adjoint op still reads it; the lowering runs
//! inside the P0.3 grad-integrity bracket with the P0.2 live set armed.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 163 lines,
//! 9 inputs ([`FaseHookLoweringInputs`]); the arm's value — the
//! lowered adjoint — is the function's result. The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the emitted accumulate calls on
//! every FASE-deferred fixture.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt::ParamHookEntry;

/// Every binding of `compile_train_block_inner` the FASE-hook lowering
/// reads; names are the driver's.
pub(crate) struct FaseHookLoweringInputs<'a> {
    /// The gradient-accumulation buffer list (`Some` whenever the hook is active).
    pub(crate) accum_list: Option<Value>,
    /// Adjoint VarId → (param name, primal value, accumulation index).
    pub(crate) adj_vid_to_hook_entry: &'a std::collections::HashMap<crate::wengert::VarId, ParamHookEntry>,
    /// The final adjoint tape.
    pub(crate) adjoint: &'a crate::wengert::WengertList,
    pub(crate) fase_plan: &'a crate::fase::FasePlan,
    pub(crate) full_vars: &'a crate::wengert_lower::VarMap,
    /// The P0.2 gradient-integrity live set, armed only for this lowering.
    pub(crate) grad_live_set: &'a Option<std::collections::HashSet<crate::wengert::VarId>>,
    pub(crate) num_params_val: Value,
    /// The adjoint VarIds the hook consumes.
    pub(crate) param_adj_set: &'a std::collections::HashSet<crate::wengert::VarId>,
    pub(crate) param_list: Value,
}

impl Compiler<'_> {
    /// Lower the adjoint with the FASE per-parameter hook (see the module
    /// header); returns the lowered adjoint.
    pub(crate) fn emit_fase_hook_adjoint_lowering(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: FaseHookLoweringInputs<'_>,
    ) -> Result<Option<crate::wengert_lower::LoweredWengert>, CodegenError> {
        let FaseHookLoweringInputs {
            accum_list,
            adj_vid_to_hook_entry,
            adjoint,
            fase_plan,
            full_vars,
            grad_live_set,
            num_params_val,
            param_adj_set,
            param_list,
        } = inputs;

        // FASE Deferred: consume each param gradient immediately.
        // The callback receives &mut Compiler explicitly so no
        // double-borrow occurs.
        let accum_val = accum_list.ok_or_else(|| {
            CodegenError::new(
                "fase_hook_active requires accum_list to be Some",
            )
        })?;
        let accum_scale = fase_plan.recipe.accum_scale;
        let num_params = num_params_val;
        let hook_map = adj_vid_to_hook_entry;
        let plist = param_list;
        let mut fase_cb = |c: &mut Compiler,
                           var_id: crate::wengert::VarId,
                           grad_src: crate::wengert_lower::ParamGradSource,
                           still_needed: bool,
                           b: &mut cranelift_frontend::FunctionBuilder|
         -> Result<(), CodegenError> {
            let Some(entry) = hook_map.get(&var_id) else {
                return Ok(());
            };
            // Runtime pointer-scan: find the index in param_list that
            // matches the primal param pointer, then use the same index
            // for accum_list.  This is necessary because param_list and
            // accum_list are both indexed by param_paths order, but a
            // given primal_val may appear at any runtime slot if the
            // model has shared/aliased weights.
            //
            // Fast path: use the compile-time accum_idx directly.
            // accum_list[accum_idx] == the m_partial for this param
            // (both are param_paths-ordered).
            let _ = entry.primal_val; // present for future alias detection
            let idx_val = b.ins().iconst(cranelift_codegen::ir::types::I64, entry.accum_idx);
            let _ = num_params; // captured for guard assertions if needed
            let _ = plist;
            let m_partial =
                c.compile_call_by_name(b, "nsl_list_get", &[accum_val, idx_val])?;
            let off = c.compile_options.optim_state_offload;
            // Item 7: the fused chain never materializes a
            // gradient tensor — emit the accumulating GEMM over
            // the chain's operands and we are done. There is
            // nothing to note for grad-integrity and nothing to
            // free.
            //
            // These used to be `debug_assert!`s justified by "both
            // compositions are refused at option validation". That
            // held only while clap was the sole route here: clap
            // conflicts `--fuse-wgrad-accum` with both flags at
            // PARSE time. `--pretrain-optimized` now enables the
            // fusion from `expand_pretrain_optimized`, which runs
            // after parsing and is therefore invisible to clap, so
            // the bundle's own blocker list is the only thing
            // keeping this state unreachable. A debug_assert is a
            // no-op in release (the workspace sets no
            // `[profile.release] debug-assertions`), so a lapse in
            // that list would not abort — it would silently emit a
            // grad-integrity report attesting parameters whose
            // gradients were never observed, or aim the device GEMM
            // at a host-resident `m_partial`. Fail loudly instead;
            // this is unreachable by construction, so the cost is
            // zero and the value is that it stays that way.
            let grad_ptr = match grad_src {
                crate::wengert_lower::ParamGradSource::FusedWgrad { x, g } => {
                    if c.compile_options.diagnostics.grad_integrity {
                        return Err(CodegenError::new(
                            "internal: --fuse-wgrad-accum reached lowering with \
                             --grad-integrity active. The fused chain never \
                             materializes a gradient tensor, so the integrity gate \
                             would attest parameters it never observed. Whatever \
                             enabled the fusion (clap conflict, or the \
                             --pretrain-optimized blocker list in \
                             crates/nsl-cli/src/meta_flags.rs) has a gap.",
                        ));
                    }
                    if off {
                        return Err(CodegenError::new(
                            "internal: --fuse-wgrad-accum reached lowering with \
                             --optim-state-offload active. `m_partial` is \
                             host-resident under offload and the fused device GEMM \
                             cannot write it. Whatever enabled the fusion (clap \
                             conflict, or the --pretrain-optimized blocker list in \
                             crates/nsl-cli/src/meta_flags.rs) has a gap.",
                        ));
                    }
                    let scale_val = b.ins().f64const(accum_scale);
                    c.compile_call_by_name(
                        b,
                        "nsl_tensor_wgrad_accum",
                        &[m_partial, x, g, scale_val],
                    )?;
                    return Ok(());
                }
                crate::wengert_lower::ParamGradSource::Materialized(v) => v,
            };
            // P0.3: note this parameter's gradient BEFORE accumulate
            // frees/consumes it. accum_idx == the param_paths index.
            if c.compile_options.diagnostics.grad_integrity {
                c.compile_call_by_name(
                    b,
                    "nsl_grad_integrity_note",
                    &[grad_ptr, idx_val],
                )?;
            }
            c.fase_emit_accumulate(b, m_partial, grad_ptr, accum_scale, off)?;
            // Free the raw gradient now ONLY if no later adjoint op
            // still reads it. When this param's grad adjoint is a
            // shared intermediate (a bias whose grad == d_out, which
            // the weight-grad matmul also consumes), the free is
            // DEFERRED to end-of-backward cleanup — freeing here
            // would drop the weight gradient (silently, pre-#396).
            // `fase_emit_accumulate` leaves grad_ptr intact (rc
            // unchanged), so the later op reads live data.
            if !still_needed {
                c.compile_call_by_name(b, "nsl_tensor_free", &[grad_ptr])?;
            }
            Ok(())
        };
        // P0.3: bracket the FASE backward with a grad-integrity step
        // (the hook notes each parameter's gradient between these).
        // This bracket wraps ONE micro-batch's adjoint lowering, so
        // every trainable param must be noted exactly once inside
        // it — anything else is a dropped or double-consumed
        // gradient, which is what the declared expectation catches.
        let gi = self.compile_options.diagnostics.grad_integrity;
        if gi {
            let one_note = builder.ins().iconst(cl_types::I64, 1);
            self.compile_call_by_name(
                builder,
                "nsl_grad_integrity_step_begin",
                &[num_params_val, one_note],
            )?;
        }
        // P0.2: arm the gradient-integrity guard for the FASE
        // adjoint lowering, then disarm before the match so it
        // never leaks into a later (forward / free-list) lowering.
        self.grad_live_results = grad_live_set.clone();
        let fase_lowered = crate::wengert_lower::compile_wengert_ops(
            self,
            builder,
            state,
            adjoint,
            full_vars,
            Some((param_adj_set, &mut fase_cb)),
        );
        self.grad_live_results = None;
        let fase_out = match fase_lowered {
            Ok(gv) => Some(gv),
            Err(e) => {
                nsl_runtime::nsl_log!(ERROR, "nsl", 
                    "[nsl] source AD lowering (FASE hook) failed ({}), \
                     rerun without --source-ad",
                    e
                );
                return Err(e);
            }
        };
        if gi {
            self.compile_call_by_name(
                builder,
                "nsl_grad_integrity_step_end",
                &[],
            )?;
        }
        Ok(fase_out)
    }
}
