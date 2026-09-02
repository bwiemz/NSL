//! Post-loop teardown of the train block: free the parameter, optimizer
//! state and gradient-accumulation lists the block allocated, sweep the
//! trailing CSLA partial window, restore streamed weights to the device
//! and print the CUDA-graphs capture/replay banner.
//!
//! Runs with the epoch loop's exit block current and leaves
//! `state.current_block` on the last block it emits (the runtime free
//! loops each add a header / body / exit triple). The driver
//! (`compile_train_block_inner`) still restores the `FuncState` scopes
//! and clears the per-function caches after this returns — those are not
//! emission and stay with it.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1,
//! PR 3). The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`)
//! pin the list frees and the state-buffer loop on every fixture (one and
//! two state buffers), the accumulator loop on the `grad_accumulation`
//! fixtures, the CSLA shell-only free + tail sweep on the
//! `*_layerwise_accum` / `*_state_bf16` variants and the graphs banner on
//! the `*_cuda_graphs` variants. The ZeRO-3 latch free and the
//! weight-stream teardown have no fixture and rest on the move alone.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

/// The block's runtime lists and the flags that decide how each is freed.
/// Every field is a binding of `compile_train_block_inner`, handed over
/// once the epoch loop is closed; nothing here is read by the driver
/// afterwards.
pub(crate) struct TrainTeardown {
    /// The parameter list (`nsl_list_new` + one push per tensor param).
    pub(crate) param_list: Value,
    /// The parameter count as an `iconst` (`param_paths.len()`, known at
    /// compile time); bounds the state-buffer and accumulator free loops.
    /// The CSLA sweeps are bounded by `nsl_list_len` of their own lists.
    pub(crate) num_params_val: Value,
    /// First optimizer-state list (momentum / first moment).
    pub(crate) state_list_1: Value,
    /// Second optimizer-state list, or `iconst 0` when the optimizer
    /// keeps one buffer per parameter (`num_state_buffers < 2`).
    pub(crate) state_list_2: Value,
    /// 1 or 2: whether `state_list_2` holds tensors.
    pub(crate) num_state_buffers: usize,
    /// Item C: the ZeRO-3 deferred-moment-fill latch (1 slot, no tensors).
    pub(crate) moment_fill_latch: Option<Value>,
    /// Gradient-accumulation buffers (`grad_accumulation > 1`).
    pub(crate) accum_list: Option<Value>,
    /// `--layerwise-accum`: the accumulator slots are NULL between windows,
    /// so only the shell of `accum_list` is freed; with `weight_stream`
    /// also gates the weight-stream + parameter-plan teardown.
    pub(crate) csla_active: bool,
    /// CSLA's buffered-saves and batch-dict list variables.
    pub(crate) csla_buffers: Option<(Variable, Variable)>,
    /// CSLA: `(slot index, free fn)` for every owned buffered slot of a
    /// saves entry — the trailing partial window's sweep list.
    pub(crate) csla_teardown_slots: Option<Vec<(i64, &'static str)>>,
}

/// Emit the teardown. `state.current_block` must be the epoch loop's
/// exit block on entry.
pub(crate) fn emit_train_teardown(
    c: &mut Compiler,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    lists: TrainTeardown,
) -> Result<(), CodegenError> {
    let TrainTeardown {
        param_list,
        num_params_val,
        state_list_1,
        state_list_2,
        num_state_buffers,
        moment_fill_latch,
        accum_list,
        csla_active,
        csla_buffers,
        csla_teardown_slots,
    } = lists;

    // Free param_list after training loop completes
    c.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;

    // Free optimizer state buffers (runtime lists of momentum/velocity tensors)
    // Runtime loop: free each tensor in state_list_1 and state_list_2
    {
        let free_i_var = state.new_variable();
        builder.declare_var(free_i_var, cl_types::I64);
        let f_zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(free_i_var, f_zero);
        let f_header = builder.create_block();
        let f_body = builder.create_block();
        let f_exit = builder.create_block();
        builder.ins().jump(f_header, &[]);
        builder.switch_to_block(f_header);
        let fi = builder.use_var(free_i_var);
        let fc = builder
            .ins()
            .icmp(IntCC::SignedLessThan, fi, num_params_val);
        builder.ins().brif(fc, f_body, &[], f_exit, &[]);
        builder.switch_to_block(f_body);
        builder.seal_block(f_body);
        let buf1 = c.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, fi])?;
        c.compile_call_by_name(builder, "nsl_tensor_free", &[buf1])?;
        if num_state_buffers >= 2 {
            let buf2 =
                c.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, fi])?;
            c.compile_call_by_name(builder, "nsl_tensor_free", &[buf2])?;
        }
        let f_one = builder.ins().iconst(cl_types::I64, 1);
        let f_next = builder.ins().iadd(fi, f_one);
        builder.def_var(free_i_var, f_next);
        builder.ins().jump(f_header, &[]);
        builder.seal_block(f_header);
        builder.switch_to_block(f_exit);
        builder.seal_block(f_exit);
        state.current_block = Some(f_exit);
    }
    c.compile_call_by_name(builder, "nsl_list_free", &[state_list_1])?;
    if num_state_buffers >= 2 {
        c.compile_call_by_name(builder, "nsl_list_free", &[state_list_2])?;
    }
    // Item C: the deferred-fill latch (1 slot, no tensors).
    if let Some(latch) = moment_fill_latch {
        c.compile_call_by_name(builder, "nsl_list_free", &[latch])?;
    }

    // Free gradient accumulation buffers (if allocated) — runtime loop.
    // CSLA (D1b): slots are NULL between windows (each window allocates
    // its accumulators fresh and frees them after its group updates;
    // the partial tail never allocates), so only the shell needs
    // freeing — the per-slot loop would nsl_tensor_free(0).
    if let Some(accum) = accum_list.filter(|_| csla_active) {
        c.compile_call_by_name(builder, "nsl_list_free", &[accum])?;
    } else if let Some(accum) = accum_list {
        let fa_i_var = state.new_variable();
        builder.declare_var(fa_i_var, cl_types::I64);
        let fa_z = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(fa_i_var, fa_z);
        let fa_hdr = builder.create_block();
        let fa_body = builder.create_block();
        let fa_exit = builder.create_block();
        builder.ins().jump(fa_hdr, &[]);
        builder.switch_to_block(fa_hdr);
        let fai = builder.use_var(fa_i_var);
        let fac = builder
            .ins()
            .icmp(IntCC::SignedLessThan, fai, num_params_val);
        builder.ins().brif(fac, fa_body, &[], fa_exit, &[]);
        builder.switch_to_block(fa_body);
        builder.seal_block(fa_body);
        let buf = c.compile_call_by_name(builder, "nsl_list_get", &[accum, fai])?;
        c.compile_call_by_name(builder, "nsl_tensor_free", &[buf])?;
        let fa_one = builder.ins().iconst(cl_types::I64, 1);
        let fa_next = builder.ins().iadd(fai, fa_one);
        builder.def_var(fa_i_var, fa_next);
        builder.ins().jump(fa_hdr, &[]);
        builder.seal_block(fa_hdr);
        builder.switch_to_block(fa_exit);
        builder.seal_block(fa_exit);
        state.current_block = Some(fa_exit);
        c.compile_call_by_name(builder, "nsl_list_free", &[accum])?;
    }

    // CSLA: sweep the trailing partial window. A window that never
    // reached the modulo boundary left its buffered saves + batch dicts
    // alive (the baseline discards the same tail's m_partial content —
    // its gradients never influence θ either way, so parity holds; this
    // sweep is purely against leaks). Every entry here is stale: its
    // iteration's callbacks are long done, so loss slots free
    // unconditionally too.
    if let (Some((saves_outer_var, dicts_var)), Some(sweep)) =
        (csla_buffers, &csla_teardown_slots)
    {
        let so = builder.use_var(saves_outer_var);
        let tail_len = c.compile_call_by_name(builder, "nsl_list_len", &[so])?;
        let sw_i_var = state.new_variable();
        builder.declare_var(sw_i_var, cl_types::I64);
        let sw_z = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(sw_i_var, sw_z);
        let sw_hdr = builder.create_block();
        let sw_body = builder.create_block();
        let sw_exit = builder.create_block();
        builder.ins().jump(sw_hdr, &[]);
        builder.switch_to_block(sw_hdr);
        let swi = builder.use_var(sw_i_var);
        let swc = builder.ins().icmp(IntCC::SignedLessThan, swi, tail_len);
        builder.ins().brif(swc, sw_body, &[], sw_exit, &[]);
        builder.switch_to_block(sw_body);
        builder.seal_block(sw_body);
        let inner = c.compile_call_by_name(builder, "nsl_list_get", &[so, swi])?;
        for (idx, free_fn) in sweep {
            let idx_val = builder.ins().iconst(cl_types::I64, *idx);
            let slot_val =
                c.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
            c.compile_call_by_name(builder, free_fn, &[slot_val])?;
        }
        c.compile_call_by_name(builder, "nsl_list_free", &[inner])?;
        let sw_one = builder.ins().iconst(cl_types::I64, 1);
        let sw_next = builder.ins().iadd(swi, sw_one);
        builder.def_var(sw_i_var, sw_next);
        builder.ins().jump(sw_hdr, &[]);
        builder.seal_block(sw_hdr);
        builder.switch_to_block(sw_exit);
        builder.seal_block(sw_exit);
        state.current_block = Some(sw_exit);
        c.compile_call_by_name(builder, "nsl_list_free", &[so])?;

        // Tail batch dicts: nsl_dict_free_tensor_values destroys the
        // whole dict structure (values + shell, free_dict_impl(_, true))
        // — the same call the baseline makes per iteration; the
        // DataLoader teardown never touches popped dicts.
        let dl = builder.use_var(dicts_var);
        let dl_len = c.compile_call_by_name(builder, "nsl_list_len", &[dl])?;
        let dw_i_var = state.new_variable();
        builder.declare_var(dw_i_var, cl_types::I64);
        let dw_z = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(dw_i_var, dw_z);
        let dw_hdr = builder.create_block();
        let dw_body = builder.create_block();
        let dw_exit = builder.create_block();
        builder.ins().jump(dw_hdr, &[]);
        builder.switch_to_block(dw_hdr);
        let dwi = builder.use_var(dw_i_var);
        let dwc = builder.ins().icmp(IntCC::SignedLessThan, dwi, dl_len);
        builder.ins().brif(dwc, dw_body, &[], dw_exit, &[]);
        builder.switch_to_block(dw_body);
        builder.seal_block(dw_body);
        let tail_dict = c.compile_call_by_name(builder, "nsl_list_get", &[dl, dwi])?;
        c.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[tail_dict])?;
        let dw_one = builder.ins().iconst(cl_types::I64, 1);
        let dw_next = builder.ins().iadd(dwi, dw_one);
        builder.def_var(dw_i_var, dw_next);
        builder.ins().jump(dw_hdr, &[]);
        builder.seal_block(dw_hdr);
        builder.switch_to_block(dw_exit);
        builder.seal_block(dw_exit);
        state.current_block = Some(dw_exit);
        c.compile_call_by_name(builder, "nsl_list_free", &[dl])?;
    }

    // D2b part 2: LOAD-BEARING — under whole-loop streaming every
    // streamed param exits the training loop EVICTED (the forward
    // evicts after its last primal touch each micro-batch and there is
    // no post-epilogue restore), so this teardown is the SOLE restore
    // of device residency for model_save/eval, plus the pinned-mirror
    // release. Removing or reordering it after any θ reader crashes on
    // null data pointers.
    if csla_active && c.compile_options.weight_stream {
        c.compile_call_by_name(builder, "nsl_weight_stream_teardown", &[])?;
        // Item 3: drop this block's declared plan in the same breath.
        // The residency tables are cleared above, so leaving the plan
        // behind would make a SECOND train block in the same program
        // verify its predecessor's (now unregistered, possibly freed)
        // pointers and abort on a mismatch that is really just staleness.
        c.compile_call_by_name(builder, "nsl_param_plan_teardown", &[])?;
    }

    // P5 item 19: capture/replay counter banner (anti-vacuity evidence
    // for the gates; harmless no-op when the runtime declined to arm).
    if c.compile_options.cuda_graphs {
        c.compile_call_by_name(builder, "nsl_cuda_graphs_report", &[])?;
    }

    Ok(())
}
