//! The train block's epoch close: seal the batch loop (or stay in the
//! single-step block), bind and run the `on_epoch` / `on_epoch_end`
//! callbacks, free the epoch-loss alias, increment the epoch counter and
//! jump back to the epoch header; leaves `state.current_block` on the
//! loop's exit block, where `teardown` starts.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1):
//! 129 lines, no escaping binding, no `self` write; the twelve values it
//! reads are named in [`EpochClose`] the way [`super::teardown::TrainTeardown`]
//! names the teardown's. The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the block structure on every
//! fixture and the callback bindings on the `on_epoch` fixtures; the
//! `loss`-alias non-owning mark is pinned by the callback-lifetime gates
//! in `nsl-cli` (#543, #544).

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{Block, InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::is_block_filled;

/// The loop blocks and variables the epoch close reads. Every field is a
/// binding of `compile_train_block_inner`.
pub(crate) struct EpochClose {
    /// The batch loop's header / body / exit blocks (`has_dataloader`
    /// decides whether the header is sealed here with its back-edge).
    pub(crate) batch_header_block: Block,
    pub(crate) batch_body_block: Block,
    pub(crate) batch_exit_block: Block,
    /// The DataLoader binding the batch loop iterates; `None` = one step
    /// per epoch, no batch loop to close.
    pub(crate) has_dataloader: Option<Value>,
    /// The epoch loop's header / increment / exit blocks.
    pub(crate) header_block: Block,
    pub(crate) increment_block: Block,
    pub(crate) exit_block: Block,
    pub(crate) epoch_counter_var: Variable,
    /// The last step's loss tensor, held for an `on_epoch` callback that
    /// binds `loss`; freed here after the callbacks ran.
    pub(crate) epoch_loss_var: Variable,
    pub(crate) on_epoch_binds_loss: bool,
    /// The model symbol, for the callbacks' streamed-residency guard.
    pub(crate) model_sym: nsl_ast::Symbol,
}

pub(crate) fn emit_epoch_close(
    c: &mut Compiler,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    callbacks: &[&nsl_ast::block::CallbackDef],
    close: EpochClose,
) -> Result<(), CodegenError> {
    let EpochClose {
        batch_header_block,
        batch_body_block,
        batch_exit_block,
        has_dataloader,
        header_block,
        increment_block,
        exit_block,
        epoch_counter_var,
        epoch_loss_var,
        on_epoch_binds_loss,
        model_sym,
    } = close;
    let current = state.current_block.unwrap_or(batch_body_block);
    if has_dataloader.is_some() {
        // Jump back to batch_header for next batch
        if !is_block_filled(builder, current) {
            builder.ins().jump(batch_header_block, &[]);
        }
        // Now seal batch_header — both predecessors connected (entry + back-edge)
        builder.seal_block(batch_header_block);
        // batch_exit: all batches done → run epoch callbacks then increment
        builder.switch_to_block(batch_exit_block);
        builder.seal_block(batch_exit_block);
        state.current_block = Some(batch_exit_block);
    } else {
        // No DataLoader — single step per epoch, stay in the current block for epoch callbacks
        state.current_block = Some(current);
    }

    let mut epoch_loss_alias_sym: Option<nsl_ast::Symbol> = None;
    for cb in callbacks {
        let cb_name = c.resolve_sym(cb.name).to_string();
        if cb_name == "on_epoch" || cb_name == "on_epoch_end" {
            for param in &cb.params {
                let pname = c.resolve_sym(param.name).to_string();
                match pname.as_str() {
                    "epoch" => {
                        let var = builder.declare_var(cl_types::I64);
                        let epoch_val = builder.use_var(epoch_counter_var);
                        builder.def_var(var, epoch_val);
                        state.variables.insert(param.name, (var, cl_types::I64));
                        state.param_symbols.insert(param.name);
                        // Same invariant as on_step's `step` arm: an untyped
                        // slot reads as indeterminate to any future
                        // tensor-cleanup sweep over state.variables, which
                        // would hand this raw counter to
                        // nsl_tensor_free_if_valid. No such sweep runs over
                        // epoch scope today, but the binding shouldn't rely
                        // on that staying true.
                        state.variable_types.insert(param.name, Type::Int);
                    }
                    "loss" => {
                        let var = builder.declare_var(cl_types::I64);
                        let epoch_loss = builder.use_var(epoch_loss_var);
                        builder.def_var(var, epoch_loss);
                        state.variables.insert(param.name, (var, cl_types::I64));
                        state.param_symbols.insert(param.name);
                        // Unlike `epoch`, this one really is a tensor
                        // pointer: it aliases the exact value in
                        // epoch_loss_var, which this function frees
                        // explicitly (nsl_tensor_free_if_valid) right
                        // after the callback body below runs. Typing it
                        // Int like `epoch` would misrepresent it; leaving
                        // it untyped makes it read as "indeterminate" to
                        // the step-body sweep's own filter a few hundred
                        // lines up (`is_tensor || is_unknown`) — and a
                        // future epoch-scope sweep modeled on that one
                        // would free this alias too, double-freeing the
                        // same pointer. non_owning_symbols is the flag
                        // that sweep already checks (see its
                        // `!state.non_owning_symbols.contains(sym)`
                        // filter) to skip exactly this kind of borrowed
                        // alias, so mark it and clear the mark once this
                        // callback's body is done using it.
                        state.non_owning_symbols.insert(param.name);
                        epoch_loss_alias_sym = Some(param.name);
                    }
                    _ => {
                        let var = builder.declare_var(cl_types::I64);
                        let z = builder.ins().iconst(cl_types::I64, 0);
                        builder.def_var(var, z);
                        state.variables.insert(param.name, (var, cl_types::I64));
                        state.param_symbols.insert(param.name);
                        // Typed for the same reason as `epoch` above.
                        state.variable_types.insert(param.name, Type::Int);
                    }
                }
            }
            // Item 12: same scoped-residency guard as on_step — an
            // on_epoch callback that logs / saves model state runs with
            // every streamed param evicted.
            let ws_guard =
                c.emit_callback_residency_open(builder, &cb.body, model_sym, &cb_name)?;
            for stmt in &cb.body.stmts {
                c.compile_stmt(builder, state, stmt)?;
            }
            c.emit_callback_residency_close(builder, ws_guard)?;
            // Scope the non_owning mark to this callback's body: it isn't
            // covered by the saved_variables/saved_variable_types restore
            // at the end of this train block, and `loss` is common enough
            // as a symbol name elsewhere that leaving the mark set would
            // wrongly suppress freeing unrelated tensors bound to it later.
            if let Some(sym) = epoch_loss_alias_sym.take() {
                state.non_owning_symbols.remove(&sym);
            }
        }
    }

    if on_epoch_binds_loss {
        let saved_loss = builder.use_var(epoch_loss_var);
        c.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[saved_loss])?;
        let epoch_loss_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(epoch_loss_var, epoch_loss_null);
    }

    let epoch_callback_block = state.current_block.unwrap_or(current);
    if !is_block_filled(builder, epoch_callback_block) {
        builder.ins().jump(increment_block, &[]);
    }

    builder.switch_to_block(increment_block);
    builder.seal_block(increment_block);
    state.current_block = Some(increment_block);
    let counter = builder.use_var(epoch_counter_var);
    let one = builder.ins().iconst(cl_types::I64, 1);
    let next = builder.ins().iadd(counter, one);
    builder.def_var(epoch_counter_var, next);
    builder.ins().jump(header_block, &[]);

    builder.seal_block(header_block);
    builder.switch_to_block(exit_block);
    builder.seal_block(exit_block);
    state.current_block = Some(exit_block);
    Ok(())
}
