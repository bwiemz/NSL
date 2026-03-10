use cranelift_codegen::ir::{types as cl_types, Function, InstBuilder, UserFuncName};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::Module;

use nsl_ast::decl::FnDef;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    pub fn compile_fn_def(&mut self, fn_def: &FnDef) -> Result<(), CodegenError> {
        let name = self.resolve_sym(fn_def.name).to_string();
        let (func_id, sig) = self.functions[&name].clone();

        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig.clone(),
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let mut state = FuncState::new();

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            for (i, param) in fn_def.params.iter().enumerate() {
                let param_val = builder.block_params(entry)[i];
                let cl_type = if i < sig.params.len() {
                    sig.params[i].value_type
                } else {
                    cl_types::I64
                };
                let var = state.new_variable();
                builder.declare_var(var, cl_type);
                builder.def_var(var, param_val);
                state.variables.insert(param.name, (var, cl_type));
            }

            // @no_grad: pause tape recording at function entry
            let is_no_grad = self.no_grad_fns.contains(&name);
            if is_no_grad {
                state.is_no_grad = true;
                self.compile_call_by_name(&mut builder, "nsl_tape_pause", &[])?;
            }

            for stmt in &fn_def.body.stmts {
                self.compile_stmt(&mut builder, &mut state, stmt)?;
            }

            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                // @no_grad: resume tape before implicit return
                if is_no_grad {
                    self.compile_call_by_name(&mut builder, "nsl_tape_resume", &[])?;
                }
                if sig.returns.is_empty() {
                    builder.ins().return_(&[]);
                } else {
                    let ret_type = sig.returns[0].value_type;
                    let zero = if ret_type.is_float() {
                        builder.ins().f64const(0.0)
                    } else {
                        builder.ins().iconst(ret_type, 0)
                    };
                    builder.ins().return_(&[zero]);
                }
            }

            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: fn '{}' ---\n{}", name, ctx.func.display());
        }

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define fn '{name}': {e}")))?;
        Ok(())
    }
}
