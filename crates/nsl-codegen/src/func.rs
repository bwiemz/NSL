use cranelift_codegen::ir::{types as cl_types, Function, InstBuilder, UserFuncName};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::Module;

use nsl_ast::decl::FnDef;
use nsl_ast::types::{DimExpr as AstDimExpr, TypeExprKind};

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    pub fn compile_fn_def(&mut self, fn_def: &FnDef) -> Result<(), CodegenError> {
        self.compile_fn_def_named(fn_def, None)
    }

    pub fn compile_fn_def_named(
        &mut self,
        fn_def: &FnDef,
        name_override: Option<&str>,
    ) -> Result<(), CodegenError> {
        let name = if let Some(override_name) = name_override {
            override_name.to_string()
        } else {
            self.resolve_sym(fn_def.name).to_string()
        };
        let (func_id, sig) = self.registry.functions[&name].clone();

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

            // M28: emit runtime assertions for symbolic/bounded dimensions
            for (i, param) in fn_def.params.iter().enumerate() {
                let shape = match param.type_ann.as_ref().map(|t| &t.kind) {
                    Some(TypeExprKind::Tensor { shape, .. })
                    | Some(TypeExprKind::Param { shape, .. })
                    | Some(TypeExprKind::Buffer { shape, .. })
                    | Some(TypeExprKind::Sparse { shape, .. }) => shape,
                    _ => continue,
                };
                let param_val = builder.block_params(entry)[i];
                for (dim_idx, dim_expr) in shape.iter().enumerate() {
                    let dim_idx_val = builder.ins().iconst(cl_types::I64, dim_idx as i64);
                    match dim_expr {
                        AstDimExpr::Symbolic(sym) => {
                            if !state.symbolic_dims.is_resolved(sym) {
                                let neg1 = builder.ins().iconst(cl_types::I64, -1);
                                let actual = self.compile_call_by_name(
                                    &mut builder,
                                    "nsl_tensor_assert_dim",
                                    &[param_val, dim_idx_val, neg1],
                                )?;
                                state.symbolic_dims.resolve(*sym, actual);
                            } else {
                                let expected = state.symbolic_dims.get(sym).unwrap();
                                self.compile_call_by_name(
                                    &mut builder,
                                    "nsl_tensor_assert_dim",
                                    &[param_val, dim_idx_val, expected],
                                )?;
                            }
                        }
                        AstDimExpr::Bounded { name, upper_bound } => {
                            if !state.symbolic_dims.is_resolved(name) {
                                let neg1 = builder.ins().iconst(cl_types::I64, -1);
                                let actual = self.compile_call_by_name(
                                    &mut builder,
                                    "nsl_tensor_assert_dim",
                                    &[param_val, dim_idx_val, neg1],
                                )?;
                                state.symbolic_dims.resolve(*name, actual);
                            } else {
                                let expected = state.symbolic_dims.get(name).unwrap();
                                self.compile_call_by_name(
                                    &mut builder,
                                    "nsl_tensor_assert_dim",
                                    &[param_val, dim_idx_val, expected],
                                )?;
                            }
                            let ub_val = builder.ins().iconst(cl_types::I64, *upper_bound);
                            self.compile_call_by_name(
                                &mut builder,
                                "nsl_tensor_assert_dim_bound",
                                &[param_val, dim_idx_val, ub_val],
                            )?;
                        }
                        _ => {}
                    }
                }
            }

            // @no_grad: pause tape recording at function entry
            let is_no_grad = self.registry.no_grad_fns.contains(&name);
            if is_no_grad {
                state.flags.is_no_grad = true;
                self.compile_call_by_name(&mut builder, "nsl_tape_pause", &[])?;
            }

            // @fp8_compute: use FP8 training matmul for E5M2 backward
            if self.features.fp8_compute_fns.contains(&name) {
                state.flags.is_fp8_compute = true;
            }

            // M44: Record current function name for @grammar decorator lookup in generate() calls
            state.current_function_name = Some(name.clone());

            // FBIP Phase 2: Pre-compute use counts for single-use optimizations
            // (clone elision, in-place op emission).
            state.use_counts = Some(crate::use_count::analyze_use_counts(&fn_def.body));

            // M38b: Set up ownership lowering for this function.
            // When --linear-types is active and the semantic pass produced ownership
            // metadata for this function, create an OwnershipLowering tracker and
            // install it in the function state. During expr codegen, variable use
            // compilation can consult `state.ownership.lowering` to decide whether
            // to emit `nsl_tensor_free` at consumption point (linear bindings) or
            // skip refcount ops entirely.
            if self.features.linear_types_enabled {
                if let Some(fn_ownership) = self.features.ownership_info.get(&name) {
                    let mut lowering = crate::ownership::OwnershipLowering::new();
                    for sym in &fn_ownership.linear_params {
                        lowering.mark_linear(*sym);
                    }
                    for sym in &fn_ownership.shared_params {
                        lowering.mark_shared(*sym);
                    }
                    state.ownership.lowering = Some(lowering);
                }
            }

            for stmt in &fn_def.body.stmts {
                self.compile_stmt(&mut builder, &mut state, stmt)?;
            }

            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                // Free intermediate tensor temporaries before implicit return
                self.free_tensor_temporaries(&mut builder, &mut state, None);
                // M38b: Free linear tensors consumed during the function body
                self.free_linear_consumes(&mut builder, &mut state, None);
                // Free all tensor-typed local variables (let-bound tensors that would
                // otherwise leak on implicit return — explicit return handles this in
                // stmt.rs via free_tensor_temporaries, but locals aren't temporaries).
                {
                    let param_syms: std::collections::HashSet<nsl_ast::Symbol> =
                        fn_def.params.iter().map(|p| p.name).collect();
                    let locals: Vec<_> = state.variables.iter()
                        .filter(|(sym, _)| !param_syms.contains(sym))
                        .filter_map(|(sym, (var, _))| {
                            let is_tensor = state.variable_types.get(sym)
                                .map(|t| t.is_tensor())
                                .unwrap_or(false);
                            if is_tensor { Some(*var) } else { None }
                        })
                        .collect();
                    for var in locals {
                        let val = builder.use_var(var);
                        let _ = self.compile_call_by_name(&mut builder, "nsl_tensor_free", &[val]);
                    }
                }
                // Stop and free any DataLoaders before implicit function return
                self.teardown_dataloaders(&mut builder, &mut state);
                // @no_grad: resume tape before implicit return
                if is_no_grad {
                    self.compile_call_by_name(&mut builder, "nsl_tape_resume", &[])?;
                }
                if sig.returns.is_empty() {
                    builder.ins().return_(&[]);
                } else {
                    // The function has a return type but control flow reached the
                    // end without an explicit `return`.  For scalar types (F64/F32/
                    // integers) returning zero is safe.  For pointer-sized types
                    // (I64) this would produce a null pointer — emit a trap instead
                    // so we get a clear runtime abort rather than a segfault.
                    let ret_type = sig.returns[0].value_type;
                    if ret_type == cl_types::F64 {
                        let zero = builder.ins().f64const(0.0);
                        builder.ins().return_(&[zero]);
                    } else if ret_type == cl_types::F32 {
                        let zero = builder.ins().f32const(0.0);
                        builder.ins().return_(&[zero]);
                    } else if ret_type == cl_types::I64 {
                        // I64 is used for tensor/list/string pointers — returning 0
                        // would be a null pointer.  Trap instead.
                        builder
                            .ins()
                            .trap(cranelift_codegen::ir::TrapCode::unwrap_user(2));
                    } else {
                        let zero = builder.ins().iconst(ret_type, 0);
                        builder.ins().return_(&[zero]);
                    }
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
