use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::expr::{Expr, ExprKind, SubscriptKind};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    pub(crate) fn compile_member_access(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        member: nsl_ast::Symbol,
        expr: &Expr,
    ) -> Result<Value, CodegenError> {
        let member_name = self.resolve_sym(member).to_string();

        // Check if this is a module alias access: math.clamp (non-call context)
        {
            let obj_type = self.node_type(object.id).clone();
            if matches!(obj_type, Type::Module { .. }) {
                if let Some((func_id, _sig)) = self.registry.functions.get(&member_name) {
                    let fref = self.module.declare_func_in_func(*func_id, builder.func);
                    return Ok(builder.ins().func_addr(crate::types::pointer_type(), fref));
                }
                return Err(CodegenError::new(format!(
                    "module has no export '{member_name}'"
                )));
            }
        }

        // Check if this is an enum variant access: EnumName.VariantName
        if let ExprKind::Ident(obj_sym) = &object.kind {
            let obj_name = self.resolve_sym(*obj_sym).to_string();
            if self.types.enum_defs.contains_key(&obj_name) {
                if let Some(tag) = self.lookup_qualified_variant(&obj_name, &member_name) {
                    return Ok(builder.ins().iconst(cl_types::I64, tag));
                }
                return Err(CodegenError::new(format!(
                    "enum '{obj_name}' has no variant '{member_name}'"
                )));
            }
        }

        let obj_val = self.compile_expr(builder, state, object)?;
        let obj_type = self.node_type(object.id).clone();

        if let Type::Struct { name, .. } = &obj_type {
            let struct_name = self.resolve_sym(*name).to_string();
            if let Some(layout) = self.types.struct_layouts.get(&struct_name) {
                for field in &layout.fields {
                    if field.name == member_name {
                        let val = builder.ins().load(
                            field.cl_type,
                            MemFlags::trusted(),
                            obj_val,
                            field.offset as i32,
                        );
                        return Ok(val);
                    }
                }
                return Err(CodegenError::new(format!(
                    "struct '{struct_name}' has no field '{member_name}'"
                )));
            }
        }

        if let Type::Model { name, .. } = &obj_type {
            let model_name = self.resolve_sym(*name).to_string();
            if let Some(field_type_map) = self.models.model_field_types.get(&model_name).cloned() {
                if let Some(array_marker) = field_type_map.get(&member_name).cloned() {
                    if array_marker.starts_with('[') {
                        if let Some(layout) = self.types.struct_layouts.get(&model_name).cloned() {
                            for field in &layout.fields {
                                if field.name == member_name {
                                    let offset_val = builder.ins().iconst(cl_types::I64, field.offset as i64);
                                    return Ok(builder.ins().iadd(obj_val, offset_val));
                                }
                            }
                        }
                    }
                }
            }
            if let Some(layout) = self.types.struct_layouts.get(&model_name).cloned() {
                for field in &layout.fields {
                    if field.name == member_name {
                        let val = builder.ins().load(
                            field.cl_type,
                            MemFlags::trusted(),
                            obj_val,
                            field.offset as i32,
                        );
                        if let Some(ref wmap) = self.features.weight_map {
                            let candidates = [
                                format!("{}.{}", model_name, member_name),
                                member_name.clone(),
                            ];
                            for key in &candidates {
                                if wmap.get(key).is_some() {
                                    state.weight_values.insert(val, key.clone());
                                    break;
                                }
                            }
                        }
                        return Ok(val);
                    }
                }
                return Err(CodegenError::new(format!(
                    "model '{model_name}' has no field '{member_name}'"
                )));
            }
        }

        // Tensor property access
        if obj_type.is_tensor() {
            return match member_name.as_str() {
                "shape" => self.compile_call_by_name(builder, "nsl_tensor_shape", &[obj_val]),
                "ndim" => self.compile_call_by_name(builder, "nsl_tensor_ndim", &[obj_val]),
                _ => Err(CodegenError::new(format!("unknown tensor property '.{member_name}'"))),
            };
        }

        // Dict-like member access: obj.field → nsl_dict_get_str(obj, "field")
        // This is the correct path for Dict<String, Tensor> types (e.g., train block batches).
        // Warn if the object type is Unknown — this means type inference failed upstream.
        if matches!(obj_type, Type::Unknown) {
            eprintln!(
                "[nsl-codegen] warning: member access '.{member_name}' on Unknown-typed object — \
                 falling through to dict access. This may be a type inference gap."
            );
        }
        {
            // Ensure the key string is in the string pool
            if !self.string_pool.contains_key(member_name.as_str()) {
                self.intern_string(&member_name)?;
            }
            if let Ok(key_str) = self.compile_string_literal(builder, &member_name) {
                let result = self.compile_call_by_name(builder, "nsl_dict_get_str", &[obj_val, key_str]);
                if let Ok(value) = result {
                    if self.node_type(expr.id).is_tensor() {
                        let cloned = self.compile_call_by_name(builder, "nsl_tensor_clone", &[value])?;
                        state.cleanup.tensor_temporaries.push(cloned);
                        return Ok(cloned);
                    }
                    return Ok(value);
                }
            }
        }

        Err(CodegenError::new(format!("member access not supported: .{member_name}")))
    }

    pub(crate) fn compile_subscript(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        index: &SubscriptKind,
    ) -> Result<Value, CodegenError> {
        let obj_val = self.compile_expr(builder, state, object)?;
        match index {
            SubscriptKind::Index(idx_expr) => {
                let idx_val = self.compile_expr(builder, state, idx_expr)?;
                let obj_type = self.node_type(object.id).clone();
                match &obj_type {
                    Type::Dict(_, value_ty) => {
                        let fid = self.registry.runtime_fns["nsl_dict_get_str"].0;
                        let fref = self.module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[obj_val, idx_val]);
                        let value = builder.inst_results(call)[0];
                        if value_ty.is_tensor() {
                            let cloned = self.compile_call_by_name(builder, "nsl_tensor_clone", &[value])?;
                            state.cleanup.tensor_temporaries.push(cloned);
                            return Ok(cloned);
                        }
                        Ok(value)
                    }
                    _ => {
                        // Default: list subscript. Warn if the type is Unknown.
                        if matches!(obj_type, Type::Unknown) {
                            eprintln!(
                                "[nsl-codegen] warning: subscript on Unknown-typed object — \
                                 defaulting to list access"
                            );
                        }
                        let fid = self.registry.runtime_fns["nsl_list_get"].0;
                        let fref = self.module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[obj_val, idx_val]);
                        Ok(builder.inst_results(call)[0])
                    }
                }
            }
            SubscriptKind::Slice { lower, upper, step } => {
                let sentinel = builder.ins().iconst(cl_types::I64, i64::MIN);
                let lo = if let Some(e) = lower {
                    self.compile_expr(builder, state, e)?
                } else {
                    sentinel
                };
                let hi = if let Some(e) = upper {
                    self.compile_expr(builder, state, e)?
                } else {
                    builder.ins().iconst(cl_types::I64, i64::MIN)
                };
                let st = if let Some(e) = step {
                    self.compile_expr(builder, state, e)?
                } else {
                    builder.ins().iconst(cl_types::I64, i64::MIN)
                };

                let obj_type = self.node_type(object.id).clone();
                let fn_name = if matches!(obj_type, Type::Str) {
                    "nsl_str_slice"
                } else {
                    "nsl_list_slice"
                };
                let fid = self.registry.runtime_fns[fn_name].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[obj_val, lo, hi, st]);
                Ok(builder.inst_results(call)[0])
            }
            _ => Err(CodegenError::new("multi-dim subscript not supported yet")),
        }
    }

    pub(crate) fn compile_type_conversion(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        target_type: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() != 1 {
            return Err(CodegenError::new(format!("{target_type}() takes exactly 1 argument")));
        }
        let val = self.compile_expr(builder, state, &args[0].value)?;
        let src_type = self.node_type(args[0].value.id).clone();

        match target_type {
            "int" => match &src_type {
                Type::Int | Type::Int64 => Ok(val),
                Type::Int32 => Ok(builder.ins().sextend(cl_types::I64, val)),
                Type::Int16 => Ok(builder.ins().sextend(cl_types::I64, val)),
                Type::Int8 => Ok(builder.ins().sextend(cl_types::I64, val)),
                Type::Float | Type::F64 => {
                    Ok(builder.ins().fcvt_to_sint_sat(cl_types::I64, val))
                }
                Type::F32 => {
                    let promoted = builder.ins().fpromote(cl_types::F64, val);
                    Ok(builder.ins().fcvt_to_sint_sat(cl_types::I64, promoted))
                }
                Type::Bool => Ok(builder.ins().uextend(cl_types::I64, val)),
                Type::Str => {
                    let fid = self.registry.runtime_fns["nsl_str_to_int"].0;
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[val]);
                    Ok(builder.inst_results(call)[0])
                }
                _ => Ok(val),
            },
            "float" => match &src_type {
                Type::Float | Type::F64 => Ok(val),
                Type::F32 => Ok(builder.ins().fpromote(cl_types::F64, val)),
                Type::Int | Type::Int64 => {
                    Ok(builder.ins().fcvt_from_sint(cl_types::F64, val))
                }
                Type::Int32 | Type::Int16 | Type::Int8 => {
                    let widened = builder.ins().sextend(cl_types::I64, val);
                    Ok(builder.ins().fcvt_from_sint(cl_types::F64, widened))
                }
                Type::Bool => {
                    let ext = builder.ins().uextend(cl_types::I64, val);
                    Ok(builder.ins().fcvt_from_sint(cl_types::F64, ext))
                }
                Type::Str => {
                    let fid = self.registry.runtime_fns["nsl_str_to_float"].0;
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[val]);
                    Ok(builder.inst_results(call)[0])
                }
                _ => Ok(builder.ins().fcvt_from_sint(cl_types::F64, val)),
            },
            "str" => self.value_to_string(builder, val, &src_type),
            "bool" => match &src_type {
                Type::Bool => Ok(val),
                Type::Int | Type::Int64 | Type::Int32 | Type::Int16 | Type::Int8 => {
                    Ok(builder.ins().icmp_imm(IntCC::NotEqual, val, 0))
                }
                Type::Float | Type::F64 => {
                    let zero = builder.ins().f64const(0.0);
                    Ok(builder.ins().fcmp(FloatCC::NotEqual, val, zero))
                }
                Type::F32 => {
                    let zero = builder.ins().f32const(0.0);
                    Ok(builder.ins().fcmp(FloatCC::NotEqual, val, zero))
                }
                _ => Ok(builder.ins().icmp_imm(IntCC::NotEqual, val, 0)),
            },
            _ => Err(CodegenError::new(format!("unknown type conversion: {target_type}()"))),
        }
    }
}
