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
use crate::ownership_expr::Ownership;

impl Compiler<'_> {
    pub(crate) fn compile_member_access(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        member: nsl_ast::Symbol,
        expr: &Expr,
    ) -> Result<Value, CodegenError> {
        // B.2.1 Task 3: the WRGA LoRA AST rewrite synthesizes MemberAccess
        // nodes whose field names (`lora_A_<site>`, `lora_B_<site>`) may not
        // be present in the Interner. Prefer the synth override map.
        let member_name = match self.synth_member_names.get(&expr.id) {
            Some(name) => name.clone(),
            None => self.resolve_sym(member).to_string(),
        };

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
                        // ELTLS: non-weight struct field accesses stay
                        // Unknown for Phase 1 — plumbing the receiver
                        // variable's Cranelift Variable to build a
                        // BorrowedFromVar entry is out of scope for this
                        // task. Consumers reading expr_ownership will hit
                        // the Unknown fallback and bump the counter.
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
                                    let offset_val =
                                        builder.ins().iconst(cl_types::I64, field.offset as i64);
                                    return Ok(builder.ins().iadd(obj_val, offset_val));
                                }
                            }
                        }
                    }
                }
            }
            // B.2.1: Route synthesized adapter field accesses through the
            // per-model side-table. Names of the form `lora_A_*`, `lora_B_*`,
            // `ia3_scale_*`, `gate_*` are NOT physically laid out as struct
            // fields — they live in a heap table whose pointer is stored at
            // `layout.adapter_sidetable_offset`. The index used here must
            // match the insertion order used by the init pass, i.e. the
            // flattened `synthesized_fields` sequence across all sites
            // targeting this model in `last_wrga_plan`.
            if is_synthesized_adapter_field_name(&member_name) {
                if let Some(layout) = self.types.struct_layouts.get(&model_name).cloned() {
                    if let Some(slot_off) = layout.adapter_sidetable_offset {
                        let index = self
                            .adapter_field_index(&model_name, &member_name)
                            .ok_or_else(|| {
                                CodegenError::new(format!(
                                    "synthesized adapter field '{member_name}' not found \
                                     for model '{model_name}' in current WRGA plan"
                                ))
                            })?;
                        let table_ptr = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            obj_val,
                            slot_off as i32,
                        );
                        let byte_off = (index * 8) as i32;
                        let tensor_ptr = builder.ins().load(
                            cl_types::I64,
                            MemFlags::trusted(),
                            table_ptr,
                            byte_off,
                        );
                        return Ok(tensor_ptr);
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
                        // ELTLS: register model weight field accesses as
                        // BorrowedWeight. state.weight_values is populated
                        // just above when the loaded field matches a known
                        // model weight.
                        if state.weight_values.contains_key(&val) {
                            self.set_ownership(
                                builder,
                                state,
                                val,
                                Ownership::BorrowedWeight,
                            );
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
                _ => Err(CodegenError::new(format!(
                    "unknown tensor property '.{member_name}'"
                ))),
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
                let result =
                    self.compile_call_by_name(builder, "nsl_dict_get_str", &[obj_val, key_str]);
                if let Ok(value) = result {
                    if self.node_type(expr.id).is_tensor() {
                        let cloned =
                            self.compile_call_by_name(builder, "nsl_tensor_clone", &[value])?;
                        state.cleanup.tensor_temporaries.push(cloned);
                        // ELTLS: nsl_tensor_clone returns a fresh owned tensor.
                        self.set_ownership(builder, state, cloned, Ownership::Owned);
                        return Ok(cloned);
                    }
                    return Ok(value);
                }
            }
        }

        Err(CodegenError::new(format!(
            "member access not supported: .{member_name}"
        )))
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
                            let cloned =
                                self.compile_call_by_name(builder, "nsl_tensor_clone", &[value])?;
                            state.cleanup.tensor_temporaries.push(cloned);
                            // ELTLS: subscript-via-dict clones the stored
                            // tensor, producing a fresh owned value.
                            self.set_ownership(builder, state, cloned, Ownership::Owned);
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
            return Err(CodegenError::new(format!(
                "{target_type}() takes exactly 1 argument"
            )));
        }
        let val = self.compile_expr(builder, state, &args[0].value)?;
        let src_type = self.node_type(args[0].value.id).clone();

        match target_type {
            "int" => match &src_type {
                Type::Int | Type::Int64 => Ok(val),
                Type::Int32 => Ok(builder.ins().sextend(cl_types::I64, val)),
                Type::Int16 => Ok(builder.ins().sextend(cl_types::I64, val)),
                Type::Int8 => Ok(builder.ins().sextend(cl_types::I64, val)),
                Type::Float | Type::F64 => Ok(builder.ins().fcvt_to_sint_sat(cl_types::I64, val)),
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
                Type::Int | Type::Int64 => Ok(builder.ins().fcvt_from_sint(cl_types::F64, val)),
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
            _ => Err(CodegenError::new(format!(
                "unknown type conversion: {target_type}()"
            ))),
        }
    }
}

/// B.2.1: recognise synthesized adapter field names. These are never
/// physically laid out in the model struct — they are routed through the
/// side-table pointer stored at `StructLayout::adapter_sidetable_offset`.
pub(crate) fn is_synthesized_adapter_field_name(name: &str) -> bool {
    name.starts_with("lora_A_")
        || name.starts_with("lora_B_")
        || name.starts_with("ia3_scale_")
        || name.starts_with("gate_")
}

impl Compiler<'_> {
    /// B.2.1: linear index of a synthesized adapter field within the
    /// flattened `synthesized_fields` sequence across every active adapter
    /// site targeting `model_name`, in insertion order as produced by
    /// `wrga_adapter_inject::run`.
    ///
    /// **Ordering invariant:** this MUST match the iteration order used by
    /// the future init pass that populates the side-table, otherwise an
    /// access to `self.lora_A_<site>` would dereference the wrong tensor.
    /// The insertion order is: outer loop over `plan.placements` in the
    /// order produced by WRGA; inner loop over each placement's
    /// `synthesized_fields` vec. Only placements whose target field is
    /// declared on `model_name` participate. Sites whose dims failed to
    /// resolve (input_dim == 0 || output_dim == 0) are EXCLUDED from both
    /// the init pass and this index computation to keep them in sync.
    pub(crate) fn adapter_field_index(
        &self,
        model_name: &str,
        field_name: &str,
    ) -> Option<usize> {
        let plan = self.last_wrga_plan.as_ref()?;
        let field_types = &self.models.model_field_types;
        let mut idx: usize = 0;
        for placement in &plan.placements {
            // Skip placements without a decorator (the inject pass only
            // synthesizes fields when `decorator_kind` is set).
            if placement.decorator_kind.is_none() {
                continue;
            }
            // Only count placements whose target field exists on this model.
            let target_field = placement.name.rsplit('.').next().unwrap_or("");
            let matches_model = field_types
                .get(model_name)
                .map(|m| m.contains_key(target_field))
                .unwrap_or(false);
            if !matches_model {
                continue;
            }
            for synth in &placement.synthesized_fields {
                if synth == field_name {
                    return Some(idx);
                }
                idx += 1;
            }
        }
        None
    }
}
