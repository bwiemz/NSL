//! The `quant` block lowering: `compile_quant_block` (the per-model
//! quantisation call sequence — dtype and granularity selection and the
//! walk over the model's fields and sub-models).
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1); the statement
//! dispatch (`stmt.rs::compile_stmt_dispatch`) still calls
//! `compile_quant_block`.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlagsData};
use cranelift_frontend::FunctionBuilder;
use nsl_ast::block::{QuantDtype, QuantGranularity};

use crate::compiler::Compiler;
use crate::context::{FieldKind, FuncState};
use crate::error::CodegenError;

use crate::stmt::glob_match;
use cranelift_codegen::ir::Value;

/// The runtime quantizer's arguments, as constants of the function being
/// built: the same four values go to every tensor field of a quant block.
struct QuantCodes {
    dtype: Value,
    granularity: Value,
    axis: Value,
    group_size: Value,
}

impl Compiler<'_> {
    // ── Quant block codegen ──────────────────────────────────────────

    pub(crate) fn compile_quant_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        quant: &nsl_ast::block::QuantBlock,
    ) -> Result<(), CodegenError> {
        // 1. Get source model variable
        let source_sym = quant.source;
        let source_val = {
            let (var, _) = state.variables.get(&source_sym).ok_or_else(|| {
                CodegenError::new(format!(
                    "undefined model variable '{}' in quant block",
                    self.resolve_sym(source_sym)
                ))
            })?;
            builder.use_var(*var)
        };

        // 2. The model type is the SOURCE VARIABLE's type, resolved the way a
        //    train block resolves its model. This used to scan the whole type
        //    map for any model with a layout and take the first hit; the map
        //    is a hash map, so a program with two model types (a model and
        //    its sub-model) got one or the other from run to run, and the
        //    wrong one walked the outer model's memory with the inner model's
        //    layout.
        let model_type_name = self
            .models
            .model_var_types
            .get(&source_sym)
            .cloned()
            .or_else(|| {
                state.variable_types.get(&source_sym).and_then(|ty| match ty {
                    nsl_semantic::types::Type::Model { name, .. }
                    | nsl_semantic::types::Type::Struct { name, .. } => {
                        Some(self.resolve_sym(*name).to_string())
                    }
                    _ => None,
                })
            })
            .unwrap_or_else(|| self.resolve_sym(source_sym).to_string());

        let layout = self
            .types
            .struct_layouts
            .get(&model_type_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "quant block: '{}' is not a model whose layout is known (resolved type '{}')",
                    self.resolve_sym(source_sym),
                    model_type_name
                ))
            })?;

        // 3. Compute dtype/granularity integer codes for the runtime call.
        //    The checker refuses awq4/gptq4/gptq8 (`quant_requests.rs`); a
        //    caller that skipped the checker gets the same answer here
        //    instead of a dtype code the runtime quantizer aborts on.
        let dtype_code: i64 = match quant.default_dtype {
            Some(QuantDtype::Int4) => 1,
            Some(QuantDtype::Int8) | None => 0,
            Some(QuantDtype::Awq4) | Some(QuantDtype::Gptq4) | Some(QuantDtype::Gptq8) => {
                return Err(CodegenError::new(
                    "quant blocks implement only `int8` and `int4`: `awq4`, `gptq4` and `gptq8` \
                     are not implemented"
                        .to_string(),
                ));
            }
        };
        let (gran_code, axis_val, gs_val): (i64, i64, i64) = match &quant.default_granularity {
            Some(QuantGranularity::PerChannel(a)) => (1, *a, 0),
            Some(QuantGranularity::PerGroup(a, gs)) => (2, *a, *gs),
            Some(QuantGranularity::PerTensor) | None => (0, 0, 0),
        };

        let codes = QuantCodes {
            dtype: builder.ins().iconst(cl_types::I64, dtype_code),
            granularity: builder.ins().iconst(cl_types::I64, gran_code),
            axis: builder.ins().iconst(cl_types::I64, axis_val),
            group_size: builder.ins().iconst(cl_types::I64, gs_val),
        };

        // 4. Build the quantized copy, descending into sub-models.
        let new_ptr = self.emit_quantized_model_copy(
            builder,
            &model_type_name,
            source_val,
            "",
            &quant.exclude,
            &codes,
            0,
        )?;

        // 5. Register the quantized model with the same struct layout and methods
        //    so that forward dispatch works identically to the source model.
        let quant_name = self.resolve_sym(quant.name).to_string();
        if !self.types.struct_layouts.contains_key(&quant_name) {
            self.types.struct_layouts.insert(quant_name.clone(), layout);
        }
        if let Some(field_types) = self.models.model_field_types.get(&model_type_name).cloned() {
            self.models
                .model_field_types
                .entry(quant_name.clone())
                .or_insert(field_types);
        }
        if let Some(methods) = self.models.model_methods.get(&model_type_name).cloned() {
            self.models.model_methods.insert(quant_name, methods);
        }

        // 6. Bind the new struct pointer as the output variable
        let var = builder.declare_var(cl_types::I64);
        builder.def_var(var, new_ptr);
        state.variables.insert(quant.name, (var, cl_types::I64));

        Ok(())
    }

    /// Emit a copy of the model at `src_ptr` (of type `model_name`) in which
    /// every tensor field has been quantized and dequantized, and return the
    /// pointer to the copy.
    ///
    /// The walk follows the layout's field kinds. It used to treat every
    /// field as a tensor: a sub-model's pointer went to the quantizer as a
    /// tensor handle ("invalid tensor handle"), and an `int` field did too.
    ///
    /// - A tensor or buffer field is quantized, unless an `exclude:` glob
    ///   matches its dotted path from the block's source model (`w`,
    ///   `inner.w`, `blocks.0.w`) or its own name; then it is cloned. The
    ///   result is moved back to the device the source tensor is on.
    /// - A sub-model field is copied by recursion, and each element of a
    ///   `[Model; N]` field likewise. The sub-model's methods are compiled
    ///   against its own type, so they run on the copy unchanged.
    /// - Every other slot (an `int`, a `str`, the adapter side-table) is
    ///   copied as it is.
    /// - A nested field that is neither (a struct, a list of models) is a
    ///   compile error naming the field: nothing here knows its layout.
    #[allow(clippy::too_many_arguments)]
    fn emit_quantized_model_copy(
        &mut self,
        builder: &mut FunctionBuilder,
        model_name: &str,
        src_ptr: Value,
        path_prefix: &str,
        exclude: &[String],
        codes: &QuantCodes,
        depth: usize,
    ) -> Result<Value, CodegenError> {
        // A model cannot contain itself by value, so this only guards a
        // malformed layout table from recursing without end.
        if depth > 64 {
            return Err(CodegenError::new(format!(
                "quant block: model '{model_name}' nests more than 64 levels deep"
            )));
        }
        let layout = self.types.struct_layouts.get(model_name).cloned().ok_or_else(|| {
            CodegenError::new(format!("quant block: no struct layout for model '{model_name}'"))
        })?;
        let field_types = self.models.model_field_types.get(model_name).cloned();
        let flags = MemFlagsData::trusted();

        let alloc_size = builder
            .ins()
            .iconst(cl_types::I64, layout.total_size.max(8) as i64);
        let new_ptr = self.compile_call_by_name(builder, "nsl_alloc", &[alloc_size])?;

        // Start from a slot-for-slot copy, so that whatever the loop below
        // does not rebuild (scalars, strings, the adapter side-table slot)
        // holds the source's value rather than uninitialised memory.
        let whole_slots = layout.total_size / 8;
        for slot in 0..whole_slots {
            let off = (slot * 8) as i32;
            let v = builder.ins().load(cl_types::I64, flags, src_ptr, off);
            builder.ins().store(flags, v, new_ptr, off);
        }
        // A layout that ends in a field narrower than a slot (a `bool`).
        for byte in (whole_slots * 8)..layout.total_size {
            let v = builder.ins().load(cl_types::I8, flags, src_ptr, byte as i32);
            builder.ins().store(flags, v, new_ptr, byte as i32);
        }

        for field in &layout.fields {
            let path = if path_prefix.is_empty() {
                field.name.clone()
            } else {
                format!("{path_prefix}.{}", field.name)
            };
            match field.kind {
                FieldKind::Scalar => {}
                FieldKind::Tensor | FieldKind::Buffer => {
                    let src_val = builder.ins().load(field.cl_type, flags, src_ptr, field.offset as i32);
                    let is_excluded = exclude
                        .iter()
                        .any(|pat| glob_match(pat, &path) || glob_match(pat, &field.name));
                    let out = if is_excluded {
                        // Copy as-is via clone (bumps refcount internally)
                        self.compile_call_by_name(builder, "nsl_tensor_clone", &[src_val])?
                    } else {
                        // Quantize then immediately dequantize: the copy holds
                        // a regular tensor carrying the quantization error,
                        // which the model's own methods consume directly.
                        let qt = self.compile_call_by_name(
                            builder,
                            "nsl_qtensor_quantize",
                            &[src_val, codes.dtype, codes.granularity, codes.axis, codes.group_size],
                        )?;
                        let deq = self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[qt])?;
                        self.compile_call_by_name(builder, "nsl_qtensor_release", &[qt])?;
                        // The quantizer works on the host. Put the result
                        // where the source lives, or a model moved to a GPU
                        // would get a host weight beside device activations.
                        let placed =
                            self.compile_call_by_name(builder, "nsl_tensor_to_device_like", &[deq, src_val])?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[deq])?;
                        placed
                    };
                    builder.ins().store(flags, out, new_ptr, field.offset as i32);
                }
                FieldKind::Nested => {
                    let sub_type = field_types.as_ref().and_then(|ft| ft.get(&field.name).cloned());
                    let unsupported = |what: &str| {
                        CodegenError::new(format!(
                            "quant block: field '{path}' of model '{model_name}' is {what}; a quant block \
                             can copy tensor fields, sub-model fields and fixed-size model arrays only"
                        ))
                    };
                    let Some(sub_type) = sub_type else {
                        return Err(unsupported("a nested value whose type is not recorded"));
                    };
                    if let Some(inner) = sub_type.strip_prefix('[').and_then(|r| r.strip_suffix(']')) {
                        // `[Model; N]`: N model pointers stored inline.
                        let mut parts = inner.split(';');
                        let elem_type = parts.next().unwrap_or("").trim().to_string();
                        let count = parts.next().and_then(|c| c.trim().parse::<usize>().ok());
                        let Some(count) = count else {
                            return Err(unsupported(&format!("an array `{sub_type}` whose length is not a literal")));
                        };
                        if !self.types.struct_layouts.contains_key(&elem_type) {
                            return Err(unsupported(&format!("an array of `{elem_type}`, which is not a model")));
                        }
                        for i in 0..count {
                            let off = (field.offset + i * 8) as i32;
                            let elem_ptr = builder.ins().load(cl_types::I64, flags, src_ptr, off);
                            let elem_copy = self.emit_quantized_model_copy(
                                builder,
                                &elem_type,
                                elem_ptr,
                                &format!("{path}.{i}"),
                                exclude,
                                codes,
                                depth + 1,
                            )?;
                            builder.ins().store(flags, elem_copy, new_ptr, off);
                        }
                    } else if self.types.struct_layouts.contains_key(&sub_type) {
                        let sub_ptr = builder.ins().load(cl_types::I64, flags, src_ptr, field.offset as i32);
                        let sub_copy = self.emit_quantized_model_copy(
                            builder,
                            &sub_type,
                            sub_ptr,
                            &path,
                            exclude,
                            codes,
                            depth + 1,
                        )?;
                        builder.ins().store(flags, sub_copy, new_ptr, field.offset as i32);
                    } else if !self.types.enum_defs.contains_key(&sub_type) {
                        // (An enum value is a plain integer; the slot copy
                        // above already carried it over.)
                        return Err(unsupported(&format!("a `{sub_type}`")));
                    }
                }
            }
        }

        Ok(new_ptr)
    }
}
