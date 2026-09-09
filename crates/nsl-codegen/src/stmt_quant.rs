//! The `quant` block lowering: `compile_quant_block` (the per-model
//! quantisation call sequence — dtype and granularity selection, the
//! calibration-driven AWQ/GPTQ paths, and the wire-format handoff to the
//! runtime) and `discover_awq_projections` (the forward-method walk that
//! enumerates the linear-projection call sites AWQ calibration needs;
//! `lib.rs` also calls it when it synthesises the calibration harness).
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1); the statement
//! dispatch (`stmt.rs::compile_stmt_dispatch`) still calls
//! `compile_quant_block`.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlagsData};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;
use nsl_ast::block::{QuantDtype, QuantGranularity};

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

use crate::stmt::glob_match;
use cranelift_codegen::ir::Value;
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

        // 2. Resolve model type name using the same strategy as train blocks:
        //    scan the type_map for a Model type with a known struct layout.
        let model_type_name = {
            let mut found_name = None;
            for (_node_id, ty) in self.type_map.iter() {
                match ty {
                    nsl_semantic::types::Type::Model { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.types.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                            break;
                        }
                    }
                    nsl_semantic::types::Type::Struct { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.types.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                        }
                    }
                    _ => {}
                }
            }
            found_name.unwrap_or_else(|| self.resolve_sym(source_sym).to_string())
        };

        let layout = self
            .types
            .struct_layouts
            .get(&model_type_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "no struct layout found for model '{}' in quant block",
                    model_type_name
                ))
            })?;

        // 3. Compute dtype/granularity integer codes for the runtime call
        let dtype_code: i64 = match quant.default_dtype {
            Some(QuantDtype::Int4) => 1,
            Some(QuantDtype::Awq4) => 2,
            Some(QuantDtype::Gptq4) => 3,
            Some(QuantDtype::Gptq8) => 4,
            Some(QuantDtype::Int8) | None => 0,
        };
        let (gran_code, axis_val, gs_val): (i64, i64, i64) = match &quant.default_granularity {
            Some(QuantGranularity::PerChannel(a)) => (1, *a, 0),
            Some(QuantGranularity::PerGroup(a, gs)) => (2, *a, *gs),
            Some(QuantGranularity::PerTensor) | None => (0, 0, 0),
        };

        let dtype_v = builder.ins().iconst(cl_types::I64, dtype_code);
        let gran_v = builder.ins().iconst(cl_types::I64, gran_code);
        let axis_v = builder.ins().iconst(cl_types::I64, axis_val);
        let gs_v = builder.ins().iconst(cl_types::I64, gs_val);

        // 4. Allocate a new struct with the same layout as the source model
        let alloc_size = builder
            .ins()
            .iconst(cl_types::I64, layout.total_size.max(8) as i64);
        let new_ptr = self.compile_call_by_name(builder, "nsl_alloc", &[alloc_size])?;

        // 4b. If this is an AWQ quant block and a calibration sidecar is present,
        //     decode the AWQ activation scales once.  We'll use them per-field below.
        //
        //     Key: "awq_activation_scales" in sidecar.hooks (binary blob).
        //     Projection path format: "{model_type_name}.{field_name}" — same
        //     cache key as Task 8's discovery pass.
        //
        //     Hard error when sidecar present but projection missing:
        //     silent fallback to uncalibrated is a correctness trap.
        let is_awq = matches!(quant.default_dtype, Some(QuantDtype::Awq4));
        let awq_scales_opt: Option<nsl_runtime::awq::AwqScales> = if is_awq {
            match self.compile_options.calibration.sidecar.as_ref() {
                None => None,
                Some(sidecar) => {
                    match sidecar.hooks.get("awq_activation_scales") {
                        None => None, // Sidecar present but no AWQ hook blob → treat as uncalibrated.
                        Some(blob) => {
                            match nsl_runtime::awq::AwqScales::from_blob(blob) {
                                Ok(scales) => Some(scales),
                                Err(e) => {
                                    // Blob present but malformed → hard error.
                                    return Err(CodegenError::new(format!(
                                        "AWQ calibration sidecar blob is malformed: {e}"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        } else {
            None
        };

        // AWQ calibration alpha (matches awq_quantize_with_scales default).
        let awq_alpha: f64 = 0.5;

        // 5. For each field: quantize→dequantize (or clone if excluded)
        for field in &layout.fields {
            let is_excluded = quant.exclude.iter().any(|pat| glob_match(pat, &field.name));
            let src_val = builder.ins().load(
                field.cl_type,
                MemFlagsData::trusted(),
                source_val,
                field.offset as i32,
            );

            if is_excluded {
                // Copy as-is via clone (bumps refcount internally)
                let cloned = self.compile_call_by_name(builder, "nsl_tensor_clone", &[src_val])?;
                builder
                    .ins()
                    .store(MemFlagsData::trusted(), cloned, new_ptr, field.offset as i32);
            } else {
                // For AWQ with a calibration sidecar, pre-scale the weight tensor using
                // the per-input-channel activation statistics before quantizing.
                // This embeds the scale data as compile-time constants in the object file
                // and calls nsl_awq_pre_scale_weight at runtime to apply them.
                let weight_for_quantize: Value = if is_awq {
                    match awq_scales_opt.as_ref() {
                        None => {
                            // No sidecar → uncalibrated, pass weight through unchanged.
                            src_val
                        }
                        Some(scales_map) => {
                            // Sidecar present — projection MUST have scales.
                            let projection_path =
                                format!("{}.{}", model_type_name, field.name);
                            let field_scales = scales_map
                                .by_projection
                                .get(&projection_path)
                                .ok_or_else(|| {
                                    CodegenError::missing_scales(&projection_path)
                                })?;

                            // Embed scale data as a compile-time constant in .rodata.
                            let data_label = format!(
                                "__nsl_awq_scales_{}_{}",
                                model_type_name, field.name
                            );
                            let scale_bytes: Vec<u8> = field_scales
                                .iter()
                                .flat_map(|v: &f32| v.to_le_bytes())
                                .collect();
                            let scale_data_id = self
                                .module
                                .declare_data(
                                    &data_label,
                                    cranelift_module::Linkage::Local,
                                    false,
                                    false,
                                )
                                .map_err(|e| {
                                    CodegenError::new(format!(
                                        "failed to declare AWQ scale data for \
                                         '{projection_path}': {e}"
                                    ))
                                })?;
                            let mut data_desc = cranelift_module::DataDescription::new();
                            data_desc.define(scale_bytes.into_boxed_slice());
                            self.module
                                .define_data(scale_data_id, &data_desc)
                                .map_err(|e| {
                                    CodegenError::new(format!(
                                        "failed to define AWQ scale data for \
                                         '{projection_path}': {e}"
                                    ))
                                })?;

                            // Get a pointer to the scale data in this function.
                            let scale_gv = self
                                .module
                                .declare_data_in_func(scale_data_id, builder.func);
                            let scales_ptr =
                                builder.ins().symbol_value(cl_types::I64, scale_gv);
                            let scales_len = builder
                                .ins()
                                .iconst(cl_types::I64, field_scales.len() as i64);
                            let alpha_v = builder.ins().f64const(awq_alpha);

                            // Apply calibration scaling: returns a new NslTensor.
                            self.compile_call_by_name(
                                builder,
                                "nsl_awq_pre_scale_weight",
                                &[src_val, scales_ptr, scales_len, alpha_v],
                            )?
                        }
                    }
                } else {
                    src_val
                };

                // Quantize then immediately dequantize — validates the roundtrip and
                // shows quantization effects (precision loss) while storing a regular
                // NslTensor that the original forward method can consume directly.
                let qt = self.compile_call_by_name(
                    builder,
                    "nsl_qtensor_quantize",
                    &[weight_for_quantize, dtype_v, gran_v, axis_v, gs_v],
                )?;
                let deq = self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[qt])?;
                // Release the intermediate QuantizedTensor (refcount-aware)
                self.compile_call_by_name(builder, "nsl_qtensor_release", &[qt])?;
                // If we pre-scaled the weight, release the intermediate scaled tensor too.
                if is_awq && awq_scales_opt.is_some() {
                    self.compile_call_by_name(
                        builder,
                        "nsl_tensor_release",
                        &[weight_for_quantize],
                    )?;
                }
                builder
                    .ins()
                    .store(MemFlagsData::trusted(), deq, new_ptr, field.offset as i32);
            }
        }

        // 6. Register the quantized model with the same struct layout and methods
        //    so that forward dispatch works identically to the source model.
        let quant_name = self.resolve_sym(quant.name).to_string();
        if !self.types.struct_layouts.contains_key(&quant_name) {
            self.types.struct_layouts.insert(quant_name.clone(), layout);
        }
        if let Some(methods) = self.models.model_methods.get(&model_type_name).cloned() {
            self.models.model_methods.insert(quant_name, methods);
        }

        // 7. Bind the new struct pointer as the output variable
        let var = builder.declare_var(cl_types::I64);
        builder.def_var(var, new_ptr);
        state.variables.insert(quant.name, (var, cl_types::I64));

        Ok(())
    }

    /// Walk the compiled model's `quant { ... }` blocks and produce the
    /// list of ProjectionRefs that AWQ needs calibration data for.
    /// Returns `None` when no AWQ quant block is present or discovery
    /// produces no matches.
    ///
    /// Implementation (Task 3): scans `self.features.quant_configs` for
    /// models quantised with `"awq4"`.  For each such model, retrieves the
    /// `forward` method body from `model_method_bodies`, walks its pipe chain
    /// to enumerate linear-projection call sites, and returns the sorted,
    /// deduplicated `Vec<ProjectionRef>`.  Discovery errors (e.g. empty match)
    /// are logged to stderr and treated as `None` so the harness falls back to
    /// its no-op path rather than crashing the compile.
    pub(crate) fn discover_awq_projections(
        &self,
    ) -> Option<Vec<crate::calibration::DiscoveredProjection>> {
        use crate::calibration::discover_awq_projections_from_state;

        // Collect all AWQ-quantised model names.
        let awq_models: Vec<String> = self
            .features
            .quant_configs
            .iter()
            .filter(|(_, cfg)| cfg.dtype == "awq4")
            .map(|(name, _)| name.clone())
            .collect();

        if awq_models.is_empty() {
            return None;
        }

        let mut all_projections: Vec<crate::calibration::DiscoveredProjection> = Vec::new();

        for model_name in &awq_models {
            // Retrieve the forward method body (if stored).
            let forward_body: Option<&nsl_ast::stmt::Block> = self
                .models
                .model_method_bodies
                .get(model_name)
                .and_then(|methods| methods.get("forward"))
                .map(|fn_def| &fn_def.body);

            // Retrieve field-type and shape maps for this model.
            let empty_field_types = std::collections::HashMap::new();
            let field_types = self
                .models
                .model_field_types
                .get(model_name)
                .unwrap_or(&empty_field_types);

            let empty_shapes = std::collections::HashMap::new();
            let tensor_shapes = self
                .models
                .model_tensor_field_shapes
                .get(model_name)
                .unwrap_or(&empty_shapes);

            match discover_awq_projections_from_state(
                model_name,
                forward_body,
                field_types,
                tensor_shapes,
                &[], // no exclusions from the Compiler-level stub; the QuantBlock's
                     // exclude list is stored in the AST which isn't retained here.
                self.interner,
            ) {
                Ok(discovered) => {
                    for dp in discovered {
                        all_projections.push(dp);
                    }
                }
                Err(e) => {
                    nsl_log::nsl_log!(INFO, "calibration", "[calibration] AWQ discovery for model '{model_name}': {e}");
                }
            }
        }

        if all_projections.is_empty() {
            None
        } else {
            // Sort + dedup across models (by qualified path for determinism).
            all_projections.sort_by(|a, b| a.projection.0.cmp(&b.projection.0));
            all_projections.dedup_by(|a, b| a.projection.0 == b.projection.0);
            Some(all_projections)
        }
    }
}
