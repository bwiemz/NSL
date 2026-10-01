//! WRGA: allocate and initialise a model instance's adapter side-table.
//!
//! Every model struct reserves a pointer slot at
//! `StructLayout::adapter_sidetable_offset` when an `@adapter` decorator is
//! active, and synthesized adapter field accesses (`lora_A_*`, `lora_B_*`,
//! `ia3_scale_*`, `gate_*`) read through it (`expr/access.rs`).
//!
//! The table is built in the model's CONSTRUCTOR, so every instance has
//! one: a top-level model, a sub-model field, and each element of a
//! `[Blk; N]` array, under either AD mode and outside any train block. It
//! used to be built at source-AD train-block entry, for the top-level model
//! only, so tape AD and nested instances read a null table.
//!
//! **Ordering invariant:** slot `k` holds the `k`-th field that
//! `Compiler::adapter_field_index` counts: both walk `bus.adapter_sites()`
//! in order, keep the sites whose `target_model` is this model and whose
//! dims resolved, and take each site's `synthesized_fields` in order.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::error::CodegenError;

/// The adapter sites whose tensors live in `model_type_name`'s side-table,
/// in slot order (see the module header).
fn sites_for(compiler: &Compiler<'_>, model_type_name: &str) -> Vec<PlacementEmit> {
    compiler
        .bus
        .adapter_sites()
        .iter()
        .filter(|s| s.target_model == model_type_name && s.input_dim != 0 && s.output_dim != 0)
        .map(|s| PlacementEmit {
            input_dim: i64::from(s.input_dim),
            output_dim: i64::from(s.output_dim),
            rank: s.rank.max(1),
            fields: s.synthesized_fields.clone(),
            target_field: s.target_field.clone(),
        })
        .collect()
}

/// The number of tensor slots in `model_type_name`'s side-table (0 when it
/// has no adapters).
pub(crate) fn sidetable_len(compiler: &Compiler<'_>, model_type_name: &str) -> usize {
    sites_for(compiler, model_type_name).iter().map(|s| s.fields.len()).sum()
}

/// Emit Cranelift IR that allocates the adapter side-table of the instance
/// at `model_ptr`, fills each slot with a freshly initialised tensor on the
/// target weight's device, and stores the table pointer into the instance.
///
/// Called from the model constructor after the fields are initialised (the
/// target weight is the device reference). A no-op when the model has no
/// reserved slot or no adapter targets it.
pub(crate) fn emit_adapter_init_sidetable(
    compiler: &mut Compiler<'_>,
    builder: &mut FunctionBuilder,
    model_ptr: Value,
    model_type_name: &str,
) -> Result<(), CodegenError> {
    let Some(layout) = compiler.types.struct_layouts.get(model_type_name).cloned() else {
        return Ok(());
    };
    let Some(slot_off) = layout.adapter_sidetable_offset else {
        return Ok(());
    };
    let sites = sites_for(compiler, model_type_name);
    let total_fields: usize = sites.iter().map(|s| s.fields.len()).sum();
    if total_fields == 0 {
        return Ok(());
    }
    let table_bytes = builder.ins().iconst(cl_types::I64, (total_fields * 8) as i64);
    let table_ptr = compiler.compile_call_by_name(builder, "nsl_alloc", &[table_bytes])?;

    let mut idx: i64 = 0;
    for site in &sites {
        // The target weight is the device reference: a tensor created here
        // is host-resident, and is moved next to the weight it adapts.
        let field_layout = layout
            .fields
            .iter()
            .find(|f| f.name == site.target_field)
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "adapter init: target field '{}' not in struct layout for model '{}'",
                    site.target_field, model_type_name,
                ))
            })?;
        let ref_tensor_ptr = builder.ins().load(
            cl_types::I64,
            MemFlagsData::trusted(),
            model_ptr,
            cranelift_codegen::ir::immediates::Offset32::new(field_layout.offset as i32),
        );

        for field in &site.fields {
            let tensor_ptr = emit_one_init(compiler, builder, site, field)?;
            // `to_device_like` returns the tensor itself, retained, when the
            // devices already match, and a new tensor otherwise; either way
            // the creation reference is released here.
            let placed_tensor = compiler.compile_call_by_name(
                builder,
                "nsl_tensor_to_device_like",
                &[tensor_ptr, ref_tensor_ptr],
            )?;
            compiler.compile_call_by_name(builder, "nsl_tensor_free", &[tensor_ptr])?;
            builder.ins().store(MemFlagsData::trusted(), placed_tensor, table_ptr, (idx * 8) as i32);
            idx += 1;
        }
    }

    builder.ins().store(MemFlagsData::trusted(), table_ptr, model_ptr, slot_off as i32);
    Ok(())
}

struct PlacementEmit {
    input_dim: i64,
    output_dim: i64,
    rank: i64,
    /// The site's synthesized field names, in slot order.
    fields: Vec<String>,
    /// Target weight field name on the model struct (e.g. "w"), whose
    /// tensor is the device reference for `nsl_tensor_to_device_like`.
    target_field: String,
}

/// Emit one synthesized field's tensor creation FFI. Returns the owned
/// tensor pointer.
fn emit_one_init(
    compiler: &mut Compiler<'_>,
    builder: &mut FunctionBuilder,
    site: &PlacementEmit,
    name: &str,
) -> Result<Value, CodegenError> {
    // Determine shape + init strategy from field-name prefix. These prefixes
    // are the synthesis contract defined in `wrga_adapter_inject::run`.
    // Every temporary (shape lists, the unscaled draw, the scale) is freed;
    // only the returned tensor is owned by the caller.
    if name.starts_with("lora_A_") {
        // LoRA A: [input_dim, rank], KaimingUniform (randn scaled by
        // 1/sqrt(fan_in)). The shape is [in, rank] (NOT [rank, in]) so the
        // forward rewrite can do `x @ self.lora_A` without a transpose.
        let shape = build_shape_list(compiler, builder, &[site.input_dim, site.rank])?;
        let base = compiler.compile_call_by_name(builder, "nsl_tensor_randn", &[shape])?;
        compiler.compile_call_by_name(builder, "nsl_list_free", &[shape])?;
        let fan_in = site.input_dim.max(1) as f64;
        let scale = 1.0_f64 / fan_in.sqrt();
        let one_shape = build_shape_list(compiler, builder, &[1])?;
        let scale_val = builder.ins().f64const(scale);
        let scale_tensor =
            compiler.compile_call_by_name(builder, "nsl_tensor_full", &[one_shape, scale_val])?;
        compiler.compile_call_by_name(builder, "nsl_list_free", &[one_shape])?;
        // FBIP flags byte (third arg for tensor-tensor FFIs): pass 0 —
        // neither operand is relinquished; both are freed below.
        let flags = builder.ins().iconst(cl_types::I8, 0);
        let scaled =
            compiler.compile_call_by_name(builder, "nsl_tensor_mul", &[base, scale_tensor, flags])?;
        compiler.compile_call_by_name(builder, "nsl_tensor_free", &[base])?;
        compiler.compile_call_by_name(builder, "nsl_tensor_free", &[scale_tensor])?;
        return Ok(scaled);
    }
    let (ctor, dims): (&str, Vec<i64>) = if name.starts_with("lora_B_") {
        // LoRA B: [rank, output_dim], zeros: `(x @ A) @ self.lora_B` needs
        // no transpose, and the adapted layer starts equal to its base.
        ("nsl_tensor_zeros", vec![site.rank, site.output_dim])
    } else if name.starts_with("ia3_scale_") {
        // IA³: [output_dim], ones.
        ("nsl_tensor_ones", vec![site.output_dim])
    } else if name.starts_with("gate_") {
        // GatedLoRA gate: [output_dim], zeros.
        ("nsl_tensor_zeros", vec![site.output_dim])
    } else {
        return Err(CodegenError::new(format!(
            "unknown synthesized adapter field prefix: '{name}' \
             (expected lora_A_*, lora_B_*, ia3_scale_*, gate_*)"
        )));
    };
    let shape = build_shape_list(compiler, builder, &dims)?;
    let tensor = compiler.compile_call_by_name(builder, ctor, &[shape])?;
    compiler.compile_call_by_name(builder, "nsl_list_free", &[shape])?;
    Ok(tensor)
}

/// Build an `nsl_list` of i64 dims for a tensor-creation FFI.
fn build_shape_list(
    compiler: &mut Compiler<'_>,
    builder: &mut FunctionBuilder,
    dims: &[i64],
) -> Result<Value, CodegenError> {
    let list = compiler.compile_call_by_name(builder, "nsl_list_new", &[])?;
    for &d in dims {
        let dim_val = builder.ins().iconst(cl_types::I64, d);
        compiler.compile_call_by_name(builder, "nsl_list_push", &[list, dim_val])?;
    }
    Ok(list)
}
