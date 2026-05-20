//! WRGA B.2.1 Task 2.5: emit adapter side-table allocation + init at
//! `@train` block entry.
//!
//! Task 2 (commit 417610d) reserved a pointer slot on the model struct at
//! `StructLayout::adapter_sidetable_offset` and routed synthesized adapter
//! field accesses (`lora_A_*`, `lora_B_*`, `ia3_scale_*`, `gate_*`) through
//! that slot. The slot was left null because WRGA fires *after*
//! `compile_model_constructor`, so no plan exists at construction time.
//!
//! This pass runs immediately after `invoke_wrga_if_enabled` returns a plan
//! inside the train-step codegen path. At that point:
//!
//!   * `compiler.last_wrga_plan` is populated.
//!   * The target model has already been constructed by the `let m = Toy()`
//!     statement — `model_ptr` is a live Cranelift `Value`.
//!   * The side-table slot is still null from constructor zero-init.
//!
//! We walk the plan's placements, filter to those targeting this model and
//! with resolved non-zero dims, and for each synthesized field emit the
//! appropriate tensor creation FFI. Resulting tensor pointers are written
//! into a heap `Vec<i64>` (one `i64` per field) obtained via `nsl_alloc`,
//! and finally the table pointer is stored into the model struct at
//! `adapter_sidetable_offset`.
//!
//! **Ordering invariant:** the iteration order here MUST match
//! `Compiler::adapter_field_index` in `expr/access.rs`. Both iterate
//! `plan.placements` in insertion order; within each matching placement,
//! both iterate `synthesized_fields` in declared order.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::wrga::WrgaPlan;
use crate::wrga_adapter_inject::InitKind;

/// Emit Cranelift IR that allocates the adapter side-table, populates each
/// slot with a freshly-initialised tensor, and writes the table pointer into
/// the model struct.
///
/// No-op when the model has no reserved side-table slot (i.e. no `@adapter`
/// decorator is active) or when no placement in the plan targets this model
/// with resolved non-zero dims.
pub(crate) fn emit_adapter_init_sidetable(
    compiler: &mut Compiler<'_>,
    builder: &mut FunctionBuilder,
    _state: &mut FuncState,
    model_ptr: Value,
    model_type_name: &str,
    plan: &WrgaPlan,
) -> Result<(), CodegenError> {
    // 1. Check this model has a side-table slot reserved.
    let Some(layout) = compiler.types.struct_layouts.get(model_type_name).cloned() else {
        return Ok(());
    };
    let Some(slot_off) = layout.adapter_sidetable_offset else {
        return Ok(());
    };

    // 2. Collect placements targeting this model with usable dims, in the
    //    same order `adapter_field_index` iterates. Each entry keeps the
    //    (input_dim, output_dim, rank, per-field init strategies).
    // B.2.1 Task 5.5: prefer the dedicated tensor-shape map for dim
    // resolution; fall back to model_field_types for sub-model targets.
    let tensor_shapes = compiler.models.model_tensor_field_shapes.clone();
    let model_field_types = compiler.models.model_field_types.clone();
    let mut emit_list: Vec<PlacementEmit> = Vec::new();
    for placement in &plan.placements {
        if placement.decorator_kind.is_none() {
            continue;
        }
        let target_field = placement.name.rsplit('.').next().unwrap_or("");
        let matches_model = tensor_shapes
            .get(model_type_name)
            .map(|m| m.contains_key(target_field))
            .unwrap_or(false)
            || model_field_types
                .get(model_type_name)
                .map(|m| m.contains_key(target_field))
                .unwrap_or(false);
        if !matches_model {
            continue;
        }
        // Resolve dims: shape map first, model_field_types as fallback.
        let (inp, out) = match crate::wrga_adapter_inject::resolve_dims_for_target(
            model_type_name,
            target_field,
            &tensor_shapes,
        )
        .or_else(|| {
            crate::wrga_adapter_inject::resolve_dims_for_target(
                model_type_name,
                target_field,
                &model_field_types,
            )
        }) {
            Some(d) => d,
            None => continue,
        };
        if inp == 0 || out == 0 {
            continue;
        }
        let rank = placement.suggested_rank.max(1) as i64;
        emit_list.push(PlacementEmit {
            input_dim: inp as i64,
            output_dim: out as i64,
            rank,
            strategies: placement.init_strategies.clone(),
            target_field: target_field.to_string(),
        });
    }

    if emit_list.is_empty() {
        return Ok(());
    }

    // 3. Count total synthesized fields and allocate the heap table.
    let total_fields: i64 = emit_list.iter().map(|e| e.strategies.len() as i64).sum();
    if total_fields == 0 {
        return Ok(());
    }
    let table_bytes = builder.ins().iconst(cl_types::I64, total_fields * 8);
    let table_ptr = compiler.compile_call_by_name(builder, "nsl_alloc", &[table_bytes])?;

    // 4. Emit tensor creation per site+field, in order, and store into the
    //    table at consecutive 8-byte slots.
    //
    // Commit-C placement fix: newly-created adapter tensors are CPU-resident
    // (nsl_tensor_zeros/ones/full/randn all allocate on the host). If the
    // model was moved to GPU before entering this train block (via
    // `m.to(cuda)`), the side-table tensors would otherwise stay on CPU, and
    // the fused FFI at forward time would hit its device-check and fall back
    // to CPU math. We reconcile by migrating each newly-created tensor onto
    // the same device as the target weight field (loaded from the model
    // struct at field.offset) using `nsl_tensor_to_device_like`.
    let mut idx: i64 = 0;
    for site in &emit_list {
        // Resolve the target-field byte offset on the model struct so we can
        // load its tensor pointer as the device reference.
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
            MemFlags::trusted(),
            model_ptr,
            cranelift_codegen::ir::immediates::Offset32::new(field_layout.offset as i32),
        );

        for strat in &site.strategies {
            let tensor_ptr = emit_one_init(compiler, builder, site, strat)?;
            // Migrate onto the target field's device. When devices already
            // match (e.g., model still on CPU), this is a refcount-bump
            // no-op in the runtime helper.
            let placed_tensor = compiler.compile_call_by_name(
                builder,
                "nsl_tensor_to_device_like",
                &[tensor_ptr, ref_tensor_ptr],
            )?;
            let byte_off = (idx * 8) as i32;
            builder
                .ins()
                .store(MemFlags::trusted(), placed_tensor, table_ptr, byte_off);
            idx += 1;
        }
    }

    // 5. Store the table pointer into the model struct at the reserved slot.
    builder
        .ins()
        .store(MemFlags::trusted(), table_ptr, model_ptr, slot_off as i32);

    Ok(())
}

struct PlacementEmit {
    input_dim: i64,
    output_dim: i64,
    rank: i64,
    strategies: Vec<crate::wrga_adapter_inject::InitStrategy>,
    /// Target weight field name on the model struct (e.g., "w" for
    /// placement "Toy.w"). Used at emit time to look up the field's byte
    /// offset from `StructLayout::fields` and load the field's tensor
    /// pointer as the reference device for `nsl_tensor_to_device_like`.
    target_field: String,
}

/// Emit one synthesized field's tensor creation FFI. Returns the owned
/// tensor pointer.
fn emit_one_init(
    compiler: &mut Compiler<'_>,
    builder: &mut FunctionBuilder,
    site: &PlacementEmit,
    strat: &crate::wrga_adapter_inject::InitStrategy,
) -> Result<Value, CodegenError> {
    let name = &strat.field_name;
    // Determine shape + init strategy from field-name prefix. These prefixes
    // are the synthesis contract defined in `wrga_adapter_inject::run`.
    if name.starts_with("lora_A_") {
        // LoRA A: [input_dim, rank], KaimingUniform (randn scaled by
        // 1/sqrt(fan_in)).
        // B.2.1 Task 5.5: shape is [in, rank] (NOT [rank, in]) so the
        // forward rewrite can do `x @ self.lora_A` directly without
        // a transpose.
        let shape = build_shape_list(compiler, builder, &[site.input_dim, site.rank])?;
        let base = compiler.compile_call_by_name(builder, "nsl_tensor_randn", &[shape])?;
        let fan_in = site.input_dim.max(1) as f64;
        let scale = 1.0_f64 / fan_in.sqrt();
        // Build shape-[1] and use `nsl_tensor_full` to get a scalar tensor.
        let one_shape = build_shape_list(compiler, builder, &[1])?;
        let scale_val = builder.ins().f64const(scale);
        let scale_tensor =
            compiler.compile_call_by_name(builder, "nsl_tensor_full", &[one_shape, scale_val])?;
        // FBIP flags byte (third arg for tensor-tensor FFIs): pass 0 —
        // neither operand is relinquished by the init pass.
        let flags = builder.ins().iconst(cl_types::I8, 0);
        let scaled = compiler.compile_call_by_name(
            builder,
            "nsl_tensor_mul",
            &[base, scale_tensor, flags],
        )?;
        Ok(scaled)
    } else if name.starts_with("lora_B_") {
        // LoRA B: [rank, output_dim], Zeros.
        // B.2.1 Task 5.5: shape is [rank, out] (NOT [out, rank]) so the
        // forward rewrite can do `(x @ A) @ self.lora_B` directly.
        debug_assert_eq!(strat.kind, InitKind::Zeros);
        let shape = build_shape_list(compiler, builder, &[site.rank, site.output_dim])?;
        compiler.compile_call_by_name(builder, "nsl_tensor_zeros", &[shape])
    } else if name.starts_with("ia3_scale_") {
        // IA³: [output_dim], Ones.
        debug_assert_eq!(strat.kind, InitKind::Ones);
        let shape = build_shape_list(compiler, builder, &[site.output_dim])?;
        compiler.compile_call_by_name(builder, "nsl_tensor_ones", &[shape])
    } else if name.starts_with("gate_") {
        // GatedLoRA gate: [output_dim], Zeros.
        debug_assert_eq!(strat.kind, InitKind::Zeros);
        let shape = build_shape_list(compiler, builder, &[site.output_dim])?;
        compiler.compile_call_by_name(builder, "nsl_tensor_zeros", &[shape])
    } else {
        Err(CodegenError::new(format!(
            "unknown synthesized adapter field prefix: '{name}' \
             (expected lora_A_*, lora_B_*, ia3_scale_*, gate_*)"
        )))
    }
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
