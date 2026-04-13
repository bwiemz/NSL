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
    let Some(layout) = compiler
        .types
        .struct_layouts
        .get(model_type_name)
        .cloned()
    else {
        return Ok(());
    };
    let Some(slot_off) = layout.adapter_sidetable_offset else {
        return Ok(());
    };

    // 2. Collect placements targeting this model with usable dims, in the
    //    same order `adapter_field_index` iterates. Each entry keeps the
    //    (input_dim, output_dim, rank, per-field init strategies).
    let field_types = compiler.models.model_field_types.clone();
    let mut emit_list: Vec<PlacementEmit> = Vec::new();
    for placement in &plan.placements {
        if placement.decorator_kind.is_none() {
            continue;
        }
        let target_field = placement.name.rsplit('.').next().unwrap_or("");
        let matches_model = field_types
            .get(model_type_name)
            .map(|m| m.contains_key(target_field))
            .unwrap_or(false);
        if !matches_model {
            continue;
        }
        // Resolve dims from the model_field_types registry (same path the
        // inject pass uses).  Skip sites with unresolved / zero dims.
        let (inp, out) = match crate::wrga_adapter_inject::resolve_dims_for_target(
            model_type_name,
            target_field,
            &field_types,
        ) {
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
    let mut idx: i64 = 0;
    for site in &emit_list {
        for strat in &site.strategies {
            let tensor_ptr = emit_one_init(compiler, builder, site, strat)?;
            let byte_off = (idx * 8) as i32;
            builder.ins().store(
                MemFlags::trusted(),
                tensor_ptr,
                table_ptr,
                byte_off,
            );
            idx += 1;
        }
    }

    // 5. Store the table pointer into the model struct at the reserved slot.
    builder.ins().store(
        MemFlags::trusted(),
        table_ptr,
        model_ptr,
        slot_off as i32,
    );

    Ok(())
}

struct PlacementEmit {
    input_dim: i64,
    output_dim: i64,
    rank: i64,
    strategies: Vec<crate::wrga_adapter_inject::InitStrategy>,
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
        // LoRA A: [rank, input_dim], KaimingUniform (randn scaled by
        // 1/sqrt(fan_in)).
        let shape = build_shape_list(compiler, builder, &[site.rank, site.input_dim])?;
        let base = compiler.compile_call_by_name(builder, "nsl_tensor_randn", &[shape])?;
        let fan_in = site.input_dim.max(1) as f64;
        let scale = 1.0_f64 / fan_in.sqrt();
        // Build shape-[1] and use `nsl_tensor_full` to get a scalar tensor.
        let one_shape = build_shape_list(compiler, builder, &[1])?;
        let scale_val = builder.ins().f64const(scale);
        let scale_tensor =
            compiler.compile_call_by_name(builder, "nsl_tensor_full", &[one_shape, scale_val])?;
        let scaled =
            compiler.compile_call_by_name(builder, "nsl_tensor_mul", &[base, scale_tensor])?;
        Ok(scaled)
    } else if name.starts_with("lora_B_") {
        // LoRA B: [output_dim, rank], Zeros.
        debug_assert_eq!(strat.kind, InitKind::Zeros);
        let shape = build_shape_list(compiler, builder, &[site.output_dim, site.rank])?;
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
