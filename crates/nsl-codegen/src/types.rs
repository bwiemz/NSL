use cranelift_codegen::ir::{self, types};
use cranelift_frontend::FunctionBuilder;
use nsl_semantic::types::Type;

/// Check if the given block already has a terminator instruction.
pub fn is_block_filled(builder: &FunctionBuilder, block: ir::Block) -> bool {
    builder
        .func
        .layout
        .last_inst(block)
        .map_or(false, |inst| builder.func.dfg.insts[inst].opcode().is_terminator())
}

/// Map an NSL semantic type to a Cranelift IR type.
pub fn nsl_type_to_cl(ty: &Type) -> types::Type {
    match ty {
        Type::Int | Type::Int64 => types::I64,
        Type::Int32 => types::I32,
        Type::Int16 => types::I16,
        Type::Int8 | Type::Int4 | Type::Uint8 => types::I8,
        Type::Bool => types::I8,
        Type::Float | Type::F64 => types::F64,
        Type::F32 => types::F32,
        Type::Fp16 | Type::Bf16 | Type::Fp8E4m3 | Type::Fp8E5m2 => types::F32,
        Type::Str => types::I64,
        Type::List(_) => types::I64,
        Type::Struct { .. } => types::I64,
        Type::Model { .. } => types::I64,
        Type::Dict(_, _) => types::I64,
        Type::Tuple(_) => types::I64,
        Type::Optional(_) => types::I64,
        Type::Tensor { .. } | Type::Param { .. } | Type::Buffer { .. } => types::I64,
        Type::Void => types::I8, // should not appear as value type
        Type::Unknown | Type::Error => {
            #[cfg(debug_assertions)]
            {
                use std::sync::atomic::{AtomicBool, Ordering};
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    eprintln!("[nsl-codegen] warning: Type::Unknown reached codegen (defaulting to I64)");
                }
            }
            types::I64
        }
        // Remaining variants: all pointer/opaque at IR level
        Type::Sparse { .. }
        | Type::QuantizedTensor
        | Type::Function { .. }
        | Type::Enum { .. }
        | Type::Union(_)
        | Type::TypeVar(_)
        | Type::Module { .. }
        | Type::NoneType
        | Type::FixedModelArray { .. } => types::I64,
    }
}

/// Map an NSL type to an optional Cranelift type for function return values.
/// Returns None for Void (no return value).
pub fn nsl_return_type(ty: &Type) -> Option<types::Type> {
    match ty {
        Type::Void => None,
        _ => Some(nsl_type_to_cl(ty)),
    }
}

/// The pointer type for our target (always 64-bit).
pub fn pointer_type() -> types::Type {
    types::I64
}

/// Check if an NSL type is a float type at the Cranelift level.
pub fn is_float_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Float | Type::F64 | Type::F32 | Type::Fp16 | Type::Bf16 | Type::Fp8E4m3 | Type::Fp8E5m2
    )
}

/// Check if an NSL type is an integer type at the Cranelift level.
pub fn is_int_type(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Int
            | Type::Int4
            | Type::Int8
            | Type::Int16
            | Type::Int32
            | Type::Int64
            | Type::Uint8
            | Type::Bool
    )
}
