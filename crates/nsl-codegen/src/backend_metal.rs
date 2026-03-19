// crates/nsl-codegen/src/backend_metal.rs
//! M47b: KIR -> Metal Shading Language (MSL) for Apple Silicon.
//!
//! Produces MSL source code that can be compiled via `xcrun metallib`
//! or the Metal Compiler API at build time.
//! Key differences from PTX:
//! - Memory model: device (global), threadgroup (shared), thread (local)
//! - No warp shuffle — uses SIMD shuffle: simd_shuffle_down(val, offset)
//! - No tensor cores — falls back to SIMD matrix operations
//! - Thread indexing via attributes: [[thread_position_in_grid]]

use crate::kernel_ir::*;
use std::fmt::Write;

/// Lower a KernelIR to MSL source code bytes.
pub fn lower_kir_to_msl(ir: &KernelIR) -> Vec<u8> {
    let mut msl = String::new();

    writeln!(msl, "#include <metal_stdlib>").unwrap();
    writeln!(msl, "using namespace metal;").unwrap();
    writeln!(msl).unwrap();

    // Kernel signature
    writeln!(msl, "kernel void {}(", ir.name).unwrap();
    for (i, param) in ir.params.iter().enumerate() {
        let msl_type = kir_type_to_msl(&param.ty);
        let attr = match param.address_space {
            AddressSpace::Global => "device",
            AddressSpace::Constant => "constant",
            AddressSpace::Shared => "threadgroup",
            _ => "device",
        };
        let is_ptr = matches!(param.ty, KirType::Ptr(_, _));
        if is_ptr {
            write!(msl, "    {} {}* {} [[buffer({})]]", attr, msl_type, param.name, i).unwrap();
        } else {
            write!(msl, "    {} {} [[buffer({})]]", msl_type, param.name, i).unwrap();
        }
        if i < ir.params.len() - 1 {
            writeln!(msl, ",").unwrap();
        }
    }
    // Thread indexing attributes
    if !ir.params.is_empty() {
        writeln!(msl, ",").unwrap();
    }
    writeln!(msl, "    uint tid [[thread_position_in_grid]],").unwrap();
    writeln!(msl, "    uint block_id [[threadgroup_position_in_grid]],").unwrap();
    writeln!(msl, "    uint local_id [[thread_position_in_threadgroup]]").unwrap();
    writeln!(msl, ") {{").unwrap();

    // Body
    for block in &ir.blocks {
        for op in &block.ops {
            writeln!(msl, "    {}", lower_op_to_msl(op)).unwrap();
        }
    }

    writeln!(msl, "}}").unwrap();

    msl.into_bytes()
}

fn kir_type_to_msl(ty: &KirType) -> &'static str {
    match ty {
        KirType::U32 => "uint",
        KirType::I32 => "int",
        KirType::U64 => "ulong",
        KirType::I64 => "long",
        KirType::F32 => "float",
        KirType::F64 => "double",
        KirType::F16 => "half",
        KirType::Ptr(inner, _) => kir_type_to_msl(inner),
        _ => "uint",
    }
}

fn lower_op_to_msl(op: &KirOp) -> String {
    match op {
        KirOp::Add(dst, a, b) => format!("auto v{} = v{} + v{};", dst, a, b),
        KirOp::Sub(dst, a, b) => format!("auto v{} = v{} - v{};", dst, a, b),
        KirOp::Mul(dst, a, b) => format!("auto v{} = v{} * v{};", dst, a, b),
        KirOp::Div(dst, a, b) => format!("auto v{} = v{} / v{};", dst, a, b),
        KirOp::Fma(dst, a, b, c) => format!("auto v{} = fma(v{}, v{}, v{});", dst, a, b, c),
        KirOp::Neg(dst, a) => format!("auto v{} = -v{};", dst, a),
        KirOp::Abs(dst, a) => format!("auto v{} = abs(v{});", dst, a),
        KirOp::Sqrt(dst, a) => format!("auto v{} = sqrt(v{});", dst, a),
        KirOp::Exp(dst, a) => format!("auto v{} = exp(v{});", dst, a),
        KirOp::Log(dst, a) => format!("auto v{} = log(v{});", dst, a),
        KirOp::Tanh(dst, a) => format!("auto v{} = tanh(v{});", dst, a),
        KirOp::ThreadId(dst, _) => format!("auto v{} = tid;", dst),
        KirOp::BlockIdx(dst, _) => format!("auto v{} = block_id;", dst),
        KirOp::GlobalId(dst, _) => format!("auto v{} = tid;  // Metal: global ID is thread_position_in_grid", dst),
        KirOp::Barrier => "threadgroup_barrier(mem_flags::mem_threadgroup);".to_string(),
        KirOp::Load(dst, ptr, _) => format!("auto v{} = *(v{});", dst, ptr),
        KirOp::Store(ptr, val, _) => format!("*(v{}) = v{};", ptr, val),
        KirOp::Const(dst, c) => format!("auto v{} = {:?};", dst, c),
        _ => "// unhandled op".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::KirBuilder;

    #[test]
    fn metal_empty_kernel() {
        let mut builder = KirBuilder::new("test_empty");
        let entry = builder.new_block();
        builder.set_block(entry);
        builder.terminate(KirTerminator::Return);
        let ir = builder.finalize();
        let msl = lower_kir_to_msl(&ir);
        let text = String::from_utf8_lossy(&msl);
        assert!(text.contains("#include <metal_stdlib>"));
        assert!(text.contains("kernel void test_empty"));
        assert!(text.contains("thread_position_in_grid"));
    }

    #[test]
    fn metal_barrier_emits_threadgroup_barrier() {
        let mut builder = KirBuilder::new("test_barrier");
        let entry = builder.new_block();
        builder.set_block(entry);
        builder.emit(KirOp::Barrier);
        builder.terminate(KirTerminator::Return);
        let ir = builder.finalize();
        let msl = lower_kir_to_msl(&ir);
        let text = String::from_utf8_lossy(&msl);
        assert!(text.contains("threadgroup_barrier"));
    }
}
