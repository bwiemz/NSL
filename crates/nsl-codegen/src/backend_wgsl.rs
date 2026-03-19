// crates/nsl-codegen/src/backend_wgsl.rs
//! M47b: KIR -> WGSL (WebGPU Shading Language) for browser deployment.
//!
//! The most constrained backend:
//! - No shared memory atomics on floats
//! - No warp/subgroup shuffle (subgroup ops are an extension)
//! - Workgroup size limited to 256 in practice
//! - No f64 support
//! - No pointer arithmetic — arrays only

use crate::kernel_ir::*;
use std::fmt::Write;

/// Lower a KernelIR to WGSL source code bytes.
pub fn lower_kir_to_wgsl(ir: &KernelIR) -> Vec<u8> {
    let mut wgsl = String::new();

    // Bind group declarations for each buffer parameter
    for (i, param) in ir.params.iter().enumerate() {
        if matches!(param.ty, KirType::Ptr(_, _)) {
            let access = match param.address_space {
                AddressSpace::Global => "read_write",
                AddressSpace::Constant => "read",
                _ => "read_write",
            };
            let inner_type = match &param.ty {
                KirType::Ptr(inner, _) => kir_type_to_wgsl(inner),
                _ => "f32",
            };
            writeln!(wgsl, "@group(0) @binding({}) var<storage, {}> {}: array<{}>;",
                i, access, param.name, inner_type).unwrap();
        }
    }
    writeln!(wgsl).unwrap();

    // Compute shader entry
    let wg = ir.workgroup_size;
    writeln!(wgsl, "@compute @workgroup_size({}, {}, {})", wg[0], wg[1], wg[2]).unwrap();
    writeln!(wgsl, "fn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{", ir.name).unwrap();
    writeln!(wgsl, "    let tid = gid.x;").unwrap();

    // Body
    for block in &ir.blocks {
        for op in &block.ops {
            writeln!(wgsl, "    {}", lower_op_to_wgsl(op)).unwrap();
        }
    }

    writeln!(wgsl, "}}").unwrap();

    wgsl.into_bytes()
}

fn kir_type_to_wgsl(ty: &KirType) -> &'static str {
    match ty {
        KirType::U32 => "u32",
        KirType::I32 => "i32",
        KirType::F32 => "f32",
        KirType::F16 => "f16",
        // WebGPU has no f64 or i64/u64 in storage buffers
        KirType::F64 => "f32",  // downgrade
        KirType::U64 => "u32",  // downgrade
        KirType::I64 => "i32",  // downgrade
        _ => "f32",
    }
}

fn lower_op_to_wgsl(op: &KirOp) -> String {
    match op {
        KirOp::Add(dst, a, b) => format!("var v{} = v{} + v{};", dst, a, b),
        KirOp::Sub(dst, a, b) => format!("var v{} = v{} - v{};", dst, a, b),
        KirOp::Mul(dst, a, b) => format!("var v{} = v{} * v{};", dst, a, b),
        KirOp::Div(dst, a, b) => format!("var v{} = v{} / v{};", dst, a, b),
        KirOp::Fma(dst, a, b, c) => format!("var v{} = fma(v{}, v{}, v{});", dst, a, b, c),
        KirOp::Neg(dst, a) => format!("var v{} = -v{};", dst, a),
        KirOp::Abs(dst, a) => format!("var v{} = abs(v{});", dst, a),
        KirOp::Sqrt(dst, a) => format!("var v{} = sqrt(v{});", dst, a),
        KirOp::Exp(dst, a) => format!("var v{} = exp(v{});", dst, a),
        KirOp::Log(dst, a) => format!("var v{} = log(v{});", dst, a),
        KirOp::Tanh(dst, a) => format!("var v{} = tanh(v{});", dst, a),
        KirOp::ThreadId(dst, _) => format!("var v{} = tid;", dst),
        KirOp::BlockIdx(dst, _) => format!("var v{} = gid.y;  // WGSL: use y component for block-like index", dst),
        KirOp::GlobalId(dst, _) => format!("var v{} = tid;", dst),
        KirOp::Barrier => "workgroupBarrier();".to_string(),
        KirOp::Load(dst, ptr, _) => format!("var v{} = v{}[tid];  // WGSL: array access", dst, ptr),
        KirOp::Store(ptr, val, _) => format!("v{}[tid] = v{};  // WGSL: array store", ptr, val),
        KirOp::Const(dst, c) => {
            match &c.value {
                ConstValue::F32(v) => format!("var v{}: f32 = {:.6};", dst, v),
                ConstValue::U32(v) => format!("var v{}: u32 = {}u;", dst, v),
                ConstValue::I32(v) => format!("var v{}: i32 = {};", dst, v),
                ConstValue::F64(v) => format!("var v{}: f32 = {:.6};  // WGSL: f64 downgraded to f32", dst, v),
                _ => format!("// const v{}", dst),
            }
        }
        KirOp::WarpShuffle(_, _, _) => {
            "// ERROR: WGSL does not support warp shuffle — feature gate should have caught this".to_string()
        }
        _ => "// unhandled op".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::KirBuilder;

    #[test]
    fn wgsl_empty_kernel() {
        let mut builder = KirBuilder::new("test_empty");
        let entry = builder.new_block();
        builder.set_block(entry);
        builder.terminate(KirTerminator::Return);
        let ir = builder.finalize();
        let wgsl = lower_kir_to_wgsl(&ir);
        let text = String::from_utf8_lossy(&wgsl);
        assert!(text.contains("@compute @workgroup_size"));
        assert!(text.contains("fn test_empty"));
        assert!(text.contains("global_invocation_id"));
    }

    #[test]
    fn wgsl_barrier_emits_workgroup_barrier() {
        let mut builder = KirBuilder::new("test_barrier");
        let entry = builder.new_block();
        builder.set_block(entry);
        builder.emit(KirOp::Barrier);
        builder.terminate(KirTerminator::Return);
        let ir = builder.finalize();
        let wgsl = lower_kir_to_wgsl(&ir);
        let text = String::from_utf8_lossy(&wgsl);
        assert!(text.contains("workgroupBarrier()"));
    }

    #[test]
    fn wgsl_no_f64_types() {
        let text = kir_type_to_wgsl(&KirType::F64);
        assert_eq!(text, "f32"); // downgraded
    }
}
