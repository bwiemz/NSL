// crates/nsl-codegen/src/backend_amdgpu.rs
//! M47b: KIR -> AMDGPU assembly (GCN/CDNA ISA) for ROCm.
//!
//! Produces AMDGPU assembly text that can be assembled via `amd-comgr`.
//! Key differences from PTX:
//! - Wavefront size is 64 (not 32)
//! - Shared memory is LDS (Local Data Share)
//! - Thread indexing via workgroup/workitem IDs
//! - Matrix cores (CDNA) use `v_mfma_*` instructions

use crate::kernel_ir::*;
use std::fmt::Write;

/// Lower a KernelIR to AMDGPU assembly text bytes.
pub fn lower_kir_to_amdgpu(ir: &KernelIR) -> Vec<u8> {
    let mut asm = String::new();

    // AMDGPU kernel metadata
    writeln!(asm, ".amdgcn_target \"amdgcn-amd-amdhsa--gfx90a\"").unwrap();
    writeln!(asm, ".text").unwrap();
    writeln!(asm, ".globl {}", ir.name).unwrap();
    writeln!(asm, ".p2align 8").unwrap();
    writeln!(asm, ".type {},@function", ir.name).unwrap();
    writeln!(asm, "{}:", ir.name).unwrap();
    writeln!(asm).unwrap();

    // Prologue: load kernel arguments from kernarg segment
    for (i, param) in ir.params.iter().enumerate() {
        let offset = i * 8; // 8 bytes per param (all 64-bit aligned)
        writeln!(asm, "  ; param {}: {} (offset {})", param.name, param.ty.amdgpu_type(), offset).unwrap();
        writeln!(asm, "  s_load_dwordx2 s[{}:{}], s[4:5], {:#x}",
            i * 2, i * 2 + 1, offset).unwrap();
    }
    writeln!(asm, "  s_waitcnt lgkmcnt(0)").unwrap();
    writeln!(asm).unwrap();

    // Body: lower each block
    for block in &ir.blocks {
        writeln!(asm, ".LBB{}:", block.id).unwrap();
        for op in &block.ops {
            writeln!(asm, "  {}", lower_op_to_amdgpu(op, ir)).unwrap();
        }
        if let Some(ref term) = block.terminator {
            writeln!(asm, "  {}", lower_terminator_amdgpu(term)).unwrap();
        }
    }

    writeln!(asm, "  s_endpgm").unwrap();
    writeln!(asm, ".size {}, .-{}", ir.name, ir.name).unwrap();

    asm.into_bytes()
}

fn lower_op_to_amdgpu(op: &KirOp, ir: &KernelIR) -> String {
    match op {
        KirOp::Add(dst, a, b) => format!("; add v{}, v{}, v{}", dst, a, b),
        KirOp::Sub(dst, a, b) => format!("; sub v{}, v{}, v{}", dst, a, b),
        KirOp::Mul(dst, a, b) => format!("; mul v{}, v{}, v{}", dst, a, b),
        KirOp::Div(dst, a, b) => format!("; div v{}, v{}, v{}", dst, a, b),
        KirOp::Fma(dst, a, b, c) => format!("v_fma_f32 v{}, v{}, v{}, v{}", dst, a, b, c),
        KirOp::Neg(dst, a) => format!("; neg v{}, v{}", dst, a),
        KirOp::Abs(dst, a) => format!("; abs v{}, v{}", dst, a),
        KirOp::Sqrt(dst, a) => format!("v_sqrt_f32 v{}, v{}", dst, a),
        KirOp::Exp(dst, a) => format!("v_exp_f32 v{}, v{}", dst, a),
        KirOp::Log(dst, a) => format!("v_log_f32 v{}, v{}", dst, a),
        KirOp::Tanh(dst, a) => format!("; tanh v{}, v{} (emulated)", dst, a),
        KirOp::ThreadId(dst, dim) => {
            // AMDGPU: workitem ID in v0 (x), v1 (y), v2 (z)
            format!("v_mov_b32 v{}, v{}", dst, dim)
        }
        KirOp::BlockIdx(dst, dim) => {
            // AMDGPU: workgroup ID via s_* registers
            format!("; blockId dim={} -> v{}", dim, dst)
        }
        KirOp::GlobalId(dst, dim) => {
            let _ = ir;
            format!("; globalId dim={} -> v{} (workgroup_id * workgroup_size + local_id)", dim, dst)
        }
        KirOp::Barrier => "s_barrier".to_string(),
        KirOp::WarpShuffle(dst, val, offset) => {
            // AMDGPU wavefront size is 64; use ds_permute for shuffle
            format!("ds_permute_b32 v{}, v{}, v{}", dst, offset, val)
        }
        KirOp::Load(dst, ptr, space) => {
            let prefix = match space {
                AddressSpace::Global => "flat_load",
                AddressSpace::Shared => "ds_read",
                _ => "flat_load",
            };
            format!("{}_dword v{}, v[{}:{}]", prefix, dst, ptr, ptr + 1)
        }
        KirOp::Store(ptr, val, space) => {
            let prefix = match space {
                AddressSpace::Global => "flat_store",
                AddressSpace::Shared => "ds_write",
                _ => "flat_store",
            };
            format!("{}_dword v[{}:{}], v{}", prefix, ptr, ptr + 1, val)
        }
        KirOp::Const(dst, c) => format!("; const v{} = {:?}", dst, c),
        _ => "; unhandled op".to_string(),
    }
}

fn lower_terminator_amdgpu(term: &KirTerminator) -> String {
    match term {
        KirTerminator::Branch(target) => format!("s_branch .LBB{}", target),
        KirTerminator::CondBranch(pred, t, f) => {
            format!("s_cbranch_vccnz .LBB{} ; pred=v{}, else .LBB{}", t, pred, f)
        }
        KirTerminator::Return => "s_endpgm".to_string(),
    }
}

/// Helper trait for AMDGPU type names.
trait AmdgpuType {
    fn amdgpu_type(&self) -> &'static str;
}

impl AmdgpuType for KirType {
    fn amdgpu_type(&self) -> &'static str {
        match self {
            KirType::U32 | KirType::I32 => "dword",
            KirType::U64 | KirType::I64 => "dwordx2",
            KirType::F32 => "dword",
            KirType::F64 => "dwordx2",
            KirType::Ptr(_, _) => "dwordx2",
            _ => "dword",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::KirBuilder;

    #[test]
    fn amdgpu_empty_kernel() {
        let mut builder = KirBuilder::new("test_empty");
        let entry = builder.new_block();
        builder.set_block(entry);
        builder.terminate(KirTerminator::Return);
        let ir = builder.finalize();
        let asm = lower_kir_to_amdgpu(&ir);
        let text = String::from_utf8_lossy(&asm);
        assert!(text.contains("amdgcn_target"));
        assert!(text.contains("test_empty:"));
        assert!(text.contains("s_endpgm"));
    }

    #[test]
    fn amdgpu_barrier_emits_s_barrier() {
        let mut builder = KirBuilder::new("test_barrier");
        let entry = builder.new_block();
        builder.set_block(entry);
        builder.emit(KirOp::Barrier);
        builder.terminate(KirTerminator::Return);
        let ir = builder.finalize();
        let asm = lower_kir_to_amdgpu(&ir);
        let text = String::from_utf8_lossy(&asm);
        assert!(text.contains("s_barrier"));
    }
}
