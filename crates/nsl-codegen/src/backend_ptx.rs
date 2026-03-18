// crates/nsl-codegen/src/backend_ptx.rs
//! M47: KIR -> PTX text emission backend.
//!
//! Lowers a `KernelIR` to null-terminated PTX text bytes suitable for
//! `cuModuleLoadData`. Uses PTX ISA 7.0 targeting sm_70.

use crate::kernel_ir::*;
use std::collections::HashMap;
use std::fmt::Write;

/// Lower a KernelIR to PTX text bytes (null-terminated).
pub fn lower_kir_to_ptx(ir: &KernelIR) -> Vec<u8> {
    let mut ptx = String::new();

    // Header
    writeln!(ptx, ".version 7.0").unwrap();
    writeln!(ptx, ".target sm_70").unwrap();
    writeln!(ptx, ".address_size 64").unwrap();
    writeln!(ptx).unwrap();

    // Shared memory declaration
    if ir.shared_mem_bytes > 0 {
        writeln!(ptx, ".shared .align 4 .b8 shared_mem[{}];", ir.shared_mem_bytes).unwrap();
        writeln!(ptx).unwrap();
    }

    // Entry point
    write!(ptx, ".visible .entry {}(", ir.name).unwrap();
    for (i, param) in ir.params.iter().enumerate() {
        if i > 0 {
            write!(ptx, ", ").unwrap();
        }
        let ptx_type = match &param.ty {
            KirType::Ptr(_, _) => ".u64",
            KirType::U32 => ".u32",
            KirType::I32 => ".s32",
            KirType::F32 => ".f32",
            KirType::F64 => ".f64",
            KirType::U64 => ".u64",
            KirType::I64 => ".s64",
            _ => ".u32",
        };
        write!(ptx, ".param {} param_{}", ptx_type, param.name).unwrap();
    }
    writeln!(ptx, ") {{").unwrap();

    // Pre-scan IR to count registers by type
    let mut reg_counts: HashMap<&str, u32> = HashMap::new();
    // Count all vars from var_types map
    for ty in ir.var_types.values() {
        let prefix = ty.ptx_reg_prefix();
        let entry = reg_counts.entry(prefix).or_insert(0);
        *entry += 1;
    }
    // Also count vars that appear in ops but may not be in var_types
    // (e.g., untyped vars from new_var() -- default to u32)
    let max_var = ir.blocks.iter()
        .flat_map(|b| b.ops.iter())
        .flat_map(extract_var_ids)
        .max()
        .unwrap_or(0);
    // Ensure we have enough registers for all variables
    let total_vars = std::cmp::max(max_var + 1, ir.params.len() as u32 + count_body_vars(ir));
    // Declare enough registers of each type
    let r_count = std::cmp::max(*reg_counts.get("%r").unwrap_or(&0), total_vars);
    let rd_count = std::cmp::max(*reg_counts.get("%rd").unwrap_or(&0), total_vars);
    let f_count = std::cmp::max(*reg_counts.get("%f").unwrap_or(&0), total_vars);
    let fd_count = *reg_counts.get("%fd").unwrap_or(&0);
    let h_count = *reg_counts.get("%h").unwrap_or(&0);
    let p_count = total_vars; // predicates

    if r_count > 0 { writeln!(ptx, "    .reg .u32 %r<{}>;", r_count).unwrap(); }
    if rd_count > 0 { writeln!(ptx, "    .reg .u64 %rd<{}>;", rd_count).unwrap(); }
    if f_count > 0 { writeln!(ptx, "    .reg .f32 %f<{}>;", f_count).unwrap(); }
    if fd_count > 0 { writeln!(ptx, "    .reg .f64 %fd<{}>;", fd_count).unwrap(); }
    if h_count > 0 { writeln!(ptx, "    .reg .f16 %h<{}>;", h_count).unwrap(); }
    if p_count > 0 { writeln!(ptx, "    .reg .pred %p<{}>;", p_count).unwrap(); }
    writeln!(ptx).unwrap();

    // Load parameters into registers
    for param in &ir.params {
        match &param.ty {
            KirType::Ptr(_, _) => {
                writeln!(ptx, "    ld.param.u64 %rd{}, [param_{}];", param.id, param.name).unwrap();
            }
            KirType::U32 => {
                writeln!(ptx, "    ld.param.u32 %r{}, [param_{}];", param.id, param.name).unwrap();
            }
            KirType::I32 => {
                writeln!(ptx, "    ld.param.s32 %r{}, [param_{}];", param.id, param.name).unwrap();
            }
            KirType::F32 => {
                writeln!(ptx, "    ld.param.f32 %f{}, [param_{}];", param.id, param.name).unwrap();
            }
            KirType::F64 => {
                writeln!(ptx, "    ld.param.f64 %fd{}, [param_{}];", param.id, param.name).unwrap();
            }
            _ => {
                writeln!(ptx, "    ld.param.u32 %r{}, [param_{}];", param.id, param.name).unwrap();
            }
        }
    }
    writeln!(ptx).unwrap();

    // Emit blocks
    for block in &ir.blocks {
        writeln!(ptx, "BB{}:", block.id).unwrap();
        for op in &block.ops {
            emit_op(&mut ptx, op, ir);
        }
        if let Some(ref term) = block.terminator {
            emit_terminator(&mut ptx, term);
        }
    }

    writeln!(ptx, "}}").unwrap();

    // Null-terminate
    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

fn emit_op(ptx: &mut String, op: &KirOp, ir: &KernelIR) {
    match op {
        KirOp::Add(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(ptx, "    add.{} {}{}, {}{}, {}{};", ty, prefix, dst, prefix, a, prefix, b).unwrap();
        }
        KirOp::Sub(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(ptx, "    sub.{} {}{}, {}{}, {}{};", ty, prefix, dst, prefix, a, prefix, b).unwrap();
        }
        KirOp::Mul(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(ptx, "    mul.lo.{} {}{}, {}{}, {}{};", ty, prefix, dst, prefix, a, prefix, b).unwrap();
        }
        KirOp::Div(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(ptx, "    div.{} {}{}, {}{}, {}{};", ty, prefix, dst, prefix, a, prefix, b).unwrap();
        }
        KirOp::Fma(dst, a, b, c) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(ptx, "    fma.rn.{} {}{}, {}{}, {}{}, {}{};", ty, prefix, dst, prefix, a, prefix, b, prefix, c).unwrap();
        }
        KirOp::Neg(dst, src) => {
            let ty = var_ptx_type(ir, *dst, *src);
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(ptx, "    neg.{} {}{}, {}{};", ty, prefix, dst, prefix, src).unwrap();
        }
        KirOp::Abs(dst, src) => {
            let ty = var_ptx_type(ir, *dst, *src);
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(ptx, "    abs.{} {}{}, {}{};", ty, prefix, dst, prefix, src).unwrap();
        }
        KirOp::Sqrt(dst, src) => {
            let ty = var_ptx_type(ir, *dst, *src);
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(ptx, "    sqrt.rn.{} {}{}, {}{};", ty, prefix, dst, prefix, src).unwrap();
        }
        KirOp::Exp(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(ptx, "    ex2.approx.f32 {}{}, {}{};", prefix, dst, prefix, src).unwrap();
        }
        KirOp::Log(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(ptx, "    lg2.approx.f32 {}{}, {}{};", prefix, dst, prefix, src).unwrap();
        }
        KirOp::Sin(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(ptx, "    sin.approx.f32 {}{}, {}{};", prefix, dst, prefix, src).unwrap();
        }
        KirOp::Cos(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(ptx, "    cos.approx.f32 {}{}, {}{};", prefix, dst, prefix, src).unwrap();
        }
        KirOp::Tanh(dst, _src) => {
            // PTX has no native tanh; emitted as a sequence, but for KIR we emit a placeholder call
            let prefix = var_reg_prefix(ir, *dst, *_src);
            // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1) -- simplified as a call stub
            writeln!(ptx, "    // tanh: not native in PTX, requires expansion").unwrap();
            writeln!(ptx, "    mov.f32 {}{}, {}{};", prefix, dst, prefix, _src).unwrap();
        }
        KirOp::Pow(dst, base, exp) => {
            let prefix = var_reg_prefix(ir, *dst, *base);
            // pow(a, b) = exp2(b * log2(a))
            writeln!(ptx, "    lg2.approx.f32 {}{}, {}{};", prefix, dst, prefix, base).unwrap();
            writeln!(ptx, "    mul.f32 {}{}, {}{}, {}{};", prefix, dst, prefix, dst, prefix, exp).unwrap();
            writeln!(ptx, "    ex2.approx.f32 {}{}, {}{};", prefix, dst, prefix, dst).unwrap();
        }
        KirOp::Cast(dst, src, target_ty) => {
            let src_ty = var_ptx_type(ir, *src, *src);
            let dst_ty = target_ty.ptx_type();
            let src_prefix = var_reg_prefix(ir, *src, *src);
            let dst_prefix = target_ty.ptx_reg_prefix();
            writeln!(ptx, "    cvt.{}.{} {}{}, {}{};", dst_ty, src_ty, dst_prefix, dst, src_prefix, src).unwrap();
        }
        KirOp::Load(dst, ptr, addr_space) => {
            let space = address_space_str(*addr_space);
            let ty = var_ptx_type(ir, *dst, *dst);
            let dst_prefix = var_reg_prefix(ir, *dst, *dst);
            let ptr_prefix = var_reg_prefix(ir, *ptr, *ptr);
            writeln!(ptx, "    ld.{}.{} {}{}, [{}{}];", space, ty, dst_prefix, dst, ptr_prefix, ptr).unwrap();
        }
        KirOp::Store(ptr, val, addr_space) => {
            let space = address_space_str(*addr_space);
            let ty = var_ptx_type(ir, *val, *val);
            let val_prefix = var_reg_prefix(ir, *val, *val);
            let ptr_prefix = var_reg_prefix(ir, *ptr, *ptr);
            writeln!(ptx, "    st.{}.{} [{}{}], {}{};", space, ty, ptr_prefix, ptr, val_prefix, val).unwrap();
        }
        KirOp::AtomicAdd(ptr, val, addr_space) => {
            let space = address_space_str(*addr_space);
            let ty = var_ptx_type(ir, *val, *val);
            let val_prefix = var_reg_prefix(ir, *val, *val);
            let ptr_prefix = var_reg_prefix(ir, *ptr, *ptr);
            writeln!(ptx, "    atom.{}.add.{} {}{}, [{}{}], {}{};", space, ty, val_prefix, val, ptr_prefix, ptr, val_prefix, val).unwrap();
        }
        KirOp::ThreadId(dst, dim) => {
            let dim_name = dim_char(*dim);
            writeln!(ptx, "    mov.u32 %r{}, %tid.{};", dst, dim_name).unwrap();
        }
        KirOp::BlockIdx(dst, dim) => {
            let dim_name = dim_char(*dim);
            writeln!(ptx, "    mov.u32 %r{}, %ctaid.{};", dst, dim_name).unwrap();
        }
        KirOp::BlockDim(dst, dim) => {
            let dim_name = dim_char(*dim);
            writeln!(ptx, "    mov.u32 %r{}, %ntid.{};", dst, dim_name).unwrap();
        }
        KirOp::GridDim(dst, dim) => {
            let dim_name = dim_char(*dim);
            writeln!(ptx, "    mov.u32 %r{}, %nctaid.{};", dst, dim_name).unwrap();
        }
        KirOp::GlobalId(dst, dim) => {
            // GlobalId = blockIdx * blockDim + threadIdx
            // IMPORTANT: Use mul.lo.u32 + add.u32, NOT mad.lo.u32 (causes INVALID_PTX on ISA 7.0)
            let dim_name = dim_char(*dim);
            // Use a temporary register for the intermediate multiply
            let tmp = dst + 1000; // offset to avoid collision
            writeln!(ptx, "    mov.u32 %r{}, %ctaid.{};", tmp, dim_name).unwrap();
            writeln!(ptx, "    mov.u32 %r{}, %ntid.{};", tmp + 1, dim_name).unwrap();
            writeln!(ptx, "    mul.lo.u32 %r{}, %r{}, %r{};", tmp, tmp, tmp + 1).unwrap();
            writeln!(ptx, "    mov.u32 %r{}, %tid.{};", dst, dim_name).unwrap();
            writeln!(ptx, "    add.u32 %r{}, %r{}, %r{};", dst, tmp, dst).unwrap();
        }
        KirOp::Barrier => {
            writeln!(ptx, "    bar.sync 0;").unwrap();
        }
        KirOp::WarpShuffle(dst, val, offset) => {
            let prefix = var_reg_prefix(ir, *dst, *val);
            writeln!(ptx, "    shfl.sync.down.b32 {}{}, {}{}, %r{}, 0x1f, 0xffffffff;",
                prefix, dst, prefix, val, offset).unwrap();
        }
        KirOp::Cmp(dst, a, b, cmp_op) => {
            let ty = var_ptx_type(ir, *a, *a);
            let prefix = var_reg_prefix(ir, *a, *a);
            let op_str = match cmp_op {
                CmpOp::Eq => "eq",
                CmpOp::Ne => "ne",
                CmpOp::Lt => "lt",
                CmpOp::Le => "le",
                CmpOp::Gt => "gt",
                CmpOp::Ge => "ge",
            };
            writeln!(ptx, "    setp.{}.{} %p{}, {}{}, {}{};", op_str, ty, dst, prefix, a, prefix, b).unwrap();
        }
        KirOp::Select(dst, cond, true_val, false_val) => {
            let prefix = var_reg_prefix(ir, *dst, *true_val);
            writeln!(ptx, "    selp.b32 {}{}, {}{}, {}{}, %p{};", prefix, dst, prefix, true_val, prefix, false_val, cond).unwrap();
        }
        KirOp::Const(dst, konst) => {
            match &konst.value {
                ConstValue::U32(v) => writeln!(ptx, "    mov.u32 %r{}, {};", dst, v).unwrap(),
                ConstValue::I32(v) => writeln!(ptx, "    mov.s32 %r{}, {};", dst, v).unwrap(),
                ConstValue::U64(v) => writeln!(ptx, "    mov.u64 %rd{}, {};", dst, v).unwrap(),
                ConstValue::I64(v) => writeln!(ptx, "    mov.s64 %rd{}, {};", dst, v).unwrap(),
                ConstValue::F32(v) => writeln!(ptx, "    mov.f32 %f{}, 0f{:08X};", dst, v.to_bits()).unwrap(),
                ConstValue::F64(v) => writeln!(ptx, "    mov.f64 %fd{}, 0d{:016X};", dst, v.to_bits()).unwrap(),
                ConstValue::Bool(v) => writeln!(ptx, "    setp.eq.u32 %p{}, 1, {};", dst, if *v { 1 } else { 0 }).unwrap(),
            }
        }
        KirOp::PtrOffset(dst, base, offset) => {
            // dst = base + offset * sizeof(pointee)
            // Widen offset to 64-bit, multiply by element size, add to base pointer
            let pointee_size = if let Some(KirType::Ptr(inner, _)) = ir.var_types.get(dst) {
                inner.size_bytes() as u32
            } else if let Some(KirType::Ptr(inner, _)) = ir.var_types.get(base) {
                inner.size_bytes() as u32
            } else {
                4 // default f32
            };
            writeln!(ptx, "    cvt.u64.u32 %rd{}, %r{};", dst, offset).unwrap();
            if pointee_size > 1 {
                writeln!(ptx, "    mul.lo.u64 %rd{}, %rd{}, {};", dst, dst, pointee_size).unwrap();
            }
            writeln!(ptx, "    add.u64 %rd{}, %rd{}, %rd{};", dst, base, dst).unwrap();
        }
        KirOp::SharedMemFence => {
            writeln!(ptx, "    membar.cta;").unwrap();
        }
    }
}

fn emit_terminator(ptx: &mut String, term: &KirTerminator) {
    match term {
        KirTerminator::Branch(target) => {
            writeln!(ptx, "    bra BB{};", target).unwrap();
        }
        KirTerminator::CondBranch(cond, true_bb, false_bb) => {
            writeln!(ptx, "    @%p{} bra BB{};", cond, true_bb).unwrap();
            writeln!(ptx, "    bra BB{};", false_bb).unwrap();
        }
        KirTerminator::Return => {
            writeln!(ptx, "    ret;").unwrap();
        }
    }
}

/// Get the PTX type string for a variable, looking up in var_types.
fn var_ptx_type(ir: &KernelIR, primary: VarId, fallback: VarId) -> &'static str {
    if let Some(ty) = ir.var_types.get(&primary) {
        return ty.ptx_type();
    }
    if let Some(ty) = ir.var_types.get(&fallback) {
        return ty.ptx_type();
    }
    "u32" // default
}

/// Get the PTX register prefix for a variable.
fn var_reg_prefix(ir: &KernelIR, primary: VarId, fallback: VarId) -> &'static str {
    if let Some(ty) = ir.var_types.get(&primary) {
        return ty.ptx_reg_prefix();
    }
    if let Some(ty) = ir.var_types.get(&fallback) {
        return ty.ptx_reg_prefix();
    }
    "%r" // default u32
}

fn address_space_str(space: AddressSpace) -> &'static str {
    match space {
        AddressSpace::Global => "global",
        AddressSpace::Shared => "shared",
        AddressSpace::Local => "local",
        AddressSpace::Constant => "const",
    }
}

fn dim_char(dim: u8) -> char {
    match dim {
        0 => 'x',
        1 => 'y',
        2 => 'z',
        _ => 'x',
    }
}

/// Extract all VarIds referenced by a KirOp.
fn extract_var_ids(op: &KirOp) -> Vec<VarId> {
    match op {
        KirOp::Add(d, a, b) | KirOp::Sub(d, a, b) | KirOp::Mul(d, a, b)
        | KirOp::Div(d, a, b) | KirOp::Pow(d, a, b) => vec![*d, *a, *b],
        KirOp::Fma(d, a, b, c) | KirOp::Select(d, a, b, c) => vec![*d, *a, *b, *c],
        KirOp::Neg(d, s) | KirOp::Abs(d, s) | KirOp::Sqrt(d, s)
        | KirOp::Exp(d, s) | KirOp::Log(d, s) | KirOp::Sin(d, s)
        | KirOp::Cos(d, s) | KirOp::Tanh(d, s) => vec![*d, *s],
        KirOp::Cast(d, s, _) => vec![*d, *s],
        KirOp::Load(d, p, _) | KirOp::Store(d, p, _) | KirOp::AtomicAdd(d, p, _) => vec![*d, *p],
        KirOp::ThreadId(d, _) | KirOp::BlockIdx(d, _) | KirOp::BlockDim(d, _)
        | KirOp::GridDim(d, _) | KirOp::GlobalId(d, _) => vec![*d],
        KirOp::Barrier | KirOp::SharedMemFence => vec![],
        KirOp::WarpShuffle(d, v, o) => vec![*d, *v, *o],
        KirOp::Cmp(d, a, b, _) | KirOp::PtrOffset(d, a, b) => vec![*d, *a, *b],
        KirOp::Const(d, _) => vec![*d],
    }
}

/// Count the number of body variables (non-param) in the IR.
fn count_body_vars(ir: &KernelIR) -> u32 {
    let mut max_id: u32 = 0;
    for block in &ir.blocks {
        for op in &block.ops {
            for id in extract_var_ids(op) {
                if id > max_id {
                    max_id = id;
                }
            }
        }
    }
    max_id + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_add_kernel() -> KernelIR {
        let mut b = KirBuilder::new("test_add");
        let a_ptr = b.add_param("a", KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global), AddressSpace::Global);
        let b_ptr = b.add_param("b", KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global), AddressSpace::Global);
        let out_ptr = b.add_param("out", KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global), AddressSpace::Global);
        let len = b.add_param("len", KirType::U32, AddressSpace::Local);

        let entry = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();

        b.set_block(entry);
        let tid = b.new_var();
        b.emit(KirOp::GlobalId(tid, 0));
        let in_bounds = b.new_var();
        b.emit(KirOp::Cmp(in_bounds, tid, len, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(in_bounds, body, exit));

        b.set_block(body);
        let a_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        b.emit(KirOp::PtrOffset(a_addr, a_ptr, tid));
        let a_val = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(a_val, a_addr, AddressSpace::Global));
        let b_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        b.emit(KirOp::PtrOffset(b_addr, b_ptr, tid));
        let b_val = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(b_val, b_addr, AddressSpace::Global));
        let sum = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Add(sum, a_val, b_val));
        let out_addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global));
        b.emit(KirOp::PtrOffset(out_addr, out_ptr, tid));
        b.emit(KirOp::Store(out_addr, sum, AddressSpace::Global));
        b.terminate(KirTerminator::Branch(exit));

        b.set_block(exit);
        b.terminate(KirTerminator::Return);

        b.set_workgroup_size([256, 1, 1]);
        b.finalize()
    }

    #[test]
    fn test_simple_add_ptx() {
        let ir = build_simple_add_kernel();
        let ptx_bytes = lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]); // exclude null

        assert!(ptx.contains(".version 7.0"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".visible .entry test_add"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ld.global.f32"));
        assert!(ptx.contains("st.global.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_ptx_thread_indexing() {
        // GlobalId must emit mul.lo.u32 + add.u32, NOT mad.lo.u32
        let mut b = KirBuilder::new("test_indexing");
        let entry = b.new_block();
        b.set_block(entry);
        let tid = b.new_var();
        b.emit(KirOp::GlobalId(tid, 0));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();

        let ptx_bytes = lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]);

        assert!(ptx.contains("mul.lo.u32"), "GlobalId must use mul.lo.u32");
        assert!(ptx.contains("add.u32"), "GlobalId must use add.u32");
        assert!(!ptx.contains("mad.lo.u32"), "GlobalId must NOT use mad.lo.u32 (INVALID_PTX on ISA 7.0)");
    }

    #[test]
    fn test_ptx_shared_memory_declaration() {
        let mut b = KirBuilder::new("test_shared");
        b.set_shared_mem(1024);
        let entry = b.new_block();
        b.set_block(entry);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();

        let ptx_bytes = lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]);

        assert!(ptx.contains(".shared .align 4 .b8 shared_mem[1024]"));
    }

    #[test]
    fn test_ptx_null_terminated() {
        let mut b = KirBuilder::new("test_null");
        let entry = b.new_block();
        b.set_block(entry);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();

        let ptx_bytes = lower_kir_to_ptx(&ir);
        assert_eq!(*ptx_bytes.last().unwrap(), 0u8, "PTX output must be null-terminated");
    }

    #[test]
    fn test_ptx_barrier() {
        let mut b = KirBuilder::new("test_barrier");
        let entry = b.new_block();
        b.set_block(entry);
        b.emit(KirOp::Barrier);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();

        let ptx_bytes = lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]);

        assert!(ptx.contains("bar.sync 0;"));
    }
}
