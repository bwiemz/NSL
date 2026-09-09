// crates/nsl-kir/src/backend_ptx.rs
//! M47: KIR -> PTX text emission backend.
//!
//! Lowers a `KernelIR` to null-terminated PTX text bytes suitable for
//! `cuModuleLoadData`. Uses PTX ISA 7.0 targeting sm_70.

use crate::FeatureSet;
use crate::kernel_ir::*;
use std::collections::HashMap;
use std::fmt::Write;

/// Lower a KernelIR to PTX text bytes (null-terminated).
pub fn lower_kir_to_ptx(ir: &KernelIR) -> Vec<u8> {
    let mut ptx = String::new();

    // Header. `cp.async` (FeatureSet::ASYNC_COPY) is an sm_80 instruction;
    // every other op lowers on sm_70, the floor this backend has always
    // targeted, so the bump is taken only when a kernel asks for it.
    writeln!(ptx, ".version 7.0").unwrap();
    // The tensor-core ops (`ldmatrix`, `mma.sync` m16n8k16 f16) are sm_80
    // instructions as well.
    let target = if ir.required_features.contains(FeatureSet::ASYNC_COPY)
        || ir.required_features.contains(FeatureSet::TENSOR_CORES)
    {
        "sm_80"
    } else {
        "sm_70"
    };
    writeln!(ptx, ".target {}", target).unwrap();
    writeln!(ptx, ".address_size 64").unwrap();
    writeln!(ptx).unwrap();

    // Shared memory declaration
    if ir.shared_mem_bytes > 0 {
        writeln!(
            ptx,
            ".shared .align 4 .b8 shared_mem[{}];",
            ir.shared_mem_bytes
        )
        .unwrap();
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
    let max_var = ir
        .blocks
        .iter()
        .flat_map(|b| b.ops.iter())
        .flat_map(extract_var_ids)
        .max()
        .unwrap_or(0);
    // GlobalId emits synthetic u32 temps at dst+1000 and dst+1001 to avoid
    // collision with normal VarIds (see emit_op GlobalId arm). Those indices
    // are not returned by extract_var_ids, so they are invisible to the
    // max_var scan above. Without an explicit floor the .reg .u32 %r<N>
    // declaration is too small and ptxas rejects the output.
    let global_id_reg_floor: u32 = ir
        .blocks
        .iter()
        .flat_map(|b| b.ops.iter())
        .filter_map(|op| {
            if let KirOp::GlobalId(d, _) = op {
                Some(d + 1002) // highest temp index is d+1001; need count d+1002
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0);
    // Ensure we have enough registers for all variables
    let total_vars = std::cmp::max(max_var + 1, ir.params.len() as u32 + count_body_vars(ir));
    let total_vars = std::cmp::max(total_vars, global_id_reg_floor);
    // Declare enough registers of each type
    let r_count = std::cmp::max(*reg_counts.get("%r").unwrap_or(&0), total_vars);
    let rd_count = std::cmp::max(*reg_counts.get("%rd").unwrap_or(&0), total_vars);
    let f_count = std::cmp::max(*reg_counts.get("%f").unwrap_or(&0), total_vars);
    let fd_count = *reg_counts.get("%fd").unwrap_or(&0);
    let h_count = *reg_counts.get("%h").unwrap_or(&0);
    let p_count = total_vars; // predicates
    // Packed fragments (`KirType::Vec`, one .b32 each): declared only when a
    // kernel has any, sized like the other classes so every VarId fits.
    let v_count = if reg_counts.contains_key("%v") { total_vars } else { 0 };

    if r_count > 0 {
        writeln!(ptx, "    .reg .u32 %r<{}>;", r_count).unwrap();
    }
    if rd_count > 0 {
        writeln!(ptx, "    .reg .u64 %rd<{}>;", rd_count).unwrap();
    }
    if f_count > 0 {
        writeln!(ptx, "    .reg .f32 %f<{}>;", f_count).unwrap();
    }
    if fd_count > 0 {
        writeln!(ptx, "    .reg .f64 %fd<{}>;", fd_count).unwrap();
    }
    if h_count > 0 {
        writeln!(ptx, "    .reg .f16 %h<{}>;", h_count).unwrap();
    }
    if p_count > 0 {
        writeln!(ptx, "    .reg .pred %p<{}>;", p_count).unwrap();
    }
    if v_count > 0 {
        writeln!(ptx, "    .reg .b32 %v<{}>;", v_count).unwrap();
    }
    // Roadmap A2 step 2: an edge that passes block arguments is a parallel
    // copy into the target's parameter registers. A cycle in that copy (a
    // swap) needs one scratch register of the class; declared only when a
    // kernel has block parameters, so every other kernel's text is unchanged.
    if ir.blocks.iter().any(|b| !b.params.is_empty()) {
        for (ty, name) in EDGE_SCRATCH {
            writeln!(ptx, "    .reg .{ty} {name};").unwrap();
        }
    }
    writeln!(ptx).unwrap();

    // Load parameters into registers
    for param in &ir.params {
        match &param.ty {
            KirType::Ptr(_, _) => {
                writeln!(
                    ptx,
                    "    ld.param.u64 %rd{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::U32 => {
                writeln!(
                    ptx,
                    "    ld.param.u32 %r{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::I32 => {
                writeln!(
                    ptx,
                    "    ld.param.s32 %r{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::F32 => {
                writeln!(
                    ptx,
                    "    ld.param.f32 %f{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::F64 => {
                writeln!(
                    ptx,
                    "    ld.param.f64 %fd{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            _ => {
                writeln!(
                    ptx,
                    "    ld.param.u32 %r{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
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
            emit_terminator(&mut ptx, term, ir, block.id);
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
            writeln!(
                ptx,
                "    add.{} {}{}, {}{}, {}{};",
                ty, prefix, dst, prefix, a, prefix, b
            )
            .unwrap();
        }
        KirOp::Sub(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(
                ptx,
                "    sub.{} {}{}, {}{}, {}{};",
                ty, prefix, dst, prefix, a, prefix, b
            )
            .unwrap();
        }
        KirOp::Mul(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(
                ptx,
                "    mul.lo.{} {}{}, {}{}, {}{};",
                ty, prefix, dst, prefix, a, prefix, b
            )
            .unwrap();
        }
        KirOp::Div(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(
                ptx,
                "    div.{} {}{}, {}{}, {}{};",
                ty, prefix, dst, prefix, a, prefix, b
            )
            .unwrap();
        }
        KirOp::Fma(dst, a, b, c) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(
                ptx,
                "    fma.rn.{} {}{}, {}{}, {}{}, {}{};",
                ty, prefix, dst, prefix, a, prefix, b, prefix, c
            )
            .unwrap();
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
            writeln!(
                ptx,
                "    sqrt.rn.{} {}{}, {}{};",
                ty, prefix, dst, prefix, src
            )
            .unwrap();
        }
        KirOp::Exp(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            // e^x = 2^(x * log2(e)); ex2.approx computes 2^x, so pre-scale
            // by log2(e) = 1.4426950408889634 (0f3FB8AA3B). src is read only
            // by the first instruction, so the sequence is safe when
            // dst == src.
            writeln!(
                ptx,
                "    mul.f32 {}{}, {}{}, 0f3FB8AA3B;",
                prefix, dst, prefix, src
            )
            .unwrap();
            writeln!(
                ptx,
                "    ex2.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, dst
            )
            .unwrap();
        }
        KirOp::Log(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            // ln(x) = log2(x) * ln(2); lg2.approx computes log2(x), so
            // post-scale by ln(2) = 0.6931471805599453 (0f3F317218). src is
            // read only by the first instruction, so the sequence is safe
            // when dst == src.
            writeln!(
                ptx,
                "    lg2.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, src
            )
            .unwrap();
            writeln!(
                ptx,
                "    mul.f32 {}{}, {}{}, 0f3F317218;",
                prefix, dst, prefix, dst
            )
            .unwrap();
        }
        KirOp::Sin(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(
                ptx,
                "    sin.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, src
            )
            .unwrap();
        }
        KirOp::Cos(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            writeln!(
                ptx,
                "    cos.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, src
            )
            .unwrap();
        }
        KirOp::Tanh(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            // PTX has no native tanh; expand as tanh(x) = 2*sigmoid(2x) - 1
            // = 2/(1 + exp(-2x)) - 1, with exp via ex2.approx and the base
            // conversion folded in (-2x * log2(e) = -2x * 0f3FB8AA3B).
            // Same instruction sequence as the proven "tanh" epilogue in
            // epilogue_fusion.rs. src is read only by the first instruction,
            // so the expansion is safe when dst == src.
            writeln!(
                ptx,
                "    add.f32 {}{}, {}{}, {}{};",
                prefix, dst, prefix, src, prefix, src
            )
            .unwrap();
            writeln!(ptx, "    neg.f32 {}{}, {}{};", prefix, dst, prefix, dst).unwrap();
            writeln!(
                ptx,
                "    mul.f32 {}{}, {}{}, 0f3FB8AA3B;",
                prefix, dst, prefix, dst
            )
            .unwrap();
            writeln!(
                ptx,
                "    ex2.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, dst
            )
            .unwrap();
            writeln!(
                ptx,
                "    add.f32 {}{}, {}{}, 0f3F800000;",
                prefix, dst, prefix, dst
            )
            .unwrap();
            writeln!(
                ptx,
                "    rcp.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, dst
            )
            .unwrap();
            writeln!(
                ptx,
                "    add.f32 {}{}, {}{}, {}{};",
                prefix, dst, prefix, dst, prefix, dst
            )
            .unwrap();
            writeln!(
                ptx,
                "    sub.f32 {}{}, {}{}, 0f3F800000;",
                prefix, dst, prefix, dst
            )
            .unwrap();
        }
        KirOp::Pow(dst, base, exp) => {
            let prefix = var_reg_prefix(ir, *dst, *base);
            // pow(a, b) = exp2(b * log2(a))
            writeln!(
                ptx,
                "    lg2.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, base
            )
            .unwrap();
            writeln!(
                ptx,
                "    mul.f32 {}{}, {}{}, {}{};",
                prefix, dst, prefix, dst, prefix, exp
            )
            .unwrap();
            writeln!(
                ptx,
                "    ex2.approx.f32 {}{}, {}{};",
                prefix, dst, prefix, dst
            )
            .unwrap();
        }
        KirOp::Cast(dst, src, target_ty) => {
            let src_ty = var_ptx_type(ir, *src, *src);
            let dst_ty = target_ty.ptx_type();
            let src_prefix = var_reg_prefix(ir, *src, *src);
            let dst_prefix = target_ty.ptx_reg_prefix();
            writeln!(
                ptx,
                "    cvt.{}.{} {}{}, {}{};",
                dst_ty, src_ty, dst_prefix, dst, src_prefix, src
            )
            .unwrap();
        }
        KirOp::Load(dst, ptr, addr_space) => {
            let space = address_space_str(*addr_space);
            let ty = var_ptx_type(ir, *dst, *dst);
            let dst_prefix = var_reg_prefix(ir, *dst, *dst);
            let ptr_prefix = var_reg_prefix(ir, *ptr, *ptr);
            writeln!(
                ptx,
                "    ld.{}.{} {}{}, [{}{}];",
                space, ty, dst_prefix, dst, ptr_prefix, ptr
            )
            .unwrap();
        }
        KirOp::Store(ptr, val, addr_space) => {
            let space = address_space_str(*addr_space);
            let ty = var_ptx_type(ir, *val, *val);
            let val_prefix = var_reg_prefix(ir, *val, *val);
            let ptr_prefix = var_reg_prefix(ir, *ptr, *ptr);
            writeln!(
                ptx,
                "    st.{}.{} [{}{}], {}{};",
                space, ty, ptr_prefix, ptr, val_prefix, val
            )
            .unwrap();
        }
        KirOp::AtomicAdd(ptr, val, addr_space) => {
            let space = address_space_str(*addr_space);
            let ty = var_ptx_type(ir, *val, *val);
            let val_prefix = var_reg_prefix(ir, *val, *val);
            let ptr_prefix = var_reg_prefix(ir, *ptr, *ptr);
            writeln!(
                ptx,
                "    atom.{}.add.{} {}{}, [{}{}], {}{};",
                space, ty, val_prefix, val, ptr_prefix, ptr, val_prefix, val
            )
            .unwrap();
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
            writeln!(
                ptx,
                "    shfl.sync.down.b32 {}{}, {}{}, %r{}, 0x1f, 0xffffffff;",
                prefix, dst, prefix, val, offset
            )
            .unwrap();
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
            writeln!(
                ptx,
                "    setp.{}.{} %p{}, {}{}, {}{};",
                op_str, ty, dst, prefix, a, prefix, b
            )
            .unwrap();
        }
        KirOp::Select(dst, cond, true_val, false_val) => {
            let prefix = var_reg_prefix(ir, *dst, *true_val);
            writeln!(
                ptx,
                "    selp.b32 {}{}, {}{}, {}{}, %p{};",
                prefix, dst, prefix, true_val, prefix, false_val, cond
            )
            .unwrap();
        }
        KirOp::Const(dst, konst) => match &konst.value {
            ConstValue::U32(v) => writeln!(ptx, "    mov.u32 %r{}, {};", dst, v).unwrap(),
            ConstValue::I32(v) => writeln!(ptx, "    mov.s32 %r{}, {};", dst, v).unwrap(),
            ConstValue::U64(v) => writeln!(ptx, "    mov.u64 %rd{}, {};", dst, v).unwrap(),
            ConstValue::I64(v) => writeln!(ptx, "    mov.s64 %rd{}, {};", dst, v).unwrap(),
            ConstValue::F32(v) => {
                writeln!(ptx, "    mov.f32 %f{}, 0f{:08X};", dst, v.to_bits()).unwrap()
            }
            ConstValue::F64(v) => {
                writeln!(ptx, "    mov.f64 %fd{}, 0d{:016X};", dst, v.to_bits()).unwrap()
            }
            ConstValue::Bool(v) => writeln!(
                ptx,
                "    setp.eq.u32 %p{}, 1, {};",
                dst,
                if *v { 1 } else { 0 }
            )
            .unwrap(),
        },
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
                writeln!(
                    ptx,
                    "    mul.lo.u64 %rd{}, %rd{}, {};",
                    dst, dst, pointee_size
                )
                .unwrap();
            }
            writeln!(ptx, "    add.u64 %rd{}, %rd{}, %rd{};", dst, base, dst).unwrap();
        }
        KirOp::SharedMemFence => {
            writeln!(ptx, "    membar.cta;").unwrap();
        }
        KirOp::SharedBase(dst) => {
            // The shared block is declared `shared_mem` (see the header);
            // its address is a shared-window offset, moved into the 64-bit
            // pointer register the `.shared` loads/stores already use.
            let prefix = var_reg_prefix(ir, *dst, *dst);
            writeln!(ptx, "    mov.u64 {}{}, shared_mem;", prefix, dst).unwrap();
        }
        KirOp::CpAsync { dst, src, bytes } => {
            let dst_prefix = var_reg_prefix(ir, *dst, *dst);
            let src_prefix = var_reg_prefix(ir, *src, *src);
            writeln!(
                ptx,
                "    cp.async.ca.shared.global [{}{}], [{}{}], {};",
                dst_prefix, dst, src_prefix, src, bytes
            )
            .unwrap();
        }
        KirOp::CpAsyncCommit => {
            writeln!(ptx, "    cp.async.commit_group;").unwrap();
        }
        KirOp::CpAsyncWait { pending } => {
            writeln!(ptx, "    cp.async.wait_group {};", pending).unwrap();
        }
        KirOp::LdMatrixX4 { dst, addr, trans } => {
            let regs = dst
                .iter()
                .map(|v| format!("{}{}", var_reg_prefix(ir, *v, *v), v))
                .collect::<Vec<_>>()
                .join(", ");
            let addr_prefix = var_reg_prefix(ir, *addr, *addr);
            writeln!(
                ptx,
                "    ldmatrix.sync.aligned.m8n8.x4{}.shared.b16 {{{}}}, [{}{}];",
                if *trans { ".trans" } else { "" },
                regs,
                addr_prefix,
                addr
            )
            .unwrap();
        }
        KirOp::MmaF16M16N8K16 { d, a, b, c } => {
            let list = |vars: &[VarId]| {
                vars.iter()
                    .map(|v| format!("{}{}", var_reg_prefix(ir, *v, *v), v))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            writeln!(
                ptx,
                "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                list(d),
                list(a),
                list(b),
                list(c)
            )
            .unwrap();
        }
        KirOp::Matmul { .. } | KirOp::ElementwiseAdd { .. } | KirOp::Relu { .. } => {
            unreachable!("M57 v1 structured KIR ops are only emitted for Target::Fpga; \
                          this codegen path is GPU/CPU PTX")
        }
    }
}

/// The scratch register per class for breaking a cycle in an edge's
/// parallel copy: (PTX type, register name), one per `ptx_reg_prefix`
/// class plus the predicate class.
const EDGE_SCRATCH: [(&str, &str); 7] = [
    ("u32", "%edge_r"),
    ("u64", "%edge_rd"),
    ("f32", "%edge_f"),
    ("f64", "%edge_fd"),
    ("f16", "%edge_h"),
    ("pred", "%edge_p"),
    ("b32", "%edge_v"),
];

/// The register class a variable moves in: (mov type, register prefix,
/// scratch name). `Bool` lives in the predicate class (`setp` writes `%p`,
/// `CondBranch` reads it), the 16-bit floats move as `.b16`, packed
/// fragments as `.b32`; an untyped variable is a `%r`.
fn mov_class(ir: &KernelIR, dst: VarId, src: VarId) -> (&'static str, &'static str, &'static str) {
    match ir.var_types.get(&dst).or_else(|| ir.var_types.get(&src)) {
        Some(KirType::Bool) => ("pred", "%p", "%edge_p"),
        Some(KirType::F16) | Some(KirType::Bf16) => ("b16", "%h", "%edge_h"),
        Some(KirType::Vec(_, _)) => ("b32", "%v", "%edge_v"),
        Some(KirType::F32) => ("f32", "%f", "%edge_f"),
        Some(KirType::F64) => ("f64", "%fd", "%edge_fd"),
        Some(KirType::U64) | Some(KirType::I64) | Some(KirType::Ptr(_, _)) => ("u64", "%rd", "%edge_rd"),
        // U32, I32, I8, I16, the ternary types, untyped.
        _ => ("u32", "%r", "%edge_r"),
    }
}

/// Roadmap A2 step 2: implement an edge's arguments as a parallel copy into
/// the target's parameter registers — every parameter receives its
/// argument's value as it was *before* the copy. Moves are sequenced so a
/// register is written only once nothing pending still reads it; when every
/// pending destination is still read (a cycle), the first destination's
/// current value is saved in the class's scratch register and the reads are
/// redirected there.
fn emit_edge_copies(ptx: &mut String, ir: &KernelIR, edge: &KirEdge) {
    let target = &ir.blocks[edge.target as usize];
    // (dst register text, src register text, mov type), self-moves dropped.
    let mut pending: Vec<(String, String, &'static str)> = Vec::new();
    for (param, arg) in target.params.iter().zip(&edge.args) {
        if param.id == *arg {
            continue;
        }
        let (mov_ty, prefix, _) = mov_class(ir, param.id, *arg);
        pending.push((format!("{prefix}{}", param.id), format!("{prefix}{}", arg), mov_ty));
    }
    while !pending.is_empty() {
        // A destination nobody pending still reads can be written now.
        let ready = pending
            .iter()
            .position(|(dst, _, _)| !pending.iter().any(|(_, src, _)| src == dst));
        match ready {
            Some(i) => {
                let (dst, src, mov_ty) = pending.remove(i);
                writeln!(ptx, "    mov.{mov_ty} {dst}, {src};").unwrap();
            }
            None => {
                // Every destination is still read: a cycle. Park the first
                // destination's value in scratch and redirect its readers.
                let (dst, _, mov_ty) = pending[0].clone();
                let scratch = EDGE_SCRATCH
                    .iter()
                    .find(|(ty, _)| *ty == mov_ty)
                    .map(|(_, name)| *name)
                    .expect("every mov class has a scratch register");
                writeln!(ptx, "    mov.{mov_ty} {scratch}, {dst};").unwrap();
                for (_, src, _) in pending.iter_mut() {
                    if *src == dst {
                        *src = scratch.to_string();
                    }
                }
            }
        }
    }
}

fn emit_terminator(ptx: &mut String, term: &KirTerminator, ir: &KernelIR, block: BlockId) {
    match term {
        KirTerminator::Branch(edge) => {
            emit_edge_copies(ptx, ir, edge);
            writeln!(ptx, "    bra BB{};", edge.target).unwrap();
        }
        KirTerminator::CondBranch(cond, taken, fallthrough)
            if taken.args.is_empty() && fallthrough.args.is_empty() =>
        {
            writeln!(ptx, "    @%p{} bra BB{};", cond, taken.target).unwrap();
            writeln!(ptx, "    bra BB{};", fallthrough.target).unwrap();
        }
        KirTerminator::CondBranch(cond, taken, fallthrough) => {
            // Each edge's copies run only when that edge is taken, so the
            // not-taken path gets its own label in this block.
            writeln!(ptx, "    @!%p{} bra BB{}_else;", cond, block).unwrap();
            emit_edge_copies(ptx, ir, taken);
            writeln!(ptx, "    bra BB{};", taken.target).unwrap();
            writeln!(ptx, "BB{}_else:", block).unwrap();
            emit_edge_copies(ptx, ir, fallthrough);
            writeln!(ptx, "    bra BB{};", fallthrough.target).unwrap();
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
        KirOp::Add(d, a, b)
        | KirOp::Sub(d, a, b)
        | KirOp::Mul(d, a, b)
        | KirOp::Div(d, a, b)
        | KirOp::Pow(d, a, b) => vec![*d, *a, *b],
        KirOp::Fma(d, a, b, c) | KirOp::Select(d, a, b, c) => vec![*d, *a, *b, *c],
        KirOp::Neg(d, s)
        | KirOp::Abs(d, s)
        | KirOp::Sqrt(d, s)
        | KirOp::Exp(d, s)
        | KirOp::Log(d, s)
        | KirOp::Sin(d, s)
        | KirOp::Cos(d, s)
        | KirOp::Tanh(d, s) => vec![*d, *s],
        KirOp::Cast(d, s, _) => vec![*d, *s],
        KirOp::Load(d, p, _) | KirOp::Store(d, p, _) | KirOp::AtomicAdd(d, p, _) => vec![*d, *p],
        KirOp::ThreadId(d, _)
        | KirOp::BlockIdx(d, _)
        | KirOp::BlockDim(d, _)
        | KirOp::GridDim(d, _)
        | KirOp::GlobalId(d, _) => vec![*d],
        KirOp::Barrier | KirOp::SharedMemFence | KirOp::CpAsyncCommit | KirOp::CpAsyncWait { .. } => vec![],
        KirOp::SharedBase(d) => vec![*d],
        KirOp::CpAsync { dst, src, .. } => vec![*dst, *src],
        KirOp::LdMatrixX4 { dst, addr, .. } => {
            let mut v = dst.to_vec();
            v.push(*addr);
            v
        }
        KirOp::MmaF16M16N8K16 { d, a, b, c } => {
            let mut v = d.to_vec();
            v.extend_from_slice(a);
            v.extend_from_slice(b);
            v.extend_from_slice(c);
            v
        }
        KirOp::WarpShuffle(d, v, o) => vec![*d, *v, *o],
        KirOp::Cmp(d, a, b, _) | KirOp::PtrOffset(d, a, b) => vec![*d, *a, *b],
        KirOp::Const(d, _) => vec![*d],
        KirOp::Matmul { a, b, out, .. } => vec![*a, *b, *out],
        KirOp::ElementwiseAdd { a, b, out, .. } => vec![*a, *b, *out],
        KirOp::Relu { a, out, .. } => vec![*a, *out],
    }
}

/// Count the number of body variables (non-param) in the IR: one past the
/// highest `VarId` any op, block parameter or edge argument names.
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
        for p in &block.params {
            max_id = max_id.max(p.id);
        }
        if let Some(term) = &block.terminator {
            for id in crate::kir_verify::terminator_uses(term) {
                max_id = max_id.max(id);
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
        let a_ptr = b.add_param(
            "a",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        let b_ptr = b.add_param(
            "b",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        let out_ptr = b.add_param(
            "out",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        let len = b.add_param("len", KirType::U32, AddressSpace::Local);

        let entry = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();

        b.set_block(entry);
        let tid = b.new_var();
        b.emit(KirOp::GlobalId(tid, 0));
        let in_bounds = b.new_var();
        b.emit(KirOp::Cmp(in_bounds, tid, len, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(in_bounds, body.into(), exit.into()));

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
        b.terminate(KirTerminator::Branch(exit.into()));

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
        assert!(
            !ptx.contains("mad.lo.u32"),
            "GlobalId must NOT use mad.lo.u32 (INVALID_PTX on ISA 7.0)"
        );
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
        assert_eq!(
            *ptx_bytes.last().unwrap(),
            0u8,
            "PTX output must be null-terminated"
        );
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

    /// GlobalId emits synthetic u32 temporaries at dst+1000 and dst+1001
    /// (to avoid collision with normal VarIds).  These indices are NOT
    /// returned by `extract_var_ids`, so without an explicit fix the
    /// `.reg .u32 %r<N>` declaration is too small and ptxas rejects the PTX
    /// with an "undeclared register" error at `cuModuleLoadData` time.
    #[test]
    fn test_ptx_async_copy_group_lowers_on_sm_80() {
        let mut b = KirBuilder::new("cp");
        let src = b.add_param(
            "src",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        b.set_shared_mem(64);
        let entry = b.new_block();
        b.set_block(entry);
        let smem = b.new_typed_var(KirType::Ptr(Box::new(KirType::F32), AddressSpace::Shared));
        b.emit(KirOp::SharedBase(smem));
        b.emit(KirOp::CpAsync { dst: smem, src, bytes: 16 });
        b.emit(KirOp::CpAsyncCommit);
        b.emit(KirOp::CpAsyncWait { pending: 0 });
        b.emit(KirOp::Barrier);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()));
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        assert!(ptx.contains(".target sm_80"), "{ptx}");
        assert!(!ptx.contains(".target sm_70"), "{ptx}");
        assert!(ptx.contains(&format!("    mov.u64 %rd{}, shared_mem;\n", smem)), "{ptx}");
        assert!(
            ptx.contains(&format!("    cp.async.ca.shared.global [%rd{}], [%rd{}], 16;\n", smem, src)),
            "{ptx}"
        );
        assert!(ptx.contains("    cp.async.commit_group;\n"), "{ptx}");
        assert!(ptx.contains("    cp.async.wait_group 0;\n"), "{ptx}");
        assert!(ptx.contains("    bar.sync 0;\n"), "{ptx}");
    }

    #[test]
    fn test_ptx_tensor_core_ops_lower_with_a_packed_register_class() {
        let mut b = KirBuilder::new("mma");
        let out = b.add_param(
            "out",
            KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global),
            AddressSpace::Global,
        );
        b.set_shared_mem(512);
        let entry = b.new_block();
        b.set_block(entry);
        let frag = || KirType::Vec(Box::new(KirType::F16), 2);
        let smem = b.new_typed_var(KirType::Ptr(Box::new(KirType::F16), AddressSpace::Shared));
        b.emit(KirOp::SharedBase(smem));
        let a = [b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag())];
        b.emit(KirOp::LdMatrixX4 { dst: a, addr: smem, trans: false });
        let bb = [b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag())];
        b.emit(KirOp::LdMatrixX4 { dst: bb, addr: smem, trans: true });
        let mut c = [0; 4];
        for slot in &mut c {
            *slot = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Const(*slot, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
        }
        let d = [b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32)];
        b.emit(KirOp::MmaF16M16N8K16 { d, a, b: [bb[0], bb[1]], c });
        b.emit(KirOp::Store(out, d[0], AddressSpace::Global));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()));
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        assert!(ptx.contains(".target sm_80"), "{ptx}");
        assert!(ptx.contains("    .reg .b32 %v<"), "{ptx}");
        assert!(
            ptx.contains(&format!(
                "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{%v{}, %v{}, %v{}, %v{}}}, [%rd{}];\n",
                a[0], a[1], a[2], a[3], smem
            )),
            "{ptx}"
        );
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"), "{ptx}");
        assert!(
            ptx.contains(&format!(
                "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{%f{}, %f{}, %f{}, %f{}}}, {{%v{}, %v{}, %v{}, %v{}}}, {{%v{}, %v{}}}, {{%f{}, %f{}, %f{}, %f{}}};\n",
                d[0], d[1], d[2], d[3], a[0], a[1], a[2], a[3], bb[0], bb[1], c[0], c[1], c[2], c[3]
            )),
            "{ptx}"
        );
    }

    #[test]
    fn test_ptx_target_stays_sm_70_without_async_copy() {
        let ir = build_simple_add_kernel();
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        assert!(ptx.contains(".target sm_70"), "{ptx}");
    }

    #[test]
    fn test_ptx_global_id_register_count_covers_synthetic_temps() {
        let mut b = KirBuilder::new("test_global_id_regs");
        let entry = b.new_block();
        b.set_block(entry);
        // First new_var() → VarId 0; synthetic temps land at %r1000 and %r1001.
        let tid = b.new_var();
        b.emit(KirOp::GlobalId(tid, 0));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();

        let ptx_bytes = lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]);

        let decl_count: u32 = ptx
            .lines()
            .find(|l| l.trim_start().starts_with(".reg .u32 %r<"))
            .and_then(|l| {
                let after = l.trim_start().strip_prefix(".reg .u32 %r<")?;
                after.split('>').next()?.trim().parse().ok()
            })
            .expect(".reg .u32 %r<N> declaration must be present");

        assert!(
            decl_count >= 1002,
            "GlobalId(dst=0) uses synthetic temps %r1000 and %r1001; \
             .reg .u32 %r<N> must declare N >= 1002, got {}",
            decl_count
        );
    }

    /// Build a trivial kernel with a single unary math op `emit(dst, src)`
    /// applied to two fresh f32 vars (src = VarId 0, dst = VarId 1) and
    /// return the emitted PTX text.
    fn ptx_for_unary_op(name: &str, op: fn(VarId, VarId) -> KirOp) -> String {
        let mut b = KirBuilder::new(name);
        let entry = b.new_block();
        b.set_block(entry);
        let src = b.new_typed_var(KirType::F32);
        let dst = b.new_typed_var(KirType::F32);
        b.emit(op(dst, src));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();

        let ptx_bytes = lower_kir_to_ptx(&ir);
        String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]).into_owned()
    }

    /// KirOp::Exp must compute e^x, not 2^x: ex2.approx computes 2^x, so the
    /// input must be pre-scaled by log2(e) = 0f3FB8AA3B.
    #[test]
    fn test_ptx_exp_applies_base_conversion() {
        let ptx = ptx_for_unary_op("test_exp", KirOp::Exp);

        let mul = ptx
            .find("mul.f32 %f1, %f0, 0f3FB8AA3B;")
            .expect("Exp must pre-multiply the input by log2(e) (0f3FB8AA3B)");
        let ex2 = ptx
            .find("ex2.approx.f32 %f1, %f1;")
            .expect("Exp must apply ex2.approx to the pre-scaled value");
        assert!(
            mul < ex2,
            "log2(e) pre-scale must come before ex2.approx:\n{}",
            ptx
        );
        assert!(
            !ptx.contains("ex2.approx.f32 %f1, %f0;"),
            "Exp must not feed the raw input to ex2.approx (that computes 2^x):\n{}",
            ptx
        );
    }

    /// KirOp::Log must compute ln(x), not log2(x): lg2.approx computes
    /// log2(x), so the result must be post-scaled by ln(2) = 0f3F317218.
    #[test]
    fn test_ptx_log_applies_base_conversion() {
        let ptx = ptx_for_unary_op("test_log", KirOp::Log);

        let lg2 = ptx
            .find("lg2.approx.f32 %f1, %f0;")
            .expect("Log must apply lg2.approx to the input");
        let mul = ptx
            .find("mul.f32 %f1, %f1, 0f3F317218;")
            .expect("Log must post-multiply the lg2 result by ln(2) (0f3F317218)");
        assert!(
            lg2 < mul,
            "ln(2) post-scale must come after lg2.approx:\n{}",
            ptx
        );
    }

    /// KirOp::Tanh must emit a real expansion (tanh(x) = 2*sigmoid(2x) - 1,
    /// mirroring the epilogue_fusion.rs sequence), not the old silent
    /// identity `mov.f32 dst, src`.
    #[test]
    fn test_ptx_tanh_emits_expansion_not_identity() {
        let ptx = ptx_for_unary_op("test_tanh", KirOp::Tanh);

        assert!(
            !ptx.contains("mov.f32 %f1, %f0;"),
            "Tanh must not emit a bare mov identity:\n{}",
            ptx
        );
        let expected = [
            "add.f32 %f1, %f0, %f0;",       // 2x
            "neg.f32 %f1, %f1;",            // -2x
            "mul.f32 %f1, %f1, 0f3FB8AA3B;", // -2x * log2(e)
            "ex2.approx.f32 %f1, %f1;",     // exp(-2x)
            "add.f32 %f1, %f1, 0f3F800000;", // 1 + exp(-2x)
            "rcp.approx.f32 %f1, %f1;",     // sigmoid(2x)
            "add.f32 %f1, %f1, %f1;",       // 2*sigmoid(2x)
            "sub.f32 %f1, %f1, 0f3F800000;", // 2*sigmoid(2x) - 1
        ];
        let mut cursor = 0;
        for instr in expected {
            match ptx[cursor..].find(instr) {
                Some(pos) => cursor += pos + instr.len(),
                None => panic!(
                    "Tanh expansion missing (or out of order): `{}`\nPTX:\n{}",
                    instr, ptx
                ),
            }
        }
    }

    /// The unary math expansions read the source register only in their
    /// first instruction, so they must stay correct when dst == src.
    #[test]
    fn test_ptx_unary_math_dst_src_aliasing() {
        let mut b = KirBuilder::new("test_alias");
        let entry = b.new_block();
        b.set_block(entry);
        let x = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Exp(x, x));
        b.emit(KirOp::Tanh(x, x));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();

        let ptx_bytes = lower_kir_to_ptx(&ir);
        let ptx = String::from_utf8_lossy(&ptx_bytes[..ptx_bytes.len() - 1]);

        // Exp(x, x): the log2(e) pre-scale reads %f0 before overwriting it.
        assert!(
            ptx.contains("mul.f32 %f0, %f0, 0f3FB8AA3B;"),
            "aliased Exp must pre-scale in-place:\n{}",
            ptx
        );
        // Tanh(x, x): the doubling reads %f0 twice before overwriting it.
        assert!(
            ptx.contains("add.f32 %f0, %f0, %f0;"),
            "aliased Tanh must double in-place:\n{}",
            ptx
        );
        assert!(
            ptx.contains("sub.f32 %f0, %f0, 0f3F800000;"),
            "aliased Tanh must complete the 2*sigmoid(2x) - 1 expansion:\n{}",
            ptx
        );
    }

    // ── Roadmap A2 step 2: block parameters lower to edge copies ─────

    /// A grid-stride `out[i] = a[i]` copy: the header takes `i`, the entry
    /// edge passes the thread's first index, the back edge `i + stride`.
    fn grid_stride_copy() -> KernelIR {
        let f32_ptr = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        let mut b = KirBuilder::new("grid_stride_copy");
        let a = b.add_param("a", f32_ptr.clone(), AddressSpace::Global);
        let out = b.add_param("out", f32_ptr.clone(), AddressSpace::Global);
        let n = b.add_param("n", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.add_block_param(header, KirType::U32);
        b.set_block(entry);
        let start = b.new_typed_var(KirType::U32);
        b.emit(KirOp::GlobalId(start, 0));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![start])));
        b.set_block(header);
        let more = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(more, i, n, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(more, body.into(), exit.into()));
        b.set_block(body);
        let src = b.new_typed_var(f32_ptr.clone());
        b.emit(KirOp::PtrOffset(src, a, i));
        let v = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(v, src, AddressSpace::Global));
        let dst = b.new_typed_var(f32_ptr);
        b.emit(KirOp::PtrOffset(dst, out, i));
        b.emit(KirOp::Store(dst, v, AddressSpace::Global));
        let bdim = b.new_typed_var(KirType::U32);
        b.emit(KirOp::BlockDim(bdim, 0));
        let gdim = b.new_typed_var(KirType::U32);
        b.emit(KirOp::GridDim(gdim, 0));
        let stride = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Mul(stride, bdim, gdim));
        let next = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Add(next, i, stride));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![next])));
        b.set_block(exit);
        b.terminate(KirTerminator::Return);
        b.finalize()
    }

    #[test]
    fn a_loop_edge_copies_its_argument_into_the_parameter_register() {
        let ir = grid_stride_copy();
        assert_eq!(ir.verify(), Ok(()));
        let i = ir.blocks[1].params[0].id;
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        // Entry edge: the first index moves into the parameter, then the jump.
        assert!(ptx.contains(&format!("    mov.u32 %r{i}, %r4;\n    bra BB1;\n")), "{ptx}");
        // Back edge: the next index moves into the same register.
        let next = ir.blocks[2].ops.iter().rev().find_map(crate::kir_verify::op_dst).unwrap();
        assert!(ptx.contains(&format!("    mov.u32 %r{i}, %r{next};\n    bra BB1;\n")), "{ptx}");
        // Arg-less conditional edges keep the two-line form.
        assert!(ptx.contains("    @%p5 bra BB2;\n    bra BB3;\n"), "{ptx}");
        // The scratch class is declared, once, and the parameter register fits.
        assert_eq!(ptx.matches(".reg .u32 %edge_r;").count(), 1, "{ptx}");
        let declared: u32 = ptx.split(".reg .u32 %r<").nth(1).unwrap().split('>').next().unwrap().parse().unwrap();
        assert!(declared > i, "{ptx}");
    }

    #[test]
    fn a_swap_on_an_edge_goes_through_the_scratch_register() {
        // header(p0, p1) with a back edge passing (p1, p0): a two-cycle.
        let mut b = KirBuilder::new("swap");
        let flag = b.add_param("flag", KirType::Bool, AddressSpace::Local);
        let entry = b.new_block();
        let header = b.new_block();
        let exit = b.new_block();
        let p0 = b.add_block_param(header, KirType::F32);
        let p1 = b.add_block_param(header, KirType::F32);
        b.set_block(entry);
        let x = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(x, KirConst { ty: KirType::F32, value: ConstValue::F32(1.0) }));
        let y = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(y, KirConst { ty: KirType::F32, value: ConstValue::F32(2.0) }));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![x, y])));
        b.set_block(header);
        b.terminate(KirTerminator::CondBranch(
            flag,
            KirEdge::with(header, vec![p1, p0]),
            exit.into(),
        ));
        b.set_block(exit);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()));
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        let expected = format!(
            "    @!%p{flag} bra BB1_else;\n    mov.f32 %edge_f, %f{p0};\n    mov.f32 %f{p0}, %f{p1};\n    mov.f32 %f{p1}, %edge_f;\n    bra BB1;\nBB1_else:\n    bra BB2;\n"
        );
        assert!(ptx.contains(&expected), "{ptx}");
    }

    #[test]
    fn a_chain_on_an_edge_writes_the_read_register_last() {
        // header(p0, p1) with an edge passing (v, p0): p1 <- p0 must run
        // before p0 <- v.
        let mut b = KirBuilder::new("chain");
        let entry = b.new_block();
        let header = b.new_block();
        let p0 = b.add_block_param(header, KirType::U32);
        let p1 = b.add_block_param(header, KirType::U32);
        b.set_block(entry);
        let v = b.new_typed_var(KirType::U32);
        b.emit(KirOp::ThreadId(v, 0));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![v, p0])));
        b.set_block(header);
        b.terminate(KirTerminator::Return);
        let mut ir = b.finalize();
        // Not verifiable (the entry reads p0 before the header defines it);
        // the sequencing is what is under test.
        ir.var_types.insert(p0, KirType::U32);
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        let expected = format!("    mov.u32 %r{p1}, %r{p0};\n    mov.u32 %r{p0}, %r{v};\n    bra BB1;\n");
        assert!(ptx.contains(&expected), "{ptx}");
    }

    #[test]
    fn kernels_without_block_params_declare_no_scratch() {
        let ir = grid_stride_copy();
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        assert!(ptx.contains("%edge_"));
        let mut b = KirBuilder::new("plain");
        let e = b.new_block();
        b.set_block(e);
        b.terminate(KirTerminator::Return);
        let ptx = String::from_utf8(lower_kir_to_ptx(&b.finalize())).unwrap();
        assert!(!ptx.contains("%edge_"), "{ptx}");
    }
}
