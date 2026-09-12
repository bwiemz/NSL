// crates/nsl-kir/src/backend_ptx.rs
//! M47: KIR -> PTX text emission backend.
//!
//! Lowers a `KernelIR` to null-terminated PTX text bytes suitable for
//! `cuModuleLoadData`. Uses PTX ISA 7.0 targeting sm_70.

use crate::FeatureSet;
use crate::kernel_ir::*;
use crate::regalloc::RegClass;
use std::collections::HashMap;
use std::fmt::Write;

/// Lower a KernelIR to PTX text bytes (null-terminated).
pub fn lower_kir_to_ptx(ir: &KernelIR) -> Vec<u8> {
    // Roadmap A2 step 5: the body is printed first with virtual
    // `%<class><VarId>` names, the allocator (`crate::regalloc`) renames
    // them to dense per-class indices, and the header's `.reg` declarations
    // are written last from the counts that produced.
    let mut body = String::new();

    // Load parameters into registers
    for param in &ir.params {
        match &param.ty {
            KirType::Ptr(_, _) => {
                writeln!(
                    body,
                    "    ld.param.u64 %rd{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::U32 => {
                writeln!(
                    body,
                    "    ld.param.u32 %r{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::I32 => {
                writeln!(
                    body,
                    "    ld.param.s32 %r{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::F32 => {
                writeln!(
                    body,
                    "    ld.param.f32 %f{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            KirType::F64 => {
                writeln!(
                    body,
                    "    ld.param.f64 %fd{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
            _ => {
                writeln!(
                    body,
                    "    ld.param.u32 %r{}, [param_{}];",
                    param.id, param.name
                )
                .unwrap();
            }
        }
    }
    writeln!(body).unwrap();

    // Emit blocks
    for block in &ir.blocks {
        writeln!(body, "BB{}:", block.id).unwrap();
        for op in &block.ops {
            emit_op(&mut body, op, ir);
        }
        if let Some(ref term) = block.terminator {
            emit_terminator(&mut body, term, ir, block.id);
        }
    }
    let alloc = crate::regalloc::allocate(ir);
    let (body, extra) = rename_registers(&body, &alloc);

    let mut ptx = String::new();

    // Header. `cp.async` (FeatureSet::ASYNC_COPY) is an sm_80 instruction;
    // every other op lowers on sm_70, the floor this backend has always
    // targeted, so the bump is taken only when a kernel asks for it.
    // A bf16 conversion (FeatureSet::BF16_ARITHMETIC) is PTX ISA 7.8 and
    // sm_80 as well (roadmap A2 step 4).
    let bf16 = ir.required_features.contains(FeatureSet::BF16_ARITHMETIC);
    writeln!(ptx, ".version {}", if bf16 { "7.8" } else { "7.0" }).unwrap();
    // The tensor-core ops (`ldmatrix`, `mma.sync` m16n8k16 f16) are sm_80
    // instructions as well.
    let target = if ir.required_features.contains(FeatureSet::ASYNC_COPY)
        || ir.required_features.contains(FeatureSet::TENSOR_CORES)
        || bf16
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

    // Roadmap A2 step 5: launch bounds and the register cap, as the hand
    // estate writes them.
    if let Some(lb) = ir.launch_bounds {
        writeln!(ptx, "    .maxntid {}, 1, 1", lb.max_threads).unwrap();
        if let Some(m) = lb.min_blocks_per_sm {
            writeln!(ptx, "    .minnctapersm {}", m).unwrap();
        }
    }
    if let Some(n) = ir.max_registers {
        writeln!(ptx, "    .maxnreg {}", n).unwrap();
    }

    // Register declarations: one class at its allocated count (plus any
    // registers the rename pass had to invent for a value printed in a
    // class other than its own).
    let pressure = alloc.pressure();
    for (i, class) in RegClass::ALL.iter().enumerate() {
        let count = pressure.of(*class) + extra[i];
        if count > 0 {
            writeln!(ptx, "    .reg .{} {}<{}>;", class.ptx_type(), class.prefix(), count).unwrap();
        }
    }
    // Roadmap A2 step 2: an edge that passes block arguments is a parallel
    // copy into the target's parameter registers. A cycle in that copy (a
    // swap) needs one scratch register of the class; declared only when a
    // kernel has block parameters (or selects a predicate, which lowers
    // through the same scratch), so every other kernel's text is unchanged.
    let selects_a_bool = ir.blocks.iter().flat_map(|b| b.ops.iter()).any(|op| {
        matches!(op, KirOp::Select(d, _, t, _)
            if ir.var_types.get(d).or_else(|| ir.var_types.get(t)) == Some(&KirType::Bool))
    });
    if ir.blocks.iter().any(|b| !b.params.is_empty()) || selects_a_bool {
        for (ty, name) in EDGE_SCRATCH {
            writeln!(ptx, "    .reg .{ty} {name};").unwrap();
        }
    }
    // `GlobalId` lowers through two u32 temporaries of its own.
    if ir.blocks.iter().flat_map(|b| b.ops.iter()).any(|op| matches!(op, KirOp::GlobalId(..))) {
        writeln!(ptx, "    .reg .u32 %gid0;").unwrap();
        writeln!(ptx, "    .reg .u32 %gid1;").unwrap();
    }
    writeln!(ptx).unwrap();

    ptx.push_str(&body);
    writeln!(ptx, "}}").unwrap();

    // Null-terminate
    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

/// Roadmap A2 step 5: rename every virtual `%<class><VarId>` register in
/// `text` to the allocated `%<class><index>`. A value printed in a class
/// other than its own (`extract`-style mismatches the verifier would have
/// reported) gets a fresh register of the printed class past the allocated
/// count; the per-class number of those is returned so the declarations
/// can cover them.
fn rename_registers(text: &str, alloc: &crate::regalloc::Allocation) -> (String, [u32; 7]) {
    let mut out = String::with_capacity(text.len());
    let mut extra = [0u32; 7];
    let mut invented: HashMap<(RegClass, VarId), u32> = HashMap::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'%' {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        }
        // `%` + letters + digits, not followed by an identifier character.
        let mut j = i + 1;
        while j < bytes.len() && bytes[j].is_ascii_lowercase() {
            j += 1;
        }
        let mut k = j;
        while k < bytes.len() && bytes[k].is_ascii_digit() {
            k += 1;
        }
        let followed_by_ident = k < bytes.len() && (bytes[k].is_ascii_alphanumeric() || bytes[k] == b'_');
        let class = RegClass::from_prefix(&text[i..j]);
        match class {
            Some(class) if k > j && !followed_by_ident => {
                let var: VarId = text[j..k].parse().unwrap();
                let index = match alloc.class(var) {
                    Some(c) if c == class => alloc.index(var).unwrap(),
                    _ => {
                        let slot = RegClass::ALL.iter().position(|c| *c == class).unwrap();
                        *invented.entry((class, var)).or_insert_with(|| {
                            let idx = alloc.pressure().of(class) + extra[slot];
                            extra[slot] += 1;
                            idx
                        })
                    }
                };
                out.push_str(class.prefix());
                out.push_str(&index.to_string());
                i = k;
            }
            _ => {
                out.push('%');
                i += 1;
            }
        }
    }
    (out, extra)
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
            // `.lo` (the low half of the product) is the integer form;
            // a float multiply has no such modifier (roadmap A2 step 3:
            // `mul.lo.f32` is not PTX).
            let lo = match ir.var_types.get(dst).or_else(|| ir.var_types.get(a)) {
                Some(KirType::F16) | Some(KirType::Bf16) | Some(KirType::F32) | Some(KirType::F64) => "",
                _ => ".lo",
            };
            writeln!(
                ptx,
                "    mul{}.{} {}{}, {}{}, {}{};",
                lo, ty, prefix, dst, prefix, a, prefix, b
            )
            .unwrap();
        }
        KirOp::Div(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            // A float division carries a rounding modifier (`div.f32` is
            // not PTX); integer division has none. Roadmap A2 step 3: the
            // IEEE `.rn` form, which is what `/` means.
            let round = match ir.var_types.get(dst).or_else(|| ir.var_types.get(a)) {
                Some(KirType::F32) | Some(KirType::F64) => ".rn",
                _ => "",
            };
            writeln!(
                ptx,
                "    div{}.{} {}{}, {}{}, {}{};",
                round, ty, prefix, dst, prefix, a, prefix, b
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
            emit_cvt(ptx, ir, *dst, *src, target_ty, None);
        }
        KirOp::CastRounded { dst, src, ty, mode } => {
            emit_cvt(ptx, ir, *dst, *src, ty, Some(*mode));
        }
        KirOp::Load(dst, ptr, addr_space) => {
            let space = address_space_str(*addr_space);
            let ty = mem_type(var_ptx_type(ir, *dst, *dst));
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
            let ty = mem_type(var_ptx_type(ir, *val, *val));
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
            // Two named temporaries of the printer's own (declared with the
            // classes; roadmap A2 step 5 retired the `dst + 1000` idiom).
            writeln!(ptx, "    mov.u32 %gid0, %ctaid.{};", dim_name).unwrap();
            writeln!(ptx, "    mov.u32 %gid1, %ntid.{};", dim_name).unwrap();
            writeln!(ptx, "    mul.lo.u32 %gid0, %gid0, %gid1;").unwrap();
            writeln!(ptx, "    mov.u32 %r{}, %tid.{};", dst, dim_name).unwrap();
            writeln!(ptx, "    add.u32 %r{}, %gid0, %r{};", dst, dst).unwrap();
        }
        KirOp::Barrier => {
            writeln!(ptx, "    bar.sync 0;").unwrap();
        }
        KirOp::WarpShuffle { dst, val, lane, mode, width } => {
            let prefix = var_reg_prefix(ir, *dst, *val);
            // The `c` operand packs the segment mask: `(32 - width) << 8`
            // in the upper byte, and the clamp (0x1f) in the lower byte for
            // every mode but `up`, whose clamp is 0.
            let segmask = (32u32 - u32::from(*width).clamp(1, 32)) << 8;
            let (mnemonic, c) = match mode {
                ShuffleMode::Down => ("down", segmask | 0x1f),
                ShuffleMode::Up => ("up", segmask),
                ShuffleMode::Xor => ("bfly", segmask | 0x1f),
                ShuffleMode::Idx => ("idx", segmask | 0x1f),
            };
            writeln!(
                ptx,
                "    shfl.sync.{}.b32 {}{}, {}{}, %r{}, 0x{:x}, 0xffffffff;",
                mnemonic, prefix, dst, prefix, val, lane, c
            )
            .unwrap();
        }
        KirOp::Vote { dst, pred, mode } => match mode {
            VoteMode::Any => writeln!(ptx, "    vote.sync.any.pred %p{}, %p{}, 0xffffffff;", dst, pred).unwrap(),
            VoteMode::All => writeln!(ptx, "    vote.sync.all.pred %p{}, %p{}, 0xffffffff;", dst, pred).unwrap(),
            VoteMode::Ballot => {
                writeln!(ptx, "    vote.sync.ballot.b32 %r{}, %p{}, 0xffffffff;", dst, pred).unwrap()
            }
        },
        KirOp::LaneId(dst) => {
            writeln!(ptx, "    mov.u32 %r{}, %laneid;", dst).unwrap();
        }
        KirOp::WarpId(dst) => {
            writeln!(ptx, "    mov.u32 %r{}, %warpid;", dst).unwrap();
        }
        // Roadmap A2 step 4: the integer / bitwise ISA.
        KirOp::And(dst, a, b) | KirOp::Or(dst, a, b) | KirOp::Xor(dst, a, b) => {
            let mnemonic = match op {
                KirOp::And(..) => "and",
                KirOp::Or(..) => "or",
                _ => "xor",
            };
            let (ty, prefix) = bit_class(ir, *dst, *a);
            writeln!(ptx, "    {}.{} {}{}, {}{}, {}{};", mnemonic, ty, prefix, dst, prefix, a, prefix, b)
                .unwrap();
        }
        KirOp::Not(dst, src) => {
            let (ty, prefix) = bit_class(ir, *dst, *src);
            writeln!(ptx, "    not.{} {}{}, {}{};", ty, prefix, dst, prefix, src).unwrap();
        }
        KirOp::Shl(dst, a, amount) => {
            let (ty, prefix) = bit_class(ir, *dst, *a);
            writeln!(ptx, "    shl.{} {}{}, {}{}, %r{};", ty, prefix, dst, prefix, a, amount).unwrap();
        }
        KirOp::Shr(dst, a, amount) => {
            // Arithmetic for signed types, logical otherwise.
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            let ty = match ty {
                "s32" | "s8" | "s16" => "s32",
                "s64" => "s64",
                "u64" => "u64",
                _ => "u32",
            };
            writeln!(ptx, "    shr.{} {}{}, {}{}, %r{};", ty, prefix, dst, prefix, a, amount).unwrap();
        }
        KirOp::Rem(dst, a, b) => {
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(ptx, "    rem.{} {}{}, {}{}, {}{};", ty, prefix, dst, prefix, a, prefix, b).unwrap();
        }
        KirOp::Min(dst, a, b) | KirOp::Max(dst, a, b) => {
            let mnemonic = if matches!(op, KirOp::Min(..)) { "min" } else { "max" };
            let ty = var_ptx_type(ir, *dst, *a);
            let prefix = var_reg_prefix(ir, *dst, *a);
            writeln!(ptx, "    {}.{} {}{}, {}{}, {}{};", mnemonic, ty, prefix, dst, prefix, a, prefix, b)
                .unwrap();
        }
        KirOp::Rcp(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            let mnemonic = if var_ptx_type(ir, *dst, *src) == "f64" { "rcp.rn.f64" } else { "rcp.approx.f32" };
            writeln!(ptx, "    {} {}{}, {}{};", mnemonic, prefix, dst, prefix, src).unwrap();
        }
        KirOp::Rsqrt(dst, src) => {
            let prefix = var_reg_prefix(ir, *dst, *src);
            let mnemonic =
                if var_ptx_type(ir, *dst, *src) == "f64" { "rsqrt.approx.f64" } else { "rsqrt.approx.f32" };
            writeln!(ptx, "    {} {}{}, {}{};", mnemonic, prefix, dst, prefix, src).unwrap();
        }
        KirOp::LoadVec { dsts, ptr, space } => {
            let first = dsts.first().copied().unwrap_or(0);
            let ty = mem_type(var_ptx_type(ir, first, first));
            let prefix = var_reg_prefix(ir, first, first);
            let regs: Vec<String> = dsts.iter().map(|d| format!("{prefix}{d}")).collect();
            writeln!(
                ptx,
                "    ld.{}.v{}.{} {{{}}}, [{}{}];",
                address_space_str(*space),
                dsts.len(),
                ty,
                regs.join(", "),
                var_reg_prefix(ir, *ptr, *ptr),
                ptr
            )
            .unwrap();
        }
        KirOp::StoreVec { ptr, vals, space } => {
            let first = vals.first().copied().unwrap_or(0);
            let ty = mem_type(var_ptx_type(ir, first, first));
            let prefix = var_reg_prefix(ir, first, first);
            let regs: Vec<String> = vals.iter().map(|v| format!("{prefix}{v}")).collect();
            writeln!(
                ptx,
                "    st.{}.v{}.{} [{}{}], {{{}}};",
                address_space_str(*space),
                vals.len(),
                ty,
                var_reg_prefix(ir, *ptr, *ptr),
                ptr,
                regs.join(", ")
            )
            .unwrap();
        }
        KirOp::Predicated { pred, negate, op: inner } => {
            // Lower the wrapped op on its own and guard every line of it.
            let mut body = String::new();
            emit_op(&mut body, inner, ir);
            let guard = if *negate { format!("@!%p{pred} ") } else { format!("@%p{pred} ") };
            for line in body.lines() {
                let trimmed = line.trim_start();
                if trimmed.is_empty() {
                    continue;
                }
                writeln!(ptx, "    {guard}{trimmed}").unwrap();
            }
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
            match ir.var_types.get(dst).or_else(|| ir.var_types.get(true_val)) {
                // No `selp.pred`: d = (c & t) | (!c & f) over the predicate class.
                Some(KirType::Bool) => {
                    writeln!(ptx, "    and.pred %edge_p, %p{}, %p{};", cond, true_val).unwrap();
                    writeln!(ptx, "    not.pred %p{}, %p{};", dst, cond).unwrap();
                    writeln!(ptx, "    and.pred %p{}, %p{}, %p{};", dst, dst, false_val).unwrap();
                    writeln!(ptx, "    or.pred %p{}, %p{}, %edge_p;", dst, dst).unwrap();
                }
                Some(ty) => {
                    let prefix = ty.ptx_reg_prefix();
                    // selp takes .b16/.b32/.b64 and the numeric types; the
                    // 32-bit integer classes keep the historical `.b32`.
                    let sel_ty = match ty {
                        KirType::F32 => "f32",
                        KirType::F64 => "f64",
                        KirType::U64 | KirType::I64 | KirType::Ptr(_, _) => "b64",
                        KirType::F16 | KirType::Bf16 => "b16",
                        _ => "b32",
                    };
                    writeln!(
                        ptx,
                        "    selp.{} {}{}, {}{}, {}{}, %p{};",
                        sel_ty, prefix, dst, prefix, true_val, prefix, false_val, cond
                    )
                    .unwrap();
                }
                None => {
                    writeln!(ptx, "    selp.b32 %r{}, %r{}, %r{}, %p{};", dst, true_val, false_val, cond)
                        .unwrap();
                }
            }
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
        KirOp::SharedRegion { dst, region } => {
            // One `mov` to the block, then the region's offset folded in.
            // The offset comes from `SmemLayout::offset_of` — the same
            // computation the accessors derive from, so a region cannot be
            // at one address here and another there. Rule 8 has already
            // bounds-checked `region`; `unwrap_or(0)` keeps the printer
            // total for a kernel that reached it unverified.
            let prefix = var_reg_prefix(ir, *dst, *dst);
            let offset = ir.smem_layout.offset_of(*region as usize).unwrap_or(0);
            writeln!(ptx, "    mov.u64 {}{}, shared_mem;", prefix, dst).unwrap();
            if offset != 0 {
                writeln!(
                    ptx,
                    "    add.u64 {}{}, {}{}, {};",
                    prefix, dst, prefix, dst, offset
                )
                .unwrap();
            }
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
        KirOp::LdMatrix { dst, addr, trans } => {
            let regs = dst
                .iter()
                .map(|v| format!("{}{}", var_reg_prefix(ir, *v, *v), v))
                .collect::<Vec<_>>()
                .join(", ");
            let addr_prefix = var_reg_prefix(ir, *addr, *addr);
            // The `.x{N}` token is the operand-vector arity: rule 6 has
            // already held `dst.len()` to 1, 2 or 4, the only forms PTX has.
            writeln!(
                ptx,
                "    ldmatrix.sync.aligned.m8n8.x{}{}.shared.b16 {{{}}}, [{}{}];",
                dst.len(),
                if *trans { ".trans" } else { "" },
                regs,
                addr_prefix,
                addr
            )
            .unwrap();
        }
        KirOp::Mma { shape, a_ty, d, a, b, c } => {
            let list = |vars: &[VarId]| {
                vars.iter()
                    .map(|v| format!("{}{}", var_reg_prefix(ir, *v, *v), v))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            // `.f32.<ty>.<ty>.f32` — the accumulator is f32 at every estate
            // site, so only the A/B type varies with `a_ty`.
            writeln!(
                ptx,
                "    mma.sync.aligned.{}.row.col.f32.{}.{}.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                shape.ptx_shape(),
                a_ty.ptx_type(),
                a_ty.ptx_type(),
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
    ("b16", "%edge_h"),
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

/// The memory-op type for a register type: the 16-bit floats move as
/// `.b16` (`ld.global.bf16` is not an instruction), everything else as its
/// own type.
fn mem_type(ty: &'static str) -> &'static str {
    match ty {
        "f16" | "bf16" => "b16",
        other => other,
    }
}

/// The (type, prefix) a bitwise op uses: predicates for `Bool`, otherwise
/// the untyped class of the register's width.
fn bit_class(ir: &KernelIR, primary: VarId, fallback: VarId) -> (&'static str, &'static str) {
    match ir.var_types.get(&primary).or_else(|| ir.var_types.get(&fallback)) {
        Some(KirType::Bool) => ("pred", "%p"),
        Some(KirType::U64) | Some(KirType::I64) | Some(KirType::Ptr(_, _)) => ("b64", "%rd"),
        Some(KirType::F16) | Some(KirType::Bf16) => ("b16", "%h"),
        Some(KirType::Vec(_, _)) => ("b32", "%v"),
        _ => ("b32", "%r"),
    }
}

/// `cvt` with the rounding modifier PTX requires (roadmap A2 step 4): a
/// float result of a float source that loses width, or of an integer
/// source, rounds (`.rn` by default); an integer result of a float source
/// rounds to an integer (`.rzi` by default); everything else is exact and
/// takes no modifier. An explicit `mode` overrides the default where a
/// modifier applies. The source type falls back to `u32` when untyped,
/// as the other arms do.
fn emit_cvt(ptx: &mut String, ir: &KernelIR, dst: VarId, src: VarId, target: &KirType, mode: Option<RoundMode>) {
    let src_kty = ir.var_types.get(&src);
    let src_ty = var_ptx_type(ir, src, src);
    let dst_ty = target.ptx_type();
    let src_prefix = var_reg_prefix(ir, src, src);
    let dst_prefix = target.ptx_reg_prefix();
    let is_f = |t: Option<&KirType>| {
        matches!(t, Some(KirType::F16) | Some(KirType::Bf16) | Some(KirType::F32) | Some(KirType::F64))
    };
    let src_float = is_f(src_kty);
    let dst_float = is_f(Some(target));
    let float_result_rounds = dst_float
        && (!src_float || src_kty.is_some_and(|t| t.size_bytes() > target.size_bytes()));
    let modifier = if float_result_rounds {
        match mode.unwrap_or(RoundMode::Rn) {
            RoundMode::Rn => ".rn",
            RoundMode::Rz => ".rz",
            RoundMode::Rm => ".rm",
            RoundMode::Rp => ".rp",
        }
    } else if src_float && !dst_float && *target != KirType::Bool {
        match mode.unwrap_or(RoundMode::Rz) {
            RoundMode::Rn => ".rni",
            RoundMode::Rz => ".rzi",
            RoundMode::Rm => ".rmi",
            RoundMode::Rp => ".rpi",
        }
    } else {
        ""
    };
    writeln!(ptx, "    cvt{}.{}.{} {}{}, {}{};", modifier, dst_ty, src_ty, dst_prefix, dst, src_prefix, src)
        .unwrap();
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
        b.emit(KirOp::LdMatrix { dst: a.to_vec(), addr: smem, trans: false });
        let bb = [b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag()), b.new_typed_var(frag())];
        b.emit(KirOp::LdMatrix { dst: bb.to_vec(), addr: smem, trans: true });
        let mut c = [0; 4];
        for slot in &mut c {
            *slot = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Const(*slot, KirConst { ty: KirType::F32, value: ConstValue::F32(0.0) }));
        }
        let d = [b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32), b.new_typed_var(KirType::F32)];
        b.emit(KirOp::Mma { shape: MmaShape::M16N8K16, a_ty: MmaOperandTy::F16, d: d.to_vec(), a: a.to_vec(), b: vec![bb[0], bb[1]], c: c.to_vec() });
        b.emit(KirOp::Store(out, d[0], AddressSpace::Global));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()));
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        let al = crate::regalloc::allocate(&ir);
        let n = |v: VarId| al.name(v);
        assert!(ptx.contains(".target sm_80"), "{ptx}");
        assert!(ptx.contains("    .reg .b32 %v<8>;"), "{ptx}");
        assert!(
            ptx.contains(&format!(
                "    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{}, {}, {}, {}}}, [{}];\n",
                n(a[0]), n(a[1]), n(a[2]), n(a[3]), n(smem)
            )),
            "{ptx}"
        );
        assert!(ptx.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"), "{ptx}");
        assert!(
            ptx.contains(&format!(
                "    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {{{}, {}, {}, {}}}, {{{}, {}, {}, {}}}, {{{}, {}}}, {{{}, {}, {}, {}}};\n",
                n(d[0]), n(d[1]), n(d[2]), n(d[3]), n(a[0]), n(a[1]), n(a[2]), n(a[3]), n(bb[0]), n(bb[1]), n(c[0]), n(c[1]), n(c[2]), n(c[3])
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
    fn test_ptx_global_id_uses_named_scratch() {
        // Roadmap A2 step 5: `GlobalId` lowers through two named
        // temporaries declared with the classes, not through registers at
        // `dst + 1000` that the declaration count had to be padded for.
        let mut b = KirBuilder::new("test_global_id_regs");
        let entry = b.new_block();
        b.set_block(entry);
        let tid = b.new_var();
        b.emit(KirOp::GlobalId(tid, 0));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        assert!(ptx.contains("    .reg .u32 %r<1>;\n"), "{ptx}");
        assert!(ptx.contains("    .reg .u32 %gid0;\n    .reg .u32 %gid1;\n"), "{ptx}");
        assert!(ptx.contains("    mul.lo.u32 %gid0, %gid0, %gid1;\n    mov.u32 %r0, %tid.x;\n    add.u32 %r0, %gid0, %r0;\n"), "{ptx}");
        assert!(!ptx.contains("%r1000"), "{ptx}");
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
        let al = crate::regalloc::allocate(&ir);
        let n = |v: VarId| al.name(v);
        // Entry edge: the first index moves into the parameter, then the jump.
        let start = crate::kir_verify::op_dst(&ir.blocks[0].ops[0]).unwrap();
        assert!(ptx.contains(&format!("    mov.u32 {}, {};\n    bra BB1;\n", n(i), n(start))), "{ptx}");
        // Back edge: the next index moves into the same register.
        let next = ir.blocks[2].ops.iter().rev().find_map(crate::kir_verify::op_dst).unwrap();
        assert!(ptx.contains(&format!("    mov.u32 {}, {};\n    bra BB1;\n", n(i), n(next))), "{ptx}");
        // Arg-less conditional edges keep the two-line form.
        let more = crate::kir_verify::op_dst(&ir.blocks[1].ops[0]).unwrap();
        assert!(ptx.contains(&format!("    @{} bra BB2;\n    bra BB3;\n", n(more))), "{ptx}");
        // The scratch class is declared once; the u32 class holds i, start,
        // n, bdim, gdim, stride, next at their densest.
        assert_eq!(ptx.matches(".reg .u32 %edge_r;").count(), 1, "{ptx}");
        assert!(ptx.contains(".reg .u32 %r<"), "{ptx}");
        assert_eq!(al.pressure().of(crate::regalloc::RegClass::P), 1);
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
        let al = crate::regalloc::allocate(&ir);
        let (f, r0, r1) = (al.name(flag), al.name(p0), al.name(p1));
        let expected = format!(
            "    @!{f} bra BB1_else;\n    mov.f32 %edge_f, {r0};\n    mov.f32 {r0}, {r1};\n    mov.f32 {r1}, %edge_f;\n    bra BB1;\nBB1_else:\n    bra BB2;\n"
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
        let al = crate::regalloc::allocate(&ir);
        let (r0, r1, rv) = (al.name(p0), al.name(p1), al.name(v));
        let expected = format!("    mov.u32 {r1}, {r0};\n    mov.u32 {r0}, {rv};\n    bra BB1;\n");
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

    // ── Roadmap A2 step 4: the scalar ISA ────────────────────────────

    /// Build a one-block kernel and print it; returns the text and the
    /// allocation so expectations name registers the way the printer does.
    fn ptx_of(build: impl FnOnce(&mut KirBuilder)) -> (String, crate::regalloc::Allocation) {
        let mut b = KirBuilder::new("isa");
        let e = b.new_block();
        b.set_block(e);
        build(&mut b);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        (String::from_utf8(lower_kir_to_ptx(&ir)).unwrap(), crate::regalloc::allocate(&ir))
    }

    fn u32_const(b: &mut KirBuilder, v: u32) -> VarId {
        let d = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Const(d, KirConst { ty: KirType::U32, value: ConstValue::U32(v) }));
        d
    }

    fn f32_const(b: &mut KirBuilder, v: f32) -> VarId {
        let d = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(d, KirConst { ty: KirType::F32, value: ConstValue::F32(v) }));
        d
    }

    #[test]
    fn bitwise_and_shift_ops_print_their_class() {
        let mut ids = Vec::new();
        let (ptx, al) = ptx_of(|b| {
            let x = u32_const(b, 6);
            let y = u32_const(b, 3);
            let a = b.new_typed_var(KirType::U32);
            b.emit(KirOp::And(a, x, y));
            let o = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Or(o, x, y));
            let xo = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Xor(xo, x, y));
            let n = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Not(n, x));
            let l = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Shl(l, x, y));
            let r = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Shr(r, x, y));
            let m = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Rem(m, x, y));
            let mn = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Min(mn, x, y));
            let mx = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Max(mx, x, y));
            // 64-bit and signed shapes.
            let w = b.new_typed_var(KirType::U64);
            b.emit(KirOp::Const(w, KirConst { ty: KirType::U64, value: ConstValue::U64(9) }));
            let ws = b.new_typed_var(KirType::U64);
            b.emit(KirOp::Shl(ws, w, y));
            let wa = b.new_typed_var(KirType::U64);
            b.emit(KirOp::And(wa, w, w));
            let i = b.new_typed_var(KirType::I32);
            b.emit(KirOp::Const(i, KirConst { ty: KirType::I32, value: ConstValue::I32(-8) }));
            let ia = b.new_typed_var(KirType::I32);
            b.emit(KirOp::Shr(ia, i, y));
            ids = vec![x, y, a, o, xo, n, l, r, m, mn, mx, w, ws, wa, i, ia];
        });
        let n = |k: usize| al.name(ids[k]);
        let (x, y) = (n(0), n(1));
        for expected in [
            format!("and.b32 {}, {x}, {y};", n(2)),
            format!("or.b32 {}, {x}, {y};", n(3)),
            format!("xor.b32 {}, {x}, {y};", n(4)),
            format!("not.b32 {}, {x};", n(5)),
            format!("shl.b32 {}, {x}, {y};", n(6)),
            format!("shr.u32 {}, {x}, {y};", n(7)),
            format!("rem.u32 {}, {x}, {y};", n(8)),
            format!("min.u32 {}, {x}, {y};", n(9)),
            format!("max.u32 {}, {x}, {y};", n(10)),
            format!("shl.b64 {}, {}, {y};", n(12), n(11)),
            format!("and.b64 {}, {}, {};", n(13), n(11), n(11)),
            format!("shr.s32 {}, {}, {y};", n(15), n(14)),
        ] {
            assert!(ptx.contains(&expected), "missing `{expected}` in\n{ptx}");
        }
        assert!(n(11).starts_with("%rd") && n(14).starts_with("%r") && !n(14).starts_with("%rd"));
    }

    #[test]
    fn bool_bitwise_ops_use_the_predicate_class() {
        let mut ids = (0, 0, 0);
        let (ptx, al) = ptx_of(|b| {
            let x = u32_const(b, 1);
            let p = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Cmp(p, x, x, CmpOp::Eq));
            let q = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Not(q, p));
            let r = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::And(r, p, q));
            ids = (p, q, r);
        });
        let (p, q, r) = (al.name(ids.0), al.name(ids.1), al.name(ids.2));
        assert!(p.starts_with("%p"), "{p}");
        assert!(ptx.contains(&format!("not.pred {q}, {p};")), "{ptx}");
        assert!(ptx.contains(&format!("and.pred {r}, {p}, {q};")), "{ptx}");
        assert!(ptx.contains("    .reg .pred %p<3>;"), "{ptx}");
    }

    #[test]
    fn rcp_and_rsqrt_pick_the_float_forms() {
        let mut ids = Vec::new();
        let (ptx, al) = ptx_of(|b| {
            let x = f32_const(b, 4.0);
            let r = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Rcp(r, x));
            let q = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Rsqrt(q, x));
            let d = b.new_typed_var(KirType::F64);
            b.emit(KirOp::Const(d, KirConst { ty: KirType::F64, value: ConstValue::F64(4.0) }));
            let rd = b.new_typed_var(KirType::F64);
            b.emit(KirOp::Rcp(rd, d));
            let mn = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Min(mn, x, x));
            ids = vec![x, r, q, d, rd, mn];
        });
        let n = |k: usize| al.name(ids[k]);
        assert!(ptx.contains(&format!("rcp.approx.f32 {}, {};", n(1), n(0))), "{ptx}");
        assert!(ptx.contains(&format!("rsqrt.approx.f32 {}, {};", n(2), n(0))), "{ptx}");
        assert!(ptx.contains(&format!("rcp.rn.f64 {}, {};", n(4), n(3))), "{ptx}");
        assert!(ptx.contains(&format!("min.f32 {}, {}, {};", n(5), n(0), n(0))), "{ptx}");
        assert!(n(3).starts_with("%fd"));
    }

    #[test]
    fn casts_carry_the_rounding_ptx_requires() {
        let mut ids = Vec::new();
        let (ptx, al) = ptx_of(|b| {
            let f = f32_const(b, 1.5);
            let h = b.new_typed_var(KirType::F16);
            b.emit(KirOp::Cast(h, f, KirType::F16));
            let bf = b.new_typed_var(KirType::Bf16);
            b.emit(KirOp::Cast(bf, f, KirType::Bf16));
            let back = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Cast(back, bf, KirType::F32));
            let i = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Cast(i, f, KirType::U32));
            let g = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Cast(g, i, KirType::F32));
            let w = b.new_typed_var(KirType::U64);
            b.emit(KirOp::Cast(w, i, KirType::U64));
            let d = b.new_typed_var(KirType::F64);
            b.emit(KirOp::Cast(d, f, KirType::F64));
            let fl = b.new_typed_var(KirType::I32);
            b.emit(KirOp::CastRounded { dst: fl, src: f, ty: KirType::I32, mode: RoundMode::Rm });
            let z = b.new_typed_var(KirType::F16);
            b.emit(KirOp::CastRounded { dst: z, src: f, ty: KirType::F16, mode: RoundMode::Rz });
            ids = vec![f, h, bf, back, i, g, w, d, fl, z];
        });
        let n = |k: usize| al.name(ids[k]);
        let f = n(0);
        for expected in [
            ".version 7.8".to_string(),
            ".target sm_80".to_string(),
            ".reg .b16 %h<".to_string(),
            format!("cvt.rn.f16.f32 {}, {f};", n(1)),
            format!("cvt.rn.bf16.f32 {}, {f};", n(2)),
            format!("cvt.f32.bf16 {}, {};", n(3), n(2)),
            format!("cvt.rzi.u32.f32 {}, {f};", n(4)),
            format!("cvt.rn.f32.u32 {}, {};", n(5), n(4)),
            format!("cvt.u64.u32 {}, {};", n(6), n(4)),
            format!("cvt.f64.f32 {}, {f};", n(7)),
            format!("cvt.rmi.s32.f32 {}, {f};", n(8)),
            format!("cvt.rz.f16.f32 {}, {f};", n(9)),
        ] {
            assert!(ptx.contains(&expected), "missing `{expected}` in\n{ptx}");
        }
    }

    #[test]
    fn sixteen_bit_and_f64_classes_are_declared_up_to_the_highest_register() {
        // Roadmap A2 step 4 gate regression (fixed in the pre-allocator
        // printer, kept as a property here): every register class a kernel
        // uses is declared with a count above the highest index the body
        // names. The step-4 kernel's first bf16 value had VarId 26 and the
        // old printer wrote `.reg .b16 %h<2>`, so ptxas saw an unknown `%h26`.
        let mut names = (0, 0);
        let (ptx, al) = ptx_of(|b| {
            let f = f32_const(b, 1.5);
            for _ in 0..20 {
                let t = b.new_typed_var(KirType::F32);
                b.emit(KirOp::Add(t, f, f));
            }
            let h = b.new_typed_var(KirType::Bf16);
            b.emit(KirOp::Cast(h, f, KirType::Bf16));
            let d = b.new_typed_var(KirType::F64);
            b.emit(KirOp::Cast(d, f, KirType::F64));
            names = (h, d);
        });
        let (h, d) = (al.name(names.0), al.name(names.1));
        assert!(ptx.contains(&format!("cvt.rn.bf16.f32 {h}, ")), "{ptx}");
        assert!(ptx.contains(&format!("cvt.f64.f32 {d}, ")), "{ptx}");
        for (decl, prefix) in [(".reg .b16 %h<", "%h"), (".reg .f64 %fd<", "%fd")] {
            let at = ptx.find(decl).unwrap_or_else(|| panic!("no `{decl}` in\n{ptx}")) + decl.len();
            let declared: u32 = ptx[at..].split('>').next().unwrap().parse().unwrap();
            let body = &ptx[ptx.find("\n\n").unwrap()..];
            let highest = body
                .match_indices(prefix)
                .filter_map(|(i, _)| {
                    let digits: String = body[i + prefix.len()..]
                        .chars()
                        .take_while(|c| c.is_ascii_digit())
                        .collect();
                    digits.parse::<u32>().ok()
                })
                .max()
                .unwrap_or_else(|| panic!("no `{prefix}` register used in\n{ptx}"));
            assert!(highest < declared, "`{prefix}{highest}` used but only {declared} declared:\n{ptx}");
        }
    }

    #[test]
    fn a_kernel_without_bf16_keeps_version_7_0() {
        let (ptx, _) = ptx_of(|b| {
            let f = f32_const(b, 1.5);
            let h = b.new_typed_var(KirType::F16);
            b.emit(KirOp::Cast(h, f, KirType::F16));
        });
        assert!(ptx.contains(".version 7.0\n.target sm_70"), "{ptx}");
    }

    #[test]
    fn sixteen_bit_values_move_through_memory_as_b16() {
        let bf16_ptr = KirType::Ptr(Box::new(KirType::Bf16), AddressSpace::Global);
        let mut b = KirBuilder::new("b16");
        let p = b.add_param("p", bf16_ptr, AddressSpace::Global);
        let e = b.new_block();
        b.set_block(e);
        let v = b.new_typed_var(KirType::Bf16);
        b.emit(KirOp::Load(v, p, AddressSpace::Global));
        b.emit(KirOp::Store(p, v, AddressSpace::Global));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()));
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        let al = crate::regalloc::allocate(&ir);
        let (pr, vr) = (al.name(p), al.name(v));
        assert!(ptx.contains(&format!("ld.global.b16 {vr}, [{pr}];")), "{ptx}");
        assert!(ptx.contains(&format!("st.global.b16 [{pr}], {vr};")), "{ptx}");
        assert!(ptx.contains("    .reg .b16 %h<1>;"), "{ptx}");
    }

    #[test]
    fn vector_loads_and_stores_print_the_brace_list() {
        let f32_ptr = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        let mut b = KirBuilder::new("vec");
        let p = b.add_param("p", f32_ptr, AddressSpace::Global);
        let e = b.new_block();
        b.set_block(e);
        let d: Vec<VarId> = (0..4).map(|_| b.new_typed_var(KirType::F32)).collect();
        b.emit(KirOp::LoadVec { dsts: d.clone(), ptr: p, space: AddressSpace::Global });
        b.emit(KirOp::StoreVec { ptr: p, vals: vec![d[1], d[0]], space: AddressSpace::Global });
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()));
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        let al = crate::regalloc::allocate(&ir);
        let n = |v: VarId| al.name(v);
        assert!(
            ptx.contains(&format!("ld.global.v4.f32 {{{}, {}, {}, {}}}, [{}];", n(d[0]), n(d[1]), n(d[2]), n(d[3]), n(p))),
            "{ptx}"
        );
        assert!(ptx.contains(&format!("st.global.v2.f32 [{}], {{{}, {}}};", n(p), n(d[1]), n(d[0]))), "{ptx}");
    }

    #[test]
    fn shuffle_modes_votes_and_lane_ids() {
        let mut ids = Vec::new();
        let (ptx, al) = ptx_of(|b| {
            let x = u32_const(b, 7);
            let one = u32_const(b, 1);
            ids.push(x);
            ids.push(one);
            for (mode, width) in [
                (ShuffleMode::Down, 32),
                (ShuffleMode::Up, 32),
                (ShuffleMode::Xor, 32),
                (ShuffleMode::Idx, 32),
                (ShuffleMode::Down, 16),
            ] {
                let d = b.new_typed_var(KirType::U32);
                b.emit(KirOp::WarpShuffle { dst: d, val: x, lane: one, mode, width });
                ids.push(d);
            }
            let p = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Cmp(p, x, one, CmpOp::Gt));
            let any = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Vote { dst: any, pred: p, mode: VoteMode::Any });
            let all = b.new_typed_var(KirType::Bool);
            b.emit(KirOp::Vote { dst: all, pred: p, mode: VoteMode::All });
            let bits = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Vote { dst: bits, pred: p, mode: VoteMode::Ballot });
            let lane = b.new_typed_var(KirType::U32);
            b.emit(KirOp::LaneId(lane));
            let warp = b.new_typed_var(KirType::U32);
            b.emit(KirOp::WarpId(warp));
            ids.extend([p, any, all, bits, lane, warp]);
        });
        let n = |k: usize| al.name(ids[k]);
        let (x, one, p) = (n(0), n(1), n(7));
        for expected in [
            format!("shfl.sync.down.b32 {}, {x}, {one}, 0x1f, 0xffffffff;", n(2)),
            format!("shfl.sync.up.b32 {}, {x}, {one}, 0x0, 0xffffffff;", n(3)),
            format!("shfl.sync.bfly.b32 {}, {x}, {one}, 0x1f, 0xffffffff;", n(4)),
            format!("shfl.sync.idx.b32 {}, {x}, {one}, 0x1f, 0xffffffff;", n(5)),
            format!("shfl.sync.down.b32 {}, {x}, {one}, 0x101f, 0xffffffff;", n(6)),
            format!("vote.sync.any.pred {}, {p}, 0xffffffff;", n(8)),
            format!("vote.sync.all.pred {}, {p}, 0xffffffff;", n(9)),
            format!("vote.sync.ballot.b32 {}, {p}, 0xffffffff;", n(10)),
            format!("mov.u32 {}, %laneid;", n(11)),
            format!("mov.u32 {}, %warpid;", n(12)),
        ] {
            assert!(ptx.contains(&expected), "missing `{expected}` in\n{ptx}");
        }
    }

    #[test]
    fn a_predicated_store_is_guarded_and_a_bool_select_uses_the_scratch_predicate() {
        let f32_ptr = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        let mut b = KirBuilder::new("pred");
        let p = b.add_param("p", f32_ptr, AddressSpace::Global);
        let e = b.new_block();
        b.set_block(e);
        let x = u32_const(&mut b, 1);
        let c = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(c, x, x, CmpOp::Eq));
        let v = f32_const(&mut b, 2.0);
        b.emit(KirOp::Predicated {
            pred: c,
            negate: false,
            op: Box::new(KirOp::Store(p, v, AddressSpace::Global)),
        });
        b.emit(KirOp::Predicated {
            pred: c,
            negate: true,
            op: Box::new(KirOp::AtomicAdd(p, v, AddressSpace::Global)),
        });
        let d = b.new_typed_var(KirType::Bool);
        let nc = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Not(nc, c));
        b.emit(KirOp::Select(d, c, nc, c));
        let s = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Select(s, c, v, v));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()), "{:?}", ir.verify());
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        let al = crate::regalloc::allocate(&ir);
        let n = |v: VarId| al.name(v);
        let (pc, pp, pv, pd, pnc, ps) = (n(c), n(p), n(v), n(d), n(nc), n(s));
        assert!(ptx.contains(&format!("    @{pc} st.global.f32 [{pp}], {pv};")), "{ptx}");
        assert!(ptx.contains(&format!("    @!{pc} atom.global.add.f32")), "{ptx}");
        assert!(ptx.contains(".reg .pred %edge_p;"), "{ptx}");
        assert!(
            ptx.contains(&format!("and.pred %edge_p, {pc}, {pnc};\n    not.pred {pd}, {pc};\n    and.pred {pd}, {pd}, {pc};\n    or.pred {pd}, {pd}, %edge_p;")),
            "{ptx}"
        );
        assert!(ptx.contains(&format!("selp.f32 {ps}, {pv}, {pv}, {pc};")), "{ptx}");
    }

    // ── Roadmap A2 step 5: the allocator ─────────────────────────────

    #[test]
    fn declarations_are_the_allocated_counts_and_attributes_print() {
        let f32_ptr = KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global);
        let mut b = KirBuilder::new("dense");
        let p = b.add_param("p", f32_ptr, AddressSpace::Global);
        b.set_launch_bounds(256, Some(2));
        b.set_max_registers(64);
        let e = b.new_block();
        b.set_block(e);
        // Ten f32 values, each dead after the next one is made: two
        // registers suffice.
        let mut prev = f32_const(&mut b, 1.0);
        for _ in 0..9 {
            let next = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Neg(next, prev));
            prev = next;
        }
        b.emit(KirOp::Store(p, prev, AddressSpace::Global));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        assert_eq!(ir.verify(), Ok(()));
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        assert!(ptx.contains(") {\n    .maxntid 256, 1, 1\n    .minnctapersm 2\n    .maxnreg 64\n    .reg .u64 %rd<1>;\n    .reg .f32 %f<2>;\n\n"), "{ptx}");
        assert!(!ptx.contains(".reg .u32"), "{ptx}");
        assert!(!ptx.contains(".reg .pred"), "{ptx}");
        let pressure = ir.register_pressure();
        assert_eq!(pressure.of(crate::regalloc::RegClass::F), 2);
        assert_eq!(pressure.of(crate::regalloc::RegClass::Rd), 1);
        assert_eq!(pressure.of(crate::regalloc::RegClass::R), 0);
    }

    #[test]
    fn a_value_printed_in_a_foreign_class_gets_a_register_past_the_count() {
        // `rename_registers` on its own: v0 is an f32, but the text names
        // it as a `%r`; the pass invents `%r` index `count(R)` for it.
        let mut b = KirBuilder::new("t");
        let e = b.new_block();
        b.set_block(e);
        let x = f32_const(&mut b, 1.0);
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        let al = crate::regalloc::allocate(&ir);
        let (text, extra) = rename_registers(&format!("    mov.b32 %r{x}, %f{x};\n    mov.u32 %r7, %tid.x; %edge_r %gid0\n"), &al);
        assert_eq!(text, "    mov.b32 %r0, %f0;\n    mov.u32 %r1, %tid.x; %edge_r %gid0\n");
        assert_eq!(extra[0], 2);
    }

    /// Roadmap A2 step 3: `mul.lo` and a bare `div` are the integer
    /// spellings; a float multiply has no `.lo` and a float division
    /// carries `.rn`.
    #[test]
    fn float_mul_and_div_spell_as_ptx() {
        let mut b = KirBuilder::new("fmuldiv");
        let entry = b.new_block();
        b.set_block(entry);
        let x = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(x, KirConst { ty: KirType::F32, value: ConstValue::F32(2.0) }));
        let m = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Mul(m, x, x));
        let d = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Div(d, m, x));
        let i = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Const(i, KirConst { ty: KirType::U32, value: ConstValue::U32(3) }));
        let im = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Mul(im, i, i));
        let id = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Div(id, im, i));
        b.terminate(KirTerminator::Return);
        let ir = b.finalize();
        let ptx = String::from_utf8(lower_kir_to_ptx(&ir)).unwrap();
        let alloc = crate::regalloc::allocate(&ir);
        assert!(ptx.contains(&format!("mul.f32 {}, {}, {};", alloc.name(m), alloc.name(x), alloc.name(x))), "{ptx}");
        assert!(ptx.contains(&format!("div.rn.f32 {}, {}, {};", alloc.name(d), alloc.name(m), alloc.name(x))), "{ptx}");
        assert!(ptx.contains(&format!("mul.lo.u32 {}, {}, {};", alloc.name(im), alloc.name(i), alloc.name(i))), "{ptx}");
        assert!(ptx.contains(&format!("div.u32 {}, {}, {};", alloc.name(id), alloc.name(im), alloc.name(i))), "{ptx}");
        assert!(!ptx.contains("mul.lo.f32") && !ptx.contains("div.f32 "), "{ptx}");
    }
}
