//! A PTX interpreter for the cast-kernel subset, and the differential
//! equivalence gate for roadmap A2 step 7.
//!
//! ## What this proves, and what it does not
//!
//! The migration replaces four hand-assembled PTX modules with four built
//! from KIR. The risk is not that the new text looks different — it must,
//! since register names and labels come from the allocator — but that it
//! *computes something different*. So the gate executes both.
//!
//! `HAND_*` below are the four modules exactly as the deleted emitter
//! produced them, frozen as fixtures. They are the pre-migration
//! behaviour; keeping them here is what lets the equivalence claim outlive
//! the emitter.
//!
//! Two properties are checked per kernel:
//!
//! 1. **Agreement** — over the same input buffer, launch geometry and
//!    element count, the hand module and the KIR module leave *the same
//!    bytes* in the destination.
//! 2. **Correctness** — those bytes are what a reference cast produces.
//!    Agreement alone would be satisfied by two kernels that are wrong in
//!    the same way; this pins the shared answer to the right one.
//!
//! The interpreter models `cvt` with `half`, so it does not prove the
//! hardware rounds as `half` does — the `ptxas` gate and the GPU suites
//! cover fidelity to the machine. What it proves is the thing the
//! migration actually put at risk: same control flow, same addressing,
//! same access widths, same conversions.
//!
//! Any mnemonic the interpreter does not know is a hard error, never a
//! skip. A silently-ignored instruction would make every assertion below
//! vacuous.

use std::collections::HashMap;

const SRC_BASE: u64 = 0x1000_0000;
const DST_BASE: u64 = 0x2000_0000;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Conv {
    Bf16,
    F16,
}

#[derive(Debug, Clone)]
enum Op {
    LdParam { dst: String, name: String },
    MovSpecial { dst: String, which: String },
    Mov { dst: String, src: String },
    Mul32 { d: String, a: String, b: String },
    Add32 { d: String, a: String, b: String },
    Add64 { d: String, a: String, b: String },
    MulImm64 { d: String, a: String, imm: u64 },
    ShlImm64 { d: String, a: String, imm: u32 },
    CvtU64U32 { d: String, a: String },
    SetpGeU64 { p: String, a: String, b: String },
    BraIf { p: String, target: String },
    Bra { target: String },
    LdF32 { d: String, addr: String },
    LdB16 { d: String, addr: String },
    StF32 { addr: String, v: String },
    StB16 { addr: String, v: String },
    Narrow { d: String, s: String, conv: Conv },
    Widen { d: String, s: String, conv: Conv },
    Ret,
}

struct Program {
    ops: Vec<Op>,
    labels: HashMap<String, usize>,
}

fn strip_operand(s: &str) -> String {
    s.trim().trim_end_matches(';').trim().to_string()
}

/// Parse one PTX module into the executable subset.
///
/// Declarations (`.version`, `.reg`, the entry signature, braces) are
/// skipped; anything else that is not a recognised instruction panics, so
/// the interpreter can never quietly execute a kernel it only half
/// understands.
fn parse(ptx: &str) -> Program {
    let mut ops = Vec::new();
    let mut labels = HashMap::new();

    for raw in ptx.lines() {
        let line = raw.split("//").next().unwrap().trim();
        if line.is_empty() || line == "{" || line == "}" || line == ")" || line == "(" {
            continue;
        }
        // Declarations and the entry signature.
        if line.starts_with('.') {
            continue;
        }
        // The hand module's signature spans lines; its parameter lines
        // start with `.param` (caught above) and end with `)` (caught
        // above), so nothing is left to skip here.
        if let Some(name) = line.strip_suffix(':') {
            if !name.contains(' ') {
                labels.insert(name.to_string(), ops.len());
                continue;
            }
        }

        let op = parse_instr(line)
            .unwrap_or_else(|| panic!("interpreter does not know this instruction: `{line}`"));
        ops.push(op);
    }
    Program { ops, labels }
}

fn parse_instr(line: &str) -> Option<Op> {
    let body = line.trim_end_matches(';').trim();

    if body == "ret" {
        return Some(Op::Ret);
    }
    if let Some(rest) = body.strip_prefix("bra ") {
        return Some(Op::Bra { target: strip_operand(rest) });
    }
    if let Some(rest) = body.strip_prefix('@') {
        // `@%p0 bra BB3`
        let (p, tail) = rest.split_once(' ')?;
        let target = tail.strip_prefix("bra ")?;
        return Some(Op::BraIf { p: p.trim().to_string(), target: strip_operand(target) });
    }

    // `ld.param.u64 %rd0, [param_numel]`
    if let Some(rest) = body.strip_prefix("ld.param.u64 ") {
        let (d, addr) = rest.split_once(',')?;
        let name = addr.trim().trim_start_matches('[').trim_end_matches(']').trim();
        return Some(Op::LdParam {
            dst: strip_operand(d),
            // The two emitters name the parameter slot differently
            // (`numel` vs `param_numel`); the ABI position is what matters.
            name: name.trim_start_matches("param_").to_string(),
        });
    }

    let mem = |rest: &str| -> Option<(String, String)> {
        let (a, b) = rest.split_once(',')?;
        Some((strip_operand(a), strip_operand(b)))
    };

    if let Some(rest) = body.strip_prefix("ld.global.f32 ") {
        let (d, a) = mem(rest)?;
        return Some(Op::LdF32 { d, addr: a.trim_start_matches('[').trim_end_matches(']').to_string() });
    }
    if let Some(rest) = body.strip_prefix("ld.global.b16 ") {
        let (d, a) = mem(rest)?;
        return Some(Op::LdB16 { d, addr: a.trim_start_matches('[').trim_end_matches(']').to_string() });
    }
    if let Some(rest) = body.strip_prefix("st.global.f32 ") {
        let (a, v) = mem(rest)?;
        return Some(Op::StF32 { addr: a.trim_start_matches('[').trim_end_matches(']').to_string(), v });
    }
    if let Some(rest) = body.strip_prefix("st.global.b16 ") {
        let (a, v) = mem(rest)?;
        return Some(Op::StB16 { addr: a.trim_start_matches('[').trim_end_matches(']').to_string(), v });
    }

    for (pfx, conv) in [("cvt.rn.bf16.f32 ", Conv::Bf16), ("cvt.rn.f16.f32 ", Conv::F16)] {
        if let Some(rest) = body.strip_prefix(pfx) {
            let (d, s) = mem(rest)?;
            return Some(Op::Narrow { d, s, conv });
        }
    }
    for (pfx, conv) in [("cvt.f32.bf16 ", Conv::Bf16), ("cvt.f32.f16 ", Conv::F16)] {
        if let Some(rest) = body.strip_prefix(pfx) {
            let (d, s) = mem(rest)?;
            return Some(Op::Widen { d, s, conv });
        }
    }
    if let Some(rest) = body.strip_prefix("cvt.u64.u32 ") {
        let (d, a) = mem(rest)?;
        return Some(Op::CvtU64U32 { d, a });
    }

    if let Some(rest) = body.strip_prefix("mov.u32 ").or_else(|| body.strip_prefix("mov.u64 ")) {
        let (d, s) = mem(rest)?;
        if let Some(which) = s.strip_prefix('%') {
            if which.contains('.') {
                return Some(Op::MovSpecial { dst: d, which: which.to_string() });
            }
        }
        return Some(Op::Mov { dst: d, src: s });
    }

    let three = |rest: &str| -> Option<(String, String, String)> {
        let mut it = rest.split(',');
        let d = strip_operand(it.next()?);
        let a = strip_operand(it.next()?);
        let b = strip_operand(it.next()?);
        Some((d, a, b))
    };

    if let Some(rest) = body.strip_prefix("mul.lo.u32 ") {
        let (d, a, b) = three(rest)?;
        return Some(Op::Mul32 { d, a, b });
    }
    if let Some(rest) = body.strip_prefix("add.u32 ") {
        let (d, a, b) = three(rest)?;
        return Some(Op::Add32 { d, a, b });
    }
    if let Some(rest) = body.strip_prefix("add.u64 ") {
        let (d, a, b) = three(rest)?;
        return Some(Op::Add64 { d, a, b });
    }
    if let Some(rest) = body.strip_prefix("mul.lo.u64 ") {
        let (d, a, imm) = three(rest)?;
        return Some(Op::MulImm64 { d, a, imm: imm.parse().ok()? });
    }
    if let Some(rest) = body.strip_prefix("shl.b64 ") {
        let (d, a, imm) = three(rest)?;
        return Some(Op::ShlImm64 { d, a, imm: imm.parse().ok()? });
    }
    if let Some(rest) = body.strip_prefix("setp.ge.u64 ") {
        let (p, a, b) = three(rest)?;
        return Some(Op::SetpGeU64 { p, a, b });
    }

    None
}

/// The launch geometry and buffers one kernel runs against.
struct Launch {
    grid_dim_x: u32,
    block_dim_x: u32,
    numel: u64,
    src: Vec<u8>,
    dst: Vec<u8>,
}

struct Thread<'a> {
    regs: HashMap<String, u64>,
    preds: HashMap<String, bool>,
    tid: u32,
    ctaid: u32,
    launch: &'a mut Launch,
}

impl Thread<'_> {
    fn get(&self, r: &str) -> u64 {
        *self
            .regs
            .get(r)
            .unwrap_or_else(|| panic!("read of register `{r}` before it was written"))
    }

    /// Resolve an absolute address to (buffer, offset). An address in
    /// neither buffer is a fault, not a wrap — which is how an addressing
    /// bug surfaces here instead of silently corrupting a neighbour.
    fn resolve(&mut self, addr: u64, width: usize) -> (bool, usize) {
        let (base, len, is_src) = if addr >= DST_BASE {
            (DST_BASE, self.launch.dst.len(), false)
        } else {
            (SRC_BASE, self.launch.src.len(), true)
        };
        let off = (addr - base) as usize;
        assert!(
            off + width <= len,
            "out-of-bounds {} access at offset {off} (+{width}) in a {len}-byte buffer",
            if is_src { "source" } else { "destination" }
        );
        (is_src, off)
    }
}

fn run_thread(prog: &Program, launch: &mut Launch, ctaid: u32, tid: u32) {
    let grid = launch.grid_dim_x;
    let block = launch.block_dim_x;
    let numel = launch.numel;
    let mut t = Thread {
        regs: HashMap::new(),
        preds: HashMap::new(),
        tid,
        ctaid,
        launch,
    };

    let mut pc = 0usize;
    // A grid-stride loop over `numel` elements runs at most
    // `numel/stride + 2` iterations; the budget is generous but finite so
    // a kernel that fails to advance its induction variable fails the test
    // instead of hanging it.
    let budget = 64 * (numel + 4) as usize + 4096;
    let mut steps = 0usize;

    while pc < prog.ops.len() {
        steps += 1;
        assert!(steps < budget, "kernel did not terminate within {budget} instructions");

        match &prog.ops[pc] {
            Op::Ret => return,
            Op::Bra { target } => {
                pc = prog.labels[target];
                continue;
            }
            Op::BraIf { p, target } => {
                if t.preds[p] {
                    pc = prog.labels[target];
                    continue;
                }
            }
            Op::LdParam { dst, name } => {
                let v = match name.as_str() {
                    "src_ptr" => SRC_BASE,
                    "dst_ptr" => DST_BASE,
                    "numel" => numel,
                    other => panic!("unknown kernel parameter `{other}`"),
                };
                t.regs.insert(dst.clone(), v);
            }
            Op::MovSpecial { dst, which } => {
                let v = match which.as_str() {
                    "tid.x" => t.tid as u64,
                    "ctaid.x" => t.ctaid as u64,
                    "ntid.x" => block as u64,
                    "nctaid.x" => grid as u64,
                    other => panic!("unknown special register `%{other}`"),
                };
                t.regs.insert(dst.clone(), v);
            }
            Op::Mov { dst, src } => {
                let v = t.get(src);
                t.regs.insert(dst.clone(), v);
            }
            Op::Mul32 { d, a, b } => {
                let v = (t.get(a) as u32).wrapping_mul(t.get(b) as u32);
                t.regs.insert(d.clone(), v as u64);
            }
            Op::Add32 { d, a, b } => {
                let v = (t.get(a) as u32).wrapping_add(t.get(b) as u32);
                t.regs.insert(d.clone(), v as u64);
            }
            Op::Add64 { d, a, b } => {
                let v = t.get(a).wrapping_add(t.get(b));
                t.regs.insert(d.clone(), v);
            }
            Op::MulImm64 { d, a, imm } => {
                let v = t.get(a).wrapping_mul(*imm);
                t.regs.insert(d.clone(), v);
            }
            Op::ShlImm64 { d, a, imm } => {
                let v = t.get(a) << imm;
                t.regs.insert(d.clone(), v);
            }
            Op::CvtU64U32 { d, a } => {
                let v = t.get(a) as u32 as u64;
                t.regs.insert(d.clone(), v);
            }
            Op::SetpGeU64 { p, a, b } => {
                let v = t.get(a) >= t.get(b);
                t.preds.insert(p.clone(), v);
            }
            Op::LdF32 { d, addr } => {
                let a = t.get(addr);
                let (is_src, off) = t.resolve(a, 4);
                let buf = if is_src { &t.launch.src } else { &t.launch.dst };
                let bits = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
                t.regs.insert(d.clone(), bits as u64);
            }
            Op::LdB16 { d, addr } => {
                let a = t.get(addr);
                let (is_src, off) = t.resolve(a, 2);
                let buf = if is_src { &t.launch.src } else { &t.launch.dst };
                let bits = u16::from_le_bytes(buf[off..off + 2].try_into().unwrap());
                t.regs.insert(d.clone(), bits as u64);
            }
            Op::StF32 { addr, v } => {
                let a = t.get(addr);
                let val = t.get(v) as u32;
                let (is_src, off) = t.resolve(a, 4);
                assert!(!is_src, "cast kernel must not write to its source buffer");
                t.launch.dst[off..off + 4].copy_from_slice(&val.to_le_bytes());
            }
            Op::StB16 { addr, v } => {
                let a = t.get(addr);
                let val = t.get(v) as u16;
                let (is_src, off) = t.resolve(a, 2);
                assert!(!is_src, "cast kernel must not write to its source buffer");
                t.launch.dst[off..off + 2].copy_from_slice(&val.to_le_bytes());
            }
            Op::Narrow { d, s, conv } => {
                let f = f32::from_bits(t.get(s) as u32);
                let bits = match conv {
                    Conv::Bf16 => half::bf16::from_f32(f).to_bits(),
                    Conv::F16 => half::f16::from_f32(f).to_bits(),
                };
                t.regs.insert(d.clone(), bits as u64);
            }
            Op::Widen { d, s, conv } => {
                let bits = t.get(s) as u16;
                let f = match conv {
                    Conv::Bf16 => half::bf16::from_bits(bits).to_f32(),
                    Conv::F16 => half::f16::from_bits(bits).to_f32(),
                };
                t.regs.insert(d.clone(), f.to_bits() as u64);
            }
        }
        pc += 1;
    }
    panic!("fell off the end of the kernel without `ret`");
}

/// Run every thread of the launch. Threads write disjoint elements, so a
/// sequential sweep is equivalent to any interleaving — and the
/// destination is poisoned first, so an element no thread visits stays
/// visibly unwritten rather than accidentally correct.
fn run_kernel(ptx: &str, launch: &mut Launch) {
    let prog = parse(ptx);
    for ctaid in 0..launch.grid_dim_x {
        for tid in 0..launch.block_dim_x {
            run_thread(&prog, launch, ctaid, tid);
        }
    }
}

use nsl_kir::kernels::cast::CastKind;

// ─── the four modules as the deleted emitter produced them ────────────
//
// Frozen fixtures, not a live call: `nsl_codegen::precision_cast_ptx` is
// gone. These bytes are the behaviour the migration had to preserve, and
// keeping them here is what lets the equivalence gate below outlive the
// emitter that produced them.

const HAND_F32_TO_BF16: &str = r".version 8.0
.target sm_80
.address_size 64

.visible .entry nsl_cast_f32_to_bf16 (
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u64 numel
)
{
    .reg .pred %p_done;
    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;
    .reg .u32 %r_tmp_x, %r_tmp_y;
    .reg .u64 %rd_src, %rd_dst, %rd_numel;
    .reg .u64 %rd_idx, %rd_stride, %rd_tile;
    .reg .u64 %rd_off_bytes, %rd_addr;
    .reg .f32 %f_val;
    .reg .b16 %h_val;

    ld.param.u64 %rd_src, [src_ptr];
    ld.param.u64 %rd_dst, [dst_ptr];
    ld.param.u64 %rd_numel, [numel];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %ctaid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %nctaid_x, %nctaid.x;

    mul.lo.u32 %r_tmp_x, %ctaid_x, %ntid_x;
    add.u32 %r_tmp_x, %r_tmp_x, %tid_x;
    cvt.u64.u32 %rd_idx, %r_tmp_x;
    mul.lo.u32 %r_tmp_y, %nctaid_x, %ntid_x;
    cvt.u64.u32 %rd_stride, %r_tmp_y;

CAST_LOOP:
    setp.ge.u64 %p_done, %rd_idx, %rd_numel;
    @%p_done bra CAST_DONE;
    shl.b64 %rd_off_bytes, %rd_idx, 2;
    add.u64 %rd_addr, %rd_src, %rd_off_bytes;
    ld.global.f32 %f_val, [%rd_addr];
    cvt.rn.bf16.f32 %h_val, %f_val;
    shl.b64 %rd_off_bytes, %rd_idx, 1;
    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;
    st.global.b16 [%rd_addr], %h_val;
    add.u64 %rd_idx, %rd_idx, %rd_stride;
    bra CAST_LOOP;
CAST_DONE:
    ret;
}
";

const HAND_BF16_TO_F32: &str = r".version 8.0
.target sm_80
.address_size 64

.visible .entry nsl_cast_bf16_to_f32 (
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u64 numel
)
{
    .reg .pred %p_done;
    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;
    .reg .u32 %r_tmp_x, %r_tmp_y;
    .reg .u64 %rd_src, %rd_dst, %rd_numel;
    .reg .u64 %rd_idx, %rd_stride, %rd_tile;
    .reg .u64 %rd_off_bytes, %rd_addr;
    .reg .f32 %f_val;
    .reg .b16 %h_val;

    ld.param.u64 %rd_src, [src_ptr];
    ld.param.u64 %rd_dst, [dst_ptr];
    ld.param.u64 %rd_numel, [numel];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %ctaid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %nctaid_x, %nctaid.x;

    mul.lo.u32 %r_tmp_x, %ctaid_x, %ntid_x;
    add.u32 %r_tmp_x, %r_tmp_x, %tid_x;
    cvt.u64.u32 %rd_idx, %r_tmp_x;
    mul.lo.u32 %r_tmp_y, %nctaid_x, %ntid_x;
    cvt.u64.u32 %rd_stride, %r_tmp_y;

CAST_LOOP:
    setp.ge.u64 %p_done, %rd_idx, %rd_numel;
    @%p_done bra CAST_DONE;
    shl.b64 %rd_off_bytes, %rd_idx, 1;
    add.u64 %rd_addr, %rd_src, %rd_off_bytes;
    ld.global.b16 %h_val, [%rd_addr];
    cvt.f32.bf16 %f_val, %h_val;
    shl.b64 %rd_off_bytes, %rd_idx, 2;
    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;
    st.global.f32 [%rd_addr], %f_val;
    add.u64 %rd_idx, %rd_idx, %rd_stride;
    bra CAST_LOOP;
CAST_DONE:
    ret;
}
";

const HAND_F32_TO_FP16: &str = r".version 7.0
.target sm_80
.address_size 64

.visible .entry nsl_cast_f32_to_fp16 (
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u64 numel
)
{
    .reg .pred %p_done;
    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;
    .reg .u32 %r_tmp_x, %r_tmp_y;
    .reg .u64 %rd_src, %rd_dst, %rd_numel;
    .reg .u64 %rd_idx, %rd_stride, %rd_tile;
    .reg .u64 %rd_off_bytes, %rd_addr;
    .reg .f32 %f_val;
    .reg .b16 %h_val;

    ld.param.u64 %rd_src, [src_ptr];
    ld.param.u64 %rd_dst, [dst_ptr];
    ld.param.u64 %rd_numel, [numel];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %ctaid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %nctaid_x, %nctaid.x;

    mul.lo.u32 %r_tmp_x, %ctaid_x, %ntid_x;
    add.u32 %r_tmp_x, %r_tmp_x, %tid_x;
    cvt.u64.u32 %rd_idx, %r_tmp_x;
    mul.lo.u32 %r_tmp_y, %nctaid_x, %ntid_x;
    cvt.u64.u32 %rd_stride, %r_tmp_y;

CAST_LOOP:
    setp.ge.u64 %p_done, %rd_idx, %rd_numel;
    @%p_done bra CAST_DONE;
    shl.b64 %rd_off_bytes, %rd_idx, 2;
    add.u64 %rd_addr, %rd_src, %rd_off_bytes;
    ld.global.f32 %f_val, [%rd_addr];
    cvt.rn.f16.f32 %h_val, %f_val;
    shl.b64 %rd_off_bytes, %rd_idx, 1;
    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;
    st.global.b16 [%rd_addr], %h_val;
    add.u64 %rd_idx, %rd_idx, %rd_stride;
    bra CAST_LOOP;
CAST_DONE:
    ret;
}
";

const HAND_FP16_TO_F32: &str = r".version 7.0
.target sm_80
.address_size 64

.visible .entry nsl_cast_fp16_to_f32 (
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u64 numel
)
{
    .reg .pred %p_done;
    .reg .u32 %tid_x, %ctaid_x, %ntid_x, %nctaid_x;
    .reg .u32 %r_tmp_x, %r_tmp_y;
    .reg .u64 %rd_src, %rd_dst, %rd_numel;
    .reg .u64 %rd_idx, %rd_stride, %rd_tile;
    .reg .u64 %rd_off_bytes, %rd_addr;
    .reg .f32 %f_val;
    .reg .b16 %h_val;

    ld.param.u64 %rd_src, [src_ptr];
    ld.param.u64 %rd_dst, [dst_ptr];
    ld.param.u64 %rd_numel, [numel];

    mov.u32 %tid_x, %tid.x;
    mov.u32 %ctaid_x, %ctaid.x;
    mov.u32 %ntid_x, %ntid.x;
    mov.u32 %nctaid_x, %nctaid.x;

    mul.lo.u32 %r_tmp_x, %ctaid_x, %ntid_x;
    add.u32 %r_tmp_x, %r_tmp_x, %tid_x;
    cvt.u64.u32 %rd_idx, %r_tmp_x;
    mul.lo.u32 %r_tmp_y, %nctaid_x, %ntid_x;
    cvt.u64.u32 %rd_stride, %r_tmp_y;

CAST_LOOP:
    setp.ge.u64 %p_done, %rd_idx, %rd_numel;
    @%p_done bra CAST_DONE;
    shl.b64 %rd_off_bytes, %rd_idx, 1;
    add.u64 %rd_addr, %rd_src, %rd_off_bytes;
    ld.global.b16 %h_val, [%rd_addr];
    cvt.f32.f16 %f_val, %h_val;
    shl.b64 %rd_off_bytes, %rd_idx, 2;
    add.u64 %rd_addr, %rd_dst, %rd_off_bytes;
    st.global.f32 [%rd_addr], %f_val;
    add.u64 %rd_idx, %rd_idx, %rd_stride;
    bra CAST_LOOP;
CAST_DONE:
    ret;
}
";

fn hand_ptx(kind: CastKind) -> &'static str {
    match kind {
        CastKind::F32ToBf16 => HAND_F32_TO_BF16,
        CastKind::Bf16ToF32 => HAND_BF16_TO_F32,
        CastKind::F32ToFp16 => HAND_F32_TO_FP16,
        CastKind::Fp16ToF32 => HAND_FP16_TO_F32,
    }
}

fn kir_ptx(kind: CastKind) -> String {
    let bytes = nsl_kir::kernels::cast::ptx(kind);
    assert_eq!(bytes.last(), Some(&0), "PTX must be NUL-terminated");
    String::from_utf8(bytes[..bytes.len() - 1].to_vec()).expect("PTX must be UTF-8")
}

fn src_elem_bytes(kind: CastKind) -> usize {
    match kind {
        CastKind::F32ToBf16 | CastKind::F32ToFp16 => 4,
        _ => 2,
    }
}

fn dst_elem_bytes(kind: CastKind) -> usize {
    match kind {
        CastKind::F32ToBf16 | CastKind::F32ToFp16 => 2,
        _ => 4,
    }
}

/// The inputs each cast is exercised over: ordinary values, both signs,
/// zero, subnormal, the rounding tie cases the `.rn` modifier decides, and
/// the specials whose bit patterns a narrowing conversion must carry
/// through.
fn source_bytes(kind: CastKind, numel: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..numel {
        match src_elem_bytes(kind) {
            4 => {
                let f = match i % 12 {
                    0 => 0.0f32,
                    1 => -0.0,
                    2 => 1.0,
                    3 => -1.0,
                    4 => f32::from_bits(0x3F80_0080), // ties-to-even for bf16
                    5 => f32::from_bits(0x3F80_0180), // ties-away would differ
                    6 => 65504.0,                     // f16 max normal
                    7 => 1e-8,                        // subnormal / zero in f16
                    8 => f32::INFINITY,
                    9 => f32::NEG_INFINITY,
                    10 => f32::NAN,
                    _ => (i as f32) * 0.37 - 5.0,
                };
                out.extend_from_slice(&f.to_bits().to_le_bytes());
            }
            _ => {
                // Sweep 16-bit patterns, including the exponent extremes.
                let bits = (i.wrapping_mul(2417) & 0xFFFF) as u16;
                out.extend_from_slice(&bits.to_le_bytes());
            }
        }
    }
    out
}

/// What the cast should produce, computed independently of both kernels.
fn reference(kind: CastKind, src: &[u8], numel: u64) -> Vec<u8> {
    let mut out = Vec::new();
    for i in 0..numel as usize {
        match kind {
            CastKind::F32ToBf16 => {
                let f = f32::from_bits(u32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()));
                out.extend_from_slice(&half::bf16::from_f32(f).to_bits().to_le_bytes());
            }
            CastKind::F32ToFp16 => {
                let f = f32::from_bits(u32::from_le_bytes(src[i * 4..i * 4 + 4].try_into().unwrap()));
                out.extend_from_slice(&half::f16::from_f32(f).to_bits().to_le_bytes());
            }
            CastKind::Bf16ToF32 => {
                let b = u16::from_le_bytes(src[i * 2..i * 2 + 2].try_into().unwrap());
                out.extend_from_slice(&half::bf16::from_bits(b).to_f32().to_bits().to_le_bytes());
            }
            CastKind::Fp16ToF32 => {
                let b = u16::from_le_bytes(src[i * 2..i * 2 + 2].try_into().unwrap());
                out.extend_from_slice(&half::f16::from_bits(b).to_f32().to_bits().to_le_bytes());
            }
        }
    }
    out
}

/// Poison byte the destination is filled with before a launch, so an
/// element no thread wrote is visibly untouched instead of coincidentally
/// matching the reference.
const POISON: u8 = 0xA5;

fn launch_for(kind: CastKind, numel: u64, grid: u32, block: u32) -> Launch {
    Launch {
        grid_dim_x: grid,
        block_dim_x: block,
        numel,
        src: source_bytes(kind, numel),
        dst: vec![POISON; numel as usize * dst_elem_bytes(kind)],
    }
}

/// The launch geometries each kernel is checked under.
///
/// The interesting axis is grid coverage. When `grid*block < numel` the
/// grid-stride loop must go round more than once — the case a kernel that
/// simply indexed by global id would get wrong, and the case a 32-bit
/// induction variable would get wrong at scale. `numel = 0` must write
/// nothing at all.
const GEOMETRIES: &[(u64, u32, u32)] = &[
    (0, 1, 4),    // empty: every thread must fall straight through
    (1, 1, 4),    // fewer elements than threads
    (16, 1, 16),  // exactly one element per thread
    (17, 1, 4),   // multiple strides, with a ragged final pass
    (64, 3, 8),   // several blocks, several strides
    (100, 2, 32), // grid*block = 64 < 100
];

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones() {
    for kind in CastKind::ALL {
        let hand = hand_ptx(kind);
        let kir = kir_ptx(kind);

        for &(numel, grid, block) in GEOMETRIES {
            let mut hand_launch = launch_for(kind, numel, grid, block);
            run_kernel(hand, &mut hand_launch);

            let mut kir_launch = launch_for(kind, numel, grid, block);
            run_kernel(&kir, &mut kir_launch);

            assert_eq!(
                hand_launch.dst, kir_launch.dst,
                "{:?} disagrees with the hand-written kernel at \
                 numel={numel} grid={grid} block={block}",
                kind
            );
            // The source is read-only; a kernel that scribbled on it would
            // have tripped the store assertion, but pin the buffer too.
            assert_eq!(hand_launch.src, kir_launch.src, "{:?} source buffers diverged", kind);
        }
    }
}

#[test]
fn the_kir_kernels_compute_the_right_answer() {
    for kind in CastKind::ALL {
        let kir = kir_ptx(kind);
        for &(numel, grid, block) in GEOMETRIES {
            let mut launch = launch_for(kind, numel, grid, block);
            let expected = reference(kind, &launch.src, numel);
            run_kernel(&kir, &mut launch);
            assert_eq!(
                launch.dst, expected,
                "{:?} produced the wrong bytes at numel={numel} grid={grid} block={block}",
                kind
            );
        }
    }
}

/// Every element must be written — including under a grid too small to
/// cover the tensor in one pass. This is the property a 32-bit induction
/// variable breaks at scale and the one the grid-stride loop exists for;
/// with `POISON` prefilled, a missed element is a mismatch rather than a
/// silent zero.
#[test]
fn every_element_is_written_even_when_the_grid_is_smaller_than_the_tensor() {
    for kind in CastKind::ALL {
        let kir = kir_ptx(kind);
        let (numel, grid, block) = (257u64, 1u32, 8u32); // 8 threads, 257 elements
        let mut launch = launch_for(kind, numel, grid, block);
        run_kernel(&kir, &mut launch);
        assert!(
            !launch.dst.windows(dst_elem_bytes(kind)).any(|w| w.iter().all(|&b| b == POISON)),
            "{:?} left an element unwritten with grid*block={} < numel={numel}",
            kind,
            grid * block
        );
    }
}

/// The interpreter must reject an instruction it does not model. Without
/// this, a mnemonic it silently skipped would make every assertion above
/// vacuous — the failure mode that matters most for a differential gate.
#[test]
#[should_panic(expected = "interpreter does not know this instruction")]
fn the_interpreter_refuses_an_instruction_it_does_not_model() {
    parse(".version 7.0\nBB0:\n    wmma.load.a.sync.aligned.m16n16k16 {%r0}, [%rd0];\n    ret;\n");
}

/// A read of a register nothing defined must fault. The printer bugs this
/// migration depended on fixing produced exactly that shape of PTX — a use
/// of `%rd2` or `%r4` that no instruction wrote — so the interpreter has
/// to treat it as an error rather than reading a default zero.
#[test]
#[should_panic(expected = "before it was written")]
fn the_interpreter_faults_on_an_undefined_register() {
    let mut launch = launch_for(CastKind::F32ToBf16, 4, 1, 4);
    run_kernel(
        ".version 7.0\nBB0:\n    add.u64 %rd9, %rd7, %rd8;\n    ret;\n",
        &mut launch,
    );
}

/// The FFI contract the runtime launches against: three 64-bit parameters
/// in `(src, dst, numel)` order, and the entry name it looks up.
#[test]
fn the_kir_kernels_keep_the_ffi_signature_and_entry_name() {
    for kind in CastKind::ALL {
        let ptx = kir_ptx(kind);
        assert!(
            ptx.contains(&format!(".visible .entry {}", kind.kernel_name())),
            "{:?} must keep the entry name the runtime looks up",
            kind
        );
        let sig = ptx
            .lines()
            .find(|l| l.contains(".visible .entry"))
            .expect("an entry line");
        let params: Vec<&str> = sig.match_indices(".param .u64").map(|(_, s)| s).collect();
        assert_eq!(params.len(), 3, "{:?}: three .u64 params, got `{sig}`", kind);
        let src_at = sig.find("src_ptr").expect("src_ptr param");
        let dst_at = sig.find("dst_ptr").expect("dst_ptr param");
        let n_at = sig.find("numel").expect("numel param");
        assert!(src_at < dst_at && dst_at < n_at, "{:?}: parameter order is (src, dst, numel)", kind);
    }
}

/// `mad.lo.u32` is invalid at PTX ISA 7.0 and banned estate-wide. The
/// index arithmetic must stay `mul.lo` + `add`.
#[test]
fn no_cast_kernel_emits_the_banned_mad_mnemonic() {
    for kind in CastKind::ALL {
        assert!(
            !kir_ptx(kind).contains("mad.lo.u32"),
            "{:?} must not emit mad.lo.u32",
            kind
        );
    }
}

/// The two header differences from the hand-written modules, pinned
/// deliberately because both are *widenings* — the new modules load
/// wherever the old ones did, and more besides.
///
/// * bf16 moves from `.version 8.0` to `7.8`, which is the ISA level the
///   bf16 `cvt` mnemonics actually require. A driver that accepted 8.0
///   accepts 7.8.
/// * the f16 pair moves from `.target sm_80` to `sm_70`. The f16 `cvt`
///   mnemonics predate sm_80; PTX compiled for a lower target runs on
///   every higher one.
///
/// Neither narrows the set of devices the runtime can JIT these on, which
/// is the property that makes them safe to take.
#[test]
fn the_header_deltas_are_widenings() {
    for kind in [CastKind::F32ToBf16, CastKind::Bf16ToF32] {
        let ptx = kir_ptx(kind);
        assert!(ptx.contains(".version 7.8"), "{:?}: bf16 needs ISA 7.8", kind);
        assert!(ptx.contains(".target sm_80"), "{:?}: bf16 cvt is sm_80", kind);
        assert!(hand_ptx(kind).contains(".version 8.0"), "fixture drifted");
    }
    for kind in [CastKind::F32ToFp16, CastKind::Fp16ToF32] {
        let ptx = kir_ptx(kind);
        assert!(ptx.contains(".version 7.0"), "{:?}: f16 stays at ISA 7.0", kind);
        assert!(ptx.contains(".target sm_70"), "{:?}: f16 cvt predates sm_80", kind);
        assert!(hand_ptx(kind).contains(".target sm_80"), "fixture drifted");
    }
}
