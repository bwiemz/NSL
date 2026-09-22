//! A cooperative-CTA PTX interpreter for the decode-attention subset, and
//! the differential equivalence gate for roadmap A2 step 9.
//!
//! ## What this proves, and what it does not
//!
//! The migration replaces the hand-assembled `nsl_cfie_decode_attn` module
//! with one built from KIR. The new text must differ — registers and labels
//! come from the allocator, addresses are element indices through
//! `PtrOffset` rather than byte offsets, and shared memory is addressed
//! with 64-bit registers where the hand kernel used 32-bit ones — so the
//! gate executes both rather than comparing them (the spec's level 3).
//!
//! `hand::emit` is the pre-migration emitter itself, frozen verbatim in
//! `tests/fixtures/cfie_decode_attn_hand.rs`. The kernel is generated per
//! configuration (its strides are baked immediates), so the generator is
//! the fixture: a text fixture would freeze one geometry.
//!
//! Three properties are checked:
//!
//! 1. **Agreement** — for every geometry, sequence length, layer and slot
//!    below, and under two thread schedules, the hand module and the KIR
//!    module leave *the same bytes* in all of global memory.
//! 2. **Correctness** — those bytes are attention: they match
//!    `cfie_decode_attention::cpu_reference` fed the same f16 K/V.
//!    Agreement alone would be satisfied by two kernels wrong the same way.
//! 3. **The gate bites** — deleting any one barrier, nudging a baked
//!    stride or the softmax scale, or dropping the tail-tile clamp in the
//!    KIR text is caught.
//!
//! The CTA is executed cooperatively: each thread runs until it reaches a
//! `bar.sync` or `ret`, and a barrier releases only when every thread of
//! the CTA is waiting at one — a thread that exits while others wait is a
//! fault, as a `bar.sync` some threads never reach is a hang on hardware.
//! Every launch runs twice, threads visited in ascending and in descending
//! order: a kernel whose result depends on the order threads run between
//! barriers has a race, and the two schedules expose each barrier this
//! kernel relies on (the mutation tests below prove that, barrier by
//! barrier).
//!
//! The interpreter models `ex2.approx.f32` as `f32::exp2` and `cvt` with
//! `half`, so it does not prove the hardware's approximation — the device
//! parity suite (`cfie_decode_attn_gpu_parity.rs`) and the `ptxas` gate
//! carry fidelity to the machine. What it proves is what the migration put
//! at risk: same control flow, same addressing, same access widths, same
//! floating-point operations in the same order.
//!
//! Any mnemonic or operand form the interpreter does not know is a hard
//! error, never a skip; so is reading a register nothing wrote, and any
//! access outside the buffers a launch provides.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_decode_attention::{
    cpu_reference, emit_decode_attention_ptx, DecodeAttentionConfig,
};

#[allow(dead_code)]
#[path = "fixtures/cfie_decode_attn_hand.rs"]
mod hand;

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Where shared memory starts in the interpreter's shared window. Non-zero
/// so a null shared pointer faults; small so the hand kernel's 32-bit
/// shared addresses hold it.
const SHARED_BASE: u64 = 0x100;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Special {
    TidX,
    CtaidX,
    NtidX,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Src {
    Reg(usize),
    Imm(u64),
    Special(Special),
}

/// `[reg]` or `[reg+imm]`.
#[derive(Debug, Clone, Copy)]
struct Addr {
    base: usize,
    offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum W {
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum IntOp {
    Add,
    Sub,
    MulLo,
    Div,
    Min,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum FOp {
    Add,
    Sub,
    Mul,
    Max,
    DivRn,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmpTy {
    U32,
    U64,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Space {
    Global,
    Shared,
}

#[derive(Debug, Clone)]
enum Op {
    LdParam { d: usize, param: String },
    Mov { d: usize, s: Src, w: W },
    Int { op: IntOp, w: W, d: usize, a: Src, b: Src },
    MulWide { d: usize, a: Src, b: Src },
    CvtU64U32 { d: usize, a: Src },
    CvtF32F16 { d: usize, a: Src },
    Setp { cmp: Cmp, ty: CmpTy, d: usize, a: Src, b: Src },
    F { op: FOp, d: usize, a: Src, b: Src },
    Fma { d: usize, a: Src, b: Src, c: Src },
    Ex2 { d: usize, a: Src },
    Ld { space: Space, bytes: usize, d: usize, addr: Addr },
    St { space: Space, bytes: usize, addr: Addr, v: Src },
    Bra { target: usize },
    Bar,
    Ret,
}

#[derive(Debug, Clone)]
struct Instr {
    /// `@%p` / `@!%p`: (predicate register, negated).
    guard: Option<(usize, bool)>,
    op: Op,
    text: String,
}

#[derive(Debug)]
struct Program {
    /// `(ptx type, name)` per entry parameter, in declaration order, the
    /// `param_` prefix the KIR printer adds stripped.
    params: Vec<(String, String)>,
    instrs: Vec<Instr>,
    reg_names: Vec<String>,
    /// Shared symbol name -> (window address, bytes).
    shared: HashMap<String, (u64, usize)>,
    shared_bytes: usize,
}

struct Parser {
    regs: HashMap<String, usize>,
    reg_names: Vec<String>,
    shared: HashMap<String, (u64, usize)>,
}

impl Parser {
    fn reg(&mut self, name: &str) -> usize {
        assert!(
            name.starts_with('%') && name.len() > 1,
            "`{name}` is not a register"
        );
        if let Some(&i) = self.regs.get(name) {
            return i;
        }
        let i = self.reg_names.len();
        self.regs.insert(name.to_string(), i);
        self.reg_names.push(name.to_string());
        i
    }

    fn src(&mut self, tok: &str) -> Src {
        let tok = tok.trim();
        match tok {
            "%tid.x" => return Src::Special(Special::TidX),
            "%ctaid.x" => return Src::Special(Special::CtaidX),
            "%ntid.x" => return Src::Special(Special::NtidX),
            _ => {}
        }
        if tok.starts_with('%') {
            assert!(!tok.contains('.'), "special register `{tok}` is not modelled");
            return Src::Reg(self.reg(tok));
        }
        if let Some(hex) = tok.strip_prefix("0f") {
            assert_eq!(hex.len(), 8, "`{tok}` is not an f32 immediate");
            return Src::Imm(u32::from_str_radix(hex, 16).expect("f32 immediate") as u64);
        }
        if let Ok(v) = tok.parse::<u64>() {
            return Src::Imm(v);
        }
        if let Some(&(addr, _)) = self.shared.get(tok) {
            return Src::Imm(addr);
        }
        panic!("operand `{tok}` is not modelled");
    }

    fn dst(&mut self, tok: &str) -> usize {
        match self.src(tok) {
            Src::Reg(r) => r,
            other => panic!("destination `{tok}` is {other:?}, not a register"),
        }
    }

    fn addr(&mut self, tok: &str) -> Addr {
        let inner = tok
            .trim()
            .strip_prefix('[')
            .and_then(|t| t.strip_suffix(']'))
            .unwrap_or_else(|| panic!("`{tok}` is not an address operand"));
        let (base, offset) = match inner.split_once('+') {
            Some((b, o)) => (b, o.trim().parse::<u64>().expect("address offset")),
            None => (inner, 0),
        };
        Addr { base: self.reg(base.trim()), offset }
    }
}

fn parse_signature(ptx: &str) -> Vec<(String, String)> {
    let start = ptx.find(".visible .entry ").expect("an entry");
    let open = start + ptx[start..].find('(').expect("a parameter list");
    let close = open + ptx[open..].find(')').expect("a closed parameter list");
    ptx[open + 1..close]
        .split(',')
        .map(|p| {
            let words: Vec<&str> = p.split_whitespace().collect();
            assert_eq!(words.len(), 3, "parameter `{p}`");
            assert_eq!(words[0], ".param");
            (words[1].to_string(), words[2].trim_start_matches("param_").to_string())
        })
        .collect()
}

/// Parse one module into the executable subset. Declarations are read for
/// the shared block and the signature and otherwise skipped; anything else
/// that is not a recognised instruction panics.
fn parse(ptx: &str) -> Program {
    let mut p = Parser { regs: HashMap::new(), reg_names: Vec::new(), shared: HashMap::new() };
    let mut shared_bytes = 0usize;

    // `.shared .align A .b8 NAME[N];`
    for line in ptx.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix(".shared ") {
            let decl = rest.split_whitespace().last().expect("a shared symbol");
            let (name, n) = decl.trim_end_matches(';').trim_end_matches(']').split_once('[').expect("a sized block");
            let n: usize = n.parse().expect("a static shared size");
            p.shared.insert(name.to_string(), (SHARED_BASE + shared_bytes as u64, n));
            shared_bytes += n;
        }
        assert!(!line.starts_with(".extern"), "dynamic shared memory is not modelled");
    }

    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut pending: Vec<(String, String, Option<(usize, bool)>)> = Vec::new(); // (mnemonic, operands, guard)

    for raw in ptx.lines() {
        let line = raw.split("//").next().unwrap().trim();
        if line.is_empty() || line.starts_with('.') || matches!(line, "{" | "}" | ")" | "(") {
            continue;
        }
        if let Some(name) = line.strip_suffix(':') {
            assert!(!name.contains(' '), "`{line}` is not a label");
            labels.insert(name.to_string(), pending.len());
            continue;
        }
        let body = line.strip_suffix(';').unwrap_or_else(|| panic!("`{line}` is not a statement"));
        let (guard, body) = match body.strip_prefix('@') {
            Some(rest) => {
                let (pred, tail) = rest.split_once(' ').expect("a guarded instruction");
                let (negated, pred) = match pred.strip_prefix('!') {
                    Some(p) => (true, p),
                    None => (false, pred),
                };
                (Some((p.reg(pred), negated)), tail.trim())
            }
            None => (None, body),
        };
        let (mnemonic, operands) = body.split_once(' ').unwrap_or((body, ""));
        pending.push((mnemonic.to_string(), operands.to_string(), guard));
    }

    let mut instrs = Vec::with_capacity(pending.len());
    for (mnemonic, operands, guard) in pending {
        let ops: Vec<&str> = if operands.is_empty() {
            vec![]
        } else {
            operands.split(',').map(str::trim).collect()
        };
        let text = format!("{mnemonic} {operands}");
        let want = |n: usize| assert_eq!(ops.len(), n, "`{text}` takes {n} operands");
        let parts: Vec<&str> = mnemonic.split('.').collect();
        let op = match parts.as_slice() {
            ["ld", "param", "u64" | "u32"] => {
                want(2);
                let name = ops[1].trim_start_matches('[').trim_end_matches(']');
                Op::LdParam { d: p.dst(ops[0]), param: name.trim_start_matches("param_").to_string() }
            }
            ["mov", ty] => {
                want(2);
                let w = match *ty {
                    "u32" | "b32" | "f32" | "pred" => W::U32,
                    "u64" | "b64" => W::U64,
                    _ => panic!("`{text}`: mov.{ty} is not modelled"),
                };
                Op::Mov { d: p.dst(ops[0]), s: p.src(ops[1]), w }
            }
            [name @ ("add" | "sub" | "div" | "min"), ty @ ("u32" | "u64")]
            | [name @ "mul", "lo", ty @ ("u32" | "u64")] => {
                want(3);
                let op = match *name {
                    "add" => IntOp::Add,
                    "sub" => IntOp::Sub,
                    "mul" => IntOp::MulLo,
                    "div" => IntOp::Div,
                    _ => IntOp::Min,
                };
                let w = if *ty == "u32" { W::U32 } else { W::U64 };
                Op::Int { op, w, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            ["mul", "wide", "u32"] => {
                want(3);
                Op::MulWide { d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            ["cvt", "u64", "u32"] => {
                want(2);
                Op::CvtU64U32 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "f32", "f16"] => {
                want(2);
                Op::CvtF32F16 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["setp", cmp, ty] => {
                want(3);
                let cmp = match *cmp {
                    "eq" => Cmp::Eq,
                    "ne" => Cmp::Ne,
                    "lt" => Cmp::Lt,
                    "le" => Cmp::Le,
                    "gt" => Cmp::Gt,
                    "ge" => Cmp::Ge,
                    _ => panic!("`{text}`: comparison not modelled"),
                };
                let ty = match *ty {
                    "u32" => CmpTy::U32,
                    "u64" => CmpTy::U64,
                    "f32" => CmpTy::F32,
                    _ => panic!("`{text}`: comparison type not modelled"),
                };
                Op::Setp { cmp, ty, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            [name @ ("add" | "sub" | "mul" | "max"), "f32"] | [name @ "div", "rn", "f32"] => {
                want(3);
                let op = match *name {
                    "add" => FOp::Add,
                    "sub" => FOp::Sub,
                    "mul" => FOp::Mul,
                    "max" => FOp::Max,
                    _ => FOp::DivRn,
                };
                Op::F { op, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            ["fma", "rn", "f32"] => {
                want(4);
                Op::Fma { d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]), c: p.src(ops[3]) }
            }
            ["ex2", "approx", "f32"] => {
                want(2);
                Op::Ex2 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["ld", space @ ("global" | "shared"), ty @ ("f32" | "b16")] => {
                want(2);
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                let bytes = if *ty == "f32" { 4 } else { 2 };
                Op::Ld { space, bytes, d: p.dst(ops[0]), addr: p.addr(ops[1]) }
            }
            ["st", space @ ("global" | "shared"), "f32"] => {
                want(2);
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                Op::St { space, bytes: 4, addr: p.addr(ops[0]), v: p.src(ops[1]) }
            }
            ["bra"] => {
                want(1);
                // Resolved below, once every label is known.
                Op::Bra { target: usize::MAX }
            }
            ["bar", "sync"] => {
                assert_eq!(ops, vec!["0"], "`{text}`: only barrier 0 is modelled");
                Op::Bar
            }
            ["ret"] => {
                want(0);
                Op::Ret
            }
            _ => panic!("interpreter does not know this instruction: `{text}`"),
        };
        let op = match op {
            Op::Bra { .. } => Op::Bra {
                target: *labels.get(ops[0]).unwrap_or_else(|| panic!("`{text}`: no label {}", ops[0])),
            },
            other => other,
        };
        instrs.push(Instr { guard, op, text });
    }

    Program {
        params: parse_signature(ptx),
        instrs,
        reg_names: p.reg_names,
        shared: p.shared,
        shared_bytes,
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// A global buffer the launch provides. Every global access must fall
/// entirely inside one.
struct Segment {
    base: u64,
    bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Order {
    Ascending,
    Descending,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Running,
    AtBarrier,
    Exited,
}

struct Thread {
    pc: usize,
    regs: Vec<u64>,
    written: Vec<bool>,
    state: State,
}

struct Launch<'a> {
    prog: &'a Program,
    args: &'a HashMap<String, u64>,
    global: &'a mut [Segment],
    shared: Vec<u8>,
    ctaid: u32,
    ntid: u32,
    steps: u64,
}

const STEP_LIMIT: u64 = 200_000_000;

impl Launch<'_> {
    fn global(&mut self, addr: u64, len: usize) -> &mut [u8] {
        for seg in self.global.iter_mut() {
            if addr >= seg.base && addr + len as u64 <= seg.base + seg.bytes.len() as u64 {
                let at = (addr - seg.base) as usize;
                return &mut seg.bytes[at..at + len];
            }
        }
        panic!("global access of {len} bytes at {addr:#x} is outside every buffer");
    }

    fn shared(&mut self, addr: u64, len: usize) -> &mut [u8] {
        let end = SHARED_BASE + self.shared.len() as u64;
        assert!(
            addr >= SHARED_BASE && addr + len as u64 <= end,
            "shared access of {len} bytes at {addr:#x} is outside [{SHARED_BASE:#x}, {end:#x})"
        );
        let at = (addr - SHARED_BASE) as usize;
        &mut self.shared[at..at + len]
    }
}

fn read(prog: &Program, t: &Thread, launch: &Launch, tid: u32, s: Src, at: &str) -> u64 {
    match s {
        Src::Reg(r) => {
            assert!(
                t.written[r],
                "thread {tid} read `{}` before anything wrote it, in `{at}`",
                prog.reg_names[r]
            );
            t.regs[r]
        }
        Src::Imm(v) => v,
        Src::Special(Special::TidX) => tid as u64,
        Src::Special(Special::CtaidX) => launch.ctaid as u64,
        Src::Special(Special::NtidX) => launch.ntid as u64,
    }
}

fn f(bits: u64) -> f32 {
    f32::from_bits(bits as u32)
}

fn fb(v: f32) -> u64 {
    v.to_bits() as u64
}

/// Run thread `tid` until it waits at a barrier or exits.
fn run_until_blocked(t: &mut Thread, launch: &mut Launch, tid: u32) {
    let prog = launch.prog;
    loop {
        launch.steps += 1;
        assert!(launch.steps < STEP_LIMIT, "kernel did not terminate");
        let instr = prog.instrs.get(t.pc).unwrap_or_else(|| panic!("thread {tid} ran off the end"));
        t.pc += 1;
        if let Some((p, negated)) = instr.guard {
            let taken = read(prog, t, launch, tid, Src::Reg(p), &instr.text) != 0;
            if taken == negated {
                continue;
            }
        }
        let rd = |t: &Thread, launch: &Launch, s: Src| read(prog, t, launch, tid, s, &instr.text);
        let write = |t: &mut Thread, d: usize, v: u64| {
            t.regs[d] = v;
            t.written[d] = true;
        };
        match &instr.op {
            Op::LdParam { d, param } => {
                let v = *launch.args.get(param).unwrap_or_else(|| panic!("no argument `{param}`"));
                write(t, *d, v);
            }
            Op::Mov { d, s, w } => {
                let v = rd(t, launch, *s);
                write(t, *d, if *w == W::U32 { v as u32 as u64 } else { v });
            }
            Op::Int { op, w, d, a, b } => {
                let (a, b) = (rd(t, launch, *a), rd(t, launch, *b));
                let v = match w {
                    W::U32 => {
                        let (a, b) = (a as u32, b as u32);
                        (match op {
                            IntOp::Add => a.wrapping_add(b),
                            IntOp::Sub => a.wrapping_sub(b),
                            IntOp::MulLo => a.wrapping_mul(b),
                            IntOp::Div => a.checked_div(b).expect("u32 division by zero"),
                            IntOp::Min => a.min(b),
                        }) as u64
                    }
                    W::U64 => match op {
                        IntOp::Add => a.wrapping_add(b),
                        IntOp::Sub => a.wrapping_sub(b),
                        IntOp::MulLo => a.wrapping_mul(b),
                        IntOp::Div => a.checked_div(b).expect("u64 division by zero"),
                        IntOp::Min => a.min(b),
                    },
                };
                write(t, *d, v);
            }
            Op::MulWide { d, a, b } => {
                let v = (rd(t, launch, *a) as u32 as u64) * (rd(t, launch, *b) as u32 as u64);
                write(t, *d, v);
            }
            Op::CvtU64U32 { d, a } => {
                let v = rd(t, launch, *a) as u32 as u64;
                write(t, *d, v);
            }
            Op::CvtF32F16 { d, a } => {
                let v = f16::from_bits(rd(t, launch, *a) as u16).to_f32();
                write(t, *d, fb(v));
            }
            Op::Setp { cmp, ty, d, a, b } => {
                let (a, b) = (rd(t, launch, *a), rd(t, launch, *b));
                let r = match ty {
                    CmpTy::U32 | CmpTy::U64 => {
                        let (a, b) = if *ty == CmpTy::U32 { (a as u32 as u64, b as u32 as u64) } else { (a, b) };
                        match cmp {
                            Cmp::Eq => a == b,
                            Cmp::Ne => a != b,
                            Cmp::Lt => a < b,
                            Cmp::Le => a <= b,
                            Cmp::Gt => a > b,
                            Cmp::Ge => a >= b,
                        }
                    }
                    // PTX's ordered comparisons are false on NaN, `ne` too
                    // (`neu` is the unordered form) — as Rust's operators.
                    CmpTy::F32 => {
                        let (a, b) = (f(a), f(b));
                        match cmp {
                            Cmp::Eq => a == b,
                            Cmp::Ne => a < b || a > b,
                            Cmp::Lt => a < b,
                            Cmp::Le => a <= b,
                            Cmp::Gt => a > b,
                            Cmp::Ge => a >= b,
                        }
                    }
                };
                write(t, *d, r as u64);
            }
            Op::F { op, d, a, b } => {
                let (a, b) = (f(rd(t, launch, *a)), f(rd(t, launch, *b)));
                let v = match op {
                    FOp::Add => a + b,
                    FOp::Sub => a - b,
                    FOp::Mul => a * b,
                    // PTX `max.f32` returns the non-NaN operand, as Rust's.
                    FOp::Max => a.max(b),
                    FOp::DivRn => a / b,
                };
                write(t, *d, fb(v));
            }
            Op::Fma { d, a, b, c } => {
                let v = f(rd(t, launch, *a)).mul_add(f(rd(t, launch, *b)), f(rd(t, launch, *c)));
                write(t, *d, fb(v));
            }
            Op::Ex2 { d, a } => {
                let v = f(rd(t, launch, *a)).exp2();
                write(t, *d, fb(v));
            }
            Op::Ld { space, bytes, d, addr } => {
                let at = rd(t, launch, Src::Reg(addr.base)).wrapping_add(addr.offset);
                let mem = match space {
                    Space::Global => launch.global(at, *bytes),
                    Space::Shared => launch.shared(at, *bytes),
                };
                let mut buf = [0u8; 8];
                buf[..*bytes].copy_from_slice(mem);
                write(t, *d, u64::from_le_bytes(buf));
            }
            Op::St { space, bytes, addr, v } => {
                let at = rd(t, launch, Src::Reg(addr.base)).wrapping_add(addr.offset);
                let val = rd(t, launch, *v).to_le_bytes();
                let mem = match space {
                    Space::Global => launch.global(at, *bytes),
                    Space::Shared => launch.shared(at, *bytes),
                };
                mem.copy_from_slice(&val[..*bytes]);
            }
            Op::Bra { target } => t.pc = *target,
            Op::Bar => {
                t.state = State::AtBarrier;
                return;
            }
            Op::Ret => {
                t.state = State::Exited;
                return;
            }
        }
    }
}

/// Run one CTA to completion under `order`.
fn run_cta(launch: &mut Launch, order: Order) {
    let n = launch.ntid;
    let regs = launch.prog.reg_names.len();
    let mut threads: Vec<Thread> = (0..n)
        .map(|_| Thread { pc: 0, regs: vec![0; regs], written: vec![false; regs], state: State::Running })
        .collect();
    let visit: Vec<u32> = match order {
        Order::Ascending => (0..n).collect(),
        Order::Descending => (0..n).rev().collect(),
    };
    loop {
        for &tid in &visit {
            let t = &mut threads[tid as usize];
            if t.state == State::Running {
                run_until_blocked(t, launch, tid);
            }
        }
        let waiting = threads.iter().filter(|t| t.state == State::AtBarrier).count();
        let exited = threads.iter().filter(|t| t.state == State::Exited).count();
        if exited == threads.len() {
            return;
        }
        assert_eq!(
            waiting,
            threads.len(),
            "CTA {}: {waiting} threads wait at a barrier that {exited} exited threads never reach",
            launch.ctaid
        );
        for t in threads.iter_mut() {
            t.state = State::Running;
        }
    }
}

// ---------------------------------------------------------------------------
// The kernel under test
// ---------------------------------------------------------------------------

const Q_BASE: u64 = 0x1000_0000;
const KV_BASE: u64 = 0x2000_0000;
const OUT_BASE: u64 = 0x3000_0000;
const BLOCK: u32 = 128;

#[derive(Debug, Clone, Copy)]
struct Geometry {
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    per_slot: u32,
    slots: u32,
}

impl Geometry {
    fn cfg(self) -> DecodeAttentionConfig {
        DecodeAttentionConfig {
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            kv_dtype_bytes: 2,
        }
    }

    fn hand_cfg(self) -> hand::DecodeAttentionConfig {
        hand::DecodeAttentionConfig {
            n_layers: self.n_layers,
            n_heads: self.n_heads,
            n_kv_heads: self.n_kv_heads,
            head_dim: self.head_dim,
            per_slot_max_tokens: self.per_slot,
            max_slots: self.slots,
            kv_dtype_bytes: 2,
            sm_version: 80,
        }
    }

    fn token_elems(self) -> usize {
        (self.n_kv_heads * self.head_dim) as usize
    }

    fn max_tokens(self) -> usize {
        (self.slots * self.per_slot) as usize
    }

    /// Element index of `(layer, plane, token, kv_head, d)` in the pool.
    fn kv_index(self, layer: u32, plane: u32, token: usize, kv_head: u32, d: u32) -> usize {
        (((layer as usize * 2 + plane as usize) * self.max_tokens() + token) * self.n_kv_heads as usize
            + kv_head as usize)
            * self.head_dim as usize
            + d as usize
    }

    fn pool_elems(self) -> usize {
        self.n_layers as usize * 2 * self.max_tokens() * self.token_elems()
    }
}

/// Deterministic values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

struct Inputs {
    q: Vec<f32>,
    /// The whole pool, every layer and slot filled, so an address that
    /// strays into a neighbouring layer, plane, slot or head reads a
    /// different value rather than a zero.
    pool: Vec<f16>,
}

fn inputs(g: Geometry, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let q = (0..g.n_heads * g.head_dim).map(|_| rng.next()).collect();
    let pool = (0..g.pool_elems()).map(|_| f16::from_f32(rng.next())).collect();
    Inputs { q, pool }
}

#[derive(Debug, Clone, Copy)]
struct Call {
    layer: u32,
    slot: u32,
    seq_len: u32,
}

/// Global memory after running `ptx` over every CTA: `[q, kv, out]`.
fn run(ptx: &str, g: Geometry, input: &Inputs, call: Call, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let q_bytes: Vec<u8> = input.q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kv_bytes: Vec<u8> = input.pool.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    // A NaN sentinel: an output element the kernel fails to write shows.
    let out_bytes: Vec<u8> = (0..g.n_heads * g.head_dim).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect();
    let mut global = vec![
        Segment { base: Q_BASE, bytes: q_bytes },
        Segment { base: KV_BASE, bytes: kv_bytes },
        Segment { base: OUT_BASE, bytes: out_bytes },
    ];
    let args: HashMap<String, u64> = [
        ("q_ptr", Q_BASE),
        ("kv_base", KV_BASE),
        ("out_ptr", OUT_BASE),
        ("layer_idx", call.layer as u64),
        ("slot_idx", call.slot as u64),
        ("seq_len", call.seq_len as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    for ctaid in 0..g.n_heads {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            // Poisoned (0xFF.. is a NaN as f32): a read of shared memory
            // no thread wrote this launch shows in the output.
            shared: vec![0xFF; prog.shared_bytes],
            ctaid,
            ntid: BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn kir_ptx(g: Geometry) -> String {
    emit_decode_attention_ptx(&g.cfg())
}

fn hand_ptx(g: Geometry) -> String {
    hand::emit_decode_attention_ptx(&g.hand_cfg())
}

fn out_f32(mem: &[Vec<u8>]) -> Vec<f32> {
    mem[2].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// What attention over the slot's first `seq_len` tokens produces, from
/// `cpu_reference` fed the same f16 values the kernel reads.
fn reference(g: Geometry, input: &Inputs, call: Call) -> Vec<f32> {
    let first = (call.slot * g.per_slot) as usize;
    let rows = |plane: u32| -> Vec<f32> {
        let mut out = Vec::new();
        for t in 0..call.seq_len as usize {
            for h in 0..g.n_kv_heads {
                for d in 0..g.head_dim {
                    out.push(input.pool[g.kv_index(call.layer, plane, first + t, h, d)].to_f32());
                }
            }
        }
        out
    };
    cpu_reference(&input.q, &rows(0), &rows(1), g.n_heads, g.n_kv_heads, g.head_dim, call.seq_len)
}

/// The geometries and calls the agreement and correctness gates sweep.
fn cases() -> Vec<(Geometry, Vec<Call>)> {
    vec![
        // GQA (group 2), a mid-pool layer and slot, and sequence lengths
        // that cross every tile edge: empty, one token, one short of a
        // tile, exactly one, one over, and three tiles with a ragged tail.
        (
            Geometry { n_layers: 2, n_heads: 2, n_kv_heads: 1, head_dim: 8, per_slot: 300, slots: 3 },
            [0, 1, 127, 128, 129, 300]
                .into_iter()
                .map(|seq_len| Call { layer: 1, slot: 1, seq_len })
                .chain([Call { layer: 0, slot: 2, seq_len: 77 }, Call { layer: 1, slot: 0, seq_len: 256 }])
                .collect(),
        ),
        // MHA with a head_dim that is not a power of two: threads 40..128
        // compute scores but own no output element.
        (
            Geometry { n_layers: 1, n_heads: 3, n_kv_heads: 3, head_dim: 40, per_slot: 200, slots: 1 },
            vec![Call { layer: 0, slot: 0, seq_len: 5 }, Call { layer: 0, slot: 0, seq_len: 200 }],
        ),
        // head_dim == block: every thread owns an output element.
        (
            Geometry { n_layers: 1, n_heads: 1, n_kv_heads: 1, head_dim: 128, per_slot: 130, slots: 1 },
            vec![Call { layer: 0, slot: 0, seq_len: 130 }],
        ),
        // head_dim 1, group 4.
        (
            Geometry { n_layers: 3, n_heads: 4, n_kv_heads: 1, head_dim: 1, per_slot: 129, slots: 2 },
            vec![Call { layer: 2, slot: 1, seq_len: 129 }],
        ),
    ]
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    for (g, calls) in cases() {
        let (hand, kir) = (hand_ptx(g), kir_ptx(g));
        let input = inputs(g, 0x5eed ^ g.head_dim as u64);
        for call in calls {
            for order in [Order::Ascending, Order::Descending] {
                let expect = run(&hand, g, &input, call, order);
                let got = run(&kir, g, &input, call, order);
                assert!(
                    expect == got,
                    "{g:?} {call:?} {order:?}: global memory differs\nhand: {:?}\nkir:  {:?}",
                    out_f32(&expect),
                    out_f32(&got)
                );
            }
        }
    }
}

#[test]
fn the_shared_answer_is_attention() {
    for (g, calls) in cases() {
        let kir = kir_ptx(g);
        let input = inputs(g, 0x5eed ^ g.head_dim as u64);
        for call in calls {
            let got = out_f32(&run(&kir, g, &input, call, Order::Ascending));
            let want = reference(g, &input, call);
            assert_eq!(got.len(), want.len());
            for (i, (a, b)) in got.iter().zip(&want).enumerate() {
                // The kernel's exp is 2^(x*log2 e) and it accumulates in a
                // different order from the reference; nothing else differs.
                assert!(
                    (a - b).abs() <= 2e-5 * (1.0 + b.abs()),
                    "{g:?} {call:?}: out[{i}] = {a}, reference {b}"
                );
            }
        }
    }
}

#[test]
fn inputs_are_left_untouched() {
    let (g, calls) = cases().remove(0);
    let input = inputs(g, 7);
    let mem = run(&kir_ptx(g), g, &input, calls[5], Order::Descending);
    let q: Vec<u8> = input.q.iter().flat_map(|v| v.to_le_bytes()).collect();
    let kv: Vec<u8> = input.pool.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
    assert!(mem[0] == q && mem[1] == kv, "the kernel wrote to its inputs");
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    for (g, _) in cases() {
        let (hand, kir) = (parse(&hand_ptx(g)), parse(&kir_ptx(g)));
        assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
        let names: Vec<&str> = kir.params.iter().map(|(_, n)| n.as_str()).collect();
        assert_eq!(names, ["q_ptr", "kv_base", "out_ptr", "layer_idx", "slot_idx", "seq_len"]);
        assert!(kir_ptx(g).contains(".visible .entry nsl_cfie_decode_attn("));
        // Same shared footprint, one static block.
        let (hand_smem, kir_smem) = (hand.shared_bytes, kir.shared_bytes);
        assert_eq!(hand_smem, kir_smem);
        assert_eq!((hand.shared.len(), kir.shared.len()), (1, 1));
    }
}

#[test]
fn the_kir_kernel_keeps_the_hand_kernels_baked_header_lines() {
    // Sibling CFIE kernels compare these lines with the decode kernel's
    // byte for byte (see cfie_persistent_ptx / cfie_speculative_ptx).
    for (g, _) in cases() {
        let header = |ptx: &str| -> Vec<String> {
            ptx.lines().take_while(|l| l.starts_with("//")).map(str::to_string).collect()
        };
        assert_eq!(header(&hand_ptx(g)), header(&kir_ptx(g)), "{g:?}");
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// The multi-tile, ragged-tail case every mutation is judged on. Its
/// baked constants are pairwise distinct (head_dim 8, token stride 16,
/// per-slot 300, kv half 14400, layer 28800, group 2, tile 128), so each
/// can be nudged on its own.
fn mutation_case() -> (Geometry, Call) {
    (
        Geometry { n_layers: 2, n_heads: 4, n_kv_heads: 2, head_dim: 8, per_slot: 300, slots: 3 },
        Call { layer: 1, slot: 1, seq_len: 300 },
    )
}

/// Whether `mutant` is told apart from the hand kernel: under either
/// schedule its memory differs, or the interpreter faults on it.
fn caught(mutant: &str) -> bool {
    let (g, call) = mutation_case();
    let input = inputs(g, 11);
    let hand = hand_ptx(g);
    [Order::Ascending, Order::Descending].into_iter().any(|order| {
        let expect = run(&hand, g, &input, call, order);
        let input = &input;
        let got = std::panic::catch_unwind(|| run(mutant, g, input, call, order));
        match got {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// `ptx` with the `n`th line equal to `line` (trimmed) removed.
fn without_nth(ptx: &str, line: &str, n: usize) -> String {
    let mut seen = 0;
    let mut out = String::new();
    for l in ptx.lines() {
        if l.trim() == line {
            seen += 1;
            if seen == n + 1 {
                continue;
            }
        }
        out.push_str(l);
        out.push('\n');
    }
    assert!(seen > n, "fewer than {} `{line}` lines", n + 1);
    out
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    // The baseline for every mutation below: without it, `caught` could
    // be reporting a difference the mutation did not cause.
    let (g, _) = mutation_case();
    assert!(!caught(&kir_ptx(g)));
}

#[test]
fn deleting_any_one_barrier_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g);
    let barriers = ptx.lines().filter(|l| l.trim() == "bar.sync 0;").count();
    assert_eq!(barriers, 5, "q load, scores, softmax, tile end, l publish");
    for n in 0..barriers {
        assert!(caught(&without_nth(&ptx, "bar.sync 0;", n)), "barrier {n} deleted went unnoticed");
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g);
    let cfg = g.cfg();
    let s = nsl_codegen::cfie_decode_attention::kv_strides(&cfg);
    let scale = format!("0f{:08X}", (1.0f32 / (g.head_dim as f32).sqrt()).to_bits());
    let nudged_scale = format!("0f{:08X}", (1.0f32 / (g.head_dim as f32).sqrt()).to_bits() + 1);
    for (from, to) in [
        (format!(", {};", s.token_stride), format!(", {};", s.token_stride + 1)),
        (format!(", {};", s.kv_half_stride), format!(", {};", s.kv_half_stride - 1)),
        (format!(", {};", s.layer_stride), format!(", {};", s.layer_stride + 1)),
        (format!(", {};", cfg.per_slot_max_tokens), format!(", {};", cfg.per_slot_max_tokens - 1)),
        (format!(", {scale};"), format!(", {nudged_scale};")),
    ] {
        let hits = ptx.lines().filter(|l| l.trim_start().starts_with("mov.") && l.ends_with(&from)).count();
        assert_eq!(hits, 1, "`{from}` should be materialised once");
        let mutant: String = ptx
            .lines()
            .map(|l| if l.trim_start().starts_with("mov.") && l.ends_with(&from) { l.replace(&from, &to) } else { l.to_string() })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(caught(&mutant), "`{from}` -> `{to}` went unnoticed");
    }
}

#[test]
fn dropping_the_tail_tile_clamp_is_caught() {
    // `tcnt = min(seq_len - tile, TILE)` -> `tcnt = TILE`: the ragged last
    // tile's softmax and P*V then run over scores no thread wrote.
    //
    // The other tail guard, pass 1's `tok < seq_len`, is deliberately not
    // mutated: loosening or deleting it is an *equivalent* mutant. The
    // extra scores it lets through land at `scores[tcnt..]`, which every
    // later loop bounds away, and the extra K rows it reads stay inside
    // the pool (a K row past the plane's last token is the V plane's
    // first). It is a guard against wasted work, not a correctness one,
    // and no execution can tell it apart — so a test claiming to would be
    // vacuous. The module's own structural test pins that it is present.
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g);
    let clamps: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("min.u32 ")).collect();
    assert_eq!(clamps.len(), 1, "one tail clamp");
    let (dst, rest) = clamps[0].trim().trim_start_matches("min.u32 ").split_once(',').unwrap();
    let tile = rest.split(',').nth(1).unwrap().trim().trim_end_matches(';');
    let mutant = ptx.replacen(clamps[0], &format!("    mov.u32 {dst}, {tile};"), 1);
    assert!(caught(&mutant), "the tail clamp removed went unnoticed");
}

#[test]
fn the_interpreter_refuses_an_instruction_it_does_not_model() {
    let (g, _) = mutation_case();
    let ptx = kir_ptx(g).replacen("bar.sync 0;", "membar.cta;", 1);
    let err = std::panic::catch_unwind(|| parse(&ptx)).expect_err("an unknown mnemonic must not parse");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("does not know this instruction"), "{msg}");
}

#[test]
fn the_interpreter_faults_on_an_undefined_register() {
    let (g, call) = mutation_case();
    let input = inputs(g, 3);
    // Drop the load of `seq_len`: its first reader must fault, not see 0.
    let ptx: String = kir_ptx(g)
        .lines()
        .filter(|l| !l.contains("[param_seq_len]"))
        .map(|l| format!("{l}\n"))
        .collect();
    let err = std::panic::catch_unwind(|| run(&ptx, g, &input, call, Order::Ascending))
        .expect_err("an undefined register must fault");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("before anything wrote it"), "{msg}");
}
