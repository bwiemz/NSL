//! A cooperative-CTA PTX interpreter for the flash-decode subset — the
//! executable half of the CFIE equivalence gates (roadmap A2 step 9).
//!
//! Shared by `cfie_decode_attn_kir_equivalence.rs`,
//! `cfie_kv_quant_kir_equivalence.rs`,
//! `cfie_spec_sampler_kir_equivalence.rs`,
//! `cfie_speculative_kir_equivalence.rs` and the other CFIE gates, which
//! include it with `#[path]`;
//! the first of them documents what it models and what it does not.
//! In short: a CTA runs cooperatively, a `bar.sync` releases only when
//! every thread waits at it, and an unknown mnemonic or operand form, a
//! read of an unwritten register, or an access outside the buffers a
//! launch provides is a hard error, never a skip.
//!
//! `ld.global.s8` sign-extends one byte into its register, as the
//! hardware does, and `cvt.rn.f32.s8` converts the low byte — exact, so
//! the rounding mode is moot. `cvt.u64.u64` is a copy: it is how the KIR
//! printer reinterprets one pointer type as another.
//!
//! The approximate instructions are modelled by their exact
//! counterparts, `ex2.approx.f32` as `exp2`, `rsqrt.approx.f32` as
//! `1 / sqrt`, and `sin.approx.f32` / `cos.approx.f32` as `sin` / `cos`:
//! the gates compare two programs on the same model, so what matters is
//! that the model is a function of its input, not that it rounds as the
//! hardware's approximation does. `sqrt.rn.f32` is IEEE and so is Rust's.
//! `cvt.rn.f16.f32` and `cvt.rn.bf16.f32` round to nearest even, as
//! `half` does, and `st.b16` stores the register's low two bytes.
//! `lg2.approx.f32` is modelled as `log2`, `rcp.approx.f32` as `1 / x`.
//!
//! One `.extern .shared` block (dynamic shared memory) is modelled: it sits
//! after the static blocks, and its size is what the launch allocates past
//! them (`Program::shared_bytes` counts the static blocks only).
//! `.s64` values are two's-complement bit patterns in a register, and a
//! negative decimal immediate is read as one; `add`, `sub` and `mul.lo` on
//! them are the unsigned operations, which give the same bits.
//! `%ctaid.y` and `%nctaid.y` are modelled for a two-dimensional grid.
//! `ld.{global,shared}.v4.f32` and `st.{global,shared}.v4.f32` move four
//! consecutive f32s through a braced register list, and a label may share
//! its line with the instruction it names (`DONE: ret;`).
//!
//! `red.{global,shared}.add.f32` is a read-modify-write in f32, atomic
//! because threads run one at a time; the order in which threads (and
//! CTAs) add is the schedule's, so two kernels that add the same values in
//! the same per-thread order agree bit for bit under the same schedule.
//!
//! Shifts follow PTX: `shl` and the unsigned `shr` take their amount from
//! the low 32 bits of the operand and produce 0 for an amount at or past
//! the width, rather than wrapping it as Rust's `<<` would.
//! `cvt.rn.f32.u32` and `cvt.rn.f32.u64` convert with round-to-nearest
//! (Rust's `as`), and `cvt.u32.u64` keeps the low 32 bits.
//!
//! A `.shared` block may be declared by element (`.shared .f32 NAME[N]`,
//! `N` elements) as well as in bytes, and an address may name it directly
//! (`[NAME]`). `div.approx.f32` is modelled as `a / b`, the same as
//! `div.rn.f32`, like every approximate form. `neg.f32` and `abs.f32` flip
//! and clear the sign bit (NaN included); `min.f32`, like `max.f32`, returns
//! the non-NaN operand.
//!
//! `shfl.sync.bfly.b32` is warp-synchronous: a lane that reaches it waits
//! until every lane of its 32-thread warp has reached the same instruction,
//! then all of them exchange at once (lane `l` reads lane `l ^ b`, clamped
//! by `c` as PTX specifies; a lane past the clamp keeps its own value). The
//! member mask must be the full warp and every lane of the warp must be
//! live — a divergent or partial shuffle is a hard error, not a guess.
//! `and.pred` / `or.pred` combine 0/1 predicates, `setp.nan.f32` is true if
//! either operand is NaN, `.u16` loads zero-extend two bytes and `.u16`
//! compares look at the low 16 bits. `cvta.shared.u64` of a shared symbol is
//! its window address: the model has one address space for shared memory.

use std::collections::HashMap;

use half::{bf16, f16};

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

/// Where shared memory starts in the interpreter's shared window. Non-zero
/// so a null shared pointer faults; small so the hand kernel's 32-bit
/// shared addresses hold it.
pub(crate) const SHARED_BASE: u64 = 0x100;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Special {
    TidX,
    /// `%laneid`: `%tid.x % 32` for the one-dimensional blocks modelled.
    LaneId,
    CtaidX,
    CtaidY,
    NctaidY,
    NtidX,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Src {
    Reg(usize),
    Imm(u64),
    Special(Special),
}

/// `[reg]`, `[reg+imm]`, or `[symbol]` (a shared block's address).
#[derive(Debug, Clone, Copy)]
pub(crate) struct Addr {
    pub(crate) base: Src,
    pub(crate) offset: u64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum W {
    U32,
    U64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum IntOp {
    Add,
    Sub,
    MulLo,
    Div,
    Rem,
    Min,
    Max,
    And,
    Or,
    Xor,
    Shl,
    Shr,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum FOp {
    Add,
    Sub,
    Mul,
    Max,
    Min,
    DivRn,
    /// `div.approx.f32`, modelled as `a / b` like every approximate form.
    DivApprox,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    /// `setp.nan.f32`: true if either operand is NaN.
    Nan,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CmpTy {
    U16,
    U32,
    U64,
    S64,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Space {
    Global,
    Shared,
}

#[derive(Debug, Clone)]
pub(crate) enum Op {
    LdParam { d: usize, param: String },
    Mov { d: usize, s: Src, w: W },
    Int { op: IntOp, w: W, d: usize, a: Src, b: Src },
    MulWide { d: usize, a: Src, b: Src },
    CvtU64U32 { d: usize, a: Src },
    /// `cvt.u32.u64`: the low 32 bits.
    CvtU32U64 { d: usize, a: Src },
    /// `cvt.u16.u32` / `cvt.u32.u16`: the low 16 bits, zero-extended.
    CvtU16 { d: usize, a: Src },
    /// `cvt.rn.f32.u32`: round to nearest.
    CvtF32U32 { d: usize, a: Src },
    /// `cvt.rn.f32.u64`: round to nearest.
    CvtF32U64 { d: usize, a: Src },
    CvtF32F16 { d: usize, a: Src },
    /// `cvt.rn.f32.s8`: the low byte, as a signed integer, to f32 (exact).
    CvtF32S8 { d: usize, a: Src },
    /// `cvt.u32.s8`: the low byte, sign-extended to 32 bits.
    CvtU32S8 { d: usize, a: Src },
    Setp { cmp: Cmp, ty: CmpTy, d: usize, a: Src, b: Src },
    /// `selp.<type>`: `d = p ? a : b`, at the type's width.
    Selp { w: W, d: usize, a: Src, b: Src, p: Src },
    F { op: FOp, d: usize, a: Src, b: Src },
    Fma { d: usize, a: Src, b: Src, c: Src },
    Ex2 { d: usize, a: Src },
    /// `neg.f32` / `abs.f32`: the sign bit flipped / cleared, NaN included.
    Neg { d: usize, a: Src },
    Abs { d: usize, a: Src },
    Rsqrt { d: usize, a: Src },
    Sqrt { d: usize, a: Src },
    Sin { d: usize, a: Src },
    Cos { d: usize, a: Src },
    /// `cvt.rn.f16.f32`: round to nearest even, into the low 16 bits.
    CvtF16F32 { d: usize, a: Src },
    CvtF32Bf16 { d: usize, a: Src },
    /// `cvt.rn.bf16.f32`: round to nearest even, into the low 16 bits.
    CvtBf16F32 { d: usize, a: Src },
    Lg2 { d: usize, a: Src },
    /// `rcp.approx.f32`, modelled as `1 / a`.
    Rcp { d: usize, a: Src },
    Ld { space: Space, bytes: usize, d: usize, addr: Addr },
    /// `ld.global.s8`: one byte, sign-extended into the register.
    LdS8 { space: Space, d: usize, addr: Addr },
    St { space: Space, bytes: usize, addr: Addr, v: Src },
    /// `ld.<space>.v4.f32 {d0, d1, d2, d3}, [addr]`.
    LdV4 { space: Space, d: [usize; 4], addr: Addr },
    /// `st.<space>.v4.f32 [addr], {v0, v1, v2, v3}`.
    StV4 { space: Space, addr: Addr, v: [Src; 4] },
    /// `red.<space>.add.f32`: `*addr += v` in f32, as one step of the thread.
    RedAddF32 { space: Space, addr: Addr, v: Src },
    /// `shfl.sync.bfly.b32 d, a, b, c, members`: warp-synchronous, resolved
    /// by `run_cta` once every lane of the warp waits at it.
    ShflBfly { d: usize, a: Src, b: Src, c: Src, members: Src },
    Bra { target: usize },
    Bar,
    Ret,
}

#[derive(Debug, Clone)]
pub(crate) struct Instr {
    /// `@%p` / `@!%p`: (predicate register, negated).
    pub(crate) guard: Option<(usize, bool)>,
    pub(crate) op: Op,
    pub(crate) text: String,
}

#[derive(Debug)]
pub(crate) struct Program {
    /// `(ptx type, name)` per entry parameter, in declaration order, the
    /// `param_` prefix the KIR printer adds stripped.
    pub(crate) params: Vec<(String, String)>,
    pub(crate) instrs: Vec<Instr>,
    pub(crate) reg_names: Vec<String>,
    /// Shared symbol name -> (window address, bytes).
    pub(crate) shared: HashMap<String, (u64, usize)>,
    /// Bytes of the static `.shared` blocks; a dynamic block starts here.
    pub(crate) shared_bytes: usize,
    /// The `.extern .shared` block's declared size, if the module has one
    /// (0 for `name[]`). The launch decides its real size.
    pub(crate) dynamic_shared: Option<usize>,
}

pub(crate) struct Parser {
    pub(crate) regs: HashMap<String, usize>,
    pub(crate) reg_names: Vec<String>,
    pub(crate) shared: HashMap<String, (u64, usize)>,
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
            "%laneid" => return Src::Special(Special::LaneId),
            "%ctaid.x" => return Src::Special(Special::CtaidX),
            "%ctaid.y" => return Src::Special(Special::CtaidY),
            "%nctaid.y" => return Src::Special(Special::NctaidY),
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
        if let Ok(v) = tok.parse::<i64>() {
            return Src::Imm(v as u64);
        }
        if let Some(hex) = tok.strip_prefix("0x") {
            return Src::Imm(u64::from_str_radix(hex, 16).expect("hex immediate"));
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
        let base = base.trim();
        let base = match self.shared.get(base) {
            Some(&(addr, _)) => Src::Imm(addr),
            None => Src::Reg(self.reg(base)),
        };
        Addr { base, offset }
    }
}

pub(crate) fn parse_signature(ptx: &str) -> Vec<(String, String)> {
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
pub(crate) fn parse(ptx: &str) -> Program {
    let mut p = Parser { regs: HashMap::new(), reg_names: Vec::new(), shared: HashMap::new() };
    let mut shared_bytes = 0usize;

    // `.shared .align A .b8 NAME[N];`, then at most one
    // `.extern .shared .align A .b8 NAME[N?];` after them.
    let mut dynamic: Option<(String, usize)> = None;
    for line in ptx.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix(".shared ") {
            let words: Vec<&str> = rest.split_whitespace().collect();
            let decl = *words.last().expect("a shared symbol");
            let (name, n) = decl.trim_end_matches(';').trim_end_matches(']').split_once('[').expect("a sized block");
            // `.b8 NAME[bytes]`, or an element-typed `.f32 NAME[count]`.
            let elem = match words.len().checked_sub(2).map(|i| words[i]) {
                Some(".b8" | ".u8" | ".s8") | None => 1,
                Some(".b16" | ".u16" | ".f16") => 2,
                Some(".f32" | ".u32" | ".b32" | ".s32") => 4,
                Some(".f64" | ".u64" | ".b64" | ".s64") => 8,
                Some(other) => panic!("`{line}`: shared element type {other} is not modelled"),
            };
            let n: usize = n.parse::<usize>().expect("a static shared size") * elem;
            p.shared.insert(name.to_string(), (SHARED_BASE + shared_bytes as u64, n));
            shared_bytes += n;
        } else if let Some(rest) = line.strip_prefix(".extern .shared ") {
            assert!(dynamic.is_none(), "one dynamic shared block is modelled");
            let decl = rest.split_whitespace().last().expect("a shared symbol");
            let (name, n) = decl.trim_end_matches(';').trim_end_matches(']').split_once('[').expect("a block");
            dynamic = Some((name.to_string(), if n.is_empty() { 0 } else { n.parse().expect("a shared size") }));
        } else {
            assert!(!line.starts_with(".extern"), "`{line}`: only `.extern .shared` is modelled");
        }
    }
    if let Some((name, n)) = &dynamic {
        p.shared.insert(name.clone(), (SHARED_BASE + shared_bytes as u64, *n));
    }

    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut pending: Vec<(String, String, Option<(usize, bool)>)> = Vec::new(); // (mnemonic, operands, guard)

    for raw in ptx.lines() {
        let line = raw.split("//").next().unwrap().trim();
        if line.is_empty() || line.starts_with('.') || matches!(line, "{" | "}" | ")" | "(" | ") {") {
            continue;
        }
        if let Some(name) = line.strip_suffix(':') {
            assert!(!name.contains(' '), "`{line}` is not a label");
            labels.insert(name.to_string(), pending.len());
            continue;
        }
        // `LABEL: instr;` names the instruction on its own line.
        let line = match line.split_once(':') {
            Some((name, rest)) if !name.contains(' ') && !name.contains('[') => {
                labels.insert(name.to_string(), pending.len());
                rest.trim()
            }
            _ => line,
        };
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
        let text = format!("{mnemonic} {operands}");
        // A braced register list (`{%f1, %f2, %f3, %f4}`) is one operand.
        let (operands, group): (String, Vec<&str>) = match operands.split_once('{') {
            Some((pre, rest)) => {
                let (inner, post) = rest.split_once('}').unwrap_or_else(|| panic!("`{text}`: unclosed `{{`"));
                (format!("{pre}{{}}{post}"), inner.split(',').map(str::trim).collect())
            }
            None => (operands.clone(), vec![]),
        };
        let ops: Vec<&str> = if operands.is_empty() {
            vec![]
        } else {
            operands.split(',').map(str::trim).collect()
        };
        let want = |n: usize| assert_eq!(ops.len(), n, "`{text}` takes {n} operands");
        let parts: Vec<&str> = mnemonic.split('.').collect();
        let op = match parts.as_slice() {
            ["ld", "param", "u64" | "u32" | "f32"] => {
                want(2);
                let name = ops[1].trim_start_matches('[').trim_end_matches(']');
                Op::LdParam { d: p.dst(ops[0]), param: name.trim_start_matches("param_").to_string() }
            }
            ["mov", ty] => {
                want(2);
                let w = match *ty {
                    // A 16-bit value lives in the low half of the register.
                    "u32" | "b32" | "f32" | "pred" | "b16" | "u16" => W::U32,
                    "u64" | "b64" | "s64" => W::U64,
                    _ => panic!("`{text}`: mov.{ty} is not modelled"),
                };
                Op::Mov { d: p.dst(ops[0]), s: p.src(ops[1]), w }
            }
            [name @ ("add" | "sub" | "div" | "rem" | "min" | "max"), ty @ ("u32" | "u64")]
            | [name @ "mul", "lo", ty @ ("u32" | "u64")]
            | [name @ ("add" | "sub"), ty @ "s64"]
            | [name @ "mul", "lo", ty @ "s64"] => {
                want(3);
                let op = match *name {
                    "add" => IntOp::Add,
                    "sub" => IntOp::Sub,
                    "mul" => IntOp::MulLo,
                    "div" => IntOp::Div,
                    "rem" => IntOp::Rem,
                    "max" => IntOp::Max,
                    _ => IntOp::Min,
                };
                let w = if *ty == "u32" { W::U32 } else { W::U64 };
                Op::Int { op, w, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            ["mul", "wide", "u32"] => {
                want(3);
                Op::MulWide { d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            // A shared symbol's generic address is its window address.
            ["cvta", "shared", "u64"] => {
                want(2);
                Op::Mov { d: p.dst(ops[0]), s: p.src(ops[1]), w: W::U64 }
            }
            // A pointer reinterpreted as another pointer type: a copy.
            ["cvt", "u64", "u64"] => {
                want(2);
                Op::Mov { d: p.dst(ops[0]), s: p.src(ops[1]), w: W::U64 }
            }
            ["cvt", "rn", "f32", "s8"] => {
                want(2);
                Op::CvtF32S8 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "u32", "s8"] => {
                want(2);
                Op::CvtU32S8 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            // u32 -> 64 bits zero-extends whatever the destination's sign.
            ["cvt", "u64" | "s64", "u32"] => {
                want(2);
                Op::CvtU64U32 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "u32", "u64"] => {
                want(2);
                Op::CvtU32U64 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            // Narrowing to / widening from 16 bits: the low half, zero-extended.
            ["cvt", "u16", "u32"] | ["cvt", "u32", "u16"] => {
                want(2);
                Op::CvtU16 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "rn", "f32", "u32"] => {
                want(2);
                Op::CvtF32U32 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "rn", "f32", "u64"] => {
                want(2);
                Op::CvtF32U64 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            [name @ ("and" | "or"), "pred"] => {
                want(3);
                let op = if *name == "and" { IntOp::And } else { IntOp::Or };
                Op::Int { op, w: W::U32, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            [name @ ("and" | "or" | "xor" | "shl"), ty @ ("b32" | "b64")] | [name @ "shr", ty @ ("b32" | "b64" | "u32" | "u64")] => {
                want(3);
                let op = match *name {
                    "and" => IntOp::And,
                    "or" => IntOp::Or,
                    "xor" => IntOp::Xor,
                    "shl" => IntOp::Shl,
                    _ => IntOp::Shr,
                };
                let w = if ty.ends_with("32") { W::U32 } else { W::U64 };
                Op::Int { op, w, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            ["cvt", "f32", "f16"] => {
                want(2);
                Op::CvtF32F16 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "rn", "f16", "f32"] => {
                want(2);
                Op::CvtF16F32 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "f32", "bf16"] => {
                want(2);
                Op::CvtF32Bf16 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cvt", "rn", "bf16", "f32"] => {
                want(2);
                Op::CvtBf16F32 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["rcp", "approx", "f32"] => {
                want(2);
                Op::Rcp { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["lg2", "approx", "f32"] => {
                want(2);
                Op::Lg2 { d: p.dst(ops[0]), a: p.src(ops[1]) }
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
                    "nan" => Cmp::Nan,
                    _ => panic!("`{text}`: comparison not modelled"),
                };
                let ty = match *ty {
                    "u16" => CmpTy::U16,
                    "u32" => CmpTy::U32,
                    "u64" => CmpTy::U64,
                    "s64" => CmpTy::S64,
                    "f32" => CmpTy::F32,
                    _ => panic!("`{text}`: comparison type not modelled"),
                };
                assert!(cmp != Cmp::Nan || ty == CmpTy::F32, "`{text}`: `nan` is a float comparison");
                Op::Setp { cmp, ty, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            ["selp", ty @ ("f32" | "b32" | "u32" | "b64" | "u64")] => {
                want(4);
                let w = if ty.ends_with("64") { W::U64 } else { W::U32 };
                Op::Selp { w, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]), p: p.src(ops[3]) }
            }
            [name @ ("add" | "sub" | "mul" | "max" | "min"), "f32"] | [name @ "div", "rn", "f32"] | [name @ "div", "approx", "f32"] => {
                want(3);
                let op = match (*name, parts[1]) {
                    ("add", _) => FOp::Add,
                    ("sub", _) => FOp::Sub,
                    ("mul", _) => FOp::Mul,
                    ("max", _) => FOp::Max,
                    ("min", _) => FOp::Min,
                    (_, "approx") => FOp::DivApprox,
                    _ => FOp::DivRn,
                };
                Op::F { op, d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]) }
            }
            ["fma", "rn", "f32"] => {
                want(4);
                Op::Fma { d: p.dst(ops[0]), a: p.src(ops[1]), b: p.src(ops[2]), c: p.src(ops[3]) }
            }
            ["neg", "f32"] => {
                want(2);
                Op::Neg { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["abs", "f32"] => {
                want(2);
                Op::Abs { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["ex2", "approx", "f32"] => {
                want(2);
                Op::Ex2 { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["rsqrt", "approx", "f32"] => {
                want(2);
                Op::Rsqrt { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["sqrt", "rn", "f32"] => {
                want(2);
                Op::Sqrt { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["sin", "approx", "f32"] => {
                want(2);
                Op::Sin { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["cos", "approx", "f32"] => {
                want(2);
                Op::Cos { d: p.dst(ops[0]), a: p.src(ops[1]) }
            }
            ["ld", space @ ("global" | "shared"), ty @ ("f32" | "b16" | "u16" | "u32" | "b32" | "s64" | "u64")] => {
                want(2);
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                let bytes = match *ty {
                    "b16" | "u16" => 2,
                    "s64" | "u64" => 8,
                    _ => 4,
                };
                Op::Ld { space, bytes, d: p.dst(ops[0]), addr: p.addr(ops[1]) }
            }
            // One byte, zero-extended into the register.
            ["ld", space @ ("global" | "shared"), "u8"] => {
                want(2);
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                Op::Ld { space, bytes: 1, d: p.dst(ops[0]), addr: p.addr(ops[1]) }
            }
            ["ld", space @ ("global" | "shared"), "s8"] => {
                want(2);
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                Op::LdS8 { space, d: p.dst(ops[0]), addr: p.addr(ops[1]) }
            }
            ["st", space @ ("global" | "shared"), ty @ ("f32" | "u32" | "b32" | "b16" | "u16")] => {
                want(2);
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                let bytes = if matches!(*ty, "b16" | "u16") { 2 } else { 4 };
                Op::St { space, bytes, addr: p.addr(ops[0]), v: p.src(ops[1]) }
            }
            ["ld", space @ ("global" | "shared"), "v4", "f32"] => {
                want(2);
                assert!(ops[0] == "{}" && group.len() == 4, "`{text}`: four destinations");
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                let d = [p.dst(group[0]), p.dst(group[1]), p.dst(group[2]), p.dst(group[3])];
                Op::LdV4 { space, d, addr: p.addr(ops[1]) }
            }
            ["st", space @ ("global" | "shared"), "v4", "f32"] => {
                want(2);
                assert!(ops[1] == "{}" && group.len() == 4, "`{text}`: four sources");
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                let v = [p.src(group[0]), p.src(group[1]), p.src(group[2]), p.src(group[3])];
                Op::StV4 { space, addr: p.addr(ops[0]), v }
            }
            ["red", space @ ("global" | "shared"), "add", "f32"] => {
                want(2);
                let space = if *space == "global" { Space::Global } else { Space::Shared };
                Op::RedAddF32 { space, addr: p.addr(ops[0]), v: p.src(ops[1]) }
            }
            ["shfl", "sync", "bfly", "b32"] => {
                want(5);
                Op::ShflBfly {
                    d: p.dst(ops[0]),
                    a: p.src(ops[1]),
                    b: p.src(ops[2]),
                    c: p.src(ops[3]),
                    members: p.src(ops[4]),
                }
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
        dynamic_shared: dynamic.map(|(_, n)| n),
    }
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// A global buffer the launch provides. Every global access must fall
/// entirely inside one.
pub(crate) struct Segment {
    pub(crate) base: u64,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Order {
    Ascending,
    Descending,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum State {
    Running,
    AtBarrier,
    /// Waiting at the `shfl.sync` at instruction `at` for the rest of its warp.
    AtShfl { at: usize },
    Exited,
}

/// A lane's half of a pending `shfl.sync.bfly`.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ShflPending {
    pub(crate) d: usize,
    pub(crate) value: u64,
    pub(crate) b: u64,
    pub(crate) c: u64,
    pub(crate) members: u64,
}

pub(crate) struct Thread {
    pub(crate) pc: usize,
    pub(crate) regs: Vec<u64>,
    pub(crate) written: Vec<bool>,
    pub(crate) state: State,
    pub(crate) shfl: ShflPending,
}

pub(crate) struct Launch<'a> {
    pub(crate) prog: &'a Program,
    pub(crate) args: &'a HashMap<String, u64>,
    pub(crate) global: &'a mut [Segment],
    pub(crate) shared: Vec<u8>,
    pub(crate) ctaid: u32,
    /// `%ctaid.y`: 0 for a one-dimensional grid.
    pub(crate) ctaid_y: u32,
    /// `%nctaid.y`: 1 for a one-dimensional grid.
    pub(crate) nctaid_y: u32,
    pub(crate) ntid: u32,
    pub(crate) steps: u64,
}

pub(crate) const STEP_LIMIT: u64 = 200_000_000;

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

pub(crate) fn read(prog: &Program, t: &Thread, launch: &Launch, tid: u32, s: Src, at: &str) -> u64 {
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
        Src::Special(Special::LaneId) => (tid % 32) as u64,
        Src::Special(Special::CtaidX) => launch.ctaid as u64,
        Src::Special(Special::CtaidY) => launch.ctaid_y as u64,
        Src::Special(Special::NctaidY) => launch.nctaid_y as u64,
        Src::Special(Special::NtidX) => launch.ntid as u64,
    }
}

pub(crate) fn f(bits: u64) -> f32 {
    f32::from_bits(bits as u32)
}

pub(crate) fn fb(v: f32) -> u64 {
    v.to_bits() as u64
}

/// Run thread `tid` until it waits at a barrier or exits.
pub(crate) fn run_until_blocked(t: &mut Thread, launch: &mut Launch, tid: u32) {
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
                            IntOp::Rem => a.checked_rem(b).expect("u32 remainder by zero"),
                            IntOp::Min => a.min(b),
                            IntOp::Max => a.max(b),
                            IntOp::And => a & b,
                            IntOp::Or => a | b,
                            IntOp::Xor => a ^ b,
                            IntOp::Shl => a.checked_shl(b).unwrap_or(0),
                            IntOp::Shr => a.checked_shr(b).unwrap_or(0),
                        }) as u64
                    }
                    W::U64 => match op {
                        IntOp::Add => a.wrapping_add(b),
                        IntOp::Sub => a.wrapping_sub(b),
                        IntOp::MulLo => a.wrapping_mul(b),
                        IntOp::Div => a.checked_div(b).expect("u64 division by zero"),
                        IntOp::Rem => a.checked_rem(b).expect("u64 remainder by zero"),
                        IntOp::Min => a.min(b),
                        IntOp::Max => a.max(b),
                        IntOp::And => a & b,
                        IntOp::Or => a | b,
                        IntOp::Xor => a ^ b,
                        // The amount is a u32 operand, even for a 64-bit shift.
                        IntOp::Shl => a.checked_shl(b as u32).unwrap_or(0),
                        IntOp::Shr => a.checked_shr(b as u32).unwrap_or(0),
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
            Op::CvtU32U64 { d, a } => {
                let v = rd(t, launch, *a) as u32 as u64;
                write(t, *d, v);
            }
            Op::CvtU16 { d, a } => {
                let v = rd(t, launch, *a) as u16 as u64;
                write(t, *d, v);
            }
            Op::CvtF32U32 { d, a } => {
                let v = rd(t, launch, *a) as u32 as f32;
                write(t, *d, fb(v));
            }
            Op::CvtF32U64 { d, a } => {
                let v = rd(t, launch, *a) as f32;
                write(t, *d, fb(v));
            }
            Op::CvtF32F16 { d, a } => {
                let v = f16::from_bits(rd(t, launch, *a) as u16).to_f32();
                write(t, *d, fb(v));
            }
            Op::CvtU32S8 { d, a } => {
                let v = rd(t, launch, *a) as u8 as i8 as i32 as u32 as u64;
                write(t, *d, v);
            }
            Op::CvtF32S8 { d, a } => {
                let v = rd(t, launch, *a) as u8 as i8 as f32;
                write(t, *d, fb(v));
            }
            Op::Setp { cmp, ty, d, a, b } => {
                let (a, b) = (rd(t, launch, *a), rd(t, launch, *b));
                let r = match ty {
                    CmpTy::S64 => {
                        let (a, b) = (a as i64, b as i64);
                        match cmp {
                            Cmp::Eq => a == b,
                            Cmp::Ne => a != b,
                            Cmp::Lt => a < b,
                            Cmp::Le => a <= b,
                            Cmp::Gt => a > b,
                            Cmp::Ge => a >= b,
                            Cmp::Nan => unreachable!("rejected at parse"),
                        }
                    }
                    CmpTy::U16 | CmpTy::U32 | CmpTy::U64 => {
                        let (a, b) = match ty {
                            CmpTy::U16 => (a as u16 as u64, b as u16 as u64),
                            CmpTy::U32 => (a as u32 as u64, b as u32 as u64),
                            _ => (a, b),
                        };
                        match cmp {
                            Cmp::Eq => a == b,
                            Cmp::Ne => a != b,
                            Cmp::Lt => a < b,
                            Cmp::Le => a <= b,
                            Cmp::Gt => a > b,
                            Cmp::Ge => a >= b,
                            Cmp::Nan => unreachable!("rejected at parse"),
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
                            Cmp::Nan => a.is_nan() || b.is_nan(),
                        }
                    }
                };
                write(t, *d, r as u64);
            }
            Op::Selp { w, d, a, b, p } => {
                let v = if rd(t, launch, *p) != 0 { rd(t, launch, *a) } else { rd(t, launch, *b) };
                write(t, *d, if *w == W::U32 { v as u32 as u64 } else { v });
            }
            Op::F { op, d, a, b } => {
                let (a, b) = (f(rd(t, launch, *a)), f(rd(t, launch, *b)));
                let v = match op {
                    FOp::Add => a + b,
                    FOp::Sub => a - b,
                    FOp::Mul => a * b,
                    // PTX `max.f32` returns the non-NaN operand, as Rust's.
                    FOp::Max => a.max(b),
                    // Likewise `min.f32`.
                    FOp::Min => a.min(b),
                    FOp::DivRn | FOp::DivApprox => a / b,
                };
                write(t, *d, fb(v));
            }
            Op::Fma { d, a, b, c } => {
                let v = f(rd(t, launch, *a)).mul_add(f(rd(t, launch, *b)), f(rd(t, launch, *c)));
                write(t, *d, fb(v));
            }
            Op::Neg { d, a } => {
                let v = (rd(t, launch, *a) as u32) ^ 0x8000_0000;
                write(t, *d, v as u64);
            }
            Op::Abs { d, a } => {
                let v = (rd(t, launch, *a) as u32) & 0x7FFF_FFFF;
                write(t, *d, v as u64);
            }
            Op::Ex2 { d, a } => {
                let v = f(rd(t, launch, *a)).exp2();
                write(t, *d, fb(v));
            }
            Op::Rsqrt { d, a } => {
                let v = 1.0 / f(rd(t, launch, *a)).sqrt();
                write(t, *d, fb(v));
            }
            Op::Sqrt { d, a } => {
                let v = f(rd(t, launch, *a)).sqrt();
                write(t, *d, fb(v));
            }
            Op::Sin { d, a } => {
                let v = f(rd(t, launch, *a)).sin();
                write(t, *d, fb(v));
            }
            Op::Cos { d, a } => {
                let v = f(rd(t, launch, *a)).cos();
                write(t, *d, fb(v));
            }
            Op::CvtF16F32 { d, a } => {
                let v = f16::from_f32(f(rd(t, launch, *a))).to_bits();
                write(t, *d, v as u64);
            }
            Op::CvtF32Bf16 { d, a } => {
                let v = bf16::from_bits(rd(t, launch, *a) as u16).to_f32();
                write(t, *d, fb(v));
            }
            Op::CvtBf16F32 { d, a } => {
                let v = bf16::from_f32(f(rd(t, launch, *a))).to_bits();
                write(t, *d, v as u64);
            }
            Op::Lg2 { d, a } => {
                let v = f(rd(t, launch, *a)).log2();
                write(t, *d, fb(v));
            }
            Op::Rcp { d, a } => {
                let v = 1.0 / f(rd(t, launch, *a));
                write(t, *d, fb(v));
            }
            Op::Ld { space, bytes, d, addr } => {
                let at = rd(t, launch, addr.base).wrapping_add(addr.offset);
                let mem = match space {
                    Space::Global => launch.global(at, *bytes),
                    Space::Shared => launch.shared(at, *bytes),
                };
                let mut buf = [0u8; 8];
                buf[..*bytes].copy_from_slice(mem);
                write(t, *d, u64::from_le_bytes(buf));
            }
            Op::LdS8 { space, d, addr } => {
                let at = rd(t, launch, addr.base).wrapping_add(addr.offset);
                let mem = match space {
                    Space::Global => launch.global(at, 1),
                    Space::Shared => launch.shared(at, 1),
                };
                write(t, *d, mem[0] as i8 as i64 as u64);
            }
            Op::St { space, bytes, addr, v } => {
                let at = rd(t, launch, addr.base).wrapping_add(addr.offset);
                let val = rd(t, launch, *v).to_le_bytes();
                let mem = match space {
                    Space::Global => launch.global(at, *bytes),
                    Space::Shared => launch.shared(at, *bytes),
                };
                mem.copy_from_slice(&val[..*bytes]);
            }
            Op::LdV4 { space, d, addr } => {
                let at = rd(t, launch, addr.base).wrapping_add(addr.offset);
                let mem = match space {
                    Space::Global => launch.global(at, 16),
                    Space::Shared => launch.shared(at, 16),
                };
                let lanes: Vec<u64> =
                    mem.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u64).collect();
                for (k, &dk) in d.iter().enumerate() {
                    write(t, dk, lanes[k]);
                }
            }
            Op::StV4 { space, addr, v } => {
                let at = rd(t, launch, addr.base).wrapping_add(addr.offset);
                let vals: Vec<u64> = v.iter().map(|&s| rd(t, launch, s)).collect();
                let mem = match space {
                    Space::Global => launch.global(at, 16),
                    Space::Shared => launch.shared(at, 16),
                };
                for (k, val) in vals.iter().enumerate() {
                    mem[4 * k..4 * k + 4].copy_from_slice(&(*val as u32).to_le_bytes());
                }
            }
            Op::RedAddF32 { space, addr, v } => {
                let at = rd(t, launch, addr.base).wrapping_add(addr.offset);
                let add = f(rd(t, launch, *v));
                let mem = match space {
                    Space::Global => launch.global(at, 4),
                    Space::Shared => launch.shared(at, 4),
                };
                let old = f32::from_le_bytes([mem[0], mem[1], mem[2], mem[3]]);
                mem.copy_from_slice(&(old + add).to_le_bytes());
            }
            Op::Bra { target } => t.pc = *target,
            Op::ShflBfly { d, a, b, c, members } => {
                t.shfl = ShflPending {
                    d: *d,
                    value: rd(t, launch, *a),
                    b: rd(t, launch, *b),
                    c: rd(t, launch, *c),
                    members: rd(t, launch, *members),
                };
                t.state = State::AtShfl { at: t.pc - 1 };
                return;
            }
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

/// Complete every warp whose lanes all wait at the same `shfl.sync.bfly`.
/// Returns whether any warp was released. A warp with some lanes at a
/// shuffle and others elsewhere (another shuffle, a barrier, exited) is
/// divergent, which the model does not guess at.
fn resolve_shuffles(threads: &mut [Thread], ctaid: u32) -> bool {
    let mut released = false;
    for (w, warp) in threads.chunks_mut(32).enumerate() {
        let Some(at) = warp.iter().find_map(|t| match t.state {
            State::AtShfl { at } => Some(at),
            _ => None,
        }) else {
            continue;
        };
        assert!(
            warp.len() == 32 && warp.iter().all(|t| t.state == State::AtShfl { at }),
            "CTA {ctaid} warp {w}: a shfl.sync at instruction {at} that not every lane of a full warp reached"
        );
        let values: Vec<u64> = warp.iter().map(|t| t.shfl.value).collect();
        for (lane, t) in warp.iter_mut().enumerate() {
            let ShflPending { d, b, c, members, .. } = t.shfl;
            assert_eq!(members as u32, u32::MAX, "CTA {ctaid} warp {w}: only full-warp shuffles are modelled");
            // PTX: c packs the clamp (bits 0-4) and the segment mask (bits 8-12).
            let seg_mask = ((c >> 8) & 0x1f) as usize;
            let max_lane = (lane & seg_mask) | ((c & 0x1f) as usize & !seg_mask);
            let src = lane ^ (b as usize & 0x1f);
            let v = if src > max_lane { values[lane] } else { values[src] };
            t.regs[d] = v as u32 as u64;
            t.written[d] = true;
            t.state = State::Running;
        }
        released = true;
    }
    released
}

/// Run one CTA to completion under `order`.
pub(crate) fn run_cta(launch: &mut Launch, order: Order) {
    let n = launch.ntid;
    let regs = launch.prog.reg_names.len();
    let mut threads: Vec<Thread> = (0..n)
        .map(|_| Thread {
            pc: 0,
            regs: vec![0; regs],
            written: vec![false; regs],
            state: State::Running,
            shfl: ShflPending::default(),
        })
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
        if resolve_shuffles(&mut threads, launch.ctaid) {
            continue;
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

