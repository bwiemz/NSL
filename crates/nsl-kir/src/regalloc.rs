// crates/nsl-kir/src/regalloc.rs
//! Register classes and the allocator (roadmap A2 step 5).
//!
//! `KernelIR` values are SSA `VarId`s. Until this step the PTX printer named
//! every value's register by its `VarId` — `%f17` was variable 17 whatever
//! its type — and declared *every* class at `max VarId + 1`, so a kernel with
//! N values declared four to five N registers, and its only register-pressure
//! number was that count. `ptxas` does the physical allocation, so none of
//! that was a correctness problem; it is why the roadmap asks for "typed
//! registers + a linear-scan allocator": a kernel should expose how many
//! registers of each class it holds live at once, and a migrated kernel's
//! text should be comparable with its hand-written predecessor, which numbers
//! densely.
//!
//! The allocator assigns each value a (class, index) pair: the class from
//! its type (`RegClass::of`), the index by linear scan over live intervals
//! on the block-order linearisation. Intervals come from the IR — a kernel
//! parameter is defined before block 0, a block parameter at its block's
//! entry *and* at the terminator of every predecessor (the edge copy writes
//! it there), an op's destinations at the op, and a value stays live to its
//! last use and to the end of every block it is live-out of (block-level
//! liveness over the CFG), so a value carried around a loop keeps its
//! register through the whole loop. Two values live at the same point never
//! share a register; a value whose interval ends at the point another's
//! begins does not share either (strict), so an edge's parallel copy never
//! reads a register it has just written. Indices are dense per class, and
//! the per-class count is the kernel's register pressure
//! (`KernelIR::register_pressure`).
//!
//! The printer keeps emitting `%<class><VarId>` names and renames them
//! through the [`Allocation`] as a last pass; scratch registers the printer
//! needs for its own lowerings (`%edge_*`, `%gid*`) have names, not numbers,
//! and are declared beside the classes.

use std::collections::{HashMap, HashSet};

use crate::kernel_ir::{BlockId, KernelIR, KirType, VarId};
use crate::kir_verify::{op_dsts, op_uses, terminator_uses};

/// A PTX register class: the `.reg` declaration a value lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// `.u32 %r` — 32-bit integers, the narrow integer types, untyped values.
    R,
    /// `.u64 %rd` — 64-bit integers and pointers.
    Rd,
    /// `.f32 %f`.
    F,
    /// `.f64 %fd`.
    Fd,
    /// `.b16 %h` — f16 and bf16.
    H,
    /// `.pred %p` — `Bool`.
    P,
    /// `.b32 %v` — packed fragments (`Vec`).
    V,
}

impl RegClass {
    /// Every class, in declaration order.
    pub const ALL: [RegClass; 7] = [
        RegClass::R,
        RegClass::Rd,
        RegClass::F,
        RegClass::Fd,
        RegClass::H,
        RegClass::P,
        RegClass::V,
    ];

    /// The register-name prefix (`%r`, `%rd`, …).
    pub fn prefix(self) -> &'static str {
        match self {
            RegClass::R => "%r",
            RegClass::Rd => "%rd",
            RegClass::F => "%f",
            RegClass::Fd => "%fd",
            RegClass::H => "%h",
            RegClass::P => "%p",
            RegClass::V => "%v",
        }
    }

    /// The `.reg` type the class is declared with.
    pub fn ptx_type(self) -> &'static str {
        match self {
            RegClass::R => "u32",
            RegClass::Rd => "u64",
            RegClass::F => "f32",
            RegClass::Fd => "f64",
            RegClass::H => "b16",
            RegClass::P => "pred",
            RegClass::V => "b32",
        }
    }

    /// The class a value of type `ty` lives in (`None`: untyped, a `%r`).
    pub fn of(ty: Option<&KirType>) -> RegClass {
        match ty {
            Some(KirType::Bool) => RegClass::P,
            Some(KirType::U64) | Some(KirType::I64) | Some(KirType::Ptr(_, _)) => RegClass::Rd,
            Some(KirType::F32) => RegClass::F,
            Some(KirType::F64) => RegClass::Fd,
            Some(KirType::F16) | Some(KirType::Bf16) => RegClass::H,
            Some(KirType::Vec(_, _)) => RegClass::V,
            _ => RegClass::R,
        }
    }

    /// The class whose prefix is `prefix`, if any.
    pub fn from_prefix(prefix: &str) -> Option<RegClass> {
        RegClass::ALL.iter().copied().find(|c| c.prefix() == prefix)
    }

    fn slot(self) -> usize {
        RegClass::ALL.iter().position(|c| *c == self).unwrap()
    }
}

/// How many registers of each class a kernel holds live at once — the
/// declared count per class after allocation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RegisterPressure {
    counts: [u32; 7],
}

impl RegisterPressure {
    /// The count for one class.
    pub fn of(&self, class: RegClass) -> u32 {
        self.counts[class.slot()]
    }

    /// Every class with a non-zero count, in declaration order.
    pub fn nonzero(&self) -> impl Iterator<Item = (RegClass, u32)> + '_ {
        RegClass::ALL.iter().copied().filter_map(|c| {
            let n = self.of(c);
            (n > 0).then_some((c, n))
        })
    }
}

/// The result of [`allocate`]: a (class, index) per value and the count per
/// class.
#[derive(Debug, Clone)]
pub struct Allocation {
    assign: HashMap<VarId, (RegClass, u32)>,
    pressure: RegisterPressure,
}

impl Allocation {
    /// The class `v` was placed in, if `v` appears in the kernel.
    pub fn class(&self, v: VarId) -> Option<RegClass> {
        self.assign.get(&v).map(|(c, _)| *c)
    }

    /// The index `v` was given within its class.
    pub fn index(&self, v: VarId) -> Option<u32> {
        self.assign.get(&v).map(|(_, i)| *i)
    }

    /// The register name `v` prints as (`%f3`).
    ///
    /// # Panics
    /// If `v` does not appear in the kernel.
    pub fn name(&self, v: VarId) -> String {
        let (class, index) = self.assign.get(&v).unwrap_or_else(|| panic!("v{v} was never allocated"));
        format!("{}{}", class.prefix(), index)
    }

    /// The per-class counts.
    pub fn pressure(&self) -> RegisterPressure {
        self.pressure
    }
}

/// A live interval on the linearised program: inclusive points.
#[derive(Debug, Clone, Copy)]
struct Interval {
    var: VarId,
    class: RegClass,
    start: u32,
    end: u32,
}

/// Allocate registers for every value of `ir`; see the module docs.
pub fn allocate(ir: &KernelIR) -> Allocation {
    // ── Program points ───────────────────────────────────────────────
    // Kernel parameters are defined at 0; block b spans [start[b], end[b]]
    // with its parameters at start[b], one point per op, and the
    // terminator at end[b].
    let n = ir.blocks.len();
    let mut start = vec![0u32; n];
    let mut end = vec![0u32; n];
    let mut point = 1u32;
    for (b, block) in ir.blocks.iter().enumerate() {
        start[b] = point;
        point += 1 + block.ops.len() as u32;
        end[b] = point;
        point += 1;
    }

    // ── Block-level liveness ─────────────────────────────────────────
    let mut use_before_def: Vec<HashSet<VarId>> = vec![HashSet::new(); n];
    let mut defs: Vec<HashSet<VarId>> = vec![HashSet::new(); n];
    let mut succs: Vec<Vec<BlockId>> = vec![Vec::new(); n];
    for (b, block) in ir.blocks.iter().enumerate() {
        for p in &block.params {
            defs[b].insert(p.id);
        }
        for op in &block.ops {
            for u in op_uses(op) {
                if !defs[b].contains(&u) {
                    use_before_def[b].insert(u);
                }
            }
            for d in op_dsts(op) {
                defs[b].insert(d);
            }
        }
        if let Some(term) = &block.terminator {
            for u in terminator_uses(term) {
                if !defs[b].contains(&u) {
                    use_before_def[b].insert(u);
                }
            }
            for edge in term.edges() {
                if (edge.target as usize) < n {
                    succs[b].push(edge.target);
                }
            }
        }
    }
    let mut live_in: Vec<HashSet<VarId>> = vec![HashSet::new(); n];
    let mut live_out: Vec<HashSet<VarId>> = vec![HashSet::new(); n];
    let mut changed = true;
    while changed {
        changed = false;
        for b in (0..n).rev() {
            let mut out = HashSet::new();
            for s in &succs[b] {
                out.extend(live_in[*s as usize].iter().copied());
            }
            let mut inn = use_before_def[b].clone();
            inn.extend(out.iter().filter(|v| !defs[b].contains(v)).copied());
            if out != live_out[b] || inn != live_in[b] {
                live_out[b] = out;
                live_in[b] = inn;
                changed = true;
            }
        }
    }

    // ── Intervals ────────────────────────────────────────────────────
    let class_of = |v: VarId| RegClass::of(ir.var_types.get(&v));
    let mut intervals: HashMap<VarId, Interval> = HashMap::new();
    let mut touch = |v: VarId, at: u32| {
        let iv = intervals.entry(v).or_insert(Interval { var: v, class: class_of(v), start: at, end: at });
        iv.start = iv.start.min(at);
        iv.end = iv.end.max(at);
    };
    for p in &ir.params {
        touch(p.id, 0);
    }
    for (b, block) in ir.blocks.iter().enumerate() {
        for p in &block.params {
            touch(p.id, start[b]);
        }
        let mut at = start[b];
        for op in &block.ops {
            at += 1;
            for d in op_dsts(op) {
                touch(d, at);
            }
            for u in op_uses(op) {
                touch(u, at);
            }
        }
        if let Some(term) = &block.terminator {
            for u in terminator_uses(term) {
                touch(u, end[b]);
            }
            // The edge copy defines the target's parameters here.
            for edge in term.edges() {
                if let Some(target) = ir.blocks.get(edge.target as usize) {
                    for p in &target.params {
                        touch(p.id, end[b]);
                    }
                }
            }
        }
        for v in &live_out[b] {
            touch(*v, end[b]);
        }
    }
    // A value used before any definition reaches it (invalid IR the
    // verifier reports; still printed) starts at 0 like a parameter.
    let defined: HashSet<VarId> = ir
        .params
        .iter()
        .map(|p| p.id)
        .chain(ir.blocks.iter().flat_map(|b| {
            b.params.iter().map(|p| p.id).chain(b.ops.iter().flat_map(op_dsts))
        }))
        .collect();
    for iv in intervals.values_mut() {
        if !defined.contains(&iv.var) {
            iv.start = 0;
        }
    }

    // ── Linear scan, per class ───────────────────────────────────────
    let mut sorted: Vec<Interval> = intervals.into_values().collect();
    sorted.sort_by_key(|iv| (iv.start, iv.var));
    let mut assign: HashMap<VarId, (RegClass, u32)> = HashMap::new();
    let mut counts = [0u32; 7];
    for class in RegClass::ALL {
        // (end, index) of the intervals holding a register.
        let mut active: Vec<(u32, u32)> = Vec::new();
        let mut free: Vec<u32> = Vec::new();
        let mut next = 0u32;
        for iv in sorted.iter().filter(|iv| iv.class == class) {
            // Expire what ended strictly before this interval starts.
            active.retain(|(e, idx)| {
                if *e < iv.start {
                    free.push(*idx);
                    false
                } else {
                    true
                }
            });
            free.sort_unstable_by(|a, b| b.cmp(a));
            let idx = match free.pop() {
                Some(idx) => idx,
                None => {
                    next += 1;
                    next - 1
                }
            };
            active.push((iv.end, idx));
            assign.insert(iv.var, (class, idx));
        }
        counts[class.slot()] = next;
    }
    Allocation { assign, pressure: RegisterPressure { counts } }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel_ir::{AddressSpace, CmpOp, KirBuilder, KirEdge, KirOp, KirTerminator};

    fn f32_ptr() -> KirType {
        KirType::Ptr(Box::new(KirType::F32), AddressSpace::Global)
    }

    /// `out[i] = a[i]` over a grid-stride loop: `n` is live across the
    /// whole loop, `i` is the header's parameter.
    fn grid_stride() -> (KernelIR, VarId, VarId, VarId, VarId) {
        let mut b = KirBuilder::new("g");
        let a = b.add_param("a", f32_ptr(), AddressSpace::Global);
        let out = b.add_param("out", f32_ptr(), AddressSpace::Global);
        let n = b.add_param("n", KirType::U32, AddressSpace::Local);
        let entry = b.new_block();
        let header = b.new_block();
        let body = b.new_block();
        let exit = b.new_block();
        let i = b.add_block_param(header, KirType::U32);
        b.set_block(entry);
        let s = b.new_typed_var(KirType::U32);
        b.emit(KirOp::GlobalId(s, 0));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![s])));
        b.set_block(header);
        let more = b.new_typed_var(KirType::Bool);
        b.emit(KirOp::Cmp(more, i, n, CmpOp::Lt));
        b.terminate(KirTerminator::CondBranch(more, body.into(), exit.into()));
        b.set_block(body);
        let src = b.new_typed_var(f32_ptr());
        b.emit(KirOp::PtrOffset(src, a, i));
        let v = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Load(v, src, AddressSpace::Global));
        let dst = b.new_typed_var(f32_ptr());
        b.emit(KirOp::PtrOffset(dst, out, i));
        b.emit(KirOp::Store(dst, v, AddressSpace::Global));
        let stride = b.new_typed_var(KirType::U32);
        b.emit(KirOp::BlockDim(stride, 0));
        let next = b.new_typed_var(KirType::U32);
        b.emit(KirOp::Add(next, i, stride));
        b.terminate(KirTerminator::Branch(KirEdge::with(header, vec![next])));
        b.set_block(exit);
        b.terminate(KirTerminator::Return);
        (b.finalize(), n, i, stride, next)
    }

    #[test]
    fn classes_follow_types_and_indices_are_dense() {
        let (ir, n, i, _, _) = grid_stride();
        let a = allocate(&ir);
        assert_eq!(a.class(n), Some(RegClass::R));
        assert_eq!(a.class(i), Some(RegClass::R));
        assert_eq!(a.class(0), Some(RegClass::Rd));
        // %rd: a, out, src, dst — src and dst never overlap.
        assert_eq!(a.pressure().of(RegClass::Rd), 3);
        assert_eq!(a.pressure().of(RegClass::P), 1);
        assert_eq!(a.pressure().of(RegClass::F), 1);
        assert_eq!(a.pressure().of(RegClass::Fd), 0);
        for class in RegClass::ALL {
            let mut idx: Vec<u32> = a
                .assign
                .values()
                .filter(|(c, _)| *c == class)
                .map(|(_, i)| *i)
                .collect();
            idx.sort_unstable();
            idx.dedup();
            assert_eq!(idx, (0..a.pressure().of(class)).collect::<Vec<_>>(), "{class:?}");
        }
    }

    #[test]
    fn a_value_live_around_the_loop_keeps_its_register_through_the_body() {
        // `n` is read only in the header, but it is live-out of the body
        // (the back edge returns to the header), so nothing in the body may
        // take its register.
        let (ir, n, i, stride, next) = grid_stride();
        let a = allocate(&ir);
        for v in [i, stride, next] {
            assert_ne!(a.index(v), a.index(n), "v{v} shares n's register");
        }
    }

    #[test]
    fn a_block_parameter_is_live_at_every_incoming_edge() {
        // `i` is written by the entry edge and the back edge; a value
        // defined after `i`'s last use in the body and live at the back
        // edge (`stride`, read by the add that computes `next`) must not
        // share `i`'s register.
        let (ir, _, i, stride, _) = grid_stride();
        let a = allocate(&ir);
        assert_ne!(a.index(stride), a.index(i));
    }

    #[test]
    fn dead_values_free_their_registers() {
        let mut b = KirBuilder::new("d");
        let e = b.new_block();
        b.set_block(e);
        let x = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(x, crate::kernel_ir::KirConst { ty: KirType::F32, value: crate::kernel_ir::ConstValue::F32(1.0) }));
        let y = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Neg(y, x));
        let z = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Neg(z, y));
        b.terminate(KirTerminator::Return);
        let a = allocate(&b.finalize());
        // x dies at y's definition; z may take x's register but not y's.
        assert_eq!(a.pressure().of(RegClass::F), 2);
        assert_eq!(a.index(z), a.index(x));
        assert_ne!(a.index(z), a.index(y));
    }

    #[test]
    fn names_carry_the_class_prefix() {
        let (ir, n, _, _, _) = grid_stride();
        let a = allocate(&ir);
        assert!(a.name(n).starts_with("%r"));
        assert!(a.name(0).starts_with("%rd"));
        assert_eq!(RegClass::from_prefix("%fd"), Some(RegClass::Fd));
        assert_eq!(RegClass::from_prefix("%x"), None);
    }
}
