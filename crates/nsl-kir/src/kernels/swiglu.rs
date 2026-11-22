// crates/nsl-kir/src/kernels/swiglu.rs
//! Fused SwiGLU: `p = silu(g) * u` and its adjoint, one kernel each.
//!
//! The unfused chain is two launches forward (`silu`, then the multiply,
//! with `silu(g)` written to memory and read back) and, in source AD's
//! backward, a multiply for the up branch (`p̄ * silu(g)`, reading the
//! recomputed silu output) beside the fused gate adjoint
//! (`nsl_swiglu_gate_backward_f32`). These kernels read `g` and `u` (and `p̄`)
//! once and never materialise `silu(g)`:
//!
//! ```text
//! Forward     (gate, up, out, n)               out   = silu(g)·u
//! UpBackward  (grad, gate, out, n)             out   = p̄·silu(g)
//! Backward    (grad, up, gate, dgate, dup, n)  dgate = (p̄·u)·silu'(g)
//!                                              dup   = p̄·silu(g)
//! BackwardRecompute                            as Backward, and
//!   (grad, up, gate, dgate, dup, out, n)       out   = silu(g)·u
//! ```
//!
//! `BackwardRecompute` also writes the forward value `p` for the down
//! projection's weight gradient, so a checkpointed step need not recompute
//! it with a forward launch (torch.compile's `mul_silu_silu_backward`).
//!
//! **Arithmetic.** f32, in the unfused chain's operation order: the sigmoid
//! is the hand kernels' (`ex2.approx`, `rcp.approx`; `elementwise::sigmoid_of`),
//! `silu(g) = g·σ(g)` as `nsl_silu_f32` computes it, the gate adjoint is
//! `nsl_swiglu_gate_backward_f32`'s sequence (`p̄·u` rounded first, then the
//! explicitly rounded source-AD derivative tail), and every multiply is
//! `mul.rn` (the IEEE product the unfused kernels' multiplies compute; the
//! explicit rounding only keeps ptxas from contracting anything into an
//! `fma`). So the all-f32 instantiations are bit-identical to the chains they
//! replace.
//!
//! **Storage.** Each tensor operand is f32 or bf16 independently (widened on
//! load, exact); every output is rounded once on store. With bf16 storage the
//! unfused chain rounds `silu(g)` to bf16 when it stores it; these kernels keep
//! it in a register, so their outputs carry one rounding fewer.
//! [`SwigluKernel::round_silu`] reinstates that rounding (round to bf16 and
//! widen back, when `g` is stored bf16): it exists so the fused kernels can be
//! held bit for bit to the unfused bf16 chain, not for production.
//!
//! **Width.** Each thread handles `width` consecutive elements (1, 4 or 8).
//! At 4 and 8 the operands are moved with `ld/st.global.v4` (8 bytes per
//! bf16 operand, 16 per f32 one; 8 elements are two such accesses), which
//! keeps enough bytes in flight per thread to reach the card's bandwidth: a
//! one-element bf16 thread has only 2 bytes outstanding per operand. The
//! launcher picks a width only when `n` is a multiple of it and every pointer
//! is aligned to `4 * element size`.

use super::elementwise::{ELEMENTWISE_BLOCK, f32_op2, sigmoid_of, srcad_silu_tail, verified_ptx};
use super::storage::{Storage, variant_name};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator,
    KirType, VarId,
};

/// Which kernel of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwigluOp {
    /// `out = silu(gate) * up`.
    Forward,
    /// `out = grad * silu(gate)`: the up branch's adjoint.
    UpBackward,
    /// `dgate = (grad * up) * silu'(gate)`, `dup = grad * silu(gate)`: both
    /// adjoints from one read of `grad`, `up` and `gate`.
    Backward,
    /// [`SwigluOp::Backward`] and `out = silu(gate) * up` (the forward value,
    /// recomputed), from the same reads.
    BackwardRecompute,
}

impl SwigluOp {
    pub const ALL: [SwigluOp; 4] = [
        SwigluOp::Forward,
        SwigluOp::UpBackward,
        SwigluOp::Backward,
        SwigluOp::BackwardRecompute,
    ];

    /// The all-f32, one-element entry name.
    pub fn kernel_name(self) -> &'static str {
        match self {
            SwigluOp::Forward => "nsl_swiglu_f32",
            SwigluOp::UpBackward => "nsl_swiglu_up_backward_f32",
            SwigluOp::Backward => "nsl_swiglu_backward_f32",
            SwigluOp::BackwardRecompute => "nsl_swiglu_backward_h_f32",
        }
    }

    /// The tensor inputs, in parameter order.
    pub fn inputs(self) -> &'static [&'static str] {
        match self {
            SwigluOp::Forward => &["gate", "up"],
            SwigluOp::UpBackward => &["grad", "gate"],
            SwigluOp::Backward | SwigluOp::BackwardRecompute => &["grad", "up", "gate"],
        }
    }

    /// The outputs, in parameter order (after the inputs, before `n`).
    pub fn outputs(self) -> &'static [&'static str] {
        match self {
            SwigluOp::Forward | SwigluOp::UpBackward => &["out"],
            SwigluOp::Backward => &["dgate", "dup"],
            SwigluOp::BackwardRecompute => &["dgate", "dup", "out"],
        }
    }

    /// Every parameter, in order: the inputs, the outputs, `n`.
    pub fn param_names(self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self.inputs().to_vec();
        v.extend_from_slice(self.outputs());
        v.push("n");
        v
    }
}

/// Elements per thread the launcher may choose.
pub const WIDTHS: [u32; 3] = [1, 4, 8];

/// One instantiation of the family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SwigluKernel {
    pub op: SwigluOp,
    /// The storage of each tensor input, in parameter order; the slots past
    /// `op.inputs().len()` are unused and must be `F32` (so equal kernels
    /// compare and hash equal).
    pub ins: [Storage; 3],
    /// The storage of every output.
    pub out: Storage,
    /// Elements per thread: 1, 4 or 8.
    pub width: u32,
    /// Round `silu(gate)` to bf16 before it multiplies, when `gate` is stored
    /// bf16 (see the module header). A no-op for an f32 `gate`.
    pub round_silu: bool,
}

impl SwigluKernel {
    /// `op` with `ins` (one storage per input, in parameter order).
    ///
    /// # Panics
    ///
    /// On a storage count that does not match the op, or a width outside
    /// [`WIDTHS`].
    pub fn new(op: SwigluOp, ins: &[Storage], out: Storage, width: u32, round_silu: bool) -> Self {
        assert_eq!(
            ins.len(),
            op.inputs().len(),
            "{}: one storage per tensor input",
            op.kernel_name()
        );
        assert!(
            WIDTHS.contains(&width),
            "{}: width {width} is not one of {WIDTHS:?}",
            op.kernel_name()
        );
        let mut slots = [Storage::F32; 3];
        slots[..ins.len()].copy_from_slice(ins);
        SwigluKernel {
            op,
            ins: slots,
            out,
            width,
            round_silu,
        }
    }

    fn inputs(&self) -> &[Storage] {
        &self.ins[..self.op.inputs().len()]
    }

    /// The storage of `gate`.
    fn gate(&self) -> Storage {
        let k = self
            .op
            .inputs()
            .iter()
            .position(|n| *n == "gate")
            .expect("every op reads gate");
        self.ins[k]
    }

    /// Whether the kernel rounds `silu(gate)` to bf16 before multiplying.
    pub fn rounds_silu(&self) -> bool {
        self.round_silu && self.gate() == Storage::Bf16
    }

    /// The `.visible .entry` name: [`variant_name`] of the op's base name,
    /// then `_x4`/`_x8` for a vector width and `_rs` when it rounds
    /// `silu(gate)`. The all-f32 one-element kernel keeps the base name.
    pub fn name(&self) -> String {
        let mut name = variant_name(self.op.kernel_name(), self.inputs(), self.out);
        if self.width > 1 {
            name.push_str(&format!("_x{}", self.width));
        }
        if self.rounds_silu() {
            name.push_str("_rs");
        }
        name
    }

    /// The kernel as KIR.
    pub fn build(&self) -> KernelIR {
        build(self)
    }

    /// The kernel lowered to a NUL-terminated PTX module.
    ///
    /// # Panics
    ///
    /// If the built kernel fails verification: a bug in this module.
    pub fn ptx(&self) -> Vec<u8> {
        verified_ptx(self.build())
    }
}

/// `silu(a) = a·σ(a)` from `s = σ(a)`, rounded through bf16 when `round`.
fn silu_from(b: &mut KirBuilder, a: VarId, s: VarId, round: bool) -> VarId {
    let y = f32_op2(b, KirOp::MulRn, a, s);
    if !round {
        return y;
    }
    let h = b.new_typed_var(KirType::Bf16);
    b.emit(KirOp::Cast(h, y, KirType::Bf16));
    let w = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Cast(w, h, KirType::F32));
    w
}

/// The outputs of one element from its widened inputs (parameter order).
fn lane(b: &mut KirBuilder, op: SwigluOp, round: bool, v: &[VarId]) -> Vec<VarId> {
    match op {
        SwigluOp::Forward => {
            let (a, u) = (v[0], v[1]);
            let s = sigmoid_of(b, a);
            let y = silu_from(b, a, s, round);
            vec![f32_op2(b, KirOp::MulRn, y, u)]
        }
        SwigluOp::UpBackward => {
            let (g, a) = (v[0], v[1]);
            let s = sigmoid_of(b, a);
            let y = silu_from(b, a, s, round);
            vec![f32_op2(b, KirOp::MulRn, g, y)]
        }
        SwigluOp::Backward | SwigluOp::BackwardRecompute => {
            let (g, u, a) = (v[0], v[1], v[2]);
            // The gate adjoint, exactly `nsl_swiglu_gate_backward_f32`'s.
            let t = f32_op2(b, KirOp::MulRn, g, u);
            let s = sigmoid_of(b, a);
            let tail = srcad_silu_tail(b, a, s);
            let dg = f32_op2(b, KirOp::MulRn, t, tail);
            // The up adjoint, from the same sigmoid.
            let y = silu_from(b, a, s, round);
            let du = f32_op2(b, KirOp::MulRn, g, y);
            if op == SwigluOp::Backward {
                return vec![dg, du];
            }
            // The forward value, as `Forward` computes it.
            let p = f32_op2(b, KirOp::MulRn, y, u);
            vec![dg, du, p]
        }
    }
}

/// Load `k` (1 or 4) consecutive elements `base[i..i+k]` and widen them.
fn load_group(b: &mut KirBuilder, base: VarId, i: VarId, s: Storage, k: usize) -> Vec<VarId> {
    let addr = b.new_typed_var(s.ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    let raw: Vec<VarId> = (0..k).map(|_| b.new_typed_var(s.elem())).collect();
    if k == 1 {
        b.emit(KirOp::Load(raw[0], addr, AddressSpace::Global));
    } else {
        b.emit(KirOp::LoadVec {
            dsts: raw.clone(),
            ptr: addr,
            space: AddressSpace::Global,
        });
    }
    match s {
        Storage::F32 => raw,
        Storage::Bf16 => raw
            .into_iter()
            .map(|h| {
                let v = b.new_typed_var(KirType::F32);
                b.emit(KirOp::Cast(v, h, KirType::F32));
                v
            })
            .collect(),
    }
}

/// Narrow `vals` (f32) to `s` and store them at `base[i..]`.
fn store_group(b: &mut KirBuilder, base: VarId, i: VarId, s: Storage, vals: &[VarId]) {
    let stored: Vec<VarId> = match s {
        Storage::F32 => vals.to_vec(),
        Storage::Bf16 => vals
            .iter()
            .map(|&v| {
                let h = b.new_typed_var(KirType::Bf16);
                b.emit(KirOp::Cast(h, v, KirType::Bf16));
                h
            })
            .collect(),
    };
    let addr = b.new_typed_var(s.ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    if stored.len() == 1 {
        b.emit(KirOp::Store(addr, stored[0], AddressSpace::Global));
    } else {
        b.emit(KirOp::StoreVec {
            ptr: addr,
            vals: stored,
            space: AddressSpace::Global,
        });
    }
}

fn build(k: &SwigluKernel) -> KernelIR {
    use AddressSpace::Global;
    let op = k.op;
    let mut b = KirBuilder::new(&k.name());
    let ins: Vec<VarId> = op
        .inputs()
        .iter()
        .zip(k.inputs())
        .map(|(name, s)| b.add_param(name, s.ptr(), Global))
        .collect();
    let outs: Vec<VarId> = op
        .outputs()
        .iter()
        .map(|name| b.add_param(name, k.out.ptr(), Global))
        .collect();
    let n = b.add_param("n", KirType::U64, Global);

    let entry = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let tid32 = b.new_typed_var(KirType::U32);
    b.emit(KirOp::GlobalId(tid32, 0));
    let tid = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(tid, tid32, KirType::U64));
    // The first element of this thread. The launcher passes `n` a multiple
    // of the width, so `i < n` covers the thread's whole run.
    let i = if k.width == 1 {
        tid
    } else {
        let w = b.new_typed_var(KirType::U64);
        b.emit(KirOp::Const(
            w,
            KirConst {
                ty: KirType::U64,
                value: ConstValue::U64(u64::from(k.width)),
            },
        ));
        let i = b.new_typed_var(KirType::U64);
        b.emit(KirOp::Mul(i, tid, w));
        i
    };
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, i, n, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(
        past,
        KirEdge::to(exit),
        KirEdge::to(body),
    ));

    b.set_block(body);
    // Groups of up to four elements, each moved by one (vector) access.
    let per = if k.width == 1 { 1 } else { 4 };
    let groups = (k.width as usize) / per;
    let starts: Vec<VarId> = (0..groups)
        .map(|g| {
            if g == 0 {
                i
            } else {
                let off = b.new_typed_var(KirType::U64);
                b.emit(KirOp::Const(
                    off,
                    KirConst {
                        ty: KirType::U64,
                        value: ConstValue::U64((g * per) as u64),
                    },
                ));
                let at = b.new_typed_var(KirType::U64);
                b.emit(KirOp::Add(at, i, off));
                at
            }
        })
        .collect();
    // Every load first, so all of the thread's reads are in flight together.
    let loaded: Vec<Vec<Vec<VarId>>> = starts
        .iter()
        .map(|&at| {
            ins.iter()
                .zip(k.inputs())
                .map(|(&p, &s)| load_group(&mut b, p, at, s, per))
                .collect()
        })
        .collect();
    let round = k.rounds_silu();
    for (g, &at) in starts.iter().enumerate() {
        // results[o][lane]
        let mut results: Vec<Vec<VarId>> = vec![Vec::with_capacity(per); outs.len()];
        for l in 0..per {
            let v: Vec<VarId> = loaded[g].iter().map(|vals| vals[l]).collect();
            for (o, r) in lane(&mut b, op, round, &v).into_iter().enumerate() {
                results[o].push(r);
            }
        }
        for (o, &p) in outs.iter().enumerate() {
            store_group(&mut b, p, at, k.out, &results[o]);
        }
    }
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([ELEMENTWISE_BLOCK, 1, 1]);
    b.set_launch_bounds(ELEMENTWISE_BLOCK, None);
    b.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(k: &SwigluKernel) -> String {
        String::from_utf8(k.ptx())
            .expect("ASCII")
            .trim_end_matches('\0')
            .to_string()
    }

    fn all() -> Vec<SwigluKernel> {
        let mut out = Vec::new();
        for op in SwigluOp::ALL {
            let n = op.inputs().len();
            for mask in 0..(1u32 << n) {
                let ins: Vec<Storage> = (0..n)
                    .map(|j| {
                        if mask >> j & 1 == 1 {
                            Storage::Bf16
                        } else {
                            Storage::F32
                        }
                    })
                    .collect();
                for out_s in Storage::ALL {
                    for w in WIDTHS {
                        for rs in [false, true] {
                            out.push(SwigluKernel::new(op, &ins, out_s, w, rs));
                        }
                    }
                }
            }
        }
        out
    }

    #[test]
    fn every_instantiation_verifies_and_is_named_uniquely_unless_identical() {
        let mut seen = std::collections::HashMap::new();
        for k in all() {
            let ptx = text(&k);
            assert!(
                ptx.contains(&format!(".visible .entry {}(", k.name())),
                "{}: entry name",
                k.name()
            );
            // `round_silu` on an f32 gate is the same kernel under the same name.
            if let Some(prev) = seen.insert(k.name(), ptx.clone()) {
                assert_eq!(
                    prev,
                    ptx,
                    "{}: two different kernels share a name",
                    k.name()
                );
            }
        }
    }

    #[test]
    fn the_f32_one_element_kernels_keep_their_base_names() {
        for op in SwigluOp::ALL {
            let ins = vec![Storage::F32; op.inputs().len()];
            assert_eq!(
                SwigluKernel::new(op, &ins, Storage::F32, 1, false).name(),
                op.kernel_name()
            );
        }
        let k = SwigluKernel::new(
            SwigluOp::Backward,
            &[Storage::Bf16; 3],
            Storage::Bf16,
            8,
            true,
        );
        assert_eq!(k.name(), "nsl_swiglu_backward_bbb_b_x8_rs");
    }

    /// Vector widths move operands with `.v4` accesses (b16 for bf16), one
    /// per input and output per group of four; one element per thread never
    /// does.
    #[test]
    fn vector_widths_use_v4_accesses() {
        for w in WIDTHS {
            let k = SwigluKernel::new(
                SwigluOp::Backward,
                &[Storage::Bf16; 3],
                Storage::Bf16,
                w,
                false,
            );
            let t = text(&k);
            let groups = (w / 4) as usize;
            assert_eq!(
                t.matches("ld.global.v4.b16").count(),
                3 * groups,
                "width {w}:\n{t}"
            );
            assert_eq!(
                t.matches("st.global.v4.b16").count(),
                2 * groups,
                "width {w}:\n{t}"
            );
            let k = SwigluKernel::new(
                SwigluOp::Forward,
                &[Storage::F32; 2],
                Storage::F32,
                w,
                false,
            );
            let t = text(&k);
            assert_eq!(
                t.matches("ld.global.v4.f32").count(),
                2 * groups,
                "width {w}:\n{t}"
            );
        }
    }

    /// No multiply-add pair is left for ptxas to contract: every float
    /// multiply and add is explicitly rounded except the sigmoid's own
    /// (whose product feeds `ex2.approx`, as in `nsl_sigmoid_f32`).
    #[test]
    fn products_are_explicitly_rounded() {
        let k = SwigluKernel::new(
            SwigluOp::Backward,
            &[Storage::F32; 3],
            Storage::F32,
            1,
            false,
        );
        let t = text(&k);
        assert_eq!(t.matches("mul.rn.f32").count(), 6, "{t}");
        assert_eq!(t.matches("mul.f32").count(), 1, "{t}");
        assert!(!t.contains("fma"), "{t}");
    }

    /// `round_silu` rounds `silu(gate)` once to bf16 and widens it back, only
    /// when the gate is bf16.
    #[test]
    fn round_silu_inserts_one_round_trip_for_a_bf16_gate() {
        let plain = SwigluKernel::new(
            SwigluOp::Forward,
            &[Storage::Bf16, Storage::Bf16],
            Storage::Bf16,
            1,
            false,
        );
        let rs = SwigluKernel::new(
            SwigluOp::Forward,
            &[Storage::Bf16, Storage::Bf16],
            Storage::Bf16,
            1,
            true,
        );
        let count = |t: &str| t.matches("cvt.rn.bf16.f32").count();
        assert_eq!(count(&text(&rs)), count(&text(&plain)) + 1);
        let f = SwigluKernel::new(
            SwigluOp::Forward,
            &[Storage::F32, Storage::Bf16],
            Storage::Bf16,
            1,
            true,
        );
        assert!(!f.rounds_silu());
    }
}
