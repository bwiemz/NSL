// crates/nsl-kir/src/kernels/optim.rs
//! The runtime's fused optimizer-step kernels, as KIR (roadmap A2 step 11,
//! new-roadmap item 5).
//!
//! `nsl_fase_fused_adamw_step_f32` is the FASE-Deferred AdamW/Adam step for
//! one parameter: one launch in place of the ~15-launch decomposed
//! `UpdateProgram`, and bit-exact with it. Per element, in the decomposed
//! program's rounding order (every arithmetic operation `.rn`, so ptxas
//! cannot contract a multiply into the add that reads it):
//!
//! ```text
//! m'  = rn(rn(m·β₁) + rn(mp·(1-β₁)))                  -> m
//! v'  = rn(rn(v·β₂) + rn(rn(mp·mp)·(1-β₂)))           -> v
//! u   = div.approx(rn(m'·bc1), rn(sqrt.rn(rn(v'·bc2)) + ε))
//! adj = rn(u·(-lr));  if has_wd != 0: adj = rn(adj + rn(θ·(-lr·wd)))
//! θ'  = rn(θ + adj)                                    -> θ
//! ```
//!
//! The quotient is `div.approx.f32` (`KirOp::DivApprox`) because the
//! decomposed program divides with `nsl_div_f32`, which is `div.approx`.
//! `mp` (the accumulated gradient) is only read.
//!
//! Signature `(theta, m, v, mp, n, b1, omb1, b2, omb2, eps, neg_lr,
//! neg_lr_wd, bc1, bc2, has_wd)`: four f32 pointers, `n` a `.u64`, nine
//! `.f32` scalars and a `.u32` flag. One thread per element, blocks of
//! [`ELEMENTWISE_BLOCK`](super::elementwise::ELEMENTWISE_BLOCK).

use super::elementwise::{f32_op1, f32_op2, f32_ptr, finish, index_and_bound, load_f32, store_f32, verified_ptx};
use crate::kernel_ir::{AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator, KirType, VarId};

/// The entry name the runtime launches.
pub const FASE_ADAMW_STEP_NAME: &str = "nsl_fase_fused_adamw_step_f32";

/// The AdamW hyperparameters and the decay flag, in the kernels' parameter
/// order: `b1, omb1, b2, omb2, eps, neg_lr, neg_lr_wd, bc1, bc2, has_wd`.
struct AdamwScalars {
    b1: VarId,
    omb1: VarId,
    b2: VarId,
    omb2: VarId,
    eps: VarId,
    neg_lr: VarId,
    neg_lr_wd: VarId,
    bc1: VarId,
    bc2: VarId,
    has_wd: VarId,
}

fn add_adamw_scalars(b: &mut KirBuilder) -> AdamwScalars {
    use AddressSpace::Global;
    let mut scalar = |name: &str| b.add_param(name, KirType::F32, Global);
    let (b1, omb1, b2, omb2, eps) = (scalar("b1"), scalar("omb1"), scalar("b2"), scalar("omb2"), scalar("eps"));
    let (neg_lr, neg_lr_wd, bc1, bc2) = (scalar("neg_lr"), scalar("neg_lr_wd"), scalar("bc1"), scalar("bc2"));
    let has_wd = b.add_param("has_wd", KirType::U32, Global);
    AdamwScalars { b1, omb1, b2, omb2, eps, neg_lr, neg_lr_wd, bc1, bc2, has_wd }
}

/// How a kernel stores the first moment `m`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FirstMoment {
    /// f32, as the update computes it.
    F32,
    /// bf16, `m'` rounded to nearest even on store (`cvt.rn.bf16.f32`). The
    /// update itself reads the f32 `m'`: only the stored copy is rounded.
    Bf16,
}

/// The per-element AdamW update every kernel here shares, in the hand
/// kernels' order: `m'` and `v'` computed and stored at `m[i]` / `v[i]`,
/// then the step, then the decay branch. Returns `θ' = rn(θ + adj)`, with
/// the builder left in the join block that defines it.
#[allow(clippy::too_many_arguments)]
fn adamw_update(
    b: &mut KirBuilder,
    s: &AdamwScalars,
    (th, mi, vi, g): (VarId, VarId, VarId, VarId),
    m: VarId,
    v: VarId,
    i: VarId,
    first_moment: FirstMoment,
) -> VarId {
    // m' = rn(rn(m·β₁) + rn(mp·(1-β₁)))
    let m_decay = f32_op2(b, KirOp::MulRn, mi, s.b1);
    let m_new_part = f32_op2(b, KirOp::MulRn, g, s.omb1);
    let m_next = f32_op2(b, KirOp::AddRn, m_decay, m_new_part);
    match first_moment {
        FirstMoment::F32 => store_f32(b, m, i, m_next),
        FirstMoment::Bf16 => store_bf16_rn(b, m, i, m_next),
    }

    // v' = rn(rn(v·β₂) + rn(rn(mp·mp)·(1-β₂)))
    let g2 = f32_op2(b, KirOp::MulRn, g, g);
    let v_new_part = f32_op2(b, KirOp::MulRn, g2, s.omb2);
    let v_decay = f32_op2(b, KirOp::MulRn, vi, s.b2);
    let v_next = f32_op2(b, KirOp::AddRn, v_decay, v_new_part);
    store_f32(b, v, i, v_next);

    // u = div.approx(rn(m'·bc1), rn(sqrt.rn(rn(v'·bc2)) + ε)); adj = rn(u·(-lr))
    let m_hat = f32_op2(b, KirOp::MulRn, m_next, s.bc1);
    let v_hat = f32_op2(b, KirOp::MulRn, v_next, s.bc2);
    let root = f32_op1(b, KirOp::Sqrt, v_hat);
    let denom = f32_op2(b, KirOp::AddRn, root, s.eps);
    let u = f32_op2(b, KirOp::DivApprox, m_hat, denom);
    let adj = f32_op2(b, KirOp::MulRn, u, s.neg_lr);

    let wd = b.new_block();
    let join = b.new_block();
    let adj_in = b.add_block_param(join, KirType::F32);
    let zero = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Const(zero, KirConst { ty: KirType::U32, value: ConstValue::U32(0) }));
    let no_wd = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(no_wd, s.has_wd, zero, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(no_wd, KirEdge::with(join, vec![adj]), KirEdge::to(wd)));

    // Decoupled weight decay: adj += rn(θ·(-lr·wd)).
    b.set_block(wd);
    let decay = f32_op2(b, KirOp::MulRn, th, s.neg_lr_wd);
    let adj_wd = f32_op2(b, KirOp::AddRn, adj, decay);
    b.terminate(KirTerminator::Branch(KirEdge::with(join, vec![adj_wd])));

    b.set_block(join);
    f32_op2(b, KirOp::AddRn, th, adj_in)
}

/// Build `nsl_fase_fused_adamw_step_f32` in the hand kernel's instruction
/// order.
///
/// ```text
/// entry:  i = blockIdx.x*blockDim.x + threadIdx.x (u32, widened)
///         if i >= n { br exit } else { br body }
/// body:   θ, m, v, mp = loads; m', v' stored; adj = rn(u·(-lr))
///         if has_wd == 0 { br join(adj) } else { br wd }
/// wd:     br join(rn(adj + rn(θ·(-lr·wd))))
/// join(adj): θ[i] = rn(θ + adj); br exit
/// exit:   ret
/// ```
pub fn build_fase_adamw_step() -> KernelIR {
    build_step(FASE_ADAMW_STEP_NAME, FirstMoment::F32)
}

/// The per-parameter step with a bf16 first moment.
pub const FASE_ADAMW_STEP_MBF16_NAME: &str = "nsl_fase_fused_adamw_step_f32_mbf16";

/// `nsl_fase_fused_adamw_step_f32_mbf16`: [`build_fase_adamw_step`] with the
/// first moment `m` stored as bf16 (`first_moment_dtype="bf16"`). Same
/// signature; `m` points at bf16. `m` is widened exactly on load
/// (`cvt.f32.bf16`), the update is the f32 kernel's op for op, and only the
/// stored `m'` is rounded, to nearest even (`cvt.rn.bf16.f32`): `θ'` is
/// computed from the f32 `m'`.
pub fn build_fase_adamw_step_mbf16() -> KernelIR {
    build_step(FASE_ADAMW_STEP_MBF16_NAME, FirstMoment::Bf16)
}

fn build_step(name: &str, first_moment: FirstMoment) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(name);
    let theta = b.add_param("theta", f32_ptr(), Global);
    let m = b.add_param("m", m_ptr(first_moment), Global);
    let v = b.add_param("v", f32_ptr(), Global);
    let mp = b.add_param("mp", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let s = add_adamw_scalars(&mut b);

    let (i, _, exit) = index_and_bound(&mut b, n);
    let th = load_f32(&mut b, theta, i);
    let mi = load_m(&mut b, m, i, first_moment);
    let vi = load_f32(&mut b, v, i);
    let g = load_f32(&mut b, mp, i);
    let th_next = adamw_update(&mut b, &s, (th, mi, vi, g), m, v, i, first_moment);
    store_f32(&mut b, theta, i, th_next);
    finish(b, exit)
}

/// The pointer type of `m` for a storage mode.
fn m_ptr(first_moment: FirstMoment) -> KirType {
    match first_moment {
        FirstMoment::F32 => f32_ptr(),
        FirstMoment::Bf16 => bf16_ptr(),
    }
}

fn bf16_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::Bf16), AddressSpace::Global)
}

/// `m[i]` as f32: a plain load, or a bf16 load widened exactly (`cvt.f32.bf16`).
fn load_m(b: &mut KirBuilder, m: VarId, i: VarId, first_moment: FirstMoment) -> VarId {
    match first_moment {
        FirstMoment::F32 => load_f32(b, m, i),
        FirstMoment::Bf16 => {
            let addr = b.new_typed_var(bf16_ptr());
            b.emit(KirOp::PtrOffset(addr, m, i));
            let h = b.new_typed_var(KirType::Bf16);
            b.emit(KirOp::Load(h, addr, AddressSpace::Global));
            let x = b.new_typed_var(KirType::F32);
            b.emit(KirOp::Cast(x, h, KirType::F32));
            x
        }
    }
}

/// `base[i] = bf16(x)`, rounded to nearest even (`cvt.rn.bf16.f32`; a NaN
/// stays a NaN, a finite value past the bf16 range rounds to infinity).
fn store_bf16_rn(b: &mut KirBuilder, base: VarId, i: VarId, x: VarId) {
    let h = b.new_typed_var(KirType::Bf16);
    b.emit(KirOp::Cast(h, x, KirType::Bf16));
    let addr = b.new_typed_var(bf16_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, h, AddressSpace::Global));
}

/// [`build_fase_adamw_step`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_step_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_step())
}

/// `base[idx]` for a table of `elem` and a u64 index.
fn table_load(b: &mut KirBuilder, base: VarId, idx: VarId, elem: KirType) -> VarId {
    let addr = b.new_typed_var(KirType::Ptr(Box::new(elem.clone()), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(addr, base, idx));
    let v = b.new_typed_var(elem);
    b.emit(KirOp::Load(v, addr, AddressSpace::Global));
    v
}

fn widen_u64(b: &mut KirBuilder, x: VarId) -> VarId {
    let w = b.new_typed_var(KirType::U64);
    b.emit(KirOp::Cast(w, x, KirType::U64));
    w
}

/// The multi-tensor kernels' flat-grid header.
struct MultiHeader {
    /// The block's parameter index, widened.
    param: VarId,
    /// The element within that parameter, `bbtab[b] + tid`, in 32 bits.
    e32: VarId,
    /// The block the element's work goes in (the builder is left in it).
    body: crate::kernel_ir::BlockId,
    exit: crate::kernel_ir::BlockId,
}

/// `p = bptab[b]; e = bbtab[b] + tid (u32); if e >= ntab[p] { exit }`. The
/// kernel reads neither `%ntid` nor `%nctaid`.
fn multi_header(b: &mut KirBuilder, bptab: VarId, bbtab: VarId, ntab: VarId) -> MultiHeader {
    let entry = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let blk = b.new_typed_var(KirType::U32);
    b.emit(KirOp::BlockIdx(blk, 0));
    let blk = widen_u64(b, blk);
    let param = table_load(b, bptab, blk, KirType::U32);
    let first = table_load(b, bbtab, blk, KirType::U32);
    let tid = b.new_typed_var(KirType::U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let e32 = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Add(e32, first, tid));
    let param = widen_u64(b, param);
    let len = table_load(b, ntab, param, KirType::U32);
    let past = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(past, e32, len, CmpOp::Ge));
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));
    MultiHeader { param, e32, body, exit }
}

/// The multi-tensor entry name the runtime launches.
pub const FASE_ADAMW_MULTI_NAME: &str = "nsl_fase_fused_adamw_multi_f32";

/// `nsl_fase_fused_adamw_multi_f32`: [`build_fase_adamw_step`] over every
/// parameter in one launch, on a FLAT grid.
///
/// Signature `(ttab, mtab, vtab, mptab, ntab, b1, omb1, b2, omb2, eps,
/// neg_lr, neg_lr_wd, bc1, bc2, has_wd, bptab, bbtab, mp_scale)`. The four
/// `*tab` pointer tables hold one device pointer per parameter, `ntab` one
/// `u32` length; `bptab[b]` / `bbtab[b]` are the parameter and element
/// offset of block `b`, built once on the host from the shape list. The
/// kernel reads neither `%ntid` nor `%nctaid`: its element is `bbtab[b] +
/// threadIdx.x`, so `blockDim.x` must be the block size the tables were
/// built with.
///
/// Per element it is the single-parameter step, with two differences:
///
/// - `mp_scale` folds the two-phase-clip Phase-B pre-scale into the read,
///   `g = rn(mp * mp_scale)`, and is branched AROUND when it equals 1.0 so
///   the unclipped path keeps NaN payloads bit-for-bit.
/// - After θ is stored, `mp[e] = 0`: the separate `nsl_tensor_zero_inplace`
///   pass folded in.
///
/// ```text
/// entry:  p = bptab[b]; e = bbtab[b] + tid (u32); if e >= ntab[p] exit
/// body:   θ, m, v, mp bases from the tables; loads at e
///         if mp_scale == 1.0 { br join(g) } else { br scale }
/// scale:  br join(rn(g·mp_scale))
/// join(g): the single-parameter body; wd branch; θ[e] stored; mp[e] = 0
/// ```
pub fn build_fase_adamw_multi() -> KernelIR {
    build_multi(
        FASE_ADAMW_MULTI_NAME,
        FirstMoment::F32,
        Theta::F32,
        MpTail::Zero,
    )
}

/// The multi-tensor step with a bf16 first moment.
pub const FASE_ADAMW_MULTI_MBF16_NAME: &str = "nsl_fase_fused_adamw_multi_f32_mbf16";

/// `nsl_fase_fused_adamw_multi_f32_mbf16`: [`build_fase_adamw_multi`] with
/// every first moment stored as bf16 (`first_moment_dtype="bf16"`). Same
/// signature; `mtab` holds bf16 pointers. Per element it is
/// [`build_fase_adamw_step_mbf16`]'s body with the multi kernel's
/// `mp_scale` fold and `mp` zeroing.
pub fn build_fase_adamw_multi_mbf16() -> KernelIR {
    build_multi(
        FASE_ADAMW_MULTI_MBF16_NAME,
        FirstMoment::Bf16,
        Theta::F32,
        MpTail::Zero,
    )
}

/// How a bf16 parameter's master-mode image is rounded from the f32 master.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Image {
    /// `cvt.rn.bf16.f32`: what the weight-cast cache casts, so the image is
    /// the bits today's GEMMs read.
    Rne,
    /// [`split_hi`]'s rounding: to nearest, a tie toward the truncation. The
    /// test-only reference for [`Theta::Bf16Split`], whose image cannot be
    /// RNE at every tie.
    HalfTowardTruncation,
}

/// How a multi-tensor AdamW kernel stores and steps the parameter θ.
///
/// Every variant computes the same f32 update, op for op ([`adamw_update`]),
/// folds the clip pre-scale into the gradient read (`mp_scale`) and zeroes
/// the consumed gradient. They differ only in where the f32 θ the update
/// reads comes from and where θ' goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Theta {
    /// f32 θ, read and written in place: today's kernels, byte for byte.
    F32,
    /// bf16 θ (the GEMMs' operand) beside an f32 master in optimizer state
    /// (`wtab`): the master is stepped exactly as [`Theta::F32`] steps θ,
    /// then θ is written as the master's image.
    Bf16Master(Image),
    /// bf16 θ = the f32 master's top half rounded ([`split_hi`]) beside the
    /// master's low 16 bits in optimizer state (`wtab`, u16): the kernel
    /// reassembles the exact master ([`join_f32`]), steps it, and splits
    /// θ' again. The master is never stored whole.
    Bf16Split,
    /// bf16 θ and no master: θ is widened exactly (`bits << 16`), and θ' is
    /// stored with stochastic rounding ([`sr_bf16_bits`]) under the launch
    /// key `sr_key` and the per-parameter counter base `ctrtab[p]`.
    Bf16Sr,
}

impl Theta {
    /// Every bf16-parameter variant, in a fixed order (the runtime's module
    /// cache indexes by it).
    pub const BF16: [Theta; 4] = [
        Theta::Bf16Master(Image::Rne),
        Theta::Bf16Master(Image::HalfTowardTruncation),
        Theta::Bf16Split,
        Theta::Bf16Sr,
    ];

    /// The element type of θ's table: bf16 storage for the master's RNE image
    /// (a `cvt.rn.bf16.f32` store), raw 16-bit words for the rest.
    fn param_elem(self) -> KirType {
        match self {
            Theta::F32 => KirType::F32,
            Theta::Bf16Master(Image::Rne) => KirType::Bf16,
            _ => KirType::U16,
        }
    }

    /// The element type of the optimizer-state table beside θ (`wtab`): the
    /// f32 master, or the split master's low half.
    fn master_elem(self) -> Option<KirType> {
        match self {
            Theta::Bf16Master(_) => Some(KirType::F32),
            Theta::Bf16Split => Some(KirType::U16),
            Theta::F32 | Theta::Bf16Sr => None,
        }
    }
}

/// The entry name of the multi-tensor step for a parameter storage and a
/// first-moment dtype.
pub fn multi_name(theta: Theta, m_bf16: bool) -> &'static str {
    match (theta, m_bf16) {
        (Theta::F32, false) => FASE_ADAMW_MULTI_NAME,
        (Theta::F32, true) => FASE_ADAMW_MULTI_MBF16_NAME,
        (Theta::Bf16Master(Image::Rne), false) => "nsl_fase_fused_adamw_multi_bf16master",
        (Theta::Bf16Master(Image::Rne), true) => "nsl_fase_fused_adamw_multi_bf16master_mbf16",
        (Theta::Bf16Master(Image::HalfTowardTruncation), false) => {
            "nsl_fase_fused_adamw_multi_bf16master_tz"
        }
        (Theta::Bf16Master(Image::HalfTowardTruncation), true) => {
            "nsl_fase_fused_adamw_multi_bf16master_tz_mbf16"
        }
        (Theta::Bf16Split, false) => "nsl_fase_fused_adamw_multi_bf16split",
        (Theta::Bf16Split, true) => "nsl_fase_fused_adamw_multi_bf16split_mbf16",
        (Theta::Bf16Sr, false) => "nsl_fase_fused_adamw_multi_bf16sr_mp",
        (Theta::Bf16Sr, true) => "nsl_fase_fused_adamw_multi_bf16sr_mp_mbf16",
    }
}

/// The multi-tensor step for a parameter storage (see [`Theta`]) and a
/// first-moment dtype. [`Theta::F32`] is [`build_fase_adamw_multi`] /
/// [`build_fase_adamw_multi_mbf16`].
///
/// Signatures, after the f32 kernel's (`ttab, mtab, vtab, mptab, ntab`, the
/// nine scalars and `has_wd`, `bptab, bbtab, mp_scale`):
///
/// - `Bf16Master`: `wtab` (f32 masters) after `ttab` (bf16 images);
/// - `Bf16Split`: `wtab` (u16 low halves) after `ttab` (u16 high halves);
/// - `Bf16Sr`: `sr_key` (u64) and `ctrtab` (u64 per parameter) after
///   `has_wd`.
pub fn build_param_multi(theta: Theta, m_bf16: bool) -> KernelIR {
    let fm = if m_bf16 {
        FirstMoment::Bf16
    } else {
        FirstMoment::F32
    };
    build_multi(multi_name(theta, m_bf16), fm, theta, MpTail::Zero)
}

/// What a multi-tensor AdamW kernel leaves in the accumulated gradient `mp`
/// once it has read it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MpTail {
    /// `mp[e] = 0`: the next window adds onto zeros. Every kernel above.
    Zero,
    /// `mp` is left as it is. The runtime marks the buffers fresh instead
    /// (`nsl-runtime`'s `accum_fresh`), and the next window's first write to
    /// each one overwrites it, so the 4-byte store per element and the first
    /// micro-batch's read of the zeros both go.
    Keep,
}

/// The entry name of a multi-tensor step: [`multi_name`], with `_keepmp`
/// appended for [`MpTail::Keep`].
pub fn multi_kernel_name(theta: Theta, m_bf16: bool, tail: MpTail) -> String {
    match tail {
        MpTail::Zero => multi_name(theta, m_bf16).to_string(),
        MpTail::Keep => format!("{}_keepmp", multi_name(theta, m_bf16)),
    }
}

/// The multi-tensor step for a parameter storage, a first-moment dtype and an
/// accumulator tail. With [`MpTail::Zero`] it is [`build_param_multi`] (and,
/// for [`Theta::F32`], [`build_fase_adamw_multi`] /
/// [`build_fase_adamw_multi_mbf16`]); [`MpTail::Keep`] drops the final store
/// and nothing else.
pub fn build_multi_kernel(theta: Theta, m_bf16: bool, tail: MpTail) -> KernelIR {
    let fm = if m_bf16 {
        FirstMoment::Bf16
    } else {
        FirstMoment::F32
    };
    build_multi(&multi_kernel_name(theta, m_bf16, tail), fm, theta, tail)
}

/// [`build_multi_kernel`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn multi_kernel_ptx(theta: Theta, m_bf16: bool, tail: MpTail) -> Vec<u8> {
    verified_ptx(build_multi_kernel(theta, m_bf16, tail))
}

/// [`build_param_multi`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn param_multi_ptx(theta: Theta, m_bf16: bool) -> Vec<u8> {
    verified_ptx(build_param_multi(theta, m_bf16))
}

/// The bf16 image of an f32 bit pattern under [`Theta::Bf16Split`]: the top
/// 16 bits rounded to nearest, a tie (`bits & 0xFFFF == 0x8000`) toward the
/// truncation, so that [`split_lo`] alone says whether the rounding carried
/// and [`join_f32`] can undo it. Differs from round-to-nearest-even only at
/// a tie whose truncation is odd (RNE carries there): 2^-17 of bit patterns.
/// A NaN maps to the quiet NaN `sign | 0x7FC0` (the SR tail's rule; the
/// carry trick would turn some NaNs into ±∞ or a zero). Host reference for
/// the kernels' split, in the runtime's `param_storage` module.
pub fn split_hi(bits: u32) -> u16 {
    if bits & 0x7FFF_FFFF > 0x7F80_0000 {
        return ((bits >> 16) & 0x8000) as u16 | 0x7FC0;
    }
    (bits.wrapping_add(0x7FFF) >> 16) as u16
}

/// The remainder [`split_hi`] leaves: the low 16 bits.
pub fn split_lo(bits: u32) -> u16 {
    bits as u16
}

/// Reassemble [`split_hi`] / [`split_lo`]: the identity on every non-NaN
/// bit pattern (a NaN comes back a NaN). `lo > 0x8000` is exactly "the
/// rounding carried", so the truncation is `hi - 1` there.
pub fn join_f32(hi: u16, lo: u16) -> u32 {
    let carry = 0x8000u32.wrapping_sub(lo as u32) >> 31;
    ((hi as u32).wrapping_sub(carry) << 16) | lo as u32
}

/// `base[i]`, a u16, zero-extended to u32.
fn load_u16(b: &mut KirBuilder, base: VarId, i: VarId) -> VarId {
    let addr = b.new_typed_var(u16_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    let h = b.new_typed_var(KirType::U16);
    b.emit(KirOp::Load(h, addr, AddressSpace::Global));
    let w = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(w, h, KirType::U32));
    w
}

/// [`join_f32`] in KIR: the f32 master from its halves (both `U32`, zero
/// extended). Branch-free: `carry = (0x8000 - lo) >> 31`.
fn join_kir(b: &mut KirBuilder, hi: VarId, lo: VarId) -> VarId {
    let half = int_const(b, KirType::U32, ConstValue::U32(0x8000));
    let diff = int_op2(b, KirType::U32, KirOp::Sub, half, lo);
    let carry = u32_imm(b, KirOp::Shr, diff, 31);
    let trunc = int_op2(b, KirType::U32, KirOp::Sub, hi, carry);
    let high = u32_imm(b, KirOp::Shl, trunc, 16);
    let bits = int_op2(b, KirType::U32, KirOp::Or, high, lo);
    let x = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Bitcast(x, bits));
    x
}

/// [`split_hi`] in KIR, of the `U32` bit pattern `bits`: `(bits + 0x7FFF)
/// >> 16`, or the quiet NaN for a NaN (a `selp`, no branch).
fn split_hi_kir(b: &mut KirBuilder, bits: VarId) -> VarId {
    let carried = u32_imm(b, KirOp::Add, bits, 0x7FFF);
    let rounded = u32_imm(b, KirOp::Shr, carried, 16);
    let mag = u32_imm(b, KirOp::And, bits, 0x7FFF_FFFF);
    let inf = int_const(b, KirType::U32, ConstValue::U32(0x7F80_0000));
    let nan = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(nan, mag, inf, CmpOp::Gt));
    let sign = u32_imm(b, KirOp::Shr, bits, 16);
    let sign = u32_imm(b, KirOp::And, sign, 0x8000);
    let qnan = u32_imm(b, KirOp::Or, sign, 0x7FC0);
    let hi = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Select(hi, nan, qnan, rounded));
    hi
}

fn build_multi(name: &str, first_moment: FirstMoment, theta_mode: Theta, tail: MpTail) -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(name);
    let table =
        |elem: KirType| KirType::Ptr(Box::new(KirType::Ptr(Box::new(elem), Global)), Global);
    let u32_ptr = || KirType::Ptr(Box::new(KirType::U32), Global);
    let elem_ptr = |elem: KirType| KirType::Ptr(Box::new(elem), Global);
    // The f32 kernel's parameters, in its order; the bf16 variants add
    // `wtab` after `ttab`, or `sr_key, ctrtab` after the scalars.
    let ttab = b.add_param("ttab", table(theta_mode.param_elem()), Global);
    let wtab = theta_mode
        .master_elem()
        .map(|e| (b.add_param("wtab", table(e.clone()), Global), e));
    let mtab = b.add_param(
        "mtab",
        KirType::Ptr(Box::new(m_ptr(first_moment)), Global),
        Global,
    );
    let vtab = b.add_param("vtab", table(KirType::F32), Global);
    let mptab = b.add_param("mptab", table(KirType::F32), Global);
    let ntab = b.add_param("ntab", u32_ptr(), Global);
    let s = add_adamw_scalars(&mut b);
    let sr = (theta_mode == Theta::Bf16Sr).then(|| {
        (
            b.add_param("sr_key", KirType::U64, Global),
            b.add_param("ctrtab", elem_ptr(KirType::U64), Global),
        )
    });
    let bptab = b.add_param("bptab", u32_ptr(), Global);
    let bbtab = b.add_param("bbtab", u32_ptr(), Global);
    let mp_scale = b.add_param("mp_scale", KirType::F32, Global);

    let MultiHeader { param, e32, body, exit } = multi_header(&mut b, bptab, bbtab, ntab);

    b.set_block(body);
    let theta = table_load(&mut b, ttab, param, elem_ptr(theta_mode.param_elem()));
    let master = wtab.map(|(w, e)| table_load(&mut b, w, param, elem_ptr(e)));
    let m = table_load(&mut b, mtab, param, m_ptr(first_moment));
    let v = table_load(&mut b, vtab, param, f32_ptr());
    let mp = table_load(&mut b, mptab, param, f32_ptr());
    let sr = sr.map(|(key, ctrtab)| (key, table_load(&mut b, ctrtab, param, KirType::U64)));
    let i = widen_u64(&mut b, e32);
    let th = match (theta_mode, master) {
        (Theta::F32, _) => load_f32(&mut b, theta, i),
        (Theta::Bf16Master(_), Some(w)) => load_f32(&mut b, w, i),
        (Theta::Bf16Split, Some(lo_base)) => {
            let hi = load_u16(&mut b, theta, i);
            let lo = load_u16(&mut b, lo_base, i);
            join_kir(&mut b, hi, lo)
        }
        (Theta::Bf16Sr, _) => load_bf16(&mut b, theta, i),
        (_, None) => unreachable!("a master-carrying variant has wtab"),
    };
    let mi = load_m(&mut b, m, i, first_moment);
    let vi = load_f32(&mut b, v, i);
    let g_raw = load_f32(&mut b, mp, i);

    // Phase-B clip pre-scale, skipped at exactly 1.0.
    let scale = b.new_block();
    let scaled = b.new_block();
    let g = b.add_block_param(scaled, KirType::F32);
    let one = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Const(one, KirConst { ty: KirType::F32, value: ConstValue::F32(1.0) }));
    let unscaled = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(unscaled, mp_scale, one, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(unscaled, KirEdge::with(scaled, vec![g_raw]), KirEdge::to(scale)));
    b.set_block(scale);
    let g_scaled = f32_op2(&mut b, KirOp::MulRn, g_raw, mp_scale);
    b.terminate(KirTerminator::Branch(KirEdge::with(scaled, vec![g_scaled])));

    b.set_block(scaled);
    let th_next = adamw_update(&mut b, &s, (th, mi, vi, g), m, v, i, first_moment);
    match (theta_mode, master) {
        (Theta::F32, _) => store_f32(&mut b, theta, i, th_next),
        (Theta::Bf16Master(image), Some(w)) => {
            store_f32(&mut b, w, i, th_next);
            match image {
                Image::Rne => store_bf16_rn(&mut b, theta, i, th_next),
                Image::HalfTowardTruncation => {
                    let bits = b.new_typed_var(KirType::U32);
                    b.emit(KirOp::Bitcast(bits, th_next));
                    let hi = split_hi_kir(&mut b, bits);
                    store_u16(&mut b, theta, i, hi);
                }
            }
        }
        (Theta::Bf16Split, Some(lo_base)) => {
            let bits = b.new_typed_var(KirType::U32);
            b.emit(KirOp::Bitcast(bits, th_next));
            let hi = split_hi_kir(&mut b, bits);
            store_u16(&mut b, theta, i, hi);
            // `cvt.u16.u32` keeps the low 16 bits: the remainder.
            store_u16(&mut b, lo_base, i, bits);
        }
        (Theta::Bf16Sr, _) => {
            let (key, ctr_base) = sr.expect("the SR variant has its key and counters");
            store_bf16_sr(&mut b, theta, i, th_next, key, ctr_base);
        }
        (_, None) => unreachable!("a master-carrying variant has wtab"),
    }
    // The accumulated gradient is consumed: zero it (was a separate launch),
    // unless the caller marks the buffers fresh instead (`MpTail::Keep`).
    if tail == MpTail::Zero {
        let fzero = b.new_typed_var(KirType::F32);
        b.emit(KirOp::Const(
            fzero,
            KirConst {
                ty: KirType::F32,
                value: ConstValue::F32(0.0),
            },
        ));
        store_f32(&mut b, mp, i, fzero);
    }
    finish(b, exit)
}

/// [`build_fase_adamw_multi`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_multi_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_multi())
}

/// [`build_fase_adamw_step_mbf16`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_step_mbf16_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_step_mbf16())
}

/// [`build_fase_adamw_multi_mbf16`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_multi_mbf16_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_multi_mbf16())
}

/// The single-parameter SR-BF16 step's entry name.
pub const FASE_ADAMW_STEP_BF16SR_NAME: &str = "nsl_fase_fused_adamw_step_bf16sr";

/// The standalone SR-BF16 rounding probe's entry name.
pub const SR_BF16_ROUND_PROBE_NAME: &str = "nsl_sr_bf16_round_probe";

/// splitmix64's counter increment, the host's `sr_bf16::SR_STEP_SALT`.
pub const SR_SPLITMIX_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// splitmix64's two finalizer multipliers.
const SR_SPLITMIX_M1: u64 = 0xBF58_476D_1CE4_E5B9;
const SR_SPLITMIX_M2: u64 = 0x94D0_49BB_1331_11EB;

fn int_const(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = b.new_typed_var(ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn int_op2(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = b.new_typed_var(ty);
    b.emit(op(dst, x, y));
    dst
}

/// `x op c` for a `u32` constant `c`.
fn u32_imm(b: &mut KirBuilder, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, c: u32) -> VarId {
    let c = int_const(b, KirType::U32, ConstValue::U32(c));
    int_op2(b, KirType::U32, op, x, c)
}

/// The SR-BF16 rounding tail: the f32 bit pattern `bits` (a `U32`) to bf16
/// storage bits, stochastically rounded. It is the hand kernels' tail and
/// the host's `sr_bf16::sr_bf16_round_counter`, bit for bit.
///
/// ```text
/// sign = (bits & 0x80000000) >> 16
/// if bits & 0x7f800000 == 0x7f800000:              special
///     NaN (mantissa != 0) -> sign | 0x7fc0; ±∞ -> sign | 0x7f80
/// z = splitmix64(key + (ctr_base + i)·γ)            dither
/// r = bits + (z & 0xffff)
/// if r & 0x7f800000 == 0x7f800000: sign | 0x7f7f    saturate
/// else r >> 16                                      truncate
/// ```
///
/// Returns the bits, a `U32`, as the parameter of the block the four paths
/// join in; the builder is left there.
fn sr_bf16_bits(b: &mut KirBuilder, bits: VarId, i: VarId, key: VarId, ctr_base: VarId) -> VarId {
    let sign = u32_imm(b, KirOp::And, bits, 0x8000_0000);
    let sign = u32_imm(b, KirOp::Shr, sign, 16);
    let exp = u32_imm(b, KirOp::And, bits, 0x7F80_0000);
    let all_ones = int_const(b, KirType::U32, ConstValue::U32(0x7F80_0000));
    let special = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(special, exp, all_ones, CmpOp::Eq));
    let dither = b.new_block();
    let truncate = b.new_block();
    let saturate = b.new_block();
    let special_blk = b.new_block();
    let inf = b.new_block();
    let qnan = b.new_block();
    let store = b.new_block();
    let out = b.add_block_param(store, KirType::U32);
    b.terminate(KirTerminator::CondBranch(special, KirEdge::to(special_blk), KirEdge::to(dither)));

    // The dither: the low 16 bits of splitmix64(key, ctr_base + i).
    b.set_block(dither);
    let u64_imm = |b: &mut KirBuilder, c: u64| int_const(b, KirType::U64, ConstValue::U64(c));
    let ctr = int_op2(b, KirType::U64, KirOp::Add, ctr_base, i);
    let gamma = u64_imm(b, SR_SPLITMIX_GAMMA);
    let z = int_op2(b, KirType::U64, KirOp::Mul, ctr, gamma);
    let mut z = int_op2(b, KirType::U64, KirOp::Add, key, z);
    for (shift, mult) in [(30, Some(SR_SPLITMIX_M1)), (27, Some(SR_SPLITMIX_M2)), (31, None)] {
        let amount = int_const(b, KirType::U32, ConstValue::U32(shift));
        let hi = int_op2(b, KirType::U64, KirOp::Shr, z, amount);
        z = int_op2(b, KirType::U64, KirOp::Xor, z, hi);
        if let Some(mult) = mult {
            let mult = u64_imm(b, mult);
            z = int_op2(b, KirType::U64, KirOp::Mul, z, mult);
        }
    }
    let low = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(low, z, KirType::U32));
    let d = u32_imm(b, KirOp::And, low, 0xFFFF);
    let r = int_op2(b, KirType::U32, KirOp::Add, bits, d);
    let r_exp = u32_imm(b, KirOp::And, r, 0x7F80_0000);
    let all_ones = int_const(b, KirType::U32, ConstValue::U32(0x7F80_0000));
    let overflow = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(overflow, r_exp, all_ones, CmpOp::Eq));
    b.terminate(KirTerminator::CondBranch(overflow, KirEdge::to(saturate), KirEdge::to(truncate)));

    b.set_block(truncate);
    let t = u32_imm(b, KirOp::Shr, r, 16);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    // Rounding carried into the all-ones exponent: ±max-normal.
    b.set_block(saturate);
    let t = u32_imm(b, KirOp::Or, sign, 0x7F7F);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    // The input was ±∞ or NaN: propagate, quieting a NaN.
    b.set_block(special_blk);
    let mant = u32_imm(b, KirOp::And, bits, 0x007F_FFFF);
    let zero = int_const(b, KirType::U32, ConstValue::U32(0));
    let nan = b.new_typed_var(KirType::Bool);
    b.emit(KirOp::Cmp(nan, mant, zero, CmpOp::Ne));
    b.terminate(KirTerminator::CondBranch(nan, KirEdge::to(qnan), KirEdge::to(inf)));

    b.set_block(inf);
    let t = u32_imm(b, KirOp::Or, sign, 0x7F80);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    b.set_block(qnan);
    let t = u32_imm(b, KirOp::Or, sign, 0x7FC0);
    b.terminate(KirTerminator::Branch(KirEdge::with(store, vec![t])));

    b.set_block(store);
    out
}

fn u16_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::U16), AddressSpace::Global)
}

/// `base[i] = bits` for a `u16` array: `cvt.u16.u32`, `st.global.u16`.
fn store_u16(b: &mut KirBuilder, base: VarId, i: VarId, bits: VarId) {
    let h = b.new_typed_var(KirType::U16);
    b.emit(KirOp::Cast(h, bits, KirType::U16));
    let addr = b.new_typed_var(u16_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    b.emit(KirOp::Store(addr, h, AddressSpace::Global));
}

/// `nsl_fase_fused_adamw_step_bf16sr`: [`build_fase_adamw_step`] with a
/// bf16 θ and a stochastically rounded θ store (P4 item 17, SR-BF16).
///
/// Signature `(theta, m, v, mp, n, b1, omb1, b2, omb2, eps, neg_lr,
/// neg_lr_wd, bc1, bc2, has_wd, sr_key, sr_ctr_base)`. `theta` holds bf16
/// bits; `m`, `v`, `mp` are f32. The θ load widens exactly (`bits << 16`,
/// a `mov.b32` into f32). The update is the f32 step's, rounding for
/// rounding. The store is [`sr_bf16_bits`] with the counter
/// `sr_ctr_base + i` and the launch key `sr_key` (the host's
/// `sr_step_key(seed, step)`).
pub fn build_fase_adamw_step_bf16sr() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(FASE_ADAMW_STEP_BF16SR_NAME);
    let theta = b.add_param("theta", u16_ptr(), Global);
    let m = b.add_param("m", f32_ptr(), Global);
    let v = b.add_param("v", f32_ptr(), Global);
    let mp = b.add_param("mp", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let s = add_adamw_scalars(&mut b);
    let key = b.add_param("sr_key", KirType::U64, Global);
    let ctr_base = b.add_param("sr_ctr_base", KirType::U64, Global);

    let (i, _, exit) = index_and_bound(&mut b, n);
    let th = load_bf16(&mut b, theta, i);
    let mi = load_f32(&mut b, m, i);
    let vi = load_f32(&mut b, v, i);
    let g = load_f32(&mut b, mp, i);
    let th_next = adamw_update(&mut b, &s, (th, mi, vi, g), m, v, i, FirstMoment::F32);
    store_bf16_sr(&mut b, theta, i, th_next, key, ctr_base);
    finish(b, exit)
}

/// `base[i]`, a bf16, widened exactly to f32: `bits << 16`, `mov.b32`.
fn load_bf16(b: &mut KirBuilder, base: VarId, i: VarId) -> VarId {
    let addr = b.new_typed_var(u16_ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    let h = b.new_typed_var(KirType::U16);
    b.emit(KirOp::Load(h, addr, AddressSpace::Global));
    let w = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Cast(w, h, KirType::U32));
    let w = u32_imm(b, KirOp::Shl, w, 16);
    let x = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Bitcast(x, w));
    x
}

/// `base[i] = sr_bf16(x)` with the counter `ctr_base + i`.
fn store_bf16_sr(b: &mut KirBuilder, base: VarId, i: VarId, x: VarId, key: VarId, ctr_base: VarId) {
    let bits = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Bitcast(bits, x));
    let out = sr_bf16_bits(b, bits, i, key, ctr_base);
    store_u16(b, base, i, out);
}

/// [`build_fase_adamw_step_bf16sr`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_step_bf16sr_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_step_bf16sr())
}

/// The multi-tensor SR-BF16 step's entry name.
pub const FASE_ADAMW_MULTI_BF16SR_NAME: &str = "nsl_fase_fused_adamw_multi_bf16sr";

/// `nsl_fase_fused_adamw_multi_bf16sr`: [`build_fase_adamw_step_bf16sr`]
/// over every parameter in one launch, on [`build_fase_adamw_multi`]'s flat
/// grid.
///
/// Signature `(ttab, mtab, vtab, mptab, ntab, b1, omb1, b2, omb2, eps,
/// neg_lr, neg_lr_wd, bc1, bc2, has_wd, sr_key, ctrtab, bptab, bbtab)`.
/// `ttab` holds bf16 θ pointers, `ctrtab` one `u64` SR counter base per
/// parameter; element `e` of parameter `p` rounds with the counter
/// `ctrtab[p] + e`, exactly the per-parameter kernel's `sr_ctr_base + i`.
///
/// Per element it is the per-parameter SR step, with neither of the f32
/// multi kernel's extras: no `mp_scale` (the SR path has no two-phase clip
/// to fold in) and no `mp` zeroing (the per-parameter SR kernel leaves `mp`
/// to the FASE-Deferred lifecycle, and this kernel replaces it
/// observationally).
///
/// ```text
/// entry:  p = bptab[b]; e = bbtab[b] + tid (u32); if e >= ntab[p] exit
/// body:   bases and ctrtab[p] from the tables; θ (bf16), m, v, mp at e
///         the single-parameter body; wd branch; SR tail; θ[e] stored
/// ```
pub fn build_fase_adamw_multi_bf16sr() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(FASE_ADAMW_MULTI_BF16SR_NAME);
    let table = |elem: KirType| KirType::Ptr(Box::new(elem), Global);
    let ttab = b.add_param("ttab", table(u16_ptr()), Global);
    let mtab = b.add_param("mtab", table(f32_ptr()), Global);
    let vtab = b.add_param("vtab", table(f32_ptr()), Global);
    let mptab = b.add_param("mptab", table(f32_ptr()), Global);
    let ntab = b.add_param("ntab", table(KirType::U32), Global);
    let s = add_adamw_scalars(&mut b);
    let key = b.add_param("sr_key", KirType::U64, Global);
    let ctrtab = b.add_param("ctrtab", table(KirType::U64), Global);
    let bptab = b.add_param("bptab", table(KirType::U32), Global);
    let bbtab = b.add_param("bbtab", table(KirType::U32), Global);

    let MultiHeader { param, e32, body, exit } = multi_header(&mut b, bptab, bbtab, ntab);

    b.set_block(body);
    let theta = table_load(&mut b, ttab, param, u16_ptr());
    let m = table_load(&mut b, mtab, param, f32_ptr());
    let v = table_load(&mut b, vtab, param, f32_ptr());
    let mp = table_load(&mut b, mptab, param, f32_ptr());
    let ctr_base = table_load(&mut b, ctrtab, param, KirType::U64);
    let i = widen_u64(&mut b, e32);
    let th = load_bf16(&mut b, theta, i);
    let mi = load_f32(&mut b, m, i);
    let vi = load_f32(&mut b, v, i);
    let g = load_f32(&mut b, mp, i);
    let th_next = adamw_update(&mut b, &s, (th, mi, vi, g), m, v, i, FirstMoment::F32);
    store_bf16_sr(&mut b, theta, i, th_next, key, ctr_base);
    finish(b, exit)
}

/// [`build_fase_adamw_multi_bf16sr`], lowered to a NUL-terminated PTX
/// module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn fase_adamw_multi_bf16sr_ptx() -> Vec<u8> {
    verified_ptx(build_fase_adamw_multi_bf16sr())
}

/// `nsl_sr_bf16_round_probe(src, dst, n, sr_key, sr_ctr_base)`: the SR-BF16
/// tail alone, `dst[i] = sr_bf16_bits(src[i])`, over f32 bit patterns
/// (`src` is read as `u32`). The parity gate drives it with adversarial
/// inputs (max-normal, subnormals, ±∞, NaN) against the host reference,
/// without the optimizer arithmetic in the way.
pub fn build_sr_bf16_round_probe() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(SR_BF16_ROUND_PROBE_NAME);
    let src = b.add_param("src", KirType::Ptr(Box::new(KirType::U32), Global), Global);
    let dst = b.add_param("dst", u16_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let key = b.add_param("sr_key", KirType::U64, Global);
    let ctr_base = b.add_param("sr_ctr_base", KirType::U64, Global);

    let (i, _, exit) = index_and_bound(&mut b, n);
    let addr = b.new_typed_var(KirType::Ptr(Box::new(KirType::U32), Global));
    b.emit(KirOp::PtrOffset(addr, src, i));
    let bits = b.new_typed_var(KirType::U32);
    b.emit(KirOp::Load(bits, addr, Global));
    let out = sr_bf16_bits(&mut b, bits, i, key, ctr_base);
    store_u16(&mut b, dst, i, out);
    finish(b, exit)
}

/// [`build_sr_bf16_round_probe`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn sr_bf16_round_probe_ptx() -> Vec<u8> {
    verified_ptx(build_sr_bf16_round_probe())
}

/// The split-master conversion's entry name.
pub const PARAM_SPLIT_NAME: &str = "nsl_param_split_f32";

/// `nsl_param_split_f32(src, hi, lo, n)`: `hi[i] = split_hi(src[i])`, `lo[i] =
/// split_lo(src[i])` over f32 bit patterns. Converts an f32 parameter to the
/// split storage ([`Theta::Bf16Split`]) at train setup, and writes the
/// half-toward-truncation image of a test-only master ([`Image`]).
pub fn build_param_split() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(PARAM_SPLIT_NAME);
    let src = b.add_param("src", KirType::Ptr(Box::new(KirType::U32), Global), Global);
    let hi = b.add_param("hi", u16_ptr(), Global);
    let lo = b.add_param("lo", u16_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let (i, _, exit) = index_and_bound(&mut b, n);
    let bits = table_load(&mut b, src, i, KirType::U32);
    let h = split_hi_kir(&mut b, bits);
    store_u16(&mut b, hi, i, h);
    store_u16(&mut b, lo, i, bits);
    finish(b, exit)
}

/// [`build_param_split`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn param_split_ptx() -> Vec<u8> {
    verified_ptx(build_param_split())
}

/// The split-master reassembly's entry name.
pub const PARAM_JOIN_NAME: &str = "nsl_param_join_f32";

/// `nsl_param_join_f32(hi, lo, dst, n)`: `dst[i] = join_f32(hi[i], lo[i])`, the
/// exact f32 master of a split parameter: what teardown widens it back to,
/// what a checkpoint saves, and what an f32 GEMM of it reads.
pub fn build_param_join() -> KernelIR {
    use AddressSpace::Global;
    let mut b = KirBuilder::new(PARAM_JOIN_NAME);
    let hi = b.add_param("hi", u16_ptr(), Global);
    let lo = b.add_param("lo", u16_ptr(), Global);
    let dst = b.add_param("dst", f32_ptr(), Global);
    let n = b.add_param("n", KirType::U64, Global);
    let (i, _, exit) = index_and_bound(&mut b, n);
    let h = load_u16(&mut b, hi, i);
    let l = load_u16(&mut b, lo, i);
    let x = join_kir(&mut b, h, l);
    store_f32(&mut b, dst, i, x);
    finish(b, exit)
}

/// [`build_param_join`], lowered to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module.
pub fn param_join_ptx() -> Vec<u8> {
    verified_ptx(build_param_join())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sr_bf16_kernels_verify_and_share_one_tail() {
        for (ir, name) in [
            (build_sr_bf16_round_probe(), SR_BF16_ROUND_PROBE_NAME),
            (build_fase_adamw_step_bf16sr(), FASE_ADAMW_STEP_BF16SR_NAME),
            (build_fase_adamw_multi_bf16sr(), FASE_ADAMW_MULTI_BF16SR_NAME),
        ] {
            crate::kir_verify::verify(&ir).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(ir.name, name);
        }
        let probe = String::from_utf8(sr_bf16_round_probe_ptx()).unwrap();
        let step = String::from_utf8(fase_adamw_step_bf16sr_ptx()).unwrap();
        let multi = String::from_utf8(fase_adamw_multi_bf16sr_ptx()).unwrap();
        for p in [&probe, &step, &multi] {
            for c in [SR_SPLITMIX_GAMMA, SR_SPLITMIX_M1, SR_SPLITMIX_M2] {
                assert_eq!(p.matches(&format!(", {c};")).count(), 1, "{c:#x}");
            }
            assert_eq!(p.matches("st.global.u16 ").count(), 1);
            assert_eq!(p.matches("cvt.u16.u32 ").count(), 1);
        }
        for p in [&step, &multi] {
            assert_eq!(p.matches("mov.b32 ").count(), 2, "θ in, θ' out");
            assert_eq!(p.matches("ld.global.u16 ").count(), 1);
            assert_eq!(p.matches("div.approx.f32").count(), 1);
            assert_eq!(p.matches("mul.rn.f32").count(), 9, "no mp_scale");
            assert_eq!(p.matches("st.global.f32 ").count(), 2, "m and v only: no mp zeroing");
        }
        assert!(!multi.contains("%ntid"), "the element is bbtab[b] + tid: {multi}");
    }

    #[test]
    fn fase_adamw_step_verifies_and_spells_its_rounding() {
        let ptx = String::from_utf8(fase_adamw_step_ptx()).unwrap();
        assert!(ptx.contains(".visible .entry nsl_fase_fused_adamw_step_f32("), "{ptx}");
        assert_eq!(ptx.matches("div.approx.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("sqrt.rn.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("mul.rn.f32").count(), 9, "{ptx}");
        assert_eq!(ptx.matches("add.rn.f32").count(), 5, "{ptx}");
        assert!(!ptx.contains("fma") && !ptx.contains("div.rn"), "{ptx}");
    }

    /// The bf16-first-moment kernels are the f32 kernels with `m` loaded by
    /// one exact widening and stored by one round-to-nearest-even narrowing,
    /// and nothing else changed: the same arithmetic instruction counts.
    #[test]
    fn the_bf16_first_moment_kernels_round_m_once_and_change_nothing_else() {
        for (ir, name) in [
            (build_fase_adamw_step_mbf16(), FASE_ADAMW_STEP_MBF16_NAME),
            (build_fase_adamw_multi_mbf16(), FASE_ADAMW_MULTI_MBF16_NAME),
        ] {
            crate::kir_verify::verify(&ir).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(ir.name, name);
        }
        let pairs = [
            (fase_adamw_step_ptx(), fase_adamw_step_mbf16_ptx(), 9),
            (fase_adamw_multi_ptx(), fase_adamw_multi_mbf16_ptx(), 10),
        ];
        for (f32_ptx, bf16_ptx, muls) in pairs {
            let (f, h) = (String::from_utf8(f32_ptx).unwrap(), String::from_utf8(bf16_ptx).unwrap());
            assert_eq!(h.matches("cvt.rn.bf16.f32").count(), 1, "one rounding, on the m store: {h}");
            assert_eq!(h.matches("cvt.f32.bf16").count(), 1, "one exact widening, on the m load: {h}");
            assert!(!h.contains("cvt.rz") && !h.contains("cvt.rm"), "{h}");
            for (op, n) in [("mul.rn.f32", muls), ("add.rn.f32", 5), ("div.approx.f32", 1), ("sqrt.rn.f32", 1)] {
                assert_eq!(h.matches(op).count(), n, "{op}: {h}");
                assert_eq!(f.matches(op).count(), n, "{op}: {f}");
            }
            // θ, v (and in the multi kernel the zeroed mp) stay f32 stores;
            // the f32 kernel's m store is the one that moved to bf16.
            assert_eq!(h.matches("st.global.f32 ").count() + 1, f.matches("st.global.f32 ").count());
            assert!(!h.contains("fma") && !h.contains("div.rn"), "{h}");
        }
    }

    #[test]
    fn fase_adamw_multi_verifies_and_spells_its_rounding() {
        let ptx = String::from_utf8(fase_adamw_multi_ptx()).unwrap();
        assert!(ptx.contains(".visible .entry nsl_fase_fused_adamw_multi_f32("), "{ptx}");
        assert_eq!(ptx.matches("div.approx.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("sqrt.rn.f32").count(), 1, "{ptx}");
        assert_eq!(ptx.matches("mul.rn.f32").count(), 10, "{ptx}");
        assert_eq!(ptx.matches("add.rn.f32").count(), 5, "{ptx}");
        assert!(!ptx.contains("fma") && !ptx.contains("div.rn"), "{ptx}");
        assert!(!ptx.contains("%ntid"), "the element is bbtab[b] + tid: {ptx}");
    }

    /// The keep-mp tail drops the zero store and nothing else: for every
    /// storage and first moment, the kernel is its zeroing twin's PTX with
    /// one `mov.f32 0f00000000` + its address arithmetic + one f32 store
    /// fewer, under a distinct `_keepmp` name; the zeroing kernels' names and
    /// PTX are unchanged.
    #[test]
    fn the_keep_tail_drops_only_the_zero_store() {
        let mut names = std::collections::HashSet::new();
        for theta in std::iter::once(Theta::F32).chain(Theta::BF16) {
            for m_bf16 in [false, true] {
                let zero =
                    String::from_utf8(multi_kernel_ptx(theta, m_bf16, MpTail::Zero)).unwrap();
                assert_eq!(zero.as_bytes(), param_multi_ptx(theta, m_bf16).as_slice());
                assert_eq!(
                    multi_kernel_name(theta, m_bf16, MpTail::Zero),
                    multi_name(theta, m_bf16)
                );
                let ir = build_multi_kernel(theta, m_bf16, MpTail::Keep);
                crate::kir_verify::verify(&ir)
                    .unwrap_or_else(|e| panic!("{theta:?}/{m_bf16}: {e:?}"));
                let name = multi_kernel_name(theta, m_bf16, MpTail::Keep);
                assert_eq!(ir.name, name);
                assert!(
                    name.ends_with("_keepmp") && names.insert(name.clone()),
                    "{name}"
                );
                let keep =
                    String::from_utf8(multi_kernel_ptx(theta, m_bf16, MpTail::Keep)).unwrap();
                assert!(keep.contains(&format!(".visible .entry {name}(")), "{keep}");
                assert_eq!(
                    keep.matches("st.global.f32 ").count() + 1,
                    zero.matches("st.global.f32 ").count(),
                    "{theta:?}/{m_bf16}"
                );
                assert_eq!(
                    keep.matches("0f00000000").count() + 1,
                    zero.matches("0f00000000").count(),
                    "{theta:?}/{m_bf16}"
                );
                for op in [
                    "mul.rn.f32",
                    "add.rn.f32",
                    "div.approx.f32",
                    "sqrt.rn.f32",
                    "ld.global",
                ] {
                    assert_eq!(
                        keep.matches(op).count(),
                        zero.matches(op).count(),
                        "{theta:?}/{m_bf16} {op}"
                    );
                }
            }
        }
    }

    /// The f32 members of the family are today's kernels: the same names,
    /// and (pinned byte for byte by `tests/f32_ptx_golden.rs`) the same PTX.
    #[test]
    fn the_family_s_f32_members_are_today_s_kernels() {
        assert_eq!(multi_name(Theta::F32, false), FASE_ADAMW_MULTI_NAME);
        assert_eq!(multi_name(Theta::F32, true), FASE_ADAMW_MULTI_MBF16_NAME);
        assert_eq!(param_multi_ptx(Theta::F32, false), fase_adamw_multi_ptx());
        assert_eq!(
            param_multi_ptx(Theta::F32, true),
            fase_adamw_multi_mbf16_ptx()
        );
    }

    /// Every bf16-parameter variant verifies, has a distinct name, keeps the
    /// f32 kernel's arithmetic (the same rounding instructions, counted), the
    /// clip fold and the gradient zeroing, and stores θ as 16-bit words.
    #[test]
    fn every_bf16_parameter_kernel_keeps_the_f32_update_the_clip_fold_and_the_zeroing() {
        let f = String::from_utf8(fase_adamw_multi_ptx()).unwrap();
        let mut names = std::collections::HashSet::new();
        for theta in Theta::BF16 {
            for m_bf16 in [false, true] {
                let ir = build_param_multi(theta, m_bf16);
                crate::kir_verify::verify(&ir)
                    .unwrap_or_else(|e| panic!("{theta:?}/{m_bf16}: {e:?}"));
                assert!(names.insert(ir.name.clone()), "duplicate name {}", ir.name);
                let p = String::from_utf8(param_multi_ptx(theta, m_bf16)).unwrap();
                assert!(
                    p.contains(&format!(".visible .entry {}(", multi_name(theta, m_bf16))),
                    "{p}"
                );
                for op in ["mul.rn.f32", "add.rn.f32", "div.approx.f32", "sqrt.rn.f32"] {
                    assert_eq!(
                        p.matches(op).count(),
                        f.matches(op).count(),
                        "{theta:?}/{m_bf16} {op}: {p}"
                    );
                }
                assert!(
                    !p.contains("fma") && !p.contains("div.rn") && !p.contains("%ntid"),
                    "{p}"
                );
                // The gradient is zeroed: one f32 store of zero after θ'.
                assert!(p.contains("mp_scale"), "{p}");
                let st16 =
                    p.matches("st.global.u16 ").count() + p.matches("st.global.b16 ").count();
                let want16 = match theta {
                    Theta::Bf16Split => 2,
                    _ => 1,
                } + usize::from(m_bf16);
                assert_eq!(st16, want16, "{theta:?}/{m_bf16}: 16-bit stores: {p}");
                let st32 = p.matches("st.global.f32 ").count();
                // v and the zeroed mp, m when f32, and the f32 master in master mode.
                let want32 =
                    2 + usize::from(!m_bf16) + usize::from(matches!(theta, Theta::Bf16Master(_)));
                assert_eq!(st32, want32, "{theta:?}/{m_bf16}: f32 stores: {p}");
            }
        }
        let sr = String::from_utf8(param_multi_ptx(Theta::Bf16Sr, false)).unwrap();
        for c in [SR_SPLITMIX_GAMMA, SR_SPLITMIX_M1, SR_SPLITMIX_M2] {
            assert_eq!(sr.matches(&format!(", {c};")).count(), 1, "{c:#x}");
        }
        let master =
            String::from_utf8(param_multi_ptx(Theta::Bf16Master(Image::Rne), false)).unwrap();
        assert_eq!(
            master.matches("cvt.rn.bf16.f32").count(),
            1,
            "the image is RNE of the master: {master}"
        );
        for theta in [
            Theta::Bf16Split,
            Theta::Bf16Master(Image::HalfTowardTruncation),
        ] {
            let p = String::from_utf8(param_multi_ptx(theta, false)).unwrap();
            assert!(
                !p.contains("cvt.rn.bf16.f32") && p.contains("selp.b32"),
                "{theta:?}: {p}"
            );
        }
    }

    /// The host split is exact: `join(split(x)) == x` for every non-NaN f32
    /// bit pattern, the image equals round-to-nearest-even except at a tie
    /// with an odd truncation, and a NaN stays a NaN. Every high half
    /// against the low halves that decide the carry, and every low half
    /// against high halves of every class (the exhaustive 2^32 sweep is
    /// `split_round_trips_every_bit_pattern`, ignored for its debug-build
    /// time).
    #[test]
    fn the_split_round_trips_and_differs_from_rne_only_at_odd_ties() {
        // Round to nearest even on the bit pattern (finite and infinite
        // values; NaN returns before it is used): the textbook form, not the
        // split's, so a slip in either shows.
        let rne =
            |bits: u32| -> u16 { (bits.wrapping_add(0x7FFF + ((bits >> 16) & 1)) >> 16) as u16 };
        let mut check = |bits: u32| {
            let (hi, lo) = (split_hi(bits), split_lo(bits));
            let x = f32::from_bits(bits);
            if x.is_nan() {
                assert!(
                    f32::from_bits(join_f32(hi, lo)).is_nan()
                        && f32::from_bits((hi as u32) << 16).is_nan(),
                    "{bits:#010x}"
                );
                return;
            }
            assert_eq!(join_f32(hi, lo), bits, "round trip {bits:#010x}");
            let odd_tie = bits & 0xFFFF == 0x8000 && bits & 0x1_0000 != 0;
            assert_eq!(
                hi == rne(bits),
                !odd_tie,
                "{bits:#010x}: hi {hi:#06x} rne {:#06x}",
                rne(bits)
            );
            if odd_tie {
                assert_eq!(hi as u32, bits >> 16, "an odd tie keeps the truncation");
            }
        };
        let lows = [
            0u32, 1, 0x7FFF, 0x8000, 0x8001, 0xFFFE, 0xFFFF, 0x1234, 0xC000,
        ];
        for top in 0..=0xFFFFu32 {
            for &l in &lows {
                check((top << 16) | l);
            }
        }
        let tops = [
            0u32, 1, 0x8000, 0x8001, 0x3F80, 0xBF81, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7F81,
            0xFFFF, 0x7FFF, 0x0080,
        ];
        for &t in &tops {
            for l in 0..=0xFFFFu32 {
                check((t << 16) | l);
            }
        }
    }

    #[test]
    #[ignore = "diagnostic: exhaustive 2^32 sweep of the split round trip (a minute in debug; run --ignored --release)"]
    fn split_round_trips_every_bit_pattern() {
        for bits in 0..=u32::MAX {
            let (hi, lo) = (split_hi(bits), split_lo(bits));
            if f32::from_bits(bits).is_nan() {
                assert!(f32::from_bits(join_f32(hi, lo)).is_nan());
            } else {
                assert_eq!(join_f32(hi, lo), bits, "{bits:#010x}");
            }
        }
    }

    #[test]
    fn the_split_and_join_kernels_verify() {
        for (ir, name) in [
            (build_param_split(), PARAM_SPLIT_NAME),
            (build_param_join(), PARAM_JOIN_NAME),
        ] {
            crate::kir_verify::verify(&ir).unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(ir.name, name);
        }
        let s = String::from_utf8(param_split_ptx()).unwrap();
        assert_eq!(s.matches("st.global.u16 ").count(), 2, "{s}");
        let j = String::from_utf8(param_join_ptx()).unwrap();
        assert_eq!(j.matches("ld.global.u16 ").count(), 2, "{j}");
    }
}
