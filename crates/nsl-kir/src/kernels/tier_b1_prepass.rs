// crates/nsl-kir/src/kernels/tier_b1_prepass.rs
//! The two CSHA Tier B.1 pre-pass kernels, as KIR (roadmap A2 step 11).
//!
//! Tier B.1's projection MMA reads its inputs in layouts the standard CSHA
//! pipeline does not produce, so `nsl_flash_attention_csha` runs these two
//! kernels first (`nsl_runtime::cuda::tier_b1_prepass` launches them and
//! documents the orchestration). They replace the runtime's hand-written
//! `CSHA_TIER_B1_PREPASS_X_PTX` and `CSHA_TIER_B1_PREPASS_W_PTX`.
//!
//! ## `csha_tier_b1_prepass_x`: RMSNorm, narrow, chunkify
//!
//! `(x_in, gamma, x_out, seq, d_model, chunk: u64, log2_chunk: u32, eps: f32)`,
//! grid `(seq, 1, 1)`, block `(256, 1, 1)`: one CTA per row of the f32
//! `[seq, d_model]` input. Each thread sums the squares of its columns
//! (`d = tid, tid + 256, ...`) into `sdata[tid]`; thread 0 adds the other
//! threads' partials to its own, in thread order, and publishes
//! `rsqrt(sum / d_model + eps)`; then every thread writes
//! `f16(x * rms_inv * gamma)` to the chunks-major `[d_model/chunk, seq,
//! chunk]` output at `(d >> log2_chunk, row, d & (chunk - 1))`.
//!
//! ## `csha_tier_b1_prepass_w`: narrow, col-major chunkify
//!
//! `(w_in, w_out, d_model, hd, chunk: u64, log2_hd, log2_chunk: u32)`, grid
//! `(ceil(d_model * hd / 256), 1, 1)`, block `(256, 1, 1)`: thread `gid`
//! reads the f32 `w_in[d_row, n]` (`d_row = gid >> log2_hd`, `n = gid &
//! (hd - 1)`) and writes it as f16 to `[d_model/chunk, hd, chunk]` at
//! `(d_row >> log2_chunk, n, d_row & (chunk - 1))`.
//!
//! ## One difference from the hand kernel
//!
//! The hand X kernel took the mean square with `div.approx.f32`; KIR's f32
//! division is the IEEE `div.rn.f32` (roadmap A2 step 3), so the mean is
//! now correctly rounded where it was within 2 ulp. It feeds
//! `rsqrt.approx.f32`, whose own error dominates; nothing else changes.

use crate::backend_ptx::lower_kir_to_ptx;
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp, KirTerminator,
    KirType, SmemLayout, SmemRegion, VarId,
};

/// The block both kernels are launched with. The X kernel strides its row
/// by this constant and sizes its reduction buffer to it.
pub const PREPASS_BLOCK: u32 = 256;

/// Which pre-pass a kernel is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prepass {
    /// RMSNorm + narrow + chunkify of the activations, per step.
    X,
    /// Narrow + col-major chunkify of a projection weight, once per weight.
    W,
}

impl Prepass {
    pub const ALL: [Prepass; 2] = [Prepass::X, Prepass::W];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self) -> &'static str {
        match self {
            Prepass::X => "csha_tier_b1_prepass_x",
            Prepass::W => "csha_tier_b1_prepass_w",
        }
    }
}

fn var(b: &mut KirBuilder, ty: KirType) -> VarId {
    b.new_typed_var(ty)
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn op2(b: &mut KirBuilder, ty: KirType, op: fn(VarId, VarId, VarId) -> KirOp, x: VarId, y: VarId) -> VarId {
    let dst = var(b, ty);
    b.emit(op(dst, x, y));
    dst
}

fn cast(b: &mut KirBuilder, src: VarId, ty: KirType) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Cast(dst, src, ty));
    dst
}

fn ptr(ty: KirType, space: AddressSpace) -> KirType {
    KirType::Ptr(Box::new(ty), space)
}

fn at(b: &mut KirBuilder, elem: KirType, space: AddressSpace, base: VarId, index: VarId) -> VarId {
    let dst = var(b, ptr(elem, space));
    b.emit(KirOp::PtrOffset(dst, base, index));
    dst
}

fn load(b: &mut KirBuilder, ty: KirType, addr: VarId, space: AddressSpace) -> VarId {
    let dst = var(b, ty);
    b.emit(KirOp::Load(dst, addr, space));
    dst
}

fn cmp(b: &mut KirBuilder, x: VarId, y: VarId, op: CmpOp) -> VarId {
    let dst = var(b, KirType::Bool);
    b.emit(KirOp::Cmp(dst, x, y, op));
    dst
}

/// Build `kind` as KIR.
pub fn build(kind: Prepass) -> KernelIR {
    match kind {
        Prepass::X => build_x(),
        Prepass::W => build_w(),
    }
}

/// ```text
/// entry:                 row = blockIdx.x; if row >= seq { exit }
/// setup:                 row_base = x_in + row*d_model; br sq(tid, 0)
/// sq(d, acc):            if d >= d_model { br sq_done(acc) }
/// sq_body:               x = row_base[d]; br sq(d + 256, fma(x, x, acc))
/// sq_done(acc):          sdata[tid] = acc; bar; if tid != 0 { br skip }
/// reduce_init:           br red(1, acc)
/// red(i, acc):           if i >= blockDim.x { br red_done(acc) }
/// red_body:              br red(i + 1, acc + sdata[i])
/// red_done(acc):         sdata[0] = rsqrt(acc / d_model + eps); br skip
/// skip:                  bar; rms = sdata[0]; br norm(tid)
/// norm(d):               if d >= d_model { exit }
/// norm_body:             x_out[(d >> log2_chunk)*seq*chunk + row*chunk + (d & (chunk-1))]
///                          = f16(row_base[d] * rms * gamma[d]); br norm(d + 256)
/// ```
fn build_x() -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{Bool, F16, F32, U32, U64};
    let mut b = KirBuilder::new(Prepass::X.kernel_name());

    let x_in = b.add_param("x_in", ptr(F32, Global), Global);
    let gamma = b.add_param("gamma", ptr(F32, Global), Global);
    let x_out = b.add_param("x_out", ptr(F16, Global), Global);
    let seq = b.add_param("seq", U64, Global);
    let d_model = b.add_param("d_model", U64, Global);
    let chunk = b.add_param("chunk", U64, Global);
    let log2_chunk = b.add_param("log2_chunk", U32, Global);
    let eps = b.add_param("eps", F32, Global);

    b.set_smem_layout(SmemLayout {
        regions: vec![SmemRegion { name: "sdata".to_string(), bytes: PREPASS_BLOCK * 4, align: 4, elem: F32 }],
        dynamic: false,
    });

    let entry = b.new_block();
    let setup = b.new_block();
    let sq = b.new_block();
    let sq_body = b.new_block();
    let sq_done = b.new_block();
    let reduce_init = b.new_block();
    let red = b.new_block();
    let red_body = b.new_block();
    let red_done = b.new_block();
    let skip = b.new_block();
    let norm = b.new_block();
    let norm_body = b.new_block();
    let exit = b.new_block();

    let sq_d = b.add_block_param(sq, U64);
    let sq_acc = b.add_block_param(sq, F32);
    let done_acc = b.add_block_param(sq_done, F32);
    let red_i = b.add_block_param(red, U32);
    let red_acc = b.add_block_param(red, F32);
    let final_acc = b.add_block_param(red_done, F32);
    let norm_d = b.add_block_param(norm, U64);
    let _ = Bool;

    // ── entry: one CTA per row ───────────────────────────────────────
    b.set_block(entry);
    let row32 = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(row32, 0));
    let row = cast(&mut b, row32, U64);
    let past = cmp(&mut b, row, seq, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(setup)));

    // ── setup: the thread's first column and the row's base ──────────
    b.set_block(setup);
    let tid32 = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid32, 0));
    let tid = cast(&mut b, tid32, U64);
    let row_off = op2(&mut b, U64, KirOp::Mul, row, d_model);
    let row_base = at(&mut b, F32, Global, x_in, row_off);
    let stride = konst(&mut b, U64, ConstValue::U64(PREPASS_BLOCK as u64));
    let f_zero = konst(&mut b, F32, ConstValue::F32(0.0));
    b.terminate(KirTerminator::Branch(KirEdge::with(sq, vec![tid, f_zero])));

    // ── pass 1: this thread's sum of squares ─────────────────────────
    b.set_block(sq);
    let sq_end = cmp(&mut b, sq_d, d_model, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(sq_end, KirEdge::with(sq_done, vec![sq_acc]), KirEdge::to(sq_body)));

    b.set_block(sq_body);
    let xa = at(&mut b, F32, Global, row_base, sq_d);
    let xv = load(&mut b, F32, xa, Global);
    let acc_next = var(&mut b, F32);
    b.emit(KirOp::Fma(acc_next, xv, xv, sq_acc));
    let d_next = op2(&mut b, U64, KirOp::Add, sq_d, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(sq, vec![d_next, acc_next])));

    // ── publish the partial; thread 0 reduces ────────────────────────
    b.set_block(sq_done);
    let sdata = var(&mut b, ptr(F32, Shared));
    b.emit(KirOp::SharedRegion { dst: sdata, region: 0 });
    let mine = at(&mut b, F32, Shared, sdata, tid32);
    b.emit(KirOp::Store(mine, done_acc, Shared));
    b.emit(KirOp::Barrier);
    let u_zero = konst(&mut b, U32, ConstValue::U32(0));
    let not_first = cmp(&mut b, tid32, u_zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_first, KirEdge::to(skip), KirEdge::to(reduce_init)));

    b.set_block(reduce_init);
    let u_one = konst(&mut b, U32, ConstValue::U32(1));
    let ntid = var(&mut b, U32);
    b.emit(KirOp::BlockDim(ntid, 0));
    b.terminate(KirTerminator::Branch(KirEdge::with(red, vec![u_one, done_acc])));

    b.set_block(red);
    let red_end = cmp(&mut b, red_i, ntid, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(red_end, KirEdge::with(red_done, vec![red_acc]), KirEdge::to(red_body)));

    b.set_block(red_body);
    let theirs = at(&mut b, F32, Shared, sdata, red_i);
    let part = load(&mut b, F32, theirs, Shared);
    let red_next = op2(&mut b, F32, KirOp::Add, red_acc, part);
    let i_next = op2(&mut b, U32, KirOp::Add, red_i, u_one);
    b.terminate(KirTerminator::Branch(KirEdge::with(red, vec![i_next, red_next])));

    b.set_block(red_done);
    let n = cast(&mut b, d_model, F32);
    let mean = op2(&mut b, F32, KirOp::Div, final_acc, n);
    let shifted = op2(&mut b, F32, KirOp::Add, mean, eps);
    let rms_inv = var(&mut b, F32);
    b.emit(KirOp::Rsqrt(rms_inv, shifted));
    b.emit(KirOp::Store(sdata, rms_inv, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(skip)));

    // ── pass 2: normalise, scale, narrow, chunkify ───────────────────
    b.set_block(skip);
    b.emit(KirOp::Barrier);
    let rms = load(&mut b, F32, sdata, Shared);
    let band = op2(&mut b, U64, KirOp::Mul, seq, chunk);
    let row_in_band = op2(&mut b, U64, KirOp::Mul, row, chunk);
    let one64 = konst(&mut b, U64, ConstValue::U64(1));
    let mask = op2(&mut b, U64, KirOp::Sub, chunk, one64);
    b.terminate(KirTerminator::Branch(KirEdge::with(norm, vec![tid])));

    b.set_block(norm);
    let norm_end = cmp(&mut b, norm_d, d_model, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(norm_end, KirEdge::to(exit), KirEdge::to(norm_body)));

    b.set_block(norm_body);
    let xa = at(&mut b, F32, Global, row_base, norm_d);
    let xv = load(&mut b, F32, xa, Global);
    let ga = at(&mut b, F32, Global, gamma, norm_d);
    let gv = load(&mut b, F32, ga, Global);
    let scaled = op2(&mut b, F32, KirOp::Mul, xv, rms);
    let scaled = op2(&mut b, F32, KirOp::Mul, scaled, gv);
    let h = cast(&mut b, scaled, F16);
    let chunk_idx = op2(&mut b, U64, KirOp::Shr, norm_d, log2_chunk);
    let c = op2(&mut b, U64, KirOp::And, norm_d, mask);
    let band_off = op2(&mut b, U64, KirOp::Mul, chunk_idx, band);
    let idx = op2(&mut b, U64, KirOp::Add, band_off, row_in_band);
    let idx = op2(&mut b, U64, KirOp::Add, idx, c);
    let out = at(&mut b, F16, Global, x_out, idx);
    b.emit(KirOp::Store(out, h, Global));
    let d_next = op2(&mut b, U64, KirOp::Add, norm_d, stride);
    b.terminate(KirTerminator::Branch(KirEdge::with(norm, vec![d_next])));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);

    b.set_workgroup_size([PREPASS_BLOCK, 1, 1]);
    b.set_launch_bounds(PREPASS_BLOCK, None);
    b.finalize()
}

/// ```text
/// entry:   gid = blockIdx.x*blockDim.x + threadIdx.x; if gid >= d_model*hd { exit }
/// body:    d_row = gid >> log2_hd; n = gid & (hd - 1)
///          w_out[(d_row >> log2_chunk)*hd*chunk + n*chunk + (d_row & (chunk-1))] = f16(w_in[gid])
/// exit:    ret
/// ```
fn build_w() -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F16, F32, U32, U64};
    let mut b = KirBuilder::new(Prepass::W.kernel_name());

    let w_in = b.add_param("w_in", ptr(F32, Global), Global);
    let w_out = b.add_param("w_out", ptr(F16, Global), Global);
    let d_model = b.add_param("d_model", U64, Global);
    let hd = b.add_param("hd", U64, Global);
    let chunk = b.add_param("chunk", U64, Global);
    let log2_hd = b.add_param("log2_hd", U32, Global);
    let log2_chunk = b.add_param("log2_chunk", U32, Global);

    let entry = b.new_block();
    let body = b.new_block();
    let exit = b.new_block();

    b.set_block(entry);
    let gid32 = var(&mut b, U32);
    b.emit(KirOp::GlobalId(gid32, 0));
    let gid = cast(&mut b, gid32, U64);
    let total = op2(&mut b, U64, KirOp::Mul, d_model, hd);
    let past = cmp(&mut b, gid, total, CmpOp::Ge);
    b.terminate(KirTerminator::CondBranch(past, KirEdge::to(exit), KirEdge::to(body)));

    b.set_block(body);
    let one = konst(&mut b, U64, ConstValue::U64(1));
    let d_row = op2(&mut b, U64, KirOp::Shr, gid, log2_hd);
    let hd_mask = op2(&mut b, U64, KirOp::Sub, hd, one);
    let n = op2(&mut b, U64, KirOp::And, gid, hd_mask);
    let src = at(&mut b, F32, Global, w_in, gid);
    let v = load(&mut b, F32, src, Global);
    let h = cast(&mut b, v, F16);
    let chunk_idx = op2(&mut b, U64, KirOp::Shr, d_row, log2_chunk);
    let chunk_mask = op2(&mut b, U64, KirOp::Sub, chunk, one);
    let k = op2(&mut b, U64, KirOp::And, d_row, chunk_mask);
    let band = op2(&mut b, U64, KirOp::Mul, hd, chunk);
    let idx = op2(&mut b, U64, KirOp::Mul, chunk_idx, band);
    let col = op2(&mut b, U64, KirOp::Mul, n, chunk);
    let idx = op2(&mut b, U64, KirOp::Add, idx, col);
    let idx = op2(&mut b, U64, KirOp::Add, idx, k);
    let dst = at(&mut b, F16, Global, w_out, idx);
    b.emit(KirOp::Store(dst, h, Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);

    b.set_workgroup_size([PREPASS_BLOCK, 1, 1]);
    b.set_launch_bounds(PREPASS_BLOCK, None);
    b.finalize()
}

/// Build `kind` and lower it to a NUL-terminated PTX module.
///
/// # Panics
///
/// If the built kernel fails verification: a bug in this module, since the
/// kernels take no input.
pub fn ptx(kind: Prepass) -> Vec<u8> {
    let ir = build(kind);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("pre-pass kernel `{}` failed KIR verification: {errors:?}", kind.kernel_name());
    }
    lower_kir_to_ptx(&ir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kir_verify::verify;

    #[test]
    fn both_kernels_verify() {
        for kind in Prepass::ALL {
            if let Err(errors) = verify(&build(kind)) {
                panic!("{kind:?} failed verification: {errors:?}");
            }
        }
    }

    /// The runtime marshals the arguments positionally; the 32-bit shift
    /// amounts and `eps` must stay 32-bit.
    #[test]
    fn the_parameter_lists_are_pinned() {
        let x = build(Prepass::X);
        let names: Vec<&str> = x.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["x_in", "gamma", "x_out", "seq", "d_model", "chunk", "log2_chunk", "eps"]);
        assert_eq!(x.params[6].ty, KirType::U32);
        assert_eq!(x.params[7].ty, KirType::F32);
        let w = build(Prepass::W);
        let names: Vec<&str> = w.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["w_in", "w_out", "d_model", "hd", "chunk", "log2_hd", "log2_chunk"]);
        assert_eq!(w.params[5].ty, KirType::U32);
        assert_eq!(w.params[6].ty, KirType::U32);
    }

    #[test]
    fn the_entry_names_are_pinned() {
        assert_eq!(Prepass::X.kernel_name(), "csha_tier_b1_prepass_x");
        assert_eq!(Prepass::W.kernel_name(), "csha_tier_b1_prepass_w");
        for kind in Prepass::ALL {
            assert_eq!(build(kind).name, kind.kernel_name());
        }
    }

    /// NUL-terminated ASCII at the ISA floor, narrowing to f16 with a 16-bit
    /// store, and the chunk index taken by shift and mask rather than
    /// division.
    #[test]
    fn the_modules_are_nul_terminated_ascii_with_the_expected_forms() {
        for kind in Prepass::ALL {
            let bytes = ptx(kind);
            assert_eq!(bytes.last(), Some(&0), "{kind:?}");
            assert!(!bytes[..bytes.len() - 1].contains(&0), "{kind:?}");
            assert!(bytes.is_ascii(), "{kind:?}");
            let text = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
            assert!(text.starts_with(".version 7.0\n"), "{kind:?}");
            assert!(text.contains("cvt.rn.f16.f32 "), "{kind:?}");
            assert!(text.contains("st.global.b16 "), "{kind:?}");
            assert!(text.contains("shr.u64 ") && text.contains("and.b64 "), "{kind:?}");
            assert!(!text.contains("div.u64") && !text.contains("rem.u64"), "{kind:?}");
        }
    }
}
