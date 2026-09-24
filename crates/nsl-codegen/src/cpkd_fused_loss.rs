//! CPKD Innovation 2 — fused KL-CE distillation loss kernel synthesis.
//!
//! Computes the standard knowledge-distillation loss
//!
//! ```text
//! loss = alpha * CE(student_logits, labels)
//!      + (1 - alpha) * T^2 * KL(softmax(teacher/T) || softmax(student/T))
//! ```
//!
//! WITHOUT materializing either `[rows, vocab]` logit tensor in HBM. One
//! CTA per token row streams the vocabulary in SMEM tiles; each tile holds
//! BOTH the student and the teacher logits for that vocab slice, and thread
//! 0 maintains three online-softmax accumulator families:
//!
//! - `(m_s1, S_s1)` — student logits at temperature 1 (for the CE term);
//! - `(m_sT, S_sT)` — student logits / T (for `LSE_T(s)`);
//! - `(m_tT, S_tT, C)` — teacher logits / T plus the KL cross-term
//!   `C = sum_v exp(t_v/T - m_tT) * (t_v - s_v)`, rescaled alongside `S_tT`
//!   whenever the running max advances (same rescale trick as the sum).
//!
//! Per valid row the kernel emits:
//!
//! ```text
//! CE  = lse_s1 - s_target                     (lse = m + ln S)
//! KL  = (C / S_tT) / T - lse_tT + lse_sT
//! loss_row = alpha * CE + (1 - alpha) * T^2 * KL
//! ```
//!
//! and saves the three per-row LSEs (`lse_s1`, `lse_sT`, `lse_tT`) to HBM —
//! `3 * rows * 4` bytes, the only forward state the backward needs.
//!
//! The backward re-computes both logits per (row, vocab) pair — mirroring
//! the fused linear-CE backward's "recompute, never save" design — and
//! forms
//!
//! ```text
//! dloss/ds_v = alpha * (p_s1(v) - 1{v==target})
//!            + (1 - alpha) * T * (p_sT(v) - p_tT(v))
//! ```
//!
//! scattering `dx_s`, `dW_s`, `dbias_s` via `red.global.add.f32`.
//! **The teacher receives no gradients by construction** — there are no
//! teacher-gradient outputs in the kernel ABI at all (composition-paper
//! invariant I-11 enforced at the ABI level, not by a runtime check).
//!
//! ## v1 scope (deferrals refuse loudly in `validate`)
//!
//! - f32 only (the fp16/bf16 mixed-precision path of `fused_linear_ce.rs`
//!   is a mechanical extension, deferred);
//! - `vocab_size <= 8192` (single-CTA path; the two-kernel large-vocab
//!   split is deferred);
//! - spectral top-k logit compression is ADVISORY in v1 (`cpkd_spectral`
//!   reports the effective rank; this kernel always runs the full vocab).
//!
//! Both kernels are built as KIR ([`build_forward`], [`build_backward`];
//! roadmap A2 step 10) and lowered to null-terminated module bytes
//! (`cuModuleLoadData` contract) at the KIR floor, `.version 7.0` /
//! `.target sm_70`, so `gpu_sm` no longer reaches them.
//! `tests/cpkd_fused_loss_kir_equivalence.rs` runs the pre-migration hand
//! emitters (`tests/fixtures/cpkd_fused_loss_hand.rs`) against them on the
//! PTX interpreter and requires the same output bits.

use serde::Serialize;

use crate::backend_ptx::lower_kir_to_ptx;
use crate::cfie_decode_attention::{at, cmp, konst, load, op2, ptr, widen};
use crate::fused_linear_ce::{bottom_tested_loop, i64_const, V1_BLOCK};
use crate::kernel_ir::{
    AddressSpace, CmpOp, ConstValue, KernelIR, KirBuilder, KirEdge, KirOp, KirTerminator, KirType, SmemLayout,
    SmemRegion, VarId,
};

/// Hard ceiling for the v1 single-CTA path (mirrors
/// `fused_linear_ce::LARGE_VOCAB_THRESHOLD`; the large-vocab two-kernel
/// variant is a deferred extension).
pub const KL_CE_MAX_VOCAB_V1: u32 = 8192;

/// Compile-time configuration for one fused KL-CE kernel pair.
///
/// Shapes are decorator-supplied compile-time constants (same static-shape
/// contract as `@fused_lm_ce`); alpha/temperature are RUNTIME kernel
/// parameters, so one PTX pair serves any (alpha, T) without respecialization.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FusedKlCeConfig {
    pub vocab_size: u32,
    /// Student hidden dim (must be % 32 == 0).
    pub student_hidden: u32,
    /// Teacher hidden dim (must be % 32 == 0; may differ from the student's).
    pub teacher_hidden: u32,
    pub batch_size: u32,
    pub seq_len: u32,
    /// Vocab tile held in SMEM (% 128 == 0, <= vocab_size).
    pub vocab_tile: u32,
    pub gpu_sm: u32,
    pub ignore_index: i64,
}

impl Default for FusedKlCeConfig {
    fn default() -> Self {
        FusedKlCeConfig {
            vocab_size: 0,
            student_hidden: 0,
            teacher_hidden: 0,
            batch_size: 0,
            seq_len: 0,
            vocab_tile: 1024,
            gpu_sm: 80,
            ignore_index: -100,
        }
    }
}

impl FusedKlCeConfig {
    /// Validate the v1 preconditions. Every violated precondition names the
    /// deferral it corresponds to (repo doctrine: refuse loudly).
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 || self.student_hidden == 0 || self.teacher_hidden == 0 {
            return Err("fused_kl_ce: vocab_size/student_hidden/teacher_hidden must be non-zero".into());
        }
        if self.vocab_size > KL_CE_MAX_VOCAB_V1 {
            return Err(format!(
                "fused_kl_ce v1 supports vocab_size <= {} (single-CTA path); got {}. \
                 The large-vocab two-kernel variant is a deferred extension",
                KL_CE_MAX_VOCAB_V1, self.vocab_size
            ));
        }
        if !self.student_hidden.is_multiple_of(32) {
            return Err(format!(
                "fused_kl_ce: student_hidden must be a multiple of 32; got {}",
                self.student_hidden
            ));
        }
        if !self.teacher_hidden.is_multiple_of(32) {
            return Err(format!(
                "fused_kl_ce: teacher_hidden must be a multiple of 32; got {}",
                self.teacher_hidden
            ));
        }
        if self.vocab_tile == 0 || !self.vocab_tile.is_multiple_of(128) {
            return Err(format!(
                "fused_kl_ce: vocab_tile must be a non-zero multiple of 128; got {}",
                self.vocab_tile
            ));
        }
        if self.vocab_tile > self.vocab_size.next_multiple_of(128) {
            return Err(format!(
                "fused_kl_ce: vocab_tile ({}) exceeds padded vocab_size ({})",
                self.vocab_tile,
                self.vocab_size.next_multiple_of(128)
            ));
        }
        if self.batch_size == 0 || self.seq_len == 0 {
            return Err("fused_kl_ce: batch_size and seq_len must be non-zero".into());
        }
        Ok(())
    }

    pub fn rows(&self) -> u32 {
        self.batch_size * self.seq_len
    }

    pub fn num_vocab_tiles(&self) -> u32 {
        self.vocab_size.div_ceil(self.vocab_tile)
    }

    /// SMEM layout: `[0 .. vtile*4)` student logit tile, `[vtile*4 ..
    /// 2*vtile*4)` teacher logit tile, one f32 scratch slot for the
    /// student's logit-at-target, padded to 32 bytes.
    pub fn shared_mem_bytes(&self) -> u32 {
        self.vocab_tile * 4 * 2 + 32
    }

    pub fn kernel_name(&self) -> String {
        format!(
            "nsl_fused_kl_ce_f32_v{}_hs{}_ht{}",
            self.vocab_size, self.student_hidden, self.teacher_hidden
        )
    }

    pub fn bwd_kernel_name(&self) -> String {
        format!(
            "nsl_fused_kl_ce_backward_f32_v{}_hs{}_ht{}",
            self.vocab_size, self.student_hidden, self.teacher_hidden
        )
    }
}

/// Synthesise the forward PTX: [`build_forward`], verified and lowered.
/// Returned bytes are null-terminated (`cuModuleLoadData` reads a C string;
/// see `fused_linear_ce.rs`).
pub fn synthesize_fused_kl_ce_ptx(cfg: &FusedKlCeConfig) -> Vec<u8> {
    lower_verified(build_forward(cfg))
}

/// Synthesise the backward PTX: [`build_backward`], verified and lowered
/// (null-terminated, same contract).
pub fn synthesize_fused_kl_ce_backward_ptx(cfg: &FusedKlCeConfig) -> Vec<u8> {
    lower_verified(build_backward(cfg))
}

/// # Panics
///
/// If the kernel fails KIR verification — a bug in this module, not a
/// condition a caller can provoke.
fn lower_verified(ir: KernelIR) -> Vec<u8> {
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!("{} failed KIR verification: {errors:?}", ir.name);
    }
    lower_kir_to_ptx(&ir)
}

// ─── KIR (roadmap A2 step 10) ───────────────────────────────────────────────
//
// Both kernels are built as KIR. They keep the hand kernels' control flow,
// barriers, loop shapes (every loop tested at the bottom, as the hand loops
// were, but the scans' vocab guard and the ignored row's zeroing, tested at
// the top) and floating-point order; `tests/cpkd_fused_loss_kir_equivalence.rs`
// runs the frozen hand emitters (`tests/fixtures/cpkd_fused_loss_hand.rs`)
// against them on the PTX interpreter and requires the same output bits.

/// Index of each shared region in [`forward_smem`].
const R_STUDENT: u32 = 0;
const R_TEACHER: u32 = 1;
const R_TARGET: u32 = 2;

/// `[student logits: vtile][teacher logits: vtile][student logit at target]`,
/// f32, dynamic (the launcher passes [`FusedKlCeConfig::shared_mem_bytes`],
/// which covers it) — the hand kernel's offsets.
fn forward_smem(cfg: &FusedKlCeConfig) -> SmemLayout {
    let region = |name: &str, elems: u32| SmemRegion { name: name.to_string(), bytes: elems * 4, align: 4, elem: KirType::F32 };
    SmemLayout {
        regions: vec![region("student", cfg.vocab_tile), region("teacher", cfg.vocab_tile), region("logit_at_target", 1)],
        dynamic: true,
    }
}

/// `sum_h fma(a[a_row + h], b[b_row + h], acc)` over `h < hidden`, from
/// zero, in index order, tested at the bottom (`hidden >= 32`).
fn dot(b: &mut KirBuilder, (a, a_row): (VarId, VarId), (m, m_row): (VarId, VarId), hidden: u32) -> VarId {
    use AddressSpace::Global;
    use KirType::{F32, U64};
    let zero = konst(b, ConstValue::U32(0));
    let one = konst(b, ConstValue::U32(1));
    let hidden = konst(b, ConstValue::U32(hidden));
    let f_zero = konst(b, ConstValue::F32(0.0));
    bottom_tested_loop(b, (zero, hidden, one), &[f_zero], |b, h, acc| {
        let h_wide = widen(b, h);
        let a_index = op2(b, U64, KirOp::Add, a_row, h_wide);
        let a_addr = at(b, F32, Global, a, a_index);
        let av = load(b, F32, a_addr, Global);
        let m_index = op2(b, U64, KirOp::Add, m_row, h_wide);
        let m_addr = at(b, F32, Global, m, m_index);
        let mv = load(b, F32, m_addr, Global);
        let next = b.new_typed_var(F32);
        b.emit(KirOp::Fma(next, av, mv, acc[0]));
        vec![next]
    })[0]
}

/// A scan over the tile's lanes that stops at the vocab: `i = 0; while
/// v_base + i < V { carried = body(i, carried); i += 1; if !(i < vtile)
/// break; }` — the hand kernel's shape. The carried values after the last
/// lane are returned.
fn tile_scan(
    b: &mut KirBuilder,
    (v_base, vocab, vtile): (VarId, VarId, VarId),
    init: &[VarId],
    body: impl FnOnce(&mut KirBuilder, VarId, &[VarId]) -> Vec<VarId>,
) -> Vec<VarId> {
    use KirType::{F32, U32};
    let zero = konst(b, ConstValue::U32(0));
    let one = konst(b, ConstValue::U32(1));
    let head = b.new_block();
    let lane = b.new_block();
    let done = b.new_block();
    let i = b.add_block_param(head, U32);
    let carried: Vec<VarId> = init.iter().map(|_| b.add_block_param(head, F32)).collect();
    let out: Vec<VarId> = init.iter().map(|_| b.add_block_param(done, F32)).collect();
    let mut entry = vec![zero];
    entry.extend_from_slice(init);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, entry)));

    b.set_block(head);
    let v = op2(b, U32, KirOp::Add, v_base, i);
    let in_vocab = cmp(b, v, vocab, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(in_vocab, KirEdge::to(lane), KirEdge::with(done, carried.clone())));

    b.set_block(lane);
    let next = body(b, i, &carried);
    let i_next = op2(b, U32, KirOp::Add, i, one);
    let more = cmp(b, i_next, vtile, CmpOp::Lt);
    let mut back = vec![i_next];
    back.extend_from_slice(&next);
    b.terminate(KirTerminator::CondBranch(more, KirEdge::with(head, back), KirEdge::with(done, next)));
    b.set_block(done);
    out
}

fn f32_exp(b: &mut KirBuilder, x: VarId) -> VarId {
    let e = b.new_typed_var(KirType::F32);
    b.emit(KirOp::Exp(e, x));
    e
}

/// Build the forward kernel as KIR.
///
/// One CTA per token row, 128 threads. The CFG, in the hand kernel's order:
///
/// ```text
/// entry     thread 0 stores -inf to the logit-at-target slot      bar
///           target = targets[row]; target == ignore_index ? skip : body
/// body      per tile (m_s1, S_s1, m_sT, S_sT, m_tT, S_tT, C):
///             per sub-tile: v = j*128 + tid + tile*vtile; v < V ?
///               s = fma-dot(xs[row], Ws[v]) + bs[v]
///               t = fma-dot(xt[row], Wt[v]) + bt[v]
///               student[slot] = s; teacher[slot] = t
///               v == target ? logit_at_target = s
///             bar
///             thread 0: max of each tile (lanes below V); rescale the
///               three families; accumulate S_s1, S_sT, S_tT and C
///             bar
///           thread 0: the three LSEs, CE, KL, the loss; four stores
/// skip      thread 0: four zeros
/// ```
pub fn build_forward(cfg: &FusedKlCeConfig) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, I64, U32, U64};

    let mut b = KirBuilder::new(&cfg.kernel_name());
    // The params, in FFI order; rows, V, HS and HT are baked.
    let fptr = || ptr(F32, Global);
    let xs = b.add_param("xs", fptr(), Global);
    let ws = b.add_param("ws", fptr(), Global);
    let bs = b.add_param("bs", fptr(), Global);
    let xt = b.add_param("xt", fptr(), Global);
    let wt = b.add_param("wt", fptr(), Global);
    let bt = b.add_param("bt", fptr(), Global);
    let targets = b.add_param("targets", ptr(I64, Global), Global);
    let loss_out = b.add_param("loss_out", fptr(), Global);
    let lse_s1_out = b.add_param("lse_s1_out", fptr(), Global);
    let lse_st_out = b.add_param("lse_st_out", fptr(), Global);
    let lse_tt_out = b.add_param("lse_tt_out", fptr(), Global);
    for name in ["rows", "V", "HS", "HT"] {
        b.add_param(name, U32, Global);
    }
    let alpha = b.add_param("alpha", F32, Global);
    let temp = b.add_param("temp", F32, Global);
    b.set_smem_layout(forward_smem(cfg));
    b.set_workgroup_size([V1_BLOCK, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let row = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(row, 0));
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let tinv = b.new_typed_var(F32);
    b.emit(KirOp::Rcp(tinv, temp));
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let region = |b: &mut KirBuilder, r: u32| {
        let dst = b.new_typed_var(ptr(F32, Shared));
        b.emit(KirOp::SharedRegion { dst, region: r });
        dst
    };
    let student = region(&mut b, R_STUDENT);
    let teacher = region(&mut b, R_TEACHER);
    let target_slot = region(&mut b, R_TARGET);
    let f_neg_inf = konst(&mut b, ConstValue::F32(f32::NEG_INFINITY));
    let f_zero = konst(&mut b, ConstValue::F32(0.0));

    // Thread 0 initialises the logit-at-target slot to -inf.
    let init = b.new_block();
    let init_done = b.new_block();
    let is_thread0 = cmp(&mut b, tid, zero, CmpOp::Eq);
    b.terminate(KirTerminator::CondBranch(is_thread0, KirEdge::to(init), KirEdge::to(init_done)));
    b.set_block(init);
    b.emit(KirOp::Store(target_slot, f_neg_inf, Shared));
    b.terminate(KirTerminator::Branch(KirEdge::to(init_done)));

    b.set_block(init_done);
    b.emit(KirOp::Barrier);
    let target_addr = at(&mut b, I64, Global, targets, row);
    let target = load(&mut b, I64, target_addr, Global);
    let ignore = i64_const(&mut b, cfg.ignore_index);
    let skips = cmp(&mut b, target, ignore, CmpOp::Eq);
    let body = b.new_block();
    let skip = b.new_block();
    let exit = b.new_block();
    b.terminate(KirTerminator::CondBranch(skips, KirEdge::to(skip), KirEdge::to(body)));

    // ── the vocab tile loop ─────────────────────────────────────────────
    b.set_block(body);
    let row_wide = widen(&mut b, row);
    let hs_wide = konst(&mut b, ConstValue::U64(cfg.student_hidden as u64));
    let ht_wide = konst(&mut b, ConstValue::U64(cfg.teacher_hidden as u64));
    let xs_row = op2(&mut b, U64, KirOp::Mul, row_wide, hs_wide);
    let xt_row = op2(&mut b, U64, KirOp::Mul, row_wide, ht_wide);
    let vocab = konst(&mut b, ConstValue::U32(cfg.vocab_size));
    let vtile = konst(&mut b, ConstValue::U32(cfg.vocab_tile));
    let block = konst(&mut b, ConstValue::U32(V1_BLOCK));
    let n_tiles = konst(&mut b, ConstValue::U32(cfg.num_vocab_tiles()));
    let per_thread = konst(&mut b, ConstValue::U32(cfg.vocab_tile / V1_BLOCK));
    let init_acc = [f_neg_inf, f_zero, f_neg_inf, f_zero, f_neg_inf, f_zero, f_zero];
    let acc = bottom_tested_loop(&mut b, (zero, n_tiles, one), &init_acc, |b, tile, acc| {
        let v_base = op2(b, U32, KirOp::Mul, tile, vtile);
        bottom_tested_loop(b, (zero, per_thread, one), &[], |b, j, _| {
            let lane_base = op2(b, U32, KirOp::Mul, j, block);
            let slot = op2(b, U32, KirOp::Add, lane_base, tid);
            let v = op2(b, U32, KirOp::Add, slot, v_base);
            let in_vocab = cmp(b, v, vocab, CmpOp::Lt);
            let fill = b.new_block();
            let fill_done = b.new_block();
            b.terminate(KirTerminator::CondBranch(in_vocab, KirEdge::to(fill), KirEdge::to(fill_done)));

            b.set_block(fill);
            let v_wide = widen(b, v);
            let ws_row = op2(b, U64, KirOp::Mul, v_wide, hs_wide);
            let s_dot = dot(b, (xs, xs_row), (ws, ws_row), cfg.student_hidden);
            let bs_addr = at(b, F32, Global, bs, v);
            let bs_v = load(b, F32, bs_addr, Global);
            let s = op2(b, F32, KirOp::Add, s_dot, bs_v);
            let wt_row = op2(b, U64, KirOp::Mul, v_wide, ht_wide);
            let t_dot = dot(b, (xt, xt_row), (wt, wt_row), cfg.teacher_hidden);
            let bt_addr = at(b, F32, Global, bt, v);
            let bt_v = load(b, F32, bt_addr, Global);
            let t = op2(b, F32, KirOp::Add, t_dot, bt_v);
            let s_addr = at(b, F32, Shared, student, slot);
            b.emit(KirOp::Store(s_addr, s, Shared));
            let t_addr = at(b, F32, Shared, teacher, slot);
            b.emit(KirOp::Store(t_addr, t, Shared));
            // Only the thread holding the target's column writes the slot.
            let v_signed = b.new_typed_var(I64);
            b.emit(KirOp::Cast(v_signed, v, I64));
            let is_target = cmp(b, v_signed, target, CmpOp::Eq);
            let record = b.new_block();
            b.terminate(KirTerminator::CondBranch(is_target, KirEdge::to(record), KirEdge::to(fill_done)));
            b.set_block(record);
            b.emit(KirOp::Store(target_slot, s, Shared));
            b.terminate(KirTerminator::Branch(KirEdge::to(fill_done)));

            b.set_block(fill_done);
            vec![]
        });
        // Every logit of the tile is in shared memory.
        b.emit(KirOp::Barrier);

        // Thread 0: the tile's maxima, the rescale, the accumulation.
        let reduce = b.new_block();
        let reduce_done = b.new_block();
        let next: Vec<VarId> = (0..acc.len()).map(|_| b.add_block_param(reduce_done, F32)).collect();
        let not_thread0 = cmp(b, tid, zero, CmpOp::Ne);
        b.terminate(KirTerminator::CondBranch(
            not_thread0,
            KirEdge::with(reduce_done, acc.to_vec()),
            KirEdge::to(reduce),
        ));

        b.set_block(reduce);
        let (m_s1, s_s1, m_st, s_st, m_tt, s_tt, cross) = (acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6]);
        let maxima = tile_scan(b, (v_base, vocab, vtile), &[f_neg_inf, f_neg_inf], |b, i, m| {
            let s_addr = at(b, F32, Shared, student, i);
            let s = load(b, F32, s_addr, Shared);
            let ms = op2(b, F32, KirOp::Max, m[0], s);
            let t_addr = at(b, F32, Shared, teacher, i);
            let t = load(b, F32, t_addr, Shared);
            let mt = op2(b, F32, KirOp::Max, m[1], t);
            vec![ms, mt]
        });
        let (tmax_s, tmax_t) = (maxima[0], maxima[1]);
        // Online rescale: student at T = 1.
        let m_s1n = op2(b, F32, KirOp::Max, m_s1, tmax_s);
        let d = op2(b, F32, KirOp::Sub, m_s1, m_s1n);
        let sc = f32_exp(b, d);
        let s_s1r = op2(b, F32, KirOp::Mul, s_s1, sc);
        // Student / T.
        let cand = op2(b, F32, KirOp::Mul, tmax_s, tinv);
        let m_stn = op2(b, F32, KirOp::Max, m_st, cand);
        let d = op2(b, F32, KirOp::Sub, m_st, m_stn);
        let sc = f32_exp(b, d);
        let s_str = op2(b, F32, KirOp::Mul, s_st, sc);
        // Teacher / T: the sum and the cross-term.
        let cand = op2(b, F32, KirOp::Mul, tmax_t, tinv);
        let m_ttn = op2(b, F32, KirOp::Max, m_tt, cand);
        let d = op2(b, F32, KirOp::Sub, m_tt, m_ttn);
        let sc = f32_exp(b, d);
        let s_ttr = op2(b, F32, KirOp::Mul, s_tt, sc);
        let crossr = op2(b, F32, KirOp::Mul, cross, sc);
        let sums = tile_scan(b, (v_base, vocab, vtile), &[s_s1r, s_str, s_ttr, crossr], |b, i, a| {
            let s_addr = at(b, F32, Shared, student, i);
            let s = load(b, F32, s_addr, Shared);
            let t_addr = at(b, F32, Shared, teacher, i);
            let t = load(b, F32, t_addr, Shared);
            let d = op2(b, F32, KirOp::Sub, s, m_s1n);
            let e = f32_exp(b, d);
            let s_s1 = op2(b, F32, KirOp::Add, a[0], e);
            let st = op2(b, F32, KirOp::Mul, s, tinv);
            let d = op2(b, F32, KirOp::Sub, st, m_stn);
            let e = f32_exp(b, d);
            let s_st = op2(b, F32, KirOp::Add, a[1], e);
            let tt = op2(b, F32, KirOp::Mul, t, tinv);
            let d = op2(b, F32, KirOp::Sub, tt, m_ttn);
            let e_t = f32_exp(b, d);
            let s_tt = op2(b, F32, KirOp::Add, a[2], e_t);
            let t_minus_s = op2(b, F32, KirOp::Sub, t, s);
            let cross = b.new_typed_var(F32);
            b.emit(KirOp::Fma(cross, e_t, t_minus_s, a[3]));
            vec![s_s1, s_st, s_tt, cross]
        });
        b.terminate(KirTerminator::Branch(KirEdge::with(
            reduce_done,
            vec![m_s1n, sums[0], m_stn, sums[1], m_ttn, sums[2], sums[3]],
        )));

        // The tiles are refilled next trip.
        b.set_block(reduce_done);
        b.emit(KirOp::Barrier);
        next
    });

    // ── thread 0 writes the loss and the three LSEs ─────────────────────
    let write = b.new_block();
    let not_writer = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_writer, KirEdge::to(exit), KirEdge::to(write)));
    b.set_block(write);
    let lse = |b: &mut KirBuilder, sum: VarId, max: VarId| {
        let l = b.new_typed_var(F32);
        b.emit(KirOp::Log(l, sum));
        op2(b, F32, KirOp::Add, l, max)
    };
    let lse_s1 = lse(&mut b, acc[1], acc[0]);
    let lse_st = lse(&mut b, acc[3], acc[2]);
    let lse_tt = lse(&mut b, acc[5], acc[4]);
    let logit_at_target = load(&mut b, F32, target_slot, Shared);
    let ce = op2(&mut b, F32, KirOp::Sub, lse_s1, logit_at_target);
    let kl = op2(&mut b, F32, KirOp::Div, acc[6], acc[5]);
    let kl = op2(&mut b, F32, KirOp::Mul, kl, tinv);
    let kl = op2(&mut b, F32, KirOp::Sub, kl, lse_tt);
    let kl = op2(&mut b, F32, KirOp::Add, kl, lse_st);
    let ce_part = op2(&mut b, F32, KirOp::Mul, alpha, ce);
    let f_one = konst(&mut b, ConstValue::F32(1.0));
    let one_minus_alpha = op2(&mut b, F32, KirOp::Sub, f_one, alpha);
    let t_squared = op2(&mut b, F32, KirOp::Mul, temp, temp);
    let kl_weight = op2(&mut b, F32, KirOp::Mul, one_minus_alpha, t_squared);
    let loss = b.new_typed_var(F32);
    b.emit(KirOp::Fma(loss, kl_weight, kl, ce_part));
    for (out, v) in [(loss_out, loss), (lse_s1_out, lse_s1), (lse_st_out, lse_st), (lse_tt_out, lse_tt)] {
        let addr = at(&mut b, F32, Global, out, row);
        b.emit(KirOp::Store(addr, v, Global));
    }
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    // ── an ignored row: thread 0 writes zeros ───────────────────────────
    b.set_block(skip);
    let zero_write = b.new_block();
    let not_zeroer = cmp(&mut b, tid, zero, CmpOp::Ne);
    b.terminate(KirTerminator::CondBranch(not_zeroer, KirEdge::to(exit), KirEdge::to(zero_write)));
    b.set_block(zero_write);
    for out in [loss_out, lse_s1_out, lse_st_out, lse_tt_out] {
        let addr = at(&mut b, F32, Global, out, row);
        b.emit(KirOp::Store(addr, f_zero, Global));
    }
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

/// Build the backward kernel as KIR.
///
/// One CTA per token row, 128 threads, no shared memory. Per (row, v) it
/// recomputes both logits and forms
///
/// ```text
/// dl = alpha*(p_s1 - [v == target]) + (1 - alpha)*T*(p_sT - p_tT)
/// ```
///
/// scaled by `grad / num_valid`, then scatters `dx_s`, `dW_s` and `dbias_s`
/// with `red.global.add.f32` (the caller zero-fills them). An ignored row
/// zeroes its `dx_s` row. There are no teacher-gradient outputs (I-11).
pub fn build_backward(cfg: &FusedKlCeConfig) -> KernelIR {
    use AddressSpace::Global;
    use KirType::{F32, I64, U32, U64};

    let mut b = KirBuilder::new(&cfg.bwd_kernel_name());
    let fptr = || ptr(F32, Global);
    let grad_output = b.add_param("grad_output", F32, Global);
    let xs = b.add_param("xs", fptr(), Global);
    let ws = b.add_param("ws", fptr(), Global);
    let bs = b.add_param("bs", fptr(), Global);
    let xt = b.add_param("xt", fptr(), Global);
    let wt = b.add_param("wt", fptr(), Global);
    let bt = b.add_param("bt", fptr(), Global);
    let targets = b.add_param("targets", ptr(I64, Global), Global);
    let lse_s1 = b.add_param("lse_s1", fptr(), Global);
    let lse_st = b.add_param("lse_st", fptr(), Global);
    let lse_tt = b.add_param("lse_tt", fptr(), Global);
    let dxs_out = b.add_param("dxs_out", fptr(), Global);
    let dws_out = b.add_param("dws_out", fptr(), Global);
    let dbs_out = b.add_param("dbs_out", fptr(), Global);
    for name in ["rows", "V", "HS", "HT"] {
        b.add_param(name, U32, Global);
    }
    let alpha = b.add_param("alpha", F32, Global);
    let temp = b.add_param("temp", F32, Global);
    let num_valid = b.add_param("num_valid", U32, Global);
    b.set_workgroup_size([V1_BLOCK, 1, 1]);

    let entry = b.new_block();
    b.set_block(entry);
    let num_valid_f = b.new_typed_var(F32);
    b.emit(KirOp::Cast(num_valid_f, num_valid, F32));
    let row = b.new_typed_var(U32);
    b.emit(KirOp::BlockIdx(row, 0));
    let tid = b.new_typed_var(U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let tinv = b.new_typed_var(F32);
    b.emit(KirOp::Rcp(tinv, temp));
    let zero = konst(&mut b, ConstValue::U32(0));
    let one = konst(&mut b, ConstValue::U32(1));
    let block = konst(&mut b, ConstValue::U32(V1_BLOCK));
    let hs = konst(&mut b, ConstValue::U32(cfg.student_hidden));
    let hs_wide = konst(&mut b, ConstValue::U64(cfg.student_hidden as u64));
    let row_wide = widen(&mut b, row);
    let xs_row = op2(&mut b, U64, KirOp::Mul, row_wide, hs_wide);
    let target_addr = at(&mut b, I64, Global, targets, row);
    let target = load(&mut b, I64, target_addr, Global);
    let ignore = i64_const(&mut b, cfg.ignore_index);
    let skips = cmp(&mut b, target, ignore, CmpOp::Eq);
    let live = b.new_block();
    let skip = b.new_block();
    let exit = b.new_block();
    b.terminate(KirTerminator::CondBranch(skips, KirEdge::to(skip), KirEdge::to(live)));

    // ── the live row ────────────────────────────────────────────────────
    b.set_block(live);
    let saved = |b: &mut KirBuilder, base: VarId| {
        let addr = at(b, F32, Global, base, row);
        load(b, F32, addr, Global)
    };
    let lse_s1_row = saved(&mut b, lse_s1);
    let lse_st_row = saved(&mut b, lse_st);
    let lse_tt_row = saved(&mut b, lse_tt);
    let ht_wide = konst(&mut b, ConstValue::U64(cfg.teacher_hidden as u64));
    let xt_row = op2(&mut b, U64, KirOp::Mul, row_wide, ht_wide);
    let scale = op2(&mut b, F32, KirOp::Div, grad_output, num_valid_f);
    let f_one = konst(&mut b, ConstValue::F32(1.0));
    let vocab = konst(&mut b, ConstValue::U32(cfg.vocab_size));
    let vtile = konst(&mut b, ConstValue::U32(cfg.vocab_tile));
    let n_tiles = konst(&mut b, ConstValue::U32(cfg.num_vocab_tiles()));
    let per_thread = konst(&mut b, ConstValue::U32(cfg.vocab_tile / V1_BLOCK));
    bottom_tested_loop(&mut b, (zero, n_tiles, one), &[], |b, tile, _| {
        let v_base = op2(b, U32, KirOp::Mul, tile, vtile);
        bottom_tested_loop(b, (zero, per_thread, one), &[], |b, j, _| {
            let lane_base = op2(b, U32, KirOp::Mul, j, block);
            let slot = op2(b, U32, KirOp::Add, lane_base, tid);
            let v = op2(b, U32, KirOp::Add, slot, v_base);
            let in_vocab = cmp(b, v, vocab, CmpOp::Lt);
            let column = b.new_block();
            let column_done = b.new_block();
            b.terminate(KirTerminator::CondBranch(in_vocab, KirEdge::to(column), KirEdge::to(column_done)));

            b.set_block(column);
            let v_wide = widen(b, v);
            let ws_row = op2(b, U64, KirOp::Mul, v_wide, hs_wide);
            let s_dot = dot(b, (xs, xs_row), (ws, ws_row), cfg.student_hidden);
            let bs_addr = at(b, F32, Global, bs, v);
            let bs_v = load(b, F32, bs_addr, Global);
            let s = op2(b, F32, KirOp::Add, s_dot, bs_v);
            let wt_row = op2(b, U64, KirOp::Mul, v_wide, ht_wide);
            let t_dot = dot(b, (xt, xt_row), (wt, wt_row), cfg.teacher_hidden);
            let bt_addr = at(b, F32, Global, bt, v);
            let bt_v = load(b, F32, bt_addr, Global);
            let t = op2(b, F32, KirOp::Add, t_dot, bt_v);
            // The three probabilities, from the saved LSEs.
            let d = op2(b, F32, KirOp::Sub, s, lse_s1_row);
            let p_s1 = f32_exp(b, d);
            let st = op2(b, F32, KirOp::Mul, s, tinv);
            let d = op2(b, F32, KirOp::Sub, st, lse_st_row);
            let p_st = f32_exp(b, d);
            let tt = op2(b, F32, KirOp::Mul, t, tinv);
            let d = op2(b, F32, KirOp::Sub, tt, lse_tt_row);
            let p_tt = f32_exp(b, d);
            // dl = alpha*(p_s1 - [v == target]) + (1 - alpha)*T*(p_sT - p_tT)
            let v_signed = b.new_typed_var(I64);
            b.emit(KirOp::Cast(v_signed, v, I64));
            let is_target = cmp(b, v_signed, target, CmpOp::Eq);
            let p_s1_minus_one = op2(b, F32, KirOp::Sub, p_s1, f_one);
            let p_ce = b.new_typed_var(F32);
            b.emit(KirOp::Select(p_ce, is_target, p_s1_minus_one, p_s1));
            let dl_ce = op2(b, F32, KirOp::Mul, alpha, p_ce);
            let p_diff = op2(b, F32, KirOp::Sub, p_st, p_tt);
            let one_minus_alpha = op2(b, F32, KirOp::Sub, f_one, alpha);
            let kl_weight = op2(b, F32, KirOp::Mul, one_minus_alpha, temp);
            let dl = b.new_typed_var(F32);
            b.emit(KirOp::Fma(dl, kl_weight, p_diff, dl_ce));
            let g = op2(b, F32, KirOp::Mul, dl, scale);
            // Scatter into the student's gradients.
            bottom_tested_loop(b, (zero, hs, one), &[], |b, h, _| {
                let h_wide = widen(b, h);
                let w_index = op2(b, U64, KirOp::Add, ws_row, h_wide);
                let w_addr = at(b, F32, Global, ws, w_index);
                let wv = load(b, F32, w_addr, Global);
                let dx_part = op2(b, F32, KirOp::Mul, g, wv);
                let x_index = op2(b, U64, KirOp::Add, xs_row, h_wide);
                let dx_addr = at(b, F32, Global, dxs_out, x_index);
                b.emit(KirOp::AtomicAdd(dx_addr, dx_part, Global));
                let x_addr = at(b, F32, Global, xs, x_index);
                let xv = load(b, F32, x_addr, Global);
                let dw_part = op2(b, F32, KirOp::Mul, g, xv);
                let dw_addr = at(b, F32, Global, dws_out, w_index);
                b.emit(KirOp::AtomicAdd(dw_addr, dw_part, Global));
                vec![]
            });
            let db_addr = at(b, F32, Global, dbs_out, v);
            b.emit(KirOp::AtomicAdd(db_addr, g, Global));
            b.terminate(KirTerminator::Branch(KirEdge::to(column_done)));

            b.set_block(column_done);
            vec![]
        });
        vec![]
    });
    b.terminate(KirTerminator::Branch(KirEdge::to(exit)));

    // ── an ignored row: its dx_s row is zeroed ──────────────────────────
    b.set_block(skip);
    let head = b.new_block();
    let body = b.new_block();
    let k = b.add_block_param(head, U32);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![zero])));
    b.set_block(head);
    let lane_base = op2(&mut b, U32, KirOp::Mul, k, block);
    let slot = op2(&mut b, U32, KirOp::Add, lane_base, tid);
    let in_row = cmp(&mut b, slot, hs, CmpOp::Lt);
    b.terminate(KirTerminator::CondBranch(in_row, KirEdge::to(body), KirEdge::to(exit)));
    b.set_block(body);
    let slot_wide = widen(&mut b, slot);
    let dx_index = op2(&mut b, U64, KirOp::Add, xs_row, slot_wide);
    let dx_addr = at(&mut b, F32, Global, dxs_out, dx_index);
    let f_zero = konst(&mut b, ConstValue::F32(0.0));
    b.emit(KirOp::Store(dx_addr, f_zero, Global));
    let k_next = op2(&mut b, U32, KirOp::Add, k, one);
    b.terminate(KirTerminator::Branch(KirEdge::with(head, vec![k_next])));

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.finalize()
}

// ─── Reference implementation (test oracle) ─────────────────────────────────

/// Pure-Rust f64 reference for the fused KL-CE forward. Returns
/// `(per_row_loss, lse_s1, lse_sT, lse_tT, num_valid)`.
///
/// Used by unit tests and the GPU parity suite; NOT a production fallback
/// (the production CPU path is the stdlib composite via tape AD).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn reference_forward_f64(
    xs: &[f64],
    ws: &[f64],
    bs: &[f64],
    xt: &[f64],
    wt: &[f64],
    bt: &[f64],
    targets: &[i64],
    rows: usize,
    vocab: usize,
    hs: usize,
    ht: usize,
    alpha: f64,
    temp: f64,
    ignore_index: i64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    let mut loss = vec![0.0; rows];
    let mut lse_s1 = vec![0.0; rows];
    let mut lse_st = vec![0.0; rows];
    let mut lse_tt = vec![0.0; rows];
    let mut num_valid = 0usize;

    for r in 0..rows {
        let tgt = targets[r];
        if tgt == ignore_index {
            continue;
        }
        num_valid += 1;
        let s_logits: Vec<f64> = (0..vocab)
            .map(|v| {
                let mut acc = bs[v];
                for h in 0..hs {
                    acc += xs[r * hs + h] * ws[v * hs + h];
                }
                acc
            })
            .collect();
        let t_logits: Vec<f64> = (0..vocab)
            .map(|v| {
                let mut acc = bt[v];
                for h in 0..ht {
                    acc += xt[r * ht + h] * wt[v * ht + h];
                }
                acc
            })
            .collect();

        let lse = |xs: &[f64]| {
            let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            m + xs.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
        };
        let l_s1 = lse(&s_logits);
        let s_t: Vec<f64> = s_logits.iter().map(|&x| x / temp).collect();
        let t_t: Vec<f64> = t_logits.iter().map(|&x| x / temp).collect();
        let l_st = lse(&s_t);
        let l_tt = lse(&t_t);

        let ce = l_s1 - s_logits[tgt as usize];
        // KL(p_t || p_s) at temperature T.
        let kl: f64 = (0..vocab)
            .map(|v| {
                let log_pt = t_t[v] - l_tt;
                let log_ps = s_t[v] - l_st;
                log_pt.exp() * (log_pt - log_ps)
            })
            .sum();

        loss[r] = alpha * ce + (1.0 - alpha) * temp * temp * kl;
        lse_s1[r] = l_s1;
        lse_st[r] = l_st;
        lse_tt[r] = l_tt;
    }

    (loss, lse_s1, lse_st, lse_tt, num_valid)
}

/// Pure-Rust f64 reference backward: returns `(dxs, dws, dbs)` — the
/// STUDENT gradients only (the teacher has none by construction).
#[allow(clippy::too_many_arguments)]
pub fn reference_backward_f64(
    xs: &[f64],
    ws: &[f64],
    bs: &[f64],
    xt: &[f64],
    wt: &[f64],
    bt: &[f64],
    targets: &[i64],
    rows: usize,
    vocab: usize,
    hs: usize,
    ht: usize,
    alpha: f64,
    temp: f64,
    ignore_index: i64,
    grad_output: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (_, lse_s1, lse_st, lse_tt, num_valid) = reference_forward_f64(
        xs, ws, bs, xt, wt, bt, targets, rows, vocab, hs, ht, alpha, temp, ignore_index,
    );
    let mut dxs = vec![0.0; rows * hs];
    let mut dws = vec![0.0; vocab * hs];
    let mut dbs = vec![0.0; vocab];
    if num_valid == 0 {
        return (dxs, dws, dbs);
    }
    let scale = grad_output / num_valid as f64;

    for r in 0..rows {
        let tgt = targets[r];
        if tgt == ignore_index {
            continue;
        }
        for v in 0..vocab {
            let mut s_v = bs[v];
            for h in 0..hs {
                s_v += xs[r * hs + h] * ws[v * hs + h];
            }
            let mut t_v = bt[v];
            for h in 0..ht {
                t_v += xt[r * ht + h] * wt[v * ht + h];
            }
            let mut p_s1 = (s_v - lse_s1[r]).exp();
            let p_st = (s_v / temp - lse_st[r]).exp();
            let p_tt = (t_v / temp - lse_tt[r]).exp();
            if v as i64 == tgt {
                p_s1 -= 1.0;
            }
            let dl = alpha * p_s1 + (1.0 - alpha) * temp * (p_st - p_tt);
            let sc = dl * scale;
            for h in 0..hs {
                dxs[r * hs + h] += sc * ws[v * hs + h];
                dws[v * hs + h] += sc * xs[r * hs + h];
            }
            dbs[v] += sc;
        }
    }
    (dxs, dws, dbs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> FusedKlCeConfig {
        FusedKlCeConfig {
            vocab_size: 256,
            student_hidden: 32,
            teacher_hidden: 64,
            batch_size: 2,
            seq_len: 4,
            vocab_tile: 128,
            gpu_sm: 80,
            ignore_index: -100,
        }
    }

    #[test]
    fn validate_accepts_small_cfg() {
        assert!(small_cfg().validate().is_ok());
    }

    #[test]
    fn validate_rejects_large_vocab() {
        let mut cfg = small_cfg();
        cfg.vocab_size = 16384;
        let err = cfg.validate().unwrap_err();
        assert!(err.contains("8192"), "err must name the v1 ceiling: {err}");
        assert!(err.contains("deferred"), "err must name the deferral: {err}");
    }

    #[test]
    fn validate_rejects_bad_hidden() {
        let mut cfg = small_cfg();
        cfg.student_hidden = 33;
        assert!(cfg.validate().is_err());
        let mut cfg = small_cfg();
        cfg.teacher_hidden = 100;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_bad_vocab_tile() {
        let mut cfg = small_cfg();
        cfg.vocab_tile = 100;
        assert!(cfg.validate().is_err());
        cfg.vocab_tile = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn ptx_is_null_terminated_and_ascii() {
        let cfg = small_cfg();
        for bytes in [
            synthesize_fused_kl_ce_ptx(&cfg),
            synthesize_fused_kl_ce_backward_ptx(&cfg),
        ] {
            assert_eq!(*bytes.last().unwrap(), 0, "must be null-terminated");
            // ptxas 13.x rejects non-ASCII anywhere in the module
            // (feedback_ptx_comment_ascii_only).
            assert!(
                bytes[..bytes.len() - 1].iter().all(|&b| b.is_ascii()),
                "PTX must be pure ASCII"
            );
        }
    }

    #[test]
    fn ptx_register_declarations_cover_uses() {
        // Guard against the `.reg .f32 %f<3>` class of bug (declares
        // %f0..%f2 while the body uses %f3 — see bugs.md 2026-07-10):
        // scan every numbered register use and assert it is below the
        // declared count for its family.
        let cfg = small_cfg();
        for ptx in [
            String::from_utf8(synthesize_fused_kl_ce_ptx(&cfg)[..].to_vec())
                .unwrap()
                .trim_end_matches('\0')
                .to_string(),
            String::from_utf8(synthesize_fused_kl_ce_backward_ptx(&cfg)[..].to_vec())
                .unwrap()
                .trim_end_matches('\0')
                .to_string(),
        ] {
            for family in ["rd", "r"] {
                let decl_marker = format!(".reg .u64 %{family}<");
                let decl_marker32 = format!(".reg .u32 %{family}<");
                let declared: u32 = ptx
                    .lines()
                    .find_map(|l| {
                        let l = l.trim();
                        let rest = l
                            .strip_prefix(&decl_marker)
                            .or_else(|| l.strip_prefix(&decl_marker32))?;
                        rest.split('>').next()?.parse().ok()
                    })
                    .unwrap_or_else(|| panic!("no decl for %{family} family"));
                let re = regex::Regex::new(&format!(r"%{family}(\d+)")).unwrap();
                let max_used = re
                    .captures_iter(&ptx)
                    .map(|c| c[1].parse::<u32>().unwrap())
                    .max()
                    .unwrap_or(0);
                assert!(
                    max_used < declared,
                    "%{family}{max_used} used but only %{family}<{declared}> declared"
                );
            }
        }
    }

    #[test]
    fn kernel_names_encode_shape() {
        let cfg = small_cfg();
        assert_eq!(cfg.kernel_name(), "nsl_fused_kl_ce_f32_v256_hs32_ht64");
        assert_eq!(
            cfg.bwd_kernel_name(),
            "nsl_fused_kl_ce_backward_f32_v256_hs32_ht64"
        );
        let fwd = String::from_utf8(synthesize_fused_kl_ce_ptx(&cfg)).unwrap();
        assert!(fwd.contains(&cfg.kernel_name()));
        let bwd = String::from_utf8(synthesize_fused_kl_ce_backward_ptx(&cfg)).unwrap();
        assert!(bwd.contains(&cfg.bwd_kernel_name()));
    }

    #[test]
    fn reference_forward_matches_composite_math() {
        // Cross-check the reference against a direct (non-online)
        // computation on a tiny problem.
        let rows = 3;
        let vocab = 5;
        let hs = 2;
        let ht = 3;
        let xs: Vec<f64> = (0..rows * hs).map(|i| (i as f64 * 0.37).sin()).collect();
        let ws: Vec<f64> = (0..vocab * hs).map(|i| (i as f64 * 0.11).cos()).collect();
        let bs: Vec<f64> = (0..vocab).map(|i| i as f64 * 0.01).collect();
        let xt: Vec<f64> = (0..rows * ht).map(|i| (i as f64 * 0.29).sin()).collect();
        let wt: Vec<f64> = (0..vocab * ht).map(|i| (i as f64 * 0.13).cos()).collect();
        let bt: Vec<f64> = (0..vocab).map(|i| -(i as f64) * 0.02).collect();
        let targets = vec![1i64, -100, 4];
        let alpha = 0.3;
        let temp = 2.0;

        let (loss, lse_s1, _lse_st, _lse_tt, num_valid) = reference_forward_f64(
            &xs, &ws, &bs, &xt, &wt, &bt, &targets, rows, vocab, hs, ht, alpha, temp, -100,
        );
        assert_eq!(num_valid, 2);
        assert_eq!(loss[1], 0.0, "ignored row contributes zero");

        // Direct softmax computation for row 0.
        let s0: Vec<f64> = (0..vocab)
            .map(|v| bs[v] + (0..hs).map(|h| xs[h] * ws[v * hs + h]).sum::<f64>())
            .collect();
        let t0: Vec<f64> = (0..vocab)
            .map(|v| bt[v] + (0..ht).map(|h| xt[h] * wt[v * ht + h]).sum::<f64>())
            .collect();
        let softmax = |xs: &[f64]| {
            let m = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = xs.iter().map(|&x| (x - m).exp()).collect();
            let s: f64 = e.iter().sum();
            e.into_iter().map(|x| x / s).collect::<Vec<f64>>()
        };
        let ps1 = softmax(&s0);
        let ce = -(ps1[1]).ln();
        assert!((lse_s1[0] - (s0[1] - ps1[1].ln())).abs() < 1e-9);
        let pst = softmax(&s0.iter().map(|&x| x / temp).collect::<Vec<_>>());
        let ptt = softmax(&t0.iter().map(|&x| x / temp).collect::<Vec<_>>());
        let kl: f64 = (0..vocab)
            .map(|v| ptt[v] * (ptt[v].ln() - pst[v].ln()))
            .sum();
        let expected = alpha * ce + (1.0 - alpha) * temp * temp * kl;
        assert!(
            (loss[0] - expected).abs() < 1e-9,
            "loss {} != expected {}",
            loss[0],
            expected
        );
    }

    #[test]
    fn reference_backward_matches_finite_differences() {
        let rows = 2;
        let vocab = 4;
        let hs = 2;
        let ht = 2;
        let xs: Vec<f64> = (0..rows * hs).map(|i| (i as f64 * 0.41).sin()).collect();
        let ws: Vec<f64> = (0..vocab * hs).map(|i| (i as f64 * 0.17).cos()).collect();
        let bs: Vec<f64> = (0..vocab).map(|i| i as f64 * 0.03).collect();
        let xt: Vec<f64> = (0..rows * ht).map(|i| (i as f64 * 0.23).sin()).collect();
        let wt: Vec<f64> = (0..vocab * ht).map(|i| (i as f64 * 0.19).cos()).collect();
        let bt: Vec<f64> = (0..vocab).map(|i| -(i as f64) * 0.01).collect();
        let targets = vec![2i64, 0];
        let alpha = 0.6;
        let temp = 3.0;

        let mean_loss = |ws_: &[f64], bs_: &[f64], xs_: &[f64]| -> f64 {
            let (loss, _, _, _, nv) = reference_forward_f64(
                xs_, ws_, bs_, &xt, &wt, &bt, &targets, rows, vocab, hs, ht, alpha, temp, -100,
            );
            loss.iter().sum::<f64>() / nv as f64
        };

        let (dxs, dws, dbs) = reference_backward_f64(
            &xs, &ws, &bs, &xt, &wt, &bt, &targets, rows, vocab, hs, ht, alpha, temp, -100, 1.0,
        );

        let eps = 1e-6;
        // dW_s finite differences.
        for i in [0usize, 3, 7] {
            let mut wp = ws.clone();
            wp[i] += eps;
            let mut wm = ws.clone();
            wm[i] -= eps;
            let fd = (mean_loss(&wp, &bs, &xs) - mean_loss(&wm, &bs, &xs)) / (2.0 * eps);
            assert!(
                (fd - dws[i]).abs() < 1e-5,
                "dW[{i}]: fd {fd} vs analytic {}",
                dws[i]
            );
        }
        // dbias_s.
        for i in [0usize, 2] {
            let mut bp = bs.clone();
            bp[i] += eps;
            let mut bm = bs.clone();
            bm[i] -= eps;
            let fd = (mean_loss(&ws, &bp, &xs) - mean_loss(&ws, &bm, &xs)) / (2.0 * eps);
            assert!(
                (fd - dbs[i]).abs() < 1e-5,
                "dbias[{i}]: fd {fd} vs analytic {}",
                dbs[i]
            );
        }
        // dx_s.
        for i in [0usize, 3] {
            let mut xp = xs.clone();
            xp[i] += eps;
            let mut xm = xs.clone();
            xm[i] -= eps;
            let fd = (mean_loss(&ws, &bs, &xp) - mean_loss(&ws, &bs, &xm)) / (2.0 * eps);
            assert!(
                (fd - dxs[i]).abs() < 1e-5,
                "dx[{i}]: fd {fd} vs analytic {}",
                dxs[i]
            );
        }
    }
}
