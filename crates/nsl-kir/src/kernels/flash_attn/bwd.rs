// crates/nsl-kir/src/kernels/flash_attn/bwd.rs
//! The flash-attention backward on tensor cores: three kernels, no output
//! atomics.
//!
//! 1. **Pre-pass** ([`build_pre`]): one warp per query row computes
//!    `D[r] = Σ_d dO[r, d] · O[r, d]` (f32) and folds `max_d |dO[r, d]|`
//!    into its head's word of `gmax` with `red.global.max.u32` on the f32
//!    bit pattern (non-negative floats order as their bits do).
//! 2. **dQ** ([`build_dq`]): one CTA per `(query tile, batch·head)`, as the
//!    forward; per key/value tile each warp recomputes `S = Q·Kᵀ`, `P =
//!    exp(S·scale - lse)`, `dP = dO·Vᵀ`, `dS = P ∘ (dP - D)` and accumulates
//!    `dQ += dS·K`. Every `dQ` row has one writer.
//! 3. **dK/dV** ([`build_dkdv`]): one CTA per `(key tile, batch·kv-head)`;
//!    it walks every query head of its group and every query tile at or
//!    after its keys, and per tile each warp (16 keys) computes `Sᵀ = K·Qᵀ`,
//!    `Pᵀ`, `dV += Pᵀ·dO`, `dPᵀ = V·dOᵀ`, `dSᵀ = Pᵀ ∘ (dPᵀ - D)`, `dK +=
//!    dSᵀ·Q`. Every `dK`/`dV` row has one writer, so grouped-query attention
//!    needs no expanded K/V and no group sum.
//!
//! # Gradient prescale
//!
//! `dO` reaches the tensor cores as f16. The backward is linear in `dO`, so
//! it is multiplied by an exact power of two `gs` before rounding (putting
//! the largest scaled magnitude in `[2^GRAD_SCALE_EXP, 2^(GRAD_SCALE_EXP+1))`,
//! see [`super::grad_scale_for_max`]) and the results by `1/gs` as they
//! leave the f32 accumulators; `D` is scaled by the same `gs` so `dP - D`
//! stays consistent. A power-of-two multiply moves only the exponent, so the
//! scaling rounds nothing itself.
//!
//! - In the dQ kernel each query row's `dS` row depends on that row's `dO`
//!   alone, so the scale is **per row**: every row's `dO` uses f16's full
//!   precision whatever the other rows hold.
//! - In the dK/dV kernel a product sums over query rows, so one scale must
//!   hold across the rows summed: it is **per batch and kv-head group**,
//!   from the largest `|dO|` over the group's query heads. Rows far below
//!   that maximum contribute proportionally little to `dK`/`dV`; 20 binades
//!   of f16's normal range sit below the scaled maximum.
//!
//! `dS` is clamped to f16's finite range before rounding (ordered compares,
//! so a NaN stays visible).

use super::fwd::{acc_to_a_frags, load_a_frags, mma_b_rows, mma_b_trans, Lane, TileMap};
use super::{
    f32_global, f32_shared, grad_scale_from_max_bits, h2, h2_shared, io_global, u32_global, G, F16_MAX,
    LOG2_E, SMEM_ROW_PAD,
};
use crate::kernels::storage::Storage;
use crate::kernel_ir::{
    AddressSpace, CmpOp, KernelIR, KirEdge, KirOp, KirTerminator, KirType, SmemLayout, SmemRegion, VarId,
};

/// One backward variant: the problem (`head_dim`, `causal`) and the
/// tilings of the dQ kernel (`dq_warps` of 16 query rows, `dq_block_n` keys
/// per tile) and the dK/dV kernel (`dkdv_warps` of 16 keys, `dkdv_block_q`
/// queries per tile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BwdConfig {
    pub head_dim: u32,
    pub causal: bool,
    pub dq_warps: u32,
    pub dq_block_n: u32,
    pub dkdv_warps: u32,
    pub dkdv_block_q: u32,
    /// How Q, K, V, O, dO, dQ, dK and dV are stored (see
    /// [`super::fwd::FwdConfig::io`]); `lse`, `D` and `gmax` stay f32.
    pub io: Storage,
}

/// Warps per CTA of the pre-pass, and rows each warp reduces.
pub const PRE_WARPS: u32 = 8;
pub const PRE_ROWS_PER_WARP: u32 = 4;

impl BwdConfig {
    pub const HEAD_DIMS: [u32; 2] = [64, 128];

    /// The tilings the runtime launches (chosen by measurement, see
    /// `tc_attention_kernel_times` in nsl-runtime's `attention_tc.rs`).
    pub fn new(head_dim: u32, causal: bool) -> BwdConfig {
        // Measured on an RTX PRO 4500 (sm_120) at Coder-1B's shape (batch
        // 2, 32 heads, seq 2048, causal): head_dim 64 dQ 0.57 ms and dK/dV
        // 0.76 ms at 4 warps x 32, against 0.57-0.84 / 0.76-1.28 for the
        // other tilings of 4 or 8 warps x 32 / 64 / 128; head_dim 128 dQ
        // 0.60 ms and dK/dV 0.76 ms at 8 warps x 32.
        let (dq_warps, dq_block_n, dkdv_warps, dkdv_block_q) =
            if head_dim > 64 { (8, 32, 8, 32) } else { (4, 32, 4, 32) };
        BwdConfig { head_dim, causal, dq_warps, dq_block_n, dkdv_warps, dkdv_block_q, io: Storage::F32 }
    }

    pub fn supported(head_dim: u32) -> bool {
        Self::HEAD_DIMS.contains(&head_dim)
    }

    fn tag(&self) -> String {
        format!("d{}_{}", self.head_dim, if self.causal { "causal" } else { "full" })
    }

    /// This variant with its activations and their gradients stored as `io`.
    pub fn with_io(self, io: Storage) -> BwdConfig {
        BwdConfig { io, ..self }
    }

    fn io_suffix(&self) -> &'static str {
        if self.io == Storage::Bf16 { "_bf16" } else { "" }
    }

    /// The same problem with the pre-pass's (tiling-independent) identity.
    pub fn pre_only(head_dim: u32) -> BwdConfig {
        BwdConfig::new(head_dim, false)
    }

    fn row_words(&self) -> u32 {
        (self.head_dim + SMEM_ROW_PAD) / 2
    }

    // ── pre-pass ──

    pub fn pre_kernel_name(&self) -> String {
        format!("nsl_flash_tc_bwd_pre_d{}{}", self.head_dim, self.io_suffix())
    }

    pub fn pre_threads(&self) -> u32 {
        32 * PRE_WARPS
    }

    /// `[ceil(rows / rows per CTA), 1, 1]` for `rows = batch · heads · seq`.
    pub fn pre_grid(&self, rows: u32) -> [u32; 3] {
        [rows.div_ceil(PRE_WARPS * PRE_ROWS_PER_WARP), 1, 1]
    }

    /// `dout, out` f32, `dvec` f32 `[rows]`, `gmax` u32 `[batch · heads]`
    /// (zeroed by the caller), `rows`, `seq` u32.
    pub fn pre_param_names() -> [&'static str; 6] {
        ["dout", "out", "dvec", "gmax", "rows", "seq"]
    }

    // ── dQ ──

    pub fn dq_kernel_name(&self) -> String {
        format!("nsl_flash_tc_bwd_dq_{}_w{}_n{}{}", self.tag(), self.dq_warps, self.dq_block_n, self.io_suffix())
    }

    pub fn dq_block_m(&self) -> u32 {
        16 * self.dq_warps
    }

    pub fn dq_block_n(&self) -> u32 {
        self.dq_block_n
    }

    pub fn dq_threads(&self) -> u32 {
        32 * self.dq_warps
    }

    pub fn dq_grid(&self, batch: u32, heads: u32, seq: u32) -> [u32; 3] {
        [seq.div_ceil(self.dq_block_m()), batch * heads, 1]
    }

    /// `q, k, v, dout, lse, dvec, dq` f32 device pointers, `seq, heads,
    /// kv_heads` u32, `scale` f32.
    pub fn dq_param_names() -> [&'static str; 11] {
        ["q", "k", "v", "dout", "lse", "dvec", "dq", "seq", "heads", "kv_heads", "scale"]
    }

    // ── dK/dV ──

    pub fn dkdv_kernel_name(&self) -> String {
        format!("nsl_flash_tc_bwd_dkdv_{}_w{}_q{}{}", self.tag(), self.dkdv_warps, self.dkdv_block_q, self.io_suffix())
    }

    pub fn dkdv_block_n(&self) -> u32 {
        16 * self.dkdv_warps
    }

    pub fn dkdv_block_q(&self) -> u32 {
        self.dkdv_block_q
    }

    pub fn dkdv_threads(&self) -> u32 {
        32 * self.dkdv_warps
    }

    pub fn dkdv_grid(&self, batch: u32, kv_heads: u32, seq: u32) -> [u32; 3] {
        [seq.div_ceil(self.dkdv_block_n()), batch * kv_heads, 1]
    }

    /// The dK/dV kernel's dynamic shared memory, in bytes (the launch
    /// requests exactly this).
    pub fn dkdv_smem_bytes(&self) -> u32 {
        dkdv_layout(self).total_bytes().expect("the layout fits u32")
    }

    /// `q, k, v, dout, lse, dvec` f32, `gmax` u32, `dk, dv` f32 device
    /// pointers, `seq, heads, kv_heads` u32, `scale` f32.
    pub fn dkdv_param_names() -> [&'static str; 13] {
        ["q", "k", "v", "dout", "lse", "dvec", "gmax", "dk", "dv", "seq", "heads", "kv_heads", "scale"]
    }
}

// ─────────────────────────────── pre-pass ───────────────────────────────

/// Build the pre-pass for one head_dim (it does not depend on causality).
pub fn build_pre(cfg: BwdConfig) -> KernelIR {
    use KirType::U32;
    let hd = cfg.head_dim;
    let per_lane = hd / 32; // 2 or 4 consecutive elements
    let names = BwdConfig::pre_param_names();
    let mut g = G::new(&cfg.pre_kernel_name());
    let dout = g.b.add_param(names[0], io_global(cfg.io), AddressSpace::Global);
    let out = g.b.add_param(names[1], io_global(cfg.io), AddressSpace::Global);
    let dvec = g.b.add_param(names[2], f32_global(), AddressSpace::Global);
    let gmax = g.b.add_param(names[3], u32_global(), AddressSpace::Global);
    let rows = g.b.add_param(names[4], U32, AddressSpace::Local);
    let seq = g.b.add_param(names[5], U32, AddressSpace::Local);
    g.b.set_workgroup_size([cfg.pre_threads(), 1, 1]);
    g.b.set_launch_bounds(cfg.pre_threads(), None);
    let entry = g.b.new_block();
    g.b.set_block(entry);

    let tid = g.var(U32);
    g.b.emit(KirOp::ThreadId(tid, 0));
    let ln = Lane::new(&mut g, tid);
    let cta = g.var(U32);
    g.b.emit(KirOp::BlockIdx(cta, 0));
    let per_cta = PRE_WARPS * PRE_ROWS_PER_WARP;
    let first = g.mul_u32c(cta, per_cta);
    let wfirst = {
        let w = g.mul_u32c(ln.warp, PRE_ROWS_PER_WARP);
        g.add(first, w)
    };
    let one = g.u32c(1);
    let rows_m1 = g.sub(rows, one);
    let lane_off = g.mul_u32c(ln.lane, per_lane);
    let zero_u = g.u32c(0);
    let lane0 = g.cmp(ln.lane, zero_u, CmpOp::Eq);
    for i in 0..PRE_ROWS_PER_WARP {
        let row = g.add_u32c(wfirst, i);
        let valid = g.cmp(row, rows, CmpOp::Lt);
        let rc = g.min(row, rows_m1);
        let off = {
            let r64 = g.cast(rc, KirType::U64);
            let hd64 = g.u64c(hd as u64);
            let ro = g.mul(r64, hd64);
            let lo = g.cast(lane_off, KirType::U64);
            g.add(ro, lo)
        };
        let pd = g.ptr(dout, off);
        let po = g.ptr(out, off);
        let dv = g.load_vec_f32(pd, per_lane as usize, AddressSpace::Global);
        let ov = g.load_vec_f32(po, per_lane as usize, AddressSpace::Global);
        let mut dot = g.mul(dv[0], ov[0]);
        let mut amax = g.abs(dv[0]);
        for e in 1..per_lane as usize {
            dot = g.fma(dv[e], ov[e], dot);
            let a = g.abs(dv[e]);
            amax = g.max(amax, a);
        }
        for m in [16u32, 8, 4, 2, 1] {
            let o = g.shfl_xor(dot, m);
            dot = g.add(dot, o);
            let a = g.shfl_xor(amax, m);
            amax = g.max(amax, a);
        }
        let w = g.and(valid, lane0);
        let pdv = g.ptr(dvec, rc);
        g.when(w, KirOp::Store(pdv, dot, AddressSpace::Global));
        let head = g.div(rc, seq);
        let bits = g.bitcast(amax, U32);
        let pg = g.ptr(gmax, head);
        g.when(w, KirOp::AtomicMax(pg, bits, AddressSpace::Global));
    }
    g.b.terminate(KirTerminator::Return);
    g.b.finalize()
}

// ─────────────────────────────── dQ ───────────────────────────────

/// Build one dQ variant.
pub fn build_dq(cfg: BwdConfig) -> KernelIR {
    use KirType::{F32, U32, U64};
    assert!(BwdConfig::supported(cfg.head_dim));
    let hd = cfg.head_dim;
    let bm = cfg.dq_block_m();
    let bn = cfg.dq_block_n();
    let rw = cfg.row_words();
    let n_dn = (hd / 8) as usize;
    let n_sn = (bn / 8) as usize;
    let names = BwdConfig::dq_param_names();

    let mut g = G::new(&cfg.dq_kernel_name());
    let q = g.b.add_param(names[0], io_global(cfg.io), AddressSpace::Global);
    let k = g.b.add_param(names[1], io_global(cfg.io), AddressSpace::Global);
    let v = g.b.add_param(names[2], io_global(cfg.io), AddressSpace::Global);
    let dout = g.b.add_param(names[3], io_global(cfg.io), AddressSpace::Global);
    let lse = g.b.add_param(names[4], f32_global(), AddressSpace::Global);
    let dvec = g.b.add_param(names[5], f32_global(), AddressSpace::Global);
    let dq = g.b.add_param(names[6], io_global(cfg.io), AddressSpace::Global);
    let seq = g.b.add_param(names[7], U32, AddressSpace::Local);
    let heads = g.b.add_param(names[8], U32, AddressSpace::Local);
    let kv_heads = g.b.add_param(names[9], U32, AddressSpace::Local);
    let scale = g.b.add_param(names[10], F32, AddressSpace::Local);

    let tile_bytes = bn * rw * 4;
    g.b.set_smem_layout(SmemLayout {
        regions: vec![
            SmemRegion { name: "k".into(), bytes: tile_bytes, align: 16, elem: h2() },
            SmemRegion { name: "v".into(), bytes: tile_bytes, align: 16, elem: h2() },
        ],
        dynamic: false,
    });
    g.b.set_workgroup_size([cfg.dq_threads(), 1, 1]);
    g.b.set_launch_bounds(cfg.dq_threads(), None);

    let entry = g.b.new_block();
    g.b.set_block(entry);
    let k_s = g.var(h2_shared());
    g.b.emit(KirOp::SharedRegion { dst: k_s, region: 0 });
    let v_s = g.var(h2_shared());
    g.b.emit(KirOp::SharedRegion { dst: v_s, region: 1 });
    let tid = g.var(U32);
    g.b.emit(KirOp::ThreadId(tid, 0));
    let ln = Lane::new(&mut g, tid);
    let qt_raw = g.var(U32);
    g.b.emit(KirOp::BlockIdx(qt_raw, 0));
    let bh = g.var(U32);
    g.b.emit(KirOp::BlockIdx(bh, 1));
    let one = g.u32c(1);

    let n_qt = {
        let s = g.add_u32c(seq, bm - 1);
        let sh = g.u32c(bm.trailing_zeros());
        g.shr(s, sh)
    };
    let q_tile = if cfg.causal {
        let last = g.sub(n_qt, one);
        g.sub(last, qt_raw)
    } else {
        qt_raw
    };
    let q0 = g.mul_u32c(q_tile, bm);
    let b_idx = g.div(bh, heads);
    let h = g.rem(bh, heads);
    let group = g.div(heads, kv_heads);
    let kvh = g.div(h, group);
    let kvb = g.mul(b_idx, kv_heads);
    let kv_bh = g.add(kvb, kvh);

    let seq64 = g.cast(seq, U64);
    let hd64 = g.u64c(hd as u64);
    let head_elems = g.mul(seq64, hd64);
    let bh64 = g.cast(bh, U64);
    let kvbh64 = g.cast(kv_bh, U64);
    let q_off = g.mul(bh64, head_elems);
    let kv_off = g.mul(kvbh64, head_elems);
    let row_off = g.mul(bh64, seq64);
    let q_base = g.ptr(q, q_off);
    let do_base = g.ptr(dout, q_off);
    let dq_base = g.ptr(dq, q_off);
    let k_base = g.ptr(k, kv_off);
    let v_base = g.ptr(v, kv_off);
    let lse_base = g.ptr(lse, row_off);
    let dvec_base = g.ptr(dvec, row_off);

    let seq_m1 = g.sub(seq, one);
    let wrow0 = {
        let w16 = g.mul_u32c(ln.warp, 16);
        g.add(q0, w16)
    };
    let r_lo = g.add(wrow0, ln.lr);
    let r_hi = g.add_u32c(r_lo, 8);
    let r_lo_c = g.min(r_lo, seq_m1);
    let r_hi_c = g.min(r_hi, seq_m1);
    let rows = [r_lo, r_hi];
    let rows_c = [r_lo_c, r_hi_c];

    let l2e = g.f32c(LOG2_E);
    let sl2 = g.mul(scale, l2e);
    let qa = load_a_frags(&mut g, q_base, r_lo_c, r_hi_c, ln.lc, hd, None);

    // dO rows: raw values, then the per-row prescale, then fragments.
    let (doa, gs, ginv) = {
        let raw = load_raw_rows(&mut g, do_base, r_lo_c, r_hi_c, ln.lc, hd);
        let mut gs = [0; 2];
        let mut ginv = [0; 2];
        for half in 0..2 {
            let vals: Vec<VarId> = raw
                .iter()
                .flat_map(|t| [t[half], t[half + 2]])
                .flat_map(|(x, y)| [x, y])
                .collect();
            let abs: Vec<VarId> = vals.iter().map(|&x| g.abs(x)).collect();
            let m = g.max_all(&abs);
            let m = g.quad_reduce(m, KirOp::Max);
            let bits = g.bitcast(m, U32);
            let (s, i) = grad_scale_from_max_bits(&mut g, bits);
            gs[half] = s;
            ginv[half] = i;
        }
        let frags: Vec<[VarId; 4]> = raw
            .iter()
            .map(|t| {
                let mut f = [0; 4];
                for (r, &(x0, x1)) in t.iter().enumerate() {
                    let s = gs[r % 2];
                    let a = g.mul(x0, s);
                    let b = g.mul(x1, s);
                    f[r] = g.pack(a, b);
                }
                f
            })
            .collect();
        (frags, gs, ginv)
    };
    // Per row: lse·log2 e and gs·D.
    let mut neg_lse2 = [0; 2];
    let mut dsc = [0; 2];
    for half in 0..2 {
        let pl = g.ptr(lse_base, rows_c[half]);
        let lv = g.load(pl, AddressSpace::Global);
        let l2 = g.mul(lv, l2e);
        neg_lse2[half] = g.neg(l2);
        let pd = g.ptr(dvec_base, rows_c[half]);
        let dv = g.load(pd, AddressSpace::Global);
        dsc[half] = g.mul(dv, gs[half]);
    }

    let kv_end = if cfg.causal {
        let e = g.add_u32c(q0, bm);
        g.min(e, seq)
    } else {
        seq
    };
    let n_kv = {
        let e = g.add_u32c(kv_end, bn - 1);
        let sh = g.u32c(bn.trailing_zeros());
        g.shr(e, sh)
    };
    let n_kv_m1 = g.sub(n_kv, one);

    let tm = TileMap::new(&mut g, tid, hd, bn, cfg.dq_threads());
    let cols4 = ln.ldm_cols4_off(&mut g, rw);
    let rows2 = ln.ldm_rows2_off(&mut g, rw);
    let k_rows = g.ptr(k_s, cols4);
    let k_trans = g.ptr(k_s, rows2);
    let v_rows = g.ptr(v_s, cols4);
    let key_lane = g.mul_u32c(ln.lc, 2);
    let neg_inf = g.f32c(f32::NEG_INFINITY);
    let zero_f = g.f32c(0.0);
    let warp_last_row = g.add_u32c(wrow0, 15);

    let zero_u = g.u32c(0);
    let pf_k0 = tm.load(&mut g, k_base, zero_u, seq, seq_m1, hd, None);
    let pf_v0 = tm.load(&mut g, v_base, zero_u, seq, seq_m1, hd, None);

    let header = g.b.new_block();
    let body = g.b.new_block();
    let exit = g.b.new_block();
    let j = g.b.add_block_param(header, U32);
    let dq_acc: Vec<[VarId; 4]> = (0..n_dn)
        .map(|_| {
            [
                g.b.add_block_param(header, F32),
                g.b.add_block_param(header, F32),
                g.b.add_block_param(header, F32),
                g.b.add_block_param(header, F32),
            ]
        })
        .collect();
    let pf_k: Vec<VarId> = (0..pf_k0.len()).map(|_| g.b.add_block_param(header, F32)).collect();
    let pf_v: Vec<VarId> = (0..pf_v0.len()).map(|_| g.b.add_block_param(header, F32)).collect();
    let mut init = vec![zero_u];
    init.extend(std::iter::repeat_n(zero_f, n_dn * 4));
    init.extend(&pf_k0);
    init.extend(&pf_v0);
    g.b.terminate(KirTerminator::Branch(KirEdge::with(header, init)));

    g.b.set_block(header);
    let done = g.cmp(j, n_kv, CmpOp::Ge);
    g.b.terminate(KirTerminator::CondBranch(done, KirEdge::to(exit), KirEdge::to(body)));

    g.b.set_block(body);
    g.b.emit(KirOp::Barrier);
    tm.store(&mut g, k_s, &pf_k, rw);
    tm.store(&mut g, v_s, &pf_v, rw);
    g.b.emit(KirOp::Barrier);
    let j1 = g.add(j, one);
    let jn = g.min(j1, n_kv_m1);
    let kvn = g.mul_u32c(jn, bn);
    let nk = tm.load(&mut g, k_base, kvn, seq, seq_m1, hd, None);
    let nv = tm.load(&mut g, v_base, kvn, seq, seq_m1, hd, None);
    let kv0 = g.mul_u32c(j, bn);
    let flat: Vec<VarId> = dq_acc.iter().flatten().copied().collect();
    let next_args = |acc: &[VarId]| -> Vec<VarId> {
        let mut a = vec![j1];
        a.extend_from_slice(acc);
        a.extend(&nk);
        a.extend(&nv);
        a
    };
    let kv_last = g.add_u32c(kv0, bn - 1);
    let kv_end_tile = g.add_u32c(kv0, bn);
    let tail = g.cmp(kv_end_tile, seq, CmpOp::Gt);
    let needs_mask = if cfg.causal {
        let diag = g.cmp(kv_last, wrow0, CmpOp::Gt);
        g.or(diag, tail)
    } else {
        tail
    };
    let masked_blk = g.b.new_block();
    let plain_blk = g.b.new_block();
    if cfg.causal {
        let skip = g.cmp(kv0, warp_last_row, CmpOp::Gt);
        let check = g.b.new_block();
        g.b.terminate(KirTerminator::CondBranch(skip, KirEdge::with(header, next_args(&flat)), KirEdge::to(check)));
        g.b.set_block(check);
    }
    g.b.terminate(KirTerminator::CondBranch(needs_mask, KirEdge::to(masked_blk), KirEdge::to(plain_blk)));

    for (blk, masked) in [(masked_blk, true), (plain_blk, false)] {
        g.b.set_block(blk);
        let z = zero_f;
        // S = Q·Kᵀ, masked.
        let mut s: Vec<[VarId; 4]> = vec![[z, z, z, z]; n_sn];
        mma_b_rows(&mut g, &mut s, &qa, k_rows, rw);
        if masked {
            mask_scores(&mut g, &mut s, kv0, key_lane, rows, seq, cfg.causal, neg_inf);
        }
        // P = 2^(S·sl2 - lse·log2 e).
        for acc in s.iter_mut() {
            for (idx, slot) in acc.iter_mut().enumerate() {
                let arg = g.fma(*slot, sl2, neg_lse2[idx / 2]);
                *slot = g.exp2(arg);
            }
        }
        // dP = dO·Vᵀ (scaled by gs per row).
        let mut dp: Vec<[VarId; 4]> = vec![[z, z, z, z]; n_sn];
        mma_b_rows(&mut g, &mut dp, &doa, v_rows, rw);
        // dS = P ∘ (dP - gs·D), clamped to f16's range.
        let ds: Vec<[VarId; 4]> = s
            .iter()
            .zip(&dp)
            .map(|(p, d)| {
                let mut out = [0; 4];
                for idx in 0..4 {
                    let t = g.sub(d[idx], dsc[idx / 2]);
                    let x = g.mul(p[idx], t);
                    out[idx] = clamp_f16(&mut g, x);
                }
                out
            })
            .collect();
        let dsa = acc_to_a_frags(&mut g, &ds);
        let mut acc = dq_acc.clone();
        mma_b_trans(&mut g, &mut acc, &dsa, k_trans, rw);
        let flat_new: Vec<VarId> = acc.into_iter().flatten().collect();
        g.b.terminate(KirTerminator::Branch(KirEdge::with(header, next_args(&flat_new))));
    }

    // dQ = scale · (1/gs) · acc
    g.b.set_block(exit);
    let c2 = g.mul_u32c(ln.lc, 2);
    for half in 0..2 {
        let f = g.mul(scale, ginv[half]);
        let valid = g.cmp(rows[half], seq, CmpOp::Lt);
        let roff = g.mul_u32c(rows[half], hd);
        let roff = g.add(roff, c2);
        let prow = g.ptr(dq_base, roff);
        for (n, acc) in dq_acc.iter().enumerate() {
            let x0 = g.mul(acc[half * 2], f);
            let x1 = g.mul(acc[half * 2 + 1], f);
            let p = g.ptr_c(prow, n as u32 * 8);
            let vals = g.narrow_to(p, vec![x0, x1]);
            g.when(valid, KirOp::StoreVec { ptr: p, vals, space: AddressSpace::Global });
        }
    }
    g.b.terminate(KirTerminator::Return);
    g.b.finalize()
}

/// The raw f32 values a lane holds of rows `lo`/`hi` in m16k16 A-fragment
/// order: per k-step, registers 0..4 as `(x, x+1)` pairs (`[lo c, hi c, lo
/// c+8, hi c+8]`).
fn load_raw_rows(
    g: &mut G,
    base: VarId,
    row_lo: VarId,
    row_hi: VarId,
    lc: VarId,
    head_dim: u32,
) -> Vec<[(VarId, VarId); 4]> {
    let lo_off = g.mul_u32c(row_lo, head_dim);
    let hi_off = g.mul_u32c(row_hi, head_dim);
    let c2 = g.mul_u32c(lc, 2);
    let lo_off = g.add(lo_off, c2);
    let hi_off = g.add(hi_off, c2);
    let p_lo = g.ptr(base, lo_off);
    let p_hi = g.ptr(base, hi_off);
    (0..head_dim / 16)
        .map(|t| {
            let mut regs = [(0, 0); 4];
            for (r, (p, col)) in [(p_lo, 0u32), (p_hi, 0), (p_lo, 8), (p_hi, 8)].into_iter().enumerate() {
                let pp = g.ptr_c(p, 16 * t + col);
                let v = g.load_vec_f32(pp, 2, AddressSpace::Global);
                regs[r] = (v[0], v[1]);
            }
            regs
        })
        .collect()
}

/// `x` saturated at ±65504 with ordered compares (a NaN stays NaN).
fn clamp_f16(g: &mut G, x: VarId) -> VarId {
    let hi = g.f32c(F16_MAX);
    let lo = g.f32c(-F16_MAX);
    let over = g.cmp(x, hi, CmpOp::Gt);
    let x = g.sel(over, hi, x);
    let under = g.cmp(x, lo, CmpOp::Lt);
    g.sel(under, lo, x)
}

/// Score-shaped masking for a tile whose keys start at `kv0`: a lane's
/// element `(row, kv0 + 8n + key_lane + e)` becomes `-inf` when the key is
/// at or past `seq`, or (causal) after the row.
#[allow(clippy::too_many_arguments)]
fn mask_scores(
    g: &mut G,
    s: &mut [[VarId; 4]],
    kv0: VarId,
    key_lane: VarId,
    rows: [VarId; 2],
    seq: VarId,
    causal: bool,
    neg_inf: VarId,
) {
    let key0 = g.add(kv0, key_lane);
    for (n, acc) in s.iter_mut().enumerate() {
        for e in 0..2u32 {
            let key = g.add_u32c(key0, n as u32 * 8 + e);
            let oob = g.cmp(key, seq, CmpOp::Ge);
            for (half, &row) in rows.iter().enumerate() {
                let idx = half * 2 + e as usize;
                let dead = if causal {
                    let future = g.cmp(key, row, CmpOp::Gt);
                    g.or(future, oob)
                } else {
                    oob
                };
                acc[idx] = g.sel(dead, neg_inf, acc[idx]);
            }
        }
    }
}

// ─────────────────────────────── dK/dV ───────────────────────────────

fn dkdv_layout(cfg: &BwdConfig) -> SmemLayout {
    let rw = cfg.row_words();
    let bn = cfg.dkdv_block_n();
    let bq = cfg.dkdv_block_q();
    SmemLayout {
        regions: vec![
            SmemRegion { name: "k".into(), bytes: bn * rw * 4, align: 16, elem: h2() },
            SmemRegion { name: "v".into(), bytes: bn * rw * 4, align: 16, elem: h2() },
            SmemRegion { name: "q".into(), bytes: bq * rw * 4, align: 16, elem: h2() },
            SmemRegion { name: "do".into(), bytes: bq * rw * 4, align: 16, elem: h2() },
            SmemRegion { name: "neg_lse2".into(), bytes: bq * 4, align: 16, elem: KirType::F32 },
            SmemRegion { name: "dsc".into(), bytes: bq * 4, align: 16, elem: KirType::F32 },
        ],
        dynamic: true,
    }
}

/// Build one dK/dV variant.
pub fn build_dkdv(cfg: BwdConfig) -> KernelIR {
    use KirType::{F32, U32, U64};
    assert!(BwdConfig::supported(cfg.head_dim));
    let hd = cfg.head_dim;
    let bn = cfg.dkdv_block_n();
    let bq = cfg.dkdv_block_q();
    let rw = cfg.row_words();
    let n_dn = (hd / 8) as usize;
    let n_qn = (bq / 8) as usize;
    let threads = cfg.dkdv_threads();
    let names = BwdConfig::dkdv_param_names();

    let mut g = G::new(&cfg.dkdv_kernel_name());
    let q = g.b.add_param(names[0], io_global(cfg.io), AddressSpace::Global);
    let k = g.b.add_param(names[1], io_global(cfg.io), AddressSpace::Global);
    let v = g.b.add_param(names[2], io_global(cfg.io), AddressSpace::Global);
    let dout = g.b.add_param(names[3], io_global(cfg.io), AddressSpace::Global);
    let lse = g.b.add_param(names[4], f32_global(), AddressSpace::Global);
    let dvec = g.b.add_param(names[5], f32_global(), AddressSpace::Global);
    let gmax = g.b.add_param(names[6], u32_global(), AddressSpace::Global);
    let dk = g.b.add_param(names[7], io_global(cfg.io), AddressSpace::Global);
    let dv = g.b.add_param(names[8], io_global(cfg.io), AddressSpace::Global);
    let seq = g.b.add_param(names[9], U32, AddressSpace::Local);
    let heads = g.b.add_param(names[10], U32, AddressSpace::Local);
    let kv_heads = g.b.add_param(names[11], U32, AddressSpace::Local);
    let scale = g.b.add_param(names[12], F32, AddressSpace::Local);

    g.b.set_smem_layout(dkdv_layout(&cfg));
    g.b.set_workgroup_size([threads, 1, 1]);
    g.b.set_launch_bounds(threads, None);

    let entry = g.b.new_block();
    g.b.set_block(entry);
    let reg = |g: &mut G, i: u32, ty: KirType| {
        let d = g.var(ty);
        g.b.emit(KirOp::SharedRegion { dst: d, region: i });
        d
    };
    let k_s = reg(&mut g, 0, h2_shared());
    let v_s = reg(&mut g, 1, h2_shared());
    let q_s = reg(&mut g, 2, h2_shared());
    let do_s = reg(&mut g, 3, h2_shared());
    let lse_s = reg(&mut g, 4, f32_shared());
    let dsc_s = reg(&mut g, 5, f32_shared());

    let tid = g.var(U32);
    g.b.emit(KirOp::ThreadId(tid, 0));
    let ln = Lane::new(&mut g, tid);
    let kt = g.var(U32);
    g.b.emit(KirOp::BlockIdx(kt, 0));
    let bkv = g.var(U32);
    g.b.emit(KirOp::BlockIdx(bkv, 1));
    let one = g.u32c(1);
    let k0 = g.mul_u32c(kt, bn);
    let b_idx = g.div(bkv, kv_heads);
    let kvh = g.rem(bkv, kv_heads);
    let group = g.div(heads, kv_heads);
    // first query head of the group, as a batch·head index
    let bh_first = {
        let bhd = g.mul(b_idx, heads);
        let kg = g.mul(kvh, group);
        g.add(bhd, kg)
    };

    let seq64 = g.cast(seq, U64);
    let hd64 = g.u64c(hd as u64);
    let head_elems = g.mul(seq64, hd64);
    let bkv64 = g.cast(bkv, U64);
    let kv_off = g.mul(bkv64, head_elems);
    let k_base = g.ptr(k, kv_off);
    let v_base = g.ptr(v, kv_off);
    let dk_base = g.ptr(dk, kv_off);
    let dv_base = g.ptr(dv, kv_off);
    let seq_m1 = g.sub(seq, one);

    // The group's prescale: the largest |dO| over its query heads.
    let (gs, ginv) = {
        let head_loop = g.b.new_block();
        let head_body = g.b.new_block();
        let head_done = g.b.new_block();
        let i = g.b.add_block_param(head_loop, U32);
        let acc = g.b.add_block_param(head_loop, U32);
        let zero_u = g.u32c(0);
        g.b.terminate(KirTerminator::Branch(KirEdge::with(head_loop, vec![zero_u, zero_u])));
        g.b.set_block(head_loop);
        let fin = g.cmp(i, group, CmpOp::Ge);
        g.b.terminate(KirTerminator::CondBranch(fin, KirEdge::to(head_done), KirEdge::to(head_body)));
        g.b.set_block(head_body);
        let hi = g.add(bh_first, i);
        let p = g.ptr(gmax, hi);
        let w = g.load(p, AddressSpace::Global);
        let m = g.max(acc, w);
        let i1 = g.add(i, one);
        g.b.terminate(KirTerminator::Branch(KirEdge::with(head_loop, vec![i1, m])));
        g.b.set_block(head_done);
        grad_scale_from_max_bits(&mut g, acc)
    };

    // Stage this CTA's K and V tiles once.
    {
        let tm_kv = TileMap::new(&mut g, tid, hd, bn, threads);
        let kv_vals = tm_kv.load(&mut g, k_base, k0, seq, seq_m1, hd, None);
        tm_kv.store(&mut g, k_s, &kv_vals, rw);
        let vv = tm_kv.load(&mut g, v_base, k0, seq, seq_m1, hd, None);
        tm_kv.store(&mut g, v_s, &vv, rw);
    }

    let l2e = g.f32c(LOG2_E);
    let neg_l2e = g.f32c(-LOG2_E);
    let sl2 = g.mul(scale, l2e);
    let cols4 = ln.ldm_cols4_off(&mut g, rw);
    let rows2 = ln.ldm_rows2_off(&mut g, rw);
    // This warp's 16 keys as A fragments come from rows 16·warp.
    let wrow_words = g.mul_u32c(ln.warp, 16 * rw);
    let a_off = g.add(rows2, wrow_words);
    let k_a = g.ptr(k_s, a_off);
    let v_a = g.ptr(v_s, a_off);
    let q_rows = g.ptr(q_s, cols4);
    let q_trans = g.ptr(q_s, rows2);
    let do_rows = g.ptr(do_s, cols4);
    let do_trans = g.ptr(do_s, rows2);
    let c2 = g.mul_u32c(ln.lc, 2);
    let lse_lane = g.ptr(lse_s, c2);
    let dsc_lane = g.ptr(dsc_s, c2);

    let wk0 = {
        let w16 = g.mul_u32c(ln.warp, 16);
        g.add(k0, w16)
    };
    let key_lo = g.add(wk0, ln.lr);
    let key_hi = g.add_u32c(key_lo, 8);
    let wk_last = g.add_u32c(wk0, 15);

    // Query tiles: causal starts at the tile holding key k0.
    let n_qt = {
        let s = g.add_u32c(seq, bq - 1);
        let sh = g.u32c(bq.trailing_zeros());
        g.shr(s, sh)
    };
    let qt_start = if cfg.causal {
        let sh = g.u32c(bq.trailing_zeros());
        g.shr(k0, sh)
    } else {
        g.u32c(0)
    };
    let nq = g.sub(n_qt, qt_start);
    let total = g.mul(nq, group);
    let total_m1 = g.sub(total, one);

    let tm = TileMap::new(&mut g, tid, hd, bq, threads);
    let neg_inf = g.f32c(f32::NEG_INFINITY);
    let zero_f = g.f32c(0.0);
    let zero_u = g.u32c(0);
    let bq_c = g.u32c(bq);
    let t_small = g.cmp(tid, bq_c, CmpOp::Lt);
    let bq_m1 = g.u32c(bq - 1);
    let tid_c = g.min(tid, bq_m1);

    // Loads for flat iteration `it`: (Q tile, gs·dO tile, lse, D).
    let load_tile = |g: &mut G, it: VarId| -> (Vec<VarId>, Vec<VarId>, VarId, VarId, VarId) {
        let gi = g.div(it, nq);
        let r = g.rem(it, nq);
        let qt = g.add(qt_start, r);
        let q0 = g.mul_u32c(qt, bq);
        let bhq = g.add(bh_first, gi);
        let bhq64 = g.cast(bhq, U64);
        let off = g.mul(bhq64, head_elems);
        let qb = g.ptr(q, off);
        let db = g.ptr(dout, off);
        let qv = tm.load(g, qb, q0, seq, seq_m1, hd, None);
        let dov = tm.load(g, db, q0, seq, seq_m1, hd, Some(gs));
        // lse and D of row q0 + tid (threads below bq).
        let row = g.add(q0, tid_c);
        let valid = g.cmp(row, seq, CmpOp::Lt);
        let rc = g.min(row, seq_m1);
        let roff64 = g.mul(bhq64, seq64);
        let rc64 = g.cast(rc, U64);
        let ro = g.add(roff64, rc64);
        let pl = g.ptr(lse, ro);
        let lv = g.load(pl, AddressSpace::Global);
        let lv = g.mul(lv, neg_l2e);
        let lv = g.sel(valid, lv, zero_f);
        let pd = g.ptr(dvec, ro);
        let dvv = g.load(pd, AddressSpace::Global);
        let dvv = g.mul(dvv, gs);
        let dvv = g.sel(valid, dvv, zero_f);
        (qv, dov, lv, dvv, q0)
    };

    let (pq0, pdo0, plse0, pd0, _) = load_tile(&mut g, zero_u);

    let header = g.b.new_block();
    let body = g.b.new_block();
    let exit = g.b.new_block();
    let it = g.b.add_block_param(header, U32);
    let mut accs: Vec<[VarId; 4]> = Vec::new(); // dK n-tiles then dV n-tiles
    for _ in 0..2 * n_dn {
        accs.push([
            g.b.add_block_param(header, F32),
            g.b.add_block_param(header, F32),
            g.b.add_block_param(header, F32),
            g.b.add_block_param(header, F32),
        ]);
    }
    let pq: Vec<VarId> = (0..pq0.len()).map(|_| g.b.add_block_param(header, F32)).collect();
    let pdo: Vec<VarId> = (0..pdo0.len()).map(|_| g.b.add_block_param(header, F32)).collect();
    let plse = g.b.add_block_param(header, F32);
    let pd = g.b.add_block_param(header, F32);
    let mut init = vec![zero_u];
    init.extend(std::iter::repeat_n(zero_f, 2 * n_dn * 4));
    init.extend(&pq0);
    init.extend(&pdo0);
    init.extend([plse0, pd0]);
    g.b.terminate(KirTerminator::Branch(KirEdge::with(header, init)));

    g.b.set_block(header);
    let done = g.cmp(it, total, CmpOp::Ge);
    g.b.terminate(KirTerminator::CondBranch(done, KirEdge::to(exit), KirEdge::to(body)));

    g.b.set_block(body);
    g.b.emit(KirOp::Barrier);
    tm.store(&mut g, q_s, &pq, rw);
    tm.store(&mut g, do_s, &pdo, rw);
    let pls = g.ptr(lse_s, tid_c);
    g.when(t_small, KirOp::Store(pls, plse, AddressSpace::Shared));
    let pds = g.ptr(dsc_s, tid_c);
    g.when(t_small, KirOp::Store(pds, pd, AddressSpace::Shared));
    g.b.emit(KirOp::Barrier);
    let it1 = g.add(it, one);
    let itn = g.min(it1, total_m1);
    let (nq_v, ndo_v, nl, nd, _) = load_tile(&mut g, itn);
    // This iteration's first query row.
    let q0 = {
        let r = g.rem(it, nq);
        let qt = g.add(qt_start, r);
        g.mul_u32c(qt, bq)
    };
    let flat: Vec<VarId> = accs.iter().flatten().copied().collect();
    let next_args = |acc: &[VarId]| -> Vec<VarId> {
        let mut a = vec![it1];
        a.extend_from_slice(acc);
        a.extend(&nq_v);
        a.extend(&ndo_v);
        a.extend([nl, nd]);
        a
    };
    let masked_blk = if cfg.causal { g.b.new_block() } else { 0 };
    let plain_blk = g.b.new_block();
    if cfg.causal {
        // Every query of the tile is before every key of this warp.
        let q_last = g.add_u32c(q0, bq - 1);
        let skip = g.cmp(q_last, wk0, CmpOp::Lt);
        let check = g.b.new_block();
        g.b.terminate(KirTerminator::CondBranch(skip, KirEdge::with(header, next_args(&flat)), KirEdge::to(check)));
        g.b.set_block(check);
        let needs_mask = g.cmp(q0, wk_last, CmpOp::Lt);
        g.b.terminate(KirTerminator::CondBranch(needs_mask, KirEdge::to(masked_blk), KirEdge::to(plain_blk)));
    } else {
        // Rows and keys past `seq` were zero-filled: their terms vanish.
        g.b.terminate(KirTerminator::Branch(KirEdge::to(plain_blk)));
    }

    let paths: &[(_, bool)] =
        if cfg.causal { &[(masked_blk, true), (plain_blk, false)] } else { &[(plain_blk, false)] };
    for &(blk, masked) in paths {
        g.b.set_block(blk);
        let z = zero_f;
        let ka: Vec<[VarId; 4]> = (0..hd / 16).map(|t| g_ldm_a(&mut g, k_a, t * 8)).collect();
        // Sᵀ = K·Qᵀ: rows are this warp's keys, columns the tile's queries.
        let mut st: Vec<[VarId; 4]> = vec![[z, z, z, z]; n_qn];
        mma_b_rows(&mut g, &mut st, &ka, q_rows, rw);
        if masked {
            let qcol0 = g.add(q0, c2);
            for (n, acc) in st.iter_mut().enumerate() {
                for e in 0..2u32 {
                    let qcol = g.add_u32c(qcol0, n as u32 * 8 + e);
                    for (half, &key) in [key_lo, key_hi].iter().enumerate() {
                        let idx = half * 2 + e as usize;
                        let future = g.cmp(key, qcol, CmpOp::Gt);
                        acc[idx] = g.sel(future, neg_inf, acc[idx]);
                    }
                }
            }
        }
        // Pᵀ = 2^(Sᵀ·sl2 - lse·log2 e[column]) (the tile holds -lse·log2 e).
        let mut cols_l = Vec::with_capacity(n_qn);
        let mut cols_d = Vec::with_capacity(n_qn);
        for n in 0..n_qn {
            let pl = g.ptr_c(lse_lane, n as u32 * 8);
            cols_l.push(g.load_vec(pl, 2, AddressSpace::Shared));
            let pd = g.ptr_c(dsc_lane, n as u32 * 8);
            cols_d.push(g.load_vec(pd, 2, AddressSpace::Shared));
        }
        for (n, acc) in st.iter_mut().enumerate() {
            for (idx, slot) in acc.iter_mut().enumerate() {
                let arg = g.fma(*slot, sl2, cols_l[n][idx % 2]);
                *slot = g.exp2(arg);
            }
        }
        let (dk_acc, dv_acc) = accs.split_at(n_dn);
        let mut dk_acc = dk_acc.to_vec();
        let mut dv_acc = dv_acc.to_vec();
        // dV += Pᵀ·(gs·dO)
        let pa = acc_to_a_frags(&mut g, &st);
        mma_b_trans(&mut g, &mut dv_acc, &pa, do_trans, rw);
        // dPᵀ = V·(gs·dO)ᵀ
        let va: Vec<[VarId; 4]> = (0..hd / 16).map(|t| g_ldm_a(&mut g, v_a, t * 8)).collect();
        let mut dpt: Vec<[VarId; 4]> = vec![[z, z, z, z]; n_qn];
        mma_b_rows(&mut g, &mut dpt, &va, do_rows, rw);
        // dSᵀ = Pᵀ ∘ (dPᵀ - gs·D[column])
        let ds: Vec<[VarId; 4]> = st
            .iter()
            .zip(&dpt)
            .enumerate()
            .map(|(n, (p, d))| {
                let mut out = [0; 4];
                for idx in 0..4 {
                    let t = g.sub(d[idx], cols_d[n][idx % 2]);
                    let x = g.mul(p[idx], t);
                    out[idx] = clamp_f16(&mut g, x);
                }
                out
            })
            .collect();
        // dK += dSᵀ·Q
        let dsa = acc_to_a_frags(&mut g, &ds);
        mma_b_trans(&mut g, &mut dk_acc, &dsa, q_trans, rw);
        let flat_new: Vec<VarId> = dk_acc.iter().chain(dv_acc.iter()).flatten().copied().collect();
        g.b.terminate(KirTerminator::Branch(KirEdge::with(header, next_args(&flat_new))));
    }

    // dK = scale/gs · acc, dV = 1/gs · acc, rows (keys) below seq.
    g.b.set_block(exit);
    let fk = g.mul(scale, ginv);
    for (half, &key) in [key_lo, key_hi].iter().enumerate() {
        let valid = g.cmp(key, seq, CmpOp::Lt);
        let roff = g.mul_u32c(key, hd);
        let roff = g.add(roff, c2);
        let pk = g.ptr(dk_base, roff);
        let pv = g.ptr(dv_base, roff);
        for n in 0..n_dn {
            let a = accs[n];
            let x0 = g.mul(a[half * 2], fk);
            let x1 = g.mul(a[half * 2 + 1], fk);
            let p = g.ptr_c(pk, n as u32 * 8);
            let vals = g.narrow_to(p, vec![x0, x1]);
            g.when(valid, KirOp::StoreVec { ptr: p, vals, space: AddressSpace::Global });
            let b = accs[n_dn + n];
            let y0 = g.mul(b[half * 2], ginv);
            let y1 = g.mul(b[half * 2 + 1], ginv);
            let p = g.ptr_c(pv, n as u32 * 8);
            let vals = g.narrow_to(p, vec![y0, y1]);
            g.when(valid, KirOp::StoreVec { ptr: p, vals, space: AddressSpace::Global });
        }
    }
    g.b.terminate(KirTerminator::Return);
    g.b.finalize()
}

/// An m16k16 A fragment from a row-major shared tile: one non-transposed
/// `ldmatrix.x4` at `base + word_off` (`base` already holds the lane's
/// rows-2 offset).
fn g_ldm_a(g: &mut G, base: VarId, word_off: u32) -> [VarId; 4] {
    let addr = g.ptr_c(base, word_off);
    g.ldmatrix_x4(addr, false)
}

pub fn pre_ptx(cfg: BwdConfig) -> Vec<u8> {
    super::super::elementwise::verified_ptx(build_pre(cfg))
}

pub fn dq_ptx(cfg: BwdConfig) -> Vec<u8> {
    super::super::elementwise::verified_ptx(build_dq(cfg))
}

pub fn dkdv_ptx(cfg: BwdConfig) -> Vec<u8> {
    super::super::elementwise::verified_ptx(build_dkdv(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_backward_variant_verifies_and_lowers() {
        for hd in BwdConfig::HEAD_DIMS {
            for causal in [false, true] {
                let cfg = BwdConfig::new(hd, causal);
                for (name, ir) in [
                    (cfg.pre_kernel_name(), build_pre(cfg)),
                    (cfg.dq_kernel_name(), build_dq(cfg)),
                    (cfg.dkdv_kernel_name(), build_dkdv(cfg)),
                ] {
                    assert_eq!(ir.verify(), Ok(()), "{name}");
                }
                let pre = String::from_utf8(pre_ptx(cfg)).unwrap();
                assert!(pre.contains("red.global.max.u32"), "{pre}");
                let dq = String::from_utf8(dq_ptx(cfg)).unwrap();
                // two paths x three products x n-tiles x k-steps
                let per = (cfg.dq_block_n() / 8) * (hd / 16);
                assert_eq!(dq.matches("mma.sync.aligned").count() as u32, 2 * 3 * per);
                let dkdv = String::from_utf8(dkdv_ptx(cfg)).unwrap();
                let per = (cfg.dkdv_block_q() / 8) * (hd / 16);
                let paths = if causal { 2 } else { 1 };
                assert_eq!(dkdv.matches("mma.sync.aligned").count() as u32, paths * 4 * per);
                assert!(dkdv.contains(".extern .shared"));
                assert!(cfg.dkdv_smem_bytes() <= crate::kernel_ir::SMEM_DYNAMIC_BUDGET);
            }
        }
    }
}
