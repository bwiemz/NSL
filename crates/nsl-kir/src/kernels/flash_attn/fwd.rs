// crates/nsl-kir/src/kernels/flash_attn/fwd.rs
//! The flash-attention forward (FlashAttention-2 schedule) on tensor cores.
//!
//! One CTA of `warps` warps per `(query tile, batch·head)`; each warp owns
//! 16 query rows, so a CTA covers `BLOCK_M = 16 · warps` rows. The
//! CTA walks the key/value tiles of `BLOCK_N` rows (causal: only up to its
//! last query row), and per tile each warp computes
//!
//! ```text
//!   S = Q · Kᵀ                      (mma, f16 operands, f32 accumulators)
//!   m' = max(m, rowmax S);  α = 2^((m - m')·scale·log2 e)
//!   P = 2^(S·scale·log2 e - m'·scale·log2 e)
//!   l = α·l + rowsum P;   O = α·O + P · V   (P rounded to f16)
//! ```
//!
//! and at the end stores `O / l` (f32) and `lse = (m·scale·log2 e +
//! log2 l) · ln 2` (the natural log-sum-exp of the scaled scores).
//!
//! `Q` lives in registers as A fragments for the whole kernel. `K` and `V`
//! tiles are read from global memory as f32 (`ld.global.v4`) one tile ahead
//! (the loads for tile `j + 1` are in flight while tile `j` is computed),
//! rounded to f16 and stored to shared memory, from where `ldmatrix` feeds
//! the B fragments (`K` as stored, `V` transposed). `P` goes from the score
//! accumulators straight to A fragments (`CvtPack`), never through shared
//! memory: an m16n8 accumulator pair `(c0, c1)` / `(c2, c3)` of two
//! adjacent n-tiles is exactly the m16k16 A fragment of the next product.
//!
//! The grid's x runs over query tiles, heaviest first under causal masking,
//! y over `batch · heads`; the launch covers the whole batch.

use super::{f32_global, h2, h2_shared, G, LN_2, LOG2_E, SMEM_ROW_PAD};
use crate::kernel_ir::{
    AddressSpace, CmpOp, KernelIR, KirEdge, KirOp, KirTerminator, KirType, SmemLayout,
    SmemRegion, VarId,
};

/// One forward variant: the problem (`head_dim`, `causal`) and the tiling
/// (`warps` of 16 query rows each per CTA, `block_n` keys per tile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FwdConfig {
    pub head_dim: u32,
    pub causal: bool,
    pub warps: u32,
    pub block_n: u32,
}

impl FwdConfig {
    /// Head dims these kernels are built for.
    pub const HEAD_DIMS: [u32; 2] = [64, 128];

    pub fn supported(head_dim: u32) -> bool {
        Self::HEAD_DIMS.contains(&head_dim)
    }

    /// The tiling the runtime launches (chosen by measurement, see
    /// `tc_attention_kernel_times` in nsl-runtime's `attention_tc.rs`).
    pub fn new(head_dim: u32, causal: bool) -> FwdConfig {
        let (warps, block_n) = default_fwd_tiling(head_dim);
        FwdConfig { head_dim, causal, warps, block_n }
    }

    pub fn block_m(&self) -> u32 {
        16 * self.warps
    }

    pub fn block_n(&self) -> u32 {
        self.block_n
    }

    pub fn threads(&self) -> u32 {
        32 * self.warps
    }

    /// The `.visible .entry` name; the launcher launches by it.
    pub fn kernel_name(&self) -> String {
        format!(
            "nsl_fa2_fwd_d{}_{}_w{}_n{}",
            self.head_dim,
            if self.causal { "causal" } else { "full" },
            self.warps,
            self.block_n
        )
    }

    /// `[query tiles, batch · heads, 1]`.
    pub fn grid(&self, batch: u32, heads: u32, seq: u32) -> [u32; 3] {
        [seq.div_ceil(self.block_m()), batch * heads, 1]
    }

    /// Parameter names in launch order: `q, k, v, o, lse` are f32 device
    /// pointers, `seq, heads, kv_heads` u32, `scale` f32.
    pub fn param_names() -> [&'static str; 9] {
        ["q", "k", "v", "o", "lse", "seq", "heads", "kv_heads", "scale"]
    }

    fn row_words(&self) -> u32 {
        (self.head_dim + SMEM_ROW_PAD) / 2
    }
}

/// `(warps, block_n)` per head_dim. Measured on an RTX PRO 4500 (sm_120)
/// at Coder-1B's shape (batch 2, 32 heads, seq 2048, causal): head_dim 64
/// 0.42 ms at (8, 32) against 0.47-0.67 for (4 or 8, 32 / 64 / 128);
/// head_dim 128 0.47 ms at (8, 64), (8, 32) 0.48.
fn default_fwd_tiling(head_dim: u32) -> (u32, u32) {
    if head_dim > 64 { (8, 64) } else { (8, 32) }
}

/// The per-thread mapping of a `rows x head_dim` f32 tile onto float4
/// chunks: thread `t` loads chunk `t + i·threads` for `i < chunks`, i.e.
/// row `t / (hd/4) + i · row_step`, column `(t % (hd/4)) · 4`.
pub(crate) struct TileMap {
    pub chunks: u32,
    pub row_step: u32,
    pub t_row: VarId,
    pub t_col: VarId,
}

impl TileMap {
    pub fn new(g: &mut G, tid: VarId, head_dim: u32, rows: u32, threads: u32) -> TileMap {
        let per_row = head_dim / 4;
        assert!(per_row.is_power_of_two() && threads % per_row == 0);
        let chunks = rows * per_row / threads;
        assert!(chunks >= 1 && (rows * per_row) % threads == 0, "tile {rows}x{head_dim} on {threads} threads");
        let sh = g.u32c(per_row.trailing_zeros());
        let t_row = g.shr(tid, sh);
        let m = g.u32c(per_row - 1);
        let tc = g.and(tid, m);
        let t_col = g.mul_u32c(tc, 4);
        TileMap { chunks, row_step: threads / per_row, t_row, t_col }
    }

    /// Load the tile whose first row is `row0` from `base` (a `[seq,
    /// head_dim]` f32 matrix): rows at or past `seq` read row `seq - 1`'s
    /// address and are replaced by zeros. Returns `chunks · 4` values,
    /// optionally each multiplied by `scale`.
    pub fn load(
        &self,
        g: &mut G,
        base: VarId,
        row0: VarId,
        seq: VarId,
        seq_m1: VarId,
        head_dim: u32,
        scale: Option<VarId>,
    ) -> Vec<VarId> {
        let mut out = Vec::with_capacity(self.chunks as usize * 4);
        let zero = g.f32c(0.0);
        let r = g.add(row0, self.t_row);
        for i in 0..self.chunks {
            let row = g.add_u32c(r, i * self.row_step);
            let valid = g.cmp(row, seq, CmpOp::Lt);
            let rc = g.min(row, seq_m1);
            let off = g.mul_u32c(rc, head_dim);
            let off = g.add(off, self.t_col);
            let p = g.ptr(base, off);
            let vals = g.load_vec(p, 4, AddressSpace::Global);
            for v in vals {
                let v = match scale {
                    Some(s) => g.mul(v, s),
                    None => v,
                };
                out.push(g.sel(valid, v, zero));
            }
        }
        out
    }

    /// Round loaded values to f16 and store them into the shared tile at
    /// `smem` (row stride `row_words` packed pairs).
    pub fn store(&self, g: &mut G, smem: VarId, vals: &[VarId], row_words: u32) {
        // word offset of this thread's first chunk: t_row·row_words + t_col/2
        let ro = g.mul_u32c(self.t_row, row_words);
        let one = g.u32c(1);
        let co = g.shr(self.t_col, one);
        let off = g.add(ro, co);
        let lane_base = g.ptr(smem, off);
        for i in 0..self.chunks {
            let p = g.ptr_c(lane_base, i * self.row_step * row_words);
            let c = &vals[i as usize * 4..i as usize * 4 + 4];
            let lo = g.pack(c[0], c[1]);
            let hi = g.pack(c[2], c[3]);
            g.store_vec(p, vec![lo, hi], AddressSpace::Shared);
        }
    }
}

/// Lane-derived constants every warp-level tile op uses.
pub(crate) struct Lane {
    pub lane: VarId,
    pub warp: VarId,
    /// `lane / 4`: the accumulator row (and `+ 8`) this lane holds.
    pub lr: VarId,
    /// `lane % 4`: the accumulator column pair `2·lc, 2·lc + 1`.
    pub lc: VarId,
}

impl Lane {
    pub fn new(g: &mut G, tid: VarId) -> Lane {
        let m31 = g.u32c(31);
        let lane = g.and(tid, m31);
        let five = g.u32c(5);
        let warp = g.shr(tid, five);
        let two = g.u32c(2);
        let lr = g.shr(lane, two);
        let m3 = g.u32c(3);
        let lc = g.and(lane, m3);
        Lane { lane, warp, lr, lc }
    }

    /// Per-lane word offset of the `ldmatrix.x4` row address for a tile
    /// stored row-major with `row_words` packed pairs per row, when the
    /// four 8x8 matrices are `(rows 0-7, cols 0-7)`, `(rows 0-7, cols
    /// 8-15)`, `(rows 0-7, cols 16-23)`, `(rows 0-7, cols 24-31)`: lane `l`
    /// addresses row `l % 8`, column `8 · (l / 8)`. This is the B-fragment
    /// pair of two consecutive k-steps of one n-tile (non-transposed load).
    pub fn ldm_cols4_off(&self, g: &mut G, row_words: u32) -> VarId {
        let m7 = g.u32c(7);
        let r = g.and(self.lane, m7);
        let three = g.u32c(3);
        let mi = g.shr(self.lane, three);
        let ro = g.mul_u32c(r, row_words);
        let co = g.mul_u32c(mi, 4); // 8 halves = 4 words
        g.add(ro, co)
    }

    /// Per-lane word offset when the four matrices are `(rows 0-7, cols
    /// 0-7)`, `(rows 8-15, cols 0-7)`, `(rows 0-7, cols 8-15)`, `(rows 8-15,
    /// cols 8-15)`: lane `l` addresses row `l % 8 + 8 · ((l / 8) % 2)`,
    /// column `8 · (l / 16)`. Non-transposed, this is an m16k16 A fragment;
    /// transposed, it is the B-fragment pair (k-rows 0-15) of two adjacent
    /// n-tiles of a row-major `[k][n]` tile.
    pub fn ldm_rows2_off(&self, g: &mut G, row_words: u32) -> VarId {
        let m7 = g.u32c(7);
        let r = g.and(self.lane, m7);
        let three = g.u32c(3);
        let mi = g.shr(self.lane, three);
        let one = g.u32c(1);
        let half = g.and(mi, one);
        let eight = g.mul_u32c(half, 8);
        let row = g.add(r, eight);
        let ro = g.mul_u32c(row, row_words);
        let four = g.u32c(4);
        let cblk = g.shr(self.lane, four);
        let co = g.mul_u32c(cblk, 4);
        g.add(ro, co)
    }
}

/// Load `rows` (a lane's `lo` and `hi` accumulator rows, already clamped
/// into the tensor) of a `[seq, head_dim]` f32 matrix at `base` as m16k16
/// A fragments, one per k-step of 16 columns, each value multiplied by
/// `scale` when given.
pub(crate) fn load_a_frags(
    g: &mut G,
    base: VarId,
    row_lo: VarId,
    row_hi: VarId,
    lc: VarId,
    head_dim: u32,
    scale: Option<(VarId, VarId)>,
) -> Vec<[VarId; 4]> {
    let lo_off = g.mul_u32c(row_lo, head_dim);
    let hi_off = g.mul_u32c(row_hi, head_dim);
    let c2 = g.mul_u32c(lc, 2);
    let lo_off = g.add(lo_off, c2);
    let hi_off = g.add(hi_off, c2);
    let p_lo = g.ptr(base, lo_off);
    let p_hi = g.ptr(base, hi_off);
    let mut frags = Vec::new();
    for t in 0..head_dim / 16 {
        let mut regs = [0; 4];
        for (r, (p, col)) in [(p_lo, 0u32), (p_hi, 0), (p_lo, 8), (p_hi, 8)].into_iter().enumerate() {
            let pp = g.ptr_c(p, 16 * t + col);
            let v = g.load_vec(pp, 2, AddressSpace::Global);
            let (x0, x1) = match scale {
                Some((s_lo, s_hi)) => {
                    let s = if r % 2 == 0 { s_lo } else { s_hi };
                    (g.mul(v[0], s), g.mul(v[1], s))
                }
                None => (v[0], v[1]),
            };
            regs[r] = g.pack(x0, x1);
        }
        frags.push(regs);
    }
    frags
}

/// The m16k16 A fragments of the next product made from score-shaped
/// accumulators `acc[n-tile][0..4]` (16 rows x 8·n columns): k-step `u`
/// takes n-tiles `2u` and `2u + 1`.
pub(crate) fn acc_to_a_frags(g: &mut G, acc: &[[VarId; 4]]) -> Vec<[VarId; 4]> {
    (0..acc.len() / 2)
        .map(|u| {
            let (a, b) = (acc[2 * u], acc[2 * u + 1]);
            [g.pack(a[0], a[1]), g.pack(a[2], a[3]), g.pack(b[0], b[1]), g.pack(b[2], b[3])]
        })
        .collect()
}

/// `acc[n] += A · B` over k-steps where the B fragments of n-tiles `n,
/// n + 1` at k-step `t` come from one `ldmatrix.x4.trans` at
/// `base + rows2_off + t·16·row_words + n·4` words: a row-major `[k][n]`
/// tile in shared memory (`V` in `P·V`).
pub(crate) fn mma_b_trans(
    g: &mut G,
    acc: &mut [[VarId; 4]],
    a: &[[VarId; 4]],
    base: VarId,
    row_words: u32,
) {
    for (t, at) in a.iter().enumerate() {
        for n in (0..acc.len()).step_by(2) {
            let addr = g.ptr_c(base, t as u32 * 16 * row_words + n as u32 * 4);
            let b = g.ldmatrix_x4(addr, true);
            acc[n] = g.mma(*at, [b[0], b[1]], acc[n]);
            acc[n + 1] = g.mma(*at, [b[2], b[3]], acc[n + 1]);
        }
    }
}

/// `acc[n] += A · Bᵀ` where `B` is a row-major `[n][k]` tile in shared
/// memory (`K` in `Q·Kᵀ`): the fragments of n-tile `n` at k-steps `t, t +
/// 1` come from one non-transposed `ldmatrix.x4` at `base + cols4_off +
/// n·8·row_words + t·8` words.
pub(crate) fn mma_b_rows(
    g: &mut G,
    acc: &mut [[VarId; 4]],
    a: &[[VarId; 4]],
    base: VarId,
    row_words: u32,
) {
    for t in (0..a.len()).step_by(2) {
        for (n, slot) in acc.iter_mut().enumerate() {
            let addr = g.ptr_c(base, n as u32 * 8 * row_words + t as u32 * 8);
            let b = g.ldmatrix_x4(addr, false);
            *slot = g.mma(a[t], [b[0], b[1]], *slot);
            *slot = g.mma(a[t + 1], [b[2], b[3]], *slot);
        }
    }
}

/// Build one forward variant.
pub fn build(cfg: FwdConfig) -> KernelIR {
    use KirType::{F32, U32, U64};
    assert!(FwdConfig::supported(cfg.head_dim), "head_dim {}", cfg.head_dim);
    let hd = cfg.head_dim;
    let bm = cfg.block_m();
    let bn = cfg.block_n();
    let rw = cfg.row_words();
    let n_dn = (hd / 8) as usize; // output n-tiles
    let n_sn = (bn / 8) as usize; // score n-tiles
    let names = FwdConfig::param_names();

    let mut g = G::new(&cfg.kernel_name());
    let q = g.b.add_param(names[0], f32_global(), AddressSpace::Global);
    let k = g.b.add_param(names[1], f32_global(), AddressSpace::Global);
    let v = g.b.add_param(names[2], f32_global(), AddressSpace::Global);
    let o = g.b.add_param(names[3], f32_global(), AddressSpace::Global);
    let lse = g.b.add_param(names[4], f32_global(), AddressSpace::Global);
    let seq = g.b.add_param(names[5], U32, AddressSpace::Local);
    let heads = g.b.add_param(names[6], U32, AddressSpace::Local);
    let kv_heads = g.b.add_param(names[7], U32, AddressSpace::Local);
    let scale = g.b.add_param(names[8], F32, AddressSpace::Local);

    let tile_bytes = bn * rw * 4;
    g.b.set_smem_layout(SmemLayout {
        regions: vec![
            SmemRegion { name: "k".into(), bytes: tile_bytes, align: 16, elem: h2() },
            SmemRegion { name: "v".into(), bytes: tile_bytes, align: 16, elem: h2() },
        ],
        dynamic: false,
    });
    g.b.set_workgroup_size([cfg.threads(), 1, 1]);
    g.b.set_launch_bounds(cfg.threads(), None);

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

    // Query tile (heaviest first when causal).
    let n_qt = {
        let s = g.add_u32c(seq, bm - 1);
        let sh = g.u32c(bm.trailing_zeros());
        g.shr(s, sh)
    };
    let q_tile = if cfg.causal {
        let one = g.u32c(1);
        let last = g.sub(n_qt, one);
        g.sub(last, qt_raw)
    } else {
        qt_raw
    };
    let q0 = g.mul_u32c(q_tile, bm);

    // Heads: kv head = h / (heads / kv_heads).
    let b_idx = g.div(bh, heads);
    let h = g.rem(bh, heads);
    let group = g.div(heads, kv_heads);
    let kvh = g.div(h, group);
    let kvb = g.mul(b_idx, kv_heads);
    let kv_bh = g.add(kvb, kvh);

    // Per-head base pointers (64-bit element offsets).
    let seq64 = g.cast(seq, U64);
    let hd64 = g.u64c(hd as u64);
    let head_elems = g.mul(seq64, hd64);
    let bh64 = g.cast(bh, U64);
    let kvbh64 = g.cast(kv_bh, U64);
    let q_off = g.mul(bh64, head_elems);
    let kv_off = g.mul(kvbh64, head_elems);
    let q_base = g.ptr(q, q_off);
    let o_base = g.ptr(o, q_off);
    let k_base = g.ptr(k, kv_off);
    let v_base = g.ptr(v, kv_off);
    let lse_off = g.mul(bh64, seq64);
    let lse_base = g.ptr(lse, lse_off);

    let one = g.u32c(1);
    let seq_m1 = g.sub(seq, one);
    let wrow0 = {
        let w16 = g.mul_u32c(ln.warp, 16);
        g.add(q0, w16)
    };
    let r_lo = g.add(wrow0, ln.lr);
    let r_hi = g.add_u32c(r_lo, 8);
    let r_lo_c = g.min(r_lo, seq_m1);
    let r_hi_c = g.min(r_hi, seq_m1);

    let sl2 = {
        let l2e = g.f32c(LOG2_E);
        g.mul(scale, l2e)
    };

    // Q as A fragments for the whole kernel.
    let qa = load_a_frags(&mut g, q_base, r_lo_c, r_hi_c, ln.lc, hd, None);

    // KV tile count.
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

    let tm = TileMap::new(&mut g, tid, hd, bn, cfg.threads());
    let k_ldm = {
        let off = ln.ldm_cols4_off(&mut g, rw);
        g.ptr(k_s, off)
    };
    let v_ldm = {
        let off = ln.ldm_rows2_off(&mut g, rw);
        g.ptr(v_s, off)
    };
    let key_lane = g.mul_u32c(ln.lc, 2);
    let neg_inf = g.f32c(f32::NEG_INFINITY);
    let zero_f = g.f32c(0.0);
    let warp_last_row = g.add_u32c(wrow0, 15);

    // Prologue: tile 0 in flight.
    let zero_u = g.u32c(0);
    let pf_k0 = tm.load(&mut g, k_base, zero_u, seq, seq_m1, hd, None);
    let pf_v0 = tm.load(&mut g, v_base, zero_u, seq, seq_m1, hd, None);

    // header(j, o[n_dn][4], m_lo, m_hi, l_lo, l_hi, pf_k[..], pf_v[..])
    let header = g.b.new_block();
    let body = g.b.new_block();
    let exit = g.b.new_block();
    let j = g.b.add_block_param(header, U32);
    let mut o_acc: Vec<[VarId; 4]> = Vec::new();
    for _ in 0..n_dn {
        o_acc.push([
            g.b.add_block_param(header, F32),
            g.b.add_block_param(header, F32),
            g.b.add_block_param(header, F32),
            g.b.add_block_param(header, F32),
        ]);
    }
    let m_lo = g.b.add_block_param(header, F32);
    let m_hi = g.b.add_block_param(header, F32);
    let l_lo = g.b.add_block_param(header, F32);
    let l_hi = g.b.add_block_param(header, F32);
    let pf_k: Vec<VarId> = (0..pf_k0.len()).map(|_| g.b.add_block_param(header, F32)).collect();
    let pf_v: Vec<VarId> = (0..pf_v0.len()).map(|_| g.b.add_block_param(header, F32)).collect();

    let mut init = vec![zero_u];
    init.extend(std::iter::repeat_n(zero_f, n_dn * 4));
    init.extend([neg_inf, neg_inf, zero_f, zero_f]);
    init.extend(&pf_k0);
    init.extend(&pf_v0);
    g.b.terminate(KirTerminator::Branch(KirEdge::with(header, init)));

    g.b.set_block(header);
    let done = g.cmp(j, n_kv, CmpOp::Ge);
    g.b.terminate(KirTerminator::CondBranch(done, KirEdge::to(exit), KirEdge::to(body)));

    // ── body: stage tile j, prefetch tile j + 1, compute ──
    g.b.set_block(body);
    g.b.emit(KirOp::Barrier); // every warp is done reading tile j - 1
    tm.store(&mut g, k_s, &pf_k, rw);
    tm.store(&mut g, v_s, &pf_v, rw);
    g.b.emit(KirOp::Barrier); // tile j is visible
    let j1 = g.add(j, one);
    let jn = g.min(j1, n_kv_m1);
    let kvn = g.mul_u32c(jn, bn);
    let nk = tm.load(&mut g, k_base, kvn, seq, seq_m1, hd, None);
    let nv = tm.load(&mut g, v_base, kvn, seq, seq_m1, hd, None);
    let kv0 = g.mul_u32c(j, bn);

    let flat_o: Vec<VarId> = o_acc.iter().flatten().copied().collect();
    let next_args = |o_new: &[VarId], st: [VarId; 4]| -> Vec<VarId> {
        let mut a = vec![j1];
        a.extend_from_slice(o_new);
        a.extend(st);
        a.extend(&nk);
        a.extend(&nv);
        a
    };

    // Which compute path: skip (causal, the whole tile is above this
    // warp's rows), masked (the tile crosses the diagonal or the end of the
    // sequence), or plain.
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
        let skip_args = next_args(&flat_o, [m_lo, m_hi, l_lo, l_hi]);
        g.b.terminate(KirTerminator::CondBranch(skip, KirEdge::with(header, skip_args), KirEdge::to(check)));
        g.b.set_block(check);
    }
    g.b.terminate(KirTerminator::CondBranch(needs_mask, KirEdge::to(masked_blk), KirEdge::to(plain_blk)));

    let st = Step {
        cfg,
        qa: &qa,
        o_acc: &o_acc,
        m: [m_lo, m_hi],
        l: [l_lo, l_hi],
        k_ldm,
        v_ldm,
        rw,
        n_sn,
        sl2,
        kv0,
        key_lane,
        rows: [r_lo, r_hi],
        seq,
        neg_inf,
        zero_f,
    };
    for (blk, masked) in [(masked_blk, true), (plain_blk, false)] {
        g.b.set_block(blk);
        let (o_new, state) = st.compute(&mut g, masked);
        g.b.terminate(KirTerminator::Branch(KirEdge::with(header, next_args(&o_new, state))));
    }

    // ── exit: normalise and store ──
    g.b.set_block(exit);
    epilogue(&mut g, &o_acc, [m_lo, m_hi], [l_lo, l_hi], [r_lo, r_hi], seq, sl2, o_base, lse_base, ln.lc, hd);
    g.b.terminate(KirTerminator::Return);
    g.b.finalize()
}

/// One tile's computation, shared by the masked and plain paths.
struct Step<'a> {
    cfg: FwdConfig,
    qa: &'a [[VarId; 4]],
    o_acc: &'a [[VarId; 4]],
    m: [VarId; 2],
    l: [VarId; 2],
    k_ldm: VarId,
    v_ldm: VarId,
    rw: u32,
    n_sn: usize,
    sl2: VarId,
    kv0: VarId,
    key_lane: VarId,
    rows: [VarId; 2],
    seq: VarId,
    neg_inf: VarId,
    zero_f: VarId,
}

impl Step<'_> {
    /// Returns the new `O` accumulators (flattened) and `[m_lo, m_hi,
    /// l_lo, l_hi]`.
    fn compute(&self, g: &mut G, masked: bool) -> (Vec<VarId>, [VarId; 4]) {
        let z = self.zero_f;
        let mut s: Vec<[VarId; 4]> = vec![[z, z, z, z]; self.n_sn];
        mma_b_rows(g, &mut s, self.qa, self.k_ldm, self.rw);

        if masked {
            let key0 = g.add(self.kv0, self.key_lane);
            for (n, acc) in s.iter_mut().enumerate() {
                for e in 0..2u32 {
                    let key = g.add_u32c(key0, n as u32 * 8 + e);
                    let oob = g.cmp(key, self.seq, CmpOp::Ge);
                    for (half, &row) in self.rows.iter().enumerate() {
                        let idx = half * 2 + e as usize;
                        let dead = if self.cfg.causal {
                            let future = g.cmp(key, row, CmpOp::Gt);
                            g.or(future, oob)
                        } else {
                            oob
                        };
                        acc[idx] = g.sel(dead, self.neg_inf, acc[idx]);
                    }
                }
            }
        }

        let mut new_m = [0; 2];
        let mut new_l = [0; 2];
        let mut alpha = [0; 2];
        let mut neg_ms = [0; 2];
        for half in 0..2 {
            let vals: Vec<VarId> = s.iter().flat_map(|a| [a[half * 2], a[half * 2 + 1]]).collect();
            let mx = g.max_all(&vals);
            let mx = g.quad_reduce(mx, KirOp::Max);
            let m_new = g.max(self.m[half], mx);
            // The scaled max to subtract; 0 for a row that has seen only
            // masked keys so far (then every term is 2^-inf = 0, not NaN).
            let scaled = g.mul(m_new, self.sl2);
            let is_inf = g.cmp(m_new, self.neg_inf, CmpOp::Eq);
            let ms = g.sel(is_inf, self.zero_f, scaled);
            let old_scaled = g.mul(self.m[half], self.sl2);
            let d = g.sub(old_scaled, ms);
            alpha[half] = g.exp2(d);
            neg_ms[half] = g.neg(ms);
            new_m[half] = m_new;
        }
        // P = 2^(S·sl2 - m'·sl2), in place.
        for acc in s.iter_mut() {
            for (idx, slot) in acc.iter_mut().enumerate() {
                let arg = g.fma(*slot, self.sl2, neg_ms[idx / 2]);
                *slot = g.exp2(arg);
            }
        }
        for half in 0..2 {
            let vals: Vec<VarId> = s.iter().flat_map(|a| [a[half * 2], a[half * 2 + 1]]).collect();
            let rs = g.sum_all(&vals);
            new_l[half] = g.fma(self.l[half], alpha[half], rs);
        }
        let mut o: Vec<[VarId; 4]> = self
            .o_acc
            .iter()
            .map(|acc| {
                [
                    g.mul(acc[0], alpha[0]),
                    g.mul(acc[1], alpha[0]),
                    g.mul(acc[2], alpha[1]),
                    g.mul(acc[3], alpha[1]),
                ]
            })
            .collect();
        let pa = acc_to_a_frags(g, &s);
        mma_b_trans(g, &mut o, &pa, self.v_ldm, self.rw);
        (o.into_iter().flatten().collect(), [new_m[0], new_m[1], new_l[0], new_l[1]])
    }
}

#[allow(clippy::too_many_arguments)]
fn epilogue(
    g: &mut G,
    o_acc: &[[VarId; 4]],
    m: [VarId; 2],
    l: [VarId; 2],
    rows: [VarId; 2],
    seq: VarId,
    sl2: VarId,
    o_base: VarId,
    lse_base: VarId,
    lc: VarId,
    hd: u32,
) {
    let one_f = g.f32c(1.0);
    let ln2 = g.f32c(LN_2);
    let zero_u = g.u32c(0);
    let lane0 = g.cmp(lc, zero_u, CmpOp::Eq);
    let c2 = g.mul_u32c(lc, 2);
    for half in 0..2 {
        let lt = g.quad_reduce(l[half], KirOp::Add);
        let inv = g.div(one_f, lt);
        let row = rows[half];
        let valid = g.cmp(row, seq, CmpOp::Lt);
        let roff = g.mul_u32c(row, hd);
        let roff = g.add(roff, c2);
        let prow = g.ptr(o_base, roff);
        for (n, acc) in o_acc.iter().enumerate() {
            let x0 = g.mul(acc[half * 2], inv);
            let x1 = g.mul(acc[half * 2 + 1], inv);
            let p = g.ptr_c(prow, n as u32 * 8);
            g.when(valid, KirOp::StoreVec { ptr: p, vals: vec![x0, x1], space: AddressSpace::Global });
        }
        // lse = (m·sl2 + log2 l) · ln 2
        let ms = g.mul(m[half], sl2);
        let lg = g.log2(lt);
        let t = g.add(ms, lg);
        let lse_v = g.mul(t, ln2);
        let pl = g.ptr(lse_base, row);
        let w = g.and(valid, lane0);
        g.when(w, KirOp::Store(pl, lse_v, AddressSpace::Global));
    }
}

/// The PTX module of one forward variant (null-terminated).
pub fn ptx(cfg: FwdConfig) -> Vec<u8> {
    super::super::elementwise::verified_ptx(build(cfg))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_verifies_and_lowers_to_tensor_core_ops() {
        for hd in FwdConfig::HEAD_DIMS {
            for causal in [false, true] {
                let cfg = FwdConfig::new(hd, causal);
                let ir = build(cfg);
                assert_eq!(ir.verify(), Ok(()), "{}", cfg.kernel_name());
                let text = String::from_utf8(ptx(cfg)).unwrap();
                assert!(text.contains(&format!(".visible .entry {}(", cfg.kernel_name())));
                let mmas = text.matches("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32").count();
                // Two paths (masked, plain) x (S and P·V) x n-tiles x k-steps.
                let per_path = 2 * (cfg.block_n() / 8) * (hd / 16);
                assert_eq!(mmas as u32, 2 * per_path, "{}", cfg.kernel_name());
                assert!(text.contains("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"));
                assert!(text.contains("cvt.rn.f16x2.f32"));
            }
        }
    }
}
