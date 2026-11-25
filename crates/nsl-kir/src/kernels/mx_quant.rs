// crates/nsl-kir/src/kernels/mx_quant.rs
//! MXFP8 quantizers: f32 or bf16 tensors to OCP E4M3 data plus one UE8M0
//! scale per 32 elements along the GEMM's K, in the layout cublasLt's
//! `CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` reads.
//!
//! cublasLt runs FP8 only in the "TN" form on sm_120: both operands K-major.
//! A row-major `[rows, k]` tensor whose K is its contiguous dimension is
//! quantized as it lies ([`MxQuant::Rows`]); one whose K is its row
//! dimension is quantized into its transpose ([`MxQuant::ColsT`]).
//!
//! | entry | input | output data | blocks of 32 along |
//! |---|---|---|---|
//! | `nsl_mx_quant_rows_{f32,bf16}` | `[rows, k]` | `[rows, k]` | each row |
//! | `nsl_mx_quant_cols_t_{f32,bf16}` | `[r, c]` | `[c, r]` (the transpose) | each column of the input |
//!
//! ## The scale rule
//!
//! For a block with largest magnitude `amax`, the scale is the smallest
//! power of two `2^e` with `amax / 2^e <= 448` (E4M3's largest finite), so
//! nothing saturates. It is computed exactly from the bits of `amax`:
//! `f = exponent(amax)`, `e = f - 8` if the mantissa is at most `0.75`
//! (`amax <= 1.75 · 2^f`, and `448 = 1.75 · 2^8`), else `f - 7`, clamped to
//! `[-127, 127]`. The scale byte is `e + 127`. An all-zero or all-subnormal
//! block gets byte 0 (`2^-127`). A block holding NaN or Inf gets byte
//! `0xFF`, the UE8M0 NaN, so the GEMM output carries the non-finite value
//! instead of a clamped one.
//!
//! Each element is `e4m3_rne_satfinite(x · 2^-e)`, the product rounded in
//! f32 (`mul.rn`). `2^-e` is a power of two, so the product is exact except
//! below f32's normal range, where E4M3 rounds it to zero anyway. The
//! magnitude order of f32 bit patterns is the order of the values (NaN
//! above Inf above every finite), so `amax` is a `max.u32` of the bits with
//! the sign cleared: a NaN or Inf anywhere in the block wins.
//!
//! [`reference`] computes the same bytes on the host.
//!
//! ## Threads
//!
//! Every thread converts four consecutive elements of one block. Eight
//! lanes (an aligned group of a warp) share a block and combine their
//! maxima with three `shfl.bfly`; the group's first lane writes the scale.
//!
//! - `Rows`: thread `g` takes elements `4g .. 4g + 4` of the flat tensor
//!   (`k % 32 == 0`, so a block never crosses a row). Grid:
//!   `ceil(rows · k / 4 / 256)` blocks of 256; threads past the end load
//!   element 0, still shuffle (`shfl.sync` wants the full warp), and skip
//!   both stores.
//! - `ColsT`: a CTA owns a 32 × 32 tile. It loads the tile row by row
//!   (coalesced) into shared memory with a 33-float pitch, then each thread
//!   reads four consecutive tile rows of one tile column and writes four
//!   consecutive bytes of one output row. Grid `(c / 32, r / 32)`.
//!
//! ## The scale layout
//!
//! cublasLt stores block scales in 128 × 4 tiles of 512 bytes: for outer
//! index `o` (the output row) and inner index `i` (the K block), the byte
//! sits at
//! `((o / 128) · (sf_inner / 4) + i / 4) · 512 + (o % 32) · 16 + ((o % 128) / 32) · 4 + i % 4`,
//! where `sf_inner = ceil(k / 32)` rounded up to a multiple of 4.
//! [`scale_bytes`] is the buffer size. Entries outside the tensor must be
//! zero: the caller clears the buffer when the rows are not a multiple of
//! 128 or `k / 32` is not a multiple of 4.

use super::storage::Storage;
use crate::backend_ptx::lower_kir_to_ptx;
use crate::kernel_ir::{
    AddressSpace, BlockId, CmpOp, ConstValue, KernelIR, KirBuilder, KirConst, KirEdge, KirOp,
    KirTerminator, KirType, ShuffleMode, SmemLayout, SmemRegion, VarId,
};

/// Elements per scale block.
pub const MX_BLOCK: usize = 32;

/// Threads per CTA, both kernels.
pub const MX_THREADS: u32 = 256;

/// Tile edge of [`MxQuant::ColsT`].
pub const MX_TILE: u32 = 32;

/// Which side the 32-element blocks run along.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MxQuant {
    /// Along each row of the input; the output has the input's layout.
    Rows,
    /// Along each column of the input; the output is the transpose.
    ColsT,
}

impl MxQuant {
    pub const ALL: [MxQuant; 2] = [MxQuant::Rows, MxQuant::ColsT];

    /// The `.visible .entry` name. Pinned: the runtime launches by it.
    pub fn kernel_name(self, src: Storage) -> &'static str {
        match (self, src) {
            (MxQuant::Rows, Storage::F32) => "nsl_mx_quant_rows_f32",
            (MxQuant::Rows, Storage::Bf16) => "nsl_mx_quant_rows_bf16",
            (MxQuant::ColsT, Storage::F32) => "nsl_mx_quant_cols_t_f32",
            (MxQuant::ColsT, Storage::Bf16) => "nsl_mx_quant_cols_t_bf16",
        }
    }
}

/// `ceil(n / m) · m`.
pub fn round_up(n: usize, m: usize) -> usize {
    n.div_ceil(m) * m
}

/// Scale blocks per output row, padded to the tile's 4: the `sf_inner`
/// both kernels take.
pub fn sf_inner(k: usize) -> usize {
    round_up(k.div_ceil(MX_BLOCK), 4)
}

/// Bytes of the scale buffer for an operand with `outer` rows and `k`
/// elements along K.
pub fn scale_bytes(outer: usize, k: usize) -> usize {
    round_up(outer, 128) * sf_inner(k)
}

/// Whether the scale buffer has entries outside the tensor, which must be
/// zero.
pub fn scales_padded(outer: usize, k: usize) -> bool {
    !outer.is_multiple_of(128) || !(k / MX_BLOCK).is_multiple_of(4)
}

/// Byte offset of the scale of outer index `o`, K block `i`.
pub fn scale_offset(o: usize, i: usize, sf_inner: usize) -> usize {
    ((o / 128) * (sf_inner / 4) + i / 4) * 512 + (o % 32) * 16 + ((o % 128) / 32) * 4 + i % 4
}

fn var(b: &mut KirBuilder, ty: KirType) -> VarId {
    b.new_typed_var(ty)
}

fn konst(b: &mut KirBuilder, ty: KirType, value: ConstValue) -> VarId {
    let dst = var(b, ty.clone());
    b.emit(KirOp::Const(dst, KirConst { ty, value }));
    dst
}

fn u32c(b: &mut KirBuilder, v: u32) -> VarId {
    konst(b, KirType::U32, ConstValue::U32(v))
}

fn op2(
    b: &mut KirBuilder,
    ty: KirType,
    op: fn(VarId, VarId, VarId) -> KirOp,
    x: VarId,
    y: VarId,
) -> VarId {
    let dst = var(b, ty);
    b.emit(op(dst, x, y));
    dst
}

fn cmp(b: &mut KirBuilder, x: VarId, y: VarId, op: CmpOp) -> VarId {
    let dst = var(b, KirType::Bool);
    b.emit(KirOp::Cmp(dst, x, y, op));
    dst
}

fn select(b: &mut KirBuilder, ty: KirType, c: VarId, t: VarId, f: VarId) -> VarId {
    let dst = var(b, ty);
    b.emit(KirOp::Select(dst, c, t, f));
    dst
}

fn widen(b: &mut KirBuilder, x: VarId) -> VarId {
    let dst = var(b, KirType::U64);
    b.emit(KirOp::Cast(dst, x, KirType::U64));
    dst
}

/// Load four consecutive elements `base[i .. i + 4]` (`i` in elements) and
/// widen them to f32 (bf16 widening is exact).
fn load4(b: &mut KirBuilder, base: VarId, i: VarId, src: Storage) -> [VarId; 4] {
    let addr = var(b, src.ptr());
    b.emit(KirOp::PtrOffset(addr, base, i));
    let raw: Vec<VarId> = (0..4).map(|_| var(b, src.elem())).collect();
    b.emit(KirOp::LoadVec {
        dsts: raw.clone(),
        ptr: addr,
        space: AddressSpace::Global,
    });
    let wide: Vec<VarId> = match src {
        Storage::F32 => raw,
        Storage::Bf16 => raw
            .into_iter()
            .map(|h| {
                let v = var(b, KirType::F32);
                b.emit(KirOp::Cast(v, h, KirType::F32));
                v
            })
            .collect(),
    };
    [wide[0], wide[1], wide[2], wide[3]]
}

/// The group's scale byte (`U32`) and this thread's four E4M3 bytes packed
/// in a `U32`, from its four f32 values. Every lane of the warp must reach
/// this: the maxima travel by `shfl.sync` over the full mask.
fn quantize_group(b: &mut KirBuilder, x: [VarId; 4]) -> (VarId, VarId) {
    use KirType::{F32, U32};
    let abs_mask = u32c(b, 0x7fff_ffff);
    let mut m: Option<VarId> = None;
    for &xi in &x {
        let bits = var(b, U32);
        b.emit(KirOp::Bitcast(bits, xi));
        let a = op2(b, U32, KirOp::And, bits, abs_mask);
        m = Some(match m {
            None => a,
            Some(prev) => op2(b, U32, KirOp::Max, prev, a),
        });
    }
    let mut m = m.expect("four values");
    for d in [1u32, 2, 4] {
        let lane = u32c(b, d);
        let other = var(b, U32);
        b.emit(KirOp::WarpShuffle {
            dst: other,
            val: m,
            lane,
            mode: ShuffleMode::Xor,
            width: 32,
        });
        m = op2(b, U32, KirOp::Max, m, other);
    }
    // byte = max(exponent + (mantissa > 0.75), 8) - 8; the NaN byte for a
    // non-finite block.
    let c23 = u32c(b, 23);
    let field = op2(b, U32, KirOp::Shr, m, c23);
    let mant_mask = u32c(b, 0x007f_ffff);
    let mant = op2(b, U32, KirOp::And, m, mant_mask);
    let three_quarters = u32c(b, 0x0060_0000);
    let above = cmp(b, mant, three_quarters, CmpOp::Gt);
    let one = u32c(b, 1);
    let zero = u32c(b, 0);
    let inc = select(b, U32, above, one, zero);
    let t = op2(b, U32, KirOp::Add, field, inc);
    let eight = u32c(b, 8);
    let big = cmp(b, t, eight, CmpOp::Gt);
    let t8 = op2(b, U32, KirOp::Sub, t, eight);
    let byte_fin = select(b, U32, big, t8, zero);
    // 2^-e = 2^(127 - byte): f32 bits (254 - byte) << 23, byte <= 248.
    let c254 = u32c(b, 254);
    let biased = op2(b, U32, KirOp::Sub, c254, byte_fin);
    let inv_fin = op2(b, U32, KirOp::Shl, biased, c23);
    let inf_bits = u32c(b, 0x7f80_0000);
    let nonfinite = cmp(b, m, inf_bits, CmpOp::Ge);
    let nan_byte = u32c(b, 0xff);
    let byte = select(b, U32, nonfinite, nan_byte, byte_fin);
    let one_bits = u32c(b, 0x3f80_0000);
    let inv_bits = select(b, U32, nonfinite, one_bits, inv_fin);
    let inv = var(b, F32);
    b.emit(KirOp::Bitcast(inv, inv_bits));
    let y: Vec<VarId> = x
        .iter()
        .map(|&xi| op2(b, F32, KirOp::MulRn, xi, inv))
        .collect();
    let packed = var(b, U32);
    b.emit(KirOp::PackE4m3x4 {
        dst: packed,
        vals: [y[0], y[1], y[2], y[3]],
    });
    (byte, packed)
}

/// Store the packed bytes as `q[word]` (a `U32` index, 4 bytes each).
fn store_word(b: &mut KirBuilder, q: VarId, word: VarId, packed: VarId) {
    let addr = var(
        b,
        KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global),
    );
    b.emit(KirOp::PtrOffset(addr, q, word));
    b.emit(KirOp::Store(addr, packed, AddressSpace::Global));
}

/// `scale_offset(o, i, sf_inner)` in `U32` arithmetic.
fn scale_index(b: &mut KirBuilder, o: VarId, i: VarId, sf_inner: VarId) -> VarId {
    use KirType::U32;
    let c7 = u32c(b, 7);
    let c2 = u32c(b, 2);
    let c3 = u32c(b, 3);
    let c31 = u32c(b, 31);
    let c4 = u32c(b, 4);
    let c5 = u32c(b, 5);
    let c9 = u32c(b, 9);
    let tile_row = op2(b, U32, KirOp::Shr, o, c7);
    let tiles_per_row = op2(b, U32, KirOp::Shr, sf_inner, c2);
    let tile_col = op2(b, U32, KirOp::Shr, i, c2);
    let row_base = op2(b, U32, KirOp::Mul, tile_row, tiles_per_row);
    let tile = op2(b, U32, KirOp::Add, row_base, tile_col);
    let tile_off = op2(b, U32, KirOp::Shl, tile, c9);
    let o32 = op2(b, U32, KirOp::And, o, c31);
    let o32_16 = op2(b, U32, KirOp::Shl, o32, c4);
    let o_quarter = op2(b, U32, KirOp::Shr, o, c5);
    let o_q = op2(b, U32, KirOp::And, o_quarter, c3);
    let o_q4 = op2(b, U32, KirOp::Shl, o_q, c2);
    let i4 = op2(b, U32, KirOp::And, i, c3);
    let a = op2(b, U32, KirOp::Add, tile_off, o32_16);
    let a2 = op2(b, U32, KirOp::Add, a, o_q4);
    op2(b, U32, KirOp::Add, a2, i4)
}

/// The group leader (`lane % 8 == 0`) writes `byte` at `s[index]`; then
/// branch to `next`.
fn store_scale_then(b: &mut KirBuilder, s: VarId, index: VarId, byte: VarId, next: BlockId) {
    use KirType::{U8, U32};
    let lane = var(b, U32);
    b.emit(KirOp::LaneId(lane));
    let c7 = u32c(b, 7);
    let sub = op2(b, U32, KirOp::And, lane, c7);
    let zero = u32c(b, 0);
    let leader = cmp(b, sub, zero, CmpOp::Eq);
    let write = b.new_block();
    b.terminate(KirTerminator::CondBranch(
        leader,
        KirEdge::to(write),
        KirEdge::to(next),
    ));
    b.set_block(write);
    let byte8 = var(b, U8);
    b.emit(KirOp::Cast(byte8, byte, U8));
    let addr = var(b, KirType::Ptr(Box::new(U8), AddressSpace::Global));
    b.emit(KirOp::PtrOffset(addr, s, index));
    b.emit(KirOp::Store(addr, byte8, AddressSpace::Global));
    b.terminate(KirTerminator::Branch(KirEdge::to(next)));
}

fn word_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::U32), AddressSpace::Global)
}

fn byte_ptr() -> KirType {
    KirType::Ptr(Box::new(KirType::U8), AddressSpace::Global)
}

/// `nsl_mx_quant_rows_*(src, q, s, nquads, k_blocks, sf_inner)`:
/// `nquads = rows · k / 4`, `k_blocks = k / 32`.
///
/// ```text
/// entry: g = global id; live = g < nquads; g = live ? g : 0
///        x = src[4g .. 4g+4]; (byte, packed) = quantize_group(x)
///        if !live: exit
/// store: q[g] = packed; blk = g / 8
///        leader: s[scale_offset(blk / k_blocks, blk % k_blocks)] = byte
/// ```
fn build_rows(src: Storage) -> KernelIR {
    use AddressSpace::Global;
    use KirType::U32;
    let mut b = KirBuilder::new(MxQuant::Rows.kernel_name(src));
    let srcp = b.add_param("src", src.ptr(), Global);
    let q = b.add_param("q", word_ptr(), Global);
    let s = b.add_param("s", byte_ptr(), Global);
    let nquads = b.add_param("nquads", U32, Global);
    let k_blocks = b.add_param("k_blocks", U32, Global);
    let sf = b.add_param("sf_inner", U32, Global);

    let entry = b.new_block();
    let store = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let g0 = var(&mut b, U32);
    b.emit(KirOp::GlobalId(g0, 0));
    let live = cmp(&mut b, g0, nquads, CmpOp::Lt);
    let zero = u32c(&mut b, 0);
    let g = select(&mut b, U32, live, g0, zero);
    let g64 = widen(&mut b, g);
    // Shift amounts are `U32` registers in PTX, whatever the shifted width.
    let c2 = u32c(&mut b, 2);
    let elem = op2(&mut b, KirType::U64, KirOp::Shl, g64, c2);
    let x = load4(&mut b, srcp, elem, src);
    let (byte, packed) = quantize_group(&mut b, x);
    b.terminate(KirTerminator::CondBranch(
        live,
        KirEdge::to(store),
        KirEdge::to(exit),
    ));

    b.set_block(store);
    store_word(&mut b, q, g64, packed);
    let c3 = u32c(&mut b, 3);
    let blk = op2(&mut b, U32, KirOp::Shr, g, c3);
    let o = op2(&mut b, U32, KirOp::Div, blk, k_blocks);
    let i = op2(&mut b, U32, KirOp::Rem, blk, k_blocks);
    let idx = scale_index(&mut b, o, i, sf);
    store_scale_then(&mut b, s, idx, byte, exit);

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([MX_THREADS, 1, 1]);
    b.set_launch_bounds(MX_THREADS, None);
    b.finalize()
}

/// `nsl_mx_quant_cols_t_*(src, q, s, r, c, sf_inner)`: input `[r, c]`,
/// output `[c, r]`, `r % 32 == c % 32 == 0`, grid `(c / 32, r / 32)`.
///
/// ```text
/// load:  t = tid; tile[t / 8][4(t % 8) ..] = src[(r0 + t / 8) · c + c0 + 4(t % 8) ..]
///        barrier
/// quant: oc = t / 8; or = 4(t % 8); x = tile[or .. or+4][oc]
///        q[((c0 + oc) · r + r0 + or) / 4] = packed
///        leader: s[scale_offset(c0 + oc, r0 / 32)] = byte
/// ```
fn build_cols_t(src: Storage) -> KernelIR {
    use AddressSpace::{Global, Shared};
    use KirType::{F32, U32, U64};
    let mut b = KirBuilder::new(MxQuant::ColsT.kernel_name(src));
    let srcp = b.add_param("src", src.ptr(), Global);
    let q = b.add_param("q", word_ptr(), Global);
    let s = b.add_param("s", byte_ptr(), Global);
    let r = b.add_param("r", U32, Global);
    let c = b.add_param("c", U32, Global);
    let sf = b.add_param("sf_inner", U32, Global);
    let pitch = MX_TILE + 1;
    b.set_smem_layout(SmemLayout {
        regions: vec![SmemRegion {
            name: "tile".to_string(),
            bytes: MX_TILE * pitch * 4,
            align: 4,
            elem: F32,
        }],
        dynamic: false,
    });

    let entry = b.new_block();
    let exit = b.new_block();
    b.set_block(entry);
    let tid = var(&mut b, U32);
    b.emit(KirOp::ThreadId(tid, 0));
    let bx = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(bx, 0));
    let by = var(&mut b, U32);
    b.emit(KirOp::BlockIdx(by, 1));
    let c5 = u32c(&mut b, 5);
    let c3 = u32c(&mut b, 3);
    let c7 = u32c(&mut b, 7);
    let c2 = u32c(&mut b, 2);
    let c0 = op2(&mut b, U32, KirOp::Shl, bx, c5);
    let r0 = op2(&mut b, U32, KirOp::Shl, by, c5);
    let hi = op2(&mut b, U32, KirOp::Shr, tid, c3);
    let lo = op2(&mut b, U32, KirOp::And, tid, c7);
    let lo4 = op2(&mut b, U32, KirOp::Shl, lo, c2);
    let tile = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
    b.emit(KirOp::SharedRegion {
        dst: tile,
        region: 0,
    });
    let pitch_c = u32c(&mut b, pitch);

    // Load: tile row `hi`, tile columns lo4 .. lo4 + 4.
    let row = op2(&mut b, U32, KirOp::Add, r0, hi);
    let row64 = widen(&mut b, row);
    let c64 = widen(&mut b, c);
    let col = op2(&mut b, U32, KirOp::Add, c0, lo4);
    let col64 = widen(&mut b, col);
    let row_base = op2(&mut b, U64, KirOp::Mul, row64, c64);
    let at = op2(&mut b, U64, KirOp::Add, row_base, col64);
    let x = load4(&mut b, srcp, at, src);
    let srow = op2(&mut b, U32, KirOp::Mul, hi, pitch_c);
    for (j, &xj) in x.iter().enumerate() {
        let jc = u32c(&mut b, j as u32);
        let sc = op2(&mut b, U32, KirOp::Add, lo4, jc);
        let si = op2(&mut b, U32, KirOp::Add, srow, sc);
        let addr = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
        b.emit(KirOp::PtrOffset(addr, tile, si));
        b.emit(KirOp::Store(addr, xj, Shared));
    }
    b.emit(KirOp::Barrier);

    // Quantize: tile column `hi`, tile rows lo4 .. lo4 + 4.
    let mut y = Vec::with_capacity(4);
    for j in 0..4u32 {
        let jc = u32c(&mut b, j);
        let tr = op2(&mut b, U32, KirOp::Add, lo4, jc);
        let trp = op2(&mut b, U32, KirOp::Mul, tr, pitch_c);
        let si = op2(&mut b, U32, KirOp::Add, trp, hi);
        let addr = var(&mut b, KirType::Ptr(Box::new(F32), Shared));
        b.emit(KirOp::PtrOffset(addr, tile, si));
        let v = var(&mut b, F32);
        b.emit(KirOp::Load(v, addr, Shared));
        y.push(v);
    }
    let (byte, packed) = quantize_group(&mut b, [y[0], y[1], y[2], y[3]]);
    let orow = op2(&mut b, U32, KirOp::Add, c0, hi);
    let orow64 = widen(&mut b, orow);
    let r64 = widen(&mut b, r);
    let obase = op2(&mut b, U64, KirOp::Mul, orow64, r64);
    let ocol = op2(&mut b, U32, KirOp::Add, r0, lo4);
    let ocol64 = widen(&mut b, ocol);
    let oelem = op2(&mut b, U64, KirOp::Add, obase, ocol64);
    let word = op2(&mut b, U64, KirOp::Shr, oelem, c2);
    store_word(&mut b, q, word, packed);
    let kblk = op2(&mut b, U32, KirOp::Shr, r0, c5);
    let idx = scale_index(&mut b, orow, kblk, sf);
    store_scale_then(&mut b, s, idx, byte, exit);

    b.set_block(exit);
    b.terminate(KirTerminator::Return);
    b.set_workgroup_size([MX_THREADS, 1, 1]);
    b.set_launch_bounds(MX_THREADS, None);
    b.finalize()
}

/// The kernel as KIR.
pub fn build(q: MxQuant, src: Storage) -> KernelIR {
    match q {
        MxQuant::Rows => build_rows(src),
        MxQuant::ColsT => build_cols_t(src),
    }
}

/// The kernel's verified PTX (null-terminated).
pub fn ptx(q: MxQuant, src: Storage) -> Vec<u8> {
    let ir = build(q, src);
    if let Err(errors) = crate::kir_verify::verify(&ir) {
        panic!(
            "mx_quant kernel `{}` failed KIR verification: {errors:?}",
            ir.name
        );
    }
    lower_kir_to_ptx(&ir)
}

/// The host computation the kernels must match byte for byte.
pub mod reference {
    use super::{MX_BLOCK, scale_bytes, scale_offset, sf_inner};

    /// The 127 non-negative finite E4M3 values, by code (`0x00 ..= 0x7E`).
    fn e4m3_values() -> [f64; 127] {
        let mut v = [0.0; 127];
        for (code, slot) in v.iter_mut().enumerate() {
            let exp = (code >> 3) as i32;
            let man = (code & 7) as f64;
            *slot = if exp == 0 {
                man * 2f64.powi(-9)
            } else {
                (1.0 + man / 8.0) * 2f64.powi(exp - 7)
            };
        }
        v
    }

    /// `cvt.rn.satfinite.e4m3x2.f32` for one value: round to nearest even,
    /// magnitudes past 448 to 448, NaN to `0x7F`.
    pub fn e4m3_rne_satfinite(x: f32) -> u8 {
        if x.is_nan() {
            return 0x7f;
        }
        let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
        let a = f64::from(x.abs());
        let table = e4m3_values();
        if a >= table[126] {
            return sign | 0x7e;
        }
        // The first code whose value is >= a; a lies in (table[hi-1], table[hi]].
        let hi = table.partition_point(|&t| t < a);
        if table[hi] == a || hi == 0 {
            return sign | hi as u8;
        }
        let lo = hi - 1;
        let (dl, dh) = (a - table[lo], table[hi] - a);
        let code = if dl < dh {
            lo
        } else if dh < dl {
            hi
        } else if lo.is_multiple_of(2) {
            lo
        } else {
            hi
        };
        sign | code as u8
    }

    /// The scale byte of a block and the multiplier `2^-e` (as the kernel
    /// computes them, from the bits of the block's largest magnitude).
    pub fn block_scale(block: &[f32]) -> (u8, f32) {
        let m = block
            .iter()
            .map(|x| x.to_bits() & 0x7fff_ffff)
            .max()
            .unwrap_or(0);
        if m >= 0x7f80_0000 {
            return (0xff, 1.0);
        }
        let t = (m >> 23) + u32::from((m & 0x007f_ffff) > 0x0060_0000);
        let byte = t.saturating_sub(8);
        (byte as u8, f32::from_bits((254 - byte) << 23))
    }

    /// Quantize `rows` rows of `k` values (row-major), blocks along each
    /// row: `(data [rows · k], scales [scale_bytes(rows, k)])`.
    pub fn quant_rows(x: &[f32], rows: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        assert_eq!(x.len(), rows * k);
        assert_eq!(k % MX_BLOCK, 0);
        let sf = sf_inner(k);
        let mut data = vec![0u8; rows * k];
        let mut scales = vec![0u8; scale_bytes(rows, k)];
        for r in 0..rows {
            for i in 0..k / MX_BLOCK {
                let at = r * k + i * MX_BLOCK;
                let block = &x[at..at + MX_BLOCK];
                let (byte, inv) = block_scale(block);
                scales[scale_offset(r, i, sf)] = byte;
                for (j, &v) in block.iter().enumerate() {
                    data[at + j] = e4m3_rne_satfinite(v * inv);
                }
            }
        }
        (data, scales)
    }

    /// Quantize the transpose of row-major `[r, c]`: blocks along each
    /// column, output `[c, r]`.
    pub fn quant_cols_t(x: &[f32], r: usize, c: usize) -> (Vec<u8>, Vec<u8>) {
        assert_eq!(x.len(), r * c);
        let mut t = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                t[j * r + i] = x[i * c + j];
            }
        }
        quant_rows(&t, c, r)
    }

    /// The value an E4M3 byte encodes (`NaN` for `0x7F` / `0xFF`).
    pub fn e4m3_value(b: u8) -> f64 {
        if b & 0x7f == 0x7f {
            return f64::NAN;
        }
        let v = e4m3_values()[usize::from(b & 0x7f)];
        if b & 0x80 != 0 { -v } else { v }
    }

    /// The value a UE8M0 byte encodes (`NaN` for `0xFF`).
    pub fn e8m0_value(b: u8) -> f64 {
        if b == 0xff {
            f64::NAN
        } else {
            2f64.powi(i32::from(b) - 127)
        }
    }

    /// Dequantize element `(o, kk)` of a quantized `[outer, k]` operand.
    pub fn dequant(data: &[u8], scales: &[u8], k: usize, o: usize, kk: usize) -> f64 {
        e4m3_value(data[o * k + kk])
            * e8m0_value(scales[scale_offset(o, kk / MX_BLOCK, sf_inner(k))])
    }
}

#[cfg(test)]
mod tests {
    use super::reference::*;
    use super::*;
    use crate::feature_set::FeatureSet;

    #[test]
    fn every_kernel_verifies_and_targets_sm_89() {
        for q in MxQuant::ALL {
            for src in Storage::ALL {
                let ir = build(q, src);
                assert!(ir.required_features.contains(FeatureSet::FP8_CONVERT));
                let text = String::from_utf8(ptx(q, src)).unwrap();
                assert!(text.contains(".version 8.1"), "{text}");
                assert!(text.contains(".target sm_89"), "{text}");
                assert!(
                    text.contains(&format!(".visible .entry {}", q.kernel_name(src))),
                    "{text}"
                );
                assert_eq!(
                    text.matches("cvt.rn.satfinite.e4m3x2.f32").count(),
                    2,
                    "{text}"
                );
                assert_eq!(text.matches("shfl.sync.bfly.b32").count(), 3, "{text}");
            }
        }
    }

    #[test]
    fn the_packed_conversion_puts_the_first_value_in_the_low_byte() {
        // PTX writes a packed conversion's FIRST source to the upper half, so
        // `lo` must convert (vals[1], vals[0]) and `hi` (vals[3], vals[2]);
        // `mov.b32 {lo, hi}` then puts `lo` low. The printer renumbers
        // registers, so the sources are told apart by the constants that
        // define them.
        use crate::kernel_ir::KirTerminator;
        let mut b = KirBuilder::new("pack_order");
        let out = b.add_param("out", word_ptr(), AddressSpace::Global);
        let entry = b.new_block();
        b.set_block(entry);
        let consts = [1.0f32, 2.0, 3.0, 4.0];
        let vals: Vec<VarId> = consts
            .iter()
            .map(|&c| konst(&mut b, KirType::F32, ConstValue::F32(c)))
            .collect();
        let packed = var(&mut b, KirType::U32);
        b.emit(KirOp::PackE4m3x4 {
            dst: packed,
            vals: [vals[0], vals[1], vals[2], vals[3]],
        });
        b.emit(KirOp::Store(out, packed, AddressSpace::Global));
        b.terminate(KirTerminator::Return);
        b.set_workgroup_size([1, 1, 1]);
        let text = String::from_utf8(lower_kir_to_ptx(&b.finalize())).unwrap();
        // The register each constant lands in.
        let reg_of = |c: f32| -> String {
            let hex = format!("0f{:08X}", c.to_bits());
            let line = text
                .lines()
                .find(|l| l.contains(&hex))
                .unwrap_or_else(|| panic!("no constant {c} ({hex}) in\n{text}"));
            line.split_whitespace()
                .nth(1)
                .unwrap()
                .trim_end_matches(',')
                .to_string()
        };
        let r: Vec<String> = consts.iter().map(|&c| reg_of(c)).collect();
        let line = text.lines().find(|l| l.contains("e4m3x2")).unwrap();
        let lo = format!("cvt.rn.satfinite.e4m3x2.f32 lo, {}, {};", r[1], r[0]);
        let hi = format!("cvt.rn.satfinite.e4m3x2.f32 hi, {}, {};", r[3], r[2]);
        assert!(line.contains(&lo), "{line}\nwant {lo}");
        assert!(line.contains(&hi), "{line}\nwant {hi}");
        assert!(
            line.contains("mov.b32") && line.contains("{lo, hi}"),
            "{line}"
        );
    }

    #[test]
    fn e4m3_rounding_matches_the_format() {
        assert_eq!(e4m3_rne_satfinite(0.0), 0x00);
        assert_eq!(e4m3_rne_satfinite(-0.0), 0x80);
        assert_eq!(e4m3_rne_satfinite(1.0), 0x38);
        assert_eq!(e4m3_rne_satfinite(448.0), 0x7e);
        assert_eq!(e4m3_rne_satfinite(1e9), 0x7e);
        assert_eq!(e4m3_rne_satfinite(f32::INFINITY), 0x7e);
        assert_eq!(e4m3_rne_satfinite(f32::NEG_INFINITY), 0xfe);
        assert_eq!(e4m3_rne_satfinite(f32::NAN) & 0x7f, 0x7f);
        // Smallest subnormal 2^-9; half of it ties to even (0).
        assert_eq!(e4m3_rne_satfinite(2f32.powi(-9)), 0x01);
        assert_eq!(e4m3_rne_satfinite(2f32.powi(-10)), 0x00);
        assert_eq!(e4m3_rne_satfinite(1.5 * 2f32.powi(-9)), 0x02);
        // 1 + 1/16 sits between 1 (0x38) and 1.125 (0x39): ties to even, 0x38.
        assert_eq!(e4m3_rne_satfinite(1.0625), 0x38);
        assert_eq!(e4m3_rne_satfinite(1.1875), 0x3a);
        // Every code round-trips.
        for code in 0u8..0x7f {
            let v = e4m3_value(code) as f32;
            assert_eq!(e4m3_rne_satfinite(v), code);
            assert_eq!(e4m3_rne_satfinite(-v) & 0x7f, code);
        }
    }

    #[test]
    fn the_scale_is_the_smallest_power_of_two_that_fits_448() {
        let check = |amax: f32| {
            let (byte, inv) = block_scale(&[amax]);
            let e = i32::from(byte) - 127;
            assert_eq!(inv, 2f32.powi(-e), "amax {amax}");
            assert!(
                f64::from(amax) * f64::from(inv) <= 448.0,
                "amax {amax} saturates"
            );
            if e > -127 {
                assert!(
                    f64::from(amax) * 2f64.powi(-(e - 1)) > 448.0,
                    "amax {amax}: e not minimal"
                );
            }
        };
        for amax in [
            448.0f32,
            449.0,
            1.0,
            1.75,
            1.7500001,
            0.001,
            3.0e38,
            1e-30,
            2f32.powi(-126),
            6.0e-39,
        ] {
            check(amax);
        }
        assert_eq!(block_scale(&[0.0, -0.0]), (0, 2f32.powi(127)));
        assert_eq!(block_scale(&[1.0, f32::NAN]).0, 0xff);
        assert_eq!(block_scale(&[f32::NEG_INFINITY]).0, 0xff);
        assert_eq!(block_scale(&[448.0]).0, 127);
    }

    #[test]
    fn the_scale_layout_matches_the_cublas_tile() {
        // Within a tile: (o % 32) * 16 + (o / 32) * 4 + i.
        assert_eq!(scale_offset(0, 0, 4), 0);
        assert_eq!(scale_offset(0, 3, 4), 3);
        assert_eq!(scale_offset(1, 0, 4), 16);
        assert_eq!(scale_offset(32, 0, 4), 4);
        assert_eq!(scale_offset(127, 3, 4), 31 * 16 + 3 * 4 + 3);
        // Tiles row-major: the next 4 K-blocks, then the next 128 rows.
        assert_eq!(scale_offset(0, 4, 8), 512);
        assert_eq!(scale_offset(128, 0, 8), 1024);
        // Every in-range entry has its own byte.
        let (outer, k) = (256, 32 * 12);
        let sf = sf_inner(k);
        let mut seen = vec![false; scale_bytes(outer, k)];
        for o in 0..outer {
            for i in 0..k / 32 {
                let at = scale_offset(o, i, sf);
                assert!(!seen[at]);
                seen[at] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
        assert!(!scales_padded(256, 32 * 12));
        assert!(scales_padded(3520, 2048));
        assert!(scales_padded(2048, 3520));
    }

    #[test]
    fn the_transposed_reference_is_the_rows_reference_of_the_transpose() {
        let (r, c) = (64, 96);
        let x: Vec<f32> = (0..r * c)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.37)
            .collect();
        let (d, s) = quant_cols_t(&x, r, c);
        for j in 0..c {
            for i in 0..r {
                let deq = dequant(&d, &s, r, j, i);
                let rel = (deq - f64::from(x[i * c + j])).abs()
                    / f64::from(x[i * c + j]).abs().max(1e-30);
                assert!(
                    rel <= 1.0 / 16.0 + 1e-12 || x[i * c + j] == 0.0,
                    "({i},{j}) {deq} vs {}",
                    x[i * c + j]
                );
            }
        }
    }
}
