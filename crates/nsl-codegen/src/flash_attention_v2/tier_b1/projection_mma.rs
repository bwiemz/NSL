//! Q/K/V projection chunk-streamed MMA emitter.
//!
//! B1.3 ships the Q projection (`emit_q_projection`). B1.5 adds the K/V
//! projection emitter (`emit_kv_projection_chunk_loop`), which mirrors the
//! Q-projection structure but accumulates into two independent sets of f32
//! registers (K and V) and scatters to the appropriate ping/pong SMEM slots.
//!
//! Layout (matches `smem_layout::tier_b1_*_offset` accessors + the
//! `validate_tier_b1_config` chunk-staging budget):
//!
//! ```text
//!   q_offset                                                      [bq * hd * 2]
//!   k_offset_ping                                                 [bkv * hd * 2]
//!   k_offset_pong                                                 [bkv * hd * 2]
//!   v_offset_ping                                                 [bkv * hd * 2]
//!   v_offset_pong                                                 [bkv * hd * 2]
//!   w_chunk_offset            slot 0: Wk chunk          [chunk*hd*2]
//!                             slot 1: Wv chunk          [chunk*hd*2]
//!                             x_q chunk staging         [bq*chunk*2]
//!                             x_kv chunk staging        [bkv*chunk*2]
//! ```
//!
//! For Q projection at B1.3:
//!   * Wq chunk staged at  w_chunk_offset (slot 0).
//!   * x_q chunk staged at w_chunk_offset + 2 * chunk * hd * 2 (after both W slots).
//!
//! For K/V projection at B1.5:
//!   * Wk chunk staged at w_chunk_offset (slot 0).
//!   * Wv chunk staged at w_chunk_offset + chunk * hd * 2 (slot 1).
//!   * x_kv chunk staged at w_chunk_offset + 2*chunk*hd*2 + bq*chunk*2.
//!   * x_kv HBM offset: kv_iter * bkv * d_model * 2 (f16, chunks-major).
//!
//! ### HBM layout assumed for x_q / x_kv (post chunk-stride fix)
//!
//! Both `csha_x_ptr` slices are expected in this layout:
//!
//! ```text
//!   x_q  : [d_model/chunk, bq,  chunk]            f16 chunks-major
//!   x_kv : [n_kv_iters, d_model/chunk, bkv, chunk] f16 chunks-major
//! ```
//!
//! Per cp.async chunk band: `(bq | bkv) * chunk * 2` bytes contiguous.
//! Per chunk_idx advance: `(bq | bkv) * chunk * 2` bytes (chunks-major).
//! Per kv_iter advance (x_kv only): `bkv * d_model * 2` bytes.
//!
//! Prior to this fix the cp.async used `chunk_idx * chunk * 4` (f32
//! stride + single-row-per-chunk addressing). Successive chunks then
//! overlapped by ~94% of their content for canonical configs
//! (chunk=128, bq=32: `chunk * 4 = 512` bytes between starts but each
//! cp.async copies `bq * chunk * 2 = 8192` bytes). The projection MMA
//! consequently fed garbage into the dot product. See commits 1ca2a62f
//! + f6c9958b for the parallel matmul_mma fragment-load helper rewrite
//! (those fixed the lane-mapping bug; this comment-block documents the
//! orthogonal chunk-stride fix).
//!
//! **Caller responsibility**: produce x_q / x_kv in this layout
//! upstream. The standard CSHA pipeline emits RMSNorm-normalised x as
//! f32 `[bq, d_model]` (or `[seq, d_model]`); a narrow-and-chunkify
//! pre-pass (out of scope for Tier B.1's codegen) must rearrange that
//! buffer into the f16 chunks-major form before this kernel runs.
//!
//! Warp distribution (per spec §5.3):
//!   * 8 warps per CTA.
//!   * Q output tile is (bq × hd), tiled by m16n8k16 into
//!     (bq/16) × (hd/8) tiles.
//!   * K/V output tile is (bkv × hd), tiled by m16n8k16 into
//!     (bkv/16) × (hd/8) tiles.
//!   * Warp `w` owns linear tile indices `t` where `t % 8 == w`.
//!   * `tiles_per_warp = max((bq/16) * (hd/8) / 8, 1)`.
//!   * `tiles_per_warp_kv = max((bkv/16) * (hd/8) / 8, 1)`. Each MMA
//!     produces 4 f32 lanes per thread.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    tier_b1_k_offset_ping, tier_b1_k_offset_pong, tier_b1_q_offset, tier_b1_v_offset_ping,
    tier_b1_v_offset_pong, tier_b1_w_chunk_offset,
};
use crate::matmul_mma::{
    emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
};

/// Emit a multi-iter per-thread `cp.async.cg.shared.global` sequence that
/// stages `smem_dest_size_bytes` from HBM into SMEM. With 256 threads per
/// CTA and 16 bytes per cp.async, each sub-iter stages 4096 bytes; we emit
/// `ceil(smem_dest_size_bytes / 4096)` sub-iters at codegen time, each
/// shifting both the SMEM destination and HBM source addresses by
/// `sub_iter * 4096` bytes.
///
/// Resolves B1.6 deferral #3 multi-iter unroll (chunks > 4096 bytes).
///
/// Caller is responsible for declaring + populating the named scratch
/// registers (`smem_addr_u32`, `hbm_addr_u64`) and the thread-local offset
/// register (`tid_off_u32` = `%tid.x << 4`). Caller also passes:
/// - `smem_base_u32`: u32 SMEM base reg for this destination slot
/// - `hbm_ptr_u64`: u64 reg holding the HBM source pointer
/// - `hbm_chunk_base_off_bytes`: codegen-time offset within the HBM
///   buffer (e.g. `chunk_idx * chunk * hd * 2` for the per-chunk shift)
///
/// `hbm_off_u32` is **legacy / unused** — earlier revisions computed HBM
/// addresses through a u32 scratch register. On sm_80+ (Hopper, Blackwell)
/// HBM device pointers occupy the full 64-bit virtual address space and
/// the upper 32 bits are routinely non-zero, so the u32 narrowing
/// produced an HBM address with the upper bits zeroed — virtually
/// guaranteed to fault with `CUDA_ERROR_ILLEGAL_ADDRESS` at launch.
/// Discovered during N4 GPU disambiguation (2026-05-15). The parameter
/// is retained for call-site signature stability; the helper no longer
/// touches the register, so callers may keep declaring it (ptxas prunes
/// unused virtual regs) or strip it in a follow-up cleanup.
fn emit_cp_async_multi_iter(
    ptx: &mut String,
    smem_dest_size_bytes: u32,
    smem_base_u32: &str,
    tid_off_u32: &str,
    hbm_ptr_u64: &str,
    hbm_chunk_base_off_bytes: u32,
    smem_addr_u32: &str,
    hbm_off_u32: &str,
    hbm_addr_u64: &str,
) {
    // 256 threads × 16 bytes = 4096 bytes per cp.async sub-iter.
    let n_subiters = smem_dest_size_bytes.div_ceil(4096).max(1);

    for sub in 0..n_subiters {
        let sub_off = sub * 4096;
        if n_subiters > 1 {
            ptx.push_str(&format!(
                "    // cp.async sub_iter {}/{} (sub_off={} bytes)\n",
                sub + 1,
                n_subiters,
                sub_off
            ));
        }
        // SMEM addr (u32) = smem_base + tid_off + sub_off.
        // SMEM-space addresses on sm_80+ are 32-bit, so u32 arithmetic is
        // correct for the destination.
        ptx.push_str(&format!(
            "    add.u32 {}, {}, {};\n",
            smem_addr_u32, smem_base_u32, tid_off_u32
        ));
        if sub_off > 0 {
            ptx.push_str(&format!(
                "    add.u32 {}, {}, {};\n",
                smem_addr_u32, smem_addr_u32, sub_off
            ));
        }
        // HBM addr (u64) = hbm_ptr + (hbm_chunk_base_off + sub_off) + tid_off.
        // **CRITICAL:** keep the addressing in u64. On sm_80+ (Hopper,
        // Blackwell) HBM device pointers occupy the full 64-bit virtual
        // address space; the upper 32 bits are routinely non-zero. The
        // earlier emission narrowed to u32 via `cvt.u32.u64` and operated
        // on the truncated low half, which produced an HBM address with
        // the upper 32 bits zeroed — virtually guaranteed to fault with
        // CUDA_ERROR_ILLEGAL_ADDRESS at launch.
        // The `hbm_off_u32` register passed by the caller is now unused;
        // callers may keep declaring it (ptxas prunes unused virtual regs)
        // but the helper no longer touches it.
        let _ = hbm_off_u32;
        // Widen tid_off (u32) into hbm_addr_u64 via cvt; then add the u64
        // base + the codegen-time chunk + sub byte offset.
        ptx.push_str(&format!(
            "    cvt.u64.u32 {}, {};\n",
            hbm_addr_u64, tid_off_u32
        ));
        ptx.push_str(&format!(
            "    add.u64 {}, {}, {};\n",
            hbm_addr_u64, hbm_addr_u64, hbm_ptr_u64
        ));
        let total_chunk_off = hbm_chunk_base_off_bytes + sub_off;
        if total_chunk_off > 0 {
            ptx.push_str(&format!(
                "    add.u64 {}, {}, {};\n",
                hbm_addr_u64, hbm_addr_u64, total_chunk_off
            ));
        }
        ptx.push_str(&format!(
            "    cp.async.cg.shared.global [{}], [{}], 16;\n",
            smem_addr_u32, hbm_addr_u64
        ));
    }
}

/// Emit a D-fragment-aware scatter from per-warp m16n8k16 accumulators
/// (`%<acc_prefix>_<t>_<lane>`) to a row-major SMEM tile
/// `[m_dim_block × head_dim]` f16. Each thread writes the 4 f16 values
/// it holds per tile per the standard m16n8k16 D-fragment lane mapping
/// (spec §5.5):
///   D0 = (row=lo_row,     col=lo_col_base    ) → +0 bytes
///   D1 = (row=lo_row,     col=lo_col_base + 1) → +2 bytes
///   D2 = (row=lo_row + 8, col=lo_col_base    ) → +8*head_dim*2 bytes
///   D3 = (row=lo_row + 8, col=lo_col_base + 1) → +8*head_dim*2 + 2 bytes
///
/// Where `lo_row = laneid / 4` and `lo_col_base = (laneid % 4) * 2`.
///
/// **N3 resolution:** each warp's slot `local_t` maps to a DISTINCT
/// global tile via `global_t = warp_id + local_t * 8`. The tile's
/// (m_tile, n_tile) coordinates are computed from `global_t` at PTX
/// runtime, so every warp writes to a different output tile. No
/// store-time predicate; all 8 warps fire each iteration.
///
/// Requires `head_dim / 8` to be a power of 2 (asserted).
///
/// `prefix` is the namespace for this call's scratch registers — caller
/// MUST use a unique value per call site on the same ptx string (e.g.
/// "qsc", "ksc", "vsc") or ptxas rejects with duplicate-declaration.
///
/// `acc_prefix` is the accumulator register prefix (e.g. "q_acc",
/// "k_acc", "v_acc"). The accumulator is read as `%<acc_prefix>_<t>_<i>`.
///
/// `smem_u64_base` is the u64 SMEM base register expression — converted
/// inside this helper to a u32 sister via `cvt.u32.u64`.
fn emit_dfragment_scatter(
    ptx: &mut String,
    prefix: &str,
    acc_prefix: &str,
    smem_u64_base: &str,
    head_dim: u32,
    tpw: u32,
) {
    let n_tiles_d = (head_dim / 8).max(1);
    assert!(
        n_tiles_d.is_power_of_two(),
        "N3: emit_dfragment_scatter requires head_dim/8 to be a power of 2; got {}",
        n_tiles_d
    );
    let log2_n_tiles_d = n_tiles_d.trailing_zeros();
    let n_tiles_d_mask = n_tiles_d - 1;
    let m_stride_bytes = 16 * head_dim * 2;
    assert!(
        m_stride_bytes.is_power_of_two(),
        "N3: emit_dfragment_scatter requires 16*head_dim*2 to be a power of 2; got {}",
        m_stride_bytes
    );
    let log2_m_stride = m_stride_bytes.trailing_zeros();

    ptx.push_str(&format!(
        "    // === {} D-fragment scatter ({} tiles, head_dim={}, N3) ===\n",
        prefix, tpw, head_dim
    ));

    ptx.push_str(&format!(
        "    .reg .u32 %{}_laneid, %{}_lo_row, %{}_lo_col_base;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_row_off, %{}_col_off, %{}_lane_off;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_base_u32, %{}_addr_u32, %{}_addr_i;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_global_t, %{}_m_tile, %{}_n_tile;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_m_off, %{}_n_off, %{}_tile_off, %{}_tile_u32;\n",
        prefix, prefix, prefix, prefix
    ));
    ptx.push_str(&format!("    .reg .b16 %{}_h;\n", prefix));

    // Per-lane (lo_row, lo_col_base, lane_off) computed once.
    ptx.push_str(&format!("    mov.u32 %{}_laneid, %tid.x;\n", prefix));
    ptx.push_str(&format!(
        "    and.b32 %{}_laneid, %{}_laneid, 31;\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    shr.u32 %{}_lo_row, %{}_laneid, 2;       // l/4\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    and.b32 %{}_lo_col_base, %{}_laneid, 3;  // l%4\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    shl.b32 %{}_lo_col_base, %{}_lo_col_base, 1; // (l%4)*2 = col base\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    mul.lo.u32 %{}_row_off, %{}_lo_row, {};  // row * head_dim*2\n",
        prefix,
        prefix,
        head_dim * 2
    ));
    ptx.push_str(&format!(
        "    shl.b32 %{}_col_off, %{}_lo_col_base, 1;     // col * 2 bytes (f16)\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    add.u32 %{}_lane_off, %{}_row_off, %{}_col_off;\n",
        prefix, prefix, prefix
    ));

    // u32 sister for the SMEM base.
    ptx.push_str(&format!(
        "    cvt.u32.u64 %{}_base_u32, {};\n",
        prefix, smem_u64_base
    ));

    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // {} slot local_t={} (global_t = warp_id + {}*8 at runtime; N3)\n",
            prefix,
            t,
            t
        ));

        // N3: global_t = warp_id + local_t * 8
        if t == 0 {
            ptx.push_str(&format!("    mov.u32 %{}_global_t, %warp_id;\n", prefix));
        } else {
            ptx.push_str(&format!(
                "    add.u32 %{}_global_t, %warp_id, {};\n",
                prefix,
                t * 8
            ));
        }
        // (m_tile, n_tile) from global_t (shr/and since power of 2).
        ptx.push_str(&format!(
            "    shr.u32 %{}_m_tile, %{}_global_t, {};\n",
            prefix, prefix, log2_n_tiles_d
        ));
        ptx.push_str(&format!(
            "    and.b32 %{}_n_tile, %{}_global_t, {};\n",
            prefix, prefix, n_tiles_d_mask
        ));
        // m_off = m_tile * (16 * head_dim * 2) bytes (shl since pow2).
        ptx.push_str(&format!(
            "    shl.b32 %{}_m_off, %{}_m_tile, {};\n",
            prefix, prefix, log2_m_stride
        ));
        // n_off = n_tile * 8 * 2 = n_tile * 16 bytes (shl 4).
        ptx.push_str(&format!(
            "    shl.b32 %{}_n_off, %{}_n_tile, 4;\n",
            prefix, prefix
        ));
        ptx.push_str(&format!(
            "    add.u32 %{}_tile_off, %{}_m_off, %{}_n_off;\n",
            prefix, prefix, prefix
        ));
        ptx.push_str(&format!(
            "    add.u32 %{}_tile_u32, %{}_base_u32, %{}_tile_off;\n",
            prefix, prefix, prefix
        ));
        ptx.push_str(&format!(
            "    add.u32 %{}_addr_u32, %{}_tile_u32, %{}_lane_off;\n",
            prefix, prefix, prefix
        ));

        for i in 0..4u32 {
            let off_bytes: u32 = match i {
                0 => 0,
                1 => 2,
                2 => 8 * head_dim * 2,
                3 => 8 * head_dim * 2 + 2,
                _ => unreachable!(),
            };
            if off_bytes > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %{}_addr_i, %{}_addr_u32, {};\n",
                    prefix, prefix, off_bytes
                ));
            } else {
                ptx.push_str(&format!(
                    "    mov.u32 %{}_addr_i, %{}_addr_u32;\n",
                    prefix, prefix
                ));
            }
            ptx.push_str(&format!(
                "    cvt.rn.f16.f32 %{}_h, %{}_{}_{};\n",
                prefix, acc_prefix, t, i
            ));
            // N3: unconditional store — each warp writes its own distinct
            // (m_tile, n_tile) region of SMEM.
            ptx.push_str(&format!(
                "    st.shared.b16 [%{}_addr_i], %{}_h;\n",
                prefix, prefix
            ));
        }
    }
}

/// Like `emit_dfragment_scatter` but writes a COL-MAJOR `[m_dim_block ×
/// head_dim]` f16 tile to SMEM. Used for V only — the PV MMA's
/// B-fragment requires V col-major so that 2 k-adjacent f16 are
/// contiguous in a single b32 read.
///
/// Byte offset for V[k=m_tile*16 + lo_row + row_off, n=n_tile*8 + lo_col_base + col_off]
/// in col-major storage with `m_dim_block` rows and `head_dim` cols:
///   addr = base + n_idx * (m_dim_block * 2) + k_idx * 2
///
/// For the 4 D-fragment positions held per lane:
///   D0 = (lo_row,     lo_col_base    ): tile_off + lo_col_base*(m_block*2) + lo_row*2 + 0
///   D1 = (lo_row,     lo_col_base + 1): D0 + m_block*2
///   D2 = (lo_row + 8, lo_col_base    ): D0 + 16  (8 rows × 2 bytes)
///   D3 = (lo_row + 8, lo_col_base + 1): D0 + m_block*2 + 16
///
/// Per spec section 5.5 the m_tile and n_tile offsets are runtime
/// (warp-distributed via N3 round-robin), same as the row-major variant.
/// Tile-off (per (m_tile, n_tile)) = n_tile*8 * (m_block*2) + m_tile*16*2.
///
/// `m_dim_block` is the rows-of-the-tile-in-SMEM dim. For V that's
/// `block_kv`. `head_dim` is the cols dim (NUMBER of n_tiles is
/// `head_dim / 8`).
///
/// Requires `head_dim / 8` to be a power of 2 (asserted) AND
/// `m_dim_block * 2` to be representable in u32 (always true).
fn emit_dfragment_scatter_col_major(
    ptx: &mut String,
    prefix: &str,
    acc_prefix: &str,
    smem_u64_base: &str,
    m_dim_block: u32,
    head_dim: u32,
    tpw: u32,
) {
    let n_tiles_d = (head_dim / 8).max(1);
    assert!(
        n_tiles_d.is_power_of_two(),
        "emit_dfragment_scatter_col_major requires head_dim/8 to be a power of 2; got {}",
        n_tiles_d
    );
    let log2_n_tiles_d = n_tiles_d.trailing_zeros();
    let n_tiles_d_mask = n_tiles_d - 1;
    let col_stride_bytes = m_dim_block * 2;

    ptx.push_str(&format!(
        "    // === {} D-fragment scatter COL-MAJOR ({} tiles, m_dim={}, head_dim={}, N3) ===\n",
        prefix, tpw, m_dim_block, head_dim
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_laneid, %{}_lo_row, %{}_lo_col_base;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_row_off, %{}_col_off, %{}_lane_off;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_base_u32, %{}_addr_u32, %{}_addr_i;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_global_t, %{}_m_tile, %{}_n_tile;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %{}_m_off, %{}_n_off, %{}_tile_off;\n",
        prefix, prefix, prefix
    ));
    ptx.push_str(&format!("    .reg .b16 %{}_h;\n", prefix));

    // Per-lane (lo_row, lo_col_base, lane_off) computed once.
    ptx.push_str(&format!("    mov.u32 %{}_laneid, %tid.x;\n", prefix));
    ptx.push_str(&format!(
        "    and.b32 %{}_laneid, %{}_laneid, 31;\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    shr.u32 %{}_lo_row, %{}_laneid, 2;       // l/4\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    and.b32 %{}_lo_col_base, %{}_laneid, 3;  // l%4\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    shl.b32 %{}_lo_col_base, %{}_lo_col_base, 1; // *2\n",
        prefix, prefix
    ));
    // lane_off = lo_col_base * col_stride + lo_row * 2
    ptx.push_str(&format!(
        "    mul.lo.u32 %{}_col_off, %{}_lo_col_base, {};  // lo_col_base * (m_block*2)\n",
        prefix, prefix, col_stride_bytes
    ));
    ptx.push_str(&format!(
        "    shl.b32 %{}_row_off, %{}_lo_row, 1;     // lo_row * 2 bytes\n",
        prefix, prefix
    ));
    ptx.push_str(&format!(
        "    add.u32 %{}_lane_off, %{}_col_off, %{}_row_off;\n",
        prefix, prefix, prefix
    ));

    // u32 sister for the SMEM base.
    ptx.push_str(&format!(
        "    cvt.u32.u64 %{}_base_u32, {};\n",
        prefix, smem_u64_base
    ));

    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // {} slot local_t={} (global_t = warp_id + {}*8 at runtime; N3)\n",
            prefix, t, t
        ));
        if t == 0 {
            ptx.push_str(&format!("    mov.u32 %{}_global_t, %warp_id;\n", prefix));
        } else {
            ptx.push_str(&format!(
                "    add.u32 %{}_global_t, %warp_id, {};\n",
                prefix,
                t * 8
            ));
        }
        ptx.push_str(&format!(
            "    shr.u32 %{}_m_tile, %{}_global_t, {};\n",
            prefix, prefix, log2_n_tiles_d
        ));
        ptx.push_str(&format!(
            "    and.b32 %{}_n_tile, %{}_global_t, {};\n",
            prefix, prefix, n_tiles_d_mask
        ));
        // n_off = n_tile * 8 * col_stride_bytes
        ptx.push_str(&format!(
            "    mul.lo.u32 %{}_n_off, %{}_n_tile, {};\n",
            prefix,
            prefix,
            8 * col_stride_bytes
        ));
        // m_off = m_tile * 16 * 2 bytes = m_tile * 32 bytes
        ptx.push_str(&format!(
            "    shl.b32 %{}_m_off, %{}_m_tile, 5;\n",
            prefix, prefix
        ));
        ptx.push_str(&format!(
            "    add.u32 %{}_tile_off, %{}_n_off, %{}_m_off;\n",
            prefix, prefix, prefix
        ));
        ptx.push_str(&format!(
            "    add.u32 %{}_addr_u32, %{}_base_u32, %{}_tile_off;\n",
            prefix, prefix, prefix
        ));
        ptx.push_str(&format!(
            "    add.u32 %{}_addr_u32, %{}_addr_u32, %{}_lane_off;\n",
            prefix, prefix, prefix
        ));

        // 4 D-frag positions:
        //   i=0: +0
        //   i=1: +col_stride (next n col)
        //   i=2: +16 (next 8 m rows: 8*2 bytes)
        //   i=3: +col_stride + 16
        for i in 0..4u32 {
            let off_bytes: u32 = match i {
                0 => 0,
                1 => col_stride_bytes,
                2 => 16,
                3 => col_stride_bytes + 16,
                _ => unreachable!(),
            };
            if off_bytes > 0 {
                ptx.push_str(&format!(
                    "    add.u32 %{}_addr_i, %{}_addr_u32, {};\n",
                    prefix, prefix, off_bytes
                ));
            } else {
                ptx.push_str(&format!(
                    "    mov.u32 %{}_addr_i, %{}_addr_u32;\n",
                    prefix, prefix
                ));
            }
            ptx.push_str(&format!(
                "    cvt.rn.f16.f32 %{}_h, %{}_{}_{};\n",
                prefix, acc_prefix, t, i
            ));
            ptx.push_str(&format!(
                "    st.shared.b16 [%{}_addr_i], %{}_h;\n",
                prefix, prefix
            ));
        }
    }
}

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the Q
/// projection output (bq × hd). Always at least 1 — small configs
/// (e.g. bq=32, hd=32 → 8 tiles total → 1 per warp) bottom out here.
pub(crate) fn tiles_per_warp(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let hd = config.head_dim as u32;
    let m_tiles = bq / 16;
    let n_tiles = hd / 8;
    let total = m_tiles.max(1) * n_tiles.max(1);
    (total / 8).max(1)
}

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the K or V
/// projection output (bkv × hd). Always at least 1 — small configs
/// (e.g. bkv=32, hd=32 → 8 tiles total → 1 per warp) bottom out here.
/// Analogous to `tiles_per_warp` but uses `bkv` as the M dimension.
pub(crate) fn tiles_per_warp_kv(config: &FlashAttentionConfig) -> u32 {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    let m_tiles = bkv / 16;
    let n_tiles = hd / 8;
    let total = m_tiles.max(1) * n_tiles.max(1);
    (total / 8).max(1)
}

/// Emit the Q projection sub-kernel for Tier B.1.
///
/// Behaviour:
///   1. Declare + zero per-warp Q accumulators (`%q_acc_<t>_<lane>`).
///   2. Null-guard `csha_x_ptr` AND `csha_wq_ptr`; on null, branch to
///      `V2_TIER_B1_Q_PROJ_SKIP`.
///   3. Outer loop over `n_chunks = d_model / chunk`:
///       * cp.async stage `x_q_chunk[chunk_idx]` to SMEM at
///         `w_chunk_offset + 2*chunk*hd*2` (x_q slot).
///       * cp.async stage `Wq_chunk[chunk_idx]` to SMEM at
///         `w_chunk_offset` (W slot 0).
///       * `cp.async.commit_group; cp.async.wait_group 0; bar.sync 0`.
///       * Inner MMA loop: for `t in 0..tiles_per_warp` emit one
///         `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` consuming
///         the staged x_q + Wq SMEM and accumulating into `%q_acc_t_*`.
///   4. Pack f32 accumulators to f16 and scatter into Q SMEM at
///      `tier_b1_q_offset(config)`. End with `bar.sync 0`.
///   5. Land the `V2_TIER_B1_Q_PROJ_SKIP:` label.
///
/// `chunk` is the d_model_chunk size selected by `chunk_config::select`
/// (one of {128, 64, 32, FLOOR}).
///
/// # B1.3 deferrals (FOUR classes of placeholder for B1.4 to replace)
///
/// B1.3 ships structural FSM scaffolding. The kernel ptxas-validates on
/// sm_80 + sm_120, but **will NOT compute correct numerics if launched**
/// — there are four independent layers of placeholder work, each of
/// which B1.4 must replace before the kernel is numerically meaningful:
///
/// 1. **A/B fragment loads use a uniform SMEM address.** Every lane reads
///    from `[%tb1_q_smem_w]` and `[%tb1_q_smem_x]` with no per-(warp, lane,
///    K-step) offset. Real CUTLASS-style per-thread fragment loads should
///    replace this via `matmul_mma::emit_load_a_fragment_smem` and
///    `emit_load_b_fragment_smem` (those helpers already exist).
/// 2. **Warp-ownership predicate (`t % 8 == warp_id`) is NOT emitted.**
///    All 8 warps currently execute every MMA in the inner loop. B1.4 must
///    gate each MMA on a `setp` + `@%pred` against `%warp_id`, or the
///    accumulators end up 8× the intended sum.
/// 3. **cp.async stages 16 bytes per chunk, not the chunk's full
///    `chunk * hd * 2` bytes.** 16 bytes is the only legal cp.async width
///    in PTX ISA 7.0; staging a full chunk requires a per-thread loop with
///    per-thread destination offsets. B1.4 must add that loop.
/// 4. **`d_model % chunk == 0` is a precondition with no upstream guard.**
///    `chunk_config::select` doesn't enforce divisibility today. A
///    debug_assert is added at the top of this function as a tripwire;
///    long-term either the assert moves into `chunk_config::select` as a
///    rejection criterion, OR the chunk loop emits a tail-iteration
///    handler for the remainder.
///
/// Caller guarantees `d_model % chunk == 0` for admitted configs (V3
/// supported-matrix CSV property; debug_assert enforces it at this layer).
pub fn emit_q_projection(ptx: &mut String, config: &FlashAttentionConfig, chunk: u32) {
    let csha = match &config.csha {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier B.1 Q projection: csha=None, no emission\n");
            return;
        }
    };
    let d_model = csha.d_model;
    if d_model == 0 || chunk == 0 {
        ptx.push_str("    // Tier B.1 Q projection: d_model=0 or chunk=0, no emission\n");
        return;
    }
    // B1.3 deferral #4 tripwire: chunk_config::select does not currently
    // reject non-divisible (d_model, chunk) pairs. If a non-divisible pair
    // reaches this emitter (e.g., user-overridden chunk param, or future
    // model_config with non-power-of-2 d_model), the integer division
    // below silently truncates and the kernel under-computes Q. B1.4
    // should either (a) add divisibility check to chunk_config::select
    // as a rejection criterion, OR (b) emit a tail-iteration handler.
    debug_assert!(
        d_model % chunk == 0,
        "Tier B.1 Q projection: d_model ({}) must be divisible by chunk ({}); chunk_config::select missing divisibility check (see B1.4 deferral #4)",
        d_model,
        chunk
    );
    let n_chunks = d_model / chunk;
    let hd = config.head_dim as u32;
    let bq = config.block_q as u32;
    let tpw = tiles_per_warp(config);

    // N3 constants: warp-distributed (m_tile, n_tile) over (bq/16, hd/8).
    let n_tiles_d_q = (hd / 8).max(1);
    assert!(
        n_tiles_d_q.is_power_of_two(),
        "N3: Q projection requires head_dim/8 to be a power of 2; got {}",
        n_tiles_d_q
    );
    let log2_n_tiles_d_q = n_tiles_d_q.trailing_zeros();
    let n_tiles_d_q_mask = n_tiles_d_q - 1;
    // m_tile stride in x_q SMEM = 16 rows × chunk cols × 2 bytes.
    let x_m_stride_bytes = 16 * chunk * 2;
    assert!(
        x_m_stride_bytes.is_power_of_two(),
        "N3: Q projection requires 16*chunk*2 to be a power of 2; got {}",
        x_m_stride_bytes
    );
    let log2_x_m_stride = x_m_stride_bytes.trailing_zeros();
    // B-frag (Wq) n_tile band stride. W is stored COL-MAJOR within each
    // chunk band as [hd, chunk] f16, so one output column spans `chunk`
    // contiguous f16 = chunk*2 bytes, and an 8-column n_tile band spans
    // 8*chunk*2 bytes. (The prior emission used a hardcoded shift of 4 =
    // n_tile*16 bytes, which only advanced 8 f16 within column 0's chunk
    // run, so n_tile>0 warps read column-0 weights -> garbage.)
    let w_n_band_bytes = 8 * chunk * 2;
    assert!(
        w_n_band_bytes.is_power_of_two(),
        "N3: Q projection requires 8*chunk*2 (W n_tile band stride) to be a power of 2; got {}",
        w_n_band_bytes
    );
    let log2_w_n_band = w_n_band_bytes.trailing_zeros();
    // N1c: Q-projection K-loop count = chunk / 16.
    assert!(
        chunk.is_multiple_of(16),
        "N1c: Q-projection K-loop requires chunk divisible by 16; got {}",
        chunk
    );
    let n_k_iters_qproj = chunk / 16;

    // SMEM offsets (computed inline so the function is self-contained).
    let w_chunk_off = tier_b1_w_chunk_offset(config);
    // Two W slots reserved by validate_tier_b1_config; x_q starts after both.
    let x_chunk_off = w_chunk_off + 2 * chunk * hd * 2;
    let q_out_off = tier_b1_q_offset(config);

    ptx.push_str(&format!(
        "    // === Tier B.1 Q projection: bq={} hd={} d_model={} chunk={} n_chunks={} tiles_per_warp={} ===\n",
        bq, hd, d_model, chunk, n_chunks, tpw
    ));

    // ----- 1. Scratch + addressing regs used by this emitter only.
    // The per-warp Q accumulators (%q_acc_<t>_<lane>) are hoisted to the
    // orchestrator via register_budget::declare_registers (B1.6 deferral
    // #4); this helper just writes into them.
    ptx.push_str("    .reg .u64 %tb1_q_x_ptr, %tb1_q_wq_ptr;\n");
    ptx.push_str("    .reg .pred %tb1_q_xnull, %tb1_q_wqnull;\n");
    ptx.push_str("    .reg .u64 %tb1_q_smem_w, %tb1_q_smem_x, %tb1_q_smem_q;\n");
    // u32 sister registers for matmul_mma::emit_load_*_fragment_smem
    // (B1.6 deferral #1) — the helpers use 32-bit SMEM addressing.
    ptx.push_str("    .reg .u32 %tb1_q_smem_w_u32, %tb1_q_smem_x_u32;\n");
    // Per-thread cp.async byte offset (B1.6 deferral #3): each of the 256
    // threads stages its own 16-byte slot in the chunk.
    ptx.push_str("    .reg .u32 %tb1_q_cp_tid_off;\n");
    ptx.push_str("    .reg .u32 %tb1_q_cp_smem_addr, %tb1_q_cp_hbm_off;\n");
    ptx.push_str("    .reg .u64 %tb1_q_cp_hbm_addr;\n");
    ptx.push_str("    mov.u32 %tb1_q_cp_tid_off, %tid.x;\n");
    ptx.push_str("    shl.b32 %tb1_q_cp_tid_off, %tb1_q_cp_tid_off, 4; // tid * 16 (B1.6 deferral #3)\n");
    ptx.push_str("    .reg .u64 %tb1_q_scatter_addr;\n");
    ptx.push_str("    .reg .b16 %tb1_q_h0, %tb1_q_h1;\n");

    // N3 runtime tile-coords regs (warp-distributed Q-projection MMA).
    ptx.push_str("    .reg .u32 %qmma_global_t, %qmma_m_tile, %qmma_n_tile;\n");
    ptx.push_str("    .reg .u32 %qmma_x_warp_off, %qmma_w_warp_off;\n");
    ptx.push_str("    .reg .u32 %qmma_a_base, %qmma_b_base;\n");
    // N1c K-iter shifted bases.
    ptx.push_str("    .reg .u32 %qmma_a_base_k, %qmma_b_base_k;\n");

    // ----- 2. Null-guards on csha_x_ptr + csha_wq_ptr -----
    ptx.push_str("    // Null-guard csha_x_ptr; on null, skip Q projection (Tier A path).\n");
    ptx.push_str("    ld.param.u64 %tb1_q_x_ptr, [csha_x_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_q_xnull, %tb1_q_x_ptr, 0;\n");
    ptx.push_str("    @%tb1_q_xnull bra V2_TIER_B1_Q_PROJ_SKIP;\n");
    ptx.push_str("    ld.param.u64 %tb1_q_wq_ptr, [csha_wq_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_q_wqnull, %tb1_q_wq_ptr, 0;\n");
    ptx.push_str("    @%tb1_q_wqnull bra V2_TIER_B1_Q_PROJ_SKIP;\n");

    // Pre-compute SMEM bases for the W and x_q chunk slots + Q output tile.
    ptx.push_str(&format!(
        "    add.u64 %tb1_q_smem_w, %shmem_base, {}; // Wq chunk slot (w_chunk_off)\n",
        w_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_q_smem_x, %shmem_base, {}; // x_q chunk slot\n",
        x_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_q_smem_q, %shmem_base, {}; // Q output tile (tier_b1_q_offset)\n",
        q_out_off
    ));
    // Narrow SMEM bases to u32 for the matmul_mma fragment-load helpers.
    ptx.push_str("    cvt.u32.u64 %tb1_q_smem_w_u32, %tb1_q_smem_w;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_q_smem_x_u32, %tb1_q_smem_x;\n");

    // ----- 3. Outer chunk loop -----
    // For B1.3 we unroll the chunk loop so the snapshot diff captures the
    // exact iteration count per config — matches the FSM-gate intent of
    // spec §6.5 (snapshot locks the structure). Real cp.async ping-pong
    // (B1.4) will collapse this into a single rotated loop body.
    for chunk_idx in 0..n_chunks {
        ptx.push_str(&format!(
            "    // -- Q-proj chunk {}/{} (d_model offset {} .. {}) --\n",
            chunk_idx + 1,
            n_chunks,
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));

        // cp.async stage Wq chunk: bytes = chunk * hd * 2 (f16).
        //
        // Source HBM offset: chunk_idx * chunk * hd * 2 from csha_wq_ptr base.
        // Destination SMEM: w_chunk_off (slot 0).
        ptx.push_str(&format!(
            "    // cp.async Wq[{}:{}] -> SMEM[w_chunk_off]\n",
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));
        // B1.6 deferral #3 RESOLVED: multi-iter cp.async unroll.
        // Wq SMEM dest size = chunk * hd * 2 bytes. For chunks where this
        // exceeds 4096 (e.g., canonical chunk=128 hd=32 → 8192 bytes),
        // multiple sub-iters of per-thread cp.async are emitted.
        emit_cp_async_multi_iter(
            ptx,
            chunk * hd * 2,
            "%tb1_q_smem_w_u32",
            "%tb1_q_cp_tid_off",
            "%tb1_q_wq_ptr",
            chunk_idx * chunk * hd * 2,
            "%tb1_q_cp_smem_addr",
            "%tb1_q_cp_hbm_off",
            "%tb1_q_cp_hbm_addr",
        );

        // cp.async stage x_q chunk: SMEM dest = bq * chunk * 2 bytes (f16).
        //
        // **HBM layout assumption** (correct after the chunk-stride fix
        // below): `csha_x_ptr` points to x laid out as `[d_model/chunk,
        // bq, chunk]` f16 chunks-major — i.e. each chunk band is `bq *
        // chunk` f16 elements contiguous in memory, and successive
        // chunk bands are stacked along d_model. With this layout the
        // per-chunk advance is `bq * chunk * 2` bytes and the cp.async
        // fetches exactly the `bq × chunk` tile needed by the SMEM
        // slot. Prior to this fix the stride was `chunk_idx * chunk * 4`
        // — implying both f32 element stride AND single-row-per-chunk
        // addressing. Successive chunks then overlapped by ~94% of
        // their content (chunk_idx=0 → bytes [0, 8192); chunk_idx=1 →
        // bytes [512, 8704); etc.), producing garbage in the projection
        // MMA for any d_model > chunk.
        //
        // Caller responsibility: produce `csha_x_ptr` in this layout
        // upstream. The standard CSHA pipeline runs RMSNorm on x as
        // f32 [bq, d_model] and writes f32 back to `csha_x_ptr`; a
        // narrow-and-chunkify pre-pass (out of scope for Tier B.1's
        // codegen) must transform that f32 buffer into the f16
        // chunks-major layout before this kernel runs.
        ptx.push_str(&format!(
            "    // cp.async x_q[chunk={}] -> SMEM[x_chunk_off]\n",
            chunk_idx
        ));
        emit_cp_async_multi_iter(
            ptx,
            bq * chunk * 2,
            "%tb1_q_smem_x_u32",
            "%tb1_q_cp_tid_off",
            "%tb1_q_x_ptr",
            chunk_idx * bq * chunk * 2,
            "%tb1_q_cp_smem_addr",
            "%tb1_q_cp_hbm_off",
            "%tb1_q_cp_hbm_addr",
        );

        // Commit + wait + barrier: synchronous chunk for B1.3 (B1.4
        // overlaps via ping-pong).
        ptx.push_str("    cp.async.commit_group;\n");
        ptx.push_str("    cp.async.wait_group 0;\n");
        ptx.push_str("    bar.sync 0;\n");

        // ----- 3b. Inner MMA loop (N3 warp-distributed) -----
        // For each slot local_t this warp owns, emit one m16n8k16. The
        // warp's global tile index is `warp_id + local_t * 8`, mapped to
        // (m_tile, n_tile) at PTX runtime. A-frag base shifts by m_tile,
        // B-frag base shifts by n_tile, and each warp writes its own slot.
        for t in 0..tpw {
            // N3: compute warp-specific (m_tile, n_tile) at runtime.
            if t == 0 {
                ptx.push_str("    mov.u32 %qmma_global_t, %warp_id;\n");
            } else {
                ptx.push_str(&format!(
                    "    add.u32 %qmma_global_t, %warp_id, {};\n",
                    t * 8
                ));
            }
            ptx.push_str(&format!(
                "    shr.u32 %qmma_m_tile, %qmma_global_t, {};\n",
                log2_n_tiles_d_q
            ));
            ptx.push_str(&format!(
                "    and.b32 %qmma_n_tile, %qmma_global_t, {};\n",
                n_tiles_d_q_mask
            ));
            // A-frag x_q base = x_q_smem + m_tile * 16 * chunk * 2 bytes.
            ptx.push_str(&format!(
                "    shl.b32 %qmma_x_warp_off, %qmma_m_tile, {};\n",
                log2_x_m_stride
            ));
            ptx.push_str(
                "    add.u32 %qmma_a_base, %tb1_q_smem_x_u32, %qmma_x_warp_off;\n",
            );
            // B-frag Wq base = Wq_smem + n_tile * 8 * chunk * 2 bytes
            // (W is col-major [hd, chunk] f16, so an 8-col band = 8*chunk*2).
            ptx.push_str(&format!(
                "    shl.b32 %qmma_w_warp_off, %qmma_n_tile, {};\n",
                log2_w_n_band
            ));
            ptx.push_str(
                "    add.u32 %qmma_b_base, %tb1_q_smem_w_u32, %qmma_w_warp_off;\n",
            );

            ptx.push_str(&format!(
                "    .reg .b32 %tb1_q_a_{}_{}_0, %tb1_q_a_{}_{}_1, %tb1_q_a_{}_{}_2, %tb1_q_a_{}_{}_3;\n",
                chunk_idx, t, chunk_idx, t, chunk_idx, t, chunk_idx, t
            ));
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_q_b_{}_{}_0, %tb1_q_b_{}_{}_1;\n",
                chunk_idx, t, chunk_idx, t
            ));

            // N1c: Q-projection K-loop. K-dim = chunk; for chunk > 16 we
            // need chunk/16 K-iters, accumulating into the same
            // %q_acc_<t>_*. Shifts:
            //   - A (x_q[bq × chunk]) base by k_iter * 32 bytes (col shift)
            //   - B (Wq[chunk × hd]) base by k_iter * 16 * hd * 2 bytes
            //     (row shift down chunk dim, each row spans hd cols × 2 bytes)
            for k_iter in 0..n_k_iters_qproj {
                let a_base_expr: String = if k_iter == 0 {
                    "%qmma_a_base".to_string()
                } else {
                    let off = k_iter * 32;
                    ptx.push_str(&format!(
                        "    add.u32 %qmma_a_base_k, %qmma_a_base, {}; // Q-proj A k_iter={}\n",
                        off, k_iter
                    ));
                    "%qmma_a_base_k".to_string()
                };
                let b_base_expr: String = if k_iter == 0 {
                    "%qmma_b_base".to_string()
                } else {
                    // Wq is stored COL-MAJOR within each chunk band as
                    // `[hd, chunk]` (host pre-pass `w_to_col_major_
                    // chunked_f16`). K-iter advances 16 K-positions =
                    // 16 × 2 = 32 bytes along the col-major k-step.
                    // The prior emission used `k_iter * 16 * hd * 2`,
                    // which would be correct only if Wq were row-major
                    // [chunk, hd]; in the actual col-major layout that
                    // overshoots by exactly `hd` (= one full hd-tile of
                    // cols) per K-iter, so successive K-iters read
                    // entirely different n-columns rather than 16 more
                    // K-positions of the same column. The full kernel
                    // e2e exposed this because the projection accumulator
                    // mixed unrelated weight cols together; snapshot/
                    // ptxas/SASS tests don't catch it (none execute the
                    // kernel against known numerical data).
                    let off = k_iter * 32;
                    ptx.push_str(&format!(
                        "    add.u32 %qmma_b_base_k, %qmma_b_base, {}; // Q-proj B k_iter={}\n",
                        off, k_iter
                    ));
                    "%qmma_b_base_k".to_string()
                };

                let a_fragment_regs = [
                    format!("tb1_q_a_{}_{}_0", chunk_idx, t),
                    format!("tb1_q_a_{}_{}_1", chunk_idx, t),
                    format!("tb1_q_a_{}_{}_2", chunk_idx, t),
                    format!("tb1_q_a_{}_{}_3", chunk_idx, t),
                ];
                emit_load_a_fragment_smem(
                    ptx,
                    &a_fragment_regs,
                    &a_base_expr,
                    (chunk * 2) as usize,
                );
                let b_fragment_regs = [
                    format!("tb1_q_b_{}_{}_0", chunk_idx, t),
                    format!("tb1_q_b_{}_{}_1", chunk_idx, t),
                ];
                // B-frag col_stride = chunk * 2 bytes. The new B-frag
                // helper (commit f6c9958b) interprets `row_stride_bytes`
                // as the stride between adjacent N columns in COL-MAJOR
                // [hd_n, chunk_k] SMEM (each col = chunk * f16 = chunk*2
                // bytes). Prior emission passed `hd * 2 = 64` under the
                // OLD helper's row-major interpretation; with the new
                // helper that stride misaddresses the B operand by ~4×
                // (canonical chunk=128 → correct stride 256, was 64).
                // SMEM HBM layout assumption: caller must store Wq
                // col-major within each chunk band — `[d_model/chunk,
                // hd, chunk]` f16. See the rustdoc at the top of this
                // file for the full layout precondition.
                emit_load_b_fragment_smem(
                    ptx,
                    &b_fragment_regs,
                    &b_base_expr,
                    (chunk * 2) as usize,
                );
                let d_regs = [
                    format!("%q_acc_{}_0", t),
                    format!("%q_acc_{}_1", t),
                    format!("%q_acc_{}_2", t),
                    format!("%q_acc_{}_3", t),
                ];
                let a_regs = [
                    format!("%tb1_q_a_{}_{}_0", chunk_idx, t),
                    format!("%tb1_q_a_{}_{}_1", chunk_idx, t),
                    format!("%tb1_q_a_{}_{}_2", chunk_idx, t),
                    format!("%tb1_q_a_{}_{}_3", chunk_idx, t),
                ];
                let b_regs = [
                    format!("%tb1_q_b_{}_{}_0", chunk_idx, t),
                    format!("%tb1_q_b_{}_{}_1", chunk_idx, t),
                ];
                let c_regs = d_regs.clone();
                // N1c: accumulate into same %q_acc_<t>_* across K-iters.
                emit_mma_instruction(ptx, &d_regs, &a_regs, &b_regs, &c_regs);
            }
        }
    }

    // ----- 4. Pack f32 -> f16 and scatter Q accumulators to SMEM tile -----
    //
    // Per spec §5.5: each thread holds 4 f32 accumulator lanes per tile
    // in the m16n8k16 D-fragment layout, scattered to a row-major
    // [bq × head_dim] f16 tile at `tier_b1_q_offset`. The previous
    // flat-address placeholder (`t*8 + lane*2`) had all 32 lanes writing
    // the same address — replaced by `emit_dfragment_scatter` which uses
    // proper D-fragment lane mapping + per-tile warp gating.
    emit_dfragment_scatter(ptx, "qsc", "q_acc", "%tb1_q_smem_q", hd, tpw);
    ptx.push_str("    bar.sync 0; // FENCE: Q tile visible before downstream phases\n");

    // ----- 5. Null-guard skip label -----
    ptx.push_str("V2_TIER_B1_Q_PROJ_SKIP:\n");
}

/// Emit the K and V projection sub-kernel for Tier B.1 (B1.5 Phase A body).
///
/// Behaviour:
///   1. Declare + zero per-warp K accumulators (`%k_acc_<t>_<lane>`) AND
///      V accumulators (`%v_acc_<t>_<lane>`), `tiles_per_warp_kv(config)`
///      tiles each.
///   2. Null-guard `csha_x_ptr`, `csha_wk_ptr`, AND `csha_wv_ptr`; on null,
///      branch to `V2_TIER_B1_KV_PROJ_SKIP_<slot>`.
///   3. Outer loop over `n_chunks = d_model / chunk`:
///       * cp.async stage `Wk_chunk[chunk_idx]` to SMEM at
///         `w_chunk_off` (slot 0).
///       * cp.async stage `Wv_chunk[chunk_idx]` to SMEM at
///         `w_chunk_off + chunk*hd*2` (slot 1).
///       * cp.async stage `x_kv_chunk[chunk_idx]` to SMEM at
///         `w_chunk_off + 2*chunk*hd*2 + bq*chunk*2` (x_kv slot).
///       * `cp.async.commit_group; cp.async.wait_group 0; bar.sync 0`.
///       * Inner MMA loop: for `t in 0..tiles_per_warp_kv` emit TWO MMAs
///         per tile — one accumulating into `%k_acc_t_*` (Wk B-fragment),
///         one into `%v_acc_t_*` (Wv B-fragment). Both share the same x_kv
///         A-fragment registers.
///   4. Pack K f32 -> f16 and scatter into K SMEM at `tier_b1_k_offset_ping`
///      (when `slot == 0`) or `tier_b1_k_offset_pong` (when `slot == 1`).
///      Pack V f32 -> f16 and scatter into V SMEM at the corresponding
///      `tier_b1_v_offset_ping` / `tier_b1_v_offset_pong`. End with `bar.sync 0`.
///   5. Land the `V2_TIER_B1_KV_PROJ_SKIP_<slot>:` label.
///
/// `chunk` is the d_model_chunk size selected by `chunk_config::select`
/// (one of {128, 64, 32, FLOOR}).
///
/// `kv_iter` is the outer KV-tile iteration index (0-based). Used to
/// compute the x_kv HBM row-block offset: `kv_iter * bkv * d_model * 4`.
///
/// `slot` is the ping/pong slot index (0 or 1), compile-time-known at
/// emission. Controls which K/V SMEM slot the accumulators scatter into.
/// The skip label is parameterised by `slot` to avoid label collision when
/// the orchestrator calls this function twice with different slots.
///
/// # B1.6 deferrals (carry-forward from Q-projection's B1.4 deferrals)
///
/// The same four independent layers of placeholder work apply here as in
/// `emit_q_projection`. This function ships structural FSM scaffolding that
/// ptxas-validates on sm_80 + sm_120 but will NOT compute correct numerics
/// if launched. B1.6 must replace all four:
///
/// 1. **A/B fragment loads use a uniform SMEM address.** (See B1.4 TODO in
///    `emit_q_projection` deferral #1.) K/V markers: `tb1_kv_smem_x`,
///    `tb1_kv_smem_wk`, `tb1_kv_smem_wv`.
/// 2. **Warp-ownership predicate (`t % 8 == warp_id`) is NOT emitted.**
///    (See B1.4 TODO in `emit_q_projection` deferral #2.) All 8 warps
///    execute every MMA tile, giving 8x the intended sum.
/// 3. **cp.async stages 16 bytes per chunk, not the chunk's full bytes.**
///    (See B1.4 TODO in `emit_q_projection` deferral #3.) Three cp.async
///    per chunk instead of Q's two; each still only stages 16 bytes.
/// 4. **`d_model % chunk == 0` has no upstream guard.**
///    (See B1.4 TODO in `emit_q_projection` deferral #4.) debug_assert
///    tripwire added below; long-term fix belongs in `chunk_config::select`.
///
/// Caller guarantees `d_model % chunk == 0` for admitted configs (V3
/// supported-matrix CSV property; debug_assert enforces it at this layer).
pub fn emit_kv_projection_chunk_loop(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    chunk: u32,
    kv_iter: u32,
    slot: u32,
) {
    let csha = match &config.csha {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier B.1 KV projection: csha=None, no emission\n");
            return;
        }
    };
    let d_model = csha.d_model;
    if d_model == 0 || chunk == 0 {
        ptx.push_str("    // Tier B.1 KV projection: d_model=0 or chunk=0, no emission\n");
        return;
    }
    // B1.6 deferral #4 tripwire: chunk_config::select does not currently
    // reject non-divisible (d_model, chunk) pairs. If a non-divisible pair
    // reaches this emitter, the integer division below silently truncates and
    // the kernel under-computes K and V. B1.6 should either (a) add
    // divisibility check to chunk_config::select as a rejection criterion,
    // OR (b) emit a tail-iteration handler. See matching marker in
    // emit_q_projection (B1.4 deferral #4).
    debug_assert!(
        d_model % chunk == 0,
        "Tier B.1 KV projection: d_model ({}) must be divisible by chunk ({}); \
         chunk_config::select missing divisibility check (see B1.6 deferral #4)",
        d_model,
        chunk
    );
    let n_chunks = d_model / chunk;
    let hd = config.head_dim as u32;
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let tpw = tiles_per_warp_kv(config);

    // N3 constants: warp-distributed (m_tile, n_tile) over (bkv/16, hd/8).
    let n_tiles_d_kv = (hd / 8).max(1);
    assert!(
        n_tiles_d_kv.is_power_of_two(),
        "N3: KV projection requires head_dim/8 to be a power of 2; got {}",
        n_tiles_d_kv
    );
    let log2_n_tiles_d_kv = n_tiles_d_kv.trailing_zeros();
    let n_tiles_d_kv_mask = n_tiles_d_kv - 1;
    // m_tile stride in x_kv SMEM = 16 rows × chunk cols × 2 bytes.
    let x_m_stride_kv_bytes = 16 * chunk * 2;
    assert!(
        x_m_stride_kv_bytes.is_power_of_two(),
        "N3: KV projection requires 16*chunk*2 to be a power of 2; got {}",
        x_m_stride_kv_bytes
    );
    let log2_x_m_stride_kv = x_m_stride_kv_bytes.trailing_zeros();
    // B-frag (Wk/Wv) n_tile band stride. W is stored COL-MAJOR within each
    // chunk band as [hd, chunk] f16, so one output column spans `chunk`
    // contiguous f16 = chunk*2 bytes, and an 8-column n_tile band spans
    // 8*chunk*2 bytes. (The prior emission used a hardcoded shift of 4 =
    // n_tile*16 bytes, which only advanced 8 f16 within column 0's chunk
    // run, so n_tile>0 warps read column-0 weights -> garbage. One band
    // stride feeds both the Wk and Wv B-frag bases.)
    let w_n_band_bytes_kv = 8 * chunk * 2;
    assert!(
        w_n_band_bytes_kv.is_power_of_two(),
        "N3: KV projection requires 8*chunk*2 (W n_tile band stride) to be a power of 2; got {}",
        w_n_band_bytes_kv
    );
    let log2_w_n_band_kv = w_n_band_bytes_kv.trailing_zeros();
    // N1d: K/V-projection K-loop count = chunk / 16.
    assert!(
        chunk.is_multiple_of(16),
        "N1d: KV-projection K-loop requires chunk divisible by 16; got {}",
        chunk
    );
    let n_k_iters_kvproj = chunk / 16;

    // SMEM offsets (computed inline so the function is self-contained).
    let w_chunk_off = tier_b1_w_chunk_offset(config);
    // slot 0: Wk chunk; slot 1: Wv chunk (each chunk*hd*2 bytes).
    let wk_chunk_off = w_chunk_off;
    let wv_chunk_off = w_chunk_off + chunk * hd * 2;
    // x_kv staging lives after both W slots + x_q slot.
    let x_kv_off = w_chunk_off + 2 * chunk * hd * 2 + bq * chunk * 2;
    // K/V output tile: compile-time slot selection (slot is a u32 parameter).
    let k_out_off = if slot == 0 {
        tier_b1_k_offset_ping(config)
    } else {
        tier_b1_k_offset_pong(config)
    };
    let v_out_off = if slot == 0 {
        tier_b1_v_offset_ping(config)
    } else {
        tier_b1_v_offset_pong(config)
    };
    // x_kv HBM row-block offset for kv_iter.
    //
    // **HBM layout assumption** (post chunk-stride fix): `csha_x_ptr`
    // (the kv slice — typically the same buffer as the q slice) is
    // laid out as `[n_kv_iters, d_model/chunk, bkv, chunk]` f16 — that
    // is, the kv_iter "block" is the outermost dim, each block is
    // `d_model * bkv` f16 elements = `bkv * d_model * 2` bytes, and
    // within a block the chunks-major layout applies.
    // Prior to the fix this was `* 4` (f32 stride) under the assumption
    // that x_kv was f32 in HBM; with the new layout the stride is `* 2`
    // (f16). The caller must produce x_kv in this layout upstream
    // (narrow + chunkify + iter-block).
    let x_kv_hbm_base_off = kv_iter * bkv * d_model * 2; // f16 = 2 bytes

    ptx.push_str(&format!(
        "    // === Tier B.1 KV projection: bkv={} hd={} d_model={} chunk={} n_chunks={} \
         kv_iter={} slot={} tiles_per_warp_kv={} ===\n",
        bkv, hd, d_model, chunk, n_chunks, kv_iter, slot, tpw
    ));

    // ----- 1. Scratch + addressing regs used by this emitter only.
    // %k_acc_<t>_<lane> and %v_acc_<t>_<lane> are hoisted to the
    // orchestrator via register_budget::declare_registers (B1.6 deferral
    // #4); this helper just writes into them.
    ptx.push_str("    .reg .u64 %tb1_kv_x_ptr, %tb1_kv_wk_ptr, %tb1_kv_wv_ptr;\n");
    ptx.push_str("    .reg .pred %tb1_kv_xnull, %tb1_kv_wknull, %tb1_kv_wvnull;\n");
    ptx.push_str(
        "    .reg .u64 %tb1_kv_smem_wk, %tb1_kv_smem_wv, %tb1_kv_smem_x;\n",
    );
    // u32 sister registers for matmul_mma::emit_load_*_fragment_smem
    // (B1.6 deferral #1) — the helpers use 32-bit SMEM addressing.
    ptx.push_str("    .reg .u32 %tb1_kv_smem_wk_u32, %tb1_kv_smem_wv_u32, %tb1_kv_smem_x_u32;\n");
    // Per-thread cp.async byte offset (B1.6 deferral #3): each of the 256
    // threads stages its own 16-byte slot.
    ptx.push_str("    .reg .u32 %tb1_kv_cp_tid_off;\n");
    ptx.push_str("    .reg .u32 %tb1_kv_cp_smem_addr, %tb1_kv_cp_hbm_off;\n");
    ptx.push_str("    .reg .u64 %tb1_kv_cp_hbm_addr;\n");
    ptx.push_str("    mov.u32 %tb1_kv_cp_tid_off, %tid.x;\n");
    ptx.push_str("    shl.b32 %tb1_kv_cp_tid_off, %tb1_kv_cp_tid_off, 4; // tid * 16 (B1.6 deferral #3)\n");
    ptx.push_str("    .reg .u64 %tb1_kv_smem_k, %tb1_kv_smem_v;\n");
    ptx.push_str("    .reg .u64 %tb1_kv_scatter_addr;\n");
    ptx.push_str("    .reg .b16 %tb1_kv_h0;\n");

    // N3 runtime tile-coords regs (warp-distributed K/V MMAs).
    ptx.push_str("    .reg .u32 %kvmma_global_t, %kvmma_m_tile, %kvmma_n_tile;\n");
    ptx.push_str("    .reg .u32 %kvmma_x_warp_off, %kvmma_w_warp_off;\n");
    ptx.push_str("    .reg .u32 %kvmma_a_base, %kvmma_bk_base, %kvmma_bv_base;\n");
    // N1d K-iter shifted bases.
    ptx.push_str(
        "    .reg .u32 %kvmma_a_base_k, %kvmma_bk_base_k, %kvmma_bv_base_k;\n",
    );

    // ----- 2. Null-guards on csha_x_ptr, csha_wk_ptr, csha_wv_ptr -----
    let skip_label = format!("V2_TIER_B1_KV_PROJ_SKIP_{}", slot);
    ptx.push_str("    // Null-guard csha_x_ptr; on null, skip KV projection.\n");
    ptx.push_str("    ld.param.u64 %tb1_kv_x_ptr, [csha_x_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_kv_xnull, %tb1_kv_x_ptr, 0;\n");
    ptx.push_str(&format!("    @%tb1_kv_xnull bra {};\n", skip_label));
    ptx.push_str("    // Null-guard csha_wk_ptr; on null, skip KV projection.\n");
    ptx.push_str("    ld.param.u64 %tb1_kv_wk_ptr, [csha_wk_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_kv_wknull, %tb1_kv_wk_ptr, 0;\n");
    ptx.push_str(&format!("    @%tb1_kv_wknull bra {};\n", skip_label));
    ptx.push_str("    // Null-guard csha_wv_ptr; on null, skip KV projection.\n");
    ptx.push_str("    ld.param.u64 %tb1_kv_wv_ptr, [csha_wv_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_kv_wvnull, %tb1_kv_wv_ptr, 0;\n");
    ptx.push_str(&format!("    @%tb1_kv_wvnull bra {};\n", skip_label));

    // Pre-compute SMEM bases for the Wk, Wv, x_kv chunk slots + K/V output tiles.
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_wk, %shmem_base, {}; // Wk chunk slot (w_chunk_off slot 0)\n",
        wk_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_wv, %shmem_base, {}; // Wv chunk slot (w_chunk_off slot 1)\n",
        wv_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_x, %shmem_base, {}; // x_kv chunk slot (will be narrowed to u32 below)\n",
        x_kv_off
    ));
    // Narrow SMEM bases to u32 for the matmul_mma fragment-load helpers.
    ptx.push_str("    cvt.u32.u64 %tb1_kv_smem_wk_u32, %tb1_kv_smem_wk;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_kv_smem_wv_u32, %tb1_kv_smem_wv;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_kv_smem_x_u32, %tb1_kv_smem_x;\n");
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_k, %shmem_base, {}; // K output tile (slot={})\n",
        k_out_off, slot
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_v, %shmem_base, {}; // V output tile (slot={})\n",
        v_out_off, slot
    ));

    // ----- 3. Outer chunk loop -----
    // For B1.5 Phase A we unroll the chunk loop so the snapshot diff captures
    // the exact iteration count per config, matching the FSM-gate intent of
    // spec §6.5. The B1.6 cp.async ping-pong pipeline will collapse this into
    // a single rotated loop body.
    for chunk_idx in 0..n_chunks {
        ptx.push_str(&format!(
            "    // -- KV-proj chunk {}/{} (d_model offset {} .. {}) --\n",
            chunk_idx + 1,
            n_chunks,
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));

        // cp.async stage Wk chunk: bytes = chunk * hd * 2 (f16).
        //
        // Source HBM offset: chunk_idx * chunk * hd * 2 from csha_wk_ptr base.
        // Destination SMEM: wk_chunk_off (slot 0).
        ptx.push_str(&format!(
            "    // cp.async Wk[{}:{}] -> SMEM[wk_chunk_off slot 0]\n",
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));
        // B1.6 deferral #3 RESOLVED: multi-iter cp.async for Wk chunk.
        emit_cp_async_multi_iter(
            ptx,
            chunk * hd * 2,
            "%tb1_kv_smem_wk_u32",
            "%tb1_kv_cp_tid_off",
            "%tb1_kv_wk_ptr",
            chunk_idx * chunk * hd * 2,
            "%tb1_kv_cp_smem_addr",
            "%tb1_kv_cp_hbm_off",
            "%tb1_kv_cp_hbm_addr",
        );

        // cp.async stage Wv chunk: bytes = chunk * hd * 2 (f16).
        //
        // Source HBM offset: chunk_idx * chunk * hd * 2 from csha_wv_ptr base.
        // Destination SMEM: wv_chunk_off (slot 1).
        ptx.push_str(&format!(
            "    // cp.async Wv[{}:{}] -> SMEM[wv_chunk_off slot 1]\n",
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));
        // B1.6 deferral #3 RESOLVED: multi-iter cp.async for Wv chunk.
        emit_cp_async_multi_iter(
            ptx,
            chunk * hd * 2,
            "%tb1_kv_smem_wv_u32",
            "%tb1_kv_cp_tid_off",
            "%tb1_kv_wv_ptr",
            chunk_idx * chunk * hd * 2,
            "%tb1_kv_cp_smem_addr",
            "%tb1_kv_cp_hbm_off",
            "%tb1_kv_cp_hbm_addr",
        );

        // cp.async stage x_kv chunk: bytes = bkv * chunk * 2 (f16).
        //
        // Source HBM offset: `x_kv_hbm_base_off + chunk_idx * bkv * chunk * 2`,
        // matching the `[n_kv_iters, d_model/chunk, bkv, chunk]` f16
        // chunks-major layout established for x_q above. Destination SMEM:
        // x_kv_off.
        //
        // Prior emission used `chunk_idx * chunk * 4` — same f32 stride
        // + single-row-per-chunk bug as x_q (chunks overlapped by ~94%).
        ptx.push_str(&format!(
            "    // cp.async x_kv[kv_iter={} chunk={}] -> SMEM[x_kv_off]\n",
            kv_iter, chunk_idx
        ));
        // Multi-iter cp.async for x_kv chunk.
        emit_cp_async_multi_iter(
            ptx,
            bkv * chunk * 2,
            "%tb1_kv_smem_x_u32",
            "%tb1_kv_cp_tid_off",
            "%tb1_kv_x_ptr",
            x_kv_hbm_base_off + chunk_idx * bkv * chunk * 2,
            "%tb1_kv_cp_smem_addr",
            "%tb1_kv_cp_hbm_off",
            "%tb1_kv_cp_hbm_addr",
        );

        // Commit + wait + barrier: synchronous chunk for B1.5 Phase A
        // (B1.6 overlaps via ping-pong).
        ptx.push_str("    cp.async.commit_group;\n");
        ptx.push_str("    cp.async.wait_group 0;\n");
        ptx.push_str("    bar.sync 0;\n");

        // ----- 3b. Inner MMA loop (TWO MMAs per tile: K and V) -----
        // For each output tile `t` owned by this warp, emit:
        //   (a) one mma.sync.aligned.m16n8k16 into K accumulators using Wk
        //       B-fragment.
        //   (b) one mma.sync.aligned.m16n8k16 into V accumulators using Wv
        //       B-fragment.
        // Both MMAs share the same x_kv A-fragment registers (x_kv is the
        // shared input to both K and V projections).
        for t in 0..tpw {
            // --- Declare A/B fragment registers ---
            // A-fragment: x_kv rows (shared between K and V MMAs).
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_kv_a_{}_{}_0, %tb1_kv_a_{}_{}_1, \
                 %tb1_kv_a_{}_{}_2, %tb1_kv_a_{}_{}_3;\n",
                chunk_idx, t, chunk_idx, t, chunk_idx, t, chunk_idx, t
            ));
            // B-fragment for K (Wk rows).
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_kv_bk_{}_{}_0, %tb1_kv_bk_{}_{}_1;\n",
                chunk_idx, t, chunk_idx, t
            ));
            // B-fragment for V (Wv rows).
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_kv_bv_{}_{}_0, %tb1_kv_bv_{}_{}_1;\n",
                chunk_idx, t, chunk_idx, t
            ));

            // N3: compute warp-specific (m_tile, n_tile) at runtime.
            if t == 0 {
                ptx.push_str("    mov.u32 %kvmma_global_t, %warp_id;\n");
            } else {
                ptx.push_str(&format!(
                    "    add.u32 %kvmma_global_t, %warp_id, {};\n",
                    t * 8
                ));
            }
            ptx.push_str(&format!(
                "    shr.u32 %kvmma_m_tile, %kvmma_global_t, {};\n",
                log2_n_tiles_d_kv
            ));
            ptx.push_str(&format!(
                "    and.b32 %kvmma_n_tile, %kvmma_global_t, {};\n",
                n_tiles_d_kv_mask
            ));
            // A-frag x_kv base = x_kv_smem + m_tile * 16 * chunk * 2 bytes.
            ptx.push_str(&format!(
                "    shl.b32 %kvmma_x_warp_off, %kvmma_m_tile, {};\n",
                log2_x_m_stride_kv
            ));
            ptx.push_str(
                "    add.u32 %kvmma_a_base, %tb1_kv_smem_x_u32, %kvmma_x_warp_off;\n",
            );
            // B-frag Wk/Wv bases = W_smem + n_tile * 8 * chunk * 2 bytes
            // (W is col-major [hd, chunk] f16, so an 8-col band = 8*chunk*2).
            ptx.push_str(&format!(
                "    shl.b32 %kvmma_w_warp_off, %kvmma_n_tile, {};\n",
                log2_w_n_band_kv
            ));
            ptx.push_str(
                "    add.u32 %kvmma_bk_base, %tb1_kv_smem_wk_u32, %kvmma_w_warp_off;\n",
            );
            ptx.push_str(
                "    add.u32 %kvmma_bv_base, %tb1_kv_smem_wv_u32, %kvmma_w_warp_off;\n",
            );

            // N1d: K/V projection K-loop. K-dim = chunk; for chunk > 16
            // we need chunk/16 K-iters, accumulating into the same
            // %k_acc_<t>_* and %v_acc_<t>_*. Shifts:
            //   - A (x_kv[bkv × chunk] row-major) base by k_iter * 32
            //     bytes (k-step = 2 bytes per col, ×16 cols per K-iter).
            //   - B (Wk/Wv stored COL-MAJOR within chunk as [hd, chunk]):
            //     base shifts by k_iter * 32 bytes (k-step = 2 bytes per
            //     row within the col, ×16 rows per K-iter). Prior emission
            //     used `k_iter * 16 * hd * 2` (row-major-stride formula),
            //     which under col-major B overshoots by `hd` (= one full
            //     hd-tile of cols) per K-iter — successive K-iters read
            //     unrelated n-cols instead of advancing along K within
            //     the same col. Same bug as Q-projection (see comment at
            //     the Q-proj K-iter site for the full diagnosis).
            for k_iter in 0..n_k_iters_kvproj {
                let a_base_expr: String = if k_iter == 0 {
                    "%kvmma_a_base".to_string()
                } else {
                    let off = k_iter * 32;
                    ptx.push_str(&format!(
                        "    add.u32 %kvmma_a_base_k, %kvmma_a_base, {}; // KV-proj A k_iter={}\n",
                        off, k_iter
                    ));
                    "%kvmma_a_base_k".to_string()
                };
                let bk_base_expr: String = if k_iter == 0 {
                    "%kvmma_bk_base".to_string()
                } else {
                    let off = k_iter * 32;
                    ptx.push_str(&format!(
                        "    add.u32 %kvmma_bk_base_k, %kvmma_bk_base, {}; // KV-proj Bk k_iter={}\n",
                        off, k_iter
                    ));
                    "%kvmma_bk_base_k".to_string()
                };
                let bv_base_expr: String = if k_iter == 0 {
                    "%kvmma_bv_base".to_string()
                } else {
                    let off = k_iter * 32;
                    ptx.push_str(&format!(
                        "    add.u32 %kvmma_bv_base_k, %kvmma_bv_base, {}; // KV-proj Bv k_iter={}\n",
                        off, k_iter
                    ));
                    "%kvmma_bv_base_k".to_string()
                };

                // A-fragment: x_kv rows. Shared between K and V MMAs.
                let a_fragment_regs = [
                    format!("tb1_kv_a_{}_{}_0", chunk_idx, t),
                    format!("tb1_kv_a_{}_{}_1", chunk_idx, t),
                    format!("tb1_kv_a_{}_{}_2", chunk_idx, t),
                    format!("tb1_kv_a_{}_{}_3", chunk_idx, t),
                ];
                emit_load_a_fragment_smem(
                    ptx,
                    &a_fragment_regs,
                    &a_base_expr,
                    (chunk * 2) as usize,
                );

                // K B-fragment: Wk cols (warp-specific via n_tile).
                // Same col_stride = chunk * 2 as Wq B-frag (see comment
                // at Q-projection B-frag call site). HBM Wk must be
                // stored col-major within each chunk band:
                // `[d_model/chunk, hd, chunk]` f16.
                let bk_fragment_regs = [
                    format!("tb1_kv_bk_{}_{}_0", chunk_idx, t),
                    format!("tb1_kv_bk_{}_{}_1", chunk_idx, t),
                ];
                emit_load_b_fragment_smem(
                    ptx,
                    &bk_fragment_regs,
                    &bk_base_expr,
                    (chunk * 2) as usize,
                );

                // V B-fragment: Wv cols. Same convention as Wk.
                let bv_fragment_regs = [
                    format!("tb1_kv_bv_{}_{}_0", chunk_idx, t),
                    format!("tb1_kv_bv_{}_{}_1", chunk_idx, t),
                ];
                emit_load_b_fragment_smem(
                    ptx,
                    &bv_fragment_regs,
                    &bv_base_expr,
                    (chunk * 2) as usize,
                );

                // --- MMA (a): K accumulation ---
                let k_d_regs = [
                    format!("%k_acc_{}_0", t),
                    format!("%k_acc_{}_1", t),
                    format!("%k_acc_{}_2", t),
                    format!("%k_acc_{}_3", t),
                ];
                let a_regs = [
                    format!("%tb1_kv_a_{}_{}_0", chunk_idx, t),
                    format!("%tb1_kv_a_{}_{}_1", chunk_idx, t),
                    format!("%tb1_kv_a_{}_{}_2", chunk_idx, t),
                    format!("%tb1_kv_a_{}_{}_3", chunk_idx, t),
                ];
                let bk_regs = [
                    format!("%tb1_kv_bk_{}_{}_0", chunk_idx, t),
                    format!("%tb1_kv_bk_{}_{}_1", chunk_idx, t),
                ];
                let k_c_regs = k_d_regs.clone();
                // N1d: accumulate into same %k_acc_<t>_* across K-iters.
                emit_mma_instruction(ptx, &k_d_regs, &a_regs, &bk_regs, &k_c_regs);

                // --- MMA (b): V accumulation ---
                let v_d_regs = [
                    format!("%v_acc_{}_0", t),
                    format!("%v_acc_{}_1", t),
                    format!("%v_acc_{}_2", t),
                    format!("%v_acc_{}_3", t),
                ];
                let bv_regs = [
                    format!("%tb1_kv_bv_{}_{}_0", chunk_idx, t),
                    format!("%tb1_kv_bv_{}_{}_1", chunk_idx, t),
                ];
                let v_c_regs = v_d_regs.clone();
                // N1d: accumulate into same %v_acc_<t>_* across K-iters.
                emit_mma_instruction(ptx, &v_d_regs, &a_regs, &bv_regs, &v_c_regs);
            }
        }
    }

    // ----- 4. Pack f32 -> f16 and scatter K + V accumulators to SMEM -----
    //
    // K written to tier_b1_k_offset_<slot> ROW-MAJOR [bkv × head_dim] f16:
    // QK^T's B-fragment treats K-transposed as col-major (B[k=hd_pos,
    // n=bkv_pos]), which coincides byte-for-byte with row-major K storage,
    // so K stays row-major.
    //
    // V written to tier_b1_v_offset_<slot> COL-MAJOR [bkv × head_dim] f16
    // (i.e. each head_dim column is contiguous in memory, k-rows step by
    // 2 bytes). PV's B-fragment expects V col-major because the MMA's
    // `.col` operand layout requires k-adjacent f16 to be contiguous in a
    // single b32 — which is true for col-major storage but NOT for
    // row-major. Storing V row-major and reading via the col-major B-frag
    // helper produced effectively P @ scrambled-V (the magnitude-bug
    // root cause documented in PR follow-up #1).
    emit_dfragment_scatter(ptx, "ksc", "k_acc", "%tb1_kv_smem_k", hd, tpw);
    emit_dfragment_scatter_col_major(
        ptx, "vsc", "v_acc", "%tb1_kv_smem_v", config.block_kv as u32, hd, tpw,
    );
    ptx.push_str("    bar.sync 0; // FENCE: K+V tiles visible before downstream phases\n");

    // ----- 5. Null-guard skip label -----
    ptx.push_str(&format!("{}:\n", skip_label));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn make_config(bq: i64, bkv: i64, hd: i64, dm: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: bq,
            block_kv: bkv,
            head_dim: hd,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: dm,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        }
    }

    #[test]
    fn null_guards_appear_exactly_twice() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        assert_eq!(ptx.matches("setp.eq.u64 %tb1_q_").count(), 2,
            "expected 2 null-guards (x_ptr + wq_ptr); got: {}", ptx);
        assert_eq!(ptx.matches("V2_TIER_B1_Q_PROJ_SKIP").count(), 3,
            "expected 3 occurrences of the skip label (2 bra + 1 landing): {}", ptx);
    }

    #[test]
    fn chunk_loop_unrolls_correctly() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        // n_chunks = 2048 / 128 = 16. We emit one cp.async pair per chunk.
        // Count cp.async.commit_group to match n_chunks.
        assert_eq!(ptx.matches("cp.async.commit_group").count(), 16,
            "expected 16 chunk iterations (2048 / 128)");
    }

    #[test]
    fn mma_count_matches_tiles_per_warp_times_chunks_times_k_iters() {
        // N1c: K-loop over chunk/16 added.
        // (bq=32, hd=32, chunk=128): tpw=1, n_chunks=16, n_k_iters=8.
        // Expected MMA count = tpw * n_chunks * n_k_iters = 1*16*8 = 128.
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        let expected = 1 * 16 * 8;
        assert_eq!(mma_count, expected,
            "expected {} MMA instructions (1 tile/warp * 16 chunks * 8 K-iters); got {}\nPTX:\n{}",
            expected, mma_count, ptx);
    }

    #[test]
    fn csha_none_emits_noop_comment() {
        let mut cfg = make_config(32, 32, 32, 2048);
        cfg.csha = None;
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        assert!(ptx.contains("csha=None"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn d_model_zero_emits_noop() {
        let cfg = make_config(32, 32, 32, 0);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        assert!(ptx.contains("d_model=0"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn tiles_per_warp_canonical_64_64_64() {
        // bq=64, hd=64 -> m_tiles=4, n_tiles=8, total=32, tpw=4.
        let cfg = make_config(64, 64, 64, 2048);
        assert_eq!(tiles_per_warp(&cfg), 4);
    }

    #[test]
    fn tiles_per_warp_small_floor() {
        // bq=32, hd=32 -> total=8, tpw = 8/8 = 1.
        let cfg = make_config(32, 32, 32, 2048);
        assert_eq!(tiles_per_warp(&cfg), 1);
    }

    // ---- K/V projection tests (B1.5) ----------------------------------------

    #[test]
    fn kv_null_guards_appear_for_wk_and_wv() {
        // Canonical config: bq=bkv=hd=32, d_model=2048, chunk=128.
        // Expect exactly 2 null-guards for Wk+Wv (plus 1 for x_ptr = 3 total
        // setp.eq.u64 instructions; we count the wk/wv-specific ones).
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        // csha_wk_ptr null-guard:
        assert_eq!(
            ptx.matches("setp.eq.u64 %tb1_kv_wknull").count(),
            1,
            "expected exactly 1 Wk null-guard; got:\n{}",
            ptx
        );
        // csha_wv_ptr null-guard:
        assert_eq!(
            ptx.matches("setp.eq.u64 %tb1_kv_wvnull").count(),
            1,
            "expected exactly 1 Wv null-guard; got:\n{}",
            ptx
        );
        // Combined: 3 setp.eq.u64 total (x_ptr + wk_ptr + wv_ptr).
        assert_eq!(
            ptx.matches("setp.eq.u64 %tb1_kv_").count(),
            3,
            "expected 3 total null-guards (x + wk + wv); got:\n{}",
            ptx
        );
    }

    #[test]
    fn kv_mma_count_is_two_times_tpw_times_chunks_times_k_iters() {
        // N1d: K-loop over chunk/16 added.
        // (bkv=32, hd=32, chunk=128): tpw_kv=1, n_chunks=16, n_k_iters=8.
        // Expected = 2 (K+V) * tpw * n_chunks * n_k_iters = 2*1*16*8 = 256.
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        let tpw_kv = tiles_per_warp_kv(&cfg);
        let n_chunks = 2048 / 128;
        let n_k_iters = 128 / 16;
        let expected = 2 * tpw_kv * n_chunks * n_k_iters;
        assert_eq!(
            mma_count, expected as usize,
            "expected {} MMA instructions (2 * {} tpw_kv * {} chunks * {} K-iters); got {}\nPTX:\n{}",
            expected, tpw_kv, n_chunks, n_k_iters, mma_count, ptx
        );
    }

    #[test]
    fn kv_commit_group_count_matches_n_chunks() {
        // n_chunks = 2048/128 = 16. One cp.async.commit_group per chunk
        // (after the three cp.async issuances for Wk, Wv, x_kv).
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        let commit_count = ptx.matches("cp.async.commit_group").count();
        assert_eq!(
            commit_count, 16,
            "expected 16 cp.async.commit_group (one per chunk); got {}\nPTX:\n{}",
            commit_count, ptx
        );
    }

    #[test]
    fn kv_skip_label_is_slot_parameterised() {
        // Calling with slot=0 and slot=1 must produce different skip labels to
        // avoid collision when the orchestrator emits both slots.
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx0 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx0, &cfg, 128, 0, 0);
        let mut ptx1 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx1, &cfg, 128, 0, 1);
        assert!(
            ptx0.contains("V2_TIER_B1_KV_PROJ_SKIP_0"),
            "slot=0 must use label V2_TIER_B1_KV_PROJ_SKIP_0"
        );
        assert!(
            ptx1.contains("V2_TIER_B1_KV_PROJ_SKIP_1"),
            "slot=1 must use label V2_TIER_B1_KV_PROJ_SKIP_1"
        );
        assert!(
            !ptx0.contains("V2_TIER_B1_KV_PROJ_SKIP_1"),
            "slot=0 output must not contain slot=1 label"
        );
    }

    #[test]
    fn kv_ping_pong_smem_offsets_differ_by_slot() {
        // slot=0 must reference k_offset_ping / v_offset_ping.
        // slot=1 must reference k_offset_pong / v_offset_pong.
        use crate::flash_attention_v2::smem_layout::{
            tier_b1_k_offset_ping, tier_b1_k_offset_pong,
            tier_b1_v_offset_ping, tier_b1_v_offset_pong,
        };
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx0 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx0, &cfg, 128, 0, 0);
        let mut ptx1 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx1, &cfg, 128, 0, 1);

        let k_ping = tier_b1_k_offset_ping(&cfg);
        let k_pong = tier_b1_k_offset_pong(&cfg);
        let v_ping = tier_b1_v_offset_ping(&cfg);
        let v_pong = tier_b1_v_offset_pong(&cfg);

        assert!(
            ptx0.contains(&format!("// K output tile (slot=0)")),
            "slot=0 K tile comment must be present; k_ping={}", k_ping
        );
        assert!(
            ptx0.contains(&format!("// V output tile (slot=0)")),
            "slot=0 V tile comment must be present; v_ping={}", v_ping
        );
        assert!(
            ptx1.contains(&format!("// K output tile (slot=1)")),
            "slot=1 K tile comment must be present; k_pong={}", k_pong
        );
        assert!(
            ptx1.contains(&format!("// V output tile (slot=1)")),
            "slot=1 V tile comment must be present; v_pong={}", v_pong
        );
    }

    #[test]
    fn kv_csha_none_emits_noop_comment() {
        let mut cfg = make_config(32, 32, 32, 2048);
        cfg.csha = None;
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        assert!(ptx.contains("csha=None"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn kv_d_model_zero_emits_noop() {
        let cfg = make_config(32, 32, 32, 0);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        assert!(ptx.contains("d_model=0"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn tiles_per_warp_kv_small_floor() {
        // bkv=32, hd=32 -> m_tiles=2, n_tiles=4, total=8, tpw_kv=1.
        let cfg = make_config(32, 32, 32, 2048);
        assert_eq!(tiles_per_warp_kv(&cfg), 1);
    }

    #[test]
    fn tiles_per_warp_kv_canonical_64_64_64() {
        // bkv=64, hd=64 -> m_tiles=4, n_tiles=8, total=32, tpw_kv=4.
        let cfg = make_config(64, 64, 64, 2048);
        assert_eq!(tiles_per_warp_kv(&cfg), 4);
    }
}
