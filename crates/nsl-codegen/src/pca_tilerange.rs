//! PCA Tier B — runtime per-tile segment-range tile-skip predicate.
//!
//! Builds a runtime range table from segment_ids in SMEM, then emits a
//! warp-uniform branch that skips QK^T matmul + softmax update + PV
//! reduction for `(q_tile, kv_tile)` pairs whose segment ranges are
//! strictly disjoint. Composes with Tier A's per-element segment_mask:
//! Tier B filters tile pairs; Tier A masks cells inside surviving tiles.
//!
//! Spec: `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md`.
//!
//! # Emitter functions
//!
//! - [`emit_range_table_preamble`] — once per kernel, in the preamble.
//! - [`emit_skip_predicate`] — once per `(q_iter, kv_iter)` boundary.
//! - [`emit_skip_decision_writeback`] — gated debug-only; M3 parity.
//!
//! # Scratch register convention
//!
//! All scratch registers introduced by this helper use the suffix
//! `_TILERANGE` (e.g. `%rs_qmin_TILERANGE`, `%p_skip_TILERANGE`) so
//! they don't collide with Tier A's `_SEGMASK` suffix or the FA2
//! emitter's seeded scratch space.

use crate::flash_attention::FlashAttentionConfig;
use crate::pca_segment::SegmentResidency;

/// Four sub-table offsets inside the Tier B range-table region.
/// Spec §3.4.3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RangeTableAddrs {
    pub qtile_min:  u32,
    pub qtile_max:  u32,
    pub kvtile_min: u32,
    pub kvtile_max: u32,
}

/// Compute the four sub-table offsets given the range-table base and
/// tile counts. `base` is assumed 2-byte aligned per
/// `tier_b_range_table_offset`'s `align_up(2)` guarantee.
///
/// Layout:
/// ```text
///   base + 0                       : qtile_min  [num_q_tiles  × u16]
///   base + num_q_tiles × 2         : qtile_max  [num_q_tiles  × u16]
///   base + 2 × num_q_tiles × 2     : kvtile_min [num_kv_tiles × u16]
///   base + (2*num_q + num_kv) × 2  : kvtile_max [num_kv_tiles × u16]
/// ```
pub fn range_table_addrs(base: u32, num_q_tiles: u32, num_kv_tiles: u32) -> RangeTableAddrs {
    let q_bytes  = num_q_tiles  * 2;
    let kv_bytes = num_kv_tiles * 2;
    RangeTableAddrs {
        qtile_min:  base,
        qtile_max:  base + q_bytes,
        kvtile_min: base + 2 * q_bytes,
        kvtile_max: base + 2 * q_bytes + kv_bytes,
    }
}

/// Total bytes consumed by the Tier B range table at the given config
/// + seq_len. Used by `shared_mem_bytes_v2{_backward}_with_seqlen` to
/// widen the dynamic-SMEM launch parameter when Tier B is admitted.
pub fn tier_b_range_table_bytes(
    config: &crate::flash_attention::FlashAttentionConfig,
    seq_len: u32,
) -> u32 {
    let num_q  = crate::pca_tile_config::num_tiles(seq_len, config.block_q as u32);
    let num_kv = crate::pca_tile_config::num_tiles(seq_len, config.block_kv as u32);
    2 * (num_q + num_kv) * 2
}

/// Compute the SMEM bytes the range table consumes for a given
/// (block_q, block_kv, seq_len) configuration.
///
/// Formula per spec §3.3:
///
/// ```text
///   bytes = 2 × (num_q_tiles + num_kv_tiles) × sizeof(seg_id_t)
/// ```
///
/// where `seg_id_t = u16` and tile counts are
/// `ceil(seq_len / block_*)`. Both `min` and `max` tables exist per
/// dimension; the `2 ×` covers that.
pub fn compute_range_table_bytes(seq_len: u64, block_q: u64, block_kv: u64) -> u64 {
    let num_q_tiles = seq_len.div_ceil(block_q);
    let num_kv_tiles = seq_len.div_ceil(block_kv);
    2 * (num_q_tiles + num_kv_tiles) * 2 // 2 bytes per u16
}

/// SMEM size threshold above which Tier B falls back to Tier A only
/// (spec §3.3 v1 bound).
pub const TIER_B_RANGE_TABLE_BUDGET_BYTES: u64 = 8192;

/// Decide whether to emit Tier B for a given FA config. Returns false
/// when the range table would exceed the v1 budget OR when Tier A's
/// segment-ID residency is `Streamed` (spec §3.4 — Tier B requires
/// `Shared`).
pub fn should_emit_tier_b(
    config: &FlashAttentionConfig,
    seq_len: u64,
    residency: SegmentResidency,
) -> bool {
    if !config.segment_masked {
        return false;
    }
    if !matches!(residency, SegmentResidency::Shared) {
        return false;
    }
    let bytes = compute_range_table_bytes(seq_len, config.block_q as u64, config.block_kv as u64);
    bytes <= TIER_B_RANGE_TABLE_BUDGET_BYTES
}

/// Emit PTX that builds the per-tile segment-range table at the top
/// of the kernel preamble.
///
/// Reads segment_ids from SMEM (already populated by Tier A), runs a
/// parallel min/max reduction per tile, and stores results into
/// `qtile_seg_min[num_q_tiles] : u16`, `qtile_seg_max[num_q_tiles] : u16`,
/// `kvtile_seg_min[num_kv_tiles] : u16`, `kvtile_seg_max[num_kv_tiles] : u16`.
///
/// Stores values as warp-uniform classified so subsequent
/// [`emit_skip_predicate`] calls compile to `BRA.U`.
///
/// Per spec §3.1 / §4: per-phase uniform-counter loop (BRA.U-eligible
/// branch), warp-shuffle butterfly min/max reduction, lane-0 predicated
/// store. Uniformity-through-load anti-pattern avoided: `%r_tile_*` is
/// register-resident, `mov`-initialized, unconditionally incremented.
pub fn emit_range_table_preamble(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    seq_len: u32,
    segment_ids_smem_base: &str,
    range_table_base: u32,
) {
    let block_q  = config.block_q  as u32;
    let block_kv = config.block_kv as u32;
    let num_q_tiles  = crate::pca_tile_config::num_tiles(seq_len, block_q);
    let num_kv_tiles = crate::pca_tile_config::num_tiles(seq_len, block_kv);
    let addrs = range_table_addrs(range_table_base, num_q_tiles, num_kv_tiles);

    ptx.push_str("    // ----- PCA Tier B: range-table preamble (spec sec 3.1, v2) -----\n");
    ptx.push_str(&format!(
        "    // num_q_tiles={num_q_tiles}, num_kv_tiles={num_kv_tiles}, base=0x{range_table_base:X}\n"
    ));

    // Register declarations (declared once at the preamble; reused
    // across q-phase and kv-phase iterations).
    ptx.push_str("    .reg .u32  %r_tile_q_TILERANGE,  %r_tile_kv_TILERANGE;\n");
    ptx.push_str("    .reg .u16  %rs_min_TILERANGE,   %rs_max_TILERANGE;\n");
    ptx.push_str("    .reg .u16  %rs_peer_min_TILERANGE, %rs_peer_max_TILERANGE;\n");
    ptx.push_str("    .reg .u32  %lane_id_TILERANGE;\n");
    ptx.push_str("    .reg .pred %p_done_TILERANGE,   %p_lane_zero_TILERANGE;\n");
    ptx.push_str("    .reg .u64  %addr_min_TILERANGE, %addr_max_TILERANGE;\n");
    ptx.push_str("    .reg .u64  %seg_smem_TILERANGE, %wide_tile_off_TILERANGE;\n");
    ptx.push_str("    .reg .u32  %seg_byte_off_TILERANGE;\n");
    ptx.push_str("\n");

    ptx.push_str(&format!(
        "    cvta.shared.u64 %seg_smem_TILERANGE, {segment_ids_smem_base};\n"
    ));
    ptx.push_str("    mov.u32 %lane_id_TILERANGE, %laneid;\n");
    ptx.push_str("\n");

    emit_phase(ptx, "q",  num_q_tiles,  block_q,  addrs.qtile_min,  addrs.qtile_max);
    emit_phase(ptx, "kv", num_kv_tiles, block_kv, addrs.kvtile_min, addrs.kvtile_max);

    ptx.push_str("    bar.sync 0;  // range tables visible to all warps before kv-tile loop reads\n");
    ptx.push_str("    // ----- end PCA Tier B range-table preamble -----\n");
}

/// Emit one phase (q-phase or kv-phase) of the range-table reduction.
///
/// Per spec §4 (warp-uniformity discipline):
///   - `%r_tile_*`: `.reg .u32`, initialized via `mov.u32 ..., 0`, incremented unconditionally.
///   - Loop branch consumes a uniform predicate (setp on uniform register vs literal).
///   - Lane-0 store uses PTX predicated execution, not thread-divergent branch.
fn emit_phase(
    ptx: &mut String,
    tag: &str,        // "q" or "kv"
    num_tiles: u32,
    block_size: u32,
    addr_min: u32,
    addr_max: u32,
) {
    let r_tile = format!("%r_tile_{tag}_TILERANGE");
    let loop_label = format!("LOOP_{}_TILERANGE", tag.to_uppercase());

    ptx.push_str(&format!(
        "    // --- {tag}-phase: {num_tiles} tiles x {block_size} tokens, addr_min=0x{addr_min:X}, addr_max=0x{addr_max:X} ---\n"
    ));

    // Uniform counter init.
    ptx.push_str(&format!("    mov.u32 {r_tile}, 0;\n"));
    ptx.push_str(&format!("{loop_label}:\n"));

    // Per-iteration: load segment_ids[tile_start + lane], optional +lane+warp_size if block > 32,
    // butterfly-reduce, lane-0 store.
    emit_inner_reduction(ptx, tag, block_size, &r_tile);
    emit_lane_zero_store(ptx, tag, &r_tile, addr_min, addr_max);

    // Uniform increment + uniform-comparison branch (compiles to BRA.U).
    ptx.push_str(&format!("    add.u32 {r_tile}, {r_tile}, 1;\n"));
    ptx.push_str(&format!(
        "    setp.lt.u32 %p_done_TILERANGE, {r_tile}, {num_tiles};\n"
    ));
    ptx.push_str(&format!("    @%p_done_TILERANGE bra {loop_label};\n\n"));
}

/// Per-iteration body: load segment IDs into lane-local `%rs_min`/`%rs_max`,
/// then 5-step warp-shuffle butterfly reduction.
fn emit_inner_reduction(
    ptx: &mut String,
    tag: &str,
    block_size: u32,
    r_tile: &str,
) {
    // tile_start = %r_tile * block_size
    ptx.push_str(&format!(
        "    mul.lo.u32 %seg_byte_off_TILERANGE, {r_tile}, {block_size};\n"
    ));
    // Add lane to the tile start.
    ptx.push_str(
        "    add.u32 %seg_byte_off_TILERANGE, %seg_byte_off_TILERANGE, %lane_id_TILERANGE;\n",
    );
    // Convert token-index → byte offset (x sizeof(u16) = x 2).
    ptx.push_str(
        "    shl.b32 %seg_byte_off_TILERANGE, %seg_byte_off_TILERANGE, 1;\n",
    );
    // Compute byte address into segment_ids SMEM.
    ptx.push_str(
        "    cvt.u64.u32 %wide_tile_off_TILERANGE, %seg_byte_off_TILERANGE;\n",
    );
    ptx.push_str(
        "    add.u64 %wide_tile_off_TILERANGE, %seg_smem_TILERANGE, %wide_tile_off_TILERANGE;\n",
    );
    // Lane-local initial: load one u16 from SMEM.
    ptx.push_str(
        "    ld.shared.u16 %rs_min_TILERANGE, [%wide_tile_off_TILERANGE];\n",
    );
    ptx.push_str(
        "    mov.u16 %rs_max_TILERANGE, %rs_min_TILERANGE;\n",
    );

    // If block_size > warp_size, fold the second half (lane + 32, lane + 64, ...).
    let warp_size: u32 = 32;
    let mut extra_offset = warp_size;
    while extra_offset < block_size {
        let _ = tag; // silenced; tag may be referenced for debug comments in future
        ptx.push_str(&format!(
            "    // fold lane + {extra_offset} into lane-local min/max\n"
        ));
        ptx.push_str(&format!(
            "    add.u64 %wide_tile_off_TILERANGE, %wide_tile_off_TILERANGE, {};\n",
            extra_offset * 2 /* byte stride */
        ));
        ptx.push_str(
            "    ld.shared.u16 %rs_peer_min_TILERANGE, [%wide_tile_off_TILERANGE];\n",
        );
        ptx.push_str(
            "    min.u16 %rs_min_TILERANGE, %rs_min_TILERANGE, %rs_peer_min_TILERANGE;\n",
        );
        ptx.push_str(
            "    max.u16 %rs_max_TILERANGE, %rs_max_TILERANGE, %rs_peer_min_TILERANGE;\n",
        );
        extra_offset += warp_size;
    }

    // 5-step butterfly reduction (offsets 16, 8, 4, 2, 1).
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %rs_peer_min_TILERANGE, %rs_min_TILERANGE, {offset}, 0x1F, 0xFFFFFFFF;\n"
        ));
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %rs_peer_max_TILERANGE, %rs_max_TILERANGE, {offset}, 0x1F, 0xFFFFFFFF;\n"
        ));
        ptx.push_str(
            "    min.u16 %rs_min_TILERANGE, %rs_min_TILERANGE, %rs_peer_min_TILERANGE;\n",
        );
        ptx.push_str(
            "    max.u16 %rs_max_TILERANGE, %rs_max_TILERANGE, %rs_peer_max_TILERANGE;\n",
        );
    }
}

/// Lane-0 predicated store of (min, max) to `range_table[tile_idx]`.
fn emit_lane_zero_store(
    ptx: &mut String,
    tag: &str,
    r_tile: &str,
    addr_min: u32,
    addr_max: u32,
) {
    let _ = tag;
    // Compute byte offset = 2 * %r_tile (u16 stride).
    ptx.push_str(&format!(
        "    shl.b32 %seg_byte_off_TILERANGE, {r_tile}, 1;\n"
    ));
    ptx.push_str(
        "    cvt.u64.u32 %wide_tile_off_TILERANGE, %seg_byte_off_TILERANGE;\n",
    );
    // qtile_min/kvtile_min address.
    ptx.push_str(
        "    cvta.shared.u64 %addr_min_TILERANGE, shmem;\n"
    );
    ptx.push_str(&format!(
        "    add.u64 %addr_min_TILERANGE, %addr_min_TILERANGE, {addr_min};\n"
    ));
    ptx.push_str(
        "    add.u64 %addr_min_TILERANGE, %addr_min_TILERANGE, %wide_tile_off_TILERANGE;\n",
    );
    // qtile_max/kvtile_max address.
    ptx.push_str(
        "    cvta.shared.u64 %addr_max_TILERANGE, shmem;\n"
    );
    ptx.push_str(&format!(
        "    add.u64 %addr_max_TILERANGE, %addr_max_TILERANGE, {addr_max};\n"
    ));
    ptx.push_str(
        "    add.u64 %addr_max_TILERANGE, %addr_max_TILERANGE, %wide_tile_off_TILERANGE;\n",
    );
    // Lane-0 predicate.
    ptx.push_str(
        "    setp.eq.u32 %p_lane_zero_TILERANGE, %lane_id_TILERANGE, 0;\n",
    );
    // Predicated stores (NOT divergent branch -- see spec sec 4.3).
    ptx.push_str(
        "    @%p_lane_zero_TILERANGE st.shared.u16 [%addr_min_TILERANGE], %rs_min_TILERANGE;\n",
    );
    ptx.push_str(
        "    @%p_lane_zero_TILERANGE st.shared.u16 [%addr_max_TILERANGE], %rs_max_TILERANGE;\n",
    );
}

/// Emit PTX that branches to `on_skip_label` if the `(qt, kvt)` tile
/// pair has strictly disjoint segment ranges.
///
/// Loads `qtile_seg_min[qt]`, `qtile_seg_max[qt]`, `kvtile_seg_min[kvt]`,
/// `kvtile_seg_max[kvt]` from the range table in SMEM, then checks:
///   disjoint = (qmax < kvmin) || (qmin > kvmax)
///
/// When disjoint, branches to `on_skip_label` (warp-uniform → BRA.U).
/// All scratch registers carry the `_TB` suffix to avoid collisions.
///
/// `qt_reg` and `kvt_reg` are u32 PTX operands (register names or
/// immediates) holding the current q-tile / kv-tile ordinal.
/// `range_table_base` is the SMEM byte offset from
/// `smem_layout::tier_b_range_table_offset`.
pub fn emit_skip_predicate(
    ptx: &mut String,
    config: &crate::flash_attention::FlashAttentionConfig,
    seq_len: u32,
    qt_reg: &str,            // PTX register holding the current q-tile index
    kvt_reg: &str,           // PTX register holding the current kv-tile index
    range_table_base: u32,   // from smem_layout::tier_b_range_table_offset
    on_skip_label: &str,     // PTX label to branch to when ranges disjoint
) {
    let num_q  = crate::pca_tile_config::num_tiles(seq_len, config.block_q  as u32);
    let num_kv = crate::pca_tile_config::num_tiles(seq_len, config.block_kv as u32);
    let addrs = range_table_addrs(range_table_base, num_q, num_kv);

    ptx.push_str("    // ----- PCA Tier B: skip predicate (spec sec 3.2) -----\n");
    ptx.push_str("    .reg .u16  %qmin_TB, %qmax_TB, %kvmin_TB, %kvmax_TB;\n");
    ptx.push_str("    .reg .pred %p_lt_TB, %p_gt_TB, %p_skip_TB;\n");
    ptx.push_str("    .reg .u32  %tile_byte_TB;\n");
    ptx.push_str("    .reg .u64  %addr_TB;\n");
    ptx.push_str("\n");

    // qmin[qt], qmax[qt]
    ptx.push_str(&format!("    shl.b32 %tile_byte_TB, {qt_reg}, 1;\n"));
    ptx.push_str("    cvt.u64.u32 %addr_TB, %tile_byte_TB;\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem;\n    add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.qtile_min
    ));
    ptx.push_str("    ld.shared.u16 %qmin_TB, [%addr_TB];\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem;\n    add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.qtile_max
    ));
    ptx.push_str("    ld.shared.u16 %qmax_TB, [%addr_TB];\n");

    // kvmin[kvt], kvmax[kvt]
    ptx.push_str(&format!("    shl.b32 %tile_byte_TB, {kvt_reg}, 1;\n"));
    ptx.push_str("    cvt.u64.u32 %addr_TB, %tile_byte_TB;\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem;\n    add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.kvtile_min
    ));
    ptx.push_str("    ld.shared.u16 %kvmin_TB, [%addr_TB];\n");
    ptx.push_str(&format!(
        "    cvta.shared.u64 %addr_TB, shmem;\n    add.u64 %addr_TB, %addr_TB, {};\n",
        addrs.kvtile_max
    ));
    ptx.push_str("    ld.shared.u16 %kvmax_TB, [%addr_TB];\n");

    // disjoint = (qmax < kvmin) || (qmin > kvmax)
    ptx.push_str("    setp.lt.u16 %p_lt_TB, %qmax_TB, %kvmin_TB;\n");
    ptx.push_str("    setp.gt.u16 %p_gt_TB, %qmin_TB, %kvmax_TB;\n");
    ptx.push_str("    or.pred %p_skip_TB, %p_lt_TB, %p_gt_TB;\n");
    ptx.push_str(&format!("    @%p_skip_TB bra {on_skip_label};\n"));
    ptx.push_str("    // ----- end PCA Tier B skip predicate -----\n");
}

/// Emit PTX that writes the skip decision for `(b, h, qt, kvt)` to
/// the debug instrumentation buffer at the matching slot.
///
/// Lane 0 of warp 0 within the CTA writes; other threads no-op.
/// Decision: `1` = disjoint (skipped), `0` = kept.
///
/// Buffer shape: `[batch, head, num_q_tiles, num_kv_tiles] : u8`
/// per spec §4.3.1. Slot index = `(bh_idx * num_q_tiles + qt) * num_kv + kvt`.
///
/// Gated behind `cfg!(debug_kernel_instrumentation)` at codegen time
/// — production builds never call this function.
///
/// # Arguments
///
/// - `seq_len` — the sequence length; used to derive tile counts at
///   codegen time (not a runtime register).
/// - `qt_reg` / `kvt_reg` — PTX operands (register names or immediates)
///   holding the current q-tile / kv-tile ordinal.
/// - `is_skip_pred` — the predicate register produced by
///   [`emit_skip_predicate`] (e.g. `%p_skip_TB`).
/// - `decisions_buf_param` — the `.param .u64` parameter name holding
///   the HBM buffer base pointer.
///
/// TODO(tier-b.2): replace warp-0 approximation with a real
/// `owning_warp(qt, kvt)` helper when multi-warp ownership is needed.
pub fn emit_skip_decision_writeback(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    seq_len: u32,
    qt_reg: &str,
    kvt_reg: &str,
    is_skip_pred: &str,           // %p_skip_TB from emit_skip_predicate
    decisions_buf_param: &str,    // .param .u64 holding the HBM buffer ptr
) {
    let num_kv = crate::pca_tile_config::num_tiles(seq_len, config.block_kv as u32);
    let num_q_tiles_const = crate::pca_tile_config::num_tiles(seq_len, config.block_q as u32);

    ptx.push_str("    // ----- PCA Tier B: skip-decision writeback (debug, spec sec 4.3) -----\n");
    ptx.push_str("    .reg .u64 %dec_buf_TB, %dec_slot_TB;\n");
    ptx.push_str("    .reg .u32 %slot_off_TB, %bh_TB, %bh_slot_TB;\n");
    ptx.push_str("    .reg .u16 %dec_val_TB;\n");
    ptx.push_str("    .reg .pred %p_owner_TB, %p_lane_TB;\n");
    ptx.push_str("    .reg .u32 %warp_id_TB, %lane_TB;\n");
    ptx.push_str("\n");

    // (batch * num_heads + head) precomputed elsewhere as %bh_idx; reuse if available.
    ptx.push_str("    mov.u32 %bh_TB, %bh_idx;\n");
    ptx.push_str(&format!(
        "    mad.lo.u32 %bh_slot_TB, %bh_TB, {num_q_tiles_const}, {qt_reg};\n"
    ));
    ptx.push_str(&format!(
        "    mad.lo.u32 %slot_off_TB, %bh_slot_TB, {num_kv}, {kvt_reg};\n"
    ));

    // Load buffer base + add offset.
    ptx.push_str(&format!(
        "    ld.param.u64 %dec_buf_TB, [{decisions_buf_param}];\n"
    ));
    ptx.push_str("    cvt.u64.u32 %dec_slot_TB, %slot_off_TB;\n");
    ptx.push_str("    add.u64 %dec_slot_TB, %dec_buf_TB, %dec_slot_TB;\n");

    // Owner = lane 0 of owning warp. v1 approximates as warp 0 lane 0 within the CTA.
    // TODO(tier-b.2): replace with the real owning_warp(qt, kvt) helper.
    ptx.push_str("    mov.u32 %warp_id_TB, %warpid;\n");
    ptx.push_str("    setp.eq.u32 %p_owner_TB, %warp_id_TB, 0;\n");
    ptx.push_str("    mov.u32 %lane_TB, %laneid;\n");
    ptx.push_str("    setp.eq.u32 %p_lane_TB, %lane_TB, 0;\n");
    ptx.push_str("    and.pred %p_owner_TB, %p_owner_TB, %p_lane_TB;\n");

    // Decision value: 1 if disjoint (skipped), 0 if kept.
    ptx.push_str(&format!(
        "    selp.u16 %dec_val_TB, 1, 0, {is_skip_pred};\n"
    ));
    ptx.push_str("    @%p_owner_TB st.global.u8 [%dec_slot_TB], %dec_val_TB;\n");
    ptx.push_str("    // ----- end skip-decision writeback -----\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fa_base_for_test() -> FlashAttentionConfig {
        // Mirrors pca_segment.rs::tests::fa_base — same shape so
        // skeleton tests share the precedent's config surface.
        use crate::flash_attention::RopeStyle;
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 64,
            causal: true,
            paged: false,
            rope_q: true,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 2,
            tree_mask: false,
            gpu_sm: 90,
            segment_masked: true,
            csha: None,
        }
    }

    #[test]
    fn range_table_bytes_4k_block_64() {
        // 4096 / 64 = 64 q-tiles, 64 kv-tiles.
        // 2 × (64 + 64) × 2 = 512 B.
        assert_eq!(compute_range_table_bytes(4096, 64, 64), 512);
    }

    #[test]
    fn range_table_bytes_16k_block_64() {
        // 16384 / 64 = 256 tiles each.
        // 2 × (256 + 256) × 2 = 2048 B.
        assert_eq!(compute_range_table_bytes(16384, 64, 64), 2048);
    }

    #[test]
    fn range_table_bytes_asymmetric_block() {
        // 16K with block_q=32, block_kv=64, head_dim=256.
        // num_q = 512, num_kv = 256. 2 × (512+256) × 2 = 3072 B.
        assert_eq!(compute_range_table_bytes(16384, 32, 64), 3072);
    }

    #[test]
    fn should_emit_tier_b_inside_envelope() {
        let cfg = fa_base_for_test();
        assert!(should_emit_tier_b(&cfg, 4096, SegmentResidency::Shared));
    }

    #[test]
    fn should_skip_tier_b_when_streamed() {
        let cfg = fa_base_for_test();
        assert!(!should_emit_tier_b(&cfg, 4096, SegmentResidency::Streamed));
    }

    #[test]
    fn should_skip_tier_b_when_segment_masked_off() {
        let mut cfg = fa_base_for_test();
        cfg.segment_masked = false;
        assert!(!should_emit_tier_b(&cfg, 4096, SegmentResidency::Shared));
    }

    #[test]
    fn should_skip_tier_b_when_table_exceeds_budget() {
        // Hypothetical 1 M sequence → 16384 q tiles each side → 128 KB.
        let cfg = fa_base_for_test();
        assert!(!should_emit_tier_b(&cfg, 1_048_576, SegmentResidency::Shared));
    }

    /// Boundary-pair test that anchors directly on `TIER_B_RANGE_TABLE_BUDGET_BYTES`.
    /// If the const drifts, at least one assertion fails immediately —
    /// neither `should_emit_tier_b_inside_envelope` (4 K seq → 512 B, ¹⁄₁₆-budget)
    /// nor `should_skip_tier_b_when_table_exceeds_budget` (1 M seq → 128 KB, 16×-budget)
    /// catches drift to 4 KB or 16 KB.
    #[test]
    fn at_range_table_budget_boundary_decides_correctly() {
        // compute_range_table_bytes(16384, 16, 16) = 2 * (1024 + 1024) * 2 = 8192.
        assert_eq!(compute_range_table_bytes(16_384, 16, 16), 8192);
        // compute_range_table_bytes(16384, 8, 8)   = 2 * (2048 + 2048) * 2 = 16384.
        assert_eq!(compute_range_table_bytes(16_384, 8, 8), 16_384);

        // Exactly at budget → fits (`<=` semantics in should_emit_tier_b).
        let mut cfg_at = fa_base_for_test();
        cfg_at.block_q = 16;
        cfg_at.block_kv = 16;
        assert!(should_emit_tier_b(&cfg_at, 16_384, SegmentResidency::Shared));

        // Just past budget → falls back.
        let mut cfg_over = fa_base_for_test();
        cfg_over.block_q = 8;
        cfg_over.block_kv = 8;
        assert!(!should_emit_tier_b(&cfg_over, 16_384, SegmentResidency::Shared));
    }

    #[test]
    fn skeleton_emitters_emit_marker_comments() {
        let mut s = String::new();
        let cfg = fa_base_for_test();
        emit_range_table_preamble(&mut s, &cfg, 4096, "seg_smem", 0);
        emit_skip_predicate(&mut s, &cfg, 4096, "%qt", "%kvt", 0, "skip_kv_iter");
        emit_skip_decision_writeback(&mut s, &cfg, 4096, "%qt", "%kvt", "%p_skip_TB", "%dec_buf");
        assert!(s.contains("PCA Tier B"), "output missing marker:\n{}", s);
    }
}
