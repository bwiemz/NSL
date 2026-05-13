//! Tier B.1 native output scatter — replaces the Tier A `finalize::emit`
//! call. Reads %o_acc_<t>_<lane> + %s_sum_<t>_<half> + %s_max_<t>_<half>
//! and writes the final O[batch, head, q_row, d_col] = o_acc / row_sum to
//! global memory (f16), plus LSE = row_max + ln(row_sum) for the backward
//! pass. Per spec section 4.2 finalize step.
//!
//! ## Why this isn't `phases::forward::finalize::emit`
//!
//! Tier A's finalize is one-q-row-per-warp: each warp's 32 lanes each
//! write head_dim/32 contiguous elements of one row, divided by a single
//! `%row_sum` scalar. Tier B.1 instead has m16n8k16 D-fragment-shaped
//! per-tile accumulators: each warp owns up to `tpw_pv` tiles where each
//! tile is 16 q-rows × 8 head_dim cols, and the four accumulator lanes
//! per thread map to four distinct (row, col) positions per the standard
//! D-fragment layout. The two layouts don't share a register namespace —
//! a Tier-B1-native scatter is the cleaner solution than a rename.
//!
//! ## D-fragment layout (per spec section 5.5)
//!
//! For lane l ∈ [0, 32) within a warp:
//!   * `%o_acc_<t>_0` = O[m_tile*16 + l/4,     n_tile*8 + (l%4)*2    ]
//!   * `%o_acc_<t>_1` = O[m_tile*16 + l/4,     n_tile*8 + (l%4)*2 + 1]
//!   * `%o_acc_<t>_2` = O[m_tile*16 + l/4 + 8, n_tile*8 + (l%4)*2    ]
//!   * `%o_acc_<t>_3` = O[m_tile*16 + l/4 + 8, n_tile*8 + (l%4)*2 + 1]
//!
//! Where `m_tile = t / (head_dim/8)` and `n_tile = t % (head_dim/8)`.
//!
//! Lanes 0,1,2,3 within each thread index correspond to the LO half
//! (rows 0..7 of the 16-row tile) and HI half (rows 8..15) respectively,
//! so lanes {0,1} divide by `%s_sum_<t>_lo` and {2,3} by `%s_sum_<t>_hi`.
//!
//! ## Warp ownership
//!
//! Matches the PV MMA gate in `attention_mma::emit_pv_mma`: tile t fires
//! only when `%warp_id == (t % 8)`. The address compute runs across all
//! warps (negligible perf loss) but the `st.global.b16` is predicated, so
//! only the owning warp writes per tile.
//!
//! ## LSE store
//!
//! For each tile, lanes with `(lane % 4) == 0` (lanes 0,4,8,...,28) each
//! write 2 LSE values: one for the LO row they own (row = m_tile*16 + l/4)
//! and one for the HI row (row = m_tile*16 + l/4 + 8). Both are gated on
//! `%p_has_lse` (null-guarded `%logsumexp_base`). LSE = row_max + ln(row_sum)
//! computed from `%s_max_<t>_<half>` and `%s_sum_<t>_<half>`.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::tier_b1::attention_mma::tiles_per_warp_pv;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig) {
    let tpw = tiles_per_warp_pv(config);
    let head_dim = config.head_dim as u32;
    let n_tiles_d = (head_dim / 8).max(1);

    ptx.push_str("    // === Tier B.1 native output scatter (B1.6 deferral #9) ===\n");

    // Per-phase scratch. Named registers avoid collision with the %rd<64>
    // numbered pool used elsewhere in the kernel.
    ptx.push_str("    .reg .f32 %fin_norm;\n");
    ptx.push_str("    .reg .u32 %fin_lo_row, %fin_lo_col_base;\n");
    ptx.push_str("    .reg .u32 %fin_row_in_tile, %fin_col_in_tile;\n");
    ptx.push_str("    .reg .u32 %fin_q_row_u32, %fin_d_col_u32;\n");
    ptx.push_str("    .reg .u32 %fin_laneid, %fin_lane_mod4;\n");
    ptx.push_str("    .reg .u64 %fin_q_row, %fin_d_col, %fin_idx, %fin_idx2;\n");
    ptx.push_str("    .reg .u64 %fin_addr;\n");
    ptx.push_str("    .reg .b16 %fin_h;\n");
    ptx.push_str("    .reg .pred %fin_warp_pred, %fin_lane0_pred;\n");

    // Per-lane D-fragment row/col base. These are constant across all
    // tiles for a given lane, so compute once.
    ptx.push_str("    mov.u32 %fin_laneid, %tid.x;\n");
    ptx.push_str("    and.b32 %fin_laneid, %fin_laneid, 31;\n");
    ptx.push_str("    shr.u32 %fin_lo_row, %fin_laneid, 2;       // l/4 = D-fragment row within lo half\n");
    ptx.push_str("    and.b32 %fin_lo_col_base, %fin_laneid, 3;  // l%4\n");
    ptx.push_str("    shl.b32 %fin_lo_col_base, %fin_lo_col_base, 1; // (l%4)*2 = D-fragment col base\n");
    ptx.push_str("    and.b32 %fin_lane_mod4, %fin_laneid, 3;    // for LSE lane-0-of-quad gate\n");

    for t in 0..tpw {
        let m_tile = t / n_tiles_d;
        let n_tile = t % n_tiles_d;

        ptx.push_str(&format!(
            "    // ---- Output tile t={} (m_tile={}, n_tile={}, warp owner = {}) ----\n",
            t,
            m_tile,
            n_tile,
            t % 8
        ));
        ptx.push_str(&format!(
            "    setp.eq.u32 %fin_warp_pred, %warp_id, {};\n",
            t % 8
        ));

        for i in 0..4u32 {
            let half = if i < 2 { "lo" } else { "hi" };
            let row_offset = if i < 2 { 0u32 } else { 8 };
            let col_offset = i % 2;

            // Normalize: o_acc_t_i / s_sum_t_<half>.
            ptx.push_str(&format!(
                "    div.approx.f32 %fin_norm, %o_acc_{}_{}, %s_sum_{}_{};\n",
                t, i, t, half
            ));

            // Compute (row_in_tile, col_in_tile).
            ptx.push_str(&format!(
                "    add.u32 %fin_row_in_tile, %fin_lo_row, {};\n",
                row_offset
            ));
            ptx.push_str(&format!(
                "    add.u32 %fin_col_in_tile, %fin_lo_col_base, {};\n",
                col_offset
            ));

            // q_row_global_u32 = m_tile*16 + row_in_tile (added to %q_start
            // which is u64 below). d_col_global_u32 = n_tile*8 + col_in_tile.
            ptx.push_str(&format!(
                "    add.u32 %fin_q_row_u32, %fin_row_in_tile, {};\n",
                m_tile * 16
            ));
            ptx.push_str(&format!(
                "    add.u32 %fin_d_col_u32, %fin_col_in_tile, {};\n",
                n_tile * 8
            ));
            ptx.push_str("    cvt.u64.u32 %fin_q_row, %fin_q_row_u32;\n");
            ptx.push_str("    cvt.u64.u32 %fin_d_col, %fin_d_col_u32;\n");
            ptx.push_str("    add.u64 %fin_q_row, %fin_q_row, %q_start;\n");

            // addr = out_ptr + ((batch*heads + head)*seq + q_row)*hd + d_col) * 2
            ptx.push_str("    mul.lo.u64 %fin_idx, %batch_idx, %rd5;\n");
            ptx.push_str("    add.u64 %fin_idx, %fin_idx, %head_idx;\n");
            ptx.push_str("    mul.lo.u64 %fin_idx, %fin_idx, %rd6;\n");
            ptx.push_str("    add.u64 %fin_idx, %fin_idx, %fin_q_row;\n");
            ptx.push_str("    mul.lo.u64 %fin_idx, %fin_idx, %rd7;\n");
            ptx.push_str("    add.u64 %fin_idx, %fin_idx, %fin_d_col;\n");
            ptx.push_str("    shl.b64 %fin_idx, %fin_idx, 1;  // f16 = 2 bytes\n");
            ptx.push_str("    add.u64 %fin_addr, %rd3, %fin_idx;\n");

            // f32 -> f16 + predicated store.
            ptx.push_str("    cvt.rn.f16.f32 %fin_h, %fin_norm;\n");
            ptx.push_str("    @%fin_warp_pred st.global.b16 [%fin_addr], %fin_h;\n");
        }

        // LSE store: lanes with l%4 == 0 each write 2 LSE entries (lo + hi
        // row of the tile). Gated additionally on %p_has_lse and the
        // warp-owner predicate.
        ptx.push_str("    setp.eq.u32 %fin_lane0_pred, %fin_lane_mod4, 0;\n");
        ptx.push_str("    and.pred %fin_lane0_pred, %fin_lane0_pred, %fin_warp_pred;\n");
        ptx.push_str("    and.pred %fin_lane0_pred, %fin_lane0_pred, %p_has_lse;\n");

        for half_idx in 0..2u32 {
            let half = if half_idx == 0 { "lo" } else { "hi" };
            let row_offset = half_idx * 8;

            // lse = row_max + log2(row_sum) * ln(2).
            ptx.push_str(&format!(
                "    lg2.approx.f32 %log_sum, %s_sum_{}_{};\n",
                t, half
            ));
            ptx.push_str("    mul.f32 %log_sum, %log_sum, 0f3F317218; // * ln(2)\n");
            ptx.push_str(&format!(
                "    add.f32 %lse, %s_max_{}_{}, %log_sum;\n",
                t, half
            ));

            // q_row_global_u32 = m_tile*16 + row_offset + lo_row
            ptx.push_str(&format!(
                "    add.u32 %fin_q_row_u32, %fin_lo_row, {};\n",
                m_tile * 16 + row_offset
            ));
            ptx.push_str("    cvt.u64.u32 %fin_q_row, %fin_q_row_u32;\n");
            ptx.push_str("    add.u64 %fin_q_row, %fin_q_row, %q_start;\n");

            // addr = lse_base + ((batch*heads + head)*seq + q_row) * 4
            ptx.push_str("    mul.lo.u64 %fin_idx2, %batch_idx, %rd5;\n");
            ptx.push_str("    add.u64 %fin_idx2, %fin_idx2, %head_idx;\n");
            ptx.push_str("    mul.lo.u64 %fin_idx2, %fin_idx2, %rd6;\n");
            ptx.push_str("    add.u64 %fin_idx2, %fin_idx2, %fin_q_row;\n");
            ptx.push_str("    shl.b64 %fin_idx2, %fin_idx2, 2;  // f32 = 4 bytes\n");
            ptx.push_str("    add.u64 %fin_addr, %logsumexp_base, %fin_idx2;\n");

            ptx.push_str("    @%fin_lane0_pred st.global.f32 [%fin_addr], %lse;\n");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn make_config(bq: i64, bkv: i64, hd: i64) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: bq,
            block_kv: bkv,
            head_dim: hd,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: 2048,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn emits_one_div_per_accumulator_lane_per_tile() {
        let cfg = make_config(32, 32, 32);
        let tpw = tiles_per_warp_pv(&cfg);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // 4 accumulator lanes per tile × tpw tiles.
        let expected = (tpw * 4) as usize;
        assert_eq!(
            ptx.matches("div.approx.f32 %fin_norm,").count(),
            expected,
            "expected {} divides for {} tiles × 4 lanes",
            expected,
            tpw
        );
    }

    #[test]
    fn emits_two_lse_stores_per_tile() {
        let cfg = make_config(32, 32, 32);
        let tpw = tiles_per_warp_pv(&cfg);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert_eq!(
            ptx.matches("@%fin_lane0_pred st.global.f32").count(),
            (tpw * 2) as usize,
            "expected 2 LSE stores per tile (lo + hi half)"
        );
    }

    #[test]
    fn emits_warp_predicated_b16_stores() {
        let cfg = make_config(32, 32, 32);
        let tpw = tiles_per_warp_pv(&cfg);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert_eq!(
            ptx.matches("@%fin_warp_pred st.global.b16").count(),
            (tpw * 4) as usize,
            "expected 4 predicated f16 stores per tile (D-fragment lanes 0..3)"
        );
    }

    #[test]
    fn no_reference_to_tier_a_register_namespace() {
        let cfg = make_config(64, 64, 64);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // Tier A finalize uses %f48 (O_BASE). Tier B.1 native MUST NOT.
        assert!(
            !ptx.contains("%f48"),
            "Tier-B1-native finalize must not reference Tier A's %f<O_BASE> registers"
        );
        // And must not touch the named %row_sum / %row_max scalars (those
        // are Tier A's one-row-per-warp namespace).
        assert!(
            !ptx.contains("%row_sum"),
            "Tier-B1-native finalize uses %s_sum_<t>_<half>, not Tier A's %row_sum"
        );
    }
}
