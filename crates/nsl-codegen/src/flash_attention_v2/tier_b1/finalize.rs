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
//! ## Warp ownership (N3 resolution)
//!
//! Each warp owns its own `tpw_pool` accumulator slots. For slot
//! `local_t in 0..tpw_pool` the warp's global tile is
//! `global_t = warp_id + local_t * 8` (round-robin across 8 warps).
//! All 8 warps execute every iteration; the address compute uses
//! runtime `%warp_id`-derived `(m_tile, n_tile)` so each warp writes
//! to a DISTINCT region of HBM. No store-time predicate is needed.
//!
//! ## LSE store
//!
//! For each tile, lanes with `(lane % 4) == 0` (lanes 0,4,8,...,28) each
//! write 2 LSE values: one for the LO row they own (row = m_tile*16 + l/4)
//! and one for the HI row (row = m_tile*16 + l/4 + 8). Both are gated on
//! `%p_has_lse` (null-guarded `%logsumexp_base`). LSE = row_max + ln(row_sum)
//! computed from `%s_max_<t>_<half>` and `%s_sum_<t>_<half>`.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    tier_b1_reduced_stats_offset, tier_b1_reduced_stats_sum_offset,
};
use crate::flash_attention_v2::tier_b1::attention_mma::tiles_per_warp_pv;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig) {
    // Phase 2.6 (T2.5) save-activation retrofit gate. Only fires when the
    // caller requests backward-activation saves (gated on @train mode by
    // the compiler). Inference builds leave this false; finalize emits no
    // extra param loads, predicates, or scatters and pays zero HBM cost.
    let save_activations = config
        .csha
        .as_ref()
        .is_some_and(|c| c.save_activations_for_backward);

    let tpw = tiles_per_warp_pv(config);
    let head_dim = config.head_dim as u32;
    let n_tiles_d = (head_dim / 8).max(1);
    assert!(
        n_tiles_d.is_power_of_two(),
        "Tier B.1 finalize requires head_dim/8 to be a power of 2 \
         (for shr/and tile-coords compute); got n_tiles_d={}",
        n_tiles_d
    );
    let log2_n_tiles_d = n_tiles_d.trailing_zeros();
    let n_tiles_d_mask = n_tiles_d - 1;

    ptx.push_str("    // === Tier B.1 native output scatter (B1.6 deferral #9 + N3) ===\n");

    // Per-phase scratch. Named registers avoid collision with the %rd<64>
    // numbered pool used elsewhere in the kernel.
    ptx.push_str("    .reg .f32 %fin_norm;\n");
    ptx.push_str("    .reg .u32 %fin_lo_row, %fin_lo_col_base;\n");
    ptx.push_str("    .reg .u32 %fin_row_in_tile, %fin_col_in_tile;\n");
    ptx.push_str("    .reg .u32 %fin_q_row_u32, %fin_d_col_u32;\n");
    ptx.push_str("    .reg .u32 %fin_laneid, %fin_lane_mod4;\n");
    // N3 runtime tile coords: each warp computes its OWN (m_tile, n_tile)
    // from %warp_id, so every warp does distinct work each iter.
    ptx.push_str("    .reg .u32 %fin_global_t, %fin_m_tile, %fin_n_tile;\n");
    ptx.push_str("    .reg .u32 %fin_m_tile_x16, %fin_n_tile_x8;\n");
    ptx.push_str("    .reg .u64 %fin_q_row, %fin_d_col, %fin_idx, %fin_idx2;\n");
    ptx.push_str("    .reg .u64 %fin_addr;\n");
    ptx.push_str("    .reg .b16 %fin_h;\n");
    ptx.push_str("    .reg .pred %fin_lane0_pred;\n");

    // Reduced softmax-stats SMEM reads (hd > block_kv fix). The row-max/row-sum
    // are sourced from a dedicated SMEM region keyed by ABSOLUTE global query
    // row, NOT from the QK^T-slot-keyed %s_max_<t>/%s_sum_<t> registers. This
    // makes finalize independent of the QK^T-vs-PV slot->m_tile divergence that
    // breaks at head_dim > block_kv (see attention_mma.rs STEP 4.5 + smem_layout
    // ::tier_b1_reduced_stats_offset). %rstat_row is the LOCAL global_row
    // (pre-q_start); %rstat_addr is the byte address; %rstat_max/%rstat_sum hold
    // the loaded values for the LSE / save sites.
    ptx.push_str("    .reg .u32 %rstat_max_base, %rstat_sum_base, %rstat_row, %rstat_addr;\n");
    ptx.push_str("    .reg .f32 %rstat_max, %rstat_sum;\n");
    ptx.push_str("    .reg .u64 %rstat_base_u64;\n");
    // Compute the reduced-stats SMEM base addresses once (u32 for st/ld.shared).
    let reduced_max_base = tier_b1_reduced_stats_offset(config);
    let reduced_sum_base = tier_b1_reduced_stats_sum_offset(config);
    ptx.push_str(&format!(
        "    add.u64 %rstat_base_u64, %shmem_base, {}; // reduced row-max base\n",
        reduced_max_base
    ));
    ptx.push_str("    cvt.u32.u64 %rstat_max_base, %rstat_base_u64;\n");
    ptx.push_str(&format!(
        "    add.u64 %rstat_base_u64, %shmem_base, {}; // reduced row-sum base\n",
        reduced_sum_base
    ));
    ptx.push_str("    cvt.u32.u64 %rstat_sum_base, %rstat_base_u64;\n");

    // Phase 2.6 (T2.5): row_max/row_sum HBM save scratch. These reuse the
    // %fin_idx2-derived row-major [B,H,S] index that the LSE store builds;
    // only the base pointer and the stored value differ (raw global max /
    // sum vs LSE). Declared once; consumed inside the per-half LSE loop.
    if save_activations {
        ptx.push_str("    .reg .u64 %psv_row_max_base, %psv_row_sum_base;\n");
        ptx.push_str("    .reg .u64 %psv_row_max_addr, %psv_row_sum_addr;\n");
        ptx.push_str("    .reg .pred %psv_has_row_max, %psv_has_row_sum;\n");
        ptx.push_str("    .reg .pred %psv_row_max_gate, %psv_row_sum_gate;\n");
        // Load the two save base pointers once (mirrors prelude's
        // %logsumexp_base load for LSE) and build their null-guard
        // predicates up front, ANDed into the lane-0-of-quad gate below.
        ptx.push_str("    ld.param.u64 %psv_row_max_base, [row_max_ptr];\n");
        ptx.push_str("    ld.param.u64 %psv_row_sum_base, [row_sum_ptr];\n");
        ptx.push_str("    setp.ne.u64 %psv_has_row_max, %psv_row_max_base, 0;\n");
        ptx.push_str("    setp.ne.u64 %psv_has_row_sum, %psv_row_sum_base, 0;\n");
    }

    // Initialize the LSE null-guard predicate. %p_has_lse is DECLARED by the
    // shared prelude (prelude.rs:200) but only SET by Tier A's finalize
    // (phases/forward/finalize.rs:68). Tier B.1 has its own native finalize
    // and never invokes Tier A's, so without this setp the predicate is left
    // UNDEFINED -- on the GPU it reads false, gating off every LSE / row_max /
    // row_sum store (all three buffers came back all-zero in T2.7). Mirror
    // Tier A's guard so the LSE store fires whenever logsumexp_ptr is non-null.
    ptx.push_str(
        "    setp.ne.u64 %p_has_lse, %logsumexp_base, 0;  // B.1: set LSE null-guard (Tier A sets it; B.1 native finalize must too)\n",
    );

    // Per-lane D-fragment row/col base. These are constant across all
    // tiles for a given lane, so compute once.
    ptx.push_str("    mov.u32 %fin_laneid, %tid.x;\n");
    ptx.push_str("    and.b32 %fin_laneid, %fin_laneid, 31;\n");
    ptx.push_str("    shr.u32 %fin_lo_row, %fin_laneid, 2;       // l/4 = D-fragment row within lo half\n");
    ptx.push_str("    and.b32 %fin_lo_col_base, %fin_laneid, 3;  // l%4\n");
    ptx.push_str("    shl.b32 %fin_lo_col_base, %fin_lo_col_base, 1; // (l%4)*2 = D-fragment col base\n");
    ptx.push_str("    and.b32 %fin_lane_mod4, %fin_laneid, 3;    // for LSE lane-0-of-quad gate\n");

    for t in 0..tpw {
        ptx.push_str(&format!(
            "    // ---- Output slot local_t={} (global_t = warp_id + {}*8 at runtime; N3) ----\n",
            t,
            t
        ));

        // global_t = warp_id + local_t * 8 (or just warp_id when local_t == 0)
        if t == 0 {
            ptx.push_str("    mov.u32 %fin_global_t, %warp_id;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u32 %fin_global_t, %warp_id, {};\n",
                t * 8
            ));
        }
        // m_tile = global_t / n_tiles_d (shr since power of 2)
        // n_tile = global_t & (n_tiles_d - 1)
        ptx.push_str(&format!(
            "    shr.u32 %fin_m_tile, %fin_global_t, {};\n",
            log2_n_tiles_d
        ));
        ptx.push_str(&format!(
            "    and.b32 %fin_n_tile, %fin_global_t, {};\n",
            n_tiles_d_mask
        ));
        // m_tile*16 and n_tile*8 precomputed once per tile.
        ptx.push_str("    shl.b32 %fin_m_tile_x16, %fin_m_tile, 4;   // m_tile * 16\n");
        ptx.push_str("    shl.b32 %fin_n_tile_x8, %fin_n_tile, 3;    // n_tile * 8\n");

        for i in 0..4u32 {
            let half = if i < 2 { "lo" } else { "hi" };
            let row_offset = if i < 2 { 0u32 } else { 8 };
            let col_offset = i % 2;

            // Normalize: o_acc_t_i / row_sum[global_row]. The row_sum is read
            // from the reduced-stats SMEM by ABSOLUTE global query row (hd>bkv
            // fix), NOT from the QK^T-slot-keyed %s_sum_<t> register.
            // local global_row = m_tile*16 + lo_row + row_offset(i).
            let _ = half; // half no longer selects a register; row_offset does.
            ptx.push_str(&format!(
                "    add.u32 %rstat_row, %fin_lo_row, {};\n",
                row_offset
            ));
            ptx.push_str("    add.u32 %rstat_row, %rstat_row, %fin_m_tile_x16;\n");
            ptx.push_str("    shl.b32 %rstat_addr, %rstat_row, 2; // global_row * 4 (f32)\n");
            ptx.push_str("    add.u32 %rstat_addr, %rstat_addr, %rstat_sum_base;\n");
            ptx.push_str("    ld.shared.f32 %rstat_sum, [%rstat_addr];\n");
            // Each warp's slot t holds its OWN global tile's accumulator — no gate needed.
            ptx.push_str(&format!(
                "    div.approx.f32 %fin_norm, %o_acc_{}_{}, %rstat_sum;\n",
                t, i
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

            // q_row_global_u32 = m_tile*16 + row_in_tile (runtime now).
            // d_col_global_u32 = n_tile*8 + col_in_tile (runtime).
            ptx.push_str(
                "    add.u32 %fin_q_row_u32, %fin_row_in_tile, %fin_m_tile_x16;\n",
            );
            ptx.push_str(
                "    add.u32 %fin_d_col_u32, %fin_col_in_tile, %fin_n_tile_x8;\n",
            );
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

            // f32 -> f16 + unconditional store (N3: no warp gate; each
            // warp writes to a DISTINCT global tile address).
            ptx.push_str("    cvt.rn.f16.f32 %fin_h, %fin_norm;\n");
            ptx.push_str("    st.global.b16 [%fin_addr], %fin_h;\n");
        }

        // LSE store: lanes with l%4 == 0 each write 2 LSE entries (lo + hi
        // row of this warp's global tile). Gated only on %p_has_lse +
        // lane_mod4 == 0 (no warp gate; each warp writes its own LSE).
        ptx.push_str("    setp.eq.u32 %fin_lane0_pred, %fin_lane_mod4, 0;\n");
        ptx.push_str("    and.pred %fin_lane0_pred, %fin_lane0_pred, %p_has_lse;\n");

        for half_idx in 0..2u32 {
            let row_offset = half_idx * 8;

            // local global_row = m_tile*16 + lo_row + row_offset (pre-q_start).
            // This indexes the reduced-stats SMEM written by the producer's
            // STEP 4.5, keyed by absolute query row (hd>bkv fix). The SAME
            // value (before adding q_start) is the LSE/save HBM seq index below.
            ptx.push_str(&format!(
                "    add.u32 %rstat_row, %fin_lo_row, {};\n",
                row_offset
            ));
            ptx.push_str("    add.u32 %rstat_row, %rstat_row, %fin_m_tile_x16;\n");
            ptx.push_str("    shl.b32 %rstat_addr, %rstat_row, 2; // global_row * 4 (f32)\n");
            // row_max[global_row].
            ptx.push_str("    add.u32 %rstat_addr, %rstat_addr, %rstat_max_base;\n");
            ptx.push_str("    ld.shared.f32 %rstat_max, [%rstat_addr];\n");
            // row_sum[global_row].
            ptx.push_str("    shl.b32 %rstat_addr, %rstat_row, 2;\n");
            ptx.push_str("    add.u32 %rstat_addr, %rstat_addr, %rstat_sum_base;\n");
            ptx.push_str("    ld.shared.f32 %rstat_sum, [%rstat_addr];\n");

            // lse = row_max + log2(row_sum) * ln(2).
            ptx.push_str("    lg2.approx.f32 %log_sum, %rstat_sum;\n");
            ptx.push_str("    mul.f32 %log_sum, %log_sum, 0f3F317218; // * ln(2)\n");
            ptx.push_str("    add.f32 %lse, %rstat_max, %log_sum;\n");

            // q_row_global_u32 = local global_row (== m_tile*16 + row_offset
            // + lo_row); reuse %rstat_row computed above for the SMEM read.
            ptx.push_str("    mov.u32 %fin_q_row_u32, %rstat_row;\n");
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

            // Phase 2.6 (T2.5): sibling row_max/row_sum saves. Reuse the
            // %fin_idx2 byte offset the LSE store just built (row-major
            // [B,H,S] f32, *4 already applied at the shl above); swap only
            // the base pointer and store the RAW global softmax max/sum
            // (NOT the LSE = max + ln(sum) combination) for the dQ-kernel's
            // P = exp(S - row_max)/row_sum recompute (R4 linear-domain
            // convention). Same lane-0-of-quad row ownership as LSE, ANDed
            // with each pointer's null-guard so a caller that passes a null
            // row_max_ptr/row_sum_ptr skips that store without fault.
            if save_activations {
                ptx.push_str(
                    "    add.u64 %psv_row_max_addr, %psv_row_max_base, %fin_idx2;\n",
                );
                ptx.push_str(
                    "    add.u64 %psv_row_sum_addr, %psv_row_sum_base, %fin_idx2;\n",
                );
                ptx.push_str(
                    "    and.pred %psv_row_max_gate, %fin_lane0_pred, %psv_has_row_max;\n",
                );
                ptx.push_str(
                    "    and.pred %psv_row_sum_gate, %fin_lane0_pred, %psv_has_row_sum;\n",
                );
                // Store the SMEM-sourced (m-tile-keyed) reduced stats — NOT the
                // raw %s_max_<t>/%s_sum_<t> registers, which alias the wrong
                // m-tile at hd>bkv. %rstat_max / %rstat_sum were loaded for THIS
                // half's global_row above (hd>bkv fix; must move in lockstep).
                ptx.push_str(
                    "    @%psv_row_max_gate st.global.f32 [%psv_row_max_addr], %rstat_max;  // row_max_ptr write\n",
                );
                ptx.push_str(
                    "    @%psv_row_sum_gate st.global.f32 [%psv_row_sum_addr], %rstat_sum;  // row_sum_ptr write\n",
                );
            }
        }
    }

    // === Phase 2.6 (T2.5): q/k/v_proj save-activation scatter ===
    //
    // Purely additive SMEM->HBM scatter of the projected Q/K/V tiles for
    // the Tier B.2 backward dQ-kernel. Runs AFTER the per-tile O/LSE loop so
    // finalize's %fin_* scratch is free; the sweep uses its own %psv_*
    // namespace (R6). Each save is null-guarded on its FFI pointer.
    //
    // SMEM-survival (design V-Phase-2.6-B section 2.2): the Q tile lives in
    // a DEDICATED slab at offset 0 (no downstream phase writes into it);
    // K-ping/V-ping survive because the current Tier B.1 orchestrator emits
    // a SINGLE kv_iter (slot=0/ping), so the attention phase never clobbers
    // them. VALID ONLY WHILE B.1 EMITS A SINGLE kv_iter -- revisit when the
    // B1.6 multi-iter ping/pong loop lands (the K/V save must then move
    // inside the kv loop, scattering each block before the slab swap).
    if save_activations {
        use crate::flash_attention_v2::smem_layout::{
            tier_b1_k_offset_ping, tier_b1_q_offset, tier_b1_v_offset_ping,
        };
        ptx.push_str(
            "    // -- Phase 2.6 (T2.5) projection saves: Q/K/V SMEM -> HBM --\n",
        );
        ptx.push_str(
            "    bar.sync 0;  // FENCE: Q/K/V SMEM tiles fully visible to save sweep\n",
        );
        // Q: row-major SMEM, HBM seq index = q_start + r (R1: Q rows ARE
        // the q-tile's absolute rows).
        emit_proj_save(
            ptx,
            config,
            "Q",
            "q_proj_ptr",
            tier_b1_q_offset(config),
            /* row_count = */ config.block_q as u32,
            /* col_major = */ false,
            /* use_q_start = */ true,
        );
        // K: row-major SMEM, HBM seq index = r (R1: absolute K position,
        // NOT q_start + r -- otherwise multiple q-tile CTAs disagree).
        emit_proj_save(
            ptx,
            config,
            "K",
            "k_proj_ptr",
            tier_b1_k_offset_ping(config),
            /* row_count = */ config.block_kv as u32,
            /* col_major = */ false,
            /* use_q_start = */ false,
        );
        // V: COL-major SMEM (R2 transpose), HBM seq index = r (same as K).
        emit_proj_save(
            ptx,
            config,
            "V",
            "v_proj_ptr",
            tier_b1_v_offset_ping(config),
            /* row_count = */ config.block_kv as u32,
            /* col_major = */ true,
            /* use_q_start = */ false,
        );
    }
}

/// Emit a single projection-save sweep: scatter one SMEM tile (Q, K, or V)
/// to its HBM `[B,H,S,D]` row-major f16 buffer.
///
/// Cooperative warp-per-row / lane-per-col sweep modeled on Tier-A's
/// `csha_hooks::emit_save_activations_subset`: warp `w` (in 0..8) owns SMEM
/// rows `{w, w+8, w+16, ...} < row_count`; lane `l` owns cols
/// `{l, l+32, ...} < head_dim`. For each owned (row r, col c) the thread
/// reads one f16 from SMEM and stores it to HBM.
///
/// * `label` — "Q"/"K"/"V"; namespaces the SKIP label and register suffix.
/// * `ptr_param` — the kernel param to `ld.param.u64` for the HBM base.
/// * `smem_off` — byte offset of the tile within the extern `.shared` slab.
/// * `row_count` — `block_q` for Q, `block_kv` for K/V.
/// * `col_major` — false: row-major SMEM read `r*(hd*2)+c*2`; true (V only):
///   col-major SMEM read `c*(bkv*2)+r*2` (R2, the load_transposed bug
///   class -- the ONLY index that differs from row-major).
/// * `use_q_start` — true (Q): HBM seq index = `q_start + r`; false (K/V):
///   HBM seq index = `r` (absolute K position; R1, load-bearing).
///
/// Fresh `%psv_<label>_*` register namespace (R6) -- suffixed by label so
/// the three sequential calls never collide on a `.reg` redeclaration.
fn emit_proj_save(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    label: &str,
    ptr_param: &str,
    smem_off: u32,
    row_count: u32,
    col_major: bool,
    use_q_start: bool,
) {
    let hd = config.head_dim as u32;
    let bkv = config.block_kv as u32;

    // rows_per_warp: warp w covers rows w, w+8, ... < row_count.
    let rows_per_warp = row_count.div_ceil(8);
    // cols_per_lane: lane l covers cols l, l+32, ... < head_dim.
    let cols_per_lane = hd.div_ceil(32);

    ptx.push_str(&format!(
        "    // ---- Save {label} projection ({}-major SMEM, rows={}, hd={}, seq={}) ----\n",
        if col_major { "col" } else { "row" },
        row_count,
        hd,
        if use_q_start { "q_start+r" } else { "r (abs)" },
    ));

    // Fresh per-label register namespace (R6 -- avoids %fin_*, %qsc_/%ksc_/
    // %vsc_, and collisions across the 3 sequential calls).
    ptx.push_str(&format!(
        "    .reg .u64 %psv_{label}_ptr;\n"
    ));
    ptx.push_str(&format!(
        "    .reg .u32 %psv_{label}_r, %psv_{label}_c, %psv_{label}_wid, %psv_{label}_lane;\n"
    ));
    ptx.push_str(&format!(
        "    .reg .u64 %psv_{label}_r64, %psv_{label}_c64, %psv_{label}_seq;\n"
    ));
    ptx.push_str(&format!(
        "    .reg .u64 %psv_{label}_idx, %psv_{label}_smem_b, %psv_{label}_hbm;\n"
    ));
    ptx.push_str(&format!(
        "    .reg .u64 %psv_{label}_smem_addr;\n"
    ));
    ptx.push_str(&format!(
        "    .reg .b16 %psv_{label}_h;\n"
    ));
    ptx.push_str(&format!(
        "    .reg .pred %psv_{label}_null;\n"
    ));
    // Range predicates, only when row_count / head_dim are not exact
    // multiples of the warp / lane stride (declared once here -- never
    // inside the loops, which would redeclare them and trip ptxas's
    // duplicate-.reg rejection). For canonical 32x32x32 / 64x64x64 configs
    // both are exact and no guard is emitted.
    let row_guarded = (row_count % 8) != 0;
    let col_guarded = (hd % 32) != 0;
    if row_guarded {
        ptx.push_str(&format!(
            "    .reg .pred %psv_{label}_rok;\n"
        ));
    }
    if col_guarded {
        ptx.push_str(&format!(
            "    .reg .pred %psv_{label}_cok;\n"
        ));
    }
    if row_guarded && col_guarded {
        ptx.push_str(&format!(
            "    .reg .pred %psv_{label}_ok;\n"
        ));
    }

    // Null-guard: skip the whole sweep if the caller passed a null pointer.
    ptx.push_str(&format!(
        "    ld.param.u64 %psv_{label}_ptr, [{ptr_param}];\n"
    ));
    ptx.push_str(&format!(
        "    setp.eq.u64 %psv_{label}_null, %psv_{label}_ptr, 0;\n"
    ));
    ptx.push_str(&format!(
        "    @%psv_{label}_null bra V2_TIER_B1_SAVE_{label}_SKIP;\n"
    ));

    // warp_id and lane (recomputed locally from %tid.x to stay independent
    // of any earlier register lifetime).
    ptx.push_str(&format!(
        "    mov.u32 %psv_{label}_wid, %tid.x;\n"
    ));
    ptx.push_str(&format!(
        "    shr.u32 %psv_{label}_wid, %psv_{label}_wid, 5;   // warp_id = tid.x / 32\n"
    ));
    ptx.push_str(&format!(
        "    mov.u32 %psv_{label}_lane, %tid.x;\n"
    ));
    ptx.push_str(&format!(
        "    and.b32 %psv_{label}_lane, %psv_{label}_lane, 31; // lane = tid.x % 32\n"
    ));

    // warp-per-row / lane-per-col sweep. Unrolled over the (rows_per_warp x
    // cols_per_lane) cells each thread owns. row r = warp_id + rw*8;
    // col c = lane + cl*32. Out-of-range cells are skipped at compile time
    // when row_count/hd are exact multiples of 8/32 (canonical 32x32x32:
    // rows_per_warp=4, cols_per_lane=1).
    for rw in 0..rows_per_warp {
        // r = warp_id + rw*8
        ptx.push_str(&format!(
            "    add.u32 %psv_{label}_r, %psv_{label}_wid, {};\n",
            rw * 8
        ));
        // r < row_count guard only needed if row_count is not a multiple of 8.
        if row_guarded {
            ptx.push_str(&format!(
                "    setp.lt.u32 %psv_{label}_rok, %psv_{label}_r, {};\n",
                row_count
            ));
        }
        for cl in 0..cols_per_lane {
            // c = lane + cl*32
            ptx.push_str(&format!(
                "    add.u32 %psv_{label}_c, %psv_{label}_lane, {};\n",
                cl * 32
            ));
            // c < head_dim guard only needed if head_dim is not a multiple of 32.
            if col_guarded {
                ptx.push_str(&format!(
                    "    setp.lt.u32 %psv_{label}_cok, %psv_{label}_c, {};\n",
                    hd
                ));
            }

            // SMEM read byte:
            //   row-major (Q/K): r*(hd*2) + c*2
            //   col-major (V):   c*(bkv*2) + r*2   (R2 transpose)
            ptx.push_str(&format!(
                "    cvt.u64.u32 %psv_{label}_r64, %psv_{label}_r;\n"
            ));
            ptx.push_str(&format!(
                "    cvt.u64.u32 %psv_{label}_c64, %psv_{label}_c;\n"
            ));
            if col_major {
                // smem_byte = c*(bkv*2) + r*2
                ptx.push_str(&format!(
                    "    mul.lo.u64 %psv_{label}_smem_b, %psv_{label}_c64, {};  // c * (bkv*2)\n",
                    bkv * 2
                ));
                ptx.push_str(&format!(
                    "    shl.b64 %psv_{label}_r64, %psv_{label}_r64, 1;  // r*2\n"
                ));
                ptx.push_str(&format!(
                    "    add.u64 %psv_{label}_smem_b, %psv_{label}_smem_b, %psv_{label}_r64;\n"
                ));
            } else {
                // smem_byte = r*(hd*2) + c*2
                ptx.push_str(&format!(
                    "    mul.lo.u64 %psv_{label}_smem_b, %psv_{label}_r64, {};  // r * (hd*2)\n",
                    hd * 2
                ));
                ptx.push_str(&format!(
                    "    shl.b64 %psv_{label}_c64, %psv_{label}_c64, 1;  // c*2\n"
                ));
                ptx.push_str(&format!(
                    "    add.u64 %psv_{label}_smem_b, %psv_{label}_smem_b, %psv_{label}_c64;\n"
                ));
            }
            // smem_addr = shmem_base + smem_off + smem_byte
            if smem_off != 0 {
                ptx.push_str(&format!(
                    "    add.u64 %psv_{label}_smem_addr, %shmem_base, {};\n",
                    smem_off
                ));
                ptx.push_str(&format!(
                    "    add.u64 %psv_{label}_smem_addr, %psv_{label}_smem_addr, %psv_{label}_smem_b;\n"
                ));
            } else {
                ptx.push_str(&format!(
                    "    add.u64 %psv_{label}_smem_addr, %shmem_base, %psv_{label}_smem_b;\n"
                ));
            }

            // HBM seq index:
            //   Q:   seq = q_start + r
            //   K/V: seq = r            (R1, absolute K position)
            // Recompute c64 (it may have been clobbered by the row-major
            // shl above) for the HBM column term.
            ptx.push_str(&format!(
                "    cvt.u64.u32 %psv_{label}_c64, %psv_{label}_c;\n"
            ));
            ptx.push_str(&format!(
                "    cvt.u64.u32 %psv_{label}_seq, %psv_{label}_r;\n"
            ));
            if use_q_start {
                ptx.push_str(&format!(
                    "    add.u64 %psv_{label}_seq, %psv_{label}_seq, %q_start;  // R1: Q seq = q_start + r\n"
                ));
            } else {
                ptx.push_str(&format!(
                    "    // R1: K/V seq = r (absolute K position, NOT q_start+r)\n"
                ));
            }
            // hbm_elem = (((batch*heads + head)*seq + s)*hd + c)
            // hbm_byte = hbm_elem * 2  (f16)
            ptx.push_str(&format!(
                "    mul.lo.u64 %psv_{label}_idx, %batch_idx, %rd5;\n"
            ));
            ptx.push_str(&format!(
                "    add.u64 %psv_{label}_idx, %psv_{label}_idx, %head_idx;\n"
            ));
            ptx.push_str(&format!(
                "    mul.lo.u64 %psv_{label}_idx, %psv_{label}_idx, %rd6;\n"
            ));
            ptx.push_str(&format!(
                "    add.u64 %psv_{label}_idx, %psv_{label}_idx, %psv_{label}_seq;\n"
            ));
            ptx.push_str(&format!(
                "    mul.lo.u64 %psv_{label}_idx, %psv_{label}_idx, %rd7;\n"
            ));
            ptx.push_str(&format!(
                "    add.u64 %psv_{label}_idx, %psv_{label}_idx, %psv_{label}_c64;\n"
            ));
            ptx.push_str(&format!(
                "    shl.b64 %psv_{label}_idx, %psv_{label}_idx, 1;  // f16 = 2 bytes\n"
            ));
            ptx.push_str(&format!(
                "    add.u64 %psv_{label}_hbm, %psv_{label}_ptr, %psv_{label}_idx;\n"
            ));

            // ld.shared + st.global, optionally row/col-range guarded.
            ptx.push_str(&format!(
                "    ld.shared.b16 %psv_{label}_h, [%psv_{label}_smem_addr];\n"
            ));
            let gate = match (row_guarded, col_guarded) {
                (false, false) => None,
                (true, false) => Some(format!("%psv_{label}_rok")),
                (false, true) => Some(format!("%psv_{label}_cok")),
                (true, true) => {
                    ptx.push_str(&format!(
                        "    and.pred %psv_{label}_ok, %psv_{label}_rok, %psv_{label}_cok;\n"
                    ));
                    Some(format!("%psv_{label}_ok"))
                }
            };
            match gate {
                Some(g) => ptx.push_str(&format!(
                    "    @{g} st.global.b16 [%psv_{label}_hbm], %psv_{label}_h;  // {ptr_param} write\n"
                )),
                None => ptx.push_str(&format!(
                    "    st.global.b16 [%psv_{label}_hbm], %psv_{label}_h;  // {ptr_param} write\n"
                )),
            }
        }
    }

    ptx.push_str(&format!("V2_TIER_B1_SAVE_{label}_SKIP:\n"));
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

    /// Same as `make_config` but with `save_activations_for_backward=true`,
    /// which triggers the Phase 2.6 (T2.5) retrofit emission.
    fn make_save_config(bq: i64, bkv: i64, hd: i64) -> FlashAttentionConfig {
        let mut cfg = make_config(bq, bkv, hd);
        if let Some(c) = cfg.csha.as_mut() {
            c.save_activations_for_backward = true;
        }
        cfg
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
    fn emits_unpredicated_b16_stores_post_n3() {
        // After N3 refactor, each warp writes to its OWN distinct global
        // tile, so the b16 stores are no longer warp-predicated (each
        // address is unique per warp). 4 stores per slot × tpw slots.
        let cfg = make_config(32, 32, 32);
        let tpw = tiles_per_warp_pv(&cfg);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert_eq!(
            ptx.matches("    st.global.b16 [%fin_addr], %fin_h;\n").count(),
            (tpw * 4) as usize,
            "expected 4 unpredicated f16 stores per tile (D-fragment lanes 0..3)"
        );
        assert!(
            !ptx.contains("%fin_warp_pred"),
            "%fin_warp_pred register should no longer be declared/used after N3"
        );
    }

    #[test]
    fn computes_runtime_tile_coords_from_warp_id() {
        // N3 invariant: every per-slot iter must compute m_tile/n_tile
        // from %warp_id at runtime.
        let cfg = make_config(32, 32, 32);
        let tpw = tiles_per_warp_pv(&cfg);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // First slot uses `mov %fin_global_t, %warp_id`.
        assert!(
            ptx.contains("mov.u32 %fin_global_t, %warp_id;"),
            "slot 0 must initialize %fin_global_t from %warp_id"
        );
        // Each slot emits one shr (m_tile = global_t >> log2(n_tiles_d)).
        assert_eq!(
            ptx.matches("shr.u32 %fin_m_tile, %fin_global_t,").count(),
            tpw as usize,
            "expected one m_tile shr per slot"
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

    // ===== Phase 2.6 (T2.5) save-activation retrofit unit tests =====

    #[test]
    fn no_save_emission_when_flag_off() {
        // The default config (save_activations_for_backward=false) must emit
        // ZERO retrofit code -- no param loads, no %psv_* registers, no save
        // stores. Inference builds pay nothing.
        let cfg = make_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert!(!ptx.contains("q_proj_ptr"), "no q_proj_ptr load when flag off");
        assert!(!ptx.contains("k_proj_ptr"), "no k_proj_ptr load when flag off");
        assert!(!ptx.contains("v_proj_ptr"), "no v_proj_ptr load when flag off");
        assert!(!ptx.contains("row_max_ptr"), "no row_max_ptr load when flag off");
        assert!(!ptx.contains("row_sum_ptr"), "no row_sum_ptr load when flag off");
        assert!(!ptx.contains("%psv_"), "no %psv_ registers when flag off");
        assert!(
            !ptx.contains("V2_TIER_B1_SAVE_"),
            "no save SKIP labels when flag off"
        );
    }

    #[test]
    fn emits_param_loads_for_all_five_saves() {
        let cfg = make_save_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        for p in ["q_proj_ptr", "k_proj_ptr", "v_proj_ptr", "row_max_ptr", "row_sum_ptr"] {
            assert_eq!(
                ptx.matches(&format!(", [{p}];")).count(),
                1,
                "expected exactly one ld.param.u64 for {p}"
            );
        }
    }

    #[test]
    fn emits_null_guards_for_projection_pointers() {
        let cfg = make_save_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // Each projection save null-guards + branches to its SKIP label.
        for label in ["Q", "K", "V"] {
            assert!(
                ptx.contains(&format!("setp.eq.u64 %psv_{label}_null, %psv_{label}_ptr, 0;")),
                "missing null-guard setp for {label}"
            );
            assert!(
                ptx.contains(&format!("@%psv_{label}_null bra V2_TIER_B1_SAVE_{label}_SKIP;")),
                "missing null-guard branch for {label}"
            );
            assert!(
                ptx.contains(&format!("V2_TIER_B1_SAVE_{label}_SKIP:")),
                "missing SKIP label for {label}"
            );
        }
        // row_max/row_sum use setp.ne.u64 null-guards (ANDed into the gate).
        assert!(
            ptx.contains("setp.ne.u64 %psv_has_row_max, %psv_row_max_base, 0;"),
            "missing row_max null-guard"
        );
        assert!(
            ptx.contains("setp.ne.u64 %psv_has_row_sum, %psv_row_sum_base, 0;"),
            "missing row_sum null-guard"
        );
    }

    #[test]
    fn emits_expected_b16_store_count_per_projection() {
        // Canonical 32x32x32: rows_per_warp=4, cols_per_lane=1.
        // Each projection sweep emits rows_per_warp * cols_per_lane stores
        // unrolled in PTX (the per-thread cell count). row_count is a
        // multiple of 8 and hd a multiple of 32, so stores are unguarded.
        let cfg = make_save_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // 3 projections (Q/K/V) each emit 4 unguarded f16 save stores.
        let q_writes = ptx.matches("// q_proj_ptr write").count();
        let k_writes = ptx.matches("// k_proj_ptr write").count();
        let v_writes = ptx.matches("// v_proj_ptr write").count();
        assert_eq!(q_writes, 4, "expected 4 Q save stores (rows_per_warp*cols_per_lane)");
        assert_eq!(k_writes, 4, "expected 4 K save stores");
        assert_eq!(v_writes, 4, "expected 4 V save stores");
    }

    #[test]
    fn emits_row_max_row_sum_stores_inline_with_lse() {
        // row_max/row_sum mirror the LSE store: tpw*2 gated f32 stores each.
        let cfg = make_save_config(32, 32, 32);
        let tpw = tiles_per_warp_pv(&cfg);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert_eq!(
            ptx.matches("// row_max_ptr write").count(),
            (tpw * 2) as usize,
            "expected 2 row_max stores per tile"
        );
        assert_eq!(
            ptx.matches("// row_sum_ptr write").count(),
            (tpw * 2) as usize,
            "expected 2 row_sum stores per tile"
        );
        // Stored value must be the reduced-stats SMEM-sourced (m-tile-keyed)
        // %rstat_max / %rstat_sum, NOT the QK^T-slot-keyed %s_max_<t>/%s_sum_<t>
        // registers (which alias the wrong m-tile at hd>bkv). The value is the
        // RAW global max/sum (not the LSE = max + ln(sum) combo).
        assert!(
            ptx.contains("@%psv_row_max_gate st.global.f32 [%psv_row_max_addr], %rstat_max;"),
            "row_max store must write the SMEM-sourced %rstat_max (hd>bkv fix)"
        );
        assert!(
            ptx.contains("@%psv_row_sum_gate st.global.f32 [%psv_row_sum_addr], %rstat_sum;"),
            "row_sum store must write the SMEM-sourced %rstat_sum (hd>bkv fix)"
        );
        // The stores must NO LONGER reference the slot-keyed registers directly.
        assert!(
            !ptx.contains("[%psv_row_max_addr], %s_max_"),
            "row_max store must not source the QK^T-slot-keyed %s_max_<t> register"
        );
        assert!(
            !ptx.contains("[%psv_row_sum_addr], %s_sum_"),
            "row_sum store must not source the QK^T-slot-keyed %s_sum_<t> register"
        );
        // The reduced-stats SMEM reads that feed the stores must be present.
        assert!(
            ptx.contains("ld.shared.f32 %rstat_max, [%rstat_addr];")
                && ptx.contains("ld.shared.f32 %rstat_sum, [%rstat_addr];"),
            "finalize must read row_max/row_sum from the reduced-stats SMEM region"
        );
    }

    #[test]
    fn finalize_reads_softmax_stats_from_smem_not_slot_registers() {
        // hd > block_kv fix: finalize must source row_max/row_sum from the
        // reduced-stats SMEM region keyed by absolute global_row, NOT from the
        // QK^T-slot-keyed %s_max_<t>/%s_sum_<t> registers (which the producer
        // only declares over tiles_per_warp_qkt, and which map slot->wrong
        // m_tile at hd>bkv). Exercise an hd>bkv config (32x32x64) where the
        // old code path would have referenced undeclared %s_*_1_* registers.
        let cfg = make_save_config(32, 32, 64);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // The divide must consume the SMEM-loaded %rstat_sum, not %s_sum_<t>.
        assert!(
            ptx.contains("div.approx.f32 %fin_norm, %o_acc_0_0, %rstat_sum;"),
            "divide must normalize by the SMEM-sourced %rstat_sum"
        );
        // No emitted instruction may read a %s_sum_<t>/%s_max_<t> register
        // (comments are allowed; instruction operands are not). Check the
        // operand forms the old code used.
        assert!(
            !ptx.contains(", %s_sum_0_") && !ptx.contains(", %s_sum_1_"),
            "finalize must not read %s_sum_<t>_<half> registers (hd>bkv fix)"
        );
        assert!(
            !ptx.contains(", %s_max_0_") && !ptx.contains(", %s_max_1_"),
            "finalize must not read %s_max_<t>_<half> registers (hd>bkv fix)"
        );
        // Reduced-stats SMEM base addresses must be set up.
        assert!(
            ptx.contains("cvt.u32.u64 %rstat_max_base, %rstat_base_u64;")
                && ptx.contains("cvt.u32.u64 %rstat_sum_base, %rstat_base_u64;"),
            "finalize must compute the reduced-stats SMEM base addresses"
        );
    }

    #[test]
    fn v_save_uses_col_major_smem_read() {
        // R2: V reads col-major SMEM (c*(bkv*2)+r*2), Q/K read row-major
        // (r*(hd*2)+c*2). For 32x32x32, bkv*2 = 64 and hd*2 = 64 -- the
        // STRIDE is the same integer, so distinguish by which operand is
        // multiplied: V multiplies the COL (c64), Q/K multiply the ROW (r64).
        let cfg = make_save_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // V col-major: mul of the column index by (bkv*2).
        assert!(
            ptx.contains("mul.lo.u64 %psv_V_smem_b, %psv_V_c64, 64;  // c * (bkv*2)"),
            "V save must read col-major SMEM (multiply COL by bkv*2)"
        );
        // Q/K row-major: mul of the row index by (hd*2).
        assert!(
            ptx.contains("mul.lo.u64 %psv_Q_smem_b, %psv_Q_r64, 64;  // r * (hd*2)"),
            "Q save must read row-major SMEM (multiply ROW by hd*2)"
        );
        assert!(
            ptx.contains("mul.lo.u64 %psv_K_smem_b, %psv_K_r64, 64;  // r * (hd*2)"),
            "K save must read row-major SMEM (multiply ROW by hd*2)"
        );
        // V must NOT emit the row-major pattern.
        assert!(
            !ptx.contains("mul.lo.u64 %psv_V_smem_b, %psv_V_r64,"),
            "V save must NOT use the row-major (row-multiplied) SMEM index"
        );
    }

    #[test]
    fn k_v_use_absolute_seq_index_q_uses_q_start() {
        // R1 (load-bearing): Q seq = q_start + r; K/V seq = r (absolute).
        let cfg = make_save_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        // Q adds %q_start to the seq term.
        assert!(
            ptx.contains("add.u64 %psv_Q_seq, %psv_Q_seq, %q_start;  // R1: Q seq = q_start + r"),
            "Q save must add q_start to the seq index"
        );
        // K/V must NOT add %q_start to their seq term.
        assert!(
            !ptx.contains("add.u64 %psv_K_seq, %psv_K_seq, %q_start"),
            "K save must use absolute seq index r (NOT q_start + r)"
        );
        assert!(
            !ptx.contains("add.u64 %psv_V_seq, %psv_V_seq, %q_start"),
            "V save must use absolute seq index r (NOT q_start + r)"
        );
    }

    #[test]
    fn projection_save_registers_are_label_namespaced() {
        // R6: the 3 calls must not collide on a duplicate .reg. Each save
        // declares its own %psv_<label>_* pool, so the three .reg lines for
        // (e.g.) the HBM pointer are distinct.
        let cfg = make_save_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert!(ptx.contains(".reg .u64 %psv_Q_ptr;"));
        assert!(ptx.contains(".reg .u64 %psv_K_ptr;"));
        assert!(ptx.contains(".reg .u64 %psv_V_ptr;"));
        // No bare un-namespaced %psv_ptr that would collide across calls.
        assert!(!ptx.contains(".reg .u64 %psv_ptr;"));
    }

    #[test]
    fn save_sweep_has_fence_before_projection_scatter() {
        let cfg = make_save_config(32, 32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert!(
            ptx.contains("bar.sync 0;  // FENCE: Q/K/V SMEM tiles fully visible to save sweep"),
            "projection save sweep must be preceded by a visibility fence"
        );
    }
}
