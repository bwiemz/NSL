//! D pre-pass kernel emitter for Tier B.2 backward.
//!
//! Computes D[batch, head, q] = sum_d (dO[b,h,q,d] * O[b,h,q,d])
//! for use by dQ-kernel and dK/dV-kernel as part of dS = P * (dP - D).
//!
//! Grid: (ceil(seq_len / 32), heads, batch); Block: (32, 1, 1) — one warp.
//! SMEM: none. Per-warp reduction via shfl.sync.bfly butterfly.
//! HBM in:  dO, O — both f16 [B, H, S, D].
//! HBM out: D — f32 [B, H, S] (batch * heads * seq * 4 bytes).
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §3.3

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::tier_b2::backward::BackwardSynthError;

/// Emit a single-warp PTX kernel that computes the D pre-pass.
///
/// Each lane in the 32-thread CTA handles exactly one row (q index).
/// Within that row, each lane processes `head_dim / 32` f16 elements
/// in an interleaved pattern (`lane_id + iter * 32`) for coalesced HBM
/// access, fma's them into an f32 accumulator, then a 5-stage
/// `shfl.sync.bfly` reduces the 32 partial sums to lane 0, which
/// writes the result to D.
///
/// # Errors
/// Returns `BackwardSynthError::UnsupportedHeadDim` when `head_dim` is
/// not divisible by 32.
pub fn synthesize_d_prepass(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    let hd = config.head_dim as u32;
    if !hd.is_multiple_of(32) {
        return Err(BackwardSynthError::UnsupportedHeadDim(hd));
    }
    let elements_per_lane = hd / 32; // >= 1

    let mut ptx = String::new();

    // ── Prelude ──────────────────────────────────────────────────────────
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n\n");

    // ── Entry signature ──────────────────────────────────────────────────
    // `batch` is recovered from %ctaid.z (grid dim z) and `heads` is read for
    // the [B, H, S] row-index flattening below — no separate `batch` param
    // is needed because the grid-z dimension determines that index at launch.
    ptx.push_str(".visible .entry tier_b2_d_prepass(\n");
    ptx.push_str("    .param .u64 d_o_ptr,\n");
    ptx.push_str("    .param .u64 o_ptr,\n");
    ptx.push_str("    .param .u64 d_out_ptr,\n");
    ptx.push_str("    .param .u32 seq_len,\n");
    ptx.push_str("    .param .u32 heads\n");
    ptx.push_str(")\n");
    ptx.push_str(".maxntid 32, 1, 1\n");
    ptx.push_str("{\n");

    // ── Register declarations ────────────────────────────────────────────
    ptx.push_str("    .reg .u32  %lane_id, %q_strip, %head, %batch_idx, %row_global;\n");
    ptx.push_str("    .reg .u32  %seq_len_r, %heads_r;\n");
    ptx.push_str("    .reg .u32  %row_index;\n");
    ptx.push_str("    .reg .u64  %d_o_base, %o_base, %d_out_base, %addr_64, %offset_64, %lane_off;\n");
    ptx.push_str("    .reg .b16  %dO_h, %o_h;\n");
    ptx.push_str("    .reg .f32  %dO_f, %o_f, %acc, %tmp;\n");
    ptx.push_str("    .reg .pred %p_in_range, %p_lane0;\n");
    ptx.push('\n');

    // ── Load kernel parameters ───────────────────────────────────────────
    ptx.push_str("    ld.param.u64 %d_o_base,   [d_o_ptr];\n");
    ptx.push_str("    ld.param.u64 %o_base,      [o_ptr];\n");
    ptx.push_str("    ld.param.u64 %d_out_base,  [d_out_ptr];\n");
    ptx.push_str("    ld.param.u32 %seq_len_r,   [seq_len];\n");
    ptx.push_str("    ld.param.u32 %heads_r,     [heads];\n");
    ptx.push('\n');

    // ── Compute logical row index ─────────────────────────────────────────
    // lane_id  = tid.x  (0..31 within warp)
    // q_strip  = ctaid.x (which group of 32 rows this CTA handles)
    // head     = ctaid.y
    // batch_idx = ctaid.z
    // row_global = q_strip * 32 + lane_id  (the q index this lane owns)
    ptx.push_str("    mov.u32 %lane_id,    %tid.x;\n");
    ptx.push_str("    mov.u32 %q_strip,    %ctaid.x;\n");
    ptx.push_str("    mov.u32 %head,       %ctaid.y;\n");
    ptx.push_str("    mov.u32 %batch_idx,  %ctaid.z;\n");
    ptx.push('\n');
    ptx.push_str("    mul.lo.u32 %row_global, %q_strip, 32;\n");
    ptx.push_str("    add.u32    %row_global, %row_global, %lane_id;\n");
    ptx.push('\n');

    // ── Bounds check (last CTA strip may be partial) ─────────────────────
    ptx.push_str("    setp.lt.u32 %p_in_range, %row_global, %seq_len_r;\n");
    ptx.push_str("    @!%p_in_range bra D_PREPASS_DONE;\n");
    ptx.push('\n');

    // ── Flat row index in [B, H, S] layout ──────────────────────────────
    // row_index = (batch_idx * heads + head) * seq_len + row_global
    //
    // u32 range: at the largest canonical config (batch=8, heads=32, seq=8192)
    // the max row_index is (8*32 + 31) * 8192 + 8191 ≈ 2.1M, well below 2^32.
    // The downstream `mul.wide.u32 %offset_64, %row_index, hd*2` widens to u64
    // before any byte-offset arithmetic, so overflow risk is bounded to the
    // u32 intermediate above (and confirmed safe).
    ptx.push_str("    mul.lo.u32 %row_index, %batch_idx, %heads_r;\n");
    ptx.push_str("    add.u32    %row_index, %row_index, %head;\n");
    ptx.push_str("    mul.lo.u32 %row_index, %row_index, %seq_len_r;\n");
    ptx.push_str("    add.u32    %row_index, %row_index, %row_global;\n");
    ptx.push('\n');

    // ── Initialise accumulator ───────────────────────────────────────────
    ptx.push_str("    mov.f32 %acc, 0.0;\n");
    ptx.push('\n');

    // ── Pre-compute per-lane base byte offset: lane_id * 2 bytes ─────────
    // lane_off = (u64) lane_id * 2
    ptx.push_str("    mul.wide.u32 %lane_off, %lane_id, 2;\n");
    ptx.push('\n');

    // ── Per-lane element loop ─────────────────────────────────────────────
    // HBM layout: dO / O are [B, H, S, D] in row-major f16.
    // Element at (row_index, d) has byte offset = row_index * hd * 2 + d * 2.
    // Lane `lane_id` owns the interleaved elements:
    //   d = lane_id + iter * 32   for iter in 0..elements_per_lane
    // Byte offset within the row: (lane_id + iter*32) * 2
    //                           = lane_id*2 + iter*64
    //
    // Total byte address = base + row_index * (hd*2) + lane_id*2 + iter*64
    //
    // We pre-compute `row_byte_base = base + row_index * (hd*2) + lane_id*2`
    // then add the constant `iter*64` per iteration.
    let row_stride_bytes = hd * 2; // bytes per full row of f16

    for i in 0..elements_per_lane {
        let iter_extra_bytes = i * 64u32; // iter * 32 lanes * 2 bytes/f16
        ptx.push_str(&format!("    // Element {}/{} — d = lane_id + {}*32\n", i + 1, elements_per_lane, i));

        // offset_64 = row_index * row_stride_bytes
        ptx.push_str(&format!(
            "    mul.wide.u32 %offset_64, %row_index, {};\n",
            row_stride_bytes
        ));
        // offset_64 += lane_id * 2
        ptx.push_str("    add.u64 %offset_64, %offset_64, %lane_off;\n");
        // offset_64 += iter * 64  (constant, skip for i==0)
        if iter_extra_bytes > 0 {
            ptx.push_str(&format!(
                "    add.u64 %offset_64, %offset_64, {};\n",
                iter_extra_bytes
            ));
        }

        // Load dO[row_global, d]
        ptx.push_str("    add.u64 %addr_64, %d_o_base, %offset_64;\n");
        ptx.push_str("    ld.global.b16 %dO_h, [%addr_64];\n");

        // Load O[row_global, d]
        ptx.push_str("    add.u64 %addr_64, %o_base, %offset_64;\n");
        ptx.push_str("    ld.global.b16 %o_h, [%addr_64];\n");

        // Convert f16 -> f32
        ptx.push_str("    cvt.f32.f16 %dO_f, %dO_h;\n");
        ptx.push_str("    cvt.f32.f16 %o_f, %o_h;\n");

        // Accumulate: acc += dO_f * o_f
        ptx.push_str("    fma.rn.f32 %acc, %dO_f, %o_f, %acc;\n");
        ptx.push('\n');
    }

    // ── Warp butterfly reduction: 32 partial sums → lane 0 ──────────────
    // 5 stages: stride 16, 8, 4, 2, 1
    ptx.push_str("    // Warp-shfl butterfly reduction (32 -> 1)\n");
    for stride in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %tmp, %acc, {}, 0x1f, 0xffffffff;\n",
            stride
        ));
        ptx.push_str("    add.f32 %acc, %acc, %tmp;\n");
    }
    ptx.push('\n');

    // ── Lane 0 stores D[batch, head, row_global] ─────────────────────────
    // D layout: [B, H, S] f32 — byte offset = row_index * 4
    ptx.push_str("    setp.eq.u32 %p_lane0, %lane_id, 0;\n");
    ptx.push_str("    @!%p_lane0 bra D_PREPASS_DONE;\n");
    ptx.push('\n');
    ptx.push_str("    mul.wide.u32 %offset_64, %row_index, 4;\n");
    ptx.push_str("    add.u64 %addr_64, %d_out_base, %offset_64;\n");
    ptx.push_str("    st.global.f32 [%addr_64], %acc;\n");
    ptx.push('\n');

    ptx.push_str("D_PREPASS_DONE:\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 128,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
        }
    }

    #[test]
    fn d_prepass_emits_target_sm80_single_warp_block() {
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains(".maxntid 32"));
    }

    #[test]
    fn d_prepass_emits_warp_shfl_reduction() {
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        assert!(ptx.contains("shfl.sync.bfly"));
        // 5 butterfly stages for 32->1 reduction
        assert_eq!(
            ptx.matches("shfl.sync.bfly").count(),
            5,
            "expected exactly 5 shfl.sync.bfly stages"
        );
    }

    #[test]
    fn d_prepass_writes_f32_d_output() {
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        assert!(ptx.contains("st.global.f32"));
    }

    #[test]
    fn d_prepass_uses_no_smem() {
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        assert!(!ptx.contains(".shared .align"));
        assert!(!ptx.contains(".extern .shared"));
    }

    #[test]
    fn d_prepass_loads_d_o_and_o_from_global() {
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        // Both dO and O must be loaded from HBM (f16).
        // hd=128 -> 4 elements/lane -> 4 dO loads + 4 O loads = 8 ld.global
        assert!(
            ptx.matches("ld.global").count() >= 2,
            "expected at least 2 ld.global (dO + O), got:\n{ptx}"
        );
    }

    #[test]
    fn d_prepass_uses_f32_fma_accumulation() {
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        // f16 inputs converted to f32 for accumulation (fma.rn.f32 or mul+add)
        assert!(
            ptx.contains("fma.rn.f32") || (ptx.contains("mul.f32") && ptx.contains("add.f32")),
            "expected f32 fma or mul+add accumulation, got:\n{ptx}",
        );
    }

    #[test]
    fn d_prepass_rejects_head_dim_not_divisible_by_32() {
        let mut cfg = canonical_cfg();
        cfg.head_dim = 48;
        let result = synthesize_d_prepass(&cfg);
        assert_eq!(result, Err(BackwardSynthError::UnsupportedHeadDim(48)));
    }

    #[test]
    fn d_prepass_scales_per_lane_load_count_with_head_dim() {
        // hd=128 -> 4 elements/lane -> 8 ld.global (4 dO + 4 O)
        // hd=32  -> 1 element/lane  -> 2 ld.global (1 dO + 1 O)
        let mut cfg = canonical_cfg();
        cfg.head_dim = 128;
        let ptx_128 = synthesize_d_prepass(&cfg).unwrap();
        cfg.head_dim = 32;
        let ptx_32 = synthesize_d_prepass(&cfg).unwrap();
        // hd=128 must emit MORE per-lane load+fma pairs than hd=32
        assert!(
            ptx_128.matches("ld.global").count() > ptx_32.matches("ld.global").count(),
            "hd=128 should have more ld.global than hd=32"
        );
    }
}
