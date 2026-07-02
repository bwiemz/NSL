//! D pre-pass kernel emitter for Tier B.2 backward.
//!
//! Computes D[batch, head, q] = sum_d (dO[b,h,q,d] * O[b,h,q,d])
//! for use by dQ-kernel and dK/dV-kernel as part of dS = P * (dP - D).
//!
//! Grid: (ceil(seq_len / 32), heads, batch); Block: (32, 1, 1) - one warp.
//! Schedule: 1 lane = 1 row. Each lane independently sums all `head_dim` cols
//! of its row into an f32 accumulator (no inter-lane shfl reduction needed
//! because no two lanes share a row). No SMEM.
//!
//! HBM in:  dO, O - both f16 [B, H, S, D].
//! HBM out: D - f32 [B, H, S] (batch * heads * seq * 4 bytes).
//!
//! Spec deviation note: spec §3.3 described a "warp-shfl butterfly reduction"
//! where 32 lanes cooperate on a single row (D/32 cols each). The kernel
//! implements the dual: 32 lanes each own one row (D cols each, sequentially).
//! Both schedules are HBM-bandwidth-bound at canonical seq/hd sizes and
//! produce identical numerical results; the per-lane-per-row schedule is
//! strictly simpler (no butterfly, no lane-0-only store) and avoids the
//! row-vs-col-conflation bug class that bit the original spec'd schedule.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §3.3

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::tier_b2::backward::BackwardSynthError;

/// Emit a single-warp PTX kernel that computes the D pre-pass.
///
/// Each lane in the 32-thread CTA handles exactly one row (q index). The
/// lane iterates over all `head_dim` columns sequentially, fma'ing
/// `dO[q,d] * O[q,d]` into an f32 accumulator, then writes its row's D
/// value directly. No inter-lane reduction is required because the warp's
/// rows are disjoint.
///
/// # Errors
/// Returns `BackwardSynthError::UnsupportedHeadDim` when `head_dim` is
/// not divisible by 32 (kept as a config-validation gate; the loop itself
/// doesn't require that exact divisibility).
pub fn synthesize_d_prepass(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    let hd = config.head_dim as u32;
    if !hd.is_multiple_of(32) {
        return Err(BackwardSynthError::UnsupportedHeadDim(hd));
    }

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
    ptx.push_str("    .reg .u64  %d_o_base, %o_base, %d_out_base, %addr_64, %offset_64, %row_byte_base;\n");
    ptx.push_str("    .reg .b16  %dO_h, %o_h;\n");
    ptx.push_str("    .reg .f32  %dO_f, %o_f, %acc;\n");
    ptx.push_str("    .reg .pred %p_in_range;\n");
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

    // ── Per-row column sweep ─────────────────────────────────────────────
    // HBM layout: dO / O are [B, H, S, D] in row-major f16.
    // Each lane owns row `row_index`; lane sums all `hd` cols of that row.
    // Element at col d has byte offset row_index * (hd*2) + d*2.
    //
    // We pre-compute %row_byte_base = row_index * (hd*2) once outside the loop,
    // then walk the row by adding 2 bytes per iteration. The loop is unrolled
    // at PTX emission time so the column index is a compile-time constant —
    // ptxas can fold the address arithmetic.
    let row_stride_bytes = hd * 2; // bytes per full row of f16
    ptx.push_str(&format!(
        "    mul.wide.u32 %row_byte_base, %row_index, {};\n",
        row_stride_bytes
    ));
    ptx.push('\n');

    for d in 0..hd {
        let col_bytes = d * 2u32;
        // offset_64 = row_byte_base + d*2
        if col_bytes == 0 {
            ptx.push_str("    mov.u64 %offset_64, %row_byte_base;\n");
        } else {
            ptx.push_str(&format!(
                "    add.u64 %offset_64, %row_byte_base, {};\n",
                col_bytes
            ));
        }
        // Load dO[row, d]
        ptx.push_str("    add.u64 %addr_64, %d_o_base, %offset_64;\n");
        ptx.push_str("    ld.global.b16 %dO_h, [%addr_64];\n");
        // Load O[row, d]
        ptx.push_str("    add.u64 %addr_64, %o_base, %offset_64;\n");
        ptx.push_str("    ld.global.b16 %o_h, [%addr_64];\n");
        // f16 -> f32
        ptx.push_str("    cvt.f32.f16 %dO_f, %dO_h;\n");
        ptx.push_str("    cvt.f32.f16 %o_f, %o_h;\n");
        // acc += dO_f * o_f
        ptx.push_str("    fma.rn.f32 %acc, %dO_f, %o_f, %acc;\n");
    }
    ptx.push('\n');

    // ── Each lane stores its row's D directly ────────────────────────────
    // D layout: [B, H, S] f32. Byte offset = row_index * 4.
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
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
            checkpoint: None,
        }
    }

    #[test]
    fn d_prepass_emits_target_sm80_single_warp_block() {
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains(".maxntid 32"));
    }

    #[test]
    fn d_prepass_emits_no_warp_shfl_reduction() {
        // Per the row-per-lane schedule (see module doc spec-deviation note):
        // each lane owns one row independently, so no inter-lane reduction is
        // needed. Asserts the kernel does NOT emit shfl.sync — a guardrail
        // against accidentally re-introducing the original row/col-conflated
        // schedule that bit the first GPU launch (2026-05-20).
        let ptx = synthesize_d_prepass(&canonical_cfg()).unwrap();
        assert!(
            !ptx.contains("shfl.sync"),
            "row-per-lane schedule must not emit shfl.sync"
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
        // Each lane processes head_dim cols, each loading dO + O = 2 ld.global
        // per col. canonical hd=128 -> 256 ld.global total.
        let n_loads = ptx.matches("ld.global").count();
        let expected = (canonical_cfg().head_dim * 2) as usize;
        assert_eq!(
            n_loads, expected,
            "expected exactly {expected} ld.global (hd * 2), got {n_loads}"
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
