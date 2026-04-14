//! Phase 6 - finalize O = O_acc / row_sum, store f16 output + LSE.
//!
//! After the K-tile loop completes: O_acc holds the warp-row's
//! accumulated sum_k P[q_row, k] * V[k, d]. Divide by row_sum, cvt to
//! f16, store coalesced to global memory. Lane 0 of each warp also
//! writes the logsumexp (row_max + ln(row_sum)) for the backward pass,
//! null-guarded on the logsumexp_base pointer.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::phases::pv_accum::O_BASE;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let head_dim = config.head_dim as u32;
    let slices = head_dim / 32;

    ptx.push_str(&format!(
        "    // Phase 6: finalize + output store (q_tile_iter = {})\n",
        q_tile_iter
    ));

    // Reciprocal of row_sum.
    ptx.push_str("    rcp.approx.f32 %f0, %row_sum;\n");

    // Normalise each O_acc slice.
    for i in 0..slices {
        ptx.push_str(&format!(
            "    mul.f32 %f{}, %f{}, %f0;\n",
            O_BASE + i,
            O_BASE + i
        ));
    }

    // Output base: out_ptr + (batch*heads*seq_len*head_dim
    //                        + head_idx*seq_len*head_dim
    //                        + q_row_global*head_dim) * 2 (f16 bytes).
    ptx.push_str(&format!(
        "    add.u32 %r7, %warp_id, {};             // q_row_local = warp_id + q_tile_iter*4\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %rd46, %r7;\n");
    ptx.push_str("    add.u64 %rd47, %q_start, %rd46;           // q_row_global\n");
    ptx.push_str("    mul.lo.u64 %rd48, %batch_idx, %rd5;       // batch*heads\n");
    ptx.push_str("    add.u64 %rd48, %rd48, %head_idx;          // + head\n");
    ptx.push_str("    mul.lo.u64 %rd48, %rd48, %rd6;            // * seq_len\n");
    ptx.push_str("    add.u64 %rd48, %rd48, %rd47;              // + q_row_global\n");
    ptx.push_str("    mul.lo.u64 %rd48, %rd48, %rd7;            // * head_dim\n");
    ptx.push_str("    shl.b64 %rd48, %rd48, 1;                  // * 2 bytes f16\n");
    ptx.push_str("    add.u64 %rd48, %rd3, %rd48;               // out_base_global\n");

    // Each lane writes head_dim/32 f16 values.
    for i in 0..slices {
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %h0, %f{};\n",
            O_BASE + i
        ));
        ptx.push_str("    cvt.u64.u32 %rd49, %lane;\n");
        if i > 0 {
            ptx.push_str(&format!("    add.u64 %rd49, %rd49, {};\n", i * 32));
        }
        ptx.push_str("    shl.b64 %rd49, %rd49, 1;                  // * 2 bytes\n");
        ptx.push_str("    add.u64 %rd49, %rd48, %rd49;              // out_base + d*2\n");
        ptx.push_str("    st.global.b16 [%rd49], %h0;\n");
    }

    // LSE: lane 0 of each warp writes logsumexp[batch, head, q_row_global].
    ptx.push_str("    // LSE store (lane 0 only, null-guarded)\n");
    ptx.push_str("    setp.eq.u32 %p1, %lane, 0;\n");
    ptx.push_str("    setp.ne.u64 %p_has_lse, %logsumexp_base, 0;\n");
    ptx.push_str("    and.pred %p1, %p1, %p_has_lse;\n");
    ptx.push_str("    lg2.approx.f32 %log_sum, %row_sum;\n");
    ptx.push_str("    mov.f32 %f1, 0f3F317218;                  // ln(2)\n");
    ptx.push_str("    mul.f32 %log_sum, %log_sum, %f1;          // log_sum = ln(row_sum)\n");
    ptx.push_str("    add.f32 %lse, %row_max, %log_sum;\n");
    // lse_addr = logsumexp_base + (batch*heads*seq_len + head*seq_len + q_row_global) * 4
    ptx.push_str("    mul.lo.u64 %rd50, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd50, %rd50, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd50, %rd50, %rd6;\n");
    ptx.push_str("    add.u64 %rd50, %rd50, %rd47;\n");
    ptx.push_str("    shl.b64 %rd50, %rd50, 2;\n");
    ptx.push_str("    add.u64 %rd50, %logsumexp_base, %rd50;\n");
    ptx.push_str("    @%p1 st.global.f32 [%rd50], %lse;\n");
}
