//! Phase 0 (prelude): PTX header, param block, register declarations,
//! and thread/block-index computation. See spec §1 for the register
//! budget this phase allocates.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::total_bytes;

/// Emit the PTX file header up through the index-computation block.
///
/// After this returns, the following registers hold useful values:
///   %tid_x     (u32) = threadIdx.x
///   %warp_id   (u32) = tid_x / 32
///   %lane      (u32) = tid_x % 32
///   %bid_x     (u32) = blockIdx.x
///   %bid_y     (u32) = blockIdx.y
///   %q_start   (u64) = bid_x * block_q
///   %head_idx  (u64) = bid_y % heads
///   %batch_idx (u64) = bid_y / heads
///
/// Q/K/V/out pointers loaded into %rd0/%rd1/%rd2/%rd3; batch/heads/
/// seq_len/head_dim into %rd4/%rd5/%rd6/%rd7; logsumexp into
/// %logsumexp_base; %scale holds the softmax scale. Shared-memory base
/// pointer lives in %shmem_base after a `cvta.shared.u64` from the
/// `shmem` byte array declared here.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig) {
    // File header.
    ptx.push_str(".version 8.7\n");
    ptx.push_str(".target sm_75\n");
    ptx.push_str(".address_size 64\n\n");

    // Kernel entry + param block. All 30 params declared even when a
    // variant ignores some -- keeps the 30-arg FFI launch list stable.
    let name = crate::flash_attention_v2::flash_attention_kernel_name_v2(config);
    ptx.push_str(&format!(".visible .entry {} (\n", name));
    let params = [
        (".param .u64", "q_ptr"), (".param .u64", "k_ptr"), (".param .u64", "v_ptr"),
        (".param .u64", "out_ptr"), (".param .f32", "scale"),
        (".param .u64", "batch"), (".param .u64", "heads"), (".param .u64", "seq_len"),
        (".param .u64", "head_dim"), (".param .u64", "block_table_ptr"),
        (".param .u64", "k_pool_ptr"), (".param .u64", "v_pool_ptr"),
        (".param .u64", "block_size"), (".param .u64", "cos_ptr"),
        (".param .u64", "sin_ptr"), (".param .u64", "seq_ids_ptr"),
        (".param .u64", "seq_lens_ptr"), (".param .u64", "dfs_enter_ptr"),
        (".param .u64", "dfs_exit_ptr"), (".param .u64", "num_tree_nodes"),
        (".param .u64", "param_logsumexp"),
        (".param .u64", "csha_x_ptr"), (".param .u64", "csha_norm_weight_ptr"),
        (".param .u64", "csha_wq_ptr"), (".param .u64", "csha_wk_ptr"),
        (".param .u64", "csha_wv_ptr"), (".param .u64", "csha_wo_ptr"),
        (".param .f32", "csha_eps"), (".param .u32", "csha_active_heads"),
        (".param .u32", "csha_d_model"),
    ];
    for (i, (ty, pname)) in params.iter().enumerate() {
        let comma = if i + 1 < params.len() { "," } else { "" };
        ptx.push_str(&format!("    {} {}{}\n", ty, pname, comma));
    }
    ptx.push_str(")\n{\n");

    // Static shared memory -- ASCII-only, no em-dashes.
    ptx.push_str(&format!(
        "    .shared .align 16 .b8 shmem[{}];\n",
        total_bytes(config)
    ));

    // Register declarations. f32 pool must cover the highest-indexed
    // register any phase writes:
    //   %f0..%f31    — scratch (softmax partials, P broadcasts, cvt temps)
    //   %f32..       — Q row slice,  Q_BASE=32, head_dim/32 registers
    //   %f48..       — O_acc slice, O_BASE=48, head_dim/32 registers
    // Pool size N declares %f<N> = %f0..%f(N-1), so N = 48 + head_dim/32
    // to make %f{O_BASE + head_dim/32 - 1} a valid register.
    let f32_pool = 48 + (config.head_dim / 32) as u32;
    ptx.push_str("    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;\n");
    ptx.push_str("    .reg .u64 %rd<64>;\n");
    ptx.push_str(&format!("    .reg .f32 %f<{}>;\n", f32_pool));
    ptx.push_str("    .reg .b16 %h<32>;\n");
    ptx.push_str("    .reg .pred %p<8>;\n");
    ptx.push_str("    .reg .u32 %r<16>;\n");
    ptx.push_str("    .reg .f32 %scale, %log2e, %row_max, %row_sum, %correction;\n");
    ptx.push_str("    .reg .f32 %new_max, %old_max, %shfl_tmp;\n");
    ptx.push_str("    .reg .u64 %q_start, %head_idx, %batch_idx, %k_start, %k_max;\n");
    ptx.push_str("    .reg .u64 %shmem_base, %smem_addr;\n");
    ptx.push_str("    .reg .f32 %log_sum, %lse;\n");
    ptx.push_str("    .reg .u64 %logsumexp_base;\n");
    ptx.push_str("    .reg .pred %p_has_lse;\n");

    // CSHA A.2.3 projection registers (only when fused_projections is set).
    // Named register pools: one accumulator, counter, scratch, and output
    // register per (label ∈ {Q,K,V}) × (slice ∈ 0..slices_per_lane).
    if config.csha.as_ref().map_or(false, |c| c.fused_projections) {
        let slices_per_lane = ((config.head_dim as u32) / 32).max(1);
        for label in ["Q", "K", "V"] {
            for slice in 0..slices_per_lane {
                ptx.push_str(&format!(
                    "    .reg .f32 %f_acc_{}_{}, %f_x_{}_{}, %f_w_{}_{}, %f_red_{}_{};\n",
                    label, slice, label, slice, label, slice, label, slice
                ));
                ptx.push_str(&format!(
                    "    .reg .b16 %h_x_{}_{}, %h_w_{}_{}, %h_out_{}_{};\n",
                    label, slice, label, slice, label, slice
                ));
                ptx.push_str(&format!(
                    "    .reg .u32 %r_indim_{}_{};\n",
                    label, slice
                ));
                ptx.push_str(&format!(
                    "    .reg .pred %p_indim_{}_{};\n",
                    label, slice
                ));
            }
        }
        // Weight-tile load scratch registers (shared across all three tile loads).
        ptx.push_str("    .reg .u64 %rd_wt, %rd_wt_idx, %rd_wt_off, %rd_wt_src, %rd_wt_dst;\n");
        ptx.push_str("    .reg .b16 %h_wt;\n");
        ptx.push_str("    .reg .pred %p_wt;\n");
        // SMEM base registers for Q/K/V output tiles and x_norm input tile.
        ptx.push_str("    .reg .u64 %q_smem_base, %k_smem_base, %v_smem_base;\n");
        ptx.push_str("    .reg .u64 %x_norm_base, %warp_row;\n");
        // SMEM tile pointer registers for weight matrices and inner-loop use.
        ptx.push_str("    .reg .u64 %q_tile, %k_tile, %v_tile;\n");
    }

    // CSHA A.2.4 RoPE epilogue registers (only when rope_q=true and csha is set).
    if config.rope_q && config.csha.is_some() {
        // HBM pointer registers for cos/sin tables.
        ptx.push_str("    .reg .u64 %rd_rope_cos, %rd_rope_sin, %rd_rope_addr;\n");
        ptx.push_str("    .reg .u64 %rd_rope_cs_idx, %rd_rope_x0_off, %rd_rope_x1_off;\n");
        // f32 accumulators for rotation math.
        ptx.push_str("    .reg .f32 %f_rope_cos, %f_rope_sin;\n");
        ptx.push_str("    .reg .f32 %f_rope_x0, %f_rope_x1, %f_rope_y0, %f_rope_y1;\n");
        ptx.push_str("    .reg .f32 %f_rope_neg_x1;\n");
        // f16 scratch for pair loads/stores.
        ptx.push_str("    .reg .b16 %h_rope_pair, %h_rope_y0, %h_rope_y1;\n");
        // u32 loop/index registers.
        ptx.push_str("    .reg .u32 %r_rope_tid, %r_rope_pair_idx;\n");
        ptx.push_str("    .reg .u32 %r_rope_row, %r_rope_dim_pair;\n");
        ptx.push_str("    .reg .u32 %r_rope_cs_off, %r_rope_smem_row_off;\n");
        ptx.push_str("    .reg .u32 %r_rope_x0_col, %r_rope_x0_off, %r_rope_x1_off;\n");
        // Predicate registers for null-guard and loop exit.
        ptx.push_str("    .reg .pred %p_rope_cos_null, %p_rope_sin_null, %p_rope_skip, %p_rope_done;\n");
    }

    ptx.push_str("    cvta.shared.u64 %shmem_base, shmem;\n");
    ptx.push_str("    mov.f32 %log2e, 0f3FB8AA3B;  // 1.4426950408 (log2(e))\n");

    // Load scalar params.
    ptx.push_str("    ld.param.f32 %scale, [scale];\n");
    ptx.push_str("    ld.param.u64 %rd0, [q_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd1, [k_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd2, [v_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd3, [out_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd4, [batch];\n");
    ptx.push_str("    ld.param.u64 %rd5, [heads];\n");
    ptx.push_str("    ld.param.u64 %rd6, [seq_len];\n");
    ptx.push_str("    ld.param.u64 %rd7, [head_dim];\n");
    ptx.push_str("    ld.param.u64 %logsumexp_base, [param_logsumexp];\n");

    // Thread/block indices.
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    shr.u32 %warp_id, %tid_x, 5;       // warp_id = tid_x / 32\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;          // lane = tid_x % 32\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");

    // q_start = bid_x * block_q.
    ptx.push_str("    cvt.u64.u32 %q_start, %bid_x;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %q_start, %q_start, {};   // * block_q\n",
        config.block_q
    ));

    // batch/head routing from bid_y.
    ptx.push_str("    cvt.u64.u32 %rd16, %bid_y;\n");
    ptx.push_str("    rem.u64 %head_idx,  %rd16, %rd5;   // head_idx  = bid_y % heads\n");
    ptx.push_str("    div.u64 %batch_idx, %rd16, %rd5;   // batch_idx = bid_y / heads\n");
}
