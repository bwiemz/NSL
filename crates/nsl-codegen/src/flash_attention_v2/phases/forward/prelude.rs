//! Phase 0 (prelude): PTX header, param block, register declarations,
//! and thread/block-index computation. See spec §1 for the register
//! budget this phase allocates.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{needs_dynamic_smem, total_bytes, SMEM_BUDGET_BYTES};
use crate::pca_segment::{DEFAULT_SMEM_SEGMENT_BUDGET, SegmentResidency};

/// Forward SMEM-routing gate that also accounts for PCA Tier A's
/// `seg_smem` static array (when `config.segment_masked`) and the PCA
/// §4.3 `smem_doc_starts[1028]` array (when `segment_masked && rope_q`).
///
/// Without both overheads, CSHA fused+saves + segment_masked configs can
/// push the combined static SMEM past the 48 KB cap; ptxas accepts but
/// launch fails with CUDA_ERROR_ILLEGAL_ADDRESS at sync time because the
/// main shmem[] is silently over-allocated.  Mirrors
/// `backward_needs_dynamic_smem`.
///
/// `seg_smem` is sized to `pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET`
/// (the upper bound used by `pca_segment::plan_kernel` when deciding
/// `SegmentResidency::Shared`); the FA emitter's allocation must match
/// the planner so cooperative loads never write past the buffer.
///
/// `smem_doc_starts` is `(MAX_NUM_DOCS + 1) * 4` = 1028 bytes; sized
/// by `pca_rope::MAX_NUM_DOCS` so this gate and the emitter stay in sync.
fn fwd_needs_dynamic_smem(config: &FlashAttentionConfig) -> bool {
    let seg_overhead = if config.segment_masked {
        DEFAULT_SMEM_SEGMENT_BUDGET as u32
    } else {
        0
    };
    let rope_overhead = if config.segment_masked && config.rope_q {
        (crate::pca_rope::MAX_NUM_DOCS + 1) * 4
    } else {
        0
    };
    total_bytes(config) + seg_overhead + rope_overhead > SMEM_BUDGET_BYTES
        || needs_dynamic_smem(config)
}
use crate::kernel_skeleton::indexing::emit_thread_lane_warp_register_decl;

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
/// `tier_b` — when `Some((seq_len, residency))`, emits the PCA Tier B
/// range-table preamble after the Tier A segment_ids SMEM load + bar.sync.
/// When `None` (all existing callers), the output is byte-identical to the
/// pre-Tier-B baseline (spec §3.4.6 no-op guarantee).
pub fn emit(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    tier_b: Option<(u32, SegmentResidency)>,
) {
    emit_with_smem_override(ptx, config, total_bytes(config), tier_b);
}

/// Same as `emit`, but the caller supplies the exact byte count for the
/// static `.shared` array. Used by Tier B.1's `synthesize` to declare
/// `shmem[N]` at the Tier-B.1-sized total (q + 4×KV slabs + p_scratch +
/// chunk_staging) rather than the Tier-A baseline returned by
/// `total_bytes`. Without this override the prelude under-declares SMEM
/// and Tier B.1's later `add.u64 %tb1_*_smem_*, %shmem_base, <offset>`
/// arithmetic addresses bytes past the end of `shmem[]`, silently
/// stomping neighboring GPU state at launch (ptxas can't detect this
/// because SMEM offsets are computed dynamically at runtime).
///
/// `tier_b` is forwarded to the PCA Tier B range-table emission. Tier
/// B.1 callers pass `None` here — CSHA Tier B.1 and PCA Tier B are
/// different subsystems (see `synthesize_flash_attention_ptx_v2_with_tier_b`
/// docstring).
pub fn emit_with_smem_override(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    smem_bytes: u32,
    tier_b: Option<(u32, SegmentResidency)>,
) {
    // File header. Target SM is config-aware: Tier B.1 (cp.async +
    // m16n8k16) requires sm_80+; legacy Tier A defaults to sm_75 since
    // `gpu_sm` is 75 for all its test configs. This resolves B1.6
    // deferral #10 (stopgap `String::replace` removed from
    // `tier_b1::synthesize`).
    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    let target_sm = if config.gpu_sm >= 80 {
        TargetSm::Sm80
    } else {
        TargetSm::Sm75
    };
    emit_ptx_header(ptx, PtxVersion::V8_7, target_sm);

    // Dynamic SMEM module-scope declaration (must precede the .visible .entry).
    // In PTX, `.extern .shared` is a module-level directive — it CANNOT appear
    // inside a function body.  For configs that exceed the 48 KB static SMEM cap
    // the declaration moves here; the static `.shared .align 16 .b8 shmem[N]`
    // form (which CAN appear inside a function body) is used for smaller configs.
    if fwd_needs_dynamic_smem(config) {
        use crate::kernel_skeleton::smem::emit_dynamic_smem_extern;
        emit_dynamic_smem_extern(ptx);
    }

    // Kernel entry + param block. All 30 params declared even when a
    // variant ignores some -- keeps the 30-arg FFI launch list stable.
    // PCA Tier A: segment_ids_ptr is appended conditionally at the END
    // when config.segment_masked, keeping the existing 36-param layout
    // byte-stable for segment_masked=false kernels.
    let name = crate::flash_attention_v2::flash_attention_kernel_name_v2(config);
    ptx.push_str(&format!(".visible .entry {} (\n", name));
    let mut params: Vec<(&str, &str)> = vec![
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
        // Tier C: post-RoPE activation save pointers (null when not in use).
        (".param .u64", "q_proj_ptr"), (".param .u64", "k_proj_ptr"),
        (".param .u64", "v_proj_ptr"),
        (".param .u64", "row_max_ptr"), (".param .u64", "row_sum_ptr"),
        // Tier C: pre-RMSNorm raw-x save (null = skip). Forward stages a
        // copy here BEFORE writing x_normed back into csha_x_ptr; backward
        // dRMSNorm reads from this slot to recover the un-normed x its
        // closed-form needs.
        (".param .u64", "x_raw_ptr"),
    ];
    // PCA Tier A: segment_ids pointer only in the signature when
    // segment_masked is true. Kept at the END so the existing layout
    // stays byte-stable for segment_masked=false kernels.
    if config.segment_masked {
        params.push((".param .u64", "segment_ids_ptr"));
    }
    // PCA §4.3 RoPE-reset: doc_starts pointer only when both
    // segment_masked AND rope_q are set. Appended after segment_ids_ptr
    // so segment_masked=true && rope_q=false signatures stay byte-stable.
    if config.segment_masked && config.rope_q {
        params.push((".param .u64", "doc_starts_ptr"));
    }
    // PCA Tier B M3 instrumentation (B1.5-3): per-tile skip-decision HBM
    // buffer pointer. Only declared when:
    //   1. Tier B is being emitted (tier_b is Some + budget admits), AND
    //   2. the `debug_kernel_instrumentation` Cargo feature is enabled.
    // The bench binary's kernel-args list passes this slot unconditionally
    // when tier_b_on; production callers omit the feature and the param
    // disappears from the signature, keeping the byte-identical no-op
    // guarantee.
    let tier_b_admitted = tier_b
        .map(|(seq_len, residency)| {
            crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency)
        })
        .unwrap_or(false);
    if cfg!(feature = "debug_kernel_instrumentation") && tier_b_admitted {
        params.push((".param .u64", "skip_decisions_ptr"));
    }
    for (i, (ty, pname)) in params.iter().enumerate() {
        let comma = if i + 1 < params.len() { "," } else { "" };
        ptx.push_str(&format!("    {} {}{}\n", ty, pname, comma));
    }
    ptx.push_str(")\n{\n");

    // Static shared memory declaration (inside function body).
    // Dynamic SMEM configs use `.extern .shared` at module scope (emitted above,
    // before the .visible .entry).  Static configs declare the array here.
    // For Tier B.1, `smem_bytes` is the caller-supplied Tier-B.1 total (q +
    // 4×KV slabs + p_scratch + chunk_staging) rather than the Tier-A baseline.
    if !fwd_needs_dynamic_smem(config) {
        use crate::kernel_skeleton::smem::emit_static_smem_decl;
        emit_static_smem_decl(ptx, smem_bytes as usize);
    }

    // Register declarations. f32 pool must cover the highest-indexed
    // register any phase writes:
    //   %f0..%f31    — scratch (softmax partials, P broadcasts, cvt temps)
    //   %f32..       — Q row slice,  Q_BASE=32, head_dim/32 registers
    //   %f48..       — O_acc slice, O_BASE=48, head_dim/32 registers
    // Pool size N declares %f<N> = %f0..%f(N-1), so N = 48 + head_dim/32
    // to make %f{O_BASE + head_dim/32 - 1} a valid register.
    let f32_pool = 48 + (config.head_dim / 32) as u32;
    emit_thread_lane_warp_register_decl(ptx);
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
    //
    // EXCEPTION: the three SMEM base registers `%q_smem_base`, `%k_smem_base`,
    // `%v_smem_base` are ALSO required when `save_activations_for_backward=true`
    // — the save-emission path in `emit_save_activations` (csha_hooks.rs) reads
    // from these SMEM tiles to write post-RoPE Q/K/V to HBM for the backward
    // kernel.  Without the wider gate, setting `save_activations_for_backward=true
    // && fused_projections=false` emits `add.u64 %..., %q_smem_base, ...` with
    // no preceding `.reg` declaration → ptxas rejects with `CUDA_ERROR_INVALID_PTX`.
    // The wider declaration is cheap (three u64 registers) and only materialises
    // when the CSHA branch actually needs SMEM-base plumbing.  See
    // `phases/forward/csha_hooks.rs` for the mirrored init guard.
    let needs_qkv_smem_base = config.csha.as_ref().is_some_and(|c| {
        c.fused_projections || c.save_activations_for_backward
    });
    if config.csha.as_ref().is_some_and(|c| c.fused_projections) {
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
        // x_norm_base / warp_row stay gated on fused_projections — only the
        // projection-sweep path initialises them.  SMEM base registers move
        // out below so they can be declared under the wider gate.
        ptx.push_str("    .reg .u64 %x_norm_base, %warp_row;\n");
        // SMEM tile pointer registers for weight matrices and inner-loop use.
        ptx.push_str("    .reg .u64 %q_tile, %k_tile, %v_tile;\n");
        // KV-tile load bypass registers: used by emit_k_tile_load / emit_v_tile_load
        // to skip the HBM load when the projection sweep already filled SMEM.
        ptx.push_str("    .reg .u64 %rd_wk_chk, %rd_wv_chk;\n");
        ptx.push_str("    .reg .pred %p_wk_fused, %p_wv_fused;\n");
    }

    // Q/K/V SMEM base registers — required whenever the fused projection
    // sweep OR the Tier C save-activations path is live.  Declared under
    // the wider gate so `save_activations_for_backward=true &&
    // fused_projections=false` still sees valid register declarations.
    if needs_qkv_smem_base {
        ptx.push_str("    .reg .u64 %q_smem_base, %k_smem_base, %v_smem_base;\n");
    }

    // CSHA Tier C save_activations scratch registers (only when flag is set).
    // Used by emit_save_activations to write post-RoPE Q/K/V to HBM for the
    // fused source-AD backward kernel. `%f_diag` and `%r_save_qlo` are J-A2
    // diagnostic scratch regs used only when `NSL_CSHA_DUMP_SAVE_STATE` is set;
    // `%f_sdx_fmax` / `%f_sdx_nmax` / `%f_sdx_fsum` are J-A3 softmax-internal
    // capture regs written in softmax.rs at the three decisive points
    // (post-butterfly-max, post-online-update, post-butterfly-sum). All are
    // declared unconditionally because ptxas prunes unused virtual regs.
    if config.csha.as_ref().is_some_and(|c| c.save_activations_for_backward) {
        ptx.push_str(
            "    .reg .u64 %rd_save_base, %rd_save_off, %rd_save_elem, %rd_save_smem, %rd_save_wrow, %rd_save_col, %rd_save_colb;\n",
        );
        ptx.push_str("    .reg .u32 %r_save_wrow, %r_save_qlo;\n");
        ptx.push_str("    .reg .f32 %f_diag, %f_sdx_fmax, %f_sdx_nmax, %f_sdx_fsum;\n");
        ptx.push_str("    .reg .b16 %h_save_v;\n");
        ptx.push_str("    .reg .pred %p_save_null, %p_rowmax_null, %p_rowsum_null, %p_skip_rm, %p_skip_rs;\n");
    }

    // CSHA A5 Wo output projection stub registers (only when fused_output_proj is set).
    // Actual Wo @ O computation is delegated to a separate follow-up kernel (spec R2);
    // these registers are used only for the null-check dispatch stub.
    if config.csha.as_ref().is_some_and(|c| c.fused_output_proj) {
        ptx.push_str("    .reg .u64 %rd_wo_ptr;\n");
        ptx.push_str("    .reg .pred %p_wo_null, %p_x_null;\n");
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

    // PCA §4.3 RoPE-reset registers (sites 1-4 + CTA prologue).
    // Gated on segment_masked && rope_q so sentinel-disabled (segment_masked=false)
    // paths stay byte-stable. Registers consumed by:
    //   * emit_doc_starts_smem_load (this prelude, below) — CTA prologue
    //   * Tasks 7/8 (forward Q/K rotation sites) — read smem_doc_starts
    //   * Task 9 (backward dQ/dK sites) — read smem_doc_starts
    if config.segment_masked && config.rope_q {
        ptx.push_str("    // PCA §4.3 RoPE-reset registers (sites 1-4 + CTA prologue)\n");
        ptx.push_str("    .reg .u64 %rd_doc_starts_ptr, %rd_doc_starts_addr;\n");
        // %r_doc_smem_base / %rd_doc_smem_addr: generic-space u64 SMEM base + per-iter
        // store addr (ptxas rejects [symbol + %reg] in shared stores, so we
        // mirror the Tier A seg_smem cvta.shared.u64 pattern).
        ptx.push_str("    .reg .u64 %r_doc_smem_base, %rd_doc_smem_addr;\n");
        ptx.push_str("    .reg .u32 %r_doc_starts_idx, %r_doc_starts_byte_off, %r_doc_starts_stride;\n");
        ptx.push_str("    .reg .u32 %r_batch_idx, %r_row_offset_elems;\n");
        // %r_abs_pos: abs_row = q_start (or kv_start in fused path) + tile-local
        // row, used as the segment_ids[] index during Tasks 7/8 effective_pos
        // computation. Distinct from %r_rope_row (tile-local) so SMEM addressing
        // stays correct after cs_idx reroutes through effective_pos.
        ptx.push_str("    .reg .u32 %r_abs_pos;\n");
        // %rs_doc_seg: u16 scratch for ld.shared.u16 of segment_ids[abs_row].
        // Distinct from segment_mask's %rs_q_SEGMASK / %rs_k_SEGMASK to avoid
        // collisions if both helpers fire in the same kernel.
        ptx.push_str("    .reg .b16 %rs_doc_seg;\n");
        ptx.push_str("    .reg .s32 %r_doc_start, %r_effective_pos_q, %r_effective_pos_k;\n");
        ptx.push_str("    .reg .pred %p_doc_load_done, %p_doc_null;\n");
    }

    // PCA Tier A: segment-mask helper scratch registers + SMEM buffer.
    // Only emitted when segment_masked is set; segment_masked=false kernels
    // remain byte-identical to pre-Task-3B.
    //
    // Named registers are used (not numbered pool slots) to avoid
    // collisions with the existing %rd<64> / %r<16> / %p<8> numbered pools.
    //
    // The .shared seg_smem declaration MUST appear inside the function body
    // (it is a local static, not module-scope .extern .shared). It is
    // kept separate from the main `shmem` array so the Q/K/V tile arithmetic
    // via total_bytes() is unaffected.
    if config.segment_masked {
        // u64 pair: global segment_ids pointer + SMEM generic-space pointer.
        // %seg_base is u64 (full generic address) to avoid the cvt.u32.u64
        // truncation bug on Blackwell (sm_120+): at shared offset 0 the low 32
        // bits of the generic address are NOT zero, so the u32 truncation gives
        // a wrong shared-space address.  Use the full u64 generic address with
        // ld/st.shared instead.
        ptx.push_str("    .reg .u64 %rd_seg_global, %rd_seg_smem, %seg_base;\n");
        // Scratch registers used by segment_mask::emit_segment_mask_predicate.
        // Now u64 (matching seg_base) to avoid mixing address widths.
        ptx.push_str("    .reg .u64 %rd_q_SEGMASK, %rd_k_SEGMASK;\n");
        ptx.push_str("    .reg .b16 %rs_q_SEGMASK, %rs_k_SEGMASK;\n");
        ptx.push_str("    .reg .pred %p_seg_SEGMASK;\n");
        // Cooperative-load scratch for the warp-0 prelude. The load loop
        // iterates over the runtime seq_len; the SMEM buffer below caps
        // the supported seq_len at DEFAULT_SMEM_SEGMENT_BUDGET / 2 entries
        // (the same ceiling pca_segment::plan_kernel uses to pick Shared).
        ptx.push_str("    .reg .u32 %r_pca_i, %r_pca_seq;\n");
        ptx.push_str("    .reg .u64 %rd_pca_off;\n");
        ptx.push_str("    .reg .b16 %rs_pca;\n");
        ptx.push_str("    .reg .pred %p_pca_load, %p_pca_done, %p_seg_null;\n");
        // SMEM buffer sized to pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET so
        // it matches the ceiling pca_segment::plan_kernel uses to pick
        // SegmentResidency::Shared. Kept separate from `shmem` so Q/K/V
        // tile offset arithmetic is unaffected. Align 4 is safe for u16
        // loads (offsets are 2-byte aligned from a 4-byte-aligned base).
        ptx.push_str(&format!(
            "    .shared .align 4 .b8 seg_smem[{}];\n",
            DEFAULT_SMEM_SEGMENT_BUDGET
        ));
    }

    crate::kernel_skeleton::smem::emit_shmem_base_cvta(ptx);
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

    // PCA Tier A: cooperative warp-0 global→shared load of segment_ids.
    // Runs immediately after block-index computation so segment_ids are
    // ready in SMEM before the first KV-tile loop iteration.
    //
    // Design: warp 0 (threads 0-31) loops over segment_ids in strides of 32
    // (warp-width). At seq_len=2048, that is 64 iterations per thread.
    // All other warps skip straight to PCA_LOAD_DONE and wait at bar.sync.
    //
    // Task-3C NOTE: the launch wrapper is expected to pass a pre-indexed
    // pointer (i.e., pointing to the start of this batch sample's row),
    // so the kernel does NOT add a batch_idx offset here. If Task 3C
    // decides to pass a raw [B, S] base pointer instead, add:
    //   mul.lo.u64 %rd_seg_global, batch_idx, seq_len_bytes;
    //   add.u64    %rd_seg_global, %rd_seg_global, base_ptr;
    // before the load loop.
    if config.segment_masked {
        ptx.push_str("\n    // --- PCA Tier A: load segment_ids from global to shared ---\n");
        ptx.push_str("    ld.param.u64 %rd_seg_global, [segment_ids_ptr];\n");
        // PCA Tier A null-guard (spec §4.2): null segment_ids_ptr → write the
        // all-zero sentinel (every position in segment 0 → seg[q]!=seg[k] is
        // uniformly false → no masking). Computed once; predicates the per-
        // iteration ld.global below so we skip the dereference, never load-then-check.
        ptx.push_str("    setp.eq.u64 %p_seg_null, %rd_seg_global, 0;\n");
        // Get SMEM generic-space address of seg_smem.  Keep as u64 — do NOT
        // truncate to u32.  On Blackwell (sm_120+) the low 32 bits of the
        // generic address are NOT the raw shared-space offset when the label
        // sits at shared offset 0, so the old cvt.u32.u64 trick is incorrect.
        ptx.push_str("    cvta.shared.u64 %seg_base, seg_smem;\n");
        // Only warp 0 participates in the load.
        ptx.push_str("    setp.lt.u32 %p_pca_load, %tid_x, 32;\n");
        ptx.push_str("    @!%p_pca_load bra PCA_LOAD_DONE;\n");
        // seq_len is in %rd6 (u64); narrow to u32 for loop arithmetic.
        ptx.push_str("    cvt.u32.u64 %r_pca_seq, %rd6;\n");
        // Starting index = lane ID (threads 0..31 cover first 32 entries).
        ptx.push_str("    mov.u32 %r_pca_i, %tid_x;             // starting index = lane\n");
        ptx.push_str("PCA_LOAD_LOOP:\n");
        ptx.push_str("    setp.ge.u32 %p_pca_done, %r_pca_i, %r_pca_seq;\n");
        ptx.push_str("    @%p_pca_done bra PCA_LOAD_DONE;\n");
        // Global address = rd_seg_global + i * 2  (u16 = 2 bytes)
        ptx.push_str("    cvt.u64.u32 %rd_seg_smem, %r_pca_i;\n");
        ptx.push_str("    shl.b64 %rd_seg_smem, %rd_seg_smem, 1;\n");
        ptx.push_str("    add.u64 %rd_seg_smem, %rd_seg_smem, %rd_seg_global;\n");
        ptx.push_str("    @%p_seg_null bra PCA_SEG_NULL_LD;\n");
        ptx.push_str("    ld.global.u16 %rs_pca, [%rd_seg_smem];\n");
        ptx.push_str("    bra PCA_SEG_LD_DONE;\n");
        ptx.push_str("PCA_SEG_NULL_LD:\n");
        ptx.push_str("    mov.u16 %rs_pca, 0;\n");
        ptx.push_str("PCA_SEG_LD_DONE:\n");
        // Shared address = seg_base (u64 generic) + i * 2
        ptx.push_str("    cvt.u64.u32 %rd_pca_off, %r_pca_i;\n");
        ptx.push_str("    shl.b64 %rd_pca_off, %rd_pca_off, 1;\n");
        ptx.push_str("    add.u64 %rd_pca_off, %rd_pca_off, %seg_base;\n");
        ptx.push_str("    st.shared.u16 [%rd_pca_off], %rs_pca;\n");
        // Advance by warp size (32) for next stride.
        ptx.push_str("    add.u32 %r_pca_i, %r_pca_i, 32;\n");
        ptx.push_str("    bra PCA_LOAD_LOOP;\n");
        ptx.push_str("PCA_LOAD_DONE:\n");
        // Fence: all threads (including warps 1+) see segment_ids before use.
        ptx.push_str("    bar.sync 0;\n");
        ptx.push_str("    // --- end PCA Tier A segment_ids load ---\n");

        // PCA §4.3 RoPE-reset: CTA-prologue load of this row's doc_starts
        // subtable into SMEM. Runs once per CTA, immediately after the
        // segment_ids fence. Gated on rope_q so segment_masked && !rope_q
        // configs stay byte-stable. The emitter includes its own bar.sync
        // 0 so subsequent reads from smem_doc_starts are well-defined.
        if config.rope_q {
            crate::pca_rope::emit_doc_starts_smem_load(ptx);
        }

        // PCA Tier B range-table preamble (only when admitted).
        // Runs after bar.sync 0 so segment_ids SMEM values are visible to all threads.
        // The preamble builds per-tile min/max tables used by emit_skip_predicate.
        if let Some((seq_len, residency)) = tier_b {
            if crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency) {
                let range_table_base =
                    crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
                        config,
                        crate::flash_attention_v2::smem_layout::Direction::Forward,
                    );
                crate::pca_tilerange::emit_range_table_preamble(
                    ptx,
                    config,
                    seq_len,
                    "seg_smem",
                    range_table_base,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
    use crate::flash_attention_v2::smem_layout::{total_bytes, SMEM_BUDGET_BYTES};
    use crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET;
    use crate::pca_rope::MAX_NUM_DOCS;

    fn rope_segment_config() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: true, paged: false,
            rope_q: true,
            rope_style: RopeStyle::Adjacent,
            gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
            segment_masked: true,
            csha: Some(CshaExtras::level2(1e-5, 32)),
        }
    }

    /// `fwd_needs_dynamic_smem` must include the 1028-byte `smem_doc_starts`
    /// overhead when `segment_masked && rope_q`.  The fixture config has
    /// `total_bytes + seg_overhead + rope_overhead > SMEM_BUDGET_BYTES`,
    /// so the forward SMEM path must be dynamic, and the emitted PTX must
    /// contain `.extern .shared` rather than the static `shmem[N]` form.
    #[test]
    fn fwd_smem_budget_counts_rope_doc_starts_overhead() {
        let cfg = rope_segment_config();
        let t = total_bytes(&cfg);
        let seg = DEFAULT_SMEM_SEGMENT_BUDGET as u32;
        let rope = (MAX_NUM_DOCS + 1) * 4;

        // Verify rope_overhead constant matches pca_rope emission (1028 bytes).
        assert_eq!(rope, 1028, "smem_doc_starts must be (MAX_NUM_DOCS+1)*4 = 1028 bytes");

        // For this fixture total exceeds 48 KB once all three overheads are summed.
        assert!(
            t + seg + rope > SMEM_BUDGET_BYTES,
            "total ({t}) + seg ({seg}) + rope ({rope}) must exceed {SMEM_BUDGET_BYTES}; \
             fwd_needs_dynamic_smem must return true"
        );

        // Confirm the emitted PTX uses dynamic SMEM (extern .shared shmem[]).
        let ptx = String::from_utf8(
            crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg)
        ).expect("PTX must be valid UTF-8");
        assert!(
            ptx.contains(".extern .shared .align 16 .b8 shmem[]"),
            "segment_masked+rope_q forward kernel must use extern .shared (dynamic SMEM); \
             got PTX snippet:\n{}", &ptx[..ptx.len().min(400)]
        );
    }

    /// Sentinel-disabled (segment_masked=false) path must NOT have its budget
    /// inflated by rope_overhead — rope_overhead is only charged when both
    /// segment_masked AND rope_q are set.
    #[test]
    fn fwd_smem_budget_skips_rope_overhead_when_segment_masked_false() {
        let mut cfg = rope_segment_config();
        cfg.segment_masked = false;
        // Without segment_masked the config uses small static SMEM; the
        // overhead must not be charged and `fwd_needs_dynamic_smem` must
        // not incorrectly force dynamic SMEM.
        let ptx = String::from_utf8(
            crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg)
        ).expect("PTX must be valid UTF-8");
        assert!(
            !ptx.contains(".extern .shared .align 16 .b8 shmem[]"),
            "segment_masked=false forward kernel must use static shmem (no extern .shared)"
        );
    }
}
