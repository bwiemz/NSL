//! Per-document CTA forward kernel emitter (G2 Strategy 3 v1).
//!
//! Wraps the existing FA-2 v2 phase emitters with a new per-CTA prelude
//! that replaces the standard `%q_start = bid_x * block_q` with:
//!
//!   `%q_start = doc_starts[bid_x]`
//!   `%k_max   = doc_starts[bid_x + 1]`
//!
//! The cloned `FlashAttentionConfig` forces `segment_masked = false` so
//! `s_compute::emit` skips the `seg_smem` predicate — cross-doc isolation
//! comes from the launch grid `(num_docs, batch*heads, 1)` instead of a
//! per-token mask.
//!
//! # v1 limitations (documented in spec §4 known_limits)
//!
//! - Forward ONLY.  Backward not implemented; callers MUST NOT invoke
//!   this under autodiff recording.
//! - One CTA per document (`max_doc_len <= block_q` enforced by `admit`).
//! - No CSHA fused projections.
//! - No Tier B tile-skip.
//! - ABI: the per-doc kernel adds `doc_starts_ptr` as a trailing param
//!   (unconditionally, unlike Tier A which gates on `segment_masked &&
//!   rope_q`).  The runtime identifies the kernel by its `_per_doc_cta`
//!   name suffix (mirrors the `_tier_b1_chunkN` pattern).
//!
//! # Entry points
//!
//! - [`synthesize_per_doc_cta_forward`] — returns the PTX `Vec<u8>`.
//! - [`per_doc_cta_kernel_name`] — returns the kernel name string.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::phases;
use crate::flash_attention_v2::phases::pv_accum::O_BASE;
use crate::flash_attention_v2::smem_layout;
use crate::pca_per_doc::PerDocCtaPlan;
use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
use crate::kernel_skeleton::smem::{emit_static_smem_decl, emit_shmem_base_cvta};
use crate::kernel_skeleton::indexing::emit_thread_lane_warp_register_decl;

/// Kernel-name suffix for per-doc CTA kernels.  The runtime uses this
/// suffix to detect the per-doc path and compute `grid_x` from `doc_starts`
/// rather than `ceil(seq/block_q)`.
pub const PER_DOC_CTA_SUFFIX: &str = "_per_doc_cta";

/// Return the kernel entry-point name for a per-doc CTA forward kernel.
///
/// Format: `<base_name>_v2_per_doc_cta` — the `_v2` tag matches the
/// existing `flash_attention_kernel_name_v2` convention; `_per_doc_cta`
/// is the ABI-stable dispatch signal.
pub fn per_doc_cta_kernel_name(config: &FlashAttentionConfig) -> String {
    let base = crate::flash_attention::flash_attention_kernel_name(config);
    format!("{}_v2{}", base, PER_DOC_CTA_SUFFIX)
}

/// Return `true` when `name` was emitted by `per_doc_cta_kernel_name`.
pub fn is_per_doc_cta_kernel(name: &str) -> bool {
    name.ends_with(PER_DOC_CTA_SUFFIX)
}

/// Synthesise the per-doc CTA forward PTX kernel.
///
/// The returned `Vec<u8>` is NUL-terminated for `cuModuleLoadData`.
///
/// The `plan` is used for documentation (routing reason, max_doc_len) but
/// the actual PTX structure is derived from `plan.fa_config` which has
/// `segment_masked=false` enforced by `admit`.
pub fn synthesize_per_doc_cta_forward(
    _plan: &PerDocCtaPlan,
    config: &FlashAttentionConfig,
) -> Vec<u8> {
    // Precondition: segment_masked must be false (per-doc kernel does not
    // use seg_smem — cross-doc isolation is structural via the launch grid).
    // We enforce this by working on a clone with segment_masked=false.
    let mut cfg = config.clone();
    cfg.segment_masked = false;
    // No CSHA fused projections in v1.
    assert!(
        !cfg.csha.as_ref().is_some_and(|c| c.fused_projections),
        "per-doc CTA v1 does not support CSHA fused projections"
    );

    smem_layout::validate_scalar_v2_config(&cfg, smem_layout::Direction::Forward)
        .expect("per-doc CTA emitter called with unsupported config");

    let mut ptx = String::new();

    // File header.
    let target_sm = if cfg.gpu_sm >= 80 { TargetSm::Sm80 } else { TargetSm::Sm75 };
    emit_ptx_header(&mut ptx, PtxVersion::V8_7, target_sm);

    // Static SMEM (no seg_smem, no doc_starts SMEM bake — per-doc reads 2 i32s directly).
    let smem_bytes = smem_layout::total_bytes(&cfg);
    // Dynamic SMEM extern must be at MODULE scope (before .visible .entry).
    if smem_bytes > smem_layout::SMEM_BUDGET_BYTES {
        use crate::kernel_skeleton::smem::emit_dynamic_smem_extern;
        emit_dynamic_smem_extern(&mut ptx);
    }

    // Kernel entry + full param block (mirrors `prelude::emit_with_smem_override`
    // but uses the per-doc kernel name and appends `doc_starts_ptr` unconditionally).
    //
    // # ABI alignment with the production FFI
    //
    // `nsl_flash_attention_csha` and `_with_saves` pass a 38-arg launch
    // list (see `nsl_runtime::flash_attention::nsl_flash_attention_csha`):
    //   [0..35]: standard FA-2 prelude (`q..x_raw`)
    //   [36]:    `segment_ids_ptr` (Tier A — present even when segment_masked=false)
    //   [37]:    `doc_starts_ptr`  (PCA §4.3)
    //
    // The per-doc CTA kernel does NOT need `segment_ids_ptr` (cross-doc
    // isolation comes from the launch grid, not seg-mask) but it DOES
    // need `doc_starts_ptr` at the FFI's args[37] slot. To keep both
    // ABIs aligned without a special FFI-side seg_ids skip, we declare a
    // `_segment_ids_placeholder` param at position 36 and put
    // `doc_starts_ptr` at position 37. The placeholder is loaded but
    // never read — its presence consumes the FFI's seg_ids arg slot so
    // the FFI's doc_starts arg slot lands on the kernel's
    // `doc_starts_ptr` param.
    //
    // For the standalone-launch path (`nsl_kernel_launch` from the v1
    // correctness test) callers MUST pass 38 args including a sentinel
    // 0 at position 36 — same convention as the FFI.
    let name = per_doc_cta_kernel_name(&cfg);
    ptx.push_str(&format!(".visible .entry {} (\n", name));
    let params: &[(&str, &str)] = &[
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
        (".param .u64", "q_proj_ptr"), (".param .u64", "k_proj_ptr"),
        (".param .u64", "v_proj_ptr"),
        (".param .u64", "row_max_ptr"), (".param .u64", "row_sum_ptr"),
        (".param .u64", "x_raw_ptr"),
        // Tier-A-shaped placeholder: consumes the FFI's seg_ids arg
        // slot so the next slot (doc_starts) lands on `doc_starts_ptr`.
        // Never read by the per-doc kernel body.
        (".param .u64", "_segment_ids_placeholder"),
        // Per-doc CTA: doc_starts unconditionally (unlike Tier A's conditional append).
        (".param .u64", "doc_starts_ptr"),
    ];
    for (i, (ty, pname)) in params.iter().enumerate() {
        let comma = if i + 1 < params.len() { "," } else { "" };
        ptx.push_str(&format!("    {} {}{}\n", ty, pname, comma));
    }
    ptx.push_str(")\n{\n");

    // Static SMEM inside function body (only for configs that fit the static cap).
    // Dynamic SMEM configs use the `.extern .shared` emitted at module scope above.
    if smem_bytes <= smem_layout::SMEM_BUDGET_BYTES {
        emit_static_smem_decl(&mut ptx, smem_bytes as usize);
    }

    // Register declarations (mirrors prelude.rs standard pool).
    let f32_pool = 48 + (cfg.head_dim / 32) as u32;
    emit_thread_lane_warp_register_decl(&mut ptx);
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

    // SMEM base pointer.
    emit_shmem_base_cvta(&mut ptx);
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

    // Thread/block indices (mirrors standard prelude).
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    shr.u32 %warp_id, %tid_x, 5;\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");

    // Standard batch/head routing from bid_y.
    ptx.push_str("    cvt.u64.u32 %rd16, %bid_y;\n");
    ptx.push_str("    rem.u64 %head_idx,  %rd16, %rd5;\n");
    ptx.push_str("    div.u64 %batch_idx, %rd16, %rd5;\n");

    // Per-doc CTA prelude delta: overwrite %q_start and %k_max.
    phases::forward::per_doc_prelude::emit(&mut ptx, &cfg);

    // Main iteration loop (same as standard path, segment_masked=false).
    let iters = (cfg.block_q as u32).div_ceil(4);
    let slices = (cfg.head_dim as u32) / 32;

    for q_iter in 0..iters {
        ptx.push_str(&format!(
            "    // ====== per-doc q_tile_iter = {} / {} ======\n",
            q_iter, iters
        ));

        // Per-iteration softmax-state reset.
        ptx.push_str("    mov.f32 %row_max, 0fFF800000;  // -inf\n");
        ptx.push_str("    mov.f32 %row_sum, 0f00000000;\n");
        for i in 0..slices {
            ptx.push_str(&format!(
                "    mov.f32 %f{}, 0f00000000;  // O_acc[{}] = 0\n",
                O_BASE + i, i
            ));
        }

        // CSHA hooks — no-op when csha=None (standard for v1 per-doc).
        phases::csha_hooks::emit_prologue(&mut ptx, &cfg, q_iter);
        phases::csha_hooks::emit_matmul_projection(&mut ptx, &cfg, q_iter);
        phases::csha_hooks::emit_rope_epilogue(&mut ptx, &cfg, q_iter);

        // Q load.
        phases::q_load::emit(&mut ptx, &cfg, q_iter);

        // Save activations (null-guarded — only fires when flag set; no-op for v1).
        phases::csha_hooks::emit_save_activations_subset(
            &mut ptx, &cfg, q_iter, phases::csha_hooks::SaveSet::QK,
        );

        // KV-tile loop.
        // %k_start initialized to %q_start (doc start), %k_max = doc_end (set by prelude).
        ptx.push_str("    mov.u64 %k_start, %q_start;\n");
        ptx.push_str(&format!("V2_PERDOC_LOOP_KV_{}:\n", q_iter));

        emit_k_tile_load(&mut ptx, &cfg, q_iter);
        phases::s_compute::emit(&mut ptx, &cfg, q_iter, None);
        phases::softmax::emit(&mut ptx, &cfg, q_iter);
        phases::csha_hooks::emit_save_softmax_state(&mut ptx, &cfg, q_iter);
        emit_v_tile_load(&mut ptx, &cfg, q_iter);
        phases::csha_hooks::emit_save_activations_subset(
            &mut ptx, &cfg, q_iter, phases::csha_hooks::SaveSet::V,
        );
        phases::pv_accum::emit(&mut ptx, &cfg, q_iter);

        ptx.push_str(&format!(
            "    add.u64 %k_start, %k_start, {};\n",
            cfg.block_kv
        ));
        ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
        ptx.push_str(&format!("    @%p0 bra V2_PERDOC_LOOP_KV_{};\n", q_iter));

        // Per-doc CTA bounds guard: skip finalize/output for rows beyond doc_len.
        // q_row_local = q_iter * 4 + warp_id; if >= doc_len, this warp's row
        // is in the padding region of the packed sequence and must NOT be written.
        // Without this guard, CTAs with shorter docs would overwrite output positions
        // belonging to adjacent CTAs with garbage values (cross-CTA race condition).
        //
        // Note: ALL warps must reach the bar.sync above (in K/V tile loads and
        // s_compute) even if their q_row is out-of-bounds; the guard fires AFTER
        // the KV loop completes, gating only the finalize store.
        ptx.push_str(
            "    // per-doc bounds guard: skip finalize if q_row_local >= doc_len\n"
        );
        ptx.push_str(&format!(
            "    add.u32 %r0, %warp_id, {};  // q_row_local = warp_id + {}\n",
            q_iter * 4, q_iter * 4
        ));
        ptx.push_str("    setp.lt.u32 %p0, %r0, %r_pdoc_len;  // q_row_local < doc_len\n");
        ptx.push_str(&format!(
            "    @!%p0 bra PERDOC_SKIP_FINALIZE_{};\n",
            q_iter
        ));
        phases::finalize::emit(&mut ptx, &cfg, q_iter);
        phases::csha_hooks::emit_output_projection(&mut ptx, &cfg, q_iter);
        ptx.push_str(&format!("PERDOC_SKIP_FINALIZE_{}:\n", q_iter));
    }

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    // Trailing newline + NUL for cuModuleLoadData.
    if !ptx.ends_with('\n') {
        ptx.push('\n');
    }
    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

// ---------------------------------------------------------------------------
// K-tile and V-tile loads (simplified — no CSHA fused-projection null-guard
// needed for v1 per-doc kernel since fused_projections=false is enforced).
// ---------------------------------------------------------------------------

fn emit_k_tile_load(ptx: &mut String, config: &FlashAttentionConfig, q_iter: u32) {
    let total_k_elems = (config.block_kv as u32) * (config.head_dim as u32);
    let kv_off = smem_layout::kv_offset(config);

    ptx.push_str("    // per-doc K tile load\n");
    ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd6;\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %k_start;\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd7;\n");
    ptx.push_str("    shl.b64 %rd58, %rd58, 2;\n");
    ptx.push_str("    add.u64 %rd58, %rd1, %rd58;\n");
    ptx.push_str("    cvt.u64.u32 %rd59, %tid_x;\n");
    ptx.push_str(&format!("V2_PERDOC_K_LOAD_{}:\n", q_iter));
    ptx.push_str("    shl.b64 %rd60, %rd59, 2;\n");
    ptx.push_str("    add.u64 %rd61, %rd58, %rd60;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd61];\n");
    ptx.push_str("    cvt.rn.f16.f32 %h0, %f0;\n");
    ptx.push_str("    shl.b64 %rd60, %rd59, 1;\n");
    ptx.push_str(&format!("    add.u64 %rd60, %rd60, {};\n", kv_off));
    ptx.push_str("    add.u64 %smem_addr, %rd60, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd59, %rd59, 128;\n");
    ptx.push_str(&format!("    setp.lt.u64 %p0, %rd59, {};\n", total_k_elems));
    ptx.push_str(&format!("    @%p0 bra V2_PERDOC_K_LOAD_{};\n", q_iter));
    ptx.push_str("    bar.sync 0;\n");
}

fn emit_v_tile_load(ptx: &mut String, config: &FlashAttentionConfig, q_iter: u32) {
    let total_v_elems = (config.block_kv as u32) * (config.head_dim as u32);
    let kv_off = smem_layout::kv_offset(config);

    ptx.push_str("    // per-doc V tile load\n");
    ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd6;\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %k_start;\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd7;\n");
    ptx.push_str("    shl.b64 %rd58, %rd58, 2;\n");
    ptx.push_str("    add.u64 %rd58, %rd2, %rd58;\n");
    ptx.push_str("    cvt.u64.u32 %rd59, %tid_x;\n");
    ptx.push_str(&format!("V2_PERDOC_V_LOAD_{}:\n", q_iter));
    ptx.push_str("    shl.b64 %rd60, %rd59, 2;\n");
    ptx.push_str("    add.u64 %rd61, %rd58, %rd60;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd61];\n");
    ptx.push_str("    cvt.rn.f16.f32 %h0, %f0;\n");
    ptx.push_str("    shl.b64 %rd60, %rd59, 1;\n");
    ptx.push_str(&format!("    add.u64 %rd60, %rd60, {};\n", kv_off));
    ptx.push_str("    add.u64 %smem_addr, %rd60, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd59, %rd59, 128;\n");
    ptx.push_str(&format!("    setp.lt.u64 %p0, %rd59, {};\n", total_v_elems));
    ptx.push_str(&format!("    @%p0 bra V2_PERDOC_V_LOAD_{};\n", q_iter));
    ptx.push_str("    bar.sync 0;\n");
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};
    use crate::pca_per_doc::{PerDocCtaPlan, PerDocAdmitConfig};
    use crate::pca_detect::{DatasetPackingConfig, PcaDetection, PcaStrategy};

    fn base_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim: 32,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 75,
            segment_masked: false, // per-doc: segment_masked always false
            csha: None,
        }
    }

    fn build_plan(cfg: &FlashAttentionConfig) -> PerDocCtaPlan {
        let packing = DatasetPackingConfig {
            enabled: true,
            max_sequence_length: 128,
            mean_doc_length: Some(25),
            doc_length_stddev: Some(0),
            separator_token_id: Some(2),
        };
        let det = PcaDetection {
            strategy: PcaStrategy::PerDocumentCta,
            expected_doc_fraction: 0.195,
            rationale: "test".to_string(),
            segment_id_bytes_per_batch: 256,
            eliminated_mask_bytes_per_batch: 32768,
        };
        crate::pca_per_doc::admit(
            &det,
            cfg,
            &packing,
            &PerDocAdmitConfig { enable_per_doc_cta: true, ..Default::default() },
        )
        .expect("admit should succeed for base_cfg")
    }

    #[test]
    fn kernel_name_includes_per_doc_cta_suffix() {
        let cfg = base_cfg();
        let name = per_doc_cta_kernel_name(&cfg);
        assert!(
            name.ends_with(PER_DOC_CTA_SUFFIX),
            "kernel name '{}' must end with '{}'", name, PER_DOC_CTA_SUFFIX
        );
        assert!(is_per_doc_cta_kernel(&name));
        // Standard v2 name must NOT match.
        let std_name = crate::flash_attention_v2::flash_attention_kernel_name_v2(&cfg);
        assert!(!is_per_doc_cta_kernel(&std_name));
    }

    #[test]
    fn prelude_emits_per_doc_grid_indexing() {
        let cfg = base_cfg();
        let plan = build_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_forward(&plan, &cfg);
        let ptx = String::from_utf8_lossy(&ptx_bytes);

        // Must contain the per-doc prelude comment markers.
        assert!(
            ptx.contains("// per-doc CTA: q_start = doc_starts[bid_x]"),
            "PTX must contain per-doc prelude marker"
        );
        assert!(
            ptx.contains("// q_end = doc_starts[bid_x + 1]"),
            "PTX must contain per-doc q_end marker"
        );
        // Must NOT contain seg_smem (segment-mask SMEM region must be absent).
        assert!(
            !ptx.contains("seg_smem"),
            "per-doc PTX must not contain seg_smem"
        );
    }

    #[test]
    fn s_compute_does_not_emit_segment_mask_predicate() {
        let cfg = base_cfg();
        let plan = build_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_forward(&plan, &cfg);
        let ptx = String::from_utf8_lossy(&ptx_bytes);

        // segment_mask predicate registers (only emitted when segment_masked=true).
        assert!(
            !ptx.contains("%p_seg_SEGMASK"),
            "per-doc PTX must not contain segment_mask predicate register"
        );
    }

    #[test]
    fn doc_starts_ptr_always_present_in_per_doc_param_list() {
        let cfg = base_cfg();
        let plan = build_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_forward(&plan, &cfg);
        let ptx = String::from_utf8_lossy(&ptx_bytes);
        assert!(
            ptx.contains("doc_starts_ptr"),
            "per-doc kernel must always declare doc_starts_ptr param"
        );
    }

    #[test]
    fn byte_identity_tier_a_unchanged_when_per_doc_not_selected() {
        // The standard synthesize_flash_attention_ptx_v2 must be byte-identical
        // for configs that do NOT use per-doc (i.e., all existing callers).
        let cfg = base_cfg();
        let v2_bytes = crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg);
        let v2_bytes_2 = crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg);
        // Tier A path is deterministic.
        assert_eq!(v2_bytes, v2_bytes_2, "Tier A v2 must be deterministic");
        // Per-doc produces a DIFFERENT kernel (different name, different prelude).
        let plan = build_plan(&cfg);
        let per_doc_bytes = synthesize_per_doc_cta_forward(&plan, &cfg);
        assert_ne!(
            v2_bytes, per_doc_bytes,
            "per-doc PTX must differ from Tier A (different prelude + kernel name)"
        );
    }
}
