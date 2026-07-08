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
//! - One CTA per document (`max_doc_len <= block_q` enforced by `admit`).
//! - No CSHA fused projections (forward and backward both gate this off).
//! - No Tier B tile-skip.
//! - causal=true required (per-doc launch geometry assumes causal).
//! - default-OFF (`enable_per_doc_cta=false` in `PerDocAdmitConfig`).
//! - ABI: the per-doc kernel adds `doc_starts_ptr` as a trailing param
//!   (unconditionally, unlike Tier A which gates on `segment_masked &&
//!   rope_q`).  The runtime identifies the kernel by its `_per_doc_cta`
//!   name suffix (mirrors the `_tier_b1_chunkN` pattern).
//!
//! # Backward (CFTP v2 follow-on Sprint 5)
//!
//! The backward kernel ([`synthesize_per_doc_cta_backward`]) reuses every
//! standard FA-2 v2 backward phase (`ds_compute`, `dv_accum`, `dqdk_accum`,
//! `finalize`) and only patches the prelude's `%q_start`/`%k_max` to load
//! from `doc_starts[bid_x]` and `doc_starts[bid_x+1]`. Cross-CTA dQ/dK/dV
//! HBM writes are bounded by `%k_max` (= doc_end) via two targeted
//! `setp.lt.u64 ..., %k_max` substitutions in `finalize::emit_store_dq_only`
//! and `emit_store_kv_only`, so out-of-doc rows never touch HBM.
//!
//! Same v1 caveats as forward apply: max_doc_len ≤ block_q, no CSHA fused
//! projections, causal-only, default-OFF.
//!
//! # Entry points
//!
//! - [`synthesize_per_doc_cta_forward`] — returns the PTX `Vec<u8>`.
//! - [`synthesize_per_doc_cta_backward`] — returns the PTX `Vec<u8>`.
//! - [`per_doc_cta_kernel_name`] — forward kernel name string.
//! - [`per_doc_cta_backward_kernel_name`] — backward kernel name string.

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
// Per-doc CTA backward (CFTP v2 follow-on Sprint 5).
//
// Strategy: synthesize the standard FA-2 v2 backward via
// `synthesize_backward(config_with_segment_masked_false)`, then post-process
// to (1) rename the kernel with `_per_doc_cta` suffix, (2) inject the
// FFI-ABI-aligned trailing params `_segment_ids_placeholder` + `doc_starts_ptr`,
// (3) replace the standard `%q_start = bid_x * block_q + q_launch_base` prelude
// with the per-doc `%q_start = doc_starts[bid_x]; %k_max = doc_starts[bid_x+1]`
// load (mirrors `phases::forward::per_doc_prelude`), (4) override the
// orchestrator's `%k_start = 0; %k_max = seq_len` with `%k_start = %q_start;`
// (keeping the prelude-set `%k_max = doc_end`), and (5) bound the dQ/dK/dV
// finalize HBM stores by `%k_max` instead of `%rd6` (seq_len) so out-of-doc
// rows never write to HBM.
//
// All other backward phases (ds_compute, dv_accum, dqdk_accum, csha_hooks,
// finalize) are reused UNCHANGED — the per-doc semantics flow through via
// the patched `%q_start` / `%k_start` / `%k_max` register values plus the
// existing causal mask.
// ---------------------------------------------------------------------------

/// Return the kernel entry-point name for a per-doc CTA backward kernel.
///
/// Format: `<bwd_base_name>_per_doc_cta` — the prefix matches the standard
/// backward `kernel_name` (which itself starts with `flash_attn_backward_v2_`);
/// the `_per_doc_cta` suffix is the ABI-stable dispatch signal recognised by
/// `nsl_flash_attention_csha_backward`'s `csha_is_per_doc_cta_kernel` helper.
pub fn per_doc_cta_backward_kernel_name(config: &FlashAttentionConfig) -> String {
    let mut cfg = config.clone();
    cfg.segment_masked = false;
    let base = crate::flash_attention_v2::phases::backward::prelude::kernel_name(&cfg);
    format!("{}{}", base, PER_DOC_CTA_SUFFIX)
}

/// Synthesise the per-doc CTA backward PTX kernel.
///
/// Returns the PTX as a NUL-terminated `Vec<u8>` suitable for
/// `cuModuleLoadData`. On unsupported configs (validator rejection) returns
/// `Err(message)`.
///
/// # ABI contract
///
/// The emitted kernel's param list extends the standard backward's
/// `segment_masked=false` param list with two trailing params:
///
///   * `_segment_ids_placeholder` (slot 47 in the FFI args[] array) — never
///     read; present only so the FFI's `seg_ids` slot lines up.
///   * `doc_starts_ptr` (slot 48 in the FFI args[] array) — read by the
///     per-doc prelude to derive `%q_start` and `%k_max`.
///
/// The FFI also appends `num_docs_or_zero` as a trailing scalar arg (added
/// by Sprint 5 mirror of Sprint 1's forward FFI change) so the per-doc
/// `grid_x = num_docs` override has a runtime value.
///
/// # Preconditions
///
/// * `config.causal == true` (per-doc is causal-only in v1).
/// * `config.csha.is_some_and(|c| !c.fused_projections)` OR `config.csha.is_none()`
///   (no fused projections in v1).
/// * Validator accepts `Direction::Backward` for the (segment_masked=false)
///   clone of the config.
pub fn synthesize_per_doc_cta_backward(
    _plan: &PerDocCtaPlan,
    config: &FlashAttentionConfig,
) -> Result<Vec<u8>, String> {
    // Force segment_masked=false on the working clone — per-doc never uses
    // seg_smem. Cross-doc isolation is structural via the launch grid.
    let mut cfg = config.clone();
    cfg.segment_masked = false;
    if cfg.csha.as_ref().is_some_and(|c| c.fused_projections) {
        return Err(
            "per-doc CTA v1 backward does not support CSHA fused projections".to_string(),
        );
    }
    if !cfg.causal {
        return Err("per-doc CTA v1 backward requires causal=true".to_string());
    }

    // 1. Synthesise the standard backward PTX (no Tier B).
    let std_ptx = crate::flash_attention_v2::synthesize_backward(&cfg)?;

    // The standard backward emits a NUL terminator at the end — strip it
    // for string-based patching, re-append before return.
    let mut ptx = std_ptx;
    if ptx.ends_with('\0') {
        ptx.pop();
    }

    // 2. Rename the kernel: append `_per_doc_cta` to the `.visible .entry NAME (`
    //    declaration. The standard kernel name is `phases::backward::prelude::kernel_name(&cfg)`.
    let std_name = crate::flash_attention_v2::phases::backward::prelude::kernel_name(&cfg);
    let new_name = format!("{}{}", std_name, PER_DOC_CTA_SUFFIX);
    let old_entry = format!(".visible .entry {} (", std_name);
    let new_entry = format!(".visible .entry {} (", new_name);
    if !ptx.contains(&old_entry) {
        return Err(format!(
            "per-doc backward synth: expected to find `{}` in standard backward PTX",
            old_entry
        ));
    }
    ptx = ptx.replacen(&old_entry, &new_entry, 1);

    // 3. Inject the FFI-ABI-aligned trailing params:
    //    `_segment_ids_placeholder` (slot 47) + `doc_starts_ptr` (slot 48).
    //    The standard segment_masked=false backward's param list ends with
    //    `.param .u64 dv_scratch_ptr` followed by `)`. We find that closing
    //    paren and insert the two extra params before it (with proper commas).
    //
    //    Standard tail looks like:
    //        .param .u64 dv_scratch_ptr
    //    )
    //
    //    We rewrite to:
    //        .param .u64 dv_scratch_ptr,
    //        .param .u64 _segment_ids_placeholder,
    //        .param .u64 doc_starts_ptr
    //    )
    // Under `csha_cycle19_probe` feature, the backward prelude appends
    // two trailing `.param .u64 probe_{ds,dv}_out_ptr` slots — so the
    // standard tail is `probe_dv_out_ptr\n)` instead of `dv_scratch_ptr\n)`.
    // Both match paths keep the segment_ids + doc_starts insertion
    // between the last standard param and the closing paren.
    // Phase 1.1 (pretraining): the standard backward prelude now ends the param
    // list with the three dW f32-scratch pointers (dwq/dwk/dwv_scratch_ptr)
    // right after dk/dv scratch, so the non-probe tail is `dwv_scratch_ptr\n)`
    // (was `dv_scratch_ptr\n)` before the dW-scratch params landed). The probe
    // slots still trail everything, so the probe tail is unchanged.
    #[cfg(not(feature = "csha_cycle19_probe"))]
    let std_tail = "    .param .u64 dwv_scratch_ptr\n)";
    #[cfg(not(feature = "csha_cycle19_probe"))]
    let new_tail = "    .param .u64 dwv_scratch_ptr,\n    .param .u64 _segment_ids_placeholder,\n    .param .u64 doc_starts_ptr\n)";
    #[cfg(feature = "csha_cycle19_probe")]
    let std_tail = "    .param .u64 probe_dv_out_ptr\n)";
    #[cfg(feature = "csha_cycle19_probe")]
    let new_tail = "    .param .u64 probe_dv_out_ptr,\n    .param .u64 _segment_ids_placeholder,\n    .param .u64 doc_starts_ptr\n)";
    if !ptx.contains(std_tail) {
        return Err(format!(
            "per-doc backward synth: expected standard backward param-list tail \
             `{std_tail}` not found"
        ));
    }
    ptx = ptx.replacen(std_tail, new_tail, 1);

    // 4. Replace the standard prelude's `%q_start = bid_x * block_q + q_launch_base`
    //    derivation with the per-doc `%q_start = doc_starts[bid_x]; %k_max = doc_starts[bid_x+1]`
    //    load. The standard prelude emits exactly the four lines below
    //    (see `phases/backward/prelude.rs:383-388` + `mov %k_start, 0;`).
    //
    //    We replace q_start derivation only — `%k_start = 0` and
    //    `%k_max = %rd6` are emitted by the orchestrator (not the prelude)
    //    and are patched in step 5.
    let block_q = cfg.block_q;
    let std_q_start_block = format!(
        "    cvt.u64.u32 %q_start, %bid_x;\n    mul.lo.u64 %q_start, %q_start, {};\n    add.u64 %q_start, %q_start, %q_launch_base;\n",
        block_q,
    );
    let new_q_start_block = String::from(
        "    // ===== per-doc CTA prelude override (Sprint 5 backward) =====\n\
         \x20   // %q_start = doc_starts[bid_x]; %k_max_pdoc = doc_starts[bid_x+1]\n\
         \x20   .reg .u64 %rd_pdoc_base, %rd_pdoc_addr;\n\
         \x20   .reg .s32 %r_pdoc_start, %r_pdoc_end;\n\
         \x20   .reg .u32 %r_pdoc_len;\n\
         \x20   .reg .u64 %k_max_pdoc;\n\
         \x20   ld.param.u64 %rd_pdoc_base, [doc_starts_ptr];\n\
         \x20   cvt.u64.u32 %rd_pdoc_addr, %bid_x;\n\
         \x20   shl.b64 %rd_pdoc_addr, %rd_pdoc_addr, 2;  // bid_x * 4\n\
         \x20   add.u64 %rd_pdoc_addr, %rd_pdoc_base, %rd_pdoc_addr;\n\
         \x20   ld.global.s32 %r_pdoc_start, [%rd_pdoc_addr];\n\
         \x20   add.u64 %rd_pdoc_addr, %rd_pdoc_addr, 4;\n\
         \x20   ld.global.s32 %r_pdoc_end, [%rd_pdoc_addr];\n\
         \x20   sub.s32 %r_pdoc_len, %r_pdoc_end, %r_pdoc_start;\n\
         \x20   cvt.u64.s32 %q_start, %r_pdoc_start;  // q_start = doc_start\n\
         \x20   cvt.u64.s32 %k_max_pdoc, %r_pdoc_end; // doc_end (saved for k_max patch)\n\
         \x20   // ===== end per-doc prelude override =====\n",
    );
    if !ptx.contains(&std_q_start_block) {
        return Err(format!(
            "per-doc backward synth: standard q_start derivation block not found \
             (expected:\n{})",
            std_q_start_block
        ));
    }
    ptx = ptx.replacen(&std_q_start_block, &new_q_start_block, 1);

    // 5. Replace ALL occurrences of `mov.u64 %k_start, 0; mov.u64 %k_max, %rd6;`
    //    with `mov.u64 %k_start, %q_start; mov.u64 %k_max, %k_max_pdoc;`.
    //
    //    There are TWO occurrences in the standard backward:
    //      a. `prelude.rs:396-397` — a defensive init so kv_load doesn't fault
    //         if SMEM has garbage (k_start=0 forces k_global=lane indexing).
    //      b. `mod.rs:893-894` — the operational init immediately before
    //         the `V2_BWD_LOOP_KV:` label that drives the kv-loop.
    //
    //    Both should be replaced — per-doc kernel needs `k_start = doc_start`
    //    and `k_max = doc_end` consistently across the kernel body.
    let std_kstart_block = "    mov.u64 %k_start, 0;\n    mov.u64 %k_max, %rd6;\n";
    let new_kstart_block =
        "    mov.u64 %k_start, %q_start;  // per-doc: sweep starts at doc_start\n    mov.u64 %k_max, %k_max_pdoc;  // per-doc: doc_end\n";
    if !ptx.contains(std_kstart_block) {
        return Err(
            "per-doc backward synth: k_start/k_max init block not found"
                .to_string(),
        );
    }
    ptx = ptx.replace(std_kstart_block, new_kstart_block);

    // 6. Bound dQ/dK/dV HBM stores by `%k_max` instead of `%rd6` (seq_len).
    //    The standard backward finalize bounds the per-cell row index against
    //    `%rd6` (seq_len), allowing writes to rows up to seq_len. For per-doc,
    //    we want writes bounded by doc_end (= %k_max). Two textually-identical
    //    `setp.lt.u64 %p0, %rd43, %rd6;` instances exist in `phases/backward/
    //    finalize.rs` — one in `emit_store_dq_only`, one in `emit_store_kv_only`.
    //    Replace ALL with the doc-end bound.
    let std_bound = "    setp.lt.u64 %p0, %rd43, %rd6;\n";
    let new_bound = "    setp.lt.u64 %p0, %rd43, %k_max;  // per-doc: bound by doc_end\n";
    if !ptx.contains(std_bound) {
        return Err(
            "per-doc backward synth: finalize row-bound predicate not found".to_string(),
        );
    }
    ptx = ptx.replace(std_bound, new_bound);

    // 7. Bound d_correction + ds_compute padding-row skip by `%k_max` instead
    //    of `%rd6`. Both emit `setp.ge.u64 %p0, %rd39, %rd6;` at the top of
    //    each per-warp emission to zero/skip padding rows (q_row_global >=
    //    seq_len). For per-doc, padding rows have q_row_global ∈ [doc_end,
    //    seq_len) — these MUST be skipped to prevent garbage row_max/row_sum
    //    reads from inflating P and propagating into dV/dK/dQ. Replace all
    //    32+ occurrences (one per q_tile_iter * each emit site) with the
    //    doc-end bound.
    let std_padding_bound = "    setp.ge.u64 %p0, %rd39, %rd6;\n";
    let new_padding_bound = "    setp.ge.u64 %p0, %rd39, %k_max;  // per-doc: skip padding past doc_end\n";
    if !ptx.contains(std_padding_bound) {
        return Err(
            "per-doc backward synth: d_correction/ds_compute padding-row \
             skip predicate not found"
                .to_string(),
        );
    }
    ptx = ptx.replace(std_padding_bound, new_padding_bound);

    // 8. Bound ds_compute's k-column mask by `%k_max` instead of `%rd6`.
    //    `ds_compute::emit` line 207 emits `setp.ge.u64 %p1, %rd42, %rd6;`
    //    where %rd42 = lane + k_start = k_global. This masks k-columns past
    //    seq_len to -INF in the recomputed P. For per-doc, k-columns in
    //    [doc_end, seq_len) are cross-doc and MUST also be masked.
    //    Replacing the bound forces P=0 for those k positions so dV[k] and
    //    dK[k] for cross-doc k receive zero contribution.
    let std_k_bound = "    setp.ge.u64 %p1, %rd42, %rd6;\n";
    let new_k_bound = "    setp.ge.u64 %p1, %rd42, %k_max;  // per-doc: mask k-columns past doc_end\n";
    if !ptx.contains(std_k_bound) {
        return Err(
            "per-doc backward synth: ds_compute k-column mask predicate not found"
                .to_string(),
        );
    }
    ptx = ptx.replace(std_k_bound, new_k_bound);

    // 9. Bound q_load HBM read by `%k_max` instead of `%rd6` (seq_len).
    //    `phases/backward/q_load.rs:59` emits a per-q_tile_iter guard
    //    `setp.ge.u64 %p1, %rd34, %rd6` where %rd34 = q_row_global. Standard
    //    backward skips rows past seq_len. For per-doc, the guard MUST also
    //    skip rows in `[doc_end, doc_start + block_q)` — without this, when
    //    `doc_start + block_q > seq_len` (block_q-aligned doc_start within
    //    `seq_len - block_q .. seq_len`), the load would read past the
    //    q_proj buffer end and fault with `CUDA_ERROR_ILLEGAL_ADDRESS`. The
    //    downstream d_correction/finalize guards already produce zero
    //    contributions for these polluted SMEM cells, but reading past HBM
    //    is unsafe. Replace all occurrences.
    let std_qload_bound =
        "    setp.ge.u64 %p1, %rd34, %rd6;         // q_row_global >= seq_len?\n";
    let new_qload_bound = "    setp.ge.u64 %p1, %rd34, %k_max;         // per-doc: skip past doc_end\n";
    if !ptx.contains(std_qload_bound) {
        return Err(
            "per-doc backward synth: q_load HBM bound predicate not found".to_string(),
        );
    }
    ptx = ptx.replace(std_qload_bound, new_qload_bound);

    // 10. Bound kv_load HBM read by `%k_max` instead of `%rd6` (seq_len).
    //     `phases/backward/kv_load.rs:54` emits `setp.ge.u64 %p1, %rd35, %rd6`
    //     where %rd35 = k_row_global. Same rationale as step 9 — per-doc K/V
    //     rows past doc_end must be skipped to prevent HBM faults on configs
    //     where `k_start + block_kv > seq_len`. Replace all occurrences.
    let std_kvload_bound = "    setp.ge.u64 %p1, %rd35, %rd6;\n";
    let new_kvload_bound = "    setp.ge.u64 %p1, %rd35, %k_max;  // per-doc: skip past doc_end\n";
    if !ptx.contains(std_kvload_bound) {
        return Err(
            "per-doc backward synth: kv_load HBM bound predicate not found".to_string(),
        );
    }
    ptx = ptx.replace(std_kvload_bound, new_kvload_bound);

    // 11. NUL-terminate for cuModuleLoadData.
    if !ptx.ends_with('\n') {
        ptx.push('\n');
    }
    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    Ok(bytes)
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
            num_sink_tokens: 0,
            gpu_sm: 75,
            segment_masked: false, // per-doc: segment_masked always false
            csha: None,
            checkpoint: None,
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

    // -------------------------------------------------------------------
    // Backward synthesis tests (CFTP v2 follow-on Sprint 5).
    // -------------------------------------------------------------------

    /// Backward-friendly base config (block_q/block_kv/head_dim = 32,
    /// segment_masked=false, csha=None, causal=true).
    fn bwd_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: true, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75,
            segment_masked: false, csha: None,
            checkpoint: None,
        }
    }

    fn build_bwd_plan(cfg: &FlashAttentionConfig) -> PerDocCtaPlan {
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
            &det, cfg, &packing,
            &PerDocAdmitConfig { enable_per_doc_cta: true, ..Default::default() },
        )
        .expect("admit should succeed for bwd_cfg")
    }

    #[test]
    fn backward_kernel_name_includes_per_doc_cta_suffix() {
        let cfg = bwd_cfg();
        let name = per_doc_cta_backward_kernel_name(&cfg);
        assert!(
            name.ends_with(PER_DOC_CTA_SUFFIX),
            "backward kernel name '{}' must end with '{}'", name, PER_DOC_CTA_SUFFIX
        );
        // Sprint 1 FFI dispatcher recognises the same suffix on either
        // forward or backward, so `is_per_doc_cta_kernel` should match.
        assert!(is_per_doc_cta_kernel(&name));
        // Distinct from forward kernel name.
        let fwd = per_doc_cta_kernel_name(&cfg);
        assert_ne!(name, fwd, "backward and forward names must differ");
        // Backward name carries the `backward_` prefix marker.
        assert!(
            name.contains("backward"),
            "backward kernel name must contain `backward` marker, got '{}'", name,
        );
    }

    #[test]
    fn backward_synthesis_emits_per_doc_prelude_override() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);

        assert!(
            ptx.contains("per-doc CTA prelude override (Sprint 5 backward)"),
            "backward PTX must contain per-doc prelude marker"
        );
        assert!(
            ptx.contains("ld.param.u64 %rd_pdoc_base, [doc_starts_ptr];"),
            "backward PTX must load doc_starts_ptr in prelude override"
        );
        assert!(
            ptx.contains("ld.global.s32 %r_pdoc_start, [%rd_pdoc_addr];"),
            "backward PTX must load doc_starts[bid_x] entries"
        );
    }

    #[test]
    fn backward_synthesis_replaces_k_start_init() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);

        // The standard `mov.u64 %k_start, 0;` is replaced by `... %k_start, %q_start;`.
        assert!(
            !ptx.contains("    mov.u64 %k_start, 0;\n    mov.u64 %k_max, %rd6;\n"),
            "standard k_start=0 / k_max=seq_len block must be replaced"
        );
        assert!(
            ptx.contains("mov.u64 %k_start, %q_start;"),
            "backward PTX must initialise k_start from q_start"
        );
        assert!(
            ptx.contains("mov.u64 %k_max, %k_max_pdoc;"),
            "backward PTX must initialise k_max from doc_end"
        );
    }

    #[test]
    fn backward_synthesis_bounds_d_correction_padding_skip_by_k_max() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);
        // d_correction / ds_compute padding-row skip MUST be bounded by
        // %k_max (doc_end), not %rd6 (seq_len), so padding q-rows don't
        // inflate dV/dK via garbage row_max/row_sum reads.
        assert!(
            !ptx.contains("setp.ge.u64 %p0, %rd39, %rd6;"),
            "all d_correction/ds_compute padding-skip predicates must be patched"
        );
        assert!(
            ptx.contains("setp.ge.u64 %p0, %rd39, %k_max;"),
            "padding-skip predicate must use %k_max bound"
        );
        assert!(
            ptx.contains("per-doc: skip padding past doc_end"),
            "padding-skip replacement must carry the per-doc annotation"
        );
    }

    #[test]
    fn backward_synthesis_bounds_finalize_by_k_max() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);

        // The textually-identical `setp.lt.u64 %p0, %rd43, %rd6;` must NOT
        // appear (we replaced all occurrences with `..., %k_max;`).
        assert!(
            !ptx.contains("setp.lt.u64 %p0, %rd43, %rd6;"),
            "standard seq_len bound on finalize row index must be replaced"
        );
        assert!(
            ptx.contains("setp.lt.u64 %p0, %rd43, %k_max;"),
            "backward PTX must bound finalize stores by k_max (doc_end)"
        );
        // The per-doc marker comment must annotate the replacement.
        assert!(
            ptx.contains("per-doc: bound by doc_end"),
            "k_max-bound replacement must carry the per-doc annotation"
        );
    }

    #[test]
    fn backward_synthesis_appends_doc_starts_param() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);

        assert!(
            ptx.contains(".param .u64 _segment_ids_placeholder"),
            "backward PTX must declare seg_ids placeholder param (FFI ABI alignment)"
        );
        assert!(
            ptx.contains(".param .u64 doc_starts_ptr"),
            "backward PTX must declare doc_starts_ptr param"
        );
    }

    #[test]
    fn backward_synthesis_kernel_name_in_entry_decl() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);

        let expected_name = per_doc_cta_backward_kernel_name(&cfg);
        let needle = format!(".visible .entry {} (", expected_name);
        assert!(
            ptx.contains(&needle),
            "backward PTX must declare entry with the per-doc backward kernel name, got missing '{}'", needle,
        );
    }

    #[test]
    fn backward_synthesis_rejects_fused_projections() {
        let mut cfg = bwd_cfg();
        cfg.csha = Some(crate::flash_attention::CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..crate::flash_attention::CshaExtras::default()
        });
        let plan = build_bwd_plan(&bwd_cfg());
        let result = synthesize_per_doc_cta_backward(&plan, &cfg);
        assert!(result.is_err(), "fused_projections must be rejected in v1");
        let err = result.unwrap_err();
        assert!(
            err.contains("fused projections"),
            "rejection message must explain the reason, got '{}'", err,
        );
    }

    #[test]
    fn backward_synthesis_rejects_non_causal() {
        let mut cfg = bwd_cfg();
        cfg.causal = false;
        let plan = build_bwd_plan(&bwd_cfg());
        let result = synthesize_per_doc_cta_backward(&plan, &cfg);
        assert!(result.is_err(), "non-causal must be rejected in v1");
        let err = result.unwrap_err();
        assert!(err.contains("causal"), "rejection message must mention causal, got '{}'", err);
    }

    #[test]
    fn backward_synthesis_nul_terminated() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        assert_eq!(
            ptx_bytes.last(),
            Some(&0u8),
            "backward PTX must be NUL-terminated for cuModuleLoadData"
        );
    }

    #[test]
    fn backward_synthesis_no_seg_smem() {
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);
        // segment_masked=false is enforced on the working clone, so the
        // backward must not emit any seg_smem load/predicate code.
        assert!(
            !ptx.contains("BW_PCA_LOAD_LOOP"),
            "per-doc backward must not emit segment_ids SMEM load loop"
        );
        assert!(
            !ptx.contains("%p_seg_SEGMASK"),
            "per-doc backward must not declare seg_mask predicate"
        );
    }

    #[test]
    fn backward_synthesis_param_list_byte_aligned_with_forward_ffi() {
        // The runtime FFI's args[47] holds seg_ids, args[48] holds doc_starts
        // (mirrors forward's _per_doc_cta ABI). Therefore the per-doc backward's
        // param list MUST end with `_segment_ids_placeholder` then `doc_starts_ptr`
        // in that order, with `doc_starts_ptr` immediately before the closing `)`.
        let cfg = bwd_cfg();
        let plan = build_bwd_plan(&cfg);
        let ptx_bytes = synthesize_per_doc_cta_backward(&plan, &cfg)
            .expect("backward synth must succeed");
        let ptx = String::from_utf8_lossy(&ptx_bytes);
        // Find the ordering: _segment_ids_placeholder appears before doc_starts_ptr.
        let pos_placeholder = ptx
            .find("_segment_ids_placeholder")
            .expect("placeholder param missing");
        let pos_doc_starts = ptx
            .find("doc_starts_ptr")
            .expect("doc_starts_ptr param missing");
        assert!(
            pos_placeholder < pos_doc_starts,
            "param order must be _segment_ids_placeholder THEN doc_starts_ptr"
        );
    }
}
