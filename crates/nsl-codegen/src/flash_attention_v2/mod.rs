//! FlashAttention-2 scalar-path emitter v2.
//!
//! Replaces the structurally incorrect v1 scalar forward path with a
//! warp-per-row thread-mapping contract. See
//! `docs/superpowers/specs/2026-04-14-fa-scalar-emitter-rewrite-design.md`
//! for the phase-level algorithm and constraints.
//!
//! Routed via `flash_attention_selector::select_emitter` when
//! `NSL_FA_EMITTER=v2` and `gpu_sm < 80`. The MMA path (sm>=80) stays on
//! v1 until a separate spec covers MMA correctness.

pub mod smem_layout;
pub mod register_budget;
pub mod phases;

use crate::flash_attention::FlashAttentionConfig;
use phases::pv_accum::O_BASE;

/// v2 entry point. Returns a byte vector ending with a single trailing
/// newline followed by a NUL terminator so `cuModuleLoadData` accepts it.
pub fn synthesize_flash_attention_ptx_v2(config: &FlashAttentionConfig) -> Vec<u8> {
    smem_layout::validate_scalar_v2_config(config, smem_layout::Direction::Forward)
        .expect("v2 emitter called with unsupported config -- selector must gate this");

    let mut ptx = String::new();

    // Phase 0: file header, param block, register decls, indices.
    phases::prelude::emit(&mut ptx, config);

    // CSHA A.4: head pruning guard (runs ONCE, before any q_tile work).
    phases::csha_hooks::emit_active_heads_guard(&mut ptx, config);

    // Outer q_tile_iter loop: iterates ceil(block_q / 4) times. Each
    // iteration processes 4 rows (one per warp).
    let iters = (config.block_q as u32).div_ceil(4);
    let slices = (config.head_dim as u32) / 32;
    let fused_proj = config.csha.as_ref().map_or(false, |c| c.fused_projections);

    // ── CSHA K/V fused-projection pre-passes ──────────────────────────────
    //
    // When fused_projections=true the per-warp-row K and V sweeps each write
    // only 4 rows per q_tile_iter (one per warp × 4 warps = 4 rows).  The full
    // K tile (block_kv rows) is only complete after ALL q_tile_iters have run
    // their K sweeps.  S-compute and PV-accum need the FULL K/V tiles.
    //
    // Solution: run K pre-pass (all q_tile_iters) before the main attention
    // loop to populate K at kv_offset.  For V: run V pre-pass (all q_tile_iters)
    // BETWEEN the last S-compute and the first PV-accum, then do a PV-only loop.
    //
    // Softmax state (%row_max, %row_sum, P in SP-SMEM) is saved to a dedicated
    // SMEM save area between the S-pass and PV-pass (2 f32 per q_iter × iters).
    // The V pre-pass then overwrites K at kv_offset without corrupting S-state.
    //
    // When fused_projections=false: standard interleaved S+PV loop runs.
    if fused_proj {
        let base = smem_layout::sp_offset(config) + smem_layout::sp_bytes(config);
        let wt_bytes = smem_layout::wq_tile_bytes(config);

        // ── Step 1: RMSNorm pre-pass — normalize all x rows in-place before any
        //   projection.  Both K pre-pass and Q sweep in S-compute read normalized x.
        //   Without this step, K pre-pass would read raw (un-normalized) x.
        ptx.push_str("    // CSHA RMSNorm pre-pass: normalize all x rows before projection\n");
        for q_iter in 0..iters {
            phases::csha_hooks::emit_prologue(&mut ptx, config, q_iter);
        }

        // ── Step 2: Load weight tiles once (all three: Wq/Wk/Wv) cooperatively.
        ptx.push_str("    // CSHA K/V pre-pass: load weight tiles (Wq/Wk/Wv) once\n");
        emit_weight_tile_load(&mut ptx, config, "Wq", "csha_wq_ptr", base, 0);
        emit_weight_tile_load(&mut ptx, config, "Wk", "csha_wk_ptr", base + wt_bytes, 0);
        emit_weight_tile_load(&mut ptx, config, "Wv", "csha_wv_ptr", base + 2 * wt_bytes, 0);

        // ── Step 3: K pre-pass — all q_tile_iters write K rows to kv_offset.
        //   Reads normalized x (written by RMSNorm pre-pass above).
        ptx.push_str("    // CSHA K pre-pass: populate full K SMEM tile\n");
        for q_iter in 0..iters {
            phases::csha_hooks::emit_k_prepass_sweep(&mut ptx, config, q_iter);
        }
        ptx.push_str("    bar.sync 0; // K tile complete; safe for all S-computes\n");

        // ── Step 4: S-compute pass — Q sweep + RoPE + Q-load + S-compute + softmax.
        // RMSNorm already done above; do NOT call emit_prologue again here.
        // Row_max/row_sum saved to SMEM after each iter so the V pre-pass can
        // overwrite kv_offset without losing softmax state.
        ptx.push_str("    // CSHA S-compute pass\n");
        for q_iter in 0..iters {
            ptx.push_str(&format!(
                "    // ====== q_tile_iter = {} / {} (S-pass) ======\n",
                q_iter, iters
            ));
            ptx.push_str("    mov.f32 %row_max, 0fFF800000;\n");
            ptx.push_str("    mov.f32 %row_sum, 0f00000000;\n");
            for i in 0..slices {
                ptx.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", O_BASE + i));
            }

            // Q sweep only — K was done in the K pre-pass; RMSNorm done above.
            phases::csha_hooks::emit_q_projection_only(&mut ptx, config, q_iter);

            // RoPE epilogue.
            phases::csha_hooks::emit_rope_epilogue(&mut ptx, config, q_iter);

            // Tier C: save post-RoPE activations for backward (gated on flag).
            phases::csha_hooks::emit_save_activations(&mut ptx, config, q_iter);

            // Q load (q_smem → registers).
            phases::q_load::emit(&mut ptx, config, q_iter);

            // KV loop: K tile load null-guarded (skip when wk≠null; K in SMEM).
            ptx.push_str("    mov.u64 %k_start, 0;\n");
            ptx.push_str("    mov.u64 %k_max, %rd6;\n");
            ptx.push_str(&format!("V2_LOOP_KV_S_{}:\n", q_iter));
            emit_k_tile_load(&mut ptx, config, q_iter);
            phases::s_compute::emit(&mut ptx, config, q_iter);
            phases::softmax::emit(&mut ptx, config, q_iter);
            ptx.push_str(&format!("    add.u64 %k_start, %k_start, {};\n", config.block_kv));
            ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
            ptx.push_str(&format!("    @%p0 bra V2_LOOP_KV_S_{};\n", q_iter));

            // Save row_max and row_sum to SMEM so V pre-pass can run without
            // overwriting these registers.
            phases::csha_hooks::emit_save_softmax_state(&mut ptx, config, q_iter);
        }

        // V pre-pass: all q_tile_iters write V rows to kv_offset.
        // All S-computes are done; K at kv_offset is no longer needed.
        ptx.push_str("    bar.sync 0; // S-pass done; V pre-pass overwrites K SMEM\n");
        ptx.push_str("    // CSHA V pre-pass: populate full V SMEM tile\n");
        for q_iter in 0..iters {
            phases::csha_hooks::emit_v_prepass_sweep(&mut ptx, config, q_iter);
        }
        ptx.push_str("    bar.sync 0; // V tile complete; safe for all PV-accums\n");

        // PV-accum pass: restore softmax state + PV-accum + finalize per iter.
        ptx.push_str("    // CSHA PV-accum pass\n");
        for q_iter in 0..iters {
            ptx.push_str(&format!(
                "    // ====== q_tile_iter = {} / {} (PV-pass) ======\n",
                q_iter, iters
            ));

            // Restore row_max and row_sum saved during S-pass.
            // O_acc is initialized to 0 here (it was never updated during S-pass).
            for i in 0..slices {
                ptx.push_str(&format!("    mov.f32 %f{}, 0f00000000;\n", O_BASE + i));
            }
            phases::csha_hooks::emit_restore_softmax_state(&mut ptx, config, q_iter);

            // KV loop: V tile load null-guarded (skip when wv≠null; V in SMEM).
            ptx.push_str("    mov.u64 %k_start, 0;\n");
            ptx.push_str("    mov.u64 %k_max, %rd6;\n");
            ptx.push_str(&format!("V2_LOOP_KV_PV_{}:\n", q_iter));
            emit_v_tile_load(&mut ptx, config, q_iter);
            phases::pv_accum::emit(&mut ptx, config, q_iter);
            ptx.push_str(&format!("    add.u64 %k_start, %k_start, {};\n", config.block_kv));
            ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
            ptx.push_str(&format!("    @%p0 bra V2_LOOP_KV_PV_{};\n", q_iter));

            phases::finalize::emit(&mut ptx, config, q_iter);
            phases::csha_hooks::emit_output_projection(&mut ptx, config, q_iter);
        }
    } else {
        // ── Standard path (no fused projections) ──────────────────────────────
        for q_iter in 0..iters {
            ptx.push_str(&format!(
                "    // ====== q_tile_iter = {} / {} ======\n",
                q_iter, iters
            ));

            // Per-iteration softmax-state reset.
            ptx.push_str("    mov.f32 %row_max, 0fFF800000;              // -inf\n");
            ptx.push_str("    mov.f32 %row_sum, 0f00000000;\n");
            for i in 0..slices {
                ptx.push_str(&format!(
                    "    mov.f32 %f{}, 0f00000000;                  // O_acc[{}] = 0\n",
                    O_BASE + i,
                    i
                ));
            }

            // CSHA hooks (no-op when csha=None).
            phases::csha_hooks::emit_prologue(&mut ptx, config, q_iter);

            phases::csha_hooks::emit_matmul_projection(&mut ptx, config, q_iter);

            // CSHA A.2.4 RoPE epilogue: runs immediately after projection so
            // Q/K SMEM tiles are rotated BEFORE Q-load / S-compute consume them
            // for QK^T.
            phases::csha_hooks::emit_rope_epilogue(&mut ptx, config, q_iter);

            // Tier C: save post-RoPE activations for backward (gated on flag).
            phases::csha_hooks::emit_save_activations(&mut ptx, config, q_iter);

            // Phase 1: Q load.
            phases::q_load::emit(&mut ptx, config, q_iter);

            // K/V-tile loop.
            ptx.push_str("    mov.u64 %k_start, 0;\n");
            ptx.push_str("    mov.u64 %k_max, %rd6;                        // seq_len\n");
            ptx.push_str(&format!("V2_LOOP_KV_START_{}:\n", q_iter));

            emit_k_tile_load(&mut ptx, config, q_iter);
            phases::s_compute::emit(&mut ptx, config, q_iter);
            phases::softmax::emit(&mut ptx, config, q_iter);
            emit_v_tile_load(&mut ptx, config, q_iter);
            phases::pv_accum::emit(&mut ptx, config, q_iter);

            ptx.push_str(&format!(
                "    add.u64 %k_start, %k_start, {};\n",
                config.block_kv
            ));
            ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
            ptx.push_str(&format!("    @%p0 bra V2_LOOP_KV_START_{};\n", q_iter));

            phases::finalize::emit(&mut ptx, config, q_iter);
            phases::csha_hooks::emit_output_projection(&mut ptx, config, q_iter);
        }
    }

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    // Ensure trailing newline + single NUL for cuModuleLoadData.
    if !ptx.ends_with('\n') {
        ptx.push('\n');
    }
    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

/// Cooperative K-tile load. 128 threads load block_kv*head_dim f32
/// values from k_ptr and cvt-store them as f16 into shmem at kv_offset.
///
/// When `csha.fused_projections` is true and `csha_wk_ptr` is non-null the
/// projection sweep in `emit_matmul_projection` has already written the
/// projected K into SMEM at kv_offset — skip the HBM load to avoid
/// overwriting the fused result.  When `csha_wk_ptr` is null (caller
/// pre-projected K/V via classic k_ptr) the load runs normally.
fn emit_k_tile_load(ptx: &mut String, config: &FlashAttentionConfig, q_iter: u32) {
    let total_k_elems = (config.block_kv as u32) * (config.head_dim as u32);
    let fused_k = config.csha.as_ref().map_or(false, |c| c.fused_projections);

    ptx.push_str("    // K tile load: 128 threads cooperatively load block_kv*head_dim elems\n");

    // When fused K projection is enabled, null-guard the HBM load: if
    // csha_wk_ptr is non-null the projection already filled SMEM; skip.
    if fused_k {
        ptx.push_str("    ld.param.u64 %rd_wk_chk, [csha_wk_ptr];\n");
        ptx.push_str("    setp.ne.u64 %p_wk_fused, %rd_wk_chk, 0;\n");
        ptx.push_str(&format!(
            "    @%p_wk_fused bra V2_K_LOAD_SKIP_{}; // K already in SMEM from projection\n",
            q_iter
        ));
    }

    // K base global address.
    ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;      // batch*heads\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %head_idx;         // + head\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd6;            // * seq_len\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %k_start;           // + k_start\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd7;            // * head_dim\n");
    ptx.push_str("    shl.b64 %rd58, %rd58, 2;                  // * 4 bytes (f32 source)\n");
    ptx.push_str("    add.u64 %rd58, %rd1, %rd58;               // k_base global\n");
    ptx.push_str("    cvt.u64.u32 %rd59, %tid_x;\n");
    ptx.push_str(&format!("V2_LOOP_K_LOAD_{}:\n", q_iter));
    ptx.push_str("    shl.b64 %rd60, %rd59, 2;\n");
    ptx.push_str("    add.u64 %rd61, %rd58, %rd60;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd61];\n");
    ptx.push_str("    cvt.rn.f16.f32 %h0, %f0;\n");
    ptx.push_str("    shl.b64 %rd60, %rd59, 1;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd60, %rd60, {};                 // + kv_offset\n",
        smem_layout::kv_offset(config)
    ));
    ptx.push_str("    add.u64 %smem_addr, %rd60, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd59, %rd59, 128;\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p0, %rd59, {};\n",
        total_k_elems
    ));
    ptx.push_str(&format!("    @%p0 bra V2_LOOP_K_LOAD_{};\n", q_iter));

    if fused_k {
        ptx.push_str(&format!("V2_K_LOAD_SKIP_{}:\n", q_iter));
    }
    ptx.push_str("    bar.sync 0;  // FENCE: K tile in shmem\n");
}

/// Cooperative V-tile load. Same shape as K load but reads from v_ptr
/// (%rd2) and reuses the KV shmem region (overwriting K).
///
/// When `csha.fused_projections` is true and `csha_wv_ptr` is non-null the
/// projection sweep has already written projected V into SMEM — skip the
/// HBM load to avoid overwriting the fused result.
fn emit_v_tile_load(ptx: &mut String, config: &FlashAttentionConfig, q_iter: u32) {
    let total_v_elems = (config.block_kv as u32) * (config.head_dim as u32);
    let fused_v = config.csha.as_ref().map_or(false, |c| c.fused_projections);

    ptx.push_str("    // V tile load: cooperative, reuses K region\n");

    // When fused V projection is enabled, null-guard: skip if csha_wv_ptr != 0.
    if fused_v {
        ptx.push_str("    ld.param.u64 %rd_wv_chk, [csha_wv_ptr];\n");
        ptx.push_str("    setp.ne.u64 %p_wv_fused, %rd_wv_chk, 0;\n");
        ptx.push_str(&format!(
            "    @%p_wv_fused bra V2_V_LOAD_SKIP_{}; // V already in SMEM from projection\n",
            q_iter
        ));
    }

    ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd6;\n");
    ptx.push_str("    add.u64 %rd58, %rd58, %k_start;\n");
    ptx.push_str("    mul.lo.u64 %rd58, %rd58, %rd7;\n");
    ptx.push_str("    shl.b64 %rd58, %rd58, 2;\n");
    ptx.push_str("    add.u64 %rd58, %rd2, %rd58;               // v_base global\n");
    ptx.push_str("    cvt.u64.u32 %rd59, %tid_x;\n");
    ptx.push_str(&format!("V2_LOOP_V_LOAD_{}:\n", q_iter));
    ptx.push_str("    shl.b64 %rd60, %rd59, 2;\n");
    ptx.push_str("    add.u64 %rd61, %rd58, %rd60;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd61];\n");
    ptx.push_str("    cvt.rn.f16.f32 %h0, %f0;\n");
    ptx.push_str("    shl.b64 %rd60, %rd59, 1;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd60, %rd60, {};\n",
        smem_layout::kv_offset(config)
    ));
    ptx.push_str("    add.u64 %smem_addr, %rd60, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd59, %rd59, 128;\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p0, %rd59, {};\n",
        total_v_elems
    ));
    ptx.push_str(&format!("    @%p0 bra V2_LOOP_V_LOAD_{};\n", q_iter));

    if fused_v {
        ptx.push_str(&format!("V2_V_LOAD_SKIP_{}:\n", q_iter));
    }
    ptx.push_str("    bar.sync 0;  // FENCE: V tile in shmem\n");
}

/// Cooperative HBM→SMEM load for one CSHA projection weight tile.
///
/// 128 threads cooperatively load `d_model × head_dim` f16 values from
/// `weight_param` into the SMEM region at `shmem_base + smem_byte_offset`.
/// Null-guarded: if the weight pointer is 0 the load is skipped.
///
/// `label` is used only for comments and loop-label naming (e.g., "Wq").
/// `q_iter` suffixes labels to prevent duplicates when the outer loop
/// calls this function multiple times.
fn emit_weight_tile_load(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    label: &str,              // e.g. "Wq"
    weight_param: &str,       // PTX param name, e.g. "csha_wq_ptr"
    smem_byte_offset: u32,    // byte offset from shmem[] base
    q_iter: u32,
) {
    let csha = match &config.csha {
        Some(c) if c.fused_projections => c,
        _ => return,
    };
    let d_model    = csha.d_model as u64;
    let head_dim   = config.head_dim as u64;
    let total_elems = d_model * head_dim; // number of f16 elements

    let loop_label = format!("V2_WT_LOAD_{}_{}", label.to_uppercase(), q_iter);
    let skip_label = format!("V2_WT_SKIP_{}_{}", label.to_uppercase(), q_iter);

    ptx.push_str(&format!(
        "    // Cooperative HBM->SMEM load: {} (d_model={}, head_dim={}, smem_off={})\n",
        label, d_model, head_dim, smem_byte_offset
    ));
    // Null-guard: skip load if the weight pointer is 0.
    ptx.push_str(&format!(
        "    ld.param.u64 %rd_wt, [{}];\n",
        weight_param
    ));
    ptx.push_str("    setp.eq.u64 %p_wt, %rd_wt, 0;\n");
    ptx.push_str(&format!("    @%p_wt bra {};\n", skip_label));

    // Compute SMEM base for this tile: shmem_base + smem_byte_offset.
    ptx.push_str(&format!(
        "    add.u64 %rd_wt_dst, %shmem_base, {};\n",
        smem_byte_offset
    ));
    // Each thread loads its elements: idx = tid_x, tid_x+128, tid_x+256, ...
    ptx.push_str("    cvt.u64.u32 %rd_wt_idx, %tid_x;\n");
    ptx.push_str(&format!("{}:\n", loop_label));
    // Byte offset within the tile (2 bytes per f16 element).
    ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_idx, 1;\n");
    // HBM source address.
    ptx.push_str("    add.u64 %rd_wt_src, %rd_wt, %rd_wt_off;\n");
    ptx.push_str("    ld.global.b16 %h_wt, [%rd_wt_src];\n");
    // SMEM destination address = tile_base + element_offset.
    ptx.push_str("    add.u64 %rd_wt_src, %rd_wt_dst, %rd_wt_off;  // reuse %rd_wt_src\n");
    ptx.push_str("    st.shared.b16 [%rd_wt_src], %h_wt;\n");
    // Advance by 128 and loop.
    ptx.push_str("    add.u64 %rd_wt_idx, %rd_wt_idx, 128;\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p_wt, %rd_wt_idx, {};\n",
        total_elems
    ));
    ptx.push_str(&format!("    @%p_wt bra {};\n", loop_label));
    ptx.push_str(&format!("{}:\n", skip_label));
    ptx.push_str("    bar.sync 0;  // FENCE: weight tile in SMEM\n");
}

/// Kernel entry-point name for v2. Same format as v1 with a `_v2` suffix
/// so module caches never collide between versions.
pub fn flash_attention_kernel_name_v2(config: &FlashAttentionConfig) -> String {
    format!("{}_v2", crate::flash_attention::flash_attention_kernel_name(config))
}

/// SMEM byte count for a v2 kernel. Computed from the layout module so
/// static-shmem declaration and launch-arg stay in sync.
pub fn shared_mem_bytes_v2(config: &FlashAttentionConfig) -> u32 {
    smem_layout::total_bytes(config)
}
