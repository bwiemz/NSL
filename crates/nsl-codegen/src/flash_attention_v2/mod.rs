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

        // ── Step 3b: K RoPE rotation — rotate the full K SMEM tile in-place.
        //   Must run ONCE after K pre-pass (all rows populated) and BEFORE any
        //   S-compute reads K for QK^T.  Q rotation runs per-q_iter inside
        //   emit_rope_epilogue; K rotation runs once here for the whole tile.
        phases::csha_hooks::emit_rope_k_epilogue(&mut ptx, config);

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

            // Tier C: save post-RoPE Q and K only here — V SMEM tile aliases
            // K during the S-pass (same `kv_offset`) and is only populated
            // with real V values after the V pre-pass, so V save is deferred.
            phases::csha_hooks::emit_save_activations_subset(
                &mut ptx, config, q_iter, phases::csha_hooks::SaveSet::QK,
            );

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

        // Tier C: V save runs AFTER the V pre-pass so v_smem_base actually
        // holds V projection (during S-pass it aliased K).
        for q_iter in 0..iters {
            phases::csha_hooks::emit_save_activations_subset(
                &mut ptx, config, q_iter, phases::csha_hooks::SaveSet::V,
            );
        }

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

/// Tier C backward orchestrator — emits the full backward PTX kernel by
/// wiring every Phase 3 emitter in the correct execution order.
///
/// Order (mirrors the CPU reference in `tests/csha_reference.rs`):
///   1. prelude — .visible .entry, register pool, pointer loads.
///   2. q_load — cooperative HBM load of saved post-RoPE Q_proj.
///   3. Per q_tile_iter (4 Q rows each):
///        accumulator reset (%f_dq_*, %f_dk_*, %f_dv_* ← 0)
///        ds_compute — P recompute + dP + dS with softmax Jacobian
///        dv_accum   — P^T @ dO
///        dqdk_accum — dS @ K, dS^T @ Q
///   4. dRoPE (Q + K inverse rotation, skipped when rope_q=false).
///   5. dproj — dWq/dWk/dWv weight gradient accumulations.
///   6. dRMSNorm — closed-form dx.
///   7. finalize — cooperative HBM stores of all 7 gradients + final
///      bar.sync.
///
/// Returns `Err` if the validator rejects the config in
/// `Direction::Backward` (the 99 KB budget check includes
/// `backward_extra_bytes` for the gradient accumulator tiles).
pub fn synthesize_backward(config: &FlashAttentionConfig) -> Result<String, String> {
    smem_layout::validate_scalar_v2_config(config, smem_layout::Direction::Backward)
        .map_err(|e| format!("backward validator rejected: {e}"))?;

    let mut ptx = String::new();

    // Phase 0: header, .visible .entry, SMEM, register pool, indices.
    phases::backward::prelude::emit(&mut ptx, config);

    // Phase 1: load saved post-RoPE Q/K/V from HBM into their SMEM
    // tiles. Fired once; subsequent q_tile_iter iterations read from
    // these tiles. K and V are loaded via kv_load so ds_compute can
    // recompute S = Q @ K^T and dP = dO · V^T with real addressing.
    //
    // Q load must cover ALL block_q rows — each q_tile_iter loads 4 rows
    // (one per warp), so we iterate all q_tile_iters to fill the full
    // Q SMEM tile. Without this, only rows 0..3 would be populated and
    // ds_compute for q_tile_iter > 0 would read uninitialised SMEM.
    let q_load_iters = (config.block_q as u32).div_ceil(4);
    for qi in 0..q_load_iters {
        phases::backward::q_load::emit(&mut ptx, config, qi);
    }
    phases::backward::kv_load::emit_k(&mut ptx, config);
    phases::backward::kv_load::emit_v(&mut ptx, config);

    // Phase 2: per q_tile_iter KV loop. One iter per 4-row warp group
    // (matches the forward orchestrator's tile cadence).
    let iters = (config.block_q as u32).div_ceil(4);
    let slices_per_lane = ((config.head_dim as u32) / 32).max(1);

    // Cooperatively zero the dV and dK SMEM tiles ONCE before the KV
    // loop. Both tiles are block_kv × head_dim × 4 bytes f32; 128
    // threads per block cooperatively zero
    // `block_kv*head_dim*2 / 128` floats each.
    let dv_off = smem_layout::backward_dv_offset(config);
    let dk_off = smem_layout::backward_dk_offset(config);
    let dq_off = smem_layout::backward_dq_offset(config);
    let dk_dv_cells = (config.block_kv * config.head_dim) as u32;
    let dq_cells = (config.block_q * config.head_dim) as u32;
    let cells_per_thread = dk_dv_cells.div_ceil(128);
    let dq_cells_per_thread = dq_cells.div_ceil(128);
    for (tag, off, total, per_thread) in [
        ("DV", dv_off, dk_dv_cells, cells_per_thread),
        ("DK", dk_off, dk_dv_cells, cells_per_thread),
        ("DQ", dq_off, dq_cells, dq_cells_per_thread),
    ] {
        ptx.push_str(&format!(
            "    // BWD zero-init {tag} SMEM tile ({total} cells, \
             {per_thread}/thread)\n"
        ));
        for k in 0..per_thread {
            let thread_cell = k * 128;
            // cell_idx = tid + k*128; byte_off = off + cell_idx*4
            ptx.push_str("    cvt.u64.u32 %rd_zero_idx, %tid_x;\n");
            if thread_cell > 0 {
                ptx.push_str(&format!(
                    "    add.u64 %rd_zero_idx, %rd_zero_idx, {};\n",
                    thread_cell
                ));
            }
            // Guard (only stores in-range cells): cell_idx < total
            ptx.push_str(&format!(
                "    setp.lt.u64 %p_zero, %rd_zero_idx, {};\n", total
            ));
            ptx.push_str("    shl.b64 %rd_zero_idx, %rd_zero_idx, 2;\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_zero_idx, %rd_zero_idx, {off};\n"
            ));
            ptx.push_str("    add.u64 %rd_zero_idx, %shmem_base, %rd_zero_idx;\n");
            ptx.push_str("    mov.f32 %f_zero_val, 0f00000000;\n");
            ptx.push_str("    @%p_zero st.shared.f32 [%rd_zero_idx], %f_zero_val;\n");
        }
    }
    ptx.push_str("    bar.sync 0;  // dV + dK + dQ tiles zeroed\n");

    for q_iter in 0..iters {
        ptx.push_str(&format!(
            "    // ====== BWD q_tile_iter = {q_iter} / {iters} ======\n"
        ));
        // %f_dq_{slice} register-held accumulator reset per iter.
        // (dV/dK now use SMEM tiles zeroed above, so no per-iter reset.)
        for slice in 0..slices_per_lane {
            ptx.push_str(&format!("    mov.f32 %f_dq_{slice}, 0f00000000;\n"));
        }
        phases::backward::ds_compute::emit(&mut ptx, config, q_iter);
        phases::backward::dv_accum::emit(&mut ptx, config, q_iter);
        phases::backward::dqdk_accum::emit(&mut ptx, config, q_iter);

        // Flush %f_dq_{slice} registers into the dQ SMEM tile so the
        // per-iter register values survive across iters. Warp owns row
        // (warp_row = warp_id + q_iter*4), lane owns d-slice cols.
        let hd = config.head_dim as u32;
        let row_stride = hd * 4; // f32
        let slices = slices_per_lane;
        ptx.push_str(&format!(
            "    // BWD flush %f_dq -> dQ SMEM tile (q_tile_iter={q_iter})\n"
        ));
        ptx.push_str(&format!(
            "    add.u32 %r0, %warp_id, {};\n", q_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd_dqs_row, %r0;\n");
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_dqs_row, %rd_dqs_row, {row_stride};\n"
        ));
        ptx.push_str(&format!(
            "    add.u64 %rd_dqs_row, %rd_dqs_row, {dq_off};\n"
        ));
        ptx.push_str("    add.u64 %rd_dqs_row, %shmem_base, %rd_dqs_row;\n");
        for slice in 0..slices {
            // col = lane * slices + slice; byte off = col * 4
            ptx.push_str("    cvt.u64.u32 %rd_dqs_col, %lane;\n");
            if slices > 1 {
                ptx.push_str(&format!(
                    "    mul.lo.u64 %rd_dqs_col, %rd_dqs_col, {slices};\n"
                ));
            }
            if slice > 0 {
                ptx.push_str(&format!(
                    "    add.u64 %rd_dqs_col, %rd_dqs_col, {slice};\n"
                ));
            }
            ptx.push_str("    shl.b64 %rd_dqs_col, %rd_dqs_col, 2;\n");
            ptx.push_str(
                "    add.u64 %rd_dqs_addr, %rd_dqs_row, %rd_dqs_col;\n",
            );
            ptx.push_str(&format!(
                "    st.shared.f32 [%rd_dqs_addr], %f_dq_{slice};\n"
            ));
        }
    }
    ptx.push_str("    bar.sync 0;  // dQ SMEM tile complete\n");

    // Phase 3: CSHA hooks (x_norm recompute, inverse RoPE, dW{q,k,v},
    // dRMSNorm). Each writes directly to HBM except the dRoPE rotation
    // which mutates the dQ/dK SMEM tiles in place.
    phases::backward::csha_hooks_backward::emit_xnorm_recompute(&mut ptx, config);
    phases::backward::csha_hooks_backward::emit_drope(&mut ptx, config, 0);
    phases::backward::csha_hooks_backward::emit_dproj(&mut ptx, config, 0);
    phases::backward::csha_hooks_backward::emit_drmsnorm(&mut ptx, config, 0);
    // phases::backward::csha_hooks_backward::emit_dproj(&mut ptx, config, 0);
    // phases::backward::csha_hooks_backward::emit_drmsnorm(&mut ptx, config, 0);

    // Phase 4: cooperative global stores of the 7 gradients + final fence.
    phases::backward::finalize::emit(&mut ptx, config, 0);

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
    // NUL-terminate so cuModuleLoadData accepts the byte slice.
    ptx.push('\0');
    Ok(ptx)
}

#[cfg(test)]
mod backward_orchestrator_tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg_fused_backward(
        block_q: i64, block_kv: i64, head_dim: i64, heads: u32, d_model: u32,
    ) -> FlashAttentionConfig {
        let _ = heads;
        FlashAttentionConfig {
            block_q, block_kv, head_dim,
            causal: false, paged: false, rope_q: true,
            rope_style: RopeStyle::Adjacent,
            gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn synthesize_backward_emits_all_phases_in_order() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let ptx = synthesize_backward(&cfg).expect("synth backward");

        let idx_prelude = ptx.find(".visible .entry").expect(".visible .entry missing");
        let idx_qload = ptx.find("V2_BWD_Q_LOAD_0:").expect("q_load label missing");
        let idx_ds = ptx.find("V2_BWD_DS_0:").expect("dS label missing");
        let idx_dv = ptx.find("V2_BWD_DV_ACCUM_0:").expect("dV label missing");
        let idx_dq = ptx.find("V2_BWD_DQ_ACCUM_0:").expect("dQ label missing");
        let idx_drope = ptx.find("V2_BWD_DROPE_Q_LOOP_0:").expect("dRoPE label missing");
        let idx_dproj = ptx.find("V2_BWD_DPROJ_WQ_LOOP_0:").expect("dproj label missing");
        let idx_drmsnorm = ptx.find("V2_BWD_DRMSNORM_0:").expect("dRMSNorm label missing");
        let idx_final = ptx.find("ret;").expect("ret missing");

        assert!(idx_prelude < idx_qload, "prelude before q_load");
        assert!(idx_qload < idx_ds, "q_load before ds");
        assert!(idx_ds < idx_dv, "ds before dV");
        assert!(idx_dv < idx_dq, "dV before dQ");
        assert!(idx_dq < idx_drope, "dQ/dK before dRoPE");
        assert!(idx_drope < idx_dproj, "dRoPE before dproj");
        assert!(idx_dproj < idx_drmsnorm, "dproj before dRMSNorm");
        assert!(idx_drmsnorm < idx_final, "dRMSNorm before ret");
    }

    #[test]
    fn synthesize_backward_rejects_over_budget_config() {
        // head_dim=64, heads=8, block_q=64 with backward tiles should
        // blow the 99 KB cap (see T2.1's rejection test).
        let cfg = base_cfg_fused_backward(64, 64, 64, 8, 64);
        let err = synthesize_backward(&cfg)
            .expect_err("expected backward validator rejection");
        assert!(err.contains("backward validator rejected"), "err: {err}");
        assert!(err.contains("Backward"), "err must name direction: {err}");
    }

    #[test]
    fn synthesize_backward_nul_terminated() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let ptx = synthesize_backward(&cfg).expect("synth backward");
        assert!(ptx.ends_with('\0'),
            "cuModuleLoadData requires NUL terminator");
    }

    #[test]
    fn synthesize_backward_emits_accumulator_resets() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let ptx = synthesize_backward(&cfg).expect("synth backward");
        // dQ is register-held — orchestrator zero-inits %f_dq_{slice}
        // per q_tile_iter (cross-iter accumulation not needed; dQ has
        // unique per-thread cell ownership).
        assert!(ptx.contains("mov.f32 %f_dq_0, 0f00000000"));
        // dK and dV migrated to SMEM tiles (backward_dk_offset /
        // backward_dv_offset). Orchestrator cooperatively zeros the
        // tiles ONCE before the KV loop rather than per-iter resets.
        assert!(ptx.contains("BWD zero-init DK SMEM tile"),
            "orchestrator must zero dK SMEM tile");
        assert!(ptx.contains("BWD zero-init DV SMEM tile"),
            "orchestrator must zero dV SMEM tile");
    }
}
