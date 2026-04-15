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
    smem_layout::validate_scalar_v2_config(config)
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

        // Cooperative HBM→SMEM weight-tile loads for Wq/Wk/Wv.
        // Must execute BEFORE emit_matmul_projection so the projection
        // sweeps can read W from SMEM. Gated on fused_projections.
        if config.csha.as_ref().map_or(false, |c| c.fused_projections) {
            let base = smem_layout::sp_offset(config) + 4 * (config.block_kv as u32) * 4;
            let wt_bytes = smem_layout::wq_tile_bytes(config);
            emit_weight_tile_load(&mut ptx, config, "Wq", "csha_wq_ptr", base, q_iter);
            emit_weight_tile_load(&mut ptx, config, "Wk", "csha_wk_ptr", base + wt_bytes, q_iter);
            emit_weight_tile_load(&mut ptx, config, "Wv", "csha_wv_ptr", base + 2 * wt_bytes, q_iter);
        }

        phases::csha_hooks::emit_matmul_projection(&mut ptx, config, q_iter);

        // CSHA A.2.4 RoPE epilogue: runs immediately after projection so
        // Q/K SMEM tiles are rotated BEFORE Q-load / S-compute consume them
        // for QK^T.  Mirrors v1's emit_csha_rope_epilogue placement (after
        // emit_csha_matmul_projection, before emit_q_tile_load).
        // No-op when csha=None or rope_q=false.
        phases::csha_hooks::emit_rope_epilogue(&mut ptx, config, q_iter);

        // Phase 1: Q load.
        phases::q_load::emit(&mut ptx, config, q_iter);

        // K/V-tile loop.
        ptx.push_str("    mov.u64 %k_start, 0;\n");
        ptx.push_str("    mov.u64 %k_max, %rd6;                        // seq_len\n");
        ptx.push_str(&format!("V2_LOOP_KV_START_{}:\n", q_iter));

        emit_k_tile_load(&mut ptx, config, q_iter);
        phases::s_compute::emit(&mut ptx, config, q_iter);
        phases::softmax::emit(&mut ptx, config);
        emit_v_tile_load(&mut ptx, config, q_iter);
        phases::pv_accum::emit(&mut ptx, config, q_iter);

        ptx.push_str(&format!(
            "    add.u64 %k_start, %k_start, {};\n",
            config.block_kv
        ));
        ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
        ptx.push_str(&format!("    @%p0 bra V2_LOOP_KV_START_{};\n", q_iter));

        // Phase 6: finalize + output store + LSE.
        // When fused_output_proj=true the Wo projection is a separate follow-up
        // kernel (spec R2: Wo tile does not fit in 48 KB SMEM budget inline).
        // finalize always emits its output store; emit_output_projection adds
        // the Wo null-check dispatch stub + skip labels after it.
        phases::finalize::emit(&mut ptx, config, q_iter);

        // CSHA A5: Wo output projection stub. No-op when csha=None or
        // fused_output_proj=false. Emits pointer null-checks + skip labels;
        // actual Wo @ O and residual add are dispatched as a separate kernel.
        phases::csha_hooks::emit_output_projection(&mut ptx, config, q_iter);
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
fn emit_k_tile_load(ptx: &mut String, config: &FlashAttentionConfig, q_iter: u32) {
    let total_k_elems = (config.block_kv as u32) * (config.head_dim as u32);
    ptx.push_str("    // K tile load: 128 threads cooperatively load block_kv*head_dim elems\n");
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
    ptx.push_str("    bar.sync 0;  // FENCE: K tile in shmem\n");
}

/// Cooperative V-tile load. Same shape as K load but reads from v_ptr
/// (%rd2) and reuses the KV shmem region (overwriting K).
fn emit_v_tile_load(ptx: &mut String, config: &FlashAttentionConfig, q_iter: u32) {
    let total_v_elems = (config.block_kv as u32) * (config.head_dim as u32);
    ptx.push_str("    // V tile load: cooperative, reuses K region\n");
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
