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
pub mod tier_b1;

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
    // K/V pre-pass iteration count — decoupled from `iters` so the fused
    // path supports asymmetric tiles (block_q != block_kv). Each K/V
    // pre-pass iter writes 4 tile rows (one per warp); `kv_iters` rounds
    // up to cover the full block_kv-row tile.
    let kv_iters = (config.block_kv as u32).div_ceil(4);
    let slices = (config.head_dim as u32) / 32;
    let fused_proj = config.csha.as_ref().is_some_and(|c| c.fused_projections);

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

        // ── Step 3: K pre-pass — all kv_iters write K rows to kv_offset.
        //   Reads normalized x (written by RMSNorm pre-pass above).  Uses
        //   `kv_iters` (not `iters`) so asymmetric tiles (block_q !=
        //   block_kv) populate exactly block_kv K rows.
        ptx.push_str("    // CSHA K pre-pass: populate full K SMEM tile\n");
        for kv_iter in 0..kv_iters {
            phases::csha_hooks::emit_k_prepass_sweep(&mut ptx, config, kv_iter);
        }
        ptx.push_str("    bar.sync 0; // K tile complete; safe for all S-computes\n");

        // ── Step 3b: K RoPE rotation — rotate the full K SMEM tile in-place.
        //   Must run ONCE after K pre-pass (all rows populated) and BEFORE any
        //   S-compute reads K for QK^T.  Q rotation runs per-q_iter inside
        //   emit_rope_epilogue; K rotation runs once here for the whole tile.
        phases::csha_hooks::emit_rope_k_epilogue(&mut ptx, config);

        // ── Step 3c: K save — post-RoPE K save runs here so asymmetric
        //   tiles (block_q != block_kv) cover exactly block_kv K rows.
        //   Iteration count is `kv_iters`, matching the K pre-pass.
        //
        //   Single-tile assumption inherited from the symmetric path: the
        //   K save helper writes to HBM at `%q_start + warp_row` because
        //   the K pre-pass reads x at `%q_start + warp_row` (see
        //   `csha_hooks::emit_kv_prepass_reginit`), so the write address
        //   is consistent with the data's sequence position.  The backward
        //   `kv_load::emit_k` reads HBM without a `q_start` term, which
        //   forces `q_start == 0` (single-tile workloads — all current
        //   test harnesses and the prescribed Llama-3 proxy shape).
        //   Multi-tile K-save addressing is a pre-existing gap orthogonal
        //   to asymmetric tile support; tracked as a follow-up.
        //
        //   Gated on `save_activations_for_backward` at the orchestrator
        //   level to avoid emitting N×"skip" comment lines when saves are
        //   disabled (the shipped-binary common case).
        if config.csha.as_ref().is_some_and(|c| c.save_activations_for_backward) {
            for kv_iter in 0..kv_iters {
                phases::csha_hooks::emit_save_activations_subset(
                    &mut ptx, config, kv_iter, phases::csha_hooks::SaveSet::K,
                );
            }
        }

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

            // Tier C: save post-RoPE Q only here.  K was saved in Step 3c
            // (after K RoPE, before the S-compute loop) to decouple from
            // block_q — asymmetric-tile configs need kv_iters worth of K
            // rows, not iters.  V SMEM tile aliases K during the S-pass
            // and is saved after the V pre-pass below (Step 5).
            phases::csha_hooks::emit_save_activations_subset(
                &mut ptx, config, q_iter, phases::csha_hooks::SaveSet::Q,
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

        // V pre-pass: all kv_iters write V rows to kv_offset.
        // All S-computes are done; K at kv_offset is no longer needed.
        // Uses `kv_iters` (not `iters`) so asymmetric tiles populate
        // exactly block_kv V rows regardless of block_q.
        ptx.push_str("    bar.sync 0; // S-pass done; V pre-pass overwrites K SMEM\n");
        ptx.push_str("    // CSHA V pre-pass: populate full V SMEM tile\n");
        for kv_iter in 0..kv_iters {
            phases::csha_hooks::emit_v_prepass_sweep(&mut ptx, config, kv_iter);
        }
        ptx.push_str("    bar.sync 0; // V tile complete; safe for all PV-accums\n");

        // Tier C: V save runs AFTER the V pre-pass so v_smem_base actually
        // holds V projection (during S-pass it aliased K).  Iterates
        // `kv_iters` to cover all block_kv V rows.
        for kv_iter in 0..kv_iters {
            phases::csha_hooks::emit_save_activations_subset(
                &mut ptx, config, kv_iter, phases::csha_hooks::SaveSet::V,
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

            // Phase 1: Q load — populates Q SMEM tile (q_offset) with post-RoPE
            // values that the Tier C save path will read below.
            phases::q_load::emit(&mut ptx, config, q_iter);

            // Tier C: save post-RoPE Q/K activations for backward (gated on flag).
            // Split QK vs V: Q lives in q_offset and is stable after q_load; K
            // lives at kv_offset but V aliases the same slot, so K must be
            // saved immediately after its HBM load and BEFORE v_tile_load
            // overwrites the slot.  SaveSet::V runs after v_tile_load below.
            phases::csha_hooks::emit_save_activations_subset(
                &mut ptx, config, q_iter, phases::csha_hooks::SaveSet::QK,
            );

            // K/V-tile loop.
            ptx.push_str("    mov.u64 %k_start, 0;\n");
            ptx.push_str("    mov.u64 %k_max, %rd6;                        // seq_len\n");
            ptx.push_str(&format!("V2_LOOP_KV_START_{}:\n", q_iter));

            emit_k_tile_load(&mut ptx, config, q_iter);
            phases::s_compute::emit(&mut ptx, config, q_iter);
            phases::softmax::emit(&mut ptx, config, q_iter);
            // Tier C: persist row_max/row_sum to HBM IMMEDIATELY after
            // softmax's online update, inside the KV loop. The prior
            // placement (after the KV loop exit) allowed PV-accum to write
            // back to physical f32 registers that ptxas had coalesced with
            // `%row_max` / `%f_sdx_fmax` / `%f_sdx_nmax`, clobbering the
            // captured softmax state before the save could fire (confirmed
            // by J-A3 measurement: fsum was correct but fmax/newmax read
            // `~±1e-30` uniform — same as default %row_max read-back). For
            // multi-tile KV loops this fires per-tile, writing to the same
            // HBM address each time; the final tile's values win, which IS
            // the final committed softmax state — identical semantics to
            // the previous post-loop placement for correctness.
            phases::csha_hooks::emit_save_softmax_state(&mut ptx, config, q_iter);
            emit_v_tile_load(&mut ptx, config, q_iter);
            // Tier C: save V from v_smem (which aliases K after v_tile_load).
            // NOTE: under the non-fused path the save addressing uses
            // `q_start+warp_row` which is Q-indexed; under the standard KV
            // loop this matches the K/V tile layout only at k_start=0.
            // Backward numerical correctness for non-fused path is NOT the
            // goal of this edit — the goal is to ensure the forward PTX
            // assembles and launch rc=0 so structural gradients flow; a
            // proper addressing rewrite is tracked as a separate follow-up.
            phases::csha_hooks::emit_save_activations_subset(
                &mut ptx, config, q_iter, phases::csha_hooks::SaveSet::V,
            );
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
    let fused_k = config.csha.as_ref().is_some_and(|c| c.fused_projections);

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
    let fused_v = config.csha.as_ref().is_some_and(|c| c.fused_projections);

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

/// Dynamic SMEM byte count to pass to `cuLaunchKernel` for the v2 Tier C
/// **backward** kernel. Covers the backward shmem region
/// (`backward_total_bytes` = forward total + dQ/dK/dV/P/dS/v_in tiles +
/// CSHA dRMSNorm strips) plus PCA Tier A's embedded `seg_smem` tail when
/// `config.segment_masked`. The forward shmem helper above does not
/// suffice — backward needs `backward_extra_bytes` on top of the forward
/// layout, and the seg_smem region lives in the same extern shmem
/// allocation per the Blackwell static+extern fix
/// (see `phases/backward/prelude.rs::backward_needs_dynamic_smem`).
pub fn shared_mem_bytes_v2_backward(config: &FlashAttentionConfig) -> u32 {
    let seg_overhead = if config.segment_masked {
        crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32
    } else {
        0
    };
    phases::backward::prelude::backward_total_bytes(config) + seg_overhead
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

    // CSHA A.4: head pruning guard — mirror forward. Runs ONCE before
    // any backward phase so blocks whose head_idx >= csha_active_heads
    // early-`ret` and skip the entire backward pass. Reuses forward's
    // emitter: the guard's scratch (%r10/%r11/%p0) and %head_idx are
    // declared by the backward prelude's register pool, and
    // `csha_active_heads` is in the backward param list. The guard's
    // label `V2_CSHA_ACTIVE_HEADS_SKIP` is function-scoped in PTX so
    // the forward copy cannot collide with the backward copy.
    //
    // Placement: must come AFTER the PCA `bar.sync 0` at the end of
    // `backward::prelude::emit` (segment_masked path) so the warp-0
    // cooperative segment_ids load completes for all 128 threads of
    // the block before any thread takes the early-`ret` branch.
    phases::csha_hooks::emit_active_heads_guard(&mut ptx, config);

    // Phase 1: load saved post-RoPE Q from HBM into its SMEM tile.
    // K and V are reloaded inside the outer KV-tile loop so `%k_start`
    // advances across the full sequence instead of freezing on tile 0.
    //
    // Q load must cover ALL block_q rows — each q_tile_iter loads 4 rows
    // (one per warp), so we iterate all q_tile_iters to fill the full
    // Q SMEM tile. Without this, only rows 0..3 would be populated and
    // ds_compute for q_tile_iter > 0 would read uninitialised SMEM.
    let q_load_iters = (config.block_q as u32).div_ceil(4);
    for qi in 0..q_load_iters {
        phases::backward::q_load::emit(&mut ptx, config, qi);
    }

    // Phase 2: per q_tile_iter KV loop. One iter per 4-row warp group
    // (matches the forward orchestrator's tile cadence).
    let iters = (config.block_q as u32).div_ceil(4);
    let slices_per_lane = ((config.head_dim as u32) / 32).max(1);

    // Cooperatively zero the dQ SMEM tile ONCE before the KV loop. dQ
    // accumulates across all KV tiles for this q-block, so its backing
    // tile must persist for the whole outer k_start sweep.
    let dv_off = smem_layout::backward_dv_offset(config);
    let dk_off = smem_layout::backward_dk_offset(config);
    let dq_off = smem_layout::backward_dq_offset(config);
    let corr_off = smem_layout::backward_rms_strip_offset(config);
    let dk_dv_cells = (config.block_kv * config.head_dim) as u32;
    let dq_cells = (config.block_q * config.head_dim) as u32;
    let corr_cells = config.block_q as u32;
    let cells_per_thread = dk_dv_cells.div_ceil(128);
    let dq_cells_per_thread = dq_cells.div_ceil(128);
    let corr_cells_per_thread = corr_cells.div_ceil(128);
    // Zero-init the dQ SMEM tile. The row-correction strip is written
    // unconditionally by `emit_d_correction` below (overwrite, not RMW),
    // so it does NOT need pre-zeroing.
    let _ = (corr_off, corr_cells, corr_cells_per_thread);
    ptx.push_str(&format!(
        "    // BWD zero-init DQ SMEM tile ({dq_cells} cells, \
         {dq_cells_per_thread}/thread)\n"
    ));
    for k in 0..dq_cells_per_thread {
        let thread_cell = k * 128;
        ptx.push_str("    cvt.u64.u32 %rd_zero_idx, %tid_x;\n");
        if thread_cell > 0 {
            ptx.push_str(&format!(
                "    add.u64 %rd_zero_idx, %rd_zero_idx, {};\n",
                thread_cell
            ));
        }
        ptx.push_str(&format!(
            "    setp.lt.u64 %p_zero, %rd_zero_idx, {};\n", dq_cells
        ));
        ptx.push_str("    shl.b64 %rd_zero_idx, %rd_zero_idx, 2;\n");
        ptx.push_str(&format!(
            "    add.u64 %rd_zero_idx, %rd_zero_idx, {dq_off};\n"
        ));
        ptx.push_str("    add.u64 %rd_zero_idx, %shmem_base, %rd_zero_idx;\n");
        ptx.push_str("    mov.f32 %f_zero_val, 0f00000000;\n");
        ptx.push_str("    @%p_zero st.shared.f32 [%rd_zero_idx], %f_zero_val;\n");
    }
    ptx.push_str("    bar.sync 0;  // dQ tile zeroed\n");

    // One-shot D-correction phase: D[i] = dO[i] . O[i] for each row.
    // Replaces the previous ROWPRE KV loop + cross-tile strip RMW.
    // Mathematically equivalent to sum_c P[i,c]*dP[i,c] but computed in
    // a single row-wise pass — matches the CPU reference's `d_corr`
    // formulation and avoids the accumulation bug that made the old
    // strip ~62× too large on the sq128 fixture.
    ptx.push_str("    // --- one-shot D correction strip (dO . O) ---\n");
    for q_iter in 0..iters {
        phases::backward::ds_compute::emit_d_correction(&mut ptx, config, q_iter);
    }
    ptx.push_str("    bar.sync 0;  // D correction strip complete\n");

    ptx.push_str("    mov.u64 %k_start, 0;\n");
    ptx.push_str("    mov.u64 %k_max, %rd6;\n");
    ptx.push_str("V2_BWD_LOOP_KV:\n");
    phases::backward::kv_load::emit_k_suffixed(&mut ptx, config, "MAIN");
    phases::backward::kv_load::emit_v_suffixed(&mut ptx, config, "MAIN");
    for (tag, off, total, per_thread) in [
        ("DV", dv_off, dk_dv_cells, cells_per_thread),
        ("DK", dk_off, dk_dv_cells, cells_per_thread),
    ] {
        ptx.push_str(&format!(
            "    // BWD zero-init {tag} SMEM tile ({total} cells, \
             {per_thread}/thread)\n"
        ));
        for k in 0..per_thread {
            let thread_cell = k * 128;
            ptx.push_str("    cvt.u64.u32 %rd_zero_idx, %tid_x;\n");
            if thread_cell > 0 {
                ptx.push_str(&format!(
                    "    add.u64 %rd_zero_idx, %rd_zero_idx, {};\n",
                    thread_cell
                ));
            }
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
    ptx.push_str("    bar.sync 0;  // dV + dK tiles zeroed for this KV tile\n");

    for q_iter in 0..iters {
        ptx.push_str(&format!(
            "    // ====== BWD q_tile_iter = {q_iter} / {iters} ======\n"
        ));
        let hd = config.head_dim as u32;
        let row_stride = hd * 4; // f32
        let slices = slices_per_lane;
        ptx.push_str(&format!(
            "    // BWD reload dQ SMEM tile -> %f_dq (q_tile_iter={q_iter})\n"
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
            ptx.push_str("    add.u64 %rd_dqs_addr, %rd_dqs_row, %rd_dqs_col;\n");
            ptx.push_str(&format!(
                "    ld.shared.f32 %f_dq_{slice}, [%rd_dqs_addr];\n"
            ));
        }

        phases::backward::ds_compute::emit(&mut ptx, config, q_iter);
        phases::backward::dv_accum::emit(&mut ptx, config, q_iter);
        phases::backward::dqdk_accum::emit(&mut ptx, config, q_iter);

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
    phases::backward::finalize::emit_store_kv_only(&mut ptx, config, 0);
    ptx.push_str(&format!("    add.u64 %k_start, %k_start, {};\n", config.block_kv));
    ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
    ptx.push_str("    @%p0 bra V2_BWD_LOOP_KV;\n");
    ptx.push_str("    bar.sync 0;  // dQ SMEM tile complete across all KV tiles\n");

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
    phases::backward::finalize::emit_store_dq_only(&mut ptx, config, 0);

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
            segment_masked: false,
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
        // Use rfind so we match the trailing `ret;` that closes the
        // kernel body, not the guarded `@%p0 ret;` inside the A.4
        // dead-head guard that sits between prelude and q_load.
        let idx_final = ptx.rfind("ret;").expect("ret missing");

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
    fn synthesize_backward_emits_active_heads_guard_when_csha_is_some() {
        // With csha=Some, backward must emit the A.4 active_heads guard
        // so dead-head blocks early-`ret` and skip the entire backward
        // pass (same contract as forward, closing the Tier C follow-up
        // for dead-head elimination on backward).
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let ptx = synthesize_backward(&cfg).expect("synth backward");

        assert!(
            ptx.contains("CSHA A.4: active_heads guard"),
            "backward must emit A.4 guard annotation when csha=Some"
        );
        assert!(
            ptx.contains("ld.param.u32 %r10, [csha_active_heads];"),
            "backward A.4 guard must load csha_active_heads param"
        );
        assert!(
            ptx.contains("V2_CSHA_ACTIVE_HEADS_SKIP:"),
            "backward A.4 guard must emit the skip-label"
        );

        // The guard must run BEFORE any backward phase (q_load / ds /
        // dv / dq / finalize). Placing it after the prelude's
        // bar.sync but before q_load is what ensures dead-head blocks
        // pay only the prelude setup and then `ret`.
        let idx_guard = ptx
            .find("CSHA A.4: active_heads guard")
            .expect("guard annotation missing");
        let idx_qload = ptx
            .find("V2_BWD_Q_LOAD_0:")
            .expect("q_load label missing");
        assert!(
            idx_guard < idx_qload,
            "A.4 guard must precede q_load so dead-head blocks skip all backward work"
        );
    }

    #[test]
    fn synthesize_backward_emits_no_guard_when_csha_is_none() {
        // With csha=None the A.4 guard must no-op (emit only the
        // `csha=None, no emission` annotation) so non-CSHA backward
        // kernels are byte-identical to pre-guard emission modulo the
        // single comment line.
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::Adjacent,
            gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
            segment_masked: false,
            csha: None,
        };
        let ptx = synthesize_backward(&cfg).expect("synth backward");

        assert!(
            ptx.contains("CSHA A.4 active_heads guard: csha=None, no emission"),
            "csha=None backward must emit the no-emission annotation"
        );
        assert!(
            !ptx.contains("ld.param.u32 %r10, [csha_active_heads];"),
            "csha=None backward must NOT load csha_active_heads"
        );
        assert!(
            !ptx.contains("V2_CSHA_ACTIVE_HEADS_SKIP:"),
            "csha=None backward must NOT emit the skip-label"
        );
    }

    #[test]
    fn synthesize_backward_emits_accumulator_resets() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let ptx = synthesize_backward(&cfg).expect("synth backward");
        // WIP q-block serialization architecture: dQ, dK, dV all migrated
        // to SMEM tiles zeroed cooperatively before their respective loops.
        // dQ is zeroed ONCE before the KV loop (cumulates across KV iters
        // via register-SMEM cycle inside each q_tile_iter). dK/dV are
        // zeroed at the TOP of each KV iter since they flush to f32
        // scratch (Option A) after every iter.
        assert!(ptx.contains("BWD zero-init DQ SMEM tile"),
            "orchestrator must zero dQ SMEM tile");
        assert!(ptx.contains("BWD zero-init DK SMEM tile"),
            "orchestrator must zero dK SMEM tile");
        assert!(ptx.contains("BWD zero-init DV SMEM tile"),
            "orchestrator must zero dV SMEM tile");
    }

    /// `shared_mem_bytes_v2_backward` must be strictly larger than the
    /// forward `shared_mem_bytes_v2` for any non-trivial backward config —
    /// `backward_extra_bytes` (P, dS, dQ/dK/dV, V_in, dRMSNorm strips) is
    /// always non-zero. A regression that returns the forward total here
    /// silently short-allocates dynamic SMEM at every backward launch.
    #[test]
    fn backward_shmem_strictly_exceeds_forward_shmem() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let fwd = shared_mem_bytes_v2(&cfg);
        let bwd = shared_mem_bytes_v2_backward(&cfg);
        assert!(
            bwd > fwd,
            "backward shmem ({bwd}) must exceed forward shmem ({fwd}) by backward_extra_bytes"
        );
    }

    /// When `segment_masked`, the backward shmem must also include the
    /// embedded `seg_smem` tail (sized to `DEFAULT_SMEM_SEGMENT_BUDGET`)
    /// so the launcher's dynamic allocation covers both `backward_total`
    /// and the trailing segment_ids region.
    #[test]
    fn backward_shmem_includes_segment_budget_when_masked() {
        let base = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let masked = FlashAttentionConfig { segment_masked: true, ..base.clone() };
        let unmasked = shared_mem_bytes_v2_backward(&base);
        let with_seg = shared_mem_bytes_v2_backward(&masked);
        assert_eq!(
            with_seg - unmasked,
            crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32,
            "segment_masked must add exactly DEFAULT_SMEM_SEGMENT_BUDGET bytes"
        );
    }
}
