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
pub mod sinks;
pub mod tier_b1;
pub mod tier_b2;
pub mod per_doc_cta;

use crate::flash_attention::FlashAttentionConfig;
use crate::pca_segment::SegmentResidency;
use phases::pv_accum::O_BASE;

/// v2 entry point. Returns a byte vector ending with a single trailing
/// newline followed by a NUL terminator so `cuModuleLoadData` accepts it.
///
/// Calls `synthesize_flash_attention_ptx_v2_with_tier_b(config, None)` —
/// output is byte-identical to all pre-Tier-B baselines (spec §3.4.6).
pub fn synthesize_flash_attention_ptx_v2(config: &FlashAttentionConfig) -> Vec<u8> {
    synthesize_flash_attention_ptx_v2_with_tier_b(config, None)
}

/// v2 entry point with optional PCA Tier B support.
///
/// When `tier_b` is `None` this produces byte-identical output to
/// `synthesize_flash_attention_ptx_v2` (no-op guarantee, spec §3.4.6).
/// When `tier_b` is `Some((seq_len, residency))` and `should_emit_tier_b`
/// returns true, the emitted PTX includes:
///   1. Range-table preamble in the forward prelude (after Tier A bar.sync 0).
///   2. Per-KV-tile skip predicate via `s_compute::emit(... tier_b)`.
///   3. `KV_TILE_SKIP_TB_{q_iter}:` labels at the bottom of each KV tile loop.
///
/// **CSHA Tier B.1 dispatch (orthogonal to PCA Tier B)**: when
/// `csha.level >= 2` AND `gpu_sm >= 80`, the kernel takes the
/// pipelined-MMA path via `tier_b1::synthesize` (Hopper/Blackwell
/// cp.async + m16n8k16). CSHA Tier B.1 and PCA Tier B are different
/// subsystems; the `tier_b` parameter here refers ONLY to PCA Tier B's
/// range-table optimisation. When CSHA Level 2 + sm_80+ matches, Tier
/// B.1 takes priority and the PCA-Tier-B `tier_b` parameter is
/// ignored (the two optimisations target different paths and don't
/// currently compose).
///
/// `chunk_config::select` gates the Tier B.1 dispatch. On failure (no
/// chunk fits SMEM, register pressure, etc.) it returns a
/// `DowngradeReason` and we fall through to the Tier A v2 path below.
/// Under normal operation the upstream planner (`csha_pipeline::plan_layer`)
/// has already determined L2 fits per the cost model in spec section 7,
/// so this fall-through is a safety net for cost-model/budget
/// disagreements.
pub fn synthesize_flash_attention_ptx_v2_with_tier_b(
    config: &FlashAttentionConfig,
    tier_b: Option<(u32, SegmentResidency)>,
) -> Vec<u8> {
    // CSHA Tier B.1 dispatch (orthogonal to PCA Tier B; see docstring).
    // Sprint 3 cycle-7 (§4.3 attention sinks v1): defense-in-depth refusal —
    // mirrors `is_tier_b1_dispatch`. The pipelined-MMA forward does not
    // understand sinks; falling through to Tier A v2 here picks up Sprint
    // 1b's sink-aware narrow path. Lift point: a future sprint with
    // sink-aware Tier B.1 kernels.
    let want_tier_b1 = config.csha.as_ref().is_some_and(|c| c.level >= 2)
        && config.gpu_sm >= 80
        && config.num_sink_tokens == 0;
    if want_tier_b1 {
        match tier_b1::chunk_config::select(config) {
            Ok(chunk) => {
                // Tier B.1 dispatch ALWAYS expects the caller-side pre-pass
                // (`nsl-runtime::cuda::tier_b1_prepass::launch_x_prepass`)
                // to have already RMSNormalized + narrowed + chunkified x.
                // The runtime FFI orchestrates the pre-pass automatically
                // when it sees the `_tier_b1_chunk<N>` kernel-name suffix.
                // Force-override `skip_rmsnorm_prologue` so the in-kernel
                // RMSNorm prologue is suppressed — otherwise x would be
                // RMSNormalized twice (once by the pre-pass, once again
                // here), and the second pass would re-divide an already-
                // unit-RMS row by its own RMS, producing roughly-identity-
                // looking but subtly drifted x_normed.
                //
                // Cloning the config (rather than mutating the borrow) keeps
                // upstream config-builders' assumptions intact; the cloned
                // value only escapes into `tier_b1::synthesize`.
                let mut cfg = config.clone();
                if let Some(c) = cfg.csha.as_mut() {
                    c.skip_rmsnorm_prologue = true;
                }
                return tier_b1::synthesize(&cfg, chunk);
            }
            Err(_reason) => {
                // chunk_config::select failed despite planner having
                // admitted L2.  Fall through to Tier A v2 path.  The
                // snapshot regression test on the emitted PTX will catch
                // any inadvertent fall-through in CI-monitored variants.
            }
        }
    }

    smem_layout::validate_scalar_v2_config(config, smem_layout::Direction::Forward)
        .expect("v2 emitter called with unsupported config -- selector must gate this");

    let mut ptx = String::new();

    // Phase 0: file header, param block, register decls, indices.
    // tier_b forwarded so the Tier B range-table preamble (if admitted) is
    // emitted after the Tier A segment_ids load + bar.sync 0.
    phases::prelude::emit(&mut ptx, config, tier_b);

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
            phases::s_compute::emit(&mut ptx, config, q_iter, tier_b);
            phases::softmax::emit(&mut ptx, config, q_iter);
            // PCA Tier B: KV_TILE_SKIP_TB_{q_iter} label is emitted here ONLY
            // when tier_b.is_some() — preserves byte-identical output for non-Tier-B
            // kernels (spec §3.4.6 no-op guarantee).
            if tier_b.is_some() {
                ptx.push_str(&format!("KV_TILE_SKIP_TB_{}:\n", q_iter));
            }
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

            // K/V-tile loop.
            ptx.push_str("    mov.u64 %k_start, 0;\n");
            ptx.push_str("    mov.u64 %k_max, %rd6;                        // seq_len\n");
            ptx.push_str(&format!("V2_LOOP_KV_START_{}:\n", q_iter));

            emit_k_tile_load(&mut ptx, config, q_iter);
            // Cycle 20 T3 BUG A FIX: Tier C QK save moved to AFTER
            // emit_k_tile_load. Previously the save ran BEFORE the K tile
            // load, reading uninitialised SMEM at %k_smem_base. Q is
            // loop-invariant (post-RoPE Q in q_offset is stable across KV
            // iterations), so re-emitting the save per KV iter is a wasted
            // HBM write but correctness-safe (same Q values). K SMEM is
            // now populated when the save reads it. SaveSet::V still runs
            // after v_tile_load below.
            phases::csha_hooks::emit_save_activations_subset(
                &mut ptx, config, q_iter, phases::csha_hooks::SaveSet::QK,
            );
            phases::s_compute::emit(&mut ptx, config, q_iter, tier_b);
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
            // Cycle 20 T3 BUG B FIX: inline V-save with K/V-indexed row
            // addressing (uses %k_start, not %q_start). Previously called
            // emit_save_activations_subset(SaveSet::V) which hard-codes
            // %q_start+warp_row row indexing — correct only at k_start=0.
            // This inline version covers all block_kv rows per KV-loop
            // iteration by looping (block_kv/4) rows per warp, and writes
            // to HBM row = k_start + warp_id + row_step*4. See BUG B in
            // 2026-07-01 T3 report.
            if config
                .csha
                .as_ref()
                .is_some_and(|c| c.save_activations_for_backward)
            {
                let head_dim = config.head_dim as u32;
                let slices_per_lane = (head_dim / 32).max(1);
                let kv_off = smem_layout::kv_offset(config);
                let rows_per_warp = ((config.block_kv as u32) / 4).max(1);
                ptx.push_str("    // -- Cycle 20 T3: inline V-save (k_start-indexed) --\n");
                // Init v_smem_base (non-fused: V aliases K at kv_offset).
                ptx.push_str(&format!(
                    "    add.u64 %v_smem_base, %shmem_base, {}; // T3: V tile at kv_offset\n",
                    kv_off
                ));
                ptx.push_str("    bar.sync 0;  // T3 FENCE: V SMEM tile complete\n");
                ptx.push_str("    ld.param.u64 %rd_save_base, [v_proj_ptr];\n");
                ptx.push_str("    setp.eq.u64 %p_save_null, %rd_save_base, 0;\n");
                ptx.push_str(&format!(
                    "    @%p_save_null bra V2_CSHA_SAVE_V_T3_SKIP_{};\n",
                    q_iter
                ));
                // batch*heads*seq_len base contribution (row-independent).
                ptx.push_str("    mul.lo.u64 %rd_save_bh, %batch_idx, %rd5;\n");
                ptx.push_str("    add.u64 %rd_save_bh, %rd_save_bh, %head_idx;\n");
                ptx.push_str("    mul.lo.u64 %rd_save_bh, %rd_save_bh, %rd6;\n");
                for row_step in 0..rows_per_warp {
                    // kv_row = warp_id + row_step*4 (row within block_kv tile)
                    ptx.push_str(&format!(
                        "    add.u32 %r_save_wrow, %warp_id, {}; // T3 kv_row (step {})\n",
                        row_step * 4,
                        row_step
                    ));
                    ptx.push_str("    cvt.u64.u32 %rd_save_wrow, %r_save_wrow;\n");
                    // HBM row = k_start + kv_row  (T3 BUG B FIX: was q_start)
                    ptx.push_str("    add.u64 %rd_save_off, %rd_save_bh, %k_start;\n");
                    ptx.push_str("    add.u64 %rd_save_off, %rd_save_off, %rd_save_wrow;\n");
                    ptx.push_str("    mul.lo.u64 %rd_save_off, %rd_save_off, %rd7;\n");
                    for slice in 0..slices_per_lane {
                        ptx.push_str("    cvt.u64.u32 %rd_save_col, %lane;\n");
                        if slices_per_lane > 1 {
                            ptx.push_str(&format!(
                                "    mul.lo.u64 %rd_save_col, %rd_save_col, {};\n",
                                slices_per_lane
                            ));
                        }
                        if slice > 0 {
                            ptx.push_str(&format!(
                                "    add.u64 %rd_save_col, %rd_save_col, {};\n",
                                slice
                            ));
                        }
                        ptx.push_str("    add.u64 %rd_save_elem, %rd_save_off, %rd_save_col;\n");
                        ptx.push_str("    shl.b64 %rd_save_elem, %rd_save_elem, 1;\n");
                        ptx.push_str("    add.u64 %rd_save_elem, %rd_save_base, %rd_save_elem;\n");
                        ptx.push_str(&format!(
                            "    mul.lo.u64 %rd_save_smem, %rd_save_wrow, {};\n",
                            head_dim * 2
                        ));
                        ptx.push_str(
                            "    add.u64 %rd_save_smem, %v_smem_base, %rd_save_smem;\n",
                        );
                        ptx.push_str("    shl.b64 %rd_save_colb, %rd_save_col, 1;\n");
                        ptx.push_str(
                            "    add.u64 %rd_save_smem, %rd_save_smem, %rd_save_colb;\n",
                        );
                        ptx.push_str("    ld.shared.b16 %h_save_v, [%rd_save_smem];\n");
                        ptx.push_str("    st.global.b16 [%rd_save_elem], %h_save_v;\n");
                    }
                }
                ptx.push_str(&format!("V2_CSHA_SAVE_V_T3_SKIP_{}:\n", q_iter));
            } else {
                ptx.push_str("    // CSHA Tier C V-save: save_activations=false, no emission\n");
            }
            phases::pv_accum::emit(&mut ptx, config, q_iter);

            // PCA Tier B: KV_TILE_SKIP_TB_{q_iter} label emitted ONLY when
            // tier_b.is_some() — preserves byte-identical output for non-Tier-B
            // kernels (spec §3.4.6 no-op guarantee).
            if tier_b.is_some() {
                ptx.push_str(&format!("KV_TILE_SKIP_TB_{}:\n", q_iter));
            }
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
    // §4.3 attention sinks (Sprint 1a precursor): the rolling load covers
    // `block_kv` rows; when sinks are enabled the sink pre-load below
    // covers an additional `num_sink_tokens` rows so the K SMEM slab
    // matches `effective_block_kv` rows that s_compute / softmax /
    // pv_accum / sp_bytes consume. At num_sink_tokens==0 this is
    // identical to `config.block_kv` and emits byte-identical PTX
    // (snapshot invariant).
    let rolling_k_elems = (config.block_kv as u32) * (config.head_dim as u32);
    let fused_k = config.csha.as_ref().is_some_and(|c| c.fused_projections);
    let kv_off = smem_layout::kv_offset(config);
    let n_sink = config.num_sink_tokens;
    let head_dim = config.head_dim as u32;
    // SMEM byte offset of the rolling K rows: skip past the sink slab if
    // sinks are enabled. At num_sink_tokens==0 this is `kv_off` —
    // byte-identical to pre-Sprint-1b.
    let rolling_kv_off = kv_off + n_sink * head_dim * 2;

    ptx.push_str("    // K tile load: 128 threads cooperatively load block_kv*head_dim elems\n");

    // §4.3 sinks (Sprint 1b cycle-7): emit sink K pre-load at the front
    // of the KV SMEM slab. Reads `num_sink_tokens * head_dim` f16 values
    // from `sink_k_ptr` (the producer materialises sinks at narrow
    // precision; see runtime FFI Task E for the null-ptr guard). Falls
    // through to the rolling K load below which writes to a shifted SMEM
    // offset so it doesn't stomp on the sinks. Skipped entirely when
    // num_sink_tokens==0 → byte-identical to pre-Sprint-1b.
    //
    // Sprint 2 cycle-8 §4.3 multi-tile: the sink-load is wrapped with a
    // PTX-level conditional gating on `%k_start == 0` so the sinks load
    // ONLY when the live K outer-loop iterator is at iter-0. For
    // subsequent iters within the SAME outer K loop, the sink rows
    // persist immutable in SMEM (the rolling K load writes into the
    // shifted `rolling_kv_off` slot which sits past the sink slab, so
    // the slab is never clobbered). The conditional itself is gated on
    // `num_sink_tokens > 0` at codegen time — at zero sinks NEITHER the
    // setp/bra NOR the load body emits → byte-identical PTX (cycle-7
    // Sprint 1a snapshot invariant).
    //
    // NOTE on fused-path semantics (cycle-8 holistic-review finding):
    // In the fused S/PV orchestrator, the K outer loop is RESTARTED at
    // each q_iter (`%k_start = 0` reset at the top of each q_iter's S
    // pass). This means `setp.ne %k_start, 0` fires false at iter-0 of
    // EVERY q_iter, so the sink K rows re-load once per q_iter rather
    // than once per kernel invocation. The data is idempotent (same f16
    // values written to the same SMEM slots), so this is a PERFORMANCE
    // regression for multi-q_iter configs (~num_sink_tokens × head_dim
    // wasted bytes per q_iter), NOT a correctness bug. A future cycle
    // could add a function-scope `%p_sinks_loaded` predicate to load
    // truly once per kernel. v1 accepts the per-q_iter reload as the
    // honest cost of the simplest correct lift.
    if n_sink > 0 {
        let sink_elems = n_sink * head_dim;
        ptx.push_str(&format!(
            "    // sinks v1: pre-load {n_sink} sink K rows from sink_k_ptr -> kv_offset (front of slab)\n"
        ));
        // Sprint 2 cycle-8 §4.3 multi-tile: sinks load ONLY when kv_iter == 0.
        // For subsequent iters, sink rows persist in SMEM from the first load.
        ptx.push_str("    setp.ne.u64 %p_skip_sinks, %k_start, 0;\n");
        ptx.push_str(&format!(
            "    @%p_skip_sinks bra V2_K_SINK_LOAD_SKIP_{};\n",
            q_iter
        ));
        ptx.push_str("    ld.param.u64 %rd_sink_base, [sink_k_ptr];\n");
        ptx.push_str("    cvt.u64.u32 %rd_sink_idx, %tid_x;\n");
        ptx.push_str(&format!("V2_LOOP_K_SINK_LOAD_{}:\n", q_iter));
        ptx.push_str(&format!(
            "    setp.lt.u64 %p_sink, %rd_sink_idx, {sink_elems};\n"
        ));
        ptx.push_str(&format!(
            "    @!%p_sink bra V2_K_SINK_LOAD_DONE_{};\n",
            q_iter
        ));
        // Sink table is f16 (2 bytes/elem) — matches sink_slab_bytes().
        ptx.push_str("    shl.b64 %rd_sink_off, %rd_sink_idx, 1;\n");
        ptx.push_str("    add.u64 %rd_sink_src, %rd_sink_base, %rd_sink_off;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd_sink_src];\n");
        // SMEM dest at kv_offset + idx * 2 (f16 stride).
        ptx.push_str(&format!(
            "    add.u64 %rd_sink_dst, %rd_sink_off, {kv_off};\n"
        ));
        ptx.push_str("    add.u64 %rd_sink_dst, %rd_sink_dst, %shmem_base;\n");
        ptx.push_str("    st.shared.b16 [%rd_sink_dst], %h0;\n");
        ptx.push_str("    add.u64 %rd_sink_idx, %rd_sink_idx, 128;\n");
        ptx.push_str(&format!(
            "    bra V2_LOOP_K_SINK_LOAD_{};\n",
            q_iter
        ));
        ptx.push_str(&format!("V2_K_SINK_LOAD_DONE_{}:\n", q_iter));
        ptx.push_str(&format!("V2_K_SINK_LOAD_SKIP_{}:\n", q_iter));
    }

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
    // GQA zero-copy stride (paper s4.2): when gqa_group_size > 1, divide
    // head_idx by the group size before composing the row index, so
    // multiple Q-heads alias to the same kv-head slot in the K tensor.
    // Byte-identical no-op when gqa_group_size == 1.
    if config.gqa_group_size > 1 {
        ptx.push_str(&format!(
            "    // GQA: kv_head = q_head / {} (paper s4.2 zero-copy)\n",
            config.gqa_group_size
        ));
        ptx.push_str(&format!(
            "    div.u64 %rd57, %head_idx, {};\n",
            config.gqa_group_size
        ));
        ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;      // batch*heads\n");
        ptx.push_str("    add.u64 %rd58, %rd58, %rd57;             // + kv_head\n");
    } else {
        ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;      // batch*heads\n");
        ptx.push_str("    add.u64 %rd58, %rd58, %head_idx;         // + head\n");
    }
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
    // Preserve pre-Sprint-1b comment text at num_sink_tokens=0 (snapshot
    // byte-identity invariant); add a sink-slab note only when sinks
    // shift the SMEM destination.
    if n_sink == 0 {
        ptx.push_str(&format!(
            "    add.u64 %rd60, %rd60, {};                 // + kv_offset\n",
            rolling_kv_off
        ));
    } else {
        ptx.push_str(&format!(
            "    add.u64 %rd60, %rd60, {};                 // + kv_offset + sink slab\n",
            rolling_kv_off
        ));
    }
    ptx.push_str("    add.u64 %smem_addr, %rd60, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd59, %rd59, 128;\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p0, %rd59, {};\n",
        rolling_k_elems
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
    // §4.3 attention sinks (Sprint 1a precursor): see emit_k_tile_load.
    // The rolling load covers `block_kv` rows; when sinks are enabled the
    // sink pre-load below covers an additional `num_sink_tokens` rows so
    // the V SMEM slab matches `effective_block_kv` rows. At
    // num_sink_tokens==0 emits byte-identical PTX to pre-Sprint-1b.
    let rolling_v_elems = (config.block_kv as u32) * (config.head_dim as u32);
    let fused_v = config.csha.as_ref().is_some_and(|c| c.fused_projections);
    let kv_off = smem_layout::kv_offset(config);
    let n_sink = config.num_sink_tokens;
    let head_dim = config.head_dim as u32;
    let rolling_kv_off = kv_off + n_sink * head_dim * 2;

    ptx.push_str("    // V tile load: cooperative, reuses K region\n");

    // §4.3 sinks (Sprint 1b cycle-7): emit sink V pre-load at the front
    // of the KV SMEM slab. Symmetric to the K sink pre-load. Skipped
    // entirely when num_sink_tokens==0 → byte-identical to pre-Sprint-1b.
    //
    // Sprint 2 cycle-8 §4.3 multi-tile: same kv_iter==0 gating as the K
    // sink pre-load — sinks load only on the first KV iteration and
    // persist across subsequent ones. Symmetric guard so the V slab
    // matches the K slab's "load once, persist" semantics.
    if n_sink > 0 {
        let sink_elems = n_sink * head_dim;
        ptx.push_str(&format!(
            "    // sinks v1: pre-load {n_sink} sink V rows from sink_v_ptr -> kv_offset (front of slab)\n"
        ));
        // Sprint 2 cycle-8 §4.3 multi-tile: sinks load ONLY when kv_iter == 0.
        ptx.push_str("    setp.ne.u64 %p_skip_sinks, %k_start, 0;\n");
        ptx.push_str(&format!(
            "    @%p_skip_sinks bra V2_V_SINK_LOAD_SKIP_{};\n",
            q_iter
        ));
        ptx.push_str("    ld.param.u64 %rd_sink_base, [sink_v_ptr];\n");
        ptx.push_str("    cvt.u64.u32 %rd_sink_idx, %tid_x;\n");
        ptx.push_str(&format!("V2_LOOP_V_SINK_LOAD_{}:\n", q_iter));
        ptx.push_str(&format!(
            "    setp.lt.u64 %p_sink, %rd_sink_idx, {sink_elems};\n"
        ));
        ptx.push_str(&format!(
            "    @!%p_sink bra V2_V_SINK_LOAD_DONE_{};\n",
            q_iter
        ));
        ptx.push_str("    shl.b64 %rd_sink_off, %rd_sink_idx, 1;\n");
        ptx.push_str("    add.u64 %rd_sink_src, %rd_sink_base, %rd_sink_off;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd_sink_src];\n");
        ptx.push_str(&format!(
            "    add.u64 %rd_sink_dst, %rd_sink_off, {kv_off};\n"
        ));
        ptx.push_str("    add.u64 %rd_sink_dst, %rd_sink_dst, %shmem_base;\n");
        ptx.push_str("    st.shared.b16 [%rd_sink_dst], %h0;\n");
        ptx.push_str("    add.u64 %rd_sink_idx, %rd_sink_idx, 128;\n");
        ptx.push_str(&format!(
            "    bra V2_LOOP_V_SINK_LOAD_{};\n",
            q_iter
        ));
        ptx.push_str(&format!("V2_V_SINK_LOAD_DONE_{}:\n", q_iter));
        ptx.push_str(&format!("V2_V_SINK_LOAD_SKIP_{}:\n", q_iter));
    }

    // When fused V projection is enabled, null-guard: skip if csha_wv_ptr != 0.
    if fused_v {
        ptx.push_str("    ld.param.u64 %rd_wv_chk, [csha_wv_ptr];\n");
        ptx.push_str("    setp.ne.u64 %p_wv_fused, %rd_wv_chk, 0;\n");
        ptx.push_str(&format!(
            "    @%p_wv_fused bra V2_V_LOAD_SKIP_{}; // V already in SMEM from projection\n",
            q_iter
        ));
    }

    // GQA zero-copy stride (paper s4.2): kv_head = q_head / gqa_group_size.
    // Byte-identical no-op when gqa_group_size == 1.
    if config.gqa_group_size > 1 {
        ptx.push_str(&format!(
            "    // GQA: kv_head = q_head / {} (paper s4.2 zero-copy)\n",
            config.gqa_group_size
        ));
        ptx.push_str(&format!(
            "    div.u64 %rd57, %head_idx, {};\n",
            config.gqa_group_size
        ));
        ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd58, %rd58, %rd57;             // + kv_head\n");
    } else {
        ptx.push_str("    mul.lo.u64 %rd58, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd58, %rd58, %head_idx;\n");
    }
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
    // Preserve pre-Sprint-1b emission verbatim at num_sink_tokens=0
    // (the V-load line had NO trailing comment originally); add a
    // sink-slab note only when sinks shift the SMEM destination.
    if n_sink == 0 {
        ptx.push_str(&format!(
            "    add.u64 %rd60, %rd60, {};\n",
            rolling_kv_off
        ));
    } else {
        ptx.push_str(&format!(
            "    add.u64 %rd60, %rd60, {};                 // + kv_offset + sink slab\n",
            rolling_kv_off
        ));
    }
    ptx.push_str("    add.u64 %smem_addr, %rd60, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd59, %rd59, 128;\n");
    ptx.push_str(&format!(
        "    setp.lt.u64 %p0, %rd59, {};\n",
        rolling_v_elems
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

/// Kernel entry-point name for v2. Same format as v1 with a `_v2`
/// suffix so module caches never collide between versions.
///
/// **Tier B.1 suffix**: when the dispatch criteria for the Tier B.1
/// pipelined-MMA path are met (`csha.level >= 2` AND `gpu_sm >= 80`)
/// AND `tier_b1::chunk_config::select` succeeds, the name is further
/// suffixed with `_tier_b1_chunk<N>` where `N` is the selected chunk
/// size. This carries TWO signals through to the runtime FFI:
///
///   1. Tier B.1 codegen requires 256 threads per CTA (8 warps for the
///      `global_t = warp_id + local_t * 8` distribution), while Tier A
///      v2 uses 128 threads (4 warps). The shared `nsl_flash_attention_csha`
///      FFI inspects the kernel-name string and picks block_x accordingly.
///   2. The chunk size is needed by the runtime pre-pass orchestration
///      (`nsl-runtime::cuda::tier_b1_prepass::launch_*`) to compute
///      output strides for the RMSNorm + narrow + chunkify pass on `x`
///      and the narrow + col-major chunkify pass on `Wq`/`Wk`/`Wv`.
///
/// Carrying both signals in the name (rather than as new FFI args) is
/// ABI-stable — Cranelift-emitted code already passes the name string
/// through; no call-site signature changes.
///
/// If `is_tier_b1_dispatch(config)` is true but `chunk_config::select`
/// fails for the config (e.g. SMEM budget overflow at all candidate
/// chunks), `synthesize_flash_attention_ptx_v2` falls through to the
/// Tier A v2 path. The name reflects that fall-through by omitting the
/// `_tier_b1_chunk<N>` suffix.
pub fn flash_attention_kernel_name_v2(config: &FlashAttentionConfig) -> String {
    let base = format!("{}_v2", crate::flash_attention::flash_attention_kernel_name(config));
    if is_tier_b1_dispatch(config) {
        match tier_b1::chunk_config::select(config) {
            Ok(chunk) => format!("{}_tier_b1_chunk{}", base, chunk),
            Err(_) => base, // fell through to Tier A v2 path
        }
    } else {
        base
    }
}

/// Returns `true` when `synthesize_flash_attention_ptx_v2` would
/// dispatch to `tier_b1::synthesize` for the given config — i.e. both
/// the CSHA pipelined level is requested AND the target GPU supports
/// the cp.async + m16n8k16 path. Mirrors the dispatch predicate in
/// `synthesize_flash_attention_ptx_v2` exactly.
///
/// Sprint 3 cycle-7 (§4.3 attention sinks v1): defense-in-depth refusal.
/// The Tier B.1 pipelined-MMA forward kernel does NOT understand the
/// persistent sink slab — neither `projection_mma::emit_kv_projection_chunk_loop`
/// nor `attention_mma::emit_phase_b_attention` reserves SMEM rows for
/// sinks, and the cp.async prologue kicks (`pipeline::emit_prologue_kicks`)
/// would not pre-load the sink rows. Returning `false` here forces the
/// dispatch to fall through to the Tier A v2 path, which Sprint 1b made
/// sink-aware for the narrow single-tile configuration. Lift point: when
/// a future sprint lands sink-aware Tier B.1 cp.async + MMA kernels.
pub fn is_tier_b1_dispatch(config: &FlashAttentionConfig) -> bool {
    config.csha.as_ref().is_some_and(|c| c.level >= 2)
        && config.gpu_sm >= 80
        && config.num_sink_tokens == 0
}

/// Central dispatch toggle for PCA Tier B PTX emission.
///
/// Returns `true` when Tier B's range-table preamble + tile-skip predicate
/// should be emitted for this FA config. Today returns `false`
/// unconditionally — Tier B is dormant infrastructure per PR #168. The
/// `nsl-codegen-bench` measurement harness overrides this gate by passing
/// `Some((seq_len, residency))` directly to
/// [`synthesize_flash_attention_ptx_v2_with_tier_b`], bypassing the
/// planner-side decision.
///
/// After the M2/M6 measurements land (task B1.5-6 of the 2026-05-13 plan),
/// this helper is updated per the §10 outcomes matrix of the design spec:
///   - `keep` outcome — returns `true` when `config.segment_masked` and the
///     fine-grained gate `pca_tilerange::should_emit_tier_b` admits.
///   - `keep-with-sparsity-gate` — adds a measured sparsity threshold.
///   - `revert` — stays at `false` (current state); the 6-month decay timer
///     starts on the day the findings doc commits.
///
/// This is the single source of truth so planner-dispatch callers see one
/// boolean, distinct from the fine-grained PTX-budget check in
/// [`crate::pca_tilerange::should_emit_tier_b`].
pub fn should_emit_tier_b(_config: &FlashAttentionConfig) -> bool {
    false
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
///
/// G16-2 fix: when `config.checkpoint = Some(Full)`, the kv_recompute path
/// writes to a recomputed x_norm scratch tile starting at
/// `smem_layout::recompute_xnorm_offset`, which sits immediately AFTER the
/// `backward_total_bytes` region. The launch must grant those extra bytes or
/// the store fires CUDA_ERROR_ILLEGAL_ADDRESS.
pub fn shared_mem_bytes_v2_backward(config: &FlashAttentionConfig) -> u32 {
    let seg_overhead = if config.segment_masked {
        crate::pca_segment::DEFAULT_SMEM_SEGMENT_BUDGET as u32
    } else {
        0
    };
    let recompute_extra = if config.checkpoint.is_some() {
        smem_layout::recompute_extra_bytes(config) as u32
    } else {
        0
    };
    phases::backward::prelude::backward_total_bytes(config) + seg_overhead + recompute_extra
}

/// SMEM byte count for a v2 forward kernel including Tier B contribution.
///
/// Called from launch sites that have access to `seq_len` and the Tier A
/// residency decision. No-op guarantee: when Tier B is not emitted, returns
/// exactly `shared_mem_bytes_v2(config)` — pre-Tier-B SMEM layout preserved
/// byte-identically for non-Tier-B configs (spec §3.4.6).
pub fn shared_mem_bytes_v2_with_seqlen(
    config: &FlashAttentionConfig,
    seq_len: u32,
    residency: crate::pca_segment::SegmentResidency,
) -> u32 {
    let tier_b_bytes = if crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency) {
        crate::pca_tilerange::tier_b_range_table_bytes(config, seq_len)
    } else {
        0
    };
    shared_mem_bytes_v2(config) + tier_b_bytes
}

/// Backward equivalent of `shared_mem_bytes_v2_with_seqlen`.
pub fn shared_mem_bytes_v2_backward_with_seqlen(
    config: &FlashAttentionConfig,
    seq_len: u32,
    residency: crate::pca_segment::SegmentResidency,
) -> u32 {
    let tier_b_bytes = if crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency) {
        crate::pca_tilerange::tier_b_range_table_bytes(config, seq_len)
    } else {
        0
    };
    shared_mem_bytes_v2_backward(config) + tier_b_bytes
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
    synthesize_backward_with_tier_b(config, None)
}

/// Tier-aware backward synthesizer. Consults the planner's backward
/// dispatch tier and routes to either the scalar v2 backward or (in
/// Phase 2+) the Tier B.2 MMA backward.
///
/// Phase 1 contract: Tier B.2 emitter returns `Err(NotImplemented)`,
/// so this function transparently falls back to scalar v2 in all cases
/// today. Phase 2 will land the real Tier B.2 emitter and this fallback
/// will only trigger for configs that fail the §6.4 SMEM ladder.
pub fn synthesize_backward_with_tier(
    config: &FlashAttentionConfig,
) -> Result<String, String> {
    use crate::csha_pipeline::backward_dispatch_tier;
    use crate::flash_attention_v2::tier_b2::BackwardTier;
    use crate::flash_attention_v2::tier_b2::backward::synthesize_tier_b2_backward;

    // Cycle-10 §5.3 Task 9 FIRST-CHECK: route to recompute path when
    // `@checkpoint(policy="full")` is active. Must run BEFORE the
    // existing sinks refusal so that R8.1 (checkpoint + sinks-v2
    // outside verified smoke set) can fire with its specific substring
    // instead of being shadowed by the generic Sprint-2 cycle-7
    // backward sinks refusal.
    if config.checkpoint.is_some() {
        return synthesize_backward_with_recompute(config);
    }

    // Sprint 2 cycle-7 defense-in-depth: refuse `num_sink_tokens > 0`
    // before dispatching to either tier. Both downstream paths
    // (`synthesize_backward` -> `synthesize_backward_with_tier_b`, and
    // `synthesize_tier_b2_backward`) would silently emit sink-unaware
    // gradients without this check. See `synthesize_backward_with_tier_b`
    // and `synthesize_backward_combined` for the same per-entry-point
    // refusal pattern.
    let (eligible, why) = sinks::attention_sinks_v1_backward_eligible(config);
    if !eligible {
        return Err(format!(
            "Sprint 2 cycle-7 backward sinks refusal at tier-dispatch entry: {}",
            why.unwrap_or("(unknown reason)")
        ));
    }

    match backward_dispatch_tier(config) {
        BackwardTier::TierB2 { .. } => {
            // Phase 1: stub returns NotImplemented; transparently fall
            // back. Phase 2: stub returns the real PTX; only the SMEM
            // ladder rejection path falls through to scalar.
            match synthesize_tier_b2_backward(config) {
                Ok(ptx) => Ok(ptx),
                Err(_) => synthesize_backward(config),
            }
        }
        BackwardTier::Scalar => synthesize_backward(config),
    }
}

/// Cycle-12 §1: backward synthesizer for `@checkpoint(policy="full")` —
/// R0 narrowed to R3/R11/R12 in cycle 12.
///
/// R0 narrowed to R3/R11/R12 in cycle 12: the unconditional R0 catch-all
/// refusal was retired; the cycle-11 structurally-validated PTX path is
/// now reachable in production for the cycle-12-eligible configuration
/// set (`level >= 1 && fused_rmsnorm && fused_projections`,
/// `!(segment_masked && rope_q)`, `!(segment_masked && checkpoint)`,
/// `num_sink_tokens == 0`, `!paged_kv_collision`, `block_q == block_kv`,
/// `!tier_b2_hybrid_backward_compile_time_eligible`). Configs outside
/// the eligible set must refuse with one of the specific R3/R7/R8.1/
/// R9/R10/R11/R12 messages — there is no catch-all in cycle 12.
///
/// Refusal cascade (cycle 12): R3 (augmented for fused_projections) →
/// R7 → R9 → R10 → R8.1 → R11 (Tier B.2 + checkpoint composition) →
/// R12 (segment_masked + checkpoint composition) → SMEM validator →
/// fall-through to `synthesize_backward_with_tier_b` for the
/// eligible-config functional emit path.
///
/// The cfg-gated `r0_bypass` field on `CheckpointExtras` remains for
/// cycle-11 test compatibility — it is now a no-op (test-helpers seam,
/// no longer needed for R0 lift). New tests should not consult it.
///
/// Byte-identity invariant (W11 / Gate 1): when `config.checkpoint.is_none()`
/// the dispatch fork in `synthesize_backward_with_tier` never reaches this
/// function, so no checkpoint-related PTX is emitted on the no-decorator
/// path — `fa_v2_snapshots` remains byte-identical.
/// Cycle-12 Phase F: cycle-5 invariant `feedback_deferral_must_refuse`
/// requires refusals to fire at EVERY entry point. Reviewer 1 (on
/// cycle-12 Phase D) found the R3/R11/R12 cascade lived ONLY inside
/// `synthesize_backward_with_recompute`, but production routes through
/// `synthesize_backward_combined` → `synthesize_backward` →
/// `synthesize_backward_with_tier_b(config, None)`, which NEVER calls
/// `synthesize_backward_with_recompute`. So R3/R11/R12 were functionally
/// dead under the production call chain.
///
/// This helper centralises the R3/R7/R8.1/R9/R10/R11/R12 sequence so
/// it can be called from BOTH the direct dispatch fork
/// (`synthesize_backward_with_recompute`) AND the production entry
/// (`synthesize_backward_with_tier_b`). When `config.checkpoint.is_none()`
/// the helper is a no-op — preserving the byte-identity invariant of
/// the no-decorator path (`fa_v2_snapshots` 25/25 byte-identical).
pub(crate) fn validate_checkpoint_eligibility(config: &FlashAttentionConfig) -> Result<(), String> {
    use crate::flash_attention::CheckpointPolicy;
    use crate::flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_compile_time_eligible;

    // No carrier ⇒ no checkpoint policy in play ⇒ nothing to gate.
    let Some(extras) = config.checkpoint.as_ref() else {
        return Ok(());
    };
    let CheckpointPolicy::Full = extras.policy;

    // ── R3 (cycle-12 augmented): Non-Level-1-fusible prologue ────────
    // Cycle 12 augmented R3 with the `fused_projections` requirement —
    // without staged x_raw + Wk/Wv on the forward path, the backward
    // kv-recompute would silently no-op (the x_raw_ptr null guard at
    // csha_hooks_backward.rs early-exits when x_raw is null, leaving
    // K/V SMEM tiles never written).
    let prologue_ok = config
        .csha
        .as_ref()
        .map(|c| c.level >= 1 && c.fused_rmsnorm && c.fused_projections)
        .unwrap_or(false);
    if !prologue_ok {
        return Err(
            "@checkpoint(policy=\"full\") requires RMSNorm + fused-projections + RoPE \
             prologue (Level-1 boundary-fusion chain with x_raw + Wk/Wv staged for \
             backward recompute) on the forward path; config has no fusible prologue \
             (level/fused_rmsnorm/fused_projections gate failed). Set \
             @csha(level=1, fused_rmsnorm=true, fused_projections=true) on the forward \
             function to enable kv-recompute backward."
                .to_string(),
        );
    }

    // ── R7: PCA packing with rope_q=true under @checkpoint ───────
    if config.segment_masked && config.rope_q {
        return Err(
            "PCA packing with rope_q=true under @checkpoint deferred to v4: \
             the segment-aware causal mask shares index machinery with the \
             RoPE-Q write-back path; safe composition requires the v4 \
             rotated-Q SMEM-staging refactor"
                .to_string(),
        );
    }

    // ── R9: @paged_kv model + @checkpoint fn composition ─────────
    if extras.paged_kv_collision {
        return Err(
            "@checkpoint composition with @paged_kv requires scatter-on-recompute; \
             deferred to v4: paged KV cache pages are not byte-stable across the \
             recompute boundary in v1"
                .to_string(),
        );
    }

    // ── R10: Asymmetric tile (block_q != block_kv) + @checkpoint ─
    if config.block_q != config.block_kv {
        return Err(format!(
            "checkpoint policy=\"full\" requires block_q == block_kv in v1; \
             asymmetric-tile composition deferred to v2 (got block_q={}, block_kv={})",
            config.block_q, config.block_kv
        ));
    }

    // ── R8.1: sinks-v2 + @checkpoint — UNCONDITIONAL in v1 ───────
    // The cycle-9 spec carved out an "unless (hd, S) is in verified
    // smoke set" exception, but Reviewer 1 found the carve-out
    // referenced `static_seq_len.unwrap_or(0)` which is ALWAYS 0 in
    // production (`CshaExtras::level1` does not set static_seq_len),
    // so the carve-out was structurally unreachable. Phase F drops
    // it and makes R8.1 honest about being unconditional. The
    // smoke-set carve-out is deferred to v2 (requires GPU
    // validation), and re-evaluating it requires plumbing
    // static_seq_len through the wire-up site first.
    if config.num_sink_tokens > 0 {
        return Err(
            "checkpoint policy=\"full\" + sinks-v2 composition refused in v1; \
             smoke-set carve-out deferred to v2 (requires GPU validation)"
                .to_string(),
        );
    }

    // ── R11 (cycle-12 new): Tier B.2 hybrid backward + @checkpoint ─
    // Tier B.2 hybrid four-kernel backward composes with @checkpoint
    // only via a coordinated dq/dkdv/d_prepass/proj recompute lift —
    // deferred to cycle 13. v1 must refuse rather than silently routing
    // the hybrid path through scalar-v2 kv-recompute (which would
    // produce wrong gradients for the dq/dkdv MMA tiles).
    if tier_b2_hybrid_backward_compile_time_eligible(config) {
        return Err(
            "@checkpoint(policy=\"full\") + Tier B.2 hybrid backward composition \
             deferred to cycle 13; configure for scalar-v2 backward or remove \
             @checkpoint"
                .to_string(),
        );
    }

    // ── R12 (cycle-12 new): segment_masked (PCA Tier-B) + @checkpoint ─
    // PCA Tier-B planner and kv-recompute are not co-emitted in v1; the
    // segment-aware tile-active predicate would gate recompute writes
    // unpredictably. Deferred to a co-emission lift in a later cycle.
    if config.segment_masked {
        return Err(
            "@checkpoint(policy=\"full\") + paged-segment-masked composition deferred; \
             PCA Tier-B planner and kv-recompute not co-emitted"
                .to_string(),
        );
    }

    Ok(())
}

pub fn synthesize_backward_with_recompute(
    config: &FlashAttentionConfig,
) -> Result<String, String> {
    // Carrier presence is guaranteed by the dispatch fork at
    // `synthesize_backward_with_tier`. If a caller hits this entry
    // point directly with `checkpoint: None`, refuse loudly rather
    // than silently routing to the no-decorator backward.
    if config.checkpoint.is_none() {
        return Err(
            "synthesize_backward_with_recompute called without \
             config.checkpoint set; v1 only services policy=\"full\""
                .to_string(),
        );
    }

    // Cycle-12 Phase F: centralised R3/R7/R8.1/R9/R10/R11/R12 cascade.
    // This is also called from `synthesize_backward_with_tier_b` so the
    // refusals fire from the production call chain too — closing the
    // cycle-5 invariant gap Reviewer 1 caught.
    validate_checkpoint_eligibility(config)?;

    // SMEM budget extension (Cut 7 / R5): re-run the scalar v2 validator
    // with the checkpoint carrier in place so `recompute_extra_bytes` is
    // counted.
    smem_layout::validate_scalar_v2_config(config, smem_layout::Direction::Backward)
        .map_err(|e| format!("backward validator (checkpoint=Full) rejected: {e}"))?;

    // Cycle-12 functional emission landing point — R0 retired. The
    // refusal cascade above narrows to the eligible-config subset; this
    // call routes into the tier-b scalar synthesizer whose
    // `checkpoint.is_some()` dispatch fork at the kv_load call site
    // swaps the HBM-load emitters for `emit_kv_recompute`, which writes
    // recomputed K/V into `%k_smem_base` / `%v_smem_base` for the
    // downstream `ds_compute` / `dv_accum` / `dqdk_accum` consumers.
    // GPU numerical validation (G4: atol=5e-4 rtol=5e-3 at hd in
    // {64,128}, S in {512, 2048, 4096}) runs in cycle 13 (cuda-gated
    // harness).
    synthesize_backward_with_tier_b(config, None)
}

/// Sprint 1 T1.4: emit a single PTX module containing BOTH the scalar
/// Tier C backward AND the Tier B.2 hybrid four-kernel backward when the
/// config is compile-time eligible for the hybrid path. Otherwise return
/// the scalar PTX unchanged.
///
/// The runtime FFI (`csha_tier_b2_backward_launch`) branches on
/// `tier_b2_active` (computed at Wengert lowering, Sprint 1 T1.3) to pick
/// either the scalar single-entry path or the four-kernel hybrid path.
/// Both entries must live in the same loaded module so a single
/// `cuModuleLoadData` + per-path `cuModuleGetFunction` lookup work; this
/// function is what production codegen at `compiler/kernel.rs:822` calls
/// to embed that combined module.
///
/// Header-union convention mirrors `synthesize_tier_b2_backward`:
///   * Single `.version 8.7 / .target sm_80 / .address_size 64` (sm_80
///     is mandatory for the dq/dkdv MMA path; sm_75-targeted bodies run
///     fine on sm_80+ via JIT).
///   * Single `.extern .shared .align 16 .b8 shmem[];` module-level
///     declaration. Both the scalar backward (when its SMEM exceeds the
///     48 KB static cap) and the hybrid kernels reference `shmem` via
///     this extern; one extern is sufficient. Static `.shared` decls
///     emitted INSIDE individual entry bodies are function-scoped and
///     shadow the extern locally — no symbol collision.
///   * Each component's body (everything from `.visible .entry`
///     onward) is concatenated verbatim via `strip_module_header`.
///
/// **No-op guarantee**: for ineligible configs (head_dim not
/// hybrid-compatible, csha.level<2, gpu_sm<80, rope_q=true,
/// active_heads!=1, d_model!=head_dim, no CSHA, ...), the output is
/// byte-identical to `synthesize_backward(config)`.
pub fn synthesize_backward_combined(
    config: &FlashAttentionConfig,
) -> Result<String, String> {
    use crate::flash_attention_v2::tier_b2::backward::{
        strip_module_header, synthesize_tier_b2_backward,
    };
    use crate::flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_compile_time_eligible;

    // Sprint 2 cycle-7 defense-in-depth: refuse `num_sink_tokens > 0`
    // at the combined-module entry too. `synthesize_backward` (called
    // immediately below) already refuses via `synthesize_backward_with_tier_b`,
    // but this explicit check keeps the per-entry-point refusal contract
    // visible at the hybrid path's front door. Otherwise, a future
    // refactor that bypasses `synthesize_backward` (e.g., directly
    // synthesizing only the hybrid four-kernel module) could silently
    // reintroduce the silent-correctness gap.
    let (eligible, why) = sinks::attention_sinks_v1_backward_eligible(config);
    if !eligible {
        return Err(format!(
            "Sprint 2 cycle-7 backward sinks refusal at combined entry: {}",
            why.unwrap_or("(unknown reason)")
        ));
    }

    // Always synthesize the scalar backward first — it is the fallback
    // path the runtime takes when `tier_b2_active=0`, and we must
    // preserve its validator-rejection behaviour (the caller in
    // `compiler/kernel.rs` swallows the Err to fall back to the legacy
    // tape-op backward).
    let scalar_ptx = synthesize_backward(config)?;

    if !tier_b2_hybrid_backward_compile_time_eligible(config) {
        // Ineligible: byte-identical to scalar. Runtime will never set
        // `tier_b2_active=1` for this config, so the hybrid entries
        // would be dead weight.
        return Ok(scalar_ptx);
    }

    // Eligible: synthesize the hybrid four-kernel module and union the
    // headers. If hybrid synthesis fails (e.g. UnsupportedHeadDim that
    // the compile-time predicate didn't catch — should be unreachable
    // today but defensive), fall back to scalar so the runtime still
    // has SOMETHING to launch.
    let hybrid_ptx = match synthesize_tier_b2_backward(config) {
        Ok(p) => p,
        Err(_) => return Ok(scalar_ptx),
    };

    let mut combined = String::new();
    // Single union header. `.version 8.7` is the highest any component
    // emits; `.target sm_80` is required by the Tier B.2 MMA kernels
    // and JIT-runs the sm_75-targeted scalar body fine. One shmem
    // extern serves all entries that reference dynamic SMEM.
    combined.push_str(".version 8.7\n");
    combined.push_str(".target sm_80\n");
    combined.push_str(".address_size 64\n\n");
    combined.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
    combined.push_str(strip_module_header(&scalar_ptx));
    combined.push_str("\n\n");
    combined.push_str(strip_module_header(&hybrid_ptx));
    Ok(combined)
}

/// Forward module for CSHA fused-projection configs with a multi-tile
/// attention twin appended.
///
/// The fused-projection forward orchestrator is single-tile by
/// construction twice over: (1) the K/V pre-passes project only the
/// CTA's own block rows, and (2) the split S-pass/PV-pass materializes
/// P per KV tile in SMEM, so a second KV tile clobbers the first's P
/// before the PV pass reads it. Multi-tile launches therefore need the
/// classic INTERLEAVED orchestration (the `else` branch: per-tile
/// S+softmax+PV with k_start-addressed HBM loads).
///
/// This synthesizer emits BOTH into one module:
///   * the fused kernel, byte-identical to
///     `synthesize_flash_attention_ptx_v2(config)` (used stand-alone or
///     as "launch A" of the runtime's two-launch multi-tile dispatch —
///     its per-block projection saves are valid at any seq_len);
///   * an attention "twin": same config with `fused_rmsnorm=false`,
///     `fused_projections=false`, `rope_q=false`, `level=1` (dodges the
///     Tier B.1 pipelined dispatch at `gpu_sm >= 80`), renamed to
///     `<fused_name>_mt_attn`. The runtime's launch B feeds it the
///     f32-widened projection saves via q/k/v (classic loads apply no
///     RMSNorm/projection/RoPE — the inputs are already projected and
///     rotated) and its else-branch `emit_save_softmax_state` writes the
///     row_max/row_sum saves the backward needs.
///
/// Configs the twin cannot serve (sinks, segment_masked, checkpoint,
/// rope_q) return the fused module unchanged — the runtime's multi-tile
/// dispatch then fails loudly at kernel-name resolution instead of
/// producing silent garbage.
///
/// Phase 1.1 SAFETY (2026-07-08): `rope_q=true` is refused here. RoPE is
/// applied in launch A (the fused per-tile kernel) via `emit_rope_pair_sweep`,
/// which indexes cos/sin by the TILE-LOCAL row under a single-tile assumption
/// (see forward/csha_hooks.rs:1519-1524). At multi-tile every tile past
/// position 0 is mis-rotated, so the saved rotated projections are wrong and
/// the whole forward is garbage (GPU-verified: forward parity 4.5-9.8 at
/// rope_q=true multi-tile vs 3e-3 at rope_q=false). Refuse (no twin -> runtime
/// rc=500) until the launch-A rope epilogue is reworked to use GLOBAL tile
/// positions (q_start for Q, k_start for K). Single-tile rope is unaffected
/// (it uses the fused entry directly, never the twin).
///
/// Output is NUL-terminated like `synthesize_flash_attention_ptx_v2`.
pub fn synthesize_forward_multi_tile_combined(config: &FlashAttentionConfig) -> Vec<u8> {
    let fused = synthesize_flash_attention_ptx_v2(config);

    let is_fused_proj = config
        .csha
        .as_ref()
        .is_some_and(|c| c.fused_projections);
    if !is_fused_proj
        || config.num_sink_tokens > 0
        || config.segment_masked
        || config.checkpoint.is_some()
        || config.paged
        || config.rope_q
    {
        return fused;
    }

    let mut twin_cfg = config.clone();
    twin_cfg.rope_q = false;
    if let Some(c) = twin_cfg.csha.as_mut() {
        c.fused_rmsnorm = false;
        c.fused_projections = false;
        c.level = 1;
    }
    let twin = synthesize_flash_attention_ptx_v2(&twin_cfg);

    // Strip trailing NULs so string ops below see clean PTX text.
    let to_str = |v: &[u8]| -> String {
        let end = v.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
        String::from_utf8_lossy(&v[..end]).into_owned()
    };
    let fused_txt = to_str(&fused);
    let twin_txt = to_str(&twin);

    let fused_name = flash_attention_kernel_name_v2(config);
    let twin_name = flash_attention_kernel_name_v2(&twin_cfg);
    let mt_name = format!("{fused_name}_mt_attn");
    let twin_renamed = twin_txt.replace(&twin_name, &mt_name);

    use crate::flash_attention_v2::tier_b2::backward::strip_module_header;
    let mut combined = String::new();
    // Union header: match the fused component's own target choice
    // (prelude emits sm_80 when gpu_sm >= 80, else sm_75).
    combined.push_str(".version 8.7\n");
    if config.gpu_sm >= 80 {
        combined.push_str(".target sm_80\n");
    } else {
        combined.push_str(".target sm_75\n");
    }
    combined.push_str(".address_size 64\n\n");
    // One module-level extern serves any entry that references dynamic
    // SMEM; entries with static `.shared` decls shadow it locally
    // (same convention as `synthesize_backward_combined`).
    combined.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
    combined.push_str(strip_module_header(&fused_txt));
    combined.push_str("\n\n");
    combined.push_str(strip_module_header(&twin_renamed));
    combined.push('\n');

    let mut bytes = combined.into_bytes();
    bytes.push(0);
    bytes
}

/// Tier C backward orchestrator with optional PCA Tier B.2 support.
///
/// When `tier_b` is `None` this produces byte-identical output to
/// `synthesize_backward(config)` (Tier B.2 no-op guarantee, spec §7.4).
/// When `tier_b` is `Some((seq_len, residency))` and `should_emit_tier_b`
/// returns true, the emitted PTX includes:
///   1. Range-table preamble in the backward prelude (after Tier A bar.sync 0).
///   2. Per-(q_iter, kvt) skip predicate at the head of each q_iter body
///      inside `V2_BWD_LOOP_KV` (KV-outer / Q-inner per V-B.2-predicate
///      case (β)).
///   3. `BWD_KV_TILE_SKIP_TB_{q_iter}:` labels at the end of each q_iter body.
///
/// Symmetric correctness (spec §7.1): skipped tiles produce P=0 ⇒ dS=0;
/// no contribution to dQ/dK/dV. dV/dK SMEM tiles are zero-initialised at the
/// top of the kv-outer loop, so a skipped (q_iter, kvt) sees a no-op RMW.
/// dQ is zero-initialised once at the top of the kernel and persists across
/// kv-iters; skipped tiles leave it unchanged.
pub fn synthesize_backward_with_tier_b(
    config: &FlashAttentionConfig,
    tier_b: Option<(u32, SegmentResidency)>,
) -> Result<String, String> {
    // Cycle-12 Phase F (Reviewer 1 NEEDS_FIXES): refuse R3/R7/R8.1/R9/
    // R10/R11/R12 from the production entry too. Production routes
    // `synthesize_backward_combined` → `synthesize_backward` → here
    // WITHOUT going through `synthesize_backward_with_recompute`, so
    // without this call the cycle-12 refusal cascade would be
    // unreachable from the production call chain — violating the
    // cycle-5 invariant `feedback_deferral_must_refuse`. When
    // `config.checkpoint.is_none()` this is a no-op, preserving the
    // `fa_v2_snapshots` 25/25 byte-identity invariant for the
    // no-decorator path.
    validate_checkpoint_eligibility(config)?;

    // Sprint 2 cycle-7 defense-in-depth: refuse `num_sink_tokens > 0`.
    // The forward eligibility predicate already refuses
    // `csha.save_activations_for_backward = true` + `num_sink_tokens > 0`
    // at the compiler/kernel.rs front door, but the cycle-5
    // `feedback_deferral_must_refuse` invariant says refusals must
    // be at EVERY entry point — not just the decorator extraction
    // site. A caller that bypasses kernel.rs (e.g., direct codegen
    // call in a test that constructs FlashAttentionConfig by hand)
    // would otherwise silently emit backward PTX that does NOT
    // understand the persistent sink slab — producing wrong
    // gradients with no error. Lift point: when a future v2 sprint
    // lands sink-aware dQ/dK/dV kernels, update
    // `sinks::attention_sinks_v1_backward_eligible` to allow it.
    let (eligible, why) = sinks::attention_sinks_v1_backward_eligible(config);
    if !eligible {
        return Err(format!(
            "Sprint 2 cycle-7 backward sinks refusal: {}",
            why.unwrap_or("(unknown reason)")
        ));
    }

    smem_layout::validate_scalar_v2_config(config, smem_layout::Direction::Backward)
        .map_err(|e| format!("backward validator rejected: {e}"))?;

    let mut ptx = String::new();

    // Phase 0: header, .visible .entry, SMEM, register pool, indices.
    // tier_b forwarded so the Tier B range-table preamble (if admitted) is
    // emitted after the Tier A segment_ids load + bar.sync 0.
    phases::backward::prelude::emit(&mut ptx, config, tier_b);

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
    // PCA Tier B.2: derive the kv-tile ordinal once per kv-outer iter and
    // hoist it into a register that lives until the bottom of the kv-loop.
    // Each per-q_iter skip predicate (below) reads this for `kvt_reg`.
    // %k_start is a u64 induction var written at the top of the kv-loop by
    // the prior iteration's add.u64; block_kv is a power-of-2 tile size, so
    // shr.b64 by log2(block_kv) yields the kv-tile ordinal directly.
    let tier_b_active = tier_b
        .map(|(seq_len, residency)| {
            crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency)
        })
        .unwrap_or(false);
    let log2_bkv = (config.block_kv as u32).trailing_zeros();
    let log2_bq = (config.block_q as u32).trailing_zeros();
    if tier_b_active {
        ptx.push_str("    { // PCA Tier B.2: kv-tile ordinal scope (per kv-outer iter)\n");
        ptx.push_str("    .reg .u64 %rd_kvt_ord_TB_BWD;\n");
        ptx.push_str("    .reg .u32 %r_kvt_ord_TB_BWD;\n");
        ptx.push_str(&format!(
            "    shr.b64 %rd_kvt_ord_TB_BWD, %k_start, {log2_bkv};\n"
        ));
        ptx.push_str("    cvt.u32.u64 %r_kvt_ord_TB_BWD, %rd_kvt_ord_TB_BWD;\n");
        // B2-2.5: q-tile ordinal derived from %q_start, not %bid_x. Backward
        // launches with `grid_x = 1` in production (q-block encoded in
        // `seq_lens_ptr` → `%q_launch_base` → `%q_start`); reading `%bid_x`
        // here would always yield 0 and the predicate would see q-tile 0's
        // segment range for every CTA. Computing qt = q_start >> log2(bq)
        // is correct under BOTH launch ABIs:
        //   grid_x = num_q_tiles (legacy bench): q_start = bid_x * bq → qt = bid_x.
        //   grid_x = 1            (production):   q_start = q_launch_base → qt = q_block.
        // qt is invariant across the kv-outer loop (q_start written once in
        // prelude), so hoisting alongside %r_kvt_ord_TB_BWD is correct.
        ptx.push_str("    .reg .u64 %rd_qt_ord_TB_BWD;\n");
        ptx.push_str("    .reg .u32 %r_qt_ord_TB_BWD;\n");
        ptx.push_str(&format!(
            "    shr.b64 %rd_qt_ord_TB_BWD, %q_start, {log2_bq};\n"
        ));
        ptx.push_str("    cvt.u32.u64 %r_qt_ord_TB_BWD, %rd_qt_ord_TB_BWD;\n");
    }
    // Cycle-11 Task 4 dispatch fork: when `@checkpoint` decorated the
    // enclosing fn, the kv_load emitters (which expect K/V to be already
    // resident in HBM) are replaced with `emit_kv_recompute`, which
    // re-derives K/V into `%k_smem_base`/`%v_smem_base` from saved
    // x_raw_ptr + W_k/W_v projection weights. R0 still guards production
    // (lifted only via the cfg-gated `bypass_r0_for_testing()` seam);
    // this fork is reached only when R0 is bypassed under test.
    if config.checkpoint.is_some() {
        phases::backward::csha_hooks_backward::emit_kv_recompute(&mut ptx, config, "MAIN");
    } else {
        phases::backward::kv_load::emit_k_suffixed(&mut ptx, config, "MAIN");
        phases::backward::kv_load::emit_v_suffixed(&mut ptx, config, "MAIN");
    }
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

        // PCA Tier B.2 skip predicate — fires once per (q_iter, kvt) at the
        // head of this q_iter body. When ranges are disjoint the predicate
        // branches to BWD_KV_TILE_SKIP_TB_{q_iter}_{kvt-from-loop}, placed at
        // the END of this q_iter body (after the dQ flush). The skip is
        // SYMMETRIC-ZERO safe (spec §7.1): P=0 ⇒ dV=P^T·dO=0, dP=dO·V^T=0,
        // dS=P⊙(dP-D)=0 ⇒ no contribution to dQ/dK/dV for this (qt, kvt).
        // The dV/dK SMEM tiles were zero-initialised at the top of THIS kv
        // iter; the dQ register reload+flush around the skip is a load-store
        // of unchanged data (idempotent overwrite).
        //
        // The label includes `q_iter` so different inner q_iters get distinct
        // PTX labels — required because we still want each q_iter to make an
        // independent skip decision (qt is the same for all q_iters in a CTA
        // — see B2-2.5: qt = %q_start >> log2(block_q) — but kvt is the same
        // for all q_iters too, so the predicate result is actually loop-
        // invariant across q_iters; we still emit per-q_iter labels for
        // ptxas's BRA.U inference and to avoid label namespace collisions).
        if tier_b_active {
            if let Some((seq_len, _residency)) = tier_b {
                let range_table_base =
                    crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
                        config,
                        crate::flash_attention_v2::smem_layout::Direction::Backward,
                    );
                let skip_label = format!("BWD_KV_TILE_SKIP_TB_{q_iter}");
                ptx.push_str(&format!(
                    "    {{ // PCA Tier B.2 per-q_iter skip-predicate scope (q_iter={q_iter})\n"
                ));
                crate::pca_tilerange::emit_skip_predicate(
                    &mut ptx,
                    config,
                    seq_len,
                    // B2-2.5: %r_qt_ord_TB_BWD = (%q_start >> log2(block_q)).
                    // Correct under BOTH grid_x=num_q_tiles (bench legacy) and
                    // grid_x=1 (production, q-block in seq_lens_ptr).
                    "%r_qt_ord_TB_BWD",
                    "%r_kvt_ord_TB_BWD",
                    range_table_base,
                    &skip_label,
                    crate::pca_tilerange::IterationOrder::KVOuter,
                );
                ptx.push_str("    } // end per-q_iter skip-predicate scope\n");
            }
        }

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

        // PCA Tier B.2 skip-predicate target — placed at the END of this
        // q_iter body so a skip branches past ds_compute/dv_accum/dqdk_accum
        // AND past the dQ flush. Skipped tiles produce S=0 ⇒ dS=0, and the
        // dQ register contents are unchanged from the reload above, so even
        // if we had flushed them back, the SMEM contents would be identical.
        // Per-q_iter label namespace is `BWD_KV_TILE_SKIP_TB_{q_iter}` —
        // distinct from forward's `KV_TILE_SKIP_TB_{q_iter}` namespace.
        if tier_b_active {
            ptx.push_str(&format!("BWD_KV_TILE_SKIP_TB_{q_iter}:\n"));
        }
    }
    if tier_b_active {
        ptx.push_str("    } // end PCA Tier B.2 kv-tile ordinal scope\n");
    }
    phases::backward::finalize::emit_store_kv_only(&mut ptx, config, 0);
    // Cycle 17 G16-1 T2: in-loop dK inverse-RoPE + dK HBM store. Cycle 16
    // emitted these POST-LOOP via emit_drope(Both) + emit_store_dk_only.
    // The post-loop emission was empirically refuted (commit c2c6879b):
    // emit_store_dk_only's predicate `(row + %k_start) < seq_len` (see
    // finalize.rs:115-116) evaluates FALSE for every row when %k_start
    // has been incremented past seq_len at loop exit — the store is
    // dead-code-masked and Path A is byte-identical to cycle 15.
    //
    // T2 fix: rotate+store dK INSIDE the loop where %k_start is the
    // current KV tile's start (in-range per iter). H2 (per-iter dK SMEM
    // zero-init) is structurally correct given in-loop store: each iter
    // zeros dK SMEM, accumulates ONE tile via dqdk_accum, rotates via
    // in-loop emit_drope_branch(K), stores to HBM via in-loop
    // emit_store_dk_only, then next iter zeros again. Q branch stays
    // post-loop because dQ uses cycle-15-correct flush-and-reload that
    // persists dQ SMEM across all KV iters.
    //
    // bar.sync after emit_store_kv_only (above, embedded in finalize.rs
    // line 206) ensures dV SMEM writes complete before in-loop emit_drope
    // (K) reads the adjacent dK SMEM region.
    phases::backward::csha_hooks_backward::emit_drope_branch(
        &mut ptx,
        config,
        0,
        phases::backward::csha_hooks_backward::DropeBranch::K,
    );
    phases::backward::finalize::emit_store_dk_only(&mut ptx, config, 0);
    ptx.push_str(&format!("    add.u64 %k_start, %k_start, {};\n", config.block_kv));
    ptx.push_str("    setp.lt.u64 %p0, %k_start, %k_max;\n");
    ptx.push_str("    @%p0 bra V2_BWD_LOOP_KV;\n");
    ptx.push_str("    bar.sync 0;  // dQ SMEM tile complete across all KV tiles\n");

    // Phase 3: CSHA hooks (x_norm recompute, inverse RoPE Q-branch only,
    // dW{q,k,v}, dRMSNorm). Each writes directly to HBM except the dRoPE
    // rotation which mutates the dQ SMEM tile in place. Cycle 17 T2: K
    // branch moved in-loop (above); Q branch stays here.
    phases::backward::csha_hooks_backward::emit_xnorm_recompute(&mut ptx, config);
    phases::backward::csha_hooks_backward::emit_drope_branch(
        &mut ptx,
        config,
        0,
        phases::backward::csha_hooks_backward::DropeBranch::Q,
    );
    phases::backward::csha_hooks_backward::emit_dproj(&mut ptx, config, 0);
    phases::backward::csha_hooks_backward::emit_drmsnorm(&mut ptx, config, 0);

    // Phase 4: cooperative global stores of the remaining gradients +
    // final fence. Cycle 17 T2: emit_store_dk_only moved in-loop (above).
    // Here we only emit dQ; dV is already stored via emit_store_kv_only
    // (in-loop f32 scratch RMW); dwq/dwk/dwv/dx are hook-managed.
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
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model,
                ..CshaExtras::default()
            }),
            checkpoint: None,
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
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: None,
            checkpoint: None,
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

    #[test]
    fn synthesize_backward_with_tier_falls_back_to_scalar_in_phase1() {
        use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
        // Small config so BOTH Tier B.2 dispatch (which triggers on
        // hd ∈ {64, 128, 256} + level=2 + sm=80) AND the scalar v2
        // backward fallback can accept it.
        //
        // Why not the canonical (bq=bkv=64, hd=128) config? Because
        // scalar v2 backward's `backward_extra_bytes` at hd=128 alone
        // exceeds the 99 KB SMEM cap (P+dS+dQ+dK+dV+V_in ~= 144 KB).
        // Scalar v2 backward is therefore structurally incapable of
        // handling hd=128 — which is precisely why Tier B.2 is the
        // ONLY viable backward for production-scale configs, not an
        // optimization. The fallback wrapper logic still needs to be
        // exercised, so we use the smallest config that both paths
        // accept.
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 64,
            causal: true, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
            checkpoint: None,
        };
        let result = synthesize_backward_with_tier(&cfg);
        // Phase 1: tier_b2 emitter is a stub; the wrapper falls back
        // to scalar v2. Scalar v2 should emit a non-empty PTX string
        // for this small config.
        let ptx = result.expect("scalar v2 backward should accept small config");
        assert!(
            ptx.contains(".visible .entry"),
            "fallback PTX should be a valid kernel"
        );
    }

    // -----------------------------------------------------------------
    // Sprint 5 — GQA zero-copy stride pattern (paper s4.2) for the
    // Tier A / v2 forward emitter. The K-tile load (emit_k_tile_load)
    // and V-tile load (emit_v_tile_load) at mod.rs must emit a
    // compile-time-literal `div.u64 %rd57, %head_idx, N` when
    // gqa_group_size > 1 and substitute %rd57 for %head_idx in the row
    // index composition. gqa_group_size==1 must remain byte-identical
    // to the pre-Sprint-5 baseline.
    // -----------------------------------------------------------------

    fn fwd_v2_cfg(gqa: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: gqa, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: None,
            checkpoint: None,
        }
    }

    #[test]
    fn v2_forward_emits_gqa_divisor_when_group_size_gt_one() {
        let cfg = fwd_v2_cfg(4);
        let ptx_bytes = synthesize_flash_attention_ptx_v2(&cfg);
        let ptx = String::from_utf8_lossy(
            &ptx_bytes[..ptx_bytes.len().saturating_sub(1)]
        ).into_owned();

        // The forward K-tile and V-tile loads each emit one div.u64
        // for the GQA group size literal.
        let div_count = ptx.matches("div.u64 %rd57, %head_idx, 4;").count();
        assert!(
            div_count >= 2,
            "forward must emit >= 2 GQA div.u64 emissions (K-load + V-load), got {div_count}.\nPTX excerpt around `div.u64`:\n{}",
            ptx.lines().filter(|l| l.contains("div.u64") || l.contains("head_idx")).collect::<Vec<_>>().join("\n")
        );

        // The kv-head intermediate %rd57 must reach the row-index chain
        // in both K-load and V-load.
        let kvh_uses = ptx.matches("add.u64 %rd58, %rd58, %rd57;").count();
        assert!(
            kvh_uses >= 2,
            "kv_head register %rd57 must feed both K-load and V-load row indices; got {kvh_uses}"
        );
    }

    #[test]
    fn v2_forward_byte_identical_at_gqa_one() {
        let cfg = fwd_v2_cfg(1);
        let ptx_bytes = synthesize_flash_attention_ptx_v2(&cfg);
        let ptx = String::from_utf8_lossy(
            &ptx_bytes[..ptx_bytes.len().saturating_sub(1)]
        ).into_owned();

        // No GQA div.u64 must be emitted with the %rd57 destination
        // and the gqa-paper s4.2 comment must be absent at gqa=1.
        assert!(
            !ptx.contains("div.u64 %rd57, %head_idx"),
            "gqa_group_size=1 must NOT emit forward GQA div.u64 (byte-identity)"
        );
        assert!(
            !ptx.contains("paper s4.2 zero-copy"),
            "gqa_group_size=1 must NOT emit the s4.2 zero-copy annotation"
        );
        // The original Q-head wiring must still be present (proves we
        // took the no-op branch, not a third branch that silently broke).
        assert!(
            ptx.contains("add.u64 %rd58, %rd58, %head_idx;         // + head"),
            "gqa_group_size=1 K-load must wire %head_idx directly"
        );
        assert!(
            ptx.contains("add.u64 %rd58, %rd58, %head_idx;\n"),
            "gqa_group_size=1 V-load must wire %head_idx directly"
        );
    }
}

#[cfg(test)]
mod sinks_tier_b1_dispatch_tests {
    //! Sprint 3 cycle-7 (§4.3 attention sinks v1): unit tests for the
    //! Tier B.1 dispatch refusal under `num_sink_tokens > 0`. Mirrors the
    //! Sprint 2 `tier_b2_hybrid_backward_eligible` sinks refusal pattern.
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn tier_b1_otherwise_eligible_cfg() -> FlashAttentionConfig {
        // Otherwise-eligible: csha.level=2 + gpu_sm=120 (Blackwell). The
        // ONLY axis we mutate per test is `num_sink_tokens` so the refusal
        // proof is sinks-axis-only.
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: true, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
            gpu_sm: 120, segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: 2048,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        }
    }

    #[test]
    fn is_tier_b1_dispatch_true_when_otherwise_eligible_and_zero_sinks() {
        // Proves the test config is on the "happy path" of dispatch so
        // the sinks-axis refusal below is the load-bearing axis.
        let cfg = tier_b1_otherwise_eligible_cfg();
        assert!(
            is_tier_b1_dispatch(&cfg),
            "otherwise-eligible config with num_sink_tokens=0 must dispatch to Tier B.1"
        );
    }

    #[test]
    fn is_tier_b1_dispatch_false_when_sinks_enabled() {
        // The Tier B.1 pipelined-MMA forward kernel does not understand
        // the persistent sink slab — the predicate must return `false`
        // so dispatch falls through to Tier A v2 (sink-aware per
        // Sprint 1b). cycle-5 `feedback_deferral_must_refuse` invariant.
        let mut cfg = tier_b1_otherwise_eligible_cfg();
        cfg.num_sink_tokens = 4;
        assert!(
            !is_tier_b1_dispatch(&cfg),
            "Tier B.1 dispatch must refuse num_sink_tokens > 0 (defense-in-depth)"
        );
    }

    #[test]
    fn is_tier_b1_dispatch_false_when_level_below_2() {
        // Pin: dropping csha.level to 1 alone makes the predicate `false`.
        // Proves the level >= 2 check is separable from the sinks check.
        let mut cfg = tier_b1_otherwise_eligible_cfg();
        cfg.csha.as_mut().unwrap().level = 1;
        assert!(!is_tier_b1_dispatch(&cfg));
    }

    #[test]
    fn is_tier_b1_dispatch_false_when_gpu_sm_below_80() {
        let mut cfg = tier_b1_otherwise_eligible_cfg();
        cfg.gpu_sm = 75;
        assert!(!is_tier_b1_dispatch(&cfg));
    }

    /// Sprint 3 cycle-7: when `synthesize_flash_attention_ptx_v2_with_tier_b`
    /// sees a sinks-enabled config that would otherwise route to Tier B.1,
    /// it must fall through to Tier A v2 (which is sink-aware per Sprint 1b)
    /// rather than emit the Tier B.1 single-iter scaffold sentinel.
    #[test]
    fn sinks_enabled_falls_through_to_tier_a_v2_not_tier_b1() {
        let mut cfg = tier_b1_otherwise_eligible_cfg();
        // Sprint 1b narrow-config so the Tier A v2 path accepts sinks.
        cfg.causal = false;
        cfg.block_q = 64;
        cfg.block_kv = 64;
        cfg.head_dim = 64;
        cfg.num_sink_tokens = 4;
        let ptx = synthesize_flash_attention_ptx_v2(&cfg);
        let s = String::from_utf8_lossy(&ptx);
        assert!(
            !s.contains("Tier B.1 single-iter scaffold complete"),
            "sinks-enabled config must NOT route to Tier B.1 scaffold (defense-in-depth); got: {}",
            &s[..s.len().min(200)]
        );
    }

    /// Sprint 3 cycle-7: direct call to `tier_b1::synthesize` with a
    /// sinks-enabled config MUST panic with a Sprint-3-citing message.
    /// Defense in depth in case a caller bypasses the v2 entry path.
    #[test]
    #[should_panic(expected = "Sprint 3 cycle-7 refusal")]
    fn tier_b1_synthesize_panics_when_sinks_enabled() {
        let mut cfg = tier_b1_otherwise_eligible_cfg();
        cfg.num_sink_tokens = 4;
        // `chunk` arg is irrelevant — the panic fires before any emission.
        let _ = tier_b1::synthesize(&cfg, 32);
    }
}
