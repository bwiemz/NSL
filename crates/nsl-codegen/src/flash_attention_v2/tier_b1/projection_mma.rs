//! Q/K/V projection chunk-streamed MMA emitter.
//!
//! B1.3 ships the Q projection (`emit_q_projection`). The matching K and V
//! projection emitters land in B1.5 alongside the cp.async ping-pong
//! pipeline plumbing from B1.4.
//!
//! Layout (matches `smem_layout::tier_b1_*_offset` accessors + the
//! `validate_tier_b1_config` chunk-staging budget):
//!
//! ```text
//!   q_offset                                                      [bq * hd * 2]
//!   k_offset_ping                                                 [bkv * hd * 2]
//!   k_offset_pong                                                 [bkv * hd * 2]
//!   v_offset_ping                                                 [bkv * hd * 2]
//!   v_offset_pong                                                 [bkv * hd * 2]
//!   w_chunk_offset            slot 0: W chunk #0 (Wq | Wk)         [chunk*hd*2]
//!                             slot 1: W chunk #1 (Wv)              [chunk*hd*2]
//!                             x_q chunk staging                    [bq*chunk*2]
//!                             x_kv chunk staging                   [bkv*chunk*2]
//! ```
//!
//! For Q projection at B1.3:
//!   * Wq chunk staged at  w_chunk_offset (slot 0).
//!   * x_q chunk staged at w_chunk_offset + 2 * chunk * hd * 2 (after both W slots).
//!
//! Warp distribution (per spec §5.3):
//!   * 8 warps per CTA.
//!   * Q output tile is (bq × hd), tiled by m16n8k16 into
//!     (bq/16) × (hd/8) tiles.
//!   * Warp `w` owns linear tile indices `t` where `t % 8 == w`.
//!   * `tiles_per_warp = max((bq/16) * (hd/8) / 8, 1)`. Each MMA produces
//!     4 f32 lanes per thread → `4 * tiles_per_warp` accumulator regs.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{tier_b1_q_offset, tier_b1_w_chunk_offset};
use crate::matmul_mma::emit_mma_instruction;

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the Q
/// projection output (bq × hd). Always at least 1 — small configs
/// (e.g. bq=32, hd=32 → 8 tiles total → 1 per warp) bottom out here.
fn tiles_per_warp(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let hd = config.head_dim as u32;
    let m_tiles = bq / 16;
    let n_tiles = hd / 8;
    let total = m_tiles.max(1) * n_tiles.max(1);
    (total / 8).max(1)
}

/// Emit the Q projection sub-kernel for Tier B.1.
///
/// Behaviour:
///   1. Declare + zero per-warp Q accumulators (`%q_acc_<t>_<lane>`).
///   2. Null-guard `csha_x_ptr` AND `csha_wq_ptr`; on null, branch to
///      `V2_TIER_B1_Q_PROJ_SKIP`.
///   3. Outer loop over `n_chunks = d_model / chunk`:
///       * cp.async stage `x_q_chunk[chunk_idx]` to SMEM at
///         `w_chunk_offset + 2*chunk*hd*2` (x_q slot).
///       * cp.async stage `Wq_chunk[chunk_idx]` to SMEM at
///         `w_chunk_offset` (W slot 0).
///       * `cp.async.commit_group; cp.async.wait_group 0; bar.sync 0`.
///       * Inner MMA loop: for `t in 0..tiles_per_warp` emit one
///         `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` consuming
///         the staged x_q + Wq SMEM and accumulating into `%q_acc_t_*`.
///   4. Pack f32 accumulators to f16 and scatter into Q SMEM at
///      `tier_b1_q_offset(config)`. End with `bar.sync 0`.
///   5. Land the `V2_TIER_B1_Q_PROJ_SKIP:` label.
///
/// `chunk` is the d_model_chunk size selected by `chunk_config::select`
/// (one of {128, 64, 32, FLOOR}).
///
/// # B1.3 deferrals (FOUR classes of placeholder for B1.4 to replace)
///
/// B1.3 ships structural FSM scaffolding. The kernel ptxas-validates on
/// sm_80 + sm_120, but **will NOT compute correct numerics if launched**
/// — there are four independent layers of placeholder work, each of
/// which B1.4 must replace before the kernel is numerically meaningful:
///
/// 1. **A/B fragment loads use a uniform SMEM address.** Every lane reads
///    from `[%tb1_q_smem_w]` and `[%tb1_q_smem_x]` with no per-(warp, lane,
///    K-step) offset. Real CUTLASS-style per-thread fragment loads should
///    replace this via `matmul_mma::emit_load_a_fragment_smem` and
///    `emit_load_b_fragment_smem` (those helpers already exist).
/// 2. **Warp-ownership predicate (`t % 8 == warp_id`) is NOT emitted.**
///    All 8 warps currently execute every MMA in the inner loop. B1.4 must
///    gate each MMA on a `setp` + `@%pred` against `%warp_id`, or the
///    accumulators end up 8× the intended sum.
/// 3. **cp.async stages 16 bytes per chunk, not the chunk's full
///    `chunk * hd * 2` bytes.** 16 bytes is the only legal cp.async width
///    in PTX ISA 7.0; staging a full chunk requires a per-thread loop with
///    per-thread destination offsets. B1.4 must add that loop.
/// 4. **`d_model % chunk == 0` is a precondition with no upstream guard.**
///    `chunk_config::select` doesn't enforce divisibility today. A
///    debug_assert is added at the top of this function as a tripwire;
///    long-term either the assert moves into `chunk_config::select` as a
///    rejection criterion, OR the chunk loop emits a tail-iteration
///    handler for the remainder.
///
/// Caller guarantees `d_model % chunk == 0` for admitted configs (V3
/// supported-matrix CSV property; debug_assert enforces it at this layer).
pub fn emit_q_projection(ptx: &mut String, config: &FlashAttentionConfig, chunk: u32) {
    let csha = match &config.csha {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier B.1 Q projection: csha=None, no emission\n");
            return;
        }
    };
    let d_model = csha.d_model;
    if d_model == 0 || chunk == 0 {
        ptx.push_str("    // Tier B.1 Q projection: d_model=0 or chunk=0, no emission\n");
        return;
    }
    // B1.3 deferral #4 tripwire: chunk_config::select does not currently
    // reject non-divisible (d_model, chunk) pairs. If a non-divisible pair
    // reaches this emitter (e.g., user-overridden chunk param, or future
    // model_config with non-power-of-2 d_model), the integer division
    // below silently truncates and the kernel under-computes Q. B1.4
    // should either (a) add divisibility check to chunk_config::select
    // as a rejection criterion, OR (b) emit a tail-iteration handler.
    debug_assert!(
        d_model % chunk == 0,
        "Tier B.1 Q projection: d_model ({}) must be divisible by chunk ({}); chunk_config::select missing divisibility check (see B1.4 deferral #4)",
        d_model,
        chunk
    );
    let n_chunks = d_model / chunk;
    let hd = config.head_dim as u32;
    let bq = config.block_q as u32;
    let tpw = tiles_per_warp(config);

    // SMEM offsets (computed inline so the function is self-contained).
    let w_chunk_off = tier_b1_w_chunk_offset(config);
    // Two W slots reserved by validate_tier_b1_config; x_q starts after both.
    let x_chunk_off = w_chunk_off + 2 * chunk * hd * 2;
    let q_out_off = tier_b1_q_offset(config);

    ptx.push_str(&format!(
        "    // === Tier B.1 Q projection: bq={} hd={} d_model={} chunk={} n_chunks={} tiles_per_warp={} ===\n",
        bq, hd, d_model, chunk, n_chunks, tpw
    ));

    // ----- 1. Per-warp Q accumulator registers + zero-init -----
    ptx.push_str("    // Declare + zero per-warp Q accumulator registers (4 f32 lanes/tile).\n");
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %q_acc_{}_{};\n", t, lane));
        }
    }
    // Plus scratch + addressing regs used by this emitter.
    ptx.push_str("    .reg .u64 %tb1_q_x_ptr, %tb1_q_wq_ptr;\n");
    ptx.push_str("    .reg .pred %tb1_q_xnull, %tb1_q_wqnull;\n");
    ptx.push_str("    .reg .u64 %tb1_q_smem_w, %tb1_q_smem_x, %tb1_q_smem_q;\n");
    ptx.push_str("    .reg .u64 %tb1_q_scatter_addr;\n");
    ptx.push_str("    .reg .b16 %tb1_q_h0, %tb1_q_h1;\n");
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!(
                "    mov.f32 %q_acc_{}_{}, 0f00000000;\n",
                t, lane
            ));
        }
    }

    // ----- 2. Null-guards on csha_x_ptr + csha_wq_ptr -----
    ptx.push_str("    // Null-guard csha_x_ptr; on null, skip Q projection (Tier A path).\n");
    ptx.push_str("    ld.param.u64 %tb1_q_x_ptr, [csha_x_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_q_xnull, %tb1_q_x_ptr, 0;\n");
    ptx.push_str("    @%tb1_q_xnull bra V2_TIER_B1_Q_PROJ_SKIP;\n");
    ptx.push_str("    ld.param.u64 %tb1_q_wq_ptr, [csha_wq_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_q_wqnull, %tb1_q_wq_ptr, 0;\n");
    ptx.push_str("    @%tb1_q_wqnull bra V2_TIER_B1_Q_PROJ_SKIP;\n");

    // Pre-compute SMEM bases for the W and x_q chunk slots + Q output tile.
    ptx.push_str(&format!(
        "    add.u64 %tb1_q_smem_w, %shmem_base, {}; // Wq chunk slot (w_chunk_off)\n",
        w_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_q_smem_x, %shmem_base, {}; // x_q chunk slot\n",
        x_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_q_smem_q, %shmem_base, {}; // Q output tile (tier_b1_q_offset)\n",
        q_out_off
    ));

    // ----- 3. Outer chunk loop -----
    // For B1.3 we unroll the chunk loop so the snapshot diff captures the
    // exact iteration count per config — matches the FSM-gate intent of
    // spec §6.5 (snapshot locks the structure). Real cp.async ping-pong
    // (B1.4) will collapse this into a single rotated loop body.
    for chunk_idx in 0..n_chunks {
        ptx.push_str(&format!(
            "    // -- Q-proj chunk {}/{} (d_model offset {} .. {}) --\n",
            chunk_idx + 1,
            n_chunks,
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));

        // cp.async stage Wq chunk: bytes = chunk * hd * 2 (f16).
        //
        // Source HBM offset: chunk_idx * chunk * hd * 2 from csha_wq_ptr base.
        // Destination SMEM: w_chunk_off (slot 0).
        ptx.push_str(&format!(
            "    // cp.async Wq[{}:{}] -> SMEM[w_chunk_off]\n",
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));
        // B1.4 TODO (deferral #3): this is a single 16-byte burst per
        // CTA-wide issuance, NOT a chunk-completing per-thread loop. A
        // full chunk = chunk*hd*2 bytes (e.g. 8192 bytes for canonical
        // 128*32*2). To stage that, B1.4 must wrap this cp.async in a
        // per-thread loop with per-thread destination offsets — likely
        // via the cp.async ping-pong helper to be added in pipeline.rs.
        // (Sized-16 is the only legal width for cp.async in PTX ISA 7.0,
        // so the issuance shape is correct; what's missing is the count.)
        ptx.push_str(&format!(
            "    cp.async.cg.shared.global [%tb1_q_smem_w], [%tb1_q_wq_ptr + {}], 16;\n",
            chunk_idx * chunk * hd * 2
        ));

        // cp.async stage x_q chunk: bytes = bq * chunk * 2 (f16 after f32->f16
        // narrowing during the load — the actual narrowing happens in B1.4).
        //
        // Source HBM offset: chunk_idx * chunk * 4 (x_q is f32 in HBM after
        // RMSNorm, so element-stride is 4 bytes).
        // Destination SMEM: x_chunk_off.
        ptx.push_str(&format!(
            "    // cp.async x_q[chunk={}] -> SMEM[x_chunk_off]\n",
            chunk_idx
        ));
        ptx.push_str(&format!(
            "    cp.async.cg.shared.global [%tb1_q_smem_x], [%tb1_q_x_ptr + {}], 16;\n",
            chunk_idx * chunk * 4
        ));

        // Commit + wait + barrier: synchronous chunk for B1.3 (B1.4
        // overlaps via ping-pong).
        ptx.push_str("    cp.async.commit_group;\n");
        ptx.push_str("    cp.async.wait_group 0;\n");
        ptx.push_str("    bar.sync 0;\n");

        // ----- 3b. Inner MMA loop -----
        // For each output tile `t` owned by this warp, emit one
        // mma.sync.aligned.m16n8k16. A-fragment is x_q rows, B-fragment
        // is Wq cols. Accumulators live in %q_acc_t_*.
        for t in 0..tpw {
            // Per spec §5.3 lane→fragment mapping: lanes within a warp
            // collectively own one m16n8k16 fragment per tile.
            // emit_mma_instruction emits the PTX instruction; the actual
            // ld.shared.b32 fragment-loads land in B1.4 once the cp.async
            // pipeline knows the per-thread tile-offset arithmetic. For
            // B1.3 we use sentinel A/B fragment register names — they're
            // declared inline so ptxas can verify the instruction shape.
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_q_a_{}_{}_0, %tb1_q_a_{}_{}_1, %tb1_q_a_{}_{}_2, %tb1_q_a_{}_{}_3;\n",
                chunk_idx, t, chunk_idx, t, chunk_idx, t, chunk_idx, t
            ));
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_q_b_{}_{}_0, %tb1_q_b_{}_{}_1;\n",
                chunk_idx, t, chunk_idx, t
            ));
            // B1.4 TODO (deferral #1): replace with matmul_mma::
            // emit_load_a_fragment_smem(...) — that helper already does the
            // CUTLASS-style per-(warp, lane, K-step) row*stride+base+offset
            // arithmetic. Current uniform `[%tb1_q_smem_x]` load is correct
            // PTX syntax (ptxas accepts) but every lane reads the same 4
            // bytes; numerics will only become correct after the swap.
            for i in 0..4 {
                ptx.push_str(&format!(
                    "    ld.shared.b32 %tb1_q_a_{}_{}_{}, [%tb1_q_smem_x];\n",
                    chunk_idx, t, i
                ));
            }
            // B1.4 TODO (deferral #1): replace with matmul_mma::
            // emit_load_b_fragment_smem(...) for per-lane B-fragment addressing.
            for i in 0..2 {
                ptx.push_str(&format!(
                    "    ld.shared.b32 %tb1_q_b_{}_{}_{}, [%tb1_q_smem_w];\n",
                    chunk_idx, t, i
                ));
            }
            let d_regs = [
                format!("%q_acc_{}_0", t),
                format!("%q_acc_{}_1", t),
                format!("%q_acc_{}_2", t),
                format!("%q_acc_{}_3", t),
            ];
            let a_regs = [
                format!("%tb1_q_a_{}_{}_0", chunk_idx, t),
                format!("%tb1_q_a_{}_{}_1", chunk_idx, t),
                format!("%tb1_q_a_{}_{}_2", chunk_idx, t),
                format!("%tb1_q_a_{}_{}_3", chunk_idx, t),
            ];
            let b_regs = [
                format!("%tb1_q_b_{}_{}_0", chunk_idx, t),
                format!("%tb1_q_b_{}_{}_1", chunk_idx, t),
            ];
            let c_regs = d_regs.clone();
            // B1.4 TODO (deferral #2): gate this MMA on the warp-ownership
            // predicate (`setp.eq.u32 %wo_pred, %warp_id, (t %% 8); @%wo_pred mma.sync ...`).
            // Currently every one of the 8 warps executes every MMA tile; the
            // accumulator end-state is therefore 8x the intended sum. ptxas
            // accepts the instruction; the gate is what makes the numerics right.
            ptx.push_str(&format!(
                "    // MMA tile t={} -- intended ownership: t %% 8 == warp_id (gate NOT yet emitted; B1.4)\n",
                t
            ));
            emit_mma_instruction(ptx, &d_regs, &a_regs, &b_regs, &c_regs);
        }
    }

    // ----- 4. Pack f32 -> f16 and scatter to Q SMEM tile -----
    //
    // Per spec §5.5 lane-coherent scatter: each thread holds 4 f32
    // accumulator lanes per tile, which pack into 2 b32 regs (each
    // holding 2 f16 lanes). For B1.3 the scatter uses one
    // `st.shared.b32` per packed register; real CUTLASS-style
    // lane→fragment offset arithmetic lands in B1.5.
    ptx.push_str("    // Pack f32 -> f16 + scatter Q accumulators to SMEM at tier_b1_q_offset\n");
    for t in 0..tpw {
        // Pack pairs (lane 0,1) and (lane 2,3) via cvt + bfi pattern.
        // Tier A's projection emitter uses a per-lane cvt.rn.f16.f32 +
        // st.shared.b16 — we mirror that here for instruction-level
        // continuity with Tier A's snapshots.
        for lane in 0..4 {
            ptx.push_str(&format!(
                "    cvt.rn.f16.f32 %tb1_q_h0, %q_acc_{}_{};\n",
                t, lane
            ));
            ptx.push_str(&format!(
                "    add.u64 %tb1_q_scatter_addr, %tb1_q_smem_q, {}; // tile={} lane={} byte off\n",
                t * 8 + lane * 2,
                t,
                lane
            ));
            ptx.push_str(
                "    st.shared.b16 [%tb1_q_scatter_addr], %tb1_q_h0;\n",
            );
        }
    }
    ptx.push_str("    bar.sync 0; // FENCE: Q tile visible before downstream phases\n");

    // ----- 5. Null-guard skip label -----
    ptx.push_str("V2_TIER_B1_Q_PROJ_SKIP:\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn make_config(bq: i64, bkv: i64, hd: i64, dm: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: bq,
            block_kv: bkv,
            head_dim: hd,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 120,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model: dm,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn null_guards_appear_exactly_twice() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        assert_eq!(ptx.matches("setp.eq.u64 %tb1_q_").count(), 2,
            "expected 2 null-guards (x_ptr + wq_ptr); got: {}", ptx);
        assert_eq!(ptx.matches("V2_TIER_B1_Q_PROJ_SKIP").count(), 3,
            "expected 3 occurrences of the skip label (2 bra + 1 landing): {}", ptx);
    }

    #[test]
    fn chunk_loop_unrolls_correctly() {
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        // n_chunks = 2048 / 128 = 16. We emit one cp.async pair per chunk.
        // Count cp.async.commit_group to match n_chunks.
        assert_eq!(ptx.matches("cp.async.commit_group").count(), 16,
            "expected 16 chunk iterations (2048 / 128)");
    }

    #[test]
    fn mma_count_matches_tiles_per_warp_times_chunks() {
        // (bq=32, hd=32): m_tiles=2, n_tiles=4, total=8 -> tpw = 1.
        // n_chunks=16. Expected MMA count = 16 * 1 = 16.
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        assert_eq!(mma_count, 16,
            "expected 16 MMA instructions (1 tile/warp * 16 chunks); got {}\nPTX:\n{}",
            mma_count, ptx);
    }

    #[test]
    fn csha_none_emits_noop_comment() {
        let mut cfg = make_config(32, 32, 32, 2048);
        cfg.csha = None;
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        assert!(ptx.contains("csha=None"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn d_model_zero_emits_noop() {
        let cfg = make_config(32, 32, 32, 0);
        let mut ptx = String::new();
        emit_q_projection(&mut ptx, &cfg, 128);
        assert!(ptx.contains("d_model=0"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn tiles_per_warp_canonical_64_64_64() {
        // bq=64, hd=64 -> m_tiles=4, n_tiles=8, total=32, tpw=4.
        let cfg = make_config(64, 64, 64, 2048);
        assert_eq!(tiles_per_warp(&cfg), 4);
    }

    #[test]
    fn tiles_per_warp_small_floor() {
        // bq=32, hd=32 -> total=8, tpw = 8/8 = 1.
        let cfg = make_config(32, 32, 32, 2048);
        assert_eq!(tiles_per_warp(&cfg), 1);
    }
}
