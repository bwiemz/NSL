//! Q/K/V projection chunk-streamed MMA emitter.
//!
//! B1.3 ships the Q projection (`emit_q_projection`). B1.5 adds the K/V
//! projection emitter (`emit_kv_projection_chunk_loop`), which mirrors the
//! Q-projection structure but accumulates into two independent sets of f32
//! registers (K and V) and scatters to the appropriate ping/pong SMEM slots.
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
//!   w_chunk_offset            slot 0: Wk chunk          [chunk*hd*2]
//!                             slot 1: Wv chunk          [chunk*hd*2]
//!                             x_q chunk staging         [bq*chunk*2]
//!                             x_kv chunk staging        [bkv*chunk*2]
//! ```
//!
//! For Q projection at B1.3:
//!   * Wq chunk staged at  w_chunk_offset (slot 0).
//!   * x_q chunk staged at w_chunk_offset + 2 * chunk * hd * 2 (after both W slots).
//!
//! For K/V projection at B1.5:
//!   * Wk chunk staged at w_chunk_offset (slot 0).
//!   * Wv chunk staged at w_chunk_offset + chunk * hd * 2 (slot 1).
//!   * x_kv chunk staged at w_chunk_offset + 2*chunk*hd*2 + bq*chunk*2.
//!   * x_kv HBM offset: kv_iter * bkv * d_model * 4 (f32 rows).
//!
//! Warp distribution (per spec §5.3):
//!   * 8 warps per CTA.
//!   * Q output tile is (bq × hd), tiled by m16n8k16 into
//!     (bq/16) × (hd/8) tiles.
//!   * K/V output tile is (bkv × hd), tiled by m16n8k16 into
//!     (bkv/16) × (hd/8) tiles.
//!   * Warp `w` owns linear tile indices `t` where `t % 8 == w`.
//!   * `tiles_per_warp = max((bq/16) * (hd/8) / 8, 1)`.
//!   * `tiles_per_warp_kv = max((bkv/16) * (hd/8) / 8, 1)`. Each MMA
//!     produces 4 f32 lanes per thread.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    tier_b1_k_offset_ping, tier_b1_k_offset_pong, tier_b1_q_offset, tier_b1_v_offset_ping,
    tier_b1_v_offset_pong, tier_b1_w_chunk_offset,
};
use crate::matmul_mma::{
    emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction_predicated,
};

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the Q
/// projection output (bq × hd). Always at least 1 — small configs
/// (e.g. bq=32, hd=32 → 8 tiles total → 1 per warp) bottom out here.
pub(crate) fn tiles_per_warp(config: &FlashAttentionConfig) -> u32 {
    let bq = config.block_q as u32;
    let hd = config.head_dim as u32;
    let m_tiles = bq / 16;
    let n_tiles = hd / 8;
    let total = m_tiles.max(1) * n_tiles.max(1);
    (total / 8).max(1)
}

/// Number of (m16, n8) MMA tiles each of the 8 warps owns for the K or V
/// projection output (bkv × hd). Always at least 1 — small configs
/// (e.g. bkv=32, hd=32 → 8 tiles total → 1 per warp) bottom out here.
/// Analogous to `tiles_per_warp` but uses `bkv` as the M dimension.
pub(crate) fn tiles_per_warp_kv(config: &FlashAttentionConfig) -> u32 {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    let m_tiles = bkv / 16;
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

    // ----- 1. Scratch + addressing regs used by this emitter only.
    // The per-warp Q accumulators (%q_acc_<t>_<lane>) are hoisted to the
    // orchestrator via register_budget::declare_registers (B1.6 deferral
    // #4); this helper just writes into them.
    ptx.push_str("    .reg .u64 %tb1_q_x_ptr, %tb1_q_wq_ptr;\n");
    ptx.push_str("    .reg .pred %tb1_q_xnull, %tb1_q_wqnull;\n");
    ptx.push_str("    .reg .u64 %tb1_q_smem_w, %tb1_q_smem_x, %tb1_q_smem_q;\n");
    // u32 sister registers for matmul_mma::emit_load_*_fragment_smem
    // (B1.6 deferral #1) — the helpers use 32-bit SMEM addressing.
    ptx.push_str("    .reg .u32 %tb1_q_smem_w_u32, %tb1_q_smem_x_u32;\n");
    ptx.push_str("    .reg .u64 %tb1_q_scatter_addr;\n");
    ptx.push_str("    .reg .b16 %tb1_q_h0, %tb1_q_h1;\n");

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
    // Narrow SMEM bases to u32 for the matmul_mma fragment-load helpers.
    ptx.push_str("    cvt.u32.u64 %tb1_q_smem_w_u32, %tb1_q_smem_w;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_q_smem_x_u32, %tb1_q_smem_x;\n");

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
            // B1.6 deferral #1 resolution: A/B-fragment loads now use the
            // matmul_mma helpers which compute per-(warp, lane) addresses
            // from %mma_a_row / %mma_b_row (set up by declare_registers).
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_q_a_{}_{}_0, %tb1_q_a_{}_{}_1, %tb1_q_a_{}_{}_2, %tb1_q_a_{}_{}_3;\n",
                chunk_idx, t, chunk_idx, t, chunk_idx, t, chunk_idx, t
            ));
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_q_b_{}_{}_0, %tb1_q_b_{}_{}_1;\n",
                chunk_idx, t, chunk_idx, t
            ));
            // A-fragment: x_q rows from SMEM at %tb1_q_smem_x. Row stride is
            // chunk * 2 bytes (the x_q chunk slab is bq rows x chunk cols, f16).
            let a_fragment_regs = [
                format!("tb1_q_a_{}_{}_0", chunk_idx, t),
                format!("tb1_q_a_{}_{}_1", chunk_idx, t),
                format!("tb1_q_a_{}_{}_2", chunk_idx, t),
                format!("tb1_q_a_{}_{}_3", chunk_idx, t),
            ];
            emit_load_a_fragment_smem(
                ptx,
                &a_fragment_regs,
                "%tb1_q_smem_x_u32",
                (chunk * 2) as usize,
            );
            // B-fragment: Wq cols from SMEM at %tb1_q_smem_w. Row stride is
            // hd * 2 bytes (Wq chunk slab is chunk rows x hd cols, f16).
            let b_fragment_regs = [
                format!("tb1_q_b_{}_{}_0", chunk_idx, t),
                format!("tb1_q_b_{}_{}_1", chunk_idx, t),
            ];
            emit_load_b_fragment_smem(
                ptx,
                &b_fragment_regs,
                "%tb1_q_smem_w_u32",
                (hd * 2) as usize,
            );
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
            // B1.6 deferral #2 resolution: warp-ownership gate. Only the
            // warp owning output tile `t` writes the MMA.
            ptx.push_str(&format!(
                "    setp.eq.u32 %wo_pred, %warp_id, {}; // Q-proj tile t={} ownership\n",
                t % 8, t
            ));
            emit_mma_instruction_predicated(ptx, &d_regs, &a_regs, &b_regs, &c_regs, "wo_pred");
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

/// Emit the K and V projection sub-kernel for Tier B.1 (B1.5 Phase A body).
///
/// Behaviour:
///   1. Declare + zero per-warp K accumulators (`%k_acc_<t>_<lane>`) AND
///      V accumulators (`%v_acc_<t>_<lane>`), `tiles_per_warp_kv(config)`
///      tiles each.
///   2. Null-guard `csha_x_ptr`, `csha_wk_ptr`, AND `csha_wv_ptr`; on null,
///      branch to `V2_TIER_B1_KV_PROJ_SKIP_<slot>`.
///   3. Outer loop over `n_chunks = d_model / chunk`:
///       * cp.async stage `Wk_chunk[chunk_idx]` to SMEM at
///         `w_chunk_off` (slot 0).
///       * cp.async stage `Wv_chunk[chunk_idx]` to SMEM at
///         `w_chunk_off + chunk*hd*2` (slot 1).
///       * cp.async stage `x_kv_chunk[chunk_idx]` to SMEM at
///         `w_chunk_off + 2*chunk*hd*2 + bq*chunk*2` (x_kv slot).
///       * `cp.async.commit_group; cp.async.wait_group 0; bar.sync 0`.
///       * Inner MMA loop: for `t in 0..tiles_per_warp_kv` emit TWO MMAs
///         per tile — one accumulating into `%k_acc_t_*` (Wk B-fragment),
///         one into `%v_acc_t_*` (Wv B-fragment). Both share the same x_kv
///         A-fragment registers.
///   4. Pack K f32 -> f16 and scatter into K SMEM at `tier_b1_k_offset_ping`
///      (when `slot == 0`) or `tier_b1_k_offset_pong` (when `slot == 1`).
///      Pack V f32 -> f16 and scatter into V SMEM at the corresponding
///      `tier_b1_v_offset_ping` / `tier_b1_v_offset_pong`. End with `bar.sync 0`.
///   5. Land the `V2_TIER_B1_KV_PROJ_SKIP_<slot>:` label.
///
/// `chunk` is the d_model_chunk size selected by `chunk_config::select`
/// (one of {128, 64, 32, FLOOR}).
///
/// `kv_iter` is the outer KV-tile iteration index (0-based). Used to
/// compute the x_kv HBM row-block offset: `kv_iter * bkv * d_model * 4`.
///
/// `slot` is the ping/pong slot index (0 or 1), compile-time-known at
/// emission. Controls which K/V SMEM slot the accumulators scatter into.
/// The skip label is parameterised by `slot` to avoid label collision when
/// the orchestrator calls this function twice with different slots.
///
/// # B1.6 deferrals (carry-forward from Q-projection's B1.4 deferrals)
///
/// The same four independent layers of placeholder work apply here as in
/// `emit_q_projection`. This function ships structural FSM scaffolding that
/// ptxas-validates on sm_80 + sm_120 but will NOT compute correct numerics
/// if launched. B1.6 must replace all four:
///
/// 1. **A/B fragment loads use a uniform SMEM address.** (See B1.4 TODO in
///    `emit_q_projection` deferral #1.) K/V markers: `tb1_kv_smem_x`,
///    `tb1_kv_smem_wk`, `tb1_kv_smem_wv`.
/// 2. **Warp-ownership predicate (`t % 8 == warp_id`) is NOT emitted.**
///    (See B1.4 TODO in `emit_q_projection` deferral #2.) All 8 warps
///    execute every MMA tile, giving 8x the intended sum.
/// 3. **cp.async stages 16 bytes per chunk, not the chunk's full bytes.**
///    (See B1.4 TODO in `emit_q_projection` deferral #3.) Three cp.async
///    per chunk instead of Q's two; each still only stages 16 bytes.
/// 4. **`d_model % chunk == 0` has no upstream guard.**
///    (See B1.4 TODO in `emit_q_projection` deferral #4.) debug_assert
///    tripwire added below; long-term fix belongs in `chunk_config::select`.
///
/// Caller guarantees `d_model % chunk == 0` for admitted configs (V3
/// supported-matrix CSV property; debug_assert enforces it at this layer).
pub fn emit_kv_projection_chunk_loop(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    chunk: u32,
    kv_iter: u32,
    slot: u32,
) {
    let csha = match &config.csha {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier B.1 KV projection: csha=None, no emission\n");
            return;
        }
    };
    let d_model = csha.d_model;
    if d_model == 0 || chunk == 0 {
        ptx.push_str("    // Tier B.1 KV projection: d_model=0 or chunk=0, no emission\n");
        return;
    }
    // B1.6 deferral #4 tripwire: chunk_config::select does not currently
    // reject non-divisible (d_model, chunk) pairs. If a non-divisible pair
    // reaches this emitter, the integer division below silently truncates and
    // the kernel under-computes K and V. B1.6 should either (a) add
    // divisibility check to chunk_config::select as a rejection criterion,
    // OR (b) emit a tail-iteration handler. See matching marker in
    // emit_q_projection (B1.4 deferral #4).
    debug_assert!(
        d_model % chunk == 0,
        "Tier B.1 KV projection: d_model ({}) must be divisible by chunk ({}); \
         chunk_config::select missing divisibility check (see B1.6 deferral #4)",
        d_model,
        chunk
    );
    let n_chunks = d_model / chunk;
    let hd = config.head_dim as u32;
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let tpw = tiles_per_warp_kv(config);

    // SMEM offsets (computed inline so the function is self-contained).
    let w_chunk_off = tier_b1_w_chunk_offset(config);
    // slot 0: Wk chunk; slot 1: Wv chunk (each chunk*hd*2 bytes).
    let wk_chunk_off = w_chunk_off;
    let wv_chunk_off = w_chunk_off + chunk * hd * 2;
    // x_kv staging lives after both W slots + x_q slot.
    let x_kv_off = w_chunk_off + 2 * chunk * hd * 2 + bq * chunk * 2;
    // K/V output tile: compile-time slot selection (slot is a u32 parameter).
    let k_out_off = if slot == 0 {
        tier_b1_k_offset_ping(config)
    } else {
        tier_b1_k_offset_pong(config)
    };
    let v_out_off = if slot == 0 {
        tier_b1_v_offset_ping(config)
    } else {
        tier_b1_v_offset_pong(config)
    };
    // x_kv HBM row-block offset: kv_iter * bkv rows, each of d_model f32 elements.
    let x_kv_hbm_base_off = kv_iter * bkv * d_model * 4; // f32 = 4 bytes

    ptx.push_str(&format!(
        "    // === Tier B.1 KV projection: bkv={} hd={} d_model={} chunk={} n_chunks={} \
         kv_iter={} slot={} tiles_per_warp_kv={} ===\n",
        bkv, hd, d_model, chunk, n_chunks, kv_iter, slot, tpw
    ));

    // ----- 1. Scratch + addressing regs used by this emitter only.
    // %k_acc_<t>_<lane> and %v_acc_<t>_<lane> are hoisted to the
    // orchestrator via register_budget::declare_registers (B1.6 deferral
    // #4); this helper just writes into them.
    ptx.push_str("    .reg .u64 %tb1_kv_x_ptr, %tb1_kv_wk_ptr, %tb1_kv_wv_ptr;\n");
    ptx.push_str("    .reg .pred %tb1_kv_xnull, %tb1_kv_wknull, %tb1_kv_wvnull;\n");
    ptx.push_str(
        "    .reg .u64 %tb1_kv_smem_wk, %tb1_kv_smem_wv, %tb1_kv_smem_x;\n",
    );
    // u32 sister registers for matmul_mma::emit_load_*_fragment_smem
    // (B1.6 deferral #1) — the helpers use 32-bit SMEM addressing.
    ptx.push_str("    .reg .u32 %tb1_kv_smem_wk_u32, %tb1_kv_smem_wv_u32, %tb1_kv_smem_x_u32;\n");
    ptx.push_str("    .reg .u64 %tb1_kv_smem_k, %tb1_kv_smem_v;\n");
    ptx.push_str("    .reg .u64 %tb1_kv_scatter_addr;\n");
    ptx.push_str("    .reg .b16 %tb1_kv_h0;\n");

    // ----- 2. Null-guards on csha_x_ptr, csha_wk_ptr, csha_wv_ptr -----
    let skip_label = format!("V2_TIER_B1_KV_PROJ_SKIP_{}", slot);
    ptx.push_str("    // Null-guard csha_x_ptr; on null, skip KV projection.\n");
    ptx.push_str("    ld.param.u64 %tb1_kv_x_ptr, [csha_x_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_kv_xnull, %tb1_kv_x_ptr, 0;\n");
    ptx.push_str(&format!("    @%tb1_kv_xnull bra {};\n", skip_label));
    ptx.push_str("    // Null-guard csha_wk_ptr; on null, skip KV projection.\n");
    ptx.push_str("    ld.param.u64 %tb1_kv_wk_ptr, [csha_wk_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_kv_wknull, %tb1_kv_wk_ptr, 0;\n");
    ptx.push_str(&format!("    @%tb1_kv_wknull bra {};\n", skip_label));
    ptx.push_str("    // Null-guard csha_wv_ptr; on null, skip KV projection.\n");
    ptx.push_str("    ld.param.u64 %tb1_kv_wv_ptr, [csha_wv_ptr];\n");
    ptx.push_str("    setp.eq.u64 %tb1_kv_wvnull, %tb1_kv_wv_ptr, 0;\n");
    ptx.push_str(&format!("    @%tb1_kv_wvnull bra {};\n", skip_label));

    // Pre-compute SMEM bases for the Wk, Wv, x_kv chunk slots + K/V output tiles.
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_wk, %shmem_base, {}; // Wk chunk slot (w_chunk_off slot 0)\n",
        wk_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_wv, %shmem_base, {}; // Wv chunk slot (w_chunk_off slot 1)\n",
        wv_chunk_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_x, %shmem_base, {}; // x_kv chunk slot (will be narrowed to u32 below)\n",
        x_kv_off
    ));
    // Narrow SMEM bases to u32 for the matmul_mma fragment-load helpers.
    ptx.push_str("    cvt.u32.u64 %tb1_kv_smem_wk_u32, %tb1_kv_smem_wk;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_kv_smem_wv_u32, %tb1_kv_smem_wv;\n");
    ptx.push_str("    cvt.u32.u64 %tb1_kv_smem_x_u32, %tb1_kv_smem_x;\n");
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_k, %shmem_base, {}; // K output tile (slot={})\n",
        k_out_off, slot
    ));
    ptx.push_str(&format!(
        "    add.u64 %tb1_kv_smem_v, %shmem_base, {}; // V output tile (slot={})\n",
        v_out_off, slot
    ));

    // ----- 3. Outer chunk loop -----
    // For B1.5 Phase A we unroll the chunk loop so the snapshot diff captures
    // the exact iteration count per config, matching the FSM-gate intent of
    // spec §6.5. The B1.6 cp.async ping-pong pipeline will collapse this into
    // a single rotated loop body.
    for chunk_idx in 0..n_chunks {
        ptx.push_str(&format!(
            "    // -- KV-proj chunk {}/{} (d_model offset {} .. {}) --\n",
            chunk_idx + 1,
            n_chunks,
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));

        // cp.async stage Wk chunk: bytes = chunk * hd * 2 (f16).
        //
        // Source HBM offset: chunk_idx * chunk * hd * 2 from csha_wk_ptr base.
        // Destination SMEM: wk_chunk_off (slot 0).
        ptx.push_str(&format!(
            "    // cp.async Wk[{}:{}] -> SMEM[wk_chunk_off slot 0]\n",
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));
        // B1.6 TODO (deferral #3): this is a single 16-byte burst, NOT a
        // chunk-completing per-thread loop. A full Wk chunk = chunk*hd*2
        // bytes (e.g. 8192 bytes for canonical 128*32*2). B1.6 must wrap
        // this cp.async in a per-thread loop with per-thread destination
        // offsets. See matching marker in emit_q_projection (B1.4 deferral #3).
        ptx.push_str(&format!(
            "    cp.async.cg.shared.global [%tb1_kv_smem_wk], [%tb1_kv_wk_ptr + {}], 16;\n",
            chunk_idx * chunk * hd * 2
        ));

        // cp.async stage Wv chunk: bytes = chunk * hd * 2 (f16).
        //
        // Source HBM offset: chunk_idx * chunk * hd * 2 from csha_wv_ptr base.
        // Destination SMEM: wv_chunk_off (slot 1).
        ptx.push_str(&format!(
            "    // cp.async Wv[{}:{}] -> SMEM[wv_chunk_off slot 1]\n",
            chunk_idx * chunk,
            (chunk_idx + 1) * chunk
        ));
        // B1.6 TODO (deferral #3): same 16-byte placeholder as Wk above.
        // Full Wv chunk requires the same per-thread loop fix. See B1.4 deferral #3.
        ptx.push_str(&format!(
            "    cp.async.cg.shared.global [%tb1_kv_smem_wv], [%tb1_kv_wv_ptr + {}], 16;\n",
            chunk_idx * chunk * hd * 2
        ));

        // cp.async stage x_kv chunk: bytes = bkv * chunk * 2 (f16 after f32->f16
        // narrowing during the load — the actual narrowing happens in B1.6).
        //
        // Source HBM offset: x_kv_hbm_base_off + chunk_idx * chunk * 4
        // (x_kv is f32 in HBM after RMSNorm; element-stride is 4 bytes).
        // Destination SMEM: x_kv_off.
        ptx.push_str(&format!(
            "    // cp.async x_kv[kv_iter={} chunk={}] -> SMEM[x_kv_off]\n",
            kv_iter, chunk_idx
        ));
        // B1.6 TODO (deferral #3): same 16-byte placeholder. Full x_kv
        // chunk = bkv*chunk*2 bytes; per-thread loop fix required. See B1.4
        // deferral #3 in emit_q_projection.
        ptx.push_str(&format!(
            "    cp.async.cg.shared.global [%tb1_kv_smem_x], \
             [%tb1_kv_x_ptr + {}], 16;\n",
            x_kv_hbm_base_off + chunk_idx * chunk * 4
        ));

        // Commit + wait + barrier: synchronous chunk for B1.5 Phase A
        // (B1.6 overlaps via ping-pong).
        ptx.push_str("    cp.async.commit_group;\n");
        ptx.push_str("    cp.async.wait_group 0;\n");
        ptx.push_str("    bar.sync 0;\n");

        // ----- 3b. Inner MMA loop (TWO MMAs per tile: K and V) -----
        // For each output tile `t` owned by this warp, emit:
        //   (a) one mma.sync.aligned.m16n8k16 into K accumulators using Wk
        //       B-fragment.
        //   (b) one mma.sync.aligned.m16n8k16 into V accumulators using Wv
        //       B-fragment.
        // Both MMAs share the same x_kv A-fragment registers (x_kv is the
        // shared input to both K and V projections).
        for t in 0..tpw {
            // --- Declare A/B fragment registers ---
            // A-fragment: x_kv rows (shared between K and V MMAs).
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_kv_a_{}_{}_0, %tb1_kv_a_{}_{}_1, \
                 %tb1_kv_a_{}_{}_2, %tb1_kv_a_{}_{}_3;\n",
                chunk_idx, t, chunk_idx, t, chunk_idx, t, chunk_idx, t
            ));
            // B-fragment for K (Wk rows).
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_kv_bk_{}_{}_0, %tb1_kv_bk_{}_{}_1;\n",
                chunk_idx, t, chunk_idx, t
            ));
            // B-fragment for V (Wv rows).
            ptx.push_str(&format!(
                "    .reg .b32 %tb1_kv_bv_{}_{}_0, %tb1_kv_bv_{}_{}_1;\n",
                chunk_idx, t, chunk_idx, t
            ));

            // B1.6 deferral #1 resolution: A/B-fragment loads now use
            // matmul_mma helpers with per-(warp, lane) addressing.
            //
            // A-fragment: x_kv rows from SMEM at %tb1_kv_smem_x_u32. Row
            // stride is chunk * 2 bytes (x_kv chunk slab is bkv rows x
            // chunk cols, f16). Shared between K and V MMAs.
            let a_fragment_regs = [
                format!("tb1_kv_a_{}_{}_0", chunk_idx, t),
                format!("tb1_kv_a_{}_{}_1", chunk_idx, t),
                format!("tb1_kv_a_{}_{}_2", chunk_idx, t),
                format!("tb1_kv_a_{}_{}_3", chunk_idx, t),
            ];
            emit_load_a_fragment_smem(
                ptx,
                &a_fragment_regs,
                "%tb1_kv_smem_x_u32",
                (chunk * 2) as usize,
            );

            // K B-fragment: Wk rows from SMEM at %tb1_kv_smem_wk_u32. Row
            // stride is hd * 2 bytes (Wk chunk slab is chunk rows x hd cols).
            let bk_fragment_regs = [
                format!("tb1_kv_bk_{}_{}_0", chunk_idx, t),
                format!("tb1_kv_bk_{}_{}_1", chunk_idx, t),
            ];
            emit_load_b_fragment_smem(
                ptx,
                &bk_fragment_regs,
                "%tb1_kv_smem_wk_u32",
                (hd * 2) as usize,
            );

            // V B-fragment: Wv rows from SMEM at %tb1_kv_smem_wv_u32.
            let bv_fragment_regs = [
                format!("tb1_kv_bv_{}_{}_0", chunk_idx, t),
                format!("tb1_kv_bv_{}_{}_1", chunk_idx, t),
            ];
            emit_load_b_fragment_smem(
                ptx,
                &bv_fragment_regs,
                "%tb1_kv_smem_wv_u32",
                (hd * 2) as usize,
            );

            // --- MMA (a): K accumulation ---
            let k_d_regs = [
                format!("%k_acc_{}_0", t),
                format!("%k_acc_{}_1", t),
                format!("%k_acc_{}_2", t),
                format!("%k_acc_{}_3", t),
            ];
            let a_regs = [
                format!("%tb1_kv_a_{}_{}_0", chunk_idx, t),
                format!("%tb1_kv_a_{}_{}_1", chunk_idx, t),
                format!("%tb1_kv_a_{}_{}_2", chunk_idx, t),
                format!("%tb1_kv_a_{}_{}_3", chunk_idx, t),
            ];
            let bk_regs = [
                format!("%tb1_kv_bk_{}_{}_0", chunk_idx, t),
                format!("%tb1_kv_bk_{}_{}_1", chunk_idx, t),
            ];
            let k_c_regs = k_d_regs.clone();
            // B1.6 deferral #2 resolution: warp-ownership gate (K MMA).
            ptx.push_str(&format!(
                "    setp.eq.u32 %wo_pred, %warp_id, {}; // K-proj tile t={} ownership\n",
                t % 8, t
            ));
            emit_mma_instruction_predicated(ptx, &k_d_regs, &a_regs, &bk_regs, &k_c_regs, "wo_pred");

            // --- MMA (b): V accumulation ---
            let v_d_regs = [
                format!("%v_acc_{}_0", t),
                format!("%v_acc_{}_1", t),
                format!("%v_acc_{}_2", t),
                format!("%v_acc_{}_3", t),
            ];
            let bv_regs = [
                format!("%tb1_kv_bv_{}_{}_0", chunk_idx, t),
                format!("%tb1_kv_bv_{}_{}_1", chunk_idx, t),
            ];
            let v_c_regs = v_d_regs.clone();
            // B1.6 deferral #2 resolution: warp-ownership gate (V MMA).
            // Same predicate value as K — both MMAs share the t % 8 ownership.
            emit_mma_instruction_predicated(ptx, &v_d_regs, &a_regs, &bv_regs, &v_c_regs, "wo_pred");
        }
    }

    // ----- 4. Pack f32 -> f16 and scatter K accumulators to K SMEM tile -----
    //
    // Per spec §5.5 lane-coherent scatter: each thread holds 4 f32
    // accumulator lanes per tile, packed and stored as f16 in SMEM.
    // B1.5 uses one `st.shared.b16` per lane, mirroring Q-projection's
    // scatter pattern. Real CUTLASS-style lane->fragment offset arithmetic
    // lands in B1.6.
    ptx.push_str("    // Pack f32 -> f16 + scatter K accumulators to SMEM at K tile offset\n");
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!(
                "    cvt.rn.f16.f32 %tb1_kv_h0, %k_acc_{}_{};\n",
                t, lane
            ));
            ptx.push_str(&format!(
                "    add.u64 %tb1_kv_scatter_addr, %tb1_kv_smem_k, {}; // tile={} lane={} byte off\n",
                t * 8 + lane * 2,
                t,
                lane
            ));
            ptx.push_str("    st.shared.b16 [%tb1_kv_scatter_addr], %tb1_kv_h0;\n");
        }
    }

    // ----- 4b. Pack f32 -> f16 and scatter V accumulators to V SMEM tile -----
    ptx.push_str("    // Pack f32 -> f16 + scatter V accumulators to SMEM at V tile offset\n");
    for t in 0..tpw {
        for lane in 0..4 {
            ptx.push_str(&format!(
                "    cvt.rn.f16.f32 %tb1_kv_h0, %v_acc_{}_{};\n",
                t, lane
            ));
            ptx.push_str(&format!(
                "    add.u64 %tb1_kv_scatter_addr, %tb1_kv_smem_v, {}; // tile={} lane={} byte off\n",
                t * 8 + lane * 2,
                t,
                lane
            ));
            ptx.push_str("    st.shared.b16 [%tb1_kv_scatter_addr], %tb1_kv_h0;\n");
        }
    }
    ptx.push_str("    bar.sync 0; // FENCE: K+V tiles visible before downstream phases\n");

    // ----- 5. Null-guard skip label -----
    ptx.push_str(&format!("{}:\n", skip_label));
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

    // ---- K/V projection tests (B1.5) ----------------------------------------

    #[test]
    fn kv_null_guards_appear_for_wk_and_wv() {
        // Canonical config: bq=bkv=hd=32, d_model=2048, chunk=128.
        // Expect exactly 2 null-guards for Wk+Wv (plus 1 for x_ptr = 3 total
        // setp.eq.u64 instructions; we count the wk/wv-specific ones).
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        // csha_wk_ptr null-guard:
        assert_eq!(
            ptx.matches("setp.eq.u64 %tb1_kv_wknull").count(),
            1,
            "expected exactly 1 Wk null-guard; got:\n{}",
            ptx
        );
        // csha_wv_ptr null-guard:
        assert_eq!(
            ptx.matches("setp.eq.u64 %tb1_kv_wvnull").count(),
            1,
            "expected exactly 1 Wv null-guard; got:\n{}",
            ptx
        );
        // Combined: 3 setp.eq.u64 total (x_ptr + wk_ptr + wv_ptr).
        assert_eq!(
            ptx.matches("setp.eq.u64 %tb1_kv_").count(),
            3,
            "expected 3 total null-guards (x + wk + wv); got:\n{}",
            ptx
        );
    }

    #[test]
    fn kv_mma_count_is_two_times_tpw_times_chunks() {
        // (bkv=32, hd=32): m_tiles=2, n_tiles=4, total=8, tpw_kv=1.
        // n_chunks = 2048/128 = 16.
        // Expected MMA count = 2 (K+V) * 1 (tpw_kv) * 16 (chunks) = 32.
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        let mma_count = ptx.matches("mma.sync.aligned.m16n8k16").count();
        let tpw_kv = tiles_per_warp_kv(&cfg);
        let n_chunks = 2048 / 128;
        let expected = 2 * tpw_kv * n_chunks;
        assert_eq!(
            mma_count, expected as usize,
            "expected {} MMA instructions (2 * {} tpw_kv * {} chunks); got {}\nPTX:\n{}",
            expected, tpw_kv, n_chunks, mma_count, ptx
        );
    }

    #[test]
    fn kv_commit_group_count_matches_n_chunks() {
        // n_chunks = 2048/128 = 16. One cp.async.commit_group per chunk
        // (after the three cp.async issuances for Wk, Wv, x_kv).
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        let commit_count = ptx.matches("cp.async.commit_group").count();
        assert_eq!(
            commit_count, 16,
            "expected 16 cp.async.commit_group (one per chunk); got {}\nPTX:\n{}",
            commit_count, ptx
        );
    }

    #[test]
    fn kv_skip_label_is_slot_parameterised() {
        // Calling with slot=0 and slot=1 must produce different skip labels to
        // avoid collision when the orchestrator emits both slots.
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx0 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx0, &cfg, 128, 0, 0);
        let mut ptx1 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx1, &cfg, 128, 0, 1);
        assert!(
            ptx0.contains("V2_TIER_B1_KV_PROJ_SKIP_0"),
            "slot=0 must use label V2_TIER_B1_KV_PROJ_SKIP_0"
        );
        assert!(
            ptx1.contains("V2_TIER_B1_KV_PROJ_SKIP_1"),
            "slot=1 must use label V2_TIER_B1_KV_PROJ_SKIP_1"
        );
        assert!(
            !ptx0.contains("V2_TIER_B1_KV_PROJ_SKIP_1"),
            "slot=0 output must not contain slot=1 label"
        );
    }

    #[test]
    fn kv_ping_pong_smem_offsets_differ_by_slot() {
        // slot=0 must reference k_offset_ping / v_offset_ping.
        // slot=1 must reference k_offset_pong / v_offset_pong.
        use crate::flash_attention_v2::smem_layout::{
            tier_b1_k_offset_ping, tier_b1_k_offset_pong,
            tier_b1_v_offset_ping, tier_b1_v_offset_pong,
        };
        let cfg = make_config(32, 32, 32, 2048);
        let mut ptx0 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx0, &cfg, 128, 0, 0);
        let mut ptx1 = String::new();
        emit_kv_projection_chunk_loop(&mut ptx1, &cfg, 128, 0, 1);

        let k_ping = tier_b1_k_offset_ping(&cfg);
        let k_pong = tier_b1_k_offset_pong(&cfg);
        let v_ping = tier_b1_v_offset_ping(&cfg);
        let v_pong = tier_b1_v_offset_pong(&cfg);

        assert!(
            ptx0.contains(&format!("// K output tile (slot=0)")),
            "slot=0 K tile comment must be present; k_ping={}", k_ping
        );
        assert!(
            ptx0.contains(&format!("// V output tile (slot=0)")),
            "slot=0 V tile comment must be present; v_ping={}", v_ping
        );
        assert!(
            ptx1.contains(&format!("// K output tile (slot=1)")),
            "slot=1 K tile comment must be present; k_pong={}", k_pong
        );
        assert!(
            ptx1.contains(&format!("// V output tile (slot=1)")),
            "slot=1 V tile comment must be present; v_pong={}", v_pong
        );
    }

    #[test]
    fn kv_csha_none_emits_noop_comment() {
        let mut cfg = make_config(32, 32, 32, 2048);
        cfg.csha = None;
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        assert!(ptx.contains("csha=None"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn kv_d_model_zero_emits_noop() {
        let cfg = make_config(32, 32, 32, 0);
        let mut ptx = String::new();
        emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
        assert!(ptx.contains("d_model=0"));
        assert!(!ptx.contains("mma.sync"));
    }

    #[test]
    fn tiles_per_warp_kv_small_floor() {
        // bkv=32, hd=32 -> m_tiles=2, n_tiles=4, total=8, tpw_kv=1.
        let cfg = make_config(32, 32, 32, 2048);
        assert_eq!(tiles_per_warp_kv(&cfg), 1);
    }

    #[test]
    fn tiles_per_warp_kv_canonical_64_64_64() {
        // bkv=64, hd=64 -> m_tiles=4, n_tiles=8, total=32, tpw_kv=4.
        let cfg = make_config(64, 64, 64, 2048);
        assert_eq!(tiles_per_warp_kv(&cfg), 4);
    }
}
