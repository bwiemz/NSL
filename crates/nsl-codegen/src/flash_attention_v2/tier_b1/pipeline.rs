//! cp.async + commit_group + wait_group helpers + ping-pong slot management.
//!
//! B1.4 Task 4.1: outer prologue + Phase A load + Phase C swap helpers.
//! The cp.async per-thread iteration arithmetic is a placeholder; real
//! per-thread burst emission lives in B1.5 (main FSM orchestrator).

use crate::flash_attention::FlashAttentionConfig;

/// Outer prologue: prime slot[0] + slot[1] cp.asyncs for the first two
/// kv tiles. Per spec section 4.2 prologue.
pub fn emit_prologue_kicks(ptx: &mut String, _config: &FlashAttentionConfig) {
    ptx.push_str("    // Tier B.1 outer prologue: prime slot[0] + slot[1] x_kv loads\n");

    ptx.push_str("    // === slot[0].x_kv <- HBM[kv_iter=0] ===\n");
    emit_cp_async_x_kv_for_slot(ptx, 0);
    ptx.push_str("    cp.async.commit_group; // group 0\n");

    ptx.push_str("    // === slot[1].x_kv <- HBM[kv_iter=1] ===\n");
    emit_cp_async_x_kv_for_slot(ptx, 1);
    ptx.push_str("    cp.async.commit_group; // group 1\n");

    ptx.push_str("    // wait for slot[0] (leaves slot[1] in flight)\n");
    ptx.push_str("    cp.async.wait_group 1;\n");
    ptx.push_str("    bar.sync 0;\n");
}

fn emit_cp_async_x_kv_for_slot(_ptx: &mut String, _slot: u32) {
    // B1.5 TODO: emit concrete cp.async.ca.shared.global instructions per slot
    // (per-thread iteration + offset arithmetic). Emits nothing until B1.5 --
    // keeping the placeholder body empty makes the commit_group cadence
    // (between the two slot-priming calls in emit_prologue_kicks) the sole
    // PTX output of this section, which is the load-bearing FSM gate per
    // spec section 6.5. Once a real PTX burst is emitted here, the snapshot
    // for prologue_kicks_snapshot will need to be regenerated.
}

/// Phase A load: cp.async load x_kv slab into slot[curr].
/// Inner FSM at depth=1 per spec section 4.3.
pub fn emit_main_loop_phase_a_load(ptx: &mut String, _config: &FlashAttentionConfig, _kv_iter: u32) {
    ptx.push_str("    // Phase A: chunk-loop body lives in projection_mma; this helper just synchronizes.\n");
    ptx.push_str("    bar.sync 0;\n");
}

/// Phase C: kick off slot[next].x_kv load for kv_iter+2 + slot swap.
/// Per spec section 4.2 Phase C.
///
/// **B1.5 prereq:** the main FSM prelude in `tier_b1::synthesize` must
/// declare these named registers before this helper is invoked, or ptxas
/// will reject with CUDA_ERROR_INVALID_PTX:
/// ```text
///   .reg .u32  %r_kv_iter_next, %r_n_kv, %r_slot_curr;
///   .reg .pred %p_prefetch;
/// ```
/// (B1.4 ships these references unanchored because the orchestrator that
/// owns the prelude is still a stub at this milestone.)
pub fn emit_main_loop_phase_c_swap(ptx: &mut String, _config: &FlashAttentionConfig, kv_iter: u32) {
    ptx.push_str(&format!(
        "    // Phase C: prefetch kv_iter={} into the slot we just consumed\n",
        kv_iter + 2
    ));
    ptx.push_str("    // (conditional on kv_iter + 2 < N_kv; emitted as runtime predicate)\n");
    ptx.push_str("    setp.lt.u32 %p_prefetch, %r_kv_iter_next, %r_n_kv;\n");
    ptx.push_str("    @%p_prefetch bra V2_TIER_B1_PHASE_C_DO_PREFETCH;\n");
    ptx.push_str("    bra V2_TIER_B1_PHASE_C_AFTER_PREFETCH;\n");
    ptx.push_str("V2_TIER_B1_PHASE_C_DO_PREFETCH:\n");
    ptx.push_str("    // cp.async issue for slot[curr] (which becomes slot[next] after swap)\n");
    ptx.push_str("    cp.async.commit_group;\n");
    ptx.push_str("V2_TIER_B1_PHASE_C_AFTER_PREFETCH:\n");
    ptx.push_str("    cp.async.wait_group 1;\n");
    ptx.push_str("    bar.sync 0;\n");
    ptx.push_str("    // slot swap: toggle slot[curr] in addressing arithmetic\n");
    ptx.push_str("    xor.b32 %r_slot_curr, %r_slot_curr, 1;\n");
}
