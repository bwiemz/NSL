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

fn emit_cp_async_x_kv_for_slot(ptx: &mut String, slot: u32) {
    // B1.5 TODO: emit concrete cp.async.ca.shared.global instructions per slot
    // (real per-thread iteration + offset arithmetic). For B1.4 the cadence
    // (commit_group/wait_group placement) is what's load-bearing per spec §6.5;
    // the body is a placeholder that the main FSM orchestrator will fill in.
    ptx.push_str(&format!(
        "    // (placeholder) cp.async.ca.shared.global slot[{}].x_kv from HBM -- B1.5 will emit real bursts\n",
        slot
    ));
}

/// Phase A load: cp.async load x_kv slab into slot[curr].
/// Inner FSM at depth=1 per spec section 4.3.
pub fn emit_main_loop_phase_a_load(ptx: &mut String, _config: &FlashAttentionConfig, _kv_iter: u32) {
    ptx.push_str("    // Phase A: chunk-loop body lives in projection_mma; this helper just synchronizes.\n");
    ptx.push_str("    bar.sync 0;\n");
}

/// Phase C: kick off slot[next].x_kv load for kv_iter+2 + slot swap.
/// Per spec section 4.2 Phase C.
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
