//! CSHA Tier B.1 -- Level 2 pipelined attention forward kernel.
//!
//! Design spec: docs/superpowers/specs/2026-05-11-csha-tier-b1-pipelined-attention-design.md
//!
//! Module map:
//!   * `pipeline`        - cp.async ping-pong helpers (commit_group, wait_group, slot mgmt)
//!   * `projection_mma`  - Q/K/V projection chunk-streamed MMA emitter
//!   * `attention_mma`   - QK^T + softmax + PV emitter
//!   * `chunk_config`    - d_model_chunk selection per variant
//!   * `register_budget` - static spill analysis pre-emission

pub mod pipeline;
pub mod projection_mma;
pub mod attention_mma;
pub mod chunk_config;
pub mod register_budget;
pub mod finalize;

use crate::flash_attention::FlashAttentionConfig;

/// Top-level Tier B.1 PTX emitter. Called from
/// `flash_attention_v2::synthesize_flash_attention_ptx_v2` when the
/// dispatch criteria are met. Returns PTX bytes terminated with NL+NUL.
///
/// # B1.5 Task 5.3 status: structural scaffold end-to-end
///
/// Emits the full FSM cadence in a single-iteration form:
///
///   1. PTX prelude (37-slot param block, reused 1:1 from Tier A per V1).
///   2. Tier B.1 outer-prologue register declarations: `%r_kv_iter_next`,
///      `%r_n_kv`, `%r_slot_curr`, `%p_prefetch` (consumed by
///      `pipeline::emit_main_loop_phase_c_swap`).
///   3. CSHA A.4 active_heads guard (reused unchanged).
///   4. RMSNorm pre-pass per q_tile_iter (reused; 8-warp partition
///      extension deferred to B1.6).
///   5. Q projection (B1.3): chunk-streamed MMA into Q SMEM tile.
///   6. cp.async prologue kicks (B1.4): prime slot[0] + slot[1] +
///      wait_group(1) (depth=2 outer pipeline setup).
///   7. **Single-iteration main loop body** (kv_iter=0, slot=0):
///        - Phase A: `pipeline::emit_main_loop_phase_a_load` +
///          `projection_mma::emit_kv_projection_chunk_loop`.
///        - Phase B: `attention_mma::emit_phase_b_attention`.
///        - Phase C: `pipeline::emit_main_loop_phase_c_swap`.
///   8. Finalize per q_tile_iter (reused).
///   9. `ret;` + closing brace.
///
/// # B1.6 deferral #4 RESOLVED (register hoist)
///
/// Per-warp accumulator registers (`%q_acc_*`, `%k_acc_*`, `%v_acc_*`,
/// `%s_acc_*`, `%o_acc_*`, `%p_packed_*`) and Phase B placeholder SMEM
/// base regs (`%tb1_phase_b_smem_q/k/v`) are now declared + zero-init'd
/// at step 2b via `register_budget::declare_registers`. The sub-helpers
/// (`emit_q_projection`, `emit_kv_projection_chunk_loop`, the three
/// `attention_mma.rs` sub-helpers) no longer emit `.reg` declarations
/// for these names. Multi-call is now ptxas-safe — which resolves the
/// architectural blocker on the `for kv_iter in 0..n_kv_iters` main loop
/// expansion. The expansion itself lands in the immediate follow-up
/// commit; this commit keeps the single-iter body for diff isolability.
///
/// O_acc cross-iter lifetime (formerly deferral #8) and softmax running
/// max/sum state register lifetime (formerly deferral #7) are both
/// architecturally the same hoist viewed from different sub-helpers —
/// all three deferrals (#4 + #7 + #8) are resolved by this commit.
///
/// # FLIP-POINT for B1.6
///
/// When B1.6 hoists the register declarations and wires the multi-iter
/// loop, the dispatch test `tier_b1_eligible_routes_to_synthesize_stub`
/// in `tests/tier_b1_dispatch.rs` (currently keyed off "Tier B.1 single-iter
/// scaffold complete") will need its assertion updated. B1.5 Task 5.3
/// replaces the stub marker with a `.visible .entry`-anchored kernel; the
/// test's marker becomes either the orchestrator's tail-comment sentinel or
/// the `synthesize`-function entry-label name.
pub fn synthesize(config: &FlashAttentionConfig, chunk: u32) -> Vec<u8> {
    let mut ptx = String::new();

    // 1. PTX header (target, version, param block, register decls).
    crate::flash_attention_v2::phases::forward::prelude::emit(&mut ptx, config);

    // 2. Tier B.1 outer-prologue register declarations.
    // Required by pipeline::emit_main_loop_phase_c_swap (the helper
    // references these names; B1.4's rustdoc lists them as a prereq).
    ptx.push_str("    .reg .u32 %r_kv_iter_next, %r_n_kv;\n");
    ptx.push_str("    .reg .b32 %r_slot_curr;\n");
    ptx.push_str("    .reg .pred %p_prefetch;\n");
    // Initialize slot_curr to 0 (slot[0] is the first slab consumed).
    ptx.push_str("    mov.b32 %r_slot_curr, 0;\n");
    // kv_iter_next / n_kv: placeholders for B1.6 (single-iter scaffold
    // so the predicate falls through to the AFTER label every time).
    ptx.push_str("    mov.u32 %r_kv_iter_next, 1; // single-iter scaffold (B1.6)\n");
    ptx.push_str("    mov.u32 %r_n_kv, 1; // single-iter scaffold (B1.6)\n");

    // 2b. Tier B.1 per-warp accumulator + scratch register declarations
    // (B1.6 deferral #4 resolution). Hoisting these out of the sub-helpers
    // means each helper can be called any number of times on the same PTX
    // string without ptxas duplicate-declaration rejection. This unlocks
    // the multi-iter main loop expansion that lands in the immediate
    // follow-up commit.
    register_budget::declare_registers(&mut ptx, config);

    // 3. CSHA A.4 active_heads guard (reused from Tier A unchanged).
    crate::flash_attention_v2::phases::forward::csha_hooks::emit_active_heads_guard(
        &mut ptx, config,
    );

    // 4. RMSNorm pre-pass. Each q_tile_iter normalises its rows into
    // csha_x_ptr. 8-warp partition extension deferred to B1.6 (Task 5.2
    // module rustdoc Phase-B deferral #4 covers the broader hoist).
    let iters = (config.block_q as u32).div_ceil(4);
    for q_iter in 0..iters {
        crate::flash_attention_v2::phases::forward::csha_hooks::emit_prologue(
            &mut ptx, config, q_iter,
        );
    }

    // 5. Q projection (chunk-streamed MMA). Per spec section 4.2 prologue.
    projection_mma::emit_q_projection(&mut ptx, config, chunk);

    // 6. cp.async outer prologue: prime slot[0] + slot[1].
    pipeline::emit_prologue_kicks(&mut ptx, config);

    // 7. Single-iteration main loop body (kv_iter=0, slot=0).
    // Loop expansion to all kv_iters is B1.6 (requires the register-hoist
    // refactor in attention_mma::emit_phase_b_attention's Phase-B
    // deferral #4 to avoid duplicate .reg declarations on multi-call).
    let kv_iter: u32 = 0;
    let slot: u32 = 0;
    pipeline::emit_main_loop_phase_a_load(&mut ptx, config, kv_iter);
    projection_mma::emit_kv_projection_chunk_loop(&mut ptx, config, chunk, kv_iter, slot);
    attention_mma::emit_phase_b_attention(&mut ptx, config, kv_iter, slot);
    pipeline::emit_main_loop_phase_c_swap(&mut ptx, config, kv_iter);

    // 8. Finalize: divide O_acc by row_sum + scatter to HBM + LSE.
    //
    // B1.6 deferral #9 RESOLVED: Tier-B1-native scatter (option b from
    // the deferral memo) is implemented in `finalize::emit`. It reads
    // `%o_acc_<t>_<lane>` + `%s_sum_<t>_<half>` + `%s_max_<t>_<half>` —
    // the Tier B.1 D-fragment-shaped accumulator namespace — and writes
    // the full block_q × head_dim output tile to HBM with per-tile warp
    // gating matching the PV MMA ownership pattern. LSE = row_max +
    // ln(row_sum) is emitted by one lane per row-quad per half.
    //
    // The Tier A `phases::forward::finalize::emit` is no longer called
    // from this orchestrator. It was structurally invalid here: Tier A's
    // one-q-row-per-warp register namespace (%f<O_BASE>, %row_sum) does
    // not exist in Tier B.1's m16n8k16 D-fragment layout.
    finalize::emit(&mut ptx, config);

    // 9. Sentinel comment + return. The sentinel is load-bearing for
    // the dispatch test (replaces the prior "Tier B.1 stub" marker).
    ptx.push_str("    // Tier B.1 single-iter scaffold complete (B1.5 Task 5.3); multi-iter loop B1.6\n");
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    // B1.6 deferral #10 RESOLVED: `phases::forward::prelude::emit` is now
    // config-aware (consults `config.gpu_sm`) and emits `.target sm_80`
    // directly when gpu_sm >= 80. The prior stopgap `String::replace`
    // post-rewrite is no longer needed.

    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}
