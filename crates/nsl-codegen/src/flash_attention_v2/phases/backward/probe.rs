//! CSHA cycle 20 T1 — dS-probe PTX emission helper.
//!
//! **Scope:** cycle-18 DEGENERATE-PROBE meta-lesson gate. This module
//! provides the predicated `st.global.f32` emission for the 8-slot probe
//! layout {row_max, row_sum, S_pre_mask, P, dP, rowsum_dP_P, dS, scale*dS}
//! sampled at the non-degenerate coordinate
//! `(batch=0, head=0, q_tile_iter=0, warp_row=1, lane=0, causal=true)`.
//!
//! **Per-slot column coordinate.**
//!   * Slots 0-6 (row_max, row_sum, S_pre_mask, P, dP, rowsum_dP_P, dS)
//!     are emitted from ds_compute BEFORE the col-loop advances, i.e. they
//!     naturally sample col=0 under the (warp_id==1, lane==0, ...) gate.
//!   * Slot 7 (scale*dS) is emitted from `dqdk_accum` INSIDE the KV-col
//!     loop. `maybe_emit_probe_store` provides no col-gate hook, so the
//!     store fires once per col iteration and slot 7 retains the LAST
//!     column's value — i.e. it samples col=block_kv-1, NOT col=0.
//!     Consumers (T5/c21 CPU-reference matching) MUST interpret slot 7
//!     at col=block_kv-1.
//!
//! **Feature gating.** The probe machinery is gated at PTX-emission time
//! by the `csha_cycle19_probe` Cargo feature — when the feature is OFF,
//! `maybe_emit_probe_store` is a compile-time no-op AND the prelude does
//! not widen the param block. This guarantees the 25/25 `fa_v2_snapshots`
//! remain byte-identical on default feature configurations.
//!
//! **Runtime gating (feature ON).** Even when the feature is enabled the
//! probe stores are gated by a PTX predicate `%p_probe_active` that OR's
//! together:
//!   1. `probe_ds_out_ptr != 0` (loaded once in prelude), AND
//!   2. `q_tile_iter == 0` (compile-time — this helper is only called
//!       from q_tile_iter=0 emission sites; the `q_tile_iter != 0`
//!       caller path is a no-op), AND
//!   3. `%warp_id == 1 && %lane == 0` (composed into `%p_probe_gate`
//!       at each store site), AND
//!   4. `%batch_idx == 0 && %head_idx == 0` (composed the same way).
//!
//! **Global-slot address computation.** The 8 slots are contiguous f32:
//!   slot_addr = probe_ds_out_ptr + slot_idx * 4
//!
//! Each store predicate falls through on ANY of the gate conditions
//! failing — the compile-time `q_tile_iter != 0` check simply skips
//! emission entirely. **No col-gate hook** — see the per-slot column
//! coordinate note above; slot 7's col=block_kv-1 sampling is a
//! consequence of this limitation.
//!
//! **Non-CSHA config safety.** Callers should invoke this helper only
//! from backward emission sites — the register prerequisites (`%warp_id`,
//! `%lane`, `%batch_idx`, `%head_idx`, `%rd_probe_ds`, `%p_probe_active`)
//! are declared unconditionally by the backward prelude when the feature
//! is enabled.

use crate::flash_attention::FlashAttentionConfig;

/// Emit a predicated `st.global.f32` for one probe slot.
///
/// * `slot_idx` — one of 0..=7 mapping to the layout above.
/// * `value_reg` — PTX register name holding the f32 value to store
///   (e.g. `"%f_P"`, `"%f_dS"`).
/// * `q_tile_iter` — compile-time iter index; emission is skipped when
///   nonzero.
///
/// **Feature OFF:** unconditional no-op (function body compiled out).
/// **Feature ON, q_tile_iter != 0:** unconditional no-op.
/// **Feature ON, q_tile_iter == 0:** emits 3 setp/and.pred + one
/// predicated `st.global.f32`.
///
/// **Col-gate limitation (future maintainers).** This helper composes a
/// fixed gate (probe_active AND warp_id==1 AND lane==0 AND batch==0 AND
/// head==0) into `%p_probe_gate`. It does NOT accept an extra predicate
/// to compose an additional gate (e.g. `col==0`) BEFORE the internal
/// `setp.eq.u32 %p_probe_w, %warp_id, 1` overwrites `%p_probe_w`. As a
/// result, callers emitting from inside a col-loop (slot 7 at
/// `dqdk_accum`) cause the store to fire once per col iteration; the
/// slot retains the LAST column's value. If a tight col=0 (or arbitrary
/// extra-predicate) gate is required, extend this API with an
/// `Option<&str>` extra-predicate register that composes into
/// `%p_probe_gate` before line 73 — see R11 (cycle-20 T1-fixup) for the
/// design.
#[cfg(feature = "csha_cycle19_probe")]
pub fn maybe_emit_probe_store(
    ptx: &mut String,
    _config: &FlashAttentionConfig,
    slot_idx: u32,
    value_reg: &str,
    q_tile_iter: u32,
) {
    if q_tile_iter != 0 {
        return;
    }
    assert!(slot_idx < 8, "cycle-20 T1 probe layout is 8 slots wide (got {slot_idx})");
    let byte_off = slot_idx * 4;

    ptx.push_str(&format!(
        "    // ── cycle-20 T1 probe slot {slot_idx} := {value_reg} ──\n"
    ));
    // Compose per-site gate: p_probe_active (from prelude) &&
    // %warp_id == 1 && %lane == 0 && %batch_idx == 0 && %head_idx == 0.
    // Result is placed in %p_probe_gate (reserved register in prelude).
    ptx.push_str("    setp.eq.u32 %p_probe_w, %warp_id, 1;\n");
    ptx.push_str("    setp.eq.u32 %p_probe_l, %lane, 0;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_w, %p_probe_l;\n");
    ptx.push_str("    setp.eq.u64 %p_probe_b, %batch_idx, 0;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_gate, %p_probe_b;\n");
    ptx.push_str("    setp.eq.u64 %p_probe_h, %head_idx, 0;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_gate, %p_probe_h;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_gate, %p_probe_active;\n");
    // Compute slot address: %rd_probe_slot = %rd_probe_ds + byte_off.
    if byte_off == 0 {
        ptx.push_str("    mov.u64 %rd_probe_slot, %rd_probe_ds;\n");
    } else {
        ptx.push_str(&format!(
            "    add.u64 %rd_probe_slot, %rd_probe_ds, {byte_off};\n"
        ));
    }
    ptx.push_str(&format!(
        "    @%p_probe_gate st.global.f32 [%rd_probe_slot], {value_reg};\n"
    ));
}

/// No-op stub when the feature is disabled — ensures the emitter
/// call sites compile unconditionally without cfg noise at each site.
#[cfg(not(feature = "csha_cycle19_probe"))]
#[inline(always)]
pub fn maybe_emit_probe_store(
    _ptx: &mut String,
    _config: &FlashAttentionConfig,
    _slot_idx: u32,
    _value_reg: &str,
    _q_tile_iter: u32,
) {
    // Feature off — probe emission is a compile-time no-op.
}

/// CSHA cycle 20 T2 — dV-probe emission helper.
///
/// Analogous to `maybe_emit_probe_store` but writes to `%rd_probe_dv`
/// (loaded once in prelude from `.param .u64 probe_dv_out_ptr`). Uses a
/// 5-slot layout for the dV-side chain:
///
///   Slot 0: dV_P            — %f_P read at dv_accum FMA site
///   Slot 1: dV_accum_pre    — %f0 SMEM cell PRE-FMA at dv_accum
///   Slot 2: dV_dO_f32       — %f1 dO cvt.f32.f16 result at dv_accum
///   Slot 3: dV_final        — final dV SMEM cell at finalize readback
///   Slot 4: dV_cross_check  — reconstructed slot0 * slot2 + slot1
///                             (FMA reconstruction, matches slot 3 iff
///                             the SMEM path is intact — bisects
///                             hypothesis (iii) inter-phase corruption
///                             vs (iv) accumulator write path)
///
/// **Coord gate.** The 5 stores are gated at the same runtime coord as
/// the T1 dS probe (batch=0, head=0, warp_id==1, lane==0) so both probes
/// sample the SAME representative cell. The additional coord fields
/// documented in the T2 task (row_global=1, r1=0, lane=20, warp of d=37,
/// kv_outer_iter=8) are the ORACLE-SIDE coordinates the CPU reference
/// will compare against; the PTX-side gate stays uniform with dS to keep
/// the helper reusable. Follow-on c21 refinements can widen the gate
/// signature if per-slot coord differentiation becomes necessary.
///
/// **Feature OFF:** unconditional no-op (function body compiled out).
/// **Feature ON, q_tile_iter != 0:** unconditional no-op.
/// **Feature ON, q_tile_iter == 0:** emits gate composition + one
/// predicated `st.global.f32` per call.
#[cfg(feature = "csha_cycle19_probe")]
pub fn maybe_emit_dv_probe_store(
    ptx: &mut String,
    _config: &FlashAttentionConfig,
    slot_idx: u32,
    value_reg: &str,
    q_tile_iter: u32,
) {
    if q_tile_iter != 0 {
        return;
    }
    assert!(slot_idx < 5, "cycle-20 T2 dV probe layout is 5 slots wide (got {slot_idx})");
    let byte_off = slot_idx * 4;

    ptx.push_str(&format!(
        "    // ── cycle-20 T2 dV probe slot {slot_idx} := {value_reg} ──\n"
    ));
    // Compose per-site gate: %p_probe_active_dv (rd_probe_dv != 0, set
    // ONCE in prelude) && %warp_id == 1 && %lane == 0 && %batch_idx == 0
    // && %head_idx == 0.
    //
    // R11 fix (cycle-20 T2 fixup): earlier drafts re-derived
    // %p_probe_active from %rd_probe_dv here, which CLOBBERED the
    // prelude-set %p_probe_active (derived from %rd_probe_ds). Since
    // dv_accum emission runs BEFORE dqdk_accum in the backward pipeline,
    // that clobber silently suppressed the dS slot-7 store (probe_ds!=0
    // && probe_dv==0 caller) and null-faulted the dS pipeline (probe_ds==0
    // && probe_dv!=0 caller). The fix: the prelude now emits a SEPARATE
    // %p_probe_active_dv predicate; the ds-side helper still consumes
    // %p_probe_active, and this helper references %p_probe_active_dv —
    // making dS and dV probe activation ORDER-INDEPENDENT across the
    // entire backward emission sequence.
    ptx.push_str("    setp.eq.u32 %p_probe_w, %warp_id, 1;\n");
    ptx.push_str("    setp.eq.u32 %p_probe_l, %lane, 0;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_w, %p_probe_l;\n");
    ptx.push_str("    setp.eq.u64 %p_probe_b, %batch_idx, 0;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_gate, %p_probe_b;\n");
    ptx.push_str("    setp.eq.u64 %p_probe_h, %head_idx, 0;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_gate, %p_probe_h;\n");
    ptx.push_str("    and.pred %p_probe_gate, %p_probe_gate, %p_probe_active_dv;\n");
    // Compute slot address: %rd_probe_slot = %rd_probe_dv + byte_off.
    if byte_off == 0 {
        ptx.push_str("    mov.u64 %rd_probe_slot, %rd_probe_dv;\n");
    } else {
        ptx.push_str(&format!(
            "    add.u64 %rd_probe_slot, %rd_probe_dv, {byte_off};\n"
        ));
    }
    ptx.push_str(&format!(
        "    @%p_probe_gate st.global.f32 [%rd_probe_slot], {value_reg};\n"
    ));
}

/// Feature-OFF stub — compile-time no-op keeping call sites cfg-free.
#[cfg(not(feature = "csha_cycle19_probe"))]
#[inline(always)]
pub fn maybe_emit_dv_probe_store(
    _ptx: &mut String,
    _config: &FlashAttentionConfig,
    _slot_idx: u32,
    _value_reg: &str,
    _q_tile_iter: u32,
) {
    // Feature off — dV probe emission is a compile-time no-op.
}

#[cfg(all(test, feature = "csha_cycle19_probe"))]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: true, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model: 32,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        }
    }

    #[test]
    fn probe_store_emits_gated_stg_at_q_tile_iter_zero() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_probe_store(&mut ptx, &c, 6, "%f_dS", 0);
        assert!(ptx.contains("st.global.f32"), "must emit store");
        assert!(ptx.contains("%p_probe_gate"), "must be predicated");
        assert!(ptx.contains("%rd_probe_ds"),
            "must reference prelude-declared probe pointer register");
    }

    #[test]
    fn probe_store_is_noop_at_nonzero_q_tile_iter() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_probe_store(&mut ptx, &c, 6, "%f_dS", 1);
        assert!(ptx.is_empty(),
            "cycle-20 T1: q_tile_iter != 0 emission is a no-op");
    }

    #[test]
    fn probe_store_slot_zero_uses_direct_mov() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_probe_store(&mut ptx, &c, 0, "%row_max", 0);
        assert!(ptx.contains("mov.u64 %rd_probe_slot, %rd_probe_ds"),
            "slot 0 should use mov not add");
    }

    #[test]
    fn probe_store_slot_seven_uses_add_28_offset() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_probe_store(&mut ptx, &c, 7, "%f_dS", 0);
        assert!(ptx.contains("add.u64 %rd_probe_slot, %rd_probe_ds, 28"),
            "slot 7 byte offset is 7*4=28");
    }

    // ── cycle 20 T2 dV-probe helper tests ──

    #[test]
    fn dv_probe_store_emits_gated_stg_at_q_tile_iter_zero() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_dv_probe_store(&mut ptx, &c, 0, "%f_P", 0);
        assert!(ptx.contains("st.global.f32"), "must emit store");
        assert!(ptx.contains("%p_probe_gate"), "must be predicated");
        assert!(ptx.contains("%rd_probe_dv"),
            "must reference prelude-declared dV probe pointer register");
    }

    #[test]
    fn dv_probe_store_is_noop_at_nonzero_q_tile_iter() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_dv_probe_store(&mut ptx, &c, 0, "%f_P", 1);
        assert!(ptx.is_empty(),
            "cycle-20 T2 dV probe: q_tile_iter != 0 emission is a no-op");
    }

    #[test]
    fn dv_probe_store_slot_zero_uses_direct_mov() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_dv_probe_store(&mut ptx, &c, 0, "%f_P", 0);
        assert!(ptx.contains("mov.u64 %rd_probe_slot, %rd_probe_dv"),
            "slot 0 should use mov not add");
    }

    #[test]
    fn dv_probe_store_slot_four_uses_add_16_offset() {
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_dv_probe_store(&mut ptx, &c, 4, "%f_dk_tmp", 0);
        assert!(ptx.contains("add.u64 %rd_probe_slot, %rd_probe_dv, 16"),
            "slot 4 byte offset is 4*4=16");
    }

    #[test]
    fn dv_probe_uses_separate_active_predicate() {
        // R11 fix (cycle-20 T2 fixup): the dV helper must reference the
        // prelude-set %p_probe_active_dv register (NOT %p_probe_active).
        // This ensures dS and dV probes have independent activation and
        // ordering of ds_compute / dv_accum / dqdk_accum / finalize does
        // not silently suppress or null-fault either probe pipeline.
        let c = cfg();
        let mut ptx = String::new();
        maybe_emit_dv_probe_store(&mut ptx, &c, 2, "%f1", 0);
        assert!(
            ptx.contains("and.pred %p_probe_gate, %p_probe_gate, %p_probe_active_dv"),
            "dV gate AND-chain must consume %p_probe_active_dv (separate predicate)"
        );
        // And crucially — the helper must NOT re-derive %p_probe_active
        // (would clobber the ds-side prelude register).
        assert!(
            !ptx.contains("setp.ne.u64 %p_probe_active,"),
            "dV helper must not re-derive %p_probe_active — that would \
             clobber the prelude-set dS predicate (R11)"
        );
        // Nor should the AND-chain reference the ds-side %p_probe_active.
        assert!(
            !ptx.contains("and.pred %p_probe_gate, %p_probe_gate, %p_probe_active;"),
            "dV gate must NOT reference %p_probe_active (ds-side predicate)"
        );
    }
}
