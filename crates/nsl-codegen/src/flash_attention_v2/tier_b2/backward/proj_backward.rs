//! Tier B.2 projection-backward kernel (`tier_b2_proj_backward`) building block.
//!
//! The projection-backward kernel reuses the scalar `emit_dproj` /
//! `emit_drmsnorm` emitters (in `phases/backward/csha_hooks_backward.rs`).
//! Those emitters READ dQ/dK/dV from SMEM as row-major `[block_q, head_dim]`
//! f32 tiles anchored at `backward_d{q,k,v}_offset(config)` off `%shmem_base`.
//!
//! The Tier B.2 dQ/dK/dV kernels, however, write their gradients to HBM as
//! f32 (see `tier_b2/backward/dq.rs` / `dkdv.rs` finalize stores, all
//! `st.global.f32`). This module bridges the two: `emit_dqkv_hbm_to_smem_load`
//! cooperatively loads each HBM gradient buffer into the EXACT SMEM slot that
//! `emit_dproj` subsequently reads, so the scalar dproj/dRMSNorm math runs
//! unchanged against the Tier B.2 gradients.
//!
//! Dtype: f32 throughout. The HBM dQ/dK/dV buffers are f32 (Tier B.2 dQ/dK/dV
//! kernels emit `st.global.f32`) and `emit_dproj` reads f32 from SMEM, so this
//! emitter does a straight `ld.global.f32` -> `st.shared.f32` with no narrowing.
//! It is READ-ONLY on the HBM buffers (loads only; never stores back).

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::phases;
use crate::flash_attention_v2::smem_layout::{
    backward_dk_offset, backward_dq_offset, backward_dv_offset,
    tier_b2_proj_backward_smem_base,
};
use crate::flash_attention_v2::tier_b2::backward::BackwardSynthError;

/// Cooperatively load the three HBM dQ/dK/dV gradient buffers into the SMEM
/// slots that `emit_dproj` reads (`backward_d{q,k,v}_offset`).
///
/// SMEM layout produced (mirrors `emit_dproj`'s read addressing):
///   SMEM[backward_dq_offset + (s*head_dim + j)*4] = dQ[q_start+s, j]   (f32)
///   SMEM[backward_dk_offset + (s*head_dim + j)*4] = dK[q_start+s, j]   (f32)
///   SMEM[backward_dv_offset + (s*head_dim + j)*4] = dV[q_start+s, j]   (f32)
///
/// HBM source layout (mirrors `emit_xnorm_recompute`'s row addressing for the
/// `[head, seq, head_dim]` f32 buffers under the smoke scope):
///   row_global = q_start + s
///   flat       = head_idx*seq + row_global        (seq in %rd6)
///   byte_off   = flat * head_dim * 4 + j*4        (head_dim in %rd7)
///   src        = base_ptr + byte_off
///
/// Work partition (mirrors `emit_dproj`): 128 threads over `block_q*head_dim`
/// cells. Thread owns `cell = tid_x + k*128`, guarded by `cell < block_q*head_dim`;
/// `s = cell / head_dim`, `j = cell % head_dim`.
///
/// ASSUMES the kernel prelude (orchestrator task) has already declared the
/// shared register pool: `%shmem_base`, `%tid_x`, `%q_start`, `%head_idx`,
/// `%rd6` (seq_len), `%rd7` (head_dim), scratch `%rd_c*`, `%f_dy`, predicate
/// `%p_c0`. This emitter does NOT redeclare `.reg`.
pub fn emit_dqkv_hbm_to_smem_load(ptx: &mut String, config: &FlashAttentionConfig) {
    let csha = match config.csha.as_ref() {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier B.2 proj-backward load: csha=None, no emission\n");
            return;
        }
    };
    // d_model gates the broader proj-backward path; the load itself is keyed on
    // head_dim, but keep the same guard the sibling emitters use for parity.
    if csha.d_model == 0 {
        ptx.push_str("    // Tier B.2 proj-backward load: d_model=0, no emission\n");
        return;
    }

    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    if head_dim == 0 || block_q == 0 {
        ptx.push_str("    // Tier B.2 proj-backward load: head_dim/block_q=0, no emission\n");
        return;
    }
    let total_cells = block_q * head_dim;
    let cells_per_thread = total_cells.div_ceil(128).max(1);

    ptx.push_str(
        "    // Tier B.2 proj-backward: stage HBM dQ/dK/dV into emit_dproj SMEM slots (f32).\n",
    );

    for (label, src_reg, dst_off) in [
        ("DQ", "%rd_bwd_dq", backward_dq_offset(config)),
        ("DK", "%rd_bwd_dk", backward_dk_offset(config)),
        ("DV", "%rd_bwd_dv", backward_dv_offset(config)),
    ] {
        ptx.push_str(&format!("V2_PROJBWD_LOAD_{label}:\n"));
        // Null-guard the HBM source pointer; skip the whole buffer if absent.
        ptx.push_str(&format!("    setp.eq.u64 %p_c0, {src_reg}, 0;\n"));
        ptx.push_str(&format!("    @%p_c0 bra V2_PROJBWD_LOAD_{label}_SKIP;\n"));

        for k in 0..cells_per_thread {
            let thread_cell = k * 128;
            // cell = tid_x + k*128
            ptx.push_str("    cvt.u64.u32 %rd_c0, %tid_x;\n");
            if thread_cell > 0 {
                ptx.push_str(&format!("    add.u64 %rd_c0, %rd_c0, {thread_cell};\n"));
            }
            // guard cell < total_cells
            ptx.push_str(&format!("    setp.lt.u64 %p_c0, %rd_c0, {total_cells};\n"));
            ptx.push_str(&format!(
                "    @!%p_c0 bra V2_PROJBWD_LOAD_{label}_CELL_{k}_SKIP;\n"
            ));

            // s = cell / head_dim; j = cell % head_dim
            ptx.push_str(&format!("    div.u64 %rd_c1, %rd_c0, {head_dim};  // s\n"));
            ptx.push_str(&format!("    rem.u64 %rd_c2, %rd_c0, {head_dim};  // j\n"));

            // --- HBM source address (emit_xnorm_recompute convention) ---
            // row_global = q_start + s
            ptx.push_str("    add.u64 %rd_c3, %rd_c1, %q_start;  // row_global = q_start + s\n");
            // flat = head_idx*seq + row_global  (seq in %rd6)
            ptx.push_str("    mul.lo.u64 %rd_c4, %head_idx, %rd6;  // head_idx*seq\n");
            ptx.push_str("    add.u64 %rd_c4, %rd_c4, %rd_c3;  // + row_global\n");
            // elem = flat*head_dim + j   (head_dim in %rd7)
            ptx.push_str("    mul.lo.u64 %rd_c4, %rd_c4, %rd7;  // flat*head_dim\n");
            ptx.push_str("    add.u64 %rd_c4, %rd_c4, %rd_c2;  // + j\n");
            // byte_off = elem*4 (f32)
            ptx.push_str("    shl.b64 %rd_c4, %rd_c4, 2;  // *4 bytes (f32)\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_c4, {src_reg}, %rd_c4;  // HBM src\n"
            ));
            ptx.push_str("    ld.global.f32 %f_dy, [%rd_c4];\n");

            // --- SMEM dest address (emit_dproj layout) ---
            //   dst = %shmem_base + dst_off + (s*head_dim + j)*4
            // smem_elem = s*head_dim + j = (cell), so reuse %rd_c0 (the cell idx)
            // but recompute to keep the layout explicit and independent of cell.
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_c3, %rd_c1, {head_dim};  // s*head_dim\n"
            ));
            ptx.push_str("    add.u64 %rd_c3, %rd_c3, %rd_c2;  // + j\n");
            ptx.push_str("    shl.b64 %rd_c3, %rd_c3, 2;  // *4 bytes (f32)\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_c3, %rd_c3, {dst_off};  // + dst offset\n"
            ));
            ptx.push_str("    add.u64 %rd_c3, %shmem_base, %rd_c3;  // SMEM dest\n");
            ptx.push_str("    st.shared.f32 [%rd_c3], %f_dy;\n");

            ptx.push_str(&format!("V2_PROJBWD_LOAD_{label}_CELL_{k}_SKIP:\n"));
        }

        ptx.push_str(&format!("V2_PROJBWD_LOAD_{label}_SKIP:\n"));
    }
    ptx.push_str("    bar.sync 0;  // dQ/dK/dV SMEM tiles visible before dproj reads\n");
}

/// Synthesize the complete `tier_b2_proj_backward` kernel.
///
/// This is the 4th "projection backward" kernel of the hybrid CSHA Tier B.2
/// backward. The dQ/dK/dV gradients are produced by the (validated) `dq` and
/// `dkdv` kernels and written to HBM. THIS kernel:
///   1. loads dQ/dK/dV from HBM into the SMEM slots that the scalar
///      `emit_dproj` reads (via `emit_dqkv_hbm_to_smem_load`),
///   2. recomputes `x_norm` + `rms` from `x_raw_ptr`
///      (`emit_xnorm_recompute`),
///   3. runs the EXISTING scalar `emit_dproj` (dWq/dWk/dWv) and
///      `emit_drmsnorm` (dx) emitters UNCHANGED.
///
/// It is READ-ONLY on the HBM dQ/dK/dV buffers — the final dQ/dK/dV are the
/// other kernels' outputs, so there is NO dRoPE and NO dQ/dK/dV HBM finalize
/// store here. The only HBM writes are the dW*/dx/dx_norm stores emitted by
/// the reused `emit_dproj` / `emit_drmsnorm` (which target `%rd_bwd_dwq` /
/// `%rd_bwd_dwk` / `%rd_bwd_dwv` / `dx_ptr` / `%rd_bwd_dxn`, never the
/// dQ/dK/dV input pointers).
///
/// ## Prelude strategy: REUSE the scalar backward prelude, rename the entry.
///
/// `phases::backward::prelude::emit` already declares EVERY register and
/// param the four reused emitters reference (`%shmem_base`, `%tid_x`,
/// `%q_start`, `%head_idx`, `%batch_idx`, `%rd6`=seq, `%rd7`=head_dim, the
/// `%rd_c*` / `%f_*` / `%h_tmp` / `%p_c*` / `%p0` / `%rd30`/`%rd31` scratch,
/// the `%rd_bwd_d*` pointer regs, and the `x_raw_ptr` / `csha_eps` /
/// `csha_norm_weight_ptr` / `csha_w{q,k,v}_ptr` / `dx_ptr` params) AND emits a
/// launchable `.visible .entry` whose PARAMETER LIST is exactly the scalar
/// backward kernel's — which is precisely the requirement (so the runtime can
/// pass the same pointers). The ONLY required difference is the entry NAME.
///
/// We therefore call `prelude::emit` (with `tier_b = None` so the output is
/// the byte-identical pre-Tier-B-2 baseline) and then rename the single
/// `.visible .entry <scalar_name> (` line to `.visible .entry
/// tier_b2_proj_backward (`. This touches ZERO shared files, so the scalar
/// `synthesize_backward(config)` output is trivially unchanged. The rename
/// targets exactly one line (the generated `kernel_name` appears only in the
/// entry header).
///
/// Guards `head_dim % 32 != 0` (warp-reduction precondition shared with the
/// dQ / dK/dV kernels).
pub fn synthesize_proj_backward(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    let head_dim = config.head_dim as u32;
    if !head_dim.is_multiple_of(32) {
        return Err(BackwardSynthError::UnsupportedHeadDim(head_dim));
    }

    let mut ptx = String::new();

    // Phase 0: header, .visible .entry, .extern .shared (when dynamic SMEM),
    // SMEM decl, full register pool, scalar param loads, and grid/index setup
    // that establishes %shmem_base / %tid_x / %q_start / %head_idx / %rd6 /
    // %rd7 and all the %rd_c* / %f_* / %h_tmp / %p* scratch the reused
    // emitters use. `tier_b = None` keeps this byte-identical to the scalar
    // backward prelude baseline (Tier B.2 no-op guarantee, prelude.rs spec
    // §7.4) — we only rename the entry below.
    phases::backward::prelude::emit(&mut ptx, config, None);

    // Rename the entry from the scalar backward name to the proj-backward
    // name. The param list and body register pool are unchanged. The scalar
    // `kernel_name` string only ever appears in the `.visible .entry ... (`
    // header line, so a single targeted replace is exact and unambiguous.
    let scalar_name = phases::backward::prelude::kernel_name(config);
    let from = format!(".visible .entry {scalar_name} (");
    let to = ".visible .entry tier_b2_proj_backward (";
    debug_assert_eq!(
        ptx.matches(&from).count(),
        1,
        "expected exactly one scalar entry header to rename"
    );
    ptx = ptx.replacen(&from, to, 1);

    // SMEM rebase (T8 fix): the prelude declares + bases SMEM for the FULL
    // scalar fused-backward layout (forward Q/KV/SP/weight tiles + P/dS +
    // dQ/dK/dV + x_norm/dx_norm/rms), whose absolute footprint through the rms
    // strip is ~137 KB (hd=64) / ~113 KB (hd=128) — past the 99 KB dynamic-SMEM
    // device cap, so the standalone launch fails with CUDA_ERROR_INVALID_VALUE.
    // This kernel only references the dQ/dK/dV/x_norm/dx_norm/rms tiles (all at
    // offsets >= backward_dq_offset). Shift %shmem_base DOWN by that base so the
    // SAME emitter offsets (`%shmem_base + backward_*_offset`) land in a
    // compacted region starting at allocation byte 0. The launch then allocates
    // only `tier_b2_proj_backward_smem_bytes` (~88 KB). One subtract, applied
    // before any proj phase touches SMEM (the active_heads guard uses none).
    let proj_base = tier_b2_proj_backward_smem_base(config);
    if proj_base > 0 {
        ptx.push_str(&format!(
            "    // T8: rebase %shmem_base for compacted standalone proj SMEM \
             (base={proj_base})\n"
        ));
        ptx.push_str(&format!(
            "    sub.u64 %shmem_base, %shmem_base, {proj_base};\n"
        ));
    }

    // CSHA A.4 head-pruning guard — mirror the scalar backward orchestrator
    // so blocks whose head_idx >= csha_active_heads early-`ret` before doing
    // any projection-backward work. Reuses the forward emitter; its scratch
    // (%r10/%r11/%p0) and %head_idx are declared by the prelude pool, and
    // `csha_active_heads` is in the param list. Placement is after the
    // prelude's (segment_masked) bar.sync 0, identical to the scalar path.
    phases::csha_hooks::emit_active_heads_guard(&mut ptx, config);

    // T2 load: stage HBM dQ/dK/dV into the exact SMEM slots emit_dproj reads.
    emit_dqkv_hbm_to_smem_load(&mut ptx, config);
    ptx.push_str("    bar.sync 0;  // dQ/dK/dV SMEM staged before x_norm recompute\n");

    // Recompute x_norm + rms (the forward did not persist them for backward).
    phases::backward::csha_hooks_backward::emit_xnorm_recompute(&mut ptx, config);
    ptx.push_str("    bar.sync 0;  // x_norm + rms visible before dproj / dRMSNorm\n");

    // Sprint 10: rope_q integration. When rope_q=true, dQ/dK live in HBM in
    // the post-RoPE basis (the basis of q_saved/k_saved that the dq/dkdv
    // kernels backproped through), but emit_dproj needs them in the pre-RoPE
    // basis so dWq = x_norm^T @ dQ_preRoPE. `emit_drope` rotates the dQ/dK
    // SMEM tiles in-place using the same inverse rotation the scalar Tier C
    // path applies (see `mod.rs:1224`, matching phase order
    // xnorm_recompute -> drope -> dproj -> drmsnorm). emit_drope already
    // includes its own internal null-guard on cos_ptr/sin_ptr and a final
    // bar.sync, and is a no-op when rope_q=false / csha=None (so the
    // rope_q=false PTX is byte-identical to the pre-Sprint-10 baseline).
    //
    // Preconditions vs proj_backward's scope:
    //   - dQ/dK SMEM tiles are staged into backward_d{q,k}_offset by
    //     emit_dqkv_hbm_to_smem_load above + bar.sync — emit_drope reads
    //     and writes those exact slots.
    //   - The hybrid path's segment_masked-RoPE-reset (config.segment_masked
    //     && config.rope_q) branch inside emit_drope requires %seg_base +
    //     smem_doc_starts to be initialized; that is the scalar prelude's
    //     responsibility (gated identically on segment_masked && rope_q in
    //     prelude.rs:271) and the proj_backward kernel reuses that prelude
    //     unchanged. Sentinel-disabled paths (segment_masked=false) are
    //     trivially safe.
    //
    // Byte-identity gate: emit_drope itself emits a "no emission" comment
    // when rope_q=false (csha_hooks_backward.rs:192-194), which would drift
    // the rope_q=false PTX away from the pre-Sprint-10 baseline. We therefore
    // gate the CALL externally — when !config.rope_q the call site emits
    // ZERO bytes, preserving exact byte-for-byte equality on the historical
    // rope_q=false path that all existing tests + the production wengert
    // lowering exercise.
    if config.rope_q {
        phases::backward::csha_hooks_backward::emit_drope(&mut ptx, config, 0);
    }

    // dproj weight gradients (dWq/dWk/dWv) and dRMSNorm (dx) — reused scalar
    // emitters, UNCHANGED. q_tile_iter = 0: under the smoke scope seq ==
    // block_q so a single q-block covers the whole tile (matches how the
    // scalar dproj/dRMSNorm phases are invoked for the leading q-block).
    phases::backward::csha_hooks_backward::emit_dproj(&mut ptx, config, 0);
    phases::backward::csha_hooks_backward::emit_drmsnorm(&mut ptx, config, 0);

    // Entry close.
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");

    Ok(ptx)
}

#[cfg(test)]
mod synth_tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, RopeStyle};

    fn cfg(head_dim: i64, d_model: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64,
            block_kv: 64,
            head_dim,
            causal: true,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 80,
            segment_masked: false,
            csha: Some(CshaExtras {
                level: 2,
                d_model,
                active_heads: 1,
                ..Default::default()
            }),
            checkpoint: None,
        }
    }

    #[test]
    fn entry_renamed_and_phases_present() {
        let ptx = synthesize_proj_backward(&cfg(64, 64)).expect("synth ok");
        assert!(ptx.contains(".visible .entry tier_b2_proj_backward ("));
        // The scalar backward name must NOT survive the rename.
        let scalar = phases::backward::prelude::kernel_name(&cfg(64, 64));
        assert!(
            !ptx.contains(&format!(".visible .entry {scalar} (")),
            "scalar entry header should have been renamed"
        );
        assert!(ptx.contains("V2_BWD_XNORM_RECOMPUTE"));
        assert!(ptx.contains("V2_BWD_DPROJ_WQ"));
        assert!(ptx.contains("V2_BWD_DRMSNORM"));
        // Read-only on HBM dQ/dK/dV: no dRoPE phase, no dQ/dK/dV finalize.
        assert!(!ptx.contains("V2_BWD_DROPE"));
        assert!(ptx.trim_end().ends_with('}'));
    }

    #[test]
    fn rejects_bad_head_dim() {
        let err = synthesize_proj_backward(&cfg(48, 48)).unwrap_err();
        assert_eq!(err, BackwardSynthError::UnsupportedHeadDim(48));
    }

    /// Sprint 10: rope_q=true integration. When the config has rope_q=true,
    /// emit_drope MUST be inserted between emit_xnorm_recompute and
    /// emit_dproj so dQ/dK are rotated from post-RoPE (the basis q_saved/
    /// k_saved live in, which the dq/dkdv kernels backproped through) to
    /// pre-RoPE before emit_dproj computes dWq = x_norm^T @ dQ_preRoPE.
    /// Both the Q-side and K-side dRoPE labels must appear (V is never
    /// rotated by RoPE so V_LOOP is absent — same forward Adjacent-layout
    /// epilogue rule as the scalar Tier C path).
    #[test]
    fn rope_q_true_emits_drope_chain() {
        let mut c = cfg(64, 64);
        c.rope_q = true;
        let ptx = synthesize_proj_backward(&c).expect("synth ok");
        // The Q + K rotation loops AND the shared null-guard must be present.
        assert!(
            ptx.contains("V2_BWD_DROPE_GUARD_0:"),
            "rope_q=true: expected dRoPE null-guard label"
        );
        assert!(
            ptx.contains("V2_BWD_DROPE_Q_LOOP_0:"),
            "rope_q=true: expected Q-side dRoPE rotation loop"
        );
        assert!(
            ptx.contains("V2_BWD_DROPE_K_LOOP_0:"),
            "rope_q=true: expected K-side dRoPE rotation loop"
        );
        assert!(
            !ptx.contains("V2_BWD_DROPE_V_LOOP"),
            "V is never RoPE-rotated; V_LOOP must not be emitted"
        );

        // Ordering invariant: x_norm_recompute -> dRoPE -> dproj.
        // The scalar Tier C orchestrator (mod.rs:1223-1225) emits the same
        // sequence; this gate guards against a future refactor moving the
        // call before the x_norm recompute (would still be correct on its
        // own, but breaks parity with the scalar phase order).
        let xnorm_pos = ptx.find("V2_BWD_XNORM_RECOMPUTE")
            .expect("xnorm recompute label present");
        let drope_pos = ptx.find("V2_BWD_DROPE_GUARD_0:")
            .expect("dRoPE guard label present");
        let dproj_pos = ptx.find("V2_BWD_DPROJ_WQ")
            .expect("dproj label present");
        assert!(
            xnorm_pos < drope_pos && drope_pos < dproj_pos,
            "phase order must be xnorm_recompute < dRoPE < dproj \
             (xnorm={xnorm_pos}, drope={drope_pos}, dproj={dproj_pos})"
        );
    }

    /// Sprint 10: byte-identity guard for the rope_q=false path. The
    /// instructions require that adding the rope_q=true wiring MUST NOT
    /// change the emitted PTX one byte for any rope_q=false config; this
    /// test verifies the external gate (`if config.rope_q { ... }` in
    /// `synthesize_proj_backward`) achieves that. Without the external
    /// gate, emit_drope still emits a "no emission" comment which would
    /// drift the rope_q=false PTX away from the pre-Sprint-10 baseline.
    #[test]
    fn rope_q_false_emits_no_drope_marker() {
        let ptx = synthesize_proj_backward(&cfg(64, 64)).expect("synth ok");
        assert!(
            !ptx.contains("V2_BWD_DROPE"),
            "rope_q=false: NO dRoPE label may appear"
        );
        // The "no emission" comment itself must also be absent — emit_drope
        // is gated at the CALL site, not internally, so it is never invoked
        // for rope_q=false. This is the byte-identity guarantee.
        assert!(
            !ptx.contains("Tier C dRoPE: rope_q=false"),
            "rope_q=false: emit_drope's internal no-op comment MUST NOT appear \
             (call must be gated externally for byte identity)"
        );
    }
}
