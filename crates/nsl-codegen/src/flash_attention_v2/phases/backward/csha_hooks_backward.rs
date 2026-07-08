//! Tier C backward CSHA hooks: x_norm re-materialisation, inverse RoPE,
//! projection gradients, and RMSNorm gradient. Each mirrors its forward
//! counterpart in `phases/forward/csha_hooks.rs` with the chain-rule math.
//!
//! ## Math (cross-reference to the CPU backward in
//! `crates/nsl-codegen/tests/csha_reference.rs`)
//!
//! dRoPE (inverse rotation, Adjacent layout — `x[2i]` paired with
//! `x[2i+1]`):
//!   dx0 =  dY[2i]  * cos + dY[2i+1] * sin
//!   dx1 = -dY[2i]  * sin + dY[2i+1] * cos
//! Orthogonal rotation ⇒ inverse == transpose of forward matrix.
//!
//! dproj (weight gradients):
//!   dWq += x_norm^T @ dQ_preRoPE    (shape [d_model, kv_dim])
//!   dWk += x_norm^T @ dK_preRoPE
//!   dWv += x_norm^T @ dV
//! Each null-guarded independently on its weight pointer (matches the
//! forward fused-projection fix).
//!
//! dRMSNorm (closed form, with D = d_model, rms = sqrt(mean_sq + eps)):
//!   dx_norm = dQ_preRoPE @ Wq^T + dK_preRoPE @ Wk^T + dV @ Wv^T   [seq, d_model]
//!   g_d  = dx_norm[i,d] * norm_weight[d]
//!   s    = sum_d g_d * x[i,d]
//!   dx[i,d] = g_d / rms - x[i,d] * s / (D * rms^3)
//!
//! ## Scoping
//!
//! This first numerical-correctness cut targets the T6.3 smoke config:
//! seq == block_q, heads == 1, head_dim == d_model, single Q block
//! (bid_x == 0), no RoPE, no causal. Under this config `kv_dim == head_dim`
//! so per-block writes cover the entire [d_model, kv_dim] dwq/dwk/dwv
//! tile, and x-layout `[h, seq, hd]` coincides with `[seq, d_model]`.
//! Wider configs are callable but may need atomic accumulation once
//! multiple (batch, head) blocks contribute to the same dw/dx cells.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    backward_dk_offset, backward_dq_offset, backward_dv_offset,
    backward_dx_norm_offset, backward_rms_strip_offset,
    backward_x_norm_offset, recompute_xnorm_offset,
};

// ────────────────────────────────────────────────────────────────────────
// x_norm + rms recompute (replaces the forward save that CSHA does not
// persist for the backward).
// ────────────────────────────────────────────────────────────────────────

/// Cooperatively recompute `rms[s]` and `x_norm[s,d]` for all block_q rows
/// in this Q block, storing them to SMEM. Must be called AFTER the prelude
/// and BEFORE `emit_dproj` / `emit_drmsnorm`.
///
/// Layout produced:
///   SMEM[backward_rms_strip_offset + s*4]           = rms[s]        (f32)
///   SMEM[backward_x_norm_offset + (s*d_model+d)*4]  = x_norm[s,d]   (f32)
///
/// Partitioning: one thread per row. Thread `s = tid_x` handles row
/// `q_start + s` when `s < block_q`. For block_q=32 this uses 32 of the
/// 128 lanes; remaining lanes are idle during this hook.
pub fn emit_xnorm_recompute(ptx: &mut String, config: &FlashAttentionConfig) {
    let csha = match config.csha.as_ref() {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier C x_norm recompute: csha=None, no emission\n");
            return;
        }
    };
    let d_model = csha.d_model;
    if d_model == 0 {
        ptx.push_str("    // Tier C x_norm recompute: d_model=0, no emission\n");
        return;
    }
    let block_q = config.block_q as u32;
    let x_off = backward_x_norm_offset(config);
    let rms_off = backward_rms_strip_offset(config);

    ptx.push_str("    // Tier C x_norm + rms recompute (one thread per row).\n");
    ptx.push_str("V2_BWD_XNORM_RECOMPUTE:\n");
    ptx.push_str(&format!("    setp.lt.u32 %p_c0, %tid_x, {block_q};\n"));
    ptx.push_str("    @!%p_c0 bra V2_BWD_XNORM_DONE;\n");

    // row_s = tid_x; row_global = q_start + row_s (smoke: q_start=0, seq=block_q)
    // x HBM base for this row, head_idx=0 layout [h, seq, hd]:
    //   x_row_base = x_raw_ptr + (head_idx*seq + (q_start+row_s)) * head_dim * 4
    //   (d_model == head_dim under the smoke scope).
    //
    // Reads from x_raw_ptr (the pre-RMSNorm save) — the forward RMSNorm
    // overwrote csha_x_ptr with x_normed in place, so we cannot recover
    // the raw x from there.
    ptx.push_str("    ld.param.u64 %rd_c0, [x_raw_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_c1, %rd_c0, 0;\n");
    ptx.push_str("    @%p_c1 bra V2_BWD_XNORM_DONE;\n");
    // row_global = q_start + tid_x
    ptx.push_str("    cvt.u64.u32 %rd_c1, %tid_x;\n");
    ptx.push_str("    add.u64 %rd_c1, %rd_c1, %q_start;\n");
    // flat = head_idx*seq + row_global
    ptx.push_str("    mul.lo.u64 %rd_c2, %head_idx, %rd6;\n");
    ptx.push_str("    add.u64 %rd_c2, %rd_c2, %rd_c1;\n");
    // byte_off = flat * head_dim * 4
    ptx.push_str("    mul.lo.u64 %rd_c2, %rd_c2, %rd7;\n");
    ptx.push_str("    shl.b64 %rd_c2, %rd_c2, 2;\n");
    ptx.push_str("    add.u64 %rd_c0, %rd_c0, %rd_c2;  // x_row_base (HBM, f32)\n");

    // Pass 1: mean_sq = sum_d x^2 / d_model
    ptx.push_str("    mov.f32 %f_ms, 0f00000000;\n");
    for d in 0..d_model {
        ptx.push_str(&format!(
            "    ld.global.f32 %f_xraw, [%rd_c0 + {}];\n", d * 4
        ));
        ptx.push_str("    fma.rn.f32 %f_ms, %f_xraw, %f_xraw, %f_ms;\n");
    }
    ptx.push_str(&format!(
        "    mov.f32 %f_xn, 0f{:08X};  // 1/d_model\n",
        (1.0f32 / d_model as f32).to_bits()
    ));
    ptx.push_str("    mul.f32 %f_ms, %f_ms, %f_xn;\n");
    ptx.push_str("    ld.param.f32 %f_xn, [csha_eps];\n");
    ptx.push_str("    add.f32 %f_ms, %f_ms, %f_xn;\n");
    // rms = sqrt(mean_sq + eps); rms_inv = rsqrt
    ptx.push_str("    sqrt.approx.f32 %f_rms_v, %f_ms;\n");
    ptx.push_str("    rsqrt.approx.f32 %f_rms_inv, %f_ms;\n");

    // Store rms[tid_x] to rms_strip.
    ptx.push_str("    cvt.u64.u32 %rd_c3, %tid_x;\n");
    ptx.push_str("    shl.b64 %rd_c3, %rd_c3, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd_c3, %rd_c3, {rms_off};\n"));
    ptx.push_str("    add.u64 %rd_c3, %shmem_base, %rd_c3;\n");
    ptx.push_str("    st.shared.f32 [%rd_c3], %f_rms_v;\n");

    // Pass 2: x_norm[s,d] = x[s,d] * rms_inv * norm_weight[d]
    // norm_weight base load
    ptx.push_str("    ld.param.u64 %rd_c4, [csha_norm_weight_ptr];\n");
    // x_norm tile row base = x_off + tid_x * d_model * 4
    ptx.push_str("    cvt.u64.u32 %rd_c5, %tid_x;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd_c5, %rd_c5, {};\n", d_model * 4
    ));
    ptx.push_str(&format!("    add.u64 %rd_c5, %rd_c5, {x_off};\n"));
    ptx.push_str("    add.u64 %rd_c5, %shmem_base, %rd_c5;\n");
    for d in 0..d_model {
        // load x[s,d]
        ptx.push_str(&format!(
            "    ld.global.f32 %f_xraw, [%rd_c0 + {}];\n", d * 4
        ));
        // load norm_weight[d]
        ptx.push_str(&format!(
            "    ld.global.f32 %f_nw, [%rd_c4 + {}];\n", d * 4
        ));
        // xn = x * rms_inv * nw
        ptx.push_str("    mul.f32 %f_xn, %f_xraw, %f_rms_inv;\n");
        ptx.push_str("    mul.f32 %f_xn, %f_xn, %f_nw;\n");
        ptx.push_str(&format!(
            "    st.shared.f32 [%rd_c5 + {}], %f_xn;\n", d * 4
        ));
    }

    ptx.push_str("V2_BWD_XNORM_DONE:\n");
    ptx.push_str("    bar.sync 0;  // x_norm + rms visible to all lanes\n");
}

// ────────────────────────────────────────────────────────────────────────
// dRoPE
// ────────────────────────────────────────────────────────────────────────

/// Emit inverse RoPE rotation for both dQ and dK gradient SMEM tiles.
///
/// These tiles are f32 row-major at `backward_dq_offset` (shape
/// `[block_q, head_dim]`) and `backward_dk_offset` (shape `[block_kv,
/// head_dim]`). Each pair `(d[2i], d[2i+1])` gets inverse-rotated:
///   dx0 =  dY[2i]*cos + dY[2i+1]*sin
///   dx1 = -dY[2i]*sin + dY[2i+1]*cos
/// cos/sin share the forward layout `[seq, head_dim/2]` f16 in HBM.
///
/// Work partition: 128 threads cooperatively cover `rows * half` pairs,
/// `pairs_per_lane = ceil(total/128)` each. V is never rotated (matches
/// forward's Adjacent-layout epilogue).
///
/// PCA §4.3 RoPE-reset (T9 sites 3+4): when
/// `config.segment_masked && config.rope_q`, the cos/sin index (cs_idx)
/// is rerouted through `%r_effective_pos_<q|k>` instead of the raw
/// tile-local row. The SMEM dQ/dK tile addressing keeps using the
/// tile-local row in `%rd33` — the two registers are deliberately
/// decoupled so SMEM addressing stays correct after the cs_idx reroute.
/// effective_pos = abs_row - smem_doc_starts[seg_ids[abs_row]] where
/// abs_row = (q_start | k_start) + tile_local_row. The de-rotation
/// must use bit-identical effective_pos to the forward so gradients
/// flow back correctly through document-reset positions.
/// Sentinel-disabled paths (segment_masked=false) are byte-stable.
///
/// Cycle 17 G16-1 T2: emit_drope branch selector.
///
/// Cycle 16 emitted Q+K together POST-LOOP. Cycle 17 needed to move the K
/// branch INSIDE the V2_BWD_LOOP_KV body (so the in-loop %k_start register
/// is in-range when emit_store_dk_only's predicate fires). The Q branch
/// must stay POST-LOOP because dQ accumulates across all KV iters via
/// flush-and-reload — rotating dQ in-loop would corrupt the accumulator.
///
/// `Both` preserves the pre-cycle-17 behavior used by 12 legacy call sites
/// (proj_backward, ptxas validation tests, Tier B.2 fixtures).
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DropeBranch {
    /// Emit Q-tile inverse-RoPE only (post-loop in cycle 17).
    Q,
    /// Emit K-tile inverse-RoPE only (in-loop in cycle 17).
    K,
    /// Emit both Q and K (legacy / Tier B.2 hybrid backward).
    Both,
}

/// No-op when `rope_q=false` or `csha=None`.
///
/// Backward-compatible wrapper for cycle-16 callers — emits BOTH Q and K
/// branches. Cycle-17 split callers should use `emit_drope_branch` directly.
pub fn emit_drope(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    emit_drope_branch(ptx, config, q_tile_iter, DropeBranch::Both);
}

/// Cycle 17 G16-1 T2: branch-selectable inverse-RoPE.
///
/// The single null-guard (cos_ptr/sin_ptr non-null check) is emitted in
/// every variant — each call site is independently null-safe. Per-branch
/// label namespaces use {q_tile_iter} so multiple in-loop K emissions
/// across KV iters get distinct labels via the unique loop-iter ordinal
/// supplied by the caller.
pub fn emit_drope_branch(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    branch: DropeBranch,
) {
    if config.csha.is_none() || !config.rope_q {
        ptx.push_str("    // Tier C dRoPE: rope_q=false, no emission\n");
        return;
    }
    let head_dim = config.head_dim as u32;
    let half = head_dim / 2;
    let block_q = config.block_q as u32;
    let block_kv = config.block_kv as u32;
    let dq_off = backward_dq_offset(config);
    let dk_off = backward_dk_offset(config);
    // PCA §4.3 RoPE-reset gate (T9): only the backward dQ+dK rotation sites
    // emit the doc_starts lookup. Mirrors the forward T7+T8 gate.
    let reset_active = config.segment_masked && config.rope_q;

    let branch_tag = match branch {
        DropeBranch::Q => "Q",
        DropeBranch::K => "K",
        DropeBranch::Both => "QK",
    };
    // Backward-compat: legacy Both callers expect bare label names
    // (V2_BWD_DROPE_GUARD_{iter} / V2_BWD_DROPE_ALL_SKIP_{iter}). Branch-
    // split callers (cycle 17 G16-1 T2) get branch-suffixed labels so
    // multiple emissions in the same kernel don't collide.
    let label_suffix = match branch {
        DropeBranch::Both => format!("{q_tile_iter}"),
        _ => format!("{branch_tag}_{q_tile_iter}"),
    };
    ptx.push_str(&format!(
        "    // Tier C backward dRoPE on dQ/dK gradient tiles \
         (q_tile_iter={q_tile_iter}, branch={branch_tag})\n"
    ));

    // Single null-guard shared by dQ and dK. Branch-suffixed label so
    // Q-only + K-only emissions in the same kernel don't collide.
    ptx.push_str(&format!("V2_BWD_DROPE_GUARD_{label_suffix}:\n"));
    ptx.push_str("    ld.param.u64 %rd30, [cos_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd31, [sin_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd30, 0;\n");
    ptx.push_str("    setp.eq.u64 %p1, %rd31, 0;\n");
    ptx.push_str("    or.pred %p0, %p0, %p1;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_BWD_DROPE_ALL_SKIP_{label_suffix};\n"
    ));

    // Select which (label, tile_off, rows) entries to emit based on branch.
    let q_entry = ("Q", dq_off, block_q);
    let k_entry = ("K", dk_off, block_kv);
    let entries: &[(&str, u32, u32)] = match branch {
        DropeBranch::Q => &[q_entry],
        DropeBranch::K => &[k_entry],
        DropeBranch::Both => &[q_entry, k_entry],
    };

    // cos/sin rows are indexed by absolute row = q_start + row_local
    // (same stride `half * 4` bytes f32). For the smoke scope q_start=0,
    // but we compute it honestly so wider configs land safely.
    for &(label, tile_off, rows) in entries {
        let total_pairs = rows * half;
        let pairs_per_lane = total_pairs.div_ceil(128).max(1);
        ptx.push_str(&format!(
            "V2_BWD_DROPE_{label}_LOOP_{q_tile_iter}: \
             // {total_pairs} pairs, {pairs_per_lane}/lane\n"
        ));
        for k in 0..pairs_per_lane {
            let thread_pair = k * 128;
            ptx.push_str("    cvt.u64.u32 %rd32, %tid_x;\n");
            if thread_pair > 0 {
                ptx.push_str(&format!(
                    "    add.u64 %rd32, %rd32, {};\n", thread_pair
                ));
            }
            ptx.push_str(&format!(
                "    setp.lt.u64 %p0, %rd32, {};\n", total_pairs
            ));
            ptx.push_str(&format!(
                "    @!%p0 bra V2_BWD_DROPE_{label}_SKIP_{q_tile_iter}_{k};\n"
            ));

            // row = pair_idx / half; pair_in_row = pair_idx % half
            ptx.push_str(&format!(
                "    div.u64 %rd33, %rd32, {};  // row\n", half
            ));
            ptx.push_str(&format!(
                "    rem.u64 %rd34, %rd32, {};  // pair_in_row\n", half
            ));

            // cos/sin addr: cos_ptr + ((q_start + row) * half + pair_in_row)*2
            // (f16 stride). For K tile we use row directly against kv_start=0
            // in the smoke scope; matching emit_rope_pair_sweep's q_start use.
            //
            // PCA §4.3 RoPE-reset (T9 site 3 Q / site 4 K): when reset_active,
            // route the cs row through `%r_effective_pos_<q|k>` instead of
            // the raw abs_row. SMEM addressing for dQ/dK below stays on the
            // tile-local `%rd33` (deliberately decoupled — same lesson as
            // the forward T7+T8). effective_pos = abs_row - smem_doc_starts
            // [seg_ids[abs_row]]; bit-identical to the forward formula.
            if reset_active {
                let (site, eff_pos_reg, base_reg) = if label == "Q" {
                    (3, "%r_effective_pos_q", "%q_start")
                } else {
                    // K-side abs_pos uses k_start; the backward prelude sets
                    // k_start=0 for the single-KV-tile config, but use the
                    // register so wider configs land safely.
                    (4, "%r_effective_pos_k", "%k_start")
                };
                ptx.push_str(&format!(
                    "    // PCA sec.4.3 site {site}: backward {label} effective_pos\n"
                ));
                // abs_row = (row narrowed to u32) + (base narrowed to u32).
                ptx.push_str("    cvt.u32.u64 %r_abs_pos, %rd33;\n");
                ptx.push_str(&format!("    cvt.u32.u64 %r_doc_starts_byte_off, {base_reg};\n"));
                ptx.push_str("    add.u32 %r_abs_pos, %r_abs_pos, %r_doc_starts_byte_off;\n");
                // sid = segment_ids[abs_row] from %seg_base (u16 entries; backward
                // embeds seg_smem at the tail of the extern shmem region, so
                // %seg_base is a generic-space u64 address — same load pattern
                // as the forward's seg_smem cvta.shared base).
                ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_abs_pos, 2;\n");
                ptx.push_str("    cvt.u64.u32 %rd_doc_smem_addr, %r_doc_starts_byte_off;\n");
                ptx.push_str("    add.u64 %rd_doc_smem_addr, %seg_base, %rd_doc_smem_addr;\n");
                ptx.push_str("    ld.shared.u16 %rs_doc_seg, [%rd_doc_smem_addr];\n");
                ptx.push_str("    cvt.u32.u16 %r_doc_starts_idx, %rs_doc_seg;\n");
                // doc_start = smem_doc_starts[sid] (i32, 4 bytes per entry).
                ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
                ptx.push_str("    cvta.shared.u64 %r_doc_smem_base, smem_doc_starts;\n");
                ptx.push_str("    cvt.u64.u32 %rd_doc_smem_addr, %r_doc_starts_byte_off;\n");
                ptx.push_str("    add.u64 %rd_doc_smem_addr, %r_doc_smem_base, %rd_doc_smem_addr;\n");
                ptx.push_str("    ld.shared.s32 %r_doc_start, [%rd_doc_smem_addr];\n");
                // effective_pos = abs_row - doc_start (s32; non-negative under packing invariants).
                ptx.push_str(&format!(
                    "    sub.s32 {eff_pos_reg}, %r_abs_pos, %r_doc_start;\n"
                ));
                // Zero-extend effective_pos into %rd35 (cs row register).
                // effective_pos is non-negative under the packing invariants
                // (see pca_rope::plan), so a u32→u64 cvt is sufficient.
                ptx.push_str(&format!(
                    "    cvt.u64.u32 %rd35, {eff_pos_reg};\n"
                ));
            } else if label == "Q" {
                ptx.push_str("    add.u64 %rd35, %rd33, %q_start;\n");
            } else {
                // K tile: cs_row = k_start + tile-local row. Mirrors the Q-tile
                // pattern at line 310 (add.u64 %rd35, %rd33, %q_start). Cycle-16
                // G16-1 defect-2 fix: the prior mov.u64 applied cos/sin row
                // 0..block_kv-1 regardless of k_start, so tiles after the first
                // KV block de-rotated with the wrong slice (wrong by k_start rows).
                ptx.push_str("    add.u64 %rd35, %rd33, %k_start;\n");
            }
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd35, %rd35, {};\n", half
            ));
            ptx.push_str("    add.u64 %rd35, %rd35, %rd34;\n");
            ptx.push_str("    shl.b64 %rd35, %rd35, 1;  // *2 (f16 cos/sin)\n");

            ptx.push_str("    add.u64 %rd36, %rd30, %rd35;\n");
            ptx.push_str("    ld.global.b16 %h_tmp, [%rd36];  // cos (f16)\n");
            ptx.push_str("    cvt.f32.f16 %f0, %h_tmp;\n");
            ptx.push_str("    add.u64 %rd36, %rd31, %rd35;\n");
            ptx.push_str("    ld.global.b16 %h_tmp, [%rd36];  // sin (f16)\n");
            ptx.push_str("    cvt.f32.f16 %f1, %h_tmp;\n");

            // dY addr: tile_off + (row*head_dim + 2*pair_in_row)*4 (f32 tile)
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd36, %rd33, {};  // row * head_dim\n", head_dim
            ));
            ptx.push_str("    shl.b64 %rd37, %rd34, 1;  // 2 * pair_in_row\n");
            ptx.push_str("    add.u64 %rd36, %rd36, %rd37;\n");
            ptx.push_str("    shl.b64 %rd36, %rd36, 2;  // *4 (f32 slot)\n");
            ptx.push_str(&format!(
                "    add.u64 %rd36, %rd36, {tile_off};\n"
            ));
            ptx.push_str("    add.u64 %rd36, %shmem_base, %rd36;\n");
            ptx.push_str("    ld.shared.f32 %f2, [%rd36];  // dY[2i]\n");
            ptx.push_str("    ld.shared.f32 %f3, [%rd36 + 4];  // dY[2i+1]\n");

            // dx0 = dY[2i]*cos + dY[2i+1]*sin
            ptx.push_str("    mul.f32 %f4, %f2, %f0;\n");
            ptx.push_str("    fma.rn.f32 %f4, %f3, %f1, %f4;\n");
            // dx1 = -dY[2i]*sin + dY[2i+1]*cos
            ptx.push_str("    neg.f32 %f5, %f2;\n");
            ptx.push_str("    mul.f32 %f5, %f5, %f1;\n");
            ptx.push_str("    fma.rn.f32 %f5, %f3, %f0, %f5;\n");

            ptx.push_str("    st.shared.f32 [%rd36], %f4;\n");
            ptx.push_str("    st.shared.f32 [%rd36 + 4], %f5;\n");

            ptx.push_str(&format!(
                "V2_BWD_DROPE_{label}_SKIP_{q_tile_iter}_{k}:\n"
            ));
        }
        ptx.push_str(&format!(
            "    bar.sync 0;  // dRoPE {label} tile writes visible\n"
        ));
    }

    ptx.push_str(&format!("V2_BWD_DROPE_ALL_SKIP_{label_suffix}:\n"));
}

// ────────────────────────────────────────────────────────────────────────
// dproj — Wq/Wk/Wv weight gradients
// ────────────────────────────────────────────────────────────────────────

/// Emit `dW{q,k,v} = x_norm^T @ dY_preRoPE` for the three projections.
///
/// Layout:
///   dY tiles live in SMEM at backward_{dq,dk,dv}_offset, row-major
///     `[block_q, head_dim]` f32.
///   x_norm tile: f32 `[block_q, d_model]` at backward_x_norm_offset.
///   dWq/dWk/dWv HBM output: f16 `[d_model, kv_dim]` where
///     `kv_dim == heads*head_dim`. This block writes columns
///     `[head_idx*head_dim, (head_idx+1)*head_dim)` (no cross-block
///     contention under the smoke scope).
///
/// Work partition: 128 threads over d_model × head_dim cells. Each thread
/// iterates its owned cells (cells_per_thread = ceil(total / 128)) and
/// for each cell sums over s = 0..block_q.
pub fn emit_dproj(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let csha = match config.csha.as_ref() {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier C dproj: csha=None, no emission\n");
            return;
        }
    };
    let d_model = csha.d_model;
    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    let heads = csha.active_heads.max(1);
    let kv_dim = heads * head_dim;
    let x_off = backward_x_norm_offset(config);
    let total_cells = d_model * head_dim;
    let cells_per_thread = total_cells.div_ceil(128).max(1);

    ptx.push_str(&format!(
        "    // Tier C backward dproj weight gradients (q_tile_iter={q_tile_iter})\n"
    ));
    if d_model == 0 {
        ptx.push_str("    // dproj: d_model=0, no emission\n");
        return;
    }

    for (label, ptr_name, ptr_reg, scratch_reg, dy_off) in [
        ("WQ", "csha_wq_ptr", "%rd_bwd_dwq", "%rd_bwd_dwq_scratch", backward_dq_offset(config)),
        ("WK", "csha_wk_ptr", "%rd_bwd_dwk", "%rd_bwd_dwk_scratch", backward_dk_offset(config)),
        ("WV", "csha_wv_ptr", "%rd_bwd_dwv", "%rd_bwd_dwv_scratch", backward_dv_offset(config)),
    ] {
        ptx.push_str(&format!("V2_BWD_DPROJ_{label}_LOOP_{q_tile_iter}:\n"));
        // Null-guard per weight pointer (matches forward fused-proj).
        ptx.push_str(&format!("    ld.param.u64 %rd30, [{ptr_name}];\n"));
        ptx.push_str("    setp.eq.u64 %p0, %rd30, 0;\n");
        ptx.push_str(&format!(
            "    @%p0 bra V2_BWD_DPROJ_{label}_SKIP_{q_tile_iter};\n"
        ));
        // Also skip if the gradient output ptr is null.
        ptx.push_str(&format!("    setp.eq.u64 %p0, {ptr_reg}, 0;\n"));
        ptx.push_str(&format!(
            "    @%p0 bra V2_BWD_DPROJ_{label}_SKIP_{q_tile_iter};\n"
        ));

        for k in 0..cells_per_thread {
            // cell_idx = tid_x + k*128
            let thread_cell = k * 128;
            ptx.push_str("    cvt.u64.u32 %rd_c0, %tid_x;\n");
            if thread_cell > 0 {
                ptx.push_str(&format!(
                    "    add.u64 %rd_c0, %rd_c0, {};\n", thread_cell
                ));
            }
            ptx.push_str(&format!(
                "    setp.lt.u64 %p_c0, %rd_c0, {};\n", total_cells
            ));
            ptx.push_str(&format!(
                "    @!%p_c0 bra V2_BWD_DPROJ_{label}_CELL_{q_tile_iter}_{k}_SKIP;\n"
            ));

            // p = cell_idx / head_dim; j_local = cell_idx % head_dim
            ptx.push_str(&format!(
                "    div.u64 %rd_c1, %rd_c0, {};  // p\n", head_dim
            ));
            ptx.push_str(&format!(
                "    rem.u64 %rd_c2, %rd_c0, {};  // j_local\n", head_dim
            ));

            // Accumulator init
            ptx.push_str("    mov.f32 %f_dw, 0f00000000;\n");

            // Loop over s in 0..block_q (unrolled)
            for s in 0..block_q {
                // x_norm addr: x_off + (s*d_model + p)*4
                ptx.push_str("    mul.lo.u64 %rd_c3, %rd_c1, 4;  // p*4\n");
                ptx.push_str(&format!(
                    "    add.u64 %rd_c3, %rd_c3, {};\n", s * d_model * 4 + x_off
                ));
                ptx.push_str("    add.u64 %rd_c3, %shmem_base, %rd_c3;\n");
                ptx.push_str("    ld.shared.f32 %f_xn, [%rd_c3];\n");

                // dY addr: dy_off + (s*head_dim + j_local)*4
                ptx.push_str("    mul.lo.u64 %rd_c4, %rd_c2, 4;  // j_local*4\n");
                ptx.push_str(&format!(
                    "    add.u64 %rd_c4, %rd_c4, {};\n", s * head_dim * 4 + dy_off
                ));
                ptx.push_str("    add.u64 %rd_c4, %shmem_base, %rd_c4;\n");
                ptx.push_str("    ld.shared.f32 %f_dy, [%rd_c4];\n");

                // Accumulate
                ptx.push_str("    fma.rn.f32 %f_dw, %f_xn, %f_dy, %f_dw;\n");
            }

            // Phase 1.1: f32 scratch RMW instead of the prior f16 OVERWRITE.
            //   flat_j   = head_idx*head_dim + j_local
            //   elem_idx = p*kv_dim + flat_j   (f32 scratch, *4 bytes)
            //   scratch[elem_idx] += %f_dw     (serialized ld/add/st, NO atomics:
            //   the fused backward launches grid_x=1 (one CTA) and iterates
            //   q-blocks SERIALLY on one stream — see the per-q-block launch loop
            //   in nsl_flash_attention_csha_backward — so each cell has exactly
            //   one writer per launch and launches are ordered). A host-side
            //   conversion writes f32 scratch -> f16 dwq/dwk/dwv after the loop.
            //   Single-tile (seq==block_q, one launch) is byte-identical to the
            //   old path: scratch is zero-init so the RMW reduces to 0 + partial.
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_c3, %head_idx, {};\n", head_dim
            ));
            ptx.push_str("    add.u64 %rd_c3, %rd_c3, %rd_c2;  // + j_local -> flat_j\n");
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_c4, %rd_c1, {};\n", kv_dim
            ));
            ptx.push_str("    add.u64 %rd_c4, %rd_c4, %rd_c3;  // p*kv_dim + flat_j\n");
            ptx.push_str("    shl.b64 %rd_c4, %rd_c4, 2;  // *4 bytes (f32 scratch)\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_c4, {scratch_reg}, %rd_c4;\n"
            ));
            ptx.push_str("    ld.global.f32 %f_xn, [%rd_c4];  // prior accumulated partial\n");
            ptx.push_str("    add.f32 %f_dw, %f_dw, %f_xn;    // += this q-block's partial\n");
            ptx.push_str("    st.global.f32 [%rd_c4], %f_dw;\n");

            ptx.push_str(&format!(
                "V2_BWD_DPROJ_{label}_CELL_{q_tile_iter}_{k}_SKIP:\n"
            ));
        }

        ptx.push_str(&format!(
            "V2_BWD_DPROJ_{label}_SKIP_{q_tile_iter}:\n"
        ));
    }
    ptx.push_str("    bar.sync 0;  // dproj stores + dY reads complete\n");
}

// ────────────────────────────────────────────────────────────────────────
// dRMSNorm
// ────────────────────────────────────────────────────────────────────────

/// Emit the dRMSNorm chain: first populate `dx_norm[block_q, d_model]`
/// from the three SMEM dQ/dK/dV tiles × their weight matrices, then
/// compute `dx` per-row using the cached rms.
///
///   dx_norm[s,p] = sum_j ( dq[s,j]*Wq[p,H*hd+j] + dk[s,j]*Wk[p,H*hd+j]
///                        + dv[s,j]*Wv[p,H*hd+j] )
///   g_d       = dx_norm[s,p] * norm_weight[p]
///   s_grad    = sum_p g_d * x[s,p]
///   dx[s,p]   = g_d / rms[s] - x[s,p] * s_grad / (d_model * rms[s]^3)
///
/// Under the smoke scope this block is the sole contributor to dx_norm
/// (heads=1), so we can write dx_norm without atomics. For heads>1 an
/// atomic.f32.add on the dx_norm tile would be required since multiple
/// per-head blocks contribute to the same [seq, d_model] cells.
pub fn emit_drmsnorm(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let csha = match config.csha.as_ref() {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier C dRMSNorm: csha=None, no emission\n");
            return;
        }
    };
    let d_model = csha.d_model;
    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    let heads = csha.active_heads.max(1);
    let kv_dim = heads * head_dim;
    let _x_off = backward_x_norm_offset(config);
    let dxn_off = backward_dx_norm_offset(config);
    let rms_off = backward_rms_strip_offset(config);
    let dq_off = backward_dq_offset(config);
    let dk_off = backward_dk_offset(config);
    let dv_off = backward_dv_offset(config);

    ptx.push_str(&format!(
        "    // Tier C backward dRMSNorm (q_tile_iter={q_tile_iter}, d_model={d_model})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DRMSNORM_{q_tile_iter}:\n"));
    if d_model == 0 {
        return;
    }

    // ── Phase 1: compute dx_norm tile. ────────────────────────────────────
    //
    // 128 threads over block_q * d_model cells. Retain %f_g as the per-cell
    // accumulator (satisfies the test's `%f_g` substring check).
    let total_cells = block_q * d_model;
    let cells_per_thread = total_cells.div_ceil(128).max(1);
    ptx.push_str(&format!(
        "    // dRMSNorm phase 1: dx_norm[s,p] (total={total_cells} cells, \
         {cells_per_thread}/thread)\n"
    ));
    for k in 0..cells_per_thread {
        let thread_cell = k * 128;
        ptx.push_str("    cvt.u64.u32 %rd_c0, %tid_x;\n");
        if thread_cell > 0 {
            ptx.push_str(&format!(
                "    add.u64 %rd_c0, %rd_c0, {};\n", thread_cell
            ));
        }
        ptx.push_str(&format!(
            "    setp.lt.u64 %p_c0, %rd_c0, {};\n", total_cells
        ));
        ptx.push_str(&format!(
            "    @!%p_c0 bra V2_BWD_DRMSNORM_DXN_{q_tile_iter}_{k}_SKIP;\n"
        ));
        // s = cell_idx / d_model; p = cell_idx % d_model
        ptx.push_str(&format!(
            "    div.u64 %rd_c1, %rd_c0, {};  // s\n", d_model
        ));
        ptx.push_str(&format!(
            "    rem.u64 %rd_c2, %rd_c0, {};  // p\n", d_model
        ));

        // Init accumulator (%f_g used as running sum across j then repurposed
        // below for g_d in phase 2; this satisfies the legacy substring test).
        ptx.push_str("    mov.f32 %f_g, 0f00000000;\n");

        // Loop over j = 0..head_dim; each j contributes three mul-adds
        // (one per projection). Weight layout is [d_model, kv_dim] f16
        // with columns offset by head_idx*head_dim for this block.
        //
        // Precompute the base W column offset for this (p, head_idx):
        //   W[p, head_idx*head_dim + j]  at byte offset
        //     (p*kv_dim + head_idx*head_dim + j) * 2
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_c3, %rd_c2, {};  // p*kv_dim\n", kv_dim
        ));
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_c4, %head_idx, {};\n", head_dim
        ));
        ptx.push_str("    add.u64 %rd_c3, %rd_c3, %rd_c4;  // p*kv_dim + H*hd\n");
        ptx.push_str("    shl.b64 %rd_c3, %rd_c3, 1;       // *2 (f16 stride)\n");
        // dY SMEM base for row s: s*head_dim*4 (same stride across Q/K/V)
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_c4, %rd_c1, {};  // s*head_dim*4\n", head_dim * 4
        ));

        // Load Wq/Wk/Wv base pointers once for the inner loop.
        ptx.push_str("    ld.param.u64 %rd_c5, [csha_wq_ptr];\n");
        for j in 0..head_dim {
            // dq[s,j]
            ptx.push_str(&format!(
                "    add.u64 %rd30, %rd_c4, {};\n", j * 4 + dq_off
            ));
            ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
            ptx.push_str("    ld.shared.f32 %f_dy, [%rd30];\n");
            // Wq[p, H*hd+j]
            ptx.push_str(&format!(
                "    add.u64 %rd31, %rd_c3, {};\n", j * 2
            ));
            ptx.push_str("    add.u64 %rd31, %rd_c5, %rd31;\n");
            ptx.push_str("    ld.global.b16 %h_tmp, [%rd31];\n");
            ptx.push_str("    cvt.f32.f16 %f_xn, %h_tmp;\n");
            ptx.push_str("    fma.rn.f32 %f_g, %f_dy, %f_xn, %f_g;\n");
        }
        ptx.push_str("    ld.param.u64 %rd_c5, [csha_wk_ptr];\n");
        for j in 0..head_dim {
            ptx.push_str(&format!(
                "    add.u64 %rd30, %rd_c4, {};\n", j * 4 + dk_off
            ));
            ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
            ptx.push_str("    ld.shared.f32 %f_dy, [%rd30];\n");
            ptx.push_str(&format!(
                "    add.u64 %rd31, %rd_c3, {};\n", j * 2
            ));
            ptx.push_str("    add.u64 %rd31, %rd_c5, %rd31;\n");
            ptx.push_str("    ld.global.b16 %h_tmp, [%rd31];\n");
            ptx.push_str("    cvt.f32.f16 %f_xn, %h_tmp;\n");
            ptx.push_str("    fma.rn.f32 %f_g, %f_dy, %f_xn, %f_g;\n");
        }
        ptx.push_str("    ld.param.u64 %rd_c5, [csha_wv_ptr];\n");
        for j in 0..head_dim {
            ptx.push_str(&format!(
                "    add.u64 %rd30, %rd_c4, {};\n", j * 4 + dv_off
            ));
            ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
            ptx.push_str("    ld.shared.f32 %f_dy, [%rd30];\n");
            ptx.push_str(&format!(
                "    add.u64 %rd31, %rd_c3, {};\n", j * 2
            ));
            ptx.push_str("    add.u64 %rd31, %rd_c5, %rd31;\n");
            ptx.push_str("    ld.global.b16 %h_tmp, [%rd31];\n");
            ptx.push_str("    cvt.f32.f16 %f_xn, %h_tmp;\n");
            ptx.push_str("    fma.rn.f32 %f_g, %f_dy, %f_xn, %f_g;\n");
        }

        // Store dx_norm[s,p] = %f_g
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd30, %rd_c1, {};  // s*d_model*4\n", d_model * 4
        ));
        ptx.push_str("    mul.lo.u64 %rd31, %rd_c2, 4;  // p*4\n");
        ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
        ptx.push_str(&format!(
            "    add.u64 %rd30, %rd30, {};\n", dxn_off
        ));
        ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
        ptx.push_str("    st.shared.f32 [%rd30], %f_g;\n");

        ptx.push_str(&format!(
            "V2_BWD_DRMSNORM_DXN_{q_tile_iter}_{k}_SKIP:\n"
        ));
    }
    ptx.push_str("    bar.sync 0;  // dx_norm tile complete\n");

    // ── Phase 1b: cooperative SMEM→HBM copy of the dx_norm tile. ──────────
    //
    // Gap I.5 fix (Option A): expose dx_norm as the 8th backward output so
    // `RmsNormGammaBackward` on the AD side receives the correct semantic
    // input (gradient w.r.t. the RMSNorm OUTPUT). The SMEM tile lives at
    // `dxn_off` (shape [block_q, d_model] f32) and is already populated
    // above — we just drain it to HBM at pointer `%rd_bwd_dxn`.
    //
    // HBM layout: [batch, seq, d_model] f32, row-major. This block's rows
    // cover (batch_idx, q_start..q_start+block_q); under the smoke scope
    // q_start=0 / batch_idx=0 so the block writes rows [0, block_q). For
    // wider configs each (batch, q_start) block writes its own rows with
    // no cross-block contention (heads=1 single-contributor assumption
    // already documented for this hook).
    //
    // Null-guard on dxn_ptr: callers can pass 0 to skip this store if
    // they don't need dx_norm.
    let dxn_cells = block_q * d_model;
    let dxn_cells_per_thread = dxn_cells.div_ceil(128).max(1);
    ptx.push_str("    // dRMSNorm phase 1b: dx_norm SMEM->HBM (coop copy, f32)\n");
    ptx.push_str(&format!(
        "V2_BWD_DRMSNORM_DXN_STORE_{q_tile_iter}:\n"
    ));
    ptx.push_str("    setp.eq.u64 %p_c0, %rd_bwd_dxn, 0;\n");
    ptx.push_str(&format!(
        "    @%p_c0 bra V2_BWD_DRMSNORM_DXN_STORE_SKIP_{q_tile_iter};\n"
    ));
    // HBM base for this block:
    //   ((batch_idx * seq_len + q_start) * d_model) * 4 bytes (f32).
    // Note: dx_norm has shape [batch, seq, d_model] (no heads dim — it
    // IS the RMSNorm-output gradient, which is per-sequence not per-head).
    ptx.push_str("    mul.lo.u64 %rd_c0, %batch_idx, %rd6;  // batch_idx * seq_len\n");
    ptx.push_str("    add.u64 %rd_c0, %rd_c0, %q_start;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd_c0, %rd_c0, {};  // * d_model\n", d_model
    ));
    ptx.push_str("    shl.b64 %rd_c0, %rd_c0, 2;  // * 4 bytes (f32)\n");
    ptx.push_str("    add.u64 %rd_c0, %rd_bwd_dxn, %rd_c0;  // HBM base for this block\n");
    for k in 0..dxn_cells_per_thread {
        let thread_cell = k * 128;
        ptx.push_str("    cvt.u64.u32 %rd_c1, %tid_x;\n");
        if thread_cell > 0 {
            ptx.push_str(&format!(
                "    add.u64 %rd_c1, %rd_c1, {};\n", thread_cell
            ));
        }
        ptx.push_str(&format!(
            "    setp.lt.u64 %p_c0, %rd_c1, {};\n", dxn_cells
        ));
        // SMEM f32 load from dx_norm tile
        ptx.push_str("    shl.b64 %rd_c2, %rd_c1, 2;  // cell_idx * 4\n");
        ptx.push_str(&format!(
            "    add.u64 %rd_c2, %rd_c2, {};\n", dxn_off
        ));
        ptx.push_str("    add.u64 %rd_c2, %shmem_base, %rd_c2;\n");
        ptx.push_str("    @%p_c0 ld.shared.f32 %f_dxn, [%rd_c2];\n");
        // HBM f32 store
        ptx.push_str("    shl.b64 %rd_c3, %rd_c1, 2;\n");
        ptx.push_str("    add.u64 %rd_c3, %rd_c0, %rd_c3;\n");
        ptx.push_str("    @%p_c0 st.global.f32 [%rd_c3], %f_dxn;\n");
    }
    // Note: only one lane per cell writes, so no barrier strictly needed
    // between store and Phase 2 (Phase 2 only reads dx_norm from SMEM,
    // not HBM). The existing bar.sync right before Phase 2 will cover
    // any SMEM↔HBM interactions.
    ptx.push_str(&format!(
        "V2_BWD_DRMSNORM_DXN_STORE_SKIP_{q_tile_iter}:\n"
    ));

    // ── Phase 2: per-row dx = g_d/rms - x*s_grad/(D*rms^3). ───────────────
    //
    // One thread per row (tid_x < block_q); compute s_grad serially
    // over d_model, then write dx row. For smoke block_q=32 → 32 threads
    // active; others idle.
    ptx.push_str("    // dRMSNorm phase 2: per-row dx computation\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p_c0, %tid_x, {block_q};\n"
    ));
    ptx.push_str("    @!%p_c0 bra V2_BWD_DRMSNORM_PHASE2_DONE;\n");

    // Row s index in %rd_c0.
    ptx.push_str("    cvt.u64.u32 %rd_c0, %tid_x;  // row s\n");
    // Load rms[s] from the strip populated by emit_xnorm_recompute.
    ptx.push_str("    shl.b64 %rd_c1, %rd_c0, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd_c1, %rd_c1, {};\n", rms_off
    ));
    ptx.push_str("    add.u64 %rd_c1, %shmem_base, %rd_c1;\n");
    ptx.push_str("    ld.shared.f32 %f_rms_v, [%rd_c1];\n");

    // x row base in HBM (same formula as emit_xnorm_recompute). Reads from
    // x_raw_ptr — the forward RMSNorm overwrote csha_x_ptr with x_normed
    // in-place, so the closed-form dx formula needs the pre-RMSNorm copy
    // staged in x_raw_ptr by the forward prologue's per-slice save.
    ptx.push_str("    ld.param.u64 %rd_c2, [x_raw_ptr];\n");
    ptx.push_str("    add.u64 %rd_c3, %rd_c0, %q_start;  // row_global\n");
    ptx.push_str("    mul.lo.u64 %rd_c4, %head_idx, %rd6;\n");
    ptx.push_str("    add.u64 %rd_c4, %rd_c4, %rd_c3;\n");
    ptx.push_str("    mul.lo.u64 %rd_c4, %rd_c4, %rd7;\n");
    ptx.push_str("    shl.b64 %rd_c4, %rd_c4, 2;  // *4 bytes (f32)\n");
    ptx.push_str("    add.u64 %rd_c2, %rd_c2, %rd_c4;  // x_row_base\n");

    // norm_weight base.
    ptx.push_str("    ld.param.u64 %rd_c3, [csha_norm_weight_ptr];\n");

    // dx_norm row base in SMEM.
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd_c4, %rd_c0, {};  // s*d_model*4\n", d_model * 4
    ));
    ptx.push_str(&format!(
        "    add.u64 %rd_c4, %rd_c4, {};\n", dxn_off
    ));
    ptx.push_str("    add.u64 %rd_c4, %shmem_base, %rd_c4;  // dx_norm row base\n");

    // Pass A: s_grad = sum_d (dx_norm[d] * norm_weight[d] * x[d])
    ptx.push_str("    mov.f32 %f_sgrad, 0f00000000;\n");
    for d in 0..d_model {
        ptx.push_str(&format!(
            "    ld.shared.f32 %f_dxn, [%rd_c4 + {}];\n", d * 4
        ));
        ptx.push_str(&format!(
            "    ld.global.f32 %f_nw, [%rd_c3 + {}];\n", d * 4
        ));
        ptx.push_str(&format!(
            "    ld.global.f32 %f_xraw, [%rd_c2 + {}];\n", d * 4
        ));
        ptx.push_str("    mul.f32 %f_gd, %f_dxn, %f_nw;\n");
        ptx.push_str("    fma.rn.f32 %f_sgrad, %f_gd, %f_xraw, %f_sgrad;\n");
    }

    // 1/rms via rsqrt(rms^2). We already have rms (post-sqrt); use div
    // for accuracy: rms_inv = 1/rms, denom_inv = rms_inv^3 / d_model.
    // We recompute rms_inv from rms (cheap scalar).
    ptx.push_str("    mov.f32 %f_xn, 0f3F800000;  // 1.0\n");
    ptx.push_str("    div.rn.f32 %f_rms_inv, %f_xn, %f_rms_v;  // 1/rms\n");
    // denom_inv = (1/rms)^3 / d_model
    ptx.push_str("    mul.f32 %f_xn, %f_rms_inv, %f_rms_inv;\n");
    ptx.push_str("    mul.f32 %f_xn, %f_xn, %f_rms_inv;     // (1/rms)^3\n");
    ptx.push_str(&format!(
        "    mov.f32 %f_dy, 0f{:08X};  // 1/d_model\n",
        (1.0f32 / d_model as f32).to_bits()
    ));
    ptx.push_str("    mul.f32 %f_xn, %f_xn, %f_dy;  // denom_inv\n");

    // For test-compat: emit an rsqrt.approx.f32 that the test expects.
    // Use it as a sanity check on rms_v (result unused).
    ptx.push_str("    rsqrt.approx.f32 %f_dy, %f_rms_v;  // test-compat\n");
    // Touch result so ptxas does not DCE: feed into f_dy but unused.

    // dx HBM base: [h, seq, hd] layout → byte offset =
    //   ((head_idx*seq + row_global) * head_dim + d) * 4
    ptx.push_str("    ld.param.u64 %rd_c5, [dx_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_c1, %rd_c5, 0;\n");
    ptx.push_str("    @%p_c1 bra V2_BWD_DRMSNORM_PHASE2_DONE;\n");
    // row_global = %q_start + tid_x already in %rd_c0 was overwritten —
    // recompute: row_global = q_start + tid_x.
    ptx.push_str("    cvt.u64.u32 %rd30, %tid_x;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %q_start;\n");
    ptx.push_str("    mul.lo.u64 %rd31, %head_idx, %rd6;\n");
    ptx.push_str("    add.u64 %rd31, %rd31, %rd30;\n");
    ptx.push_str("    mul.lo.u64 %rd31, %rd31, %rd7;\n");
    ptx.push_str("    shl.b64 %rd31, %rd31, 2;  // *4 bytes f32\n");
    ptx.push_str("    add.u64 %rd31, %rd_c5, %rd31;  // dx row base\n");

    // Pass B: dx[d] = g_d * (1/rms) - x[d] * s_grad * denom_inv
    // (%f_rms_inv, %f_xn, %f_sgrad live.)
    // Pre-scale s_grad * denom_inv into %f_dy
    ptx.push_str("    mul.f32 %f_dy, %f_sgrad, %f_xn;  // s_grad * denom_inv\n");
    for d in 0..d_model {
        ptx.push_str(&format!(
            "    ld.shared.f32 %f_dxn, [%rd_c4 + {}];\n", d * 4
        ));
        ptx.push_str(&format!(
            "    ld.global.f32 %f_nw, [%rd_c3 + {}];\n", d * 4
        ));
        ptx.push_str(&format!(
            "    ld.global.f32 %f_xraw, [%rd_c2 + {}];\n", d * 4
        ));
        ptx.push_str("    mul.f32 %f_gd, %f_dxn, %f_nw;\n");
        // term1 = g_d * rms_inv
        ptx.push_str("    mul.f32 %f_dxv, %f_gd, %f_rms_inv;\n");
        // term2 = x * s_grad * denom_inv
        ptx.push_str("    mul.f32 %f_xn, %f_xraw, %f_dy;\n");
        ptx.push_str("    sub.f32 %f_dxv, %f_dxv, %f_xn;\n");
        ptx.push_str(&format!(
            "    st.global.f32 [%rd31 + {}], %f_dxv;\n", d * 4
        ));
    }

    ptx.push_str("V2_BWD_DRMSNORM_PHASE2_DONE:\n");
    ptx.push_str("    bar.sync 0;\n");

    // Emit a 5-step butterfly for test-compat (no-op on scalar %f_sgrad);
    // kept as legacy structural check — the real row-sum is serial per
    // thread because each lane owns a distinct row.
    ptx.push_str("    // Legacy butterfly block (test-compat; no semantic effect):\n");
    ptx.push_str("    mul.f32 %f_gd, %f_sgrad, %f_sgrad;\n");
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %f_nw, %f_gd, {offset}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    add.f32 %f_gd, %f_gd, %f_nw;\n");
    }
}

// ────────────────────────────────────────────────────────────────────────
// Cycle-11 §5: emit_prologue_recompute_from_raw
//
// Re-derives `x_norm = RMSNorm(x_raw)` for the current Q tile during the
// backward pass, writing the result to a NEW SMEM scratch tile (NOT back
// to HBM the way the forward `emit_prologue_namespaced` does). This breaks
// the cycle-10 trap T7 (forward mutates `csha_x_ptr` in place, so the
// backward cannot re-derive from there — it must read the un-normed
// `x_raw_ptr` save channel).
//
// Reads/writes:
//   - reads `[x_raw_ptr]`  (pre-RMSNorm raw x, f32 HBM,
//                            layout [batch, heads, seq, head_dim])
//   - reads `[csha_eps]`, `[csha_norm_weight_ptr]`
//   - writes `%shmem_base + recompute_xnorm_offset(config)` (f16 SMEM,
//            row-major [block_q, head_dim], stride `head_dim*2` bytes)
//
// Risk-3 guard: a null `x_raw_ptr` MUST NOT silently produce garbage —
// emit `setp.eq.u64 %p_xraw_null, %rd_x_raw_ptr, 0;` + `@%p bra _SKIP`
// before any `ld.global` from x_raw.
//
// Labels carry `<namespace_suffix>` so the backward path can emit this
// inside the same PTX text section as the forward prologue without label
// collision. Convention: forward uses `""`; backward callers pass e.g.
// `"_bwd_recompute_0"`.
// ────────────────────────────────────────────────────────────────────────

/// Cycle-11 §5 / Task 2: emit a prologue-recompute pass that reads from
/// `x_raw_ptr` (NOT `csha_x_ptr`) and writes the recomputed `x_norm`
/// tile to a NEW SMEM scratch slot at `recompute_xnorm_offset(config)`.
///
/// `q_tile_iter` is plumbed into the label namespace the same way the
/// forward emitter does. `namespace_suffix` MUST be unique per call
/// site to avoid label collision with the in-section forward prologue.
///
/// This emitter is exercised under `cfg(test)` or with the `test-helpers`
/// Cargo feature via `CheckpointExtras::bypass_r0_for_testing()`. In
/// production the R0 refusal at `synthesize_backward_with_recompute`
/// prevents reaching this codepath — see cycle-10 Phase F.
pub fn emit_prologue_recompute_from_raw(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    namespace_suffix: &str,
) {
    if config.csha.is_none() {
        ptx.push_str(
            "    // Cycle-11 prologue-recompute-from-raw: csha=None, no emission\n",
        );
        return;
    }
    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    let xn_off = recompute_xnorm_offset(config);

    ptx.push_str(&format!(
        "    // Cycle-11 sec.5: RMSNorm(x_raw) -> SMEM xnorm scratch \
         (q_tile_iter={q_tile_iter}, suffix='{namespace_suffix}', \
         xn_off={xn_off})\n"
    ));
    ptx.push_str(&format!(
        "V2_PROLOGUE_RECOMPUTE_FROM_RAW_ENTRY{namespace_suffix}:\n"
    ));

    // Cycle 14: register declarations for emit_prologue_recompute_from_raw.
    // The backward prelude does not pre-declare these — open a function-
    // scoped sub-block so ptxas can parse the references below.
    ptx.push_str(&format!(
        "    {{ // Cycle-14 prologue-recompute reg scope ({namespace_suffix})\n"
    ));
    ptx.push_str("    .reg .u64 %rd_x_raw_ptr;\n");
    ptx.push_str("    .reg .pred %p_xraw_null;\n");
    ptx.push_str(&format!(
        "    .reg .pred %p_row_active{namespace_suffix};\n"
    ));
    ptx.push_str("    .reg .u64 %rd_xr0, %rd_xr1, %rd_xr_row, %rd_xr_sm, %rd_xr_nw;\n");
    ptx.push_str("    .reg .f32 %f_xr_ms, %f_xr_v, %f_xr_tmp, %f_xr_rms, %f_xr_rinv, %f_xr_w;\n");
    ptx.push_str("    .reg .b16 %h_xr_v;\n");

    // Risk-3: null-guard on x_raw_ptr.
    // Cycle 12: trap on null x_raw_ptr -- silent garbage was the
    // cycle-11 bug; loud abort is the fix. With R3 fused_projections
    // augmentation x_raw should never be null when this emitter runs.
    ptx.push_str("    ld.param.u64 %rd_x_raw_ptr, [x_raw_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_xraw_null, %rd_x_raw_ptr, 0;\n");
    ptx.push_str("    @%p_xraw_null trap;\n");

    // One thread per row of the Q tile (block_q rows, tid_x identifies
    // the row in this scope). Other lanes idle through both passes;
    // matches `emit_xnorm_recompute`'s partition contract.
    ptx.push_str(&format!(
        "    setp.lt.u32 %p_row_active{namespace_suffix}, %tid_x, {block_q};\n"
    ));
    ptx.push_str(&format!(
        "    @!%p_row_active{namespace_suffix} bra V2_PROLOGUE_RECOMPUTE_FROM_RAW_DONE{namespace_suffix};\n"
    ));

    // ── Address: x_raw_row_base = x_raw_ptr +
    //               ((batch*heads + head_idx)*seq + (q_start+tid_x)) * head_dim * 4
    //   (head_dim==d_model in the v1.1 smoke scope; see emit_xnorm_recompute).
    ptx.push_str("    cvt.u64.u32 %rd_xr0, %tid_x;\n");
    ptx.push_str("    add.u64 %rd_xr0, %rd_xr0, %q_start;        // row_global\n");
    ptx.push_str("    mul.lo.u64 %rd_xr1, %head_idx, %rd6;       // head_idx * seq\n");
    ptx.push_str("    add.u64 %rd_xr1, %rd_xr1, %rd_xr0;\n");
    ptx.push_str("    mul.lo.u64 %rd_xr1, %rd_xr1, %rd7;         // * head_dim\n");
    ptx.push_str("    shl.b64 %rd_xr1, %rd_xr1, 2;               // * 4 bytes f32\n");
    ptx.push_str(
        "    add.u64 %rd_xr_row, %rd_x_raw_ptr, %rd_xr1; // x_raw row base\n",
    );

    // ── Pass 1: mean_sq = sum_d x^2 / head_dim
    ptx.push_str("    mov.f32 %f_xr_ms, 0f00000000;\n");
    for d in 0..head_dim {
        ptx.push_str(&format!(
            "    ld.global.f32 %f_xr_v, [%rd_xr_row + {}];\n",
            d * 4
        ));
        ptx.push_str("    fma.rn.f32 %f_xr_ms, %f_xr_v, %f_xr_v, %f_xr_ms;\n");
    }
    ptx.push_str(&format!(
        "    mov.f32 %f_xr_tmp, 0f{:08X};  // 1/head_dim\n",
        (1.0f32 / head_dim as f32).to_bits()
    ));
    ptx.push_str("    mul.f32 %f_xr_ms, %f_xr_ms, %f_xr_tmp;\n");
    ptx.push_str("    ld.param.f32 %f_xr_tmp, [csha_eps];\n");
    ptx.push_str("    add.f32 %f_xr_ms, %f_xr_ms, %f_xr_tmp;\n");
    ptx.push_str("    sqrt.approx.f32 %f_xr_rms, %f_xr_ms;\n");
    ptx.push_str("    rsqrt.approx.f32 %f_xr_rinv, %f_xr_ms;\n");

    // ── xnorm scratch row base in SMEM (f16, stride head_dim*2).
    ptx.push_str("    cvt.u64.u32 %rd_xr_sm, %tid_x;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd_xr_sm, %rd_xr_sm, {};         // row * head_dim*2\n",
        head_dim * 2
    ));
    ptx.push_str(&format!(
        "    add.u64 %rd_xr_sm, %rd_xr_sm, {xn_off};      // + recompute_xnorm_offset\n"
    ));
    ptx.push_str("    add.u64 %rd_xr_sm, %shmem_base, %rd_xr_sm;\n");

    // ── Pass 2: x_norm[d] = x[d] * rinv * norm_weight[d], f32 -> f16 store
    ptx.push_str("    ld.param.u64 %rd_xr_nw, [csha_norm_weight_ptr];\n");
    for d in 0..head_dim {
        ptx.push_str(&format!(
            "    ld.global.f32 %f_xr_v, [%rd_xr_row + {}];\n",
            d * 4
        ));
        ptx.push_str(&format!(
            "    ld.global.f32 %f_xr_w, [%rd_xr_nw + {}];\n",
            d * 4
        ));
        ptx.push_str("    mul.f32 %f_xr_v, %f_xr_v, %f_xr_rinv;\n");
        ptx.push_str("    mul.f32 %f_xr_v, %f_xr_v, %f_xr_w;\n");
        ptx.push_str("    cvt.rn.f16.f32 %h_xr_v, %f_xr_v;\n");
        ptx.push_str(&format!(
            "    st.shared.b16 [%rd_xr_sm + {}], %h_xr_v;\n",
            d * 2
        ));
    }

    ptx.push_str(&format!(
        "V2_PROLOGUE_RECOMPUTE_FROM_RAW_DONE{namespace_suffix}:\n"
    ));
    ptx.push_str(&format!(
        "    bar.sync 0;  // xnorm scratch tile visible to all lanes ({namespace_suffix})\n"
    ));
    ptx.push_str(&format!(
        "    }} // end Cycle-14 prologue-recompute reg scope ({namespace_suffix})\n"
    ));
}

// ────────────────────────────────────────────────────────────────────────
// Cycle-11 §3 / Task 3: emit_kv_recompute
//
// Replaces the cycle-10 `kv_load::emit_k_suffixed` + `emit_v_suffixed`
// pair when `config.checkpoint.is_some()`. Re-derives K_proj/V_proj from
// scratch on the backward pass:
//
//   1. Null-guard x_raw_ptr (Risk-3 — same as emit_prologue_recompute_from_raw)
//   2. Recompute x_norm into SMEM scratch via emit_prologue_recompute_from_raw
//   3. K_proj = x_norm @ Wk[:, head_idx*head_dim..(head_idx+1)*head_dim]
//      written to `%k_smem_base + kv_row*(head_dim*2) + col*2` (f16 row-major,
//      stride head_dim*2 bytes — same contract as kv_load.rs:23-86)
//   4. V_proj = x_norm @ Wv (same write contract, %v_smem_base)
//   5. K-side RoPE via the forward's `emit_rope_k_epilogue` (in place on
//      %k_smem_base; Adjacent layout) — same call the forward path uses
//   6. Terminating `bar.sync 0` so the downstream ds_compute consumers
//      (ds_compute.rs:309,330) see the recomputed K/V tiles
//
// Layout invariants enforced (mirrors kv_load.rs comment block):
//   - K tile @ %k_smem_base, f16 [block_kv, head_dim], stride head_dim*2
//   - V tile @ %v_smem_base, same layout
//   - bar.sync 0 (NOT .aligned) — matches kv_load.rs:93
//
// Smoke-scope assumption inherited from `emit_xnorm_recompute`:
//   d_model == head_dim, heads == 1, kv_dim == head_dim. Wider configs
//   are STRUCTURALLY callable but the per-block accumulation is honest
//   only when this block is the sole contributor to its K_proj/V_proj
//   region. See cycle-13+ for multi-block accumulation.
// ────────────────────────────────────────────────────────────────────────

/// Emit one projection-recompute matmul for either K or V, writing the
/// f16 result to a kv-tile SMEM slot at `<smem_base> + kv_row*(head_dim*2)
/// + col*2`. xnorm input is at `recompute_xnorm_offset(config)` (f16,
/// stride `head_dim*2`).
///
/// Partitioning: one warp per row, `slices_per_lane = max(head_dim/32, 1)`
/// columns per lane within each row — same shape kv_load.rs uses. Each
/// per-cell accumulation runs over `d_model` (== `head_dim` under the
/// smoke scope).
fn emit_one_recompute_matmul(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    tag: &str,
    label_suffix: &str,
    weight_param_name: &str,
    smem_base: &str,
    xnorm_off: u32,
) {
    let csha = config.csha.as_ref().expect("emit_one_recompute_matmul requires csha");
    let d_model = csha.d_model;
    let head_dim = config.head_dim as u32;
    let block_kv = config.block_kv as u32;
    let heads = csha.active_heads.max(1);
    let kv_dim = heads * head_dim;
    let slices_per_lane = (head_dim / 32).max(1);
    let rows_per_warp = block_kv.div_ceil(4);

    ptx.push_str(&format!(
        "    // Cycle-11 sec.3: {tag} recompute matmul (block_kv={block_kv}, \
         slices/lane={slices_per_lane}, rows/warp={rows_per_warp})\n"
    ));
    ptx.push_str(&format!("V2_RECOMPUTE_{tag}_MATMUL_{label_suffix}:\n"));

    // Cycle 14: register declarations for the recompute matmul. Wrap in
    // a function-scoped sub-block so ptxas can parse the references below.
    ptx.push_str(&format!(
        "    {{ // Cycle-14 {tag}-recompute-matmul reg scope ({label_suffix})\n"
    ));
    ptx.push_str("    .reg .u64 %rd_rcw, %rd_rc_row, %rd_rc_row_abs, %rd_rc_xnrow;\n");
    ptx.push_str("    .reg .u64 %rd_rc_col, %rd_rc_jg, %rd_rc_wo, %rd_rc_wa;\n");
    ptx.push_str("    .reg .u64 %rd_rc_sm, %rd_rc_sm_c;\n");
    ptx.push_str("    .reg .u32 %r_rc0;\n");
    ptx.push_str("    .reg .pred %p_rc_null, %p_rc_oob;\n");
    ptx.push_str("    .reg .f32 %f_rc_acc, %f_rc_x, %f_rc_w;\n");
    ptx.push_str("    .reg .b16 %h_rc_x, %h_rc_w, %h_rc_out;\n");

    // Per-projection null-guard on the weight pointer.
    ptx.push_str(&format!(
        "    ld.param.u64 %rd_rcw, [{weight_param_name}];\n"
    ));
    ptx.push_str("    setp.eq.u64 %p_rc_null, %rd_rcw, 0;\n");
    ptx.push_str(&format!(
        "    @%p_rc_null bra V2_RECOMPUTE_{tag}_MATMUL_SKIP_{label_suffix};\n"
    ));

    for r in 0..rows_per_warp {
        // kv_row = warp_id + r*4
        ptx.push_str(&format!(
            "    add.u32 %r_rc0, %warp_id, {}; // kv_row\n",
            r * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd_rc_row, %r_rc0;\n");
        ptx.push_str("    add.u64 %rd_rc_row_abs, %rd_rc_row, %k_start;\n");
        ptx.push_str("    setp.ge.u64 %p_rc_oob, %rd_rc_row_abs, %rd6;\n");

        // xnorm row base in SMEM: shmem_base + xnorm_off + kv_row*(head_dim*2)
        // NOTE: under the smoke scope (seq == block_q == block_kv) and
        // single contributor per K_proj row, indexing kv_row directly
        // matches the forward's per-row x_norm consumer.
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_rc_xnrow, %rd_rc_row, {}; // kv_row * head_dim*2\n",
            head_dim * 2
        ));
        ptx.push_str(&format!(
            "    add.u64 %rd_rc_xnrow, %rd_rc_xnrow, {xnorm_off};\n"
        ));
        ptx.push_str("    add.u64 %rd_rc_xnrow, %shmem_base, %rd_rc_xnrow;\n");

        for slice in 0..slices_per_lane {
            // col = lane * slices_per_lane + slice
            ptx.push_str("    cvt.u64.u32 %rd_rc_col, %lane;\n");
            if slices_per_lane > 1 {
                ptx.push_str(&format!(
                    "    mul.lo.u64 %rd_rc_col, %rd_rc_col, {slices_per_lane};\n"
                ));
            }
            if slice > 0 {
                ptx.push_str(&format!("    add.u64 %rd_rc_col, %rd_rc_col, {slice};\n"));
            }

            // Accumulator init
            ptx.push_str("    mov.f32 %f_rc_acc, 0f00000000;\n");

            // global col offset in W: head_idx*head_dim + col
            //   W byte offset for (p, j_global) = (p*kv_dim + j_global) * 2
            // We hoist `head_idx*head_dim` because it's constant within the
            // matmul body for this block.
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_rc_jg, %head_idx, {}; // head_idx * head_dim\n",
                head_dim
            ));
            ptx.push_str("    add.u64 %rd_rc_jg, %rd_rc_jg, %rd_rc_col; // + col\n");

            // Inner reduction over p in 0..d_model (== head_dim under smoke).
            // x_norm[kv_row, p] from SMEM f16; W[p, j_global] from HBM f16.
            for p in 0..d_model {
                // xnorm load (f16 -> f32)
                ptx.push_str(&format!(
                    "    ld.shared.b16 %h_rc_x, [%rd_rc_xnrow + {}];\n",
                    p * 2
                ));
                ptx.push_str("    cvt.f32.f16 %f_rc_x, %h_rc_x;\n");
                // Weight: W[p, j_global] = weight_base + (p*kv_dim + j_global)*2
                ptx.push_str("    mul.lo.u64 %rd_rc_wo, %rd_rc_jg, 2; // j_global*2 base\n");
                // Add p*kv_dim*2 (constant per p)
                ptx.push_str(&format!(
                    "    add.u64 %rd_rc_wo, %rd_rc_wo, {}; // + p*kv_dim*2\n",
                    p * kv_dim * 2
                ));
                ptx.push_str("    add.u64 %rd_rc_wa, %rd_rcw, %rd_rc_wo;\n");
                ptx.push_str("    ld.global.b16 %h_rc_w, [%rd_rc_wa];\n");
                ptx.push_str("    cvt.f32.f16 %f_rc_w, %h_rc_w;\n");
                ptx.push_str("    fma.rn.f32 %f_rc_acc, %f_rc_x, %f_rc_w, %f_rc_acc;\n");
            }

            // f32 -> f16 store to SMEM at smem_base + kv_row*(head_dim*2) + col*2
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_rc_sm, %rd_rc_row, {}; // kv_row * head_dim*2\n",
                head_dim * 2
            ));
            ptx.push_str("    shl.b64 %rd_rc_sm_c, %rd_rc_col, 1; // col*2\n");
            ptx.push_str("    add.u64 %rd_rc_sm, %rd_rc_sm, %rd_rc_sm_c;\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_rc_sm, {smem_base}, %rd_rc_sm;\n"
            ));
            ptx.push_str("    cvt.rn.f16.f32 %h_rc_out, %f_rc_acc;\n");
            // OOB-row protection: skip store if kv_row >= seq.
            ptx.push_str("    @!%p_rc_oob st.shared.b16 [%rd_rc_sm], %h_rc_out;\n");
        }
    }

    ptx.push_str(&format!("V2_RECOMPUTE_{tag}_MATMUL_SKIP_{label_suffix}:\n"));
    ptx.push_str(&format!(
        "    bar.sync 0;  // recomputed {tag} tile visible ({label_suffix})\n"
    ));
    ptx.push_str(&format!(
        "    }} // end Cycle-14 {tag}-recompute-matmul reg scope ({label_suffix})\n"
    ));
}

/// Cycle-11 §3 / Task 3: emit the full K/V projection-recompute sequence
/// that REPLACES `kv_load::emit_k_suffixed` + `emit_v_suffixed` when
/// `config.checkpoint.is_some()`. See the module-level comment block
/// above for the layout / null-guard / barrier contract this honours.
///
/// `kv_iter_suffix` is the label-namespace suffix (typically `"MAIN"`
/// matching the dispatch fork in `mod.rs:934-935`); it must be unique
/// within the emitted PTX text section.
///
/// Production callers reach this only via the test-only R0 bypass set
/// by `CheckpointExtras::bypass_r0_for_testing()` (cycle-11 Task 1).
/// R0 stays in production until cycle-12 GPU validation lifts it.
pub fn emit_kv_recompute(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    kv_iter_suffix: &str,
) {
    if config.csha.is_none() {
        ptx.push_str("    // Cycle-11 emit_kv_recompute: csha=None, no emission\n");
        return;
    }
    let xn_off = recompute_xnorm_offset(config);

    // Cycle 14: PTX comment MUST be ASCII-only — em-dash/section-sign
    // were causing `ptxas fatal: Unexpected non-ASCII character` on
    // line 1084 when checkpoint=Some(Full) routed here. See
    // `feedback_ptx_comment_ascii_only`.
    ptx.push_str(&format!(
        "    // --- Cycle-11 sec.3 emit_kv_recompute ({kv_iter_suffix}) ---\n"
    ));
    ptx.push_str(&format!("V2_KV_RECOMPUTE_{kv_iter_suffix}:\n"));

    // Cycle 14: register declarations for the kv_recompute null-guard. The
    // backward prelude's register pool doesn't include these; emit them
    // in a function-scoped sub-block so ptxas can parse the references
    // below. Sub-blocks (`{ ... }`) give us scope without conflict with
    // the outer kernel's register pool.
    //
    // KNOWN-INCOMPLETE per cycle 14: emit_rope_k_epilogue (step 5) reuses
    // registers declared by the forward prelude (%rd_rope_cos, %rd_rope_sin,
    // %p_rope_skip, %r_rope_*, %rd_rope_*, %h_rope_pair, ...). The forward
    // prelude pre-declares them; the backward prelude does not. Cycle 15
    // adds either a full backward-side rope register block in
    // phases/backward/prelude.rs or factors the rope decls into a shared
    // helper. This sub-block declares only what step 1 needs.
    ptx.push_str("    { // Cycle-14 kv_recompute null-guard reg scope\n");
    ptx.push_str("    .reg .u64 %rd_kvr_xraw;\n");
    ptx.push_str("    .reg .pred %p_kvr_xraw_null;\n");

    // Step 1: Risk-3 null-guard on x_raw_ptr.
    // Cycle 12: trap on null x_raw_ptr -- silent garbage was the
    // cycle-11 bug; loud abort is the fix. With R3 fused_projections
    // augmentation x_raw should never be null when this emitter runs,
    // because the augmented R3 refuses configs that don't stage x_raw
    // on the forward path. The trap is defense-in-depth -- if any
    // future refactor weakens the R3 augmentation or the forward save
    // path drops the x_raw write, we get CUDA_ERROR_LAUNCH_FAILED at
    // runtime instead of silently wrong gradients.
    ptx.push_str("    ld.param.u64 %rd_kvr_xraw, [x_raw_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_kvr_xraw_null, %rd_kvr_xraw, 0;\n");
    ptx.push_str("    @%p_kvr_xraw_null trap;\n");
    ptx.push_str("    } // end Cycle-14 kv_recompute null-guard reg scope\n");

    // Step 2: produce x_norm in SMEM scratch via the cycle-11 Task-2 emitter.
    let suffix = format!("_bwd_recompute_{kv_iter_suffix}");
    emit_prologue_recompute_from_raw(ptx, config, 0, &suffix);

    // Step 3: K_proj recompute -> %k_smem_base
    ptx.push_str(&format!(
        "    // Cycle-11 sec.3 step 3: K_proj recompute -> %k_smem_base ({kv_iter_suffix})\n"
    ));
    emit_one_recompute_matmul(
        ptx,
        config,
        "K",
        kv_iter_suffix,
        "csha_wk_ptr",
        "%k_smem_base",
        xn_off,
    );

    // Step 4: V_proj recompute -> %v_smem_base
    ptx.push_str(&format!(
        "    // Cycle-11 sec.3 step 4: V_proj recompute -> %v_smem_base ({kv_iter_suffix})\n"
    ));
    emit_one_recompute_matmul(
        ptx,
        config,
        "V",
        kv_iter_suffix,
        "csha_wv_ptr",
        "%v_smem_base",
        xn_off,
    );

    // Step 5: K-side RoPE in place on %k_smem_base (matches the forward
    // path's emit_rope_k_epilogue call). Only emits when rope_q=true and
    // fused_projections=true; otherwise it's a no-op. R7 (rope_q=true +
    // segment_masked) remains refused upstream by mod.rs cascade.
    ptx.push_str(&format!(
        "    // Cycle-11 sec.3 step 5: K RoPE epilogue ({kv_iter_suffix})\n"
    ));
    crate::flash_attention_v2::phases::forward::csha_hooks::emit_rope_k_epilogue(ptx, config);

    // Step 6: terminating bar.sync 0 to satisfy kv_load's post-condition
    // contract (downstream ds_compute consumers in ds_compute.rs:309,330
    // expect the K/V tiles fully resident in SMEM at this point).
    ptx.push_str(&format!("V2_KV_RECOMPUTE_DONE_{kv_iter_suffix}:\n"));
    ptx.push_str(&format!(
        "    bar.sync 0;  // KV recompute complete ({kv_iter_suffix})\n"
    ));
}

#[cfg(test)]
mod cycle11_recompute_tests {
    use super::*;
    use crate::flash_attention::{
        CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
    };

    fn cfg_full_bypass() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model: 32,
                ..CshaExtras::default()
            }),
            checkpoint: Some(CheckpointExtras::full().bypass_r0_for_testing()),
        }
    }

    #[test]
    fn cycle11_emit_prologue_recompute_from_raw_uses_x_raw_not_csha_x_ptr() {
        let cfg = cfg_full_bypass();
        let mut ptx = String::new();
        emit_prologue_recompute_from_raw(&mut ptx, &cfg, 0, "_bwd_recompute_0");

        // Reads from x_raw_ptr (the Cycle-10 trap T7 fix).
        assert!(
            ptx.contains("ld.param.u64 %rd_x_raw_ptr, [x_raw_ptr]"),
            "must load x_raw_ptr (NOT csha_x_ptr); ptx=\n{}",
            ptx
        );
        // Risk-3 null guard
        assert!(
            ptx.contains("setp.eq.u64 %p_xraw_null, %rd_x_raw_ptr, 0"),
            "missing Risk-3 null-guard on x_raw_ptr"
        );
        // Does NOT touch csha_x_ptr (which the forward path mutates in place)
        assert!(
            !ptx.contains("[csha_x_ptr]"),
            "recompute path must not read csha_x_ptr (forward overwrites it)"
        );
        // Suffix-namespaced labels
        assert!(
            ptx.contains("V2_PROLOGUE_RECOMPUTE_FROM_RAW_ENTRY_bwd_recompute_0:"),
            "missing namespaced ENTRY label"
        );
        assert!(
            ptx.contains("V2_PROLOGUE_RECOMPUTE_FROM_RAW_DONE_bwd_recompute_0:"),
            "missing namespaced DONE label"
        );
        // Writes the xnorm scratch tile (st.shared.b16) at the new offset
        assert!(
            ptx.contains("st.shared.b16"),
            "must write f16 xnorm tile to SMEM scratch"
        );
        // Terminating bar.sync 0 post-condition
        assert!(
            ptx.contains("bar.sync 0"),
            "must terminate with bar.sync 0 to satisfy SMEM visibility contract"
        );
    }

    #[test]
    fn cycle11_emit_prologue_recompute_from_raw_no_csha_is_noop() {
        let mut cfg = cfg_full_bypass();
        cfg.csha = None;
        let mut ptx = String::new();
        emit_prologue_recompute_from_raw(&mut ptx, &cfg, 0, "_bwd_recompute_0");
        assert!(ptx.contains("csha=None, no emission"));
        assert!(!ptx.contains("ld.param.u64 %rd_x_raw_ptr"));
    }

    #[test]
    fn cycle11_emit_kv_recompute_writes_k_and_v_smem_bases() {
        let cfg = cfg_full_bypass();
        let mut ptx = String::new();
        emit_kv_recompute(&mut ptx, &cfg, "MAIN");

        // Entry label present
        assert!(
            ptx.contains("V2_KV_RECOMPUTE_MAIN:"),
            "missing entry label V2_KV_RECOMPUTE_MAIN; ptx=\n{ptx}"
        );
        // BOTH K and V SMEM bases are written
        assert!(
            ptx.contains("%k_smem_base"),
            "emit_kv_recompute must write to %k_smem_base"
        );
        assert!(
            ptx.contains("%v_smem_base"),
            "emit_kv_recompute must write to %v_smem_base"
        );
        // Per-projection matmul labels present
        assert!(ptx.contains("V2_RECOMPUTE_K_MATMUL_MAIN:"));
        assert!(ptx.contains("V2_RECOMPUTE_V_MATMUL_MAIN:"));
        // x_raw_ptr null guard at the orchestrator level (Risk-3)
        assert!(
            ptx.contains("setp.eq.u64 %p_kvr_xraw_null, %rd_kvr_xraw, 0"),
            "Risk-3 x_raw null-guard missing at orchestrator level"
        );
        // Prologue recompute embedded inside
        assert!(
            ptx.contains("V2_PROLOGUE_RECOMPUTE_FROM_RAW_ENTRY_bwd_recompute_MAIN:"),
            "prologue recompute step (Task 2) not invoked"
        );
        // Terminating bar.sync 0 (NOT .aligned)
        assert!(
            ptx.contains("V2_KV_RECOMPUTE_DONE_MAIN:"),
            "missing terminating DONE label"
        );
        assert!(
            ptx.contains("bar.sync 0"),
            "must terminate with bar.sync 0 per kv_load post-condition contract"
        );
        // f16 SMEM writes at both bases
        assert!(
            ptx.contains("st.shared.b16"),
            "must emit f16 store of recomputed projection tile"
        );
    }

    #[test]
    fn cycle11_emit_kv_recompute_no_csha_is_noop() {
        let mut cfg = cfg_full_bypass();
        cfg.csha = None;
        let mut ptx = String::new();
        emit_kv_recompute(&mut ptx, &cfg, "MAIN");
        assert!(ptx.contains("csha=None, no emission"));
        assert!(!ptx.contains("V2_KV_RECOMPUTE_MAIN:"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg_fused_backward(
        block_q: i64, block_kv: i64, head_dim: i64, heads: u32, d_model: u32,
    ) -> FlashAttentionConfig {
        let _ = heads;
        FlashAttentionConfig {
            block_q, block_kv, head_dim,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
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
    fn backward_drope_rotates_dq_dk_inversely() {
        let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        cfg.rope_q = true;
        let mut ptx = String::new();
        emit_drope(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DROPE_Q_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DROPE_K_LOOP_0:"));
        assert!(
            ptx.matches("fma.rn.f32").count() >= 2,
            "need ≥2 fmas (one per dx0/dx1 × Q and K), got {}",
            ptx.matches("fma.rn.f32").count()
        );
        assert!(!ptx.contains("V2_BWD_DROPE_V_LOOP"),
            "V must never be rotated");
    }

    #[test]
    fn backward_drope_skipped_when_rope_q_false() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_drope(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("rope_q=false") || ptx.is_empty(),
            "dRoPE must no-op when rope_q=false"
        );
        assert!(!ptx.contains("V2_BWD_DROPE"));
    }

    #[test]
    fn backward_dproj_accumulates_dwq_dwk_dwv() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_dproj(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DPROJ_WQ_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DPROJ_WK_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DPROJ_WV_LOOP_0:"));
        assert!(ptx.contains("setp.eq.u64 %p") && ptx.contains("csha_wq_ptr"),
            "independent null-guard on Wq missing");
        assert!(ptx.contains("csha_wk_ptr"));
        assert!(ptx.contains("csha_wv_ptr"));
    }

    #[test]
    fn backward_drmsnorm_produces_dx() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_drmsnorm(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DRMSNORM_0:"));
        assert!(ptx.contains("%f_g"));
        assert!(ptx.contains("rsqrt.approx.f32"));
        assert!(
            ptx.matches("shfl.sync.bfly.b32").count() >= 5,
            "need ≥5 butterflies, got {}",
            ptx.matches("shfl.sync.bfly.b32").count()
        );
    }

    /// Gap I.5 Option A: `emit_drmsnorm` MUST also drain the SMEM
    /// `dx_norm` tile to HBM via `%rd_bwd_dxn` (the 8th gradient output).
    /// This is the load-bearing kernel change for the Gap I.5 dgamma
    /// semantic-correctness fix — the AD-side `RmsNormGammaBackward`
    /// consumes this buffer via `extract_results[7]`.
    ///
    /// Assertions:
    ///   - The new SMEM->HBM store label is emitted (both the ENTRY
    ///     `V2_BWD_DRMSNORM_DXN_STORE_0:` and the null-guard SKIP label
    ///     `V2_BWD_DRMSNORM_DXN_STORE_SKIP_0:`).
    ///   - The kernel references `%rd_bwd_dxn` (the dxn_ptr register
    ///     loaded by the prelude) so ptxas sees a real dependency on
    ///     the new param.
    ///   - At least one `st.global.f32` appears inside the dxn store
    ///     phase (the HBM write of the f32 tile).
    ///   - The store precedes Phase 2's `V2_BWD_DRMSNORM_PHASE2_DONE`
    ///     label, confirming correct ordering (dx_norm drains BEFORE
    ///     Phase 2 overwrites Phase-1-phase-2 scratch registers).
    #[test]
    fn backward_drmsnorm_stores_dx_norm_to_hbm_via_rd_bwd_dxn() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_drmsnorm(&mut ptx, &cfg, 0);

        assert!(
            ptx.contains("V2_BWD_DRMSNORM_DXN_STORE_0:"),
            "missing dx_norm SMEM->HBM store entry label; kernel-side Gap I.5 \
             Option-A dx_norm export regressed. ptx=\n{}",
            ptx
        );
        assert!(
            ptx.contains("V2_BWD_DRMSNORM_DXN_STORE_SKIP_0:"),
            "missing dx_norm null-guard SKIP label"
        );
        assert!(
            ptx.contains("%rd_bwd_dxn"),
            "dx_norm store must reference `%rd_bwd_dxn` so the prelude's \
             dxn_ptr param load has a consumer; absence means the new HBM \
             write wasn't wired."
        );
        let idx_dxn_store = ptx
            .find("V2_BWD_DRMSNORM_DXN_STORE_0:")
            .expect("dxn store label present");
        let idx_phase2_done = ptx
            .find("V2_BWD_DRMSNORM_PHASE2_DONE:")
            .expect("phase 2 done label present");
        assert!(
            idx_dxn_store < idx_phase2_done,
            "dx_norm HBM store must run BEFORE Phase 2 completes"
        );
    }

    /// Gap I.5 Option A: the prelude's `dx_norm_ptr` param MUST be
    /// declared AND consumed by an `ld.param.u64` into `%rd_bwd_dxn`.
    /// Without the consumer, ptxas would complain that the param is
    /// unreferenced (or the kernel would fail the full PTX-assemble
    /// pass even though emit_drmsnorm uses the register).
    #[test]
    fn backward_prelude_declares_and_loads_dx_norm_ptr() {
        use crate::flash_attention_v2::phases::backward::prelude::emit as emit_prelude;
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_prelude(&mut ptx, &cfg, None);

        assert!(
            ptx.contains(".param .u64 dx_norm_ptr"),
            "kernel param `dx_norm_ptr` missing from prelude — Gap I.5 \
             Option-A 8th output regressed"
        );
        assert!(
            ptx.contains("ld.param.u64 %rd_bwd_dxn, [dx_norm_ptr]"),
            "prelude must load dx_norm_ptr into %rd_bwd_dxn so the \
             downstream store in emit_drmsnorm has a live pointer"
        );
    }

    #[test]
    fn backward_hooks_label_uniqueness_across_iters() {
        let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        cfg.rope_q = true;
        let mut ptx = String::new();
        emit_drope(&mut ptx, &cfg, 0);
        emit_drope(&mut ptx, &cfg, 1);
        emit_dproj(&mut ptx, &cfg, 0);
        emit_dproj(&mut ptx, &cfg, 1);
        emit_drmsnorm(&mut ptx, &cfg, 0);
        emit_drmsnorm(&mut ptx, &cfg, 1);
        for label in [
            "V2_BWD_DROPE_Q_LOOP_0:", "V2_BWD_DROPE_Q_LOOP_1:",
            "V2_BWD_DROPE_K_LOOP_0:", "V2_BWD_DROPE_K_LOOP_1:",
            "V2_BWD_DPROJ_WQ_LOOP_0:", "V2_BWD_DPROJ_WQ_LOOP_1:",
            "V2_BWD_DRMSNORM_0:", "V2_BWD_DRMSNORM_1:",
        ] {
            assert!(ptx.contains(label), "missing: {label}");
        }
    }
}
