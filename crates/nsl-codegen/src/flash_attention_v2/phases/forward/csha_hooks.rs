//! CSHA Tier A extras - prologue (RMSNorm), matmul projection (Q/K/V/O),
//! RoPE epilogue, active_heads guard. Each hook is null-guarded: if the
//! respective CSHA pointer is 0 (e.g. `csha: None`), the kernel skips
//! the phase and falls through to the classic Q-from-HBM path.
//!
//! All hooks obey the warp-per-row contract. Labels are parameterised
//! on `q_tile_iter` so the orchestrator (Task 11) can call them multiple
//! times for block_q > 4 configs without duplicate-label errors.

use crate::flash_attention::{FlashAttentionConfig, RopeStyle};

/// # CSHA paper §5.2 (dead-head elimination) — v1 envelope
///
/// Two-tier elimination, both shipped pre-cycle-2:
///   * **Launcher truncation** (primary): when `csha.active_heads > 0` and
///     < total heads, the `nsl_flash_attention_csha*` runtime launches
///     `gridDim.y = batch * effective_heads` (see
///     `nsl-runtime/src/flash_attention.rs` near line 654 for the forward
///     CSHA path, with parallel sites near 976 / 1428 / 2145 for
///     scalar+fused backward variants). Blocks for dead heads NEVER
///     launch — no wasted SM cycles. Formula pinned by the
///     `a4_grid_y_*` unit tests in that file.
///   * **In-kernel guard** (this function, defense in depth): runtime
///     predicate that short-circuits the block prologue when the block's
///     head index would have been classified dead. Cheap, ASCII-PTX,
///     no SMEM cost. The guard reads the `csha_active_heads` *kernel
///     param* (not the codegen-time `config.csha.active_heads` literal)
///     so a single emitted kernel handles any pruning count the launcher
///     supplies, including the "no pruning" sentinel.
///
/// v1 envelope (cycle-2 audit):
///   * `csha_active_heads == 0` is a SENTINEL meaning "all heads live" —
///     the guard's first `setp.eq.u32 %r10, 0` branches over the
///     head-index check, so no head is ejected. In a paired launch the
///     launcher reports `effective_heads = heads` so the grid runs the
///     full head count too.
///   * `csha_active_heads > 0` triggers BOTH launcher truncation AND the
///     in-kernel `ret` — defense in depth: kernel stays safe if a future
///     caller forgets to truncate `gridDim.y` or constructs a grid
///     against a different active_heads than the kernel was specialised
///     against (cache-hit misrouting).
///   * Emission is gated only on `config.csha.is_some()`; the v2 hooks
///     variant emits the same PTX for any `csha.active_heads` value
///     because the predicate lives in the kernel param, not in a
///     compile-time literal. The legacy classic-path emitter at
///     `flash_attention.rs::emit_csha_active_heads_guard` IS literal-
///     gated (used by the non-v2 emitter); these two emitters cover
///     different dispatch paths and must stay numerically equivalent
///     at runtime.
///
/// Deferred to a future cycle (paper §5.2 v2):
///   * Per-active-head codegen specialization — emit N variant kernels,
///     one per active-head set, replacing the runtime guard with
///     compile-time dead code elimination. Trades binary size for
///     hot-path predicate cost.
///   * Per-head mixed precision (depends on FP8 M35 — paper §3.3).
///
/// # Emission contract
///
/// When `config.csha.is_none()`: emits a single comment line only — no
/// PTX instructions, no labels. This is what the non-CSHA dispatch
/// (classic FA-2 path with `csha: None`) ships.
///
/// When `config.csha.is_some()`: emits the full guard prelude — `ld.param`
/// of `csha_active_heads`, sentinel-zero skip branch (`@%p0 bra
/// V2_CSHA_ACTIVE_HEADS_SKIP`), head-index compare, conditional `ret`,
/// and the `V2_CSHA_ACTIVE_HEADS_SKIP:` label.
pub fn emit_active_heads_guard(ptx: &mut String, config: &FlashAttentionConfig) {
    if config.csha.is_none() {
        ptx.push_str("    // CSHA A.4 active_heads guard: csha=None, no emission\n");
        return;
    }
    ptx.push_str("    // CSHA A.4: active_heads guard\n");
    ptx.push_str("    ld.param.u32 %r10, [csha_active_heads];\n");
    ptx.push_str("    setp.eq.u32 %p0, %r10, 0;\n");
    ptx.push_str("    @%p0 bra V2_CSHA_ACTIVE_HEADS_SKIP;\n");
    // If head_idx >= active_heads, early-exit.
    ptx.push_str("    cvt.u32.u64 %r11, %head_idx;\n");
    ptx.push_str("    setp.ge.u32 %p0, %r11, %r10;\n");
    ptx.push_str("    @%p0 ret;\n");
    ptx.push_str("V2_CSHA_ACTIVE_HEADS_SKIP:\n");
}

/// Emit the §A.2.2 RMSNorm prologue. Computes
///     x_normed = x / sqrt(mean(x^2) + eps) * norm_weight
/// for the warp's query row and writes the result back into the x
/// buffer in-place. Null-guarded on `csha_x_ptr`.
pub fn emit_prologue(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    // Forward-path entry point: delegate to the namespaced variant with
    // an empty suffix to preserve byte-identity. Verified against the
    // `fa_v2_snapshots` baseline (25/25 byte-identical post-cycle-10).
    emit_prologue_namespaced(ptx, config, q_tile_iter, "");
}

/// Cycle-10 §5.3 Task 8: namespaced `emit_prologue` variant.
///
/// `namespace_suffix` is appended to every PTX *label* this function
/// emits (`V2_CSHA_PROLOGUE_SKIP_{q_tile_iter}{suffix}`). PTX registers
/// (`%rd52`, `%f0`, etc.) are NOT suffixed because they are declared
/// in the kernel prelude and shared across all phase emitters; the
/// backward recompute path will reload them per call, not re-declare
/// them.
///
/// Forward path passes `suffix = ""` to preserve byte-identity. Backward
/// recompute path (Task 9) will pass a backward-suitable suffix like
/// `"_bwd_{q_tile_iter}"` via `emit_prologue_recompute` below.
pub fn emit_prologue_namespaced(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    namespace_suffix: &str,
) {
    if config.csha.is_none() {
        ptx.push_str("    // CSHA A.2.2 prologue: csha=None, no emission\n");
        return;
    }
    let head_dim = config.head_dim as u32;
    let slices = head_dim / 32;

    ptx.push_str(&format!(
        "    // CSHA A.2.2: RMSNorm prologue (q_tile_iter = {})\n",
        q_tile_iter
    ));
    // Null-guard on x_ptr.
    ptx.push_str("    ld.param.u64 %rd52, [csha_x_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd52, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_CSHA_PROLOGUE_SKIP_{}{};\n",
        q_tile_iter, namespace_suffix
    ));

    // Tier C: load x_raw_ptr once (for the per-slice raw-x save below).
    // Null x_raw_ptr -> %p1 set, save stores are gated off per slice.
    ptx.push_str("    ld.param.u64 %rd58, [x_raw_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p1, %rd58, 0;\n");

    // Each warp normalizes its own x_row. Lane-strided sumsq across
    // head_dim slices, warp butterfly reduce, divide, multiply by
    // per-dim norm_weight.
    ptx.push_str("    mov.f32 %f0, 0f00000000;             // sumsq = 0\n");
    for i in 0..slices {
        ptx.push_str(&format!("    // x slice {}: load, square, accumulate\n", i));
        // Compute x row global offset.
        // x layout: [batch, heads, seq, head_dim] row-major, f32.
        ptx.push_str("    cvt.u64.u32 %rd53, %lane;\n");
        if i > 0 {
            ptx.push_str(&format!("    add.u64 %rd53, %rd53, {};\n", i * 32));
        }
        ptx.push_str("    mul.lo.u64 %rd54, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %head_idx;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd6;\n");
        ptx.push_str(&format!(
            "    add.u32 %r12, %warp_id, {};\n",
            q_tile_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd55, %r12;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %q_start;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd55;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd7;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd53;\n");
        ptx.push_str("    shl.b64 %rd54, %rd54, 2;\n");
        ptx.push_str("    add.u64 %rd54, %rd52, %rd54;\n");
        ptx.push_str("    ld.global.f32 %f1, [%rd54];\n");
        ptx.push_str("    fma.rn.f32 %f0, %f1, %f1, %f0;            // sumsq += x*x\n");
        // Tier C: save raw x[d] to x_raw_ptr at the same offset (skip if null).
        // %rd58 holds x_raw_ptr base, %p1 = (x_raw_ptr == 0).
        ptx.push_str("    sub.u64 %rd59, %rd54, %rd52;        // byte_off = abs - x_base\n");
        ptx.push_str("    add.u64 %rd59, %rd58, %rd59;        // x_raw addr\n");
        ptx.push_str("    @!%p1 st.global.f32 [%rd59], %f1;   // x_raw save (null-gated)\n");
    }
    // 5-step butterfly sum.
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f0, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    add.f32 %f0, %f0, %shfl_tmp;\n");
    }
    // mean = sumsq / head_dim; rms = sqrt(mean + eps); norm = 1/rms
    ptx.push_str(&format!(
        "    mov.f32 %f1, 0f{:08X};       // 1.0 / head_dim\n",
        (1.0f32 / head_dim as f32).to_bits()
    ));
    ptx.push_str("    mul.f32 %f0, %f0, %f1;\n");
    ptx.push_str("    ld.param.f32 %f1, [csha_eps];\n");
    ptx.push_str("    add.f32 %f0, %f0, %f1;\n");
    ptx.push_str("    sqrt.approx.f32 %f0, %f0;\n");
    ptx.push_str("    rcp.approx.f32 %f0, %f0;                  // 1/rms\n");

    // Second pass: x_normed[d] = x[d] * (1/rms) * norm_weight[d], writeback.
    for i in 0..slices {
        ptx.push_str(&format!("    // x slice {}: normalize + scale, writeback\n", i));
        ptx.push_str("    cvt.u64.u32 %rd53, %lane;\n");
        if i > 0 {
            ptx.push_str(&format!("    add.u64 %rd53, %rd53, {};\n", i * 32));
        }
        ptx.push_str("    mul.lo.u64 %rd54, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %head_idx;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd6;\n");
        ptx.push_str(&format!(
            "    add.u32 %r12, %warp_id, {};\n",
            q_tile_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd55, %r12;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %q_start;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd55;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd7;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd53;\n");
        ptx.push_str("    shl.b64 %rd54, %rd54, 2;\n");
        ptx.push_str("    add.u64 %rd54, %rd52, %rd54;\n");
        ptx.push_str("    ld.global.f32 %f2, [%rd54];\n");
        ptx.push_str("    mul.f32 %f2, %f2, %f0;                    // x * 1/rms\n");
        // norm_weight[d] load
        ptx.push_str("    ld.param.u64 %rd56, [csha_norm_weight_ptr];\n");
        ptx.push_str("    shl.b64 %rd57, %rd53, 2;\n");
        ptx.push_str("    add.u64 %rd56, %rd56, %rd57;\n");
        ptx.push_str("    ld.global.f32 %f3, [%rd56];\n");
        ptx.push_str("    mul.f32 %f2, %f2, %f3;\n");
        ptx.push_str("    st.global.f32 [%rd54], %f2;\n");
    }

    ptx.push_str(&format!(
        "V2_CSHA_PROLOGUE_SKIP_{}{}:\n",
        q_tile_iter, namespace_suffix
    ));
    ptx.push_str("    bar.sync 0;  // FENCE: all prologue writes complete\n");
}

/// Cycle-10 §5.3 Task 8: backward-suitable wrapper around
/// `emit_prologue_namespaced`.
///
/// Used by the backward dispatch fork (Task 9) under
/// `CheckpointPolicy::Full` to re-emit the prologue at backward-pass
/// time with a backward-distinguished label namespace, so the
/// recomputed labels (e.g. `V2_CSHA_PROLOGUE_SKIP_0_bwd_0`) do not
/// collide with the forward-pass labels still resident in the same
/// PTX text section.
pub fn emit_prologue_recompute(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    namespace_suffix: &str,
) {
    emit_prologue_namespaced(ptx, config, q_tile_iter, namespace_suffix);
}

/// Emit the §A.2.3 matmul projection (Q/K/V fused projection).
///
/// Warp-per-row contract: each warp owns one output row; lanes distribute
/// the output's d dimension in slices of `head_dim/32`. Inner dot-product
/// uses the 5-step warp butterfly sum idiom from Phase 2 S compute.
/// A.2.3.2 lane-coherent scatter becomes a per-lane direct write within a
/// single row (no inter-row scatter needed because each warp owns its row
/// completely).
///
/// When `csha.fused_projections` is false (or `csha` is None) this is a
/// no-op. When all three weight pointers are non-null, three sweeps are
/// emitted for Q, K, and V respectively.  If any pointer is zero the
/// entire projection block is skipped (null-guard on the triple).
pub fn emit_matmul_projection(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let csha = match &config.csha {
        Some(c) if c.fused_projections => c,
        _ => {
            ptx.push_str("    // CSHA A.2.3 projection: csha=None or fused_projections=false\n");
            return;
        }
    };
    let d_model = csha.d_model;
    let head_dim = config.head_dim as u32;

    ptx.push_str(&format!(
        "    // CSHA A.2.3: Q/K/V matmul projection (q_tile_iter={}), d_model={}, head_dim={}\n",
        q_tile_iter, d_model, head_dim
    ));

    // Independent null-guards per weight pointer.  Each sweep is individually
    // gated: if Wq is null the Q sweep is skipped; likewise for Wk and Wv.
    // This replaces the C2 compound-guard that skipped K/V unconditionally.
    ptx.push_str("    ld.param.u64 %rd60, [csha_wq_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd61, [csha_wk_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd62, [csha_wv_ptr];\n");
    // Gate the whole projection block on Wq being non-null (Wq is the primary
    // signal that fused projection is active; Wk/Wv have their own per-sweep
    // null-checks inside each respective sweep block below).
    ptx.push_str("    setp.eq.u64 %p0, %rd60, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_CSHA_PROJECTION_SKIP_{};\n",
        q_tile_iter
    ));

    // ── A.2.3 register initialisation ────────────────────────────────────
    // %warp_row, %q_smem_base, %k_smem_base, %v_smem_base, %q_tile,
    // %k_tile, %v_tile, and %x_norm_base must be initialised here because
    // they are declared in the prelude but never assigned elsewhere.
    //
    // warp_row: 0-based row index of this warp within the current q-tile.
    //   warp_row = warp_id + q_tile_iter * 4
    //   Stored as u64 for address arithmetic.
    ptx.push_str(&format!(
        "    add.u32 %r_indim_Q_0, %warp_id, {}; // warp_row = warp_id + iter*4\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r_indim_Q_0;\n");

    // Q/K/V output SMEM base registers (absolute SMEM byte addresses).
    //   %q_smem_base → Q tile   at byte 0
    //   %k_smem_base → KV tile  at byte kv_offset
    //   %v_smem_base → KV tile  at byte kv_offset (K and V share the KV
    //                            region; V projection fires after K projection
    //                            and S-compute, so overwriting K with V is safe)
    let kv_off = crate::flash_attention_v2::smem_layout::kv_offset(config);
    ptx.push_str("    mov.u64 %q_smem_base, %shmem_base;\n");
    ptx.push_str(&format!(
        "    add.u64 %k_smem_base, %shmem_base, {}; // + kv_offset\n",
        kv_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %v_smem_base, %shmem_base, {}; // + kv_offset (shared with K)\n",
        kv_off
    ));

    // Weight-tile SMEM base registers (%q_tile / %k_tile / %v_tile).
    // Each tile occupies wq_tile_bytes = d_model * head_dim * 2 bytes.
    // Use sp_bytes() so the base is correct whether fused_projections expands
    // the SP region (iters×4_warps×block_kv×4) or not (4_warps×block_kv×4).
    let wt_base = crate::flash_attention_v2::smem_layout::sp_offset(config)
        + crate::flash_attention_v2::smem_layout::sp_bytes(config);
    let wt_bytes = crate::flash_attention_v2::smem_layout::wq_tile_bytes(config);
    ptx.push_str(&format!(
        "    add.u64 %q_tile, %shmem_base, {}; // Wq tile base in SMEM\n",
        wt_base
    ));
    ptx.push_str(&format!(
        "    add.u64 %k_tile, %shmem_base, {}; // Wk tile base in SMEM\n",
        wt_base + wt_bytes
    ));
    ptx.push_str(&format!(
        "    add.u64 %v_tile, %shmem_base, {}; // Wv tile base in SMEM\n",
        wt_base + 2 * wt_bytes
    ));

    // %x_norm_base: x_normed row for this warp in global memory (f32).
    // After the prologue, x has been normalised in-place and written back
    // to csha_x_ptr.  Address of this warp's row:
    //   x_ptr + ((head_idx * seq_len + q_start + warp_row) * head_dim) * 4
    // NOTE: the inner loop uses `ld.global.f32` (see emit_warp_per_row_sweep).
    ptx.push_str("    ld.param.u64 %x_norm_base, [csha_x_ptr];\n");
    // head_idx * seq_len
    ptx.push_str("    mul.lo.u64 %rd_wt_off, %head_idx, %rd6;\n");
    // + q_start (already computed)
    ptx.push_str("    add.u64 %rd_wt_off, %rd_wt_off, %q_start;\n");
    // + warp_row (= warp_id + q_tile_iter*4)
    ptx.push_str("    add.u64 %rd_wt_off, %rd_wt_off, %warp_row;\n");
    // * head_dim
    ptx.push_str("    mul.lo.u64 %rd_wt_off, %rd_wt_off, %rd7;\n");
    // * 4 (f32 byte size)
    ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_off, 2;\n");
    ptx.push_str("    add.u64 %x_norm_base, %x_norm_base, %rd_wt_off;\n");
    // ── end register initialisation ───────────────────────────────────────

    // Project Q and K into their respective SMEM tiles before the KV loop.
    // V projection is deferred to `emit_v_projection_in_kv_loop` (called from
    // mod.rs between softmax and PV-accum) because V and K share the same SMEM
    // region (both at kv_offset); emitting V here would overwrite K before
    // s_compute reads it.
    //
    // K sweep is additionally guarded on its own weight pointer so that callers
    // who supply pre-projected K via the classic k_ptr param can pass wk_ptr=null
    // and only the Q sweep fires.

    // Q sweep (gated above on %rd60 != 0).
    emit_warp_per_row_sweep(ptx, config, q_tile_iter, "Q", "%q_smem_base");

    // K sweep — skip if Wk is null (caller supplies pre-projected K via k_ptr).
    ptx.push_str("    setp.eq.u64 %p0, %rd61, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_CSHA_PROJ_K_SKIP_{};\n",
        q_tile_iter
    ));
    emit_warp_per_row_sweep(ptx, config, q_tile_iter, "K", "%k_smem_base");
    ptx.push_str(&format!("V2_CSHA_PROJ_K_SKIP_{}:\n", q_tile_iter));

    ptx.push_str(&format!("V2_CSHA_PROJECTION_SKIP_{}:\n", q_tile_iter));
    ptx.push_str("    bar.sync 0;  // FENCE: all projection writes visible to all threads\n");
}

/// Emit the V projection sweep into SMEM from within the KV-tile loop.
///
/// # Placement
///
/// Must be called **after softmax and before PV-accum** inside the KV loop in
/// `flash_attention_v2/mod.rs`.  K and V share `kv_offset` in SMEM; writing V
/// before s_compute would overwrite K.  This deferred sweep writes V into SMEM
/// after s_compute has consumed K so PV-accum sees the correct projected V.
///
/// When `csha.fused_projections` is false (or `csha` is None), or `csha_wv_ptr`
/// was null at kernel entry, this is a no-op (the classic `emit_v_tile_load`
/// path handled V from HBM).
pub fn emit_v_projection_in_kv_loop(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let fused = config.csha.as_ref().is_some_and(|c| c.fused_projections);
    if !fused {
        ptx.push_str("    // CSHA V projection (deferred): fused_projections=false, no-op\n");
        return;
    }
    ptx.push_str(&format!(
        "    // CSHA A.2.3 V projection (deferred, q_tile_iter={}): write V into kv_offset SMEM\n",
        q_tile_iter
    ));
    // Only fire when Wv is non-null (same gate as emit_matmul_projection K sweep).
    // %rd62 was loaded at entry to emit_matmul_projection; re-load here for correctness.
    ptx.push_str("    ld.param.u64 %rd62, [csha_wv_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd62, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_CSHA_V_PROJ_KV_SKIP_{};\n",
        q_tile_iter
    ));
    emit_warp_per_row_sweep(ptx, config, q_tile_iter, "V", "%v_smem_base");
    ptx.push_str(&format!("V2_CSHA_V_PROJ_KV_SKIP_{}:\n", q_tile_iter));
    ptx.push_str("    bar.sync 0;  // FENCE: V projection visible before PV-accum\n");
}

/// Emit one warp-per-row sweep computing `out_row = x_normed_row @ W`
/// where W is already loaded into the SMEM weight tile. The loop label
/// `V2_CSHA_PROJ_{label}_LOOP_{q_tile_iter}:` uniquely identifies this
/// sweep for ptxas label dedup. Each lane owns `head_dim/32` output
/// d-dimension slices and accumulates across d_model input elements using
/// a 5-step warp butterfly reduction per slice.
fn emit_warp_per_row_sweep(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    label: &str,       // "Q" / "K" / "V"
    smem_base: &str,   // destination SMEM base register name
) {
    let csha    = config.csha.as_ref().expect("fused_projections checked by caller");
    let d_model = csha.d_model;
    let head_dim = config.head_dim as u32;
    // slices_per_lane: each lane owns this many output d positions.
    // With head_dim=32 each lane owns exactly 1 slice.
    let slices_per_lane = (head_dim / 32).max(1);
    let label_lc = label.to_lowercase();

    ptx.push_str(&format!(
        "    // A.2.3 warp-per-row sweep: {} (q_tile_iter={}), slices/lane={}\n",
        label, q_tile_iter, slices_per_lane
    ));
    ptx.push_str(&format!("V2_CSHA_PROJ_{}_LOOP_{}:\n", label, q_tile_iter));

    for slice in 0..slices_per_lane {
        // Initialise f32 accumulator for this slice.
        ptx.push_str(&format!(
            "    mov.f32 %f_acc_{}_{}, 0f00000000;    // acc[{}][{}] = 0\n",
            label, slice, label, slice
        ));
        // Initialise in_dim loop counter.
        ptx.push_str(&format!(
            "    mov.u32 %r_indim_{}_{}, 0;           // in_dim loop counter\n",
            label, slice
        ));
        // Inner loop: dot-product accumulation over d_model input elements.
        ptx.push_str(&format!(
            "V2_CSHA_PROJ_{}_INDIM_{}_{}:\n",
            label, slice, q_tile_iter
        ));
        // Load x_normed[warp_row, in_dim] from global memory (f32).
        // x_norm_base points to HBM after the prologue wrote normed data back.
        // Address = x_norm_base + in_dim * 4 (f32 stride = 4 bytes).
        ptx.push_str(&format!(
            "    cvt.u64.u32 %rd_wt_off, %r_indim_{}_{};\n",
            label, slice
        ));
        ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_off, 2;    // in_dim * 4 bytes (f32)\n");
        ptx.push_str("    add.u64 %rd_wt_src, %x_norm_base, %rd_wt_off;\n");
        // Use f32 register directly — skip the f16 intermediate.
        ptx.push_str(&format!(
            "    ld.global.f32 %f_x_{}_{}, [%rd_wt_src];\n",
            label, slice
        ));
        // Load W[in_dim, lane_col] from SMEM weight tile (f16).
        // W layout: [d_model, head_dim], row-major, f16.
        // Row-stride = head_dim * 2 bytes.
        // Lane column = lane * slices_per_lane + slice.
        // Address = {label_lc}_tile + in_dim * head_dim * 2 + lane_col * 2.
        // Recompute in_dim offset cleanly (do NOT reuse %rd_wt_off which is f32-scaled).
        ptx.push_str(&format!(
            "    cvt.u64.u32 %rd_wt_off, %r_indim_{}_{};\n",
            label, slice
        ));
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_wt_off, %rd_wt_off, {}; // in_dim * head_dim * 2 (f16 W row-stride)\n",
            head_dim * 2
        ));
        ptx.push_str(&format!(
            "    add.u64 %rd_wt_dst, %{}_tile, %rd_wt_off;\n",
            label_lc
        ));
        // Add lane column offset: (lane * slices_per_lane + slice) * 2.
        ptx.push_str("    cvt.u64.u32 %rd_wt_off, %lane;\n");
        if slices_per_lane > 1 {
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_wt_off, %rd_wt_off, {};\n",
                slices_per_lane
            ));
        }
        if slice > 0 {
            ptx.push_str(&format!(
                "    add.u64 %rd_wt_off, %rd_wt_off, {};\n",
                slice
            ));
        }
        ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_off, 1; // * 2 bytes (f16)\n");
        ptx.push_str("    add.u64 %rd_wt_dst, %rd_wt_dst, %rd_wt_off;\n");
        ptx.push_str(&format!(
            "    ld.shared.b16 %h_w_{}_{}, [%rd_wt_dst];\n",
            label, slice
        ));
        // f_x is already f32 (loaded directly above); convert f_w f16→f32.
        ptx.push_str(&format!(
            "    cvt.f32.f16 %f_w_{}_{}, %h_w_{}_{};\n",
            label, slice, label, slice
        ));
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_acc_{}_{}, %f_x_{}_{}, %f_w_{}_{}, %f_acc_{}_{};\n",
            label, slice, label, slice, label, slice, label, slice
        ));
        // Advance in_dim and loop.
        ptx.push_str(&format!(
            "    add.u32 %r_indim_{}_{}, %r_indim_{}_{}, 1;\n",
            label, slice, label, slice
        ));
        ptx.push_str(&format!(
            "    setp.lt.u32 %p_indim_{}_{}, %r_indim_{}_{}, {};\n",
            label, slice, label, slice, d_model
        ));
        ptx.push_str(&format!(
            "    @%p_indim_{}_{} bra V2_CSHA_PROJ_{}_INDIM_{}_{};\n",
            label, slice, label, slice, q_tile_iter
        ));

        // No warp butterfly: each lane independently accumulates its OWN output
        // column (lane_col = lane * slices_per_lane + slice).  The partial sums
        // are already complete after the inner loop — each lane has the full
        // dot product for its own column.  Butterfly would incorrectly sum
        // independent outputs across lanes.

        // Convert accumulated f32 to f16 and store to SMEM output tile.
        // Layout: out_tile[warp_row, lane * slices_per_lane + slice] (f16).
        // Address = smem_base + warp_row * (head_dim * 2) + (lane*slices + slice) * 2
        // Use %rd_wt_dst as scratch; do NOT mutate smem_base so it stays valid for
        // subsequent slices, sweeps, and the q-from-smem load in q_load::emit.
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %h_out_{}_{}, %f_acc_{}_{};\n",
            label, slice, label, slice
        ));
        // row offset: warp_row * row_stride
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_wt_off, %warp_row, {};\n",
            head_dim * 2
        ));
        ptx.push_str(&format!(
            "    add.u64 %rd_wt_dst, {}, %rd_wt_off;\n",
            smem_base
        ));
        // column offset: (lane * slices_per_lane + slice) * 2 bytes
        ptx.push_str("    cvt.u64.u32 %rd_wt_off, %lane;\n");
        if slices_per_lane > 1 {
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_wt_off, %rd_wt_off, {};\n",
                slices_per_lane
            ));
        }
        if slice > 0 {
            ptx.push_str(&format!(
                "    add.u64 %rd_wt_off, %rd_wt_off, {};\n",
                slice
            ));
        }
        ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_off, 1;\n");
        ptx.push_str("    add.u64 %rd_wt_dst, %rd_wt_dst, %rd_wt_off;\n");
        // Store into the correct SMEM slot.
        ptx.push_str(&format!(
            "    st.shared.b16 [%rd_wt_dst], %h_out_{}_{};\n",
            label, slice
        ));
    }
}

// ── K/V pre-pass helpers for the fused-projection split-loop design ─────────
//
// When fused_projections=true the main synthesize function uses a 3-pass
// structure: K pre-pass → S-compute pass (with Q sweep) → V pre-pass →
// PV-accum pass. Each *_prepass_sweep function emits the per-warp-row setup
// (warp_row, smem bases, x_norm_base) plus the relevant sweep, mirroring the
// register init block in emit_matmul_projection.

/// Emit the register-init block used by K and V pre-pass sweeps.
/// Sets up warp_row, k/v_smem_base, q/k/v_tile, and x_norm_base for the
/// given q_tile_iter.  Does NOT emit the Q smem base (not needed for K/V).
fn emit_kv_prepass_reginit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let kv_off = crate::flash_attention_v2::smem_layout::kv_offset(config);
    let wt_base = crate::flash_attention_v2::smem_layout::sp_offset(config)
        + crate::flash_attention_v2::smem_layout::sp_bytes(config);
    let wt_bytes = crate::flash_attention_v2::smem_layout::wq_tile_bytes(config);

    // warp_row = warp_id + q_tile_iter * 4.  Reuse %r_indim_Q_0 as scratch.
    ptx.push_str(&format!(
        "    add.u32 %r_indim_Q_0, %warp_id, {}; // warp_row = warp_id + iter*4\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r_indim_Q_0;\n");

    // Q smem base (needed for q_tile register, even though Q sweep is not emitted here).
    ptx.push_str("    mov.u64 %q_smem_base, %shmem_base;\n");
    ptx.push_str(&format!(
        "    add.u64 %k_smem_base, %shmem_base, {};\n",
        kv_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %v_smem_base, %shmem_base, {};\n",
        kv_off
    ));

    // Weight tile SMEM bases.
    ptx.push_str(&format!(
        "    add.u64 %q_tile, %shmem_base, {};\n",
        wt_base
    ));
    ptx.push_str(&format!(
        "    add.u64 %k_tile, %shmem_base, {};\n",
        wt_base + wt_bytes
    ));
    ptx.push_str(&format!(
        "    add.u64 %v_tile, %shmem_base, {};\n",
        wt_base + 2 * wt_bytes
    ));

    // x_norm_base: global address of this warp's row in the normalized x tensor.
    ptx.push_str("    ld.param.u64 %x_norm_base, [csha_x_ptr];\n");
    ptx.push_str("    mul.lo.u64 %rd_wt_off, %head_idx, %rd6;\n");
    ptx.push_str("    add.u64 %rd_wt_off, %rd_wt_off, %q_start;\n");
    ptx.push_str("    add.u64 %rd_wt_off, %rd_wt_off, %warp_row;\n");
    ptx.push_str("    mul.lo.u64 %rd_wt_off, %rd_wt_off, %rd7;\n");
    ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_off, 2;\n");
    ptx.push_str("    add.u64 %x_norm_base, %x_norm_base, %rd_wt_off;\n");
}

/// Emit the K projection sweep for the K pre-pass (before any S-compute).
/// Writes K rows for this q_tile_iter's warps into SMEM at kv_offset.
pub fn emit_k_prepass_sweep(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if !config.csha.as_ref().is_some_and(|c| c.fused_projections) {
        return;
    }
    ptx.push_str(&format!(
        "    // K pre-pass sweep q_tile_iter={}\n", q_tile_iter
    ));
    // Load Wk pointer (already in SMEM from the one-time weight tile load;
    // re-load into %rd61 for the sweep's null-guard pattern).
    ptx.push_str("    ld.param.u64 %rd61, [csha_wk_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd61, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_K_PREPASS_SKIP_{};\n", q_tile_iter
    ));
    emit_kv_prepass_reginit(ptx, config, q_tile_iter);
    emit_warp_per_row_sweep(ptx, config, q_tile_iter, "K", "%k_smem_base");
    ptx.push_str(&format!("V2_K_PREPASS_SKIP_{}:\n", q_tile_iter));
}

/// Emit only the Q projection sweep (no K) for the S-compute pass.
/// Writes Q rows for this q_tile_iter into q_smem (q_offset).
/// Assumes x has been normalized by the prologue and weight tiles are in SMEM.
pub fn emit_q_projection_only(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if !config.csha.as_ref().is_some_and(|c| c.fused_projections) {
        return;
    }
    ptx.push_str(&format!(
        "    // Q-only projection sweep q_tile_iter={}\n", q_tile_iter
    ));
    // Load Wq pointer for null guard.
    ptx.push_str("    ld.param.u64 %rd60, [csha_wq_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd60, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_Q_ONLY_PROJ_SKIP_{};\n", q_tile_iter
    ));
    emit_kv_prepass_reginit(ptx, config, q_tile_iter);
    emit_warp_per_row_sweep(ptx, config, q_tile_iter, "Q", "%q_smem_base");
    ptx.push_str(&format!("V2_Q_ONLY_PROJ_SKIP_{}:\n", q_tile_iter));
    ptx.push_str("    bar.sync 0; // Q rows visible before Q-load\n");
}

/// Emit the V projection sweep for the V pre-pass (after all S-computes).
/// Writes V rows for this q_tile_iter's warps into SMEM at kv_offset.
pub fn emit_v_prepass_sweep(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if !config.csha.as_ref().is_some_and(|c| c.fused_projections) {
        return;
    }
    ptx.push_str(&format!(
        "    // V pre-pass sweep q_tile_iter={}\n", q_tile_iter
    ));
    ptx.push_str("    ld.param.u64 %rd62, [csha_wv_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd62, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_V_PREPASS_SKIP_{};\n", q_tile_iter
    ));
    emit_kv_prepass_reginit(ptx, config, q_tile_iter);
    emit_warp_per_row_sweep(ptx, config, q_tile_iter, "V", "%v_smem_base");
    ptx.push_str(&format!("V2_V_PREPASS_SKIP_{}:\n", q_tile_iter));
}

/// Emit cooperative HBM writes of post-RoPE Q/K/V activations for the
/// Tier C backward pass.
///
/// Fires immediately after `emit_rope_epilogue` when
/// `CshaExtras::save_activations_for_backward=true`.  Emits:
///
/// 1. `bar.sync 0` fence — ensures the attention body has not yet begun
///    consuming the SMEM Q/K/V tiles when the save path reads them.
/// 2. Per-row cooperative store of `Q_proj` from the Q SMEM tile to HBM
///    at `q_proj_ptr + (batch*heads*seq + head*seq + (q_start+warp_row))
///    * head_dim + lane_col`.
/// 3. Same for `K_proj` (from k_smem) and `V_proj` (from v_smem).
///
/// Each lane owns `head_dim/32` columns (same contract as
/// `emit_warp_per_row_sweep`).  Kernel params `q_proj_ptr`, `k_proj_ptr`,
/// `v_proj_ptr` are expected in the kernel's param block (added alongside
/// this hook in the prelude); null pointers on any of them skip the
/// corresponding save sweep without error.
///
/// Null-gated invariants:
/// - `save_activations_for_backward=false` (or `csha=None`) → no-op comment.
/// - Per-tensor pointer null-guards allow partial-save configs (e.g. only Q
///   saved during early bring-up).
pub fn emit_save_activations(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    emit_save_activations_subset(ptx, config, q_tile_iter, SaveSet::All);
}

/// Which of the Q/K/V saves to emit. `QK` runs in the S-pass (while Q and K
/// SMEM tiles are live and post-RoPE); `V` runs after the V pre-pass (V SMEM
/// aliases K during the S-pass, so an early V save would store K's bytes).
#[derive(Copy, Clone)]
/// Which post-RoPE activations `emit_save_activations_subset` writes to HBM.
///
/// The `Q` / `K` / `V` variants each save a single tile and are used by the
/// asymmetric-tile fused path (`block_q != block_kv`) where Q, K, V all
/// have independent row counts and are written from distinct orchestrator
/// loops.  `QK` and `All` are retained for callers in the symmetric fused
/// path and the non-fused path that save multiple tiles at a single site.
pub enum SaveSet { All, QK, V, Q, K }

pub fn emit_save_activations_subset(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    set: SaveSet,
) {
    let save = config
        .csha
        .as_ref()
        .is_some_and(|c| c.save_activations_for_backward);
    if !save {
        ptx.push_str("    // CSHA Tier C save_activations: save_activations=false, no emission\n");
        return;
    }
    let head_dim = config.head_dim as u32;
    let slices_per_lane = (head_dim / 32).max(1);
    let fused = config
        .csha
        .as_ref()
        .is_some_and(|c| c.fused_projections);

    ptx.push_str(&format!(
        "    // CSHA Tier C: save post-RoPE activations (q_tile_iter={}, slices/lane={}, set={:?})\n",
        q_tile_iter, slices_per_lane,
        match set {
            SaveSet::All => "All",
            SaveSet::QK => "QK",
            SaveSet::V => "V",
            SaveSet::Q => "Q",
            SaveSet::K => "K",
        }
    ));

    // Non-fused path: initialise %q_smem_base / %k_smem_base / %v_smem_base.
    // In the fused-projections path these were already initialised by
    // `emit_matmul_projection` (csha_hooks.rs:215).  In the non-fused path
    // the Q tile lives at q_offset(=0) and the K/V tiles alias kv_offset
    // (K is loaded by `emit_k_tile_load`, V by `emit_v_tile_load` — both
    // write into the kv_offset slot).  Initialise here so the save sweep
    // below can read from %q/k/v_smem_base without hitting an undeclared-
    // register ptxas error.  Registers are guaranteed to be declared by
    // the matching widened gate in `prelude.rs:needs_qkv_smem_base`.
    if !fused {
        let kv_off = crate::flash_attention_v2::smem_layout::kv_offset(config);
        ptx.push_str("    // save_activations && !fused_projections: init SMEM base registers\n");
        ptx.push_str("    mov.u64 %q_smem_base, %shmem_base;              // Q tile at q_offset(=0)\n");
        ptx.push_str(&format!(
            "    add.u64 %k_smem_base, %shmem_base, {}; // K tile at kv_offset\n",
            kv_off
        ));
        ptx.push_str(&format!(
            "    add.u64 %v_smem_base, %shmem_base, {}; // V tile at kv_offset (aliases K)\n",
            kv_off
        ));
    }

    // Fence: ensure no attention-body reads of Q/K/V SMEM have commenced.
    ptx.push_str("    bar.sync 0;  // FENCE: save path must see final SMEM tiles\n");

    let all = [
        ("Q", "q_proj_ptr", "%q_smem_base"),
        ("K", "k_proj_ptr", "%k_smem_base"),
        ("V", "v_proj_ptr", "%v_smem_base"),
    ];
    let entries: &[(&str, &str, &str)] = match set {
        SaveSet::All => &all,
        SaveSet::QK => &all[0..2],
        SaveSet::Q  => &all[0..1],
        SaveSet::K  => &all[1..2],
        SaveSet::V  => &all[2..3],
    };
    for &(label, ptr_name, smem_base) in entries {
        ptx.push_str(&format!(
            "    // -- Save {label} activation to HBM via [{ptr_name}] --\n"
        ));
        ptx.push_str(&format!("    ld.param.u64 %rd_save_base, [{ptr_name}];\n"));
        ptx.push_str("    setp.eq.u64 %p_save_null, %rd_save_base, 0;\n");
        ptx.push_str(&format!(
            "    @%p_save_null bra V2_CSHA_SAVE_{label}_SKIP_{q_tile_iter};\n"
        ));

        // row_idx = batch_idx*heads*seq + head_idx*seq + (q_start + warp_row)
        ptx.push_str(&format!(
            "    add.u32 %r_save_wrow, %warp_id, {}; // warp_row = warp_id + iter*4\n",
            q_tile_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd_save_wrow, %r_save_wrow;\n");
        // batch*heads
        ptx.push_str("    mul.lo.u64 %rd_save_off, %batch_idx, %rd5;\n");
        // + head_idx
        ptx.push_str("    add.u64 %rd_save_off, %rd_save_off, %head_idx;\n");
        // * seq_len
        ptx.push_str("    mul.lo.u64 %rd_save_off, %rd_save_off, %rd6;\n");
        // + q_start + warp_row
        ptx.push_str("    add.u64 %rd_save_off, %rd_save_off, %q_start;\n");
        ptx.push_str("    add.u64 %rd_save_off, %rd_save_off, %rd_save_wrow;\n");
        // * head_dim
        ptx.push_str("    mul.lo.u64 %rd_save_off, %rd_save_off, %rd7;\n");

        // Per-slice: add lane*slices_per_lane + slice, load from SMEM, store to HBM (f16).
        for slice in 0..slices_per_lane {
            // col = lane * slices_per_lane + slice
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
            // hbm_addr = base + (row_off + col) * 2 (f16)
            ptx.push_str("    add.u64 %rd_save_elem, %rd_save_off, %rd_save_col;\n");
            ptx.push_str("    shl.b64 %rd_save_elem, %rd_save_elem, 1; // * 2 (f16)\n");
            ptx.push_str("    add.u64 %rd_save_elem, %rd_save_base, %rd_save_elem;\n");
            // smem_addr = smem_base + warp_row*(head_dim*2) + col*2
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_save_smem, %rd_save_wrow, {};\n",
                head_dim * 2
            ));
            ptx.push_str(&format!(
                "    add.u64 %rd_save_smem, {smem_base}, %rd_save_smem;\n"
            ));
            ptx.push_str("    shl.b64 %rd_save_colb, %rd_save_col, 1;\n");
            ptx.push_str("    add.u64 %rd_save_smem, %rd_save_smem, %rd_save_colb;\n");
            ptx.push_str("    ld.shared.b16 %h_save_v, [%rd_save_smem];\n");
            ptx.push_str(&format!(
                "    st.global.b16 [%rd_save_elem], %h_save_v;  // {ptr_name} write\n"
            ));
        }
        ptx.push_str(&format!("V2_CSHA_SAVE_{label}_SKIP_{q_tile_iter}:\n"));
    }
}

/// Save %row_max and %row_sum to the softmax-state SMEM save area.
///
/// Each (warp_id, q_tile_iter) pair has its own 2×f32 slot to avoid races.
/// Layout: `softmax_save_offset + (warp_id * iters + q_tile_iter) * 8` bytes.
///
/// Runs when `fused_projections=true` (the S+PV split design) OR when
/// `save_activations_for_backward=true` (Tier C training path). The SMEM
/// save is only needed for the fused path's S+PV split; the HBM save is
/// needed for the backward kernel regardless of fused_projections.
pub fn emit_save_softmax_state(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let diag = std::env::var("NSL_CSHA_DUMP_SAVE_STATE")
        .ok()
        .unwrap_or_default();
    emit_save_softmax_state_with_diag(ptx, config, q_tile_iter, &diag);
}

/// J-A2 diagnostic-aware emitter. Callers usually use the env-reading wrapper
/// `emit_save_softmax_state`; tests call this directly to avoid mucking with
/// process env. `diag` values: `""` (real %row_max/%row_sum), `"wrow"`,
/// `"wid"`, or `"qstart"`.
pub(crate) fn emit_save_softmax_state_with_diag(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    diag: &str,
) {
    let fused = config.csha.as_ref().is_some_and(|c| c.fused_projections);
    let save = config
        .csha
        .as_ref()
        .is_some_and(|c| c.save_activations_for_backward);
    if !fused && !save {
        return;
    }
    if fused {
        let save_base = crate::flash_attention_v2::smem_layout::softmax_save_offset(config);
        let iters = (config.block_q as u32).div_ceil(4);
        // Slot base for this (warp_id, q_tile_iter): save_base + (warp_id * iters + q_tile_iter) * 8.
        // Computed at runtime using %warp_id.
        // Static part: save_base + q_tile_iter * 8 + warp_id * iters * 8.

        ptx.push_str(&format!(
            "    // Save softmax state for q_tile_iter={} (warp-indexed)\n",
            q_tile_iter
        ));
        // Compute warp offset: warp_id * iters * 8 (bytes).
        ptx.push_str("    cvt.u64.u32 %rd_wt_off, %warp_id;\n");
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_wt_off, %rd_wt_off, {}; // warp_id * iters * 8\n",
            iters * 8
        ));
        ptx.push_str(&format!(
            "    add.u64 %rd_wt_dst, %shmem_base, {}; // save_base + q_tile_iter*8\n",
            save_base + q_tile_iter * 8
        ));
        ptx.push_str("    add.u64 %rd_wt_dst, %rd_wt_dst, %rd_wt_off;\n");
        // Save row_max and row_sum at consecutive 4-byte slots.
        ptx.push_str("    st.shared.f32 [%rd_wt_dst], %row_max;\n");
        ptx.push_str("    add.u64 %rd_wt_dst, %rd_wt_dst, 4;\n");
        ptx.push_str("    st.shared.f32 [%rd_wt_dst], %row_sum;\n");
    }

    // Tier C: persist row_max/row_sum to HBM so the backward kernel can
    // recompute P = exp(S - row_max) / row_sum identically to forward.
    // Layout: [batch, heads, seq] f32, row_major. Stored by lane 0 only
    // (all 32 lanes share identical values post-butterfly-reduction).
    // Fires for both fused and non-fused paths when save_activations_for_backward=true.
    if save {
        // J-A2 + J-A3 diagnostic: when `diag` is non-empty, replace the real
        // `%row_max` / `%row_sum` source operands with known-provenance
        // scalars. J-A2 modes (save-site origin) distinguish "wrong value
        // computed by softmax" from "wrong address in save path"; J-A3 modes
        // (softmax-captured scratch regs populated in softmax.rs) distinguish
        // "softmax computed wrong value" from "ptxas corrupted %row_max
        // between softmax and save."
        //
        // J-A2 save-site modes:
        //   - "wrow"   → both slots store f32(warp_row). Expect [0..N-1].
        //   - "wid"    → row_max = f32(warp_id), row_sum = f32(lane). Verifies
        //                the lane-0 gate and warp identity (row_sum reads 0.0
        //                everywhere since only lane 0 stores).
        //   - "qstart" → row_max = f32(q_start.lo), row_sum = f32(warp_row).
        //
        // J-A3 softmax-internal modes (softmax.rs populates these regs):
        //   - "fmax"   → both slots store %f_sdx_fmax (post-butterfly max,
        //                BEFORE online update with %row_max). If fmax shows
        //                the expected per-row max but default row_max reads
        //                1e-30, ptxas is corrupting the reg between the
        //                softmax update and the save.
        //   - "newmax" → both slots store %f_sdx_nmax (post-online-update,
        //                the value softmax commits to %row_max this tile).
        //   - "fsum"   → both slots store %f_sdx_fsum (post-butterfly sum,
        //                BEFORE online add to %row_sum).
        //
        //   - anything else or empty → real %row_max / %row_sum.
        ptx.push_str(&format!(
            "    // Tier C: persist row_max/row_sum to HBM (q_tile_iter={})\n",
            q_tile_iter
        ));
        if !diag.is_empty() {
            ptx.push_str(&format!(
                "    // J-A2 diagnostic: NSL_CSHA_DUMP_SAVE_STATE={}\n",
                diag
            ));
        }
        // `%p_save_null` is the "skip-this-lane" predicate — TRUE when this
        // lane should NOT save. We want only lane 0 of each warp to write,
        // so skip-predicate = (lane != 0). Composed with the null-ptr
        // predicate via OR at each store site.
        ptx.push_str("    setp.ne.u32 %p_save_null, %lane, 0;\n");
        // warp_row = warp_id + iter*4
        ptx.push_str(&format!(
            "    add.u32 %r_save_wrow, %warp_id, {}; // warp_row = warp_id + iter*4\n",
            q_tile_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd_save_wrow, %r_save_wrow;\n");
        // row_idx = batch_idx*(heads*seq) + head_idx*seq + (q_start + warp_row)
        ptx.push_str("    mul.lo.u64 %rd_save_off, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd_save_off, %rd_save_off, %head_idx;\n");
        ptx.push_str("    mul.lo.u64 %rd_save_off, %rd_save_off, %rd6;\n");
        ptx.push_str("    add.u64 %rd_save_off, %rd_save_off, %q_start;\n");
        ptx.push_str("    add.u64 %rd_save_off, %rd_save_off, %rd_save_wrow;\n");
        // * 4 bytes (f32)
        ptx.push_str("    shl.b64 %rd_save_off, %rd_save_off, 2;\n");

        // Pick diagnostic source operands per slot if `diag` is set.
        let (rm_setup, rm_src): (&str, &str) = match diag {
            "wrow" => (
                "    cvt.rn.f32.u32 %f_diag, %r_save_wrow;\n",
                "%f_diag",
            ),
            "wid" => (
                "    cvt.rn.f32.u32 %f_diag, %warp_id;\n",
                "%f_diag",
            ),
            "qstart" => (
                "    cvt.u32.u64 %r_save_qlo, %q_start;\n    cvt.rn.f32.u32 %f_diag, %r_save_qlo;\n",
                "%f_diag",
            ),
            // J-A3: read from softmax-captured scratch regs (populated in softmax.rs).
            "fmax"   => ("", "%f_sdx_fmax"),
            "newmax" => ("", "%f_sdx_nmax"),
            "fsum"   => ("", "%f_sdx_fsum"),
            _ => ("", "%row_max"),
        };

        // J-A4: under `direct_*` modes softmax.rs already wrote the row_max
        // buffer directly from within softmax (bypassing scratch regs). Skip
        // this save-site's row_max write to avoid overwriting the direct
        // store with the stale-register value we're trying to sidestep.
        let skip_rowmax_write = diag.starts_with("direct_");

        // row_max_ptr
        if !skip_rowmax_write {
            ptx.push_str("    ld.param.u64 %rd_save_base, [row_max_ptr];\n");
            ptx.push_str("    add.u64 %rd_save_elem, %rd_save_base, %rd_save_off;\n");
            ptx.push_str("    setp.eq.u64 %p_rowmax_null, %rd_save_base, 0;\n");
            ptx.push_str("    or.pred %p_skip_rm, %p_rowmax_null, %p_save_null;\n");
            ptx.push_str(rm_setup);
            ptx.push_str(&format!(
                "    @!%p_skip_rm st.global.f32 [%rd_save_elem], {};  // row_max_ptr write\n",
                rm_src
            ));
        } else {
            ptx.push_str(
                "    // row_max write suppressed - direct_* mode wrote it from softmax\n",
            );
        }
        let _ = q_tile_iter;

        let (rs_setup, rs_src): (&str, &str) = match diag {
            "wrow" => (
                "    cvt.rn.f32.u32 %f_diag, %r_save_wrow;\n",
                "%f_diag",
            ),
            "wid" => (
                "    cvt.rn.f32.u32 %f_diag, %lane;\n",
                "%f_diag",
            ),
            "qstart" => (
                "    cvt.rn.f32.u32 %f_diag, %r_save_wrow;\n",
                "%f_diag",
            ),
            // J-A3: both slots store the same softmax-captured scratch — makes
            // it unambiguous when reading back which register we're inspecting.
            "fmax"   => ("", "%f_sdx_fmax"),
            "newmax" => ("", "%f_sdx_nmax"),
            "fsum"   => ("", "%f_sdx_fsum"),
            _ => ("", "%row_sum"),
        };

        // row_sum_ptr
        ptx.push_str("    ld.param.u64 %rd_save_base, [row_sum_ptr];\n");
        ptx.push_str("    add.u64 %rd_save_elem, %rd_save_base, %rd_save_off;\n");
        ptx.push_str("    setp.eq.u64 %p_rowsum_null, %rd_save_base, 0;\n");
        ptx.push_str("    or.pred %p_skip_rs, %p_rowsum_null, %p_save_null;\n");
        ptx.push_str(rs_setup);
        ptx.push_str(&format!(
            "    @!%p_skip_rs st.global.f32 [%rd_save_elem], {};  // row_sum_ptr write\n",
            rs_src
        ));
    }
}

/// Restore %row_max and %row_sum from the softmax-state SMEM save area.
///
/// Symmetric to `emit_save_softmax_state`.  Called at the start of the PV-accum
/// pass for each q_tile_iter.
pub fn emit_restore_softmax_state(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if !config.csha.as_ref().is_some_and(|c| c.fused_projections) {
        return;
    }
    let save_base = crate::flash_attention_v2::smem_layout::softmax_save_offset(config);
    let iters = (config.block_q as u32).div_ceil(4);

    ptx.push_str(&format!(
        "    // Restore softmax state for q_tile_iter={} (warp-indexed)\n",
        q_tile_iter
    ));
    ptx.push_str("    cvt.u64.u32 %rd_wt_off, %warp_id;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd_wt_off, %rd_wt_off, {};\n",
        iters * 8
    ));
    ptx.push_str(&format!(
        "    add.u64 %rd_wt_dst, %shmem_base, {};\n",
        save_base + q_tile_iter * 8
    ));
    ptx.push_str("    add.u64 %rd_wt_dst, %rd_wt_dst, %rd_wt_off;\n");
    ptx.push_str("    ld.shared.f32 %row_max, [%rd_wt_dst];\n");
    ptx.push_str("    add.u64 %rd_wt_dst, %rd_wt_dst, 4;\n");
    ptx.push_str("    ld.shared.f32 %row_sum, [%rd_wt_dst];\n");
}

/// Emit the §A.2.5 Wo output projection (post-attention epilogue).
///
/// # SMEM budget status (spec R2 decision point)
///
/// A5.0 measured 99328 bytes for the worst-case matrix config (block_q=64,
/// block_kv=64, head_dim=64, d_model=128, fused_projections=true,
/// fused_output_proj=true) — 2.02× the 48 KB budget.  Inline fusion is
/// therefore NOT viable.
///
/// Per spec R2: this function is a **dispatch stub**.  It emits the Wo
/// pointer null-check so the kernel can signal "Wo ready" and sets up
/// skip labels, but does NOT perform the matrix multiply inline.  The
/// actual `O @ Wo` + residual add is delegated to a separate follow-up
/// kernel call emitted by the surrounding codegen after the FA kernel
/// returns.
///
/// When `fused_output_proj=false` (or `csha=None`) the function is a
/// complete no-op (emits a comment only).
pub fn emit_output_projection(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let fused_output = config.csha.as_ref().is_some_and(|c| c.fused_output_proj);
    if !fused_output {
        ptx.push_str("    // CSHA A5: fused_output=false, no emission\n");
        return;
    }

    let d_model = config.csha.as_ref().map_or(0, |c| c.d_model);

    // Keep PTX comments ASCII-only — ptxas rejects UTF-8 bytes like em-dash.
    ptx.push_str("    // CSHA A5: Wo output projection stub (spec R2 -- separate kernel path)\n");
    ptx.push_str(&format!(
        "    // d_model={d_model}, q_tile_iter={q_tile_iter}\n"
    ));

    // Wo pointer null-check. If Wo=null the kernel has no output projection
    // to perform; skip all Wo/residual logic.
    // Uses named registers declared in prelude (%rd_wo_ptr, %p_wo_null).
    ptx.push_str("    ld.param.u64 %rd_wo_ptr, [csha_wo_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_wo_null, %rd_wo_ptr, 0;\n");
    ptx.push_str(&format!(
        "    @%p_wo_null bra V2_CSHA_WO_SKIP_{};\n",
        q_tile_iter
    ));

    // x_ptr null-check for residual add. If x_ptr=null, residual is skipped.
    // Uses %rd52 which is already declared in the 64-register pool and reused
    // here post-prologue (prologue is complete at this phase).
    ptx.push_str("    ld.param.u64 %rd52, [csha_x_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_x_null, %rd52, 0;\n");
    ptx.push_str(&format!(
        "    @%p_x_null bra V2_CSHA_WO_SKIP_RESIDUAL_{};\n",
        q_tile_iter
    ));

    // Wo loop dispatch point. In the separate-kernel path this label marks
    // where a follow-up kernel call takes over.  The label is retained so
    // downstream tests can verify orchestration ordering.
    ptx.push_str(&format!("V2_CSHA_WO_LOOP_{}:\n", q_tile_iter));
    ptx.push_str("    // Wo @ O and residual add delegated to follow-up kernel (spec R2)\n");

    // Residual skip label (inline path not implemented; used by null-x guard).
    ptx.push_str(&format!("V2_CSHA_WO_SKIP_RESIDUAL_{}:\n", q_tile_iter));

    ptx.push_str(&format!("V2_CSHA_WO_SKIP_{}:\n", q_tile_iter));
}

/// Emit the §A.2.4 RoPE Q/K-rotation epilogue.
///
/// # Placement (pre-attention, immediately after `emit_matmul_projection`)
///
/// This hook fires **immediately after `emit_matmul_projection`** and
/// **before** the Q-load / S-compute / softmax / PV-accumulate body.
/// `emit_matmul_projection` writes projected Q/K/V fragments into SMEM
/// tiles; this hook then rotates the Q and K tiles in-place so that the
/// subsequent `QK^T` computation (in `s_compute`) consumes already-rotated
/// queries and keys — exactly matching standard pre-attention RoPE semantics
/// (`RoPE(Q); RoPE(K); S = Q @ K^T / sqrt(d); softmax; O = P @ V`).
///
/// This mirrors v1's equivalent: `emit_csha_rope_epilogue` in
/// `flash_attention.rs` is called after `emit_csha_matmul_projection` and
/// before `emit_q_tile_load`.  The v2 orchestrator (`mod.rs`) follows the
/// same ordering.
///
/// Null-guarded on `cos_ptr` AND `sin_ptr` — if either is zero the entire
/// rotation body is skipped.  Only emits when `rope_q=true` AND
/// `csha.is_some()`.  V is never rotated (standard attention).
pub fn emit_rope_epilogue(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if config.csha.is_none() || !config.rope_q {
        ptx.push_str("    // CSHA A.2.4 RoPE epilogue: rope_q=false, no emission\n");
        return;
    }

    // emit_rope_pair_sweep implements RopeStyle::Adjacent (GPT-NeoX / GPT-J layout):
    //   pair i rotates (x[2i], x[2i+1]).
    // RopeStyle::HalfSplit (LLaMA / Qwen layout: x[i] paired with x[i+head_dim/2])
    // is NOT implemented here.  If you need HalfSplit, add a separate sweep variant.
    assert!(
        matches!(config.rope_style, RopeStyle::Adjacent),
        "emit_rope_pair_sweep only implements RopeStyle::Adjacent; got {:?}",
        config.rope_style
    );

    let block_q  = config.block_q  as u32;
    let head_dim = config.head_dim as u32;
    let half_dim = head_dim / 2;
    // Each of 128 threads covers ceil(block_q * half_dim / 128) pairs.
    let total_pairs = block_q * half_dim;
    let pairs_per_lane = total_pairs.div_ceil(128);

    ptx.push_str(&format!(
        "    // CSHA A.2.4: RoPE Q/K rotation epilogue (q_tile_iter={}, block_q={}, head_dim={}, pairs_per_lane={})\n",
        q_tile_iter, block_q, head_dim, pairs_per_lane
    ));

    // Null-guard: skip if either cos_ptr or sin_ptr is zero.
    ptx.push_str("    ld.param.u64 %rd_rope_cos, [cos_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_rope_sin, [sin_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_rope_cos_null, %rd_rope_cos, 0;\n");
    ptx.push_str("    setp.eq.u64 %p_rope_sin_null, %rd_rope_sin, 0;\n");
    ptx.push_str("    or.pred %p_rope_skip, %p_rope_cos_null, %p_rope_sin_null;\n");
    ptx.push_str(&format!(
        "    @%p_rope_skip bra V2_CSHA_ROPE_SKIP_{};\n",
        q_tile_iter
    ));

    // Rotate Q tile in-place.  K rotation for the fused_projections path
    // is handled by `emit_rope_k_epilogue`, called ONCE after the K
    // pre-pass completes (before the S-compute loop).
    emit_rope_pair_sweep(
        ptx,
        config,
        q_tile_iter,
        "Q",
        "%q_smem_base",
        block_q,
        head_dim,
        half_dim,
        pairs_per_lane,
    );

    ptx.push_str(&format!("V2_CSHA_ROPE_SKIP_{}:\n", q_tile_iter));
    ptx.push_str("    bar.sync 0;  // FENCE: RoPE rotation writes visible to all threads\n");
}

/// Rotate the full K SMEM tile in-place using Adjacent RoPE.
///
/// Called ONCE after the K pre-pass completes and before the S-compute loop
/// in the fused_projections path.  The K pre-pass writes `Wk * x_norm` into
/// `%k_smem_base` (kv_offset) but does NOT apply RoPE, so without this call
/// the S-compute pass would compute `QK^T` with rotated Q but unrotated K,
/// breaking attention semantics and all backward gradients.
///
/// Uses the same rotation math as `emit_rope_pair_sweep` (Adjacent layout):
///   new_k[2i]   = k[2i]*cos[i] - k[2i+1]*sin[i]
///   new_k[2i+1] = k[2i]*sin[i] + k[2i+1]*cos[i]
///
/// Null-guarded on `cos_ptr` AND `sin_ptr`.  Only emits when `rope_q=true`
/// AND `csha.fused_projections=true`.
pub fn emit_rope_k_epilogue(ptx: &mut String, config: &FlashAttentionConfig) {
    if config.csha.is_none() || !config.rope_q {
        ptx.push_str("    // CSHA RoPE K epilogue: rope_q=false or no CSHA, skip\n");
        return;
    }
    if !config.csha.as_ref().is_some_and(|c| c.fused_projections) {
        ptx.push_str("    // CSHA RoPE K epilogue: fused_projections=false, skip\n");
        return;
    }

    assert!(
        matches!(config.rope_style, RopeStyle::Adjacent),
        "emit_rope_k_epilogue only implements RopeStyle::Adjacent; got {:?}",
        config.rope_style
    );

    let block_kv = config.block_kv as u32;
    let head_dim = config.head_dim as u32;
    let half_dim = head_dim / 2;
    let total_pairs = block_kv * half_dim;
    let pairs_per_lane = total_pairs.div_ceil(128);

    ptx.push_str(&format!(
        "    // CSHA RoPE K rotation (fused path): block_kv={}, head_dim={}, pairs_per_lane={}\n",
        block_kv, head_dim, pairs_per_lane
    ));

    // Null-guard: skip if either cos_ptr or sin_ptr is zero.
    ptx.push_str("    ld.param.u64 %rd_rope_cos, [cos_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_rope_sin, [sin_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p_rope_cos_null, %rd_rope_cos, 0;\n");
    ptx.push_str("    setp.eq.u64 %p_rope_sin_null, %rd_rope_sin, 0;\n");
    ptx.push_str("    or.pred %p_rope_skip, %p_rope_cos_null, %p_rope_sin_null;\n");
    ptx.push_str("    @%p_rope_skip bra V2_CSHA_ROPE_K_FUSED_SKIP;\n");

    emit_rope_pair_sweep(
        ptx,
        config,
        0, // not per-q_iter; runs once for the whole K tile
        "K",
        "%k_smem_base",
        block_kv,
        head_dim,
        half_dim,
        pairs_per_lane,
    );

    ptx.push_str("V2_CSHA_ROPE_K_FUSED_SKIP:\n");
    ptx.push_str("    bar.sync 0;  // FENCE: K RoPE rotation writes visible to all threads\n");
}

/// Emit one cooperative pair-loop sweep that applies RoPE to a single
/// SMEM tile (Q or K).  Each lane handles `pairs_per_lane` consecutive
/// (row, dim_pair) pairs in the (block_q × head_dim/2) space.
///
/// Rotation math per pair:
///   cos, sin  = cos_ptr[row * half_dim + dim_pair], sin_ptr[same]  (f16→f32)
///   x0        = tile[row, 2*dim_pair]     (f16→f32)
///   x1        = tile[row, 2*dim_pair + 1] (f16→f32)
///   new_x0    = x0*cos - x1*sin           (2× fma)
///   new_x1    = x0*sin + x1*cos           (2× fma)
///   store f32→f16, write back to SMEM
///
/// All SMEM addresses are precomputed into u64 registers before use
/// (bracket-register-arithmetic rejected by ptxas).
///
/// PCA §4.3 RoPE-reset (T7 forward Q / T8 forward K): when
/// `config.segment_masked && config.rope_q`, the cos/sin index (cs_idx)
/// is rerouted through `%r_effective_pos_<q|k>` instead of the raw
/// tile-local row. The SMEM tile addressing (`%r_rope_smem_row_off`)
/// keeps using the tile-local `%r_rope_row` — the two registers are
/// deliberately decoupled so SMEM addressing stays correct after the
/// cs_idx reroute. effective_pos = (q_start + tile_local_row)
///                                  - smem_doc_starts[seg_ids[abs_row]].
/// Sentinel-disabled paths (segment_masked=false) are byte-stable.
#[allow(clippy::too_many_arguments)]
fn emit_rope_pair_sweep(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    tile_label: &str,    // "Q" or "K"
    smem_base_reg: &str, // "%q_smem_base" or "%k_smem_base"
    block_q: u32,
    head_dim: u32,
    half_dim: u32,
    pairs_per_lane: u32,
) {
    let tl = tile_label; // short alias for label generation
    // PCA §4.3 RoPE-reset gate: only the forward Q+K rotation sites (T7+T8)
    // emit the doc_starts lookup. tile_label drives which effective_pos
    // register name is used (Q vs K-side).
    let reset_active = config.segment_masked && config.rope_q;
    let effective_pos_reg = match tile_label {
        "Q" => "%r_effective_pos_q",
        "K" => "%r_effective_pos_k",
        // Defensive: emit_rope_pair_sweep is only ever called with "Q" or "K"
        // from the two epilogues; an unexpected label would silently fall
        // through to the non-reset cs_idx (no semantic effect) but flag it.
        other => panic!("emit_rope_pair_sweep: unexpected tile_label {other:?}, expected Q or K"),
    };
    ptx.push_str(&format!(
        "    // A.2.4 RoPE {tl} sweep: block_q={block_q}, half_dim={half_dim}, pairs_per_lane={pairs_per_lane}\n"
    ));

    // Linear thread index within the block (tid_x = warp_id*32 + lane).
    ptx.push_str("    cvt.u32.u32 %r_rope_tid, %tid_x;\n");

    // Loop counter: start = tid_x (first pair for this lane).
    ptx.push_str("    mov.u32 %r_rope_pair_idx, %r_rope_tid;\n");

    // Total pairs constant for loop-exit predicate.
    let total_pairs = block_q * half_dim;

    ptx.push_str(&format!("V2_CSHA_ROPE_{tl}_LOOP_{q_tile_iter}:\n"));

    // Guard: if pair_idx >= total_pairs, exit loop.
    ptx.push_str(&format!(
        "    setp.ge.u32 %p_rope_done, %r_rope_pair_idx, {total_pairs};\n"
    ));
    ptx.push_str(&format!(
        "    @%p_rope_done bra V2_CSHA_ROPE_{tl}_END_{q_tile_iter};\n"
    ));

    // Decompose pair_idx into (row, dim_pair).
    //   row      = pair_idx / half_dim
    //   dim_pair = pair_idx % half_dim
    ptx.push_str(&format!(
        "    div.u32 %r_rope_row,      %r_rope_pair_idx, {half_dim};\n"
    ));
    ptx.push_str(&format!(
        "    rem.u32 %r_rope_dim_pair, %r_rope_pair_idx, {half_dim};\n"
    ));

    // PCA §4.3 RoPE-reset (T7 Q / T8 K): compute effective_pos for cs_idx.
    //   abs_row       = q_start (u64 → u32) + tile_local_row   (Q and K
    //                   share q_start in the fused path: the K pre-pass
    //                   writes K rows at q_start + warp_row, so the SMEM
    //                   tile-local row offsets match absolute positions
    //                   under the single-tile assumption that the fused
    //                   path inherits from the symmetric path.)
    //   sid           = segment_ids[abs_row]   (read from seg_smem; u16)
    //   doc_start     = smem_doc_starts[sid]   (read from smem_doc_starts; s32)
    //   effective_pos = abs_row - doc_start    (s32 result; non-negative
    //                   under the packing invariants from pca_rope::plan)
    //
    // %r_rope_row is left unchanged so the SMEM offset computation below
    // (mul.lo.u32 %r_rope_smem_row_off, %r_rope_row, head_dim*2) still
    // addresses the correct tile-local row.
    if reset_active {
        ptx.push_str(&format!(
            "    // PCA §4.3 site {site}: forward {tl} effective_pos\n",
            site = if tl == "Q" { 1 } else { 2 }
        ));
        // abs_row = (q_start narrowed to u32) + tile_local_row.
        ptx.push_str("    cvt.u32.u64 %r_abs_pos, %q_start;\n");
        ptx.push_str("    add.u32 %r_abs_pos, %r_abs_pos, %r_rope_row;\n");
        // sid = segment_ids[abs_row] from seg_smem (u16 entries).
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
        // effective_pos = abs_row - doc_start (s32; abs_row reinterpreted via mov).
        ptx.push_str(&format!(
            "    sub.s32 {effective_pos_reg}, %r_abs_pos, %r_doc_start;\n"
        ));
    }

    // cos/sin HBM address:
    //   byte_offset = (row * half_dim + dim_pair) * 2   (f16 = 2 bytes)
    //   row * half_dim + dim_pair
    //
    // When PCA §4.3 RoPE-reset is active the cs_idx is computed against
    // effective_pos (which encodes the document-relative position) instead
    // of the tile-local row, so packed-batch positions wrap back to 0 at
    // each document boundary. SMEM addressing further below still uses
    // %r_rope_row.
    let cs_row_reg = if reset_active { effective_pos_reg } else { "%r_rope_row" };
    ptx.push_str(&format!(
        "    mul.lo.u32 %r_rope_cs_off, {cs_row_reg}, {half_dim};\n"
    ));
    ptx.push_str("    add.u32 %r_rope_cs_off, %r_rope_cs_off, %r_rope_dim_pair;\n");
    ptx.push_str("    cvt.u64.u32 %rd_rope_cs_idx, %r_rope_cs_off;\n");
    ptx.push_str("    shl.b64 %rd_rope_cs_idx, %rd_rope_cs_idx, 1;  // *2 for f16\n");

    // Load cos (f16) from HBM, convert to f32.
    ptx.push_str("    add.u64 %rd_rope_addr, %rd_rope_cos, %rd_rope_cs_idx;\n");
    ptx.push_str("    ld.global.b16 %h_rope_pair, [%rd_rope_addr];\n");
    ptx.push_str("    cvt.f32.f16 %f_rope_cos, %h_rope_pair;\n");

    // Load sin (f16) from HBM, convert to f32.
    ptx.push_str("    add.u64 %rd_rope_addr, %rd_rope_sin, %rd_rope_cs_idx;\n");
    ptx.push_str("    ld.global.b16 %h_rope_pair, [%rd_rope_addr];\n");
    ptx.push_str("    cvt.f32.f16 %f_rope_sin, %h_rope_pair;\n");

    // SMEM tile addresses for x0 and x1.
    //   tile[row, col] at byte offset = (row * head_dim + col) * 2  (f16)
    //   x0 col = 2 * dim_pair
    //   x1 col = 2 * dim_pair + 1
    ptx.push_str(&format!(
        "    mul.lo.u32 %r_rope_smem_row_off, %r_rope_row, {head_dim_x2};\n",
        head_dim_x2 = head_dim * 2
    ));
    // x0: col = 2*dim_pair → byte offset = 2*dim_pair*2 = 4*dim_pair
    ptx.push_str("    shl.b32 %r_rope_x0_col, %r_rope_dim_pair, 2;  // 4*dim_pair\n");
    ptx.push_str("    add.u32 %r_rope_x0_off, %r_rope_smem_row_off, %r_rope_x0_col;\n");
    // x1: col = 2*dim_pair+1 → byte offset = (2*dim_pair+1)*2 = 4*dim_pair+2
    ptx.push_str("    add.u32 %r_rope_x1_off, %r_rope_x0_off, 2;    // +2 bytes\n");

    // Precompute full SMEM addresses for x0 and x1 into u64 regs.
    ptx.push_str("    cvt.u64.u32 %rd_rope_x0_off, %r_rope_x0_off;\n");
    ptx.push_str("    cvt.u64.u32 %rd_rope_x1_off, %r_rope_x1_off;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd_rope_addr, {smem_base_reg}, %rd_rope_x0_off;\n"
    ));
    ptx.push_str("    ld.shared.b16 %h_rope_pair, [%rd_rope_addr];\n");
    ptx.push_str("    cvt.f32.f16 %f_rope_x0, %h_rope_pair;\n");

    ptx.push_str(&format!(
        "    add.u64 %rd_rope_addr, {smem_base_reg}, %rd_rope_x1_off;\n"
    ));
    ptx.push_str("    ld.shared.b16 %h_rope_pair, [%rd_rope_addr];\n");
    ptx.push_str("    cvt.f32.f16 %f_rope_x1, %h_rope_pair;\n");

    // Rotation:
    //   new_x0 = x0*cos - x1*sin   →  fma.rn.f32 new_x0, x0, cos, 0
    //                                  fma.rn.f32 new_x0, -x1, sin, new_x0
    //   new_x1 = x0*sin + x1*cos   →  fma.rn.f32 new_x1, x0, sin, 0
    //                                  fma.rn.f32 new_x1, x1, cos, new_x1
    ptx.push_str("    mov.f32 %f_rope_y0, 0f00000000;\n");
    ptx.push_str("    fma.rn.f32 %f_rope_y0, %f_rope_x0, %f_rope_cos, %f_rope_y0;\n");
    ptx.push_str("    neg.f32 %f_rope_neg_x1, %f_rope_x1;\n");
    ptx.push_str("    fma.rn.f32 %f_rope_y0, %f_rope_neg_x1, %f_rope_sin, %f_rope_y0;\n");

    ptx.push_str("    mov.f32 %f_rope_y1, 0f00000000;\n");
    ptx.push_str("    fma.rn.f32 %f_rope_y1, %f_rope_x0, %f_rope_sin, %f_rope_y1;\n");
    ptx.push_str("    fma.rn.f32 %f_rope_y1, %f_rope_x1, %f_rope_cos, %f_rope_y1;\n");

    // Convert new_x0, new_x1 to f16 and store back to SMEM.
    ptx.push_str("    cvt.rn.f16.f32 %h_rope_y0, %f_rope_y0;\n");
    ptx.push_str("    cvt.rn.f16.f32 %h_rope_y1, %f_rope_y1;\n");

    // Store x0 back.
    ptx.push_str(&format!(
        "    add.u64 %rd_rope_addr, {smem_base_reg}, %rd_rope_x0_off;\n"
    ));
    ptx.push_str("    st.shared.b16 [%rd_rope_addr], %h_rope_y0;\n");

    // Store x1 back.
    ptx.push_str(&format!(
        "    add.u64 %rd_rope_addr, {smem_base_reg}, %rd_rope_x1_off;\n"
    ));
    ptx.push_str("    st.shared.b16 [%rd_rope_addr], %h_rope_y1;\n");

    // Advance by 128 (one full warp-block stride).
    ptx.push_str("    add.u32 %r_rope_pair_idx, %r_rope_pair_idx, 128;\n");
    ptx.push_str(&format!(
        "    bra V2_CSHA_ROPE_{tl}_LOOP_{q_tile_iter};\n"
    ));

    ptx.push_str(&format!("V2_CSHA_ROPE_{tl}_END_{q_tile_iter}:\n"));
    ptx.push_str("    bar.sync 0;  // FENCE: RoPE tile writes complete\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn cfg_with_projections() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras { fused_projections: true, d_model: 128, ..CshaExtras::default() }),
            checkpoint: None,
        }
    }

    /// Base config for A4 RoPE tests.  rope_q=true, csha set with fused_projections.
    fn base_cfg_for_rope_test() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: true,
            rope_style: RopeStyle::Adjacent,  // emit_rope_pair_sweep implements Adjacent (x[2i], x[2i+1])
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras { fused_projections: true, d_model: 128, ..CshaExtras::default() }),
            checkpoint: None,
        }
    }

    #[test]
    fn a4_rope_epilogue_emits_q_and_k_rotation_sweeps() {
        let cfg = base_cfg_for_rope_test();
        let mut ptx = String::new();
        emit_rope_epilogue(&mut ptx, &cfg, 0);

        assert!(ptx.contains("ld.param.u64 %rd_rope_cos, [cos_ptr];"), "cos_ptr load missing");
        assert!(ptx.contains("ld.param.u64 %rd_rope_sin, [sin_ptr];"), "sin_ptr load missing");
        // emit_rope_epilogue only rotates Q; K rotation is in emit_rope_k_epilogue (called once after K pre-pass).
        assert!(ptx.contains("V2_CSHA_ROPE_Q_LOOP_0:"), "Q rotation loop label missing");
        assert!(!ptx.contains("V2_CSHA_ROPE_K_LOOP_0:"), "K rotation belongs in emit_rope_k_epilogue, not here");
        assert!(!ptx.contains("V2_CSHA_ROPE_V_LOOP"), "V must not be rotated");
        // 4 fma.rn.f32 per pair (2 for new_x0, 2 for new_x1) for Q sweep only
        assert!(
            ptx.matches("fma.rn.f32").count() >= 2,
            "expected at least 2 fma.rn.f32, got {}",
            ptx.matches("fma.rn.f32").count()
        );
        assert!(ptx.contains("cvt.rn.f16.f32"), "f16 conversion for store missing");
    }

    #[test]
    fn a4_rope_epilogue_skipped_when_rope_q_false() {
        let mut cfg = base_cfg_for_rope_test();
        cfg.rope_q = false;
        let mut ptx = String::new();
        emit_rope_epilogue(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("rope_q=false, no emission") || ptx.is_empty(),
            "expected no-emit comment or empty string, got: {ptx}"
        );
        assert!(!ptx.contains("V2_CSHA_ROPE_Q_LOOP"));
    }

    #[test]
    fn a4_rope_epilogue_label_uniqueness_across_q_tile_iters() {
        let mut cfg = base_cfg_for_rope_test();
        cfg.block_q = 64;
        let mut ptx = String::new();
        emit_rope_epilogue(&mut ptx, &cfg, 0);
        emit_rope_epilogue(&mut ptx, &cfg, 1);
        // Q labels must be unique across q_tile_iters.
        assert!(ptx.contains("V2_CSHA_ROPE_Q_LOOP_0:"));
        assert!(ptx.contains("V2_CSHA_ROPE_Q_LOOP_1:"));
        // K rotation is in emit_rope_k_epilogue, not here.
        assert!(!ptx.contains("V2_CSHA_ROPE_K_LOOP_0:"), "K rotation belongs in emit_rope_k_epilogue");
        assert!(!ptx.contains("V2_CSHA_ROPE_K_LOOP_1:"), "K rotation belongs in emit_rope_k_epilogue");
    }

    #[test]
    fn a4_rope_k_epilogue_emits_k_rotation_for_fused_path() {
        let cfg = base_cfg_for_rope_test();
        let mut ptx = String::new();
        emit_rope_k_epilogue(&mut ptx, &cfg);

        // K rotation loop must be present.
        assert!(ptx.contains("V2_CSHA_ROPE_K_LOOP_0:"), "K rotation loop label missing");
        assert!(ptx.contains("V2_CSHA_ROPE_K_FUSED_SKIP:"), "K skip label missing");
        // Null-guard on cos/sin.
        assert!(ptx.contains("ld.param.u64 %rd_rope_cos, [cos_ptr];"), "cos_ptr load missing");
        assert!(ptx.contains("ld.param.u64 %rd_rope_sin, [sin_ptr];"), "sin_ptr load missing");
        // Must NOT rotate Q or V.
        assert!(!ptx.contains("V2_CSHA_ROPE_Q_LOOP"), "Q must not be rotated in K epilogue");
        assert!(!ptx.contains("V2_CSHA_ROPE_V_LOOP"), "V must not be rotated");
        // fma instructions for rotation math.
        assert!(
            ptx.matches("fma.rn.f32").count() >= 2,
            "expected at least 2 fma.rn.f32 for K rotation"
        );
    }

    #[test]
    fn a4_rope_k_epilogue_skipped_when_not_fused() {
        let mut cfg = base_cfg_for_rope_test();
        cfg.csha = Some(CshaExtras { fused_projections: false, d_model: 128, ..CshaExtras::default() });
        let mut ptx = String::new();
        emit_rope_k_epilogue(&mut ptx, &cfg);
        assert!(!ptx.contains("V2_CSHA_ROPE_K_LOOP"), "K rotation should not emit when fused_projections=false");
    }

    #[test]
    fn a4_rope_k_epilogue_skipped_when_rope_q_false() {
        let mut cfg = base_cfg_for_rope_test();
        cfg.rope_q = false;
        let mut ptx = String::new();
        emit_rope_k_epilogue(&mut ptx, &cfg);
        assert!(!ptx.contains("V2_CSHA_ROPE_K_LOOP"), "K rotation should not emit when rope_q=false");
    }

    #[test]
    fn a3_matmul_projection_emits_q_and_k_sweeps() {
        let cfg = cfg_with_projections();
        let mut ptx = String::new();
        emit_matmul_projection(&mut ptx, &cfg, 0);

        // All three weight pointer loads present with independent null-guards.
        assert!(ptx.contains("ld.param.u64 %rd60, [csha_wq_ptr];"), "missing Wq null-check load");
        assert!(ptx.contains("ld.param.u64 %rd61, [csha_wk_ptr];"), "missing Wk null-check load");
        assert!(ptx.contains("ld.param.u64 %rd62, [csha_wv_ptr];"), "missing Wv null-check load");
        // Q and K sweeps emitted here (§9a); V is deferred to emit_v_projection_in_kv_loop
        // because K and V share kv_offset — emitting V here would overwrite K before s_compute.
        assert!(ptx.contains("V2_CSHA_PROJ_Q_LOOP_0:"), "missing Q loop label");
        assert!(ptx.contains("V2_CSHA_PROJ_K_LOOP_0:"), "missing K loop label");
        assert!(!ptx.contains("V2_CSHA_PROJ_V_LOOP_0:"),
            "V sweep must be deferred to emit_v_projection_in_kv_loop (K/V share kv_offset)");
        // No warp butterfly reduction: each lane independently accumulates its own output column.
        assert_eq!(
            ptx.matches("shfl.sync.bfly.b32").count(),
            0,
            "expected 0 shfl.sync.bfly (butterfly removed; each lane owns its output column), got {}",
            ptx.matches("shfl.sync.bfly.b32").count()
        );
        // Output store uses %rd_wt_dst scratch (smem_base is NOT mutated).
        assert!(ptx.contains("st.shared.b16 [%rd_wt_dst]"), "missing Q SMEM store via %rd_wt_dst");
    }

    #[test]
    fn a3_v_projection_deferred_to_kv_loop() {
        let cfg = cfg_with_projections();
        let mut ptx = String::new();
        emit_v_projection_in_kv_loop(&mut ptx, &cfg, 0);

        // V sweep is emitted by the deferred hook (for kv_loop placement).
        assert!(ptx.contains("V2_CSHA_PROJ_V_LOOP_0:"), "missing deferred V loop label");
        // Q and K must NOT appear here (they belong to emit_matmul_projection).
        assert!(!ptx.contains("V2_CSHA_PROJ_Q_LOOP_0:"), "Q must not appear in deferred V hook");
        assert!(!ptx.contains("V2_CSHA_PROJ_K_LOOP_0:"), "K must not appear in deferred V hook");
        // Null-guard re-loads Wv pointer.
        assert!(ptx.contains("[csha_wv_ptr]"), "missing Wv null-guard reload");
    }

    #[test]
    fn a3_label_uniqueness_across_q_tile_iters() {
        let mut cfg = cfg_with_projections();
        cfg.block_q = 64;
        let mut ptx = String::new();
        emit_matmul_projection(&mut ptx, &cfg, 0);
        emit_matmul_projection(&mut ptx, &cfg, 1);
        emit_v_projection_in_kv_loop(&mut ptx, &cfg, 0);
        emit_v_projection_in_kv_loop(&mut ptx, &cfg, 1);

        // Every label must include its q_tile_iter suffix — for all three sweeps.
        assert!(ptx.contains("V2_CSHA_PROJ_Q_LOOP_0:"), "missing iter-0 Q label");
        assert!(ptx.contains("V2_CSHA_PROJ_Q_LOOP_1:"), "missing iter-1 Q label");
        assert!(ptx.contains("V2_CSHA_PROJ_K_LOOP_0:"), "missing iter-0 K label");
        assert!(ptx.contains("V2_CSHA_PROJ_K_LOOP_1:"), "missing iter-1 K label");
        assert!(ptx.contains("V2_CSHA_PROJ_V_LOOP_0:"), "missing iter-0 V label");
        assert!(ptx.contains("V2_CSHA_PROJ_V_LOOP_1:"), "missing iter-1 V label");
        // No unsuffixed labels for any sweep
        assert!(!ptx.contains("V2_CSHA_PROJ_Q_LOOP:"), "found unsuffixed Q label");
        assert!(!ptx.contains("V2_CSHA_PROJ_K_LOOP:"), "found unsuffixed K label");
        assert!(!ptx.contains("V2_CSHA_PROJ_V_LOOP:"), "found unsuffixed V label");
    }

    /// Regression test: RoPE epilogue must appear BEFORE the attention body
    /// (S-compute / softmax / PV-accum) in the synthesized PTX.
    ///
    /// Verifies the fix that moved `emit_rope_epilogue` from post-KV-loop
    /// to immediately after `emit_matmul_projection`.  The canonical ordering
    /// is: projection → RoPE(Q,K) → Q-load → `QK^T` → softmax → `PV`.
    fn base_cfg_for_a5() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
                fused_projections: true,
                fused_output_proj: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        }
    }

    /// A5.1 — stub emits Wo null-check + x_ptr null-check + loop label +
    /// skip labels, delegating the matrix multiply to a separate kernel.
    #[test]
    fn a5_emit_output_projection_stub_for_separate_kernel_path() {
        let cfg = base_cfg_for_a5();
        let mut ptx = String::new();
        emit_output_projection(&mut ptx, &cfg, 0);

        // Wo pointer load (null-check guard) — uses named %rd_wo_ptr register
        assert!(
            ptx.contains("ld.param.u64 %rd_wo_ptr, [csha_wo_ptr];"),
            "Wo pointer load missing"
        );
        // x_ptr load for residual null-check
        assert!(
            ptx.contains("[csha_x_ptr]"),
            "x_ptr load for residual null-check missing"
        );
        // Wo sweep dispatch label (loop entry point for follow-up kernel)
        assert!(ptx.contains("V2_CSHA_WO_LOOP_0:"), "Wo loop label missing");
        // Residual skip label (for null x_ptr branch)
        assert!(
            ptx.contains("V2_CSHA_WO_SKIP_RESIDUAL_0:"),
            "residual-skip label for null x_ptr missing"
        );
        // Overall Wo skip label (for null Wo ptr branch)
        assert!(ptx.contains("V2_CSHA_WO_SKIP_0:"), "Wo overall skip label missing");
        // Spec R2 note in emitted PTX
        assert!(
            ptx.contains("separate kernel") || ptx.contains("spec R2"),
            "spec R2 / separate-kernel comment missing"
        );
    }

    /// A5.1 — when fused_output_proj=false the function emits nothing but a comment.
    #[test]
    fn a5_emit_output_projection_skipped_when_fused_output_false() {
        let mut cfg = base_cfg_for_a5();
        cfg.csha = Some(CshaExtras {
            fused_output_proj: false,
            ..CshaExtras::default()
        });

        let mut ptx = String::new();
        emit_output_projection(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("fused_output=false") || ptx.is_empty(),
            "expected no-emit marker or empty string, got: {ptx}"
        );
        assert!(!ptx.contains("V2_CSHA_WO_LOOP"), "should not emit WO loop when disabled");
    }

    /// A5.1 — null x_ptr skip label exists (tested at stub level; runtime
    /// null-path coverage lives in C3's dedicated integration test row).
    #[test]
    fn a5_null_x_ptr_skips_residual_add() {
        let cfg = base_cfg_for_a5();
        let mut ptx = String::new();
        emit_output_projection(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("V2_CSHA_WO_SKIP_RESIDUAL_0"),
            "null x_ptr skip-residual branch missing"
        );
        assert!(
            ptx.contains("p_x_null"),
            "x_ptr null predicate register missing"
        );
    }

    // ── T1.3 save_activations tests ────────────────────────────────────────

    #[test]
    fn save_activations_emits_post_rope_writes() {
        let mut cfg = base_cfg_for_rope_test();
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 128,
            ..CshaExtras::default()
        });

        let mut ptx = String::new();
        emit_save_activations(&mut ptx, &cfg, 0);

        assert!(
            ptx.contains("st.global.b16") && ptx.contains("q_proj_ptr"),
            "Q_proj save missing: {ptx}"
        );
        assert!(
            ptx.contains("st.global.b16") && ptx.contains("k_proj_ptr"),
            "K_proj save missing"
        );
        assert!(
            ptx.contains("st.global.b16") && ptx.contains("v_proj_ptr"),
            "V_proj save missing"
        );
        // Fence before cooperative save path.
        assert!(ptx.contains("bar.sync 0"), "fence before save missing");
    }

    #[test]
    fn save_activations_zero_emission_when_flag_false() {
        let mut cfg = base_cfg_for_rope_test();
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: false,
            d_model: 128,
            ..CshaExtras::default()
        });
        let mut ptx = String::new();
        emit_save_activations(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("save_activations=false, no emission") || ptx.is_empty(),
            "expected no-emit comment or empty, got: {ptx}"
        );
        assert!(!ptx.contains("st.global.b16"));
    }

    #[test]
    fn save_activations_label_uniqueness_across_q_tile_iters() {
        let mut cfg = base_cfg_for_rope_test();
        cfg.block_q = 64;
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 128,
            ..CshaExtras::default()
        });
        let mut ptx = String::new();
        emit_save_activations(&mut ptx, &cfg, 0);
        emit_save_activations(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_CSHA_SAVE_Q_SKIP_0:"));
        assert!(ptx.contains("V2_CSHA_SAVE_Q_SKIP_1:"));
        assert!(ptx.contains("V2_CSHA_SAVE_K_SKIP_0:"));
        assert!(ptx.contains("V2_CSHA_SAVE_V_SKIP_1:"));
    }

    #[test]
    fn save_softmax_state_emits_hbm_writes_when_backward_flag_set() {
        let cfg = FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state(&mut ptx, &cfg, 0);
        // SMEM save always present (fused_projections=true).
        assert!(ptx.contains("st.shared.f32 [%rd_wt_dst], %row_max;"));
        assert!(ptx.contains("st.shared.f32 [%rd_wt_dst], %row_sum;"));
        // HBM saves gated on save_activations=true.
        assert!(ptx.contains("row_max_ptr"), "row_max_ptr load missing");
        assert!(ptx.contains("row_sum_ptr"), "row_sum_ptr load missing");
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %row_max;"),
            "row_max HBM store missing"
        );
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %row_sum;"),
            "row_sum HBM store missing"
        );
    }

    /// J-A2 lane-gate invariant: the HBM save must skip lanes 1..31 so that
    /// only lane 0 writes. The predicate composition is "skip if (null_ptr OR
    /// this_lane_should_skip)"; "this_lane_should_skip" = (lane != 0), so the
    /// setp must be `setp.ne`, not `setp.eq`. Pre-Gap J this was `.eq` which
    /// inverted the gate and caused 31-way races (benign because all lanes hold
    /// identical post-butterfly values, but semantically wrong).
    #[test]
    fn save_softmax_state_lane_gate_uses_setp_ne() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state_with_diag(&mut ptx, &cfg, 0, "");
        assert!(
            ptx.contains("setp.ne.u32 %p_save_null, %lane, 0"),
            "lane-0 skip predicate must be setp.ne (got: {ptx})"
        );
        assert!(
            !ptx.contains("setp.eq.u32 %p_save_null, %lane, 0"),
            "pre-Gap J setp.eq lane gate still present — inverted semantics"
        );
    }

    /// J-A2 diag mode "wrow": both HBM slots store `f32(warp_row)` instead of
    /// the real row_max/row_sum. Used to verify that the ADDRESSING math lays
    /// out correctly (expected read-back: [0, 1, 2, ..., N-1]).
    #[test]
    fn save_softmax_state_diag_wrow_replaces_source_operand() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state_with_diag(&mut ptx, &cfg, 0, "wrow");
        assert!(
            ptx.contains("cvt.rn.f32.u32 %f_diag, %r_save_wrow"),
            "wrow mode must convert %r_save_wrow to %f_diag"
        );
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_diag;  // row_max_ptr write"),
            "row_max HBM store must source from %f_diag under wrow mode"
        );
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_diag;  // row_sum_ptr write"),
            "row_sum HBM store must source from %f_diag under wrow mode"
        );
        // Real operands must NOT appear as store sources.
        assert!(
            !ptx.contains("st.global.f32 [%rd_save_elem], %row_max;"),
            "diag mode must not store real %row_max"
        );
        assert!(
            !ptx.contains("st.global.f32 [%rd_save_elem], %row_sum;"),
            "diag mode must not store real %row_sum"
        );
    }

    /// J-A2 diag mode "wid": row_max stores `f32(warp_id)`, row_sum stores
    /// `f32(lane)`. With a correct lane-0 gate, row_sum must read back 0.0
    /// everywhere (only lane 0 stores, lane == 0).
    #[test]
    fn save_softmax_state_diag_wid_splits_sources() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state_with_diag(&mut ptx, &cfg, 0, "wid");
        assert!(
            ptx.contains("cvt.rn.f32.u32 %f_diag, %warp_id"),
            "wid mode row_max setup must convert %warp_id"
        );
        assert!(
            ptx.contains("cvt.rn.f32.u32 %f_diag, %lane"),
            "wid mode row_sum setup must convert %lane"
        );
    }

    /// J-A2 diag mode "qstart": row_max stores `f32(q_start.lo)`, row_sum
    /// stores `f32(warp_row)`. Verifies the carrier register %q_start.
    #[test]
    fn save_softmax_state_diag_qstart_dumps_carrier_reg() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state_with_diag(&mut ptx, &cfg, 0, "qstart");
        assert!(
            ptx.contains("cvt.u32.u64 %r_save_qlo, %q_start"),
            "qstart mode must narrow %q_start to u32"
        );
        assert!(
            ptx.contains("cvt.rn.f32.u32 %f_diag, %r_save_qlo"),
            "qstart mode row_max setup must convert %r_save_qlo"
        );
    }

    /// J-A2 diag annotation comment appears only when mode is non-empty.
    #[test]
    fn save_softmax_state_diag_annotation_gated() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut off = String::new();
        emit_save_softmax_state_with_diag(&mut off, &cfg, 0, "");
        assert!(
            !off.contains("J-A2 diagnostic"),
            "off mode must not emit diag annotation"
        );
        let mut on = String::new();
        emit_save_softmax_state_with_diag(&mut on, &cfg, 0, "wrow");
        assert!(
            on.contains("J-A2 diagnostic: NSL_CSHA_DUMP_SAVE_STATE=wrow"),
            "on mode must emit diag annotation with mode name"
        );
    }

    /// J-A3 fmax mode: both slots source from `%f_sdx_fmax` (populated by
    /// softmax.rs post-butterfly-max). No cvt setup — direct mov-source.
    #[test]
    fn save_softmax_state_diag_fmax_sources_from_softmax_capture() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state_with_diag(&mut ptx, &cfg, 0, "fmax");
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_sdx_fmax;  // row_max_ptr write"),
            "row_max slot must source from %f_sdx_fmax"
        );
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_sdx_fmax;  // row_sum_ptr write"),
            "row_sum slot must source from %f_sdx_fmax (both slots for unambiguous read-back)"
        );
        // No cvt setup for J-A3 modes — source regs are populated upstream.
        assert!(
            !ptx.contains("cvt.rn.f32.u32 %f_diag"),
            "fmax mode must not re-cvt into %f_diag"
        );
    }

    /// J-A3 newmax mode: both slots source from `%f_sdx_nmax`.
    #[test]
    fn save_softmax_state_diag_newmax_sources_from_online_update() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state_with_diag(&mut ptx, &cfg, 0, "newmax");
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_sdx_nmax;  // row_max_ptr write"),
            "row_max slot must source from %f_sdx_nmax"
        );
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_sdx_nmax;  // row_sum_ptr write"),
            "row_sum slot must source from %f_sdx_nmax"
        );
    }

    /// J-A3 fsum mode: both slots source from `%f_sdx_fsum`.
    #[test]
    fn save_softmax_state_diag_fsum_sources_from_butterfly_sum() {
        let cfg = FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: true,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state_with_diag(&mut ptx, &cfg, 0, "fsum");
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_sdx_fsum;  // row_max_ptr write"),
            "row_max slot must source from %f_sdx_fsum"
        );
        assert!(
            ptx.contains("st.global.f32 [%rd_save_elem], %f_sdx_fsum;  // row_sum_ptr write"),
            "row_sum slot must source from %f_sdx_fsum"
        );
    }

    #[test]
    fn save_softmax_state_skips_hbm_when_backward_flag_false() {
        let cfg = FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            num_sink_tokens: 0,
            gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: false,
                d_model: 128,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        };
        let mut ptx = String::new();
        emit_save_softmax_state(&mut ptx, &cfg, 0);
        assert!(!ptx.contains("row_max_ptr"));
        assert!(!ptx.contains("row_sum_ptr"));
        assert!(!ptx.contains("st.global.f32"));
    }

    #[test]
    fn a4_rope_epilogue_placed_before_attention_body() {
        let cfg = base_cfg_for_rope_test();
        let ptx_bytes =
            crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg);
        // synthesize returns a NUL-terminated byte vec; drop the trailing NUL.
        let ptx = std::str::from_utf8(&ptx_bytes[..ptx_bytes.len().saturating_sub(1)])
            .expect("PTX should be valid UTF-8");

        let rope_q_idx = ptx
            .find("V2_CSHA_ROPE_Q_LOOP_")
            .expect("ROPE_Q label missing — emit_rope_epilogue did not fire");
        // V2_LOOP_S_OVER_K_{iter} is emitted by s_compute::emit, which is the
        // first phase that consumes Q and K for QK^T.  RoPE must precede it.
        let attn_body_idx = ptx
            .find("V2_LOOP_S_OVER_K_")
            .expect("S-compute loop label missing — s_compute did not fire");

        assert!(
            rope_q_idx < attn_body_idx,
            "RoPE must run pre-attention (before QK^T / S-compute); \
             found ROPE_Q @ byte {rope_q_idx} but S-compute @ byte {attn_body_idx}"
        );
    }

    /// Verifies that `emit_rope_k_epilogue` fires in the fused path and
    /// produces a K rotation sweep before the S-compute body.
    #[test]
    fn a4_rope_k_epilogue_placed_before_s_compute_in_full_ptx() {
        let cfg = base_cfg_for_rope_test();
        let ptx_bytes =
            crate::flash_attention_v2::synthesize_flash_attention_ptx_v2(&cfg);
        let ptx = std::str::from_utf8(&ptx_bytes[..ptx_bytes.len().saturating_sub(1)])
            .expect("PTX should be valid UTF-8");

        let rope_k_idx = ptx
            .find("V2_CSHA_ROPE_K_LOOP_0:")
            .expect("ROPE_K label missing — emit_rope_k_epilogue did not fire in fused path");
        let attn_body_idx = ptx
            .find("V2_LOOP_S_OVER_K_")
            .expect("S-compute loop label missing");

        assert!(
            rope_k_idx < attn_body_idx,
            "K RoPE must run before S-compute; \
             found ROPE_K @ byte {rope_k_idx} but S-compute @ byte {attn_body_idx}"
        );
    }

    // ── Cycle-10 §5.3 Task 8: namespaced emit_prologue tests ──────────────

    #[test]
    fn cycle10_task8_forward_path_byte_identical() {
        // emit_prologue is now a thin shim around emit_prologue_namespaced
        // with suffix="". The PTX bytes must match a direct
        // emit_prologue_namespaced(..., "") call exactly — this is the
        // contract that guarantees fa_v2_snapshots byte-identity.
        let cfg = cfg_with_projections();
        let mut a = String::new();
        let mut b = String::new();
        emit_prologue(&mut a, &cfg, 0);
        emit_prologue_namespaced(&mut b, &cfg, 0, "");
        assert_eq!(a, b, "forward path must equal namespaced empty-suffix call");

        // Specifically: emitted label uses the unsuffixed form.
        assert!(
            a.contains("V2_CSHA_PROLOGUE_SKIP_0:"),
            "forward label must be V2_CSHA_PROLOGUE_SKIP_0 (no suffix); got: {}",
            a
        );
        assert!(
            !a.contains("V2_CSHA_PROLOGUE_SKIP_0_bwd"),
            "forward must NOT include backward-suffixed label"
        );
    }

    #[test]
    fn cycle10_task8_emit_prologue_recompute_uses_suffix() {
        // emit_prologue_recompute is the backward-path entry point. With
        // a backward-distinguishing suffix the emitted labels must carry
        // the suffix so they don't collide with forward labels in the
        // same PTX text section.
        let cfg = cfg_with_projections();
        let mut ptx = String::new();
        emit_prologue_recompute(&mut ptx, &cfg, 0, "_bwd_0");
        assert!(
            ptx.contains("V2_CSHA_PROLOGUE_SKIP_0_bwd_0:"),
            "backward-suffixed label V2_CSHA_PROLOGUE_SKIP_0_bwd_0 missing; got: {}",
            ptx
        );
        assert!(
            ptx.contains("V2_CSHA_PROLOGUE_SKIP_0_bwd_0;"),
            "branch target to V2_CSHA_PROLOGUE_SKIP_0_bwd_0 missing; got: {}",
            ptx
        );
    }

    #[test]
    fn cycle10_task8_no_csha_short_circuits_with_or_without_suffix() {
        // The csha=None bail-out is independent of the suffix.
        let mut cfg = cfg_with_projections();
        cfg.csha = None;
        let mut a = String::new();
        let mut b = String::new();
        emit_prologue(&mut a, &cfg, 0);
        emit_prologue_recompute(&mut b, &cfg, 0, "_bwd_0");
        assert_eq!(
            a, b,
            "csha=None short-circuit must be suffix-independent"
        );
        assert!(a.contains("csha=None"));
    }
}
