//! Tensor-core (m16n8k16) forward for the plain decorator-free SDPA path.
//!
//! The v2 scalar forward computes Q@K^T and P@V with per-lane `fma.rn.f32`
//! chains — on any sm_80+ part that leaves the tensor pipes idle while the
//! FP32 vector units bound the whole training step (measured 52% of GPU time
//! on Coder-50M, RTX PRO 4500). This module emits the same kernel contract
//! (identical 36-param signature, SMEM `total_bytes`, 128-thread CTA, grid,
//! f16 output staging, f32 LSE) with the two matmuls on `mma.sync`.
//!
//! ## Structure (per CTA = one block_q tile)
//!
//! Unlike the scalar path's per-4-row `q_tile_iter` re-emission (which
//! reloads K/V from HBM `block_q/4` times), this body loads the full Q tile
//! once and runs a single KV loop:
//!
//! ```text
//!   Q tile   -> SMEM f16 [block_q][head_dim]   (cooperative, once)
//!   loop kv:
//!     K tile -> SMEM f16 [block_kv][head_dim]  (emit_k_tile_load, row-major)
//!     S      = Q @ K^T via MMA                 (fragments, f32 accum)
//!     scale + causal mask on the S fragments
//!     online softmax, all in registers          (see "warp-per-m-tile")
//!     V^T    -> SMEM f16 [head_dim][block_kv]  (transposed store)
//!     O     += P @ V via MMA                    (P repacked in registers)
//!   O / l -> out (f16), m + ln(l) -> logsumexp (f32)
//! ```
//!
//! ## Warp-per-m-tile
//!
//! Warp `w` owns output rows `16w .. 16w+15` (m-tile `w`). Every softmax
//! row therefore lives inside ONE warp: in the m16n8k16 accumulator layout
//! lane `t` holds rows `g = t/4` and `g+8` at columns `2(t%4)` and
//! `2(t%4)+1` of each n-tile, so a full row is spread over exactly the 4
//! lanes sharing `g` — a 2-step `shfl.bfly` (xor 1, xor 2) reduces it.
//! No cross-warp softmax scratch, no SMEM round-trip for S or P (CSHA
//! Tier B.1 needed both because it split n-tiles across warps).
//!
//! For `block_q < 64` the spare warps redundantly compute m-tile
//! `w % m_tiles` (the backward's proven redundant-warp pattern) and are
//! excluded from stores — every warp still reaches every `bar.sync`.
//!
//! ## Layout facts the fragment loaders rely on
//!
//! * Q A-fragments: row-major f16, `row_stride = head_dim*2`.
//! * K B-fragments: the byte-alias trick — row-major K `[bkv][hd]` read
//!   with `col_stride = head_dim*2` IS col-major K^T (matmul_mma.rs docs).
//! * V B-fragments need col-major V, so the V load stores TRANSPOSED
//!   (`[hd][bkv]`, dst = d*(bkv*2) + k*2) — CSHA Tier B.1's bug #4, fixed
//!   here by construction. Advancing 16 k-rows in that layout is +32 bytes.
//! * P A-fragments repack straight from the S accumulator registers
//!   (2× `cvt.rn.f16.f32` + `mov.b32` per register) — the accumulator
//!   (row, col) map equals the A-fragment map for the same lane.
//!
//! SMEM base registers are true shared-window offsets obtained with
//! `mov.u32 %fm_smem32, shmem;` — NOT `cvt.u32.u64` of the generic
//! `%shmem_base`, whose low 32 bits are not guaranteed to be the window
//! offset on sm_120 (see the segment-ids loader comment in prelude.rs).
//!
//! ## Numerics
//!
//! Inputs were already f16-narrowed by the scalar path (Q/K/V SMEM tiles
//! are f16 there too); this path additionally quantizes P to f16 for the
//! PV MMA and changes reduction order. The mask sentinel is -inf exactly
//! like s_compute.rs; every row of KV tile 0 has column 0 unmasked, so the
//! running max is finite from the first tile and no inf-inf NaN path
//! exists (the correction still carries the scalar path's NaN guard).
//! LSE definition matches finalize.rs: `row_max + lg2(row_sum)*ln2`.
//!
//! Kill switch: `NSL_FA_FWD_MMA=0` at codegen time forces the scalar body
//! (mirrors `NSL_FUSED_LCE_GEMM`'s opt-out shape; see the selector's
//! retired `NSL_FA_EMITTER` for the precedent of codegen-side gates).

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::phases;
use crate::matmul_mma::{
    emit_load_a_fragment_smem, emit_load_b_fragment_smem, emit_mma_instruction,
};

/// Config-shape admission (env-independent; tests target this directly).
///
/// v1 scope is exactly the decorator-free SDPA table: plain non-CSHA
/// configs, GQA pre-expanded (`gqa_group_size == 1`), no paged/rope/tree/
/// segment/sink/checkpoint extras. Every `TILE_CANDIDATES` × head_dim the
/// table produces satisfies the tile-divisibility rules below.
pub fn admits_shape(config: &FlashAttentionConfig) -> bool {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    config.gpu_sm >= 80
        && config.csha.is_none()
        && config.checkpoint.is_none()
        && !config.paged
        && !config.rope_q
        && !config.tree_mask
        && !config.segment_masked
        && config.num_sink_tokens == 0
        && config.gqa_group_size == 1
        // block_q rows split into 16-row m-tiles owned by the 4 warps.
        // m_tiles must itself be a power of two (1/2/4) — the warp->m-tile
        // map is `warp_id & (m_tiles-1)`, so bq=48 (m_tiles=3) would leave
        // rows 16..31 never computed. bq in {16, 32, 64} exactly.
        && bq.is_multiple_of(16)
        && bq <= 64
        && (bq / 16).is_power_of_two()
        // block_kv is the S n-dim (%8) AND the PV reduction dim (%16).
        // Bounded at 64: bkv=128 is emitter-legal on paper (the SMEM math
        // holds) but sits outside every ptxas/parity gate — widen only
        // together with coverage.
        && bkv.is_multiple_of(16)
        && bkv <= 64
        // head_dim is the QK reduction dim (%16) and the PV n-dim (%8);
        // power-of-two lets the V^T store use shifts instead of divides.
        // 32..128 is the validated ALLOWED_HEAD_DIM intersection.
        && hd.is_multiple_of(16)
        && hd.is_power_of_two()
        && (32..=128).contains(&hd)
}

/// Full dispatch decision: config shape + the codegen-side kill switch.
/// Consulted by BOTH `flash_attention_kernel_name_v2` (for the `_mma`
/// suffix) and `synthesize_flash_attention_ptx_v2_with_tier_b` (for the
/// body), mirroring the `is_tier_b1_dispatch` pattern so name and body
/// can never disagree within one compile.
pub fn is_mma_forward_dispatch(config: &FlashAttentionConfig) -> bool {
    if std::env::var("NSL_FA_FWD_MMA").ok().as_deref() == Some("0") {
        return false;
    }
    admits_shape(config)
}

/// Every SMEM tile row is padded by 4 bytes. Unpadded, the natural
/// strides are multiples of 128 bytes (hd*2 or bkv*2 for the table
/// shapes), so all 8 fragment rows land on ONE of the 32 4-byte banks
/// (8-way conflict on every A/B fragment load) and the V^T transposed
/// store is a full 32-way conflict. +4 bytes makes the stride an odd
/// multiple of 4 (bank rotation 1): consecutive rows visit consecutive
/// banks and every access above is conflict-free. The backward's
/// `hd_padded = hd + 4` exists for the same reason. 4-byte alignment of
/// `ld.shared.b32` is preserved (stride % 4 == 0).
const ROW_PAD_BYTES: u32 = 4;

fn q_stride(hd: u32) -> u32 { hd * 2 + ROW_PAD_BYTES }
fn k_stride(hd: u32) -> u32 { hd * 2 + ROW_PAD_BYTES }
fn vt_stride(bkv: u32) -> u32 { bkv * 2 + ROW_PAD_BYTES }

/// Q band bytes (block_q padded rows).
fn q_band(config: &FlashAttentionConfig) -> u32 {
    config.block_q as u32 * q_stride(config.head_dim as u32)
}

/// K / V^T shared band bytes: K is [bkv][hd] padded, V^T is [hd][bkv]
/// padded; the band must hold whichever is larger.
fn kv_band(config: &FlashAttentionConfig) -> u32 {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    (bkv * k_stride(hd)).max(hd * vt_stride(bkv))
}

/// Total static SMEM the MMA body declares (via the prelude's
/// `emit_with_smem_override`, the Tier B.1 precedent). Always below the
/// scalar `total_bytes` + 48 KB static cap for every table shape; the
/// runtime's launch-side dynamic-SMEM request is computed from the
/// scalar layout and is simply unused extra against a static array.
fn padded_total(config: &FlashAttentionConfig) -> u32 {
    q_band(config) + kv_band(config)
}

/// Emit the complete MMA forward kernel. The caller (v2 dispatch) has
/// already checked `is_mma_forward_dispatch`.
pub fn synthesize(config: &FlashAttentionConfig) -> Vec<u8> {
    let bq = config.block_q as u32;
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    let m_tiles = bq / 16;
    let n_tiles_s = (bkv / 8) as usize;
    let n_tiles_o = (hd / 8) as usize;
    let k_iters_qk = (hd / 16) as usize;
    let k_iters_pv = (bkv / 16) as usize;
    let q_off = 0u32;
    let kv_off = q_band(config);

    let mut ptx = String::new();

    // Same header / entry / params / pools / index math as the scalar
    // body, but with the padded SMEM total (see ROW_PAD_BYTES).
    // tier_b is None by contract: segment_masked configs are refused above,
    // so no PCA range table can ever be admitted here.
    phases::prelude::emit_with_smem_override(&mut ptx, config, padded_total(config), None);

    emit_mma_registers(&mut ptx, n_tiles_s, n_tiles_o);
    emit_prolog(&mut ptx, config, m_tiles);
    emit_q_tile_load(&mut ptx, config, q_off);

    // ── KV loop ──────────────────────────────────────────────────────
    ptx.push_str("    mov.u64 %k_start, 0;\n");
    ptx.push_str("FMMA_LOOP_KV:\n");

    emit_k_tile_load_padded(&mut ptx, config, kv_off);

    emit_qk_mma(&mut ptx, config, n_tiles_s, k_iters_qk, q_off, kv_off);
    emit_scale_and_mask(&mut ptx, config, n_tiles_s);
    emit_online_softmax(&mut ptx, n_tiles_s, n_tiles_o);

    // All warps' QK fragment loads are complete (register-resident S);
    // safe to overwrite the K band with V^T.
    ptx.push_str("    bar.sync 0;  // FENCE: K consumed, V^T may overwrite\n");
    emit_vt_tile_load(&mut ptx, config, kv_off);
    emit_pv_mma(&mut ptx, n_tiles_o, k_iters_pv, bkv, kv_off);
    ptx.push_str("    bar.sync 0;  // FENCE: V consumed, next K load may overwrite\n");

    ptx.push_str(&format!("    add.u64 %k_start, %k_start, {};\n", bkv));
    ptx.push_str("    setp.lt.u64 %p0, %k_start, %fm_kbound;\n");
    ptx.push_str("    @%p0 bra FMMA_LOOP_KV;\n");

    emit_epilogue(&mut ptx, config, n_tiles_o);

    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
    if !ptx.ends_with('\n') {
        ptx.push('\n');
    }
    let mut bytes = ptx.into_bytes();
    bytes.push(0);
    bytes
}

/// Named register declarations. Everything is named (never `%f<N>`-pool
/// indexed) so the scalar pools declared by the prelude stay untouched;
/// `%mma_addr`/`%mma_a_row`/`%mma_b_row` are the scratch the matmul_mma
/// fragment loaders clobber and require in scope.
fn emit_mma_registers(ptx: &mut String, n_tiles_s: usize, n_tiles_o: usize) {
    ptx.push_str("    // -- MMA forward registers --\n");
    ptx.push_str("    .reg .u32 %mma_addr, %mma_a_row, %mma_b_row;\n");
    ptx.push_str("    .reg .u32 %fm_smem32, %fm_abase, %fm_bbase, %fm_qwoff;\n");
    ptx.push_str("    .reg .u32 %fm_g, %fm_tid2, %fm_mtile, %fm_qrow32, %fm_krow32, %fm_kb32, %fm_u0;\n");
    ptx.push_str("    .reg .u64 %fm_kbound;\n");
    ptx.push_str("    .reg .f32 %fm_m_lo, %fm_m_hi, %fm_l_lo, %fm_l_hi;\n");
    ptx.push_str("    .reg .f32 %fm_alpha_lo, %fm_alpha_hi, %fm_tmax_lo, %fm_tmax_hi;\n");
    ptx.push_str("    .reg .f32 %fm_psum_lo, %fm_psum_hi, %fm_t0, %fm_t1;\n");
    ptx.push_str("    .reg .b16 %fm_h0, %fm_h1;\n");
    ptx.push_str("    .reg .pred %fm_p0, %fm_p1, %fm_pstore;\n");
    ptx.push_str("    .reg .b32 %fm_qa_0, %fm_qa_1, %fm_qa_2, %fm_qa_3;\n");
    ptx.push_str("    .reg .b32 %fm_pa_0, %fm_pa_1, %fm_pa_2, %fm_pa_3;\n");
    ptx.push_str("    .reg .b32 %fm_b_0, %fm_b_1;\n");
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!(
            "    .reg .f32 %fm_s_{nt}_0, %fm_s_{nt}_1, %fm_s_{nt}_2, %fm_s_{nt}_3;\n"
        ));
    }
    for nt in 0..n_tiles_o {
        ptx.push_str(&format!(
            "    .reg .f32 %fm_o_{nt}_0, %fm_o_{nt}_1, %fm_o_{nt}_2, %fm_o_{nt}_3;\n"
        ));
    }
}

fn emit_prolog(ptx: &mut String, config: &FlashAttentionConfig, m_tiles: u32) {
    let hd = config.head_dim as u32;
    ptx.push_str("    // -- MMA prolog: lane map, warp m-tile, stats init --\n");
    // True shared-window byte offset of the shmem[] symbol (see module doc).
    ptx.push_str("    mov.u32 %fm_smem32, shmem;\n");
    // Accumulator lane map: g = lane/4 (row-in-tile), tid2 = (lane%4)*2 (col).
    ptx.push_str("    shr.u32 %fm_g, %lane, 2;\n");
    ptx.push_str("    and.b32 %fm_tid2, %lane, 3;\n");
    ptx.push_str("    shl.b32 %fm_tid2, %fm_tid2, 1;\n");
    // m_tile = warp_id % m_tiles (m_tiles is a power of two: 1/2/4).
    ptx.push_str(&format!(
        "    and.b32 %fm_mtile, %warp_id, {};\n",
        m_tiles - 1
    ));
    // Byte offset of this warp's 16-row A-tile inside the Q band.
    ptx.push_str("    mov.u32 %fm_qwoff, %fm_mtile;\n");
    ptx.push_str(&format!(
        "    mul.lo.u32 %fm_qwoff, %fm_qwoff, {};\n",
        16 * q_stride(hd)
    ));
    // Only warps with a distinct m-tile store output.
    ptx.push_str(&format!(
        "    setp.lt.u32 %fm_pstore, %warp_id, {};\n",
        m_tiles
    ));
    // Global row_lo this lane owns: q_start + m_tile*16 + g (u32; the
    // runtime declines seq >= 2^31 shapes long before here).
    ptx.push_str("    cvt.u32.u64 %fm_qrow32, %q_start;\n");
    ptx.push_str("    shl.b32 %fm_u0, %fm_mtile, 4;\n");
    ptx.push_str("    add.u32 %fm_qrow32, %fm_qrow32, %fm_u0;\n");
    ptx.push_str("    add.u32 %fm_qrow32, %fm_qrow32, %fm_g;\n");
    // KV loop bound: causal tiles past the diagonal contribute exactly 0
    // (every column masked), so stop at the diagonal tile. k_start is a
    // multiple of block_kv, so `k_start < q_start + block_q` is exact.
    ptx.push_str("    mov.u64 %fm_kbound, %rd6;                 // seq_len\n");
    if config.causal {
        ptx.push_str(&format!(
            "    add.u64 %rd52, %q_start, {};\n",
            config.block_q
        ));
        ptx.push_str("    min.u64 %fm_kbound, %fm_kbound, %rd52;\n");
    }
    // Online-softmax running state.
    ptx.push_str("    mov.f32 %fm_m_lo, 0fFF800000;\n");
    ptx.push_str("    mov.f32 %fm_m_hi, 0fFF800000;\n");
    ptx.push_str("    mov.f32 %fm_l_lo, 0f00000000;\n");
    ptx.push_str("    mov.f32 %fm_l_hi, 0f00000000;\n");
    let n_tiles_o = (hd / 8) as usize;
    for nt in 0..n_tiles_o {
        for r in 0..4 {
            ptx.push_str(&format!("    mov.f32 %fm_o_{nt}_{r}, 0f00000000;\n"));
        }
    }
}

/// Cooperative tile load with a PADDED row stride: flat idx over
/// rows*cols f32 source elements (contiguous in HBM), f16 store at
/// `band_off + row*stride + col*2` where row = idx >> log2(cols),
/// col = idx & (cols-1) (cols is a power of two for every table shape).
fn emit_padded_tile_load(
    ptx: &mut String,
    label: &str,
    src_reg: &str,
    row_base_u64: &str, // register holding (bh*seq + first_row), pre-scaled by caller
    rows: u32,
    cols: u32,
    stride: u32,
    band_off: u32,
) {
    let elems = (rows * cols) as u64;
    let log2_cols = cols.trailing_zeros();
    ptx.push_str(&format!("    mul.lo.u64 %rd52, {row_base_u64}, %rd7;\n"));
    ptx.push_str("    shl.b64 %rd52, %rd52, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd52, {src_reg}, %rd52;          // tile base (f32)\n"));
    ptx.push_str("    cvt.u64.u32 %rd53, %tid_x;\n");
    ptx.push_str(&format!("{label}:\n"));
    ptx.push_str("    shl.b64 %rd54, %rd53, 2;\n");
    ptx.push_str("    add.u64 %rd55, %rd52, %rd54;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd55];\n");
    ptx.push_str("    cvt.rn.f16.f32 %h0, %f0;\n");
    ptx.push_str(&format!("    shr.u64 %rd54, %rd53, {};\n", log2_cols));
    ptx.push_str(&format!("    mul.lo.u64 %rd54, %rd54, {};              // row * stride\n", stride));
    ptx.push_str(&format!("    and.b64 %rd55, %rd53, {};\n", cols - 1));
    ptx.push_str("    shl.b64 %rd55, %rd55, 1;                  // col * 2\n");
    ptx.push_str("    add.u64 %rd54, %rd54, %rd55;\n");
    if band_off > 0 {
        ptx.push_str(&format!("    add.u64 %rd54, %rd54, {};\n", band_off));
    }
    ptx.push_str("    add.u64 %smem_addr, %rd54, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd53, %rd53, 128;\n");
    ptx.push_str(&format!("    setp.lt.u64 %p0, %rd53, {};\n", elems));
    ptx.push_str(&format!("    @%p0 bra {label};\n"));
}

/// Full block_q x head_dim Q tile, padded rows at band offset 0.
fn emit_q_tile_load(ptx: &mut String, config: &FlashAttentionConfig, q_off: u32) {
    let hd = config.head_dim as u32;
    ptx.push_str("    // Q tile load: 128 threads, full tile, padded rows\n");
    ptx.push_str("    mul.lo.u64 %rd56, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd56, %rd56, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd56, %rd56, %rd6;\n");
    ptx.push_str("    add.u64 %rd56, %rd56, %q_start;\n");
    emit_padded_tile_load(
        ptx, "FMMA_Q_LOAD", "%rd0", "%rd56",
        config.block_q as u32, hd, q_stride(hd), q_off,
    );
    ptx.push_str("    bar.sync 0;  // FENCE: Q tile in shmem\n");
}

/// K tile [block_kv][head_dim] f16, padded rows, at kv_off. Replaces the
/// scalar emit_k_tile_load (whose unpadded rows would conflict on every
/// B-fragment load).
fn emit_k_tile_load_padded(ptx: &mut String, config: &FlashAttentionConfig, kv_off: u32) {
    let hd = config.head_dim as u32;
    ptx.push_str("    // K tile load (padded rows)\n");
    ptx.push_str("    mul.lo.u64 %rd56, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd56, %rd56, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd56, %rd56, %rd6;\n");
    ptx.push_str("    add.u64 %rd56, %rd56, %k_start;\n");
    emit_padded_tile_load(
        ptx, "FMMA_K_LOAD", "%rd1", "%rd56",
        config.block_kv as u32, hd, k_stride(hd), kv_off,
    );
    ptx.push_str("    bar.sync 0;  // FENCE: K tile in shmem\n");
}

/// Cooperative V^T load: reads the same contiguous [block_kv][head_dim]
/// f32 rows as the scalar V load but stores TRANSPOSED f16 ([hd][bkv],
/// dst = d*(bkv*2) + k*2) so `emit_load_b_fragment_smem` with
/// `col_stride = bkv*2` reads true col-major V (Tier B.1 bug #4, avoided
/// by construction).
fn emit_vt_tile_load(ptx: &mut String, config: &FlashAttentionConfig, kv_off: u32) {
    let bkv = config.block_kv as u32;
    let hd = config.head_dim as u32;
    let elems = (bkv * hd) as u64;
    let log2_hd = hd.trailing_zeros();
    let stride = vt_stride(bkv);
    ptx.push_str("    // V^T tile load: transposed padded f16 store into the K band\n");
    ptx.push_str("    mul.lo.u64 %rd52, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd52, %rd52, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd52, %rd52, %rd6;\n");
    ptx.push_str("    add.u64 %rd52, %rd52, %k_start;\n");
    ptx.push_str("    mul.lo.u64 %rd52, %rd52, %rd7;\n");
    ptx.push_str("    shl.b64 %rd52, %rd52, 2;\n");
    ptx.push_str("    add.u64 %rd52, %rd2, %rd52;               // v tile base (f32)\n");
    ptx.push_str("    cvt.u64.u32 %rd53, %tid_x;\n");
    ptx.push_str("FMMA_VT_LOAD:\n");
    ptx.push_str("    shl.b64 %rd54, %rd53, 2;\n");
    ptx.push_str("    add.u64 %rd55, %rd52, %rd54;\n");
    ptx.push_str("    ld.global.f32 %f0, [%rd55];\n");
    ptx.push_str("    cvt.rn.f16.f32 %h0, %f0;\n");
    // k = idx >> log2(hd), d = idx & (hd-1); dst = kv_off + d*stride + k*2.
    ptx.push_str(&format!("    shr.u64 %rd54, %rd53, {};\n", log2_hd));
    ptx.push_str("    shl.b64 %rd54, %rd54, 1;                  // k * 2\n");
    ptx.push_str(&format!("    and.b64 %rd55, %rd53, {};\n", hd - 1));
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd55, %rd55, {};              // d * vt_stride\n",
        stride
    ));
    ptx.push_str("    add.u64 %rd54, %rd54, %rd55;\n");
    ptx.push_str(&format!("    add.u64 %rd54, %rd54, {};\n", kv_off));
    ptx.push_str("    add.u64 %smem_addr, %rd54, %shmem_base;\n");
    ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
    ptx.push_str("    add.u64 %rd53, %rd53, 128;\n");
    ptx.push_str(&format!("    setp.lt.u64 %p0, %rd53, {};\n", elems));
    ptx.push_str("    @%p0 bra FMMA_VT_LOAD;\n");
    ptx.push_str("    bar.sync 0;  // FENCE: V^T tile in shmem\n");
}

/// S = Q @ K^T. A-fragments from the warp's Q rows (row-major), B-fragments
/// from K via the byte-alias trick (row-major K [bkv][hd] with
/// col_stride = hd*2 IS col-major K^T). Accumulates into %fm_s_{nt}_{r}.
fn emit_qk_mma(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    n_tiles_s: usize,
    k_iters: usize,
    q_off: u32,
    kv_off: u32,
) {
    let hd = config.head_dim as u32;
    ptx.push_str("    // -- QK^T MMA --\n");
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!("    mov.f32 %fm_s_{nt}_{r}, 0f00000000;\n"));
        }
    }
    let qa: [String; 4] = core::array::from_fn(|i| format!("fm_qa_{i}"));
    let b: [String; 2] = [String::from("fm_b_0"), String::from("fm_b_1")];
    let b_pct: [String; 2] = [String::from("%fm_b_0"), String::from("%fm_b_1")];
    let qa_pct: [String; 4] = core::array::from_fn(|i| format!("%fm_qa_{i}"));
    for k_iter in 0..k_iters {
        // A base = smem32 + q_off + warp_m_tile_bytes + k_iter*32.
        ptx.push_str("    add.u32 %fm_abase, %fm_smem32, %fm_qwoff;\n");
        let a_const = q_off + (k_iter as u32) * 32;
        if a_const > 0 {
            ptx.push_str(&format!("    add.u32 %fm_abase, %fm_abase, {};\n", a_const));
        }
        emit_load_a_fragment_smem(ptx, &qa, "%fm_abase", q_stride(hd) as usize);
        for nt in 0..n_tiles_s {
            let b_const = kv_off + (nt as u32) * 8 * k_stride(hd) + (k_iter as u32) * 32;
            ptx.push_str(&format!(
                "    add.u32 %fm_bbase, %fm_smem32, {};\n",
                b_const
            ));
            emit_load_b_fragment_smem(ptx, &b, "%fm_bbase", k_stride(hd) as usize);
            let s: [String; 4] = core::array::from_fn(|r| format!("%fm_s_{nt}_{r}"));
            emit_mma_instruction(ptx, &s, &qa_pct, &b_pct, &s);
        }
    }
}

/// scale, then causal mask with the -inf sentinel (identical semantics to
/// s_compute.rs). Accumulator (row, col) per register r:
/// row = qrow32 (+8 when r>=2), col = k_start + nt*8 + tid2 (+1 when r odd).
fn emit_scale_and_mask(ptx: &mut String, config: &FlashAttentionConfig, n_tiles_s: usize) {
    ptx.push_str("    // scale S\n");
    for nt in 0..n_tiles_s {
        for r in 0..4 {
            ptx.push_str(&format!(
                "    mul.f32 %fm_s_{nt}_{r}, %fm_s_{nt}_{r}, %scale;\n"
            ));
        }
    }
    if !config.causal {
        return;
    }
    ptx.push_str("    // causal: k_global > q_global -> S = -inf\n");
    ptx.push_str("    cvt.u32.u64 %fm_kb32, %k_start;\n");
    for nt in 0..n_tiles_s {
        ptx.push_str("    mov.u32 %fm_krow32, %fm_kb32;\n");
        let nt_off = (nt as u32) * 8;
        if nt_off > 0 {
            ptx.push_str(&format!(
                "    add.u32 %fm_krow32, %fm_krow32, {};\n",
                nt_off
            ));
        }
        ptx.push_str("    add.u32 %fm_krow32, %fm_krow32, %fm_tid2;\n");
        // row_hi = row_lo + 8 (scratch %fm_u0).
        ptx.push_str("    add.u32 %fm_u0, %fm_qrow32, 8;\n");
        // even column (r0 row_lo, r2 row_hi)
        ptx.push_str("    setp.gt.u32 %fm_p0, %fm_krow32, %fm_qrow32;\n");
        ptx.push_str(&format!("    @%fm_p0 mov.f32 %fm_s_{nt}_0, 0fFF800000;\n"));
        ptx.push_str("    setp.gt.u32 %fm_p1, %fm_krow32, %fm_u0;\n");
        ptx.push_str(&format!("    @%fm_p1 mov.f32 %fm_s_{nt}_2, 0fFF800000;\n"));
        // odd column (r1 row_lo, r3 row_hi)
        ptx.push_str("    add.u32 %fm_krow32, %fm_krow32, 1;\n");
        ptx.push_str("    setp.gt.u32 %fm_p0, %fm_krow32, %fm_qrow32;\n");
        ptx.push_str(&format!("    @%fm_p0 mov.f32 %fm_s_{nt}_1, 0fFF800000;\n"));
        ptx.push_str("    setp.gt.u32 %fm_p1, %fm_krow32, %fm_u0;\n");
        ptx.push_str(&format!("    @%fm_p1 mov.f32 %fm_s_{nt}_3, 0fFF800000;\n"));
    }
}

/// Online softmax across this KV tile, entirely in registers.
/// Row-halves: `lo` = row g of the m-tile, `hi` = row g+8. The 4 lanes
/// sharing g hold the full row; bfly xor 1 + xor 2 reduces across them.
fn emit_online_softmax(ptx: &mut String, n_tiles_s: usize, n_tiles_o: usize) {
    ptx.push_str("    // -- online softmax (register-resident) --\n");
    // Tile row max per half.
    ptx.push_str("    mov.f32 %fm_tmax_lo, 0fFF800000;\n");
    ptx.push_str("    mov.f32 %fm_tmax_hi, 0fFF800000;\n");
    for nt in 0..n_tiles_s {
        ptx.push_str(&format!("    max.f32 %fm_tmax_lo, %fm_tmax_lo, %fm_s_{nt}_0;\n"));
        ptx.push_str(&format!("    max.f32 %fm_tmax_lo, %fm_tmax_lo, %fm_s_{nt}_1;\n"));
        ptx.push_str(&format!("    max.f32 %fm_tmax_hi, %fm_tmax_hi, %fm_s_{nt}_2;\n"));
        ptx.push_str(&format!("    max.f32 %fm_tmax_hi, %fm_tmax_hi, %fm_s_{nt}_3;\n"));
    }
    for off in [1u32, 2] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %fm_tmax_lo, {off}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    max.f32 %fm_tmax_lo, %fm_tmax_lo, %shfl_tmp;\n");
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %fm_tmax_hi, {off}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    max.f32 %fm_tmax_hi, %fm_tmax_hi, %shfl_tmp;\n");
    }
    // new_m, correction alpha = exp(old - new), l *= alpha. NaN guard as in
    // softmax.rs (only reachable if a whole row was masked in every tile
    // so far, which full-tile-0 coverage rules out — kept for defense).
    for half in ["lo", "hi"] {
        ptx.push_str(&format!("    mov.f32 %fm_t0, %fm_m_{half};\n"));
        ptx.push_str(&format!(
            "    max.f32 %fm_m_{half}, %fm_m_{half}, %fm_tmax_{half};\n"
        ));
        ptx.push_str(&format!("    sub.f32 %fm_t0, %fm_t0, %fm_m_{half};\n"));
        ptx.push_str("    setp.nan.f32 %fm_p0, %fm_t0, %fm_t0;\n");
        ptx.push_str("    @%fm_p0 mov.f32 %fm_t0, 0f00000000;\n");
        ptx.push_str("    mul.f32 %fm_t0, %fm_t0, %log2e;\n");
        ptx.push_str(&format!("    ex2.approx.f32 %fm_alpha_{half}, %fm_t0;\n"));
        ptx.push_str(&format!(
            "    mul.f32 %fm_l_{half}, %fm_l_{half}, %fm_alpha_{half};\n"
        ));
    }
    // P = exp(S - m) in place + per-lane partial row sums.
    ptx.push_str("    mov.f32 %fm_psum_lo, 0f00000000;\n");
    ptx.push_str("    mov.f32 %fm_psum_hi, 0f00000000;\n");
    for nt in 0..n_tiles_s {
        for (r, half) in [(0, "lo"), (1, "lo"), (2, "hi"), (3, "hi")] {
            ptx.push_str(&format!(
                "    sub.f32 %fm_t0, %fm_s_{nt}_{r}, %fm_m_{half};\n"
            ));
            ptx.push_str("    mul.f32 %fm_t0, %fm_t0, %log2e;\n");
            ptx.push_str(&format!("    ex2.approx.f32 %fm_s_{nt}_{r}, %fm_t0;\n"));
            ptx.push_str(&format!(
                "    add.f32 %fm_psum_{half}, %fm_psum_{half}, %fm_s_{nt}_{r};\n"
            ));
        }
    }
    for off in [1u32, 2] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %fm_psum_lo, {off}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    add.f32 %fm_psum_lo, %fm_psum_lo, %shfl_tmp;\n");
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %fm_psum_hi, {off}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    add.f32 %fm_psum_hi, %fm_psum_hi, %shfl_tmp;\n");
    }
    ptx.push_str("    add.f32 %fm_l_lo, %fm_l_lo, %fm_psum_lo;\n");
    ptx.push_str("    add.f32 %fm_l_hi, %fm_l_hi, %fm_psum_hi;\n");
    // Rescale the O accumulators by alpha (rows lo: r0/r1, hi: r2/r3).
    for nt in 0..n_tiles_o {
        ptx.push_str(&format!("    mul.f32 %fm_o_{nt}_0, %fm_o_{nt}_0, %fm_alpha_lo;\n"));
        ptx.push_str(&format!("    mul.f32 %fm_o_{nt}_1, %fm_o_{nt}_1, %fm_alpha_lo;\n"));
        ptx.push_str(&format!("    mul.f32 %fm_o_{nt}_2, %fm_o_{nt}_2, %fm_alpha_hi;\n"));
        ptx.push_str(&format!("    mul.f32 %fm_o_{nt}_3, %fm_o_{nt}_3, %fm_alpha_hi;\n"));
    }
}

/// O += P @ V. P A-fragments repack in registers from the S accumulators:
/// for PV k-tile j (16 kv columns), a0/a1 come from S n-tile 2j (cols
/// tid2, tid2+1 = A cols 2t, 2t+1) and a2/a3 from n-tile 2j+1 (A cols
/// 2t+8, 2t+8+1) — rows g (a0/a2) and g+8 (a1/a3), exactly the
/// m16n8k16 A-fragment map for this lane.
fn emit_pv_mma(
    ptx: &mut String,
    n_tiles_o: usize,
    k_iters: usize,
    bkv: u32,
    kv_off: u32,
) {
    ptx.push_str("    // -- PV MMA --\n");
    let pa_pct: [String; 4] = core::array::from_fn(|i| format!("%fm_pa_{i}"));
    let b: [String; 2] = [String::from("fm_b_0"), String::from("fm_b_1")];
    let b_pct: [String; 2] = [String::from("%fm_b_0"), String::from("%fm_b_1")];
    for k_iter in 0..k_iters {
        let (lo_nt, hi_nt) = (2 * k_iter, 2 * k_iter + 1);
        ptx.push_str(&format!("    // P A-frags from S n-tiles {lo_nt}/{hi_nt}\n"));
        for (pa, s_nt, r_even, r_odd) in [
            (0, lo_nt, 0, 1), // (row g,   cols 2t..2t+1)
            (1, lo_nt, 2, 3), // (row g+8, cols 2t..2t+1)
            (2, hi_nt, 0, 1), // (row g,   cols 2t+8..)
            (3, hi_nt, 2, 3), // (row g+8, cols 2t+8..)
        ] {
            ptx.push_str(&format!("    cvt.rn.f16.f32 %fm_h0, %fm_s_{s_nt}_{r_even};\n"));
            ptx.push_str(&format!("    cvt.rn.f16.f32 %fm_h1, %fm_s_{s_nt}_{r_odd};\n"));
            ptx.push_str(&format!("    mov.b32 %fm_pa_{pa}, {{%fm_h0, %fm_h1}};\n"));
        }
        for nt in 0..n_tiles_o {
            // V^T is col-major V: advancing 16 k-rows = +32 bytes,
            // advancing 8 n-cols = +8 * vt_stride bytes.
            let b_const = kv_off + (nt as u32) * 8 * vt_stride(bkv) + (k_iter as u32) * 32;
            ptx.push_str(&format!(
                "    add.u32 %fm_bbase, %fm_smem32, {};\n",
                b_const
            ));
            emit_load_b_fragment_smem(ptx, &b, "%fm_bbase", vt_stride(bkv) as usize);
            let o: [String; 4] = core::array::from_fn(|r| format!("%fm_o_{nt}_{r}"));
            emit_mma_instruction(ptx, &o, &pa_pct, &b_pct, &o);
        }
    }
}

/// O / l -> f16 out, m + ln(l) -> f32 logsumexp. Store layout, dtype and
/// LSE definition match finalize.rs exactly; redundant warps skip via
/// %fm_pstore (no bar.sync below this point).
fn emit_epilogue(ptx: &mut String, config: &FlashAttentionConfig, n_tiles_o: usize) {
    let hd = config.head_dim as u32;
    ptx.push_str("    // -- epilogue: normalize + store O (f16) + LSE (f32) --\n");
    ptx.push_str("    @!%fm_pstore bra FMMA_DONE;\n");
    ptx.push_str("    rcp.approx.f32 %fm_alpha_lo, %fm_l_lo;\n");
    ptx.push_str("    rcp.approx.f32 %fm_alpha_hi, %fm_l_hi;\n");
    // (batch*heads + head) * seq — shared by the out and LSE addresses.
    ptx.push_str("    mul.lo.u64 %rd52, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd52, %rd52, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd52, %rd52, %rd6;\n");
    ptx.push_str("    cvt.u64.u32 %rd53, %fm_qrow32;\n");
    // out row_lo base = out_ptr + (bh*seq + row_lo) * hd * 2.
    ptx.push_str("    add.u64 %rd54, %rd52, %rd53;\n");
    ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd7;\n");
    ptx.push_str("    shl.b64 %rd54, %rd54, 1;\n");
    ptx.push_str("    add.u64 %rd54, %rd3, %rd54;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd55, %rd54, {};                 // row_hi base (+8 rows)\n",
        8 * hd * 2
    ));
    // Column byte offset for this lane: (nt*8 + tid2) * 2.
    ptx.push_str("    cvt.u64.u32 %rd56, %fm_tid2;\n");
    ptx.push_str("    shl.b64 %rd56, %rd56, 1;\n");
    for nt in 0..n_tiles_o {
        let col_const = (nt as u32) * 8 * 2;
        ptx.push_str("    add.u64 %rd57, %rd54, %rd56;\n");
        if col_const > 0 {
            ptx.push_str(&format!("    add.u64 %rd57, %rd57, {};\n", col_const));
        }
        ptx.push_str(&format!("    mul.f32 %fm_t0, %fm_o_{nt}_0, %fm_alpha_lo;\n"));
        ptx.push_str("    cvt.rn.f16.f32 %fm_h0, %fm_t0;\n");
        ptx.push_str("    st.global.b16 [%rd57], %fm_h0;\n");
        ptx.push_str(&format!("    mul.f32 %fm_t0, %fm_o_{nt}_1, %fm_alpha_lo;\n"));
        ptx.push_str("    cvt.rn.f16.f32 %fm_h0, %fm_t0;\n");
        ptx.push_str("    st.global.b16 [%rd57 + 2], %fm_h0;\n");
        ptx.push_str("    add.u64 %rd57, %rd55, %rd56;\n");
        if col_const > 0 {
            ptx.push_str(&format!("    add.u64 %rd57, %rd57, {};\n", col_const));
        }
        ptx.push_str(&format!("    mul.f32 %fm_t0, %fm_o_{nt}_2, %fm_alpha_hi;\n"));
        ptx.push_str("    cvt.rn.f16.f32 %fm_h0, %fm_t0;\n");
        ptx.push_str("    st.global.b16 [%rd57], %fm_h0;\n");
        ptx.push_str(&format!("    mul.f32 %fm_t0, %fm_o_{nt}_3, %fm_alpha_hi;\n"));
        ptx.push_str("    cvt.rn.f16.f32 %fm_h0, %fm_t0;\n");
        ptx.push_str("    st.global.b16 [%rd57 + 2], %fm_h0;\n");
    }
    // LSE: one writer lane per row (tid2 == 0), rows g and g+8.
    ptx.push_str("    // LSE store (col-0 lanes, null-guarded)\n");
    ptx.push_str("    and.b32 %fm_u0, %lane, 3;\n");
    ptx.push_str("    setp.eq.u32 %fm_p0, %fm_u0, 0;\n");
    ptx.push_str("    setp.ne.u64 %p_has_lse, %logsumexp_base, 0;\n");
    ptx.push_str("    and.pred %fm_p0, %fm_p0, %p_has_lse;\n");
    ptx.push_str("    add.u64 %rd56, %rd52, %rd53;\n");
    ptx.push_str("    shl.b64 %rd56, %rd56, 2;\n");
    ptx.push_str("    add.u64 %rd56, %logsumexp_base, %rd56;\n");
    // lse_lo = m_lo + lg2(l_lo) * ln2 (finalize.rs definition).
    ptx.push_str("    lg2.approx.f32 %fm_t0, %fm_l_lo;\n");
    ptx.push_str("    mov.f32 %fm_t1, 0f3F317218;               // ln(2)\n");
    ptx.push_str("    mul.f32 %fm_t0, %fm_t0, %fm_t1;\n");
    ptx.push_str("    add.f32 %fm_t0, %fm_t0, %fm_m_lo;\n");
    ptx.push_str("    @%fm_p0 st.global.f32 [%rd56], %fm_t0;\n");
    ptx.push_str("    lg2.approx.f32 %fm_t0, %fm_l_hi;\n");
    ptx.push_str("    mul.f32 %fm_t0, %fm_t0, %fm_t1;\n");
    ptx.push_str("    add.f32 %fm_t0, %fm_t0, %fm_m_hi;\n");
    ptx.push_str("    @%fm_p0 st.global.f32 [%rd56 + 32], %fm_t0;  // row_hi = +8 rows\n");
    ptx.push_str("FMMA_DONE:\n");
}
