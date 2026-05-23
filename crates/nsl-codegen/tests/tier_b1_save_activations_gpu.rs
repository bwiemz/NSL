//! GPU numerical validation of the Tier B.1 save-activation retrofit
//! (Phase 2.6 Task 2.7).
//!
//! This is the NUMERICAL CORRECTNESS GATE for the save-activation retrofit
//! landed in commit `7c00ae1c` (`tier_b1/finalize.rs`). The structural tests
//! (`tier_b1_save_activations_snapshot.rs`, the unit tests in `finalize.rs`,
//! and the ptxas/SASS validation) check STRUCTURE only — that the saves emit
//! the right store count, null guards, col-major V index, and absolute-K
//! seq index. Per the institutional meta-lesson
//! (`csha-tier-b1-numerical-correctness`):
//!
//!   "snapshot/ptxas/SASS validate structure only — end-to-end numerical
//!    CPU-reference comparison is non-negotiable for fused-MMA kernels."
//!
//! This test launches Tier B.1 with `save_activations_for_backward=true` on
//! the GPU, reads back the 5 saved tensors (q_proj, k_proj, v_proj f16
//! `[B,H,S,D]`; row_max, row_sum f32 `[B,H,S]`), and compares each against a
//! plain-Rust f32 CPU forward reference.
//!
//! ## Config + invariants
//!
//! Per the V-B design (`2026-05-22-tier-b2-phase2.6-v-B-save-emission-design.md`
//! §2.1, §7, §8):
//!
//! * **Single-block invariant (§2.1):** B.1 currently emits a SINGLE kv_iter,
//!   so it is numerically complete ONLY when `block_kv >= seq_len`. We use
//!   `bq=bkv=hd=32, seq_len=32` so the whole sequence is one block.
//! * **R1 exercise (§8):** `B=H=2` (4 CTAs) confirms the saves' per-(b,h) HBM
//!   write addressing (`((b*H+h)*S+s)*D+c`). Note the B.1 projection load
//!   addresses for x/Wq/Wk/Wv are CTA-INDEPENDENT (they carry no batch_idx /
//!   head_idx offset — see `projection_mma.rs:836-846`), so all 4 CTAs read
//!   the SAME x and SAME W and therefore project the SAME q/k/v. The test
//!   value of B=H=2 is thus confirming the *write* addressing places identical
//!   projections into 4 DISTINCT (b,h) regions without scrambling/overlap —
//!   a wrong per-(b,h) address would corrupt one or more regions. The
//!   multi-q-tile R1 path (`block_q < seq_len`) cannot be tested in this
//!   single-block regime without the B1.6 multi-iter loop; that limitation is
//!   documented and out of scope for T2.7.
//! * **R2 exercise (§8):** V is stored COL-major in SMEM and re-staged to
//!   row-major HBM by a swapped read index. This is the highest-risk path; V
//!   is compared at FULL tolerance (NOT a relaxed band).
//!
//! ## RMSNorm convention
//!
//! The kernel-under-test runs with `skip_rmsnorm_prologue=true` (production
//! B.1 mode — the host pre-pass normalizes x before launch). So we feed the
//! kernel x that is ALREADY RMSNormalized + narrowed to f16 + chunkified, and
//! the CPU reference does plain projections on the SAME f16-narrowed x_norm —
//! no RMSNorm in the reference. This matches production and the existing
//! `tier_b1_full_kernel_e2e.rs` harness exactly.
//!
//! ## Running
//!
//! ```bash
//! cargo test -p nsl-codegen --features cuda \
//!     --test tier_b1_save_activations_gpu -- --ignored --nocapture
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    flash_attention_kernel_name_v2, synthesize_flash_attention_ptx_v2,
};
use std::ffi::{c_void, CString};

use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
};

extern "C" {
    fn nsl_kernel_launch(
        ptx_ptr: i64,
        name_ptr: i64,
        grid_x: i64,
        grid_y: i64,
        grid_z: i64,
        block_x: i64,
        block_y: i64,
        block_z: i64,
        args_ptr: i64,
        num_args: i64,
        shared_mem_bytes: i64,
    ) -> i64;
}

// -- f16 / f32 helpers (identical to tier_b1_full_kernel_e2e.rs) -------------

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e: i32 = -1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let e = (127 + e - 14) as u32;
            (sign << 31) | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        (sign << 31) | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

fn f32_to_f16_bits(x: f32) -> u16 {
    if x.is_nan() {
        return 0x7E00;
    }
    let bits = x.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 255 {
        return ((sign << 15) | 0x7C00 | if mant != 0 { 0x200 } else { 0 }) as u16;
    }
    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        let shift = (1 - exp_f16).min(24) as u32;
        let shifted = (mant | 0x800000) >> shift;
        let rounded = (shifted + 0x1000) >> 13;
        return ((sign << 15) | rounded) as u16;
    }
    if exp_f16 >= 31 {
        return ((sign << 15) | 0x7C00) as u16;
    }
    let mant16 = (mant + 0x1000) >> 13;
    let overflow = (mant16 >> 10) & 1;
    let exp16 = (exp_f16 as u32 + overflow) & 0x1F;
    ((sign << 15) | (exp16 << 10) | (mant16 & 0x3FF)) as u16
}

/// Round-trip f32 through f16 (what the kernel sees once x_norm/W are narrowed
/// for cp.async). The CPU reference's projection dot-products consume these
/// narrowed values so the comparison isolates the SAVE addressing/transpose,
/// not the f16 quantization (which both sides incur identically).
fn narrow(x: f32) -> f32 {
    f16_to_f32(f32_to_f16_bits(x))
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    let rc = nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {rc}");
        return false;
    }
    true
}

// CSHA tiered tolerance for projections (O(sqrt(d)·eps_f16)): 5e-3 for hd<=32.
fn tol_for_head_dim(hd: u32) -> f32 {
    if hd >= 128 {
        4e-2
    } else if hd >= 64 {
        2e-2
    } else {
        5e-3
    }
}

// -- Layout transforms (host side, matches kernel cp.async expectations) -----
// Identical to tier_b1_full_kernel_e2e.rs.

fn rmsnorm_rows_full_dmodel(x: &[f32], seq: usize, d_model: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; seq * d_model];
    for s in 0..seq {
        let row = &x[s * d_model..(s + 1) * d_model];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / d_model as f32;
        let rms = (mean_sq + eps).sqrt();
        for d in 0..d_model {
            out[s * d_model + d] = row[d] / rms;
        }
    }
    out
}

/// `x[seq, d_model]` row-major f32 -> `[d_model/chunk, seq, chunk]` f16.
fn x_to_chunks_major_f16(x: &[f32], seq: usize, d_model: usize, chunk: usize) -> Vec<u16> {
    assert!(d_model % chunk == 0, "d_model must be divisible by chunk");
    let n_chunks = d_model / chunk;
    let mut out = vec![0u16; n_chunks * seq * chunk];
    for chunk_idx in 0..n_chunks {
        for s in 0..seq {
            for c in 0..chunk {
                let d_col = chunk_idx * chunk + c;
                out[chunk_idx * seq * chunk + s * chunk + c] =
                    f32_to_f16_bits(x[s * d_model + d_col]);
            }
        }
    }
    out
}

/// `w[d_model, hd]` row-major f32 -> `[d_model/chunk, hd, chunk]` col-major-
/// within-chunk f16.
fn w_to_col_major_chunked_f16(w: &[f32], d_model: usize, hd: usize, chunk: usize) -> Vec<u16> {
    assert!(d_model % chunk == 0, "d_model must be divisible by chunk");
    let n_chunks = d_model / chunk;
    let mut out = vec![0u16; n_chunks * hd * chunk];
    for chunk_idx in 0..n_chunks {
        for n in 0..hd {
            for k_in_chunk in 0..chunk {
                let d_row = chunk_idx * chunk + k_in_chunk;
                out[chunk_idx * hd * chunk + n * chunk + k_in_chunk] =
                    f32_to_f16_bits(w[d_row * hd + n]);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// CPU forward reference for the 5 saves.
// ---------------------------------------------------------------------------

struct SaveRef {
    q_proj: Vec<f32>, // [B,H,S,D] row-major
    k_proj: Vec<f32>,
    v_proj: Vec<f32>,
    row_max: Vec<f32>, // [B,H,S]
    row_sum: Vec<f32>,
}

/// Replicate B.1's forward for the saves. `x_norm` is the already-RMSNormed
/// row-major `[seq, d_model]`; Wq/Wk/Wv are `[d_model, hd]` row-major. All
/// inputs are narrowed to f16-then-f32 so the projection dot products match
/// the kernel's f16 SMEM operands.
///
/// Because B.1's projection load is CTA-independent (no batch/head offset),
/// every (b,h) CTA projects the SAME q/k/v; the reference therefore computes
/// one (b,h) slice and replicates it into all `B*H` HBM regions. This mirrors
/// what the kernel writes, so the per-(b,h) save addressing (R1) is what the
/// comparison actually validates.
fn cpu_reference_saves(
    x_norm: &[f32],
    wq: &[f32],
    wk: &[f32],
    wv: &[f32],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
    d_model: usize,
) -> SaveRef {
    let scale = 1.0f32 / (hd as f32).sqrt();

    // Single (b,h) slice projections: proj[s, c] = sum_d x_norm[s,d]*W[d,c].
    let project = |w: &[f32]| -> Vec<f32> {
        let mut p = vec![0f32; seq * hd];
        for s in 0..seq {
            for c in 0..hd {
                let mut acc = 0f32;
                for d in 0..d_model {
                    acc += narrow(x_norm[s * d_model + d]) * narrow(w[d * hd + c]);
                }
                p[s * hd + c] = acc;
            }
        }
        p
    };
    let q1 = project(wq);
    let k1 = project(wk);
    let v1 = project(wv);

    // Softmax stats per query row: row_max = max_k S, row_sum = sum_k exp(S-max)
    // (LINEAR sum-of-exp, FA-2 convention). S[qi,ki] = scale * q1[qi,:].k1[ki,:].
    // The kernel-under-test config is NON-causal (canonical hd32 dQ config) so
    // every key participates.
    let mut rmax1 = vec![0f32; seq];
    let mut rsum1 = vec![0f32; seq];
    for qi in 0..seq {
        let s_row: Vec<f32> = (0..seq)
            .map(|ki| {
                let dot: f32 = (0..hd).map(|d| q1[qi * hd + d] * k1[ki * hd + d]).sum();
                dot * scale
            })
            .collect();
        let m = s_row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = s_row.iter().map(|&s| (s - m).exp()).sum();
        rmax1[qi] = m;
        rsum1[qi] = sum;
    }

    // Replicate the single (b,h) slice into all B*H regions.
    let bh = batch * heads;
    let mut q_proj = vec![0f32; bh * seq * hd];
    let mut k_proj = vec![0f32; bh * seq * hd];
    let mut v_proj = vec![0f32; bh * seq * hd];
    let mut row_max = vec![0f32; bh * seq];
    let mut row_sum = vec![0f32; bh * seq];
    for r in 0..bh {
        q_proj[r * seq * hd..(r + 1) * seq * hd].copy_from_slice(&q1);
        k_proj[r * seq * hd..(r + 1) * seq * hd].copy_from_slice(&k1);
        v_proj[r * seq * hd..(r + 1) * seq * hd].copy_from_slice(&v1);
        row_max[r * seq..(r + 1) * seq].copy_from_slice(&rmax1);
        row_sum[r * seq..(r + 1) * seq].copy_from_slice(&rsum1);
    }
    SaveRef { q_proj, k_proj, v_proj, row_max, row_sum }
}

// ---------------------------------------------------------------------------
// Config: bq=bkv=hd=32, seq=32 (single block), B=H=2, d_model=128
// (chunk=128 -> n_chunks=1). gpu_sm=120. save_activations_for_backward=true.
// ---------------------------------------------------------------------------

fn save_config(head_dim: i64) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim,
        causal: false, // single-block, no causal mask (every key participates)
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 128,
            skip_rmsnorm_prologue: true,
            save_activations_for_backward: true,
            ..CshaExtras::default()
        }),
    }
}

fn max_abs(a: &[f32], b: &[f32]) -> (f32, usize) {
    let mut m = 0f32;
    let mut idx = 0usize;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > m {
            m = d;
            idx = i;
        }
    }
    (m, idx)
}

#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_save_activations_match_cpu_reference_hd32() {
    run_save_validation(32);
}

/// hd=64 > block_kv=32 — exercises the softmax stat-register re-key fix
/// (tier_b1_reduced_stats SMEM region). Pre-fix this config failed ptxas with
/// `Unknown symbol %s_sum_1_lo` (tpw_pv=2 > tpw_qkt=1).
#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_save_activations_match_cpu_reference_hd64() {
    run_save_validation(64);
}

/// hd=128 > block_kv=32 — exercises the deepest divergence (tpw_pv=4 >
/// tpw_qkt=1; pre-fix `Unknown symbol %s_*_{1,2,3}_*`).
#[test]
#[ignore = "requires CUDA GPU"]
fn tier_b1_save_activations_match_cpu_reference_hd128() {
    run_save_validation(128);
}

fn run_save_validation(head_dim: i64) {
    if !cuda_available() {
        return;
    }

    let config = save_config(head_dim);
    let batch = 2usize;
    let heads = 2usize;
    let seq = config.block_q as usize; // single q-tile = single block
    let hd = config.head_dim as usize;
    // Single-head projection: W is [d_model, hd] (the kernel's projection
    // treats head_dim as the output width and ignores head_idx in the load).
    let kv_dim = hd;
    let d_model = config.csha.as_ref().unwrap().d_model as usize;
    // Use the chunk the kernel was actually synthesized with. `synthesize_*`
    // resolves the chunk internally via `chunk_config::select` (largest chunk
    // in {128,64,32,FLOOR} that fits the 99 KB SMEM budget). At hd<=64 this is
    // 128 (n_chunks=1); at hd=128 chunk=128 overflows so it downgrades to 64
    // (n_chunks=2). The host x/W chunkification MUST match the kernel's chunk
    // or the projection reads scrambled HBM (the projection math is otherwise
    // chunk-independent: it sums over the full d_model either way).
    let chunk = nsl_codegen::flash_attention_v2::tier_b1::chunk_config::select(&config)
        .expect("chunk_config::select must admit the save-activation config") as usize;
    eprintln!("[B1-save][hd{hd}] kernel chunk = {chunk} (d_model={d_model}, n_chunks={})", d_model / chunk);
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (hd as f32).sqrt();
    let bh = batch * heads;

    // ---- Host inputs (small deterministic) --------------------------------
    let mut x_host = vec![0f32; seq * d_model];
    for s in 0..seq {
        for d in 0..d_model {
            x_host[s * d_model + d] = (((s + d) as f32).sin() * 0.1) + 0.05;
        }
    }
    let mut wq_f32 = vec![0f32; d_model * kv_dim];
    let mut wk_f32 = vec![0f32; d_model * kv_dim];
    let mut wv_f32 = vec![0f32; d_model * kv_dim];
    for d in 0..d_model {
        for n in 0..kv_dim {
            wq_f32[d * kv_dim + n] = ((d * 7 + n) as f32).sin() * 0.05;
            wk_f32[d * kv_dim + n] = ((d * 11 + n) as f32).cos() * 0.05;
            wv_f32[d * kv_dim + n] = ((d * 13 + n) as f32).sin() * 0.05;
        }
    }

    // ---- CPU reference (on the SAME f16-narrowed x_norm + W) ---------------
    let x_normed = rmsnorm_rows_full_dmodel(&x_host, seq, d_model, norm_eps);
    let cpu = cpu_reference_saves(
        &x_normed, &wq_f32, &wk_f32, &wv_f32, batch, heads, seq, hd, d_model,
    );

    // ---- Pre-prepare GPU inputs in the kernel's layouts -------------------
    let x_chunked = x_to_chunks_major_f16(&x_normed, seq, d_model, chunk);
    let wq_chunked = w_to_col_major_chunked_f16(&wq_f32, d_model, kv_dim, chunk);
    let wk_chunked = w_to_col_major_chunked_f16(&wk_f32, d_model, kv_dim, chunk);
    let wv_chunked = w_to_col_major_chunked_f16(&wv_f32, d_model, kv_dim, chunk);
    let norm_weight = vec![1.0f32; d_model];

    // ---- Device allocations + H2D -----------------------------------------
    let x_bytes = (x_chunked.len() * 2) as i64;
    let w_bytes = (wq_chunked.len() * 2) as i64;
    let nw_bytes = (norm_weight.len() * 4) as i64;
    let qkv_elems = bh * seq * hd;
    let stat_elems = bh * seq;
    let qkv_f32_bytes = (qkv_elems * 4) as i64; // input q/k/v slots (unused; f32-sized like e2e)
    let out_bytes = (qkv_elems * 2) as i64; // O f16
    let lse_bytes = (stat_elems * 4) as i64;
    let save_qkv_bytes = (qkv_elems * 2) as i64; // q/k/v_proj f16 [B,H,S,D]
    let save_stat_bytes = (stat_elems * 4) as i64; // row_max/row_sum f32 [B,H,S]

    let x_dev = nsl_test_cuda_alloc(x_bytes);
    let wq_dev = nsl_test_cuda_alloc(w_bytes);
    let wk_dev = nsl_test_cuda_alloc(w_bytes);
    let wv_dev = nsl_test_cuda_alloc(w_bytes);
    let nw_dev = nsl_test_cuda_alloc(nw_bytes);
    let q_dev = nsl_test_cuda_alloc(qkv_f32_bytes);
    let k_dev = nsl_test_cuda_alloc(qkv_f32_bytes);
    let v_dev = nsl_test_cuda_alloc(qkv_f32_bytes);
    let out_dev = nsl_test_cuda_alloc(out_bytes);
    let lse_dev = nsl_test_cuda_alloc(lse_bytes);
    // The 5 save slots.
    let qproj_dev = nsl_test_cuda_alloc(save_qkv_bytes);
    let kproj_dev = nsl_test_cuda_alloc(save_qkv_bytes);
    let vproj_dev = nsl_test_cuda_alloc(save_qkv_bytes);
    let rmax_dev = nsl_test_cuda_alloc(save_stat_bytes);
    let rsum_dev = nsl_test_cuda_alloc(save_stat_bytes);

    let all_ptrs = [
        x_dev, wq_dev, wk_dev, wv_dev, nw_dev, q_dev, k_dev, v_dev, out_dev, lse_dev,
        qproj_dev, kproj_dev, vproj_dev, rmax_dev, rsum_dev,
    ];
    if !all_ptrs.iter().all(|&p| p != 0) {
        for &p in &all_ptrs {
            if p != 0 {
                nsl_test_cuda_free(p);
            }
        }
        panic!("device alloc returned null");
    }

    nsl_test_cuda_h2d(x_dev, x_chunked.as_ptr() as i64, x_bytes);
    nsl_test_cuda_h2d(wq_dev, wq_chunked.as_ptr() as i64, w_bytes);
    nsl_test_cuda_h2d(wk_dev, wk_chunked.as_ptr() as i64, w_bytes);
    nsl_test_cuda_h2d(wv_dev, wv_chunked.as_ptr() as i64, w_bytes);
    nsl_test_cuda_h2d(nw_dev, norm_weight.as_ptr() as i64, nw_bytes);

    // ---- PTX synthesis ----------------------------------------------------
    let mut ptx = synthesize_flash_attention_ptx_v2(&config);
    while ptx.last() == Some(&0) {
        ptx.pop();
    }
    if ptx.last() != Some(&b'\n') {
        ptx.push(b'\n');
    }
    let dump = std::env::temp_dir().join(format!("tier_b1_save_activations_hd{hd}.ptx"));
    std::fs::write(&dump, &ptx).ok();
    eprintln!("[B1-save][hd{hd}] PTX dumped to: {}", dump.display());
    ptx.push(0);

    let kernel_name = CString::new(flash_attention_kernel_name_v2(&config)).unwrap();
    eprintln!("[B1-save] kernel name: {kernel_name:?}");

    // ---- Build 37-arg list (mirrors nsl_flash_attention_csha_with_saves) --
    // Slot indices match the FFI launch list at flash_attention.rs:1073-1114.
    // The 5 save pointers occupy slots 30..34 (q_proj, k_proj, v_proj,
    // row_max, row_sum); x_raw at 35; segment_ids at 36.
    let mut q = q_dev as u64;
    let mut k = k_dev as u64;
    let mut v = v_dev as u64;
    let mut out = out_dev as u64;
    let mut s = scale;
    let mut b = batch as u64;
    let mut h = heads as u64;
    let mut sl = seq as u64;
    let mut hdv = hd as u64;
    let mut bt: u64 = 0;
    let mut kp: u64 = 0;
    let mut vp: u64 = 0;
    let mut bs: u64 = 0;
    let mut cos_ptr: u64 = 0;
    let mut sin_ptr: u64 = 0;
    let mut sids: u64 = 0;
    let mut slens: u64 = 0;
    let mut dfs_enter: u64 = 0;
    let mut dfs_exit: u64 = 0;
    let mut num_tree_nodes: u64 = 0;
    let mut lse = lse_dev as u64;
    let mut x = x_dev as u64;
    let mut nw = nw_dev as u64;
    let mut wq = wq_dev as u64;
    let mut wk = wk_dev as u64;
    let mut wv = wv_dev as u64;
    let mut wo: u64 = 0;
    let mut eps = norm_eps;
    let mut ah: u32 = 0;
    let mut dm = d_model as u32;
    let mut q_proj = qproj_dev as u64;
    let mut k_proj = kproj_dev as u64;
    let mut v_proj = vproj_dev as u64;
    let mut rmax = rmax_dev as u64;
    let mut rsum = rsum_dev as u64;
    let mut xraw: u64 = 0; // x_raw not saved by B.1 (skip_rmsnorm path)
    let mut seg_ids: u64 = 0;

    let args: [*mut c_void; 37] = [
        &mut q as *mut _ as *mut c_void,
        &mut k as *mut _ as *mut c_void,
        &mut v as *mut _ as *mut c_void,
        &mut out as *mut _ as *mut c_void,
        &mut s as *mut _ as *mut c_void,
        &mut b as *mut _ as *mut c_void,
        &mut h as *mut _ as *mut c_void,
        &mut sl as *mut _ as *mut c_void,
        &mut hdv as *mut _ as *mut c_void,
        &mut bt as *mut _ as *mut c_void,
        &mut kp as *mut _ as *mut c_void,
        &mut vp as *mut _ as *mut c_void,
        &mut bs as *mut _ as *mut c_void,
        &mut cos_ptr as *mut _ as *mut c_void,
        &mut sin_ptr as *mut _ as *mut c_void,
        &mut sids as *mut _ as *mut c_void,
        &mut slens as *mut _ as *mut c_void,
        &mut dfs_enter as *mut _ as *mut c_void,
        &mut dfs_exit as *mut _ as *mut c_void,
        &mut num_tree_nodes as *mut _ as *mut c_void,
        &mut lse as *mut _ as *mut c_void,
        &mut x as *mut _ as *mut c_void,
        &mut nw as *mut _ as *mut c_void,
        &mut wq as *mut _ as *mut c_void,
        &mut wk as *mut _ as *mut c_void,
        &mut wv as *mut _ as *mut c_void,
        &mut wo as *mut _ as *mut c_void,
        &mut eps as *mut _ as *mut c_void,
        &mut ah as *mut _ as *mut c_void,
        &mut dm as *mut _ as *mut c_void,
        &mut q_proj as *mut _ as *mut c_void, // slot 30
        &mut k_proj as *mut _ as *mut c_void, // slot 31
        &mut v_proj as *mut _ as *mut c_void, // slot 32
        &mut rmax as *mut _ as *mut c_void,   // slot 33
        &mut rsum as *mut _ as *mut c_void,   // slot 34
        &mut xraw as *mut _ as *mut c_void,   // slot 35
        &mut seg_ids as *mut _ as *mut c_void, // slot 36 (segment_ids; 0)
    ];

    // ---- Launch (grid = (ceil(seq/bq), B*H, 1); block = 256 / 8 warps) ----
    let grid_x = ((seq as i64) + config.block_q - 1) / config.block_q;
    let rc = unsafe {
        nsl_kernel_launch(
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            grid_x,
            bh as i64,
            1,
            256, 1, 1,
            args.as_ptr() as i64,
            args.len() as i64,
            0, // static SMEM (Tier B.1 declares shmem[N])
        )
    };
    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                std::ffi::CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no log>".into()
        };
        for &p in &all_ptrs {
            nsl_test_cuda_free(p);
        }
        panic!("Tier B.1 with-saves launch failed rc={rc}\nJIT log:\n{log}");
    }

    // ---- Readback the 5 saves + LSE ---------------------------------------
    let mut qproj_f16 = vec![0u16; qkv_elems];
    let mut kproj_f16 = vec![0u16; qkv_elems];
    let mut vproj_f16 = vec![0u16; qkv_elems];
    let mut rmax_gpu = vec![0f32; stat_elems];
    let mut rsum_gpu = vec![0f32; stat_elems];
    let mut lse_gpu = vec![0f32; stat_elems];
    nsl_test_cuda_d2h(qproj_f16.as_mut_ptr() as i64, qproj_dev, save_qkv_bytes);
    nsl_test_cuda_d2h(kproj_f16.as_mut_ptr() as i64, kproj_dev, save_qkv_bytes);
    nsl_test_cuda_d2h(vproj_f16.as_mut_ptr() as i64, vproj_dev, save_qkv_bytes);
    nsl_test_cuda_d2h(rmax_gpu.as_mut_ptr() as i64, rmax_dev, save_stat_bytes);
    nsl_test_cuda_d2h(rsum_gpu.as_mut_ptr() as i64, rsum_dev, save_stat_bytes);
    nsl_test_cuda_d2h(lse_gpu.as_mut_ptr() as i64, lse_dev, lse_bytes);
    for &p in &all_ptrs {
        nsl_test_cuda_free(p);
    }

    let qproj_gpu: Vec<f32> = qproj_f16.iter().map(|&b| f16_to_f32(b)).collect();
    let kproj_gpu: Vec<f32> = kproj_f16.iter().map(|&b| f16_to_f32(b)).collect();
    let vproj_gpu: Vec<f32> = vproj_f16.iter().map(|&b| f16_to_f32(b)).collect();

    // NaN guard.
    let n_nan = qproj_gpu.iter().chain(&kproj_gpu).chain(&vproj_gpu)
        .chain(&rmax_gpu).chain(&rsum_gpu)
        .filter(|v| !v.is_finite())
        .count();

    // ---- Compare ----------------------------------------------------------
    let tol_proj = tol_for_head_dim(hd as u32); // 5e-3 for hd=32
    // row_max is a `max.f32` select -> EXACT; hold it to a tight absolute
    // tolerance. row_sum is `sum_k ex2.approx.f32(...)` -> the kernel uses the
    // approximate exp instruction (attention_mma.rs STEP 3) where the CPU
    // reference uses exact `f32::exp`. `ex2.approx.f32` carries ~3.6e-7 max
    // relative error per term (CUDA PTX ISA), and summing ~S terms compounds
    // it; the observed worst REL err is ~6.6e-6. So row_sum is checked with a
    // RELATIVE tolerance keyed to ex2.approx, NOT an absolute one (an absolute
    // 1e-4 on a magnitude-~30 sum-of-exps is ~3e-6 relative -- below the
    // instruction's own noise floor). A real addressing/transposition bug
    // produces O(1) relative error (the pre-fix all-zero case was rel ~1.0),
    // so 5e-5 still catches every structural failure mode this gate guards.
    const TOL_ROW_MAX: f32 = 1e-4; // exact max -> tight absolute
    const TOL_ROW_SUM_REL: f32 = 5e-5; // ex2.approx sum -> relative
    let (q_max, q_idx) = max_abs(&qproj_gpu, &cpu.q_proj);
    let (k_max, k_idx) = max_abs(&kproj_gpu, &cpu.k_proj);
    let (v_max, v_idx) = max_abs(&vproj_gpu, &cpu.v_proj);
    let (rmax_max, _) = max_abs(&rmax_gpu, &cpu.row_max);
    let (rsum_max, _) = max_abs(&rsum_gpu, &cpu.row_sum);
    let rsum_rel = {
        let mut m = 0f32;
        for i in 0..stat_elems {
            let rel = (rsum_gpu[i] - cpu.row_sum[i]).abs() / cpu.row_sum[i].abs().max(1e-9);
            if rel > m {
                m = rel;
            }
        }
        m
    };

    // LSE internal-consistency cross-check (§7 step 6): LSE == row_max +
    // ln(row_sum), computed from the saved row_max/row_sum vs B.1's
    // independently-written LSE buffer. Catches a row_max/row_sum addressing
    // transposition even without the full CPU reference.
    let mut lse_max = 0f32;
    for i in 0..stat_elems {
        let expect = rmax_gpu[i] + rsum_gpu[i].ln();
        let d = (lse_gpu[i] - expect).abs();
        if d > lse_max {
            lse_max = d;
        }
    }

    // Per-(b,h)-region and per-column localization. On failure these signals
    // localize the root cause (see the assert comments below). Per-region
    // equality across all B*H regions confirms the save's per-(b,h) write
    // addressing (R1); a per-column band split localizes the failing tile.
    {
        let region = qkv_elems / bh; // elems per (b,h) = S*D
        let (r0, _) = max_abs(&qproj_gpu[..region], &cpu.q_proj[..region]);
        let mut regions_equal = true;
        for r in 1..bh {
            let lo = r * region;
            for i in 0..region {
                if qproj_gpu[lo + i].to_bits() != qproj_gpu[i].to_bits() {
                    regions_equal = false;
                    break;
                }
            }
        }
        eprintln!(
            "[B1-save][diag] q_proj region0 max_abs={r0:.3e}; all {bh} (b,h) regions byte-identical={regions_equal} \
             (identical => per-(b,h) write addressing R1 is correct)"
        );
        eprint!("[B1-save][diag] q region0 per-column-band(8) max_abs: ");
        for nt in 0..(hd / 8) {
            let mut m = 0f32;
            for s in 0..seq {
                for d in (nt * 8)..(nt * 8 + 8) {
                    m = m.max((qproj_gpu[s * hd + d] - cpu.q_proj[s * hd + d]).abs());
                }
            }
            eprint!("cols[{}..{}]={m:.3e} ", nt * 8, nt * 8 + 8);
        }
        eprintln!();
    }

    eprintln!("\n[B1-save] ── numerical summary (B={batch} H={heads} S={seq} D={hd}) ──");
    eprintln!("[B1-save] non-finite save elems = {n_nan}");
    eprintln!("[B1-save] q_proj  max_abs = {q_max:.4e}  (tol {tol_proj:.1e})  worst idx={q_idx}");
    eprintln!("[B1-save] k_proj  max_abs = {k_max:.4e}  (tol {tol_proj:.1e})  worst idx={k_idx}");
    eprintln!("[B1-save] v_proj  max_abs = {v_max:.4e}  (tol {tol_proj:.1e}, FULL — R2)  worst idx={v_idx}");
    eprintln!("[B1-save] row_max max_abs = {rmax_max:.4e}  (tol {TOL_ROW_MAX:.1e}, exact max)");
    eprintln!("[B1-save] row_sum max_abs = {rsum_max:.4e}  (abs, informational)  REL = {rsum_rel:.3e}  (tol {TOL_ROW_SUM_REL:.1e}, ex2.approx)");
    eprintln!("[B1-save] LSE==row_max+ln(row_sum) cross-check max_abs = {lse_max:.4e}");
    eprintln!("[B1-save] # nonzero row_max={}, row_sum={}, lse={} (expect {stat_elems} each)",
        rmax_gpu.iter().filter(|&&v| v != 0.0).count(),
        rsum_gpu.iter().filter(|&&v| v != 0.0).count(),
        lse_gpu.iter().filter(|&&v| v != 0.0).count());
    eprintln!("[B1-save] q_proj[0..4] gpu={:?} cpu={:?}",
        &qproj_gpu[..4.min(qproj_gpu.len())], &cpu.q_proj[..4.min(cpu.q_proj.len())]);
    eprintln!("[B1-save] v_proj[0..4] gpu={:?} cpu={:?}",
        &vproj_gpu[..4.min(vproj_gpu.len())], &cpu.v_proj[..4.min(cpu.v_proj.len())]);

    assert_eq!(n_nan, 0, "[B1-save] FAIL: {n_nan} non-finite saved elements");

    // DO NOT relax these tolerances to make this pass. A failure here
    // localizes a real retrofit bug (see the module/diagnosis notes in the
    // task):
    //   * V fails while Q/K pass     -> R2 col-major SMEM read index is wrong.
    //   * K fails but Q passes       -> R1 absolute-K seq-index rule is wrong.
    //   * row_max/row_sum/LSE ALL 0  -> the LSE null-guard predicate
    //     (%p_has_lse) is never set in B.1's native finalize (T2.9 fix:
    //     finalize.rs now mirrors Tier A's `setp.ne.u64 %p_has_lse,
    //     %logsumexp_base, 0`). A real addressing/transposition bug shows up
    //     as O(1) RELATIVE error, far above the ex2.approx noise floor below.
    assert!(q_max <= tol_proj, "q_proj max_abs {q_max:.4e} > tol {tol_proj:.1e}");
    assert!(k_max <= tol_proj, "k_proj (R1 abs-seq) max_abs {k_max:.4e} > tol {tol_proj:.1e}");
    assert!(v_max <= tol_proj, "v_proj (R2 col-major) max_abs {v_max:.4e} > tol {tol_proj:.1e}");
    assert!(rmax_max <= TOL_ROW_MAX, "row_max max_abs {rmax_max:.4e} > tol {TOL_ROW_MAX:.1e}");
    // row_sum: relative tolerance keyed to ex2.approx.f32 (see the const decl).
    assert!(
        rsum_rel <= TOL_ROW_SUM_REL,
        "row_sum REL err {rsum_rel:.4e} > tol {TOL_ROW_SUM_REL:.1e} \
         (abs max_abs {rsum_max:.4e}) -- ex2.approx noise is ~6.6e-6; an O(1) \
         relative error here means a real addressing/transposition bug"
    );
    assert!(
        lse_max <= 1e-3,
        "LSE consistency (row_max+ln(row_sum)) max_abs {lse_max:.4e} > 1e-3 \
         -- row_max/row_sum or LSE addressing transposed (or %p_has_lse unset)"
    );

    eprintln!("\n[B1-save][hd{hd}] PASS: all 5 saves match CPU reference; LSE consistent.");
}
