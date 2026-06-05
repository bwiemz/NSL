//! CPU-side B.1 -> row-major adapter (Phase 2.6 T6+T7).
//!
//! Per V-Phase-2.6-A (audit) + the B.1 save-activation retrofit
//! (commits 7c00ae1c / cda47e40 / 9e2148f1) + T2.7 GPU validation: B.1 writes
//! ALL save-activation tensors in row-major layout, byte-identical to the
//! dQ-kernel's read convention. So this adapter is IDENTITY passthrough --
//! it validates lengths and moves the buffers into a ForwardOutputs.
//!
//! (The original Phase 2.6 plan anticipated a chunked B.1 layout requiring
//! an index remap; the retrofit targets row-major directly, so no remap is
//! needed. If a future B.1 layout change reintroduces chunking, the per-tensor
//! remap would live here.)

use half::f16;
use crate::cpu_naive_forward::ForwardOutputs;

/// Raw d2h bytes from B.1's `_with_saves` slots, row-major (post-retrofit).
pub struct B1Saves {
    pub q_proj:  Vec<f16>,
    pub k_proj:  Vec<f16>,
    pub v_proj:  Vec<f16>,
    pub row_max: Vec<f32>,
    pub row_sum: Vec<f32>,
    pub o:       Vec<f16>,
}

/// Move B.1's row-major saves into a ForwardOutputs (identity passthrough,
/// length-validated). `batch/heads/seq/hd` give the expected lengths.
pub fn reshape_b1_saves_to_row_major(
    saves: B1Saves,
    batch: usize, heads: usize, seq: usize, hd: usize,
) -> ForwardOutputs {
    let proj_len = batch * heads * seq * hd;
    let stat_len = batch * heads * seq;
    assert_eq!(saves.q_proj.len(), proj_len, "q_proj length: got {} expected {}", saves.q_proj.len(), proj_len);
    assert_eq!(saves.k_proj.len(), proj_len, "k_proj length: got {} expected {}", saves.k_proj.len(), proj_len);
    assert_eq!(saves.v_proj.len(), proj_len, "v_proj length: got {} expected {}", saves.v_proj.len(), proj_len);
    assert_eq!(saves.row_max.len(), stat_len, "row_max length: got {} expected {}", saves.row_max.len(), stat_len);
    assert_eq!(saves.row_sum.len(), stat_len, "row_sum length: got {} expected {}", saves.row_sum.len(), stat_len);
    assert_eq!(saves.o.len(), proj_len, "o length: got {} expected {}", saves.o.len(), proj_len);
    ForwardOutputs {
        q_saved: saves.q_proj,
        k_saved: saves.k_proj,
        v_saved: saves.v_proj,
        row_max: saves.row_max,
        row_sum: saves.row_sum,
        o:       saves.o,
    }
}

// ===== Phase 2.6 T9 — GPU launcher: run B.1-with-saves and adapt =============
//
// Ports the proven launch from
// `crates/nsl-codegen/tests/tier_b1_save_activations_gpu.rs` (the T2.7
// validation gate). Builds a dedicated B.1-forward launch config with
// `block_kv >= seq` (single-block precondition), RMSNorms + chunkifies the raw
// f16 inputs the way the kernel's cp.async expects, launches via the 37-arg
// `nsl_kernel_launch`, reads back the 5 saves + O, and reshapes to row-major.

/// f16 bits -> f32 (matches `tier_b1_save_activations_gpu.rs::f16_to_f32`).
#[cfg(feature = "cuda")]
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

/// f32 -> f16 bits (matches `tier_b1_save_activations_gpu.rs::f32_to_f16_bits`).
#[cfg(feature = "cuda")]
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

/// Per-row RMSNorm over the full `d_model` (matches T2.7's
/// `rmsnorm_rows_full_dmodel`). `x` is `[seq, d_model]` row-major f32.
#[cfg(feature = "cuda")]
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

/// `x[seq, d_model]` row-major f32 -> `[d_model/chunk, seq, chunk]` f16
/// (matches T2.7's `x_to_chunks_major_f16`).
#[cfg(feature = "cuda")]
fn x_to_chunks_major_f16(x: &[f32], seq: usize, d_model: usize, chunk: usize) -> Vec<u16> {
    assert!(d_model.is_multiple_of(chunk), "d_model must be divisible by chunk");
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
/// within-chunk f16 (matches T2.7's `w_to_col_major_chunked_f16`).
#[cfg(feature = "cuda")]
fn w_to_col_major_chunked_f16(w: &[f32], d_model: usize, hd: usize, chunk: usize) -> Vec<u16> {
    assert!(d_model.is_multiple_of(chunk), "d_model must be divisible by chunk");
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

#[cfg(feature = "cuda")]
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

/// Launch Tier B.1 forward (with saves) on the GPU and adapt the readback to a
/// row-major `ForwardOutputs`.
///
/// `x` is raw row-major `[B,S,d_model]` f16; `wq/wk/wv` are `[d_model, H*D]`
/// row-major f16; `norm_weight` is `[d_model]` f16. This fn RMSNorms x in f32,
/// narrows + chunkifies (x to chunks-major, W to col-major-chunked), launches
/// the kernel, then d2h's q_proj/k_proj/v_proj (f16), row_max/row_sum (f32),
/// and O (f16). Test defaults: B=1, H=1.
///
/// `seq` is passed explicitly. B.1 forward is numerically complete only for a
/// single kv_iter (block_kv >= seq), so the launcher sets block_q=block_kv=seq
/// and asserts seq <= 32 (the SMEM-safe single-block max for this test config).
#[cfg(feature = "cuda")]
pub fn run_b1_forward_and_adapt(
    x: &[f16],
    wq: &[f16],
    wk: &[f16],
    wv: &[f16],
    norm_weight: &[f16],
    cfg: &nsl_codegen::flash_attention::FlashAttentionConfig,
    seq: usize,
) -> ForwardOutputs {
    use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig};
    use nsl_codegen::flash_attention_v2::{
        flash_attention_kernel_name_v2, synthesize_flash_attention_ptx_v2,
    };
    use nsl_runtime::{
        nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free,
        nsl_test_cuda_h2d, nsl_test_cuda_jit_log,
    };
    use std::ffi::{c_void, CStr, CString};

    // ---- 1. Dimensions (test defaults) ------------------------------------
    let batch = 1usize;
    let heads = 1usize;
    // B.1 single-block precondition: block_kv must cover the whole sequence in
    // one kv_iter. The closure gate runs at seq=32.
    assert!(
        seq <= 32,
        "run_b1_forward_and_adapt: seq {seq} exceeds the single-block max (32)"
    );
    let hd = cfg.head_dim as usize;
    let d_model = cfg.csha.as_ref().map(|c| c.d_model as usize).unwrap_or(128);
    let kv_dim = hd; // single-head projection: W is [d_model, hd]
    let bh = batch * heads;
    let norm_eps = 1e-5f32;
    let scale = 1.0f32 / (hd as f32).sqrt();

    // ---- Input length validation ------------------------------------------
    assert_eq!(x.len(), batch * seq * d_model, "x length");
    assert_eq!(wq.len(), d_model * heads * hd, "wq length");
    assert_eq!(wk.len(), d_model * heads * hd, "wk length");
    assert_eq!(wv.len(), d_model * heads * hd, "wv length");
    assert_eq!(norm_weight.len(), d_model, "norm_weight length");

    // ---- 2. B.1-forward launch config (single-block: block_kv >= seq) -----
    // B.1 forward is numerically complete only when block_kv >= seq_len (single
    // kv_iter), so force block_q=block_kv=seq. This is a DIFFERENT kernel launch
    // from the dQ-kernel's cfg (which may carry bq=64) — clone cfg and override
    // the tiling + save flags.
    let mut b1_csha = cfg.csha.clone().unwrap_or_else(|| CshaExtras {
        level: 2,
        d_model: d_model as u32,
        ..CshaExtras::default()
    });
    b1_csha.skip_rmsnorm_prologue = true;
    b1_csha.save_activations_for_backward = true;
    if b1_csha.d_model == 0 {
        b1_csha.d_model = d_model as u32;
    }
    let b1_cfg = FlashAttentionConfig {
        block_q: seq as i64,
        block_kv: seq as i64,
        head_dim: hd as i64,
        causal: cfg.causal,
        paged: false,
        rope_q: false,
        rope_style: cfg.rope_style,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: cfg.gpu_sm,
        segment_masked: false,
        csha: Some(b1_csha),
    };
    assert!(
        seq <= b1_cfg.block_kv as usize,
        "B.1 single-block precondition violated: seq {seq} > block_kv {}",
        b1_cfg.block_kv
    );
    // chunk MUST match what the kernel resolves via chunk_config::select for
    // b1_cfg. At hd=128, chunk downgrades from d_model (SMEM budget), so
    // hardcoding chunk=d_model would scramble the chunkified HBM layout the
    // kernel reads (the exact hd=128 mismatch found in tier_b1_save_activations_gpu).
    let chunk = nsl_codegen::flash_attention_v2::tier_b1::chunk_config::select(&b1_cfg)
        .expect("chunk_config::select must admit the B.1-forward single-block config")
        as usize;

    // ---- 3. f16 -> f32, RMSNorm, narrow + chunkify ------------------------
    let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let wq_f32: Vec<f32> = wq.iter().map(|v| v.to_f32()).collect();
    let wk_f32: Vec<f32> = wk.iter().map(|v| v.to_f32()).collect();
    let wv_f32: Vec<f32> = wv.iter().map(|v| v.to_f32()).collect();

    let x_normed = rmsnorm_rows_full_dmodel(&x_f32, seq, d_model, norm_eps);
    let x_chunked = x_to_chunks_major_f16(&x_normed, seq, d_model, chunk);
    let wq_chunked = w_to_col_major_chunked_f16(&wq_f32, d_model, kv_dim, chunk);
    let wk_chunked = w_to_col_major_chunked_f16(&wk_f32, d_model, kv_dim, chunk);
    let wv_chunked = w_to_col_major_chunked_f16(&wv_f32, d_model, kv_dim, chunk);
    let norm_weight_f32: Vec<f32> = norm_weight.iter().map(|v| v.to_f32()).collect();

    // ---- 4. CUDA init + device allocations + H2D --------------------------
    let rc_init = nsl_cuda_init();
    assert_eq!(rc_init, 0, "nsl_cuda_init failed rc={rc_init}");

    let x_bytes = (x_chunked.len() * 2) as i64;
    let w_bytes = (wq_chunked.len() * 2) as i64;
    let nw_bytes = (norm_weight_f32.len() * 4) as i64;
    let qkv_elems = bh * seq * hd;
    let stat_elems = bh * seq;
    let qkv_f32_bytes = (qkv_elems * 4) as i64; // unused q/k/v input slots (f32-sized)
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
    let qproj_dev = nsl_test_cuda_alloc(save_qkv_bytes);
    let kproj_dev = nsl_test_cuda_alloc(save_qkv_bytes);
    let vproj_dev = nsl_test_cuda_alloc(save_qkv_bytes);
    let rmax_dev = nsl_test_cuda_alloc(save_stat_bytes);
    let rsum_dev = nsl_test_cuda_alloc(save_stat_bytes);

    let all_ptrs = [
        x_dev, wq_dev, wk_dev, wv_dev, nw_dev, q_dev, k_dev, v_dev, out_dev, lse_dev, qproj_dev,
        kproj_dev, vproj_dev, rmax_dev, rsum_dev,
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
    nsl_test_cuda_h2d(nw_dev, norm_weight_f32.as_ptr() as i64, nw_bytes);

    // ---- 5. PTX synthesis -------------------------------------------------
    let mut ptx = synthesize_flash_attention_ptx_v2(&b1_cfg);
    while ptx.last() == Some(&0) {
        ptx.pop();
    }
    if ptx.last() != Some(&b'\n') {
        ptx.push(b'\n');
    }
    ptx.push(0);
    let kernel_name = CString::new(flash_attention_kernel_name_v2(&b1_cfg)).unwrap();

    // ---- 6. Build the 37-arg list (saves at slots 30..34) -----------------
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
    let mut x_arg = x_dev as u64;
    let mut nw = nw_dev as u64;
    let mut wq_arg = wq_dev as u64;
    let mut wk_arg = wk_dev as u64;
    let mut wv_arg = wv_dev as u64;
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
        &mut x_arg as *mut _ as *mut c_void,
        &mut nw as *mut _ as *mut c_void,
        &mut wq_arg as *mut _ as *mut c_void,
        &mut wk_arg as *mut _ as *mut c_void,
        &mut wv_arg as *mut _ as *mut c_void,
        &mut wo as *mut _ as *mut c_void,
        &mut eps as *mut _ as *mut c_void,
        &mut ah as *mut _ as *mut c_void,
        &mut dm as *mut _ as *mut c_void,
        &mut q_proj as *mut _ as *mut c_void,  // slot 30
        &mut k_proj as *mut _ as *mut c_void,  // slot 31
        &mut v_proj as *mut _ as *mut c_void,  // slot 32
        &mut rmax as *mut _ as *mut c_void,    // slot 33
        &mut rsum as *mut _ as *mut c_void,    // slot 34
        &mut xraw as *mut _ as *mut c_void,    // slot 35
        &mut seg_ids as *mut _ as *mut c_void, // slot 36
    ];

    // ---- 7. Launch (grid = (ceil(seq/bq), B*H, 1); block = 256) -----------
    let grid_x = ((seq as i64) + b1_cfg.block_q - 1) / b1_cfg.block_q;
    let rc = unsafe {
        nsl_kernel_launch(
            ptx.as_ptr() as i64,
            kernel_name.as_ptr() as i64,
            grid_x,
            bh as i64,
            1,
            256,
            1,
            1,
            args.as_ptr() as i64,
            args.len() as i64,
            0, // static SMEM (Tier B.1 declares shmem[N])
        )
    };
    if rc != 0 {
        let log_ptr = nsl_test_cuda_jit_log(ptx.as_ptr() as i64);
        let log = if log_ptr != 0 {
            unsafe {
                CStr::from_ptr(log_ptr as *const i8)
                    .to_string_lossy()
                    .into_owned()
            }
        } else {
            "<no log>".into()
        };
        for &p in &all_ptrs {
            nsl_test_cuda_free(p);
        }
        panic!("run_b1_forward_and_adapt: launch failed rc={rc}\nJIT log:\n{log}");
    }

    // ---- 8. Readback (5 saves + O) ----------------------------------------
    let mut qproj_f16 = vec![0u16; qkv_elems];
    let mut kproj_f16 = vec![0u16; qkv_elems];
    let mut vproj_f16 = vec![0u16; qkv_elems];
    let mut rmax_gpu = vec![0f32; stat_elems];
    let mut rsum_gpu = vec![0f32; stat_elems];
    let mut o_f16 = vec![0u16; qkv_elems];
    nsl_test_cuda_d2h(qproj_f16.as_mut_ptr() as i64, qproj_dev, save_qkv_bytes);
    nsl_test_cuda_d2h(kproj_f16.as_mut_ptr() as i64, kproj_dev, save_qkv_bytes);
    nsl_test_cuda_d2h(vproj_f16.as_mut_ptr() as i64, vproj_dev, save_qkv_bytes);
    nsl_test_cuda_d2h(rmax_gpu.as_mut_ptr() as i64, rmax_dev, save_stat_bytes);
    nsl_test_cuda_d2h(rsum_gpu.as_mut_ptr() as i64, rsum_dev, save_stat_bytes);
    nsl_test_cuda_d2h(o_f16.as_mut_ptr() as i64, out_dev, out_bytes);

    for &p in &all_ptrs {
        nsl_test_cuda_free(p);
    }

    // ---- 9. f16 bits -> half::f16; assemble B1Saves -----------------------
    let to_f16 = |bits: u16| f16::from_f32(f16_to_f32(bits));
    let saves = B1Saves {
        q_proj: qproj_f16.iter().map(|&b| to_f16(b)).collect(),
        k_proj: kproj_f16.iter().map(|&b| to_f16(b)).collect(),
        v_proj: vproj_f16.iter().map(|&b| to_f16(b)).collect(),
        row_max: rmax_gpu,
        row_sum: rsum_gpu,
        o: o_f16.iter().map(|&b| to_f16(b)).collect(),
    };

    reshape_b1_saves_to_row_major(saves, batch, heads, seq, hd)
}

/// CPU-only build: the B1Forward path requires a GPU.
#[cfg(not(feature = "cuda"))]
pub fn run_b1_forward_and_adapt(
    _x: &[f16],
    _wq: &[f16],
    _wk: &[f16],
    _wv: &[f16],
    _norm_weight: &[f16],
    _cfg: &nsl_codegen::flash_attention::FlashAttentionConfig,
    _seq: usize,
) -> ForwardOutputs {
    panic!("run_b1_forward_and_adapt requires feature='cuda'")
}
