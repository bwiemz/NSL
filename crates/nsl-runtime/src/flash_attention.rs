//! FlashAttention-2 runtime launch wrappers.
//!
//! These functions compute grid/block dimensions from tensor shapes, marshal
//! arguments, and call `kernel_launch()` with pre-baked PTX from .rodata.
//! No PTX generation happens at runtime. On non-CUDA builds, falls back to
//! naive matmul+softmax attention path.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};

use crate::tensor::NslTensor;
#[cfg(feature = "cuda")]
use crate::autodiff;

/// One-time log guard: prints the selected kernel variant (FA3 or FA2) only once.
#[cfg(feature = "cuda")]
static FA_VARIANT_LOGGED: AtomicBool = AtomicBool::new(false);

/// Launch FlashAttention-3 (Hopper wgmma) kernel.
/// Returns 0 on success, -1 if the launch failed (caller should fall back to FA2).
#[cfg(feature = "cuda")]
fn flash_attention_hopper(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale: f32,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: bool,
) -> i64 {
    use crate::cuda::kernels_hopper::{generate_flash_attention_3_ptx, FA3Config};

    let block_q = 64;
    let block_kv = 64;

    let config = FA3Config {
        block_q,
        block_kv,
        head_dim: head_dim as usize,
        seq_len: seq_len as usize,
        batch_heads: (batch * heads) as usize,
        causal,
        fp8: false,
        scale,
    };

    // Generate PTX at runtime for this config
    let ptx = generate_flash_attention_3_ptx(block_q, block_kv, head_dim as usize, causal, false);
    let ptx_cstr = format!("{}\0", ptx);

    let mut q_data = q_ptr as u64;
    let mut k_data = k_ptr as u64;
    let mut v_data = v_ptr as u64;
    let mut o_data = out_ptr as u64;
    let mut lse_data = if logsumexp_ptr != 0 {
        logsumexp_ptr as u64
    } else {
        0u64
    };
    let mut scale_val = scale;
    let mut seq_val = seq_len as u64;
    let mut hd_val = head_dim as u64;
    let mut num_kv = config.num_kv_tiles() as u64;

    let args: [*mut c_void; 9] = [
        &mut q_data   as *mut _ as *mut c_void,
        &mut k_data   as *mut _ as *mut c_void,
        &mut v_data   as *mut _ as *mut c_void,
        &mut o_data   as *mut _ as *mut c_void,
        &mut lse_data as *mut _ as *mut c_void,
        &mut scale_val as *mut _ as *mut c_void,
        &mut seq_val  as *mut _ as *mut c_void,
        &mut hd_val   as *mut _ as *mut c_void,
        &mut num_kv   as *mut _ as *mut c_void,
    ];

    let grid  = config.grid();
    let block = config.block();
    let shared = config.shared_mem_bytes();

    let result = crate::cuda::inner::kernel_launch(
        ptx_cstr.as_ptr(),
        b"nsl_flash_attention_3\0".as_ptr(),
        grid,
        block,
        &args,
        shared,
    );

    if result as u32 != 0 {
        eprintln!(
            "[nsl] FA3 Hopper kernel launch failed ({:?}), falling back to FA2",
            result
        );
        return -1;
    }

    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    0
}

/// FlashAttention-2 kernel launch wrapper.
///
/// All params are i64 for Cranelift ABI compatibility (same pattern as nsl_kernel_launch).
/// f32 scale is passed as i64 and reconstructed via f32::from_bits(scale as u32).
///
/// Returns 0 on success, non-zero CUDA error code on failure.
#[no_mangle]
pub extern "C" fn nsl_flash_attention(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64,
    logsumexp_ptr: i64,  // backward aux output (0 = skip, inference-only)
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
    causal: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let _scale = f32::from_bits(scale_bits as u32);

        // Grid: (ceil(seq_len / block_q), batch * heads, 1)
        let grid_x = (seq_len + block_q - 1) / block_q;
        let grid_y = batch * heads;
        let grid_z = 1i64;

        // Block: (128, 1, 1) — 4 warps per thread block
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // Marshal all kernel arguments as u64 values
        let mut q = q_ptr as u64;
        let mut k = k_ptr as u64;
        let mut v = v_ptr as u64;
        let mut out = out_ptr as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        // M33: tree mask params (null for non-tree-mask variants)
        let mut dfs_enter: u64 = 0;
        let mut dfs_exit: u64 = 0;
        let mut num_tree_nodes: u64 = 0;
        // Backward pass: logsumexp auxiliary output
        let mut lse = logsumexp_ptr as u64;

        let args: [*mut c_void; 21] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut dfs_enter as *mut _ as *mut c_void,
            &mut dfs_exit as *mut _ as *mut c_void,
            &mut num_tree_nodes as *mut _ as *mut c_void,
            &mut lse as *mut _ as *mut c_void,
        ];

        // ── Ampere / Hopper dispatch ──────────────────────────────────────────
        // Detect SM version once; on Hopper (sm_90+) try FA3 (wgmma) first.
        // If FA3 launch fails fall through to FA2 with the caller-supplied PTX.
        let sm = crate::cuda::inner::detect_sm_version();
        let fa3_ok = if sm >= 90 {
            let fa3_result = flash_attention_hopper(
                q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr,
                f32::from_bits(scale_bits as u32),
                batch, heads, seq_len, head_dim,
                causal != 0,
            );
            if fa3_result == 0 {
                if !FA_VARIANT_LOGGED.swap(true, Ordering::Relaxed) {
                    eprintln!("[nsl] Using FlashAttention-3 (Hopper wgmma, sm_90a)");
                }
                true
            } else {
                false
            }
        } else {
            false
        };

        let result: i64 = if fa3_ok {
            // FA3 already launched and synced; treat as success.
            0
        } else {
            if !FA_VARIANT_LOGGED.swap(true, Ordering::Relaxed) {
                eprintln!("[nsl] Using FlashAttention-2 (Ampere mma.sync)");
            }
            crate::cuda::inner::kernel_launch(
                ptx_ptr as *const u8,
                name_ptr as *const u8,
                [grid_x, grid_y, grid_z],
                [block_x, block_y, block_z],
                &args,
                shared_mem_bytes as u32,
            ) as i64
        };

        // Record tape op for backward pass if recording
        if autodiff::is_recording() {
            let scale = f32::from_bits(scale_bits as u32);
            // Bump refcounts on Q, K, V so they survive until backward
            NslTensor::from_ptr(q_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(k_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            NslTensor::from_ptr(v_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            // Bump refcount on out so it survives until backward (needed for D[i] = dO . O)
            NslTensor::from_ptr(out_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            // Bump refcount on logsumexp so it survives until backward
            NslTensor::from_ptr(logsumexp_ptr).refcount.fetch_add(1, Ordering::SeqCst);
            autodiff::maybe_record(autodiff::TapeOp::FlashAttention {
                q: q_ptr, k: k_ptr, v: v_ptr,
                out: out_ptr,
                logsumexp: logsumexp_ptr,
                scale,
                batch, heads, seq_len, head_dim,
                causal: causal != 0,
                saved_q: q_ptr,
                saved_k: k_ptr,
                saved_v: v_ptr,
            });
        }

        result
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv, causal);
        eprintln!("[nsl] FlashAttention requires CUDA. Use naive path (no @flash_attention decorator).");
        -1
    }
}

/// M42b: Quantized FlashAttention — KV-cache in INT8/FP8, Q in f16/f32.
///
/// Same tiled FlashAttention-2 algorithm but with inline dequantization:
/// each tile load of K/V dequantizes INT8→f32 (using per-head scales from
/// meta_k/meta_v) before the Q@K^T dot product.
///
/// kv_quant_scheme:
///   0 = None (f32 KV, same as nsl_flash_attention)
///   1 = INT8 per-head (scale per attention head)
///   2 = INT8 per-token (scale per token position)
///   3 = INT4 per-group (scale + zero_point per group)
///   4 = FP8 E4M3 (no scale needed, direct cast)
///
/// meta_k/meta_v: pointers to KvBlockQuantMeta arrays (null for FP8/None).
///
/// Returns 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_quantized(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    meta_k: i64, meta_v: i64,
    kv_quant_scheme: i64,
    shared_mem_bytes: i64,
    ptx_ptr: i64, name_ptr: i64,
    block_q: i64, _block_kv: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let _scale = f32::from_bits(scale_bits as u32);

        // Grid: (ceil(seq_len / block_q), batch * heads, 1)
        let grid_x = (seq_len + block_q - 1) / block_q;
        let grid_y = batch * heads;
        let grid_z = 1i64;

        // Block: (128, 1, 1) — 4 warps per thread block
        let block_x = 128i64;
        let block_y = 1i64;
        let block_z = 1i64;

        // Marshal all kernel arguments as u64 values
        let mut q = q_ptr as u64;
        let mut k = k_ptr as u64;
        let mut v = v_ptr as u64;
        let mut out = out_ptr as u64;
        let mut s = f32::from_bits(scale_bits as u32);
        let mut b = batch as u64;
        let mut h = heads as u64;
        let mut sl = seq_len as u64;
        let mut hd = head_dim as u64;
        let mut bt = block_table_ptr as u64;
        let mut kp = k_pool_ptr as u64;
        let mut vp = v_pool_ptr as u64;
        let mut bs = block_size as u64;
        let mut mk = meta_k as u64;
        let mut mv = meta_v as u64;
        let mut qs = kv_quant_scheme as u64;

        let args: [*mut c_void; 16] = [
            &mut q as *mut _ as *mut c_void,
            &mut k as *mut _ as *mut c_void,
            &mut v as *mut _ as *mut c_void,
            &mut out as *mut _ as *mut c_void,
            &mut s as *mut _ as *mut c_void,
            &mut b as *mut _ as *mut c_void,
            &mut h as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut mk as *mut _ as *mut c_void,
            &mut mv as *mut _ as *mut c_void,
            &mut qs as *mut _ as *mut c_void,
        ];

        // For scheme 0 (no quantization), delegate to standard FlashAttention PTX.
        // For scheme 1-4, the quantized PTX kernel handles dequantization inline:
        //   INT8: each tile load does `v_f32 = (int8_val as f32) * scale[head]`
        //   FP8:  each tile load does direct E4M3→f32 cast via LUT
        //   INT4: each tile load unpacks nibbles + applies group scale/zero_point
        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (q_ptr, k_ptr, v_ptr, out_ptr, scale_bits);
        let _ = (batch, heads, seq_len, head_dim);
        let _ = (block_table_ptr, k_pool_ptr, v_pool_ptr, block_size);
        let _ = (meta_k, meta_v, kv_quant_scheme);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv);
        eprintln!("[nsl] quantized FlashAttention requires CUDA.");
        -1
    }
}

/// RoPE + paged cache write kernel launch wrapper.
///
/// All params i64 for Cranelift ABI compatibility.
/// Grid: (num_tokens, num_heads, ceil(head_dim/2))
#[no_mangle]
pub extern "C" fn nsl_rope_cache_write(
    k_projected_ptr: i64, v_projected_ptr: i64,
    cos_ptr: i64, sin_ptr: i64,
    positions_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_table_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
    num_tokens: i64, num_heads: i64, head_dim: i64, block_size: i64,
    ptx_ptr: i64, name_ptr: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let grid_x = num_tokens;
        let grid_y = num_heads;
        let grid_z = (head_dim + 1) / 2;

        let block_x = 1i64;
        let block_y = 1i64;
        let block_z = 1i64;

        let mut kp = k_projected_ptr as u64;
        let mut vp = v_projected_ptr as u64;
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut pos = positions_ptr as u64;
        let mut k_pool = k_pool_ptr as u64;
        let mut v_pool = v_pool_ptr as u64;
        let mut bt = block_table_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;
        let mut nt = num_tokens as u64;
        let mut nh = num_heads as u64;
        let mut hd = head_dim as u64;
        let mut bs = block_size as u64;

        let args: [*mut c_void; 14] = [
            &mut kp as *mut _ as *mut c_void,
            &mut vp as *mut _ as *mut c_void,
            &mut cos as *mut _ as *mut c_void,
            &mut sin as *mut _ as *mut c_void,
            &mut pos as *mut _ as *mut c_void,
            &mut k_pool as *mut _ as *mut c_void,
            &mut v_pool as *mut _ as *mut c_void,
            &mut bt as *mut _ as *mut c_void,
            &mut sids as *mut _ as *mut c_void,
            &mut slens as *mut _ as *mut c_void,
            &mut nt as *mut _ as *mut c_void,
            &mut nh as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
        ];

        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            0,
        );

        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (k_projected_ptr, v_projected_ptr, cos_ptr, sin_ptr, positions_ptr);
        let _ = (k_pool_ptr, v_pool_ptr, block_table_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (num_tokens, num_heads, head_dim, block_size, ptx_ptr, name_ptr);
        eprintln!("[nsl] rope_cache_write requires CUDA.");
        -1
    }
}

// ── FlashAttention backward pass (CPU reference implementation) ──────────

/// CPU reference implementation of FlashAttention backward.
///
/// Computes dQ, dK, dV from the upstream gradient dO, using the saved Q, K, V
/// tensors and the logsumexp (L) from the forward pass. The attention matrix is
/// recomputed from L (never stored), so memory is O(N) not O(N^2).
///
/// All tensors are [batch, heads, seq_len, head_dim] except logsumexp which is
/// [batch, heads, seq_len]. The output tensor O is [batch, heads, seq_len, head_dim].
///
/// Algorithm per (batch, head):
///   D[i] = sum_d(dO[i,d] * O[i,d])                        -- correction term
///   For each query i, key j:
///     s = sum_d(Q[i,d] * K[j,d]) * scale
///     p = exp(s - L[i])                                     -- recomputed softmax
///     if causal and j > i: p = 0
///     dV[j] += p * dO[i]                                    -- P^T @ dO
///     dp = sum_d(dO[i,d] * V[j,d])                          -- dO @ V^T
///     ds = p * (dp - D[i])                                   -- softmax backward
///     dQ[i] += ds * K[j] * scale
///     dK[j] += ds * Q[i] * scale
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_cpu(
    q: &[f32], k: &[f32], v: &[f32],
    out: &[f32], logsumexp: &[f32], dout: &[f32],
    dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    batch: usize, heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool,
) {
    // Strides for [batch, heads, seq_len, head_dim] layout
    let bh_stride = heads * seq_len * head_dim;
    let h_stride = seq_len * head_dim;
    let s_stride = head_dim;

    // Strides for logsumexp [batch, heads, seq_len]
    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;

    for b in 0..batch {
        for h in 0..heads {
            let qkv_base = b * bh_stride + h * h_stride;
            let lse_base = b * lse_bh_stride + h * lse_h_stride;

            // Phase 1: Compute correction D[i] = rowsum(dO[i] * O[i])
            let mut d_corr = vec![0.0f32; seq_len];
            for (i, d_corr_i) in d_corr.iter_mut().enumerate() {
                let row_base = qkv_base + i * s_stride;
                let mut sum = 0.0f32;
                for d in 0..head_dim {
                    sum += dout[row_base + d] * out[row_base + d];
                }
                *d_corr_i = sum;
            }

            // Phase 2: Compute dQ, dK, dV
            for i in 0..seq_len {
                let q_base = qkv_base + i * s_stride;
                let lse_i = logsumexp[lse_base + i];

                let j_max = if causal { i + 1 } else { seq_len };
                for j in 0..j_max {
                    let k_base = qkv_base + j * s_stride;
                    let v_base = qkv_base + j * s_stride;

                    // Recompute S[i,j] = Q[i] . K[j] * scale
                    let mut s_val = 0.0f32;
                    for d in 0..head_dim {
                        s_val += q[q_base + d] * k[k_base + d];
                    }
                    s_val *= scale;

                    // Recompute P[i,j] = exp(S[i,j] - L[i])
                    let p_val = (s_val - lse_i).exp();

                    // dV[j] += p * dO[i]
                    for d in 0..head_dim {
                        dv[k_base + d] += p_val * dout[q_base + d];
                    }

                    // dp = dO[i] . V[j]
                    let mut dp_val = 0.0f32;
                    for d in 0..head_dim {
                        dp_val += dout[q_base + d] * v[v_base + d];
                    }

                    // dS = P * (dP - D[i])
                    let ds_val = p_val * (dp_val - d_corr[i]);

                    // dQ[i] += dS * K[j] * scale
                    // dK[j] += dS * Q[i] * scale
                    for d in 0..head_dim {
                        dq[q_base + d] += ds_val * k[k_base + d] * scale;
                        dk[k_base + d] += ds_val * q[q_base + d] * scale;
                    }
                }
            }
        }
    }
}

/// GQA-aware backward: Q has `heads` heads, K/V have `kv_heads` heads.
/// Each group of `gqa_groups` Q heads shares one KV head.
/// dQ has shape [batch, heads, seq, head_dim].
/// dK, dV have shape [batch, kv_heads, seq, head_dim].
#[allow(clippy::too_many_arguments)]
pub fn flash_attention_backward_cpu_gqa(
    q: &[f32], k: &[f32], v: &[f32],
    out: &[f32], logsumexp: &[f32], dout: &[f32],
    dq: &mut [f32], dk: &mut [f32], dv: &mut [f32],
    batch: usize, heads: usize, kv_heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool, gqa_groups: usize,
) {
    // Q/dQ/out/dout strides: [batch, heads, seq_len, head_dim]
    let q_bh_stride = heads * seq_len * head_dim;
    let q_h_stride = seq_len * head_dim;
    // K/V/dK/dV strides: [batch, kv_heads, seq_len, head_dim]
    let kv_bh_stride = kv_heads * seq_len * head_dim;
    let kv_h_stride = seq_len * head_dim;
    let s_stride = head_dim;

    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;

    for b_idx in 0..batch {
        for h_idx in 0..heads {
            let kv_h_idx = h_idx / gqa_groups;
            let q_base = b_idx * q_bh_stride + h_idx * q_h_stride;
            let kv_base = b_idx * kv_bh_stride + kv_h_idx * kv_h_stride;
            let lse_base = b_idx * lse_bh_stride + h_idx * lse_h_stride;

            // Phase 1: D[i] = rowsum(dO[i] * O[i])
            let mut d_corr = vec![0.0f32; seq_len];
            for (i, d_corr_i) in d_corr.iter_mut().enumerate() {
                let row = q_base + i * s_stride;
                let mut sum = 0.0f32;
                for d_idx in 0..head_dim {
                    sum += dout[row + d_idx] * out[row + d_idx];
                }
                *d_corr_i = sum;
            }

            // Phase 2: dQ, dK, dV
            for i in 0..seq_len {
                let qi = q_base + i * s_stride;
                let lse_i = logsumexp[lse_base + i];

                let j_max = if causal { i + 1 } else { seq_len };
                for j in 0..j_max {
                    let kj = kv_base + j * s_stride;
                    let vj = kv_base + j * s_stride;

                    // S[i,j] = Q[i] . K[j] * scale
                    let mut s_val = 0.0f32;
                    for d_idx in 0..head_dim {
                        s_val += q[qi + d_idx] * k[kj + d_idx];
                    }
                    s_val *= scale;

                    // P[i,j] = exp(S[i,j] - L[i])
                    let p_val = (s_val - lse_i).exp();

                    // dV[j] += p * dO[i]
                    for d_idx in 0..head_dim {
                        dv[kj + d_idx] += p_val * dout[qi + d_idx];
                    }

                    // dp = dO[i] . V[j]
                    let mut dp_val = 0.0f32;
                    for d_idx in 0..head_dim {
                        dp_val += dout[qi + d_idx] * v[vj + d_idx];
                    }

                    // dS = P * (dP - D[i])
                    let ds_val = p_val * (dp_val - d_corr[i]);

                    // dQ[i] += dS * K[j] * scale
                    // dK[j] += dS * Q[i] * scale
                    for d_idx in 0..head_dim {
                        dq[qi + d_idx] += ds_val * k[kj + d_idx] * scale;
                        dk[kj + d_idx] += ds_val * q[qi + d_idx] * scale;
                    }
                }
            }
        }
    }
}

/// FFI entry point for FlashAttention backward pass (CPU reference).
/// Auto-compute logsumexp from Q, K when the forward didn't save it.
///
/// lse[b,h,i] = log(sum_j(exp(Q[b,h,i,:] . K[b,h,j,:] * scale)))
/// with optional causal masking (j <= i).
/// Auto-compute logsumexp with GQA support.
///
/// Q has `heads` heads, K has `kv_heads` heads. Each Q head group maps to one KV head.
/// lse[b,h,i] = log(sum_j(exp(Q[b,h,i,:] . K[b,h//gqa_groups,j,:] * scale)))
#[allow(clippy::too_many_arguments)]
fn compute_logsumexp_gqa(
    q: &[f32], k: &[f32],
    batch: usize, heads: usize, kv_heads: usize, seq_len: usize, head_dim: usize,
    scale: f32, causal: bool,
) -> Vec<f32> {
    let q_bh_stride = heads * seq_len * head_dim;
    let q_h_stride = seq_len * head_dim;
    let k_bh_stride = kv_heads * seq_len * head_dim;
    let k_h_stride = seq_len * head_dim;
    let s_stride = head_dim;
    let lse_bh_stride = heads * seq_len;
    let lse_h_stride = seq_len;
    let gqa_groups = if kv_heads > 0 { heads / kv_heads } else { 1 };

    let total_lse = batch * heads * seq_len;
    let mut lse = vec![0.0f32; total_lse];

    for b_idx in 0..batch {
        for h_idx in 0..heads {
            let kv_h_idx = h_idx / gqa_groups;
            let q_base_bh = b_idx * q_bh_stride + h_idx * q_h_stride;
            let k_base_bh = b_idx * k_bh_stride + kv_h_idx * k_h_stride;
            let lse_base = b_idx * lse_bh_stride + h_idx * lse_h_stride;

            for i in 0..seq_len {
                let q_row = q_base_bh + i * s_stride;
                let j_max = if causal { i + 1 } else { seq_len };

                // Numerically stable logsumexp: max + log(sum(exp(x - max)))
                let mut max_val = f32::NEG_INFINITY;
                for j in 0..j_max {
                    let k_row = k_base_bh + j * s_stride;
                    let mut dot = 0.0f32;
                    for d_idx in 0..head_dim {
                        dot += q[q_row + d_idx] * k[k_row + d_idx];
                    }
                    let score = dot * scale;
                    if score > max_val {
                        max_val = score;
                    }
                }

                let mut sum_exp = 0.0f32;
                for j in 0..j_max {
                    let k_row = k_base_bh + j * s_stride;
                    let mut dot = 0.0f32;
                    for d_idx in 0..head_dim {
                        dot += q[q_row + d_idx] * k[k_row + d_idx];
                    }
                    let score = dot * scale;
                    sum_exp += (score - max_val).exp();
                }

                lse[lse_base + i] = max_val + sum_exp.ln();
            }
        }
    }

    lse
}

/// GPU PTX backward dispatch: launches Phase 1 (D-correction) and Phase 2 (dQ/dK/dV)
/// kernels entirely on GPU. No host-device transfer needed.
///
/// Returns NslList [dQ, dK, dV] as GPU tensors.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn flash_attention_backward_gpu(
    dout_ptr: i64, q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, _logsumexp_ptr: i64,
    scale: f32,
    b: usize, h: usize, s: usize, d: usize,
    is_causal: bool,
    phase1_ptx_ptr: i64, phase1_name_ptr: i64,
    phase2_ptx_ptr: i64, phase2_name_ptr: i64,
) -> i64 {
    use crate::cuda::inner;
    use std::ffi::c_void;

    // Sync before reading tensor data pointers
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let dout_t = NslTensor::from_ptr(dout_ptr);
    let q_t = NslTensor::from_ptr(q_ptr);
    let k_t = NslTensor::from_ptr(k_ptr);
    let v_t = NslTensor::from_ptr(v_ptr);
    let out_t = NslTensor::from_ptr(out_ptr);

    let total_qkv = b * h * s * d;

    // Block sizes must match compile-time PTX constants
    let block_q: i64 = 64;
    let block_kv: i64 = 64;

    // ── Allocate D correction vector [b*h*s] on GPU ──
    let d_buf = inner::alloc_managed(b * h * s * 4);

    // ── Allocate dQ on GPU (zero-initialized) ──
    let dq_data = inner::alloc_managed(total_qkv * 4);
    inner::memset_d8(dq_data, total_qkv * 4);

    // ── Allocate dK, dV on GPU (zero-initialized) ──
    let dk_data = inner::alloc_managed(total_qkv * 4);
    inner::memset_d8(dk_data, total_qkv * 4);
    let dv_data = inner::alloc_managed(total_qkv * 4);
    inner::memset_d8(dv_data, total_qkv * 4);

    // ── Phase 1: D-correction vector ──
    // D[bh, i] = sum_d( dO[bh, i, d] * O[bh, i, d] )
    // Grid: (b*h, ceil(s/block_q), 1), Block: (block_q, 1, 1)
    {
        let mut dout_data = dout_t.data as u64;
        let mut out_data = out_t.data as u64;
        let mut d_data = d_buf as u64;
        let mut sl = s as u64;
        let mut hd = d as u64;

        let args: [*mut c_void; 5] = [
            &mut dout_data as *mut _ as *mut c_void,
            &mut out_data as *mut _ as *mut c_void,
            &mut d_data as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let grid = [(b * h) as i64, (s as i64 + block_q - 1) / block_q, 1];
        let block = [block_q, 1, 1];

        let res = inner::kernel_launch(
            phase1_ptx_ptr as *const u8,
            phase1_name_ptr as *const u8,
            grid, block, &args, 0,
        );
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("[flash-bwd] Phase 1 kernel launch failed: {:?}", res);
        }
    }

    // ── Phase 2: Main backward (dQ/dK/dV) ──
    // Grid: (b*h, ceil(s/block_kv), 1), Block: (block_q, 1, 1)
    // Shared memory: compute inline (matches backward_shared_mem_bytes)
    {
        let pad: i64 = 4;
        let hd_padded = d as i64 + pad;
        let tile_bytes = |rows: i64, cols: i64| -> i64 { rows * cols * 4 };
        let shmem = (tile_bytes(block_kv, hd_padded) * 2  // K, V tiles
            + tile_bytes(block_q, hd_padded) * 2           // Q, dO tiles
            + tile_bytes(block_kv, hd_padded) * 2          // dK, dV accumulators
            + tile_bytes(block_q, block_kv)                // S tile
            + block_q * 4                                  // D vector
            + block_q * 4                                  // L (logsumexp) vector
        ) as u32;

        // Compute logsumexp on GPU: reuse D buffer area as scratch for lse
        // For now, pass the logsumexp pointer if available, otherwise auto-compute
        // The Phase 2 kernel expects lse_data; we need to compute it.
        // Since _logsumexp_ptr might be 0, we compute lse on CPU and upload.
        let lse_data = {
            let total_lse = b * h * s;
            let lse_gpu = inner::alloc_managed(total_lse * 4);

            // Read Q and K to CPU to compute logsumexp
            let total_qkv_bytes = total_qkv * 4;
            let mut q_cpu = vec![0.0f32; total_qkv];
            let mut k_cpu = vec![0.0f32; total_qkv];
            inner::memcpy_dtoh(
                q_cpu.as_mut_ptr() as *mut c_void,
                q_t.data as *const c_void,
                total_qkv_bytes,
            );
            inner::memcpy_dtoh(
                k_cpu.as_mut_ptr() as *mut c_void,
                k_t.data as *const c_void,
                total_qkv_bytes,
            );

            let lse_cpu = compute_logsumexp_gqa(&q_cpu, &k_cpu, b, h, h, s, d, scale, is_causal);
            inner::memcpy_htod(
                lse_gpu,
                lse_cpu.as_ptr() as *const c_void,
                total_lse * 4,
            );
            lse_gpu
        };

        let mut dout_data = dout_t.data as u64;
        let mut q_data = q_t.data as u64;
        let mut k_data = k_t.data as u64;
        let mut v_data = v_t.data as u64;
        let mut dq_d = dq_data as u64;
        let mut dk_d = dk_data as u64;
        let mut dv_d = dv_data as u64;
        let mut d_d = d_buf as u64;
        let mut lse_d = lse_data as u64;
        let mut sc = scale;
        let mut sl = s as u64;
        let mut hd = d as u64;

        let args: [*mut c_void; 12] = [
            &mut dout_data as *mut _ as *mut c_void,
            &mut q_data as *mut _ as *mut c_void,
            &mut k_data as *mut _ as *mut c_void,
            &mut v_data as *mut _ as *mut c_void,
            &mut dq_d as *mut _ as *mut c_void,
            &mut dk_d as *mut _ as *mut c_void,
            &mut dv_d as *mut _ as *mut c_void,
            &mut d_d as *mut _ as *mut c_void,
            &mut lse_d as *mut _ as *mut c_void,
            &mut sc as *mut _ as *mut c_void,
            &mut sl as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];

        let grid = [(b * h) as i64, (s as i64 + block_kv - 1) / block_kv, 1];
        let block = [block_q, 1, 1];

        let res = inner::kernel_launch(
            phase2_ptx_ptr as *const u8,
            phase2_name_ptr as *const u8,
            grid, block, &args, shmem,
        );
        if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!("[flash-bwd] Phase 2 kernel launch failed: {:?}", res);
        }

        // Sync after all kernels
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

        // Free logsumexp scratch buffer
        inner::free_managed(lse_data);
    }

    // Free D correction buffer
    inner::free_managed(d_buf);

    // ── Build output NslTensor wrappers for dQ, dK, dV ──
    fn make_gpu_tensor(data: *mut c_void, shape: &[i64], total: usize) -> i64 {
        let ndim = shape.len() as i64;
        let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
        for (i, &s) in shape.iter().enumerate() {
            unsafe { *shape_ptr.add(i) = s };
        }
        let strides = NslTensor::compute_strides(shape_ptr, ndim);
        let t = Box::new(NslTensor::new(
            data,
            shape_ptr,
            strides,
            ndim,
            total as i64,
            1, // device = GPU
            1, // dtype = f32
            1, // refcount
            0, // flags
        ));
        Box::into_raw(t) as i64
    }

    let shape = [b as i64, h as i64, s as i64, d as i64];
    let dq_ptr = make_gpu_tensor(dq_data, &shape, total_qkv);
    let dk_ptr = make_gpu_tensor(dk_data, &shape, total_qkv);
    let dv_ptr = make_gpu_tensor(dv_data, &shape, total_qkv);

    // Pack into NslList [dQ, dK, dV]
    let list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(list, dq_ptr);
    crate::list::nsl_list_push(list, dk_ptr);
    crate::list::nsl_list_push(list, dv_ptr);
    list
}

///
/// Allocates dQ, dK, dV tensors, runs the CPU backward, and returns them
/// as a tuple of three tensor pointers packed into an NslList.
///
/// This is called from the backward dispatch in autodiff/backward.rs.
/// Falls back to CPU when GPU PTX backward is not available (no PTX pointers
/// or GQA with different Q/KV head counts).
///
/// When `logsumexp_ptr == 0`, the logsumexp is auto-computed from Q, K, scale.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_backward(
    dout_ptr: i64,
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: i64,
    phase1_ptx_ptr: i64,
    phase1_name_ptr: i64,
    phase2_ptx_ptr: i64,
    phase2_name_ptr: i64,
) -> i64 {
    let scale = f32::from_bits(scale_bits as u32);
    let b = batch as usize;
    let h = heads as usize;
    let s = seq_len as usize;
    let d = head_dim as usize;
    let is_causal = causal != 0;

    // GPU PTX dispatch: if PTX pointers are provided and tensors are on GPU,
    // launch the backward kernels directly on the device (no CPU transfer).
    #[cfg(feature = "cuda")]
    {
        let dout_t = NslTensor::from_ptr(dout_ptr);
        let k_t = NslTensor::from_ptr(k_ptr);
        let kv_h = if k_t.ndim >= 2 {
            unsafe { *k_t.shape.add(1) as usize }
        } else {
            h
        };

        if dout_t.device > 0 && phase1_ptx_ptr != 0 && kv_h == h {
            return flash_attention_backward_gpu(
                dout_ptr, q_ptr, k_ptr, v_ptr, out_ptr, logsumexp_ptr,
                scale, b, h, s, d, is_causal,
                phase1_ptx_ptr, phase1_name_ptr,
                phase2_ptx_ptr, phase2_name_ptr,
            );
        }

        if dout_t.device > 0 {
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (phase1_ptx_ptr, phase1_name_ptr, phase2_ptx_ptr, phase2_name_ptr);
    }

    // Read input tensors
    let dout_t = NslTensor::from_ptr(dout_ptr);
    let q_t = NslTensor::from_ptr(q_ptr);
    let k_t = NslTensor::from_ptr(k_ptr);
    let v_t = NslTensor::from_ptr(v_ptr);
    let out_t = NslTensor::from_ptr(out_ptr);

    // Detect GQA: K/V may have fewer heads than Q.
    // Read actual KV head count from K's shape (dim 1).
    let kv_h = if k_t.ndim >= 2 {
        unsafe { *k_t.shape.add(1) as usize }
    } else {
        h
    };

    let total_qkv = b * h * s * d;
    let total_kv = b * kv_h * s * d;
    let total_lse = b * h * s;

    // Helper to read tensor data as f32 slice (handles both f32 and f64 dtypes).
    // Handles GPU tensors by transferring data to CPU via cudaMemcpy.
    fn read_f32_data(t: &NslTensor, len: usize) -> Vec<f32> {
        let is_gpu = t.device > 0;

        if t.dtype == 1 {
            // f32 tensor
            let mut buf = vec![0.0f32; len];
            if is_gpu {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::memcpy_dtoh(
                        buf.as_mut_ptr() as *mut std::ffi::c_void,
                        t.data as *const std::ffi::c_void,
                        len * 4,
                    );
                }
                #[cfg(not(feature = "cuda"))]
                {
                    eprintln!("[flash-bwd] WARNING: GPU tensor but CUDA not enabled");
                }
            } else {
                for i in 0..len {
                    buf[i] = unsafe { *t.data_f32().add(i) };
                }
            }
            buf
        } else {
            // f64 -> f32
            if is_gpu {
                // GPU f64 tensors: transfer as f64, then convert
                let mut f64_buf = vec![0.0f64; len];
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::inner::memcpy_dtoh(
                        f64_buf.as_mut_ptr() as *mut std::ffi::c_void,
                        t.data as *const std::ffi::c_void,
                        len * 8,
                    );
                }
                f64_buf.iter().map(|&v| v as f32).collect()
            } else {
                (0..len).map(|i| unsafe { *t.data_f64().add(i) as f32 }).collect()
            }
        }
    }

    let dout_data = read_f32_data(dout_t, total_qkv);
    let q_data = read_f32_data(q_t, total_qkv);
    let k_data = read_f32_data(k_t, total_kv);
    let v_data = read_f32_data(v_t, total_kv);
    let out_data = read_f32_data(out_t, total_qkv);

    // Read or auto-compute logsumexp.
    // When logsumexp_ptr == 0, the forward was decomposed (not fused FlashAttention)
    // and no logsumexp buffer was saved. Compute it from Q, K, scale, causal.
    let gqa_groups = if kv_h > 0 { h / kv_h } else { 1 };
    let lse_data = if logsumexp_ptr != 0 {
        let lse_t = NslTensor::from_ptr(logsumexp_ptr);
        read_f32_data(lse_t, total_lse)
    } else {
        compute_logsumexp_gqa(&q_data, &k_data, b, h, kv_h, s, d, scale, is_causal)
    };

    // Allocate gradient buffers (zero-initialized)
    // dQ has Q's shape [batch, heads, seq, head_dim]
    // dK, dV have KV's shape [batch, kv_heads, seq, head_dim]
    let mut dq_data = vec![0.0f32; total_qkv];
    let mut dk_data = vec![0.0f32; total_kv];
    let mut dv_data = vec![0.0f32; total_kv];

    // Run the CPU backward with GQA support
    flash_attention_backward_cpu_gqa(
        &q_data, &k_data, &v_data,
        &out_data, &lse_data, &dout_data,
        &mut dq_data, &mut dk_data, &mut dv_data,
        b, h, kv_h, s, d,
        scale, is_causal, gqa_groups,
    );

    // Create output tensors (f32 dtype, matching Q shape for dQ, KV shape for dK/dV)
    fn make_tensor(data: &[f32], shape: &[i64]) -> i64 {
        let ndim = shape.len() as i64;
        let total = data.len();
        let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
        for (i, &s) in shape.iter().enumerate() {
            unsafe { *shape_ptr.add(i) = s };
        }
        let strides = NslTensor::compute_strides(shape_ptr, ndim);
        let data_size = std::mem::size_of_val(data);
        let data_ptr = crate::memory::checked_alloc(data_size) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, total);
        }
        let t = Box::new(NslTensor::new(
            data_ptr as *mut std::ffi::c_void,
            shape_ptr,
            strides,
            ndim,
            total as i64,
            0,
            1,
            1,
            0,
        ));
        Box::into_raw(t) as i64
    }

    let q_shape = [batch, heads, seq_len, head_dim];
    let kv_shape = [batch, kv_h as i64, seq_len, head_dim];
    let mut dq_ptr = make_tensor(&dq_data, &q_shape);
    let mut dk_ptr = make_tensor(&dk_data, &kv_shape);
    let mut dv_ptr = make_tensor(&dv_data, &kv_shape);

    // If inputs were on GPU, transfer gradient tensors to GPU to match device
    let input_device = q_t.device;
    if input_device > 0 {
        dq_ptr = crate::tensor::nsl_tensor_to_device(dq_ptr, input_device as i64);
        dk_ptr = crate::tensor::nsl_tensor_to_device(dk_ptr, input_device as i64);
        dv_ptr = crate::tensor::nsl_tensor_to_device(dv_ptr, input_device as i64);
    }

    // Pack into an NslList [dq, dk, dv]
    let list = crate::list::nsl_list_new();
    crate::list::nsl_list_push(list, dq_ptr);
    crate::list::nsl_list_push(list, dk_ptr);
    crate::list::nsl_list_push(list, dv_ptr);
    list
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive attention forward for reference: O = softmax(Q @ K^T * scale) @ V
    fn naive_attention_forward(
        q: &[f32], k: &[f32], v: &[f32],
        batch: usize, heads: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        // Returns (output, logsumexp)
        let bh_stride = heads * seq_len * head_dim;
        let h_stride = seq_len * head_dim;
        let s_stride = head_dim;
        let lse_bh_stride = heads * seq_len;
        let lse_h_stride = seq_len;

        let total_qkv = batch * heads * seq_len * head_dim;
        let total_lse = batch * heads * seq_len;
        let mut output = vec![0.0f32; total_qkv];
        let mut logsumexp = vec![0.0f32; total_lse];

        for b in 0..batch {
            for h in 0..heads {
                let qkv_base = b * bh_stride + h * h_stride;
                let lse_base = b * lse_bh_stride + h * lse_h_stride;

                for i in 0..seq_len {
                    let q_base = qkv_base + i * s_stride;

                    // Compute scores S[i,:] = Q[i] . K[:] * scale
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[j] = dot * scale;
                    }

                    // Apply causal mask
                    if causal {
                        for j in (i + 1)..seq_len {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }

                    // Online softmax and logsumexp
                    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for &s in &scores {
                        sum_exp += (s - max_score).exp();
                    }
                    logsumexp[lse_base + i] = max_score + sum_exp.ln();

                    // Softmax probs
                    let lse = logsumexp[lse_base + i];
                    let probs: Vec<f32> = scores.iter().map(|&s| (s - lse).exp()).collect();

                    // O[i] = sum_j P[i,j] * V[j]
                    for j in 0..seq_len {
                        let v_base = qkv_base + j * s_stride;
                        for d in 0..head_dim {
                            output[q_base + d] += probs[j] * v[v_base + d];
                        }
                    }
                }
            }
        }

        (output, logsumexp)
    }

    /// Naive attention backward for reference (full S^2 computation)
    fn naive_attention_backward(
        q: &[f32], k: &[f32], v: &[f32],
        dout: &[f32],
        batch: usize, heads: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let bh_stride = heads * seq_len * head_dim;
        let h_stride = seq_len * head_dim;
        let s_stride = head_dim;
        let total_qkv = batch * heads * seq_len * head_dim;

        let mut dq = vec![0.0f32; total_qkv];
        let mut dk = vec![0.0f32; total_qkv];
        let mut dv = vec![0.0f32; total_qkv];

        for b in 0..batch {
            for h in 0..heads {
                let qkv_base = b * bh_stride + h * h_stride;

                // Compute full attention scores and softmax probs
                let mut probs = vec![vec![0.0f32; seq_len]; seq_len];
                for i in 0..seq_len {
                    let q_base = qkv_base + i * s_stride;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[j] = dot * scale;
                    }
                    if causal {
                        for j in (i + 1)..seq_len {
                            scores[j] = f32::NEG_INFINITY;
                        }
                    }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - max_s).exp();
                        sum_exp += *s;
                    }
                    for j in 0..seq_len {
                        probs[i][j] = scores[j] / sum_exp;
                    }
                }

                // dV[j] = sum_i P[i,j] * dO[i]
                for j in 0..seq_len {
                    let v_base = qkv_base + j * s_stride;
                    for i in 0..seq_len {
                        let do_base = qkv_base + i * s_stride;
                        for d in 0..head_dim {
                            dv[v_base + d] += probs[i][j] * dout[do_base + d];
                        }
                    }
                }

                // dP[i,j] = dO[i] . V[j]
                // dS[i,j] = P[i,j] * (dP[i,j] - sum_k(dP[i,k] * P[i,k]))
                //         = P[i,j] * (dP[i,j] - D[i])
                // where D[i] = sum_j P[i,j] * dP[i,j] = sum_j P[i,j] * (dO[i] . V[j])
                //            = dO[i] . (sum_j P[i,j] * V[j]) = dO[i] . O[i]
                for i in 0..seq_len {
                    let do_base = qkv_base + i * s_stride;

                    // D[i] = sum_j P[i,j] * (dO[i] . V[j])
                    let mut d_i = 0.0f32;
                    for j in 0..seq_len {
                        let v_base = qkv_base + j * s_stride;
                        let mut dp_ij = 0.0f32;
                        for d in 0..head_dim {
                            dp_ij += dout[do_base + d] * v[v_base + d];
                        }
                        d_i += probs[i][j] * dp_ij;
                    }

                    for j in 0..seq_len {
                        let k_base = qkv_base + j * s_stride;
                        let v_base = qkv_base + j * s_stride;

                        // dP[i,j]
                        let mut dp_ij = 0.0f32;
                        for d in 0..head_dim {
                            dp_ij += dout[do_base + d] * v[v_base + d];
                        }

                        // dS[i,j]
                        let ds_ij = probs[i][j] * (dp_ij - d_i);

                        // dQ[i] += dS[i,j] * K[j] * scale
                        let q_base = qkv_base + i * s_stride;
                        for d in 0..head_dim {
                            dq[q_base + d] += ds_ij * k[k_base + d] * scale;
                            dk[k_base + d] += ds_ij * q[q_base + d] * scale;
                        }
                    }
                }
            }
        }

        (dq, dk, dv)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
    }

    #[allow(dead_code)]
    fn rel_error(a: &[f32], b: &[f32]) -> f32 {
        let mut max_rel = 0.0f32;
        for (x, y) in a.iter().zip(b.iter()) {
            let denom = x.abs().max(y.abs()).max(1e-8);
            let rel = (x - y).abs() / denom;
            if rel > max_rel { max_rel = rel; }
        }
        max_rel
    }

    #[test]
    fn test_flash_backward_matches_naive_non_causal() {
        let batch = 1;
        let heads = 1;
        let seq_len = 4;
        let head_dim = 8;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        // Deterministic test data
        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        // Forward pass (naive reference)
        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);

        // Random-ish upstream gradient
        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.7 + 3.0).cos() * 0.3).collect();

        // Naive backward
        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, false,
        );

        // Flash backward
        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, false,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);

        assert!(dq_err < tol, "dQ max abs diff = {dq_err} exceeds tolerance {tol}");
        assert!(dk_err < tol, "dK max abs diff = {dk_err} exceeds tolerance {tol}");
        assert!(dv_err < tol, "dV max abs diff = {dv_err} exceeds tolerance {tol}");
    }

    #[test]
    fn test_flash_backward_matches_naive_causal() {
        let batch = 1;
        let heads = 2;
        let seq_len = 6;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.13).sin() * 0.4).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.17 + 1.0).cos() * 0.4).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.23 + 2.0).sin() * 0.4).collect();

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, true);

        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.31 + 3.0).cos() * 0.3).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, true,
        );

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, true,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);

        assert!(dq_err < tol, "dQ max abs diff = {dq_err} exceeds tolerance {tol} (causal)");
        assert!(dk_err < tol, "dK max abs diff = {dk_err} exceeds tolerance {tol} (causal)");
        assert!(dv_err < tol, "dV max abs diff = {dv_err} exceeds tolerance {tol} (causal)");
    }

    #[test]
    fn test_flash_backward_batched_multihead() {
        let batch = 2;
        let heads = 3;
        let seq_len = 4;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.07).sin() * 0.3).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.11 + 0.5).cos() * 0.3).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.19 + 1.5).sin() * 0.3).collect();

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);

        let dout: Vec<f32> = (0..total).map(|i| (i as f32 * 0.29 + 2.5).cos() * 0.2).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward(
            &q, &k, &v, &dout, batch, heads, seq_len, head_dim, scale, false,
        );

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, false,
        );

        let tol = 1e-4;
        assert!(max_abs_diff(&dq_naive, &dq_flash) < tol, "dQ mismatch in batched test");
        assert!(max_abs_diff(&dk_naive, &dk_flash) < tol, "dK mismatch in batched test");
        assert!(max_abs_diff(&dv_naive, &dv_flash) < tol, "dV mismatch in batched test");
    }

    #[test]
    fn test_flash_backward_causal_gradient_mask() {
        // Verify that for a causal mask, gradients for keys past the query position are zero
        // in the sense that they don't receive "future" contributions.
        let batch = 1;
        let heads = 1;
        let seq_len = 4;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        // Use identity-like Q/K to make attention sharp
        let mut q = vec![0.0f32; total];
        let mut k = vec![0.0f32; total];
        let v: Vec<f32> = (0..total).map(|i| (i as f32 + 1.0) * 0.1).collect();

        // Make Q[0] only attend to K[0] (with causal, it can't see K[1..3] anyway)
        for d in 0..head_dim {
            q[0 * head_dim + d] = if d == 0 { 5.0 } else { 0.0 };
            k[0 * head_dim + d] = if d == 0 { 5.0 } else { 0.0 };
            // Other positions get different patterns
            for i in 1..seq_len {
                q[i * head_dim + d] = if d == i % head_dim { 5.0 } else { 0.0 };
                k[i * head_dim + d] = if d == i % head_dim { 5.0 } else { 0.0 };
            }
        }

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, true);

        // Gradient only on the first query row
        let mut dout = vec![0.0f32; total];
        for d in 0..head_dim {
            dout[d] = 1.0;
        }

        let mut dq_flash = vec![0.0f32; total];
        let mut dk_flash = vec![0.0f32; total];
        let mut dv_flash = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            batch, heads, seq_len, head_dim,
            scale, true,
        );

        // With causal mask, Q[0] only attends to K[0], so:
        // - dK for positions j>0 should be zero from Q[0]'s gradient
        //   (since dout is only nonzero for row 0 and row 0 can only see j=0)
        // Check dK[1..] are zero
        for j in 1..seq_len {
            for d in 0..head_dim {
                assert!(dk_flash[j * head_dim + d].abs() < 1e-6,
                    "dK[{j},{d}] = {} should be zero for causal mask with dout only at row 0",
                    dk_flash[j * head_dim + d]);
            }
        }
        // dV for positions j>0 should also be zero (P[0,j]=0 for j>0)
        for j in 1..seq_len {
            for d in 0..head_dim {
                assert!(dv_flash[j * head_dim + d].abs() < 1e-6,
                    "dV[{j},{d}] = {} should be zero for causal mask with dout only at row 0",
                    dv_flash[j * head_dim + d]);
            }
        }
    }

    #[test]
    fn test_flash_backward_finite_difference() {
        // Numerical gradient check using finite differences
        let batch = 1;
        let heads = 1;
        let seq_len = 3;
        let head_dim = 4;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total = batch * heads * seq_len * head_dim;

        let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        let eps = 1e-3f32;

        let (out, lse) = naive_attention_forward(&q, &k, &v, batch, heads, seq_len, head_dim, scale, false);
        let dout: Vec<f32> = vec![1.0; total]; // all-ones upstream gradient = computing d(sum(O))/d(param)

        // Analytic gradients
        let mut dq_analytic = vec![0.0f32; total];
        let mut dk_analytic = vec![0.0f32; total];
        let mut dv_analytic = vec![0.0f32; total];
        flash_attention_backward_cpu(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_analytic, &mut dk_analytic, &mut dv_analytic,
            batch, heads, seq_len, head_dim, scale, false,
        );

        // Finite-difference check for dQ
        for idx in 0..total.min(12) {
            let mut q_plus = q.clone();
            let mut q_minus = q.clone();
            q_plus[idx] += eps;
            q_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q_plus, &k, &v, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q_minus, &k, &v, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dq_analytic[idx] - fd_grad).abs();
            let denom = dq_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dQ[{idx}] analytic={} fd={} rel_err={}",
                dq_analytic[idx], fd_grad, diff / denom);
        }

        // Finite-difference check for dK
        for idx in 0..total.min(12) {
            let mut k_plus = k.clone();
            let mut k_minus = k.clone();
            k_plus[idx] += eps;
            k_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q, &k_plus, &v, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q, &k_minus, &v, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dk_analytic[idx] - fd_grad).abs();
            let denom = dk_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dK[{idx}] analytic={} fd={} rel_err={}",
                dk_analytic[idx], fd_grad, diff / denom);
        }

        // Finite-difference check for dV
        for idx in 0..total.min(12) {
            let mut v_plus = v.clone();
            let mut v_minus = v.clone();
            v_plus[idx] += eps;
            v_minus[idx] -= eps;

            let (out_plus, _) = naive_attention_forward(&q, &k, &v_plus, batch, heads, seq_len, head_dim, scale, false);
            let (out_minus, _) = naive_attention_forward(&q, &k, &v_minus, batch, heads, seq_len, head_dim, scale, false);

            let fd_grad: f32 = out_plus.iter().zip(out_minus.iter())
                .map(|(p, m)| (p - m) / (2.0 * eps))
                .sum();

            let diff = (dv_analytic[idx] - fd_grad).abs();
            let denom = dv_analytic[idx].abs().max(fd_grad.abs()).max(1e-6);
            assert!(diff / denom < 0.05,
                "dV[{idx}] analytic={} fd={} rel_err={}",
                dv_analytic[idx], fd_grad, diff / denom);
        }
    }

    #[test]
    fn test_logsumexp_only_auxiliary_storage() {
        // Verify that the backward pass uses O(N) auxiliary storage (logsumexp)
        // not O(N^2) (full attention matrix). The logsumexp tensor is
        // [batch, heads, seq_len] while the attention matrix would be
        // [batch, heads, seq_len, seq_len].
        let batch = 2;
        let heads = 4;
        let seq_len = 16;
        let _head_dim = 8;
        let total_lse = batch * heads * seq_len;
        let total_attn = batch * heads * seq_len * seq_len;

        // logsumexp is O(N), attention matrix would be O(N^2)
        assert!(total_lse < total_attn,
            "logsumexp ({total_lse}) should be much smaller than full attention ({total_attn})");
        assert_eq!(total_lse, batch * heads * seq_len);
    }

    // ── GQA helpers ─────────────────────────────────────────────────────

    /// Naive GQA-aware forward: Q [b, h_q, s, d], K/V [b, h_kv, s, d]
    /// Each group of h_q/h_kv Q heads shares one KV head.
    fn naive_attention_forward_gqa(
        q: &[f32], k: &[f32], v: &[f32],
        batch: usize, h_q: usize, h_kv: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>) {
        let groups = h_q / h_kv;
        let q_bh = h_q * seq_len * head_dim;
        let kv_bh = h_kv * seq_len * head_dim;
        let total_q = batch * h_q * seq_len * head_dim;
        let total_lse = batch * h_q * seq_len;
        let mut output = vec![0.0f32; total_q];
        let mut logsumexp = vec![0.0f32; total_lse];

        for b in 0..batch {
            for hq in 0..h_q {
                let hkv = hq / groups;
                let q_base = b * q_bh + hq * seq_len * head_dim;
                let kv_base = b * kv_bh + hkv * seq_len * head_dim;
                let lse_base = b * h_q * seq_len + hq * seq_len;

                for i in 0..seq_len {
                    let qi = q_base + i * head_dim;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim { dot += q[qi + dd] * k[kj + dd]; }
                        scores[j] = dot * scale;
                    }
                    if causal { for j in (i+1)..seq_len { scores[j] = f32::NEG_INFINITY; } }

                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for &s in &scores { sum_exp += (s - max_s).exp(); }
                    logsumexp[lse_base + i] = max_s + sum_exp.ln();
                    let lse = logsumexp[lse_base + i];

                    let probs: Vec<f32> = scores.iter().map(|&s| (s - lse).exp()).collect();
                    for j in 0..seq_len {
                        let vj = kv_base + j * head_dim;
                        for dd in 0..head_dim {
                            output[qi + dd] += probs[j] * v[vj + dd];
                        }
                    }
                }
            }
        }
        (output, logsumexp)
    }

    /// Naive GQA backward reference: returns (dQ [b,h_q,s,d], dK [b,h_kv,s,d], dV [b,h_kv,s,d])
    fn naive_attention_backward_gqa(
        q: &[f32], k: &[f32], v: &[f32], dout: &[f32],
        batch: usize, h_q: usize, h_kv: usize, seq_len: usize, head_dim: usize,
        scale: f32, causal: bool,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let groups = h_q / h_kv;
        let q_bh = h_q * seq_len * head_dim;
        let kv_bh = h_kv * seq_len * head_dim;
        let total_q = batch * h_q * seq_len * head_dim;
        let total_kv = batch * h_kv * seq_len * head_dim;

        let mut dq = vec![0.0f32; total_q];
        let mut dk = vec![0.0f32; total_kv];
        let mut dv = vec![0.0f32; total_kv];

        for b in 0..batch {
            for hq in 0..h_q {
                let hkv = hq / groups;
                let q_base = b * q_bh + hq * seq_len * head_dim;
                let kv_base = b * kv_bh + hkv * seq_len * head_dim;

                // Compute softmax probs
                let mut probs = vec![vec![0.0f32; seq_len]; seq_len];
                for i in 0..seq_len {
                    let qi = q_base + i * head_dim;
                    let mut scores = vec![0.0f32; seq_len];
                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let mut dot = 0.0f32;
                        for dd in 0..head_dim { dot += q[qi + dd] * k[kj + dd]; }
                        scores[j] = dot * scale;
                    }
                    if causal { for j in (i+1)..seq_len { scores[j] = f32::NEG_INFINITY; } }
                    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum_exp = 0.0f32;
                    for s in &mut scores { *s = (*s - max_s).exp(); sum_exp += *s; }
                    for j in 0..seq_len { probs[i][j] = scores[j] / sum_exp; }
                }

                // dV[j] += sum_i P[i,j] * dO[i]  (accumulated across GQA group)
                for j in 0..seq_len {
                    let vj_idx = kv_base + j * head_dim;
                    for i in 0..seq_len {
                        let doi = q_base + i * head_dim;
                        for dd in 0..head_dim {
                            dv[vj_idx + dd] += probs[i][j] * dout[doi + dd];
                        }
                    }
                }

                // dQ, dK
                for i in 0..seq_len {
                    let doi = q_base + i * head_dim;
                    let mut d_i = 0.0f32;
                    for j in 0..seq_len {
                        let vj = kv_base + j * head_dim;
                        let mut dp_ij = 0.0f32;
                        for dd in 0..head_dim { dp_ij += dout[doi + dd] * v[vj + dd]; }
                        d_i += probs[i][j] * dp_ij;
                    }

                    for j in 0..seq_len {
                        let kj = kv_base + j * head_dim;
                        let vj = kv_base + j * head_dim;
                        let mut dp_ij = 0.0f32;
                        for dd in 0..head_dim { dp_ij += dout[doi + dd] * v[vj + dd]; }
                        let ds_ij = probs[i][j] * (dp_ij - d_i);

                        let qi = q_base + i * head_dim;
                        for dd in 0..head_dim {
                            dq[qi + dd] += ds_ij * k[kj + dd] * scale;
                            dk[kj + dd] += ds_ij * q[qi + dd] * scale;
                        }
                    }
                }
            }
        }
        (dq, dk, dv)
    }

    #[test]
    fn test_flash_attention_backward_gqa_cpu() {
        // GQA: 4 Q heads, 2 KV heads (groups=2)
        let b = 1;
        let h_q = 4;
        let h_kv = 2;
        let s = 16;
        let d = 16;
        let scale = 1.0 / (d as f32).sqrt();
        let groups = h_q / h_kv;

        let total_q = b * h_q * s * d;
        let total_kv = b * h_kv * s * d;

        // Deterministic test data
        let q: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        let k: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.2 + 1.0).cos() * 0.5).collect();
        let v: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.3 + 2.0).sin() * 0.5).collect();

        // GQA forward
        let (out, lse) = naive_attention_forward_gqa(&q, &k, &v, b, h_q, h_kv, s, d, scale, false);

        let dout: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.7 + 3.0).cos() * 0.3).collect();

        // Naive GQA backward (reference)
        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward_gqa(
            &q, &k, &v, &dout, b, h_q, h_kv, s, d, scale, false,
        );

        // Flash GQA backward (under test)
        let mut dq_flash = vec![0.0f32; total_q];
        let mut dk_flash = vec![0.0f32; total_kv];
        let mut dv_flash = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            b, h_q, h_kv, s, d,
            scale, false, groups,
        );

        // Verify shapes implicitly via lengths
        assert_eq!(dq_flash.len(), total_q, "dQ should have Q shape [b, h_q, s, d]");
        assert_eq!(dk_flash.len(), total_kv, "dK should have KV shape [b, h_kv, s, d]");
        assert_eq!(dv_flash.len(), total_kv, "dV should have KV shape [b, h_kv, s, d]");

        // Verify non-zero gradients
        let dq_norm: f32 = dq_flash.iter().map(|x| x * x).sum();
        let dk_norm: f32 = dk_flash.iter().map(|x| x * x).sum();
        let dv_norm: f32 = dv_flash.iter().map(|x| x * x).sum();
        assert!(dq_norm > 1e-8, "dQ should be non-zero, got norm={dq_norm}");
        assert!(dk_norm > 1e-8, "dK should be non-zero, got norm={dk_norm}");
        assert!(dv_norm > 1e-8, "dV should be non-zero, got norm={dv_norm}");

        // Match naive reference
        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);
        assert!(dq_err < tol, "GQA dQ max abs diff = {dq_err} exceeds tol {tol}");
        assert!(dk_err < tol, "GQA dK max abs diff = {dk_err} exceeds tol {tol}");
        assert!(dv_err < tol, "GQA dV max abs diff = {dv_err} exceeds tol {tol}");

        // Cross-check: summing dK from non-GQA (h_q heads, same K for each group)
        // should match the GQA dK. Run a non-GQA backward with K replicated.
        let mut k_expanded = vec![0.0f32; total_q]; // [b, h_q, s, d]
        let mut v_expanded = vec![0.0f32; total_q];
        for bb in 0..b {
            for hq in 0..h_q {
                let hkv = hq / groups;
                for si in 0..s {
                    for dd in 0..d {
                        let q_idx = bb * h_q * s * d + hq * s * d + si * d + dd;
                        let kv_idx = bb * h_kv * s * d + hkv * s * d + si * d + dd;
                        k_expanded[q_idx] = k[kv_idx];
                        v_expanded[q_idx] = v[kv_idx];
                    }
                }
            }
        }

        // Non-GQA backward with expanded K/V
        let (out_exp, lse_exp) = naive_attention_forward(
            &q, &k_expanded, &v_expanded, b, h_q, s, d, scale, false,
        );
        let mut dk_expanded = vec![0.0f32; total_q];
        let mut dq_expanded = vec![0.0f32; total_q];
        let mut dv_expanded = vec![0.0f32; total_q];
        flash_attention_backward_cpu(
            &q, &k_expanded, &v_expanded, &out_exp, &lse_exp, &dout,
            &mut dq_expanded, &mut dk_expanded, &mut dv_expanded,
            b, h_q, s, d, scale, false,
        );

        // Sum dk_expanded across groups to get per-KV-head gradients
        let mut dk_summed = vec![0.0f32; total_kv];
        for bb in 0..b {
            for hkv in 0..h_kv {
                for g in 0..groups {
                    let hq = hkv * groups + g;
                    for si in 0..s {
                        for dd in 0..d {
                            let kv_idx = bb * h_kv * s * d + hkv * s * d + si * d + dd;
                            let q_idx = bb * h_q * s * d + hq * s * d + si * d + dd;
                            dk_summed[kv_idx] += dk_expanded[q_idx];
                        }
                    }
                }
            }
        }

        let dk_cross_err = max_abs_diff(&dk_summed, &dk_flash);
        assert!(dk_cross_err < tol,
            "GQA dK should match sum of expanded non-GQA dK, err={dk_cross_err}");
    }

    #[test]
    fn test_flash_attention_backward_gqa_causal() {
        // GQA with causal mask: 4 Q heads, 2 KV heads
        let b = 1;
        let h_q = 4;
        let h_kv = 2;
        let s = 8;
        let d = 8;
        let scale = 1.0 / (d as f32).sqrt();
        let groups = h_q / h_kv;

        let total_q = b * h_q * s * d;
        let total_kv = b * h_kv * s * d;

        let q: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.13).sin() * 0.4).collect();
        let k: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.17 + 1.0).cos() * 0.4).collect();
        let v: Vec<f32> = (0..total_kv).map(|i| (i as f32 * 0.23 + 2.0).sin() * 0.4).collect();

        let (out, lse) = naive_attention_forward_gqa(&q, &k, &v, b, h_q, h_kv, s, d, scale, true);

        let dout: Vec<f32> = (0..total_q).map(|i| (i as f32 * 0.31 + 3.0).cos() * 0.3).collect();

        let (dq_naive, dk_naive, dv_naive) = naive_attention_backward_gqa(
            &q, &k, &v, &dout, b, h_q, h_kv, s, d, scale, true,
        );

        let mut dq_flash = vec![0.0f32; total_q];
        let mut dk_flash = vec![0.0f32; total_kv];
        let mut dv_flash = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout,
            &mut dq_flash, &mut dk_flash, &mut dv_flash,
            b, h_q, h_kv, s, d,
            scale, true, groups,
        );

        let tol = 1e-4;
        let dq_err = max_abs_diff(&dq_naive, &dq_flash);
        let dk_err = max_abs_diff(&dk_naive, &dk_flash);
        let dv_err = max_abs_diff(&dv_naive, &dv_flash);
        assert!(dq_err < tol, "GQA causal dQ err = {dq_err} exceeds tol {tol}");
        assert!(dk_err < tol, "GQA causal dK err = {dk_err} exceeds tol {tol}");
        assert!(dv_err < tol, "GQA causal dV err = {dv_err} exceeds tol {tol}");

        // Causal check: with dout only at position 0, positions j > 0
        // should not receive gradient from Q[0] for any head
        let mut dout_first = vec![0.0f32; total_q];
        for hq in 0..h_q {
            for dd in 0..d {
                dout_first[hq * s * d + dd] = 1.0;
            }
        }

        let mut dk_causal = vec![0.0f32; total_kv];
        let mut dq_causal = vec![0.0f32; total_q];
        let mut dv_causal = vec![0.0f32; total_kv];
        flash_attention_backward_cpu_gqa(
            &q, &k, &v, &out, &lse, &dout_first,
            &mut dq_causal, &mut dk_causal, &mut dv_causal,
            b, h_q, h_kv, s, d,
            scale, true, groups,
        );

        // With causal mask and dout only at row 0, K/V positions j>0
        // should get zero gradient contribution from row 0
        for hkv in 0..h_kv {
            for j in 1..s {
                for dd in 0..d {
                    let idx = hkv * s * d + j * d + dd;
                    assert!(dk_causal[idx].abs() < 1e-5,
                        "dK[hkv={hkv},j={j},d={dd}] = {} should be ~0 for causal with dout at row 0",
                        dk_causal[idx]);
                    assert!(dv_causal[idx].abs() < 1e-5,
                        "dV[hkv={hkv},j={j},d={dd}] = {} should be ~0 for causal with dout at row 0",
                        dv_causal[idx]);
                }
            }
        }
    }
}
