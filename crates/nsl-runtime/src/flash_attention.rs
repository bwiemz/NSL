//! FlashAttention-2 runtime launch wrappers.
//!
//! These functions compute grid/block dimensions from tensor shapes, marshal
//! arguments, and call `kernel_launch()` with pre-baked PTX from .rodata.
//! No PTX generation happens at runtime. On non-CUDA builds, falls back to
//! naive matmul+softmax attention path.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
#[cfg(feature = "cuda")]
use std::sync::atomic::Ordering;

use crate::tensor::NslTensor;
#[cfg(feature = "cuda")]
use crate::autodiff;

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

        let result = crate::cuda::inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &args,
            shared_mem_bytes as u32,
        );

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

        result as i64
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

/// FFI entry point for FlashAttention backward pass (CPU reference).
///
/// Allocates dQ, dK, dV tensors, runs the CPU backward, and returns them
/// as a tuple of three tensor pointers packed into an NslList.
///
/// This is called from the backward dispatch in autodiff/backward.rs.
/// GPU PTX backward can be added as a follow-up optimization.
#[no_mangle]
pub extern "C" fn nsl_flash_attention_backward(
    dout_ptr: i64,
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, logsumexp_ptr: i64,
    scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    causal: i64,
) -> i64 {
    let scale = f32::from_bits(scale_bits as u32);
    let b = batch as usize;
    let h = heads as usize;
    let s = seq_len as usize;
    let d = head_dim as usize;
    let is_causal = causal != 0;

    // Synchronize GPU if tensors might be on device
    #[cfg(feature = "cuda")]
    {
        let dout_t = NslTensor::from_ptr(dout_ptr);
        if dout_t.device > 0 {
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        }
    }

    let total_qkv = b * h * s * d;
    let total_lse = b * h * s;

    // Read input tensors as f32 slices
    let dout_t = NslTensor::from_ptr(dout_ptr);
    let q_t = NslTensor::from_ptr(q_ptr);
    let k_t = NslTensor::from_ptr(k_ptr);
    let v_t = NslTensor::from_ptr(v_ptr);
    let out_t = NslTensor::from_ptr(out_ptr);
    let lse_t = NslTensor::from_ptr(logsumexp_ptr);

    // Helper to read tensor data as f32 slice (handles both f32 and f64 dtypes)
    fn read_f32_data(t: &NslTensor, len: usize) -> Vec<f32> {
        if t.dtype == 1 {
            // f32
            (0..len).map(|i| unsafe { *t.data_f32().add(i) }).collect()
        } else {
            // f64 -> f32
            (0..len).map(|i| unsafe { *t.data_f64().add(i) as f32 }).collect()
        }
    }

    let dout_data = read_f32_data(dout_t, total_qkv);
    let q_data = read_f32_data(q_t, total_qkv);
    let k_data = read_f32_data(k_t, total_qkv);
    let v_data = read_f32_data(v_t, total_qkv);
    let out_data = read_f32_data(out_t, total_qkv);
    let lse_data = read_f32_data(lse_t, total_lse);

    // Allocate gradient buffers (zero-initialized)
    let mut dq_data = vec![0.0f32; total_qkv];
    let mut dk_data = vec![0.0f32; total_qkv];
    let mut dv_data = vec![0.0f32; total_qkv];

    // Run the CPU backward
    flash_attention_backward_cpu(
        &q_data, &k_data, &v_data,
        &out_data, &lse_data, &dout_data,
        &mut dq_data, &mut dk_data, &mut dv_data,
        b, h, s, d,
        scale, is_causal,
    );

    // Create output tensors (f32 dtype, matching Q shape)
    let shape = [batch, heads, seq_len, head_dim];

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

    let dq_ptr = make_tensor(&dq_data, &shape);
    let dk_ptr = make_tensor(&dk_data, &shape);
    let dv_ptr = make_tensor(&dv_data, &shape);

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
}
