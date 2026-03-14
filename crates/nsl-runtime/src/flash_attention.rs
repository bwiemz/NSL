//! FlashAttention-2 runtime launch wrappers.
//!
//! These functions compute grid/block dimensions from tensor shapes, marshal
//! arguments, and call `kernel_launch()` with pre-baked PTX from .rodata.
//! No PTX generation happens at runtime. On non-CUDA builds, falls back to
//! naive matmul+softmax attention path.

#[cfg(feature = "cuda")]
use std::ffi::c_void;

/// FlashAttention-2 kernel launch wrapper.
///
/// All params are i64 for Cranelift ABI compatibility (same pattern as nsl_kernel_launch).
/// f32 scale is passed as i64 and reconstructed via f32::from_bits(scale as u32).
///
/// Returns 0 on success, non-zero CUDA error code on failure.
#[no_mangle]
pub extern "C" fn nsl_flash_attention(
    q_ptr: i64, k_ptr: i64, v_ptr: i64,
    out_ptr: i64, scale_bits: i64,
    batch: i64, heads: i64, seq_len: i64, head_dim: i64,
    block_table_ptr: i64,
    k_pool_ptr: i64, v_pool_ptr: i64,
    block_size: i64,
    cos_ptr: i64, sin_ptr: i64,
    seq_ids_ptr: i64, seq_lens_ptr: i64,
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
        let mut cos = cos_ptr as u64;
        let mut sin = sin_ptr as u64;
        let mut sids = seq_ids_ptr as u64;
        let mut slens = seq_lens_ptr as u64;

        let args: [*mut c_void; 17] = [
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
        ];

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
        let _ = (cos_ptr, sin_ptr, seq_ids_ptr, seq_lens_ptr);
        let _ = (shared_mem_bytes, ptx_ptr, name_ptr, block_q, _block_kv);
        eprintln!("[nsl] FlashAttention requires CUDA. Use naive path (no @flash_attention decorator).");
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
