// crates/nsl-runtime/src/kv_compress/ffi.rs
//! M42: FFI exports for KV-cache compression.

use std::sync::Mutex;

use super::KvQuantScheme;
use super::quantize::{self, KvBlockQuantMeta};
use super::sliding_window::SlidingWindowManager;
use super::h2o::H2OManager;

// ---------------------------------------------------------------------------
// Quantization FFI
// ---------------------------------------------------------------------------

/// Quantize incoming K/V values and store in compressed format.
///
/// Parameters (all i64 for Cranelift):
/// - raw_k/raw_v: pointers to f32 K/V values for new tokens
/// - block_k/block_v: pointers to target blocks (u8 for quantized, f32 for none)
/// - meta_k/meta_v: pointers to KvBlockQuantMeta (null for Fp8/None)
/// - token_offset: position within block
/// - num_tokens: tokens to quantize
/// - num_heads: attention heads
/// - head_dim: dimension per head
/// - scheme: KvQuantScheme discriminant
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_kv_quantize_and_store(
    raw_k: i64,
    raw_v: i64,
    block_k: i64,
    block_v: i64,
    meta_k: i64,
    meta_v: i64,
    _token_offset: i64,
    num_tokens: i64,
    num_heads: i64,
    head_dim: i64,
    scheme: i64,
) -> i64 {
    let qs = KvQuantScheme::from_i64(scheme);
    let n = num_heads as usize * num_tokens as usize * head_dim as usize;

    if raw_k == 0 || raw_v == 0 || block_k == 0 || block_v == 0 {
        return -1;
    }

    // Read input values
    let k_values = unsafe { std::slice::from_raw_parts(raw_k as *const f32, n) };
    let v_values = unsafe { std::slice::from_raw_parts(raw_v as *const f32, n) };

    match qs {
        KvQuantScheme::None => {
            // Direct copy (no quantization)
            unsafe {
                std::ptr::copy_nonoverlapping(raw_k as *const u8, block_k as *mut u8, n * 4);
                std::ptr::copy_nonoverlapping(raw_v as *const u8, block_v as *mut u8, n * 4);
            }
        }
        KvQuantScheme::Int8PerHead => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut i8, n) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut i8, n) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            quantize::quantize_int8_per_head(k_values, k_out, k_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
            quantize::quantize_int8_per_head(v_values, v_out, v_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
        }
        KvQuantScheme::Int8PerToken => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut i8, n) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut i8, n) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            quantize::quantize_int8_per_token(k_values, k_out, k_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
            quantize::quantize_int8_per_token(v_values, v_out, v_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
        }
        KvQuantScheme::Int4PerGroup => {
            let packed_len = n.div_ceil(2);
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut u8, packed_len) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut u8, packed_len) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            quantize::quantize_int4_per_group(k_values, k_out, k_meta, quantize::INT4_GROUP_SIZE);
            quantize::quantize_int4_per_group(v_values, v_out, v_meta, quantize::INT4_GROUP_SIZE);
        }
        KvQuantScheme::Fp8 => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut u8, n) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut u8, n) };
            quantize::quantize_fp8(k_values, k_out);
            quantize::quantize_fp8(v_values, v_out);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Sliding Window FFI
// ---------------------------------------------------------------------------

static SW_CTX: Mutex<Option<SlidingWindowManager>> = Mutex::new(None);

/// Initialize sliding window manager.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_kv_sliding_window_init(
    window: i64,
    sinks: i64,
    block_size: i64,
) -> i64 {
    let mut guard = SW_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(SlidingWindowManager::new(
        window as usize,
        sinks as usize,
        block_size as usize,
    ));
    0
}

/// Check sliding window eviction. Returns number of blocks to evict.
/// Evicted block indices are written to `evict_out_ptr` if non-null.
#[no_mangle]
pub extern "C" fn nsl_kv_sliding_window_check(
    _seq_id: i64,
    current_len: i64,
    evict_out_ptr: i64,
    max_evict: i64,
) -> i64 {
    let guard = SW_CTX.lock().unwrap();
    let mgr = guard.as_ref().expect("nsl_kv_sliding_window_init not called");

    let evicted = mgr.check_eviction(current_len as usize);
    let count = evicted.len().min(max_evict as usize);

    if evict_out_ptr != 0 && count > 0 {
        let out = unsafe { std::slice::from_raw_parts_mut(evict_out_ptr as *mut u32, count) };
        for (i, &block_idx) in evicted.iter().take(count).enumerate() {
            out[i] = block_idx as u32;
        }
    }
    count as i64
}

/// Destroy sliding window manager.
#[no_mangle]
pub extern "C" fn nsl_kv_sliding_window_destroy() -> i64 {
    let mut guard = SW_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// H2O FFI
// ---------------------------------------------------------------------------

static H2O_CTX: Mutex<Option<H2OManager>> = Mutex::new(None);

/// Initialize H2O manager.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_init(budget: i64, sinks: i64, block_size: i64) -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(H2OManager::new(
        budget as usize,
        sinks as usize,
        block_size as usize,
    ));
    0
}

/// Accumulate attention scores for a sequence.
/// scores_ptr: *const f32, [seq_len] averaged scores from latest decode step.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_accumulate(
    seq_id: i64,
    scores_ptr: i64,
    seq_len: i64,
) -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    let mgr = guard.as_mut().expect("nsl_kv_h2o_init not called");

    if scores_ptr == 0 || seq_len <= 0 {
        return -1;
    }

    let scores = unsafe { std::slice::from_raw_parts(scores_ptr as *const f32, seq_len as usize) };
    mgr.accumulate_scores(seq_id as u64, scores);
    0
}

/// Check H2O eviction. Returns number of blocks to evict.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_check(
    seq_id: i64,
    current_len: i64,
    evict_out_ptr: i64,
    max_evict: i64,
) -> i64 {
    let guard = H2O_CTX.lock().unwrap();
    let mgr = guard.as_ref().expect("nsl_kv_h2o_init not called");

    let evicted = mgr.check_eviction(seq_id as u64, current_len as usize);
    let count = evicted.len().min(max_evict as usize);

    if evict_out_ptr != 0 && count > 0 {
        let out = unsafe { std::slice::from_raw_parts_mut(evict_out_ptr as *mut u32, count) };
        for (i, &block_idx) in evicted.iter().take(count).enumerate() {
            out[i] = block_idx as u32;
        }
    }
    count as i64
}

/// Remove sequence tracking from H2O manager.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_remove_sequence(seq_id: i64) -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    let mgr = guard.as_mut().expect("nsl_kv_h2o_init not called");
    mgr.remove_sequence(seq_id as u64);
    0
}

/// Destroy H2O manager.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_destroy() -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// Compression statistics
// ---------------------------------------------------------------------------

/// Get compression ratio for the active scheme.
/// Returns the compression ratio as f64 bits (e.g., 2.0 for INT8 vs FP16).
#[no_mangle]
pub extern "C" fn nsl_kv_compress_ratio(scheme: i64) -> i64 {
    let qs = KvQuantScheme::from_i64(scheme);
    let ratio = 2.0 / qs.bytes_per_element(); // relative to FP16 baseline
    f64::to_bits(ratio) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_kv_sliding_window_destroy();
        nsl_kv_h2o_destroy();
        guard
    }

    #[test]
    fn ffi_quantize_none_copies() {
        let k = vec![1.0f32, 2.0, 3.0, 4.0];
        let v = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut k_out = vec![0.0f32; 4];
        let mut v_out = vec![0.0f32; 4];

        let rc = nsl_kv_quantize_and_store(
            k.as_ptr() as i64, v.as_ptr() as i64,
            k_out.as_mut_ptr() as i64, v_out.as_mut_ptr() as i64,
            0, 0, // no metadata for None
            0, 1, 1, 4, // offset=0, tokens=1, heads=1, dim=4
            0, // scheme=None
        );
        assert_eq!(rc, 0);
        assert_eq!(k_out, k);
        assert_eq!(v_out, v);
    }

    #[test]
    fn ffi_quantize_int8() {
        let n = 2 * 1 * 4; // heads=2, tokens=1, dim=4
        let k: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * -0.3).collect();
        let mut k_out = vec![0i8; n];
        let mut v_out = vec![0i8; n];
        let mut k_meta = KvBlockQuantMeta::default();
        let mut v_meta = KvBlockQuantMeta::default();

        let rc = nsl_kv_quantize_and_store(
            k.as_ptr() as i64, v.as_ptr() as i64,
            k_out.as_mut_ptr() as i64, v_out.as_mut_ptr() as i64,
            &mut k_meta as *mut _ as i64, &mut v_meta as *mut _ as i64,
            0, 1, 2, 4, // offset=0, tokens=1, heads=2, dim=4
            1, // scheme=Int8PerHead
        );
        assert_eq!(rc, 0);
        assert_eq!(k_meta.scheme, 1);
        assert_eq!(k_meta.num_scales, 2);
    }

    #[test]
    fn ffi_sliding_window_lifecycle() {
        let _lock = setup();

        assert_eq!(nsl_kv_sliding_window_init(32, 4, 8), 0);
        assert_eq!(nsl_kv_sliding_window_init(32, 4, 8), -1); // double init

        // No eviction at 30 tokens
        let count = nsl_kv_sliding_window_check(0, 30, 0, 0);
        assert_eq!(count, 0);

        // Eviction at 100 tokens
        let mut out = vec![0u32; 16];
        let count = nsl_kv_sliding_window_check(0, 100, out.as_mut_ptr() as i64, 16);
        assert!(count > 0);

        assert_eq!(nsl_kv_sliding_window_destroy(), 0);
    }

    #[test]
    fn ffi_h2o_lifecycle() {
        let _lock = setup();

        assert_eq!(nsl_kv_h2o_init(4, 0, 1), 0);
        assert_eq!(nsl_kv_h2o_init(4, 0, 1), -1); // double init

        // Accumulate scores
        let scores = vec![5.0f32, 3.0, 1.0, 4.0, 0.5, 6.0];
        let rc = nsl_kv_h2o_accumulate(0, scores.as_ptr() as i64, 6);
        assert_eq!(rc, 0);

        // Check eviction (6 tokens, budget=4 -> evict 2)
        let mut out = vec![0u32; 4];
        let count = nsl_kv_h2o_check(0, 6, out.as_mut_ptr() as i64, 4);
        assert_eq!(count, 2);

        assert_eq!(nsl_kv_h2o_remove_sequence(0), 0);
        assert_eq!(nsl_kv_h2o_destroy(), 0);
    }

    #[test]
    fn ffi_null_pointer_returns_error() {
        let rc = nsl_kv_quantize_and_store(0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 0);
        assert_eq!(rc, -1);
    }

    #[test]
    fn ffi_compress_ratio() {
        let ratio_none = f64::from_bits(nsl_kv_compress_ratio(0) as u64);
        assert_eq!(ratio_none, 1.0); // FP16/FP16

        let ratio_int8 = f64::from_bits(nsl_kv_compress_ratio(1) as u64);
        assert_eq!(ratio_int8, 2.0); // FP16/INT8

        let ratio_int4 = f64::from_bits(nsl_kv_compress_ratio(3) as u64);
        assert_eq!(ratio_int4, 4.0); // FP16/INT4
    }
}
