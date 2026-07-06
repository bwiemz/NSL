//! CFIE Cycle 12: host token-buffer <-> tokenizer-tensor bridge FFIs.
//!
//! The generation driver (`nsl_cfie_generate`, engine.rs) speaks host
//! i64 arrays; the tokenizer surface (`nsl_tokenizer_encode` /
//! `nsl_tokenizer_decode`, tokenizer.rs) speaks 1-D f64 `NslTensor`s.
//! These two FFIs are the missing conversion in each direction, closing
//! the Cycle-11 deferrals ("prompt is byte-baked", "generate() prints a
//! count, not text"):
//!
//!   * `nsl_cfie_tokens_to_tensor`: host i64 token ids -> NEW 1-D f64
//!     tensor (feeds `nsl_tokenizer_decode` so generated ids become
//!     TEXT).
//!   * `nsl_cfie_tensor_to_tokens`: 1-D f64 tensor (the
//!     `nsl_tokenizer_encode` output) -> host i64 buffer (feeds
//!     `nsl_cfie_generate`'s prompt ABI so the configured prompt is
//!     runtime-encoded by the REAL tokenizer).
//!
//! Both are pure host copies — no CUDA gate — and follow the house
//! contracts: honest refusals (never fabricate token ids), the Cycle-9
//! `isize::MAX` host-read guard, and the runtime's tensor ownership
//! convention (the returned tensor is a fresh refcount-1 owner the
//! caller frees with `nsl_tensor_free`).

use std::ffi::c_void;

use crate::memory::checked_alloc;
use crate::tensor::{NslTensor, TENSOR_MAGIC};

/// Byte count for a host read/write of `count` 8-byte elements (i64 or
/// f64).  `slice::from_raw_parts` is UB past `isize::MAX` total bytes (a
/// process abort, NOT a recoverable error), so both FFIs must refuse
/// oversized counts up front — the Cycle-9 `checked_host_f32_bytes`
/// pattern (engine.rs).  Returns `None` to refuse (`count <= 0` or the
/// byte total exceeding `isize::MAX`).
fn checked_host_8byte_len(count: i64) -> Option<usize> {
    if count <= 0 {
        return None;
    }
    (count as u64)
        .checked_mul(8)
        .filter(|&b| b <= isize::MAX as u64)
        .map(|_| count as usize)
}

/// Build a fresh 1-D f64 CPU `NslTensor` from `values`, following the
/// tokenizer module's `make_1d_tensor` layout (tokenizer.rs — the house
/// per-file convention for constructing the tokenizer-ABI tensor):
/// heap shape/strides/data via `checked_alloc`, `NslTensor::new` (magic
/// set, refcount 1, owns_data 1), published through `NslTensor::publish`
/// so scope tracking + alloc accounting see it.  The caller owns the
/// reference and frees it with `nsl_tensor_free`.
fn make_1d_f64_tensor(values: &[f64]) -> i64 {
    let len = values.len() as i64;
    let ndim: i64 = 1;

    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = len };

    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let data = checked_alloc(std::mem::size_of_val(values)) as *mut f64;
    for (i, &v) in values.iter().enumerate() {
        unsafe { *data.add(i) = v };
    }

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0, // device: CPU
        0, // dtype: f64 (the tokenizer ABI dtype)
        1, // owns_data
        0, // data_owner: self
    ));
    NslTensor::publish(tensor)
}

/// Copy `count` i64 token ids from host memory at `tokens_ptr` into a
/// NEW 1-D f64 `NslTensor` (the dtype `nsl_tokenizer_decode` consumes).
/// Returns the tensor pointer (> 0) — a fresh refcount-1 owner the
/// caller frees with `nsl_tensor_free` — or 0 on bad args: null
/// `tokens_ptr`, `count <= 0`, or a `count` whose byte total would
/// exceed `isize::MAX` (the Cycle-9 host-read guard; reading past it is
/// UB, so an oversized count must refuse, not abort).
///
/// The i64 -> f64 value copy is exact for every real token id (f64 holds
/// integers up to 2^53 exactly; vocab ids are far below), matching the
/// precision contract of `nsl_tokenizer_encode`'s own f64 ids.
///
/// # Safety
/// `tokens_ptr` must point to at least `count` readable i64s.
#[no_mangle]
pub extern "C" fn nsl_cfie_tokens_to_tensor(tokens_ptr: i64, count: i64) -> i64 {
    if tokens_ptr == 0 {
        return 0;
    }
    let n = match checked_host_8byte_len(count) {
        Some(n) => n,
        None => return 0,
    };
    let src = unsafe { std::slice::from_raw_parts(tokens_ptr as *const i64, n) };
    let values: Vec<f64> = src.iter().map(|&t| t as f64).collect();
    make_1d_f64_tensor(&values)
}

/// Read a 1-D f64 `NslTensor` of token ids (the `nsl_tokenizer_encode`
/// output shape) and write them as i64 into the host buffer at
/// `out_ptr`, clamped to `cap` (no overrun).  Returns the FULL token
/// count — the caller detects truncation by `count > cap`, mirroring
/// `nsl_cfie_generate`'s out-buffer contract.  A 0-len tensor returns 0
/// (nothing written).
///
/// Refuses (-1) on: null/invalid tensor (magic check against
/// `TENSOR_MAGIC`), a non-CPU or non-f64 tensor, a non-1-D tensor, a
/// null `out_ptr` (when anything would be written), a negative `cap`,
/// a tensor length whose byte total exceeds `isize::MAX` (host-read
/// guard), or ANY element that is not a non-negative integer
/// representable in i64 (NaN / negative / fractional / overflow) —
/// a corrupt encode must never silently become token 0.
///
/// # Safety
/// `tensor_ptr` must be a live `NslTensor*`; `out_ptr` must point to at
/// least `min(len, cap)` writable i64s.
#[no_mangle]
pub extern "C" fn nsl_cfie_tensor_to_tokens(tensor_ptr: i64, out_ptr: i64, cap: i64) -> i64 {
    if tensor_ptr == 0 || cap < 0 {
        return -1;
    }
    // Raw read + magic check (NOT NslTensor::from_ptr — its debug_assert
    // would abort a debug build on garbage instead of refusing).
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    if t.magic != TENSOR_MAGIC {
        return -1;
    }
    // The tokenizer-encode contract: 1-D f64 host tensor.
    if t.device != 0 || t.dtype != 0 || t.ndim != 1 || t.data.is_null() {
        return -1;
    }
    if t.len < 0 {
        return -1;
    }
    if t.len == 0 {
        return 0;
    }
    // Host-read guard (Cycle-9 pattern): len * 8 bytes must stay within
    // isize::MAX or the slice below is UB.
    let n = match checked_host_8byte_len(t.len) {
        Some(n) => n,
        None => return -1,
    };
    if out_ptr == 0 {
        return -1;
    }
    let src = unsafe { std::slice::from_raw_parts(t.data as *const f64, n) };
    // Validate EVERY element (not just the ones that fit in cap) before
    // writing anything: a corrupt encode must refuse, never silently
    // truncate into token 0.  2^63 is the first f64 double boundary
    // above i64::MAX; `v < 2^63` exactly bounds representable i64s.
    const I64_BOUND: f64 = 9_223_372_036_854_775_808.0; // 2^63
    for &v in src {
        if !v.is_finite() || v < 0.0 || v.trunc() != v || v >= I64_BOUND {
            return -1;
        }
    }
    let write_n = n.min(cap as usize);
    if write_n > 0 {
        let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut i64, write_n) };
        for (d, &v) in dst.iter_mut().zip(src.iter()) {
            *d = v as i64;
        }
    }
    t.len
}

// ---------------------------------------------------------------------------
// Unit tests — pure host logic, run on every build (no CUDA gate).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::nsl_tensor_free;

    #[test]
    fn tokens_to_tensor_roundtrip_exact() {
        let tokens: [i64; 5] = [0, 3, 17, 127, 4095];
        let tptr = nsl_cfie_tokens_to_tensor(tokens.as_ptr() as i64, tokens.len() as i64);
        assert!(tptr > 0, "valid args must produce a tensor");
        {
            let t = NslTensor::from_ptr(tptr);
            assert_eq!(t.ndim, 1);
            assert_eq!(t.len, 5);
            assert_eq!(t.dtype, 0, "tokenizer ABI dtype is f64");
            assert_eq!(t.device, 0);
            let data = t.data as *const f64;
            for (i, &tok) in tokens.iter().enumerate() {
                assert_eq!(unsafe { *data.add(i) }, tok as f64);
            }
        }
        // Round-trip back through tensor_to_tokens.
        let mut out = [0i64; 5];
        let n = nsl_cfie_tensor_to_tokens(tptr, out.as_mut_ptr() as i64, out.len() as i64);
        assert_eq!(n, 5);
        assert_eq!(out, tokens);
        nsl_tensor_free(tptr);
    }

    #[test]
    fn tokens_to_tensor_refuses_bad_args() {
        let tokens: [i64; 2] = [1, 2];
        // Null pointer.
        assert_eq!(nsl_cfie_tokens_to_tensor(0, 2), 0);
        // Non-positive count.
        assert_eq!(nsl_cfie_tokens_to_tensor(tokens.as_ptr() as i64, 0), 0);
        assert_eq!(nsl_cfie_tokens_to_tensor(tokens.as_ptr() as i64, -3), 0);
        // isize guard: count * 8 overflows isize::MAX -> refuse, not abort.
        assert_eq!(
            nsl_cfie_tokens_to_tensor(tokens.as_ptr() as i64, i64::MAX / 4),
            0
        );
    }

    #[test]
    fn tensor_to_tokens_truncation_returns_full_count() {
        let tokens: [i64; 4] = [10, 20, 30, 40];
        let tptr = nsl_cfie_tokens_to_tensor(tokens.as_ptr() as i64, 4);
        assert!(tptr > 0);
        // cap 2 < len 4: first 2 written, sentinel intact, FULL count back.
        let mut out = [0i64; 3];
        let sentinel = -777i64;
        out[2] = sentinel;
        let n = nsl_cfie_tensor_to_tokens(tptr, out.as_mut_ptr() as i64, 2);
        assert_eq!(n, 4, "return value is the FULL count (truncation signal)");
        assert_eq!(&out[..2], &[10, 20]);
        assert_eq!(out[2], sentinel, "no write past cap");
        // cap 0: nothing written, full count still returned.
        let n = nsl_cfie_tensor_to_tokens(tptr, out.as_mut_ptr() as i64, 0);
        assert_eq!(n, 4);
        assert_eq!(&out[..2], &[10, 20], "cap 0 must not write");
        nsl_tensor_free(tptr);
    }

    #[test]
    fn tensor_to_tokens_zero_len_tensor_returns_zero() {
        let tptr = make_1d_f64_tensor(&[]);
        assert!(tptr > 0);
        let mut out = [0i64; 1];
        assert_eq!(
            nsl_cfie_tensor_to_tokens(tptr, out.as_mut_ptr() as i64, 1),
            0
        );
        nsl_tensor_free(tptr);
    }

    #[test]
    fn tensor_to_tokens_refuses_null_and_bad_meta() {
        let mut out = [0i64; 4];
        let out_ptr = out.as_mut_ptr() as i64;
        // Null tensor / negative cap.
        assert_eq!(nsl_cfie_tensor_to_tokens(0, out_ptr, 4), -1);
        let tptr = make_1d_f64_tensor(&[1.0, 2.0]);
        assert_eq!(nsl_cfie_tensor_to_tokens(tptr, out_ptr, -1), -1);
        // Null out_ptr with a non-empty tensor.
        assert_eq!(nsl_cfie_tensor_to_tokens(tptr, 0, 4), -1);
        // Bad magic (freed tensor poisons the magic to TENSOR_FREED — a
        // stale pointer must refuse, not read).
        {
            let t = unsafe { &mut *(tptr as *mut NslTensor) };
            let live = t.magic;
            t.magic = 0x0000DEAD;
            assert_eq!(nsl_cfie_tensor_to_tokens(tptr, out_ptr, 4), -1);
            t.magic = live;
        }
        // Non-f64 dtype (the encode ABI is f64; refuse f32 rather than
        // silently reinterpret the bytes).
        {
            let t = unsafe { &mut *(tptr as *mut NslTensor) };
            t.dtype = 1;
            assert_eq!(nsl_cfie_tensor_to_tokens(tptr, out_ptr, 4), -1);
            t.dtype = 0;
        }
        // Non-1-D.
        {
            let t = unsafe { &mut *(tptr as *mut NslTensor) };
            t.ndim = 2;
            assert_eq!(nsl_cfie_tensor_to_tokens(tptr, out_ptr, 4), -1);
            t.ndim = 1;
        }
        nsl_tensor_free(tptr);
    }

    #[test]
    fn tensor_to_tokens_refuses_non_token_values() {
        let mut out = [0i64; 4];
        let out_ptr = out.as_mut_ptr() as i64;
        // NaN, negative, fractional, and >= 2^63 each refuse — a corrupt
        // encode must not silently become token 0.
        for bad in [f64::NAN, -1.0, 2.5, 9.3e18, f64::INFINITY] {
            let tptr = make_1d_f64_tensor(&[1.0, bad, 3.0]);
            assert_eq!(
                nsl_cfie_tensor_to_tokens(tptr, out_ptr, 4),
                -1,
                "value {bad} must refuse"
            );
            nsl_tensor_free(tptr);
        }
        // Validation covers elements PAST cap too: a corrupt tail refuses
        // even when the write window is clean.
        let tptr = make_1d_f64_tensor(&[1.0, 2.0, f64::NAN]);
        assert_eq!(nsl_cfie_tensor_to_tokens(tptr, out_ptr, 2), -1);
        nsl_tensor_free(tptr);
    }
}
