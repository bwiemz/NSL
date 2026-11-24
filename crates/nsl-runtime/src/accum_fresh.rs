//! Gradient accumulators that are logically zero.
//!
//! Under the FASE-Deferred source-AD hook every parameter's gradient is added
//! into its accumulation buffer (`m_partial`) as it is computed, and the
//! optimizer step used to store zeros into every buffer once it had read
//! them, so the next window could add onto zeros. At Coder-1B that store is 4
//! bytes per parameter, 6.5 ms of the 47 ms step, and the first micro-batch
//! of each window then read those zeros back in every weight-gradient GEMM's
//! `beta = 1` epilogue.
//!
//! When the compiler admits it (see `stmt_train::optimizer_step`), the step
//! instead keeps its buffers and marks them *fresh*: their bytes are stale,
//! their value is zero. Every writer then asks this module before it adds:
//!
//! - a writer that covers the whole buffer ([`claim`]) takes the mark and
//!   overwrites, a GEMM with `beta = 0`. `0 + x` and `x` are the same bits for
//!   every `x` but `-0`: an epilogue product that underflows to `-0` is stored
//!   `-0` where the add onto `+0` stored `+0`. Nothing downstream tells them
//!   apart: the clip squares it, and AdamW's `m` and `v` start at `+0`, where
//!   `+0 + -0` is `+0`, so they and θ are the same bits;
//! - any other writer ([`materialize`]) takes the mark and zeroes the buffer
//!   first, then adds as before;
//! - before anything reads the buffers as gradients (the clip's sum of
//!   squares, the optimizer, the trailing window's rescale),
//!   [`nsl_accum_materialize`] zeroes whatever is still marked: a parameter
//!   no gradient reached in this window.
//!
//! So the values every reader sees are the ones the zeroing step gave. The
//! marks are keyed by the buffer's data address; the train block's teardown
//! drops them ([`nsl_accum_forget`]) before it frees the buffers, so an
//! address the allocator hands out again is never mistaken for a fresh
//! accumulator.
//!
//! `NSL_ACCUM_COUNTER=1` prints at exit how many buffers were marked, taken
//! by an overwrite, and zeroed: the gate's evidence that the overwrite ran.
//! `NSL_ACCUM_TEST_POISON=1` (gates only) fills each buffer with NaN when it
//! is marked, so a writer that adds onto a fresh buffer, or a read that comes
//! before the zeroing, turns the run's numbers into NaN instead of into the
//! previous window's gradient, which for a buffer no gradient reaches is zero
//! and would go unseen. A `beta = 0` GEMM must not read it either.

use std::collections::HashSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::tensor::NslTensor;

/// The data addresses of the fresh buffers.
static FRESH: Mutex<Option<HashSet<u64>>> = Mutex::new(None);
/// `FRESH`'s size, so a writer outside the protocol skips the lock.
static FRESH_LEN: AtomicUsize = AtomicUsize::new(0);

/// Buffers marked fresh, over the process.
pub static MARKED: AtomicU64 = AtomicU64::new(0);
/// Marks taken by a writer that overwrote the whole buffer.
pub static CLAIMED: AtomicU64 = AtomicU64::new(0);
/// Marks resolved by zeroing the buffer.
pub static ZEROED: AtomicU64 = AtomicU64::new(0);

fn with_fresh<R>(f: impl FnOnce(&mut HashSet<u64>) -> R) -> R {
    let mut guard = FRESH
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let set = guard.get_or_insert_with(HashSet::new);
    let r = f(set);
    FRESH_LEN.store(set.len(), Ordering::Release);
    r
}

/// Take `data`'s mark: `true` when the buffer was fresh, and is not any more.
fn take(data: u64) -> bool {
    if data == 0 || FRESH_LEN.load(Ordering::Acquire) == 0 {
        return false;
    }
    with_fresh(|s| s.remove(&data))
}

/// For a writer about to overwrite all of `t_ptr`: `true` when the buffer was
/// fresh, so the write must not read it (`beta = 0`). The mark is gone either
/// way; a `false` means add as usual.
pub fn claim(t_ptr: i64) -> bool {
    let fresh = take(NslTensor::from_ptr_ref(t_ptr).data as u64);
    if fresh {
        CLAIMED.fetch_add(1, Ordering::Relaxed);
    }
    fresh
}

/// For a writer that adds into `t_ptr` without covering all of it (or that
/// reads it): zero the buffer first if it is fresh.
pub fn materialize(t_ptr: i64) {
    if take(NslTensor::from_ptr_ref(t_ptr).data as u64) {
        crate::tensor::nsl_tensor_zero_inplace(t_ptr);
        ZEROED.fetch_add(1, Ordering::Relaxed);
    }
}

/// The tensors of an `NslList`, skipping null entries.
fn list_tensors(list_ptr: i64) -> Vec<i64> {
    if list_ptr == 0 {
        return Vec::new();
    }
    let list = crate::list::NslList::from_ptr(list_ptr);
    (0..list.len as usize)
        .map(|i| unsafe { *list.data.add(i) })
        .filter(|&t| t != 0)
        .collect()
}

/// `NSL_ACCUM_TEST_POISON=1`: see the module header.
fn poison_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NSL_ACCUM_TEST_POISON").ok().as_deref() == Some("1"))
}

/// Fill `t`'s bytes with 0xFF: a NaN in every f32.
fn poison(t: &NslTensor) {
    let bytes = (t.len as usize) * t.element_size();
    #[cfg(feature = "cuda")]
    if t.device > 0 {
        crate::cuda::inner::memset_d8_value(t.data, 0xFF, bytes);
        return;
    }
    if t.device == 0 && t.has_writable_storage() {
        unsafe { std::ptr::write_bytes(t.data as *mut u8, 0xFF, bytes) };
    }
}

/// After the optimizer step has read every accumulator in `list_ptr` and left
/// it in place: mark each one fresh (zero in value, its bytes stale).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_accum_mark_fresh(list_ptr: i64) {
    let tensors = list_tensors(list_ptr);
    if poison_enabled() {
        for &t in &tensors {
            poison(NslTensor::from_ptr_ref(t));
        }
    }
    let datas: Vec<u64> = tensors
        .into_iter()
        .map(|t| NslTensor::from_ptr_ref(t).data as u64)
        .filter(|&d| d != 0)
        .collect();
    let n = datas.len() as u64;
    with_fresh(|s| s.extend(datas));
    MARKED.fetch_add(n, Ordering::Relaxed);
}

/// Before the accumulators in `list_ptr` are read as gradients: zero every
/// one still fresh (no gradient reached it since it was marked).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_accum_materialize(list_ptr: i64) {
    if FRESH_LEN.load(Ordering::Acquire) == 0 {
        return;
    }
    for t in list_tensors(list_ptr) {
        materialize(t);
    }
}

/// Before the accumulators in `list_ptr` are freed: drop their marks without
/// writing them.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_accum_forget(list_ptr: i64) {
    if FRESH_LEN.load(Ordering::Acquire) == 0 {
        return;
    }
    for t in list_tensors(list_ptr) {
        take(NslTensor::from_ptr_ref(t).data as u64);
    }
}

/// `(marked, claimed, zeroed)` so far.
pub fn counts() -> (u64, u64, u64) {
    (
        MARKED.load(Ordering::Relaxed),
        CLAIMED.load(Ordering::Relaxed),
        ZEROED.load(Ordering::Relaxed),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The marks are process-global; the tests that touch them run in one
    /// test so they cannot interleave.
    #[test]
    fn marks_are_taken_once_and_zero_what_they_cover() {
        let a = crate::cpu::create_tensor_with_shape_rs_dtype(&[4], 1);
        let b = crate::cpu::create_tensor_with_shape_rs_dtype(&[3], 1);
        for (t, v) in [(a, 2.5f32), (b, -1.0f32)] {
            let tt = NslTensor::from_ptr(t);
            for i in 0..tt.len as usize {
                unsafe { *(tt.data as *mut f32).add(i) = v };
            }
        }
        let list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(list, a);
        crate::list::nsl_list_push(list, b);
        let read = |t: i64| -> Vec<f32> {
            let tt = NslTensor::from_ptr_ref(t);
            (0..tt.len as usize)
                .map(|i| unsafe { *(tt.data as *const f32).add(i) })
                .collect()
        };

        // Unmarked: nothing is fresh, nothing is written.
        assert!(!claim(a));
        nsl_accum_materialize(list);
        assert_eq!(read(a), vec![2.5; 4]);

        // Marked: an overwriting writer takes `a` once; `b`, untouched, is
        // zeroed by the materialize before the read.
        let (m0, c0, z0) = counts();
        nsl_accum_mark_fresh(list);
        assert!(claim(a));
        assert!(!claim(a), "a mark is taken once");
        assert_eq!(
            read(a),
            vec![2.5; 4],
            "claim leaves the bytes to the overwrite"
        );
        nsl_accum_materialize(list);
        assert_eq!(read(b), vec![0.0; 3]);
        assert_eq!(read(a), vec![2.5; 4], "a taken mark is not zeroed later");
        let (m1, c1, z1) = counts();
        assert_eq!((m1 - m0, c1 - c0, z1 - z0), (2, 1, 1));

        // An adding writer zeroes a fresh buffer before it adds.
        nsl_accum_mark_fresh(list);
        materialize(a);
        assert_eq!(read(a), vec![0.0; 4]);

        // Forgotten marks write nothing.
        let tb = NslTensor::from_ptr(b);
        unsafe { *(tb.data as *mut f32) = 7.0 };
        nsl_accum_forget(list);
        nsl_accum_materialize(list);
        assert_eq!(read(b)[0], 7.0);
        assert!(!claim(b));

        crate::list::nsl_list_free(list);
        crate::tensor::nsl_tensor_free(a);
        crate::tensor::nsl_tensor_free(b);
    }
}
