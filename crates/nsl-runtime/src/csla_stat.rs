//! CSLA Stage-2 (`--layerwise-accum`): window-backward anti-vacuity counter.
//!
//! The layerwise emitter calls `nsl_csla_window_mark` once per accumulation
//! window right before the buffered backward loop runs. The differential gate
//! asserts the count is > 0 on the `--layerwise-accum` arm and == 0 on the
//! baseline arm, so a silently-inert flag can never pass the parity check.
//! The counter is always live (one relaxed atomic per optimizer step); the
//! `NSL_CSLA_COUNTER=1` env var only gates the atexit report (see `args.rs`).

use std::sync::atomic::{AtomicU64, Ordering};

/// Count of CSLA window-backward phases executed.
pub static CSLA_WINDOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Emitted at the head of every CSLA window-backward phase.
#[no_mangle]
pub extern "C" fn nsl_csla_window_mark() {
    CSLA_WINDOW_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// In-process numeric getter (same family as `nsl_fase_fused_step_count`):
/// lets gates assert the layerwise path actually fired without stderr scraping.
#[no_mangle]
pub extern "C" fn nsl_csla_window_count() -> i64 {
    CSLA_WINDOW_COUNT.load(Ordering::Relaxed) as i64
}

/// D1b: one-time pointer-tie guard, emitted at train setup under
/// `--layerwise-accum`.
///
/// Pointer-tied weights (two model fields aliasing one storage) are
/// invisible to the compile-time layerwise analysis — each alias would
/// classify independently, and a per-layer in-place θ update through one
/// alias would corrupt the pending backward reads through the other.
/// Structural ties (the same compound field read twice) are handled by
/// source-AD memoization + the CrossLayer classification; this guard covers
/// only the runtime-aliasing representation, which the compiler cannot see.
///
/// Aborts loudly on the first aliased pair: NslTensor-pointer identity
/// (assignment ties), shared `data_owner` (two views of one buffer), and
/// byte-interval OVERLAP of the backing storage (offset views into one
/// allocation — plain data-pointer equality misses those; review D1b-2).
#[no_mangle]
pub extern "C" fn nsl_csla_assert_params_unaliased(list_ptr: i64) {
    if list_ptr == 0 {
        return;
    }
    let list = unsafe { &*(list_ptr as *const crate::list::NslList) };
    let n = list.len as usize;
    let slots: &[i64] = unsafe { std::slice::from_raw_parts(list.data, n) };
    // Conservative element width per dtype; training params are f64/f32
    // (exact), half-family dtypes get 2, everything else 1 (an
    // under-estimate can only shrink intervals, i.e. miss exotic-dtype
    // overlaps — never false-abort).
    let elem_bytes = |dtype: u16| -> usize {
        match dtype {
            crate::tensor::DTYPE_F64 => 8,
            crate::tensor::DTYPE_F32 => 4,
            crate::tensor::DTYPE_FP16 | crate::tensor::DTYPE_BF16 => 2,
            _ => 1,
        }
    };
    for i in 0..n {
        if slots[i] == 0 {
            continue;
        }
        let ti = unsafe { &*(slots[i] as *const crate::tensor::NslTensor) };
        let start_i = ti.data as usize;
        let end_i = start_i + (ti.len.max(0) as usize) * elem_bytes(ti.dtype);
        for (j, &sj) in slots.iter().enumerate().skip(i + 1) {
            if sj == 0 {
                continue;
            }
            let tj = unsafe { &*(sj as *const crate::tensor::NslTensor) };
            let start_j = tj.data as usize;
            let end_j = start_j + (tj.len.max(0) as usize) * elem_bytes(tj.dtype);
            let same_tensor = slots[i] == sj;
            let shared_owner =
                ti.data_owner != 0 && ti.data_owner == tj.data_owner;
            let view_of_each_other =
                ti.data_owner == sj || tj.data_owner == slots[i];
            let overlap = start_i < end_j && start_j < end_i;
            if same_tensor || shared_owner || view_of_each_other || overlap {
                eprintln!(
                    "[csla] FATAL: model parameters {i} and {j} alias the same \
                     storage (tensor {} vs {}, data [{start_i:#x},{end_i:#x}) vs \
                     [{start_j:#x},{end_j:#x}), owners {} / {}). Pointer-tied or \
                     view-tied weights are unsupported under --layerwise-accum: \
                     a per-layer in-place update through one alias would corrupt \
                     the other alias's pending backward. Untie the weights or \
                     drop --layerwise-accum.",
                    slots[i], sj, ti.data_owner, tj.data_owner
                );
                std::process::abort();
            }
        }
    }
}
