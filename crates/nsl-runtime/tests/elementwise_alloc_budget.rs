//! Drift gate: a CPU elementwise op on a small tensor performs exactly FOUR
//! heap allocations — the result's `NslTensor` struct, its shape, its strides
//! and its data — in both the f32 and the f64 kernel, whether or not the
//! autodiff tape is recording.
//!
//! Before roadmap item C4 the same op made THIRTEEN: `nsl_tensor_contiguous`
//! allocated the expected strides just to compare them (twice per op, once per
//! operand), each recorded tape node boxed its two operand shapes in a `Vec`
//! apiece, and the elementwise kernel built five scratch `Vec`s (two shapes,
//! two padded shapes, the output shape) per call before touching a single
//! element. At `[8, 8]` the op is a few dozen flops, so those nine
//! allocations WERE the op: the `record/elementwise_*` tape benches dropped by
//! a third when they went. This test pins the count so the next helper that
//! quietly `collect()`s a shape or `to_vec()`s a stride list on the hot path
//! fails here instead of showing up as a slow-creeping bench regression.
//!
//! Counting is done with a `#[global_allocator]` that tallies every call into
//! the system allocator (allocations and reallocations only — frees are not
//! counted, so the per-op frees of the chain's intermediates are invisible to
//! it). The count is taken over the ops alone: the tape is started, the
//! counter zeroed, the chain run, the counter read, the tape stopped. Every
//! arm runs the same chain once first, un-counted: that hoists first-touch
//! costs (the tape thread-local's destructor registration, which allocates
//! on Windows) out of the window, and under recording leaves the tape's
//! `ops` vector holding the capacity the chain needs (`nsl_tape_stop`
//! clears it without shrinking) — a cold tape's growth would otherwise be
//! charged to the ops.
//!
//! All four arms live in ONE test function: the counter is process-global,
//! and a second test in this binary would run on its own thread and add its
//! allocations to whichever arm was mid-window.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use nsl_runtime::autodiff::{nsl_tape_start, nsl_tape_stop};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_add, nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get_dtype,
    nsl_tensor_mul, nsl_tensor_randn,
};

struct CountingAlloc;

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // A growth is a new allocation as far as the budget is concerned: a
        // `Vec::push` that spills is exactly the kind of hot-path cost this
        // gate exists to catch.
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

/// The result tensor's struct, shape, strides and data. Nothing else.
const ALLOCS_PER_OP: usize = 4;

const OPS: usize = 64;

const DTYPE_F64: i64 = 0;
const DTYPE_F32: i64 = 1;

fn shape_list(dims: &[i64]) -> i64 {
    let list = nsl_list_new();
    for &d in dims {
        nsl_list_push(list, d);
    }
    list
}

/// What NSL-level code makes: `randn` places an f32 tensor on the host.
fn randn_f32(dims: &[i64]) -> i64 {
    let list = shape_list(dims);
    let t = nsl_tensor_randn(list);
    nsl_list_free(list);
    t
}

/// An f64 `[8, 8]` over static data (`owns_data = 0`): no public constructor
/// makes an f64 tensor directly, and the f64 kernel is a separate
/// monomorphization of the same broadcast plan, so it gets its own arm.
fn static_f64_8x8(which: usize) -> i64 {
    static DATA: [[f64; 64]; 2] = {
        let mut d = [[0.0; 64]; 2];
        let mut i = 0;
        while i < 64 {
            d[0][i] = 0.25 * (i as f64) - 8.0;
            d[1][i] = 1.0 + 0.5 * (i as f64);
            i += 1;
        }
        d
    };
    let list = shape_list(&[8, 8]);
    let t = nsl_tensor_from_static(DATA[which].as_ptr() as i64, list, DTYPE_F64);
    nsl_list_free(list);
    t
}

/// The `record/elementwise_*` bench body: `OPS` alternating `add`/`mul` ops
/// on one `[8, 8]` tensor, each intermediate freed as soon as its successor
/// exists. Returns the last result for the caller to free.
fn chain(x: i64, w: i64) -> i64 {
    let mut cur = x;
    for i in 0..OPS {
        let next = if i % 2 == 0 { nsl_tensor_add(cur, w, 0) } else { nsl_tensor_mul(cur, w, 0) };
        if cur != x {
            nsl_tensor_free(cur);
        }
        cur = next;
    }
    cur
}

/// Allocations made by the chain alone, tape recording or not.
fn count_chain(x: i64, w: i64, params: i64, record: bool) -> usize {
    // Warm up on both arms: the first `is_recording()` on a thread is the
    // first touch of the `TAPE` thread-local, and on targets without a
    // native TLS destructor hook (Windows among the CI lanes) registering
    // its destructor pushes onto a `Vec` through this allocator. Under
    // recording the same chain length also leaves `ops` with the capacity
    // the measured chain needs.
    if record {
        nsl_tape_start(params);
    }
    nsl_tensor_free(chain(x, w));
    if record {
        nsl_tape_stop();
        nsl_tape_start(params);
    }
    ALLOCS.store(0, Ordering::Relaxed);
    let last = chain(x, w);
    let n = ALLOCS.load(Ordering::Relaxed);
    if record {
        nsl_tape_stop();
    }
    nsl_tensor_free(last);
    n
}

/// Both tape arms for one operand pair; returns `(unrecorded, recorded)`.
fn count_both_arms(x: i64, w: i64, dtype: i64, label: &str) -> (usize, usize) {
    // The budget is per kernel; a dtype flip in a constructor would move the
    // count to a different code path and the number would mean something else.
    assert_eq!(nsl_tensor_get_dtype(x), dtype, "{label}: x has the wrong dtype");
    assert_eq!(nsl_tensor_get_dtype(w), dtype, "{label}: w has the wrong dtype");
    let params = shape_list(&[w]);
    let unrecorded = count_chain(x, w, params, false);
    let recorded = count_chain(x, w, params, true);
    nsl_list_free(params);
    nsl_tensor_free(w);
    nsl_tensor_free(x);
    (unrecorded, recorded)
}

#[test]
fn cpu_elementwise_op_allocates_exactly_its_result_on_every_arm() {
    let arms = [
        ("f32", count_both_arms(randn_f32(&[8, 8]), randn_f32(&[8, 8]), DTYPE_F32, "f32")),
        ("f64", count_both_arms(static_f64_8x8(0), static_f64_8x8(1), DTYPE_F64, "f64")),
    ];
    for (label, (unrecorded, recorded)) in arms {
        assert_eq!(
            unrecorded,
            OPS * ALLOCS_PER_OP,
            "{label} unrecorded chain: {unrecorded} allocations over {OPS} ops ({:.2}/op), \
             budget is {ALLOCS_PER_OP}/op",
            unrecorded as f64 / OPS as f64
        );
        assert_eq!(
            recorded,
            OPS * ALLOCS_PER_OP,
            "{label} recorded chain: {recorded} allocations over {OPS} ops ({:.2}/op), \
             budget is {ALLOCS_PER_OP}/op — a tape node must not allocate (its shapes \
             are inline `TapeShape`s)",
            recorded as f64 / OPS as f64
        );
    }
}
