//! Lazy-init dedicated CUstream for inspect copies.
//!
//! Sync model: codegen-emitted hook calls cuEventRecord on the compute stream
//! after the producing kernel, then cuStreamWaitEvent on this inspect stream
//! BEFORE issuing the memcpy. That ordering is not enforced here; this module
//! only owns the thread-local stream handle itself.

#![cfg(feature = "cuda")]

use cudarc::driver::sys;
use std::cell::RefCell;

thread_local! {
    static INSPECT_STREAM: RefCell<Option<sys::CUstream>> = const { RefCell::new(None) };
}

/// Returns the thread-local inspect stream, creating it on first access.
pub fn current_inspect_stream() -> sys::CUstream {
    INSPECT_STREAM.with(|s| {
        let mut g = s.borrow_mut();
        if g.is_none() {
            let mut stream: sys::CUstream = std::ptr::null_mut();
            unsafe {
                let res = sys::cuStreamCreate(&mut stream, 0);
                if res != sys::CUresult::CUDA_SUCCESS {
                    panic!("cuStreamCreate failed: {:?}", res);
                }
            }
            *g = Some(stream);
        }
        g.unwrap()
    })
}
