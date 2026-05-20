//! M56 v1 runtime FFI surface. Codegen (Task 18) emits Cranelift function
//! calls to these `#[no_mangle] extern "C"` symbols; the symbols live here
//! as thin wrappers around the Rust pool/scheduler/mailbox APIs.
//!
//! Spec §3.4, refinement #3 (PortMessage payloads, not raw NslTensor).
//!
//! Pointer ownership convention for `PortMessage`:
//! - Producer constructs a `Box<PortMessage>` and `Box::into_raw` it for the FFI.
//! - The FFI moves the pointer ownership to the receiver.
//! - The consumer reconstructs `Box::from_raw` and reads/drops normally.
//! - `nsl_agent_mailbox_write` takes ownership of the pointer; the caller
//!   must not free the original Box.
//! - `nsl_agent_mailbox_read` returns ownership; the caller is responsible
//!   for freeing via `Box::from_raw`.

use crate::agent::mailbox::{PortMailbox, PortMessage};
use crate::agent::pool::{AcquireError, PipelineContext, PipelineContextPool};
use crate::agent::scheduler::ReactorScheduler;

/// Construct a pool with `pool_size` contexts. Returns null on failure
/// (e.g., size exceeds OS/GPU memory or v1 hard limit).
#[no_mangle]
pub extern "C" fn nsl_agent_pool_new(
    pool_size: u64,
    _pipeline_fn_id: u64,
) -> *mut PipelineContextPool {
    match PipelineContextPool::try_new(pool_size as usize, || {
        PipelineContext::new(ReactorScheduler::new())
    }) {
        Ok(p) => Box::into_raw(Box::new(p)),
        Err(e) => {
            eprintln!("[m56 ffi] pool_new construction error: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Destroy a pool created by `nsl_agent_pool_new`. No-op on null.
#[no_mangle]
pub extern "C" fn nsl_agent_pool_destroy(pool: *mut PipelineContextPool) {
    if pool.is_null() {
        return;
    }
    // SAFETY: pool was created via `Box::into_raw` and ownership is being
    // transferred back to drop.
    unsafe {
        drop(Box::from_raw(pool));
    }
}

/// Acquire a lease. Returns the lease's index (>=0), or:
/// - `-1` if the pool is exhausted.
/// - `-2` if `pool` is null.
#[no_mangle]
pub extern "C" fn nsl_agent_pool_acquire(pool: *mut PipelineContextPool, _timeout_ms: u64) -> i64 {
    if pool.is_null() {
        return -2;
    }
    // SAFETY: pool is a valid pointer from `nsl_agent_pool_new`.
    let pool = unsafe { &mut *pool };
    match pool.acquire() {
        Ok(lease) => lease.index() as i64,
        Err(AcquireError::Exhausted) => -1,
        Err(AcquireError::Timeout) => -1, // v1 acquires are non-blocking
    }
}

/// Release a lease by its index (returned from `acquire`). No-op on null
/// or negative index.
#[no_mangle]
pub extern "C" fn nsl_agent_pool_release(pool: *mut PipelineContextPool, lease_id: i64) {
    if pool.is_null() || lease_id < 0 {
        return;
    }
    // SAFETY: pool is a valid pointer.
    let pool = unsafe { &mut *pool };
    pool.release_by_index(lease_id as usize);
}

/// Run one logical-time step on the scheduler. Returns 0 on success.
/// Returns 1 if `sched` is null (sentinel for v1; codegen should never pass null).
#[no_mangle]
pub extern "C" fn nsl_agent_scheduler_step(sched: *mut ReactorScheduler) -> i32 {
    if sched.is_null() {
        return 1;
    }
    // SAFETY: sched is valid; the codegen owns the scheduler for the
    // pipeline function's duration.
    let sched = unsafe { &mut *sched };
    let _ = sched.step();
    0
}

/// Write a `PortMessage` (Box-owned by caller) to the mailbox at `time`.
/// Returns 0 on success, 1 on null mailbox, 2 on null msg.
///
/// Ownership: this function takes ownership of `msg` (consumes the Box).
/// The caller MUST NOT call `Box::from_raw(msg)` again after this returns.
#[no_mangle]
pub extern "C" fn nsl_agent_mailbox_write(
    mb: *mut PortMailbox,
    msg: *mut PortMessage,
    time: u64,
) -> i32 {
    if mb.is_null() {
        return 1;
    }
    if msg.is_null() {
        return 2;
    }
    // SAFETY: mb is a valid mailbox pointer; msg was produced by
    // `Box::into_raw` and ownership is now transferred here.
    let mb = unsafe { &mut *mb };
    let msg = unsafe { Box::from_raw(msg) };
    mb.write(*msg, time);
    0
}

/// Read a `PortMessage` from the mailbox. Returns null if the mailbox is
/// empty or the pointer is null. The caller takes ownership of the
/// returned `PortMessage` and is responsible for freeing it via
/// `Box::from_raw`.
#[no_mangle]
pub extern "C" fn nsl_agent_mailbox_read(mb: *mut PortMailbox) -> *mut PortMessage {
    if mb.is_null() {
        return std::ptr::null_mut();
    }
    // SAFETY: mb is a valid mailbox pointer.
    let mb = unsafe { &mut *mb };
    match mb.read() {
        Some(msg) => Box::into_raw(Box::new(msg)),
        None => std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::mailbox::StructPayload;

    #[test]
    fn ffi_pool_new_acquire_release_destroy() {
        let pool = nsl_agent_pool_new(2, 0);
        assert!(!pool.is_null());

        let l1 = nsl_agent_pool_acquire(pool, 1000);
        assert!(l1 >= 0);
        let l2 = nsl_agent_pool_acquire(pool, 1000);
        assert!(l2 >= 0);
        let l3 = nsl_agent_pool_acquire(pool, 1000);
        assert!(l3 < 0, "3rd acquire should fail (exhausted)");

        nsl_agent_pool_release(pool, l1);
        nsl_agent_pool_release(pool, l2);
        nsl_agent_pool_destroy(pool);
    }

    #[test]
    fn ffi_pool_new_oversize_returns_null() {
        // usize::MAX easily exceeds the v1 HARD_LIMIT.
        let pool = nsl_agent_pool_new(u64::MAX, 0);
        assert!(pool.is_null(), "oversize pool should return null pointer");
    }

    #[test]
    fn ffi_pool_acquire_null_pool_returns_minus_two() {
        let r = nsl_agent_pool_acquire(std::ptr::null_mut(), 0);
        assert_eq!(r, -2);
    }

    #[test]
    fn ffi_pool_destroy_null_no_panic() {
        nsl_agent_pool_destroy(std::ptr::null_mut());
        // Just must not panic.
    }

    #[test]
    fn ffi_scheduler_step_advances_time() {
        let mut sched = ReactorScheduler::new();
        sched.register_agent(|_| {});
        let sched_ptr: *mut ReactorScheduler = &mut sched;
        let r = nsl_agent_scheduler_step(sched_ptr);
        assert_eq!(r, 0);
        assert_eq!(sched.logical_time(), 1);
    }

    #[test]
    fn ffi_scheduler_step_null_returns_one() {
        let r = nsl_agent_scheduler_step(std::ptr::null_mut());
        assert_eq!(r, 1);
    }

    #[test]
    fn ffi_mailbox_write_read_round_trip() {
        let mut mb = PortMailbox::new();
        let mb_ptr: *mut PortMailbox = &mut mb;

        let msg = Box::new(PortMessage::Struct(Box::new(StructPayload::new(vec![
            7, 8, 9,
        ]))));
        let msg_ptr = Box::into_raw(msg);
        let r = nsl_agent_mailbox_write(mb_ptr, msg_ptr, 42);
        assert_eq!(r, 0);

        let read_ptr = nsl_agent_mailbox_read(mb_ptr);
        assert!(!read_ptr.is_null());
        // SAFETY: read_ptr ownership transferred to us.
        let read_msg = unsafe { Box::from_raw(read_ptr) };
        match *read_msg {
            PortMessage::Struct(payload) => {
                assert_eq!(payload.as_bytes(), &[7, 8, 9]);
            }
            _ => panic!("expected Struct variant"),
        }

        // Mailbox is now empty.
        let empty = nsl_agent_mailbox_read(mb_ptr);
        assert!(empty.is_null(), "second read should return null");
    }

    #[test]
    fn ffi_mailbox_read_empty_returns_null() {
        let mut mb = PortMailbox::new();
        let r = nsl_agent_mailbox_read(&mut mb);
        assert!(r.is_null());
    }

    #[test]
    fn ffi_mailbox_write_null_mailbox_returns_one() {
        let msg = Box::into_raw(Box::new(PortMessage::Struct(Box::new(StructPayload::new(
            vec![],
        )))));
        let r = nsl_agent_mailbox_write(std::ptr::null_mut(), msg, 0);
        assert_eq!(r, 1);
        // Free the leaked msg to avoid the leak; FFI didn't take ownership
        // because mb was null.
        unsafe {
            drop(Box::from_raw(msg));
        }
    }

    #[test]
    fn ffi_mailbox_write_null_msg_returns_two() {
        let mut mb = PortMailbox::new();
        let r = nsl_agent_mailbox_write(&mut mb, std::ptr::null_mut(), 0);
        assert_eq!(r, 2);
    }
}
