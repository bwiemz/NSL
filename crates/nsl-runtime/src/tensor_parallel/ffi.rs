//! FFI exports for tensor parallelism.
//!
//! Provides a global `TpContext` behind a `Mutex<Option<TpContext>>` that
//! Cranelift-generated code calls into via `extern "C"` functions.
//! All parameters are `i64` to match the Cranelift calling convention.

use std::ffi::c_void;
use std::sync::Mutex;

use super::collective::{CollectiveBackend, DtypeId, SimulatedBackend, StreamHandle};

// ---------------------------------------------------------------------------
// Global context
// ---------------------------------------------------------------------------

static TP_CTX: Mutex<Option<TpContext>> = Mutex::new(None);

struct TpContext {
    rank: i32,
    world_size: i32,
    backend: Box<dyn CollectiveBackend>,
}

// SAFETY: CollectiveBackend implementors (SimulatedBackend) are Send+Sync.
unsafe impl Send for TpContext {}

// ---------------------------------------------------------------------------
// Shared-memory helper
// ---------------------------------------------------------------------------

/// Open a file-backed shared memory region via `memmap2` and leak the mapping
/// so it lives for the process lifetime.
fn open_shm(path: &str) -> (*mut u8, usize) {
    use std::fs::OpenOptions;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .expect("failed to open shared memory file");
    let mmap = unsafe {
        memmap2::MmapMut::map_mut(&file).expect("failed to mmap")
    };
    let ptr = mmap.as_ptr() as *mut u8;
    let len = mmap.len();
    std::mem::forget(mmap); // keep mapped for process lifetime
    (ptr, len)
}

// ---------------------------------------------------------------------------
// FFI functions
// ---------------------------------------------------------------------------

/// Initialise the tensor-parallelism context.
///
/// Reads configuration from environment variables:
/// - `NSL_LOCAL_RANK`   — this process's rank (default: 0)
/// - `NSL_WORLD_SIZE`   — total number of ranks (default: 1)
/// - `NSL_SIMULATED_TP` — if "1", use `SimulatedBackend` (default: 1)
/// - `NSL_TP_SHM_PATH`  — path to the shared-memory file (required when world_size > 1)
///
/// Returns 0 on success, -1 if already initialised.
#[no_mangle]
pub extern "C" fn nsl_tp_init() -> i64 {
    let rank: i32 = std::env::var("NSL_LOCAL_RANK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    let world_size: i32 = std::env::var("NSL_WORLD_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);

    let _simulated: bool = std::env::var("NSL_SIMULATED_TP")
        .ok()
        .map(|v| v == "1")
        .unwrap_or(true);

    // For world_size > 1 with simulated backend, open the shared memory file.
    let (shm_ptr, shm_len) = if world_size > 1 {
        let shm_path = std::env::var("NSL_TP_SHM_PATH")
            .expect("NSL_TP_SHM_PATH must be set when NSL_WORLD_SIZE > 1");
        open_shm(&shm_path)
    } else {
        (std::ptr::null_mut(), 0)
    };

    let backend = Box::new(SimulatedBackend::new(rank, world_size, shm_ptr, shm_len));

    let mut guard = TP_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(TpContext {
        rank,
        world_size,
        backend,
    });
    0
}

/// Returns this process's tensor-parallel rank.
#[no_mangle]
pub extern "C" fn nsl_tp_rank() -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_init not called");
    ctx.rank as i64
}

/// Returns the tensor-parallel world size.
#[no_mangle]
pub extern "C" fn nsl_tp_world_size() -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_init not called");
    ctx.world_size as i64
}

/// Element-wise sum across all ranks.
///
/// All parameters are `i64` (Cranelift convention):
/// - `sendbuf` / `recvbuf` — pointers to data buffers
/// - `count` — number of elements
/// - `dtype` — `DtypeId` constant (0=f64, 1=f32, …)
/// - `stream` — CUDA stream handle (0/null for CPU)
///
/// Returns 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn nsl_tp_all_reduce_sum(
    sendbuf: i64,
    recvbuf: i64,
    count: i64,
    dtype: i64,
    stream: i64,
) -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_init not called");
    let rc = ctx.backend.all_reduce_sum(
        sendbuf as *const c_void,
        recvbuf as *mut c_void,
        count as usize,
        dtype as DtypeId,
        stream as StreamHandle,
    );
    rc as i64
}

/// Gather `send_count` elements from each rank into `recvbuf`.
///
/// Total elements in recvbuf = send_count * world_size.
/// Returns 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn nsl_tp_all_gather(
    sendbuf: i64,
    recvbuf: i64,
    send_count: i64,
    dtype: i64,
    stream: i64,
) -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_init not called");
    let rc = ctx.backend.all_gather(
        sendbuf as *const c_void,
        recvbuf as *mut c_void,
        send_count as usize,
        dtype as DtypeId,
        stream as StreamHandle,
    );
    rc as i64
}

/// Broadcast `count` elements from `root_rank` to all ranks.
///
/// Returns 0 on success, negative on error.
#[no_mangle]
pub extern "C" fn nsl_tp_broadcast(
    buf: i64,
    count: i64,
    dtype: i64,
    root_rank: i64,
    stream: i64,
) -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_init not called");
    let rc = ctx.backend.broadcast(
        buf as *mut c_void,
        count as usize,
        dtype as DtypeId,
        root_rank as i32,
        stream as StreamHandle,
    );
    rc as i64
}

/// Block until all ranks have reached this point.
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_tp_barrier() -> i64 {
    let guard = TP_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_tp_init not called");
    ctx.backend.barrier() as i64
}

/// Tear down the tensor-parallelism context and release resources.
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_tp_destroy() -> i64 {
    let mut guard = TP_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialise all FFI tests — they share the global `TP_CTX` and env vars,
    /// so concurrent execution causes data races.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Helper: acquire the serialisation lock, destroy any stale context, and
    /// clear env vars so each test starts from a known state.
    /// Returns the `MutexGuard` — hold it for the duration of the test.
    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_tp_destroy();
        std::env::remove_var("NSL_LOCAL_RANK");
        std::env::remove_var("NSL_WORLD_SIZE");
        std::env::remove_var("NSL_SIMULATED_TP");
        std::env::remove_var("NSL_TP_SHM_PATH");
        guard
    }

    #[test]
    fn ffi_init_destroy_lifecycle() {
        let _lock = setup();

        assert_eq!(nsl_tp_init(), 0);
        assert_eq!(nsl_tp_rank(), 0);
        assert_eq!(nsl_tp_world_size(), 1);

        // all_reduce on single rank = identity
        let mut data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let ptr = data.as_mut_ptr() as i64;
        assert_eq!(nsl_tp_all_reduce_sum(ptr, ptr, 3, 1, 0), 0); // dtype 1 = F32
        assert_eq!(data, vec![1.0, 2.0, 3.0]);

        assert_eq!(nsl_tp_barrier(), 0);
        assert_eq!(nsl_tp_destroy(), 0);
    }

    #[test]
    fn ffi_double_init_returns_error() {
        let _lock = setup();

        assert_eq!(nsl_tp_init(), 0);
        assert_eq!(nsl_tp_init(), -1); // already initialised
        assert_eq!(nsl_tp_destroy(), 0);
    }

    #[test]
    fn ffi_all_gather_single_rank() {
        let _lock = setup();

        assert_eq!(nsl_tp_init(), 0);

        let src: Vec<f32> = vec![10.0, 20.0, 30.0];
        let mut dst: Vec<f32> = vec![0.0; 3];
        let rc = nsl_tp_all_gather(
            src.as_ptr() as i64,
            dst.as_mut_ptr() as i64,
            3,
            1, // F32
            0,
        );
        assert_eq!(rc, 0);
        assert_eq!(dst, vec![10.0, 20.0, 30.0]);

        assert_eq!(nsl_tp_destroy(), 0);
    }

    #[test]
    fn ffi_broadcast_single_rank() {
        let _lock = setup();

        assert_eq!(nsl_tp_init(), 0);

        let mut data: Vec<f64> = vec![42.0, 99.0];
        let rc = nsl_tp_broadcast(
            data.as_mut_ptr() as i64,
            2,
            0, // F64
            0, // root rank
            0,
        );
        assert_eq!(rc, 0);
        assert_eq!(data, vec![42.0, 99.0]);

        assert_eq!(nsl_tp_destroy(), 0);
    }

    #[test]
    fn ffi_destroy_without_init() {
        let _lock = setup();
        // Destroying when no context exists should still return 0.
        assert_eq!(nsl_tp_destroy(), 0);
    }
}
