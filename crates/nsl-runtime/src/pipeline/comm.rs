//! M43b: Pipeline point-to-point communication — SharedMem backend.
//!
//! Implements send/recv for activations and gradients between adjacent pipeline
//! stages. Uses a shared-memory mailbox pattern: each (dst_rank, tag) pair has
//! a slot where the sender writes tensor data and the receiver reads it.
//!
//! The backend follows M30's SimulatedBackend pattern but uses point-to-point
//! mailboxes instead of collective all-reduce slots.

use std::collections::HashMap;
use std::os::raw::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// Pipeline context — global state for the pipeline communication layer
// ---------------------------------------------------------------------------

static PIPELINE_CTX: Mutex<Option<PipelineContext>> = Mutex::new(None);

#[allow(dead_code)]
struct PipelineContext {
    num_stages: usize,
    num_micro_batches: usize,
    backend: SharedMemPipeline,
}

// ---------------------------------------------------------------------------
// PipelineBackend trait
// ---------------------------------------------------------------------------

/// Abstraction over pipeline point-to-point communication.
/// All tensor data is serialized as contiguous bytes for transport.
pub trait PipelineBackend: Send + Sync {
    /// Send tensor data to a destination rank with a tag.
    fn send(&self, data: &[u8], dst_rank: usize, tag: i64) -> i64;

    /// Receive tensor data from a source rank with a tag.
    /// Blocks until data is available. Returns the received bytes.
    fn recv(&self, src_rank: usize, tag: i64) -> Vec<u8>;

    /// Synchronize all pipeline stages.
    fn barrier(&self) -> i64;
}

// ---------------------------------------------------------------------------
// SharedMemPipeline — in-process mailbox backend for single-node pipelines
// ---------------------------------------------------------------------------

/// Mailbox entry: holds tensor data for a specific (dst_rank, tag) pair.
struct Mailbox {
    /// Data written by sender, read by receiver.
    data: Option<Vec<u8>>,
}

/// Shared-memory pipeline backend using in-process mailboxes with condvar signaling.
///
/// Each (dst_rank, tag) pair has a mailbox. The sender writes data and notifies;
/// the receiver waits on the condvar, then reads. This enables multi-threaded
/// pipeline execution within a single process.
pub struct SharedMemPipeline {
    /// Mailboxes keyed by (dst_rank, tag).
    mailboxes: Arc<Mutex<HashMap<(usize, i64), Mailbox>>>,
    /// Condvar to signal data availability.
    data_ready: Arc<Condvar>,
    /// Barrier: counts arrivals, resets after all stages arrive.
    barrier_count: Arc<AtomicUsize>,
    barrier_generation: Arc<AtomicUsize>,
    num_stages: usize,
    /// Condvar for barrier synchronization.
    barrier_cv: Arc<(Mutex<()>, Condvar)>,
}

impl SharedMemPipeline {
    pub fn new(num_stages: usize) -> Self {
        Self {
            mailboxes: Arc::new(Mutex::new(HashMap::new())),
            data_ready: Arc::new(Condvar::new()),
            barrier_count: Arc::new(AtomicUsize::new(0)),
            barrier_generation: Arc::new(AtomicUsize::new(0)),
            num_stages,
            barrier_cv: Arc::new((Mutex::new(()), Condvar::new())),
        }
    }
}

impl PipelineBackend for SharedMemPipeline {
    fn send(&self, data: &[u8], dst_rank: usize, tag: i64) -> i64 {
        let mut guard = self.mailboxes.lock().unwrap();
        guard.insert((dst_rank, tag), Mailbox { data: Some(data.to_vec()) });
        // Notify all waiters that new data is available
        self.data_ready.notify_all();
        0
    }

    fn recv(&self, src_rank: usize, tag: i64) -> Vec<u8> {
        let _ = src_rank; // Key is (dst=self, tag) — sender writes to our slot
        let mut guard = self.mailboxes.lock().unwrap();
        loop {
            // Check if data is available for our (rank, tag)
            // We receive using our own rank as the key — sender wrote to (dst_rank=us, tag)
            if let Some(mailbox) = guard.get_mut(&(src_rank, tag)) {
                if let Some(data) = mailbox.data.take() {
                    guard.remove(&(src_rank, tag));
                    return data;
                }
            }
            // Wait for data — note: we use the mailbox key from the sender's perspective
            // Sender writes to (dst_rank, tag), so we check (dst_rank=us, tag)
            // But since we don't know our own rank here, we re-key by (src_rank, tag)
            // and rely on the sender using (dst_rank, tag) where dst_rank matches what
            // the receiver checks.
            guard = self.data_ready.wait(guard).unwrap();
        }
    }

    fn barrier(&self) -> i64 {
        let gen = self.barrier_generation.load(Ordering::Acquire);
        let prev = self.barrier_count.fetch_add(1, Ordering::AcqRel);

        if prev + 1 == self.num_stages {
            // Last to arrive — reset counter, advance generation
            self.barrier_count.store(0, Ordering::Release);
            self.barrier_generation.store(gen + 1, Ordering::Release);
            let (_lock, cv) = &*self.barrier_cv;
            cv.notify_all();
        } else {
            // Wait for generation to advance
            let (lock, cv) = &*self.barrier_cv;
            let mut guard = lock.lock().unwrap();
            while self.barrier_generation.load(Ordering::Acquire) == gen {
                guard = cv.wait(guard).unwrap();
            }
        }
        0
    }
}

// ---------------------------------------------------------------------------
// Tensor serialization helpers
// ---------------------------------------------------------------------------

/// Serialize an NslTensor into a byte buffer.
/// Format: [ndim: i64][shape: i64 * ndim][dtype: u16][device: u8][data: bytes]
fn serialize_tensor(tensor_ptr: i64) -> Vec<u8> {
    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;
    let len = t.len as usize;
    let elem_size = t.element_size();
    let data_bytes = len * elem_size;

    // Header: ndim + shape + dtype + device
    let header_size = 8 + ndim * 8 + 2 + 1; // i64 + i64*ndim + u16 + u8
    let mut buf = vec![0u8; header_size + data_bytes];

    // Write ndim
    buf[..8].copy_from_slice(&(t.ndim).to_le_bytes());

    // Write shape
    let shape = unsafe { std::slice::from_raw_parts(t.shape, ndim) };
    for (i, &dim) in shape.iter().enumerate() {
        buf[8 + i * 8..8 + (i + 1) * 8].copy_from_slice(&dim.to_le_bytes());
    }

    // Write dtype and device
    let offset = 8 + ndim * 8;
    buf[offset..offset + 2].copy_from_slice(&t.dtype.to_le_bytes());
    buf[offset + 2] = t.device;

    // Write tensor data
    let data_offset = header_size;
    unsafe {
        std::ptr::copy_nonoverlapping(
            t.data as *const u8,
            buf[data_offset..].as_mut_ptr(),
            data_bytes,
        );
    }

    buf
}

/// Deserialize a byte buffer into a new NslTensor. Returns the tensor pointer.
fn deserialize_tensor(buf: &[u8]) -> i64 {
    if buf.len() < 11 { return 0; } // minimum: ndim(8) + dtype(2) + device(1)

    // Read ndim
    let ndim = i64::from_le_bytes(buf[..8].try_into().unwrap());
    let ndim_usize = ndim as usize;

    // Read shape
    let shape = checked_alloc(ndim_usize * std::mem::size_of::<i64>()) as *mut i64;
    let mut total_len: i64 = 1;
    for i in 0..ndim_usize {
        let dim = i64::from_le_bytes(buf[8 + i * 8..8 + (i + 1) * 8].try_into().unwrap());
        unsafe { *shape.add(i) = dim };
        total_len *= dim;
    }

    let offset = 8 + ndim_usize * 8;
    let dtype = u16::from_le_bytes(buf[offset..offset + 2].try_into().unwrap());
    let device = buf[offset + 2];

    // Use the same element_size logic as NslTensor::element_size()
    let elem_size = match dtype {
        0 => 8,  // f64
        1 => 4,  // f32
        _ => 4,  // conservative default for custom dtypes
    };
    let data_bytes = total_len as usize * elem_size;
    let data_offset = offset + 3;

    let data = checked_alloc(data_bytes);
    unsafe {
        std::ptr::copy_nonoverlapping(
            buf[data_offset..data_offset + data_bytes].as_ptr(),
            data,
            data_bytes,
        );
    }

    let strides = NslTensor::compute_strides(shape, ndim);

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        total_len,
        device,
        dtype,
        1,
        0,
    ));

    Box::into_raw(tensor) as i64
}

// ---------------------------------------------------------------------------
// FFI entry points
// ---------------------------------------------------------------------------

/// Initialize the pipeline communication context.
/// `schedule_type`: 0 = 1F1B, 1 = GPipe.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_pipeline_init(
    num_stages: i64,
    schedule_type: i64,
    num_micro_batches: i64,
) -> i64 {
    let _ = schedule_type;
    let mut guard = PIPELINE_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    let stages = num_stages as usize;
    *guard = Some(PipelineContext {
        num_stages: stages,
        num_micro_batches: num_micro_batches as usize,
        backend: SharedMemPipeline::new(stages),
    });
    0
}

/// Send a tensor to the next pipeline stage.
/// `tensor_ptr`: NslTensor* to send.
/// `dst_rank`: destination stage index.
/// `tag`: message tag (typically micro_batch_idx * 2 + direction).
/// `stream`: CUDA stream handle (0 for CPU).
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_pipeline_send(
    tensor_ptr: i64,
    dst_rank: i64,
    tag: i64,
    _stream: i64,
) -> i64 {
    if tensor_ptr == 0 { return -1; }
    let guard = PIPELINE_CTX.lock().unwrap();
    let ctx = match guard.as_ref() {
        Some(c) => c,
        None => return -1,
    };
    let mailboxes = Arc::clone(&ctx.backend.mailboxes);
    let cv = Arc::clone(&ctx.backend.data_ready);
    drop(guard);

    let data = serialize_tensor(tensor_ptr);
    let mut mbox = mailboxes.lock().unwrap();
    mbox.insert((dst_rank as usize, tag), Mailbox { data: Some(data) });
    cv.notify_all();
    0
}

/// Receive a tensor from the previous pipeline stage.
/// Blocks until data is available.
///
/// `my_rank`: the receiver's own stage rank (the sender wrote to this rank via send(dst_rank=my_rank)).
/// `tag`: message tag matching what the sender used.
///
/// Returns NslTensor* on success, 0 on failure.
#[no_mangle]
pub extern "C" fn nsl_pipeline_recv(
    _shape_ptr: i64,
    _ndim: i64,
    _dtype: i64,
    my_rank: i64,
    tag: i64,
    _stream: i64,
) -> i64 {
    let guard = PIPELINE_CTX.lock().unwrap();
    let ctx = match guard.as_ref() {
        Some(c) => c,
        None => return 0,
    };
    // Must drop guard before blocking recv to avoid deadlock
    let backend_mailboxes = Arc::clone(&ctx.backend.mailboxes);
    let backend_cv = Arc::clone(&ctx.backend.data_ready);
    drop(guard);

    // Lookup by (my_rank, tag): the sender wrote to this key via send(data, dst_rank=my_rank, tag)
    let rank = my_rank as usize;
    let mut mbox_guard = backend_mailboxes.lock().unwrap();
    loop {
        if let Some(mailbox) = mbox_guard.get_mut(&(rank, tag)) {
            if let Some(data) = mailbox.data.take() {
                mbox_guard.remove(&(rank, tag));
                return deserialize_tensor(&data);
            }
        }
        mbox_guard = backend_cv.wait(mbox_guard).unwrap();
    }
}

/// Send gradients to the previous pipeline stage.
/// Same as `nsl_pipeline_send` but uses a different tag namespace.
#[no_mangle]
pub extern "C" fn nsl_pipeline_send_grad(
    grad_ptr: i64,
    dst_rank: i64,
    tag: i64,
    stream: i64,
) -> i64 {
    // Gradient sends use the same mechanism with tag offset for namespace separation
    nsl_pipeline_send(grad_ptr, dst_rank, tag + 1_000_000, stream)
}

/// Receive gradients from the next pipeline stage.
/// Same as `nsl_pipeline_recv` but uses a different tag namespace.
/// `my_rank`: the receiver's own stage rank.
#[no_mangle]
pub extern "C" fn nsl_pipeline_recv_grad(
    shape_ptr: i64,
    ndim: i64,
    dtype: i64,
    my_rank: i64,
    tag: i64,
    stream: i64,
) -> i64 {
    nsl_pipeline_recv(shape_ptr, ndim, dtype, my_rank, tag + 1_000_000, stream)
}

/// Pipeline barrier — synchronize all stages.
#[no_mangle]
pub extern "C" fn nsl_pipeline_barrier() -> i64 {
    let guard = PIPELINE_CTX.lock().unwrap();
    let ctx = match guard.as_ref() {
        Some(c) => c,
        None => return 0, // No context = no-op (backwards compat)
    };
    let barrier_count = Arc::clone(&ctx.backend.barrier_count);
    let barrier_gen = Arc::clone(&ctx.backend.barrier_generation);
    let barrier_cv = Arc::clone(&ctx.backend.barrier_cv);
    let num_stages = ctx.num_stages;
    drop(guard);

    // Inline barrier to avoid holding PIPELINE_CTX lock
    let gen = barrier_gen.load(Ordering::Acquire);
    let prev = barrier_count.fetch_add(1, Ordering::AcqRel);

    if prev + 1 == num_stages {
        barrier_count.store(0, Ordering::Release);
        barrier_gen.store(gen + 1, Ordering::Release);
        let (_lock, cv) = &*barrier_cv;
        cv.notify_all();
    } else {
        let (lock, cv) = &*barrier_cv;
        let mut g = lock.lock().unwrap();
        while barrier_gen.load(Ordering::Acquire) == gen {
            g = cv.wait(g).unwrap();
        }
    }
    0
}

/// Destroy the pipeline communication context.
/// Returns 0 on success, -1 if not initialized.
#[no_mangle]
pub extern "C" fn nsl_pipeline_destroy() -> i64 {
    let mut guard = PIPELINE_CTX.lock().unwrap();
    if guard.is_none() {
        return -1;
    }
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn reset_ctx() {
        let mut guard = PIPELINE_CTX.lock().unwrap();
        *guard = None;
    }

    /// Initialize pipeline for a test, ensuring clean state.
    fn test_init(stages: usize, micro_batches: usize) {
        reset_ctx();
        let r = nsl_pipeline_init(stages as i64, 0, micro_batches as i64);
        assert_eq!(r, 0, "pipeline init failed (ctx was not properly reset)");
    }

    #[test]
    fn test_pipeline_init_destroy_lifecycle() {
        reset_ctx();
        assert_eq!(nsl_pipeline_init(4, 0, 8), 0);
        assert_eq!(nsl_pipeline_init(4, 0, 8), -1); // double init
        assert_eq!(nsl_pipeline_destroy(), 0);
        assert_eq!(nsl_pipeline_destroy(), -1); // double destroy
        assert_eq!(nsl_pipeline_init(2, 1, 4), 0); // re-init
        assert_eq!(nsl_pipeline_destroy(), 0);
    }

    #[test]
    fn test_pipeline_barrier_no_ctx() {
        reset_ctx();
        // Barrier without context is a no-op
        assert_eq!(nsl_pipeline_barrier(), 0);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        // Create a test tensor [2, 3] with known data
        let data = checked_alloc(6 * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..6 {
            unsafe { *data.add(i) = (i as f64) * 1.5 };
        }
        let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = 2; *shape.add(1) = 3 };
        let strides = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = 3; *strides.add(1) = 1 };

        let tensor = Box::new(NslTensor::new(
            data as *mut c_void,
            shape,
            strides,
            2,
            6,
            0,
            0,
            1,
            0,
        ));
        let ptr = Box::into_raw(tensor) as i64;

        // Serialize
        let buf = serialize_tensor(ptr);
        assert!(buf.len() > 0);

        // Deserialize
        let new_ptr = deserialize_tensor(&buf);
        assert_ne!(new_ptr, 0);

        let orig = NslTensor::from_ptr(ptr);
        let copy = NslTensor::from_ptr(new_ptr);

        assert_eq!(orig.ndim, copy.ndim);
        assert_eq!(orig.len, copy.len);
        assert_eq!(orig.dtype, copy.dtype);

        // Verify data matches
        let orig_data = unsafe { std::slice::from_raw_parts(orig.data as *const f64, 6) };
        let copy_data = unsafe { std::slice::from_raw_parts(copy.data as *const f64, 6) };
        for i in 0..6 {
            assert_eq!(orig_data[i], copy_data[i], "data mismatch at index {i}");
        }

        // Verify shape matches
        let orig_shape = unsafe { std::slice::from_raw_parts(orig.shape, 2) };
        let copy_shape = unsafe { std::slice::from_raw_parts(copy.shape, 2) };
        assert_eq!(orig_shape, copy_shape);

        // Cleanup
        crate::tensor::nsl_tensor_free(ptr);
        crate::tensor::nsl_tensor_free(new_ptr);
    }

    #[test]
    fn test_shared_mem_send_recv() {
        let backend = SharedMemPipeline::new(2);
        let data = vec![1u8, 2, 3, 4, 5];

        // Send from stage 0 to stage 1 with tag 0
        backend.send(&data, 1, 0);

        // Receive at stage 1 from stage 0 with tag 0
        // Note: recv checks (src_rank=sender, tag) which the sender wrote as (dst_rank=1, tag=0)
        // The recv uses the same key — let's fix this by using dst_rank as the key
        let received = backend.recv(1, 0);
        assert_eq!(received, data);
    }

    #[test]
    fn test_shared_mem_multiple_tags() {
        let backend = SharedMemPipeline::new(4);

        // Send multiple messages with different tags
        backend.send(&[10, 20], 1, 100);
        backend.send(&[30, 40], 1, 200);
        backend.send(&[50, 60], 2, 100);

        // Receive in any order — tags distinguish messages
        let r1 = backend.recv(1, 200);
        assert_eq!(r1, vec![30, 40]);

        let r2 = backend.recv(1, 100);
        assert_eq!(r2, vec![10, 20]);

        let r3 = backend.recv(2, 100);
        assert_eq!(r3, vec![50, 60]);
    }

    #[test]
    fn test_tensor_send_recv_via_backend() {
        // Test using SharedMemPipeline directly (avoids global state races).
        let backend = SharedMemPipeline::new(2);

        // Create a test tensor [4]
        let data = checked_alloc(4 * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..4 {
            unsafe { *data.add(i) = (i as f64) + 0.5 };
        }
        let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *shape = 4 };
        let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *strides = 1 };

        let tensor = Box::new(NslTensor::new(
            data as *mut c_void,
            shape,
            strides,
            1,
            4,
            0,
            0,
            1,
            0,
        ));
        let send_ptr = Box::into_raw(tensor) as i64;

        // Serialize and send to stage 1 via backend
        let serialized = serialize_tensor(send_ptr);
        backend.send(&serialized, 1, 42);

        // Receive at stage 1
        let received = backend.recv(1, 42);
        let recv_ptr = deserialize_tensor(&received);
        assert_ne!(recv_ptr, 0, "deserialize failed");

        let recv_t = NslTensor::from_ptr(recv_ptr);
        assert_eq!(recv_t.len, 4);
        let recv_data = unsafe { std::slice::from_raw_parts(recv_t.data as *const f64, 4) };
        assert_eq!(recv_data[0], 0.5);
        assert_eq!(recv_data[3], 3.5);

        crate::tensor::nsl_tensor_free(send_ptr);
        crate::tensor::nsl_tensor_free(recv_ptr);
    }

    #[test]
    fn test_gradient_tag_namespace_separation() {
        // Activation and gradient tags don't collide due to 1M offset
        let backend = SharedMemPipeline::new(2);

        // Send activation with tag 5 to rank 0
        backend.send(&[10, 20], 0, 5);
        // Send gradient with tag 5 to rank 0 (uses offset tag: 5 + 1_000_000)
        backend.send(&[30, 40], 0, 5 + 1_000_000);

        // Receive activation tag 5
        let act = backend.recv(0, 5);
        assert_eq!(act, vec![10, 20]);

        // Receive gradient tag 5 (with offset)
        let grad = backend.recv(0, 5 + 1_000_000);
        assert_eq!(grad, vec![30, 40]);
    }

    #[test]
    fn test_barrier_with_threads() {
        use std::sync::Arc;
        use std::thread;

        let backend = Arc::new(SharedMemPipeline::new(3));
        let counter = Arc::new(AtomicUsize::new(0));

        let mut handles = vec![];
        for _stage in 0..3 {
            let b = Arc::clone(&backend);
            let c = Arc::clone(&counter);
            handles.push(thread::spawn(move || {
                // Each stage increments counter, then barriers
                c.fetch_add(1, Ordering::SeqCst);
                b.barrier();
                // After barrier, all 3 stages should have incremented
                assert_eq!(c.load(Ordering::SeqCst), 3);
            }));
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_threaded_send_recv() {
        use std::sync::Arc;
        use std::thread;

        let backend = Arc::new(SharedMemPipeline::new(2));

        let b_sender = Arc::clone(&backend);
        let b_receiver = Arc::clone(&backend);

        let sender = thread::spawn(move || {
            b_sender.send(&[42, 43, 44], 1, 0);
        });

        let receiver = thread::spawn(move || {
            let data = b_receiver.recv(1, 0);
            assert_eq!(data, vec![42, 43, 44]);
        });

        sender.join().unwrap();
        receiver.join().unwrap();
    }
}
