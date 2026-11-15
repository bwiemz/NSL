//! N-stage prefetch pipeline for overlapping I/O with GPU compute.
//!
//! ```text
//! Stage 1 (I/O thread):   read shard → decode → tokenize → batch
//! Stage 2 (CPU thread):   collate + pad → pin memory
//! Stage 3 (GPU stream):   cudaMemcpyAsync H2D (or GDS direct)
//! Stage 4 (compute):      training step on GPU
//!
//! All 4 stages run concurrently on different data:
//! While GPU trains on batch[i], CPU prepares batch[i+1],
//! I/O reads batch[i+2], GDS prefetches batch[i+3]
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// ---------------------------------------------------------------------------
// Batch — a loaded and prepared mini-batch
// ---------------------------------------------------------------------------

/// A prepared batch of training data.
#[derive(Debug)]
pub struct Batch {
    /// Raw sample data (concatenated, variable length per sample).
    pub data: Vec<u8>,
    /// Offsets into `data` for each sample in the batch.
    pub sample_offsets: Vec<usize>,
    /// Number of samples in this batch.
    pub batch_size: usize,
    /// Sequence number for ordering.
    pub seq_num: u64,
}

// ---------------------------------------------------------------------------
// Bounded channel (backpressure queue)
// ---------------------------------------------------------------------------

/// A bounded multi-producer, multi-consumer channel with backpressure.
/// Producers block when the queue is full; consumers block when empty.
///
/// `closed` lives under the queue's mutex. It used to be an atomic that
/// `close` set without taking the lock, so a waiter could read it (false),
/// `close` could then set it and notify, and only then the waiter would
/// enter `wait`: the notification was lost and the waiter slept forever.
/// That is how `PrefetchPipeline::shutdown` hung, its I/O worker parked in
/// `push` on a full queue and the caller in `join`. Under the lock, `close`
/// cannot run between a waiter's check and its wait.
struct BoundedChannel<T> {
    state: Mutex<ChannelState<T>>,
    capacity: usize,
    not_full: Condvar,
    not_empty: Condvar,
}

struct ChannelState<T> {
    queue: VecDeque<T>,
    closed: bool,
}

impl<T> BoundedChannel<T> {
    fn new(capacity: usize) -> Self {
        Self {
            state: Mutex::new(ChannelState {
                queue: VecDeque::with_capacity(capacity),
                closed: false,
            }),
            capacity,
            not_full: Condvar::new(),
            not_empty: Condvar::new(),
        }
    }

    /// Push an item, blocking if the queue is full. Returns false (and
    /// drops the item) once the channel is closed.
    fn push(&self, item: T) -> bool {
        let mut guard = self.state.lock().unwrap();
        loop {
            if guard.closed {
                return false;
            }
            if guard.queue.len() < self.capacity {
                break;
            }
            guard = self.not_full.wait(guard).unwrap();
        }
        guard.queue.push_back(item);
        self.not_empty.notify_one();
        true
    }

    /// Pop an item, blocking if the queue is empty.
    /// Returns None if the channel is closed and empty.
    fn pop(&self) -> Option<T> {
        let mut guard = self.state.lock().unwrap();
        loop {
            if let Some(item) = guard.queue.pop_front() {
                self.not_full.notify_one();
                return Some(item);
            }
            if guard.closed {
                return None;
            }
            guard = self.not_empty.wait(guard).unwrap();
        }
    }

    /// Close the channel. Wakes all waiting threads.
    fn close(&self) {
        let mut guard = self.state.lock().unwrap();
        guard.closed = true;
        drop(guard);
        self.not_full.notify_all();
        self.not_empty.notify_all();
    }

    /// Number of items currently in the queue.
    fn len(&self) -> usize {
        self.state.lock().unwrap().queue.len()
    }
}

// ---------------------------------------------------------------------------
// PrefetchPipeline — multi-stage data loading
// ---------------------------------------------------------------------------

/// Configuration for the prefetch pipeline.
#[derive(Debug, Clone)]
pub struct PrefetchConfig {
    /// Number of batches to buffer ahead of GPU consumption.
    pub prefetch_depth: usize,
    /// Batch size (samples per batch).
    pub batch_size: usize,
    /// Number of I/O worker threads.
    pub num_io_workers: usize,
}

impl Default for PrefetchConfig {
    fn default() -> Self {
        PrefetchConfig {
            prefetch_depth: 2,
            batch_size: 32,
            num_io_workers: 1,
        }
    }
}

/// Multi-stage prefetch pipeline that overlaps I/O with GPU compute.
///
/// The pipeline runs I/O workers on background threads that fill a bounded
/// queue. The main thread (GPU compute) pops batches from the queue.
/// Backpressure ensures memory usage stays bounded.
pub struct PrefetchPipeline {
    /// Bounded queue connecting I/O threads to GPU consumer.
    output_queue: Arc<BoundedChannel<Batch>>,
    /// Handle to the I/O worker thread(s).
    io_handles: Vec<thread::JoinHandle<()>>,
    /// Flag to signal shutdown.
    shutdown: Arc<AtomicBool>,
    /// Next expected batch sequence number.
    next_seq: u64,
}

impl PrefetchPipeline {
    /// Create a new prefetch pipeline with the given data source.
    ///
    /// `data_fn`: A function that produces raw sample bytes given a sample index.
    ///            Called by I/O worker threads.
    pub fn new<F>(config: PrefetchConfig, data_fn: F) -> Self
    where
        F: Fn(u64) -> Vec<u8> + Send + Sync + 'static,
    {
        let output_queue = Arc::new(BoundedChannel::new(config.prefetch_depth));
        let shutdown = Arc::new(AtomicBool::new(false));
        let data_fn = Arc::new(data_fn);

        let mut io_handles = Vec::new();

        for worker_id in 0..config.num_io_workers {
            let queue = Arc::clone(&output_queue);
            let stop = Arc::clone(&shutdown);
            let fetch = Arc::clone(&data_fn);
            let batch_size = config.batch_size;

            let handle = thread::spawn(move || {
                let mut seq = worker_id as u64;
                let stride = config.num_io_workers as u64;

                while !stop.load(Ordering::Acquire) {
                    // Build one batch
                    let mut batch_data = Vec::new();
                    let mut offsets = Vec::new();

                    for sample_idx in 0..batch_size {
                        let global_idx = seq * batch_size as u64 + sample_idx as u64;
                        let sample = fetch(global_idx);
                        offsets.push(batch_data.len());
                        batch_data.extend_from_slice(&sample);
                    }

                    let batch = Batch {
                        data: batch_data,
                        sample_offsets: offsets,
                        batch_size,
                        seq_num: seq,
                    };

                    if !queue.push(batch) {
                        break; // channel closed
                    }
                    seq += stride;
                }
            });

            io_handles.push(handle);
        }

        PrefetchPipeline {
            output_queue,
            io_handles,
            shutdown,
            next_seq: 0,
        }
    }

    /// Get the next batch from the pipeline.
    /// Blocks until a batch is available or the pipeline is shut down.
    pub fn next_batch(&mut self) -> Option<Batch> {
        let batch = self.output_queue.pop()?;
        self.next_seq = batch.seq_num + 1;
        Some(batch)
    }

    /// Number of batches currently buffered.
    pub fn buffered_count(&self) -> usize {
        self.output_queue.len()
    }

    /// Shut down the pipeline. Signals all I/O workers to stop and joins them.
    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    /// Signal shutdown and join worker threads.
    fn stop_and_join(&mut self) {
        self.shutdown.store(true, Ordering::Release);
        self.output_queue.close();
        for handle in self.io_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl Drop for PrefetchPipeline {
    fn drop(&mut self) {
        self.stop_and_join();
    }
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// Create a prefetch pipeline. Returns a handle.
/// `prefetch_depth`: number of batches to buffer ahead.
/// `batch_size`: samples per batch.
/// `num_workers`: I/O threads.
/// Returns: pipeline handle (pointer), or 0 on error.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_data_prefetch_create(
    prefetch_depth: i64,
    batch_size: i64,
    num_workers: i64,
) -> i64 {
    let config = PrefetchConfig {
        prefetch_depth: (prefetch_depth as usize).max(1),
        batch_size: (batch_size as usize).max(1),
        num_io_workers: (num_workers as usize).max(1),
    };
    // Create with a no-op data function (caller wires the real one via set_data_fn)
    let pipeline = PrefetchPipeline::new(config, |_idx| Vec::new());
    let boxed = Box::new(pipeline);
    Box::into_raw(boxed) as i64
}

/// Destroy a prefetch pipeline.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_data_prefetch_destroy(handle: i64) -> i64 {
    if handle == 0 { return -1; }
    let pipeline = unsafe { Box::from_raw(handle as *mut PrefetchPipeline) };
    pipeline.shutdown();
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_channel_basic() {
        let ch = BoundedChannel::new(2);
        assert!(ch.push(1));
        assert!(ch.push(2));
        // Queue is now full (capacity=2)

        assert_eq!(ch.pop(), Some(1));
        assert_eq!(ch.pop(), Some(2));
    }

    #[test]
    fn test_bounded_channel_close() {
        let ch: Arc<BoundedChannel<i32>> = Arc::new(BoundedChannel::new(2));
        let ch2 = Arc::clone(&ch);

        let handle = thread::spawn(move || {
            ch2.pop() // will block until close
        });

        // Close the channel — waiting pop should return None
        thread::sleep(std::time::Duration::from_millis(10));
        ch.close();
        let result = handle.join().unwrap();
        assert_eq!(result, None);
    }

    #[test]
    fn test_prefetch_pipeline_produces_batches() {
        let config = PrefetchConfig {
            prefetch_depth: 2,
            batch_size: 4,
            num_io_workers: 1,
        };

        let mut pipeline = PrefetchPipeline::new(config, |idx| {
            format!("sample_{idx}").into_bytes()
        });

        // Get first batch
        let batch = pipeline.next_batch().unwrap();
        assert_eq!(batch.batch_size, 4);
        assert_eq!(batch.sample_offsets.len(), 4);

        // Verify samples are present
        assert!(batch.data.len() > 0);

        pipeline.shutdown();
    }

    #[test]
    fn test_prefetch_pipeline_multiple_batches() {
        let config = PrefetchConfig {
            prefetch_depth: 4,
            batch_size: 2,
            num_io_workers: 1,
        };

        let mut pipeline = PrefetchPipeline::new(config, |idx| {
            vec![idx as u8; 10]
        });

        // Consume 3 batches
        for _ in 0..3 {
            let batch = pipeline.next_batch().unwrap();
            assert_eq!(batch.batch_size, 2);
        }

        pipeline.shutdown();
    }

    #[test]
    fn test_bounded_channel_backpressure() {
        // Verify that push blocks when queue is full
        let ch = Arc::new(BoundedChannel::new(1));
        ch.push(42);
        assert_eq!(ch.len(), 1);

        // Pop to unblock
        assert_eq!(ch.pop(), Some(42));
        assert_eq!(ch.len(), 0);
    }

    /// The lost wakeup, deterministically: a thread between its check of
    /// `closed` and its `wait` holds the channel's lock, and `close` must
    /// not get past it. Holding the lock here stands in for that thread.
    /// When `closed` was an atomic set without the lock, `close` finished
    /// at once, which is the interleaving that lost the notification.
    #[test]
    fn close_cannot_slip_between_a_waiters_check_and_its_wait() {
        let ch: Arc<BoundedChannel<u64>> = Arc::new(BoundedChannel::new(1));
        let held = ch.state.lock().unwrap();
        let ch2 = Arc::clone(&ch);
        let (tx, rx) = std::sync::mpsc::channel();
        let closer = thread::spawn(move || {
            ch2.close();
            tx.send(()).unwrap();
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_millis(300)).is_err(),
            "close() completed while another thread held the channel's lock"
        );
        assert!(!held.closed);
        drop(held);
        rx.recv_timeout(std::time::Duration::from_secs(30))
            .expect("close() did not complete once the lock was free");
        closer.join().unwrap();
        assert!(!ch.push(7), "a closed channel accepts no item");
    }

    /// The race itself, swept: a producer blocks in `push` on a full queue
    /// while `close` lands after a varying delay, so over the iterations it
    /// falls on every point of the producer's way into `wait`. On the old
    /// channel this hung within 8,000 to 131,000 iterations in three runs
    /// (the deterministic test above is the one that always fails there);
    /// a hang is reported by the watchdog instead of stalling the suite.
    #[test]
    fn close_wakes_a_blocked_producer_whenever_it_lands() {
        let (tx, rx) = std::sync::mpsc::channel();
        thread::spawn(move || {
            for i in 0..20_000u64 {
                let ch: Arc<BoundedChannel<u64>> = Arc::new(BoundedChannel::new(1));
                assert!(ch.push(0));
                let ch2 = Arc::clone(&ch);
                let producer = thread::spawn(move || ch2.push(1));
                for _ in 0..(i * 7919) % 40_000 {
                    std::hint::spin_loop();
                }
                ch.close();
                assert!(!producer.join().unwrap(), "a push past close() succeeded");
                if i % 1000 == 0 {
                    tx.send(i).unwrap();
                }
            }
            tx.send(u64::MAX).unwrap();
        });
        let mut last = 0;
        loop {
            match rx.recv_timeout(std::time::Duration::from_secs(60)) {
                Ok(u64::MAX) => break,
                Ok(i) => last = i,
                Err(_) => panic!("close() lost a producer's wakeup after iteration {last}"),
            }
        }
    }
}
