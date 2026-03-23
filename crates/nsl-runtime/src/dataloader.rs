//! Multi-threaded DataLoader runtime with reorder buffer.
//!
//! Spawns worker threads that produce batches in parallel, stores them in a
//! reorder buffer keyed by batch ID, and yields them to the caller in
//! deterministic (sequential) order.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use crate::cpu::create_tensor_with_shape_rs_dtype;
use crate::dict::{nsl_dict_free, nsl_dict_new, nsl_dict_set_str};
use crate::packing::{pack_batch, packed_batch_to_dict};
use crate::string::nsl_str_from_rust;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

// Fields are reserved for future use (parallelism, memory-pinning, etc.)
#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DataLoaderConfig {
    batch_size: usize,
    seq_len: usize,
    shuffle: bool,
    num_workers: usize,
    prefetch: usize,
    pin_memory: bool,
    drop_last: bool,
    packing: bool,
    pack_separator: i64,
}

impl DataLoaderConfig {
    fn from_json(json_ptr: *const u8, json_len: usize) -> Self {
        let slice = unsafe { std::slice::from_raw_parts(json_ptr, json_len) };
        let text = std::str::from_utf8(slice).expect("config JSON must be valid UTF-8");
        let v: serde_json::Value =
            serde_json::from_str(text).expect("config must be valid JSON");

        DataLoaderConfig {
            batch_size: v["batch_size"].as_u64().unwrap_or(1) as usize,
            seq_len: v["seq_len"].as_u64().unwrap_or(128) as usize,
            shuffle: v["shuffle"].as_bool().unwrap_or(false),
            num_workers: v["num_workers"].as_u64().unwrap_or(1) as usize,
            prefetch: v["prefetch"].as_u64().unwrap_or(2) as usize,
            pin_memory: v["pin_memory"].as_bool().unwrap_or(false),
            drop_last: v["drop_last"].as_bool().unwrap_or(false),
            packing: v["packing"].as_bool().unwrap_or(false),
            pack_separator: v["pack_separator"].as_i64().unwrap_or(0),
        }
    }
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

struct DataLoader {
    data: *const f64,
    data_len: usize,
    /// Source data dtype: 0=f64, 1=f32, 3=u16 (pretokenized).
    /// When dtype=3, `data` is actually `*const u16` cast to `*const f64`.
    data_dtype: u16,
    config: DataLoaderConfig,
    cursor: Arc<AtomicUsize>,
    stop_flag: Arc<AtomicBool>,
    reorder_buffer: Arc<Mutex<HashMap<usize, i64>>>,
    expected_batch_id: Arc<AtomicUsize>,
    condvar: Arc<Condvar>,
    worker_handles: Vec<JoinHandle<()>>,
    total_batches: usize,
    _shuffle_offsets: Vec<usize>,
}

unsafe impl Send for DataLoader {}
unsafe impl Sync for DataLoader {}

impl DataLoader {
    fn new(data: *const f64, data_len: usize, data_dtype: u16, config: DataLoaderConfig) -> Self {
        let tokens_per_batch = config.batch_size * config.seq_len;
        let total_batches = if tokens_per_batch > 0 {
            data_len / tokens_per_batch
        } else {
            0
        };

        DataLoader {
            data,
            data_len,
            data_dtype,
            config,
            cursor: Arc::new(AtomicUsize::new(0)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            reorder_buffer: Arc::new(Mutex::new(HashMap::new())),
            expected_batch_id: Arc::new(AtomicUsize::new(0)),
            condvar: Arc::new(Condvar::new()),
            worker_handles: Vec::new(),
            total_batches,
            _shuffle_offsets: Vec::new(),
        }
    }

    fn start_workers(&mut self) {
        self.stop_flag.store(false, Ordering::SeqCst);
        self.cursor.store(0, Ordering::SeqCst);
        self.expected_batch_id.store(0, Ordering::SeqCst);

        let num_workers = self.config.num_workers.max(1);
        let data_usize = self.data as usize; // Convert to usize for Send
        let data_len = self.data_len;
        let data_dtype = self.data_dtype;
        let batch_size = self.config.batch_size;
        let seq_len = self.config.seq_len;
        let packing = self.config.packing;
        let pack_separator = self.config.pack_separator;
        let total_batches = self.total_batches;

        for _ in 0..num_workers {
            let cursor = Arc::clone(&self.cursor);
            let stop_flag = Arc::clone(&self.stop_flag);
            let reorder_buffer = Arc::clone(&self.reorder_buffer);
            let condvar = Arc::clone(&self.condvar);

            let handle = thread::spawn(move || {
                let data_ptr = data_usize as *const f64;

                loop {
                    if stop_flag.load(Ordering::SeqCst) {
                        break;
                    }

                    // Atomically claim the next batch_id
                    let batch_id = cursor.fetch_add(1, Ordering::SeqCst);
                    if batch_id >= total_batches {
                        // Put back so other workers also see exhaustion
                        break;
                    }

                    let tokens_per_batch = batch_size * seq_len;
                    let dict_ptr = if packing {
                        let mut cur = batch_id * tokens_per_batch;
                        match pack_batch(
                            data_ptr,
                            data_len,
                            &mut cur,
                            batch_size,
                            seq_len,
                            pack_separator,
                        ) {
                            Some(batch) => packed_batch_to_dict(&batch),
                            None => continue,
                        }
                    } else {
                        build_simple_batch(
                            data_ptr,
                            data_len,
                            batch_id * tokens_per_batch,
                            batch_size,
                            seq_len,
                            data_dtype,
                        )
                    };

                    // Insert into reorder buffer
                    {
                        let mut buf = reorder_buffer.lock().unwrap();
                        buf.insert(batch_id, dict_ptr);
                    }
                    condvar.notify_all();
                }
            });
            self.worker_handles.push(handle);
        }
    }

    fn stop_workers(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.condvar.notify_all();
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Simple (non-packed) batch builder
// ---------------------------------------------------------------------------

/// Build a standard batch dict without packing: sequential token slice,
/// labels shifted by 1, standard causal mask.
fn build_simple_batch(
    data: *const f64,
    data_len: usize,
    offset: usize,
    batch_size: usize,
    seq_len: usize,
    data_dtype: u16,
) -> i64 {
    let total = batch_size * seq_len;
    if offset + total > data_len {
        return 0;
    }

    // Read input_ids — dtype-aware: f64(0), f32(1), u16(3)
    let mut input_ids = Vec::with_capacity(total);
    match data_dtype {
        3 => {
            // u16 pretokenized data (zero-copy mmap)
            let src = data as *const u16;
            if offset + total > data_len {
                return 0;
            }
            for i in 0..total {
                let val = unsafe { *src.add(offset + i) } as i64;
                input_ids.push(val);
            }
        }
        1 => {
            // f32 data
            let src = data as *const f32;
            for i in 0..total {
                let val = unsafe { *src.add(offset + i) } as i64;
                input_ids.push(val);
            }
        }
        _ => {
            // f64 data (default)
            for i in 0..total {
                let val = unsafe { *data.add(offset + i) } as i64;
                input_ids.push(val);
            }
        }
    }

    // Build labels: shifted by 1 within each sequence, -100 at last position
    let mut labels = vec![0i64; total];
    for b in 0..batch_size {
        let base = b * seq_len;
        if seq_len > 0 {
            for i in 0..seq_len - 1 {
                labels[base + i] = input_ids[base + i + 1];
            }
            labels[base + seq_len - 1] = -100;
        }
    }

    // NOTE: Causal attention mask is NOT generated here.
    // The model's GQA layer calls causal_mask(seq_len) internally, which creates
    // a [seq_len, seq_len] mask that broadcasts across the batch dimension.
    // Generating a [batch, seq, seq] mask here wastes 134MB/batch at batch=32,
    // seq=1024 — pure PCIe/memory bandwidth waste for a static pattern.

    // Create tensors and dict (f32, dtype=1, to match default tensor dtype)
    let b = batch_size as i64;
    let s = seq_len as i64;

    let ids_ptr = create_tensor_with_shape_rs_dtype(&[b, s], 1);
    let ids_tensor = NslTensor::from_ptr(ids_ptr);
    let ids_data = ids_tensor.data_f32();
    for (i, &v) in input_ids.iter().enumerate() {
        unsafe { *ids_data.add(i) = v as f32 };
    }

    let lbl_ptr = create_tensor_with_shape_rs_dtype(&[b, s], 1);
    let lbl_tensor = NslTensor::from_ptr(lbl_ptr);
    let lbl_data = lbl_tensor.data_f32();
    for (i, &v) in labels.iter().enumerate() {
        unsafe { *lbl_data.add(i) = v as f32 };
    }

    let dict = nsl_dict_new();
    let k_ids = nsl_str_from_rust("input_ids");
    let k_lbl = nsl_str_from_rust("labels");
    nsl_dict_set_str(dict, k_ids, ids_ptr);
    nsl_dict_set_str(dict, k_lbl, lbl_ptr);

    dict
}

// ---------------------------------------------------------------------------
// FFI functions
// ---------------------------------------------------------------------------

/// Create a new DataLoader.
///
/// `data_ptr` — pointer to flat f64 token array (cast to i64)
/// `data_len` — number of tokens
/// `config_ptr` — pointer to UTF-8 JSON config string (cast to i64)
/// `config_len` — byte length of config string
///
/// Returns an opaque DataLoader handle (i64).
#[no_mangle]
pub extern "C" fn nsl_dataloader_create(
    data_ptr: i64,
    data_len: i64,
    config_ptr: i64,
    config_len: i64,
    data_dtype: i64,
) -> i64 {
    let data = data_ptr as *const f64;
    let config =
        DataLoaderConfig::from_json(config_ptr as *const u8, config_len as usize);
    let dl = DataLoader::new(data, data_len as usize, data_dtype as u16, config);
    Box::into_raw(Box::new(dl)) as i64
}

/// Start (or restart) the worker threads.
#[no_mangle]
pub extern "C" fn nsl_dataloader_start(dl_ptr: i64) {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    dl.start_workers();
}

/// Retrieve the next batch in deterministic order.
///
/// Returns a dict pointer (i64) or 0 when the epoch is exhausted.
#[no_mangle]
pub extern "C" fn nsl_dataloader_next_batch(dl_ptr: i64) -> i64 {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    let expected = dl.expected_batch_id.load(Ordering::SeqCst);

    if expected >= dl.total_batches {
        return 0;
    }

    // Wait until the expected batch is in the reorder buffer
    let dict_ptr = {
        let mut buf = dl.reorder_buffer.lock().unwrap();
        loop {
            if let Some(ptr) = buf.remove(&expected) {
                break ptr;
            }
            if dl.stop_flag.load(Ordering::SeqCst) {
                return 0;
            }
            // If all workers finished, do one final check — a worker may have
            // inserted the batch between our remove() and this is_finished() check.
            if dl.worker_handles.iter().all(|h| h.is_finished()) {
                if let Some(ptr) = buf.remove(&expected) {
                    break ptr;
                }
                return 0;
            }
            buf = dl.condvar.wait(buf).unwrap();
        }
    };

    dl.expected_batch_id.fetch_add(1, Ordering::SeqCst);
    dict_ptr
}

/// Reset the dataloader for a new epoch: stop workers, clear state, restart.
#[no_mangle]
pub extern "C" fn nsl_dataloader_reset(dl_ptr: i64) {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    dl.stop_workers();

    // Clear any remaining batches in the reorder buffer
    {
        let mut buf = dl.reorder_buffer.lock().unwrap();
        for (_, dict_ptr) in buf.drain() {
            nsl_dict_free(dict_ptr);
        }
    }

    dl.cursor.store(0, Ordering::SeqCst);
    dl.expected_batch_id.store(0, Ordering::SeqCst);
    dl.start_workers();
}

/// Stop all worker threads (without freeing the DataLoader).
#[no_mangle]
pub extern "C" fn nsl_dataloader_stop(dl_ptr: i64) {
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    dl.stop_workers();
}

/// Free the DataLoader and all resources.
#[no_mangle]
pub extern "C" fn nsl_dataloader_free(dl_ptr: i64) {
    if dl_ptr == 0 {
        return;
    }
    let mut dl = unsafe { *Box::from_raw(dl_ptr as *mut DataLoader) };
    dl.stop_workers();

    // Free any remaining batches in the reorder buffer
    {
        let mut buf = dl.reorder_buffer.lock().unwrap();
        for (_, dict_ptr) in buf.drain() {
            nsl_dict_free(dict_ptr);
        }
    }

    // DataLoader is dropped here
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::nsl_dict_len;

    fn make_config_json(
        batch_size: usize,
        seq_len: usize,
        num_workers: usize,
        packing: bool,
    ) -> String {
        format!(
            r#"{{"batch_size":{},"seq_len":{},"num_workers":{},"packing":{},"shuffle":false,"prefetch":2,"pin_memory":false,"drop_last":false,"pack_separator":0}}"#,
            batch_size, seq_len, num_workers, packing
        )
    }

    #[test]
    fn test_dataloader_basic() {
        // 32 tokens: [0..31]
        let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let config_json = make_config_json(2, 4, 1, false);

        let dl_ptr = nsl_dataloader_create(
            data.as_ptr() as i64,
            data.len() as i64,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
            0, // dtype=0 (f64)
        );
        assert!(dl_ptr != 0);

        nsl_dataloader_start(dl_ptr);

        // total_batches = 32 / (2 * 4) = 4
        let mut batch_count = 0;
        loop {
            let batch = nsl_dataloader_next_batch(dl_ptr);
            if batch == 0 {
                break;
            }
            assert_eq!(nsl_dict_len(batch), 2); // input_ids + labels (no attention_mask)
            nsl_dict_free(batch);
            batch_count += 1;
        }
        assert_eq!(batch_count, 4);

        nsl_dataloader_stop(dl_ptr);
        nsl_dataloader_free(dl_ptr);
    }

    #[test]
    fn test_dataloader_deterministic_order() {
        // 16 tokens: [0..15], 2 workers
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let config_json = make_config_json(1, 4, 2, false);

        let dl_ptr = nsl_dataloader_create(
            data.as_ptr() as i64,
            data.len() as i64,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
            0, // dtype=0 (f64)
        );

        nsl_dataloader_start(dl_ptr);

        // First batch should be tokens [0, 1, 2, 3] regardless of which worker produced it
        let batch = nsl_dataloader_next_batch(dl_ptr);
        assert!(batch != 0);

        let k = nsl_str_from_rust("input_ids");
        let tensor_ptr = crate::dict::nsl_dict_get_str(batch, k);
        let tensor = NslTensor::from_ptr(tensor_ptr);

        let ids_data = tensor.data_f32();
        for i in 0..4 {
            let val = unsafe { *ids_data.add(i) } as i64;
            assert_eq!(val, i as i64, "token at position {} should be {}", i, i);
        }

        nsl_dict_free(batch);

        // Drain remaining batches
        while nsl_dataloader_next_batch(dl_ptr) != 0 {}

        nsl_dataloader_stop(dl_ptr);
        nsl_dataloader_free(dl_ptr);
    }
}
