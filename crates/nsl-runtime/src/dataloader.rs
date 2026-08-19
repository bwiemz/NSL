//! Multi-threaded DataLoader runtime with reorder buffer.
//!
//! Spawns worker threads that produce batches in parallel, stores them in a
//! reorder buffer keyed by batch ID, and yields them to the caller in
//! deterministic (sequential) order.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Default cross-rank shuffle seed for data-parallel (world_size > 1). All
/// ranks MUST derive the identical global permutation each epoch, so the seed
/// is a fixed constant (XORed with the epoch), never entropy — otherwise the
/// strided per-rank shard partition would be neither disjoint nor complete.
const DEFAULT_DP_SHUFFLE_SEED: u64 = 0x1970_D31A_10AD_5EED;

use crate::cpu::create_tensor_with_shape_rs_dtype;
use crate::dict::{nsl_dict_free_tensor_values, nsl_dict_new, nsl_dict_set_str};
use crate::packing::{pack_batch, packed_batch_to_dict};
use crate::string::nsl_str_from_rust;
use crate::tensor::{DTYPE_I32, DTYPE_U16_TOKEN, NslTensor};

#[inline]
fn supports_flat_value_dtype(dtype: u16) -> bool {
    matches!(dtype, 0 | 1 | DTYPE_U16_TOKEN)
}

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
    /// Opt-in dense `[b, s, s]` attention_mask in packed batches (Stage-B
    /// `forward_masked` consumers). Default OFF: the packed-attention path
    /// keys off `segment_ids`, and the decomposed fallback derives the mask
    /// on demand (`nsl_packed_mask_from_segment_ids`).
    emit_attention_mask: bool,
    /// D3 v3: data-parallel world size baked by codegen (from
    /// `self.features.world_size` when `--zero-stage >= 1`). `> 1` turns on
    /// rank-aware sharding: rank `r` (from `NSL_LOCAL_RANK`) consumes the
    /// global batches at slots where `slot % world_size == r`, so all-reduced
    /// gradients average over the true global batch instead of replicated
    /// data. Default 1 = rank-blind (unchanged single-rank behavior).
    world_size: usize,
    /// D3 v3: opt-in rank-aware sharding. Only when `true` (and world_size > 1)
    /// does the loader hand each rank a disjoint 1/world_size shard. Default
    /// false = every rank sees the full data (replicated). The TRAINING loader
    /// under data-parallel sets this; eval/validation loaders must NOT (they
    /// have no gradient all-reduce to recombine a shard).
    shard_by_rank: bool,
    /// Cross-rank shuffle seed (only load-bearing when sharding is active).
    seed: u64,
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
            emit_attention_mask: v["emit_attention_mask"].as_bool().unwrap_or(false),
            world_size: v["world_size"].as_u64().unwrap_or(1).max(1) as usize,
            shard_by_rank: v["shard_by_rank"].as_bool().unwrap_or(false),
            // Item 8: with no explicit loader seed, fall back to the global
            // `--seed` store when the program set one, so `--seed` controls
            // the data ORDER too (it already controls init and dropout).
            // Unseeded runs keep the historical constant.
            seed: v["seed"].as_u64().unwrap_or_else(|| {
                crate::deterministic_ops::explicit_rng_seed()
                    .unwrap_or(DEFAULT_DP_SHUFFLE_SEED)
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// DataLoader
// ---------------------------------------------------------------------------

struct DataLoader {
    data_tensor_ptr: i64,
    data: *const c_void,
    data_len: usize,
    /// Source data dtype: 0=f64, 1=f32, internal u16 token buffer dtype.
    data_dtype: u16,
    labels_tensor_ptr: i64,
    labels: *const c_void,
    labels_len: usize,
    labels_dtype: u16,
    has_labels: bool,
    config: DataLoaderConfig,
    cursor: Arc<AtomicUsize>,
    stop_flag: Arc<AtomicBool>,
    reorder_buffer: Arc<Mutex<HashMap<usize, i64>>>,
    expected_batch_id: Arc<AtomicUsize>,
    condvar: Arc<Condvar>,
    worker_handles: Vec<JoinHandle<()>>,
    /// D3 v3: batches THIS rank yields per epoch = global_total_batches /
    /// world_size (FLOOR). Every rank yields exactly this many so all ranks
    /// reach the same number of gradient all-reduce/broadcast collectives —
    /// an unequal count would hang the ZeRO spin-barrier. Used by next_batch
    /// termination and the worker slot space.
    total_batches: usize,
    /// D3 v3: total batches across ALL ranks (the un-sharded count). The full
    /// global order is built from this, then strided to this rank's shard.
    global_total_batches: usize,
    /// D3 v3: this rank's DP index (from NSL_LOCAL_RANK; 0 when rank-blind).
    rank: usize,
    /// D3 v3: epoch counter, advanced by reset(), folded into the shuffle seed
    /// so each epoch's global permutation differs but stays identical across
    /// ranks (lockstep resets in SPMD).
    epoch: Arc<AtomicUsize>,
    _shuffle_offsets: Vec<usize>,
    /// Item 8: a pending checkpoint resume, `(epoch, slot)`, consumed by the
    /// NEXT [`nsl_dataloader_reset`]. That reset pins the epoch instead of
    /// incrementing it (so the permutation is the saved epoch's) and starts
    /// the cursor at `slot` instead of 0 (so the batches already trained on
    /// are not re-delivered). Armed rather than applied immediately because
    /// the train block's epoch loop resets at the top of every epoch,
    /// including the first one after the resume.
    pending_resume: Option<(usize, usize)>,
}

unsafe impl Send for DataLoader {}
unsafe impl Sync for DataLoader {}

impl DataLoader {
    fn new(
        data_tensor_ptr: i64,
        labels_tensor_ptr: i64,
        config: DataLoaderConfig,
    ) -> Self {
        if data_tensor_ptr == 0 {
            eprintln!("nsl: DataLoader requires a data tensor");
            std::process::abort();
        }

        let data_tensor = NslTensor::from_ptr(data_tensor_ptr);
        if data_tensor.device != 0 {
            eprintln!("nsl: DataLoader data tensor must be on CPU");
            std::process::abort();
        }
        if !data_tensor.is_contiguous() {
            eprintln!("nsl: DataLoader data tensor must be contiguous");
            std::process::abort();
        }

        let data = data_tensor.data as *const c_void;
        let data_len = data_tensor.len as usize;
        let data_dtype = data_tensor.dtype;
        if !supports_flat_value_dtype(data_dtype) {
            eprintln!("nsl: DataLoader does not support source dtype {}", data_dtype);
            std::process::abort();
        }

        let (labels, labels_len, labels_dtype) = if labels_tensor_ptr != 0 {
            let labels_tensor = NslTensor::from_ptr(labels_tensor_ptr);
            if labels_tensor.device != 0 {
                eprintln!("nsl: DataLoader labels tensor must be on CPU");
                std::process::abort();
            }
            if !labels_tensor.is_contiguous() {
                eprintln!("nsl: DataLoader labels tensor must be contiguous");
                std::process::abort();
            }
            (
                labels_tensor.data as *const c_void,
                labels_tensor.len as usize,
                labels_tensor.dtype,
            )
        } else {
            (std::ptr::null(), 0, 0)
        };

        let has_labels = !labels.is_null() && labels_len > 0;
        if has_labels && !supports_flat_value_dtype(labels_dtype) {
            eprintln!("nsl: DataLoader does not support label dtype {}", labels_dtype);
            std::process::abort();
        }
        if has_labels && labels_len != data_len {
            eprintln!(
                "nsl: DataLoader labels length mismatch: inputs={} labels={}",
                data_len, labels_len
            );
            std::process::abort();
        }
        let tokens_per_batch = config.batch_size * config.seq_len;

        // D3 v3: rank-aware data-parallel sharding is OPT-IN via `shard_by_rank`
        // (default off) and only meaningful across multiple ranks. It must NOT
        // auto-activate on every DataLoader: an eval/validation loader is not
        // wrapped by the gradient all-reduce, so sharding it would make each
        // rank report metrics over only 1/world_size of the data. The training
        // loader opts in explicitly (`DataLoader(..., shard_by_rank=true)`).
        let world_size = config.world_size.max(1);
        let sharding = world_size > 1 && config.shard_by_rank;

        // rank comes from NSL_LOCAL_RANK (set by the `nsl run --devices N`
        // spawner), mirroring zero.rs.
        let local_rank_env = std::env::var("NSL_LOCAL_RANK").ok();
        // Refuse loudly if sharding is requested but this process was NOT
        // launched by the --devices spawner (NSL_LOCAL_RANK unset): sharding a
        // lone process would silently train on only its 1/world_size slice.
        if sharding && local_rank_env.is_none() {
            eprintln!(
                "nsl: DataLoader(shard_by_rank=true) with world_size={world_size} but \
                 NSL_LOCAL_RANK is unset — run under `nsl run --devices {world_size} \
                 --zero-stage N`. Refusing to shard a single process (would train on \
                 only 1/{world_size} of the data)."
            );
            std::process::abort();
        }
        let rank = local_rank_env
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            .min(world_size - 1);

        // Under sharding, drop the ragged partial tail (FLOOR, regardless of
        // drop_last) so no batch_id points at a short slice: a PACKED partial
        // tail makes a worker skip a claimed delivery slot, yielding an unequal
        // per-rank count that hangs the ZeRO spin-barrier. Dropping the tail is
        // standard DDP behavior (re-randomized across epochs via the shuffle).
        let global_total_batches = if tokens_per_batch == 0 {
            0
        } else if config.drop_last || sharding {
            data_len / tokens_per_batch
        } else {
            data_len.div_ceil(tokens_per_batch)
        };

        // Per-rank count is the FLOOR so every rank yields the SAME number of
        // batches (equal collective count → no spin-barrier deadlock).
        let total_batches = if sharding {
            global_total_batches / world_size
        } else {
            global_total_batches
        };

        // Refuse loudly if the corpus can't give every rank ≥1 batch (floor
        // would be 0 → a silent no-op, frozen-model, exit-0 run — the M43b
        // anti-pattern).
        if sharding && total_batches == 0 {
            eprintln!(
                "nsl: DataLoader can't shard {global_total_batches} batches across \
                 {world_size} ranks (< 1 per rank). Reduce --devices, batch_size, or \
                 seq_len, or provide more data."
            );
            std::process::abort();
        }

        DataLoader {
            data_tensor_ptr,
            data,
            data_len,
            data_dtype,
            labels_tensor_ptr,
            labels,
            labels_len,
            labels_dtype,
            has_labels,
            config,
            cursor: Arc::new(AtomicUsize::new(0)),
            stop_flag: Arc::new(AtomicBool::new(false)),
            reorder_buffer: Arc::new(Mutex::new(HashMap::new())),
            expected_batch_id: Arc::new(AtomicUsize::new(0)),
            condvar: Arc::new(Condvar::new()),
            worker_handles: Vec::new(),
            total_batches,
            global_total_batches,
            rank,
            epoch: Arc::new(AtomicUsize::new(0)),
            _shuffle_offsets: Vec::new(),
            pending_resume: None,
        }
    }

    /// Start (or restart) workers at delivery slot `start_slot`. Everything
    /// before `start_slot` in this epoch's permutation is never claimed, so a
    /// resume skips it in O(1) instead of building and discarding batches.
    fn start_workers_at(&mut self, start_slot: usize) {
        self.stop_flag.store(false, Ordering::Release);
        let start_slot = start_slot.min(self.total_batches);
        self.cursor.store(start_slot, Ordering::Relaxed);
        self.expected_batch_id.store(start_slot, Ordering::Relaxed);

        // D3 v3: build the GLOBAL batch order first, then (only when sharding
        // is opted in) stride to this rank's shard. When NOT sharding this
        // reduces exactly to the original `(0..total_batches)` + entropy
        // shuffle — byte-identical, whether world_size is 1 OR a replicated
        // multi-rank loader (shard_by_rank=false).
        let sharding = self.config.world_size > 1 && self.config.shard_by_rank;
        let mut batch_order: Vec<usize> = (0..self.global_total_batches).collect();
        if self.config.shuffle {
            // Deterministic shuffle keyed by (seed, epoch) on EVERY path.
            //
            // Sharded: every rank must derive the identical global permutation
            // or the strided partition is neither disjoint nor complete.
            //
            // Single-rank: this used to be `rand::rng()` — entropy. Item 8
            // makes it seeded because a checkpoint that records "resume at
            // epoch 3, slot 412" is meaningless if epoch 3's permutation is a
            // fresh random draw each process. The cost is that an unseeded
            // run now repeats the same order run-to-run (standard seeded-
            // training behavior); epochs still differ via the XOR.
            // MIXED, not `seed ^ epoch`. XOR is not injective across small
            // (seed, epoch) pairs — seed 1 epoch 0 and seed 0 epoch 1 are the
            // same key. That was harmless while `config.seed` was almost
            // always the large `DEFAULT_DP_SHUFFLE_SEED` constant, but the
            // fallback added just above routes a small `--seed` in, which
            // makes the collision reachable: an 8-seed sweep would have its
            // arms enumerate ONE shared set of permutations in a swapped
            // order. All ranks compute this identically, which is the only
            // property the sharded path requires of the key.
            let epoch = self.epoch.load(Ordering::Relaxed) as u64;
            let key = crate::sr_bf16::sr_mix64(self.config.seed, epoch);
            let mut rng = StdRng::seed_from_u64(key);
            batch_order.shuffle(&mut rng);
        }
        // Strided per-rank shard: rank r takes global slots where slot % ws == r,
        // capped to total_batches (floor) so every rank yields an equal count.
        if sharding {
            let rank = self.rank;
            let ws = self.config.world_size;
            batch_order = batch_order
                .into_iter()
                .enumerate()
                .filter(|(slot, _)| slot % ws == rank)
                .map(|(_, b)| b)
                .take(self.total_batches)
                .collect();
        }
        self._shuffle_offsets = batch_order.clone();
        let batch_order = Arc::new(batch_order);

        let num_workers = self.config.num_workers.max(1);
        let data_usize = self.data as usize; // Convert to usize for Send
        let data_len = self.data_len;
        let data_dtype = self.data_dtype;
        let labels_usize = self.labels as usize;
        let labels_len = self.labels_len;
        let labels_dtype = self.labels_dtype;
        let has_labels = self.has_labels;
        let batch_size = self.config.batch_size;
        let seq_len = self.config.seq_len;
        let packing = self.config.packing;
        let pack_separator = self.config.pack_separator;
        let emit_attention_mask = self.config.emit_attention_mask;
        let total_batches = self.total_batches;
        let prefetch_limit = self.config.prefetch.max(1);

        for _ in 0..num_workers {
            let cursor = Arc::clone(&self.cursor);
            let stop_flag = Arc::clone(&self.stop_flag);
            let reorder_buffer = Arc::clone(&self.reorder_buffer);
            let condvar = Arc::clone(&self.condvar);
            let batch_order = Arc::clone(&batch_order);

            let handle = thread::spawn(move || {
                let data_ptr = data_usize as *const c_void;
                let labels_ptr = labels_usize as *const c_void;

                loop {
                    if stop_flag.load(Ordering::Acquire) {
                        break;
                    }

                    {
                        let mut buf = reorder_buffer.lock().unwrap();
                        while buf.len() >= prefetch_limit && !stop_flag.load(Ordering::Acquire) {
                            buf = condvar.wait(buf).unwrap();
                        }
                    }

                    if stop_flag.load(Ordering::Acquire) {
                        break;
                    }

                    // Atomically claim the next delivery slot. The shuffled order,
                    // when enabled, is applied when mapping this slot to source data.
                    let work_index = cursor.fetch_add(1, Ordering::Relaxed);
                    if work_index >= total_batches {
                        // Put back so other workers also see exhaustion
                        break;
                    }
                    let batch_id = batch_order[work_index];

                    let tokens_per_batch = batch_size * seq_len;
                    let dict_ptr = if packing {
                        let mut cur = batch_id * tokens_per_batch;
                        match pack_batch(
                            data_ptr,
                            data_len,
                            data_dtype,
                            &mut cur,
                            batch_size,
                            seq_len,
                            pack_separator,
                            emit_attention_mask,
                        ) {
                            Some(batch) => packed_batch_to_dict(&batch),
                            // Ragged packed tail: this CLAIMED work_index would
                            // otherwise become a permanent HOLE in the reorder
                            // sequence — under shuffle the tail id can land at
                            // an EARLY slot, the consumer parks on the hole,
                            // workers fill the buffer and park on the full
                            // gate: a wakeup-independent deadlock (review H1).
                            // Insert a 0 sentinel so the consumer skips the
                            // slot instead of waiting on it forever.
                            None => 0,
                        }
                    } else {
                        build_simple_batch(
                            data_ptr,
                            data_len,
                            labels_ptr,
                            labels_len,
                            batch_id * tokens_per_batch,
                            batch_size,
                            seq_len,
                            data_dtype,
                            labels_dtype,
                            has_labels,
                        )
                    };

                    // Insert into reorder buffer
                    {
                        let mut buf = reorder_buffer.lock().unwrap();
                        buf.insert(work_index, dict_ptr);
                    }
                    // TWO waiter classes share this condvar: the consumer (waiting
                    // for `expected` to appear) and workers (waiting for buffer
                    // space at the prefetch gate). notify_one can hand this signal
                    // to a full-gated worker, which rechecks its predicate and goes
                    // back to sleep — the consumer then waits forever on a batch
                    // that is already in the buffer while every worker is parked on
                    // the full gate (observed as a multi-hour hang on Windows CI).
                    // Broadcast so the consumer always re-evaluates after an insert.
                    condvar.notify_all();
                }
                // Review M1: a worker can exit WITHOUT a final insert (cursor
                // exhaustion, stop). The consumer's is_finished() escape only
                // runs while awake — wake it unconditionally so the last
                // worker's exit is never a lost signal.
                condvar.notify_all();
            });
            self.worker_handles.push(handle);
        }
    }

    fn start_workers(&mut self) {
        self.start_workers_at(0);
    }

    /// FNV-1a over the loader's geometry and a corpus fingerprint. Recorded in
    /// the checkpoint and compared on resume: "resume at epoch 3, slot 412"
    /// only means anything against the same corpus, batch geometry and shuffle
    /// key. Fingerprint (not full hash) because the corpus is multi-GB and
    /// this runs on every save — length + dtype + the first and last MiB catch
    /// a swapped, truncated or re-tokenized corpus, which is what the guard is
    /// for. It is a pairing check, not an integrity check (same doctrine as
    /// `model_file_sig`).
    fn identity_sig(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        let mut mix = |bytes: &[u8]| {
            for &b in bytes {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        };
        let c = &self.config;
        for v in [
            c.batch_size as u64,
            c.seq_len as u64,
            c.shuffle as u64,
            c.drop_last as u64,
            c.packing as u64,
            c.pack_separator as u64,
            c.world_size as u64,
            c.shard_by_rank as u64,
            c.seed,
            self.rank as u64,
            self.total_batches as u64,
            self.global_total_batches as u64,
            self.data_len as u64,
            self.data_dtype as u64,
            self.has_labels as u64,
            self.labels_len as u64,
            self.labels_dtype as u64,
        ] {
            mix(&v.to_le_bytes());
        }
        // Corpus fingerprint: the loader holds CPU-contiguous buffers, so
        // these are plain host reads. Labels are fingerprinted too — a run
        // that swapped only the label stream trains on entirely different
        // targets while the token stream (and every count above) matches.
        let elem_size = |dtype: u16| match dtype {
            DTYPE_U16_TOKEN => 2usize,
            1 => 4,
            _ => 8,
        };
        for (ptr, len, dtype) in [
            (self.data, self.data_len, self.data_dtype),
            (self.labels, self.labels_len, self.labels_dtype),
        ] {
            let nbytes = len * elem_size(dtype);
            if ptr.is_null() || nbytes == 0 {
                continue;
            }
            let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, nbytes) };
            const CHUNK: usize = 1 << 20;
            mix(&bytes[..CHUNK.min(nbytes)]);
            if nbytes > CHUNK {
                mix(&bytes[nbytes - CHUNK..]);
            }
        }
        h
    }

    fn stop_workers(&mut self) {
        // Review M2: store the flag and notify while HOLDING the buffer mutex.
        // Without it, a worker can evaluate the full-gate predicate (sees
        // stop=false), lose the CPU, miss this notify, and park after it fired
        // — join below then blocks forever. Holding the mutex orders the
        // store+notify against the worker's check-then-wait, which happens
        // under the same mutex.
        {
            let _buf = self.reorder_buffer.lock().unwrap();
            self.stop_flag.store(true, Ordering::Release);
            self.condvar.notify_all();
        }
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
fn read_flat_value(data: *const c_void, dtype: u16, index: usize) -> i64 {
    match dtype {
        DTYPE_U16_TOKEN => unsafe { *(data as *const u16).add(index) as i64 },
        1 => unsafe { *(data as *const f32).add(index) as i64 },
        0 => unsafe { *(data as *const f64).add(index) as i64 },
        _ => panic!("read_flat_value() unsupported dtype {}", dtype),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_simple_batch(
    data: *const c_void,
    data_len: usize,
    labels: *const c_void,
    labels_len: usize,
    offset: usize,
    batch_size: usize,
    seq_len: usize,
    data_dtype: u16,
    labels_dtype: u16,
    has_labels: bool,
) -> i64 {
    let total = batch_size * seq_len;
    let available = data_len.saturating_sub(offset);
    if available == 0 {
        return 0;
    }

    // Read input_ids — pad with 0 beyond available data
    let mut input_ids = vec![0i64; total];
    let read_count = available.min(total);
    for i in 0..read_count {
        input_ids[i] = read_flat_value(data, data_dtype, offset + i);
    }

    // Build labels from the provided tensor when available; otherwise shift the
    // input_ids within each sequence and ignore only the final position.
    // Padded positions get -100 (ignore_index) so they don't contribute to loss.
    let mut labels_vec = vec![-100i64; total];
    if has_labels {
        let label_available = labels_len.saturating_sub(offset);
        let label_read = label_available.min(total);
        for i in 0..label_read {
            labels_vec[i] = read_flat_value(labels, labels_dtype, offset + i);
        }
    } else {
        for b in 0..batch_size {
            let base = b * seq_len;
            if seq_len > 0 {
                for i in 0..seq_len - 1 {
                    if base + i + 1 < read_count {
                        labels_vec[base + i] = input_ids[base + i + 1];
                    }
                    // positions beyond read_count stay -100
                }
                // Last position in each sequence is always -100
            }
        }
    }

    // NOTE: Causal attention mask is NOT generated here.
    // The model's GQA layer calls causal_mask(seq_len) internally, which creates
    // a [seq_len, seq_len] mask that broadcasts across the batch dimension.
    // Generating a [batch, seq, seq] mask here wastes 134MB/batch at batch=32,
    // seq=1024 — pure PCIe/memory bandwidth waste for a static pattern.

    // Create tensors and dict using DTYPE_I32 for token IDs.
    // Token IDs are integers that only need 32 bits (vocab < 2^31).
    // Using i32 instead of f32 halves bandwidth vs f64 and avoids
    // precision loss for large token IDs (f32 mantissa is only 24 bits).
    // The cast to float is deferred to the embedding lookup boundary.
    let b = batch_size as i64;
    let s = seq_len as i64;

    let ids_ptr = create_tensor_with_shape_rs_dtype(&[b, s], DTYPE_I32);
    let ids_tensor = NslTensor::from_ptr(ids_ptr);
    let ids_data = ids_tensor.data as *mut i32;
    for (i, &v) in input_ids.iter().enumerate() {
        unsafe { *ids_data.add(i) = v as i32 };
    }

    let lbl_ptr = create_tensor_with_shape_rs_dtype(&[b, s], DTYPE_I32);
    let lbl_tensor = NslTensor::from_ptr(lbl_ptr);
    let lbl_data = lbl_tensor.data as *mut i32;
    for (i, &v) in labels_vec.iter().enumerate() {
        unsafe { *lbl_data.add(i) = v as i32 };
    }

    let dict = nsl_dict_new();
    let k_ids = nsl_str_from_rust("input_ids");
    let k_lbl = nsl_str_from_rust("labels");
    nsl_dict_set_str(dict, k_ids, ids_ptr);
    nsl_dict_set_str(dict, k_lbl, lbl_ptr);
    crate::string::nsl_string_free(k_ids);
    crate::string::nsl_string_free(k_lbl);

    dict
}

// ---------------------------------------------------------------------------
// FFI functions
// ---------------------------------------------------------------------------

/// Create a new DataLoader.
///
/// `data_tensor_ptr` — CPU-contiguous tensor containing the flat token stream
/// `labels_tensor_ptr` — optional CPU-contiguous tensor with precomputed labels, or 0
/// `config_ptr` — pointer to UTF-8 JSON config string (cast to i64)
/// `config_len` — byte length of config string
///
/// Returns an opaque DataLoader handle (i64).
#[no_mangle]
pub extern "C" fn nsl_dataloader_create(
    data_tensor_ptr: i64,
    labels_tensor_ptr: i64,
    config_ptr: i64,
    config_len: i64,
) -> i64 {
    let config =
        DataLoaderConfig::from_json(config_ptr as *const u8, config_len as usize);
    let dl = DataLoader::new(data_tensor_ptr, labels_tensor_ptr, config);
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
    let mut expected = dl.expected_batch_id.load(Ordering::Relaxed);

    if expected >= dl.total_batches {
        return 0;
    }

    // Wait until the expected batch is in the reorder buffer
    let dict_ptr = {
        let mut buf = dl.reorder_buffer.lock().unwrap();
        loop {
            if let Some(ptr) = buf.remove(&expected) {
                if ptr == 0 {
                    // Ragged-packed-tail sentinel (see the worker's pack_batch
                    // None arm): the slot exists but yields no batch. Skip it,
                    // wake any full-gated producer (the buffer just shrank),
                    // and keep waiting for the next slot.
                    dl.expected_batch_id.fetch_add(1, Ordering::Relaxed);
                    expected += 1;
                    dl.condvar.notify_all();
                    if expected >= dl.total_batches {
                        return 0;
                    }
                    continue;
                }
                break ptr;
            }
            if dl.stop_flag.load(Ordering::Acquire) {
                return 0;
            }
            // If all workers finished, do one final check — a worker may have
            // inserted the batch between our remove() and this is_finished() check.
            if dl.worker_handles.iter().all(|h| h.is_finished()) {
                match buf.remove(&expected) {
                    Some(0) | None => return 0,
                    Some(ptr) => break ptr,
                }
            }
            buf = dl.condvar.wait(buf).unwrap();
        }
    };

    dl.expected_batch_id.fetch_add(1, Ordering::Relaxed);
    dl.condvar.notify_all();
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
            nsl_dict_free_tensor_values(dict_ptr);
        }
    }

    // Item 8: a pending checkpoint resume replaces this reset's epoch advance
    // and start slot. Consumed exactly once — later epochs advance normally.
    if let Some((epoch, slot)) = dl.pending_resume.take() {
        dl.epoch.store(epoch, Ordering::Relaxed);
        if slot >= dl.total_batches {
            // The saved run finished this epoch; nothing of it is left to
            // replay. Yield an empty epoch rather than clamping to the last
            // batch (which would train on it a second time) — the epoch loop
            // then moves on to the next epoch, freshly permuted.
            eprintln!(
                "[checkpoint] resume slot {slot} is at/after this epoch's \
                 {} batches — resuming at the next epoch boundary",
                dl.total_batches
            );
        }
        dl.start_workers_at(slot);
        return;
    }

    dl.cursor.store(0, Ordering::Relaxed);
    dl.expected_batch_id.store(0, Ordering::Relaxed);
    // D3 v3: advance the epoch BEFORE restarting workers so each epoch's global
    // permutation differs (fed into the seeded shuffle). SPMD ranks reset in
    // lockstep (the train block emits nsl_dataloader_reset at every epoch
    // boundary on all ranks), so their epoch counters — and thus permutations
    // — stay identical, keeping the strided shard partition disjoint+complete.
    dl.epoch.fetch_add(1, Ordering::Relaxed);
    dl.start_workers();
}

// ---------------------------------------------------------------------------
// Item 8: resumable-position FFIs
// ---------------------------------------------------------------------------

/// This loader's current epoch counter (the key its permutation is derived
/// from). Recorded in a training checkpoint; `0` for a null handle so the
/// checkpoint writer can pass "no loader" without a branch.
#[no_mangle]
pub extern "C" fn nsl_dataloader_epoch(dl_ptr: i64) -> i64 {
    if dl_ptr == 0 {
        return 0;
    }
    let dl = unsafe { &*(dl_ptr as *const DataLoader) };
    dl.epoch.load(Ordering::Relaxed) as i64
}

/// The NEXT delivery slot this loader would yield.
///
/// This is `expected_batch_id`, deliberately NOT a count of batches handed
/// out: the ragged-packed-tail sentinel advances the slot without yielding a
/// batch, so the two diverge — and only the slot indexes the permutation.
/// Called after `next_batch` returned the batch currently being trained on,
/// it is already the correct resume point.
#[no_mangle]
pub extern "C" fn nsl_dataloader_slot(dl_ptr: i64) -> i64 {
    if dl_ptr == 0 {
        return 0;
    }
    let dl = unsafe { &*(dl_ptr as *const DataLoader) };
    dl.expected_batch_id.load(Ordering::Relaxed) as i64
}

/// Delivery slots this loader yields per epoch. Used by the checkpoint writer
/// to tell "part-way through epoch E" from "epoch E is finished", which are
/// the same `(epoch, slot)` pair apart from this bound.
#[no_mangle]
pub extern "C" fn nsl_dataloader_total_batches(dl_ptr: i64) -> i64 {
    if dl_ptr == 0 {
        return 0;
    }
    let dl = unsafe { &*(dl_ptr as *const DataLoader) };
    dl.total_batches as i64
}

/// Fingerprint of the corpus + loader geometry — see `identity_sig`.
#[no_mangle]
pub extern "C" fn nsl_dataloader_identity(dl_ptr: i64) -> i64 {
    if dl_ptr == 0 {
        return 0;
    }
    let dl = unsafe { &*(dl_ptr as *const DataLoader) };
    dl.identity_sig() as i64
}

/// Arm a checkpoint resume: the next [`nsl_dataloader_reset`] pins `epoch` and
/// starts at delivery `slot`. See `DataLoader::pending_resume`.
#[no_mangle]
pub extern "C" fn nsl_dataloader_resume_to(dl_ptr: i64, epoch: i64, slot: i64) {
    if dl_ptr == 0 {
        return;
    }
    if epoch < 0 || slot < 0 {
        eprintln!(
            "nsl: dataloader_resume_to: negative position (epoch {epoch}, slot \
             {slot}) — refusing rather than wrapping to a huge usize"
        );
        std::process::abort();
    }
    let dl = unsafe { &mut *(dl_ptr as *mut DataLoader) };
    dl.pending_resume = Some((epoch as usize, slot as usize));
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
            nsl_dict_free_tensor_values(dict_ptr);
        }
    }

    if dl.labels_tensor_ptr != 0 {
        crate::tensor::nsl_tensor_free(dl.labels_tensor_ptr);
    }
    crate::tensor::nsl_tensor_free(dl.data_tensor_ptr);

    // DataLoader is dropped here
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::{nsl_dict_free, nsl_dict_len};
    use std::fs;

    fn tensor_from_f64_slice(values: &[f64]) -> i64 {
        let tensor_ptr = create_tensor_with_shape_rs_dtype(&[values.len() as i64], 0);
        let tensor = NslTensor::from_ptr(tensor_ptr);
        let data = tensor.data_f64();
        for (index, value) in values.iter().enumerate() {
            unsafe { *data.add(index) = *value; }
        }
        tensor_ptr
    }

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

    fn make_shuffled_config_json(batch_size: usize, seq_len: usize, seed: u64) -> String {
        format!(
            r#"{{"batch_size":{batch_size},"seq_len":{seq_len},"num_workers":2,"packing":false,"shuffle":true,"prefetch":2,"pin_memory":false,"drop_last":true,"pack_separator":0,"seed":{seed}}}"#
        )
    }

    /// Drain a loader, returning the first token id of every delivered batch —
    /// a cheap fingerprint of WHICH slice of the corpus each batch came from,
    /// and therefore of the permutation.
    fn drain_first_tokens(dl_ptr: i64) -> Vec<i64> {
        let mut out = Vec::new();
        loop {
            let batch = nsl_dataloader_next_batch(dl_ptr);
            if batch == 0 {
                break;
            }
            let k = nsl_str_from_rust("input_ids");
            let tensor_ptr = crate::dict::nsl_dict_get_str(batch, k);
            let tensor = NslTensor::from_ptr(tensor_ptr);
            out.push(unsafe { *(tensor.data as *const i32) } as i64);
            crate::string::nsl_string_free(k);
            nsl_dict_free(batch);
        }
        out
    }

    /// Item 8: the single-rank shuffle must be a function of (seed, epoch) and
    /// nothing else. Before this it was `rand::rng()` — entropy — so "resume
    /// at epoch 3, slot 412" named a position in a permutation that no longer
    /// existed. Two independently constructed loaders on the same corpus must
    /// agree batch for batch, and consecutive epochs must still differ (or the
    /// XOR-by-epoch decorrelation is silently broken).
    #[test]
    fn shuffle_is_reproducible_across_loaders() {
        let data: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let cfg = make_shuffled_config_json(1, 4, 7);

        let mut per_epoch: Vec<Vec<Vec<i64>>> = Vec::new();
        for _run in 0..2 {
            let dl = nsl_dataloader_create(
                tensor_from_f64_slice(&data),
                0,
                cfg.as_ptr() as i64,
                cfg.len() as i64,
            );
            nsl_dataloader_start(dl);
            let mut epochs = Vec::new();
            for _ in 0..3 {
                nsl_dataloader_reset(dl);
                epochs.push(drain_first_tokens(dl));
            }
            nsl_dataloader_stop(dl);
            nsl_dataloader_free(dl);
            per_epoch.push(epochs);
        }

        assert_eq!(
            per_epoch[0], per_epoch[1],
            "two loaders with the same seed must produce identical permutations \
             — an entropy-seeded shuffle makes a checkpointed data position \
             meaningless"
        );
        assert_ne!(
            per_epoch[0][0], per_epoch[0][1],
            "consecutive epochs must differ (seed ^ epoch); identical epochs \
             would mean the epoch is not reaching the shuffle key"
        );
        assert_eq!(per_epoch[0][0].len(), 16, "16 batches of 4 tokens from 64");
    }

    /// Item 8: resuming at delivery slot `k` must yield exactly the batches an
    /// uninterrupted loader would have yielded from `k` on — same permutation,
    /// no repeats, no skips. This is what makes a checkpoint a continuation
    /// rather than a restart.
    #[test]
    fn dataloader_resume_yields_the_exact_tail() {
        let data: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let cfg = make_shuffled_config_json(1, 4, 11);

        // Reference: epoch 2 in full.
        let dl = nsl_dataloader_create(
            tensor_from_f64_slice(&data),
            0,
            cfg.as_ptr() as i64,
            cfg.len() as i64,
        );
        nsl_dataloader_start(dl);
        nsl_dataloader_reset(dl); // epoch 1
        nsl_dataloader_reset(dl); // epoch 2
        let full = drain_first_tokens(dl);
        nsl_dataloader_stop(dl);
        nsl_dataloader_free(dl);
        assert_eq!(full.len(), 16);

        // Resumed: a fresh loader armed for (epoch 2, slot 9).
        const SLOT: usize = 9;
        let dl2 = nsl_dataloader_create(
            tensor_from_f64_slice(&data),
            0,
            cfg.as_ptr() as i64,
            cfg.len() as i64,
        );
        nsl_dataloader_start(dl2);
        nsl_dataloader_resume_to(dl2, 2, SLOT as i64);
        assert_eq!(
            nsl_dataloader_slot(dl2),
            0,
            "arming must not move the position — the pending resume is applied \
             by the next reset, which is where the epoch loop enters"
        );
        nsl_dataloader_reset(dl2);
        assert_eq!(
            nsl_dataloader_epoch(dl2),
            2,
            "the consuming reset must PIN the saved epoch, not increment past it"
        );
        let tail = drain_first_tokens(dl2);

        // And the resume is consumed exactly once: the following epoch must
        // advance and start from slot 0 like any other.
        nsl_dataloader_reset(dl2);
        assert_eq!(nsl_dataloader_epoch(dl2), 3);
        let next_epoch = drain_first_tokens(dl2);
        nsl_dataloader_stop(dl2);
        nsl_dataloader_free(dl2);

        assert_eq!(
            tail,
            full[SLOT..].to_vec(),
            "resumed tail must equal the uninterrupted loader's tail"
        );
        assert_eq!(
            next_epoch.len(),
            16,
            "the epoch after a resume must be complete — a sticky pending \
             resume would truncate every later epoch"
        );
    }

    /// A resume slot past the end of the epoch (checkpointed on the epoch's
    /// last batch) yields an EMPTY epoch rather than replaying the final
    /// batch — the epoch loop then moves on to the next, freshly permuted one.
    #[test]
    fn resume_at_or_past_the_epoch_end_yields_nothing() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let cfg = make_shuffled_config_json(1, 4, 3);
        let dl = nsl_dataloader_create(
            tensor_from_f64_slice(&data),
            0,
            cfg.as_ptr() as i64,
            cfg.len() as i64,
        );
        nsl_dataloader_start(dl);
        nsl_dataloader_resume_to(dl, 1, 4); // 16 tokens / 4 = exactly 4 batches
        nsl_dataloader_reset(dl);
        assert!(
            drain_first_tokens(dl).is_empty(),
            "slot == total_batches must deliver nothing, not clamp back onto \
             the last batch (which would train on it twice)"
        );
        nsl_dataloader_stop(dl);
        nsl_dataloader_free(dl);
    }

    /// `drop_last=false` over a corpus that does not divide evenly keeps a
    /// PADDED final batch (`div_ceil`), so the slot space is one longer than
    /// the whole-batch count. A resume must land on the same slots either way
    /// — the padded tail is a real delivery, not a boundary.
    #[test]
    fn resume_tail_is_exact_with_a_padded_final_batch() {
        // 22 tokens at seq 4 → ceil = 6 slots, the last one padded.
        let data: Vec<f64> = (0..22).map(|i| i as f64).collect();
        let cfg = format!(
            r#"{{"batch_size":1,"seq_len":4,"num_workers":2,"packing":false,"shuffle":true,"prefetch":2,"pin_memory":false,"drop_last":false,"pack_separator":0,"seed":21}}"#
        );

        let full = {
            let dl = nsl_dataloader_create(
                tensor_from_f64_slice(&data),
                0,
                cfg.as_ptr() as i64,
                cfg.len() as i64,
            );
            nsl_dataloader_start(dl);
            nsl_dataloader_reset(dl);
            let v = drain_first_tokens(dl);
            nsl_dataloader_stop(dl);
            nsl_dataloader_free(dl);
            v
        };
        assert_eq!(full.len(), 6, "drop_last=false keeps the padded tail slot");

        const SLOT: usize = 4;
        let dl2 = nsl_dataloader_create(
            tensor_from_f64_slice(&data),
            0,
            cfg.as_ptr() as i64,
            cfg.len() as i64,
        );
        nsl_dataloader_start(dl2);
        nsl_dataloader_resume_to(dl2, 1, SLOT as i64);
        nsl_dataloader_reset(dl2);
        let tail = drain_first_tokens(dl2);
        nsl_dataloader_stop(dl2);
        nsl_dataloader_free(dl2);
        assert_eq!(tail, full[SLOT..].to_vec());
    }

    /// Packing makes delivery slots and delivered batches DIFFER: a ragged
    /// tail inserts a sentinel that advances the slot without yielding a
    /// batch. The resume position is the slot, so it must still land exactly
    /// — this is the case a "count the batches we handed out" cursor gets
    /// wrong.
    #[test]
    fn resume_tail_is_exact_under_packing_with_a_ragged_tail() {
        // Same corpus shape as test_dataloader_packed_ragged_tail_terminates:
        // 4 docs of 5 tokens + 3 strays = 23 tokens, 1x8 → 3 slots, the last
        // of which cannot pack and yields the sentinel.
        let mut data: Vec<f64> = Vec::new();
        for d in 0..4 {
            for t in 0..4 {
                data.push((d * 10 + t + 1) as f64);
            }
            data.push(0.0);
        }
        data.extend([91.0, 92.0, 93.0]);
        let cfg = r#"{"batch_size":1,"seq_len":8,"num_workers":2,"packing":true,"shuffle":false,"prefetch":2,"pin_memory":false,"drop_last":false,"pack_separator":0}"#;

        let mk = || {
            let dl = nsl_dataloader_create(
                tensor_from_f64_slice(&data),
                0,
                cfg.as_ptr() as i64,
                cfg.len() as i64,
            );
            nsl_dataloader_start(dl);
            dl
        };

        let dl = mk();
        nsl_dataloader_reset(dl);
        let full = drain_first_tokens(dl);
        let slots_consumed = nsl_dataloader_slot(dl);
        nsl_dataloader_stop(dl);
        nsl_dataloader_free(dl);
        assert!(
            (slots_consumed as usize) > full.len(),
            "the sentinel must advance the SLOT past the number of delivered \
             batches ({slots_consumed} slots vs {} batches) — otherwise this \
             test is not exercising the case it exists for",
            full.len()
        );

        const SLOT: usize = 1;
        let dl2 = mk();
        nsl_dataloader_resume_to(dl2, 1, SLOT as i64);
        nsl_dataloader_reset(dl2);
        let tail = drain_first_tokens(dl2);
        nsl_dataloader_stop(dl2);
        nsl_dataloader_free(dl2);
        assert_eq!(
            tail,
            full[SLOT..].to_vec(),
            "resuming at a slot must skip exactly the slots before it, \
             sentinel included"
        );
    }

    /// The identity fingerprint must separate corpora and geometries — it is
    /// the guard that stops a resume from continuing "slot 9" of a
    /// permutation over different data.
    #[test]
    fn identity_separates_corpus_and_geometry() {
        let a: Vec<f64> = (0..64).map(|i| i as f64).collect();
        let mut b = a.clone();
        b[17] = 999.0;
        let cfg = make_shuffled_config_json(1, 4, 5);
        let cfg_wide = make_shuffled_config_json(2, 4, 5);
        let cfg_seed = make_shuffled_config_json(1, 4, 6);

        let mk = |data: &[f64], cfg: &str| -> i64 {
            let dl = nsl_dataloader_create(
                tensor_from_f64_slice(data),
                0,
                cfg.as_ptr() as i64,
                cfg.len() as i64,
            );
            let id = nsl_dataloader_identity(dl);
            nsl_dataloader_free(dl);
            id
        };

        let base = mk(&a, &cfg);
        assert_eq!(base, mk(&a, &cfg), "identity must be stable for the same inputs");
        assert_ne!(base, mk(&b, &cfg), "a changed corpus must change the identity");
        assert_ne!(base, mk(&a, &cfg_wide), "batch geometry is part of the identity");
        assert_ne!(base, mk(&a, &cfg_seed), "the shuffle seed is part of the identity");

        // Same tokens, different LABELS: every count matches, so only the
        // label fingerprint separates these two runs.
        let labels_a: Vec<f64> = (0..64).map(|i| (i % 7) as f64).collect();
        let mut labels_b = labels_a.clone();
        labels_b[5] = 42.0;
        let mk_labelled = |labels: &[f64]| -> i64 {
            let dl = nsl_dataloader_create(
                tensor_from_f64_slice(&a),
                tensor_from_f64_slice(labels),
                cfg.as_ptr() as i64,
                cfg.len() as i64,
            );
            let id = nsl_dataloader_identity(dl);
            nsl_dataloader_free(dl);
            id
        };
        assert_ne!(
            mk_labelled(&labels_a),
            mk_labelled(&labels_b),
            "a changed label stream must change the identity — the token \
             stream and every count are identical here"
        );
    }

    /// Review H1: a ragged packed tail must not become a permanent HOLE in
    /// the reorder sequence. The claimed work_index whose pack_batch returns
    /// None now inserts a 0 sentinel the consumer skips — before the fix this
    /// config (packing, non-divisible token count, shuffle-free tail) could
    /// deadlock the consumer against full-gated workers, or silently drop
    /// trailing batches after an early hole under shuffle.
    #[test]
    fn test_dataloader_packed_ragged_tail_terminates() {
        // 4 docs of 5 tokens (incl. separator 0) + 3 stray tokens = 23 tokens.
        // batch 1x8: ceil(23/8) = 3 slots; the tail slot cannot pack a full
        // batch and returns None.
        let mut data: Vec<f64> = Vec::new();
        for d in 0..4 {
            for t in 0..4 {
                data.push((d * 10 + t + 1) as f64);
            }
            data.push(0.0);
        }
        data.extend([91.0, 92.0, 93.0]);
        let data_tensor = tensor_from_f64_slice(&data);
        let config_json = make_config_json(1, 8, 2, true);

        let dl_ptr = nsl_dataloader_create(
            data_tensor,
            0,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
        );
        nsl_dataloader_start(dl_ptr);

        // Must terminate (no consumer/worker deadlock) and yield ≥1 real
        // batch; the ragged tail contributes a skipped sentinel, not a hang.
        let mut real_batches = 0;
        loop {
            let batch = nsl_dataloader_next_batch(dl_ptr);
            if batch == 0 {
                break;
            }
            nsl_dict_free(batch);
            real_batches += 1;
        }
        assert!(real_batches >= 1, "expected at least one packed batch");

        nsl_dataloader_stop(dl_ptr);
        nsl_dataloader_free(dl_ptr);
    }

    #[test]
    fn test_dataloader_basic() {
        // 32 tokens: [0..31]
        let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let data_tensor = tensor_from_f64_slice(&data);
        let config_json = make_config_json(2, 4, 1, false);

        let dl_ptr = nsl_dataloader_create(
            data_tensor,
            0,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
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
        let data_tensor = tensor_from_f64_slice(&data);
        let config_json = make_config_json(1, 4, 2, false);

        let dl_ptr = nsl_dataloader_create(
            data_tensor,
            0,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
        );

        nsl_dataloader_start(dl_ptr);

        // First batch should be tokens [0, 1, 2, 3] regardless of which worker produced it
        let batch = nsl_dataloader_next_batch(dl_ptr);
        assert!(batch != 0);

        let k = nsl_str_from_rust("input_ids");
        let tensor_ptr = crate::dict::nsl_dict_get_str(batch, k);
        let tensor = NslTensor::from_ptr(tensor_ptr);

        // Token IDs are stored as i32 (DTYPE_I32)
        assert_eq!(tensor.dtype, DTYPE_I32, "input_ids should be i32 dtype");
        let ids_data = tensor.data as *const i32;
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

    #[test]
    fn test_dataloader_uses_precomputed_labels() {
        let data: Vec<f64> = vec![10.0, 11.0, 0.0, 0.0];
        let labels: Vec<f64> = vec![11.0, -100.0, -100.0, -100.0];
        let data_tensor = tensor_from_f64_slice(&data);
        let labels_tensor = tensor_from_f64_slice(&labels);
        let config_json = make_config_json(1, 4, 1, false);

        let dl_ptr = nsl_dataloader_create(
            data_tensor,
            labels_tensor,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
        );

        nsl_dataloader_start(dl_ptr);

        let batch = nsl_dataloader_next_batch(dl_ptr);
        assert!(batch != 0);

        let k = nsl_str_from_rust("labels");
        let tensor_ptr = crate::dict::nsl_dict_get_str(batch, k);
        let tensor = NslTensor::from_ptr(tensor_ptr);

        assert_eq!(tensor.dtype, DTYPE_I32, "labels should be i32 dtype");
        let labels_data = tensor.data as *const i32;
        let expected = [11, -100, -100, -100];
        for (i, value) in expected.iter().enumerate() {
            let actual = unsafe { *labels_data.add(i) };
            assert_eq!(actual, *value, "label at position {} should match precomputed label", i);
        }

        nsl_dict_free(batch);
        nsl_dataloader_stop(dl_ptr);
        nsl_dataloader_free(dl_ptr);
    }

    #[test]
    fn test_dataloader_handles_mmap_u16_tokens() {
        let dir = std::env::temp_dir().join("nsl_test_mmap_dataloader");
        let _ = fs::create_dir_all(&dir);
        let path = dir.join("tokens_u16.bin");
        {
            let data: [u16; 5] = [100, 200, 50256, 42, 7];
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    data.as_ptr() as *const u8,
                    data.len() * std::mem::size_of::<u16>(),
                )
            };
            fs::write(&path, bytes).unwrap();
        }

        let path_str = path.to_str().unwrap();
        let data_tensor = crate::data_source::nsl_load_mmap(
            path_str.as_ptr() as i64,
            path_str.len() as i64,
            3,
        );

        let config_json = format!(
            r#"{{"batch_size":1,"seq_len":4,"num_workers":1,"packing":false,"shuffle":false,"prefetch":1,"pin_memory":false,"drop_last":true,"pack_separator":0}}"#
        );

        let dl_ptr = nsl_dataloader_create(
            data_tensor,
            0,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
        );

        nsl_dataloader_start(dl_ptr);

        let batch = nsl_dataloader_next_batch(dl_ptr);
        assert!(batch != 0, "expected one batch from mmap-backed u16 tokens");

        let k_ids = nsl_str_from_rust("input_ids");
        let ids_ptr = crate::dict::nsl_dict_get_str(batch, k_ids);
        let ids_tensor = NslTensor::from_ptr(ids_ptr);
        assert_eq!(ids_tensor.dtype, DTYPE_I32, "input_ids should be materialized as i32 tokens");

        let ids_data = ids_tensor.data as *const i32;
        let expected_ids = [100, 200, 50256, 42];
        for (i, value) in expected_ids.iter().enumerate() {
            let actual = unsafe { *ids_data.add(i) };
            assert_eq!(actual, *value, "token {} should match source data", i);
        }

        let k_lbl = nsl_str_from_rust("labels");
        let lbl_ptr = crate::dict::nsl_dict_get_str(batch, k_lbl);
        let lbl_tensor = NslTensor::from_ptr(lbl_ptr);
        let lbl_data = lbl_tensor.data as *const i32;
        let expected_labels = [200, 50256, 42, -100];
        for (i, value) in expected_labels.iter().enumerate() {
            let actual = unsafe { *lbl_data.add(i) };
            assert_eq!(actual, *value, "label {} should match shifted token data", i);
        }

        nsl_dict_free(batch);
        nsl_dataloader_stop(dl_ptr);
        nsl_dataloader_free(dl_ptr);
    }

    #[test]
    fn test_dataloader_partial_tail_batch() {
        // 10 tokens with batch_size=1, seq_len=4 → 2 full batches + 1 partial (2 tokens, padded)
        let data: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let data_tensor = tensor_from_f64_slice(&data);
        // drop_last=false so the partial tail batch is included
        let config_json = format!(
            r#"{{"batch_size":1,"seq_len":4,"num_workers":1,"packing":false,"shuffle":false,"prefetch":1,"pin_memory":false,"drop_last":false,"pack_separator":0}}"#
        );

        let dl_ptr = nsl_dataloader_create(
            data_tensor,
            0,
            config_json.as_ptr() as i64,
            config_json.len() as i64,
        );

        nsl_dataloader_start(dl_ptr);

        // Batch 1: tokens 0-3
        let batch1 = nsl_dataloader_next_batch(dl_ptr);
        assert!(batch1 != 0, "first batch should exist");
        nsl_dict_free(batch1);

        // Batch 2: tokens 4-7
        let batch2 = nsl_dataloader_next_batch(dl_ptr);
        assert!(batch2 != 0, "second batch should exist");
        nsl_dict_free(batch2);

        // Batch 3: tokens 8-9 + padding
        let batch3 = nsl_dataloader_next_batch(dl_ptr);
        assert!(batch3 != 0, "partial tail batch should exist with drop_last=false");

        // Check that padded positions have 0 for input_ids
        let k_ids = nsl_str_from_rust("input_ids");
        let ids_ptr = crate::dict::nsl_dict_get_str(batch3, k_ids);
        let ids_tensor = NslTensor::from_ptr(ids_ptr);
        let ids_data = ids_tensor.data as *const i32;
        assert_eq!(unsafe { *ids_data.add(0) }, 8, "first token should be 8");
        assert_eq!(unsafe { *ids_data.add(1) }, 9, "second token should be 9");
        assert_eq!(unsafe { *ids_data.add(2) }, 0, "third position should be padded with 0");
        assert_eq!(unsafe { *ids_data.add(3) }, 0, "fourth position should be padded with 0");

        // Check that padded label positions have -100
        let k_lbl = nsl_str_from_rust("labels");
        let lbl_ptr = crate::dict::nsl_dict_get_str(batch3, k_lbl);
        let lbl_tensor = NslTensor::from_ptr(lbl_ptr);
        let lbl_data = lbl_tensor.data as *const i32;
        assert_eq!(unsafe { *lbl_data.add(0) }, 9, "label for position 0 should be next token (9)");
        assert_eq!(unsafe { *lbl_data.add(1) }, -100, "label at position 1 should be -100 (last real token)");
        assert_eq!(unsafe { *lbl_data.add(2) }, -100, "label at padded position should be -100");
        assert_eq!(unsafe { *lbl_data.add(3) }, -100, "label at padded position should be -100");

        nsl_dict_free(batch3);

        // No more batches
        let batch4 = nsl_dataloader_next_batch(dl_ptr);
        assert_eq!(batch4, 0, "should be no more batches after tail");

        nsl_dataloader_stop(dl_ptr);
        nsl_dataloader_free(dl_ptr);
    }
}
