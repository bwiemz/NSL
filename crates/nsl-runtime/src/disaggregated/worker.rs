//! M41: Prefill and decode worker loop implementations.
//!
//! Each worker runs as a separate OS process. The loops receive messages
//! from the router, process requests, and send results back.
//! Workers call the model's forward pass via `nsl_model_forward` when a
//! valid `model_ptr` is provided, falling back to stub behavior otherwise.

use std::sync::Mutex;

use super::kv_transfer::{
    KvTransferHeader, KvBlockTransferEntry, KvTransferBackend,
    SharedMemBackend, KV_TRANSFER_MAGIC,
};
use super::messages::{WorkerRole, RouterMessage, serialize_message, deserialize_message};

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------

static WORKER_CTX: Mutex<Option<WorkerContext>> = Mutex::new(None);

struct WorkerContext {
    role: WorkerRole,
    rank: i32,
    model_ptr: i64,  // opaque model handle from codegen
    kv_cache_handle: i64, // opaque KvCacheManager handle (0 = not set)
}

/// Configuration for worker loops, passed from codegen/router FFI.
#[repr(C)]
#[derive(Clone, Debug)]
pub struct WorkerConfig {
    /// Maximum sequence length before forcing EOS.
    pub max_seq_len: u32,
    /// Number of KV-cache blocks per worker.
    pub kv_blocks_per_worker: u32,
    /// Block size (tokens per block).
    pub block_size: u32,
    /// Number of KV heads.
    pub num_kv_heads: u32,
    /// Head dimension.
    pub head_dim: u32,
    /// Number of transformer layers.
    pub num_layers: u32,
    /// Number of speculative draft tokens reserved per decode iteration.
    pub speculative_tokens: u32,
    /// Speculative method selector: 0=Draft, 1=Medusa, 2=Eagle2, 3=Lookahead.
    pub speculative_method: u32,
    /// Branching factor for tree-based speculative methods.
    pub speculative_tree_width: u32,
    /// `f32::to_bits()` for the configured speculative temperature.
    pub speculative_temperature_bits: u32,
    /// EOS token ID.
    pub eos_token_id: i64,
}

impl Default for WorkerConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 2048,
            kv_blocks_per_worker: 256,
            block_size: 16,
            num_kv_heads: 32,
            head_dim: 128,
            num_layers: 32,
            speculative_tokens: 0,
            speculative_method: 0,
            speculative_tree_width: 1,
            speculative_temperature_bits: 0,
            eos_token_id: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Message mailbox (in-memory for single-node, shared-memory for multi-process)
// ---------------------------------------------------------------------------

static MAILBOX: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Post a message to the worker mailbox (used by router to send work).
pub fn post_message(msg: &RouterMessage) {
    let bytes = serialize_message(msg);
    let mut guard = MAILBOX.lock().unwrap();
    guard.push(bytes);
}

/// Try to receive a message from the mailbox (non-blocking).
/// Returns None if no messages are available.
fn try_recv_message() -> Option<RouterMessage> {
    let mut guard = MAILBOX.lock().unwrap();
    if guard.is_empty() {
        return None;
    }
    let bytes = guard.remove(0);
    deserialize_message(&bytes)
}

/// Wait for a message with a timeout.
/// Returns None on timeout.
fn recv_message_timeout(timeout_ms: u64) -> Option<RouterMessage> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms);
    loop {
        if let Some(msg) = try_recv_message() {
            return Some(msg);
        }
        if std::time::Instant::now() >= deadline {
            return None;
        }
        std::thread::yield_now();
    }
}

/// Response mailbox (workers post results here for the router to collect).
static RESPONSE_MAILBOX: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Post a response message (used by workers to send results to router).
fn post_response(msg: &RouterMessage) {
    let bytes = serialize_message(msg);
    let mut guard = RESPONSE_MAILBOX.lock().unwrap();
    guard.push(bytes);
}

/// Drain all response messages (used by router to collect results).
pub fn drain_responses() -> Vec<RouterMessage> {
    let mut guard = RESPONSE_MAILBOX.lock().unwrap();
    let bytes_list: Vec<Vec<u8>> = guard.drain(..).collect();
    bytes_list.iter().filter_map(|b| deserialize_message(b)).collect()
}

// ---------------------------------------------------------------------------
// FFI: Initialization
// ---------------------------------------------------------------------------

/// Initialize a disaggregated worker (called after role dispatch).
///
/// `role`: 1 = prefill, 2 = decode
/// `rank`: this worker's local rank
/// `model_ptr`: opaque pointer to the compiled model
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_disagg_worker_init(role: i64, rank: i64, model_ptr: i64) -> i64 {
    let worker_role = match role {
        1 => WorkerRole::Prefill,
        2 => WorkerRole::Decode,
        _ => return -1,
    };

    let mut guard = match WORKER_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1, // mutex poisoned
    };
    if guard.is_some() {
        return -1;
    }
    *guard = Some(WorkerContext {
        role: worker_role,
        rank: rank as i32,
        model_ptr,
        kv_cache_handle: 0,
    });
    0
}

/// Set the KV cache handle on the current worker context.
///
/// Must be called after `nsl_disagg_worker_init`. The handle is an opaque
/// pointer returned by `nsl_kv_cache_init` / `nsl_kv_cache_init_gpu`.
/// The prefill worker uses this handle to serialize KV data via
/// `nsl_kv_serialize` before transferring to the decode worker.
///
/// Returns 0 on success, -1 if the worker is not initialized.
#[no_mangle]
pub extern "C" fn nsl_disagg_worker_set_kv_cache(kv_cache_handle: i64) -> i64 {
    let mut guard = match WORKER_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1,
    };
    match guard.as_mut() {
        Some(ctx) => {
            ctx.kv_cache_handle = kv_cache_handle;
            0
        }
        None => -1,
    }
}

// ---------------------------------------------------------------------------
// Model forward helpers
// ---------------------------------------------------------------------------

/// Call the model's forward pass via the C API.
///
/// `model_ptr`: opaque NslModel handle (from `nsl_model_create`)
/// `input_tensor_ptr`: NslTensor* containing the input (e.g., token IDs)
///
/// Returns a Vec of output NslTensor pointers (caller must free them),
/// or an empty Vec if the model has no registered forward function.
fn call_model_forward(model_ptr: i64, input_tensor_ptr: i64) -> Vec<i64> {
    use crate::c_api::NslTensorDesc;
    use crate::tensor::NslTensor;

    if model_ptr == 0 || input_tensor_ptr == 0 {
        return Vec::new();
    }

    // Build an NslTensorDesc from the input tensor
    let tensor = NslTensor::from_ptr(input_tensor_ptr);
    let input_desc = NslTensorDesc {
        data: tensor.data,
        shape: tensor.shape,
        strides: tensor.strides,
        ndim: tensor.ndim as i32,
        dtype: crate::c_api::nsl_dtype_to_capi(tensor.dtype),
        device_type: if tensor.device > 0 { 1 } else { 0 },
        device_id: if tensor.device > 0 { (tensor.device - 1) as i32 } else { 0 },
    };

    // Prepare output descriptor
    let mut output_desc = NslTensorDesc {
        data: std::ptr::null_mut(),
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        ndim: 0,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    };

    let rc = crate::c_api::nsl_model_forward(
        model_ptr,
        &input_desc as *const NslTensorDesc as i64,
        1, // num_inputs
        &mut output_desc as *mut NslTensorDesc as i64,
        1, // num_outputs
    );

    if rc != 0 {
        return Vec::new();
    }

    // If the output desc was populated (non-null data), convert to tensor pointer
    if !output_desc.data.is_null() && output_desc.ndim > 0 {
        let out_ptr = crate::c_api::desc_to_nsl_tensor_pub(&output_desc);
        if out_ptr != 0 {
            return vec![out_ptr];
        }
    }

    Vec::new()
}

/// Create a 1-D NslTensor containing token IDs from a raw pointer.
///
/// `token_ids_ptr`: pointer to an array of i64 token IDs (0 = no tokens available)
/// `num_tokens`: number of tokens
///
/// Returns an NslTensor* (as i64) with shape [num_tokens] and dtype f64 (NSL default).
/// The tensor owns a copy of the data. Returns 0 if token_ids_ptr is null/zero.
fn create_token_tensor(token_ids_ptr: i64, num_tokens: u32) -> i64 {
    use crate::tensor::NslTensor;
    use crate::memory::checked_alloc;

    if token_ids_ptr == 0 || num_tokens == 0 {
        return 0;
    }

    let n = num_tokens as usize;
    let data_bytes = n * std::mem::size_of::<f64>();
    let data_ptr = checked_alloc(data_bytes) as *mut f64;

    // Copy token IDs (i64) into f64 buffer (NSL's default CPU dtype)
    let src = token_ids_ptr as *const i64;
    for i in 0..n {
        unsafe { *data_ptr.add(i) = *src.add(i) as f64; }
    }

    let shape_ptr = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape_ptr = n as i64; }

    let strides_ptr = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides_ptr = 1; }

    let tensor = Box::new(NslTensor::new(
        data_ptr as *mut std::ffi::c_void,
        shape_ptr,
        strides_ptr,
        1,      // ndim
        n as i64,
        0,      // device = CPU
        0,      // dtype = f64
        1,      // owns_data
        0,      // data_owner
    ));
    NslTensor::publish(tensor)
}

/// Sample a single token ID from a logits tensor via argmax.
///
/// `logits_ptr`: NslTensor* with shape [..., vocab_size] (last dim is vocab)
///
/// Returns the token ID with the highest logit value.
fn sample_argmax(logits_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;

    if logits_ptr == 0 {
        return 0;
    }

    let tensor = NslTensor::from_ptr(logits_ptr);
    let len = tensor.len as usize;
    if len == 0 {
        return 0;
    }

    // Read logits as f64 or f32 depending on dtype
    let mut best_idx: usize = 0;
    let mut best_val: f64 = f64::NEG_INFINITY;

    if tensor.dtype == 0 {
        // f64
        let data = unsafe { std::slice::from_raw_parts(tensor.data as *const f64, len) };
        // Argmax over the last dimension (last `vocab_size` elements)
        let vocab_size = if tensor.ndim > 0 {
            (unsafe { *tensor.shape.add(tensor.ndim as usize - 1) }) as usize
        } else {
            len
        };
        let start = len.saturating_sub(vocab_size);
        for (i, &v) in data[start..].iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
    } else {
        // f32
        let data = unsafe { std::slice::from_raw_parts(tensor.data as *const f32, len) };
        let vocab_size = if tensor.ndim > 0 {
            (unsafe { *tensor.shape.add(tensor.ndim as usize - 1) }) as usize
        } else {
            len
        };
        let start = len.saturating_sub(vocab_size);
        for (i, &v) in data[start..].iter().enumerate() {
            if (v as f64) > best_val {
                best_val = v as f64;
                best_idx = i;
            }
        }
    }

    best_idx as i64
}

// ---------------------------------------------------------------------------
// Prefill worker loop
// ---------------------------------------------------------------------------

/// Run the prefill worker loop.
///
/// This is the main entry point for a prefill worker process.
/// The loop:
/// 1. Receives StartPrefill messages from the router
/// 2. Runs the model's forward pass on the prompt tokens (stub: fills KV with sentinel)
/// 3. Serializes the KV-cache and transfers to the assigned decode worker
/// 4. Sends PrefillComplete back to the router
///
/// `config_ptr`: pointer to WorkerConfig (0 = use defaults)
///
/// Returns 0 on clean shutdown, negative on error.
#[no_mangle]
pub extern "C" fn nsl_disagg_prefill_loop(config_ptr: i64) -> i64 {
    // Verify role and extract context
    let (rank, model_ptr, kv_cache_handle) = {
        let guard = match WORKER_CTX.lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        let ctx = match guard.as_ref() {
            Some(c) => c,
            None => return -2,
        };
        if ctx.role != WorkerRole::Prefill {
            return -3;
        }
        (ctx.rank, ctx.model_ptr, ctx.kv_cache_handle)
    };

    let config = if config_ptr != 0 {
        unsafe { &*(config_ptr as *const WorkerConfig) }.clone()
    } else {
        WorkerConfig::default()
    };

    // Create a local KV transfer backend for sending to decode workers
    let kv_backend = SharedMemBackend::new(rank);

    // Prefill processing loop — runs until Shutdown or timeout with no work
    while let Some(msg) = recv_message_timeout(100) {
        match msg {
            RouterMessage::StartPrefill {
                request_id,
                token_ids_ptr,
                num_tokens,
                target_decode_rank,
            } => {
                // Step 1: Run model forward pass on prompt tokens.
                // When model_ptr is valid, call the registered forward function
                // which populates the KV cache as a side effect.
                if model_ptr != 0 {
                    let input_tensor = create_token_tensor(token_ids_ptr, num_tokens);
                    if input_tensor != 0 {
                        let outputs = call_model_forward(model_ptr, input_tensor);
                        // Free output logits — prefill only needs KV cache side effect
                        for out_ptr in &outputs {
                            if *out_ptr != 0 {
                                crate::tensor::nsl_tensor_free(*out_ptr);
                            }
                        }
                        crate::tensor::nsl_tensor_free(input_tensor);
                    }
                }

                let num_blocks = (num_tokens as u64).div_ceil(config.block_size as u64);

                // Step 2: Build KV transfer header and block entries
                let mut header = KvTransferHeader {
                    magic: KV_TRANSFER_MAGIC,
                    request_id,
                    num_layers: config.num_layers,
                    num_kv_heads: config.num_kv_heads,
                    head_dim: config.head_dim,
                    block_size: config.block_size,
                    num_blocks: num_blocks as u32,
                    dtype: 1, // f32
                    compressed: 0,
                    _padding: 0,
                    total_bytes: 0,
                };
                header.total_bytes = header.compute_kv_bytes();

                let mut entries = Vec::with_capacity(num_blocks as usize);
                for i in 0..num_blocks as u32 {
                    let valid_tokens = if i == num_blocks as u32 - 1 {
                        let remainder = num_tokens % config.block_size;
                        if remainder == 0 { config.block_size } else { remainder }
                    } else {
                        config.block_size
                    };
                    entries.push(KvBlockTransferEntry {
                        logical_block_id: i,
                        num_valid_tokens: valid_tokens,
                    });
                }

                // Step 3: Transfer KV cache to decode worker.
                // Allocate buffers and serialize KV data from the block allocator
                // when a KV cache handle is available. Falls back to empty buffers
                // (header-only transfer) when the handle is not set.
                // Step 3: Transfer KV cache to decode worker.
                // Buffers must outlive the send_kv call — declared here so they
                // stay alive through the call and drop naturally afterwards.
                let total_bytes = header.total_bytes as usize;
                let mut k_buf: Vec<u8> = vec![0u8; total_bytes];
                let mut v_buf: Vec<u8> = vec![0u8; total_bytes];

                if total_bytes > 0 && kv_cache_handle != 0 {
                    let _rc = super::kv_transfer::nsl_kv_serialize(
                        kv_cache_handle,
                        request_id as i64,
                        request_id as i64,
                        &mut header as *mut KvTransferHeader as i64,
                        k_buf.as_mut_ptr() as i64,
                        v_buf.as_mut_ptr() as i64,
                    );
                }

                let k_data_ptr = if total_bytes > 0 { k_buf.as_ptr() as *const std::ffi::c_void } else { std::ptr::null() };
                let v_data_ptr = if total_bytes > 0 { v_buf.as_ptr() as *const std::ffi::c_void } else { std::ptr::null() };

                let _rc = kv_backend.send_kv(
                    target_decode_rank,
                    &header,
                    &entries,
                    k_data_ptr,
                    v_data_ptr,
                );
                // k_buf and v_buf drop here — no leak

                // Step 4: Notify router that prefill is complete
                post_response(&RouterMessage::PrefillComplete {
                    request_id,
                    num_kv_blocks: num_blocks as u32,
                });
            }
            _ => {
                // Unexpected message type — ignore
            }
        }
    }

    0
}

// ---------------------------------------------------------------------------
// Decode worker loop
// ---------------------------------------------------------------------------

/// Run the decode worker loop.
///
/// This is the main entry point for a decode worker process.
/// The loop:
/// 1. Checks for incoming KV transfers (non-blocking)
/// 2. Receives StartDecode messages from router
/// 3. Runs autoregressive decode steps
/// 4. Streams generated tokens back to the router
/// 5. Cleans up completed sequences
///
/// `config_ptr`: pointer to WorkerConfig (0 = use defaults)
///
/// Returns 0 on clean shutdown, negative on error.
#[no_mangle]
pub extern "C" fn nsl_disagg_decode_loop(config_ptr: i64) -> i64 {
    // Verify role
    let (rank, model_ptr) = {
        let guard = match WORKER_CTX.lock() {
            Ok(g) => g,
            Err(_) => return -1,
        };
        let ctx = match guard.as_ref() {
            Some(c) => c,
            None => return -2,
        };
        if ctx.role != WorkerRole::Decode {
            return -3;
        }
        (ctx.rank, ctx.model_ptr)
    };

    let config = if config_ptr != 0 {
        unsafe { &*(config_ptr as *const WorkerConfig) }.clone()
    } else {
        WorkerConfig::default()
    };

    // Structured speculative config is now threaded into the worker loop via
    // WorkerConfig instead of the old non-null sentinel-pointer hack.
    let _speculative_tokens = config.speculative_tokens;
    let _speculative_method = config.speculative_method;
    let _speculative_tree_width = config.speculative_tree_width;
    let _speculative_temperature = f32::from_bits(config.speculative_temperature_bits);

    // Active sequences being decoded:
    // request_id -> (prompt_len, tokens_generated, max_tokens, last_token_id)
    let mut active_sequences: std::collections::HashMap<u64, (u32, u32, u32, i64)> =
        std::collections::HashMap::new();

    // KV transfer backend for receiving from prefill workers
    let _kv_backend = SharedMemBackend::new(rank);

    // Decode processing loop
    loop {
        // Step 1: Check for StartDecode messages from router
        let msg = match recv_message_timeout(100) {
            Some(m) => m,
            None => {
                // No new messages — run one decode step for all active sequences
                if active_sequences.is_empty() {
                    break; // Nothing to do, exit for test/single-iteration mode
                }
                // Process one decode step per active sequence
                let mut completed = Vec::new();
                for (&request_id, (prompt_len, tokens_generated, max_tokens, last_token_id)) in active_sequences.iter_mut() {
                    if *tokens_generated >= *max_tokens
                        || prompt_len.saturating_add(*tokens_generated) >= config.max_seq_len
                    {
                        post_response(&RouterMessage::DecodeComplete {
                            request_id,
                            total_tokens: *tokens_generated,
                        });
                        completed.push(request_id);
                        continue;
                    }

                    // Step 2: Run model decode step.
                    // When model_ptr is valid, pass the last token through the model
                    // and sample from the output logits.
                    let token_id = if model_ptr != 0 {
                        // Create a 1-element tensor with the last generated token
                        let single_token = [*last_token_id];
                        let input_tensor = create_token_tensor(
                            single_token.as_ptr() as i64,
                            1,
                        );
                        if input_tensor != 0 {
                            let outputs = call_model_forward(model_ptr, input_tensor);
                            crate::tensor::nsl_tensor_free(input_tensor);
                            if let Some(&logits_ptr) = outputs.first() {
                                let sampled = sample_argmax(logits_ptr);
                                for out_ptr in &outputs {
                                    if *out_ptr != 0 {
                                        crate::tensor::nsl_tensor_free(*out_ptr);
                                    }
                                }
                                sampled
                            } else {
                                // Forward returned no outputs — fall back to stub
                                *tokens_generated as i64 + 100
                            }
                        } else {
                            *tokens_generated as i64 + 100
                        }
                    } else {
                        // Stub: generate sequential token IDs when no model is loaded
                        *tokens_generated as i64 + 100
                    };

                    *last_token_id = token_id;
                    *tokens_generated += 1;

                    let total_seq_len = prompt_len.saturating_add(*tokens_generated);
                    let is_eos = token_id == config.eos_token_id
                        || *tokens_generated >= *max_tokens
                        || total_seq_len >= config.max_seq_len;

                    // Step 3: Stream token to router
                    post_response(&RouterMessage::TokenGenerated {
                        request_id,
                        token_id,
                        is_eos: if is_eos { 1 } else { 0 },
                    });

                    if is_eos {
                        // Step 4: Mark complete and clean up
                        post_response(&RouterMessage::DecodeComplete {
                            request_id,
                            total_tokens: *tokens_generated,
                        });
                        completed.push(request_id);
                    }
                }
                for id in completed {
                    active_sequences.remove(&id);
                }
                continue;
            }
        };

        match msg {
            RouterMessage::StartDecode {
                request_id,
                prompt_len,
                max_tokens,
                temperature_bits: _,
                top_p_bits: _,
            } => {
                // Admit new sequence into active set.
                // last_token_id starts at 0 (BOS); prefill would have set this
                // via the KV transfer but for stub mode we use 0.
                active_sequences.insert(request_id, (prompt_len, 0, max_tokens, 0));
            }
            _ => {
                // Unexpected message — ignore
            }
        }
    }

    0
}

/// Destroy the worker context.
#[no_mangle]
pub extern "C" fn nsl_disagg_worker_destroy() -> i64 {
    let mut guard = match WORKER_CTX.lock() {
        Ok(g) => g,
        Err(_) => return -1, // mutex poisoned
    };
    *guard = None;
    // Also clear mailboxes
    if let Ok(mut mb) = MAILBOX.lock() {
        mb.clear();
    }
    if let Ok(mut rb) = RESPONSE_MAILBOX.lock() {
        rb.clear();
    }
    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_disagg_worker_destroy();
        guard
    }

    #[test]
    fn worker_init_prefill() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);
        // Prefill loop exits after timeout with no messages
        assert_eq!(nsl_disagg_prefill_loop(0), 0);
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_init_decode() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(2, 0, 0), 0);
        // Decode loop exits after timeout with no active sequences
        assert_eq!(nsl_disagg_decode_loop(0), 0);
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_double_init_fails() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);
        assert_eq!(nsl_disagg_worker_init(2, 0, 0), -1);
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_invalid_role_fails() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(99, 0, 0), -1);
    }

    #[test]
    fn worker_config_struct_carries_speculative_fields() {
        let _lock = setup();
        let cfg = WorkerConfig {
            max_seq_len: 1024,
            kv_blocks_per_worker: 96,
            block_size: 16,
            num_kv_heads: 32,
            head_dim: 128,
            num_layers: 32,
            speculative_tokens: 4,
            speculative_method: 2,
            speculative_tree_width: 3,
            speculative_temperature_bits: 0.75f32.to_bits(),
            eos_token_id: 2,
        };

        assert_eq!(cfg.max_seq_len, 1024);
        assert_eq!(cfg.kv_blocks_per_worker, 96);
        assert_eq!(cfg.speculative_tokens, 4);
        assert_eq!(cfg.speculative_method, 2);
        assert_eq!(cfg.speculative_tree_width, 3);
        assert_eq!(cfg.speculative_temperature_bits, 0.75f32.to_bits());
    }

    #[test]
    fn prefill_processes_start_prefill_message() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);

        // Post a StartPrefill message
        post_message(&RouterMessage::StartPrefill {
            request_id: 42,
            token_ids_ptr: 0,
            num_tokens: 128,
            target_decode_rank: 1,
        });

        // Run prefill loop (processes message, then exits on timeout)
        assert_eq!(nsl_disagg_prefill_loop(0), 0);

        // Check that PrefillComplete was posted
        let responses = drain_responses();
        assert!(!responses.is_empty(), "prefill should post PrefillComplete");
        match &responses[0] {
            RouterMessage::PrefillComplete { request_id, num_kv_blocks } => {
                assert_eq!(*request_id, 42);
                // 128 tokens / 16 block_size = 8 blocks
                assert_eq!(*num_kv_blocks, 8);
            }
            other => panic!("expected PrefillComplete, got {:?}", other),
        }

        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn prefill_loop_respects_custom_worker_config() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);

        post_message(&RouterMessage::StartPrefill {
            request_id: 77,
            token_ids_ptr: 0,
            num_tokens: 17,
            target_decode_rank: 1,
        });

        let cfg = WorkerConfig {
            max_seq_len: 2048,
            kv_blocks_per_worker: 64,
            block_size: 8,
            num_kv_heads: 32,
            head_dim: 128,
            num_layers: 32,
            speculative_tokens: 0,
            speculative_method: 0,
            speculative_tree_width: 1,
            speculative_temperature_bits: 0,
            eos_token_id: 2,
        };
        assert_eq!(nsl_disagg_prefill_loop(&cfg as *const WorkerConfig as i64), 0);

        let responses = drain_responses();
        let complete = responses.iter().find(|m| matches!(m, RouterMessage::PrefillComplete { .. }));
        assert!(complete.is_some(), "prefill should complete request 77");
        if let Some(RouterMessage::PrefillComplete { request_id, num_kv_blocks }) = complete {
            assert_eq!(*request_id, 77);
            assert_eq!(*num_kv_blocks, 3, "17 tokens with block_size=8 should use 3 KV blocks");
        }

        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn decode_processes_start_decode_and_generates_tokens() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(2, 1, 0), 0);

        // Post StartDecode with max_tokens=3
        post_message(&RouterMessage::StartDecode {
            request_id: 99,
            prompt_len: 10,
            max_tokens: 3,
            temperature_bits: 0.7f64.to_bits(),
            top_p_bits: 0.9f64.to_bits(),
        });

        // Run decode loop
        assert_eq!(nsl_disagg_decode_loop(0), 0);

        // Check generated tokens
        let responses = drain_responses();
        // Should have 3 TokenGenerated + 1 DecodeComplete = 4 messages
        let token_msgs: Vec<_> = responses.iter()
            .filter(|m| matches!(m, RouterMessage::TokenGenerated { .. }))
            .collect();
        let complete_msgs: Vec<_> = responses.iter()
            .filter(|m| matches!(m, RouterMessage::DecodeComplete { .. }))
            .collect();

        assert_eq!(token_msgs.len(), 3, "should generate 3 tokens");
        assert_eq!(complete_msgs.len(), 1, "should send 1 DecodeComplete");

        // Last token should be EOS
        if let RouterMessage::TokenGenerated { is_eos, .. } = token_msgs[2] {
            assert_eq!(*is_eos, 1, "last token should be EOS");
        }

        // DecodeComplete should have total_tokens=3
        if let RouterMessage::DecodeComplete { request_id, total_tokens } = complete_msgs[0] {
            assert_eq!(*request_id, 99);
            assert_eq!(*total_tokens, 3);
        }

        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn decode_loop_respects_custom_worker_config() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(2, 1, 0), 0);

        post_message(&RouterMessage::StartDecode {
            request_id: 7,
            prompt_len: 10,
            max_tokens: 3,
            temperature_bits: 0.7f64.to_bits(),
            top_p_bits: 0.9f64.to_bits(),
        });

        let cfg = WorkerConfig {
            max_seq_len: 1,
            kv_blocks_per_worker: 64,
            block_size: 16,
            num_kv_heads: 32,
            head_dim: 128,
            num_layers: 32,
            speculative_tokens: 2,
            speculative_method: 1,
            speculative_tree_width: 4,
            speculative_temperature_bits: 0.25f32.to_bits(),
            eos_token_id: 2,
        };
        assert_eq!(nsl_disagg_decode_loop(&cfg as *const WorkerConfig as i64), 0);

        let responses = drain_responses();
        let token_msgs: Vec<_> = responses.iter()
            .filter(|m| matches!(m, RouterMessage::TokenGenerated { .. }))
            .collect();
        let complete_msgs: Vec<_> = responses.iter()
            .filter(|m| matches!(m, RouterMessage::DecodeComplete { .. }))
            .collect();

        assert!(token_msgs.is_empty(), "prompt_len already exceeds max_seq_len, so decode should emit no tokens");
        assert_eq!(complete_msgs.len(), 1, "custom max_seq_len should still complete the request");
        if let RouterMessage::DecodeComplete { request_id, total_tokens } = complete_msgs[0] {
            assert_eq!(*request_id, 7);
            assert_eq!(*total_tokens, 0);
        }

        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn decode_loop_respects_zero_max_tokens() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(2, 1, 0), 0);

        post_message(&RouterMessage::StartDecode {
            request_id: 8,
            prompt_len: 0,
            max_tokens: 0,
            temperature_bits: 1.0f64.to_bits(),
            top_p_bits: 1.0f64.to_bits(),
        });

        assert_eq!(nsl_disagg_decode_loop(0), 0);

        let responses = drain_responses();
        let token_msgs: Vec<_> = responses
            .iter()
            .filter(|m| matches!(m, RouterMessage::TokenGenerated { .. }))
            .collect();
        let complete_msgs: Vec<_> = responses
            .iter()
            .filter(|m| matches!(m, RouterMessage::DecodeComplete { .. }))
            .collect();

        assert!(token_msgs.is_empty(), "max_tokens=0 should emit no tokens");
        assert_eq!(complete_msgs.len(), 1, "max_tokens=0 should complete immediately");
        if let RouterMessage::DecodeComplete { request_id, total_tokens } = complete_msgs[0] {
            assert_eq!(*request_id, 8);
            assert_eq!(*total_tokens, 0);
        }

        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn decode_loop_stops_exactly_at_sequence_length_boundary() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(2, 1, 0), 0);

        post_message(&RouterMessage::StartDecode {
            request_id: 9,
            prompt_len: 2,
            max_tokens: 5,
            temperature_bits: 1.0f64.to_bits(),
            top_p_bits: 1.0f64.to_bits(),
        });

        let cfg = WorkerConfig {
            max_seq_len: 3,
            kv_blocks_per_worker: 64,
            block_size: 16,
            num_kv_heads: 32,
            head_dim: 128,
            num_layers: 32,
            speculative_tokens: 0,
            speculative_method: 0,
            speculative_tree_width: 1,
            speculative_temperature_bits: 0,
            eos_token_id: 2,
        };
        assert_eq!(nsl_disagg_decode_loop(&cfg as *const WorkerConfig as i64), 0);

        let responses = drain_responses();
        let token_msgs: Vec<_> = responses
            .iter()
            .filter(|m| matches!(m, RouterMessage::TokenGenerated { .. }))
            .collect();
        let complete_msgs: Vec<_> = responses
            .iter()
            .filter(|m| matches!(m, RouterMessage::DecodeComplete { .. }))
            .collect();

        assert_eq!(token_msgs.len(), 1, "one final token should be emitted when it lands exactly on max_seq_len");
        assert_eq!(complete_msgs.len(), 1, "request should complete at the exact sequence-length boundary");
        if let RouterMessage::TokenGenerated { is_eos, .. } = token_msgs[0] {
            assert_eq!(*is_eos, 1, "the boundary token should terminate the sequence");
        }
        if let RouterMessage::DecodeComplete { request_id, total_tokens } = complete_msgs[0] {
            assert_eq!(*request_id, 9);
            assert_eq!(*total_tokens, 1);
        }

        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn prefill_then_decode_roundtrip() {
        let _lock = setup();

        // === Prefill phase ===
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);
        post_message(&RouterMessage::StartPrefill {
            request_id: 1,
            token_ids_ptr: 0,
            num_tokens: 32,
            target_decode_rank: 1,
        });
        assert_eq!(nsl_disagg_prefill_loop(0), 0);
        let prefill_responses = drain_responses();
        assert!(
            prefill_responses.iter().any(|m| matches!(m, RouterMessage::PrefillComplete { request_id: 1, .. })),
            "prefill should complete request 1"
        );
        assert_eq!(nsl_disagg_worker_destroy(), 0);

        // === Decode phase ===
        assert_eq!(nsl_disagg_worker_init(2, 1, 0), 0);
        post_message(&RouterMessage::StartDecode {
            request_id: 1,
            prompt_len: 32,
            max_tokens: 5,
            temperature_bits: 1.0f64.to_bits(),
            top_p_bits: 1.0f64.to_bits(),
        });
        assert_eq!(nsl_disagg_decode_loop(0), 0);
        let decode_responses = drain_responses();

        let complete = decode_responses.iter().find(|m| matches!(m, RouterMessage::DecodeComplete { .. }));
        assert!(complete.is_some(), "decode should complete");
        if let Some(RouterMessage::DecodeComplete { request_id, total_tokens }) = complete {
            assert_eq!(*request_id, 1);
            assert_eq!(*total_tokens, 5);
        }

        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_wrong_role_rejected() {
        let _lock = setup();
        // Init as prefill, try to run decode loop
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);
        assert_eq!(nsl_disagg_decode_loop(0), -3); // wrong role
        assert_eq!(nsl_disagg_worker_destroy(), 0);

        // Init as decode, try to run prefill loop
        assert_eq!(nsl_disagg_worker_init(2, 0, 0), 0);
        assert_eq!(nsl_disagg_prefill_loop(0), -3); // wrong role
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn multiple_prefill_requests() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(1, 0, 0), 0);

        // Post 3 prefill requests
        for i in 0..3 {
            post_message(&RouterMessage::StartPrefill {
                request_id: i as u64,
                token_ids_ptr: 0,
                num_tokens: 64,
                target_decode_rank: 1,
            });
        }

        assert_eq!(nsl_disagg_prefill_loop(0), 0);
        let responses = drain_responses();
        let completes: Vec<_> = responses.iter()
            .filter(|m| matches!(m, RouterMessage::PrefillComplete { .. }))
            .collect();
        assert_eq!(completes.len(), 3, "should complete all 3 prefill requests");
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }
}
