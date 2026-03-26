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
}

/// Configuration for worker loops, passed from the router.
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
    });
    0
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
        if ctx.role != WorkerRole::Prefill {
            return -3;
        }
        (ctx.rank, ctx.model_ptr)
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
                // When model_ptr != 0 the forward pass has populated KV cache
                // memory. For now we still send the header + entries (the
                // SharedMemBackend will copy any non-null K/V data). A future
                // change will wire nsl_kv_serialize to extract actual block
                // data from the BlockAllocator.
                let _rc = kv_backend.send_kv(
                    target_decode_rank,
                    &header,
                    &entries,
                    std::ptr::null(), // TODO: wire to BlockAllocator K data
                    std::ptr::null(), // TODO: wire to BlockAllocator V data
                );

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

    // Active sequences being decoded: request_id -> (tokens_generated, max_tokens, last_token_id)
    let mut active_sequences: std::collections::HashMap<u64, (u32, u32, i64)> =
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
                for (&request_id, (tokens_generated, max_tokens, last_token_id)) in active_sequences.iter_mut() {
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

                    let is_eos = token_id == config.eos_token_id
                        || *tokens_generated >= *max_tokens
                        || *tokens_generated >= config.max_seq_len;

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
                prompt_len: _,
                max_tokens,
                temperature_bits: _,
                top_p_bits: _,
            } => {
                // Admit new sequence into active set.
                // last_token_id starts at 0 (BOS); prefill would have set this
                // via the KV transfer but for stub mode we use 0.
                active_sequences.insert(request_id, (0, max_tokens, 0));
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
