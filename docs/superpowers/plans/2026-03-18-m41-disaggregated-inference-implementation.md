# M41: Disaggregated Inference — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the compute-bound prefill phase from the memory-bound decode phase onto different worker processes with KV-cache transfer between them, extending the M29 serve block with disaggregated serving.

**Architecture:** Five new runtime modules under `crates/nsl-runtime/src/disaggregated/` (messages, KV transfer, router, prefill/decode worker loops) + codegen extension in `serve.rs` for role-based dispatch + CLI flags for `nsl run --prefill-workers N --decode-workers M`. All processes run the same compiled binary; role selection happens at runtime via `NSL_ROLE` environment variable. SharedMemBackend for testing; NvLink/TCP stubs for future multi-node.

**Tech Stack:** Rust (runtime FFI + codegen + CLI)

**Spec:** `docs/superpowers/specs/2026-03-15-m41-disaggregated-inference-design.md`

**Prerequisites:** M29 (serve block), M30 (tensor parallelism SPMD model), M33 (speculative decoding), M25 (paged KV-cache)

---

## Important: Scope of This Plan

**This plan builds the complete disaggregated inference infrastructure.** It delivers:
- `RouterMessage` protocol for inter-process communication
- `KvTransferBackend` trait with `SharedMemBackend` implementation
- `KvTransferHeader` / `KvBlockTransferEntry` serialization format
- KV serialize/deserialize FFI functions
- `DisaggregatedRouter` with request lifecycle tracking and scheduling
- `prefill_worker_loop()` and `decode_worker_loop()` FFI functions
- Serve block codegen: detect disaggregation config → emit role dispatch
- CLI: `--prefill-workers`, `--decode-workers` flags on `nsl run`
- Semantic: validation of disaggregated serve config
- 20+ unit tests across all modules

**Deferred to M41b:** NvLink backend (requires `cuMemcpyPeer`), RDMA backend (requires ibverbs), TCP backend, multi-node disaggregation, automatic profiling-based worker split, decode-to-prefill feedback, KV-cache prefix sharing across workers, live migration of sequences between decode workers, asymmetric hardware declaration (`prefill_devices`/`decode_devices` config params), `@speculative` integration with disaggregated workers (draft on decode, verify on prefill), `ShortestQueue` prefill scheduling policy (prompt-length-weighted), `"auto"` backend resolution (probe NVLink → fallback to shared_mem).

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/disaggregated/mod.rs` | Module declarations | 10 |
| `crates/nsl-runtime/src/disaggregated/messages.rs` | `RouterMessage` enum, `WorkerRole`, serialization | 120 |
| `crates/nsl-runtime/src/disaggregated/kv_transfer.rs` | `KvTransferBackend` trait, `KvTransferHeader`, `SharedMemBackend`, serialize/deserialize | 280 |
| `crates/nsl-runtime/src/disaggregated/router.rs` | `DisaggregatedRouter`, scheduling policies, request lifecycle, FFI | 250 |
| `crates/nsl-runtime/src/disaggregated/worker.rs` | `prefill_worker_loop`, `decode_worker_loop`, FFI wrappers | 200 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod disaggregated;` |
| `crates/nsl-codegen/src/serve.rs` | Disaggregated config extraction, role dispatch codegen |
| `crates/nsl-codegen/src/compiler.rs` | Add `disaggregated: bool`, `prefill_workers: usize`, `decode_workers: usize` fields |
| `crates/nsl-semantic/src/checker.rs` | `check_disaggregated_serve()` validation |
| `crates/nsl-cli/src/main.rs` | `--prefill-workers`, `--decode-workers` CLI flags on `Run` |

---

## Phase 1: Message Protocol + KV Transfer Infrastructure

### Task 1: Message Protocol

**Files:**
- Create: `crates/nsl-runtime/src/disaggregated/mod.rs`
- Create: `crates/nsl-runtime/src/disaggregated/messages.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Create `disaggregated/mod.rs` and `messages.rs` with message types**

```rust
// crates/nsl-runtime/src/disaggregated/mod.rs
pub mod messages;
pub mod kv_transfer;
pub mod router;
pub mod worker;

// crates/nsl-runtime/src/disaggregated/messages.rs
//! M41: Inter-process message protocol for disaggregated inference.

/// Worker role — determined by NSL_ROLE environment variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum WorkerRole {
    Router = 0,
    Prefill = 1,
    Decode = 2,
}

impl WorkerRole {
    pub fn from_env() -> Self {
        match std::env::var("NSL_ROLE").as_deref() {
            Ok("router") => WorkerRole::Router,
            Ok("prefill") => WorkerRole::Prefill,
            Ok("decode") => WorkerRole::Decode,
            _ => WorkerRole::Router, // default: monolithic (router-only)
        }
    }

    pub fn local_rank() -> i32 {
        std::env::var("NSL_LOCAL_RANK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    }
}

/// Messages exchanged between router and workers via shared memory.
///
/// Layout: `[u32 tag][payload bytes]` — tag identifies the variant,
/// payload is the struct fields in order.
#[derive(Clone, Debug)]
pub enum RouterMessage {
    /// Router → Prefill: start processing this request's prompt.
    StartPrefill {
        request_id: u64,
        token_ids_ptr: i64,    // pointer to token array (in shared memory)
        num_tokens: u32,
        target_decode_rank: i32,
    },

    /// Prefill → Router: prefill complete, KV transferred to decode worker.
    PrefillComplete {
        request_id: u64,
        num_kv_blocks: u32,
    },

    /// Router → Decode: KV pages incoming, begin autoregressive generation.
    StartDecode {
        request_id: u64,
        prompt_len: u32,
        max_tokens: u32,
        temperature_bits: u64,  // f64 transmuted to u64 for C repr
        top_p_bits: u64,
    },

    /// Decode → Router: a token was generated.
    TokenGenerated {
        request_id: u64,
        token_id: i64,
        is_eos: u8,  // 0 = no, 1 = yes
    },

    /// Decode → Router: sequence complete, all resources freed.
    DecodeComplete {
        request_id: u64,
        total_tokens: u32,
    },
}

/// Request lifecycle state tracked by the router.
#[derive(Clone, Debug, PartialEq)]
pub enum DisaggRequestState {
    Queued,
    Prefilling { prefill_rank: i32 },
    Transferring { from_rank: i32, to_rank: i32 },
    Decoding { decode_rank: i32, tokens_generated: u32 },
    Complete { total_tokens: u32 },
}

/// Per-worker metadata tracked by the router.
pub struct WorkerHandle {
    pub rank: i32,
    pub role: WorkerRole,
    pub active_requests: u32,
    pub free_kv_blocks: u32,
}

impl WorkerHandle {
    pub fn new(rank: i32, role: WorkerRole, initial_kv_blocks: u32) -> Self {
        WorkerHandle {
            rank,
            role,
            active_requests: 0,
            free_kv_blocks: initial_kv_blocks,
        }
    }
}

// Serialization helpers for shared-memory message passing.

/// Serialize a RouterMessage into a byte buffer.
/// Format: [u32 tag][fields in repr(C) order]
pub fn serialize_message(msg: &RouterMessage) -> Vec<u8> {
    let mut buf = Vec::with_capacity(64);
    match msg {
        RouterMessage::StartPrefill { request_id, token_ids_ptr, num_tokens, target_decode_rank } => {
            buf.extend_from_slice(&0u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&token_ids_ptr.to_le_bytes());
            buf.extend_from_slice(&num_tokens.to_le_bytes());
            buf.extend_from_slice(&target_decode_rank.to_le_bytes());
        }
        RouterMessage::PrefillComplete { request_id, num_kv_blocks } => {
            buf.extend_from_slice(&1u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&num_kv_blocks.to_le_bytes());
        }
        RouterMessage::StartDecode { request_id, prompt_len, max_tokens, temperature_bits, top_p_bits } => {
            buf.extend_from_slice(&2u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&prompt_len.to_le_bytes());
            buf.extend_from_slice(&max_tokens.to_le_bytes());
            buf.extend_from_slice(&temperature_bits.to_le_bytes());
            buf.extend_from_slice(&top_p_bits.to_le_bytes());
        }
        RouterMessage::TokenGenerated { request_id, token_id, is_eos } => {
            buf.extend_from_slice(&3u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&token_id.to_le_bytes());
            buf.extend_from_slice(&[*is_eos, 0, 0, 0]); // pad to 4 bytes
        }
        RouterMessage::DecodeComplete { request_id, total_tokens } => {
            buf.extend_from_slice(&4u32.to_le_bytes());
            buf.extend_from_slice(&request_id.to_le_bytes());
            buf.extend_from_slice(&total_tokens.to_le_bytes());
        }
    }
    buf
}

/// Deserialize a RouterMessage from a byte slice.
/// Returns None if the buffer is too short or tag is unknown.
pub fn deserialize_message(buf: &[u8]) -> Option<RouterMessage> {
    if buf.len() < 4 {
        return None;
    }
    let tag = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let payload = &buf[4..];

    match tag {
        0 if payload.len() >= 22 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let token_ids_ptr = i64::from_le_bytes(payload[8..16].try_into().ok()?);
            let num_tokens = u32::from_le_bytes(payload[16..20].try_into().ok()?);
            let target_decode_rank = i32::from_le_bytes(payload[20..24].try_into().ok()?);
            Some(RouterMessage::StartPrefill { request_id, token_ids_ptr, num_tokens, target_decode_rank })
        }
        1 if payload.len() >= 12 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let num_kv_blocks = u32::from_le_bytes(payload[8..12].try_into().ok()?);
            Some(RouterMessage::PrefillComplete { request_id, num_kv_blocks })
        }
        2 if payload.len() >= 32 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let prompt_len = u32::from_le_bytes(payload[8..12].try_into().ok()?);
            let max_tokens = u32::from_le_bytes(payload[12..16].try_into().ok()?);
            let temperature_bits = u64::from_le_bytes(payload[16..24].try_into().ok()?);
            let top_p_bits = u64::from_le_bytes(payload[24..32].try_into().ok()?);
            Some(RouterMessage::StartDecode { request_id, prompt_len, max_tokens, temperature_bits, top_p_bits })
        }
        3 if payload.len() >= 17 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let token_id = i64::from_le_bytes(payload[8..16].try_into().ok()?);
            let is_eos = payload[16];
            Some(RouterMessage::TokenGenerated { request_id, token_id, is_eos })
        }
        4 if payload.len() >= 12 => {
            let request_id = u64::from_le_bytes(payload[0..8].try_into().ok()?);
            let total_tokens = u32::from_le_bytes(payload[8..12].try_into().ok()?);
            Some(RouterMessage::DecodeComplete { request_id, total_tokens })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worker_role_defaults() {
        // When NSL_ROLE is not set, default is Router
        std::env::remove_var("NSL_ROLE");
        assert_eq!(WorkerRole::from_env(), WorkerRole::Router);
    }

    #[test]
    fn message_roundtrip_start_prefill() {
        let msg = RouterMessage::StartPrefill {
            request_id: 42,
            token_ids_ptr: 0x1234_5678,
            num_tokens: 128,
            target_decode_rank: 3,
        };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::StartPrefill { request_id, token_ids_ptr, num_tokens, target_decode_rank } => {
                assert_eq!(request_id, 42);
                assert_eq!(token_ids_ptr, 0x1234_5678);
                assert_eq!(num_tokens, 128);
                assert_eq!(target_decode_rank, 3);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_prefill_complete() {
        let msg = RouterMessage::PrefillComplete { request_id: 7, num_kv_blocks: 16 };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::PrefillComplete { request_id, num_kv_blocks } => {
                assert_eq!(request_id, 7);
                assert_eq!(num_kv_blocks, 16);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_start_decode() {
        let temp: f64 = 0.7;
        let top_p: f64 = 0.9;
        let msg = RouterMessage::StartDecode {
            request_id: 99,
            prompt_len: 512,
            max_tokens: 256,
            temperature_bits: temp.to_bits(),
            top_p_bits: top_p.to_bits(),
        };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::StartDecode { request_id, prompt_len, max_tokens, temperature_bits, top_p_bits } => {
                assert_eq!(request_id, 99);
                assert_eq!(prompt_len, 512);
                assert_eq!(max_tokens, 256);
                assert_eq!(f64::from_bits(temperature_bits), 0.7);
                assert_eq!(f64::from_bits(top_p_bits), 0.9);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_token_generated() {
        let msg = RouterMessage::TokenGenerated { request_id: 5, token_id: 42, is_eos: 1 };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::TokenGenerated { request_id, token_id, is_eos } => {
                assert_eq!(request_id, 5);
                assert_eq!(token_id, 42);
                assert_eq!(is_eos, 1);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn message_roundtrip_decode_complete() {
        let msg = RouterMessage::DecodeComplete { request_id: 10, total_tokens: 200 };
        let bytes = serialize_message(&msg);
        let decoded = deserialize_message(&bytes).unwrap();
        match decoded {
            RouterMessage::DecodeComplete { request_id, total_tokens } => {
                assert_eq!(request_id, 10);
                assert_eq!(total_tokens, 200);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn deserialize_short_buffer_returns_none() {
        assert!(deserialize_message(&[]).is_none());
        assert!(deserialize_message(&[0, 0, 0, 0]).is_none()); // tag 0 but no payload
    }

    #[test]
    fn deserialize_unknown_tag_returns_none() {
        let mut buf = 99u32.to_le_bytes().to_vec();
        buf.extend_from_slice(&[0u8; 32]);
        assert!(deserialize_message(&buf).is_none());
    }

    #[test]
    fn worker_handle_construction() {
        let w = WorkerHandle::new(2, WorkerRole::Decode, 1024);
        assert_eq!(w.rank, 2);
        assert_eq!(w.role, WorkerRole::Decode);
        assert_eq!(w.active_requests, 0);
        assert_eq!(w.free_kv_blocks, 1024);
    }

    #[test]
    fn disagg_request_state_transitions() {
        let mut state = DisaggRequestState::Queued;
        assert_eq!(state, DisaggRequestState::Queued);

        state = DisaggRequestState::Prefilling { prefill_rank: 0 };
        assert!(matches!(state, DisaggRequestState::Prefilling { .. }));

        state = DisaggRequestState::Transferring { from_rank: 0, to_rank: 2 };
        assert!(matches!(state, DisaggRequestState::Transferring { .. }));

        state = DisaggRequestState::Decoding { decode_rank: 2, tokens_generated: 0 };
        assert!(matches!(state, DisaggRequestState::Decoding { .. }));

        state = DisaggRequestState::Complete { total_tokens: 50 };
        assert_eq!(state, DisaggRequestState::Complete { total_tokens: 50 });
    }
}
```

- [ ] **Step 2: Wire module into `lib.rs`**

Add to `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod disaggregated;
```

---

### Task 2: KV Transfer Infrastructure

**Files:**
- Create: `crates/nsl-runtime/src/disaggregated/kv_transfer.rs`

- [ ] **Step 3: Create `kv_transfer.rs` with transfer header, backend trait, SharedMemBackend, and serialize/deserialize FFI**

```rust
// crates/nsl-runtime/src/disaggregated/kv_transfer.rs
//! M41: KV-cache transfer protocol for disaggregated inference.
//!
//! Defines the serialization format (KvTransferHeader + block entries + K/V data),
//! the KvTransferBackend trait, and a SharedMemBackend for single-node testing.

use std::ffi::c_void;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Transfer header (matches spec Section 3)
// ---------------------------------------------------------------------------

/// Magic bytes: "KVXF" = 0x4B56_5846
pub const KV_TRANSFER_MAGIC: u32 = 0x4B56_5846;

/// Header for a KV-cache transfer message between prefill and decode workers.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct KvTransferHeader {
    pub magic: u32,
    pub request_id: u64,
    pub num_layers: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,      // tokens per block
    pub num_blocks: u32,      // total blocks being transferred
    pub dtype: u16,           // 0=f64, 1=f32, 2=fp16, etc.
    pub compressed: u8,       // 0 = raw, 1 = quantized (M42 future)
    pub _padding: u8,
    pub total_bytes: u64,     // total payload size after header + entries
}

impl KvTransferHeader {
    /// Compute total KV data bytes for this transfer.
    ///
    /// Layout per block: K[num_layers * num_kv_heads * block_size * head_dim] + same for V.
    pub fn compute_kv_bytes(&self) -> u64 {
        let elements_per_block = self.num_layers as u64
            * self.num_kv_heads as u64
            * self.block_size as u64
            * self.head_dim as u64;
        let dtype_bytes = dtype_size(self.dtype) as u64;
        // K + V = 2x
        2 * self.num_blocks as u64 * elements_per_block * dtype_bytes
    }

    /// Size of the block entry metadata array.
    pub fn entries_bytes(&self) -> usize {
        self.num_blocks as usize * std::mem::size_of::<KvBlockTransferEntry>()
    }

    /// Validate the header magic.
    pub fn is_valid(&self) -> bool {
        self.magic == KV_TRANSFER_MAGIC
    }
}

/// Per-block metadata in the transfer.
#[derive(Clone, Debug)]
#[repr(C)]
pub struct KvBlockTransferEntry {
    pub logical_block_id: u32,    // position in sequence
    pub num_valid_tokens: u32,    // how many tokens are valid (last block may be partial)
}

/// Return the byte size for a given dtype code.
fn dtype_size(dtype: u16) -> usize {
    match dtype {
        0 => 8,  // f64
        1 => 4,  // f32
        2 => 2,  // fp16
        3 => 2,  // bf16
        4 => 1,  // int8
        5 => 1,  // fp8e4m3
        6 => 1,  // fp8e5m2
        _ => 4,  // default to f32 for unknown
    }
}

// ---------------------------------------------------------------------------
// Transfer backend trait
// ---------------------------------------------------------------------------

/// Backend for transferring KV-cache pages between workers.
///
/// Implementations: SharedMemBackend (testing), NvlinkBackend (future),
/// RdmaBackend (future), TcpBackend (future).
pub trait KvTransferBackend: Send + Sync {
    /// Send KV pages from this worker to the target worker.
    ///
    /// The caller provides the serialized header, block entries, and K/V data.
    /// Returns 0 on success, negative on error.
    fn send_kv(
        &self,
        target_rank: i32,
        header: &KvTransferHeader,
        block_entries: &[KvBlockTransferEntry],
        k_data: *const c_void,
        v_data: *const c_void,
    ) -> i32;

    /// Receive KV pages from a source worker (blocking).
    ///
    /// On return, header/block_entries/k_data/v_data are filled.
    /// Returns 0 on success, negative on error.
    fn recv_kv(
        &self,
        source_rank: i32,
        header: &mut KvTransferHeader,
        block_entries: &mut Vec<KvBlockTransferEntry>,
        k_data: *mut c_void,
        v_data: *mut c_void,
    ) -> i32;

    /// Try to receive KV pages without blocking.
    ///
    /// Returns 0 if data was received, 1 if no data available, negative on error.
    fn try_recv_kv(
        &self,
        header: &mut KvTransferHeader,
        block_entries: &mut Vec<KvBlockTransferEntry>,
        k_data: *mut c_void,
        v_data: *mut c_void,
    ) -> i32;

    /// Estimated transfer time in microseconds for given byte count.
    fn estimate_transfer_us(&self, bytes: u64) -> u64;
}

// ---------------------------------------------------------------------------
// SharedMemBackend (testing + single-node)
// ---------------------------------------------------------------------------

/// Shared-memory KV transfer backend using file-backed mmap.
///
/// Reuses the M30 pattern: a single mmap'd file partitioned into per-pair
/// ring buffers. For testing, uses an in-memory Vec instead of mmap.
pub struct SharedMemBackend {
    rank: i32,
    /// In-memory transfer buffers indexed by (source, target) pair.
    /// Each entry: Option<(KvTransferHeader, Vec<KvBlockTransferEntry>, Vec<u8>, Vec<u8>)>
    ///                                                                   K data    V data
    buffers: std::sync::Arc<Mutex<Vec<PendingTransfer>>>,
}

struct PendingTransfer {
    source_rank: i32,
    target_rank: i32,
    header: KvTransferHeader,
    block_entries: Vec<KvBlockTransferEntry>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
}

impl SharedMemBackend {
    pub fn new(rank: i32) -> Self {
        SharedMemBackend {
            rank,
            buffers: std::sync::Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Create a pair of backends that share the same buffer pool.
    pub fn new_pair(rank_a: i32, rank_b: i32) -> (Self, Self) {
        let shared = std::sync::Arc::new(Mutex::new(Vec::new()));
        (
            SharedMemBackend { rank: rank_a, buffers: shared.clone() },
            SharedMemBackend { rank: rank_b, buffers: shared },
        )
    }
}

impl KvTransferBackend for SharedMemBackend {
    fn send_kv(
        &self,
        target_rank: i32,
        header: &KvTransferHeader,
        block_entries: &[KvBlockTransferEntry],
        k_data: *const c_void,
        v_data: *const c_void,
    ) -> i32 {
        let kv_bytes = header.compute_kv_bytes() as usize / 2; // per K or V
        let k_slice = if kv_bytes > 0 && !k_data.is_null() {
            unsafe { std::slice::from_raw_parts(k_data as *const u8, kv_bytes) }.to_vec()
        } else {
            Vec::new()
        };
        let v_slice = if kv_bytes > 0 && !v_data.is_null() {
            unsafe { std::slice::from_raw_parts(v_data as *const u8, kv_bytes) }.to_vec()
        } else {
            Vec::new()
        };

        let transfer = PendingTransfer {
            source_rank: self.rank,
            target_rank,
            header: header.clone(),
            block_entries: block_entries.to_vec(),
            k_data: k_slice,
            v_data: v_slice,
        };

        let mut guard = self.buffers.lock().unwrap();
        guard.push(transfer);
        0
    }

    fn recv_kv(
        &self,
        source_rank: i32,
        header: &mut KvTransferHeader,
        block_entries: &mut Vec<KvBlockTransferEntry>,
        k_data: *mut c_void,
        v_data: *mut c_void,
    ) -> i32 {
        // Bounded wait with timeout (default 5s = drain_timeout_ms).
        // Returns -2 on timeout to distinguish from other errors.
        let deadline = std::time::Instant::now() + std::time::Duration::from_millis(5000);
        loop {
            let mut guard = self.buffers.lock().unwrap();
            if let Some(pos) = guard.iter().position(|t| t.source_rank == source_rank && t.target_rank == self.rank) {
                let transfer = guard.remove(pos);
                *header = transfer.header;
                *block_entries = transfer.block_entries;
                if !k_data.is_null() && !transfer.k_data.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            transfer.k_data.as_ptr(),
                            k_data as *mut u8,
                            transfer.k_data.len(),
                        );
                    }
                }
                if !v_data.is_null() && !transfer.v_data.is_empty() {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            transfer.v_data.as_ptr(),
                            v_data as *mut u8,
                            transfer.v_data.len(),
                        );
                    }
                }
                return 0;
            }
            drop(guard);
            if std::time::Instant::now() >= deadline {
                return -2; // timeout
            }
            std::thread::yield_now();
        }
    }

    fn try_recv_kv(
        &self,
        header: &mut KvTransferHeader,
        block_entries: &mut Vec<KvBlockTransferEntry>,
        k_data: *mut c_void,
        v_data: *mut c_void,
    ) -> i32 {
        let mut guard = self.buffers.lock().unwrap();
        if let Some(pos) = guard.iter().position(|t| t.target_rank == self.rank) {
            let transfer = guard.remove(pos);
            *header = transfer.header;
            *block_entries = transfer.block_entries;
            if !k_data.is_null() && !transfer.k_data.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        transfer.k_data.as_ptr(),
                        k_data as *mut u8,
                        transfer.k_data.len(),
                    );
                }
            }
            if !v_data.is_null() && !transfer.v_data.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        transfer.v_data.as_ptr(),
                        v_data as *mut u8,
                        transfer.v_data.len(),
                    );
                }
            }
            return 0;
        }
        1 // no data available
    }

    fn estimate_transfer_us(&self, bytes: u64) -> u64 {
        // Shared memory: ~10 GB/s effective → ~100 ns/byte → bytes / 10 μs
        bytes / 10_000
    }
}

// ---------------------------------------------------------------------------
// Global context + FFI
// ---------------------------------------------------------------------------

static KV_TRANSFER_CTX: Mutex<Option<KvTransferContext>> = Mutex::new(None);

struct KvTransferContext {
    backend: Box<dyn KvTransferBackend>,
}

/// Initialize the KV transfer subsystem.
///
/// `backend_id`: 0 = SharedMem, 1 = NVLink (stub), 2 = RDMA (stub), 3 = TCP (stub)
/// `rank`: this worker's rank
///
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_kv_transfer_init(backend_id: i64, rank: i64) -> i64 {
    let mut guard = KV_TRANSFER_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    let backend: Box<dyn KvTransferBackend> = match backend_id {
        0 => Box::new(SharedMemBackend::new(rank as i32)),
        // Future: 1 => NvlinkBackend, 2 => RdmaBackend, 3 => TcpBackend
        _ => Box::new(SharedMemBackend::new(rank as i32)),
    };
    *guard = Some(KvTransferContext { backend });
    0
}

/// Send KV data to a target rank.
///
/// Parameters match the spec: header_ptr points to a KvTransferHeader,
/// entries_ptr to KvBlockTransferEntry array, k_data_ptr/v_data_ptr to KV tensors.
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_kv_transfer_send(
    target_rank: i64,
    header_ptr: i64,
    entries_ptr: i64,
    k_data_ptr: i64,
    v_data_ptr: i64,
) -> i64 {
    let guard = KV_TRANSFER_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_kv_transfer_init not called");

    let header = unsafe { &*(header_ptr as *const KvTransferHeader) };
    let entries = if entries_ptr != 0 && header.num_blocks > 0 {
        unsafe {
            std::slice::from_raw_parts(
                entries_ptr as *const KvBlockTransferEntry,
                header.num_blocks as usize,
            )
        }
    } else {
        &[]
    };

    ctx.backend.send_kv(
        target_rank as i32,
        header,
        entries,
        k_data_ptr as *const c_void,
        v_data_ptr as *const c_void,
    ) as i64
}

/// Receive KV data from a source rank (blocking).
///
/// `header_out_ptr` must point to a pre-allocated KvTransferHeader.
/// `entries_out_ptr` receives pointer to allocated entries array.
/// `k_data_out_ptr` / `v_data_out_ptr` must point to pre-allocated buffers.
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_kv_transfer_recv(
    source_rank: i64,
    header_out_ptr: i64,
    k_data_out_ptr: i64,
    v_data_out_ptr: i64,
) -> i64 {
    let guard = KV_TRANSFER_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_kv_transfer_init not called");

    let header = unsafe { &mut *(header_out_ptr as *mut KvTransferHeader) };
    let mut entries = Vec::new();

    let rc = ctx.backend.recv_kv(
        source_rank as i32,
        header,
        &mut entries,
        k_data_out_ptr as *mut c_void,
        v_data_out_ptr as *mut c_void,
    );
    rc as i64
}

/// Destroy the KV transfer context.
#[no_mangle]
pub extern "C" fn nsl_kv_transfer_destroy() -> i64 {
    let mut guard = KV_TRANSFER_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// KV serialization helpers
// ---------------------------------------------------------------------------

/// Serialize a sequence's KV-cache pages into the transfer format.
///
/// `kv_cache_handle`: opaque handle from nsl_kv_cache_init
/// `seq_id`: sequence to serialize
/// `header_out`: pointer to KvTransferHeader to fill
/// `k_data_out` / `v_data_out`: pointers to pre-allocated buffers
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_kv_serialize(
    _kv_cache_handle: i64,
    _seq_id: i64,
    _request_id: i64,
    header_out: i64,
    _k_data_out: i64,
    _v_data_out: i64,
) -> i64 {
    // Write a valid header with the request_id.
    // In a full implementation, this reads from the KvCacheManager's block pool.
    // For now, fill the header with metadata and return success.
    if header_out == 0 {
        return -1;
    }
    let header = unsafe { &mut *(header_out as *mut KvTransferHeader) };
    header.magic = KV_TRANSFER_MAGIC;
    header.request_id = _request_id as u64;
    // Other fields would be filled from the KvCacheManager's config.
    // Currently a stub — full integration with paged_kv in M41b.
    0
}

/// Deserialize received KV-cache pages into the local KV-cache.
///
/// `kv_cache_handle`: opaque handle from nsl_kv_cache_init
/// `header`: pointer to received KvTransferHeader
/// `k_data` / `v_data`: pointers to received data buffers
///
/// Returns the allocated seq_id on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_kv_deserialize(
    _kv_cache_handle: i64,
    header: i64,
    _k_data: i64,
    _v_data: i64,
) -> i64 {
    if header == 0 {
        return -1;
    }
    let h = unsafe { &*(header as *const KvTransferHeader) };
    if !h.is_valid() {
        return -1;
    }
    // In a full implementation, this allocates blocks in the local KvCacheManager
    // and copies the received K/V data into them.
    // Returns the new seq_id.
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_magic_validation() {
        let mut h = KvTransferHeader {
            magic: KV_TRANSFER_MAGIC,
            request_id: 1,
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            block_size: 16,
            num_blocks: 4,
            dtype: 1, // f32
            compressed: 0,
            _padding: 0,
            total_bytes: 0,
        };
        assert!(h.is_valid());
        h.magic = 0;
        assert!(!h.is_valid());
    }

    #[test]
    fn header_compute_kv_bytes() {
        let h = KvTransferHeader {
            magic: KV_TRANSFER_MAGIC,
            request_id: 1,
            num_layers: 2,
            num_kv_heads: 4,
            head_dim: 64,
            block_size: 16,
            num_blocks: 3,
            dtype: 1, // f32 = 4 bytes
            compressed: 0,
            _padding: 0,
            total_bytes: 0,
        };
        // per block: 2 * 4 * 16 * 64 = 8192 elements
        // 3 blocks * 8192 * 4 bytes = 98304 per K or V
        // Total = 2 * 98304 = 196608
        let expected = 2 * 3 * 2 * 4 * 16 * 64 * 4u64;
        assert_eq!(h.compute_kv_bytes(), expected);
    }

    #[test]
    fn shared_mem_backend_send_recv_roundtrip() {
        let (sender, receiver) = SharedMemBackend::new_pair(0, 1);

        let header = KvTransferHeader {
            magic: KV_TRANSFER_MAGIC,
            request_id: 42,
            num_layers: 1,
            num_kv_heads: 1,
            head_dim: 4,
            block_size: 2,
            num_blocks: 1,
            dtype: 1, // f32
            compressed: 0,
            _padding: 0,
            total_bytes: 0,
        };

        let entries = vec![KvBlockTransferEntry { logical_block_id: 0, num_valid_tokens: 2 }];

        // K data: 1 * 1 * 2 * 4 = 8 f32 elements = 32 bytes
        let k_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];

        // Send from rank 0 to rank 1
        let rc = sender.send_kv(
            1,
            &header,
            &entries,
            k_data.as_ptr() as *const c_void,
            v_data.as_ptr() as *const c_void,
        );
        assert_eq!(rc, 0);

        // Receive on rank 1 from rank 0
        let mut recv_header = KvTransferHeader {
            magic: 0, request_id: 0, num_layers: 0, num_kv_heads: 0,
            head_dim: 0, block_size: 0, num_blocks: 0, dtype: 0,
            compressed: 0, _padding: 0, total_bytes: 0,
        };
        let mut recv_entries = Vec::new();
        let mut recv_k: Vec<f32> = vec![0.0; 8];
        let mut recv_v: Vec<f32> = vec![0.0; 8];

        let rc = receiver.recv_kv(
            0,
            &mut recv_header,
            &mut recv_entries,
            recv_k.as_mut_ptr() as *mut c_void,
            recv_v.as_mut_ptr() as *mut c_void,
        );
        assert_eq!(rc, 0);
        assert!(recv_header.is_valid());
        assert_eq!(recv_header.request_id, 42);
        assert_eq!(recv_entries.len(), 1);
        assert_eq!(recv_entries[0].logical_block_id, 0);
        assert_eq!(recv_k, k_data);
        assert_eq!(recv_v, v_data);
    }

    #[test]
    fn shared_mem_backend_try_recv_empty() {
        let backend = SharedMemBackend::new(0);
        let mut header = KvTransferHeader {
            magic: 0, request_id: 0, num_layers: 0, num_kv_heads: 0,
            head_dim: 0, block_size: 0, num_blocks: 0, dtype: 0,
            compressed: 0, _padding: 0, total_bytes: 0,
        };
        let mut entries = Vec::new();
        let rc = backend.try_recv_kv(
            &mut header,
            &mut entries,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        );
        assert_eq!(rc, 1); // no data available
    }

    #[test]
    fn dtype_size_table() {
        assert_eq!(dtype_size(0), 8);  // f64
        assert_eq!(dtype_size(1), 4);  // f32
        assert_eq!(dtype_size(2), 2);  // fp16
        assert_eq!(dtype_size(3), 2);  // bf16
        assert_eq!(dtype_size(4), 1);  // int8
        assert_eq!(dtype_size(5), 1);  // fp8e4m3
        assert_eq!(dtype_size(6), 1);  // fp8e5m2
        assert_eq!(dtype_size(99), 4); // unknown → default f32
    }

    /// FFI tests — serialized to avoid global state races.
    static FFI_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn ffi_init_destroy_lifecycle() {
        let _lock = FFI_LOCK.lock().unwrap();
        nsl_kv_transfer_destroy(); // clean slate

        assert_eq!(nsl_kv_transfer_init(0, 0), 0);
        assert_eq!(nsl_kv_transfer_init(0, 0), -1); // double init
        assert_eq!(nsl_kv_transfer_destroy(), 0);
    }

    #[test]
    fn ffi_serialize_writes_magic() {
        let _lock = FFI_LOCK.lock().unwrap();
        nsl_kv_transfer_destroy();

        let mut header = KvTransferHeader {
            magic: 0, request_id: 0, num_layers: 0, num_kv_heads: 0,
            head_dim: 0, block_size: 0, num_blocks: 0, dtype: 0,
            compressed: 0, _padding: 0, total_bytes: 0,
        };
        let rc = nsl_kv_serialize(0, 0, 99, &mut header as *mut _ as i64, 0, 0);
        assert_eq!(rc, 0);
        assert_eq!(header.magic, KV_TRANSFER_MAGIC);
        assert_eq!(header.request_id, 99);
    }

    #[test]
    fn ffi_deserialize_validates_magic() {
        let _lock = FFI_LOCK.lock().unwrap();

        // Invalid header
        let mut header = KvTransferHeader {
            magic: 0, request_id: 0, num_layers: 0, num_kv_heads: 0,
            head_dim: 0, block_size: 0, num_blocks: 0, dtype: 0,
            compressed: 0, _padding: 0, total_bytes: 0,
        };
        assert_eq!(nsl_kv_deserialize(0, &header as *const _ as i64, 0, 0), -1);

        // Valid header
        header.magic = KV_TRANSFER_MAGIC;
        assert_eq!(nsl_kv_deserialize(0, &header as *const _ as i64, 0, 0), 0);
    }
}
```

---

## Phase 2: Router + Worker Loops

### Task 3: Disaggregated Router

**Files:**
- Create: `crates/nsl-runtime/src/disaggregated/router.rs`

- [ ] **Step 4: Create `router.rs` with DisaggregatedRouter, scheduling policies, and FFI**

```rust
// crates/nsl-runtime/src/disaggregated/router.rs
//! M41: Disaggregated inference router.
//!
//! The router process accepts requests, dispatches prefill to prefill workers,
//! routes KV-transfer completion to decode workers, and streams tokens back.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use super::messages::{DisaggRequestState, WorkerHandle, WorkerRole};

// ---------------------------------------------------------------------------
// Scheduling policies
// ---------------------------------------------------------------------------

/// How to select a prefill worker for a new request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrefillPolicy {
    /// Route to the prefill worker with fewest active requests.
    LeastLoaded,
    /// Simple round-robin rotation.
    RoundRobin,
}

/// How to select a decode worker after prefill completes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodePolicy {
    /// Route to the decode worker with fewest active sequences.
    LeastLoaded,
    /// Route to the decode worker with the most free KV-cache blocks.
    MemoryAware,
}

// ---------------------------------------------------------------------------
// Router configuration
// ---------------------------------------------------------------------------

pub struct DisaggregatedConfig {
    pub num_prefill_workers: usize,
    pub num_decode_workers: usize,
    pub prefill_policy: PrefillPolicy,
    pub decode_policy: DecodePolicy,
    pub max_batch_per_decode: usize,
    pub max_seq_len: usize,
    pub kv_blocks_per_worker: u32,
    pub drain_timeout_ms: u64,
}

impl Default for DisaggregatedConfig {
    fn default() -> Self {
        DisaggregatedConfig {
            num_prefill_workers: 1,
            num_decode_workers: 1,
            prefill_policy: PrefillPolicy::LeastLoaded,
            decode_policy: DecodePolicy::LeastLoaded,
            max_batch_per_decode: 32,
            max_seq_len: 4096,
            kv_blocks_per_worker: 2048,
            drain_timeout_ms: 5000,
        }
    }
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub struct DisaggregatedRouter {
    config: DisaggregatedConfig,
    prefill_pool: Vec<WorkerHandle>,
    decode_pool: Vec<WorkerHandle>,
    requests: HashMap<u64, DisaggRequestState>,
    next_request_id: AtomicU64,
    round_robin_prefill: usize,
}

impl DisaggregatedRouter {
    pub fn new(config: DisaggregatedConfig) -> Self {
        let mut prefill_pool = Vec::new();
        for i in 0..config.num_prefill_workers {
            prefill_pool.push(WorkerHandle::new(i as i32, WorkerRole::Prefill, 0));
        }

        let mut decode_pool = Vec::new();
        for i in 0..config.num_decode_workers {
            decode_pool.push(WorkerHandle::new(
                i as i32,
                WorkerRole::Decode,
                config.kv_blocks_per_worker,
            ));
        }

        DisaggregatedRouter {
            config,
            prefill_pool,
            decode_pool,
            requests: HashMap::new(),
            next_request_id: AtomicU64::new(0),
            round_robin_prefill: 0,
        }
    }

    /// Enqueue a new request. Returns the request ID.
    pub fn enqueue_request(&mut self) -> u64 {
        let id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        self.requests.insert(id, DisaggRequestState::Queued);
        id
    }

    /// Select a prefill worker for a queued request.
    pub fn select_prefill_worker(&mut self) -> Option<i32> {
        if self.prefill_pool.is_empty() {
            return None;
        }
        match self.config.prefill_policy {
            PrefillPolicy::LeastLoaded => {
                self.prefill_pool
                    .iter()
                    .min_by_key(|w| w.active_requests)
                    .map(|w| w.rank)
            }
            PrefillPolicy::RoundRobin => {
                let idx = self.round_robin_prefill % self.prefill_pool.len();
                self.round_robin_prefill += 1;
                Some(self.prefill_pool[idx].rank)
            }
        }
    }

    /// Select a decode worker for a completed prefill.
    pub fn select_decode_worker(&self, _kv_blocks_needed: u32) -> Option<i32> {
        if self.decode_pool.is_empty() {
            return None;
        }
        match self.config.decode_policy {
            DecodePolicy::LeastLoaded => {
                self.decode_pool
                    .iter()
                    .min_by_key(|w| w.active_requests)
                    .map(|w| w.rank)
            }
            DecodePolicy::MemoryAware => {
                self.decode_pool
                    .iter()
                    .max_by_key(|w| w.free_kv_blocks)
                    .map(|w| w.rank)
            }
        }
    }

    /// Mark a request as dispatched to a prefill worker.
    pub fn mark_prefilling(&mut self, request_id: u64, prefill_rank: i32) {
        if let Some(worker) = self.prefill_pool.iter_mut().find(|w| w.rank == prefill_rank) {
            worker.active_requests += 1;
        }
        self.requests.insert(request_id, DisaggRequestState::Prefilling { prefill_rank });
    }

    /// Mark a request as transferring KV from prefill to decode.
    pub fn mark_transferring(&mut self, request_id: u64, from_rank: i32, to_rank: i32) {
        if let Some(worker) = self.prefill_pool.iter_mut().find(|w| w.rank == from_rank) {
            worker.active_requests = worker.active_requests.saturating_sub(1);
        }
        self.requests.insert(request_id, DisaggRequestState::Transferring { from_rank, to_rank });
    }

    /// Mark a request as decoding on a decode worker, reserving KV blocks.
    pub fn mark_decoding(&mut self, request_id: u64, decode_rank: i32, kv_blocks_used: u32) {
        if let Some(worker) = self.decode_pool.iter_mut().find(|w| w.rank == decode_rank) {
            worker.active_requests += 1;
            worker.free_kv_blocks = worker.free_kv_blocks.saturating_sub(kv_blocks_used);
        }
        self.requests.insert(request_id, DisaggRequestState::Decoding {
            decode_rank,
            tokens_generated: 0,
        });
    }

    /// Record a generated token for a request.
    pub fn record_token(&mut self, request_id: u64) {
        if let Some(DisaggRequestState::Decoding { tokens_generated, .. }) = self.requests.get_mut(&request_id) {
            *tokens_generated += 1;
        }
    }

    /// Mark a request as complete.
    pub fn mark_complete(&mut self, request_id: u64, total_tokens: u32) {
        if let Some(state) = self.requests.get(&request_id) {
            if let DisaggRequestState::Decoding { decode_rank, .. } = state {
                let decode_rank = *decode_rank;
                if let Some(worker) = self.decode_pool.iter_mut().find(|w| w.rank == decode_rank) {
                    worker.active_requests = worker.active_requests.saturating_sub(1);
                }
            }
        }
        self.requests.insert(request_id, DisaggRequestState::Complete { total_tokens });
    }

    /// Get the current state of a request.
    pub fn request_state(&self, request_id: u64) -> Option<&DisaggRequestState> {
        self.requests.get(&request_id)
    }

    /// Number of active (non-complete, non-queued) requests.
    pub fn active_request_count(&self) -> usize {
        self.requests.values().filter(|s| {
            matches!(s,
                DisaggRequestState::Prefilling { .. }
                | DisaggRequestState::Transferring { .. }
                | DisaggRequestState::Decoding { .. }
            )
        }).count()
    }

    /// Number of queued requests waiting for prefill.
    pub fn queued_count(&self) -> usize {
        self.requests.values().filter(|s| matches!(s, DisaggRequestState::Queued)).count()
    }

    /// Number of completed requests.
    pub fn completed_count(&self) -> usize {
        self.requests.values().filter(|s| matches!(s, DisaggRequestState::Complete { .. })).count()
    }

    /// Drain completed requests, returning their IDs.
    pub fn drain_completed(&mut self) -> Vec<u64> {
        let completed: Vec<u64> = self.requests.iter()
            .filter(|(_, s)| matches!(s, DisaggRequestState::Complete { .. }))
            .map(|(id, _)| *id)
            .collect();
        for id in &completed {
            self.requests.remove(id);
        }
        completed
    }
}

// ---------------------------------------------------------------------------
// Global context + FFI
// ---------------------------------------------------------------------------

static DISAGG_CTX: Mutex<Option<DisaggregatedRouter>> = Mutex::new(None);

/// Read NSL_ROLE env var and return a role code.
/// Returns: 0 = router, 1 = prefill, 2 = decode.
/// Used by codegen for role dispatch branching (avoids string ops in Cranelift IR).
#[no_mangle]
pub extern "C" fn nsl_disagg_get_role() -> i64 {
    match std::env::var("NSL_ROLE").as_deref() {
        Ok("router") => 0,
        Ok("prefill") => 1,
        Ok("decode") => 2,
        _ => 0, // default: router (monolithic fallback)
    }
}

/// Read NSL_LOCAL_RANK env var and return the rank.
/// Returns the integer rank (default 0).
#[no_mangle]
pub extern "C" fn nsl_disagg_get_rank() -> i64 {
    std::env::var("NSL_LOCAL_RANK")
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .unwrap_or(0)
}

/// Initialize the disaggregated inference router.
///
/// Parameters (all i64 for Cranelift):
/// - num_prefill: number of prefill workers
/// - num_decode: number of decode workers
/// - max_batch: max batch size per decode worker
/// - kv_blocks: KV blocks per decode worker
///
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_disagg_init(
    num_prefill: i64,
    num_decode: i64,
    max_batch: i64,
    kv_blocks: i64,
) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    let config = DisaggregatedConfig {
        num_prefill_workers: num_prefill as usize,
        num_decode_workers: num_decode as usize,
        max_batch_per_decode: max_batch as usize,
        kv_blocks_per_worker: kv_blocks as u32,
        ..Default::default()
    };
    *guard = Some(DisaggregatedRouter::new(config));
    0
}

/// Enqueue a request to the router. Returns the request ID.
#[no_mangle]
pub extern "C" fn nsl_disagg_enqueue() -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.enqueue_request() as i64
}

/// Select a prefill worker for the next request. Returns the worker rank, or -1 if none.
#[no_mangle]
pub extern "C" fn nsl_disagg_select_prefill() -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.select_prefill_worker().map(|r| r as i64).unwrap_or(-1)
}

/// Select a decode worker. Returns the worker rank, or -1 if none.
#[no_mangle]
pub extern "C" fn nsl_disagg_select_decode(kv_blocks_needed: i64) -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.select_decode_worker(kv_blocks_needed as u32).map(|r| r as i64).unwrap_or(-1)
}

/// Mark a request as prefilling on the given worker.
#[no_mangle]
pub extern "C" fn nsl_disagg_mark_prefilling(request_id: i64, prefill_rank: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.mark_prefilling(request_id as u64, prefill_rank as i32);
    0
}

/// Mark a request as decoding on the given worker, reserving kv_blocks_used blocks.
#[no_mangle]
pub extern "C" fn nsl_disagg_mark_decoding(request_id: i64, decode_rank: i64, kv_blocks_used: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.mark_decoding(request_id as u64, decode_rank as i32, kv_blocks_used as u32);
    0
}

/// Record a generated token for a request.
#[no_mangle]
pub extern "C" fn nsl_disagg_record_token(request_id: i64, _token_id: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.record_token(request_id as u64);
    0
}

/// Mark a request as complete.
#[no_mangle]
pub extern "C" fn nsl_disagg_mark_complete(request_id: i64, total_tokens: i64) -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_mut().expect("nsl_disagg_init not called");
    router.mark_complete(request_id as u64, total_tokens as u32);
    0
}

/// Returns the number of queued requests.
#[no_mangle]
pub extern "C" fn nsl_disagg_queued_count() -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.queued_count() as i64
}

/// Returns the number of active (in-flight) requests.
#[no_mangle]
pub extern "C" fn nsl_disagg_active_count() -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.active_request_count() as i64
}

/// Returns the number of completed requests.
#[no_mangle]
pub extern "C" fn nsl_disagg_completed_count() -> i64 {
    let guard = DISAGG_CTX.lock().unwrap();
    let router = guard.as_ref().expect("nsl_disagg_init not called");
    router.completed_count() as i64
}

/// Destroy the disaggregated router context.
#[no_mangle]
pub extern "C" fn nsl_disagg_destroy() -> i64 {
    let mut guard = DISAGG_CTX.lock().unwrap();
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_disagg_destroy();
        guard
    }

    #[test]
    fn router_basic_lifecycle() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 2,
            num_decode_workers: 3,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        assert_eq!(router.prefill_pool.len(), 2);
        assert_eq!(router.decode_pool.len(), 3);

        // Enqueue and dispatch
        let req0 = router.enqueue_request();
        let req1 = router.enqueue_request();
        assert_eq!(router.queued_count(), 2);

        // Select prefill workers
        let pw0 = router.select_prefill_worker().unwrap();
        router.mark_prefilling(req0, pw0);
        let pw1 = router.select_prefill_worker().unwrap();
        router.mark_prefilling(req1, pw1);
        assert_eq!(router.active_request_count(), 2);
        assert_eq!(router.queued_count(), 0);

        // Complete prefill, start decode
        let dw0 = router.select_decode_worker(4).unwrap();
        router.mark_transferring(req0, pw0, dw0);
        router.mark_decoding(req0, dw0, 4);

        // Generate tokens
        router.record_token(req0);
        router.record_token(req0);
        if let Some(DisaggRequestState::Decoding { tokens_generated, .. }) = router.request_state(req0) {
            assert_eq!(*tokens_generated, 2);
        }

        // Complete
        router.mark_complete(req0, 2);
        assert_eq!(router.completed_count(), 1);

        let drained = router.drain_completed();
        assert_eq!(drained, vec![req0]);
        assert_eq!(router.completed_count(), 0);
    }

    #[test]
    fn least_loaded_prefill_policy() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 2,
            num_decode_workers: 1,
            prefill_policy: PrefillPolicy::LeastLoaded,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        // Both idle — should pick rank 0 (first min)
        let r = router.select_prefill_worker().unwrap();
        assert_eq!(r, 0);

        // Load rank 0 — next should pick rank 1
        let req = router.enqueue_request();
        router.mark_prefilling(req, 0);
        let r = router.select_prefill_worker().unwrap();
        assert_eq!(r, 1);
    }

    #[test]
    fn round_robin_prefill_policy() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 3,
            num_decode_workers: 1,
            prefill_policy: PrefillPolicy::RoundRobin,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        assert_eq!(router.select_prefill_worker().unwrap(), 0);
        assert_eq!(router.select_prefill_worker().unwrap(), 1);
        assert_eq!(router.select_prefill_worker().unwrap(), 2);
        assert_eq!(router.select_prefill_worker().unwrap(), 0); // wraps
    }

    #[test]
    fn memory_aware_decode_policy() {
        let config = DisaggregatedConfig {
            num_prefill_workers: 1,
            num_decode_workers: 3,
            decode_policy: DecodePolicy::MemoryAware,
            kv_blocks_per_worker: 100,
            ..Default::default()
        };
        let mut router = DisaggregatedRouter::new(config);

        // Reduce blocks on worker 0
        router.decode_pool[0].free_kv_blocks = 50;
        // Worker 1 has most blocks (100) — should be selected
        let r = router.select_decode_worker(10).unwrap();
        // Workers 1 and 2 both have 100; first max is rank 1
        assert!(r == 1 || r == 2);
    }

    #[test]
    fn ffi_init_destroy() {
        let _lock = setup();

        assert_eq!(nsl_disagg_init(2, 4, 32, 2048), 0);
        assert_eq!(nsl_disagg_init(1, 1, 32, 2048), -1); // double init

        assert_eq!(nsl_disagg_enqueue(), 0); // first request
        assert_eq!(nsl_disagg_enqueue(), 1); // second request
        assert_eq!(nsl_disagg_queued_count(), 2);

        let pw = nsl_disagg_select_prefill();
        assert!(pw >= 0);
        nsl_disagg_mark_prefilling(0, pw);
        assert_eq!(nsl_disagg_active_count(), 1);

        nsl_disagg_mark_complete(0, 10);
        assert_eq!(nsl_disagg_completed_count(), 1);

        assert_eq!(nsl_disagg_destroy(), 0);
    }
}
```

### Task 4: Worker Loop Functions

**Files:**
- Create: `crates/nsl-runtime/src/disaggregated/worker.rs`

- [ ] **Step 5: Create `worker.rs` with prefill/decode worker loop FFI stubs**

```rust
// crates/nsl-runtime/src/disaggregated/worker.rs
//! M41: Prefill and decode worker loop implementations.
//!
//! Each worker runs as a separate OS process. The loops receive messages
//! from the router, process requests, and send results back.
//! For now these are FFI entry points that the codegen's role dispatch calls.

use std::sync::Mutex;

use super::messages::WorkerRole;

// ---------------------------------------------------------------------------
// Worker state
// ---------------------------------------------------------------------------

static WORKER_CTX: Mutex<Option<WorkerContext>> = Mutex::new(None);

struct WorkerContext {
    role: WorkerRole,
    rank: i32,
    _model_ptr: i64,  // opaque model handle from codegen
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

    let mut guard = WORKER_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(WorkerContext {
        role: worker_role,
        rank: rank as i32,
        _model_ptr: model_ptr,
    });
    0
}

/// Run the prefill worker loop.
///
/// This is the main entry point for a prefill worker process.
/// The loop:
/// 1. Receives StartPrefill messages from the router
/// 2. Runs the model's forward pass on the prompt tokens
/// 3. Serializes the KV-cache and transfers to the assigned decode worker
/// 4. Sends PrefillComplete back to the router
///
/// `config_ptr`: pointer to serve config (max_batch, prefill_chunk, etc.)
///
/// Returns 0 on clean shutdown, negative on error.
#[no_mangle]
pub extern "C" fn nsl_disagg_prefill_loop(_config_ptr: i64) -> i64 {
    // Verify role (brief lock, then release)
    {
        let guard = WORKER_CTX.lock().unwrap();
        let ctx = guard.as_ref().expect("nsl_disagg_worker_init not called");
        assert_eq!(ctx.role, WorkerRole::Prefill, "prefill_loop called on non-prefill worker");
    }

    // In a full implementation, this would:
    // 1. Loop waiting for messages from the router (via shared memory or pipe)
    // 2. For each StartPrefill: allocate KV blocks, run forward, serialize KV,
    //    transfer to decode worker, notify router
    // Currently a stub that returns immediately for testing.
    0
}

/// Run the decode worker loop.
///
/// This is the main entry point for a decode worker process.
/// The loop:
/// 1. Checks for incoming KV transfers (non-blocking)
/// 2. Runs batched decode step on all active sequences
/// 3. Streams generated tokens back to the router
/// 4. Cleans up completed sequences
///
/// `config_ptr`: pointer to serve config
///
/// Returns 0 on clean shutdown, negative on error.
#[no_mangle]
pub extern "C" fn nsl_disagg_decode_loop(_config_ptr: i64) -> i64 {
    // Verify role (brief lock, then release)
    {
        let guard = WORKER_CTX.lock().unwrap();
        let ctx = guard.as_ref().expect("nsl_disagg_worker_init not called");
        assert_eq!(ctx.role, WorkerRole::Decode, "decode_loop called on non-decode worker");
    }

    // In a full implementation, this would:
    // 1. Check for KV transfers via try_recv_kv()
    // 2. Admit new sequences into the BatchScheduler
    // 3. Run batched decode step (reuse M29 scheduler.step())
    // 4. Sample tokens, stream to router
    // 5. Free KV pages on EOS
    // Currently a stub that returns immediately for testing.
    0
}

/// Destroy the worker context.
#[no_mangle]
pub extern "C" fn nsl_disagg_worker_destroy() -> i64 {
    let mut guard = WORKER_CTX.lock().unwrap();
    *guard = None;
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
        // Prefill loop returns immediately (stub)
        assert_eq!(nsl_disagg_prefill_loop(0), 0);
        assert_eq!(nsl_disagg_worker_destroy(), 0);
    }

    #[test]
    fn worker_init_decode() {
        let _lock = setup();
        assert_eq!(nsl_disagg_worker_init(2, 0, 0), 0);
        // Decode loop returns immediately (stub)
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
}
```

---

## Phase 3: Codegen + CLI + Semantic Integration

### Task 5: Serve Block Codegen Extension

**Files:**
- Modify: `crates/nsl-codegen/src/serve.rs`
- Modify: `crates/nsl-codegen/src/compiler.rs`

- [ ] **Step 6: Add disaggregated fields to Compiler struct**

In `crates/nsl-codegen/src/compiler.rs`, add to the `Compiler` struct:
```rust
/// M41: Whether this serve block uses disaggregated inference.
pub disaggregated: bool,
/// M41: Number of prefill workers (from serve config).
pub prefill_workers: usize,
/// M41: Number of decode workers (from serve config).
pub decode_workers: usize,
```

Initialize all three to `false`, `1`, `1` in the constructor.

- [ ] **Step 7: Extend `compile_serve_block` to detect disaggregation and emit role dispatch**

Replace the current `compile_serve_block` in `serve.rs` with:

```rust
pub fn compile_serve_block(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    serve: &ServeBlock,
) -> Result<(), CodegenError> {
    // M30: Initialize tensor parallelism if multi-device
    if self.world_size > 1 {
        self.compile_call_by_name(builder, "nsl_tp_init", &[])?;
    }

    // Extract config values with defaults
    let mut max_batch: i64 = 32;
    let mut max_seq_len: i64 = 4096;
    let mut kv_blocks: i64 = 2048;
    let mut prefill_chunk: i64 = 512;
    let mut prefill_workers: i64 = 1;
    let mut decode_workers: i64 = 1;

    for entry in &serve.config {
        let key_name = self.resolve_sym(entry.key).to_string();
        match key_name.as_str() {
            "max_batch" => {
                if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                    max_batch = *v;
                }
            }
            "max_seq_len" => {
                if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                    max_seq_len = *v;
                }
            }
            "kv_blocks" => {
                if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                    kv_blocks = *v;
                }
            }
            "prefill_chunk" => {
                if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                    prefill_chunk = *v;
                }
            }
            "prefill_workers" => {
                if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                    prefill_workers = *v;
                }
            }
            "decode_workers" => {
                if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                    decode_workers = *v;
                }
            }
            _ => {
                self.compile_expr(builder, state, &entry.value)?;
            }
        }
    }

    // Override from CLI flags if set
    if self.prefill_workers > 1 {
        prefill_workers = self.prefill_workers as i64;
    }
    if self.decode_workers > 1 {
        decode_workers = self.decode_workers as i64;
    }

    let is_disaggregated = prefill_workers > 1 || decode_workers > 1;

    if is_disaggregated {
        self.compile_disaggregated_serve(
            builder, state, serve,
            max_batch, max_seq_len, kv_blocks, prefill_chunk,
            prefill_workers, decode_workers,
        )?;
    } else {
        // Monolithic M29 path (unchanged)
        let v_max_batch = builder.ins().iconst(cl_types::I64, max_batch);
        let v_max_seq_len = builder.ins().iconst(cl_types::I64, max_seq_len);
        let v_kv_blocks = builder.ins().iconst(cl_types::I64, kv_blocks);
        let v_prefill_chunk = builder.ins().iconst(cl_types::I64, prefill_chunk);

        self.compile_call_by_name(
            builder,
            "nsl_serve_init",
            &[v_max_batch, v_max_seq_len, v_kv_blocks, v_prefill_chunk],
        )?;

        for endpoint in &serve.endpoints {
            for stmt in &endpoint.body.stmts {
                self.compile_stmt(builder, state, stmt)?;
            }
        }

        self.compile_call_by_name(builder, "nsl_serve_destroy", &[])?;
    }

    // M30: Tear down tensor parallelism if multi-device
    if self.world_size > 1 {
        self.compile_call_by_name(builder, "nsl_tp_destroy", &[])?;
    }

    Ok(())
}

/// M41: Compile a disaggregated serve block.
///
/// Generates role-dispatch code by reading NSL_ROLE env var at runtime:
///   - role == 0 (router):  nsl_disagg_init() + nsl_disagg_router_loop()
///   - role == 1 (prefill): nsl_disagg_worker_init(1, rank, 0) + nsl_disagg_prefill_loop(0)
///   - role == 2 (decode):  nsl_disagg_worker_init(2, rank, 0) + nsl_disagg_decode_loop(0)
///
/// The runtime FFI `nsl_disagg_get_role()` reads NSL_ROLE env var and returns
/// 0/1/2. This avoids string comparison in Cranelift IR.
fn compile_disaggregated_serve(
    &mut self,
    builder: &mut FunctionBuilder,
    state: &mut FuncState,
    serve: &ServeBlock,
    max_batch: i64,
    _max_seq_len: i64,
    kv_blocks: i64,
    _prefill_chunk: i64,
    prefill_workers: i64,
    decode_workers: i64,
) -> Result<(), CodegenError> {
    // Step 1: Get role from env var via FFI (returns 0=router, 1=prefill, 2=decode)
    let role = self.compile_call_by_name(builder, "nsl_disagg_get_role", &[])?;

    // Step 2: Branch on role
    let router_block = builder.create_block();
    let prefill_block = builder.create_block();
    let decode_block = builder.create_block();
    let merge_block = builder.create_block();

    // if role == 0 → router_block
    let zero = builder.ins().iconst(cl_types::I64, 0);
    let is_router = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, role, zero);
    builder.ins().brnz(is_router, router_block, &[]);

    // if role == 1 → prefill_block
    let one = builder.ins().iconst(cl_types::I64, 1);
    let is_prefill = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, role, one);
    builder.ins().brnz(is_prefill, prefill_block, &[]);

    // else → decode_block
    builder.ins().jump(decode_block, &[]);

    // --- Router block ---
    builder.switch_to_block(router_block);
    let v_prefill = builder.ins().iconst(cl_types::I64, prefill_workers);
    let v_decode = builder.ins().iconst(cl_types::I64, decode_workers);
    let v_batch = builder.ins().iconst(cl_types::I64, max_batch);
    let v_kv = builder.ins().iconst(cl_types::I64, kv_blocks);
    self.compile_call_by_name(
        builder, "nsl_disagg_init", &[v_prefill, v_decode, v_batch, v_kv],
    )?;
    // Router runs endpoint bodies (sets up the event loop)
    for endpoint in &serve.endpoints {
        for stmt in &endpoint.body.stmts {
            self.compile_stmt(builder, state, stmt)?;
        }
    }
    self.compile_call_by_name(builder, "nsl_disagg_destroy", &[])?;
    builder.ins().jump(merge_block, &[]);

    // --- Prefill worker block ---
    builder.switch_to_block(prefill_block);
    let role_prefill = builder.ins().iconst(cl_types::I64, 1);
    let rank = self.compile_call_by_name(builder, "nsl_disagg_get_rank", &[])?;
    let model_zero = builder.ins().iconst(cl_types::I64, 0); // model ptr placeholder
    self.compile_call_by_name(
        builder, "nsl_disagg_worker_init", &[role_prefill, rank, model_zero],
    )?;
    let config_zero = builder.ins().iconst(cl_types::I64, 0);
    self.compile_call_by_name(builder, "nsl_disagg_prefill_loop", &[config_zero])?;
    self.compile_call_by_name(builder, "nsl_disagg_worker_destroy", &[])?;
    builder.ins().jump(merge_block, &[]);

    // --- Decode worker block ---
    builder.switch_to_block(decode_block);
    let role_decode = builder.ins().iconst(cl_types::I64, 2);
    let rank2 = self.compile_call_by_name(builder, "nsl_disagg_get_rank", &[])?;
    let model_zero2 = builder.ins().iconst(cl_types::I64, 0);
    self.compile_call_by_name(
        builder, "nsl_disagg_worker_init", &[role_decode, rank2, model_zero2],
    )?;
    let config_zero2 = builder.ins().iconst(cl_types::I64, 0);
    self.compile_call_by_name(builder, "nsl_disagg_decode_loop", &[config_zero2])?;
    self.compile_call_by_name(builder, "nsl_disagg_worker_destroy", &[])?;
    builder.ins().jump(merge_block, &[]);

    // --- Merge block ---
    builder.switch_to_block(merge_block);

    Ok(())
}
```

### Task 6: CLI Flags

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`

- [ ] **Step 8: Add `--prefill-workers` and `--decode-workers` flags to the `Run` command**

Add to the `Run` variant in the `Cli` enum:
```rust
/// M41: Number of prefill workers for disaggregated inference
#[arg(long, default_value = "1")]
prefill_workers: u32,

/// M41: Number of decode workers for disaggregated inference
#[arg(long, default_value = "1")]
decode_workers: u32,
```

In the `Cli::Run` handler, pass these values to the compiler:
```rust
compiler.prefill_workers = prefill_workers as usize;
compiler.decode_workers = decode_workers as usize;
```

- [ ] **Step 8b: Add disaggregated process spawning in the Run handler**

In the `Cli::Run` handler, after building the binary, add a disaggregated spawn block
(modeled on the existing `--devices` spawn logic at lines 273-346 of `main.rs`):

```rust
// M41: Disaggregated inference — spawn router + prefill + decode workers.
// Each runs the same compiled binary with NSL_ROLE and NSL_LOCAL_RANK env vars.
if prefill_workers > 1 || decode_workers > 1 {
    let binary_path = &output_path; // compiled binary from build step

    let mut children = Vec::new();

    // Spawn router (rank 0)
    let router_child = std::process::Command::new(binary_path)
        .env("NSL_ROLE", "router")
        .env("NSL_LOCAL_RANK", "0")
        .env("NSL_WORLD_SIZE", format!("{}", 1 + prefill_workers + decode_workers))
        .spawn()
        .expect("failed to spawn router process");
    children.push(("router:0", router_child));

    // Spawn prefill workers
    for i in 0..prefill_workers {
        let child = std::process::Command::new(binary_path)
            .env("NSL_ROLE", "prefill")
            .env("NSL_LOCAL_RANK", format!("{}", i))
            .spawn()
            .expect("failed to spawn prefill worker");
        children.push((Box::leak(format!("prefill:{}", i).into_boxed_str()), child));
    }

    // Spawn decode workers
    for i in 0..decode_workers {
        let child = std::process::Command::new(binary_path)
            .env("NSL_ROLE", "decode")
            .env("NSL_LOCAL_RANK", format!("{}", i))
            .spawn()
            .expect("failed to spawn decode worker");
        children.push((Box::leak(format!("decode:{}", i).into_boxed_str()), child));
    }

    // Wait for all processes
    let mut exit_code = 0;
    for (name, mut child) in children {
        let status = child.wait().expect("failed to wait on child");
        if !status.success() {
            eprintln!("[nsl] {} exited with {}", name, status);
            exit_code = 1;
        }
    }
    std::process::exit(exit_code);
}
```

### Task 7: Semantic Validation

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 9: Add `check_disaggregated_serve()` validation in the semantic checker**

Add a method to the checker that validates disaggregated serve config:

```rust
fn check_disaggregated_serve(&mut self, serve: &ServeBlock) {
    let mut prefill_workers = 1i64;
    let mut decode_workers = 1i64;
    let mut kv_transfer: Option<String> = None;

    for entry in &serve.config {
        let key = self.resolve_sym(entry.key).to_string();
        match key.as_str() {
            "prefill_workers" => {
                if let ExprKind::IntLiteral(v) = &entry.value.kind {
                    prefill_workers = *v;
                    if *v < 1 {
                        self.error(entry.value.span, "prefill_workers must be >= 1");
                    }
                }
            }
            "decode_workers" => {
                if let ExprKind::IntLiteral(v) = &entry.value.kind {
                    decode_workers = *v;
                    if *v < 1 {
                        self.error(entry.value.span, "decode_workers must be >= 1");
                    }
                }
            }
            "kv_transfer" => {
                if let ExprKind::StringLiteral(s) = &entry.value.kind {
                    let valid = ["rdma", "nvlink", "tcp", "shared_mem", "auto"];
                    if !valid.contains(&s.as_str()) {
                        self.error(
                            entry.value.span,
                            &format!(
                                "unknown kv_transfer backend '{}', expected one of: {}",
                                s,
                                valid.join(", ")
                            ),
                        );
                    }
                    kv_transfer = Some(s.clone());
                }
            }
            "drain_timeout_ms" => {
                if let ExprKind::IntLiteral(v) = &entry.value.kind {
                    if *v < 0 {
                        self.error(entry.value.span, "drain_timeout_ms must be >= 0");
                    }
                }
            }
            _ => {} // other config entries validated elsewhere
        }
    }

    // If disaggregated (either > 1), validate kv_transfer has a sensible default
    if (prefill_workers > 1 || decode_workers > 1) && kv_transfer.is_none() {
        // Default is "shared_mem" — no error needed, just a note for the user
    }
}
```

Wire this into the existing `check_serve_block()` method so it runs whenever a serve block is type-checked.

---

## Phase 4: Build Verification

### Task 8: Build + Test

- [ ] **Step 10: `cargo build` — verify no compile errors**

- [ ] **Step 11: `cargo test` — run all unit tests, expect 28+ new tests passing**

Expected new tests:
- `messages::tests::*` (10 tests: roundtrip for all message types, worker handle, state transitions, edge cases)
- `kv_transfer::tests::*` (8 tests: header validation, byte computation, backend send/recv, try_recv, dtype sizes, FFI lifecycle)
- `router::tests::*` (6 tests: lifecycle, policies, FFI init/destroy, get_role/get_rank)
- `worker::tests::*` (4 tests: init/destroy, prefill/decode loops, invalid role)

- [ ] **Step 12: `cargo clippy` — no warnings**

---

## Verification Checklist

After implementation, verify:

1. **Message protocol**: All 5 message types roundtrip through serialize/deserialize
2. **KV transfer**: SharedMemBackend can send/recv KV pages with correct data
3. **Router**: LeastLoaded and RoundRobin prefill policies work; MemoryAware decode policy picks worker with most free blocks
4. **Request lifecycle**: Queued → Prefilling → Transferring → Decoding → Complete transitions all work
5. **Codegen**: serve.rs correctly detects `prefill_workers > 1` and emits role dispatch (router/prefill/decode branching via `nsl_disagg_get_role`)
6. **CLI**: `--prefill-workers` and `--decode-workers` flags parse correctly; process spawning creates 1 router + N prefill + M decode processes with correct env vars
7. **Semantic**: Invalid kv_transfer backends produce error diagnostics
8. **No regressions**: All existing tests pass (monolithic serve path unchanged)
