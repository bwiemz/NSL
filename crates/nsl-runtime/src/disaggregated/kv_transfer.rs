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
