//! Collective communication backend trait and simulated (shared-memory) implementation.

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

// ---------------------------------------------------------------------------
// DtypeId constants — mirrors tensor.rs built-in IDs
// ---------------------------------------------------------------------------

pub type DtypeId = u16;

pub const DTYPE_F64: DtypeId = 0;
pub const DTYPE_F32: DtypeId = 1;
pub const DTYPE_F16: DtypeId = 2;
pub const DTYPE_BF16: DtypeId = 3;
pub const DTYPE_I8: DtypeId = 4;
pub const DTYPE_FP8: DtypeId = 5;

// P0.4 dtype/ABI cleanup: this module historically re-declared the runtime
// dtype tags by copy. Pin them to the canonical `crate::tensor::DTYPE_*` so any
// future divergence is a COMPILE error, not a silent wire mismatch. (The names
// differ by convention — F16/I8/FP8 here vs FP16/INT8/FP8E4M3 there — but the
// numeric tags MUST agree.)
const _: () = {
    assert!(DTYPE_F64 == crate::tensor::DTYPE_F64);
    assert!(DTYPE_F32 == crate::tensor::DTYPE_F32);
    assert!(DTYPE_F16 == crate::tensor::DTYPE_FP16);
    assert!(DTYPE_BF16 == crate::tensor::DTYPE_BF16);
    assert!(DTYPE_I8 == crate::tensor::DTYPE_INT8);
    assert!(DTYPE_FP8 == crate::tensor::DTYPE_FP8E4M3);
};

/// Returns the byte width for a built-in dtype, or 0 for unknown.
pub fn dtype_byte_width(dtype: DtypeId) -> usize {
    match dtype {
        DTYPE_F64 => 8,
        DTYPE_F32 => 4,
        DTYPE_F16 => 2,
        DTYPE_BF16 => 2,
        DTYPE_I8 => 1,
        DTYPE_FP8 => 1,
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Opaque stream handle (placeholder for CUDA stream pointers)
// ---------------------------------------------------------------------------

pub type StreamHandle = *mut c_void;

// ---------------------------------------------------------------------------
// CollectiveBackend trait
// ---------------------------------------------------------------------------

/// Abstraction over collective communication (NCCL, simulated, etc.).
///
/// All buffer pointers are device or host memory depending on the backend.
/// Return value: 0 on success, negative on error.
pub trait CollectiveBackend {
    /// Element-wise sum across all ranks.  `count` = number of elements.
    fn all_reduce_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32;

    /// Gather `send_count` elements from each rank into `recvbuf` (total = send_count * world_size).
    fn all_gather(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        send_count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32;

    /// Broadcast `count` elements from `root_rank` to all ranks.
    fn broadcast(
        &self,
        buf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        root_rank: i32,
        stream: StreamHandle,
    ) -> i32;

    /// D3 (ZeRO-2 backward): element-wise sum across ranks, each rank
    /// keeping only its `count / world_size` slice (rank r receives the
    /// summed elements `[r*count/ws, (r+1)*count/ws)`). `count` MUST be
    /// divisible by `world_size` (return -3 otherwise). CONTRACT (pinned
    /// by the gate): the result is bit-equal to `all_reduce_sum` followed
    /// by a local slice — implementations must use the same fixed
    /// reduction order so ZeRO-sharded arithmetic stays bit-exact against
    /// the unsharded baseline.
    fn reduce_scatter_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32;

    /// Block until all ranks have reached this point.
    fn barrier(&self) -> i32;

    /// Point-to-point send to a specific rank.
    fn send(
        &self,
        sendbuf: *const c_void,
        count: usize,
        dtype_bytes: usize,
        dst_rank: i32,
    ) -> i32;

    /// Point-to-point receive from a specific rank.
    fn recv(
        &self,
        recvbuf: *mut c_void,
        count: usize,
        dtype_bytes: usize,
        src_rank: i32,
    ) -> i32;

    /// This process's rank (0-based).
    fn rank(&self) -> i32;

    /// Total number of participating ranks.
    fn world_size(&self) -> i32;
}

// ---------------------------------------------------------------------------
// Shared-memory header for SimulatedBackend (multi-process)
// ---------------------------------------------------------------------------

/// Layout at the start of the shared memory region.
/// After the header, per-rank data slots follow at offset `DATA_OFFSET`.
#[repr(C)]
pub struct ShmHeader {
    /// Monotonically increasing generation counter (even = idle, odd = in-barrier).
    pub generation: AtomicU32,
    /// Number of ranks that have arrived at the current barrier.
    pub arrival: AtomicU32,
}

/// Byte offset where per-rank data slots begin.
const DATA_OFFSET: usize = 64; // cache-line aligned, leaves room for header growth

// ---------------------------------------------------------------------------
// f16 / bf16 conversion helpers (software, no hardware dependency)
// ---------------------------------------------------------------------------

/// Convert IEEE 754 half-precision (f16) bits to f32.
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        // Subnormal or zero
        if mant == 0 { return f32::from_bits(sign); }
        let mut e = 1u32;
        let mut m = mant;
        while (m & 0x400) == 0 { m <<= 1; e += 1; }
        let f_exp = (127 - 15 + 1 - e) << 23;
        let f_mant = (m & 0x3FF) << 13;
        f32::from_bits(sign | f_exp | f_mant)
    } else if exp == 31 {
        // Inf or NaN
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        let f_exp = (exp + 127 - 15) << 23;
        f32::from_bits(sign | f_exp | (mant << 13))
    }
}

/// Convert f32 to IEEE 754 half-precision (f16) bits (round-to-nearest-even).
fn f32_to_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp >= 143 { // overflow → inf
        return sign | 0x7C00;
    }
    if exp <= 102 { // underflow → zero
        return sign;
    }
    if exp <= 112 { // subnormal
        let shift = 113 - exp;
        let m = (0x800000 | mant) >> (shift + 13);
        return sign | m as u16;
    }
    let f16_exp = ((exp - 112) as u16) << 10;
    let f16_mant = (mant >> 13) as u16;
    sign | f16_exp | f16_mant
}

/// Convert bfloat16 bits to f32 (simple: bf16 is just the top 16 bits of f32).
fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Convert f32 to bfloat16 bits (truncate lower 16 bits).
fn f32_to_bf16(val: f32) -> u16 {
    (val.to_bits() >> 16) as u16
}

// ---------------------------------------------------------------------------
// SimulatedBackend — file-backed shared memory for testing without NCCL
// ---------------------------------------------------------------------------

/// A collective backend that uses a memory-mapped file for cross-process IPC.
///
/// For `world_size == 1` no shared memory is needed — operations are simple memcpy.
/// For `world_size > 1` each rank writes its data into a per-rank slot, executes
/// a spin-barrier, then reads peer data to perform the reduction / gather.
pub struct SimulatedBackend {
    rank: i32,
    world_size: i32,
    /// Pointer to the start of the shared memory region (contains ShmHeader + data slots).
    shm_ptr: *mut u8,
    /// Total length of the shared memory region in bytes.
    shm_len: usize,
}

// SAFETY: The raw pointer targets a memory-mapped region that outlives the backend.
// Synchronization is handled via atomics in ShmHeader.
unsafe impl Send for SimulatedBackend {}
unsafe impl Sync for SimulatedBackend {}

impl SimulatedBackend {
    /// Create a new simulated backend.
    ///
    /// * `rank` — this process's rank (0-based).
    /// * `world_size` — total number of ranks.
    /// * `shm_ptr` — pointer to mmap'd shared region; may be null when `world_size == 1`.
    /// * `shm_len` — byte length of the shared region.
    pub fn new(rank: i32, world_size: i32, shm_ptr: *mut u8, shm_len: usize) -> Self {
        Self { rank, world_size, shm_ptr, shm_len }
    }

    // -- internal helpers ---------------------------------------------------

    fn header(&self) -> &ShmHeader {
        assert!(!self.shm_ptr.is_null(), "shared memory not mapped");
        unsafe { &*(self.shm_ptr as *const ShmHeader) }
    }

    /// Returns a mutable pointer to the start of rank `r`'s data slot.
    fn slot_ptr(&self, r: i32, slot_bytes: usize) -> *mut u8 {
        let offset = DATA_OFFSET + (r as usize) * slot_bytes;
        assert!(offset + slot_bytes <= self.shm_len, "slot exceeds shm bounds");
        unsafe { self.shm_ptr.add(offset) }
    }

    /// D3 (review): does a per-rank slot of `slot_bytes` fit for ALL ranks?
    /// Every rank processes the same tensor order and computes the same
    /// `slot_bytes`, so a failure here is symmetric — all ranks return the
    /// same error at the same op instead of one panicking in `slot_ptr`
    /// while peers spin forever on the next barrier.
    fn slot_fits(&self, slot_bytes: usize) -> bool {
        DATA_OFFSET + (self.world_size as usize) * slot_bytes <= self.shm_len
    }

    /// Spin-barrier using double-buffered generation counter.
    ///
    /// All ranks increment `arrival`; the last rank to arrive bumps `generation`
    /// and resets `arrival` for the next round.  Other ranks spin on `generation`.
    ///
    /// D3 (review): bounded so a dead peer aborts the job loudly instead of
    /// spinning all survivors at 100% CPU forever. Deadline is
    /// `NSL_TP_BARRIER_TIMEOUT_SECS` (default 300s) — generous enough for
    /// slow legitimate steps, finite for a crashed rank. Checked every 64K
    /// spins so the hot path stays a bare load + spin_loop.
    fn spin_barrier(&self) {
        let hdr = self.header();
        let gen_before = hdr.generation.load(Ordering::Acquire);

        let prev = hdr.arrival.fetch_add(1, Ordering::AcqRel);
        if prev + 1 == self.world_size as u32 {
            // Last to arrive — reset arrival and bump generation.
            hdr.arrival.store(0, Ordering::Release);
            hdr.generation.store(gen_before.wrapping_add(1), Ordering::Release);
        } else {
            let timeout_secs: u64 = std::env::var("NSL_TP_BARRIER_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300);
            let start = std::time::Instant::now();
            let mut spins: u64 = 0;
            while hdr.generation.load(Ordering::Acquire) == gen_before {
                std::hint::spin_loop();
                spins += 1;
                if spins & 0xFFFF == 0
                    && start.elapsed().as_secs() >= timeout_secs
                {
                    eprintln!(
                        "nsl: collective barrier timed out after {timeout_secs}s \
                         on rank {} — a peer rank likely died. Aborting (set \
                         NSL_TP_BARRIER_TIMEOUT_SECS to tune).",
                        self.rank
                    );
                    std::process::abort();
                }
            }
        }
    }
}

impl CollectiveBackend for SimulatedBackend {
    fn all_reduce_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        let bw = dtype_byte_width(dtype);
        if bw == 0 { return -1; }
        let nbytes = count * bw;

        // Single-rank fast path: just copy.
        if self.world_size == 1 {
            if !std::ptr::eq(sendbuf, recvbuf) {
                unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, recvbuf as *mut u8, nbytes); }
            }
            return 0;
        }
        if !self.slot_fits(nbytes) {
            eprintln!(
                "nsl: all_reduce_sum: {nbytes}-byte tensor exceeds the per-rank \
                 shm slot ({} ranks, {} bytes total) — raise the --devices shm \
                 budget or shard/chunk the tensor",
                self.world_size, self.shm_len
            );
            return -4;
        }

        // Write local data into our slot.
        let slot = self.slot_ptr(self.rank, nbytes);
        unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, slot, nbytes); }

        // Wait for all ranks to finish writing.
        self.spin_barrier();

        // Reduce: sum across all ranks.
        match dtype {
            DTYPE_F32 => {
                let dst = recvbuf as *mut f32;
                for i in 0..count {
                    let mut acc: f32 = 0.0;
                    for r in 0..self.world_size {
                        let src = self.slot_ptr(r, nbytes) as *const f32;
                        acc += unsafe { *src.add(i) };
                    }
                    unsafe { *dst.add(i) = acc; }
                }
            }
            DTYPE_F64 => {
                let dst = recvbuf as *mut f64;
                for i in 0..count {
                    let mut acc: f64 = 0.0;
                    for r in 0..self.world_size {
                        let src = self.slot_ptr(r, nbytes) as *const f64;
                        acc += unsafe { *src.add(i) };
                    }
                    unsafe { *dst.add(i) = acc; }
                }
            }
            DTYPE_F16 | DTYPE_BF16 => {
                // f16/bf16 reduction: accumulate in f32 for numerical stability,
                // then convert back. Uses bit manipulation for f16<->f32 conversion.
                let dst = recvbuf as *mut u16;
                for i in 0..count {
                    let mut acc: f32 = 0.0;
                    for r in 0..self.world_size {
                        let src = self.slot_ptr(r, nbytes) as *const u16;
                        let bits = unsafe { *src.add(i) };
                        let val = if dtype == DTYPE_F16 {
                            half_to_f32(bits)
                        } else {
                            bf16_to_f32(bits)
                        };
                        acc += val;
                    }
                    let result_bits = if dtype == DTYPE_F16 {
                        f32_to_half(acc)
                    } else {
                        f32_to_bf16(acc)
                    };
                    unsafe { *dst.add(i) = result_bits; }
                }
            }
            _ => return -2, // unsupported dtype for reduction
        }

        // Second barrier so no rank overwrites its slot before peers finish reading.
        self.spin_barrier();
        0
    }

    fn reduce_scatter_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        let bw = dtype_byte_width(dtype);
        if bw == 0 {
            return -1;
        }
        if !count.is_multiple_of(self.world_size.max(1) as usize) {
            return -3;
        }
        let nbytes = count * bw;
        let per = count / self.world_size as usize;

        if self.world_size == 1 {
            if !std::ptr::eq(sendbuf, recvbuf) {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        sendbuf as *const u8,
                        recvbuf as *mut u8,
                        nbytes,
                    );
                }
            }
            return 0;
        }
        if !self.slot_fits(nbytes) {
            eprintln!(
                "nsl: reduce_scatter_sum: {nbytes}-byte tensor exceeds the \
                 per-rank shm slot ({} ranks) — raise the --devices shm budget",
                self.world_size
            );
            return -4;
        }

        let slot = self.slot_ptr(self.rank, nbytes);
        unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, slot, nbytes) };
        self.spin_barrier();

        // The local slice of the SAME fixed rank-order per-element sum
        // `all_reduce_sum` computes — bit-equal to all-reduce + slice by
        // construction (the gate pins this contract). Only f32/f64 are
        // supported (the ZeRO path is full-precision).
        let base = self.rank as usize * per;
        match dtype {
            DTYPE_F32 => {
                let dst = recvbuf as *mut f32;
                for i in 0..per {
                    let mut acc: f32 = 0.0;
                    for r in 0..self.world_size {
                        let src = self.slot_ptr(r, nbytes) as *const f32;
                        acc += unsafe { *src.add(base + i) };
                    }
                    unsafe { *dst.add(i) = acc };
                }
            }
            DTYPE_F64 => {
                let dst = recvbuf as *mut f64;
                for i in 0..per {
                    let mut acc: f64 = 0.0;
                    for r in 0..self.world_size {
                        let src = self.slot_ptr(r, nbytes) as *const f64;
                        acc += unsafe { *src.add(base + i) };
                    }
                    unsafe { *dst.add(i) = acc };
                }
            }
            _ => return -2,
        }

        self.spin_barrier();
        0
    }

    fn all_gather(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        send_count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        let bw = dtype_byte_width(dtype);
        if bw == 0 { return -1; }
        let nbytes = send_count * bw;

        // Single-rank fast path.
        if self.world_size == 1 {
            if !std::ptr::eq(sendbuf, recvbuf) {
                unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, recvbuf as *mut u8, nbytes); }
            }
            return 0;
        }

        // Write local chunk into our slot.
        let slot = self.slot_ptr(self.rank, nbytes);
        unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, slot, nbytes); }

        self.spin_barrier();

        // Copy each rank's slot into the corresponding segment of recvbuf.
        let dst = recvbuf as *mut u8;
        for r in 0..self.world_size {
            let src = self.slot_ptr(r, nbytes);
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst.add((r as usize) * nbytes), nbytes);
            }
        }

        self.spin_barrier();
        0
    }

    fn broadcast(
        &self,
        buf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        root_rank: i32,
        _stream: StreamHandle,
    ) -> i32 {
        let bw = dtype_byte_width(dtype);
        if bw == 0 { return -1; }
        let nbytes = count * bw;

        // Single-rank fast path: buf already contains the data.
        if self.world_size == 1 {
            return 0;
        }
        if !self.slot_fits(nbytes) {
            eprintln!(
                "nsl: broadcast: {nbytes}-byte tensor exceeds the per-rank shm \
                 slot ({} ranks) — raise the --devices shm budget",
                self.world_size
            );
            return -4;
        }

        // Root writes data into its slot.
        if self.rank == root_rank {
            let slot = self.slot_ptr(root_rank, nbytes);
            unsafe { std::ptr::copy_nonoverlapping(buf as *const u8, slot, nbytes); }
        }

        self.spin_barrier();

        // Non-root ranks copy from root's slot.
        if self.rank != root_rank {
            let src = self.slot_ptr(root_rank, nbytes);
            unsafe { std::ptr::copy_nonoverlapping(src, buf as *mut u8, nbytes); }
        }

        self.spin_barrier();
        0
    }

    fn barrier(&self) -> i32 {
        if self.world_size > 1 {
            self.spin_barrier();
        }
        0
    }

    fn send(
        &self,
        sendbuf: *const c_void,
        count: usize,
        dtype_bytes: usize,
        _dst_rank: i32,
    ) -> i32 {
        let nbytes = count * dtype_bytes;

        if self.world_size == 1 {
            // Single-rank: write to our slot for later recv.
            // Caller must ensure shm is allocated even for single-rank p2p.
            if !self.shm_ptr.is_null() {
                let slot = self.slot_ptr(self.rank, nbytes);
                unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, slot, nbytes); }
            }
            return 0;
        }

        // Multi-rank: write to our slot, barrier so receiver can read, barrier again to
        // protect against slot reuse before receiver finishes.
        let slot = self.slot_ptr(self.rank, nbytes);
        unsafe { std::ptr::copy_nonoverlapping(sendbuf as *const u8, slot, nbytes); }
        self.spin_barrier(); // 1: data visible
        self.spin_barrier(); // 2: receiver done reading
        0
    }

    fn recv(
        &self,
        recvbuf: *mut c_void,
        count: usize,
        dtype_bytes: usize,
        src_rank: i32,
    ) -> i32 {
        let nbytes = count * dtype_bytes;

        if self.world_size == 1 {
            // Single-rank: read from our slot (written by previous send).
            if !self.shm_ptr.is_null() {
                let slot = self.slot_ptr(src_rank, nbytes);
                unsafe { std::ptr::copy_nonoverlapping(slot, recvbuf as *mut u8, nbytes); }
            }
            return 0;
        }

        // Multi-rank: barrier (wait for sender), read, barrier (signal done).
        self.spin_barrier(); // 1: data visible
        let slot = self.slot_ptr(src_rank, nbytes);
        unsafe { std::ptr::copy_nonoverlapping(slot, recvbuf as *mut u8, nbytes); }
        self.spin_barrier(); // 2: done reading
        0
    }

    fn rank(&self) -> i32 {
        self.rank
    }

    fn world_size(&self) -> i32 {
        self.world_size
    }
}

// ---------------------------------------------------------------------------
// P4 item 14: GpuStagedBackend — the CUDA-aware TEST backend.
// ---------------------------------------------------------------------------
//
// Same contract as NcclBackend (buffers are DEVICE pointers) but the data
// plane stages through the CPU-shm SimulatedBackend: DtoH → host collective →
// HtoD. Exists so the ZeRO CUDA-aware plumbing (device-pointer collectives,
// on-device averaging, broadcast roots, GPU parity) is validatable on a
// SINGLE-GPU machine, where NCCL refuses multiple ranks per device
// (ncclInvalidUsage — verified empirically on NCCL 2.30). Selected with
// `--collectives sim-gpu`; NCCL remains the real multi-GPU transport.

#[cfg(feature = "cuda")]
pub struct GpuStagedBackend {
    host: SimulatedBackend,
}

#[cfg(feature = "cuda")]
unsafe impl Send for GpuStagedBackend {}

#[cfg(feature = "cuda")]
impl GpuStagedBackend {
    pub fn new(rank: i32, world_size: i32, shm_ptr: *mut u8, shm_len: usize) -> Self {
        Self {
            host: SimulatedBackend::new(rank, world_size, shm_ptr, shm_len),
        }
    }

    fn dtoh(dev: *const c_void, bytes: usize) -> Vec<u8> {
        let mut buf = vec![0u8; bytes];
        crate::cuda::inner::ensure_context();
        crate::cuda::inner::memcpy_dtoh(buf.as_mut_ptr() as *mut c_void, dev, bytes);
        buf
    }

    fn htod(dev: *mut c_void, buf: &[u8]) {
        crate::cuda::inner::memcpy_htod(dev, buf.as_ptr() as *const c_void, buf.len());
    }
}

#[cfg(feature = "cuda")]
impl CollectiveBackend for GpuStagedBackend {
    fn all_reduce_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32 {
        let bytes = count * dtype_byte_width(dtype);
        if bytes == 0 {
            return -1;
        }
        let mut staged = Self::dtoh(sendbuf, bytes);
        let rc = self.host.all_reduce_sum(
            staged.as_ptr() as *const c_void,
            staged.as_mut_ptr() as *mut c_void,
            count,
            dtype,
            stream,
        );
        if rc != 0 {
            return rc;
        }
        Self::htod(recvbuf, &staged);
        0
    }

    fn all_gather(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        send_count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32 {
        let bw = dtype_byte_width(dtype);
        let bytes = send_count * bw;
        if bytes == 0 {
            return -1;
        }
        let staged = Self::dtoh(sendbuf, bytes);
        let ws = self.host.world_size() as usize;
        let mut gathered = vec![0u8; bytes * ws];
        let rc = self.host.all_gather(
            staged.as_ptr() as *const c_void,
            gathered.as_mut_ptr() as *mut c_void,
            send_count,
            dtype,
            stream,
        );
        if rc != 0 {
            return rc;
        }
        Self::htod(recvbuf, &gathered);
        0
    }

    fn broadcast(
        &self,
        buf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        root_rank: i32,
        stream: StreamHandle,
    ) -> i32 {
        let bytes = count * dtype_byte_width(dtype);
        if bytes == 0 {
            return -1;
        }
        let mut staged = Self::dtoh(buf, bytes);
        let rc = self.host.broadcast(
            staged.as_mut_ptr() as *mut c_void,
            count,
            dtype,
            root_rank,
            stream,
        );
        if rc != 0 {
            return rc;
        }
        Self::htod(buf, &staged);
        0
    }

    fn reduce_scatter_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        stream: StreamHandle,
    ) -> i32 {
        let bw = dtype_byte_width(dtype);
        let ws = self.host.world_size() as usize;
        if bw == 0 {
            return -1;
        }
        if count % ws != 0 {
            return -3;
        }
        let staged = Self::dtoh(sendbuf, count * bw);
        let mut out = vec![0u8; (count / ws) * bw];
        let rc = self.host.reduce_scatter_sum(
            staged.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            count,
            dtype,
            stream,
        );
        if rc != 0 {
            return rc;
        }
        Self::htod(recvbuf, &out);
        0
    }

    fn barrier(&self) -> i32 {
        self.host.barrier()
    }

    fn send(&self, sendbuf: *const c_void, count: usize, dtype_bytes: usize, dst_rank: i32) -> i32 {
        self.host.send(sendbuf, count, dtype_bytes, dst_rank)
    }

    fn recv(&self, recvbuf: *mut c_void, count: usize, dtype_bytes: usize, src_rank: i32) -> i32 {
        self.host.recv(recvbuf, count, dtype_bytes, src_rank)
    }

    fn rank(&self) -> i32 {
        self.host.rank()
    }

    fn world_size(&self) -> i32 {
        self.host.world_size()
    }
}

// ---------------------------------------------------------------------------
// P4 item 14: NcclBackend — real CUDA-aware collectives over NCCL.
// ---------------------------------------------------------------------------
//
// Buffers are DEVICE pointers (unlike SimulatedBackend's host buffers).
// Bootstrap rides the spawner's existing shm file: rank 0 mints the
// ncclUniqueId and distributes it with the host backend's `broadcast`
// (inheriting its spin-barrier timeout/abort semantics), then every rank
// calls ncclCommInitRank under a watchdog thread. Each collective is issued
// on the calling thread's COMPUTE stream and then drained with a
// deadline-polled cuStreamQuery — a dead peer turns into a loud, symmetric
// abort on every surviving rank instead of an unbounded hang. Timeout env:
// NSL_NCCL_TIMEOUT_SECS (default 300, same convention as the shm barrier).

/// NCCL collective timeout (seconds).
#[cfg(feature = "nccl")]
fn nccl_timeout_secs() -> u64 {
    std::env::var("NSL_NCCL_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300)
}

#[cfg(feature = "nccl")]
pub struct NcclBackend {
    /// Host-side bootstrap + barrier companion (uniqueId exchange, host
    /// fallbacks for send/recv). Shares the spawner's shm mapping.
    host: SimulatedBackend,
    comm: cudarc::nccl::sys::ncclComm_t,
    /// Set once `ncclCommAbort` has run: Drop must then SKIP
    /// `ncclCommDestroy` — destroying an aborted communicator is undefined
    /// (NCCL documents abort as the terminal call for that comm) and can
    /// block or crash during process teardown.
    aborted: std::sync::atomic::AtomicBool,
}

// SAFETY: the ncclComm_t is only used from the (single-threaded) training
// path under the owning context's mutex — same argument as SimulatedBackend.
#[cfg(feature = "nccl")]
unsafe impl Send for NcclBackend {}

#[cfg(feature = "nccl")]
impl NcclBackend {
    /// Build the communicator clique. Every rank must call this in lockstep
    /// (the shm broadcast + ncclCommInitRank are collective). Returns a
    /// descriptive error string on any NCCL failure — the caller refuses
    /// loudly rather than training unsynchronized.
    pub fn new(
        rank: i32,
        world_size: i32,
        shm_ptr: *mut u8,
        shm_len: usize,
    ) -> Result<Self, String> {
        use cudarc::nccl::sys as nccl;
        let host = SimulatedBackend::new(rank, world_size, shm_ptr, shm_len);

        // Bind this rank's device BEFORE any NCCL call (rank-striped in
        // cuda::select_device_ordinal; on a single-GPU box every rank binds
        // device 0 and NCCL itself decides whether it accepts the topology).
        crate::cuda::inner::ensure_context();

        // Rank 0 mints the id; the host shm broadcast distributes it.
        let mut id = nccl::ncclUniqueId { internal: [0; 128] };
        if rank == 0 {
            let rc = unsafe { nccl::ncclGetUniqueId(&mut id) };
            if rc != nccl::ncclResult_t::ncclSuccess {
                return Err(format!("ncclGetUniqueId failed: {rc:?}"));
            }
        }
        let brc = host.broadcast(
            id.internal.as_mut_ptr() as *mut c_void,
            id.internal.len(),
            DTYPE_I8,
            0,
            std::ptr::null_mut(),
        );
        if brc != 0 {
            return Err(format!("uniqueId shm broadcast failed rc={brc}"));
        }

        // Init under a watchdog: ncclCommInitRank blocks until the whole
        // clique arrives — if a peer died after the broadcast above, this
        // would hang forever without the deadline.
        let done = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let watchdog = {
            let done = done.clone();
            let secs = nccl_timeout_secs();
            std::thread::spawn(move || {
                let deadline =
                    std::time::Instant::now() + std::time::Duration::from_secs(secs);
                while !done.load(std::sync::atomic::Ordering::Acquire) {
                    if std::time::Instant::now() >= deadline {
                        eprintln!(
                            "nsl: ncclCommInitRank timed out after {secs}s on rank {rank} \
                             — a peer rank likely died before joining the clique. \
                             Aborting (set NSL_NCCL_TIMEOUT_SECS to tune)."
                        );
                        std::process::abort();
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                }
            })
        };
        let mut comm: nccl::ncclComm_t = std::ptr::null_mut();
        let rc = unsafe { nccl::ncclCommInitRank(&mut comm, world_size, id, rank) };
        done.store(true, std::sync::atomic::Ordering::Release);
        let _ = watchdog.join();
        if rc != nccl::ncclResult_t::ncclSuccess {
            return Err(format!(
                "ncclCommInitRank(rank {rank}/{world_size}) failed: {rc:?} — on a \
                 single-GPU machine NCCL may refuse multiple ranks per device"
            ));
        }
        Ok(Self {
            host,
            comm,
            aborted: std::sync::atomic::AtomicBool::new(false),
        })
    }

    /// Abort the communicator exactly once and remember it for Drop.
    fn abort_comm(&self) {
        if !self
            .aborted
            .swap(true, std::sync::atomic::Ordering::AcqRel)
        {
            unsafe { cudarc::nccl::sys::ncclCommAbort(self.comm) };
        }
    }

    fn nccl_dtype(dtype: DtypeId) -> Option<cudarc::nccl::sys::ncclDataType_t> {
        use cudarc::nccl::sys::ncclDataType_t as T;
        match dtype {
            DTYPE_F64 => Some(T::ncclFloat64),
            DTYPE_F32 => Some(T::ncclFloat32),
            DTYPE_F16 => Some(T::ncclFloat16),
            DTYPE_BF16 => Some(T::ncclBfloat16),
            _ => None,
        }
    }

    /// Issue-side error check + watchdog-bounded completion. On failure or
    /// timeout: abort the communicator and the process — SYMMETRIC error
    /// propagation (peers observe our death through their own watchdogs, so
    /// no rank trains on while the clique is broken).
    fn finish(&self, rc: cudarc::nccl::sys::ncclResult_t, what: &str) -> i32 {
        use cudarc::nccl::sys as nccl;
        if rc != nccl::ncclResult_t::ncclSuccess {
            eprintln!(
                "nsl: {what} failed on rank {}: {rc:?} — aborting the communicator",
                self.host.rank()
            );
            self.abort_comm();
            return -1;
        }
        let secs = nccl_timeout_secs();
        if !crate::cuda::inner::sync_compute_stream_with_deadline(secs, what) {
            eprintln!(
                "nsl: {what} timed out after {secs}s on rank {} — a peer rank likely \
                 died mid-collective. Aborting (set NSL_NCCL_TIMEOUT_SECS to tune).",
                self.host.rank()
            );
            self.abort_comm();
            std::process::abort();
        }
        // Surface async (background-thread) NCCL errors loudly too.
        let mut aerr = nccl::ncclResult_t::ncclSuccess;
        unsafe { nccl::ncclCommGetAsyncError(self.comm, &mut aerr) };
        if aerr != nccl::ncclResult_t::ncclSuccess {
            eprintln!(
                "nsl: {what}: async NCCL error on rank {}: {aerr:?} — aborting",
                self.host.rank()
            );
            self.abort_comm();
            return -1;
        }
        0
    }

    fn compute_stream(&self) -> cudarc::nccl::sys::cudaStream_t {
        // CUstream (driver) and cudaStream_t (runtime) are the same object.
        crate::cuda::inner::current_stream() as cudarc::nccl::sys::cudaStream_t
    }
}

#[cfg(feature = "nccl")]
impl Drop for NcclBackend {
    fn drop(&mut self) {
        if !self.comm.is_null()
            && !self.aborted.load(std::sync::atomic::Ordering::Acquire)
        {
            unsafe { cudarc::nccl::sys::ncclCommDestroy(self.comm) };
        }
    }
}

#[cfg(feature = "nccl")]
impl CollectiveBackend for NcclBackend {
    fn all_reduce_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        use cudarc::nccl::sys as nccl;
        let Some(dt) = Self::nccl_dtype(dtype) else {
            return -2;
        };
        if self.host.world_size() == 1 {
            if sendbuf as *const u8 != recvbuf as *const u8 {
                let n = count * dtype_byte_width(dtype);
                crate::cuda::inner::memcpy_dtod(recvbuf, sendbuf, n);
            }
            return 0;
        }
        let rc = unsafe {
            nccl::ncclAllReduce(
                sendbuf,
                recvbuf,
                count,
                dt,
                nccl::ncclRedOp_t::ncclSum,
                self.comm,
                self.compute_stream(),
            )
        };
        self.finish(rc, "nccl all_reduce_sum")
    }

    fn all_gather(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        send_count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        use cudarc::nccl::sys as nccl;
        let Some(dt) = Self::nccl_dtype(dtype) else {
            return -2;
        };
        if self.host.world_size() == 1 {
            if sendbuf as *const u8 != recvbuf as *const u8 {
                let n = send_count * dtype_byte_width(dtype);
                crate::cuda::inner::memcpy_dtod(recvbuf, sendbuf, n);
            }
            return 0;
        }
        let rc = unsafe {
            nccl::ncclAllGather(sendbuf, recvbuf, send_count, dt, self.comm, self.compute_stream())
        };
        self.finish(rc, "nccl all_gather")
    }

    fn broadcast(
        &self,
        buf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        root_rank: i32,
        _stream: StreamHandle,
    ) -> i32 {
        use cudarc::nccl::sys as nccl;
        let Some(dt) = Self::nccl_dtype(dtype) else {
            return -2;
        };
        if self.host.world_size() == 1 {
            return 0;
        }
        let rc = unsafe {
            nccl::ncclBroadcast(buf, buf, count, dt, root_rank, self.comm, self.compute_stream())
        };
        self.finish(rc, "nccl broadcast")
    }

    fn reduce_scatter_sum(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        dtype: DtypeId,
        _stream: StreamHandle,
    ) -> i32 {
        use cudarc::nccl::sys as nccl;
        let Some(dt) = Self::nccl_dtype(dtype) else {
            return -2;
        };
        let ws = self.host.world_size() as usize;
        if count % ws != 0 {
            return -3;
        }
        if ws == 1 {
            if sendbuf as *const u8 != recvbuf as *const u8 {
                let n = count * dtype_byte_width(dtype);
                crate::cuda::inner::memcpy_dtod(recvbuf, sendbuf, n);
            }
            return 0;
        }
        // NOTE: NCCL's ring reduction order is deterministic per topology but
        // NOT the SimulatedBackend's fixed rank order — the bit-exact
        // reduce==all-reduce-slice CONTRACT is per-backend, not cross-backend.
        let rc = unsafe {
            nccl::ncclReduceScatter(
                sendbuf,
                recvbuf,
                count / ws,
                dt,
                nccl::ncclRedOp_t::ncclSum,
                self.comm,
                self.compute_stream(),
            )
        };
        self.finish(rc, "nccl reduce_scatter_sum")
    }

    fn barrier(&self) -> i32 {
        // Host-level barrier suffices for the control plane (data-plane
        // ordering is by stream); it also inherits the shm timeout/abort.
        self.host.barrier()
    }

    fn send(&self, sendbuf: *const c_void, count: usize, dtype_bytes: usize, dst_rank: i32) -> i32 {
        // Host-buffer point-to-point stays on the shm path (TP control use).
        self.host.send(sendbuf, count, dtype_bytes, dst_rank)
    }

    fn recv(&self, recvbuf: *mut c_void, count: usize, dtype_bytes: usize, src_rank: i32) -> i32 {
        self.host.recv(recvbuf, count, dtype_bytes, src_rank)
    }

    fn rank(&self) -> i32 {
        self.host.rank()
    }

    fn world_size(&self) -> i32 {
        self.host.world_size()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn dtype_byte_width_values() {
        assert_eq!(dtype_byte_width(DTYPE_F64), 8);
        assert_eq!(dtype_byte_width(DTYPE_F32), 4);
        assert_eq!(dtype_byte_width(DTYPE_F16), 2);
        assert_eq!(dtype_byte_width(DTYPE_BF16), 2);
        assert_eq!(dtype_byte_width(DTYPE_I8), 1);
        assert_eq!(dtype_byte_width(DTYPE_FP8), 1);
        assert_eq!(dtype_byte_width(999), 0); // unknown
    }

    #[test]
    fn simulated_backend_rank_and_world_size() {
        let backend = SimulatedBackend::new(0, 1, ptr::null_mut(), 0);
        assert_eq!(backend.rank(), 0);
        assert_eq!(backend.world_size(), 1);

        let backend2 = SimulatedBackend::new(3, 8, ptr::null_mut(), 0);
        assert_eq!(backend2.rank(), 3);
        assert_eq!(backend2.world_size(), 8);
    }

    #[test]
    fn simulated_backend_single_rank_all_reduce() {
        let backend = SimulatedBackend::new(0, 1, ptr::null_mut(), 0);
        let input: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut output: [f32; 4] = [0.0; 4];

        let rc = backend.all_reduce_sum(
            input.as_ptr() as *const c_void,
            output.as_mut_ptr() as *mut c_void,
            4,
            DTYPE_F32,
            ptr::null_mut(),
        );
        assert_eq!(rc, 0);
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0]);

        // Also test f64
        let input64: [f64; 3] = [10.0, 20.0, 30.0];
        let mut output64: [f64; 3] = [0.0; 3];
        let rc64 = backend.all_reduce_sum(
            input64.as_ptr() as *const c_void,
            output64.as_mut_ptr() as *mut c_void,
            3,
            DTYPE_F64,
            ptr::null_mut(),
        );
        assert_eq!(rc64, 0);
        assert_eq!(output64, [10.0, 20.0, 30.0]);
    }

    #[test]
    fn simulated_backend_single_rank_all_gather() {
        let backend = SimulatedBackend::new(0, 1, ptr::null_mut(), 0);
        let input: [f32; 3] = [5.0, 6.0, 7.0];
        let mut output: [f32; 3] = [0.0; 3];

        let rc = backend.all_gather(
            input.as_ptr() as *const c_void,
            output.as_mut_ptr() as *mut c_void,
            3,
            DTYPE_F32,
            ptr::null_mut(),
        );
        assert_eq!(rc, 0);
        assert_eq!(output, [5.0, 6.0, 7.0]);
    }

    #[test]
    fn simulated_backend_single_rank_broadcast() {
        let backend = SimulatedBackend::new(0, 1, ptr::null_mut(), 0);
        let mut data: [f64; 2] = [42.0, 99.0];

        let rc = backend.broadcast(
            data.as_mut_ptr() as *mut c_void,
            2,
            DTYPE_F64,
            0,
            ptr::null_mut(),
        );
        assert_eq!(rc, 0);
        assert_eq!(data, [42.0, 99.0]);
    }

    #[test]
    fn simulated_backend_single_rank_barrier() {
        let backend = SimulatedBackend::new(0, 1, ptr::null_mut(), 0);
        assert_eq!(backend.barrier(), 0);
    }

    #[test]
    fn simulated_backend_multi_rank_all_reduce() {
        // Simulate 2 ranks in the same process using a heap-allocated shared region.
        let world_size = 2;
        let count = 4;
        let slot_bytes = count * dtype_byte_width(DTYPE_F32);
        let shm_len = DATA_OFFSET + world_size * slot_bytes;
        let mut shm = vec![0u8; shm_len];
        let shm_ptr = shm.as_mut_ptr();

        // Initialize header.
        let hdr = unsafe { &*(shm_ptr as *const ShmHeader) };
        hdr.generation.store(0, Ordering::Release);
        hdr.arrival.store(0, Ordering::Release);

        let b0 = SimulatedBackend::new(0, world_size as i32, shm_ptr, shm_len);
        let b1 = SimulatedBackend::new(1, world_size as i32, shm_ptr, shm_len);

        let input0: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let input1: [f32; 4] = [10.0, 20.0, 30.0, 40.0];
        let mut out0: [f32; 4] = [0.0; 4];
        let mut out1: [f32; 4] = [0.0; 4];

        // Run from two threads to exercise the barrier.
        std::thread::scope(|s| {
            s.spawn(|| {
                let rc = b0.all_reduce_sum(
                    input0.as_ptr() as *const c_void,
                    out0.as_mut_ptr() as *mut c_void,
                    count,
                    DTYPE_F32,
                    ptr::null_mut(),
                );
                assert_eq!(rc, 0);
            });
            s.spawn(|| {
                let rc = b1.all_reduce_sum(
                    input1.as_ptr() as *const c_void,
                    out1.as_mut_ptr() as *mut c_void,
                    count,
                    DTYPE_F32,
                    ptr::null_mut(),
                );
                assert_eq!(rc, 0);
            });
        });

        assert_eq!(out0, [11.0, 22.0, 33.0, 44.0]);
        assert_eq!(out1, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn simulated_backend_multi_rank_all_gather() {
        let world_size = 2;
        let send_count = 3;
        let slot_bytes = send_count * dtype_byte_width(DTYPE_F64);
        let shm_len = DATA_OFFSET + world_size * slot_bytes;
        let mut shm = vec![0u8; shm_len];
        let shm_ptr = shm.as_mut_ptr();

        let hdr = unsafe { &*(shm_ptr as *const ShmHeader) };
        hdr.generation.store(0, Ordering::Release);
        hdr.arrival.store(0, Ordering::Release);

        let b0 = SimulatedBackend::new(0, world_size as i32, shm_ptr, shm_len);
        let b1 = SimulatedBackend::new(1, world_size as i32, shm_ptr, shm_len);

        let input0: [f64; 3] = [1.0, 2.0, 3.0];
        let input1: [f64; 3] = [4.0, 5.0, 6.0];
        let mut out0: [f64; 6] = [0.0; 6];
        let mut out1: [f64; 6] = [0.0; 6];

        std::thread::scope(|s| {
            s.spawn(|| {
                let rc = b0.all_gather(
                    input0.as_ptr() as *const c_void,
                    out0.as_mut_ptr() as *mut c_void,
                    send_count,
                    DTYPE_F64,
                    ptr::null_mut(),
                );
                assert_eq!(rc, 0);
            });
            s.spawn(|| {
                let rc = b1.all_gather(
                    input1.as_ptr() as *const c_void,
                    out1.as_mut_ptr() as *mut c_void,
                    send_count,
                    DTYPE_F64,
                    ptr::null_mut(),
                );
                assert_eq!(rc, 0);
            });
        });

        assert_eq!(out0, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(out1, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn simulated_backend_single_rank_send_recv() {
        // Allocate a small shm region for single-rank p2p staging.
        let slot_bytes = 4 * std::mem::size_of::<f32>();
        let shm_len = DATA_OFFSET + slot_bytes;
        let mut shm = vec![0u8; shm_len];
        let shm_ptr = shm.as_mut_ptr();

        let backend = SimulatedBackend::new(0, 1, shm_ptr, shm_len);
        let send_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut recv_data = vec![0.0f32; 4];

        let rc = backend.send(
            send_data.as_ptr() as *const c_void,
            4, std::mem::size_of::<f32>(), 0,
        );
        assert_eq!(rc, 0);

        let rc = backend.recv(
            recv_data.as_mut_ptr() as *mut c_void,
            4, std::mem::size_of::<f32>(), 0,
        );
        assert_eq!(rc, 0);
        assert_eq!(recv_data, send_data);
    }

    #[test]
    fn simulated_backend_multi_rank_send_recv() {
        let world_size = 2;
        let count = 4;
        let slot_bytes = count * std::mem::size_of::<f32>();
        let shm_len = DATA_OFFSET + world_size * slot_bytes;
        let mut shm = vec![0u8; shm_len];
        let shm_ptr = shm.as_mut_ptr();

        let hdr = unsafe { &*(shm_ptr as *const ShmHeader) };
        hdr.generation.store(0, Ordering::Release);
        hdr.arrival.store(0, Ordering::Release);

        let b0 = SimulatedBackend::new(0, world_size as i32, shm_ptr, shm_len);
        let b1 = SimulatedBackend::new(1, world_size as i32, shm_ptr, shm_len);

        let send0: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let mut recv1: [f32; 4] = [0.0; 4];

        // Rank 0 sends, rank 1 receives.
        std::thread::scope(|s| {
            s.spawn(|| {
                let rc = b0.send(
                    send0.as_ptr() as *const c_void,
                    count, std::mem::size_of::<f32>(), 1,
                );
                assert_eq!(rc, 0);
            });
            s.spawn(|| {
                let rc = b1.recv(
                    recv1.as_mut_ptr() as *mut c_void,
                    count, std::mem::size_of::<f32>(), 0,
                );
                assert_eq!(rc, 0);
            });
        });

        assert_eq!(recv1, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn simulated_backend_multi_rank_broadcast() {
        let world_size = 2;
        let count = 4;
        let slot_bytes = count * dtype_byte_width(DTYPE_F32);
        let shm_len = DATA_OFFSET + world_size * slot_bytes;
        let mut shm = vec![0u8; shm_len];
        let shm_ptr = shm.as_mut_ptr();

        let hdr = unsafe { &*(shm_ptr as *const ShmHeader) };
        hdr.generation.store(0, Ordering::Release);
        hdr.arrival.store(0, Ordering::Release);

        let b0 = SimulatedBackend::new(0, world_size as i32, shm_ptr, shm_len);
        let b1 = SimulatedBackend::new(1, world_size as i32, shm_ptr, shm_len);

        let mut data0: [f32; 4] = [100.0, 200.0, 300.0, 400.0]; // root data
        let mut data1: [f32; 4] = [0.0; 4]; // will be overwritten

        std::thread::scope(|s| {
            s.spawn(|| {
                let rc = b0.broadcast(
                    data0.as_mut_ptr() as *mut c_void,
                    count,
                    DTYPE_F32,
                    0, // root = rank 0
                    ptr::null_mut(),
                );
                assert_eq!(rc, 0);
            });
            s.spawn(|| {
                let rc = b1.broadcast(
                    data1.as_mut_ptr() as *mut c_void,
                    count,
                    DTYPE_F32,
                    0,
                    ptr::null_mut(),
                );
                assert_eq!(rc, 0);
            });
        });

        assert_eq!(data0, [100.0, 200.0, 300.0, 400.0]);
        assert_eq!(data1, [100.0, 200.0, 300.0, 400.0]);
    }

    // ── D3 (ZeRO): reduce_scatter_sum contract tests ────────────────────

    /// The load-bearing contract: rank r's reduce_scatter output is
    /// BIT-EQUAL to `all_reduce_sum` followed by slicing
    /// `[r*count/ws, (r+1)*count/ws)` — the property that keeps
    /// ZeRO-sharded arithmetic bit-exact against the unsharded baseline.
    #[test]
    fn simulated_backend_reduce_scatter_equals_all_reduce_slice() {
        let world_size = 4usize;
        let count = 8usize;
        let slot_bytes = count * dtype_byte_width(DTYPE_F64);
        let shm_len = DATA_OFFSET + world_size * slot_bytes;
        let mut shm = vec![0u8; shm_len];
        let shm_ptr = shm.as_mut_ptr();
        let hdr = unsafe { &*(shm_ptr as *const ShmHeader) };
        hdr.generation.store(0, Ordering::Release);
        hdr.arrival.store(0, Ordering::Release);

        let backends: Vec<SimulatedBackend> = (0..world_size)
            .map(|r| SimulatedBackend::new(r as i32, world_size as i32, shm_ptr, shm_len))
            .collect();
        let inputs: Vec<[f64; 8]> = (0..world_size)
            .map(|r| std::array::from_fn(|i| (r * 8 + i) as f64 * 0.5 + 0.125))
            .collect();

        // Phase 1: all_reduce on every rank (reference).
        let mut full: Vec<[f64; 8]> = vec![[0.0; 8]; world_size];
        std::thread::scope(|s| {
            for (r, (b, out)) in backends.iter().zip(full.iter_mut()).enumerate() {
                let input = &inputs[r];
                s.spawn(move || {
                    let rc = b.all_reduce_sum(
                        input.as_ptr() as *const c_void,
                        out.as_mut_ptr() as *mut c_void,
                        8,
                        DTYPE_F64,
                        ptr::null_mut(),
                    );
                    assert_eq!(rc, 0);
                });
            }
        });
        // Phase 2: reduce_scatter on every rank.
        let mut scat: Vec<[f64; 2]> = vec![[0.0; 2]; world_size];
        std::thread::scope(|s| {
            for (r, (b, out)) in backends.iter().zip(scat.iter_mut()).enumerate() {
                let input = &inputs[r];
                s.spawn(move || {
                    let rc = b.reduce_scatter_sum(
                        input.as_ptr() as *const c_void,
                        out.as_mut_ptr() as *mut c_void,
                        8,
                        DTYPE_F64,
                        ptr::null_mut(),
                    );
                    assert_eq!(rc, 0);
                });
            }
        });
        for r in 0..world_size {
            let per = 2;
            assert_eq!(
                scat[r].to_vec(),
                full[0][r * per..(r + 1) * per].to_vec(),
                "rank {r} scatter slice != all-reduce slice"
            );
        }
    }

    /// f32 case with order-sensitive values (1e8 + 1 − 1e8 loses the 1.0
    /// unless summed in the fixed rank order both paths share) — pins the
    /// fixed reduction order itself, not just the values.
    #[test]
    fn simulated_backend_reduce_scatter_fixed_order_f32() {
        let world_size = 4usize;
        let count = 4usize;
        let slot_bytes = count * dtype_byte_width(DTYPE_F32);
        let shm_len = DATA_OFFSET + world_size * slot_bytes;
        let mut shm = vec![0u8; shm_len];
        let shm_ptr = shm.as_mut_ptr();
        let hdr = unsafe { &*(shm_ptr as *const ShmHeader) };
        hdr.generation.store(0, Ordering::Release);
        hdr.arrival.store(0, Ordering::Release);

        let backends: Vec<SimulatedBackend> = (0..world_size)
            .map(|r| SimulatedBackend::new(r as i32, world_size as i32, shm_ptr, shm_len))
            .collect();
        // Element 0 (rank-order contributions): 1e8, 1.0, -1e8, 1.0 —
        // fixed-order f32 sum = (1e8 + 1.0) − 1e8 + 1.0 = 1.0 exactly
        // (1e8+1.0 rounds to 1e8). Any other order changes the result.
        let inputs: Vec<[f32; 4]> = vec![
            [1e8, 5.0, 0.25, 7.0],
            [1.0, 5.0, 0.25, 7.0],
            [-1e8, 5.0, 0.25, 7.0],
            [1.0, 5.0, 0.25, 7.0],
        ];
        let mut scat: Vec<[f32; 1]> = vec![[0.0]; world_size];
        std::thread::scope(|s| {
            for (r, (b, out)) in backends.iter().zip(scat.iter_mut()).enumerate() {
                let input = &inputs[r];
                s.spawn(move || {
                    let rc = b.reduce_scatter_sum(
                        input.as_ptr() as *const c_void,
                        out.as_mut_ptr() as *mut c_void,
                        4,
                        DTYPE_F32,
                        ptr::null_mut(),
                    );
                    assert_eq!(rc, 0);
                });
            }
        });
        assert_eq!(scat[0][0], 1.0, "fixed rank-order f32 sum drifted");
        assert_eq!(scat[1][0], 20.0);
        assert_eq!(scat[2][0], 1.0);
        assert_eq!(scat[3][0], 28.0);
    }

    /// Non-divisible count refuses with -3.
    #[test]
    fn simulated_backend_reduce_scatter_refuses_indivisible() {
        let b = SimulatedBackend::new(0, 1, ptr::null_mut(), 0);
        let input: [f32; 3] = [1.0, 2.0, 3.0];
        let mut out: [f32; 3] = [0.0; 3];
        // world_size=1: divisible trivially — exercise via a 2-rank
        // backend with no shm (never reaches the slot copy on the guard).
        let b2 = SimulatedBackend::new(0, 2, ptr::null_mut(), 0);
        let rc = b2.reduce_scatter_sum(
            input.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            3,
            DTYPE_F32,
            ptr::null_mut(),
        );
        assert_eq!(rc, -3);
        // Single-rank fast path still copies.
        let rc1 = b.reduce_scatter_sum(
            input.as_ptr() as *const c_void,
            out.as_mut_ptr() as *mut c_void,
            3,
            DTYPE_F32,
            ptr::null_mut(),
        );
        assert_eq!(rc1, 0);
        assert_eq!(out, [1.0, 2.0, 3.0]);
    }
}
