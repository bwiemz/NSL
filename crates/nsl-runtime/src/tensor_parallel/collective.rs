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

    /// Spin-barrier using double-buffered generation counter.
    ///
    /// All ranks increment `arrival`; the last rank to arrive bumps `generation`
    /// and resets `arrival` for the next round.  Other ranks spin on `generation`.
    fn spin_barrier(&self) {
        let hdr = self.header();
        let gen_before = hdr.generation.load(Ordering::Acquire);

        let prev = hdr.arrival.fetch_add(1, Ordering::AcqRel);
        if prev + 1 == self.world_size as u32 {
            // Last to arrive — reset arrival and bump generation.
            hdr.arrival.store(0, Ordering::Release);
            hdr.generation.store(gen_before.wrapping_add(1), Ordering::Release);
        } else {
            // Spin until generation advances.
            while hdr.generation.load(Ordering::Acquire) == gen_before {
                std::hint::spin_loop();
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
}
