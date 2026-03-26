//! M54b: Unikernel runtime — bare-metal execution environment.
//!
//! Provides the core runtime infrastructure for NSL unikernel deployment:
//! - Bump allocator for the model memory pool
//! - Serial console output (COM1 port 0x3F8) for diagnostics
//! - Boot config parsing
//! - PCI device discovery (for GPU init)
//!
//! This module is compiled into the unikernel binary and runs without an OS.
//! All functions use `no_std`-compatible code paths.

pub mod gpu_init;

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};

// ---------------------------------------------------------------------------
// Serial console (COM1 at 0x3F8)
// ---------------------------------------------------------------------------

/// COM1 I/O port base address.
const COM1_PORT: u16 = 0x3F8;

/// Write a byte to the serial console (COM1).
///
/// In bare-metal mode, this uses x86 `outb` instructions.
/// In hosted mode (testing), this writes to stderr.
#[inline]
pub fn serial_putc(byte: u8) {
    #[cfg(target_os = "none")]
    unsafe {
        // Wait for transmit holding register to be empty
        loop {
            let status: u8;
            core::arch::asm!("in al, dx", out("al") status, in("dx") COM1_PORT + 5);
            if status & 0x20 != 0 { break; }
        }
        // Write byte
        core::arch::asm!("out dx, al", in("al") byte, in("dx") COM1_PORT);
    }

    #[cfg(not(target_os = "none"))]
    {
        // Hosted mode: write to stderr
        let _ = byte;
    }
}

/// Write a string to the serial console.
pub fn serial_puts(s: &str) {
    for byte in s.bytes() {
        serial_putc(byte);
    }
}

/// Initialize the serial port (COM1).
///
/// Sets 115200 baud, 8N1 (8 data bits, no parity, 1 stop bit).
pub fn serial_init() {
    #[cfg(target_os = "none")]
    unsafe {
        // Disable interrupts
        port_outb(COM1_PORT + 1, 0x00);
        // Enable DLAB (set baud rate divisor)
        port_outb(COM1_PORT + 3, 0x80);
        // Set divisor to 1 (115200 baud)
        port_outb(COM1_PORT + 0, 0x01); // low byte
        port_outb(COM1_PORT + 1, 0x00); // high byte
        // 8 bits, no parity, one stop bit
        port_outb(COM1_PORT + 3, 0x03);
        // Enable FIFO, clear, 14-byte threshold
        port_outb(COM1_PORT + 2, 0xC7);
        // IRQs enabled, RTS/DSR set
        port_outb(COM1_PORT + 4, 0x0B);
    }

    #[cfg(not(target_os = "none"))]
    {
        // Nothing to do in hosted mode
    }
}

#[cfg(target_os = "none")]
unsafe fn port_outb(port: u16, val: u8) {
    core::arch::asm!("out dx, al", in("al") val, in("dx") port);
}

// ---------------------------------------------------------------------------
// Bump allocator
// ---------------------------------------------------------------------------

/// A simple bump allocator for the unikernel memory pool.
///
/// Thread-safe via atomic pointer bump. Does not support deallocation
/// (all memory freed when the unikernel exits). This is appropriate for
/// model weight allocation where all weights are loaded once at boot.
pub struct BumpAllocator {
    /// Start of the memory region.
    base: u64,
    /// Current allocation pointer (bumps upward).
    next: AtomicU64,
    /// End of the memory region (exclusive).
    limit: u64,
    /// Whether the allocator has been initialized.
    initialized: AtomicBool,
}

impl BumpAllocator {
    /// Create a new uninitialized bump allocator.
    pub const fn uninit() -> Self {
        Self {
            base: 0,
            next: AtomicU64::new(0),
            limit: 0,
            initialized: AtomicBool::new(false),
        }
    }

    /// Initialize the allocator with a memory region [base, base + size).
    pub fn init(&mut self, base: u64, size: u64) {
        self.base = base;
        self.next.store(base, Ordering::Release);
        self.limit = base + size;
        self.initialized.store(true, Ordering::Release);
    }

    /// Allocate `size` bytes with the given alignment.
    ///
    /// Returns the physical address of the allocation, or 0 if out of memory.
    pub fn alloc(&self, size: u64, align: u64) -> u64 {
        if !self.initialized.load(Ordering::Acquire) {
            return 0;
        }

        loop {
            let current = self.next.load(Ordering::Acquire);
            // Align up
            let aligned = (current + align - 1) & !(align - 1);
            let new_next = aligned + size;

            if new_next > self.limit {
                return 0; // out of memory
            }

            if self.next.compare_exchange_weak(
                current, new_next,
                Ordering::AcqRel, Ordering::Relaxed,
            ).is_ok() {
                return aligned;
            }
            // CAS failed — retry
        }
    }

    /// Total capacity in bytes.
    pub fn capacity(&self) -> u64 {
        self.limit - self.base
    }

    /// Used bytes so far.
    pub fn used(&self) -> u64 {
        self.next.load(Ordering::Acquire) - self.base
    }

    /// Remaining free bytes.
    pub fn free(&self) -> u64 {
        self.limit - self.next.load(Ordering::Acquire)
    }

    /// Reset the allocator (free all memory).
    pub fn reset(&self) {
        self.next.store(self.base, Ordering::Release);
    }
}

// Global allocators for different memory pools
static MODEL_POOL: Mutex<BumpAllocator> = Mutex::new(BumpAllocator::uninit());
static KV_CACHE_POOL: Mutex<BumpAllocator> = Mutex::new(BumpAllocator::uninit());

// ---------------------------------------------------------------------------
// Boot config parsing
// ---------------------------------------------------------------------------

/// Parsed boot configuration (extracted from embedded JSON).
#[derive(Debug, Clone)]
pub struct BootConfig {
    pub listen_host: [u8; 4],
    pub listen_port: u16,
    pub memory_bytes: u64,
    pub model_pool_bytes: u64,
    pub kv_cache_pool_bytes: u64,
    pub weight_source: String,
    pub gpu_init: String,
}

impl Default for BootConfig {
    fn default() -> Self {
        Self {
            listen_host: [0, 0, 0, 0],
            listen_port: 8080,
            memory_bytes: 0,
            model_pool_bytes: 0,
            kv_cache_pool_bytes: 0,
            weight_source: "embedded".to_string(),
            gpu_init: "vfio-passthrough".to_string(),
        }
    }
}

/// Parse a boot config JSON string into a BootConfig.
///
/// Uses a simple hand-written parser (no serde dependency in bare-metal).
pub fn parse_boot_config(json: &str) -> BootConfig {
    let mut config = BootConfig::default();

    // Simple key-value extraction from JSON
    for line in json.lines() {
        let line = line.trim().trim_matches(',');
        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().trim_matches('"');
            let value = value.trim().trim_matches('"').trim_matches(',');

            match key {
                "listen_port" => {
                    if let Ok(port) = value.parse::<u16>() {
                        config.listen_port = port;
                    }
                }
                "listen_host" => {
                    let octets: Vec<u8> = value.split('.')
                        .filter_map(|o| o.trim_matches('"').parse().ok())
                        .collect();
                    if octets.len() == 4 {
                        config.listen_host = [octets[0], octets[1], octets[2], octets[3]];
                    }
                }
                "memory_bytes" => {
                    if let Ok(bytes) = value.parse::<u64>() {
                        config.memory_bytes = bytes;
                    }
                }
                "weight_source" => {
                    config.weight_source = value.trim_matches('"').to_string();
                }
                "gpu_init" => {
                    config.gpu_init = value.trim_matches('"').to_string();
                }
                "model_pool" => {
                    if let Ok(bytes) = value.parse::<u64>() {
                        config.model_pool_bytes = bytes;
                    }
                }
                "kv_cache_pool" => {
                    if let Ok(bytes) = value.parse::<u64>() {
                        config.kv_cache_pool_bytes = bytes;
                    }
                }
                _ => {}
            }
        }
    }
    config
}

// ---------------------------------------------------------------------------
// FFI entry points
// ---------------------------------------------------------------------------

/// Initialize the unikernel runtime.
///
/// Called from the boot stub after long mode is established.
/// Sets up memory pools, serial console, and GPU.
///
/// `config_json_ptr`: pointer to null-terminated boot config JSON string
/// `config_json_len`: length of the JSON string
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_unikernel_init(config_json_ptr: i64, config_json_len: i64) -> i64 {
    serial_init();
    serial_puts("[nsl] Unikernel runtime initializing...\n");

    let config = if config_json_ptr != 0 && config_json_len > 0 {
        let json = unsafe {
            let ptr = config_json_ptr as *const u8;
            let len = config_json_len as usize;
            std::str::from_utf8_unchecked(std::slice::from_raw_parts(ptr, len))
        };
        parse_boot_config(json)
    } else {
        BootConfig::default()
    };

    serial_puts("[nsl] Boot config parsed\n");

    // Initialize memory pools if sizes are known
    if config.model_pool_bytes > 0 {
        if let Ok(mut pool) = MODEL_POOL.lock() {
            // In bare-metal, base address comes from linker script (__heap_start)
            // In hosted mode, we allocate from the system heap
            let base = crate::memory::checked_alloc(config.model_pool_bytes as usize) as u64;
            pool.init(base, config.model_pool_bytes);
            serial_puts("[nsl] Model pool initialized\n");
        }
    }

    if config.kv_cache_pool_bytes > 0 {
        if let Ok(mut pool) = KV_CACHE_POOL.lock() {
            let base = crate::memory::checked_alloc(config.kv_cache_pool_bytes as usize) as u64;
            pool.init(base, config.kv_cache_pool_bytes);
            serial_puts("[nsl] KV-cache pool initialized\n");
        }
    }

    serial_puts("[nsl] Unikernel runtime ready\n");
    0
}

/// Allocate from the model memory pool.
///
/// Returns physical address or 0 on failure.
#[no_mangle]
pub extern "C" fn nsl_unikernel_model_alloc(size: i64, align: i64) -> i64 {
    if let Ok(pool) = MODEL_POOL.lock() {
        pool.alloc(size as u64, align.max(1) as u64) as i64
    } else {
        0
    }
}

/// Allocate from the KV-cache memory pool.
#[no_mangle]
pub extern "C" fn nsl_unikernel_kv_alloc(size: i64, align: i64) -> i64 {
    if let Ok(pool) = KV_CACHE_POOL.lock() {
        pool.alloc(size as u64, align.max(1) as u64) as i64
    } else {
        0
    }
}

/// Get model pool usage stats.
/// Returns (used_bytes << 32) | free_bytes, or -1 on error.
#[no_mangle]
pub extern "C" fn nsl_unikernel_model_pool_stats() -> i64 {
    if let Ok(pool) = MODEL_POOL.lock() {
        let used = pool.used() as i64;
        let free = pool.free() as i64;
        (used << 32) | (free & 0xFFFF_FFFF)
    } else {
        -1
    }
}

/// Shut down the unikernel runtime.
#[no_mangle]
pub extern "C" fn nsl_unikernel_shutdown() -> i64 {
    serial_puts("[nsl] Unikernel shutting down\n");

    // Reset pools
    if let Ok(pool) = MODEL_POOL.lock() {
        pool.reset();
    }
    if let Ok(pool) = KV_CACHE_POOL.lock() {
        pool.reset();
    }

    serial_puts("[nsl] Goodbye.\n");

    // In bare-metal, this would trigger a VM exit or triple fault
    #[cfg(target_os = "none")]
    unsafe {
        // ACPI shutdown: write to port 0x604 (QEMU specific)
        core::arch::asm!("out dx, ax", in("ax") 0x2000u16, in("dx") 0x604u16);
    }

    0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_allocator_basic() {
        let mut alloc = BumpAllocator::uninit();
        alloc.init(0x1000, 4096);

        assert_eq!(alloc.capacity(), 4096);
        assert_eq!(alloc.used(), 0);
        assert_eq!(alloc.free(), 4096);

        let addr1 = alloc.alloc(64, 8);
        assert_eq!(addr1, 0x1000); // base aligned to 8
        assert_eq!(alloc.used(), 64);

        let addr2 = alloc.alloc(128, 16);
        assert_eq!(addr2, 0x1040); // 0x1000 + 64, already 16-aligned
        assert_eq!(alloc.used(), 192);
    }

    #[test]
    fn test_bump_allocator_alignment() {
        let mut alloc = BumpAllocator::uninit();
        alloc.init(0x1001, 4096); // misaligned base

        let addr = alloc.alloc(32, 16);
        assert_eq!(addr, 0x1010); // aligned up to 16
        assert_eq!(addr % 16, 0);
    }

    #[test]
    fn test_bump_allocator_oom() {
        let mut alloc = BumpAllocator::uninit();
        alloc.init(0x1000, 64);

        let addr1 = alloc.alloc(32, 1);
        assert_ne!(addr1, 0);
        let addr2 = alloc.alloc(32, 1);
        assert_ne!(addr2, 0);
        let addr3 = alloc.alloc(1, 1); // out of memory
        assert_eq!(addr3, 0);
    }

    #[test]
    fn test_bump_allocator_reset() {
        let mut alloc = BumpAllocator::uninit();
        alloc.init(0x1000, 4096);

        alloc.alloc(1024, 1);
        assert_eq!(alloc.used(), 1024);

        alloc.reset();
        assert_eq!(alloc.used(), 0);
        assert_eq!(alloc.free(), 4096);
    }

    #[test]
    fn test_bump_allocator_uninit_returns_zero() {
        let alloc = BumpAllocator::uninit();
        assert_eq!(alloc.alloc(64, 8), 0);
    }

    #[test]
    fn test_parse_boot_config_basic() {
        let json = r#"{
  "target": "kvm",
  "listen_host": "10.0.0.1",
  "listen_port": 9090,
  "weight_source": "embedded",
  "gpu_init": "vfio-passthrough",
  "memory_bytes": 17179869184
}"#;
        let config = parse_boot_config(json);
        assert_eq!(config.listen_host, [10, 0, 0, 1]);
        assert_eq!(config.listen_port, 9090);
        assert_eq!(config.memory_bytes, 17179869184);
        assert_eq!(config.weight_source, "embedded");
        assert_eq!(config.gpu_init, "vfio-passthrough");
    }

    #[test]
    fn test_parse_boot_config_with_pools() {
        let json = r#"{
  "listen_port": 8080,
  "model_pool": 1073741824,
  "kv_cache_pool": 536870912
}"#;
        let config = parse_boot_config(json);
        assert_eq!(config.model_pool_bytes, 1073741824); // 1 GiB
        assert_eq!(config.kv_cache_pool_bytes, 536870912); // 512 MiB
    }

    #[test]
    fn test_parse_boot_config_empty() {
        let config = parse_boot_config("{}");
        assert_eq!(config.listen_port, 8080); // default
        assert_eq!(config.memory_bytes, 0);
    }

    #[test]
    fn test_ffi_init_shutdown() {
        // Test with no config
        assert_eq!(nsl_unikernel_init(0, 0), 0);
        assert_eq!(nsl_unikernel_shutdown(), 0);
    }
}
