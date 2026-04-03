//! M54a: Unikernel configuration and codegen support.
//!
//! This module provides the configuration types and generation functions
//! for bare-metal unikernel deployment. Boot-stub generation and image-builder
//! support now live in `unikernel_boot.rs`; this module still owns the config,
//! linker script, boot config JSON, and memory layout calculations that feed
//! that build pipeline.

use std::fmt;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Target hypervisor for the unikernel image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HypervisorTarget {
    /// KVM / QEMU with direct kernel boot (`-kernel` flag).
    Kvm,
    /// AWS Firecracker microVM.
    Firecracker,
}

impl fmt::Display for HypervisorTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Kvm => write!(f, "kvm"),
            Self::Firecracker => write!(f, "firecracker"),
        }
    }
}

/// How model weights are provided to the unikernel at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightSource {
    /// Weights embedded in `.rodata` section (small models, from M24).
    Embedded,
    /// Weights loaded from virtio-blk device at boot.
    Disk,
    /// Weights fetched via network (PXE / HTTP).
    Network,
}

impl fmt::Display for WeightSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Embedded => write!(f, "embedded"),
            Self::Disk => write!(f, "disk"),
            Self::Network => write!(f, "network"),
        }
    }
}

/// GPU initialization strategy inside the unikernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuInitStrategy {
    /// VFIO passthrough: hypervisor assigns GPU to guest, guest uses CUDA Driver API.
    VfioPassthrough,
    /// Direct register init: bare-metal GPU access without any driver (experimental).
    DirectRegister,
}

impl fmt::Display for GpuInitStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VfioPassthrough => write!(f, "vfio-passthrough"),
            Self::DirectRegister => write!(f, "direct-register"),
        }
    }
}

/// Boot protocol for the unikernel ELF.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootProtocol {
    /// Multiboot2 — works with GRUB and some hypervisors.
    Multiboot2,
    /// Linux boot protocol (bzImage header) — works with QEMU `-kernel`.
    LinuxBoot,
}

impl fmt::Display for BootProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Multiboot2 => write!(f, "multiboot2"),
            Self::LinuxBoot => write!(f, "linux-boot"),
        }
    }
}

// ---------------------------------------------------------------------------
// ListenAddr
// ---------------------------------------------------------------------------

/// Network listen address for the unikernel HTTP server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListenAddr {
    pub host: [u8; 4],
    pub port: u16,
}

impl Default for ListenAddr {
    fn default() -> Self {
        Self {
            host: [0, 0, 0, 0],
            port: 8080,
        }
    }
}

impl fmt::Display for ListenAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{}.{}.{}:{}",
            self.host[0], self.host[1], self.host[2], self.host[3], self.port
        )
    }
}

/// Parse a `host:port` string into a [`ListenAddr`].
///
/// Accepts `"0.0.0.0:8080"` or `"10.0.0.1:9090"` style strings.
pub fn parse_listen_addr(s: &str) -> Result<ListenAddr, String> {
    let (host_str, port_str) = s
        .rsplit_once(':')
        .ok_or_else(|| format!("expected host:port, got '{s}'"))?;

    let port: u16 = port_str
        .parse()
        .map_err(|_| format!("invalid port '{port_str}'"))?;

    let octets: Vec<u8> = host_str
        .split('.')
        .map(|o| o.parse::<u8>().map_err(|_| format!("invalid octet '{o}'")))
        .collect::<Result<Vec<_>, _>>()?;

    if octets.len() != 4 {
        return Err(format!(
            "expected 4 octets in host, got {}",
            octets.len()
        ));
    }

    Ok(ListenAddr {
        host: [octets[0], octets[1], octets[2], octets[3]],
        port,
    })
}

// ---------------------------------------------------------------------------
// MemoryLayout
// ---------------------------------------------------------------------------

/// Physical memory layout for the unikernel.
///
/// All sizes are in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryLayout {
    /// Total physical RAM available to the unikernel.
    pub total_bytes: u64,
    /// Memory reserved for kernel structures (GDT, IDT, page tables, stack).
    pub kernel_reserved: u64,
    /// Slab for model weight storage (host-side mirror or embedded).
    pub model_pool: u64,
    /// Slab for KV-cache and activations.
    pub kv_cache_pool: u64,
    /// Remaining memory for general-purpose heap allocation.
    pub heap_available: u64,
}

/// Default kernel reservation: 16 MiB (page tables, GDT, IDT, stacks, boot info).
const KERNEL_RESERVED_BYTES: u64 = 16 * 1024 * 1024;

/// Compute a [`MemoryLayout`] given the total available RAM.
///
/// Allocation strategy:
///   1. First 16 MiB reserved for kernel structures.
///   2. 60% of remaining RAM → model weight pool.
///   3. 30% of remaining RAM → KV-cache / activation pool.
///   4. 10% of remaining RAM → general heap.
///
/// If `total_bytes` is less than or equal to the kernel reservation, the pools
/// are all zero and only kernel_reserved is set.
pub fn compute_memory_layout(total_bytes: u64) -> MemoryLayout {
    if total_bytes <= KERNEL_RESERVED_BYTES {
        return MemoryLayout {
            total_bytes,
            kernel_reserved: total_bytes,
            model_pool: 0,
            kv_cache_pool: 0,
            heap_available: 0,
        };
    }

    let usable = total_bytes - KERNEL_RESERVED_BYTES;
    // Use integer percentages to avoid floating-point.
    let model_pool = usable * 60 / 100;
    let kv_cache_pool = usable * 30 / 100;
    let heap_available = usable - model_pool - kv_cache_pool;

    MemoryLayout {
        total_bytes,
        kernel_reserved: KERNEL_RESERVED_BYTES,
        model_pool,
        kv_cache_pool,
        heap_available,
    }
}

/// Parse a human-readable memory size string (e.g., `"16G"`, `"512M"`, `"2048"`)
/// into bytes.
pub fn parse_memory_size(s: &str) -> Result<u64, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty memory size string".to_string());
    }

    let (digits, suffix) = if s.ends_with('G') || s.ends_with('g') {
        (&s[..s.len() - 1], 1024u64 * 1024 * 1024)
    } else if s.ends_with("GB") || s.ends_with("gb") || s.ends_with("Gb") {
        (&s[..s.len() - 2], 1024u64 * 1024 * 1024)
    } else if s.ends_with('M') || s.ends_with('m') {
        (&s[..s.len() - 1], 1024u64 * 1024)
    } else if s.ends_with("MB") || s.ends_with("mb") || s.ends_with("Mb") {
        (&s[..s.len() - 2], 1024u64 * 1024)
    } else {
        // Bare number = bytes
        (s, 1u64)
    };

    let value: u64 = digits
        .parse()
        .map_err(|_| format!("invalid number '{digits}'"))?;

    Ok(value * suffix)
}

// ---------------------------------------------------------------------------
// UnikernelConfig
// ---------------------------------------------------------------------------

/// Complete configuration for unikernel binary generation.
#[derive(Debug, Clone)]
pub struct UnikernelConfig {
    /// Hypervisor target.
    pub target: HypervisorTarget,
    /// Network listen address.
    pub listen_addr: ListenAddr,
    /// Total physical memory in bytes (0 = auto-detect at boot).
    pub memory_bytes: u64,
    /// How model weights are provided.
    pub weight_source: WeightSource,
    /// GPU initialization strategy.
    pub gpu_init: GpuInitStrategy,
    /// Boot protocol.
    pub boot_protocol: BootProtocol,
}

impl Default for UnikernelConfig {
    fn default() -> Self {
        Self {
            target: HypervisorTarget::Kvm,
            listen_addr: ListenAddr::default(),
            memory_bytes: 0,
            weight_source: WeightSource::Embedded,
            gpu_init: GpuInitStrategy::VfioPassthrough,
            boot_protocol: BootProtocol::LinuxBoot,
        }
    }
}

impl UnikernelConfig {
    /// Compute the memory layout for this configuration.
    ///
    /// Returns `None` if `memory_bytes` is 0 (auto-detect mode).
    pub fn memory_layout(&self) -> Option<MemoryLayout> {
        if self.memory_bytes == 0 {
            None
        } else {
            Some(compute_memory_layout(self.memory_bytes))
        }
    }

    /// Print a human-readable summary of the unikernel configuration to stderr.
    pub fn print_summary(&self) {
        eprintln!("[nsl] Unikernel build configured:");
        eprintln!("  target:        {}", self.target);
        eprintln!("  boot protocol: {}", self.boot_protocol);
        eprintln!("  listen:        {}", self.listen_addr);
        eprintln!("  weights:       {}", self.weight_source);
        eprintln!("  gpu init:      {}", self.gpu_init);

        if let Some(layout) = self.memory_layout() {
            eprintln!("  memory layout:");
            eprintln!(
                "    total:          {} MiB",
                layout.total_bytes / (1024 * 1024)
            );
            eprintln!(
                "    kernel reserved: {} MiB",
                layout.kernel_reserved / (1024 * 1024)
            );
            eprintln!(
                "    model pool:     {} MiB",
                layout.model_pool / (1024 * 1024)
            );
            eprintln!(
                "    kv-cache pool:  {} MiB",
                layout.kv_cache_pool / (1024 * 1024)
            );
            eprintln!(
                "    heap available: {} MiB",
                layout.heap_available / (1024 * 1024)
            );
        } else {
            eprintln!("  memory:        auto-detect at boot");
        }
    }
}

// ---------------------------------------------------------------------------
// Linker script generation
// ---------------------------------------------------------------------------

/// Generate a linker script string for the unikernel ELF layout.
///
/// The entry address is chosen based on the boot protocol.
/// The script places the boot stub first in `.text`, embeds weight data
/// in a dedicated `.weights` section, and reserves a stack region.
pub fn generate_linker_script(config: &UnikernelConfig) -> String {
    let entry_addr = match config.boot_protocol {
        BootProtocol::LinuxBoot => "0x100000",    // 1 MiB (Linux boot protocol)
        BootProtocol::Multiboot2 => "0x100000",   // 1 MiB (Multiboot2)
    };

    format!(
        r#"/* NSL unikernel linker script — generated by nsl build --unikernel */
ENTRY(_start)

SECTIONS {{
    . = {entry_addr};

    .text : {{
        *(.text.boot)       /* Boot stub must be first */
        *(.text .text.*)    /* Model code + runtime */
    }}

    .rodata : ALIGN(4096) {{
        *(.rodata .rodata.*)
        __weights_start = .;
        *(.weights)
        __weights_end = .;
    }}

    .data : ALIGN(4096) {{
        *(.data .data.*)
    }}

    .bss : ALIGN(4096) {{
        __bss_start = .;
        *(.bss .bss.*)
        __bss_end = .;
    }}

    /* Stack: 1 MiB */
    . = ALIGN(4096);
    __stack_bottom = .;
    . += 0x100000;
    __stack_top = .;

    /* Heap: remaining memory managed by slab allocator */
    . = ALIGN(4096);
    __heap_start = .;

    /DISCARD/ : {{
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }}
}}
"#
    )
}

// ---------------------------------------------------------------------------
// Boot config generation
// ---------------------------------------------------------------------------

/// Generate a JSON boot configuration string that a boot stub would read.
///
/// The JSON is deterministic (fields in a fixed order) so that builds are
/// reproducible.
pub fn generate_boot_config(config: &UnikernelConfig) -> String {
    let layout_json = if let Some(layout) = config.memory_layout() {
        format!(
            r#"  "memory_layout": {{
    "total_bytes": {},
    "kernel_reserved": {},
    "model_pool": {},
    "kv_cache_pool": {},
    "heap_available": {}
  }}"#,
            layout.total_bytes,
            layout.kernel_reserved,
            layout.model_pool,
            layout.kv_cache_pool,
            layout.heap_available,
        )
    } else {
        r#"  "memory_layout": null"#.to_string()
    };

    format!(
        r#"{{
  "target": "{}",
  "boot_protocol": "{}",
  "listen_host": "{}.{}.{}.{}",
  "listen_port": {},
  "weight_source": "{}",
  "gpu_init": "{}",
  "memory_bytes": {},
{layout_json}
}}"#,
        config.target,
        config.boot_protocol,
        config.listen_addr.host[0],
        config.listen_addr.host[1],
        config.listen_addr.host[2],
        config.listen_addr.host[3],
        config.listen_addr.port,
        config.weight_source,
        config.gpu_init,
        config.memory_bytes,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let cfg = UnikernelConfig::default();
        assert_eq!(cfg.target, HypervisorTarget::Kvm);
        assert_eq!(cfg.listen_addr.port, 8080);
        assert_eq!(cfg.listen_addr.host, [0, 0, 0, 0]);
        assert_eq!(cfg.memory_bytes, 0);
        assert_eq!(cfg.weight_source, WeightSource::Embedded);
        assert_eq!(cfg.gpu_init, GpuInitStrategy::VfioPassthrough);
        assert_eq!(cfg.boot_protocol, BootProtocol::LinuxBoot);
    }

    #[test]
    fn test_parse_listen_addr_valid() {
        let addr = parse_listen_addr("0.0.0.0:8080").unwrap();
        assert_eq!(addr.host, [0, 0, 0, 0]);
        assert_eq!(addr.port, 8080);

        let addr2 = parse_listen_addr("10.0.0.1:9090").unwrap();
        assert_eq!(addr2.host, [10, 0, 0, 1]);
        assert_eq!(addr2.port, 9090);

        let addr3 = parse_listen_addr("192.168.1.100:443").unwrap();
        assert_eq!(addr3.host, [192, 168, 1, 100]);
        assert_eq!(addr3.port, 443);
    }

    #[test]
    fn test_parse_listen_addr_invalid() {
        assert!(parse_listen_addr("").is_err());
        assert!(parse_listen_addr("no-colon").is_err());
        assert!(parse_listen_addr("1.2.3:80").is_err()); // only 3 octets
        assert!(parse_listen_addr("1.2.3.4.5:80").is_err()); // 5 octets
        assert!(parse_listen_addr("1.2.3.4:99999").is_err()); // port overflow
        assert!(parse_listen_addr("256.0.0.1:80").is_err()); // octet overflow
    }

    #[test]
    fn test_parse_memory_size() {
        assert_eq!(parse_memory_size("16G").unwrap(), 16 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("16g").unwrap(), 16 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("16GB").unwrap(), 16 * 1024 * 1024 * 1024);
        assert_eq!(parse_memory_size("512M").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_memory_size("512m").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_memory_size("512MB").unwrap(), 512 * 1024 * 1024);
        assert_eq!(parse_memory_size("65536").unwrap(), 65536);
        assert!(parse_memory_size("").is_err());
        assert!(parse_memory_size("abc").is_err());
    }

    #[test]
    fn test_compute_memory_layout_16g() {
        let total = 16u64 * 1024 * 1024 * 1024; // 16 GiB
        let layout = compute_memory_layout(total);

        assert_eq!(layout.total_bytes, total);
        assert_eq!(layout.kernel_reserved, KERNEL_RESERVED_BYTES);

        let usable = total - KERNEL_RESERVED_BYTES;
        assert_eq!(layout.model_pool, usable * 60 / 100);
        assert_eq!(layout.kv_cache_pool, usable * 30 / 100);
        assert_eq!(layout.heap_available, usable - layout.model_pool - layout.kv_cache_pool);

        // All pools sum to total
        assert_eq!(
            layout.kernel_reserved + layout.model_pool + layout.kv_cache_pool + layout.heap_available,
            total
        );
    }

    #[test]
    fn test_compute_memory_layout_tiny() {
        // Less than kernel reservation
        let total = 8 * 1024 * 1024; // 8 MiB
        let layout = compute_memory_layout(total);
        assert_eq!(layout.total_bytes, total);
        assert_eq!(layout.kernel_reserved, total);
        assert_eq!(layout.model_pool, 0);
        assert_eq!(layout.kv_cache_pool, 0);
        assert_eq!(layout.heap_available, 0);
    }

    #[test]
    fn test_compute_memory_layout_exact_reservation() {
        let layout = compute_memory_layout(KERNEL_RESERVED_BYTES);
        assert_eq!(layout.kernel_reserved, KERNEL_RESERVED_BYTES);
        assert_eq!(layout.model_pool, 0);
        assert_eq!(layout.kv_cache_pool, 0);
        assert_eq!(layout.heap_available, 0);
    }

    #[test]
    fn test_generate_linker_script_linux_boot() {
        let cfg = UnikernelConfig {
            boot_protocol: BootProtocol::LinuxBoot,
            ..Default::default()
        };
        let script = generate_linker_script(&cfg);
        assert!(script.contains("ENTRY(_start)"));
        assert!(script.contains(". = 0x100000;"));
        assert!(script.contains("*(.text.boot)"));
        assert!(script.contains("__weights_start"));
        assert!(script.contains("__weights_end"));
        assert!(script.contains("__stack_top"));
        assert!(script.contains("__heap_start"));
        assert!(script.contains("/DISCARD/"));
    }

    #[test]
    fn test_generate_linker_script_multiboot2() {
        let cfg = UnikernelConfig {
            boot_protocol: BootProtocol::Multiboot2,
            ..Default::default()
        };
        let script = generate_linker_script(&cfg);
        assert!(script.contains(". = 0x100000;"));
    }

    #[test]
    fn test_generate_boot_config_with_memory() {
        let cfg = UnikernelConfig {
            target: HypervisorTarget::Firecracker,
            listen_addr: ListenAddr {
                host: [10, 0, 0, 1],
                port: 9090,
            },
            memory_bytes: 16 * 1024 * 1024 * 1024, // 16 GiB
            weight_source: WeightSource::Disk,
            gpu_init: GpuInitStrategy::DirectRegister,
            boot_protocol: BootProtocol::Multiboot2,
        };

        let json = generate_boot_config(&cfg);
        assert!(json.contains(r#""target": "firecracker""#));
        assert!(json.contains(r#""boot_protocol": "multiboot2""#));
        assert!(json.contains(r#""listen_host": "10.0.0.1""#));
        assert!(json.contains(r#""listen_port": 9090"#));
        assert!(json.contains(r#""weight_source": "disk""#));
        assert!(json.contains(r#""gpu_init": "direct-register""#));
        assert!(json.contains(r#""memory_layout": {"#));
        assert!(json.contains(r#""kernel_reserved":"#));
        assert!(json.contains(r#""model_pool":"#));
    }

    #[test]
    fn test_generate_boot_config_auto_memory() {
        let cfg = UnikernelConfig::default(); // memory_bytes = 0
        let json = generate_boot_config(&cfg);
        assert!(json.contains(r#""memory_layout": null"#));
        assert!(json.contains(r#""memory_bytes": 0"#));
    }

    #[test]
    fn test_memory_layout_method() {
        let cfg = UnikernelConfig::default();
        assert!(cfg.memory_layout().is_none());

        let cfg2 = UnikernelConfig {
            memory_bytes: 1024 * 1024 * 1024, // 1 GiB
            ..Default::default()
        };
        let layout = cfg2.memory_layout().unwrap();
        assert_eq!(layout.total_bytes, 1024 * 1024 * 1024);
    }

    #[test]
    fn test_display_impls() {
        assert_eq!(format!("{}", HypervisorTarget::Kvm), "kvm");
        assert_eq!(format!("{}", HypervisorTarget::Firecracker), "firecracker");
        assert_eq!(format!("{}", WeightSource::Embedded), "embedded");
        assert_eq!(format!("{}", WeightSource::Disk), "disk");
        assert_eq!(format!("{}", WeightSource::Network), "network");
        assert_eq!(format!("{}", GpuInitStrategy::VfioPassthrough), "vfio-passthrough");
        assert_eq!(format!("{}", GpuInitStrategy::DirectRegister), "direct-register");
        assert_eq!(format!("{}", BootProtocol::Multiboot2), "multiboot2");
        assert_eq!(format!("{}", BootProtocol::LinuxBoot), "linux-boot");

        let addr = ListenAddr { host: [127, 0, 0, 1], port: 3000 };
        assert_eq!(format!("{addr}"), "127.0.0.1:3000");
    }
}
