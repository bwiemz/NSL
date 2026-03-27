//! M54b: GPU initialization for bare-metal unikernels.
//!
//! Provides PCI device discovery and GPU initialization paths:
//! - **VFIO passthrough** (recommended): hypervisor assigns GPU to guest,
//!   guest uses standard CUDA Driver API via `cuInit()`.
//! - **Direct register** (experimental): bare-metal MMIO register access
//!   for environments without a hypervisor GPU manager.
//!
//! In hosted mode (testing), these functions are no-ops.

use std::sync::Mutex;

// ---------------------------------------------------------------------------
// PCI device discovery
// ---------------------------------------------------------------------------

/// NVIDIA PCI vendor ID.
const NVIDIA_VENDOR_ID: u16 = 0x10DE;

/// PCI configuration space access port (x86).
#[allow(dead_code)]
const PCI_CONFIG_ADDR: u16 = 0xCF8;
#[allow(dead_code)]
const PCI_CONFIG_DATA: u16 = 0xCFC;

/// A discovered PCI device.
#[derive(Debug, Clone)]
pub struct PciDevice {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u8,
    pub subclass: u8,
    /// Base Address Registers (BARs) — physical addresses of MMIO regions.
    pub bars: [u64; 6],
    /// BAR sizes in bytes.
    pub bar_sizes: [u64; 6],
}

impl PciDevice {
    /// Check if this is a display controller (class 0x03) from NVIDIA.
    pub fn is_nvidia_gpu(&self) -> bool {
        self.vendor_id == NVIDIA_VENDOR_ID && self.class_code == 0x03
    }

    /// Get the primary MMIO BAR address (usually BAR0 for GPU registers).
    pub fn mmio_bar(&self) -> u64 {
        self.bars[0]
    }

    /// Get the framebuffer BAR address (usually BAR1).
    pub fn framebuffer_bar(&self) -> u64 {
        self.bars[1]
    }
}

/// Read a 32-bit value from PCI configuration space.
///
/// Uses the legacy x86 I/O port mechanism (CF8h/CFCh).
fn pci_config_read(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let address: u32 = (1 << 31) // enable bit
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    #[cfg(target_os = "none")]
    unsafe {
        core::arch::asm!("out dx, eax", in("eax") address, in("dx") PCI_CONFIG_ADDR);
        let value: u32;
        core::arch::asm!("in eax, dx", out("eax") value, in("dx") PCI_CONFIG_DATA);
        value
    }

    #[cfg(not(target_os = "none"))]
    {
        // Hosted mode: return fake values for testing
        let _ = (address, bus, device, function, offset);
        0xFFFF_FFFF // no device
    }
}

/// Scan the PCI bus for NVIDIA GPUs.
///
/// Returns a list of discovered NVIDIA GPU devices.
pub fn scan_pci_for_gpus() -> Vec<PciDevice> {
    let mut gpus = Vec::new();

    for bus in 0..=255u16 {
        for device in 0..32u8 {
            let config0 = pci_config_read(bus as u8, device, 0, 0);
            let vendor_id = (config0 & 0xFFFF) as u16;
            let device_id = ((config0 >> 16) & 0xFFFF) as u16;

            if vendor_id == 0xFFFF || vendor_id == 0 {
                continue; // no device
            }

            if vendor_id != NVIDIA_VENDOR_ID {
                continue; // not NVIDIA
            }

            // Read class/subclass
            let config8 = pci_config_read(bus as u8, device, 0, 8);
            let class_code = ((config8 >> 24) & 0xFF) as u8;
            let subclass = ((config8 >> 16) & 0xFF) as u8;

            // Read BARs
            let mut bars = [0u64; 6];
            for i in 0..6 {
                let bar_offset = (0x10 + i * 4) as u8;
                let bar_val = pci_config_read(bus as u8, device, 0, bar_offset);

                if bar_val & 0x01 == 0 {
                    // Memory BAR
                    let bar_type = (bar_val >> 1) & 0x03;
                    let addr = (bar_val & 0xFFFFF000) as u64;

                    if bar_type == 0x02 && i < 5 {
                        // 64-bit BAR: combine with next BAR
                        let bar_high = pci_config_read(bus as u8, device, 0, bar_offset + 4);
                        bars[i as usize] = addr | ((bar_high as u64) << 32);
                    } else {
                        bars[i as usize] = addr;
                    }
                }
            }

            gpus.push(PciDevice {
                bus: bus as u8,
                device,
                function: 0,
                vendor_id,
                device_id,
                class_code,
                subclass,
                bars,
                bar_sizes: [0; 6], // sizes determined by BAR sizing protocol
            });
        }
    }

    gpus
}

// ---------------------------------------------------------------------------
// GPU initialization strategies
// ---------------------------------------------------------------------------

/// GPU initialization result.
#[derive(Debug, Clone)]
pub struct GpuInitResult {
    /// Whether GPU initialization succeeded.
    pub success: bool,
    /// GPU device info (if found).
    pub device: Option<PciDevice>,
    /// Strategy that was used.
    pub strategy: &'static str,
    /// Error message (if failed).
    pub error: Option<String>,
}

/// Initialize GPU via VFIO passthrough.
///
/// In this mode, the hypervisor has already assigned the GPU to the guest.
/// We just need to:
/// 1. Discover the GPU on the PCI bus
/// 2. Map its BARs into our address space (identity-mapped by boot stub)
/// 3. Call `cuInit(0)` via the CUDA driver API
///
/// This is the recommended path for production unikernel deployment.
pub fn init_gpu_vfio() -> GpuInitResult {
    let gpus = scan_pci_for_gpus();

    if gpus.is_empty() {
        return GpuInitResult {
            success: false,
            device: None,
            strategy: "vfio-passthrough",
            error: Some("No NVIDIA GPU found on PCI bus".to_string()),
        };
    }

    let gpu = gpus.into_iter().next().unwrap();
    super::serial_puts("[nsl-gpu] NVIDIA GPU found: ");

    // In bare-metal with VFIO, the GPU BARs are already mapped by the hypervisor.
    // We can directly call CUDA driver API functions.
    #[cfg(feature = "cuda")]
    {
        extern "C" {
            fn cuInit(flags: u32) -> u32;
        }
        let rc = unsafe { cuInit(0) };
        if rc != 0 {
            return GpuInitResult {
                success: false,
                device: Some(gpu),
                strategy: "vfio-passthrough",
                error: Some(format!("cuInit failed with rc={}", rc)),
            };
        }
    }

    GpuInitResult {
        success: true,
        device: Some(gpu),
        strategy: "vfio-passthrough",
        error: None,
    }
}

/// Initialize GPU via direct register access (experimental).
///
/// This path attempts to initialize the GPU by directly writing to its
/// MMIO registers. This is significantly more complex and less reliable
/// than VFIO, as it requires knowledge of the GPU's register interface.
///
/// Steps:
/// 1. Discover GPU on PCI bus
/// 2. Map GPU BAR0 (registers) via identity-mapped pages
/// 3. Write GPU initialization sequences to MMIO registers
/// 4. Set up GPU command submission ring
///
/// WARNING: This is experimental and may not work on all GPU architectures.
pub fn init_gpu_direct() -> GpuInitResult {
    let gpus = scan_pci_for_gpus();

    if gpus.is_empty() {
        return GpuInitResult {
            success: false,
            device: None,
            strategy: "direct-register",
            error: Some("No NVIDIA GPU found on PCI bus".to_string()),
        };
    }

    let gpu = gpus.into_iter().next().unwrap();
    let bar0 = gpu.mmio_bar();

    if bar0 == 0 {
        return GpuInitResult {
            success: false,
            device: Some(gpu),
            strategy: "direct-register",
            error: Some("GPU BAR0 is zero — cannot access registers".to_string()),
        };
    }

    // Direct register init is highly GPU-architecture-specific.
    // For now, we report the discovered BAR and defer actual register
    // poke sequences to future work (requires reverse-engineering per arch).
    super::serial_puts("[nsl-gpu] Direct register init: BAR0 discovered\n");

    GpuInitResult {
        success: false, // Not yet implemented for real
        device: Some(gpu),
        strategy: "direct-register",
        error: Some("Direct register init not yet implemented — use VFIO".to_string()),
    }
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

static GPU_STATE: Mutex<Option<GpuInitResult>> = Mutex::new(None);

/// Initialize the GPU in the unikernel.
///
/// `strategy`: 0 = VFIO passthrough, 1 = direct register
///
/// Returns 0 on success, -1 if no GPU found, -2 if init failed.
#[no_mangle]
pub extern "C" fn nsl_unikernel_gpu_init(strategy: i64) -> i64 {
    let result = match strategy {
        0 => init_gpu_vfio(),
        1 => init_gpu_direct(),
        _ => init_gpu_vfio(), // default to VFIO
    };

    let rc = if result.success { 0 } else if result.device.is_none() { -1 } else { -2 };

    if let Ok(mut guard) = GPU_STATE.lock() {
        *guard = Some(result);
    }

    rc
}

/// Check if GPU is initialized.
///
/// Returns 1 if GPU is ready, 0 otherwise.
#[no_mangle]
pub extern "C" fn nsl_unikernel_gpu_ready() -> i64 {
    if let Ok(guard) = GPU_STATE.lock() {
        if let Some(ref result) = *guard {
            if result.success { return 1; }
        }
    }
    0
}

/// Get the GPU PCI device ID (for diagnostics).
///
/// Returns the device ID (u16) or 0 if no GPU.
#[no_mangle]
pub extern "C" fn nsl_unikernel_gpu_device_id() -> i64 {
    if let Ok(guard) = GPU_STATE.lock() {
        if let Some(ref result) = *guard {
            if let Some(ref dev) = result.device {
                return dev.device_id as i64;
            }
        }
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
    fn test_pci_device_is_nvidia_gpu() {
        let dev = PciDevice {
            bus: 0, device: 1, function: 0,
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: 0x2684, // RTX 4090
            class_code: 0x03, // Display controller
            subclass: 0x00,
            bars: [0xFB00_0000, 0xE000_0000, 0, 0, 0, 0],
            bar_sizes: [0; 6],
        };
        assert!(dev.is_nvidia_gpu());
        assert_eq!(dev.mmio_bar(), 0xFB00_0000);
        assert_eq!(dev.framebuffer_bar(), 0xE000_0000);
    }

    #[test]
    fn test_pci_device_not_nvidia() {
        let dev = PciDevice {
            bus: 0, device: 0, function: 0,
            vendor_id: 0x8086, // Intel
            device_id: 0x1234,
            class_code: 0x06, // Bridge
            subclass: 0x00,
            bars: [0; 6],
            bar_sizes: [0; 6],
        };
        assert!(!dev.is_nvidia_gpu());
    }

    #[test]
    fn test_scan_pci_hosted_returns_empty() {
        // In hosted mode, PCI reads return 0xFFFFFFFF (no device)
        let gpus = scan_pci_for_gpus();
        assert!(gpus.is_empty(), "hosted PCI scan should find no GPUs");
    }

    #[test]
    fn test_init_gpu_vfio_no_gpu() {
        // In hosted mode, no GPU will be found
        let result = init_gpu_vfio();
        assert!(!result.success);
        assert!(result.device.is_none());
        assert!(result.error.is_some());
    }

    #[test]
    fn test_init_gpu_direct_no_gpu() {
        let result = init_gpu_direct();
        assert!(!result.success);
        assert!(result.device.is_none());
    }

    #[test]
    fn test_gpu_ffi_init_no_gpu() {
        assert_eq!(nsl_unikernel_gpu_init(0), -1); // no GPU
        assert_eq!(nsl_unikernel_gpu_ready(), 0);
        assert_eq!(nsl_unikernel_gpu_device_id(), 0);
    }
}
