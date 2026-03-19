# M54: Bare-Metal Unikernels — Zero-OS Deployment — Design Specification

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M54
**Prerequisites:** M24 (Standalone Export), M29 (Continuous Batching / Serve Block), M52 (Weight-Aware Compilation)
**Dependencies:** M53 (WCET Proofs) benefits from unikernel's interrupt-free execution guarantee

## Overview

M54 compiles the entire inference stack — model weights, HTTP server, KV-cache, serve pipeline — into a bootable x86_64 ELF image that runs directly on a hypervisor without an operating system. The result is a unikernel: a single-purpose, single-address-space binary where the model *is* the operating system.

This is **structurally impossible in Python/PyTorch.** Python requires CPython (or PyPy), which requires glibc, which requires a Linux kernel. PyTorch additionally requires libcuda, libnccl, cuDNN, and various shared libraries. A typical PyTorch inference container is 2-8 GB. An NSL unikernel for the same model is under 100 MB (excluding weights), boots in under 500 ms, and has zero attack surface because there is no shell, no filesystem, no process model, no users, and no system calls.

The practical value is threefold:

1. **Security.** No OS means no OS vulnerabilities. No shell means no shell injection. No filesystem means no data exfiltration. The binary's attack surface is exactly: (a) the HTTP parser, (b) the GPU driver interface, (c) the model inference code. All three are compiled from NSL/Rust and benefit from memory safety guarantees.

2. **Latency.** No OS means no context switches, no page table walks (identity-mapped physical memory), no scheduler jitter, no interrupt coalescing. The NIC interrupt fires, the virtio driver reads the HTTP request, the GPU processes the model forward pass, and the response is written back — in a single, uninterrupted execution path. This is critical for M53 (WCET Proofs): the unikernel provides the interrupt-free execution guarantee that makes WCET bounds tight.

3. **Density.** A 50 MB unikernel boots in 200 ms on Firecracker. A single bare-metal server can run hundreds of unikernel instances, each serving a different model, with sub-second cold start. This is the infrastructure for serverless AI inference at scale.

---

## Section 1: Language Surface

### CLI Interface

```bash
# Build a bootable unikernel image
nsl build model.nsl --unikernel --weights model.safetensors -o model.elf

# Specify hypervisor target
nsl build model.nsl --unikernel --target kvm -o model.elf
nsl build model.nsl --unikernel --target firecracker -o model.elf

# Configure network
nsl build model.nsl --unikernel --listen 0.0.0.0:8080 -o model.elf

# Configure memory pool size (default: all available RAM)
nsl build model.nsl --unikernel --memory 16G -o model.elf

# Build unikernel with WCET certification (M53)
nsl build model.nsl --unikernel --wcet --weights model.safetensors -o model.elf

# Boot the unikernel in QEMU for testing
nsl run --qemu model.elf --memory 4G --gpu passthrough

# Weight-separate unikernel (weights loaded from virtio-blk at boot)
nsl build model.nsl --unikernel --weights-disk -o model.elf
# Boot with: qemu ... -drive file=weights.img,format=raw,if=virtio
```

### NSL Source: Serve Block in Unikernel Mode

No source changes are required for unikernel deployment. The same `serve` block (M29) that runs on Linux runs as a unikernel. The `--unikernel` flag changes the linker target, not the source code.

```python
model LlamaServe:
    transformer: Transformer
    tokenizer: BPETokenizer

    @endpoint("/generate", method="POST")
    fn generate(self, prompt: str, max_tokens: int = 256) -> str:
        tokens = self.tokenizer.encode(prompt)
        output = self.transformer.forward(tokens, max_tokens=max_tokens)
        return self.tokenizer.decode(output)

serve LlamaServe:
    host: "0.0.0.0"
    port: 8080
    max_batch_size: 32
    max_concurrent: 128

# Compiled to unikernel:
# nsl build llama.nsl --unikernel --weights llama-7b.safetensors -o llama.elf
# qemu-system-x86_64 -kernel llama.elf -m 32G -device vfio-pci,host=01:00.0
```

---

## Section 2: Architecture

### Boot Sequence

```
UEFI/Multiboot2 Entry
    │
    ├── 1. Early init (64-bit long mode, GDT, IDT, page tables)
    │       - Identity-map all physical RAM (no virtual memory overhead)
    │       - Set up interrupt handlers (timer, NIC, GPU)
    │
    ├── 2. Memory pool init
    │       - Detect available RAM via UEFI memory map or E820
    │       - Reserve first 16 MB for kernel structures
    │       - Remaining RAM = model memory pool (slab allocator from M36)
    │
    ├── 3. PCI-e enumeration
    │       - Walk PCI configuration space
    │       - Identify NVIDIA GPU (vendor 0x10DE)
    │       - Map GPU BAR0 (MMIO registers) and BAR1 (framebuffer/VRAM)
    │       - Identify virtio-net device (vendor 0x1AF4, device 0x1000/0x1041)
    │
    ├── 4. GPU initialization (without OS)
    │       - Initialize NVIDIA GPU via direct register writes (RM API)
    │       - Or: use VFIO passthrough (hypervisor manages GPU, guest uses CUDA Driver API)
    │       - Load PTX kernels into GPU
    │       - Allocate VRAM slab for model weights + activations
    │
    ├── 5. Weight loading
    │       - If embedded: weights already in .rodata section
    │       - If disk: read from virtio-blk device into VRAM
    │       - Verify SHA-256 hash (M52 integrity check)
    │
    ├── 6. virtio-net driver init
    │       - Configure virtio queues (rx/tx descriptor rings)
    │       - Set MAC address, enable promiscuous mode
    │       - ARP responder for IP address assignment
    │
    ├── 7. HTTP server listen
    │       - Bind to configured port
    │       - Enter main event loop: poll virtio-net → parse HTTP → dispatch to model → respond
    │
    └── Total boot time target: < 500 ms (< 200 ms on Firecracker)
```

### Core Data Structures

```rust
// crates/nsl-cli/src/unikernel/mod.rs

/// Configuration for unikernel binary generation.
pub struct UnikernelConfig {
    /// Hypervisor target
    pub target: HypervisorTarget,
    /// Network listen address
    pub listen_addr: ListenAddr,
    /// Memory pool size (0 = auto-detect all available RAM)
    pub memory_bytes: u64,
    /// Whether weights are embedded in binary or loaded from disk
    pub weight_source: WeightSource,
    /// GPU initialization strategy
    pub gpu_init: GpuInitStrategy,
    /// Boot protocol
    pub boot_protocol: BootProtocol,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HypervisorTarget {
    /// KVM/QEMU with direct kernel boot (-kernel flag)
    KVM,
    /// AWS Firecracker microVM
    Firecracker,
    /// Google gVisor (experimental)
    GVisor,
}

#[derive(Debug, Clone)]
pub struct ListenAddr {
    pub ip: [u8; 4],
    pub port: u16,
}

impl Default for ListenAddr {
    fn default() -> Self {
        Self { ip: [0, 0, 0, 0], port: 8080 }
    }
}

#[derive(Debug, Clone)]
pub enum WeightSource {
    /// Weights embedded in .rodata (small models, from M24)
    Embedded,
    /// Weights loaded from virtio-blk device at boot
    VirtioBlk,
    /// Weights loaded via network (future: PXE boot)
    Network,
}

#[derive(Debug, Clone, Copy)]
pub enum GpuInitStrategy {
    /// VFIO passthrough: hypervisor assigns GPU to guest, guest uses CUDA Driver API
    /// This is the practical path: NVIDIA's driver model is too complex for bare-metal
    VfioPassthrough,
    /// Direct register init: bare-metal GPU access without any driver (research/experimental)
    DirectRegister,
    /// CPU-only: no GPU, all inference on CPU (for edge/embedded use cases)
    CpuOnly,
}

#[derive(Debug, Clone, Copy)]
pub enum BootProtocol {
    /// Linux boot protocol (bzImage header) — works with QEMU -kernel
    LinuxBoot,
    /// Multiboot2 — works with GRUB and some hypervisors
    Multiboot2,
    /// UEFI application — works with OVMF firmware
    Uefi,
}
```

---

## Section 3: Unikernel Linker

### ELF Generation

The `UnikernelLinker` takes the Cranelift-compiled object file (model + serve pipeline) and links it with the boot stub, virtio drivers, and memory manager into a bootable ELF:

```rust
// crates/nsl-cli/src/unikernel/linker.rs

use std::path::{Path, PathBuf};

/// The unikernel linker: combines compiled model code with boot infrastructure.
pub struct UnikernelLinker {
    /// Configuration
    config: UnikernelConfig,
    /// Path to the Cranelift-compiled model object file
    model_object: PathBuf,
    /// Path to the boot stub assembly/object
    boot_stub: PathBuf,
    /// Path to the virtio driver object
    virtio_object: PathBuf,
    /// Path to the memory manager object
    memory_object: PathBuf,
    /// Path to the HTTP parser object
    http_object: PathBuf,
    /// Path to the GPU init object (if GPU enabled)
    gpu_object: Option<PathBuf>,
    /// Weight data to embed (if embedded mode)
    weight_data: Option<Vec<u8>>,
    /// Output path
    output: PathBuf,
}

impl UnikernelLinker {
    /// Link all components into a bootable ELF.
    pub fn link(&self) -> Result<(), LinkError> {
        // 1. Generate linker script
        let linker_script = self.generate_linker_script();

        // 2. Invoke the linker (lld or system ld)
        let mut cmd = std::process::Command::new("ld.lld");
        cmd.arg("-T").arg(&linker_script);
        cmd.arg("--oformat=elf64-x86-64");
        cmd.arg("--entry=_start");
        cmd.arg("-z").arg("max-page-size=4096");

        // Add all object files
        cmd.arg(&self.boot_stub);
        cmd.arg(&self.model_object);
        cmd.arg(&self.virtio_object);
        cmd.arg(&self.memory_object);
        cmd.arg(&self.http_object);
        if let Some(gpu) = &self.gpu_object {
            cmd.arg(gpu);
        }

        cmd.arg("-o").arg(&self.output);

        let status = cmd.status().map_err(|e| LinkError::LinkerFailed(e.to_string()))?;
        if !status.success() {
            return Err(LinkError::LinkerFailed("ld.lld returned non-zero".into()));
        }

        // 3. If weights embedded, append to .rodata via objcopy
        if let Some(weight_data) = &self.weight_data {
            self.embed_weights(weight_data)?;
        }

        Ok(())
    }

    /// Generate the linker script for unikernel layout.
    fn generate_linker_script(&self) -> PathBuf {
        let entry_addr = match self.config.boot_protocol {
            BootProtocol::LinuxBoot => "0x100000",   // 1 MB (Linux boot protocol)
            BootProtocol::Multiboot2 => "0x100000",
            BootProtocol::Uefi => "0x00100000",
        };

        let script = format!(r#"
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

    /* Stack: 1 MB */
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
"#);

        let script_path = self.output.with_extension("ld");
        std::fs::write(&script_path, script).expect("failed to write linker script");
        script_path
    }

    /// Embed weight data into the .weights section of the ELF.
    fn embed_weights(&self, data: &[u8]) -> Result<(), LinkError> {
        // Write raw weight data to a temporary file
        let weight_bin = self.output.with_extension("weights.bin");
        std::fs::write(&weight_bin, data)
            .map_err(|e| LinkError::IoError(e.to_string()))?;

        // Use objcopy to add the .weights section
        let status = std::process::Command::new("llvm-objcopy")
            .arg("--add-section")
            .arg(format!(".weights={}", weight_bin.display()))
            .arg("--set-section-flags")
            .arg(".weights=readonly,data")
            .arg(&self.output)
            .status()
            .map_err(|e| LinkError::LinkerFailed(e.to_string()))?;

        if !status.success() {
            return Err(LinkError::LinkerFailed("llvm-objcopy failed".into()));
        }

        // Clean up temp file
        let _ = std::fs::remove_file(&weight_bin);
        Ok(())
    }
}

#[derive(Debug)]
pub enum LinkError {
    LinkerFailed(String),
    IoError(String),
    MissingComponent(String),
}
```

---

## Section 4: Boot Stub

### x86_64 Entry Point

The boot stub is a minimal assembly/Rust `no_std` module that sets up the CPU environment and transfers control to the Rust runtime:

```rust
// crates/nsl-cli/src/unikernel/boot.rs
//
// This module compiles as a separate #![no_std] #![no_main] crate
// that provides the _start entry point for the unikernel.

#![no_std]
#![no_main]

use core::panic::PanicInfo;

/// Physical memory layout detected from boot protocol.
#[repr(C)]
pub struct BootInfo {
    /// Total physical memory in bytes
    pub total_memory: u64,
    /// Pointer to memory map entries
    pub memory_map: *const MemoryMapEntry,
    /// Number of memory map entries
    pub memory_map_count: u32,
    /// Command line string pointer
    pub cmdline: *const u8,
    /// Command line length
    pub cmdline_len: u32,
    /// ACPI RSDP physical address (for PCI enumeration)
    pub acpi_rsdp: u64,
}

#[repr(C)]
pub struct MemoryMapEntry {
    pub base: u64,
    pub length: u64,
    pub mem_type: MemoryType,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryType {
    Available = 1,
    Reserved = 2,
    AcpiReclaimable = 3,
    AcpiNvs = 4,
    Unusable = 5,
}

// Defined in assembly: the actual _start entry point
// Sets up 64-bit long mode, GDT, IDT, page tables (identity map),
// then calls rust_entry(boot_info)
extern "C" {
    fn _start();
}

/// Rust entry point, called from assembly after CPU init.
#[no_mangle]
pub extern "C" fn rust_entry(boot_info: *const BootInfo) -> ! {
    let info = unsafe { &*boot_info };

    // 1. Initialize BSS to zero
    unsafe { zero_bss(); }

    // 2. Initialize memory pool
    let pool = memory::init_pool(info);

    // 3. PCI enumeration
    let pci_devices = pci::enumerate(info.acpi_rsdp);

    // 4. GPU initialization (if GPU found)
    let gpu = pci_devices.iter()
        .find(|d| d.vendor == 0x10DE)  // NVIDIA
        .map(|dev| gpu::init_vfio(dev, &pool));

    // 5. virtio-net initialization
    let nic = pci_devices.iter()
        .find(|d| d.vendor == 0x1AF4 && (d.device == 0x1000 || d.device == 0x1041))
        .map(|dev| virtio_net::init(dev, &pool))
        .expect("no virtio-net device found");

    // 6. Load weights
    let weights = weight_loader::load(info, &pool, gpu.as_ref());

    // 7. Initialize model (generated code)
    let model = nsl_model_init(weights, gpu.as_ref());

    // 8. Start HTTP server
    http::serve_forever(&nic, &model, &pool);
}

unsafe fn zero_bss() {
    extern "C" {
        static __bss_start: u8;
        static __bss_end: u8;
    }
    let start = &__bss_start as *const u8 as *mut u8;
    let end = &__bss_end as *const u8;
    let len = end as usize - start as usize;
    core::ptr::write_bytes(start, 0, len);
}

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    // Write panic message to serial port (COM1) for debugging
    serial::write_str("PANIC: ");
    if let Some(msg) = info.message() {
        serial::write_fmt(*msg);
    }
    if let Some(loc) = info.location() {
        serial::write_str(" at ");
        serial::write_str(loc.file());
        serial::write_str(":");
        serial::write_u64(loc.line() as u64);
    }
    serial::write_str("\n");

    // Halt the CPU
    loop {
        unsafe { core::arch::asm!("hlt"); }
    }
}
```

### x86_64 Assembly Entry (Linux Boot Protocol)

```rust
// crates/nsl-cli/src/unikernel/boot_asm.rs
//
// This is the raw assembly that QEMU -kernel jumps to.
// It sets up 64-bit long mode if not already in it,
// configures page tables, GDT, IDT, and calls rust_entry().

/// Assembly entry point (compiled via global_asm! or as a .S file).
///
/// Linux boot protocol: QEMU loads the ELF at the entry point address,
/// CPU is in 32-bit protected mode (or 64-bit if using -kernel with bzImage64).
/// We set up identity-mapped page tables and jump to rust_entry.
pub const BOOT_ASM: &str = r#"
.section .text.boot, "ax"
.global _start
.code64

_start:
    # Disable interrupts during init
    cli

    # Set up stack
    lea rsp, [__stack_top]

    # Clear direction flag
    cld

    # Set up identity-mapped page tables
    # PML4[0] -> PDPT, PDPT[0..512] -> 1GB pages (identity map first 512 GB)
    call setup_page_tables

    # Load GDT
    lgdt [gdt_ptr]

    # Reload segments
    mov ax, 0x10    # data segment
    mov ds, ax
    mov es, ax
    mov fs, ax
    mov gs, ax
    mov ss, ax

    # Set up basic IDT (timer + NIC interrupts)
    call setup_idt

    # Build BootInfo struct on stack
    sub rsp, 64
    mov rdi, rsp
    call detect_memory

    # Enable interrupts
    sti

    # Call Rust entry point
    mov rdi, rsp    # boot_info pointer
    call rust_entry

    # Should never return
    hlt
    jmp $

# Page table setup: identity map using 1GB pages (2MB for first 4GB for finer control)
setup_page_tables:
    # ... (PML4 -> PDPT -> PD entries with 1GB/2MB page mappings)
    ret

setup_idt:
    # ... (IDT entries for timer, virtio-net, GPU interrupts)
    ret

detect_memory:
    # Read E820 memory map or parse multiboot2 info
    # Fill BootInfo struct at [rdi]
    ret

.section .rodata
gdt:
    .quad 0                 # null descriptor
    .quad 0x00AF9A000000FFFF # code segment (64-bit)
    .quad 0x00CF92000000FFFF # data segment
gdt_ptr:
    .word gdt_ptr - gdt - 1
    .quad gdt
"#;
```

---

## Section 5: virtio-net Driver

### Minimal Network Stack

The unikernel includes a minimal network stack: virtio-net driver, Ethernet frame parsing, ARP responder, IPv4, TCP (simplified), and HTTP/1.1 parsing. This is not a full TCP/IP stack — it handles exactly the use case of receiving HTTP requests and sending HTTP responses.

```rust
// crates/nsl-cli/src/unikernel/virtio_net.rs

/// virtio-net device state.
pub struct VirtioNet {
    /// MMIO base address of the virtio device
    mmio_base: usize,
    /// Receive virtqueue
    rx_queue: VirtQueue,
    /// Transmit virtqueue
    tx_queue: VirtQueue,
    /// MAC address
    mac: [u8; 6],
    /// Assigned IP address
    ip: [u8; 4],
}

/// A virtio virtqueue (descriptor ring + available ring + used ring).
pub struct VirtQueue {
    /// Descriptor table base address
    desc_base: usize,
    /// Available ring base address
    avail_base: usize,
    /// Used ring base address
    used_base: usize,
    /// Number of descriptors
    num_descs: u16,
    /// Next descriptor index to use
    next_desc: u16,
    /// Last seen used index
    last_used_idx: u16,
}

/// virtio descriptor (4.2.4.2 in virtio spec).
#[repr(C)]
pub struct VirtqDesc {
    pub addr: u64,   // physical address of buffer
    pub len: u32,    // buffer length
    pub flags: u16,  // VIRTQ_DESC_F_NEXT, VIRTQ_DESC_F_WRITE, etc.
    pub next: u16,   // next descriptor in chain
}

// virtio descriptor flags
pub const VIRTQ_DESC_F_NEXT: u16 = 1;
pub const VIRTQ_DESC_F_WRITE: u16 = 2;

impl VirtioNet {
    /// Initialize the virtio-net device from a PCI BAR.
    pub fn init(pci_device: &PciDevice, pool: &MemoryPool) -> Self {
        let mmio_base = pci_device.bar0 as usize;

        // 1. Reset device
        Self::write_reg(mmio_base, VIRTIO_STATUS, 0);

        // 2. Set ACKNOWLEDGE and DRIVER status bits
        Self::write_reg(mmio_base, VIRTIO_STATUS, VIRTIO_STATUS_ACKNOWLEDGE | VIRTIO_STATUS_DRIVER);

        // 3. Negotiate features (we need: VIRTIO_NET_F_MAC)
        let features = Self::read_reg(mmio_base, VIRTIO_DEVICE_FEATURES);
        Self::write_reg(mmio_base, VIRTIO_DRIVER_FEATURES, features & VIRTIO_NET_F_MAC);

        // 4. Set FEATURES_OK
        Self::write_reg(mmio_base, VIRTIO_STATUS,
            VIRTIO_STATUS_ACKNOWLEDGE | VIRTIO_STATUS_DRIVER | VIRTIO_STATUS_FEATURES_OK);

        // 5. Allocate virtqueues
        let rx_queue = VirtQueue::alloc(pool, 256); // 256 descriptors
        let tx_queue = VirtQueue::alloc(pool, 256);

        // 6. Configure queues
        Self::write_reg(mmio_base, VIRTIO_QUEUE_SEL, 0); // RX queue
        Self::write_reg64(mmio_base, VIRTIO_QUEUE_DESC, rx_queue.desc_base as u64);
        Self::write_reg64(mmio_base, VIRTIO_QUEUE_AVAIL, rx_queue.avail_base as u64);
        Self::write_reg64(mmio_base, VIRTIO_QUEUE_USED, rx_queue.used_base as u64);
        Self::write_reg(mmio_base, VIRTIO_QUEUE_READY, 1);

        Self::write_reg(mmio_base, VIRTIO_QUEUE_SEL, 1); // TX queue
        Self::write_reg64(mmio_base, VIRTIO_QUEUE_DESC, tx_queue.desc_base as u64);
        Self::write_reg64(mmio_base, VIRTIO_QUEUE_AVAIL, tx_queue.avail_base as u64);
        Self::write_reg64(mmio_base, VIRTIO_QUEUE_USED, tx_queue.used_base as u64);
        Self::write_reg(mmio_base, VIRTIO_QUEUE_READY, 1);

        // 7. Set DRIVER_OK
        Self::write_reg(mmio_base, VIRTIO_STATUS,
            VIRTIO_STATUS_ACKNOWLEDGE | VIRTIO_STATUS_DRIVER |
            VIRTIO_STATUS_FEATURES_OK | VIRTIO_STATUS_DRIVER_OK);

        // 8. Read MAC address
        let mut mac = [0u8; 6];
        for i in 0..6 {
            mac[i] = Self::read_reg8(mmio_base, VIRTIO_NET_MAC_OFFSET + i);
        }

        // 9. Pre-populate RX queue with buffers
        let mut net = VirtioNet {
            mmio_base,
            rx_queue,
            tx_queue,
            mac,
            ip: [10, 0, 0, 2], // Default IP, configurable
        };
        net.populate_rx_buffers(pool);

        net
    }

    /// Poll for received packets (non-blocking).
    pub fn poll_rx(&mut self) -> Option<&[u8]> {
        self.rx_queue.poll_used()
    }

    /// Send a packet.
    pub fn send(&mut self, data: &[u8]) {
        self.tx_queue.enqueue(data);
        // Notify device
        Self::write_reg(self.mmio_base, VIRTIO_QUEUE_NOTIFY, 1); // TX queue index
    }

    fn populate_rx_buffers(&mut self, pool: &MemoryPool) {
        for _ in 0..128 {
            let buf = pool.alloc(2048); // Standard Ethernet MTU + headers
            self.rx_queue.enqueue_empty(buf, 2048);
        }
        Self::write_reg(self.mmio_base, VIRTIO_QUEUE_NOTIFY, 0); // RX queue index
    }

    fn write_reg(base: usize, offset: usize, value: u32) {
        unsafe { core::ptr::write_volatile((base + offset) as *mut u32, value); }
    }

    fn write_reg64(base: usize, offset: usize, value: u64) {
        unsafe { core::ptr::write_volatile((base + offset) as *mut u64, value); }
    }

    fn read_reg(base: usize, offset: usize) -> u32 {
        unsafe { core::ptr::read_volatile((base + offset) as *const u32) }
    }

    fn read_reg8(base: usize, offset: usize) -> u8 {
        unsafe { core::ptr::read_volatile((base + offset) as *const u8) }
    }
}

// virtio MMIO register offsets
const VIRTIO_STATUS: usize = 0x070;
const VIRTIO_DEVICE_FEATURES: usize = 0x010;
const VIRTIO_DRIVER_FEATURES: usize = 0x020;
const VIRTIO_QUEUE_SEL: usize = 0x030;
const VIRTIO_QUEUE_NUM_MAX: usize = 0x034;
const VIRTIO_QUEUE_NUM: usize = 0x038;
const VIRTIO_QUEUE_READY: usize = 0x044;
const VIRTIO_QUEUE_NOTIFY: usize = 0x050;
const VIRTIO_QUEUE_DESC: usize = 0x080;
const VIRTIO_QUEUE_AVAIL: usize = 0x090;
const VIRTIO_QUEUE_USED: usize = 0x0A0;
const VIRTIO_NET_MAC_OFFSET: usize = 0x100;

// virtio status bits
const VIRTIO_STATUS_ACKNOWLEDGE: u32 = 1;
const VIRTIO_STATUS_DRIVER: u32 = 2;
const VIRTIO_STATUS_FEATURES_OK: u32 = 8;
const VIRTIO_STATUS_DRIVER_OK: u32 = 4;
const VIRTIO_NET_F_MAC: u32 = 1 << 5;
```

---

## Section 6: Embedded HTTP Server

### HTTP/1.1 Request Parser

The unikernel includes a minimal HTTP/1.1 parser that handles exactly the request types needed for model serving: POST with JSON body, GET for health checks.

```rust
// crates/nsl-cli/src/unikernel/http.rs

/// Parsed HTTP request.
pub struct HttpRequest<'a> {
    pub method: HttpMethod,
    pub path: &'a str,
    pub content_length: usize,
    pub body: &'a [u8],
    pub connection_keep_alive: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HttpMethod {
    Get,
    Post,
    Options,
    Unknown,
}

/// HTTP response builder.
pub struct HttpResponse {
    pub status: u16,
    pub status_text: &'static str,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn ok_json(body: &[u8]) -> Self {
        Self {
            status: 200,
            status_text: "OK",
            content_type: "application/json",
            body: body.to_vec(),
        }
    }

    pub fn bad_request(message: &str) -> Self {
        Self {
            status: 400,
            status_text: "Bad Request",
            content_type: "text/plain",
            body: message.as_bytes().to_vec(),
        }
    }

    pub fn not_found() -> Self {
        Self {
            status: 404,
            status_text: "Not Found",
            content_type: "text/plain",
            body: b"Not Found".to_vec(),
        }
    }

    pub fn internal_error(message: &str) -> Self {
        Self {
            status: 500,
            status_text: "Internal Server Error",
            content_type: "text/plain",
            body: message.as_bytes().to_vec(),
        }
    }

    /// Serialize to HTTP/1.1 response bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256 + self.body.len());
        buf.extend_from_slice(b"HTTP/1.1 ");
        buf.extend_from_slice(itoa(self.status).as_bytes());
        buf.extend_from_slice(b" ");
        buf.extend_from_slice(self.status_text.as_bytes());
        buf.extend_from_slice(b"\r\nContent-Type: ");
        buf.extend_from_slice(self.content_type.as_bytes());
        buf.extend_from_slice(b"\r\nContent-Length: ");
        buf.extend_from_slice(itoa(self.body.len() as u16).as_bytes());
        buf.extend_from_slice(b"\r\nConnection: keep-alive\r\n\r\n");
        buf.extend_from_slice(&self.body);
        buf
    }
}

/// Parse an HTTP/1.1 request from a raw byte buffer.
/// Returns None if the buffer does not contain a complete request.
pub fn parse_request(buf: &[u8]) -> Option<HttpRequest<'_>> {
    // Find end of headers (\r\n\r\n)
    let header_end = find_subsequence(buf, b"\r\n\r\n")?;
    let headers = core::str::from_utf8(&buf[..header_end]).ok()?;

    // Parse request line
    let mut lines = headers.lines();
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method_str = parts.next()?;
    let path = parts.next()?;

    let method = match method_str {
        "GET" => HttpMethod::Get,
        "POST" => HttpMethod::Post,
        "OPTIONS" => HttpMethod::Options,
        _ => HttpMethod::Unknown,
    };

    // Parse headers
    let mut content_length = 0usize;
    let mut keep_alive = true;

    for line in lines {
        if let Some(val) = strip_header(line, "Content-Length") {
            content_length = val.trim().parse().unwrap_or(0);
        }
        if let Some(val) = strip_header(line, "Connection") {
            keep_alive = !val.trim().eq_ignore_ascii_case("close");
        }
    }

    // Check if we have the full body
    let body_start = header_end + 4; // skip \r\n\r\n
    if buf.len() < body_start + content_length {
        return None; // incomplete request
    }

    let body = &buf[body_start..body_start + content_length];

    Some(HttpRequest {
        method,
        path,
        content_length,
        body,
        connection_keep_alive: keep_alive,
    })
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|window| window == needle)
}

fn strip_header<'a>(line: &'a str, header: &str) -> Option<&'a str> {
    let line_lower = line.to_ascii_lowercase();
    let header_lower = header.to_ascii_lowercase();
    if line_lower.starts_with(&header_lower) {
        let rest = &line[header.len()..];
        if rest.starts_with(':') {
            return Some(&rest[1..]);
        }
    }
    None
}

fn itoa(n: u16) -> &'static str {
    // Simple integer-to-string for status codes and small lengths
    // In practice, use a stack buffer + formatting
    match n {
        200 => "200",
        400 => "400",
        404 => "404",
        500 => "500",
        _ => "0",
    }
}
```

### Main Event Loop

```rust
// crates/nsl-cli/src/unikernel/http.rs (continued)

/// The main HTTP serve loop. This function never returns.
/// It polls the virtio-net device for incoming packets, parses HTTP requests,
/// dispatches to the model, and sends responses.
pub fn serve_forever(
    nic: &mut VirtioNet,
    model: &dyn ModelInterface,
    pool: &MemoryPool,
) -> ! {
    let mut rx_buf = [0u8; 65536];  // receive buffer
    let mut rx_len = 0usize;

    loop {
        // Poll for incoming packets
        if let Some(packet) = nic.poll_rx() {
            // Parse Ethernet + IP + TCP headers
            if let Some(tcp_payload) = parse_tcp_payload(packet) {
                // Append to receive buffer
                let copy_len = tcp_payload.len().min(rx_buf.len() - rx_len);
                rx_buf[rx_len..rx_len + copy_len].copy_from_slice(&tcp_payload[..copy_len]);
                rx_len += copy_len;

                // Try to parse HTTP request
                if let Some(request) = parse_request(&rx_buf[..rx_len]) {
                    let response = dispatch_request(&request, model, pool);
                    let response_bytes = response.to_bytes();

                    // Build TCP + IP + Ethernet response
                    let full_packet = build_tcp_response(packet, &response_bytes);
                    nic.send(&full_packet);

                    // Reset buffer
                    rx_len = 0;
                }
            }
        }

        // Yield CPU briefly if no packets (avoid busy-wait power consumption)
        // On bare metal, this is a PAUSE instruction (reduces power in spin loops)
        unsafe { core::arch::asm!("pause"); }
    }
}

/// Dispatch an HTTP request to the appropriate handler.
fn dispatch_request(
    request: &HttpRequest,
    model: &dyn ModelInterface,
    pool: &MemoryPool,
) -> HttpResponse {
    match (request.method, request.path) {
        (HttpMethod::Get, "/health") => {
            HttpResponse::ok_json(b"{\"status\":\"ok\"}")
        }
        (HttpMethod::Get, "/metrics") => {
            let metrics = model.get_metrics();
            HttpResponse::ok_json(metrics.as_bytes())
        }
        (HttpMethod::Post, "/generate") | (HttpMethod::Post, "/v1/completions") => {
            match model.inference(request.body) {
                Ok(output) => HttpResponse::ok_json(&output),
                Err(e) => HttpResponse::bad_request(&e),
            }
        }
        (HttpMethod::Options, _) => {
            // CORS preflight
            HttpResponse::ok_json(b"{}")
        }
        _ => HttpResponse::not_found(),
    }
}

/// Trait that the compiled model implements for HTTP dispatch.
pub trait ModelInterface {
    /// Run inference on a JSON-encoded request body.
    /// Returns JSON-encoded response body.
    fn inference(&self, request_json: &[u8]) -> Result<Vec<u8>, String>;

    /// Return metrics as a JSON string.
    fn get_metrics(&self) -> String;
}
```

---

## Section 7: GPU Initialization Without OS

### VFIO Passthrough Strategy

Direct bare-metal GPU initialization is impractical because NVIDIA's GPU firmware and register interface are proprietary. Instead, M54 uses VFIO passthrough: the hypervisor (KVM/QEMU) assigns the physical GPU to the unikernel guest using VFIO, and the guest accesses the GPU via the CUDA Driver API — which is statically linked into the unikernel binary from the CUDA runtime library.

```rust
// crates/nsl-cli/src/unikernel/gpu.rs

/// GPU state in unikernel context.
pub struct UnikernelGpu {
    /// CUDA device handle (from cuDevicePrimaryCtxRetain)
    pub device: i32,
    /// CUDA context
    pub context: u64,
    /// VRAM base address (BAR1 mapping)
    pub vram_base: usize,
    /// VRAM size in bytes
    pub vram_size: u64,
    /// Compiled PTX modules loaded on this device
    pub modules: Vec<CudaModule>,
}

pub struct CudaModule {
    pub name: String,
    pub module_handle: u64,
    pub functions: Vec<CudaFunction>,
}

pub struct CudaFunction {
    pub name: String,
    pub handle: u64,
}

impl UnikernelGpu {
    /// Initialize GPU via CUDA Driver API.
    /// In VFIO mode, the GPU's PCI BAR is already mapped into guest physical memory
    /// by the hypervisor, so cuInit() finds the device normally.
    pub fn init_vfio(pci_device: &PciDevice, pool: &MemoryPool) -> Result<Self, GpuError> {
        // cuInit(0) — initializes CUDA driver
        // In unikernel, this accesses GPU registers via the VFIO-mapped BAR
        let result = unsafe { cuda_driver::cuInit(0) };
        if result != 0 {
            return Err(GpuError::InitFailed(result));
        }

        // Get device 0
        let mut device: i32 = 0;
        let result = unsafe { cuda_driver::cuDeviceGet(&mut device, 0) };
        if result != 0 {
            return Err(GpuError::NoDevice(result));
        }

        // Retain primary context (M17 pattern: cuDevicePrimaryCtxRetain, not cuCtxCreate)
        let mut context: u64 = 0;
        let result = unsafe { cuda_driver::cuDevicePrimaryCtxRetain(&mut context, device) };
        if result != 0 {
            return Err(GpuError::ContextFailed(result));
        }

        // Set context as current
        unsafe { cuda_driver::cuCtxSetCurrent(context); }

        // Query VRAM size
        let mut vram_size: u64 = 0;
        unsafe { cuda_driver::cuDeviceTotalMem_v2(&mut vram_size, device); }

        Ok(UnikernelGpu {
            device,
            context,
            vram_base: pci_device.bar1 as usize,
            vram_size,
            modules: Vec::new(),
        })
    }

    /// Load a PTX module from embedded .rodata into the GPU.
    pub fn load_ptx(&mut self, name: &str, ptx_data: &[u8]) -> Result<(), GpuError> {
        let mut module: u64 = 0;
        let result = unsafe {
            cuda_driver::cuModuleLoadData(&mut module, ptx_data.as_ptr() as *const _)
        };
        if result != 0 {
            return Err(GpuError::ModuleLoadFailed(name.to_string(), result));
        }

        self.modules.push(CudaModule {
            name: name.to_string(),
            module_handle: module,
            functions: Vec::new(),
        });

        Ok(())
    }

    /// Allocate VRAM using cuMemAlloc (for slab from M36).
    pub fn alloc_vram(&self, size: u64) -> Result<u64, GpuError> {
        let mut ptr: u64 = 0;
        let result = unsafe { cuda_driver::cuMemAlloc_v2(&mut ptr, size as usize) };
        if result != 0 {
            return Err(GpuError::AllocFailed(size, result));
        }
        Ok(ptr)
    }
}

#[derive(Debug)]
pub enum GpuError {
    InitFailed(i32),
    NoDevice(i32),
    ContextFailed(i32),
    ModuleLoadFailed(String, i32),
    AllocFailed(u64, i32),
}

/// PCI device descriptor from PCI-e enumeration.
pub struct PciDevice {
    pub vendor: u16,
    pub device: u16,
    pub class: u8,
    pub subclass: u8,
    pub bar0: u64,
    pub bar1: u64,
    pub irq: u8,
}

/// Minimal CUDA Driver API FFI bindings (statically linked).
/// These are the subset of the CUDA Driver API needed for unikernel operation.
mod cuda_driver {
    extern "C" {
        pub fn cuInit(flags: u32) -> i32;
        pub fn cuDeviceGet(device: *mut i32, ordinal: i32) -> i32;
        pub fn cuDevicePrimaryCtxRetain(ctx: *mut u64, device: i32) -> i32;
        pub fn cuCtxSetCurrent(ctx: u64) -> i32;
        pub fn cuDeviceTotalMem_v2(bytes: *mut u64, device: i32) -> i32;
        pub fn cuModuleLoadData(module: *mut u64, data: *const core::ffi::c_void) -> i32;
        pub fn cuMemAlloc_v2(ptr: *mut u64, size: usize) -> i32;
        pub fn cuMemFree_v2(ptr: u64) -> i32;
        pub fn cuLaunchKernel(
            func: u64, gx: u32, gy: u32, gz: u32,
            bx: u32, by: u32, bz: u32,
            shared_mem: u32, stream: u64,
            args: *mut *mut core::ffi::c_void, extra: *mut *mut core::ffi::c_void,
        ) -> i32;
        pub fn cuCtxSynchronize() -> i32;
        pub fn cuMemcpyHtoD_v2(dst: u64, src: *const core::ffi::c_void, size: usize) -> i32;
        pub fn cuMemcpyDtoH_v2(dst: *mut core::ffi::c_void, src: u64, size: usize) -> i32;
    }
}
```

---

## Section 8: Memory Model

### Physical Memory Pool

The unikernel uses all available physical RAM as a single memory pool. No virtual memory overhead, no page tables beyond the identity map.

```rust
// crates/nsl-cli/src/unikernel/memory.rs

/// Physical memory pool for the unikernel.
/// All allocations come from this pool. No system allocator, no malloc/free.
pub struct MemoryPool {
    /// Base physical address of the pool
    base: usize,
    /// Total size in bytes
    size: u64,
    /// Current allocation watermark (bump allocator for simplicity)
    watermark: core::sync::atomic::AtomicUsize,
    /// Page size for alignment
    page_size: usize,
}

impl MemoryPool {
    /// Initialize the memory pool from boot info.
    /// Finds the largest available memory region and uses it as the pool.
    pub fn init(boot_info: &BootInfo) -> Self {
        let entries = unsafe {
            core::slice::from_raw_parts(boot_info.memory_map, boot_info.memory_map_count as usize)
        };

        // Find the largest available region
        let mut best_base: u64 = 0;
        let mut best_size: u64 = 0;
        for entry in entries {
            if entry.mem_type == MemoryType::Available && entry.length > best_size {
                // Skip the first 16 MB (reserved for kernel structures)
                let base = if entry.base < 0x1000000 {
                    0x1000000
                } else {
                    entry.base
                };
                let end = entry.base + entry.length;
                if end > base {
                    best_base = base;
                    best_size = end - base;
                }
            }
        }

        MemoryPool {
            base: best_base as usize,
            size: best_size,
            watermark: core::sync::atomic::AtomicUsize::new(best_base as usize),
            page_size: 4096,
        }
    }

    /// Allocate a block of memory from the pool (bump allocator).
    /// Returns a physical address. Thread-safe via atomic fetch_add.
    pub fn alloc(&self, size: usize) -> *mut u8 {
        let aligned_size = (size + self.page_size - 1) & !(self.page_size - 1);
        let old_watermark = self.watermark.fetch_add(
            aligned_size,
            core::sync::atomic::Ordering::SeqCst,
        );
        if old_watermark + aligned_size > self.base + self.size as usize {
            panic!("MemoryPool: out of memory (requested {}, available {})",
                   aligned_size,
                   self.base + self.size as usize - old_watermark);
        }
        old_watermark as *mut u8
    }

    /// Allocate a block aligned to the specified alignment.
    pub fn alloc_aligned(&self, size: usize, alignment: usize) -> *mut u8 {
        // Bump watermark up to alignment boundary first
        loop {
            let current = self.watermark.load(core::sync::atomic::Ordering::SeqCst);
            let aligned = (current + alignment - 1) & !(alignment - 1);
            let new = aligned + size;
            if new > self.base + self.size as usize {
                panic!("MemoryPool: out of memory");
            }
            if self.watermark.compare_exchange(
                current, new,
                core::sync::atomic::Ordering::SeqCst,
                core::sync::atomic::Ordering::SeqCst,
            ).is_ok() {
                return aligned as *mut u8;
            }
        }
    }

    /// Total pool size.
    pub fn total_bytes(&self) -> u64 {
        self.size
    }

    /// Used bytes.
    pub fn used_bytes(&self) -> u64 {
        let wm = self.watermark.load(core::sync::atomic::Ordering::SeqCst);
        (wm - self.base) as u64
    }

    /// Free bytes remaining.
    pub fn free_bytes(&self) -> u64 {
        self.total_bytes() - self.used_bytes()
    }
}
```

---

## Section 9: PCI-e Enumeration

### Minimal PCI Configuration Space Walker

```rust
// crates/nsl-cli/src/unikernel/pci.rs

/// Enumerate PCI devices by walking configuration space.
/// Uses I/O port access (0xCF8/0xCFC) on x86_64.
pub fn enumerate(_acpi_rsdp: u64) -> Vec<PciDevice> {
    let mut devices = Vec::new();

    // Brute-force scan: bus 0-255, device 0-31, function 0-7
    // (In practice, most hypervisors expose only a few devices)
    for bus in 0..=255u8 {
        for device in 0..32u8 {
            for function in 0..8u8 {
                let vendor = pci_read_u16(bus, device, function, 0x00);
                if vendor == 0xFFFF { continue; } // no device

                let device_id = pci_read_u16(bus, device, function, 0x02);
                let class = pci_read_u8(bus, device, function, 0x0B);
                let subclass = pci_read_u8(bus, device, function, 0x0A);

                // Read BARs
                let bar0 = pci_read_bar(bus, device, function, 0x10);
                let bar1 = pci_read_bar(bus, device, function, 0x14);

                // Read interrupt line
                let irq = pci_read_u8(bus, device, function, 0x3C);

                devices.push(PciDevice {
                    vendor,
                    device: device_id,
                    class,
                    subclass,
                    bar0,
                    bar1,
                    irq,
                });

                // Check if multi-function device
                if function == 0 {
                    let header_type = pci_read_u8(bus, device, function, 0x0E);
                    if header_type & 0x80 == 0 {
                        break; // single-function device
                    }
                }
            }
        }
    }

    devices
}

/// Read a 32-bit value from PCI configuration space via I/O ports.
fn pci_config_read(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let address: u32 = 0x80000000
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);

    unsafe {
        // Write address to CONFIG_ADDRESS (0xCF8)
        core::arch::asm!(
            "out dx, eax",
            in("dx") 0xCF8u16,
            in("eax") address,
        );
        // Read data from CONFIG_DATA (0xCFC)
        let mut data: u32;
        core::arch::asm!(
            "in eax, dx",
            in("dx") 0xCFCu16,
            out("eax") data,
        );
        data
    }
}

fn pci_read_u16(bus: u8, device: u8, function: u8, offset: u8) -> u16 {
    let val = pci_config_read(bus, device, function, offset & 0xFC);
    ((val >> ((offset & 2) * 8)) & 0xFFFF) as u16
}

fn pci_read_u8(bus: u8, device: u8, function: u8, offset: u8) -> u8 {
    let val = pci_config_read(bus, device, function, offset & 0xFC);
    ((val >> ((offset & 3) * 8)) & 0xFF) as u8
}

fn pci_read_bar(bus: u8, device: u8, function: u8, bar_offset: u8) -> u64 {
    let low = pci_config_read(bus, device, function, bar_offset);
    if low & 0x04 != 0 {
        // 64-bit BAR: read high 32 bits from next register
        let high = pci_config_read(bus, device, function, bar_offset + 4);
        ((high as u64) << 32) | ((low as u64) & 0xFFFFFFF0)
    } else {
        (low & 0xFFFFFFF0) as u64
    }
}
```

---

## Section 10: Type System

### Unikernel Mode Constraints

When `--unikernel` is specified, the semantic checker enforces additional constraints:

```rust
// Extension to crates/nsl-semantic/src/checker.rs

/// Constraints enforced in unikernel mode.
pub struct UnikernelConstraints;

impl UnikernelConstraints {
    /// Operations banned in unikernel mode:
    pub fn check(func: &FunctionDef) -> Vec<UnikernelViolation> {
        let mut violations = Vec::new();

        // 1. No file system operations (no filesystem in unikernel)
        // (NSL doesn't have general file I/O, but check for weight loading paths)

        // 2. No py.call() (no Python interpreter)
        // 3. No subprocess/exec calls
        // 4. No dynamic library loading
        // 5. print() is redirected to serial port, not stdout

        violations
    }
}

#[derive(Debug)]
pub struct UnikernelViolation {
    pub source_loc: String,
    pub operation: String,
    pub reason: String,
}
```

The type system also verifies that:
- All tensor shapes are statically known (required for slab allocation in the memory pool)
- The serve block has at least one `@endpoint` (the unikernel needs something to serve)
- The model does not use training features (train blocks are not supported in unikernel mode)

---

## Section 11: Testing Strategy

### Unit Tests

| Test | Description |
|------|-------------|
| `test_http_parse_get` | Parse `GET /health HTTP/1.1\r\n\r\n`. Verify method=Get, path="/health". |
| `test_http_parse_post` | Parse `POST /generate HTTP/1.1\r\nContent-Length: 13\r\n\r\n{"prompt":"hi"}`. Verify body parsed correctly. |
| `test_http_parse_incomplete` | Parse partial request (missing body). Verify returns None. |
| `test_http_response_json` | Build 200 OK JSON response. Verify Content-Type, Content-Length, body. |
| `test_http_parse_keep_alive` | Request with `Connection: keep-alive`. Verify flag is true. |
| `test_http_parse_close` | Request with `Connection: close`. Verify flag is false. |
| `test_memory_pool_basic` | Allocate 3 blocks from a 1MB pool. Verify addresses are sequential and aligned. |
| `test_memory_pool_oom` | Allocate more than pool size. Verify panic (or error in test mode). |
| `test_memory_pool_alignment` | Allocate with 256-byte alignment. Verify returned address is aligned. |
| `test_linker_script_gen` | Generate linker script for KVM target. Verify entry point, section layout. |
| `test_pci_vendor_ids` | Known vendor IDs: NVIDIA=0x10DE, virtio=0x1AF4. Verify detection logic. |
| `test_boot_info_parse` | Construct BootInfo with 3 memory regions. Verify largest Available region selected. |

### E2E Tests

| Test | NSL Source | Description |
|------|-----------|-------------|
| `examples/m54_unikernel_basic.nsl` | Tiny MLP with serve block. | Build unikernel ELF. Boot in QEMU. Send HTTP POST. Verify response. |
| `examples/m54_health_check.nsl` | Model with `/health` endpoint. | Boot in QEMU. Send GET /health. Verify `{"status":"ok"}`. |
| `examples/m54_cpu_only.nsl` | Model with `--unikernel` and no GPU. | Build CPU-only unikernel. Boot in QEMU without GPU. Verify inference works. |
| `examples/m54_weights_disk.nsl` | Model with `--weights-disk`. | Build unikernel + weights image. Boot with virtio-blk. Verify weights loaded. |
| `examples/m54_size_check.nsl` | 7B model unikernel. | Verify ELF size < 100 MB (excluding weights). |
| `examples/m54_boot_time.nsl` | Model with `/health` endpoint. | Boot in Firecracker. Measure time from VM start to first successful HTTP response. Target: < 500ms. |

### QEMU Test Harness

```bash
#!/bin/bash
# Test script for unikernel E2E tests

set -euo pipefail

# Build the unikernel
nsl build examples/m54_unikernel_basic.nsl --unikernel --weights tiny_weights.safetensors -o /tmp/test.elf

# Boot in QEMU with virtio-net and no display
qemu-system-x86_64 \
    -kernel /tmp/test.elf \
    -m 4G \
    -nographic \
    -serial stdio \
    -netdev user,id=net0,hostfwd=tcp::18080-:8080 \
    -device virtio-net-pci,netdev=net0 \
    -no-reboot \
    &

QEMU_PID=$!

# Wait for boot (poll /health endpoint)
for i in $(seq 1 30); do
    if curl -s http://localhost:18080/health 2>/dev/null | grep -q "ok"; then
        echo "BOOT SUCCESS (attempt $i)"
        break
    fi
    sleep 0.5
done

# Send inference request
RESPONSE=$(curl -s -X POST http://localhost:18080/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "test input", "max_tokens": 10}')

echo "Response: $RESPONSE"

# Verify response is valid JSON
echo "$RESPONSE" | python3 -c "import sys, json; json.load(sys.stdin)"
echo "RESPONSE VALID JSON: OK"

# Clean up
kill $QEMU_PID 2>/dev/null || true
wait $QEMU_PID 2>/dev/null || true
echo "TEST PASSED"
```

---

## Section 12: Modified Files

### New Files

| File | Responsibility |
|------|----------------|
| `crates/nsl-cli/src/unikernel/mod.rs` | UnikernelConfig, top-level orchestration |
| `crates/nsl-cli/src/unikernel/boot.rs` | Boot stub (Rust `no_std` entry point, BootInfo, panic handler) |
| `crates/nsl-cli/src/unikernel/boot_asm.rs` | x86_64 assembly entry point, GDT, IDT, page tables |
| `crates/nsl-cli/src/unikernel/linker.rs` | UnikernelLinker: ELF generation, linker script, weight embedding |
| `crates/nsl-cli/src/unikernel/memory.rs` | MemoryPool: physical memory bump allocator |
| `crates/nsl-cli/src/unikernel/pci.rs` | PCI-e enumeration via I/O ports |
| `crates/nsl-cli/src/unikernel/virtio_net.rs` | virtio-net driver: init, rx, tx, virtqueue management |
| `crates/nsl-cli/src/unikernel/gpu.rs` | GPU initialization via VFIO/CUDA Driver API |
| `crates/nsl-cli/src/unikernel/http.rs` | HTTP/1.1 parser, response builder, main event loop, model dispatch |
| `crates/nsl-cli/src/unikernel/serial.rs` | Serial port (COM1) output for debug logging |
| `crates/nsl-cli/src/unikernel/tcp.rs` | Minimal TCP state machine (SYN/ACK, data, FIN) |
| `crates/nsl-cli/src/unikernel/ip.rs` | IPv4 packet parsing/construction, ARP responder |

### Modified Files

| File | Change |
|------|--------|
| `crates/nsl-cli/src/main.rs` | `--unikernel`, `--target kvm|firecracker`, `--listen`, `--memory`, `--weights-disk`, `nsl run --qemu` subcommand |
| `crates/nsl-codegen/src/compiler.rs` | `unikernel_config: Option<UnikernelConfig>` field; when set, emit `no_std`-compatible code, skip libc dependencies |
| `crates/nsl-codegen/src/lib.rs` | `unikernel: bool` in `CompileOptions` |
| `crates/nsl-codegen/src/serve.rs` | When unikernel mode, emit `ModelInterface` trait impl instead of tokio-based HTTP server |
| `crates/nsl-semantic/src/checker.rs` | `check_unikernel_constraints()`: reject py.call(), file I/O, train blocks in unikernel mode |
| `crates/nsl-runtime/src/lib.rs` | Conditional compilation: `#[cfg(not(feature = "unikernel"))]` gates for OS-dependent code |
| `Cargo.toml` (workspace) | New `unikernel` feature flag; `nsl-cli` depends on `lld` for linking |

---

## Section 13: Deliverables

1. `nsl build --unikernel` CLI flag that produces a bootable x86_64 ELF image
2. Boot stub with x86_64 long mode setup, identity-mapped page tables, GDT/IDT
3. Physical memory pool allocator (bump allocator, zero OS overhead)
4. PCI-e enumeration for GPU and virtio device discovery
5. virtio-net driver with rx/tx virtqueue management
6. Minimal TCP/IP stack: ARP, IPv4, TCP (connection-oriented, no UDP)
7. HTTP/1.1 request parser and response builder
8. Main event loop: poll NIC, parse HTTP, dispatch to model, send response
9. GPU initialization via VFIO passthrough and CUDA Driver API
10. Weight loading from embedded .rodata or virtio-blk device
11. Linker script and ELF generation via lld
12. Serial port debug output (COM1)
13. QEMU test harness for automated E2E testing
14. Hypervisor targets: KVM/QEMU (primary), Firecracker (stretch)

## Out of Scope

- Full TCP/IP stack (no UDP, no ICMP beyond ping, no DNS, no TLS/HTTPS)
- DHCP client (IP address is statically configured or hardcoded)
- Filesystem (no virtio-fs, no ext4; weights loaded from raw virtio-blk or embedded)
- Multi-GPU support in unikernel (single GPU per unikernel instance; scale by running multiple instances)
- SMP/multi-core (unikernel runs on a single core; GPU does the parallel work)
- Direct bare-metal GPU initialization without VFIO (NVIDIA's register interface is proprietary)
- ARM/RISC-V unikernel targets (x86_64 only for now)
- Hot model reloading (updating weights requires rebooting the unikernel)
- Streaming HTTP responses (chunked transfer encoding) — full response buffered then sent
- TLS termination (terminate TLS at the load balancer, not in the unikernel)
- Windows Hyper-V target (KVM/QEMU and Firecracker only)
- virtio-gpu (GPU accessed via VFIO passthrough, not virtio-gpu paravirtualization)
- Power management / ACPI (unikernel runs at full power, no sleep states)

## Success Criteria

1. A unikernel built from a tiny MLP model with a serve block boots in QEMU, accepts an HTTP request on port 8080, runs inference, and returns a correct JSON response.
2. The health check endpoint (`GET /health`) responds with `{"status":"ok"}` within 1 second of boot.
3. The unikernel ELF size (excluding weights) is under 100 MB.
4. Boot time from QEMU start to first successful HTTP response is under 500 ms.
5. On Firecracker, boot time is under 200 ms.
6. A CPU-only unikernel (no GPU) runs inference correctly using CPU tensor operations.
7. Weight integrity verification works: a unikernel compiled with `--weights a.safetensors` rejects weights with a different SHA-256 hash.
8. The virtio-net driver correctly sends and receives Ethernet frames with proper checksums.
9. The HTTP parser handles malformed requests gracefully (returns 400, does not crash).
10. Memory usage is bounded: the bump allocator does not exceed the configured `--memory` limit.
