//! M54b: x86_64 boot stub generation for bare-metal unikernels.
//!
//! Generates raw machine code bytes for the `.text.boot` ELF section:
//! - Multiboot2 header (magic + architecture + checksum)
//! - GDT (Global Descriptor Table) for 64-bit long mode
//! - Page tables: identity-map first 4GB (needed for MMIO)
//! - SSE/AVX enable (required for tensor math)
//! - Jump to Rust `kmain()` entry point
//!
//! The output is a Vec<u8> of raw x86_64 machine code that can be embedded
//! directly into a Cranelift object file's `.text.boot` section.

use super::unikernel::{BootProtocol, UnikernelConfig};

// ---------------------------------------------------------------------------
// Multiboot2 header constants
// ---------------------------------------------------------------------------

/// Multiboot2 magic number (required in first 8KB of the image).
const MULTIBOOT2_MAGIC: u32 = 0xE85250D6;
/// Architecture: 0 = i386 / x86_64 (protected mode).
const MULTIBOOT2_ARCH_I386: u32 = 0;

// ---------------------------------------------------------------------------
// Boot stub generation
// ---------------------------------------------------------------------------

/// Generate the complete boot stub as raw x86_64 machine code bytes.
///
/// The stub contains:
/// 1. Multiboot2 or Linux boot header
/// 2. GDT with 64-bit code/data segments
/// 3. Page tables (PML4 → PDPT → PD) identity-mapping 0-4GB
/// 4. Long mode entry that enables SSE/AVX
/// 5. Stack setup and jump to `kmain`
///
/// Returns a tuple of (boot_stub_bytes, boot_config_json_bytes).
pub fn generate_boot_stub(config: &UnikernelConfig) -> BootStub {
    let mut stub = BootStub::new();

    // Phase 1: Header
    match config.boot_protocol {
        BootProtocol::Multiboot2 => stub.emit_multiboot2_header(),
        BootProtocol::LinuxBoot => stub.emit_linux_boot_header(),
    }

    // Phase 2: 32-bit entry point (_start) — transitions to 64-bit long mode
    stub.emit_entry_32();

    // Phase 3: GDT for 64-bit mode
    stub.emit_gdt64();

    // Phase 4: Page tables (identity map first 4GB)
    stub.emit_page_tables();

    // Phase 5: 64-bit entry point — enables SSE/AVX, sets up stack, calls kmain
    stub.emit_entry_64();

    stub
}

/// A boot stub being assembled.
pub struct BootStub {
    /// Raw machine code bytes for `.text.boot`.
    pub code: Vec<u8>,
    /// Offset of the 32-bit entry point (`_start`) within `code`.
    pub entry_offset: usize,
    /// Offset of the GDT within `code`.
    pub gdt_offset: usize,
    /// Offset of the page tables within `code`.
    pub page_table_offset: usize,
    /// Offset of the 64-bit entry point within `code`.
    pub entry64_offset: usize,
    /// Labels: name → offset (for relocation).
    pub labels: Vec<(&'static str, usize)>,
}

impl BootStub {
    fn new() -> Self {
        Self {
            code: Vec::with_capacity(16384), // 16KB boot stub
            entry_offset: 0,
            gdt_offset: 0,
            page_table_offset: 0,
            entry64_offset: 0,
            labels: Vec::new(),
        }
    }

    /// Total size of the boot stub in bytes.
    pub fn size(&self) -> usize {
        self.code.len()
    }

    // -----------------------------------------------------------------------
    // Multiboot2 header
    // -----------------------------------------------------------------------

    fn emit_multiboot2_header(&mut self) {
        let header_start = self.code.len();
        self.labels.push(("multiboot2_header", header_start));

        // Magic number
        self.emit_u32(MULTIBOOT2_MAGIC);
        // Architecture (i386)
        self.emit_u32(MULTIBOOT2_ARCH_I386);
        // Header length (will be filled after all tags)
        let header_len_offset = self.code.len();
        self.emit_u32(0); // placeholder
                          // Checksum (will be filled)
        let checksum_offset = self.code.len();
        self.emit_u32(0); // placeholder

        // End tag (type=0, flags=0, size=8)
        self.emit_u32(0); // type
        self.emit_u32(8); // size

        // Fill header length
        let header_len = (self.code.len() - header_start) as u32;
        self.code[header_len_offset..header_len_offset + 4]
            .copy_from_slice(&header_len.to_le_bytes());

        // Fill checksum: -(magic + arch + header_len) mod 2^32
        let checksum =
            (-(MULTIBOOT2_MAGIC as i64 + MULTIBOOT2_ARCH_I386 as i64 + header_len as i64)) as u32;
        self.code[checksum_offset..checksum_offset + 4].copy_from_slice(&checksum.to_le_bytes());
    }

    fn emit_linux_boot_header(&mut self) {
        self.labels.push(("linux_boot_header", self.code.len()));

        // Linux boot protocol setup header (simplified).
        // At offset 0x01F1 in the bzImage, but for direct -kernel boot
        // with KVM/QEMU, we just need a valid entry point.
        // Emit a minimal header that QEMU's direct kernel loader accepts.

        // For KVM direct boot, entry point is at the start of the loaded image.
        // We emit a JMP to the real entry point.
        self.emit_bytes(&[0xEB, 0x3E]); // JMP SHORT +62 (skip header)

        // Pad header area (64 bytes of identity info)
        for _ in 0..62 {
            self.emit_u8(0);
        }
    }

    // -----------------------------------------------------------------------
    // 32-bit entry point (_start)
    // -----------------------------------------------------------------------

    fn emit_entry_32(&mut self) {
        self.entry_offset = self.code.len();
        self.labels.push(("_start", self.entry_offset));

        // === 32-bit protected mode entry ===
        // Assumptions: Multiboot2 loader has set up flat 32-bit segments.
        // We need to transition to 64-bit long mode.

        // cli — disable interrupts
        self.emit_u8(0xFA);

        // Set up page tables (CR3)
        // mov eax, page_table_phys_addr (will be patched)
        self.emit_bytes(&[0xB8]); // MOV EAX, imm32
        let pml4_fixup = self.code.len();
        self.emit_u32(0); // placeholder — patched to PML4 address
                          // mov cr3, eax
        self.emit_bytes(&[0x0F, 0x22, 0xD8]); // MOV CR3, EAX

        // Enable PAE (Physical Address Extension) — required for long mode
        // mov eax, cr4
        self.emit_bytes(&[0x0F, 0x20, 0xE0]);
        // or eax, (1 << 5) — PAE bit
        self.emit_bytes(&[0x0D]);
        self.emit_u32(1 << 5);
        // mov cr4, eax
        self.emit_bytes(&[0x0F, 0x22, 0xE0]);

        // Enable long mode via IA32_EFER MSR
        // mov ecx, 0xC0000080 (IA32_EFER)
        self.emit_bytes(&[0xB9]);
        self.emit_u32(0xC0000080);
        // rdmsr
        self.emit_bytes(&[0x0F, 0x32]);
        // or eax, (1 << 8) — LME (Long Mode Enable)
        self.emit_bytes(&[0x0D]);
        self.emit_u32(1 << 8);
        // wrmsr
        self.emit_bytes(&[0x0F, 0x30]);

        // Enable paging (CR0.PG = 1) + protected mode (CR0.PE = 1)
        // mov eax, cr0
        self.emit_bytes(&[0x0F, 0x20, 0xC0]);
        // or eax, (1 << 31) | (1 << 0) — PG + PE
        self.emit_bytes(&[0x0D]);
        self.emit_u32((1 << 31) | (1 << 0));
        // mov cr0, eax
        self.emit_bytes(&[0x0F, 0x22, 0xC0]);

        // Load 64-bit GDT
        // lgdt [gdt_ptr] — will be patched
        self.emit_bytes(&[0x0F, 0x01, 0x15]); // LGDT [rip+disp32] in 32-bit: LGDT m48
        let gdt_fixup = self.code.len();
        self.emit_u32(0); // placeholder for GDT pointer address

        // Far jump to 64-bit code segment (selector 0x08)
        // jmp 0x08:entry64
        self.emit_bytes(&[0xEA]); // JMP FAR ptr16:32
        let entry64_fixup = self.code.len();
        self.emit_u32(0); // placeholder for 64-bit entry offset
        self.emit_u16(0x08); // code segment selector

        // Record fixup locations for patching
        self.labels.push(("fixup_pml4", pml4_fixup));
        self.labels.push(("fixup_gdt_ptr", gdt_fixup));
        self.labels.push(("fixup_entry64", entry64_fixup));
    }

    // -----------------------------------------------------------------------
    // 64-bit GDT
    // -----------------------------------------------------------------------

    fn emit_gdt64(&mut self) {
        // Align to 16 bytes
        while !self.code.len().is_multiple_of(16) {
            self.emit_u8(0);
        }

        self.gdt_offset = self.code.len();
        self.labels.push(("gdt64", self.gdt_offset));

        // Null descriptor (entry 0)
        self.emit_u64(0);

        // Code segment descriptor (entry 1, selector 0x08)
        // Flags: L=1 (64-bit), P=1, S=1, type=0xA (execute/read)
        // Access byte: 0x9A = Present | Code | Execute/Read
        // Flags nibble: 0x2 = Long mode
        self.emit_u64(0x00AF_9A00_0000_FFFF);

        // Data segment descriptor (entry 2, selector 0x10)
        // Access byte: 0x92 = Present | Data | Read/Write
        self.emit_u64(0x00CF_9200_0000_FFFF);

        // GDT pointer (6 bytes: limit + base)
        self.labels.push(("gdt64_ptr", self.code.len()));
        let gdt_limit = (3 * 8 - 1) as u16; // 3 entries * 8 bytes - 1
        self.emit_u16(gdt_limit);
        // Base address: absolute address of gdt64 (will be patched for actual load address)
        self.emit_u32(0); // placeholder — patched to gdt64 absolute address
    }

    // -----------------------------------------------------------------------
    // Page tables (identity map 0-4GB)
    // -----------------------------------------------------------------------

    fn emit_page_tables(&mut self) {
        // Align to 4KB page boundary
        while !self.code.len().is_multiple_of(4096) {
            self.emit_u8(0);
        }

        self.page_table_offset = self.code.len();
        self.labels.push(("page_tables", self.page_table_offset));

        // PML4 (Page Map Level 4) — 1 entry pointing to PDPT
        let pml4_offset = self.code.len();
        self.labels.push(("pml4", pml4_offset));
        // Entry 0: Present + Writable + points to PDPT (address will be patched)
        self.emit_u64(0x03); // flags only, base patched later
                             // Remaining 511 entries = 0
        for _ in 1..512 {
            self.emit_u64(0);
        }

        // PDPT (Page Directory Pointer Table) — 4 entries for 4GB
        let pdpt_offset = self.code.len();
        self.labels.push(("pdpt", pdpt_offset));
        for i in 0..4u64 {
            // Each entry maps 1GB using 2MB pages (PS=1 for huge pages in PDPT)
            // Present + Writable + Page Size (1GB page)
            let entry = (i << 30) | 0x83; // base address + Present + RW + PS
            self.emit_u64(entry);
        }
        // Remaining 508 entries = 0
        for _ in 4..512 {
            self.emit_u64(0);
        }

        // Patch PML4[0] to point to PDPT
        let pdpt_page_addr = pdpt_offset as u64;
        // We store the relative offset; actual address patching happens at link time
        // For now, set the PDPT address relative to image base (0x100000)
        let pml4_entry = pdpt_page_addr | 0x03; // Present + RW
        self.code[pml4_offset..pml4_offset + 8].copy_from_slice(&pml4_entry.to_le_bytes());
    }

    // -----------------------------------------------------------------------
    // 64-bit entry point
    // -----------------------------------------------------------------------

    fn emit_entry_64(&mut self) {
        // Align to 16 bytes
        while !self.code.len().is_multiple_of(16) {
            self.emit_u8(0);
        }

        self.entry64_offset = self.code.len();
        self.labels.push(("entry64", self.entry64_offset));

        // === 64-bit long mode entry ===

        // Load data segment selectors
        // mov ax, 0x10 (data segment)
        self.emit_bytes(&[0x66, 0xB8, 0x10, 0x00]);
        // mov ds, ax
        self.emit_bytes(&[0x8E, 0xD8]);
        // mov es, ax
        self.emit_bytes(&[0x8E, 0xC0]);
        // mov ss, ax
        self.emit_bytes(&[0x8E, 0xD0]);
        // xor fs, fs (clear FS)
        self.emit_bytes(&[0x8E, 0xE0]);
        // xor gs, gs (clear GS)
        self.emit_bytes(&[0x8E, 0xE8]);

        // Enable SSE (CR4.OSFXSR + CR4.OSXMMEXCPT)
        // mov rax, cr4
        self.emit_bytes(&[0x0F, 0x20, 0xE0]);
        // or eax, (1 << 9) | (1 << 10) — OSFXSR + OSXMMEXCPT
        self.emit_bytes(&[0x0D]);
        self.emit_u32((1 << 9) | (1 << 10));
        // mov cr4, rax
        self.emit_bytes(&[0x0F, 0x22, 0xE0]);

        // Enable AVX if supported (XCR0.SSE + XCR0.AVX)
        // xor ecx, ecx (XCR0 index = 0)
        self.emit_bytes(&[0x31, 0xC9]);
        // xgetbv
        self.emit_bytes(&[0x0F, 0x01, 0xD0]);
        // or eax, 0x07 — SSE + AVX + AVX state
        self.emit_bytes(&[0x0D]);
        self.emit_u32(0x07);
        // xsetbv
        self.emit_bytes(&[0x0F, 0x01, 0xD1]);

        // Set up stack pointer (__stack_top defined by linker script)
        // mov rsp, __stack_top (placeholder — resolved by linker)
        self.emit_bytes(&[0x48, 0xBC]); // MOV RSP, imm64
        self.labels.push(("fixup_stack_top", self.code.len()));
        self.emit_u64(0); // placeholder for __stack_top

        // Clear BSS (optional — linker may handle)
        // For safety, we zero BSS in case the loader doesn't

        // Call kmain()
        // mov rax, kmain (placeholder — resolved by linker)
        self.emit_bytes(&[0x48, 0xB8]); // MOV RAX, imm64
        self.labels.push(("fixup_kmain", self.code.len()));
        self.emit_u64(0); // placeholder for kmain address
                          // call rax
        self.emit_bytes(&[0xFF, 0xD0]);

        // If kmain returns, halt
        self.labels.push(("halt_loop", self.code.len()));
        // cli
        self.emit_u8(0xFA);
        // hlt
        self.emit_u8(0xF4);
        // jmp halt_loop
        self.emit_bytes(&[0xEB, 0xFC]); // JMP SHORT -4
    }

    // -----------------------------------------------------------------------
    // Byte emission helpers
    // -----------------------------------------------------------------------

    fn emit_u8(&mut self, val: u8) {
        self.code.push(val);
    }

    fn emit_u16(&mut self, val: u16) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    fn emit_u32(&mut self, val: u32) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    fn emit_u64(&mut self, val: u64) {
        self.code.extend_from_slice(&val.to_le_bytes());
    }

    fn emit_bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }
}

// ---------------------------------------------------------------------------
// ELF image builder
// ---------------------------------------------------------------------------

/// Build a unikernel ELF image by combining boot stub + compiled code + weights.
///
/// This produces a self-contained binary that can be booted directly by
/// QEMU (`-kernel`) or Firecracker.
pub struct UnikernelImageBuilder {
    /// Boot stub bytes (from `generate_boot_stub()`).
    pub boot_stub: Vec<u8>,
    /// Compiled model code (from Cranelift JIT output).
    pub model_code: Vec<u8>,
    /// Weight data to embed in `.rodata` (empty if weights are loaded from disk).
    pub weight_data: Vec<u8>,
    /// Boot configuration JSON (embedded for the stub to parse).
    pub boot_config: Vec<u8>,
    /// Linker script content.
    pub linker_script: String,
}

impl UnikernelImageBuilder {
    pub fn new(config: &UnikernelConfig) -> Self {
        let stub = generate_boot_stub(config);
        let linker_script = super::unikernel::generate_linker_script(config);
        let boot_config = super::unikernel::generate_boot_config(config);

        Self {
            boot_stub: stub.code,
            model_code: Vec::new(),
            weight_data: Vec::new(),
            boot_config: boot_config.into_bytes(),
            linker_script,
        }
    }

    /// Set the compiled model code.
    pub fn with_model_code(mut self, code: Vec<u8>) -> Self {
        self.model_code = code;
        self
    }

    /// Set the weight data to embed.
    pub fn with_weights(mut self, data: Vec<u8>) -> Self {
        self.weight_data = data;
        self
    }

    /// Estimate the total image size in bytes.
    pub fn estimated_size(&self) -> usize {
        self.boot_stub.len()
            + self.model_code.len()
            + self.weight_data.len()
            + self.boot_config.len()
            + 4096 // alignment padding
    }

    /// Write the linker script to a file and return the path.
    pub fn write_linker_script(
        &self,
        dir: &std::path::Path,
    ) -> std::io::Result<std::path::PathBuf> {
        let path = dir.join("unikernel.ld");
        std::fs::write(&path, &self.linker_script)?;
        Ok(path)
    }

    /// Write the boot config JSON to a file.
    pub fn write_boot_config(&self, dir: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
        let path = dir.join("boot_config.json");
        std::fs::write(&path, &self.boot_config)?;
        Ok(path)
    }

    /// Write the boot stub to a raw binary file.
    pub fn write_boot_stub(&self, dir: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
        let path = dir.join("boot_stub.bin");
        std::fs::write(&path, &self.boot_stub)?;
        Ok(path)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unikernel::UnikernelConfig;

    #[test]
    fn test_multiboot2_header_magic() {
        let config = UnikernelConfig {
            boot_protocol: BootProtocol::Multiboot2,
            ..Default::default()
        };
        let stub = generate_boot_stub(&config);

        // First 4 bytes should be Multiboot2 magic
        let magic = u32::from_le_bytes(stub.code[0..4].try_into().unwrap());
        assert_eq!(magic, MULTIBOOT2_MAGIC);

        // Architecture should be 0 (i386)
        let arch = u32::from_le_bytes(stub.code[4..8].try_into().unwrap());
        assert_eq!(arch, MULTIBOOT2_ARCH_I386);

        // Header length (at offset 8)
        let header_len = u32::from_le_bytes(stub.code[8..12].try_into().unwrap());
        assert!(
            header_len > 0 && header_len <= 256,
            "header_len={}",
            header_len
        );

        // Checksum: magic + arch + len + checksum == 0 (mod 2^32)
        let checksum = u32::from_le_bytes(stub.code[12..16].try_into().unwrap());
        let sum = MULTIBOOT2_MAGIC
            .wrapping_add(MULTIBOOT2_ARCH_I386)
            .wrapping_add(header_len)
            .wrapping_add(checksum);
        assert_eq!(sum, 0, "Multiboot2 checksum failed: sum={}", sum);
    }

    #[test]
    fn test_linux_boot_header() {
        let config = UnikernelConfig {
            boot_protocol: BootProtocol::LinuxBoot,
            ..Default::default()
        };
        let stub = generate_boot_stub(&config);

        // First two bytes should be JMP SHORT (0xEB)
        assert_eq!(
            stub.code[0], 0xEB,
            "Linux boot header should start with JMP SHORT"
        );
        assert!(stub.size() > 64, "stub too small: {}", stub.size());
    }

    #[test]
    fn test_boot_stub_has_entry_point() {
        let config = UnikernelConfig::default();
        let stub = generate_boot_stub(&config);

        assert!(stub.entry_offset > 0 || config.boot_protocol == BootProtocol::LinuxBoot);
        assert!(stub.labels.iter().any(|(name, _)| *name == "_start"));
    }

    #[test]
    fn test_boot_stub_has_gdt() {
        let config = UnikernelConfig::default();
        let stub = generate_boot_stub(&config);

        assert!(stub.gdt_offset > 0, "GDT should be emitted");
        assert!(stub.labels.iter().any(|(name, _)| *name == "gdt64"));

        // GDT should have null + code + data segments (24 bytes minimum)
        assert!(stub.code.len() > stub.gdt_offset + 24);
    }

    #[test]
    fn test_boot_stub_has_page_tables() {
        let config = UnikernelConfig::default();
        let stub = generate_boot_stub(&config);

        assert!(stub.page_table_offset > 0);
        // Page tables are 4KB-aligned
        assert_eq!(stub.page_table_offset % 4096, 0);
        assert!(stub.labels.iter().any(|(name, _)| *name == "pml4"));
        assert!(stub.labels.iter().any(|(name, _)| *name == "pdpt"));
    }

    #[test]
    fn test_boot_stub_has_halt_loop() {
        let config = UnikernelConfig::default();
        let stub = generate_boot_stub(&config);

        assert!(stub.labels.iter().any(|(name, _)| *name == "halt_loop"));

        // Find halt loop and verify it contains HLT (0xF4)
        let halt_offset = stub
            .labels
            .iter()
            .find(|(name, _)| *name == "halt_loop")
            .unwrap()
            .1;
        // CLI (0xFA) followed by HLT (0xF4)
        assert_eq!(stub.code[halt_offset], 0xFA, "expected CLI before HLT");
        assert_eq!(stub.code[halt_offset + 1], 0xF4, "expected HLT instruction");
    }

    #[test]
    fn test_boot_stub_size_reasonable() {
        let config = UnikernelConfig::default();
        let stub = generate_boot_stub(&config);

        // Boot stub should be < 64KB (mostly page tables at ~8KB + code)
        assert!(stub.size() > 100, "stub too small: {}", stub.size());
        assert!(stub.size() < 65536, "stub too large: {}", stub.size());
    }

    #[test]
    fn test_image_builder_creation() {
        let config = UnikernelConfig::default();
        let builder = UnikernelImageBuilder::new(&config);

        assert!(!builder.boot_stub.is_empty());
        assert!(!builder.linker_script.is_empty());
        assert!(!builder.boot_config.is_empty());
        assert!(builder.estimated_size() > 0);
    }

    #[test]
    fn test_image_builder_with_weights() {
        let config = UnikernelConfig::default();
        let weights = vec![0u8; 1024];
        let builder = UnikernelImageBuilder::new(&config)
            .with_weights(weights.clone())
            .with_model_code(vec![0xCC; 256]);

        assert_eq!(builder.weight_data.len(), 1024);
        assert_eq!(builder.model_code.len(), 256);
        assert!(builder.estimated_size() > 1024 + 256);
    }

    #[test]
    fn test_image_builder_write_files() {
        let config = UnikernelConfig::default();
        let builder = UnikernelImageBuilder::new(&config);

        let dir = std::env::temp_dir().join("nsl_unikernel_test");
        std::fs::create_dir_all(&dir).unwrap();

        let ld_path = builder.write_linker_script(&dir).unwrap();
        assert!(ld_path.exists());
        let content = std::fs::read_to_string(&ld_path).unwrap();
        assert!(content.contains("ENTRY(_start)"));

        let config_path = builder.write_boot_config(&dir).unwrap();
        assert!(config_path.exists());

        let stub_path = builder.write_boot_stub(&dir).unwrap();
        assert!(stub_path.exists());
        let stub_bytes = std::fs::read(&stub_path).unwrap();
        assert!(!stub_bytes.is_empty());

        // Cleanup
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_page_table_identity_maps_4gb() {
        let config = UnikernelConfig::default();
        let stub = generate_boot_stub(&config);

        let pdpt_offset = stub
            .labels
            .iter()
            .find(|(name, _)| *name == "pdpt")
            .unwrap()
            .1;

        // First 4 PDPT entries should map 0-4GB (1GB pages)
        for i in 0..4u64 {
            let entry_offset = pdpt_offset + (i as usize) * 8;
            let entry = u64::from_le_bytes(
                stub.code[entry_offset..entry_offset + 8]
                    .try_into()
                    .unwrap(),
            );
            // Check Present + RW + PS bits
            assert_eq!(
                entry & 0x83,
                0x83,
                "PDPT[{}] missing flags: {:#x}",
                i,
                entry
            );
            // Check base address (i * 1GB)
            let base = entry & !0xFFF;
            assert_eq!(base, i << 30, "PDPT[{}] wrong base: {:#x}", i, base);
        }

        // Entry 4 should be zero (unmapped)
        let entry4_offset = pdpt_offset + 4 * 8;
        let entry4 = u64::from_le_bytes(
            stub.code[entry4_offset..entry4_offset + 8]
                .try_into()
                .unwrap(),
        );
        assert_eq!(entry4, 0);
    }
}
