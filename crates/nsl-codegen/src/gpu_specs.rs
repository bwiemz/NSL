//! M37: GPU specification database for roofline analysis.

/// Hardware specifications for a GPU model.
#[derive(Debug, Clone)]
pub struct GpuSpec {
    pub name: &'static str,
    pub sm_version: u32,
    /// Peak FP16 Tensor Core throughput in TFLOPS.
    pub peak_fp16_tflops: f64,
    /// Peak FP8 Tensor Core throughput in TFLOPS (0 if unsupported).
    pub peak_fp8_tflops: f64,
    /// Peak FP32 throughput in TFLOPS (non-Tensor Core).
    pub peak_fp32_tflops: f64,
    /// Peak memory bandwidth in GB/s.
    pub peak_bandwidth_gbs: f64,
    /// VRAM capacity in GB.
    pub vram_gb: f64,
    /// L2 cache size in MB.
    pub l2_cache_mb: f64,
    /// Crossover: peak_fp16_tflops*1e12 / peak_bandwidth_gbs*1e9 (FLOPs/byte).
    pub crossover_fp16: f64,
    /// Crossover for FP8 (0 if unsupported).
    pub crossover_fp8: f64,
    /// Crossover for FP32.
    pub crossover_fp32: f64,
    /// Base clock in MHz (worst case — no boost).
    pub base_clock_mhz: u32,
    /// Kernel launch overhead in nanoseconds (worst case).
    pub kernel_launch_overhead_ns: u64,
    /// cuCtxSynchronize overhead in nanoseconds (worst case).
    pub sync_overhead_ns: u64,
    /// PCIe bandwidth for host-device transfers in GB/s.
    pub pcie_bandwidth_gbps: f64,
    /// Worst-case SM occupancy factor (0.0-1.0).
    pub occupancy_worst_case: f64,
    /// L2 cache size in bytes.
    pub l2_cache_bytes: u64,
    // --- Multi-level memory hierarchy (for cost model) ---
    /// L1 cache / shared memory size per SM in KB.
    pub l1_cache_kb: u32,
    /// Aggregate L1 bandwidth in GB/s (across all SMs).
    pub l1_bandwidth_gbs: f64,
    /// L2 cache bandwidth in GB/s.
    pub l2_bandwidth_gbs: f64,
    /// Maximum concurrent warps per SM.
    pub max_warps_per_sm: u32,
    /// Total 32-bit registers per SM.
    pub registers_per_sm: u32,
    /// Number of SMs.
    pub num_sms: u32,
}

impl GpuSpec {
    /// Crossover point for a given dtype byte width (FLOPs/byte threshold).
    pub fn crossover(&self, dtype_bytes: usize) -> f64 {
        match dtype_bytes {
            1 => self.crossover_fp8,   // FP8
            2 => self.crossover_fp16,  // FP16/BF16
            _ => self.crossover_fp32,  // FP32/F64
        }
    }

    /// Peak TFLOPS for a given dtype byte width.
    pub fn peak_tflops(&self, dtype_bytes: usize) -> f64 {
        match dtype_bytes {
            1 => self.peak_fp8_tflops,
            2 => self.peak_fp16_tflops,
            _ => self.peak_fp32_tflops,
        }
    }

    /// Returns true if the GPU supports FP8 (e4m3/e5m2) tensor core MMA.
    /// Requires sm_90 (H100/H200) or later.
    pub fn supports_fp8_mma(&self) -> bool {
        self.sm_version >= 90
    }

    /// Returns true if the GPU supports FP16 tensor core MMA (mma.sync).
    /// Requires sm_80 (A100) or later.
    pub fn supports_fp16_mma(&self) -> bool {
        self.sm_version >= 80
    }

    /// Returns true if the GPU supports wgmma.mma_async (128-thread warp groups).
    /// Requires sm_90 (H100/H200). Enables ~37% more tensor core utilization
    /// over mma.sync via asynchronous warp group execution.
    pub fn supports_wgmma(&self) -> bool {
        self.sm_version >= 90
    }

    /// Returns true if the GPU supports MXFP8 per-block scaling with E8M0 scales.
    /// Requires sm_100 (Blackwell B100/B200) or later.
    pub fn supports_mxfp8(&self) -> bool {
        self.sm_version >= 100
    }

    /// Returns true if the GPU supports NVFP4 (E2M1) tensor core operations.
    /// Requires sm_100 (Blackwell B100/B200) or later.
    pub fn supports_fp4(&self) -> bool {
        self.sm_version >= 100
    }

    /// Returns the appropriate PTX version string for this GPU's features.
    pub fn ptx_version(&self) -> &'static str {
        if self.sm_version >= 90 { "8.0" }
        else { "7.0" }
    }

    /// Returns the PTX target string for this GPU.
    pub fn ptx_target(&self) -> &'static str {
        if self.sm_version >= 90 { "sm_90" }
        else if self.sm_version >= 89 { "sm_89" }
        else if self.sm_version >= 87 { "sm_87" }
        else if self.sm_version >= 86 { "sm_86" }
        else if self.sm_version >= 80 { "sm_80" }
        else { "sm_52" }
    }

    /// Warp group size: 128 threads on Hopper (4 warps collaborate), 32 on older.
    pub fn warp_group_size(&self) -> u32 {
        if self.sm_version >= 90 { 128 } else { 32 }
    }
}

/// Built-in GPU specification database.
pub const GPU_DATABASE: &[GpuSpec] = &[
    GpuSpec {
        name: "A100-SXM", sm_version: 80,
        peak_fp16_tflops: 312.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 19.5,
        peak_bandwidth_gbs: 2039.0, vram_gb: 80.0, l2_cache_mb: 40.0,
        crossover_fp16: 153.0, crossover_fp8: 0.0, crossover_fp32: 9.6,
        base_clock_mhz: 765, kernel_launch_overhead_ns: 6000, sync_overhead_ns: 3000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 40 * 1024 * 1024,
        l1_cache_kb: 192, l1_bandwidth_gbs: 14400.0, l2_bandwidth_gbs: 4800.0,
        max_warps_per_sm: 64, registers_per_sm: 65536, num_sms: 108,
    },
    GpuSpec {
        name: "A100-PCIe", sm_version: 80,
        peak_fp16_tflops: 312.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 19.5,
        peak_bandwidth_gbs: 1555.0, vram_gb: 40.0, l2_cache_mb: 40.0,
        crossover_fp16: 200.6, crossover_fp8: 0.0, crossover_fp32: 12.5,
        base_clock_mhz: 765, kernel_launch_overhead_ns: 6000, sync_overhead_ns: 3000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 40 * 1024 * 1024,
        l1_cache_kb: 192, l1_bandwidth_gbs: 14400.0, l2_bandwidth_gbs: 4800.0,
        max_warps_per_sm: 64, registers_per_sm: 65536, num_sms: 108,
    },
    GpuSpec {
        name: "H100-SXM", sm_version: 90,
        peak_fp16_tflops: 989.0, peak_fp8_tflops: 1979.0, peak_fp32_tflops: 67.0,
        peak_bandwidth_gbs: 3350.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 295.2, crossover_fp8: 590.7, crossover_fp32: 20.0,
        base_clock_mhz: 1095, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 64.0, occupancy_worst_case: 0.5, l2_cache_bytes: 50 * 1024 * 1024,
        l1_cache_kb: 256, l1_bandwidth_gbs: 19200.0, l2_bandwidth_gbs: 6000.0,
        max_warps_per_sm: 64, registers_per_sm: 65536, num_sms: 132,
    },
    GpuSpec {
        name: "H100-PCIe", sm_version: 90,
        peak_fp16_tflops: 756.0, peak_fp8_tflops: 1513.0, peak_fp32_tflops: 51.0,
        peak_bandwidth_gbs: 2039.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 370.8, crossover_fp8: 741.9, crossover_fp32: 25.0,
        base_clock_mhz: 1095, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 50 * 1024 * 1024,
        l1_cache_kb: 256, l1_bandwidth_gbs: 19200.0, l2_bandwidth_gbs: 6000.0,
        max_warps_per_sm: 64, registers_per_sm: 65536, num_sms: 114,
    },
    GpuSpec {
        name: "RTX-4090", sm_version: 89,
        peak_fp16_tflops: 330.0, peak_fp8_tflops: 661.0, peak_fp32_tflops: 82.6,
        peak_bandwidth_gbs: 1008.0, vram_gb: 24.0, l2_cache_mb: 72.0,
        crossover_fp16: 327.4, crossover_fp8: 655.8, crossover_fp32: 81.9,
        base_clock_mhz: 2235, kernel_launch_overhead_ns: 4000, sync_overhead_ns: 1500,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 72 * 1024 * 1024,
        l1_cache_kb: 128, l1_bandwidth_gbs: 12800.0, l2_bandwidth_gbs: 5600.0,
        max_warps_per_sm: 48, registers_per_sm: 65536, num_sms: 128,
    },
    GpuSpec {
        name: "RTX-3090", sm_version: 86,
        peak_fp16_tflops: 142.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 35.6,
        peak_bandwidth_gbs: 936.2, vram_gb: 24.0, l2_cache_mb: 6.0,
        crossover_fp16: 151.7, crossover_fp8: 0.0, crossover_fp32: 38.0,
        base_clock_mhz: 1395, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 6 * 1024 * 1024,
        l1_cache_kb: 128, l1_bandwidth_gbs: 9600.0, l2_bandwidth_gbs: 2400.0,
        max_warps_per_sm: 48, registers_per_sm: 65536, num_sms: 82,
    },
    GpuSpec {
        name: "L40S", sm_version: 89,
        peak_fp16_tflops: 362.0, peak_fp8_tflops: 733.0, peak_fp32_tflops: 91.6,
        peak_bandwidth_gbs: 864.0, vram_gb: 48.0, l2_cache_mb: 96.0,
        crossover_fp16: 419.0, crossover_fp8: 848.4, crossover_fp32: 106.0,
        base_clock_mhz: 1110, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 48 * 1024 * 1024,
        l1_cache_kb: 128, l1_bandwidth_gbs: 12800.0, l2_bandwidth_gbs: 5600.0,
        max_warps_per_sm: 48, registers_per_sm: 65536, num_sms: 142,
    },
    // NVIDIA Jetson AGX Orin
    GpuSpec {
        name: "Orin", sm_version: 87,
        peak_fp16_tflops: 170.0, peak_fp8_tflops: 170.0, peak_fp32_tflops: 5.3,
        peak_bandwidth_gbs: 204.8, vram_gb: 64.0, l2_cache_mb: 4.0,
        crossover_fp16: 830.0, crossover_fp8: 830.0, crossover_fp32: 25.9,
        base_clock_mhz: 624, kernel_launch_overhead_ns: 8000, sync_overhead_ns: 4000,
        pcie_bandwidth_gbps: 0.0, occupancy_worst_case: 0.4, l2_cache_bytes: 4 * 1024 * 1024,
        l1_cache_kb: 128, l1_bandwidth_gbs: 3200.0, l2_bandwidth_gbs: 800.0,
        max_warps_per_sm: 48, registers_per_sm: 65536, num_sms: 16,
    },
    // NVIDIA Jetson Orin NX (smaller edge)
    GpuSpec {
        name: "Orin-NX", sm_version: 87,
        peak_fp16_tflops: 100.0, peak_fp8_tflops: 100.0, peak_fp32_tflops: 3.1,
        peak_bandwidth_gbs: 102.4, vram_gb: 16.0, l2_cache_mb: 2.0,
        crossover_fp16: 976.0, crossover_fp8: 976.0, crossover_fp32: 30.3,
        base_clock_mhz: 624, kernel_launch_overhead_ns: 8000, sync_overhead_ns: 4000,
        pcie_bandwidth_gbps: 0.0, occupancy_worst_case: 0.35, l2_cache_bytes: 2 * 1024 * 1024,
        l1_cache_kb: 128, l1_bandwidth_gbs: 1600.0, l2_bandwidth_gbs: 400.0,
        max_warps_per_sm: 48, registers_per_sm: 65536, num_sms: 8,
    },
    // NVIDIA B200 (Blackwell, sm_100) — MXFP8 per-block scaling + NVFP4
    GpuSpec {
        name: "B200", sm_version: 100,
        peak_fp16_tflops: 2250.0, peak_fp8_tflops: 4500.0, peak_fp32_tflops: 70.0,
        peak_bandwidth_gbs: 8000.0, vram_gb: 192.0, l2_cache_mb: 128.0,
        crossover_fp16: 281.3, crossover_fp8: 562.5, crossover_fp32: 8.75,
        base_clock_mhz: 1800, kernel_launch_overhead_ns: 3000, sync_overhead_ns: 1200,
        pcie_bandwidth_gbps: 64.0, occupancy_worst_case: 0.6, l2_cache_bytes: 128 * 1024 * 1024,
        l1_cache_kb: 256, l1_bandwidth_gbs: 25600.0, l2_bandwidth_gbs: 12000.0,
        max_warps_per_sm: 64, registers_per_sm: 65536, num_sms: 160,
    },
    // NVIDIA B100 (Blackwell, sm_100) — MXFP8 + NVFP4 (PCIe variant)
    GpuSpec {
        name: "B100", sm_version: 100,
        peak_fp16_tflops: 1750.0, peak_fp8_tflops: 3500.0, peak_fp32_tflops: 56.0,
        peak_bandwidth_gbs: 6400.0, vram_gb: 192.0, l2_cache_mb: 96.0,
        crossover_fp16: 273.4, crossover_fp8: 546.9, crossover_fp32: 8.75,
        base_clock_mhz: 1600, kernel_launch_overhead_ns: 3500, sync_overhead_ns: 1500,
        pcie_bandwidth_gbps: 64.0, occupancy_worst_case: 0.55, l2_cache_bytes: 96 * 1024 * 1024,
        l1_cache_kb: 256, l1_bandwidth_gbs: 20000.0, l2_bandwidth_gbs: 10000.0,
        max_warps_per_sm: 64, registers_per_sm: 65536, num_sms: 128,
    },
];

/// Find a GPU by name. Case-insensitive prefix match; prefers SXM variants.
pub fn find_gpu(name: &str) -> Option<&'static GpuSpec> {
    let name_upper = name.to_uppercase().replace(' ', "-");

    // Exact match first
    if let Some(gpu) = GPU_DATABASE.iter().find(|g| g.name.to_uppercase() == name_upper) {
        return Some(gpu);
    }

    // Prefix match
    let matches: Vec<&GpuSpec> = GPU_DATABASE
        .iter()
        .filter(|g| g.name.to_uppercase().starts_with(&name_upper))
        .collect();

    match matches.len() {
        0 => None,
        1 => Some(matches[0]),
        _ => matches.iter().find(|g| g.name.contains("SXM")).copied().or(Some(matches[0])),
    }
}

/// Default GPU when none specified and auto-detect unavailable.
pub fn default_gpu() -> &'static GpuSpec {
    find_gpu("A100-SXM").unwrap()
}

/// Hardware specifications for a CPU model (used for WCET analysis on CPU targets).
#[derive(Debug, Clone)]
pub struct CpuSpec {
    pub name: &'static str,
    pub base_clock_mhz: u32,
    pub fp32_flops_per_cycle: u32,
    pub fp16_flops_per_cycle: Option<u32>,
    pub l1d_cache_bytes: u32,
    pub l2_cache_bytes: u64,
    pub l3_cache_bytes: Option<u64>,
    pub memory_bandwidth_gbps: f64,
    pub cache_line_bytes: u32,
    pub num_cores: u32,
}

/// Built-in CPU specification database for WCET analysis.
pub const CPU_DATABASE: &[CpuSpec] = &[
    CpuSpec {
        name: "cortex-a78",
        base_clock_mhz: 2000,
        fp32_flops_per_cycle: 8,
        fp16_flops_per_cycle: Some(16),
        l1d_cache_bytes: 64 * 1024,
        l2_cache_bytes: 512 * 1024,
        l3_cache_bytes: Some(4 * 1024 * 1024),
        memory_bandwidth_gbps: 51.2,
        cache_line_bytes: 64,
        num_cores: 4,
    },
    CpuSpec {
        name: "x86-64-v4",
        base_clock_mhz: 2100,
        fp32_flops_per_cycle: 32,
        fp16_flops_per_cycle: Some(64),
        l1d_cache_bytes: 48 * 1024,
        l2_cache_bytes: 2 * 1024 * 1024,
        l3_cache_bytes: Some(36 * 1024 * 1024),
        memory_bandwidth_gbps: 102.4,
        cache_line_bytes: 64,
        num_cores: 16,
    },
];

/// Find a CPU by name. Exact match only.
pub fn find_cpu(name: &str) -> Option<&'static CpuSpec> {
    CPU_DATABASE.iter().find(|c| c.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_gpu_exact() {
        let gpu = find_gpu("H100-SXM").unwrap();
        assert_eq!(gpu.name, "H100-SXM");
        assert_eq!(gpu.sm_version, 90);
    }

    #[test]
    fn test_find_gpu_case_insensitive() {
        let gpu = find_gpu("h100-sxm").unwrap();
        assert_eq!(gpu.name, "H100-SXM");
    }

    #[test]
    fn test_find_gpu_prefix_prefers_sxm() {
        let gpu = find_gpu("H100").unwrap();
        assert_eq!(gpu.name, "H100-SXM");
    }

    #[test]
    fn test_find_gpu_prefix_unique() {
        let gpu = find_gpu("RTX-4090").unwrap();
        assert_eq!(gpu.name, "RTX-4090");
    }

    #[test]
    fn test_find_gpu_not_found() {
        assert!(find_gpu("NONEXISTENT").is_none());
    }

    #[test]
    fn test_default_gpu() {
        assert_eq!(default_gpu().name, "A100-SXM");
    }

    #[test]
    fn test_crossover_by_dtype() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert!((h100.crossover(2) - 295.2).abs() < 0.1);
        assert!((h100.crossover(1) - 590.7).abs() < 0.1);
        assert!((h100.crossover(4) - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_peak_tflops_by_dtype() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert!((h100.peak_tflops(2) - 989.0).abs() < 0.1);
        assert!((h100.peak_tflops(1) - 1979.0).abs() < 0.1);
        assert!((h100.peak_tflops(4) - 67.0).abs() < 0.1);
    }

    #[test]
    fn test_database_has_all_gpus() {
        assert_eq!(GPU_DATABASE.len(), 9);
        let names: Vec<&str> = GPU_DATABASE.iter().map(|g| g.name).collect();
        assert!(names.contains(&"A100-SXM"));
        assert!(names.contains(&"H100-SXM"));
        assert!(names.contains(&"RTX-4090"));
        assert!(names.contains(&"RTX-3090"));
        assert!(names.contains(&"L40S"));
        assert!(names.contains(&"Orin"));
        assert!(names.contains(&"Orin-NX"));
    }

    #[test]
    fn test_wcet_fields_present() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert_eq!(h100.base_clock_mhz, 1095);
        assert_eq!(h100.kernel_launch_overhead_ns, 5000);
        assert_eq!(h100.sync_overhead_ns, 2000);
        assert!((h100.pcie_bandwidth_gbps - 64.0).abs() < 0.1);
        assert!((h100.occupancy_worst_case - 0.5).abs() < 0.01);
        assert_eq!(h100.l2_cache_bytes, 50 * 1024 * 1024);
    }

    #[test]
    fn test_edge_gpu_orin() {
        let orin = find_gpu("Orin").unwrap();
        assert_eq!(orin.name, "Orin");
        assert_eq!(orin.sm_version, 87);
        assert_eq!(orin.base_clock_mhz, 624);
        assert!((orin.pcie_bandwidth_gbps).abs() < 0.01); // unified memory, no PCIe
        assert!((orin.occupancy_worst_case - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_find_cpu() {
        let arm = find_cpu("cortex-a78").unwrap();
        assert_eq!(arm.base_clock_mhz, 2000);
        assert_eq!(arm.fp32_flops_per_cycle, 8);
        assert_eq!(arm.fp16_flops_per_cycle, Some(16));
        assert_eq!(arm.num_cores, 4);

        let x86 = find_cpu("x86-64-v4").unwrap();
        assert_eq!(x86.base_clock_mhz, 2100);
        assert_eq!(x86.fp32_flops_per_cycle, 32);
        assert_eq!(x86.num_cores, 16);

        assert!(find_cpu("nonexistent").is_none());
    }

    #[test]
    fn test_cpu_database_size() {
        assert_eq!(CPU_DATABASE.len(), 2);
    }

    #[test]
    fn test_fp8_mma_requires_sm90() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert!(h100.supports_fp8_mma(), "H100 (sm_90) should support FP8 MMA");

        let a100 = find_gpu("A100-SXM").unwrap();
        assert!(!a100.supports_fp8_mma(), "A100 (sm_80) should NOT support FP8 MMA");

        let rtx4090 = find_gpu("RTX-4090").unwrap();
        assert!(!rtx4090.supports_fp8_mma(), "RTX-4090 (sm_89) should NOT support FP8 MMA");
    }

    #[test]
    fn test_fp16_mma_requires_sm80() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert!(h100.supports_fp16_mma(), "H100 should support FP16 MMA");

        let a100 = find_gpu("A100-SXM").unwrap();
        assert!(a100.supports_fp16_mma(), "A100 should support FP16 MMA");

        let rtx3090 = find_gpu("RTX-3090").unwrap();
        assert!(rtx3090.supports_fp16_mma(), "RTX-3090 (sm_86) should support FP16 MMA");

        let orin = find_gpu("Orin").unwrap();
        assert!(orin.supports_fp16_mma(), "Orin (sm_87) should support FP16 MMA");
    }

    #[test]
    fn test_wgmma_requires_sm90() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert!(h100.supports_wgmma(), "H100 (sm_90) should support wgmma");

        let a100 = find_gpu("A100-SXM").unwrap();
        assert!(!a100.supports_wgmma(), "A100 (sm_80) should NOT support wgmma");

        let rtx4090 = find_gpu("RTX-4090").unwrap();
        assert!(!rtx4090.supports_wgmma(), "RTX-4090 (sm_89) should NOT support wgmma");
    }

    #[test]
    fn test_ptx_target_selection() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert_eq!(h100.ptx_target(), "sm_90");
        assert_eq!(h100.ptx_version(), "8.0");

        let a100 = find_gpu("A100-SXM").unwrap();
        assert_eq!(a100.ptx_target(), "sm_80");
        assert_eq!(a100.ptx_version(), "7.0");

        let rtx4090 = find_gpu("RTX-4090").unwrap();
        assert_eq!(rtx4090.ptx_target(), "sm_89");
    }

    #[test]
    fn test_warp_group_size() {
        let h100 = find_gpu("H100-SXM").unwrap();
        assert_eq!(h100.warp_group_size(), 128, "Hopper uses 128-thread warp groups");

        let a100 = find_gpu("A100-SXM").unwrap();
        assert_eq!(a100.warp_group_size(), 32, "Ampere uses 32-thread warps");
    }
}
