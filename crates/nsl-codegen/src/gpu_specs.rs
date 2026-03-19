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
    },
    GpuSpec {
        name: "A100-PCIe", sm_version: 80,
        peak_fp16_tflops: 312.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 19.5,
        peak_bandwidth_gbs: 1555.0, vram_gb: 40.0, l2_cache_mb: 40.0,
        crossover_fp16: 200.6, crossover_fp8: 0.0, crossover_fp32: 12.5,
        base_clock_mhz: 765, kernel_launch_overhead_ns: 6000, sync_overhead_ns: 3000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 40 * 1024 * 1024,
    },
    GpuSpec {
        name: "H100-SXM", sm_version: 90,
        peak_fp16_tflops: 989.0, peak_fp8_tflops: 1979.0, peak_fp32_tflops: 67.0,
        peak_bandwidth_gbs: 3350.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 295.2, crossover_fp8: 590.7, crossover_fp32: 20.0,
        base_clock_mhz: 1095, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 64.0, occupancy_worst_case: 0.5, l2_cache_bytes: 50 * 1024 * 1024,
    },
    GpuSpec {
        name: "H100-PCIe", sm_version: 90,
        peak_fp16_tflops: 756.0, peak_fp8_tflops: 1513.0, peak_fp32_tflops: 51.0,
        peak_bandwidth_gbs: 2039.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 370.8, crossover_fp8: 741.9, crossover_fp32: 25.0,
        base_clock_mhz: 1095, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 50 * 1024 * 1024,
    },
    GpuSpec {
        name: "RTX-4090", sm_version: 89,
        peak_fp16_tflops: 330.0, peak_fp8_tflops: 661.0, peak_fp32_tflops: 82.6,
        peak_bandwidth_gbs: 1008.0, vram_gb: 24.0, l2_cache_mb: 72.0,
        crossover_fp16: 327.4, crossover_fp8: 655.8, crossover_fp32: 81.9,
        base_clock_mhz: 2235, kernel_launch_overhead_ns: 4000, sync_overhead_ns: 1500,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 72 * 1024 * 1024,
    },
    GpuSpec {
        name: "RTX-3090", sm_version: 86,
        peak_fp16_tflops: 142.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 35.6,
        peak_bandwidth_gbs: 936.2, vram_gb: 24.0, l2_cache_mb: 6.0,
        crossover_fp16: 151.7, crossover_fp8: 0.0, crossover_fp32: 38.0,
        base_clock_mhz: 1395, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 6 * 1024 * 1024,
    },
    GpuSpec {
        name: "L40S", sm_version: 89,
        peak_fp16_tflops: 362.0, peak_fp8_tflops: 733.0, peak_fp32_tflops: 91.6,
        peak_bandwidth_gbs: 864.0, vram_gb: 48.0, l2_cache_mb: 96.0,
        crossover_fp16: 419.0, crossover_fp8: 848.4, crossover_fp32: 106.0,
        base_clock_mhz: 1110, kernel_launch_overhead_ns: 5000, sync_overhead_ns: 2000,
        pcie_bandwidth_gbps: 32.0, occupancy_worst_case: 0.5, l2_cache_bytes: 48 * 1024 * 1024,
    },
    // NVIDIA Jetson AGX Orin
    GpuSpec {
        name: "Orin", sm_version: 87,
        peak_fp16_tflops: 170.0, peak_fp8_tflops: 170.0, peak_fp32_tflops: 5.3,
        peak_bandwidth_gbs: 204.8, vram_gb: 64.0, l2_cache_mb: 4.0,
        crossover_fp16: 830.0, crossover_fp8: 830.0, crossover_fp32: 25.9,
        base_clock_mhz: 624, kernel_launch_overhead_ns: 8000, sync_overhead_ns: 4000,
        pcie_bandwidth_gbps: 0.0, occupancy_worst_case: 0.4, l2_cache_bytes: 4 * 1024 * 1024,
    },
    // NVIDIA Jetson Orin NX (smaller edge)
    GpuSpec {
        name: "Orin-NX", sm_version: 87,
        peak_fp16_tflops: 100.0, peak_fp8_tflops: 100.0, peak_fp32_tflops: 3.1,
        peak_bandwidth_gbs: 102.4, vram_gb: 16.0, l2_cache_mb: 2.0,
        crossover_fp16: 976.0, crossover_fp8: 976.0, crossover_fp32: 30.3,
        base_clock_mhz: 624, kernel_launch_overhead_ns: 8000, sync_overhead_ns: 4000,
        pcie_bandwidth_gbps: 0.0, occupancy_worst_case: 0.35, l2_cache_bytes: 2 * 1024 * 1024,
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
}
