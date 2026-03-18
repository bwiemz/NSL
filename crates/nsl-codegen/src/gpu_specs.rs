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
    },
    GpuSpec {
        name: "A100-PCIe", sm_version: 80,
        peak_fp16_tflops: 312.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 19.5,
        peak_bandwidth_gbs: 1555.0, vram_gb: 40.0, l2_cache_mb: 40.0,
        crossover_fp16: 200.6, crossover_fp8: 0.0, crossover_fp32: 12.5,
    },
    GpuSpec {
        name: "H100-SXM", sm_version: 90,
        peak_fp16_tflops: 989.0, peak_fp8_tflops: 1979.0, peak_fp32_tflops: 67.0,
        peak_bandwidth_gbs: 3350.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 295.2, crossover_fp8: 590.7, crossover_fp32: 20.0,
    },
    GpuSpec {
        name: "H100-PCIe", sm_version: 90,
        peak_fp16_tflops: 756.0, peak_fp8_tflops: 1513.0, peak_fp32_tflops: 51.0,
        peak_bandwidth_gbs: 2039.0, vram_gb: 80.0, l2_cache_mb: 50.0,
        crossover_fp16: 370.8, crossover_fp8: 741.9, crossover_fp32: 25.0,
    },
    GpuSpec {
        name: "RTX-4090", sm_version: 89,
        peak_fp16_tflops: 330.0, peak_fp8_tflops: 661.0, peak_fp32_tflops: 82.6,
        peak_bandwidth_gbs: 1008.0, vram_gb: 24.0, l2_cache_mb: 72.0,
        crossover_fp16: 327.4, crossover_fp8: 655.8, crossover_fp32: 81.9,
    },
    GpuSpec {
        name: "RTX-3090", sm_version: 86,
        peak_fp16_tflops: 142.0, peak_fp8_tflops: 0.0, peak_fp32_tflops: 35.6,
        peak_bandwidth_gbs: 936.2, vram_gb: 24.0, l2_cache_mb: 6.0,
        crossover_fp16: 151.7, crossover_fp8: 0.0, crossover_fp32: 38.0,
    },
    GpuSpec {
        name: "L40S", sm_version: 89,
        peak_fp16_tflops: 362.0, peak_fp8_tflops: 733.0, peak_fp32_tflops: 91.6,
        peak_bandwidth_gbs: 864.0, vram_gb: 48.0, l2_cache_mb: 96.0,
        crossover_fp16: 419.0, crossover_fp8: 848.4, crossover_fp32: 106.0,
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
        assert_eq!(GPU_DATABASE.len(), 7);
        let names: Vec<&str> = GPU_DATABASE.iter().map(|g| g.name).collect();
        assert!(names.contains(&"A100-SXM"));
        assert!(names.contains(&"H100-SXM"));
        assert!(names.contains(&"RTX-4090"));
        assert!(names.contains(&"RTX-3090"));
        assert!(names.contains(&"L40S"));
    }
}
