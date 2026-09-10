// crates/nsl-codegen/src/gpu_target.rs
//! M47: GPU target selection and feature capability detection.

/// GPU compilation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    Cuda,
    Rocm,
    Metal,
    WebGpu,
    Fpga,  // M57 v1: FPGA Verilog backend
}

impl GpuTarget {
    /// Parse from CLI string.
    pub fn parse_target(s: &str) -> Option<Self> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "cuda" => Some(GpuTarget::Cuda),
            "rocm" | "amd" | "hip" => Some(GpuTarget::Rocm),
            "metal" | "apple" | "mps" => Some(GpuTarget::Metal),
            "webgpu" | "wgsl" => Some(GpuTarget::WebGpu),
            "fpga" => Some(GpuTarget::Fpga),  // M57.1 §3.2

            // WRGA B.3 Task 4: accept `cuda_sm<N>` / `sm<N>` variants.
            s if s.starts_with("cuda_sm") || s.starts_with("sm") => Some(GpuTarget::Cuda),
            _ => None,
        }
    }

    /// WRGA B.3 Task 4: extract a CUDA sm version from the raw target string,
    /// if present.  Accepts `cuda_sm80`, `sm_80`, `sm80`.  Returns `None`
    /// when the target is non-CUDA or carries no sm qualifier.
    pub fn parse_sm_version(s: &str) -> Option<u32> {
        let lower = s.to_lowercase();
        let tail = if let Some(rest) = lower.strip_prefix("cuda_sm") {
            rest
        } else if let Some(rest) = lower.strip_prefix("sm_") {
            rest
        } else { lower.strip_prefix("sm")? };
        // Trim any trailing non-digit (e.g. `sm_80a`).
        let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    }

    /// Parse from CLI string, defaulting to Cuda when empty or unrecognized.
    pub fn from_target_string(s: &str) -> Self {
        if s.is_empty() {
            return GpuTarget::Cuda;
        }
        Self::parse_target(s).unwrap_or(GpuTarget::Cuda)
    }

    /// Display name for error messages.
    pub fn name(&self) -> &'static str {
        match self {
            GpuTarget::Cuda => "cuda",
            GpuTarget::Rocm => "rocm",
            GpuTarget::Metal => "metal",
            GpuTarget::WebGpu => "webgpu",
            GpuTarget::Fpga => "fpga",
        }
    }

    /// Supported features for this target.
    pub fn features(&self) -> FeatureSet {
        match self {
            GpuTarget::Cuda => {
                FeatureSet::SHARED_MEMORY
                    | FeatureSet::WARP_SHUFFLE
                    | FeatureSet::TENSOR_CORES
                    | FeatureSet::ATOMIC_FLOAT
                    | FeatureSet::F16_ARITHMETIC
                    | FeatureSet::BF16_ARITHMETIC
                    | FeatureSet::ASYNC_COPY
            }
            GpuTarget::Rocm => {
                FeatureSet::SHARED_MEMORY
                    | FeatureSet::WARP_SHUFFLE
                    | FeatureSet::TENSOR_CORES
                    | FeatureSet::ATOMIC_FLOAT
                    | FeatureSet::F16_ARITHMETIC
                    | FeatureSet::BF16_ARITHMETIC
            }
            GpuTarget::Metal => {
                FeatureSet::SHARED_MEMORY
                    | FeatureSet::WARP_SHUFFLE
                    | FeatureSet::ATOMIC_FLOAT
                    | FeatureSet::F16_ARITHMETIC
            }
            GpuTarget::WebGpu => FeatureSet::SHARED_MEMORY | FeatureSet::F16_ARITHMETIC,
            GpuTarget::Fpga => {
                // FPGA target uses the HIR pipeline (kernel_ir → hir::lower →
                // backend_verilog), not the SIMT FeatureSet model — return NONE
                // because no SIMT feature flags apply to dataflow hardware.
                FeatureSet::NONE
            }
        }
    }

    /// Default warp/wavefront/SIMD width.
    pub fn warp_size(&self) -> u32 {
        match self {
            GpuTarget::Cuda => 32,
            GpuTarget::Rocm => 64,
            GpuTarget::Metal => 32,
            GpuTarget::WebGpu => 0, // no subgroup guarantees
            GpuTarget::Fpga => 1, // dataflow — no warp concept; 1 is a benign sentinel
        }
    }
}

/// `FeatureSet` lives in `nsl-kir` (the leaf crate that owns `KernelIR`,
/// its verifier and the PTX printer; roadmap A2 step 1) and is re-exported
/// here at its historical path.
pub use nsl_kir::FeatureSet;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_parse() {
        assert_eq!(GpuTarget::parse_target("cuda"), Some(GpuTarget::Cuda));
        assert_eq!(GpuTarget::parse_target("ROCM"), Some(GpuTarget::Rocm));
        assert_eq!(GpuTarget::parse_target("hip"), Some(GpuTarget::Rocm));
        assert_eq!(GpuTarget::parse_target("metal"), Some(GpuTarget::Metal));
        assert_eq!(GpuTarget::parse_target("mps"), Some(GpuTarget::Metal));
        assert_eq!(GpuTarget::parse_target("webgpu"), Some(GpuTarget::WebGpu));
        assert_eq!(GpuTarget::parse_target("vulkan"), None);
    }

    #[test]
    fn parse_target_recognizes_fpga() {
        // M57.1 §3.2: parse_target returns GpuTarget::Fpga for "fpga"/"FPGA".
        // This activates the previously-runtime-unreachable compiler/kernel.rs:170 arm.
        assert_eq!(GpuTarget::parse_target("fpga"), Some(GpuTarget::Fpga));
        assert_eq!(GpuTarget::parse_target("FPGA"), Some(GpuTarget::Fpga));
    }

    #[test]
    fn from_target_string_defaults_to_cuda() {
        assert_eq!(GpuTarget::from_target_string(""), GpuTarget::Cuda);
        assert_eq!(GpuTarget::from_target_string("unknown"), GpuTarget::Cuda);
        assert_eq!(GpuTarget::from_target_string("rocm"), GpuTarget::Rocm);
        assert_eq!(GpuTarget::from_target_string("metal"), GpuTarget::Metal);
        assert_eq!(GpuTarget::from_target_string("webgpu"), GpuTarget::WebGpu);
    }

    #[test]
    fn cuda_has_all_features() {
        let f = GpuTarget::Cuda.features();
        assert!(f.contains(FeatureSet::SHARED_MEMORY));
        assert!(f.contains(FeatureSet::WARP_SHUFFLE));
        assert!(f.contains(FeatureSet::TENSOR_CORES));
        assert!(f.contains(FeatureSet::BF16_ARITHMETIC));
    }

    #[test]
    fn webgpu_lacks_shuffle_and_tensor_cores() {
        let f = GpuTarget::WebGpu.features();
        assert!(f.contains(FeatureSet::SHARED_MEMORY));
        assert!(!f.contains(FeatureSet::WARP_SHUFFLE));
        assert!(!f.contains(FeatureSet::TENSOR_CORES));
    }

    #[test]
    fn feature_missing_detection() {
        let target = GpuTarget::WebGpu.features();
        let required = FeatureSet::SHARED_MEMORY | FeatureSet::TENSOR_CORES;
        let missing = target.missing(required);
        assert!(missing.contains(FeatureSet::TENSOR_CORES));
        assert!(!missing.contains(FeatureSet::SHARED_MEMORY));
    }

    #[test]
    fn feature_names() {
        let f = FeatureSet::WARP_SHUFFLE | FeatureSet::TENSOR_CORES;
        let names = f.names();
        assert!(names.contains(&"warp_shuffle"));
        assert!(names.contains(&"tensor_cores"));
        assert_eq!(names.len(), 2);
    }
}
