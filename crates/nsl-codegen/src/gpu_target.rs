// crates/nsl-codegen/src/gpu_target.rs
//! M47: GPU target selection and feature capability detection.

/// GPU compilation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    Cuda,
    Rocm,
    Metal,
    WebGpu,
}

impl GpuTarget {
    /// Parse from CLI string.
    pub fn parse_target(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cuda" => Some(GpuTarget::Cuda),
            "rocm" | "amd" | "hip" => Some(GpuTarget::Rocm),
            "metal" | "apple" | "mps" => Some(GpuTarget::Metal),
            "webgpu" | "wgsl" => Some(GpuTarget::WebGpu),
            _ => None,
        }
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
        }
    }

    /// Default warp/wavefront/SIMD width.
    pub fn warp_size(&self) -> u32 {
        match self {
            GpuTarget::Cuda => 32,
            GpuTarget::Rocm => 64,
            GpuTarget::Metal => 32,
            GpuTarget::WebGpu => 0, // no subgroup guarantees
        }
    }
}

/// Feature flags for GPU capabilities.
///
/// Used to validate that a kernel's required features are supported by the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureSet(u32);

impl FeatureSet {
    pub const NONE: Self = FeatureSet(0);
    pub const SHARED_MEMORY: Self = FeatureSet(0x01);
    pub const WARP_SHUFFLE: Self = FeatureSet(0x02);
    pub const TENSOR_CORES: Self = FeatureSet(0x04);
    pub const ATOMIC_FLOAT: Self = FeatureSet(0x08);
    pub const SUBGROUP_OPS: Self = FeatureSet(0x10);
    pub const F16_ARITHMETIC: Self = FeatureSet(0x20);
    pub const BF16_ARITHMETIC: Self = FeatureSet(0x40);

    pub fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Features required by kernel but not supported by target.
    pub fn missing(self, required: Self) -> Self {
        FeatureSet(required.0 & !self.0)
    }

    /// Human-readable list of feature names.
    pub fn names(self) -> Vec<&'static str> {
        let mut names = Vec::new();
        if self.0 & 0x01 != 0 {
            names.push("shared_memory");
        }
        if self.0 & 0x02 != 0 {
            names.push("warp_shuffle");
        }
        if self.0 & 0x04 != 0 {
            names.push("tensor_cores");
        }
        if self.0 & 0x08 != 0 {
            names.push("atomic_float");
        }
        if self.0 & 0x10 != 0 {
            names.push("subgroup_ops");
        }
        if self.0 & 0x20 != 0 {
            names.push("f16_arithmetic");
        }
        if self.0 & 0x40 != 0 {
            names.push("bf16_arithmetic");
        }
        names
    }
}

impl std::ops::BitOr for FeatureSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        FeatureSet(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for FeatureSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for FeatureSet {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        FeatureSet(self.0 & rhs.0)
    }
}

impl std::ops::Not for FeatureSet {
    type Output = Self;
    fn not(self) -> Self {
        FeatureSet(!self.0)
    }
}

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
