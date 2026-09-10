// crates/nsl-kir/src/feature_set.rs
//! GPU capability flags: what a kernel requires and what a target offers.
//! Moved here from `nsl_codegen::gpu_target` with the rest of the IR
//! (roadmap A2 step 1); `nsl_codegen::gpu_target::FeatureSet` re-exports it.

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
    /// Asynchronous global → shared copies with commit/wait groups
    /// (`cp.async` on sm_80+). Roadmap A2 step 2.
    pub const ASYNC_COPY: Self = FeatureSet(0x80);

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
