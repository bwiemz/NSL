// crates/nsl-codegen/src/gpu_target.rs
//! M47: GPU target selection and feature capability detection.
//!
//! CUDA is the only GPU backend. The ROCm/AMDGPU, Metal and WebGPU/WGSL
//! printers and the FPGA/Verilog backend were removed in the Phase 0.6 scope
//! freeze; the code is preserved at tag [`REMOVED_BACKENDS_ATTIC_TAG`].

/// The tag that preserves the removed ROCm/AMDGPU, Metal, WebGPU/WGSL and
/// FPGA/Verilog backends. Named in every refusal of those targets.
pub const REMOVED_BACKENDS_ATTIC_TAG: &str = "attic/scope-freeze-2026-10";

/// GPU compilation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    Cuda,
}

impl GpuTarget {
    /// Parse from CLI string.
    pub fn parse_target(s: &str) -> Option<Self> {
        let lower = s.to_lowercase();
        match lower.as_str() {
            "cuda" => Some(GpuTarget::Cuda),

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

    /// Parse from a target string, defaulting to Cuda when empty or
    /// unrecognized.
    ///
    /// The fallback serves library callers that build `CompileOptions` by
    /// hand. It is also why the CLI checks `--target` with
    /// [`validate_cli_target`] first: without that check a misspelt or
    /// removed target (`rocm`, `metal`, `webgpu`, `fpga`) would compile CUDA
    /// kernels without any error.
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
        }
    }

    /// Default warp/wavefront/SIMD width.
    pub fn warp_size(&self) -> u32 {
        match self {
            GpuTarget::Cuda => 32,
        }
    }
}

/// Names the removed backends answered to (every alias `parse_target` used
/// to accept), with the backend each one selected. A `--target` naming one
/// is refused with the reason, not reported as an unknown name.
const REMOVED_BACKEND_TARGETS: &[(&str, &str)] = &[
    ("rocm", "ROCm/AMDGPU"),
    ("amd", "ROCm/AMDGPU"),
    ("hip", "ROCm/AMDGPU"),
    ("metal", "Metal"),
    ("apple", "Metal"),
    ("mps", "Metal"),
    ("webgpu", "WebGPU/WGSL"),
    ("wgsl", "WebGPU/WGSL"),
    ("fpga", "FPGA/Verilog"),
];

/// `true` for `<prefix><digits>` with at least one digit and nothing after.
fn is_sm_spelling(s: &str, prefix: &str) -> bool {
    s.strip_prefix(prefix)
        .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
}

/// Check a user-supplied `--target` value (`nsl build` / `nsl run`).
///
/// [`GpuTarget::from_target_string`] maps every unrecognised string to CUDA,
/// so the CLI checks the value here before the compiler sees it. Accepted
/// spellings, all lowercase:
///
/// - `cuda`: the default. Fused-attention PTX is generated for sm_80.
/// - `sm_<N>` (e.g. `sm_120`): CUDA pinned to an SM. This is the form the
///   fused-attention tables and the packed-attention planner read.
/// - `sm<N>` / `cuda_sm<N>`: CUDA, with the SM read by
///   [`GpuTarget::parse_sm_version`] (the WRGA fused-adapter gate). The
///   fused SDPA tables are not generated for these spellings.
/// - `cpu`: a host-only compile. The CUDA-only fused SDPA variant tables
///   are not embedded.
///
/// `parse_gpu_sm_from_target` (`compiler/kernel.rs`) parses only `cuda` and
/// `sm_<N>`, and it panics on any other spelling. So an `@flash_attention`
/// model compiled with `sm<N>`, `cuda_sm<N>` or `cpu` fails there.
/// That defect predates this check, and this check does not fix it.
///
/// `<N>` must be all digits. A suffix such as `sm_90a` is refused, because
/// the fused-attention path parses `sm_<N>` strictly and would panic on it.
pub fn validate_cli_target(s: &str) -> Result<(), String> {
    const ACCEPTED: &str = "`cuda`, `sm_<N>` (e.g. `sm_120`; also `sm<N>` and \
                            `cuda_sm<N>`) or `cpu`";
    if matches!(s, "cuda" | "cpu")
        || is_sm_spelling(s, "sm_")
        || is_sm_spelling(s, "sm")
        || is_sm_spelling(s, "cuda_sm")
    {
        return Ok(());
    }
    let lower = s.to_lowercase();
    if let Some((_, backend)) = REMOVED_BACKEND_TARGETS.iter().find(|(n, _)| *n == lower) {
        return Err(format!(
            "the {backend} backend was removed in the Phase 0.6 scope freeze; the code is \
             preserved at tag `{REMOVED_BACKENDS_ATTIC_TAG}`. CUDA is the only GPU backend; \
             accepted targets: {ACCEPTED}"
        ));
    }
    if lower != s && validate_cli_target(&lower).is_ok() {
        return Err(format!("target names are lowercase: use `{lower}`"));
    }
    Err(format!(
        "unknown target; expected {ACCEPTED}. The ROCm/AMDGPU, Metal, WebGPU/WGSL and \
         FPGA/Verilog backends were removed (preserved at tag \
         `{REMOVED_BACKENDS_ATTIC_TAG}`)"
    ))
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
        assert_eq!(GpuTarget::parse_target("CUDA"), Some(GpuTarget::Cuda));
        assert_eq!(GpuTarget::parse_target("sm_120"), Some(GpuTarget::Cuda));
        assert_eq!(GpuTarget::parse_target("cuda_sm80"), Some(GpuTarget::Cuda));
        assert_eq!(GpuTarget::parse_target("vulkan"), None);
        // The removed backends no longer parse to a target of their own.
        for removed in ["rocm", "hip", "metal", "mps", "webgpu", "wgsl", "fpga", "FPGA"] {
            assert_eq!(GpuTarget::parse_target(removed), None, "{removed}");
        }
    }

    #[test]
    fn from_target_string_defaults_to_cuda() {
        assert_eq!(GpuTarget::from_target_string(""), GpuTarget::Cuda);
        assert_eq!(GpuTarget::from_target_string("unknown"), GpuTarget::Cuda);
    }

    #[test]
    fn cli_target_accepts_the_cuda_spellings_and_cpu() {
        for ok in ["cuda", "sm_120", "sm_89", "sm_75", "sm80", "cuda_sm80", "cuda_sm70", "cpu"] {
            assert_eq!(validate_cli_target(ok), Ok(()), "{ok} must be accepted");
        }
    }

    #[test]
    fn cli_target_refuses_a_removed_backend_naming_the_attic_tag() {
        for (name, backend) in REMOVED_BACKEND_TARGETS {
            for spelled in [name.to_string(), name.to_uppercase()] {
                let err = validate_cli_target(&spelled)
                    .expect_err("a removed backend must be refused, not mapped to CUDA");
                assert!(err.contains(backend), "{spelled}: {err}");
                assert!(err.contains("removed"), "{spelled}: {err}");
                assert!(err.contains(REMOVED_BACKENDS_ATTIC_TAG), "{spelled}: {err}");
            }
        }
    }

    #[test]
    fn cli_target_refuses_unknown_and_malformed_spellings() {
        // Every one of these used to compile as CUDA without an error,
        // through `from_target_string`'s fallback or `parse_target`'s loose
        // `sm` prefix match.
        for bad in ["", "vulkan", "h100", "sm", "sm_", "sm_90a", "smx", "cuda_sm", "cuda:sm_90", " cuda"] {
            let err = validate_cli_target(bad).expect_err(bad);
            assert!(err.contains("expected"), "{bad:?}: {err}");
            assert!(err.contains(REMOVED_BACKENDS_ATTIC_TAG), "{bad:?}: {err}");
        }
    }

    #[test]
    fn cli_target_refuses_uppercase_with_the_lowercase_spelling() {
        // `CUDA` parses as CUDA, but the fused-attention tables compare the
        // raw string with `cuda`. Accepting it would compile without them.
        let err = validate_cli_target("CUDA").expect_err("CUDA");
        assert!(err.contains("`cuda`"), "{err}");
        let err = validate_cli_target("SM_120").expect_err("SM_120");
        assert!(err.contains("`sm_120`"), "{err}");
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
    fn feature_missing_detection() {
        let target = FeatureSet::SHARED_MEMORY | FeatureSet::F16_ARITHMETIC;
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
