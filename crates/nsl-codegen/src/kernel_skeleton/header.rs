//! PTX file header emission (.version, .target, .address_size).
//!
//! Variants tested:
//! - (V8_7, Sm75) — FA v2 forward and backward
//! - (V7_0, Sm80) — WRGA fused LoRA and IA³

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtxVersion {
    V7_0,
    V8_7,
}

impl PtxVersion {
    fn as_str(self) -> &'static str {
        match self {
            PtxVersion::V7_0 => "7.0",
            PtxVersion::V8_7 => "8.7",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetSm {
    Sm75,
    Sm80,
}

impl TargetSm {
    fn as_str(self) -> &'static str {
        match self {
            TargetSm::Sm75 => "sm_75",
            TargetSm::Sm80 => "sm_80",
        }
    }
}

/// Emit the three-line PTX file header plus trailing blank line.
///
/// Variants tested: (V8_7, Sm75) for FA; (V7_0, Sm80) for WRGA.
/// Extend by adding a PtxVersion / TargetSm variant AND a new snapshot.
pub fn emit_ptx_header(ptx: &mut String, version: PtxVersion, sm: TargetSm) {
    ptx.push_str(&format!(".version {}\n", version.as_str()));
    ptx.push_str(&format!(".target {}\n", sm.as_str()));
    ptx.push_str(".address_size 64\n\n");
}
