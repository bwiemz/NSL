//! Profiling report types shared across NSL dev-tools (cost model, fusion,
//! memory planner, CLI reports).

use crate::cost_model::OpCost;
use crate::wggo::WggoPlan;
use crate::wrga_fusion::FusionPlan;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EntryKind {
    Auto,
    Train,
    Function(String),
}

impl EntryKind {
    /// Parse an `--entry` CLI flag: `"auto"`, `"train"`, or `"fn:<name>"`.
    pub fn parse_flag(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(EntryKind::Auto),
            "train" => Some(EntryKind::Train),
            other => {
                if let Some(name) = other.strip_prefix("fn:") {
                    if name.is_empty() {
                        None
                    } else {
                        Some(EntryKind::Function(name.to_string()))
                    }
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warn,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    pub code: String,
    pub message: String,
    pub severity: Severity,
}

impl Recommendation {
    pub fn memory_bound_batch_hint(op: &str) -> Self {
        Recommendation {
            code: "R01".to_string(),
            message: format!(
                "Op `{op}` is memory-bound; consider increasing batch size or fusing with neighbors."
            ),
            severity: Severity::Info,
        }
    }

    pub fn dominating_op(op: &str, pct: f64) -> Self {
        Recommendation {
            code: "R02".to_string(),
            message: format!("Op `{op}` dominates runtime ({pct:.1}% of total)."),
            severity: Severity::Info,
        }
    }

    pub fn fusion_strongly_recommended(kernels_saved: usize) -> Self {
        Recommendation {
            code: "R03".to_string(),
            message: format!(
                "Fusion strongly recommended: ~{kernels_saved} kernel launches saved."
            ),
            severity: Severity::Warn,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTimelineEntry {
    pub program_point: u32,
    pub live_bytes: u64,
    pub phase: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileReport {
    pub target_gpu: String,
    pub dtype: String,
    pub entry: EntryKind,
    pub ops: Vec<OpCost>,
    pub total_flops: u64,
    pub total_hbm_bytes: u64,
    pub total_estimated_us: f64,
    pub fusion: Option<FusionPlan>,
    pub memory_timeline: Option<Vec<MemoryTimelineEntry>>,
    pub recommendations: Vec<Recommendation>,
    /// Populated by `nsl profile --explain-wggo`. `WggoPlan` is serialize-only,
    /// so this field is skipped on deserialization and defaults to `None`.
    #[serde(default, skip_deserializing)]
    pub wggo_explain: Option<WggoPlan>,
}
