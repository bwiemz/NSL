//! Training-pipeline decision audit emitted by `nsl check --training-report`.
//!
//! Walks the AST for `train(...)` blocks, invokes the FASE and PCA
//! planners for each, and produces a report suitable for text display
//! or JSON serialization.  Pure data computation — no codegen.
//!
//! Task 2 establishes data structures + text formatter.  Task 3 adds
//! the AST walker (`build_report`).

use serde::Serialize;

use crate::fase::{self, FasePlan};
use crate::fase_memory::MemorySchedule;
use crate::pca_detect::{DatasetPackingConfig, PcaDetection};

#[derive(Debug, Clone, Serialize)]
pub struct TrainingReport {
    pub source_path: String,
    pub train_blocks: Vec<TrainBlockReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainBlockReport {
    pub model_name: Option<String>,
    pub fase: FaseSection,
    pub pca: Option<PcaSection>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FaseSection {
    pub plan: FasePlan,
    pub memory: Option<MemorySchedule>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PcaSection {
    pub detection: PcaDetection,
    pub packing_config: DatasetPackingConfig,
}

impl std::fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Training Pipeline Report ===")?;
        writeln!(f, "File: {}", self.source_path)?;
        writeln!(f, "Train blocks found: {}", self.train_blocks.len())?;
        writeln!(f)?;

        if self.train_blocks.is_empty() {
            writeln!(f, "No train blocks found in {}.", self.source_path)?;
            return Ok(());
        }

        for (i, block) in self.train_blocks.iter().enumerate() {
            writeln!(f, "[Block {}]", i + 1)?;
            if let Some(name) = &block.model_name {
                writeln!(f, "  Model: {}", name)?;
            }
            writeln!(f)?;
            render_fase_section(f, &block.fase)?;
            if let Some(pca) = &block.pca {
                writeln!(f)?;
                render_pca_section(f, pca)?;
            }
            if i + 1 < self.train_blocks.len() {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

fn render_fase_section(f: &mut std::fmt::Formatter<'_>, sec: &FaseSection) -> std::fmt::Result {
    writeln!(f, "  FASE (Fused Accumulation-Step Elimination):")?;
    writeln!(f, "    grad_accumulation: {}", sec.plan.accumulation)?;
    writeln!(f, "    optimizer:         {:?}", sec.plan.recipe.optimizer)?;
    writeln!(f, "    mode:              {:?}", sec.plan.mode)?;
    writeln!(f, "    rationale:         {}", sec.plan.rationale)?;
    writeln!(
        f,
        "    backward_phases:   {}",
        format_phases(&sec.plan.backward_phases)
    )?;
    writeln!(f, "    two_phase_clip:    {}", sec.plan.two_phase_clip)?;
    if let Some(mem) = &sec.memory {
        writeln!(f, "    memory:")?;
        writeln!(
            f,
            "      standard peak: {} bytes",
            format_thousands(mem.standard.peak)
        )?;
        writeln!(
            f,
            "      FASE peak:     {} bytes",
            format_thousands(mem.fase.peak)
        )?;
        let savings = mem.standard.peak.saturating_sub(mem.fase.peak);
        let pct = if mem.standard.peak > 0 {
            (savings as f64 / mem.standard.peak as f64) * 100.0
        } else {
            0.0
        };
        writeln!(
            f,
            "      savings:       {} bytes ({:.1}%)",
            format_thousands(savings),
            pct
        )?;
    }
    Ok(())
}

fn render_pca_section(f: &mut std::fmt::Formatter<'_>, sec: &PcaSection) -> std::fmt::Result {
    writeln!(f, "  PCA (Packed Causal Attention):")?;
    if !sec.packing_config.enabled {
        writeln!(f, "    packing: disabled")?;
        return Ok(());
    }
    writeln!(f, "    packing:              enabled")?;
    writeln!(f, "    strategy:             {:?}", sec.detection.strategy)?;
    writeln!(
        f,
        "    max_sequence_length:  {}",
        sec.packing_config.max_sequence_length
    )?;
    if let Some(m) = sec.packing_config.mean_doc_length {
        writeln!(f, "    mean_doc_length:      {}", m)?;
    }
    if let Some(s) = sec.packing_config.doc_length_stddev {
        writeln!(f, "    doc_length_stddev:    {}", s)?;
    }
    Ok(())
}

fn format_phases(phases: &[fase::BackwardPhase]) -> String {
    if phases.is_empty() {
        return "(none)".to_string();
    }
    let mut runs: Vec<(String, usize)> = Vec::new();
    for p in phases {
        let label = format!("{:?}", p);
        if let Some(last) = runs.last_mut() {
            if last.0 == label {
                last.1 += 1;
                continue;
            }
        }
        runs.push((label, 1));
    }
    runs.into_iter()
        .map(|(label, n)| {
            if n == 1 {
                label
            } else {
                format!("{} x {}", label, n)
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_thousands(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out.chars().rev().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::{FaseConfig, FaseOptimizer};

    fn sample_report_deferred_adamw() -> TrainingReport {
        let cfg = FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            grad_clip: None,
            allow_v_approx: true,
            ..Default::default()
        };
        let plan = fase::plan(&cfg);
        TrainingReport {
            source_path: "pretrain.nsl".to_string(),
            train_blocks: vec![TrainBlockReport {
                model_name: Some("NSLCoder".to_string()),
                fase: FaseSection { plan, memory: None },
                pca: None,
            }],
        }
    }

    #[test]
    fn text_formatter_mentions_fase_mode_and_accumulation() {
        let r = sample_report_deferred_adamw();
        let text = format!("{}", r);
        assert!(
            text.contains("FASE"),
            "report text missing FASE section:\n{}",
            text
        );
        assert!(
            text.contains("Deferred"),
            "report text missing mode: Deferred:\n{}",
            text
        );
        assert!(
            text.contains("4") && text.contains("accumulation"),
            "report text missing grad_accumulation: 4:\n{}",
            text
        );
    }

    #[test]
    fn json_serialization_round_trips_plan_mode() {
        let r = sample_report_deferred_adamw();
        let json = serde_json::to_string(&r).expect("serialize");
        assert!(
            json.contains("\"Deferred\""),
            "json missing mode:\n{}",
            json
        );
        assert!(
            json.contains("\"AdamW\""),
            "json missing optimizer:\n{}",
            json
        );
    }

    #[test]
    fn no_train_blocks_is_a_valid_empty_report() {
        let r = TrainingReport {
            source_path: "model_only.nsl".to_string(),
            train_blocks: vec![],
        };
        let text = format!("{}", r);
        assert!(
            text.contains("No train blocks found"),
            "empty report text missing the no-blocks message:\n{}",
            text
        );
    }
}
