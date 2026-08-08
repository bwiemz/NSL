//! Training-pipeline decision audit emitted by `nsl check --training-report`.
//!
//! Walks the AST for training blocks — `train(...)` **and** `distill(...)` —
//! invokes the FASE and PCA planners for each, and produces a report suitable
//! for text display or JSON serialization.  Pure data computation — no codegen.
//!
//! Task 2 establishes data structures + text formatter.  Task 3 adds
//! the AST walker (`build_report`).
//!
//! A `distill(...)` block is a training loop over the student, and it reaches
//! the planners through the same synthetic `TrainBlock` codegen lowers it
//! through ([`crate::cpkd::distill_as_train_block`]). Before that it matched
//! nothing here, so the audit whose entire job is explaining the accumulation
//! decision answered "Training blocks found: 0" for a `distill(..., epochs = 8,
//! grad_accumulation = 4)` block that was, in fact, planning FASE-Deferred
//! with a window of 4.

use std::collections::HashMap;

use nsl_ast::block::{TrainBlock, TrainSection};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;
use serde::Serialize;

use crate::fase::{self, FaseConfig, FaseOptimizer, FasePlan};
use crate::fase_memory::MemorySchedule;
use crate::pca_activation::{collect_dataset_configs, extract_dataset_ref};
use crate::pca_detect::{DatasetPackingConfig, PcaDetection, PcaDetectConfig};

#[derive(Debug, Clone, Serialize)]
pub struct TrainingReport {
    pub source_path: String,
    pub train_blocks: Vec<TrainBlockReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TrainBlockReport {
    /// Which block spelling produced this entry. Rendered for BOTH kinds:
    /// showing it only for `distill` would read as "train blocks are the
    /// untyped default", and a JSON consumer diffing two reports needs to
    /// know which block moved.
    pub kind: BlockKind,
    /// For a `distill` block this is the STUDENT — the teacher's fields are
    /// frozen `Input` leaves with no adjoints (invariant I-11) and are not in
    /// the block's parameter list at all.
    pub model_name: Option<String>,
    pub fase: FaseSection,
    pub pca: Option<PcaSection>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BlockKind {
    Train,
    Distill,
}

impl BlockKind {
    fn as_str(self) -> &'static str {
        match self {
            BlockKind::Train => "train",
            BlockKind::Distill => "distill",
        }
    }
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

// ─── AST walker ─────────────────────────────────────────────────────────────

/// Walk an AST's top-level items for training blocks — `train(...)` and
/// `distill(...)` — and build a report.
///
/// Pure-data traversal — no codegen, no semantic re-analysis.  Each train
/// block's config is extracted from its AST args and optimizer section, fed
/// to [`fase::plan`] (and optionally [`pca_detect::detect`] when a packing-
/// enabled dataset is referenced), and the results assembled into
/// [`TrainBlockReport`]s.
///
/// Fields that cannot be resolved from the AST (e.g. a non-literal
/// `grad_accumulation` expression) fall back to [`FaseConfig::default`]
/// values so the planner sees the most conservative intent without panicking.
///
/// **Scope note (Task 3):** `model_name` extraction, dataset-block packing
/// config, and `data: source = <ident>` resolution are all implemented.
/// `MemorySchedule` (`FaseSection::memory`) is always `None` because it
/// requires a `ModelFootprint` that cannot be cheaply derived from the AST
/// without running semantic analysis; this is deferred to a later task.
pub fn build_report(ast: &nsl_ast::Module, interner: &Interner, source_path: &std::path::Path) -> TrainingReport {
    // Item 2 step 2: this is a real pass driver — it produces the report BY
    // running the FASE and PCA planners — but it is not the compile pipeline.
    // Naming it is what stops `nsl check --training-report` reporting an
    // unattributed pass.
    let _phase = crate::pass_trace::enter_phase(crate::pass_registry::CompilePhase::Analysis);
    let dataset_configs = collect_dataset_configs(&ast.stmts, interner);
    let mut blocks = Vec::new();
    for stmt in &ast.stmts {
        // Both spellings, and both bare and decorated — a train block often
        // sits inside a test/bench decorator.
        let inner = match &stmt.kind {
            StmtKind::Decorated { stmt: inner, .. } => &inner.kind,
            other => other,
        };
        match inner {
            StmtKind::TrainBlock(train) => {
                blocks.push(build_block_report(
                    BlockKind::Train,
                    train,
                    None,
                    interner,
                    &dataset_configs,
                ));
            }
            StmtKind::DistillBlock(distill) => {
                // A header the presentation refuses cannot compile either (the
                // semantic checker refuses the same spellings with a span), so
                // no successful build reaches here with one. But `build_report`
                // is `pub` and runs no semantic analysis, so a caller that
                // hands it an unchecked AST would otherwise get "Training
                // blocks found: 0" — character for character the defect this
                // arm exists to fix. Say why instead of dropping it silently.
                match crate::cpkd::distill_as_train_block(distill, interner) {
                    Ok(presented) => {
                        let student = presented
                            .student_sym
                            .map(|s| resolve(interner, s).to_string());
                        blocks.push(build_block_report(
                            BlockKind::Distill,
                            &presented.train,
                            student,
                            interner,
                            &dataset_configs,
                        ));
                    }
                    Err(why) => eprintln!(
                        "note: --training-report skipped a distill block whose \
                         header cannot be planned: {why}"
                    ),
                }
            }
            _ => {}
        }
    }
    TrainingReport {
        source_path: source_path.display().to_string(),
        train_blocks: blocks,
    }
}

/// `model_name_override` carries the name a synthetic block cannot express in
/// its config: a distill block trains the STUDENT, and `student =` is not a
/// train config key, so the presentation returns the symbol separately rather
/// than smuggling it in under `model =` (see
/// [`crate::cpkd::distill_as_train_block`]).
fn build_block_report(
    kind: BlockKind,
    train: &TrainBlock,
    model_name_override: Option<String>,
    interner: &Interner,
    datasets: &HashMap<String, DatasetPackingConfig>,
) -> TrainBlockReport {
    let model_name = model_name_override.or_else(|| extract_model_name(train, interner));
    let fase_config = extract_fase_config(train, interner);
    let plan = fase::plan(&fase_config);

    // MemorySchedule requires a ModelFootprint derivable only from semantic
    // analysis — always None at the AST-walk stage.
    let memory = None;

    let pca = extract_dataset_ref(train, interner)
        .and_then(|ds| datasets.get(&ds))
        .filter(|cfg| cfg.enabled)
        .map(|cfg| PcaSection {
            detection: crate::pca_detect::detect(cfg, &PcaDetectConfig::default(), 2),
            packing_config: cfg.clone(),
        });

    TrainBlockReport {
        kind,
        model_name,
        fase: FaseSection { plan, memory },
        pca,
    }
}

/// Resolve a `Symbol` to a `&str` using the interner.
fn resolve(interner: &Interner, sym: nsl_ast::Symbol) -> &str {
    interner.resolve(sym.0).unwrap_or("<unknown>")
}

fn extract_model_name(train: &TrainBlock, interner: &Interner) -> Option<String> {
    for arg in &train.config {
        if let Some(name_sym) = arg.name {
            if resolve(interner, name_sym) == "model" {
                if let ExprKind::Ident(sym) = &arg.value.kind {
                    return Some(resolve(interner, *sym).to_string());
                }
            }
        }
    }
    None
}

fn extract_fase_config(train: &TrainBlock, interner: &Interner) -> FaseConfig {
    let mut accumulation: u32 = 1;
    let mut grad_clip: Option<f64> = None;

    for arg in &train.config {
        if let Some(name_sym) = arg.name {
            match resolve(interner, name_sym) {
                "grad_accumulation" => {
                    if let ExprKind::IntLiteral(n) = &arg.value.kind {
                        accumulation = (*n).max(1) as u32;
                    }
                }
                "grad_clip" => {
                    if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                        grad_clip = Some(*f);
                    } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                        grad_clip = Some(*n as f64);
                    }
                }
                _ => {}
            }
        }
    }

    let optimizer = extract_optimizer(train, interner);

    FaseConfig {
        optimizer,
        accumulation,
        grad_clip,
        allow_v_approx: true,
        ..Default::default()
    }
}

fn extract_optimizer(train: &TrainBlock, interner: &Interner) -> FaseOptimizer {
    for section in &train.sections {
        if let TrainSection::Optimizer(expr) = section {
            if let ExprKind::Call { callee, .. } = &expr.kind {
                if let ExprKind::Ident(sym) = &callee.kind {
                    let name = resolve(interner, *sym);
                    return FaseOptimizer::parse(name);
                }
            }
            // Bare identifier (no args): `optimizer: AdamW`
            if let ExprKind::Ident(sym) = &expr.kind {
                let name = resolve(interner, *sym);
                return FaseOptimizer::parse(name);
            }
        }
    }
    FaseOptimizer::Unknown
}

// ─── Display / formatting ────────────────────────────────────────────────────

impl std::fmt::Display for TrainingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "=== Training Pipeline Report ===")?;
        writeln!(f, "File: {}", self.source_path)?;
        writeln!(f, "Training blocks found: {}", self.train_blocks.len())?;
        writeln!(f)?;

        if self.train_blocks.is_empty() {
            writeln!(f, "No training blocks found in {}.", self.source_path)?;
            return Ok(());
        }

        for (i, block) in self.train_blocks.iter().enumerate() {
            writeln!(f, "[Block {}]", i + 1)?;
            writeln!(f, "  Kind: {}", block.kind.as_str())?;
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
                kind: BlockKind::Train,
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
            text.contains("No training blocks found"),
            "empty report text missing the no-blocks message:\n{}",
            text
        );
    }

    // ── Task 3 tests ─────────────────────────────────────────────────────────

    /// Compile-time signature check.  If `build_report` doesn't have the right
    /// signature this won't compile.
    #[test]
    fn build_report_has_expected_signature() {
        fn _assert_sig() {
            let _f: fn(
                &nsl_ast::Module,
                &nsl_lexer::Interner,
                &std::path::Path,
            ) -> TrainingReport = build_report;
        }
    }

    /// Parse the `fase_deferred_adamw_equivalence.nsl` fixture, call
    /// `build_report`, and verify the FASE planner sees 1 train block in
    /// Deferred mode with accumulation == 4.
    #[test]
    fn build_report_on_parsed_fixture_finds_train_block() {
        let fixture = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/fase_deferred_adamw_equivalence.nsl");

        let source = std::fs::read_to_string(&fixture)
            .unwrap_or_else(|e| panic!("failed to read fixture {:?}: {}", fixture, e));

        let mut interner = nsl_lexer::Interner::new();
        let (tokens, lex_diags) =
            nsl_lexer::tokenize(&source, nsl_errors::FileId(0), &mut interner);
        let lex_errors: Vec<_> = lex_diags
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .collect();
        assert!(lex_errors.is_empty(), "lex errors: {:?}", lex_errors);

        let parsed = nsl_parser::parse(&tokens, &mut interner);
        let parse_errors: Vec<_> = parsed
            .diagnostics
            .iter()
            .filter(|d| matches!(d.level, nsl_errors::Level::Error))
            .collect();
        assert!(parse_errors.is_empty(), "parse errors: {:?}", parse_errors);

        let report = build_report(&parsed.module, &interner, &fixture);

        assert_eq!(
            report.train_blocks.len(),
            1,
            "expected 1 train block, found {}",
            report.train_blocks.len()
        );

        let block = &report.train_blocks[0];

        assert_eq!(
            block.fase.plan.accumulation, 4,
            "expected accumulation == 4, got {}",
            block.fase.plan.accumulation
        );

        assert_eq!(
            block.fase.plan.mode,
            crate::fase::FaseMode::Deferred,
            "expected FaseMode::Deferred, got {:?}",
            block.fase.plan.mode
        );

        // Model name should be extracted as "m" from `train(model = m, ...)`.
        assert_eq!(
            block.model_name.as_deref(),
            Some("m"),
            "expected model_name == Some(\"m\"), got {:?}",
            block.model_name
        );
    }

    /// A source file with no train blocks should yield an empty report.
    #[test]
    fn build_report_no_train_blocks() {
        let source = "let x = 1\nlet y = 2\n";
        let mut interner = nsl_lexer::Interner::new();
        let (tokens, _) = nsl_lexer::tokenize(source, nsl_errors::FileId(0), &mut interner);
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        let path = std::path::Path::new("dummy.nsl");
        let report = build_report(&parsed.module, &interner, path);
        assert!(report.train_blocks.is_empty(), "expected no train blocks");
    }
}
