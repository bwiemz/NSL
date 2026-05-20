//! Training-pipeline decision audit emitted by `nsl check --training-report`.
//!
//! Walks the AST for `train(...)` blocks, invokes the FASE and PCA
//! planners for each, and produces a report suitable for text display
//! or JSON serialization.  Pure data computation — no codegen.
//!
//! Task 2 establishes data structures + text formatter.  Task 3 adds
//! the AST walker (`build_report`).

use std::collections::HashMap;

use nsl_ast::block::{DatasetDef, TrainBlock, TrainSection};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::StmtKind;
use nsl_lexer::Interner;
use serde::Serialize;

use crate::fase::{self, FaseConfig, FaseOptimizer, FasePlan};
use crate::fase_memory::MemorySchedule;
use crate::pca_detect::{DatasetPackingConfig, PcaDetectConfig, PcaDetection};

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

// ─── AST walker ─────────────────────────────────────────────────────────────

/// Walk an AST's top-level items for `train(...)` blocks; build a report.
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
pub fn build_report(
    ast: &nsl_ast::Module,
    interner: &Interner,
    source_path: &std::path::Path,
) -> TrainingReport {
    let dataset_configs = collect_dataset_configs(ast, interner);
    let mut blocks = Vec::new();
    for stmt in &ast.stmts {
        match &stmt.kind {
            StmtKind::TrainBlock(train) => {
                blocks.push(build_block_report(train, interner, &dataset_configs));
            }
            StmtKind::Decorated { stmt: inner, .. } => {
                if let StmtKind::TrainBlock(train) = &inner.kind {
                    blocks.push(build_block_report(train, interner, &dataset_configs));
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

fn build_block_report(
    train: &TrainBlock,
    interner: &Interner,
    datasets: &HashMap<String, DatasetPackingConfig>,
) -> TrainBlockReport {
    let model_name = extract_model_name(train, interner);
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

/// Look inside `data:` sections for an assignment like `source = <ident>`.
///
/// In the AST a data section is `TrainSection::Data(Vec<Stmt>)`.  The parser
/// renders `source = ds_name` as a `StmtKind::Assign` with a plain-ident
/// target and plain-ident value, or alternatively as a `StmtKind::VarDecl`.
fn extract_dataset_ref(train: &TrainBlock, interner: &Interner) -> Option<String> {
    for section in &train.sections {
        if let TrainSection::Data(stmts) = section {
            for stmt in stmts {
                // Pattern: `source = <ident>` → StmtKind::Assign
                if let StmtKind::Assign { target, value, .. } = &stmt.kind {
                    if let ExprKind::Ident(target_sym) = &target.kind {
                        if resolve(interner, *target_sym) == "source" {
                            if let ExprKind::Ident(val_sym) = &value.kind {
                                return Some(resolve(interner, *val_sym).to_string());
                            }
                        }
                    }
                }
                // Pattern: `let source = <ident>` → StmtKind::VarDecl
                if let StmtKind::VarDecl {
                    pattern,
                    value: Some(val),
                    ..
                } = &stmt.kind
                {
                    if let nsl_ast::pattern::PatternKind::Ident(name) = &pattern.kind {
                        if resolve(interner, *name) == "source" {
                            if let ExprKind::Ident(val_sym) = &val.kind {
                                return Some(resolve(interner, *val_sym).to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

/// Walk top-level items for `dataset <Name>(...)` blocks and build a
/// packing-config map keyed by dataset name.
///
/// **Scope note:** Only the `packing`, `max_sequence_length`,
/// `mean_doc_length`, `doc_length_stddev`, and `separator_token_id`
/// keys inside the dataset body are extracted.  Unknown keys are silently
/// ignored for forward-compatibility.
fn collect_dataset_configs(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> HashMap<String, DatasetPackingConfig> {
    let mut map = HashMap::new();
    for stmt in &ast.stmts {
        if let StmtKind::DatasetDef(def) = &stmt.kind {
            let name = resolve(interner, def.name).to_string();
            let cfg = extract_dataset_packing(def, interner);
            map.insert(name, cfg);
        }
    }
    map
}

fn extract_dataset_packing(def: &DatasetDef, interner: &Interner) -> DatasetPackingConfig {
    let mut enabled = false;
    let mut max_sequence_length: u32 = 0;
    let mut mean_doc_length: Option<u32> = None;
    let mut doc_length_stddev: Option<u32> = None;
    let mut separator_token_id: Option<i64> = None;

    for entry in &def.body {
        let key = resolve(interner, entry.key);
        match key {
            "packing" => {
                if let ExprKind::BoolLiteral(b) = &entry.value.kind {
                    enabled = *b;
                }
            }
            "max_sequence_length" | "max_seq_len" => {
                if let ExprKind::IntLiteral(n) = &entry.value.kind {
                    max_sequence_length = (*n).max(0) as u32;
                }
            }
            "mean_doc_length" => {
                if let ExprKind::IntLiteral(n) = &entry.value.kind {
                    mean_doc_length = Some((*n).max(0) as u32);
                }
            }
            "doc_length_stddev" => {
                if let ExprKind::IntLiteral(n) = &entry.value.kind {
                    doc_length_stddev = Some((*n).max(0) as u32);
                }
            }
            "separator_token_id" => {
                if let ExprKind::IntLiteral(n) = &entry.value.kind {
                    separator_token_id = Some(*n);
                }
            }
            _ => {}
        }
    }

    DatasetPackingConfig {
        enabled,
        max_sequence_length,
        mean_doc_length,
        doc_length_stddev,
        separator_token_id,
    }
}

// ─── Display / formatting ────────────────────────────────────────────────────

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

    // ── Task 3 tests ─────────────────────────────────────────────────────────

    /// Compile-time signature check.  If `build_report` doesn't have the right
    /// signature this won't compile.
    #[test]
    fn build_report_has_expected_signature() {
        fn _assert_sig() {
            let _f: fn(&nsl_ast::Module, &nsl_lexer::Interner, &std::path::Path) -> TrainingReport =
                build_report;
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
