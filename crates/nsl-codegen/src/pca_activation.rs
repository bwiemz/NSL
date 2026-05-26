//! Single source of truth for "is this module's training data packed?".
//!
//! Used by BOTH the training-report path and the kernel-synthesis path so
//! the synthesis `segment_masked` flag and the runtime registry setter can
//! never compute divergent packing decisions (spec §2.4).
//!
//! Institutional note (2026-05-17 §5 correction): the FA-2 entry points were
//! claimed to null-check segment_ids_ptr/doc_starts_ptr; they did NOT — the
//! masked kernel dereferenced null. The guards were added in this Tier A
//! activation (forward/backward seg_ids preludes + emit_doc_starts_smem_load)
//! and are verified by the identity-equals-unmasked GPU tests
//! (masked_null_seg_ptr_equals_unmasked_{forward,backward}: masked+null ==
//! unmasked, max_abs 0.0).

use std::collections::HashMap;

use nsl_ast::block::{DatasetDef, TrainBlock, TrainSection};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_lexer::Interner;

use crate::pca_detect::{self, DatasetPackingConfig, PcaDetectConfig, PcaStrategy};

/// Returns true iff the module contains a train block whose referenced
/// dataset selects a packing strategy. Conservative: any resolution
/// failure → false.
pub fn detect_packing_for_stmts(stmts: &[Stmt], interner: &Interner) -> bool {
    let datasets = collect_dataset_configs(stmts, interner);
    for stmt in stmts {
        match &stmt.kind {
            StmtKind::TrainBlock(train)
                if check_train_packs(train, &datasets, interner) =>
            {
                return true;
            }
            StmtKind::TrainBlock(_) => {}
            StmtKind::Decorated { stmt: inner, .. } => {
                if let StmtKind::TrainBlock(train) = &inner.kind {
                    if check_train_packs(train, &datasets, interner) {
                        return true;
                    }
                }
            }
            _ => {}
        }
    }
    false
}

fn check_train_packs(
    train: &TrainBlock,
    datasets: &std::collections::HashMap<String, DatasetPackingConfig>,
    interner: &Interner,
) -> bool {
    if let Some(ds_name) = extract_dataset_ref(train, interner) {
        if let Some(cfg) = datasets.get(&ds_name) {
            let det = pca_detect::detect(cfg, &PcaDetectConfig::default(), 2);
            return det.strategy != PcaStrategy::NoPacking;
        }
    }
    false
}

/// Walk top-level items for `dataset <Name>(...)` blocks and build a
/// packing-config map keyed by dataset name.
pub(crate) fn collect_dataset_configs(
    stmts: &[Stmt],
    interner: &Interner,
) -> HashMap<String, DatasetPackingConfig> {
    let mut map = HashMap::new();
    for stmt in stmts {
        if let StmtKind::DatasetDef(def) = &stmt.kind {
            let name = resolve(interner, def.name).to_string();
            let cfg = extract_dataset_packing(def, interner);
            map.insert(name, cfg);
        }
    }
    map
}

/// Look inside `data:` sections for an assignment like `source = <ident>`.
///
/// In the AST a data section is `TrainSection::Data(Vec<Stmt>)`. The parser
/// renders `source = ds_name` as a `StmtKind::Assign` with a plain-ident
/// target and plain-ident value, or alternatively as a `StmtKind::VarDecl`.
pub(crate) fn extract_dataset_ref(train: &TrainBlock, interner: &Interner) -> Option<String> {
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

/// Extract packing configuration from a `dataset` definition block.
pub(crate) fn extract_dataset_packing(
    def: &DatasetDef,
    interner: &Interner,
) -> DatasetPackingConfig {
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

/// Resolve a `Symbol` to a `&str` using the interner.
fn resolve(interner: &Interner, sym: nsl_ast::Symbol) -> &str {
    interner.resolve(sym.0).unwrap_or("<unknown>")
}
