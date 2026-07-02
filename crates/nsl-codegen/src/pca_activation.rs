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

/// CFTP §4.3 G2 Strategy 3 (Item 4): resolve the first train block's
/// referenced `DatasetPackingConfig`. Returns `None` when no train block
/// is present, when its dataset reference cannot be resolved, or when
/// the resolved dataset has `enabled = false`. Mirrors the resolution
/// path used by [`detect_packing_for_stmts`] but returns the concrete
/// config so the planner site can pass it to `pca_per_doc::admit`.
pub fn resolve_packing_config_for_stmts(
    stmts: &[Stmt],
    interner: &Interner,
) -> Option<DatasetPackingConfig> {
    let datasets = collect_dataset_configs(stmts, interner);
    for stmt in stmts {
        let train_opt = match &stmt.kind {
            StmtKind::TrainBlock(train) => Some(train),
            StmtKind::Decorated { stmt: inner, .. } => {
                if let StmtKind::TrainBlock(train) = &inner.kind {
                    Some(train)
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(train) = train_opt {
            if let Some(ds_name) = extract_dataset_ref(train, interner) {
                if let Some(cfg) = datasets.get(&ds_name) {
                    if cfg.enabled {
                        return Some(cfg.clone());
                    }
                }
            }
        }
    }
    None
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
            // v9 M4: `mean_doc_length` and `doc_length_stddev` accept both
            // Int and Float literals. Calibration tools that measure these
            // stats on real corpora naturally emit floats (e.g. 384.7);
            // forcing users to hand-round before writing them in the
            // dataset body was a papercut.
            //
            // Semantic-time non-negative validation (v9 L4) refuses
            // negatives up-front, so `n < 0.0` here is unreachable in
            // normal flow. The `.max(0.0)` guard is defensive against a
            // possible test-only path that bypasses semantic — a negative
            // float rounded down would otherwise wrap to a huge u32 and
            // silently misroute PCA planning.
            "mean_doc_length" => {
                if let Some(v) = int_or_float_as_u32(&entry.value.kind) {
                    mean_doc_length = Some(v);
                }
            }
            "doc_length_stddev" => {
                if let Some(v) = int_or_float_as_u32(&entry.value.kind) {
                    doc_length_stddev = Some(v);
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

/// v9 M4: coerce an Int or Float literal into a `u32` for PCA length stats.
///
/// Behavior (uniform across Int and Float; review-driven consistency fix):
///
/// * Negative inputs saturate to `0`.
/// * Values above `u32::MAX` saturate to `u32::MAX`.
/// * `FloatLiteral` with NaN / ±Infinity returns `None` (the writer chose
///   a non-representable value; the caller drops the field so PCA planning
///   sees "not present" rather than a silently coerced garbage max).
/// * Everything else (BoolLiteral, StringLiteral, …) returns `None`
///   (matches the pre-v9 no-op fallthrough).
///
/// Rounding uses `f64::round` (half-away-from-zero) because calibration
/// tools generally quote real-corpus averages to 1–2 decimals (`384.7`)
/// and users expect `385`, not `384`.
///
/// The explicit `clamp` on both paths is load-bearing. Without it the Int
/// path went through `i64 as u32` which TRUNCATES modulo 2^32 — so
/// `mean_doc_length = 5_000_000_000` silently became `705_032_704`, a
/// materially misleading small-mid-range value that would misroute PCA
/// planning without any diagnostic. The Float path used the implicit
/// saturating cast (`as u32` on f64 saturates since Rust 1.45), so the
/// two paths disagreed on the same input class. Both now saturate.
fn int_or_float_as_u32(kind: &ExprKind) -> Option<u32> {
    match kind {
        ExprKind::IntLiteral(n) => Some((*n).clamp(0, u32::MAX as i64) as u32),
        ExprKind::FloatLiteral(f) if f.is_finite() => {
            Some(f.clamp(0.0, u32::MAX as f64).round() as u32)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v9_m4_int_literal_rounds_trivially() {
        assert_eq!(int_or_float_as_u32(&ExprKind::IntLiteral(384)), Some(384));
        assert_eq!(int_or_float_as_u32(&ExprKind::IntLiteral(0)), Some(0));
    }

    #[test]
    fn v9_m4_float_literal_rounds_nearest() {
        // 384.7 → 385  (nearest-even, .7 rounds up)
        assert_eq!(int_or_float_as_u32(&ExprKind::FloatLiteral(384.7)), Some(385));
        // 384.4 → 384
        assert_eq!(int_or_float_as_u32(&ExprKind::FloatLiteral(384.4)), Some(384));
        // 384.5 → 385  (Rust's f64::round is half-away-from-zero, not banker's)
        assert_eq!(int_or_float_as_u32(&ExprKind::FloatLiteral(384.5)), Some(385));
    }

    #[test]
    fn v9_m4_negative_int_saturates_to_zero() {
        // Defensive fallback: semantic L4 refuses negatives at parse time,
        // but the codegen path still saturates to 0 rather than wrapping
        // to a huge u32. Pins the invariant against a future refactor
        // that drops the semantic check.
        assert_eq!(int_or_float_as_u32(&ExprKind::IntLiteral(-100)), Some(0));
    }

    #[test]
    fn v9_m4_negative_float_saturates_to_zero() {
        assert_eq!(int_or_float_as_u32(&ExprKind::FloatLiteral(-42.5)), Some(0));
    }

    #[test]
    fn v9_m4_nan_and_infinity_return_none() {
        // Non-finite values are refused rather than silently coerced to
        // u32::MAX. The caller drops the field (Some/None pattern), so
        // PCA planning sees "not present" instead of a garbage max value.
        assert_eq!(int_or_float_as_u32(&ExprKind::FloatLiteral(f64::NAN)), None);
        assert_eq!(int_or_float_as_u32(&ExprKind::FloatLiteral(f64::INFINITY)), None);
        assert_eq!(int_or_float_as_u32(&ExprKind::FloatLiteral(f64::NEG_INFINITY)), None);
    }

    #[test]
    fn v9_m4_non_numeric_variants_return_none() {
        // BoolLiteral, StringLiteral etc. fall through — matches the
        // pre-v9 behavior where the whole match arm was a no-op fallthrough.
        assert_eq!(int_or_float_as_u32(&ExprKind::BoolLiteral(true)), None);
    }

    // ─── Review-driven overflow-consistency pins ────────────────────────

    #[test]
    fn v9_m4_int_larger_than_u32_max_saturates_to_u32_max() {
        // Pre-fix: `i64 as u32` TRUNCATED via modulo, so 5_000_000_000
        // silently became 705_032_704 — a materially misleading value
        // that would misroute PCA planning without diagnostic. This
        // test pins the explicit-clamp fix so a future regression
        // (e.g. dropping `.clamp(...)` back to `.max(0)`) trips here.
        let five_billion: i64 = 5_000_000_000;
        assert_eq!(
            int_or_float_as_u32(&ExprKind::IntLiteral(five_billion)),
            Some(u32::MAX),
            "int > u32::MAX must saturate to u32::MAX (not truncate)"
        );
        // Exactly u32::MAX still round-trips cleanly.
        assert_eq!(
            int_or_float_as_u32(&ExprKind::IntLiteral(u32::MAX as i64)),
            Some(u32::MAX)
        );
        // i64::MAX also saturates rather than wrapping.
        assert_eq!(
            int_or_float_as_u32(&ExprKind::IntLiteral(i64::MAX)),
            Some(u32::MAX)
        );
    }

    #[test]
    fn v9_m4_float_larger_than_u32_max_saturates_to_u32_max() {
        // Rust's `f64 as u32` already saturates since 1.45, so the
        // implicit path would have worked; the explicit clamp makes
        // intent visible and pins the Int/Float parity. `1e20` is a
        // finite value well above `u32::MAX ≈ 4.29e9`.
        assert_eq!(
            int_or_float_as_u32(&ExprKind::FloatLiteral(1e20)),
            Some(u32::MAX),
            "finite float > u32::MAX must saturate to u32::MAX"
        );
    }

    #[test]
    fn v9_m4_int_and_float_paths_agree_on_overflow() {
        // Consistency guarantee: for a large input class that fits in
        // both variants, the two paths must produce the same u32.
        let large_int = 4_500_000_000i64;
        let large_float = 4_500_000_000f64;
        assert_eq!(
            int_or_float_as_u32(&ExprKind::IntLiteral(large_int)),
            int_or_float_as_u32(&ExprKind::FloatLiteral(large_float)),
            "Int and Float paths must agree on the same numeric value"
        );
    }
}
