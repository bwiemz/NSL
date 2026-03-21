//! Format-agnostic sparse tensor layout annotations — `@layout` decorator.
//!
//! Allows users to annotate tensors with per-dimension storage formats:
//!   @layout("CSR")              — predefined format macro
//!   @layout(dense, compressed)  — per-dimension level formats
//!
//! The compiler uses these annotations to auto-generate optimal sparse
//! iteration code via TACO-style concrete index notation lowering.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

// ---------------------------------------------------------------------------
// Level format — per-dimension storage mode
// ---------------------------------------------------------------------------

/// How to traverse one dimension of a sparse tensor.
///
/// These match TACO's level format algebra. The compiler generates
/// different loop structures depending on the level format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LevelFormat {
    /// Iterate 0..N (all positions, including zeros). No index compression.
    Dense,
    /// Compressed arrays: ptr[i] → start, ptr[i+1] → end, with sorted indices.
    /// This is the CSR/CSC format for the compressed dimension.
    Compressed,
    /// Single coordinate per entry (COO-style). Unsorted within parent level.
    Singleton,
    /// Compressed, but allows duplicate indices (used in COO with batched insertions).
    CompressedNonUnique,
    /// Hash-map backed level. O(1) lookup but non-sequential iteration.
    Hashed,
}

// ---------------------------------------------------------------------------
// Layout info — parsed from @layout decorator
// ---------------------------------------------------------------------------

/// Parsed configuration from `@layout("CSR")` or `@layout(dense, compressed)`.
#[derive(Debug, Clone, PartialEq)]
pub struct LayoutInfo {
    /// Per-dimension level formats, in dimension order (outermost first).
    pub levels: Vec<LevelFormat>,
    /// Human-readable name (e.g., "CSR", "COO", or "custom").
    pub name: String,
}

impl LayoutInfo {
    /// Create a layout from a predefined format name.
    pub fn from_format_name(name: &str) -> Option<Self> {
        let levels = match name.to_uppercase().as_str() {
            "CSR" => vec![LevelFormat::Dense, LevelFormat::Compressed],
            "CSC" => vec![LevelFormat::Compressed, LevelFormat::Dense],
            "COO" => vec![LevelFormat::CompressedNonUnique, LevelFormat::Singleton],
            "BSR" | "BCSR" => vec![
                LevelFormat::Dense,
                LevelFormat::Compressed,
                LevelFormat::Dense,
                LevelFormat::Dense,
            ],
            "DENSE" => vec![LevelFormat::Dense, LevelFormat::Dense],
            _ => return None,
        };
        Some(LayoutInfo {
            levels,
            name: name.to_uppercase(),
        })
    }

    /// Check if this layout has any compressed/sparse dimensions.
    pub fn is_sparse(&self) -> bool {
        self.levels.iter().any(|l| !matches!(l, LevelFormat::Dense))
    }

    /// Number of dimensions in this layout.
    pub fn ndim(&self) -> usize {
        self.levels.len()
    }

    /// Map to the M50 format ID (0=COO, 1=CSR, 2=CSC, 3=BSR).
    /// Returns None for non-standard layouts.
    pub fn to_sparse_format_id(&self) -> Option<u8> {
        match self.name.as_str() {
            "COO" => Some(0),
            "CSR" => Some(1),
            "CSC" => Some(2),
            "BSR" | "BCSR" => Some(3),
            _ => None,
        }
    }

    /// Preferred format for matmul when this is the left operand.
    /// CSR is preferred for SpMM (row-major iteration); CSC for column-major.
    pub fn preferred_matmul_format(&self) -> u8 {
        match self.name.as_str() {
            "CSC" => 2,
            "BSR" | "BCSR" => 3,
            _ => 1, // CSR is the default for matmul
        }
    }
}

// ---------------------------------------------------------------------------
// @layout decorator validator
// ---------------------------------------------------------------------------

/// Parse a level format from a string identifier.
fn parse_level_format(s: &str) -> Option<LevelFormat> {
    match s.to_lowercase().as_str() {
        "dense" | "d" => Some(LevelFormat::Dense),
        "compressed" | "c" => Some(LevelFormat::Compressed),
        "singleton" | "s" => Some(LevelFormat::Singleton),
        "compressed_nonunique" | "cu" => Some(LevelFormat::CompressedNonUnique),
        "hashed" | "h" => Some(LevelFormat::Hashed),
        _ => None,
    }
}

/// Validate `@layout` decorator arguments.
///
/// Two forms:
///   1. `@layout("CSR")` — predefined format macro (string argument)
///   2. `@layout(dense, compressed)` — per-dimension level formats (positional string args)
///
/// Returns `Some(LayoutInfo)` on success, `None` if validation fails.
pub fn validate_layout_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<LayoutInfo> {
    let args = match &deco.args {
        Some(a) if !a.is_empty() => a,
        _ => {
            diagnostics.push(
                Diagnostic::error("@layout: at least one argument required".to_string())
                    .with_label(deco.span, "missing layout specification"),
            );
            return None;
        }
    };

    // Check for named "format" argument first (backward compat)
    for arg in args {
        if let Some(name_sym) = arg.name {
            let aname = resolve_sym(name_sym);
            if aname == "format" {
                if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                    return match LayoutInfo::from_format_name(s) {
                        Some(info) => Some(info),
                        None => {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@layout: unknown format \"{}\". Valid: CSR, CSC, COO, BSR, DENSE",
                                    s
                                ))
                                .with_label(arg.span, "unknown format"),
                            );
                            None
                        }
                    };
                }
            }
        }
    }

    // Form 1: single string argument → predefined format macro
    if args.len() == 1 {
        if let ExprKind::StringLiteral(ref s) = args[0].value.kind {
            return match LayoutInfo::from_format_name(s) {
                Some(info) => Some(info),
                None => {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "@layout: unknown format \"{}\". Valid: CSR, CSC, COO, BSR, DENSE",
                            s
                        ))
                        .with_label(args[0].span, "unknown format"),
                    );
                    None
                }
            };
        }
    }

    // Form 2: per-dimension level format identifiers
    // e.g., @layout(dense, compressed) or @layout("dense", "compressed")
    let mut levels = Vec::with_capacity(args.len());
    for arg in args {
        let level_str = match &arg.value.kind {
            ExprKind::StringLiteral(s) => s.clone(),
            ExprKind::Ident(sym) => resolve_sym(*sym),
            _ => {
                diagnostics.push(
                    Diagnostic::error(
                        "@layout: each argument must be a level format (dense, compressed, singleton)"
                            .to_string(),
                    )
                    .with_label(arg.span, "expected level format"),
                );
                return None;
            }
        };

        match parse_level_format(&level_str) {
            Some(lf) => levels.push(lf),
            None => {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "@layout: unknown level format \"{}\". Valid: dense, compressed, singleton, hashed",
                        level_str
                    ))
                    .with_label(arg.span, "unknown level format"),
                );
                return None;
            }
        }
    }

    if levels.is_empty() {
        diagnostics.push(
            Diagnostic::error("@layout: no level formats specified".to_string())
                .with_label(deco.span, "empty layout"),
        );
        return None;
    }

    // Determine name from level pattern
    let name = match levels.as_slice() {
        [LevelFormat::Dense, LevelFormat::Compressed] => "CSR".to_string(),
        [LevelFormat::Compressed, LevelFormat::Dense] => "CSC".to_string(),
        [LevelFormat::CompressedNonUnique, LevelFormat::Singleton] => "COO".to_string(),
        [LevelFormat::Dense, LevelFormat::Compressed, LevelFormat::Dense, LevelFormat::Dense] => {
            "BSR".to_string()
        }
        [LevelFormat::Dense, LevelFormat::Dense] => "DENSE".to_string(),
        _ => "CUSTOM".to_string(),
    };

    Some(LayoutInfo { levels, name })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::decl::Decorator;
    use nsl_ast::expr::{Arg, Expr, ExprKind};
    use nsl_ast::Span;
    use nsl_errors::{BytePos, FileId};

    const ZERO_SPAN: Span = Span {
        file_id: FileId(0),
        start: BytePos(0),
        end: BytePos(0),
    };

    fn make_sym(n: u32) -> Symbol {
        Symbol(unsafe { std::mem::transmute::<u32, string_interner::DefaultSymbol>(n) })
    }

    fn make_string_arg(name_id: Option<u32>, value: &str) -> Arg {
        Arg {
            name: name_id.map(make_sym),
            value: Expr {
                kind: ExprKind::StringLiteral(value.to_string()),
                span: ZERO_SPAN,
                id: nsl_ast::NodeId(0),
            },
            span: ZERO_SPAN,
        }
    }

    fn resolver(sym: Symbol) -> String {
        match unsafe { std::mem::transmute::<string_interner::DefaultSymbol, u32>(sym.0) } {
            10 => "format".to_string(),
            20 => "dense".to_string(),
            21 => "compressed".to_string(),
            22 => "singleton".to_string(),
            _ => "unknown".to_string(),
        }
    }

    // --- Predefined format macros ---

    #[test]
    fn layout_csr() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(None, "CSR")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_layout_decorator(&deco, &resolver, &mut diags);
        assert!(diags.is_empty(), "diags: {:?}", diags);
        let info = result.unwrap();
        assert_eq!(info.name, "CSR");
        assert_eq!(info.levels, vec![LevelFormat::Dense, LevelFormat::Compressed]);
        assert!(info.is_sparse());
        assert_eq!(info.to_sparse_format_id(), Some(1));
    }

    #[test]
    fn layout_csc() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(None, "CSC")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "CSC");
        assert_eq!(info.levels, vec![LevelFormat::Compressed, LevelFormat::Dense]);
        assert_eq!(info.to_sparse_format_id(), Some(2));
    }

    #[test]
    fn layout_coo() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(None, "COO")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "COO");
        assert_eq!(
            info.levels,
            vec![LevelFormat::CompressedNonUnique, LevelFormat::Singleton]
        );
    }

    #[test]
    fn layout_bsr() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(None, "BSR")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "BSR");
        assert_eq!(info.ndim(), 4);
    }

    #[test]
    fn layout_case_insensitive() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(None, "csr")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "CSR");
    }

    #[test]
    fn layout_unknown_format_error() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(None, "ELLPACK")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_layout_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_none());
        assert!(!diags.is_empty());
    }

    #[test]
    fn layout_no_args_error() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_layout_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_none());
    }

    #[test]
    fn layout_none_args_error() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: None,
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_layout_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_none());
    }

    // --- Per-dimension level formats ---

    #[test]
    fn layout_per_dimension_csr() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![
                make_string_arg(None, "dense"),
                make_string_arg(None, "compressed"),
            ]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "CSR");
        assert_eq!(info.levels, vec![LevelFormat::Dense, LevelFormat::Compressed]);
    }

    #[test]
    fn layout_per_dimension_custom() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![
                make_string_arg(None, "compressed"),
                make_string_arg(None, "compressed"),
            ]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "CUSTOM");
        assert!(info.is_sparse());
    }

    #[test]
    fn layout_per_dimension_dense() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![
                make_string_arg(None, "dense"),
                make_string_arg(None, "dense"),
            ]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "DENSE");
        assert!(!info.is_sparse());
    }

    #[test]
    fn layout_unknown_level_error() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![
                make_string_arg(None, "dense"),
                make_string_arg(None, "radix"),
            ]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_layout_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_none());
        assert!(!diags.is_empty());
    }

    // --- LayoutInfo methods ---

    #[test]
    fn layout_preferred_matmul() {
        let csr = LayoutInfo::from_format_name("CSR").unwrap();
        assert_eq!(csr.preferred_matmul_format(), 1);

        let csc = LayoutInfo::from_format_name("CSC").unwrap();
        assert_eq!(csc.preferred_matmul_format(), 2);

        let bsr = LayoutInfo::from_format_name("BSR").unwrap();
        assert_eq!(bsr.preferred_matmul_format(), 3);
    }

    #[test]
    fn layout_named_format_arg() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(Some(10), "CSR")]), // format="CSR"
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let info = validate_layout_decorator(&deco, &resolver, &mut diags).unwrap();
        assert_eq!(info.name, "CSR");
    }
}
