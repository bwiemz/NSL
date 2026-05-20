//! Compile-time shape-propagation trace for `nsl check --shapes`.
//!
//! Read-only formatter over the semantic analyzer's output. Walks the typed
//! AST and prints one line per let-bound expression (or top-level expr) that
//! produces a `Tensor<...>`, with a green check on clean nodes and a red X
//! when any diagnostic's primary label overlaps the node's span.

use std::fmt::Write;

use nsl_ast::{
    decl::FnDef,
    expr::{Expr, ExprKind},
    stmt::{Block, Stmt, StmtKind},
    Module, Symbol,
};
use nsl_errors::{Diagnostic, Level, Span};
use nsl_lexer::Interner;
use nsl_semantic::{
    checker::TypeMap,
    types::{display_type, Dim, Shape, Type},
    AnalysisResult,
};

/// Bundle of everything needed to render a shape trace.
pub struct ShapeDebugInput {
    pub source: String,
    pub file_name: String,
    pub module: Module,
    pub analysis: AnalysisResult,
    pub interner: Interner,
    pub parse_diagnostics: Vec<Diagnostic>,
}

impl ShapeDebugInput {
    /// Run lex → parse → semantic-analyze on `src` and bundle the results.
    /// The file name is used only for diagnostic rendering.
    pub fn from_source(src: &str, _file: &str) -> Result<Self, String> {
        let mut interner = Interner::new();
        // We don't have a SourceMap here since we don't emit diagnostics; use
        // a fresh FileId(0) for the lexer.
        let file_id = nsl_errors::FileId(0);
        let (tokens, lex_errors) = nsl_lexer::tokenize(src, file_id, &mut interner);
        // Parse
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        let mut parse_diagnostics = lex_errors;
        parse_diagnostics.extend(parse_result.diagnostics.clone());
        // Semantic analysis (no imports, no linear types)
        let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
        Ok(Self {
            source: src.to_string(),
            file_name: _file.to_string(),
            module: parse_result.module,
            analysis,
            interner,
            parse_diagnostics,
        })
    }
}

/// Render the trace. See module-level docs for the format.
pub fn format_trace(input: &ShapeDebugInput) -> String {
    let mut out = String::new();
    writeln!(out, "=== Shape Debugger ===").unwrap();

    render_signature(&mut out, &input.module, &input.interner, &input.source);
    writeln!(out, "\nPropagation:").unwrap();

    let mut total_flops: u64 = 0;
    let mut error_count: usize = 0;

    for stmt in &input.module.stmts {
        walk_stmt(stmt, input, &mut out, &mut total_flops, &mut error_count);
    }

    // After the line-by-line trace, emit a detailed Expected/Cause/Fix block
    // for each error diagnostic.
    let shape_errors: Vec<&Diagnostic> = input
        .analysis
        .diagnostics
        .iter()
        .chain(input.parse_diagnostics.iter())
        .filter(|d| d.level == Level::Error)
        .collect();

    for d in &shape_errors {
        render_error_block(&mut out, d, &input.source);
    }

    if shape_errors.is_empty() && error_count == 0 {
        writeln!(out, "\nAll shapes valid. No mismatches detected.").unwrap();
    } else {
        writeln!(
            out,
            "\n{} mismatch(es) detected.",
            shape_errors.len().max(error_count)
        )
        .unwrap();
    }
    writeln!(
        out,
        "Total FLOPs: {:.2} GFLOP per forward pass.",
        total_flops as f64 / 1e9
    )
    .unwrap();
    out
}

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

fn render_signature(out: &mut String, m: &Module, interner: &Interner, source: &str) {
    for stmt in &m.stmts {
        if let StmtKind::FnDef(f) = &stmt.kind {
            let name = resolve(interner, f.name);
            let params: Vec<String> = f
                .params
                .iter()
                .map(|p| {
                    let ty_src = p
                        .type_ann
                        .as_ref()
                        .and_then(|t| snippet(source, t_span(t)))
                        .unwrap_or_else(|| "?".to_string());
                    format!("{}: {}", resolve(interner, p.name), ty_src)
                })
                .collect();
            writeln!(out, "Input: fn {}({})", name, params.join(", ")).unwrap();
            return;
        }
    }
    writeln!(out, "Input: <no top-level fn>").unwrap();
}

fn walk_stmt(
    stmt: &Stmt,
    input: &ShapeDebugInput,
    out: &mut String,
    total_flops: &mut u64,
    error_count: &mut usize,
) {
    match &stmt.kind {
        StmtKind::FnDef(f) => walk_block(&f.body, input, out, total_flops, error_count),
        StmtKind::Decorated { stmt: inner, .. } => {
            walk_stmt(inner, input, out, total_flops, error_count)
        }
        StmtKind::VarDecl {
            value: Some(expr), ..
        } => {
            render_expr_line(expr, stmt.span, input, out, total_flops, error_count);
        }
        StmtKind::Return(Some(expr)) | StmtKind::Expr(expr) => {
            render_expr_line(expr, stmt.span, input, out, total_flops, error_count);
        }
        _ => {}
    }
}

fn walk_block(
    block: &Block,
    input: &ShapeDebugInput,
    out: &mut String,
    total_flops: &mut u64,
    error_count: &mut usize,
) {
    for s in &block.stmts {
        walk_stmt(s, input, out, total_flops, error_count);
    }
}

fn render_expr_line(
    expr: &Expr,
    stmt_span: Span,
    input: &ShapeDebugInput,
    out: &mut String,
    total_flops: &mut u64,
    error_count: &mut usize,
) {
    let ty = input.analysis.type_map.get(&expr.id);
    // Skip non-tensor expressions — we only want shape-carrying lines.
    let is_tensor = ty
        .map(|t| t.is_tensor() || matches!(t, Type::Unknown))
        .unwrap_or(false);
    if !is_tensor {
        return;
    }
    let snippet_str = snippet(&input.source, expr.span)
        .unwrap_or_else(|| "<expr>".to_string())
        .trim()
        .replace('\n', " ");
    let shape_str = ty
        .map(render_shape_only)
        .unwrap_or_else(|| "<unknown>".to_string());

    let has_err = diagnostic_overlaps(&input.analysis.diagnostics, stmt_span)
        || diagnostic_overlaps(&input.parse_diagnostics, stmt_span);
    let mark = if has_err {
        *error_count += 1;
        "\u{274C}"
    } else {
        "\u{2705}"
    };

    writeln!(
        out,
        "  {:<40} \u{2192} {:<20} {}",
        truncate(&snippet_str, 40),
        truncate(&shape_str, 20),
        mark
    )
    .unwrap();

    if let Some(t) = ty {
        *total_flops += estimate_flops_for_call(expr, t, &input.analysis.type_map, &input.interner);
    }
}

fn render_shape_only(ty: &Type) -> String {
    if let Some((shape, _dt, _dev)) = ty.as_tensor_parts() {
        format_shape_brief(shape)
    } else {
        display_type(ty)
    }
}

fn format_shape_brief(shape: &Shape) -> String {
    let parts: Vec<String> = shape
        .dims
        .iter()
        .map(|d| match d {
            Dim::Concrete(n) => n.to_string(),
            Dim::Symbolic(_) => "?".to_string(),
            Dim::Named { size, .. } => match &**size {
                Dim::Concrete(n) => n.to_string(),
                _ => "?".to_string(),
            },
            Dim::Bounded { upper_bound, .. } => format!("<={}", upper_bound),
            Dim::Computed(_) => "?".to_string(),
            Dim::Wildcard => "*".to_string(),
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        let cut: String = s.chars().take(n.saturating_sub(1)).collect();
        format!("{cut}…")
    }
}

fn diagnostic_overlaps(diags: &[Diagnostic], span: Span) -> bool {
    diags.iter().filter(|d| d.level == Level::Error).any(|d| {
        d.labels.iter().any(|l| {
            l.span.file_id == span.file_id
                && l.span.start.0 < span.end.0
                && l.span.end.0 > span.start.0
        })
    })
}

fn render_error_block(out: &mut String, d: &Diagnostic, source: &str) {
    writeln!(out, "\n  \u{274C} MISMATCH").unwrap();
    // Heuristic decomposition of the diagnostic into Expected / Cause / Fix /
    // Source. The `nsl-errors::Diagnostic` type doesn't have dedicated fields
    // for these, so we fall back to rendering the message verbatim under
    // "Cause" and the first label's snippet under "Source".
    let (expected, cause) = split_message(&d.message);
    writeln!(out, "    Expected: {}", expected).unwrap();
    writeln!(out, "    Cause:    {}", cause).unwrap();
    let fix = d
        .notes
        .first()
        .cloned()
        .unwrap_or_else(|| "(no suggested fix)".to_string());
    writeln!(out, "    Fix:      {}", fix).unwrap();
    let src_line = d
        .labels
        .first()
        .and_then(|l| snippet(source, l.span))
        .map(|s| s.trim().replace('\n', " "))
        .unwrap_or_else(|| "<no source>".to_string());
    writeln!(out, "    Source:   {}", src_line).unwrap();
}

fn split_message(msg: &str) -> (String, String) {
    // If the message looks like "expected X, found Y" pull X out as Expected
    // and leave the rest as Cause. Otherwise use the whole message as Cause.
    if let Some(rest) = msg.strip_prefix("expected ") {
        if let Some(idx) = rest.find(", found ") {
            let expected = rest[..idx].to_string();
            let found = &rest[idx + ", found ".len()..];
            return (expected, format!("found {found}"));
        }
    }
    ("(see message)".to_string(), msg.to_string())
}

fn snippet(src: &str, span: Span) -> Option<String> {
    let s = span.start.0 as usize;
    let e = span.end.0 as usize;
    if e <= s || e > src.len() {
        return None;
    }
    Some(src[s..e].to_string())
}

fn t_span(t: &nsl_ast::types::TypeExpr) -> Span {
    t.span
}

fn resolve(interner: &Interner, s: Symbol) -> String {
    interner
        .resolve(s.0)
        .map(|x| x.to_string())
        .unwrap_or_else(|| "<sym>".to_string())
}

// ---------------------------------------------------------------------------
// FLOP estimation
// ---------------------------------------------------------------------------

fn estimate_flops_for_call(
    expr: &Expr,
    out_ty: &Type,
    type_map: &TypeMap,
    interner: &Interner,
) -> u64 {
    let ExprKind::Call { callee, args } = &expr.kind else {
        return 0;
    };
    let ExprKind::Ident(sym) = &callee.kind else {
        return 0;
    };
    let name = resolve(interner, *sym);
    let Some((out_shape, _dt, _dev)) = out_ty.as_tensor_parts() else {
        return 0;
    };
    let dtype_bytes: u64 = 2; // bf16/fp16 assumption; close enough for display

    // Resolve concrete sizes on the output shape.
    let out_dims: Vec<u64> = out_shape.dims.iter().map(dim_to_u64).collect();

    match name.as_str() {
        "matmul" => {
            // matmul(A, B): A: [..., M, K], B: [..., K, N], out: [..., M, N]
            // Get K from first arg's last dim.
            let k = args
                .first()
                .and_then(|a| type_map.get(&a.value.id))
                .and_then(|t| t.as_tensor_parts())
                .and_then(|(s, _, _)| s.dims.last().cloned())
                .map(|d| dim_to_u64(&d))
                .unwrap_or(0);
            if out_dims.len() >= 2 && k > 0 {
                let n = *out_dims.last().unwrap();
                let m = out_dims[out_dims.len() - 2];
                let batch: u64 = out_dims[..out_dims.len() - 2].iter().product();
                let batch = batch.max(1);
                let (f, _, _) =
                    nsl_codegen::cost_model::batched_matmul_cost(batch, m, k, n, dtype_bytes);
                return f;
            }
            0
        }
        "layernorm" | "rmsnorm" => {
            // [B, S, D] or [B, D]
            if out_dims.len() == 3 {
                let (b, s, d) = (out_dims[0], out_dims[1], out_dims[2]);
                let (f, _, _) = if name == "layernorm" {
                    nsl_codegen::cost_model::layernorm_cost(b, s, d, dtype_bytes)
                } else {
                    nsl_codegen::cost_model::rmsnorm_cost(b, s, d, dtype_bytes)
                };
                return f;
            }
            0
        }
        "softmax" => {
            let total: u64 = out_dims.iter().product::<u64>().max(1);
            let last = *out_dims.last().unwrap_or(&1);
            let b = total / last.max(1);
            let (f, _, _) = nsl_codegen::cost_model::softmax_cost(b, last, dtype_bytes);
            f
        }
        "flash_attn" | "flash_attention" => {
            if out_dims.len() == 4 {
                let (b, h, s, d) = (out_dims[0], out_dims[1], out_dims[2], out_dims[3]);
                let (f, _, _) =
                    nsl_codegen::cost_model::flash_attention_cost(b, h, s, d, dtype_bytes);
                return f;
            }
            0
        }
        "embedding" => {
            if out_dims.len() == 3 {
                let (b, s, d) = (out_dims[0], out_dims[1], out_dims[2]);
                let (f, _, _) = nsl_codegen::cost_model::embedding_cost(b, s, d, dtype_bytes);
                return f;
            }
            0
        }
        _ => 0,
    }
}

fn dim_to_u64(d: &Dim) -> u64 {
    match d {
        Dim::Concrete(n) => (*n).max(0) as u64,
        Dim::Named { size, .. } => dim_to_u64(size),
        Dim::Bounded { upper_bound, .. } => (*upper_bound).max(0) as u64,
        _ => 0,
    }
}

// Unused FnDef import placeholder — the type is surfaced for downstream use.
#[allow(dead_code)]
fn _touch(_: &FnDef) {}
