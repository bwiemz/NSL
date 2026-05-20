//! Task 8 of M35.1: parser + semantic integration for the BitNet ternary dtypes.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md` §1.1 + §2.
//!
//! These tests verify two ends of the lexer-parser-semantic pipeline:
//!   1. `ternary` and `ternary_unpacked` lex as bare identifiers (NSL does not
//!      promote dtype names to lexer keywords; they are resolved at semantic
//!      time, mirroring `bf16` / `fp16` / `f32`).
//!   2. A `Tensor<[...], ternary>` parameter annotation parses without error
//!      AND resolves to `DType::TernaryPacked` (similarly for `ternary_unpacked`
//!      → `DType::TernaryUnpacked`) inside the semantic type map.

use nsl_ast::stmt::StmtKind;
use nsl_ast::types::TypeExprKind;
use nsl_errors::FileId;
use nsl_lexer::token::TokenKind;
use nsl_lexer::Interner;
use nsl_semantic::types::{DType, Type};

/// Tokenize a source snippet and return the resulting tokens + diagnostics.
fn lex(src: &str) -> (Vec<nsl_lexer::token::Token>, Vec<nsl_errors::Diagnostic>) {
    let mut interner = Interner::new();
    nsl_lexer::tokenize(src, FileId(0), &mut interner)
}

/// Tokenize + look up the identifier text of any `Ident` tokens.
/// Returns `(token, resolved_name)` pairs for non-trivia tokens.
fn lex_with_names(src: &str) -> Vec<(TokenKind, String)> {
    let mut interner = Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    tokens
        .into_iter()
        .filter_map(|t| match t.kind {
            TokenKind::Newline | TokenKind::Eof => None,
            TokenKind::Ident(sym) => {
                // `TokenKind::Ident` wraps a raw `string_interner::DefaultSymbol`,
                // so we pass it directly — no `.0` extraction.
                let name = interner.resolve(sym).unwrap_or("").to_string();
                Some((TokenKind::Ident(sym), name))
            }
            other => Some((other, String::new())),
        })
        .collect()
}

#[test]
fn ternary_keyword_lexes_as_identifier() {
    // Dtype names in NSL are identifiers, not lexer keywords. Confirm `ternary`
    // tokenizes cleanly as a single Ident — no Error tokens, no diagnostics.
    let (tokens, diags) = lex("ternary\n");
    assert!(diags.is_empty(), "unexpected lex diagnostics: {diags:?}");
    assert!(
        tokens.iter().any(|t| matches!(t.kind, TokenKind::Ident(_))),
        "expected an Ident token for `ternary`, got {:?}",
        tokens.iter().map(|t| &t.kind).collect::<Vec<_>>()
    );
    // The resolved string must be exactly "ternary".
    let names: Vec<String> = lex_with_names("ternary\n")
        .into_iter()
        .filter_map(|(k, n)| {
            if matches!(k, TokenKind::Ident(_)) {
                Some(n)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(names, vec!["ternary".to_string()]);
}

#[test]
fn ternary_unpacked_keyword_lexes_as_identifier() {
    let (tokens, diags) = lex("ternary_unpacked\n");
    assert!(diags.is_empty(), "unexpected lex diagnostics: {diags:?}");
    let names: Vec<String> = lex_with_names("ternary_unpacked\n")
        .into_iter()
        .filter_map(|(k, n)| {
            if matches!(k, TokenKind::Ident(_)) {
                Some(n)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(names, vec!["ternary_unpacked".to_string()]);
    assert!(
        tokens.iter().any(|t| matches!(t.kind, TokenKind::Ident(_))),
        "expected an Ident token for `ternary_unpacked`"
    );
}

#[test]
fn parses_tensor_with_ternary_dtype() {
    // A `Tensor<[1024, 1024], ternary>` parameter annotation must parse with no
    // diagnostics, and the dtype symbol on the resulting AST node must resolve
    // back to "ternary".
    let src = "fn dummy(w: Tensor<[1024, 1024], ternary>) -> Tensor<[1024], f16>:\n    return w\n";

    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    assert!(lex_diags.is_empty(), "lex errors: {lex_diags:?}");

    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parse_result.diagnostics.is_empty(),
        "parse errors: {:?}",
        parse_result.diagnostics
    );

    // Walk: top-level stmt 0 should be a FnDef whose first param has a
    // `TypeExprKind::Tensor { dtype, .. }` where `dtype` resolves to "ternary".
    let stmt = &parse_result.module.stmts[0];
    let f = match &stmt.kind {
        StmtKind::FnDef(f) => f,
        other => panic!("expected FnDef, got {other:?}"),
    };
    let param = f.params.first().expect("missing parameter");
    let type_ann = param
        .type_ann
        .as_ref()
        .expect("parameter has no type annotation");
    let dtype_sym = match &type_ann.kind {
        TypeExprKind::Tensor { dtype, .. } => *dtype,
        other => panic!("expected TypeExprKind::Tensor, got {other:?}"),
    };
    let dtype_name = interner.resolve(dtype_sym.0).unwrap_or("");
    assert_eq!(dtype_name, "ternary");

    // And the semantic layer must resolve it to DType::TernaryPacked.
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    // The function-parameter NodeId is registered in the type map during
    // type-checking. Look for any Tensor entry whose dtype is TernaryPacked.
    let found = analysis.type_map.values().any(|ty| match ty {
        Type::Tensor { dtype, .. } => *dtype == DType::TernaryPacked,
        _ => false,
    });
    assert!(
        found,
        "expected a Tensor with DType::TernaryPacked in the type map; got: {:?}",
        analysis
            .type_map
            .values()
            .filter(|ty| matches!(ty, Type::Tensor { .. }))
            .collect::<Vec<_>>()
    );
    // No semantic errors about an undefined `ternary` dtype.
    let errs: Vec<String> = analysis
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message.clone())
        .collect();
    assert!(
        !errs.iter().any(|m| m.contains("undefined type `ternary`")),
        "semantic resolver still treats `ternary` as undefined; errors = {errs:?}"
    );
}

#[test]
fn parses_tensor_with_ternary_unpacked_dtype() {
    let src =
        "fn dummy(w: Tensor<[1024, 1024], ternary_unpacked>) -> Tensor<[1024], f16>:\n    return w\n";

    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    assert!(lex_diags.is_empty(), "lex errors: {lex_diags:?}");

    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parse_result.diagnostics.is_empty(),
        "parse errors: {:?}",
        parse_result.diagnostics
    );

    let stmt = &parse_result.module.stmts[0];
    let f = match &stmt.kind {
        StmtKind::FnDef(f) => f,
        other => panic!("expected FnDef, got {other:?}"),
    };
    let param = f.params.first().expect("missing parameter");
    let type_ann = param
        .type_ann
        .as_ref()
        .expect("parameter has no type annotation");
    let dtype_sym = match &type_ann.kind {
        TypeExprKind::Tensor { dtype, .. } => *dtype,
        other => panic!("expected TypeExprKind::Tensor, got {other:?}"),
    };
    let dtype_name = interner.resolve(dtype_sym.0).unwrap_or("");
    assert_eq!(dtype_name, "ternary_unpacked");

    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    let found = analysis.type_map.values().any(|ty| match ty {
        Type::Tensor { dtype, .. } => *dtype == DType::TernaryUnpacked,
        _ => false,
    });
    assert!(
        found,
        "expected a Tensor with DType::TernaryUnpacked in the type map; got: {:?}",
        analysis
            .type_map
            .values()
            .filter(|ty| matches!(ty, Type::Tensor { .. }))
            .collect::<Vec<_>>()
    );
    let errs: Vec<String> = analysis
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message.clone())
        .collect();
    assert!(
        !errs
            .iter()
            .any(|m| m.contains("undefined type `ternary_unpacked`")),
        "semantic resolver still treats `ternary_unpacked` as undefined; errors = {errs:?}"
    );
}

#[test]
fn dtype_byte_width_for_ternary_is_one() {
    // Both packed and unpacked ternary occupy one storage byte per atom (packed
    // = 4 trits/byte but the atom is still 1 byte; unpacked = 1 trit per i8).
    assert_eq!(DType::TernaryPacked.byte_width(), 1);
    assert_eq!(DType::TernaryUnpacked.byte_width(), 1);
}

#[test]
fn display_type_for_ternary_variants() {
    use nsl_semantic::types::display_type;
    assert_eq!(display_type(&Type::TernaryPacked), "ternary");
    assert_eq!(display_type(&Type::TernaryUnpacked), "ternary_unpacked");
}
