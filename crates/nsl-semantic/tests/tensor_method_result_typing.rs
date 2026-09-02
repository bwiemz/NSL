//! Drift gate for the tensor-method member table in `check_member_access`
//! (roadmap item 1 residual, 2026-07-28).
//!
//! The table used to cover only `reshape`/`transpose` (plus scalar-ish
//! accessors); `expand`, `contiguous`, `unsqueeze`, `select`, `slice` and
//! `cumsum` fell to the `_ => Type::Unknown` arm, so the RESULT of any such
//! call was Unknown-typed — and so was every LATER link of a method chain,
//! because Unknown propagates. That was not just a diagnostics gap: codegen
//! ownership tracking (`track_owned_tensor_expr_result`) filters on the
//! result type, so each Unknown-typed anonymous chain link stranded its
//! handle. Measured on `GroupedQueryAttention::forward` ([2,1024,512],
//! RTX 5070 Ti): 4 stranded GPU blocks per call, two per
//! `expand(..).contiguous().reshape(..)` chain — the expand view pinning the
//! RoPE output block plus the contiguous materialisation itself. A whole
//! Coder-50M forward stranded +65 blocks / +228 MB per call, of which this
//! accounted for 32 blocks / ~96 MB.
//!
//! These tests type-check exact snippets of that shape and assert every
//! chain link is Tensor-typed. If a method is dropped from the member table
//! again, its link goes back to Unknown and the matching assertion here goes
//! red — BEFORE the leak has to be re-measured on hardware.

use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::stmt::StmtKind;
use nsl_errors::FileId;
use nsl_lexer::Interner;
use nsl_semantic::types::Type;

/// Parse + type-check `src`, then return the checked type of the initializer
/// of `let <name> = ...` at top level.
fn type_of_let_init(src: &str, name: &str) -> Type {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    assert!(lex_diags.is_empty(), "lex errors: {lex_diags:?}");
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parse_result.diagnostics.is_empty(),
        "parse errors: {:?}",
        parse_result.diagnostics
    );

    // Find the initializer expression id BEFORE analysis moves the module.
    let mut init_id = None;
    for stmt in &parse_result.module.stmts {
        if let StmtKind::VarDecl { pattern, value: Some(value), .. } = &stmt.kind
            && pattern_name(pattern, &interner).as_deref() == Some(name)
        {
            init_id = Some(value.id);
        }
    }
    let init_id = init_id.unwrap_or_else(|| panic!("no `let {name} = ...` in snippet"));

    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    analysis
        .type_map
        .get(&init_id)
        .cloned()
        .unwrap_or_else(|| panic!("no type recorded for `let {name}` initializer"))
}

fn pattern_name(pattern: &nsl_ast::pattern::Pattern, interner: &Interner) -> Option<String> {
    use nsl_ast::pattern::PatternKind;
    match &pattern.kind {
        PatternKind::Ident(sym) => interner.resolve(sym.0).map(|s| s.to_string()),
        _ => None,
    }
}

fn assert_tensor(src: &str, name: &str) {
    let ty = type_of_let_init(src, name);
    assert!(
        ty.is_tensor(),
        "`let {name}` initializer should be Tensor-typed, got {ty:?}. \
         A non-Tensor (especially Unknown) result here means the method was \
         dropped from the member table in check_member_access \
         (nsl-semantic/src/checker/ops.rs) — codegen ownership tracking \
         filters on this type, so Unknown re-opens the view-chain leak \
         (view_chain_leak_gate.rs)."
    );
}

const PRELUDE: &str = "let x = full([2, 4, 8], 1.0)\n";

#[test]
fn expand_result_is_tensor_typed() {
    let src = format!("{PRELUDE}let e = x.reshape([2, 4, 1, 8]).expand([2, 4, 2, 8])\n");
    assert_tensor(&src, "e");
}

#[test]
fn expand_gets_its_target_shape_at_the_call_site() {
    // Like reshape, a literal target list should become the checked shape —
    // expand legitimately changes the element count, so there is no product
    // proof, but the declared dims are taken as-is.
    let src = format!("{PRELUDE}let e = x.reshape([2, 4, 1, 8]).expand([2, 4, 2, 8])\n");
    let ty = type_of_let_init(&src, "e");
    match ty {
        Type::Tensor { ref shape, .. } => {
            let dims: Vec<i64> = shape
                .dims
                .iter()
                .filter_map(|d| match d {
                    nsl_semantic::types::Dim::Concrete(v) => Some(*v),
                    _ => None,
                })
                .collect();
            assert_eq!(
                dims,
                vec![2, 4, 2, 8],
                "expand target shape not propagated: {ty:?}"
            );
        }
        other => panic!("expected Tensor, got {other:?}"),
    }
}

#[test]
fn contiguous_result_is_tensor_typed() {
    let src = format!("{PRELUDE}let c = x.transpose(0, 1).contiguous()\n");
    assert_tensor(&src, "c");
}

#[test]
fn unsqueeze_select_slice_cumsum_results_are_tensor_typed() {
    let src = format!(
        "{PRELUDE}let u = x.unsqueeze(0)\nlet s = x.select(0, 1)\nlet sl = x.slice(0, 0, 2)\nlet cs = x.cumsum(0)\n"
    );
    for name in ["u", "s", "sl", "cs"] {
        assert_tensor(&src, name);
    }
}

/// The exact GQA regression shape: reshape → expand → contiguous → reshape,
/// all anonymous links. The FINAL link's type is what the codegen consults
/// when deciding whether the bound value participates in cleanup, and every
/// intermediate link's type gates its own tracking. One Unknown anywhere in
/// the chain re-opens the leak.
#[test]
fn the_gqa_expand_contiguous_reshape_chain_stays_tensor_typed_end_to_end() {
    let src = format!(
        "{PRELUDE}let k_exp = x.reshape([2, 4, 1, 8]).expand([2, 4, 2, 8]).contiguous().reshape([2, 8, 8])\n"
    );
    // The bound (final) link:
    assert_tensor(&src, "k_exp");

    // And every intermediate link. Walk the initializer's nested call spine
    // and assert each node in the chain is Tensor-typed.
    let mut interner = Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(&src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    let mut chain_ids = Vec::new();
    for stmt in &parse_result.module.stmts {
        if let StmtKind::VarDecl { pattern, value: Some(value), .. } = &stmt.kind
            && pattern_name(pattern, &interner).as_deref() == Some("k_exp")
        {
            let mut cur: &Expr = value;
            loop {
                chain_ids.push(cur.id);
                let ExprKind::Call { callee, .. } = &cur.kind else { break };
                let ExprKind::MemberAccess { object, .. } = &callee.kind else { break };
                cur = object;
            }
        }
    }
    assert!(
        chain_ids.len() >= 4,
        "expected at least 4 chain links (reshape/expand/contiguous/reshape), got {}",
        chain_ids.len()
    );
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);
    for id in chain_ids {
        let ty = analysis.type_map.get(&id).cloned().unwrap_or(Type::Unknown);
        assert!(
            ty.is_tensor(),
            "chain link {id:?} is {ty:?}, not Tensor — an Unknown link makes \
             codegen ownership tracking drop it and the handle strands \
             (roadmap item 1 residual)"
        );
    }
}
