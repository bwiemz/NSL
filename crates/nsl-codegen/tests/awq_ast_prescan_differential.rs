use std::collections::HashMap;
use std::path::PathBuf;

use nsl_ast::decl::ModelMember;
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::StmtKind;
use nsl_ast::types::TypeExprKind;
use nsl_codegen::calibration::{
    discover_awq_projections_from_state, pre_scan_awq_projections_from_ast, DiscoveredProjection,
};
use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn fixture(name: &str) -> PathBuf {
    repo_root().join("tests").join("fixtures").join(name)
}

fn is_awq_quantize_model(stmt: &nsl_ast::stmt::Stmt, interner: &Interner) -> bool {
    match &stmt.kind {
        StmtKind::Decorated { decorators, stmt } => {
            if !matches!(stmt.kind, StmtKind::ModelDef(_)) {
                return false;
            }
            decorators.iter().any(|decorator| {
                if decorator.name.len() != 1 {
                    return false;
                }
                interner.resolve(decorator.name[0].0).unwrap_or("") == "quantize"
                    && decorator.args.as_ref().is_some_and(|args| {
                    args.iter().any(|arg| {
                        arg.name.is_some_and(|arg_name| {
                            interner.resolve(arg_name.0).unwrap_or("") == "dtype"
                        })
                            && matches!(&arg.value.kind, ExprKind::StringLiteral(value) if value == "awq4")
                    })
                })
            })
        }
        _ => false,
    }
}

fn model_def_from_stmt(stmt: &nsl_ast::stmt::Stmt) -> Option<&nsl_ast::decl::ModelDef> {
    match &stmt.kind {
        StmtKind::ModelDef(md) => Some(md),
        StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
            StmtKind::ModelDef(md) => Some(md),
            _ => None,
        },
        _ => None,
    }
}

fn extract_shape_from_tensor_init(expr: &nsl_ast::expr::Expr) -> Option<[u32; 2]> {
    let ExprKind::Call { args, .. } = &expr.kind else {
        return None;
    };
    let first = args.first()?;
    let ExprKind::ListLiteral(elements) = &first.value.kind else {
        return None;
    };
    if elements.len() != 2 {
        return None;
    }
    let mut dims = [0u32; 2];
    for (index, element) in elements.iter().enumerate() {
        let ExprKind::IntLiteral(value) = element.kind else {
            return None;
        };
        dims[index] = u32::try_from(value).ok()?;
    }
    Some(dims)
}

fn collect_in_compile_discovery(
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> Vec<DiscoveredProjection> {
    let mut discovered = Vec::new();

    for stmt in &ast.stmts {
        if !is_awq_quantize_model(stmt, interner) {
            continue;
        }
        let Some(model_def) = model_def_from_stmt(stmt) else {
            continue;
        };

        let model_name = interner.resolve(model_def.name.0).unwrap_or("");
        let mut tensor_shapes = HashMap::new();
        let mut forward_body = None;

        for member in &model_def.members {
            match member {
                ModelMember::LayerDecl {
                    name,
                    type_ann,
                    init,
                    ..
                } => {
                    let TypeExprKind::Named(type_sym) = &type_ann.kind else {
                        continue;
                    };
                    if interner.resolve(type_sym.0).unwrap_or("") != "Tensor" {
                        continue;
                    }
                    let Some(init_expr) = init else {
                        continue;
                    };
                    let Some(shape) = extract_shape_from_tensor_init(init_expr) else {
                        continue;
                    };
                    tensor_shapes.insert(
                        interner.resolve(name.0).unwrap_or("").to_string(),
                        format!("Tensor<[{}, {}], f32>", shape[0], shape[1]),
                    );
                }
                ModelMember::Method(fn_def, _) => {
                    if interner.resolve(fn_def.name.0).unwrap_or("") == "forward" {
                        forward_body = Some(&fn_def.body);
                    }
                }
            }
        }

        let mut per_model = discover_awq_projections_from_state(
            model_name,
            forward_body,
            &HashMap::new(),
            &tensor_shapes,
            &[],
            interner,
        )
        .expect("in-compile discovery must succeed for AWQ fixture");
        discovered.append(&mut per_model);
    }

    discovered.sort_by(|left, right| left.projection.0.cmp(&right.projection.0));
    discovered
}

#[test]
fn differential_agrees_on_awq_calibration_mlp_fixture() {
    let source = std::fs::read_to_string(fixture("awq_calibration_mlp.nsl"))
        .expect("fixture readable");

    let mut interner = Interner::new();
    let (tokens, lex_diags) = tokenize(&source, FileId(0), &mut interner);
    assert!(
        lex_diags.iter().all(|diag| !matches!(diag.level, Level::Error)),
        "fixture must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.level, Level::Error)),
        "fixture must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let pre_scan = pre_scan_awq_projections_from_ast(&parsed.module, &interner);
    let in_compile = collect_in_compile_discovery(&parsed.module, &interner);

    assert_eq!(
        pre_scan, in_compile,
        "AST pre-scan and in-compile discovery diverged on awq_calibration_mlp.nsl"
    );
}
