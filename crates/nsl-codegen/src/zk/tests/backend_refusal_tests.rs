//! M55: honest backend refusal tests for `compile_zk`.
//!
//! `nsl build --zk-backend halo2` (or `plonky3`) must refuse loudly with
//! [`ZkError::BackendNotImplemented`] instead of silently running the folding
//! backend and mislabeling the resulting proof as the requested backend.
//! The guard lives at the top of `compile_zk` (before `ast_to_zkdag`) because
//! `compile_zk_from_dag` errors are swallowed by a `.ok()` inside `compile_zk`.

use nsl_errors::{FileId, Level};
use nsl_lexer::{tokenize, Interner};

use crate::zk::backend::{ZkBackendType, ZkConfig, ZkError, ZkMode};

/// Run the frontend (lex -> parse -> semantic analysis) on an NSL snippet and
/// return everything `compile_zk` needs. The module must contain exactly one
/// top-level `fn`.
fn frontend(
    source: &str,
) -> (
    nsl_ast::Module,
    Interner,
    nsl_semantic::checker::TypeMap,
) {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = tokenize(source, FileId(0), &mut interner);
    assert!(
        lex_diags
            .iter()
            .all(|diag| !matches!(diag.level, Level::Error)),
        "test source must lex cleanly: {lex_diags:?}"
    );

    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        parsed
            .diagnostics
            .iter()
            .all(|diag| !matches!(diag.level, Level::Error)),
        "test source must parse cleanly: {:?}",
        parsed.diagnostics
    );

    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    (parsed.module, interner, analysis.type_map)
}

/// Extract the first top-level `FnDef` from a parsed module (bare or decorated).
fn first_fn_def(module: &nsl_ast::Module) -> &nsl_ast::decl::FnDef {
    for stmt in &module.stmts {
        match &stmt.kind {
            nsl_ast::stmt::StmtKind::FnDef(fd) => return fd,
            nsl_ast::stmt::StmtKind::Decorated { stmt: inner, .. } => {
                if let nsl_ast::stmt::StmtKind::FnDef(fd) = &inner.kind {
                    return fd;
                }
            }
            _ => {}
        }
    }
    panic!("test module must contain a top-level fn");
}

/// Minimal `@zk_proof`-shaped function: one input, one arithmetic op.
const ZK_FN_SOURCE: &str = "fn zk_infer(x: float) -> float:\n    return x * x\n";

fn compile_with_backend(
    backend: ZkBackendType,
) -> Result<crate::zk::ZkCompileResult, ZkError> {
    let (module, interner, type_map) = frontend(ZK_FN_SOURCE);
    let fn_def = first_fn_def(&module);
    let config = ZkConfig {
        backend,
        ..Default::default()
    };
    crate::zk::compile_zk(
        fn_def,
        ZkMode::ArchitectureAttestation,
        &config,
        &type_map,
        &interner,
    )
}

#[test]
fn halo2_backend_is_refused_not_silently_folded() {
    let err = compile_with_backend(ZkBackendType::Halo2)
        .err()
        .expect("halo2 must be refused, not silently run the folding backend");
    match err {
        ZkError::BackendNotImplemented { backend, message } => {
            assert_eq!(backend, "halo2");
            assert!(
                message.contains("deprecated"),
                "halo2 refusal must note the backend is deprecated/removed: {message}"
            );
            assert!(
                message.contains("--zk-backend folding"),
                "halo2 refusal must point the user at the folding backend: {message}"
            );
        }
        other => panic!("expected BackendNotImplemented, got: {other}"),
    }
}

#[test]
fn plonky3_backend_is_refused_not_silently_folded() {
    let err = compile_with_backend(ZkBackendType::Plonky3)
        .err()
        .expect("plonky3 must be refused, not silently run the folding backend");
    match err {
        ZkError::BackendNotImplemented { backend, message } => {
            assert_eq!(backend, "plonky3");
            assert!(
                message.contains("not yet wired"),
                "plonky3 refusal must note the prover is not wired into compilation: {message}"
            );
            assert!(
                message.contains("--zk-backend folding"),
                "plonky3 refusal must point the user at the folding backend: {message}"
            );
        }
        other => panic!("expected BackendNotImplemented, got: {other}"),
    }
}

#[test]
fn folding_backend_still_compiles_and_proves() {
    let result = compile_with_backend(ZkBackendType::Folding)
        .expect("folding backend must keep compiling unchanged");
    assert!(
        !result.zkir.instructions.is_empty(),
        "folding path must still lower the DAG to ZK-IR"
    );
    let proof = result
        .proof
        .expect("folding path must still produce a proof");
    assert!(proof.num_folds > 0, "proof must contain at least one fold");
    assert!(!proof.data.is_empty(), "proof data must be non-empty");
}
