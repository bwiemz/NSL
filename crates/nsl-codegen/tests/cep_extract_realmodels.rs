//! Real-model exercise of the CEP extractor: coder-rl recognized, gpt2 refused.
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}
fn parse(path: &PathBuf) -> (nsl_ast::Module, nsl_lexer::Interner) {
    let src = std::fs::read_to_string(path).unwrap();
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _d) = nsl_lexer::tokenize(&src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    (parsed.module, interner)
}

#[test]
fn coder_rl_is_recognized() {
    let (module, interner) = parse(&workspace_root().join("models/coder-rl/model.nsl"));
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
    let spec = nsl_codegen::cep_extract::extract_model_spec(&module, &resolve)
        .expect("coder-rl is the canonical GQA model");
    assert_eq!(spec.d_model, 384);
    assert_eq!(spec.n_layers, 6);
    assert_eq!(spec.n_heads, vec![6; 6]);
    assert_eq!(spec.n_kv_heads, vec![3; 6]);
    assert_eq!(spec.vocab, 4096);
}

#[test]
fn gpt2_is_refused() {
    let (module, interner) = parse(&workspace_root().join("examples/gpt2.nsl"));
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();
    let err = nsl_codegen::cep_extract::extract_model_spec(&module, &resolve).unwrap_err();
    assert!(
        err.to_string().contains("GroupedQueryAttention") || err.to_string().contains("recognize")
    );
}
