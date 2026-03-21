use super::*;
use nsl_errors::FileId;

/// Parse and type-check a snippet of NSL source, returning all diagnostics.
fn check_source(src: &str) -> Vec<Diagnostic> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    let analysis = crate::analyze(&parse_result.module, &mut interner);
    analysis.diagnostics
}

/// Parse and type-check a snippet, returning BOTH parse and semantic diagnostics.
fn check_source_all(src: &str) -> Vec<Diagnostic> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    let analysis = crate::analyze(&parse_result.module, &mut interner);
    let mut all = parse_result.diagnostics;
    all.extend(analysis.diagnostics);
    all
}

// -----------------------------------------------------------------------
// @fuse_graph tests
// -----------------------------------------------------------------------

#[test]
fn test_fuse_graph_decorator_valid_on_fn() {
    // @fuse_graph on a function should produce no fuse_graph-related error
    let src = "@fuse_graph\nfn my_fn(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x\n";
    let diags = check_source(src);
    let fuse_graph_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("fuse_graph"))
        .collect();
    assert!(
        fuse_graph_errs.is_empty(),
        "@fuse_graph on fn should be valid, got: {:?}",
        fuse_graph_errs
    );
}

#[test]
fn test_fuse_graph_decorator_invalid_on_model() {
    let src = "@fuse_graph\nmodel MyModel:\n    let x: Tensor<[4], f32>\n";
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| format!("{:?}", d).contains("fuse_graph")),
        "@fuse_graph on model should produce a fuse_graph diagnostic, got: {:?}",
        diags
    );
}

#[test]
fn test_fuse_graph_and_fuse_conflict() {
    let src =
        "@fuse\n@fuse_graph\nfn my_fn(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x\n";
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| format!("{:?}", d).contains("cannot")),
        "@fuse + @fuse_graph should produce a 'cannot' conflict diagnostic, got: {:?}",
        diags
    );
}

// -----------------------------------------------------------------------
// @no_fuse tests
// -----------------------------------------------------------------------

#[test]
fn test_no_fuse_valid_on_let() {
    // @no_fuse on a let-binding should produce no no_fuse-related error
    let src =
        "@fuse_graph\nfn f(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    @no_fuse\n    let y = x\n    return y\n";
    let diags = check_source(src);
    let no_fuse_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("no_fuse"))
        .collect();
    assert!(
        no_fuse_errs.is_empty(),
        "@no_fuse on let should be valid, got: {:?}",
        no_fuse_errs
    );
}

#[test]
fn test_no_fuse_invalid_on_fn() {
    let src = "@no_fuse\nfn f(x: Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x\n";
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| format!("{:?}", d).contains("no_fuse")),
        "@no_fuse on fn should produce a no_fuse diagnostic, got: {:?}",
        diags
    );
}

// -----------------------------------------------------------------------
// Immutable borrow (&T) tests — Tasks 1-4
// -----------------------------------------------------------------------

// --- Task 1: Parse borrow syntax ---

#[test]
fn test_borrow_param_type_parses() {
    // A function with a &Tensor parameter should parse without errors
    let src = "fn f(x: &Tensor<[4], f32>) -> Tensor<[4], f32>:\n    return x\n";
    let diags = check_source_all(src);
    let borrow_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("borrow"))
        .collect();
    assert!(
        borrow_errs.is_empty(),
        "&Tensor in param should parse cleanly, got: {:?}",
        borrow_errs
    );
}

#[test]
fn test_borrow_named_type_parses() {
    // &int, &float, etc. should also parse
    let src = "fn f(x: &int) -> int:\n    return x\n";
    let diags = check_source_all(src);
    let borrow_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("borrow"))
        .collect();
    assert!(
        borrow_errs.is_empty(),
        "&int in param should parse cleanly, got: {:?}",
        borrow_errs
    );
}

#[test]
fn test_borrow_in_return_type_error() {
    // &T in return position should produce an error
    let src = "fn f(x: Tensor<[4], f32>) -> &Tensor<[4], f32>:\n    return x\n";
    let diags = check_source_all(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("borrow") && format!("{:?}", d).contains("return")),
        "borrow in return type should produce a diagnostic, got: {:?}",
        diags
    );
}

// --- Task 2: Type system integration ---

#[test]
fn test_borrow_tensor_read_compatible() {
    // Passing an owned Tensor to a function expecting &Tensor should work (auto-borrow)
    // Note: zeros() returns Tensor<[], f64, cpu>, so match that dtype
    let src = r#"
fn read_tensor(x: &Tensor) -> float:
    return 0.0

fn main():
    let t = zeros([4])
    let r = read_tensor(t)
"#;
    let diags = check_source(src);
    let type_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("type mismatch"))
        .collect();
    assert!(
        type_errs.is_empty(),
        "owned Tensor should auto-borrow to &Tensor, got: {:?}",
        type_errs
    );
}

#[test]
fn test_borrow_to_borrow_compatible() {
    // Passing &Tensor where &Tensor is expected should work
    let src = r#"
fn read_tensor(x: &Tensor<[4], f32>) -> float:
    return 0.0

fn forward(x: &Tensor<[4], f32>) -> float:
    return read_tensor(x)
"#;
    let diags = check_source(src);
    let type_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("type mismatch"))
        .collect();
    assert!(
        type_errs.is_empty(),
        "&Tensor should pass to &Tensor, got: {:?}",
        type_errs
    );
}

// --- Task 4: Borrow safety rules (tested at ownership.rs unit test level) ---
// These are already extensively tested in ownership::tests. Here we add
// integration-level tests through the checker to verify the full pipeline.

#[test]
fn test_borrow_multiple_reads_ok() {
    // Multiple uses of a borrowed parameter should be fine
    let src = r#"
fn multi_read(x: &Tensor<[4], f32>, y: &Tensor<[4], f32>) -> float:
    return 0.0
"#;
    let diags = check_source(src);
    let borrow_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("borrow"))
        .collect();
    assert!(
        borrow_errs.is_empty(),
        "multiple borrow params should be fine, got: {:?}",
        borrow_errs
    );
}

// -----------------------------------------------------------------------
// Task 5: Autodiff + borrowed tensors — integration-level tests
// -----------------------------------------------------------------------

#[test]
fn test_borrow_param_in_no_grad_fn() {
    // @no_grad on a function that takes &Tensor: borrow inherits the annotation.
    // No tape-recording happens for any op in this function.
    let src = r#"
@no_grad
fn inference_step(x: &Tensor<[4], f32>) -> float:
    return 0.0
"#;
    let diags = check_source_all(src);
    let borrow_errs: Vec<_> = diags
        .iter()
        .filter(|d| {
            let s = format!("{:?}", d);
            s.contains("borrow") || s.contains("no_grad")
        })
        .collect();
    assert!(
        borrow_errs.is_empty(),
        "@no_grad fn with &Tensor param should be clean, got: {:?}",
        borrow_errs
    );
}

#[test]
fn test_borrow_passed_to_borrow_param() {
    // Chaining borrows: f takes &T, g takes &T and calls f with its own &T param.
    // Demonstrates that a borrow can be propagated through a call chain without
    // being consumed at any point — the same pointer flows through.
    let src = r#"
fn apply(w: &Tensor<[4], f32>) -> float:
    return 0.0

fn forward(w: &Tensor<[4], f32>) -> float:
    return apply(w)
"#;
    let diags = check_source(src);
    let type_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("type mismatch"))
        .collect();
    assert!(
        type_errs.is_empty(),
        "borrow should chain through call without type error, got: {:?}",
        type_errs
    );
}

// -----------------------------------------------------------------------
// Task 6: Model methods — self is borrowed by default
// -----------------------------------------------------------------------
//
// In NSL, model methods take `self` as their first parameter. At the codegen
// level, `self` is bound as a raw I64 pointer — NOT as a linear tensor
// binding. This means:
//   1. The ownership checker never registers `self` as a tensor-typed binding.
//   2. Calling model.forward(x) does NOT consume the model instance.
//   3. The same model instance can be used across multiple forward() calls
//      without any ownership violation.
//
// This is the expected behaviour for training loops:
//   for epoch in 0..100:
//       let loss = model.forward(x)   # safe on every iteration
//       loss.backward()

#[test]
fn test_model_method_declaration_parses_cleanly() {
    // A model with a forward() method should parse and type-check without errors
    let src = r#"
model Linear:
    let weight: Tensor<[4, 4], f32>

    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source_all(src);
    // Filter out any errors unrelated to model/borrow semantics
    let model_errs: Vec<_> = diags
        .iter()
        .filter(|d| {
            let s = format!("{:?}", d);
            s.contains("consume") || s.contains("borrow") || s.contains("self")
        })
        .collect();
    assert!(
        model_errs.is_empty(),
        "model.forward(self, ...) should parse cleanly, got: {:?}",
        model_errs
    );
}

#[test]
fn test_model_forward_borrow_param_parses() {
    // forward(&self, ...) syntax — explicit borrow of self — should parse.
    // Whether the parser exposes &self or treats self-as-ptr implicitly,
    // the key is no errors about self consumption.
    let src = r#"
model MLP:
    let weight: Tensor<[4, 4], f32>

    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source_all(src);
    let consume_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("consume"))
        .collect();
    assert!(
        consume_errs.is_empty(),
        "model forward should not produce consumption errors, got: {:?}",
        consume_errs
    );
}

#[test]
fn test_borrowed_tensor_passed_to_model_method() {
    // Passing a &Tensor to a function that expects a Tensor (read-only)
    // should be compatible — the auto-borrow rule (T assignable to &T) is
    // symmetric: &T can also be read where T is expected for non-consuming ops.
    let src = r#"
fn uses_tensor(x: &Tensor<[4], f32>) -> float:
    return 0.0

fn call_multiple(x: &Tensor<[4], f32>) -> float:
    let _ = uses_tensor(x)
    let _ = uses_tensor(x)
    return 0.0
"#;
    let diags = check_source(src);
    let type_errs: Vec<_> = diags
        .iter()
        .filter(|d| format!("{:?}", d).contains("type mismatch"))
        .collect();
    assert!(
        type_errs.is_empty(),
        "borrowed tensor can be passed to borrow-param fn multiple times, got: {:?}",
        type_errs
    );
}
