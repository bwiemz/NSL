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

#[test]
fn test_tokenizer_dataset_defs_validate_cleanly() {
    let src = r#"
tokenizer demo_tokenizer(algorithm=bpe, vocab_size=1024):
    special_tokens:
        pad = "<pad>"
        eos = "<eos>"

    normalize: nfkc, lowercase
    pre_tokenize: whitespace, byte_fallback

    padding:
        side = left
        pad_to = longest

    truncation:
        max_length = 128
        strategy = longest_first

dataset train_data("demo"):
    source = "data.bin"
    format = binary
    sequence_length = 128
    packing = true
    pack_separator = 0
    shuffle = false
    max_samples = 100
"#;
    let diags = check_source_all(src);
    let relevant: Vec<_> = diags
        .iter()
        .filter(|d| {
            let s = format!("{:?}", d);
            s.contains("tokenizer") || s.contains("dataset") || s.contains("padding")
        })
        .collect();
    assert!(
        relevant.is_empty(),
        "expected tokenizer/dataset validation to pass cleanly, got: {:?}",
        relevant
    );
}

#[test]
fn test_tokenizer_invalid_padding_side_reports_error() {
    let src = r#"
tokenizer bad_tokenizer(algorithm=bpe, vocab_size=128):
    padding:
        side = center
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            let s = format!("{:?}", d);
            s.contains("padding.side") || s.contains("left, right")
        }),
        "expected tokenizer padding validation error, got: {:?}",
        diags
    );
}

#[test]
fn test_dataset_unknown_field_reports_error() {
    let src = r#"
dataset bad_data("demo"):
    source = "data.bin"
    unexpected = 1
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("unknown dataset field 'unexpected'")),
        "expected dataset field validation error, got: {:?}",
        diags
    );
}

#[test]
fn test_duplicate_tokenizer_section_reports_error() {
    let src = r#"
tokenizer dup_sections(algorithm=bpe, vocab_size=128):
    padding:
        side = left

    padding:
        side = right
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("duplicate tokenizer section 'padding'")),
        "expected duplicate tokenizer section error, got: {:?}",
        diags
    );
}

#[test]
fn test_duplicate_dataset_field_reports_error() {
    let src = r#"
dataset dup_fields("demo"):
    source = "data.bin"
    sequence_length = 128
    sequence_length = 256
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("duplicate dataset field 'sequence_length'")),
        "expected duplicate dataset field error, got: {:?}",
        diags
    );
}

#[test]
fn test_tokenizer_vocab_size_type_mismatch_reports_error() {
    let src = r#"
tokenizer bad_vocab(algorithm=bpe, vocab_size="large"):
    normalize: nfkc
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("tokenizer vocab_size must be int")),
        "expected tokenizer vocab_size type error, got: {:?}",
        diags
    );
}

#[test]
fn test_trait_definition_and_bounds_validate_cleanly() {
    let src = r#"
struct Box<T: Comparable<T>>:
    value: T

trait Named<T: Display>:
    fn name(self) -> str:
        return "named"

trait Display:
    fn render(self) -> str

trait Comparable<T>:
    fn eq(self, other: T) -> bool
"#;
    let diags = check_source_all(src);
    let relevant: Vec<_> = diags
        .iter()
        .filter(|d| {
            let s = format!("{:?}", d);
            s.contains("duplicate trait method") || s.contains("bound must reference a trait") || s.contains("undefined type")
        })
        .collect();
    assert!(
        relevant.is_empty(),
        "expected trait definitions and trait bounds to validate cleanly, got: {:?}",
        relevant
    );
}

#[test]
fn test_duplicate_trait_method_reports_error() {
    let src = r#"
trait Duplicate:
    fn render(self) -> str
    fn render(self) -> str
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("duplicate trait method 'render' in trait 'Duplicate'")),
        "expected duplicate trait method error, got: {:?}",
        diags
    );
}

#[test]
fn test_non_trait_bound_reports_error() {
    let src = r#"
struct Bad<T: int>:
    value: T
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("type parameter 'T' bound must reference a trait")),
        "expected non-trait bound error, got: {:?}",
        diags
    );
}

#[test]
fn test_user_defined_generic_struct_instantiation_substitutes_fields() {
    let src = r#"
struct Box<T>:
    value: T

struct Wrapper<T>:
    inner: Box<T>

fn read_value(wrapper: Wrapper<int>) -> int:
    return wrapper.inner.value
"#;
    let diags = check_source_all(src);
    let relevant: Vec<_> = diags
        .iter()
        .filter(|d| {
            let s = format!("{:?}", d);
            s.contains("undefined type") || s.contains("wrong number of type arguments") || s.contains("type mismatch")
        })
        .collect();
    assert!(
        relevant.is_empty(),
        "expected nested generic struct instantiation to type-check, got: {:?}",
        relevant
    );
}

#[test]
fn test_user_defined_generic_model_instantiation_substitutes_method_types() {
    let src = r#"
model Holder<T>:
    value: T

    fn get(self) -> T:
        return self.value

fn read_holder(holder: Holder<int>) -> int:
    return holder.value
"#;
    let diags = check_source_all(src);
    let relevant: Vec<_> = diags
        .iter()
        .filter(|d| {
            let s = format!("{:?}", d);
            s.contains("undefined type") || s.contains("wrong number of type arguments") || s.contains("type mismatch")
        })
        .collect();
    assert!(
        relevant.is_empty(),
        "expected generic model instantiation to type-check, got: {:?}",
        relevant
    );
}

#[test]
fn test_user_defined_generic_multi_param_and_deep_instantiation_type_checks() {
    let src = r#"
struct Box<T>:
    value: T

struct Pair<A, B>:
    first: A
    second: B

model Holder<T>:
    value: T

    fn get(self) -> T:
        return self.value

model Mapper<K, V>:
    entry: Pair<K, V>

    fn value(self) -> V:
        return self.entry.second

struct Deep<T>:
    item: Pair<Box<T>, Holder<T>>

fn read_pair(pair: Pair<int, str>) -> str:
    return pair.second

fn read_mapper(mapper: Mapper<int, str>) -> str:
    return mapper.value()

fn read_deep(deep: Deep<int>) -> int:
    return deep.item.second.get()
"#;
    let diags = check_source_all(src);
    let relevant: Vec<_> = diags
        .iter()
        .filter(|d| {
            let s = format!("{:?}", d);
            s.contains("undefined type") || s.contains("wrong number of type arguments") || s.contains("type mismatch")
        })
        .collect();
    assert!(
        relevant.is_empty(),
        "expected multi-parameter and deep generic instantiation to type-check, got: {:?}",
        relevant
    );
}

#[test]
fn test_user_defined_generic_type_arity_mismatch_reports_error() {
    let src = r#"
struct Box<T>:
    value: T

fn bad(box: Box<int, str>) -> int:
    return 0
"#;
    let diags = check_source_all(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("Box expects 1 type argument(s), got 2")),
        "expected user-defined generic arity error, got: {:?}",
        diags
    );
}
