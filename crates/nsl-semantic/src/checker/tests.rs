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

// -----------------------------------------------------------------------
// CFTP v9 M4: `mean_doc_length` / `doc_length_stddev` accept Float literals
// -----------------------------------------------------------------------

#[test]
fn v9_m4_mean_doc_length_accepts_float_literal() {
    // Calibration tools naturally emit fractional averages like 384.7.
    // The semantic checker must accept these; codegen rounds to u32.
    let src = r#"
dataset calibrated("demo"):
    source = "data.bin"
    sequence_length = 1024
    packing = true
    mean_doc_length = 384.7
    doc_length_stddev = 42.5
"#;
    let diags = check_source(src);
    let errs: Vec<_> = diags.iter().filter(|d| d.level == nsl_errors::Level::Error).collect();
    assert!(
        errs.is_empty(),
        "v9 M4: Float literal for mean_doc_length/doc_length_stddev must not error; got: {:?}",
        errs
    );
}

#[test]
fn v9_m4_mean_doc_length_still_accepts_int_literal() {
    // Backwards compatibility: existing Int-literal fixtures must keep working.
    let src = r#"
dataset intcorpus("demo"):
    source = "data.bin"
    sequence_length = 1024
    packing = true
    mean_doc_length = 384
    doc_length_stddev = 42
"#;
    let diags = check_source(src);
    let errs: Vec<_> = diags.iter().filter(|d| d.level == nsl_errors::Level::Error).collect();
    assert!(
        errs.is_empty(),
        "v9 M4: Int literal must still be accepted (regression guard); got: {:?}",
        errs
    );
}

#[test]
fn v9_m4_mean_doc_length_rejects_string() {
    // The widening is Int|Float only — not "anything goes". String must still error.
    let src = r#"
dataset weird("demo"):
    source = "data.bin"
    sequence_length = 1024
    packing = true
    mean_doc_length = "not a number"
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("mean_doc_length must be int or float")),
        "v9 M4: String must error with 'int or float' message; got: {:?}",
        diags
    );
}

// -----------------------------------------------------------------------
// CFTP v9 L4: non-negative literal validation for length stats
// -----------------------------------------------------------------------

#[test]
fn v9_l4_mean_doc_length_rejects_negative_int() {
    let src = r#"
dataset bad_stats("demo"):
    source = "data.bin"
    sequence_length = 1024
    packing = true
    mean_doc_length = -50
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("dataset mean_doc_length must be non-negative")),
        "v9 L4: negative Int must be refused up-front; got: {:?}",
        diags
    );
}

#[test]
fn v9_l4_doc_length_stddev_rejects_negative_float() {
    let src = r#"
dataset bad_stats("demo"):
    source = "data.bin"
    sequence_length = 1024
    packing = true
    doc_length_stddev = -3.14
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("dataset doc_length_stddev must be non-negative")),
        "v9 L4: negative Float must be refused up-front; got: {:?}",
        diags
    );
}

#[test]
fn v9_l4_sequence_length_rejects_negative() {
    // Pre-v9 gap: `sequence_length` also silently wraps to a huge u32
    // downstream when negative. v9 L4 closes this for all length fields.
    let src = r#"
dataset seq_len_bad("demo"):
    source = "data.bin"
    sequence_length = -100
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("dataset sequence_length must be non-negative")),
        "v9 L4: negative sequence_length must be refused up-front; got: {:?}",
        diags
    );
}

#[test]
fn v9_l4_max_sequence_length_rejects_negative() {
    let src = r#"
dataset ms_bad("demo"):
    source = "data.bin"
    sequence_length = 1024
    packing = true
    max_sequence_length = -1024
"#;
    let diags = check_source(src);
    assert!(
        diags
            .iter()
            .any(|d| format!("{:?}", d).contains("dataset max_sequence_length must be non-negative")),
        "v9 L4: negative max_sequence_length must be refused up-front; got: {:?}",
        diags
    );
}

#[test]
fn v9_l4_separator_token_id_still_allows_any_int() {
    // separator_token_id is a token id (which can legally be negative
    // as a sentinel). It intentionally stays out of the L4 non-negative
    // check. This test pins that behavior so a future L4 extension
    // doesn't accidentally sweep it in.
    let src = r#"
dataset with_neg_sep("demo"):
    source = "data.bin"
    sequence_length = 1024
    packing = true
    separator_token_id = -1
"#;
    let diags = check_source(src);
    let errs: Vec<_> = diags.iter().filter(|d| d.level == nsl_errors::Level::Error).collect();
    assert!(
        errs.is_empty(),
        "v9 L4: separator_token_id must continue accepting any i64 (including negative sentinels); got: {:?}",
        errs
    );
}

#[test]
fn v9_l4_zero_length_still_allowed() {
    // Zero is not negative — the L4 check must not accidentally refuse it.
    // Downstream `pca_activation::validate_config` refuses zero mean_doc_length
    // separately (that's a semantic error, not a "wrong sign" one).
    let src = r#"
dataset zero_ok("demo"):
    source = "data.bin"
    sequence_length = 0
"#;
    let diags = check_source(src);
    // The L4 check itself should not fire. (Other checks may fire but not
    // the "must be non-negative" message.)
    assert!(
        !diags
            .iter()
            .any(|d| format!("{:?}", d).contains("must be non-negative")),
        "v9 L4: zero is not negative; must not trip the L4 check; got: {:?}",
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

// -----------------------------------------------------------------------
// Decorator collection tests (WRGA + freeze + adapter + CFTP fused_lm_ce)
// -----------------------------------------------------------------------

/// Parse and analyze a snippet, returning the full AnalysisResult.
fn analyze_source(src: &str) -> crate::AnalysisResult {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    crate::analyze(&parse_result.module, &mut interner)
}

// -----------------------------------------------------------------------
// CFTP §4.4 G3 @fused_lm_ce decorator collection tests (Sprint 2)
// -----------------------------------------------------------------------

#[test]
fn fused_lm_ce_decorator_on_train_block_is_captured() {
    let src = r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])
let y = zeros([4, 1])

@fused_lm_ce(enabled = true, vocab_tile = 1024)
train(model = m, epochs = 5):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.fused_ce_configs.len(),
        1,
        "expected one @fused_lm_ce config, got {:?}",
        res.fused_ce_configs.len()
    );
    let cfg = &res.fused_ce_configs[0];
    assert!(cfg.enabled, "enabled=true should round-trip");
    assert_eq!(cfg.vocab_tile, Some(1024));
}

#[test]
fn fused_lm_ce_decorator_on_non_train_target_errors() {
    let src = r#"
@fused_lm_ce(enabled = true)
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    // The decorator must be rejected when not on a `train` block.  The
    // config is NOT captured (only train-attached configs flow through).
    assert!(
        res.fused_ce_configs.is_empty(),
        "@fused_lm_ce on non-train target must NOT be captured, got {:?}",
        res.fused_ce_configs.len()
    );
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d)
                .contains("@fused_lm_ce may only be applied to a `train` block")),
        "expected target-mismatch diagnostic, got: {:?}",
        res.diagnostics
    );
}

#[test]
fn fused_lm_ce_decorator_vocab_tile_alignment_errors() {
    // vocab_tile must be a multiple of 128 (per the v1 fused-CE kernel
    // invariant — see validate_fused_ce_decorator in cftp.rs).
    let src = r#"
let m = Tiny()
@fused_lm_ce(enabled = true, vocab_tile = 100)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = 0
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("must be a multiple of 128")),
        "expected vocab_tile alignment diagnostic, got: {:?}",
        res.diagnostics
    );
}

/// CFTP v10 (item 3) LIFTED: two `@fused_lm_ce` decorators on DIFFERENT
/// train blocks in the same compilation unit are now ACCEPTED.  Codegen
/// dispatches each block to its own config via the
/// `train_block_stmt_id` field.
///
/// Pre-v10 (v5 follow-on Finding 1 HIGH), codegen read
/// `fused_ce_configs.first()` for every train block, silently binding
/// EVERY block to the FIRST decorator's dtype/vocab hint — so the
/// checker had to refuse the second decorator entirely. Item 3
/// installs per-block dispatch (`active_fused_ce_config` +
/// `set_active_fused_ce_config_for_train_block`), lifting the refusal.
///
/// This test pins:
///   * both configs land in `AnalysisResult.fused_ce_configs`
///   * each config's `train_block_stmt_id` matches its train-block Stmt id
///   * no refusal diagnostic fires
#[test]
fn fused_lm_ce_two_train_blocks_each_get_own_config() {
    let src = r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])
let y = zeros([4, 1])

@fused_lm_ce(enabled = true, vocab_tile = 1024)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)

@fused_lm_ce(enabled = true, vocab_tile = 128)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
"#;
    let res = analyze_source(src);
    // Both configs collected — item 3 lift.
    assert_eq!(
        res.fused_ce_configs.len(),
        2,
        "expected TWO collected configs (item 3 per-block dispatch), got {:?}",
        res.fused_ce_configs.len()
    );
    // Each carries a distinct train_block_stmt_id.
    assert_ne!(
        res.fused_ce_configs[0].train_block_stmt_id,
        res.fused_ce_configs[1].train_block_stmt_id,
        "two decorators on two different train blocks must carry distinct \
         `train_block_stmt_id` so codegen can route them independently",
    );
    // The dummy id (validator's placeholder) must NOT survive — the
    // push site is responsible for overwriting with the real Stmt.id.
    assert_ne!(
        res.fused_ce_configs[0].train_block_stmt_id,
        nsl_ast::NodeId::dummy(),
        "the push-site overwrite of `train_block_stmt_id` did not fire — \
         codegen would then match on `dummy()` and dispatch would silently \
         fail for BOTH blocks",
    );
    assert_ne!(
        res.fused_ce_configs[1].train_block_stmt_id,
        nsl_ast::NodeId::dummy(),
        "the push-site overwrite of `train_block_stmt_id` did not fire for \
         the second block",
    );
    // First config's vocab_tile matches source order; second config's does too.
    assert_eq!(res.fused_ce_configs[0].vocab_tile, Some(1024));
    assert_eq!(res.fused_ce_configs[1].vocab_tile, Some(128));
    // No refusal diagnostic (pre-v10 test would assert this fires; now
    // its absence pins the lift).
    assert!(
        !res.diagnostics.iter().any(|d| format!("{:?}", d)
            .contains("at most one decorator")),
        "unexpected duplicate-decorator refusal after item 3 lift: {:?}",
        res.diagnostics
    );
}

/// CFTP v10 (item 3): same-block duplicate `@fused_lm_ce` STILL refused.
/// The lift is per-train-block, not per-compilation-unit; two decorators
/// on the SAME train block would fight over the single dispatch slot
/// (`active_fused_ce_config` is 1:1 with the train block).  Refuse per
/// `feedback_deferral_must_refuse`.
#[test]
fn fused_lm_ce_same_block_duplicate_decorator_still_refused() {
    let src = r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])

@fused_lm_ce(enabled = true, vocab_tile = 1024)
@fused_lm_ce(enabled = true, vocab_tile = 128)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.fused_ce_configs.len(),
        1,
        "same-block duplicate: exactly one config survives (the first), got {:?}",
        res.fused_ce_configs.len()
    );
    assert!(
        res.diagnostics.iter().any(|d| format!("{:?}", d)
            .contains("at most one decorator per")),
        "expected same-block duplicate-decorator refusal diagnostic, got: {:?}",
        res.diagnostics
    );
}

// -----------------------------------------------------------------------
// WRGA decorator collection tests
// -----------------------------------------------------------------------

#[test]
fn wrga_decorator_is_captured() {
    let src = r#"
@wrga(mode=auto, budget=100000, target=rtx5070ti)
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.wrga_configs.len(),
        1,
        "expected one @wrga config, got {:?}",
        res.wrga_configs
    );
    let cfg = &res.wrga_configs[0];
    assert_eq!(cfg.block.budget, Some(100000));
    assert!(
        matches!(cfg.block.mode, nsl_ast::block::WrgaMode::Auto),
        "mode mismatch: {:?}",
        cfg.block.mode
    );
}

/// WRGA paper §8.2 — custom adapter DSL acceptance.  `@wrga(adapter=<Ident>)`
/// must parse, capture the symbol's resolved name on `WrgaConfig.adapter_name`,
/// and survive the post-pass validator when the named model exists with both a
/// Tensor field and a `forward` method.
#[test]
fn wrga_decorator_custom_adapter_parses_and_resolves_name() {
    let src = r#"
model GatedLoRA:
    a: Tensor = zeros([4, 2])
    b: Tensor = zeros([2, 4])
    gate: Tensor = ones([4])

    fn forward(self, x: Tensor) -> Tensor:
        return self.gate * (x @ self.a @ self.b)

@wrga(mode=auto, adapter=GatedLoRA, target=h100)
model Toy:
    w: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "no semantic errors expected, got: {:?}",
        res.diagnostics,
    );
    assert_eq!(res.wrga_configs.len(), 1);
    let cfg = &res.wrga_configs[0];
    assert!(cfg.block.adapter.is_some(), "block.adapter symbol must be set");
    assert_eq!(
        cfg.adapter_name.as_deref(),
        Some("GatedLoRA"),
        "adapter_name must carry the resolved string form",
    );
}

/// Typo in the adapter name must produce a clear error pointing at the
/// `@wrga(...)` site.  Defends against silently using the wrong (or no)
/// adapter at compile time.
#[test]
fn wrga_decorator_custom_adapter_undeclared_errors() {
    let src = r#"
@wrga(mode=auto, adapter=DoesNotExist)
model Toy:
    w: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    let errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(
        errors
            .iter()
            .any(|d| format!("{:?}", d).contains("DoesNotExist")
                && format!("{:?}", d).contains("not a declared model")),
        "expected an undeclared-adapter error mentioning 'DoesNotExist', got: {:?}",
        errors,
    );
}

/// A custom adapter with NO Tensor field is rejected — WRGA needs at least
/// one trainable parameter to place at each site.
#[test]
fn wrga_decorator_custom_adapter_missing_tensor_field_errors() {
    let src = r#"
model EmptyAdapter:
    fn forward(self, x: Tensor) -> Tensor:
        return x

@wrga(mode=auto, adapter=EmptyAdapter)
model Toy:
    w: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    let errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(
        errors
            .iter()
            .any(|d| format!("{:?}", d).contains("no `Tensor` field")),
        "expected a no-Tensor-field error, got: {:?}",
        errors,
    );
}

/// A custom adapter that omits `forward` is rejected — the rewrite has no
/// entry point to call into.
#[test]
fn wrga_decorator_custom_adapter_missing_forward_errors() {
    let src = r#"
model ForwardlessAdapter:
    a: Tensor = zeros([4, 4])

@wrga(mode=auto, adapter=ForwardlessAdapter)
model Toy:
    w: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    let errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(
        errors
            .iter()
            .any(|d| format!("{:?}", d).contains("no `forward` method")),
        "expected a no-forward-method error, got: {:?}",
        errors,
    );
}

/// A same-file `struct` named as `adapter=` is not a cross-module import: the
/// post-pass can see its full shape, and a `struct` can never declare a
/// `forward` method, so it can never satisfy the §8.2 adapter contract. It must
/// be rejected outright rather than trusted via the import fallback. (Before
/// this fix the struct fell into the cross-module trust path and was accepted.)
#[test]
fn wrga_decorator_custom_adapter_naming_a_same_file_struct_is_rejected() {
    let src = r#"
struct GatedLoRA:
    placeholder: int

@wrga(mode=auto, adapter=GatedLoRA)
model Toy:
    w: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    let wrong_kind_errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .filter(|d| format!("{:?}", d).contains("is a `struct`, not a `model`"))
        .collect();
    assert!(
        !wrong_kind_errors.is_empty(),
        "post-pass must reject a same-file struct named as `adapter=`; \
         got diagnostics: {:?}",
        res.diagnostics,
    );
}

/// A genuinely external adapter symbol — one that is in scope (e.g. a builtin,
/// standing in for `from foo.peft import GatedLoRA`) but has no declaration of
/// any kind in this file's `stmts` — must NOT trigger an error. The post-pass
/// falls back to a scope lookup and trusts the import; the deeper contract
/// check lands at codegen.
#[test]
fn wrga_decorator_custom_adapter_in_scope_via_non_local_symbol_is_accepted() {
    let src = r#"
@wrga(mode=auto, adapter=print)
model Toy:
    w: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics.iter().all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "no semantic errors expected for an in-scope external adapter symbol, got: {:?}",
        res.diagnostics,
    );
}

/// Sanity: an existing @wrga decorator without `adapter=` must continue to
/// parse with `adapter_name == None` (no behaviour regression).
#[test]
fn wrga_decorator_without_adapter_leaves_adapter_name_none() {
    let src = r#"
@wrga(mode=auto, budget=10000, target=h100)
model Toy:
    w: Tensor = zeros([4, 4])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w
"#;
    let res = analyze_source(src);
    assert_eq!(res.wrga_configs.len(), 1);
    let cfg = &res.wrga_configs[0];
    assert!(cfg.block.adapter.is_none());
    assert!(cfg.adapter_name.is_none());
}

#[test]
fn freeze_decorator_is_captured() {
    let src = r#"
@freeze(exclude=["blocks.6.*", "blocks.7.*"])
let coder = load_model()
"#;
    let res = analyze_source(src);
    assert_eq!(res.freeze_configs.len(), 1);
    let cfg = &res.freeze_configs[0];
    assert_eq!(cfg.exclude, vec!["blocks.6.*".to_string(), "blocks.7.*".to_string()]);
    assert!(cfg.include.is_empty());
}

#[test]
fn adapter_decorator_is_captured() {
    let src = r#"
@adapter(type=lora, target=["blocks.*.attn.q"], rank=8)
let coder = load_model()
"#;
    let res = analyze_source(src);
    assert_eq!(res.adapter_configs.len(), 1);
    let cfg = &res.adapter_configs[0];
    assert!(matches!(cfg.kind, crate::wrga::AdapterKind::Lora));
    assert_eq!(cfg.rank, Some(8));
    assert_eq!(cfg.targets, vec!["blocks.*.attn.q".to_string()]);
}

#[test]
fn adapter_rank_above_16_is_rejected() {
    // rank > 16 was previously accepted here and then SILENTLY dropped at
    // codegen (the adapter never fused). It must be a validation error.
    let src = r#"
@adapter(type=lora, target=["m.w"], rank=32)
let m = MyModel()
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("rank must be <= 16")),
        "expected rank-too-large error, got: {:?}",
        res.diagnostics
    );
}

#[test]
fn adapter_rank_exactly_16_is_accepted() {
    let src = r#"
@adapter(type=lora, target=["m.w"], rank=16)
let m = MyModel()
"#;
    let res = analyze_source(src);
    assert!(
        !res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("rank must be")),
        "rank=16 is the boundary and must be accepted, got: {:?}",
        res.diagnostics
    );
    assert_eq!(res.adapter_configs[0].rank, Some(16));
}

#[test]
fn adapter_alpha_is_captured() {
    let src = r#"
@adapter(type=lora, target=["m.w"], rank=4, alpha=8)
let m = MyModel()
"#;
    let res = analyze_source(src);
    assert_eq!(res.adapter_configs.len(), 1);
    let cfg = &res.adapter_configs[0];
    assert_eq!(cfg.rank, Some(4));
    assert_eq!(cfg.alpha, Some(8), "alpha must be captured from decorator");
}

#[test]
fn adapter_alpha_defaults_to_rank_when_absent() {
    let src = r#"
@adapter(type=lora, target=["m.w"], rank=4)
let m = MyModel()
"#;
    let res = analyze_source(src);
    let cfg = &res.adapter_configs[0];
    assert_eq!(cfg.alpha, None, "absent alpha stays None; codegen defaults to rank");
}

#[test]
fn wrga_hybrid_without_layers_errors() {
    let src = r#"
@wrga(mode=hybrid)
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("hybrid) requires a `layers=")),
        "expected hybrid-mode layer-scope error, got: {:?}",
        res.diagnostics
    );
}

#[test]
fn freeze_include_and_exclude_conflict_errors() {
    let src = r#"
@freeze(include=["a.*"], exclude=["b.*"])
let coder = load_model()
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("include` and `exclude` cannot be combined")),
        "expected freeze include/exclude conflict error, got: {:?}",
        res.diagnostics
    );
}

#[test]
fn adapter_missing_type_errors() {
    let src = r#"
@adapter(target=["blocks.*.attn.q"])
let coder = load_model()
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("requires a `type=")),
        "expected missing-type error, got: {:?}",
        res.diagnostics
    );
    assert!(res.adapter_configs.is_empty(), "invalid adapter should not be captured");
}

#[test]
fn freeze_on_non_model_let_binding_errors() {
    let src = r#"
@freeze(exclude=["a.*"])
let x: int = 42
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("@freeze can only be applied to a model")),
        "expected @freeze target error, got: {:?}",
        res.diagnostics
    );
}

#[test]
fn adapter_on_non_model_let_binding_errors() {
    let src = r#"
@adapter(type=lora, target=["blocks.*.attn.q"], rank=8)
let x: int = 42
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("@adapter can only be applied to a model")),
        "expected @adapter target error, got: {:?}",
        res.diagnostics
    );
}

// -----------------------------------------------------------------------
// @wggo_target placement validation tests (WGGO Phase 2 Task 1)
// -----------------------------------------------------------------------

#[test]
fn wggo_target_on_non_forward_method_errors() {
    // @wggo_target must be on the model's `forward` method. Placing it on
    // any other method (e.g. `cached_forward`) must emit a diagnostic
    // containing the exact phrase "@wggo_target must be on the model's
    // 'forward' method" so users can immediately locate the misplacement.
    //
    // The fixture intentionally uses a no-arg `@wggo_target` decorator to
    // isolate the placement rule (Task 1) from later argument-validation
    // tasks. Subsequent tasks (2/3/4) cover arg presence, types, etc.
    let src = r#"
model AttentionBlock:
    let weight: Tensor<[4, 4], f32>

    @wggo_target
    fn cached_forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let placement_errs: Vec<_> = diags
        .iter()
        .filter(|d| {
            d.message
                .contains("@wggo_target must be on the model's 'forward' method")
        })
        .collect();
    assert!(
        !placement_errs.is_empty(),
        "@wggo_target on a non-`forward` method should emit the placement diagnostic; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_on_forward_method_is_clean() {
    // The happy path: @wggo_target on `forward` should NOT emit the
    // placement diagnostic. (Other validation rules — arg presence, types,
    // etc. — may still fire but are out of scope for Task 1.)
    let src = r#"
model AttentionBlock:
    let weight: Tensor<[4, 4], f32>

    @wggo_target
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        !diags.iter().any(|d| d
            .message
            .contains("@wggo_target must be on the model's 'forward' method")),
        "@wggo_target on `forward` should not emit the placement diagnostic; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_on_layer_decl_errors() {
    // Covers the `ModelMember::LayerDecl` arm in `checker/model.rs`:
    // `@wggo_target` on a layer/field declaration (not a method) must emit
    // the "@wggo_target can only be applied to fn declarations" diagnostic.
    //
    // Field-decl form mirrors `examples/m30_shard_validation.nsl` —
    // `name: Type = init` with the decorator on the preceding line — which
    // is the syntax the parser actually accepts inside a `model:` body.
    let src = r#"
model AttentionBlock:
    @wggo_target
    weight: Tensor<[4, 4], f32> = zeros([4, 4])

    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let placement_errs: Vec<_> = diags
        .iter()
        .filter(|d| {
            d.message
                .contains("@wggo_target can only be applied to fn declarations")
        })
        .collect();
    assert!(
        !placement_errs.is_empty(),
        "@wggo_target on a layer declaration should emit the fn-only diagnostic; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_on_top_level_non_fn_errors() {
    // Covers the top-level `_ =>` arm in `StmtKind::Decorated` in
    // `checker/stmt.rs`: `@wggo_target` decorating any non-`fn` top-level
    // statement (here, a `model` decl) must emit the
    // "@wggo_target can only be applied to fn declarations" diagnostic.
    let src = r#"
@wggo_target
model AttentionBlock:
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let placement_errs: Vec<_> = diags
        .iter()
        .filter(|d| {
            d.message
                .contains("@wggo_target can only be applied to fn declarations")
        })
        .collect();
    assert!(
        !placement_errs.is_empty(),
        "@wggo_target on a top-level non-fn statement should emit the fn-only diagnostic; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_on_both_methods_fires_only_for_wrong_one() {
    // Edge case: a model with BOTH `@wggo_target fn forward` (clean) AND
    // `@wggo_target fn cached_forward` (wrong-name) verifies that the
    // iteration in `checker/model.rs` continues past the first match and
    // emits exactly one placement diagnostic — for `cached_forward` only.
    //
    // The exact-count assertion guards against regressions where the loop
    // either short-circuits on the happy method or double-reports.
    let src = r#"
model AttentionBlock:
    let q_proj: Tensor<[4, 4], f32>

    @wggo_target
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x

    @wggo_target
    fn cached_forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let placement_errs: Vec<_> = diags
        .iter()
        .filter(|d| {
            d.message
                .contains("@wggo_target must be on the model's 'forward' method")
        })
        .collect();
    assert_eq!(
        placement_errs.len(),
        1,
        "expected exactly one placement diagnostic (for `cached_forward`); got {} from diags: {:?}",
        placement_errs.len(),
        diags
    );
    // Sanity: the single firing must be the cached_forward one, not forward.
    assert!(
        placement_errs[0].message.contains("'cached_forward'"),
        "the single placement diagnostic must name 'cached_forward'; got: {:?}",
        placement_errs[0]
    );
}

// -----------------------------------------------------------------------
// WGGO Phase 2 Task 2: @wggo_target required-arguments validation
// -----------------------------------------------------------------------
// The @wggo_target decorator requires exactly five named arguments:
//   w_q, w_k, w_v, w_o, head_dim
// Missing any is a semantic-check error. These tests exercise the
// `ModelMember::Method` arm in `checker/model.rs` (the canonical case)
// since the standalone-fn arm in `checker/stmt.rs` always emits the
// placement diagnostic for the non-`forward` case but otherwise mirrors
// the same validate path.

#[test]
fn wggo_target_missing_head_dim_errors() {
    // Four of the five required args are present; `head_dim` is missing.
    // The diagnostic message must mention both the required-args header
    // and `head_dim` in the missing list so users know exactly what's
    // missing.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let arg_errs: Vec<_> = diags
        .iter()
        .filter(|d| {
            d.message.contains("@wggo_target requires arguments")
                && d.message.contains("head_dim")
        })
        .collect();
    assert!(
        !arg_errs.is_empty(),
        "@wggo_target with missing head_dim should emit the required-args diagnostic mentioning head_dim; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_missing_w_k_errors() {
    // Four of the five required args are present; `w_k` is missing.
    // Uses Tensor fields (matching Task 1's fixture form) for `q_proj`
    // and `head_dim` so the parser is happy. The exact field-type or
    // self.x reference shape isn't validated yet (Tasks 3/4); only the
    // arg *name* presence matters here.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_v=self.q_proj, w_o=self.q_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let arg_errs: Vec<_> = diags
        .iter()
        .filter(|d| d.message.contains("missing:") && d.message.contains("w_k"))
        .collect();
    assert!(
        !arg_errs.is_empty(),
        "@wggo_target with missing w_k should emit a diagnostic containing `missing:` and `w_k`; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_all_five_args_present_is_clean() {
    // Happy path: all five required args present. The required-args
    // diagnostic must NOT fire. Other validation rules (Tasks 3/4) may
    // still fire but are out of scope for Task 2.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        !diags
            .iter()
            .any(|d| d.message.contains("@wggo_target requires arguments")),
        "@wggo_target with all five args present should not emit the required-args diagnostic; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_missing_multiple_args_lists_all() {
    // Only `w_q` provided; the diagnostic must list all four missing
    // names so users see them in one shot rather than chasing one error
    // at a time.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let arg_errs: Vec<_> = diags
        .iter()
        .filter(|d| d.message.contains("@wggo_target requires arguments"))
        .collect();
    assert_eq!(
        arg_errs.len(),
        1,
        "expected exactly one required-args diagnostic; got: {:?}",
        diags
    );
    let msg = &arg_errs[0].message;
    for missing in ["w_k", "w_v", "w_o", "head_dim"] {
        assert!(
            msg.contains(missing),
            "required-args diagnostic must list missing arg '{}'; got: {}",
            missing,
            msg
        );
    }
    // `w_q` was provided so it must NOT appear in the missing list.
    // Look for the exact `"w_q"` quoted form (the message header
    // mentions w_q in the required-args list, but not in the missing
    // vector debug print).
    let after_missing = msg.split("missing:").nth(1).unwrap_or("");
    assert!(
        !after_missing.contains("\"w_q\""),
        "the missing list must not contain w_q (it was provided); got: {}",
        msg
    );
}

#[test]
fn wggo_target_no_args_lists_all_five_missing() {
    // No-arg `@wggo_target` (the Task 1 fixture form): all five required
    // args are missing. The diagnostic must list all five.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let arg_errs: Vec<_> = diags
        .iter()
        .filter(|d| d.message.contains("@wggo_target requires arguments"))
        .collect();
    assert_eq!(
        arg_errs.len(),
        1,
        "expected exactly one required-args diagnostic; got: {:?}",
        diags
    );
    let msg = &arg_errs[0].message;
    for missing in ["w_q", "w_k", "w_v", "w_o", "head_dim"] {
        assert!(
            msg.contains(missing),
            "required-args diagnostic must list missing arg '{}'; got: {}",
            missing,
            msg
        );
    }
}

// -----------------------------------------------------------------------
// WGGO Phase 2 Task 3: @wggo_target argument-expression validation
//
// Each of the five required arguments (w_q, w_k, w_v, w_o, head_dim) must
// be a `self.<field>` reference. Any other expression shape (literal,
// bare identifier, complex expression, etc.) emits a diagnostic of the
// form:
//   @wggo_target argument 'NAME' must be a self.<field> reference; got KIND
// -----------------------------------------------------------------------

#[test]
fn wggo_target_bare_ident_arg_errors() {
    // `w_q=q_proj` (bare identifier — missing `self.` prefix) must emit
    // the self.<field> diagnostic for w_q.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'w_q' must be a self.<field> reference")
        }),
        "bare-ident `w_q=q_proj` must emit the self.<field> diagnostic for w_q; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_int_literal_head_dim_errors() {
    // `head_dim=32` (integer literal) must emit the self.<field>
    // diagnostic for head_dim.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.q_proj, w_v=self.q_proj, w_o=self.q_proj, head_dim=32)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'head_dim' must be a self.<field> reference")
        }),
        "int-literal `head_dim=32` must emit the self.<field> diagnostic for head_dim; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_all_self_field_args_no_self_field_error() {
    // Happy path: all five args are `self.<field>` — the self.<field>
    // diagnostic must NOT fire. (Other diagnostics from later tasks may
    // fire, but not THIS specific error.)
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        !diags
            .iter()
            .any(|d| d.message.contains("must be a self.<field> reference")),
        "all-`self.<field>` args must not emit the self.<field> diagnostic; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_multiple_wrong_args_all_diagnose() {
    // Two wrong args (`w_q=q_proj` bare-ident and `head_dim=32` literal)
    // must both produce diagnostics — validation does not stop at the
    // first wrong arg.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=32)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'w_q' must be a self.<field> reference")
        }),
        "expected w_q self.<field> diagnostic; got: {:?}",
        diags
    );
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'head_dim' must be a self.<field> reference")
        }),
        "expected head_dim self.<field> diagnostic; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_bare_ident_summary_mentions_kind() {
    // The diagnostic message should include the expression-kind summary
    // after `got `. For a bare identifier this is "bare identifier".
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let msg = diags
        .iter()
        .find(|d| {
            d.message
                .contains("argument 'w_q' must be a self.<field> reference")
        })
        .map(|d| d.message.clone())
        .unwrap_or_default();
    assert!(
        msg.contains("got bare identifier"),
        "diagnostic should mention the kind summary `got bare identifier`; got: {}",
        msg
    );
}

#[test]
fn wggo_target_int_literal_summary_mentions_kind() {
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.q_proj, w_v=self.q_proj, w_o=self.q_proj, head_dim=32)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    let msg = diags
        .iter()
        .find(|d| {
            d.message
                .contains("argument 'head_dim' must be a self.<field> reference")
        })
        .map(|d| d.message.clone())
        .unwrap_or_default();
    assert!(
        msg.contains("got int literal"),
        "diagnostic should mention the kind summary `got int literal`; got: {}",
        msg
    );
}

// -------------------------------------------------------------------------
// WGGO Phase 2 Task 4: @wggo_target field-existence + field-type validation
//
// Each `@wggo_target` argument that passes Tasks 1-3 (placement,
// required-args, self.<field>) must reference a field that:
//   - exists in the enclosing model's `LayerDecl` member list, AND
//   - has the right type (Tensor for w_q/w_k/w_v/w_o; int for head_dim)
//
// Diagnostic shapes:
//   @wggo_target field reference 'self.<F>' not found in model '<M>'
//   @wggo_target argument '<ARG>' must reference a Tensor field; got <KIND>
//   @wggo_target argument 'head_dim' must reference an int field; got <KIND>
// -------------------------------------------------------------------------

#[test]
fn wggo_target_unknown_field_errors() {
    // `self.unknown_field` is referenced but no such field exists on
    // model `Attention`. The diagnostic must name both the missing field
    // and the model.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.unknown_field, w_k=self.q_proj, w_v=self.q_proj, w_o=self.q_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("'self.unknown_field' not found in model 'Attention'")
        }),
        "expected unknown-field diagnostic naming `self.unknown_field` and `Attention`; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_w_q_points_at_int_field_errors() {
    // `head_dim` is an `int` field. Pointing `w_q` (which must be Tensor)
    // at it should emit the wrong-field-type diagnostic with `got int`.
    let src = r#"
model Attention:
    head_dim: int = 4

    @wggo_target(w_q=self.head_dim, w_k=self.head_dim, w_v=self.head_dim, w_o=self.head_dim, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'w_q' must reference a Tensor field; got int")
        }),
        "expected `w_q must reference a Tensor field; got int`; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_head_dim_points_at_tensor_field_errors() {
    // `q_proj` is a Tensor field. Using it as `head_dim` (which must be
    // int) should emit the wrong-field-type diagnostic with `got Tensor`.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.q_proj, w_v=self.q_proj, w_o=self.q_proj, head_dim=self.q_proj)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'head_dim' must reference an int field; got Tensor")
        }),
        "expected `head_dim must reference an int field; got Tensor`; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_correctly_typed_fields_no_field_type_error() {
    // Happy path: all four projection args reference Tensor fields and
    // head_dim references an int field. None of the Task 4 diagnostics
    // should fire.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: int = 4

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        !diags.iter().any(|d| d.message.contains("not found in model")),
        "happy path must not emit `not found in model` diagnostics; got: {:?}",
        diags
    );
    assert!(
        !diags.iter().any(|d| d.message.contains("must reference a Tensor field")
            || d.message.contains("must reference an int field")),
        "happy path must not emit field-type diagnostics; got: {:?}",
        diags
    );
}

#[test]
fn wggo_target_multiple_wrong_fields_all_diagnose() {
    // Two errors: `w_q` points at an int field and `head_dim` points at a
    // Tensor field. Both diagnostics must fire — no short-circuit.
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim_int: int = 4

    @wggo_target(w_q=self.head_dim_int, w_k=self.q_proj, w_v=self.q_proj, w_o=self.q_proj, head_dim=self.q_proj)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'w_q' must reference a Tensor field; got int")
        }),
        "expected w_q-int diagnostic; got: {:?}",
        diags
    );
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("argument 'head_dim' must reference an int field; got Tensor")
        }),
        "expected head_dim-Tensor diagnostic; got: {:?}",
        diags
    );
}

// -------------------------------------------------------------------------
// Fast-follow coverage gaps from PR #132 review
// -------------------------------------------------------------------------

/// Task 2 M4: a top-level standalone `fn forward` decorated with
/// `@wggo_target(<missing args>)` exercises the `checker/stmt.rs` arm
/// directly (not the `checker/model.rs` arm). The placement check
/// passes because the function name IS `forward`, so the required-args
/// validator must still run and emit the missing-args diagnostic.
///
/// This complements `wggo_target_on_top_level_non_fn_errors` (which
/// only exercises the `_ =>` non-fn arm) and the model-method tests
/// (which exercise `model.rs`).
#[test]
fn wggo_target_top_level_forward_runs_required_args_validation() {
    let src = r#"
@wggo_target(w_q=self.q_proj)
fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let diags = check_source(src);
    let arg_errs: Vec<_> = diags
        .iter()
        .filter(|d| d.message.contains("@wggo_target requires arguments"))
        .collect();
    assert_eq!(
        arg_errs.len(),
        1,
        "stmt.rs arm must run required-args validation on top-level `fn forward` with @wggo_target; got: {:?}",
        diags
    );
    let msg = &arg_errs[0].message;
    for missing in ["w_k", "w_v", "w_o", "head_dim"] {
        assert!(
            msg.contains(missing),
            "missing-args diagnostic must list '{}'; got: {}",
            missing,
            msg
        );
    }
}

/// Task 3 fast-follow: `@wggo_target` arguments whose names are NOT in
/// `WGGO_TARGET_REQUIRED_ARGS` (e.g. an extra `foo=...`) must be
/// silently skipped by the self-field validator. They are neither
/// required (Task 2) nor in scope for the self.<field> rule (Task 3).
///
/// Concretely: passing `foo=42` (an int-literal that would otherwise
/// trigger the "must be a self.<field> reference" diagnostic if it were
/// in scope) MUST NOT emit that diagnostic for `foo`.
#[test]
fn wggo_target_unknown_arg_name_is_silently_skipped_by_self_field_check() {
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    @wggo_target(w_q=self.q_proj, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim, foo=42)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        !diags.iter().any(|d| {
            d.message
                .contains("argument 'foo' must be a self.<field> reference")
        }),
        "extra arg 'foo' must NOT trigger the self.<field> diagnostic; got: {:?}",
        diags
    );
}

/// Task 4 fast-follow: a `self.<field>` arg that resolves to a method on
/// the model (not a `LayerDecl`) must report `not found in model` —
/// methods are absent from the field-type map by design. This locks in
/// the contract that fields and methods live in disjoint namespaces for
/// `@wggo_target`.
#[test]
fn wggo_target_method_arg_falls_through_to_unknown_field() {
    let src = r#"
model Attention:
    q_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    k_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    v_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    o_proj: Tensor<[4, 4], f32> = zeros([4, 4])
    head_dim: Tensor<[4, 4], f32> = zeros([4, 4])

    fn helper(self) -> int:
        return 4

    @wggo_target(w_q=self.helper, w_k=self.k_proj, w_v=self.v_proj, w_o=self.o_proj, head_dim=self.head_dim)
    fn forward(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
    let diags = check_source(src);
    assert!(
        diags.iter().any(|d| {
            d.message
                .contains("'self.helper' not found in model 'Attention'")
        }),
        "method-typed self.<method> arg must surface as `not found in model` (methods are not in the field-type map); got: {:?}",
        diags
    );
}

// -----------------------------------------------------------------------
// CFTP §4.3 G2 Strategy 3 (Item 4): @pca decorator collection tests
// -----------------------------------------------------------------------

#[test]
fn pca_decorator_per_document_strategy_is_captured() {
    let src = r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])

@pca(strategy = per_document)
train(model = m, epochs = 5):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.pca_configs.len(),
        1,
        "expected one @pca config, got {:?}",
        res.pca_configs.len()
    );
    assert!(
        matches!(
            res.pca_configs[0].strategy,
            crate::cftp::PcaStrategy::PerDocument
        ),
        "strategy mismatch: {:?}",
        res.pca_configs[0].strategy
    );
}

#[test]
fn pca_decorator_default_strategy_is_auto() {
    let src = r#"
let m = Tiny()
@pca
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = 0
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.pca_configs.len(),
        1,
        "bare @pca must still produce a config (defaults to auto)"
    );
    assert!(
        matches!(
            res.pca_configs[0].strategy,
            crate::cftp::PcaStrategy::Auto
        ),
        "bare @pca should default to Auto, got {:?}",
        res.pca_configs[0].strategy
    );
}

#[test]
fn pca_decorator_invalid_strategy_errors_and_drops_config() {
    let src = r#"
let m = Tiny()
@pca(strategy = wibble)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = 0
"#;
    let res = analyze_source(src);
    assert!(
        res.diagnostics
            .iter()
            .any(|d| format!("{:?}", d).contains("@pca: strategy must be")),
        "expected invalid-strategy diagnostic, got: {:?}",
        res.diagnostics
    );
    // The validator still returns a config (default `Auto`) on an
    // invalid strategy — we just emit the diagnostic.  This preserves
    // partial-recovery semantics so downstream codegen can still
    // proceed with the safe (Auto) default.
    assert_eq!(res.pca_configs.len(), 1);
    assert!(matches!(
        res.pca_configs[0].strategy,
        crate::cftp::PcaStrategy::Auto
    ));
}

#[test]
fn pca_decorator_multiple_occurrences_are_collected() {
    let src = r#"
let m = Tiny()
@pca(strategy = per_document)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = 0

@pca(strategy = segment_id)
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = 0
"#;
    let res = analyze_source(src);
    assert_eq!(
        res.pca_configs.len(),
        2,
        "expected both @pca configs to be collected"
    );
}

#[test]
fn pca_decorator_absent_yields_empty_collection() {
    let src = r#"
let m = Tiny()
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.01)
    step(batch):
        let _ = 0
"#;
    let res = analyze_source(src);
    assert!(
        res.pca_configs.is_empty(),
        "absent @pca must leave the collection empty"
    );
}

// -----------------------------------------------------------------------
// Training Configuration Contract (train_config.rs): the train(...) key
// namespace is CLOSED at check time. Each test wraps one violation in an
// otherwise-valid program; the valid-control test pins that the closed
// set is not over-broad (the cpkd guard-test pattern).
// -----------------------------------------------------------------------

/// A minimal valid program with `{header}` spliced as the train header.
fn train_fixture(header: &str) -> String {
    format!(
        r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])
let y = zeros([4, 1])

{header}
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = m.forward(x)
"#
    )
}

fn train_config_errors(header: &str) -> Vec<String> {
    check_source(&train_fixture(header))
        .into_iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message)
        .collect()
}

#[test]
fn train_unknown_key_is_refused_at_check_time() {
    // The motivating typo: epochz=8 used to type-check clean and lower
    // with epochs' DEFAULT — a silently different training run.
    let errs = train_config_errors("train(model = m, epochz = 8):");
    assert!(
        errs.iter().any(|m| m.contains("unknown train config key 'epochz'")),
        "expected unknown-key refusal, got {errs:?}"
    );
}

#[test]
fn train_duplicate_key_is_refused() {
    let errs = train_config_errors("train(model = m, epochs = 2, epochs = 3):");
    assert!(
        errs.iter().any(|m| m.contains("duplicate train config key 'epochs'")),
        "expected duplicate-key refusal, got {errs:?}"
    );
}

#[test]
fn train_positional_config_arg_is_refused() {
    // `train(m, 5):` used to parse and silently ignore BOTH args (every
    // consumer guarded `if let Some(name) = arg.name`).
    let errs = train_config_errors("train(m, 5):");
    assert!(
        errs.iter().any(|m| m.contains("named arguments only")),
        "expected positional-arg refusal, got {errs:?}"
    );
}

#[test]
fn train_epochs_zero_and_negative_are_refused() {
    let errs = train_config_errors("train(model = m, epochs = 0):");
    assert!(
        errs.iter().any(|m| m.contains("'epochs' must be >= 1")),
        "epochs=0 used to silently train zero epochs; got {errs:?}"
    );
    let errs = train_config_errors("train(model = m, epochs = -3):");
    assert!(
        errs.iter().any(|m| m.contains("'epochs' must be >= 1")),
        "negative epochs must refuse, got {errs:?}"
    );
}

#[test]
fn train_non_literal_epochs_is_refused_at_check_time() {
    // Previously refused only at CODEGEN ("nsl check" was green) — the
    // layer asymmetry cpkd_distill_e2e documents. Both layers agree now.
    let errs = train_config_errors("train(model = m, epochs = n_ep):");
    assert!(
        errs.iter().any(|m| m.contains("'epochs' must be an integer literal")),
        "expected non-literal epochs refusal, got {errs:?}"
    );
}

#[test]
fn train_non_literal_grad_accumulation_is_refused_not_clamped() {
    // The old behavior silently clamped to 1 (accumulation OFF while the
    // source asked for it) — recorded as GradAccumulationDecl::NonLiteral
    // but never surfaced.
    let errs = train_config_errors("train(model = m, grad_accumulation = ga):");
    assert!(
        errs.iter()
            .any(|m| m.contains("'grad_accumulation' must be a positive integer literal")),
        "expected non-literal grad_accumulation refusal, got {errs:?}"
    );
    let errs = train_config_errors("train(model = m, grad_accumulation = 0):");
    assert!(
        errs.iter()
            .any(|m| m.contains("'grad_accumulation' must be a positive integer literal")),
        "grad_accumulation=0 used to silently clamp to 1, got {errs:?}"
    );
}

#[test]
fn train_grad_clip_zero_negative_and_non_literal_are_refused() {
    // grad_clip=0 zeroes every gradient each step (training silently
    // frozen); negative flips gradient signs; a non-literal used to be the
    // one key with a FULLY silent bad-value path (no else arm at all).
    let errs = train_config_errors("train(model = m, grad_clip = 0.0):");
    assert!(
        errs.iter()
            .any(|m| m.contains("'grad_clip' must be a positive finite threshold")),
        "grad_clip=0 must refuse, got {errs:?}"
    );
    let errs = train_config_errors("train(model = m, grad_clip = -1.0):");
    assert!(
        errs.iter()
            .any(|m| m.contains("'grad_clip' must be a positive finite threshold")),
        "negative grad_clip must refuse, got {errs:?}"
    );
    let errs = train_config_errors("train(model = m, grad_clip = gc):");
    assert!(
        errs.iter().any(|m| m.contains("'grad_clip' must be a numeric literal")),
        "non-literal grad_clip must refuse, got {errs:?}"
    );
}

#[test]
fn train_model_non_identifier_is_refused() {
    let errs = train_config_errors("train(model = 5):");
    assert!(
        errs.iter().any(|m| m.contains("'model' arg must be an identifier")),
        "model=5 must refuse with the identifier message, got {errs:?}"
    );
}

#[test]
fn train_checkpoint_value_shapes_are_refused() {
    // Each refusal arm exercised so none can be silently deleted (review
    // finding: the value arms had zero negative coverage).
    let errs = train_config_errors(
        r#"train(model = m, checkpoint_save = "", checkpoint_every = 5):"#,
    );
    assert!(
        errs.iter().any(|m| m.contains("'checkpoint_save' arg must be a non-empty")),
        "empty save path must refuse, got {errs:?}"
    );
    let errs = train_config_errors(
        r#"train(model = m, checkpoint_save = "ck.nslm", checkpoint_every = 0):"#,
    );
    assert!(
        errs.iter().any(|m| m.contains("'checkpoint_every' must be a positive")),
        "checkpoint_every=0 must refuse, got {errs:?}"
    );
    // The range error must be the ONLY pairing-adjacent diagnostic: the
    // cadence key IS present, so the "requires checkpoint_every" pairing
    // message would be factually false here.
    assert!(
        !errs.iter().any(|m| m.contains("requires checkpoint_every")),
        "a present-but-invalid cadence must not be reported as missing, got {errs:?}"
    );
    let errs = train_config_errors(
        r#"train(model = m, checkpoint_save = "ck.nslm", checkpoint_every = n):"#,
    );
    assert!(
        errs.iter().any(|m| m.contains("'checkpoint_every' must be a positive")),
        "non-literal checkpoint_every must refuse, got {errs:?}"
    );
    let errs = train_config_errors("train(model = m, checkpoint_load = 5):");
    assert!(
        errs.iter().any(|m| m.contains("'checkpoint_load' arg must be a non-empty")),
        "non-string checkpoint_load must refuse, got {errs:?}"
    );
}

#[test]
fn train_parenthesized_and_infinite_values_fold_and_refuse_correctly() {
    // (2) folds through Paren — an improvement over the old codegen match,
    // which refused any parenthesized literal outright.
    let errs = train_config_errors("train(model = m, epochs = (2)):");
    assert!(
        !errs.iter().any(|m| m.contains("train '")),
        "parenthesized literal epochs must fold clean, got {errs:?}"
    );
    // An infinite threshold collides with codegen's f64::MAX no-clipping
    // sentinel — accepted-but-inert, so it refuses.
    let errs = train_config_errors("train(model = m, grad_clip = 1e400):");
    assert!(
        errs.iter()
            .any(|m| m.contains("'grad_clip' must be a positive finite threshold")),
        "an infinite grad_clip must refuse, got {errs:?}"
    );
}

#[test]
fn train_checkpoint_pairing_is_enforced_both_directions() {
    let errs =
        train_config_errors(r#"train(model = m, checkpoint_save = "ck.nslm"):"#);
    assert!(
        errs.iter().any(|m| m.contains("requires checkpoint_every")),
        "save without cadence must refuse, got {errs:?}"
    );
    let errs = train_config_errors("train(model = m, checkpoint_every = 5):");
    assert!(
        errs.iter().any(|m| m.contains("without checkpoint_save= is inert")),
        "cadence without save used to be silently inert, got {errs:?}"
    );
}

#[test]
fn train_missing_model_is_refused_at_check_time() {
    // Previously only caught at codegen; `nsl check` was green.
    let errs = train_config_errors("train(epochs = 2):");
    assert!(
        errs.iter().any(|m| m.contains("requires a 'model' config arg")),
        "expected missing-model refusal, got {errs:?}"
    );
}

#[test]
fn train_valid_full_header_produces_no_config_errors() {
    // Anti-overbreadth control: every accepted key, all valid — the
    // contract must not fire (the cpkd guard-test pattern: pin the error's
    // ABSENCE on valid programs).
    let errs = train_config_errors(
        r#"train(model = m, epochs = 2, grad_accumulation = 4, grad_clip = 1.0, checkpoint_save = "ck.nslm", checkpoint_every = 5, checkpoint_load = "ck.nslm"):"#,
    );
    let config_errs: Vec<_> = errs
        .iter()
        .filter(|m| {
            // Every contract message shape, including the missing-model one
            // ("train block requires a 'model' config arg"), which matches
            // neither of the first two patterns (review finding: the control
            // was blind to a missing-model false positive).
            m.contains("train config")
                || m.contains("train '")
                || m.contains("train block requires")
        })
        .collect();
    assert!(
        config_errs.is_empty(),
        "valid header must produce no contract errors, got {config_errs:?}"
    );
}

/// A minimal distill program with `{header}` spliced as the distill header.
fn distill_fixture(header: &str) -> String {
    format!(
        r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let t = Tiny()
let s = Tiny()
let x = ones([4, 2])

{header}
    optimizer: SGD(lr = 0.01)
    step(batch):
        let pred = s.forward(x)
"#
    )
}

#[test]
fn distill_epochs_zero_is_refused_like_train() {
    // epochs rides the typed DistillAsTrain field, never the train
    // resolver — without the checker's own >= 1 arm, distill(epochs = 0)
    // silently trained the student ZERO epochs while the identical train
    // header refuses (review finding on 7e3aa7fa).
    let errs: Vec<String> = check_source(&distill_fixture(
        "distill(teacher = t, student = s, epochs = 0):",
    ))
    .into_iter()
    .filter(|d| matches!(d.level, nsl_errors::Level::Error))
    .map(|d| d.message)
    .collect();
    assert!(
        errs.iter().any(|m| m.contains("distill 'epochs' must be >= 1")),
        "distill epochs=0 must refuse, got {errs:?}"
    );
}

#[test]
fn distill_duplicate_config_key_is_refused_at_check_time() {
    // Without this, a duplicated key passed `nsl check` and refused only at
    // build time via the synthesized TrainBlock — with a message naming
    // 'train', a construct the program does not contain.
    let errs: Vec<String> = check_source(&distill_fixture(
        "distill(teacher = t, student = s, epochs = 1, grad_accumulation = 2, grad_accumulation = 4):",
    ))
    .into_iter()
    .filter(|d| matches!(d.level, nsl_errors::Level::Error))
    .map(|d| d.message)
    .collect();
    assert!(
        errs.iter()
            .any(|m| m.contains("duplicate distill config key 'grad_accumulation'")),
        "expected the distill-named duplicate refusal, got {errs:?}"
    );
}

#[test]
fn train_model_written_but_invalid_is_not_reported_as_missing() {
    // Presence is judged from the key WRITTEN: `model = 5` gets the
    // identifier error and must NOT also get "requires a 'model' config
    // arg" — a factually false second diagnostic.
    let errs = train_config_errors("train(model = 5):");
    assert!(
        !errs.iter().any(|m| m.contains("requires a 'model' config arg")),
        "a written-but-invalid model must not be reported as missing, got {errs:?}"
    );
}

// -----------------------------------------------------------------------
// Optimizer/scheduler/callbacks section contract (optim_config.rs): the
// section namespaces are CLOSED at check time — per-optimizer kwarg
// tables, per-scheduler kwarg tables, callback names. Same shape as the
// header contract above: one violation per test, plus an anti-overbreadth
// valid control.
// -----------------------------------------------------------------------

/// A minimal valid program with `{sections}` spliced as the train
/// sections (4-space indented; a step section is appended).
fn train_sections_fixture(sections: &str) -> String {
    format!(
        r#"
model Tiny:
    w: Tensor = ones([2, 1])

    fn forward(self, x: Tensor) -> Tensor:
        return x @ self.w

let m = Tiny()
let x = ones([4, 2])

train(model = m):
{sections}
    step(batch):
        let pred = m.forward(x)
"#
    )
}

fn section_contract_errors(sections: &str) -> Vec<String> {
    check_source(&train_sections_fixture(sections))
        .into_iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message)
        .collect()
}

#[test]
fn optimizer_typo_kwarg_is_refused_at_check_time() {
    // The motivating typo: AdamW(lrr=0.01) used to hit codegen's
    // `_ => {}` and silently train at the default lr=0.01.
    let errs = section_contract_errors("    optimizer: AdamW(lrr = 0.01)");
    assert!(
        errs.iter().any(|m| m.contains("unknown AdamW kwarg 'lrr'")),
        "expected unknown-kwarg refusal, got {errs:?}"
    );
}

#[test]
fn optimizer_inapplicable_kwarg_is_refused() {
    // beta2 parsed fine on SGD and was simply never read by the emitted
    // sgd_step call — accepted-but-ignored, the defect class the contract
    // exists to prevent. The kwarg tables are the stdlib step signatures.
    let errs = section_contract_errors("    optimizer: SGD(lr = 0.01, beta2 = 0.5)");
    assert!(
        errs.iter().any(|m| m.contains("unknown SGD kwarg 'beta2'")),
        "expected inapplicable-kwarg refusal, got {errs:?}"
    );
    // soap_step takes no weight_decay parameter at all.
    let errs = section_contract_errors("    optimizer: SOAP(lr = 0.01, weight_decay = 0.1)");
    assert!(
        errs.iter().any(|m| m.contains("unknown SOAP kwarg 'weight_decay'")),
        "expected SOAP weight_decay refusal, got {errs:?}"
    );
}

#[test]
fn optimizer_duplicate_kwarg_is_refused() {
    // Previously last-wins (each arm reassigned its local as the loop ran).
    let errs = section_contract_errors("    optimizer: AdamW(lr = 0.01, lr = 0.02)");
    assert!(
        errs.iter().any(|m| m.contains("duplicate optimizer kwarg 'lr'")),
        "expected duplicate-kwarg refusal, got {errs:?}"
    );
}

#[test]
fn optimizer_positional_arg_is_refused() {
    // Previously skipped before any name matched: SGD(0.05) compiled and
    // trained at the default lr.
    let errs = section_contract_errors("    optimizer: SGD(0.05)");
    assert!(
        errs.iter().any(|m| m.contains("optimizer takes named arguments only")),
        "expected positional-arg refusal, got {errs:?}"
    );
}

#[test]
fn optimizer_non_literal_value_is_refused_not_defaulted() {
    // The silent half of the defect: a KNOWN key whose value shape didn't
    // match fell through its own arm — `lr = base_lr` kept lr=0.01.
    let errs = section_contract_errors("    optimizer: AdamW(lr = base_lr)");
    assert!(
        errs.iter().any(|m| m.contains("optimizer 'lr' must be a numeric literal")),
        "expected non-literal value refusal, got {errs:?}"
    );
}

#[test]
fn optimizer_negative_lr_folds_and_refuses_with_range_message() {
    // -0.001 parses as UnaryOp(Neg, FloatLiteral) — pre-contract it was
    // silently dropped (lr stayed 0.01); the resolver folds it and gives
    // the range message, not a bogus "not a literal".
    let errs = section_contract_errors("    optimizer: AdamW(lr = -0.001)");
    assert!(
        errs.iter().any(|m| m.contains("optimizer 'lr' must be a non-negative")
            && m.contains("-0.001")),
        "expected folded range refusal, got {errs:?}"
    );
}

#[test]
fn optimizer_beta_out_of_range_is_refused() {
    // beta = 1.0 divides by zero in bias correction (1 - beta^t) at t=1.
    let errs = section_contract_errors("    optimizer: AdamW(lr = 0.01, beta1 = 1.0)");
    assert!(
        errs.iter().any(|m| m.contains("optimizer 'beta1' must be a numeric literal in [0, 1)")),
        "expected beta range refusal, got {errs:?}"
    );
}

#[test]
fn unknown_optimizer_name_is_refused_at_check_time() {
    // Previously refused only deep in codegen (step-fn mangling); `nsl
    // check` was green.
    let errs = section_contract_errors("    optimizer: RMSProp(lr = 0.01)");
    assert!(
        errs.iter().any(|m| m.contains("unknown optimizer 'RMSProp'")),
        "expected unknown-name refusal, got {errs:?}"
    );
}

#[test]
fn bare_ident_optimizer_is_refused_as_not_a_call() {
    // `optimizer: AdamW` (no parens) passed `nsl check` and then refused
    // at build with the misleading "requires an optimizer section".
    let errs = section_contract_errors("    optimizer: AdamW");
    assert!(
        errs.iter().any(|m| m.contains("optimizer must be a constructor call")),
        "expected constructor-call refusal, got {errs:?}"
    );
}

#[test]
fn missing_optimizer_section_is_refused_at_check_time() {
    // Previously codegen-only.
    let errs = section_contract_errors("    data:\n        source = 1");
    assert!(
        errs.iter().any(|m| m.contains("train block requires an optimizer section")),
        "expected missing-section refusal, got {errs:?}"
    );
}

#[test]
fn duplicate_optimizer_sections_are_refused() {
    // Pre-contract the consumers disagreed: the lowering took the LAST
    // section, CPDT and the training report took the FIRST.
    let errs = section_contract_errors(
        "    optimizer: SGD(lr = 0.01)\n    optimizer: AdamW(lr = 0.001)",
    );
    assert!(
        errs.iter().any(|m| m.contains("duplicate optimizer: section")),
        "expected duplicate-section refusal, got {errs:?}"
    );
}

#[test]
fn ns_steps_refusals_fire_at_check_time_with_the_pinned_wording() {
    // The muon_p1_gate wording contract, now surfaced by `nsl check` too.
    for (val, expect) in [
        ("0", "ns_steps must be a positive integer"),
        ("-3", "ns_steps must be a positive integer"),
        ("3.5", "cannot be fractional"),
    ] {
        let errs = section_contract_errors(&format!(
            "    optimizer: Muon(lr = 0.01, ns_steps = {val})"
        ));
        assert!(
            errs.iter().any(|m| m.contains(expect)),
            "ns_steps={val}: expected '{expect}', got {errs:?}"
        );
    }
}

#[test]
fn no_decay_unknown_role_is_refused_at_check_time() {
    let errs = section_contract_errors(
        r#"    optimizer: AdamW(lr = 0.01, no_decay = ["vectors"])"#,
    );
    assert!(
        errs.iter().any(|m| m.contains("unknown parameter role")),
        "expected unknown-role refusal, got {errs:?}"
    );
}

#[test]
fn unknown_scheduler_name_is_refused_at_check_time() {
    // Previously fell through the lowering's name map to `_ =>
    // base_lr_val // fallback: no change` — compiled clean and silently
    // trained at constant lr. (spec/05 advertised `Cosine`, which was one
    // of the silent victims.)
    let errs = section_contract_errors(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: Cosine(t_max = 100)",
    );
    assert!(
        errs.iter().any(|m| m.contains("unknown scheduler 'Cosine'")),
        "expected unknown-scheduler refusal, got {errs:?}"
    );
}

#[test]
fn scheduler_typo_kwarg_is_refused_not_defaulted() {
    // A typo'd kwarg was pushed into scheduler_args, never matched by any
    // `.find()`, and the hardcoded default silently applied.
    let errs = section_contract_errors(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: warmup_cosine(warmup_step = 5)",
    );
    assert!(
        errs.iter().any(|m| m.contains("unknown warmup_cosine kwarg 'warmup_step'")),
        "expected scheduler typo refusal, got {errs:?}"
    );
}

#[test]
fn scheduler_non_literal_value_is_refused() {
    let errs = section_contract_errors(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: warmup_cosine(total_steps = ts)",
    );
    assert!(
        errs.iter().any(|m| m.contains("scheduler 'total_steps' must be a numeric literal")),
        "expected scheduler non-literal refusal, got {errs:?}"
    );
}

#[test]
fn scheduler_duplicate_kwarg_and_duplicate_section_are_refused() {
    // Duplicate kwargs were FIRST-wins in the scheduler (Vec + .find) but
    // last-wins in the optimizer — the divergence is refused away.
    let errs = section_contract_errors(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: step_lr(gamma = 0.1, gamma = 0.2)",
    );
    assert!(
        errs.iter().any(|m| m.contains("duplicate scheduler kwarg 'gamma'")),
        "expected duplicate-kwarg refusal, got {errs:?}"
    );
    let errs = section_contract_errors(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: constant_lr()\n    scheduler: step_lr()",
    );
    assert!(
        errs.iter().any(|m| m.contains("duplicate scheduler: section")),
        "expected duplicate-section refusal, got {errs:?}"
    );
}

#[test]
fn unknown_callback_name_is_refused_at_check_time() {
    // Previously collected and silently never emitted — the dispatch only
    // knows on_step/on_epoch/on_epoch_end.
    let errs = section_contract_errors(concat!(
        "    optimizer: AdamW(lr = 0.001)\n",
        "    callbacks:\n",
        "        on_stpe(step, loss):\n",
        "            let s = step\n",
    ));
    assert!(
        errs.iter().any(|m| m.contains("unknown callback 'on_stpe'")),
        "expected unknown-callback refusal, got {errs:?}"
    );
}

#[test]
fn valid_full_sections_produce_no_contract_errors() {
    // Anti-overbreadth control: a fully-loaded valid program — every
    // AdamW kwarg, int-literal scheduler steps (every committed scheduler
    // call passes ints; the stdlib types them float), warmup > total (the
    // coder-rl 12-step repro shape), and both callback names. The filter
    // covers every contract message shape (audited against
    // optim_config.rs: "optimizer", "scheduler", "callback", "kwarg",
    // "no_decay", "ns_steps", "adamw_lr").
    let errs = section_contract_errors(concat!(
        "    optimizer: AdamW(lr = 0.0003, weight_decay = 0.1, beta1 = 0.9, ",
        "beta2 = 0.95, eps = 1e-8, no_decay = [\"vector\"])\n",
        "    scheduler: warmup_cosine(warmup_steps = 20, total_steps = 12, min_lr = 0.00003)\n",
        "    callbacks:\n",
        "        on_step(step, loss):\n",
        "            let s = step\n",
        "        on_epoch(epoch, loss):\n",
        "            let e = epoch\n",
    ));
    let contract_errs: Vec<_> = errs
        .iter()
        .filter(|m| {
            m.contains("optimizer")
                || m.contains("scheduler")
                || m.contains("callback")
                || m.contains("kwarg")
                || m.contains("no_decay")
                || m.contains("ns_steps")
                || m.contains("adamw_lr")
        })
        .collect();
    assert!(
        contract_errs.is_empty(),
        "valid sections must produce no contract errors, got {contract_errs:?}"
    );
}

#[test]
fn muon_camelcase_scheduler_spelling_is_accepted() {
    // WarmupCosine (CamelCase) lowercases to the collapsed spelling the
    // canonical map accepts — the pre-contract acceptance is preserved.
    let errs = section_contract_errors(
        "    optimizer: AdamW(lr = 0.001)\n    scheduler: WarmupCosine(warmup_steps = 10, total_steps = 100)",
    );
    assert!(
        !errs.iter().any(|m| m.contains("unknown scheduler")),
        "WarmupCosine must stay accepted, got {errs:?}"
    );
}

#[test]
fn distill_sections_get_the_same_contract_at_check_time() {
    // Distill sections travel verbatim into the synthesized TrainBlock —
    // without the distill-side call a typo'd optimizer kwarg passed
    // `nsl check` and refused only at build.
    let errs: Vec<String> = check_source(&distill_fixture(
        "distill(teacher = t, student = s, epochs = 1):",
    ))
    .into_iter()
    .filter(|d| matches!(d.level, nsl_errors::Level::Error))
    .map(|d| d.message)
    .collect();
    // The shared fixture's SGD(lr = 0.01) is valid — control.
    assert!(
        !errs.iter().any(|m| m.contains("kwarg") || m.contains("unknown optimizer")),
        "valid distill optimizer must not refuse, got {errs:?}"
    );

    let src = distill_fixture("distill(teacher = t, student = s, epochs = 1):")
        .replace("optimizer: SGD(lr = 0.01)", "optimizer: SGD(lrr = 0.01)");
    let errs: Vec<String> = check_source(&src)
        .into_iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message)
        .collect();
    assert!(
        errs.iter().any(|m| m.contains("unknown SGD kwarg 'lrr'")),
        "distill optimizer typo must refuse at check time, got {errs:?}"
    );
}

#[test]
fn typo_callback_param_is_refused_not_bound_to_zero() {
    // Codegen binds callback params BY NAME with a silent zero fallback:
    // on_step(step, lss) type-checked clean and logged a constant 0
    // forever (review finding). The param namespace is closed per
    // callback: on_step provides step/loss, on_epoch* provide epoch/loss.
    let errs = section_contract_errors(concat!(
        "    optimizer: AdamW(lr = 0.001)\n",
        "    callbacks:\n",
        "        on_step(step, lss):\n",
        "            let s = step\n",
    ));
    assert!(
        errs.iter().any(|m| m.contains("unknown on_step callback parameter 'lss'")),
        "expected the param refusal, got {errs:?}"
    );
    // Cross-context names refuse too: on_epoch provides epoch, not step.
    let errs = section_contract_errors(concat!(
        "    optimizer: AdamW(lr = 0.001)\n",
        "    callbacks:\n",
        "        on_epoch(step, loss):\n",
        "            let s = step\n",
    ));
    assert!(
        errs.iter().any(|m| m.contains("unknown on_epoch callback parameter 'step'")),
        "expected the cross-context param refusal, got {errs:?}"
    );
    // Subsets stay legal: on_step(step) is a committed corpus shape.
    let errs = section_contract_errors(concat!(
        "    optimizer: AdamW(lr = 0.001)\n",
        "    callbacks:\n",
        "        on_step(step):\n",
        "            let s = step\n",
    ));
    assert!(
        !errs.iter().any(|m| m.contains("callback parameter")),
        "on_step(step) must stay legal, got {errs:?}"
    );
}

#[test]
fn duplicate_callback_definition_is_refused() {
    // Codegen compiles EVERY matching definition — a user redefining
    // on_step (expecting override) silently got both bodies run per step
    // (review finding). Same contract as duplicate optimizer:/scheduler:.
    let errs = section_contract_errors(concat!(
        "    optimizer: AdamW(lr = 0.001)\n",
        "    callbacks:\n",
        "        on_step(step, loss):\n",
        "            let a = step\n",
        "        on_step(step, loss):\n",
        "            let b = step\n",
    ));
    assert!(
        errs.iter().any(|m| m.contains("duplicate callback 'on_step'")),
        "expected the duplicate-callback refusal, got {errs:?}"
    );
    // Distinct callbacks across two callbacks: sections stay legal.
    let errs = section_contract_errors(concat!(
        "    optimizer: AdamW(lr = 0.001)\n",
        "    callbacks:\n",
        "        on_step(step, loss):\n",
        "            let a = step\n",
        "    callbacks:\n",
        "        on_epoch(epoch, loss):\n",
        "            let b = epoch\n",
    ));
    assert!(
        !errs.iter().any(|m| m.contains("duplicate callback")),
        "distinct callbacks across sections must stay legal, got {errs:?}"
    );
}
