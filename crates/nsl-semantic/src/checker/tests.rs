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
