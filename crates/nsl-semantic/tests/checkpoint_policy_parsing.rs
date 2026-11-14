//! Cycle-10 §5.3 Task 3: `@checkpoint(policy="...")` kwarg parsing,
//! R0 deprecation warning (Path B per T3), R4 message update, and the
//! `EffectChecker::checkpoint_policies` side-table.

use nsl_errors::FileId;
use nsl_lexer::Interner;
use nsl_semantic::effects::CheckpointPolicy;

fn analyze_source(src: &str) -> nsl_semantic::AnalysisResult {
    let mut interner = Interner::new();
    let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    nsl_semantic::analyze(&parse_result.module, &mut interner)
}

#[test]
fn checkpoint_policy_full_is_recorded() {
    let src = r#"
@pure
@checkpoint(policy="full")
fn block(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
"#;
    let res = analyze_source(src);
    // No errors at semantic time.
    let errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    assert_eq!(
        res.checkpoint_policies.get("block").copied(),
        Some(CheckpointPolicy::Full),
        "policy=\"full\" must populate the side-table"
    );
}

#[test]
fn checkpoint_policy_none_is_tape_fallback() {
    let src = r#"
@pure
@checkpoint(policy="none")
fn block(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
"#;
    let res = analyze_source(src);
    let errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(errors.is_empty(), "unexpected errors: {errors:?}");
    assert!(
        !res.checkpoint_policies.contains_key("block"),
        "policy=\"none\" must NOT populate the codegen-side policy map"
    );
}

/// The error messages of a source, for the refusal tests below.
fn error_messages(src: &str) -> Vec<String> {
    analyze_source(src)
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .map(|d| d.message.clone())
        .collect()
}

#[test]
fn checkpoint_policy_selective_is_refused() {
    // The selective recompute policy belongs to a train block
    // (`--checkpoint-selective`). Accepted on a function, it switched the
    // policy on for every train block of the program, called or not.
    let src = r#"
@pure
@checkpoint(policy="selective")
fn block(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
"#;
    let res = analyze_source(src);
    let errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(
        errors.iter().any(|d| d.message.contains("policy=\"selective\"")
            && d.message.contains("not supported")
            && d.message.contains("--checkpoint-selective")),
        "selective must be refused and point at the flag; got: {errors:?}"
    );
    assert!(
        !res.checkpoint_policies.contains_key("block"),
        "a refused policy must not reach the codegen-side policy map"
    );
}

#[test]
fn checkpoint_on_a_model_is_refused() {
    // Accepted before with any arguments, and ignored.
    for args in ["(budget=0.5)", "(policy=\"bogus_value\")", "(policy=\"full\")", ""] {
        let src = format!(
            r#"
@checkpoint{args}
model M:
    w: Tensor<[4, 4], f32> = ones([4, 4])

    fn forward(self, x: Tensor<[4, 4], f32>) -> Tensor<[4, 4], f32>:
        return x @ self.w
"#
        );
        let errors = error_messages(&src);
        assert!(
            errors
                .iter()
                .any(|m| m.contains("@checkpoint is not supported on a model")),
            "@checkpoint{args} on a model must be refused; got: {errors:?}"
        );
    }
}

#[test]
fn checkpoint_on_a_model_method_is_refused() {
    for args in ["(budget=0.5)", "(policy=\"bogus_value\")", "(policy=\"full\")", ""] {
        let src = format!(
            r#"
model M:
    w: Tensor<[4, 4], f32> = ones([4, 4])

    @checkpoint{args}
    fn forward(self, x: Tensor<[4, 4], f32>) -> Tensor<[4, 4], f32>:
        return x @ self.w
"#
        );
        let errors = error_messages(&src);
        assert!(
            errors
                .iter()
                .any(|m| m.contains("@checkpoint is not supported on a model method")),
            "@checkpoint{args} on a method must be refused; got: {errors:?}"
        );
    }
}

#[test]
fn checkpoint_policy_custom_is_refused() {
    let src = r#"
@pure
@checkpoint(policy="custom")
fn block(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
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
            .any(|d| d.message.contains("reserved for §5.3-v2/v3")),
        "expected v2/v3 refusal; got: {errors:?}"
    );
}

#[test]
fn checkpoint_policy_bogus_lists_valid_choices() {
    let src = r#"
@pure
@checkpoint(policy="bogus")
fn block(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
"#;
    let res = analyze_source(src);
    let errors: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Error))
        .collect();
    assert!(
        errors.iter().any(|d| d.message.contains("must be one of")
            && d.message.contains("\"full\"")
            && d.message.contains("\"none\"")),
        "expected valid-list diagnostic; got: {errors:?}"
    );
}

#[test]
fn bare_checkpoint_emits_one_deprecation_warning_per_file() {
    // Two bare @checkpoint sites in the same file → ONE warning.
    let src = r#"
@pure
@checkpoint
fn block_a(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x

@pure
@checkpoint
fn block_b(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
"#;
    let res = analyze_source(src);
    let warnings: Vec<_> = res
        .diagnostics
        .iter()
        .filter(|d| matches!(d.level, nsl_errors::Level::Warning))
        .filter(|d| d.message.contains("deprecated"))
        .collect();
    assert_eq!(
        warnings.len(),
        1,
        "expected exactly one deprecation warning per source file; got {warnings:?}"
    );
    // Both functions still register as @checkpoint for R4 purposes
    // (they appear via the M14 tape fallback path).
    assert!(
        !res.checkpoint_policies.contains_key("block_a"),
        "bare @checkpoint must NOT carry a CheckpointPolicy"
    );
    assert!(
        !res.checkpoint_policies.contains_key("block_b"),
        "bare @checkpoint must NOT carry a CheckpointPolicy"
    );
}
