//! TDD tests for `pca_activation::detect_packing_for_stmts`.
//!
//! These tests constitute the acceptance criteria for Task A-1 of the
//! PCA Tier A Activation plan: a single shared helper that is the sole
//! source of truth for "is this module's training data packed?" (spec §2.4).

use nsl_codegen::pca_activation::detect_packing_for_stmts;
use nsl_errors::FileId;
use nsl_lexer::{tokenize, Interner};

fn parse(src: &str) -> (nsl_ast::Module, Interner) {
    let mut interner = Interner::new();
    let (tokens, _lex_diags) = tokenize(src, FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    (parsed.module, interner)
}

/// A dataset with `packing = true` referenced by a train block → true.
///
/// Correct NSL dataset syntax: `dataset Name("source_path"):` with an
/// indented key-value body (parser-verified in nsl-parser block.rs:1098).
#[test]
fn detects_packing_true() {
    let src = "dataset PretrainCorpus(\"/data/pile\"):\n    packing = true\n    max_sequence_length = 2048\n    mean_doc_length = 400\n\ntrain(model = m, grad_accumulation = 4):\n    data:\n        source = PretrainCorpus\n    optimizer: AdamW\n";
    let (module, interner) = parse(src);
    assert!(
        detect_packing_for_stmts(&module.stmts, &interner),
        "expected detect_packing_for_stmts to return true for a dataset with packing=true"
    );
}

/// A dataset without the `packing` key (absent) and a train block → false.
#[test]
fn no_packing_when_flag_absent_or_false() {
    let src = "dataset PretrainCorpus(\"/data/pile\"):\n    max_sequence_length = 2048\n\ntrain(model = m, grad_accumulation = 4):\n    data:\n        source = PretrainCorpus\n    optimizer: AdamW\n";
    let (module, interner) = parse(src);
    assert!(
        !detect_packing_for_stmts(&module.stmts, &interner),
        "expected detect_packing_for_stmts to return false when packing key is absent"
    );
}

/// A dataset with `packing = true` but NO train block → false.
/// Conservative: only returns true when a train block explicitly references
/// a packing-enabled dataset (spec §2.4 single-source invariant).
#[test]
fn no_packing_when_no_train_block() {
    let src = "dataset PretrainCorpus(\"/data/pile\"):\n    packing = true\n    max_sequence_length = 2048\n    mean_doc_length = 400\n";
    let (module, interner) = parse(src);
    assert!(
        !detect_packing_for_stmts(&module.stmts, &interner),
        "expected detect_packing_for_stmts to return false when there is no train block"
    );
}
