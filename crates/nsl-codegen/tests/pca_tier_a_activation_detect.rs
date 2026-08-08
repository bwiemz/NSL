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

/// A dataset with `packing = true` but NO training block → false.
/// Conservative: only returns true when a training block explicitly references
/// a packing-enabled dataset (spec §2.4 single-source invariant).
#[test]
fn no_packing_when_no_train_block() {
    let src = "dataset PretrainCorpus(\"/data/pile\"):\n    packing = true\n    max_sequence_length = 2048\n    mean_doc_length = 400\n";
    let (module, interner) = parse(src);
    assert!(
        !detect_packing_for_stmts(&module.stmts, &interner),
        "expected detect_packing_for_stmts to return false when there is no training block"
    );
}

// ─── distill blocks are training blocks ─────────────────────────────────────
//
// `compiler::kernel`'s PCA Tier-A synthesis admits a distill block via
// `stmts_contain_train_block` (whose `DistillBlock` arm says "the CSHA
// training-PTX synthesis must fire for it exactly as for `train`") and then,
// twelve lines later, asks THIS function whether the corpus is packed. While
// the walk here matched `TrainBlock` alone the two answers contradicted each
// other, and the losing side was the mask: a packed distill corpus got
// UNMASKED training kernels, so tokens attended across document boundaries
// with nothing on stderr. These two arms differ in one token — `distill` vs
// `train` — which is the whole point.

fn distill_src(packing: &str) -> String {
    format!(
        "dataset PretrainCorpus(\"/data/pile\"):\n    packing = {packing}\n    \
         max_sequence_length = 2048\n    mean_doc_length = 400\n\n\
         distill(teacher = t, student = s, epochs = 2, grad_accumulation = 4):\n    \
         data:\n        source = PretrainCorpus\n    optimizer: AdamW\n"
    )
}

#[test]
fn a_distill_block_on_a_packed_corpus_selects_packing() {
    let (module, interner) = parse(&distill_src("true"));
    assert!(
        detect_packing_for_stmts(&module.stmts, &interner),
        "a distill block IS a training loop: the packed corpus it names must \
         select the segment-masked kernels, exactly as under `train`"
    );
    assert!(
        nsl_codegen::pca_activation::resolve_packing_config_for_stmts(
            &module.stmts,
            &interner
        )
        .is_some(),
        "the per-document CTA admission resolves from the same walk; if it \
         disagreed with the mask decision above, one of the two would be \
         planning for a corpus shape the other does not believe in"
    );
}

/// The control that makes the arm above mean something: the SAME distill
/// block, one token different in the dataset, must select nothing. Without
/// it, an arm that returned `true` unconditionally would pass.
#[test]
fn the_same_distill_block_selects_nothing_when_the_corpus_is_unpacked() {
    let (module, interner) = parse(&distill_src("false"));
    assert!(
        !detect_packing_for_stmts(&module.stmts, &interner),
        "packing = false must not select the masked kernels"
    );
    assert!(
        nsl_codegen::pca_activation::resolve_packing_config_for_stmts(
            &module.stmts,
            &interner
        )
        .is_none(),
        "packing = false must not admit per-document CTAs"
    );
}

/// The mask decision and the training report must not disagree about the same
/// program. The report renders a PCA section from `extract_dataset_ref` on the
/// presented block; the kernel path calls `detect_packing_for_stmts`. Pinning
/// them equal is what stops a future edit fixing one surface and leaving the
/// other — the exact shape of the defect these tests were added for.
#[test]
fn the_training_report_and_the_kernel_path_agree_about_a_distill_block() {
    for packing in ["true", "false"] {
        let (module, interner) = parse(&distill_src(packing));
        let detected = detect_packing_for_stmts(&module.stmts, &interner);
        let report = nsl_codegen::training_report::build_report(
            &module,
            &interner,
            std::path::Path::new("mem.nsl"),
        );
        assert_eq!(
            report.train_blocks.len(),
            1,
            "the report must see the distill block at all (packing = {packing})"
        );
        assert_eq!(
            report.train_blocks[0].pca.is_some(),
            detected,
            "report PCA section and kernel-path packing detection disagree \
             for packing = {packing}"
        );
    }
}
