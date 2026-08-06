//! Static wiring pins for the pass manager: the mechanism only enforces
//! anything if (1) every Compiler anchors an epoch and (2) the train-block
//! wrapper actually consults the decision. Both are one-line calls a
//! refactor could drop without any test noticing — the enforcement would
//! silently degrade to "never fires" (epoch 0 for everything, or a decision
//! nobody asks for), which is the declared-but-inert defect class this
//! repo has now shipped five times. These scans make the wiring load-bearing.

use std::path::{Path, PathBuf};

fn src(rel: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Strip `//` comments then all whitespace — the multi-line-call-chain
/// lesson: a line-based scan misses a call split across lines and certifies
/// what it cannot see.
fn code_only(text: &str) -> String {
    text.lines()
        .map(|l| match l.find("//") {
            Some(i) if !l[..i].contains('"') => &l[..i],
            _ => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .split_whitespace()
        .collect()
}

#[test]
fn every_compiler_anchors_a_compile_epoch() {
    let code = code_only(&src("src/compiler/mod.rs"));
    assert!(
        code.contains("passes:crate::pass_manager::PassManager::begin()"),
        "Compiler's construction no longer anchors a compile epoch — the \
         per-compile view is empty and enforcement can never fire. If the \
         construction moved, update this scan with it."
    );
}

#[test]
fn the_train_block_wrapper_consults_the_ordering_decision() {
    let code = code_only(&src("src/stmt.rs"));
    assert!(
        code.contains(".enforce_dependency_order()"),
        "compile_train_block no longer consults \
         PassManager::enforce_dependency_order — the ordering decision is \
         computed for nobody. If the enforcement point moved, update this \
         scan with it."
    );
}

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/cpdt_precision_fp16_loop.nsl")
}

/// The loop-bound fixture is the live witness for WGGO's TrainBlock phase
/// declaration (the prepass structurally cannot plan it, so WGGO's first
/// record lands in the in-place site). The declaration comment points here.
#[test]
fn the_loop_bound_fixture_exists_and_keeps_its_shape() {
    let s = std::fs::read_to_string(fixture()).unwrap_or_else(|e| {
        panic!("missing {}: {e}", fixture().display());
    });
    assert!(
        s.contains("for member in ens.members:") && s.contains("train(model = member"),
        "the loop fixture no longer binds its train model in a loop — it no \
         longer witnesses the TrainBlock phase declaration"
    );
}
