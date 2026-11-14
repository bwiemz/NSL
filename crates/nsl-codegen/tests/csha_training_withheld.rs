//! A train step may not go through the CSHA fused attention pair.
//!
//! With `--csha`, a train block whose attention sits in an
//! `@flash_attention` method, directly behind an RMSNorm and three
//! projections, used to lower to the fused forward that saves activations
//! and the fused backward. Run on a GPU, that pair gives a forward output
//! that is right over gradients that are not: dQ, dK and dV are 43% to 130%
//! off the f64 reference beyond one tile, more than one head stops on a
//! shape mismatch, and one head with one tile has not been shown right.
//! The lowering now refuses with a compile error that names the decorator
//! and the flag.
//!
//! This file replaces `csha_gap_f_toy_pretrain_smoke.rs`, which compiled the
//! same program and asserted that the fused backward's symbol was referenced
//! by a relocation: it pinned the path being reachable and nothing about
//! what the path computes.
//!
//! What is checked, all at compile time, no GPU (that the same program
//! compiles and trains correctly without `--csha` is checked through the CLI
//! by `crates/nsl-cli/tests/csha_toy_pretrain_gate.rs`):
//!
//! * the refusal fires for a train block on the claimed chain, and its
//!   message names `@flash_attention`, `--csha` and the way out;
//! * a forward-only program (the same method, no train block) compiles with
//!   `--csha`: inference through the decorated method is not refused.

#![cfg(feature = "test-helpers")]

use nsl_codegen::CompileOptions;
use nsl_errors::{FileId, Level};
use nsl_lexer::Interner;
use object::{Object, ObjectSection, ObjectSymbol};
use std::collections::{HashMap, HashSet};

/// RMSNorm, three projections named wq / wk / wv, attention: the chain the
/// CSHA boundary scan claims.
const MODEL: &str = r#"
model TinyAttn:
    w_norm: Tensor = ones([32])
    wq: Tensor = ones([32, 32])
    wk: Tensor = ones([32, 32])
    wv: Tensor = ones([32, 32])

    @flash_attention(head_dim=32)
    fn forward(self, x: Tensor) -> Tensor:
        let x_norm = rmsnorm(x, self.w_norm, 0.00001)
        let q = x_norm @ self.wq
        let k = x_norm @ self.wk
        let v = x_norm @ self.wv
        let scale = 1.0 / sqrt(32.0)
        return scaled_dot_product_attention(q, k, v, scale)

let m = TinyAttn()
let x = ones([1, 1, 32, 32])
"#;

const TRAIN: &str = r#"
train(model = m, epochs = 1):
    optimizer: SGD(lr = 0.001)
    step(batch):
        let out = m.forward(x)
        let loss = sum(out)
"#;

const FORWARD_ONLY: &str = r#"
let y: Tensor = m.forward(x)
print(sum(y))
"#;

/// Compile `src` to an object. `Err` carries the compile error's message.
fn compile(src: &str, csha: bool) -> Result<Vec<u8>, String> {
    let mut interner = Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
    let errors = |d: &[nsl_errors::Diagnostic]| -> Vec<String> {
        d.iter().filter(|d| matches!(d.level, Level::Error)).map(|d| d.message.clone()).collect()
    };
    assert!(errors(&lex_diags).is_empty(), "lex errors: {:?}", errors(&lex_diags));
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(errors(&parsed.diagnostics).is_empty(), "parse errors: {:?}", errors(&parsed.diagnostics));
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(errors(&analysis.diagnostics).is_empty(), "semantic errors: {:?}", errors(&analysis.diagnostics));
    let opts = CompileOptions {
        csha: nsl_codegen::CshaOptions {
            mode: csha.then(|| "auto".to_string()),
            ..Default::default()
        },
        source_ad: true,
        target: "sm_75".to_string(),
        ..Default::default()
    };
    let imported_fns = nsl_codegen::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        &opts,
    )
    .unwrap_or_else(|e| panic!("stdlib loader: {}", e.message));
    nsl_codegen::compile_entry(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &imported_fns,
        HashMap::new(),
        HashSet::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        false,
        &opts,
    )
    .map_err(|e| e.message)
}

/// The names referenced by at least one relocation in the object. Membership
/// in the symbol table alone does not show a call was emitted.
fn called_symbols(obj_bytes: &[u8]) -> HashSet<String> {
    let file = object::File::parse(obj_bytes).expect("object::File::parse");
    let name_by_index: HashMap<_, _> = file
        .symbols()
        .filter_map(|s| {
            let n = s.name().ok()?;
            Some((s.index(), nsl_codegen::linker::strip_host_symbol_prefix(n).to_string()))
        })
        .collect();
    let mut out = HashSet::new();
    for section in file.sections() {
        for (_offset, reloc) in section.relocations() {
            if let object::RelocationTarget::Symbol(idx) = reloc.target()
                && let Some(name) = name_by_index.get(&idx)
            {
                out.insert(name.clone());
            }
        }
    }
    out
}

#[test]
fn a_train_step_through_csha_claimed_attention_is_a_compile_error() {
    let err = match compile(&format!("{MODEL}{TRAIN}"), true) {
        Ok(_) => panic!("a train block on a CSHA-claimed attention compiled"),
        Err(e) => e,
    };
    for needle in ["@flash_attention", "--csha", "withheld", "do not match the reference", "without `--csha`"] {
        assert!(err.contains(needle), "the error does not say {needle:?}:\n{err}");
    }
}

#[test]
fn a_forward_only_use_of_the_decorated_method_compiles_with_csha() {
    let obj = compile(&format!("{MODEL}{FORWARD_ONLY}"), true).expect("forward-only compile with --csha");
    let called = called_symbols(&obj);
    assert!(
        called.iter().any(|s| s.starts_with("nsl_flash_attention")),
        "the decorated method launched no attention kernel: {called:?}"
    );
    assert!(
        !called.contains("nsl_flash_attention_csha_backward"),
        "a program with no train block references the fused backward"
    );
}
