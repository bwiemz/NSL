//! Smoke test: FASE Deferred mode should compile a train block with
//! grad_accumulation=4 and AdamW once Tasks 8 + 9 land.  Until then this
//! test is expected to FAIL (test is `#[ignore]`d to keep CI green).
//!
//! Flip `#[ignore]` to a plain `#[test]` after Task 9 lands the stmt.rs
//! dispatch that wires `Compiler::fase_emit_accumulate` into the micro-batch
//! accumulation loop.

use nsl_codegen::CompileOptions;
use std::path::PathBuf;

fn fixture(name: &str) -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p.push(name);
    std::fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("failed to read fixture {:?}: {}", p, e))
}

#[test]
#[ignore = "enabled after Task 9 lands the stmt.rs dispatch"]
fn fase_deferred_compiles_adamw_grad_accum_4() {
    let src = fixture("fase_deferred_grad_accum_4.nsl");
    let opts = CompileOptions {
        source_ad: true,
        ..Default::default()
    };
    nsl_codegen::debug_compile_and_return_plan(&src, &opts)
        .expect("FASE deferred train block with grad_accumulation=4 must compile without error");
}
