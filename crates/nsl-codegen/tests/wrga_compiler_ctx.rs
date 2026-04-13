//! Task 3: WrgaInputs reach the Compiler context.
//!
//! Compile a minimal valid NSL module with `options.wrga_inputs = Some(...)`
//! and verify no panic/error; a later task asserts the plan runs.

use nsl_codegen::{CompileOptions, FreezeDecoratorConfig, WrgaInputs};

const TRIVIAL_SRC: &str = "fn main():\n    let _x = 0\n";

#[test]
fn wrga_inputs_reach_compiler_without_crashing() {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _) = nsl_lexer::tokenize(TRIVIAL_SRC, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    assert!(
        analysis
            .diagnostics
            .iter()
            .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
        "baseline must type-check clean: {:?}",
        analysis.diagnostics
    );

    let opts = CompileOptions {
        wrga_inputs: Some(WrgaInputs {
            freeze: vec![FreezeDecoratorConfig {
                exclude: vec!["blocks.6.*".into()],
                include: vec![],
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    let _obj = nsl_codegen::compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        &opts,
    )
    .expect("compile should succeed");
}
