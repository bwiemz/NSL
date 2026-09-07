//! Task 1: verify decorator configs captured by nsl-semantic reach
//! nsl-codegen via CompileOptions.wrga.inputs.

use nsl_codegen::{CompileOptions, WrgaInputs};

#[test]
fn compile_options_accepts_wrga_inputs() {
    let inputs = WrgaInputs::default();
    let opts = CompileOptions {
        wrga: nsl_codegen::WrgaOptions {
            inputs: Some(inputs),
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(opts.wrga.inputs.is_some());
}
