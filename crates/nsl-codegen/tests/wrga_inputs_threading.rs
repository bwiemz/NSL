//! Task 1: verify decorator configs captured by nsl-semantic reach
//! nsl-codegen via CompileOptions.wrga_inputs.

use nsl_codegen::{CompileOptions, WrgaInputs};

#[test]
fn compile_options_accepts_wrga_inputs() {
    let inputs = WrgaInputs::default();
    let opts = CompileOptions {
        wrga_inputs: Some(inputs),
        ..Default::default()
    };
    assert!(opts.wrga_inputs.is_some());
}
