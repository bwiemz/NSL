//! Per M57.1 §3.2, `nsl build --target fpga <non-model.nsl>` returns a
//! structured redirect error pointing the user at `nsl fpga-compile`.
//!
//! This test pins the production-arm message text via the shared
//! `FPGA_TARGET_REDIRECT_MSG` const. Changing the const without updating
//! both the test substrings and the user-visible message coherently will
//! surface here.

use nsl_codegen::error::CodegenError;
use nsl_codegen::FPGA_TARGET_REDIRECT_MSG;

#[test]
fn fpga_target_redirect_msg_names_rejected_target_and_redirect_destination() {
    // Format check on the shared const that the production GpuTarget::Fpga
    // arm at compiler/kernel.rs returns. Asserting on FPGA_TARGET_REDIRECT_MSG
    // directly (rather than building a fresh CodegenError) ensures the test
    // tracks the production message verbatim.
    assert!(
        FPGA_TARGET_REDIRECT_MSG.contains("--target fpga"),
        "redirect message should name the rejected target: {FPGA_TARGET_REDIRECT_MSG}"
    );
    assert!(
        FPGA_TARGET_REDIRECT_MSG.contains("fpga-compile"),
        "redirect message should redirect to fpga-compile: {FPGA_TARGET_REDIRECT_MSG}"
    );
    assert!(
        FPGA_TARGET_REDIRECT_MSG.contains("model-block"),
        "redirect message should explain the scope: {FPGA_TARGET_REDIRECT_MSG}"
    );
}

#[test]
fn fpga_target_redirect_msg_surfaces_in_codegen_error() {
    // Cross-check: when wrapped in CodegenError (as the production arm does),
    // the Display output preserves the message.
    let err = CodegenError::new(FPGA_TARGET_REDIRECT_MSG);
    let msg = err.to_string();
    assert!(msg.contains("--target fpga"));
    assert!(msg.contains("fpga-compile"));
    assert!(msg.contains("model-block"));
}
