//! Stub implementations of interop FFI symbols.
//!
//! When the `interop` feature is disabled, the codegen still declares these
//! as external imports. These stubs satisfy the linker and panic with a
//! clear message if called at runtime.

fn interop_panic() -> ! {
    eprintln!("error: this NSL program uses interop features (safetensors/HuggingFace/ONNX)");
    eprintln!("but the runtime was built without the 'interop' feature.");
    eprintln!("Rebuild with: cargo build -p nsl-runtime --features interop");
    std::process::exit(1);
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_safetensors_load(_a: i64, _b: i64, _c: i64) -> i64 {
    interop_panic();
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_safetensors_save(_a: i64, _b: i64, _c: i64) {
    interop_panic();
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_hf_load(_a: i64, _b: i64, _c: i64, _d: i64, _e: i64, _f: i64) -> i64 {
    interop_panic();
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_trace_start() {
    interop_panic();
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_trace_register_input(_a: i64, _b: i64) {
    interop_panic();
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_trace_register_output(_a: i64, _b: i64) {
    interop_panic();
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_trace_stop() -> i64 {
    interop_panic();
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_onnx_export(_a: i64, _b: i64, _c: i64) {
    interop_panic();
}
