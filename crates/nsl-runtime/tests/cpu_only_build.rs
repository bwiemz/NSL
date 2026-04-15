//! Compile-time guard: if this file builds (i.e. `cargo test -p nsl-runtime`
//! compiles the test binary without the `cuda` feature), then the runtime's
//! calibration-facing public API is CPU-only-buildable.
//!
//! A regression that re-introduces unconditional CUDA linkage in the runtime's
//! calibration-facing path would fail the compile of this test binary.
//!
//! Symbol inventory (calibration binary imports):
//!   - `nsl_calibration_load`      — no CUDA dependency
//!   - `nsl_calibration_batch_at`  — no CUDA dependency
//!   - `nsl_calibration_count`     — no CUDA dependency
//!   - `nsl_calibration_free`      — no CUDA dependency
//!   - `nsl_awq_write_sidecar`     — no CUDA dependency
//!   - `nsl_model_load`            — GPU branch behind `#[cfg(feature = "cuda")]`
//!   - `nsl_write_file`            — no CUDA dependency
//!
//! Note: the task spec refers to `nsl_checkpoint_load`; the actual runtime
//! export for checkpoint loading is `nsl_model_load` (in checkpoint.rs).

#[test]
fn calibration_api_compiles_without_cuda_feature() {
    // Exercise the key symbols the calibration binary links against.
    // All must be reachable without the `cuda` feature being enabled.
    let _ = nsl_runtime::nsl_calibration_load as *const ();
    let _ = nsl_runtime::nsl_calibration_batch_at as *const ();
    let _ = nsl_runtime::nsl_calibration_count as *const ();
    let _ = nsl_runtime::nsl_calibration_free as *const ();
    let _ = nsl_runtime::nsl_awq_write_sidecar as *const ();
    // Checkpoint loader: the calibration binary calls nsl_model_load (not a
    // GPU-init entry point — GPU branch is behind #[cfg(feature = "cuda")]).
    let _ = nsl_runtime::checkpoint::nsl_model_load as *const ();
    // File I/O: used by the calibration binary to write the sidecar JSON.
    let _ = nsl_runtime::file_io::nsl_write_file as *const ();
}
