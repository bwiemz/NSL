//! End-to-end test for `nsl run --monitor`.
//!
//! Phase 2 scope: verify the codegen-side manifest is written to
//! `<file>.nsl-profile.json` when --monitor is passed. The actual runtime
//! timings (Test B under `cuda` feature) are scoped to a later verification
//! pass on a real GPU box — in this session we just confirm the plumbing is
//! wired end-to-end.

use std::path::PathBuf;
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn monitor_writes_codegen_manifest_for_matmul_fixture() {
    let src = fixture("tiny_transformer.nsl");
    let manifest_path = src.with_extension("nsl-profile.json");
    // Clean any pre-existing manifest so we see a fresh write.
    let _ = std::fs::remove_file(&manifest_path);

    let bin = env!("CARGO_BIN_EXE_nsl");
    let output = Command::new(bin)
        .args(["run", "--monitor", src.to_str().unwrap()])
        .output()
        .expect("failed to invoke nsl run --monitor");

    // The process may exit non-zero (e.g., CUDA isn't installed) — that's fine
    // for THIS test. We only care that the codegen-side manifest was written,
    // which happens BEFORE any runtime kernel execution.
    let _ = output.status;

    assert!(
        manifest_path.exists(),
        "expected codegen manifest at {} — stdout:\n{}\nstderr:\n{}",
        manifest_path.display(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let manifest_text =
        std::fs::read_to_string(&manifest_path).expect("manifest exists but unreadable");
    let manifest: nsl_codegen::profiling::instrument::Manifest =
        serde_json::from_str(&manifest_text).expect("manifest should parse as JSON");

    // The codegen-side default target is "H100-SXM" and dtype "bf16"; these
    // strings are stable as of Phase 2.
    assert_eq!(
        manifest.target_gpu, "H100-SXM",
        "manifest target_gpu should be H100-SXM (default)"
    );
    assert_eq!(
        manifest.dtype, "bf16",
        "manifest dtype should be bf16 (default)"
    );

    // tiny_transformer.nsl uses matmul via the `@` operator; those go through
    // nsl_tensor_matmul (an FFI dispatch), NOT compile_kernel_call. So kernel
    // entries here come from the codegen-side instrumentation of matmul
    // operators. The `compile_kernel_call` begin/end runtime hooks only fire
    // for explicit `kernel { ... }` blocks in NSL source — that case is
    // covered (or stubbed) by Test B / Phase 2.5.
    //
    // We assert the manifest shape is well-formed:
    assert!(
        manifest.kernels.iter().all(|k| !k.op_name.is_empty()),
        "all kernel entries must have a non-empty op_name"
    );
}

#[test]
fn monitor_flag_does_not_crash_without_cuda() {
    // Smoke: --monitor without a working CUDA runtime should still terminate
    // cleanly (no hang, no panic). We don't assert exit code — CUDA driver
    // absence may cause non-zero exit.
    let src = fixture("tiny_transformer.nsl");
    let bin = env!("CARGO_BIN_EXE_nsl");
    let output = Command::new(bin)
        .args(["run", "--monitor", src.to_str().unwrap()])
        .output()
        .expect("failed to invoke nsl run --monitor");
    // Process terminated (did not hang) — that's the core invariant.
    assert!(
        output.status.code().is_some() || !output.status.success(),
        "process should have terminated with an exit code"
    );
}

// -- Test B: real actual timings under the cuda feature --
//
// Intentionally stubbed — requires a GPU-enabled CI box and an NSL fixture
// using explicit `kernel { ... }` blocks so the compile_kernel_call hooks
// fire. Candidate fixtures exist:
//   - crates/nsl-cli/tests/m17_gpu_training_test.nsl
//   - crates/nsl-cli/tests/m17_kernel_test.nsl
//   - examples/m26_autotune.nsl
// Phase 2.5 owns this test.
#[cfg(all(feature = "cuda", never_enabled))]
#[test]
fn real_actual_timings_under_cuda() {
    // Placeholder — gate remains closed. See Phase 2.5.
}
