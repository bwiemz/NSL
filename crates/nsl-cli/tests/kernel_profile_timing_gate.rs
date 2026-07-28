//! `--profile-kernels` must produce REAL timings, not a file full of zeros.
//!
//! # The bug this pins
//!
//! `kernel_profile.json` reported every kernel as `"dur": 0.0`, every
//! timestamp as `"ts": 0.0`, and `total_kernel_time_ms: 0.0` — on a run that
//! launched 412 kernels and took tens of milliseconds of GPU time. It had
//! been that way long enough that a previous campaign attributed kernel cost
//! by launch count and bytes moved instead, because the durations were known
//! to be useless.
//!
//! Two independent defects produced it:
//!
//! 1. **`atexit` ordering.** Handlers run LIFO. The profiler registered its
//!    flush from `nsl_kernel_profiler_start`, which runs inside
//!    `nsl_args_init` — before any kernel launch, therefore before the CUDA
//!    context exists, therefore before the driver registers its own
//!    primary-context teardown. LIFO then ran the driver's teardown FIRST,
//!    and the flush met a dead context. (The same hazard is already
//!    documented for the cuBLAS handle, which `cuda/mod.rs` leaks on purpose
//!    because destroying it after context teardown "produces spurious driver
//!    errors".) The registration now happens from
//!    `ensure_event_pool_initialized`, which runs from inside `kernel_launch`
//!    after `ensure_context()`.
//!
//! 2. **A discarded status.** `cu_event_elapsed_time` threw away its
//!    `CUresult`, and `cuEventElapsedTime_v2` leaves its out-param UNTOUCHED
//!    on failure. So every failed query became a 0.0 that is
//!    indistinguishable from a measurement. It now returns the status, and
//!    the flush counts failures, refuses in stderr, and stamps
//!    `timing_valid: false` in the JSON.
//!
//! # Why this must be an e2e test
//!
//! The failure only exists at process exit. `matmul_batch_collapse` and
//! `matmul_transposed_operand` both drive the profiler in-process
//! (start/stop/flush inline) and were fully green throughout — they never
//! touch the atexit path. Only a real compiled program that exits normally
//! reproduces it.

use std::path::PathBuf;
use std::process::{Command, Stdio};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// Run the CUDA-graph fixture (a pure-tensor residual MLP, device-resident
/// before the train block) under `--profile-kernels` and return the parsed
/// profile plus the run's stderr.
fn run_profiled() -> (String, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_kprof_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Same fixture and rewrite marker the cuda-graph gate uses: no embedding,
    // no DataLoader, no norms — every training input device-resident before
    // the train block, so the run is a dense stream of real kernels.
    let mut src = std::fs::read_to_string(
        root.join("crates/nsl-cli/tests/fixtures/cuda_graph_gate.nsl"),
    )
    .unwrap();
    const MARKER: &str = "# GPU_PLACEMENT";
    assert!(src.contains(MARKER), "rewrite marker missing in fixture");
    src = src.replace(MARKER, "m.to(cuda)\nlet xg = x.to(cuda)\nlet yg = y.to(cuda)");
    src = src
        .replace("m.forward_train(x)", "m.forward_train(xg)")
        .replace("l1_loss(pred, y)", "l1_loss(pred, yg)");

    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, src).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["run", "--source-ad", "--profile-kernels"])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn nsl run");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(
        out.status.success(),
        "profiled run failed\nstderr:\n{stderr}"
    );

    let json = std::fs::read_to_string(tmp.join("kernel_profile.json")).unwrap_or_else(|e| {
        panic!("--profile-kernels wrote no kernel_profile.json ({e})\nstderr:\n{stderr}")
    });
    let _ = std::fs::remove_dir_all(&tmp);
    (json, stderr)
}

/// Extract every `"dur":<number>` from the trace. Deliberately a hand-rolled
/// scan rather than a JSON dependency — the assertion is about numbers being
/// non-zero, and the file format is ours.
fn durations(json: &str) -> Vec<f64> {
    let mut out = Vec::new();
    let mut rest = json;
    while let Some(i) = rest.find("\"dur\":") {
        rest = &rest[i + 6..];
        let end = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
            .unwrap_or(rest.len());
        if let Ok(v) = rest[..end].parse::<f64>() {
            out.push(v);
        }
        rest = &rest[end..];
    }
    out
}

fn metadata_number(json: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{key}\":");
    let i = json.rfind(&pat)? + pat.len();
    let rest = &json[i..];
    let end = rest
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e'))
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn profile_kernels_reports_real_durations() {
    let (json, stderr) = run_profiled();

    let durs = durations(&json);
    assert!(
        durs.len() > 50,
        "expected a dense kernel stream from the fixture, got {} events. \
         The gate cannot detect a zeroed profile if there is nothing in it.\
         \nstderr:\n{stderr}",
        durs.len()
    );

    // The actual regression: EVERY duration was 0.0.
    let zeros = durs.iter().filter(|d| **d == 0.0).count();
    assert_eq!(
        zeros,
        0,
        "{zeros} of {} kernel durations are exactly 0.0. That is the \
         signature of an unresolved cuEventElapsedTime, not a fast kernel — \
         see this file's header. stderr:\n{stderr}",
        durs.len()
    );

    // And the aggregate, which is what a human actually reads.
    let total = metadata_number(&json, "total_kernel_time_ms")
        .expect("metadata.total_kernel_time_ms present");
    assert!(
        total > 0.0,
        "total_kernel_time_ms is {total} across {} launches\nstderr:\n{stderr}",
        durs.len()
    );

    // The honesty flag: the flush must be ASSERTING the timings are real,
    // not merely happening to write non-zero numbers.
    assert!(
        json.contains("\"timing_valid\":true"),
        "the profile does not claim timing_valid — either a query failed or \
         the flag was dropped\nstderr:\n{stderr}"
    );
    assert_eq!(
        metadata_number(&json, "timing_query_failures"),
        Some(0.0),
        "some cuEventElapsedTime queries failed\nstderr:\n{stderr}"
    );
}

/// The other half of the fix: when timing genuinely cannot be resolved, the
/// profiler must SAY SO rather than emit zeros that read as measurements.
///
/// Pinned structurally (the failure path needs a dead CUDA context, which a
/// passing test cannot manufacture without also breaking the process). This
/// asserts the diagnostic and the flag exist and are wired to the failure
/// counter — a regression that deletes them is caught here even though the
/// runtime path is not exercised.
#[test]
fn a_failed_timing_query_is_reported_not_silently_zeroed() {
    let root = repo_root();
    let src =
        std::fs::read_to_string(root.join("crates/nsl-runtime/src/kernel_profiler.rs")).unwrap();
    assert!(
        src.contains("timing_failures += 1"),
        "the flush no longer counts failed cuEventElapsedTime queries — a \
         failed query leaves its out-param untouched, so it would once again \
         be written as a 0.0 measurement"
    );
    assert!(
        src.contains("timing_valid") && src.contains("timing_query_failures"),
        "the profile JSON no longer carries the timing-validity metadata a \
         consumer needs to reject a zeroed file"
    );
    assert!(
        src.contains("NOT measured"),
        "the stderr diagnostic that tells a human the durations are \
         unresolved has been removed"
    );

    let cuda = std::fs::read_to_string(root.join("crates/nsl-runtime/src/cuda/mod.rs")).unwrap();
    let i = cuda
        .find("pub unsafe fn cu_event_elapsed_time(")
        .expect("cu_event_elapsed_time still exists");
    let sig = &cuda[i..i + 200];
    assert!(
        sig.contains("-> CUresult"),
        "cu_event_elapsed_time discards its status again; a failed query \
         becomes a silent 0.0"
    );
}
