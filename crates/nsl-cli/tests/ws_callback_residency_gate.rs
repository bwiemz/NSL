//! Item 12: a train-loop callback that touches model θ under
//! `--weight-stream` must NOT crash on evicted (null) params. The compile
//! pass detects the model reference and brackets the callback body with a
//! scoped `upload_all` / `reevict_all` (residency window), and
//! `nsl_model_save` materializes evicted params from their mirrors as a
//! runtime belt.
//!
//! CPU tests assert the compile-time detection (the diagnostic fires exactly
//! when a callback references the model, with the correct writeback verdict).
//! The GPU test is the real end-to-end proof: the same program that would
//! crash on a null pointer now runs and produces a valid checkpoint.
//!
//! GPU: `cargo test -p nsl-cli --features cuda --test ws_callback_residency_gate -- --ignored`

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn fixture_src(name: &str) -> String {
    let p = repo_root().join("crates/nsl-cli/tests/fixtures").join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()))
}

/// Compile-only (`nsl build`) so the callback codegen runs — and its
/// `[weight-stream]` diagnostic prints — without executing (a CPU run under
/// `--weight-stream` would abort at the GPU-placement register belt). Returns
/// (success, stderr).
fn build_only(source: &str, tag: &str) -> (bool, String) {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_ws_cb_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, source).unwrap();
    let out = tmp.join("prog.out");

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "build",
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--weight-stream",
        ])
        .arg(&prog)
        .arg("-o")
        .arg(&out)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl build");
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

const DIAG: &str = "callback 'on_step' reads model state";

#[test]
fn callback_model_read_inserts_scoped_upload() {
    // The fixture's on_step reads `m.norm_out.sum()` and calls model_save.
    let (ok, stderr) = build_only(&fixture_src("csla_ws_callback_model_read.nsl"), "read");
    assert!(ok, "compile failed:\n{stderr}");
    assert!(
        stderr.contains(DIAG),
        "expected the scoped-upload diagnostic; stderr:\n{stderr}"
    );
    // Functional reads (`.sum()`, model_save) never mutate θ -> no writeback.
    assert!(
        stderr.contains("(without writeback)"),
        "read-only callback must skip the writeback DtoH; stderr:\n{stderr}"
    );
}

#[test]
fn loss_only_callback_no_scoped_upload() {
    // The stock FFN fixture's on_step only prints `loss` — no model touch, so
    // NO residency bracket is inserted (steady-state transfer arithmetic must
    // stay exactly as the CSLA gates assert).
    let (ok, stderr) = build_only(&fixture_src("csla_layerwise_ffn.nsl"), "lossonly");
    assert!(ok, "compile failed:\n{stderr}");
    assert!(
        !stderr.contains("reads model state"),
        "a loss-only callback must NOT trigger a residency bracket; stderr:\n{stderr}"
    );
}

#[test]
fn aliased_model_field_callback_flags_writeback() {
    // Binding a model field to a local (`let w = m.norm_out`) aliases the
    // resident streamed buffer; a later in-place mutation through `w` would
    // never have `model_sym` as its root, so the analysis conservatively
    // treats any model-rooted binding as a possible write (writeback=1 is
    // always safe). This guards the silent dropped-weight-update hazard.
    let src = fixture_src("csla_ws_callback_model_read.nsl").replace(
        "print(m.norm_out.sum())",
        "let wloc = m.norm_out\n            print(wloc.sum())",
    );
    let (ok, stderr) = build_only(&src, "alias");
    assert!(ok, "compile failed:\n{stderr}");
    assert!(
        stderr.contains(DIAG) && stderr.contains("(with writeback)"),
        "a model-field alias must be bracketed with writeback; stderr:\n{stderr}"
    );
}

#[test]
fn inplace_mutator_callback_flags_writeback() {
    // A callback that mutates θ in place (`copy_data(m.norm_out, ..)`) must be
    // bracketed WITH writeback so the change survives the next window upload.
    let src = fixture_src("csla_ws_callback_model_read.nsl").replace(
        "print(m.norm_out.sum())",
        "copy_data(m.norm_out, m.norm_out)",
    );
    let (ok, stderr) = build_only(&src, "mutate");
    assert!(ok, "compile failed:\n{stderr}");
    assert!(
        stderr.contains(DIAG) && stderr.contains("(with writeback)"),
        "in-place mutator callback must be bracketed with writeback; stderr:\n{stderr}"
    );
}

/// End-to-end GPU proof: with `# GPU_PLACEMENT` -> `m.to(cuda)` and weight
/// streaming ON, the on_step callback reads `m.norm_out` and saves a
/// checkpoint mid-loop — both of which dereference evicted params without
/// Item 12. The run must succeed, print the norm sum, and leave a loadable
/// checkpoint.
#[test]
#[ignore = "requires CUDA GPU"]
fn callback_model_read_runs_on_gpu_without_crash() {
    let root = repo_root();
    let tmp = std::env::temp_dir().join(format!("nsl_ws_cb_gpu_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let ckpt = tmp.join("mid_loop.nslm");
    let src = fixture_src("csla_ws_callback_model_read.nsl")
        .replace("# GPU_PLACEMENT", "m.to(cuda)")
        .replace("CKPT_PATH", ckpt.to_str().unwrap());
    let prog = tmp.join("prog.nsl");
    std::fs::write(&prog, &src).unwrap();

    let output = Command::new(env!("CARGO"))
        .args(["run", "-q", "--features", "cuda", "--manifest-path"])
        .arg(root.join("Cargo.toml"))
        .args([
            "-p",
            "nsl-cli",
            "--",
            "run",
            "--source-ad",
            "--checkpoint-blocks",
            "--layerwise-accum",
            "--weight-stream",
        ])
        .arg(&prog)
        .current_dir(&tmp)
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .env("NSL_WS_COUNTER", "1")
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "GPU run crashed (the Item 12 hazard):\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // The model-field read produced a real value (not a null-pointer abort).
    let norm = stdout
        .split_once("NORMSUM_BEGIN")
        .and_then(|(_, r)| r.split_once("NORMSUM_END"))
        .map(|(v, _)| v)
        .unwrap_or("");
    assert!(
        norm.chars().any(|c| c.is_ascii_digit()),
        "expected a printed norm sum between the markers; got {norm:?}\nstderr:\n{stderr}"
    );
    // The mid-loop checkpoint exists and is a real .nslm (magic "NSLM").
    let bytes = std::fs::read(&ckpt).expect("mid-loop checkpoint missing");
    assert!(
        bytes.len() > 16 && &bytes[0..4] == b"NSLM",
        "checkpoint is not a valid .nslm (len {}, magic {:?})",
        bytes.len(),
        &bytes[0..4.min(bytes.len())]
    );
}
